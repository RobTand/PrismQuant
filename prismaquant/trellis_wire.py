"""Independent writer and decoder for ``gridbook.trellis.wire.v1``.

PrismaQuant may neither import nor vendor Gridbook.  This module therefore
implements the published byte contract from first principles, just as the
GGUF producer owns its writer and proves cross-repository agreement with
golden vectors.  The wire blob is the primary object: schedule, offsets,
alphabets, scale plane, and row bodies all live here and nowhere else.

The body is LSB-first and tight across physical 256-column blocks.  Only the
end of a row is padded, to a 16-byte boundary.  Decoding is scan-free: the
eight predecessor input bits determine the convolutional-code subset for a
shaped position.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping, Sequence

import torch

from .trellis_formats import (
    E2M1_FAMILY,
    E4M3_FAMILY,
    E4M3FN_NAN_CODES,
    GENERATOR_OCTAL,
    LAYOUT_FIXED_QUOTA,
    LAYOUT_TIGHT_OFFSETS,
    MIN_TRELLIS_STEPS,
    STATE_MEMORY_BITS,
    SUPERBLOCK_WEIGHTS,
    TRELLIS_WIRE_SCHEMA,
    get_trellis_family,
    native_code_value,
    validate_alphabets,
    validate_schedule,
)


MAGIC = b"GBTCQ1\0\0"
VERSION = 1
ROW_ALIGNMENT_BYTES = 16
HEADER = struct.Struct("<8sBBBBIIIIIIQQf32s")

_FAMILY_CODE = {E2M1_FAMILY: 1, E4M3_FAMILY: 2}
_CODE_FAMILY = {value: key for key, value in _FAMILY_CODE.items()}
_LAYOUT_CODE = {LAYOUT_TIGHT_OFFSETS: 1, LAYOUT_FIXED_QUOTA: 2}
_CODE_LAYOUT = {value: key for key, value in _LAYOUT_CODE.items()}
_GENERATOR_0 = int(GENERATOR_OCTAL[0], 8)
_GENERATOR_1 = int(GENERATOR_OCTAL[1], 8)

# An E2M1 wire stores one E4M3FN scale byte per group of sixteen weights.
# Constructing the set below happens once (and is intentionally independent of
# torch's float8 implementation).  Validation can then scan an arbitrarily
# large scale plane in C via ``set(bytes)`` instead of calling Python once per
# group.  This matters for full-model receipts, where the scale plane contains
# tens of millions of bytes but at most 256 distinct values.
_POSITIVE_E4M3_SCALE_CODES = frozenset(
    code
    for code in range(256)
    if code not in E4M3FN_NAN_CODES
    and native_code_value(E4M3_FAMILY, code) > 0.0
)


class TrellisWireError(ValueError):
    """A blob is not a canonical wire-v1 payload."""


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _align(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _block_offsets(schedule: Sequence[int]) -> tuple[int, ...]:
    total = 0
    offsets = [0]
    for first in range(0, len(schedule), SUPERBLOCK_WEIGHTS):
        total += sum(schedule[first:first + SUPERBLOCK_WEIGHTS])
        offsets.append(total)
    return tuple(offsets)


def _pack_nibbles(values: Sequence[int]) -> bytes:
    out = bytearray(_ceil_div(len(values), 2))
    for index, raw in enumerate(values):
        value = int(raw)
        if not 0 <= value <= 15:
            raise TrellisWireError(f"schedule nibble outside [0,15]: {value}")
        out[index // 2] |= value << (4 * (index & 1))
    return bytes(out)


def _unpack_nibbles(data: bytes, count: int) -> tuple[int, ...]:
    return tuple(
        (data[index // 2] >> (4 * (index & 1))) & 15
        for index in range(count)
    )


def _alphabet_blob(alphabets: Mapping[int, Sequence[int]]) -> bytes:
    out = bytearray()
    for rate, codes in sorted(alphabets.items()):
        out += struct.pack("<BH", int(rate), len(codes))
        out += bytes(int(code) for code in codes)
    return bytes(out)


def _parse_alphabet_blob(blob: bytes) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    cursor = 0
    while cursor < len(blob):
        if len(blob) - cursor < 3:
            raise TrellisWireError("truncated trellis alphabet directory")
        rate, count = struct.unpack_from("<BH", blob, cursor)
        cursor += 3
        if len(blob) - cursor < count:
            raise TrellisWireError("truncated trellis alphabet payload")
        if rate in out:
            raise TrellisWireError(f"duplicate trellis alphabet rate {rate}")
        out[int(rate)] = tuple(blob[cursor:cursor + count])
        cursor += count
    return out


def _wire_tensor(blob: bytes | bytearray | memoryview | torch.Tensor) -> torch.Tensor:
    if isinstance(blob, torch.Tensor):
        if blob.dtype != torch.uint8 or blob.ndim != 1:
            raise TrellisWireError("wire tensor must be one-dimensional uint8")
        return blob.detach().contiguous()
    return torch.frombuffer(bytearray(bytes(blob)), dtype=torch.uint8)


def _wire_bytes(blob: bytes | bytearray | memoryview | torch.Tensor) -> bytes:
    if isinstance(blob, torch.Tensor):
        tensor = _wire_tensor(blob).to(device="cpu").contiguous()
        return tensor.numpy().tobytes()
    return bytes(blob)


@dataclass(frozen=True, slots=True)
class TrellisWire:
    family: str
    layout: str
    rows: int
    columns: int
    body_rate_q256: int
    schedule: tuple[int, ...]
    block_offsets_bits: tuple[int, ...]
    alphabets: Mapping[int, tuple[int, ...]]
    scale_blob: bytes
    global_scale_real: float
    row_body_bits: int
    row_stride_bytes: int
    payload: bytes

    @property
    def format_name(self) -> str:
        return get_trellis_family(self.family).format_name(
            self.body_rate_q256
        )

    @property
    def rendered_wire_identity_sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def row_payload(self, row: int) -> bytes:
        if not 0 <= int(row) < self.rows:
            raise IndexError(row)
        start = int(row) * self.row_stride_bytes
        return self.payload[start:start + self.row_stride_bytes]

    def validate(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise TrellisWireError("trellis rows and columns must be positive")
        spec = get_trellis_family(self.family)
        schedule = validate_schedule(
            spec,
            int(self.body_rate_q256),
            self.schedule,
            layout=self.layout,
        )
        if len(schedule) != self.columns:
            raise TrellisWireError(
                f"trellis schedule has {len(schedule)} values, expected "
                f"{self.columns}"
            )
        expected_offsets = _block_offsets(schedule)
        if tuple(self.block_offsets_bits) != expected_offsets:
            raise TrellisWireError("block offsets do not match the schedule")
        if int(self.row_body_bits) != expected_offsets[-1]:
            raise TrellisWireError("row body bits do not match block offsets")
        expected_stride = _align(_ceil_div(self.row_body_bits, 8), 16)
        if self.row_stride_bytes != expected_stride:
            raise TrellisWireError("trellis row stride is not canonical")
        if len(self.payload) != self.rows * self.row_stride_bytes:
            raise TrellisWireError("trellis payload size differs from row layout")
        checked = validate_alphabets(spec, schedule, self.alphabets)
        if dict(checked) != dict(self.alphabets):
            raise TrellisWireError("trellis alphabets are not canonical")

        if spec.family == E2M1_FAMILY:
            expected_scales = self.rows * _ceil_div(self.columns, 16)
            if len(self.scale_blob) != expected_scales:
                raise TrellisWireError(
                    f"E2M1 scale plane is {len(self.scale_blob)} bytes, "
                    f"expected {expected_scales}"
                )
            if (
                not math.isfinite(self.global_scale_real)
                or self.global_scale_real <= 0.0
            ):
                raise TrellisWireError(
                    "E2M1 global_scale_real must be finite and positive"
                )
            if not set(self.scale_blob).issubset(_POSITIVE_E4M3_SCALE_CODES):
                raise TrellisWireError(
                    "E2M1 scale codes must decode finite and positive"
                )
        else:
            if self.global_scale_real != 1.0:
                raise TrellisWireError(
                    "E4M3 global_scale_real is fixed at exactly 1.0"
                )
            if len(self.scale_blob) != self.rows * 4:
                raise TrellisWireError("E4M3 requires one fp32 scale per row")
            scales = struct.unpack(f"<{self.rows}f", self.scale_blob)
            if any(not math.isfinite(value) or value <= 0 for value in scales):
                raise TrellisWireError(
                    "E4M3 row scales must be finite and positive"
                )

        used_bytes = _ceil_div(self.row_body_bits, 8)
        body_tensor = torch.frombuffer(
            bytearray(self.payload), dtype=torch.uint8
        ).reshape(self.rows, self.row_stride_bytes)
        if self.row_body_bits & 7:
            mask = (1 << (self.row_body_bits & 7)) - 1
            if bool((body_tensor[:, used_bytes - 1] & ~mask).any().item()):
                raise TrellisWireError(
                    "nonzero high padding bits in final body byte"
                )
        if used_bytes < self.row_stride_bytes and bool(
            body_tensor[:, used_bytes:].any().item()
        ):
            raise TrellisWireError("nonzero trellis row padding bytes")

        if spec.family == E4M3_FAMILY:
            for block, first in enumerate(
                range(0, self.columns, SUPERBLOCK_WEIGHTS)
            ):
                stop = min(first + SUPERBLOCK_WEIGHTS, self.columns)
                rates = torch.tensor(schedule[first:stop], dtype=torch.int64)
                bypass = torch.nonzero(rates == spec.bypass_rate).flatten()
                if not bypass.numel():
                    continue
                starts = torch.cumsum(rates, 0) - rates
                starts += int(self.block_offsets_bits[block])
                codes = _gather_bits(
                    body_tensor,
                    starts.index_select(0, bypass),
                    spec.grid_bits,
                )
                if bool(((codes == 0x7F) | (codes == 0xFF)).any().item()):
                    raise TrellisWireError(
                        "wire contains an E4M3 NaN bypass code"
                    )

    def to_bytes(self) -> bytes:
        self.validate()
        schedule_blob = _pack_nibbles(self.schedule)
        alphabet_blob = _alphabet_blob(self.alphabets)
        if self.layout == LAYOUT_TIGHT_OFFSETS:
            offset_width = 4 if self.row_body_bits <= 0xFFFFFFFF else 8
            offset_code = "I" if offset_width == 4 else "Q"
            offsets_blob = struct.pack(
                f"<{len(self.block_offsets_bits)}{offset_code}",
                *self.block_offsets_bits,
            )
        else:
            offset_width = 0
            offsets_blob = b""
        header = HEADER.pack(
            MAGIC,
            VERSION,
            _FAMILY_CODE[self.family],
            _LAYOUT_CODE[self.layout],
            offset_width,
            self.rows,
            self.columns,
            self.body_rate_q256,
            len(self.schedule),
            len(alphabet_blob),
            len(self.scale_blob),
            self.row_body_bits,
            self.row_stride_bytes,
            float(self.global_scale_real),
            hashlib.sha256(alphabet_blob).digest(),
        )
        return (
            header
            + schedule_blob
            + offsets_blob
            + alphabet_blob
            + self.scale_blob
            + self.payload
        )

    @classmethod
    def from_bytes(
        cls, data: bytes | bytearray | memoryview | torch.Tensor
    ) -> "TrellisWire":
        blob = _wire_bytes(data)
        if len(blob) < HEADER.size:
            raise TrellisWireError("truncated trellis header")
        (
            magic,
            version,
            family_code,
            layout_code,
            offset_width,
            rows,
            columns,
            body_rate_q256,
            schedule_count,
            alphabet_size,
            scale_size,
            row_body_bits,
            row_stride,
            global_scale,
            alphabet_digest,
        ) = HEADER.unpack_from(blob)
        if magic != MAGIC or version != VERSION:
            raise TrellisWireError(
                f"not a {TRELLIS_WIRE_SCHEMA} payload"
            )
        try:
            family = _CODE_FAMILY[family_code]
            layout = _CODE_LAYOUT[layout_code]
        except KeyError as exc:
            raise TrellisWireError("unknown trellis family/layout code") from exc
        allowed_widths = (
            (4, 8) if layout == LAYOUT_TIGHT_OFFSETS else (0,)
        )
        if offset_width not in allowed_widths:
            raise TrellisWireError(
                f"invalid offset width {offset_width} for {layout}"
            )

        cursor = HEADER.size
        schedule_size = _ceil_div(schedule_count, 2)
        end = cursor + schedule_size
        if end > len(blob):
            raise TrellisWireError("truncated trellis schedule")
        schedule_blob = blob[cursor:end]
        schedule = _unpack_nibbles(schedule_blob, schedule_count)
        if schedule_count & 1 and schedule_blob[-1] & 0xF0:
            raise TrellisWireError("nonzero schedule padding nibble")
        cursor = end

        blocks = _ceil_div(columns, SUPERBLOCK_WEIGHTS)
        offset_count = blocks + 1 if offset_width else 0
        end = cursor + offset_count * offset_width
        if end > len(blob):
            raise TrellisWireError("truncated trellis block offsets")
        if offset_width:
            code = "I" if offset_width == 4 else "Q"
            offsets = struct.unpack(
                f"<{offset_count}{code}", blob[cursor:end]
            )
        else:
            offsets = _block_offsets(schedule)
        cursor = end

        end = cursor + alphabet_size
        if end > len(blob):
            raise TrellisWireError("truncated trellis alphabets")
        alphabet_blob = blob[cursor:end]
        if hashlib.sha256(alphabet_blob).digest() != alphabet_digest:
            raise TrellisWireError("trellis alphabet digest mismatch")
        alphabets = _parse_alphabet_blob(alphabet_blob)
        cursor = end

        end = cursor + scale_size
        if end > len(blob):
            raise TrellisWireError("truncated trellis scale plane")
        scale_blob = blob[cursor:end]
        cursor = end
        expected_payload = rows * row_stride
        if len(blob) - cursor != expected_payload:
            raise TrellisWireError(
                f"trellis body is {len(blob)-cursor} bytes, expected "
                f"{expected_payload}"
            )
        wire = cls(
            family=family,
            layout=layout,
            rows=int(rows),
            columns=int(columns),
            body_rate_q256=int(body_rate_q256),
            schedule=tuple(schedule),
            block_offsets_bits=tuple(int(value) for value in offsets),
            alphabets=alphabets,
            scale_blob=bytes(scale_blob),
            global_scale_real=float(global_scale),
            row_body_bits=int(row_body_bits),
            row_stride_bytes=int(row_stride),
            payload=bytes(blob[cursor:]),
        )
        wire.validate()
        return wire


def _pack_body_torch(
    schedule: Sequence[int],
    u_bits: torch.Tensor,
    point_indices: torch.Tensor,
    bypass_codes: torch.Tensor,
    *,
    terminal_rate: int,
    row_stride_bytes: int,
) -> bytes:
    """Pack full rectangular planes on their resident device."""
    if (
        u_bits.shape != point_indices.shape
        or u_bits.shape != bypass_codes.shape
        or u_bits.ndim != 2
    ):
        raise TrellisWireError(
            "u_bits, point_indices, and bypass_codes must be equal rank-2 "
            "planes"
        )
    rows, columns = map(int, u_bits.shape)
    if columns != len(schedule):
        raise TrellisWireError("wire plane width differs from schedule")
    device = u_bits.device
    if point_indices.device != device or bypass_codes.device != device:
        raise TrellisWireError("wire planes must share a device")
    rates = torch.tensor(schedule, dtype=torch.int64, device=device)
    starts = torch.cumsum(rates, 0) - rates
    row_base = (
        torch.arange(rows, device=device, dtype=torch.int64)
        * int(row_stride_bytes * 8)
    )[:, None]
    absolute = row_base + starts[None, :]
    shaped = rates < int(terminal_rate)
    values = torch.where(
        shaped[None, :],
        u_bits.to(torch.int64)
        | (point_indices.to(torch.int64) << 1),
        bypass_codes.to(torch.int64),
    )
    shaped_plane = u_bits[:, shaped]
    if bool(((shaped_plane < 0) | (shaped_plane > 1)).any().item()):
        raise TrellisWireError("trellis coded-bit plane is not binary")
    shaped_rates = rates[shaped]
    shaped_limits = torch.bitwise_left_shift(
        torch.ones_like(shaped_rates), shaped_rates - 1
    )
    shaped_points = point_indices[:, shaped].to(torch.int64)
    if bool(
        ((shaped_points < 0) | (shaped_points >= shaped_limits[None, :]))
        .any()
        .item()
    ):
        raise TrellisWireError("point index exceeds its rate subset width")
    bypass_plane = bypass_codes[:, ~shaped].to(torch.int64)
    if bool(
        ((bypass_plane < 0) | (bypass_plane >= (1 << terminal_rate)))
        .any()
        .item()
    ):
        raise TrellisWireError("bypass code exceeds native width")

    flat = torch.zeros(
        rows * row_stride_bytes,
        dtype=torch.int32,
        device=device,
    )
    for bit in range(terminal_rate):
        active = rates > bit
        positions = absolute[:, active] + bit
        indices = (positions >> 3).reshape(-1)
        contribution = (
            ((values[:, active] >> bit) & 1)
            << (positions & 7)
        ).reshape(-1).to(torch.int32)
        flat.scatter_add_(0, indices, contribution)
    if bool((flat > 255).any().item()):
        raise AssertionError("wire body bit packing overlapped a bit slot")
    return flat.to(torch.uint8).to(device="cpu").numpy().tobytes()


def pack_planes(
    *,
    family: str,
    body_rate_q256: int,
    schedule: Sequence[int],
    layout: str,
    u_bits: torch.Tensor,
    point_indices: torch.Tensor,
    bypass_codes: torch.Tensor,
    alphabets: Mapping[int, Sequence[int]],
    scale_blob: bytes,
    global_scale_real: float = 1.0,
) -> TrellisWire:
    """Pack already-encoded planes into one canonical wire."""
    spec = get_trellis_family(family)
    if u_bits.ndim != 2:
        raise TrellisWireError("trellis planes must be rank two")
    rows, columns = map(int, u_bits.shape)
    normalized_schedule = validate_schedule(
        spec, int(body_rate_q256), schedule, layout=layout
    )
    if len(normalized_schedule) != columns:
        raise TrellisWireError("schedule length differs from plane width")
    checked_alphabets = validate_alphabets(
        spec, normalized_schedule, alphabets
    )
    offsets = _block_offsets(normalized_schedule)
    row_bits = offsets[-1]
    row_stride = _align(_ceil_div(row_bits, 8), ROW_ALIGNMENT_BYTES)
    payload = _pack_body_torch(
        normalized_schedule,
        u_bits,
        point_indices,
        bypass_codes,
        terminal_rate=spec.bypass_rate,
        row_stride_bytes=row_stride,
    )
    canonical_global = struct.unpack(
        "<f", struct.pack("<f", float(global_scale_real))
    )[0]
    wire = TrellisWire(
        family=spec.family,
        layout=layout,
        rows=rows,
        columns=columns,
        body_rate_q256=int(body_rate_q256),
        schedule=tuple(normalized_schedule),
        block_offsets_bits=offsets,
        alphabets=checked_alphabets,
        scale_blob=bytes(scale_blob),
        global_scale_real=canonical_global,
        row_body_bits=row_bits,
        row_stride_bytes=row_stride,
        payload=payload,
    )
    wire.validate()
    return wire


def _gather_bits(
    body: torch.Tensor, offsets: torch.Tensor, width: int | torch.Tensor
) -> torch.Tensor:
    """Gather LSB-first bitfields from ``body[rows, stride]``."""
    rows = body.shape[0]
    offsets = offsets.to(device=body.device, dtype=torch.int64)
    if isinstance(width, int):
        widths = torch.full_like(offsets, int(width))
        max_width = int(width)
    else:
        widths = width.to(device=body.device, dtype=torch.int64)
        max_width = int(widths.max().item()) if widths.numel() else 0
    result = torch.zeros(
        rows, int(offsets.numel()), dtype=torch.int64, device=body.device
    )
    for bit in range(max_width):
        absolute = offsets + bit
        byte = body.index_select(1, absolute >> 3).to(torch.int64)
        selected = (byte >> (absolute & 7)[None, :]) & 1
        selected = selected * (widths > bit)[None, :]
        result |= selected << bit
    return result


def decode_codes_torch(
    wire_or_blob: TrellisWire | bytes | bytearray | memoryview | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decode native E2M1 nibbles / E4M3 bytes without a state scan."""
    wire = (
        wire_or_blob
        if isinstance(wire_or_blob, TrellisWire)
        else TrellisWire.from_bytes(wire_or_blob)
    )
    wire.validate()
    target = torch.device(device)
    body = torch.frombuffer(
        bytearray(wire.payload), dtype=torch.uint8
    ).reshape(wire.rows, wire.row_stride_bytes).to(target)
    schedule = torch.tensor(wire.schedule, dtype=torch.int64, device=target)
    result = torch.empty(
        wire.rows, wire.columns, dtype=torch.uint8, device=target
    )
    spec = get_trellis_family(wire.family)
    parity = torch.tensor(
        [
            2 * ((register & _GENERATOR_0).bit_count() & 1)
            + ((register & _GENERATOR_1).bit_count() & 1)
            for register in range(1 << (STATE_MEMORY_BITS + 1))
        ],
        dtype=torch.int64,
        device=target,
    )
    for block, first in enumerate(
        range(0, wire.columns, SUPERBLOCK_WEIGHTS)
    ):
        stop = min(first + SUPERBLOCK_WEIGHTS, wire.columns)
        block_rates = schedule[first:stop]
        starts = torch.cumsum(block_rates, 0) - block_rates
        starts += int(wire.block_offsets_bits[block])
        shaped_local = torch.nonzero(
            block_rates < spec.bypass_rate
        ).flatten()
        bypass_local = torch.nonzero(
            block_rates == spec.bypass_rate
        ).flatten()
        if shaped_local.numel() < MIN_TRELLIS_STEPS:
            raise TrellisWireError("trellis decode block violates tail-bite floor")
        if bypass_local.numel():
            codes = _gather_bits(
                body,
                starts.index_select(0, bypass_local),
                spec.grid_bits,
            ).to(torch.uint8)
            if (
                spec.family == E4M3_FAMILY
                and bool(
                    ((codes == 0x7F) | (codes == 0xFF)).any().item()
                )
            ):
                raise TrellisWireError("wire contains an E4M3 NaN bypass code")
            result[:, first + bypass_local] = codes
        shaped_offsets = starts.index_select(0, shaped_local)
        u = _gather_bits(body, shaped_offsets, 1)
        state = torch.zeros_like(u)
        for previous in range(1, STATE_MEMORY_BITS + 1):
            state |= torch.roll(u, shifts=previous, dims=1) << (
                STATE_MEMORY_BITS - previous
            )
        subset = parity[(u << STATE_MEMORY_BITS) | state]
        shaped_rates = block_rates.index_select(0, shaped_local)
        points = _gather_bits(body, shaped_offsets + 1, shaped_rates - 1)
        for rate in sorted(set(int(value) for value in shaped_rates.tolist())):
            at = torch.nonzero(shaped_rates == rate).flatten()
            alphabet = torch.tensor(
                wire.alphabets[rate], dtype=torch.uint8, device=target
            )
            logical = subset.index_select(1, at) + 4 * points.index_select(1, at)
            result[:, first + shaped_local.index_select(0, at)] = alphabet[logical]
    return result


def _e4m3_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float("nan") if code in E4M3FN_NAN_CODES
            else native_code_value(E4M3_FAMILY, code)
            for code in range(256)
        ],
        dtype=torch.float32,
        device=device,
    )


def decode_values_torch(
    wire_or_blob: TrellisWire | bytes | bytearray | memoryview | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode and apply the exact scale plane on the requested device."""
    wire = (
        wire_or_blob
        if isinstance(wire_or_blob, TrellisWire)
        else TrellisWire.from_bytes(wire_or_blob)
    )
    target = torch.device(device)
    codes = decode_codes_torch(wire, device=target).to(torch.int64)
    if wire.family == E2M1_FAMILY:
        code_values = torch.tensor(
            [native_code_value(E2M1_FAMILY, code) for code in range(16)],
            dtype=torch.float32,
            device=target,
        )
        scale_codes = torch.frombuffer(
            bytearray(wire.scale_blob), dtype=torch.uint8
        ).reshape(wire.rows, _ceil_div(wire.columns, 16)).to(target)
        scales = _e4m3_table(target)[scale_codes.to(torch.int64)]
        scales = scales.repeat_interleave(16, dim=1)[:, :wire.columns]
        values = code_values[codes] * scales * float(wire.global_scale_real)
    else:
        values = _e4m3_table(target)[codes]
        scales = torch.frombuffer(
            bytearray(wire.scale_blob), dtype=torch.float32
        ).reshape(wire.rows, 1).to(target)
        values = values * scales
    if not bool(torch.isfinite(values).all().item()):
        raise TrellisWireError("decoded trellis values are non-finite")
    return values.to(dtype=dtype).contiguous()


__all__ = [
    "HEADER",
    "MAGIC",
    "ROW_ALIGNMENT_BYTES",
    "TrellisWire",
    "TrellisWireError",
    "VERSION",
    "decode_codes_torch",
    "decode_values_torch",
    "pack_planes",
]
