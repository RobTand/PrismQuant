"""Side-information-inclusive tensor accounting for Gridbook trellis wires.

The result describes serialized tensor payload bytes, not safetensors JSON or
filesystem/container overhead.  Schedule, block offsets, alphabets, row
padding, and the family-specific scale plane are all included; body rate alone
is intentionally never reported as artifact bpw.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .trellis_formats import (
    E2M1_FAMILY,
    LAYOUTS,
    LAYOUT_TIGHT_OFFSETS,
    SUPERBLOCK_WEIGHTS,
    TRELLIS_WIRE_SCHEMA,
    TrellisFamily,
    TrellisFormatError,
    get_trellis_family,
    validate_alphabets,
    validate_body_rate_q256,
    validate_schedule,
)


TRELLIS_TENSOR_PAYLOAD_SCHEMA = "prismaquant.trellis_tensor_payload.v1"
ROW_ALIGNMENT_BYTES = 16
BLOCK_OFFSET_BITS = 32
WIRE_HEADER_BYTES = 88
WIRE_V1_UINT32_MAX = (1 << 32) - 1
SCHEDULE_BITS_PER_CODE = 4
ALPHABET_DIRECTORY_BYTES_PER_RATE = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_PAYLOAD_FIELDS = frozenset({
    "schema",
    "wire_schema",
    "family",
    "format",
    "grid",
    "shape",
    "body_rate_q256",
    "body_bpw",
    "layout",
    "superblock_weights",
    "block_count",
    "body_bits_per_row",
    "unpadded_body_bytes_per_row",
    "body_row_stride_bytes",
    "body_padding_bytes",
    "body_bytes",
    "wire_header_bytes",
    "scale_contract",
    "scale_bytes",
    "schedule_scope",
    "schedule_bits_per_code",
    "schedule_bytes",
    "schedule_identity_sha256",
    "block_offset_bits",
    "block_offset_bytes",
    "alphabet_bytes_by_rate",
    "alphabet_bytes",
    "alphabet_identity_sha256",
    "sidecar_header_bytes",
    "structural_side_information_bytes",
    "wire_side_information_bytes",
    "side_information_bytes",
    "total_bytes",
    "exact_bpw",
    "expanded_weight_resident_bytes",
    "pre_render_recipe_identity_scope",
    "rendered_wire_identity_sha256",
    "producer_eligible",
    "pre_render_recipe_identity_sha256",
})


def _sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _align(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _plain_json_copy(value: object) -> object:
    """Copy JSON-shaped footprint data out of caller-owned containers."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TrellisFormatError(
                "trellis footprint mappings must use canonical string keys"
            )
        return {
            key: _plain_json_copy(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json_copy(item) for item in value]
    return value


def _payload_integer(
    payload: Mapping[str, object],
    field: str,
    *,
    nonnegative: bool = True,
) -> int:
    value = payload.get(field)
    if type(value) is not int or (nonnegative and value < 0):
        qualifier = "nonnegative " if nonnegative else ""
        raise TrellisFormatError(
            f"trellis footprint {field} must be a {qualifier}JSON integer"
        )
    return value


def _payload_float(
    payload: Mapping[str, object],
    field: str,
    *,
    nonnegative: bool = True,
) -> float:
    value = payload.get(field)
    if type(value) is not float:
        raise TrellisFormatError(
            f"trellis footprint {field} must be a canonical binary64 number"
        )
    result = value
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise TrellisFormatError(
            f"trellis footprint {field} must be a finite nonnegative number"
        )
    return result


def validate_trellis_tensor_payload_breakdown(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Copy and validate a content-addressed footprint at an API boundary.

    The recipe digest is a content address, not an authorization signature.
    Recomputing it detects stale or accidentally mutated reports. Independent
    schema and arithmetic checks keep a caller from making an internally
    inconsistent report self-consistent merely by recomputing that digest.
    """

    if not isinstance(payload, Mapping):
        raise TrellisFormatError("trellis footprint must be a mapping")
    copied = _plain_json_copy(payload)
    if not isinstance(copied, dict):
        raise AssertionError("mapping copy did not produce a dictionary")
    if set(copied) != _CANONICAL_PAYLOAD_FIELDS:
        missing = sorted(_CANONICAL_PAYLOAD_FIELDS - set(copied))
        extra = sorted(set(copied) - _CANONICAL_PAYLOAD_FIELDS)
        raise TrellisFormatError(
            "trellis footprint fields differ from the canonical schema: "
            f"missing={missing}, extra={extra}"
        )
    if copied.get("schema") != TRELLIS_TENSOR_PAYLOAD_SCHEMA:
        raise TrellisFormatError(
            "trellis footprint has an unsupported payload schema"
        )
    if copied.get("wire_schema") != TRELLIS_WIRE_SCHEMA:
        raise TrellisFormatError(
            "trellis footprint has an unsupported Gridbook wire schema"
        )

    claimed_identity = copied.get("pre_render_recipe_identity_sha256")
    if (
        not isinstance(claimed_identity, str)
        or _SHA256.fullmatch(claimed_identity) is None
    ):
        raise TrellisFormatError(
            "trellis footprint pre-render recipe identity must be lowercase "
            "SHA-256"
        )
    digest_body = dict(copied)
    digest_body.pop("pre_render_recipe_identity_sha256")
    try:
        expected_identity = _sha256(digest_body)
    except (TypeError, ValueError) as exc:
        raise TrellisFormatError(
            "trellis footprint must contain canonical finite JSON data"
        ) from exc
    if claimed_identity != expected_identity:
        raise TrellisFormatError(
            "trellis footprint pre-render recipe identity does not match "
            "its canonical contents"
        )

    shape = copied.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise TrellisFormatError(
            "trellis footprint shape must be two positive JSON integers"
        )
    rows, columns = shape
    if rows > WIRE_V1_UINT32_MAX or columns > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "trellis footprint shape exceeds Gridbook wire-v1 uint32 fields"
        )
    spec = get_trellis_family(str(copied.get("family")))
    rate = validate_body_rate_q256(
        spec,
        _payload_integer(copied, "body_rate_q256"),
    )
    layout = copied.get("layout")
    if layout not in LAYOUTS:
        raise TrellisFormatError("trellis footprint has an invalid layout")
    if copied.get("format") != spec.format_name(rate):
        raise TrellisFormatError("trellis footprint format does not match rate")
    if copied.get("grid") != spec.grid:
        raise TrellisFormatError("trellis footprint grid does not match family")
    if copied.get("scale_contract") != spec.scale_contract:
        raise TrellisFormatError(
            "trellis footprint scale contract does not match family"
        )
    if _payload_integer(copied, "superblock_weights") != SUPERBLOCK_WEIGHTS:
        raise TrellisFormatError("trellis footprint superblock size drifted")

    block_count = _payload_integer(copied, "block_count")
    if block_count != _ceil_div(columns, SUPERBLOCK_WEIGHTS):
        raise TrellisFormatError("trellis footprint block_count arithmetic drifted")
    body_bits_per_row = _payload_integer(copied, "body_bits_per_row")
    if not columns <= body_bits_per_row <= columns * spec.bypass_rate:
        raise TrellisFormatError(
            "trellis footprint body-bit total is outside schedule bounds"
        )
    if layout == LAYOUT_TIGHT_OFFSETS:
        if abs(
            body_bits_per_row * SUPERBLOCK_WEIGHTS - rate * columns
        ) >= SUPERBLOCK_WEIGHTS:
            raise TrellisFormatError(
                "trellis footprint tight-offset q256 arithmetic drifted"
            )
    else:
        complete_blocks, tail_columns = divmod(columns, SUPERBLOCK_WEIGHTS)
        complete_bits = complete_blocks * rate
        if tail_columns == 0:
            tail_minimum = tail_maximum = 0
        elif tail_columns < 8:
            raise TrellisFormatError(
                "trellis footprint tail is too short for tail biting"
            )
        else:
            tail_minimum = tail_columns
            tail_maximum = tail_columns * spec.bypass_rate - 8
        if not (
            complete_bits + tail_minimum
            <= body_bits_per_row
            <= complete_bits + tail_maximum
        ):
            raise TrellisFormatError(
                "trellis footprint fixed-quota body-bit arithmetic drifted"
            )
    unpadded = _payload_integer(copied, "unpadded_body_bytes_per_row")
    if unpadded != _ceil_div(body_bits_per_row, 8):
        raise TrellisFormatError(
            "trellis footprint unpadded body-byte arithmetic drifted"
        )
    row_stride = _payload_integer(copied, "body_row_stride_bytes")
    if row_stride != _align(unpadded, ROW_ALIGNMENT_BYTES):
        raise TrellisFormatError(
            "trellis footprint body row-stride arithmetic drifted"
        )
    padding = _payload_integer(copied, "body_padding_bytes")
    if padding != rows * (row_stride - unpadded):
        raise TrellisFormatError(
            "trellis footprint body-padding arithmetic drifted"
        )
    body_bytes = _payload_integer(copied, "body_bytes")
    if body_bytes != rows * row_stride:
        raise TrellisFormatError("trellis footprint body-byte arithmetic drifted")

    if _payload_integer(copied, "wire_header_bytes") != WIRE_HEADER_BYTES:
        raise TrellisFormatError("trellis footprint wire header size drifted")
    if copied.get("schedule_scope") != "tensor_input_column_shared_across_rows":
        raise TrellisFormatError("trellis footprint schedule scope drifted")
    if (
        _payload_integer(copied, "schedule_bits_per_code")
        != SCHEDULE_BITS_PER_CODE
    ):
        raise TrellisFormatError("trellis footprint schedule code width drifted")
    schedule_bytes = _payload_integer(copied, "schedule_bytes")
    if schedule_bytes != _ceil_div(columns * SCHEDULE_BITS_PER_CODE, 8):
        raise TrellisFormatError(
            "trellis footprint schedule-byte arithmetic drifted"
        )
    expected_offset_bits = (
        BLOCK_OFFSET_BITS if body_bits_per_row <= WIRE_V1_UINT32_MAX else 64
    ) if layout == LAYOUT_TIGHT_OFFSETS else 0
    offset_bits = _payload_integer(copied, "block_offset_bits")
    if offset_bits != expected_offset_bits:
        raise TrellisFormatError(
            "trellis footprint block-offset width arithmetic drifted"
        )
    offset_bytes = _payload_integer(copied, "block_offset_bytes")
    expected_offset_bytes = (
        (block_count + 1) * (expected_offset_bits // 8)
        if expected_offset_bits
        else 0
    )
    if offset_bytes != expected_offset_bytes:
        raise TrellisFormatError(
            "trellis footprint block-offset byte arithmetic drifted"
        )

    alphabet_by_rate = copied.get("alphabet_bytes_by_rate")
    if not isinstance(alphabet_by_rate, dict) or any(
        not isinstance(key, str)
        or type(value) is not int
        or value < 0
        for key, value in alphabet_by_rate.items()
    ):
        raise TrellisFormatError(
            "trellis footprint alphabet byte table is not canonical"
        )
    for key, value in alphabet_by_rate.items():
        try:
            alphabet_rate = int(key)
        except ValueError as exc:
            raise TrellisFormatError(
                "trellis footprint alphabet byte-table rate is invalid"
            ) from exc
        if (
            str(alphabet_rate) != key
            or not 1 <= alphabet_rate <= spec.shaped_max_rate
            or value != (
                ALPHABET_DIRECTORY_BYTES_PER_RATE
                + (1 << (alphabet_rate + 1))
            )
        ):
            raise TrellisFormatError(
                "trellis footprint alphabet byte-table entry drifted"
            )
    alphabet_bytes = _payload_integer(copied, "alphabet_bytes")
    if alphabet_bytes != sum(alphabet_by_rate.values()):
        raise TrellisFormatError(
            "trellis footprint alphabet-byte arithmetic drifted"
        )
    if alphabet_bytes > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "trellis footprint alphabet blob exceeds the wire-v1 uint32 field"
        )
    expected_scale_bytes = (
        rows * _ceil_div(columns, 16)
        if spec.family == E2M1_FAMILY
        else rows * 4
    )
    scale_bytes = _payload_integer(copied, "scale_bytes")
    if scale_bytes != expected_scale_bytes:
        raise TrellisFormatError(
            "trellis footprint scale-byte arithmetic drifted"
        )
    if scale_bytes > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "trellis footprint scale plane exceeds the wire-v1 uint32 field"
        )

    sidecar_bytes = _payload_integer(copied, "sidecar_header_bytes")
    structural = _payload_integer(
        copied, "structural_side_information_bytes"
    )
    if structural != (
        WIRE_HEADER_BYTES + schedule_bytes + offset_bytes + alphabet_bytes
    ):
        raise TrellisFormatError(
            "trellis footprint structural side-information arithmetic drifted"
        )
    wire_side = _payload_integer(copied, "wire_side_information_bytes")
    if wire_side != structural + scale_bytes:
        raise TrellisFormatError(
            "trellis footprint wire side-information arithmetic drifted"
        )
    side = _payload_integer(copied, "side_information_bytes")
    if side != wire_side + sidecar_bytes:
        raise TrellisFormatError(
            "trellis footprint side-information arithmetic drifted"
        )
    total = _payload_integer(copied, "total_bytes")
    if total != body_bytes + side:
        raise TrellisFormatError("trellis footprint total-byte arithmetic drifted")

    body_bpw = _payload_float(copied, "body_bpw")
    if body_bpw != body_bits_per_row / columns:
        raise TrellisFormatError("trellis footprint body_bpw arithmetic drifted")
    exact_bpw = _payload_float(copied, "exact_bpw")
    if exact_bpw != 8.0 * total / (rows * columns):
        raise TrellisFormatError("trellis footprint exact_bpw arithmetic drifted")
    if _payload_integer(copied, "expanded_weight_resident_bytes") != 0:
        raise TrellisFormatError(
            "trellis footprint cannot claim expanded weight residency"
        )
    for identity_field in (
        "schedule_identity_sha256",
        "alphabet_identity_sha256",
    ):
        identity = copied.get(identity_field)
        if not isinstance(identity, str) or _SHA256.fullmatch(identity) is None:
            raise TrellisFormatError(
                f"trellis footprint {identity_field} must be lowercase SHA-256"
            )
    if copied.get("pre_render_recipe_identity_scope") != (
        "layout_and_byte_recipe_without_encoded_body_or_scale_values"
    ):
        raise TrellisFormatError(
            "trellis footprint pre-render recipe identity scope drifted"
        )
    if copied.get("rendered_wire_identity_sha256") is not None:
        raise TrellisFormatError(
            "a pre-render footprint cannot claim a rendered wire identity"
        )
    if copied.get("producer_eligible") is not False:
        raise TrellisFormatError(
            "a Trellis research footprint cannot claim producer eligibility"
        )
    return copied


def trellis_tensor_payload_breakdown(
    shape: Sequence[int],
    *,
    family: str | TrellisFamily,
    body_rate_q256: int,
    layout: str,
    schedule: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    sidecar_header_bytes: int = 0,
) -> dict[str, object]:
    """Return exact serialized tensor-data bytes for one 2-D Linear weight.

    The 88-byte ``gridbook.trellis.wire.v1`` binary header is mandatory and
    always charged.  ``sidecar_header_bytes`` is additional exporter metadata:
    it may be zero when carried by safetensors metadata or nonzero for a
    dedicated sidecar.  Container/filesystem overhead is intentionally not
    estimated here.
    """

    spec = get_trellis_family(family)
    rate = validate_body_rate_q256(spec, body_rate_q256)
    dims = tuple(shape)
    if (
        len(dims) != 2
        or any(type(value) is not int or value <= 0 for value in dims)
    ):
        raise TrellisFormatError(
            f"trellis tensor shape must be two positive integers, got {dims}"
        )
    rows, columns = dims
    if rows > WIRE_V1_UINT32_MAX or columns > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "gridbook.trellis.wire.v1 rows and columns must fit the "
            "unsigned 32-bit header fields"
        )
    normalized_schedule = validate_schedule(
        spec, rate, schedule, layout=layout,
    )
    if len(normalized_schedule) != columns:
        raise TrellisFormatError(
            f"schedule has {len(normalized_schedule)} columns for shape {dims}"
        )
    normalized_alphabets = validate_alphabets(
        spec, normalized_schedule, alphabets,
    )
    if type(sidecar_header_bytes) is not int or sidecar_header_bytes < 0:
        raise TrellisFormatError("sidecar_header_bytes must be nonnegative")

    block_count = _ceil_div(columns, SUPERBLOCK_WEIGHTS)
    body_bits_per_row = sum(normalized_schedule)
    unpadded_body_bytes_per_row = _ceil_div(body_bits_per_row, 8)
    body_row_stride_bytes = _align(
        unpadded_body_bytes_per_row, ROW_ALIGNMENT_BYTES,
    )
    body_padding_bytes = rows * (
        body_row_stride_bytes - unpadded_body_bytes_per_row
    )
    body_bytes = rows * body_row_stride_bytes

    # gridbook.trellis.wire.v1 serializes the full tensor-shared schedule as
    # nibbles for both families and both layouts.  Fixed quota removes only
    # the redundant offset table; it does not repeat one 256-column template.
    schedule_bits_per_code = SCHEDULE_BITS_PER_CODE
    schedule_bytes = _ceil_div(columns * schedule_bits_per_code, 8)
    offset_width_bits = (
        BLOCK_OFFSET_BITS if body_bits_per_row <= 0xFFFFFFFF else 64
    )
    block_offset_bytes = (
        (block_count + 1) * (offset_width_bits // 8)
        if layout == LAYOUT_TIGHT_OFFSETS
        else 0
    )
    # The v1 alphabet blob has a uint8 rate + uint16 count directory entry and
    # then one native code byte per slot.  E2M1 nibbles are intentionally not
    # repacked here; the CUDA LUT ABI consumes code bytes directly.  Duplicate
    # E4M3 R7 slots remain physical entries.
    alphabet_bytes_by_rate = {
        str(alphabet_rate): ALPHABET_DIRECTORY_BYTES_PER_RATE + len(codes)
        for alphabet_rate, codes in normalized_alphabets.items()
    }
    alphabet_bytes = sum(alphabet_bytes_by_rate.values())

    if spec.family == E2M1_FAMILY:
        # One native E4M3 scale byte per group of 16 weights.
        scale_bytes = rows * _ceil_div(columns, 16)
    else:
        # One FP32 scale per output row.
        scale_bytes = rows * 4
    if alphabet_bytes > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "trellis alphabet blob does not fit the wire-v1 uint32 header"
        )
    if scale_bytes > WIRE_V1_UINT32_MAX:
        raise TrellisFormatError(
            "trellis scale plane does not fit the wire-v1 uint32 header"
        )

    structural_side_information_bytes = (
        WIRE_HEADER_BYTES
        + schedule_bytes
        + block_offset_bytes
        + alphabet_bytes
    )
    # Gridbook account().side_bytes includes the scale plane.  Keep a named
    # structural subtotal for callers that need the non-scale components, but
    # never report that subtotal as complete wire side information.
    wire_side_information_bytes = (
        structural_side_information_bytes + scale_bytes
    )
    side_information_bytes = (
        wire_side_information_bytes + sidecar_header_bytes
    )
    total_bytes = body_bytes + side_information_bytes
    weights = rows * columns
    body_bpw = body_bits_per_row / columns
    exact_bpw = 8.0 * total_bytes / weights

    # Aggregate byte counts are insufficient recipe identities: two schedules
    # can have the same sum and two alphabets can have the same slot count while
    # describing different layouts.  The body indices, scale-plane values, and
    # E2M1 global scale are renderer outputs and are deliberately absent here;
    # consequently the digest below is a pre-render recipe identity, never a
    # physical Gridbook wire-content identity.
    schedule_identity_sha256 = _sha256({
        "schema": "prismaquant.trellis_schedule.v1",
        "scope": "tensor_input_column_shared_across_rows",
        "values": list(normalized_schedule),
    })
    alphabet_identity_sha256 = _sha256({
        "schema": "prismaquant.trellis_alphabets.v1",
        "family": spec.family,
        "native_codes_by_rate": {
            str(rate): list(codes)
            for rate, codes in normalized_alphabets.items()
        },
    })
    body: dict[str, object] = {
        "schema": TRELLIS_TENSOR_PAYLOAD_SCHEMA,
        "wire_schema": TRELLIS_WIRE_SCHEMA,
        "family": spec.family,
        "format": spec.format_name(rate),
        "grid": spec.grid,
        "shape": [rows, columns],
        "body_rate_q256": rate,
        "body_bpw": body_bpw,
        "layout": layout,
        "superblock_weights": SUPERBLOCK_WEIGHTS,
        "block_count": block_count,
        "body_bits_per_row": body_bits_per_row,
        "unpadded_body_bytes_per_row": unpadded_body_bytes_per_row,
        "body_row_stride_bytes": body_row_stride_bytes,
        "body_padding_bytes": body_padding_bytes,
        "body_bytes": body_bytes,
        "wire_header_bytes": WIRE_HEADER_BYTES,
        "scale_contract": spec.scale_contract,
        "scale_bytes": scale_bytes,
        "schedule_scope": "tensor_input_column_shared_across_rows",
        "schedule_bits_per_code": schedule_bits_per_code,
        "schedule_bytes": schedule_bytes,
        "schedule_identity_sha256": schedule_identity_sha256,
        "block_offset_bits": (
            offset_width_bits if layout == LAYOUT_TIGHT_OFFSETS else 0
        ),
        "block_offset_bytes": block_offset_bytes,
        "alphabet_bytes_by_rate": alphabet_bytes_by_rate,
        "alphabet_bytes": alphabet_bytes,
        "alphabet_identity_sha256": alphabet_identity_sha256,
        "sidecar_header_bytes": sidecar_header_bytes,
        "structural_side_information_bytes": (
            structural_side_information_bytes
        ),
        "wire_side_information_bytes": wire_side_information_bytes,
        "side_information_bytes": side_information_bytes,
        "total_bytes": total_bytes,
        "exact_bpw": exact_bpw,
        "expanded_weight_resident_bytes": 0,
        "pre_render_recipe_identity_scope": (
            "layout_and_byte_recipe_without_encoded_body_or_scale_values"
        ),
        "rendered_wire_identity_sha256": None,
        "producer_eligible": False,
    }
    identity_sha256 = _sha256(body)
    return {
        **body,
        "pre_render_recipe_identity_sha256": identity_sha256,
    }


__all__ = [
    "ALPHABET_DIRECTORY_BYTES_PER_RATE",
    "BLOCK_OFFSET_BITS",
    "ROW_ALIGNMENT_BYTES",
    "SCHEDULE_BITS_PER_CODE",
    "TRELLIS_TENSOR_PAYLOAD_SCHEMA",
    "WIRE_HEADER_BYTES",
    "WIRE_V1_UINT32_MAX",
    "trellis_tensor_payload_breakdown",
    "validate_trellis_tensor_payload_breakdown",
]
