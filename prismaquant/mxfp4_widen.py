"""Exact MXFP4 -> MXFP8 widening: a transcode with no quantization step.

WHY THIS EXISTS. DSv4-Flash stores its routed experts — body *and* MTP — as
OCP-MX FP4: E2M1 nibble pairs packed two-per-byte along the reduce dim, with
one E8M0 (UE8M0) power-of-two scale per 32 logical elements. Gridbook's served
lanes for that wire are format-specific, and the MXFP8 dense lane
(``mxfp8_e4m3_e8m0_g32``) reads a *different* element plane. Widening lets a
unit that is only available as MXFP4 in the source reach the MXFP8 lane without
inventing any numerics.

WHY IT IS EXACT, AND WHY THAT IS A PROVABLE CLAIM RATHER THAN A HOPE. The E2M1
value set is::

    {0, 0.5, 1, 1.5, 2, 3, 4, 6} x {+, -}

Every one of those 15 distinct values is exactly representable in E4M3: each is
a 2-significant-bit number with an exponent well inside E4M3's range, so the
E4M3 encoding of the value is the value. Therefore widening changes the *width*
of the element plane and nothing about the numbers on it. This is a transcode,
not a requantization: there is no rounding rule, no scale search, no error term
to price. ``tests/test_mxfp4_widen.py`` pins the claim by exhausting all 256
byte codes rather than sampling.

WHY THE SCALE PLANE IS CARRIED BYTE-IDENTICAL. Both wires use the SAME grouping
— one E8M0 byte per 32 logical elements along the reduce dim — so the scale
tensor is already in the target's layout. Re-deriving it would be strictly worse
than copying it: a recomputed scale could differ in the one place the source is
deliberately unusual (0xFF, which the OCP MX v1.0 spec defines as NaN rather
than 2^128), and a NaN block must stay a NaN block. Copying makes the scale side
bit-exact by construction rather than by agreement.

THE ONE NORMALIZATION, STATED OUT LOUD. E2M1 code ``0x8`` is negative zero. This
module emits POSITIVE zero for it, matching the repo's own MXFP4 decode LUT in
``layer_streaming._read_layer_to_device`` (whose ``pair_lut`` maps both zero
codes to ``+0.0``). The dequantized *values* are therefore identical to the
reference decoder's for every input, which is the property that matters; a
consumer that distinguished -0.0 from +0.0 in a weight would already disagree
with how this repo reads the source.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not decide *whether* a unit
should be widened, does not touch the exporter's declaration, and does not
claim a serving route. Widening doubles the element plane (4 -> 8 bits/weight,
scales unchanged), and the MXFP8 route for *grouped/MoE* stacks has never been
audited on sm_121 — and since 2026-09-02 the DENSE half is gone too, because
the only MXFP8 lane that ever served it was Gridbook's
(archive/gridbook_lane_2026-09-02/). Producing these bytes is a capability;
serving them is a separate, evidence-gated question, and today no sanctioned
lane answers it either way.

Standard-library + torch only, so the exactness property is testable on CPU
with neither compressed-tensors nor vLLM present.
"""
from __future__ import annotations

import torch


__all__ = [
    "E2M1_VALUES",
    "GROUP_SIZE",
    "MXFP8_GROUPED_ROUTE_STATUS",
    "MXFP8_GROUPED_ROUTE_EVIDENCE",
    "Mxfp8Widened",
    "e2m1_to_e4m3_table",
    "mxfp4_source_to_mxfp8",
    "dequantize_mxfp4_source",
    "dequantize_mxfp8",
]

#: SERVING STATUS OF THE BYTES THIS MODULE PRODUCES — recorded here, next to
#: the code that makes them, so the capability cannot be mistaken for a
#: serving claim.
#:
#: The string is the ``route_status`` vocabulary from
#: ``allocator_candidates`` (BACKED / PENDING / BLOCKED), spelled literally
#: rather than imported: this module is torch-only by design so its exactness
#: property stays testable without the allocator's dependency stack. The value
#: is pinned against the real enum in ``tests/test_mxfp4_widen.py``.
#:
#: Widening is only interesting for a unit that is MXFP4 in the source, and in
#: DSv4-Flash every such unit is a ROUTED-EXPERT stack — a grouped/MoE GEMM.
#: Gridbook's MXFP8 lane (``mxfp8_e4m3_e8m0_g32``) was DENSE-ONLY; the audited
#: grouped route on sm_121 is Marlin, and Marlin was audited for MXFP4, not
#: MXFP8. So there was no measured grouped MXFP8 route on our target, and this
#: module must not imply one. The status did not improve when that lane was
#: retired on 2026-09-02 (``archive/gridbook_lane_2026-09-02/``): it got
#: strictly worse, because the dense route went with it. ``pending`` is still
#: the honest word -- nothing was ever MEASURED and refused for the grouped
#: case, which is what ``blocked`` would claim.
MXFP8_GROUPED_ROUTE_STATUS = "pending"
MXFP8_GROUPED_ROUTE_EVIDENCE = (
    "Gridbook's MXFP8 dense lane served LINEAR units only "
    "(gridbook/mxfp8_dense_lane.py, unit_kind='linear'); no grouped/MoE MXFP8 "
    "kernel was ever audited on sm_121. The sm_121 grouped route that IS "
    "audited is Marlin, for mxfp4_e2m1_ue8m0_g32. Since 2026-09-02 the dense "
    "half is gone too: that lane was retired "
    "(archive/gridbook_lane_2026-09-02/) and no sanctioned lane -- vanilla "
    "vLLM, llama.cpp/vLLM-GGUF, the Tessera plugin -- serves MXFP8_UE8M0_G32 "
    "at any unit kind. Widened stacks are producible and servable nowhere; "
    "the DENSE case, which used to ride an OPT-IN Gridbook lane with no new "
    "audit, now needs a route before it needs an audit."
)

#: The 16 E2M1 code points in code order (index == nibble value). Code 0x8 is
#: negative zero in the format; it is listed as ``0.0`` because that is the
#: value this module and ``layer_streaming``'s decode LUT both materialize.
E2M1_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

#: Elements per E8M0 scale, on BOTH wires. Shared value, so a mismatch is a
#: geometry bug rather than a conversion parameter.
GROUP_SIZE = 32


class Mxfp8Widened:
    """The two planes an MXFP8 unit ships. Plain container, no behaviour."""

    __slots__ = ("weight", "weight_scale")

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        self.weight = weight
        self.weight_scale = weight_scale

    def __iter__(self):
        yield self.weight
        yield self.weight_scale

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Mxfp8Widened(weight={tuple(self.weight.shape)}"
                f":{self.weight.dtype}, weight_scale="
                f"{tuple(self.weight_scale.shape)}:{self.weight_scale.dtype})")


def e2m1_to_e4m3_table(device: torch.device | None = None) -> torch.Tensor:
    """``(256, 2)`` byte -> (low nibble, high nibble) E4M3 element pair.

    Low nibble is the EVEN logical element, so flattening the trailing pair
    dimension lands elements in logical order — the same packing convention
    ``layer_streaming`` decodes with, and the one ``inference/convert.py``'s
    ``FP4_TABLE`` documents for this checkpoint family.
    """

    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    codes = torch.arange(256, device=device)
    pair = torch.stack([values[codes & 0x0F], values[codes >> 4]], dim=-1)
    # Cast once, here: every entry is exactly representable, so this is a
    # relabelling of the table rather than a rounding of the data.
    return pair.to(torch.float8_e4m3fn)


def _check_geometry(packed: torch.Tensor, scale: torch.Tensor) -> int:
    """Validate the pair and return the logical reduce-dim length."""

    if packed.dim() < 2:
        raise ValueError(
            f"MXFP4 element plane must be at least 2-D, got shape "
            f"{tuple(packed.shape)}")
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError(
            f"MXFP4 element plane must be int8/uint8 nibble-pack, got "
            f"{packed.dtype}. A widened or already-decoded tensor is not a "
            f"legal input here — this function transcodes the SOURCE wire.")
    logical_k = int(packed.shape[-1]) * 2
    if logical_k % GROUP_SIZE != 0:
        raise ValueError(
            f"logical reduce dim {logical_k} is not a multiple of "
            f"{GROUP_SIZE}; MXFP4 and MXFP8 share the group-of-{GROUP_SIZE} "
            f"grid, so a ragged tail has no defined scale")
    expected = packed.shape[:-1] + (logical_k // GROUP_SIZE,)
    if tuple(scale.shape) != tuple(expected):
        raise ValueError(
            f"scale plane shape {tuple(scale.shape)} does not match the "
            f"element plane: expected {tuple(expected)} for a "
            f"{tuple(packed.shape)} nibble-pack (logical K={logical_k}, one "
            f"E8M0 byte per {GROUP_SIZE} elements). A transposed or "
            f"block-128 scale grid is numel-compatible with some of these "
            f"shapes and would silently mis-scale every group.")
    return logical_k


def mxfp4_source_to_mxfp8(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> Mxfp8Widened:
    """Widen a source MXFP4 weight/scale pair to the MXFP8 wire, exactly.

    ``packed`` is the checkpoint's ``[..., N, K/2]`` int8/uint8 nibble-pack;
    ``scale`` is its ``[..., N, K/32]`` E8M0 plane (``float8_e8m0fnu`` or the
    raw ``uint8`` it is stored as). Returns ``[..., N, K]``
    ``float8_e4m3fn`` elements plus the scale plane **carried through
    byte-identically** as ``float8_e8m0fnu``.

    Exactness: ``dequantize_mxfp8(*result)`` equals
    ``dequantize_mxfp4_source(packed, scale)`` elementwise, including NaN
    placement for 0xFF scale groups.
    """

    _check_geometry(packed, scale)
    table = e2m1_to_e4m3_table(device=packed.device)
    # int32 gather indices: the index VALUES are byte codes (0..255), so int32
    # is exact and halves the transient vs int64.
    codes = packed.view(torch.uint8).to(torch.int32)
    widened = table[codes]                      # [..., N, K/2, 2] e4m3
    widened = widened.reshape(*packed.shape[:-1], -1).contiguous()
    # Byte-identical scale carry. ``view`` reinterprets the same bytes; the
    # clone detaches the result from the caller's storage without touching a
    # single bit of it.
    carried = scale.view(torch.uint8).clone().view(torch.float8_e8m0fnu)
    return Mxfp8Widened(widened, carried)


def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes -> fp32 powers of two, with 0xFF as NaN per OCP MX v1.0.

    0xFF is NaN in the spec, not 2^128: ``exp2(128)`` is ``+inf``, which turns
    a 0xFF group into a mix of +-inf (nonzero elements) and NaN (zero elements,
    ``0 * inf``) instead of a uniformly NaN group.
    """

    raw = scale.view(torch.uint8)
    decoded = torch.exp2(raw.to(torch.float32) - 127.0)
    return torch.where(raw == 0xFF, torch.full_like(decoded, float("nan")),
                       decoded)


def dequantize_mxfp4_source(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference decode of the SOURCE MXFP4 wire. The oracle for the property.

    Mirrors ``layer_streaming._read_layer_to_device``'s step 3b exactly (same
    LUT, same low-nibble-first order, same 0xFF NaN rule), kept here so the
    exactness test does not depend on the streaming loader's batching path.
    """

    logical_k = _check_geometry(packed, scale)
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32,
                          device=packed.device)
    codes = packed.view(torch.uint8).to(torch.int32)
    pair = torch.stack([values[codes & 0x0F], values[codes >> 4]], dim=-1)
    elements = pair.reshape(*packed.shape[:-1], logical_k)
    grouped = elements.reshape(*packed.shape[:-1], logical_k // GROUP_SIZE,
                               GROUP_SIZE)
    grouped = grouped * _decode_e8m0(scale).unsqueeze(-1)
    return grouped.reshape(*packed.shape[:-1], logical_k).to(dtype)


def dequantize_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference decode of the MXFP8 wire, for parity checking."""

    logical_k = int(weight.shape[-1])
    if logical_k % GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP8 reduce dim {logical_k} is not a multiple of {GROUP_SIZE}")
    grouped = weight.to(torch.float32).reshape(
        *weight.shape[:-1], logical_k // GROUP_SIZE, GROUP_SIZE)
    grouped = grouped * _decode_e8m0(scale).unsqueeze(-1)
    return grouped.reshape(*weight.shape[:-1], logical_k).to(dtype)
