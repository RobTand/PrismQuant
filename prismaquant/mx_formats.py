"""Shared OCP-MX quantization adapters backed by compressed-tensors.

MXFP8_E4M3 is a commodity compressed-tensors/vLLM format: grouped FP8
values with E8M0 power-of-two scales. Keep the qparam and cast semantics in
one place and defer scale generation to compressed-tensors, which is the
load/export authority for this on-disk representation.

MXFP8_UE8M0_G32 (bottom half of this module) is the SECOND MX-FP8 contract
this repo ships. Same element grid and same group of 32, but a different
scale RULE and a different on-disk scale dtype, and the difference is
load-bearing rather than cosmetic — see ``mxfp8_ue8m0_shared_exponent``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.quant_scheme import MXFP8
from compressed_tensors.quantization.utils.helpers import calculate_qparams

from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm


@dataclass(frozen=True)
class MXFP8Result:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def e8m0_to_scale(
    e8m0_uint8: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decode compressed-tensors E8M0 scale bytes to float32 powers of two."""
    target_device = device if device is not None else e8m0_uint8.device
    return decompress_mx_scale(e8m0_uint8.to(device=target_device)).to(
        device=target_device,
        dtype=torch.float32,
    )


def mxfp8_e4m3_qdq(
    values: torch.Tensor,
    *,
    group_size: int = 32,
    fallback_plain_activation: bool = False,
) -> MXFP8Result:
    """Quantize/dequantize values as MXFP8_E4M3 using compressed-tensors.

    ``values`` may be any rank as long as the final dimension is the feature
    dimension. Scales are generated per final-dimension group and returned as
    uint8 E8M0 metadata with shape ``values.shape[:-1] + (K / group_size,)``.
    """
    orig_shape = values.shape
    if len(orig_shape) == 0:
        raise ValueError("MXFP8_E4M3 requires at least one tensor dimension")
    cols = int(orig_shape[-1])
    if cols % group_size != 0:
        if fallback_plain_activation:
            result = fp8_dynamic_activation_qdq_vllm(
                values,
                element_dtype=torch.float8_e4m3fn,
                element_max=float(torch.finfo(torch.float8_e4m3fn).max),
            )
            return MXFP8Result(
                quant=result.quant,
                scale=result.scale,
                dequant=result.dequant,
            )
        raise ValueError(
            f"MXFP8_E4M3 group_size={group_size} does not divide K={cols}"
        )

    rows = values.to(torch.float32).reshape(-1, cols)
    grouped = rows.reshape(rows.shape[0], cols // group_size, group_size)
    args = MXFP8["weights"]
    scale, zero_point = calculate_qparams(
        grouped.amin(dim=-1),
        grouped.amax(dim=-1),
        args,
    )
    quant_rows = quantize(
        rows,
        scale,
        zero_point,
        args,
        dtype=torch.float8_e4m3fn,
    )
    e8m0 = compress_mx_scale(scale, torch.uint8)
    scale_f = e8m0_to_scale(e8m0, device=values.device)
    dequant = (
        quant_rows.reshape_as(grouped).to(torch.float32)
        * scale_f.unsqueeze(-1)
    )
    scale_shape = (*orig_shape[:-1], cols // group_size)
    return MXFP8Result(
        quant=quant_rows.reshape(orig_shape),
        scale=e8m0.reshape(scale_shape),
        dequant=dequant.reshape(orig_shape),
    )


def mxfp8_e4m3_weight_qdq(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
) -> MXFP8Result:
    return mxfp8_e4m3_qdq(weight, group_size=group_size)


def mxfp8_e4m3_activation_qdq_vllm(
    activation: torch.Tensor,
    *,
    group_size: int = 32,
) -> MXFP8Result:
    return mxfp8_e4m3_qdq(
        activation,
        group_size=group_size,
        fallback_plain_activation=True,
    )


# ---------------------------------------------------------------------------
# MXFP8_UE8M0_G32 — the saturating-ceil MX-FP8 contract
# ---------------------------------------------------------------------------

MXFP8_UE8M0_GROUP = 32
E4M3_MAX = 448.0                 # OCP FP8 E4M3 max finite magnitude
E8M0_BIAS = 127
E8M0_MIN_EXP = -127              # byte 0
E8M0_MAX_EXP = 127               # byte 254; byte 255 is the E8M0 NaN code


def mxfp8_ue8m0_shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Smallest integer ``e`` with ``amax / 2**e <= 448``, clamped to E8M0.

    THIS IS THE WHOLE POINT OF THE FORMAT, so it is worth being explicit
    about how it differs from the OCP/compressed-tensors rule that
    ``mxfp8_e4m3_qdq`` above delegates to.

    The OCP rule derives the shared exponent by rounding the block amax to a
    power of two and subtracting the element-format exponent. That can pick an
    exponent SMALLER than the one the data was already stored at, which scales
    the group UP — and scaling up is not free at the bottom of the E4M3 grid,
    because the smallest normals fall off the subnormal ladder when halved.
    Concretely, a group holding both 448 and 1.125*2^-6 gets shared exponent
    +1 from the OCP rule, and 1.125*2^-6 / 2 = 9*2^-10 is not an E4M3 value:
    it rounds, and the round-trip is lossy.

    The ceil rule below picks the smallest exponent that does not CLIP, i.e.
    it never scales a group up past the point where its own maximum still
    fits. For any input whose values are already E4M3 codes times a shared
    power of two — which is exactly the DeepSeek ``fp8_e4m3_ue8m0_block128``
    body convention — this yields ``e <= E_block``, so every element is an
    E4M3 code scaled DOWN by a power of two, which is always exactly
    representable. That makes the re-encode lossless; see
    ``tests/test_mxfp8_ue8m0.py``.

    Computed from ``frexp`` rather than ``log2`` so it is exact integer
    arithmetic with no rounding at the boundary: ``amax = f * 2**E`` with
    ``f in [0.5, 1)`` and ``448 = 0.875 * 2**9``, so the smallest admissible
    exponent is ``E - 9`` when ``f <= 0.875`` and ``E - 8`` otherwise.

    That is not a stylistic preference. The obvious spelling,
    ``ceil(log2(amax / 448))``, is wrong about once in 4e5 group maxima in
    float32: when ``amax / 448`` rounds to exactly a power of two the ``log2``
    lands on an integer and ``ceil`` returns it unchanged, giving an exponent
    one too small and an element that saturates at 448 — the very thing the
    rule exists to prevent. Measured over 6e6 sampled maxima: 15 saturations
    for the log2 form, 0 for this one. The Gridbook consumer's reference
    quantizer (gridbook/mxfp8.py ``quantize_mxfp8``) currently uses the log2
    form on its ACTIVATION path, so until that is reconciled the served A side
    can saturate on a group this emulation would not. Producer-side weights
    are unaffected: they are encoded here.

    An all-zero group has no constraint; it is pinned to the bottom of the
    range (byte 0), which is also what compressed-tensors emits, so the two
    encoders agree on the degenerate case.
    """
    amax_f = amax.to(torch.float32)
    frac, exp = torch.frexp(amax_f)
    shared = exp - 9 + (frac > 0.875).to(exp.dtype)
    shared = torch.where(
        amax_f > 0, shared, torch.full_like(shared, E8M0_MIN_EXP)
    )
    return shared.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)


def mxfp8_ue8m0_qdq(
    values: torch.Tensor,
    *,
    group_size: int = MXFP8_UE8M0_GROUP,
) -> MXFP8Result:
    """Quantize/dequantize as MXFP8_UE8M0_G32.

    ``values`` may be any rank; groups are taken along the final (reduce/K)
    dimension. Returns the E4M3 elements, the shared scales as
    ``float8_e8m0fnu`` of shape ``values.shape[:-1] + (K / group_size,)``, and
    the dequantized reconstruction.

    A final dimension that is not a multiple of ``group_size`` is zero-padded
    to one and sliced back. Zeros cannot move a group's max-abs, so the pad is
    exact for the real columns; the returned scale plane then covers
    ``ceil(K / group_size)`` groups. The exporter requires exact divisibility
    (``check_format_applicability`` masks the format otherwise) and asserts it
    separately — this path exists so direct callers (cost measurement on an
    odd synthetic layer) get a sane answer instead of an exception.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    orig_shape = values.shape
    if len(orig_shape) == 0:
        raise ValueError("MXFP8_UE8M0_G32 requires at least one dimension")
    cols = int(orig_shape[-1])
    pad = (-cols) % group_size
    rows = values.to(torch.float32).reshape(-1, cols)
    if pad:
        rows = torch.nn.functional.pad(rows, (0, pad))
    n_groups = rows.shape[-1] // group_size
    grouped = rows.reshape(rows.shape[0], n_groups, group_size)

    shared = mxfp8_ue8m0_shared_exponent(grouped.abs().amax(dim=-1))
    # ldexp scales by a power of two exactly, and never materializes 2**e
    # (which would be subnormal in fp32 near the bottom of the E8M0 range).
    scaled = torch.ldexp(grouped, (-shared).unsqueeze(-1))
    # The ceil rule already guarantees |scaled| <= 448 for every in-range
    # input; the clamp only bites when `shared` hit the E8M0 rail, and it is
    # what keeps an out-of-range magnitude from casting to the E4M3 NaN code.
    quant = scaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    dequant = torch.ldexp(quant.to(torch.float32), shared.unsqueeze(-1))

    quant = quant.reshape(rows.shape)
    dequant = dequant.reshape(rows.shape)
    if pad:
        quant = quant[:, :cols]
        dequant = dequant[:, :cols]
    e8m0 = (shared + E8M0_BIAS).to(torch.uint8).view(torch.float8_e8m0fnu)
    return MXFP8Result(
        quant=quant.reshape(orig_shape),
        scale=e8m0.reshape(*orig_shape[:-1], n_groups),
        dequant=dequant.reshape(orig_shape),
    )


def mxfp8_ue8m0_weight_qdq(
    weight: torch.Tensor,
    *,
    group_size: int = MXFP8_UE8M0_GROUP,
) -> MXFP8Result:
    return mxfp8_ue8m0_qdq(weight, group_size=group_size)


def mxfp8_ue8m0_activation_qdq(
    activation: torch.Tensor,
    *,
    group_size: int = MXFP8_UE8M0_GROUP,
) -> MXFP8Result:
    """The A side of the MXFP8_UE8M0_G32 lane: DYNAMIC per-32 E8M0 + E4M3.

    Deliberately the same function as the weight path, because the serving
    lane uses the same rule on both sides: activations are quantized per 32
    contiguous reduction-axis elements with the same saturating-ceil shared
    exponent, computed per forward (dynamic — there is no static global scale
    to fit, unlike NVFP4's ``nv_fp4_with_static_gs`` activation contract).

    Kept as a named entry point anyway so the registry's A side reads as a
    deliberate declaration rather than a reuse, and so a future divergence
    between the two sides has an obvious place to land.
    """
    return mxfp8_ue8m0_qdq(activation, group_size=group_size)
