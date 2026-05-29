#!/usr/bin/env python3
"""export_native_compressed.py — materialize a PrismaQuant recipe as a
standard `compressed-tensors` checkpoint that vLLM serves natively.

This is the unified export path. Decoder layers are streamed from
safetensors one at a time: the model skeleton is built on meta via
`init_empty_weights`, head + embed + norm + lm_head + rotary stay
resident, and each decoder layer flows disk → quantize → emit → unload.
Small models pay the no-op cost of a LayerCache large enough to keep
everything resident; big models (Qwen3.5-122B at 244 GB BF16) fit
through the same path on a 121 GB host.

Reads the per-tensor format assignment produced by `allocator.py`
(layer_config.json) and emits a directory containing:

  - `model-*.safetensors` (sharded), with each Linear / packed-MoE
    tensor written under the standard compressed-tensors schema:
        <name>.weight_packed         (uint8, 4-bit packed for NVFP4)
        <name>.weight_scale          (fp8_e4m3fn for NVFP4 / e8m0 for MXFP8_E4M3/E5M2)
        <name>.weight_global_scale   (fp32, NVFP4 only)
        <name>.input_global_scale    (fp32, A4/A8 formats only)
    OR `<name>.weight` (passthrough in the source storage dtype) for
    layers in the uncompressed bucket.

  - `model.safetensors.index.json` matching the safetensors layout

  - `config.json` carrying a `quantization_config` with
    `format = mixed-precision` and one config_group per nominated
    format. Targets are explicit per-Linear regex anchors so vLLM's
    compressed-tensors dispatcher routes every parameter to the right
    scheme without ambiguity.

  - `mixed_native_manifest.json` summarizing the export (format
    histogram, ignore list, source recipe path) for traceability.

  - tokenizer / config files copied verbatim from the source.

Why this exists separately from llmcompressor's oneshot:
  - llmcompressor's QuantizationModifier matches nn.Linear modules. It
    does not handle 3D packed-expert tensors (Qwen3.5/3.6's
    `gate_up_proj` / `down_proj`), which silently fall back to dense
    bf16 in the standard pipeline.
  - llmcompressor pins transformers <5; transformers v5 is required to
    load Qwen3.6 (`qwen3_5_moe`). The two cannot coexist.

This exporter pins to transformers v5 for model load, uses the
compressed-tensors lib's `pack_fp4_to_uint8` reference (inlined to
avoid the lib's transformers-coupled `__init__`), and writes the
on-disk layout directly. vLLM's existing `compressed_tensors` and
`compressed_tensors_moe_w4a4_nvfp4` schemes load the result without
patches.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales
try:
    from accelerate import init_empty_weights
except ModuleNotFoundError:
    @contextmanager
    def init_empty_weights():
        with torch.device("meta"):
            yield
from safetensors.torch import save_file

from .allocator_candidates import check_format_applicability
from .fp8_dynamic import fp8_dynamic_weight_qdq
from .mx_formats import e8m0_to_scale, mxfp8_e4m3_qdq
from .serving_profiles import resolve_target_profile
from .layer_config import (
    canonicalize_assignment as _canonicalize_assignment,
    canonicalize_format,
)
from .model_profiles.qwen3_5 import Qwen3_5Profile
from .schemas import validate_layer_config_payload

# ---------------------------------------------------------------------------
# NVFP4 packing. The byte layout (two 4-bit indices/byte, element-0 low nibble,
# element-1 high nibble) matches compressed-tensors'
# `compressed_tensors.compressors.nvfp4.helpers.pack_fp4_to_uint8` and is
# verified byte-identical in tests. We pack indices directly rather than call
# that helper because (a) it takes a FLOAT tensor and runs its own argmin
# codebook assignment + clamping, whereas our scale-rule / JSO / four-over-six
# `_round_to_codebook` path already produces the indices; and (b) importing the
# library's package __init__ pulls in transformers internals that are not
# stable across the transformers 4.x -> 5.x break.
# ---------------------------------------------------------------------------
FLOAT_TO_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_MAX = 6.0     # max(|FLOAT_TO_E2M1|)
FP8_E4M3_MAX = 448.0  # max representable in torch.float8_e4m3fn
NVFP4_SCALE_RULE_ENV = "PRISMAQUANT_NVFP4_SCALE_RULE"
NVFP4_SCALE_RULE_STATIC_6 = "static_6"
NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE = "four_over_six_mse"
NVFP4_SCALE_RULE_JOINT_MSE = "joint_mse"
_NVFP4_SCALE_RULE_ALIASES = {
    "": NVFP4_SCALE_RULE_STATIC_6,
    "default": NVFP4_SCALE_RULE_STATIC_6,
    "static": NVFP4_SCALE_RULE_STATIC_6,
    "static_6": NVFP4_SCALE_RULE_STATIC_6,
    "six": NVFP4_SCALE_RULE_STATIC_6,
    "6": NVFP4_SCALE_RULE_STATIC_6,
    "4/6": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "4over6": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "four_over_six": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "four_over_six_mse": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "mse": NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
    "joint": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_mse": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale_opt": NVFP4_SCALE_RULE_JOINT_MSE,
    "joint_scale_optimization": NVFP4_SCALE_RULE_JOINT_MSE,
    "codebook_mse": NVFP4_SCALE_RULE_JOINT_MSE,
}

# Back-compat exports for unit tests that validate the Qwen3.5 naming
# and per-expert catch-all contract via the historical helper symbols.
_COMPAT_QWEN_PROFILE = Qwen3_5Profile()
PER_EXPERT_MOE_REGEX = _COMPAT_QWEN_PROFILE.per_expert_moe_regex()


def _to_vllm_internal_name(checkpoint_name: str) -> str:
    """Compatibility helper kept for unit tests.

    The production path is profile-driven via `profile.to_vllm_internal_name`;
    this helper preserves the historical Qwen3.5/3.6 mapping semantics
    without depending on a local vLLM install.
    """
    name = checkpoint_name
    if name.startswith("mtp."):
        return name
    if name == "lm_head":
        return "language_model.lm_head"
    if name.startswith("model.visual."):
        return name[len("model."):]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if (name.startswith("model.layers.")
            or name.startswith("model.embed_tokens")
            or name.startswith("model.norm")
            or name == "model"):
        return "language_model.model." + name[len("model."):]
    return name


def _nvfp4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(FLOAT_TO_E2M1, device=device, dtype=dtype)


def resolve_nvfp4_scale_rule(raw: str | None = None) -> str:
    """Canonicalize the NVFP4 block-scale rule.

    ``static_6`` is the compressed-tensors/llm-compressor default: every
    16-value block maps its maximum magnitude to FP4 code ±6.  FourOverSix
    evaluates max-to-6 and max-to-4 and keeps the lower block-MSE scale while
    preserving the same NVFP4 on-disk schema and vLLM runtime kernel.
    ``joint_mse`` extends that packer-compatible candidate set to every
    positive NVFP4 codebook level, making FourOverSix a strict subset.
    """
    if raw is None:
        raw = os.environ.get(NVFP4_SCALE_RULE_ENV, NVFP4_SCALE_RULE_STATIC_6)
    key = str(raw).strip().lower().replace("-", "_")
    try:
        return _NVFP4_SCALE_RULE_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted({
            NVFP4_SCALE_RULE_STATIC_6,
            NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
            NVFP4_SCALE_RULE_JOINT_MSE,
        }))
        raise ValueError(
            f"unsupported {NVFP4_SCALE_RULE_ENV}={raw!r}; "
            f"expected one of: {allowed}"
        ) from exc


def _nvfp4_scale_rule_from_env() -> str:
    override = globals().get("_NVFP4_SCALE_RULE", None)
    if override is not None:
        return resolve_nvfp4_scale_rule(str(override))
    return resolve_nvfp4_scale_rule()


def _decode_nvfp4_indices(
    fp4_idx: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _decode_nvfp4_indices_with_eff_scale(fp4_idx, scale.unsqueeze(-1))


def _decode_nvfp4_indices_with_eff_scale(
    fp4_idx: torch.Tensor,
    eff_scale: torch.Tensor,
) -> torch.Tensor:
    cb = _nvfp4_codebook(fp4_idx.device, dtype=torch.float32)
    abs_idx = fp4_idx & 0x7
    sign = -((fp4_idx >> 3).to(torch.float32) * 2 - 1)
    return sign * cb[abs_idx] * eff_scale


def _nvfp4_mse_for_group_scale(
    grouped: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    _idx, dq = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped,
        scale.unsqueeze(-1),
    )
    return (grouped - dq).pow(2).sum(dim=-1)


def _nvfp4_best_max_to_level_scale(
    grouped: torch.Tensor,
    levels: Sequence[float],
) -> torch.Tensor:
    """Pick the best max-to-codebook-level scale for each NVFP4 group.

    ``four_over_six_mse`` is the two-level subset ``levels=(6, 4)``. The
    joint scale rule extends that candidate set while staying final-pack
    compatible because every chosen scale is ``max_abs / codebook_level``.
    """

    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for level in levels:
        scale = max_abs / float(level)
        mse = _nvfp4_mse_for_group_scale(grouped, scale)
        if best_mse is None:
            best_mse = mse
            best_scale = scale
            continue
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        assert best_scale is not None
        best_scale = torch.where(take, scale, best_scale)
    assert best_scale is not None
    return best_scale


def _select_nvfp4_group_scales(
    grouped: torch.Tensor,
    *,
    scale_rule: str | None = None,
) -> torch.Tensor:
    """Return per-block real NVFP4 scales for ``grouped[..., group_size]``.

    The returned tensor has shape ``grouped.shape[:-1]``.  This function is
    the single scale-selection point used by RTN, GPTQ block quantization,
    scale-sweep initialization, packed experts, and final export packing.
    """
    rule = (
        _nvfp4_scale_rule_from_env()
        if scale_rule is None
        else resolve_nvfp4_scale_rule(scale_rule)
    )
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    scale_6 = max_abs / NVFP4_MAX
    if rule == NVFP4_SCALE_RULE_STATIC_6:
        return scale_6
    if rule == NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE:
        return _nvfp4_best_max_to_level_scale(grouped, (6.0, 4.0))
    if rule == NVFP4_SCALE_RULE_JOINT_MSE:
        return _nvfp4_best_max_to_level_scale(
            grouped,
            (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5),
        )
    raise AssertionError(f"unhandled NVFP4 scale rule: {rule!r}")


_NVFP4_JOINT_SCALE_LEVELS = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5)


def _env_int_clamped(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(lo), min(int(hi), int(value)))


def _nvfp4_effective_scale_from_real(
    scale_real: torch.Tensor,
    global_real: torch.Tensor,
    *,
    quantize_fp8: bool,
) -> torch.Tensor:
    fp8_scale = _nvfp4_fp8_scale_from_real(
        scale_real,
        global_real,
        quantize_fp8=quantize_fp8,
    )
    return _nvfp4_effective_scale_from_fp8(fp8_scale, global_real)


def _nvfp4_fp8_scale_from_real(
    scale_real: torch.Tensor,
    global_real: torch.Tensor,
    *,
    quantize_fp8: bool = True,
) -> torch.Tensor:
    fp8_scale = (
        scale_real / global_real.to(scale_real.device, dtype=torch.float32)
    ).clamp(0, FP8_E4M3_MAX)
    if quantize_fp8:
        fp8_scale = fp8_scale.to(torch.float8_e4m3fn)
    return fp8_scale


def _nvfp4_effective_scale_from_fp8(
    fp8_scale: torch.Tensor,
    global_real: torch.Tensor,
) -> torch.Tensor:
    return (
        fp8_scale.to(torch.float32)
        * global_real.to(fp8_scale.device, dtype=torch.float32)
    ).clamp_min(1e-12)


def _nvfp4_quantize_dequantize_with_eff_scale(
    values: torch.Tensor,
    eff_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_grid = (values / eff_scale.clamp_min(1e-12)).clamp(
        -NVFP4_MAX,
        NVFP4_MAX,
    )
    fp4_idx = _round_to_codebook(in_grid)
    return fp4_idx, _decode_nvfp4_indices_with_eff_scale(fp4_idx, eff_scale)


def _nvfp4_quant_dequant_with_eff_scale(
    values: torch.Tensor,
    eff_scale: torch.Tensor,
) -> torch.Tensor:
    _idx, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        values,
        eff_scale,
    )
    return dequant


def _nvfp4_quantize_grouped_codec(
    grouped: torch.Tensor,
    *,
    global_real: torch.Tensor,
    scale_real: torch.Tensor | None = None,
    scale_rule: str | None = None,
) -> _NVFP4CodecResult:
    grouped_f = grouped.to(torch.float32)
    if scale_real is None:
        scale_real = _select_nvfp4_group_scales(
            grouped_f,
            scale_rule=scale_rule,
        )
    scale_fp8 = _nvfp4_fp8_scale_from_real(
        scale_real.to(grouped_f.device, dtype=torch.float32),
        global_real,
        quantize_fp8=True,
    )
    eff_scale = _nvfp4_effective_scale_from_fp8(
        scale_fp8,
        global_real,
    ).unsqueeze(-1)
    fp4_idx, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped_f,
        eff_scale,
    )
    return _NVFP4CodecResult(
        indices=fp4_idx,
        scale=scale_fp8,
        dequant=dequant,
    )


def _select_nvfp4_joint_gptq_eff_scale(
    grouped: torch.Tensor,
    global_real: torch.Tensor,
    *,
    col_importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return GPTQ-time effective scales for Lift-style joint scale search.

    Candidate group scales include max-to-6 and max-to-4, so FourOverSix is a
    strict subset. Additional max-to-codebook-level choices keep the output
    representable by the same NVFP4 compressed-tensors metadata when the
    final packer uses ``joint_mse``.
    """

    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    weight = None
    if col_importance is not None:
        weight = col_importance.to(grouped.device, dtype=torch.float32)
        view_shape = (1,) * (grouped.dim() - 1) + (grouped.shape[-1],)
        weight = weight.reshape(view_shape)

    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for level in _NVFP4_JOINT_SCALE_LEVELS:
        scale = max_abs / float(level)
        eff = _nvfp4_effective_scale_from_real(
            scale,
            global_real,
            quantize_fp8=True,
        )
        dq = _nvfp4_quant_dequant_with_eff_scale(grouped, eff.unsqueeze(-1))
        err = (grouped - dq).pow(2)
        if weight is not None:
            err = err * weight
        mse = err.sum(dim=-1)
        if best_mse is None:
            best_mse = mse
            best_scale = eff
            continue
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        assert best_scale is not None
        best_scale = torch.where(take, eff, best_scale)
    assert best_scale is not None
    return best_scale.clamp_min(1e-12)


def _optimize_nvfp4_joint_global_real(
    weight: torch.Tensor,
    *,
    group_size: int,
    base_global_real: torch.Tensor,
) -> torch.Tensor:
    """Choose a tensor global scale jointly with group-scale candidates.

    The search is deliberately small and opt-in. It scores candidate tensor
    globals after FP8 realization of group scales, chunking rows so large
    Linears do not materialize a candidate dimension over the full weight.
    """

    grid = _env_int_clamped(
        "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID",
        5,
        1,
        33,
    )
    if grid <= 1:
        return base_global_real.clamp_min(1e-12)
    span_lo = float(os.environ.get(
        "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO",
        "0.75",
    ))
    span_hi = float(os.environ.get(
        "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI",
        "1.25",
    ))
    if not math.isfinite(span_lo) or span_lo <= 0.0:
        span_lo = 0.75
    if not math.isfinite(span_hi) or span_hi < span_lo:
        span_hi = max(span_lo, 1.25)
    W = weight.to(torch.float32)
    rows, cols = W.shape
    grouped = W.reshape(rows, cols // group_size, group_size)
    candidates = (
        base_global_real.to(W.device, dtype=torch.float32).reshape(())
        * torch.linspace(span_lo, span_hi, grid, device=W.device, dtype=torch.float32)
    ).clamp_min(1e-12)

    n_groups = cols // group_size
    bytes_per_row = max(1, n_groups * group_size * 4 * 4)
    row_chunk = min(rows, max(1, (512 * 1024 * 1024) // bytes_per_row))
    scores = torch.zeros((grid,), device=W.device, dtype=torch.float64)
    for idx, global_real in enumerate(candidates):
        total = torch.zeros((), device=W.device, dtype=torch.float64)
        for r0 in range(0, rows, row_chunk):
            r1 = min(r0 + row_chunk, rows)
            chunk = grouped[r0:r1]
            eff = _select_nvfp4_joint_gptq_eff_scale(chunk, global_real)
            dq = _nvfp4_quant_dequant_with_eff_scale(chunk, eff.unsqueeze(-1))
            total = total + (chunk - dq).pow(2).sum().to(torch.float64)
        scores[idx] = total
    best = int(scores.argmin().item())
    return candidates[best].reshape(()).clamp_min(1e-12)


def _round_to_codebook(values_in_grid: torch.Tensor) -> torch.Tensor:
    """Round per-element values (already scaled into the [-6, +6]
    NVFP4 grid) to the nearest codebook entry, using bucketize on the
    sorted absolute codebook. O(N log K) instead of O(N · K).

    Returns a Long tensor of 4-bit indices in [0, 15], where bit 3 is
    the sign bit and bits 0-2 are the abs-codebook index.
    """
    cb = _nvfp4_codebook(values_in_grid.device, dtype=torch.float32)
    abs_x = values_in_grid.abs().contiguous()
    idx = torch.bucketize(abs_x, cb)        # insertion: cb[idx-1] <= x < cb[idx]
    idx_lo = (idx - 1).clamp_min(0).clamp_max(cb.numel() - 1)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo_v = cb[idx_lo]
    hi_v = cb[idx_hi]
    pick_hi = (hi_v - abs_x).abs() < (abs_x - lo_v).abs()
    abs_idx = torch.where(pick_hi, idx_hi, idx_lo).long()
    sign_bit = torch.signbit(values_in_grid).to(torch.long) << 3
    return abs_idx + sign_bit                # [..., shape]; values 0-15


MXFP8_LEGACY_ALIAS = "MXFP8"
MXFP8_EXPLICIT_FORMATS = {"MXFP8_E4M3", "MXFP8_E5M2"}


@dataclass(frozen=True)
class _NVFP4CodecResult:
    indices: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _FP8CodecResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _MXFP8CodecResult:
    quant: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


@dataclass(frozen=True)
class _MXFP4CodecResult:
    indices: torch.Tensor
    packed: torch.Tensor
    scale: torch.Tensor
    dequant: torch.Tensor


def _canonical_export_format(fmt: str) -> str:
    fmt_u = str(fmt).strip().upper()
    if fmt_u == MXFP8_LEGACY_ALIAS:
        return "MXFP8_E4M3"
    return fmt_u


def _resolve_act_clip_quantile(default: str = "0.999") -> float | None:
    """Return the effective activation-clip quantile for GPTQ scoring."""
    raw = os.environ.get("PRISMAQUANT_ACT_CLIP_QUANTILE", default)
    if not raw:
        return None
    try:
        q = float(raw)
    except ValueError:
        return None
    return q if 0.0 < q < 1.0 else None


def _normalize_act_clip_rescale(mode: str | None) -> str:
    if mode is None:
        mode = "none"
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return "none"
    raise ValueError("activation clip rescaling is not supported")


def _rescale_clipped_activation_matrix(
    original: torch.Tensor,
    clipped: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Return clipped activations, rejecting retired row-rescale modes."""
    mode = _normalize_act_clip_rescale(mode)
    if mode == "none" or original.numel() == 0:
        return clipped
    raise ValueError("activation clip rescaling is not supported")


def _activation_matrix_for_gptq(
    activations: torch.Tensor,
    cols: int,
    *,
    device: torch.device | None = None,
    clip_threshold: float | None = None,
    clip_quantile: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten activations and apply the same optional clipping used by GPTQ.

    This is intentionally shared by the Hessian build, damping sweep
    evaluator, and do-no-harm gate. Mixing clipped optimization with
    unclipped local gates caused the full quality-win stack to undo part
    of the activation-clipping gain.
    """
    X = activations.detach().to(torch.float32)
    if device is not None:
        X = X.to(device)
    X = X.reshape(-1, cols)
    if clip_threshold is not None and clip_threshold > 0.0 and X.numel() > 0:
        thresh = torch.tensor(
            float(clip_threshold), device=X.device, dtype=X.dtype)
        X_clipped = X.clamp(min=-thresh, max=thresh)
        X = _rescale_clipped_activation_matrix(
            X,
            X_clipped,
            mode=_normalize_act_clip_rescale(clip_rescale),
        )
    else:
        q = _resolve_act_clip_quantile() if clip_quantile is None else clip_quantile
        if q is not None and 0.0 < q < 1.0 and X.numel() > 0:
            thresh = X.abs().quantile(float(q), dim=1, keepdim=True)
            X = X.clamp(min=-thresh, max=thresh)
    if row_weights is not None and X.numel() > 0:
        rw = _normalize_fisher_row_weights(row_weights, X.shape[0], X.device)
        if rw is not None:
            X = X * rw.sqrt().unsqueeze(1).to(dtype=X.dtype)
    return X


def _normalize_fisher_row_weights(
    row_weights: torch.Tensor | None,
    n_rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return non-negative Fisher row weights normalized to mean 1.

    The h-detail probe stores per-token gradient² weights.  For a local
    least-squares GPTQ objective, `X.T @ diag(g²) @ X` is equivalent to
    scaling each activation row by `sqrt(g²)`.  Normalizing the selected
    slice to mean 1 preserves the scale expected by the existing damping
    candidates and local error gates.
    """
    if row_weights is None or n_rows <= 0:
        return None
    try:
        rw = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
    except Exception:
        return None
    if rw.numel() < n_rows:
        return None
    rw = rw[:n_rows]
    rw = torch.where(torch.isfinite(rw), rw, torch.zeros_like(rw))
    rw = rw.clamp_min(0.0)
    mean = rw.mean()
    if not torch.isfinite(mean) or float(mean.item()) <= 0.0:
        return None
    rw = rw / mean.clamp_min(1e-12)
    try:
        clip = float(os.environ.get("PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP", "64"))
    except Exception:
        clip = 64.0
    if clip > 0.0:
        rw = rw.clamp_max(float(clip))
        mean2 = rw.mean()
        if torch.isfinite(mean2) and float(mean2.item()) > 0.0:
            rw = rw / mean2.clamp_min(1e-12)
    return rw


def _activation_col_importance_for_gptq(
    activations: torch.Tensor,
    cols: int,
    *,
    device: torch.device | None = None,
    clip_threshold: float | None = None,
    clip_quantile: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=device,
        clip_threshold=clip_threshold,
        clip_quantile=clip_quantile,
        clip_rescale=clip_rescale,
        row_weights=row_weights,
    )
    if X.numel() == 0:
        return torch.ones(cols, device=device, dtype=torch.float32)
    return X.pow(2).mean(dim=0).clamp_min(1e-12)


def _activation_weighted_weight_error(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    row_weights: torch.Tensor | None = None,
) -> float:
    """Score a rendered weight with the same cheap local gate as NVFP4.

    This is the diagonal form of output MSE: columns with larger calibration
    activation energy matter more. It is intentionally used only as a local
    do-no-harm gate; the production cache still uses the shared output scorer.
    """
    ref = reference_weight.to(torch.float32)
    cand = rendered_weight.to(device=ref.device, dtype=torch.float32)
    a2 = _activation_col_importance_for_gptq(
        activations,
        ref.shape[1],
        device=ref.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=row_weights,
    )
    return float((a2 * (ref - cand).pow(2).sum(dim=0)).sum())


def pack_fp4_indices(fp4_indices: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Pack a tensor of 4-bit indices (final dim must be even) into
    uint8, two indices per byte. Preserves leading dimensions.
    """
    if last_dim % 2 != 0:
        raise ValueError("nvfp4 pack requires an even last dim")
    pairs = fp4_indices.reshape(*fp4_indices.shape[:-1], last_dim // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


DEFAULT_INPUT_GLOBAL_SCALE = 1.0  # placeholder; overridden by calibration

# FP4 E2M1 maximum representable value. Used to rescale activations so
# they fit inside the FP4 grid after the per-tensor scale divide.
_FP4_E2M1_MAX = 6.0


def compute_nvfp4_input_global_scale(activations: torch.Tensor) -> float:
    """Per-tensor input_global_scale from cached activations.

    Returns `max(|activations|) / 6.0` so that `a / input_global_scale`
    lies in [-6, 6] — the representable range of FP4 E2M1 for per-group
    quant downstream. Activations can be any shape; we flatten for the
    max.
    """
    max_abs = float(activations.detach().abs().max().item())
    if max_abs <= 0.0:
        return float(DEFAULT_INPUT_GLOBAL_SCALE)
    # Use reciprocal convention matching vLLM's CompressedTensorsW4A4Nvfp4
    # which interprets input_global_scale as a *reciprocal* scale factor
    # applied when computing activation-quant group scales: a_q = a * s.
    # So s = FP4_MAX / max_abs means scaled_a ∈ [-FP4_MAX, +FP4_MAX].
    return _FP4_E2M1_MAX / max_abs


# Module-level cache populated by main() when --activation-cache-dir is
# provided. `_quantize_2d`'s NVFP4 branch consults it by recipe-name
# when no explicit override is passed in. Keyed by the recipe name
# (post-profile.live_to_recipe_name remap). None means "not computed".
_INPUT_GLOBAL_SCALES: dict[str, float] | None = None

# Module-level raw-activation cache populated by main() when
# --activation-cache-dir is provided AND any of the activation-aware
# passes (--gptq / --act-weighted-round / --scale-sweep) are enabled. Keyed
# by recipe name; values are 2D `[N, in_features]` float32 tensors
# (lazily upcast from the on-disk bfloat16 for numerical stability
# during Hessian + per-channel stats). None means "not loaded".
_CACHED_ACTIVATIONS: object | None = None
_ACTIVATION_CACHE_FINGERPRINT: dict[str, object] | None = None


class _LazyActivationCache:
    """ActivationIndex-backed mapping with a dict-like `.get()`.

    Export only needs a Linear's calibration rows while quantizing that
    one Linear. Preloading every activation tensor as float32 keeps
    tens of GiB resident for the entire export and OOMs large MoE
    checkpoints before the sharded writer runs. Keep scale calibration
    eager, but make raw activation reads demand-driven.

    TODO(perf): this whole probe -> cost -> export activation flow needs
    a larger redesign. Thousands of tiny `.pt` activation files plus
    late whole-checkpoint materialization are avoidable; use per-layer
    activation bundles and streaming safetensors writes.
    """

    def __init__(self, index):
        self.index = index
        self.loads = 0

    def get(self, name: str):
        if name not in self.index:
            return None
        self.loads += 1
        return self.index.load(name).to(torch.float32)


def _resolve_perturbed_x_export_inputs(root: str | Path) -> tuple[Path, Path]:
    """Return (layer_config, activation_cache_dir) from an iteration output."""
    root = Path(root)
    layer_config = root / "final_layer_config.json"
    summary_path = root / "summary.json"
    cache_dir: Path | None = None
    if summary_path.is_file():
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("final_layer_config"):
            layer_config = Path(summary["final_layer_config"])
            if not layer_config.is_absolute():
                layer_config = root / layer_config
        iterations = summary.get("iterations") or []
        if iterations:
            cache_info = iterations[-1].get("cache", {})
            cache_raw = cache_info.get("cache_dir")
            if cache_raw:
                cache_dir = Path(cache_raw)
                if not cache_dir.is_absolute():
                    cache_dir = root / cache_dir
    if cache_dir is None:
        caches = sorted(root.glob("activation_cache_iter_*"))
        if caches:
            cache_dir = caches[-1]
    if not layer_config.is_file():
        raise FileNotFoundError(
            f"perturbed-X layer config not found at {layer_config}"
        )
    if cache_dir is None or not cache_dir.is_dir():
        raise FileNotFoundError(
            f"perturbed-X activation cache not found under {root}"
        )
    return layer_config, cache_dir


class _LazyFisherDiagCache:
    """HDetailIndex-backed lazy cache for per-Linear Fisher diagonal.

    Mirrors `_LazyActivationCache` but loads `h_diag` tensors of shape
    `[out, in]` from the probe's per-Linear `.pt` blobs. Returns None
    when the requested name isn't in the index — caller (GPTQ wrapper)
    falls back to unweighted Hessian. Loads are demand-driven so the
    full Fisher cache (typically a few GB) doesn't sit resident."""

    def __init__(self, index):
        self.index = index
        self.loads = 0

    def get(self, name: str):
        if name not in self.index:
            return None
        self.loads += 1
        try:
            return self.index.load(name).to(torch.float32)
        except Exception:
            return None


def _activation_index_fingerprint(index, cache_dir: Path) -> dict[str, object]:
    """Cheap cache identity for export-cache invalidation.

    The layer export cache stores quantized tensors whose values depend
    on activation-cache contents. Hash names plus file size/mtime so
    changing the activation cache or pointing at a different cache dir
    invalidates stale layer_NNN.pt files without reading tensor bytes.
    """
    import hashlib
    import json as _json

    paths = getattr(index, "_paths", {})
    rows = []
    for name, path in sorted(paths.items()):
        st = path.stat()
        rows.append([name, path.name, st.st_size, st.st_mtime_ns])
    digest = hashlib.sha256(
        _json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "path": str(cache_dir.resolve()),
        "n_files": len(rows),
        "hash": digest,
    }


def _production_cache_format_candidates(fmt: str) -> tuple[str, ...]:
    fmt_u = str(fmt).upper()
    if fmt_u == MXFP8_LEGACY_ALIAS:
        return ("MXFP8_E4M3", MXFP8_LEGACY_ALIAS)
    if fmt_u == "MXFP8_E4M3":
        return ("MXFP8_E4M3", MXFP8_LEGACY_ALIAS)
    return (fmt_u,)


def _production_cache_name_candidates(name: str) -> tuple[str, ...]:
    names = [name]
    if name.endswith(".weight"):
        names.append(name[:-len(".weight")])
    if name.startswith("model.language_model."):
        names.append("model." + name[len("model.language_model."):])
    return tuple(dict.fromkeys(names))


def _production_cache_lookup_key(name: str, fmt: str):
    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    if hasattr(cache, "resolve_key"):
        key = cache.resolve_key(name, fmt)
        if key is not None:
            return key
    weights = getattr(cache, "weights", {}) or {}
    for cand_name in _production_cache_name_candidates(name):
        for cand_fmt in _production_cache_format_candidates(fmt):
            key = (cand_name, cand_fmt)
            if key in weights:
                return key
    return None

def _production_cache_expected_keys(
    assignment: dict[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    from prismaquant.production_weight_cache import is_uncached_packed_expert_qname

    keys: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for qname, fmt in assignment.items():
        cache_fmt = str(fmt).upper()
        if _canonical_export_format(cache_fmt) == "BF16":
            continue
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is None:
            if is_uncached_packed_expert_qname(qname):
                continue
            missing.append((qname, cache_fmt))
        else:
            keys.append(key)
    return keys, missing


def _production_cache_fingerprint(
    cache,
    expected_keys: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Cheap identity for direct production-cache export.

    The direct path packs already-rendered GPTQ/scale-sweep weights. Bind the
    export layer cache to the backing shard names, mtimes, lever metadata, and
    activation-scale summary so a stale export cache cannot be reused across
    production-cache changes.
    """
    import hashlib
    import json as _json

    weights = getattr(cache, "weights", {}) or {}
    cache_dir = getattr(cache, "cache_dir", None)
    rows = []
    for key in sorted(set(expected_keys)):
        value = weights.get(key)
        if isinstance(value, torch.Tensor):
            rows.append([key[0], key[1], "tensor",
                         list(value.shape), str(value.dtype)])
            continue
        if value is None:
            rows.append([key[0], key[1], "missing"])
            continue
        path = Path(str(value))
        if cache_dir and not path.is_absolute():
            path = Path(cache_dir) / path
        try:
            st = path.stat()
            rows.append([key[0], key[1], path.name, st.st_size, st.st_mtime_ns])
        except OSError:
            rows.append([key[0], key[1], path.name, "missing"])
    act = getattr(cache, "activation_max_abs", None) or {}
    act_digest = hashlib.sha256(
        _json.dumps(act, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    metadata = dict(getattr(cache, "metadata", {}) or {})
    metadata_digest = hashlib.sha256(
        _json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()[:16]
    digest = hashlib.sha256(
        _json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "path": str(Path(cache_dir).resolve()) if cache_dir else None,
        "n_entries": len(rows),
        "hash": digest,
        "activation_max_abs_hash": act_digest,
        "metadata_hash": metadata_digest,
        "levers": dict(getattr(cache, "levers", {}) or {}),
    }


def _production_cache_scales(cache) -> dict[str, float]:
    activation_max_abs = getattr(cache, "activation_max_abs", None) or {}
    scales = {
        name: (6.0 / float(max_abs))
        for name, max_abs in activation_max_abs.items()
        if max_abs and float(max_abs) > 0.0
    }
    return _unify_input_global_scales_across_fused_siblings(scales)



def _source_weight_shape_for_recipe(
    src_model: str,
    recipe_key: str,
    profile=None,
) -> list[int] | None:
    idx_path = Path(src_model) / "model.safetensors.index.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    candidates = [recipe_key + ".weight"]
    if profile is not None:
        source_name = profile.source_tensor_name(recipe_key)
        candidates.append(source_name + ".weight")
        candidates.append(profile.source_tensor_name(recipe_key + ".weight"))
    if recipe_key.startswith("model."):
        candidates.append(
            "model.language_model." + recipe_key[len("model."):] + ".weight"
        )
    for ckpt_key in dict.fromkeys(candidates):
        shard = weight_map.get(ckpt_key)
        if shard is None:
            continue
        from safetensors import safe_open
        with safe_open(str(Path(src_model) / shard), framework="pt") as sf:
            return list(sf.get_slice(ckpt_key).get_shape())
    return None


def _coerce_runtime_legal_assignment(
    src_model: str,
    assignment: dict[str, str],
    profile=None,
) -> tuple[dict[str, str], list[tuple[str, list[int], str]]]:
    """Adjust assignments that the target runtime cannot execute.

    Shape and format legality comes from serving-profile config. BF16 is the
    conservative runtime fallback when an assigned format is not executable.
    """
    out = dict(assignment)
    coerced: list[tuple[str, list[int], str]] = []
    target_profile = _allocator_target_profile_for_audit(profile) or "research"
    for qname, fmt in assignment.items():
        fmt_canonical = _canonical_export_format(fmt)
        out[qname] = fmt_canonical
        if fmt_canonical == "BF16":
            continue
        if fmt_canonical not in FORMAT_SCHEME:
            shape = _source_weight_shape_for_recipe(src_model, qname, profile)
            out[qname] = "BF16"
            coerced.append((qname, shape or [], fmt_canonical))
            continue
        shape = _source_weight_shape_for_recipe(src_model, qname, profile)
        if shape is None or len(shape) != 2:
            continue
        verdict = check_format_applicability(
            tuple(shape),
            fmt,
            qname=qname,
            target_profile=target_profile,
        )
        if not verdict.legal:
            out[qname] = "BF16"
            coerced.append((qname, shape, fmt_canonical))
    return out, coerced


def _allocator_target_profile_for_audit(profile) -> str | None:
    if profile is None:
        return None
    return resolve_target_profile(profile, None)


def _bf16_upgrade_audit(
    src_model: str,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    runtime_coerced: Sequence[tuple],
    profile,
) -> dict[str, object]:
    """Classify BF16 entries by immutability, runtime gates, or allocation.

    This is intentionally a manifest audit, not a policy change. It tells us
    which BF16 Linears are immutable/passthrough, which were forced by runtime
    format support, and which are real numerical/budget choices where enhanced
    MXFP8_E4M3/MXFP8_E5M2/FP8 may be worth trying next.
    """
    coerced: dict[str, tuple[list[int], str]] = {}
    for row in runtime_coerced:
        if len(row) >= 3:
            name, shape, from_fmt = row[:3]
        else:
            name, shape = row[:2]
            from_fmt = "MXFP8_E4M3"
        coerced[str(name)] = (shape, str(from_fmt))
    target_profile = _allocator_target_profile_for_audit(profile)
    candidate_formats = ("MXFP8_E4M3", "MXFP8_E5M2", "FP8_E4M3", "FP8_E5M2")
    entries: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    for qname, fmt in sorted(assignment.items()):
        if _canonical_export_format(fmt) != "BF16":
            continue
        coerced_entry = coerced.get(qname)
        shape = (
            coerced_entry[0]
            if coerced_entry is not None
            else _source_weight_shape_for_recipe(src_model, qname, profile)
        )
        shape_tuple = tuple(shape) if shape is not None else None
        if qname in bf16_passthrough:
            reason = "passthrough_or_immutable"
        elif coerced_entry is not None:
            reason = (
                "runtime_coerced_from_"
                + coerced_entry[1].lower().replace("-", "_")
            )
        else:
            verdicts: dict[str, dict[str, object]] = {}
            any_legal = False
            for cand_fmt in candidate_formats:
                if shape_tuple is None or len(shape_tuple) != 2:
                    verdicts[cand_fmt] = {
                        "legal": False,
                        "reason": "shape_unknown",
                        "detail": "",
                    }
                    continue
                verdict = check_format_applicability(
                    shape_tuple,
                    cand_fmt,
                    qname=qname,
                    target_profile=target_profile,
                )
                verdicts[cand_fmt] = {
                    "legal": bool(verdict.legal),
                    "reason": verdict.reason,
                    "detail": verdict.detail,
                }
                any_legal = any_legal or bool(verdict.legal)
            if not any_legal:
                reason = "no_vllm_supported_8bit_candidate"
            elif verdicts.get("MXFP8_E4M3", {}).get("legal"):
                reason = "allocator_selected_bf16_mxfp8_legal"
            else:
                reason = "allocator_selected_bf16_alternate_8bit_legal"
        counts[reason] = counts.get(reason, 0) + 1
        entry: dict[str, object] = {
            "name": qname,
            "reason": reason,
            "shape": list(shape) if shape is not None else None,
        }
        if reason.startswith("allocator_selected") or reason == "no_vllm_supported_8bit_candidate":
            verdicts = {}
            for cand_fmt in candidate_formats:
                if shape_tuple is None or len(shape_tuple) != 2:
                    verdicts[cand_fmt] = {"legal": False, "reason": "shape_unknown"}
                else:
                    verdict = check_format_applicability(
                        shape_tuple,
                        cand_fmt,
                        qname=qname,
                        target_profile=target_profile,
                    )
                    verdicts[cand_fmt] = {
                        "legal": bool(verdict.legal),
                        "reason": verdict.reason,
                        "detail": verdict.detail,
                    }
            entry["eight_bit_candidates"] = verdicts
        entries.append(entry)

    return {
        "counts": counts,
        "entries": entries,
        "target_profile": target_profile,
    }


def _production_cache_prefetch_assignment(
    assignment: dict[str, str],
    *,
    prefix: str | None = None,
) -> int:
    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None or not hasattr(cache, "prefetch"):
        return 0
    keys: list[tuple[str, str]] = []
    for qname, fmt in assignment.items():
        if prefix is not None and not (qname == prefix or qname.startswith(prefix + ".")):
            continue
        cache_fmt = str(fmt).upper()
        if _canonical_export_format(cache_fmt) == "BF16":
            continue
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is not None:
            keys.append(key)
    if not keys:
        return 0
    return int(cache.prefetch(keys, max_workers=_PRODUCTION_CACHE_PREFETCH_WORKERS))


def _pack_production_cached_2d(
    linear_name: str,
    fmt: str,
    *,
    nvfp4_global_real_override: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor] | None:
    """Pack a pre-rendered production weight for export.

    ProductionWeightCache stores dequantized weights after the production
    numerical passes. For export we only need to re-pack those weights into the
    native compressed-tensors layout; running GPTQ/scale-sweep again would
    measure a different artifact.
    """
    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    cache_fmt = str(fmt).upper()
    fmt = _canonical_export_format(cache_fmt)
    key = _production_cache_lookup_key(linear_name, cache_fmt)
    if key is None:
        return None
    w = cache.get(key[0], key[1])
    if w is None:
        return None
    target_device = device or torch.device("cpu")
    if fmt == "NVFP4":
        w_work = w.to(device=target_device, dtype=torch.float32)
        wp, ws, wg = quantize_dequantize_nvfp4(
            w_work,
            group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        input_scale = (
            _INPUT_GLOBAL_SCALES.get(linear_name) if _INPUT_GLOBAL_SCALES
            else None
        )
        if input_scale is None:
            input_scale = DEFAULT_INPUT_GLOBAL_SCALE
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg.reshape(1) if wg.dim() == 0 else wg,
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32, device=target_device,
            ),
        }
    if fmt in MXFP8_EXPLICIT_FORMATS:
        # MXFP8_E4M3/MXFP8_E5M2 activation-aware renders choose explicit
        # E8M0 group scales.
        # The production cache stores dequantized weights for KL/polish, not
        # the uint8 scale tensor. Repacking a dequantized tensor can legally
        # choose a different E8M0 scale, so when the activation cache is
        # available we recompute from source weights through `_quantize_2d`.
        cache_levers = getattr(cache, "levers", {}) or {}
        if (
            (
                bool(cache_levers.get("scale_sweep", False))
                or bool(cache_levers.get("gptq", False))
                or bool(cache_levers.get("joint_scale_opt", False))
            )
            and _CACHED_ACTIVATIONS is not None
        ):
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        dtype, max_value = _fp8_element_dtype_and_max(fmt)
        q, qs = quantize_dequantize_mxfp8(
            w_work,
            group_size=32,
            element_dtype=dtype,
            element_max=max_value,
        )
        return {"weight": q, "weight_scale": qs}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        cache_levers = getattr(cache, "levers", {}) or {}
        if (
            (
                bool(cache_levers.get("scale_sweep", False))
                or bool(cache_levers.get("gptq", False))
            )
            and _CACHED_ACTIVATIONS is not None
        ):
            return None
        if fmt == "FP8_E5M2":
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        q, qs = quantize_dequantize_fp8_dynamic(w_work)
        return {"weight": q, "weight_scale": qs}
    if fmt == "MXFP4":
        cache_levers = getattr(cache, "levers", {}) or {}
        if bool(cache_levers.get("gptq", False)) and _CACHED_ACTIVATIONS is not None:
            return None
        w_work = w.to(device=target_device, dtype=torch.float32)
        q, qs = quantize_dequantize_mxfp4(w_work, group_size=32)
        return {"weight_packed": q, "weight_scale": qs}
    if fmt == "BF16":
        return {"weight": w.to(device=target_device, dtype=torch.bfloat16)}
    return None

# Module-level flag bundle that controls which activation-aware
# passes run when `_quantize_2d` is invoked from main()'s streaming
# loop. Kept as module-level state (mirroring _INPUT_GLOBAL_SCALES)
# so we don't have to thread 3 boolean kwargs through every call
# site — unit tests pass the flags directly via kwargs.
_ACT_AWARE_FLAGS: dict[str, bool] = {
    "gptq": False,
    "scale_sweep": False,
    "static_act_order": False,
    "joint_scale_opt": False,
}
_NVFP4_SCALE_RULE: str | None = None
_PRODUCTION_WEIGHT_CACHE = None
_PRODUCTION_CACHE_FINGERPRINT: dict[str, object] | None = None
_PRODUCTION_CACHE_PREFETCH_WORKERS = 4


def _gptq_column_block_size(cols: int) -> int:
    raw = os.environ.get(
        "PRISMAQUANT_GPTQ_BLOCK_SIZE",
        os.environ.get("PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE", "128"),
    )
    try:
        value = int(raw)
    except Exception:
        value = 128
    return max(1, min(int(cols), int(value)))


def _gptq_columnwise_update(
    W: torch.Tensor,
    U: torch.Tensor,
    *,
    block_size: int,
    quantize_column: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Run the FP-Quant/GPTQ column update with fixed quantizer params.

    This matches FP-Quant's block loop: quantize one column, propagate the
    OBS error through the remaining columns in the current GPTQ block, then
    apply the accumulated block error to later blocks.
    """
    _rows, cols = W.shape
    block_size = max(1, min(int(block_size), int(cols)))
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        ncols = block_end - block_start
        block = W[:, block_start:block_end].clone()
        errs = torch.zeros_like(block)
        U_block = U[block_start:block_end, block_start:block_end]
        for i in range(ncols):
            col = block[:, i]
            col_idx = block_start + i
            col_dq = quantize_column(col, col_idx).to(
                device=W.device,
                dtype=W.dtype,
            )
            W[:, col_idx] = col_dq
            denom = U_block[i, i].clamp_min(1e-12)
            err = (col - col_dq) / denom
            block[:, i:].addr_(err, U_block[i, i:], alpha=-1)
            errs[:, i] = err
        if block_end < cols:
            W[:, block_end:].addmm_(
                errs,
                U[block_start:block_end, block_end:],
                alpha=-1,
            )
    return W


def _gptq_obs_rounding_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16, damp: float = 0.01,
    global_real_override: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
    joint_scale_opt: bool = False,
) -> torch.Tensor:
    """GPTQ one-shot OBS rounding for NVFP4 weights.

    Standard GPTQ (Frantar et al. 2022): build the activation covariance
    `H = X^T X + λ·diag(H)`, invert via Cholesky, then round columns with
    fixed per-group scales. Error from each column's quant is propagated via
    `H_inv`, which is the closed-form OBS update for least-squares loss
    `||W - W_q||_H^2`.

    Returns the dequantized, error-propagated weight `[out, in]`
    (float32). The caller still runs NVFP4 packing on this tensor to
    produce on-disk storage — the bits end up the same as if we had
    quantized `weight` directly but with a smaller output-space error.

    `damp = 0.01` adds `0.01·mean(diag(H))` to `diag(H)` for Cholesky
    stability. `global_real_override` threads through for fused-sibling
    consistency (same semantics as `quantize_dequantize_nvfp4`).

    `static_act_order` applies Lift/MR-GPTQ style activation ordering without
    requiring a runtime column permutation: scales are selected in the original
    NVFP4 group layout, columns are processed in descending activation
    importance, then the result is unpermuted before packing.

    `joint_scale_opt` jointly searches the NVFP4 tensor global and per-group
    max-to-codebook-level scale choices used by GPTQ. Its candidate set
    contains max-to-6 and max-to-4, so FourOverSix is a strict subset.
    """
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"GPTQ requires group_size={group_size} ∤ {cols}")

    # #42: per-token activation clipping to reduce Hessian condition
    # number. PRISMAQUANT_ACT_CLIP_QUANTILE in (0,1) clamps each token's
    # activations to ±|q-th percentile| of |x|. 0.999 removes ~4 extreme
    # outliers per 4k-dim row; condition number drops materially with
    # near-zero impact on bulk distribution. Set "0" or out-of-range to
    # disable. The same clipped matrix is used by local gates/sweeps so
    # those gates score the objective the candidate was optimized under.
    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    # H = X^T X; guard against near-zero diagonal (dead channels).
    H = X.t() @ X                                         # [in, in]
    diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
    H.diagonal().add_(damp * diag_mean)

    # Dead-channel handling (standard GPTQ trick): columns with zero
    # diagonal get set to identity-like so the Cholesky succeeds, and
    # we zero those weight columns.
    dead = torch.diagonal(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    # Target NVFP4 grid. Pre-compute the per-tensor global_real so the
    # per-block quantization uses the same outer scale as the final
    # on-disk packing (otherwise error propagation would be under an
    # inconsistent scale). This mirrors quantize_dequantize_nvfp4.
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12).float()
    else:
        grouped_full = W.reshape(rows, cols // group_size, group_size)
        scale_rule = (
            NVFP4_SCALE_RULE_JOINT_MSE
            if joint_scale_opt
            else None
        )
        s_g_real_full = _select_nvfp4_group_scales(
            grouped_full,
            scale_rule=scale_rule,
        )
        global_real = (s_g_real_full.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
        if joint_scale_opt:
            global_real = _optimize_nvfp4_joint_global_real(
                W,
                group_size=group_size,
                base_global_real=global_real,
            )

    scales_by_group = torch.empty(
        (rows, cols // group_size),
        dtype=torch.float32,
        device=W.device,
    )
    for group_idx, block_start in enumerate(range(0, cols, group_size)):
        block_end = block_start + group_size
        block = W[:, block_start:block_end]
        if joint_scale_opt:
            eff = _select_nvfp4_joint_gptq_eff_scale(
                block,
                global_real,
                col_importance=col_importance[block_start:block_end],
            )
        else:
            s_g_real = _select_nvfp4_group_scales(block)
            eff = _nvfp4_effective_scale_from_real(
                s_g_real,
                global_real,
                quantize_fp8=True,
            )
        scales_by_group[:, group_idx] = eff
    scale_by_col = scales_by_group.repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    # Compute Cholesky + inverse. We follow the GPTQ paper's trick of
    # computing an upper-triangular inverse (`torch.cholesky_inverse`
    # then Cholesky again) so the column-wise update becomes a simple
    # multiplication by an upper-triangular factor.
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        # Upper-triangular factor U such that U^T U = Hinv (GPTQ uses U
        # directly for the column updates).
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        # Fall back to RTN if the Cholesky numerically fails (rare:
        # extreme activation degeneracy).  Returning the original weight
        # here is not a valid NVFP4 render in compute_only/cache paths and
        # can make downstream local-MSE gates see an impossible zero error.
        return _rtn_dequant_nvfp4(
            weight,
            group_size=group_size,
            global_real_override=global_real_override,
        )

    def _quantize_nvfp4_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        eff_scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(1e-12)
        _idx, col_dq = _nvfp4_quantize_dequantize_with_eff_scale(
            col.unsqueeze(1),
            eff_scale,
        )
        return col_dq.squeeze(1)

    W = _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols),
        quantize_column=_quantize_nvfp4_col,
    )
    if static_act_order:
        assert inverse_perm is not None
        return W.index_select(1, inverse_perm).contiguous()
    return W


def _gptq_obs_rounding_nvfp4_swept(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
    joint_scale_opt: bool = False,
    linear_name: str | None = None,
) -> torch.Tensor:
    """Per-Linear GPTQ damping sweep.

    For each candidate damping value, run the standard
    `_gptq_obs_rounding_nvfp4` and measure the Hessian-weighted
    reconstruction error `tr((W − W_q)^T H (W − W_q))`. Return the
    rounded weight from the candidate with the smallest error.

    Cost: ~|candidates|× the unswept call (Cholesky+propagation
    repeats per candidate). Memory: `H` is recomputed each pass; we
    keep only the best `W_q` so far. For the typical 5-candidate
    sweep on a 4k×4k Linear, total wallclock ≈ 5× single-damp.

    Quality: typically 0.02–0.05 PPL gain on Llama-class models
    because the optimal damping varies by Linear (attention out-proj
    likes higher damp; MLP gate/up like lower).

    Caller convention matches `_gptq_obs_rounding_nvfp4`. When the
    Cholesky fallback fires (degenerate H), we return the best
    successful pass; if all fail, we return the unswept fallback.
    """
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        weight.shape[1],
        device=weight.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    H_full = X.t() @ X  # [in, in], shared evaluator

    # Optional research instrumentation (#46-followup): log per-Linear
    # H spectrum + per-damp errors so we can fit an analytical damp
    # picker. Env-gated; cost is one eigvalsh per Linear (~O(n^3)
    # where n=in_features; tens of ms on 4k×4k).
    log_path = os.environ.get("PRISMAQUANT_DAMP_SWEEP_LOG")
    if log_path:
        try:
            eigvals = torch.linalg.eigvalsh(H_full.to(torch.float64)).to(torch.float32)
            lambda_max = float(eigvals[-1].item())
            positive = eigvals[eigvals > 1e-30]
            lambda_min = float(positive.min().item()) if positive.numel() > 0 else 0.0
            mean_diag = float(torch.diagonal(H_full).mean().item())
        except Exception:
            lambda_max = float("nan")
            lambda_min = float("nan")
            mean_diag = float("nan")

    # Optional analytical damp picker: skip the 5-candidate sweep and
    # pick damp = c * lambda_max(H) / mean(diag(H)) directly.
    # Equivalent to a kappa-target with K=10 (kappa target reduces to
    # this form when lambda_min ~= 0, which is true on nearly every
    # production Linear). c=1.784e-5 fitted on Qwen3-4B's 450 logged
    # damp-sweep winners (log-MSE 0.172 = typical prediction within
    # ~2.4x of the parabolic-interpolated continuous optimum, well
    # inside the 5x gap between sweep candidates).
    # Cost: 1 GPTQ pass + ~10-iter power iteration for lambda_max,
    # vs 5 GPTQ passes for the sweep. Net ~5x speedup.
    if os.environ.get("PRISMAQUANT_DAMP_ANALYTICAL", "").lower() in {
        "1", "true", "yes", "on", "kappa_target",
    }:
        try:
            c = float(os.environ.get("PRISMAQUANT_DAMP_ANALYTICAL_C", "1.784e-5"))
            mean_diag_a = float(torch.diagonal(H_full).mean().item())
            if mean_diag_a > 0:
                # Power iteration for the dominant eigenvalue of H.
                n = H_full.shape[0]
                H_f = H_full.to(torch.float32)
                v = torch.randn(n, device=H_full.device, dtype=torch.float32)
                v = v / v.norm().clamp_min(1e-30)
                for _ in range(10):
                    v = H_f @ v
                    v = v / v.norm().clamp_min(1e-30)
                lambda_max_est = float((v @ (H_f @ v)).item())
                damp_pred = c * lambda_max_est / mean_diag_a
                damp_pred = min(max(damp_pred, 0.001), 0.1)
                w_q = _gptq_obs_rounding_nvfp4(
                    weight, activations, group_size=group_size,
                    damp=damp_pred, global_real_override=global_real_override,
                    clip_threshold=clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=static_act_order,
                    joint_scale_opt=joint_scale_opt,
                )
                diff = W_orig - w_q.to(torch.float32)
                err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
                if math.isfinite(err) and err > 0:
                    if log_path:
                        import json as _json
                        entry = {
                            "linear_name": linear_name,
                            "shape": list(weight.shape),
                            "lambda_max_est": lambda_max_est,
                            "mean_diag": mean_diag_a,
                            "analytical_damp": damp_pred,
                            "analytical_err": err,
                        }
                        try:
                            with open(log_path, "a") as f:
                                f.write(_json.dumps(entry) + "\n")
                        except Exception:
                            pass
                    return w_q
        except Exception:
            pass  # fall through to the 5-candidate sweep

    best_w = None
    best_err = float("inf")
    best_damp: float | None = None
    per_damp_err: dict[float, float] = {}
    for damp in damp_candidates:
        try:
            w_q = _gptq_obs_rounding_nvfp4(
                weight, activations, group_size=group_size,
                damp=damp, global_real_override=global_real_override,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                static_act_order=static_act_order,
                joint_scale_opt=joint_scale_opt,
            )
        except Exception:
            per_damp_err[damp] = float("inf")
            continue
        # Hessian-weighted reconstruction error (no damp injected here —
        # we want raw H for fair comparison across candidates).
        diff = W_orig - w_q.to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        per_damp_err[damp] = err
        if err < best_err:
            best_err = err
            best_w = w_q
            best_damp = damp
    if log_path:
        import hashlib
        import json as _json
        entry = {
            "linear_name": linear_name,
            "shape": list(weight.shape),
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
            "mean_diag": mean_diag,
            "best_damp": best_damp,
            "best_err": best_err if best_err != float("inf") else None,
            "per_damp_err": {f"{k:.4g}": v for k, v in per_damp_err.items()},
        }
        try:
            with open(log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass
    if best_w is None:
        return _rtn_dequant_nvfp4(
            W_orig,
            group_size=group_size,
            global_real_override=global_real_override,
        )
    return best_w


def _scale_sweep_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    grid: int = 32,
    span: tuple[float, float] = (0.5, 1.5),
    reference_weight: torch.Tensor | None = None,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-group joint (scale, rounding) closed-form polish.

    For each NVFP4 group, sweep `grid` candidate scales spanning
    `[span[0]·s0, span[1]·s0]`, where s0 is the default max-abs scale
    derived from the pre-pass weight. For each candidate scale, run RTN
    on the NVFP4 codebook and compute the activation-weighted MSE
    `sum_j a_j²·(w_orig,j - w_q,j)²` against the ORIGINAL (pre-pass)
    weight. Keep the configuration minimizing MSE per group, with an
    improve-or-keep gate against whatever `weight` is coming in.

    `reference_weight`: the pre-pass (float32) weight used to measure
    MSE. Defaults to `weight` (the post-pass state) when not supplied
    — in that case the gate degenerates to "improve over no-op" which
    is not useful. Callers who want the gate to work should pass the
    original weight explicitly.

    Closed-form analog of AutoRound's SGD on per-weight V offsets:
    AutoRound searches a continuous relaxation; we enumerate the
    discrete scale dimension directly. Per-weight rounding at each
    scale is RTN (optimal conditional on the scale).

    Output is a dequantized tensor on valid NVFP4 grid points under the
    new per-group scales. The downstream packer re-derives fp8_scale
    through the active NVFP4 scale rule; FourOverSix gives the packer a
    second legal max-to-4 representation when max-to-6 would lose a swept
    scale choice.
    """
    W_in = weight.to(torch.float32).contiguous()
    W_ref = (reference_weight if reference_weight is not None else W_in
             ).to(torch.float32).contiguous()
    if W_in.shape != W_ref.shape:
        raise ValueError(
            f"scale-sweep: weight shape {tuple(W_in.shape)} != "
            f"reference_weight shape {tuple(W_ref.shape)}")
    rows, cols = W_in.shape
    if cols % group_size != 0:
        raise ValueError(f"scale-sweep requires group_size={group_size} ∤ {cols}")

    if clip_threshold is not None and clip_threshold > 0.0:
        a = _activation_matrix_for_gptq(
            activations,
            cols,
            device=W_in.device,
            clip_threshold=clip_threshold,
            clip_rescale=clip_rescale,
            row_weights=fisher_row_weights,
        )
    else:
        a = _activation_matrix_for_gptq(
            activations,
            cols,
            device=W_in.device,
            clip_quantile=0.0,
            row_weights=fisher_row_weights,
        )
    col_importance = a.pow(2).mean(dim=0).clamp_min(1e-12)  # [in]

    # Use the REFERENCE weight to set the default per-group scale (s0)
    # and to measure MSE against.
    ref_grouped = W_ref.reshape(rows, cols // group_size, group_size)
    in_grouped = W_in.reshape(rows, cols // group_size, group_size)
    s_g_real = _select_nvfp4_group_scales(ref_grouped)
    if global_real_override is not None:
        global_real = global_real_override.to(W_in.device).clamp_min(1e-12).float()
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    eff_scale0 = _nvfp4_effective_scale_from_real(
        s_g_real,
        global_real,
        quantize_fp8=True,
    ).unsqueeze(-1)
    col_imp = col_importance.reshape(1, cols // group_size, group_size)  # [1, n_g, gs]

    # Incoming per-group MSE against reference.
    init_mse = (col_imp * (ref_grouped - in_grouped).pow(2)).sum(dim=-1)  # [rows, n_g]

    # Sweep scales. The full intermediate tensor
    # [rows, n_g, grid, gs, 15] would peak at >70 GB for a 12288-row
    # Linear × 192 groups × 32 scales × 16 weights × 15 codes × 4 B.
    # Chunk over rows so peak memory stays bounded regardless of size.
    mults = torch.linspace(span[0], span[1], grid,
                           device=W_in.device, dtype=torch.float32)  # [grid]

    # Target per-chunk intermediate budget: ~2 GB max on the biggest
    # tensor `d = [chunk, n_g, grid, gs, len(cb)]` (float32).
    n_g = cols // group_size
    bytes_per_row = n_g * grid * group_size * (2 * len(FLOAT_TO_E2M1) - 1) * 4
    chunk_target = max(1, (2 * 1024 * 1024 * 1024) // max(1, bytes_per_row))
    row_chunk = min(rows, int(chunk_target))

    result_groups = torch.empty_like(ref_grouped)
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        scales_c = eff_scale0[r0:r1].squeeze(-1).unsqueeze(-1) * mults  # [c, n_g, grid]
        ref_c = ref_grouped[r0:r1]
        in_c = in_grouped[r0:r1]
        init_mse_c = init_mse[r0:r1]

        gexp = ref_c.unsqueeze(2)                   # [c, n_g, 1, gs]
        sexp = scales_c.unsqueeze(3)                # [c, n_g, grid, 1]
        _idx, Wq_cand = _nvfp4_quantize_dequantize_with_eff_scale(
            gexp,
            sexp,
        )                                           # [c, n_g, grid, gs]
        err = col_imp.unsqueeze(2) * (gexp - Wq_cand).pow(2)  # [c, n_g, grid, gs]
        mse = err.sum(dim=-1)                        # [c, n_g, grid]
        del err
        best = mse.argmin(dim=-1)                    # [c, n_g]
        bidx = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, group_size)
        chosen_Wq = Wq_cand.gather(2, bidx).squeeze(2)           # [c, n_g, gs]
        chosen_mse = mse.gather(2, best.unsqueeze(-1)).squeeze(-1)  # [c, n_g]
        del Wq_cand, mse, best, bidx

        use_new = chosen_mse < init_mse_c
        result_groups[r0:r1] = torch.where(
            use_new.unsqueeze(-1).expand(-1, -1, group_size),
            chosen_Wq,
            in_c,
        )
    return result_groups.reshape(rows, cols)


def compute_nvfp4_global_real(weight: torch.Tensor, group_size: int = 16
                              ) -> torch.Tensor:
    """Return the per-tensor `global_real` that NVFP4 packing would
    pick for `weight` alone. Useful for fused-sibling pre-pass: caller
    takes the max across siblings and passes the joint value back into
    `quantize_dequantize_nvfp4(global_real_override=...)`."""
    rows, cols = weight.shape
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    s_g_real = _select_nvfp4_group_scales(grouped)
    return (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)


def quantize_dequantize_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NVFP4 RTN to a 2D `[rows, cols]` weight and return the
    on-disk triple `(weight_packed, weight_scale, weight_global_scale)`
    in the **compressed-tensors NVFP4 convention**:

      - per-group dequant scale  s_g_real from active NVFP4 scale rule
        (`static_6` maps max-abs(group) to ±6; FourOverSix also tests ±4)
      - per-tensor outer scale   global   = max(s_g_real) / FP8_E4M3_MAX
        (so the fp8-stored per-group scale stays inside [0, 448])
      - on-disk weight_scale (fp8) = s_g_real / global  ∈ [0, 448]
      - on-disk weight_global_scale = 1 / global  (DIVISOR)
        vLLM inverts on load: `layer.weight_global_scale = 1/loaded`
        → recovers `global` and applies it as the per-tensor multiplier
        in the NVFP4 GEMM.

    Dequant in the kernel: `weight ≈ codebook[index] · weight_scale_fp8 · global`

    `global_real_override` lets a caller force a particular per-tensor
    scale — used for fused siblings (q/k/v, gate/up) that vLLM expects
    to share one global_scale slot. Pass the max across the sibling
    group's natural global_real values.
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    grouped = weight.float().reshape(rows, n_groups, group_size)
    s_g_real = _select_nvfp4_group_scales(grouped)                       # the actual per-group scale
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12)
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)  # scalar
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=s_g_real,
    )
    fp4_idx = codec.indices.reshape(rows, cols)
    weight_packed = pack_fp4_indices(fp4_idx, cols)
    return (
        weight_packed,
        codec.scale,
        (1.0 / global_real).to(torch.float32).reshape(1),  # divisor convention
    )


def _rtn_dequant_nvfp4(
    weight: torch.Tensor, group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """RTN to NVFP4 grid, returning FP32 dequantized weights (no GPTQ
    error propagation, no scale sweep). Used by the do-no-harm gate
    (#do-no-harm) to compare against post-GPTQ/sweep state and revert
    if a Linear locally regressed."""
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    n_groups = cols // group_size
    W = weight.float()
    grouped = W.reshape(rows, n_groups, group_size)
    s_g_real = _select_nvfp4_group_scales(grouped)
    if global_real_override is not None:
        global_real = global_real_override.to(weight.device).clamp_min(1e-12).float()
    else:
        global_real = (s_g_real.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=s_g_real,
    )
    return codec.dequant.reshape(rows, cols)


def quantize_dequantize_nvfp4_packed(
    packed: torch.Tensor, group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert NVFP4 packing for a 3D `[E, M, N]` packed tensor.
    Each expert gets its own `global_real` (so the weight_global_scale
    output has shape `[E]`); the on-disk values are divisors (1/scale)
    matching the compressed-tensors convention.
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {N}")
    g = N // group_size
    grouped = packed.float().reshape(E, M, g, group_size)
    s_g_real = _select_nvfp4_group_scales(grouped)                          # [E, M, g]
    global_real = (s_g_real.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX).clamp_min(1e-12)  # [E]
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real.view(E, 1, 1),
        scale_real=s_g_real,
    )
    fp4_idx = codec.indices.reshape(E, M, N)
    weight_packed = pack_fp4_indices(fp4_idx, N)
    return (
        weight_packed,
        codec.scale,
        (1.0 / global_real).to(torch.float32),
    )


# ---------------------------------------------------------------------------
# MXFP8_E4M3/MXFP8_E5M2 packing (FP8 element format, E8M0 per-group scale).
# ---------------------------------------------------------------------------
MXFP8_E4M3_MAX = 448.0   # max representable in fp8_e4m3fn
MXFP4_E2M1_MAX = NVFP4_MAX


def _fp8_element_dtype_and_max(fmt: str) -> tuple[torch.dtype, float]:
    fmt_u = str(fmt).upper()
    if fmt_u.endswith("E5M2"):
        dtype = torch.float8_e5m2
    else:
        dtype = torch.float8_e4m3fn
    try:
        max_value = float(torch.finfo(dtype).max)
    except Exception:
        max_value = MXFP8_E4M3_MAX if dtype is torch.float8_e4m3fn else 57344.0
    return dtype, max_value


def _fp8_codec(
    values: torch.Tensor,
    *,
    scale: torch.Tensor,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> _FP8CodecResult:
    values_f = values.to(torch.float32)
    scale_f = scale.to(
        device=values_f.device,
        dtype=torch.float32,
    ).clamp_min(2.0 ** -127)
    quant = (values_f / scale_f).clamp(
        -float(element_max),
        float(element_max),
    ).to(element_dtype)
    dequant = quant.to(torch.float32) * scale_f
    return _FP8CodecResult(
        quant=quant,
        scale=scale_f,
        dequant=dequant,
    )


def _fp8_dynamic_codec(
    values: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> _FP8CodecResult:
    result = fp8_dynamic_weight_qdq(
        values,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _FP8CodecResult(
        quant=result.quant,
        scale=result.scale,
        dequant=result.dequant,
    )


def _fp8_dequantize(
    quant: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return quant.to(torch.float32) * scale.to(
        device=quant.device,
        dtype=torch.float32,
    )


def _mx_rounded_amax_power2(amax: torch.Tensor) -> torch.Tensor:
    """Match compressed-tensors' MX scale power-of-two rounding.

    compressed-tensors derives MXFP4/MXFP8 E8M0 scales by first rounding the
    block amax to a power of two with the FP4 mantissa-aware bit-mask rule,
    then subtracting the element-format exponent offset. Reusing that rule
    keeps PrismaQuant's stored scales byte-identical to the served consumer.
    """
    x = amax.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    raw = x.view(torch.int32).to(torch.int64)
    val_to_add = 1 << (23 - 1 - 1)
    sign_exponent_mask = ((1 << (8 + 1)) - 1) << 23
    rounded = torch.bitwise_and(
        raw + val_to_add,
        sign_exponent_mask,
    )
    return rounded.to(torch.int32).view(torch.float32)


def _mx_base_exponent_from_amax(
    amax: torch.Tensor,
    *,
    element_max: float,
) -> torch.Tensor:
    if math.isclose(float(element_max), MXFP4_E2M1_MAX, rel_tol=0.0, abs_tol=1e-6):
        return generate_mx_scales(amax, num_bits=4).to(torch.float32) - 127.0
    if math.isclose(float(element_max), MXFP8_E4M3_MAX, rel_tol=0.0, abs_tol=1e-6):
        return generate_mx_scales(amax, num_bits=8).to(torch.float32) - 127.0

    # compressed-tensors only exposes MX scale generation for FP4 E2M1 and
    # FP8 E4M3. Keep the local fallback for research-only FP8_E5M2.
    rounded = _mx_rounded_amax_power2(amax)
    element_offset = int(math.floor(math.log2(float(element_max))))
    exponent = torch.floor(torch.log2(rounded)) - float(element_offset)
    return exponent.clamp(-127, 127)


def _mxfp8_base_exponent(
    grouped: torch.Tensor,
    *,
    element_max: float,
) -> torch.Tensor:
    return _mx_base_exponent_from_amax(
        grouped.abs().amax(dim=-1),
        element_max=element_max,
    )


def _mxfp8_grouped_codec(
    grouped: torch.Tensor,
    *,
    e8m0_unbiased: torch.Tensor | None = None,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> _MXFP8CodecResult:
    if (
        e8m0_unbiased is None
        and element_dtype == torch.float8_e4m3fn
        and float(element_max) == float(MXFP8_E4M3_MAX)
    ):
        ungrouped = grouped.reshape(
            *grouped.shape[:-2],
            grouped.shape[-2] * grouped.shape[-1],
        )
        result = mxfp8_e4m3_qdq(ungrouped)
        return _MXFP8CodecResult(
            quant=result.quant.reshape_as(grouped),
            scale=result.scale,
            dequant=result.dequant.reshape_as(grouped),
        )

    grouped_f = grouped.to(torch.float32)
    if e8m0_unbiased is None:
        e8m0_unbiased = _mxfp8_base_exponent(
            grouped_f,
            element_max=element_max,
        )
    e = e8m0_unbiased.to(grouped_f.device, dtype=torch.float32).clamp(-127, 127)
    scale = torch.pow(
        torch.tensor(2.0, device=grouped_f.device, dtype=torch.float32),
        e,
    )
    fp8 = _fp8_codec(
        grouped_f,
        scale=scale.unsqueeze(-1),
        element_dtype=element_dtype,
        element_max=element_max,
    )
    e8m0_uint8 = (e + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return _MXFP8CodecResult(
        quant=fp8.quant,
        scale=e8m0_uint8,
        dequant=fp8.dequant,
    )


def _mxfp4_grouped_codec(grouped: torch.Tensor) -> _MXFP4CodecResult:
    """Return MXFP4 packed codes, E8M0 scale, and reconstructed values.

    ``grouped`` must have the 32-value MX block axis last. The same primitive
    is used by dense export, packed-expert export, and renderer-side checks so
    the E8M0 scale and FP4 codebook math cannot drift across call sites.
    """
    grouped_f = grouped.to(torch.float32)
    group_size = grouped_f.shape[-1]
    if group_size % 2 != 0:
        raise ValueError("MXFP4 packing requires an even group size")
    e8m0 = _mx_base_exponent_from_amax(
        grouped_f.abs().amax(dim=-1),
        element_max=MXFP4_E2M1_MAX,
    )
    scale = torch.pow(
        torch.tensor(2.0, device=grouped_f.device, dtype=torch.float32),
        e8m0,
    )
    indices, dequant = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped_f,
        scale.unsqueeze(-1),
    )
    packed = pack_fp4_indices(indices, group_size)
    e8m0_uint8 = (e8m0 + 127).to(torch.int32).clamp(0, 255).to(torch.uint8)
    return _MXFP4CodecResult(
        indices=indices,
        packed=packed,
        scale=e8m0_uint8,
        dequant=dequant,
    )


def _mxfp4_quantize_grouped(grouped: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP4 packed E2M1 values + E8M0 scale for grouped weights.

    `grouped` must have the per-group axis in the final dimension
    (size 32 for the OCP MXFP4 layout). The returned packed tensor has
    two FP4 codes per byte along that final axis, and the scale tensor is
    uint8 E8M0 with the final group axis removed.
    """
    codec = _mxfp4_grouped_codec(grouped)
    return codec.packed, codec.scale


def quantize_dequantize_mxfp4(weight: torch.Tensor, group_size: int = 32
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP4 RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp4-pack-quantized` format):
      - weight_packed: uint8, shape (rows, cols // 2)
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    packed, e8m0_uint8 = _mxfp4_quantize_grouped(grouped)
    return packed.reshape(rows, cols // 2), e8m0_uint8


def _mxfp4_dequantize_2d(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {cols}")
    lo = (weight_packed & 0xF).to(torch.long)
    hi = ((weight_packed >> 4) & 0xF).to(torch.long)
    fp4_idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    scale_by_col = e8m0_to_scale(
        scale,
        device=weight_packed.device,
    ).repeat_interleave(group_size, dim=1)
    return _decode_nvfp4_indices_with_eff_scale(fp4_idx, scale_by_col)


def quantize_dequantize_mxfp4_packed(packed: torch.Tensor, group_size: int = 32
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP4 RTN to a 3D packed-experts tensor `[E, M, N]`."""
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP4 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    qpacked, e8m0_uint8 = _mxfp4_quantize_grouped(grouped)
    return qpacked.reshape(E, M, N // 2), e8m0_uint8


def _mxfp8_quantize_grouped(
    grouped: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MXFP8_E4M3/MXFP8_E5M2 values + E8M0 scale for an arbitrary
    rank-N tensor whose LAST dim is the per-group axis (size group_size).

    Returns:
      - quant_fp8: same shape as `grouped`, dtype torch.float8_*
      - e8m0_uint8: same shape minus the last dim, uint8 (E8M0)

    Scale generation follows compressed-tensors' MX rule: round the block
    amax to the nearest representable power-of-two, subtract the element
    exponent offset, then store the biased E8M0 exponent. The resulting grid
    may place the block amax above the finite FP8 max; `_fp8_codec` clamps
    before casting so overflow cannot become NaN.
    """
    codec = _mxfp8_grouped_codec(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def quantize_dequantize_mxfp8(
    weight: torch.Tensor,
    group_size: int = 32,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8_E4M3/MXFP8_E5M2 RTN with E8M0 per-group scale to a 2D weight.

    On-disk schema (compressed-tensors `mxfp8-quantized` format):
      - weight_packed: torch.float8_*, same shape as weight
      - weight_scale:  uint8 E8M0, shape (rows, cols // group_size)
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(rows, cols // group_size, group_size)
    quant_fp8, e8m0_uint8 = _mxfp8_quantize_grouped(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return quant_fp8.reshape(rows, cols), e8m0_uint8


def _mxfp8_dequantize_grouped(
    quant_fp8: torch.Tensor,
    e8m0_uint8: torch.Tensor,
) -> torch.Tensor:
    scale = e8m0_to_scale(e8m0_uint8, device=quant_fp8.device)
    return quant_fp8.to(torch.float32) * scale.unsqueeze(-1)


def _mxfp8_dequantize_2d(
    quant_fp8: torch.Tensor,
    e8m0_uint8: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    rows, cols = quant_fp8.shape
    grouped = quant_fp8.reshape(rows, cols // group_size, group_size)
    return _mxfp8_dequantize_grouped(grouped, e8m0_uint8).reshape(rows, cols)


def _mxfp8_scale_sweep_quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted E8M0 scale search for MXFP8_E4M3.

    MXFP8_E4M3 has enough mantissa precision that GPTQ-style error propagation is
    usually not worth the cost. The scale, however, is exponent-only E8M0;
    max-abs/ceil is conservative and can waste resolution. Searching nearby
    exponents per row/group is cheap, vLLM-compatible, and preserves the exact
    compressed-tensors MXFP8_E4M3 representation.
    """
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3 scale sweep requires group_size={group_size} ∤ {cols}")
    W = weight.to(torch.float32)
    grouped = W.reshape(rows, cols // group_size, group_size)
    raw_shifts = os.environ.get("PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS", "0")
    try:
        shifts = [int(x.strip()) for x in raw_shifts.split(",") if x.strip()]
    except Exception:
        shifts = [0]
    if not shifts:
        shifts = [0]
    if shifts == [0]:
        codec = _mxfp8_grouped_codec(grouped)
        return (
            codec.quant.reshape(rows, cols),
            codec.scale,
            codec.dequant.reshape(rows, cols),
        )

    col_importance = _activation_col_importance_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    ).reshape(1, cols // group_size, group_size)

    base_e = _mxfp8_base_exponent(grouped, element_max=MXFP8_E4M3_MAX)
    shift_t = torch.tensor(shifts, device=W.device, dtype=torch.float32)

    best_err: torch.Tensor | None = None
    best_e: torch.Tensor | None = None
    for shift in shift_t:
        e = (base_e + shift).clamp(-127, 127)
        codec = _mxfp8_grouped_codec(
            grouped,
            e8m0_unbiased=e,
        )
        err = ((grouped - codec.dequant).pow(2) * col_importance).sum(dim=-1)
        if best_err is None:
            best_err = err
            best_e = e
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_e = torch.where(take, e, best_e)

    assert best_e is not None
    codec = _mxfp8_grouped_codec(
        grouped,
        e8m0_unbiased=best_e,
    )
    return (
        codec.quant.reshape(rows, cols),
        codec.scale,
        codec.dequant.reshape(rows, cols),
    )


def quantize_dequantize_mxfp8_packed(
    packed: torch.Tensor,
    group_size: int = 32,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MXFP8_E4M3/MXFP8_E5M2 RTN to a 3D packed-experts tensor `[E, M, N]`.

    Returns:
      - weight_packed: float8 `[E, M, N]`
      - weight_scale:  uint8 E8M0   `[E, M, N//group_size]`
    """
    E, M, N = packed.shape
    if N % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 group_size={group_size} ∤ {N}")
    grouped = packed.float().reshape(E, M, N // group_size, group_size)
    codec = _mxfp8_grouped_codec(
        grouped,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant.reshape(E, M, N), codec.scale


def quantize_dequantize_fp8_dynamic(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 W8A8 dynamic per-channel weight quantization.

    Matches vLLM's CompressedTensorsW8A8Fp8 expectation:
      - weight: torch.float8_*, shape `[out, in]`
      - weight_scale: torch.float32, shape `[out, 1]` (per-channel)

    Per-channel scale = max-abs(row) / fp8_max. Dynamic-token activation
    quantization is handled at runtime by vLLM (no on-disk activation
    scale needed).
    """
    codec = _fp8_dynamic_codec(
        weight,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def _dequantize_fp8_dynamic(
    quant: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _fp8_dequantize(quant, scale)


def _rtn_dequant_fp8_dynamic(
    weight: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    q, s = quantize_dequantize_fp8_dynamic(
        weight.to(torch.float32),
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _dequantize_fp8_dynamic(q, s)


def _rtn_dequant_mxfp8(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = MXFP8_E4M3_MAX,
) -> torch.Tensor:
    q, s = quantize_dequantize_mxfp8(
        weight.to(torch.float32),
        group_size=group_size,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return _mxfp8_dequantize_2d(q, s, group_size=group_size)


def _parse_int_set_env(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    vals: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            continue
    vals.append(0)
    return tuple(sorted(set(vals)))


def _mxfp8_joint_scale_shifts() -> tuple[int, ...]:
    # {-1, 0}: ceil-log2 (the canonical NVIDIA recipe, never saturates) plus
    # ceil-1 (lets the block trade one ULP of saturation on the amax for
    # one bit more precision on the rest). The synthetic-LLM oracle search
    # picks -1 for ~5% of blocks and 0 for the rest; deeper negative shifts
    # over-saturate and never win on unweighted MSE.
    return _parse_int_set_env(
        "PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS",
        "-1,0",
    )


def _fp8_quantize_dequantize_with_scale(
    values: torch.Tensor,
    scale: torch.Tensor,
    *,
    element_dtype: torch.dtype,
    element_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    codec = _fp8_codec(
        values,
        scale=scale,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.dequant


def _mxfp8_quantize_dequantize_block(
    block: torch.Tensor,
    *,
    col_importance: torch.Tensor | None,
    joint_scale_opt: bool,
    element_dtype: torch.dtype,
    element_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MXFP8_E4M3/MXFP8_E5M2 q/scale/dequant for one `[rows, group_size]` block.

    The JSO path searches legal E8M0 block scales by unweighted block MSE.
    Activation-weighted MSE was tried and produced rankings that disagreed
    with end-task quality; unweighted block MSE aligns with the allocator's
    h_trace * weight_mse cost surrogate. `col_importance` is accepted for
    interface compatibility but no longer consulted.
    """
    del col_importance  # unused after switch to unweighted block-MSE
    base_e = _mxfp8_base_exponent(block, element_max=element_max)
    if joint_scale_opt:
        best_err: torch.Tensor | None = None
        best_e: torch.Tensor | None = None
        for shift in _mxfp8_joint_scale_shifts():
            e = (base_e + float(shift)).clamp(-127, 127)
            codec = _mxfp8_grouped_codec(
                block,
                element_dtype=element_dtype,
                element_max=element_max,
                e8m0_unbiased=e,
            )
            score = (block - codec.dequant).pow(2).sum(dim=-1)
            if best_err is None:
                best_err = score
                best_e = e
                continue
            take = score < best_err
            best_err = torch.where(take, score, best_err)
            assert best_e is not None
            best_e = torch.where(take, e, best_e)
        assert best_e is not None
        e = best_e
    else:
        e = base_e
    codec = _mxfp8_grouped_codec(
        block,
        e8m0_unbiased=e,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale, codec.dequant


def _gptq_obs_rounding_mxfp4(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    damp: float = 0.01,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ one-shot OBS rounding for MXFP4 weights.

    MXFP4 uses the same E2M1 FP4 codebook as NVFP4, but with an E8M0
    per-32-value block scale and no tensor-global FP8 scale. The returned
    tuple is directly exportable: packed FP4 bytes, E8M0 block scales, and
    the served dequantized weight.
    """
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(f"MXFP4 GPTQ requires group_size={group_size} ∤ {cols}")

    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    H = X.t() @ X
    diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
    H.diagonal().add_(float(damp) * diag_mean)
    dead = torch.diagonal(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    scale_out = torch.empty(
        (rows, cols // group_size),
        device=W.device,
        dtype=torch.uint8,
    )
    for group_idx, block_start in enumerate(range(0, cols, group_size)):
        block_end = block_start + group_size
        codec = _mxfp4_grouped_codec(W[:, block_start:block_end])
        scale_out[:, group_idx] = codec.scale
    scale_by_col = e8m0_to_scale(
        scale_out,
        device=W.device,
    ).repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        q, scale = quantize_dequantize_mxfp4(
            weight.to(torch.float32),
            group_size=group_size,
        )
        return q, scale, _mxfp4_dequantize_2d(q, scale, group_size=group_size)

    idx_work = torch.empty((rows, cols), device=W.device, dtype=torch.uint8)

    def _quantize_mxfp4_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(1e-12)
        idx_col, col_dq = _nvfp4_quantize_dequantize_with_eff_scale(
            col.unsqueeze(1),
            scale,
        )
        idx_work[:, col_idx] = idx_col.squeeze(1)
        return col_dq.squeeze(1)

    _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols),
        quantize_column=_quantize_mxfp4_col,
    )

    idx_out = (
        idx_work.index_select(1, inverse_perm).contiguous()
        if inverse_perm is not None else
        idx_work
    )
    q_out = pack_fp4_indices(idx_out, cols)
    dequant = _mxfp4_dequantize_2d(q_out, scale_out, group_size=group_size)
    return q_out.contiguous(), scale_out.contiguous(), dequant.contiguous()


def _gptq_obs_rounding_mxfp4_swept(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    group_size: int = 32,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    static_act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        W_orig.shape[1],
        device=W_orig.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    H_full = X.t() @ X
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_err = float("inf")
    for damp in damp_candidates:
        try:
            candidate = _gptq_obs_rounding_mxfp4(
                W_orig,
                activations,
                group_size=group_size,
                damp=damp,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                static_act_order=static_act_order,
            )
        except Exception:
            continue
        diff = W_orig - candidate[2].to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        if err < best_err:
            best_err = err
            best = candidate
    if best is not None:
        return best
    return _gptq_obs_rounding_mxfp4(
        W_orig,
        activations,
        group_size=group_size,
        damp=0.01,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        fisher_row_weights=fisher_row_weights,
        static_act_order=static_act_order,
    )


def _gptq_obs_rounding_fp8_like(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    fmt: str,
    group_size: int = 32,
    damp: float = 0.01,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    joint_scale_opt: bool = False,
    static_act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ one-shot OBS rounding for FP8_E4M3/E5M2 and MXFP8_E4M3/E5M2 weights.

    Returns `(quant_weight, scale, dequant_weight)` in the exact representation
    the export path can serialize. Plain FP8 uses one fp32 scale per output
    row; MXFP8_E4M3/MXFP8_E5M2 use one uint8 E8M0 scale per row/group.
    """
    fmt_u = _canonical_export_format(fmt)
    is_mx = fmt_u in MXFP8_EXPLICIT_FORMATS
    is_plain = fmt_u in {"FP8_E4M3", "FP8_E5M2"}
    if not (is_mx or is_plain):
        raise ValueError(f"unsupported FP8 GPTQ format: {fmt}")
    if is_mx:
        # MXFP8 uses the canonical E8M0 block scale. The historical
        # joint_scale_opt hook searched nearby legal exponents, but that
        # is not part of the production MXFP8 recipe.
        joint_scale_opt = False
    else:
        # Plain FP8 has a per-output-row dynamic scale, so static activation
        # ordering is not part of the current production recipe.
        static_act_order = False

    element_dtype, element_max = _fp8_element_dtype_and_max(fmt_u)
    W = weight.to(torch.float32).clone()
    rows, cols = W.shape
    if is_mx and cols % group_size != 0:
        raise ValueError(f"MXFP8_E4M3/MXFP8_E5M2 GPTQ requires group_size={group_size} ∤ {cols}")

    X = _activation_matrix_for_gptq(
        activations,
        cols,
        device=W.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    H = X.t() @ X
    diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
    H.diagonal().add_(float(damp) * diag_mean)
    dead = torch.diagonal(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

    if is_plain:
        scale_out = _fp8_dynamic_codec(
            weight.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        ).scale
        scale_by_col = scale_out.expand(rows, cols)
    else:
        scale_out = torch.empty(
            (rows, cols // group_size),
            device=W.device,
            dtype=torch.uint8,
        )
        for group_idx, block_start in enumerate(range(0, cols, group_size)):
            block_end = block_start + group_size
            _q_block, scale_block, _block_dq = _mxfp8_quantize_dequantize_block(
                W[:, block_start:block_end],
                col_importance=col_importance[block_start:block_end],
                joint_scale_opt=joint_scale_opt,
                element_dtype=element_dtype,
                element_max=element_max,
            )
            scale_out[:, group_idx] = scale_block
        scale_by_col = e8m0_to_scale(
            scale_out,
            device=W.device,
        ).repeat_interleave(group_size, dim=1)

    inverse_perm: torch.Tensor | None = None
    if static_act_order:
        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(cols, device=W.device)
        W = W.index_select(1, perm).contiguous()
        H = H.index_select(0, perm).index_select(1, perm).contiguous()
        scale_by_col = scale_by_col.index_select(1, perm).contiguous()

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        if is_mx:
            q, scale = quantize_dequantize_mxfp8(
                weight.to(torch.float32),
                group_size=group_size,
                element_dtype=element_dtype,
                element_max=element_max,
            )
            return q, scale, _mxfp8_dequantize_2d(q, scale, group_size=group_size)
        q, scale = quantize_dequantize_fp8_dynamic(
            weight.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return q, scale, _dequantize_fp8_dynamic(q, scale)

    q_work = torch.empty((rows, cols), device=W.device, dtype=element_dtype)

    def _quantize_fp8_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        scale = scale_by_col[:, col_idx:col_idx + 1].clamp_min(2.0 ** -127)
        q_col, col_dq = _fp8_quantize_dequantize_with_scale(
            col.unsqueeze(1),
            scale,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        q_work[:, col_idx] = q_col.squeeze(1)
        return col_dq.squeeze(1)

    W = _gptq_columnwise_update(
        W,
        U,
        block_size=_gptq_column_block_size(cols),
        quantize_column=_quantize_fp8_col,
    )

    q_out = (
        q_work.index_select(1, inverse_perm).contiguous()
        if inverse_perm is not None else
        q_work
    )
    if is_mx:
        dequant = _mxfp8_dequantize_2d(q_out, scale_out, group_size=group_size)
    else:
        dequant = _dequantize_fp8_dynamic(q_out, scale_out)
    return q_out.contiguous(), scale_out.contiguous(), dequant.contiguous()


def _gptq_obs_rounding_fp8_like_swept(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    fmt: str,
    group_size: int = 32,
    damp_candidates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1),
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    joint_scale_opt: bool = False,
    static_act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W_orig = weight.to(torch.float32)
    X = _activation_matrix_for_gptq(
        activations,
        W_orig.shape[1],
        device=W_orig.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    )
    H_full = X.t() @ X
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    best_err = float("inf")
    for damp in damp_candidates:
        try:
            candidate = _gptq_obs_rounding_fp8_like(
                W_orig,
                activations,
                fmt=fmt,
                group_size=group_size,
                damp=damp,
                clip_threshold=clip_threshold,
                clip_rescale=clip_rescale,
                fisher_row_weights=fisher_row_weights,
                joint_scale_opt=joint_scale_opt,
                static_act_order=static_act_order,
            )
        except Exception:
            continue
        diff = W_orig - candidate[2].to(torch.float32)
        err = float(torch.einsum("oi,ij,oj->", diff, H_full, diff))
        if err < best_err:
            best_err = err
            best = candidate
    if best is not None:
        return best
    return _gptq_obs_rounding_fp8_like(
        W_orig,
        activations,
        fmt=fmt,
        group_size=group_size,
        damp=0.01,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        fisher_row_weights=fisher_row_weights,
        joint_scale_opt=joint_scale_opt,
        static_act_order=static_act_order,
    )


def _fp8_scale_sweep_factors() -> tuple[float, ...]:
    raw = os.environ.get(
        "PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS",
        "0.25,0.3535533906,0.5,0.7071067812,0.8408964153,"
        "1.0,1.189207115,1.414213562,2.0",
    )
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0.0:
            vals.append(value)
    vals.append(1.0)
    return tuple(sorted(set(vals)))


def _fp8_dynamic_scale_sweep_quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    clip_threshold: float | None = None,
    clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-weighted per-row scale search for vLLM FP8 E4M3."""
    if weight.dim() != 2:
        raise ValueError("FP8 scale sweep expects a 2D Linear weight")
    rows, cols = weight.shape
    w_f = weight.detach().to(torch.float32)
    if activations.shape[-1] != cols:
        codec = _fp8_dynamic_codec(w_f)
        return codec.quant, codec.scale, codec.dequant
    col_importance = _activation_col_importance_for_gptq(
        activations,
        cols,
        device=w_f.device,
        clip_threshold=clip_threshold,
        clip_rescale=clip_rescale,
        row_weights=fisher_row_weights,
    ).to(device=w_f.device, dtype=torch.float32)
    base = (
        w_f.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -127)
        / FP8_E4M3_MAX
    )
    best_score = torch.full((rows,), float("inf"), device=w_f.device)
    best_dequant = torch.empty_like(w_f)
    best_scale = torch.empty((rows, 1), device=w_f.device, dtype=torch.float32)
    for factor in _fp8_scale_sweep_factors():
        scale = base * float(factor)
        codec = _fp8_codec(
            w_f,
            scale=scale,
            element_dtype=torch.float8_e4m3fn,
            element_max=FP8_E4M3_MAX,
        )
        score = (
            (w_f - codec.dequant).pow(2) * col_importance.unsqueeze(0)
        ).sum(dim=1)
        take = score < best_score
        if bool(take.any().item()):
            best_score = torch.where(take, score, best_score)
            best_dequant = torch.where(
                take.unsqueeze(1),
                codec.dequant,
                best_dequant,
            )
            best_scale = torch.where(
                take.unsqueeze(1),
                codec.scale,
                best_scale,
            )
    best = _fp8_codec(
        best_dequant,
        scale=best_scale,
        element_dtype=torch.float8_e4m3fn,
        element_max=FP8_E4M3_MAX,
    )
    return (
        best.quant.contiguous(),
        best.scale.contiguous(),
        best.dequant.contiguous(),
    )


def quantize_dequantize_fp8_dynamic_packed(
    packed: torch.Tensor,
    *,
    element_dtype: torch.dtype = torch.float8_e4m3fn,
    element_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert FP8 W8A8 dynamic per-channel for `[E, M, N]` packed.

    Returns weight `[E, M, N]` fp8 and scale `[E, M, 1]` fp32.
    """
    codec = _fp8_dynamic_codec(
        packed,
        element_dtype=element_dtype,
        element_max=element_max,
    )
    return codec.quant, codec.scale


def _explicit_regex(name: str) -> str:
    """Anchor a Linear name as a compressed-tensors regex target."""
    return f"re:^{name.replace('.', '[.]')}$"


# Matches a vLLM-internal per-expert Linear qname, e.g.
#   model.layers.10.mlp.experts.0.gate_proj
# (Qwen3.5 / MiniMax / Gemma4 layouts all normalize to this form via the
# profile's `to_vllm_internal_name`.)
_PER_EXPERT_LINEAR_RE = re.compile(
    r"^(?P<prefix>.*[.])layers[.](?P<L>\d+)[.](?P<inner>.*mlp)[.]"
    r"experts[.](?P<E>\d+)[.](?P<proj>[^.]+)$"
)


def _build_target_list(vllm_names: list[str]) -> list[str]:
    """Emit compressed-tensors regex targets with per-expert Linears
    collapsed from 1-per-expert enumerations to one compact regex per
    (layer-prefix, projection) pair.

    Why: without collapsing, a 256-expert / 62-layer MoE produces ~47k
    explicit regex targets in config_groups. vLLM's
    `find_matched_target` does an O(n²) per-Linear walk through this
    list with Python's built-in `re.match` LRU cache (bounded to ~512
    distinct patterns), so the cache thrashes and scheme dispatch
    takes hours. Collapsing shrinks that to ~(layers × projs × active
    formats) regexes — typically a few hundred — and scheme dispatch
    completes in seconds.

    Names that aren't per-expert Linears pass through as explicit
    `re:^...$` regexes (same output as before).

    Within a (layer, proj) bucket, if every expert index 0..N-1 is
    present we emit a `[0-9]+` regex; sparse subsets get an enumerated
    alternation.
    """
    from collections import defaultdict

    bucketed: dict[tuple[str, int, str, str], set[int]] = defaultdict(set)
    passthrough: list[str] = []
    # Pre-formed regex targets (e.g. the packed-MoE per-expert regex
    # build_quantization_config emits) must pass through verbatim.
    # Double-wrapping them via _explicit_regex would produce an
    # unmatchable `re:^re:^...$$`.
    preformed_regex: list[str] = []
    for n in vllm_names:
        if n.startswith("re:"):
            preformed_regex.append(n)
            continue
        m = _PER_EXPERT_LINEAR_RE.match(n)
        if not m:
            passthrough.append(n)
            continue
        prefix = m.group("prefix")
        L = int(m.group("L"))
        inner = m.group("inner")
        proj = m.group("proj")
        E = int(m.group("E"))
        bucketed[(prefix, L, inner, proj)].add(E)

    collapsed: list[str] = []
    for (prefix, L, inner, proj), _experts in sorted(bucketed.items()):
        prefix_r = prefix.replace(".", "[.]")
        inner_r = inner.replace(".", "[.]")
        # Always emit the [0-9]+ wildcard for the expert position. vLLM's
        # FusedMoE.get_moe_method probes the synthetic name `experts.0.X_proj`
        # against this regex, and every expert in a layer shares the same
        # scheme, so wildcarding is semantically correct.
        expr = "[0-9]+"
        collapsed.append(
            f"re:^{prefix_r}layers[.]{L}[.]{inner_r}[.]experts[.]{expr}"
            f"[.]{proj}$"
        )

    out = (
        [_explicit_regex(n) for n in sorted(passthrough)]
        + sorted(preformed_regex)
        + sorted(collapsed)
    )
    return out


# ---------------------------------------------------------------------------
# Module / parameter discovery — mirrors what install_packed_expert_hooks
# detects, so the export sees the same units as the probe.
# ---------------------------------------------------------------------------
def _packed_expert_param_name_set(profile=None) -> set[str]:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return set(profile.packed_expert_param_names())
        except Exception:
            pass
    return set()


def _is_packed_experts_module(module: nn.Module, profile=None) -> bool:
    names = _packed_expert_param_name_set(profile)
    cls_name = type(module).__name__.lower()
    if "expert" not in cls_name:
        return False
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in names):
            return True
    return False


def _packed_experts_param_names(module: nn.Module, profile=None) -> list[str]:
    names = _packed_expert_param_name_set(profile)
    return sorted(
        n for n, p in module.named_parameters(recurse=False)
        if (isinstance(p, nn.Parameter)
            and p.dim() == 3
            and n in names)
    )


def _packed_expert_projection_names(profile, param_name: str) -> tuple[str, ...]:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            projections = tuple(profile.packed_expert_projection_names(param_name))
            if projections:
                return projections
        except Exception:
            pass
    return (str(param_name),)


def _packed_expert_parent_for_projection(profile, projection_name: str) -> str | None:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return profile.packed_expert_parent_for_projection(projection_name)
        except Exception:
            pass
    return None


def _vllm_moe_scheme_projection_names(profile, param_name: str) -> tuple[str, ...]:
    """vLLM FusedMoE scheme-probe / ignore projection names for a packed
    expert param — the canonical ``gate_proj``/``up_proj``/``down_proj``
    that vLLM's ``get_moe_method`` and ignore matching dispatch on,
    regardless of the on-disk weight names. Used ONLY for config_groups
    targets + ignore regexes; weight export still uses the on-disk names.
    See ModelProfile.vllm_fused_moe_scheme_projection_names."""
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        getter = getattr(profile, "vllm_fused_moe_scheme_projection_names", None)
        if callable(getter):
            try:
                projections = tuple(getter(param_name))
                if projections:
                    return projections
            except Exception:
                pass
    return _packed_expert_projection_names(profile, param_name)


def _all_packed_expert_projection_names(profile) -> tuple[str, ...]:
    projections: list[str] = []
    seen: set[str] = set()
    for param_name in sorted(_packed_expert_param_name_set(profile)):
        for projection in _packed_expert_projection_names(profile, param_name):
            if projection in seen:
                continue
            projections.append(projection)
            seen.add(projection)
    return tuple(projections)


def _split_packed_expert_tensor(
    packed_param: torch.Tensor,
    param_name: str,
    profile,
) -> list[tuple[str, torch.Tensor]]:
    projections = _packed_expert_projection_names(profile, param_name)
    if projections == (param_name,):
        return [(param_name, packed_param)]
    rows = int(packed_param.shape[1])
    n_parts = len(projections)
    if rows % n_parts != 0:
        raise ValueError(
            f"packed expert tensor {param_name!r} with rows={rows} cannot "
            f"split evenly into configured projections {projections!r}"
        )
    chunk = rows // n_parts
    return [
        (proj_name, packed_param[:, i * chunk:(i + 1) * chunk, :])
        for i, proj_name in enumerate(projections)
    ]


# ---------------------------------------------------------------------------
# Fused-sibling joint global_scale (for dense Linears)
# ---------------------------------------------------------------------------
# vLLM's compressed_tensors_w4a4_nvfp4.process_weights_after_loading warns
# (and reduces accuracy) when q/k/v or gate/up have different
# weight_global_scale. We compute the max over each fused group's natural
# global_scale and force every sibling to use it.
#
# Legacy fallback for callers that do not pass a ModelProfile. New model
# families should declare fused groups in their profile structure spec.
_FUSED_DENSE_PATTERNS = [
    (re.compile(r"^(?P<pre>.+)\.self_attn\.(?P<sib>q_proj|k_proj|v_proj)$"),
     ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.mlp\.shared_expert\.(?P<sib>gate_proj|up_proj)$"),
     ("gate_proj", "up_proj")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_qkv|in_proj_z)$"),
     ("in_proj_qkv", "in_proj_z")),
    (re.compile(r"^(?P<pre>.+)\.linear_attn\.(?P<sib>in_proj_a|in_proj_b)$"),
     ("in_proj_a", "in_proj_b")),
]


def _fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return (group_key, sibling_member_names) if `name` is part of a
    known fused dense Linear group; else None. group_key is the parent
    prefix used to bucket siblings together."""
    for pat, members in _FUSED_DENSE_PATTERNS:
        m = pat.match(name)
        if m:
            return (m.group("pre"), members)
    return None


def _fused_group_key_for_name(name: str, profile=None) -> str | None:
    group_fn = getattr(profile, "fused_sibling_group", None)
    if callable(group_fn):
        try:
            group = group_fn(name)
        except Exception:
            group = None
        if group:
            return str(group)
    mapping_fn = getattr(profile, "fused_sibling_leaf_mapping", None)
    if callable(mapping_fn) and "." in name:
        try:
            mapping = mapping_fn()
        except Exception:
            mapping = None
        if mapping:
            prefix, leaf = name.rsplit(".", 1)
            for fused, members in mapping.items():
                if leaf in set(str(member) for member in members):
                    return f"{prefix}.{fused}"
    fallback = _fused_dense_group(name)
    if fallback is None:
        return None
    prefix, members = fallback
    return f"{prefix}::__fused__:{','.join(members)}"


def _unify_input_global_scales_across_fused_siblings(
    scales: dict[str, float],
    *,
    profile=None,
) -> dict[str, float]:
    """Post-process per-Linear input_global_scale values so fused-
    sibling groups share one scale.

    vLLM concatenates q/k/v (and gate/up) into a single fused Linear
    at load time and applies ONE input_global_scale to the forward
    pass. If the siblings' scales don't match, vLLM warns and reduces
    accuracy.

    Siblings receive the same upstream activation, so their
    `compute_nvfp4_input_global_scale` outputs are theoretically
    identical — but capture + subsampling order introduces float-
    precision drift in practice.  The stored values are reciprocals
    (s = 6 / max_abs); the conservative join is therefore ``min(vals)``
    (smallest reciprocal == largest max_abs == loosest clipping), so
    the fused Linear never truncates any sibling's activations.
    Siblings that weren't NVFP4-assigned pass through unchanged.
    """
    # Bucket siblings by fused group.
    groups: dict[str, list[str]] = {}
    for name in scales:
        g = _fused_group_key_for_name(name, profile)
        if g is None:
            continue
        groups.setdefault(g, []).append(name)

    out = dict(scales)
    n_unified = 0
    max_drift = 0.0
    for key, members in groups.items():
        members = [m for m in members if m in scales]
        if len(members) < 2:
            continue
        vals = [scales[m] for m in members]
        # input_global_scale stores 6 / max_abs (reciprocal convention,
        # see compute_nvfp4_input_global_scale).  To pick a JOINT scale
        # that doesn't over-clip ANY sibling's activations we want the
        # smallest reciprocal == largest max_abs == loosest clipping.
        # Previously this used max(vals), which under the reciprocal
        # convention yields the TIGHTEST clipping — over-clipping the
        # sibling with the largest activation range.  In practice fused
        # siblings have similar activation distributions, so the drift
        # is small, but min() is the correct conservative join.
        joint = min(vals)
        drift = max(abs(joint - v) for v in vals)
        max_drift = max(max_drift, drift)
        for m in members:
            out[m] = joint
        n_unified += 1
    if n_unified:
        print(f"[export-stream] unified input_global_scale across "
              f"{n_unified} fused-sibling groups "
              f"(max pre-unify drift: {max_drift:.3e})", flush=True)
    return out


def _compute_nvfp4_joint_global(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    profile=None,
) -> dict[str, torch.Tensor]:
    """Pre-pass over the model: for each fused-sibling group whose
    members are all assigned to NVFP4, compute the joint global_real
    (max across siblings). Return a dict mapping each sibling's qname
    to the shared global_real tensor."""
    # Bucket siblings by (parent_prefix, kind). Missing siblings are
    # OK — vLLM's loader handles partial fusion fine.
    groups: dict[str, list[tuple[str, nn.Linear]]] = {}
    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        live_to_recipe = getattr(profile, "live_to_recipe_name", None)
        if callable(live_to_recipe):
            try:
                recipe_qname = live_to_recipe(qname)
            except Exception:
                recipe_qname = qname
        else:
            recipe_qname = qname
        if _canonical_export_format(assignment.get(recipe_qname, "BF16")) != "NVFP4":
            continue
        g = _fused_group_key_for_name(recipe_qname, profile)
        if g is None:
            continue
        groups.setdefault(g, []).append((recipe_qname, mod))

    out: dict[str, torch.Tensor] = {}
    for _group_key, siblings in groups.items():
        # Need every sibling to also be NVFP4 — otherwise vLLM allocates
        # the fused tensor under a different scheme and our joint scale
        # wouldn't apply consistently. The allocator's promote_fused
        # already enforces this; here we just verify and skip on partial
        # consistency (defensive — a mixed-format fused group is a bug
        # upstream of the export and would fail the load anyway).
        candidates = [
            compute_nvfp4_global_real(mod.weight.detach().float())
            for _, mod in siblings
        ]
        joint = torch.stack(candidates).max()
        for qname, _ in siblings:
            out[qname] = joint
    return out


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------
def _quantize_2d(
    weight: torch.Tensor, fmt: str,
    nvfp4_global_real_override: torch.Tensor | None = None,
    input_global_scale_override: float | None = None,
    act_clip_threshold: float | None = None,
    act_clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    linear_name: str | None = None,
    gptq_enabled: bool = False,
    scale_sweep_enabled: bool = False,
    static_act_order_enabled: bool = False,
    joint_scale_opt_enabled: bool = False,
    cached_activations: torch.Tensor | None = None,
    compute_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Compress a 2D Linear weight under format `fmt`.

    Returns the dict of on-disk tensors keyed by the suffix
    (`weight_packed`, `weight_scale`, `weight_global_scale`, ...).

    `nvfp4_global_real_override`: when this Linear is one shard of a
    fused parameter (q/k/v/o, gate/up), pass the joint per-tensor
    scale shared across all siblings. vLLM warns when sibling scales
    differ and reports degraded accuracy; sharing avoids both.

    `input_global_scale_override`: per-Linear activation scale computed
    from calibration — `max_abs(cached_activations) / 6.0` so scaled
    activations fit in FP4 E2M1's ±6 range before per-group quant. If
    None, falls back to `DEFAULT_INPUT_GLOBAL_SCALE` (1.0). Calibrated
    values typically improve PPL noticeably on NVFP4 weights because
    otherwise vLLM's runtime activation quant uses an undersized
    dynamic range.

    `gptq_enabled` and `scale_sweep_enabled` compose activation-aware passes
    on NVFP4, MXFP4, FP8_E4M3/FP8_E5M2, and MXFP8_E4M3/MXFP8_E5M2 paths. Each
    requires `cached_activations` (looked up from _CACHED_ACTIVATIONS by
    `linear_name` when not supplied explicitly). For MXFP8_E4M3/MXFP8_E5M2,
    `joint_scale_opt_enabled` searches legal E8M0 block scales during GPTQ
    instead of reusing NVFP4's max-to-4/max-to-6 heuristic.

    `cached_activations`: optional `[N, in_features]` float tensor of
    probe-captured inputs for this Linear. If None and `linear_name`
    is set, `_CACHED_ACTIVATIONS[linear_name]` is used.

    `act_clip_threshold`: optional scalar clamp for the render-time
    activation-aware NVFP4/MXFP4/MXFP8_E4M3/MXFP8_E5M2 passes.  When None, legacy behavior is
    preserved: GPTQ/do-no-harm honor PRISMAQUANT_ACT_CLIP_QUANTILE,
    while scale_sweep uses raw cached activations.

    `fisher_row_weights`: optional per-token gradient² weights aligned to
    cached activation rows. When provided, GPTQ/scale-sweep local objectives
    become output/Fisher-weighted by scaling activation rows by sqrt(weight).

    `fmt = MXFP8_E4M3` and `fmt = MXFP8_E5M2` emit fp8 weights plus E8M0
    uint8 per-group scales (group_size=32). A bare `MXFP8` input is accepted
    only as a legacy alias for `MXFP8_E4M3`.
    """
    fmt = _canonical_export_format(fmt)

    # Resolve activations from the module-level cache when not passed.
    acts = cached_activations
    if (acts is None and linear_name is not None
            and _CACHED_ACTIVATIONS is not None):
        acts = _CACHED_ACTIVATIONS.get(linear_name)

    # Device fix: cached activations are stored on CPU (float32) to
    # amortize load cost across many quant calls; weights land on the
    # export device (typically CUDA). Move activations to the weight's
    # device here so every downstream op (GPTQ H matrix,
    # act-weighted rounding) runs on a consistent device. Repairs
    # `Expected all tensors to be on the same device, but found at
    # least two devices, cuda:0 and cpu!` in live Qwen3.6-35B export.
    if acts is not None and acts.device != weight.device:
        acts = acts.to(weight.device, non_blocking=True)

    # Resolve act-aware flags from the module-level config when none
    # were explicitly enabled via kwargs — lets main() turn them on
    # once without threading through every call site. Kwargs still
    # win when any is set True (unit tests pass them explicitly).
    if not (
        gptq_enabled
        or scale_sweep_enabled
        or static_act_order_enabled
        or joint_scale_opt_enabled
    ):
        gptq_enabled = bool(_ACT_AWARE_FLAGS.get("gptq"))
        scale_sweep_enabled = bool(_ACT_AWARE_FLAGS.get("scale_sweep"))
        static_act_order_enabled = bool(_ACT_AWARE_FLAGS.get("static_act_order"))
        joint_scale_opt_enabled = bool(_ACT_AWARE_FLAGS.get("joint_scale_opt"))
    static_act_order_enabled = bool(gptq_enabled and static_act_order_enabled)
    joint_scale_opt_enabled = bool(gptq_enabled and joint_scale_opt_enabled)

    if fmt == "NVFP4":
        w_work = weight.to(torch.float32)

        def _acts_for_error_passes() -> torch.Tensor | None:
            """Return cached activations aligned to this Linear."""
            if acts is None or acts.shape[-1] != w_work.shape[1]:
                return None
            return acts

        # Step 2: GPTQ one-shot OBS rounding (block-wise error prop).
        # Produces an already-dequantized tensor living on the NVFP4
        # grid; subsequent packing is lossless wrt this tensor.
        if gptq_enabled:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                # Env-gated per-Linear damping sweep (#46). When set,
                # try multiple λ values for the Hessian regularizer and
                # pick the one with smallest output-space error. ~5×
                # GPTQ wallclock; ~0.02–0.05 PPL gain on Llama-class.
                # Default ON (validated on Qwen3-0.6B audit: −0.19 PPL
                # vs single-damp). PRISMAQUANT_GPTQ_DAMP_SWEEP=0 disables.
                if os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0":
                    w_work = _gptq_obs_rounding_nvfp4_swept(
                        w_work, acts_work, group_size=16,
                        global_real_override=nvfp4_global_real_override,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=static_act_order_enabled,
                        joint_scale_opt=joint_scale_opt_enabled,
                        linear_name=linear_name,
                    )
                else:
                    w_work = _gptq_obs_rounding_nvfp4(
                        w_work, acts_work, group_size=16,
                        global_real_override=nvfp4_global_real_override,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=static_act_order_enabled,
                        joint_scale_opt=joint_scale_opt_enabled,
                    )

        # Step 3: closed-form per-group scale sweep. Joint (scale,
        # rounding-set) search on the NVFP4 codebook, activation-
        # weighted MSE against the ORIGINAL pre-pass weight, with an
        # improve-or-keep gate against the current w_work. Recovers
        # most of AutoRound's benefit without its 200-iter SGD.
        if scale_sweep_enabled:
            acts_work = _acts_for_error_passes()
            if acts_work is not None:
                w_work = _scale_sweep_nvfp4(
                    w_work, acts_work, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                    reference_weight=weight.to(torch.float32),
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )

        # Do-no-harm gate (codex review #3): if GPTQ ran and we have
        # cached activations, compute the activation-weighted
        # reconstruction MSE for both the post-pass weight (`w_work`)
        # and a pure-RTN baseline against the original. If RTN is
        # better, revert. Catches per-Linear cases where GPTQ + sweep
        # locally degraded reconstruction. Env-gated; default on
        # because the cost is one RTN dequant + two MSE sums (cheap).
        if (gptq_enabled and acts is not None
                and os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0"):
            try:
                w_orig_f = weight.to(torch.float32)
                w_rtn = _rtn_dequant_nvfp4(
                    w_orig_f, group_size=16,
                    global_real_override=nvfp4_global_real_override,
                )
                a2 = _activation_col_importance_for_gptq(
                    acts,
                    w_orig_f.shape[1],
                    device=w_orig_f.device,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                )
                mse_rtn = float((a2 * (w_orig_f - w_rtn).pow(2)
                                 .sum(dim=0)).sum())
                mse_work = float((a2 * (w_orig_f - w_work).pow(2)
                                  .sum(dim=0)).sum())
                if mse_rtn < mse_work:
                    if os.environ.get(
                        "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                        print(f"[do-no-harm] {linear_name}: "
                              f"reverted to RTN "
                              f"(mse {mse_work:.3e} → {mse_rtn:.3e})",
                              flush=True)
                    w_work = w_rtn
            except Exception as _e:
                pass  # never fail the export over the gate

        # Step 4: final NVFP4 pack. `w_work` is the post-GPTQ,
        # post-act-round, post-scale-sweep weight.
        input_scale = input_global_scale_override
        if input_scale is None and linear_name is not None and _INPUT_GLOBAL_SCALES:
            input_scale = _INPUT_GLOBAL_SCALES.get(linear_name)
        if input_scale is None:
            input_scale = DEFAULT_INPUT_GLOBAL_SCALE

        # compute_only path (#12): defer final pack so block-output
        # match can refine the dequantized weight before it's frozen
        # into FP4 codes. Caller invokes _finalize_compute_only() to
        # produce the final packed dict.
        if compute_only:
            return {
                "_compute_only": True,
                "_fmt": "NVFP4",
                "_w_dq": w_work,
                "_nvfp4_global_real": nvfp4_global_real_override,
                "_input_scale": float(input_scale),
            }

        wp, ws, wg = quantize_dequantize_nvfp4(
            w_work, group_size=16,
            global_real_override=nvfp4_global_real_override,
        )
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
            # Required by vLLM's CompressedTensorsW4A4Nvfp4 process; see
            # compressed_tensors_w4a4_nvfp4.py:115. Without it vLLM
            # initializes input_global_scale to zeros and computes
            # 1/zero on activation quant → degenerate output.
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32,
            ),
        }
    if fmt in MXFP8_EXPLICIT_FORMATS:
        # MXFP8_E4M3/MXFP8_E5M2 GPTQ is export-faithful: it returns the FP8
        # codes and E8M0 scale tensor directly. MXFP8 deliberately does not
        # consume joint_scale_opt; it uses the canonical E8M0 scale rule.
        w_work = weight.to(torch.float32)
        acts_work = acts
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        has_acts = (
            acts_work is not None and acts_work.shape[-1] == w_work.shape[1]
        )

        def _mxfp8_rtn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_rtn, s_rtn = quantize_dequantize_mxfp8(
                w_work,
                group_size=32,
                element_dtype=element_dtype,
                element_max=element_max,
            )
            return q_rtn, s_rtn, _mxfp8_dequantize_2d(q_rtn, s_rtn, group_size=32)

        if gptq_enabled and has_acts:
            assert acts_work is not None

            def _mxfp8_gptq_candidate(
                use_static_act_order: bool,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0":
                    return _gptq_obs_rounding_fp8_like_swept(
                        w_work,
                        acts_work,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=False,
                        static_act_order=use_static_act_order,
                    )
                return _gptq_obs_rounding_fp8_like(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                    static_act_order=use_static_act_order,
                )

            candidates = [_mxfp8_gptq_candidate(False)]
            if static_act_order_enabled:
                candidates.append(_mxfp8_gptq_candidate(True))
            w, ws, dq = min(
                candidates,
                key=lambda cand: _activation_weighted_weight_error(
                    w_work,
                    cand[2],
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                ),
            )
            if scale_sweep_enabled and fmt == "MXFP8_E4M3":
                w, ws, dq = _mxfp8_scale_sweep_quantize(
                    dq,
                    acts_work,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            if os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0":
                try:
                    q_rtn, s_rtn, dq_rtn = _mxfp8_rtn()
                    err_rtn = _activation_weighted_weight_error(
                        w_work,
                        dq_rtn,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                    )
                    err_work = _activation_weighted_weight_error(
                        w_work,
                        dq,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                    )
                    if err_rtn < err_work:
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP8 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        w, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception:
                    pass
        elif scale_sweep_enabled and has_acts and fmt == "MXFP8_E4M3":
            assert acts_work is not None
            w, ws, _ = _mxfp8_scale_sweep_quantize(
                w_work,
                acts_work,
                group_size=32,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
            )
        else:
            w, ws, _ = _mxfp8_rtn()
        return {"weight": w, "weight_scale": ws}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        w_work = weight.to(torch.float32)
        acts_work = acts
        if (gptq_enabled and acts_work is not None
                and acts_work.shape[-1] == w_work.shape[1]):
            if os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0":
                w, ws, dq = _gptq_obs_rounding_fp8_like_swept(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                )
            else:
                w, ws, dq = _gptq_obs_rounding_fp8_like(
                    w_work,
                    acts_work,
                    fmt=fmt,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    joint_scale_opt=False,
                )
            if scale_sweep_enabled and fmt == "FP8_E4M3":
                w, ws, _ = _fp8_dynamic_scale_sweep_quantize(
                    dq,
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
        elif (scale_sweep_enabled and acts_work is not None
              and acts_work.shape[-1] == w_work.shape[1]
              and fmt == "FP8_E4M3"):
            w, ws, _ = _fp8_dynamic_scale_sweep_quantize(
                w_work,
                acts_work,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
            )
        else:
            if fmt == "FP8_E5M2":
                raise ValueError("FP8_E5M2 export packing is research-only")
            w, ws = quantize_dequantize_fp8_dynamic(w_work)
        return {"weight": w, "weight_scale": ws}
    if fmt == "MXFP4":
        w_work = weight.to(torch.float32)
        acts_work = acts
        has_acts = (
            acts_work is not None and acts_work.shape[-1] == w_work.shape[1]
        )

        def _mxfp4_rtn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_rtn, s_rtn = quantize_dequantize_mxfp4(w_work, group_size=32)
            return q_rtn, s_rtn, _mxfp4_dequantize_2d(q_rtn, s_rtn, group_size=32)

        if gptq_enabled and has_acts:
            assert acts_work is not None

            def _mxfp4_gptq_candidate(
                use_static_act_order: bool,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0":
                    return _gptq_obs_rounding_mxfp4_swept(
                        w_work,
                        acts_work,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        static_act_order=use_static_act_order,
                    )
                return _gptq_obs_rounding_mxfp4(
                    w_work,
                    acts_work,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=use_static_act_order,
                )

            candidates = [_mxfp4_gptq_candidate(False)]
            if static_act_order_enabled:
                candidates.append(_mxfp4_gptq_candidate(True))
            wp, ws, dq = min(
                candidates,
                key=lambda cand: _activation_weighted_weight_error(
                    w_work,
                    cand[2],
                    acts_work,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    row_weights=fisher_row_weights,
                ),
            )
            if os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0":
                try:
                    q_rtn, s_rtn, dq_rtn = _mxfp4_rtn()
                    err_rtn = _activation_weighted_weight_error(
                        w_work,
                        dq_rtn,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                    )
                    err_work = _activation_weighted_weight_error(
                        w_work,
                        dq,
                        acts_work,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=act_clip_rescale,
                        row_weights=fisher_row_weights,
                    )
                    if err_rtn < err_work:
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP4 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        wp, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception:
                    pass
        else:
            wp, ws, _ = _mxfp4_rtn()
        return {"weight_packed": wp, "weight_scale": ws}
    if fmt == "BF16":
        return {"weight": weight.to(torch.bfloat16)}
    raise ValueError(f"unsupported format: {fmt}")


def _quantize_3d_packed(packed: torch.Tensor, fmt: str) -> dict[str, torch.Tensor]:
    """Compress a 3D packed-expert tensor `[E, M, N]` as a single
    batched op (per-expert independent scales).

    Returns tensors with leading expert dim preserved, matching what
    vLLM's `compressed_tensors_moe_w4a4_nvfp4` allocates internally
    (uint8 packed weights, fp8/uint8 per-group scales, per-expert
    global scales for NVFP4).
    """
    fmt = _canonical_export_format(fmt)
    if fmt == "BF16":
        return {"weight": packed.to(torch.bfloat16)}
    if fmt == "NVFP4":
        wp, ws, wg = quantize_dequantize_nvfp4_packed(packed, group_size=16)
        return {
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg,
        }
    if fmt in MXFP8_EXPLICIT_FORMATS:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_mxfp8_packed(
            packed,
            group_size=32,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_fp8_dynamic_packed(
            packed.to(torch.float32),
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    if fmt == "MXFP4":
        wp, ws = quantize_dequantize_mxfp4_packed(packed, group_size=32)
        return {"weight_packed": wp, "weight_scale": ws}
    raise ValueError(f"unsupported format for packed-MoE: {fmt}")


def _finalize_compute_only(compute_dict: dict, *,
                           weight_override: torch.Tensor | None = None
                           ) -> dict[str, torch.Tensor]:
    """Pack a compute_only result from `_quantize_2d` into the final
    on-disk tensor dict. When `weight_override` is supplied (e.g. after
    block-output match modified the dequantized weight), pack that
    instead of the original `_w_dq`.

    Currently only NVFP4 is supported in compute_only mode. Other
    formats fall through to a clear error so a misuse fails loudly
    rather than silently silently corrupting the artifact.
    """
    fmt = compute_dict.get("_fmt")
    if fmt != "NVFP4":
        raise ValueError(
            f"_finalize_compute_only: only NVFP4 is supported "
            f"(got fmt={fmt}). Other formats should not be in "
            f"compute_only mode.")
    w = compute_dict["_w_dq"] if weight_override is None else weight_override
    nvfp4_global_real = compute_dict["_nvfp4_global_real"]
    input_scale = compute_dict["_input_scale"]

    wp, ws, wg = quantize_dequantize_nvfp4(
        w, group_size=16,
        global_real_override=nvfp4_global_real,
    )
    return {
        "weight_packed": wp,
        "weight_scale": ws,
        "weight_global_scale": wg,
        "input_global_scale": torch.tensor(
            [float(input_scale)], dtype=torch.float32,
        ),
    }


def _quantize_2d_group_same_shape(
    stacked_weights: torch.Tensor,
    fmt: str,
) -> dict[str, torch.Tensor]:
    """Compress a batch of same-shape 2D weights in one vectorized op.

    `stacked_weights` is `[B, out, in]`. Returned tensors keep the leading
    batch dimension so the caller can split them back to per-Linear keys.
    This is deliberately limited to RTN-only formats: activation-aware NVFP4
    remains scalar until its GPTQ/scale-sweep passes are vectorized too.
    """
    fmt = _canonical_export_format(fmt)
    if stacked_weights.dim() != 3:
        raise ValueError(
            "same-shape export grouping expects [B, out, in] weights; "
            f"got shape={tuple(stacked_weights.shape)}"
        )
    if fmt in MXFP8_EXPLICIT_FORMATS:
        element_dtype, element_max = _fp8_element_dtype_and_max(fmt)
        w, ws = quantize_dequantize_mxfp8_packed(
            stacked_weights.to(torch.float32),
            group_size=32,
            element_dtype=element_dtype,
            element_max=element_max,
        )
        return {"weight": w, "weight_scale": ws}
    raise ValueError(f"unsupported grouped 2D export format: {fmt}")


def _quantize_2d_nvfp4_group_batched(
    items: list,
    joint_globals: dict,
    device: torch.device,
    expert_chunk: int = 32,
) -> list[dict]:
    """Batched NVFP4 quantization for a same-shape group of Linears
    when activation-aware passes (GPTQ / scale_sweep) are enabled.

    Replaces the per-Linear `_quantize_2d` flow's slow steps (GPTQ +
    scale_sweep) with the batched analogs in
    `prismaquant.export_batched_gptq`. The fast steps (final NVFP4
    pack, input-global-scale lookup) stay per-Linear since they are
    already cheap.

    Items: list of `(full, emit_full, recipe_key, mod)` tuples. All
    `mod.weight` must share `(out, in)` shape. The function returns a
    list of compressed dicts in the same order, ready to be merged
    into the export's `out` dict by the caller.

    The reference weight passed to scale_sweep is the same original weight
    used by the per-Linear path's `weight.to(float32)` argument.
    """
    from .export_batched_gptq import (
        gptq_obs_rounding_nvfp4_batched,
        scale_sweep_nvfp4_batched,
    )

    n = len(items)
    if n == 0:
        return []

    # Stack weights into [E, out, in]. All shapes must match.
    weights = torch.stack(
        [it[3].weight.detach().to(torch.float32) for it in items], dim=0,
    ).to(device)
    reference_weights = weights.clone()  # pre-pass reference for scale_sweep

    # Per-Linear activation tensors (None where missing).
    acts_list: list = []
    for full, emit_full, recipe_key, mod in items:
        a = None
        if _CACHED_ACTIVATIONS is not None:
            raw = _CACHED_ACTIVATIONS.get(recipe_key)
            if raw is not None and raw.shape[-1] == mod.weight.shape[1]:
                a = raw.to(torch.float32).reshape(-1, raw.shape[-1])
        acts_list.append(a if a is not None else torch.zeros(
            0, weights.shape[2], dtype=torch.float32, device=device))

    # Per-Linear NVFP4 global_real overrides (from joint fused-sibling).
    # When recipe_key isn't in joint_globals, the batched path computes
    # per-Linear from the weights — pass `None` for that Linear. We
    # represent the override array as a [E] tensor with NaN for "no
    # override"; the batched function expects a single tensor of shape
    # [E], so we must split into "all overridden" or "none overridden"
    # groups within this function or fall back to per-Linear when mixed.
    overrides_list = [joint_globals.get(it[2]) for it in items]
    if all(v is not None for v in overrides_list):
        global_real_overrides = torch.stack(
            [v.to(device, dtype=torch.float32) for v in overrides_list]
        ).reshape(n)
    elif all(v is None for v in overrides_list):
        global_real_overrides = None
    else:
        # Mixed — split into homogeneous sub-groups and recurse.
        with_idx = [i for i, v in enumerate(overrides_list) if v is not None]
        without_idx = [i for i, v in enumerate(overrides_list) if v is None]
        results: list[dict] = [None] * n  # type: ignore[list-item]
        if with_idx:
            sub = [items[i] for i in with_idx]
            sub_results = _quantize_2d_nvfp4_group_batched(
                sub, joint_globals, device, expert_chunk=expert_chunk,
            )
            for i, r in zip(with_idx, sub_results):
                results[i] = r
        if without_idx:
            sub = [items[i] for i in without_idx]
            sub_results = _quantize_2d_nvfp4_group_batched(
                sub, joint_globals, device, expert_chunk=expert_chunk,
            )
            for i, r in zip(without_idx, sub_results):
                results[i] = r
        return results

    # Run the batched activation-aware passes. Match the per-Linear
    # `_quantize_2d` ordering: GPTQ → scale_sweep.
    # Codex review #46 batched extension: per-Linear damping sweep.
    # Run GPTQ at each candidate damp, measure activation-weighted
    # output MSE per Linear, keep the best per Linear. Cost is
    # n_candidates × the unswept GPTQ pass; gated by env so prod
    # default keeps the existing single-damp speed.
    if _ACT_AWARE_FLAGS["gptq"]:
        # Default ON (validated on Qwen3-0.6B audit). =0 to disable.
        damp_sweep_on = (
            os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0")
        if damp_sweep_on:
            damp_candidates = (0.001, 0.005, 0.01, 0.05, 0.1)
            best_w = None
            best_err = None  # [E] of activation-weighted MSE
            # Pre-compute per-Linear column importance for the gate.
            col_imp = torch.empty(
                (n, weights.shape[2]), device=device, dtype=torch.float32)
            for j, a in enumerate(acts_list):
                if a is None or a.numel() == 0:
                    col_imp[j] = 1.0
                else:
                    col_imp[j] = _activation_col_importance_for_gptq(
                        a, weights.shape[2], device=device)
            for damp in damp_candidates:
                cand_w = gptq_obs_rounding_nvfp4_batched(
                    weights, acts_list,
                    damp=damp,
                    global_real_overrides=global_real_overrides,
                    expert_chunk=expert_chunk,
                    static_act_order=bool(
                        _ACT_AWARE_FLAGS.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        _ACT_AWARE_FLAGS.get("joint_scale_opt", False)
                    ),
                )
                # Per-Linear activation-weighted MSE vs reference.
                diff = reference_weights - cand_w
                err = (col_imp.unsqueeze(1) * diff.pow(2)).sum(dim=(1, 2))
                if best_w is None:
                    best_w = cand_w
                    best_err = err
                else:
                    take = err < best_err
                    if take.any():
                        idx = take.nonzero(as_tuple=True)[0]
                        best_w[idx] = cand_w[idx]
                        best_err[idx] = err[idx]
            weights = best_w
        else:
            weights = gptq_obs_rounding_nvfp4_batched(
                weights, acts_list,
                global_real_overrides=global_real_overrides,
                expert_chunk=expert_chunk,
                static_act_order=bool(
                    _ACT_AWARE_FLAGS.get("static_act_order", False)
                ),
                joint_scale_opt=bool(
                    _ACT_AWARE_FLAGS.get("joint_scale_opt", False)
                ),
            )
    if _ACT_AWARE_FLAGS["scale_sweep"]:
        weights = scale_sweep_nvfp4_batched(
            weights, acts_list,
            reference_weights=reference_weights,
            global_real_overrides=global_real_overrides,
            expert_chunk=expert_chunk,
        )

    # Codex review #47 batched extension: per-Linear do-no-harm gate.
    # If the post-pass weight is worse on activation-weighted MSE than
    # a pure RTN of the original, swap that single Linear back to RTN.
    # Same default-on as the per-Linear path; PRISMAQUANT_DO_NO_HARM=0
    # disables. Cost: one RTN dequant + two MSE sums per Linear.
    if (_ACT_AWARE_FLAGS["gptq"]
            and os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0"):
        try:
            # Per-Linear activation column importance.
            col_imp = torch.empty(
                (n, weights.shape[2]), device=device, dtype=torch.float32)
            n_acts_avail = 0
            for j, a in enumerate(acts_list):
                if a is None or a.numel() == 0:
                    col_imp[j] = 1.0
                else:
                    col_imp[j] = _activation_col_importance_for_gptq(
                        a, weights.shape[2], device=device)
                    n_acts_avail += 1
            n_reverted = 0
            for i in range(n):
                if acts_list[i] is None or acts_list[i].numel() == 0:
                    continue  # no activations → can't gate; trust the pass
                override = overrides_list[i]
                w_rtn = _rtn_dequant_nvfp4(
                    reference_weights[i], group_size=16,
                    global_real_override=override,
                )
                ref_i = reference_weights[i]
                imp = col_imp[i]
                mse_pass = float(
                    (imp * (ref_i - weights[i]).pow(2).sum(dim=0)).sum())
                mse_rtn = float(
                    (imp * (ref_i - w_rtn).pow(2).sum(dim=0)).sum())
                if mse_rtn < mse_pass:
                    weights[i] = w_rtn
                    n_reverted += 1
            if n_reverted and os.environ.get(
                    "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                print(f"[do-no-harm batched] reverted {n_reverted}/{n} "
                      f"Linears to RTN", flush=True)
        except Exception as _e:
            print(f"[do-no-harm batched] WARN failed: {_e}", flush=True)

    # Per-Linear final NVFP4 pack (cheap; reuses the existing function).
    out: list[dict] = []
    for i, (full, emit_full, recipe_key, mod) in enumerate(items):
        override = overrides_list[i]
        wp, ws, wg = quantize_dequantize_nvfp4(
            weights[i], group_size=16,
            global_real_override=override,
        )
        input_scale = (
            _INPUT_GLOBAL_SCALES.get(recipe_key) if _INPUT_GLOBAL_SCALES
            else None
        )
        if input_scale is None:
            input_scale = DEFAULT_INPUT_GLOBAL_SCALE
        out.append({
            "weight_packed": wp,
            "weight_scale": ws,
            "weight_global_scale": wg.reshape(1)
            if wg.dim() == 0 else wg,
            "input_global_scale": torch.tensor(
                [float(input_scale)], dtype=torch.float32),
        })
    return out


def _host_mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 30


def _export_vector_chunk_len(
    shape: tuple[int, int],
    max_items: int,
    device: torch.device,
) -> int:
    """Choose a conservative grouped-export chunk size.

    `PQ_EXPORT_VECTOR_CHUNK=<int>` pins the upper bound. The default `auto`
    keeps one path for all model sizes while scaling down when available
    memory is tight.
    """
    env = os.getenv("PQ_EXPORT_VECTOR_CHUNK", "auto").strip().lower()
    if env and env != "auto":
        try:
            cap = max(1, int(env))
        except ValueError:
            cap = 128
    else:
        cap = 128

    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except RuntimeError:
            free_bytes = _host_mem_available_bytes()
    else:
        free_bytes = _host_mem_available_bytes()

    # Quantization creates grouped float32 views, integer code tensors, scale
    # tensors, and packed outputs. Budget for several live copies per item.
    per_item = max(1, int(math.prod(shape)) * 4)
    budget = max(16 << 20, min(int(free_bytes * 0.08), 2 << 30))
    by_mem = max(1, budget // max(per_item * 6, 1))
    return max(1, min(max_items, cap, by_mem))


# ---------------------------------------------------------------------------
# Fused-sibling joint NVFP4 scale (per-layer scope, used by the streaming
# materializer below). The whole-model variant `_compute_nvfp4_joint_global`
# lives above and is kept for the MTP path + unit tests.
# ---------------------------------------------------------------------------
def _compute_layer_joint_nvfp4(layer_mod: nn.Module,
                               layer_qname: str,
                               assignment: dict[str, str],
                               profile,
                               ) -> dict[str, torch.Tensor]:
    """Return {recipe_key -> joint global scale} for NVFP4 fused-sibling
    groups inside this decoder layer. Only keys assigned NVFP4 get an
    override entry; the rest compute per-Linear scales at quantize time.

    Semantically equivalent to a scoped `_compute_nvfp4_joint_global`
    across just this layer's modules."""
    groups: dict[str, list[tuple[str, str, nn.Linear]]] = defaultdict(list)
    for sub_name, mod in layer_mod.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        full = f"{layer_qname}.{sub_name}" if sub_name else layer_qname
        try:
            recipe_key = profile.live_to_recipe_name(full)
        except Exception:
            recipe_key = full
        group_key = _fused_group_key_for_name(recipe_key, profile)
        if group_key is None:
            continue
        groups[group_key].append((full, recipe_key, mod))

    out: dict[str, torch.Tensor] = {}
    for _group_key, members in groups.items():
        fqn_fmt = []
        for full, recipe_key, mod in members:
            fmt = assignment.get(recipe_key)
            fqn_fmt.append((full, recipe_key, fmt, mod))
        fmts = {_canonical_export_format(f) for _, _, f, _ in fqn_fmt}
        if fmts != {"NVFP4"}:
            continue
        candidates = [
            compute_nvfp4_global_real(mod.weight.detach().float(),
                                      group_size=16)
            for _, _, _, mod in fqn_fmt
        ]
        joint = torch.stack(candidates).max()
        for full, recipe_key, _, _ in fqn_fmt:
            out[recipe_key] = joint
    return out


_SAFETENSORS_DTYPE_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _build_source_dtype_map(
    model_to_shard: dict[str, str],
    model_to_ckpt: dict[str, str],
) -> dict[str, torch.dtype]:
    """Return live tensor qname -> dtype from source safetensors metadata."""
    from safetensors import safe_open

    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        by_shard[shard].append((model_name, model_to_ckpt[model_name]))

    out: dict[str, torch.dtype] = {}
    for shard, pairs in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            for model_name, ckpt_name in pairs:
                if ckpt_name not in keys:
                    continue
                label = f.get_slice(ckpt_name).get_dtype()
                dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(label)
                if dtype is not None:
                    out[model_name] = dtype
    return out


def _dtype_hist_label(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "BF16"
    if dtype == torch.float16:
        return "FP16"
    if dtype == torch.float32:
        return "FP32"
    if dtype == torch.float64:
        return "FP64"
    if dtype == torch.float8_e4m3fn:
        return "FP8_E4M3"
    if dtype == torch.float8_e5m2:
        return "FP8_E5M2"
    return str(dtype).replace("torch.", "").upper()


def _passthrough_dtype(
    qname: str,
    source_dtype: torch.dtype | None = None,
    *,
    fallback_dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Pick the storage dtype for a passthrough (non-quantized) param.

    Passthrough means source-preserving. Do not silently upcast norms or
    other parameters here; if a future recipe wants FP32 norms, it should
    request that as an explicit transform and record it in the manifest.
    """
    if source_dtype is not None:
        return source_dtype
    if fallback_dtype is not None:
        return fallback_dtype
    raise ValueError(f"missing source dtype for passthrough tensor {qname}")


def _passthrough_tensor(
    qname: str,
    tensor: torch.Tensor,
    source_dtype_by_name: dict[str, torch.dtype] | None = None,
) -> tuple[torch.Tensor, str]:
    dtype = _passthrough_dtype(
        qname,
        None if source_dtype_by_name is None else source_dtype_by_name.get(qname),
        fallback_dtype=tensor.dtype,
    )
    return tensor.detach().to(dtype).cpu(), _dtype_hist_label(dtype)


def _init_rotary_inplace(base_model: nn.Module, device: torch.device,
                         dtype: torch.dtype) -> None:
    """After init_empty_weights, rotary modules exist but their
    `inv_freq` buffers are on meta. Re-run the module's own rope init
    (which is deterministic from config) so `inv_freq` lives on the
    exec device with correct values — matching what `from_pretrained`
    would have produced."""
    from .layer_streaming import _get_rotary
    rotary = _get_rotary(base_model)
    if rotary is None:
        return
    cfg = getattr(rotary, "config", None)
    if cfg is None:
        return
    try:
        rope_init_fn = rotary.compute_default_rope_parameters
    except AttributeError:
        return
    if hasattr(rotary, "reset_rope_cache"):
        rotary.reset_rope_cache(device)
        return
    inv_freq, attention_scaling = rope_init_fn(cfg, device)
    rotary.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32,
                                                   device=device),
                           persistent=False)
    if hasattr(rotary, "original_inv_freq"):
        rotary.register_buffer(
            "original_inv_freq",
            inv_freq.to(dtype=torch.float32, device=device).clone(),
            persistent=False)
    rotary.attention_scaling = attention_scaling


def _build_fp8_source_map(
    model_path: str, *, multimodal: bool = False,
) -> dict[str, tuple[str, str]]:
    """Scan the source safetensors index for native-FP8 block-scaled
    Linears and return `{live_base_name: (shard_path, ckpt_scale_inv_key)}`.

    A tensor qualifies as FP8-sourced when `<base>.weight` has a sibling
    `<base>.weight_scale_inv` in the index (the 128×128 block-scale
    convention MiniMax-M2, DeepSeek-V3, and NVIDIA FP8 checkpoints use).
    The returned keys are the LIVE-MODEL attribute paths (i.e., the same
    form as `full` in the per-layer loop), obtained by applying the same
    source → live name rewrite that `layer_streaming._build_weight_map`
    performs for the `.weight` tensors — so the exporter can look up
    directly by `live_base` without re-running the rewrite.

    `multimodal` must match what was passed to `_build_weight_map`:
    text-only path strips `model.language_model.` prefix; multimodal
    preserves it. (MiniMax-M2 is text-only; set False.)

    Returns `{}` when the source has no `.weight_scale_inv` sibling for
    any `.weight` — i.e., the source is not FP8-block quantized. In that
    case the FP8_SOURCE format is inert (allocator's passthrough-
    integrity filter drops it from every Linear's candidate set).
    """
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            return {}
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}
    else:
        with open(idx_path) as f:
            raw = json.load(f)["weight_map"]

    def _rename(k: str) -> str | None:
        # Mirror `layer_streaming._rename_text_only`, but WITHOUT the
        # `.weight_scale_inv` drop — we need those keys preserved.
        if not multimodal:
            if (k.startswith("model.visual.")
                    or k.startswith("model.audio_tower.")
                    or k.startswith("model.vision_tower.")
                    or k.startswith("model.embed_vision.")
                    or k.startswith("model.embed_audio.")
                    or k.startswith("mtp.")):
                return None
            if k.startswith("model.language_model."):
                return "model." + k[len("model.language_model."):]
            return k
        # multimodal umbrella
        if k.startswith("mtp."):
            return None
        return k

    # Group by `<live_base>`: the live-model qname without `.weight` /
    # `.weight_scale_inv` suffix.
    bases: dict[str, dict[str, tuple[str, str]]] = {}
    for ck_key, shard in raw.items():
        for suffix in (".weight_scale_inv", ".weight"):
            if ck_key.endswith(suffix):
                ck_base = ck_key[: -len(suffix)]
                live_base = _rename(ck_base)
                if live_base is None:
                    break
                bases.setdefault(live_base, {})[suffix[1:]] = (
                    os.path.join(model_path, shard), ck_key,
                )
                break

    out: dict[str, tuple[str, str]] = {}
    for live_base, kinds in bases.items():
        if "weight" in kinds and "weight_scale_inv" in kinds:
            # Only the scale_inv half is new information — the `.weight`
            # shard+ckpt_key is already in `weight_ckpt` from the main
            # loader. Callers combine the two.
            shard, ckpt_scale_inv_key = kinds["weight_scale_inv"]
            out[live_base] = (shard, ckpt_scale_inv_key)
    return out


def materialize_tensors_streaming(
    model_path: str,
    assignment: dict[str, str],
    *,
    profile,
    bf16_passthrough: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    offload_folder: str | None = None,
    tensor_sink: Callable[[dict[str, torch.Tensor]], None] | None = None,
    export_cache_dir: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stream decoder layers through quantize → emit → unload. Never
    holds the full model in memory. Small models still exercise this
    path — the LayerCache just keeps everything resident, so load/
    unload degenerates to a no-op.

    Output: `(out_tensors, hist)` matching the shape the monolithic
    materialize used to return, ready for `write_sharded_safetensors`.
    When `tensor_sink` is supplied, each emitted head/layer batch is
    passed to the sink and cleared immediately; the returned tensor dict
    is then intentionally empty."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .layer_streaming import (
        _build_expert_packer,
        _build_fp8_scale_inv_map,
        _build_install_resolver,
        _build_weight_map,
        _fast_install,
        _get_layer_list,
        _head_prefixes,
        _materialize,
        _read_layer_to_device,
        _resolve_base_prefix,
        _unload,
    )
    from .sensitivity_probe import stage_text_only

    # ----- 1. Meta skeleton + manual head materialization -----
    # Pure `init_empty_weights` path — avoids accelerate's
    # `from_pretrained` which would write ~244 GB of offload files to
    # disk on Qwen3.5-122B before we ever read them. Instead we:
    #   (a) build the full skeleton on meta (0 bytes),
    #   (b) read head/embed/norm/lm_head tensors directly from the
    #       source safetensors and install on the exec device,
    #   (c) re-run rotary's init_fn to populate `inv_freq` (not in
    #       state_dict — computed from config),
    #   (d) leave decoder layers on meta until the per-layer loop
    #       streams them in.
    staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    # _init_weights is globally no-op'd by prismaquant.__init__'s
    # _polyfill_transformers (wasted work + transformers-5.x compat
    # landmine on remote modeling files).
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    source_dtype_by_name = _build_source_dtype_map(weight_shard, weight_ckpt)
    # Per-expert -> packed-3D bridge for checkpoints that ship MoE experts
    # unfused while the live module is packed (driven by the model profile;
    # None for every other model). Keeps the exporter's source read aligned
    # with the streaming probe/cost path — a raw checkpoint exports without
    # an out-of-band pre-pack.
    expert_packer = _build_expert_packer(model, weight_ckpt)
    # Native-FP8 dequant map, keyed by live weight-qname. Passed to
    # every `_read_layer_to_device` / `_materialize` call so fp8 source
    # weights land on the module as TRUE dequanted bf16 — not raw fp8
    # codes (range ±448) cast to bf16. Every downstream pass
    # (_quantize_2d for non-passthrough formats, probe Fisher, cost
    # RTN) then operates on the real weight values instead of scaled-
    # by-hidden-factor codes. Empty dict for BF16-native checkpoints.
    fp8_scale_inv_map = _build_fp8_scale_inv_map(model_path)
    if fp8_scale_inv_map:
        print(f"[export-stream] fp8 scale_inv map: "
              f"{len(fp8_scale_inv_map)} weights will be dequanted "
              f"inline at layer-load", flush=True)

    # FP8_SOURCE passthrough-emit map: keyed by live base name (no
    # `.weight` suffix), used by the `fmt == 'FP8_SOURCE'` emit branch
    # to copy source fp8 + scale_inv bytes verbatim into the output.
    # Distinct key format from the loader-side dequant map above.
    fp8_source_map = _build_fp8_source_map(model_path)
    if fp8_source_map:
        print(f"[export-stream] fp8 source-emit map: {len(fp8_source_map)} "
              f"Linears available for FP8_SOURCE passthrough", flush=True)

    # Materialize head (embed + norm + lm_head). These are in the
    # safetensors and get populated via `set_module_tensor_to_device`.
    print(f"[export-stream] base_prefix={base_prefix!r}  layers={num_layers}",
          flush=True)
    t0 = time.time()
    head_pfxs = _head_prefixes(None, base_prefix)
    loaded_n = _materialize(model, head_pfxs, weight_shard, weight_ckpt,
                            device, dtype,
                            fp8_scale_inv_map=fp8_scale_inv_map)

    # Rotary's `inv_freq` isn't in the state_dict — compute from config.
    _init_rotary_inplace(base_model, device, dtype)
    print(f"[export-stream] head materialized ({loaded_n} tensors, rotary "
          f"re-init) in {time.time()-t0:.1f}s", flush=True)

    out: dict[str, torch.Tensor] = {}
    hist: Counter = Counter()
    unmapped_keys: list[str] = []

    # ----- 2. Head / embed / norm / lm_head / rotary passthrough -----
    # These are resident on `device` already. Emit as source-dtype
    # passthrough UNLESS `lm_head` (or similar) is explicitly in the
    # assignment.
    t_head = time.time()

    def _emit_head_param(full_qname: str, param: nn.Parameter):
        recipe_key = profile.live_to_recipe_name(full_qname)
        # Recipe keys are module qnames (e.g. "lm_head"), not parameter
        # qnames ("lm_head.weight"). Strip the trailing `.weight` so the
        # assignment lookup hits — otherwise head params always fall
        # through to source-dtype passthrough regardless of what the
        # allocator chose for them.
        if recipe_key.endswith(".weight"):
            recipe_key = recipe_key[:-len(".weight")]
        recipe_fmt = assignment.get(recipe_key)
        fmt = recipe_fmt
        # Respect the passthrough set (e.g. `--ignore lm_head`) even if
        # the allocator assigned NVFP4/MXFP8_E4M3/MXFP8_E5M2 to this head module. See
        # the --ignore docstring for why lm_head is passthrough by
        # default despite vLLM rejecting quantized ParallelLMHead.
        if recipe_key in bf16_passthrough:
            fmt = "BF16"
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt == "FP8_SOURCE":
            raise NotImplementedError(
                f"[export-stream] FP8_SOURCE not wired for head params "
                f"(at {full_qname}). Native-FP8 checkpoints (MiniMax, "
                f"DeepSeek) keep lm_head/embed/norm in BF16 — the "
                f"allocator's passthrough-integrity filter should reject "
                f"FP8_SOURCE for these. If a future model ships FP8 "
                f"head weights, add the passthrough path here.")
        if fmt is not None and fmt != "BF16":
            joint = None
            compressed = _pack_production_cached_2d(
                recipe_key,
                recipe_fmt if recipe_fmt is not None else fmt,
                nvfp4_global_real_override=joint,
                device=device,
            )
            if compressed is None:
                compressed = _quantize_2d(
                    param.detach().float(), fmt,
                    nvfp4_global_real_override=joint,
                    linear_name=recipe_key,
                )
            for suffix, t in compressed.items():
                base_name = (full_qname[:-len(".weight")]
                             if full_qname.endswith(".weight")
                             else full_qname)
                out_key = (base_name
                           if suffix == "weight"
                           else f"{base_name}.{suffix}")
                out[out_key] = t.cpu()
            hist[("head", fmt)] += 1
        else:
            out[full_qname], label = _passthrough_tensor(
                full_qname, param, source_dtype_by_name)
            hist[("head_passthrough", label)] += 1

    for name, p in model.named_parameters():
        if p.is_meta:
            continue  # only head/embed/norm/lm_head resident here
        if name.startswith(layers_prefix):
            continue
        _emit_head_param(name, p)

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            if buf.is_meta:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if full.startswith(layers_prefix):
                continue
            if full in out:
                continue
            out[full], label = _passthrough_tensor(
                full, buf, source_dtype_by_name)
            hist[("head_buffer", label)] += 1
    print(f"[export-stream] head+embed+norm+lm_head passthrough: "
          f"{time.time()-t_head:.1f}s  keys={len(out)}", flush=True)
    if tensor_sink is not None:
        tensor_sink(out)
        out = {}

    # ----- 3. Per-layer streaming quantize loop -----
    # v25: per-layer cache. When --export-cache-dir is set, each
    # layer's emitted tensor dict is torch.save'd to a per-layer file
    # AFTER quantization succeeds. On a restart the loop checks each
    # layer's cache file and SKIPS the quantization work for any layer
    # already cached — instead loads the saved dict and replays it
    # into tensor_sink. Recovers full progress from a mid-flight kill.
    cache_path = Path(export_cache_dir) if export_cache_dir else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

        # Cache fingerprint (codex review #2): bind the cache to the
        # quality-affecting state. If any of these change between runs,
        # the cache is silently wrong because the saved layer tensors
        # were quantized under a different recipe. Write/check a
        # manifest.json; mismatch invalidates the cache wholesale.
        import json as _json
        fp_state = {
            "PRISMAQUANT_DO_NO_HARM": os.environ.get(
                "PRISMAQUANT_DO_NO_HARM", "1"),
            "PRISMAQUANT_GPTQ_DAMP_SWEEP": os.environ.get(
                "PRISMAQUANT_GPTQ_DAMP_SWEEP", "1"),
            "PRISMAQUANT_ACT_CLIP_QUANTILE": os.environ.get(
                "PRISMAQUANT_ACT_CLIP_QUANTILE", "0.999"),
            "PRISMAQUANT_BLOCK_OUTPUT_MATCH": os.environ.get(
                "PRISMAQUANT_BLOCK_OUTPUT_MATCH", "1"),
            "PRISMAQUANT_BATCHED_NVFP4_EXPORT": os.environ.get(
                "PRISMAQUANT_BATCHED_NVFP4_EXPORT", "1"),
            NVFP4_SCALE_RULE_ENV: _nvfp4_scale_rule_from_env(),
            "ACT_AWARE_FLAGS": dict(sorted(_ACT_AWARE_FLAGS.items())),
            "activation_cache_fingerprint": _ACTIVATION_CACHE_FINGERPRINT,
            "production_cache_fingerprint": _PRODUCTION_CACHE_FINGERPRINT,
        }
        # Hash the assignment dict (layer_config recipe) too — recipe
        # changes invalidate per-Linear quantization output.
        try:
            fp_state["assignment_hash"] = hashlib.sha256(
                _json.dumps(assignment, sort_keys=True).encode()
            ).hexdigest()[:16]
        except Exception:
            fp_state["assignment_hash"] = None

        manifest_path = cache_path / "manifest.json"
        if manifest_path.exists():
            try:
                with manifest_path.open() as _f:
                    prev = _json.load(_f)
                if prev != fp_state:
                    diff_keys = sorted(
                        set(prev.keys()) | set(fp_state.keys())
                    )
                    diffs = [
                        k for k in diff_keys
                        if prev.get(k) != fp_state.get(k)
                    ]
                    print(f"[export-stream] cache fingerprint MISMATCH "
                          f"(differs in: {diffs}); invalidating cache",
                          flush=True)
                    for _f in cache_path.glob("layer_*.pt"):
                        _f.unlink()
                    with manifest_path.open("w") as _f:
                        _json.dump(fp_state, _f, indent=2)
                else:
                    print(f"[export-stream] cache fingerprint match — "
                          f"resumable from {len(list(cache_path.glob('layer_*.pt')))} "
                          f"layers", flush=True)
            except Exception as _e:
                print(f"[export-stream] cache manifest unreadable "
                      f"({_e}); invalidating cache", flush=True)
                for _f in cache_path.glob("layer_*.pt"):
                    _f.unlink()
                with manifest_path.open("w") as _f:
                    _json.dump(fp_state, _f, indent=2)
        else:
            with manifest_path.open("w") as _f:
                _json.dump(fp_state, _f, indent=2)
            print(f"[export-stream] wrote cache fingerprint to {manifest_path}",
                  flush=True)

    def _layer_cache_file(L: int) -> Path | None:
        return None if cache_path is None else cache_path / f"layer_{L:03d}.pt"

    t_layers = time.time()
    cache_hits = 0
    for L in range(num_layers):
        if tensor_sink is not None:
            out = {}
        layer_t0 = time.time()
        layer_qname = f"{layers_prefix}{L}".rstrip(".")
        if layer_qname.endswith("."):
            layer_qname = layer_qname[:-1]
        if _PRODUCTION_WEIGHT_CACHE is not None:
            layer_recipe_prefix = profile.live_to_recipe_name(layer_qname)
            prefetched = _production_cache_prefetch_assignment(
                assignment,
                prefix=layer_recipe_prefix,
            )
            if prefetched and (L % 4 == 0 or L == num_layers - 1):
                print(
                    f"[export-stream] layer {L:02d} production-cache "
                    f"prefetch={prefetched}",
                    flush=True,
                )

        # v25: cache hit — skip quantization, replay cached tensor dict.
        cf = _layer_cache_file(L)
        if cf is not None and cf.exists():
            cached = torch.load(str(cf), weights_only=False, map_location="cpu")
            if tensor_sink is not None:
                tensor_sink(cached)
            else:
                out.update(cached)
            cache_hits += 1
            if L % 4 == 0 or L == num_layers - 1:
                print(f"[export-stream] layer {L:02d}  CACHED "
                      f"keys={len(cached)}", flush=True)
            del cached
            continue

        # 3a. Load layer from safetensors (direct to device). When
        # `fp8_scale_inv_map` is non-empty, the loader applies the
        # 128x128 block dequant inline, so `mod.weight` receives the
        # true dequanted weight rather than raw fp8 codes cast to bf16.
        load_t0 = time.time()
        tensors = _read_layer_to_device(
            f"{layers_prefix}{L}.", weight_shard, weight_ckpt, dtype, device,
            fp8_scale_inv_map=fp8_scale_inv_map, pack_experts=expert_packer)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Joint NVFP4 scales across fused siblings in this layer.
        joint_globals = _compute_layer_joint_nvfp4(
            layer_mod, layer_qname, assignment, profile,
        )

        # 3c. Emit Linears.
        covered: set[str] = set()
        linear_count = 0
        grouped_linears: dict[
            tuple[str, tuple[int, int]],
            list[tuple[str, str, str, nn.Linear]]  # (full, emit_full, recipe_key, mod)
        ] = defaultdict(list)
        # v23 (opt-in): batch NVFP4 same-shape Linears when act-aware
        # passes (GPTQ / scale_sweep) are on. Activated by env var
        # PRISMAQUANT_BATCHED_NVFP4_EXPORT=1 — disabled by default while
        # the path is being validated against the per-Linear baseline.
        # When inactive, NVFP4 Linears go through the per-Linear
        # `_quantize_2d` exactly as before.
        grouped_nvfp4_batched: dict[
            tuple[int, int],
            list[tuple[str, str, str, nn.Linear]]
        ] = defaultdict(list)
        # v26: default ON. Set PRISMAQUANT_BATCHED_NVFP4_EXPORT=0 to revert
        # to per-Linear NVFP4 quantization (slower but provably correct).
        _raw_batched = os.environ.get("PRISMAQUANT_BATCHED_NVFP4_EXPORT")
        _batched_env_on = (
            True if _raw_batched is None
            else _raw_batched not in ("0", "", "false", "False", "FALSE", "no", "NO")
        )
        _batched_nvfp4_enabled = (
            _batched_env_on
            and (_ACT_AWARE_FLAGS["gptq"] or _ACT_AWARE_FLAGS["scale_sweep"])
            and _CACHED_ACTIVATIONS is not None
        )

        # #12 Block-output match deferred-pack list. Per-layer scope.
        _BLOCK_COMPUTE_PENDING: list[dict] = []
        # Capture FP16 snapshots of the layer's standard block Linears
        # so we can run a reference (pre-quantization) forward pass for
        # block-output match. Cheap: a layer's q/k/v/o + gate/up/down at
        # FP32 ≈ 64-128 MB.
        _FP16_BLOCK_SNAPSHOTS: dict[str, torch.Tensor] = {}
        if os.environ.get("PRISMAQUANT_BLOCK_OUTPUT_MATCH", "1") != "0":
            for _sn, _m in layer_mod.named_modules():
                if not isinstance(_m, nn.Linear):
                    continue
                _leaf = _sn.rsplit(".", 1)[-1] if _sn else ""
                if _leaf in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                    "gate_proj", "up_proj", "down_proj",
                ):
                    _FP16_BLOCK_SNAPSHOTS[_sn] = _m.weight.detach().clone()

        for sub_name, mod in layer_mod.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            linear_count += 1
            full = f"{layer_qname}.{sub_name}"
            emit_full = full

            recipe_key = profile.live_to_recipe_name(full)
            recipe_fmt = assignment.get(recipe_key)
            fmt = _canonical_export_format(recipe_fmt) if recipe_fmt is not None else None
            source_weight_key = f"{full}.weight"
            source_weight_dtype = source_dtype_by_name.get(source_weight_key)
            source_is_fp8_scaled = (
                source_weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and source_weight_key in fp8_scale_inv_map
            )
            if (source_is_fp8_scaled
                    and (fmt is None
                         or fmt == "BF16"
                         or recipe_key in bf16_passthrough)):
                fmt = "FP8_SOURCE"
            if fmt is None:
                # No assignment -> source-dtype passthrough.
                if not mod.weight.is_meta:
                    out[f"{emit_full}.weight"], label = _passthrough_tensor(
                        source_weight_key, mod.weight, source_dtype_by_name)
                    if mod.bias is not None and not mod.bias.is_meta:
                        out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                            f"{full}.bias", mod.bias, source_dtype_by_name)
                    hist[("linear", label)] += 1
                    covered.add(full)
                continue

            if fmt == "BF16" or recipe_key in bf16_passthrough:
                out[f"{emit_full}.weight"], label = _passthrough_tensor(
                    source_weight_key, mod.weight, source_dtype_by_name)
                if mod.bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", label)] += 1
                covered.add(full)
                continue

            if fmt == "FP8_SOURCE":
                # Passthrough: copy source `.weight` (fp8_e4m3fn) and
                # `.weight_scale_inv` (fp32, 128×128 block) verbatim.
                # The live model holds a BF16 dequant of the source
                # tensor — skip it and go back to the safetensors.
                scale_entry = fp8_source_map.get(full)
                weight_ckpt_key = weight_ckpt.get(f"{full}.weight")
                weight_shard_path = weight_shard.get(f"{full}.weight")
                if (scale_entry is None or weight_ckpt_key is None
                        or weight_shard_path is None):
                    raise RuntimeError(
                        f"[export-stream] FP8_SOURCE assigned to {full} "
                        f"but source is missing `.weight_scale_inv` "
                        f"(scale={scale_entry}, weight_shard="
                        f"{weight_shard_path}). The allocator's "
                        f"passthrough-integrity filter should have "
                        f"prevented this — source manifest is out of "
                        f"sync with the actual checkpoint.")
                scale_shard, scale_ckpt_key = scale_entry
                from safetensors import safe_open
                with safe_open(weight_shard_path, framework="pt") as sf:
                    w_fp8 = sf.get_tensor(weight_ckpt_key)
                    # Common case: scale lives in the same shard. Avoid
                    # a second `safe_open` when we can satisfy both
                    # reads from one file handle.
                    if scale_shard == weight_shard_path:
                        w_scale = sf.get_tensor(scale_ckpt_key)
                    else:
                        w_scale = None
                if w_scale is None:
                    with safe_open(scale_shard, framework="pt") as sf:
                        w_scale = sf.get_tensor(scale_ckpt_key)
                # Sanity check: source dtype must be fp8_e4m3fn; scale
                # must be fp32. Any deviation means the FP8_SOURCE
                # format is being misapplied.
                if w_fp8.dtype != torch.float8_e4m3fn:
                    raise RuntimeError(
                        f"[export-stream] FP8_SOURCE: expected "
                        f"fp8_e4m3fn at {weight_ckpt_key}, got "
                        f"{w_fp8.dtype}")
                out[f"{emit_full}.weight"] = w_fp8.cpu().contiguous()
                out[f"{emit_full}.weight_scale"] = w_scale.to(
                    torch.float32).cpu().contiguous()
                if mod.bias is not None and not mod.bias.is_meta:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", "FP8_SOURCE")] += 1
                covered.add(full)
                continue

            override = joint_globals.get(recipe_key) if fmt == "NVFP4" else None
            cached_compressed = _pack_production_cached_2d(
                recipe_key,
                recipe_fmt if recipe_fmt is not None else fmt,
                nvfp4_global_real_override=override,
                device=device,
            )
            if cached_compressed is not None:
                for suffix, t in cached_compressed.items():
                    out[f"{emit_full}.{suffix}"] = t.cpu()
                if mod.bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{full}.bias", mod.bias, source_dtype_by_name)
                hist[("linear", f"{fmt}_PRODUCTION_CACHE")] += 1
                covered.add(full)
                continue

            if fmt in MXFP8_EXPLICIT_FORMATS and mod.weight.dim() == 2:
                shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
                grouped_linears[(fmt, shape)].append((full, emit_full, recipe_key, mod))
                continue

            # v23: route NVFP4 same-shape Linears through the batched
            # GPTQ + scale_sweep path when env-gated and act-aware.
            if (_batched_nvfp4_enabled
                    and fmt == "NVFP4"
                    and mod.weight.dim() == 2):
                shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
                grouped_nvfp4_batched[shape].append(
                    (full, emit_full, recipe_key, mod))
                continue

            # #12 Block-output match: when enabled AND this is a
            # standard "block" Linear (q/k/v/o or gate/up/down) on
            # NVFP4, defer the final pack so we can refine its
            # dequantized weight using block-level output MSE before
            # freezing it into FP4 codes. The compute_dict + post-pack
            # state is saved into _BLOCK_COMPUTE_PENDING; the post-loop
            # phase invokes refine_block_scales then _finalize_compute_only.
            sub_leaf = sub_name.rsplit(".", 1)[-1] if sub_name else ""
            is_block_linear = (
                fmt == "NVFP4"
                and os.environ.get("PRISMAQUANT_BLOCK_OUTPUT_MATCH", "1") != "0"
                and sub_leaf in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                    "gate_proj", "up_proj", "down_proj",
                )
            )
            if is_block_linear:
                compute_dict = _quantize_2d(
                    mod.weight.detach().float(), fmt,
                    nvfp4_global_real_override=override,
                    linear_name=recipe_key,
                    compute_only=True,
                )
                _BLOCK_COMPUTE_PENDING.append({
                    "full": full, "emit_full": emit_full,
                    "sub_name": sub_name, "sub_leaf": sub_leaf, "mod": mod,
                    "compute_dict": compute_dict, "fmt": fmt,
                })
                continue  # skip immediate emit; finalized post-loop

            compressed = _quantize_2d(
                mod.weight.detach().float(), fmt,
                nvfp4_global_real_override=override,
                linear_name=recipe_key,
            )
            for suffix, t in compressed.items():
                out[f"{emit_full}.{suffix}"] = t.cpu()
            if mod.bias is not None:
                out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                    f"{full}.bias", mod.bias, source_dtype_by_name)
            hist[("linear", fmt)] += 1
            covered.add(full)

        # RTN-only formats can be emitted in same-shape batches. MiniMax has
        # hundreds of expert Linears per layer with identical shapes; doing
        # those one at a time keeps the export CPU/Python-bound even though the
        # math itself is vectorized.
        export_dev = torch.device(device)
        for (fmt, shape), items in grouped_linears.items():
            chunk_len = _export_vector_chunk_len(shape, len(items), export_dev)
            for start in range(0, len(items), chunk_len):
                chunk = items[start:start + chunk_len]
                stacked = torch.stack(
                    [mod.weight.detach().to(torch.float32) for _, _, _, mod in chunk],
                    dim=0,
                )
                compressed_batch = _quantize_2d_group_same_shape(stacked, fmt)
                del stacked
                for i, (full, emit_full, _recipe_key, mod) in enumerate(chunk):
                    for suffix, tensor in compressed_batch.items():
                        piece = tensor[i]
                        if suffix == "weight_global_scale":
                            piece = piece.reshape(1)
                        out[f"{emit_full}.{suffix}"] = piece.cpu()
                    if mod.bias is not None:
                        out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                            f"{full}.bias", mod.bias, source_dtype_by_name)
                    hist[("linear", fmt)] += 1
                    covered.add(full)
                del compressed_batch

        # v23: batched NVFP4 emission for same-shape groups when
        # _batched_nvfp4_enabled. Mirrors the INT/MXFP8_E4M3 grouped path
        # above but routes through the activation-aware batched path.
        if grouped_nvfp4_batched:
            export_dev = torch.device(device)
            for shape, items in grouped_nvfp4_batched.items():
                # Re-use the same E-chunk sizing as the INT/MXFP8_E4M3 path
                # so memory peaks stay bounded.
                chunk_len = _export_vector_chunk_len(
                    shape, len(items), export_dev)
                for start in range(0, len(items), chunk_len):
                    chunk = items[start:start + chunk_len]
                    compressed_per_linear = _quantize_2d_nvfp4_group_batched(
                        chunk, joint_globals, export_dev,
                        expert_chunk=chunk_len,
                    )
                    for (full, emit_full, _recipe_key, mod), compressed in zip(
                        chunk, compressed_per_linear,
                    ):
                        for suffix, t in compressed.items():
                            out[f"{emit_full}.{suffix}"] = t.cpu()
                        if mod.bias is not None:
                            out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                                f"{full}.bias", mod.bias, source_dtype_by_name)
                        hist[("linear", "NVFP4")] += 1
                        covered.add(full)

        # 3c'. Block-output match (#12). When PRISMAQUANT_BLOCK_OUTPUT_MATCH=1
        # the per-Linear loop above deferred packing for standard block
        # Linears (q/k/v/o, gate/up/down). Now run greedy refinement of
        # per-Linear scale perturbations against an FP16 reference forward,
        # then finalize the pack. Skipped if no compute-only entries
        # accumulated (env flag off, or no eligible Linears in this layer).
        if _BLOCK_COMPUTE_PENDING:
            try:
                from .block_output_match import (
                    block_output_mse,
                    make_attention_block_spec, make_mlp_block_spec,
                    refine_block_scales,
                )
                # Group pending entries by sub_leaf so we can index
                # them when applying refined scales. Also recover
                # the FP16 reference weights from _FP16_BLOCK_SNAPSHOTS.
                pending_by_sub = {p["sub_leaf"]: p
                                  for p in _BLOCK_COMPUTE_PENDING}

                # Use a small calibration input drawn from the cached
                # activation of q_proj (its input == post-norm of the
                # residual stream, which is the natural attn-block
                # input). For MLP block, gate_proj input is the same
                # post-norm residual after attention. If activations
                # aren't cached for this layer, skip refinement —
                # there's no reference signal.
                cal_input_attn = None
                cal_input_mlp = None
                if _CACHED_ACTIVATIONS is not None:
                    # cached keys are recipe_keys; pull from any
                    # block-Linear that's pending so naming variation
                    # across profiles still works.
                    for p in _BLOCK_COMPUTE_PENDING:
                        if p["sub_leaf"] in ("q_proj",) and cal_input_attn is None:
                            cal_input_attn = _CACHED_ACTIVATIONS.get(
                                profile.live_to_recipe_name(p["full"]))
                        if p["sub_leaf"] in ("gate_proj",) and cal_input_mlp is None:
                            cal_input_mlp = _CACHED_ACTIVATIONS.get(
                                profile.live_to_recipe_name(p["full"]))

                # Run refinement for each block we have a cal input for.
                # Candidates are simple multiplicative perturbations of
                # the current dequantized weight; refine_block_scales
                # picks the per-Linear scale that minimizes block MSE.
                cands = [torch.tensor(s) for s in (0.95, 1.0, 1.05)]

                block_logs: list[str] = []

                def _apply_refined_scales(label: str, spec_factory, cal_input):
                    if cal_input is None:
                        block_logs.append(f"{label}=no_cal")
                        return
                    ref_spec = spec_factory(layer_mod, layer_qname)
                    if ref_spec is None:
                        block_logs.append(f"{label}=no_spec")
                        return
                    # Cap the cal_input to a small batch to keep refinement fast.
                    ci = cal_input.to(layer_mod.input_layernorm.weight.device
                                      if hasattr(layer_mod, "input_layernorm")
                                      else next(iter(layer_mod.parameters())).device)
                    if ci.dim() == 2:
                        ci = ci[:32]
                    elif ci.dim() == 3:
                        ci = ci[:8]
                    run_dtype = next(
                        (p["mod"].weight.dtype for p in _BLOCK_COMPUTE_PENDING
                         if p["mod"].weight.dtype.is_floating_point),
                        torch.float32,
                    )
                    ci_run = ci.to(dtype=run_dtype)
                    # Full-precision reference first, while the live layer
                    # still holds original weights. Earlier code built the
                    # reference and candidates from the same live weights,
                    # making scale=1.0 perfect and the pass a silent no-op.
                    with torch.no_grad():
                        ref = ref_spec.forward_fn(ci_run).float().clone()

                    touched: list[dict] = []
                    for ln in ref_spec.linears:
                        p = pending_by_sub.get(ln)
                        if p is None:
                            continue
                        mod = p["mod"]
                        touched.append(p)
                        q_weight = p["compute_dict"]["_w_dq"].to(
                            device=mod.weight.device, dtype=mod.weight.dtype)
                        mod.weight.data.copy_(q_weight)

                    if not touched:
                        block_logs.append(f"{label}=no_pending")
                        return

                    try:
                        spec = spec_factory(layer_mod, layer_qname)
                        if spec is None:
                            block_logs.append(f"{label}=lost_spec")
                            return
                        candidates = {
                            ln: cands for ln in spec.linears
                            if ln in pending_by_sub
                        }
                        before = block_output_mse(spec, ci_run, ref)
                        final = refine_block_scales(
                            spec, ci_run, ref, candidates, max_passes=2)
                        n_changed = 0
                        n_eval = 0
                        for ln in spec.linears:
                            p = pending_by_sub.get(ln)
                            if p is None:
                                continue
                            n_eval += len(cands) * 2
                            s = float(spec.scale_getter(ln))
                            if abs(s - 1.0) < 1e-8:
                                continue
                            p["compute_dict"]["_w_dq"] = (
                                p["compute_dict"]["_w_dq"] * s)
                            n_changed += 1
                        block_logs.append(
                            f"{label}=spec evals={n_eval} "
                            f"changed={n_changed} "
                            f"mse={before:.3e}->{final:.3e}")
                    finally:
                        for p in touched:
                            snap = _FP16_BLOCK_SNAPSHOTS.get(p["sub_name"])
                            if snap is not None:
                                p["mod"].weight.data.copy_(
                                    snap.to(device=p["mod"].weight.device,
                                            dtype=p["mod"].weight.dtype))

                _apply_refined_scales(
                    "attn", make_attention_block_spec, cal_input_attn)
                _apply_refined_scales(
                    "mlp", make_mlp_block_spec, cal_input_mlp)
                print(
                    f"[block-output-match] {layer_qname}: "
                    f"pending={len(_BLOCK_COMPUTE_PENDING)} "
                    + " ".join(block_logs),
                    flush=True,
                )

            except Exception as e:
                print(f"[block-output-match] WARN refinement failed for "
                      f"{layer_qname}: {e}", flush=True)

            # Finalize the pack for every pending Linear (refined or not).
            for p in _BLOCK_COMPUTE_PENDING:
                compressed = _finalize_compute_only(p["compute_dict"])
                emit_full = p["emit_full"]
                for suffix, t in compressed.items():
                    out[f"{emit_full}.{suffix}"] = t.cpu()
                if p["mod"].bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{p['full']}.bias", p["mod"].bias,
                        source_dtype_by_name)
                hist[("linear", "NVFP4_block_match")] += 1
                covered.add(p["full"])

            del _BLOCK_COMPUTE_PENDING, _FP16_BLOCK_SNAPSHOTS

        # 3d. Emit packed MoE experts, scoped to this layer.
        packed_count = 0
        for sub_name, mod in layer_mod.named_modules():
            if not _is_packed_experts_module(mod, profile):
                continue
            packed_count += 1
            for pn in _packed_experts_param_names(mod, profile):
                experts_qname = (f"{layer_qname}.{sub_name}"
                                 if sub_name else layer_qname)
                full = f"{experts_qname}.{pn}"
                recipe_key = profile.live_to_recipe_name(full)
                fmt = assignment.get(recipe_key)
                if fmt is not None:
                    fmt = _canonical_export_format(fmt)
                if fmt is None:
                    unmapped_keys.append(full)
                    continue
                if fmt == "FP8_SOURCE":
                    raise NotImplementedError(
                        f"[export-stream] FP8_SOURCE not wired for "
                        f"packed-MoE tensors (at {full}). MiniMax-M2/M2.7 "
                        f"— the only natively-FP8 MoE today — uses "
                        f"per-expert `nn.Linear`s, so its experts go "
                        f"through the Linear emit path above, not here. "
                        f"If a new FP8-native MoE arch ships with a "
                        f"packed-expert live module, extend this branch "
                        f"to read per-expert `.weight` + "
                        f"`.weight_scale_inv` from source and emit the "
                        f"per-expert compressed-tensors pairs.")
                packed_param_src = getattr(mod, pn).detach()
                packed_param = packed_param_src.float()
                E, M, N = packed_param.shape
                proj_split = _split_packed_expert_tensor(
                    packed_param,
                    pn,
                    profile,
                )

                is_bf16 = fmt == "BF16" or full in bf16_passthrough
                disk_qname = profile.on_disk_expert_qname(experts_qname)
                should_split = profile.split_packed_experts_for_format(fmt)

                iter_experts = [(e, e) for e in range(E)]

                if not should_split:
                    out[f"{disk_qname}.{pn}"], label = _passthrough_tensor(
                        full, packed_param_src, source_dtype_by_name)
                    covered.add(full)
                    hist[("packed_moe", label if is_bf16 else fmt)] += 1
                    del packed_param, packed_param_src
                    continue

                # Per-expert joint global scale when NVFP4 splits gate+up.
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    for orig_e, _ in iter_experts:
                        cands = [
                            compute_nvfp4_global_real(sp[orig_e].float(),
                                                      group_size=16)
                            for _, sp in proj_split
                        ]
                        per_expert_joint[orig_e] = torch.stack(cands).max()

                for proj_name, sub_packed in proj_split:
                    for orig_e, new_e in iter_experts:
                        expert_2d = sub_packed[orig_e]
                        base = f"{disk_qname}.{new_e}.{proj_name}"
                        if is_bf16:
                            out[f"{base}.weight"], label = _passthrough_tensor(
                                full, expert_2d, source_dtype_by_name)
                        else:
                            compressed = _quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=per_expert_joint[orig_e],
                            )
                            for suffix, t in compressed.items():
                                out[f"{base}.{suffix}"] = t.cpu()
                covered.add(full)
                hist[("packed_moe_per_expert", label if is_bf16 else fmt)] += 1
                del packed_param, packed_param_src, proj_split

        # 3e. Remaining layer-scoped params (norms, conv1d, biases on
        # passthrough-only modules) and persistent buffers.
        for sub_name, param in layer_mod.named_parameters():
            full = f"{layer_qname}.{sub_name}"
            if full in out:
                continue
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if param.is_meta:
                continue
            out[full], label = _passthrough_tensor(
                full, param, source_dtype_by_name)
            hist[("layer_passthrough", label)] += 1
        for mod_name, mod in layer_mod.named_modules():
            non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
            for buf_name, buf in mod.named_buffers(recurse=False):
                if buf_name in non_persistent:
                    continue
                full_modpath = (f"{layer_qname}.{mod_name}"
                                if mod_name else layer_qname)
                full = f"{full_modpath}.{buf_name}"
                if full in out or buf.is_meta:
                    continue
                out[full], label = _passthrough_tensor(
                    full, buf, source_dtype_by_name)
                hist[("layer_buffer", label)] += 1

        # 3f. Unload.
        _unload(model, [f"{layers_prefix}{L}."])
        del tensors, resolver, joint_globals
        # Aggressive GPU cleanup — we've already `.cpu()`'d every
        # quantized output into `out`, so the per-layer GPU working
        # set (fp32 weight copies, grouped/packed intermediates) can
        # be released immediately. Keeps per-layer peak bounded.
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if L % 4 == 0:
            gc.collect()
        if L % 4 == 0 or L == num_layers - 1:
            elapsed = time.time() - layer_t0
            print(f"[export-stream] layer {L:02d}  linears={linear_count} "
                  f"packed={packed_count}  load={load_s:.2f}s  "
                  f"total={elapsed:.2f}s  out_keys={len(out)}", flush=True)
        # v25: save layer cache BEFORE tensor_sink consumes the dict.
        # Use a tmp + rename to keep the cache file atomic — a kill in
        # the middle of torch.save leaves a .tmp behind which we'll
        # ignore on the next run (skip and recompute the layer).
        cf = _layer_cache_file(L)
        if cf is not None and out:
            tmp = cf.with_suffix(".pt.tmp")
            torch.save(out, str(tmp))
            tmp.rename(cf)
        if tensor_sink is not None:
            tensor_sink(out)
            out = {}

    print(f"[export-stream] layer sweep: {time.time()-t_layers:.1f}s "
          f"(cache_hits={cache_hits}/{num_layers})",
          flush=True)

    if unmapped_keys:
        print(f"[export-stream] WARN {len(unmapped_keys)} unmapped assignment "
              f"keys — first 5: {unmapped_keys[:5]}", flush=True)

    return out, dict(hist)


def _materialize_tensors_inmemory(
    model: nn.Module,
    assignment: dict[str, str],
    *,
    bf16_passthrough: set[str],
    profile: "ModelProfile | None" = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Whole-model quantizer used for small auxiliary modules (notably the
    MTP wrapper) that fit in RAM. The main decoder export path uses the
    streaming materializer above; this helper exists because MTP is
    built standalone from safetensors and its root module is orders of
    magnitude smaller than the decoder body."""
    from .model_profiles import DefaultProfile
    profile = profile or DefaultProfile()
    remap = profile.live_to_recipe_name

    out: dict[str, torch.Tensor] = {}
    hist = Counter()
    covered: set[str] = set()

    # Pre-pass: joint NVFP4 global_scale per fused-sibling group so
    # q/k/v (or gate/up, etc.) share one weight_global_scale slot.
    nvfp4_joint_global = _compute_nvfp4_joint_global(
        model,
        assignment,
        profile=profile,
    )

    for qname, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        fmt_key = remap(qname)
        fmt = assignment.get(fmt_key)
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt is None:
            continue
        if fmt == "BF16" or fmt_key in bf16_passthrough:
            out[f"{qname}.weight"], label = _passthrough_tensor(
                f"{qname}.weight", mod.weight)
            if mod.bias is not None:
                out[f"{qname}.bias"], _ = _passthrough_tensor(
                    f"{qname}.bias", mod.bias)
            covered.add(qname)
            hist[("linear", label)] += 1
            continue
        joint = nvfp4_joint_global.get(fmt_key) if fmt == "NVFP4" else None
        compressed = _quantize_2d(
            mod.weight.detach().float(), fmt,
            nvfp4_global_real_override=joint,
            linear_name=fmt_key,
        )
        for suffix, tensor in compressed.items():
            out[f"{qname}.{suffix}"] = tensor.cpu()
        if mod.bias is not None:
            out[f"{qname}.bias"], _ = _passthrough_tensor(
                f"{qname}.bias", mod.bias)
        covered.add(qname)
        hist[("linear", fmt)] += 1

    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        for pn in _packed_experts_param_names(mod, profile):
            full_name = f"{qname}.{pn}" if qname else pn
            recipe_key = remap(full_name)
            fmt = assignment.get(recipe_key)
            if fmt is not None:
                fmt = _canonical_export_format(fmt)
            if fmt is None:
                continue
            packed_param_src = getattr(mod, pn).detach()
            packed_param = packed_param_src.float()
            E, M, N = packed_param.shape
            proj_split = _split_packed_expert_tensor(
                packed_param,
                pn,
                profile,
            )

            is_bf16 = fmt == "BF16" or full_name in bf16_passthrough
            disk_qname = profile.on_disk_expert_qname(qname)
            should_split = profile.split_packed_experts_for_format(fmt)

            if not should_split:
                out[f"{disk_qname}.{pn}"], label = _passthrough_tensor(
                    full_name, packed_param_src)
                covered.add(full_name)
                hist[("packed_moe", label if is_bf16 else fmt)] += 1
                continue

            per_expert_joint: list[torch.Tensor | None] = [None] * E
            if fmt == "NVFP4" and len(proj_split) > 1:
                for e in range(E):
                    candidates = [
                        compute_nvfp4_global_real(sub_packed[e].float(),
                                                  group_size=16)
                        for _, sub_packed in proj_split
                    ]
                    per_expert_joint[e] = torch.stack(candidates).max()

            for proj_name, sub_packed in proj_split:
                E_p, Mp, Np = sub_packed.shape
                for e in range(E_p):
                    expert_2d = sub_packed[e]
                    base = f"{disk_qname}.{e}.{proj_name}"
                    if is_bf16:
                        out[f"{base}.weight"], label = _passthrough_tensor(
                            full_name, expert_2d)
                    else:
                        compressed = _quantize_2d(
                            expert_2d, fmt,
                            nvfp4_global_real_override=per_expert_joint[e],
                        )
                        for suffix, tensor in compressed.items():
                            out[f"{base}.{suffix}"] = tensor.cpu()
            covered.add(full_name)
            hist[("packed_moe_per_expert", label if is_bf16 else fmt)] += 1

    for name, p in model.named_parameters():
        if any(name.startswith(c + ".") or name == c for c in covered):
            continue
        if name in out:
            continue
        out[name], label = _passthrough_tensor(name, p)
        hist[("passthrough", label)] += 1

    for mod_name, mod in model.named_modules():
        non_persistent = getattr(mod, "_non_persistent_buffers_set", set())
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            full = f"{mod_name}.{buf_name}" if mod_name else buf_name
            if any(full.startswith(c + ".") or full == c for c in covered):
                continue
            if full in out:
                continue
            out[full], label = _passthrough_tensor(full, buf)
            hist[("passthrough_buffer", label)] += 1

    return out, dict(hist)


# ---------------------------------------------------------------------------
# Compressed-tensors quantization_config
# ---------------------------------------------------------------------------
NVFP4_SCHEME = {
    "format": "nvfp4-pack-quantized",
    "weights": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 4, "type": "float", "strategy": "tensor_group",
        "group_size": 16, "symmetric": True,
        "dynamic": "local", "observer": "static_minmax",
        "scale_dtype": "torch.float8_e4m3fn",
        "zp_dtype": "torch.float8_e4m3fn",
    },
}
MXFP8_SCHEME = {
    "format": "mxfp8-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": True,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
    },
}
MXFP4_SCHEME = {
    "format": "mxfp4-pack-quantized",
    "weights": {
        "num_bits": 4, "type": "float", "strategy": "group",
        "group_size": 32,
        "symmetric": True, "dynamic": False,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
        "observer": "memoryless_minmax",
    },
}
# Source-FP8 passthrough. Emitted for Linears whose source checkpoint
# already stores `.weight` as fp8_e4m3fn + `.weight_scale_inv` fp32 at
# 128×128 block granularity (MiniMax-M2/M2.7, DeepSeek V3, several
# NVIDIA FP8 releases). vLLM's compressed-tensors dispatcher routes
# this scheme to `_is_fp8_w8a8` which accepts BLOCK-strategy symmetric
# static FP8 weights with dynamic FP8 activations — matching the
# native MiniMax inference configuration.
#
# Compressed-tensors' `weight_scale` (forward-direction dequant scale:
# `w_bf16 = w_fp8 * weight_scale`) is semantically identical to
# MiniMax's `weight_scale_inv`; the tensor bytes are copied verbatim
# and only the suffix is renamed on export. No _quantize_2d pass runs.
FP8_SOURCE_SCHEME = {
    "format": "float-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "block",
        "block_structure": [128, 128],
        "symmetric": True, "dynamic": False,
        "observer": "memoryless_minmax",
    },
    # Per-tensor dynamic activation scaling (NOT per-token). vLLM's
    # FP8 MoE path `fp8_w8a8_moe_quant_config` asserts
    # `not per_act_token_quant` whenever weight `block_structure` is
    # set — block-scaled weight + per-token act isn't wired. This
    # matches MiniMax's native-serving `activation_scheme: dynamic`,
    # which is per-tensor dynamic in DeepSeek / MiniMax conventions.
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "tensor",
        "symmetric": True, "dynamic": True,
        "observer": "memoryless_minmax",
    },
}
FP8_E4M3_SCHEME = {
    "format": "float-quantized",
    "weights": {
        "num_bits": 8, "type": "float", "strategy": "channel",
        "symmetric": True, "dynamic": False,
        "observer": "memoryless_minmax",
    },
    "input_activations": {
        "num_bits": 8, "type": "float", "strategy": "token",
        "symmetric": True, "dynamic": True,
    },
}


def _pin_regex_to_layer(body: str, layer_idx: str | None) -> str | None:
    if layer_idx is None:
        return None
    return re.sub(
        r"layers\[\.\]\[0-9\]\+",
        f"layers[.]{layer_idx}",
        str(body),
        count=1,
    )


def _constrain_per_expert_projection_regex(
    body: str,
    proj_options: str,
) -> str:
    """Constrain a profile per-expert regex to selected projections.

    Profile specs own projection names.  Older specs spell Qwen-style
    projections as ``(gate|up|down)_proj``; newer/custom specs may provide
    complete alternatives like ``(w1_proj|w3_proj|w2)``.  This helper rewrites
    the final projection segment after ``experts.<id>.`` without hardcoding
    either naming family into the export path.
    """
    replacement = f"({proj_options})"
    for legacy in (
        "(gate|up|down)_proj",
        "(gate_proj|up_proj|down_proj)",
    ):
        if legacy in body:
            return body.replace(legacy, replacement)

    for pattern in (
        r"(?P<prefix>experts\[\.\]\[0-9\]\+\[\.\])(?P<proj>.+?)(?P<suffix>\$)$",
        r"(?P<prefix>experts\\\.\[0-9\]\+\\\.)(?P<proj>.+?)(?P<suffix>\$)$",
        r"(?P<prefix>experts\.\[0-9\]\+\.)(?P<proj>.+?)(?P<suffix>\$)$",
    ):
        constrained, count = re.subn(
            pattern,
            rf"\g<prefix>{replacement}\g<suffix>",
            body,
            count=1,
        )
        if count:
            return constrained
    return body


def _bf16_packed_expert_ignore_regex(
        recipe_key: str,
        profile,
) -> list[str]:
    """If `recipe_key` names a BF16 packed-MoE tensor, return regex
    strings for the corresponding per-expert Linear qnames at
    scheme-dispatch time.

    The packed-parameter to per-projection decomposition comes from the
    active model profile/spec, so export metadata stays aligned with model
    structure config instead of baking Qwen-specific names into this path.
    """
    import re as _re

    # Does this recipe key name a packed-expert tensor?
    if ".experts." not in recipe_key:
        return []
    pn = recipe_key.rsplit(".", 1)[-1]
    if pn not in _packed_expert_param_name_set(profile):
        return []

    # Convert the recipe parent prefix to a live-model prefix by
    # asking the profile. `profile.live_to_recipe_name` is the
    # opposite direction, so we'd need its inverse — instead emit a
    # regex loose enough to match both live forms on both sides of
    # the remap (text-only-style `...layers.X.experts.Y.*` and
    # multimodal `language_model.model.layers.X.moe.experts.Y.*`).
    # The profile's `per_expert_moe_regex` already encodes the live
    # form; we narrow it to this specific layer by pinning the layer
    # index.
    # Distinguish MTP (`mtp.layers.N.*`) from body (`model.layers.N.*`)
    # — both can have layer index N but they're DIFFERENT layers, and
    # emitting a body-prefixed regex for a BF16 MTP assignment
    # accidentally ignores the body's NVFP4 experts at that layer idx.
    is_mtp = recipe_key.startswith("mtp.")
    layer_idx = None
    lm = _re.search(r"\.layers\.(\d+)\.", recipe_key)
    if lm:
        layer_idx = lm.group(1)
    # vLLM's should_ignore_layer probes the canonical gate_proj/up_proj/
    # down_proj names; emit the ignore regex with those (not the on-disk
    # w1/w3/w2), else BF16 experts are not recognized as ignored and fall
    # through to a quantized catch-all scheme.
    projections = _vllm_moe_scheme_projection_names(profile, pn)
    proj_options = "|".join(_re.escape(proj) for proj in projections)

    # Use the profile's own regex as the base; swap its `(gate|up|down)_proj`
    # group with the exact projections we emit, and constrain to this
    # layer.
    # MTP layers live under a `mtp.layers.N.*` prefix — separate
    # layer-index namespace from the body. Use the profile's dedicated
    # per_expert_mtp_regex (if any) instead of the body one.
    if is_mtp:
        mtp_base = profile.per_expert_mtp_regex() if profile else None
        if mtp_base and mtp_base.startswith("re:"):
            body = mtp_base[len("re:"):]
            pinned = _pin_regex_to_layer(body, layer_idx)
            if pinned is None:
                return []
            pinned = _constrain_per_expert_projection_regex(pinned, proj_options)
            return [f"re:{pinned}"]
        # Fallback: emit an `mtp.layers.N.*` regex directly.
        if layer_idx is None:
            return []
        return [
            rf"re:^mtp[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        ]

    base = profile.per_expert_moe_regex() if profile else None
    if not base or not base.startswith("re:"):
        # No profile regex — emit a conservative default spanning
        # both common live-module conventions.
        patterns = []
        if layer_idx is None:
            return patterns
        # Try the multimodal (Gemma / Qwen3.6) layout first.
        patterns.append(
            rf"re:^language_model[.]model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        # And the text-only / dense layout.
        patterns.append(
            rf"re:^model[.]layers[.]{layer_idx}[.]"
            rf"(?:moe[.])?experts[.][0-9]+[.]({proj_options})$"
        )
        return patterns

    # Profile-provided regex. Strip the `re:` prefix, pin to this
    # layer index, constrain to the emitted projections.
    body = base[len("re:"):]
    pinned = _pin_regex_to_layer(body, layer_idx)
    if pinned is None:
        return []
    pinned = _constrain_per_expert_projection_regex(pinned, proj_options)
    return [f"re:{pinned}"]


FORMAT_SCHEME = {
    "NVFP4": NVFP4_SCHEME,
    "MXFP4": MXFP4_SCHEME,
    "MXFP8": MXFP8_SCHEME,
    "MXFP8_E4M3": MXFP8_SCHEME,
    "MXFP8_E5M2": MXFP8_SCHEME,
    "FP8_E4M3": FP8_E4M3_SCHEME,
    "FP8_SOURCE": FP8_SOURCE_SCHEME,
}


def _fused_modules_mapping_for_profile(profile) -> dict[str, tuple[str, ...]]:
    """Return fused-module leaf mapping for target emission.

    The returned shape mirrors vLLM's ``packed_modules_mapping``:
    ``{"qkv_proj": ("q_proj", ...)}``.
    """
    if profile is None:
        return {}

    getter = getattr(profile, "fused_sibling_leaf_mapping", None)
    if callable(getter):
        try:
            mapping = getter()
        except Exception:
            mapping = None
        if mapping:
            return {
                str(fused): tuple(str(sibling) for sibling in siblings)
                for fused, siblings in mapping.items()
            }

    try:
        from .model_profiles.vllm_registry import (
            packed_modules_mapping_from_class,
            vllm_class_for_architecture,
        )
        vllm_cls = vllm_class_for_architecture(
            profile.vllm_architecture_class() or ""
        )
        packed_mapping = packed_modules_mapping_from_class(vllm_cls)
        if packed_mapping:
            return {
                str(fused): tuple(str(sibling) for sibling in siblings)
                for fused, siblings in packed_mapping.items()
            }
    except Exception:
        pass

    spec_getter = getattr(profile, "structure_spec", None)
    spec = spec_getter() if callable(spec_getter) else None
    if spec is None:
        return {}

    mapping: dict[str, tuple[str, ...]] = {}
    for group in getattr(spec, "fused_groups", ()):
        target_parent, target_leaf = _suffix_parent_leaf(group.target_suffix)
        member_leafs: list[str] = []
        valid = True
        for member in group.member_suffixes:
            member_parent, member_leaf = _suffix_parent_leaf(member)
            if target_parent and member_parent and member_parent != target_parent:
                valid = False
                break
            member_leafs.append(member_leaf)
        if valid and len(member_leafs) > 1:
            mapping[target_leaf] = tuple(member_leafs)
    return mapping


def _suffix_parent_leaf(suffix: str) -> tuple[str, str]:
    if "." not in suffix:
        return "", str(suffix)
    parent, leaf = str(suffix).rsplit(".", 1)
    return parent, leaf


def build_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
    *,
    profile: "ModelProfile | None" = None,
) -> dict:
    """Emit a `quantization_config` dict with explicit per-name targets
    grouped by format. Targets and ignore are remapped to vLLM's
    internal naming via the supplied `profile` so `find_matched_target`
    matches.

    `extra_ignore` is for module qnames that aren't in the recipe at
    all but should be excluded from any catch-all group (e.g. routers).
    The catch-all default group is the format with the most non-BF16
    members (typically NVFP4).

    `profile` controls the architecture-specific bits: name remap,
    per-expert MoE / MTP regexes. Defaults to `DefaultProfile()` (plain
    names, no catch-all regexes) when omitted.
    """
    from .model_profiles import DefaultProfile
    profile = profile or DefaultProfile()

    by_fmt: dict[str, list[str]] = {}
    ignore: list[str] = []
    for n in bf16_passthrough:
        ignore.append(profile.to_vllm_internal_name(n))
    for n in extra_ignore:
        ignore.append(profile.to_vllm_internal_name(n))
    for name, fmt in sorted(assignment.items()):
        fmt = _canonical_export_format(fmt)
        vllm_name = profile.to_vllm_internal_name(name)
        if fmt == "BF16":
            ignore.append(vllm_name)
            # Packed MoE tensors in BF16 are emitted as per-expert
            # per-projection splits (not as the 3D packed tensor). vLLM
            # scheme-dispatches against the per-expert Linear qnames
            # (e.g. `...experts.0.gate_proj`), not the packed parent —
            # so the `ignore` for a BF16 packed-expert recipe entry
            # must cover every per-expert per-projection for that layer.
            # We emit a narrow regex per layer rather than enumerating
            # hundreds of explicit names.
            regex_list = _bf16_packed_expert_ignore_regex(name, profile)
            for r in regex_list:
                ignore.append(r)
            continue
        by_fmt.setdefault(fmt, []).append(vllm_name)

    # Fill in fused-sibling members that exist in the serving model
    # but weren't in the probe assignment — e.g. Gemma 4's
    # full_attention layers have no v_proj on disk, so the probe
    # never saw it, but vLLM's QKVParallelLinear still instantiates
    # a v_proj sub-module that gets k_proj's weights at load. Scheme
    # dispatch requires all fused siblings to have consistent
    # scheme. We infer missing siblings by walking the assignment for
    # fused groups that landed in `ignore` and filling in every
    # sibling from vLLM's `packed_modules_mapping` or the declarative
    # model-structure spec — including ones we never saw weights for.
    packed_mapping = _fused_modules_mapping_for_profile(profile)
    if packed_mapping:
        # Reverse map: sibling-leaf-name -> fused-name (e.g.
        # q_proj -> qkv_proj).
        leaf_to_fused: dict[str, str] = {}
        for fused_name, siblings in packed_mapping.items():
            for s in siblings:
                leaf_to_fused[s] = fused_name
        # Set of leaf suffixes we should have. We'll only fill in
        # siblings under names that match known fused patterns.
        bf16_name_set = set(ignore)
        for name, fmt in list(assignment.items()):
            fmt = _canonical_export_format(fmt)
            if fmt != "BF16":
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in leaf_to_fused:
                continue
            fused = leaf_to_fused[leaf]
            expected_siblings = packed_mapping[fused]
            parent = name[: -(len(leaf))]
            for sib in expected_siblings:
                full = parent + sib
                vllm_name = profile.to_vllm_internal_name(full)
                if vllm_name not in bf16_name_set:
                    ignore.append(vllm_name)
                    bf16_name_set.add(vllm_name)

    # Packed-3D MoE target emission. Serving runtimes such as vLLM load
    # packed expert tensors through one FusedMoE module at qname
    # `<block>.experts`. Scheme dispatch
    # (`get_moe_method`) probes targets via THREE synthetic layer
    # names built off the FusedMoE prefix:
    #   `<block>.experts.0.gate_proj`
    #   `<block>.experts.0.up_proj`
    #   `<block>.experts.0.down_proj`
    # — this is the "Linear-before-fusion" naming convention, not the
    # packed-tensor qnames (for example `experts.gate_up_proj`) we emit
    # in the safetensors. Without matching targets on that
    # per-expert form, no scheme binds to FusedMoE, `w2_input_global_scale`
    # etc. are never registered, and load_weights KeyErrors on our
    # per-expert input scale keys.
    #
    # Fix: for each packed recipe entry under `by_fmt` or `ignore`,
    # replace it with a per-expert regex pinned to that layer index so
    # vLLM's scheme dispatch gets a match on expert 0's projection
    # names. One regex per layer covers all (expert, projection)
    # combinations. The profile's packed-expert format groups ensure the
    # projections of a single FusedMoE share a scheme — we crash loud on
    # mismatch.
    packed_fused_states: dict[str, set[str]] = {}
    packed_fused_projections: dict[str, list[str]] = {}

    def _packed_expert_vllm_match(vname: str) -> tuple[str, str] | None:
        if vname.startswith("re:"):
            return None
        if "." not in vname:
            return None
        fused_qname, leaf = vname.rsplit(".", 1)
        if not fused_qname.endswith(".experts"):
            return None
        if leaf not in _packed_expert_param_name_set(profile):
            return None
        return fused_qname, leaf

    def _packed_format_group_members(fused_qname: str, leaf: str) -> tuple[str, ...]:
        group_getter = getattr(profile, "packed_expert_format_group", None)
        if callable(group_getter):
            group_key = group_getter(f"{fused_qname}.{leaf}")
            marker = "::__packed_format__:"
            if group_key and marker in group_key:
                return tuple(
                    member for member in group_key.split(marker, 1)[1].split(",")
                    if member
                )
        return (leaf,)

    def _record_packed_fused_state(fused_qname: str, leaf: str, state: str) -> None:
        packed_fused_states.setdefault(fused_qname, set()).add(state)
        seen = set(packed_fused_projections.setdefault(fused_qname, []))
        for member in _packed_format_group_members(fused_qname, leaf):
            # vLLM scheme dispatch probes canonical gate_proj/up_proj/down_proj
            # names, not the on-disk projection names (w1/w3/w2 on LFM2.5).
            for projection in _vllm_moe_scheme_projection_names(profile, member):
                if projection in seen:
                    continue
                packed_fused_projections[fused_qname].append(projection)
                seen.add(projection)

    for fmt, names in list(by_fmt.items()):
        kept = []
        for vname in names:
            packed = _packed_expert_vllm_match(vname)
            if packed is not None:
                fused_qname, leaf = packed
                _record_packed_fused_state(fused_qname, leaf, fmt)
            else:
                kept.append(vname)
        by_fmt[fmt] = kept
    ignore_kept = []
    for vname in ignore:
        # Preserve regex-prefixed ignores (our
        # _bf16_packed_expert_ignore_regex emits those); they already
        # cover the per-expert forms vLLM dispatches on.
        if vname.startswith("re:"):
            ignore_kept.append(vname)
            continue
        packed = _packed_expert_vllm_match(vname)
        if packed is not None:
            fused_qname, leaf = packed
            _record_packed_fused_state(fused_qname, leaf, "IGNORE")
        else:
            ignore_kept.append(vname)
    ignore = ignore_kept

    def _per_expert_regex_for(
        fused_qname: str,
        projections: list[str],
    ) -> str:
        """Regex matching any `<fused_qname>.<eid>.<proj>` where
        proj is one of the configured per-expert projections. Uses `[.]`
        for literal-dot escapes, matching the rest of this file's regex
        target style."""
        escaped = fused_qname.replace(".", "[.]")
        if not projections:
            projections = list(_all_packed_expert_projection_names(profile))
        proj_options = "|".join(re.escape(proj) for proj in projections)
        return (
            f"re:^{escaped}[.][0-9]+[.]({proj_options})$"
        )

    for fused_qname, states in packed_fused_states.items():
        if len(states) > 1:
            raise RuntimeError(
                f"[export-stream] FusedMoE at {fused_qname!r} has mixed "
                f"states across packed expert projections {states}; "
                f"the allocator's packed-expert format group should have "
                f"forced one scheme before this point."
            )
        state = next(iter(states))
        regex = _per_expert_regex_for(
            fused_qname,
            packed_fused_projections.get(fused_qname, []),
        )
        if state == "IGNORE":
            ignore.append(regex)
        else:
            by_fmt.setdefault(state, []).append(regex)

    # Fused-linear target emission. vLLM's model-loading time fuses
    # siblings from `packed_modules_mapping` into a single packed Linear
    # (e.g. Qwen3.5 DeltaNet's `in_proj_qkv + in_proj_z → in_proj_qkvz`,
    # standard `q_proj + k_proj + v_proj → qkv_proj`). Scheme dispatch
    # keys off the FUSED module's prefix, so our config must list that
    # fused name alongside the siblings. When all expected siblings
    # share one format, emit the fused name into that format's target
    # list; when all land in ignore, emit the fused name into ignore.
    # Mixed-format fused groups are blocked upstream by the allocator's
    # `fused_sibling_group` pre-pass — but we defensively skip emitting
    # a fused target in that case rather than guess.
    if packed_mapping:
        # Map leaf sibling → fused-name, using packed_mapping that vLLM
        # reads at load time.
        leaf_to_fused = {s: fused for fused, sibs in packed_mapping.items()
                         for s in sibs}

        # Build parent-path → {leaf: (fmt|IGNORE, vllm_name)} for every
        # live entry (assignment + extra_ignore + bf16_passthrough).
        def _parent_leaf(vname: str):
            parts = vname.rsplit(".", 1)
            if len(parts) != 2:
                return None, vname
            return parts[0], parts[1]

        # (parent, leaf) → (fmt or "IGNORE")
        leaf_state: dict[tuple[str, str], str] = {}
        for fmt, names in by_fmt.items():
            for vname in names:
                parent, leaf = _parent_leaf(vname)
                if parent is None:
                    continue
                leaf_state[(parent, leaf)] = fmt
        ignore_set = set(ignore)
        for vname in ignore_set:
            parent, leaf = _parent_leaf(vname)
            if parent is None:
                continue
            leaf_state.setdefault((parent, leaf), "IGNORE")

        # For each (parent, fused) pair where all siblings are present
        # and share a state, emit the fused-name target.
        fused_emitted: set[str] = set()
        parents = {p for (p, _) in leaf_state}
        for parent in parents:
            for fused_name, sibs in packed_mapping.items():
                # Skip degenerate fused definitions (single-sibling).
                if len(sibs) < 2:
                    continue
                states = [leaf_state.get((parent, s)) for s in sibs]
                if any(s is None for s in states):
                    continue  # not all siblings present → skip
                if len(set(states)) != 1:
                    continue  # mixed formats → caller's bug; don't emit
                state = states[0]
                fused_vllm_name = f"{parent}.{fused_name}"
                if fused_vllm_name in fused_emitted:
                    continue
                fused_emitted.add(fused_vllm_name)
                if state == "IGNORE":
                    ignore.append(fused_vllm_name)
                else:
                    by_fmt.setdefault(state, []).append(fused_vllm_name)

    if not by_fmt:
        return {}

    sizes = {k: len(v) for k, v in by_fmt.items()}
    catchall = max(sizes, key=sizes.get) if sizes else None
    config_groups = {}
    idx = 0
    for fmt, names in by_fmt.items():
        if fmt == catchall:
            continue
        scheme = deepcopy(FORMAT_SCHEME[fmt])
        scheme["targets"] = _build_target_list(names)
        config_groups[f"group_{idx}"] = scheme
        idx += 1
    if catchall is not None:
        scheme = deepcopy(FORMAT_SCHEME[catchall])
        # Explicit per-name targets, NOT a class-name catch-all
        # ("Linear"). The class-name catch-all matches via a substring
        # check against module class (e.g. MergedColumnParallelLinear)
        # and short-circuits vLLM's fused-layer regex resolution, which
        # is needed to route the explicit per-component MXFP8_E4M3 targets
        # to vLLM's fused parameter (in_proj_qkvz, qkv_proj, etc.).
        # `_build_target_list` collapses per-expert enumerations into
        # compact regexes so a 256-expert / 62-layer MoE emits
        # a few hundred targets instead of ~47k. The profile's
        # per-expert regexes remain as a safety-net for any
        # per-expert Linear not captured by the collapse (e.g.
        # stray experts the recipe didn't enumerate).
        # The profile per-expert regexes name on-disk projections; vLLM's
        # scheme probe uses canonical gate_proj/up_proj/down_proj. Rewrite
        # the projection group ONLY when the on-disk names differ from
        # canonical (LFM2.5's w1/w3/w2) — left verbatim when the profile is
        # already canonical (e.g. Qwen), so shipped configs don't churn.
        ondisk: set[str] = set()
        canon: set[str] = set()
        for pname in sorted(_packed_expert_param_name_set(profile)):
            ondisk.update(_packed_expert_projection_names(profile, pname))
            canon.update(_vllm_moe_scheme_projection_names(profile, pname))
        need_canon = ondisk != canon and bool(canon)
        canon_opts = "|".join(sorted(canon)) or "gate_proj|up_proj|down_proj"
        expert_regexes = []
        for getter in (profile.per_expert_moe_regex, profile.per_expert_mtp_regex):
            r = getter()
            if r is None:
                continue
            if need_canon:
                body = r[len("re:"):] if r.startswith("re:") else r
                r = f"re:{_constrain_per_expert_projection_regex(body, canon_opts)}"
            expert_regexes.append(r)
        scheme["targets"] = _build_target_list(by_fmt[catchall]) + expert_regexes
        config_groups[f"group_{idx}"] = scheme

    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        "quantization_status": "compressed",
    }


# ---------------------------------------------------------------------------
# Recipe canonicalization + Main
# ---------------------------------------------------------------------------
# Per-expert siblings map to a fused packed parent at recipe level.
# If the parent IS quantized, the per-expert source keys are already
# covered and must NOT be added to `extra_ignore` — otherwise vLLM's
# compressed-tensors loader marks the FusedMoE layer as un-quantized
# and the NVFP4 scale params (w2_input_global_scale, ...) never get
# registered, crashing at weight-load.
_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.\d+\.(?P<proj>[^.]+)$")


def _per_expert_parent(base: str, profile=None) -> str | None:
    """Map a per-expert source tensor base like
    `model.layers.0.mlp.experts.3.gate_proj` to its packed parent
    (for example `model.layers.0.mlp.experts.gate_up_proj`), or None
    if `base` is not a per-expert tensor."""
    m = _PER_EXPERT_RE.match(base)
    if not m:
        return None
    parent = _packed_expert_parent_for_projection(profile, m.group("proj"))
    if parent is None:
        return None
    return f"{m.group('prefix')}.{parent}"


def compute_extra_ignore(
    source_shape_iter,
    assignment: dict[str, str],
    profile=None,
) -> list[str]:
    """Return the list of 2D `.weight` basenames that must be added to
    the compressed-tensors `ignore` set because the recipe doesn't cover
    them.

    `source_shape_iter` yields `(ckpt_key, shape)` for every tensor in
    the source checkpoint (or None for shape when unknown — treated as
    non-2D and skipped). `assignment` maps recipe names to formats.

    Per-expert source keys (e.g. `...experts.3.gate_proj.weight`) are
    NOT added to `extra_ignore` when their packed parent is in the
    assignment — the parent's emitted compressed-tensors scheme already
    covers them at vLLM load time, and adding the per-expert name to
    `ignore` would mark the FusedMoE layer as un-quantized.

    """
    extra_ignore: list[str] = []
    seen_recipe = set(assignment)
    for ckpt_key, shape in source_shape_iter:
        if not ckpt_key.endswith(".weight"):
            continue
        base = ckpt_key[:-7]
        if profile is not None:
            recipe_name = profile.live_to_recipe_name(base)
        else:
            recipe_name = ("model." + base[len("model.language_model."):]
                           if base.startswith("model.language_model.")
                           else base)
        if recipe_name in seen_recipe:
            continue
        parent = _per_expert_parent(recipe_name, profile)
        if parent is not None and parent in seen_recipe:
            continue
        if shape is None or len(shape) != 2:
            continue
        extra_ignore.append(base)
    return extra_ignore


def main():
    global _INPUT_GLOBAL_SCALES, _CACHED_ACTIVATIONS, _ACTIVATION_CACHE_FINGERPRINT
    global _PRODUCTION_WEIGHT_CACHE, _PRODUCTION_CACHE_FINGERPRINT
    global _PRODUCTION_CACHE_PREFETCH_WORKERS, _NVFP4_SCALE_RULE
    _INPUT_GLOBAL_SCALES = None
    _CACHED_ACTIVATIONS = None
    _ACTIVATION_CACHE_FINGERPRINT = None
    _PRODUCTION_WEIGHT_CACHE = None
    _PRODUCTION_CACHE_FINGERPRINT = None
    _NVFP4_SCALE_RULE = resolve_nvfp4_scale_rule()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", default=None,
                    help="layer_config.json from allocator.py. Optional when "
                         "--perturbed-x-dir is supplied.")
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024**3,
                    help="Approx per-shard size in bytes (default 5 GiB)")
    ap.add_argument("--device", default="cuda",
                    help="CUDA device for quantization arithmetic. Layer "
                         "weights are read into this device; "
                         "_quantize_2d / _quantize_3d_packed run here; "
                         "outputs are moved to CPU before storage. CPU is "
                         "rejected for production quantization.")
    ap.add_argument("--offload-folder", default=None,
                    help="Accelerate disk-offload folder (defaults to "
                         "sibling of output).")
    ap.add_argument("--ignore", nargs="*", default=None,
                    help="Module qnames to keep at bf16 even if the "
                         "allocator assigned another format. Default: the "
                         "active model profile's pinned_names (typically "
                         "lm_head/head for current vLLM serving targets). "
                         "Pass --ignore with no values to disable profile "
                         "pinning for a runtime that supports quantized heads.")
    ap.add_argument("--activation-cache-dir", default=None,
                    help="Probe's activation cache directory. When "
                         "supplied, per-Linear input_global_scale is "
                         "computed from cached activations "
                         "(max_abs/6.0) instead of the 1.0 default. "
                         "Typically ~1-3%% PPL improvement on NVFP4.")
    ap.add_argument("--production-weight-cache", default=None,
                    help="Pickled ProductionWeightCache containing "
                         "already-rendered production weights. When "
                         "supplied, export packs those weights directly "
                         "instead of recomputing GPTQ/scale-sweep from "
                         "raw activations. This is the faithful path for "
                         "candidates measured with production_weight_cache.")
    ap.add_argument("--production-cache-dir-override", default=None,
                    help="Override the backing shard directory stored "
                         "inside --production-weight-cache, for caches "
                         "moved between containers or host paths.")
    ap.add_argument("--production-cache-lru-gb", type=float, default=24.0,
                    help="Resident tensor budget for disk-backed production "
                         "cache loads. Layer export always prefetches the "
                         "current layer into this LRU.")
    ap.add_argument("--production-cache-prefetch-workers", type=int, default=4,
                    help="Thread count for production-cache prefetch.")
    ap.add_argument("--perturbed-x-dir", default=None,
                    help="Directory containing final_layer_config.json and "
                         "activation cache files from a prior production "
                         "calibration/polish run. When supplied, defaults "
                         "--layer-config and --activation-cache-dir from it.")
    ap.add_argument("--gptq", dest="gptq", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="GPTQ one-shot OBS rounding with block-wise "
                         "error propagation (NVFP4, FP8_E4M3/FP8_E5M2, "
                         "MXFP8_E4M3/MXFP8_E5M2). Auto-on when --activation-cache-dir "
                         "is supplied. Measured -2.7%% PPL on Qwen3.6-35B.")
    ap.add_argument("--gptq-static-act-order", dest="gptq_static_act_order",
                    default=None, action=argparse.BooleanOptionalAction,
                    help="Opt-in Lift/MR-GPTQ static activation ordering. "
                         "Columns are processed by activation importance "
                         "during GPTQ but restored before export, so no "
                         "runtime permutation is introduced.")
    ap.add_argument("--gptq-joint-scale-opt", dest="gptq_joint_scale_opt",
                    default=None, action=argparse.BooleanOptionalAction,
                    help="Opt-in Lift/MR-GPTQ joint NVFP4 scale search inside "
                         "GPTQ. The candidate set includes FourOverSix and "
                         "additional codebook-aligned max-to-level scales.")
    ap.add_argument("--scale-sweep", dest="scale_sweep", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Per-group 1-D scale sweep with RTN rounding on "
                         "NVFP4 — closed-form analog of AutoRound's SGD. "
                         "Auto-on when --activation-cache-dir is supplied. "
                         "Measured best-in-bake-off when composed after "
                         "GPTQ: geomean out_mse ratio = 0.33 vs GPTQ-only "
                         "0.41 vs RTN 1.0, on Qwen3.6-35B visual+MTP "
                         "Linears.")
    ap.add_argument("--export-cache-dir", default=None,
                    help="Per-layer cache dir for resumable export. When "
                         "set, each layer's emitted tensor dict is "
                         "torch.save'd to <cache_dir>/layer_NNN.pt right "
                         "after quantization. On a restart, layers whose "
                         "cache file exists are SKIPPED — their tensors "
                         "are loaded from cache and replayed into the "
                         "shard writer without redoing the GPTQ + "
                         "scale_sweep work. Recovers full progress on a "
                         "mid-flight kill (which today restarts from "
                         "layer 0 every time). Cache is removed at end of "
                         "successful export. Disk overhead: ~2 GB per "
                         "MoE layer = ~120 GB transient on a 62-layer "
                         "MiniMax-class model, freed on completion.")
    ap.add_argument("--keep-export-cache", action="store_true",
                    default=False,
                    help="Don't remove --export-cache-dir on success. "
                         "Useful for debugging or comparing two exports "
                         "against the same cache.")
    args = ap.parse_args()

    from .model_profiles import detect_profile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)

    if args.perturbed_x_dir:
        px_layer_config, px_cache_dir = _resolve_perturbed_x_export_inputs(
            args.perturbed_x_dir
        )
        if args.layer_config is None:
            args.layer_config = str(px_layer_config)
        if args.activation_cache_dir is None:
            args.activation_cache_dir = str(px_cache_dir)
        print("[export-stream] perturbed-X inputs: "
              f"layer_config={args.layer_config} "
              f"activation_cache_dir={args.activation_cache_dir}",
              flush=True)
    if args.layer_config is None:
        ap.error("--layer-config is required unless --perturbed-x-dir is supplied")

    with open(args.layer_config) as _lc_for_cache:
        _layer_config_payload_for_cache = json.load(_lc_for_cache)
    validate_layer_config_payload(_layer_config_payload_for_cache, args.layer_config)
    _assignment_for_cache = _canonicalize_assignment(_layer_config_payload_for_cache)
    _assignment_for_cache, _ = _coerce_runtime_legal_assignment(
        args.model,
        _assignment_for_cache,
        profile,
    )

    if args.production_weight_cache:
        import pickle

        with open(args.production_weight_cache, "rb") as fh:
            production_cache = pickle.load(fh)
        if args.production_cache_dir_override:
            production_cache.relocate(args.production_cache_dir_override)
        if args.production_cache_lru_gb and args.production_cache_lru_gb > 0:
            production_cache.enable_lru(
                int(float(args.production_cache_lru_gb) * 1024**3)
            )
        _PRODUCTION_WEIGHT_CACHE = production_cache
        _PRODUCTION_CACHE_PREFETCH_WORKERS = max(
            1, int(args.production_cache_prefetch_workers)
        )
        expected_keys, missing_keys = _production_cache_expected_keys(
            _assignment_for_cache
        )
        if missing_keys:
            raise RuntimeError(
                "[export-stream] production-weight-cache missing recipe "
                f"entries: {len(missing_keys)} sample={missing_keys[:8]}"
            )
        files = production_cache.verify_files(expected_keys)
        if files["missing"]:
            raise RuntimeError(
                "[export-stream] production-weight-cache backing files "
                f"missing: {len(files['missing'])} sample={files['missing'][:8]}"
            )
        _PRODUCTION_CACHE_FINGERPRINT = _production_cache_fingerprint(
            production_cache,
            expected_keys,
        )
        _INPUT_GLOBAL_SCALES = _production_cache_scales(production_cache)
        print(
            "[export-stream] production-weight-cache direct path: "
            f"{len(expected_keys)} entries, "
            f"lru={args.production_cache_lru_gb:.1f} GiB, "
            f"prefetch_workers={_PRODUCTION_CACHE_PREFETCH_WORKERS}",
            flush=True,
        )

    # Resolve flag defaults.
    cache_supplied = bool(args.activation_cache_dir)
    # GPTQ + scale-sweep: ON iff activation cache supplied.
    gptq_enabled = args.gptq if args.gptq is not None else cache_supplied
    # scale_sweep: ON iff activation cache supplied.
    scale_sweep_enabled = (args.scale_sweep if args.scale_sweep is not None
                           else cache_supplied)
    static_act_order_enabled = (
        args.gptq_static_act_order
        if args.gptq_static_act_order is not None
        else os.environ.get(
            "PRISMAQUANT_GPTQ_STATIC_ACT_ORDER",
            "0",
        ).strip().lower() not in {"", "0", "false", "no", "off"}
    )
    joint_scale_opt_enabled = (
        args.gptq_joint_scale_opt
        if args.gptq_joint_scale_opt is not None
        else os.environ.get(
            "PRISMAQUANT_NVFP4_JOINT_SCALE_OPT",
            "0",
        ).strip().lower() not in {"", "0", "false", "no", "off"}
    )
    static_act_order_enabled = bool(gptq_enabled and static_act_order_enabled)
    joint_scale_opt_enabled = bool(gptq_enabled and joint_scale_opt_enabled)
    if (
        joint_scale_opt_enabled
        and NVFP4_SCALE_RULE_ENV not in os.environ
        and _NVFP4_SCALE_RULE == NVFP4_SCALE_RULE_STATIC_6
    ):
        _NVFP4_SCALE_RULE = NVFP4_SCALE_RULE_JOINT_MSE
    act_passes_any = gptq_enabled or scale_sweep_enabled
    # The activation-aware passes need the actual activations, not just
    # the scale summary. We only load raw activations when at least one
    # pass is enabled.
    if act_passes_any and not cache_supplied:
        print("[export-stream] WARN activation-aware passes requested "
              "but no --activation-cache-dir; disabling.", flush=True)
        gptq_enabled = False
        scale_sweep_enabled = False
        static_act_order_enabled = False
        joint_scale_opt_enabled = False
        act_passes_any = False
    print(f"[export-stream] act-aware passes: "
          f"gptq={gptq_enabled} "
          f"scale_sweep={scale_sweep_enabled} "
          f"static_act_order={static_act_order_enabled} "
          f"joint_scale_opt={joint_scale_opt_enabled}", flush=True)
    print(f"[export-stream] NVFP4 scale rule: {_nvfp4_scale_rule_from_env()}",
          flush=True)
    # Publish to the module-level config so `_quantize_2d` picks them
    # up from every call site without needing the flags threaded
    # through `materialize_tensors_streaming` + MTP helpers.
    _ACT_AWARE_FLAGS["gptq"] = gptq_enabled
    _ACT_AWARE_FLAGS["scale_sweep"] = scale_sweep_enabled
    _ACT_AWARE_FLAGS["static_act_order"] = static_act_order_enabled
    _ACT_AWARE_FLAGS["joint_scale_opt"] = joint_scale_opt_enabled

    # Populate the module-level input-global-scale cache (used by
    # `_quantize_2d` for NVFP4 linears) from cached activations.
    # Same cache is reused to populate _CACHED_ACTIVATIONS when any
    # act-aware pass is enabled.
    if args.activation_cache_dir and _PRODUCTION_WEIGHT_CACHE is not None:
        print("[export-stream] production-weight-cache supplied; using its "
              "activation scales and pre-rendered weights for assigned "
              "Linears. Raw activation cache will not drive body export.",
              flush=True)
    elif args.activation_cache_dir:
        from .measure_quant_cost import ActivationIndex
        cache_dir = Path(args.activation_cache_dir)
        if not cache_dir.exists():
            print(f"[export-stream] WARN activation cache dir {cache_dir} "
                  f"missing; input_global_scale falls back to "
                  f"{DEFAULT_INPUT_GLOBAL_SCALE}", flush=True)
            _ACTIVATION_CACHE_FINGERPRINT = {
                "path": str(cache_dir.resolve()),
                "missing": True,
            }
        else:
            # Pull candidate names from the recipe — ActivationIndex
            # only loads for names that actually have a cached file.
            with open(args.layer_config) as _lc:
                _recipe_payload = json.load(_lc)
            validate_layer_config_payload(_recipe_payload, args.layer_config)
            _recipe_names = list(_recipe_payload.keys())
            idx = ActivationIndex(cache_dir, _recipe_names)
            _ACTIVATION_CACHE_FINGERPRINT = _activation_index_fingerprint(
                idx, cache_dir)
            scales: dict[str, float] = {}
            for name in idx.names():
                try:
                    acts = idx.load(name)
                    scales[name] = compute_nvfp4_input_global_scale(acts)
                except Exception as e:
                    print(f"[export-stream] WARN could not load "
                          f"activations for {name}: {e}", flush=True)
            # Unify input_global_scale across fused-sibling groups.
            # vLLM's fused Linear loader concatenates q/k/v (and gate/up)
            # into a single tensor and applies ONE input scale at
            # forward time. If q/k/v scales differ the warning
            #   "global scale for input or weight are different for
            #    parallel layers (e.g. q_proj, k_proj, v_proj). This
            #    will likely result in reduced accuracy."
            # fires at vLLM load. q/k/v siblings receive the same
            # upstream activation in principle, but captured per-
            # Linear from different shard subsamples, so the computed
            # max/6 values can drift by a float-precision tick. Take
            # the max over the group so vLLM runs on the conservative
            # (larger) scale for every sibling.
            scales = _unify_input_global_scales_across_fused_siblings(
                scales,
                profile=profile,
            )
            _INPUT_GLOBAL_SCALES = scales
            if act_passes_any:
                _CACHED_ACTIVATIONS = _LazyActivationCache(idx)
                print(f"[export-stream] raw activations will be loaded "
                      f"lazily for GPTQ/round/scale-sweep passes "
                      f"({len(idx)}/{len(_recipe_names)} Linears indexed)",
                      flush=True)
            print(f"[export-stream] input_global_scale calibrated for "
                  f"{len(scales)}/{len(_recipe_names)} Linears from "
                  f"{cache_dir}", flush=True)

    with open(args.layer_config) as f:
        raw_recipe = json.load(f)
    validate_layer_config_payload(raw_recipe, args.layer_config)
    assignment = _canonicalize_assignment(raw_recipe)
    assignment, runtime_coerced = _coerce_runtime_legal_assignment(
        args.model,
        assignment,
        profile,
    )
    if runtime_coerced:
        print(
            "[export-stream] runtime format coercions: "
            f"{len(runtime_coerced)} Linears -> BF16 "
            "(target runtime does not support those format/shape pairs). "
            f"sample={runtime_coerced[:6]}",
            flush=True,
        )
    validate_mtp_assignment_coverage(args.model, assignment, profile)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    from prismaquant.gpu_guard import require_cuda_hot_path

    dtype = torch.bfloat16
    device = require_cuda_hot_path(
        "export_native_compressed",
        args.device,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_passthrough = set(
        args.ignore
        if args.ignore is not None
        else profile.pinned_names()
    )
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    def _rename_body_batch(
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {profile.export_tensor_name(k): v for k, v in batch.items()}

    writer = IncrementalSafetensorsWriter(out_dir, args.shard_bytes)
    sample_recipe_key = "model.layers.0.self_attn.q_proj.weight"
    sample_source_key = profile.export_tensor_name(sample_recipe_key)
    if sample_source_key != sample_recipe_key:
        print(
            "[export-stream] streaming source-name remap via profile: "
            f"{sample_recipe_key} -> {sample_source_key}",
            flush=True,
        )

    tensors, hist = materialize_tensors_streaming(
        args.model, assignment,
        profile=profile, bf16_passthrough=bf16_passthrough,
        dtype=dtype, device=device,
        offload_folder=args.offload_folder,
        tensor_sink=lambda batch: writer.add_tensors(_rename_body_batch(batch)),
        export_cache_dir=args.export_cache_dir,
    )
    print(f"[export-stream] streamed materialization complete "
          f"resident_tensors={len(tensors)}  hist={hist}",
          flush=True)

    # MTP materialization if the profile has heads. Uses the in-memory
    # helper — MTP heads are small enough that full-model residency
    # isn't a concern.
    mtp_tensors: dict[str, torch.Tensor] = {}
    if profile.has_mtp():
        print("[export-stream] materializing MTP tensors ...", flush=True)
        mtp_tensors = _materialize_mtp_tensors(
            args.model, assignment,
            bf16_passthrough=bf16_passthrough, hist=hist,
            device=device)
        print(f"[export-stream] MTP: {len(mtp_tensors)} tensors", flush=True)
    else:
        print(f"[export-stream] profile '{profile.name}' has no MTP — "
              "skipping", flush=True)

    # Merge source passthrough (visual/audio towers etc.) that aren't
    # part of our streaming pass. Drop entries that MTP materialize
    # already covered.
    passthrough_prefixes = tuple(profile.source_passthrough_prefixes())
    if passthrough_prefixes:
        src_extra = _load_source_passthrough(
            args.model, prefix_filters=passthrough_prefixes)
        src_extra = _filter_source_passthrough_against_materialized(
            src_extra,
            mtp_tensors,
            profile=profile,
            seen_keys=writer.seen_keys,
        )

        # Phase 1 visual-encoder quant: when the allocator's recipe
        # assigns a non-BF16 format to a visual Linear, run its 2D
        # weight through `_quantize_2d` before emit. BF16 entries and
        # non-Linear tensors (norms, conv1d, biases, buffers) pass
        # through unchanged. See allocator's `--visual-format` docstring
        # for why this is a uniform override rather than a per-Linear
        # decision — text-only probe never exercises the visual tower.
        src_extra = _apply_visual_recipe_quant(
            src_extra, assignment, device=device)

        writer.add_tensors(mtp_tensors)
        writer.add_tensors(src_extra)
        print(f"[export-stream] merged {len(src_extra)} source-passthrough + "
              f"{len(mtp_tensors)} MTP tensors", flush=True)
    else:
        writer.add_tensors(mtp_tensors)

    print("[export-stream] finalizing safetensors shards ...", flush=True)
    t_write = time.time()
    writer.finalize()
    print(f"[export-stream] sharded write: {time.time()-t_write:.1f}s",
          flush=True)

    # Scan source safetensors for 2D `.weight` keys not covered by the
    # recipe — these are visual encoder / unmapped Linears that vLLM
    # instantiates during model-construction time. Without an explicit
    # ignore entry, compressed-tensors' `find_matched_target` raises
    # `ValueError: Unable to find matching target for visual.merger.*`.
    src_dir = Path(args.model)

    def _source_shape_iter():
        if not src_dir.exists():
            return
        from safetensors import safe_open
        import os as _os
        for f in sorted(_os.listdir(src_dir)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src_dir / f), framework="pt") as sf:
                for k in sf.keys():
                    try:
                        shape = list(sf.get_slice(k).get_shape())
                    except Exception:
                        shape = None
                    yield k, shape

    extra_ignore = compute_extra_ignore(_source_shape_iter(), assignment, profile)
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, assignment, bf16_passthrough,
        extra_ignore=extra_ignore,
        transform_config=None)
    _copy_tokenizer(args.model, out_dir)

    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(assignment),
            "runtime_coercions": [
                {"name": name, "shape": shape, "from": from_fmt, "to": "BF16"}
                for name, shape, from_fmt in runtime_coerced
            ],
            "bf16_audit": _bf16_upgrade_audit(
                args.model,
                assignment,
                bf16_passthrough,
                runtime_coerced,
                profile,
            ),
            "ignore": sorted(bf16_passthrough),
        }, f, indent=2)

    # v25: clear the per-layer cache on successful export. --keep-export-cache
    # leaves it intact (debugging / comparison). On a failed run the cache
    # stays anyway since this code wouldn't be reached.
    if (args.export_cache_dir
            and not args.keep_export_cache
            and Path(args.export_cache_dir).exists()):
        import shutil
        try:
            shutil.rmtree(args.export_cache_dir)
            print(f"[export-stream] removed export cache "
                  f"{args.export_cache_dir}", flush=True)
        except Exception as e:
            print(f"[export-stream] WARN cache cleanup failed: {e!r}",
                  flush=True)

    print(f"[export-stream] done. Serve with:\n"
          f"  vllm serve {out_dir.resolve()} --quantization compressed-tensors",
          flush=True)


# ---------------------------------------------------------------------------
# Sharded safetensors writer (mirrors HF transformers' shard layout so
# the index file is the same one transformers + vLLM expect).
# ---------------------------------------------------------------------------
def _clone_shared_storage_for_safetensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a save-ready dict without same-file shared storage ties."""
    out = dict(tensors)
    seen_storage: dict[int, str] = {}
    for k, t in list(out.items()):
        try:
            sid = t.untyped_storage().data_ptr()
        except Exception:
            continue
        if sid in seen_storage:
            # This tensor shares storage with an earlier one. Deep-copy
            # so safetensors treats them independently.
            out[k] = t.detach().clone().contiguous()
        else:
            seen_storage[sid] = k
    return out


class IncrementalSafetensorsWriter:
    """Write HF-style safetensor shards while batches are produced.

    The legacy writer receives the entire tensor dict and therefore needs
    enough host RAM for the full compressed checkpoint. Large MoE exports
    can exceed that before the final write phase. This writer keeps only
    one output shard resident, writes temporary shard files as soon as
    they reach the byte budget, then renames them to the final
    `model-00001-of-000NN.safetensors` layout and writes the index once
    the final shard count is known.
    """

    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out_dir = out_dir
        self.shard_bytes = int(shard_bytes)
        self.current: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self.total_size = 0
        self.tmp_shards: list[tuple[Path, list[str]]] = []
        self.weight_map: dict[str, str] = {}
        self.seen_keys: set[str] = set()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def add_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        if not tensors:
            return
        for key in sorted(tensors):
            if key in self.seen_keys:
                raise RuntimeError(
                    f"duplicate tensor key emitted during export: {key}"
                )
            tensor = tensors[key].detach().cpu()
            size = self._tensor_size(tensor)
            if (self.current
                    and self.current_size + size > self.shard_bytes):
                self._flush_current()
            self.current[key] = tensor
            self.current_size += size
            self.total_size += size
            self.seen_keys.add(key)
            # A single tensor can exceed the target shard size. Flush it
            # immediately so the next shard starts cleanly.
            if self.current_size >= self.shard_bytes:
                self._flush_current()

    def _flush_current(self) -> None:
        if not self.current:
            return
        idx = len(self.tmp_shards) + 1
        tmp_path = self.out_dir / f".model-{idx:05d}.safetensors.tmp"
        save_file(
            {k: v.contiguous() for k, v in
             _clone_shared_storage_for_safetensors(self.current).items()},
            str(tmp_path),
            metadata={"format": "pt"},
        )
        self.tmp_shards.append((tmp_path, list(self.current.keys())))
        print(
            f"[export-stream] wrote temp shard {idx:05d} "
            f"keys={len(self.current)} bytes={self.current_size}",
            flush=True,
        )
        self.current = {}
        self.current_size = 0
        gc.collect()

    def finalize(self) -> None:
        self._flush_current()
        if not self.tmp_shards:
            raise RuntimeError("no tensors were written")

        if len(self.tmp_shards) == 1:
            tmp_path, keys = self.tmp_shards[0]
            final_name = "model.safetensors"
            os.replace(tmp_path, self.out_dir / final_name)
            for key in keys:
                self.weight_map[key] = final_name
            print("[export-stream] finalized single safetensors shard",
                  flush=True)
            return

        n = len(self.tmp_shards)
        for i, (tmp_path, keys) in enumerate(self.tmp_shards, start=1):
            final_name = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp_path, self.out_dir / final_name)
            for key in keys:
                self.weight_map[key] = final_name

        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump({
                "metadata": {"total_size": self.total_size},
                "weight_map": self.weight_map,
            }, f, indent=2)
        print(f"[export-stream] finalized {n} safetensors shards",
              flush=True)


def write_sharded_safetensors(
    tensors: dict[str, torch.Tensor],
    out_dir: Path,
    shard_bytes: int,
) -> None:
    # Detach + clone any tensors that share underlying storage so
    # safetensors' dedup check doesn't raise. This covers tied
    # embeddings (Gemma 4: `lm_head.weight` ≡ `embed_tokens.weight`)
    # and any other view-ties produced by HF's
    # `_tied_weights_keys`. Cost: one extra copy of the embed matrix;
    # correctness: identical bytes on disk, no runtime semantic change.
    tensors = _clone_shared_storage_for_safetensors(tensors)

    keys = sorted(tensors.keys())
    sizes = {k: tensors[k].numel() * tensors[k].element_size() for k in keys}
    total = sum(sizes.values())
    n_shards = max(1, math.ceil(total / shard_bytes))
    target = math.ceil(total / n_shards)

    shards: list[list[str]] = [[]]
    cur = 0
    for k in keys:
        if cur + sizes[k] > target and shards[-1]:
            shards.append([])
            cur = 0
        shards[-1].append(k)
        cur += sizes[k]

    if len(shards) == 1:
        path = out_dir / "model.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shards[0]},
            str(path),
            metadata={"format": "pt"},
        )
        return

    weight_map: dict[str, str] = {}
    n = len(shards)
    for i, shard_keys in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        save_file(
            {k: tensors[k].contiguous() for k in shard_keys},
            str(out_dir / shard_name),
            metadata={"format": "pt"},
        )
        for k in shard_keys:
            weight_map[k] = shard_name

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump({
            "metadata": {"total_size": total},
            "weight_map": weight_map,
        }, f, indent=2)


def write_config_with_quantization(
    src_model: str, out_dir: Path,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    extra_ignore: Iterable[str] = (),
    transform_config: dict | None = None,
) -> None:
    from .model_profiles import detect_profile
    profile = detect_profile(src_model)
    src_cfg_path = Path(src_model) / "config.json"
    cfg = json.load(open(src_cfg_path))
    qc = build_quantization_config(assignment, bf16_passthrough,
                                   extra_ignore, profile=profile)
    if qc:
        if transform_config:
            qc["transform_config"] = transform_config
        cfg["quantization_config"] = qc

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _materialize_mtp_tensors(src_model: str,
                             assignment: dict[str, str],
                             *,
                             bf16_passthrough: set[str],
                             hist: dict,
                             device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so the streaming decoder-layer sweep never sees any `mtp.*` entry in
    `assignment`. We build a standalone MTP module, load the source
    `mtp.*` weights into it, wrap it in a parent module named `mtp` (so
    qualified names come out as `mtp.fc`, `mtp.layers.0.self_attn.q_proj`,
    ...), and run the in-memory materialize helper.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from .mtp_module import MtpModule, _load_into_mtp, _load_mtp_state_dict
    from transformers import AutoConfig

    # Build an MTP wrapper with source weights.
    cfg = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    text_config = getattr(cfg, "text_config", cfg)
    inner = MtpModule(text_config)
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=torch.bfloat16)
    raw = _load_mtp_state_dict(src_model)
    _load_into_mtp(inner, raw)
    # Move the whole MTP module to the export device so
    # _materialize_tensors_inmemory's per-linear quant runs on GPU when
    # EXPORT_DEVICE=cuda. Previously defaulted to CPU, costing ~10× on
    # MTP quant. The input weights (raw) are CPU, so we move after load.
    wrapper.to(device=device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries.
    mtp_assignment = {k: v for k, v in assignment.items() if k.startswith("mtp.")}
    if not mtp_assignment:
        return {}

    out, sub_hist = _materialize_tensors_inmemory(
        wrapper, mtp_assignment, bf16_passthrough=bf16_passthrough,
    )
    # Merge MTP histogram into caller's.
    for k, v in sub_hist.items():
        hist[("mtp_" + k[0], k[1])] = hist.get(("mtp_" + k[0], k[1]), 0) + v
    return out


def _load_source_passthrough(src_model: str,
                             prefix_filters: tuple[str, ...]
                             ) -> dict[str, torch.Tensor]:
    """Pull tensors from the source safetensors whose key begins with
    any of `prefix_filters`. Returns the loaded tensors so they can be
    written back verbatim into the export. Used for visual encoder +
    MTP head weights that the recipe doesn't touch but vLLM expects to
    find at load time.
    """
    import os
    from safetensors.torch import safe_open
    src_dir = Path(src_model)
    out: dict[str, torch.Tensor] = {}
    for f in sorted(os.listdir(src_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(str(src_dir / f), framework="pt") as sf:
            for k in sf.keys():
                if any(k.startswith(p) for p in prefix_filters):
                    out[k] = sf.get_tensor(k)
    return out


def _filter_source_passthrough_against_materialized(
    src_extra: dict[str, torch.Tensor],
    materialized: dict[str, torch.Tensor],
    *,
    profile,
    seen_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Drop source passthrough tensors already represented by materialized output.

    MTP is synthesized separately from raw `mtp.*` source tensors. For BF16
    packed MTP experts, the synthesized form is the vLLM-loader aggregate
    tensor (`...experts.gate_up_proj` / `...experts.down_proj`), while the
    source checkpoint stores per-expert children
    (`...experts.0.gate_proj.weight`, etc.). Those children must not be copied
    too: vLLM loads the aggregate and then warns on the duplicate children.
    """
    materialized_bases: set[str] = set()
    for key in materialized:
        base = key
        for suffix in (".weight_packed", ".weight_scale",
                       ".weight_global_scale", ".input_global_scale",
                       ".weight"):
            if key.endswith(suffix):
                base = key[:-len(suffix)] + ".weight"
                break
        materialized_bases.add(base)
        if base.endswith(".weight"):
            parent = _per_expert_parent(base[:-len(".weight")], profile)
            if parent is not None:
                materialized_bases.add(parent)

    seen_keys = seen_keys or set()

    def _covered_by_materialized_source_form(key: str) -> bool:
        if key in materialized or key in materialized_bases or key in seen_keys:
            return True
        if key.endswith(".weight"):
            parent = _per_expert_parent(key[:-len(".weight")], profile)
            if parent is not None and parent in materialized_bases:
                return True
        return False

    return {
        key: value for key, value in src_extra.items()
        if not _covered_by_materialized_source_form(key)
    }


_VISUAL_KEY_RE = re.compile(r"^(?:model\.)?visual\.")


def _apply_visual_recipe_quant(
    src_extra: dict[str, torch.Tensor],
    assignment: dict[str, str],
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Rewrite visual-encoder `.weight` entries in `src_extra` under the
    recipe's per-Linear format assignment.

    The allocator's `--visual-format` flag stamps every visual Linear
    with a uniform format (`BF16` | `NVFP4` | `MXFP8_E4M3`). For BF16 we do
    nothing — the passthrough tensor is already in the right dtype
    (typically bf16 in the source). For NVFP4 / MXFP8_E4M3 we route the
    rank-2 weight through `_quantize_2d` and replace the single
    `<name>.weight` key with the compressed-tensors tensor set
    (`<name>.weight_packed`, `<name>.weight_scale`,
    `<name>.weight_global_scale`, `<name>.input_global_scale` for NVFP4;
    `<name>.weight`, `<name>.weight_scale` for MXFP8_E4M3).

    Non-Linear tensors (norms, conv1d, biases, buffers) and visual
    keys WITHOUT a recipe entry are passed through unchanged —
    consistent with the Phase 1 uniform-override contract: only
    Linears discovered by `discover_visual_linears_from_source` end up
    with a recipe entry, and that helper rejects anything that isn't
    rank-2.

    `device` is the compute device for quant arithmetic; outputs are
    moved to CPU before storage so they're ready for the sharded
    safetensors writer.
    """
    out: dict[str, torch.Tensor] = {}
    touched = 0
    for key, tensor in src_extra.items():
        if not key.endswith(".weight"):
            out[key] = tensor
            continue
        if not _VISUAL_KEY_RE.match(key):
            out[key] = tensor
            continue
        base = key[:-len(".weight")]
        fmt = assignment.get(base)
        if fmt is not None:
            fmt = _canonical_export_format(fmt)
        if fmt is None or fmt == "BF16":
            out[key] = tensor
            continue
        if tensor.ndim != 2:
            # Non-2D visual weights aren't Linear modules — skip them.
            out[key] = tensor
            continue
        weight = tensor.to(device=device, dtype=torch.float32)
        try:
            compressed = _quantize_2d(
                weight, fmt,
                nvfp4_global_real_override=None,
                linear_name=base,
            )
        except Exception as e:
            # Fail-safe: fall back to passthrough on any arithmetic
            # error. Better to land a BF16 visual Linear than crash
            # the whole export — the rest of the body/MTP are already
            # materialized.
            print(f"[export-stream] WARN visual quant failed for {base} "
                  f"({fmt}): {e}; falling back to BF16 passthrough",
                  flush=True)
            out[key] = tensor
            continue
        for suffix, t in compressed.items():
            out[f"{base}.{suffix}"] = t.cpu()
        touched += 1
    if touched:
        print(f"[export-stream] quantized {touched} visual Linear(s) "
              f"from recipe", flush=True)
    return out


def _copy_tokenizer(src_model: str, out_dir: Path) -> None:
    src = Path(src_model)
    for name in (
        "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
        "special_tokens_map.json", "merges.txt", "vocab.json",
        "added_tokens.json", "generation_config.json", "configuration.json",
        # Multimodal preprocessor configs — vLLM's loader for
        # qwen3_vl_moe constructs the multimodal processor even for
        # text-only requests, so the preprocessor files must travel
        # with the checkpoint.
        "preprocessor_config.json", "video_preprocessor_config.json",
        "processor_config.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out_dir / name)
    # Custom architecture modules (trust_remote_code). MiniMax-M2 ships
    # `configuration_minimax_m2.py` + `modeling_minimax_m2.py`;
    # DeepSeek-V3 and similar use the same pattern. vLLM's config loader
    # re-reads these via `get_class_from_dynamic_module` when the
    # exported config's `auto_map` still references them, so they must
    # travel with the checkpoint. Copy every `.py` at the source root
    # (there's only ever a handful — the custom modules and occasionally
    # a `modular_*.py` generator; the autogen header warns not to ship
    # both but copying is harmless).
    for py in src.glob("*.py"):
        shutil.copy2(py, out_dir / py.name)


def _source_has_prefixed_weights(src_model: str, prefix: str) -> bool:
    """Return True when the source safetensors index contains any key
    beginning with `prefix`.

    Export-time validation should use the index rather than a loaded HF
    model because transformers intentionally drops `mtp.*` on load for
    Qwen3.5/3.6, which would otherwise make missing recipe coverage look
    benign.
    """
    idx_path = Path(src_model) / "model.safetensors.index.json"
    if not idx_path.exists():
        return False
    with open(idx_path) as f:
        weight_map = json.load(f).get("weight_map", {})
    return any(k.startswith(prefix) for k in weight_map)


def validate_mtp_assignment_coverage(src_model: str,
                                     assignment: dict[str, str],
                                     profile) -> None:
    """Fail fast when an architecture with MTP source weights is being
    exported without any allocator coverage for `mtp.*`.

    Passing raw MTP weights through silently produces a checkpoint that
    looks complete but violates PrismaQuant's intended contract: MTP must
    participate in the same probe/cost/allocation loop as the body. This
    exact state was observed on Qwen3.5-122B where the body artifacts on
    disk were generated without merged MTP probe/cost results.
    """
    if not profile.has_mtp():
        return
    if not _source_has_prefixed_weights(src_model, "mtp."):
        return
    if any(k.startswith("mtp.") for k in assignment):
        return
    raise RuntimeError(
        "source checkpoint contains mtp.* weights but the allocator recipe "
        "contains no mtp.* entries. Re-run the incremental probe + cost "
        "with --include-mtp (the default) so mtp.* tensors are measured, "
        "then rerun allocator/export."
    )


if __name__ == "__main__":
    main()
