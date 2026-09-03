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
import resource
import json
import math
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

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

from . import nvfp4_activation_contract as _nvfp4_activation_contract
from .allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _scan_source_dtype_manifest,
    check_format_applicability,
)
from .fp8_dynamic import fp8_dynamic_weight_qdq
from .mx_formats import e8m0_to_scale, mxfp8_e4m3_qdq
from .serving_profiles import resolve_target_profile
from .layer_config import (
    canonicalize_assignment as _canonicalize_assignment,
    canonicalize_format,
)
from .model_profiles.qwen3_5 import Qwen3_5Profile
from .layer_config import (
    is_layer_config_meta_key as _is_layer_config_meta_key,
    layer_config_metadata as _layer_config_metadata,
)
from .schemas import validate_layer_config_payload
from .export_output_safety import (
    prepare_fresh_export_directory,
    transactional_export_directory,
    validate_fresh_export_directory,
)
from .nvfp4_cb_footprint import (
    enforce_whole_artifact_budget,
    whole_artifact_budget_from_assignment_payload,
)

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
NVFP4_MAX = _nvfp4_activation_contract.FP4_E2M1_MAX
FP8_E4M3_MAX = _nvfp4_activation_contract.FP8_E4M3_MAX
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
_DO_NO_HARM_STATS: Counter = Counter()


def _record_do_no_harm_revert(fmt: str) -> None:
    _DO_NO_HARM_STATS[f"{fmt}_reverts"] += 1


def _record_do_no_harm_failure(fmt: str, linear_name: str | None, exc: Exception) -> None:
    _DO_NO_HARM_STATS[f"{fmt}_failures"] += 1
    name = linear_name or "<unknown>"
    print(
        f"[do-no-harm] WARN {name} {fmt} gate failed: {exc!r}",
        flush=True,
    )

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


# The NVFP4 codebook is a CONSTANT -- the eight positive E2M1 levels -- but it
# was being rebuilt from a Python list on every call, and the GPTQ render calls
# it once per column-quantize. Measured on one 5120x5120 Linear with
# gptq+static_act_order+joint_scale_opt: 11,562 calls, 2.279 s of a 3.705 s
# render. Building a CUDA tensor from a Python list is ~200 us (Python-side
# element walk, then an H2D copy), so 62% of the render was materializing 8
# floats over and over while the GPU sat at 33 W.
#
# Memoizing is safe because the value never varies and both call sites are
# strictly read-only (`cb[idx]`, `torch.bucketize(x, cb)`, `cb.numel()`); no
# caller mutates or keeps an escaping alias. Keyed on (device, dtype) so a
# mixed-device run cannot be handed a tensor from the wrong device, which would
# be a silent correctness bug rather than a slow path.
#
# The cached tensor must be BIT-IDENTICAL to the freshly built one -- this runs
# inside the production render, and a render that differs from the exported
# bytes is the rendering confound principle 8 exists to prevent. Pinned by
# tests/test_nvfp4_codebook_cache.py, which compares cached against fresh and
# renders a whole Linear both ways.
_NVFP4_CODEBOOK_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _nvfp4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    cb = _NVFP4_CODEBOOK_CACHE.get(key)
    if cb is None:
        cb = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=dtype)
        _NVFP4_CODEBOOK_CACHE[key] = cb
    return cb


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
    *,
    global_real: torch.Tensor | None = None,
) -> torch.Tensor:
    if global_real is not None:
        eff_scale = _nvfp4_effective_scale_from_real(
            scale,
            global_real,
            quantize_fp8=True,
        ).unsqueeze(-1)
    else:
        eff_scale = scale.unsqueeze(-1)
    _idx, dq = _nvfp4_quantize_dequantize_with_eff_scale(
        grouped,
        eff_scale,
    )
    return (grouped - dq).pow(2).sum(dim=-1)


def _nvfp4_best_max_to_level_scale(
    grouped: torch.Tensor,
    levels: Sequence[float],
    *,
    global_real: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick the best max-to-codebook-level scale for each NVFP4 group.

    ``four_over_six_mse`` is the two-level set ``levels=(6, 4)``. The joint
    scale rule uses ``_NVFP4_JOINT_SCALE_LEVELS`` (also ``(6, 4)`` by default;
    extendable via ``PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS``) and stays
    final-pack compatible because every chosen scale is
    ``max_abs / codebook_level``.
    """

    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    best_scale: torch.Tensor | None = None
    best_mse: torch.Tensor | None = None
    for level in levels:
        scale = max_abs / float(level)
        mse = _nvfp4_mse_for_group_scale(
            grouped,
            scale,
            global_real=global_real,
        )
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


def _parse_joint_scale_levels() -> tuple[float, ...]:
    """JSO per-group scale levels. Default ``(6.0, 4.0)`` — the FourOverSix
    pair. On both Qwen3.5-0.8B and Gemma4-31B the full 7-level grid collapses
    to {6,4} for 99.998% of groups (aggregate weight-MSE cost of restricting to
    {6,4} = +0.009%). The rare residual is self-correcting at allocation time:
    the format allocator scores cost under this same recipe and {6,4} ⊆ the
    full grid ⇒ a group's cost is monotone non-decreasing under the trim, so a
    genuinely-hurt Linear can only be *promoted* to FP8/BF16, never silently
    degraded. Set ``PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`` (comma/space
    separated, e.g. ``"6,4,3,2,1.5,1,0.5"``) to restore the full grid."""
    raw = os.environ.get("PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS")
    if raw:
        try:
            levels = tuple(float(x) for x in raw.replace(",", " ").split())
        except ValueError:
            levels = ()
        if levels:
            return levels
    return (6.0, 4.0)


_NVFP4_JOINT_SCALE_LEVELS = _parse_joint_scale_levels()


def _select_nvfp4_group_scales(
    grouped: torch.Tensor,
    *,
    scale_rule: str | None = None,
    global_real: torch.Tensor | None = None,
    joint_scale_levels: tuple[float, ...] | None = None,
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
        return _nvfp4_best_max_to_level_scale(
            grouped,
            (6.0, 4.0),
            global_real=global_real,
        )
    if rule == NVFP4_SCALE_RULE_JOINT_MSE:
        return _nvfp4_best_max_to_level_scale(
            grouped,
            (
                _NVFP4_JOINT_SCALE_LEVELS
                if joint_scale_levels is None
                else joint_scale_levels
            ),
            global_real=global_real,
        )
    raise AssertionError(f"unhandled NVFP4 scale rule: {rule!r}")


def _nvfp4_snapped_scale_scoring_enabled() -> bool:
    """Score NVFP4 scale candidates under the FP8-SNAPPED effective scale.

    Research lever (default OFF). Scoring under the snapped scale the
    served kernel actually uses is more faithful in principle, but it
    changes the shipped NVFP4 bytes for the joint_mse (JSO, a production
    default lever) and four_over_six rules and has not cleared a served
    gold-metric A/B (QC finding on review-batch M21). Promote per the
    ladder: served KL+PPL at matched bpp on 4B, then a 27B confirmation.
    """
    return os.environ.get(
        "PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING", "0") != "0"


def _select_nvfp4_pack_scales_and_global(
    grouped: torch.Tensor,
    *,
    global_real_override: torch.Tensor | None = None,
    scale_rule: str | None = None,
    snapped_scale_scoring: bool | None = None,
    joint_scale_levels: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    score_snapped = (
        _nvfp4_snapped_scale_scoring_enabled()
        if snapped_scale_scoring is None
        else bool(snapped_scale_scoring)
    )
    if not score_snapped:
        # Pre-2026-06-12 behavior (byte-stable with shipped artifacts):
        # scales scored under the RAW real scale; the tensor global is
        # derived once from the chosen scales (or taken verbatim from the
        # fused-sibling override) without re-scoring.
        scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            joint_scale_levels=joint_scale_levels,
        )
        if global_real_override is not None:
            global_real = global_real_override.to(
                grouped.device, dtype=torch.float32).clamp_min(1e-12)
        else:
            global_real = (scale.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
        return scale, global_real
    scale = _select_nvfp4_group_scales(
        grouped,
        scale_rule=scale_rule,
        joint_scale_levels=joint_scale_levels,
    )
    if global_real_override is not None:
        global_real = global_real_override.to(
            grouped.device,
            dtype=torch.float32,
        ).clamp_min(1e-12)
        scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            global_real=global_real,
            joint_scale_levels=joint_scale_levels,
        )
        return scale, global_real

    global_real = (scale.amax() / FP8_E4M3_MAX).clamp_min(1e-12)
    for _ in range(3):
        snapped_scale = _select_nvfp4_group_scales(
            grouped,
            scale_rule=scale_rule,
            global_real=global_real,
            joint_scale_levels=joint_scale_levels,
        )
        next_global = (
            snapped_scale.amax() / FP8_E4M3_MAX
        ).clamp_min(1e-12)
        scale = snapped_scale
        if torch.allclose(next_global, global_real, rtol=0.0, atol=1e-12):
            break
        global_real = next_global
    return scale, global_real


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
            global_real=global_real,
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


DEFAULT_INPUT_GLOBAL_SCALE = (
    _nvfp4_activation_contract.UNCALIBRATED_INPUT_GLOBAL_SCALE
)  # uncalibrated fallback; matches
# compressed-tensors' 1.0 "no global scaling" fallback for uncalibrated or
# degenerate nonpositive tensors.

# Compatibility aliases retained for external callers/tests.  The versioned
# activation-contract module is the sole owner of their values and policy.
_FP4_E2M1_MAX = _nvfp4_activation_contract.FP4_E2M1_MAX
_FP8_E4M3_MAX = _nvfp4_activation_contract.FP8_E4M3_MAX


def _nvfp4_input_gscale_fp8_range_enabled() -> bool:
    """Whether input_global_scale uses the compressed-tensors/vLLM
    convention ``FP8_MAX * FP4_MAX / amax`` (generate_gparam) instead of
    the legacy ``FP4_MAX / amax``.

    Default OFF (legacy bytes). The convention places serve-time
    FP8-stored activation block scales in (0, 448] instead of (0, 1] —
    rescuing blocks >64x below calibration amax from FP8 subnormals at
    the cost of CLIPPING any serve block whose amax exceeds calibration
    amax. Served A/Bs 2026-07-02 (weights byte-identical, only
    input_global_scale x448): Qwen3.6-35B-A3B MoE frontier -14.1% KL
    (WIN), LFM2.5 thin-calib smoke +5.8% (loss), Qwen3.6-27B regen dense
    +37.5% (LOSS). Strongly artifact-dependent, so the default stays
    backwards-compatible; set ``PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=1``
    only behind a per-artifact served A/B (the scale is a free
    post-export knob: patch input_global_scale in place, re-measure).
    """
    return (
        _nvfp4_activation_contract.resolve_input_global_scale_policy()
        == _nvfp4_activation_contract.FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    )


def _nvfp4_input_global_scale_from_max_abs(max_abs: float) -> float:
    """input_global_scale for a calibrated activation ``max_abs``.

    The shared policy owner selects the legacy ``6 / amax`` bytes by default
    or the explicit compressed-tensors ``448 * 6 / amax`` opt-in.  This
    compatibility wrapper never defines a second formula.
    """
    return _nvfp4_activation_contract.input_global_scale_from_max_abs(
        max_abs,
        policy=(
            _nvfp4_activation_contract.resolve_input_global_scale_policy()
        ),
        nonpositive_fallback=(
            _nvfp4_activation_contract.UNCALIBRATED_INPUT_GLOBAL_SCALE
        ),
    )


def compute_nvfp4_input_global_scale(activations: torch.Tensor) -> float:
    """Per-tensor input_global_scale from cached activations.

    Activations can have any shape.  The shared activation-contract policy
    converts their maximum absolute value and preserves the legacy all-zero
    fallback required for byte-compatible native export.
    """
    max_abs = float(activations.detach().abs().max().item())
    return _nvfp4_input_global_scale_from_max_abs(max_abs)


# Module-level cache populated by main() when --activation-cache-dir is
# provided. `_quantize_2d`'s NVFP4 branch consults it by recipe-name
# when no explicit override is passed in. Keyed by the recipe name
# (post-profile.live_to_recipe_name remap). None means "not computed".
_INPUT_GLOBAL_SCALES: dict[str, float] | None = None


def _resolve_nvfp4_input_global_scale(
    override: float | None = None,
    *,
    target: str | None = None,
) -> float:
    """Legacy native-export compatibility delegate.

    The uncalibrated fallback preserves existing native artifact bytes.  Its
    presence is also why this exporter does not claim the versioned Gridbook
    fused-W4A4 activation contract.
    """

    return _nvfp4_activation_contract.resolve_input_global_scale_value(
        override,
        target=target,
        calibrated_scales=_INPUT_GLOBAL_SCALES,
        allow_uncalibrated_fallback=True,
    )

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
    """Cache keys export will actually READ, and which of them are absent.

    Passthrough formats are excluded, not just BF16: their emit paths copy
    source bytes and never call `_pack_production_cached_2d` (the
    FP8_SOURCE branch in the streaming Linear loop returns before the
    cache lookup). `build_production_cache --render-scope assignment` does
    render them — FP8_SOURCE's `quantize_dequantize` is identity — so the
    keys usually exist, but demanding a render this export cannot consume
    would fail an otherwise-valid FP8-source export over an unused entry.
    Before issue #29 the question never arose: the runtime-legality guard
    rewrote every FP8_SOURCE entry to BF16 before this check saw it.
    """
    from prismaquant.production_weight_cache import is_uncached_packed_expert_qname

    keys: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for qname, fmt in assignment.items():
        cache_fmt = str(fmt).upper()
        canonical = _canonical_export_format(cache_fmt)
        if canonical in PASSTHROUGH_SOURCE_REQUIREMENTS:
            continue
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is None:
            if (
                _ALLOW_PACKED_EXPERT_RTN
                and is_uncached_packed_expert_qname(qname)
            ):
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
    # Hook enumeration the shipped bytes were rendered against (#147,
    # consumer 3): a shipped artifact records which enumeration its bytes
    # saw, so a cost table priced from one rendering cannot be silently
    # served from another. ``None`` for caches that predate the stamp.
    from prismaquant.production_weight_cache import activation_hook_scope_of

    return {
        "path": str(Path(cache_dir).resolve()) if cache_dir else None,
        "n_entries": len(rows),
        "hash": digest,
        "activation_max_abs_hash": act_digest,
        "metadata_hash": metadata_digest,
        "levers": dict(getattr(cache, "levers", {}) or {}),
        "activation_hook_scope": activation_hook_scope_of(cache),
    }


def _production_cache_scales(cache, *, profile=None) -> dict[str, float]:
    activation_max_abs = getattr(cache, "activation_max_abs", None) or {}
    scales = {
        name: _nvfp4_input_global_scale_from_max_abs(float(max_abs))
        for name, max_abs in activation_max_abs.items()
        if max_abs and float(max_abs) > 0.0
    }
    return _unify_input_global_scales_across_fused_siblings(
        scales,
        profile=profile,
    )



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


class _RuntimeCoercion(NamedTuple):
    """One Linear the runtime-legality guard rewrote to BF16.

    Positionally tuple-compatible with the legacy ``(name, shape,
    from_fmt)`` row this function used to return: `_bf16_upgrade_audit`
    and the manifest read the first three fields, so old readers keep
    working while the serving-group fields carry WHY a Linear the
    allocator chose a quantized format for is shipping unquantized.
    """
    name: str
    shape: list[int] | None
    from_fmt: str
    reason: str = ""
    detail: str = ""
    # Set only when the coercion was forced by serving-atomicity rather
    # than by this Linear's own legality verdict.
    serving_group: str | None = None
    serving_group_kind: str | None = None
    serving_group_members: tuple[str, ...] = ()
    trigger: str | None = None
    delta_bytes: int | None = None


@dataclass(frozen=True)
class _ServingUnit:
    """A set of Linears that must carry ONE format to be servable.

    ``kind`` is ``fused_siblings`` (vLLM merges q/k/v, gate/up into one
    packed Linear with one scheme) or ``packed_moe_experts`` (vLLM's
    ``CompressedTensorsMoEMethod`` selects one scheme per FusedMoE
    layer). Both are the hard serving invariants of CLAUDE.md §6.
    """
    kind: str
    key: str
    members: tuple[str, ...]


def _serving_atomic_units(
    names: Iterable[str],
    profile,
) -> tuple[tuple[_ServingUnit, ...], dict[str, str]]:
    """Serving-atomic units among ``names``, from the profile accessors.

    Grouping is asked of the shared activation-contract policy through the
    compatibility delegate `_fused_group_key_for_name` (the same
    `fused_sibling_group` / `fused_sibling_leaf_mapping` chain the
    fused-coherence gate in `build_quantization_config` derives its
    sibling sets from) and `profile.packed_expert_format_group` (what
    that gate's `_packed_format_group_members` uses). Nothing here
    parses Linear names: a new architecture declares its couplings once,
    in its profile/structure spec, and every consumer sees them.

    Returns the units with >= 2 members present (a lone present member
    cannot disagree with anything) plus, for fail-closed handling, the
    names whose packed-expert grouping accessor RAISED — a profile that
    cannot answer "which unit is this expert projection in" must not be
    silently treated as "no unit".
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    failures: dict[str, str] = {}
    packed_getter = (
        getattr(profile, "packed_expert_format_group", None)
        if profile is not None
        else None
    )
    for name in names:
        fused_key = _fused_group_key_for_name(name, profile)
        if fused_key:
            grouped.setdefault(("fused_siblings", str(fused_key)), []).append(name)
        if not callable(packed_getter):
            continue
        try:
            packed_key = packed_getter(name)
        except Exception as exc:  # PackedExpertRoleUnknown and friends
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        if packed_key:
            grouped.setdefault(
                ("packed_moe_experts", str(packed_key)), []
            ).append(name)
    units = tuple(
        _ServingUnit(kind, key, tuple(sorted(members)))
        for (kind, key), members in sorted(grouped.items())
        if len(members) >= 2
    )
    return units, failures


def _serving_atomic_components(
    names: Iterable[str],
    profile,
) -> tuple[dict[str, list[str]], dict[str, tuple[_ServingUnit, ...]], dict[str, str]]:
    """Connected components of the serving-atomic units over ``names``.

    Units are unioned rather than handled one at a time because they can
    overlap: on the split per-expert representation
    ``...experts.7.gate_proj`` is both a fused sibling of
    ``...experts.7.up_proj`` and a member of the layer's packed-expert
    unit. Coercing one unit at a time could then still leave the other
    mixed. This mirrors the allocator's own union-find serving-unit
    promotion (`allocator_solver._promote_group_components`).

    Returns ``(members_by_name, units_by_name, grouping_failures)`` where
    the first two are keyed by EVERY name in its component (so a lookup
    needs no root bookkeeping at the call site).
    """
    all_names = list(names)
    units, failures = _serving_atomic_units(all_names, profile)
    parent = {name: name for name in all_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for unit in units:
        members = [m for m in unit.members if m in parent]
        for member in members[1:]:
            ra, rb = find(members[0]), find(member)
            if ra != rb:
                parent[rb] = ra

    members_by_root: dict[str, list[str]] = {}
    for name in all_names:
        members_by_root.setdefault(find(name), []).append(name)
    units_by_root: dict[str, list[_ServingUnit]] = {}
    for unit in units:
        present = [m for m in unit.members if m in parent]
        if present:
            units_by_root.setdefault(find(present[0]), []).append(unit)

    members_by_name: dict[str, list[str]] = {}
    units_by_name: dict[str, tuple[_ServingUnit, ...]] = {}
    for root, members in members_by_root.items():
        component = sorted(members)
        component_units = tuple(units_by_root.get(root, ()))
        for name in members:
            members_by_name[name] = component
            units_by_name[name] = component_units
    return members_by_name, units_by_name, failures


def _bf16_coercion_delta_bytes(
    shape: Sequence[int] | None,
    from_fmt: str,
) -> int | None:
    """Bytes a Linear GAINS by shipping BF16 instead of ``from_fmt``."""
    if not shape:
        return None
    from .format_registry import get_format

    try:
        src = get_format(from_fmt)
        bf16 = get_format("BF16")
    except KeyError:
        return None
    shape_t = tuple(int(dim) for dim in shape)
    return int(
        bf16.memory_bytes_for_shape(shape_t) - src.memory_bytes_for_shape(shape_t)
    )


def _human_bytes(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unknown"
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} GiB"


def _group_legal_quantized_formats(
    shapes: dict[str, list[int] | None],
    members: Sequence[str],
    target_profile: str,
) -> list[tuple[str, float]]:
    """Emittable QUANTIZED formats legal for every checkable member.

    "Checkable" is the same bar the coercion itself uses: a member with a
    known 2-D source shape. Passthrough formats are excluded: BF16 is the
    fallback under discussion, and FP8_SOURCE is a *source-dependent*
    rung (legal only where every member's source is already fp8), so
    offering it as a substitute would flip refuse-vs-coerce on FP8-source
    models — a unit that must fall back to BF16 today would instead be
    refused. That rung is already in the allocator's menu, which is where
    picking it belongs; export names alternatives, it does not choose.

    A non-empty answer is the decisive fact for refuse-vs-coerce: it says
    the allocation is repairable upstream at a quantized bit rate, so
    rewriting the whole serving unit to 16 bpp is NOT what a
    shape-aware allocator would have produced.
    """
    from .format_registry import get_format

    checkable = [
        (name, tuple(int(d) for d in shapes[name]))
        for name in members
        if shapes.get(name) is not None and len(shapes[name] or ()) == 2
    ]
    if not checkable:
        return []
    legal: list[tuple[str, float]] = []
    for fmt in sorted(EXPORTABLE_FORMATS - set(PASSTHROUGH_SOURCE_REQUIREMENTS)):
        if not all(
            check_format_applicability(
                shape,
                fmt,
                qname=name,
                target_profile=target_profile,
            ).legal
            for name, shape in checkable
        ):
            continue
        try:
            bits = get_format(fmt).effective_bits_for_shape(checkable[0][1])
        except KeyError:
            bits = float("nan")
        legal.append((fmt, float(bits)))
    return sorted(legal, key=lambda row: (row[1], row[0]))


def _coerce_runtime_legal_assignment(
    src_model: str,
    assignment: dict[str, str],
    profile=None,
) -> tuple[dict[str, str], list[_RuntimeCoercion]]:
    """Adjust assignments that the target runtime cannot execute.

    Shape and format legality comes from serving-profile config. BF16 is the
    conservative runtime fallback when a format this exporter CAN emit turns
    out to be illegal on this Linear's shape or denied by the resolved
    serving profile.

    A format the exporter cannot emit at all (not in `EXPORTABLE_FORMATS`)
    is NOT coerced — it raises. Rewriting it to BF16 would ship that Linear
    at 16 bpp, blowing the byte budget the allocation was selected under and
    producing an artifact whose real bpp disagrees with its own
    `layer_config.json`. The serving profile's export lane already bounds
    the allocator's menu by this exporter's declaration, so reaching here
    means that bound regressed; masking it with a rewrite is the
    post-allocator band-aid CLAUDE.md §4.1 vetoes.

    **Serving-atomic groups (issue #28).** The coercion is group-aware. A
    packed-MoE expert unit and a fused-sibling set must carry ONE format
    (vLLM selects one scheme per FusedMoE layer / per merged-column
    Linear), and their members do NOT share a shape — an odd
    `moe_intermediate_size` can make `down_proj` indivisible while
    `gate_up_proj` is fine. Rewriting only the offending member would
    produce a quantized + BF16 mix inside one serving unit: a
    hard-serving-invariant violation that `build_quantization_config`'s
    fused-coherence gate then reports as a wrong-model-profile problem it
    is not. So when a member of a serving unit is illegal, this function
    resolves the WHOLE unit, and never leaves a unit mixed:

      * if some emittable QUANTIZED format is legal for every member, it
        **raises**. Coercing would ship the whole unit at 16 bpp — for a
        packed-expert unit that is `num_experts x` the per-Linear cost
        this branch is justified by, and the model-wide dimension that
        made one member illegal makes it every layer's unit — when a
        re-solve lands the unit on that legal format for free. Export
        must not pick the substitute itself: the format it picked is the
        one the production weight cache holds a deliberate render for
        (a substitute is a cache miss at best, an RTN render at worst),
        so naming the legal rung and refusing is the honest move.
      * if NO quantized format is legal for every member, BF16 is not a
        band-aid but the only representable answer — exactly what a
        shape-aware allocator would have had to pick — so the whole unit
        is coerced, loudly, and every member is recorded in the returned
        rows (and from there in `runtime_coercions` +
        `bf16_audit.serving_group`).

    **Passthrough source integrity (issue #29).**
    `PASSTHROUGH_SOURCE_REQUIREMENTS` makes FP8_SOURCE legal only where the
    source tensor is ALREADY fp8, which `check_format_applicability` can
    only judge with a `source_kind`. That argument used to be omitted, so
    every FP8_SOURCE Linear came back `source_dtype_mismatch` and was
    rewritten to BF16 —
    inert in the bytes (materialization copies the source fp8 verbatim and
    `_fp8_source_config_overlay` restores the config), but it filled every
    DSv4 / Hy3 / MiniMax manifest's `runtime_coercions` with rows for
    demotions that never happened, hiding any real one. The `source_kind`
    now comes from `_scan_source_dtype_manifest` — the SAME recipe-keyed
    map `build_candidates` gates the allocator's passthrough candidates
    on — so export's verdict and the gate that admitted the allocation
    cannot disagree, and a passthrough row now means the source really is
    not fp8 (i.e. an upstream `PASSTHROUGH_SOURCE_REQUIREMENTS` failure)
    and the bytes really do change. Such a verdict is therefore treated
    like any other illegality, group escalation included.

    Upstream is the real fix: the allocator's candidate mask intersects
    shape legality per member, so a promoted format is legal for every
    member by construction and this path is unreachable in normal
    operation. Treat any firing as an upstream regression worth reporting
    — `_runtime_coercion_report` is written to be impossible to miss.
    """
    out = dict(assignment)
    target_profile = _allocator_target_profile_for_audit(profile) or "research"
    shapes: dict[str, list[int] | None] = {}
    # Recipe-keyed source dtypes, read lazily: only passthrough formats
    # consult `source_kind` inside `check_format_applicability`, and the
    # scan is safetensors-header IO that a BF16-source export never needs
    # (CLAUDE.md §4.7 — no disk work on a path that cannot use it).
    source_kinds: dict[str, str] | None = None

    def _source_kind_for(qname: str) -> str | None:
        nonlocal source_kinds
        if source_kinds is None:
            source_kinds = _scan_source_dtype_manifest(src_model, profile)
        return source_kinds.get(qname)

    # qname -> (shape, from_fmt, reason, detail)
    illegal: dict[str, tuple[list[int], str, str, str]] = {}
    for qname, fmt in assignment.items():
        fmt_canonical = _canonical_export_format(fmt)
        out[qname] = fmt_canonical
        if fmt_canonical == "BF16":
            continue
        if fmt_canonical not in EXPORTABLE_FORMATS:
            from prismaquant.gguf_formats import GGUF_BLOCK_BYTES
            from prismaquant.cb_layout import CB_FORMAT_NAMES

            if fmt_canonical in CB_FORMAT_NAMES:
                # Wrong container: an NVFP4-CB / FP8-CB assignment reaching the
                # compressed-tensors exporter means the pipeline was launched
                # without EXPORT_CONTAINER=nvfp4_cb. Stock compressed-tensors
                # schemes cannot express codebooks; coercing to BF16 would ship
                # a ~16 bpp artifact unrelated to the allocated budget.
                raise ValueError(
                    f"{qname}: format {fmt_canonical} ships via the nvfp4_cb "
                    f"container (prismaquant.export_nvfp4_cb / "
                    f"EXPORT_CONTAINER=nvfp4_cb), not compressed-tensors"
                )
            if fmt_canonical in GGUF_BLOCK_BYTES:
                # Wrong container, not a research format: a GGUF assignment
                # reaching the compressed-tensors exporter means the pipeline
                # was launched without EXPORT_CONTAINER=gguf. Silently
                # coercing to BF16 would ship a ~16 bpp artifact unrelated to
                # the allocated budget.
                raise ValueError(
                    f"{qname}: format {fmt_canonical} ships via the GGUF "
                    f"container (prismaquant.export_gguf / "
                    f"EXPORT_CONTAINER=gguf), not compressed-tensors"
                )
            raise ValueError(
                f"{qname}: format {fmt_canonical} has no compressed-tensors "
                f"emit path -- it is absent from FORMAT_SCHEME, so there is "
                f"no `config_groups` scheme for vLLM to dispatch on "
                f"(emittable={sorted(EXPORTABLE_FORMATS)}). Rewriting it to "
                f"BF16 here would ship this Linear at 16 bpp and silently "
                f"blow the byte budget the allocation was selected under, "
                f"leaving the artifact's real bpp disagreeing with its own "
                f"layer_config.json. The serving profile that admitted it "
                f"(target_profile={target_profile!r}) is supposed to bound "
                f"its menu by this exporter's EXPORTABLE_FORMATS via its "
                f"export lane, so this is a regression in that bound (or an "
                f"allocation solved under a different profile than "
                f"PRISMAQUANT_TARGET_PROFILE resolves to here) -- re-solve "
                f"the allocation under the serving profile you are exporting "
                f"for rather than letting export rewrite it."
            )
        shape = _source_weight_shape_for_recipe(src_model, qname, profile)
        shapes[qname] = shape
        if shape is None or len(shape) != 2:
            continue
        verdict = check_format_applicability(
            tuple(shape),
            fmt,
            qname=qname,
            source_kind=(
                _source_kind_for(qname)
                if fmt_canonical in PASSTHROUGH_SOURCE_REQUIREMENTS
                else None
            ),
            target_profile=target_profile,
        )
        if not verdict.legal:
            illegal[qname] = (
                shape,
                fmt_canonical,
                verdict.reason or "illegal",
                verdict.detail or "",
            )

    if not illegal:
        return out, []

    coerced: list[_RuntimeCoercion] = []
    handled: set[str] = set()

    # No passthrough exemption (issue #29). It existed only because the
    # FP8_SOURCE verdict was an artifact of the missing `source_kind` —
    # escalating a bogus demotion would have coerced whole packed-expert
    # units to BF16 on every FP8-source model. Now that the verdict is
    # source-aware, a passthrough mismatch is a real illegality with a
    # real byte cost, and gets the same serving-atomic treatment as a
    # shape or policy one: refused when the unit has a legal quantized
    # rung, coerced as a whole unit when it does not.
    members_by_name, units_by_name, grouping_failures = (
        _serving_atomic_components(out.keys(), profile)
    )
    if grouping_failures:
        # Fail closed. A name whose serving-unit accessor raised is absent
        # from every unit, so "coerce the whole unit" would silently leave
        # it behind -- i.e. produce exactly the mixed FusedMoE this branch
        # exists to prevent. An undeclared packed-expert role is a profile
        # declaration gap; guessing it is how you ship an allocation nobody
        # selected.
        raise ValueError(
            f"cannot verify serving-atomic coherence: the model profile "
            f"could not name the serving unit for "
            f"{len(grouping_failures)} Linear(s), and "
            f"{len(illegal) - len(handled)} Linear(s) need a runtime-legality "
            f"coercion that must be applied to a whole unit or not at all. "
            + "; ".join(
                f"{name} ({err})"
                for name, err in sorted(grouping_failures.items())[:8]
            )
            + ". Declare the packed-expert projection roles for this "
            "architecture (model_profiles/specs/*.json "
            "`packed_experts.projection_splits`) rather than letting export "
            "guess which projections share a FusedMoE scheme."
        )
    refusals: list[str] = []
    for qname in sorted(illegal):
        if qname in handled:
            continue
        shape, from_fmt, reason, detail = illegal[qname]
        units = units_by_name.get(qname, ())
        if not units:
            # Not serving-atomic: coerce this Linear alone, as before.
            handled.add(qname)
            out[qname] = "BF16"
            coerced.append(
                _RuntimeCoercion(
                    qname, shape, from_fmt, reason, detail,
                    delta_bytes=_bf16_coercion_delta_bytes(shape, from_fmt),
                )
            )
            continue
        members = members_by_name.get(qname, [qname])
        co_triggers = sorted(m for m in members if m in illegal)
        handled.update(co_triggers)
        alternatives = _group_legal_quantized_formats(
            shapes, members, target_profile
        )
        # Name the component by its WIDEST unit — the FusedMoE / merged
        # column that actually constrains it. Concatenating every unit key
        # would put a 128-expert layer's whole key list on every row of
        # the manifest for no extra information: `serving_group_members`
        # already carries the exact membership.
        widest = max(units, key=lambda unit: (len(unit.members), unit.key))
        group_key = widest.key
        group_kind = "+".join(sorted({unit.kind for unit in units}))
        would_cost = sum(
            delta
            for delta in (
                _bf16_coercion_delta_bytes(
                    shapes.get(member), _canonical_export_format(out[member])
                )
                for member in members
                if _canonical_export_format(out[member]) != "BF16"
            )
            if delta is not None
        )
        if alternatives:
            refusals.append(
                _describe_serving_group_refusal(
                    trigger=qname,
                    shape=shape,
                    from_fmt=from_fmt,
                    reason=reason,
                    detail=detail,
                    group_key=group_key,
                    group_kind=group_kind,
                    members=members,
                    co_triggers=co_triggers,
                    alternatives=alternatives,
                    would_cost=would_cost,
                    target_profile=target_profile,
                )
            )
            continue
        # BF16 is the only representable format left for this unit: it is
        # what a shape-aware allocator must pick, so take it for the WHOLE
        # unit. Never one member — that is the mixed-scheme artifact.
        for member in members:
            member_fmt = _canonical_export_format(out[member])
            if member_fmt == "BF16":
                continue
            member_shape = shapes.get(member)
            out[member] = "BF16"
            if member in illegal:
                member_reason, member_detail = illegal[member][2], illegal[member][3]
            else:
                member_reason = "serving_group_coherence"
                member_detail = (
                    f"coerced with its serving-atomic unit; {qname} is "
                    f"illegal for {from_fmt} ({reason})"
                )
            coerced.append(
                _RuntimeCoercion(
                    member,
                    member_shape,
                    member_fmt,
                    member_reason,
                    member_detail,
                    group_key,
                    group_kind,
                    tuple(members),
                    qname,
                    _bf16_coercion_delta_bytes(member_shape, member_fmt),
                )
            )

    if refusals:
        raise ValueError(
            f"serving-atomic group is not runtime-legal for its allocated "
            f"format in {len(refusals)} unit(s). vLLM selects ONE scheme per "
            f"FusedMoE layer and one per merged-column Linear, so every "
            f"member of these units must carry ONE format; export refuses to "
            f"rewrite them rather than ship a mixed (unservable) unit or a "
            f"whole unit at 16 bpp when a quantized format legal for every "
            f"member exists. "
            + " || ".join(refusals)
        )
    return out, coerced


def _describe_serving_group_refusal(
    *,
    trigger: str,
    shape: Sequence[int] | None,
    from_fmt: str,
    reason: str,
    detail: str,
    group_key: str,
    group_kind: str,
    members: Sequence[str],
    co_triggers: Sequence[str],
    alternatives: Sequence[tuple[str, float]],
    would_cost: int,
    target_profile: str,
) -> str:
    """One refusal clause: what is illegal, and what to do about it."""
    alt_text = ", ".join(f"{fmt} (~{bits:.3f} bpp)" for fmt, bits in alternatives)
    others = [name for name in co_triggers if name != trigger]
    parts = [
        f"unit {group_key!r} [{group_kind}] allocated {from_fmt} with "
        f"{len(members)} member(s): {trigger} is ILLEGAL at shape "
        f"{list(shape) if shape else None} "
        f"(reason={reason}; {detail or 'no detail'}) under "
        f"target_profile={target_profile!r}"
        + (f"; also illegal in this unit: {others}" if others else ""),
        f"but these emittable formats ARE legal for every member of the "
        f"unit: {alt_text}. Coercing the unit to BF16 instead would add "
        f"{_human_bytes(would_cost)} for nothing, and the "
        + ("source precision"
           if reason == "source_dtype_mismatch" else "dimension")
        + f" that made {trigger.rsplit('.', 1)[-1]} illegal is model-wide, "
        f"so it is that much per unit across every layer",
    ]
    parts.append(
        f"members={sorted(members)[:8]}"
        + (f" (+{len(members) - 8} more)" if len(members) > 8 else "")
    )
    parts.append(
        f"FIX: re-solve the allocation for target_profile={target_profile!r} "
        f"so this unit's promoted format is legal for every member -- "
        f"legality (shape, serving policy, and passthrough source precision) "
        f"has to be intersected ACROSS a serving unit, because that "
        f"unit is one scheme at serve time (issue #28); or, when "
        f"re-exporting an allocation that predates that, set every member of "
        f"the unit to one of the legal formats above in layer_config.json. "
        f"Export deliberately does not substitute the format itself: the "
        f"production weight cache holds a deliberate render for the format "
        f"that was ALLOCATED, so a substitution is a cache miss (or an RTN "
        f"render), and picking formats is the allocator's job"
    )
    return " -- ".join(parts)


def _runtime_coercion_report(coerced: Sequence[_RuntimeCoercion]) -> str:
    """Operator-facing report for runtime coercions, group-aware.

    A whole-serving-unit coercion means an allocation reached export with
    a format that is illegal for a member of a unit that must be uniform.
    Upstream is supposed to make that unrepresentable, so this is loud on
    purpose: it is a safety net firing, i.e. evidence of an upstream
    regression (or a pre-#28 `layer_config.json`), not routine hygiene.
    """
    if not coerced:
        return ""
    deltas = [
        row.delta_bytes for row in coerced
        if getattr(row, "delta_bytes", None) is not None
    ]
    lines = [
        f"[export-stream] runtime format coercions: {len(coerced)} Linears "
        f"-> BF16 (target runtime does not support those format/shape pairs)"
        + (
            f"; +{_human_bytes(sum(deltas))} over the allocated formats"
            if deltas else ""
        ),
    ]
    # A passthrough verdict is not a runtime-support fact but a broken
    # allocator contract: FP8_SOURCE was assigned to a Linear whose source
    # is not fp8, which `PASSTHROUGH_SOURCE_REQUIREMENTS` is supposed to
    # make unreachable (issue #29). Since #29 this can no longer be the
    # missing-`source_kind` false positive, so say so in those words.
    passthrough = [
        row for row in coerced
        if getattr(row, "reason", "") == "source_dtype_mismatch"
    ]
    if passthrough:
        lines.append(
            f"[export-stream] WARNING: {len(passthrough)} PASSTHROUGH "
            f"SOURCE MISMATCH(ES): a passthrough format was allocated to a "
            f"Linear whose source is not that precision, so the source bytes "
            f"cannot be copied and BF16 is the only representable answer. "
            f"PASSTHROUGH_SOURCE_REQUIREMENTS is supposed to make this "
            f"unreachable -- re-solve the allocation against this checkpoint "
            f"(the allocator's source manifest disagrees with the source "
            f"safetensors). "
            + str([(row.name, row.from_fmt, row.detail)
                   for row in passthrough[:6]])
        )
    groups: dict[str, list[_RuntimeCoercion]] = {}
    singles: list[_RuntimeCoercion] = []
    for row in coerced:
        key = getattr(row, "serving_group", None)
        if key:
            groups.setdefault(str(key), []).append(row)
        else:
            singles.append(row)
    if groups:
        lines.append(
            "[export-stream] " + "=" * 62
        )
        lines.append(
            f"[export-stream] WARNING: {len(groups)} SERVING-ATOMIC UNIT(S) "
            f"COERCED TO BF16 IN FULL."
        )
        lines.append(
            "[export-stream]   vLLM needs ONE format per FusedMoE layer / "
            "merged-column Linear, and no"
        )
        lines.append(
            "[export-stream]   emittable quantized format was legal for "
            "every member, so the whole unit"
        )
        lines.append(
            "[export-stream]   ships unquantized. The allocator is supposed "
            "to make this unreachable, so"
        )
        lines.append(
            "[export-stream]   treat it as an UPSTREAM REGRESSION (or a "
            "pre-#28 layer_config.json) and"
        )
        lines.append(
            "[export-stream]   report it: this artifact's real bpp EXCEEDS "
            "its own layer_config.json."
        )
        for key, rows in sorted(groups.items()):
            first = rows[0]
            delta = sum(
                row.delta_bytes for row in rows
                if row.delta_bytes is not None
            )
            lines.append(
                f"[export-stream]   unit {key} [{first.serving_group_kind}]: "
                f"{len(rows)} Linears -> BF16, +{_human_bytes(delta)}"
            )
            lines.append(
                f"[export-stream]     trigger={first.trigger} "
                f"from={first.from_fmt} reason={first.reason} "
                f"detail={first.detail}"
            )
            lines.append(
                "[export-stream]     members="
                + str(list(first.serving_group_members)[:8])
                + (
                    f" (+{len(first.serving_group_members) - 8} more)"
                    if len(first.serving_group_members) > 8
                    else ""
                )
            )
        lines.append("[export-stream] " + "=" * 62)
    if singles:
        lines.append(
            "[export-stream]   ungrouped: "
            + str([(row.name, row.shape, row.from_fmt) for row in singles[:6]])
        )
    return "\n".join(lines)


def _runtime_coercion_manifest_rows(
    coerced: Sequence[_RuntimeCoercion],
) -> list[dict[str, object]]:
    """`runtime_coercions` records for `mixed_native_manifest.json`."""
    rows: list[dict[str, object]] = []
    for row in coerced:
        entry: dict[str, object] = {
            "name": row[0],
            "shape": row[1],
            "from": row[2],
            "to": "BF16",
            "reason": getattr(row, "reason", "") or None,
            "detail": getattr(row, "detail", "") or None,
            "delta_bytes": getattr(row, "delta_bytes", None),
        }
        group = getattr(row, "serving_group", None)
        if group:
            entry["serving_group"] = {
                "key": group,
                "kind": row.serving_group_kind,
                "members": list(row.serving_group_members),
                "trigger": row.trigger,
            }
        rows.append(entry)
    return rows


# The serving profile the allocator actually solved with, read out of
# layer_config.json's reserved metadata block at load time (re-vet R11).
_ALLOCATOR_TARGET_PROFILE: str | None = None


def _allocator_target_profile_for_audit(profile) -> str | None:
    # Export must audit legality under the SAME serving profile the allocator
    # solved with. When they differ, export coerces every format the profile
    # it resolved does not serve — 2026-07-11: 226 dense FP8 Linears silently
    # -> BF16 on the Hy3 CT export, because the allocator ran vllm_packed_moe
    # while export re-resolved hy_v3's declared `gguf`.
    #
    # Precedence: PRISMAQUANT_TARGET_PROFILE (explicit operator override for
    # direct exporter invocations) > the allocator's stamp in
    # layer_config.json > the architecture spec default. The stamp travels
    # with the artifact, so the pipeline needs no env plumbing at all.
    requested = (
        os.environ.get("PRISMAQUANT_TARGET_PROFILE")
        or _ALLOCATOR_TARGET_PROFILE
        or None
    )
    if profile is None and requested is None:
        return None
    return resolve_target_profile(profile, requested)


def _bf16_passthrough_for_assignment(
    explicit_ignore: Sequence[str] | None,
    profile,
    allocator_meta: Mapping[str, object] | None,
) -> set[str]:
    """Resolve profile pins without undoing an explicit head assignment.

    Old layer configs carry no head metadata and retain the historical profile
    pin. New allocator configs stamp ``lm_head_mode``: ``fixed`` means the
    measured ``lm_head_format`` is auxiliary to body bpp, while ``dp`` means
    ``--allow-pinned lm_head`` selected it. In either case the assignment — not
    the exporter's default pin — is authoritative for that one head. An
    explicit ``--ignore`` remains the highest-precedence operator override.
    """
    if explicit_ignore is not None:
        return {str(name) for name in explicit_ignore}

    from .fixed_head import remaining_profile_pins

    meta = dict(allocator_meta or {})
    mode = str(meta.get("lm_head_mode") or "")
    fmt = str(meta.get("lm_head_format") or "BF16")
    lift_head = mode == "dp" or (
        mode == "fixed" and _canonical_export_format(fmt) != "BF16"
    )
    return set(remaining_profile_pins(
        profile,
        fixed_lm_head_quantized=lift_head,
    ))


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
    # Serving-unit coercions (issue #28) are reported as such: a whole
    # FusedMoE / merged-column unit shipping unquantized is a different
    # (and louder) fact than one Linear whose own shape was illegal.
    coerced_groups: dict[str, dict[str, object]] = {}
    for row in runtime_coerced:
        if len(row) >= 3:
            name, shape, from_fmt = row[:3]
        else:
            name, shape = row[:2]
            from_fmt = "MXFP8_E4M3"
        coerced[str(name)] = (shape, str(from_fmt))
        group_key = getattr(row, "serving_group", None)
        if group_key:
            coerced_groups[str(name)] = {
                "key": str(group_key),
                "kind": row.serving_group_kind,
                "members": list(row.serving_group_members),
                "trigger": row.trigger,
                "reason": getattr(row, "reason", "") or None,
                "detail": getattr(row, "detail", "") or None,
            }
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
                "runtime_coerced_"
                + ("serving_group_" if qname in coerced_groups else "")
                + "from_"
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
        group_entry = coerced_groups.get(qname)
        if group_entry is not None:
            entry["serving_group"] = group_entry
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
    mode: str | None = None,
) -> int:
    """Prefetch this layer's rendered weights out of the production cache.

    ``mode`` mirrors ``ProductionWeightCache.prefetch_assignment(require=…)``:
    under ``require`` a cache that cannot supply the assignment is a hard
    failure. Every failure path here used to return ``0`` and the caller only
    logged when it prefetched something, so a TOTAL miss was invisible and the
    export silently went NVMe-bound, tensor by tensor (re-vet R24 / debt D8).
    """
    mode = str(mode or _PRODUCTION_CACHE_PREFETCH_MODE or "warn").lower()
    require = mode == "require"

    def _refuse(reason: str) -> None:
        if require:
            raise RuntimeError(
                f"production-cache prefetch (require): {reason}. The export "
                "would fall back to per-tensor NVMe reads; pass "
                "--production-cache-prefetch warn to accept that explicitly."
            )

    cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None or not hasattr(cache, "prefetch"):
        _refuse("no production weight cache is installed")
        return 0
    keys: list[tuple[str, str]] = []
    quantized = 0
    for qname, fmt in assignment.items():
        if prefix is not None and not (qname == prefix or qname.startswith(prefix + ".")):
            continue
        cache_fmt = str(fmt).upper()
        if _canonical_export_format(cache_fmt) == "BF16":
            continue
        quantized += 1
        key = _production_cache_lookup_key(qname, cache_fmt)
        if key is not None:
            keys.append(key)
    if not keys:
        # No quantized entries under this prefix is legitimate (an all-BF16
        # layer); quantized entries with no cache keys is a total miss.
        if quantized:
            _refuse(
                f"{quantized} quantized entries under prefix {prefix!r} "
                "resolved to zero production-cache keys"
            )
        return 0
    loaded = int(cache.prefetch(keys, max_workers=_PRODUCTION_CACHE_PREFETCH_WORKERS))
    if loaded == 0:
        _refuse(
            f"{len(keys)} cache keys under prefix {prefix!r} loaded nothing"
        )
    return loaded


@contextmanager
def _temporary_export_nvfp4_scale_rule(rule_name: str | None):
    """Temporarily set the active NVFP4 scale rule for an export re-derive.

    No-op when ``rule_name`` is falsy (keeps the export-entry default)."""
    global _NVFP4_SCALE_RULE
    if not rule_name:
        yield
        return
    prev = _NVFP4_SCALE_RULE
    try:
        _NVFP4_SCALE_RULE = resolve_nvfp4_scale_rule(rule_name)
        yield
    finally:
        _NVFP4_SCALE_RULE = prev


def _export_match_render_scale_rule(cache) -> str | None:
    """M19: the NVFP4 scale rule the render used, for a byte-faithful re-derive.

    The production cache stores the render's fp32 weights as bf16; export then
    re-quantizes that dequant. Re-deriving under the render's RECORDED scale
    rule (e.g. ``joint_mse`` when JSO was on) instead of the export-entry
    default (``static_6``) makes the re-quant near-idempotent — the shipped
    NVFP4 bytes track the KL-validated render rather than diverging to a
    different per-group scale choice. Returns ``None`` (legacy behavior, bytes
    unchanged) when ``PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE`` is ``0`` or
    the cache carries no recorded rule.
    """
    if os.environ.get(
            "PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE", "1") == "0":
        return None
    levers = getattr(cache, "levers", None) or {}
    rule = levers.get("nvfp4_scale_rule")
    return str(rule) if rule else None


def _packed_expert_render_scale_rule(
    cache: "ProductionWeightCache | None" = None,
) -> str | None:
    """M2 (2026-07-02 audit): render scale rule for packed-expert re-derives.

    The packed-expert re-pack re-derives NVFP4 codes/scales from the cached
    3-D dequant exactly like the dense ``_pack_production_cached_2d`` path,
    so it needs the same M19 match-render-scale wrap: without it a
    joint_mse/four_over_six-rendered expert re-derived under the export-entry
    default (``static_6``) cannot recover its codes (measured 43% packed-byte
    flips). The cache's ``nvfp4_scale_rule`` lever is cache-global — the
    packed-expert render (both ``batched`` and ``per_expert`` modes in
    ``fill_packed_expert_cache_entries``) runs under the same module-level
    rule the lever records — so the cache-level lever IS the expert render's
    rule. Resolution: recorded rule if present, else ``None`` (the current
    env default; behavior unchanged when nothing is recorded). Residual:
    ``PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`` is not recorded in the lever
    dict, so non-default joint levels must still match between cache-build
    and export env. Gated by ``PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE``
    (the existing M19 flag, default on).

    ``cache`` overrides the module-level production cache: the inline-render
    export path passes its transient per-layer cache so the re-derive keys off
    the SAME recorded rule the inline render used.
    """
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    return _export_match_render_scale_rule(cache)


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
        with _temporary_export_nvfp4_scale_rule(
                _export_match_render_scale_rule(cache)):
            wp, ws, wg = quantize_dequantize_nvfp4(
                w_work,
                group_size=16,
                global_real_override=nvfp4_global_real_override,
            )
        input_scale = _resolve_nvfp4_input_global_scale(target=linear_name)
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


def _read_cached_packed_expert(
    experts_param_name: str,
    fmt: str,
    *,
    device: torch.device | None = None,
    cache: "ProductionWeightCache | None" = None,
) -> torch.Tensor | None:
    """Return the GPTQ-rendered 3-D dequant ``[E, out, in]`` for a packed-MoE
    expert tensor from the production cache, or ``None`` if absent.

    ``cache`` overrides the module-level production cache — the inline-render
    export path (``PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ``) passes a transient
    per-layer cache produced by ``fill_packed_expert_cache_entries`` so no
    588 GB dequant cache has to be materialized to disk first.

    Mirrors ``_pack_production_cached_2d`` but for packed experts: the cache
    stores the per-expert GPTQ dequant (in the model dtype, usually bf16).  The
    caller splits this into per-expert slices and re-packs each by re-deriving
    NVFP4 codes + per-group scales from the cached dequant — the SAME
    re-derive-from-dequant approximation the shipped 2-D path uses
    (``_pack_production_cached_2d``).  This is NOT bit-lossless (the re-derived
    group scales differ slightly from those used during the render, ~1e-3
    weight error), but it is the established production contract and within
    NVFP4 noise; served KL is the final arbiter.
    """
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    w = cache.get(experts_param_name, str(fmt).upper())
    if w is None:
        return None
    target_device = device or torch.device("cpu")
    return w.to(device=target_device, dtype=torch.float32)


def _packed_expert_input_global_scale(
    experts_param_name: str,
    *,
    cache: "ProductionWeightCache | None" = None,
) -> float | None:
    """Policy-resolved W4A4 input_global_scale for a packed-expert tensor,
    read from the production cache's per-param activation max_abs.

    Returns ``None`` when no cache/scale is available, so the caller falls back
    to ``DEFAULT_INPUT_GLOBAL_SCALE`` (only on the no-cache research path).

    ``cache`` overrides the module-level production cache (inline-render path).
    """
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    if cache is None:
        return None
    max_abs_map = getattr(cache, "activation_max_abs", None) or {}
    mx = max_abs_map.get(experts_param_name)
    if mx is None or float(mx) <= 0:
        return None
    return _nvfp4_input_global_scale_from_max_abs(float(mx))


def _packed_expert_stage_attestation(
    experts_param_name: str,
    *,
    cache: "ProductionWeightCache | None" = None,
    profile=None,
) -> dict[str, Any] | None:
    """Attest one packed FusedMoE module's w13/w2 stages (ROADMAP K0.2).

    This legacy container deliberately publishes no
    ``execution_contracts.nvfp4_w4a4`` record (its activation scalars are
    optional/defaultable and cannot carry the strict Gridbook fused-W4A4
    claim), but it must not be able to emit a routed-MoE artifact whose two
    stages were not both calibrated.  The section is built by the same shared
    builder both CB exporters use, from the same calibration-source vocabulary,
    so all three emit paths agree on stage identity, framing, and digests.

    Returns ``None`` for anything that is not a packed routed-expert stage.
    Raises when the sibling stage of the same FusedMoE module has no calibrated
    max-abs — the exact half-calibrated state that makes fused MoE fail closed
    at serving time.
    """
    parsed = _nvfp4_activation_contract.routed_moe_stage(
        experts_param_name, profile=profile
    )
    if parsed is None:
        return None
    module, stage = parsed
    if cache is None:
        cache = _PRODUCTION_WEIGHT_CACHE
    max_abs_map = (
        getattr(cache, "activation_max_abs", None) or {}
    ) if cache is not None else {}
    max_abs_by_target: dict[str, float] = {}
    for name, value in max_abs_map.items():
        found = _nvfp4_activation_contract.routed_moe_stage(
            str(name), profile=profile
        )
        if found is None or found[0] != module:
            continue
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            continue
        max_abs_by_target[str(name)] = float(value)
    staged = {
        _nvfp4_activation_contract.routed_moe_stage(
            name, profile=profile
        )[1]: name
        for name in sorted(max_abs_by_target)
    }
    missing = [
        s for s in _nvfp4_activation_contract.NVFP4_ROUTED_MOE_STAGES
        if s not in staged
    ]
    if missing:
        raise RuntimeError(
            f"[export-native] packed FusedMoE module {module!r} has a "
            f"calibrated {stage} activation scale but no calibrated "
            f"{missing} stage. A routed-MoE artifact must attest BOTH stages "
            "(w13 from the experts-module input, w2 from the routed "
            "intermediate); re-run build_production_cache with the packed "
            "experts in scope so the missing packed_expert_max_abs entry is "
            "recomputed."
        )
    policy = _nvfp4_activation_contract.resolve_input_global_scale_policy()
    return _nvfp4_activation_contract.build_routed_moe_stage_attestation(
        {
            name: _nvfp4_input_global_scale_from_max_abs(value)
            for name, value in max_abs_by_target.items()
        },
        policy=policy,
        calibration_sources={
            name: (
                _nvfp4_activation_contract
                .CALIBRATION_SOURCE_PACKED_EXPERT_RENDER
            )
            for name in max_abs_by_target
        },
        profile=profile,
    )


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
# Research/A-B escape hatch (Codex-recommended): when set, packed experts skip
# the production-cache GPTQ read and the RTN-by-omission hard-fail, exporting
# source RTN instead. ONLY for the served RTN-vs-GPTQ expert A/B — NEVER a
# production path (NVFP4 experts under RTN is a severe quality regression).
_ALLOW_PACKED_EXPERT_RTN = (
    os.environ.get("PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN", "0") == "1"
)
# Inline packed-expert GPTQ render at export time. When set, and NO production
# weight cache is active, packed experts are rendered ON THE FLY during the
# streaming export — one layer's stack at a time — through the SAME
# ``fill_packed_expert_cache_entries`` batched-GPTQ path the cache builder uses,
# sourcing the experts-module input snapshot from the probe's activation cache.
# This is the 295B-MoE path (Tencent Hy3): a full dequant cache is ~588 GB and
# cannot coexist with the 557 GiB source on a 1.8 TB disk, so we never write it.
# Unset (default) preserves the RTN-by-omission hard-fail exactly.
_INLINE_EXPERT_GPTQ = (
    os.environ.get("PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ", "0") == "1"
)
_PRODUCTION_CACHE_FINGERPRINT: dict[str, object] | None = None
_PRODUCTION_CACHE_PREFETCH_WORKERS = 4
# "require" | "warn" (D8). run-pipeline.sh passes require on the native
# lane, matching VALIDATED_SOURCE_PREFETCH=require.
_PRODUCTION_CACHE_PREFETCH_MODE = "warn"


def _packed_expert_render_hist_label(
    fmt: str,
    *,
    is_bf16: bool,
    source_label: str,
    cached_3d: torch.Tensor | None,
) -> str:
    if is_bf16:
        return source_label
    return f"{fmt}+cached" if cached_3d is not None else f"{fmt}+rtn"


def _packed_expert_export_provenance() -> dict[str, object]:
    cache = _PRODUCTION_WEIGHT_CACHE
    metadata = dict(getattr(cache, "metadata", {}) or {}) if cache is not None else {}
    coverage = metadata.get("packed_expert_coverage")
    return {
        "rtn_escape_enabled": bool(_ALLOW_PACKED_EXPERT_RTN),
        "inline_expert_gptq_enabled": bool(_INLINE_EXPERT_GPTQ),
        "cache_has_packed_expert_coverage": coverage is not None,
        "cache_packed_expert_coverage": coverage or {},
    }


def _inline_expert_render_levers() -> dict[str, object]:
    """Render levers for the inline packed-expert path, taken from the export's
    resolved act-aware flags (the same flags that drive the dense inline
    ``_quantize_2d`` GPTQ path). The batched NVFP4 render ignores JSO/act-order
    and uses fixed damp — matching the cache builder's ``"batched"`` mode — but
    FP8 experts fall to per-expert ``render_production_weight``, which consumes
    these levers, so we forward the real flags."""
    return {
        "gptq": bool(_ACT_AWARE_FLAGS.get("gptq", False)),
        "scale_sweep": bool(_ACT_AWARE_FLAGS.get("scale_sweep", False)),
        "static_act_order": bool(_ACT_AWARE_FLAGS.get("static_act_order", False)),
        "joint_scale_opt": bool(_ACT_AWARE_FLAGS.get("joint_scale_opt", False)),
    }


def _dump_live_cuda_tensors(where: str, min_mb: float = 200.0) -> None:
    """Diagnostic (PRISMAQUANT_EXPORT_MEM_DUMP=1): aggregate every live CUDA
    tensor above ``min_mb`` by (shape, dtype) and print one referrer summary
    per group — the direct attribution for a cuda_alloc ramp that survives
    gc.collect()+empty_cache() (a live reference is not garbage; only its
    holder's name finds it)."""
    groups: dict[tuple, list] = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and not obj.is_meta:
                nbytes = obj.numel() * obj.element_size()
                if nbytes >= min_mb * 1024 * 1024:
                    groups.setdefault(
                        (tuple(obj.shape), str(obj.dtype)), []).append(obj)
        except Exception:
            continue
    total = sum(
        t.numel() * t.element_size() for ts in groups.values() for t in ts)
    print(f"[mem-dump] {where}: cuda_alloc="
          f"{torch.cuda.memory_allocated()/1024**3:.1f}G, "
          f"{sum(len(v) for v in groups.values())} tensors >= {min_mb:.0f}MB "
          f"({total/1024**3:.1f}G python-visible)", flush=True)
    import types as _types
    own_ids = {id(groups)} | {id(ts) for ts in groups.values()}

    def _chase_iterator(it, depth: int = 0) -> str:
        """Name the frame/generator that keeps an iterator alive."""
        if depth > 4:
            return "?"
        for r in gc.get_referrers(it):
            if id(r) in own_ids or isinstance(r, type):
                continue
            if isinstance(r, _types.FrameType):
                owner = ""
                for r2 in gc.get_referrers(r):
                    if isinstance(r2, _types.GeneratorType):
                        owner = " (suspended generator)"
                        break
                return (f"frame {r.f_code.co_filename.rsplit('/', 1)[-1]}:"
                        f"{r.f_lineno} {r.f_code.co_name}{owner}")
            if isinstance(r, _types.GeneratorType):
                fr = r.gi_frame
                return (f"generator {r.gi_code.co_name}"
                        f":{fr.f_lineno if fr else 'done'}")
            if type(r).__name__ in (
                    "enumerate", "list_iterator", "tuple_iterator",
                    "zip", "map", "filter", "chain", "islice",
                    "dict_itemiterator", "dict_valueiterator"):
                return f"{type(r).__name__} <- {_chase_iterator(r, depth+1)}"
            if isinstance(r, (list, tuple, dict)):
                return f"{type(r).__name__}(n={len(r)}) <- " + _chase_iterator(
                    r, depth + 1)
        return "no-gc-referrer"

    def _describe(r, depth: int = 0) -> str:
        if isinstance(r, dict):
            holder = ""
            if depth < 2:
                for r2 in gc.get_referrers(r):
                    if id(r2) in own_ids or isinstance(
                            r2, (_types.FrameType, type)):
                        continue
                    holder = f" held-by {_describe(r2, depth + 1)}"
                    break
            return f"dict(n={len(r)}){holder}"
        if isinstance(r, (list, tuple)):
            holder = ""
            if depth < 2:
                for r2 in gc.get_referrers(r):
                    if id(r2) in own_ids or isinstance(
                            r2, (_types.FrameType, type)):
                        continue
                    holder = f" held-by {_describe(r2, depth + 1)}"
                    break
            return f"{type(r).__name__}[{len(r)}]{holder}"
        if isinstance(r, _types.FrameType):
            return (f"frame {r.f_code.co_name}:"
                    f"{[k for k, v in r.f_locals.items()][:6]}")
        return type(r).__name__
    for (shape, dtype), ts in sorted(
            groups.items(),
            key=lambda kv: -sum(t.numel() * t.element_size()
                                for t in kv[1])):
        gb = sum(t.numel() * t.element_size() for t in ts) / 1024**3
        ptrs = sorted({t.untyped_storage().data_ptr() for t in ts})
        for i, t in enumerate(ts):
            refs: list[str] = []
            for r in gc.get_referrers(t):
                if id(r) in own_ids or isinstance(r, type):
                    continue
                if isinstance(r, _types.FrameType):
                    names = [k for k, v in r.f_locals.items() if v is t]
                    refs.append(
                        f"frame {r.f_code.co_filename.rsplit('/', 1)[-1]}:"
                        f"{r.f_lineno} {r.f_code.co_name} locals{names}")
                elif isinstance(r, dict):
                    keys = [str(k) for k, v in r.items() if v is t][:2]
                    refs.append(f"dict keys{keys} {_describe(r)}")
                elif isinstance(r, (list, tuple)):
                    refs.append(
                        f"{type(r).__name__}[{len(r)}] <- "
                        f"{_chase_iterator(r)}")
                else:
                    refs.append(
                        f"{_describe(r)} <- {_chase_iterator(r)}")
                if len(refs) >= 3:
                    break
            print(f"[mem-dump]   [{i}] {list(shape)} {dtype} = "
                  f"{t.numel()*t.element_size()/1024**3:.2f}G "
                  f"storage@{hex(t.untyped_storage().data_ptr())} "
                  f"refs: {refs}", flush=True)
        print(f"[mem-dump]   group {list(shape)} {dtype}: {len(ts)} tensors "
              f"{gb:.2f}G across {len(ptrs)} storages", flush=True)


def _inline_render_packed_expert_module(
    model: nn.Module,
    experts_qname: str,
    assignment: dict[str, str],
    profile,
) -> "ProductionWeightCache | None":
    """Render ONE packed-experts module's stack into a transient in-memory
    ``ProductionWeightCache`` at export time (no disk, no whole dequant cache).

    Gated by ``PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ`` and only active when no
    production weight cache is supplied. The experts-module input snapshot X is
    sourced from the probe's activation cache (``_CACHED_ACTIVATIONS`` keyed by
    the experts-module qname); routing is recomputed offline from X + the
    resident gate weight by ``fill_packed_expert_cache_entries`` — the SAME
    batched GPTQ path (``module_acts_override``) the streaming cache builder
    uses, so the rendered dequant is identical to what the cache path would have
    produced for this (stack, format, activations).

    Returns the transient cache (its ``.weights`` hold the ``(full, fmt)`` 3-D
    dequant and ``.activation_max_abs`` the calibrated W4A4 input scale), or
    ``None`` when the gate is off, a production cache is active, or no
    activation snapshot exists for this module (the caller then decides whether
    to hard-fail or fall through to the RTN research path).
    """
    if not _INLINE_EXPERT_GPTQ:
        return None
    if _PRODUCTION_WEIGHT_CACHE is not None or _ALLOW_PACKED_EXPERT_RTN:
        return None
    if _CACHED_ACTIVATIONS is None:
        return None
    # Activation-residency landmine: _LazyActivationCache.get() returns a
    # CPU-resident fp32 tensor; fill_packed_expert_cache_entries' override path
    # reshapes + moves it to the compute device itself (derive_per_expert_
    # activations runs the router on `device`), so no manual .to() here.
    X = _CACHED_ACTIVATIONS.get(experts_qname)
    if X is None or X.numel() == 0:
        return None

    from .production_weight_cache import (
        ProductionWeightCache,
        _resolve_production_render_levers,
        fill_packed_expert_cache_entries,
    )

    # Resolve through the SAME contract the cache builder uses so the transient
    # cache records the identical nvfp4_scale_rule / damp provenance — the
    # export re-derive keys the NVFP4 codes off that recorded rule (M19/M2), so
    # a mismatch would flip the packed bytes vs the prebuilt-cache path.
    levers = _resolve_production_render_levers(_inline_expert_render_levers())
    transient = ProductionWeightCache(
        weights={},
        levers=dict(levers),
        activation_max_abs={},
        failed={},
        cache_dir=None,  # in-memory only — never spill the dequant to disk
        metadata={},
    )
    fill_packed_expert_cache_entries(
        transient,
        model,
        None,  # calib_ids unused: module_acts_override supplies X
        render_assignment=assignment,
        levers=levers,
        profile=profile,
        cache_dir=None,
        render_mode="batched",
        module_acts_override={experts_qname: X},
        progress=False,
    )
    return transient


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


def gptq_damp_sweep_enabled() -> bool:
    """Whether GPTQ runs the legacy 5-candidate in-sample damp sweep.

    DEFAULT OFF as of 2026-06-12 (Robert: "hard code it for now to your
    empirical best finding"). The V1 served A/B found the sweep's
    in-sample evaluator picks inverted winners: fixed damp 0.3 beat the
    sweep on every gold-lane readout across two calibration draws
    (all-position KL −6.6/−11.5%, WikiText-test PPL −0.9/−1.4%, tail NLL
    improved) at ~4.4x less render time (docs/unified_render_theory.md
    §8 V1). Set PRISMAQUANT_GPTQ_DAMP_SWEEP=1 to reproduce historical
    sweep-rendered artifacts.
    """
    return os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0") != "0"


def _resolve_gptq_fixed_damp(default: float = 1.0) -> float:
    """Fixed GPTQ damp used when the damp sweep is disabled.

    Default 1.0 = the V1 served A/B winner over THREE calibration draws:
    wins all-position KL and confident KL 3/3 draws vs damp 0.3, PPL 2/3
    and on average (26.80 vs 27.03); 0.3's only consistent edge was the
    max-of-16-chunks tail at shrinking margins (+0.3% on the last draw).
    ``PRISMAQUANT_GPTQ_DAMP`` overrides (0.01 reproduces vanilla GPTQ).
    Sweep paths pass explicit candidates and ignore this. Open research:
    derive the per-Linear optimum from weights/activations alone
    (docs/unified_render_theory.md §9.7 — two closed forms refuted so far).
    """
    raw = os.environ.get("PRISMAQUANT_GPTQ_DAMP", "")
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v > 0.0 else default


# Per-role GPTQ damp override (research lever, default-off). The unified-render
# V0b/V0c held-out basins are ROLE-structured (attention/o_proj -> 1.0,
# gate/up -> ~0.3, down_proj -> ~3.0); the optimal damp tracks the activation-
# conditioning role, NOT downstream sensitivity (measured Spearman(h_trace,
# opt_damp) ~= 0 on the 31 logged 4B Linears), while the *cost* of one global
# constant concentrates on the high-h_trace gate/up Linears (Spearman ~= +0.6).
# This lever lets a served A/B test the per-role table against the fixed-1.0
# default without reviving the in-sample sweep (docs/unified_render_theory.md
# §7-8). Spec, comma-separated role=damp pairs:
#   PRISMAQUANT_GPTQ_DAMP_ROLES="qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0"
# Roles: qkv, o_proj, gate_up, down, other. Unmatched roles and unparseable
# entries fall back to _resolve_gptq_fixed_damp(); unset => exact no-op.
_GPTQ_DAMP_ROLE_CACHE: "dict[str, dict[str, float]]" = {}


def _gptq_role_of(qname: str) -> str:
    """Map a Linear name to a render role for per-role damp selection."""
    q = str(qname)
    if "down_proj" in q:
        return "down"
    if "gate_proj" in q or "up_proj" in q:
        return "gate_up"
    if "o_proj" in q:
        return "o_proj"
    if "q_proj" in q or "k_proj" in q or "v_proj" in q:
        return "qkv"
    return "other"


def _parse_gptq_damp_roles(spec: str) -> "dict[str, float]":
    cached = _GPTQ_DAMP_ROLE_CACHE.get(spec)
    if cached is not None:
        return cached
    table: "dict[str, float]" = {}
    for part in spec.split(","):
        role, sep, val = part.strip().partition("=")
        role = role.strip()
        if not sep or not role:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if v > 0.0:
            table[role] = v
    _GPTQ_DAMP_ROLE_CACHE[spec] = table
    return table


def _resolve_gptq_damp_for_role(qname: str) -> float:
    """Per-role GPTQ damp (research lever); falls back to the fixed default.

    Returns ``_resolve_gptq_fixed_damp()`` unchanged when
    ``PRISMAQUANT_GPTQ_DAMP_ROLES`` is unset or the role is unlisted, so the
    production default (global fixed 1.0) is preserved bit-for-bit.
    """
    base = _resolve_gptq_fixed_damp()
    spec = os.environ.get("PRISMAQUANT_GPTQ_DAMP_ROLES", "")
    if not spec:
        return base
    return _parse_gptq_damp_roles(spec).get(_gptq_role_of(qname), base)


def _gptq_obs_rounding_nvfp4(
    weight: torch.Tensor, activations: torch.Tensor,
    group_size: int = 16, damp: float | None = None,
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
    if damp is None:
        damp = _resolve_gptq_fixed_damp()
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
    # Dead-channel handling: columns whose H diagonal is non-positive
    # (all-zero activation channel). Detect BEFORE damping — damping
    # lifts every diagonal above zero, which made this check
    # unreachable — and exclude dead entries from the damp reference
    # mean so a mass of dead channels can't deflate it. Dead diagonals
    # get an identity entry so the Cholesky succeeds. We do NOT zero
    # the weights of dead columns (deliberate serving-safe deviation
    # from reference GPTQ — a column unexercised by calibration must
    # not be destroyed for serving traffic); with an identity-like
    # row/col their OBS error propagation is a no-op and they quantize
    # as plain RTN.
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(damp * diag_mean)

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
        # Pass `global_real` — the fused-sibling override when supplied,
        # else the per-tensor global computed above (which is the
        # JSO-optimized joint global under joint_scale_opt) — so the
        # fallback doesn't silently discard the joint-global pick.
        return _rtn_dequant_nvfp4(
            weight,
            group_size=group_size,
            global_real_override=global_real,
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
    _s_g_real, global_real = _select_nvfp4_pack_scales_and_global(grouped)
    return global_real


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
    s_g_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
    )
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
    s_g_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
    )
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=s_g_real,
    )
    return codec.dequant.reshape(rows, cols)


def render_nvfp4_dequant(
    weight: torch.Tensor,
    *,
    group_size: int = 16,
    global_real_override: torch.Tensor | None = None,
    scale_rule: str = NVFP4_SCALE_RULE_STATIC_6,
    snapped_scale_scoring: bool = False,
    joint_scale_levels: tuple[float, ...] = (6.0, 4.0),
) -> torch.Tensor:
    """Render a 2-D weight through the exact production NVFP4 codec.

    Unlike the historical private RTN helper, every scale-plane choice is an
    explicit argument.  Offline selectors such as PrismaSnap must be
    reproducible from their receipt and therefore cannot inherit either
    ``PRISMAQUANT_NVFP4_SCALE_RULE`` or the research-only snapped-scale
    scoring switch from the process environment.  Existing exporter call
    sites continue to use their unchanged environment/default behavior.
    """
    if weight.ndim != 2:
        raise ValueError(
            "production NVFP4 dequant render requires a rank-2 weight; "
            f"got shape={tuple(weight.shape)}"
        )
    rows, cols = weight.shape
    if cols % int(group_size) != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(
        rows, cols // int(group_size), int(group_size)
    )
    scale_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        global_real_override=global_real_override,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
        snapped_scale_scoring=snapped_scale_scoring,
        joint_scale_levels=joint_scale_levels,
    )
    codec = _nvfp4_quantize_grouped_codec(
        grouped,
        global_real=global_real,
        scale_real=scale_real,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
    )
    return codec.dequant.reshape(rows, cols)


def nvfp4_global_real(
    weight: torch.Tensor,
    *,
    group_size: int = 16,
    scale_rule: str = NVFP4_SCALE_RULE_STATIC_6,
    snapped_scale_scoring: bool = False,
    joint_scale_levels: tuple[float, ...] = (6.0, 4.0),
) -> torch.Tensor:
    """Return the explicit production-codec global multiplier for a weight.

    This is the companion to :func:`render_nvfp4_dequant` used when vLLM
    fuses sibling projections into one runtime parameter and therefore makes
    them share the maximum of their natural globals.
    """
    if weight.ndim != 2:
        raise ValueError(
            "production NVFP4 global calculation requires a rank-2 weight; "
            f"got shape={tuple(weight.shape)}"
        )
    rows, cols = weight.shape
    if cols % int(group_size) != 0:
        raise ValueError(f"NVFP4 group_size={group_size} ∤ {cols}")
    grouped = weight.float().reshape(
        rows, cols // int(group_size), int(group_size)
    )
    _scale_real, global_real = _select_nvfp4_pack_scales_and_global(
        grouped,
        scale_rule=resolve_nvfp4_scale_rule(scale_rule),
        snapped_scale_scoring=snapped_scale_scoring,
        joint_scale_levels=joint_scale_levels,
    )
    return global_real


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
    for _ in range(3):
        snapped = _select_nvfp4_group_scales(
            grouped,
            global_real=global_real.view(E, 1, 1),
        )
        next_global = (
            snapped.reshape(E, -1).amax(dim=-1) / FP8_E4M3_MAX
        ).clamp_min(1e-12)
        s_g_real = snapped
        if torch.allclose(next_global, global_real, rtol=0.0, atol=1e-12):
            break
        global_real = next_global
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
    damp: float | None = None,
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
    if damp is None:
        damp = _resolve_gptq_fixed_damp()
    H = X.t() @ X
    # Dead-channel handling: detect diag(H) <= 0 BEFORE damping (see
    # _gptq_obs_rounding_nvfp4), exclude dead entries from the damp
    # reference mean, give dead diagonals an identity entry. We do NOT
    # zero the weights of dead columns (deliberate serving-safe
    # deviation from reference GPTQ — a column unexercised by
    # calibration must not be destroyed for serving traffic).
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(float(damp) * diag_mean)
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
    damp: float | None = None,
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
    if damp is None:
        damp = _resolve_gptq_fixed_damp()
    H = X.t() @ X
    # Dead-channel handling: detect diag(H) <= 0 BEFORE damping (see
    # _gptq_obs_rounding_nvfp4), exclude dead entries from the damp
    # reference mean, give dead diagonals an identity entry. We do NOT
    # zero the weights of dead columns (deliberate serving-safe
    # deviation from reference GPTQ — a column unexercised by
    # calibration must not be destroyed for serving traffic).
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if dead.any():
        H[dead, dead] = 1.0
    H.diagonal().add_(float(damp) * diag_mean)
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

    Within a (layer, proj) bucket we always emit a `[0-9]+` expert-index
    wildcard. The exporter relies on the allocator/export coherence gates to
    keep every expert in a layer on one scheme for a given projection; sparse
    within-layer expert subsets are not represented as separate alternations.
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
def _fused_dense_group(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Compatibility delegate for the shared fusion catalog."""

    return _nvfp4_activation_contract.fused_dense_group(name)


def _fused_group_key_for_name(name: str, profile=None) -> str | None:
    """Legacy error-tolerant delegate to the shared fusion policy."""

    return _nvfp4_activation_contract.fused_sibling_group_key(
        name,
        profile=profile,
        tolerate_profile_errors=True,
    )


def _unify_input_global_scales_across_fused_siblings(
    scales: dict[str, float],
    *,
    profile=None,
) -> dict[str, float]:
    """Compatibility delegate for the shared conservative scale join."""

    return _nvfp4_activation_contract.unify_fused_sibling_input_global_scales(
        scales,
        profile=profile,
        tolerate_profile_errors=True,
        diagnostic_prefix="[export-stream]",
    )


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

    `input_global_scale_override`: explicit per-Linear activation scale.  If
    absent, the shared resolver checks the calibrated mapping and finally the
    legacy `DEFAULT_INPUT_GLOBAL_SCALE` (1.0).  That fallback preserves native
    artifact semantics but does not qualify the versioned fused contract.

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
                # GPTQ wallclock.
                # Default OFF since 2026-06-12 (see
                # gptq_damp_sweep_enabled(), :1845-1857): its evaluator
                # is in-sample, so its "winners" invert on held-out
                # basins (31/31) and it lost the V1 served A/B to a
                # fixed damp. Production uses the fixed damp from
                # _resolve_gptq_fixed_damp() (1.0, :1860).
                # PRISMAQUANT_GPTQ_DAMP_SWEEP=1 reproduces historical
                # sweep-rendered artifacts.
                if gptq_damp_sweep_enabled():
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
                    _record_do_no_harm_revert("NVFP4")
                    if os.environ.get(
                        "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                        print(f"[do-no-harm] {linear_name}: "
                              f"reverted to RTN "
                              f"(mse {mse_work:.3e} → {mse_rtn:.3e})",
                              flush=True)
                    w_work = w_rtn
            except Exception as _e:
                _record_do_no_harm_failure("NVFP4", linear_name, _e)

        # Step 4: final NVFP4 pack. `w_work` is the post-GPTQ,
        # post-act-round, post-scale-sweep weight.
        input_scale = _resolve_nvfp4_input_global_scale(
            input_global_scale_override,
            target=linear_name,
        )

        # compute_only path: return the dequantized weight WITHOUT freezing
        # it into FP4 codes. Its only production consumer was block-output
        # match, walled 2026-07-30 (archive/block_output_match_2026-07-30/,
        # re-vet R25) along with the `_finalize_compute_only` packer that
        # closed the loop. Kept as a lever-threading introspection hook —
        # nothing in the export path sets it.
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
                if gptq_damp_sweep_enabled():
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
                        _record_do_no_harm_revert(fmt)
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP8 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        w, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception as _e:
                    _record_do_no_harm_failure(fmt, linear_name, _e)
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
            if gptq_damp_sweep_enabled():
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
                if gptq_damp_sweep_enabled():
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
                        _record_do_no_harm_revert("MXFP4")
                        if os.environ.get(
                            "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                            print(f"[do-no-harm] {linear_name}: "
                                  f"reverted MXFP4 to RTN "
                                  f"(mse {err_work:.3e} → {err_rtn:.3e})",
                                  flush=True)
                        wp, ws, dq = q_rtn, s_rtn, dq_rtn
                except Exception as _e:
                    _record_do_no_harm_failure("MXFP4", linear_name, _e)
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
            gptq_damp_sweep_enabled())
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
            if n_reverted:
                _DO_NO_HARM_STATS["NVFP4_batched_reverts"] += int(n_reverted)
            if n_reverted and os.environ.get(
                    "PRISMAQUANT_DO_NO_HARM_VERBOSE") == "1":
                print(f"[do-no-harm batched] reverted {n_reverted}/{n} "
                      f"Linears to RTN", flush=True)
        except Exception as _e:
            _record_do_no_harm_failure("NVFP4_batched", None, _e)

    # Per-Linear final NVFP4 pack (cheap; reuses the existing function).
    out: list[dict] = []
    for i, (full, emit_full, recipe_key, mod) in enumerate(items):
        override = overrides_list[i]
        wp, ws, wg = quantize_dequantize_nvfp4(
            weights[i], group_size=16,
            global_real_override=override,
        )
        input_scale = _resolve_nvfp4_input_global_scale(target=recipe_key)
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


# NOTE: `_init_rotary_inplace` is imported from `streaming_model` (single
# source of truth). It includes the profile-driven `init_rotaries` dispatch
# for multi-layer-type rotaries (DSv4/Gemma3/Gemma4); a stale duplicate here
# previously lacked it and would crash Gemma4 export at rotary init
# (KeyError: None on rope_parameters[None]).


def _build_fp8_source_map(
    model_path: str, *, profile=None, multimodal: bool = False,
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
    if profile is not None:
        pairs_fn = getattr(profile, "fp8_scale_pairs", None)
        if callable(pairs_fn):
            explicit = pairs_fn(model_path)
            if explicit is not None:
                return {
                    (key[:-7] if key.endswith(".weight") else key): value
                    for key, value in explicit.items()
                }

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

    def _rename_weight(k: str) -> str | None:
        weight_key = f"{k}.weight"
        if profile is not None:
            mapper = getattr(profile, "checkpoint_to_live_name", None)
            if callable(mapper):
                try:
                    live_weight = mapper(weight_key, multimodal=multimodal)
                except TypeError:
                    live_weight = mapper(weight_key)
                except Exception:
                    live_weight = None
                if live_weight is None:
                    return None
                live_weight = str(live_weight)
                return (
                    live_weight[:-7]
                    if live_weight.endswith(".weight")
                    else live_weight
                )

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
                live_base = _rename_weight(ck_base)
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


def _recipe_name_for_live_qname(qname: str, profile) -> str:
    if profile is None:
        return qname
    mapper = getattr(profile, "live_to_recipe_name", None)
    if not callable(mapper):
        return qname
    try:
        return str(mapper(qname))
    except Exception:
        return qname


def _fp8_source_passthrough_recipe_keys(
    fp8_source_map: dict[str, tuple[str, str]],
    source_dtype_by_name: dict[str, torch.dtype],
    profile,
) -> set[str]:
    """Return recipe keys that can be emitted as source-native FP8 bytes."""
    out: set[str] = set()
    for live_base in fp8_source_map:
        source_weight_key = f"{live_base}.weight"
        source_weight_dtype = source_dtype_by_name.get(source_weight_key)
        if source_weight_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        out.add(_recipe_name_for_live_qname(live_base, profile))
    return out


def _fp8_source_config_overlay(
    model_path: str,
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    profile,
) -> tuple[dict[str, str], set[str], set[str]]:
    """Mirror streaming materialization's FP8_SOURCE passthrough in config.

    Native-FP8 checkpoints load source weights as dequanted BF16 for render
    work, but BF16/unassigned source-FP8 Linears are emitted by copying the
    original FP8 bytes and scale sidecars. The compressed-tensors config must
    describe those emitted bytes as FP8_SOURCE instead of putting the same
    names in ignore.
    """
    fp8_source_map = _build_fp8_source_map(model_path, profile=profile)
    if not fp8_source_map:
        return dict(assignment), set(bf16_passthrough), set()

    from .layer_streaming import _build_weight_map

    weight_shard, weight_ckpt = _build_weight_map(model_path)
    source_dtype_by_name = _build_source_dtype_map(weight_shard, weight_ckpt)
    source_recipe_keys = _fp8_source_passthrough_recipe_keys(
        fp8_source_map,
        source_dtype_by_name,
        profile,
    )
    if not source_recipe_keys:
        return dict(assignment), set(bf16_passthrough), set()

    config_assignment = dict(assignment)
    overrides: set[str] = set()
    # Packed-expert parents in the assignment (e.g.
    # `model.layers.10.mlp.experts.gate_up_proj`) own their per-expert
    # leaves: the packed emit writes those bytes in the allocated format,
    # so a per-expert source-FP8 leaf (`...mlp.experts.7.up_proj`) must
    # NOT be re-described as FP8_SOURCE — the config would contradict the
    # emitted bytes (GLM-5.3: 168 such entries double-covered the NVFP4
    # experts with the FP8 scheme).
    packed_expert_prefixes = tuple(
        {k.rsplit(".", 1)[0] + "." for k in assignment
         if ".mlp.experts." in k or k.endswith(".mlp.experts")})
    _per_expert_leaf = re.compile(r"\.experts\.[0-9]+\.")
    for recipe_key in sorted(source_recipe_keys):
        if (_per_expert_leaf.search(recipe_key)
                and recipe_key.startswith(packed_expert_prefixes)):
            continue
        recipe_fmt = config_assignment.get(recipe_key)
        fmt = (
            _canonical_export_format(recipe_fmt)
            if recipe_fmt is not None
            else None
        )
        if fmt is None or fmt == "BF16" or recipe_key in bf16_passthrough:
            config_assignment[recipe_key] = "FP8_SOURCE"
            overrides.add(recipe_key)

    config_bf16_passthrough = set(bf16_passthrough) - overrides
    return config_assignment, config_bf16_passthrough, overrides


def _compressed_tensor_key(base_name: str, suffix: str) -> str:
    return f"{base_name}.{suffix}"


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
        _build_concat_merger,
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
    from .sensitivity_probe import stage_multimodal, stage_text_only
    # Canonical rotary init (profile-driven multi-layer-type dispatch).
    from .streaming_model import (
        _init_rotary_inplace,
        _mask_cuda_queries_during_meta_init,
        _skeleton_config_and_class,
    )

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
    # A family with no `<Arch>ForCausalLM` auto-route (glm5_next on
    # transformers 5.16) cannot build a text-only skeleton at all; the
    # profile declares that fact and the export inherits the same flip
    # `_build_streaming_context` applies for probe/cost streaming. The
    # visual tower stays on meta either way — export ships it via the
    # source-passthrough merge, never through the body walk.
    multimodal_skeleton = bool(profile.requires_multimodal_skeleton())
    if multimodal_skeleton:
        print("[export-stream] profile has no text-only skeleton route; "
              "using the multimodal construction", flush=True)
        staged = stage_multimodal(model_path)
    else:
        staged = stage_text_only(model_path)
    config = AutoConfig.from_pretrained(staged, trust_remote_code=True)
    config, model_cls = _skeleton_config_and_class(
        config, multimodal=multimodal_skeleton,
        log_prefix="[export-stream]")
    # _init_weights is globally no-op'd by prismaquant.__init__'s
    # _polyfill_transformers (wasted work + transformers-5.x compat
    # landmine on remote modeling files).
    with _mask_cuda_queries_during_meta_init("[export-stream]"):
        with init_empty_weights():
            if model_cls is AutoModelForCausalLM:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True)
            else:
                model = model_cls._from_config(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model, layers = _get_layer_list(model)
    base_prefix = _resolve_base_prefix(model, base_model)
    num_layers = len(layers)
    layers_prefix = f"{base_prefix}.layers." if base_prefix else "layers."

    weight_shard, weight_ckpt = _build_weight_map(
        model_path, multimodal=multimodal_skeleton)
    source_dtype_by_name = _build_source_dtype_map(weight_shard, weight_ckpt)
    # Per-expert -> packed-3D bridge for checkpoints that ship MoE experts
    # unfused while the live module is packed (driven by the model profile;
    # None for every other model). Keeps the exporter's source read aligned
    # with the streaming probe/cost path — a raw checkpoint exports without
    # an out-of-band pre-pack.
    expert_packer = _build_expert_packer(model, weight_ckpt)
    # Sibling bridge for checkpoints that store one live parameter as several
    # source tensors (transformers' `Concatenate(dim=...)` merges, declared as
    # the profile spec's `concat_merges`). Same reason as the expert packer:
    # the exporter must load exactly what the streaming probe/cost path loads.
    concat_merger = _build_concat_merger(model, weight_ckpt)
    # Emit-side inverse of `concat_merges`: the live module holds ONE
    # merged tensor (glm5_next `self_attn.conv1d` <- q/k/v_conv1d) whose
    # key never existed in the source checkpoint. Emitting the merged
    # spelling would ship a key the serving runtime's loader does not
    # know and drop the three it expects, so the 3e passthrough skips the
    # merged live param and copies the SOURCE tensors verbatim instead.
    concat_groups = tuple(profile.concat_merge_groups())
    _src_tensor_index: dict[str, Path] | None = None

    def _read_source_tensor_verbatim(ckpt_key: str) -> torch.Tensor:
        nonlocal _src_tensor_index
        from safetensors import safe_open
        if _src_tensor_index is None:
            _src_tensor_index = {}
            src_root = Path(model_path)
            index_path = src_root / "model.safetensors.index.json"
            if index_path.exists():
                with open(index_path) as fh:
                    for key, shard in json.load(fh)["weight_map"].items():
                        _src_tensor_index[key] = src_root / shard
            else:
                for shard in sorted(src_root.glob("*.safetensors")):
                    with safe_open(str(shard), framework="pt") as sf:
                        for key in sf.keys():
                            _src_tensor_index[key] = shard
        shard_path = _src_tensor_index.get(ckpt_key)
        if shard_path is None:
            raise KeyError(
                f"[export-stream] concat-merge source tensor {ckpt_key!r} "
                f"not present in the source checkpoint — the profile's "
                f"concat_merges declaration disagrees with the shards.")
        with safe_open(str(shard_path), framework="pt") as sf:
            return sf.get_tensor(ckpt_key)
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
    fp8_source_map = _build_fp8_source_map(model_path, profile=profile)
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
                out_key = _compressed_tensor_key(base_name, suffix)
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
        fp_state = _render_lever_provenance()
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
            fp8_scale_inv_map=fp8_scale_inv_map, pack_experts=expert_packer,
            merge_concat=concat_merger)
        resolver = _build_install_resolver(model, layer_qname)
        _fast_install(resolver, tensors, device, model=model)
        load_s = time.time() - load_t0

        layer_mod = model.get_submodule(layer_qname)

        # 3b. Joint NVFP4 scales across fused siblings in this layer.
        # M2: computed under the render's recorded scale rule so the
        # fused joint-global pre-pass is consistent with the
        # match-render-scale re-derive (_pack_production_cached_2d).
        with _temporary_export_nvfp4_scale_rule(
                _export_match_render_scale_rule(_PRODUCTION_WEIGHT_CACHE)):
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
        # Batch same-shape NVFP4 Linears when act-aware passes
        # (GPTQ / scale_sweep) are on. Default ON; set
        # PRISMAQUANT_BATCHED_NVFP4_EXPORT=0 to force the slower
        # per-Linear `_quantize_2d` path for bit-exact A/Bs.
        grouped_nvfp4_batched: dict[
            tuple[int, int],
            list[tuple[str, str, str, nn.Linear]]
        ] = defaultdict(list)
        # v26: default ON.
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
                # Runtime-pinned modules (e.g. GLM MLA attention
                # projections) are dequantized by the serving runtime at
                # load, keyed on the SOURCE scale name
                # (`weight_scale_inv`); everything else serves through
                # compressed-tensors and gets the CT name.
                scale_key = ("weight_scale_inv"
                             if profile.runtime_loads_source_fp8(full)
                             else "weight_scale")
                out[f"{emit_full}.{scale_key}"] = w_scale.to(
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


        # 3d. Emit packed MoE experts, scoped to this layer.
        packed_count = 0
        # Inline packed-expert renders (PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ):
        # one transient in-memory cache per experts module in THIS layer, built
        # on first use and freed at layer end. Peak stays ~one layer's stack.
        _inline_expert_caches: dict[str, "ProductionWeightCache | None"] = {}
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

                # Inline render (no production cache): render this experts
                # module's stack on the fly into a transient cache, memoized so
                # gate_up + down share the one fill_packed_expert_cache_entries
                # call. Falls back to _PRODUCTION_WEIGHT_CACHE reads when the
                # gate is off (active_cache stays None -> module default). Done
                # BEFORE the scale-rule + joint-global pre-pass so the re-derive
                # keys off the inline render's recorded nvfp4_scale_rule.
                if not is_bf16 and experts_qname not in _inline_expert_caches:
                    _inline_expert_caches[experts_qname] = (
                        _inline_render_packed_expert_module(
                            model, experts_qname, assignment, profile))
                active_cache = _inline_expert_caches.get(experts_qname)

                # Per-expert joint global scale when NVFP4 splits gate+up.
                # M2: computed under the render's recorded scale rule so
                # the joint global is consistent with the re-derive below.
                packed_render_rule = _packed_expert_render_scale_rule(
                    active_cache)
                per_expert_joint: list[torch.Tensor | None] = [None] * E
                if fmt == "NVFP4" and len(proj_split) > 1:
                    with _temporary_export_nvfp4_scale_rule(
                            packed_render_rule):
                        for orig_e, _ in iter_experts:
                            cands = [
                                compute_nvfp4_global_real(
                                    sp[orig_e].float(), group_size=16)
                                for _, sp in proj_split
                            ]
                            per_expert_joint[orig_e] = (
                                torch.stack(cands).max())

                # Pull the GPTQ-rendered dequant from the production cache (or
                # the inline transient cache above). Packed experts go through
                # the SAME deliberate render as 2-D Linears
                # (fill_packed_expert_cache_entries); export re-packs the cached
                # dequant per expert. RTN-by-omission on packed experts is a
                # severe NVFP4 quality regression, so when a cache is active (or
                # the inline gate is set) we HARD-FAIL rather than silently RTN.
                cached_3d = None
                cached_split = None
                if not is_bf16:
                    if not _ALLOW_PACKED_EXPERT_RTN:
                        cached_3d = _read_cached_packed_expert(
                            full, fmt, device=device, cache=active_cache)
                        if cached_3d is not None and active_cache is not None:
                            # The transient inline render is read exactly once
                            # per packed param — drop the CPU fp32 entry now
                            # (18G for a GLM-class gate_up stack) instead of
                            # holding it until layer teardown.
                            _k = active_cache.resolve_key(full, fmt)
                            if _k is not None:
                                active_cache.weights.pop(_k, None)
                    if cached_3d is None:
                        if _INLINE_EXPERT_GPTQ and not _ALLOW_PACKED_EXPERT_RTN:
                            raise RuntimeError(
                                f"[export-stream] packed expert {full} @ {fmt} "
                                f"could not be rendered inline "
                                f"(PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ=1) — no "
                                f"experts-module activation snapshot for "
                                f"{experts_qname} in the activation cache. RTN "
                                f"on NVFP4 experts is banned; supply an "
                                f"--activation-cache-dir whose probe covers "
                                f"every packed-experts module, set the expert to "
                                f"BF16, or set "
                                f"PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                                f"explicit research/A-B RTN export.")
                        if (_PRODUCTION_WEIGHT_CACHE is not None
                                and not _ALLOW_PACKED_EXPERT_RTN):
                            raise RuntimeError(
                                f"[export-stream] packed expert {full} @ {fmt} "
                                f"has no production-cache render. Non-BF16 "
                                f"packed experts MUST be rendered through the "
                                f"deliberate GPTQ+JSO path "
                                f"(fill_packed_expert_cache_entries) — RTN on "
                                f"NVFP4 experts is a silent quality regression "
                                f"and is banned. Re-run build_production_cache "
                                f"with the packed experts in scope, set the "
                                f"expert to BF16, or set "
                                f"PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                                f"explicit research/A-B RTN export.")
                        print(
                            f"[export-stream] WARNING: RTN-rendering packed "
                            f"expert {full} @ {fmt} "
                            f"({'PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN research/A-B export' if _ALLOW_PACKED_EXPERT_RTN else 'no production cache active'}). "
                            f"This is NOT a production path — NVFP4 experts "
                            f"need GPTQ+JSO.",
                            flush=True)
                    else:
                        cached_split = _split_packed_expert_tensor(
                            cached_3d, pn, profile)

                # Calibrated input_global_scale (W4A4 activation clip): one
                # per packed param, calibrated from the routed activations at
                # cache-build time (or the inline render). Without it experts
                # ship the 1.0 placeholder.
                expert_input_scale = _packed_expert_input_global_scale(
                    full, cache=active_cache)
                if (cached_3d is not None and expert_input_scale is None
                        and fmt == "NVFP4"):
                    raise RuntimeError(
                        f"[export-stream] packed expert {full} @ {fmt} has a "
                        f"cached render but no calibrated input_global_scale "
                        f"(would ship the 1.0 placeholder). The cache's "
                        f"packed_expert_max_abs sidecar is missing this entry "
                        f"— re-run build_production_cache so the scale is "
                        f"recomputed, or delete the expert shard to force a "
                        f"full re-render.")
                if expert_input_scale is not None and fmt == "NVFP4":
                    # K0.2: a calibrated stage may only ship alongside its
                    # attested sibling stage.
                    _packed_expert_stage_attestation(
                        full, cache=active_cache, profile=profile)

                # M2: re-derive under the render's RECORDED NVFP4 scale
                # rule (the dense _pack_production_cached_2d wrap, lifted
                # to packed experts) — a joint_mse-rendered expert
                # re-derived under the export-entry default (static_6)
                # cannot recover its codes.
                with _temporary_export_nvfp4_scale_rule(packed_render_rule):
                    # Index-based on purpose: `enumerate` caches its last
                    # yielded result tuple internally (CPython reuse), and a
                    # surviving enumerate pinned the (proj_name, 9G-view)
                    # tuple — and through it an 18G fp32 stack — across
                    # layers (mem-dump attribution, GLM-5.3 probe runs).
                    # Never pass tensor-bearing tuples through enumerate in
                    # this function's scope.
                    for pi in range(len(proj_split)):
                        proj_name = proj_split[pi][0]
                        sub_packed = proj_split[pi][1]
                        cached_sub = (
                            cached_split[pi][1]
                            if cached_split is not None else None
                        )
                        for orig_e, new_e in iter_experts:
                            base = f"{disk_qname}.{new_e}.{proj_name}"
                            if is_bf16:
                                expert_2d = sub_packed[orig_e]
                                out[f"{base}.weight"], label = (
                                    _passthrough_tensor(
                                        full, expert_2d,
                                        source_dtype_by_name))
                            else:
                                # Re-pack the GPTQ-rendered dequant when
                                # cached (re-derives codes from the dequant,
                                # same ~1e-3 approximation as the 2-D path —
                                # not bit-lossless); fall back to source only
                                # on the no-cache warning.
                                expert_2d = (
                                    cached_sub[orig_e]
                                    if cached_sub is not None
                                    else sub_packed[orig_e]
                                )
                                compressed = _quantize_2d(
                                    expert_2d, fmt,
                                    nvfp4_global_real_override=(
                                        per_expert_joint[orig_e]),
                                    input_global_scale_override=(
                                        expert_input_scale),
                                )
                                for suffix, t in compressed.items():
                                    out[f"{base}.{suffix}"] = t.cpu()
                covered.add(full)
                hist[(
                    "packed_moe_per_expert",
                    _packed_expert_render_hist_label(
                        fmt,
                        is_bf16=is_bf16,
                        source_label=label if is_bf16 else "",
                        cached_3d=cached_3d,
                    ),
                )] += 1
                del packed_param, packed_param_src, proj_split
                if cached_3d is not None:
                    del cached_3d, cached_split
                # The inner-loop locals are function-scoped and survive this
                # module (and this LAYER) otherwise: `sub_packed` views the
                # 18G fp32 `packed_param` promotion, `cached_sub` views the
                # 18G fp32 `cached_3d` — together they pinned 36G through the
                # NEXT layer's render peak on GLM-5.3 (mem-dump attribution,
                # probe runs 1-2; the steady-state OOM of export attempts
                # 4-5 on the 118G unified pool).
                sub_packed = cached_sub = expert_2d = None

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
            concat_split = False
            for target_suffix, source_suffixes, _dim in concat_groups:
                if not (full == target_suffix
                        or full.endswith("." + target_suffix)):
                    continue
                stem = full[: -len(target_suffix)]
                for src_suffix in source_suffixes:
                    live_src = stem + src_suffix
                    ckpt_key = profile.export_tensor_name(
                        profile.live_to_recipe_name(live_src))
                    out[live_src] = _read_source_tensor_verbatim(ckpt_key)
                    hist[("layer_concat_source", "verbatim")] += 1
                concat_split = True
                break
            if concat_split:
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
        # Free this layer's inline expert renders (the 3-D dequant stacks) so
        # peak stays ~one layer's stack even across the whole sweep.
        del tensors, resolver, joint_globals, _inline_expert_caches
        # The per-Linear walk's loop locals survive the layer iteration:
        # `active_cache` in particular still references the layer's transient
        # expert cache (27G CPU fp32 on GLM-5.3) until the NEXT layer's packed
        # emit reassigns it — i.e. after that layer's full render peak. That
        # exact residue OOM-killed export attempt 3 at layer 4 on the 118G
        # unified pool. Drop it, then collect cycles BEFORE empty_cache so
        # freed CUDA blocks actually return to the pool, every layer.
        active_cache = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure outputs are CPU-resident
            torch.cuda.empty_cache()
        if os.environ.get("PRISMAQUANT_EXPORT_MEM_DUMP", "0") == "1":
            _dump_live_cuda_tensors(f"post-cleanup layer {L:02d}")
        # Every layer, unconditionally: a 306B layer can take minutes of
        # GPU render, and a silent multi-minute phase is an observability
        # defect on first-live-use exports.
        elapsed = time.time() - layer_t0
        done_n = L + 1
        sweep_rate = (time.time() - t_layers) / max(done_n, 1)
        eta_min = sweep_rate * (num_layers - done_n) / 60.0
        # Memory attribution per layer: rss (process anon), cuda_alloc
        # (live tensors), cuda_reserved (allocator pool). A layer-over-layer
        # ramp in alloc = a real reference leak; in reserved-only =
        # fragmentation; in rss-only = CPU-side. On the 118G unified pool a
        # silent ~14G/layer ramp is fatal by layer 6 — attribute, don't guess.
        try:
            rss_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      / 1024 / 1024)
            with open("/proc/self/statm") as f:
                cur_rss_gb = (int(f.read().split()[1])
                              * os.sysconf("SC_PAGE_SIZE") / 1024**3)
        except Exception:
            rss_gb = cur_rss_gb = float("nan")
        if device.type == "cuda":
            mem_note = (f"  rss={cur_rss_gb:.1f}G(max {rss_gb:.1f}) "
                        f"cuda_alloc={torch.cuda.memory_allocated()/1024**3:.1f}G "
                        f"reserved={torch.cuda.memory_reserved()/1024**3:.1f}G")
        else:
            mem_note = f"  rss={cur_rss_gb:.1f}G(max {rss_gb:.1f})"
        print(f"[export-stream] layer {L:02d}  linears={linear_count} "
              f"packed={packed_count}  load={load_s:.2f}s  "
              f"total={elapsed:.2f}s  out_keys={len(out)}  "
              f"({done_n}/{num_layers}, {sweep_rate:.1f}s/layer, "
              f"ETA {eta_min:.1f} min){mem_note}", flush=True)
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
        _stop_after = os.environ.get("PRISMAQUANT_EXPORT_STOP_AFTER_LAYER")
        if _stop_after is not None and L >= int(_stop_after):
            # Diagnostic runs only: exit AFTER the layer cache save + sink
            # flush so the rendered layer is resumable. Artifact INCOMPLETE.
            print(f"[export-stream] PRISMAQUANT_EXPORT_STOP_AFTER_LAYER="
                  f"{_stop_after}: stopping after layer {L:02d} "
                  f"(diagnostic run — artifact INCOMPLETE)", flush=True)
            sys.exit(0)

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
    # M2: under the render's recorded scale rule, consistent with the
    # _pack_production_cached_2d match-render-scale re-derive below.
    with _temporary_export_nvfp4_scale_rule(
            _export_match_render_scale_rule(_PRODUCTION_WEIGHT_CACHE)):
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
        compressed = _pack_production_cached_2d(
            fmt_key,
            fmt,
            nvfp4_global_real_override=joint,
            device=mod.weight.device,
        )
        if compressed is None and _PRODUCTION_WEIGHT_CACHE is not None:
            raise RuntimeError(
                f"[export-inmemory] auxiliary Linear {fmt_key} @ {fmt} has "
                "no production-cache render. Non-BF16 MTP/sidecar Linears "
                "must be rendered through ProductionWeightCache before "
                "export; RTN-by-omission would ship bytes that validation "
                "did not measure."
            )
        cache_hit = compressed is not None
        if compressed is None:
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
        hist[("linear", f"{fmt}_PRODUCTION_CACHE" if cache_hit else fmt)] += 1

    # Inline packed-expert renders (PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ):
    # one transient in-memory cache per experts module, built on first use.
    _inline_expert_caches: dict[str, "ProductionWeightCache | None"] = {}
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

            # Inline render (no production cache): render this experts module's
            # stack into a transient cache, memoized. Done BEFORE the scale-rule
            # + joint-global pre-pass so the re-derive keys off the inline
            # render's recorded nvfp4_scale_rule.
            if not is_bf16 and qname not in _inline_expert_caches:
                _inline_expert_caches[qname] = (
                    _inline_render_packed_expert_module(
                        model, qname, assignment, profile))
            active_cache = _inline_expert_caches.get(qname)

            # M2: joint globals + re-derive below run under the render's
            # recorded NVFP4 scale rule (see _packed_expert_render_scale_rule).
            packed_render_rule = _packed_expert_render_scale_rule(active_cache)
            per_expert_joint: list[torch.Tensor | None] = [None] * E
            if fmt == "NVFP4" and len(proj_split) > 1:
                with _temporary_export_nvfp4_scale_rule(packed_render_rule):
                    for e in range(E):
                        candidates = [
                            compute_nvfp4_global_real(sub_packed[e].float(),
                                                      group_size=16)
                            for _, sub_packed in proj_split
                        ]
                        per_expert_joint[e] = torch.stack(candidates).max()

            # Read the GPTQ-rendered dequant from the production cache (same
            # contract as the streaming path); hard-fail on RTN-by-omission
            # when a cache is active.
            cached_3d = None
            cached_split = None
            if not is_bf16:
                if not _ALLOW_PACKED_EXPERT_RTN:
                    cached_3d = _read_cached_packed_expert(
                        full_name, fmt, cache=active_cache)
                if cached_3d is None:
                    if _INLINE_EXPERT_GPTQ and not _ALLOW_PACKED_EXPERT_RTN:
                        raise RuntimeError(
                            f"[export-inmemory] packed expert {full_name} @ "
                            f"{fmt} could not be rendered inline "
                            f"(PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ=1) — no "
                            f"experts-module activation snapshot for {qname} in "
                            f"the activation cache. RTN on NVFP4 experts is "
                            f"banned; supply an --activation-cache-dir whose "
                            f"probe covers this module, set the expert to BF16, "
                            f"or set PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 for an "
                            f"explicit research/A-B RTN export.")
                    if (_PRODUCTION_WEIGHT_CACHE is not None
                            and not _ALLOW_PACKED_EXPERT_RTN):
                        raise RuntimeError(
                            f"[export-inmemory] packed expert {full_name} @ "
                            f"{fmt} has no production-cache render. Non-BF16 "
                            f"packed experts MUST be rendered through the "
                            f"deliberate GPTQ path "
                            f"(fill_packed_expert_cache_entries) — RTN on NVFP4 "
                            f"experts is a silent quality regression and is "
                            f"banned. Set PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN=1 "
                            f"for an explicit research/A-B RTN export.")
                    print(
                        f"[export-inmemory] WARNING: RTN-rendering packed "
                        f"expert {full_name} @ {fmt} "
                        f"({'research/A-B flag' if _ALLOW_PACKED_EXPERT_RTN else 'no production cache'}).",
                        flush=True)
                else:
                    cached_split = _split_packed_expert_tensor(
                        cached_3d, pn, profile)

            expert_input_scale = _packed_expert_input_global_scale(
                full_name, cache=active_cache)
            if (cached_3d is not None and expert_input_scale is None
                    and fmt == "NVFP4"):
                raise RuntimeError(
                    f"[export-inmemory] packed expert {full_name} @ {fmt} has "
                    f"a cached render but no calibrated input_global_scale "
                    f"(would ship the 1.0 placeholder). Re-run "
                    f"build_production_cache to recompute the scale, or delete "
                    f"the expert shard to force a full re-render.")
            if expert_input_scale is not None and fmt == "NVFP4":
                # K0.2: a calibrated stage may only ship alongside its
                # attested sibling stage.
                _packed_expert_stage_attestation(
                    full_name, cache=active_cache, profile=profile)

            # M2: re-derive under the render's RECORDED NVFP4 scale rule
            # (same wrap as the streaming packed-expert path).
            with _temporary_export_nvfp4_scale_rule(packed_render_rule):
                # Index-based on purpose — see the streaming emit: enumerate's
                # result-tuple reuse pins tensor-bearing tuples across layers.
                for pi in range(len(proj_split)):
                    proj_name = proj_split[pi][0]
                    sub_packed = proj_split[pi][1]
                    cached_sub = (
                        cached_split[pi][1]
                        if cached_split is not None else None
                    )
                    E_p, Mp, Np = sub_packed.shape
                    for e in range(E_p):
                        base = f"{disk_qname}.{e}.{proj_name}"
                        if is_bf16:
                            out[f"{base}.weight"], label = _passthrough_tensor(
                                full_name, sub_packed[e])
                        else:
                            expert_2d = (
                                cached_sub[e] if cached_sub is not None
                                else sub_packed[e]
                            )
                            compressed = _quantize_2d(
                                expert_2d, fmt,
                                nvfp4_global_real_override=per_expert_joint[e],
                                input_global_scale_override=expert_input_scale,
                            )
                            for suffix, tensor in compressed.items():
                                out[f"{base}.{suffix}"] = tensor.cpu()
            covered.add(full_name)
            hist[(
                "packed_moe_per_expert",
                _packed_expert_render_hist_label(
                    fmt,
                    is_bf16=is_bf16,
                    source_label=label if is_bf16 else "",
                    cached_3d=cached_3d,
                ),
            )] += 1
            # Same loop-local pinning class as the streaming emit: drop the
            # views so the fp32 stacks free with their owners.
            sub_packed = cached_sub = None

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

# Formats this container emits *without* a `config_groups` scheme, so they
# cannot appear in `FORMAT_SCHEME` by construction. BF16 is written as a
# plain safetensors bf16 tensor and named on the checkpoint's `ignore`
# list. (FP8_SOURCE is *also* a verbatim-copy passthrough — no
# `_quantize_2d` pass runs for it — but it still describes itself to vLLM
# through `FP8_SOURCE_SCHEME`, so `FORMAT_SCHEME` already covers it.)
CONTAINER_PASSTHROUGH_FORMATS = frozenset({"BF16"})

# The authoritative set of formats THIS exporter can emit: everything it
# can describe in `config_groups` metadata plus the container
# passthroughs. Derived, never hand-listed, so a new scheme becomes
# exportable in the same commit that adds it.
#
# It is deliberately NOT derived from the `_quantize_2d` byte-packer
# branches, which do not agree with what is shippable: `FP8_E5M2` has a
# packer branch but no scheme (it raises "research-only", and bytes with
# no metadata are bytes vLLM cannot dispatch — CLAUDE.md gate #9), while
# `FP8_SOURCE` has a scheme and no packer branch at all.
#
# `serving_profile_specs/vllm_packed_moe.json` reads this constant as its
# export-lane bound, which is what keeps the allocator's menu from ever
# containing a rung export would have to rewrite (issue #22 part 2).
# `_coerce_runtime_legal_assignment` hard-fails on anything outside it.
EXPORTABLE_FORMATS = frozenset(
    {_canonical_export_format(name) for name in FORMAT_SCHEME}
    | CONTAINER_PASSTHROUGH_FORMATS
)


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

    This legacy compressed-tensors container intentionally does not publish
    ``execution_contracts.nvfp4_w4a4``.  Its optional/defaulted activation
    scalars preserve existing native artifact bytes but cannot attest the
    strict Gridbook fused-W4A4 activation contract; only the versioned CB
    export path may emit that record after complete calibration.
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
        if fmt == "FP8_SOURCE" and profile.runtime_loads_source_fp8(name):
            # The pinned runtime dequantizes these modules itself at load
            # (source-style keys, module built without a quant config), so
            # they are BF16 in the serving model — not CT-schemed. A
            # config_groups entry here would double-describe them.
            ignore.append(vllm_name)
            continue
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

    def _targets_name_experts(names: list[str]) -> bool:
        """True when any target names a per-expert Linear.

        `by_fmt` holds a mix of plain vLLM module names and pre-formed
        regexes, and by the time this runs every per-expert entry is a
        regex: the loop above moves plain `...experts.<eid>.<proj>` names
        out of `by_fmt` into `packed_fused_states`, then re-adds them via
        `_per_expert_regex_for`, whose literal dots are bracket-escaped
        (`layers[.]0[.]mlp[.]experts[.][0-9]+`). So a bare `".experts."`
        test matches nothing on any packed-MoE artifact. Undo the escape
        before testing, and accept either form.
        """
        for name in names:
            probe = name[len("re:"):] if name.startswith("re:") else name
            probe = probe.replace("[.]", ".")
            if ".experts." in probe or probe.endswith(".experts"):
                return True
        return False

    for fused_qname, states in packed_fused_states.items():
        if len(states) > 1:
            raise RuntimeError(
                f"[export-stream] FusedMoE at {fused_qname!r} has mixed "
                f"states across packed expert projections {states}; vLLM "
                "fuses the experts into one kernel needing a single scheme, so "
                "this is unservable. The allocator's packed-expert format group "
                "should have forced one scheme -- this is the same failure "
                "class as the fused-sibling coherence violation: usually an "
                "allocation produced under the WRONG model profile (probe "
                "lacked meta['model'] -> DefaultProfile never saw the packed-"
                "expert grouping). Re-run the allocator with --model-override "
                "<model> so detect_profile resolves the real profile."
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
        # vLLM fuses these siblings into ONE packed Linear with ONE scheme, so
        # any group whose present siblings carry MIXED formats yields a wrong
        # artifact -- either a crash at load (>=2 distinct quantized schemes) or
        # a silent corruption (quantized + BF16). Collect such groups and fail
        # the export rather than silently emit. Each entry is
        # (fused_vllm_name, kind, {sibling_leaf: format}).
        fused_coherence_violations: list[tuple[str, str, dict[str, str]]] = []
        parents = {p for (p, _) in leaf_state}
        for parent in sorted(parents):  # deterministic violation ordering
            for fused_name, sibs in packed_mapping.items():
                # Skip degenerate fused definitions (single-sibling).
                if len(sibs) < 2:
                    continue
                states = [leaf_state.get((parent, s)) for s in sibs]
                present = [s for s in states if s is not None]
                if not present:
                    continue  # none of this group present here
                fused_vllm_name = f"{parent}.{fused_name}"
                if fused_vllm_name in fused_emitted:
                    continue
                absent_leaves = [sibs[i] for i, s in enumerate(states)
                                 if s is None]
                if absent_leaves:
                    # Incomplete fused group: a sibling is absent from the
                    # checkpoint (e.g. Gemma4 ``attention_k_eq_v`` synthesizes
                    # v=k, so v_proj is never materialized). vLLM fuses the
                    # group into one packed Linear and requires a single scheme
                    # across q/k/v. If every PRESENT sibling is BF16-ignored,
                    # the synthesized absent sibling is BF16 too → emit the
                    # absent siblings AND the fused name into ignore so the
                    # fused module loads uniformly unquantized. (Quantized
                    # incomplete groups are pinned to BF16 upstream by the
                    # allocator's incomplete-fused-group rule, so a
                    # mixed-and-incomplete group should not reach here.)
                    #
                    # A single present state can still be a partial recipe, so
                    # leave it alone. Mixed present states are unambiguous:
                    # adding an absent sibling to a quantized+BF16 or
                    # multi-quant group cannot produce one coherent fused
                    # scheme.
                    if len(set(present)) != 1:
                        members = {s: leaf_state.get((parent, s), "ABSENT")
                                   for s in sibs}
                        quant_states = {s for s in present if s != "IGNORE"}
                        kind = ("crash@load" if len(quant_states) > 1
                                else "silent-corruption")
                        fused_coherence_violations.append(
                            (fused_vllm_name, kind, members))
                        continue
                    if set(present) == {"IGNORE"}:
                        fused_emitted.add(fused_vllm_name)
                        for leaf in absent_leaves:
                            ignore.append(f"{parent}.{leaf}")
                        ignore.append(fused_vllm_name)
                    continue
                if len(set(states)) != 1:
                    # Present siblings disagree -> a fused-coherence violation.
                    # vLLM fuses them into ONE packed Linear with ONE scheme,
                    # and BOTH mixed cases produce a wrong artifact:
                    #  - >=2 distinct QUANTIZED schemes (e.g. FP8 + NVFP4):
                    #    hard CRASH at load (merged-column scale-shape assert);
                    #  - quantized + BF16/IGNORE: LOADS but SILENTLY CORRUPTS
                    #    the merged Linear (the BF16 slice is read under the
                    #    quant sibling's scheme). Measured 4.3x worse served KL
                    #    on Qwen3.x DeltaNet in_proj_ba gate projections
                    #    (0.106 -> 0.025 at matched bpp; surgical isolation).
                    # Refuse to emit either -- a coherent allocation (correct
                    # model profile) never reaches this branch.
                    members = {s: leaf_state.get((parent, s), "ABSENT")
                               for s in sibs}
                    quant_states = {s for s in present if s != "IGNORE"}
                    kind = ("crash@load" if len(quant_states) > 1
                            else "silent-corruption")
                    fused_coherence_violations.append(
                        (fused_vllm_name, kind, members))
                    continue  # do not emit a fused target for a mixed group
                state = states[0]
                fused_emitted.add(fused_vllm_name)
                if state == "IGNORE":
                    ignore.append(fused_vllm_name)
                else:
                    by_fmt.setdefault(state, []).append(fused_vllm_name)

        if fused_coherence_violations:
            detail = "; ".join(
                f"{fused} [{kind}] <- " + ", ".join(
                    f"{leaf}={fmt}" for leaf, fmt in members.items())
                for fused, kind, members in fused_coherence_violations)
            raise RuntimeError(
                "fused-sibling coherence violation: "
                f"{len(fused_coherence_violations)} merged-column group(s) "
                "carry MIXED formats among their siblings. vLLM fuses each "
                "group into one packed Linear with one scheme, so a mix either "
                "CRASHES at load (>=2 distinct quantized schemes -> scale-shape "
                "assert) or SILENTLY CORRUPTS the merged Linear (quantized + "
                "BF16 -> the BF16 slice is read under the quant scheme; "
                f"measured 4.3x worse served KL on DeltaNet gates): {detail}. "
                "This is almost always an allocation produced under the WRONG "
                "model profile -- e.g. the probe lacked meta['model'], so the "
                "allocator fell back to DefaultProfile and never saw this "
                "architecture's fused group (in_proj_ba / in_proj_qkvz on "
                "Qwen3.x DeltaNet, etc.). Re-run the allocator with "
                "--model-override <model> (or rebuild the probe with "
                "meta['model'] set) so detect_profile resolves the real "
                "profile and promote_fused coerces each group to one format.")

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
        # Only attach the safety-net when the catch-all group actually
        # holds packed/per-expert units. When the largest group is a
        # non-expert format (GLM-5.3: FP8_SOURCE dense half vs NVFP4
        # experts), the net would claim every per-expert Linear for the
        # WRONG scheme — vLLM's get_moe_method probes `experts.0.X_proj`
        # and a match here overrides the experts' real group.
        catchall_has_experts = _targets_name_experts(by_fmt[catchall])
        if catchall_has_experts:
            for getter in (profile.per_expert_moe_regex,
                           profile.per_expert_mtp_regex):
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


def _preflight_quantization_config(
    assignment: dict[str, str],
    bf16_passthrough: set[str],
    *,
    profile: "ModelProfile | None",
) -> None:
    """Run config-only export gates before GPU render and shard writes."""
    try:
        build_quantization_config(
            assignment,
            bf16_passthrough,
            profile=profile,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[export-stream] quantization-config preflight failed before "
            f"rendering or shard writes: {exc}"
        ) from exc


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


def _render_lever_provenance() -> dict:
    """The quality-affecting render state of this export run.

    Two consumers, one definition: the per-layer export cache binds its
    `manifest.json` to this dict (a change means the cached layer tensors were
    quantized under a different recipe and the cache is silently wrong), and the
    shipcard echoes it so an artifact carries the levers it was rendered under.
    Keep the key set stable — changing it invalidates every in-flight export
    cache.
    """
    return {
        "PRISMAQUANT_DO_NO_HARM": os.environ.get(
            "PRISMAQUANT_DO_NO_HARM", "1"),
        "PRISMAQUANT_GPTQ_DAMP_SWEEP": os.environ.get(
            "PRISMAQUANT_GPTQ_DAMP_SWEEP", "0"),
        "PRISMAQUANT_GPTQ_DAMP": os.environ.get(
            "PRISMAQUANT_GPTQ_DAMP", ""),
        "PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING": os.environ.get(
            "PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING", "0"),
        "PRISMAQUANT_ACT_CLIP_QUANTILE": os.environ.get(
            "PRISMAQUANT_ACT_CLIP_QUANTILE", "0.999"),
        # Walled 2026-07-30 (re-vet R25): the lever no longer exists, so the
        # key records that fact rather than echoing an env var nothing reads.
        # This DOES move the fingerprint once — intended: any in-flight export
        # cache was built by a binary that still carried the (unreachable)
        # branch, and re-rendering is the honest outcome.
        "PRISMAQUANT_BLOCK_OUTPUT_MATCH": "archived_2026-07-30",
        "PRISMAQUANT_BATCHED_NVFP4_EXPORT": os.environ.get(
            "PRISMAQUANT_BATCHED_NVFP4_EXPORT", "1"),
        NVFP4_SCALE_RULE_ENV: _nvfp4_scale_rule_from_env(),
        "ACT_AWARE_FLAGS": dict(sorted(_ACT_AWARE_FLAGS.items())),
        "activation_cache_fingerprint": _ACTIVATION_CACHE_FINGERPRINT,
        "production_cache_fingerprint": _PRODUCTION_CACHE_FINGERPRINT,
    }


def _write_shipcard(
    out_dir: Path,
    *,
    source_model: str,
    layer_config_path: str | None,
    assignment: dict,
    config_assignment: dict,
    hist: dict,
) -> None:
    """Open the ship record (R13): build-lane facts + empty serve-lane slots.

    The build lane cannot run a quality gate — `vllm` is not importable in the
    build venv, and embedding a docker serve here would make the exporter own
    the serving stack. What it can do is state, on the artifact, exactly which
    serve-lane verdicts are still missing, so "we never ran the ship gate"
    becomes a refusal (`python -m prismaquant.shipcard_cli verify`) instead of an omission.
    """
    import hashlib

    from . import read_traffic as _read_traffic
    from . import shipcard as _shipcard

    def _hash(payload) -> str | None:
        try:
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()[:16]
        except Exception:
            return None

    build = {
        "git": _shipcard.git_provenance(),
        "source_model": source_model,
        "layer_config": layer_config_path,
        "layer_config_sha": (
            _shipcard.file_sha256(layer_config_path)
            if layer_config_path else None),
        "assignment_hash": _hash(assignment),
        "config_assignment_hash": _hash(config_assignment),
        "n_assignment_entries": len(config_assignment),
        "achieved_bpp": _shipcard.allocator_achieved_bpp(layer_config_path),
        # Disk bytes are not what decode throughput is made of: a dense
        # weight is streamed every token while a routed expert stack is
        # streamed topk/E of the time, so bpp and per-token read bytes rank
        # artifacts differently on a sparse MoE. Stamped beside the bpp,
        # measured from the shards this export just wrote.
        "read_gb_per_token": _read_traffic.read_traffic_claim(out_dir),
        "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
        "render_levers": _render_lever_provenance(),
        "kv_shared_fisher": _shipcard.kv_shared_fisher_echo(),
    }
    card = _shipcard.build_shipcard(out_dir, build=build)
    path = _shipcard.write_shipcard(
        out_dir / _shipcard.SHIPCARD_FILENAME, card)
    print(f"[export-stream] shipcard opened: {path}", flush=True)
    print(f"[export-stream]   serve-lane slots still UNFILLED: "
          f"{', '.join(_shipcard.unfilled_slots(card))}", flush=True)
    print(f"[export-stream]   close them, then: python3 -m prismaquant.shipcard_cli "
          f"verify {path}", flush=True)


def _refuse_archived_block_output_match() -> None:
    """Fail loudly if an old script still asks for block-output match.

    Walled 2026-07-30 (re-vet R25, archive/block_output_match_2026-07-30/).
    Silently ignoring `PRISMAQUANT_BLOCK_OUTPUT_MATCH=1` would mean an old
    launcher exports *differently* than it did with no signal at all — the
    band-aid this house forbids. `=0` (the explicit disable) is accepted: it
    already asked for what now always happens.
    """
    raw = os.environ.get("PRISMAQUANT_BLOCK_OUTPUT_MATCH")
    if raw is None:
        return
    if str(raw).strip().lower() in {"0", "false", "no", "off", ""}:
        return
    raise SystemExit(
        "[export] ERROR: PRISMAQUANT_BLOCK_OUTPUT_MATCH="
        f"{raw} — block-output match is archived under "
        "archive/block_output_match_2026-07-30. It was UNREACHABLE on the "
        "shipping recipe (the production-cache pack fires first and "
        "`continue`s, so with PRODUCTION_CACHE=1 no dense NVFP4 Linear ever "
        "reached the branch; zero hits in two real production export logs), "
        "and had it been reached it would have re-derived NVFP4 group scales "
        "outside _export_match_render_scale_rule and discarded the render's "
        "joint_mse scales — the -6.6% KL defect M19 fixed everywhere else. "
        "Its {0.95, 1.0, 1.05} per-tensor gain re-search is subsumed by JSO. "
        "Unset the variable (or set it to 0) to export."
    )


def _replace_cli_option(argv: Sequence[str], option: str, value: str) -> list[str]:
    """Replace every spelling of one required argparse option."""
    rewritten = list(argv)
    found = False
    index = 0
    while index < len(rewritten):
        item = rewritten[index]
        if item == option:
            if index + 1 >= len(rewritten):
                break
            rewritten[index + 1] = value
            found = True
            index += 2
            continue
        if item.startswith(option + "="):
            rewritten[index] = option + "=" + value
            found = True
        index += 1
    if not found:
        raise RuntimeError(f"required CLI option {option!r} was not found")
    return rewritten


def _cleanup_successful_export_cache(
    export_cache_dir: str | None,
    *,
    keep_export_cache: bool,
) -> None:
    """Remove resumable layer state only after final artifact publication."""
    if (
        not export_cache_dir
        or keep_export_cache
        or not Path(export_cache_dir).exists()
    ):
        return
    try:
        shutil.rmtree(export_cache_dir)
        print(
            f"[export-stream] removed export cache {export_cache_dir}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[export-stream] WARN cache cleanup failed: {exc!r}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None):
    """Run one native export inside an artifact-level output transaction."""
    raw_argv = list(argv) if argv is not None else None
    parse_argv = raw_argv if raw_argv is not None else os.sys.argv[1:]
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument("--model")
    preflight.add_argument("--output")
    preflight.add_argument("--export-cache-dir")
    preflight.add_argument("--keep-export-cache", action="store_true")
    known, _unknown = preflight.parse_known_args(parse_argv)
    if known.model is None or known.output is None:
        # Preserve the complete parser's native --help/missing-argument errors.
        return _main_impl(parse_argv)

    # Refuse a stale/tampered or merely MATERIALIZED PrismaSnap source before
    # opening the export transaction or doing any GPU render work.  Ordinary
    # unsnapped sources remain a no-op in this preflight.
    from .prismasnap_contract import require_verified_prismasnap_if_present

    require_verified_prismasnap_if_present(known.model)

    requested_output = Path(known.output)
    with transactional_export_directory(
        known.model,
        requested_output,
        where="export_native_compressed",
    ) as staged_output:
        staged_argv = _replace_cli_option(
            parse_argv,
            "--output",
            str(staged_output),
        )
        result = _main_impl(staged_argv)

    _cleanup_successful_export_cache(
        known.export_cache_dir,
        keep_export_cache=bool(known.keep_export_cache),
    )
    print(
        "[export-stream] done. Serve with:\n"
        f"  vllm serve {requested_output.resolve()} "
        "--quantization compressed-tensors",
        flush=True,
    )
    return result


def _main_impl(argv: Sequence[str] | None = None):
    global _INPUT_GLOBAL_SCALES, _CACHED_ACTIVATIONS, _ACTIVATION_CACHE_FINGERPRINT
    global _PRODUCTION_WEIGHT_CACHE, _PRODUCTION_CACHE_FINGERPRINT
    global _PRODUCTION_CACHE_PREFETCH_WORKERS, _NVFP4_SCALE_RULE
    global _PRODUCTION_CACHE_PREFETCH_MODE, _ALLOCATOR_TARGET_PROFILE
    _refuse_archived_block_output_match()
    _INPUT_GLOBAL_SCALES = None
    _CACHED_ACTIVATIONS = None
    _ACTIVATION_CACHE_FINGERPRINT = None
    _PRODUCTION_WEIGHT_CACHE = None
    _PRODUCTION_CACHE_FINGERPRINT = None
    _NVFP4_SCALE_RULE = resolve_nvfp4_scale_rule()
    _DO_NO_HARM_STATS.clear()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model dir (source safetensors + config.json)")
    ap.add_argument("--layer-config", default=None,
                    help="layer_config.json from allocator.py. Optional when "
                         "--perturbed-x-dir is supplied.")
    ap.add_argument(
        "--allow-research-cost-selection",
        action="store_true",
        help="explicitly acknowledge export of a research-stamped assembled "
             "cost selection; the default ship gate refuses it",
    )
    ap.add_argument("--output", required=True,
                    help="Output directory for the compressed checkpoint")
    ap.add_argument("--shard-bytes", type=int, default=1024**3,
                    help="Approx per-shard size in bytes (default 1 GiB). "
                         "Robert's standing packaging preference for every "
                         "published artifact since 2026-08-20; it was 5 GiB "
                         "before. Smaller shards resume better on a flaky "
                         "download and let a client fetch a subset, at the "
                         "cost of more files in the index. A single tensor "
                         "larger than this still gets its own shard -- the "
                         "writer flushes it whole rather than splitting it.")
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
                         "lm_head/head for current vLLM serving targets), "
                         "except when allocator metadata explicitly stamps "
                         "a fixed or DP-unpinned lm_head assignment. "
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
    ap.add_argument("--production-cache-prefetch",
                    choices=("require", "warn"), default="warn",
                    help="`require` fails the export when the production "
                         "cache cannot supply a layer's assignment instead of "
                         "silently falling back to per-tensor NVMe reads "
                         "(re-vet R24/D8). run-pipeline.sh passes require on "
                         "the native lane; the bare-CLI default stays warn.")
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
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    # Path safety is the first preflight: an in-place/aliased/stale target must
    # be rejected before parsing model tensors or allocating any exporter state.
    # Directory creation remains delayed until the recipe/config gates pass.
    validate_fresh_export_directory(
        args.model,
        out_dir,
        where="export_native_compressed",
    )

    from .model_profiles import detect_profile, DefaultProfile
    profile = detect_profile(args.model)
    print(f"[export-stream] model profile: {profile.name}", flush=True)
    if isinstance(profile, DefaultProfile):
        # The export's fused-coherence gate keys off this profile's
        # packed_modules_mapping plus fallback fused groups. DefaultProfile
        # checks qkv_proj/gate_up_proj and the known DeltaNet
        # in_proj_ba/in_proj_qkvz groups, but unknown architecture-specific
        # merged columns and packed-MoE expert constraints are still invisible
        # when --model's architecture is not registered. Make that blind spot
        # loud.
        print(
            "[export-stream] WARNING: resolved DefaultProfile for "
            f"{args.model!r} -- architecture not registered. Fused-coherence "
            "checks cover only qkv_proj/gate_up_proj; arch-specific merged "
            "columns / packed-MoE experts are NOT verified and may ship "
            "silently corrupt. Register a ModelProfile for this architecture "
            "or confirm it is a vanilla transformer.", flush=True)

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
    from .research_cost_acceptance import enforce_research_export_acknowledgement
    enforce_research_export_acknowledgement(
        _layer_config_payload_for_cache,
        acknowledged=args.allow_research_cost_selection,
        where="export_native_compressed",
    )
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
        _PRODUCTION_CACHE_PREFETCH_MODE = str(
            args.production_cache_prefetch or "warn").lower()
        expected_keys, missing_keys = _production_cache_expected_keys(
            _assignment_for_cache
        )
        if missing_keys:
            mtp_missing = [k for k in missing_keys
                           if str(k[0]).startswith("mtp.")]
            mtp_hint = ""
            if mtp_missing:
                # Fail at attach time with the producer contract, not hours
                # later in _materialize_tensors_inmemory. The shared cache
                # builder can synthesize profile MTP now, but it needs the
                # probe activation cache and the concrete non-BF16 assignment.
                mtp_hint = (
                    f" {len(mtp_missing)} of these are MTP sidecar entries "
                    f"(e.g. {mtp_missing[0][0]}). Rebuild with the current "
                    "build_production_cache, the same --render-layer-config, "
                    "and --activation-cache-dir from the probe so its "
                    "profile-synthesized MTP producer can render them; or "
                    "set MTP_FORMAT=BF16."
                )
            raise RuntimeError(
                "[export-stream] production-weight-cache missing recipe "
                f"entries: {len(missing_keys)} sample={missing_keys[:8]}."
                + mtp_hint
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
        _INPUT_GLOBAL_SCALES = _production_cache_scales(
            production_cache,
            profile=profile,
        )
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
            _recipe_names = [n for n in _recipe_payload.keys()
                             if not _is_layer_config_meta_key(n)]
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
            # reciprocal (FP8_MAX·FP4_MAX/max_abs) values can drift by
            # a float-precision tick. Take the MIN over the group —
            # the smallest reciprocal == the largest max_abs == the
            # conservative (loosest-clipping) scale for every sibling.
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
    _allocator_meta = _layer_config_metadata(raw_recipe)
    if _allocator_meta.get("target_profile"):
        _ALLOCATOR_TARGET_PROFILE = str(_allocator_meta["target_profile"])
        print(f"[export] allocator target profile (from layer_config): "
              f"{_ALLOCATOR_TARGET_PROFILE}", flush=True)
    assignment = _canonicalize_assignment(raw_recipe)
    assignment, runtime_coerced = _coerce_runtime_legal_assignment(
        args.model,
        assignment,
        profile,
    )
    if runtime_coerced:
        print(_runtime_coercion_report(runtime_coerced), flush=True)
    whole_artifact_budget_from_assignment_payload(
        raw_recipe,
        where="export_native_compressed preflight",
        assignment=assignment,
    )
    validate_mtp_assignment_coverage(args.model, assignment, profile)
    fmts = Counter(assignment.values())
    print(f"[export-stream] recipe: {len(assignment)} entries  mix={dict(fmts)}",
          flush=True)

    bf16_passthrough = _bf16_passthrough_for_assignment(
        args.ignore,
        profile,
        _allocator_meta,
    )
    config_assignment, config_bf16_passthrough, fp8_source_overrides = (
        _fp8_source_config_overlay(
            args.model,
            assignment,
            bf16_passthrough,
            profile,
        )
    )
    if fp8_source_overrides:
        print(
            "[export-stream] config FP8_SOURCE passthrough overrides: "
            f"{len(fp8_source_overrides)} source-FP8 Linears",
            flush=True,
        )
    _preflight_quantization_config(
        config_assignment,
        config_bf16_passthrough,
        profile=profile,
    )
    print("[export-stream] quantization-config preflight passed", flush=True)
    out_dir = prepare_fresh_export_directory(
        args.model,
        out_dir,
        where="export_native_compressed",
    )

    from prismaquant.gpu_guard import require_cuda_hot_path

    dtype = torch.bfloat16
    device = require_cuda_hot_path(
        "export_native_compressed",
        args.device,
    )
    if args.offload_folder is None:
        args.offload_folder = str(out_dir / "_streaming_offload")

    def _rename_body_batch(
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Body-walk keys are LIVE module qnames. `export_tensor_name`
        # speaks the RECIPE namespace, so normalize first — a no-op for
        # every family whose live skeleton already uses recipe names
        # (`live_to_recipe_name` is idempotent prefix/segment rewriting),
        # and the only correct spelling on multimodal-forced skeletons
        # (glm5_next: `model.language_model.` + `.forget_gate.` segments
        # exist live but not in the recipe or the checkpoint).
        return {
            profile.export_tensor_name(profile.live_to_recipe_name(k)): v
            for k, v in batch.items()
        }

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
            args.model, assignment, profile=profile,
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

    extra_ignore = compute_extra_ignore(
        _source_shape_iter(),
        config_assignment,
        profile,
    )
    print(f"[export-stream] extra ignore (unmapped Linears): "
          f"{len(extra_ignore)}", flush=True)

    write_config_with_quantization(
        args.model, out_dir, config_assignment, config_bf16_passthrough,
        extra_ignore=extra_ignore,
        transform_config=None)
    _copy_tokenizer(args.model, out_dir)

    with open(out_dir / "mixed_native_manifest.json", "w") as f:
        json.dump({
            "source_model": args.model,
            "source_recipe": args.layer_config,
            "format_histogram": {f"{k[0]}/{k[1]}": v for k, v in hist.items()},
            "n_assignment_entries": len(config_assignment),
            "source_assignment_entries": len(assignment),
            "fp8_source_passthrough_overrides": sorted(fp8_source_overrides),
            "runtime_coercions": _runtime_coercion_manifest_rows(runtime_coerced),
            "bf16_audit": _bf16_upgrade_audit(
                args.model,
                config_assignment,
                config_bf16_passthrough,
                runtime_coerced,
                profile,
            ),
            "do_no_harm": dict(_DO_NO_HARM_STATS),
            "packed_expert_export": _packed_expert_export_provenance(),
            "ignore": sorted(config_bf16_passthrough),
        }, f, indent=2)

    # R13: open the ship record. Build-lane facts are final here; the
    # serve-lane slots stay empty until the validators and the gold lane fill
    # them (docs/ARCHITECTURE.md §7.1).
    try:
        _write_shipcard(
            out_dir,
            source_model=args.model,
            layer_config_path=args.layer_config,
            assignment=assignment,
            config_assignment=config_assignment,
            hist=hist,
        )
    except Exception as e:
        print(f"[export-stream] WARN shipcard not written: {e!r}", flush=True)

    budget_attestation = enforce_whole_artifact_budget(
        out_dir,
        raw_recipe,
        where="export_native_compressed",
        assignment=assignment,
    )
    if budget_attestation is not None:
        print(
            "[export-stream] whole-artifact budget passed: "
            f"{budget_attestation['actual_bytes']}B <= "
            f"{budget_attestation['budget_bytes']}B",
            flush=True,
        )

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
        if os.path.lexists(self.out_dir):
            if self.out_dir.is_symlink() or not self.out_dir.is_dir():
                raise RuntimeError(
                    f"{self.out_dir}: native shard output must be a real "
                    "directory, not a file or symlink"
                )
        else:
            self.out_dir.mkdir(parents=True, exist_ok=False)
        stale_temps = sorted(
            path.name
            for path in self.out_dir.iterdir()
            if re.fullmatch(
                r"[.]model-[0-9]+[.]safetensors[.]tmp",
                path.name,
            ) is not None
        )
        if stale_temps:
            raise RuntimeError(
                f"{self.out_dir}: refusing preexisting native temporary "
                f"shard(s) {stale_temps[:12]}; use a fresh output directory"
            )

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
        if os.path.lexists(tmp_path):
            raise RuntimeError(
                f"{tmp_path}: native temporary shard appeared during "
                "export; refusing to overwrite a concurrent/stale writer"
            )
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

    @staticmethod
    def _is_model_artifact_name(name: str) -> bool:
        """Files this writer owns and may replace across export attempts."""
        return (
            name in {"model.safetensors", "model.safetensors.index.json"}
            or re.fullmatch(
                r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name
            ) is not None
        )

    def _remove_stale_model_artifacts(self, planned_names: set[str]) -> None:
        """Remove only obsolete model containers from an earlier export.

        Tokenizer/config/processor files are intentionally outside this
        writer's ownership. A directory at a reserved model filename is an
        error rather than a recursive-delete target.
        """
        removed: list[str] = []
        for path in self.out_dir.iterdir():
            if (
                not self._is_model_artifact_name(path.name)
                or path.name in planned_names
            ):
                continue
            if path.is_dir() and not path.is_symlink():
                raise RuntimeError(
                    f"cannot replace stale model artifact {path}: expected a "
                    "file, found a directory"
                )
            path.unlink()
            removed.append(path.name)
        if removed:
            print(
                "[export-stream] removed stale model artifact(s): "
                f"{sorted(removed)}",
                flush=True,
            )

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
            self._remove_stale_model_artifacts({final_name})
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
        self._remove_stale_model_artifacts(
            {"model.safetensors.index.json", *set(self.weight_map.values())}
        )
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
                             profile,
                             bf16_passthrough: set[str],
                             hist: dict,
                             device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Quantize MTP weights per the allocator recipe.

    Transformers v5 does not instantiate MTP modules when loading
    Qwen3.5/3.6 MoE checkpoints (see `_keys_to_ignore_on_load_unexpected`),
    so the streaming decoder-layer sweep never sees any `mtp.*` entry in
    `assignment`. We ask the model profile to build a standalone MTP
    module, load the source MTP weights into it (the source prefix is
    the profile's `mtp_source_prefix()`), wrap it in a parent module
    named `mtp` (so qualified names come out as `mtp.fc`,
    `mtp.layers.0.self_attn.q_proj`, ... per `build_mtp_module`'s
    naming contract), and run the in-memory materialize helper.

    Output tensor names match the checkpoint convention (`mtp.fc.*`,
    `mtp.layers.0.<rest>`). vLLM's `qwen3_5_mtp.load_weights` remaps
    `mtp.→model.` at load time.
    """
    from transformers import AutoConfig

    # Build an MTP wrapper with source weights.
    cfg = AutoConfig.from_pretrained(src_model, trust_remote_code=True)
    text_config = getattr(cfg, "text_config", cfg)
    inner = profile.build_mtp_module(text_config)
    if inner is None:
        raise RuntimeError(
            f"profile '{profile.name}' declares has_mtp() but "
            f"build_mtp_module() returned None — MTP tensors cannot be "
            f"rendered. Either implement build_mtp_module() or take the "
            f"passthrough route (has_mtp() -> False + "
            f"source_passthrough_prefixes()).")
    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=torch.bfloat16)
    raw = profile.read_mtp_source_state_dict(src_model)
    profile.load_mtp_state_dict(inner, raw)
    # Move the whole MTP module to the export device so
    # _materialize_tensors_inmemory's per-linear quant runs on GPU when
    # EXPORT_DEVICE=cuda. Previously defaulted to CPU, costing ~10× on
    # MTP quant. The input weights (raw) are CPU, so we move after load.
    wrapper.to(device=device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Filter assignment to just `mtp.*` entries. Recipe-name prefix, not
    # the source prefix: `build_mtp_module`'s contract fixes it at `mtp.`.
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
            raise RuntimeError(
                f"[export-stream] visual quant failed for {base} ({fmt}); "
                "refusing to emit BF16 bytes under a quantized config"
            ) from e
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

    # PrismaSnap is an additive source-preparation pass.  Its compact,
    # self-digested provenance must survive the otherwise unchanged native
    # exporter so the final shipcard's model hash binds the treatment.  The
    # absent-source path performs no write and remains byte-for-byte identical
    # to every pre-PrismaSnap export.
    snap_path = src / "prismasnap_provenance.json"
    if os.path.lexists(snap_path):
        if snap_path.is_symlink() or not snap_path.is_file():
            raise RuntimeError(
                f"PrismaSnap provenance is not a regular file: {snap_path}"
            )
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"PrismaSnap provenance is unreadable: {snap_path}"
            ) from exc
        if (
            not isinstance(snap, dict)
            or snap.get("schema")
            not in {
                "prismaquant.prismasnap.provenance.v1",
                "prismaquant.prismasnap.provenance.v2",
            }
        ):
            raise RuntimeError(
                f"PrismaSnap provenance has an unsupported contract: {snap_path}"
            )
        from .prismasnap_validation import (
            validate_prismasnap_checkpoint,
            validate_prismasnap_provenance_payload,
        )

        # Export can run for hours.  Replay the index/shard content identity
        # immediately before the provenance is copied into the staged output;
        # a source mutation after preflight must abort the transaction.
        validate_prismasnap_checkpoint(src, require_verified=True)

        validate_prismasnap_provenance_payload(
            snap,
            require_verified=True,
            where=f"native export PrismaSnap provenance {snap_path}",
        )
        # This receipt describes the BF16 *source* tree, not the compressed
        # output shards.  Preserve it under an unambiguous name so the final
        # artifact never claims its own bytes are the materialized checkpoint.
        shutil.copy2(snap_path, out_dir / "source_prismasnap_provenance.json")


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
    src_prefix = profile.mtp_source_prefix()
    if not src_prefix or not _source_has_prefixed_weights(src_model, src_prefix):
        return
    # Recipe names are always `mtp.*` — `build_mtp_module`'s contract
    # wraps the module in a parent named `mtp` regardless of what prefix
    # the source checkpoint used.
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
