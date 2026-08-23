#!/usr/bin/env python3
"""sensitivity_probe.py — per-Linear empirical Fisher diagonal trace.

What this measures
------------------
For each tracked Linear with weight W, this script estimates the
per-token empirical Fisher diagonal trace

    H_trace = Σ_w E_token[(∂L/∂W_w)²]

where L is the per-token negative log-likelihood (same loss a language
model is trained against) and the expectation is over a calibration
corpus. This quantity is used by allocator.py as the sensitivity score
in the closed-form predicted Δloss

    Δloss ≈ 0.5 · H_trace · MSE_W                           (eq. 3 in
                                                             allocator.py)

Naming. The literature uses several names for E[(∂L/∂W)²]: "empirical
Fisher", "Gauss-Newton diagonal", "gradient-squared". This is NOT the
true Hessian diagonal — for that you would need a vHv-style probe
(Hutchinson). The empirical Fisher equals the Hessian only at a true
loss minimum, which a calibration corpus does not in general satisfy.
For ranking layers and predicting first-order quantization sensitivity,
empirical Fisher is the standard HAWQ-V1 choice and works well.

How the per-token estimator stays unbiased
------------------------------------------
HuggingFace's `out.loss` is `mean(CE)` over the T tokens in a batch, so
its gradient is `(1/T) · Σ_t ∂CE_t/∂W`. Squaring that under-estimates
per-token Fisher by a factor of T (under the standard assumption of
independent per-token gradients). To avoid that, we reconstruct CE with
`reduction="sum"` for the backward pass; the gradient then aggregates
per-token gradients without the 1/T factor, and dividing the
accumulated `||grad_W||²_F` by total tokens recovers the per-token
Fisher trace estimator directly.

Other features:
  - route-aware MoE normalization (discover routers by walking module
    tree; each expert's H_trace is divided by its ROUTED token count,
    i.e. the per-routed-token mean — the single implicit ÷token-fraction
    convention. This matches `incremental_probe`, the production
    backend: the two backends now agree on expert h_trace. The observed
    routing probability is recorded as `route_prob` metadata only; it is
    no longer applied as a second division, which used to make this
    backend disagree with the incremental one by ~1/p_e — audit M4.)
  - per-token importance weighting (harder tokens count more); this
    reweights the loss but preserves the per-token-Fisher units when
    used with sum reduction
  - activation snapshot cache for measure_quant_cost.py

Memory:
  - params requires_grad_(False)   → no gradient tensor storage
  - gradient checkpointing on      → activations are recomputed during backward
  - backward hooks reduce grad_w to a scalar inline and drop it
Result: peak ≈ model weights + one-block activation, fits in 128 GB for 35 B.

Model-agnostic:
  - Router discovered via module walk (any Linear whose out_features equals a
    sibling ModuleList named experts, gates, etc.)
  - Top-k read from model.config (num_experts_per_tok)
  - Dense models just skip RouterTracker
"""
from __future__ import annotations

import json
import atexit
import os
import pickle
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .name_projection import NameProjection


_STAGED_TEMP_DIRS: list[Path] = []


def _prismaquant_temp_parent() -> Path:
    root = (
        os.environ.get("PRISMAQUANT_TMPDIR")
        or os.environ.get("TMPDIR")
    )
    parent = Path(root) if root else Path.cwd() / ".prismaquant_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _mk_stage_dir(prefix: str) -> Path:
    staged = Path(tempfile.mkdtemp(
        prefix=prefix,
        dir=str(_prismaquant_temp_parent()),
    ))
    _STAGED_TEMP_DIRS.append(staged)
    return staged


def _cleanup_stage_dirs() -> None:
    for path in reversed(_STAGED_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_stage_dirs)


# ---------------------------------------------------------------------------
# Text-only staging
# ---------------------------------------------------------------------------
def stage_text_only(model_path: str) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Profile-driven: ask the registered ModelProfile which config keys
    # to strip and whether to promote `text_config.model_type`.
    try:
        from .model_profiles import detect_profile
        profile = detect_profile(str(src))
    except Exception:
        profile = None
    strip_keys = (list(profile.stage_text_only_strip_keys())
                  if profile is not None
                  else ["vision_config", "audio_config", "speech_config",
                        "image_token_id", "video_token_id",
                        "vision_start_token_id", "vision_end_token_id"])
    # Run staging when either (a) the model is multimodal (has a
    # config section that must be stripped for text-only load),
    # (b) any of the profile-declared strip_keys is present, or
    # (c) the config declares `num_local_experts` — we need to
    # alias it as `num_experts` for transformers 5.x's FP8 MoE
    # integration, which does `getattr(cfg, "num_local_experts",
    # cfg.num_experts)` with the fallback arg eagerly evaluated and
    # raises on the attribute access when only the long-form name
    # is defined.
    needs_num_experts_alias = (
        "num_local_experts" in cfg and "num_experts" not in cfg)
    if not any(k in cfg for k in
               ("vision_config", "text_config", "audio_config", "speech_config")) \
            and not any(k in cfg for k in strip_keys) \
            and not needs_num_experts_alias:
        return str(src)
    promote_inner_mt = (profile.stage_text_only_promote_inner_model_type()
                        if profile is not None else False)

    import tempfile
    for k in strip_keys:
        cfg.pop(k, None)

    # Transformers 5.x's `integrations/finegrained_fp8.py:__init__`
    # does `getattr(config, "num_local_experts", config.num_experts)` —
    # the fallback is evaluated eagerly and raises on MoE configs that
    # only declare `num_local_experts` (e.g. MiniMax M2). Alias both
    # names so the attribute access always succeeds.
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]

    if "text_config" in cfg:
        tc = cfg.pop("text_config")
        for k, v in tc.items():
            if k == "model_type":
                # Some families (Gemma 4) want the inner model_type
                # (e.g. gemma4_text) to take over so AutoConfig
                # resolves to the text-specific config class. Others
                # (Qwen 3.5 MoE) want the outer model_type to stay
                # because the ForCausalLM class is annotated with the
                # multimodal-umbrella config. Profile decides.
                if promote_inner_mt:
                    cfg[k] = v
                continue
            # text_config OVERRIDES top-level for the text-only staged
            # model. Multimodal models (Gemma 4, some Qwen variants)
            # carry their multimodal-combined hidden_size / num_layers /
            # head_dim at the top level and the text-specific values in
            # text_config. Loading weights needs the text_config values
            # since that's what the text checkpoint tensors were trained
            # with. If we didn't override, the loader sees mismatched
            # shapes ([2304] top-level vs [2816] actual weights) and
            # either raises or silently zero-inits the mismatched params.
            cfg[k] = v
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]

    staged = _mk_stage_dir("prismaquant_stage_")
    skip = {"config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "processor_config.json"}
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


# ---------------------------------------------------------------------------
# Multimodal staging — Phase 2 visual calibration (mirror of stage_text_only)
# ---------------------------------------------------------------------------
def stage_multimodal(model_path: str) -> str:
    """Stage a multimodal model PRESERVING `vision_config` (and `audio_config`
    where present). This is the mirror of `stage_text_only`, used by the
    Phase 2 visual Fisher probe path: AutoModelForCausalLM.from_pretrained
    must see the complete multimodal config so the visual tower actually
    materializes; AutoProcessor must see the preprocessor_config shard
    to tokenize image+text pairs.

    Unlike `stage_text_only` we don't strip multimodal keys, don't promote
    `text_config.model_type`, and don't rewrite architectures. We symlink
    every source file (including preprocessor_config.json,
    video_preprocessor_config.json, processor_config.json) into the
    staged dir verbatim.

    When the source has no multimodal config keys, this is effectively a
    no-op and we return the source path unchanged — matching
    `stage_text_only`'s fast path.
    """
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    # If this checkpoint has no multimodal bits, nothing to do — return
    # the source directly so the caller doesn't pay for a symlink tree
    # on pure-text checkpoints.
    if not any(k in cfg for k in ("vision_config", "audio_config",
                                  "speech_config")):
        return str(src)

    staged = _mk_stage_dir("prismaquant_mm_stage_")
    for p in src.iterdir():
        if p.name == "config.json":
            continue
        (staged / p.name).symlink_to(p.resolve())
    # Write the source config through verbatim. We intentionally do NOT
    # strip vision_config/audio_config, do NOT rewrite architectures,
    # and do NOT promote text_config.model_type — the multimodal loader
    # needs the full config as-is.
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


# ---------------------------------------------------------------------------
# Multimodal calibration loader — Phase 2 visual Fisher
# ---------------------------------------------------------------------------
def _samples_from_encoding(
    enc: dict,
    max_text_len: int,
) -> dict | None:
    """Turn a processor(...) output dict into a forward-kwargs dict
    suitable for `model(**sample)`. Preserves every tensor the processor
    emitted (including vision-specific keys like `image_grid_thw`,
    `video_grid_thw`, `attention_mask`) and adds `labels` = `input_ids`.

    Returns None on malformed encodings (missing `pixel_values` or
    `input_ids`).
    """
    pixel_values = enc.get("pixel_values")
    input_ids = enc.get("input_ids")
    if pixel_values is None or input_ids is None:
        return None
    if input_ids.size(-1) > max_text_len:
        input_ids = input_ids[..., :max_text_len]
    sample: dict = {}
    for k, v in dict(enc).items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[..., :max_text_len] if k == "input_ids" else v
    sample["labels"] = input_ids.clone()
    return sample


def _synthetic_multimodal_calibration_samples(
    processor, n_samples: int, max_text_len: int,
) -> list[dict]:
    """Offline synthetic fallback: n small PIL images + short captions
    fed through the processor. Exercises the visual Fisher path without
    needing network access. Real COCO/HF datasets replace this when
    `--mm-dataset` is a HuggingFace id.

    Each sample is a 224x224 RGB image with a deterministic flat color
    (enough to make patch-embed + attention gradients non-trivial) and
    a short caption matched to its index.

    Returns a list of `dict` — each contains every tensor the processor
    emitted (`pixel_values`, `input_ids`, `attention_mask`, `image_grid_thw`,
    etc. — schema-dependent) plus a `labels` key. Caller unpacks as
    `model(**sample)`. Was originally a `(pixel_values, input_ids, labels)`
    tuple, which lost vision-specific kwargs; Qwen3.6's multimodal
    forward requires `image_grid_thw`.
    """
    try:
        from PIL import Image
    except Exception:
        return []
    captions = [
        "a red apple on a wooden table",
        "a blue sky with scattered clouds",
        "a cat sleeping on a patterned rug",
        "a yellow sunflower in a green field",
        "a coffee mug next to an open book",
        "a golden retriever chasing a frisbee",
        "a mountain lake reflecting tall peaks",
        "a busy city street at night",
    ]
    out: list[dict] = []
    for i in range(n_samples):
        size = 224
        img = Image.new("RGB", (size, size),
                        color=(30 + (i * 37) % 220,
                               40 + (i * 53) % 200,
                               50 + (i * 67) % 180))
        caption = captions[i % len(captions)]
        prompt = f"<|image|>Describe: {caption}"
        # Processor may be None (e.g. tests without a real processor) —
        # in that case we build the sample directly as a rank-4 pixel
        # tensor + a random integer id sequence.  No vision-specific
        # kwargs (this path is only exercised by unit tests, not by
        # real multimodal Fisher probes).
        if processor is None:
            import torch as _t
            pixel_values = _t.rand(1, 3, size, size, dtype=_t.float32)
            input_ids = _t.randint(1, 100, (1, min(max_text_len, 16)),
                                   dtype=_t.long)
            out.append({
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            })
            continue
        # Prefer apply_chat_template — it inserts the correct
        # number of image-placeholder tokens to match what the vision
        # tower will emit (critical for Qwen3.5/3.6, MiniMax VL,
        # Gemma-3 VL: they use `<|vision_start|>...<|vision_end|>` or
        # similar markers that a literal `<|image|>` prompt doesn't
        # produce, leading to "Image features and image tokens do not
        # match" errors at forward time). Fall back to raw
        # `processor(text=, images=)` only if chat template isn't
        # implemented on this processor.
        enc = None
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Describe: {caption}"},
            ],
        }]
        try:
            enc = processor.apply_chat_template(
                messages, add_generation_prompt=False,
                tokenize=True, return_dict=True, return_tensors="pt")
        except Exception:
            try:
                enc = processor(text=prompt, images=img, return_tensors="pt")
            except Exception:
                continue
        if enc is None:
            continue
        sample = _samples_from_encoding(enc, max_text_len)
        if sample is not None:
            out.append(sample)
    return out


def load_multimodal_calibration(
    processor,
    dataset_name: str,
    n_samples: int,
    max_text_len: int,
) -> list[dict]:
    """Build a list of forward-kwargs dicts for multimodal Fisher
    calibration.

    `processor` is an `AutoProcessor` — usually loaded via
    `AutoProcessor.from_pretrained(model_path, trust_remote_code=True)`.

    `dataset_name`:
      - `"synthetic"` (default): built-in offline stub. No network, no
        HF datasets dependency beyond what transformers itself needs.
        Used to exercise the visual Fisher code path under unit tests
        and in environments without dataset access.
      - any other string: treated as a HuggingFace dataset id. We stream
        up to `n_samples * 4` rows and filter for rows that have an
        image (+ a caption/text field). Falls back to the synthetic
        stub on any load failure (offline, rate-limited, schema
        mismatch, etc.) so the probe always makes forward progress.

    Each returned sample is a `dict` of tensors suitable for
    `model(**sample)`. Contains every tensor the processor emitted
    (pixel_values, input_ids, image_grid_thw, attention_mask, ...)
    plus a `labels` key. Caller pops `labels` before unpacking.

    Labels default to `input_ids.clone()` (teacher-forced CE on the
    joint image+text sequence). Processors that emit `-100` sentinel
    ids for masked positions are handled by the probe's CE backward —
    not by this loader.
    """
    triples: list[dict] = []
    if dataset_name == "synthetic":
        return _synthetic_multimodal_calibration_samples(
            processor, n_samples, max_text_len)
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        iterator = iter(ds)
        for _ in range(n_samples * 4):
            try:
                row = next(iterator)
            except StopIteration:
                break
            img = row.get("image")
            caption = (row.get("caption")
                       or row.get("sentences", [{}])[0].get("raw")
                       or row.get("text")
                       or "describe this image")
            if img is None:
                continue
            prompt = f"<|image|>Describe: {caption}"
            try:
                enc = processor(text=prompt, images=img, return_tensors="pt")
            except Exception:
                continue
            sample = _samples_from_encoding(enc, max_text_len)
            if sample is None:
                continue
            triples.append(sample)
            if len(triples) >= n_samples:
                break
    except Exception as e:
        print(f"[probe/mm] dataset {dataset_name!r} unreachable ({e}); "
              f"falling back to synthetic stub", flush=True)
    if len(triples) < n_samples:
        synth = _synthetic_multimodal_calibration_samples(
            processor, n_samples - len(triples), max_text_len)
        triples.extend(synth)
    return triples[:n_samples]


_ALLOW_SUMSQ_PACKED_FISHER_ENV = "PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER"


def h_detail_blob(h_raw: torch.Tensor, global_tokens: int, name: str, *,
                  kind: str = "linear",
                  g2_per_token: torch.Tensor | None = None) -> dict:
    """Normalize a token-SUMMED Fisher diagonal accumulator to per-token
    units and wrap it in the canonical h-detail blob schema.

    ``global_tokens`` must be the GLOBAL calibration token count — the
    same denominator `finalize_fisher_stats` applies to the scalar
    ``h_trace`` — never a per-row routed-token count. v4 pins this:
    passing an unpacked expert Linear's own ``n_tokens_seen`` here left
    the detail blob (global/routed)× hotter than the scalar it must
    agree with, so `predicted_dloss` fallback rows priced expert rows on
    a different scale than the rest of the knapsack.

    Every h-detail writer (this module's `FisherAccumulator.finalize` and
    `incremental_probe`'s two writer sites) goes through this helper so
    the on-disk units are per-token everywhere and stamped with an
    explicit ``units: "per_token"`` marker (audit M9 — the incremental
    writer used to save the raw token-summed ``H``, leaving the consumer
    `HDetailIndex.h_diag_from_blob` ~n_tokens× off depending on which
    probe produced the directory). ``g2_per_token`` is already a
    per-token vector and is stored as-is.
    """
    tokens = max(int(global_tokens), 1)
    h = h_raw.detach().to("cpu", torch.float32) / tokens
    blob = {
        "h_diag": h,
        "name": name,
        "kind": kind,
        "shape": list(h.shape),
        "units": "per_token",
        # v3: unit marker; no route_prob term (M4).
        # v4: denominator is the GLOBAL calib token count for every row
        #     (v3 unpacked-expert blobs were per-ROUTED-token).
        "h_detail_version": 4,
        "norm_tokens": tokens,
    }
    if g2_per_token is not None:
        blob["g2_per_token"] = g2_per_token
    return blob


def finalize_fisher_stats(merged_stats: dict, global_tokens: int) -> None:
    """Normalize raw Fisher accumulators into ``h_trace`` (etc.), in place.

    Fisher normalization must share ONE denominator across every row: the
    global calibration token count (nsamples x seqlen). Dense trunk Linears
    accumulate exactly that many tokens in ``n_tokens_seen``, so their
    values are unchanged by dividing by the global count. Per-expert
    Linears, however, only see their ROUTED tokens; a per-row
    ``h_trace_raw / n_tokens_seen`` inflates a rarely-routed expert's
    Fisher by (global/routed) — exactly inverted importance weighting (the
    least-used experts look the most sensitive). Tokens never routed to an
    expert contribute zero gradient, so the empirical Fisher over the
    calibration set divides by the GLOBAL count for every row.

    HISTORY — this deliberately REVERSES a documented convention. Audit M4
    removed an explicit ÷route_prob on the grounds that dividing by the
    routed-token count was "the one implicit ÷token-fraction the MoE
    convention prescribes", and both backends then shipped per-routed-token
    as the single normalization (pinned by the original
    tests/test_packed_expert_per_token_fisher.py). M4's *agreement* goal
    stands — one division, both backends, route_prob as metadata only —
    but per-routed-token was the wrong denominator for the mean-Δloss
    objective the allocator optimizes: it is the same 1/p_e inflation M4
    removed, merely implicit. CLAUDE.md §3 records the same reversal.

    ``n_tokens_seen`` stays raw (routed count, metadata). Every h-detail
    blob is normalized by the SAME global count (`h_detail_blob` v4) so
    scalar and per-weight detail Fisher share one denominator.
    """
    for s in merged_stats.values():
        s["h_trace"] = s.get("h_trace_raw", 0.0) / global_tokens
        s["h_w2_sum"] = s.get("h_w2_sum_raw", 0.0) / global_tokens
        # Per-expert Fisher trace (only present on packed-3D stat entries;
        # dense Linears have no per-expert dimension). Normalize by the
        # same token count so it shares units with `h_trace`.
        per = s.get("h_trace_per_expert_raw")
        if per is not None:
            s["h_trace_per_expert"] = [float(v) / global_tokens for v in per]
        # Per-GROUP Fisher trace (grouped-BMM operands like DSv4's wo_a).
        # Same rule, same denominator: the group axis is a contraction
        # structure ON one logical tensor, and every group sees every
        # calibration token (attention output projection — nothing is
        # routed), so the global count is exact for it, not just
        # consistent.
        per_group = s.get("h_trace_per_group_raw")
        if per_group is not None:
            s["h_trace_per_group"] = [
                float(v) / global_tokens for v in per_group]
        s["h_trace_norm_tokens"] = global_tokens


# ---------------------------------------------------------------------------
# Grouped-BMM Linear Fisher (`wo_a` shape)
#
# A declared grouped module (spec `probe.grouped_module_class_names`) is an
# nn.Linear subclass whose forward consumes its `[G*R, D]` weight plane as
# `W[g, r, d]`: `y[..., g, r] = sum_d x[..., g, d] * W[g, r, d]`
# (view + bmm in `DeepseekV4GroupedLinear.forward`). The dense accumulator's
# flatten-to-2D cannot represent this consumption: its chunk_h comes out
# [R, D] against an [G*R, D] weight (the `chunk_h * w.pow(2)` broadcast that
# motivated the original probe skip), and pooling groups into the token axis
# destroys the per-(g,r) channel marginals. The functions below are the ONE
# grouped accumulation mechanism; both backends fold its outputs into the
# same stats keys the dense rows use.
#
# TP note (walker campaign addendum, 2026-08-22): identity, dispositions,
# and every field below are properties of the WHOLE logical tensor. A future
# Tensor-Parallel shard of `wo_a` would cut the stored plane's ROW axis
# (axis 0 = G*R; the profile's `base_model_tp_plan` declares `rowwise`),
# so a shard boundary must not straddle a group (a cut inside g's R rows
# would make any future group-aware format illegal at that TP degree).
# Nothing here bakes rank/shard identity into keys or sizes; byte totals
# are TOTAL-of-logical, per-device splits are a serving-runtime concern.


def grouped_linear_groups(mod, profile=None) -> int | None:
    """Group count `G` for a profile-declared grouped-BMM Linear, else None.

    Dispatch is EXPLICIT: only classes the profile declares under
    ``probe.grouped_module_class_names`` are grouped — never a shape
    heuristic. A declared class whose instance lacks a usable ``n_groups``
    fails fast: silently falling through to the dense accumulator would
    produce the inflated, mis-shaped numbers the old probe skip existed to
    prevent.
    """
    import torch.nn as _nn
    if not isinstance(mod, _nn.Linear) or profile is None:
        return None
    try:
        declared = tuple(profile.probe_grouped_module_class_names())
    except AttributeError:
        declared = ()
    if type(mod).__name__ not in declared:
        return None
    n_groups = getattr(mod, "n_groups", None)
    if n_groups is None:
        raise ValueError(
            f"{type(mod).__name__} is declared under "
            "probe.grouped_module_class_names but carries no n_groups "
            "attribute; refusing to fall through to the dense Fisher "
            "accumulator (its numbers would be wrong-shaped)")
    g = int(n_groups)
    out_features = int(mod.out_features)
    if g <= 0 or out_features % g != 0:
        raise ValueError(
            f"{type(mod).__name__}: n_groups={g} does not divide "
            f"out_features={out_features}")
    return g


def grouped_linear_stats_entry(mod, num_groups: int,
                               w_max_abs: float | None = None,
                               w_norm_sq: float | None = None) -> dict:
    """Stats-schema entry for a grouped operand, mirroring the dense row.

    Shape convention follows the serialized truth, NOT a slice fiction:
    `out_features`=G*R and `in_features`=D name the flat `[G*R, D]` plane
    the exporter quantizes, and the 1-D marginals index that same plane
    (`fisher_row[g*R + r]`). `num_groups` is the distinguishing field —
    deliberately NOT `num_experts`, which downstream code
    (`_shape_from_stats`, `_stats_indicates_packed_expert`) reads as a
    packed expert stack. Per-group structure rides on
    `h_trace_per_group_raw`; the unit stays the whole logical tensor."""
    w = mod.weight
    return {
        "h_trace_raw": 0.0,
        "h_w2_sum_raw": 0.0,
        "w_max_abs": w_max_abs,
        "w_norm_sq": w_norm_sq,
        "n_params": int(w.numel()),
        "in_features": int(mod.in_features),
        "out_features": int(mod.out_features),
        "num_groups": int(num_groups),
        "n_tokens_seen": 0,
        # Grouped attention output projections are not routed: route
        # metadata exists for schema parity with dense rows, always None.
        "route_prob": None,
        "router_path": None,
        "expert_id": None,
    }


def grouped_linear_fisher_chunk(
    x: torch.Tensor,
    gy: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One backward hook's worth of grouped empirical-Fisher pieces.

    Input contract (validated): `x` is the module input `[..., G, D]`,
    `gy = dL/dy` is `[..., G, R]`, `weight` is the `[G*R, D]` plane. The
    per-token gradient of the weight is `grad_t W[g,r,d] =
    gy[t,g,r] * x[t,g,d]`, so the per-token-summed empirical Fisher —
    the SAME estimator as every dense row (`Σ_t ‖∇_t‖²_F`, audit M3;
    never `‖Σ_t ∇_t‖²`) factorizes exactly:

        h_trace = Σ_t Σ_g ‖gy[t,g,:]‖² · ‖x[t,g,:]‖²
                = Σ_g Σ_{r,d} chunk_h[g,r,d],
        chunk_h[g,r,d] = Σ_t gy[t,g,r]² · x[t,g,d]²

    The true elementwise Fisher is block-diagonal in `g`
    (`H[(g,r),(g',d)] = 0` for `g' ≠ g`), so the per-group bmm is EXACT,
    not an approximation. Everything downstream reduces the ONE fp32
    `chunk_h`, which makes `sum(fisher_row) == sum(fisher_col) ==
    h_trace` hold BY CONSTRUCTION — the wiring discipline the dense
    sites document at their own accumulation points.

    Returns device-resident fp32 tensors:
      h_trace         0-dim, raw (pre-`finalize_fisher_stats`)
      h_w2            0-dim ⟨chunk_h, W²⟩ proxy (None when weight absent/meta)
      fisher_row     [G*R] flat-plane output-channel marginal
      fisher_col     [D]    input-channel marginal (pools groups: column d
                      of the plane serves every group's contraction)
      g_sq_sum       [G*R] flat gy² marginal
      act_sq_sum     [D]    imatrix second moment (pools groups)
      act_absmax     [D]    max |x| bound (pools groups; merge rule MAX)
      trace_per_group [G]  decomposition of h_trace across the group axis
      chunk_flat     [G*R, D] chunk_h reshaped onto the stored plane, for
                      `h_full` collection

    Normalization is NOT applied here: rows keep raw token-SUMMED values
    and `finalize_fisher_stats` divides every row by the GLOBAL calibration
    token count. Callers bump `n_tokens_seen` by TOKENS (= x rows before
    the group dim), never T*G — group slices of one token are one token's
    evidence, and the shared global denominator is what keeps grouped rows
    commensurable with dense ones.
    """
    g = int(num_groups)
    if x.dim() < 2 or gy.dim() < 2 or x.shape[-2] != g or gy.shape[-2] != g:
        raise ValueError(
            f"grouped Fisher expects x [..., {g}, D] and gy [..., {g}, R]; "
            f"got x {tuple(x.shape)}, gy {tuple(gy.shape)}")
    d_in = int(x.shape[-1])
    r_out = int(gy.shape[-1])
    if weight is not None and tuple(weight.shape) != (g * r_out, d_in):
        raise ValueError(
            f"grouped weight plane expected {(g * r_out, d_in)}, "
            f"got {tuple(weight.shape)}")

    xf = x.detach().reshape(-1, g, d_in)     # [T, G, D]
    gyf = gy.detach().reshape(-1, g, r_out)  # [T, G, R]
    tokens = int(xf.size(0))
    gy_sq = gyf.pow(2)                       # input dtype, like the dense path
    x_sq = xf.pow(2)
    # One batched matmul over the group axis: [G,R,T] @ [G,T,D] -> [G,R,D].
    # Same cost class as the dense chunk_h matmul; the [T*G, ...] flattening
    # the dense path would apply instead pools groups into the token axis
    # and loses exactly the structure this function exists to keep.
    chunk_h = torch.bmm(
        gy_sq.permute(1, 2, 0), x_sq.permute(1, 0, 2)).float()
    if tokens == 0:
        # An empty reduction would make amax/amin raise (same guard as
        # _marginal_chunk); every sum here is already zero.
        act_absmax = torch.zeros(d_in, dtype=torch.float32, device=xf.device)
    else:
        hi = torch.maximum(xf.amax(dim=0).abs(), xf.amin(dim=0).abs())
        act_absmax = hi.amax(dim=0).to(torch.float32)

    out = {
        "h_trace": chunk_h.sum(),
        "fisher_row": chunk_h.sum(dim=-1).reshape(-1),
        "fisher_col": chunk_h.sum(dim=(0, 1)),
        "g_sq_sum": gy_sq.sum(dim=0).reshape(-1).to(torch.float32),
        "act_sq_sum": x_sq.sum(dim=(0, 1)).to(torch.float32),
        "act_absmax": act_absmax,
        "trace_per_group": chunk_h.sum(dim=(1, 2)),
        "chunk_flat": chunk_h.reshape(g * r_out, d_in),
    }
    if weight is not None and not weight.is_meta:
        w_view = weight.detach().view(g, r_out, d_in)
        out["h_w2"] = (chunk_h * w_view.float().pow(2)).sum()
    else:
        out["h_w2"] = None
    return out


def _scalar_acc_add(acc: dict, name: str, value: torch.Tensor) -> None:
    """Add a 0-dim fp32 tensor into `acc[name]`, keeping the running total
    device-resident so the backward hot path never forces a CUDA→CPU sync.
    Consumers flush with `float(acc[name])` (works for float and tensor)."""
    cur = acc.get(name)
    if cur is None:
        acc[name] = value.detach().to(torch.float32).clone()
    elif torch.is_tensor(cur):
        cur.add_(value.detach().to(cur.device, torch.float32))
    else:
        # Legacy float entry (e.g. written by the env-gated sum-sq
        # fallback) — keep the float representation.
        acc[name] = float(cur) + float(value)


def _accumulate_packed_per_token_fisher(
    name: str,
    expert_idx: int,
    num_experts: int,
    x: torch.Tensor,
    gy: torch.Tensor,
    scalar_acc: dict,
    channel_acc: dict | None,
    full_acc: dict | None,
    marginal_acc: dict | None = None,
) -> None:
    """Per-token-summed empirical Fisher moments for one expert's Linear
    application inside a packed [E, M, N] MoE parameter.

    For expert e with routed tokens t (x_t = input row, gy_t = grad of the
    expert's output row), the per-token outer-product identities give:

      scalar  : Σ_t ‖gy_t‖²·‖x_t‖²          (= Σ_t ‖∇_t W_e‖²_F)
      channel : [E, M] row e += Σ_t gy_{t,m}²·‖x_t‖²
      full    : [E, M, N] slice e += (gy²)ᵀ @ (x²)
                                             (= Σ_t gy_{t,m}²·x_{t,n}²)

    This is the same per-token estimator the nn.Linear paths use
    (`FisherAccumulator._make_bwd`, incremental_probe's resident/deferred
    hooks) — NOT the sum-then-square ‖Σ_t ∇_t‖² that the packed path used
    to compute from the token-summed weight gradient (audit M3: inflated
    by the cross-token gradient covariance, 5-50× on correlated
    sequences, and non-convergent as calibration grows).

    Mixed precision matches the nn.Linear paths: square in the incoming
    dtype (bf16 squaring is safe — bf16 shares fp32's exponent range),
    reduce/accumulate in fp32. Scalar and channel accumulators stay
    device-resident (flushed once by the consumers); the full [E, M, N]
    accumulator stays on CPU, receiving one [M, N] fp32 slice per routed
    expert per backward, so GPU memory stays bounded.
    """
    gy2 = gy.detach().reshape(-1, gy.size(-1))   # (T_e, M)
    x2 = x.detach().reshape(-1, x.size(-1))      # (T_e, N)
    gy_sq = gy2.pow(2)                           # incoming dtype
    x_sq = x2.pow(2)
    x_norm = x_sq.sum(dim=1, dtype=torch.float32)          # (T_e,) fp32
    if channel_acc is not None:
        # Σ_t gy_{t,m}² · ‖x_t‖²  → (M,) fp32
        per_ch = gy_sq.float().t().mv(x_norm)
        ch = channel_acc.get(name)
        if ch is None:
            ch = torch.zeros(num_experts, per_ch.numel(),
                             dtype=torch.float32, device=per_ch.device)
            channel_acc[name] = ch
        ch[expert_idx].add_(per_ch.to(ch.device))
        trace = per_ch.sum()
    else:
        gy_norm = gy_sq.sum(dim=1, dtype=torch.float32)    # (T_e,)
        trace = (gy_norm * x_norm).sum()
    _scalar_acc_add(scalar_acc, name, trace)
    if full_acc is not None:
        cur = full_acc.get(name)
        if cur is None:
            cur = torch.zeros(num_experts, gy2.size(1), x2.size(1),
                              dtype=torch.float32, device="cpu")
            full_acc[name] = cur
        chunk_h = (gy_sq.t() @ x_sq).float()               # (M, N) fp32
        cur[expert_idx].add_(chunk_h.to("cpu", torch.float32))
        del chunk_h

    if marginal_acc is not None:
        # AQUA A-side marginals for THIS expert slice. Both are pure
        # reductions of tensors the Fisher path already materialized, so
        # they add no matmul and no extra device→host sync (the flush is
        # batched by the caller, exactly like `h_trace_per_expert_raw`).
        #
        #   g_sq_sum [E, M] : Σ_{t routed to e} gy_{t,m}²   -- RAW sum.
        #   act_sq_sum [E, N]: Σ_{t routed to e} x_{t,n}²   -- RAW sum.
        #   act_absmax [E, N]: max_t |x_{t,n}|              -- a BOUND.
        #   tokens [E]      : |{t routed to e}|             -- see below.
        #
        # `tokens` is the piece with no dense analogue and it is not
        # optional. The two raw sums feed two DIFFERENT normalizations
        # downstream and only one of them is global:
        #
        #   * g_sq_sum is divided by the GLOBAL token count, because that
        #     is what carries an expert's share of the mean-Δloss
        #     objective (a rarely-routed expert SHOULD price low). This is
        #     the PR #14 convention `finalize_fisher_stats` already
        #     enforces for h_trace.
        #   * act_sq_sum is divided by this expert's ROUTED count, because
        #     it is fitting the per-token noise magnitude of the rows that
        #     actually flow through the expert. Dividing it globally too
        #     would discount a rare expert TWICE -- once correctly in g,
        #     once wrongly in the variance -- which is PR #14's inverted
        #     importance weighting in mirror image.
        #
        # The packed stat entry's `n_tokens_seen` is the layer's global
        # count, so the routed denominator has to be recorded here.
        slot = marginal_acc.get(name)
        if slot is None:
            slot = {
                "expert_g_sq_sum": torch.zeros(
                    num_experts, gy2.size(1), dtype=torch.float32,
                    device=gy2.device),
                "expert_act_sq_sum": torch.zeros(
                    num_experts, x2.size(1), dtype=torch.float32,
                    device=x2.device),
                "expert_act_absmax": torch.zeros(
                    num_experts, x2.size(1), dtype=torch.float32,
                    device=x2.device),
                "expert_tokens": torch.zeros(
                    num_experts, dtype=torch.float64, device=x2.device),
            }
            marginal_acc[name] = slot
        slot["expert_g_sq_sum"][expert_idx].add_(
            gy_sq.sum(dim=0, dtype=torch.float32))
        slot["expert_act_sq_sum"][expert_idx].add_(
            x_sq.sum(dim=0, dtype=torch.float32))
        torch.maximum(
            slot["expert_act_absmax"][expert_idx],
            x2.abs().amax(dim=0).to(torch.float32),
            out=slot["expert_act_absmax"][expert_idx])
        slot["expert_tokens"][expert_idx].add_(float(x2.size(0)))


def _packed_expert_slice_index(weight: torch.Tensor,
                               param: torch.Tensor) -> int | None:
    """Expert index e such that `weight` is `param[e]` — a dim-0 select
    view of the packed [E, M, N] parameter. Returns None when `weight`
    is not such a slice (the caller then falls back to the plain linear
    and `_PackedWeightGradFallback` catches the weight gradient)."""
    if weight.dim() != 2 or param.dim() != 3:
        return None
    if weight.device.type == "meta" or param.device.type == "meta":
        return None
    if tuple(weight.shape) != tuple(param.shape[1:]):
        return None
    if tuple(weight.stride()) != tuple(param.stride()[1:]):
        return None
    stride0 = param.stride(0)
    if stride0 <= 0:
        return None
    try:
        if (weight.untyped_storage().data_ptr()
                != param.untyped_storage().data_ptr()):
            return None
    except Exception:
        return None
    off = weight.storage_offset() - param.storage_offset()
    if off < 0 or off % stride0 != 0:
        return None
    e = off // stride0
    if e >= param.shape[0]:
        return None
    return int(e)


class _PackedLinearPerTokenFisher(torch.autograd.Function):
    """`F.linear` on one expert's [M, N] slice of a packed [E, M, N] MoE
    parameter, with per-token empirical-Fisher capture in backward.

    Forward is exactly `orig_linear(x, weight, bias)`. Backward returns
    the exact input/bias gradients but None for the weight slice: the
    packed leaf's `.grad` stays None (no [E, M, N] gradient tensor is
    ever materialized) and the per-token-summed Fisher moments are
    accumulated instead via `_accumulate_packed_per_token_fisher`.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, orig_linear, name, expert_idx,
                num_experts, scalar_acc, channel_acc, full_acc,
                marginal_acc=None):
        ctx.name = name
        ctx.expert_idx = expert_idx
        ctx.num_experts = num_experts
        ctx.scalar_acc = scalar_acc
        ctx.channel_acc = channel_acc
        ctx.full_acc = full_acc
        ctx.marginal_acc = marginal_acc
        ctx.has_bias = bias is not None
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weight)
        return orig_linear(x, weight, bias)

    @staticmethod
    def backward(ctx, gy):
        if gy is None:
            return (None,) * 12
        x, w = ctx.saved_tensors
        _accumulate_packed_per_token_fisher(
            ctx.name, ctx.expert_idx, ctx.num_experts, x, gy,
            ctx.scalar_acc, ctx.channel_acc, ctx.full_acc,
            ctx.marginal_acc)
        grad_x = gy.matmul(w) if ctx.needs_input_grad[0] else None
        grad_b = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_b = gy.reshape(-1, gy.size(-1)).sum(dim=0)
        return (grad_x, None, grad_b) + (None,) * 9


class _PackedWeightGradFallback(torch.autograd.Function):
    """Identity wrapper on the packed [E, M, N] parameter (formerly
    `_GradNormCapture`). Returns None for the weight gradient so the
    leaf's `.grad` is never populated (40 MoE layers × 2 packed params ×
    ~5 GB of grads = 400 GB if retained).

    With the per-token `F.linear` interception in `patched_forward`
    (see `install_packed_expert_hooks`) covering every use of the packed
    weight, no gradient ever reaches this node — the interception
    returns None for the weight slice, so autograd prunes the path here.

    If this backward DOES run, the experts module consumed the packed
    weight through an op the interception does not cover (bmm /
    grouped-mm / einsum / a non-view slice). The only quantity available
    at this boundary is the token-SUMMED weight gradient, whose square
    is the ‖Σ_t ∇_t‖² sum-then-square estimator — the exact 5-50×
    cross-token-covariance inflation documented at the nn.Linear per-token
    path (audit M3). Rather than silently feed the allocator inflated
    expert rows, fail fast; PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1 opts
    into the biased estimator (mirrors the KV-shared Fisher guard,
    MINOR-M33).
    """

    @staticmethod
    def forward(ctx, weight, name, scalar_accumulator, channel_accumulator,
                full_accumulator):
        ctx.name = name
        ctx.scalar_acc = scalar_accumulator
        ctx.channel_acc = channel_accumulator
        ctx.full_acc = full_accumulator
        ctx.set_materialize_grads(False)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None
        if os.environ.get(_ALLOW_SUMSQ_PACKED_FISHER_ENV, "0") == "0":
            raise RuntimeError(
                f"[probe] packed-expert param {ctx.name!r} received a "
                "token-summed weight gradient: its expert compute was not "
                "intercepted at an F.linear(x, packed[e]) boundary, so a "
                "faithful per-token Fisher cannot be captured for it. "
                "Squaring this gradient would be the sum-then-square "
                "estimator (audit M3: 5-50× cross-token-covariance "
                "inflation, layer-non-uniform). Set "
                f"{_ALLOW_SUMSQ_PACKED_FISHER_ENV}=1 to accept the biased "
                "estimator for this module, or extend the interception in "
                "install_packed_expert_hooks to cover its compute pattern."
            )
        g = grad_output.detach()
        # --- env-gated legacy sum-then-square path (biased; see above) ---
        # Scalar Frobenius-norm squared — streamed to avoid materializing
        # the full squared tensor for very large packed params.
        flat = g.reshape(-1)
        chunk = 1_000_000
        total = 0.0
        for i in range(0, flat.numel(), chunk):
            total += float(flat[i:i + chunk].float().pow(2).sum().item())
        cur = ctx.scalar_acc.get(ctx.name, 0.0)
        ctx.scalar_acc[ctx.name] = (
            float(cur) if not torch.is_tensor(cur) else float(cur)) + total
        # Per-expert per-output-channel diagonal: reduce along the
        # in-feature axis (the last dim). For a [E, M, N] packed param,
        # result is [E, M]. Accumulated across backward samples.
        if ctx.channel_acc is not None:
            if g.dim() == 3:
                per_ch = g.float().pow(2).sum(dim=-1)      # [E, M]
            elif g.dim() == 2:
                per_ch = g.float().pow(2).sum(dim=-1, keepdim=False)
            else:
                per_ch = None
            if per_ch is not None:
                cur = ctx.channel_acc.get(ctx.name)
                if cur is None:
                    ctx.channel_acc[ctx.name] = per_ch.cpu()
                else:
                    cur.add_(per_ch.to(cur.device))
        # Full per-weight Fisher: chunk along the expert dim so CPU peak
        # stays bounded even on 128-256 expert packed params.
        if ctx.full_acc is not None and g.dim() >= 2:
            name = ctx.name
            cur = ctx.full_acc.get(name)
            if cur is None:
                cur = torch.zeros(*g.shape, dtype=torch.float32,
                                  device="cpu")
                ctx.full_acc[name] = cur
            if g.dim() == 3:
                E = g.size(0)
                chunk_e = max(1, min(16, E))
                for e in range(0, E, chunk_e):
                    g_c = g[e:e + chunk_e].cpu()
                    cur[e:e + chunk_e].add_(g_c.float().pow(2))
                    del g_c
            else:
                g_c = g.cpu()
                cur.add_(g_c.float().pow(2))
                del g_c
        return None, None, None, None, None


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
    """A module qualifies as a packed-experts container iff (a) its
    class name contains "Experts" (case-insensitive), and (b) it owns
    at least one profile-declared 3D packed expert parameter.

    The class-name check excludes other modules that happen to own 3D
    parameters — most importantly Conv1d in linear-attention paths,
    whose `weight` is shape `[out, in, kernel]`. The param-name check
    is a second safety net against unusual modules with unrelated 3D
    state.
    """
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
    """Return the attribute names of all 3D packed parameters on
    `module`, restricted to the known MoE expert names. Order is stable
    across Python runs."""
    names_allowed = _packed_expert_param_name_set(profile)
    names = []
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in names_allowed):
            names.append(n)
    return sorted(names)


def _packed_expert_projection_candidate_names(profile=None) -> tuple[str, ...]:
    names = set(_packed_expert_param_name_set(profile))
    if profile is not None:
        for name in list(names):
            try:
                names.update(profile.packed_expert_projection_names(name))
            except Exception:
                pass
    return tuple(sorted(names))


_PRISMAQUANT_PATCH_SENTINEL = "_prismaquant_packed_expert_patch"
_PRISMAQUANT_CHANNEL_SENTINEL = "_prismaquant_packed_expert_channel_patch"
_PRISMAQUANT_FULL_SENTINEL = "_prismaquant_packed_expert_full_patch"
_PRISMAQUANT_MARGINAL_SENTINEL = "_prismaquant_packed_expert_marginal_patch"


def install_packed_expert_hooks(
    model: nn.Module,
    accumulator: dict,
    channel_accumulator: dict | None = None,
    full_accumulator: dict | None = None,
    marginal_accumulator: dict | None = None,
    profile=None,
) -> dict[str, dict]:
    """Patch every packed-experts module's forward so each expert-slice
    `F.linear(x, packed[e])` call inside it routes through
    `_PackedLinearPerTokenFisher`, which captures the per-token-summed
    empirical Fisher (Σ_t ‖∇_t‖², the same estimator as the nn.Linear
    paths — NOT the sum-then-square ‖Σ_t ∇_t‖² of the token-summed
    weight gradient; audit M3). The 3D parameters are additionally
    wrapped by `_PackedWeightGradFallback`, which fail-fasts if any
    packed-weight use escapes the interception (env-gated sum-sq
    fallback via PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1).

    `accumulator` collects the scalar Fisher trace (matches the
    nn.Linear `h_trace_raw`; values may be device-resident 0-dim fp32
    tensors — flush with `float(...)`). `channel_accumulator` collects
    the per-expert per-output-channel diagonal as [E, M] fp32 tensors
    (device-resident on the compute device; for the full per-weight
    Fisher cost model). The channel accumulator is optional for backward
    compatibility; when None, packed experts contribute only their
    scalar trace and the allocator falls back to the scalar proxy for
    those entries.

    Returns a metadata dict keyed by `<module_qname>.<param_name>` with
    the same shape/role information stored for nn.Linear modules in
    `FisherAccumulator`. The probe inserts these into its main `stats`
    dict so the allocator can treat them uniformly.

    Idempotent across calls. If a module has already been patched (by
    a prior call within the same Python process), we re-bind both the
    scalar and channel accumulator references to the new dicts rather
    than wrapping the patch again. This is essential for the incremental
    probe path, which constructs a fresh FisherAccumulator per shard
    against a single loaded model.

    Activation snapshotting for measure_quant_cost is handled by
    `FisherAccumulator` directly (forward hook on the experts module).
    """
    meta: dict[str, dict] = {}
    for qname, module in model.named_modules():
        if not _is_packed_experts_module(module, profile):
            continue
        param_names = _packed_experts_param_names(module, profile)
        if not param_names:
            continue

        # transformers 5.x dispatches the packed-experts forward through an
        # MoE interface (`config._experts_implementation`, e.g. batched_mm /
        # grouped_mm) that consumes the packed weight via an advanced-index
        # gather + a batched matmul — there is no per-expert `F.linear(x,
        # packed[e])` boundary, so the per-token Fisher interception below
        # never sees the weight and the fail-fast guard in
        # `_PackedWeightGradFallback` trips (audit M3). Force the reference
        # "eager" per-expert implementation for the probe forward: it is the
        # same math, just the interceptable kernel, so the captured Fisher is
        # faithful. No-op on transformers builds without the MoE interface
        # (older per-expert layouts) — the attribute is simply absent.
        _cfg = getattr(module, "config", None)
        if _cfg is not None and hasattr(_cfg, "_experts_implementation"):
            _cfg._experts_implementation = "eager"

        # Idempotent re-bind path. The sentinel holds a reference to the
        # mutable accumulator dict that patched_forward writes to. We
        # rebind it (clear contents and adopt the new dict's identity by
        # swapping references) — but the simpler primitive is to update
        # the closure's *target dict identity* via attribute, since
        # patched_forward's closure already binds the original dict by
        # reference. Easiest: store the live accumulator on the module
        # and have patched_forward read it indirectly each call.
        if hasattr(module, _PRISMAQUANT_PATCH_SENTINEL):
            # Update the live accumulator binding for this module's patch.
            setattr(module, _PRISMAQUANT_PATCH_SENTINEL, accumulator)
            setattr(module, _PRISMAQUANT_CHANNEL_SENTINEL, channel_accumulator)
            setattr(module, _PRISMAQUANT_FULL_SENTINEL, full_accumulator)
            setattr(module, _PRISMAQUANT_MARGINAL_SENTINEL, marginal_accumulator)
            # Still report metadata so callers can refresh their stats dict.
            for pn in param_names:
                p_existing = module._parameters.get(pn)
                if p_existing is None:
                    continue
                shape = tuple(p_existing.shape)
                full_name = f"{qname}.{pn}" if qname else pn
                if p_existing.is_meta:
                    w_max_abs = None
                    w_norm_sq = None
                else:
                    w_max_abs = float(p_existing.detach().abs().max().item())
                    w_norm_sq = float(p_existing.detach().pow(2).sum().item())
                meta[full_name] = {
                    "h_trace_raw": 0.0,
                    "h_w2_sum_raw": 0.0,
                    "w_max_abs": w_max_abs,
                    "w_norm_sq": w_norm_sq,
                    "n_params": int(p_existing.numel()),
                    "in_features": int(shape[2]),
                    "out_features": int(shape[1]),
                    "num_experts": int(shape[0]),
                    "n_tokens_seen": 0,
                    "route_prob": None,
                    "router_path": None,
                    "expert_id": None,
                    "_packed_experts_module": qname,
                    "_packed_param": pn,
                }
            continue

        # Enable grad on packed params so autograd computes their gradient
        # through our identity wrapper.
        for pn in param_names:
            getattr(module, pn).requires_grad_(True)

        for pn in param_names:
            p: nn.Parameter = getattr(module, pn)
            full_name = f"{qname}.{pn}" if qname else pn
            shape = tuple(p.shape)
            # Convention: shape[0] = num_experts; the per-expert matrix is
            # the trailing two dims. Use (out_features, in_features) =
            # (shape[1], shape[2]) to match nn.Linear's convention; the
            # allocator's predicted_dloss only needs n_params correct.
            num_experts = int(shape[0])
            out_features = int(shape[1])
            in_features = int(shape[2])
            n_params = int(p.numel())
            # When the model is loaded with accelerate's disk offload
            # (`device_map="auto"` on hardware too small to fit the full
            # model), packed params start out on the meta device and are
            # materialized lazily at forward time. Defer the max-abs /
            # norm-sq scalar statistics until they can be measured.
            if p.is_meta:
                w_max_abs = None
                w_norm_sq = None
            else:
                w_max_abs = float(p.detach().abs().max().item())
                w_norm_sq = float(p.detach().pow(2).sum().item())
            meta[full_name] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,  # not measured for packed; kept for schema
                "w_max_abs": w_max_abs,
                "w_norm_sq": w_norm_sq,
                "n_params": n_params,
                "in_features": in_features,
                "out_features": out_features,
                "num_experts": num_experts,
                "n_tokens_seen": 0,
                "route_prob": None,  # rolled into per-expert sensitivity by sum
                "router_path": None,
                "expert_id": None,
                "_packed_experts_module": qname,
                "_packed_param": pn,
            }

        # Patch forward to (a) wrap each packed param with the
        # _PackedWeightGradFallback sentinel and (b) intercept F.linear
        # calls whose weight is a dim-0 slice of a packed param, routing
        # them through _PackedLinearPerTokenFisher for per-token Fisher
        # capture. The original forward uses self.<pn>; we shadow those
        # attributes with the wrapped tensors for the duration of the
        # call. nn.Module __getattribute__ checks _parameters before
        # __dict__, so we have to temporarily move the param out of
        # _parameters and shadow via __dict__ to make the wrapped tensor
        # visible to the original forward.
        original_forward = module.forward
        ns = list(param_names)
        full_names = [f"{qname}.{pn}" if qname else pn for pn in ns]
        mod_ref = module

        # Store the live accumulators as attributes so subsequent calls
        # to install_packed_expert_hooks can re-bind them (per-shard) by
        # just updating these attributes. patched_forward reads them
        # indirectly each invocation via getattr.
        setattr(mod_ref, _PRISMAQUANT_PATCH_SENTINEL, accumulator)
        setattr(mod_ref, _PRISMAQUANT_CHANNEL_SENTINEL, channel_accumulator)
        setattr(mod_ref, _PRISMAQUANT_FULL_SENTINEL, full_accumulator)
        setattr(mod_ref, _PRISMAQUANT_MARGINAL_SENTINEL, marginal_accumulator)

        def patched_forward(*args, _ns=ns, _full=full_names, _orig=original_forward,
                            _mod=mod_ref, **kwargs):
            acc = getattr(_mod, _PRISMAQUANT_PATCH_SENTINEL, None)
            ch_acc = getattr(_mod, _PRISMAQUANT_CHANNEL_SENTINEL, None)
            fu_acc = getattr(_mod, _PRISMAQUANT_FULL_SENTINEL, None)
            mg_acc = getattr(_mod, _PRISMAQUANT_MARGINAL_SENTINEL, None)
            if acc is None:
                # Should not happen, but degrade gracefully.
                return _orig(*args, **kwargs)
            saved_params = {}
            wrapped = {}
            # Identity map for the F.linear interception: a dim-0 slice
            # `wrapped[e]` is an autograd view whose ._base is the
            # original Parameter (the wrapped tensor itself is a view of
            # the param); key on both identities to be robust to either
            # aliasing convention.
            slice_targets: dict[int, tuple[str, torch.Tensor]] = {}
            for pn, fn in zip(_ns, _full):
                saved_params[pn] = _mod._parameters.pop(pn)
                wrapped[pn] = _PackedWeightGradFallback.apply(
                    saved_params[pn], fn, acc, ch_acc, fu_acc)
                _mod.__dict__[pn] = wrapped[pn]
                slice_targets[id(saved_params[pn])] = (fn, saved_params[pn])
                slice_targets[id(wrapped[pn])] = (fn, saved_params[pn])
            orig_linear = F.linear

            def _intercepting_linear(input, weight, bias=None):
                base = weight._base if weight._is_view() else weight
                hit = slice_targets.get(id(base))
                if hit is not None:
                    fn, param = hit
                    e = _packed_expert_slice_index(weight, param)
                    if e is not None:
                        return _PackedLinearPerTokenFisher.apply(
                            input, weight, bias, orig_linear, fn, e,
                            int(param.shape[0]), acc, ch_acc, fu_acc,
                            mg_acc)
                return orig_linear(input, weight, bias)

            F.linear = _intercepting_linear
            try:
                return _orig(*args, **kwargs)
            finally:
                F.linear = orig_linear
                for pn in _ns:
                    _mod.__dict__.pop(pn, None)
                    _mod._parameters[pn] = saved_params[pn]

        module.forward = patched_forward

    return meta


def resolve_execution_device(model: nn.Module, requested_device: str) -> torch.device:
    """Choose the device used for input ids / embeddings during probing.

    When `device_map="auto"` is used for model load, the model can be sharded
    across CPU and GPU. In that case we want to feed tokens to the device that
    owns the input embedding weights rather than assuming a single global
    `cuda`/`cpu` target.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device(requested_device)


# ---------------------------------------------------------------------------
# Model-agnostic MoE discovery
# ---------------------------------------------------------------------------
def discover_moe_structure(
    model: nn.Module,
    profile=None,
) -> dict[str, tuple[str, str]]:
    """Return {expert_linear_qname: (router_qname, expert_id_str)}.

    Walk the module tree.  For any module that has a child attribute named
    `experts` or `block_sparse_moe_experts` that is a ModuleList, find a
    sibling Linear in the same parent whose out_features equals len(experts).
    That Linear is the router.
    """
    if profile is None:
        try:
            from .model_profiles import profile_from_model
            profile = profile_from_model(model)
        except Exception:
            profile = None
    projection_candidates = _packed_expert_projection_candidate_names(profile)

    def _router_matches_num_experts(child: nn.Module, num_experts: int) -> bool:
        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return True
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0]) == num_experts
        return False

    expert_info: dict[str, tuple[str, str]] = {}
    for parent_qname, parent in model.named_modules():
        candidates = []
        for attr in ("experts", "block_sparse_moe_experts",
                     "moe_experts", "expert_layer"):
            experts_container = getattr(parent, attr, None)
            if experts_container is None or not isinstance(experts_container, nn.Module):
                continue
            # Two possible layouts:
            #   A) experts_container IS the list (nn.ModuleList / nn.Sequential /
            #      AutoRound's SequentialQwen3_5MoeExperts which subclasses ModuleList)
            #   B) experts_container is a plain nn.Module with numbered children
            #      (e.g. Qwen3_5MoeExperts after in-place unfuse: children are
            #      named "0", "1", ..., each holding per-expert Linears).
            #
            # Both layouts are detected by looking at child names that are
            # consecutive integer strings starting from 0.
            child_dict = dict(experts_container.named_children())
            numeric_keys = sorted(
                [k for k in child_dict if k.isdigit()],
                key=int,
            )
            if numeric_keys:
                # Require the numeric children to be 0..N-1 (no gaps)
                if [int(k) for k in numeric_keys] != list(range(len(numeric_keys))):
                    continue
                if not all(isinstance(child_dict[k], nn.Module) for k in numeric_keys):
                    continue
                candidates.append((attr, experts_container, "nested", numeric_keys))
                continue

            # Linear-loop layout after MoE unfuse: experts container itself
            # remains a module, but its packed projections become ModuleLists:
            #   experts.gate_up_proj.<expert_idx>
            #   experts.down_proj.<expert_idx>
            projection_lists = {}
            for proj_name in projection_candidates:
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                proj_numeric = sorted([k for k in proj_children if k.isdigit()], key=int)
                if not proj_numeric:
                    continue
                if [int(k) for k in proj_numeric] != list(range(len(proj_numeric))):
                    continue
                if not all(isinstance(proj_children[k], nn.Module) for k in proj_numeric):
                    continue
                projection_lists[proj_name] = proj_numeric
            if projection_lists:
                # Require a consistent expert count across projections.
                expert_lists = list(projection_lists.values())
                if all(v == expert_lists[0] for v in expert_lists[1:]):
                    candidates.append((attr, experts_container, "linear_loop", expert_lists[0]))
        if not candidates:
            continue
        attr_name, experts_container, layout, numeric_keys = candidates[0]
        num_experts = len(numeric_keys)

        # Find sibling Linear (or any module whose output feature dim
        # equals num_experts) that acts as the router.
        router_qname = None
        for child_name, child in parent.named_children():
            if child is experts_container:
                continue
            if _router_matches_num_experts(child, num_experts):
                router_qname = (f"{parent_qname}.{child_name}"
                                if parent_qname else child_name)
                break
        if router_qname is None:
            continue

        experts_root = (f"{parent_qname}.{attr_name}"
                        if parent_qname else attr_name)
        if layout == "nested":
            for eid_str in numeric_keys:
                expert_mod = child_dict[eid_str]
                for sub_name, sub_mod in expert_mod.named_modules():
                    if not isinstance(sub_mod, nn.Linear) or sub_name == "":
                        continue
                    leaf = f"{experts_root}.{eid_str}.{sub_name}"
                    expert_info[leaf] = (router_qname, eid_str)
        else:
            for proj_name in projection_candidates:
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                for eid_str in numeric_keys:
                    sub_mod = proj_children.get(eid_str)
                    if not isinstance(sub_mod, nn.Linear):
                        continue
                    leaf = f"{experts_root}.{proj_name}.{eid_str}"
                    expert_info[leaf] = (router_qname, eid_str)

    return expert_info


def discover_moe_routers(
    model: nn.Module,
    profile=None,
) -> dict[str, int]:
    """Return router module qnames and expert counts for all MoE blocks.

    ``discover_moe_structure`` intentionally describes *per-expert Linear*
    layouts.  Packed Qwen3.5/3.6 experts instead keep their projections as
    3-D parameters, so that function has no Linear leaves to return even
    though the sibling router is fully observable.  Coverage accounting only
    needs the router module itself; discover it independently from either a
    numbered expert container or a profile-declared packed 3-D parameter.
    """
    if profile is None:
        try:
            from .model_profiles import profile_from_model
            profile = profile_from_model(model)
        except Exception:
            profile = None
    packed_names = _packed_expert_param_name_set(profile)
    routers: dict[str, int] = {}

    def _matches(child: nn.Module, num_experts: int) -> bool:
        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return True
        weight = getattr(child, "weight", None)
        return bool(
            isinstance(weight, torch.Tensor)
            and weight.ndim >= 1
            and int(weight.shape[0]) == num_experts
        )

    for parent_qname, parent in model.named_modules():
        for attr in ("experts", "block_sparse_moe_experts",
                     "moe_experts", "expert_layer"):
            experts = getattr(parent, attr, None)
            if not isinstance(experts, nn.Module):
                continue
            numeric = [name for name, _child in experts.named_children()
                       if name.isdigit()]
            num_experts = len(numeric)
            if not num_experts:
                for name, param in experts.named_parameters(recurse=False):
                    if (name in packed_names and isinstance(param, torch.Tensor)
                            and param.ndim == 3 and int(param.shape[0]) > 0):
                        num_experts = int(param.shape[0])
                        break
            if not num_experts:
                continue
            for child_name, child in parent.named_children():
                if child is experts:
                    continue
                if _matches(child, num_experts):
                    qname = (f"{parent_qname}.{child_name}"
                             if parent_qname else child_name)
                    routers[qname] = num_experts
                    break
    return routers


def read_top_k(model: nn.Module, default: int = 2) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return default
    for attr in ("num_experts_per_tok", "moe_top_k", "num_active_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ("num_experts_per_tok", "moe_top_k"):
            v = getattr(text_cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return default


# ---------------------------------------------------------------------------
# The KV-cotangent path: Fisher cotangents for cross-layer SHARED state
# ---------------------------------------------------------------------------
def kv_cotangent_path_enabled() -> bool:
    """Whether the reverse sweep routes shared-state cotangents back to the
    producing layer (default: yes).

    `PRISMAQUANT_KV_COTANGENT=0` restores the pre-fix severed-cotangent
    behavior for an A/B — and, because that reintroduces the under-count, it
    also re-arms `incremental_probe.kv_shared_fisher_block_reason`. One
    definition, read by both the probe sweeps and that guard, so the flag can
    never mean two different things."""
    raw = os.environ.get("PRISMAQUANT_KV_COTANGENT")
    if raw is None:
        return True
    return raw not in ("0", "", "false", "False", "FALSE", "no", "NO")


class SharedStateCotangents:
    """Route the Fisher cotangent of cross-layer SHARED forward state back to
    the layer that PRODUCED it.

    **The measurement gap.** Some architectures share activations across layers
    within one forward pass: Gemma4's KV-sharing layers reuse the K/V computed
    by an earlier *storing* layer, threaded through the ``shared_kv_states``
    dict that ``ModelProfile.new_forward_pass_state`` declares. The Fisher
    probe's phase-3 forwards each layer in ISOLATION and hands a sharing layer
    the phase-1 *capture* of that K/V, which is necessarily detached (it is a
    CPU snapshot taken from a ``torch.no_grad()`` pass). A sharing layer's
    backward therefore stops dead at the borrowed K/V, and ``h_trace`` for the
    storing layer's ``k_proj``/``v_proj`` counts only its OWN layer's gradient
    — it misses the gradient flowing through every layer that consumes its
    K/V. The under-count lands precisely on the layers that feed other layers,
    i.e. exactly the ones the allocator should be most careful with. It also
    truncates the chained ``grad_out`` handed to every layer BELOW the storing
    layer, since that input gradient is missing the same paths.

    **The fix, in one reverse pass.** ``graft`` replaces each borrowed tensor
    with a grad-enabled LEAF clone. After that consumer's backward, ``harvest``
    reads ``leaf.grad`` — exactly the cotangent this consumer contributes to
    the producer's shared tensor — and sums it per (kwarg, source key,
    position); gradients from several consumers of one source add, which is
    what the sum is. When the sweep later reaches the producing layer,
    ``produced_roots`` returns the tensors that layer wrote into the container
    paired with those accumulated gradients, so the caller can drive one
    ``torch.autograd.backward([out, *roots], [grad_out, *grads])``. Autograd
    accumulates both contributions at the producer's K/V node before its
    ``grad_fn`` runs, so each Linear's full-backward hook still fires ONCE,
    with the total — the same gy an end-to-end backward would have delivered.

    **Why one pass suffices.** A consumer always comes AFTER its producer in
    the forward order (Gemma4 derives ``kv_shared_layer_index`` from the layers
    strictly BEFORE the sharing point), and phase-3 sweeps in REVERSE, so every
    consumer has been harvested by the time its producer is forwarded.

    Bookkeeping rules that keep this honest:

    - A borrowed leaf is never seeded as a root. It has no ``grad_fn`` (and its
      identity is tracked), so a source whose cotangent is still accumulating
      cannot be fed back into a later consumer and double-counted.
    - A seeded key is POPPED. Two layers writing one key would otherwise each
      collect the same consumers' cotangent; the sweep-end ``pending_keys``
      diagnostic reports anything never consumed.
    - Accumulation is promoted to at least fp32, so many bf16 consumer
      cotangents don't lose the small ones; the seed is cast back to the
      producer tensor's dtype.
    - ``graft`` of an empty/absent pass state returns the caller's own object
      and records nothing, so architectures that declare no shared state take
      byte-for-byte the same path they did before this class existed.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        # (kwarg, source key, position) -> summed cotangent (>= fp32).
        self._acc: dict[tuple, torch.Tensor] = {}
        # Per-layer: grafted leaves awaiting harvest, and the containers this
        # layer may have written into.
        self._live: list[tuple[tuple, torch.Tensor]] = []
        self._live_ids: set[int] = set()
        self._containers: list[tuple[str, dict]] = []
        # Diagnostics (reported at sweep end; never used to make a decision).
        self.n_grafted = 0
        self.n_harvested = 0
        self.n_seeded = 0
        self.n_no_grad = 0
        self.nondifferentiable: list[str] = []

    # -- consumer side ----------------------------------------------------
    def graft(self, pass_state):
        """Return `pass_state` with every borrowed tensor replaced by a
        grad-enabled leaf clone, recording the leaves for `harvest` and the
        containers for `produced_roots`. Hand the RESULT to the layer call."""
        if not self.enabled or not pass_state:
            return pass_state
        grafted: dict = {}
        tracked = False
        for kwarg, container in pass_state.items():
            if not isinstance(container, dict):
                # Not a keyed shared-state container; pass through untouched
                # rather than guess at its structure.
                grafted[kwarg] = container
                continue
            fresh = {
                key: self._graft_value(kwarg, key, value)
                for key, value in container.items()
            }
            grafted[kwarg] = fresh
            self._containers.append((kwarg, fresh))
            tracked = True
        return grafted if tracked else pass_state

    def _graft_value(self, kwarg, key, value):
        if isinstance(value, torch.Tensor):
            return self._leaf(kwarg, key, None, value)
        if isinstance(value, (tuple, list)):
            out = [
                self._leaf(kwarg, key, pos, item)
                if isinstance(item, torch.Tensor) else item
                for pos, item in enumerate(value)
            ]
            return tuple(out) if isinstance(value, tuple) else out
        return value

    def _leaf(self, kwarg, key, pos, tensor: torch.Tensor) -> torch.Tensor:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            # Integer/bool shared state carries no cotangent at all — the
            # producer's Fisher genuinely cannot be recovered through it.
            self.nondifferentiable.append(f"{kwarg}[{key!r}][{pos}]:{tensor.dtype}")
            return tensor
        try:
            slot = (kwarg, key, pos)
            hash(slot)
        except TypeError:
            self.nondifferentiable.append(f"{kwarg}[{key!r}]:unhashable-key")
            return tensor
        leaf = tensor.detach().clone().requires_grad_(True)
        self._live.append((slot, leaf))
        self._live_ids.add(id(leaf))
        self.n_grafted += 1
        return leaf

    def harvest(self) -> int:
        """Fold this layer's leaf gradients into the per-source accumulator and
        end the layer. Call AFTER the backward; returns the number folded."""
        folded = 0
        for slot, leaf in self._live:
            grad = leaf.grad
            if grad is None:
                # The layer never used the borrowed tensor on a path that
                # reaches its output (or the cotangent was structurally zero).
                self.n_no_grad += 1
                continue
            acc_dtype = torch.promote_types(grad.dtype, torch.float32)
            contrib = grad.detach().to(acc_dtype)
            prev = self._acc.get(slot)
            if prev is None:
                self._acc[slot] = contrib.clone()
            else:
                prev.add_(contrib)
            leaf.grad = None
            folded += 1
        self.n_harvested += folded
        self._live.clear()
        self._live_ids.clear()
        self._containers.clear()
        return folded

    # -- producer side ----------------------------------------------------
    def produced_roots(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """`(tensors, grads)` — extra backward roots for the layer just
        forwarded: the shared tensors IT produced, paired with the cotangent
        its consumers accumulated. Call after the forward, before the
        backward. Empty when this layer produced nothing consumers wanted."""
        if not self.enabled or not self._acc:
            return [], []
        order: list[int] = []
        by_id: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for kwarg, container in self._containers:
            for key, value in list(container.items()):
                for pos, tensor in self._iter_positions(value):
                    slot = (kwarg, key, pos)
                    grad = self._acc.get(slot)
                    if grad is None:
                        continue
                    if id(tensor) in self._live_ids or tensor.grad_fn is None:
                        # A tensor this layer BORROWED, not produced: it is a
                        # leaf clone with no graph. Seeding it would inject
                        # other consumers' cotangent into this layer's harvest.
                        continue
                    self._acc.pop(slot)
                    seed = grad.to(device=tensor.device, dtype=tensor.dtype)
                    if id(tensor) in by_id:
                        # Same object written at two positions (Gemma4's
                        # `value_states = key_states` k_eq_v layers): one root,
                        # summed grad — duplicate roots would be legal but this
                        # is unambiguous.
                        prev_t, prev_g = by_id[id(tensor)]
                        by_id[id(tensor)] = (prev_t, prev_g + seed)
                    else:
                        by_id[id(tensor)] = (tensor, seed)
                        order.append(id(tensor))
                    self.n_seeded += 1
        roots = [by_id[i][0] for i in order]
        grads = [by_id[i][1] for i in order]
        return roots, grads

    @staticmethod
    def _iter_positions(value):
        if isinstance(value, torch.Tensor):
            yield None, value
        elif isinstance(value, (tuple, list)):
            for pos, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    yield pos, item

    # -- diagnostics ------------------------------------------------------
    def pending_keys(self) -> list[tuple]:
        """Accumulated cotangents no producer ever claimed. Non-empty means the
        sweep never forwarded the producing layer (or it stopped writing the
        key), so that producer's Fisher is still under-counted."""
        return sorted(self._acc, key=repr)

    def summary(self) -> str:
        pending = self.pending_keys()
        parts = [
            f"enabled={self.enabled}",
            f"grafted={self.n_grafted}",
            f"harvested={self.n_harvested}",
            f"seeded={self.n_seeded}",
        ]
        if self.n_no_grad:
            parts.append(f"no_grad={self.n_no_grad}")
        if self.nondifferentiable:
            parts.append(f"nondifferentiable={len(self.nondifferentiable)}")
        if pending:
            parts.append(f"UNCLAIMED={[repr(k) for k in pending]}")
        return "kv_cotangent[" + " ".join(parts) + "]"


def _streaming_visual_layer_kwargs(
    profile,
    *,
    input_ids=None,
    pass_state: dict | None = None,
    captured_pass_state=None,
    layer=None,
    cotangents: "SharedStateCotangents | None" = None,
) -> dict:
    """Resolve the profile kwargs for one layer call in the streaming visual
    probe: per-layer `extra_layer_kwargs` plus the per-pass SHARED state —
    either the live dict threaded through the forward pass (`pass_state`, one
    object for the whole pass) or, for the isolated reverse-sweep forwards,
    the slice rebuilt from what that pass captured (`captured_pass_state`).

    Merging goes through `merge_pass_state_kwargs` so the shallow-merge /
    no-op-when-empty / raise-on-collision rule has one definition shared with
    `_call_layer`.

    `cotangents` (reverse sweep only) grafts grad-enabled leaves over the
    borrowed tensors so the producing layer's Fisher gets its consumers'
    cotangent — see `SharedStateCotangents`. Grafting happens on the isolated
    slice, before the merge, so only declared shared containers are touched."""
    from .layer_streaming import merge_pass_state_kwargs

    ctx = type(layer).__name__ if layer is not None else "streaming layer"
    kwargs = dict(profile.extra_layer_kwargs(input_ids=input_ids))
    kwargs = merge_pass_state_kwargs(kwargs, pass_state, context=ctx)
    if captured_pass_state is not None and layer is not None:
        isolated = profile.isolated_layer_pass_state(captured_pass_state, layer)
        if cotangents is not None:
            isolated = cotangents.graft(isolated)
        kwargs = merge_pass_state_kwargs(kwargs, isolated, context=ctx)
    return kwargs


# ---------------------------------------------------------------------------
# Router tracker: per-(router, expert) activation probability
# ---------------------------------------------------------------------------
class RouterTracker:
    def __init__(self, model: nn.Module, routers: list[str], top_k: int):
        self.top_k = top_k
        self.counts_t: dict[str, torch.Tensor] = {}
        self.active_counts_t: dict[str, torch.Tensor] = {}
        self.total_tokens: dict[str, int] = defaultdict(int)
        self._handles = []
        for rq in routers:
            try:
                mod = model.get_submodule(rq)
            except AttributeError:
                continue
            n_experts = None
            if isinstance(mod, nn.Linear):
                n_experts = mod.out_features
            else:
                weight = getattr(mod, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
                    n_experts = int(weight.shape[0])
            if not isinstance(n_experts, int) or n_experts <= 0:
                continue
            self.counts_t[rq] = torch.zeros(n_experts, dtype=torch.float64)
            self.active_counts_t[rq] = torch.zeros(n_experts, dtype=torch.int64)
            self._handles.append(mod.register_forward_hook(self._make_hook(rq)))

    def _make_hook(self, router_qname: str):
        def hook(module, inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            flat = scores.detach().reshape(-1, scores.size(-1))
            k = min(self.top_k, flat.size(-1))
            topk_v, topk_i = flat.topk(k, dim=-1)
            probs = F.softmax(topk_v, dim=-1)
            weighted = torch.bincount(
                topk_i.reshape(-1),
                weights=probs.reshape(-1).to(torch.float64),
                minlength=int(scores.size(-1)),
            )
            active = torch.bincount(
                topk_i.reshape(-1),
                minlength=int(scores.size(-1)),
            )
            self.total_tokens[router_qname] += flat.size(0)
            self.counts_t[router_qname].add_(weighted.cpu())
            self.active_counts_t[router_qname].add_(active.cpu().to(torch.int64))
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def prob(self, router_qname: str, eid: str) -> float:
        total = self.total_tokens.get(router_qname, 0)
        if total == 0:
            return 0.0
        counts = self.counts_t.get(router_qname)
        if counts is None:
            return 0.0
        idx = int(eid)
        if idx < 0 or idx >= counts.numel():
            return 0.0
        return float(counts[idx].item()) / total

    @property
    def counts(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for router, counts in self.counts_t.items():
            nz = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
            out[router] = {
                str(int(i)): float(counts[int(i)].item())
                for i in nz.tolist()
            }
        return out

    @property
    def active_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for router, counts in self.active_counts_t.items():
            nz = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
            out[router] = {
                str(int(i)): int(counts[int(i)].item())
                for i in nz.tolist()
            }
        return out

    @property
    def route_stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        active = self.active_counts
        for router, counts in self.counts.items():
            total = int(self.total_tokens.get(router, 0))
            denom = max(total, 1)
            out[router] = {
                "total_tokens": total,
                "mass": counts,
                "active_count": active.get(router, {}),
                "prob": {
                    eid: float(mass) / denom
                    for eid, mass in counts.items()
                },
            }
        return out


# ---------------------------------------------------------------------------
# Fisher accumulator with activation snapshot cache
# ---------------------------------------------------------------------------
class FisherAccumulator:
    def __init__(self, model: nn.Module, tracked: list[str],
                 expert_info: dict[str, tuple[str, str]],
                 act_cache_dir: Path | None = None,
                 input_rows: int = 256,
                 hook_packed_experts: bool = True,
                 h_detail_dir: Path | None = None,
                 model_profile=None):
        self.stats: dict[str, dict] = {}
        self._saved_inputs: dict[str, torch.Tensor] = {}
        self._fwd_handles, self._bwd_handles = [], []
        self.tracked = set(tracked)
        self.expert_info = expert_info
        self.cache_dir = act_cache_dir
        self.h_detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self.input_rows = input_rows
        self._input_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._input_row_indices: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._input_token_offsets: dict[str, int] = defaultdict(int)
        self._rows_got: dict[str, int] = defaultdict(int)
        # Packed expert Fisher-trace accumulator: written by
        # _PackedLinearPerTokenFisher during backward (per-token-summed;
        # values may be device-resident 0-dim fp32 tensors), read in
        # finalize().
        self._packed_grad_acc: dict[str, float | torch.Tensor] = {}
        # AQUA per-expert A-side marginals, same estimator and same merge
        # rules as the incremental backend (the two backends agree on the
        # packed path or a card built by one mis-prices under the other).
        self._packed_marginal_acc: dict[str, dict[str, torch.Tensor]] = {}
        # Per-(experts module qname) sample count (one per backward),
        # populated by the experts forward hook below.
        self._packed_sample_count: dict[str, int] = defaultdict(int)
        # Per-experts-module activation snapshots, captured live so
        # measure_quant_cost can read packed expert inputs.
        self._packed_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._packed_act_rows: dict[str, int] = defaultdict(int)
        if model_profile is None:
            try:
                from .model_profiles import profile_from_model
                model_profile = profile_from_model(model)
            except Exception:
                model_profile = None
        self.model_profile = model_profile
        # One shared name projection (R5 consumer migration). Block
        # identity and the declared structural spellings are read through
        # prismaquant.name_projection, never re-derived here from string
        # surgery. When the model's own profile is unavailable, fall back
        # to DefaultProfile exactly like `_packed_expert_param_name_set`
        # above; NameProjection itself refuses a None profile by contract,
        # so the fallback is an explicit declaration, not a silent degrade.
        if self.model_profile is not None:
            self._name_projection = NameProjection(self.model_profile)
        else:
            from .model_profiles import DefaultProfile
            self._name_projection = NameProjection(DefaultProfile())

        # Per-layer accumulator for full per-weight Fisher diagonal.
        # Keyed by Linear qname -> CPU fp64 tensor of shape [out, in]
        # matching the weight shape. Accumulated by the backward hook;
        # written to `probe_detail/<sanitized_name>.pt` in finalize()
        # so the probe pickle stays small while downstream consumers
        # can lazy-load the full H per layer.
        self._h_full: dict[str, torch.Tensor] = {}
        # Per-token squared gradient norm: ‖∇W_t‖²_F = ‖gy_t‖²·‖x_t‖² as a
        # scalar per (Linear, token). Stored as a list of per-chunk CPU
        # vectors during the hook; concatenated at finalize() and saved to
        # h_detail so research reconstruction tools can build the
        # Fisher-weighted GPTQ Hessian H = X^T diag(g²_t) X. Storage cost
        # is ~4 GB total at 32k tokens × 33k Linears × 4 bytes.
        self._per_token_grad_norm: dict[str, list[torch.Tensor]] = {}
        # GPU-resident scalar accumulators. Used by the backward hook to
        # avoid `.item()` syncs in the hot path — every `.sum().item()`
        # forces a CUDA→CPU stall, and with 32k Linears × multiple syncs
        # per backward pass, those compound into seconds of serialization.
        # We accumulate to a 0-dim GPU tensor here, then flush to the
        # Python scalar `stats[name]["h_trace_raw"]` once at finalize().
        self._gpu_h_trace: dict[str, torch.Tensor] = {}
        self._gpu_h_w2_sum: dict[str, torch.Tensor] = {}
        # Linears we never want Fisher signal for (always BF16 in our
        # serving stack, so accumulating the full per-weight matrix is
        # dead work). Populated in install_hooks() based on a regex; the
        # hook short-circuits for any name in this set.
        self._fisher_skip: set[str] = set()
        # Batched MoE Fisher (Task #48). When a Linear is part of a
        # detected MoE expert block, the per-Linear backward hook only
        # SAVES (X, gy); the parent block's backward hook then does ONE
        # batched bmm per (block, w_name) instead of N=num_experts
        # per-Linear matmuls. Reduces kernel launches by ~100× for the
        # MoE-heavy backward sweep.
        #
        # _moe_linear_to_block: linear_qname -> (block_qname, eid, w_name)
        # _moe_block_to_linears: block_qname -> {(eid, w_name): linear_qname}
        # _moe_pending: block_qname -> {(eid, w_name): (X_2d, gy_2d)}
        self._moe_linear_to_block: dict[str, tuple[str, int, str]] = {}
        self._moe_block_to_linears: dict[str, dict[tuple[int, str], str]] = {}
        self._moe_pending: dict[str, dict[tuple[int, str], tuple]] = {}
        self._moe_block_handles: list = []
        # Same idea for packed experts, but reduced to per-expert
        # per-output-channel: [E, M] instead of [E, M, N]. Full
        # per-weight for 80 packed tensors at 35B scale is 160+ GB;
        # per-channel is 160 MB total — still a vector form.
        self._h_packed_channel: dict[str, torch.Tensor] = {}
        # Pre-compute the BF16-skip set: profile-pinned Linears end up BF16
        # in serving, so accumulating their full per-weight Fisher matrix is
        # dead work. The embedding spelling is the profile's DECLARATION
        # (ModelProfile.embedding_name), not a probe substring test; the
        # embedding fallback only fires for profiles that implement the
        # table as an nn.Linear (embeddings are normally not nn.Linear and
        # therefore never enter this loop).
        #
        # Both accessors are read defensively, for the same reason: skipping
        # here is an OPTIMIZATION, never a correctness gate, so a profile that
        # declares neither simply does the extra work. `model_profile` is a
        # loosely-typed optional kwarg that duck-typed objects reach (see
        # tests/test_incremental_measure_quant_cost.py), and a probe must not
        # start refusing profiles over a field that only saves it work.
        _embedding_decl = getattr(
            self._name_projection.profile, "embedding_name", None)
        _declared_embedding = (
            str(_embedding_decl()) if _embedding_decl is not None else None)
        def _skip_fisher_name(qname: str) -> bool:
            checker = getattr(self.model_profile, "is_pinned_name", None)
            if checker is not None and checker(qname):
                return True
            return qname == _declared_embedding

        # Detect MoE expert blocks for batched Fisher accumulation. A block
        # qualifies if its immediate children all expose w1/w2/w3 nn.Linear
        # attributes (the MiniMaxM2Experts / Qwen3.5MoeExpert layout). Each
        # qualifying block gets a parent-level backward hook installed that
        # batches per-expert Fisher into one bmm per w_name. Returns the
        # set of Linear qnames that will be deferred (handled by the block).
        def _scan_moe_blocks() -> set[str]:
            deferred: set[str] = set()
            # Per-expert module attribute names come from the model profile so
            # families whose experts expose gate/up/down_proj (e.g. DSv4) are
            # detected instead of silently no-opping the batched path. The
            # default ('w1','w2','w3') keeps Qwen3/Qwen3.5 behavior unchanged.
            _proj = getattr(
                self.model_profile, "unpacked_expert_projection_names", None)
            w_attrs = tuple(_proj()) if callable(_proj) else ("w1", "w2", "w3")
            for block_name, block in model.named_modules():
                children = list(block.named_children())
                if not children or len(children) < 2:
                    continue
                # Each child must look like a MoE expert with per-projection
                # nn.Linear attributes named by w_attrs (above).
                ok = True
                for _, child in children:
                    for w in w_attrs:
                        sub = getattr(child, w, None)
                        if not isinstance(sub, nn.Linear):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Block is an MoE expert container.
                self._moe_block_to_linears[block_name] = {}
                for cname, child in children:
                    try:
                        eid = int(cname)
                    except ValueError:
                        # Non-integer child name — not a regular MoE expert
                        # container; skip the whole block.
                        del self._moe_block_to_linears[block_name]
                        ok = False
                        break
                    for w in w_attrs:
                        linear = getattr(child, w)
                        linear_qname = f"{block_name}.{cname}.{w}"
                        if linear_qname not in self.tracked:
                            continue
                        self._moe_linear_to_block[linear_qname] = (block_name, eid, w)
                        self._moe_block_to_linears[block_name][(eid, w)] = linear_qname
                        deferred.add(linear_qname)
                if not ok:
                    continue
                if not self._moe_block_to_linears.get(block_name):
                    # No tracked Linears under this block in this shard —
                    # don't install a block hook (would never have data
                    # to flush).
                    self._moe_block_to_linears.pop(block_name, None)
                    continue
                # Install the block-level backward flush hook.
                handle = block.register_full_backward_hook(
                    self._make_moe_block_flush(block_name))
                self._moe_block_handles.append(handle)
            return deferred

        self._moe_deferred_linears = _scan_moe_blocks()
        # Always log so we can confirm detection ran and see why the count
        # might be zero (regex-tracked vs named-modules-name mismatch is
        # the most common failure mode).
        _example_tracked = next(iter(self.tracked), "<empty>")
        print(f"[fisher] batched-MoE detect: {len(self._moe_block_to_linears)} "
              f"expert blocks, {len(self._moe_deferred_linears)} deferred "
              f"Linears | tracked_count={len(self.tracked)} "
              f"tracked_example={_example_tracked!r}",
              flush=True)

        for name, mod in model.named_modules():
            if name not in self.tracked or not isinstance(mod, nn.Linear):
                continue
            if _skip_fisher_name(name):
                self._fisher_skip.add(name)
            w = mod.weight
            # Grouped-BMM operand (wo_a shape): route to the grouped
            # accumulator BEFORE the dense registration below. Its stats
            # entry carries `num_groups`; the flat-plane dims let every
            # dense consumer (byte math, card validate, shape gates) read
            # it unchanged.
            num_groups = grouped_linear_groups(mod, self.model_profile)
            if num_groups is not None:
                self.stats[name] = grouped_linear_stats_entry(
                    mod, num_groups,
                    w_max_abs=None if w.is_meta else float(
                        w.detach().abs().max().item()),
                    w_norm_sq=None if w.is_meta else float(
                        w.detach().pow(2).sum().item()),
                )
                self._h_full[name] = (
                    None if w.is_meta
                    else torch.zeros(int(w.shape[0]), int(w.shape[1]),
                                     dtype=torch.float32, device=w.device))
                self._fwd_handles.append(
                    mod.register_forward_hook(self._make_fwd(name)))
                self._bwd_handles.append(mod.register_full_backward_hook(
                    self._make_grouped_bwd(name, num_groups)))
                continue
            router_qname, eid = expert_info.get(name, (None, None))
            # Weights loaded under accelerate disk offload start on the
            # meta device and materialize lazily during forward. Defer
            # the scalar weight statistics and the H-full accumulator
            # allocation until `_make_fwd`/`_make_bwd` sees a real tensor.
            if w.is_meta:
                w_max_abs = None
                w_norm_sq = None
            else:
                w_max_abs = float(w.detach().abs().max().item())
                w_norm_sq = float(w.detach().pow(2).sum().item())
            self.stats[name] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,
                "w_max_abs": w_max_abs,
                "w_norm_sq": w_norm_sq,
                "n_params": int(w.numel()),
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "n_tokens_seen": 0,
                "route_prob": None,
                "router_path": router_qname,
                "expert_id": eid,
            }
            if w.is_meta:
                # Leave the accumulator slot empty; `_make_bwd` will
                # allocate it the first time a real-device gradient
                # flows through this Linear.
                self._h_full[name] = None
            else:
                self._h_full[name] = torch.zeros(
                    mod.out_features, mod.in_features,
                    dtype=torch.float32, device=w.device,
                )
            self._fwd_handles.append(
                mod.register_forward_hook(self._make_fwd(name)))
            self._bwd_handles.append(
                mod.register_full_backward_hook(self._make_bwd(name, mod)))

        if hook_packed_experts:
            packed_meta = install_packed_expert_hooks(
                model,
                accumulator=self._packed_grad_acc,
                channel_accumulator=self._h_packed_channel,
                marginal_accumulator=self._packed_marginal_acc,
                profile=self.model_profile,
            )
            for full_name, meta in packed_meta.items():
                # Filter against the tracked set when tracked is a regex
                # match; here we accept any packed param under a tracked
                # parent (the regex from run_probe_pass already filters
                # by layer).
                experts_qname = meta.pop("_packed_experts_module")
                meta.pop("_packed_param", None)
                # Include the packed entry iff its BLOCK has tracked
                # Linears, so shard regexes (`model.layers.X.`) work
                # cleanly. The block id comes from the shared name
                # projection (decision_units.block_id_from_qname via
                # NameProjection.block_id) — not a positional slice of
                # the qname, which is only correct by accident when the
                # layer prefix happens to be two components wide.
                parent_layer = self._name_projection.block_id(experts_qname)
                if any(t.startswith(parent_layer + ".") for t in self.tracked):
                    self.stats[full_name] = meta
                    # Register a forward hook on the experts module to
                    # bump the per-backward sample count (used to keep
                    # n_tokens_seen aligned with the Linear path's
                    # accounting).
                    try:
                        experts_mod = model.get_submodule(experts_qname)
                    except AttributeError:
                        continue

                    def _exp_fwd(_mod, inp, _out, _qn=experts_qname,
                                 _full=full_name, _x_acc=self._packed_act_snaps,
                                 _r=self._packed_act_rows):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        if isinstance(x, torch.Tensor):
                            self._packed_sample_count[_full] += int(
                                x.detach().reshape(-1, x.size(-1)).size(0))
                            if act_cache_dir is not None:
                                need = self.input_rows - _r[_qn]
                                if need > 0:
                                    flat = x.detach().reshape(-1, x.size(-1))
                                    if flat.size(0) > need:
                                        idx = torch.randperm(flat.size(0),
                                                             device=flat.device)[:need]
                                        flat = flat.index_select(0, idx)
                                    _x_acc[_qn].append(flat.to("cpu"))
                                    _r[_qn] += flat.size(0)

                    self._fwd_handles.append(
                        experts_mod.register_forward_hook(_exp_fwd))

    def _make_fwd(self, name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self._saved_inputs[name] = x.detach()
            # Fill in deferred weight stats for disk-offloaded Linears
            # the first time a real tensor becomes available.
            stats = self.stats.get(name)
            if stats is not None and stats.get("w_max_abs") is None:
                w = module.weight
                if w is not None and not w.is_meta:
                    wd = w.detach()
                    stats["w_max_abs"] = float(wd.abs().max().item())
                    stats["w_norm_sq"] = float(wd.pow(2).sum().item())
            if self.cache_dir is not None:
                need = self.input_rows - self._rows_got[name]
                flat = x.detach().reshape(-1, x.size(-1))
                base = int(self._input_token_offsets[name])
                self._input_token_offsets[name] += int(flat.size(0))
                if need > 0:
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                        flat = flat.index_select(0, idx)
                    else:
                        idx = torch.arange(flat.size(0), device=flat.device)
                    self._input_snaps[name].append(flat.to("cpu"))
                    self._input_row_indices[name].append(
                        (idx.detach().to("cpu", dtype=torch.long) + base)
                    )
                    self._rows_got[name] += flat.size(0)
        return hook

    def _make_grouped_bwd(self, name: str, num_groups: int):
        """Backward hook for a grouped-BMM operand (wo_a shape).

        Same accumulators and same flush discipline as the dense
        `_make_bwd` (device-resident 0-dim fp32 scalars, `h_full`,
        per-token grad norms), but the reductions run through
        `grouped_linear_fisher_chunk`, which keeps the group axis
        explicit instead of flattening it into the token axis. Scalars
        land in `self._gpu_h_trace` / `self._gpu_h_w2_sum`, so the
        existing finalize() flush and the ONE global-token normalization
        in `finalize_fisher_stats` apply unchanged."""
        def hook(module, grad_input, grad_output):
            # Fast skip: same contract as the dense hook.
            if name in self._fisher_skip:
                self._saved_inputs.pop(name, None)
                return
            gy = grad_output[0]
            x = self._saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            pieces = grouped_linear_fisher_chunk(
                x, gy, num_groups, module.weight)
            gpu_trace = self._gpu_h_trace.get(name)
            if gpu_trace is None:
                gpu_trace = torch.zeros(
                    (), dtype=torch.float32, device=pieces["h_trace"].device)
                self._gpu_h_trace[name] = gpu_trace
            gpu_trace.add_(pieces["h_trace"])
            w2 = pieces.get("h_w2")
            if w2 is not None:
                gpu_w2 = self._gpu_h_w2_sum.get(name)
                if gpu_w2 is None:
                    gpu_w2 = torch.zeros(
                        (), dtype=torch.float32, device=w2.device)
                    self._gpu_h_w2_sum[name] = gpu_w2
                gpu_w2.add_(w2)
            acc = self._h_full.get(name)
            if acc is None:
                acc = torch.zeros(
                    int(pieces["chunk_flat"].shape[0]),
                    int(pieces["chunk_flat"].shape[1]),
                    dtype=torch.float32, device="cpu")
                self._h_full[name] = acc
            acc.add_(pieces["chunk_flat"].to(acc.device))
            # Per-(token, group) grad-norm² rows: ‖gy_t,g‖²·‖x_t,g‖² —
            # the plane-coordinate analog of the dense per-token vector,
            # so h_detail reconstruction tooling reads it unchanged.
            gy_sq = gy.detach().reshape(-1, num_groups, gy.shape[-1]).pow(2)
            x_sq = x.detach().reshape(-1, num_groups, x.shape[-1]).pow(2)
            pt = (gy_sq.sum(dim=-1) * x_sq.sum(dim=-1)).reshape(-1)
            self._per_token_grad_norm.setdefault(name, []).append(
                pt.detach().to("cpu", dtype=torch.float32))
            # TOKENS, not token-group pairs: one token's group slices are
            # one token's evidence. The normalization denominator is the
            # global calibration count either way (`finalize_fisher_stats`);
            # this field is metadata and must count what a dense row counts.
            tokens = int(x.detach().numel()) // (num_groups * int(x.shape[-1]))
            self.stats[name]["n_tokens_seen"] += tokens
        return hook

    def _make_moe_block_flush(self, block_name: str):
        """Block-level backward hook: pops the per-Linear pending data
        gathered by individual MoE-expert Linear hooks, batches the
        Fisher accumulation by (w_name) group, and scatters the per-
        Linear results into the existing per-Linear accumulators
        (gpu_h_trace, gpu_h_w2_sum, h_full, per_token_grad_norm).

        Numerically equivalent to running `_make_bwd`'s per-Linear path
        for each expert separately — same per-token-summed empirical
        Fisher (Σ_t gy²·x² element-wise), just executed as one bmm per
        w_name instead of `num_experts` independent matmuls.
        """
        def flush(module, grad_input, grad_output):
            pending = self._moe_pending.pop(block_name, None)
            if not pending:
                return
            # Group by w_name: list of (eid, name, X, gy, T, weight)
            from collections import defaultdict as _dd
            by_w: dict[str, list] = _dd(list)
            for (eid, w_name), (X, gy, lname, T, w) in pending.items():
                by_w[w_name].append((eid, lname, X, gy, T, w))

            for w_name, items in by_w.items():
                if not items:
                    continue
                # All items in this group share in_features (X.size(1)) and
                # out_features (gy.size(1)). Token counts vary per expert
                # — pad to max_T so we can stack into bmm-shaped tensors.
                in_dim = items[0][2].size(1)
                out_dim = items[0][3].size(1)
                max_T = max(it[4] for it in items)
                n_e = len(items)
                device = items[0][2].device
                # Pad-stacked tensors in float32 for accumulation
                # precision (mantissa) in the bmm reductions below. Note
                # bf16 squaring itself would NOT underflow — bf16 shares
                # fp32's exponent range; the incremental path's
                # bf16-square + fp32-accumulate is equally sound.
                X_pad = torch.zeros(n_e, max_T, in_dim,
                                    dtype=torch.float32, device=device)
                gy_pad = torch.zeros(n_e, max_T, out_dim,
                                     dtype=torch.float32, device=device)
                T_valid = torch.empty(n_e, dtype=torch.long, device=device)
                for i, (_eid, _lname, X, gy, T, _w) in enumerate(items):
                    X_pad[i, :T, :] = X.float()
                    gy_pad[i, :T, :] = gy.float()
                    T_valid[i] = T
                # Element-wise squared, then bmm — same arithmetic as the
                # single-Linear path but done across all experts at once.
                X_sq = X_pad.pow(2)               # (n_e, max_T, in_dim)
                gy_sq = gy_pad.pow(2)             # (n_e, max_T, out_dim)
                # chunk_h_batch[e] = gy_sq[e].T @ X_sq[e]  -> (out, in)
                chunk_h_batch = gy_sq.transpose(1, 2).bmm(X_sq)  # (n_e, out, in)
                # Per-token Fisher norm (for trace + per_token_g²)
                gy_norm = gy_sq.sum(dim=2)        # (n_e, max_T)
                x_norm = X_sq.sum(dim=2)          # (n_e, max_T)
                per_token = gy_norm * x_norm      # (n_e, max_T)
                # Mask padded slots
                mask = (torch.arange(max_T, device=device).unsqueeze(0)
                        < T_valid.unsqueeze(1)).to(per_token.dtype)
                per_token = per_token * mask
                trace_per_e = per_token.sum(dim=1)            # (n_e,)
                # Scatter to per-Linear accumulators.
                for i, (_eid, lname, _X, _gy, T, w) in enumerate(items):
                    # Trace
                    gpu_t = self._gpu_h_trace.get(lname)
                    if gpu_t is None:
                        gpu_t = torch.zeros((), dtype=torch.float32,
                                            device=device)
                        self._gpu_h_trace[lname] = gpu_t
                    gpu_t.add_(trace_per_e[i])
                    # h_full (CPU-resident)
                    chunk_h_e = chunk_h_batch[i]   # (out, in), still on GPU
                    acc = self._h_full.get(lname)
                    if acc is None:
                        acc = torch.zeros(out_dim, in_dim,
                                          dtype=torch.float32, device="cpu")
                        self._h_full[lname] = acc
                    acc.add_(chunk_h_e.cpu())
                    # h_w2_sum proxy (weight-aware scalar)
                    if w is not None and not w.is_meta:
                        wd = w.detach()
                        w2_contrib = (chunk_h_e
                                      * wd.float().pow(2).to(chunk_h_e.device)).sum()
                        gpu_w2 = self._gpu_h_w2_sum.get(lname)
                        if gpu_w2 is None:
                            gpu_w2 = torch.zeros((), dtype=torch.float32,
                                                 device=device)
                            self._gpu_h_w2_sum[lname] = gpu_w2
                        gpu_w2.add_(w2_contrib)
                    # Per-token g² (only the valid-T prefix)
                    pt_valid = per_token[i, :T].detach().cpu()
                    self._per_token_grad_norm.setdefault(lname, []).append(pt_valid)
        return flush

    def _make_bwd(self, name: str, mod_ref: nn.Linear):
        def hook(module, grad_input, grad_output):
            # Fast skip: Linears we know will end up BF16 in serving don't
            # need Fisher signal (lm_head, embed_tokens). Drop the saved
            # forward input so memory doesn't leak, then bail.
            if name in self._fisher_skip:
                self._saved_inputs.pop(name, None)
                return
            gy = grad_output[0]
            x = self._saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            gy2 = gy.reshape(-1, gy.size(-1))   # (T, out)
            x2 = x.reshape(-1, x.size(-1))      # (T, in)
            T = x2.size(0)
            # Batched MoE Fisher: if this Linear is part of an MoE expert
            # block, just stash (X, gy) for the block-level flush. The
            # per-Linear matmul + accumulator update happens batched there.
            moe_meta = self._moe_linear_to_block.get(name)
            if moe_meta is not None:
                block_name, eid, w_name = moe_meta
                pending = self._moe_pending.setdefault(block_name, {})
                pending[(eid, w_name)] = (
                    x2.detach(), gy2.detach(), name, T, mod_ref.weight,
                )
                self.stats[name]["n_tokens_seen"] += T
                return
            # --- CORRECT empirical-Fisher accumulation (per-token sum) ---
            # The buggy form `(Σ_t gy_t · x_t^T).pow(2)` is
            # ‖Σ_t ∇_t‖²_F = Σ_t ‖∇_t‖²_F + 2·Σ_{t<s}⟨∇_t,∇_s⟩,
            # which inflates curvature by the cross-token gradient covariance.
            # Empirically this overestimates Fisher by 5-50× for sequences
            # with correlated gradients (notably attention layers), and the
            # bias is layer-non-uniform → biased allocator decisions.
            #
            # The corrected form below uses the outer-product norm identity
            # ‖a · b^T‖²_F = ‖a‖² · ‖b‖² for the trace, and the elementwise
            # (gy²)^T @ (x²) identity for the per-weight diagonal:
            #   Σ_t (gy_{t,i} · x_{t,j})² = Σ_t gy_{t,i}² · x_{t,j}²
            # Trace cost drops from O(out·in) to O(T·(out+in)).
            gy2_sq = gy2.float().pow(2)         # (T, out)
            x2_sq = x2.float().pow(2)           # (T, in)
            gy_norm_sq = gy2_sq.sum(dim=1)      # (T,)  = ‖gy_t‖²
            x_norm_sq = x2_sq.sum(dim=1)        # (T,)  = ‖x_t‖²
            per_token_grad_norm_sq = gy_norm_sq * x_norm_sq  # (T,) = ‖∇_t‖²_F
            # Trace = Σ_t ‖∇_t‖²_F. Accumulate on GPU (0-dim tensor) to
            # avoid the CUDA→CPU sync that .item() would force per call.
            # Flushed to the Python scalar at finalize().
            gpu_trace = self._gpu_h_trace.get(name)
            if gpu_trace is None:
                gpu_trace = torch.zeros((), dtype=torch.float32,
                                        device=per_token_grad_norm_sq.device)
                self._gpu_h_trace[name] = gpu_trace
            gpu_trace.add_(per_token_grad_norm_sq.sum())
            # Per-weight diagonal accumulator: h_full[i,j] = Σ_t gy²_{t,i}·x²_{t,j}
            # Compute the matmul once, reuse for both the accumulator and the
            # weight-aware scalar proxy. (Earlier draft computed it twice.)
            chunk_h = gy2_sq.t() @ x2_sq        # (out, in)
            acc = self._h_full.get(name)
            if acc is None:
                # Deferred allocation for disk-offloaded Linears (CPU-resident
                # so offloaded layers don't pin GPU per-Linear).
                acc = torch.zeros(
                    int(gy2.size(1)), int(x2.size(1)),
                    dtype=torch.float32, device="cpu",
                )
                self._h_full[name] = acc
            acc.add_(chunk_h.to(acc.device))
            # h_w2_sum (weight-aware scalar proxy): ⟨chunk_h, w²⟩.
            # Same GPU-tensor accumulation trick as above — keep it off the
            # synchronous CPU path during the hot per-Linear hook.
            w = mod_ref.weight
            if w is not None and not w.is_meta:
                wd = w.detach()
                w2_contrib = (chunk_h * wd.float().pow(2).to(chunk_h.device)).sum()
                gpu_w2 = self._gpu_h_w2_sum.get(name)
                if gpu_w2 is None:
                    gpu_w2 = torch.zeros((), dtype=torch.float32,
                                         device=w2_contrib.device)
                    self._gpu_h_w2_sum[name] = gpu_w2
                gpu_w2.add_(w2_contrib)
            # --- Per-token grad-norm² (Task #36, DeepSeek's correct form) ---
            # Stored as a list of per-chunk vectors; flushed to disk at
            # finalize() into h_detail so archived/research reconstruction
            # tools can build the truly Fisher-weighted GPTQ Hessian
            # H = X^T diag(g²_t) X.
            ptl = self._per_token_grad_norm.setdefault(name, [])
            ptl.append(per_token_grad_norm_sq.detach().cpu())
            self.stats[name]["n_tokens_seen"] += T
        return hook

    def finalize(self, tracker: RouterTracker | None, global_tokens: int):
        # Flush GPU-resident scalar accumulators (h_trace, h_w2_sum) into
        # the stats dict. Single sync per Linear here costs one CUDA stall
        # per name, vs. thousands during the backward sweep without it.
        for nm, gpu_t in self._gpu_h_trace.items():
            if nm in self.stats:
                self.stats[nm]["h_trace_raw"] += float(gpu_t.item())
        for nm, gpu_t in self._gpu_h_w2_sum.items():
            if nm in self.stats:
                self.stats[nm]["h_w2_sum_raw"] += float(gpu_t.item())
        # Flush packed-expert grad-norm accumulator into stats h_trace_raw.
        # The packed accumulator key matches the stats key by construction
        # (full param name `<experts_qname>.<param_name>`).
        for full_name, raw in self._packed_grad_acc.items():
            if full_name in self.stats:
                self.stats[full_name]["h_trace_raw"] += float(raw)
                self.stats[full_name]["n_tokens_seen"] = int(
                    self._packed_sample_count.get(full_name, 0))
        for full_name, slot in self._packed_marginal_acc.items():
            entry = self.stats.get(full_name)
            if entry is None:
                continue
            for key, tensor in slot.items():
                entry[key] = tensor.detach().to("cpu").numpy().astype(
                    np.float64 if key == "expert_tokens" else np.float32)

        if tracker is not None:
            for name, s in self.stats.items():
                if s["router_path"]:
                    s["route_prob"] = tracker.prob(
                        s["router_path"], s["expert_id"])

        # Single normalization convention (matches incremental_probe, the
        # production backend): ONE shared denominator for every row — the
        # GLOBAL calibration token count.
        #
        # HISTORY: this deliberately reverses the per-`n_tokens_seen`
        # division that audit M4 documented as "the one implicit
        # ÷token-fraction the MoE convention prescribes" (the explicit
        # ÷route_prob was removed then precisely because the implicit
        # per-routed-token division already applied it). That reading was
        # wrong for the mean-Δloss objective: tokens never routed to an
        # expert contribute zero gradient, so dividing an unpacked expert
        # row by its ROUTED count inflates it by (global/routed) — the
        # very 1/p_e overweighting M4 set out to remove, merely implicit.
        # See finalize_fisher_stats for the full derivation + history.
        # `route_prob` stays in the stats as metadata only.
        global_tokens = max(int(global_tokens), 1)
        finalize_fisher_stats(self.stats, global_tokens)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for name, snaps in self._input_snaps.items():
                if not snaps:
                    continue
                X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
                row_parts = self._input_row_indices.get(name, [])
                row_indices = (
                    torch.cat(row_parts, dim=0).to(torch.long).contiguous()
                    if row_parts else None
                )
                payload = {"inputs": X, "name": name}
                if row_indices is not None and row_indices.numel() == X.shape[0]:
                    payload["row_indices"] = row_indices
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
                torch.save(payload, self.cache_dir / fname)
            # Also write packed-experts module input snapshots. We key
            # these by the experts module qname (not the parameter name);
            # measure_quant_cost looks for the same input regardless of
            # which packed parameter is being measured.
            for experts_qname, snaps in self._packed_act_snaps.items():
                if not snaps:
                    continue
                X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", experts_qname) + ".pt"
                torch.save({"inputs": X, "name": experts_qname},
                           self.cache_dir / fname)

        # Write per-layer Fisher H detail files (the full per-weight
        # diagonal for Linears, per-expert per-output-channel for packed
        # experts). The probe pickle stores scalar summaries + the
        # detail-file path; measure_quant_cost loads them lazily so the
        # full allocator cost model can run without bloating probe.pkl.
        if self.h_detail_dir is not None:
            self.h_detail_dir.mkdir(parents=True, exist_ok=True)
            sub = re.compile(r"[^A-Za-z0-9_-]")
            for name, acc in self._h_full.items():
                if name not in self.stats:
                    continue
                # Same denominator as the scalar trace: the GLOBAL calib
                # token count (v4). A v3-era revision divided by this
                # row's own n_tokens_seen — per-ROUTED-token for unpacked
                # expert Linears, (global/routed)× hotter than the scalar
                # — and a pre-v3 revision additionally divided expert
                # entries by route_prob (audit M4). h-detail dirs from
                # either era are in different units on expert rows and
                # must be regenerated.
                #
                # Per-token gradient² (g²_t) — concatenate the chunk vectors
                # collected during the hook. This is the per-token Fisher
                # weight, used by archived/research Fisher-weighted GPTQ
                # reconstruction tools: H = X^T diag(g²_t) X.
                ptl = self._per_token_grad_norm.get(name, [])
                g2_per_token = (torch.cat(ptl, dim=0) if ptl
                                else torch.empty(0, dtype=torch.float32))
                fname = sub.sub("__", name) + ".pt"
                torch.save(
                    h_detail_blob(acc, global_tokens, name, kind="linear",
                                  g2_per_token=g2_per_token),
                    self.h_detail_dir / fname)
                self.stats[name]["h_detail_path"] = fname
            for full_name, ch in self._h_packed_channel.items():
                if full_name not in self.stats:
                    continue
                # Packed experts don't carry a router_path — routing is
                # baked into the Fisher signal via how often each expert
                # was selected. Normalize by the global token count, same
                # as the scalar. (The channel accumulator may be
                # device-resident; h_detail_blob lands it on CPU for the
                # saved blob.)
                fname = sub.sub("__", full_name) + ".pt"
                torch.save(h_detail_blob(ch, global_tokens, full_name,
                                         kind="packed"),
                           self.h_detail_dir / fname)
                self.stats[full_name]["h_detail_path"] = fname

    def remove_hooks(self):
        for h in self._fwd_handles + self._bwd_handles + self._moe_block_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()
        self._moe_block_handles.clear()


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def load_calibration(tokenizer, source: str, n_samples: int,
                     seqlen: int, *, calib_seed: int = 42) -> torch.Tensor:
    """Load calibration from a HuggingFace dataset id, a local .jsonl, or
    a local .txt file. JSONL rows can have either {"text": ...} or
    {"messages": [...]} for chat-style data.
    """
    import os

    # A local-file source that does not exist must NOT fall through to the
    # HuggingFace loader. It used to: the `.jsonl`/`.txt` branches were guarded
    # by os.path.exists, so a missing file skipped to the generic `else` and
    # HF reported `Dataset '/home/.../diverse-v1.jsonl' doesn't exist on the
    # Hub` -- an error that names a filesystem path as a dataset id and sends
    # the reader looking for something to download. Nothing is downloadable;
    # the file is simply absent.
    if (source.endswith((".jsonl", ".txt")) or os.sep in source) \
            and not os.path.exists(source):
        raise FileNotFoundError(
            f"calibration file not found: {source}\n"
            "This is a local path, not a HuggingFace dataset id, so there is "
            "nothing to download. Either:\n"
            "  * build the default corpus:  python tools/"
            "build_diverse_calibration.py --output <path> --tokenizer <model>\n"
            "  * point DATASET at your own .jsonl ({\"text\": ...} or "
            "{\"messages\": [...]} rows) or .txt (one sample per line)\n"
            "  * point DATASET at a HuggingFace dataset id "
            "(e.g. ultrachat_200k)"
        )

    texts: list[str] = []
    if source.endswith(".jsonl") and os.path.exists(source):
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    try:
                        texts.append(tokenizer.apply_chat_template(
                            obj["messages"], tokenize=False))
                    except Exception:
                        # Tokenizer has no chat template (base / pt models
                        # like DSv4-Flash-Base). Fall back to concatenating
                        # the message contents — good enough as calibration
                        # text, mirrors the ultrachat fallback below.
                        parts = []
                        for m in obj["messages"]:
                            if isinstance(m, dict):
                                c = m.get("content")
                                if isinstance(c, str) and c:
                                    parts.append(c)
                        if parts:
                            texts.append("\n\n".join(parts))
                elif "text" in obj:
                    texts.append(obj["text"])
    elif source.endswith(".txt") and os.path.exists(source):
        with open(source) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif source == "ultrachat_200k":
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            except Exception:
                # Tokenizer has no chat template (base / pt models).
                # Join plain message contents — good enough as calibration.
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content")
                        if isinstance(c, str) and c:
                            parts.append(c)
                if parts:
                    texts.append("\n\n".join(parts))
            if len(texts) >= n_samples * 8:
                break
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading HuggingFace calibration datasets requires the "
                "'datasets' package. Local .jsonl and .txt calibration files "
                "do not require it."
            ) from exc
        # Generic HF dataset loader. Handles three common schemas:
        #   1. {"text": "..."} — raw text corpora (pile, wikitext, etc.)
        #   2. {"messages": [...]} — chat-format SFT (ultrachat, tulu-3, etc.)
        #   3. anything else — falls back to first string column
        # Streaming when possible so we don't download the full dataset for
        # just 32 samples.
        try:
            ds = load_dataset(source, split="train", streaming=True)
            stream = True
        except Exception:
            ds = load_dataset(source, split="train")
            stream = False

        # Probe one row to detect schema
        iterator = iter(ds) if stream else ds
        first = next(iterator) if stream else (ds[0] if len(ds) else {})
        schema = None
        if "messages" in first:
            schema = "messages"
        elif "text" in first:
            schema = "text"
        else:
            # pick first string-valued column
            for k, v in first.items():
                if isinstance(v, str):
                    schema = k
                    break
        if schema is None:
            raise ValueError(f"Could not find text or messages field in {source}")
        print(f"[probe] {source} schema: {schema}", flush=True)

        # Re-iterate (we consumed the first row)
        if stream:
            ds = load_dataset(source, split="train", streaming=True)
            iterator = iter(ds)
        else:
            iterator = iter(ds)

        for row in iterator:
            if schema == "messages":
                msgs = row.get("messages") or row.get("conversations") or []
                if not msgs:
                    continue
                try:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                except Exception:
                    # Tokenizer has no chat template (base / pt models)
                    # or the template rejects this message shape.
                    # Fall back to concatenating the message contents —
                    # good enough as calibration text, decouples the
                    # probe from the tokenizer's chat-template state.
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get("content")
                            if isinstance(c, str) and c:
                                parts.append(c)
                    if parts:
                        texts.append("\n\n".join(parts))
            else:
                v = row.get(schema)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
            if len(texts) >= n_samples * 8:
                break

    # Two-pass sampling:
    #   1) first pass picks any sample already >= seqlen tokens
    #   2) fallback packs multiple short samples together (separated by
    #      EOS) to reach seqlen. This makes SFT/chat datasets with short
    #      turns (tulu-3, glaive) usable without lowering seqlen.
    # calib_seed controls (a) which subset of dataset texts is tried (shuffle)
    # and (b) where within each text the seqlen window starts. Default 42 is
    # the historical value and is byte-stable.
    if calib_seed != 42:
        rng_shuffle = random.Random(calib_seed)
        rng_shuffle.shuffle(texts)
    random.seed(calib_seed)
    samples = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        start = random.randint(0, ids.size(1) - seqlen)
        samples.append(ids[0, start:start + seqlen])
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # Pack short samples by concatenating with EOS separator
        buf: list[int] = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids[0].tolist()
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= seqlen and len(samples) < n_samples:
                samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
                buf = buf[seqlen:]
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        print(f"[probe] warning: only got {len(samples)}/{n_samples} samples "
              f"(even with packing). Consider wider corpus.",
              flush=True)
    return torch.stack(samples[:n_samples], dim=0)


def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), reduction="none")
    return ce.view(shift_labels.size())


def load_probe_model_and_tokenizer(model_path: str,
                                   requested_device: str,
                                   dtype: torch.dtype,
                                   device_map: str | None = None,
                                   gradient_checkpointing: bool = True,
                                   offload_folder: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged = stage_text_only(model_path)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    load_device_map = device_map if device_map is not None else requested_device

    from_pretrained_kwargs = {
        "torch_dtype": dtype,
        "device_map": load_device_map,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }
    if offload_folder is not None:
        import os as _os
        _os.makedirs(offload_folder, exist_ok=True)
        from_pretrained_kwargs["offload_folder"] = offload_folder
        from_pretrained_kwargs["offload_buffers"] = True
        # low_cpu_mem_usage is forced True internally when device_map is set;
        # disable the explicit override to let HF pick whatever's correct.
        from_pretrained_kwargs.pop("low_cpu_mem_usage", None)
    model = AutoModelForCausalLM.from_pretrained(staged, **from_pretrained_kwargs)
    model.eval()

    # Packed MoE experts (e.g. Qwen3.5/3.6's 3D `gate_up_proj` /
    # `down_proj`) are sensed natively by FisherAccumulator via
    # install_packed_expert_hooks. No unfuse step needed; auto_round is
    # not a probe-time dependency.

    exec_device = resolve_execution_device(model, requested_device)
    print(f"[probe] execution device: {exec_device} "
          f"(load device_map={load_device_map})", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    return staged, tokenizer, model, exec_device, load_device_map


def run_probe_pass(model: nn.Module,
                   tokenizer,
                   calib: torch.Tensor,
                   model_name: str,
                   dataset_name: str,
                   seqlen: int,
                   dtype_name: str,
                   requested_device: str,
                   load_device_map,
                   exec_device: torch.device,
                   linear_include: str,
                   linear_exclude: str,
                   importance_weighting: bool,
                   activation_cache_dir: str | None,
                   output_path: str,
                   h_detail_dir: str | None = None):
    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and inc.search(n) and not exc.search(n)]
    print(f"[probe] tracking {len(tracked)} Linear layers", flush=True)

    expert_info_all = discover_moe_structure(model)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(model, default=2)
    routers = sorted({r for r, _ in expert_info.values()})
    print(f"[probe] MoE: {len(expert_info)} expert linears, "
          f"{len(routers)} routers, top_k={top_k}", flush=True)
    if len(expert_info) == 0:
        diag_count = 0
        for pname, pmod in model.named_modules():
            for attr in ("experts", "block_sparse_moe_experts",
                         "moe_experts", "expert_layer"):
                child = getattr(pmod, attr, None)
                if child is None or not isinstance(child, nn.Module):
                    continue
                kids = list(child.named_children())
                numkids = [k for k, _ in kids if k.isdigit()]
                print(f"[probe/diag] parent={pname!r} attr={attr!r} "
                      f"container_cls={type(child).__name__} "
                      f"n_children={len(kids)} n_numeric_children={len(numkids)}"
                      f" first_children={[k for k,_ in kids[:5]]}",
                      flush=True)
                diag_count += 1
                if diag_count >= 3:
                    break
            if diag_count >= 3:
                break

    tracker = RouterTracker(model, routers, top_k) if routers else None
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    acc = FisherAccumulator(model, tracked, expert_info, cache_dir,
                            h_detail_dir=detail_dir)

    print(f"[probe] calibration shape: {calib.shape}", flush=True)

    model.train()
    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids = calib[i:i+1].to(exec_device)
        t0 = time.time()
        with torch.no_grad():
            embed = model.get_input_embeddings()(ids)
        embed.requires_grad_(True)
        out = model(inputs_embeds=embed, labels=ids)
        logits = out.logits
        t_fwd += time.time() - t0

        t0 = time.time()
        # Use sum-reduction CE so per-token gradients aggregate without
        # the 1/T factor that mean-reduction introduces. The accumulated
        # ||grad_W||²_F divided by total tokens then gives the per-token
        # empirical Fisher diagonal trace under the standard assumption
        # of independence across token positions.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        # .float() before log_softmax: matches the sibling Fisher CE
        # sites (incremental phase-2, per_token_ce) — bf16 log_softmax
        # loses ~0.4% rel on the CE gradient for no memory win here.
        lp = F.log_softmax(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(), dim=-1)
        gather = -lp.gather(1, shift_labels.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                tok = per_token_ce(logits.detach(), ids).reshape(-1)
                mean = float(tok.mean().item())
            # Importance weights renormalized to mean ~1 so the per-token
            # Fisher units are preserved (the weights only redistribute
            # contributions across token positions, not change the total).
            w = (tok / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).sum()
        else:
            loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        if (i + 1) % 4 == 0 or i == 0:
            n_tok = max(int(gather.numel()), 1)
            mean_loss = float(loss.detach().item()) / n_tok
            print(f"[probe] sample {i+1}/{calib.size(0)} "
                  f"loss={mean_loss:.3f} "
                  f"fwd_avg={t_fwd/(i+1):.2f}s bwd_avg={t_bwd/(i+1):.2f}s",
                  flush=True)

        del out, loss, ids, embed, logits
        acc._saved_inputs.clear()

    # Global calib token count — must equal the meta nsamples×seqlen
    # product below so the allocator's load-time renormalization is
    # idempotent on probes written by this finalize.
    fisher_norm_tokens = int(calib.size(0)) * int(seqlen)
    acc.finalize(tracker, fisher_norm_tokens)
    acc.remove_hooks()
    if tracker is not None:
        tracker.remove_hooks()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(tracker.counts) if tracker else {},
            "router_totals": dict(tracker.total_tokens) if tracker else {},
            "router_active_counts": dict(tracker.active_counts) if tracker else {},
            "expert_route_stats": dict(tracker.route_stats) if tracker else {},
            "expert_info": expert_info,
            "meta": {
                "packed_fisher_estimator": "per_token_v2",
                "model": model_name,
                "dataset": dataset_name,
                "nsamples": calib.size(0),
                "seqlen": seqlen,
                "fisher_norm_tokens": fisher_norm_tokens,
                "dtype": dtype_name,
                "device_map": str(load_device_map),
                "execution_device": str(exec_device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
            },
        }, f)
    print(f"[probe] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2 multimodal visual Fisher probe — non-streaming second pass
# ---------------------------------------------------------------------------
def run_multimodal_visual_probe_pass(
    model_path: str,
    *,
    dataset_name: str,
    n_samples: int,
    max_text_len: int,
    requested_device: str,
    dtype: torch.dtype,
    linear_include: str,
    linear_exclude: str,
    activation_cache_dir: str | None,
    output_path: str,
    h_detail_dir: str | None = None,
) -> bool:
    """Non-streaming multimodal Fisher probe focused on the visual encoder.

    Loads the FULL multimodal model via `AutoModelForCausalLM` (so the
    visual tower materializes alongside the body), runs each
    (pixel_values, input_ids, labels) triple through forward + supervised
    CE backward, and captures per-visual-Linear Fisher via
    `FisherAccumulator`. Activation snapshots go to the same
    `activation_cache_dir` the streaming body path uses, so the cost
    stage and export stage both see visual Linears under their canonical
    recipe names.

    Visual tower on Qwen3.6-35B is ~1 GB BF16 plus ~70 GB body weights —
    fits under a 128 GB budget. For very large VLMs (e.g. 122B) where the
    body alone exceeds RAM, this function catches the OOM / load error,
    logs a warning, returns False, and the caller falls back to the Phase
    1 uniform `--visual-format` override. No partial calibration is
    written on failure.

    Returns True on successful probe completion (output pickle written),
    False on graceful fallback. Raises only on unexpected errors
    (not OOM / resource shortage).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    staged = stage_multimodal(model_path)
    try:
        processor = AutoProcessor.from_pretrained(model_path,
                                                  trust_remote_code=True)
    except Exception:
        try:
            processor = AutoProcessor.from_pretrained(staged,
                                                      trust_remote_code=True)
        except Exception as e:
            print(f"[probe/mm] AutoProcessor.from_pretrained failed "
                  f"({type(e).__name__}: {e}); skipping multimodal pass. "
                  f"Falling back to --visual-format Phase 1 override.",
                  flush=True)
            return False

    triples = load_multimodal_calibration(
        processor, dataset_name, n_samples, max_text_len)
    print(f"[probe/mm] loaded {len(triples)} multimodal samples "
          f"(dataset={dataset_name!r})", flush=True)
    if not triples:
        print("[probe/mm] load_multimodal_calibration returned 0 samples; "
              "skipping multimodal pass", flush=True)
        return False

    # Load the DECLARED arch directly, bypassing AutoModelForCausalLM's
    # silent text-only downgrade path. `AutoModelForCausalLM` resolves
    # Qwen3_5MoeConfig → Qwen3_5MoeForCausalLM (text-only), which
    # matches `config.sub_configs["text_config"]` and triggers
    # transformers/auto_factory.py:132-134 to swap the composite config
    # for just the text sub-config. The visual tower tensors in the
    # safetensors are then silently dropped, and `named_modules()`
    # returns 0 Linears under `model.visual.*`. Use
    # `config.architectures[0]` to instantiate the multimodal class
    # (Qwen3_5MoeForConditionalGeneration), which keeps the visual
    # tower wired.
    model_cls = AutoModelForCausalLM
    try:
        import transformers
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(staged, trust_remote_code=True)
        arch_names = getattr(cfg, "architectures", None) or []
        if arch_names and hasattr(transformers, arch_names[0]):
            model_cls = getattr(transformers, arch_names[0])
    except Exception:
        # Config resolve failure falls through to AutoModel below.
        pass
    try:
        model = model_cls.from_pretrained(
            staged, torch_dtype=dtype, device_map=requested_device,
            low_cpu_mem_usage=False, trust_remote_code=True,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        msg = str(e).lower()
        if ("out of memory" in msg or "oom" in msg
                or isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError))):
            print(f"[probe/mm] whole-model multimodal load OOM "
                  f"({type(e).__name__}); falling back to --visual-format "
                  f"Phase 1 override. On 122B-scale models this is expected.",
                  flush=True)
            return False
        # Other RuntimeError (weight mismatch, etc.) isn't OOM — re-raise.
        raise
    model.eval()
    exec_device = resolve_execution_device(model, requested_device)

    for p in model.parameters():
        p.requires_grad_(False)

    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and inc.search(n) and not exc.search(n)]
    print(f"[probe/mm] tracking {len(tracked)} Linear layers "
          f"(include={linear_include!r})", flush=True)

    expert_info_all = discover_moe_structure(model)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(model, default=2)
    routers = sorted({r for r, _ in expert_info.values()})
    tracker = RouterTracker(model, routers, top_k) if routers else None

    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    acc = FisherAccumulator(model, tracked, expert_info, cache_dir,
                            h_detail_dir=detail_dir)

    model.train()
    t_fwd = t_bwd = 0.0
    for i, sample in enumerate(triples):
        # Move every tensor to exec_device; pixel_values specifically
        # gets cast to `dtype` (bf16 / fp16) because vision backbones
        # are mixed-precision-friendly. Labels stay as long ids.
        kwargs: dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if not isinstance(v, torch.Tensor):
                continue
            target_dtype = dtype if k == "pixel_values" else None
            kwargs[k] = (v.to(exec_device, dtype=target_dtype)
                         if target_dtype is not None
                         else v.to(exec_device))
        labels = kwargs.pop("labels", kwargs.get("input_ids"))
        t0 = time.time()
        try:
            out = model(**kwargs, labels=labels)
        except Exception as e:
            print(f"[probe/mm] sample {i}: forward raised {type(e).__name__}: "
                  f"{e}; skipping", flush=True)
            acc._saved_inputs.clear()
            continue
        logits = out.logits
        t_fwd += time.time() - t0

        t0 = time.time()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # .float() before log_softmax: consistent with the sibling
        # Fisher CE sites (visual phase-2 casts preds to fp32 first).
        lp = F.log_softmax(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(), dim=-1)
        valid = shift_labels.reshape(-1)
        mask = (valid >= 0) & (valid < shift_logits.size(-1))
        if not mask.any():
            acc._saved_inputs.clear()
            continue
        gather = -lp[mask.nonzero(as_tuple=True)[0]].gather(
            1, valid[mask].reshape(-1, 1)).squeeze(1)
        loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        print(f"[probe/mm] sample {i + 1}/{len(triples)} "
              f"loss={float(loss) / max(gather.numel(), 1):.3f} "
              f"fwd_avg={t_fwd / (i + 1):.2f}s "
              f"bwd_avg={t_bwd / (i + 1):.2f}s", flush=True)

        del out, loss, logits
        acc._saved_inputs.clear()

    # Global calib token budget. Multimodal samples vary in real token
    # count, so nsamples×max_text_len is an upper bound — but it is ONE
    # shared constant across every row (relative Fisher is what the
    # knapsack prices) and it matches the meta nsamples×seqlen product
    # the allocator renormalizes by, keeping that recompute idempotent.
    fisher_norm_tokens = max(int(len(triples)) * int(max_text_len), 1)
    acc.finalize(tracker, fisher_norm_tokens)
    acc.remove_hooks()
    if tracker is not None:
        tracker.remove_hooks()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(tracker.counts) if tracker else {},
            "router_totals": dict(tracker.total_tokens) if tracker else {},
            "router_active_counts": dict(tracker.active_counts) if tracker else {},
            "expert_route_stats": dict(tracker.route_stats) if tracker else {},
            "expert_info": expert_info,
            "meta": {
                "packed_fisher_estimator": "per_token_v2",
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": len(triples),
                "seqlen": max_text_len,
                "fisher_norm_tokens": fisher_norm_tokens,
                "dtype": str(dtype),
                "device_map": requested_device,
                "execution_device": str(exec_device),
                "top_k": top_k,
                "importance_weighting": False,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                "calibration_modality": "multimodal",
            },
        }, f)
    print(f"[probe/mm] wrote {out_path}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Streaming multimodal visual Fisher probe — the one incremental path.
# Same intent as `run_multimodal_visual_probe_pass` above, but reuses the
# shared streaming context so the body NEVER loads whole. The visual tower
# (2-3 GB even at 122B scale) stays resident on device; body decoder layers
# install/unload around a handwritten streaming forward+backward. This is
# the path that makes Qwen3.5-122B calibration work under 128 GB.
# ---------------------------------------------------------------------------
def run_streaming_multimodal_visual_probe_pass(
    model_path: str,
    *,
    dataset_name: str,
    n_samples: int,
    max_text_len: int,
    requested_device: str,
    dtype: torch.dtype,
    linear_include: str,
    linear_exclude: str,
    activation_cache_dir: str | None,
    output_path: str,
    offload_folder: str,
    h_detail_dir: str | None = None,
) -> bool:
    """Incremental multimodal Fisher probe.

    Uses `_build_streaming_context(..., multimodal=True,
    visual_requires_grad=True)` to build a skeleton where:
      - Body decoder layers are on meta, streamed layer-by-layer on demand.
      - Visual tower is fully resident on `device`.
      - Head pieces (embed/norm/lm_head/rotary) are resident.

    Forward path (per sample):
      1. `image_features = model.model.get_image_features(pixel_values,
                                                          image_grid_thw)`
         — runs through resident visual, producing per-image patches.
      2. `text_embeds = base.embed_tokens(input_ids)` — resident.
      3. Merge image features into text embeds via the model's
         `get_placeholder_mask` + `masked_scatter`. Result
         `inputs_embeds` has an autograd edge to visual tower weights.
      4. Streaming body forward (Phase 1): install layer L, run
         `_call_layer`, cache activation on CPU, unload.
      5. Phase 2: final norm + lm_head + teacher-forced CE; backward to
         get grad at the final hidden state.
      6. Phase 3: reverse sweep, per-layer install/backward/unload,
         accumulate grad into grad-at-hidden-0 (= grad at inputs_embeds).
      7. `inputs_embeds.backward(grad_at_hidden_0)` — autograd flows
         into visual (resident, requires_grad=True) and fires the
         per-visual-Linear Fisher backward hooks.

    Returns True on success (probe pickle written), False on any failure
    (caller falls back to `--visual-format` Phase 1 override). No partial
    calibration is written on failure.
    """
    from transformers import AutoProcessor

    from .layer_streaming import (
        _call_layer,
        _compute_attention_mask,
        _compute_position_embeddings,
    )
    from .model_profiles import DefaultProfile, profile_from_model
    from .streaming_model import _build_streaming_context

    try:
        processor = AutoProcessor.from_pretrained(model_path,
                                                  trust_remote_code=True)
    except Exception:
        try:
            staged = stage_multimodal(model_path)
            processor = AutoProcessor.from_pretrained(staged,
                                                     trust_remote_code=True)
        except Exception as e:
            print(f"[probe/mm-stream] AutoProcessor.from_pretrained failed "
                  f"({type(e).__name__}: {e}); skipping streaming multimodal "
                  f"pass.", flush=True)
            return False

    triples = load_multimodal_calibration(
        processor, dataset_name, n_samples, max_text_len)
    print(f"[probe/mm-stream] loaded {len(triples)} multimodal samples "
          f"(dataset={dataset_name!r})", flush=True)
    if not triples:
        print("[probe/mm-stream] no calibration samples; skipping", flush=True)
        return False

    device = torch.device(requested_device)
    try:
        ctx = _build_streaming_context(
            model_path,
            device=device, dtype=dtype,
            offload_folder=offload_folder,
            log_prefix="[probe/mm-stream]",
            multimodal=True,
            visual_requires_grad=True,
        )
    except Exception as e:
        print(f"[probe/mm-stream] streaming context build failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return False

    if ctx.visual_module is None:
        print(f"[probe/mm-stream] no visual tower detected on model; "
              f"skipping", flush=True)
        ctx.shutdown()
        return False

    model = ctx.model
    base_model = ctx.base_model
    layers = ctx.layers
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix
    visual_module = ctx.visual_module
    visual_prefix = ctx.visual_prefix or ""
    try:
        model_profile = profile_from_model(model)
    except Exception:
        model_profile = DefaultProfile()

    # Enumerate tracked Linears: per-regex visual matches + any resident
    # Linears that the regex would pick up (usually none — visual regex
    # is scoped to visual.*). Body Linears live on meta during streaming;
    # FisherAccumulator would see meta weights and stats would be garbage.
    # Scope strictly to visual for this path.
    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    tracked: list[str] = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if not inc.search(n) or exc.search(n):
            continue
        # Skip body decoder-layer Linears — they're on meta during
        # streaming; FisherAccumulator can't handle meta weights.
        if any(n.startswith(f"{layers_prefix}{L}.") for L in range(num_layers)):
            continue
        # Check the weight is really resident.
        w = getattr(m, "weight", None)
        if w is None or w.is_meta:
            continue
        tracked.append(n)
    print(f"[probe/mm-stream] tracking {len(tracked)} resident Linears "
          f"(include={linear_include!r})", flush=True)
    if not tracked:
        print(f"[probe/mm-stream] no resident Linears match "
              f"{linear_include!r}; writing empty pickle", flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "stats": {},
                "router_counts": {},
                "router_totals": {},
                "expert_info": {},
                "meta": {
                    "model": model_path,
                    "dataset": dataset_name,
                    "nsamples": len(triples),
                    "seqlen": max_text_len,
                    "dtype": str(dtype),
                    "device_map": requested_device,
                    "execution_device": str(device),
                    "top_k": read_top_k(model, default=2),
                    "importance_weighting": False,
                    "activation_cache_dir": activation_cache_dir,
                    "linear_include": linear_include,
                    "linear_exclude": linear_exclude,
                    "calibration_modality": "multimodal",
                    "streaming": True,
                },
            }, f)
        ctx.shutdown()
        return True

    expert_info_all = discover_moe_structure(model)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(model, default=2)
    routers = sorted({r for r, _ in expert_info.values()})
    tracker = RouterTracker(model, routers, top_k) if routers else None

    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    acc = FisherAccumulator(model, tracked, expert_info, cache_dir,
                            h_detail_dir=detail_dir)

    # Enable train mode on the visual tower so any dropout/BN layers behave
    # consistently; rest of the model stays eval (and params.requires_grad
    # are set up correctly already).
    visual_module.train(False)  # no dropout — Fisher estimator wants eval

    # Derive an 'mm_model' reference for the declared-arch helpers
    # (get_image_features, get_placeholder_mask). `model.model` is the
    # Qwen3_5MoeModel-style wrapper for Qwen3.5/3.6; `model` itself can
    # be the ForConditionalGeneration class which forwards these calls.
    mm_model = getattr(model, "model", model)
    get_image_features = getattr(mm_model, "get_image_features", None)
    get_placeholder_mask = getattr(mm_model, "get_placeholder_mask", None)
    if get_image_features is None or get_placeholder_mask is None:
        # Fallback: some architectures expose these on the outer model.
        get_image_features = getattr(model, "get_image_features",
                                     get_image_features)
        get_placeholder_mask = getattr(model, "get_placeholder_mask",
                                       get_placeholder_mask)
    if get_image_features is None:
        print(f"[probe/mm-stream] model has no get_image_features; "
              f"aborting streaming multimodal probe", flush=True)
        ctx.shutdown()
        return False

    prefetch_depth = 3
    total_fwd = total_bwd = 0.0
    successes = 0
    # KV-cotangent path (on by default): route the Fisher cotangent of borrowed
    # cross-layer state back to the layer that produced it. `=0` restores the
    # severed-cotangent behavior for an A/B, and makes the KV-sharing Fisher
    # guard fail loud again (`incremental_probe.kv_shared_fisher_block_reason`).
    kv_cotangent_path = kv_cotangent_path_enabled()

    for i, sample in enumerate(triples):
        # Move inputs to device, casting pixel_values to dtype.
        kwargs: dict[str, torch.Tensor] = {}
        for k, v in sample.items():
            if not isinstance(v, torch.Tensor):
                continue
            target_dtype = dtype if k == "pixel_values" else None
            kwargs[k] = (v.to(device, dtype=target_dtype)
                         if target_dtype is not None else v.to(device))
        labels = kwargs.pop("labels", kwargs.get("input_ids")).to(device)
        pixel_values = kwargs.get("pixel_values")
        input_ids = kwargs.get("input_ids")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None or input_ids is None:
            continue

        t0 = time.time()
        try:
            # ---- Visual forward -----------------------------------------
            # Keep autograd ON so gradients flow back into visual tower.
            vision_output = get_image_features(
                pixel_values, image_grid_thw=image_grid_thw)
            # vision_output shape / dtype handling mirrors
            # Qwen3_5MoeModel.forward: pooler_output → concatenated
            # image_embeds, aligned to inputs_embeds dtype/device.
            image_embeds = getattr(vision_output, "pooler_output",
                                   vision_output)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(list(image_embeds), dim=0)

            # ---- Text embeds + merge ------------------------------------
            text_embeds = base_model.embed_tokens(input_ids).to(dtype)
            image_embeds = image_embeds.to(text_embeds.device,
                                           dtype=text_embeds.dtype)
            try:
                image_mask, _vmask = get_placeholder_mask(
                    input_ids, inputs_embeds=text_embeds,
                    image_features=image_embeds)
                inputs_embeds = text_embeds.masked_scatter(
                    image_mask, image_embeds)
            except Exception:
                # Processor didn't put image tokens in input_ids, or
                # shape mismatch. Skip merge — body will still see text
                # embeds; visual grad flows via a reconstruction path.
                # Use a fallback that sums visual features into the
                # first-image-token position, preserving a grad edge.
                inputs_embeds = text_embeds + 0.0 * image_embeds.sum()

            # ---- Streaming body forward (Phase 1) ----------------------
            # Save per-layer CPU activations for the reverse sweep.
            T = inputs_embeds.size(1)
            position_ids = torch.arange(T, device=device).unsqueeze(0)
            causal_mask = _compute_attention_mask(
                base_model, inputs_embeds, position_ids)
            position_embeddings = _compute_position_embeddings(
                base_model, inputs_embeds, position_ids, model_profile)
            pass_state = model_profile.new_forward_pass_state()

            activations_cpu: list[torch.Tensor] = [inputs_embeds.detach().cpu()]
            hidden = inputs_embeds
            for d in range(prefetch_depth):
                ctx.schedule_prefetch(d)
            for L in range(num_layers):
                ctx.install(L)
                ctx.schedule_prefetch(L + prefetch_depth)
                with torch.no_grad():
                    out = _call_layer(
                        layers[L], hidden,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        **_streaming_visual_layer_kwargs(
                            model_profile,
                            input_ids=input_ids,
                            pass_state=pass_state,
                        ),
                    )
                hidden = out
                activations_cpu.append(hidden.detach().cpu())
                ctx.unload(L)
            shared_pass_state = model_profile.capture_forward_pass_state(
                pass_state)

            # ---- Phase 2: final norm + lm_head + CE --------------------
            final_hidden = (activations_cpu[-1].to(device).to(dtype)
                            .requires_grad_(True))
            norm_out = base_model.norm(final_hidden)
            preds = model.lm_head(norm_out).float()
            # Teacher-forced CE: predict token t+1 from hidden at t.
            shift_preds = preds[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lp = F.log_softmax(
                shift_preds.reshape(-1, shift_preds.size(-1)), dim=-1)
            valid = shift_labels.reshape(-1)
            mask = (valid >= 0) & (valid < shift_preds.size(-1))
            if not mask.any():
                # No valid targets; skip this sample.
                ctx.layer_cache.clear()
                acc._saved_inputs.clear()
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            gather = -lp.index_select(0, idx).gather(
                1, valid[mask].reshape(-1, 1)).squeeze(1)
            loss = gather.sum()
            loss.backward()
            grad_at_tail = final_hidden.grad.detach().clone()
            del final_hidden, norm_out, preds, lp, gather, loss

            # ---- Phase 3: streaming reverse sweep ----------------------
            grad_out = grad_at_tail
            # KV-cotangent path, scoped to THIS sample's sweep: a fresh
            # accumulator per pass, exactly like the shared state it mirrors,
            # so sample N never seeds sample N-1's cotangent.
            kv_cotangents = SharedStateCotangents(enabled=kv_cotangent_path)
            for d in range(prefetch_depth):
                ctx.schedule_prefetch(num_layers - 1 - d)
            for L in reversed(range(num_layers)):
                ctx.install(L)
                ctx.schedule_prefetch(L - prefetch_depth)
                x_in = (activations_cpu[L].to(device).to(dtype)
                        .detach().requires_grad_(True))
                out = _call_layer(
                    layers[L], x_in,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    **_streaming_visual_layer_kwargs(
                        model_profile,
                        input_ids=input_ids,
                        captured_pass_state=shared_pass_state,
                        layer=layers[L],
                        cotangents=kv_cotangents,
                    ),
                )
                # Drive the backward with this layer's output cotangent AND
                # the cotangent its consumers accumulated on the shared state
                # it produced. One backward, so each Linear's hook still fires
                # once with the summed gy.
                kv_roots, kv_grads = kv_cotangents.produced_roots()
                if kv_roots:
                    torch.autograd.backward([out, *kv_roots],
                                            [grad_out, *kv_grads])
                else:
                    out.backward(grad_out)
                kv_cotangents.harvest()
                grad_out = x_in.grad.detach().clone()
                ctx.unload(L)
                del x_in, out
            if kv_cotangents.pending_keys():
                print(f"[probe/mm-stream] sample {i}: "
                      f"{kv_cotangents.summary()}", flush=True)

            total_fwd += (time.time() - t0)

            # ---- Final: push grad into visual via inputs_embeds --------
            t1 = time.time()
            # grad_out is now grad at inputs_embeds (hidden[0]). Apply
            # backward on the autograd-connected inputs_embeds (resident
            # on device) so visual Fisher hooks fire.
            inputs_embeds.backward(grad_out.to(device))
            total_bwd += (time.time() - t1)

            successes += 1

        except Exception as e:
            print(f"[probe/mm-stream] sample {i}: "
                  f"{type(e).__name__}: {e}", flush=True)
            ctx.layer_cache.clear()
            acc._saved_inputs.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        finally:
            ctx.layer_cache.clear()
            acc._saved_inputs.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if (i + 1) % 1 == 0 or i == 0:
            print(f"[probe/mm-stream] sample {i + 1}/{len(triples)} "
                  f"fwd_avg={total_fwd / max(successes, 1):.2f}s "
                  f"bwd_avg={total_bwd / max(successes, 1):.2f}s", flush=True)

    # Same convention as the non-streaming multimodal pass: an upper-bound
    # but SHARED global token constant, equal to the meta nsamples×seqlen
    # product so the allocator's renormalization is idempotent.
    fisher_norm_tokens = max(int(successes) * int(max_text_len), 1)
    acc.finalize(tracker, fisher_norm_tokens)
    acc.remove_hooks()
    if tracker is not None:
        tracker.remove_hooks()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(tracker.counts) if tracker else {},
            "router_totals": dict(tracker.total_tokens) if tracker else {},
            "router_active_counts": dict(tracker.active_counts) if tracker else {},
            "expert_route_stats": dict(tracker.route_stats) if tracker else {},
            "expert_info": expert_info,
            "meta": {
                "packed_fisher_estimator": "per_token_v2",
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": successes,
                "seqlen": max_text_len,
                "fisher_norm_tokens": fisher_norm_tokens,
                "dtype": str(dtype),
                "device_map": requested_device,
                "execution_device": str(device),
                "top_k": top_k,
                "importance_weighting": False,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                "calibration_modality": "multimodal",
                "streaming": True,
            },
        }, f)
    print(f"[probe/mm-stream] wrote {out_path} "
          f"({successes}/{len(triples)} samples succeeded)", flush=True)
    ctx.shutdown()
    return successes > 0
