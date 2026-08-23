#!/usr/bin/env python3
"""incremental_probe.py — PrismaQuant sensitivity probe, streamed shard-by-shard.

This is the unified probe path. There is no separate "whole model fits in
RAM" branch: the model is always loaded via the layer-streaming primitives
in `layer_streaming`, with the head (embed + norm + lm_head + rotary)
resident and decoder layers offloaded to disk and streamed in on demand.
Small models just pay the no-op cost of a LayerCache that can hold every
layer resident; large models drain the cache to disk as needed.

Each shard (body layer range, MTP, lm_head) runs one streaming pass: the
exact phase-1 / phase-2 / phase-3 flow from `streaming_probe.run_streaming_probe`,
specialized to Fisher-instrument only the Linears matching that shard's
regex. MTP is a built-in shard kind: after the body forward we ask the
model profile to build its MTP module (`profile.build_mtp_module`), load
the source MTP weights straight from safetensors
(`profile.read_mtp_source_state_dict` / `profile.load_mtp_state_dict`),
and run its own forward+backward for Fisher collection. The per-shard pickle
output format matches `sensitivity_probe.run_probe_pass` / `streaming_probe`
unchanged — the allocator consumes either. The two backends also agree on
the estimator and normalization conventions: per-token-summed empirical
Fisher (Σ_t ‖∇_t‖², including packed experts via the F.linear
interception in `install_packed_expert_hooks`), divided by the GLOBAL
calibration token count for every row -- dense and MoE expert alike.
(This paragraph used to say "the tokens each entry actually saw (routed
tokens for MoE experts)". That was audit M4's convention and PR #14
reversed it: a per-routed-token denominator inflates a rarely-routed
expert by global/routed, which is inverted importance weighting for the
mean-Δloss objective. `finalize_fisher_stats` carries the derivation.)

The one quantity that is still per-routed-token is the AQUA A-side's
activation VARIANCE fit, and deliberately so: `expert_act_sq_sum` is
divided by `expert_tokens[e]` because it models the per-token noise
magnitude of the rows that flow through expert e, not that expert's
share of the objective. See `_accumulate_packed_per_token_fisher`.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import pickle
import re
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

from prismaquant.incremental_shards import (
    annotate_incremental_shard as annotate_probe_shard,
    read_pickle as _read_pickle,
)

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch


# v26: central runtime-flag helper. Each named env var defaults to the
# given value (typically True for performance flags whose math is
# equivalent to the legacy path). Set the env var to "0" to disable.
# This replaces the proliferating `os.environ.get(NAME) == "1"`
# pattern that left every perf flag opt-in indefinitely.
def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False", "FALSE", "no", "NO")


# ---- Per-channel Fisher marginals -------------------------------------
# The per-element diagonal H[o,i] = Σ_t g[t,o]²·x[t,i]² is unstorable at
# scale (47k Linears × 17 MB ≈ 800 GB — the reason only two scalars
# survive today). Its two MARGINALS are storable, and they are what a
# per-channel sensitivity contract actually needs:
#
#   fisher_row[o] = Σ_i H[o,i]      fisher_col[i] = Σ_o H[o,i]
#
# plus the two pure factors (g_sq_sum, act_sq_sum) that let a consumer
# separate "this output channel has hot gradients" from "this input
# channel has hot activations", and act_absmax for clipping decisions.
#
# The row/col marginals are read off the `chunk_h` every accumulation
# site already materializes, so they cost no extra matmul AND satisfy
# sum(fisher_row) == sum(fisher_col) == chunk_h.sum() by construction —
# that identity is the wiring check (tests/test_probe_marginals.py).
_MARGINAL_KEYS = (
    "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax")
# act_absmax is a BOUND, not a total: it merges by elementwise maximum
# across chunks/shards. Summing it would inflate it without bound.
_MARGINAL_MAX_KEYS = frozenset({"act_absmax"})

# The packed-expert (AQUA) counterparts. Separate from `_MARGINAL_KEYS`
# because these are [E, *] per-expert arrays produced by the F.linear
# interception, not the 1-D per-Linear vectors the dense backward hook
# flushes through `_marginal_flush`.
_PACKED_MARGINAL_KEYS = (
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
    "expert_tokens")


def _marginals_enabled() -> bool:
    return _env_flag("PRISMAQUANT_PROBE_MARGINALS", default=True)


def _marginal_chunk(gy2_sq: torch.Tensor, x2_sq: torch.Tensor,
                    x2: torch.Tensor,
                    chunk_h: torch.Tensor) -> list[torch.Tensor]:
    """Five per-channel reductions of one (gy², x², H) chunk, in
    `_MARGINAL_KEYS` order, device-resident fp32.

    Reductions force fp32 accumulation: the inputs are bf16 and a
    T-long running sum in bf16 loses real precision for free.
    act_absmax comes off `x2` directly via amax/amin rather than
    sqrt(x2_sq.amax) — same value, but exact in the input dtype and
    without materializing a [T, in] abs() copy on the hot path.
    """
    if x2.size(0) == 0:
        # A routed expert can be handed zero tokens; the sums are all
        # zero anyway but amax/amin raise on an empty reduction dim.
        absmax = torch.zeros(x2.size(1), dtype=torch.float32,
                             device=x2.device)
    else:
        absmax = torch.maximum(x2.amax(dim=0).abs(),
                               x2.amin(dim=0).abs()).to(torch.float32)
    return [
        chunk_h.sum(dim=1, dtype=torch.float32),
        chunk_h.sum(dim=0, dtype=torch.float32),
        gy2_sq.sum(dim=0, dtype=torch.float32),
        x2_sq.sum(dim=0, dtype=torch.float32),
        absmax,
    ]


def _marginal_accumulate(slot: dict, name: str,
                         vecs: list[torch.Tensor]) -> None:
    """Fold one chunk's marginals into `slot[name]`, staying
    DEVICE-RESIDENT. The v21 #1 optimization batches every device→host
    scalar transfer to one sync per layer; a `.cpu()` here would put
    ~94k syncs back on the backward hot path."""
    cur = slot.get(name)
    if cur is None:
        slot[name] = [v.detach().clone() for v in vecs]
        return
    for key, c, v in zip(_MARGINAL_KEYS, cur, vecs):
        if key in _MARGINAL_MAX_KEYS:
            torch.maximum(c, v, out=c)
        else:
            c.add_(v)


def merge_marginals(dst: dict, src) -> None:
    """Fold per-channel marginals from `src` into `dst` in place.

    Sums add elementwise; act_absmax merges by MAXIMUM. Used both for
    the per-layer host flush and for the cross-shard partial-stats
    merge, so the two cannot drift apart on the max-vs-sum rule.
    """
    for key in _MARGINAL_KEYS:
        new = src.get(key)
        if new is None:
            continue
        new = np.asarray(new, dtype=np.float32)
        old = dst.get(key)
        if old is None:
            dst[key] = new.copy()
        elif key in _MARGINAL_MAX_KEYS:
            dst[key] = np.maximum(old, new)
        else:
            dst[key] = old + new


def _marginal_flush(device_slot: dict, stats: dict) -> None:
    """Drain device-resident marginal accumulators into `stats` as
    numpy fp32, using ONE device→host transfer for the whole layer:
    every vector is concatenated flat, copied once, then sliced. Same
    discipline as the h_trace/h_w2_sum stack above it."""
    if not device_slot:
        return
    names = list(device_slot.keys())
    host = torch.cat(
        [v.reshape(-1) for n in names for v in device_slot[n]]).cpu()
    off = 0
    for n in names:
        payload = {}
        for key, v in zip(_MARGINAL_KEYS, device_slot[n]):
            ln = v.numel()
            # .copy() so the per-Linear arrays do not each pin the one
            # big host buffer alive.
            payload[key] = host[off:off + ln].numpy().copy()
            off += ln
        entry = stats.get(n)
        if entry is not None:
            merge_marginals(entry, payload)
    device_slot.clear()


def _marginal_zeros(out_features: int, in_features: int) -> dict:
    """Zero-initialized marginal keys for a stats entry. Zeros are the
    identity for both merge rules (sum and max over |x| ≥ 0), so a
    Linear whose hook never fires ships zeros rather than missing keys."""
    return {
        "fisher_row": np.zeros(int(out_features), dtype=np.float32),
        "fisher_col": np.zeros(int(in_features), dtype=np.float32),
        "g_sq_sum": np.zeros(int(out_features), dtype=np.float32),
        "act_sq_sum": np.zeros(int(in_features), dtype=np.float32),
        "act_absmax": np.zeros(int(in_features), dtype=np.float32),
    }


import torch.nn as nn
import torch.nn.functional as F

from .layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)
from .perturbed_x_cache import calibration_data_hash
from .sensitivity_probe import (
    FisherAccumulator,
    RouterTracker,
    SharedStateCotangents,
    discover_moe_structure,
    discover_moe_routers,
    finalize_fisher_stats,
    grouped_linear_fisher_chunk,
    grouped_linear_groups,
    grouped_linear_stats_entry,
    h_detail_blob,
    install_packed_expert_hooks,
    kv_cotangent_path_enabled,
    load_calibration,
    per_token_ce,
    read_top_k,
    run_multimodal_visual_probe_pass,
    run_streaming_multimodal_visual_probe_pass,
    stage_multimodal,
    stage_text_only,
)
from .streaming_model import (
    StreamingContext,
    _build_streaming_context,
    _classify_shard,
)


# ---------------------------------------------------------------------------
# ModuleList-of-experts fast MoE replay (MiniMax-M2 is the motivating arch)
# ---------------------------------------------------------------------------
# HF MiniMax-M2 represents the 256 experts as a ModuleList and its
# `MiniMaxM2Experts.forward` loops over every hit expert in Python:
#   torch.where(...) -> expert MLP -> index_add_
# With 4 x 256 tokens and top-k=8, almost every expert is hit, so one
# layer replay issues ~256 tiny expert MLPs. The GPU stays mostly idle
# while CPU burns time launching thousands of small ops.
#
# During Phase-3 only the shard's target layers need nn.Linear hooks.
# Non-target layers merely propagate grad_out backward to earlier
# activations, so we can replace the ModuleList loop with chunked batched
# expert matmuls for those layers. Target layers keep the original module
# path so per-expert Linear hooks still fire exactly as before.
# ---------------------------------------------------------------------------


def _is_unpacked_experts_module(
    module: nn.Module,
    proj_names: tuple[str, ...] = ("w1", "w2", "w3"),
    class_names: frozenset[str] | tuple[str, ...] = (),
) -> bool:
    """Recognize a ModuleList-style expert container the fast replay can swap.

    Two conditions, both required:

      - the container class is one the *profile* declares
        (`packed_expert_module_class_names()` -> the spec's
        `packed_experts.module_class_names`, `base.py:182-192`). This used to
        be the literal string `"MiniMaxM2Experts"` in this file. It cannot be
        dropped in favour of pure structure: the replacement forward
        (`_minimax_fast_experts_forward`) implements one specific expert-loop
        signature, so applying it to a container that merely *looks* similar
        would silently change a forward pass. Declaring the class is the
        architecture opting in.
      - the container really has the ModuleList-of-experts shape the replay
        needs: `num_experts`/`top_k`, indexable, and a first expert carrying
        the profile's per-expert projection attributes plus `act_fn`. Packed
        (3D-parameter) expert containers declare a class name too and fail
        here, which is correct — they are not what this path replays.

    A profile that declares no container class keeps today's behaviour: no
    swap, per-Linear hooks only. That is a probe-speed loss, not a
    correctness one.
    """
    if not class_names or type(module).__name__ not in set(class_names):
        return False
    try:
        return (
            hasattr(module, "num_experts")
            and hasattr(module, "top_k")
            and len(module) > 0
            and all(hasattr(module[0], n) for n in (*proj_names, "act_fn"))
        )
    except (TypeError, KeyError, IndexError):
        # Not indexable / not list-like: the swap does not apply.
        return False


def _minimax_fast_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    original = getattr(self, "_pq_original_forward")
    if not getattr(self, "_pq_fast_moe_enabled", False):
        return original(hidden_states, top_k_index, top_k_weights)

    if hidden_states.numel() == 0:
        return torch.zeros_like(hidden_states)

    device = hidden_states.device
    n_tokens, hidden_dim = hidden_states.shape
    top_k = int(top_k_index.shape[-1])
    n_experts = int(self.num_experts)
    chunk_size = max(1, int(getattr(self, "_pq_fast_moe_chunk_size", 32)))

    flat_experts = top_k_index.reshape(-1).to(torch.long)
    flat_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    token_ids = torch.arange(n_tokens, device=device).repeat_interleave(top_k)

    order = torch.argsort(flat_experts)
    experts_sorted = flat_experts.index_select(0, order)
    tokens_sorted = token_ids.index_select(0, order)
    weights_sorted = flat_weights.index_select(0, order)

    counts = torch.bincount(experts_sorted, minlength=n_experts)
    active = torch.nonzero(counts, as_tuple=False).flatten()
    if active.numel() == 0:
        return torch.zeros_like(hidden_states)

    offsets = torch.empty(n_experts + 1, device=device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)

    final_hidden_states = torch.zeros_like(hidden_states)
    act_fn = self[0].act_fn

    # v22 Fix E2: hoist all per-chunk syncs into ONE batched device→host
    # transfer at the start of the function. The original code did 4-5
    # `.item()` / `.tolist()` calls inside the loop body, each of which
    # blocks the GPU until the prior kernel finishes. With ~8 chunks per
    # layer × ~50 MoE layers per phase-1, that's ~2000 host syncs
    # serializing GPU work. Now we precompute per-chunk metadata in
    # device tensors, do one .cpu() at the top, then loop using host
    # data only — no in-loop syncs.
    chunk_list = list(active.split(chunk_size))
    n_chunks = len(chunk_list)
    if n_chunks == 0:
        return final_hidden_states

    # Per-chunk metadata: (start, end, max_count) packed into a single
    # (n_chunks, 3) device tensor.
    # start_dev[i] = offsets[chunk_list[i][0]]
    # end_dev[i]   = offsets[chunk_list[i][-1] + 1]
    # max_count_dev[i] = max(counts[expert] for expert in chunk_list[i])
    chunk_first = torch.stack([c[0] for c in chunk_list])
    chunk_last_p1 = torch.stack([c[-1] + 1 for c in chunk_list])
    starts_dev = offsets.index_select(0, chunk_first)
    ends_dev = offsets.index_select(0, chunk_last_p1)
    # Per-chunk max via bincount + max — vectorized on device.
    # Build a chunk-id-per-active-expert tensor, then segment max.
    chunk_lengths = torch.tensor(
        [c.numel() for c in chunk_list], device=device, dtype=torch.long)
    chunk_id_per_active = torch.repeat_interleave(
        torch.arange(n_chunks, device=device), chunk_lengths)
    counts_active = counts.index_select(0, active)
    max_counts_dev = torch.full((n_chunks,), 0, device=device, dtype=torch.long)
    max_counts_dev.scatter_reduce_(
        0, chunk_id_per_active, counts_active, reduce="amax")
    metadata_dev = torch.stack(
        [starts_dev, ends_dev, max_counts_dev], dim=1)
    metadata_host = metadata_dev.cpu()  # SYNC #1 (per layer, not per chunk)

    # Flat list of all active expert ids, host-side, used by the
    # ModuleList indexing below. ONE sync for all chunks.
    all_active_host = active.tolist()  # SYNC #2

    expert_offset = 0
    for chunk_i, experts in enumerate(chunk_list):
        chunk_n = experts.numel()
        expert_list = all_active_host[expert_offset:expert_offset + chunk_n]
        expert_offset += chunk_n
        start = int(metadata_host[chunk_i, 0])
        end = int(metadata_host[chunk_i, 1])
        max_count = int(metadata_host[chunk_i, 2])
        if max_count == 0:
            continue

        sl = slice(start, end)
        experts_sl = experts_sorted[sl]
        tokens_sl = tokens_sorted[sl]
        weights_sl = weights_sorted[sl]
        n_assign = int(tokens_sl.numel())
        if n_assign == 0:
            continue

        expert_to_compact = torch.empty(n_experts, device=device, dtype=torch.long)
        expert_to_compact.index_copy_(
            0, experts, torch.arange(experts.numel(), device=device)
        )
        compact = expert_to_compact.index_select(0, experts_sl)
        rank = torch.arange(start, end, device=device) - offsets.index_select(
            0, experts_sl
        )

        x_padded = hidden_states.new_zeros(
            int(experts.numel()), max_count, hidden_dim)
        x_padded.index_put_((compact, rank), hidden_states.index_select(0, tokens_sl))

        w1 = torch.stack([self[e].w1.weight for e in expert_list], dim=0)
        w3 = torch.stack([self[e].w3.weight for e in expert_list], dim=0)
        w2 = torch.stack([self[e].w2.weight for e in expert_list], dim=0)

        h1 = torch.bmm(x_padded, w1.transpose(1, 2))
        h3 = torch.bmm(x_padded, w3.transpose(1, 2))
        h_mid = act_fn(h1) * h3
        y_padded = torch.bmm(h_mid, w2.transpose(1, 2))

        # Expert-saliency accumulation (fast-MoE path). The chunked compute
        # above bypasses per-expert nn.Module forward, so the tracker's
        # per-expert forward_hooks never fire. Accumulate inline here:
        # `y_pre_gate` is the expert output BEFORE gate-weight multiply
        # (matches the per-expert-hook semantics in the slow path), and
        # `experts_sl` / `weights_sl` give (expert_id, gate) per token
        # assignment.
        y_pre_gate = y_padded[compact, rank]
        tracker = getattr(self, "_pq_saliency_tracker", None)
        router_qname = getattr(self, "_pq_saliency_router", None)
        if tracker is not None and router_qname is not None:
            tracker._ensure_accumulators(router_qname, hidden_states.device)
            acc_sum = tracker.sum_g_norm.get(router_qname)
            acc_count = tracker.count.get(router_qname)
            acc_max = tracker.max_g_norm.get(router_qname)
            acc_sum_sq = tracker.sum_g_norm_sq.get(router_qname)
            if (acc_sum is not None and acc_count is not None
                    and acc_max is not None and acc_sum_sq is not None):
                norms = y_pre_gate.to(torch.float64).norm(dim=-1)  # [n_assign]
                gates64 = weights_sl.to(torch.float64)              # [n_assign]
                contribution = gates64 * norms                      # g·||f||
                contribution_sq = gates64 * norms.pow(2)            # g·||f||²
                ones_assign = torch.ones_like(experts_sl, dtype=torch.int64)
                acc_sum.index_add_(0, experts_sl, contribution)
                acc_sum_sq.index_add_(0, experts_sl, contribution_sq)
                acc_count.index_add_(0, experts_sl, ones_assign)
                acc_max.scatter_reduce_(
                    0, experts_sl, contribution,
                    reduce="amax", include_self=True,
                )

        y_valid = y_pre_gate * weights_sl.reshape(n_assign, 1)
        final_hidden_states.index_add_(0, tokens_sl, y_valid.to(hidden_states.dtype))

    return final_hidden_states


def _set_minimax_fast_moe(
    layer: nn.Module,
    enabled: bool,
    *,
    chunk_size: int = 32,
    proj_names: tuple[str, ...] = ("w1", "w2", "w3"),
    class_names: frozenset[str] | tuple[str, ...] = (),
) -> int:
    """Enable/disable chunked batched unpacked-expert replay on a layer.

    Returns the number of expert containers patched under `layer`. The patch
    is instance-local and falls back to the original forward whenever
    `_pq_fast_moe_enabled` is False. ``proj_names`` are the per-expert
    projection attribute names and ``class_names`` the declared container
    classes — both from the model profile
    (`unpacked_expert_projection_names()` / `packed_expert_module_class_names()`),
    defaulting to the Qwen/MiniMax ``('w1','w2','w3')`` and "no class filter".
    """
    patched = 0
    for module in layer.modules():
        if not _is_unpacked_experts_module(module, proj_names, class_names):
            continue
        if not hasattr(module, "_pq_original_forward"):
            module._pq_original_forward = module.forward
            module.forward = types.MethodType(_minimax_fast_experts_forward, module)
        module._pq_fast_moe_enabled = bool(enabled)
        module._pq_fast_moe_chunk_size = int(chunk_size)
        patched += 1
    return patched


# ---------------------------------------------------------------------------
# Memory snapshot (v20 hygiene)
# ---------------------------------------------------------------------------
def _read_proc_status_kb(*keys: str) -> dict[str, int]:
    """Read /proc/self/status for the given keys (e.g. 'VmHWM', 'VmRSS').
    Returns a dict of key -> kilobytes. Missing keys map to 0."""
    out = {k: 0 for k in keys}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, rest = line.partition(":")
                k = k.strip()
                if k in out:
                    out[k] = int(rest.strip().split()[0])
    except Exception:
        pass
    return out


def _print_mem_snapshot(label: str, log_prefix: str = "[incremental]"):
    """One-line memory snapshot at a phase boundary. Reads VmHWM
    (process high-water mark RSS), VmRSS (current resident), VmSwap
    (paged out), and MemAvailable (system-wide). All values in GB."""
    proc = _read_proc_status_kb("VmHWM", "VmRSS", "VmSwap")
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = -1.0
    print(f"{log_prefix} mem[{label}] "
          f"vmhwm={proc['VmHWM']/(1024**2):.1f}GB "
          f"vmrss={proc['VmRSS']/(1024**2):.1f}GB "
          f"swap={proc['VmSwap']/(1024**2):.1f}GB "
          f"sys_avail={avail_gb:.1f}GB",
          flush=True)


# ---------------------------------------------------------------------------
# Shard regex builders (unchanged public API)
# ---------------------------------------------------------------------------
def build_layer_shard_regexes(num_hidden_layers: int,
                              layers_per_shard: int,
                              layer_prefix: str = "model.layers") -> list[str]:
    regexes: list[str] = []
    for start in range(0, num_hidden_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_hidden_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        regexes.append(body)
    return regexes


def _detect_profile_for_shards(model_path: str):
    try:
        from .model_profiles.registry import detect_profile

        return detect_profile(model_path)
    except Exception:
        from .model_profiles.default import DefaultProfile

        return DefaultProfile()


# Router/gate Linears carry routing logits, not quantizable weights.
_BASE_LINEAR_EXCLUDE = (
    r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)"
)


def resolve_linear_exclude(model_path: str) -> str:
    """The probe's Linear exclusion: the router baseline OR'd with any
    profile-declared extra (`ModelProfile.probe_linear_exclude_extra`),
    for live Linears outside the serving contract's quantizable set.
    All meta stamps and hook installs must use this one resolver so
    shard-reuse keys stay consistent."""
    profile = _detect_profile_for_shards(model_path)
    extra = ""
    try:
        extra = str(profile.probe_linear_exclude_extra() or "")
    except AttributeError:
        pass
    if extra:
        return f"(?:{_BASE_LINEAR_EXCLUDE}|{extra})"
    return _BASE_LINEAR_EXCLUDE


def build_extended_shard_regexes(
    model_path: str,
    layers_per_shard: int,
    *,
    include_body: bool = True,
    include_mtp: bool = True,
    include_visual: bool = True,
    include_lm_head: bool = True,
) -> list[str]:
    """Extended shard list covering the profile-declared probe regions:

      - body transformer layers
      - optional MTP block(s)
      - optional visual/audio tower layers
      - optional lm_head
    """
    profile = _detect_profile_for_shards(model_path)
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    body_prefix = profile.body_layer_prefix()
    mtp_prefix = profile.mtp_layer_prefix()
    visual_key = profile.visual_config_key()
    visual_prefix = profile.visual_layer_prefix()
    lm_head_name = profile.lm_head_name()

    regexes: list[str] = []

    if include_body:
        n_body = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers", 0)))
        regexes.extend(build_layer_shard_regexes(
            n_body, layers_per_shard, layer_prefix=body_prefix))

    if include_mtp:
        n_mtp_config = int(profile.mtp_layer_count(cfg) or 0)
        n_mtp_actual = _count_mtp_layers_from_safetensors(
            model_path,
            layer_prefix=mtp_prefix,
        )
        # Empirical safetensors count is ground truth: a config may
        # declare MTP layers (inherited from a base) when the finetune
        # actually stripped the weights. Conversely, local Qwen3.5/3.6
        # exports can carry `mtp.*` weights even when the text config omits
        # the count. Use actual safetensors as the fallback, and cap declared
        # counts to actual when both are present.
        if n_mtp_actual > 0:
            n_mtp = min(n_mtp_config, n_mtp_actual) if n_mtp_config > 0 else n_mtp_actual
        else:
            n_mtp = 0
        if n_mtp_config > 0 and n_mtp_actual == 0:
            print(f"[shard-schedule] config declares "
                  f"{n_mtp_config} MTP layer(s) but safetensors index "
                  f"has no `{mtp_prefix}.*` keys; skipping MTP shards "
                  f"(common on finetunes that strip MTP)",
                  flush=True)
        if n_mtp > 0:
            mtp_regexes = build_layer_shard_regexes(
                n_mtp, layers_per_shard, layer_prefix=mtp_prefix)
            if mtp_regexes and profile.mtp_extra_linear_names():
                extra = "|".join(
                    re.escape(name) for name in profile.mtp_extra_linear_names()
                )
                mtp_regexes[0] = rf"(?:{extra}|{mtp_regexes[0]})"
            regexes.extend(mtp_regexes)

    if include_visual and visual_key and visual_prefix:
        vis_cfg = cfg.get(visual_key, {})
        n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
        if n_vis > 0:
            vis_per_shard = max(layers_per_shard, 4)
            regexes.extend(build_layer_shard_regexes(
                n_vis, vis_per_shard, layer_prefix=visual_prefix))

    if include_lm_head:
        regexes.append(rf"^{re.escape(lm_head_name)}$")

    return regexes


def _count_mtp_layers_from_safetensors(
    model_path: str,
    *,
    layer_prefix: str = "mtp.layers",
) -> int:
    """Fallback for when the config doesn't carry an MTP layer count:
    scan the source safetensors index and count `<layer_prefix>.<N>.` paths."""
    src = Path(model_path)
    layer_re = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.")
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        try:
            from safetensors.torch import safe_open
            mtp_indices: set[int] = set()
            for f in os.listdir(src):
                if not f.endswith(".safetensors"):
                    continue
                with safe_open(str(src / f), framework="pt") as sf:
                    for k in sf.keys():
                        m = layer_re.match(k)
                        if m:
                            mtp_indices.add(int(m.group(1)))
            return max(mtp_indices) + 1 if mtp_indices else 0
        except Exception:
            return 0
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_indices = set()
    for k in wm:
        m = layer_re.match(k)
        if m:
            mtp_indices.add(int(m.group(1)))
    return max(mtp_indices) + 1 if mtp_indices else 0


# ---------------------------------------------------------------------------
# Predeclared shard schedule (v20 step 1)
#
# A ShardSchedule is the full, statically-known list of shards that phase-3
# will process for a chunk. Each entry pairs the linear-include regex (the
# only thing the runners themselves consume) with kind + the layer indices
# in scope, so policy code (cache mark_done, instrumentation, allocator)
# can answer "what layers are in shard S?" without re-parsing regexes.
#
# This unblocks v20 steps 2-5: mark_done events fall out of
# `layers_done_after(shard_idx)`, value-aware retention can preload the
# layers reused across all shards, etc.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ShardEntry:
    shard_idx: int
    linear_include: str
    kind: str  # "body", "mtp", "visual", "lm_head"
    layer_indices: frozenset[int]
    layer_prefix: str | None  # Profile-declared layer prefix; None for lm_head.


@dataclasses.dataclass(frozen=True)
class ShardSchedule:
    entries: tuple[ShardEntry, ...]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> ShardEntry:
        return self.entries[i]

    def regexes(self) -> list[str]:
        return [e.linear_include for e in self.entries]

    def body_layer_indices(
        self,
        layer_prefix: str | None = None,
    ) -> frozenset[int]:
        out: set[int] = set()
        for e in self.entries:
            if e.kind != "body":
                continue
            if layer_prefix is None or e.layer_prefix == layer_prefix:
                out |= e.layer_indices
        return frozenset(out)

    def layers_done_after(self, shard_idx: int,
                          layer_prefix: str | None = None) -> frozenset[int]:
        """Layer indices in shard_idx's scope that no later shard touches.

        For the canonical body-shard layout (contiguous, disjoint ranges
        per shard), this is exactly the in-scope layers of shard_idx.
        For unified-sweep (one body shard) it returns the full body
        layer set after the only shard. The cache uses this signal to
        evict layers we've provably stopped tracking stats for."""
        if shard_idx >= len(self.entries):
            return frozenset()
        cur = self.entries[shard_idx]
        if layer_prefix is None:
            layer_prefix = cur.layer_prefix
        if cur.layer_prefix != layer_prefix:
            return frozenset()
        future: set[int] = set()
        for e in self.entries[shard_idx + 1:]:
            if e.layer_prefix == layer_prefix:
                future |= e.layer_indices
        return cur.layer_indices - future


def _build_body_shard_entries(num_layers: int, layers_per_shard: int,
                              layer_prefix: str,
                              kind: str,
                              start_idx: int) -> list[ShardEntry]:
    """Mirror of build_layer_shard_regexes but emits ShardEntry list."""
    entries: list[ShardEntry] = []
    sidx = start_idx
    for start in range(0, num_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        entries.append(ShardEntry(
            shard_idx=sidx,
            linear_include=body,
            kind=kind,
            layer_indices=frozenset(range(start, end)),
            layer_prefix=layer_prefix,
        ))
        sidx += 1
    return entries


def build_shard_schedule(
    *,
    model_path: str,
    num_body_layers: int,
    body_layers_per_shard: int,
    body_layer_range: tuple[int, int],
    include_mtp: bool,
    include_visual: bool,
    include_lm_head: bool,
    unified_body_sweep: bool,
) -> ShardSchedule:
    """Single source of truth for the shard list.

    body_layer_range = (first_layer, last_layer_exclusive) — slices the
    body shard list to this range (default (0, num_body_layers))."""
    profile = _detect_profile_for_shards(model_path)
    body_prefix = profile.body_layer_prefix()
    mtp_prefix = profile.mtp_layer_prefix()
    visual_key = profile.visual_config_key()
    visual_prefix = profile.visual_layer_prefix()
    lm_head_name = profile.lm_head_name()
    sidx = 0

    # Body shards (mirror old slice semantics).
    body_entries_full = _build_body_shard_entries(
        num_body_layers, body_layers_per_shard, body_prefix, "body", sidx)
    first = body_layer_range[0] // body_layers_per_shard
    last = (body_layer_range[1] + body_layers_per_shard - 1) // body_layers_per_shard
    body_entries = body_entries_full[first:last]
    # Renumber after slice so shard_idx is contiguous from 0.
    body_entries = [
        dataclasses.replace(e, shard_idx=sidx + i)
        for i, e in enumerate(body_entries)
    ]
    sidx += len(body_entries)

    if unified_body_sweep and body_entries:
        union = "(?:" + "|".join(
            f"(?:{e.linear_include})" for e in body_entries) + ")"
        union_layers = frozenset().union(
            *(e.layer_indices for e in body_entries))
        body_entries = [ShardEntry(
            shard_idx=0,
            linear_include=union,
            kind="body",
            layer_indices=union_layers,
            layer_prefix=body_prefix,
        )]
        sidx = 1

    extras: list[ShardEntry] = []
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    if include_mtp:
        n_mtp_config = int(profile.mtp_layer_count(cfg) or 0)
        n_mtp_actual = _count_mtp_layers_from_safetensors(
            model_path,
            layer_prefix=mtp_prefix,
        )
        if n_mtp_actual > 0:
            n_mtp = min(n_mtp_config, n_mtp_actual) if n_mtp_config > 0 else n_mtp_actual
        else:
            n_mtp = 0
        if n_mtp_config > 0 and n_mtp_actual == 0:
            print(f"[shard-schedule] config declares "
                  f"{n_mtp_config} MTP layer(s) but safetensors index "
                  f"has no `{mtp_prefix}.*` keys; skipping MTP shards "
                  f"(common on finetunes that strip MTP)",
                  flush=True)
        if n_mtp > 0:
            mtp_entries = _build_body_shard_entries(
                n_mtp, body_layers_per_shard, mtp_prefix, "mtp", sidx)
            if mtp_entries and profile.mtp_extra_linear_names():
                extra = "|".join(
                    re.escape(name) for name in profile.mtp_extra_linear_names()
                )
                mtp_entries[0] = dataclasses.replace(
                    mtp_entries[0],
                    linear_include=rf"(?:{extra}|{mtp_entries[0].linear_include})",
                )
            extras.extend(mtp_entries)
            sidx += len(mtp_entries)

    if include_visual and visual_key and visual_prefix:
        vis_cfg = cfg.get(visual_key, {})
        n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
        if n_vis > 0:
            vis_per_shard = max(body_layers_per_shard, 4)
            vis_entries = _build_body_shard_entries(
                n_vis, vis_per_shard, visual_prefix, "visual", sidx)
            extras.extend(vis_entries)
            sidx += len(vis_entries)

    if include_lm_head:
        # A tied head (`tie_word_embeddings` declared AND no head tensor
        # in the index) is an alias of the input embedding: same storage,
        # no source bytes of its own. It is structurally passthrough-only
        # — re-encoding it would re-encode the non-quantizable embedding
        # — so it gets no Fisher row and no cost row. Same shape as the
        # MTP skip above: config declares it, the index does not have it.
        from .tied_embeddings import lm_head_is_tied_alias
        if lm_head_is_tied_alias(model_path, profile=profile):
            print(f"[shard-schedule] `{lm_head_name}` is a tied alias of the "
                  "input embedding (config declares tie_word_embeddings and "
                  "the safetensors index has no head tensor); skipping the "
                  "lm_head shard — a tied head shares storage with the "
                  "non-quantizable embedding and is never quantized",
                  flush=True)
        else:
            extras.append(ShardEntry(
                shard_idx=sidx,
                linear_include=rf"^{re.escape(lm_head_name)}$",
                kind="lm_head",
                layer_indices=frozenset(),
                layer_prefix=None,
            ))
            sidx += 1

    return ShardSchedule(entries=tuple(body_entries + extras))


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged)
# ---------------------------------------------------------------------------
def _merge_nested_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = tgt.get(sk, 0.0) + float(sv)


def _merge_nested_int_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = int(tgt.get(sk, 0)) + int(sv)


def _route_stats_from_counts(
    router_counts: dict,
    router_totals: dict,
    router_active_counts: dict | None = None,
) -> dict[str, dict]:
    active_counts = router_active_counts or {}
    out: dict[str, dict] = {}
    for router, counts in router_counts.items():
        total = int(router_totals.get(router, 0) or 0)
        denom = max(total, 1)
        out[router] = {
            "total_tokens": total,
            "mass": dict(counts or {}),
            "active_count": dict(active_counts.get(router, {}) or {}),
            "prob": {
                str(eid): float(mass) / denom
                for eid, mass in (counts or {}).items()
            },
        }
    return out


def _expected_probe_shard_meta(args, *,
                               linear_include: str,
                               shard_idx: int,
                               activation_cache_dir: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        "requested_device": args.device,
        "requested_device_map": str(args.device_map),
        "importance_weighting": args.importance_weighting,
        "activation_cache_dir": str(Path(activation_cache_dir)),
        "linear_include": linear_include,
        "linear_exclude": resolve_linear_exclude(args.model),
        "h_detail_dir": str(Path(args.h_detail_dir)) if args.h_detail_dir else None,
        "activation_rows_limit": int(args.activation_rows_limit),
        "shard_idx": shard_idx,
        "router_coverage_version": _ROUTER_COVERAGE_VERSION,
        # Not a grouping axis: it decides whether the shard's stats
        # carry per-channel marginals at all. Reusing a flag-off shard
        # in a flag-on run would silently ship marginal-less entries.
        "emit_marginals": _marginals_enabled(),
    }


def probe_shard_is_reusable(path: Path, expected_meta: dict[str, Any]) -> bool:
    try:
        data = _read_pickle(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "stats" not in data or "meta" not in data:
        return False
    if not isinstance(data["stats"], dict):
        return False
    meta = data.get("meta") or {}
    probe_meta = dict(meta)
    probe_meta.update(meta.get("incremental_shard", {}))
    for key, expected in expected_meta.items():
        if probe_meta.get(key) != expected:
            return False
    return True


# Fields whose equality makes two shards' per-Linear Fisher stats
# interchangeable. Notably excludes `linear_include` and `shard_idx` —
# those describe the shard *grouping*, not the Linear-level numbers.
# Swapping LAYERS_PER_SHARD between runs changes grouping but not
# numbers, so probe_shard pickles are safe to pool on these axes.
_CONTENT_META_KEYS: tuple[str, ...] = (
    "model", "dataset", "nsamples", "seqlen", "dtype",
    "requested_device", "requested_device_map",
    "importance_weighting", "activation_cache_dir",
    "linear_exclude", "h_detail_dir", "activation_rows_limit",
    "router_coverage_version",
    # Marginal emission changes WHICH KEYS a stats entry carries, not
    # just its grouping, so a flag-off shard is not poolable into a
    # flag-on run — it would contribute entries with no marginals and
    # nothing downstream would notice.
    "emit_marginals",
)


def _probe_meta_flat(raw_meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten `{meta, meta.incremental_shard}` into one dict. Shards
    written by this module stash extra fields under
    `meta["incremental_shard"]`; we want to see both layers at once."""
    meta = dict(raw_meta or {})
    meta.update(meta.get("incremental_shard") or {})
    return meta


def _content_meta_compatible(raw_meta: dict[str, Any],
                             anchor: dict[str, Any]) -> bool:
    probe_meta = _probe_meta_flat(raw_meta)
    return all(probe_meta.get(k) == anchor.get(k) for k in _CONTENT_META_KEYS)


def scan_cached_linear_stats(
    shard_dir: Path,
    content_meta_anchor: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Scan `shard_dir/probe_shard_*.pkl`. Return a flat map
    `{linear_name: stats_dict}` pooled across all shards whose meta is
    content-compatible with `content_meta_anchor` (matches on model,
    dataset, nsamples, seqlen, etc — but NOT on linear_include or
    shard_idx). First-seen wins on duplicates.

    Used for LPS-invariant shard reuse: Fisher stats are intrinsic to
    each Linear, so a shard at lps=5 (L0-L4) and a shard at lps=3
    (L0-L2) share identical numbers for L0-L2, even though neither
    pickle directly equals the other. We pool them at the Linear level
    and synthesize new shards by filtering on regex.
    """
    pooled: dict[str, dict[str, Any]] = {}
    if not shard_dir.exists():
        return pooled
    for path in sorted(shard_dir.glob("probe_shard_*.pkl")):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not _content_meta_compatible(data.get("meta") or {},
                                        content_meta_anchor):
            continue
        stats = data.get("stats") or {}
        if not isinstance(stats, dict):
            continue
        for name, s in stats.items():
            if name not in pooled:
                pooled[name] = s
    return pooled


def synthesize_shard_from_linear_cache(
    linear_include: str,
    linear_exclude: str,
    cache: dict[str, dict[str, Any]],
    expected_meta: dict[str, Any],
    output_path: Path,
    expected_layers: "frozenset[int] | set[int] | None" = None,
    layer_prefix: str | None = None,
) -> bool:
    """Produce `output_path` by filtering `cache` through the shard's
    include / exclude regexes. Returns True iff any Linear matches
    (caller decides whether to run a fresh compute for the missing
    ones — this function doesn't attempt partial fill).

    The shard's regex form is `re:<pattern>` (compressed-tensors
    convention) or a bare pattern; we strip the optional `re:` prefix
    before compiling. The written pickle mirrors the shape that
    `_run_body_streaming_shard` produces so downstream consumers
    (merge_probe_pickles, probe_shard_is_reusable) see no difference
    between a freshly-computed and a synthesized shard."""
    def _compile(pat: str) -> "re.Pattern":
        p = pat[3:] if pat.startswith("re:") else pat
        return re.compile(p)

    inc = _compile(linear_include)
    exc = _compile(linear_exclude) if linear_exclude else None

    selected: dict[str, dict[str, Any]] = {}
    for name, stats in cache.items():
        if not inc.search(name):
            continue
        if exc is not None and exc.search(name):
            continue
        selected[name] = stats
    if not selected:
        return False
    # Layer-completeness gate. "Any Linear matches" is NOT shard
    # coverage: after a mid-run LAYERS_PER_SHARD change, the pooled
    # cache can cover a strict subset of this shard's layers (Laguna
    # 2026-07-23: cache held layers 0-4 of shard [0-6]; the shard was
    # declared complete and layers 5-6 silently fell out of the probe,
    # the cost table, and the allocation). A shard may only be
    # synthesized when EVERY expected layer contributes stats.
    if expected_layers and layer_prefix:
        # Profiles return the prefix both with and without the trailing
        # dot ('model.layers' vs 'model.layers.') — normalize before
        # building name probes or every membership test silently fails.
        _lp = layer_prefix.rstrip(".") + "."
        covered = {
            i for i in expected_layers
            if any(f"{_lp}{i}." in n for n in selected)
        }
        missing = sorted(set(expected_layers) - covered)
        if missing:
            print(f"[incremental] synthesize refused: cached stats miss "
                  f"layers {missing} of this shard — running fresh compute",
                  flush=True)
            return False

    payload = {
        "stats": selected,
        "router_counts": {},
        "router_totals": {},
        "router_active_counts": {},
        "expert_info": {},
        "meta": {
            **dict(expected_meta),
            "device_map": "streaming-layerwise",
            "synthesized_from_cache": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    return True


def merge_probe_pickles(paths: list[Path], output_path: Path):
    merged = None
    merged_stats = {}
    merged_router_counts = {}
    merged_router_totals = defaultdict(int)
    merged_router_active_counts = {}
    merged_expert_info = {}
    shard_metas = []

    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if merged is None:
            merged = data
        overlap = set(merged_stats) & set(data["stats"])
        if overlap:
            raise ValueError(f"probe shards overlap on {len(overlap)} stats entries")
        merged_stats.update(data["stats"])
        _merge_nested_counts(merged_router_counts, data.get("router_counts", {}))
        _merge_nested_int_counts(
            merged_router_active_counts, data.get("router_active_counts", {})
        )
        for rk, rv in data.get("router_totals", {}).items():
            merged_router_totals[rk] += int(rv)
        merged_expert_info.update(data.get("expert_info", {}))
        shard_metas.append(data.get("meta", {}))

    if merged is None:
        raise ValueError("no probe shards to merge")

    merged["stats"] = merged_stats
    merged["router_counts"] = dict(merged_router_counts)
    merged["router_totals"] = dict(merged_router_totals)
    merged["router_active_counts"] = dict(merged_router_active_counts)
    merged["expert_route_stats"] = _route_stats_from_counts(
        merged_router_counts, merged_router_totals, merged_router_active_counts,
    )
    merged["expert_info"] = merged_expert_info
    merged_meta = {
        **merged.get("meta", {}),
        "incremental": True,
        "n_shards": len(paths),
        "shards": shard_metas,
    }
    # R14: union of the per-shard calibration identities. Multi-chunk runs give
    # each shard its own calib draw, so the merged pickle must carry the SET —
    # a single combined digest could not be intersected against a validator's
    # per-repeat hashes. Keep `calib_hash` as the single-draw convenience only
    # when the run really had one draw.
    shard_calib_hashes = sorted({
        str(meta["calib_hash"])
        for meta in shard_metas
        if isinstance(meta, dict) and meta.get("calib_hash")
    })
    if shard_calib_hashes:
        merged_meta["calib_hashes"] = shard_calib_hashes
        merged_meta["calib_hash"] = (
            shard_calib_hashes[0] if len(shard_calib_hashes) == 1 else None
        )
    else:
        merged_meta.pop("calib_hash", None)
    # Propagate the calibration-chunk domain label into the merged pickle meta.
    domain_env = os.environ.get("PRISMAQUANT_PROBE_DOMAIN")
    if domain_env:
        merged_meta["domain"] = domain_env
    merged["meta"] = merged_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)


def load_num_hidden_layers(model_path: str) -> int:
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Could not infer num_hidden_layers from {cfg_path}")
    return n


def config_num_kv_shared_layers(model_path: str) -> int:
    """``num_kv_shared_layers`` from the config (text_config or top-level).

    Returns 0 when absent. KV-sharing models (num_kv_shared_layers>0, e.g. some
    Gemma4 variants) reuse one layer's K/V in later layers. The phase-3 sweep
    forwards each layer in isolation from a ``.detach()``ed capture of that K/V,
    and that borrowed tensor severed the Fisher cotangent belonging to the
    *storing* layer's k_proj/v_proj — under-counting their h_trace (review
    finding MINOR-M33). ``SharedStateCotangents`` reconnects it, so this lookup
    now only feeds the guard that fires if that path is switched off.
    """
    staged = stage_text_only(model_path)
    cfg_path = Path(staged) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    for src in (text_cfg, cfg):
        if isinstance(src, dict):
            v = src.get("num_kv_shared_layers")
            if isinstance(v, int):
                return int(v)
    return 0


def kv_shared_fisher_block_reason(model_path: str) -> str | None:
    """Fail-fast message if the streaming Fisher probe must not run here.

    INVERTED (MINOR-M33 closed): KV-sharing models are now probed normally,
    because the reverse sweep routes each consumer's cotangent back to the
    layer that produced the borrowed K/V (``SharedStateCotangents``, verified
    against an end-to-end backward in
    ``tests/test_kv_cotangent_path.py``). The guard therefore fires only when
    that path is UNAVAILABLE — today the single way to get there is switching
    it off with ``PRISMAQUANT_KV_COTANGENT=0``, which restores the severed
    cotangent and its k/v_proj under-count. ``PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1``
    still overrides, for anyone deliberately reproducing a pre-fix probe.
    """
    kv = config_num_kv_shared_layers(model_path)
    if kv <= 0 or kv_cotangent_path_enabled():
        return None
    if os.environ.get("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "0") != "0":
        return None
    return (
        f"[incremental] model has num_kv_shared_layers={kv} and "
        "PRISMAQUANT_KV_COTANGENT=0 disables the KV-cotangent path: the "
        "streaming Fisher probe would under-count the storing layer's "
        "k_proj/v_proj h_trace (shared-consumer cotangent severed by the "
        "phase-3 K/V detach; review finding MINOR-M33). Unset "
        "PRISMAQUANT_KV_COTANGENT to probe correctly, or set "
        "PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1 to probe anyway, accepting the "
        "k/v_proj under-count."
    )


# Streaming infrastructure — `StreamingContext`, `_build_streaming_context`,
# and `_classify_shard` live in `streaming_model` so both the probe and
# the cost measurement share one implementation.


# ---------------------------------------------------------------------------
# Global precompute — Phase-1 (streaming forward) and Phase-2 (chunked CE
# backward) produce artifacts that are identical across every body shard:
# only Phase-3 (per-layer Fisher hooks + reverse sweep) depends on the
# shard's scope. Computing Phase-1 + Phase-2 once and reusing the cached
# activations + grad_at_tail across all shards roughly halves wall time
# on models with many body shards (e.g. Qwen3.5-122B).
#
# Resident linears (lm_head, root projections) must have their Fisher
# hooks fire during Phase-2's chunked CE backward, because Phase-3's
# reverse sweep doesn't re-invoke lm_head. So the global Phase-2 installs
# hooks on the union of resident linears matched by ANY shard's include
# regex; each per-shard runner later filters that union to its own scope.
# ---------------------------------------------------------------------------


def _resident_linear_fqns(model: nn.Module, layers_prefix: str,
                          num_layers: int) -> list[str]:
    """All nn.Linear fqns NOT under a decoder-layer prefix (lm_head,
    root-level projections). These are resident during streaming."""
    resident: list[str] = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(n.startswith(f"{layers_prefix}{L}.") for L in range(num_layers)):
            continue
        resident.append(n)
    return resident


def _compute_precompute_key(model_path: str, dataset_name: str,
                            nsamples: int, seqlen: int, dtype_name: str,
                            device: str, importance_weighting: bool,
                            resident_include_union: str,
                            emit_marginals: bool = False) -> dict[str, Any]:
    """Fingerprint for the global precompute cache. If any of these
    inputs change, recompute; otherwise reuse the cached tensors.

    `emit_marginals` belongs in the key because the resident marginal
    vectors are written *into* the cached stats: a cache built with the
    flag off carries no marginals, and silently reusing it with the flag
    on would yield a probe that claims marginals and has none.
    """
    return {
        "model": model_path,
        "dataset": dataset_name,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "dtype": dtype_name,
        "device": device,
        "importance_weighting": importance_weighting,
        "resident_include_union": resident_include_union,
        "router_coverage_version": _ROUTER_COVERAGE_VERSION,
        "emit_marginals": bool(emit_marginals),
    }


# In-process StreamingContext + tokenizer cache. Populated when
# `PRISMAQUANT_PROBE_CTX_CACHE=1` is set. Keyed by (model_path, device,
# dtype). Lets an in-process driver reuse a single loaded model
# across N calibration chunks instead of paying the offload + tokenizer
# rebuild cost N times.
_PROBE_CTX_CACHE: dict = {}


# v22 Fix A: lazy weight-stats cache.
#
# w_max_abs and w_norm_sq are invariants of each Linear's weight. The
# original probe code recomputed them at every shard's hook setup —
# that fires `.abs().max().item()` and `.pow(2).sum().item()` per
# tracked Linear, totaling ~94k device syncs per phase-3 sweep. Each
# sync is a ~50 us host stall AND blocks subsequent kernel issue, so
# the cumulative GPU pipeline gap was several seconds per chunk.
#
# This cache is keyed by (fqn, weight.data_ptr) so a model swap or
# in-place weight modification (for example, an export pass) invalidates
# automatically — different storage, different key. Within a single
# probe run the weights are immutable, so the cache holds for the whole
# multi-chunk driver lifetime.
_W_STATS_CACHE: dict[tuple[str, int, tuple[int, ...]], tuple[float, float]] = {}
_ROUTER_COVERAGE_VERSION = 2


def _get_or_compute_w_stats(fqn: str, weight) -> tuple[float, float]:
    """Return (w_max_abs, w_norm_sq) for `weight`, caching by FQN +
    storage pointer + shape so repeated calls within a probe run are
    free. Uses one batched .cpu() call instead of two `.item()` syncs.
    """
    try:
        ptr = int(weight.data_ptr())
    except Exception:
        ptr = 0
    key = (fqn, ptr, tuple(weight.shape))
    cached = _W_STATS_CACHE.get(key)
    if cached is not None:
        return cached
    w_det = weight.detach()
    # Stack the two reductions and pull them off the device in one sync.
    stats = torch.stack(
        [w_det.abs().max(), w_det.pow(2).sum()]
    ).float().cpu().tolist()
    out = (float(stats[0]), float(stats[1]))
    _W_STATS_CACHE[key] = out
    return out


@dataclasses.dataclass
class GlobalPrecompute:
    """Shard-independent artifacts from Phase-1 + Phase-2.

    - `activations_cpu[L]` is the hidden state at the entry to layer L;
      `activations_cpu[num_layers]` is the final hidden state (input to
      `base_model.norm`).
    - `grad_at_tail` is the gradient of CE loss wrt the final hidden
      state, used as the seed for Phase-3's reverse sweep.
    - `resident_stats` / `resident_h_full` hold Fisher for every
      resident linear matched by the union-of-shards regex. Each shard
      runner filters these dicts to its own include regex.
    - `resident_act_snaps` holds (per-fqn) CPU activation snapshots for
      resident linears, used by the cost stage's ActivationIndex.
    - `expert_info` mirrors `sensitivity_probe.discover_moe_structure`'s
      output (Linear qname -> (router_qname, expert_id_str)).
    """
    activations_cpu: list[torch.Tensor]
    grad_at_tail: torch.Tensor
    ids: torch.Tensor  # shape (N, T), dtype long, on device
    resident_stats: dict[str, dict]
    resident_h_full: dict[str, torch.Tensor]
    resident_g2_per_token: dict[str, torch.Tensor]
    resident_act_snaps: dict[str, list[torch.Tensor]]
    resident_act_row_indices: dict[str, list[torch.Tensor]]
    expert_info: dict[str, tuple[str, str]]
    router_counts: dict[str, dict[str, float]]
    router_totals: dict[str, int]
    router_active_counts: dict[str, dict[str, int]]
    expert_route_stats: dict[str, dict]
    # Per-pass shared forward state (e.g. Gemma4 shared_kv_states): captured at
    # the end of phase-1's sequential forward, reused in phase-3's isolated
    # per-layer forwards so KV-sharing layers see their borrowed K/V.
    shared_pass_state: dict | None = None
    # Reusable forward-state derivable from ids + model; recomputed on demand.


def _compute_global_precompute(
    ctx: StreamingContext,
    *,
    calib: torch.Tensor,
    importance_weighting: bool,
    prefetch_lookahead: int,
    minimax_fast_moe: bool,
    minimax_fast_moe_chunk_size: int,
    resident_include_union: str,
    resident_exclude: str,
    activation_cache_dir: str | None,
) -> GlobalPrecompute:
    """Run Phase-1 (streaming forward, cache activations on CPU) and
    Phase-2 (chunked CE backward through lm_head). Install resident
    linear hooks BEFORE Phase-2 runs so their Fisher is captured here
    — Phase-3 never re-invokes lm_head and so can't retroactively
    collect them. Returns a `GlobalPrecompute` consumed by every
    per-shard runner."""
    device = ctx.device
    dtype = ctx.dtype
    model = ctx.model
    base_model = ctx.base_model
    from .model_profiles import profile_from_model as _profile_from_model
    profile = _profile_from_model(model)
    layers = ctx.layers
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)
    ids = calib.to(device)
    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)

    prefetch_depth = prefetch_lookahead

    # Profile-driven hidden-state shape adapter (refactor #32). Default
    # profile passes through; DSv4 expands single-stream `[B, S, H]` to
    # multi-stream `[B, S, hc_mult, H]` (mirrors `DeepseekV4Model.forward`).
    from .model_profiles import profile_from_model as _profile_from_model
    _profile = _profile_from_model(base_model)

    # ---- Phase 1: streaming forward, cache activations on CPU ----
    phase1_expert_info = discover_moe_structure(model, profile=_profile)
    phase1_router_names = sorted(discover_moe_routers(
        model, profile=_profile))
    phase1_tracker = RouterTracker(
        model, phase1_router_names, top_k=read_top_k(model))

    t_phase = time.time()
    with torch.no_grad():
        hidden = base_model.embed_tokens(ids).to(dtype)
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids, _profile)
    causal_mask = _compute_attention_mask(base_model, hidden, position_ids)

    hidden = _profile.expand_hidden_for_layers(hidden, base_model)

    # Per-pass shared forward state (e.g. Gemma4 cross-layer KV sharing),
    # threaded into every layer call in the sequential loop below and
    # captured afterward for phase-3's isolated forwards.
    pass_state = _profile.new_forward_pass_state()

    print(f"[incremental/global] phase-1 N={batch_size} T={tokens_in_sample} "
          f"hidden={tuple(hidden.shape)}", flush=True)

    for d in range(prefetch_depth):
        ctx.schedule_prefetch(d)
    # Phase-1 activation capture. Default: stream each layer's activation
    # to host inside the loop — stacking all L+1 activations
    # device-resident and doing one batched .cpu() at the end (v22 Fix
    # E1) doubles the peak device memory of the activation working set at
    # the exact phase-1/2 transition where the probe's high-water mark
    # already sits, which DSv4's multi-stream hidden (hc_mult x wider)
    # can't afford. Honest accounting: the memory saving is real; the
    # relative *transfer-time* cost of per-layer vs batched copies is
    # unmeasured — PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER=1 restores the
    # v22 batched single-transfer behavior for an A/B. Read once per
    # probe run (per-layer probe path, not a per-token hot path); both
    # variants report their true copy time as `host transfer` below.
    batched_act_transfer = os.environ.get(
        "PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER", ""
    ).lower() in {"1", "true", "yes"}
    t_h2h_total = 0.0
    acts: list[torch.Tensor] = []

    def _capture_act(t: torch.Tensor) -> None:
        nonlocal t_h2h_total
        if batched_act_transfer:
            acts.append(t.detach())
            return
        t0 = time.time()
        acts.append(t.detach().to("cpu"))
        t_h2h_total += time.time() - t0

    _capture_act(hidden)
    try:
        for L in range(num_layers):
            load_t0 = time.time()
            src = ctx.install(L)
            ctx.schedule_prefetch(L + prefetch_depth)
            load_s = time.time() - load_t0
            if minimax_fast_moe:
                _set_minimax_fast_moe(
                    layers[L], True,
                    chunk_size=minimax_fast_moe_chunk_size,
                    proj_names=tuple(_profile.unpacked_expert_projection_names()),
                    class_names=tuple(_profile.packed_expert_module_class_names()),
                )
            fwd_t0 = time.time()
            with torch.no_grad():
                out = _call_layer(
                    layers[L], hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    **_profile.extra_layer_kwargs(input_ids=ids),
                    pass_state=pass_state,
                )
            fwd_s = time.time() - fwd_t0
            hidden = out
            _capture_act(hidden)
            ctx.unload(L)
            if L % 8 == 0 or L == num_layers - 1:
                print(f"[incremental/global] fwd L{L:02d}  src={src}  "
                      f"load={load_s:.2f}s  fwd={fwd_s:.2f}s", flush=True)
    finally:
        phase1_tracker.remove_hooks()
    # Snapshot the cross-layer shared state (e.g. Gemma4 shared_kv_states)
    # now that the full sequential forward is done — it holds the K/V the
    # KV-sharing layers reuse. Captured to CPU for the pickled precompute so
    # phase-3's isolated forwards can reconstruct it.
    shared_pass_state = _profile.capture_forward_pass_state(pass_state)

    if batched_act_transfer:
        # v22 Fix E1: all captures share one (B, T, ..., H) shape — stack
        # into a single (L+1, ...) tensor, one device→host copy, then
        # split back into the list layout the precompute pickle and
        # phase-3 expect.
        t0 = time.time()
        stacked = torch.stack(acts, dim=0).cpu()
        activations_cpu: list[torch.Tensor] = [
            stacked[i].clone() for i in range(stacked.size(0))
        ]
        del stacked
        t_h2h_total = time.time() - t0
    else:
        activations_cpu = acts
    acts = []
    print(f"[incremental/global] phase-1 forward: {time.time()-t_phase:.1f}s  "
          f"(host transfer {t_h2h_total:.1f}s)  "
          f"{ctx.layer_cache.summary()}", flush=True)

    phase1_router_counts = phase1_tracker.counts
    phase1_router_totals = dict(phase1_tracker.total_tokens)
    phase1_router_active_counts = phase1_tracker.active_counts
    phase1_expert_route_stats = phase1_tracker.route_stats
    print(f"[incremental/global] router coverage: "
          f"{len(phase1_router_counts)}/{len(phase1_router_names)} routers "
          f"recorded", flush=True)

    # ---- Phase 2: final norm + lm_head + CE loss; grad at final hidden ----
    ctx.layer_cache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Resident-linear Fisher hooks. We collect the union of all shards'
    # resident-scope linears here; each per-shard runner later filters to
    # its own regex. The machinery mirrors the body-layer Phase-3 hooks.
    inc = re.compile(resident_include_union)
    exc = re.compile(resident_exclude)
    all_resident = _resident_linear_fqns(model, layers_prefix, num_layers)
    resident_tracked = [n for n in all_resident
                        if inc.search(n) and not exc.search(n)]

    resident_stats: dict[str, dict] = {}
    resident_h_full: dict[str, torch.Tensor] = {}
    resident_g2_per_token: dict[str, list[torch.Tensor]] = defaultdict(list)
    # Device-resident per-channel marginals, drained once after phase-2's
    # backward completes (below) rather than per hook call.
    resident_marginals: dict[str, list[torch.Tensor]] = {}
    _emit_marginals = _marginals_enabled()
    resident_saved_inputs: dict[str, torch.Tensor] = {}
    resident_handles: list = []
    resident_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    resident_act_row_indices: dict[str, list[torch.Tensor]] = defaultdict(list)
    resident_act_rows: dict[str, int] = defaultdict(int)
    resident_act_token_offsets: dict[str, int] = defaultdict(int)
    resident_input_rows_limit = 256
    _resident_cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if _resident_cache_dir is not None:
        _resident_cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_resident_fwd(name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            resident_saved_inputs[name] = x.detach()
            if _resident_cache_dir is not None:
                need = resident_input_rows_limit - resident_act_rows[name]
                flat = x.detach().reshape(-1, x.size(-1))
                base = int(resident_act_token_offsets[name])
                resident_act_token_offsets[name] += int(flat.size(0))
                if need > 0:
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                        flat = flat.index_select(0, idx)
                    else:
                        idx = torch.arange(flat.size(0), device=flat.device)
                    resident_act_snaps[name].append(flat.to("cpu"))
                    resident_act_row_indices[name].append(
                        (idx.detach().to("cpu", dtype=torch.long) + base)
                    )
                    resident_act_rows[name] += flat.size(0)
        return hook

    def _make_resident_bwd(name: str, mod_ref: nn.Linear):
        def hook(module, grad_input, grad_output):
            gy = grad_output[0]
            x = resident_saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            gy2 = gy.reshape(-1, gy.size(-1))
            x2 = x.reshape(-1, x.size(-1))
            # CORRECT empirical-Fisher: Σ_t ‖∇_t‖² (per-token-summed),
            # not ‖Σ_t ∇_t‖² (sum-then-squared, which inflates by the
            # cross-token gradient covariance — 5-50× on autoregressive
            # sequences with correlated gradients). Outer-product norm
            # identity gives a cheaper trace too: ‖a·b^T‖²_F = ‖a‖²·‖b‖².
            # Mixed precision: bf16 squaring + matmul, fp32 result.
            gy2_sq = gy2.pow(2)                  # bf16 (T, out)
            x2_sq = x2.pow(2)                    # bf16 (T, in)
            chunk_h = (gy2_sq.t() @ x2_sq).float()  # bf16 matmul + fp32 result
            resident_g2_per_token[name].append(
                gy2_sq.sum(dim=1).detach().to("cpu", dtype=torch.float32)
            )
            acc = resident_h_full.get(name)
            if acc is None:
                acc = torch.zeros(
                    int(gy2.size(1)), int(x2.size(1)),
                    dtype=torch.float32, device="cpu")
                resident_h_full[name] = acc
            acc.add_(chunk_h.float().to("cpu"))
            # Trace from the SAME fp32 object the marginals reduce, which is
            # how both body-layer sites do it (`h_trace_dev = chunk_h.sum()`).
            # This makes `sum(fisher_row) == sum(fisher_col) == h_trace_raw`
            # hold BY CONSTRUCTION rather than as a numerical coincidence.
            #
            # It used to use the outer-product-norm identity
            # `Σ_t (Σ_o gy²)(Σ_i x²)`, justified as "avoids a second full
            # matmul". That justification was stale on THIS path: `chunk_h` is
            # materialized unconditionally three lines up (it has to be — it
            # feeds `resident_h_full`), so summing it is free and no second
            # matmul was ever avoided.
            #
            # The identity route was also numerically wrong here, and lm_head
            # is where it showed. Both reductions ran in bf16 (no
            # `dtype=torch.float32`, unlike `_marginal_chunk`, which forces
            # fp32 precisely because "a T-long running sum in bf16 loses real
            # precision for free"). `gy2_sq.sum(dim=1)` reduces over the OUTPUT
            # dim, which for lm_head is the vocabulary — ~152k bf16 addends on
            # Qwen3 against ~1-5k for a body Linear. With 8 mantissa bits that
            # cost ~1e-3 relative on lm_head and ~1e-8 elsewhere, which is
            # exactly the split the first real-model run measured: lm_head was
            # the ONLY unit of 197 whose h_trace disagreed with its own
            # marginals, and it failed `SensitivityCard.validate()` alone.
            resident_stats[name]["h_trace_raw"] += float(chunk_h.sum().item())
            if _emit_marginals:
                _marginal_accumulate(
                    resident_marginals, name,
                    _marginal_chunk(gy2_sq, x2_sq, x2, chunk_h))
            w = mod_ref.weight
            if w is not None and not w.is_meta:
                resident_stats[name]["h_w2_sum_raw"] += float(
                    (chunk_h * w.detach().float().pow(2).to(chunk_h.device))
                    .sum().item())
            resident_stats[name]["n_tokens_seen"] += x2.size(0)
        return hook

    def _make_resident_grouped_bwd(name: str, mod_ref: nn.Linear,
                                   num_groups: int):
        """Resident-path grouped fold (wo_a shape). Immediate like the
        dense resident hook; reductions via `grouped_linear_fisher_chunk`
        so both backends share one mechanism. Marginals land in the same
        five-key slot and index the flat [G*R, D] plane."""
        def hook(module, grad_input, grad_output):
            gy = grad_output[0]
            x = resident_saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            pieces = grouped_linear_fisher_chunk(x, gy, num_groups,
                                                 mod_ref.weight)
            resident_stats[name]["h_trace_raw"] += float(
                pieces["h_trace"].item())
            if _emit_marginals:
                _marginal_accumulate(
                    resident_marginals, name,
                    [pieces["fisher_row"], pieces["fisher_col"],
                     pieces["g_sq_sum"], pieces["act_sq_sum"],
                     pieces["act_absmax"]])
            w = mod_ref.weight
            if w is not None and not w.is_meta:
                if pieces["h_w2"] is not None:
                    resident_stats[name]["h_w2_sum_raw"] += float(
                        pieces["h_w2"].item())
            # TOKENS, not token-group pairs (metadata parity with dense).
            tokens = int(x.numel()) // (num_groups * int(x.shape[-1]))
            resident_stats[name]["n_tokens_seen"] += tokens
        return hook

    for fqn in resident_tracked:
        mod = model.get_submodule(fqn)
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight
        if w.is_meta:
            continue
        num_groups_res = grouped_linear_groups(mod, _profile)
        if num_groups_res is not None:
            resident_stats[fqn] = grouped_linear_stats_entry(
                mod, num_groups_res,
                w_max_abs=float(w.detach().abs().max().item()),
                w_norm_sq=float(w.detach().pow(2).sum().item()))
            if _emit_marginals:
                resident_stats[fqn].update(
                    _marginal_zeros(mod.out_features, mod.in_features))
            for p in mod.parameters():
                p.requires_grad_(True)
            resident_handles.append(
                mod.register_forward_hook(_make_resident_fwd(fqn)))
            resident_handles.append(mod.register_full_backward_hook(
                _make_resident_grouped_bwd(fqn, mod, num_groups_res)))
            continue
        resident_stats[fqn] = {
            "h_trace_raw": 0.0,
            "h_w2_sum_raw": 0.0,
            "w_max_abs": float(w.detach().abs().max().item()),
            "w_norm_sq": float(w.detach().pow(2).sum().item()),
            "n_params": int(w.numel()),
            "in_features": mod.in_features,
            "out_features": mod.out_features,
            "n_tokens_seen": 0,
            "route_prob": None,
            "router_path": None,
            "expert_id": None,
        }
        if _emit_marginals:
            resident_stats[fqn].update(
                _marginal_zeros(mod.out_features, mod.in_features))
        for p in mod.parameters():
            p.requires_grad_(True)
        resident_handles.append(mod.register_forward_hook(_make_resident_fwd(fqn)))
        resident_handles.append(
            mod.register_full_backward_hook(_make_resident_bwd(fqn, mod)))

    t_phase = time.time()
    final_hidden = activations_cpu[-1].to(device).to(dtype).requires_grad_(True)
    # Profile-driven hidden-state collapse (refactor #32). Default
    # profile passes through; DSv4 calls `base_model.hc_head(...)` to
    # fold multi-stream `[B, T, hc_mult, H]` back to `[B, T, H]`.
    final_hidden_for_norm = _profile.collapse_hidden_after_layers(
        final_hidden, base_model)
    norm_out = _get_final_norm(base_model)(final_hidden_for_norm)
    norm_out_d = norm_out.detach().requires_grad_(True)
    grad_buf = torch.zeros_like(norm_out_d)
    chunk_T = 256
    N, T, _ = norm_out_d.shape
    if importance_weighting:
        total_ce, total_count = 0.0, 0
        for start in range(0, T - 1, chunk_T):
            end = min(start + chunk_T, T)
            with torch.no_grad():
                preds = model.lm_head(norm_out_d[:, start:end, :]).float()
                cut = end - 1 - start if end >= T else end - start
                if cut <= 0:
                    continue
                preds = preds[:, :cut, :]
                tgt = ids[:, start + 1:start + 1 + cut]
                lp_c = F.log_softmax(preds.reshape(-1, preds.size(-1)), dim=-1)
                tok_ce = -lp_c.gather(1, tgt.reshape(-1, 1)).squeeze(1)
                total_ce += float(tok_ce.sum().item())
                total_count += int(tok_ce.numel())
        ce_mean = total_ce / max(total_count, 1)
    else:
        ce_mean = None

    for start in range(0, T - 1, chunk_T):
        end = min(start + chunk_T, T)
        cut = end - 1 - start if end >= T else end - start
        if cut <= 0:
            continue
        preds = model.lm_head(norm_out_d[:, start:end, :]).float()[:, :cut, :]
        tgt = ids[:, start + 1:start + 1 + cut]
        lp_c = F.log_softmax(preds.reshape(-1, preds.size(-1)), dim=-1)
        tok_ce = -lp_c.gather(1, tgt.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                w = (tok_ce.detach() / max(ce_mean, 1e-6)).clamp(0.25, 4.0)
            chunk_loss = (tok_ce * w).sum()
        else:
            chunk_loss = tok_ce.sum()
        g, = torch.autograd.grad(chunk_loss, norm_out_d, retain_graph=False)
        grad_buf.add_(g)
        del preds, lp_c, tok_ce, chunk_loss, g
    norm_out.backward(grad_buf)
    grad_at_tail = final_hidden.grad.detach().cpu().clone()
    for h in resident_handles:
        h.remove()
    resident_handles.clear()
    resident_saved_inputs.clear()
    # One device→host transfer for every resident Linear's marginals,
    # after the backward is done — the hooks themselves never synced.
    _marginal_flush(resident_marginals, resident_stats)
    del grad_buf, norm_out, norm_out_d, final_hidden
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[incremental/global] phase-2 loss+head bwd: {time.time()-t_phase:.1f}s  "
          f"(resident stats collected: {len(resident_stats)})",
          flush=True)

    return GlobalPrecompute(
        activations_cpu=activations_cpu,
        grad_at_tail=grad_at_tail,
        ids=ids,
        resident_stats=resident_stats,
        resident_h_full=resident_h_full,
        resident_g2_per_token={
            name: torch.cat(parts, dim=0)
            for name, parts in resident_g2_per_token.items()
            if parts
        },
        resident_act_snaps=dict(resident_act_snaps),
        resident_act_row_indices=dict(resident_act_row_indices),
        expert_info=phase1_expert_info,
        router_counts=phase1_router_counts,
        router_totals=phase1_router_totals,
        router_active_counts=phase1_router_active_counts,
        expert_route_stats=phase1_expert_route_stats,
        shared_pass_state=shared_pass_state,
    )


def _save_precompute_cache(path: Path, pre: GlobalPrecompute,
                           meta: dict[str, Any]) -> None:
    """Persist Phase-1 + Phase-2 artifacts to disk so an interrupted
    probe run can resume without redoing them. Tensors stay in CPU
    format; this file is on the order of (num_layers+1) * act_size,
    typically hundreds of MB for 122B with N=4 T=256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "activations_cpu": pre.activations_cpu,
        "grad_at_tail": pre.grad_at_tail,
        "ids_cpu": pre.ids.detach().cpu(),
        "resident_stats": pre.resident_stats,
        "resident_h_full": pre.resident_h_full,
        "resident_g2_per_token": pre.resident_g2_per_token,
        "resident_act_snaps": pre.resident_act_snaps,
        "resident_act_row_indices": pre.resident_act_row_indices,
        "expert_info": pre.expert_info,
        "router_counts": pre.router_counts,
        "router_totals": pre.router_totals,
        "router_active_counts": pre.router_active_counts,
        "expert_route_stats": pre.expert_route_stats,
        # Per-pass cross-layer shared state captured at the end of phase-1
        # (Gemma4 `shared_kv_states`). MUST be persisted: phase-3's isolated
        # forwards rebuild each KV-sharing layer's borrowed K/V from it, and
        # the shard runners routinely read the precompute back from this
        # cache (resume, or one shard process per body shard). Without it a
        # resumed run hands KV-sharing layers an empty dict and the layer
        # raises `KeyError: <source layer idx>` inside attention.
        "shared_pass_state": pre.shared_pass_state,
        "meta": meta,
    }, str(path))


def _load_precompute_cache(path: Path, expected_meta: dict[str, Any],
                           device: torch.device) -> GlobalPrecompute | None:
    """Load cached precompute if meta matches; return None otherwise."""
    if not path.exists():
        return None
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[incremental/global] cache load failed ({e}); recomputing",
              flush=True)
        return None
    cached_meta = data.get("meta") or {}
    for key, expected in expected_meta.items():
        if cached_meta.get(key) != expected:
            print(f"[incremental/global] cache meta mismatch on {key!r}: "
                  f"cached={cached_meta.get(key)!r} expected={expected!r}; "
                  "recomputing", flush=True)
            return None
    return GlobalPrecompute(
        activations_cpu=data["activations_cpu"],
        grad_at_tail=data["grad_at_tail"],
        ids=data["ids_cpu"].to(device),
        resident_stats=data["resident_stats"],
        resident_h_full=data["resident_h_full"],
        resident_g2_per_token=data.get("resident_g2_per_token", {}),
        resident_act_snaps=data["resident_act_snaps"],
        resident_act_row_indices=data.get("resident_act_row_indices", {}),
        expert_info=data.get("expert_info", {}),
        router_counts={},
        router_totals={},
        router_active_counts={},
        expert_route_stats={},
        # Restore the phase-1 cross-layer shared state (Gemma4
        # `shared_kv_states`); `None` for every architecture that declares no
        # per-pass shared kwargs, and for caches written before this key
        # existed — on a KV-sharing model those hit the profile's loud
        # "delete the precompute cache" error instead of a bare KeyError.
        shared_pass_state=data.get("shared_pass_state"),
    )


# `finalize_fisher_stats` lives in sensitivity_probe (next to
# h_detail_blob, so both probe backends and every h-detail writer share
# the single global-token normalization convention); it is re-exported
# here because this module is the production backend consumers import
# it from.


# ---------------------------------------------------------------------------
# Per-shard body runner — phase-3 of streaming_probe, scoped to the
# Linears matching this shard's regex. Phase-1 + Phase-2 are now global
# (see `_compute_global_precompute`); the caller passes in the cached
# `activations_cpu` + `grad_at_tail` + resident Fisher dicts.
# ---------------------------------------------------------------------------
def _run_body_streaming_shard(
    ctx: StreamingContext,
    *,
    calib: torch.Tensor,
    linear_include: str,
    linear_exclude: str,
    importance_weighting: bool,
    activation_cache_dir: str | None,
    h_detail_dir: str | None,
    output_path: str,
    dataset_name: str,
    dtype_name: str,
    seqlen: int,
    model_path: str,
    prefetch_lookahead: int = 3,
    minimax_fast_moe: bool = True,
    minimax_fast_moe_chunk_size: int = 32,
    activation_rows_limit: int = 256,
    precomputed: GlobalPrecompute | None = None,
):
    if precomputed is None:
        raise ValueError(
            "_run_body_streaming_shard requires precomputed Phase-1/Phase-2 "
            "artifacts; call _compute_global_precompute first")
    device = ctx.device
    dtype = ctx.dtype
    model = ctx.model
    base_model = ctx.base_model
    layers = ctx.layers
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    # Profile-driven Linear gathering (refactor #32). Profile decides
    # whether each Linear gets Fisher hooks. Default profile accepts
    # any `nn.Linear`; DSv4 skips `DeepseekV4GroupedLinear` (its weight
    # `[out_features, in_features_per_group]` doesn't match the per-token
    # Hessian-trace effective output dim).
    from .model_profiles import profile_from_model as _pfm
    _shard_profile = _pfm(model)
    all_linears = [
        n for n, m in model.named_modules()
        if _shard_profile.should_probe_linear(n, m)
    ]
    all_tracked = [n for n in all_linears
                   if inc.search(n) and not exc.search(n)]
    layer_linear_names: list[list[str]] = []
    for L in range(num_layers):
        pref = f"{layers_prefix}{L}."
        layer_linear_names.append([n for n in all_tracked if n.startswith(pref)])
    total_tracked = sum(len(x) for x in layer_linear_names)
    # Linears not in any decoder layer (lm_head, root-level projections,
    # visual/audio encoders wired into the model top-level) are resident
    # on device during streaming. Their Fisher was collected once during
    # the global Phase-2 (resident hooks were installed on the union of
    # shard regexes); here we filter the cached resident dicts to the
    # scope of this shard's include regex.
    resident_linears: list[str] = [
        n for n in all_tracked
        if not any(n.startswith(f"{layers_prefix}{L}.") for L in range(num_layers))
    ]
    if total_tracked == 0 and not resident_linears:
        print(f"[incremental] shard has no Linears matching "
              f"{linear_include!r} under {layers_prefix}* or model root; "
              "writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "stats": {},
                "router_counts": {},
                "router_totals": {},
                "router_active_counts": {},
                "expert_route_stats": {},
                "expert_info": {},
                "meta": {
                    "model": model_path,
                    "dataset": dataset_name,
                    "nsamples": int(calib.size(0)),
                    # R14: calibration identity, so held-out disjointness is
                    # verifiable from the artifact instead of resting on the
                    # driver passing the right --calib-skip-first.
                    "calib_hash": calibration_data_hash(calib),
                    "seqlen": seqlen,
                    "dtype": dtype_name,
                    "device_map": "streaming-layerwise",
                    "execution_device": str(device),
                    "top_k": read_top_k(model, default=2),
                    "importance_weighting": importance_weighting,
                    "activation_cache_dir": activation_cache_dir,
                    "h_detail_dir": h_detail_dir,
                    "activation_rows_limit": int(activation_rows_limit),
                    "linear_include": linear_include,
                    "linear_exclude": linear_exclude,
                },
            }, f)
        return
    print(f"[incremental] body shard: tracking {total_tracked} body Linears "
          f"across {sum(1 for x in layer_linear_names if x)} layers "
          f"+ {len(resident_linears)} resident Linears "
          f"(include={linear_include!r})", flush=True)

    # One shared Fisher denominator for every row AND every h-detail blob
    # — the global calib token count (see finalize_fisher_stats for why
    # per-row n_tokens_seen is wrong for routed-expert Linears).
    global_tokens = max(int(calib.size(0)) * int(seqlen), 1)

    top_k = read_top_k(model, default=2)

    merged_stats: dict[str, dict] = {}
    merged_h_full: dict[str, torch.Tensor] = {}
    merged_g2_per_token: dict[str, list[torch.Tensor]] = defaultdict(list)

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)

    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)

    prefetch_depth = prefetch_lookahead

    # ---- Phase 1 + Phase 2 are precomputed globally (see main()). -------
    # Use the cached activations_cpu + grad_at_tail directly and filter
    # the resident Fisher dicts down to this shard's include scope.
    activations_cpu = precomputed.activations_cpu
    grad_at_tail = precomputed.grad_at_tail.to(device)
    with torch.no_grad():
        # position_embeddings derived from the same embed output that
        # produced activations_cpu[0]; call on an on-device copy once.
        embed0 = activations_cpu[0].to(device).to(dtype)
        position_embeddings = _compute_position_embeddings(
            base_model, embed0, position_ids, _shard_profile)
        causal_mask = _compute_attention_mask(base_model, embed0, position_ids)
        del embed0
    print(f"[incremental] shard reuses global precompute "
          f"N={batch_size} T={tokens_in_sample} "
          f"layers_cached={len(activations_cpu)}", flush=True)

    # Activation snapshots for resident linears populated by the global
    # Phase-2 run. We only emit the entries whose fqn is in this shard's
    # scope (others will be claimed by another shard, or already are).
    resident_act_snaps: dict[str, list[torch.Tensor]] = {
        n: list(snaps)
        for n, snaps in precomputed.resident_act_snaps.items()
        if n in resident_linears
    }
    resident_act_row_indices: dict[str, list[torch.Tensor]] = {
        n: list(indices)
        for n, indices in precomputed.resident_act_row_indices.items()
        if n in resident_linears
    }

    # Fold resident Fisher stats + H-diag into the main accumulators so
    # downstream finalization / h-detail / pickle write paths are agnostic
    # to whether a Linear was body-scoped or resident.
    for fqn in resident_linears:
        s = precomputed.resident_stats.get(fqn)
        if s is not None:
            merged_stats[fqn] = dict(s)
        h = precomputed.resident_h_full.get(fqn)
        if h is not None:
            merged_h_full[fqn] = h.clone()
        g2 = precomputed.resident_g2_per_token.get(fqn)
        if g2 is not None:
            merged_g2_per_token[fqn].append(g2.detach().to(torch.float32).cpu())

    # Activation snap accumulators (populated during Phase-3 for body
    # Linears; resident snaps were populated during Phase-2 hooks above).
    activation_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    activation_row_indices: dict[str, list[torch.Tensor]] = defaultdict(list)
    activation_rows: dict[str, int] = defaultdict(int)
    activation_token_offsets: dict[str, int] = defaultdict(int)
    input_rows_limit = max(1, int(activation_rows_limit))
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    act_fname_sub = re.compile(r"[^A-Za-z0-9_-]")

    # v22 Fix C: async + batched activation cache writes.
    #
    # PRISMAQUANT_ACT_CACHE_ASYNC=1 (default off) defers per-Linear
    # torch.save calls to a small thread pool so the main probe thread
    # doesn't pay the file-write latency between layers. Each write is
    # short (~1-5 ms) but ~770 writes per layer × 62 layers per phase-3
    # = 47k synchronous file ops, roughly 50-200 s of wall time we
    # don't need to spend in the foreground. Pool size defaults to 4
    # workers — enough to keep up with the layer flush rate without
    # piling on the IO subsystem.
    _act_async = _env_flag("PRISMAQUANT_ACT_CACHE_ASYNC", default=True)
    _act_pool = None
    _act_pending: list = []
    if _act_async and cache_dir is not None:
        from concurrent.futures import ThreadPoolExecutor
        _act_pool = ThreadPoolExecutor(
            max_workers=int(os.environ.get("PRISMAQUANT_ACT_CACHE_WORKERS", "4")),
            thread_name_prefix="act-save",
        )

    def _act_save_one(path, payload):
        torch.save(payload, path)

    def flush_activation_snapshots(
        snaps_by_name: dict[str, list[torch.Tensor]],
        indices_by_name: dict[str, list[torch.Tensor]] | None = None,
    ):
        if cache_dir is None:
            return
        for name in list(snaps_by_name.keys()):
            snaps = snaps_by_name.pop(name)
            if not snaps:
                continue
            # If the snapshots are still on device (Fix C path), bring
            # them to host once per Linear via a non-blocking copy
            # before pickling. The fwd hook keeps them on device when
            # _act_async is on so the main thread doesn't stall on
            # device→host transfers between Linears in the same forward.
            # #43: PRISMAQUANT_ACT_CACHE_FP32 keeps activations at FP32
            # for better Hessian numerical stability in the cost step.
            # 2× storage cost; recommended when disk is plentiful.
            cache_dtype = (torch.float32
                           if os.environ.get("PRISMAQUANT_ACT_CACHE_FP32", "1") != "0"
                           else torch.bfloat16)
            X = torch.cat(snaps, dim=0).to(
                "cpu", dtype=cache_dtype
            ).contiguous()
            row_indices = None
            if indices_by_name is not None:
                index_parts = indices_by_name.pop(name, [])
                if index_parts:
                    row_indices = torch.cat(index_parts, dim=0).to(
                        torch.long
                    ).contiguous()
            payload = {"inputs": X, "name": name}
            if row_indices is not None and row_indices.numel() == X.shape[0]:
                payload["row_indices"] = row_indices
            fname = act_fname_sub.sub("__", name) + ".pt"
            target = cache_dir / fname
            if _act_pool is not None:
                fut = _act_pool.submit(_act_save_one, target, payload)
                _act_pending.append(fut)
            else:
                torch.save(payload, target)

    def drain_activation_writes():
        """Block until all background activation-cache writes have
        completed. Called at the end of the shard so the cost step sees
        a fully-flushed activation directory."""
        if _act_pool is None:
            return
        for fut in _act_pending:
            fut.result()
        _act_pending.clear()

    collect_h_full = h_detail_dir is not None
    packed_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    packed_act_rows: dict[str, int] = defaultdict(int)

    # Phase-3 reverse sweep runs only when this shard has body-scoped
    # Linears. Pure resident-scoped shards (e.g. `^lm_head$`) skip it —
    # Fisher for resident Linears was captured in Phase-2 above; the
    # tail gradient was only needed to drive the sweep over decoder
    # layers, which has no resident Linears to measure.
    if total_tracked == 0:
        print(f"[incremental] shard has only resident Linears "
              f"(n={len(resident_linears)}); skipping Phase-3 reverse sweep",
              flush=True)
        # `activations_cpu` is a shared reference into the global
        # precompute; do not free it here — the caller reuses across
        # shards. `grad_at_tail` is a per-shard device copy.
        del grad_at_tail
    else:
        # ---- Phase 3: reverse sweep, Fisher collection only on tracked Linears ----
        _print_mem_snapshot("phase-3 start")
        t_phase = time.time()
        phase_load_s = 0.0
        phase_bwd_s = 0.0
        phase_pressure_trim_bytes = 0
        load_by_src: dict[str, float] = defaultdict(float)
        count_by_src: dict[str, int] = defaultdict(int)
        grad_out = grad_at_tail
        # Smart cache: register in-scope (tracked) layers as priority so the
        # cache prefers evicting out-of-scope entries first. Also configure
        # pressure-triggered eviction (Task #3) so spikes during MoE hook
        # firing don't push the system to OOM. Threshold = max(prefetch
        # pause floor, dynamic cache reserve).
        in_scope_layers = {L for L in range(num_layers) if layer_linear_names[L]}
        ctx.layer_cache.set_priority_layers(in_scope_layers)
        ctx.configure_runtime_pressure_floor()
        # The reverse sweep exists to deliver gradients to the tracked
        # layers; backward below the LOWEST tracked layer computes VJPs
        # nobody consumes (weight-Fisher for layer a needs only the
        # cached boundary input at a and the grad at a's output). Stop
        # there — for high shards this removes most of the sweep's
        # layer loads.
        stop_L = min(in_scope_layers)
        # Cap lookahead so protected demand (priority in-scope layers +
        # pinned prefetches + the layer being consumed) fits the cache.
        # Oversubscription forces last-resort eviction of pinned entries,
        # and every such eviction is a full re-read of a multi-GB layer
        # (measured: evicted_pinned=37/sweep = ~152 GB of doubled disk
        # traffic on Laguna-117B at depth 12 with 7 in-scope layers).
        cache_slots = max(1, int(ctx.layer_cache.max_bytes
                                 // max(1, ctx.estimated_layer_bytes)))
        prefetch_depth = max(2, min(
            prefetch_depth, cache_slots - len(in_scope_layers) - 4))
        # KV-cotangent path: a fresh accumulator per SWEEP. A consumer of
        # shared state always sits above its producer, so the reverse walk
        # collects every consumer's cotangent before a producer at or above
        # stop_L is forwarded — no cross-shard or cross-sweep state. A
        # producer BELOW stop_L is untracked in this shard (its h_trace is
        # measured by the shard that tracks it, whose stop_L sits at or
        # below it); its never-delivered cotangents are discarded at sweep
        # end and surface in the pending_keys diagnostic.
        kv_cotangents = SharedStateCotangents(
            enabled=kv_cotangent_path_enabled())
        # Reverse-prefetch (Task #5): prefetcher should now look BACKWARD
        # in layer index since reverse sweep walks num_layers-1 → stop_L.
        # Schedule lookahead in the direction we're actually going.
        for d in range(prefetch_depth):
            ctx.schedule_prefetch(num_layers - 1 - d)

        for L in reversed(range(stop_L, num_layers)):
            load_t0 = time.time()
            src = ctx.install(L)
            if L - prefetch_depth >= stop_L:
                ctx.schedule_prefetch(L - prefetch_depth)
            load_s = time.time() - load_t0
            phase_load_s += load_s
            load_by_src[src] += load_s
            count_by_src[src] += 1

            tracked_here = layer_linear_names[L]
            acc_h_full: dict[str, torch.Tensor] = {}
            acc_g2_per_token: dict[str, list[torch.Tensor]] = defaultdict(list)
            acc_stats: dict[str, dict] = {}
            saved_inputs: dict[str, torch.Tensor] = {}
            handles: list = []

            # ---- Batched-MoE Fisher (Task #48) -----------------------
            # Detect MoE expert containers within this layer (modules
            # where every immediate child has w1/w2/w3 nn.Linear). For
            # tracked Linears under such a block, we DEFER the per-Linear
            # Fisher matmul to a block-level backward hook that batches
            # all experts in one bmm. Reduces kernel count from N=experts
            # × 3 weights (~768 per MoE layer) to 3 batched bmm calls.
            tracked_set = set(tracked_here)
            moe_linear_to_block: dict[str, tuple[str, int, str]] = {}
            moe_block_pending: dict[str, dict[tuple[int, str], tuple]] = {}
            moe_block_handles: list = []
            # Per-expert projection attribute names from the model profile so
            # unpacked-expert families that don't use w1/w2/w3 still get the
            # batched-Fisher block path instead of silently falling back to the
            # (correct but slower) per-Linear hooks. Default keeps Qwen behavior.
            _moe_proj = getattr(
                _shard_profile, "unpacked_expert_projection_names", None)
            moe_w_attrs = tuple(_moe_proj()) if callable(_moe_proj) else ("w1", "w2", "w3")
            for block_name, block in layers[L].named_modules():
                full_block_name = f"{layers_prefix}{L}.{block_name}" if block_name else f"{layers_prefix}{L}"
                children = list(block.named_children())
                if not children or len(children) < 2:
                    continue
                ok = True
                for _, child in children:
                    for w in moe_w_attrs:
                        if not isinstance(getattr(child, w, None), nn.Linear):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Check at least one tracked Linear lives under this block
                any_tracked = False
                for cname, child in children:
                    try:
                        eid = int(cname)
                    except ValueError:
                        ok = False; break
                    for w in moe_w_attrs:
                        ln = f"{full_block_name}.{cname}.{w}"
                        if ln in tracked_set:
                            moe_linear_to_block[ln] = (full_block_name, eid, w)
                            any_tracked = True
                if not (ok and any_tracked):
                    continue

                def _make_flush(_block_name: str):
                    def flush(module, grad_input, grad_output):
                        pending = moe_block_pending.pop(_block_name, None)
                        if not pending:
                            return
                        _emit_marginals_blk = _marginals_enabled()
                        blk_marginals: dict[str, list[torch.Tensor]] = {}
                        from collections import defaultdict as _dd
                        by_w: dict[str, list] = _dd(list)
                        for (eid, w_name), (X, gy, lname, T, w_ref) in pending.items():
                            by_w[w_name].append((eid, lname, X, gy, T, w_ref))
                        # Expert-chunk size: caps peak GPU memory per bmm.
                        # 256 experts × max_T × hidden × fp32 can hit 5+ GB
                        # for w1/w3 alone — way too much on a 121 GB box
                        # already running 110 GB of model weights. 32 experts
                        # per chunk → ~600 MB peak per bmm, safe.
                        EXPERT_CHUNK = 32
                        for w_name, items in by_w.items():
                            if not items:
                                continue
                            in_dim = items[0][2].size(1)
                            out_dim = items[0][3].size(1)
                            device = items[0][2].device
                            for cs in range(0, len(items), EXPERT_CHUNK):
                                chunk_items = items[cs:cs + EXPERT_CHUNK]
                                n_e = len(chunk_items)
                                max_T = max(it[4] for it in chunk_items)
                                X_pad = torch.zeros(n_e, max_T, in_dim,
                                                    dtype=torch.float32, device=device)
                                gy_pad = torch.zeros(n_e, max_T, out_dim,
                                                     dtype=torch.float32, device=device)
                                T_valid = torch.empty(n_e, dtype=torch.long, device=device)
                                for i, (_eid, _lname, X, gy, T, _w) in enumerate(chunk_items):
                                    X_pad[i, :T, :] = X.float()
                                    gy_pad[i, :T, :] = gy.float()
                                    T_valid[i] = T
                                X_sq = X_pad.pow(2)
                                gy_sq = gy_pad.pow(2)
                                # Drop the padded source tensors before the
                                # bmm allocates its big result, so peak is
                                # bounded by max(pad_inputs, bmm_output).
                                del X_pad, gy_pad
                                chunk_h_batch = gy_sq.transpose(1, 2).bmm(X_sq)  # (n_e, out, in)
                                gy_norm = gy_sq.sum(dim=2)
                                x_norm = X_sq.sum(dim=2)
                                # Per-channel factors must be reduced out
                                # of the batched tensors BEFORE the del.
                                # Padded rows are zero, so they are inert
                                # for both the sums and the max (x² ≥ 0)
                                # — no T_valid mask needed here.
                                if _emit_marginals_blk:
                                    g_sq_e = gy_sq.sum(dim=1, dtype=torch.float32)
                                    act_sq_e = X_sq.sum(dim=1, dtype=torch.float32)
                                    act_absmax_e = X_sq.amax(dim=1).sqrt()
                                del X_sq, gy_sq
                                per_token = gy_norm * x_norm
                                mask = (torch.arange(max_T, device=device).unsqueeze(0)
                                        < T_valid.unsqueeze(1)).to(per_token.dtype)
                                per_token = per_token * mask
                                trace_per_e = per_token.sum(dim=1)
                                for i, (_eid, lname, _X, _gy, T, w_ref) in enumerate(chunk_items):
                                    acc_stats[lname]["h_trace_raw"] += float(trace_per_e[i].item())
                                    if collect_h_full:
                                        acc_g2_per_token[lname].append(
                                            gy_norm[i, :T].detach().to(
                                                "cpu", dtype=torch.float32)
                                        )
                                    if collect_h_full:
                                        acc = acc_h_full.get(lname)
                                        if acc is None:
                                            acc = torch.zeros(out_dim, in_dim,
                                                              dtype=torch.float32, device="cpu")
                                            acc_h_full[lname] = acc
                                        acc.add_(chunk_h_batch[i].cpu())
                                    if w_ref is not None and not w_ref.is_meta:
                                        acc_stats[lname]["h_w2_sum_raw"] += float(
                                            (chunk_h_batch[i] * w_ref.detach().float().pow(2)
                                             .to(chunk_h_batch.device)).sum().item())
                                    if _emit_marginals_blk:
                                        _marginal_accumulate(
                                            blk_marginals, lname, [
                                                chunk_h_batch[i].sum(
                                                    dim=1, dtype=torch.float32),
                                                chunk_h_batch[i].sum(
                                                    dim=0, dtype=torch.float32),
                                                g_sq_e[i], act_sq_e[i],
                                                act_absmax_e[i],
                                            ])
                                del chunk_h_batch, per_token, trace_per_e
                        # One transfer for the whole block, not one per
                        # expert — same discipline as the v21 #1 stack.
                        _marginal_flush(blk_marginals, acc_stats)
                    return flush

                moe_block_handles.append(
                    block.register_full_backward_hook(_make_flush(full_block_name)))
            if moe_linear_to_block:
                # Flag so the per-Linear hook short-circuits to deferred path.
                pass  # (no-op, used as documentation; lookup happens per-call)

            def _fold_grouped_layer_stat(name: str, x, gy, mod_ref,
                                         num_groups: int):
                """Grouped-BMM Fisher fold for ONE (x, gy) pair on the
                body-shard path. Shares the dense path's storage exactly —
                device_accums scalar slots, device_marginals five-vector
                slot, acc_h_full, acc_g2_per_token — so every flush and
                merge below this line is unchanged. The math comes from
                `grouped_linear_fisher_chunk` (one mechanism with the
                non-streaming backend); only the folding differs because
                this site honors deferred_sync."""
                pieces = grouped_linear_fisher_chunk(x, gy, num_groups,
                                                     mod_ref.weight)
                per_group_slot = grouped_per_group_acc.get(name)
                if per_group_slot is None:
                    grouped_per_group_acc[name] = pieces["trace_per_group"]
                else:
                    per_group_slot.add_(pieces["trace_per_group"])
                if emit_marginals:
                    _marginal_accumulate(
                        device_marginals, name,
                        [pieces["fisher_row"], pieces["fisher_col"],
                         pieces["g_sq_sum"], pieces["act_sq_sum"],
                         pieces["act_absmax"]])
                if collect_h_full:
                    # Per-(token, group) rows: the plane-coordinate
                    # analog of the dense per-token vector.
                    gy_sq = gy.detach().reshape(
                        -1, num_groups, gy.shape[-1]).pow(2)
                    x_sq = x.detach().reshape(
                        -1, num_groups, x.shape[-1]).pow(2)
                    pt = (gy_sq.sum(dim=-1) * x_sq.sum(dim=-1)).reshape(-1)
                    acc_g2_per_token[name].append(
                        pt.detach().to("cpu", dtype=torch.float32))
                    acc = acc_h_full.get(name)
                    if acc is None:
                        acc = torch.zeros(
                            int(pieces["chunk_flat"].shape[0]),
                            int(pieces["chunk_flat"].shape[1]),
                            dtype=torch.float32, device="cpu")
                        acc_h_full[name] = acc
                    acc.add_(pieces["chunk_flat"].to(acc.device).to(acc.dtype))
                h_trace_dev = pieces["h_trace"]
                if deferred_sync:
                    slot = device_accums.get(name)
                    if slot is None:
                        slot = (
                            torch.zeros((), device=h_trace_dev.device,
                                        dtype=torch.float32),
                            torch.zeros((), device=h_trace_dev.device,
                                        dtype=torch.float32),
                        )
                        device_accums[name] = slot
                    slot[0].add_(h_trace_dev)
                    if pieces["h_w2"] is not None:
                        slot[1].add_(pieces["h_w2"])
                else:
                    acc_stats[name]["h_trace_raw"] += float(
                        h_trace_dev.item())
                    if pieces["h_w2"] is not None:
                        acc_stats[name]["h_w2_sum_raw"] += float(
                            pieces["h_w2"].item())
                tokens = int(x.numel()) // (
                    num_groups * int(x.shape[-1]))
                acc_stats[name]["n_tokens_seen"] += tokens

            def make_fwd(name: str):
                def hook(module, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    saved_inputs[name] = x.detach()
                    if cache_dir is not None:
                        need = input_rows_limit - activation_rows[name]
                        flat = x.detach().reshape(-1, x.size(-1))
                        base = int(activation_token_offsets[name])
                        activation_token_offsets[name] += int(flat.size(0))
                        if need > 0:
                            if flat.size(0) > need:
                                idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                                flat = flat.index_select(0, idx)
                            else:
                                idx = torch.arange(flat.size(0), device=flat.device)
                            # v22 Fix C: keep on device when async writes
                            # are enabled. Each per-Linear .to("cpu") in
                            # the inline path forces a device→host
                            # synchronization, stalling the forward
                            # pipeline. Deferring lets the layer's whole
                            # forward run uninterrupted; the device→host
                            # copy happens once per Linear at end-of-layer
                            # in flush_activation_snapshots.
                            if _act_async:
                                activation_snaps[name].append(flat.detach())
                            else:
                                activation_snaps[name].append(flat.to("cpu"))
                            activation_row_indices[name].append(
                                (idx.detach().to("cpu", dtype=torch.long) + base)
                            )
                            activation_rows[name] += flat.size(0)
                return hook

            # v21 #1: deferred Fisher sync. PRISMAQUANT_DEFERRED_FISHER_SYNC=1
            # accumulates h_trace_raw / h_w2_sum_raw on the device as 0-D
            # tensors and batches the host transfer to a single sync per
            # layer. The default per-Linear `.item()` calls force ~94k
            # CUDA syncs per phase-3 sweep (47k Linears × 2); deferring
            # collapses that to ~62 (one per layer) without changing the
            # math. h_full collection is unaffected (it stays on the CPU
            # path; only the device→host scalar transfers are batched).
            deferred_sync = _env_flag(
                "PRISMAQUANT_DEFERRED_FISHER_SYNC", default=True)
            emit_marginals = _marginals_enabled()
            # v22 Fix B: deferred Fisher COMPUTE. Beyond just deferring the
            # device→host syncs (above), this defers the per-Linear matmul
            # itself out of the autograd engine's per-Linear callback path.
            #
            # Why: even with #1 (no .item() syncs), every Linear's bwd
            # hook still does ~6 GPU kernel launches (gy², x², matmul,
            # sum, h_w2 multiply, sum) DURING autograd's traversal. Each
            # launch is bounced through Python and the autograd engine
            # serializes against it, leaving the GPU idle waiting for
            # Python to dispatch the next kernel. nvidia-smi dmon shows
            # 13% SM utilization during phase-3 because of this.
            #
            # With deferred_compute=on, the bwd hook just appends
            # (name, x_ref, gy_ref, mod_ref) to a per-layer queue and
            # returns immediately. The autograd graph traversal flies
            # through; the GPU stream stays busy with the layer's actual
            # bwd kernels (Q/K/V/O, attn, FFN). After `out.backward()`
            # returns, a tight Python loop drains the queue, issuing
            # the per-Linear Fisher matmuls back-to-back. The CUDA
            # driver's command queue stays full — SM utilization should
            # rise from ~13% to ~50-80%.
            #
            # Math is byte-identical to the immediate path. Memory cost:
            # the queue pins (x, gy) refs for one layer's tracked Linears
            # — typical MiniMax MoE layer ≈ 770 entries × ~2 MB = ~1.5 GB
            # peak, well within the cache budget that already accounts
            # for it.
            deferred_compute = _env_flag(
                "PRISMAQUANT_DEFERRED_FISHER_COMPUTE", default=True)
            # Per-Linear device-resident accumulators built lazily inside
            # the hook so we know the stream / device the kernel ran on.
            device_accums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            # Per-channel marginals, same device-resident discipline.
            # Kept in its own dict (not the scalar tuple) because the
            # five vectors have per-Linear shapes and cannot be stacked;
            # they are flushed with one flat cat + one .cpu() per layer.
            device_marginals: dict[str, list[torch.Tensor]] = {}
            # Per-layer deferred-compute queue: (name, x, gy, mod_ref).
            # Drained immediately after `out.backward(grad_out)` returns.
            deferred_queue: list[tuple[str, torch.Tensor, torch.Tensor, "nn.Linear"]] = []

            def make_bwd(name: str, mod_ref: nn.Linear):
                if deferred_compute:
                    # v22 Fix B path: queue refs, return fast. The Fisher
                    # math runs after out.backward() in a tight loop.
                    def hook(module, grad_input, grad_output):
                        gy = grad_output[0]
                        x = saved_inputs.pop(name, None)
                        if x is None or gy is None:
                            return
                        deferred_queue.append(
                            (name, x.detach(), gy.detach(), mod_ref))
                    return hook

                def hook(module, grad_input, grad_output):
                    gy = grad_output[0]
                    x = saved_inputs.pop(name, None)
                    if x is None or gy is None:
                        return
                    # Grouped-BMM operand: same immediate path, grouped
                    # reductions (never the dense flatten below — it would
                    # broadcast-fail against the [G*R, D] plane anyway).
                    g_count = grouped_map.get(name)
                    if g_count is not None:
                        _fold_grouped_layer_stat(
                            name, x.detach(), gy.detach(), mod_ref, g_count)
                        return
                    gy2 = gy.reshape(-1, gy.size(-1))
                    x2 = x.reshape(-1, x.size(-1))
                    T = x2.size(0)
                    # Batched-MoE deferral was attempted here (Task #48).
                    # Pinning (X, gy) for all 256 experts × 3 weights until
                    # the block-level flush peaks at ~7 GB of GPU residency
                    # which OOM'd the box on top of LayerCache + prefetch.
                    # Reverted to per-Linear path; the Fisher math fix below
                    # is the load-bearing correctness change. Proper batched
                    # implementation requires streaming partial flushes
                    # rather than holding all expert data simultaneously —
                    # filed as a follow-up (see task #48 description update).
                    # CORRECT empirical-Fisher (per-token-summed). The
                    # buggy `(Σ_t ∇_t)²` form has been replaced with
                    # `Σ_t (gy²·x²)` via the (gy²)^T @ (x²) identity.
                    # Memory-efficient mixed precision: squaring + matmul
                    # in bf16 (typical gradient magnitudes are O(1e-2 ..
                    # 1e0), so squaring stays well within bf16's safe
                    # range and the per-element precision loss averages
                    # out over T tokens and out × in matmul reductions),
                    # fp32 result for the accumulator. Halves the working
                    # set vs full-fp32 path → fits in the same memory
                    # budget the buggy bf16 code was using.
                    gy2_sq = gy2.pow(2)                        # bf16
                    x2_sq = x2.pow(2)                          # bf16
                    chunk_h = (gy2_sq.t() @ x2_sq).float()    # bf16 matmul + fp32 cast
                    if emit_marginals:
                        _marginal_accumulate(
                            device_marginals, name,
                            _marginal_chunk(gy2_sq, x2_sq, x2, chunk_h))
                    if collect_h_full:
                        acc_g2_per_token[name].append(
                            gy2_sq.sum(dim=1).detach().to(
                                "cpu", dtype=torch.float32)
                        )
                    if collect_h_full:
                        acc = acc_h_full.get(name)
                        if acc is None:
                            acc = torch.zeros(
                                int(gy2.size(1)), int(x2.size(1)),
                                dtype=torch.float32, device="cpu")
                            acc_h_full[name] = acc
                        acc.add_(chunk_h.to("cpu"))
                    # Trace from chunk_h.sum() — same value as
                    # (gy_norm·x_norm).sum() but reuses the fp32 chunk_h
                    # we already have, no extra reductions on the inputs.
                    h_trace_dev = chunk_h.sum()
                    if deferred_sync:
                        slot = device_accums.get(name)
                        if slot is None:
                            slot = (
                                torch.zeros((), device=h_trace_dev.device,
                                            dtype=torch.float32),
                                torch.zeros((), device=h_trace_dev.device,
                                            dtype=torch.float32),
                            )
                            device_accums[name] = slot
                        slot[0].add_(h_trace_dev)
                    else:
                        acc_stats[name]["h_trace_raw"] += float(
                            h_trace_dev.item())
                    # h_w2_sum is a scalar proxy used as a fallback when
                    # full per-weight Fisher isn't available. When
                    # collect_h_full is on (which is whenever the cost
                    # stage requested h_detail_dir), the per-Linear
                    # `acc_h_full` entry already encodes the full
                    # Fisher diagonal; computing the proxy on top costs
                    # ~34 MB of allocator churn per call (the weight's
                    # fp32 copy) for no extra signal.
                    if not collect_h_full:
                        w = mod_ref.weight
                        if w is not None and not w.is_meta:
                            h_w2_dev = (
                                chunk_h * w.detach().float().pow(2)
                                .to(chunk_h.device)
                            ).sum()
                            if deferred_sync:
                                # device_accums slot was created above
                                # when h_trace was computed (h_trace
                                # accum always runs first).
                                device_accums[name][1].add_(h_w2_dev)
                            else:
                                acc_stats[name]["h_w2_sum_raw"] += float(
                                    h_w2_dev.item())
                    acc_stats[name]["n_tokens_seen"] += T
                return hook

            # Grouped-BMM operands in this shard's scope (wo_a shape):
            # dispatched to `_fold_grouped_layer_stat` at backward time.
            grouped_map: dict[str, int] = {}
            for fqn in tracked_here:
                try:
                    g_mod = model.get_submodule(fqn)
                except AttributeError:
                    continue
                g_count = grouped_linear_groups(g_mod, _shard_profile)
                if g_count is not None:
                    grouped_map[fqn] = g_count

            for fqn in tracked_here:
                mod = model.get_submodule(fqn)
                if not isinstance(mod, nn.Linear):
                    continue
                w = mod.weight
                if w.is_meta:
                    continue
                # v22 Fix A: cached lookup. First call computes the
                # batched .stack().cpu() and memoizes; subsequent shards
                # / chunks return instantly with no device sync.
                w_max_abs, w_norm_sq = _get_or_compute_w_stats(fqn, w)
                if fqn in grouped_map:
                    acc_stats[fqn] = grouped_linear_stats_entry(
                        mod, grouped_map[fqn],
                        w_max_abs=w_max_abs, w_norm_sq=w_norm_sq)
                    if emit_marginals:
                        acc_stats[fqn].update(
                            _marginal_zeros(mod.out_features, mod.in_features))
                    for p in mod.parameters():
                        p.requires_grad_(True)
                    handles.append(mod.register_forward_hook(make_fwd(fqn)))
                    handles.append(mod.register_full_backward_hook(
                        make_bwd(fqn, mod)))
                    continue
                acc_stats[fqn] = {
                    "h_trace_raw": 0.0,
                    "h_w2_sum_raw": 0.0,
                    "w_max_abs": w_max_abs,
                    "w_norm_sq": w_norm_sq,
                    "n_params": int(w.numel()),
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "n_tokens_seen": 0,
                    "route_prob": None,
                    "router_path": None,
                    "expert_id": None,
                }
                if emit_marginals:
                    acc_stats[fqn].update(
                        _marginal_zeros(mod.out_features, mod.in_features))
                for p in mod.parameters():
                    p.requires_grad_(True)
                handles.append(mod.register_forward_hook(make_fwd(fqn)))
                handles.append(mod.register_full_backward_hook(make_bwd(fqn, mod)))
            # Batched-MoE deferral disabled (see make_bwd comment). Skip
            # the block-level flush hook installation entirely.
            for h in moe_block_handles:
                h.remove()
            moe_block_handles.clear()
            moe_linear_to_block.clear()

            # Scalar per-token-summed Fisher trace per packed param.
            # Values are device-resident 0-dim fp32 tensors (flushed via
            # float() below — one sync per packed param per layer).
            packed_grad_acc: dict[str, torch.Tensor] = {}
            # Per-GROUP Fisher trace [G] for grouped-BMM operands (wo_a
            # shape) in this layer. Same device-resident discipline; the
            # flush below lands it on the stats entry as a plain float
            # list (pickle/merge-safe, like `h_trace_per_expert_raw`).
            grouped_per_group_acc: dict[str, torch.Tensor] = {}
            # Per-expert per-channel Fisher [E, M] — enables per-expert
            # h_trace decomposition for the allocator's packed-3D prune
            # cost without re-measuring cost per expert. Always enabled
            # here; the accumulator's memory is ~1 MB per packed param
            # at 128 experts × 5760 channels, negligible on 121 GB RAM
            # (device-resident until the flush below).
            packed_channel_acc: dict[str, torch.Tensor] = {}
            packed_full_acc: dict[str, torch.Tensor] | None = (
                {} if h_detail_dir is not None else None)
            # AQUA A-side marginals per expert. Dense Linears get theirs
            # from `_marginal_accumulate` on the nn.Linear backward hook;
            # a packed [E, M, N] expert parameter is not an nn.Linear and
            # has no such hook, which is why an AQUA card built before
            # this carried an A-side for the dense trunk only -- 5.5% of
            # this model's parameters. The F.linear interception is the
            # equivalent site: it already holds (x, gy) for the slice.
            packed_marginal_acc: dict[str, dict[str, torch.Tensor]] = (
                {} if _marginals_enabled() else None)
            # Reverse-sweep visits every layer (gradient chain-rule needs
            # all of them), but Fisher stats should only be recorded for
            # layers in this shard's scope. Skip the packed-expert install
            # + stats merge when L is out-of-scope; backward still flows.
            layer_in_scope = bool(tracked_here) or bool(
                inc.search(f"{layers_prefix}{L}."))
            # Fast-path only layers whose Linear hooks are NOT needed
            # for this shard. In-scope MiniMax layers must run the
            # original ModuleList expert loop so per-expert nn.Linear
            # hooks collect Fisher exactly as before.
            if minimax_fast_moe:
                _mmx_proj = getattr(
                    _shard_profile, "unpacked_expert_projection_names", None)
                _mmx_cls = getattr(
                    _shard_profile, "packed_expert_module_class_names", None)
                _set_minimax_fast_moe(
                    layers[L],
                    enabled=not layer_in_scope,
                    chunk_size=minimax_fast_moe_chunk_size,
                    proj_names=tuple(_mmx_proj()) if callable(_mmx_proj) else ("w1", "w2", "w3"),
                    class_names=tuple(_mmx_cls()) if callable(_mmx_cls) else (),
                )
            packed_meta = install_packed_expert_hooks(
                layers[L], accumulator=packed_grad_acc,
                channel_accumulator=packed_channel_acc,
                full_accumulator=packed_full_acc,
                marginal_accumulator=packed_marginal_acc,
                profile=_shard_profile,
            ) if layer_in_scope else {}
            layer_prefix = f"{layers_prefix}{L}."
            layer_packed_handles: list = []
            for key, md in packed_meta.items():
                full_key = f"{layer_prefix}{key}"
                experts_qname_rel = md["_packed_experts_module"]
                md["_packed_experts_module"] = f"{layer_prefix}{experts_qname_rel}"
                acc_stats[full_key] = md
                # Capture activations for the packed-experts module so the
                # allocator can use the same input cache as nn.Linear entries.
                if cache_dir is not None:
                    try:
                        experts_mod = layers[L].get_submodule(experts_qname_rel)
                    except AttributeError:
                        experts_mod = None
                    if experts_mod is not None:
                        experts_full = f"{layer_prefix}{experts_qname_rel}"

                        def _exp_fwd(_mod, inp, _out,
                                     _q=experts_full, _rows=packed_act_rows,
                                     _snaps=packed_act_snaps,
                                     _lim=input_rows_limit):
                            x = inp[0] if isinstance(inp, tuple) else inp
                            if isinstance(x, torch.Tensor):
                                need = _lim - _rows[_q]
                                if need > 0:
                                    flat = x.detach().reshape(-1, x.size(-1))
                                    if flat.size(0) > need:
                                        idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                                        flat = flat.index_select(0, idx)
                                    _snaps[_q].append(flat.to("cpu"))
                                    _rows[_q] += flat.size(0)

                        layer_packed_handles.append(
                            experts_mod.register_forward_hook(_exp_fwd))

            # Forward + backward for this layer with the full batch.
            x_in = activations_cpu[L].to(device).to(dtype).detach().requires_grad_(True)
            bwd_t0 = time.time()
            out = _call_layer(
                layers[L], x_in,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                **_shard_profile.extra_layer_kwargs(
                    input_ids=calib.to(device) if calib is not None else None),
                # KV-sharing layers (Gemma4) reuse K/V captured in phase-1;
                # reconstruct that per-layer slice for this isolated forward.
                # One-layer scope, so the "once per pass" rule is trivially
                # satisfied: a fresh container per isolated forward.
                # `graft` swaps the borrowed tensors for grad-enabled leaves so
                # this layer's cotangent on them can be read back below; it
                # returns the caller's own object untouched when the profile
                # declares no shared state.
                pass_state=kv_cotangents.graft(
                    _shard_profile.isolated_layer_pass_state(
                        precomputed.shared_pass_state, layers[L])),
            )
            # Drive the backward with this layer's output cotangent AND the
            # cotangent its consumers accumulated on the shared state it
            # produced (empty for every architecture without cross-layer
            # sharing — then this is exactly `out.backward(grad_out)`). One
            # backward call, so autograd sums both contributions at the shared
            # tensor before its grad_fn runs and each Linear's full-backward
            # hook still fires ONCE, with the total gy.
            kv_roots, kv_grads = kv_cotangents.produced_roots()
            if kv_roots:
                torch.autograd.backward([out, *kv_roots],
                                        [grad_out.to(device), *kv_grads])
            else:
                out.backward(grad_out.to(device))
            # Fold this layer's borrowed-state gradients into the accumulator
            # (and end the layer) while the graph is still alive.
            kv_cotangents.harvest()
            bwd_s = time.time() - bwd_t0
            phase_bwd_s += bwd_s

            # v22 Fix B: drain the deferred-compute queue. The bwd hook
            # only queued (x, gy) refs; now we run the per-Linear Fisher
            # matmul in a tight Python loop, back-to-back, so the CUDA
            # driver's command queue stays full and SM utilization rises.
            #
            # The math is identical to the inline path — same sequence
            # of ops per Linear, same result. Just decoupled from the
            # autograd engine's serial Python callback dispatch. When
            # deferred_sync is also on (typical), the per-Linear
            # h_trace / h_w2_sum stay device-resident here too.
            if deferred_compute and deferred_queue:
                for name, x, gy, mod_ref in deferred_queue:
                    # Grouped-BMM operand: fold through the grouped
                    # reductions; the dense flatten below would compute
                    # the wrong-shaped marginals against the [G*R, D]
                    # plane.
                    g_count = grouped_map.get(name)
                    if g_count is not None:
                        _fold_grouped_layer_stat(name, x, gy, mod_ref,
                                                 g_count)
                        continue
                    gy2 = gy.reshape(-1, gy.size(-1))
                    x2 = x.reshape(-1, x.size(-1))
                    T = x2.size(0)
                    gy2_sq = gy2.pow(2)
                    x2_sq = x2.pow(2)
                    chunk_h = (gy2_sq.t() @ x2_sq).float()
                    if emit_marginals:
                        _marginal_accumulate(
                            device_marginals, name,
                            _marginal_chunk(gy2_sq, x2_sq, x2, chunk_h))
                    if collect_h_full:
                        acc_g2_per_token[name].append(
                            gy2_sq.sum(dim=1).detach().to(
                                "cpu", dtype=torch.float32)
                        )
                    if collect_h_full:
                        acc = acc_h_full.get(name)
                        if acc is None:
                            acc = torch.zeros(
                                int(gy2.size(1)), int(x2.size(1)),
                                dtype=torch.float32, device="cpu")
                            acc_h_full[name] = acc
                        acc.add_(chunk_h.to("cpu"))
                    h_trace_dev = chunk_h.sum()
                    if deferred_sync:
                        slot = device_accums.get(name)
                        if slot is None:
                            slot = (
                                torch.zeros((), device=h_trace_dev.device,
                                            dtype=torch.float32),
                                torch.zeros((), device=h_trace_dev.device,
                                            dtype=torch.float32),
                            )
                            device_accums[name] = slot
                        slot[0].add_(h_trace_dev)
                    else:
                        acc_stats[name]["h_trace_raw"] += float(
                            h_trace_dev.item())
                    if not collect_h_full:
                        w = mod_ref.weight
                        if w is not None and not w.is_meta:
                            h_w2_dev = (
                                chunk_h * w.detach().float().pow(2)
                                .to(chunk_h.device)
                            ).sum()
                            if deferred_sync:
                                device_accums[name][1].add_(h_w2_dev)
                            else:
                                acc_stats[name]["h_w2_sum_raw"] += float(
                                    h_w2_dev.item())
                    acc_stats[name]["n_tokens_seen"] += T
                deferred_queue.clear()

            # v21 #1: batched device→host transfer of the per-Linear
            # h_trace / h_w2_sum accumulators built up in the bwd hooks.
            # Single sync per layer instead of two per Linear (47k
            # Linears in unified-sweep × 2 = ~94k → ~62 syncs).
            if deferred_sync and device_accums:
                names = list(device_accums.keys())
                # Stack into (2, N): row 0 = h_trace, row 1 = h_w2_sum.
                # One .cpu() call → one CUDA sync.
                stacked = torch.stack(
                    [
                        torch.stack([device_accums[n][0] for n in names]),
                        torch.stack([device_accums[n][1] for n in names]),
                    ],
                    dim=0,
                )
                host = stacked.cpu().tolist()
                tr_vals, w2_vals = host[0], host[1]
                for n, tr_v, w2_v in zip(names, tr_vals, w2_vals):
                    acc_stats[n]["h_trace_raw"] += float(tr_v)
                    acc_stats[n]["h_w2_sum_raw"] += float(w2_v)
                device_accums.clear()
            # Marginals get their own flush (one flat cat, one .cpu()):
            # the five vectors are per-Linear-shaped so they cannot join
            # the (2, N) scalar stack, and they must drain even when
            # deferred_sync is off — otherwise the legacy per-Linear
            # `.item()` path would have no flush site at all.
            _marginal_flush(device_marginals, acc_stats)

            for local_key, raw in packed_grad_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                if full_key in acc_stats:
                    acc_stats[full_key]["h_trace_raw"] += float(raw)
                    acc_stats[full_key]["n_tokens_seen"] = \
                        acc_stats[full_key].get("n_tokens_seen", 0) + x_in.size(0) * x_in.size(1)
            # Per-expert Fisher trace decomposition. channel_acc[key] is
            # [E, M] (per-token-summed Σ_t gy_{e,t,m}²·‖x_{e,t}‖²);
            # summing over M collapses to [E] — per-expert Fisher trace.
            # Stored as a float list in the stat entry so it survives
            # pickle + merge without torch-device round-trips, and the
            # allocator's add_packed_prune_candidates reads it directly.
            for local_key, per_ch in packed_channel_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                if full_key not in acc_stats:
                    continue
                # per_ch is fp32 (device-resident); the .tolist() below is
                # the single flush sync for this packed param.
                per_expert_trace = per_ch.sum(dim=-1).to(torch.float64)
                prev = acc_stats[full_key].get("h_trace_per_expert_raw")
                if prev is None:
                    acc_stats[full_key]["h_trace_per_expert_raw"] = per_expert_trace.tolist()
                else:
                    summed = [p + float(q) for p, q in zip(prev, per_expert_trace.tolist())]
                    acc_stats[full_key]["h_trace_per_expert_raw"] = summed
            packed_channel_acc.clear()

            # Per-group Fisher trace flush (grouped-BMM operands). One
            # .cpu() per grouped param — same honesty as the packed
            # per-expert list: plain floats, elementwise-merged across
            # shard splits, normalized by the global token count in
            # `finalize_fisher_stats`.
            for local_key, per_group in grouped_per_group_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                entry = acc_stats.get(full_key)
                if entry is None:
                    continue
                vals = per_group.detach().to(torch.float64).tolist()
                prev = entry.get("h_trace_per_group_raw")
                if prev is None:
                    entry["h_trace_per_group_raw"] = vals
                else:
                    entry["h_trace_per_group_raw"] = [
                        p + float(v) for p, v in zip(prev, vals)]
            grouped_per_group_acc.clear()

            # Per-expert AQUA marginals. One .cpu() per array (four per
            # packed param per layer -- the arrays are [E, M] / [E, N] /
            # [E], so ~8 MB per MoE layer at E=256; the dense marginals'
            # single-concat discipline buys nothing at that count).
            for local_key, slot in (packed_marginal_acc or {}).items():
                full_key = f"{layer_prefix}{local_key}"
                entry = acc_stats.get(full_key)
                if entry is None:
                    continue
                for key, tensor in slot.items():
                    host = tensor.detach().to("cpu").numpy()
                    host = host.astype(
                        np.float64 if key == "expert_tokens" else np.float32)
                    prev = entry.get(key)
                    if prev is None:
                        entry[key] = host.copy()
                    elif key == "expert_act_absmax":
                        entry[key] = np.maximum(prev, host)
                    else:
                        entry[key] = prev + host
            if packed_marginal_acc:
                packed_marginal_acc.clear()

            grad_out = x_in.grad.detach().clone().cpu()

            for h in handles:
                h.remove()
            for h in layer_packed_handles:
                h.remove()
            for h in moe_block_handles:
                h.remove()
            for fqn, s in acc_stats.items():
                prev = merged_stats.get(fqn)
                if prev is None:
                    merged_stats[fqn] = dict(s)
                else:
                    prev["h_trace_raw"] += s.get("h_trace_raw", 0.0)
                    prev["h_w2_sum_raw"] += s.get("h_w2_sum_raw", 0.0)
                    prev["n_tokens_seen"] += s.get("n_tokens_seen", 0)
                    # Per-channel marginals: sums add, act_absmax maxes.
                    merge_marginals(prev, s)
                    # Per-expert Fisher is a list of floats on the packed
                    # stat entry; sum element-wise across shard splits.
                    per_prev = prev.get("h_trace_per_expert_raw")
                    per_new = s.get("h_trace_per_expert_raw")
                    if per_new is not None:
                        if per_prev is None:
                            prev["h_trace_per_expert_raw"] = list(per_new)
                        else:
                            prev["h_trace_per_expert_raw"] = [
                                a + b for a, b in zip(per_prev, per_new)
                            ]
                    # Per-group Fisher (grouped-BMM operands): identical
                    # elementwise-sum rule, same reason — it is a [G]
                    # decomposition of the one h_trace across shard splits.
                    grp_prev = prev.get("h_trace_per_group_raw")
                    grp_new = s.get("h_trace_per_group_raw")
                    if grp_new is not None:
                        if grp_prev is None:
                            prev["h_trace_per_group_raw"] = list(grp_new)
                        else:
                            prev["h_trace_per_group_raw"] = [
                                a + b for a, b in zip(grp_prev, grp_new)
                            ]
                    # Per-expert AQUA marginals follow the SAME merge
                    # rules as the dense ones: sums add, an absmax bound
                    # maxes. Kept beside `h_trace_per_expert_raw` rather
                    # than inside `merge_marginals` because those arrays
                    # are 1-D per-Linear and these are [E, *] per-expert.
                    for key in _PACKED_MARGINAL_KEYS:
                        new = s.get(key)
                        if new is None:
                            continue
                        old_v = prev.get(key)
                        if old_v is None:
                            prev[key] = np.asarray(new).copy()
                        elif key == "expert_act_absmax":
                            prev[key] = np.maximum(old_v, np.asarray(new))
                        else:
                            prev[key] = np.asarray(old_v) + np.asarray(new)
            if collect_h_full:
                for fqn, h in acc_h_full.items():
                    if fqn in merged_h_full:
                        merged_h_full[fqn].add_(h)
                    else:
                        merged_h_full[fqn] = h.clone()
                for fqn, parts in acc_g2_per_token.items():
                    if parts:
                        merged_g2_per_token[fqn].extend(parts)
            if packed_full_acc:
                detail_dir = Path(h_detail_dir)
                detail_dir.mkdir(parents=True, exist_ok=True)
                for local_key, tensor in packed_full_acc.items():
                    full_key = f"{layer_prefix}{local_key}"
                    fname = re.sub(r"[^A-Za-z0-9_-]", "__", full_key) + ".pt"
                    # Per-token units + explicit marker (audit M9): the
                    # accumulator is token-summed; normalize by the
                    # GLOBAL calib token count — the same denominator the
                    # scalar h_trace gets in finalize_fisher_stats.
                    # (Packed-3D rows count the full batch in
                    # n_tokens_seen, so this is numerically identical to
                    # the previous per-row count; global is used for
                    # uniformity with the per-expert-Linear writers.)
                    torch.save(
                        h_detail_blob(tensor, global_tokens,
                                      full_key, kind="packed"),
                        detail_dir / fname)
                packed_full_acc.clear()
            # Body layer FQNs are unique within the shard, so activation
            # snapshots can be flushed as soon as that layer has run.
            # Holding every target expert's sampled inputs until shard
            # finalization adds several GB of avoidable host pressure on
            # MiniMax's 256-expert layers.
            flush_activation_snapshots(activation_snaps, activation_row_indices)
            flush_activation_snapshots(packed_act_snaps)

            phase_pressure_trim_bytes += int(ctx.unload(L) or 0)
            # The `del` drops all per-layer refs; CPython ref counting
            # reclaims them synchronously. The per-layer gc + empty_cache
            # is a quick win for in-scope MoE layers where 768 hooks have
            # accumulated cached allocator blocks — releasing them after
            # each in-scope layer prevents the cumulative-residue OOM
            # we hit at the L7 transition. For out-of-scope layers (no
            # hooks fired), the empty_cache is essentially a no-op.
            del x_in, out, saved_inputs, acc_stats, acc_h_full, acc_g2_per_token, handles
            if L in in_scope_layers:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Periodic gc — every 4 in-scope layers — picks up any
                # ref-cycles before they snowball.
                if L % 4 == 0:
                    import gc as _gc
                    _gc.collect()

            if L % 8 == 0 or L == 0 or L == num_layers - 1:
                print(f"[incremental] bwd L{L:02d}  src={src}  load={load_s:.2f}s  "
                      f"bwd={bwd_s:.2f}s", flush=True)

        load_parts = ", ".join(
            f"{k}:{load_by_src[k]:.1f}s/{count_by_src[k]}"
            for k in sorted(load_by_src)
        )
        print(f"[incremental] phase-3 reverse sweep "
              f"[{num_layers-1}->{stop_L}]: {time.time()-t_phase:.1f}s  "
              f"load={phase_load_s:.1f}s bwd={phase_bwd_s:.1f}s "
              f"pressure_trim={phase_pressure_trim_bytes/(1024**3):.1f}GB "
              f"load_by_src=[{load_parts}]  "
              f"{ctx.layer_cache.summary()}  {ctx.prefetch_summary()}",
              flush=True)
        # Report the KV-cotangent path only when it did something, plus any
        # cotangent no producer claimed (which would mean a residual
        # under-count on that producer's k/v_proj — worth seeing, never
        # silently swallowed).
        if kv_cotangents.n_grafted or kv_cotangents.pending_keys():
            print(f"[incremental] {kv_cotangents.summary()}", flush=True)
        _print_mem_snapshot("phase-3 done")

        # `activations_cpu` is a shared reference into the global
        # precompute; do not free it here — the caller reuses across
        # shards. `grad_at_tail` / `grad_out` are per-shard device copies.
        del grad_at_tail, grad_out

    # ---- Finalize ----
    # One shared denominator for every row — the global calib token count
    # (hoisted above; see finalize_fisher_stats for why per-row
    # n_tokens_seen is wrong for routed-expert Linears).
    finalize_fisher_stats(merged_stats, global_tokens)

    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    if detail_dir is not None:
        detail_dir.mkdir(parents=True, exist_ok=True)
        for fqn, h in merged_h_full.items():
            fname = re.sub(r"[^A-Za-z0-9_-]", "__", fqn) + ".pt"
            g2_parts = merged_g2_per_token.get(fqn, [])
            g2_per_token = (
                torch.cat(g2_parts, dim=0).to(torch.float32).cpu()
                if g2_parts else torch.empty(0, dtype=torch.float32)
            )
            # Per-token units + explicit marker (audit M9): this writer
            # used to save the raw token-summed accumulator under "H",
            # leaving HDetailIndex consumers ~n_tokens× hotter than
            # blobs from sensitivity_probe. h_detail_blob normalizes by
            # the GLOBAL calib token count — the same denominator the
            # scalar h_trace gets in finalize_fisher_stats above, so
            # predicted_dloss fallback rows built from these blobs stay
            # on the scalar's scale (per-expert nn.Linear rows only see
            # their ROUTED tokens; dividing by that per-row count left
            # the detail (global/routed)× hotter than the scalar).
            # g2_per_token stays raw (it is already a per-token vector).
            torch.save(
                h_detail_blob(h, global_tokens, fqn, kind="linear",
                              g2_per_token=g2_per_token),
                detail_dir / fname,
            )

    # Flush activation snapshots.
    if cache_dir is not None:
        flush_activation_snapshots(activation_snaps, activation_row_indices)
        flush_activation_snapshots(packed_act_snaps)
        cache_dtype = (torch.float32
                       if os.environ.get("PRISMAQUANT_ACT_CACHE_FP32", "1") != "0"
                       else torch.bfloat16)
        for name, snaps in resident_act_snaps.items():
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to(cache_dtype).contiguous()
            row_parts = resident_act_row_indices.get(name, [])
            row_indices = (
                torch.cat(row_parts, dim=0).to(torch.long).contiguous()
                if row_parts else None
            )
            payload = {"inputs": X, "name": name}
            if row_indices is not None and row_indices.numel() == X.shape[0]:
                payload["row_indices"] = row_indices
            fname = act_fname_sub.sub("__", name) + ".pt"
            torch.save(payload, cache_dir / fname)
        # v22 Fix C: block until any async writes have completed so the
        # cost step sees a fully-flushed activation cache directory.
        drain_activation_writes()

    # Filter precomputed expert_info to the subset of routers whose experts are
    # within this shard's include-regex scope.
    shard_expert_info = {
        k: v for k, v in precomputed.expert_info.items() if k in all_tracked
    }
    shard_routers_in_scope: set[str] = {
        rq for (rq, _eid) in shard_expert_info.values()
    }
    # Packed Qwen3.5/3.6 experts have no per-expert nn.Linear leaves, so
    # ``expert_info`` is empty even though Phase 1 records their sibling
    # routers. Select those routers directly by this shard's layer regex.
    shard_routers_in_scope.update(
        rq for rq in precomputed.router_counts if inc.search(rq)
    )
    shard_router_counts = {
        rq: per_expert_map
        for rq, per_expert_map in precomputed.router_counts.items()
        if rq in shard_routers_in_scope
    }
    shard_router_totals = {
        rq: total
        for rq, total in precomputed.router_totals.items()
        if rq in shard_routers_in_scope
    }
    shard_router_active_counts = {
        rq: per_expert_map
        for rq, per_expert_map in precomputed.router_active_counts.items()
        if rq in shard_routers_in_scope
    }
    shard_expert_route_stats = {
        rq: stats
        for rq, stats in precomputed.expert_route_stats.items()
        if rq in shard_routers_in_scope
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": merged_stats,
            "router_counts": shard_router_counts,
            "router_totals": shard_router_totals,
            "router_active_counts": shard_router_active_counts,
            "expert_route_stats": shard_expert_route_stats,
            "expert_info": shard_expert_info,
            "meta": {
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": int(calib.size(0)),
                # R14: calibration identity, so held-out disjointness is
                # verifiable from the artifact instead of resting on the
                # driver passing the right --calib-skip-first.
                "calib_hash": calibration_data_hash(calib),
                "seqlen": seqlen,
                "dtype": dtype_name,
                "device_map": "streaming-layerwise",
                "execution_device": str(device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": activation_cache_dir,
                "h_detail_dir": h_detail_dir,
                "activation_rows_limit": int(activation_rows_limit),
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                # Marker: h_trace/h_w2_sum/h_trace_per_expert are divided by
                # the GLOBAL calib token count (not per-row n_tokens_seen).
                "fisher_norm_tokens": global_tokens,
            },
        }, f)
    print(f"[incremental] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# MTP shard runner — synthesize MtpModule, load `mtp.*` weights from
# safetensors, run forward+backward, collect Fisher. The body model has
# to be forwarded once (streaming phase-1) to produce final hidden states;
# no phase-3 reverse over body is needed since MTP gradients don't propagate
# back into the body.
# ---------------------------------------------------------------------------
def _run_mtp_streaming_shard(
    ctx: StreamingContext,
    *,
    calib: torch.Tensor,
    linear_include: str,
    linear_exclude: str,
    importance_weighting: bool,
    activation_cache_dir: str | None,
    h_detail_dir: str | None,
    output_path: str,
    dataset_name: str,
    dtype_name: str,
    seqlen: int,
    model_path: str,
    prefetch_lookahead: int = 3,
    activation_rows_limit: int = 256,
    precomputed: GlobalPrecompute | None = None,
):
    # Lazy import to avoid depending on transformers subpath at module load.
    from .model_profiles import profile_from_model as _profile_from_model

    if precomputed is None:
        raise ValueError(
            "_run_mtp_streaming_shard requires precomputed Phase-1 activations; "
            "call _compute_global_precompute first")

    device = ctx.device
    dtype = ctx.dtype
    model = ctx.model
    base_model = ctx.base_model

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)

    # --- Reuse globally-cached body forward activations ------------------
    # `activations_cpu[0]` is the embed output (== inputs_embeds).
    # `activations_cpu[-1]` is the hidden state at the tail of the body
    # (pre-`base_model.norm`). MTP needs the post-norm body hidden — cheap
    # to compute on CPU/device without re-running the body forward.
    t_phase = time.time()
    inputs_embeds_cpu = precomputed.activations_cpu[0]
    with torch.no_grad():
        pre_norm = precomputed.activations_cpu[-1].to(device).to(dtype)
        body_final_cpu = _get_final_norm(base_model)(pre_norm).detach().cpu()
        del pre_norm
    print(f"[incremental/mtp] body forward reused from global precompute "
          f"(norm only: {time.time()-t_phase:.1f}s)", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Synthesize MTP module, load its weights from safetensors ---
    # Both the module layout and the checkpoint prefix come from the
    # model profile (`build_mtp_module` / `mtp_source_prefix`); wrapping
    # in a parent named `mtp` is what makes the qualified names equal
    # the allocator's recipe names.
    mtp_profile = _profile_from_model(model)
    text_config = model.config
    inner_mtp = mtp_profile.build_mtp_module(text_config)
    if inner_mtp is None:
        raise RuntimeError(
            f"profile '{mtp_profile.name}' declares has_mtp() but "
            f"build_mtp_module() returned None — the MTP shard cannot be "
            f"probed. Either implement build_mtp_module() or set "
            f"has_mtp() -> False.")
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device, dtype=dtype)
    mtp_wrapper.eval()

    raw = mtp_profile.read_mtp_source_state_dict(model_path)
    if not raw:
        # No MTP weights in source — write empty pickle to satisfy the
        # schedule and return. Mirrors the text-only visual fallback.
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
                    "nsamples": int(calib.size(0)),
                    # R14: calibration identity, so held-out disjointness is
                    # verifiable from the artifact instead of resting on the
                    # driver passing the right --calib-skip-first.
                    "calib_hash": calibration_data_hash(calib),
                    "seqlen": seqlen,
                    "dtype": dtype_name,
                    "execution_device": str(device),
                    "linear_include": linear_include,
                    "linear_exclude": linear_exclude,
                    "h_detail_dir": h_detail_dir,
                    "activation_rows_limit": max(1, int(activation_rows_limit)),
                    "skipped_reason": "no MTP weights in source",
                },
            }, f)
        print(f"[incremental/mtp] no MTP weights; wrote empty shard "
              f"pickle to {output_path}", flush=True)
        return
    missing, extra = mtp_profile.load_mtp_state_dict(inner_mtp, raw)
    loaded = len(raw) - len(missing)
    print(f"[incremental/mtp] loaded {loaded}/{len(raw)} mtp weights "
          f"(missing={len(missing)}, module_params_unset={len(extra)})",
          flush=True)
    if missing:
        print(f"[incremental/mtp] unmatched checkpoint keys (first 5): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)

    # Freeze every leaf; Fisher hooks capture ||grad_w||² without
    # retaining leaf .grads.
    for p in mtp_wrapper.parameters():
        p.requires_grad_(False)

    tracked = [n for n, m in mtp_wrapper.named_modules()
               if isinstance(m, nn.Linear) and not re.search(r"mlp\.gate$", n)]
    print(f"[incremental/mtp] tracking {len(tracked)} MTP Linears", flush=True)

    expert_info_all = discover_moe_structure(mtp_wrapper, profile=mtp_profile)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(mtp_wrapper, default=2)

    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    input_rows_limit = max(1, int(activation_rows_limit))
    acc = FisherAccumulator(
        mtp_wrapper,
        tracked,
        expert_info,
        cache_dir,
        input_rows=input_rows_limit,
        h_detail_dir=detail_dir,
    )

    # lm_head lives on the body model (resident).
    lm_head = model.get_output_embeddings()
    assert isinstance(lm_head, nn.Linear), "lm_head must be Linear for MTP CE"

    from transformers.masking_utils import create_causal_mask

    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids_i = calib[i:i + 1].to(device)
        t0 = time.time()
        embed_i = inputs_embeds_cpu[i:i + 1].to(device, dtype=dtype)
        body_hidden_i = body_final_cpu[i:i + 1].to(device, dtype=dtype)

        shifted_embed = embed_i[:, 1:-1, :].contiguous()
        shifted_hidden = body_hidden_i[:, :-2, :].contiguous()
        target_ids = ids_i[:, 2:].contiguous()
        B, T2, _ = shifted_embed.shape
        trimmed_pos_ids = torch.arange(T2, device=device).view(1, T2).expand(B, T2)
        causal_mask_t2 = create_causal_mask(
            config=text_config,
            inputs_embeds=shifted_embed,
            attention_mask=None,
            past_key_values=None,
            position_ids=trimmed_pos_ids,
        )
        rot_pos = trimmed_pos_ids.view(1, B, T2).expand(3, B, T2)
        pos_emb_t2 = base_model.rotary_emb(shifted_embed, rot_pos)

        shifted_hidden = shifted_hidden.detach().requires_grad_(True)
        shifted_embed = shifted_embed.detach().requires_grad_(True)

        inner_mtp.train()
        out_hidden = inner_mtp(
            inputs_embeds=shifted_embed,
            body_hidden_states=shifted_hidden,
            position_embeddings=pos_emb_t2,
            causal_mask=causal_mask_t2,
            position_ids=trimmed_pos_ids,
        )
        logits = lm_head(out_hidden)
        t_fwd += time.time() - t0

        t0 = time.time()
        # .float() before log_softmax: matches the phase-2 chunked CE
        # sites (which cast lm_head output to fp32 first) — bf16
        # log_softmax costs ~0.4% rel on the Fisher CE gradient.
        lp = F.log_softmax(logits.reshape(-1, logits.size(-1)).float(),
                           dim=-1)
        gather = -lp.gather(1, target_ids.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                mean = float(gather.mean().item())
            w = (gather.detach() / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).sum()
        else:
            loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        n_tok = max(int(gather.numel()), 1)
        mean_loss = float(loss.detach().item()) / n_tok
        print(f"[incremental/mtp] sample {i+1}/{calib.size(0)} "
              f"loss={mean_loss:.3f} fwd_avg={t_fwd/(i+1):.2f}s "
              f"bwd_avg={t_bwd/(i+1):.2f}s", flush=True)

        del out_hidden, logits, loss, gather
        acc._saved_inputs.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Global calib token count, matching the meta nsamples×seqlen product
    # below (the MTP shift trims 2 tokens/sample — a uniform constant, and
    # keeping the meta product keeps the allocator renorm idempotent).
    fisher_norm_tokens = max(int(calib.size(0)) * int(seqlen), 1)
    acc.finalize(tracker=None, global_tokens=fisher_norm_tokens)
    acc.remove_hooks()

    renamed = dict(acc.stats)
    expert_info_renamed = dict(expert_info)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "stats": renamed,
            "router_counts": {},
            "router_totals": {},
            "router_active_counts": {},
            "expert_route_stats": {},
            "expert_info": expert_info_renamed,
            "meta": {
                "model": model_path,
                "dataset": dataset_name,
                "nsamples": int(calib.size(0)),
                # R14: calibration identity, so held-out disjointness is
                # verifiable from the artifact instead of resting on the
                # driver passing the right --calib-skip-first.
                "calib_hash": calibration_data_hash(calib),
                "seqlen": seqlen,
                "fisher_norm_tokens": fisher_norm_tokens,
                "dtype": dtype_name,
                "device_map": "streaming-layerwise",
                "execution_device": str(device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": activation_cache_dir,
                "h_detail_dir": h_detail_dir,
                "activation_rows_limit": input_rows_limit,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                "mtp_probe": True,
                "mtp_objective": "CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2})",
            },
        }, f)
    print(f"[incremental/mtp] wrote {output_path}", flush=True)

    # Free MTP before the next shard.
    del mtp_wrapper, inner_mtp, acc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k")
    ap.add_argument("--nsamples", type=int, default=0,
                    help="Calibration sample count. 0 (default since v26) "
                         "uses every line in the --dataset jsonl — useful "
                         "when the multi-chunk driver pre-shards the cal "
                         "data into per-chunk files and you want all of "
                         "each chunk consumed. Pass a positive integer to "
                         "truncate to the first N samples (smoke tests).")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--calib-seed", type=int, default=42,
                    help="Seed for calibration text-subset shuffle and "
                         "within-text window start position. Different seeds "
                         "give different sample subsets from the same dataset, "
                         "useful for multi-probe robust-Fisher experiments. "
                         "Default 42 reproduces historical behavior.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--work-dir", required=True,
                    help="Stores shard logs/pickles; safe to resume.")
    ap.add_argument("--layers-per-shard", default="1",
                    help='Int, or "auto" to derive from available RAM + model size.')
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Probe profile-declared MTP layers.")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Probe profile-declared visual encoder blocks.")
    ap.add_argument("--no-include-visual", action="store_false", dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Probe the profile-declared language-model head.")
    ap.add_argument("--no-include-lm-head", action="store_false", dest="include_lm_head")
    ap.add_argument("--h-detail-dir", default=None,
                    help="If set, write per-Linear full Fisher diagonal "
                         "(shape [out, in]) and per-packed-expert Fisher "
                         "(shape [E, M]) as .pt files in this directory. "
                         "measure_quant_cost reads them to compute the full "
                         "per-weight delta loss = 0.5 * <H, MSE_W> instead "
                         "of the scalar proxy. Omit to keep the legacy "
                         "scalar path.")
    ap.add_argument("--emit-marginals",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="Emit per-channel Fisher marginals alongside the "
                         "scalars: fisher_row [out], fisher_col [in], "
                         "g_sq_sum [out], act_sq_sum [in], act_absmax [in]. "
                         "Unlike the full [out, in] Fisher these are cheap "
                         "reductions of tensors the probe already forms; "
                         "memory is sum over Linears of "
                         "(2*out + 3*in) * 4 bytes. Default ON; overrides "
                         "PRISMAQUANT_PROBE_MARGINALS. --no-emit-marginals "
                         "restores byte-identical legacy output.")
    ap.add_argument("--unified-sweep", action="store_true", default=False,
                    help="Phase-3 in ONE reverse sweep through all 62 "
                         "layers, tracking ALL in-scope Linears at once "
                         "instead of N=ceil(num_layers/lps) per-shard "
                         "sweeps. ~16x reduction in disk reads + redundant "
                         "backward computation. Memory bounded by skipping "
                         "the per-weight h_full matrix accumulator (47k × "
                         "17 MB = 800 GB CPU, doesn't fit), keeping only "
                         "scalar h_trace + h_w2_sum. Cost stage falls "
                         "back to the scalar predicted_dloss formula "
                         "which preserves relative Linear ranking — the "
                         "load-bearing signal for the allocator's "
                         "format-choice DP. Forces --h-detail-dir off.")
    ap.add_argument("--prefetch-lookahead",
                    default=os.environ.get("PREFETCH_LOOKAHEAD", "auto"),
                    help="Number of layers to queue ahead in the disk "
                         "prefetch pool, or 'auto' to bound lookahead by "
                         "the layer-cache budget and estimated layer size.")
    ap.add_argument("--prefetch-workers",
                    default=os.environ.get("PREFETCH_WORKERS", "auto"),
                    help="Number of concurrent layer prefetch workers, or "
                         "'auto' to derive from cache budget and layer size.")
    ap.add_argument("--prefetch-min-available-gb",
                    default=os.environ.get("PREFETCH_MIN_AVAILABLE_GB", "auto"),
                    help="Pause scheduling new prefetches below this "
                         "available-memory floor, or 'auto' for two "
                         "estimated layers with an 8 GiB minimum.")
    ap.add_argument("--minimax-fast-moe", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Use chunked batched MiniMax-M2 expert replay for "
                         "non-measured layers during the probe reverse "
                         "sweep. Target layers still use the original "
                         "ModuleList path so per-Linear Fisher hooks fire.")
    ap.add_argument("--minimax-fast-moe-chunk-size", type=int, default=32,
                    help="Number of MiniMax experts to stack per batched "
                         "fast-MoE chunk. Larger chunks launch fewer "
                         "kernels but duplicate more expert weights "
                         "transiently on GPU/UMA memory.")
    ap.add_argument("--activation-rows-limit", type=int,
                    default=int(os.environ.get("ACTIVATION_ROWS_LIMIT", "256")),
                    help="Maximum sampled activation rows to keep per Linear "
                         "for the cost stage. Lower values are useful for "
                         "debug runs on very wide MoE checkpoints.")
    ap.add_argument("--calibration-modality",
                    choices=["text-only", "multimodal"],
                    default="text-only",
                    help="'text-only' (default) runs only the streaming body "
                         "Fisher probe; visual shards emit empty pickles and "
                         "the allocator's --visual-format override takes over. "
                         "'multimodal' also runs a second, non-streaming "
                         "pass that loads the full multimodal model "
                         "(vision_config preserved) and runs pixel_values + "
                         "text through a supervised CE backward. Real "
                         "per-visual-Linear Fisher + activation snapshots "
                         "land in the probe pickle + activation cache, so "
                         "the allocator treats visual Linears as regular DP "
                         "candidates and the exporter's GPTQ/AR passes "
                         "apply. Multimodal requires enough RAM for the full "
                         "model; on 122B-scale models it falls back to the "
                         "Phase 1 --visual-format override automatically on "
                         "OOM / load failure.")
    ap.add_argument("--mm-dataset", default="synthetic",
                    help="Dataset source for multimodal calibration. Accepts "
                         "a HuggingFace dataset id (e.g. `HuggingFaceM4/COCO`) "
                         "or `synthetic` (default: offline stub that exercises "
                         "the code path without network access).")
    ap.add_argument("--mm-nsamples", type=int, default=8,
                    help="Number of (image, caption) samples for the "
                         "multimodal calibration pass.")
    ap.add_argument("--mm-max-text-len", type=int, default=128,
                    help="Max text tokens per multimodal calibration sample.")
    args = ap.parse_args()
    # The marginal switch is read deep inside the hooks (and by shard
    # subprocesses), so the CLI flag lands on the env var the same way
    # allocator.py publishes --threads. Unset leaves the env default.
    if args.emit_marginals is not None:
        os.environ["PRISMAQUANT_PROBE_MARGINALS"] = (
            "1" if args.emit_marginals else "0")
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("incremental_probe", args.device)

    # MINOR-M33 (closed): KV-sharing models are probed normally now — the
    # reverse sweep seeds each producing layer's backward with the cotangent its
    # consumers accumulated on the borrowed K/V, so k_proj/v_proj h_trace is the
    # same quantity an end-to-end backward measures. The measurement gap was
    # closed rather than papered over (Principle 1). This guard only still
    # fires when PRISMAQUANT_KV_COTANGENT=0 takes that path away.
    _kv_block = kv_shared_fisher_block_reason(args.model)
    if _kv_block:
        raise SystemExit(_kv_block)

    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    # Resolve --layers-per-shard: int literal or "auto" (hardware-adaptive).
    lps_arg = str(args.layers_per_shard).strip()
    if lps_arg.lower() in ("auto", ""):
        from .autoscale import pick_layers_per_shard
        lps, lps_diag = pick_layers_per_shard(
            args.model, nsamples=args.nsamples, seqlen=args.seqlen,
        )
        print(f"[incremental] layers_per_shard=auto -> {lps} "
              f"(available={lps_diag.get('available_gb',0):.1f} GB, "
              f"per_layer_weight={lps_diag.get('per_layer_weight_gb',0):.2f} GB, "
              f"per_layer_active={lps_diag.get('per_layer_active_gb',0):.2f} GB, "
              f"cache_reserve={lps_diag.get('cache_reserve_gb',0):.1f} GB, "
              f"shard_budget={lps_diag.get('shard_budget_gb',0):.1f} GB)",
              flush=True)
        args.layers_per_shard = lps
    else:
        args.layers_per_shard = int(lps_arg)

    print("[incremental] minimax_fast_moe="
          f"{bool(args.minimax_fast_moe)} "
          f"chunk_size={args.minimax_fast_moe_chunk_size} "
          f"activation_rows_limit={args.activation_rows_limit}",
          flush=True)

    if args.unified_sweep and args.h_detail_dir:
        # h_detail off-switch must fire BEFORE schedule build so the
        # reusable-shard meta hash matches; the runners themselves only
        # see the final args.h_detail_dir.
        print("[incremental] --unified-sweep forces --h-detail-dir "
              "off (per-weight Fisher matrix would need ~800 GB CPU "
              "with all-Linears-at-once tracking)", flush=True)
        args.h_detail_dir = None

    schedule = build_shard_schedule(
        model_path=args.model,
        num_body_layers=n_layers,
        body_layers_per_shard=args.layers_per_shard,
        body_layer_range=(start, end),
        include_mtp=args.include_mtp,
        include_visual=args.include_visual,
        include_lm_head=args.include_lm_head,
        unified_body_sweep=args.unified_sweep,
    )
    shard_regexes = schedule.regexes()
    n_body = sum(1 for e in schedule if e.kind == "body")
    n_extras = len(schedule) - n_body
    if args.unified_sweep:
        # Approximate count of pre-collapse shards for the existing log line.
        pre_union = (end - start + args.layers_per_shard - 1) // args.layers_per_shard
        print(f"[incremental] --unified-sweep: collapsed {pre_union} "
              f"body shards into 1 union regex; phase-3 runs as a single "
              f"reverse sweep", flush=True)
    print(f"[incremental] shard regexes: {len(shard_regexes)} total "
          f"(body={n_body}, extras={n_extras})", flush=True)

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.activation_cache_dir).mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    exec_device = device  # streaming path runs on the requested device directly

    # Skip setup + calibration if every shard is reusable. Loading the
    # model+tokenizer is expensive — if the run is a no-op we want to
    # avoid paying that cost.
    shard_paths = [shard_dir / f"probe_shard_{i:03d}.pkl" for i in range(len(shard_regexes))]
    expected_metas = [
        _expected_probe_shard_meta(
            args,
            linear_include=linear_include,
            shard_idx=i,
            activation_cache_dir=args.activation_cache_dir,
        )
        for i, linear_include in enumerate(shard_regexes)
    ]
    all_reusable = all(
        shard_paths[i].exists()
        and probe_shard_is_reusable(shard_paths[i], expected_metas[i])
        for i in range(len(shard_regexes))
    )

    ctx: StreamingContext | None = None
    tokenizer = None
    calib: torch.Tensor | None = None
    resolved_prefetch_lookahead: int | None = None

    # Module-level cache: when set, the StreamingContext + tokenizer are
    # promoted into _PROBE_CTX_CACHE after first build and reused on
    # subsequent main() calls with the same model in the same process.
    # This is what makes the in-process multi-chunk driver fast — the
    # 244 GB BF16 source streaming offload setup + LayerCache survive
    # across chunks, so chunk_01..N hit warm caches.
    use_persistent = os.environ.get("PRISMAQUANT_PROBE_CTX_CACHE") == "1"

    def _ensure_ready():
        nonlocal ctx, tokenizer, calib
        if ctx is None and use_persistent:
            cached = _PROBE_CTX_CACHE.get((args.model, str(device), args.dtype))
            if cached is not None:
                ctx, tokenizer = cached
                # Reset accumulated state from prior chunks before reuse.
                # The in-process driver pins ~35 GB of allocator residue
                # without this — phase-3 backward then has too little
                # headroom for the MoE in-scope hooks.
                # v21 #4: PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK=1 keeps
                # layer-cache contents across chunks. Layer weights are
                # model-invariant; an entry that fit the budget at end
                # of chunk N is still valid for chunk N+1.
                retain = _env_flag(
                    "PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK", default=True)
                diag = ctx.reset_between_chunks(retain_cache=retain)
                if retain and diag.get("retained_cache_layers", 0):
                    print(f"[incremental] reused persistent ctx + tokenizer; "
                          f"between-chunk reset retained "
                          f"{diag['retained_cache_layers']} layers "
                          f"({diag['retained_cache_gb']:.1f} GB cache); "
                          f"freed {diag['freed_gb']:.1f} GB "
                          f"(avail {diag['before_avail_gb']:.0f}->{diag['after_avail_gb']:.0f} GB)",
                          flush=True)
                else:
                    print(f"[incremental] reused persistent ctx + tokenizer; "
                          f"between-chunk reset freed {diag['freed_gb']:.1f} GB "
                          f"(avail {diag['before_avail_gb']:.0f}->{diag['after_avail_gb']:.0f} GB)",
                          flush=True)
                _print_mem_snapshot("chunk start (post-reset)")
        if ctx is None:
            from transformers import AutoTokenizer
            staged = stage_text_only(args.model)
            tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
            offload_folder = str(work_dir / "streaming_offload")
            ctx = _build_streaming_context(
                args.model,
                device=device,
                dtype=dtype,
                offload_folder=offload_folder,
                prefetch_workers=args.prefetch_workers,
                prefetch_min_available_gb=args.prefetch_min_available_gb,
                log_prefix="[incremental]",
            )
            if use_persistent:
                _PROBE_CTX_CACHE[(args.model, str(device), args.dtype)] = (
                    ctx, tokenizer)
        # calib is always per-call (different chunks have different data)
        if calib is None:
            # v26: nsamples=0 means "use all lines in the dataset". The
            # prior default of 4 silently truncated multi-chunk runs that
            # pre-shard 12+ samples per chunk file. Compute the line
            # count up front when the user passed 0 so load_calibration
            # gets a positive count.
            ns = args.nsamples
            if ns == 0:
                ns_path = Path(args.dataset)
                if ns_path.exists() and ns_path.is_file():
                    with ns_path.open() as f:
                        ns = sum(1 for _ in f)
                if ns == 0:
                    ns = 4  # legacy fallback for non-jsonl datasets
            args.nsamples = ns  # write back so meta records the actual count
            calib = load_calibration(
                tokenizer, args.dataset, ns, args.seqlen,
                calib_seed=int(args.calib_seed))
            print(f"[incremental] calibration ready: {tuple(calib.shape)}",
                  flush=True)

    def _prefetch_lookahead() -> int:
        nonlocal resolved_prefetch_lookahead
        if resolved_prefetch_lookahead is not None:
            return resolved_prefetch_lookahead
        _ensure_ready()
        raw = str(args.prefetch_lookahead).strip().lower()
        if raw in ("", "auto"):
            resolved_prefetch_lookahead = ctx.suggest_prefetch_lookahead()
            print(f"[incremental] prefetch_lookahead=auto -> "
                  f"{resolved_prefetch_lookahead} "
                  f"({ctx.prefetch_summary()})", flush=True)
        else:
            resolved_prefetch_lookahead = max(1, int(raw))
            print(f"[incremental] prefetch_lookahead="
                  f"{resolved_prefetch_lookahead} (explicit)",
                  flush=True)
        return resolved_prefetch_lookahead

    # Union of all shard regexes — used for the global Phase-2 resident
    # Fisher hooks. We install hooks on every resident linear that ANY
    # shard's include regex would match; each per-shard runner filters
    # the captured dicts down to its own scope.
    linear_exclude = resolve_linear_exclude(args.model)
    resident_include_union = (
        "(?:" + "|".join(f"(?:{r})" for r in shard_regexes) + ")"
        if shard_regexes else r"(?!x)x"  # never-match fallback
    )

    precomputed: GlobalPrecompute | None = None
    precompute_cache_path = work_dir / "work" / "precomputed.pt"
    precompute_meta = _compute_precompute_key(
        model_path=args.model,
        dataset_name=args.dataset,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        dtype_name=args.dtype,
        device=str(device),
        importance_weighting=args.importance_weighting,
        resident_include_union=resident_include_union,
        # Resident marginals are written into the cached stats, so a
        # cache built with the flag off must not be reused with it on.
        emit_marginals=_marginals_enabled(),
    )

    def _ensure_precompute() -> GlobalPrecompute:
        """Load Phase-1/Phase-2 artifacts from the on-disk cache if the
        fingerprint matches; otherwise compute + persist + return."""
        nonlocal precomputed
        if precomputed is not None:
            return precomputed
        cached = _load_precompute_cache(
            precompute_cache_path, precompute_meta, device)
        if cached is not None:
            print(f"[incremental/global] reused precompute cache at "
                  f"{precompute_cache_path}", flush=True)
            precomputed = cached
            return precomputed
        _ensure_ready()
        # (Tied-embedding repair used to live here, with a hardcoded list
        # of embedding paths and a swallowed exception. It now happens
        # once, for every consumer of a streaming context, inside
        # `_build_streaming_context` via
        # `tied_embeddings.resolve_tied_output_embedding`.)
        precomputed = _compute_global_precompute(
            ctx,
            calib=calib,
            importance_weighting=args.importance_weighting,
            prefetch_lookahead=_prefetch_lookahead(),
            minimax_fast_moe=args.minimax_fast_moe,
            minimax_fast_moe_chunk_size=args.minimax_fast_moe_chunk_size,
            resident_include_union=resident_include_union,
            resident_exclude=linear_exclude,
            activation_cache_dir=args.activation_cache_dir,
        )
        _save_precompute_cache(
            precompute_cache_path, precomputed, precompute_meta)
        print(f"[incremental/global] wrote precompute cache to "
              f"{precompute_cache_path}", flush=True)
        return precomputed

    # Linear-level reuse cache (LPS-invariant): union of per-Linear
    # Fisher stats from all existing shards that share the same
    # content-level meta (model, dataset, nsamples, seqlen, dtype,
    # importance_weighting, activation_cache_dir). This lets the probe
    # resume cleanly even when LAYERS_PER_SHARD changes between runs:
    # a new shard's regex-matched Linears may already exist under
    # different shard groupings on disk, and we can synthesize the new
    # shard pickle from that cache rather than recompute.
    content_meta_anchor = {
        "model": args.model,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        "requested_device": args.device,
        "requested_device_map": str(args.device_map),
        "importance_weighting": args.importance_weighting,
        "activation_cache_dir": str(Path(args.activation_cache_dir)),
        "linear_exclude": resolve_linear_exclude(args.model),
        "h_detail_dir": (str(Path(args.h_detail_dir))
                         if args.h_detail_dir else None),
        "activation_rows_limit": int(args.activation_rows_limit),
        "router_coverage_version": _ROUTER_COVERAGE_VERSION,
        "emit_marginals": _marginals_enabled(),
    }
    linear_cache = scan_cached_linear_stats(shard_dir, content_meta_anchor)
    if linear_cache:
        print(f"[incremental] linear cache: {len(linear_cache)} stats pooled "
              f"from prior shards (LPS-invariant reuse enabled)", flush=True)

    # v20 step 2: precompute mark_done trigger. After the last body
    # shard, all body-layer tensors can be released — only non-body
    # shards (visual, lm_head) remain and they don't load body layers.
    last_body_shard_idx = max(
        (e.shard_idx for e in schedule if e.kind == "body"), default=-1)
    body_layers_marked_done = False

    def _mark_body_done_once(reason: str):
        # v20 fix #3: mark_done must fire even when the last body
        # shard is reused/synthesized (continue-skipped the old
        # in-loop call). Hoisted to a helper so we can call from
        # the body→non-body transition AND from end-of-loop.
        nonlocal body_layers_marked_done
        if body_layers_marked_done or ctx is None:
            return
        if last_body_shard_idx < 0:
            return
        transitioned = ctx.layer_cache.mark_layers_done(
            schedule.body_layer_indices())
        body_layers_marked_done = True
        if transitioned:
            print(f"[incremental] mark_done ({reason}): {transitioned} body "
                  f"layers transitioned (refuse future puts; "
                  f"refused_so_far={ctx.layer_cache.refused_puts})",
                  flush=True)

    try:
        if not all_reusable:
            _ensure_ready()

        for shard_idx, linear_include in enumerate(shard_regexes):
            # v20 fix #3: when crossing the body→non-body boundary,
            # mark body layers done before the next (non-body) shard
            # runs so its memory pressure benefits from the freed
            # cache slots. Fires regardless of how the body shards
            # were processed (computed/reused/synthesized).
            if shard_idx > last_body_shard_idx and last_body_shard_idx >= 0:
                _mark_body_done_once("body→non-body transition")
            shard_path = shard_paths[shard_idx]
            expected_meta = expected_metas[shard_idx]
            if shard_path.exists() and probe_shard_is_reusable(shard_path, expected_meta):
                print(f"[incremental] reuse shard {shard_idx}: {shard_path}",
                      flush=True)
                continue

            # LPS-invariant reuse: try to synthesize this shard from
            # cached per-Linear stats pooled from other compatible
            # shards. Skip body+lm_head+mtp kinds only — visual/empty
            # shards don't have per-Linear stats to reuse.
            kind_for_synth = _classify_shard(linear_include)
            if kind_for_synth in ("body", "mtp", "lm_head") and linear_cache:
                if synthesize_shard_from_linear_cache(
                    linear_include=linear_include,
                    linear_exclude=content_meta_anchor["linear_exclude"],
                    cache=linear_cache,
                    expected_meta=expected_meta,
                    output_path=shard_path,
                    expected_layers=schedule[shard_idx].layer_indices,
                    layer_prefix=schedule[shard_idx].layer_prefix,
                ):
                    annotate_probe_shard(shard_path, expected_meta)
                    print(f"[incremental] synthesize shard {shard_idx} "
                          f"({kind_for_synth}): reused cached Linear stats "
                          f"→ {shard_path}", flush=True)
                    continue
            if shard_path.exists():
                print(f"[incremental] stale shard {shard_idx}: "
                      f"recomputing {shard_path}", flush=True)
            kind = _classify_shard(linear_include)
            print(f"[incremental] shard {shard_idx} ({kind}): "
                  f"include={linear_include!r}", flush=True)
            _ensure_ready()

            if kind == "body":
                pre = _ensure_precompute()
                _run_body_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=linear_exclude,
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    h_detail_dir=args.h_detail_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=_prefetch_lookahead(),
                    minimax_fast_moe=args.minimax_fast_moe,
                    minimax_fast_moe_chunk_size=args.minimax_fast_moe_chunk_size,
                    activation_rows_limit=args.activation_rows_limit,
                    precomputed=pre,
                )
            elif kind == "mtp":
                pre = _ensure_precompute()
                _run_mtp_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=linear_exclude,
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    h_detail_dir=args.h_detail_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=_prefetch_lookahead(),
                    activation_rows_limit=args.activation_rows_limit,
                    precomputed=pre,
                )
            elif kind == "lm_head":
                # The lm_head Fisher is collected naturally during the
                # global Phase-2 run: its chunked CE backward runs
                # lm_head's forward+backward, and the resident Fisher
                # hooks (installed before Phase-2) capture it. The body
                # runner then filters the cached resident dicts to this
                # shard's regex and writes the shard pickle.
                pre = _ensure_precompute()
                _run_body_streaming_shard(
                    ctx,
                    calib=calib,
                    linear_include=linear_include,
                    linear_exclude=linear_exclude,
                    importance_weighting=args.importance_weighting,
                    activation_cache_dir=args.activation_cache_dir,
                    h_detail_dir=args.h_detail_dir,
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=_prefetch_lookahead(),
                    precomputed=pre,
                )
            else:
                # visual blocks are stripped by text-only staging, so the
                # streaming body never installs them. Emit an empty pickle
                # so the shard slot stays in the merged output with matching
                # metadata. When --calibration-modality=multimodal the
                # post-loop multimodal probe pass fills these in with real
                # visual Linear Fisher + activation snapshots.
                print(f"[incremental] skip shard {shard_idx} ({kind}): "
                      f"streaming path text-only; multimodal second pass "
                      f"will overlay visual stats if enabled", flush=True)
                Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
                with open(shard_path, "wb") as f:
                    pickle.dump({
                        "stats": {},
                        "router_counts": {},
                        "router_totals": {},
                        "router_active_counts": {},
                        "expert_route_stats": {},
                        "expert_info": {},
                        "meta": {
                            "model": args.model,
                            "dataset": args.dataset,
                            "nsamples": args.nsamples,
                            "seqlen": args.seqlen,
                            "dtype": args.dtype,
                            "device_map": "streaming-layerwise",
                            "execution_device": str(device),
                            "importance_weighting": args.importance_weighting,
                            "activation_cache_dir": args.activation_cache_dir,
                            "linear_include": linear_include,
                            "linear_exclude": resolve_linear_exclude(args.model),
                            "shard_kind": kind,
                        },
                    }, f)
            annotate_probe_shard(shard_path, expected_meta)
            # Force-reclaim per-shard Python state (activation snapshot lists,
            # merged_stats dicts, autograd graph leaves) before the next shard
            # allocates its own. Without this, refcount-only cleanup leaves
            # ~12-20 GB of stale refs alive across iterations — empty_cache
            # alone can't release the underlying CUDA blocks because Python
            # still holds references. gc.collect() first breaks any cycles,
            # then empty_cache reclaims the CUDA caching allocator's free list.
            gc.collect()
            if exec_device.type == "cuda":
                torch.cuda.empty_cache()
            # MiniMax-M2's per-shard merged_h_full holds ~52 GB of fp32
            # CPU tensors (4 layers × 256 experts × 3 weights × 17 MB).
            # CPython's pymalloc + glibc malloc don't return mapped pages
            # to the OS after dict deletion, so MemAvailable doesn't
            # recover and the next shard hits OOM. malloc_trim(0) forces
            # glibc to release unused arena memory. No-op on platforms
            # without malloc_trim (the ctypes.CDLL fails gracefully).
            try:
                import ctypes
                _libc = ctypes.CDLL("libc.so.6", use_errno=False)
                _libc.malloc_trim.argtypes = [ctypes.c_size_t]
                _libc.malloc_trim.restype = ctypes.c_int
                _libc.malloc_trim(0)
            except Exception:
                pass
        # v20 fix #3: end-of-loop fallback if no non-body shards
        # followed (e.g., text-only run with --no-include-{mtp,visual,lm-head}).
        # The transition check above never fired, so mark body layers
        # done now to refuse stale prefetches before the chunk ends.
        _mark_body_done_once("end of shard loop")
    finally:
        # v20 fix #2: under PRISMAQUANT_PROBE_CTX_CACHE=1 the ctx is
        # cached for chunk 1+ in the in-process driver. Shutting down
        # its prefetch_pool here would kill the executor and any
        # subsequent schedule_prefetch() in chunk 1 would raise
        # RuntimeError("cannot schedule new futures after shutdown").
        # Keep the ctx alive for the cache; reset_between_chunks()
        # handles per-chunk cleanup.
        if ctx is not None and not use_persistent:
            ctx.shutdown()

    # ---- Phase 2 multimodal visual probe (non-streaming second pass) ----
    # Runs after the streaming body / MTP / lm_head shards complete. Loads
    # the FULL multimodal model (vision_config preserved via stage_multimodal)
    # and captures per-visual-Linear Fisher + activation snapshots under the
    # same activation_cache_dir. The captured stats merge into the merged
    # probe pickle below so the allocator sees visual Linears as regular
    # DP candidates (if --visual-sensitivity=fisher).
    visual_probe_path: Path | None = None
    if args.calibration_modality == "multimodal":
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}
        mm_dtype = dtype_map[args.dtype]
        visual_probe_path = work_dir / "shards" / "probe_visual_mm.pkl"
        visual_include = r"^(?:model\.)?visual\."
        # Try the streaming path FIRST — it works on both small and huge
        # multimodal models (122B body streams; visual tower stays fully
        # resident). Fall back to the monolithic whole-model
        # `run_multimodal_visual_probe_pass` only if streaming fails
        # (e.g. unsupported architecture, missing processor).
        mm_offload = str(work_dir / "streaming_offload_mm")
        ok = run_streaming_multimodal_visual_probe_pass(
            args.model,
            dataset_name=args.mm_dataset,
            n_samples=args.mm_nsamples,
            max_text_len=args.mm_max_text_len,
            requested_device=args.device,
            dtype=mm_dtype,
            linear_include=visual_include,
            linear_exclude=linear_exclude,
            activation_cache_dir=args.activation_cache_dir,
            output_path=str(visual_probe_path),
            offload_folder=mm_offload,
            h_detail_dir=args.h_detail_dir,
        )
        if not ok:
            idx_path = Path(args.model) / "model.safetensors.index.json"
            total_size = 0
            if idx_path.exists():
                try:
                    with idx_path.open() as f:
                        total_size = int(
                            json.load(f).get("metadata", {}).get("total_size", 0)
                        )
                except Exception:
                    total_size = 0
            try:
                import psutil
                avail_bytes = int(psutil.virtual_memory().available)
            except Exception:
                avail_bytes = 0
            # The fallback loads the full multimodal model. On 122B-scale
            # checkpoints that is an OOM path, not a recovery path.
            if total_size and avail_bytes and total_size > int(avail_bytes * 0.75):
                print("[incremental] streaming multimodal probe failed; "
                      "skipping monolithic whole-model fallback because "
                      f"checkpoint total_size={total_size / (1024 ** 3):.1f} GiB "
                      f"exceeds 75% of available RAM="
                      f"{avail_bytes / (1024 ** 3):.1f} GiB", flush=True)
            else:
                print("[incremental] streaming multimodal probe failed; "
                      "trying monolithic whole-model fallback (fits only when "
                      "total model weights < RAM)", flush=True)
                ok = run_multimodal_visual_probe_pass(
                    args.model,
                    dataset_name=args.mm_dataset,
                    n_samples=args.mm_nsamples,
                    max_text_len=args.mm_max_text_len,
                    requested_device=args.device,
                    dtype=mm_dtype,
                    linear_include=visual_include,
                    linear_exclude=linear_exclude,
                    activation_cache_dir=args.activation_cache_dir,
                    output_path=str(visual_probe_path),
                    h_detail_dir=args.h_detail_dir,
                )
        if not ok:
            print("[incremental] multimodal visual probe skipped / failed; "
                  "allocator will need --visual-format for visual Linears",
                  flush=True)
            visual_probe_path = None

    all_pickles = list(shard_paths)
    if visual_probe_path is not None and visual_probe_path.exists():
        all_pickles.append(visual_probe_path)
    merge_probe_pickles(all_pickles, Path(args.output))
    # Body-coverage gate: a merged probe missing whole layers poisons
    # every downstream stage silently (cost skips them, the allocator
    # allocates around them, the export passes them through). Fail
    # fast here instead.
    with open(args.output, "rb") as _cf:
        _cov = pickle.load(_cf)
    _body_prefix = schedule[0].layer_prefix if len(schedule) else None
    if _body_prefix:
        _expected_cov = set()
        for _e in schedule:
            if _e.kind == "body":
                _expected_cov |= set(_e.layer_indices)
        _covered = set()
        _pat = re.compile(
            re.escape(_body_prefix.rstrip(".") + ".") + r"(\d+)\.")
        for _n in _cov.get("stats", {}):
            _m = _pat.search(str(_n))
            if _m:
                _covered.add(int(_m.group(1)))
        _missing_cov = sorted(_expected_cov - _covered)
        if _missing_cov:
            print(f"[incremental] FATAL: merged probe has NO stats for "
                  f"body layers {_missing_cov} — refusing to write a "
                  f"probe that would silently drop them downstream.",
                  flush=True)
            raise SystemExit(2)
    # Annotate the merged pickle with the calibration modality so
    # run-pipeline.sh's reuse guard (and any downstream tooling) can
    # reject a stale probe whose activations don't match the currently
    # requested modality. Written under the top-level `meta` dict so a
    # simple `pickle.load(...)['meta']['calibration_modality']` lookup
    # works.
    with open(args.output, "rb") as _f:
        _merged = pickle.load(_f)
    _meta = dict(_merged.get("meta", {}))
    _meta["calibration_modality"] = args.calibration_modality
    # Estimator provenance: packed-expert h_trace is the per-token
    # estimator since the 2026-07-02 M3 fix (pre-fix pickles carry the
    # sum-then-square 5-50x inflated values and are refused by
    # prepare_cost_context unless explicitly allowed).
    _meta["packed_fisher_estimator"] = "per_token_v2"
    _merged["meta"] = _meta
    with open(args.output, "wb") as _f:
        pickle.dump(_merged, _f)
    print(f"[incremental] wrote merged probe to {args.output} "
          f"(calibration_modality={args.calibration_modality})", flush=True)


if __name__ == "__main__":
    main()
