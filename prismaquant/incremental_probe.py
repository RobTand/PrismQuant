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
regex. MTP is a built-in shard kind: after the body forward we synthesize
a `MtpModule`, load `mtp.*` weights directly from safetensors, and run
its own forward+backward for Fisher collection. The per-shard pickle
output format matches `sensitivity_probe.run_probe_pass` / `streaming_probe`
unchanged — the allocator consumes either.
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
import torch.nn as nn
import torch.nn.functional as F

from .layer_streaming import (
    _call_layer,
    _compute_position_embeddings,
    _get_final_norm,
    _make_causal_mask,
)
from .sensitivity_probe import (
    FisherAccumulator,
    RouterTracker,
    discover_moe_structure,
    install_packed_expert_hooks,
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
# MiniMax-M2 fast MoE replay
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


def _is_minimax_m2_experts_module(
    module: nn.Module, proj_names: tuple[str, ...] = ("w1", "w2", "w3")
) -> bool:
    return (
        type(module).__name__ == "MiniMaxM2Experts"
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
        and len(module) > 0
        and all(hasattr(module[0], n) for n in (*proj_names, "act_fn"))
    )


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
) -> int:
    """Enable/disable chunked batched MiniMax-M2 expert replay on a layer.

    Returns the number of MiniMax expert containers patched under `layer`.
    The patch is instance-local and falls back to the original forward
    whenever `_pq_fast_moe_enabled` is False. ``proj_names`` are the per-expert
    projection attribute names (from the model profile; default Qwen/MiniMax
    ``('w1','w2','w3')``).
    """
    patched = 0
    for module in layer.modules():
        if not _is_minimax_m2_experts_module(module, proj_names):
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
        "linear_exclude": (
            r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)"
        ),
        "h_detail_dir": str(Path(args.h_detail_dir)) if args.h_detail_dir else None,
        "activation_rows_limit": int(args.activation_rows_limit),
        "shard_idx": shard_idx,
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
                            resident_include_union: str) -> dict[str, Any]:
    """Fingerprint for the global precompute cache. If any of these
    inputs change, recompute; otherwise reuse the cached tensors."""
    return {
        "model": model_path,
        "dataset": dataset_name,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "dtype": dtype_name,
        "device": device,
        "importance_weighting": importance_weighting,
        "resident_include_union": resident_include_union,
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
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)

    prefetch_depth = prefetch_lookahead

    # Profile-driven hidden-state shape adapter (refactor #32). Default
    # profile passes through; DSv4 expands single-stream `[B, S, H]` to
    # multi-stream `[B, S, hc_mult, H]` (mirrors `DeepseekV4Model.forward`).
    from .model_profiles import profile_from_model as _profile_from_model
    _profile = _profile_from_model(base_model)

    # ---- Phase 1: streaming forward, cache activations on CPU ----
    phase1_expert_info = discover_moe_structure(model, profile=_profile)

    t_phase = time.time()
    with torch.no_grad():
        hidden = base_model.embed_tokens(ids).to(dtype)
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids)

    hidden = _profile.expand_hidden_for_layers(hidden, base_model)

    print(f"[incremental/global] phase-1 N={batch_size} T={tokens_in_sample} "
          f"hidden={tuple(hidden.shape)}", flush=True)

    for d in range(prefetch_depth):
        ctx.schedule_prefetch(d)
    # v22 Fix E1: keep activations on device through phase-1 to avoid
    # the per-layer .cpu() sync that stalls the forward pipeline. We
    # batch the device→host transfer at the END of phase-1 in a single
    # call. The pickled precompute cache (and downstream phase-3) want
    # CPU tensors, which we produce after the loop.
    device_acts: list[torch.Tensor] = [hidden.detach()]
    for L in range(num_layers):
        load_t0 = time.time()
        src = ctx.install(L)
        ctx.schedule_prefetch(L + prefetch_depth)
        load_s = time.time() - load_t0
        if minimax_fast_moe:
            _set_minimax_fast_moe(
                layers[L], True, chunk_size=minimax_fast_moe_chunk_size)
        fwd_t0 = time.time()
        with torch.no_grad():
            out = _call_layer(
                layers[L], hidden,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                **_profile.extra_layer_kwargs(input_ids=ids),
            )
        fwd_s = time.time() - fwd_t0
        hidden = out
        device_acts.append(hidden.detach())
        ctx.unload(L)
        if L % 8 == 0 or L == num_layers - 1:
            print(f"[incremental/global] fwd L{L:02d}  src={src}  "
                  f"load={load_s:.2f}s  fwd={fwd_s:.2f}s", flush=True)
    # v22 Fix E1: batched device→host transfer for the activations
    # captured during phase-1. All have the same (B, T, H) shape so we
    # stack into one (L+1, B, T, H) tensor and do a single .cpu() —
    # 62 individual transfers collapsed into one. After the copy lands,
    # we split back into a list of CPU tensors so the rest of the code
    # (precompute cache pickle, phase-3 reads) sees the original layout.
    t_h2h = time.time()
    stacked = torch.stack(device_acts, dim=0).cpu()
    activations_cpu: list[torch.Tensor] = [
        stacked[i].clone() for i in range(stacked.size(0))
    ]
    del device_acts, stacked
    print(f"[incremental/global] phase-1 forward: {time.time()-t_phase:.1f}s  "
          f"(host transfer {time.time()-t_h2h:.1f}s)  "
          f"{ctx.layer_cache.summary()}", flush=True)

    phase1_router_counts = {}
    phase1_router_totals = {}
    phase1_router_active_counts = {}
    phase1_expert_route_stats = {}

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
            # Trace via the outer-product-norm identity (avoids a second
            # full matmul, just two reductions of size T).
            resident_stats[name]["h_trace_raw"] += float(
                (gy2_sq.sum(dim=1) * x2_sq.sum(dim=1)).sum().item())
            w = mod_ref.weight
            if w is not None and not w.is_meta:
                resident_stats[name]["h_w2_sum_raw"] += float(
                    (chunk_h * w.detach().float().pow(2).to(chunk_h.device))
                    .sum().item())
            resident_stats[name]["n_tokens_seen"] += x2.size(0)
        return hook

    for fqn in resident_tracked:
        mod = model.get_submodule(fqn)
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight
        if w.is_meta:
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
    )


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

    top_k = read_top_k(model, default=2)

    merged_stats: dict[str, dict] = {}
    merged_h_full: dict[str, torch.Tensor] = {}
    merged_g2_per_token: dict[str, list[torch.Tensor]] = defaultdict(list)

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)

    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)

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
            base_model, embed0, position_ids)
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
        # Reverse-prefetch (Task #5): prefetcher should now look BACKWARD
        # in layer index since reverse sweep walks num_layers-1 → 0.
        # Schedule lookahead in the direction we're actually going.
        for d in range(prefetch_depth):
            ctx.schedule_prefetch(num_layers - 1 - d)

        for L in reversed(range(num_layers)):
            load_t0 = time.time()
            src = ctx.install(L)
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
                                del chunk_h_batch, per_token, trace_per_e
                    return flush

                moe_block_handles.append(
                    block.register_full_backward_hook(_make_flush(full_block_name)))
            if moe_linear_to_block:
                # Flag so the per-Linear hook short-circuits to deferred path.
                pass  # (no-op, used as documentation; lookup happens per-call)

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

            packed_grad_acc: dict[str, float] = {}
            # Per-expert per-channel Fisher [E, M] — enables per-expert
            # h_trace decomposition for the allocator's packed-3D prune
            # cost without re-measuring cost per expert. Always enabled
            # here; the accumulator's memory is ~1 MB per packed param
            # at 128 experts × 5760 channels, negligible on 121 GB RAM.
            packed_channel_acc: dict[str, torch.Tensor] = {}
            packed_full_acc: dict[str, torch.Tensor] | None = (
                {} if h_detail_dir is not None else None)
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
                _set_minimax_fast_moe(
                    layers[L],
                    enabled=not layer_in_scope,
                    chunk_size=minimax_fast_moe_chunk_size,
                    proj_names=tuple(_mmx_proj()) if callable(_mmx_proj) else ("w1", "w2", "w3"),
                )
            packed_meta = install_packed_expert_hooks(
                layers[L], accumulator=packed_grad_acc,
                channel_accumulator=packed_channel_acc,
                full_accumulator=packed_full_acc,
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
            )
            out.backward(grad_out.to(device))
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
                    gy2 = gy.reshape(-1, gy.size(-1))
                    x2 = x.reshape(-1, x.size(-1))
                    T = x2.size(0)
                    gy2_sq = gy2.pow(2)
                    x2_sq = x2.pow(2)
                    chunk_h = (gy2_sq.t() @ x2_sq).float()
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

            for local_key, raw in packed_grad_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                if full_key in acc_stats:
                    acc_stats[full_key]["h_trace_raw"] += float(raw)
                    acc_stats[full_key]["n_tokens_seen"] = \
                        acc_stats[full_key].get("n_tokens_seen", 0) + x_in.size(0) * x_in.size(1)
            # Per-expert Fisher trace decomposition. channel_acc[key] is
            # [E, M] (grad² summed over the in-feature dim); summing over
            # M collapses to [E] — per-expert Fisher trace. Stored as a
            # float list in the stat entry so it survives pickle + merge
            # without torch-device round-trips, and the allocator's
            # add_packed_prune_candidates reads it directly.
            for local_key, per_ch in packed_channel_acc.items():
                full_key = f"{layer_prefix}{local_key}"
                if full_key not in acc_stats:
                    continue
                # per_ch is on CPU fp32; summing over the last dim gives
                # per-expert trace without a device sync.
                per_expert_trace = per_ch.sum(dim=-1).to(torch.float64)
                prev = acc_stats[full_key].get("h_trace_per_expert_raw")
                if prev is None:
                    acc_stats[full_key]["h_trace_per_expert_raw"] = per_expert_trace.tolist()
                else:
                    summed = [p + float(q) for p, q in zip(prev, per_expert_trace.tolist())]
                    acc_stats[full_key]["h_trace_per_expert_raw"] = summed
            packed_channel_acc.clear()

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
                    torch.save({"H": tensor, "name": full_key},
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
        print(f"[incremental] phase-3 reverse sweep: {time.time()-t_phase:.1f}s  "
              f"load={phase_load_s:.1f}s bwd={phase_bwd_s:.1f}s "
              f"pressure_trim={phase_pressure_trim_bytes/(1024**3):.1f}GB "
              f"load_by_src=[{load_parts}]  "
              f"{ctx.layer_cache.summary()}  {ctx.prefetch_summary()}",
              flush=True)
        _print_mem_snapshot("phase-3 done")

        # `activations_cpu` is a shared reference into the global
        # precompute; do not free it here — the caller reuses across
        # shards. `grad_at_tail` / `grad_out` are per-shard device copies.
        del grad_at_tail, grad_out

    # ---- Finalize ----
    for s in merged_stats.values():
        tokens = max(s.get("n_tokens_seen", 1), 1)
        s["h_trace"] = s.get("h_trace_raw", 0.0) / tokens
        s["h_w2_sum"] = s.get("h_w2_sum_raw", 0.0) / tokens
        # Per-expert Fisher trace (only present on packed-3D stat entries;
        # dense Linears have no per-expert dimension). Normalize by the
        # same token count so it shares units with `h_trace`.
        per = s.get("h_trace_per_expert_raw")
        if per is not None:
            s["h_trace_per_expert"] = [float(v) / tokens for v in per]

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
            torch.save(
                {
                    "H": h,
                    "name": fqn,
                    "g2_per_token": g2_per_token,
                    "h_detail_version": 2,
                },
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
    from .mtp_module import MtpModule, _load_into_mtp, _load_mtp_state_dict

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
    text_config = model.config
    inner_mtp = MtpModule(text_config)
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device, dtype=dtype)
    mtp_wrapper.eval()

    raw = _load_mtp_state_dict(model_path)
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
    missing, extra = _load_into_mtp(inner_mtp, raw)
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

    from .model_profiles import profile_from_model as _profile_from_model
    mtp_profile = _profile_from_model(model)
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
        lp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
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

    acc.finalize(tracker=None)
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
                "seqlen": seqlen,
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
    linear_exclude = (
        r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
        r"block_sparse_moe\.gate$)"
    )
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
        # Tied-embedding repair: when `tie_word_embeddings=True` (Qwen
        # 3.5/3.6 small variants, Llama-3.2-1B/3B, etc.), the streaming
        # pipeline materializes embed_tokens but leaves lm_head on meta
        # because the source has no separate lm_head shard. Manually
        # alias lm_head.weight to the materialized embedding before
        # the precompute, otherwise model.lm_head(...) returns a meta
        # tensor and `.item()` fails.
        try:
            _model = ctx.model
            _cfg = getattr(_model, "config", None)
            if _cfg is not None and getattr(_cfg, "tie_word_embeddings", False):
                _embed = None
                for _path in ("model.embed_tokens",
                              "model.language_model.embed_tokens",
                              "transformer.wte"):
                    try:
                        _m = _model.get_submodule(_path)
                        if hasattr(_m, "weight") and not _m.weight.is_meta:
                            _embed = _m
                            break
                    except (AttributeError, KeyError):
                        continue
                if _embed is not None and hasattr(_model, "lm_head"):
                    if _model.lm_head.weight.is_meta:
                        _model.lm_head.weight = _embed.weight
                        print(f"[incremental] tied lm_head.weight ← "
                              f"embed_tokens.weight (meta repair)", flush=True)
        except Exception as _e:
            print(f"[incremental] WARN tied-embedding repair: {_e}",
                  flush=True)

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
        "linear_exclude": (
            r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|"
            r"block_sparse_moe\.gate$)"
        ),
        "h_detail_dir": (str(Path(args.h_detail_dir))
                         if args.h_detail_dir else None),
        "activation_rows_limit": int(args.activation_rows_limit),
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
                            "linear_exclude": (
                                r"(?:mlp\.gate$|mlp\..*gate$|"
                                r"\.router(?:$|\.)|block_sparse_moe\.gate$)"
                            ),
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
    _merged["meta"] = _meta
    with open(args.output, "wb") as _f:
        pickle.dump(_merged, _f)
    print(f"[incremental] wrote merged probe to {args.output} "
          f"(calibration_modality={args.calibration_modality})", flush=True)


if __name__ == "__main__":
    main()
