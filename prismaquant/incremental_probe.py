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

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_streaming import (
    _call_layer,
    _compute_position_embeddings,
    _make_causal_mask,
)
from .observers import (
    ExpertSaliencyTracker,
    saliency_from_moe_structure,
    saliency_from_packed_moe,
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


def _is_minimax_m2_experts_module(module: nn.Module) -> bool:
    return (
        type(module).__name__ == "MiniMaxM2Experts"
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
        and len(module) > 0
        and all(hasattr(module[0], n) for n in ("w1", "w2", "w3", "act_fn"))
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

    for experts in active.split(chunk_size):
        expert_list = [int(e) for e in experts.tolist()]
        chunk_counts = counts.index_select(0, experts)
        max_count = int(chunk_counts.max().item())
        if max_count == 0:
            continue

        start = int(offsets[int(experts[0])].item())
        end = int(offsets[int(experts[-1]) + 1].item())
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

        # REAP saliency accumulation (fast-MoE path). The chunked compute
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
                contribution_sq = gates64 * norms.pow(2)            # g·||f||² (REAP)
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
) -> int:
    """Enable/disable chunked batched MiniMax-M2 expert replay on a layer.

    Returns the number of MiniMax expert containers patched under `layer`.
    The patch is instance-local and falls back to the original forward
    whenever `_pq_fast_moe_enabled` is False.
    """
    patched = 0
    for module in layer.modules():
        if not _is_minimax_m2_experts_module(module):
            continue
        if not hasattr(module, "_pq_original_forward"):
            module._pq_original_forward = module.forward
            module.forward = types.MethodType(_minimax_fast_experts_forward, module)
        module._pq_fast_moe_enabled = bool(enabled)
        module._pq_fast_moe_chunk_size = int(chunk_size)
        patched += 1
    return patched


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


def build_extended_shard_regexes(
    model_path: str,
    layers_per_shard: int,
    *,
    include_body: bool = True,
    include_mtp: bool = True,
    include_visual: bool = True,
    include_lm_head: bool = True,
) -> list[str]:
    """Extended shard list covering everything quantizable in a
    multimodal Qwen3.5/3.6 checkpoint:

      - body transformer    (`model.layers.X.*`)         — N shards
      - MTP block(s)        (`mtp.layers.X.*`)           — 1 shard typically
      - visual ViT blocks   (`model.visual.blocks.X.*`)  — depth/N shards
      - lm_head             (`^lm_head$`)                — 1 shard
    """
    src_cfg_path = Path(model_path) / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    vis_cfg = cfg.get("vision_config", {})

    regexes: list[str] = []

    if include_body:
        n_body = int(text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers", 0)))
        regexes.extend(build_layer_shard_regexes(
            n_body, layers_per_shard, layer_prefix="model.layers"))

    if include_mtp:
        n_mtp_config = int(
            text_cfg.get("num_nextn_predict_layers")
            or cfg.get("num_nextn_predict_layers")
            or text_cfg.get("num_mtp_layers")
            or cfg.get("num_mtp_layers")
            or 0
        )
        n_mtp_actual = _count_mtp_layers_from_safetensors(model_path)
        # Empirical safetensors count is ground truth: a config may
        # declare MTP layers (inherited from a base) when the finetune
        # actually stripped the weights. Take the min so we never
        # schedule MTP shards the runner can't fulfill.
        n_mtp = min(n_mtp_config, n_mtp_actual) if n_mtp_actual > 0 else 0
        if n_mtp_config > 0 and n_mtp_actual == 0:
            print(f"[shard-schedule] config declares "
                  f"{n_mtp_config} MTP layer(s) but safetensors index "
                  f"has no `mtp.*` keys; skipping MTP shards "
                  f"(common on Qwen3.5/3.6 finetunes that strip MTP)",
                  flush=True)
        if n_mtp > 0:
            regexes.extend(build_layer_shard_regexes(
                n_mtp, layers_per_shard, layer_prefix="mtp.layers"))

    if include_visual:
        n_vis = int(vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0)
        if n_vis > 0:
            vis_per_shard = max(layers_per_shard, 4)
            regexes.extend(build_layer_shard_regexes(
                n_vis, vis_per_shard, layer_prefix="model.visual.blocks"))

    if include_lm_head:
        regexes.append(r"^lm_head$")

    return regexes


def _count_mtp_layers_from_safetensors(model_path: str) -> int:
    """Fallback for when the config doesn't carry an MTP layer count:
    scan the source safetensors index and count `mtp.layers.<N>.` paths."""
    src = Path(model_path)
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
                        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
                        if m:
                            mtp_indices.add(int(m.group(1)))
            return max(mtp_indices) + 1 if mtp_indices else 0
        except Exception:
            return 0
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_indices = set()
    for k in wm:
        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
        if m:
            mtp_indices.add(int(m.group(1)))
    return max(mtp_indices) + 1 if mtp_indices else 0


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged)
# ---------------------------------------------------------------------------
def _merge_nested_counts(dst: dict, src: dict):
    for key, sub in src.items():
        tgt = dst.setdefault(key, {})
        for sk, sv in sub.items():
            tgt[sk] = tgt.get(sk, 0.0) + float(sv)


def _read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


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


def annotate_probe_shard(path: Path, extra_meta: dict[str, Any]) -> None:
    data = _read_pickle(path)
    meta = dict(data.get("meta", {}))
    inc = dict(meta.get("incremental_shard", {}))
    inc.update(extra_meta)
    meta["incremental_shard"] = inc
    data["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(data, f)


# Fields whose equality makes two shards' per-Linear Fisher stats
# interchangeable. Notably excludes `linear_include` and `shard_idx` —
# those describe the shard *grouping*, not the Linear-level numbers.
# Swapping LAYERS_PER_SHARD between runs changes grouping but not
# numbers, so probe_shard pickles are safe to pool on these axes.
_CONTENT_META_KEYS: tuple[str, ...] = (
    "model", "dataset", "nsamples", "seqlen", "dtype",
    "requested_device", "requested_device_map",
    "importance_weighting", "activation_cache_dir",
    "linear_exclude", "h_detail_dir",
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
    merged_expert_info = {}
    # Each shard holds saliency for routers whose expert Linears fall
    # within that shard's scope. Shards are disjoint (the body-shard
    # filter by `linear_include` regex ensures this), so the merge is
    # just a dict-union on the router_qname key — no overlap handling.
    merged_expert_saliency: dict[str, dict[int, float]] = {}
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
        for rk, rv in data.get("router_totals", {}).items():
            merged_router_totals[rk] += int(rv)
        merged_expert_info.update(data.get("expert_info", {}))
        merged_expert_saliency.update(data.get("expert_saliency", {}))
        shard_metas.append(data.get("meta", {}))

    if merged is None:
        raise ValueError("no probe shards to merge")

    merged["stats"] = merged_stats
    merged["router_counts"] = dict(merged_router_counts)
    merged["router_totals"] = dict(merged_router_totals)
    merged["expert_info"] = merged_expert_info
    merged["expert_saliency"] = merged_expert_saliency
    merged["meta"] = {
        **merged.get("meta", {}),
        "incremental": True,
        "n_shards": len(paths),
        "shards": shard_metas,
    }

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
# dtype). Lets the multi_chunk_main driver reuse a single loaded model
# across N calibration chunks instead of paying the offload + tokenizer
# rebuild cost N times.
_PROBE_CTX_CACHE: dict = {}


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
    - `expert_saliency` is the REAP dropout-loss estimate per
      (router_qname, expert_id), harvested via
      `ExpertSaliencyTracker.saliency(reduction="reap_dropout")`.
      It is populated during Phase-1 forward and empty for dense
      models. See `prismaquant.observers.expert_saliency` for both
      the ranking reductions and the dropout-loss formula.
    - `expert_info` mirrors `sensitivity_probe.discover_moe_structure`'s
      output (Linear qname -> (router_qname, expert_id_str)); the
      allocator uses it to map per-Linear quantization decisions back to
      their owning expert when DROP candidates are considered.
    """
    activations_cpu: list[torch.Tensor]
    grad_at_tail: torch.Tensor
    ids: torch.Tensor  # shape (N, T), dtype long, on device
    resident_stats: dict[str, dict]
    resident_h_full: dict[str, torch.Tensor]
    resident_act_snaps: dict[str, list[torch.Tensor]]
    expert_saliency: dict[str, dict[int, float]]
    expert_info: dict[str, tuple[str, str]]
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
    layers = ctx.layers
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    tokens_in_sample = calib.size(-1)
    batch_size = calib.size(0)
    ids = calib.to(device)
    position_ids = torch.arange(tokens_in_sample, device=device).unsqueeze(0)
    causal_mask = _make_causal_mask(tokens_in_sample, device, dtype)

    prefetch_depth = prefetch_lookahead

    # ---- Phase 1: streaming forward, cache activations on CPU ----
    #
    # Alongside the forward we install an ExpertSaliencyTracker whose hooks
    # fire whenever any MoE router/expert module is called. The tracker
    # accumulates REAP dropout-loss saliency per expert — consumed later
    # by the allocator's DROP-candidate logic. Hooks remain registered
    # across the streaming install/unload cycle because they're bound to
    # Python module objects, not to weight tensors. Dense models get an
    # empty saliency dict at no cost.
    phase1_expert_info = discover_moe_structure(model)
    _saliency_top_k = read_top_k(model, default=2)
    _saliency_triples = saliency_from_moe_structure(phase1_expert_info)
    _packed_moe_blocks = saliency_from_packed_moe(model)
    saliency_tracker = (
        ExpertSaliencyTracker(
            model, _saliency_triples, top_k=_saliency_top_k,
            packed_moe_blocks=_packed_moe_blocks,
        )
        if (_saliency_triples or _packed_moe_blocks) else None
    )
    if saliency_tracker is not None:
        n_hooked = sum(len(v) for v in saliency_tracker.registered_experts().values())
        print(f"[incremental/global] expert-saliency tracker hooked "
              f"{n_hooked} experts across "
              f"{len(_saliency_triples)} nested routers + "
              f"{len(_packed_moe_blocks)} packed-3D routers "
              f"(top_k={_saliency_top_k})", flush=True)

    t_phase = time.time()
    with torch.no_grad():
        hidden = base_model.embed_tokens(ids).to(dtype)
    position_embeddings = _compute_position_embeddings(
        base_model, hidden, position_ids)
    print(f"[incremental/global] phase-1 N={batch_size} T={tokens_in_sample} "
          f"hidden={tuple(hidden.shape)}", flush=True)

    for d in range(prefetch_depth):
        ctx.schedule_prefetch(d)
    activations_cpu: list[torch.Tensor] = [hidden.detach().cpu()]
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
            )
        fwd_s = time.time() - fwd_t0
        hidden = out
        activations_cpu.append(hidden.detach().cpu())
        ctx.unload(L)
        if L % 8 == 0 or L == num_layers - 1:
            print(f"[incremental/global] fwd L{L:02d}  src={src}  "
                  f"load={load_s:.2f}s  fwd={fwd_s:.2f}s", flush=True)
    print(f"[incremental/global] phase-1 forward: {time.time()-t_phase:.1f}s  "
          f"{ctx.layer_cache.summary()}", flush=True)

    # Collect saliency and drop hooks before Phase-2 begins (Phase-2 replays
    # embeddings + final norm + lm_head only — no MoE modules fire — but we
    # free the hook handles proactively so there's no chance of ghost
    # accumulation during downstream forward passes in the same session).
    if saliency_tracker is not None:
        # Harvest as REAP dropout loss (Δ L_j ≈ (1/T) Σ g·||f||²).
        # Units match the allocator's Fisher·weight-MSE Δloss terms,
        # so the packed-prune path can use this directly as per-expert
        # prune cost with no α/h/n scaling.
        phase1_expert_saliency = saliency_tracker.saliency(reduction="reap_dropout")
        saliency_tracker.remove_hooks()
        nonzero_counts = sum(
            1 for rq_saliencies in phase1_expert_saliency.values()
            for s in rq_saliencies.values() if s > 0.0
        )
        total_slots = sum(len(v) for v in phase1_expert_saliency.values())
        print(f"[incremental/global] expert-saliency: "
              f"{nonzero_counts}/{total_slots} experts activated "
              f"(inactive experts get S=0, allocator may DROP free)", flush=True)
    else:
        phase1_expert_saliency = {}

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
    resident_saved_inputs: dict[str, torch.Tensor] = {}
    resident_handles: list = []
    resident_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    resident_act_rows: dict[str, int] = defaultdict(int)
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
                if need > 0:
                    flat = x.detach().reshape(-1, x.size(-1))
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                        flat = flat.index_select(0, idx)
                    resident_act_snaps[name].append(flat.to("cpu"))
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
    norm_out = base_model.norm(final_hidden)
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
        resident_act_snaps=dict(resident_act_snaps),
        expert_saliency=phase1_expert_saliency,
        expert_info=phase1_expert_info,
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
        "resident_act_snaps": pre.resident_act_snaps,
        "expert_saliency": pre.expert_saliency,
        "expert_info": pre.expert_info,
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
        resident_act_snaps=data["resident_act_snaps"],
        # Older caches predate expert saliency; fall back to empty dicts.
        expert_saliency=data.get("expert_saliency", {}),
        expert_info=data.get("expert_info", {}),
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
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
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

    # Activation snap accumulators (populated during Phase-3 for body
    # Linears; resident snaps were populated during Phase-2 hooks above).
    activation_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
    activation_rows: dict[str, int] = defaultdict(int)
    input_rows_limit = max(1, int(activation_rows_limit))
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    act_fname_sub = re.compile(r"[^A-Za-z0-9_-]")

    def flush_activation_snapshots(snaps_by_name: dict[str, list[torch.Tensor]]):
        if cache_dir is None:
            return
        for name in list(snaps_by_name.keys()):
            snaps = snaps_by_name.pop(name)
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
            fname = act_fname_sub.sub("__", name) + ".pt"
            torch.save({"inputs": X, "name": name}, cache_dir / fname)

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
        t_phase = time.time()
        phase_load_s = 0.0
        phase_bwd_s = 0.0
        load_by_src: dict[str, float] = defaultdict(float)
        count_by_src: dict[str, int] = defaultdict(int)
        grad_out = grad_at_tail
        # Smart cache: register in-scope (tracked) layers as priority so the
        # cache prefers evicting out-of-scope entries first. Also configure
        # pressure-triggered eviction (Task #3) so spikes during MoE hook
        # firing don't push the system to OOM. Threshold = the same floor
        # used by the prefetcher's pause check.
        in_scope_layers = {L for L in range(num_layers) if layer_linear_names[L]}
        ctx.layer_cache.set_priority_layers(in_scope_layers)
        ctx.layer_cache.configure_pressure_threshold(
            int(ctx.prefetch_min_available_bytes or 0))
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
            for block_name, block in layers[L].named_modules():
                full_block_name = f"{layers_prefix}{L}.{block_name}" if block_name else f"{layers_prefix}{L}"
                children = list(block.named_children())
                if not children or len(children) < 2:
                    continue
                ok = True
                for _, child in children:
                    for w in ("w1", "w2", "w3"):
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
                    for w in ("w1", "w2", "w3"):
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
                        if need > 0:
                            flat = x.detach().reshape(-1, x.size(-1))
                            if flat.size(0) > need:
                                idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                                flat = flat.index_select(0, idx)
                            activation_snaps[name].append(flat.to("cpu"))
                            activation_rows[name] += flat.size(0)
                return hook

            def make_bwd(name: str, mod_ref: nn.Linear):
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
                    acc_stats[name]["h_trace_raw"] += float(chunk_h.sum().item())
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
                            acc_stats[name]["h_w2_sum_raw"] += float(
                                (chunk_h * w.detach().float().pow(2)
                                 .to(chunk_h.device)).sum().item())
                    acc_stats[name]["n_tokens_seen"] += T
                return hook

            for fqn in tracked_here:
                mod = model.get_submodule(fqn)
                if not isinstance(mod, nn.Linear):
                    continue
                w = mod.weight
                if w.is_meta:
                    continue
                acc_stats[fqn] = {
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
                _set_minimax_fast_moe(
                    layers[L],
                    enabled=not layer_in_scope,
                    chunk_size=minimax_fast_moe_chunk_size,
                )
            packed_meta = install_packed_expert_hooks(
                layers[L], accumulator=packed_grad_acc,
                channel_accumulator=packed_channel_acc,
                full_accumulator=packed_full_acc,
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
            )
            out.backward(grad_out.to(device))
            bwd_s = time.time() - bwd_t0
            phase_bwd_s += bwd_s

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
            flush_activation_snapshots(activation_snaps)
            flush_activation_snapshots(packed_act_snaps)

            ctx.unload(L)
            # The `del` drops all per-layer refs; CPython ref counting
            # reclaims them synchronously. The per-layer gc + empty_cache
            # is a quick win for in-scope MoE layers where 768 hooks have
            # accumulated cached allocator blocks — releasing them after
            # each in-scope layer prevents the cumulative-residue OOM
            # we hit at the L7 transition. For out-of-scope layers (no
            # hooks fired), the empty_cache is essentially a no-op.
            del x_in, out, saved_inputs, acc_stats, acc_h_full, handles
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
              f"load_by_src=[{load_parts}]  "
              f"{ctx.layer_cache.summary()}  {ctx.prefetch_summary()}",
              flush=True)

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
            torch.save({"H": h, "name": fqn}, detail_dir / fname)

    # Flush activation snapshots.
    if cache_dir is not None:
        flush_activation_snapshots(activation_snaps)
        flush_activation_snapshots(packed_act_snaps)
        for name, snaps in resident_act_snaps.items():
            if not snaps:
                continue
            X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
            fname = act_fname_sub.sub("__", name) + ".pt"
            torch.save({"inputs": X, "name": name}, cache_dir / fname)

    # Filter precomputed expert_info / expert_saliency to the subset of
    # routers whose experts are within this shard's include-regex scope.
    # That keeps per-shard pickles small and self-consistent — the
    # allocator merges shards via _merge_nested_counts-style logic in
    # `_merge_probe_pickles` below.
    shard_expert_info = {
        k: v for k, v in precomputed.expert_info.items() if k in all_tracked
    }
    # expert_saliency is keyed by router_qname, not expert linear qname.
    # Two layouts coexist:
    #   (a) nested — each expert is an nn.Module leaf, so we can test
    #       "router's experts are tracked in this shard" via expert_info.
    #   (b) packed-3D — no per-expert leaves, so expert_info has no
    #       entry for the router. Check whether any of the shard's
    #       tracked names belong to this router's layer prefix
    #       (strip `.gate` → MoE-block qname → `.mlp.experts.*`
    #       packed stat lives under its layer).
    shard_routers_in_scope: set[str] = {
        rq for (rq, _eid) in shard_expert_info.values()
    }
    # Packed path: walk every router in the precompute and admit it when
    # the shard tracks a stat rooted at the same MoE-block qname prefix.
    # This is O(routers · tracked) but both are tiny (40 × ~40).
    for rq in precomputed.expert_saliency:
        if rq in shard_routers_in_scope:
            continue
        # router_qname conventionally ends in ".gate"; stripping that
        # gives the MoE-block qname. Admit the router when the shard
        # tracks ANY nn.Linear under the same MoE block (e.g. the
        # shared_expert's gate_proj lives at
        # "<block>.shared_expert.gate_proj" and IS tracked). The packed
        # experts 3D param isn't a Linear and so won't appear in
        # all_tracked directly.
        if not rq.endswith(".gate"):
            continue
        block_prefix = rq[: -len(".gate")] + "."
        for tracked in all_tracked:
            if tracked.startswith(block_prefix):
                shard_routers_in_scope.add(rq)
                break
    shard_expert_saliency = {
        rq: per_expert_map
        for rq, per_expert_map in precomputed.expert_saliency.items()
        if rq in shard_routers_in_scope
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": merged_stats,
            "router_counts": {},
            "router_totals": {},
            "expert_info": shard_expert_info,
            "expert_saliency": shard_expert_saliency,
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
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
            },
        }, f)
    print(f"[incremental] wrote {out_path}  "
          f"expert_saliency={sum(len(v) for v in shard_expert_saliency.values())} entries",
          flush=True)


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
    output_path: str,
    dataset_name: str,
    dtype_name: str,
    seqlen: int,
    model_path: str,
    prefetch_lookahead: int = 3,
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
        body_final_cpu = base_model.norm(pre_norm).detach().cpu()
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
                "expert_saliency": {},
                "meta": {
                    "model": model_path,
                    "dataset": dataset_name,
                    "nsamples": int(calib.size(0)),
                    "seqlen": seqlen,
                    "dtype": dtype_name,
                    "execution_device": str(device),
                    "linear_include": linear_include,
                    "linear_exclude": linear_exclude,
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

    expert_info_all = discover_moe_structure(mtp_wrapper)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(mtp_wrapper, default=2)

    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    acc = FisherAccumulator(mtp_wrapper, tracked, expert_info, cache_dir)

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
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
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
                    help="Probe MTP layers (`mtp.layers.X.*`).")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Probe visual encoder blocks (`model.visual.blocks.X.*`).")
    ap.add_argument("--no-include-visual", action="store_false", dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Probe lm_head (`^lm_head$`).")
    ap.add_argument("--no-include-lm-head", action="store_false", dest="include_lm_head")
    ap.add_argument("--h-detail-dir", default=None,
                    help="If set, write per-Linear full Fisher diagonal "
                         "(shape [out, in]) and per-packed-expert Fisher "
                         "(shape [E, M]) as .pt files in this directory. "
                         "measure_quant_cost reads them to compute the full "
                         "per-weight delta loss = 0.5 * <H, MSE_W> instead "
                         "of the scalar proxy. Omit to keep the legacy "
                         "scalar path.")
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
                         "candidates and the exporter's AWQ/GPTQ/AR passes "
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

    body_regexes = build_layer_shard_regexes(
        n_layers, args.layers_per_shard, layer_prefix="model.layers")
    first_shard = start // args.layers_per_shard
    last_shard = (end + args.layers_per_shard - 1) // args.layers_per_shard
    shard_regexes = body_regexes[first_shard:last_shard]

    extra = build_extended_shard_regexes(
        args.model, args.layers_per_shard,
        include_body=False,
        include_mtp=args.include_mtp,
        include_visual=args.include_visual,
        include_lm_head=args.include_lm_head,
    )
    shard_regexes = shard_regexes + extra
    print(f"[incremental] shard regexes: {len(shard_regexes)} total "
          f"(body={len(body_regexes[first_shard:last_shard])}, extras={len(extra)})",
          flush=True)

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
                diag = ctx.reset_between_chunks()
                print(f"[incremental] reused persistent ctx + tokenizer; "
                      f"between-chunk reset freed {diag['freed_gb']:.1f} GB "
                      f"(avail {diag['before_avail_gb']:.0f}->{diag['after_avail_gb']:.0f} GB)",
                      flush=True)
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
            calib = load_calibration(
                tokenizer, args.dataset, args.nsamples, args.seqlen)
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
    }
    linear_cache = scan_cached_linear_stats(shard_dir, content_meta_anchor)
    if linear_cache:
        print(f"[incremental] linear cache: {len(linear_cache)} stats pooled "
              f"from prior shards (LPS-invariant reuse enabled)", flush=True)

    try:
        if not all_reusable:
            _ensure_ready()

        for shard_idx, linear_include in enumerate(shard_regexes):
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
                    output_path=str(shard_path),
                    dataset_name=args.dataset,
                    dtype_name=args.dtype,
                    seqlen=args.seqlen,
                    model_path=args.model,
                    prefetch_lookahead=_prefetch_lookahead(),
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
    finally:
        if ctx is not None:
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
