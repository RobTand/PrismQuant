"""L3 propagated-cost utilities.

This module owns the final-pass "L3 polish" path: select a small allocator
neighborhood around the converged L2 assignment, measure propagated costs for
that neighborhood, and re-solve only those measured choices while freezing the
rest of the L2 assignment.
"""
from __future__ import annotations

import inspect
import gc
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    _stats_indicates_packed_expert,
    cost_entry_predicted_dloss,
)
from prismaquant.allocator_solver import Candidate, _shape_from_stats, solve_allocation
from prismaquant.build_rtn_cache import kl_divergence
from prismaquant.memory_management import (
    GPUMemoryBudgetExceeded,
    cuda_memory_info,
    enforce_gpu_memory_budget,
    max_gpu_memory_bytes,
    register_budget_evictor,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    build_quantizable_map,
    calibration_data_hash,
    iter_calibration_forwards,
    _maybe_clip_activations,
)
from prismaquant.layer_state_cache import LayerHiddenStateCache

KLScope = Literal["last_token", "full_sequence"]


@dataclass(frozen=True)
class L3NeighborhoodEntry:
    name: str
    current_format: str
    formats: tuple[str, ...]
    margin: float
    l2_current_cost: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class FrozenBudgetError(RuntimeError):
    """Raised when frozen L2 choices make the L3 neighborhood infeasible."""


class L3UnsupportedTargetError(RuntimeError):
    """Raised when L3 selection reaches targets the hook path cannot measure."""


@dataclass(frozen=True)
class _LaneSpec:
    name: str
    fmt: str
    baseline_index: int | None
    is_baseline: bool


@dataclass
class QuantWeightCache:
    cache: dict[tuple[str, str], torch.Tensor]

    def get(self, module_name: str, fmt: str) -> torch.Tensor | None:
        seen: set[str] = set()
        for candidate in (fmt, fr.canonical_format_name(fmt), *fr.aliases_for(fmt)):
            if candidate in seen:
                continue
            seen.add(candidate)
            cached = self.cache.get((module_name, candidate))
            if cached is not None:
                return cached
        return None


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _empty_cache_each_replay_batch() -> bool:
    return _env_flag_enabled(
        "PRISMAQUANT_EMPTY_CACHE_EACH_REPLAY_BATCH",
        default=False,
    )


def resolve_kl_scope(kl_scope: KLScope | None = None) -> KLScope:
    """Resolve KL reduction scope, preserving the legacy env override.

    Passing an explicit ``kl_scope`` wins.  ``None`` means use
    ``PRISMAQUANT_FULL_SEQUENCE_KL`` for backward compatibility, with
    last-token KL as the default when the env var is unset.
    """
    if kl_scope is not None:
        if kl_scope not in {"last_token", "full_sequence"}:
            raise ValueError(
                "kl_scope must be 'last_token' or 'full_sequence', "
                f"got {kl_scope!r}"
            )
        return kl_scope
    return (
        "full_sequence"
        if _env_flag_enabled("PRISMAQUANT_FULL_SEQUENCE_KL", default=False)
        else "last_token"
    )


def _env_cuda_graphs_enabled_for_call_count(
    name: str,
    *,
    default: str | bool = "auto",
    call_count: int,
    min_calls: int,
) -> bool:
    """Return whether a CUDA graph path should run for this call pattern.

    L3 and coord-descent candidates are often one-shot graph keys. Capturing
    those graphs costs warmup + capture work without enough replays to pay it
    back, so the default is ``auto``: graph only when the same key is expected
    to run at least ``min_calls`` times. Explicit env values keep their force
    semantics: ``1``/``true`` force on, ``0``/``false`` force off.
    """

    value = os.environ.get(name)
    mode = default if value is None else value.strip().lower()
    if isinstance(mode, bool):
        return mode
    if mode in {"1", "true", "yes", "on", "force"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode != "auto":
        return bool(default) if isinstance(default, bool) else False

    threshold = _env_int(f"{name}_MIN_CALLS", int(min_calls))
    return int(call_count) >= max(int(threshold), 1)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return max(parsed, 0)


def _bytes_to_gb(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024 ** 3)


_L3_FROZEN_CACHE_MEMORY_NOTICE_EMITTED = False
_L3_PREQUANT_CACHE_MEMORY_NOTICE_EMITTED = False
_L3_PREQUANT_GROUP_SKIP_NOTICE_EMITTED = False


def _memory_status_disables_weight_cache(
    device: torch.device,
) -> tuple[bool, int, int, int, int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, 0, 0, 0, 0, 0
    budget = max_gpu_memory_bytes(device)
    info = cuda_memory_info(device)
    if budget is None or info is None:
        return False, 0, 0, 0, 0, 0
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    reserve_frac = _env_float(
        "PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION", 0.05)
    reserve_floor_gb = _env_float(
        "PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB", 2.0)
    reserve_bytes = max(
        int(total_bytes * max(reserve_frac, 0.0)),
        int(max(reserve_floor_gb, 0.0) * 1024 ** 3),
    )
    budget_slack = budget - used_bytes
    disabled = not (
        used_bytes < budget
        and free_bytes >= reserve_bytes
        and budget_slack >= reserve_bytes
    )
    return disabled, used_bytes, budget, free_bytes, total_bytes, reserve_bytes


def _maybe_disable_l3_frozen_cache_for_memory(
    device: torch.device,
    enabled: bool,
) -> bool:
    global _L3_FROZEN_CACHE_MEMORY_NOTICE_EMITTED
    if not enabled:
        return enabled
    disabled, used_bytes, budget, free_bytes, total_bytes, reserve_bytes = (
        _memory_status_disables_weight_cache(device)
    )
    if not disabled:
        return enabled
    if not _L3_FROZEN_CACHE_MEMORY_NOTICE_EMITTED:
        _L3_FROZEN_CACHE_MEMORY_NOTICE_EMITTED = True
        print(
            "[memory-aware] disabling L3 frozen weight cache: "
            f"used={_bytes_to_gb(used_bytes):.2f}GB "
            f"budget={_bytes_to_gb(budget):.2f}GB "
            f"free={_bytes_to_gb(free_bytes):.2f}GB "
            f"total={_bytes_to_gb(total_bytes):.2f}GB "
            f"reserve={_bytes_to_gb(reserve_bytes):.2f}GB; "
            "falling back to per-module quantize/restore",
            flush=True,
        )
    return False


def _maybe_disable_l3_prequant_cache_for_memory(
    device: torch.device,
    enabled: bool,
) -> bool:
    global _L3_PREQUANT_CACHE_MEMORY_NOTICE_EMITTED
    if not enabled:
        return enabled
    if device.type != "cuda" or not torch.cuda.is_available():
        return enabled
    budget = max_gpu_memory_bytes(device)
    info = cuda_memory_info(device)
    if budget is None or info is None:
        return enabled
    free_bytes, total_bytes = info
    used_bytes = total_bytes - free_bytes
    reserve_frac = _env_float(
        "PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION", 0.05)
    reserve_floor_gb = _env_float(
        "PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB", 2.0)
    reserve_bytes = max(
        int(total_bytes * max(reserve_frac, 0.0)),
        int(max(reserve_floor_gb, 0.0) * 1024 ** 3),
    )
    disabled = used_bytes >= budget or free_bytes < reserve_bytes
    if not disabled:
        return enabled
    if not _L3_PREQUANT_CACHE_MEMORY_NOTICE_EMITTED:
        _L3_PREQUANT_CACHE_MEMORY_NOTICE_EMITTED = True
        print(
            "[memory-aware] disabling L3 prequant weight cache: "
            f"used={_bytes_to_gb(used_bytes):.2f}GB "
            f"budget={_bytes_to_gb(budget):.2f}GB "
            f"free={_bytes_to_gb(free_bytes):.2f}GB "
            f"total={_bytes_to_gb(total_bytes):.2f}GB "
            f"reserve={_bytes_to_gb(reserve_bytes):.2f}GB; "
            "quantizing target weights on demand",
            flush=True,
        )
    return False


def _estimate_l3_quant_cache_bytes(
    model: nn.Module,
    neighborhood: list[L3NeighborhoodEntry],
    specs: list[fr.FormatSpec],
    *,
    skip_bf16: bool = True,
) -> int:
    quant_map = _l3_quantizable_map(model)
    seen: set[tuple[int, str, str]] = set()
    total = 0
    for entry in neighborhood:
        target = quant_map.get(entry.name)
        if target is None:
            continue
        linear, attr = target
        if not isinstance(linear, nn.Linear) or attr != "weight":
            continue
        weight = linear.weight
        weight_bytes = int(weight.numel()) * int(weight.element_size())
        for spec in specs:
            canonical = fr.canonical_format_name(spec.name)
            if skip_bf16 and canonical == "BF16":
                continue
            key = (id(linear), attr, canonical)
            if key in seen:
                continue
            seen.add(key)
            total += weight_bytes
    return int(total)


def _l3_prequant_group_cache_fits(
    model: nn.Module,
    group_entries: list[L3NeighborhoodEntry],
    specs: list[fr.FormatSpec],
    device: torch.device,
) -> bool:
    global _L3_PREQUANT_GROUP_SKIP_NOTICE_EMITTED
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    info = cuda_memory_info(device)
    if info is None:
        return True
    free_bytes, total_bytes = info
    reserve_bytes = max(
        int(total_bytes * max(
            _env_float("PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_FRACTION", 0.05),
            0.0,
        )),
        int(
            max(_env_float("PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_GB", 2.0), 0.0)
            * 1024 ** 3
        ),
    )
    cache_bytes = _estimate_l3_quant_cache_bytes(model, group_entries, specs)
    peak_multiplier = max(
        _env_float("PRISMAQUANT_L3_PREQUANT_CACHE_PEAK_MULTIPLIER", 3.0),
        1.0,
    )
    needed_bytes = int(cache_bytes * peak_multiplier)
    fits = int(free_bytes) >= reserve_bytes + needed_bytes
    if not fits and not _L3_PREQUANT_GROUP_SKIP_NOTICE_EMITTED:
        _L3_PREQUANT_GROUP_SKIP_NOTICE_EMITTED = True
        print(
            "[memory-aware] disabling L3 prequant weight cache for a depth group: "
            f"free={_bytes_to_gb(free_bytes):.2f}GB "
            f"reserve={_bytes_to_gb(reserve_bytes):.2f}GB "
            f"estimated_cache={_bytes_to_gb(cache_bytes):.2f}GB "
            f"peak_multiplier={peak_multiplier:.2f}; "
            "quantizing group weights on demand",
            flush=True,
        )
    return fits


def _host_available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _enforce_l3_host_memory_floor(*, phase: str, chunk_index: int | None = None) -> None:
    floor_gb = _env_float("PRISMAQUANT_L3_MIN_HOST_MEM_GB", 0.0)
    if floor_gb <= 0.0:
        return
    available_gb = _host_available_memory_gb()
    if available_gb is None or available_gb >= floor_gb:
        return
    chunk_text = "" if chunk_index is None else f" chunk={chunk_index}"
    raise GPUMemoryBudgetExceeded(
        f"{phase}{chunk_text}: host MemAvailable {available_gb:.1f}GB "
        f"is below PRISMAQUANT_L3_MIN_HOST_MEM_GB={floor_gb:.1f}GB"
    )


def _adjust_l3_max_lanes_for_host_floor(
    max_lanes_per_batch: int,
    *,
    phase: str,
    chunk_index: int | None = None,
) -> int:
    max_lanes = max(int(max_lanes_per_batch), 2)
    if max_lanes % 2:
        max_lanes -= 1
    floor_gb = _env_float("PRISMAQUANT_L3_MIN_HOST_MEM_GB", 0.0)
    if floor_gb <= 0.0:
        return max(max_lanes, 2)

    available_gb = _host_available_memory_gb()
    if available_gb is None:
        return max(max_lanes, 2)
    margin_gb = available_gb - floor_gb
    if margin_gb < 4.0:
        _enforce_l3_host_memory_floor(
            phase=phase,
            chunk_index=chunk_index,
        )
    if margin_gb < 16.0:
        max_lanes = min(max_lanes, 2)
    elif margin_gb < 32.0:
        max_lanes = min(max_lanes, 4)
    elif margin_gb < 48.0:
        max_lanes = min(max_lanes, 6)
    return max(max_lanes, 2)


_PRISMAQUANT_GRAPH_POOL = None
_NOCLONE_OVERRIDE_WARNED = False


def get_prismaquant_graph_pool():
    """Return the process-wide CUDA graph memory pool, or None when disabled.

    Default ON: in-process tests confirm shared and private pools produce
    bit-identical captured-graph outputs (single graph and multi-registry).
    The earlier "5% NaN with shared pool" signal was process-init noise of
    the small Qwen-0.6B smoke -- 5 separate processes with identical config
    produce 5 different KLs whether the pool is shared or private. Set
    PRISMAQUANT_GRAPH_SHARED_POOL=0 only as a diagnostic.
    """
    if not _env_flag_enabled("PRISMAQUANT_GRAPH_SHARED_POOL", default=True):
        return None
    if not torch.cuda.is_available():
        return None
    global _PRISMAQUANT_GRAPH_POOL
    if _PRISMAQUANT_GRAPH_POOL is None:
        _PRISMAQUANT_GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _PRISMAQUANT_GRAPH_POOL


def _cuda_graph_pool_id(pool) -> str:
    if pool is None:
        return "private"
    return f"shared:{id(pool):x}"


def get_prismaquant_graph_pool_id() -> str:
    if not _env_flag_enabled("PRISMAQUANT_GRAPH_SHARED_POOL", default=True):
        return "private"
    if not torch.cuda.is_available():
        return "unavailable"
    if _PRISMAQUANT_GRAPH_POOL is None:
        return "shared:uninitialized"
    return _cuda_graph_pool_id(_PRISMAQUANT_GRAPH_POOL)


def _cost_entry(costs: Mapping, name: str, fmt: str) -> dict | None:
    per_name = costs.get(name, {})
    if not isinstance(per_name, Mapping):
        return None
    for alias in fr.aliases_for(fmt):
        entry = per_name.get(alias)
        if isinstance(entry, dict) and "error" not in entry:
            return entry
    return None


def l2_cost_value(stats: Mapping, costs: Mapping, name: str, fmt: str) -> float | None:
    """Return the allocator's L2 scalar cost for one existing cost entry."""
    entry = _cost_entry(costs, name, fmt)
    if entry is None or name not in stats:
        return None
    return float(cost_entry_predicted_dloss(stats[name], entry))


def _memory_bytes_for_format(
    stats_entry: Mapping,
    spec: fr.FormatSpec,
) -> int:
    memory_map = stats_entry.get("_memory_bytes_by_format")
    if isinstance(memory_map, Mapping):
        # Legacy stats may be keyed by the pre-canonical name (e.g.
        # ``"MXFP8"`` before the alias to ``"MXFP8_E4M3"`` was added).
        # Try the canonical name first, then any registered aliases for
        # the same spec.
        for key in (spec.name, *fr.aliases_for(spec.name)):
            if key in memory_map:
                return int(memory_map[key])
    return int(spec.memory_bytes_for_shape(_shape_from_stats(dict(stats_entry))))


def assignment_bit_total(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
) -> float:
    """Return total assigned bits, not average bits."""
    total = 0.0
    for name, fmt in assignment.items():
        if name not in stats:
            continue
        spec = specs_by_name[fr.canonical_format_name(fmt)]
        total += 8.0 * _memory_bytes_for_format(stats[name], spec)
    return total


def _available_formats_for_name(
    stats: Mapping,
    costs: Mapping,
    name: str,
    specs: list[fr.FormatSpec],
) -> list[str]:
    available: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        canonical = fr.canonical_format_name(spec.name)
        if (
            canonical not in seen
            and l2_cost_value(stats, costs, name, canonical) is not None
        ):
            available.append(canonical)
            seen.add(canonical)
    return available


def _bits_for_name(stats: Mapping, name: str, spec: fr.FormatSpec) -> float:
    shape = _shape_from_stats(dict(stats[name]))
    return float(spec.effective_bits_for_shape(shape))


def select_formats_for_l3(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    name: str,
    specs: list[fr.FormatSpec],
) -> tuple[str, ...]:
    """Choose current + one cheaper + one more accurate + BF16 when present."""
    if name not in stats or name not in assignment:
        return ()
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    current = fr.canonical_format_name(assignment[name])
    available = _available_formats_for_name(stats, costs, name, specs)
    if current not in available:
        return ()

    ordered = sorted(
        available,
        key=lambda fmt: (
            _bits_for_name(stats, name, specs_by_name[fmt]),
            fmt,
        ),
    )
    idx = ordered.index(current)
    chosen = {current}
    if idx > 0:
        chosen.add(ordered[idx - 1])
    if idx + 1 < len(ordered):
        chosen.add(ordered[idx + 1])
    if "BF16" in available:
        chosen.add("BF16")
    ordered = sorted(
        chosen,
        key=lambda fmt: (
            _bits_for_name(stats, name, specs_by_name[fmt]),
            fmt,
        ),
    )
    return tuple(specs_by_name[fmt].name for fmt in ordered)


def _relative_margin(values: list[float], current_cost: float) -> float:
    margins = []
    for value in values:
        denom = max(abs(current_cost), abs(value), 1e-12)
        margins.append(abs(value - current_cost) / denom)
    if not margins:
        return float("inf")
    return float(min(margins))


def _l3_unsupported_reason(stats_entry: Mapping) -> str | None:
    """Return why this probe entry is not currently L3-hookable."""
    if _stats_indicates_packed_expert(dict(stats_entry)):
        return (
            "packed MoE expert tensor: L3 target hooks currently measure "
            "nn.Linear modules only; packed experts remain L2-priced and "
            "exportable"
        )
    return None


def _is_l3_unsupported_target(stats_entry: Mapping) -> bool:
    """Return True for probe entries whose live module is not L3-hookable."""
    return _l3_unsupported_reason(stats_entry) is not None


def _log_l3_exclusions(
    excluded: Mapping[str, str],
    *,
    scope: str,
) -> None:
    if not excluded:
        return
    reason_counts: dict[str, int] = {}
    for reason in excluded.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_text = "; ".join(
        f"{count} x {reason}" for reason, count in sorted(reason_counts.items())
    )
    sample = sorted(excluded)[:5]
    print(
        f"[l3] excluded {len(excluded)} unsupported target(s) from {scope}: "
        f"{reason_text}; sample={sample}",
        flush=True,
    )


def _current_has_cheaper_available_format(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    name: str,
    specs: list[fr.FormatSpec],
) -> bool:
    current = fr.canonical_format_name(assignment[name])
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    if current not in specs_by_name:
        return False
    current_bits = _bits_for_name(stats, name, specs_by_name[current])
    for fmt in _available_formats_for_name(stats, costs, name, specs):
        if fmt == current or fmt not in specs_by_name:
            continue
        if _bits_for_name(stats, name, specs_by_name[fmt]) < current_bits - 1e-12:
            return True
    return False


def select_l3_neighborhood(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
    *,
    uncertainty_rel_tol: float = 0.10,
    min_fraction: float = 0.05,
    max_fraction: float = 0.30,
    safety_fraction: float = 0.02,
) -> list[L3NeighborhoodEntry]:
    """Select the small L2 neighborhood that L3 is allowed to re-optimize."""
    eligible: list[L3NeighborhoodEntry] = []
    excluded: dict[str, str] = {}
    for name in sorted(set(stats) & set(assignment)):
        reason = _l3_unsupported_reason(stats[name])
        if reason is not None:
            excluded[name] = reason
            continue
        current = fr.canonical_format_name(assignment[name])
        current_cost = l2_cost_value(stats, costs, name, current)
        if current_cost is None:
            continue
        fmts = select_formats_for_l3(stats, costs, assignment, name, specs)
        if not fmts:
            continue
        alt_costs = [
            value
            for fmt in fmts
            if fmt != current
            for value in [l2_cost_value(stats, costs, name, fmt)]
            if value is not None
        ]
        margin = _relative_margin(alt_costs, current_cost)
        eligible.append(
            L3NeighborhoodEntry(
                name=name,
                current_format=current,
                formats=fmts,
                margin=margin,
                l2_current_cost=current_cost,
            )
        )
    _log_l3_exclusions(excluded, scope="bounded neighborhood selection")

    if not eligible:
        return []

    total = len(eligible)
    max_count = max(1, int(math.ceil(total * max_fraction)))
    min_count = min(max_count, max(1, int(math.ceil(total * min_fraction))))
    safety_count = int(math.ceil(total * safety_fraction))

    by_name: dict[str, L3NeighborhoodEntry] = {}

    def _add(entry: L3NeighborhoodEntry, reason: str) -> None:
        existing = by_name.get(entry.name)
        reasons = set(existing.reasons if existing is not None else entry.reasons)
        reasons.add(reason)
        by_name[entry.name] = L3NeighborhoodEntry(
            name=entry.name,
            current_format=entry.current_format,
            formats=entry.formats,
            margin=entry.margin,
            l2_current_cost=entry.l2_current_cost,
            reasons=tuple(sorted(reasons)),
        )

    def _add_until_full(entries: list[L3NeighborhoodEntry], reason: str) -> None:
        for entry in entries:
            if entry.name not in by_name and len(by_name) >= max_count:
                continue
            _add(entry, reason)

    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}

    def _expected_flip_benefit(entry: L3NeighborhoodEntry) -> float:
        current = entry.current_format
        if current not in specs_by_name:
            return float("-inf")
        current_bits = _bits_for_name(stats, entry.name, specs_by_name[current])
        best = float("-inf")
        for fmt in entry.formats:
            if fmt == current or fmt not in specs_by_name:
                continue
            if _bits_for_name(stats, entry.name, specs_by_name[fmt]) >= current_bits:
                continue
            alt_cost = l2_cost_value(stats, costs, entry.name, fmt)
            if alt_cost is not None:
                best = max(best, entry.l2_current_cost - alt_cost)
        return best

    uncertain = [
        entry
        for entry in eligible
        if entry.margin <= uncertainty_rel_tol
    ]
    uncertain.sort(key=lambda e: (e.margin, -e.l2_current_cost, e.name))

    confident_non_cheapest = [
        entry
        for entry in eligible
        if _current_has_cheaper_available_format(
            stats, costs, assignment, entry.name, specs
        )
    ]
    benefit_by_name = {
        entry.name: _expected_flip_benefit(entry)
        for entry in confident_non_cheapest
    }
    confident_non_cheapest.sort(
        key=lambda e: (
            -benefit_by_name[e.name],
            -e.l2_current_cost,
            e.margin,
            e.name,
        )
    )

    safety = sorted(eligible, key=lambda e: (-e.l2_current_cost, e.name))[:safety_count]

    _add_until_full(confident_non_cheapest, "confident_non_cheapest")
    _add_until_full(uncertain, "uncertain")
    for entry in safety:
        if entry.name not in by_name and len(by_name) >= max_count:
            continue
        _add(entry, "high_l2_cost")

    if len(by_name) < min_count:
        fill = sorted(eligible, key=lambda e: (e.margin, -e.l2_current_cost, e.name))
        for entry in fill:
            if entry.name not in by_name and len(by_name) >= max_count:
                break
            _add(entry, "fill_min_fraction")
            if len(by_name) >= min_count:
                break

    return sorted(by_name.values(), key=lambda e: e.name)


def build_global_l3_neighborhood(
    stats: Mapping,
    costs: Mapping,
    assignment: Mapping[str, str],
    specs: list[fr.FormatSpec],
) -> list[L3NeighborhoodEntry]:
    """Build an L3 measurement neighborhood covering every eligible Linear."""
    selected: list[L3NeighborhoodEntry] = []
    excluded: dict[str, str] = {}
    for name in sorted(set(stats) & set(assignment)):
        reason = _l3_unsupported_reason(stats[name])
        if reason is not None:
            excluded[name] = reason
            continue
        current = fr.canonical_format_name(assignment[name])
        current_cost = l2_cost_value(stats, costs, name, current)
        if current_cost is None:
            continue
        fmts = select_formats_for_l3(stats, costs, assignment, name, specs)
        if not fmts:
            continue
        alt_costs = [
            value
            for fmt in fmts
            if fmt != current
            for value in [l2_cost_value(stats, costs, name, fmt)]
            if value is not None
        ]
        selected.append(
            L3NeighborhoodEntry(
                name=name,
                current_format=current,
                formats=fmts,
                margin=_relative_margin(alt_costs, current_cost),
                l2_current_cost=current_cost,
                reasons=("global",),
            )
        )
    _log_l3_exclusions(excluded, scope="global selection")
    return selected


def build_l3_candidates(
    stats: Mapping,
    propagated_costs: Mapping[str, Mapping[str, Mapping]],
    specs: list[fr.FormatSpec],
) -> dict[str, list[Candidate]]:
    """Build DP candidates from propagated end-KL costs only."""
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    out: dict[str, list[Candidate]] = {}
    for name, per_name in propagated_costs.items():
        if name not in stats or not isinstance(per_name, Mapping):
            continue
        shape = _shape_from_stats(dict(stats[name]))
        cands: list[Candidate] = []
        for fmt, entry in per_name.items():
            canonical = fr.canonical_format_name(fmt)
            if canonical not in specs_by_name or not isinstance(entry, Mapping):
                continue
            if "error" in entry or "propagated_end_kl" not in entry:
                continue
            spec = specs_by_name[canonical]
            cands.append(
                Candidate(
                    fmt=canonical,
                    bits_per_param=spec.effective_bits_for_shape(shape),
                    memory_bytes=_memory_bytes_for_format(stats[name], spec),
                    predicted_dloss=max(float(entry["propagated_end_kl"]), 0.0),
                )
            )
        if cands:
            out[name] = cands
    return out


def _candidate_total_bits(candidate: Candidate) -> float:
    return 8.0 * float(candidate.memory_bytes)


def _greedy_l3_under_budget(
    open_cands: Mapping[str, list[Candidate]],
    current_assignment: Mapping[str, str],
    remaining_bits: float,
    budget_ceiling_bits: float | None = None,
) -> tuple[dict[str, str], dict[str, Candidate], dict]:
    names = sorted(open_cands)
    chosen: dict[str, Candidate] = {}
    for name in names:
        # Index candidates by canonical format so a candidate with raw
        # legacy ``c.fmt == "MXFP8"`` matches the canonical
        # ``"MXFP8_E4M3"`` after `current_assignment` is canonicalized.
        by_fmt = {fr.canonical_format_name(c.fmt): c for c in open_cands[name]}
        current_fmt = fr.canonical_format_name(current_assignment.get(name, "BF16"))
        chosen[name] = by_fmt.get(current_fmt) or min(
            open_cands[name],
            key=lambda c: (c.predicted_dloss, _candidate_total_bits(c), c.fmt),
        )

    used_bits = sum(_candidate_total_bits(c) for c in chosen.values())
    budget_ceiling_bits = (
        float(remaining_bits)
        if budget_ceiling_bits is None
        else float(budget_ceiling_bits)
    )
    eps = 1e-12
    attempts = []
    for name in names:
        current = chosen[name]
        for cand in open_cands[name]:
            if cand.fmt == current.fmt:
                continue
            improvement = current.predicted_dloss - cand.predicted_dloss
            bit_delta = _candidate_total_bits(cand) - _candidate_total_bits(current)
            if bit_delta < -eps and improvement >= -eps:
                priority = 0
            elif bit_delta < -eps:
                priority = 1
            elif improvement > eps:
                priority = 2
            else:
                priority = 3
            attempts.append((priority, improvement, name, cand))
    attempts.sort(
        key=lambda item: (
            item[0],
            -item[1],
            _candidate_total_bits(item[3]),
            item[2],
            item[3].fmt,
        )
    )

    stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_not_better": 0,
        "rejected_budget": 0,
        "accepted_budget_reducing_nonworse": 0,
        "accepted_budget_reducing_worse": 0,
        "accepted_cost_improving": 0,
        "start_bits": used_bits,
        "end_bits": None,
        "remaining_bits": float(remaining_bits),
        "budget_ceiling_bits": float(budget_ceiling_bits),
    }
    swapped_names: set[str] = set()
    for _priority, improvement, name, cand in attempts:
        if name in swapped_names:
            continue
        stats["attempts"] += 1
        current = chosen[name]
        current_bits = _candidate_total_bits(current)
        cand_bits = _candidate_total_bits(cand)
        next_bits = used_bits - current_bits + cand_bits
        if next_bits > budget_ceiling_bits + 1e-6:
            stats["rejected_budget"] += 1
            continue
        overshoot_before = max(float(used_bits) - float(remaining_bits), 0.0)
        overshoot_after = max(float(next_bits) - float(remaining_bits), 0.0)
        reduces_overshoot = overshoot_after < overshoot_before - 1e-6
        cost_worsens = improvement < -eps
        cost_improves = improvement > eps
        if reduces_overshoot:
            if cost_worsens:
                stats["accepted_budget_reducing_worse"] += 1
            else:
                stats["accepted_budget_reducing_nonworse"] += 1
        elif cost_improves:
            stats["accepted_cost_improving"] += 1
        else:
            stats["rejected_not_better"] += 1
            continue
        chosen[name] = cand
        used_bits = next_bits
        swapped_names.add(name)
        stats["accepted"] += 1

    stats["end_bits"] = used_bits
    assignment = {name: chosen[name].fmt for name in names}
    return assignment, chosen, stats


def solve_frozen_l3_neighborhood(
    stats: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    l3_candidates: Mapping[str, list[Candidate]],
    specs: list[fr.FormatSpec],
    *,
    target_bits: float,
    bit_precision: float,
    budget_tolerance: float = 0.0,
    return_metadata: bool = False,
) -> tuple[dict[str, str], dict[str, Candidate]]:
    """Solve L3 candidates while freezing all non-neighborhood L2 choices."""
    specs_by_name = {fr.canonical_format_name(s.name): s for s in specs}
    all_names = set(stats) & set(assignment)
    open_names = set(l3_candidates)
    frozen_assignment = {
        name: assignment[name]
        for name in sorted(all_names - open_names)
    }
    total_params = sum(int(stats[n].get("n_params", 0) or 0) for n in all_names)
    open_params = sum(int(stats[n].get("n_params", 0) or 0) for n in open_names)
    if total_params <= 0:
        result = (dict(assignment), {})
        if return_metadata:
            return (*result, {"frozen_dp_precision_used": "none"})
        return result

    target_total_bits = float(target_bits) * float(total_params)
    budget_tolerance_bits = max(0.0, float(budget_tolerance)) * target_total_bits
    frozen_bits = assignment_bit_total(stats, frozen_assignment, specs_by_name)
    remaining_bits = target_total_bits - frozen_bits
    if remaining_bits < -1e-6:
        raise FrozenBudgetError(
            "L3 polish infeasible: frozen L2 choices already exceed target "
            f"budget ({frozen_bits / total_params:.6f} bpp frozen vs "
            f"{target_bits:.6f} bpp target)."
        )
    if open_params <= 0:
        result = (dict(assignment), {})
        if return_metadata:
            return (*result, {"frozen_dp_precision_used": "none"})
        return result

    open_target_bits = remaining_bits / float(open_params)
    open_stats = {name: dict(stats[name]) for name in sorted(open_names)}
    open_cands = {name: list(l3_candidates[name]) for name in sorted(open_names)}
    result = solve_allocation(open_stats, open_cands, open_target_bits, bit_precision)
    precision_used: float | str = float(bit_precision)
    dp_attempts = [{"precision": float(bit_precision), "result": "ok" if result is not None else "failed"}]
    if result is None:
        print(
            f"[l3] frozen DP precision {float(bit_precision):g}: failed",
            flush=True,
        )
        for fallback_precision in (0.01, 0.05, 0.25, 0.5, 1.0):
            result = solve_allocation(
                open_stats,
                open_cands,
                open_target_bits,
                fallback_precision,
            )
            dp_attempts.append({
                "precision": fallback_precision,
                "result": "ok" if result is not None else "failed",
            })
            if result is not None:
                precision_used = fallback_precision
                print(
                    f"[l3] frozen DP precision {fallback_precision:g}: ok",
                    flush=True,
                )
                break
            print(
                f"[l3] frozen DP precision {fallback_precision:g}: failed",
                flush=True,
            )
    if result is None:
        open_current_assignment = {
            name: assignment[name]
            for name in open_cands
            if name in assignment
        }
        open_assignment, chosen, greedy_stats = _greedy_l3_under_budget(
            open_cands,
            open_current_assignment,
            remaining_bits,
            remaining_bits + budget_tolerance_bits,
        )
        result = (open_assignment, chosen)
        precision_used = "greedy"
        print(
            "[l3] frozen DP greedy: "
            f"attempts={greedy_stats['attempts']} "
            f"accepted={greedy_stats['accepted']} "
            f"rejected_not_better={greedy_stats['rejected_not_better']} "
            f"rejected_budget={greedy_stats['rejected_budget']} "
            f"budget_ceiling_bits={greedy_stats['budget_ceiling_bits']:.1f}",
            flush=True,
        )
    else:
        greedy_stats = None
    open_assignment, chosen = result
    merged = dict(assignment)
    merged.update(open_assignment)
    if return_metadata:
        return merged, chosen, {
            "frozen_dp_precision_used": precision_used,
            "frozen_dp_attempts": dp_attempts,
            "frozen_dp_greedy": greedy_stats,
            "frozen_dp_budget_tolerance": float(budget_tolerance),
            "frozen_dp_budget_tolerance_bits": float(budget_tolerance_bits),
        }
    return merged, chosen


_LAYER_DEPTH_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


def layer_depth(name: str) -> int | None:
    """Best-effort decoder-layer depth parser for depth-grouped L3 batches."""
    m = _LAYER_DEPTH_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _group_neighborhood_by_depth(
    entries: list[L3NeighborhoodEntry],
) -> list[tuple[str, list[L3NeighborhoodEntry]]]:
    grouped: dict[str, list[L3NeighborhoodEntry]] = {}
    for entry in entries:
        depth = layer_depth(entry.name)
        key = f"layer:{depth:05d}" if depth is not None else f"name:{entry.name}"
        grouped.setdefault(key, []).append(entry)
    return [(key, grouped[key]) for key in sorted(grouped)]


def _canonical_assignment(
    assignment: Mapping[str, str],
) -> dict[str, str]:
    return {
        str(name): fr.canonical_format_name(fmt)
        for name, fmt in assignment.items()
    }


def _first_tensor_batch_size(args, kwargs) -> int:
    for value in list(args) + list((kwargs or {}).values()):
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.size(0))
    raise ValueError("could not infer calibration batch size from model inputs")


def _repeat_value_for_lanes(value, lane_count: int):
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        repeats = (int(lane_count),) + (1,) * (value.dim() - 1)
        return value.repeat(repeats)
    if isinstance(value, Mapping):
        return {
            key: _repeat_value_for_lanes(child, lane_count)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_repeat_value_for_lanes(child, lane_count) for child in value)
    if isinstance(value, list):
        return [_repeat_value_for_lanes(child, lane_count) for child in value]
    return value


def _repeat_inputs_for_lanes(args, kwargs, lane_count: int):
    return (
        tuple(_repeat_value_for_lanes(value, lane_count) for value in args),
        {
            key: _repeat_value_for_lanes(value, lane_count)
            for key, value in (kwargs or {}).items()
        },
    )


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def _first_tensor_output(output) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, Mapping):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    return None


def _replace_first_tensor_output(output, replacement: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        values = list(output)
        for idx, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[idx] = replacement
                return tuple(values)
    if isinstance(output, dict):
        values = dict(output)
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = replacement
                return values
    return output


def _decoder_stack(model: nn.Module):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ]
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.append(getattr(language_model, "model", None))
    for base in candidates:
        if base is None:
            continue
        layers = getattr(base, "layers", None)
        if layers is not None and hasattr(layers, "__len__"):
            return base, layers
    return None, None


def _replace_first_tensor_call(args, kwargs, replacement: torch.Tensor):
    args = list(args)
    for idx, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            args[idx] = replacement
            return tuple(args), dict(kwargs or {})
    kwargs = dict(kwargs or {})
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            kwargs[key] = replacement
            return tuple(args), kwargs
    return (replacement, *tuple(args)), kwargs


def _repeat_layer_value_for_lanes(value, lane_count: int, base_batch: int):
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.size(0)) == int(base_batch):
            repeats = (int(lane_count),) + (1,) * (value.dim() - 1)
            return value.repeat(repeats)
        return value
    if isinstance(value, Mapping):
        return {
            key: _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for child in value
        )
    if isinstance(value, list):
        return [
            _repeat_layer_value_for_lanes(child, lane_count, base_batch)
            for child in value
        ]
    return value


def _repeat_layer_call_for_lanes(args, kwargs, lane_count: int, base_batch: int):
    return (
        tuple(
            _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for value in args
        ),
        {
            key: _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for key, value in (kwargs or {}).items()
        },
    )


class _TailLayerCaptureDone(Exception):
    pass


def _clone_layer_value_for_cache(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_layer_value_for_cache(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_layer_value_for_cache(child) for child in value)
    if isinstance(value, list):
        return [_clone_layer_value_for_cache(child) for child in value]
    return value


def _move_cached_layer_value(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {
            key: _move_cached_layer_value(child, device)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_cached_layer_value(child, device) for child in value)
    if isinstance(value, list):
        return [_move_cached_layer_value(child, device) for child in value]
    return value


def _move_cached_layer_call(cached_call, device):
    args, kwargs, base_batch = cached_call
    return (
        tuple(_move_cached_layer_value(value, device) for value in args),
        {
            key: _move_cached_layer_value(value, device)
            for key, value in kwargs.items()
        },
        base_batch,
    )


def _model_accepts_kwarg(model: nn.Module, name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _capture_layer_call(model: nn.Module, layer: nn.Module, args, kwargs):
    captured = {}

    def _hook(_module, hook_args, hook_kwargs):
        layer_kwargs = dict(hook_kwargs or {})
        if "use_cache" in layer_kwargs:
            layer_kwargs["use_cache"] = False
        if "past_key_value" in layer_kwargs:
            layer_kwargs["past_key_value"] = None
        captured["args"] = tuple(hook_args)
        captured["kwargs"] = layer_kwargs
        raise _TailLayerCaptureDone

    handle = layer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        try:
            model(*args, **(kwargs or {}))
        except _TailLayerCaptureDone:
            pass
    finally:
        handle.remove()
    if "args" not in captured:
        raise RuntimeError("tail-only L3 could not capture decoder layer inputs")
    return captured["args"], captured["kwargs"]


def _capture_all_layer_calls(
    model: nn.Module,
    layers,
    layer_indices: set[int],
    calibration_data,
    device,
) -> dict[int, list[tuple[tuple, dict, int]]]:
    captured: dict[int, list[tuple[tuple, dict, int]]] = {
        idx: [] for idx in sorted(layer_indices)
    }
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, hook_args, hook_kwargs):
            layer_kwargs = dict(hook_kwargs or {})
            if "use_cache" in layer_kwargs:
                layer_kwargs["use_cache"] = False
            if "past_key_value" in layer_kwargs:
                layer_kwargs["past_key_value"] = None
            base_batch = _first_tensor_batch_size(hook_args, layer_kwargs)
            captured[layer_idx].append(
                (
                    tuple(_clone_layer_value_for_cache(value) for value in hook_args),
                    {
                        key: _clone_layer_value_for_cache(value)
                        for key, value in layer_kwargs.items()
                    },
                    int(base_batch),
                )
            )

        return _hook

    for layer_idx in sorted(layer_indices):
        layer = layers[layer_idx]
        handles.append(layer.register_forward_pre_hook(
            _make_hook(layer_idx),
            with_kwargs=True,
        ))
    try:
        for args, kwargs in iter_calibration_forwards(calibration_data, device):
            call_kwargs = dict(kwargs or {})
            if _model_accepts_kwarg(model, "use_cache"):
                call_kwargs["use_cache"] = False
            model(*args, **call_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _tail_forward_eager(
    model: nn.Module,
    layer_idx: int,
    layer_args,
    layer_kwargs,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Run decoder layers after ``layer_idx`` plus final norm and LM head."""
    base, layers = _decoder_stack(model)
    if layers is None:
        raise RuntimeError("tail-only L3 requires a decoder layer stack")
    hidden = hidden_state
    for next_idx in range(int(layer_idx) + 1, len(layers)):
        call_args, call_kwargs = _replace_first_tensor_call(
            layer_args,
            layer_kwargs,
            hidden,
        )
        output = layers[next_idx](*call_args, **call_kwargs)
        next_hidden = _first_tensor_output(output)
        if next_hidden is None:
            raise RuntimeError("tail-only L3 decoder layer returned no tensor")
        hidden = next_hidden
    norm = getattr(base, "norm", None)
    if norm is not None:
        hidden = norm(hidden)
    lm_head = getattr(model, "lm_head", None) or getattr(base, "lm_head", None)
    if lm_head is not None:
        return lm_head(hidden)
    return hidden


def _tensor_tree_signature(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__name__,
            tuple(
                sorted(
                    (str(key), _tensor_tree_signature(child))
                    for key, child in value.items()
                )
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_signature(child) for child in value))
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_signature(child) for child in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", type(value).__name__, value)
    return ("object", type(value).__name__, id(value))


def _clone_static_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_static_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_static_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_static_tree(child) for child in value]
    return value


def _copy_static_tree(src, dst) -> bool:
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        if (
            tuple(src.shape) != tuple(dst.shape)
            or src.dtype != dst.dtype
            or src.device != dst.device
        ):
            return False
        dst.copy_(src)
        return True
    if isinstance(src, Mapping) and isinstance(dst, Mapping):
        if set(src.keys()) != set(dst.keys()):
            return False
        return all(_copy_static_tree(src[key], dst[key]) for key in src)
    if isinstance(src, tuple) and isinstance(dst, tuple):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if isinstance(src, list) and isinstance(dst, list):
        if len(src) != len(dst):
            return False
        return all(_copy_static_tree(a, b) for a, b in zip(src, dst))
    if src is dst:
        return True
    if src is None or isinstance(src, (bool, int, float, str)):
        return src == dst
    return False


def _first_cuda_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value if value.is_cuda else None
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _first_cuda_tensor(child)
            if found is not None:
                return found
    return None


def _clone_cuda_graph_output(value):
    clone_disabled = not _env_flag_enabled(
        "PRISMAQUANT_GRAPH_OUTPUT_CLONE",
        default=True,
    )
    if clone_disabled and _env_flag_enabled(
        "PRISMAQUANT_GRAPH_SHARED_POOL",
        default=True,
    ):
        global _NOCLONE_OVERRIDE_WARNED
        if not _NOCLONE_OVERRIDE_WARNED:
            _NOCLONE_OVERRIDE_WARNED = True
            print(
                "[cuda-graphs] warning: "
                "PRISMAQUANT_GRAPH_OUTPUT_CLONE=0 is unsafe with "
                "PRISMAQUANT_GRAPH_SHARED_POOL=1; cloning CUDA graph outputs instead",
                file=sys.stderr,
                flush=True,
            )
        clone_disabled = False
    if clone_disabled:
        return value
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_cuda_graph_output(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_cuda_graph_output(child) for child in value)
    if isinstance(value, list):
        return [_clone_cuda_graph_output(child) for child in value]
    return value


_CUDA_GRAPH_WARNED_LABELS: set[str] = set()


def _warn_cuda_graph_fallback_once(label: str, exc: BaseException) -> None:
    if label in _CUDA_GRAPH_WARNED_LABELS:
        return
    _CUDA_GRAPH_WARNED_LABELS.add(label)
    print(
        "[cuda-graphs] warning: "
        f"{label} capture/replay failed once; using eager for that shape "
        f"({type(exc).__name__}: {exc})",
        file=sys.stderr,
        flush=True,
    )


def _cuda_graph_debug_node_count(path: Path) -> int | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "label=" in stripped and "->" not in stripped:
            count += 1
    return count if count > 0 else None


@dataclass
class _CUDAGraphEntry:
    graph: object
    static_args: tuple
    static_kwargs: dict
    static_output: object
    keepalive: tuple[object, ...] = field(default_factory=tuple)


class CUDAGraphRegistry:
    """Bounded LRU CUDA graph cache for fixed-shape tensor forwards.

    Each entry owns graph activation memory plus static input/output tensors.
    The default cap is intentionally small and can be overridden per path with
    the registry's ``max_entries_env`` variable.

    With ``PRISMAQUANT_GRAPH_SHARED_POOL`` enabled,
    ``PRISMAQUANT_GRAPH_OUTPUT_CLONE=0`` is ignored and outputs are cloned.
    Shared-pool captures can alias pool-mate static outputs still held by callers.
    """

    def __init__(
        self,
        *,
        label: str,
        max_entries: int = 4,
        max_entries_env: str | None = None,
        warmup_iters: int = 2,
        verbose_env: str | None = None,
    ):
        self.label = str(label)
        self.default_max_entries = max(int(max_entries), 0)
        self.max_entries_env = max_entries_env
        self.warmup_iters = max(int(warmup_iters), 0)
        self.verbose_env = verbose_env
        self.entries: OrderedDict[tuple, _CUDAGraphEntry] = OrderedDict()
        self.disabled_keys: set[tuple] = set()
        self.eviction_count = 0
        register_budget_evictor(self)

    def graph_pool(self):
        return get_prismaquant_graph_pool()

    def graph_pool_id(self) -> str:
        return get_prismaquant_graph_pool_id()

    def clear(self) -> None:
        had_entries = bool(self.entries)
        self.entries.clear()
        self.disabled_keys.clear()
        if had_entries and torch.cuda.is_available():
            gc.collect()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _max_entries(self) -> int:
        if (
            self.max_entries_env is not None
            and os.environ.get(self.max_entries_env) is not None
        ):
            return _env_int(self.max_entries_env, self.default_max_entries)
        return _env_int(
            "PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH",
            self.default_max_entries,
        )

    def _verbose_enabled(self) -> bool:
        return (
            self.verbose_env is not None
            and _env_flag_enabled(self.verbose_env, default=False)
        )

    def _verbose_log(self, label: str, message: str) -> None:
        if not self._verbose_enabled():
            return
        print(
            f"[cuda-graphs][{self.label}:{label}] "
            f"{time.time():.6f} {message}",
            file=sys.stderr,
            flush=True,
        )

    def _verbose_exception(
        self,
        label: str,
        message: str,
        exc: BaseException,
    ) -> None:
        if not self._verbose_enabled():
            return
        self._verbose_log(label, f"{message}: {type(exc).__name__}: {exc}")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def _debug_graph_summary(
        self,
        graph,
        label: str,
    ) -> tuple[int | str, str | None]:
        node_count: int | str = "unavailable"
        dump_path: str | None = None
        for attr in ("num_nodes", "_num_nodes"):
            fn = getattr(graph, attr, None)
            if callable(fn):
                try:
                    node_count = int(fn())
                    break
                except Exception:
                    pass
        debug_dump = getattr(graph, "debug_dump", None)
        if callable(debug_dump):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.label}_{label}")
            path = (
                Path(tempfile.gettempdir())
                / f"prismaquant_cuda_graph_{safe}_{os.getpid()}.dot"
            )
            try:
                debug_dump(str(path))
                dump_path = str(path)
                if node_count == "unavailable":
                    parsed_count = _cuda_graph_debug_node_count(path)
                    if parsed_count is not None:
                        node_count = parsed_count
            except Exception as exc:
                self._verbose_exception(label, "debug dump failed", exc)
        return node_count, dump_path

    def _evict_if_needed(self) -> None:
        max_entries = self._max_entries()
        if max_entries <= 0:
            evicted = len(self.entries)
            self.entries.clear()
            self.eviction_count += evicted
            if evicted:
                self._log_graph_eviction(evicted, max_entries)
            return
        while len(self.entries) > max_entries:
            self._evict_oldest_graph_entry(max_entries=max_entries)

    def _log_graph_eviction(self, count: int, *, max_entries: int) -> None:
        print(
            "[cuda-graphs] "
            f"{self.label}: evicted {count} graph(s) "
            f"(max_entries={max_entries})",
            file=sys.stderr,
            flush=True,
        )

    def _evict_oldest_graph_entry(self, *, max_entries: int | None = None) -> bool:
        if not self.entries:
            return False
        self.entries.popitem(last=False)
        self.eviction_count += 1
        self._log_graph_eviction(
            1,
            max_entries=self._max_entries() if max_entries is None else max_entries,
        )
        return True

    def evict_oldest_for_memory_budget(self) -> bool:
        return self._evict_oldest_graph_entry()

    def _cleanup_failed_capture(
        self,
        graph,
        device: torch.device,
        label: str,
    ) -> None:
        if graph is not None:
            reset = getattr(graph, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:
                    self._verbose_exception(label, "failed graph reset failed", exc)
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            self._verbose_exception(label, "post-failure synchronize failed", exc)
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            self._verbose_exception(label, "post-failure empty_cache failed", exc)

    def run(
        self,
        label: str,
        key: tuple,
        fn: Callable,
        *args,
        enabled: bool = True,
        device: torch.device | None = None,
        keepalive: tuple[object, ...] = (),
        **kwargs,
    ):
        cuda_tensor = _first_cuda_tensor((args, kwargs))
        graph_device = device
        if graph_device is None and cuda_tensor is not None:
            graph_device = cuda_tensor.device
        if (
            not enabled
            or not torch.cuda.is_available()
            or graph_device is None
            or torch.device(graph_device).type != "cuda"
            or self._max_entries() <= 0
        ):
            return fn(*args, **kwargs)

        full_key = (
            self.label,
            str(label),
            tuple(key),
            _tensor_tree_signature(args),
            _tensor_tree_signature(kwargs),
        )
        entry = self.entries.get(full_key)
        if entry is not None:
            self.entries.move_to_end(full_key)
            if not (
                _copy_static_tree(tuple(args), entry.static_args)
                and _copy_static_tree(dict(kwargs), entry.static_kwargs)
            ):
                return fn(*args, **kwargs)
            try:
                entry.graph.replay()
                return _clone_cuda_graph_output(entry.static_output)
            except Exception as exc:
                self.entries.pop(full_key, None)
                self.disabled_keys.add(full_key)
                self._verbose_exception(str(label), "replay failed", exc)
                _warn_cuda_graph_fallback_once(str(label), exc)
                return fn(*args, **kwargs)
        if full_key in self.disabled_keys:
            return fn(*args, **kwargs)

        try:
            enforce_gpu_memory_budget(
                [self],
                device=torch.device(graph_device),
                reason=f"{self.label}:{label} CUDA graph capture",
            )
            entry = self._capture(
                fn,
                args,
                kwargs,
                torch.device(graph_device),
                keepalive=keepalive,
                capture_label=str(label),
            )
        except Exception as exc:
            if isinstance(exc, GPUMemoryBudgetExceeded):
                raise
            self.disabled_keys.add(full_key)
            self._verbose_exception(str(label), "capture failed", exc)
            _warn_cuda_graph_fallback_once(str(label), exc)
            return fn(*args, **kwargs)
        self.entries[full_key] = entry
        self._evict_if_needed()
        enforce_gpu_memory_budget(
            [self],
            device=torch.device(graph_device),
            reason=f"{self.label}:{label} CUDA graph capture",
        )
        return _clone_cuda_graph_output(entry.static_output)

    def _capture(
        self,
        fn: Callable,
        args: tuple,
        kwargs: Mapping,
        device: torch.device,
        *,
        keepalive: tuple[object, ...],
        capture_label: str | None = None,
    ) -> _CUDAGraphEntry:
        label = capture_label or self.label
        capture_start_wall = time.time()
        capture_start = time.perf_counter()
        self._verbose_log(
            label,
            f"capture start device={device} warmup_iters={self.warmup_iters}",
        )
        static_args = tuple(_clone_static_tree(value) for value in args)
        static_kwargs = {
            key: _clone_static_tree(value)
            for key, value in dict(kwargs).items()
        }
        current_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream), torch.no_grad():
            for _ in range(self.warmup_iters):
                fn(*static_args, **static_kwargs)
        current_stream.wait_stream(side_stream)
        if self._verbose_enabled():
            try:
                torch.cuda.synchronize(device)
            except Exception as exc:
                self._verbose_exception(label, "warmup synchronize failed", exc)
                raise

        def _new_graph():
            try:
                graph_obj = torch.cuda.CUDAGraph(
                    keep_graph=self._verbose_enabled()
                )
            except TypeError:
                graph_obj = torch.cuda.CUDAGraph()
            if self._verbose_enabled():
                enable_debug = getattr(graph_obj, "enable_debug_mode", None)
                if callable(enable_debug):
                    try:
                        enable_debug()
                    except Exception as exc:
                        self._verbose_exception(
                            label,
                            "enable debug mode failed",
                            exc,
                        )
            return graph_obj

        graph = _new_graph()
        graph_pool = None
        try:
            graph_pool = self.graph_pool()
            graph_cm = (
                torch.cuda.graph(graph, pool=graph_pool)
                if graph_pool is not None
                else torch.cuda.graph(graph)
            )
            with graph_cm, torch.no_grad():
                static_output = fn(*static_args, **static_kwargs)
        except Exception as exc:
            retry_private = (
                graph_pool is not None
                and "use_count > 0" in str(exc)
            )
            if not retry_private:
                self._verbose_exception(label, "capture body/end failed", exc)
                self._cleanup_failed_capture(graph, device, label)
                raise
            self._verbose_exception(
                label,
                "shared-pool capture failed; retrying private pool",
                exc,
            )
            self._cleanup_failed_capture(graph, device, label)
            graph = _new_graph()
            try:
                with torch.cuda.graph(graph), torch.no_grad():
                    static_output = fn(*static_args, **static_kwargs)
            except Exception as retry_exc:
                self._verbose_exception(
                    label,
                    "private-pool retry failed",
                    retry_exc,
                )
                self._cleanup_failed_capture(graph, device, label)
                raise retry_exc from exc
        try:
            instantiate = getattr(graph, "instantiate", None)
            if self._verbose_enabled() and callable(instantiate):
                instantiate()
            graph.replay()
        except Exception as exc:
            self._verbose_exception(label, "initial replay failed", exc)
            self._cleanup_failed_capture(graph, device, label)
            raise
        if self._verbose_enabled():
            try:
                torch.cuda.synchronize(device)
            except Exception as exc:
                self._verbose_exception(label, "post-capture synchronize failed", exc)
                raise
            node_count, dump_path = self._debug_graph_summary(graph, label)
            elapsed = time.perf_counter() - capture_start
            suffix = f" debug_dump={dump_path}" if dump_path is not None else ""
            self._verbose_log(
                label,
                "capture end "
                f"started_at={capture_start_wall:.6f} "
                f"elapsed={elapsed:.6f}s graph_nodes={node_count}{suffix}",
            )
        return _CUDAGraphEntry(
            graph=graph,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
            keepalive=tuple(keepalive),
        )


_COORD_LANE_CUDA_GRAPH_REGISTRY = CUDAGraphRegistry(
    label="coord-lane",
    max_entries=4,
    max_entries_env="PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE",
)


@dataclass
class _TailCudaGraphEntry:
    graph: object
    static_hidden: torch.Tensor
    static_args: tuple
    static_kwargs: dict
    static_output: torch.Tensor


class _TailCudaGraphCache:
    """Bounded tail-forward CUDA graph cache.

    With ``PRISMAQUANT_GRAPH_SHARED_POOL`` enabled,
    ``PRISMAQUANT_GRAPH_OUTPUT_CLONE=0`` is ignored and outputs are cloned.
    Another registry's capture can alias this cache's static output.
    """

    def __init__(self, *, enabled: bool):
        self.label = "l3-tail"
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.entries: OrderedDict[tuple, _TailCudaGraphEntry] = OrderedDict()
        self.disabled_keys: set[tuple] = set()
        self.eviction_count = 0
        register_budget_evictor(self)

    def graph_pool(self):
        return get_prismaquant_graph_pool()

    def graph_pool_id(self) -> str:
        return get_prismaquant_graph_pool_id()

    def clear(self) -> None:
        had_entries = bool(self.entries)
        self.entries.clear()
        self.disabled_keys.clear()
        if had_entries and torch.cuda.is_available():
            gc.collect()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _max_entries(self) -> int:
        return _env_int("PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH", 4)

    def _log_graph_eviction(self, count: int) -> None:
        print(
            "[cuda-graphs] "
            f"l3-tail: evicted {count} graph(s) "
            f"(max_entries={self._max_entries()})",
            file=sys.stderr,
            flush=True,
        )

    def _evict_oldest_graph_entry(self) -> bool:
        if not self.entries:
            return False
        self.entries.popitem(last=False)
        self.eviction_count += 1
        self._log_graph_eviction(1)
        return True

    def evict_oldest_for_memory_budget(self) -> bool:
        return self._evict_oldest_graph_entry()

    def _evict_if_needed(self) -> None:
        max_entries = self._max_entries()
        if max_entries <= 0:
            evicted = len(self.entries)
            self.entries.clear()
            self.eviction_count += evicted
            if evicted:
                self._log_graph_eviction(evicted)
            return
        while len(self.entries) > max_entries:
            self._evict_oldest_graph_entry()

    def run(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_args,
        layer_kwargs,
        hidden_state: torch.Tensor,
        *,
        lane_count: int,
        state_key: object | None = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or not isinstance(hidden_state, torch.Tensor)
            or not hidden_state.is_cuda
        ):
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        key = (
            id(model),
            int(layer_idx),
            int(lane_count),
            state_key,
            _tensor_tree_signature(hidden_state),
            _tensor_tree_signature(layer_args),
            _tensor_tree_signature(layer_kwargs or {}),
        )
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            if not (
                _copy_static_tree(hidden_state, entry.static_hidden)
                and _copy_static_tree(tuple(layer_args), entry.static_args)
                and _copy_static_tree(dict(layer_kwargs or {}), entry.static_kwargs)
            ):
                return _tail_forward_eager(
                    model,
                    layer_idx,
                    layer_args,
                    layer_kwargs,
                    hidden_state,
                )
            entry.graph.replay()
            return _clone_cuda_graph_output(entry.static_output)
        if key in self.disabled_keys:
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        try:
            enforce_gpu_memory_budget(
                [self],
                device=hidden_state.device,
                reason="l3-tail CUDA graph capture",
            )
            entry = self._capture(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        except GPUMemoryBudgetExceeded:
            raise
        except Exception:
            self.disabled_keys.add(key)
            return _tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )
        self.entries[key] = entry
        self._evict_if_needed()
        enforce_gpu_memory_budget(
            [self],
            device=hidden_state.device,
            reason="l3-tail CUDA graph capture",
        )
        return _clone_cuda_graph_output(entry.static_output)

    def _capture(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_args,
        layer_kwargs,
        hidden_state: torch.Tensor,
    ) -> _TailCudaGraphEntry:
        static_hidden = hidden_state.detach().clone()
        static_args = tuple(_clone_static_tree(value) for value in layer_args)
        static_kwargs = {
            key: _clone_static_tree(value)
            for key, value in (layer_kwargs or {}).items()
        }
        device = hidden_state.device
        current_stream = torch.cuda.current_stream(device)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                _tail_forward_eager(
                    model,
                    layer_idx,
                    static_args,
                    static_kwargs,
                    static_hidden,
                )
        current_stream.wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        graph_pool = self.graph_pool()
        graph_cm = (
            torch.cuda.graph(graph, pool=graph_pool)
            if graph_pool is not None
            else torch.cuda.graph(graph)
        )
        with graph_cm:
            static_output = _tail_forward_eager(
                model,
                layer_idx,
                static_args,
                static_kwargs,
                static_hidden,
            )
        return _TailCudaGraphEntry(
            graph=graph,
            static_hidden=static_hidden,
            static_args=static_args,
            static_kwargs=static_kwargs,
            static_output=static_output,
        )


def tail_forward_from_layer(
    model: nn.Module,
    layer_idx: int,
    layer_args,
    layer_kwargs,
    hidden_state: torch.Tensor,
    *,
    cuda_graph_cache: _TailCudaGraphCache | None = None,
    lane_count: int | None = None,
    graph_state_key: object | None = None,
) -> torch.Tensor:
    if cuda_graph_cache is not None:
        # Private-pool no-clone mode requires L3 to consume these logits before
        # another tail replay can overwrite the same static output.
        return cuda_graph_cache.run(
            model,
            layer_idx,
            layer_args,
            layer_kwargs,
            hidden_state,
            lane_count=lane_count or 1,
            state_key=graph_state_key,
        )
    return _tail_forward_eager(
        model,
        layer_idx,
        layer_args,
        layer_kwargs,
        hidden_state,
    )


def _assignment_graph_key(assignment: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(name), fr.canonical_format_name(str(fmt)))
            for name, fmt in assignment.items()
        )
    )


def _lane_specs_graph_key(lanes: Sequence[_LaneSpec]) -> tuple[tuple, ...]:
    return tuple(
        (
            str(lane.name),
            fr.canonical_format_name(str(lane.fmt)),
            None if lane.baseline_index is None else int(lane.baseline_index),
            bool(lane.is_baseline),
        )
        for lane in lanes
    )


def _override_sets_graph_key(
    overrides: Sequence[Mapping[str, str]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(_assignment_graph_key(override) for override in overrides)


def _split_lanes(tensor: torch.Tensor, base_batch: int, lane_count: int):
    if tensor.dim() == 0 or tensor.size(0) != base_batch * lane_count:
        return None
    return tensor.split(base_batch, dim=0)


def _coord_replay_target_keys(
    replay_cache: LayerHiddenStateCache,
    target_names: set[str],
) -> tuple[set[object], set[int]]:
    by_name = getattr(replay_cache, "_linear_targets_by_name", {})
    target_keys: set[object] = set()
    module_ids: set[int] = set()
    for raw_name in target_names:
        candidates = [raw_name]
        if raw_name.endswith(".weight"):
            candidates.append(raw_name[:-7])
        else:
            candidates.append(f"{raw_name}.weight")
        for name in candidates:
            target = by_name.get(name)
            if target is None:
                continue
            target_keys.add(target.key)
            module_ids.add(id(target.module))
            break
    return target_keys, module_ids


def _repeat_replay_template_for_lanes(template, lane_count: int, base_batch: int):
    return replace(
        template,
        args=tuple(
            _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for value in template.args
        ),
        kwargs={
            key: _repeat_layer_value_for_lanes(value, lane_count, base_batch)
            for key, value in template.kwargs.items()
        },
    )


def _lane_replay_cache_logits(
    replay_cache: LayerHiddenStateCache,
    layer_idx: int,
    *,
    lane_count: int,
    base_batch: int,
    target_names: set[str],
    last_token_only: bool = False,
) -> torch.Tensor:
    """Replay a populated LayerHiddenStateCache with lane-repeated state.

    LayerHiddenStateCache intentionally exposes scalar replay. Coord descent
    keeps lane semantics here by temporarily repeating the cached layer input
    and non-hidden layer-call tensors, while leaving target modules at live
    BF16 weights so _DepthGroupTargetHooks can choose the per-lane format.
    """
    original_inputs = list(replay_cache.layer_inputs)
    original_templates = list(getattr(replay_cache, "_layer_call_templates"))
    original_baseline_weights = dict(
        getattr(replay_cache, "_baseline_weight_values")
    )
    original_activation_quantizers = dict(
        getattr(replay_cache, "_activation_quantizers")
    )
    target_keys, target_module_ids = _coord_replay_target_keys(
        replay_cache,
        target_names,
    )
    try:
        replay_cache.layer_inputs = list(original_inputs)
        replay_cache.layer_inputs[layer_idx] = _repeat_layer_value_for_lanes(
            original_inputs[layer_idx],
            lane_count,
            base_batch,
        )
        repeated_templates = list(original_templates)
        for idx in range(layer_idx, len(repeated_templates)):
            repeated_templates[idx] = _repeat_replay_template_for_lanes(
                repeated_templates[idx],
                lane_count,
                base_batch,
            )
        replay_cache._layer_call_templates = repeated_templates
        replay_cache._baseline_weight_values = {
            key: value
            for key, value in original_baseline_weights.items()
            if key not in target_keys
        }
        replay_cache._activation_quantizers = {
            module_id: value
            for module_id, value in original_activation_quantizers.items()
            if module_id not in target_module_ids
        }
        return replay_cache.replay_from(
            layer_idx,
            last_token_only=last_token_only,
        )
    finally:
        replay_cache.layer_inputs = original_inputs
        replay_cache._layer_call_templates = original_templates
        replay_cache._baseline_weight_values = original_baseline_weights
        replay_cache._activation_quantizers = original_activation_quantizers


def _override_replay_cache_logits(
    replay_cache: LayerHiddenStateCache,
    layer_idx: int,
    *,
    lane_count: int,
    base_batch: int,
    target_names: set[str],
    last_token_only: bool = False,
) -> torch.Tensor:
    """Replay a populated cache with one full override-set per lane."""
    return _lane_replay_cache_logits(
        replay_cache,
        layer_idx,
        lane_count=lane_count,
        base_batch=base_batch,
        target_names=target_names,
        last_token_only=last_token_only,
    )


def _l3_quantizable_map(model: nn.Module) -> dict[str, tuple[nn.Module, str]]:
    """Map L3 names to modules, including tiny nn.Linear modules in tests."""
    out = dict(build_quantizable_map(model))
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        names = {name, f"{name}.weight"}
        if name.startswith("model."):
            suffix = name[len("model."):]
            names.add(f"model.language_model.{suffix}")
            names.add(f"model.language_model.{suffix}.weight")
        for candidate in names:
            out.setdefault(candidate, (module, "weight"))
    return out


def apply_format_quantization(
    weight: torch.Tensor,
    spec: fr.FormatSpec,
) -> torch.Tensor:
    return spec.quantize_dequantize(weight.detach().clone())


def build_quant_weight_cache(
    model: nn.Module,
    neighborhood: list[L3NeighborhoodEntry],
    specs: list[fr.FormatSpec],
    *,
    skip_bf16: bool = True,
    production_weight_cache=None,
) -> QuantWeightCache:
    quant_map = _l3_quantizable_map(model)
    cache: dict[tuple[str, str], torch.Tensor] = {}
    for entry in neighborhood:
        target = quant_map.get(entry.name)
        if target is None:
            continue
        linear, attr = target
        if not isinstance(linear, nn.Linear) or attr != "weight":
            continue
        name_keys = {
            name
            for name, (candidate_module, candidate_attr) in quant_map.items()
            if candidate_module is linear and candidate_attr == attr
        }
        name_keys.add(entry.name)
        original_weight = linear.weight.data
        for spec in specs:
            canonical = fr.canonical_format_name(spec.name)
            if skip_bf16 and canonical == "BF16":
                continue
            enforce_gpu_memory_budget(
                device=original_weight.device
                if original_weight.device.type == "cuda" else None,
                reason="L3 quant weight cache fill",
            )
            production = (
                production_weight_cache.get(entry.name, canonical)
                if production_weight_cache is not None
                else None
            )
            if production is not None:
                quantized = production.to(
                    device=original_weight.device,
                    dtype=original_weight.dtype,
                )
            else:
                if (
                    production_weight_cache is not None
                    and canonical not in {"BF16", "MXFP8", "MXFP8_E4M3"}
                    and _env_flag_enabled("PRISMAQUANT_STRICT_PRODUCTION_CACHE")
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({entry.name!r}, {canonical!r}); set "
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to fall back "
                        "to RTN, or rebuild the production cache."
                    )
                quantized = apply_format_quantization(original_weight, spec).to(
                    device=original_weight.device,
                    dtype=original_weight.dtype,
                )
            quantized = quantized.contiguous()
            fmt_keys = {canonical, spec.name, *fr.aliases_for(spec.name)}
            for name_key in name_keys:
                for fmt_key in fmt_keys:
                    cache[(name_key, fmt_key)] = quantized
    return QuantWeightCache(cache)


def _production_activation_max_abs(production_weight_cache) -> dict[str, float]:
    if production_weight_cache is None:
        return {}
    return dict(
        getattr(production_weight_cache, "activation_max_abs", None)
        or getattr(production_weight_cache, "activation_scales", None)
        or {}
    )


def _prefetch_production_weight_cache(
    production_weight_cache,
    entries: Sequence[L3NeighborhoodEntry],
) -> None:
    if production_weight_cache is None or not hasattr(production_weight_cache, "prefetch"):
        return
    if getattr(production_weight_cache, "_prismaquant_prefetch_policy", "batch") == "none":
        return
    keys: list[tuple[str, str]] = []
    for entry in entries:
        for fmt in entry.formats:
            canonical = fr.canonical_format_name(fmt)
            if canonical == "BF16":
                continue
            keys.append((entry.name, canonical))
    if keys:
        production_weight_cache.prefetch(keys)


class _DepthGroupTargetHooks:
    """Apply lane-specific target formats for one depth-group microbatch.

    The normal L2 context hooks are installed for every non-target module.
    Group targets are excluded from that context, then these hooks apply either
    the lane's candidate format, that target's paired BF16 baseline, or the
    target's original L2 format for lanes belonging to other targets in the
    same depth group. This avoids double-quantizing a target module while
    preserving "all other modules at the L2 assignment" semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        specs_by_name: Mapping[str, fr.FormatSpec],
        lanes: list[_LaneSpec],
        *,
        base_batch: int,
        quant_weight_cache: QuantWeightCache | None = None,
        include_activation_quant: bool = True,
        activation_max_abs: Mapping[str, float] | None = None,
    ):
        self.model = model
        self.assignment = _canonical_assignment(assignment)
        self.specs_by_name = specs_by_name
        self.lanes = lanes
        self.base_batch = int(base_batch)
        self.quant_weight_cache = quant_weight_cache
        self.include_activation_quant = bool(include_activation_quant)
        self.activation_max_abs = dict(activation_max_abs or {})
        self.handles = []
        self.missing: list[str] = []

    def install(self) -> None:
        quant_map = _l3_quantizable_map(self.model)
        target_names = sorted({lane.name for lane in self.lanes})
        for name in target_names:
            target = quant_map.get(name)
            if target is None:
                self.missing.append(name)
                continue
            module, _attr = target
            if not isinstance(module, nn.Linear):
                self.missing.append(name)
                continue
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _format_for_lane(self, module_name: str, lane: _LaneSpec) -> str:
        if lane.name == module_name:
            return "BF16" if lane.is_baseline else lane.fmt
        return self.assignment.get(module_name, "BF16")

    def _make_hook(self, module_name: str):
        def _hook(module, args, kwargs, output):
            y = _first_tensor_output(output)
            if y is None:
                return output
            chunks = _split_lanes(y, self.base_batch, len(self.lanes))
            if chunks is None:
                return output
            x = None
            for value in list(args) + list((kwargs or {}).values()):
                if isinstance(value, torch.Tensor):
                    x = value
                    break
            if x is None:
                return output
            x_chunks = _split_lanes(x, self.base_batch, len(self.lanes))
            if x_chunks is None:
                return output

            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None

            out_chunks = []
            for lane, y_lane, x_lane in zip(self.lanes, chunks, x_chunks):
                fmt = self._format_for_lane(module_name, lane)
                if fmt == "BF16":
                    out_chunks.append(y_lane)
                    continue
                spec = self.specs_by_name.get(fmt)
                if spec is None:
                    out_chunks.append(y_lane)
                    continue
                w_hat = None
                if self.quant_weight_cache is not None:
                    w_hat = self.quant_weight_cache.get(module_name, fmt)
                if w_hat is None:
                    w_hat = apply_format_quantization(weight, spec)
                x_hat = (
                    spec.activation_quantize_dequantize(
                        _maybe_clip_activations(
                            x_lane,
                            self.activation_max_abs,
                            module_name,
                        )
                    )
                    if self.include_activation_quant
                    else x_lane
                )
                out_chunks.append(
                    F.linear(x_hat, w_hat.to(weight.dtype), bias)
                )

            replacement = torch.cat(out_chunks, dim=0)
            return _replace_first_tensor_output(output, replacement)

        return _hook


class _OverrideSetTargetHooks:
    """Apply one override mapping per lane for interaction probes."""

    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        specs_by_name: Mapping[str, fr.FormatSpec],
        lane_overrides: list[Mapping[str, str]],
        *,
        base_batch: int,
        quant_weight_cache: QuantWeightCache | None = None,
        include_activation_quant: bool = True,
        activation_max_abs: Mapping[str, float] | None = None,
    ):
        self.model = model
        self.assignment = _canonical_assignment(assignment)
        self.specs_by_name = specs_by_name
        self.lane_overrides = [
            {
                str(name): fr.canonical_format_name(fmt)
                for name, fmt in override.items()
            }
            for override in lane_overrides
        ]
        self.base_batch = int(base_batch)
        self.quant_weight_cache = quant_weight_cache
        self.include_activation_quant = bool(include_activation_quant)
        self.activation_max_abs = dict(activation_max_abs or {})
        self.handles = []
        self.missing: list[str] = []

    def install(self) -> None:
        quant_map = _l3_quantizable_map(self.model)
        target_names = sorted(
            {
                name
                for override in self.lane_overrides
                for name in override
            }
        )
        for name in target_names:
            target = quant_map.get(name)
            if target is None:
                self.missing.append(name)
                continue
            module, _attr = target
            if not isinstance(module, nn.Linear):
                self.missing.append(name)
                continue
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _format_for_lane(self, module_name: str, lane_idx: int) -> str:
        override = self.lane_overrides[lane_idx]
        return override.get(module_name, self.assignment.get(module_name, "BF16"))

    def _make_hook(self, module_name: str):
        def _hook(module, args, kwargs, output):
            y = _first_tensor_output(output)
            if y is None:
                return output
            chunks = _split_lanes(y, self.base_batch, len(self.lane_overrides))
            if chunks is None:
                return output
            x = None
            for value in list(args) + list((kwargs or {}).values()):
                if isinstance(value, torch.Tensor):
                    x = value
                    break
            if x is None:
                return output
            x_chunks = _split_lanes(x, self.base_batch, len(self.lane_overrides))
            if x_chunks is None:
                return output

            out_chunks = []
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            for lane_idx, (y_lane, x_lane) in enumerate(zip(chunks, x_chunks)):
                fmt = self._format_for_lane(module_name, lane_idx)
                if fmt == "BF16":
                    out_chunks.append(y_lane)
                    continue
                spec = self.specs_by_name.get(fmt)
                if spec is None:
                    out_chunks.append(y_lane)
                    continue
                w_hat = None
                if self.quant_weight_cache is not None:
                    w_hat = self.quant_weight_cache.get(module_name, fmt)
                if w_hat is None:
                    w_hat = apply_format_quantization(weight, spec)
                x_hat = (
                    spec.activation_quantize_dequantize(
                        _maybe_clip_activations(
                            x_lane,
                            self.activation_max_abs,
                            module_name,
                        )
                    )
                    if self.include_activation_quant
                    else x_lane
                )
                out_chunks.append(F.linear(x_hat, w_hat.to(weight.dtype), bias))
            replacement = torch.cat(out_chunks, dim=0)
            return _replace_first_tensor_output(output, replacement)

        return _hook


class _LaneOutputMSE:
    def __init__(
        self,
        model: nn.Module,
        names: list[str],
        lanes: list[_LaneSpec],
        *,
        base_batch: int,
    ):
        self.model = model
        self.names = names
        self.lanes = lanes
        self.base_batch = int(base_batch)
        self.handles = []
        self.total_by_lane = [0.0 for _ in lanes]
        self.batch_count = 0

    def install(self) -> None:
        quant_map = _l3_quantizable_map(self.model)
        seen_modules: set[int] = set()
        for name in self.names:
            target = quant_map.get(name)
            if target is None:
                continue
            module, _attr = target
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            self.handles.append(
                module.register_forward_hook(self._hook, with_kwargs=True)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def mark_batch(self) -> None:
        self.batch_count += 1

    def _hook(self, _module, _args, _kwargs, output):
        y = _first_tensor_output(output)
        if y is None:
            return output
        chunks = _split_lanes(y.detach(), self.base_batch, len(self.lanes))
        if chunks is None:
            return output
        for idx, lane in enumerate(self.lanes):
            if lane.is_baseline or lane.baseline_index is None:
                continue
            base = chunks[lane.baseline_index].float()
            cand = chunks[idx].float()
            self.total_by_lane[idx] += float((cand - base).pow(2).mean().item())
        return output

    def value_for_lane(self, lane_index: int) -> float:
        return self.total_by_lane[lane_index] / max(self.batch_count, 1)


def _ordered_quantizable_names(model: nn.Module, assignment_names: set[str]) -> list[str]:
    quant_map = _l3_quantizable_map(model)
    module_to_names: dict[int, list[str]] = {}
    for name in assignment_names:
        target = quant_map.get(name)
        if target is None:
            continue
        module_to_names.setdefault(id(target[0]), []).append(name)

    ordered: list[str] = []
    seen: set[str] = set()
    for _qname, module in model.named_modules():
        for name in sorted(module_to_names.get(id(module), [])):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


def _downstream_names_for_group(
    ordered_names: list[str],
    group_names: set[str],
) -> list[str]:
    positions = [
        idx for idx, name in enumerate(ordered_names)
        if name in group_names
    ]
    if not positions:
        return []
    return ordered_names[min(positions):]


def _lane_specs_for_entries(
    entries: list[L3NeighborhoodEntry],
    *,
    include_baseline: bool = True,
) -> list[_LaneSpec]:
    lanes: list[_LaneSpec] = []
    for entry in entries:
        candidate_fmts = [
            fr.canonical_format_name(fmt)
            for fmt in entry.formats
            if not include_baseline or fr.canonical_format_name(fmt) != "BF16"
        ]
        if not candidate_fmts:
            continue
        if not include_baseline:
            for fmt in candidate_fmts:
                lanes.append(
                    _LaneSpec(
                        name=entry.name,
                        fmt=fmt,
                        baseline_index=None,
                        is_baseline=False,
                    )
                )
            continue
        baseline_idx = len(lanes)
        lanes.append(
            _LaneSpec(
                name=entry.name,
                fmt="BF16",
                baseline_index=None,
                is_baseline=True,
            )
        )
        for fmt in candidate_fmts:
            lanes.append(
                _LaneSpec(
                    name=entry.name,
                    fmt=fmt,
                    baseline_index=baseline_idx,
                    is_baseline=False,
                )
            )
    return lanes


def _lane_microbatches_for_entries(
    entries: list[L3NeighborhoodEntry],
    max_lanes_per_batch: int,
    *,
    include_baseline: bool = True,
) -> list[list[_LaneSpec]]:
    batches: list[list[_LaneSpec]] = []
    current: list[_LaneSpec] = []
    max_lanes = max(int(max_lanes_per_batch), 1)
    for entry in entries:
        entry_lanes = _lane_specs_for_entries(
            [entry],
            include_baseline=include_baseline,
        )
        if not entry_lanes:
            continue
        if current and len(current) + len(entry_lanes) > max_lanes:
            batches.append(current)
            current = []
        if len(entry_lanes) > max_lanes:
            batches.append(entry_lanes)
        else:
            current.extend(entry_lanes)
    if current:
        batches.append(current)
    return batches


def _specs_by_canonical_name(format_names: set[str]) -> dict[str, fr.FormatSpec]:
    specs_by_name: dict[str, fr.FormatSpec] = {}
    for fmt in sorted(format_names):
        canonical = fr.canonical_format_name(fmt)
        if canonical == "BF16":
            continue
        spec = fr.get_format(canonical)
        specs_by_name[spec.name] = spec
        specs_by_name[canonical] = spec
        for alias in fr.aliases_for(spec.name):
            specs_by_name[alias] = spec
    return specs_by_name


def _entries_for_candidate_flips(
    candidate_flips: list[tuple[str, str]],
    assignment: Mapping[str, str],
) -> list[L3NeighborhoodEntry]:
    return [
        L3NeighborhoodEntry(
            name=str(name),
            current_format=assignment.get(str(name), "BF16"),
            formats=(fr.canonical_format_name(fmt),),
            margin=0.0,
            l2_current_cost=0.0,
        )
        for name, fmt in candidate_flips
    ]


def _cuda_graph_lane_count(lane_count: int) -> int:
    for candidate in (1, 2, 4, 8, 16, 32, 64):
        if int(lane_count) <= candidate:
            return candidate
    return int(lane_count)


def _pad_lanes_for_cuda_graph(lanes: list[_LaneSpec]) -> list[_LaneSpec]:
    padded_count = _cuda_graph_lane_count(len(lanes))
    if padded_count <= len(lanes) or not lanes:
        return lanes
    dummy_source = lanes[-1]
    padded = list(lanes)
    padded.extend(
        _LaneSpec(
            name=dummy_source.name,
            fmt="BF16",
            baseline_index=None,
            is_baseline=True,
        )
        for _ in range(padded_count - len(lanes))
    )
    return padded


def _pad_override_lanes_for_cuda_graph(
    lane_overrides: list[dict[str, str]],
    target_names: set[str],
) -> list[dict[str, str]]:
    padded_count = _cuda_graph_lane_count(len(lane_overrides))
    if padded_count <= len(lane_overrides) or not lane_overrides:
        return lane_overrides
    padded = [dict(override) for override in lane_overrides]
    baseline_override = {name: "BF16" for name in sorted(target_names)}
    padded.extend(
        dict(baseline_override)
        for _ in range(padded_count - len(lane_overrides))
    )
    return padded


def _override_sets_microbatches(
    overrides: Sequence[Mapping[str, str]],
    max_lanes_per_batch: int,
) -> list[list[dict[str, str]]]:
    batch_size = max(int(max_lanes_per_batch), 1)
    normalised = [_normalise_override_set(override) for override in overrides]
    return [
        normalised[start:start + batch_size]
        for start in range(0, len(normalised), batch_size)
    ]


def _calibration_sample_tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_calibration_sample_tensor_bytes(child) for child in value.values())
    if isinstance(value, tuple | list):
        return sum(_calibration_sample_tensor_bytes(child) for child in value)
    return 0


def _estimate_l3_microbatch_memory_bytes(
    calibration_data,
    lane_count: int,
    *,
    calib_microbatch_size: int = 1,
) -> int:
    if isinstance(calibration_data, torch.Tensor):
        if calibration_data.dim() == 0 or calibration_data.size(0) == 0:
            sample = calibration_data
        else:
            sample = calibration_data[:1]
        base_bytes = _calibration_sample_tensor_bytes(sample)
    elif isinstance(calibration_data, Mapping):
        base_bytes = _calibration_sample_tensor_bytes(calibration_data)
    elif isinstance(calibration_data, (tuple, list)) and calibration_data:
        base_bytes = _calibration_sample_tensor_bytes(calibration_data[0])
    else:
        base_bytes = 0
    microbatch = max(int(calib_microbatch_size), 1)
    return int(base_bytes * microbatch * max(int(lane_count), 1) * 4)


def _calibration_call_count(calibration_data) -> int:
    if isinstance(calibration_data, torch.Tensor):
        if calibration_data.dim() == 0:
            return 1
        return max(int(calibration_data.size(0)), 1)
    if isinstance(calibration_data, Mapping):
        first_tensor = _first_cuda_tensor(calibration_data)
        if first_tensor is not None and first_tensor.dim() > 0:
            return max(int(first_tensor.size(0)), 1)
        return 1
    try:
        return max(len(calibration_data), 1)
    except TypeError:
        return 1


def _adjust_l3_max_lanes_for_memory(
    max_lanes_per_batch: int,
    calibration_data,
    device: torch.device,
    *,
    calib_microbatch_size: int = 1,
) -> int:
    requested = max(int(max_lanes_per_batch), 1)
    if device.type != "cuda" or not torch.cuda.is_available():
        return requested
    info = cuda_memory_info(device)
    if info is None:
        return requested
    free_bytes, total_bytes = info
    headroom_bytes = max(
        int(total_bytes * max(
            _env_float("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_FRACTION", 0.05),
            0.0,
        )),
        int(
            max(_env_float("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB", 2.0), 0.0)
            * 1024 ** 3
        ),
    )
    if int(free_bytes) < headroom_bytes:
        return max(requested // 2, 1)
    lanes = requested
    while lanes > 1:
        estimated_bytes = _estimate_l3_microbatch_memory_bytes(
            calibration_data,
            lanes,
            calib_microbatch_size=calib_microbatch_size,
        )
        if int(free_bytes) >= headroom_bytes + estimated_bytes:
            break
        lanes = max(lanes // 2, 1)
    return max(lanes, 1)


def _pick_l3_calib_microbatch_for_memory(
    requested: int,
    calibration_data,
    lane_count: int,
    device: torch.device,
) -> int:
    """Step a requested calibration-microbatch ceiling down until it fits.

    Mirrors :func:`_adjust_l3_max_lanes_for_memory`: assumes lane batching
    has already settled on ``lane_count`` and finds the largest power-of-two-
    style microbatch that, combined with the lane count, fits within the
    free GPU memory minus the configured headroom.  Returns 1 if memory is
    unavailable or the device is CPU, preserving historical behaviour.
    """
    requested = max(int(requested), 1)
    if requested == 1 or device.type != "cuda" or not torch.cuda.is_available():
        return 1
    info = cuda_memory_info(device)
    if info is None:
        return 1
    free_bytes, total_bytes = info
    headroom_bytes = max(
        int(total_bytes * max(
            _env_float("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_FRACTION", 0.05),
            0.0,
        )),
        int(
            max(_env_float("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB", 2.0), 0.0)
            * 1024 ** 3
        ),
    )
    micro = requested
    while micro > 1:
        estimated_bytes = _estimate_l3_microbatch_memory_bytes(
            calibration_data,
            lane_count,
            calib_microbatch_size=micro,
        )
        if int(free_bytes) >= headroom_bytes + estimated_bytes:
            break
        micro = max(micro // 2, 1)
    return max(micro, 1)


def _output_mse_names_reach_tail(
    names: list[str],
    group_depth: int | None,
) -> bool:
    if group_depth is None:
        return bool(names)
    for name in names:
        depth = layer_depth(name)
        if depth is None or depth > group_depth:
            return True
    return False


@torch.no_grad()
def measure_lane_batched_kl_deltas(
    model: nn.Module,
    baseline_assignment: Mapping[str, str],
    candidate_flips: list[tuple[str, str]],
    calib_ids: torch.Tensor,
    ref_log_probs: list[torch.Tensor],
    *,
    work_root: Path,
    max_lanes_per_batch: int = 64,
    profile=None,
    replay_cache: LayerHiddenStateCache | None = None,
    kl_scope: KLScope | None = None,
    calib_microbatch_size: int = 1,
    include_activation_quant: bool = True,
    use_cuda_graphs: bool | None = None,
    use_replay_cache: bool | None = None,
    production_weight_cache=None,
) -> list[float]:
    """Measure end-KL for each candidate flip applied to baseline_assignment.

    Each lane is one ``(Linear, format)`` override. Lanes may target different
    Linear modules; target hooks apply the candidate format for the matching
    lane and the baseline assignment for all other target modules in that lane.

    ``calib_microbatch_size`` (default 1) stacks that many calibration rows
    into each forward to amortize Python and kernel-launch overhead.  When >1,
    ``ref_log_probs`` is reorganized into matching microbatches before the
    inner KL loop; the function still expects one entry per calibration row
    on input.
    """
    if not candidate_flips:
        return []
    effective_kl_scope = resolve_kl_scope(kl_scope)
    full_sequence_kl = effective_kl_scope == "full_sequence"

    assignment_c = _canonical_assignment(baseline_assignment)
    flips = [
        (str(name), fr.canonical_format_name(fmt))
        for name, fmt in candidate_flips
    ]
    format_names = set(assignment_c.values()) | {fmt for _name, fmt in flips}
    specs_by_name = _specs_by_canonical_name(format_names)
    activation_max_abs = _production_activation_max_abs(production_weight_cache)

    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 1)
    calib_microbatch_size = max(int(calib_microbatch_size), 1)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calib_ids,
        device,
        calib_microbatch_size=calib_microbatch_size,
    )

    # Reorganize ref_log_probs into microbatches matching calib_microbatch_size
    # so the per-microbatch teacher tensor has a (B, L, V) shape that matches
    # the chunk shape the lane-split code emits.  When microbatch_size==1 this
    # is a no-op (each entry stays its original (1, L, V) shape).
    if calib_microbatch_size > 1 and isinstance(ref_log_probs, list) and ref_log_probs:
        regrouped: list[torch.Tensor] = []
        for start in range(0, len(ref_log_probs), calib_microbatch_size):
            window = ref_log_probs[start:start + calib_microbatch_size]
            if all(isinstance(t, torch.Tensor) for t in window):
                regrouped.append(torch.cat(list(window), dim=0))
            else:
                regrouped.extend(window)
        ref_log_probs = regrouped
    entries = _entries_for_candidate_flips(flips, assignment_c)
    batches = _lane_microbatches_for_entries(
        entries,
        max_lanes_per_batch,
        include_baseline=False,
    )
    cal_hash = calibration_data_hash(calib_ids)
    tmp_parent = str(work_root) if work_root is not None else None
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_prequant_cache = _maybe_disable_l3_prequant_cache_for_memory(
        device, use_prequant_cache)
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _maybe_disable_l3_frozen_cache_for_memory(
        device, use_frozen_perturbed_cache)
    calibration_call_count = max(
        len(ref_log_probs),
        _calibration_call_count(calib_ids),
    )
    assignment_key = tuple(sorted(assignment_c.items()))
    rng_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    measured: list[float] = []
    _lb_t0 = time.monotonic()
    _lb_total = sum(1 for b in batches if b)
    _lb_idx = 0
    for lanes in batches:
        if not lanes:
            continue
        _lb_idx += 1
        _lb_t_batch = time.monotonic()
        target_names = {lane.name for lane in lanes}
        context_assignment = {
            name: fmt
            for name, fmt in assignment_c.items()
            if name not in target_names
        }
        cache_entries = [
            L3NeighborhoodEntry(
                name=name,
                current_format=assignment_c.get(name, "BF16"),
                formats=tuple(
                    sorted({
                        assignment_c.get(name, "BF16"),
                        *[
                            lane.fmt
                            for lane in lanes
                            if lane.name == name
                        ],
                    })
                ),
                margin=0.0,
                l2_current_cost=0.0,
            )
            for name in sorted(target_names)
        ]
        group_quant_cache = (
            (
                _prefetch_production_weight_cache(
                    production_weight_cache,
                    cache_entries,
                )
                or build_quant_weight_cache(
                    model,
                    cache_entries,
                    list({id(spec): spec for spec in specs_by_name.values()}.values()),
                    production_weight_cache=production_weight_cache,
                )
            )
            if use_prequant_cache
            else None
        )
        target_depths = [layer_depth(lane.name) for lane in lanes]
        replay_layer_idx = (
            min(depth for depth in target_depths if depth is not None)
            if (
                replay_cache is not None
                and target_depths
                and all(depth is not None for depth in target_depths)
            )
            else None
        )
        replay_cache_enabled = (
            _env_flag_enabled(
                "PRISMAQUANT_COORD_REPLAY_CACHE",
                default=False,
            )
            if use_replay_cache is None
            else bool(use_replay_cache)
        )
        use_replay_cache_now = (
            replay_cache is not None
            and replay_layer_idx is not None
            and 0 <= replay_layer_idx < len(replay_cache.layers)
            and replay_cache_enabled
        )
        if use_replay_cache_now:
            if use_cuda_graphs is None:
                use_coord_lane_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
                    "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
                    default="auto",
                    call_count=1,
                    min_calls=8,
                )
            else:
                use_coord_lane_cuda_graphs = bool(use_cuda_graphs)
            target_hooks = None
            # GPU-resident accumulator: defer the GPU→CPU sync until the
            # whole lane-batch finishes, instead of once per (lane × sample).
            kl_totals = torch.zeros(
                len(lanes), device=device, dtype=torch.float32,
            )
            batch_count = (
                int(calib_ids.size(0))
                if isinstance(calib_ids, torch.Tensor)
                else 0
            )
            base_batch = batch_count
            rng_cm = torch.random.fork_rng(devices=rng_devices)
            try:
                with rng_cm:
                    torch.manual_seed(0)
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.manual_seed_all(0)
                    target_hooks = _DepthGroupTargetHooks(
                        model,
                        assignment_c,
                        specs_by_name,
                        lanes,
                        base_batch=base_batch,
                        quant_weight_cache=group_quant_cache,
                        include_activation_quant=include_activation_quant,
                        activation_max_abs=activation_max_abs,
                    )
                    target_hooks.install()
                    lane_key = tuple(
                        (lane.name, lane.fmt, lane.baseline_index, lane.is_baseline)
                        for lane in lanes
                    )

                    def _replay_forward():
                        logits = _extract_logits(
                            _lane_replay_cache_logits(
                                replay_cache,
                                int(replay_layer_idx),
                                lane_count=len(lanes),
                                base_batch=base_batch,
                                target_names=target_names,
                                last_token_only=not full_sequence_kl,
                            )
                        )
                        if logits.dim() >= 3:
                            if full_sequence_kl:
                                return logits.clone()
                            return logits[:, -1:, :].clone()
                        return logits

                    # In private-pool no-clone mode this returns the static graph
                    # tensor; the KL loop consumes it before the next replay.
                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-lane-replay",
                        (
                            "replay",
                            id(model),
                            id(replay_cache),
                            assignment_key,
                            cal_hash,
                            int(replay_layer_idx),
                            int(len(lanes)),
                            int(base_batch),
                            effective_kl_scope,
                            bool(include_activation_quant),
                            lane_key,
                            tuple(sorted(target_names)),
                            id(production_weight_cache)
                            if production_weight_cache is not None else 0,
                        ),
                        _replay_forward,
                        enabled=use_coord_lane_cuda_graphs,
                        device=device,
                        keepalive=(
                            replay_cache,
                            target_hooks,
                            group_quant_cache,
                        ),
                    )
                    logits = _extract_logits(logits)
                    if logits.dim() >= 3 and not full_sequence_kl:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(logits.detach(), base_batch, len(lanes))
                    if chunks is None:
                        raise RuntimeError(
                            "lane-batched coord replay logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lanes)}"
                        )
                    # Vectorize the per-lane KL across all lanes in one batched
                    # GPU op AND keep ``kl_totals`` on the GPU so the entire
                    # lane-batch incurs a single GPU→CPU sync at the end (in
                    # measured.extend), instead of ``lanes × cal_samples``
                    # times via .item().  Profiling on Qwen3-0.6B showed this
                    # Python-side sync was the dominant cost.
                    stacked = torch.stack(chunks, dim=0)
                    for batch_index, teacher in enumerate(ref_log_probs):
                        teacher = teacher.to(stacked.device).float()
                        if teacher.dim() >= 3 and not full_sequence_kl:
                            teacher = teacher[:, -1:, :]
                        student_log_probs = torch.nn.functional.log_softmax(
                            stacked[:, batch_index:batch_index + 1].float(),
                            dim=-1,
                        )
                        teacher_probs = teacher.exp()
                        kl_per_pos = (
                            teacher_probs * (teacher - student_log_probs)
                        ).sum(dim=-1)
                        kl_totals += kl_per_pos.mean(
                            dim=tuple(range(1, kl_per_pos.dim()))
                        )
                missing_targets = set(target_hooks.missing if target_hooks else [])
                if missing_targets:
                    raise RuntimeError(
                        "target module missing or unsupported for lane-batched KL: "
                        + ", ".join(sorted(missing_targets))
                    )
                # Single sync per lane-batch.
                kl_totals_local = kl_totals.detach().cpu().tolist()
                measured.extend(
                    total / max(batch_count, 1)
                    for total in kl_totals_local
                )
            finally:
                if target_hooks is not None:
                    target_hooks.remove()
                if (
                    device.type == "cuda"
                    and torch.cuda.is_available()
                    and _empty_cache_each_replay_batch()
                ):
                    torch.cuda.empty_cache()
            _lb_dt = time.monotonic() - _lb_t_batch
            _lb_elapsed = time.monotonic() - _lb_t0
            _lb_eta = (_lb_elapsed / max(_lb_idx, 1)) * (_lb_total - _lb_idx)
            print(
                f"[lane-kl][replay] batch {_lb_idx}/{_lb_total} "
                f"lanes={len(lanes)} replay_layer={int(replay_layer_idx)} "
                f"dt={_lb_dt:.1f}s elapsed={_lb_elapsed:.0f}s "
                f"ETA={_lb_eta:.0f}s",
                flush=True,
            )
            continue

        cache_dir = Path(tempfile.mkdtemp(
            prefix="prismaquant_coord_lanes_",
            dir=tmp_parent,
        ))
        context_hooks = PerturbedActivationCache(
            model,
            context_assignment,
            cache_dir,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
        )
        frozen_context = (
            context_hooks.frozen_weight_cache()
            if use_frozen_perturbed_cache
            else nullcontext(context_hooks)
        )
        target_hooks = None
        # GPU-resident accumulator: defer the GPU→CPU sync until the whole
        # lane-batch finishes, instead of once per (lane × cal sample).
        kl_totals = torch.zeros(
            len(lanes), device=device, dtype=torch.float32,
        )
        batch_count = 0
        frozen_context_entered = False
        materialized_cm = nullcontext()
        materialized_entered = False
        if use_cuda_graphs is None:
            use_coord_lane_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
                "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
                default="auto",
                call_count=calibration_call_count,
                min_calls=16,
            )
        else:
            use_coord_lane_cuda_graphs = bool(use_cuda_graphs)
        rng_cm = torch.random.fork_rng(devices=rng_devices)
        try:
            frozen_context.__enter__()
            frozen_context_entered = True
            context_hooks.install()
            # Materialize the floor-format frozen weights into the model
            # parameters for the entire calibration loop.  Without this,
            # _apply_weight_quant clones every context-Linear's weight
            # PER FORWARD and the post_hook restores it — for a 0.6B
            # model with ~190 context Linears that's ~750 MB of GPU
            # clone/restore per forward × 128 cal samples × 19 batches
            # = the dominant wall-time cost on small models.
            # measure_assignment_kl already uses this; lane-batched KL
            # was missing it.
            materialized_cm = nullcontext()
            if (
                device.type == "cuda"
                and torch.cuda.is_available()
                and use_frozen_perturbed_cache
                and context_hooks._frozen_weight_cache is not None
            ):
                materialized_cm = context_hooks.materialized_frozen_weights()
            materialized_cm.__enter__()
            materialized_entered = True
            with rng_cm:
                torch.manual_seed(0)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                _fwd_t0 = None
                _fwd_count = 0
                _fwd_log = []
                for batch_index, (args, kwargs) in enumerate(
                    iter_calibration_forwards(
                        calib_ids,
                        device,
                        microbatch_size=calib_microbatch_size,
                    )
                ):
                    base_batch = _first_tensor_batch_size(args, kwargs)
                    if target_hooks is None:
                        target_hooks = _DepthGroupTargetHooks(
                            model,
                            assignment_c,
                            specs_by_name,
                            lanes,
                            base_batch=base_batch,
                            quant_weight_cache=group_quant_cache,
                            include_activation_quant=include_activation_quant,
                            activation_max_abs=activation_max_abs,
                        )
                        target_hooks.install()
                    rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                        args,
                        kwargs,
                        len(lanes),
                    )
                    lane_key = tuple(
                        (lane.name, lane.fmt, lane.baseline_index, lane.is_baseline)
                        for lane in lanes
                    )

                    def _full_forward(*call_args, **call_kwargs):
                        logits = _extract_logits(model(*call_args, **call_kwargs))
                        if logits.dim() >= 3:
                            if full_sequence_kl:
                                return logits.clone()
                            return logits[:, -1:, :].clone()
                        return logits

                    # In private-pool no-clone mode this static output is split
                    # and reduced to scalar KLs before another replay.
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _fwd_start = time.monotonic()
                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-lane-full",
                        (
                            "full",
                            id(model),
                            assignment_key,
                            cal_hash,
                            int(len(lanes)),
                            int(base_batch),
                            effective_kl_scope,
                            bool(include_activation_quant),
                            lane_key,
                            tuple(sorted(target_names)),
                            id(production_weight_cache)
                            if production_weight_cache is not None else 0,
                        ),
                        _full_forward,
                        *rep_args,
                        enabled=use_coord_lane_cuda_graphs,
                        device=device,
                        keepalive=(
                            context_hooks,
                            target_hooks,
                            group_quant_cache,
                        ),
                        **rep_kwargs,
                    )
                    if logits.dim() >= 3 and not full_sequence_kl:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(logits.detach(), base_batch, len(lanes))
                    if chunks is None:
                        raise RuntimeError(
                            "lane-batched coord KL logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lanes)}"
                        )
                    teacher = ref_log_probs[batch_index].to(logits.device)
                    if teacher.dim() >= 3 and not full_sequence_kl:
                        teacher = teacher[:, -1:, :]
                    # Vectorize the per-lane KL across all lanes in one batched
                    # GPU op AND keep ``kl_totals`` on the GPU so the entire
                    # lane-batch incurs a single GPU→CPU sync at the end (in
                    # measured.extend), instead of ``lanes × cal_samples``
                    # times via .item().
                    stacked = torch.stack(chunks, dim=0)
                    student_log_probs = torch.nn.functional.log_softmax(
                        stacked.float(), dim=-1,
                    )
                    teacher_fp32 = teacher.float()
                    teacher_probs = teacher_fp32.exp()
                    kl_per_pos = (
                        teacher_probs * (teacher_fp32 - student_log_probs)
                    ).sum(dim=-1)
                    kl_totals += (
                        kl_per_pos.mean(dim=tuple(range(1, kl_per_pos.dim())))
                        * float(base_batch)
                    )
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _fwd_dt = time.monotonic() - _fwd_start
                    _fwd_count += 1
                    if _fwd_count <= 5 or _fwd_count % 32 == 0:
                        print(
                            f"[fwd-time] cal {_fwd_count}/{int(calibration_call_count)} "
                            f"dt={_fwd_dt*1000:.0f}ms",
                            flush=True,
                        )
                    batch_count += int(base_batch)
            missing_targets = set(target_hooks.missing if target_hooks else [])
            if missing_targets:
                raise RuntimeError(
                    "target module missing or unsupported for lane-batched KL: "
                    + ", ".join(sorted(missing_targets))
                )
            # Single sync per lane-batch.
            kl_totals_local = kl_totals.detach().cpu().tolist()
            measured.extend(
                total / max(batch_count, 1)
                for total in kl_totals_local
            )
        finally:
            if target_hooks is not None:
                target_hooks.remove()
            if context_hooks.installed:
                context_hooks.remove()
            if materialized_entered:
                materialized_cm.__exit__(None, None, None)
            if frozen_context_entered:
                frozen_context.__exit__(None, None, None)
            shutil.rmtree(cache_dir, ignore_errors=True)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        _lb_dt = time.monotonic() - _lb_t_batch
        _lb_elapsed = time.monotonic() - _lb_t0
        _lb_eta = (_lb_elapsed / max(_lb_idx, 1)) * (_lb_total - _lb_idx)
        print(
            f"[lane-kl] batch {_lb_idx}/{_lb_total} lanes={len(lanes)} "
            f"dt={_lb_dt:.1f}s elapsed={_lb_elapsed:.0f}s ETA={_lb_eta:.0f}s",
            flush=True,
        )

    if len(measured) != len(candidate_flips):
        raise RuntimeError(
            "lane-batched coord KL produced "
            f"{len(measured)} results for {len(candidate_flips)} candidates"
        )
    return measured


def _normalise_override_set(
    override: Mapping[str, str],
) -> dict[str, str]:
    return {
        str(name): fr.canonical_format_name(fmt)
        for name, fmt in sorted(override.items())
    }


@torch.no_grad()
def measure_override_set_kl(
    model: nn.Module,
    baseline_assignment: Mapping[str, str],
    candidate_overrides: list[Mapping[str, str]],
    calib_ids: torch.Tensor,
    ref_log_probs: list[torch.Tensor],
    *,
    work_root: Path,
    max_lanes_per_batch: int = 16,
    profile=None,
    replay_cache: LayerHiddenStateCache | None = None,
    kl_scope: KLScope | None = None,
    calib_microbatch_size: int = 1,
    include_activation_quant: bool = True,
    use_cuda_graphs: bool | None = None,
    use_replay_cache: bool | None = None,
    production_weight_cache=None,
) -> list[float]:
    """Measure end-KL for simultaneous multi-Linear override candidates.

    Each lane is a complete override mapping, e.g. q/k/v all set to MXFP8 for
    the vLLM-packed qkv decision unit.  The teacher remains the original BF16
    reference model; all non-target modules stay at ``baseline_assignment``.
    """
    if not candidate_overrides:
        return []

    effective_kl_scope = resolve_kl_scope(kl_scope)
    full_sequence_kl = effective_kl_scope == "full_sequence"
    assignment_c = _canonical_assignment(baseline_assignment)
    overrides = [_normalise_override_set(override) for override in candidate_overrides]

    format_names = set(assignment_c.values())
    for override in overrides:
        format_names.update(override.values())
    specs_by_name = _specs_by_canonical_name(format_names)
    activation_max_abs = _production_activation_max_abs(production_weight_cache)

    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 1)
    calib_microbatch_size = max(int(calib_microbatch_size), 1)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calib_ids,
        device,
        calib_microbatch_size=calib_microbatch_size,
    )
    if calib_microbatch_size > 1 and isinstance(ref_log_probs, list) and ref_log_probs:
        regrouped: list[torch.Tensor] = []
        for start in range(0, len(ref_log_probs), calib_microbatch_size):
            window = ref_log_probs[start:start + calib_microbatch_size]
            if all(isinstance(t, torch.Tensor) for t in window):
                regrouped.append(torch.cat(list(window), dim=0))
            else:
                regrouped.extend(window)
        ref_log_probs = regrouped

    batches = _override_sets_microbatches(overrides, max_lanes_per_batch)
    cal_hash = calibration_data_hash(calib_ids)
    tmp_parent = str(work_root) if work_root is not None else None
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_prequant_cache = _maybe_disable_l3_prequant_cache_for_memory(
        device, use_prequant_cache)
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _maybe_disable_l3_frozen_cache_for_memory(
        device, use_frozen_perturbed_cache)
    calibration_call_count = max(
        len(ref_log_probs),
        _calibration_call_count(calib_ids),
    )
    assignment_key = tuple(sorted(assignment_c.items()))
    rng_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        rng_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    measured: list[float] = []
    _t0 = time.monotonic()
    _total = sum(1 for batch in batches if batch)
    for batch_idx, lane_overrides in enumerate(batches, start=1):
        if not lane_overrides:
            continue
        _batch_t0 = time.monotonic()
        target_names = {
            name
            for override in lane_overrides
            for name in override
        }
        context_assignment = {
            name: fmt
            for name, fmt in assignment_c.items()
            if name not in target_names
        }
        cache_entries = [
            L3NeighborhoodEntry(
                name=name,
                current_format=assignment_c.get(name, "BF16"),
                formats=tuple(
                    sorted({
                        assignment_c.get(name, "BF16"),
                        *[
                            override[name]
                            for override in lane_overrides
                            if name in override
                        ],
                    })
                ),
                margin=0.0,
                l2_current_cost=0.0,
            )
            for name in sorted(target_names)
        ]
        group_quant_cache = (
            (
                _prefetch_production_weight_cache(
                    production_weight_cache,
                    cache_entries,
                )
                or build_quant_weight_cache(
                    model,
                    cache_entries,
                    list({id(spec): spec for spec in specs_by_name.values()}.values()),
                    production_weight_cache=production_weight_cache,
                )
            )
            if use_prequant_cache
            else None
        )
        target_depths = [layer_depth(name) for name in target_names]
        replay_layer_idx = (
            min(depth for depth in target_depths if depth is not None)
            if (
                replay_cache is not None
                and target_depths
                and all(depth is not None for depth in target_depths)
            )
            else None
        )
        replay_cache_enabled = (
            _env_flag_enabled(
                "PRISMAQUANT_COORD_REPLAY_CACHE",
                default=False,
            )
            if use_replay_cache is None
            else bool(use_replay_cache)
        )
        use_replay_cache_now = (
            replay_cache is not None
            and replay_layer_idx is not None
            and 0 <= replay_layer_idx < len(replay_cache.layers)
            and replay_cache_enabled
        )

        if use_replay_cache_now:
            if use_cuda_graphs is None:
                use_coord_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
                    "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
                    default="auto",
                    call_count=1,
                    min_calls=8,
                )
            else:
                use_coord_cuda_graphs = bool(use_cuda_graphs)
            base_batch = (
                int(calib_ids.size(0))
                if isinstance(calib_ids, torch.Tensor)
                else len(ref_log_probs)
            )
            target_hooks = None
            rng_cm = torch.random.fork_rng(devices=rng_devices)
            try:
                with rng_cm:
                    torch.manual_seed(0)
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.manual_seed_all(0)
                    target_hooks = _OverrideSetTargetHooks(
                        model,
                        assignment_c,
                        specs_by_name,
                        lane_overrides,
                        base_batch=base_batch,
                        quant_weight_cache=group_quant_cache,
                        include_activation_quant=include_activation_quant,
                        activation_max_abs=activation_max_abs,
                    )
                    target_hooks.install()

                    def _replay_forward():
                        logits = _extract_logits(
                            _override_replay_cache_logits(
                                replay_cache,
                                int(replay_layer_idx),
                                lane_count=len(lane_overrides),
                                base_batch=base_batch,
                                target_names=target_names,
                                last_token_only=not full_sequence_kl,
                            )
                        )
                        if logits.dim() >= 3:
                            if full_sequence_kl:
                                return logits.clone()
                            return logits[:, -1:, :].clone()
                        return logits

                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-override-replay",
                        (
                            "override-replay",
                            id(model),
                            id(replay_cache),
                            assignment_key,
                            cal_hash,
                            int(replay_layer_idx),
                            int(len(lane_overrides)),
                            int(base_batch),
                            effective_kl_scope,
                            bool(include_activation_quant),
                            _override_sets_graph_key(lane_overrides),
                            tuple(sorted(target_names)),
                            id(production_weight_cache)
                            if production_weight_cache is not None else 0,
                        ),
                        _replay_forward,
                        enabled=use_coord_cuda_graphs,
                        device=device,
                        keepalive=(
                            replay_cache,
                            target_hooks,
                            group_quant_cache,
                        ),
                    )
                    logits = _extract_logits(logits)
                    if logits.dim() >= 3 and not full_sequence_kl:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(
                        logits.detach(), base_batch, len(lane_overrides),
                    )
                    if chunks is None:
                        raise RuntimeError(
                            "override-set replay logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lane_overrides)}"
                        )
                    teacher = torch.cat(
                        [t.to(device).float() for t in ref_log_probs],
                        dim=0,
                    )
                    if teacher.dim() >= 3 and not full_sequence_kl:
                        teacher = teacher[:, -1:, :]
                    stacked = torch.stack(chunks, dim=0).float()
                    student_log_probs = F.log_softmax(stacked, dim=-1)
                    teacher_probs = teacher.exp().unsqueeze(0)
                    teacher_log_probs = teacher.unsqueeze(0)
                    kl_per_pos = (
                        teacher_probs * (teacher_log_probs - student_log_probs)
                    ).sum(dim=-1)
                    kl_values = kl_per_pos.mean(
                        dim=tuple(range(1, kl_per_pos.dim()))
                    )
                    measured.extend(float(v) for v in kl_values.detach().cpu())
                missing_targets = set(target_hooks.missing if target_hooks else [])
                if missing_targets:
                    raise RuntimeError(
                        "target module missing or unsupported for override-set KL: "
                        + ", ".join(sorted(missing_targets))
                    )
            finally:
                if target_hooks is not None:
                    target_hooks.remove()
                if (
                    device.type == "cuda"
                    and torch.cuda.is_available()
                    and _empty_cache_each_replay_batch()
                ):
                    torch.cuda.empty_cache()
            dt = time.monotonic() - _batch_t0
            elapsed = time.monotonic() - _t0
            eta = (elapsed / max(batch_idx, 1)) * (_total - batch_idx)
            print(
                f"[override-kl][replay] batch {batch_idx}/{_total} "
                f"lanes={len(lane_overrides)} targets={len(target_names)} "
                f"replay_layer={int(replay_layer_idx)} dt={dt:.1f}s "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                flush=True,
            )
            continue

        cache_dir = Path(tempfile.mkdtemp(
            prefix="prismaquant_override_lanes_",
            dir=tmp_parent,
        ))
        context_hooks = PerturbedActivationCache(
            model,
            context_assignment,
            cache_dir,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
        )
        frozen_context = (
            context_hooks.frozen_weight_cache()
            if use_frozen_perturbed_cache
            else nullcontext(context_hooks)
        )
        target_hooks = None
        kl_totals = torch.zeros(
            len(lane_overrides), device=device, dtype=torch.float32,
        )
        batch_count = 0
        frozen_context_entered = False
        materialized_cm = nullcontext()
        materialized_entered = False
        if use_cuda_graphs is None:
            use_coord_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
                "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
                default="auto",
                call_count=calibration_call_count,
                min_calls=16,
            )
        else:
            use_coord_cuda_graphs = bool(use_cuda_graphs)
        rng_cm = torch.random.fork_rng(devices=rng_devices)
        try:
            frozen_context.__enter__()
            frozen_context_entered = True
            context_hooks.install()
            if (
                device.type == "cuda"
                and torch.cuda.is_available()
                and use_frozen_perturbed_cache
                and context_hooks._frozen_weight_cache is not None
            ):
                materialized_cm = context_hooks.materialized_frozen_weights()
            materialized_cm.__enter__()
            materialized_entered = True
            with rng_cm:
                torch.manual_seed(0)
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                for batch_index, (args, kwargs) in enumerate(
                    iter_calibration_forwards(
                        calib_ids,
                        device,
                        microbatch_size=calib_microbatch_size,
                    )
                ):
                    base_batch = _first_tensor_batch_size(args, kwargs)
                    if target_hooks is None:
                        target_hooks = _OverrideSetTargetHooks(
                            model,
                            assignment_c,
                            specs_by_name,
                            lane_overrides,
                            base_batch=base_batch,
                            quant_weight_cache=group_quant_cache,
                            include_activation_quant=include_activation_quant,
                            activation_max_abs=activation_max_abs,
                        )
                        target_hooks.install()
                    rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                        args,
                        kwargs,
                        len(lane_overrides),
                    )

                    def _full_forward(*call_args, **call_kwargs):
                        logits = _extract_logits(model(*call_args, **call_kwargs))
                        if logits.dim() >= 3:
                            if full_sequence_kl:
                                return logits.clone()
                            return logits[:, -1:, :].clone()
                        return logits

                    logits = _COORD_LANE_CUDA_GRAPH_REGISTRY.run(
                        "coord-override-full",
                        (
                            "override-full",
                            id(model),
                            assignment_key,
                            cal_hash,
                            int(len(lane_overrides)),
                            int(base_batch),
                            effective_kl_scope,
                            bool(include_activation_quant),
                            _override_sets_graph_key(lane_overrides),
                            tuple(sorted(target_names)),
                            id(production_weight_cache)
                            if production_weight_cache is not None else 0,
                        ),
                        _full_forward,
                        *rep_args,
                        enabled=use_coord_cuda_graphs,
                        device=device,
                        keepalive=(
                            context_hooks,
                            target_hooks,
                            group_quant_cache,
                        ),
                        **rep_kwargs,
                    )
                    if logits.dim() >= 3 and not full_sequence_kl:
                        logits = logits[:, -1:, :]
                    chunks = _split_lanes(
                        logits.detach(), base_batch, len(lane_overrides),
                    )
                    if chunks is None:
                        raise RuntimeError(
                            "override-set KL logits did not preserve lane "
                            f"batching: shape={tuple(logits.shape)} "
                            f"base_batch={base_batch} lanes={len(lane_overrides)}"
                        )
                    teacher = ref_log_probs[batch_index].to(logits.device).float()
                    if teacher.dim() >= 3 and not full_sequence_kl:
                        teacher = teacher[:, -1:, :]
                    stacked = torch.stack(chunks, dim=0).float()
                    student_log_probs = F.log_softmax(stacked, dim=-1)
                    teacher_probs = teacher.exp().unsqueeze(0)
                    teacher_log_probs = teacher.unsqueeze(0)
                    kl_per_pos = (
                        teacher_probs * (teacher_log_probs - student_log_probs)
                    ).sum(dim=-1)
                    kl_totals += kl_per_pos.mean(
                        dim=tuple(range(1, kl_per_pos.dim()))
                    ) * float(base_batch)
                    batch_count += int(base_batch)
            missing_targets = set(target_hooks.missing if target_hooks else [])
            if missing_targets:
                raise RuntimeError(
                    "target module missing or unsupported for override-set KL: "
                    + ", ".join(sorted(missing_targets))
                )
            measured.extend(
                total / max(batch_count, 1)
                for total in kl_totals.detach().cpu().tolist()
            )
        finally:
            if target_hooks is not None:
                target_hooks.remove()
            if context_hooks.installed:
                context_hooks.remove()
            if materialized_entered:
                materialized_cm.__exit__(None, None, None)
            if frozen_context_entered:
                frozen_context.__exit__(None, None, None)
            shutil.rmtree(cache_dir, ignore_errors=True)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        dt = time.monotonic() - _batch_t0
        elapsed = time.monotonic() - _t0
        eta = (elapsed / max(batch_idx, 1)) * (_total - batch_idx)
        print(
            f"[override-kl] batch {batch_idx}/{_total} "
            f"lanes={len(lane_overrides)} targets={len(target_names)} "
            f"dt={dt:.1f}s elapsed={elapsed:.0f}s ETA={eta:.0f}s",
            flush=True,
        )

    if len(measured) != len(candidate_overrides):
        raise RuntimeError(
            "override-set KL produced "
            f"{len(measured)} results for {len(candidate_overrides)} candidates"
        )
    return measured


@torch.no_grad()
def measure_override_paired_kl_deltas(
    model: nn.Module,
    baseline_assignment: Mapping[str, str],
    candidate_overrides: list[Mapping[str, str]],
    calib_ids: torch.Tensor,
    *,
    work_root: Path,
    max_lanes_per_batch: int = 64,
    profile=None,
    progress_callback: Callable[[dict], None] | None = None,
    tail_only: bool = True,
    cache_tail_layer_inputs: bool = True,
) -> list[float]:
    """Measure paired propagated KL for multi-target override sets.

    Each candidate override is paired with a lane where the same target modules
    are forced to BF16 while all other modules stay at ``baseline_assignment``.
    This gives pair/block interaction probes the same baseline semantics as
    ``measure_propagated_costs`` uses for unary L3 costs.
    """

    if not candidate_overrides:
        return []

    assignment_c = _canonical_assignment(baseline_assignment)
    overrides = [
        _normalise_override_set(override)
        for override in candidate_overrides
    ]
    format_names = set(assignment_c.values())
    for override in overrides:
        format_names.update(override.values())
    specs_by_name = _specs_by_canonical_name(format_names)

    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 2)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calib_ids,
        device,
    )
    max_pairs_per_batch = max(int(max_lanes_per_batch) // 2, 1)
    cal_hash = calibration_data_hash(calib_ids)
    tmp_parent = str(work_root) if work_root is not None else None
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_prequant_cache = _maybe_disable_l3_prequant_cache_for_memory(
        device, use_prequant_cache)
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _maybe_disable_l3_frozen_cache_for_memory(
        device, use_frozen_perturbed_cache)
    _decoder_base, decoder_layers = _decoder_stack(model)
    tail_call_cache: dict[int, list[tuple[tuple, dict, int]]] = {}
    tail_graph_cache = _TailCudaGraphCache(enabled=torch.cuda.is_available())
    if (
        bool(tail_only)
        and bool(cache_tail_layer_inputs)
        and decoder_layers is not None
    ):
        needed_depths = set()
        for override in overrides:
            depths = [layer_depth(name) for name in override]
            if depths and all(depth is not None for depth in depths):
                depth = min(int(depth) for depth in depths if depth is not None)
                if 0 <= depth < len(decoder_layers):
                    needed_depths.add(depth)
        if needed_depths:
            cache_dir = Path(tempfile.mkdtemp(
                prefix="prismaquant_pairwise_baseline_context_",
                dir=tmp_parent,
            ))
            context_hooks = PerturbedActivationCache(
                model,
                assignment_c,
                cache_dir,
                input_rows=0,
                cal_hash=cal_hash,
                profile=profile,
            )
            frozen_context = (
                context_hooks.frozen_weight_cache()
                if use_frozen_perturbed_cache
                else nullcontext(context_hooks)
            )
            with frozen_context:
                context_hooks.install()
                try:
                    captured = _capture_all_layer_calls(
                        model,
                        decoder_layers,
                        needed_depths,
                        calib_ids,
                        device,
                    )
                    tail_call_cache = {
                        depth: captured.get(depth, [])
                        for depth in needed_depths
                    }
                finally:
                    context_hooks.remove()
                    shutil.rmtree(cache_dir, ignore_errors=True)

    measured: list[float] = []
    chunk_count = int(math.ceil(len(overrides) / float(max_pairs_per_batch)))
    chunk_index = 0
    start = 0
    while start < len(overrides):
        chunk_index += 1
        _enforce_l3_host_memory_floor(
            phase="paired_override_kl",
            chunk_index=chunk_index,
        )
        chunk_lanes_per_batch = _adjust_l3_max_lanes_for_host_floor(
            max_lanes_per_batch,
            phase="paired_override_kl",
            chunk_index=chunk_index,
        )
        chunk_pairs_per_batch = max(int(chunk_lanes_per_batch) // 2, 1)
        remaining_count = len(overrides) - start
        chunk_count = max(
            chunk_count,
            chunk_index - 1
            + int(math.ceil(remaining_count / float(chunk_pairs_per_batch))),
        )
        chunk = overrides[start:start + chunk_pairs_per_batch]
        chunk_start = time.monotonic()
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "paired_override_chunk_start",
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "override_count": len(chunk),
                    "lane_count": len(chunk) * 2,
                }
            )
        lane_overrides: list[dict[str, str]] = []
        paired_indices: list[tuple[int, int]] = []
        for override in chunk:
            target_names = sorted(override)
            baseline_override = {name: "BF16" for name in target_names}
            baseline_idx = len(lane_overrides)
            lane_overrides.append(baseline_override)
            candidate_idx = len(lane_overrides)
            lane_overrides.append(dict(override))
            paired_indices.append((baseline_idx, candidate_idx))

        target_names = {
            name
            for override in lane_overrides
            for name in override
        }
        target_depths = [layer_depth(name) for name in target_names]
        replay_layer_idx = (
            min(int(depth) for depth in target_depths if depth is not None)
            if (
                bool(tail_only)
                and decoder_layers is not None
                and target_depths
                and all(depth is not None for depth in target_depths)
            )
            else None
        )
        use_tail_chunk = (
            replay_layer_idx is not None
            and 0 <= int(replay_layer_idx) < len(decoder_layers)
        )
        cached_calls_for_graph = (
            tail_call_cache.get(int(replay_layer_idx), [])
            if use_tail_chunk and cache_tail_layer_inputs
            else None
        )
        if not cached_calls_for_graph:
            cached_calls_for_graph = None
        l3_tail_graph_call_count = (
            len(cached_calls_for_graph)
            if cached_calls_for_graph is not None
            else _calibration_call_count(calib_ids)
        )
        use_l3_tail_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
            "PRISMAQUANT_L3_CUDA_GRAPHS",
            default="auto",
            call_count=l3_tail_graph_call_count,
            min_calls=8,
        )
        tail_graph_safe = (
            use_tail_chunk
            and tail_graph_cache.enabled
            and use_l3_tail_cuda_graphs
        )
        execution_lane_overrides = (
            _pad_override_lanes_for_cuda_graph(lane_overrides, target_names)
            if tail_graph_safe
            else lane_overrides
        )
        context_assignment = {
            name: fmt
            for name, fmt in assignment_c.items()
            if name not in target_names
        }
        graph_state_key = (
            _assignment_graph_key(context_assignment),
            _override_sets_graph_key(execution_lane_overrides),
        )
        cache_entries = [
            L3NeighborhoodEntry(
                name=name,
                current_format=assignment_c.get(name, "BF16"),
                formats=tuple(
                    sorted(
                        {
                            assignment_c.get(name, "BF16"),
                            *[
                                override[name]
                                for override in lane_overrides
                                if name in override
                            ],
                        }
                    )
                ),
                margin=0.0,
                l2_current_cost=0.0,
            )
            for name in sorted(target_names)
        ]
        group_quant_cache = (
            build_quant_weight_cache(
                model,
                cache_entries,
                list({id(spec): spec for spec in specs_by_name.values()}.values()),
            )
            if use_prequant_cache
            else None
        )
        cache_dir = Path(tempfile.mkdtemp(
            prefix="prismaquant_pairwise_lanes_",
            dir=tmp_parent,
        ))
        context_hooks = PerturbedActivationCache(
            model,
            context_assignment,
            cache_dir,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
        )
        frozen_context = (
            context_hooks.frozen_weight_cache()
            if use_frozen_perturbed_cache
            else nullcontext(context_hooks)
        )
        target_hooks = None
        kl_totals = [0.0 for _override in chunk]
        batch_count = 0
        frozen_context_entered = False
        try:
            frozen_context.__enter__()
            frozen_context_entered = True
            context_hooks.install()
            call_iter = (
                cached_calls_for_graph
                if cached_calls_for_graph is not None
                else iter_calibration_forwards(calib_ids, device)
            )
            for call_item in call_iter:
                if cached_calls_for_graph is not None:
                    args, kwargs, base_batch = _move_cached_layer_call(
                        call_item,
                        device,
                    )
                else:
                    args, kwargs = call_item
                    base_batch = _first_tensor_batch_size(args, kwargs)
                if target_hooks is None:
                    target_hooks = _OverrideSetTargetHooks(
                        model,
                        assignment_c,
                        specs_by_name,
                        execution_lane_overrides,
                        base_batch=base_batch,
                        quant_weight_cache=group_quant_cache,
                    )
                    target_hooks.install()
                if use_tail_chunk:
                    layer = decoder_layers[int(replay_layer_idx)]
                    if cached_calls_for_graph is not None:
                        layer_args, layer_kwargs = args, kwargs
                    else:
                        layer_args, layer_kwargs = _capture_layer_call(
                            model,
                            layer,
                            args,
                            kwargs,
                        )
                    rep_args, rep_kwargs = _repeat_layer_call_for_lanes(
                        layer_args,
                        layer_kwargs,
                        len(execution_lane_overrides),
                        base_batch,
                    )
                    layer_output = layer(*rep_args, **rep_kwargs)
                    hidden = _first_tensor_output(layer_output)
                    if hidden is None:
                        raise RuntimeError(
                            "tail-only paired override decoder layer returned no tensor"
                        )
                    logits = tail_forward_from_layer(
                        model,
                        int(replay_layer_idx),
                        rep_args,
                        rep_kwargs,
                        hidden,
                        cuda_graph_cache=(
                            tail_graph_cache if tail_graph_safe else None
                        ),
                        lane_count=len(execution_lane_overrides),
                        graph_state_key=graph_state_key,
                    )
                else:
                    rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                        args,
                        kwargs,
                        len(execution_lane_overrides),
                    )
                    logits = _extract_logits(model(*rep_args, **rep_kwargs))
                if logits.dim() >= 3:
                    logits = logits[:, -1:, :]
                chunks = _split_lanes(
                    logits.detach(),
                    base_batch,
                    len(execution_lane_overrides),
                )
                if chunks is None:
                    raise RuntimeError(
                        "paired override KL logits did not preserve lane "
                        f"batching: shape={tuple(logits.shape)} "
                        f"base_batch={base_batch} "
                        f"lanes={len(execution_lane_overrides)}"
                    )
                for idx, (baseline_idx, candidate_idx) in enumerate(paired_indices):
                    teacher = F.log_softmax(
                        chunks[baseline_idx].float(),
                        dim=-1,
                    )
                    kl_totals[idx] += float(
                        kl_divergence(chunks[candidate_idx], teacher).item()
                    )
                batch_count += 1

            missing_targets = set(target_hooks.missing if target_hooks else [])
            if missing_targets:
                raise RuntimeError(
                    "target module missing or unsupported for paired override KL: "
                    + ", ".join(sorted(missing_targets))
                )
            measured.extend(total / max(batch_count, 1) for total in kl_totals)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "paired_override_chunk_end",
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "override_count": len(chunk),
                        "lane_count": len(execution_lane_overrides),
                        "batch_count": batch_count,
                        "elapsed_seconds": time.monotonic() - chunk_start,
                    }
                )
        finally:
            if target_hooks is not None:
                target_hooks.remove()
            if context_hooks.installed:
                context_hooks.remove()
            if frozen_context_entered:
                frozen_context.__exit__(None, None, None)
            shutil.rmtree(cache_dir, ignore_errors=True)
            target_hooks = None
            context_hooks = None
            frozen_context = None
            group_quant_cache = None
            lane_overrides = []
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()
            _enforce_l3_host_memory_floor(
                phase="paired_override_kl_cleanup",
                chunk_index=chunk_index,
            )
        start += len(chunk)

    tail_graph_cache.clear()
    if len(measured) != len(candidate_overrides):
        raise RuntimeError(
            "paired override KL produced "
            f"{len(measured)} results for {len(candidate_overrides)} candidates"
        )
    return measured


@torch.no_grad()
def measure_propagated_costs(
    model: nn.Module,
    assignment: Mapping[str, str],
    neighborhood: list[L3NeighborhoodEntry],
    calibration_data,
    specs: list[fr.FormatSpec],
    *,
    work_root: str | Path | None = None,
    profile=None,
    max_lanes_per_batch: int = 16,
    tail_only: bool = True,
    cache_tail_layer_inputs: bool = True,
    output_mse_names: list[str] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, dict[str, dict]]:
    """Measure paired end-KL and downstream output-MSE for L3 candidates.

    Each non-BF16 candidate lane is paired with a target-specific BF16 lane in
    the same model call, while all non-target modules run under the converged
    L2 assignment. Depth groups are microbatched by lane count so memory stays
    bounded.
    """
    if not neighborhood:
        return {}

    specs_by_name: dict[str, fr.FormatSpec] = {}
    for spec in specs:
        specs_by_name[spec.name] = spec
        specs_by_name[fr.canonical_format_name(spec.name)] = spec
    assignment_c = _canonical_assignment(assignment)
    results: dict[str, dict[str, dict]] = {
        entry.name: {
            "BF16": {
                "propagated_end_kl": 0.0,
                "downstream_output_mse": 0.0,
                "paired_baseline": "target_bf16_under_l2_assignment",
            }
        }
        for entry in neighborhood
        if "BF16" in entry.formats
    }
    ordered_names = _ordered_quantizable_names(model, set(assignment_c))
    all_output_names = ordered_names if output_mse_names is None else output_mse_names
    device = next(model.parameters()).device
    requested_max_lanes_per_batch = max(int(max_lanes_per_batch), 1)
    max_lanes_per_batch = _adjust_l3_max_lanes_for_memory(
        requested_max_lanes_per_batch,
        calibration_data,
        device,
    )
    if (
        progress_callback is not None
        and max_lanes_per_batch != requested_max_lanes_per_batch
    ):
        progress_callback({
            "event": "lane_batch_memory_adjusted",
            "requested_max_lanes_per_batch": requested_max_lanes_per_batch,
            "max_lanes_per_batch": max_lanes_per_batch,
        })
    cal_hash = calibration_data_hash(calibration_data)
    tmp_parent = str(work_root) if work_root is not None else None

    depth_groups = _group_neighborhood_by_depth(neighborhood)
    _decoder_base, decoder_layers = _decoder_stack(model)
    tail_call_cache: dict[int, list[tuple[tuple, dict, int]]] = {}
    use_prequant_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_PREQUANT_CACHE",
        default=True,
    )
    use_prequant_cache = _maybe_disable_l3_prequant_cache_for_memory(
        device, use_prequant_cache)
    use_frozen_perturbed_cache = _env_flag_enabled(
        "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
        default=True,
    )
    use_frozen_perturbed_cache = _maybe_disable_l3_frozen_cache_for_memory(
        device, use_frozen_perturbed_cache)
    tail_graph_cache = _TailCudaGraphCache(enabled=torch.cuda.is_available())
    if (
        bool(tail_only)
        and bool(cache_tail_layer_inputs)
        and decoder_layers is not None
    ):
        needed_depths = {
            layer_depth(group_entries[0].name)
            for _group_key, group_entries in depth_groups
            if group_entries
        }
        needed_depths = {
            depth
            for depth in needed_depths
            if depth is not None and 0 <= depth < len(decoder_layers)
        }
        if needed_depths:
            cache_dir = Path(tempfile.mkdtemp(
                prefix="prismaquant_l3_baseline_context_",
                dir=tmp_parent,
            ))
            context_hooks = PerturbedActivationCache(
                model,
                assignment_c,
                cache_dir,
                input_rows=0,
                cal_hash=cal_hash,
                profile=profile,
            )
            frozen_context = (
                context_hooks.frozen_weight_cache()
                if use_frozen_perturbed_cache
                else nullcontext(context_hooks)
            )
            with frozen_context:
                context_hooks.install()
                try:
                    all_layer_calls = _capture_all_layer_calls(
                        model,
                        decoder_layers,
                        needed_depths,
                        calibration_data,
                        device,
                    )
                    tail_call_cache = {
                        depth: all_layer_calls.get(depth, [])
                        for depth in needed_depths
                    }
                finally:
                    context_hooks.remove()
                    shutil.rmtree(cache_dir, ignore_errors=True)
    for group_index, (group_key, group_entries) in enumerate(depth_groups, start=1):
        group_depth = layer_depth(group_entries[0].name) if group_entries else None
        use_tail_group = (
            bool(tail_only)
            and group_depth is not None
            and decoder_layers is not None
            and 0 <= group_depth < len(decoder_layers)
        )
        group_start = time.monotonic()
        group_lane_count = sum(
            len(_lane_specs_for_entries([entry]))
            for entry in group_entries
        )
        if progress_callback is not None:
            progress_callback({
                "event": "depth_group_start",
                "group": group_key,
                "group_index": group_index,
                "group_count": len(depth_groups),
                "entry_count": len(group_entries),
                "lane_count": group_lane_count,
                "mode": "tail-only" if use_tail_group else "full-forward",
            })
        group_use_prequant_cache = (
            use_prequant_cache
            and _l3_prequant_group_cache_fits(
                model,
                group_entries,
                specs,
                device,
            )
        )
        group_quant_cache = (
            build_quant_weight_cache(model, group_entries, specs)
            if group_use_prequant_cache
            else None
        )
        lane_batches = list(_lane_microbatches_for_entries(
            group_entries,
            max_lanes_per_batch,
        ))
        for lane_batch_index, lanes in enumerate(lane_batches, start=1):
            if not lanes:
                continue
            target_names = {lane.name for lane in lanes}
            context_assignment = {
                name: fmt
                for name, fmt in assignment_c.items()
                if name not in target_names
            }
            downstream_names = [
                name for name in _downstream_names_for_group(ordered_names, target_names)
                if name in set(all_output_names)
            ]
            cached_calls_for_graph = (
                tail_call_cache.get(group_depth, [])
                if use_tail_group and cache_tail_layer_inputs
                else None
            )
            if not cached_calls_for_graph:
                cached_calls_for_graph = None
            l3_tail_graph_call_count = (
                len(cached_calls_for_graph)
                if cached_calls_for_graph is not None
                else _calibration_call_count(calibration_data)
            )
            use_l3_tail_cuda_graphs = _env_cuda_graphs_enabled_for_call_count(
                "PRISMAQUANT_L3_CUDA_GRAPHS",
                default="auto",
                call_count=l3_tail_graph_call_count,
                min_calls=8,
            )
            tail_graph_safe = (
                use_tail_group
                and tail_graph_cache.enabled
                and use_l3_tail_cuda_graphs
                and not _output_mse_names_reach_tail(downstream_names, group_depth)
            )
            execution_lanes = (
                _pad_lanes_for_cuda_graph(lanes)
                if tail_graph_safe
                else lanes
            )
            if progress_callback is not None:
                progress_callback({
                    "event": "depth_group_microbatch_start",
                    "group": group_key,
                    "group_index": group_index,
                    "group_count": len(depth_groups),
                    "microbatch_index": lane_batch_index,
                    "microbatch_count": len(lane_batches),
                    "lane_count": len(lanes),
                    "execution_lane_count": len(execution_lanes),
                })
            graph_state_key = (
                _assignment_graph_key(context_assignment),
                _lane_specs_graph_key(execution_lanes),
            )
            cache_dir = Path(tempfile.mkdtemp(
                prefix="prismaquant_l3_context_",
                dir=tmp_parent,
            ))
            context_hooks = PerturbedActivationCache(
                model,
                context_assignment,
                cache_dir,
                input_rows=0,
                cal_hash=cal_hash,
                profile=profile,
            )
            frozen_context = (
                context_hooks.frozen_weight_cache()
                if use_frozen_perturbed_cache
                else nullcontext(context_hooks)
            )
            frozen_context_entered = False
            target_hooks = None
            output_mse = None
            try:
                frozen_context.__enter__()
                frozen_context_entered = True
                context_hooks.install()
                first_batch = True
                kl_totals = [0.0 for _ in lanes]
                batch_count = 0
                cached_calls = cached_calls_for_graph
                call_iter = (
                    cached_calls
                    if cached_calls is not None
                    else iter_calibration_forwards(calibration_data, device)
                )
                try:
                    for call_item in call_iter:
                        if cached_calls is not None:
                            args, kwargs, base_batch = _move_cached_layer_call(
                                call_item,
                                device,
                            )
                        else:
                            args, kwargs = call_item
                            base_batch = _first_tensor_batch_size(args, kwargs)
                        if first_batch:
                            target_hooks = _DepthGroupTargetHooks(
                                model,
                                assignment_c,
                                specs_by_name,
                                execution_lanes,
                                base_batch=base_batch,
                                quant_weight_cache=group_quant_cache,
                            )
                            target_hooks.install()
                            output_mse = _LaneOutputMSE(
                                model,
                                downstream_names,
                                execution_lanes,
                                base_batch=base_batch,
                            )
                            output_mse.install()
                            first_batch = False

                        if use_tail_group:
                            layer = decoder_layers[group_depth]
                            if cached_calls is not None:
                                layer_args, layer_kwargs = args, kwargs
                            else:
                                layer_args, layer_kwargs = _capture_layer_call(
                                    model,
                                    layer,
                                    args,
                                    kwargs,
                                )
                            rep_args, rep_kwargs = _repeat_layer_call_for_lanes(
                                layer_args,
                                layer_kwargs,
                                len(execution_lanes),
                                base_batch,
                            )
                            layer_output = layer(*rep_args, **rep_kwargs)
                            hidden = _first_tensor_output(layer_output)
                            if hidden is None:
                                raise RuntimeError(
                                    "tail-only L3 decoder layer returned no tensor"
                                )
                            logits = tail_forward_from_layer(
                                model,
                                group_depth,
                                rep_args,
                                rep_kwargs,
                                hidden,
                                cuda_graph_cache=(
                                    tail_graph_cache if tail_graph_safe else None
                                ),
                                lane_count=len(execution_lanes),
                                graph_state_key=graph_state_key,
                            )
                        else:
                            rep_args, rep_kwargs = _repeat_inputs_for_lanes(
                                args,
                                kwargs,
                                len(execution_lanes),
                            )
                            logits = _extract_logits(model(*rep_args, **rep_kwargs))

                        chunks = _split_lanes(
                            logits.detach(),
                            base_batch,
                            len(execution_lanes),
                        )
                        if chunks is None:
                            raise RuntimeError(
                                "L3 propagated-cost logits did not preserve lane "
                                f"batching: shape={tuple(logits.shape)} "
                                f"base_batch={base_batch} "
                                f"lanes={len(execution_lanes)}"
                            )
                        teacher_by_baseline: dict[int, torch.Tensor] = {}
                        for idx, lane in enumerate(lanes):
                            if lane.is_baseline or lane.baseline_index is None:
                                continue
                            baseline_index = int(lane.baseline_index)
                            teacher = teacher_by_baseline.get(baseline_index)
                            if teacher is None:
                                teacher = F.log_softmax(
                                    chunks[baseline_index].float(),
                                    dim=-1,
                                )
                                teacher_by_baseline[baseline_index] = teacher
                            kl_totals[idx] += float(
                                kl_divergence(chunks[idx], teacher).item()
                            )
                        batch_count += 1
                        if output_mse is not None:
                            output_mse.mark_batch()
                    if progress_callback is not None:
                        progress_callback({
                            "event": "depth_group_microbatch_end",
                            "group": group_key,
                            "group_index": group_index,
                            "group_count": len(depth_groups),
                            "microbatch_index": lane_batch_index,
                            "microbatch_count": len(lane_batches),
                            "lane_count": len(lanes),
                            "execution_lane_count": len(execution_lanes),
                            "batch_count": batch_count,
                        })
                finally:
                    if target_hooks is not None:
                        target_hooks.remove()
                    if output_mse is not None:
                        output_mse.remove()

                missing_targets = set(target_hooks.missing if target_hooks else [])
                for idx, lane in enumerate(lanes):
                    if lane.is_baseline:
                        continue
                    per_name = results.setdefault(lane.name, {})
                    if lane.name in missing_targets:
                        per_name[lane.fmt] = {
                            "error": "target module missing or unsupported for L3"
                        }
                        continue
                    per_name[lane.fmt] = {
                        "propagated_end_kl": kl_totals[idx] / max(batch_count, 1),
                        "downstream_output_mse": (
                            output_mse.value_for_lane(idx)
                            if output_mse is not None
                            else 0.0
                        ),
                        "paired_baseline": "target_bf16_under_l2_assignment",
                    }
            finally:
                context_hooks.remove()
                if frozen_context_entered:
                    frozen_context.__exit__(None, None, None)
                shutil.rmtree(cache_dir, ignore_errors=True)
        if progress_callback is not None:
            progress_callback({
                "event": "depth_group_end",
                "group": group_key,
                "group_index": group_index,
                "group_count": len(depth_groups),
                "entry_count": len(group_entries),
                "lane_count": group_lane_count,
                "elapsed_seconds": time.monotonic() - group_start,
            })

    tail_graph_cache.clear()
    return results
