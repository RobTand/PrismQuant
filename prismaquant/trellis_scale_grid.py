"""Research-only exact E4M3 scale-grid selection for E2M1 reconstruction.

This is numerical scaffolding, not a production renderer.  It searches the
same 34 multiplier probes used by the 2026-08-30 range study, but snaps every
candidate through ``float8_e4m3fn`` relative to the fixed whole-tensor global
scale.  Candidate zero is the multiplier-1 identity, so strict-improvement
updates give it deterministic priority on ties and guarantee that selection
cannot regress the static plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


E2M1_LEVELS = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)
E4M3_MAX = 448.0
GROUP_SIZE = 16


def scale_grid_multipliers() -> tuple[float, ...]:
    """Identity first, then the 33 preregistered log-spaced probes."""

    probes = torch.logspace(
        math.log10(0.55), math.log10(1.30), 33, dtype=torch.float64
    ).tolist()
    # Python floats from logspace do not include exactly 1.0.  Still filter by
    # bitwise equality so this contract remains correct if endpoints change.
    return (1.0, *(sorted(float(value) for value in probes if float(value) != 1.0)))


SCALE_GRID_MULTIPLIERS = scale_grid_multipliers()


@dataclass(frozen=True)
class ScaleGridSelection:
    effective_scales: torch.Tensor
    multiplier_indices: torch.Tensor
    multipliers: tuple[float, ...]
    group_sse: torch.Tensor
    total_sse: float
    baseline_sse: float


def exact_e4m3_scale_candidates(
    s_real: torch.Tensor,
    global_scale: torch.Tensor | float,
    *,
    multipliers: Sequence[float] = SCALE_GRID_MULTIPLIERS,
) -> torch.Tensor:
    """Return ``[candidate, ...s_real.shape]`` legal positive scales."""

    real = torch.as_tensor(s_real, dtype=torch.float32)
    global_value = torch.as_tensor(global_scale, device=real.device, dtype=torch.float32)
    if real.numel() == 0 or not bool(torch.isfinite(real).all()) or bool((real <= 0).any()):
        raise ValueError("s_real must be finite, positive, and nonempty")
    if (
        global_value.numel() != 1
        or not bool(torch.isfinite(global_value))
        or float(global_value) <= 0
    ):
        raise ValueError("global_scale must be one finite positive scalar")
    values = tuple(float(value) for value in multipliers)
    if not values or values[0] != 1.0:
        raise ValueError("candidate zero must be multiplier 1.0")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("all scale multipliers must be finite and positive")
    multiplier_tensor = torch.tensor(values, device=real.device, dtype=torch.float32)
    normalized = (
        real.unsqueeze(0) * multiplier_tensor.reshape((-1,) + (1,) * real.ndim)
        / global_value
    ).clamp(0.0, E4M3_MAX)
    snapped = normalized.to(torch.float8_e4m3fn).to(torch.float32)
    if bool((snapped <= 0).any()):
        raise ValueError("a scale candidate underflowed to zero on the legal E4M3 grid")
    candidates = snapped * global_value
    if not bool(torch.isfinite(candidates).all()) or bool((candidates <= 0).any()):
        raise ValueError("snapped scale candidates must be finite and positive")
    # Representation proof: dividing out the immutable global scalar must be
    # an exact E4M3 round-trip for every candidate.
    normalized_again = candidates / global_value
    if not torch.equal(
        normalized_again,
        normalized_again.to(torch.float8_e4m3fn).to(torch.float32),
    ):
        raise AssertionError("candidate escaped the fixed-global E4M3 grid")
    return candidates.contiguous()


def _nearest_levels(value: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    index = torch.bucketize(value.contiguous(), levels)
    low = levels[(index - 1).clamp(0, levels.numel() - 1)]
    high = levels[index.clamp(max=levels.numel() - 1)]
    # Equal-distance ties select low, matching the study's deterministic RTN.
    return torch.where((high - value).abs() < (value - low).abs(), high, low)


def select_e2m1_scale_grid(
    weight: torch.Tensor,
    metric_weight: torch.Tensor,
    s_real: torch.Tensor,
    global_scale: torch.Tensor | float,
    *,
    group_size: int = GROUP_SIZE,
    levels: Sequence[float] = E2M1_LEVELS,
    multipliers: Sequence[float] = SCALE_GRID_MULTIPLIERS,
) -> ScaleGridSelection:
    """Choose the lowest weighted-SSE legal scale independently per group."""

    value = torch.as_tensor(weight, dtype=torch.float32)
    if value.ndim != 2 or value.shape[1] % group_size:
        raise ValueError("weight must be rank 2 with columns divisible by group_size")
    rows, columns = value.shape
    groups = columns // group_size
    real = torch.as_tensor(s_real, device=value.device, dtype=torch.float32)
    if tuple(real.shape) != (rows, groups):
        raise ValueError(f"s_real shape must be {(rows, groups)}, got {tuple(real.shape)}")
    importance = torch.as_tensor(metric_weight, device=value.device, dtype=torch.float32)
    try:
        importance = torch.broadcast_to(importance, value.shape)
    except RuntimeError as exc:
        raise ValueError("metric_weight must broadcast to weight") from exc
    if not bool(torch.isfinite(value).all()):
        raise ValueError("weight must be finite")
    if not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()):
        raise ValueError("metric_weight must be finite and nonnegative")
    level_tensor = torch.tensor(tuple(float(x) for x in levels), device=value.device)
    if (
        level_tensor.ndim != 1
        or level_tensor.numel() < 2
        or not bool(torch.isfinite(level_tensor).all())
        or not bool(torch.all(level_tensor[1:] > level_tensor[:-1]))
    ):
        raise ValueError("levels must be finite, strictly increasing, and nontrivial")

    candidates = exact_e4m3_scale_candidates(
        real, global_scale, multipliers=multipliers
    )
    grouped = value.reshape(rows, groups, group_size)
    grouped_importance = importance.reshape(rows, groups, group_size)
    best_cost = torch.full((rows, groups), float("inf"), dtype=torch.float64,
                           device=value.device)
    best_index = torch.zeros((rows, groups), dtype=torch.int64, device=value.device)
    baseline_cost: torch.Tensor | None = None
    for index, scale in enumerate(candidates):
        normalized = grouped / scale.unsqueeze(-1)
        reconstruction = _nearest_levels(normalized, level_tensor) * scale.unsqueeze(-1)
        cost = (
            (grouped - reconstruction).to(torch.float64).square()
            * grouped_importance.to(torch.float64)
        ).sum(-1)
        if index == 0:
            baseline_cost = cost.clone()
        improves = cost < best_cost
        best_cost = torch.where(improves, cost, best_cost)
        best_index = torch.where(improves, index, best_index)
    assert baseline_cost is not None
    if bool((best_cost > baseline_cost).any()):
        raise AssertionError("scale-grid selection regressed its identity baseline")
    chosen = torch.gather(candidates, 0, best_index.unsqueeze(0)).squeeze(0)
    return ScaleGridSelection(
        effective_scales=chosen.contiguous(),
        multiplier_indices=best_index.contiguous(),
        multipliers=tuple(float(value) for value in multipliers),
        group_sse=best_cost.contiguous(),
        total_sse=float(best_cost.sum()),
        baseline_sse=float(baseline_cost.sum()),
    )


__all__ = [
    "E2M1_LEVELS",
    "SCALE_GRID_MULTIPLIERS",
    "ScaleGridSelection",
    "exact_e4m3_scale_candidates",
    "scale_grid_multipliers",
    "select_e2m1_scale_grid",
]
