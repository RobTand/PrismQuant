from __future__ import annotations

import torch

from prismaquant.trellis_scale_grid import (
    E2M1_LEVELS,
    SCALE_GRID_MULTIPLIERS,
    exact_e4m3_scale_candidates,
    select_e2m1_scale_grid,
)


def _reference(weight, importance, candidates):
    levels = torch.tensor(E2M1_LEVELS)
    best = []
    indices = []
    for group in range(weight.shape[1] // 2):
        current = None
        current_index = None
        values = weight[:, group * 2:(group + 1) * 2]
        metric = importance[:, group * 2:(group + 1) * 2]
        for index, scale in enumerate(candidates[:, :, group]):
            normalized = values / scale[:, None]
            quantized = []
            for row in normalized:
                quantized.append(torch.tensor([
                    min(E2M1_LEVELS, key=lambda level: (abs(float(x) - level), level))
                    for x in row
                ]))
            reconstruction = torch.stack(quantized) * scale[:, None]
            cost = ((values - reconstruction).double().square() * metric).sum(1)
            if current is None:
                current, current_index = cost, torch.full_like(cost, index, dtype=torch.int64)
            else:
                improves = cost < current
                current = torch.where(improves, cost, current)
                current_index = torch.where(improves, index, current_index)
        best.append(current)
        indices.append(current_index)
    return torch.stack(best, 1), torch.stack(indices, 1)


def test_candidate_contract_has_identity_plus_33_legal_positive_values():
    assert len(SCALE_GRID_MULTIPLIERS) == 34
    assert SCALE_GRID_MULTIPLIERS[0] == 1.0
    real = torch.tensor([[0.25, 0.5], [1.0, 2.0]])
    global_scale = torch.tensor(0.25 / 64)
    candidates = exact_e4m3_scale_candidates(real, global_scale)
    assert candidates.shape == (34, 2, 2)
    assert bool(torch.isfinite(candidates).all()) and bool((candidates > 0).all())
    normalized = candidates / global_scale
    assert torch.equal(
        normalized, normalized.to(torch.float8_e4m3fn).to(torch.float32)
    )
    expected_identity = (
        (real / global_scale).clamp(0, 448)
        .to(torch.float8_e4m3fn).to(torch.float32) * global_scale
    )
    assert torch.equal(candidates[0], expected_identity)


def test_selector_matches_bruteforce_and_never_regresses_identity():
    weight = torch.tensor([
        [-1.7, -0.6, 0.4, 1.8],
        [-3.2, -0.2, 0.9, 2.7],
    ])
    importance = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    real = torch.tensor([[0.4, 0.7], [0.8, 0.6]])
    global_scale = torch.tensor(0.4 / 64)
    multipliers = (1.0, 0.75, 1.25)
    candidates = exact_e4m3_scale_candidates(
        real, global_scale, multipliers=multipliers
    )
    result = select_e2m1_scale_grid(
        weight, importance, real, global_scale,
        group_size=2, multipliers=multipliers,
    )
    reference_cost, reference_index = _reference(
        weight, importance.expand_as(weight), candidates
    )
    assert torch.equal(result.multiplier_indices, reference_index)
    assert torch.allclose(result.group_sse, reference_cost, rtol=0, atol=0)
    assert result.total_sse <= result.baseline_sse


def test_identity_wins_exact_ties_deterministically():
    weight = torch.zeros(1, 2)
    importance = torch.ones_like(weight)
    result = select_e2m1_scale_grid(
        weight, importance, torch.ones(1, 1), 1 / 64,
        group_size=2, multipliers=(1.0, 0.75, 1.25),
    )
    assert result.multiplier_indices.item() == 0
    assert result.total_sse == result.baseline_sse == 0.0
