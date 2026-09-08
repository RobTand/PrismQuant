"""The feasible per-unit quality optimum must survive budget-bin rounding."""
import pytest

from prismaquant.allocator_solver import Candidate, solve_with_promotion


def _inputs():
    stats = {f'u{i}': {'n_params': 8000} for i in range(8)}
    candidates = {n: [Candidate('NVFP4', 4.004, 4004, 1.0),
                      Candidate('BF16', 16.0, 16000, 0.0)] for n in stats}
    return stats, candidates


def test_feasible_quality_endpoint_is_not_excluded_by_bin_rounding():
    stats, candidates = _inputs()
    assignment, achieved = solve_with_promotion(
        stats, candidates, 16.0, {}, {'NVFP4': 0, 'BF16': 1}, 0.001,
        overshoot_tolerance=0.0)
    assert set(assignment.values()) == {'BF16'}
    assert achieved == 16.0


def test_quality_endpoint_respects_exact_budget():
    stats, candidates = _inputs()
    assignment, achieved = solve_with_promotion(
        stats, candidates, 15.0, {}, {'NVFP4': 0, 'BF16': 1}, 0.001,
        overshoot_tolerance=0.0)
    assert assignment is not None
    assert achieved <= 15.0
    assert 'NVFP4' in assignment.values()


def test_quality_endpoint_is_measured_per_unit_not_bf16_policy():
    stats, candidates = _inputs()
    for values in candidates.values():
        values[0] = Candidate('NVFP4', 4.004, 4004, 0.0)
        values[1] = Candidate('BF16', 16.0, 16000, 1.0)
    assignment, achieved = solve_with_promotion(
        stats, candidates, 16.0, {}, {'NVFP4': 0, 'BF16': 1}, 0.001,
        overshoot_tolerance=0.0)
    assert set(assignment.values()) == {'NVFP4'}
    assert achieved == pytest.approx(4.004)


@pytest.mark.parametrize('background_loss', [0.0, 1e16])
def test_promotion_must_preserve_each_units_minimum(background_loss):
    from prismaquant.model_profiles import DefaultProfile
    q, k, other = 'model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj', 'background'
    stats = {n: {'n_params': 8000} for n in (q, k, other)}
    candidates = {
        q: [Candidate('NVFP4', 4.0, 4000, 0.0), Candidate('BF16', 16.0, 16000, 1.0)],
        k: [Candidate('NVFP4', 4.0, 4000, 2.0), Candidate('BF16', 16.0, 16000, 0.0)],
        other: [Candidate('BF16', 16.0, 16000, background_loss)],
    }
    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, candidates, 16.0, {}, {'NVFP4': 0, 'BF16': 1}, 0.001,
        profile=DefaultProfile(), overshoot_tolerance=0.0, diagnostics=diag)
    assert assignment[q] == assignment[k]
    assert achieved <= 16.0
    assert diag['quality_endpoint_preserves_minimum_loss'] is False
    assert diag['quality_endpoint_selected'] is False
    assert diag['evals'] >= 1


def test_extracted_lfm_endpoint_has_zero_loss_at_exact_16_bpp():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'lfm_quality_endpoint.json').read_text())
    stats = {f'u{i}': {'n_params': row['n_params']} for i, row in enumerate(fixture['units'])}
    candidates = {f'u{i}': [Candidate(**c) for c in row['candidates']]
                  for i, row in enumerate(fixture['units'])}
    assignment, achieved = solve_with_promotion(
        stats, candidates, 16.0, {}, {}, 0.0001, overshoot_tolerance=0.0)
    assert set(assignment.values()) == {'BF16'}
    assert achieved == 16.0
    assert all(next(c for c in candidates[n] if c.fmt == fmt).predicted_dloss == 0
               for n, fmt in assignment.items())
