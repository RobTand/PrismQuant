import torch
from harness.split_knapsack import RUNGS, solve_split, sweep_budgets, uniform_mse, total_bytes_for_ks
from harness.cbl_encode import synthetic_per_group_mse, group_boundaries

def test_solve_split_simple():
    # G=2, synthetic MSE where group 0 prefers high K, group1 prefers low K
    per = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5],
                        [5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0]], dtype=torch.float64)
    # per is (2,11) for K28..K38; RUNG indices 0..10
    # Budget K32 uniform (index for K32 is 4): K32 is 32 -> idx 4
    from harness.byte_accounting import bytes_for_budget
    budget = bytes_for_budget("gate_proj", 32)
    ks, mse = solve_split(per, "gate_proj", 2, budget, per_group_scale=False)
    assert ks is not None
    assert len(ks) == 2
    assert sum(ks) == 64  # 2*32
    # Check that solver picks lower MSE than uniform (uniform is [32,32] mse = 6+4.6=10.6)
    umse = uniform_mse(per, 32)
    assert mse <= umse + 1e-9

def test_sweep_budgets():
    per = synthetic_per_group_mse(groups=4, seed=123)
    sweeps = sweep_budgets(per, "gate_proj", 4)
    assert len(sweeps) == 8  # K29..K36
    for e in sweeps:
        assert "gain" in e
        if e["feasible"]:
            assert e["split_ks"] is not None
            assert sum(e["split_ks"]) == 4*e["budget_k"]

def test_group_boundaries_alignment():
    for inn, G in [(4096,2),(4096,4),(4096,8),(2048,2),(2048,4),(2048,8)]:
        bounds = group_boundaries(inn, G)
        assert len(bounds) == G
        for s,e in bounds:
            assert (e-s) % 256 == 0
            assert (e-s) % 8 == 0
    # Misaligned should raise
    import pytest
    with pytest.raises(ValueError):
        group_boundaries(4096, 3)

def test_no_feasible_when_out_of_range():
    per = synthetic_per_group_mse(groups=2, seed=0)
    # Budget K50 not in rungs, but we use exact bytes for K50 -> sum 100 impossible with max 38*2=76
    from harness.byte_accounting import bytes_for_budget
    # craft impossible budget using shared model: need sum 100 but max sum 76
    # We can pass a huge budget directly
    ks, mse = solve_split(per, "gate_proj", 2, 999999999, per_group_scale=False)
    assert ks is None
