"""Additivity sanity check: per-group MSE sum == whole-unit MSE for uniform rung.

Uses synthetic CPU tensors with real nvfp4_cb_formats encode (if available) or
pure synthetic MSE that is additive by construction.
"""
import torch
from harness.cbl_encode import synthetic_per_group_mse

def test_synthetic_additivity_exact():
    # Synthetic per-group MSE is defined as additive by construction: we treat
    # whole mse as sum_g per_group[g, rung]. Check tolerance 1e-6 relative.
    per = synthetic_per_group_mse(groups=4, seed=42)
    # Pick a rung index 5 (K33)
    ri = 5
    sums = float(per[:, ri].sum().item())
    # Whole would be sum, so this test just verifies our synthetic helper is internally consistent
    # More relevant: sweep solver's uniform_mse should equal per_group sum
    from harness.split_knapsack import uniform_mse, RUNGS
    assert abs(uniform_mse(per, RUNGS[ri]) - sums) < 1e-9

def test_synthetic_per_group_decreasing():
    per = synthetic_per_group_mse(groups=2, seed=0)
    # Higher rung should have lower mse per group
    for g in range(per.shape[0]):
        for ri in range(1, per.shape[1]):
            assert per[g, ri] < per[g, ri-1] + 1e-12

def test_content_key_deterministic():
    from harness.content_key import unit_content_key
    k1 = unit_content_key(layer=3, projection="gate_proj", expert=5,
                          source_digest="a"*64, col_weights_digest="b"*64,
                          book_shas={28:"c"*64, 33:"d"*64}, groups=2)
    k2 = unit_content_key(layer=3, projection="gate_proj", expert=5,
                          source_digest="a"*64, col_weights_digest="b"*64,
                          book_shas={28:"c"*64, 33:"d"*64}, groups=2)
    assert k1 == k2
    k3 = unit_content_key(layer=3, projection="gate_proj", expert=6,
                          source_digest="a"*64, col_weights_digest="b"*64,
                          book_shas={28:"c"*64, 33:"d"*64}, groups=2)
    assert k1 != k3
