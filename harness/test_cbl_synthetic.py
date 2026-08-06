"""Synthetic encode path tests (CPU only)."""
import torch
from harness.cbl_encode import group_boundaries, synthetic_per_group_mse

def test_group_boundaries_cover():
    inn = 4096
    for G in [2,4,8]:
        bounds = group_boundaries(inn, G)
        # Must tile without gap/overlap and cover [0, inn)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == inn
        for i in range(len(bounds)-1):
            assert bounds[i][1] == bounds[i+1][0]

def test_synthetic_shape():
    per = synthetic_per_group_mse(out=64, inn=512, groups=8, seed=1)
    assert per.shape == (8, 11)
    assert (per > 0).all()
