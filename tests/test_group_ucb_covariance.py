"""Grouped UCB cannot assume independent errors from shared AURA probes."""
import math

import pytest

from prismaquant.allocator_candidates import _super_item_ucb_hedge


def _stderr(samples):
    n = len(samples)
    mean = sum(samples) / n
    return 0.5 * math.sqrt(sum((x - mean) ** 2 for x in samples) / (n - 1) / n)


@pytest.mark.parametrize("second", [[0.0, 4.0], [4.0, 0.0]])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("include_samples", [False, True])
def test_group_bound_covers_correlated_errors_without_verified_alignment(second, scales, include_samples):
    samples = [[0.0, 4.0], second]
    terms = []
    for row, scale in zip(samples, scales):
        entry = {"predicted_dloss": 1.0, "predicted_dloss_stderr": _stderr(row)}
        if include_samples:
            # Equal-length arrays alone do not prove joint sample identity.
            entry["x2_per_probe"] = row
        terms.append(({}, entry, scale * 3.0, scale))
    removed, bound = _super_item_ucb_hedge(terms, 2.0)
    actual = _stderr([sum(scale * row[i] for row, scale in zip(samples, scales))
                      for i in range(2)])
    assert bound >= actual
    assert bound == sum(scale * _stderr(row) for row, scale in zip(samples, scales))
    assert removed == 2.0 * bound
    if second == samples[0]:
        assert bound == actual


def test_zero_z_keeps_member_sum_bit_exact():
    terms = [({}, {"predicted_dloss": value, "predicted_dloss_stderr": 1.0}, value, scale)
             for value, scale in [(0.1, 2.0), (0.2, 0.5)]]
    total = sum(term[2] for term in terms)
    removed, bound = _super_item_ucb_hedge(terms, 0.0)
    assert removed == 0.0
    assert (total - removed + 0.0 * bound).hex() == total.hex()
