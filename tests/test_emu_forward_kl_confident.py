"""Confident-lane contract for ``prismaquant.emu_forward_kl`` (issue #160).

A cut that keeps zero positions must not emit ``kl_confident = 0.0`` — the
best possible score, computed over nothing. Every emitted confident number
must stamp the cut that produced it, and the 0.5 cut itself is pinned by the
majority property it implements, not by its literal value.
"""

import math

import pytest
import torch

from prismaquant.emu_forward_kl import _CONFIDENT_PROB, _KLAccumulator


def _lp(probs: list[list[float]]) -> torch.Tensor:
    return torch.tensor(probs, dtype=torch.float32).log()


def test_empty_confident_lane_refuses_instead_of_zero():
    """An all-unconfident batch must not yield kl_confident == 0.0.

    Fails against the pre-fix code, which divided by max(n_conf, 1) and
    returned a perfect 0.0 over an empty lane.
    """
    acc = _KLAccumulator()
    # Uniform teacher over 4 tokens: top-1 = 0.25, nothing clears the cut.
    teacher = _lp([[0.25, 0.25, 0.25, 0.25],
                   [0.25, 0.25, 0.25, 0.25]])
    student = _lp([[0.7, 0.1, 0.1, 0.1],
                   [0.1, 0.1, 0.1, 0.7]])
    acc.add(teacher, student)
    assert acc.n_conf == 0
    with pytest.raises(ValueError, match=r"confident lane is empty"):
        acc.result()


def test_empty_lane_error_names_cut_and_position_count():
    acc = _KLAccumulator()
    teacher = _lp([[0.25, 0.25, 0.25, 0.25]] * 3)
    student = _lp([[0.25, 0.25, 0.25, 0.25]] * 3)
    acc.add(teacher, student)
    with pytest.raises(ValueError) as exc_info:
        acc.result()
    message = str(exc_info.value)
    assert str(_CONFIDENT_PROB) in message
    assert "3" in message  # the position count the cut kept nothing from


def test_result_stamps_confident_cut():
    """The emitted result carries the cut that produced kl_confident."""
    acc = _KLAccumulator()
    teacher = _lp([[0.8, 0.1, 0.1],
                   [0.34, 0.33, 0.33]])
    student = _lp([[0.7, 0.2, 0.1],
                   [0.4, 0.3, 0.3]])
    acc.add(teacher, student)
    result = acc.result()
    assert result["confident_prob_cut"] == _CONFIDENT_PROB
    assert result["confident_prob_cut"] == 0.5
    assert result["n_confident"] == 1


def test_confident_cut_is_majority_boundary():
    """Pin the derivation operationally: the cut admits exactly the positions
    whose argmax survives ANY redistribution of the remaining mass.

    Distributions are derived from ``_CONFIDENT_PROB`` itself (cut +/- eps),
    so this tests the rule, not the roster: just above the cut the argmax is
    redistribution-invariant; just below it a redistribution exists that
    flips the argmax, which the lane correctly excludes.
    """
    eps = 0.1
    top_admitted = _CONFIDENT_PROB + eps
    rest = 1.0 - top_admitted
    # Worst case for the leader: pile ALL residual mass onto the runner-up.
    worst_case = [top_admitted, rest, 0.0]
    assert worst_case[0] > worst_case[1]  # leader still wins outright
    assert sum(worst_case) == pytest.approx(1.0)

    top_rejected = _CONFIDENT_PROB - eps
    rest_rejected = 1.0 - top_rejected
    flipped = [top_rejected, rest_rejected, 0.0]
    assert flipped[1] > flipped[0]  # same redistribution flips the argmax

    # And the accumulator agrees on membership for both sides of the cut.
    acc = _KLAccumulator()
    acc.add(_lp([[top_admitted, rest / 2, rest / 2]]),
            _lp([[top_admitted, rest / 2, rest / 2]]))
    acc.add(_lp([[top_rejected, rest_rejected / 2, rest_rejected / 2]]),
            _lp([[top_rejected, rest_rejected / 2, rest_rejected / 2]]))
    assert acc.n_conf == 1


def test_nonempty_lane_values_unchanged():
    """Non-empty lanes keep the exact pre-fix arithmetic: means over the
    confident positions, divided by n_conf (not max(n_conf, 1) — identical
    whenever n_conf > 0), plus the new stamp key.
    """
    teacher_p = [[0.8, 0.1, 0.1],
                 [0.6, 0.3, 0.1],
                 [0.3, 0.3, 0.4]]
    student_p = [[0.7, 0.2, 0.1],
                 [0.5, 0.4, 0.1],
                 [0.3, 0.3, 0.4]]
    acc = _KLAccumulator()
    acc.add(_lp(teacher_p), _lp(student_p))
    result = acc.result()

    assert acc.n_conf == 2
    expected_kl, expected_agree = 0.0, 0
    for tp, sp in zip(teacher_p, student_p):
        if max(tp) > _CONFIDENT_PROB:
            expected_kl += sum(p * (math.log(p) - math.log(q))
                               for p, q in zip(tp, sp))
            if tp.index(max(tp)) == sp.index(max(sp)):
                expected_agree += 1
    assert result["kl_confident"] == pytest.approx(expected_kl / 2, rel=1e-5)
    assert result["top1_agreement"] == pytest.approx(expected_agree / 2)
    assert result["n_confident"] == 2
    assert result["n_positions"] == 3
