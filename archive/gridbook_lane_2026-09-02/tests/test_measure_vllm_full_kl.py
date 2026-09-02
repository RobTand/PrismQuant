"""Position-KL math in tools/measure_vllm_full_kl.py (audit §3.6).

Pad entries (-1, -inf) at positions with fewer than top-K logprobs used to
produce 0 * (-inf) = NaN and poison the entire run's mean; they must be
masked out of both the KL sum and the accounted probability mass.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

# These tests exercise repo-root `tools/` scripts, which are not part of the
# installed package, so they can only run from a checkout. The release
# pipeline runs the suite against a non-editable install from outside the
# checkout (docs/RELEASING.md); skip there instead of failing collection.
if not (Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "measure_vllm_full_kl.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "measure_vllm_full_kl", _TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _log_softmax(values):
    return torch.log_softmax(torch.tensor(values, dtype=torch.float64), dim=-1)


def test_position_kl_matches_hand_computed_unpadded():
    t_lps = _log_softmax([2.0, 1.0, 0.0, -1.0])
    s_lps = _log_softmax([1.5, 1.2, 0.1, -0.5])
    t_ids = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
    s_ids = torch.tensor([3, 5, 7, 9], dtype=torch.int32)

    kl, top1 = tool._position_kl(t_ids, t_lps.float(), s_ids, s_lps.float())

    p = t_lps.float().double().exp()
    tlp = t_lps.float().double()
    q = s_lps.float().double()
    expected = float((p * (tlp - q)).sum())
    pt = max(1.0 - float(p.sum()), 1e-12)
    qt = max(1.0 - float(q.exp().sum()), 1e-12)
    expected += pt * (math.log(pt) - math.log(qt))

    assert kl == pytest.approx(expected, rel=1e-9)
    assert top1 == pytest.approx(float(p.max()), rel=1e-9)


def test_position_kl_masks_pad_entries_instead_of_nan():
    # Teacher position with only 2 real entries, padded to K=4 with
    # (-1, -inf): the pads contributed 0 * (-inf) = NaN before the fix,
    # and their floor-substituted q wrongly consumed student tail mass.
    real_t = _log_softmax([1.0, 0.0])
    t_ids = torch.tensor([3, 5, -1, -1], dtype=torch.int32)
    t_lps = torch.tensor(
        [float(real_t[0]), float(real_t[1]), float("-inf"), float("-inf")],
        dtype=torch.float32,
    )
    s_lps = _log_softmax([0.5, 0.5, -1.0, -2.0]).float()
    s_ids = torch.tensor([3, 5, 7, 9], dtype=torch.int32)

    kl, top1 = tool._position_kl(t_ids, t_lps, s_ids, s_lps)

    assert math.isfinite(kl)
    # Hand-computed over the two valid entries + tail bucket.
    tlp = torch.tensor(
        [float(t_lps[0]), float(t_lps[1])], dtype=torch.float64,
    )
    q = torch.tensor(
        [float(s_lps[0]), float(s_lps[1])], dtype=torch.float64,
    )
    p = tlp.exp()
    expected = float((p * (tlp - q)).sum())
    pt = max(1.0 - float(p.sum()), 1e-12)
    qt = max(1.0 - float(q.exp().sum()), 1e-12)
    expected += pt * (math.log(pt) - math.log(qt))

    assert kl == pytest.approx(expected, rel=1e-9)
    assert top1 == pytest.approx(float(p.max()), rel=1e-9)


def test_position_kl_pad_entries_do_not_consume_student_tail_mass():
    # With pads mapped to the student floor (the old behavior), Sigma q over
    # the teacher support would include exp(floor) per pad, shrinking the
    # student tail bucket qt. Verify the padded result equals the unpadded
    # 2-entry result exactly — pads must be complete no-ops.
    real_t = _log_softmax([1.0, 0.0])
    s_lps = _log_softmax([0.5, 0.5, -1.0, -2.0]).float()
    s_ids = torch.tensor([3, 5, 7, 9], dtype=torch.int32)

    padded_ids = torch.tensor([3, 5, -1, -1], dtype=torch.int32)
    padded_lps = torch.tensor(
        [float(real_t[0]), float(real_t[1]), float("-inf"), float("-inf")],
        dtype=torch.float32,
    )
    bare_ids = torch.tensor([3, 5], dtype=torch.int32)
    bare_lps = padded_lps[:2]

    kl_padded, top1_padded = tool._position_kl(padded_ids, padded_lps, s_ids, s_lps)
    kl_bare, top1_bare = tool._position_kl(bare_ids, bare_lps, s_ids, s_lps)

    assert kl_padded == kl_bare
    assert top1_padded == top1_bare


def test_position_kl_rejects_duplicate_student_token_ids():
    t_ids = torch.tensor([3, 5], dtype=torch.int32)
    t_lps = _log_softmax([1.0, 0.0]).float()
    s_ids = torch.tensor([3, 3], dtype=torch.int32)
    s_lps = _log_softmax([1.0, 0.0]).float()

    with pytest.raises(RuntimeError, match="duplicate token id 3"):
        tool._position_kl(t_ids, t_lps, s_ids, s_lps)


@pytest.mark.parametrize("bad_logprob", [float("nan"), float("inf"), float("-inf")])
def test_position_kl_rejects_nonfinite_student_logprobs(bad_logprob):
    t_ids = torch.tensor([3, 5], dtype=torch.int32)
    t_lps = _log_softmax([1.0, 0.0]).float()
    s_ids = torch.tensor([3, 5], dtype=torch.int32)
    s_lps = torch.tensor([-0.25, bad_logprob], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="student.*non-finite logprob"):
        tool._position_kl(t_ids, t_lps, s_ids, s_lps)


def test_position_kl_rejects_positive_student_logprob():
    t_ids = torch.tensor([3, 5], dtype=torch.int32)
    t_lps = _log_softmax([1.0, 0.0]).float()
    s_ids = torch.tensor([3, 5], dtype=torch.int32)
    s_lps = torch.tensor([-0.25, 1e-7], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="student.*positive logprob"):
        tool._position_kl(t_ids, t_lps, s_ids, s_lps)


def test_position_kl_rejects_nonfinite_computed_output():
    # Two individually finite contributions overflow when summed. The row is
    # malformed as a distribution, but the result boundary must still reject
    # the non-finite value instead of allowing JSON's Infinity extension.
    t_ids = torch.tensor([3, 5], dtype=torch.int32)
    t_lps = torch.tensor([-0.1, -0.1], dtype=torch.float64)
    s_ids = torch.tensor([3, 5], dtype=torch.int32)
    s_lps = torch.tensor(
        [-sys.float_info.max, -sys.float_info.max], dtype=torch.float64,
    )

    with pytest.raises(RuntimeError, match="non-finite KL output"):
        tool._position_kl(t_ids, t_lps, s_ids, s_lps)


def test_strict_json_text_rejects_nonfinite_output():
    with pytest.raises(RuntimeError, match="non-finite full-KL evidence"):
        tool._strict_json_text({"kl_mean": float("nan")})
