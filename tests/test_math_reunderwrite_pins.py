from __future__ import annotations

import random

import pytest

import torch

from prismaquant import cb_layout
from prismaquant.allocator_solver import (
    Candidate,
    _charged_bins,
    predicted_dloss,
    selected_rung_dual_intervals,
    solve_allocation,
    solve_with_promotion,
)
from prismaquant.expert_empirical_cost import _cb_ladder_rate_factor
from prismaquant.kl_fisher import fisher_probe_scalar, fisher_quadratic_form
from prismaquant.nvfp4_cb_formats import TWO_TIER_SUB_TABLE, TWO_TIER_SUPER_BIAS


def test_charged_bins_conservative_table():
    assert _charged_bins(0.00049, 0.001) == 1
    assert _charged_bins(0.0, 0.001) == 0
    assert _charged_bins(-0.0004, 0.001) == 0
    assert _charged_bins(-0.002, 0.001) == -2
    assert round(0.5) == 0
    assert _charged_bins(0.0016, 0.001) == 2


def test_charged_bins_sub_half_bin_positive_delta_charges_one_bin():
    # A strictly positive delta is conservatively charged >= one bin even
    # though int(round(0.5)) == 0 under round-half-even: the clamp at
    # allocator_solver._charged_bins overrides the rounding result. This is
    # deliberate (the audit-fix that closed the free-upgrade hole); pinned by
    # value so the clamp/rounding interaction cannot silently change.
    assert _charged_bins(0.0005, 0.001) == 1


def test_charged_bins_seeded_positive_deltas_rounding_to_zero_charge_one_bin():
    precision = 0.001
    rng = random.Random(20260821)
    checked = 0
    for _ in range(200):
        delta = rng.uniform(1e-9, 9.9e-6)
        assert 0.0 < delta < 0.01 * precision
        if int(round(delta / precision)) == 0:
            assert _charged_bins(delta, precision) == 1
            checked += 1
    assert checked > 0


def test_predicted_dloss_gain_is_multiplicative_semantics():
    assert predicted_dloss(2.0, 3.0, gain=3.7) == 0.5 * 2.0 * 3.0 * 3.7
    assert predicted_dloss(2.0, 3.0) == 3.0


def test_kl_fisher_probe_covariance_closed_form():
    probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    teacher = torch.log(probs)
    n_probes = 4_000_000
    for temperature in (1.0, 0.7):
        p_tilde = torch.softmax(teacher / temperature, dim=-1)
        fisher = torch.diag(p_tilde) - torch.outer(p_tilde, p_tilde)
        logits = teacher.view(1, 1, 6).expand(n_probes, 1, 6).clone()
        logits.requires_grad_(True)
        scalar = fisher_probe_scalar(
            logits, seed=20260821, temperature=temperature
        )
        scalar.backward()
        grads = logits.grad.detach().reshape(n_probes, 6).double()
        second_moment = (grads.T @ grads) * (temperature * temperature)
        rel_err = ((second_moment - fisher.double()) / fisher.double()).abs()
        assert rel_err.max().item() < 0.05


def test_fisher_quadratic_form_exact():
    """`fisher_quadratic_form` equals 0.5 * dz~^T (diag(p~) - p~p~^T) dz~.

    The reference is computed in **float64**, and the tolerance is derived from
    float32 precision rather than pinned to a magic constant.

    Both choices are load-bearing.  The implementation evaluates the variance
    form ``sum p (dz - mean)^2`` while the identity above is the explicit
    matrix form; they are algebraically identical but reassociate differently
    in float32.  An earlier version of this test compared the two *float32*
    expressions against an absolute ``1e-10``.  That bound sits BELOW the
    float32 noise floor of the quantity itself -- at the T=0.5 scale of 0.2509
    one float32 ulp is ~1.5e-8 -- so it asserted nothing about the math and
    everything about summation order.  It passed on aarch64, where the two
    paths happen to reassociate bit-identically (observed |lhs - rhs| = 0), and
    failed on x86-64 CI at 7.45e-09, which is sub-ulp and therefore not an
    error at all.

    Comparing against float64 instead makes the assertion the real claim -- the
    float32 implementation is correct to float32 precision -- and is strictly
    stronger than the old one, which could not catch both float32 paths being
    wrong together.  16 ulps covers the softmax plus a 6-element reduction
    chain; the observed error is ~5.5e-09 against a 4.8e-07 bound, and any
    genuine algebraic error would be larger by orders of magnitude.
    """
    probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    z = torch.log(probs).view(1, 6)
    dz = torch.tensor([0.31, -0.52, 0.17, 0.02, -0.44, 0.63]).view(1, 6)
    float32_eps = torch.finfo(torch.float32).eps
    for temperature in (1.0, 0.5):
        lhs = fisher_quadratic_form(z, dz, temperature=temperature)

        p_tilde = torch.softmax(z.double() / temperature, dim=-1).squeeze(0)
        dz_tilde = (dz.double() / temperature).squeeze(0)
        fisher = torch.diag(p_tilde) - torch.outer(p_tilde, p_tilde)
        exact = 0.5 * (dz_tilde @ fisher @ dz_tilde)

        tolerance = 16.0 * float32_eps * abs(exact.item())
        assert abs(lhs.item() - exact.item()) <= tolerance, (
            f"T={temperature}: |{lhs.item():.10g} - {exact.item():.10g}| "
            f"exceeds {tolerance:.4g} (16 float32 ulps at this scale)")


def test_bit_split_ceil_first_pinned():
    assert cb_layout.bit_split(13, 2) == (7, 6)
    assert cb_layout.bit_split(29, 4) == (8, 7, 7, 7)
    assert cb_layout.bit_split(33, 4) == (9, 8, 8, 8)
    assert cb_layout.bit_split(12, 2) == (6, 6)
    for k in range(12, 49):
        for n_sub in (1, 2, 4):
            parts = cb_layout.bit_split(k, n_sub)
            assert sum(parts) == k
            assert all(parts[i] >= parts[i + 1] for i in range(len(parts) - 1))


def test_two_tier_constants_pinned_by_value():
    assert TWO_TIER_SUPER_BIAS == 127
    assert TWO_TIER_SUB_TABLE == (
        1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
    )


def test_scale_plane_and_type_size_pinned():
    assert cb_layout.SCALE_PLANE_BYTES[("fp4", "v1")] == 16
    assert cb_layout.SCALE_PLANE_BYTES[("fp4", "two_tier")] == 9
    assert cb_layout.SCALE_PLANE_BYTES[("fp8", "v1")] == 0
    assert cb_layout.INDEX_BYTES_PER_K == 4
    assert cb_layout.CODEWORDS_PER_SUPERBLOCK == 32
    assert cb_layout.type_size(28, "fp8") == 112
    assert cb_layout.type_size(16, "fp4", "two_tier") == 73
    assert cb_layout.type_size(16, "fp4", "v1") == 80
    assert cb_layout.type_size(24, "fp4", "two_tier") == 105


def test_cb_ladder_rate_factor_pinned():
    assert _cb_ladder_rate_factor("FP8_CB_K33", 33) == pytest.approx(
        0.013671875, rel=1e-12
    )
    assert _cb_ladder_rate_factor("FP8_CB_K28", 28) == pytest.approx(
        0.03125, rel=1e-12
    )


def test_dual_interval_empty_on_equal_byte_cheaper():
    stats = {"unit": {"n_params": 8}}
    candidates = {
        "unit": [
            Candidate("DOMINATED", 4.0, 4, 2.0),
            Candidate("BETTER", 4.0, 4, 1.0),
        ]
    }
    interval = selected_rung_dual_intervals(
        stats, candidates, {"unit": "DOMINATED"}, bit_precision=1.0
    )["unit"]
    assert interval.is_empty is True


def test_solve_allocation_raw_overshoot_is_bounded_rounding_slack():
    """TRUE contract of raw ``solve_allocation`` (audit 2026-08-21).

    The bits-DP bounds the CHARGED bins (total <= round(excess/bp) + 1),
    while each unit's actual share-scaled bit delta is rounded to the nearest
    bin (minimum 1 bin when strictly positive). The returned assignment's
    average bits can therefore EXCEED ``target_bits`` — by at most
    ``bit_precision * (n_units + 3) / 2``: n/2 bins of per-unit
    round-to-nearest slack plus the DP's 1.5 bins of capacity slack. This is
    a projection, not a budget-feasibility certificate; feasibility
    (``achieved - target <= overshoot_tolerance``) is enforced upstream by
    ``solve_with_promotion`` and the byte-budget exact filter (see
    docs/design/constrained_pareto_allocation.md, serve_constraints.py).

    The construction drives every unit's scaled delta to just under a
    half-bin boundary above one bin (fractional part .4999 -> rounds DOWN,
    so each unit is charged 1 bin for ~1.5 bins of real bits) and fills the
    DP capacity exactly, realizing ~97% of the provable worst case. n=74
    reproduces the ~0.036 bpp overshoot the numeric audit measured on random
    instances.
    """
    bp = 0.001
    n = 74
    p = 10_000
    stats = {f"u{i}": {"n_params": p} for i in range(n)}
    candidates = {}
    for i in range(n):
        delta = 1.4999 * bp * n          # unscaled; f_i = 1/n -> 1.4999 bins
        candidates[f"u{i}"] = [
            Candidate("LO", 4.0, p // 2, 10.0),
            Candidate("HI", 4.0 + delta, int(p * (4.0 + delta) / 8), 0.0),
        ]
    min_bits = 4.0
    target = min_bits + (n - 1 + 0.49) * bp

    result = solve_allocation(stats, candidates, target, bit_precision=bp)
    assert result is not None
    assign, chosen = result
    # Every upgrade is charged 1 bin and attractive, so the DP takes all of
    # them and fills its charged-bin capacity exactly.
    assert all(fmt == "HI" for fmt in assign.values())
    achieved = sum(
        chosen[k].bits_per_param * stats[k]["n_params"] for k in chosen
    ) / (n * p)
    overshoot = achieved - target
    # The point of the pin: the raw result is NOT budget-feasible...
    assert overshoot > 1e-9
    # ...but the exceedance is bounded rounding slack, proven above.
    assert overshoot == pytest.approx(0.0375, abs=5e-4)
    assert overshoot <= bp * (n + 3) / 2 + 1e-12

    # Upstream repair: the promotion wrapper enforces its tolerance band on
    # the achieved bits it returns.
    format_rank = {"LO": 0, "HI": 1}
    promoted, promoted_achieved = solve_with_promotion(
        stats, candidates, target, {}, format_rank, bp,
        overshoot_tolerance=0.01,
    )
    assert promoted is not None
    assert promoted_achieved <= target + 0.01
