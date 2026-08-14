"""Unit tests for prismaquant.mtp_rung_selection (CPU, no torch).

Validates the canon selector (docs/design/mtp_rung_selection.md) against hand-derived
synthetic constants: the degenerate branch picks the highest-fidelity rung under
the memory gate; the memory gate excludes rungs; the interior argmax matches an
independent brute force; the acceptance fit degrades correctly with 0/1/2+
points; larger k pushes the argmax to a higher-fidelity rung; provenance is
complete and JSON-serialisable.
"""
from __future__ import annotations

import json
import math

import pytest

from prismaquant.mtp_rung_selection import (
    AcceptancePoint,
    DraftMemoryLedger,
    MeasuredDraftConfiguration,
    RungPoint,
    ServeConstants,
    fit_acceptance,
    select_measured_configuration,
    select_rung,
)

_BIG = 1 << 40  # a memory budget that admits every rung in these tests


def _ideal_menu(bits, bytes_by_bits=None):
    """Menu with the idealised E(b)=2^{-2b} law (sqrt(E)=2^{-b}), so a(b) is
    exactly linear in sqrt(E) and a 2-point fit is exact."""
    out = []
    for b in bits:
        rb = bytes_by_bits[b] if bytes_by_bits else 1_000_000
        out.append(RungPoint(name=f"b{b}", bits=float(b),
                             resident_bytes=int(rb), E=2.0 ** (-2.0 * b)))
    return out


# --------------------------------------------------------------------------- #
# fit_acceptance: 0 / 1 / 2+ points
# --------------------------------------------------------------------------- #
def test_fit_zero_points_is_no_data():
    a_inf, beta, mode = fit_acceptance([], {})
    assert (a_inf, beta, mode) == (None, None, "no_data")


def test_fit_one_point_is_single_point_no_slope():
    # a_inf = the one measured acceptance; no slope (beta None) per the doc.
    pts = [AcceptancePoint(0.83, rung_name="b4")]
    a_inf, beta, mode = fit_acceptance(pts, {"b4": 2.0 ** -8})
    assert mode == "single_point"
    assert beta is None
    assert a_inf == pytest.approx(0.83)


def test_fit_two_points_least_squares_recovers_line():
    # points (sqrt E, a) = (0.2, 0.6) and (0.1, 0.9) -> beta=3, a_inf=1.2.
    pts = [AcceptancePoint(0.6, rung_name="lo"),
           AcceptancePoint(0.9, rung_name="hi")]
    a_inf, beta, mode = fit_acceptance(pts, {"lo": 0.04, "hi": 0.01})
    assert mode == "least_squares"
    assert beta == pytest.approx(3.0)
    assert a_inf == pytest.approx(1.2)


def test_fit_same_E_multiple_points_has_no_slope():
    # >=2 points but identical E -> no fidelity spread -> single_point (mean).
    pts = [AcceptancePoint(0.80, rung_name="a"),
           AcceptancePoint(0.84, rung_name="b")]
    a_inf, beta, mode = fit_acceptance(pts, {"a": 0.01, "b": 0.01})
    assert mode == "single_point"
    assert beta is None
    assert a_inf == pytest.approx(0.82)


def test_fit_missing_E_is_hard_error():
    with pytest.raises(KeyError):
        fit_acceptance([AcceptancePoint(0.8, rung_name="ghost")], {"real": 0.01})


# --------------------------------------------------------------------------- #
# Interior argmax matches an independent brute force
# --------------------------------------------------------------------------- #
def test_interior_argmax_matches_brute_force():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)  # c large
    # calibrate at the two spanning rungs using their exact a(b) values
    a_inf_true, beta_true = 0.98, 1.5
    accepts = [
        AcceptancePoint(a_inf_true - beta_true * 2.0 ** -2, rung_name="b2"),
        AcceptancePoint(a_inf_true - beta_true * 2.0 ** -5, rung_name="b5"),
    ]
    res = select_rung(menu, const, accepts, _BIG, k=1)

    # independent brute force over the same fitted curve
    a_inf, beta, _ = fit_acceptance(accepts, {r.name: r.E for r in menu})
    assert (a_inf, beta) == pytest.approx((a_inf_true, beta_true))

    def T(r):
        a = a_inf - beta * math.sqrt(r.E)
        d = const.d0_ms + const.c_ms_per_bit * r.bits
        return (1.0 + a) / (const.t_ms + d)

    best = max(menu, key=T)
    assert res.regime == "interior"
    assert res.rung.name == best.name == "b3"


def test_continuous_bstar_present_and_interior():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.98 - 1.5 * 2.0 ** -2, rung_name="b2"),
               AcceptancePoint(0.98 - 1.5 * 2.0 ** -5, rung_name="b5")]
    res = select_rung(menu, const, accepts, _BIG, k=1)
    bstar = res.provenance["continuous_bstar"]
    assert bstar is not None and 2.0 < bstar < 4.0  # ~2.92, near the discrete b3


def test_lambertw_agrees_with_fixed_point_when_scipy_present():
    pytest.importorskip("scipy")
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.98 - 1.5 * 2.0 ** -2, rung_name="b2"),
               AcceptancePoint(0.98 - 1.5 * 2.0 ** -5, rung_name="b5")]
    prov = select_rung(menu, const, accepts, _BIG, k=1).provenance
    fp, lw = prov["continuous_bstar"], prov["continuous_bstar_lambertw"]
    assert lw is not None
    assert abs(fp - lw) < 0.5


# --------------------------------------------------------------------------- #
# k>1 pushes the argmax up
# --------------------------------------------------------------------------- #
def test_larger_k_pushes_argmax_to_higher_fidelity():
    # Two rungs; a(b3)=0.6, a(b4)=0.9 via E=(0.04, 0.01). With t=1,d0=0,c=0.5,
    # k=1 prefers the cheaper b3; k=2 amplifies acceptance and flips to b4.
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    accepts = [AcceptancePoint(0.6, rung_name="b3"),
               AcceptancePoint(0.9, rung_name="b4")]

    r1 = select_rung(menu, const, accepts, _BIG, k=1)
    r2 = select_rung(menu, const, accepts, _BIG, k=2)
    assert r1.regime == "interior" and r2.regime == "interior"
    assert r1.rung.name == "b3"
    assert r2.rung.name == "b4"
    assert r2.rung.bits > r1.rung.bits


# --------------------------------------------------------------------------- #
# Degenerate regime (cost-flat) picks highest fidelity under the gate
# --------------------------------------------------------------------------- #
def _hy3_menu():
    # NVFP4_CB K14..K20 (k/8+0.5 bpw) then FP8_CB K28..K44 (k/8 bpw); resident
    # bytes grow with bits so the gate can bind on the top rungs.
    bits = [2.25, 2.75, 3.5, 4.5, 5.5]
    return [RungPoint(name=f"r{b}", bits=b, resident_bytes=int(b * 1e8),
                      E=2.0 ** (-2.0 * b)) for b in bits]


def test_degenerate_regime_picks_highest_fidelity():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)  # eager d0
    accepts = [AcceptancePoint(0.78, rung_name="r2.75"),
               AcceptancePoint(0.92, rung_name="r5.5")]
    res = select_rung(menu, const, accepts, _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["degenerate_reason"] == "cost_flat"
    assert res.provenance["degenerate_test"]["ratio"] < 0.01
    # highest fidelity == lowest E == highest bits == the 5.5 rung
    assert res.rung.name == "r5.5"


def test_memory_gate_excludes_rungs_and_reshapes_choice():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.78, rung_name="r2.75"),
               AcceptancePoint(0.92, rung_name="r5.5")]
    # budget below the 4.5 and 5.5 rungs' bytes -> only <=3.5 pass
    budget = int(3.5 * 1e8)
    res = select_rung(menu, const, accepts, budget, k=1)
    excluded = {e["name"] for e in res.provenance["memory"]["excluded"]}
    assert excluded == {"r4.5", "r5.5"}
    assert res.provenance["memory"]["passing"] == ["r2.25", "r2.75", "r3.5"]
    # degenerate -> highest fidelity AMONG PASSING == r3.5
    assert res.regime == "degenerate"
    assert res.rung.name == "r3.5"


def test_no_rung_fits_budget_raises():
    menu = _hy3_menu()
    const = ServeConstants(t_ms=76.0, d0_ms=50.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.9, rung_name="r5.5")]
    with pytest.raises(ValueError, match="no rung fits"):
        select_rung(menu, const, accepts, 1, k=1)


# --------------------------------------------------------------------------- #
# Single-point / no-data fall back to the degenerate branch (not cost_flat)
# --------------------------------------------------------------------------- #
def test_single_point_falls_back_to_degenerate():
    # c is large (cost NOT flat), so the ONLY reason to be degenerate is the
    # missing acceptance slope from a single calibration point.
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    res = select_rung(menu, const, [AcceptancePoint(0.8, rung_name="b4")],
                      _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["degenerate_reason"] == "insufficient_acceptance_data"
    assert res.provenance["degenerate_test"]["cost_flat"] is False
    assert res.provenance["fit"]["fit_mode"] == "single_point"
    assert res.rung.name == "b4"  # highest fidelity (lowest E)


def test_no_acceptance_points_still_selects_highest_fidelity():
    menu = [RungPoint("b3", 3.0, 1_000_000, 0.04),
            RungPoint("b4", 4.0, 1_000_000, 0.01)]
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.5)
    res = select_rung(menu, const, [], _BIG, k=1)
    assert res.regime == "degenerate"
    assert res.provenance["fit"]["fit_mode"] == "no_data"
    assert res.rung.name == "b4"
    assert all(v is None for v in res.per_rung_T.values())  # no curve to score


# --------------------------------------------------------------------------- #
# Provenance completeness / serialisability
# --------------------------------------------------------------------------- #
def test_provenance_fields_present_and_json_serialisable():
    menu = _ideal_menu([2, 3, 4, 5])
    const = ServeConstants(t_ms=1.0, d0_ms=0.0, c_ms_per_bit=0.1)
    accepts = [AcceptancePoint(0.6, rung_name="b2"),
               AcceptancePoint(0.9, rung_name="b5")]
    res = select_rung(menu, const, accepts, _BIG, k=1, h_source="uniform")
    p = res.provenance
    for key in ("schema", "selected_rung", "regime", "degenerate_reason", "k",
                "h_source", "constants", "fit", "memory", "menu",
                "degenerate_test", "per_rung_T", "continuous_bstar",
                "continuous_method"):
        assert key in p, f"missing provenance key {key}"
    assert p["h_source"] == "uniform"
    assert p["fit"]["fit_mode"] == "least_squares"
    assert set(p["per_rung_T"]) == {"b2", "b3", "b4", "b5"}
    # documented sub-fields present
    for k in ("a_inf", "beta", "fit_mode", "n_points", "beta_negative", "points"):
        assert k in p["fit"]
    for k in ("cost_span_ms", "cycle_ms", "ratio", "cost_flat",
              "insufficient_slope"):
        assert k in p["degenerate_test"]
    assert "a_clamped" in p and "continuous_bstar_lambertw" in p
    assert p["fit"]["n_points"] == 2
    # must round-trip through JSON (doc §3.7)
    assert json.loads(json.dumps(p))["selected_rung"] == res.rung.name


def test_input_validation_fail_fast():
    with pytest.raises(ValueError):
        RungPoint("bad", -1.0, 10, 0.01)
    with pytest.raises(ValueError):
        ServeConstants(t_ms=0.0, d0_ms=0.0, c_ms_per_bit=0.1)
    with pytest.raises(ValueError):
        AcceptancePoint(1.5, rung_name="x")  # acceptance outside [0,1]
    with pytest.raises(ValueError):
        AcceptancePoint(0.5)  # neither rung_name nor bits


# --------------------------------------------------------------------------- #
# Block-parallel / directly measured configuration selector
# --------------------------------------------------------------------------- #
def _measured(
    name,
    *,
    resident,
    cycle,
    survival,
    load=0.0,
    k=5,
    E=None,
    measurement_id="dsv4-c1-fixed-v1",
    scratch=0,
):
    return MeasuredDraftConfiguration(
        name=name,
        k=k,
        resident_bytes=resident,
        cycle_ms=cycle,
        position_survival=survival,
        measurement_id=measurement_id,
        load_ms=load,
        peak_scratch_bytes=scratch,
        E=E,
        measurement_source="prometheus-counter-delta",
    )


def _ledger(*, safety=100, mode="production"):
    return DraftMemoryLedger(
        usable_pool_bytes=1_000,
        fixed_runtime_bytes=400,
        target_kv_bytes=100,
        draft_kv_bytes=20,
        profiling_peak_bytes=30,
        safety_margin_bytes=safety,
        admission_mode=mode,
    )


def test_block_parallel_selector_uses_one_measured_cycle_not_k_times_cost():
    # The larger draft accepts more positions, but its complete block cycle is
    # slow enough that the smaller draft wins. The selector consumes cycle_ms
    # exactly once; multiplying either value by k would be the DSpark bug.
    small = _measured(
        "K4", resident=100, cycle=40,
        survival=(0.70, 0.50, 0.35, 0.20, 0.10), E=2.0,
    )
    large = _measured(
        "K12", resident=200, cycle=75,
        survival=(0.95, 0.90, 0.82, 0.73, 0.65), E=0.5,
    )
    result = select_measured_configuration(
        [small, large], _ledger(safety=0), minimum_k=5
    )
    assert result.configuration.name == "K4"
    expected = 1000 * (1 + sum(small.position_survival)) / small.cycle_ms
    assert result.per_configuration["K4"][
        "steady_state_tokens_per_second"
    ] == pytest.approx(expected)


def test_memory_ledger_can_relax_testing_without_changing_production_gate():
    compact = _measured(
        "compact", resident=200, cycle=60,
        survival=(0.8, 0.6, 0.4, 0.2, 0.1),
    )
    accurate = _measured(
        "accurate", resident=380, cycle=45,
        survival=(0.95, 0.9, 0.8, 0.7, 0.6),
    )
    production = select_measured_configuration(
        [compact, accurate], _ledger(safety=100), minimum_k=5
    )
    assert production.configuration.name == "compact"
    assert production.provenance["memory"]["admission_mode"] == "production"
    assert production.per_configuration["accurate"]["passes_memory"] is False

    relaxed = select_measured_configuration(
        [compact, accurate],
        _ledger(safety=0, mode="test-only-relaxed"),
        minimum_k=5,
    )
    assert relaxed.configuration.name == "accurate"
    assert relaxed.provenance["memory"]["safety_margin_bytes"] == 0
    assert relaxed.provenance["memory"]["admission_mode"] == \
        "test-only-relaxed"


def test_candidate_specific_peak_scratch_is_in_memory_gate():
    steady_small_peak_large = _measured(
        "scratchy", resident=100, scratch=260, cycle=40,
        survival=(0.9, 0.8, 0.7, 0.6, 0.5),
    )
    stable = _measured(
        "stable", resident=220, scratch=0, cycle=50,
        survival=(0.8, 0.7, 0.6, 0.5, 0.4),
    )
    result = select_measured_configuration(
        [steady_small_peak_large, stable], _ledger(safety=100), minimum_k=5
    )
    assert result.configuration.name == "stable"
    assert result.per_configuration["scratchy"]["passes_memory"] is False


def test_startup_amortization_can_choose_the_faster_loading_candidate():
    slow_load = _measured(
        "slow-load", resident=100, cycle=20,
        survival=(0.8, 0.7, 0.6, 0.5, 0.4), load=10_000,
    )
    fast_load = _measured(
        "fast-load", resident=100, cycle=22,
        survival=(0.8, 0.7, 0.6, 0.5, 0.4), load=100,
    )
    steady = select_measured_configuration(
        [slow_load, fast_load], _ledger(safety=0), minimum_k=5
    )
    assert steady.configuration.name == "slow-load"
    short_service = select_measured_configuration(
        [slow_load, fast_load], _ledger(safety=0),
        expected_cycles=10, minimum_k=5,
    )
    assert short_service.configuration.name == "fast-load"
    assert short_service.provenance["objective"] == \
        "amortized_tokens_per_second"


def test_joint_k_menu_respects_artifact_floor_and_scores_position_counters():
    invalid_k4 = _measured(
        "k4", resident=100, cycle=20, k=4,
        survival=(0.9, 0.8, 0.7, 0.6),
    )
    valid_k5 = _measured(
        "k5", resident=100, cycle=40, k=5,
        survival=(0.7, 0.5, 0.3, 0.2, 0.1),
    )
    result = select_measured_configuration(
        [invalid_k4, valid_k5], _ledger(safety=0), minimum_k=5
    )
    assert result.configuration.name == "k5"
    assert result.per_configuration["k4"]["passes_k_floor"] is False
    assert result.provenance["excluded"] == [
        {"name": "k4", "reasons": ["below_k_floor"]}
    ]


def test_measured_selector_reports_pareto_frontier_and_json_provenance():
    tiny = _measured(
        "tiny", resident=100, cycle=50,
        survival=(0.7, 0.5, 0.3, 0.2, 0.1), load=100,
    )
    fast = _measured(
        "fast", resident=200, cycle=30,
        survival=(0.8, 0.7, 0.6, 0.5, 0.4), load=200,
    )
    dominated = _measured(
        "dominated", resident=250, cycle=60,
        survival=(0.7, 0.5, 0.3, 0.2, 0.1), load=300,
    )
    result = select_measured_configuration(
        [tiny, fast, dominated], _ledger(safety=0), minimum_k=5
    )
    assert result.pareto_frontier == ("fast", "tiny")
    assert "dominated" not in result.pareto_frontier
    assert json.loads(json.dumps(result.provenance))[
        "selected_configuration"
    ] == result.configuration.name


def test_measured_selector_rejects_mixed_workloads_and_bad_survival():
    good = _measured(
        "a", resident=100, cycle=50,
        survival=(0.8, 0.6, 0.4, 0.2, 0.1),
    )
    other_workload = _measured(
        "b", resident=100, cycle=50,
        survival=(0.8, 0.6, 0.4, 0.2, 0.1),
        measurement_id="different",
    )
    with pytest.raises(ValueError, match="different measurement_id"):
        select_measured_configuration(
            [good, other_workload], _ledger(safety=0), minimum_k=5
        )
    with pytest.raises(ValueError, match="non-increasing"):
        _measured(
            "bad", resident=100, cycle=50,
            survival=(0.7, 0.8, 0.4, 0.2, 0.1),
        )


def test_measured_selector_fails_when_no_candidate_fits():
    candidate = _measured(
        "too-large", resident=500, cycle=50,
        survival=(0.8, 0.6, 0.4, 0.2, 0.1),
    )
    with pytest.raises(ValueError, match="no candidate passes"):
        select_measured_configuration(
            [candidate], _ledger(safety=100), minimum_k=5
        )
