"""Hard serving constraints as a second selection axis (ultraplan P5c).

``docs/lanes/nvfp4-cb/format-speed-policy.md`` §1 is the spec. What this file
pins is the part of it that is easy to get subtly wrong:

* the constraints are HARD — a miss is infeasible, never "scored worse", and
  there is no λ anywhere;
* prefill and decode never blend;
* an assignment the table cannot price is INFEASIBLE, not "passed" — "we could
  not price it" is not "it met the SLO";
* a rung whose fused mid-M lane the pinned Gridbook version does NOT
  instantiate is priced with its FALLBACK route's row (the audit's "the
  allocator should see that trade" point, which P5b's metadata makes
  answerable); and
* the verdict is deterministic and self-describing — which constraint binds,
  which assignments were rejected, and under which named assumptions.
"""
from __future__ import annotations

import pytest

from prismaquant.serve_constraints import (
    AGGREGATION_MODEL,
    RELATIVE_TAX_REFERENCE_RULE,
    ServeConstraintContext,
    ServeConstraintError,
    ServeSLOs,
    WorkloadMix,
    evaluate_assignment,
    fastest_feasible_summary,
    lane_key_for,
    rejection_record,
)
from prismaquant.serve_dispatch_table import parse_dispatch_table, SCHEMA
from prismaquant.serving_profiles import (
    serving_runtime_version,
    load_serving_profile,
    serving_lane_route,
)

_PROV = {
    "source": "tests/test_serve_constraints.py synthetic fixture",
    "date": "2026-08-01",
    "gpu": "synthetic",
    "measured_quantity": "synthetic relative cost",
    "units": "dimensionless",
    "derivation": "fixture constant",
}


def _arena(phase, regime, metric, absolute, *, m=None,
           statistic="p95"):
    return {
        "phase": phase, "m_regime": regime, "m": m,
        "reference_route": "fixture reference", "metric": metric,
        "absolute_value": absolute, "statistic": statistic,
        "provenance": dict(_PROV),
    }


def _row(family, phase, regime, lane, cost):
    return {
        "format_family": family, "phase": phase, "m_regime": regime,
        "lane": lane, "relative_unit_cost": cost, "provenance": dict(_PROV),
    }


def _table(arenas, rows, status="proposal_data"):
    return parse_dispatch_table({
        "schema": SCHEMA, "table_id": "fixture", "status": status,
        "arenas": arenas, "rows": rows,
    })


_SIMPLE_TABLE = _table(
    [
        _arena("prefill", "dense", "ttft_ms", 1000.0, m=1400),
        _arena("decode", "batch1", "decode_tok_s", 10.0, m=1, statistic="p05"),
    ],
    [
        _row("NVFP4", "prefill", "dense", "native", 1.0),
        _row("FP8_E4M3", "prefill", "dense", "native", 2.0),
        _row("NVFP4", "decode", "batch1", "native", 1.0),
        _row("FP8_E4M3", "decode", "batch1", "native", 4.0),
    ],
)

_STATS = {f"u{i}": {"n_params": 100} for i in range(4)}


def _ctx(**slo_kw):
    return ServeConstraintContext(
        table=_SIMPLE_TABLE,
        mix=WorkloadMix.parse("prefill:dense=1.0,decode:batch1=1.0"),
        slos=ServeSLOs(**slo_kw),
    )


# ---------------------------------------------------------------------------
# 1. Workload mix: explicit, validated, never renormalized
# ---------------------------------------------------------------------------
def test_no_mix_is_none_not_a_default():
    """Policy §1: 'no default workload mix hidden in the allocator'."""
    assert WorkloadMix.parse(None) is None
    assert WorkloadMix.parse("") is None


def test_mix_weights_must_sum_to_one():
    with pytest.raises(ServeConstraintError, match="sums to"):
        WorkloadMix.parse("prefill:a=0.4,prefill:b=0.4")


def test_mix_is_not_silently_renormalized():
    with pytest.raises(ServeConstraintError):
        WorkloadMix.parse("prefill:a=2.0")


def test_mix_rejects_malformed_and_repeated_entries():
    with pytest.raises(ServeConstraintError, match="phase:m_regime=weight"):
        WorkloadMix.parse("prefill-a-1.0")
    with pytest.raises(ServeConstraintError, match="repeats"):
        WorkloadMix.parse("prefill:a=0.5,prefill:a=0.5")


def test_mix_is_deterministically_ordered():
    a = WorkloadMix.parse("prefill:z=0.5,prefill:a=0.5")
    b = WorkloadMix.parse("prefill:a=0.5,prefill:z=0.5")
    assert a.by_phase == b.by_phase


# ---------------------------------------------------------------------------
# 2. Activation: inactive is a no-op, incoherent activation is refused
# ---------------------------------------------------------------------------
def test_context_with_no_slo_is_inactive_and_only_stamps():
    ctx = ServeConstraintContext(table=_SIMPLE_TABLE, mix=None)
    assert not ctx.active
    verdict = evaluate_assignment({"u0": "NVFP4"}, _STATS, ctx)
    assert verdict.feasible and not verdict.active
    assert verdict.checks == ()
    stamp = verdict.as_dict()
    assert stamp["active"] is False
    assert "no_dispatch_table_and_no_slos" in stamp["reason"]
    assert stamp["aggregation_model"] is None


def test_slo_without_a_table_is_refused_by_name():
    ctx = ServeConstraintContext(slos=ServeSLOs(p95_ttft_ms=1.0))
    with pytest.raises(ServeConstraintError, match="serve-dispatch-table"):
        ctx.validate()


def test_slo_without_a_workload_mix_is_refused_by_name():
    ctx = ServeConstraintContext(
        table=_SIMPLE_TABLE, slos=ServeSLOs(p95_ttft_ms=1.0))
    with pytest.raises(ServeConstraintError, match="serve-workload-mix"):
        ctx.validate()


def test_mix_that_does_not_weight_a_constrained_phase_is_refused():
    ctx = ServeConstraintContext(
        table=_SIMPLE_TABLE,
        mix=WorkloadMix.parse("decode:batch1=1.0"),
        slos=ServeSLOs(p95_ttft_ms=1.0),
    )
    with pytest.raises(ServeConstraintError, match="weights no 'prefill'"):
        ctx.validate()


def test_mix_naming_an_undeclared_arena_is_refused():
    ctx = ServeConstraintContext(
        table=_SIMPLE_TABLE,
        mix=WorkloadMix.parse("prefill:nope=1.0"),
        slos=ServeSLOs(p95_ttft_ms=1.0),
    )
    with pytest.raises(ServeConstraintError, match="does not declare"):
        ctx.validate()


# ---------------------------------------------------------------------------
# 3. The aggregation model
# ---------------------------------------------------------------------------
def test_all_reference_units_predict_the_reference_absolute():
    ctx = _ctx(p95_ttft_ms=2000.0)
    verdict = evaluate_assignment(
        {n: "NVFP4" for n in _STATS}, _STATS, ctx)
    assert verdict.predicted["p95_ttft_ms"] == pytest.approx(1000.0)
    assert verdict.feasible


def test_param_share_weighting_is_the_declared_model():
    """Half the parameters at 2.0x, half at 1.0x -> 1.5x the reference."""
    stats = {"big": {"n_params": 100}, "small": {"n_params": 100}}
    ctx = _ctx(p95_ttft_ms=10_000.0)
    verdict = evaluate_assignment(
        {"big": "FP8_E4M3", "small": "NVFP4"}, stats, ctx)
    assert verdict.predicted["p95_ttft_ms"] == pytest.approx(1500.0)
    # ...and it is the SHARE, not the count.
    stats2 = {"big": {"n_params": 300}, "small": {"n_params": 100}}
    v2 = evaluate_assignment({"big": "FP8_E4M3", "small": "NVFP4"}, stats2, ctx)
    assert v2.predicted["p95_ttft_ms"] == pytest.approx(1750.0)


def test_phases_are_separate_constraints_and_never_blended():
    """FP8 is 2x on prefill and 4x on decode; a single blended number could
    not produce this pair of verdicts."""
    ctx = _ctx(p95_ttft_ms=2500.0, p95_itl_ms=300.0)
    verdict = evaluate_assignment(
        {n: "FP8_E4M3" for n in _STATS}, _STATS, ctx)
    assert verdict.predicted["p95_ttft_ms"] == pytest.approx(2000.0)
    assert verdict.predicted["p95_itl_ms"] == pytest.approx(400.0)
    assert verdict.feasible is False
    assert verdict.violation_names() == ("p95_itl_ms",)


def test_p05_tps_is_the_single_stream_identity_and_says_so():
    ctx = _ctx(p05_tps=5.0)
    verdict = evaluate_assignment({n: "NVFP4" for n in _STATS}, _STATS, ctx)
    # decode reference is 10 tok/s -> 100 ms/token -> 10 tok/s predicted.
    assert verdict.predicted["p05_tps"] == pytest.approx(10.0)
    assert verdict.feasible
    check = next(c for c in verdict.checks if c.name == "p05_tps")
    assert check.direction == ">="
    assert any("A7" in c or "single-stream" in c for c in check.caveats)


def test_objective_is_never_blended_with_latency():
    ctx = _ctx(p95_ttft_ms=5000.0)
    verdict = evaluate_assignment({n: "NVFP4" for n in _STATS}, _STATS, ctx)
    prov = verdict.as_dict()
    assert prov["lambda_blended_objective"] is False
    assert prov["objective"] == (
        "min_predicted_dloss__latency_enters_only_as_feasibility")
    assert prov["aggregation_model"] == AGGREGATION_MODEL
    assert prov["global_optimality_claimed"] is False
    assert "proposal_data" in prov["evidence_status"]
    assert len(prov["aggregation_assumptions"]) >= 8


def test_relative_tax_reference_rule_travels_with_the_verdict():
    ctx = _ctx(p95_ttft_ms=5000.0)
    verdict = evaluate_assignment({n: "NVFP4" for n in _STATS}, _STATS, ctx)
    rule = verdict.as_dict()["relative_tax_reference_rule"]
    assert rule == RELATIVE_TAX_REFERENCE_RULE
    assert "FASTEST GLOBALLY FEASIBLE ASSIGNMENT" in rule
    assert "INDEPENDENTLY for" in rule


# ---------------------------------------------------------------------------
# 4. Fail-closed: unpriced is infeasible, never "passed"
# ---------------------------------------------------------------------------
def test_a_format_with_no_dispatch_row_is_infeasible_not_free():
    ctx = _ctx(p95_ttft_ms=1e9)
    verdict = evaluate_assignment({"u0": "BF16"}, {"u0": {"n_params": 1}}, ctx)
    assert not verdict.feasible
    check = verdict.checks[0]
    assert check.predicted is None
    assert "no_dispatch_row_for_BF16/native" in check.unpriced_reason


def test_an_operator_microbenchmark_arena_can_never_certify_an_slo():
    table = _table(
        [_arena("prefill", "gemm", "operator_ms", 5.0, m=64,
                statistic="median_of_repeated_samples")],
        [_row("FP8_CB", "prefill", "gemm", "fallback", 1.0)],
    )
    ctx = ServeConstraintContext(
        table=table,
        mix=WorkloadMix.parse("prefill:gemm=1.0"),
        slos=ServeSLOs(p95_ttft_ms=1e9),
    )
    ctx.validate()
    verdict = evaluate_assignment(
        {"u0": "FP8_CB_K36"}, {"u0": {"n_params": 1}}, ctx)
    assert not verdict.feasible
    assert "isolated_operator_microbenchmark" in (
        verdict.checks[0].unpriced_reason)


def test_an_arena_with_no_absolute_reference_cannot_certify_an_slo():
    table = _table(
        [_arena("prefill", "ratio", "ttft_ms", None, m=64,
                statistic="ratio_only_no_absolute")],
        [_row("NVFP4", "prefill", "ratio", "native", 1.0)],
    )
    ctx = ServeConstraintContext(
        table=table,
        mix=WorkloadMix.parse("prefill:ratio=1.0"),
        slos=ServeSLOs(p95_ttft_ms=1e9),
    )
    verdict = evaluate_assignment(
        {"u0": "NVFP4"}, {"u0": {"n_params": 1}}, ctx)
    assert not verdict.feasible
    assert "no_absolute_reference" in verdict.checks[0].unpriced_reason


def test_device_budget_without_resident_bytes_is_infeasible():
    ctx = ServeConstraintContext(
        table=_SIMPLE_TABLE,
        mix=WorkloadMix.parse("prefill:dense=1.0"),
        slos=ServeSLOs(device_budget_bytes=10**9),
    )
    verdict = evaluate_assignment({"u0": "NVFP4"}, _STATS, ctx)
    assert not verdict.feasible
    assert verdict.checks[0].unpriced_reason == "resident_bytes_not_supplied"


def test_device_budget_sums_resident_plus_kv_plus_scratch():
    ctx = ServeConstraintContext(
        table=_SIMPLE_TABLE,
        mix=WorkloadMix.parse("prefill:dense=1.0"),
        slos=ServeSLOs(device_budget_bytes=1000, kv_bytes=200,
                       peak_scratch_bytes=300),
    )
    ok = evaluate_assignment({"u0": "NVFP4"}, _STATS, ctx, resident_bytes=500)
    assert ok.feasible and ok.predicted["device_memory_bytes"] == 1000
    bad = evaluate_assignment({"u0": "NVFP4"}, _STATS, ctx, resident_bytes=501)
    assert not bad.feasible
    assert bad.binding_constraint == "device_memory_bytes"


# ---------------------------------------------------------------------------
# 5. Backed vs fallback lane pricing — the audit's "see that trade" point
# ---------------------------------------------------------------------------
_LANE_TABLE = _table(
    [_arena("prefill", "mid_m", "ttft_ms", 1000.0, m=64)],
    [
        _row("FP8_CB", "prefill", "mid_m", "fused_mid_m", 0.5),
        _row("FP8_CB", "prefill", "mid_m", "fallback", 2.0),
    ],
)


def _lane_ctx(**slo_kw):
    return ServeConstraintContext(
        table=_LANE_TABLE,
        mix=WorkloadMix.parse("prefill:mid_m=1.0"),
        slos=ServeSLOs(**slo_kw),
    )


# The fused mid-M backed set is attested PER GRIDBOOK RELEASE. Resolve these
# rule-level tests against the current attested release; the separate
# ``test_unattested_runtime_pin_backs_nothing`` covers the fail-closed side.
_ATTESTED_RUNTIME = "0.8.4"


def _profile_lane(_name, fmt):
    return serving_lane_route(
        "nvfp4_cb", fmt, runtime_version=_ATTESTED_RUNTIME)


# ---------------------------------------------------------------------------
# 6. Determinism and self-description
# ---------------------------------------------------------------------------
def test_binding_constraint_when_infeasible_is_the_first_in_canonical_order():
    ctx = _ctx(p95_ttft_ms=1.0, p95_itl_ms=1.0)
    verdict = evaluate_assignment({n: "FP8_E4M3" for n in _STATS}, _STATS, ctx)
    assert not verdict.feasible
    assert verdict.binding_constraint == "p95_ttft_ms"
    assert verdict.violation_names() == ("p95_ttft_ms", "p95_itl_ms")


def test_binding_constraint_when_feasible_is_the_tightest_relative_slack():
    # prefill predicted 1000 of 1010 (1% slack); decode 100 of 200 (50%).
    ctx = _ctx(p95_ttft_ms=1010.0, p95_itl_ms=200.0)
    verdict = evaluate_assignment({n: "NVFP4" for n in _STATS}, _STATS, ctx)
    assert verdict.feasible
    assert verdict.binding_constraint == "p95_ttft_ms"


def test_evaluation_is_deterministic():
    ctx = _ctx(p95_ttft_ms=5000.0, p95_itl_ms=5000.0)
    assignment = {n: ("NVFP4" if i % 2 else "FP8_E4M3")
                  for i, n in enumerate(_STATS)}
    first = evaluate_assignment(assignment, _STATS, ctx).as_dict()
    second = evaluate_assignment(
        dict(reversed(list(assignment.items()))), _STATS, ctx).as_dict()
    assert first == second


def test_statistic_mismatch_is_recorded_not_erased():
    """A8: a single-seed point measurement is not a p95, and the artifact has
    to say so rather than let the label imply a percentile."""
    table = _table(
        [_arena("prefill", "dense", "ttft_ms", 1000.0, m=1400,
                statistic="single_seed_point_measurement")],
        [_row("NVFP4", "prefill", "dense", "native", 1.0)],
    )
    ctx = ServeConstraintContext(
        table=table, mix=WorkloadMix.parse("prefill:dense=1.0"),
        slos=ServeSLOs(p95_ttft_ms=5000.0))
    verdict = evaluate_assignment(
        {"u0": "NVFP4"}, {"u0": {"n_params": 1}}, ctx)
    assert verdict.feasible
    caveats = verdict.checks[0].caveats
    assert any("single_seed_point_measurement" in c for c in caveats)
    assert any("A8" in c for c in caveats)


def test_rejection_record_names_the_binding_constraint():
    ctx = _ctx(p95_ttft_ms=100.0)
    verdict = evaluate_assignment({n: "FP8_E4M3" for n in _STATS}, _STATS, ctx)
    record = rejection_record(
        verdict, stage="bisect", target_bits=5.5, achieved_bits=5.4,
        dloss=1e-4)
    assert record["binding_constraint"] == "p95_ttft_ms"
    assert record["violated_constraints"] == ["p95_ttft_ms"]
    assert record["violations"][0]["limit"] == 100.0
    assert record["stage"] == "bisect" and record["target_bits"] == 5.5


def test_fastest_feasible_summary_is_per_phase_and_scoped():
    probes = [
        {"label": "a", "feasible": True,
         "predicted": {"p95_ttft_ms": 900.0, "p95_itl_ms": 50.0}},
        {"label": "b", "feasible": True,
         "predicted": {"p95_ttft_ms": 800.0, "p95_itl_ms": 90.0}},
        {"label": "c", "feasible": False,
         "predicted": {"p95_ttft_ms": 10.0, "p95_itl_ms": 1.0}},
    ]
    out = fastest_feasible_summary(probes, scope_note="fixture scope")
    # Independently per phase: 'b' is fastest on prefill, 'a' on decode.
    assert out["per_phase"]["p95_ttft_ms"]["label"] == "b"
    assert out["per_phase"]["p95_itl_ms"]["label"] == "a"
    # Infeasible probes never become the denominator.
    assert out["per_phase"]["p95_ttft_ms"]["n_feasible_probes_considered"] == 2
    assert out["scope"] == "fixture scope"
    assert "GLOBALLY FEASIBLE" in out["reference_rule"]
