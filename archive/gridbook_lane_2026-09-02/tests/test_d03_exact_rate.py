"""The D0.3 exact-rate experiment harness (ultraplan P5d).

gridbook ``ROADMAP.md`` D0.3 names two experiments; this file drives the real
harness (``prismaquant.d03_exact_rate.main``) on synthetic fixtures for both
shapes, and pins the things that make its output evidence rather than a
number:

* **Exact byte accounting.** Experiment (i)'s arms are compared at exact
  whole-artifact bytes, with shared CB codebook sidecars charged ONCE. The
  test computes both arms' bytes independently and asserts equality to the
  byte — the additive per-candidate cost would double-charge the sidecar, and
  a matched-bytes contest cannot survive that.
* **The byte-neutral discipline.** Policy §4: "every NVFP4 promotion must be
  funded by lower CB rungs elsewhere [...] never compare an isolated promoted
  layer against an unfunded baseline." Every sweep point must sit at or under
  the baseline's exact bytes, or be reported as NOT byte-neutral and carry no
  quality claim.
* **The two refusals.** No cross-family verdict when P5a's band check failed;
  no quality verdict when the two arms are not byte-matched.
* **The scope exclusion.** Packed-expert vanilla NVFP4 is out of the contest
  (gridbook D0.2), recorded explicitly in every report.
"""
from __future__ import annotations

import json
import pickle

import pytest

from prismaquant.serving_profiles import (
    gridbook_runtime_version,
    serving_lane_route,
)

from prismaquant import d03_exact_rate as d03
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
)
from prismaquant.serve_dispatch_table import SCHEMA as TABLE_SCHEMA
from prismaquant import format_registry as fr
from prismaquant.footprint import nvfp4_global_sidecar_bytes

_DENSE = [f"model.layers.{i}.self_attn.o_proj" for i in range(6)]
_EXPERTS = [f"model.layers.{i}.mlp.experts.gate_up_proj" for i in range(2)]
_OUT = _IN = 512
_NPARAMS = _OUT * _IN
_MENU = ["FP8_CB_K28", "FP8_CB_K32", "FP8_CB_K36", "NVFP4", "BF16"]
_CTX = CBSerializationContext(
    scale_coding="two_tier", codebook_source="lattice",
    scale_sweep=True, encode_tier="balanced")

_PROV = {
    "source": "tests/test_d03_exact_rate.py synthetic fixture",
    "date": "2026-08-01",
    "gpu": "synthetic",
    "measured_quantity": "synthetic relative serving cost",
    "units": "dimensionless",
    "derivation": "fixture constant",
}


def _fixture(tmp_path, *, cross_family_verdict=None, with_experts=False):
    names = list(_DENSE) + (list(_EXPERTS) if with_experts else [])
    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": _NPARAMS,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(names)
    }
    # Monotone in rung: more index bits -> lower Δloss, so the promotion and
    # funding orders are well defined and the sweep is not degenerate.
    per_format = {
        "FP8_CB_K28": 4.0e-4,
        "FP8_CB_K32": 3.0e-4,
        "FP8_CB_K36": 2.0e-4,
        "NVFP4": 1.0e-4,
        "BF16": 0.0,
    }
    costs_payload = {
        "costs": {
            n: {
                fmt: {"weight_mse": v, "output_mse": v,
                      "output_mse_measured": True, "predicted_dloss": v}
                for fmt, v in per_format.items()
            }
            for n in names
        },
        "meta": {"formats": list(per_format)},
    }
    if cross_family_verdict is not None:
        costs_payload["provenance"] = {
            "expert_empirical_cost": {
                "cb_ladder_cross_family_verdict": dict(cross_family_verdict),
            },
        }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(
        {"stats": stats, "meta": {"model": "/nonexistent"}}))
    cost_p.write_bytes(pickle.dumps(costs_payload))
    return probe_p, cost_p, stats


def _run(tmp_path, probe_p, cost_p, *, extra=(), out="report.json"):
    out_path = tmp_path / out
    argv = [
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--out", str(out_path),
        "--formats", ",".join(_MENU),
        "--target-profile", "research",
        "--promote-counts", "1,2,4",
        *extra,
    ]
    assert d03.main(argv) == 0
    return json.loads(out_path.read_text())


def _expected_bytes(assignment, stats):
    """Independent exact-byte expectation.

    CB rows go through ``cb_assignment_payload_breakdown`` (which charges the
    shared codebook sidecar once per physical identity); non-CB rows through
    the format's own serialized payload plus NVFP4's two fp32 global scalars.
    """
    total = 0
    cb = {}
    shapes = {}
    for name, fmt in assignment.items():
        shape = (int(stats[name]["out_features"]),
                 int(stats[name]["in_features"]))
        if fmt.startswith(("FP8_CB_", "NVFP4_CB_")):
            cb[name] = fmt
            shapes[name] = shape
            continue
        total += fr.get_format(fmt).memory_bytes_for_shape(shape)
        if fmt == "NVFP4":
            total += nvfp4_global_sidecar_bytes(name, shape)
    if cb:
        payload = cb_assignment_payload_breakdown(cb, shapes, context=_CTX)
        total += int(payload["tensor_payload_bytes"])
        total += int(payload["codebook_sidecar_bytes"])
    return total


# ---------------------------------------------------------------------------
# 1. Experiment (i): matched exact whole-artifact bytes
# ---------------------------------------------------------------------------
def test_matched_bytes_arms_are_exactly_the_independently_computed_bytes(
        tmp_path):
    probe_p, cost_p, stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    exp = report["experiments"]["matched_bytes"]
    assert exp["scope"]["n_eligible_units"] == len(_DENSE)

    arms = {a["label"]: a for a in exp["arms"]}
    cb_arm = arms["cb:FP8_CB_K36"]
    native_arm = arms["native:NVFP4"]
    assert cb_arm["exact_bytes"] == _expected_bytes(
        {n: "FP8_CB_K36" for n in _DENSE}, stats)
    assert native_arm["exact_bytes"] == _expected_bytes(
        {n: "NVFP4" for n in _DENSE}, stats)
    # The delta is the integer difference of those two exact numbers.
    assert exp["bytes_delta_bytes"] == (
        cb_arm["exact_bytes"] - native_arm["exact_bytes"])
    assert exp["bytes_delta_fraction"] == pytest.approx(
        abs(exp["bytes_delta_bytes"]) / max(
            cb_arm["exact_bytes"], native_arm["exact_bytes"]))
    # Shared CB sidecars are charged once, not per tensor.
    assert cb_arm["cb_shared_sidecar_bytes"] > 0
    assert native_arm["cb_shared_sidecar_bytes"] == 0


def test_matched_bytes_gate_uses_the_policy_target_and_refuses_a_mismatch(
        tmp_path):
    """Policy §5 names <=0.1% as the formal whole-artifact byte-match target.
    On this small fixture the shared sidecar makes the arms miss it, and the
    harness must SAY so rather than publish a same-rate verdict."""
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    exp = report["experiments"]["matched_bytes"]
    assert exp["matched_bytes_tolerance_fraction"] == 0.001
    assert "§5" in exp["matched_bytes_tolerance_source"]
    assert exp["matched_bytes_gate"] is (
        exp["bytes_delta_fraction"] <= 0.001)
    assert exp["matched_bytes_gate"] is False


def test_matched_bytes_gate_passes_when_the_arms_really_do_match(tmp_path):
    """The gate is a real test, not always-false: a self-contest (both arms
    on the same format) matches exactly, delta 0, gate true."""
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p, out="same.json",
        extra=["--skip-byte-neutral", "--native-format", "FP8_CB_K36"])
    exp = report["experiments"]["matched_bytes"]
    assert exp["bytes_delta_bytes"] == 0
    assert exp["bytes_delta_fraction"] == 0.0
    assert exp["matched_bytes_gate"] is True


def test_matched_bytes_arms_differ_only_on_contested_rows(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p,
        extra=["--skip-byte-neutral", "--baseline-format", "BF16"])
    exp = report["experiments"]["matched_bytes"]
    arms = {a["label"]: a["assignment"] for a in exp["arms"]}
    cb, native = arms["cb:FP8_CB_K36"], arms["native:NVFP4"]
    assert set(cb) == set(native)
    differing = {n for n in cb if cb[n] != native[n]}
    assert differing == set(_DENSE)


def test_no_contest_units_is_a_named_error(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    with pytest.raises(d03.D03Error, match="no unit admits BOTH"):
        d03.run(
            probe_path=str(probe_p), costs_path=str(cost_p),
            formats=["NVFP4", "BF16"], target_profile="research",
            cb_format="FP8_CB_K36", native_format="NVFP4",
            baseline_format=None, baseline_cb_format="FP8_CB_K32",
            cb_grid="fp8", promote_counts=[1], exclude_markers=[],
            cb_serialization_context=_CTX,
            serve_context=d03.ServeConstraintContext(),
            skip_byte_neutral=True,
        )


# ---------------------------------------------------------------------------
# 2. Experiment (ii): byte-neutral sweep
# ---------------------------------------------------------------------------
def test_byte_neutral_sweep_respects_the_baseline_byte_budget(tmp_path):
    probe_p, cost_p, stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p,
        extra=["--skip-matched-bytes", "--baseline-cb-format", "FP8_CB_K32"])
    sweep = report["experiments"]["byte_neutral_sweep"]
    budget = sweep["byte_budget_bytes"]
    assert budget == _expected_bytes({n: "FP8_CB_K32" for n in _DENSE}, stats)
    assert sweep["points"]
    for point in sweep["points"]:
        arm = point["arm"]
        # Every point is re-priced with the EXACT accounting...
        assert arm["exact_bytes"] == _expected_bytes(arm["assignment"], stats)
        assert point["bytes_vs_baseline"] == arm["exact_bytes"] - budget
        # ...and byte_neutral is exactly "at or under the baseline".
        assert point["byte_neutral"] is (arm["exact_bytes"] <= budget)
        if point["byte_neutral"]:
            assert arm["exact_bytes"] <= budget
        else:
            assert any("NOT byte-neutral" in n for n in arm["notes"])
            assert any("unfunded" in n for n in arm["notes"])


def test_byte_neutral_promotions_are_funded_by_cheaper_cb_rungs(tmp_path):
    """The discipline policy §4 requires: a promotion is paid for by demoting
    other units down their own CB ladder, not taken for free."""
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p, out="funded.json",
        extra=["--skip-matched-bytes", "--baseline-cb-format", "FP8_CB_K32"])
    sweep = report["experiments"]["byte_neutral_sweep"]
    first = sweep["points"][0]
    assert first["n_funding_moves"] >= 1
    for move in first["funding_moves"]:
        assert move["unit"] not in first["promoted_units"]
        assert move["from"] == "FP8_CB_K32"
        assert move["to"] == "FP8_CB_K28"          # down its own ladder
        assert move["additive_bytes_freed"] > 0
        assert move["predicted_dloss_increase"] > 0
    assert set(first["promoted_units"]).issubset(set(_DENSE))
    assert all(first["arm"]["assignment"][u] == "NVFP4"
               for u in first["promoted_units"])


def test_byte_neutral_sweep_is_deterministic(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    kw = dict(extra=["--skip-matched-bytes",
                     "--baseline-cb-format", "FP8_CB_K32"])
    a = _run(tmp_path, probe_p, cost_p, out="a.json", **kw)
    b = _run(tmp_path, probe_p, cost_p, out="b.json", **kw)
    assert (a["experiments"]["byte_neutral_sweep"]
            == b["experiments"]["byte_neutral_sweep"])


def test_unfundable_promotion_is_reported_not_shipped(tmp_path):
    """When the ladder cannot free enough bytes the point must fail the gate.
    With the baseline already at the cheapest rung there is nothing to demote,
    so a promotion that costs bytes can never be funded."""
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p, out="unfunded.json",
        extra=["--skip-matched-bytes", "--baseline-cb-format", "FP8_CB_K28"])
    sweep = report["experiments"]["byte_neutral_sweep"]
    assert all(p["n_funding_moves"] == 0 for p in sweep["points"])
    assert all(not p["byte_neutral"] for p in sweep["points"])
    assert all(p["bytes_vs_baseline"] > 0 for p in sweep["points"])


# ---------------------------------------------------------------------------
# 3. The cross-family refusal — the whole purpose of the P5a band check
# ---------------------------------------------------------------------------
def test_cross_family_verdict_is_withheld_when_the_band_check_failed(
        tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path, cross_family_verdict={
        "verdict": "asymmetric",
        "cross_family_comparison_publishable": False,
        "detail": "fp8_cb residual sits 0.4 log2 above nvfp4_cb's",
    })
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    cross = report["cross_family"]
    assert cross["publishable"] is False
    assert cross["verdict"] is None
    assert cross["refusal_reason"] == "p5a_cross_family_band_check_failed"
    # The numbers behind the failure are kept so it is actionable...
    assert cross["band_check"]["detail"].startswith("fp8_cb residual")
    # ...but the verdict string never appears in the operator summary.
    summary = d03.summarize(report)
    assert "WITHHELD" in summary
    assert "asymmetric" not in summary


def test_cross_family_verdict_is_published_when_the_band_check_passed(
        tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path, cross_family_verdict={
        "verdict": "symmetric",
        "cross_family_comparison_publishable": True,
        "detail": "both families inside the derived band",
    })
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    cross = report["cross_family"]
    assert cross["publishable"] is True
    assert cross["verdict"] == "symmetric"
    # ...and it still is not a promotion.
    assert "does not make predicted" in cross["still_not_a_promotion"]
    assert "symmetric" in d03.summarize(report)


def test_cross_family_verdict_is_withheld_when_no_band_check_ran(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    cross = report["cross_family"]
    assert cross["publishable"] is False
    assert cross["refusal_reason"] == (
        "no_cross_family_band_check_in_the_cost_payload")


# ---------------------------------------------------------------------------
# 4. Scope, labelling and constraint provenance
# ---------------------------------------------------------------------------
def test_packed_expert_exclusion_is_recorded_and_cites_d0_2(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path, with_experts=True)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    exclusion = report["packed_expert_exclusion"]
    assert exclusion["excluded"] == "packed_expert_vanilla_nvfp4"
    assert "D0.2" in exclusion["unlocked_by"]
    assert "no-new-packer" in exclusion["reason"]
    scope = report["experiments"]["matched_bytes"]["scope"]
    assert scope["n_excluded_by_marker"] == len(_EXPERTS)
    assert scope["n_eligible_units"] == len(_DENSE)
    assert scope["packed_expert_exclusion"] == exclusion
    assert "D0.2" in d03.summarize(report)


def test_every_report_is_labelled_proposal_data(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p)
    assert "PROPOSAL DATA" in report["evidence_status"]
    assert "NATIVE-PARITY" in report["evidence_status"]
    for exp in report["experiments"].values():
        assert "PROPOSAL DATA" in exp["evidence_status"]
    assert "PROPOSAL DATA" in d03.summarize(report)


def test_report_records_the_activation_fair_pricing_it_used(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    pricing = report["activation_fair_pricing"]
    assert "enabled" in pricing and "functional_form" in pricing
    arm = report["experiments"]["matched_bytes"]["arms"][0]
    assert "activation_fair_pricing" in arm["predicted_dloss_pricing"]


def test_report_records_serving_lanes_per_arm(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(
        tmp_path, probe_p, cost_p,
        extra=["--skip-byte-neutral", "--target-profile", "nvfp4_cb"])
    lanes = report["serving_lane_provenance_by_arm"]
    cb = lanes["cb:FP8_CB_K36"]
    native = lanes["native:NVFP4"]

    # Vanilla NVFP4 has no declared CB lane at all.
    assert native["units_without_declared_lane"] == len(_DENSE)

    # The CB arm's units are fully accounted for, and the report stamps the
    # runtime the route was resolved against -- that provenance is the point
    # of this record, because the backed set is attested PER GRIDBOOK RELEASE
    # and therefore moves with the pin.
    assert cb["gridbook_runtime_version"] == gridbook_runtime_version()
    assert (
        cb["units_on_backed_fused_mid_m_lane"]
        + cb["units_on_fallback_route"]
        + cb["units_without_declared_lane"]
    ) == len(_DENSE)

    # Which side of that split K36 lands on is a function of the pin, so it is
    # asserted against the pin rather than hardcoded. The repo currently pins
    # Whether K36 is backed is resolved from the immutable released-runtime
    # table; a future pin with no key must fall back rather than inheriting a
    # prior release's evidence.
    if 36 in serving_profile_backed_rungs("nvfp4_cb"):
        assert cb["units_on_backed_fused_mid_m_lane"] == len(_DENSE)
        assert cb["selected_rungs_fused_mid_m_backed"] == [36]
    else:
        assert cb["units_on_fallback_route"] == len(_DENSE)
        assert cb["selected_rungs_fused_mid_m_backed"] == []
        assert cb["selected_rungs_on_fallback_route"] == [36]


def serving_profile_backed_rungs(profile_id: str) -> tuple[int, ...]:
    """The fused mid-M rungs the CURRENT pin attests for ``profile_id``."""
    lane = serving_lane_route(profile_id, "FP8_CB_K36")
    if lane is None or not lane.fused_mid_m_backed:
        return ()
    return tuple(lane.fused_mid_m_rungs)


def test_no_serving_constraints_stamps_that_none_were_evaluated(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    report = _run(tmp_path, probe_p, cost_p, extra=["--skip-byte-neutral"])
    stamp = report["serve_constraints_context"]
    assert stamp["active"] is False
    assert "no latency" in stamp["note"]
    assert "fastest_feasible_reference" not in report
    arm = report["experiments"]["matched_bytes"]["arms"][0]
    assert arm["serve_constraints"]["active"] is False


def test_serving_constraints_are_evaluated_per_arm_when_supplied(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({
        "schema": TABLE_SCHEMA, "table_id": "fixture",
        "status": "proposal_data",
        "arenas": [{
            "phase": "prefill", "m_regime": "dense", "m": 1400,
            "reference_route": "fixture native", "metric": "ttft_ms",
            "absolute_value": 1000.0, "statistic": "p95",
            "provenance": dict(_PROV),
        }],
        "rows": [
            {"format_family": "NVFP4", "phase": "prefill",
             "m_regime": "dense", "lane": "native",
             "relative_unit_cost": 1.0, "provenance": dict(_PROV)},
            {"format_family": "FP8_CB", "phase": "prefill",
             "m_regime": "dense", "lane": "fallback",
             "relative_unit_cost": 1.44, "provenance": dict(_PROV)},
        ],
    }))
    report = _run(
        tmp_path, probe_p, cost_p, out="constrained.json",
        extra=[
            "--skip-byte-neutral",
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--slo-prefill-p95-ttft-ms", "1200",
            "--target-profile", "nvfp4_cb",
        ])
    arms = {a["label"]: a for a in
            report["experiments"]["matched_bytes"]["arms"]}
    cb = arms["cb:FP8_CB_K36"]["serve_constraints"]
    native = arms["native:NVFP4"]["serve_constraints"]
    # The audit's headline trade, made mechanical: the CB arm costs 1.44x the
    # native arm on dense prefill and misses a 1200 ms SLO the native arm
    # clears.
    assert cb["predicted"]["p95_ttft_ms"] == pytest.approx(1440.0)
    assert native["predicted"]["p95_ttft_ms"] == pytest.approx(1000.0)
    assert cb["feasible"] is False and native["feasible"] is True
    assert cb["binding_constraint"] == "p95_ttft_ms"
    # The relative-tax reference rule travels with the contest and says how
    # narrow this denominator is.
    ref = report["fastest_feasible_reference"]
    assert ref["per_phase"]["p95_ttft_ms"]["label"] == "native:NVFP4"
    assert "not an enumeration" in ref["scope"]
    assert "GLOBALLY FEASIBLE" in ref["reference_rule"]


def test_harness_reports_a_bad_dispatch_table_as_an_error(tmp_path):
    probe_p, cost_p, _stats = _fixture(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "nope"}')
    assert d03.main([
        "--probe", str(probe_p), "--costs", str(cost_p),
        "--out", str(tmp_path / "never.json"),
        "--serve-dispatch-table", str(bad),
    ]) == 2
