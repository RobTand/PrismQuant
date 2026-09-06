"""CPU replay contracts; synthetic receipts are not producer qualification."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from prismaquant.cost_stage_checkpoint import MANIFEST_SCHEMA, canonical_json_sha256, write_unit
from prismaquant.tessera_anchored_surface import (
    CAMPAIGN_SCHEMA, CURRENCY, PLAN_SCHEMA, ReplayError, load_campaign_measurements, replay_campaign,
)


def seal(value):
    return canonical_json_sha256(value, where="test fixture")


def make_campaign(tmp_path, *, altered=None, missing=(), extra=()):
    """Real journal envelopes; inert blobs explicitly exercise hash-only replay."""
    coordinates = {f"F_R{r}": r for r in range(1, 6)}
    pilot = {"p1": ["F_R1", "F_R2", "F_R5"], "p2": ["F_R1", "F_R2", "F_R5"]}
    heldout = {"h": {"anchors": ["F_R1", "F_R5"], "audit": ["F_R3"]}}
    observed = {**pilot, "h": ["F_R1", "F_R3", "F_R5", *extra], "h_sibling": []}
    recipe = {"grid": "synthetic-grid", "body": "synthetic-test-only"}
    descriptor = {"family": "TESSERA_SYNTHETIC", "activation_contract": "fp4_e2m1",
                  "geometry": [8, 8], "wire_recipe": recipe, "hessian_applied": False,
                  "role": "profile-declared-q-proj"}
    wire_dir = tmp_path / "wires"
    wire_dir.mkdir()
    source = {"shape": [8, 8], "dtype": "bfloat16", "sha256": "a" * 64}
    units = {unit: {"weight": {**source, "sha256": hashlib.sha256(unit.encode()).hexdigest()},
                    "scoring_rows": {"shape": [4, 8], "sha256": "b" * 64},
                    "hessian": None, "input_global_scale": None, "menu": list(coordinates)}
             for unit in observed}
    identity = {"campaign_schema": CAMPAIGN_SCHEMA, "currency": CURRENCY,
                "units": units, "prismaquant_source_sha256": "c" * 64,
                "encoder_source_sha256": "d" * 64, "calibration": {"fit_ids_sha256": "e" * 64}}
    checkpoint = tmp_path / "cost.anchors.json"
    manifest = {"schema": MANIFEST_SCHEMA, "stage": "Tessera campaign", "identity": identity,
                "identity_sha256": seal(identity),
                "units": [{"qname": unit} for unit in observed]}
    checkpoint.write_text(json.dumps(manifest))
    costs = {}
    values = {}
    for unit, keys in observed.items():
        anchors, records = [], {}
        costs[unit] = {}
        for key in keys:
            if (unit, key) in missing:
                continue
            r = coordinates[key]
            log_value = -.15 * r - .02 * r * r
            log_value += {"p1": 0, "p2": .4, "h": .2 + .025 * r}[unit]
            value = (altered or {}).get((unit, key), 10 ** log_value)
            values[(unit, key)] = value
            blob = f"inert fixture {unit} {key}".encode()
            filename = f"{unit}-{key}.wire"
            (wire_dir / filename).write_bytes(blob)
            anchor = {"qname": unit, "format_name": key, "family": descriptor["family"],
                      "body_rate_q256": r, "dloss": value, "activation_contract": "fp4_e2m1",
                      "activation_quantized": True, "wire_bytes": len(blob), "hessian_applied": False}
            anchors.append(anchor)
            records[key] = {"file": filename, "blob_bytes": len(blob),
                            "blob_sha256": hashlib.sha256(blob).hexdigest(),
                            "identity": {"unit": unit, "source": units[unit]["weight"],
                                         "encoder_source_sha256": "d" * 64, "calibration": None,
                                         "recipe": {**recipe, "q256": r}}}
            costs[unit][key] = {"output_mse": value, "output_mse_measured": True,
                               "cost_source": "tessera_campaign_measured", "tessera_provenance": "measured",
                               "currency": CURRENCY, "tessera_family": descriptor["family"],
                               "tessera_body_rate_q256": r, "activation_contract": "fp4_e2m1",
                               "activation_quantized": True, "wire_bytes": len(blob),
                               "hessian_identity": {"applied": False, "supplied": False}}
        write_unit(checkpoint.with_name(checkpoint.name + ".parts"), stage="Tessera campaign",
                   qname=unit, identity_sha256=seal(identity), state={"anchors": anchors, "wire_records": records})
    payload = {"schema": CAMPAIGN_SCHEMA, "currency": CURRENCY, "costs": costs,
               "provenance": {"cost_mode": "production-render-score", "wire_dir": str(wire_dir),
                              "hessian": {"supplied": False}, "anchor_groups": {"group": ["h", "h_sibling"]}}}
    cost_path = tmp_path / "cost.pkl"
    cost_path.write_bytes(pickle.dumps(payload))
    plan = {"schema": PLAN_SCHEMA, "currency": CURRENCY,
            "input": {"payload_sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest(),
                      "checkpoint_identity_sha256": seal(identity)},
            "segments": [{"id": "fp4-q", "descriptor": descriptor, "pilot": pilot, "heldout": heldout,
                          "coordinates": coordinates, "features_by_key": {key: [r, r*r] for key, r in coordinates.items()},
                          "max_absolute_log10_error": .01, "refit_after_audit": False}]}
    return cost_path, checkpoint, plan, values


def unit_report(report):
    return report["segments"][0]["units"]["h"]


def test_two_anchor_curvature_reconstruction_and_exact_measured_rows(tmp_path):
    costs, checkpoint, plan, values = make_campaign(tmp_path)
    report = replay_campaign(costs, checkpoint, plan)
    unit = unit_report(report)
    assert unit["status"] == "audit_accepted_research_only"
    assert unit["predictions"]["F_R2"]["value"] == pytest.approx(10 ** (-.15*2-.02*4+.2+.025*2))
    for key in ("F_R1", "F_R3", "F_R5"):
        assert unit["predictions"][key] == {"value": values[("h", key)], "source": "measured"}
    assert not report["production_qualified"] and not report["allocator_payload"]
    assert report["latency_evidence"] is None and report["currency"] == CURRENCY
    assert report["measurement_requests"] == []
    assert unit["pwl_audit"][0]["absolute_log10_error"] > .05
    assert unit["audit"][0]["held_out_axes"] == ["unit", "rung"]
    assert replay_campaign(costs, checkpoint, plan) == report


def test_one_anchor_does_not_invent_the_missing_slope(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    plan["segments"][0]["heldout"]["h"]["anchors"] = ["F_R1"]
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["status"] == "measure_more"
    assert unit_report(report)["pwl_audit"][0]["unavailable"] == "needs_two_valid_anchors"
    requests = report["measurement_requests"]
    assert {r["unit"] for r in requests} == {"h", "h_sibling"}
    assert {r["key"] for r in requests} == set(plan["segments"][0]["coordinates"])


def test_pwl_baseline_refuses_nonmonotone_anchors_like_current_surface(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path, altered={("h", "F_R5"): 100.0})
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["pwl_audit"][0]["unavailable"] == "nonmonotone_anchors"


@pytest.mark.parametrize("anchor_scale,audit_value", [(1e200, 1e-200), (1e-200, 1e200)])
def test_pwl_audit_extreme_finite_costs_produce_a_serializable_failure(
    tmp_path, anchor_scale, audit_value,
):
    costs, checkpoint, plan, _ = make_campaign(tmp_path, altered={
        ("h", "F_R1"): anchor_scale, ("h", "F_R5"): anchor_scale / 2,
        ("h", "F_R3"): audit_value,
    })
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["status"] == "measure_more"
    assert unit_report(report)["pwl_audit"][0]["absolute_log10_error"] == pytest.approx(
        abs((math.log10(anchor_scale) + math.log10(anchor_scale / 2)) / 2
            - math.log10(audit_value)))
    json.dumps(report, allow_nan=False)


def test_audit_prediction_is_frozen_before_refit(tmp_path):
    costs, checkpoint, plan, values = make_campaign(tmp_path)
    base = replay_campaign(costs, checkpoint, plan)
    other = tmp_path / "other"
    other.mkdir()
    costs2, checkpoint2, plan2, _ = make_campaign(other, altered={("h", "F_R3"): values[("h", "F_R3")] * 1.01})
    plan2["segments"][0]["refit_after_audit"] = True
    changed = replay_campaign(costs2, checkpoint2, plan2)
    assert unit_report(base)["audit"][0]["predicted"] == unit_report(changed)["audit"][0]["predicted"]
    assert unit_report(changed)["refit_after_audit"] is True
    assert unit_report(changed)["audit"][0]["measured"] != unit_report(base)["audit"][0]["measured"]


def test_missing_audit_and_rank_failure_emit_measurement_requests(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path, missing={("h", "F_R3")})
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["reason"] == "missing_or_invalid_anchor_or_audit"
    assert {row["key"] for row in report["measurement_requests"]} == {"F_R3"}
    other = tmp_path / "rank"
    other.mkdir()
    costs, checkpoint, plan, _ = make_campaign(other)
    plan["segments"][0]["features_by_key"] = {key: [1, 1] for key in plan["segments"][0]["coordinates"]}
    report = replay_campaign(costs, checkpoint, plan)
    assert report["segments"][0]["pilot_fit_error"]
    assert unit_report(report)["predictions"] == {}


@pytest.mark.parametrize("field,value", [("activation_contract", "fp8_e4m3"), ("family", "OTHER"),
                                         ("geometry", [16, 8]), ("hessian_applied", True),
                                         ("wire_recipe", {"grid": "other"})])
def test_cross_segment_data_refuses(tmp_path, field, value):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    plan["segments"][0]["descriptor"][field] = value
    with pytest.raises(ReplayError, match="cross-segment"):
        replay_campaign(costs, checkpoint, plan)


def test_current_mse_cannot_be_relabelled_aura(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    plan["currency"] = "aura"
    with pytest.raises(ReplayError, match="output-MSE"):
        replay_campaign(costs, checkpoint, plan)


@pytest.mark.parametrize("kind", ["payload", "checkpoint", "wire", "source"])
def test_integrity_boundaries_refuse(tmp_path, kind):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    if kind == "payload":
        costs.write_bytes(costs.read_bytes() + b"changed")
    elif kind == "checkpoint":
        manifest = json.loads(checkpoint.read_text())
        manifest["identity"]["encoder_source_sha256"] = "f" * 64
        checkpoint.write_text(json.dumps(manifest))
    elif kind == "wire":
        (tmp_path / "wires" / "h-F_R1.wire").write_bytes(b"corrupt")
    else:
        payload = pickle.loads(costs.read_bytes())
        payload["costs"]["h"]["F_R1"]["output_mse"] = .003
        costs.write_bytes(pickle.dumps(payload))
        plan["input"]["payload_sha256"] = hashlib.sha256(costs.read_bytes()).hexdigest()
    with pytest.raises((ReplayError, RuntimeError), match="mismatch"):
        replay_campaign(costs, checkpoint, plan)


def test_disjoint_split_enforced(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    plan["segments"][0]["heldout"]["h"]["audit"] = ["F_R1"]
    with pytest.raises(ReplayError, match="disjoint"):
        replay_campaign(costs, checkpoint, plan)


def test_domain_cannot_extrapolate_beyond_pilot(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    for keys in plan["segments"][0]["pilot"].values():
        keys.remove("F_R5")
    with pytest.raises(ReplayError, match="pilot coordinate envelope"):
        replay_campaign(costs, checkpoint, plan)


def test_unfitted_measured_row_cannot_break_monotonicity(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(
        tmp_path, altered={("h", "F_R2"): 10.0}, extra=["F_R2"])
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["reason"] == "measured_overlay_nonmonotone"
    assert unit_report(report)["predictions"] == {}


def test_invalid_extra_measurement_is_not_replaced_by_a_prediction(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(
        tmp_path, altered={("h", "F_R2"): 0.0}, extra=["F_R2"])
    report = replay_campaign(costs, checkpoint, plan)
    assert unit_report(report)["status"] == "measure_more"
    assert unit_report(report)["predictions"] == {}
    assert {row["key"] for row in report["measurement_requests"]} == {"F_R2"}


def test_unrecorded_prediction_rung_refuses(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    segment = plan["segments"][0]
    segment["coordinates"]["FAKE"] = 4
    segment["features_by_key"]["FAKE"] = segment["features_by_key"].pop("F_R4")
    del segment["coordinates"]["F_R4"]
    with pytest.raises(ReplayError, match="recorded source menu"):
        replay_campaign(costs, checkpoint, plan)


def test_actual_campaign_main_output_imports(monkeypatch, tmp_path):
    # Existing fixture runs real campaign main, current journal and wire producer
    # on a tiny CPU Linear. Only model/capture/menu inputs are synthetic.
    from test_tessera_campaign_resume import UNIT, _fresh_priced_campaign
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    (_campaign, checkpoint, _argv, _model, _inputs), payload = _fresh_priced_campaign(
        monkeypatch, tmp_path)
    cost_path = tmp_path / "cost.pkl"
    manifest = json.loads(checkpoint.read_text())
    plan = {"schema": PLAN_SCHEMA, "currency": CURRENCY,
            "input": {"payload_sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest(),
                      "checkpoint_identity_sha256": manifest["identity_sha256"]}, "segments": []}
    measurements, identity, _groups = load_campaign_measurements(cost_path, checkpoint, plan)
    key = "TESSERA_E4M3_K1_R1024"
    assert measurements[(UNIT, key)]["value"] == payload["costs"][UNIT][key]["output_mse"]
    assert identity["source_recomputed"] is False


def test_cli_real_file_replay_idempotence_and_no_clobber(tmp_path):
    costs, checkpoint, plan, _ = make_campaign(tmp_path)
    plan_path, output = tmp_path / "plan.json", tmp_path / "report.json"
    plan_path.write_text(json.dumps(plan))
    tool = Path(__file__).resolve().parents[1] / "tools" / "tessera_surface_replay.py"
    command = [sys.executable, str(tool), "--costs", str(costs), "--checkpoint", str(checkpoint),
               "--plan", str(plan_path), "--out", str(output)]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
        assert result.returncode == 0, result.stderr
    assert unit_report(json.loads(output.read_text()))["status"] == "audit_accepted_research_only"
    plan["segments"][0]["max_absolute_log10_error"] = .02
    plan_path.write_text(json.dumps(plan))
    before = output.read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode == 2 and "different replay" in refused.stderr
    assert output.read_bytes() == before
