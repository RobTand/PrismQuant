"""Identity regressions for the existing pipeline diagnostic (CPU only)."""
from __future__ import annotations

import copy
import json
import math
import pickle

import pytest

from prismaquant.aura_additivity_gate import additivity_gate, main, paired_assignment_report
from test_joint_aura_assignment_diagnostics import UNIT_A, UNIT_B, _rebuild, _row


def _payload(a, b):
    return {"n_probes": 3, "costs": {
        UNIT_A: {"FP8_E4M3": a}, UNIT_B: {"FP8_E4M3": b}}}


ASSIGNMENT = {UNIT_A: "FP8_E4M3", UNIT_B: "FP8_E4M3"}


@pytest.mark.parametrize("field", ["seed_base", "calibration_sha256", "producer_source_sha256"])
def test_equal_length_joint_rows_from_different_measurements_refuse(field):
    a, b = _row(UNIT_A, [1, 2, 3]), _row(UNIT_B, [3, 2, 1])
    replacement = 237001 if field == "seed_base" else "f" * 64
    b = _rebuild(b, probe_change=lambda probe: probe.update({field: replacement}))
    # Both rows are complete and internally valid; only their alignment differs.
    with pytest.raises(ValueError, match="probe|alignment"):
        additivity_gate(_payload(a, b), ASSIGNMENT, measured_kl=0.1)


def test_gate_validates_render_binding_before_sampling_claim():
    a, b = _row(UNIT_A, [1, 2, 3]), _row(UNIT_B, [3, 2, 1])
    b["joint_operator_identity"]["rendered_weight"]["content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        additivity_gate(_payload(a, b), ASSIGNMENT, measured_kl=0.1)


def test_bare_arrays_never_claim_verified_probe_alignment():
    a = {"predicted_dloss": 7 / 3, "predicted_dloss_stderr": 7 / 6,
         "x2_per_probe": [1.0, 4.0, 9.0]}
    b = copy.deepcopy(a)
    result = additivity_gate(_payload(a, b), ASSIGNMENT, measured_kl=0.1)
    assert result["stderr_method"] == "per_probe_unverified"
    assert result["probe_alignment_verified"] is False


def test_gate_verified_joint_covariance_keeps_sequence_uncertainty_separate():
    a, b = _row(UNIT_A, [2, -4, 6]), _row(UNIT_B, [-2, 4, -6])
    result = additivity_gate(_payload(a, b), ASSIGNMENT, measured_kl=20,
                             measured_kl_stderr=0.25)
    assert result["predicted_sum"] == pytest.approx(56 / 3)
    assert result["predicted_stderr"] == pytest.approx(28 / 3)
    assert result["measured_kl_stderr"] == 0.25
    assert result["stderr_method"] == "per_probe_aligned_empirical"
    assert result["probe_alignment_verified"] is True
    assert result["probe_identity_sha256"] == a["probe_identity_sha256"]
    assert result["predicted_uncertainty_scope"] == "probe_sampling_conditional_on_fixed_calibration"
    assert result["measured_uncertainty_scope"] == "caller_supplied_heldout_sequence_standard_error"
    assert result["residual_z"] == pytest.approx((20 - 56 / 3) / math.hypot(28 / 3, 0.25))


@pytest.mark.parametrize("mutation", ["format", "qname", "n_probes", "mixed_currency", "zero_shortcut"])
def test_gate_refuses_incompatible_payload_coordinates_and_legacy_rows(mutation):
    a, b = _row(UNIT_A, [1, 2, 3]), _row(UNIT_B, [3, 2, 1])
    payload = _payload(a, b)
    if mutation in ("format", "qname"):
        b = _rebuild(b, operator_change=lambda op: op.update(
            {mutation: "NVFP4A16" if mutation == "format" else UNIT_A}))
        payload["costs"][UNIT_B]["FP8_E4M3"] = b
    elif mutation == "n_probes":
        payload["n_probes"] = 4
    else:
        payload["costs"][UNIT_B]["FP8_E4M3"] = {
            "predicted_dloss": 0.0 if mutation == "zero_shortcut" else 1.0}
    with pytest.raises(ValueError):
        additivity_gate(payload, ASSIGNMENT, measured_kl=0.1)


def test_report_cli_emits_explicit_paired_objective_from_validated_rows(tmp_path, capsys):
    a, b = _row(UNIT_A, [-1, 0, 1]), _row(UNIT_B, [10, 10, 10])
    payload = _payload(a, b)
    payload["costs"][UNIT_A]["FP8_E5M2"] = _row(UNIT_A, [1, 0, -1], fmt="FP8_E5M2")
    comparison = {**ASSIGNMENT, UNIT_A: "FP8_E5M2"}
    cost_path, assignment_path, comparison_path, output_path = [
        tmp_path / name for name in ("cost.pkl", "a.json", "b.json", "report.json")]
    cost_path.write_bytes(pickle.dumps(payload))
    assignment_path.write_text(json.dumps(ASSIGNMENT))
    comparison_path.write_text(json.dumps(comparison))
    assert main(["--costs", str(cost_path), "--assignment", str(assignment_path),
                 "--measured-kl", "0.1", "--measured-kl-stderr", "0.002",
                 "--comparison-assignment", str(comparison_path),
                 "--paired-objective", "joint_quadratic", "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert report["objective"] == "additive"
    assert report["measured_kl_stderr"] == 0.002
    paired = report["paired_assignment"]
    assert paired["objective"] == "joint_quadratic"
    assert paired["difference_per_probe"] == [-20.0, 0.0, 20.0]
    assert paired["paired_standard_error"] == pytest.approx(20 / math.sqrt(3))
    assert "measured_kl_stderr" not in paired


def test_paired_report_requires_measured_rows_for_the_whole_roster():
    a, b = _row(UNIT_A, [1, 2, 3]), _row(UNIT_B, [3, 2, 1])
    payload = _payload(a, b)
    with pytest.raises(ValueError, match="missing"):
        paired_assignment_report(payload, ASSIGNMENT, {**ASSIGNMENT, UNIT_A: "BF16"})
    payload["costs"][UNIT_A]["BF16"] = {"predicted_dloss": 0.0}
    with pytest.raises(ValueError, match="complete joint"):
        paired_assignment_report(payload, ASSIGNMENT, {**ASSIGNMENT, UNIT_A: "BF16"})
