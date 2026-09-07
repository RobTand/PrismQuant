"""CPU scalar oracles for fixed-calibration assignment diagnostics.

Synthetic complete rows exercise identity admission and paired arithmetic;
these tests make no model-quality or serving-runtime claim.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import joint_aura as joint
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA


UNIT_A = "model.layers.0.mlp.down_proj"
UNIT_B = "model.layers.7.mlp.down_proj"
SCOPE = "probe_sampling_conditional_on_fixed_calibration"
OBJECTIVES = ("additive", "joint_quadratic")


def _row(name, signed, *, fmt="FP8_E4M3"):
    # The allocator currency fixture runs a model to build rows. These scalar
    # oracles instead use the same complete row builder and identity contract.
    source_content = {
        "config": {"fixture": "assignment diagnostics"},
        "weight_map": {"fixture.weight": "fixture.safetensors"},
        "shards": [{"path": "/fixture/fixture.safetensors", "size": 1,
                    "sha256": "a" * 64}],
    }
    source_model = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic",
        "resolved_commit": None, **source_content,
        "content_sha256": joint.identity_sha256(source_content),
    }
    arithmetic = joint.arithmetic_identity(torch.float32)
    probe = {
        "schema": "prismaquant.joint_aura.probes.v2", "seed_base": 237000,
        "n_probes": len(signed), "calibration_sha256": "c" * 64,
        "producer_source_sha256": "d" * 64, "source_model": source_model,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "temperature": 1.0, "arithmetic": arithmetic,
    }
    operator = {
        "schema": "prismaquant.joint_aura.operator.v2", "qname": name,
        "format": fmt, "probe_identity_sha256": joint.identity_sha256(probe),
        "source_weight": {
            "content_sha256": joint.identity_sha256({"source": name}),
            "shape": [2, 2], "dtype": "torch.float32", "logical_bytes": 16,
        },
        "rendered_weight": {
            "content_sha256": joint.identity_sha256({"render": name, "format": fmt}),
            "shape": [2, 2], "dtype": "torch.float32", "logical_bytes": 16,
        },
        "activation": joint.activation_identity(fr.get_format(fmt), {}, name),
        "arithmetic": arithmetic,
    }
    return joint.make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[dict(weight=float(x), activation=0.0, mixed=0.0,
                                total=float(x)) for x in signed],
    )


def _rebuild(row, *, operator_change=None, probe_change=None):
    """Re-sign changed identities so pairing, not a stale digest, must refuse."""
    operator = copy.deepcopy(row["joint_operator_identity"])
    probe = copy.deepcopy(row["probe_identity"])
    if operator_change:
        operator_change(operator)
    if probe_change:
        probe_change(probe)
    operator["probe_identity_sha256"] = joint.identity_sha256(probe)
    operator["arithmetic"] = copy.deepcopy(probe["arithmetic"])
    return joint.make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=copy.deepcopy(row["signed_components_per_probe"]),
    )


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_single_unit_summary_and_pair_match_independent_samples(objective):
    a = _row(UNIT_A, [2, 4, -6])
    b = _row(UNIT_A, [1, -3, 5], fmt="FP8_E5M2")
    summary = joint.assignment_probe_summary({UNIT_A: a}, objective=objective)
    assert summary["per_probe"] == [2.0, 8.0, 18.0]
    assert summary["mean"] == pytest.approx(28 / 3)
    assert summary["standard_error"] == pytest.approx(14 / 3)
    assert summary["probe_ids"] == [237000, 237001, 237002]
    assert summary["probe_identity_sha256"] == a["probe_identity_sha256"]
    paired = joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b},
                                              objective=objective)
    legacy = joint.paired_candidate_difference(a, b)
    for field in ("mean_difference", "paired_standard_error", "difference_per_probe"):
        assert paired[field] == pytest.approx(legacy[field])
    assert paired["difference_per_probe"] == [1.5, 3.5, 5.5]
    assert paired["paired_standard_error"] == pytest.approx(2 / math.sqrt(3))
    assert paired["uncertainty_scope"] == SCOPE


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_pair_preserves_common_probe_covariance(objective):
    a = _row(UNIT_A, [5, 4, -5, -4])
    b = _row(UNIT_A, [3, 0, 3, 0], fmt="FP8_E5M2")
    assert a["predicted_dloss_stderr"] > 0
    assert b["predicted_dloss_stderr"] > 0
    result = joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b},
                                              objective=objective)
    assert result["difference_per_probe"] == [8.0] * 4
    assert result["mean_difference"] == 8.0
    assert result["paired_standard_error"] == 0.0


def test_interunit_cancellation_keeps_objectives_distinct():
    rows = {UNIT_A: _row(UNIT_A, [2, -4, 6]),
            UNIT_B: _row(UNIT_B, [-2, 4, -6])}
    additive = joint.assignment_probe_summary(rows, objective="additive")
    quadratic = joint.assignment_probe_summary(rows, objective="joint_quadratic")
    assert additive["per_probe"] == [4.0, 16.0, 36.0]
    assert additive["mean"] == pytest.approx(56 / 3)
    assert additive["standard_error"] == pytest.approx(28 / 3)
    assert quadratic["per_probe"] == [0.0] * 3
    assert quadratic["mean"] == quadratic["standard_error"] == 0.0
    assert joint.assignment_probe_summary(rows) == additive


def test_unchanged_unit_remains_in_joint_quadratic_cross_terms():
    background = _row(UNIT_B, [10, 10, 10])
    a = {UNIT_A: _row(UNIT_A, [-1, 0, 1]), UNIT_B: background}
    b = {UNIT_A: _row(UNIT_A, [1, 0, -1], fmt="FP8_E5M2"),
         UNIT_B: copy.deepcopy(background)}
    additive = joint.paired_assignment_difference(a, b, objective="additive")
    quadratic = joint.paired_assignment_difference(a, b, objective="joint_quadratic")
    assert additive["difference_per_probe"] == [0.0] * 3
    assert quadratic["difference_per_probe"] == [-20.0, 0.0, 20.0]
    assert quadratic["mean_difference"] == 0.0
    assert quadratic["paired_standard_error"] == pytest.approx(20 / math.sqrt(3))
    assert joint.paired_assignment_difference(a, b) == additive


def test_close_swap_is_not_erased_by_rounding_large_background_totals():
    background = _row(UNIT_B, [1e16, 1e16, 1e16])
    a = {UNIT_A: _row(UNIT_A, [1, 2, 3]), UNIT_B: background}
    b = {UNIT_A: _row(UNIT_A, [0, 0, 0], fmt="FP8_E5M2"), UNIT_B: background}
    # Rounded full-assignment totals coincide; subtract local contributions
    # before the background sum to preserve the actual close-swap samples.
    assert joint.paired_assignment_difference(a, b)["difference_per_probe"] == [0.5, 2.0, 4.5]
    quadratic = joint.paired_assignment_difference(a, b, objective="joint_quadratic")
    assert quadratic["difference_per_probe"] == pytest.approx([1e16, 2e16, 3e16])


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_unchanged_assignment_has_zero_paired_uncertainty(objective):
    rows = {UNIT_A: _row(UNIT_A, [-2, 3, 5]), UNIT_B: _row(UNIT_B, [1, 4, -6])}
    result = joint.paired_assignment_difference(rows, copy.deepcopy(rows), objective=objective)
    assert result["difference_per_probe"] == [0.0] * 3
    assert result["mean_difference"] == result["paired_standard_error"] == 0.0
    assert result["uncertainty_scope"] == SCOPE


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_mapping_order_is_irrelevant_and_inputs_are_not_mutated(objective):
    a = {UNIT_A: _row(UNIT_A, [-2, 3, 5]), UNIT_B: _row(UNIT_B, [1, 4, -6])}
    b = {UNIT_A: _row(UNIT_A, [2, 1, 4], fmt="FP8_E5M2"),
         UNIT_B: _row(UNIT_B, [-3, 2, 1], fmt="FP8_E5M2")}
    before = copy.deepcopy((a, b))
    reversed_a, reversed_b = dict(reversed(a.items())), dict(reversed(b.items()))
    assert joint.assignment_probe_summary(a, objective=objective) == joint.assignment_probe_summary(
        reversed_a, objective=objective)
    assert joint.paired_assignment_difference(a, b, objective=objective) == joint.paired_assignment_difference(
        reversed_a, reversed_b, objective=objective)
    assert (a, b) == before


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_paired_assignments_require_same_complete_unit_roster(objective):
    a = {UNIT_A: _row(UNIT_A, [1, 2, 3]), UNIT_B: _row(UNIT_B, [4, 5, 6])}
    with pytest.raises(ValueError):
        joint.paired_assignment_difference(a, {UNIT_A: a[UNIT_A]}, objective=objective)


@pytest.mark.parametrize("mutation", ["source", "render", "activation"])
def test_rehashed_unchanged_format_identity_changes_are_refused(mutation):
    a = _row(UNIT_A, [1, 2, 3])
    def change(operator):
        if mutation == "activation":
            operator["activation"]["activation_max_abs"] = 7.0
        else:
            operator["source_weight" if mutation == "source" else "rendered_weight"]["content_sha256"] = "e" * 64
    b = _rebuild(a, operator_change=change)
    assert joint.validate_joint_aura_entry(b)
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b})


def test_changed_format_does_not_allow_different_source_weight():
    a = _row(UNIT_A, [1, 2, 3])
    b = _rebuild(_row(UNIT_A, [3, 2, 1], fmt="FP8_E5M2"),
                 operator_change=lambda op: op["source_weight"].update(content_sha256="e" * 64))
    assert joint.validate_joint_aura_entry(b)
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b})


@pytest.mark.parametrize("mutation", ["seed", "calibration", "arithmetic", "source_model"])
def test_rehashed_probe_contract_mismatches_fail_within_and_between_assignments(mutation):
    def change(probe):
        if mutation == "seed":
            probe["seed_base"] += 1
        elif mutation == "calibration":
            probe["calibration_sha256"] = "e" * 64
        elif mutation == "arithmetic":
            probe["arithmetic"]["measurement_dtype"] = "torch.bfloat16"
        else:
            model = probe["source_model"]
            model["config"]["revision"] = "different"
            model["content_sha256"] = joint.identity_sha256(
                {key: model[key] for key in ("config", "weight_map", "shards")})
    a = _row(UNIT_A, [1, 2, 3])
    b = _rebuild(_row(UNIT_B, [4, 5, 6]), probe_change=change)
    assert joint.validate_joint_aura_entry(b)
    with pytest.raises(ValueError):
        joint.assignment_probe_summary({UNIT_A: a, UNIT_B: b})
    changed_a = _rebuild(_row(UNIT_A, [3, 2, 1], fmt="FP8_E5M2"), probe_change=change)
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: changed_a})


@pytest.mark.parametrize("mutation", ["missing_probe", "permuted_probes", "duplicate_probe",
                                     "currency", "signed_component", "loss", "operator_digest"])
def test_both_diagnostics_apply_complete_joint_row_validation(mutation):
    good = _row(UNIT_A, [1, 2, 3])
    bad = copy.deepcopy(good)
    if mutation == "missing_probe":
        bad["probe_ids"].pop()
    elif mutation == "permuted_probes":
        bad["probe_ids"].reverse()
    elif mutation == "duplicate_probe":
        bad["probe_ids"][1] = bad["probe_ids"][0]
    elif mutation == "currency":
        bad["cost_currency"] = "output_mse"
    elif mutation == "signed_component":
        bad["signed_components_per_probe"][0]["mixed"] = 1.0
    elif mutation == "loss":
        bad["predicted_dloss"] += 1.0
    else:
        bad["joint_operator_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        joint.assignment_probe_summary({UNIT_A: bad})
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_A: good}, {UNIT_A: bad})


def test_mapping_coordinates_must_match_operator_qname():
    row = _row(UNIT_A, [1, 2, 3])
    with pytest.raises(ValueError):
        joint.assignment_probe_summary({UNIT_B: row})
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_B: row}, {UNIT_B: copy.deepcopy(row)})


def test_empty_assignments_cannot_publish_zero_uncertainty():
    with pytest.raises(ValueError):
        joint.assignment_probe_summary({})
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({}, {})


@pytest.mark.parametrize("objective", ["weight_only", "", None])
def test_unsupported_objective_refuses_instead_of_silently_mixing_costs(objective):
    rows = {UNIT_A: _row(UNIT_A, [1, 2, 3])}
    with pytest.raises(ValueError):
        joint.assignment_probe_summary(rows, objective=objective)
    with pytest.raises(ValueError):
        joint.paired_assignment_difference(rows, rows, objective=objective)


@pytest.mark.parametrize("consumer", ["summary", "assignment_pair", "candidate_pair"])
def test_probe_pairing_requires_canonical_identity_not_python_numeric_equality(consumer):
    a = _row(UNIT_A, [1, 2, 3])
    name = UNIT_B if consumer == "summary" else UNIT_A
    b = _rebuild(_row(name, [3, 2, 1], fmt="FP8_E5M2"),
                 probe_change=lambda probe: probe.update(temperature=True))
    # JSON true and 1.0 are distinct recorded identities, although Python's
    # nested equality treats them as equal. Both rows pass local validation.
    assert joint.validate_joint_aura_entry(b)
    assert a["probe_identity"] == b["probe_identity"]
    assert a["probe_identity_sha256"] != b["probe_identity_sha256"]
    with pytest.raises(ValueError):
        if consumer == "summary":
            joint.assignment_probe_summary({UNIT_A: a, UNIT_B: b})
        elif consumer == "assignment_pair":
            joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b})
        else:
            joint.paired_candidate_difference(a, b)


def test_same_candidate_requires_canonical_operator_identity():
    a = _rebuild(_row(UNIT_A, [1, 2, 3]), operator_change=lambda op:
                 op["activation"].update(clip_enabled=True))
    b = _rebuild(a, operator_change=lambda op: op["activation"].update(clip_enabled=1))
    assert joint.validate_joint_aura_entry(a)
    assert joint.validate_joint_aura_entry(b)
    assert a["probe_identity_sha256"] == b["probe_identity_sha256"]
    assert a["joint_operator_identity"] == b["joint_operator_identity"]
    assert a["joint_operator_identity_sha256"] != b["joint_operator_identity_sha256"]
    with pytest.raises(ValueError):
        joint.paired_assignment_difference({UNIT_A: a}, {UNIT_A: b})


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_pair_retains_small_residual_across_large_cancelling_unit_changes(objective):
    unit_c = "model.layers.21.mlp.down_proj"
    names = [UNIT_A, UNIT_B, unit_c]
    a = {name: _row(name, [value] * 3)
         for name, value in zip(names, [1e16, -1e16, 1.0])}
    b = {name: _row(name, [value] * 3, fmt="FP8_E5M2")
         for name, value in zip(names, [0.0, -1e16, 1e16])}
    # Additive: .5 * (1e32 + 1e32 + 1 - 0 - 1e32 - 1e32) = .5.
    # Joint quadratic: .5 * (1**2 - 0**2) = .5. Subtracting
    # rounded unit totals first loses the 1 in either expression.
    result = joint.paired_assignment_difference(a, b, objective=objective)
    assert result["difference_per_probe"] == [0.5] * 3
    assert result["mean_difference"] == 0.5
    assert result["paired_standard_error"] == 0.0
