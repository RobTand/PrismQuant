"""Exercise allocator admission using actual streamed joint measurement rows."""
import copy

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant.cost_currency import CostCurrencyError, require_run_currency
from prismaquant.joint_aura import identity_sha256, make_joint_aura_entry
from test_joint_aura_streamed import _fixture, _run


@pytest.fixture(scope="module")
def measured_payload():
    _, _, runner, cache = _fixture()
    return _run(runner, cache)


def _first(payload):
    name = next(iter(payload["costs"]))
    return name, payload["costs"][name]["FP8_E4M3"]


def test_actual_joint_table_admits_in_one_aura_currency(measured_payload):
    report = require_run_currency(measured_payload)
    assert report["joint_aura_rows"] == sum(map(len, measured_payload["costs"].values()))
    assert report["activation_quantization_included"] is True


def test_joint_cost_consumed_without_scalar_sensitivity_or_activation_transfer(measured_payload, monkeypatch):
    _, row = _first(measured_payload)
    monkeypatch.setattr(ac, "_activation_penalty", lambda *args: 1000.0)
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    assert ac.cost_entry_predicted_dloss(
        {"h_trace": 1e8}, row, format_name="FP8_E4M3") == row["predicted_dloss"]
    assert ac.cost_entry_activation_pricing_branch({}, row, "FP8_E4M3") == "joint_aura"


@pytest.mark.parametrize("gain", [0.5, 2.0, float("nan")])
def test_joint_cost_refuses_second_gain(measured_payload, gain):
    _, row = _first(measured_payload)
    with pytest.raises(ValueError, match="gain"):
        ac.cost_entry_predicted_dloss({}, row, gain=gain, format_name="FP8_E4M3")


def test_joint_row_cannot_price_another_format(measured_payload):
    _, row = _first(measured_payload)
    with pytest.raises(ValueError, match="different format"):
        ac.cost_entry_predicted_dloss({}, row, format_name="NVFP4A16")


@pytest.mark.parametrize("field,value", [("fisher_application_count", True),
    ("activation_pricing_applied", True), ("act_dloss", 1.0),
    ("output_mse", 0.0), ("predicted_dloss_stderr", float("nan"))])
def test_malformed_joint_claim_never_falls_into_generic_pricing(measured_payload, field, value):
    _, source = _first(measured_payload)
    row = copy.deepcopy(source)
    row[field] = value
    with pytest.raises(ValueError, match="joint AURA"):
        ac.cost_entry_predicted_dloss({"h_trace": 1.0}, row, format_name="FP8_E4M3")


def test_joint_table_refuses_weight_only_control(measured_payload):
    payload = copy.deepcopy(measured_payload)
    name, _ = _first(payload)
    payload["costs"][name]["BF16"] = {"predicted_dloss": 0.0}
    with pytest.raises(CostCurrencyError, match="mixes joint"):
        require_run_currency(payload)


def test_joint_table_binds_row_coordinates(measured_payload):
    payload = copy.deepcopy(measured_payload)
    name, row = _first(payload)
    payload["costs"][name]["NVFP4A16"] = row
    with pytest.raises(CostCurrencyError, match="cost-table key"):
        require_run_currency(payload)


def test_joint_table_refuses_unaligned_probe_population(measured_payload):
    payload = copy.deepcopy(measured_payload)
    _, row = _first(payload)
    probe = copy.deepcopy(row["probe_identity"])
    probe["seed_base"] += 1
    operator = copy.deepcopy(row["joint_operator_identity"])
    operator["probe_identity_sha256"] = identity_sha256(probe)
    replacement = make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
        signed_components=row["signed_components_per_probe"])
    payload["costs"][operator["qname"]][operator["format"]] = replacement
    with pytest.raises(CostCurrencyError, match="probe/calibration identity"):
        require_run_currency(payload)


def test_group_uncertainty_does_not_reapply_activation_penalty(measured_payload):
    _, row = _first(measured_payload)
    linear, bound = ac._super_item_ucb_hedge(
        [({}, row, row["predicted_dloss"], 1000.0)], 2.0)
    assert bound == row["predicted_dloss_stderr"]
    assert linear == 2.0 * bound
