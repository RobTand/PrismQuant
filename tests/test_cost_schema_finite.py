"""Finite-number contract for cost rows before allocator consumption."""
from __future__ import annotations

import copy

import pytest

from prismaquant.schemas import SchemaValidationError, validate_cost_payload


_UNIT = "model.layers.0.self_attn.o_proj"
_FORMAT = "NVFP4"
_PATH = "external costs.pkl"
_SCALAR_FIELDS = (
    "weight_mse", "predicted_dloss", "output_mse", "fisher_output_mse",
)


def _payload(entry):
    return {"costs": {_UNIT: {_FORMAT: entry}}}


@pytest.mark.parametrize("field", _SCALAR_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_nonfinite_cost_scalar_refuses_at_exact_field(field, value):
    payload = _payload({field: value})
    with pytest.raises(SchemaValidationError) as caught:
        validate_cost_payload(payload, _PATH)
    assert str(caught.value) == (
        f"{_PATH}:.costs[{_UNIT!r}][{_FORMAT!r}].{field}: "
        "expected a finite number"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_nonfinite_per_expert_cost_refuses_at_exact_element(value):
    payload = _payload({
        "weight_mse": 1.0,
        "weight_mse_per_expert": [0.0, value],
    })
    with pytest.raises(SchemaValidationError) as caught:
        validate_cost_payload(payload, _PATH)
    assert str(caught.value) == (
        f"{_PATH}:.costs[{_UNIT!r}][{_FORMAT!r}].weight_mse_per_expert[1]: "
        "expected a finite number"
    )


def test_valid_signal_cannot_hide_another_nonfinite_scalar():
    payload = _payload({"predicted_dloss": 1.0, "output_mse": float("nan")})
    with pytest.raises(SchemaValidationError, match="output_mse: expected a finite number"):
        validate_cost_payload(payload, _PATH)


@pytest.mark.parametrize("value", [0, 0.0, -1.0, 1e-300, -1e300, 1e300])
def test_finite_cost_values_have_no_new_threshold_and_are_not_rewritten(value):
    payload = _payload({
        **{field: value for field in _SCALAR_FIELDS},
        "weight_mse_per_expert": [value],
    })
    original = copy.deepcopy(payload)
    assert validate_cost_payload(payload, _PATH) is payload
    assert payload == original


def test_failed_measurement_row_keeps_nonfinite_diagnostics_uninterpreted():
    payload = _payload({"error": "measurement failed", "weight_mse": float("inf")})
    assert validate_cost_payload(payload, _PATH) is payload
