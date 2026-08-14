from __future__ import annotations

import pytest

from prismaquant.schemas import (
    SchemaValidationError,
    validate_cost_payload,
    validate_layer_config_payload,
    validate_probe_payload,
)


def _probe_payload():
    return {
        "stats": {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "h_trace": 0.25,
                "n_params": 64,
                "in_features": 8,
                "out_features": 8,
                "num_experts": 4,
            }
        },
        "meta": {"model": "/models/tiny", "top_k": 2},
        "expert_info": {
            "model.layers.0.mlp.experts.0.gate_proj": (
                "model.layers.0.mlp.gate",
                "0",
            )
        },
    }


def _cost_payload():
    return {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "NVFP4": {"weight_mse": 0.01, "predicted_dloss": 0.001},
                "BF16": {"weight_mse": 0.0},
            }
        },
    }


def _layer_config_payload():
    return {
        "model.layers.0.mlp.experts.gate_up_proj": {
            "data_type": "nv_fp",
            "bits": 4,
            "group_size": 16,
        },
        "model.layers.0.self_attn.o_proj": "BF16",
        "model.layers.0.mlp.down_proj": 8,
    }


def test_valid_handoff_payloads_pass():
    assert validate_probe_payload(_probe_payload(), "probe.pkl") is not None
    assert validate_cost_payload(_cost_payload(), "cost.pkl") is not None
    assert validate_layer_config_payload(_layer_config_payload(), "layer.json") is not None


def test_probe_payload_rejects_missing_required_stat_field():
    payload = _probe_payload()
    del payload["stats"]["model.layers.0.mlp.experts.gate_up_proj"]["h_trace"]
    with pytest.raises(SchemaValidationError, match="h_trace"):
        validate_probe_payload(payload, "probe.pkl")


def test_cost_payload_rejects_usable_entry_without_error_or_signal():
    payload = _cost_payload()
    payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"]["NVFP4"] = {}
    with pytest.raises(SchemaValidationError, match="usable cost entry"):
        validate_cost_payload(payload, "cost.pkl")


def test_cost_payload_validates_cost_source_metadata():
    payload = _cost_payload()
    payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"]["NVFP4"] = {
        "predicted_dloss": 0.001,
        "cost_source": 17,
    }
    with pytest.raises(SchemaValidationError, match="cost_source"):
        validate_cost_payload(payload, "cost.pkl")


def test_cost_payload_validates_output_mse_measured_metadata():
    payload = _cost_payload()
    payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"]["NVFP4"] = {
        "output_mse": 0.0,
        "output_mse_measured": "false",
    }
    with pytest.raises(SchemaValidationError, match="output_mse_measured"):
        validate_cost_payload(payload, "cost.pkl")


def test_cost_payload_validates_optional_activation_arm_scores():
    payload = _cost_payload()
    entry = payload["costs"][
        "model.layers.0.mlp.experts.gate_up_proj"
    ]["NVFP4"]
    entry.update({
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "lattice": 0.25,
            "learned": 0.125,
        },
    })
    assert validate_cost_payload(payload, "cost.pkl") is payload

    entry["activation_output_mse_by_codebook_source"] = {"unknown": 0.5}
    with pytest.raises(SchemaValidationError, match="unknown codebook source"):
        validate_cost_payload(payload, "cost.pkl")


def test_layer_config_rejects_malformed_dict_entry():
    payload = _layer_config_payload()
    payload["model.layers.0.self_attn.q_proj"] = {"bits": 4}
    with pytest.raises(SchemaValidationError, match="data_type"):
        validate_layer_config_payload(payload, "layer.json")
