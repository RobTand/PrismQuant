from __future__ import annotations

import json

import pytest

from prismaquant.layer_config import (
    canonicalize_assignment,
    canonicalize_format,
    load_assignment,
    strip_weight,
)
from prismaquant.production_recache import _load_assignment as recache_load_assignment
from prismaquant.validate_assignments_kl import load_assignment_json


def test_canonicalize_format_accepts_supported_recipe_shapes():
    assert canonicalize_format("nvfp4") == "NVFP4"
    assert canonicalize_format("MXFP4") == "MXFP4"
    assert canonicalize_format("8") == "FP8_E4M3"
    assert canonicalize_format("fp8_dynamic") == "FP8_E4M3"
    assert canonicalize_format("fp8_e4m3") == "FP8_E4M3"
    assert canonicalize_format("fp8_e5m2") == "FP8_E5M2"
    assert canonicalize_format("mxfp8_e5m2") == "MXFP8_E5M2"
    assert canonicalize_format("bf16") == "BF16"
    assert canonicalize_format(4) == "NVFP4"
    assert canonicalize_format(8) == "FP8_E4M3"
    assert canonicalize_format(16) == "BF16"
    assert canonicalize_format({"data_type": "nv_fp", "bits": 4}) == "NVFP4"
    assert canonicalize_format({"data_type": "mx_fp", "bits": 4}) == "MXFP4"
    assert canonicalize_format({"data_type": "mx_fp", "bits": 8}) == "MXFP8_E4M3"
    assert (
        canonicalize_format({
            "data_type": "mx_fp",
            "bits": 8,
            "weight_element_dtype": "fp8_e5m2",
        })
        == "MXFP8_E5M2"
    )
    assert canonicalize_format({"data_type": "float", "bits": 16}) == "BF16"
    assert (
        canonicalize_format({"data_type": "fp8_e4m3", "bits": 8, "group_size": 0})
        == "FP8_E4M3"
    )
    assert (
        canonicalize_format({"data_type": "fp8_e4m3", "bits": 8, "group_size": 128})
        == "FP8_SOURCE"
    )


def test_canonicalize_format_rejects_unknown_formats():
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_format("not_a_real_format")
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_format({"data_type": "int", "bits": 3})


def test_canonicalize_assignment_strips_weight_suffix():
    assert strip_weight("model.layers.0.mlp.down_proj.weight") == (
        "model.layers.0.mlp.down_proj"
    )
    assert canonicalize_assignment(
        {
            "model.layers.0.mlp.down_proj.weight": "nvfp4",
            "model.layers.0.self_attn.o_proj": "bf16",
        }
    ) == {
        "model.layers.0.mlp.down_proj": "NVFP4",
        "model.layers.0.self_attn.o_proj": "BF16",
    }


def test_load_assignment_validates_and_is_shared_by_recache(tmp_path):
    path = tmp_path / "layer_config.json"
    path.write_text(
        json.dumps(
            {
                "layer.weight": {
                    "bits": 4,
                    "group_size": 16,
                    "data_type": "nv_fp",
                    "act_bits": 4,
                    "act_group_size": 16,
                    "act_data_type": "nv_fp",
                },
                "other": "bf16",
            }
        )
    )

    expected = {"layer": "NVFP4", "other": "BF16"}
    assert load_assignment(path) == expected
    assert recache_load_assignment(path) == expected


def test_assignment_loader_rejects_malformed_layer_config(tmp_path):
    path = tmp_path / "bad_layer_config.json"
    path.write_text(json.dumps({"layer": {"bits": 4}}))

    with pytest.raises(ValueError, match="data_type"):
        load_assignment(path)


def test_kl_assignment_loader_uses_shared_canonicalizer(tmp_path):
    base = {
        "model.layers.0.mlp.down_proj": "bf16",
        "model.layers.0.self_attn.o_proj": "nvfp4",
    }
    path = tmp_path / "solve_result.json"
    path.write_text(
        json.dumps(
            {
                "assignment": {
                    "model.layers.0.mlp.down_proj": {
                        "data_type": "fp8_e4m3",
                        "bits": 8,
                        "group_size": 0,
                    }
                }
            }
        )
    )

    assert load_assignment_json(path, base=base) == {
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
        "model.layers.0.self_attn.o_proj": "NVFP4",
    }
