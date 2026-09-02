import json

import pytest

from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_serialization_context_stamp,
    codebook_sidecar_payload_bytes,
)
from scripts import build_hy3_mtp_cb_inputs as hy3
from scripts import exp1_nvfp4_cb_0p6b as exp1


def test_hy3_auto_context_loader_rejects_unstamped_assignment(tmp_path):
    body = tmp_path / "body_layer_config.json"
    body.write_text(json.dumps({"model.layers.0.self_attn.q_proj": {}}))

    with pytest.raises(ValueError, match="explicit cb_serialized_payload"):
        hy3._load_body_cb_serialization_context(body)

    expected = CBSerializationContext.production()
    body.write_text(json.dumps({
        "__prismaquant__": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                expected,
                formats=("NVFP4_CB_K14", "FP8_CB_K28"),
            ),
        },
    }))
    assert hy3._load_body_cb_serialization_context(body) == expected


def test_hy3_auto_menu_uses_exact_row_scales_and_deduplicated_sidecar():
    context = CBSerializationContext.production()
    dense_shapes = {
        f"model.layers.80.dense_{index}": (4, 512)
        for index in range(7)
    }
    expert_shapes = {
        "model.layers.80.mlp.experts.gate_up_proj": (2, 8, 512),
        "model.layers.80.mlp.experts.down_proj": (2, 4, 512),
    }
    e_table = {
        f"K{k}": float(k)
        for k in (*hy3._EXPERT_FP4_KS, *hy3._EXPERT_FP8_KS)
    }

    menu, metadata = hy3._build_auto_menu_from_shapes(
        num_experts=2,
        dense_shapes=dense_shapes,
        expert_shapes=expert_shapes,
        e_table=e_table,
        cb_context=context,
    )
    record = metadata["pairing"]["K28"]
    assert record["expert_format"] == "FP8_CB_K28"
    assert record["dense_format"] == "FP8_CB_K28"
    assignment = {
        **{name: record["expert_format"] for name in expert_shapes},
        **{name: record["dense_format"] for name in dense_shapes},
    }
    expected = cb_assignment_payload_breakdown(
        assignment,
        {**dense_shapes, **expert_shapes},
        context=context,
    )

    assert record["resident_bytes"] == expected["total_bytes"]
    assert record["serialized_payload"]["fp8_row_scale_bytes"] > 0
    assert record["serialized_payload"]["fp8_row_scale_bytes"] == (
        expected["fp8_row_scale_bytes"]
    )
    assert record["serialized_payload"]["codebook_sidecar_bytes"] == (
        codebook_sidecar_payload_bytes("FP8_CB_K28")
    )
    assert len(expected["sidecars"]) == 1
    assert next(point for point in menu if point.name == "K28").resident_bytes == (
        expected["total_bytes"]
    )


def test_exp1c_analytic_footprint_has_no_nonexistent_global_scale():
    arm = exp1.Arm1c("CB16_v2", "cb_v2", k=16)
    targets = {
        "model.layers.0.self_attn.q_proj": (4, 256),
        "model.layers.1.self_attn.q_proj": (4, 256),
    }

    footprint = exp1.footprint_1c(arm, targets)

    assert footprint["global_scale_bytes"] == 0
    assert footprint["total_bytes"] == (
        footprint["body_bytes"] + footprint["sidecar_bytes"]
    )
    assert footprint["byte_scope"] == (
        "research_analytic_cb_body_plus_packed_fp4_role_codebooks"
    )
    assert footprint["production_exact"] is False
