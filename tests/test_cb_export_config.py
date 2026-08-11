"""Focused tests for the shared resident/streaming CB config emitter."""
from __future__ import annotations

import pytest
import torch

from prismaquant.cb_export_config import (
    build_cb_scheme,
    build_quant_config,
    cb_scheme_reuse_signature,
    codebook_tensors,
)
from prismaquant.nvfp4_cb_footprint import CBSerializationContext


def _config_inputs():
    q_product = "model.layers.0.self_attn.q_proj"
    q_signed = "model.layers.0.self_attn.k_proj"
    q_fp8 = "model.layers.0.self_attn.v_proj"
    q_stock = "model.layers.0.self_attn.o_proj"
    q_source = "model.layers.0.mlp.gate_proj"
    cb_targets = {
        q_product: ("fp4", "product", 12),
        q_signed: ("fp4", "signed", 13),
        q_fp8: ("fp8", "product", 28),
    }
    by_group = {
        ("lattice", "NVFP4_CB_K12"): [q_product],
        ("k_proj", "NVFP4_CB_S13"): [q_signed],
        ("lattice", "FP8_CB_K28"): [q_fp8],
    }
    codebooks = {
        ("lattice", "NVFP4_CB_K12"): (
            torch.zeros(64, 4),
            torch.zeros(64, 4),
        ),
        ("k_proj", "NVFP4_CB_S13"): torch.zeros(32, 8),
        ("lattice", "FP8_CB_K28"): tuple(
            torch.zeros(128, 2) for _ in range(4)
        ),
    }
    blobs = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in codebook_tensors(ref, fmt, codebook).items()
    }
    assignment = {
        q_product: "NVFP4_CB_K12",
        q_signed: "NVFP4_CB_S13",
        q_fp8: "FP8_CB_K28",
        q_stock: "NVFP4",
        q_source: "FP8_SOURCE",
    }
    return {
        "assignment": assignment,
        "cb_targets": cb_targets,
        "source_targets": [q_source],
        "stock_targets": {q_stock: "NVFP4"},
        "by_group": by_group,
        "codebooks": codebooks,
        "col_weights": {
            q_product: torch.tensor([1.0]),
            q_signed: torch.tensor([2.0]),
            q_fp8: torch.tensor([3.0]),
        },
        "codebook_tensors_by_name": blobs,
        "ignore": ["model.embed_tokens"],
        "codebook_file": "cb_codebooks.pqcb",
        "scale_coding": "two_tier",
        "codebook_source": "lattice",
        "serialized_payload_summary": {"total_bytes": 123},
        "serialization_context": CBSerializationContext.production(),
        "cb_render_identity": {"schema": "test.render.v1"},
        "git_commit": "0123456789abcdef",
    }


def _schemes(config):
    return {
        group["format"]: group["scheme"]
        for group in config["config_groups"].values()
        if "scheme" in group
    }


def test_resident_and_streaming_emit_identical_schemes_for_same_inputs():
    inputs = _config_inputs()
    resident = build_quant_config(
        **inputs,
        streaming_provenance=None,
        include_tensor_formats=True,
    )
    streaming = build_quant_config(
        **inputs,
        streaming_provenance=True,
        include_tensor_formats=False,
    )

    assert resident["config_groups"] == streaming["config_groups"]
    assert _schemes(resident) == _schemes(streaming)
    assert set(_schemes(resident)) == {
        "NVFP4_CB_K12",
        "NVFP4_CB_S13",
        "FP8_CB_K28",
    }
    assert resident["layout_version"] == streaming["layout_version"] == 2
    assert "streaming" not in resident["provenance"]
    assert streaming["provenance"]["streaming"] is True
    assert "tensor_formats" in resident["provenance"]
    assert "tensor_formats" not in streaming["provenance"]


def test_cb_group_target_names_emit_distinct_logical_moe_role_groups():
    inputs = _config_inputs()
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    fmt = "FP8_CBL_K28"
    gate_key = ("layer0-gate", fmt)
    up_key = ("layer0-up", fmt)
    inputs["assignment"] = {packed: fmt}
    inputs["cb_targets"] = {packed: ("fp8", "product", 28)}
    inputs["by_group"] = {
        gate_key: [packed],
        up_key: [packed],
    }
    inputs["codebooks"] = {
        gate_key: tuple(torch.zeros(128, 2) for _ in range(4)),
        up_key: tuple(torch.ones(128, 2) for _ in range(4)),
    }
    inputs["codebook_tensors_by_name"] = {
        name: tensor
        for (ref, group_fmt), codebook in inputs["codebooks"].items()
        for name, tensor in codebook_tensors(ref, group_fmt, codebook).items()
    }
    inputs["col_weights"] = {packed: torch.tensor([1.0])}
    inputs["stock_targets"] = {}
    inputs["source_targets"] = []

    gate_target = "model.layers.0.mlp.experts.gate_proj"
    up_target = "model.layers.0.mlp.experts.up_proj"
    config = build_quant_config(
        **inputs,
        cb_group_target_names={
            gate_key: [gate_target],
            up_key: [up_target],
        },
        # Exact overrides bypass the default/custom serialization hook.
        cb_target_name=lambda qname: f"mapped.{qname}",
    )

    cb_groups = [
        group for group in config["config_groups"].values()
        if group.get("format") == fmt
    ]
    assert {tuple(group["targets"]) for group in cb_groups} == {
        (gate_target,),
        (up_target,),
    }
    scheme_by_target = {
        group["targets"][0]: group["scheme"] for group in cb_groups
    }
    assert scheme_by_target[gate_target]["codebook_group"] == "layer0-gate"
    assert scheme_by_target[up_target]["codebook_group"] == "layer0-up"
    assert (
        scheme_by_target[gate_target]["codebook_ref"]
        != scheme_by_target[up_target]["codebook_ref"]
    )


def test_cb_group_target_names_unset_preserves_legacy_config_bytes():
    inputs = _config_inputs()

    legacy = build_quant_config(**inputs)
    explicit_empty = build_quant_config(
        **inputs,
        cb_group_target_names={},
    )

    assert explicit_empty == legacy


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({("missing", "FP8_CB_K28"): ["logical"]}, "absent from by_group"),
        ({("lattice", "FP8_CB_K28"): []}, "must be nonempty"),
        (
            {("lattice", "FP8_CB_K28"): ["logical", "logical"]},
            "must contain unique targets",
        ),
    ],
)
def test_cb_group_target_names_refuse_invalid_overrides(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_quant_config(
            **_config_inputs(),
            cb_group_target_names=overrides,
        )


def test_delta_reuse_signature_comes_from_the_canonical_scheme():
    codebook = (torch.zeros(64, 4), torch.zeros(64, 4))
    scheme = build_cb_scheme(
        ref="lattice",
        fmt="NVFP4_CB_K12",
        grid="fp4",
        mode="product",
        k=12,
        codebook=codebook,
        scale_coding="two_tier",
    )

    assert cb_scheme_reuse_signature(scheme) == {
        "grid": "fp4",
        "mode": "product",
        "k": 12,
        "n_sub": 2,
        "type_size": 57,
        "codebook_ref": [
            "cb_codebook.lattice.NVFP4_CB_K12.sub0",
            "cb_codebook.lattice.NVFP4_CB_K12.sub1",
        ],
        "scale_coding": "two_tier",
    }


def test_scheme_rejects_noncanonical_sidecar_type_rank_and_shape():
    base = {
        "ref": "lattice",
        "fmt": "NVFP4_CB_K12",
        "grid": "fp4",
        "mode": "product",
        "k": 12,
        "scale_coding": "two_tier",
    }
    with pytest.raises(TypeError, match="subtable 1 must be a torch.Tensor"):
        build_cb_scheme(
            **base,
            codebook=(torch.zeros(64, 4), [[0.0]]),
        )
    with pytest.raises(ValueError, match="must have rank 2"):
        build_cb_scheme(
            **base,
            codebook=(torch.zeros(64, 4), torch.zeros(64, 4, 1)),
        )
    with pytest.raises(ValueError, match="canonical shape"):
        build_cb_scheme(
            **base,
            codebook=(torch.zeros(64, 4), torch.zeros(63, 4)),
        )


def test_weight_only_stock_policy_is_an_explicit_builder_input():
    inputs = _config_inputs()
    body = next(iter(inputs["stock_targets"]))
    visual = "model.visual.blocks.0.attn.proj"
    inputs["assignment"][visual] = "NVFP4"
    inputs["stock_targets"][visual] = "NVFP4"

    config = build_quant_config(
        **inputs,
        delegated_target_name=lambda qname: (
            qname[len("model."):] if qname == visual else qname
        ),
        weight_only_stock_targets={visual},
    )
    stock_groups = [
        group for group in config["config_groups"].values()
        if "scheme" not in group and group["format"] == "nvfp4-pack-quantized"
    ]
    assert len(stock_groups) == 2
    weight_only = next(
        group for group in stock_groups if group["input_activations"] is None
    )
    activated = next(
        group for group in stock_groups if group["input_activations"] is not None
    )
    assert weight_only["targets"] == [
        "re:^visual[.]blocks[.]0[.]attn[.]proj$"
    ]
    assert activated["targets"] == [
        "re:^model[.]layers[.]0[.]self_attn[.]o_proj$"
    ]
    assert body in inputs["stock_targets"]
