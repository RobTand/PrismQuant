"""Focused tests for the shared resident/streaming CB config emitter."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from prismaquant.cb_export_config import (
    build_cb_scheme,
    build_quant_config,
    cb_scheme_reuse_signature,
    codebook_tensors,
)
from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.format_registry import format_is_producer_eligible


def _config_inputs():
    q_product = "model.layers.0.self_attn.q_proj"
    q_second = "model.layers.0.self_attn.k_proj"
    q_fp8 = "model.layers.0.self_attn.v_proj"
    q_stock = "model.layers.0.self_attn.o_proj"
    q_source = "model.layers.0.mlp.gate_proj"
    cb_targets = {
        q_product: ("fp4", "product", 12),
        q_second: ("fp4", "product", 13),
        q_fp8: ("fp8", "product", 40),
    }
    by_group = {
        ("lattice", "NVFP4_CB_K12"): [q_product],
        ("k_proj", "NVFP4_CB_K13"): [q_second],
        ("lattice", "FP8_CB_K40"): [q_fp8],
    }
    codebooks = {
        ("lattice", "NVFP4_CB_K12"): (
            torch.zeros(64, 4),
            torch.zeros(64, 4),
        ),
        ("k_proj", "NVFP4_CB_K13"): (
            torch.zeros(128, 4),
            torch.zeros(64, 4),
        ),
        ("lattice", "FP8_CB_K40"): tuple(
            torch.zeros(1024, 2) for _ in range(4)
        ),
    }
    blobs = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in codebook_tensors(ref, fmt, codebook).items()
    }
    assignment = {
        q_product: "NVFP4_CB_K12",
        q_second: "NVFP4_CB_K13",
        q_fp8: "FP8_CB_K40",
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
            q_second: torch.tensor([2.0]),
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
        "NVFP4_CB_K13",
        "FP8_CB_K40",
    }
    assert resident["layout_version"] == streaming["layout_version"] == 2
    assert "streaming" not in resident["provenance"]
    assert streaming["provenance"]["streaming"] is True
    assert "tensor_formats" in resident["provenance"]
    assert "tensor_formats" not in streaming["provenance"]


def test_cb_group_target_names_emit_distinct_logical_moe_role_groups():
    inputs = _config_inputs()
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    fmt = "FP8_CB_K40"
    gate_key = ("layer0-gate", fmt)
    up_key = ("layer0-up", fmt)
    inputs["assignment"] = {packed: fmt}
    inputs["cb_targets"] = {packed: ("fp8", "product", 40)}
    inputs["by_group"] = {
        gate_key: [packed],
        up_key: [packed],
    }
    inputs["codebooks"] = {
        gate_key: tuple(torch.zeros(1024, 2) for _ in range(4)),
        up_key: tuple(torch.ones(1024, 2) for _ in range(4)),
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
        ({("missing", "FP8_CB_K40"): ["logical"]}, "absent from by_group"),
        ({("lattice", "FP8_CB_K40"): []}, "must be nonempty"),
        (
            {("lattice", "FP8_CB_K40"): ["logical", "logical"]},
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


@pytest.mark.parametrize(
    ("k", "shapes", "type_size"),
    [
        (12, ((64, 4), (64, 4)), 57),
        (24, ((4096, 4), (4096, 4)), 105),
    ],
)
def test_nvfp4_endpoint_schemes_have_exact_geometry(k, shapes, type_size):
    fmt = f"NVFP4_CB_K{k}"
    codebook = tuple(torch.zeros(shape) for shape in shapes)
    scheme = build_cb_scheme(
        ref="lattice",
        fmt=fmt,
        grid="fp4",
        mode="product",
        k=k,
        codebook=codebook,
        scale_coding="two_tier",
    )
    assert scheme["k"] == k
    assert scheme["n_sub"] == 2
    assert scheme["type_size"] == type_size
    assert scheme["codebook_source"] == "lattice"
    assert scheme["codebook_ref"] == [
        f"cb_codebook.lattice.{fmt}.sub0",
        f"cb_codebook.lattice.{fmt}.sub1",
    ]


@pytest.mark.parametrize("k", [0, 33])
def test_nvfp4_formats_outside_endpoint_domain_cannot_enter_scheme(k):
    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(
            ref="lattice",
            fmt=f"NVFP4_CB_K{k}",
            grid="fp4",
            mode="product",
            k=k,
            codebook=(),
            scale_coding="two_tier",
        )


def test_reader_only_fp8_rung_cannot_enter_a_new_scheme():
    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(
            ref="lattice",
            fmt="FP8_CB_K29",
            grid="fp8",
            mode="product",
            k=29,
            codebook=(),
            scale_coding="v1",
        )


def test_routed_learned_bank_exception_is_explicit_and_narrow():
    codebook = tuple(torch.zeros(128, 2) for _ in range(4))
    base = {
        "ref": "model.layers.0.mlp.experts.gate_proj",
        "fmt": "FP8_CB_K28",
        "grid": "fp8",
        "mode": "product",
        "k": 28,
        "codebook": codebook,
        "scale_coding": "v1",
        "codebook_source": "learned",
    }
    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(**base)
    scheme = build_cb_scheme(**base, routed_learned_bank=True)
    assert scheme["k"] == 28
    assert scheme["codebook_source"] == "learned"

    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(
            **{**base, "fmt": "FP8_CB_K36", "k": 36},
            routed_learned_bank=True,
        )


@pytest.mark.parametrize("k", [28, 36])
def test_legacy_w8a16_scheme_exception_is_explicit_learned_only(k):
    fmt = f"FP8_CB_K{k}"
    codebook = tuple(torch.zeros(2 ** (k // 4), 2) for _ in range(4))
    base = {
        "ref": f"sealed-k{k}",
        "fmt": fmt,
        "grid": "fp8",
        "mode": "product",
        "k": k,
        "codebook": codebook,
        "scale_coding": "v1",
        "codebook_source": "learned",
    }

    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(**base)
    scheme = build_cb_scheme(**base, legacy_w8a16_compatibility=True)
    assert scheme["k"] == k
    assert scheme["codebook_source"] == "learned"
    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(
            **{**base, "codebook_source": "lattice"},
            legacy_w8a16_compatibility=True,
        )


def test_legacy_w8a16_config_stamps_only_exact_exception_groups():
    inputs = _config_inputs()
    k28_qname = "model.layers.18.mlp.experts.down_proj"
    k36_qname = "model.layers.0.self_attn.wq_b"
    k28_key = (k28_qname, "FP8_CB_K28")
    k36_key = (k36_qname, "FP8_CB_K36")
    inputs["assignment"] = {
        k28_qname: "FP8_CB_K28",
        k36_qname: "FP8_CB_K36",
    }
    inputs["cb_targets"] = {
        k28_qname: ("fp8", "product", 28),
        k36_qname: ("fp8", "product", 36),
    }
    inputs["source_targets"] = []
    inputs["stock_targets"] = {}
    inputs["by_group"] = {k28_key: [k28_qname], k36_key: [k36_qname]}
    inputs["codebooks"] = {
        k28_key: tuple(torch.zeros(128, 2) for _ in range(4)),
        k36_key: tuple(torch.zeros(512, 2) for _ in range(4)),
    }
    inputs["codebook_tensors_by_name"] = {
        name: tensor
        for (ref, fmt), codebook in inputs["codebooks"].items()
        for name, tensor in codebook_tensors(ref, fmt, codebook).items()
    }
    inputs["col_weights"] = {
        k28_qname: torch.ones(1),
        k36_qname: torch.ones(1),
    }
    inputs["serialization_context"] = CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="learned",
        codebook_source_by_format={
            "FP8_CB_K28": "learned",
            "FP8_CB_K36": "learned",
        },
    )
    stamp = {
        "schema": "prismaquant.dsv4_w8a16.legacy_compatibility.v1",
        "handoff_receipt_identity_sha256": "a" * 64,
        "publication_identity_sha256": "1" * 64,
        "assignment_sha256": "b" * 64,
        "layer_config_file_sha256": "c" * 64,
        "output_path": "/sealed/output",
        "source_identity_file_sha256": "2" * 64,
        "source_content_sha256": "3" * 64,
        "source_model_identity": {"content_sha256": "3" * 64},
        "codebook_bundle_file_sha256": "4" * 64,
        "codebook_bundle_content_sha256": "5" * 64,
        "runtime_pin_sha256": "6" * 64,
        "runtime_closure_identity_sha256": "7" * 64,
        "col_weights_content_sha256": "d" * 64,
        "exception_map": {
            "routed_fp8_cb_k28": {
                "format": "FP8_CB_K28",
                "count": 6144,
                "qnames_sha256": "8" * 64,
            },
            "dense_fp8_cb_k36": {
                "format": "FP8_CB_K36",
                "count": 3,
                "qnames_sha256": "9" * 64,
            },
        },
    }
    stamp["identity_sha256"] = hashlib.sha256(json.dumps(
        stamp,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()

    config = build_quant_config(
        **inputs,
        legacy_w8a16_compatibility_groups={k28_key, k36_key},
        legacy_w8a16_compatibility=stamp,
    )

    assert config["legacy_w8a16_compatibility"] == stamp
    assert config["provenance"]["legacy_w8a16_compatibility"] == stamp
    assert {scheme["k"] for scheme in _schemes(config).values()} == {28, 36}


def test_legacy_w8a16_representative_config_covers_exact_historical_mix():
    source = "model.layers.2.self_attn.wq_a"
    cb_formats = {
        "model.layers.18.mlp.experts.down_proj": "FP8_CB_K28",
        "model.layers.0.self_attn.wq_b": "FP8_CB_K36",
        "model.layers.3.self_attn.wq_b": "FP8_CB_K44",
        "model.layers.4.self_attn.wq_b": "FP8_CB_K48",
        "model.layers.5.mlp.shared_experts.down_proj": "NVFP4_CB_K16",
        "model.layers.6.mlp.shared_experts.down_proj": "NVFP4_CB_K18",
    }
    assignment = {source: "FP8_BLOCK_UE8M0_SOURCE", **cb_formats}
    widths = {
        "FP8_CB_K28": (128, 2, 4),
        "FP8_CB_K36": (512, 2, 4),
        "FP8_CB_K44": (2048, 2, 4),
        "FP8_CB_K48": (4096, 2, 4),
        "NVFP4_CB_K16": (256, 4, 2),
        "NVFP4_CB_K18": (512, 4, 2),
    }
    sources = {
        "FP8_CB_K28": "learned",
        "FP8_CB_K36": "learned",
        "FP8_CB_K44": "learned",
        "FP8_CB_K48": "lattice",
        "NVFP4_CB_K16": "lattice",
        "NVFP4_CB_K18": "lattice",
    }
    by_group = {(qname, fmt): [qname] for qname, fmt in cb_formats.items()}
    codebooks = {
        (qname, fmt): tuple(
            torch.zeros(rows, columns) for _ in range(subtables)
        )
        for qname, fmt in cb_formats.items()
        for rows, columns, subtables in (widths[fmt],)
    }
    codebook_blobs = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in codebook_tensors(ref, fmt, codebook).items()
    }
    legacy_groups = {
        (qname, fmt)
        for qname, fmt in cb_formats.items()
        if fmt in {"FP8_CB_K28", "FP8_CB_K36"}
    }
    stamp = {
        "schema": "prismaquant.dsv4_w8a16.legacy_compatibility.v1",
        "handoff_receipt_identity_sha256": "a" * 64,
        "publication_identity_sha256": "1" * 64,
        "assignment_sha256": "b" * 64,
        "layer_config_file_sha256": "c" * 64,
        "output_path": "/sealed/output",
        "source_identity_file_sha256": "2" * 64,
        "source_content_sha256": "3" * 64,
        "source_model_identity": {"content_sha256": "3" * 64},
        "codebook_bundle_file_sha256": "4" * 64,
        "codebook_bundle_content_sha256": "5" * 64,
        "runtime_pin_sha256": "6" * 64,
        "runtime_closure_identity_sha256": "7" * 64,
        "col_weights_content_sha256": "d" * 64,
        "exception_map": {
            "routed_fp8_cb_k28": {
                "format": "FP8_CB_K28",
                "count": 6144,
                "qnames_sha256": "8" * 64,
            },
            "dense_fp8_cb_k36": {
                "format": "FP8_CB_K36",
                "count": 3,
                "qnames_sha256": "9" * 64,
            },
        },
    }
    stamp["identity_sha256"] = hashlib.sha256(json.dumps(
        stamp,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()

    config = build_quant_config(
        assignment=assignment,
        cb_targets={
            qname: (
                "fp8" if fmt.startswith("FP8_") else "fp4",
                "product",
                int(fmt.rsplit("K", 1)[1]),
            )
            for qname, fmt in cb_formats.items()
        },
        source_targets=[],
        native_source_targets={source: "FP8_BLOCK_UE8M0_SOURCE"},
        stock_targets={},
        by_group=by_group,
        codebooks=codebooks,
        col_weights={qname: torch.ones(1) for qname in cb_formats},
        codebook_tensors_by_name=codebook_blobs,
        ignore=[],
        codebook_file="cb_codebooks.pqcb",
        scale_coding="two_tier",
        codebook_source="learned",
        serialized_payload_summary={"total_bytes": 123},
        serialization_context=CBSerializationContext(
            scale_coding="two_tier",
            codebook_source="learned",
            codebook_source_scope="fp8",
            codebook_source_by_format=sources,
        ),
        cb_render_identity={"schema": "test.render.v1"},
        git_commit="0123456789abcdef",
        source_passthrough_units={source: "FP8_BLOCK_UE8M0_SOURCE"},
        legacy_w8a16_compatibility_groups=legacy_groups,
        legacy_w8a16_compatibility=stamp,
    )

    assert set(_schemes(config)) == set(cb_formats.values())
    assert config["source_passthrough"] == {
        "version": 1,
        "units": {source: "fp8_e4m3_ue8m0_block128"},
    }
    assert config["legacy_w8a16_compatibility"] == stamp
    assert config["provenance"]["legacy_w8a16_compatibility"] == stamp
    assert not format_is_producer_eligible("FP8_CB_K28")
    assert not format_is_producer_eligible("FP8_CB_K36")
    assert format_is_producer_eligible("FP8_CB_K44")
    assert format_is_producer_eligible("FP8_CB_K48")


def test_legacy_w8a16_config_refuses_nonexception_groups():
    with pytest.raises(ValueError, match="only exact FP8_CB_K28/K36"):
        build_quant_config(
            **_config_inputs(),
            legacy_w8a16_compatibility_groups={
                ("lattice", "FP8_CB_K40")
            },
        )


@pytest.mark.parametrize("k", [26, 32])
def test_unsupported_nvfp4_rung_cannot_enter_a_new_scheme(k):
    with pytest.raises(ValueError, match="CB producer format/fields disagree"):
        build_cb_scheme(
            ref="lattice",
            fmt=f"NVFP4_CB_K{k}",
            grid="fp4",
            mode="product",
            k=k,
            codebook=(),
            scale_coding="two_tier",
        )


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
