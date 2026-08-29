"""Fail-closed source/route/header census tests for the RTX4090 artifact."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.rtx4090_artifact_census import (
    RTX4090ArtifactCensusError,
    _canonical_digest,
    _canonical_aura_configs_from_source_config,
    _expected_artifact_manifest,
    _reconcile_routes_and_ignore,
    _validate_codebook_sidecar,
    expected_qwen38_source_layout,
    preflight_rtx4090_source_census,
    scan_indexed_safetensors,
    validate_rtx4090_finalized_artifact_census,
    validate_rtx4090_serialized_tensor_manifest,
)
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_QWEN38_LAYER_TYPES,
    validate_qwen38_dense_config,
)
from prismaquant.nvfp4_cb_footprint import (
    _safetensors_tensor_payload_sha256,
)
from prismaquant.shard_layout import tensor_payload_identity
from prismaquant.shipcard import build_weight_content_manifest


def _official_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "head_dim": 256,
            "num_key_value_heads": 4,
            "num_attention_heads": 24,
            "max_position_embeddings": 262144,
            "layer_types": list(RTX4090_QWEN38_LAYER_TYPES),
            "tie_word_embeddings": False,
            "attention_bias": False,
            "attn_output_gate": True,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def _small_source_census():
    source = {
        "lm_head.weight": {"dtype": "BF16", "shape": [16, 8]},
        "model.language_model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [16, 8],
        },
        "model.language_model.layers.0.input_layernorm.weight": {
            "dtype": "BF16",
            "shape": [8],
        },
        "model.language_model.layers.0.mlp.down_proj.weight": {
            "dtype": "BF16",
            "shape": [8, 256],
        },
    }
    linears = {
        "lm_head": "lm_head.weight",
        "model.layers.0.mlp.down_proj": (
            "model.language_model.layers.0.mlp.down_proj.weight"
        ),
    }
    return source, linears


def test_closed_official_layout_is_the_exact_released_checkpoint_census():
    config = _official_config()
    validated = validate_qwen38_dense_config(config)
    manifest, linears = expected_qwen38_source_layout(
        config, validated, where="test source"
    )

    assert len(manifest) == 1199
    assert len(linears) == 615
    assert linears["lm_head"] == "lm_head.weight"
    assert linears["model.layers.0.linear_attn.in_proj_qkv"] == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert linears["model.visual.blocks.26.attn.qkv"] == (
        "model.visual.blocks.26.attn.qkv.weight"
    )
    assert linears["mtp.fc"] == "mtp.fc.weight"

    quantized_source = deepcopy(config)
    quantized_source["quantization_config"] = {"quant_method": "nvfp4"}
    with pytest.raises(RTX4090ArtifactCensusError, match="must not carry"):
        expected_qwen38_source_layout(
            quantized_source, validated, where="test source"
        )


def test_aura_staged_config_digest_input_is_wrapper_canonical_and_stable():
    staged, normalized = _canonical_aura_configs_from_source_config(
        _official_config(), where="test AURA config"
    )
    assert staged["model_type"] == "qwen3_5_text"
    assert staged["architectures"] == ["Qwen3_5ForCausalLM"]
    assert staged["hidden_size"] == 5120
    assert "text_config" not in staged
    assert "vision_config" not in staged
    assert normalized["model_type"] == "qwen3_5_text"
    assert "_name_or_path" not in normalized


def test_source_preflight_requires_complete_linear_assignment_and_identity_map(
    monkeypatch, tmp_path
):
    config = _official_config()
    validated = validate_qwen38_dense_config(config)
    manifest, linears = expected_qwen38_source_layout(
        config, validated, where="test source"
    )
    checkpoint_map = {name: "model.safetensors" for name in manifest}
    current_aura_config = deepcopy(config)
    identity = {
        "config": deepcopy(config),
        "checkpoint_weight_map": checkpoint_map,
    }
    monkeypatch.setenv(
        "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE",
        str(tmp_path / "streamed_model_identity.json"),
    )
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census.validate_cached_streamed_model_identity",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census.compact_streamed_model_identity",
        lambda *_args, **_kwargs: {
            "schema": "prismaquant.streamed_model.identity.v1",
            "content_sha256": "a" * 64,
            "resolved_commit": None,
            "checkpoint_shards": 1,
            "checkpoint_tensors": len(manifest),
        },
    )
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census.scan_indexed_safetensors",
        lambda *_args, **_kwargs: (manifest, checkpoint_map),
    )
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census."
        "_canonical_aura_configs_from_source_config",
        lambda *_args, **_kwargs: (
            deepcopy(current_aura_config),
            deepcopy(current_aura_config),
        ),
    )
    assignment = {qname: "BF16" for qname in linears}
    census = preflight_rtx4090_source_census(
        model_dir=tmp_path,
        config=config,
        validated_config=validated,
        assignment=assignment,
        validate_config=validate_qwen38_dense_config,
        where="test preflight",
    )
    assert census["source_tensor_count"] == 1199
    assert census["source_linear_count"] == 615
    assert census["source_model_identity"]["checkpoint_tensors"] == 1199

    identity["config"]["rope_parameters"] = {"rope_theta": 123.0}
    with pytest.raises(RTX4090ArtifactCensusError, match="config is stale"):
        preflight_rtx4090_source_census(
            model_dir=tmp_path,
            config=config,
            validated_config=validated,
            assignment=assignment,
            validate_config=validate_qwen38_dense_config,
            where="test preflight",
        )
    identity["config"].pop("rope_parameters")

    assignment.pop("model.visual.blocks.0.attn.qkv")
    with pytest.raises(RTX4090ArtifactCensusError, match="exact source Linear"):
        preflight_rtx4090_source_census(
            model_dir=tmp_path,
            config=config,
            validated_config=validated,
            assignment=assignment,
            validate_config=validate_qwen38_dense_config,
            where="test preflight",
        )


def test_config_groups_and_ignore_must_exactly_replay_tensor_formats():
    source, linears = _small_source_census()
    assignment = {
        "lm_head": "BF16",
        "model.layers.0.mlp.down_proj": "FP8_CB_K20",
    }
    quant_config = {
        "config_groups": {
            "group_0": {
                "format": "FP8_CB_K20",
                "targets": [
                    "model.language_model.layers.0.mlp.down_proj"
                ],
            }
        },
        "ignore": [
            "lm_head",
            "model.language_model.embed_tokens",
        ],
    }
    _reconcile_routes_and_ignore(
        quant_config,
        assignment=assignment,
        linears=linears,
        source_manifest=source,
        source_layout="official_wrapper",
        where="test manifest",
    )

    wrong_group = deepcopy(quant_config)
    wrong_group["config_groups"]["group_0"]["format"] = "FP8_CB_K24"
    with pytest.raises(RTX4090ArtifactCensusError, match="disagree"):
        _reconcile_routes_and_ignore(
            wrong_group,
            assignment=assignment,
            linears=linears,
            source_manifest=source,
            source_layout="official_wrapper",
            where="test manifest",
        )

    missing_ignore = deepcopy(quant_config)
    missing_ignore["ignore"].remove("lm_head")
    with pytest.raises(RTX4090ArtifactCensusError, match="ignore differs"):
        _reconcile_routes_and_ignore(
            missing_ignore,
            assignment=assignment,
            linears=linears,
            source_manifest=source,
            source_layout="official_wrapper",
            where="test manifest",
        )


def test_delegated_fp8_target_must_use_exact_vllm_wrapper_namespace():
    source, linears = _small_source_census()
    assignment = {
        "lm_head": "BF16",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
    }
    base = {
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": [
                    "re:^language_model[.]model[.]layers[.]0[.]mlp[.]down_proj$"
                ],
            }
        },
        "ignore": ["lm_head", "model.language_model.embed_tokens"],
    }
    _reconcile_routes_and_ignore(
        base,
        assignment=assignment,
        linears=linears,
        source_manifest=source,
        source_layout="official_wrapper",
        where="test delegated",
    )
    wildcard = deepcopy(base)
    wildcard["config_groups"]["group_0"]["targets"] = [
        "re:^language_model.*down_proj$"
    ]
    with pytest.raises(RTX4090ArtifactCensusError, match="wildcard"):
        _reconcile_routes_and_ignore(
            wildcard,
            assignment=assignment,
            linears=linears,
            source_manifest=source,
            source_layout="official_wrapper",
            where="test delegated",
        )


def test_final_tensor_manifest_rejects_hidden_fp4_or_unclaimed_planes():
    source, linears = _small_source_census()
    assignment = {
        "lm_head": "BF16",
        "model.layers.0.mlp.down_proj": "FP8_CB_K20",
    }
    expected = _expected_artifact_manifest(source, linears, assignment)
    assert expected[
        "model.language_model.layers.0.mlp.down_proj.cb_qweight"
    ] == {"dtype": "U8", "shape": [8, 80]}
    assert "model.language_model.layers.0.mlp.down_proj.weight" not in expected

    observed = deepcopy(expected)
    observed[
        "model.language_model.layers.0.mlp.down_proj.input_global_scale"
    ] = {"dtype": "F32", "shape": [1]}
    with pytest.raises(RTX4090ArtifactCensusError, match="extra=.*input_global"):
        validate_rtx4090_serialized_tensor_manifest(
            observed, expected, where="test finalized"
        )


def test_safetensors_scan_rejects_unindexed_hidden_container(tmp_path):
    save_file({"weight": torch.zeros(1)}, tmp_path / "model.safetensors")
    save_file({"nvfp4_plane": torch.zeros(1)}, tmp_path / "hidden.safetensors")
    with pytest.raises(RTX4090ArtifactCensusError, match="files on disk"):
        scan_indexed_safetensors(tmp_path, where="test scan")

    (tmp_path / "hidden").mkdir()
    save_file(
        {"nvfp4_nested": torch.zeros(1)},
        tmp_path / "hidden" / "nested.safetensors",
    )
    with pytest.raises(RTX4090ArtifactCensusError, match="nested safetensors"):
        scan_indexed_safetensors(tmp_path, where="test nested scan")


def test_safetensors_scan_rejects_shape_span_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros(2, dtype=torch.float32)}, path)
    with path.open("r+b") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        raw_header = handle.read(header_length)
        assert b'"shape":[2]' in raw_header
        corrupted = raw_header.replace(b'"shape":[2]', b'"shape":[3]', 1)
        handle.seek(8)
        handle.write(corrupted)
    with pytest.raises(RTX4090ArtifactCensusError, match="cannot read safetensors"):
        scan_indexed_safetensors(tmp_path, where="test span mismatch")


def test_codebook_sidecar_is_exactly_the_referenced_fp16_lattice_set(tmp_path):
    refs = [f"lattice_k4_sub{index}" for index in range(4)]
    quant_config = {
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_0": {
                "format": "FP8_CB_K4",
                "scheme": {
                    "k": 4,
                    "mode": "product",
                    "n_sub": 4,
                    "codebook_ref": refs,
                },
            }
        },
    }
    tensors = {
        ref: torch.zeros((2, 2), dtype=torch.float16) for ref in refs
    }
    save_file(tensors, tmp_path / "cb_codebooks.pqcb")
    _validate_codebook_sidecar(
        tmp_path, quant_config, where="test sidecar"
    )

    tensors["hidden_nvfp4_table"] = torch.zeros(
        (2, 2), dtype=torch.float16
    )
    save_file(tensors, tmp_path / "cb_codebooks.pqcb")
    with pytest.raises(RTX4090ArtifactCensusError, match="differ from exact"):
        _validate_codebook_sidecar(
            tmp_path, quant_config, where="test sidecar"
        )


def test_final_census_replays_stable_staged_config_digest(
    tmp_path, monkeypatch
):
    source, linears = _small_source_census()
    assignment = {
        "lm_head": "BF16",
        "model.layers.0.mlp.down_proj": "FP8_CB_K20",
    }
    source_config = {"model_type": "synthetic_qwen38"}
    artifact_config = {
        **source_config,
        "quantization_config": {
            "quant_method": "gridbook",
            "format": "fp8_cb",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(artifact_config))
    expected_artifact = _expected_artifact_manifest(
        source, linears, assignment
    )
    tensors = {}
    dtype_by_name = {
        "BF16": torch.bfloat16,
        "U8": torch.uint8,
        "F32": torch.float32,
    }
    for name, descriptor in expected_artifact.items():
        tensors[name] = torch.zeros(
            tuple(descriptor["shape"]),
            dtype=dtype_by_name[descriptor["dtype"]],
        )
    save_file(tensors, tmp_path / "model.safetensors")

    refs = [f"lattice_k20_sub{index}" for index in range(4)]
    sidecar_shapes = ((32, 2), (32, 2), (32, 2), (32, 2))
    save_file(
        {
            ref: torch.zeros(shape, dtype=torch.float16)
            for ref, shape in zip(refs, sidecar_shapes, strict=True)
        },
        tmp_path / "cb_codebooks.pqcb",
    )
    codebook_digests = _safetensors_tensor_payload_sha256(
        tmp_path / "cb_codebooks.pqcb", refs
    )
    tensor_digests = _safetensors_tensor_payload_sha256(
        tmp_path / "model.safetensors", sorted(expected_artifact)
    )
    git_commit = "d" * 40
    (tmp_path / "shipcard.json").write_text(json.dumps({
        "build": {"git": {"commit": git_commit, "dirty": False}},
        "slots": {},
    }))
    staged_config = {"model_type": "qwen3_5_text", "hidden_size": 5120}
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census.expected_qwen38_source_layout",
        lambda *_args, **_kwargs: (source, linears),
    )
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census."
        "_canonical_aura_configs_from_source_config",
        lambda *_args, **_kwargs: (staged_config, staged_config),
    )
    validated = {
        "source_layout": "official_wrapper",
        "layer_types": [],
    }
    compact_identity = {
        "schema": "prismaquant.streamed_model.identity.v1",
        "content_sha256": "a" * 64,
        "resolved_commit": None,
        "checkpoint_shards": 1,
        "checkpoint_tensors": len(source),
    }
    census = {
        "schema": "prismaquant.rtx4090_qwen38_source_census.v1",
        "source_layout": "official_wrapper",
        "source_config_sha256": _canonical_digest(source_config),
        "aura_staged_config_sha256": _canonical_digest(staged_config),
        "aura_execution_config_sha256": "b" * 64,
        "source_tensor_manifest_sha256": _canonical_digest(source),
        "source_tensor_count": len(source),
        "source_linear_count": len(linears),
        "assignment_sha256": _canonical_digest(assignment),
        "source_model_identity": compact_identity,
    }
    quant_config = {
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_0": {
                "format": "FP8_CB_K20",
                "targets": [
                    "model.language_model.layers.0.mlp.down_proj"
                ],
                "scheme": {
                    "k": 20,
                    "mode": "product",
                    "n_sub": 4,
                    "codebook_ref": refs,
                },
            }
        },
        "ignore": ["lm_head", "model.language_model.embed_tokens"],
        "provenance": {
            "producer_policy": {"source_census": census},
            "source_model_identity": compact_identity,
            "git_commit": git_commit,
            "codebook_sha256": codebook_digests,
            "tensor_payload_identity": tensor_payload_identity(
                tensor_digests, include_tensor_sha256=True
            ),
            "weight_content_manifest": build_weight_content_manifest(
                tmp_path
            ),
            "artifact_inventory": {
                "cb_codebook_content_sha256": codebook_digests,
            },
        },
    }
    result = validate_rtx4090_finalized_artifact_census(
        artifact_dir=tmp_path,
        quant_config=quant_config,
        assignment=assignment,
        validate_config=lambda *_args, **_kwargs: validated,
        where="test integrated census",
    )
    assert result["artifact_tensors"] == len(expected_artifact)

    bad_codebook = deepcopy(quant_config)
    bad_codebook["provenance"]["codebook_sha256"][refs[0]] = "e" * 64
    with pytest.raises(RTX4090ArtifactCensusError, match="codebook_sha256"):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=bad_codebook,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test bad codebook digest",
        )

    bad_tensor_identity = deepcopy(quant_config)
    bad_tensor_identity["provenance"]["tensor_payload_identity"][
        "payload_sha256"
    ] = "e" * 64
    with pytest.raises(RTX4090ArtifactCensusError, match="payload count/hash"):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=bad_tensor_identity,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test bad tensor digest",
        )

    false_tensor_ledger = deepcopy(quant_config)
    false_tensor_digests = dict(tensor_digests)
    false_tensor_digests["lm_head.weight"] = "e" * 64
    false_tensor_ledger["provenance"]["tensor_payload_identity"] = (
        tensor_payload_identity(
            false_tensor_digests, include_tensor_sha256=True
        )
    )
    with pytest.raises(
        RTX4090ArtifactCensusError,
        match="tensor digest ledger differs from finalized bytes",
    ):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=false_tensor_ledger,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test self-consistent false tensor ledger",
        )

    bad_weight_manifest = deepcopy(quant_config)
    bad_weight_manifest["provenance"]["weight_content_manifest"][
        "future_field"
    ] = True
    with pytest.raises(RTX4090ArtifactCensusError, match="closed schema"):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=bad_weight_manifest,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test open weight manifest",
        )

    tampered_tensors = dict(tensors)
    tampered_tensors["lm_head.weight"] = torch.ones_like(
        tampered_tensors["lm_head.weight"]
    )
    save_file(tampered_tensors, tmp_path / "model.safetensors")
    with pytest.raises(
        RTX4090ArtifactCensusError,
        match=(
            "(?:tensor digest ledger|weight content manifest) "
            "differs from finalized bytes"
        ),
    ):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=quant_config,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test same-shape weight payload tamper",
        )
    save_file(tensors, tmp_path / "model.safetensors")

    bad_git = deepcopy(quant_config)
    bad_git["provenance"]["git_commit"] = "unknown"
    with pytest.raises(RTX4090ArtifactCensusError, match="full lowercase"):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=bad_git,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test bad git",
        )

    (tmp_path / "shipcard.json").write_text(json.dumps({
        "build": {"git": {"commit": git_commit, "dirty": True}},
        "slots": {},
    }))
    with pytest.raises(RTX4090ArtifactCensusError, match="exact matching clean"):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=quant_config,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test dirty git",
        )
    (tmp_path / "shipcard.json").write_text(json.dumps({
        "build": {"git": {"commit": git_commit, "dirty": False}},
        "slots": {},
    }))

    census["aura_staged_config_sha256"] = "c" * 64
    with pytest.raises(
        RTX4090ArtifactCensusError,
        match="aura_staged_config_sha256",
    ):
        validate_rtx4090_finalized_artifact_census(
            artifact_dir=tmp_path,
            quant_config=quant_config,
            assignment=assignment,
            validate_config=lambda *_args, **_kwargs: validated,
            where="test integrated census",
        )
