"""End-to-end streaming producer contract for a quantized DSpark sidecar."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.artifact_completeness import assert_artifact_complete
from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming
import prismaquant.export_nvfp4_cb_streaming as streaming_exporter
import scripts.build_dsv4_dspark_cb_sidecar_inputs as input_builder
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    assignment_serialization_sha256,
)
from prismaquant.production_weight_cache import (
    CBRenderSourceIdentityCollector,
    _canonical_cb_col_weights_identity,
    _combined_source_weights_sha256,
    build_production_cache_cb_render_identity,
    validate_cb_render_identity_metadata,
)
from prismaquant.validate_cb_endpoint import (
    validate_dspark_production_render_attestation,
)


_BODY_LAYERS = 3
_STAGES = 3
_EXPERTS = 2
_WIDTH = 256

# This module builds synthetic CB bodies on CPU and never serves them.
# Gridbook 0.9.1's v12 table names no CB cell on sm_121, so the route gate
# refuses these exports unless the artifact declares what it is.  See
# tests/cb_synthetic_target.py; the real sm_121 refusal stays asserted in
# tests/test_cb_route_status_gate.py.
pytestmark = pytest.mark.usefixtures("synthetic_cb_target")



def _e8m0_ones(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.full(shape, 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )


def _source_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": _WIDTH,
        "num_hidden_layers": _BODY_LAYERS,
        "num_attention_heads": 1,
        "head_dim": _WIDTH,
        "q_lora_rank": _WIDTH,
        "o_groups": 1,
        "o_lora_rank": _WIDTH,
        "moe_intermediate_size": _WIDTH,
        "n_routed_experts": _EXPERTS,
        "n_shared_experts": 1,
        "vocab_size": _WIDTH,
        "expert_dtype": "fp4",
        "dspark_block_size": 5,
        "dspark_markov_rank": 8,
        "dspark_target_layer_ids": [0, 1, 2],
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _source_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}

    def fp8(base: str, shape: tuple[int, int]) -> None:
        tensors[base + ".weight"] = torch.ones(shape).to(
            torch.float8_e4m3fn
        )
        tensors[base + ".scale"] = _e8m0_ones(tuple(
            -(-dimension // 128) for dimension in shape
        ))

    def mxfp4(base: str, shape: tuple[int, int]) -> None:
        out_features, in_features = shape
        tensors[base + ".weight"] = torch.zeros(
            out_features, in_features // 2, dtype=torch.int8
        )
        tensors[base + ".scale"] = _e8m0_ones(
            (out_features, in_features // 32)
        )

    for stage in range(_STAGES):
        prefix = f"mtp.{stage}."
        for rest in (
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ):
            fp8(prefix + rest, (_WIDTH, _WIDTH))
        for expert in range(_EXPERTS):
            for leaf in ("w1", "w2", "w3"):
                mxfp4(
                    f"{prefix}ffn.experts.{expert}.{leaf}",
                    (_WIDTH, _WIDTH),
                )

        for rest in (
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ):
            tensors[prefix + rest] = torch.ones(
                _WIDTH, dtype=torch.bfloat16
            )
        tensors[prefix + "ffn.gate.weight"] = torch.ones(
            _EXPERTS, _WIDTH, dtype=torch.bfloat16
        )
        tensors[prefix + "ffn.gate.bias"] = torch.ones(
            _EXPERTS, dtype=torch.float32
        )
        tensors[prefix + "attn.attn_sink"] = torch.ones(
            1, dtype=torch.float32
        )
        for branch in ("attn", "ffn"):
            tensors[prefix + f"hc_{branch}_fn"] = torch.ones(
                24, 4 * _WIDTH, dtype=torch.float32
            )
            tensors[prefix + f"hc_{branch}_base"] = torch.ones(
                24, dtype=torch.float32
            )
            tensors[prefix + f"hc_{branch}_scale"] = torch.ones(
                3, dtype=torch.float32
            )

    fp8("mtp.0.main_proj", (_WIDTH, 3 * _WIDTH))
    tensors["mtp.0.main_norm.weight"] = torch.ones(
        _WIDTH, dtype=torch.bfloat16
    )
    tensors["mtp.2.norm.weight"] = torch.ones(
        _WIDTH, dtype=torch.bfloat16
    )
    tensors["mtp.2.confidence_head.proj.weight"] = torch.ones(
        1, _WIDTH + 8, dtype=torch.bfloat16
    )
    for leaf in ("markov_w1", "markov_w2"):
        tensors[f"mtp.2.markov_head.{leaf}.weight"] = torch.ones(
            _WIDTH, 8, dtype=torch.bfloat16
        )
    tensors["mtp.2.hc_head_fn"] = torch.ones(
        4, 4 * _WIDTH, dtype=torch.float32
    )
    tensors["mtp.2.hc_head_base"] = torch.ones(4, dtype=torch.float32)
    tensors["mtp.2.hc_head_scale"] = torch.ones(1, dtype=torch.float32)
    return tensors


def _recipe() -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    assignment: dict[str, str] = {}
    col_weights: dict[str, torch.Tensor] = {}
    for stage in range(_STAGES):
        prefix = f"mtp.{stage}."
        for rest in (
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ):
            qname = prefix + rest
            assignment[qname] = "NVFP4_CB_K12"
            col_weights[qname] = torch.ones(_WIDTH)
        assignment[prefix + "attn.wo_a"] = "FP8_BLOCK_UE8M0_SOURCE"
        for expert in range(_EXPERTS):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                qname = f"{prefix}ffn.experts.{expert}.{projection}"
                assignment[qname] = "NVFP4_CB_K12"
                col_weights[qname] = torch.ones(_WIDTH)
    return assignment, col_weights


def _source_identity(tensor_count: int) -> dict:
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "content_sha256": "a" * 64,
        "resolved_commit": "source-revision",
        "checkpoint_shards": 1,
        "checkpoint_tensors": tensor_count,
    }


def _unanchor(target: str) -> str:
    if target.startswith("re:^") and target.endswith("$"):
        return target[4:-1].replace("[.]", ".")
    return target


def test_streaming_export_builds_self_contained_k12_dspark_sidecar(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    source_tensors = _source_tensors()
    save_file(source_tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(_source_config()))
    source_identity = _source_identity(len(source_tensors))
    inputs = tmp_path / "inputs"
    input_builder.build(
        source,
        inputs,
        "NVFP4_CB_K12",
        source_model_identity=source_identity,
        serialization_context=CBSerializationContext.production(),
    )
    recipe_path = inputs / "dspark_layer_config.json"
    recipe_metadata = json.loads(recipe_path.read_text())["__prismaquant__"]
    with (inputs / "dspark_col_weights.pkl").open("rb") as handle:
        col_weights = pickle.load(handle)
    monkeypatch.setattr(
        streaming_exporter,
        "_source_model_identity_from_env",
        lambda _source: dict(source_identity),
    )
    artifact = tmp_path / "draft"

    counts = export_nvfp4_cb_streaming(
        source,
        recipe_path,
        artifact,
        col_weights,
        shared_codebook_spec={"source": "lattice"},
        device="cpu",
        scale_sweep=True,
        subset_prefixes=["mtp."],
        dspark_cb_sidecar=True,
    )

    config = json.loads((artifact / "config.json").read_text())
    quant = json.loads((artifact / "quant_config.json").read_text())
    emitted = load_file(str(artifact / "model.safetensors"))
    assert counts["NVFP4_CB_K12"] == 27
    assert config["n_mtp_layers"] == 3
    assert config["quantization_config"]["quant_method"] == "gridbook"
    assert "dspark_source_overlay" not in quant["provenance"]
    assert quant["provenance"]["dspark_cb_sidecar"]["schema"] == (
        "prismaquant.dspark_cb_sidecar.v1"
    )
    assert quant["provenance"]["render_identity_verified"] is True
    assert quant["provenance"]["cb_render_identity"][
        "source_weights_complete"
    ] is True
    assert len(quant["provenance"]["cb_render_identity"][
        "source_weights_content_sha256"
    ]) == 39
    assert quant["provenance"]["dspark_render_attestation"][
        "source_weights_entries"
    ] == 39
    render_attestation = quant["provenance"]["dspark_render_attestation"]
    assert render_attestation["recipe"] == recipe_metadata[
        "dspark_render_recipe"
    ]
    assert render_attestation["recipe_sha256"] == hashlib.sha256(
        json.dumps(
            render_attestation["recipe"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert quant["provenance"]["source_model_identity"] == source_identity
    production_attestation = validate_dspark_production_render_attestation(
        quant
    )
    assert production_attestation["source_weights_entries"] == 39
    unstamped = json.loads(json.dumps(quant))
    unstamped["provenance"]["render_identity_verified"] = False
    with pytest.raises(RuntimeError, match="does not attest a verified render"):
        validate_dspark_production_render_attestation(unstamped)

    # Adversarially shrink every self-attested scope and recompute every
    # dependent digest.  The render identity, source closure, recipe, and
    # attestation are internally valid after this mutation; only the new
    # cross-authority equality with finalized tensor_formats exposes that one
    # serialized CB member has been erased.
    reduced = deepcopy(quant)
    reduced_provenance = reduced["provenance"]
    reduced_identity = reduced_provenance["cb_render_identity"]
    victim = reduced_identity["col_weights_qnames"][0]
    reduced_qnames = [
        qname
        for qname in reduced_identity["col_weights_qnames"]
        if qname != victim
    ]
    reduced_col_weights = {
        qname: col_weights[qname] for qname in reduced_qnames
    }
    col_digest, col_shapes, col_content = (
        _canonical_cb_col_weights_identity(
            reduced_col_weights, reduced_qnames
        )
    )
    reduced_identity["col_weights_qnames"] = reduced_qnames
    reduced_identity["col_weights_entries"] = len(reduced_qnames)
    reduced_identity["col_weights_sha256"] = col_digest
    reduced_identity["col_weights_shapes"] = col_shapes
    reduced_identity["col_weights_content_sha256"] = col_content
    reduced_identity["cb_formats_by_qname"].pop(victim)
    reduced_identity["source_weights_shapes"].pop(victim)
    reduced_identity["source_weights_content_sha256"].pop(victim)
    reduced_identity["source_weights_sha256"] = (
        _combined_source_weights_sha256(
            reduced_identity["source_weights_shapes"],
            reduced_identity["source_weights_content_sha256"],
        )
    )
    validate_cb_render_identity_metadata(
        reduced_identity,
        require_source_complete=True,
        where="self-consistent reduced attack",
    )

    # Keep the finalized assignment authoritative and intact.  Recompute the
    # assignment digest anyway so the attack is self-consistent with every
    # field it is entitled to change; it still cannot redefine that scope.
    reduced_recipe = reduced_provenance[
        "dspark_render_attestation"
    ]["recipe"]
    recipe_assignment = dict(reduced_provenance["tensor_formats"])
    assert recipe_assignment.pop("mtp.0.main_proj") is not None
    reduced_recipe["assignment_sha256"] = (
        assignment_serialization_sha256(recipe_assignment)
    )
    reduced_recipe["col_weights_sha256"] = col_digest
    render_seed = deepcopy(reduced_identity)
    render_seed["source_weights_complete"] = False
    render_seed["source_weights_shapes"] = {}
    render_seed["source_weights_content_sha256"] = {}
    render_seed["source_weights_sha256"] = None
    reduced_recipe["render_identity_seed_sha256"] = hashlib.sha256(
        json.dumps(
            render_seed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    reduced_attestation = reduced_provenance["dspark_render_attestation"]
    reduced_attestation["recipe_sha256"] = hashlib.sha256(
        json.dumps(
            reduced_recipe, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    reduced_attestation["source_weights_sha256"] = reduced_identity[
        "source_weights_sha256"
    ]
    reduced_attestation["source_weights_entries"] = len(reduced_qnames)
    with pytest.raises(
        RuntimeError, match="finalized tensor_formats CB scope differs"
    ):
        validate_dspark_production_render_attestation(reduced)
    assert "dspark_target_bridge" not in quant  # weight-only experiment

    cb_targets = {
        target
        for group in quant["config_groups"].values()
        if "scheme" in group
        for target in group["targets"]
    }
    assert "model.layers.3.attn.wq_a" in cb_targets
    assert "model.layers.3.attn.wo_a" not in cb_targets
    assert "model.layers.4.ffn.experts.gate_up_proj" in cb_targets
    assert "model.layers.5.ffn.experts.down_proj" in cb_targets
    assert all(not target.startswith("mtp.") for target in cb_targets)

    assert "mtp.0.attn.wq_a.cb_qweight" in emitted
    assert "mtp.0.attn.wo_a.cb_qweight" not in emitted
    assert "mtp.1.ffn.experts.gate_up_proj.cb_qweight" in emitted
    assert "mtp.2.ffn.experts.down_proj.cb_qweight" in emitted
    assert "mtp.0.attn.wq_a.weight" not in emitted
    assert not any(
        name.startswith("mtp.0.ffn.experts.0.") for name in emitted
    )
    expected_source_mapping = {
        "mtp.0.main_proj": "model.main_proj",
        "mtp.0.attn.wo_a": "model.layers.3.attn.wo_a",
        "mtp.1.attn.wo_a": "model.layers.4.attn.wo_a",
        "mtp.2.attn.wo_a": "model.layers.5.attn.wo_a",
    }
    assert set(quant["source_passthrough"]["units"]) == set(
        expected_source_mapping.values()
    )
    sidecar = quant["provenance"]["dspark_cb_sidecar"]
    assert sidecar["physical_cb_targets"] == sorted(
        name.removesuffix(".cb_qweight")
        for name in emitted
        if name.endswith(".cb_qweight")
    )
    assert len(sidecar["physical_cb_targets"]) == 27
    assert sidecar["source_passthrough_targets"] == sorted(
        expected_source_mapping
    )
    assert sidecar["source_passthrough_physical_to_construction"] == dict(
        sorted(expected_source_mapping.items())
    )
    source_groups = [
        group for group in quant["config_groups"].values()
        if group.get("source_format") == "FP8_BLOCK_UE8M0_SOURCE"
    ]
    assert len(source_groups) == 1
    assert {_unanchor(target) for target in source_groups[0]["targets"]} == {
        *expected_source_mapping,
    }
    for base in expected_source_mapping:
        for suffix in ("weight", "scale"):
            name = f"{base}.{suffix}"
            assert torch.equal(
                emitted[name].view(torch.uint8),
                source_tensors[name].view(torch.uint8),
            )
    assert len(emitted) == 82  # 27 qweights + 47 glue + 8 source planes
    assert "mtp.2.markov_head.markov_w1.weight" in emitted
    assert_artifact_complete(artifact, verbatim_prefixes=())


def test_streaming_export_refuses_cb_wo_a_grouped_bmm(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    save_file(_source_tensors(), str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(_source_config()))
    assignment, col_weights = _recipe()
    assignment["mtp.1.attn.wo_a"] = "NVFP4_CB_K12"
    col_weights["mtp.1.attn.wo_a"] = torch.ones(_WIDTH)
    recipe_path = tmp_path / "assignment.json"
    recipe_path.write_text(json.dumps(assignment))

    with pytest.raises(ValueError, match="grouped-BMM wo_a"):
        export_nvfp4_cb_streaming(
            source,
            recipe_path,
            tmp_path / "draft",
            col_weights,
            shared_codebook_spec={"source": "lattice"},
            device="cpu",
            scale_sweep=True,
            subset_prefixes=["mtp."],
            allow_unstamped_research=True,
            dspark_cb_sidecar=True,
        )


def test_dense_source_attestation_observes_host_value_before_device_copy(
    monkeypatch,
):

    """A CUDA encode must not round-trip its source tensor just to hash it."""

    qname = "mtp.0.attn.wq_a"
    source = torch.arange(2 * _WIDTH, dtype=torch.float32).reshape(2, _WIDTH)
    col_weights = {qname: torch.ones(_WIDTH)}
    seed = build_production_cache_cb_render_identity(
        {qname: ["NVFP4_CB_K12"]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    collector = CBRenderSourceIdentityCollector(seed, where="host observation")

    # ``meta`` stands in for a non-host encode device.  The old ordering moved
    # ``source`` to meta and then tried to copy it back inside the digest,
    # raising "Cannot copy out of meta tensor".  The packer is irrelevant to
    # this ordering check, so return a tiny deterministic encoded payload.
    monkeypatch.setattr(
        streaming_exporter,
        "_pack_with_optional_warm_state",
        lambda *_args, **_kwargs: (
            torch.zeros((2, 1), dtype=torch.uint8),
            {"scales": torch.ones(2)},
        ),
    )
    verified: set[str] = set()
    packed, scale = streaming_exporter._encode_prefetched_cb_tensor(
        source,
        qname=qname,
        grid="fp4",
        mode="product",
        k=12,
        codebook=torch.ones(1),
        cw=col_weights[qname],
        scale_sweep=True,
        coding="two_tier",
        device="meta",
        encode_tier="balanced",
        cb_render_identity=seed,
        cb_render_source_collector=collector,
        verified_source_qnames=verified,
        format_name="NVFP4_CB_K12",
    )

    completed = collector.finalize()
    assert completed["source_weights_complete"] is True
    assert completed["source_weights_shapes"] == {qname: [2, _WIDTH]}
    assert verified == {qname}
    assert packed.device.type == "cpu"
    assert scale is None
