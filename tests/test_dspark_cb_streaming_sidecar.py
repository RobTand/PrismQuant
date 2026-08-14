"""End-to-end streaming producer contract for a quantized DSpark sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.artifact_completeness import assert_artifact_complete
from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming


_BODY_LAYERS = 3
_STAGES = 3
_EXPERTS = 2
_WIDTH = 256


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


def _unanchor(target: str) -> str:
    if target.startswith("re:^") and target.endswith("$"):
        return target[4:-1].replace("[.]", ".")
    return target


def test_streaming_export_builds_self_contained_k12_dspark_sidecar(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    source_tensors = _source_tensors()
    save_file(source_tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(_source_config()))
    assignment, col_weights = _recipe()
    recipe_path = tmp_path / "assignment.json"
    recipe_path.write_text(json.dumps(assignment))
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
        allow_unstamped_research=True,
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
