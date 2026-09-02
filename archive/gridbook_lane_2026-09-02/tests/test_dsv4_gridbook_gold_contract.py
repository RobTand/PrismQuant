from __future__ import annotations

import json

import pytest

from prismaquant.gridbook_environment import (
    CANONICAL_GOLD_ENVIRONMENT,
    CANONICAL_GOLD_SET_ENVIRONMENT,
)
from tools.dsv4_gridbook_contract import exact_llm_contract


def _artifact(tmp_path, *, mxfp4: bool):
    root = tmp_path / "artifact"
    root.mkdir()
    payload = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "source_passthrough": {
            "version": 1,
            "units": {
                "model.layers.0.mlp.experts": (
                    "mxfp4_e2m1_ue8m0_g32"
                    if mxfp4 else "fp8_e4m3_ue8m0_block128"
                ),
            },
        },
    }
    (root / "quant_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.mark.parametrize("mxfp4", [False, True])
def test_exact_contract_closes_dspark_runtime_kwargs(tmp_path, monkeypatch, mxfp4):
    monkeypatch.delenv("GRIDBOOK_MXFP8_DENSE", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    root = _artifact(tmp_path, mxfp4=mxfp4)

    kwargs, receipt = exact_llm_contract(root)

    assert kwargs["quantization"] == "gridbook"
    assert kwargs["kv_cache_dtype"] == "fp8"
    assert kwargs["tokenizer_mode"] == "deepseek_v4"
    assert kwargs["generation_config"] == "vllm"
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["max_model_len"] == 8192
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["max_num_batched_tokens"] == 512
    assert kwargs["kv_cache_memory_bytes"] == 1_073_741_824
    assert kwargs["enforce_eager"] is True
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["trust_remote_code"] is True
    assert kwargs["gpu_memory_utilization"] == 0.84
    assert kwargs["disable_log_stats"] is True
    assert kwargs.get("moe_backend") == ("marlin" if mxfp4 else None)
    assert receipt["requires_moe_backend_marlin"] is mxfp4
    assert receipt["speculative_decoding"] is False
    assert receipt["gpu_memory_utilization"] == 0.84
    assert receipt["disable_log_stats"] is True
    assert receipt["environment"] == dict(CANONICAL_GOLD_ENVIRONMENT)


def test_exact_contract_refuses_non_gridbook_artifact(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "quant_config.json").write_text(
        '{"quant_method":"compressed-tensors"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Gridbook artifact"):
        exact_llm_contract(root)


def test_exact_contract_clears_inherited_non_gold_runtime_knobs(
    tmp_path, monkeypatch,
):
    root = _artifact(tmp_path, mxfp4=False)
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE", "cuda")
    monkeypatch.setenv("PRISMAQUANT_PRELOAD_FUSED", "1")

    _kwargs, receipt = exact_llm_contract(root)

    assert receipt["environment"]["PRISMAQUANT_CB_DECODE"] is None
    assert receipt["environment"]["PRISMAQUANT_PRELOAD_FUSED"] == "0"
    assert "PRISMAQUANT_CB_DECODE" not in __import__("os").environ
    assert __import__("os").environ["PRISMAQUANT_PRELOAD_FUSED"] == "0"
    assert {
        name: __import__("os").environ[name]
        for name in CANONICAL_GOLD_SET_ENVIRONMENT
    } == dict(CANONICAL_GOLD_SET_ENVIRONMENT)


def test_marlin_route_ignores_non_assignment_metadata(tmp_path):
    root = _artifact(tmp_path, mxfp4=False)
    payload = json.loads((root / "quant_config.json").read_text())
    payload["provenance"] = {
        "note": "rejected menu included mxfp4_e2m1_ue8m0_g32",
    }
    (root / "quant_config.json").write_text(json.dumps(payload))

    kwargs, receipt = exact_llm_contract(root)

    assert "moe_backend" not in kwargs
    assert receipt["requires_moe_backend_marlin"] is False


def test_marlin_route_reads_per_expert_live_assignment(tmp_path):
    root = _artifact(tmp_path, mxfp4=False)
    payload = json.loads((root / "quant_config.json").read_text())
    payload.pop("source_passthrough")
    payload["per_expert_format_groups"] = {
        "version": 1,
        "layers": {
            "0": {
                "w13": [{
                    "format_wire_id": "NVFP4_CB_K16",
                    "expert_ids": [0],
                    "tensor_prefix": "model.layers.0.mlp.experts.gate_up_proj",
                }],
                "w2": [{
                    "format_wire_id": "mxfp4_e2m1_ue8m0_g32",
                    "expert_ids": [0],
                    "tensor_prefix": "model.layers.0.mlp.experts",
                }],
            },
        },
    }
    (root / "quant_config.json").write_text(json.dumps(payload))

    kwargs, receipt = exact_llm_contract(root)

    assert kwargs["moe_backend"] == "marlin"
    assert receipt["requires_moe_backend_marlin"] is True
