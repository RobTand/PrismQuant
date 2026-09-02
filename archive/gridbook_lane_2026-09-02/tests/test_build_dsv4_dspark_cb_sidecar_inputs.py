from __future__ import annotations

from types import SimpleNamespace
import json
import pickle

import torch

import scripts.build_dsv4_dspark_cb_sidecar_inputs as builder
from prismaquant.dspark_source_metadata import (
    FP8_BLOCK_UE8M0_SOURCE_FORMAT,
    MXFP4_SOURCE_FORMAT,
)
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.nvfp4_cb_footprint import CBSerializationContext


def _production_config() -> dict:
    return {
        "num_hidden_layers": 43,
        "n_mtp_layers": 3,
        "dspark_target_layer_ids": [40, 41, 42],
        "n_routed_experts": 256,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "moe_intermediate_size": 2048,
    }


def _source_identity() -> dict:
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "content_sha256": "a" * 64,
        "resolved_commit": "source-revision",
        "checkpoint_shards": 1,
        "checkpoint_tensors": 4705,
    }


def _physical_targets() -> dict[str, str]:
    targets: dict[str, str] = {}
    for stage in range(3):
        prefix = f"mtp.{stage}."
        for tail in (
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w2",
            "ffn.shared_experts.w3",
        ):
            targets[prefix + tail] = FP8_BLOCK_UE8M0_SOURCE_FORMAT
        for expert in range(256):
            for leaf in ("w1", "w2", "w3"):
                targets[f"{prefix}ffn.experts.{expert}.{leaf}"] = (
                    MXFP4_SOURCE_FORMAT
                )
    targets["mtp.0.main_proj"] = FP8_BLOCK_UE8M0_SOURCE_FORMAT
    return targets


def _shape(base: str) -> tuple[int, int]:
    tail = ".".join(base.split(".")[2:])
    if ".experts." in base:
        leaf = base.rsplit(".", 1)[1]
        return (4096, 2048) if leaf == "w2" else (2048, 4096)
    return {
        "attn.wq_a": (1024, 4096),
        "attn.wkv": (512, 4096),
        "attn.wq_b": (32768, 1024),
        "attn.wo_a": (8192, 4096),
        "attn.wo_b": (4096, 8192),
        "ffn.shared_experts.w1": (2048, 4096),
        "ffn.shared_experts.w2": (4096, 2048),
        "ffn.shared_experts.w3": (2048, 4096),
    }[tail]


class _FakeSkeleton:
    def keys(self):
        return ()

    def logical_shape(self, name: str) -> tuple[int, int]:
        assert name.endswith(".weight")
        return _shape(name.removesuffix(".weight"))


def test_builder_emits_exact_2325_cb_plus_three_wo_a_recipe(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_production_config()))
    targets = _physical_targets()
    overlay = SimpleNamespace(
        physical_targets=targets,
        num_hidden_layers=43,
        n_mtp_layers=3,
    )
    monkeypatch.setattr(builder, "_LazySkeleton", lambda _source: _FakeSkeleton())
    monkeypatch.setattr(
        builder,
        "discover_dspark_source_overlay",
        lambda _skeleton, _config: overlay,
    )
    real_ones = torch.ones
    monkeypatch.setattr(
        builder.torch,
        "ones",
        lambda *_args, **_kwargs: real_ones(1, dtype=torch.float32),
    )

    out_dir = tmp_path / "inputs"
    manifest = builder.build(
        source,
        out_dir,
        "NVFP4_CB_K12",
        source_model_identity=_source_identity(),
        serialization_context=CBSerializationContext.production(),
    )
    assignment = json.loads(
        (out_dir / "dspark_layer_config.json").read_text()
    )
    with (out_dir / "dspark_col_weights.pkl").open("rb") as handle:
        col_weights = pickle.load(handle)

    expected_source = {
        f"mtp.{stage}.attn.wo_a" for stage in range(3)
    }
    assert manifest["schema"] == "prismaquant.dspark_cb_sidecar_inputs.v2"
    metadata = assignment.pop("__prismaquant__")
    assert metadata["cb_render_identity"]["source_weights_complete"] is False
    assert metadata["dspark_render_recipe"]["source_model_identity"] == (
        _source_identity()
    )
    assert manifest["source_model_identity"] == _source_identity()
    assert manifest["render_identity_seed_sha256"] == metadata[
        "dspark_render_recipe"
    ]["render_identity_seed_sha256"]
    assert manifest["decoder_recipe_entry_count"] == 2328
    assert manifest["decoder_linear_count"] == 2325
    assert manifest["source_passthrough_decoder_linear_count"] == 3
    assert manifest["quantized_parameters"] == 19_623_051_264
    assert {
        name for name, fmt in assignment.items()
        if fmt == FP8_BLOCK_UE8M0_SOURCE_FORMAT
    } == expected_source
    assert expected_source.isdisjoint(col_weights)
    assert len(col_weights) == 2325
    assert manifest["source_passthrough_targets"] == {
        "mtp.0.attn.wo_a": "model.layers.43.attn.wo_a",
        "mtp.0.main_proj": "model.main_proj",
        "mtp.1.attn.wo_a": "model.layers.44.attn.wo_a",
        "mtp.2.attn.wo_a": "model.layers.45.attn.wo_a",
    }
