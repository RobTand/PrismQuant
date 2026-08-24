from __future__ import annotations

import json
from pathlib import Path

from prismaquant import sensitivity_probe
from prismaquant.aura_cost import _stage_aura_model
from prismaquant.layer_streaming import _build_weight_map


_CHECKPOINT_KEY = "model.language_model.layers.0.self_attn.q_proj.weight"
_LIVE_KEY = "model.layers.0.self_attn.q_proj.weight"
_SHARD = "model-00001-of-00001.safetensors"


def _write_source(tmp_path: Path, name: str, config: dict) -> Path:
    source = tmp_path / name
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config, indent=2))
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {_CHECKPOINT_KEY: _SHARD},
    }))
    (source / _SHARD).write_bytes(b"fixture")
    (source / "tokenizer_config.json").write_text("{}")
    return source


def test_aura_wrapper_staging_prefers_nested_text_execution_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    source = _write_source(tmp_path, "wrapper", {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "intermediate_size": 2048,
        "rope_theta": 10_000,
        "tie_word_embeddings": True,
        "vision_config": {"hidden_size": 256},
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 17_408,
            "rope_theta": 10_000_000,
            "tie_word_embeddings": False,
        },
    })

    staged = Path(_stage_aura_model(str(source)))
    config = json.loads((staged / "config.json").read_text())

    assert staged != source
    assert config["model_type"] == "qwen3_5_text"
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert "text_config" not in config
    assert "vision_config" not in config
    assert {
        key: config[key]
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "rope_theta",
            "tie_word_embeddings",
        )
    } == {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 17_408,
        "rope_theta": 10_000_000,
        "tie_word_embeddings": False,
    }

    # The streaming builder applies the same stager once more.  AURA's staged
    # tree is already canonical, so this second pass must preserve its path.
    assert sensitivity_probe.stage_text_only(str(staged)) == str(staged)

    # Staging preserves the original source index/shards as symlinks, and the
    # streaming weight-map adapter still translates wrapper checkpoint names
    # to the flattened live skeleton without rewriting source identity.
    assert (staged / "model.safetensors.index.json").is_symlink()
    assert (staged / _SHARD).is_symlink()
    weight_shard, weight_checkpoint = _build_weight_map(str(staged))
    assert weight_shard[_LIVE_KEY] == str(staged / _SHARD)
    assert weight_checkpoint[_LIVE_KEY] == _CHECKPOINT_KEY

    # The shared stager, not AURA, owns cleanup.  Registration keeps the tree
    # alive until resident/streaming teardown has completed at process exit.
    assert staged in sensitivity_probe._STAGED_TEMP_DIRS


def test_aura_flattened_text_source_is_an_exact_staging_noop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    source = _write_source(tmp_path, "flattened", {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 17_408,
        "rope_theta": 10_000_000,
        "tie_word_embeddings": False,
    })
    registered_before = list(sensitivity_probe._STAGED_TEMP_DIRS)

    staged = _stage_aura_model(str(source))

    assert staged == str(source)
    assert sensitivity_probe._STAGED_TEMP_DIRS == registered_before
