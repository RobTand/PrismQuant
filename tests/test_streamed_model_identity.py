from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig

import prismaquant.cost_streaming as cost_streaming
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.export_nvfp4_cb_streaming import (
    _SOURCE_MODEL_IDENTITY_CACHE_ENV,
    _bind_source_model_identity_provenance,
    _require_production_source_model_identity,
    _source_model_identity_from_env,
)


class _Config:
    _commit_hash = "source-revision"

    def __init__(self, source: str = ""):
        self._config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=32,
        )
        self._config._name_or_path = source

    def to_dict(self):
        return self._config.to_dict()


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    config = _Config(str(source))
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    body = source / "model-00001-of-00002.safetensors"
    sidecar = source / "model-00002-of-00002.safetensors"
    body.write_bytes(b"body-weights")
    sidecar.write_bytes(b"mtp-passthrough")
    checkpoint_map = {
        "model.layers.0.proj.weight": body.name,
        "mtp.0.proj.weight": sidecar.name,
    }
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": checkpoint_map,
    }))
    runner = SimpleNamespace(
        model=SimpleNamespace(config=config),
        context=SimpleNamespace(
            # Deliberately omit the MTP shard, mirroring a streamed decoder
            # runner.  Complete identity must still cover the export sidecar.
            weight_shard={"model.layers.0.proj.weight": body},
            weight_ckpt={
                "model.layers.0.proj.weight":
                "model.layers.0.proj.weight",
            },
        ),
    )
    return source, body, sidecar, checkpoint_map, runner


def test_cached_streamed_identity_refuses_live_semantic_config_drift(
    tmp_path,
):
    source, _body, _sidecar, _checkpoint_map, runner = _fixture(tmp_path)
    cache = tmp_path / "streamed_model_identity.json"
    cost_streaming.build_streamed_model_identity(
        runner, str(source), identity_cache_path=cache,
    )
    before = json.loads((source / "config.json").read_text())
    before["attention_bias"] = not bool(before.get("attention_bias", False))
    (source / "config.json").write_text(json.dumps(before))

    with pytest.raises(RuntimeError, match="live config differs"):
        cost_streaming.validate_cached_streamed_model_identity(source, cache)


def test_portable_streamed_identity_joins_independent_roots_and_staging_paths(
    tmp_path,
):
    host_a = tmp_path / "sparky"
    host_b = tmp_path / "sparklina"
    host_a.mkdir()
    host_b.mkdir()
    source_a, _body_a, _sidecar_a, _map_a, runner_a = _fixture(host_a)
    source_b, _body_b, _sidecar_b, _map_b, runner_b = _fixture(host_b)
    runner_a.model.config = _Config("/tmp/prismaquant_stage_sparky")
    runner_b.model.config = _Config("/tmp/prismaquant_stage_sparklina")
    cache_a = host_a / "streamed_model_identity.json"
    cache_b = host_b / "streamed_model_identity.json"

    identity_a = cost_streaming.build_streamed_model_identity(
        runner_a, str(source_a), identity_cache_path=cache_a,
    )
    identity_b = cost_streaming.build_streamed_model_identity(
        runner_b, str(source_b), identity_cache_path=cache_b,
    )
    # v1 deliberately remains host-local for backward compatibility.
    assert identity_a["content_sha256"] != identity_b["content_sha256"]
    assert identity_a["config"]["_name_or_path"] != (
        identity_b["config"]["_name_or_path"]
    )
    assert {Path(row["path"]).parent for row in identity_a["shards"]} == {
        source_a.resolve()
    }
    assert {Path(row["path"]).parent for row in identity_b["shards"]} == {
        source_b.resolve()
    }

    validated_a = cost_streaming.validate_cached_streamed_model_identity(
        source_a, cache_a,
    )
    validated_b = cost_streaming.validate_cached_streamed_model_identity(
        source_b, cache_b,
    )
    portable_a = cost_streaming.portable_streamed_model_content_identity(
        validated_a,
    )
    portable_b = cost_streaming.portable_streamed_model_content_identity(
        validated_b,
    )
    assert portable_a == portable_b

    # The path-neutral join does not weaken each cache's local source binding.
    with pytest.raises(RuntimeError, match="does not bind source"):
        cost_streaming.validate_cached_streamed_model_identity(source_b, cache_a)


def test_streamed_identity_covers_index_passthrough_and_validates_by_stat(
    tmp_path, monkeypatch,
):
    source, body, sidecar, checkpoint_map, runner = _fixture(tmp_path)
    cache = tmp_path / "streamed_model_identity.json"

    identity = cost_streaming.build_streamed_model_identity(
        runner, str(source), identity_cache_path=cache
    )
    assert {Path(row["path"]) for row in identity["shards"]} == {
        body.resolve(), sidecar.resolve(),
    }
    assert identity["checkpoint_weight_map"] == checkpoint_map

    # Cache validation and exact reuse must not reread either checkpoint file.
    monkeypatch.setattr(
        cost_streaming,
        "_file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("valid cache unexpectedly rehashed source bytes")
        ),
    )
    assert cost_streaming.validate_cached_streamed_model_identity(
        source, cache
    )["content_sha256"] == identity["content_sha256"]
    assert cost_streaming.build_streamed_model_identity(
        runner, str(source), identity_cache_path=cache
    ) == identity

    # Same-size content drift changes ctime and invalidates the cached digest.
    sidecar.write_bytes(b"MTP-passthrough")
    with pytest.raises(RuntimeError, match="stat drifted"):
        cost_streaming.validate_cached_streamed_model_identity(source, cache)


def test_streamed_identity_upgrades_partial_cache_without_rehashing_body(
    tmp_path, monkeypatch,
):
    source, body, sidecar, _checkpoint_map, runner = _fixture(tmp_path)
    cache = tmp_path / "streamed_model_identity.json"
    body_row = {
        "path": str(body.resolve()),
        "size": body.stat().st_size,
        "sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
    }
    value_bearing = {
        "config": _Config().to_dict(),
        "weight_map": {
            "model.layers.0.proj.weight": "model.layers.0.proj.weight",
        },
        "shards": [body_row],
    }
    old_identity = {
        "schema": cost_streaming.STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": str(source),
        "resolved_commit": "source-revision",
        "content_sha256": canonical_json_sha256(
            value_bearing, where="old streamed identity fixture"
        ),
        **value_bearing,
    }
    cache.write_text(json.dumps({
        "schema": cost_streaming.STREAMED_MODEL_IDENTITY_CACHE_SCHEMA,
        "source": str(source),
        "fingerprints": [
            cost_streaming._streamed_identity_stat_fingerprint(body)
        ],
        "identity": old_identity,
    }))
    with pytest.raises(RuntimeError, match="complete source checkpoint"):
        cost_streaming.validate_cached_streamed_model_identity(source, cache)

    hashed: list[Path] = []
    real_sha = cost_streaming._file_sha256

    def track_sha(path):
        hashed.append(Path(path))
        return real_sha(path)

    monkeypatch.setattr(cost_streaming, "_file_sha256", track_sha)
    upgraded = cost_streaming.build_streamed_model_identity(
        runner, str(source), identity_cache_path=cache
    )
    assert hashed == [sidecar.resolve()]
    assert len(upgraded["shards"]) == 2
    assert cost_streaming.validate_cached_streamed_model_identity(
        source, cache
    ) == upgraded


def test_export_identity_env_returns_compact_value_bearing_provenance(
    tmp_path, monkeypatch,
):
    source, _body, _sidecar, checkpoint_map, runner = _fixture(tmp_path)
    cache = tmp_path / "streamed_model_identity.json"
    identity = cost_streaming.build_streamed_model_identity(
        runner, str(source), identity_cache_path=cache
    )
    monkeypatch.setenv(_SOURCE_MODEL_IDENTITY_CACHE_ENV, str(cache))

    compact = _source_model_identity_from_env(source)
    assert compact == {
        "schema": cost_streaming.STREAMED_MODEL_IDENTITY_SCHEMA,
        "content_sha256": identity["content_sha256"],
        "resolved_commit": "source-revision",
        "checkpoint_shards": 2,
        "checkpoint_tensors": len(checkpoint_map),
    }
    quant_config = {"provenance": {"streaming": True}}
    _bind_source_model_identity_provenance(quant_config, compact)
    assert quant_config["provenance"]["source_model_identity"] == compact


def test_production_dsv4_export_requires_complete_source_identity(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
    }))
    monkeypatch.delenv(_SOURCE_MODEL_IDENTITY_CACHE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="complete index-referenced checkpoint"):
        _require_production_source_model_identity(
            source, None, allow_unstamped_research=False
        )

    # The escape hatch is explicit and already stamped as research by the
    # exporter; tiny synthetic DSv4 fixtures continue to use it.
    _require_production_source_model_identity(
        source, None, allow_unstamped_research=True
    )
    _require_production_source_model_identity(
        source,
        {"content_sha256": "0" * 64},
        allow_unstamped_research=False,
    )
