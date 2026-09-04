"""The paired instrument binds actual native weight bytes, not their sizes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from prismaquant import boundary_control as bc
from tools import measure_boundary_control as tool
from tools import serve_fingerprint as sf


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    weight = model / "model.safetensors"
    weight.write_bytes(b"original weight bytes")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "prompts": [{"text": "9 squared?", "stratum": "numeric"}],
        "seeds": [0], "temperature": 1.0,
        "initial_max_tokens": 16, "max_steps": 1,
    }))
    manifest = {
        "residency_readable": True, "serve_session_id": "same-live-server",
        "models_endpoint_binding": {"model": {"id": "nonce", "max_model_len": 70}},
        "model": str(model), "launch_argv": ["vllm", "serve", str(model),
                                               "--dtype", "bfloat16"],
        "quantization": None, "performance_stack_fingerprint": "a" * 64,
        "host_identity": {"boot_id": "same-boot"},
    }
    monkeypatch.setattr(tool, "activate_prismaquant_source",
                        lambda: Path(__file__).resolve().parents[1])
    monkeypatch.setattr(tool, "_install_exact_package_namespace", lambda _root: None)
    monkeypatch.setitem(sys.modules, "serve_fingerprint", sf)
    monkeypatch.setattr(sf, "collect_manifest", lambda **_kwargs: copy.deepcopy(manifest))
    requests = []
    def post(url, body):
        requests.append(url)
        if url.endswith("/tokenize"):
            return {"count": 2, "tokens": [9, 81]}
        return {"choices": [{"message": {"content": "<think>9*9</think>81"},
                              "finish_reason": "stop"}],
                "usage": {"completion_tokens": 12}}
    monkeypatch.setattr(bc.vqm, "_post_json", post)
    def run(name):
        output = tmp_path / name
        result = tool.main([
            "control", "--base-url", "http://fixture", "--model-name", "nonce",
            "--model-dir", str(model), "--image", "fixture@sha256:" + "a" * 64,
            "--contract", str(contract), "--campaign-id", "same-campaign",
            "--out", str(output),
        ])
        assert result == 0
        return json.loads(output.read_text())
    return model, weight, run, post, requests


def test_same_size_weight_edit_changes_boundary_artifact_identity(campaign):
    _model, weight, run, _post, _requests = campaign
    first = run("first.json")
    weight.write_bytes(b"replacement of weight")
    assert len(b"replacement of weight") == len(b"original weight bytes")
    second = run("second.json")
    assert first["measurement"]["binding"]["artifact_id"] != second["measurement"]["binding"]["artifact_id"]


def test_same_size_weight_edit_during_measurement_refuses_receipt(campaign, monkeypatch):
    model, weight, run, post, _requests = campaign
    def mutate(url, body):
        response = post(url, body)
        if url.endswith("/chat/completions"):
            weight.write_bytes(b"replacement of weight")
        return response
    monkeypatch.setattr(bc.vqm, "_post_json", mutate)
    with pytest.raises(ValueError, match="artifact changed during measurement"):
        run("changed.json")
    assert not (model.parent / "changed.json").exists()


def test_unchanged_content_is_stable_and_carries_exact_pre_post_manifests(campaign):
    _model, weight, run, _post, _requests = campaign
    first = run("first.json")
    weight.touch()
    second = run("second.json")
    assert first["measurement"]["binding"]["artifact_id"] == second["measurement"]["binding"]["artifact_id"]
    assert first["artifact_pre"] == first["artifact_post"] == second["artifact_pre"]
    content = first["artifact_pre"]["weight_content_manifest"]
    assert content["files"][weight.name]["sha256"] == bc.hashlib.sha256(weight.read_bytes()).hexdigest()
    assert first["measurement"]["binding"]["artifact_id"] == bc.digest(first["artifact_pre"])


def test_missing_native_weights_refuses_before_http(campaign):
    _model, weight, run, _post, requests = campaign
    weight.unlink()
    with pytest.raises(FileNotFoundError, match="no safetensors weights"):
        run("absent.json")
    assert requests == []


def test_mutation_restored_before_post_hash_still_refuses(campaign, monkeypatch):
    _model, weight, run, post, _requests = campaign
    original = weight.read_bytes()
    def mutate_restore(url, body):
        response = post(url, body)
        if url.endswith("/chat/completions"):
            weight.write_bytes(b"replacement of weight")
            weight.write_bytes(original)
        return response
    monkeypatch.setattr(bc.vqm, "_post_json", mutate_restore)
    with pytest.raises(ValueError, match="artifact changed during measurement"):
        run("restored.json")


def test_offline_replay_refuses_content_manifest_inconsistent_with_identity(campaign):
    _model, weight, run, _post, _requests = campaign
    measurement = run("first.json")["measurement"]
    manifest = measurement["binding"]["artifact_content"]["weight_content_manifest"]
    manifest["files"][weight.name]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="artifact_id"):
        bc.replay_control(measurement)
