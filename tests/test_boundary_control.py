"""Paired boundary measurement must not mistake a token budget for a defect."""
from __future__ import annotations

import copy
import importlib

import pytest


def _api():
    return importlib.import_module("prismaquant.boundary_control")


def _response(cap, *, finish="stop", text="<think>work</think> answer"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"completion_tokens": min(cap, 12)}}


def _contract():
    return {"prompts": [{"text": "9²", "stratum": "numeric"}],
            "seeds": [31, 47], "temperature": 1.0,
            "initial_max_tokens": 8, "max_steps": 5}


def _binding(artifact="bf16"):
    from prismaquant.shipcard import WEIGHT_CONTENT_MANIFEST_SCHEMA
    content = {"model_sha": _api().digest(artifact), "weight_content_manifest": {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA, "algorithm": "sha256",
        "files": {"model.safetensors": {"bytes": len(artifact), "sha256": _api().digest(artifact)}},
    }}
    return {"campaign_id": "paired-test", "artifact_id": _api().artifact_content_id(content),
            "artifact_content": content,
            "serve_session_id": artifact + "-session",
            "serve_fingerprint": "a" * 64, "host_boot_id": "boot-1",
            "model_context_tokens": 70, "prompt_tokens": [6],
            "prompt_token_ids": [[1, 2, 3, 4, 5, 6]],
            "producer_source_sha256": "c" * 64}


def _control(monkeypatch, *, finish_at=16):
    api = _api()
    def post(_url, body):
        cap = body["max_tokens"]
        return _response(cap, finish="stop" if cap >= finish_at else "length")
    monkeypatch.setattr(api.vqm, "_post_json", post)
    return api.measure_control("http://server", "model", _contract(), _binding())


def test_control_grows_until_uncensored_instead_of_zero_defects(monkeypatch):
    api = _api()
    def post(_url, body):
        cap = body["max_tokens"]
        return _response(cap, finish="stop" if cap >= 16 else "length",
                         text="answer without a reasoning tag")
    monkeypatch.setattr(api.vqm, "_post_json", post)
    control = api.measure_control("http://server", "model", _contract(), _binding())
    assert [row["max_tokens"] for row in control["steps"]] == [8, 16]
    assert control["fixed_point"] is True
    assert control["steps"][-1]["n_defects"] == 2


def test_control_uses_exact_context_remainder_and_refuses_a_backstop(monkeypatch):
    api = _api()
    monkeypatch.setattr(api.vqm, "_post_json", lambda _url, body: _response(
        body["max_tokens"], finish="length"))
    control = api.measure_control("http://server", "model", _contract(), _binding())
    assert [row["max_tokens"] for row in control["steps"]] == [8, 16, 32, 64]
    assert control["fixed_point"] is False
    assert control["stop_reason"] == "context_bound"
    with pytest.raises(ValueError, match="fixed point"):
        api.measure_candidate("http://server", "model", control, _binding("quant"))


def test_iteration_backstop_never_certifies_fixed_point(monkeypatch):
    api = _api()
    monkeypatch.setattr(api.vqm, "_post_json", lambda _url, body: _response(
        body["max_tokens"], finish="length"))
    contract = {**_contract(), "max_steps": 1}
    control = api.measure_control("http://server", "model", contract, _binding())
    assert control["stop_reason"] == "iteration_backstop"
    assert control["fixed_point"] is False


def test_candidate_matches_control_seeds_cap_and_reports_relative_counts(monkeypatch):
    api = _api()
    control = _control(monkeypatch)
    requests = []
    def post(_url, body):
        requests.append(body)
        return _response(body["max_tokens"], text="runaway", finish="length")
    monkeypatch.setattr(api.vqm, "_post_json", post)
    candidate = api.measure_candidate("http://server", "model", control, _binding("quant"))
    comparison = api.compare(control, candidate)
    assert [row["seed"] for row in requests] == _contract()["seeds"]
    assert {row["max_tokens"] for row in requests} == {16}
    assert comparison["strata"]["numeric"]["delta_defects"] == 2
    assert comparison["advisory_only"] is True
    assert "passed" not in comparison


@pytest.mark.parametrize("field,value", [
    ("campaign_id", "other-campaign"), ("host_boot_id", "other-boot"),
    ("serve_fingerprint", "b" * 64), ("prompt_tokens", [7]),
    ("prompt_token_ids", [[6, 5, 4, 3, 2, 1]]),
    ("producer_source_sha256", "d" * 64),
])
def test_candidate_refuses_unpaired_stack_or_tokenization(monkeypatch, field, value):
    api = _api()
    control = _control(monkeypatch)
    binding = {**_binding("quant"), field: value}
    with pytest.raises(ValueError, match=field):
        api.measure_candidate("http://server", "model", control, binding)


@pytest.mark.parametrize("dtype,quantization", [("half", None), ("float16", None),
                                                ("bfloat16", "fp8")])
def test_control_refuses_live_dtype_or_quantization_override(dtype, quantization):
    with pytest.raises(ValueError, match="BF16 control"):
        _api().require_bf16_control({"torch_dtype": "bfloat16"}, dtype, quantization)


@pytest.mark.parametrize("argv", [
    ["vllm", "serve", "/bf16", "-q", "fp8"],
    ["vllm", "serve", "/bf16", "--dtype", "bfloat16", "--dtype", "half"],
    ["vllm", "serve", "/bf16", "--hf-overrides", "{}"],
])
def test_control_refuses_ambiguous_or_aliased_live_overrides(argv):
    with pytest.raises(ValueError, match="BF16 control"):
        _api().require_bf16_control({"torch_dtype": "bfloat16"}, "bfloat16", None,
                                   launch_argv=argv)


def test_replay_rejects_forged_counts_and_missing_generations(monkeypatch):
    api = _api()
    control = _control(monkeypatch)
    for mutation in ("count", "generation", "fixed_point", "cap"):
        forged = copy.deepcopy(control)
        if mutation == "count":
            forged["steps"][-1]["n_defects"] = 123
        elif mutation == "generation":
            forged["steps"][-1]["outcomes"].pop()
        elif mutation == "fixed_point":
            forged["steps"][0]["outcomes"][0]["finish_reason"] = "stop"
        else:
            forged["steps"][-1]["max_tokens"] += 1
        with pytest.raises(ValueError):
            api.replay_control(forged)


@pytest.mark.parametrize("field,value", [
    ("temperature", float("nan")), ("seeds", [1, 1]),
    ("prompts", []), ("max_steps", 0),
])
def test_invalid_measurement_contract_refuses_before_request(monkeypatch, field, value):
    api = _api()
    def forbidden(*_args):
        pytest.fail("invalid contract reached the server")
    monkeypatch.setattr(api.vqm, "_post_json", forbidden)
    with pytest.raises(ValueError):
        api.measure_control("http://server", "model", {**_contract(), field: value}, _binding())
