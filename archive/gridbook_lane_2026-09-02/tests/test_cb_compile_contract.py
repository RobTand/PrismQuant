"""Focused no-GPU tests for the shared CB fail-closed compile contract."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from prismaquant import cb_compile_contract as contract
from prismaquant import cb_ldlq_atoms as atoms
from prismaquant import nvfp4_cb_formats as cb
from prismaquant import rtx4090_cb_compile_proof as campaign


def _strict_env(monkeypatch) -> None:
    monkeypatch.setenv(contract.CB_COMPILE_FAIL_CLOSED_ENV, "1")


def test_strict_compiled_callable_refuses_runtime_failure(monkeypatch):
    _strict_env(monkeypatch)

    def fake_compile(_fn, **kwargs):
        assert kwargs["backend"]
        assert kwargs["dynamic"] is True
        assert kwargs["fullgraph"] is True

        def fail(*_args, **_kwargs):
            raise RuntimeError("compiler runtime exploded")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    token = contract.begin_cb_compile_execution_proof()
    try:
        compiled = contract.compile_cb_callable(
            lambda value: value + 1,
            helper=contract.ENCODE_SCORE_MIN,
        )
        with pytest.raises(
            contract.CBCompileContractError,
            match="failed at runtime",
        ):
            compiled(torch.ones(2))
    finally:
        contract.abort_cb_compile_execution_proof(token)


def test_strict_compiled_callable_refuses_silent_eager_return(monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **_kwargs: fn,
    )
    token = contract.begin_cb_compile_execution_proof()
    try:
        compiled = contract.compile_cb_callable(
            lambda value: value + 1,
            helper=contract.ENCODE_SCORE_MIN_BATCHED,
        )
        with pytest.raises(
            contract.CBCompileContractError,
            match="without a compiled backend dispatch",
        ):
            compiled(torch.ones(2))
    finally:
        contract.abort_cb_compile_execution_proof(token)


def test_strict_execution_proof_records_fullgraph_dispatch(monkeypatch):
    _strict_env(monkeypatch)

    def fake_compile(fn, *, backend, dynamic, fullgraph):
        assert dynamic is True
        assert fullgraph is True
        lowered = None

        def optimized(*args, **kwargs):
            nonlocal lowered
            if lowered is None:
                lowered = backend(fn, args)
            return lowered(*args, **kwargs)

        return optimized

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        contract,
        "_inductor_backend",
        lambda graph_module, _example_inputs: graph_module,
    )
    token = contract.begin_cb_compile_execution_proof()
    compiled = contract.compile_cb_callable(
        lambda value: value + 2,
        helper=contract.ENCODE_SCORE_MIN,
    )
    torch.testing.assert_close(compiled(torch.ones(3)), torch.full((3,), 3.0))
    proof = contract.finish_cb_compile_execution_proof(token)
    validated = contract.validate_cb_compile_execution_proof(
        proof,
        require_live_calls=True,
        allowed_helper_prefixes=("encode.",),
    )
    assert validated["totals"] == {
        "attempted_calls": 1,
        "cuda_calls": 0,
        "compiled_dispatches": 1,
        "graph_compiles": 1,
        "compile_failures": 0,
        "runtime_failures": 0,
        "eager_fallbacks": 0,
    }
    with pytest.raises(
        contract.CBCompileContractError,
        match="exclusively on CUDA",
    ):
        contract.validate_cb_compile_execution_proof(
            proof,
            require_live_calls=True,
            require_cuda_calls=True,
            allowed_helper_prefixes=("encode.",),
        )


def test_execution_proof_refuses_more_dispatches_than_attempts(monkeypatch):
    _strict_env(monkeypatch)
    proof = {
        "schema": contract.CB_COMPILE_EXECUTION_PROOF_SCHEMA,
        "strict_setting": {contract.CB_COMPILE_FAIL_CLOSED_ENV: "1"},
        "policy": {
            "compiler": "torch.compile",
            "backend": "inductor",
            "dynamic": True,
            "fullgraph": True,
            "suppress_errors": False,
            "fallback": "refuse",
        },
        "helpers": {
            contract.ENCODE_SCORE_MIN: {
                "attempted_calls": 1,
                "cuda_calls": 1,
                "compiled_dispatches": 2,
                "graph_compiles": 1,
                "compile_failures": 0,
                "runtime_failures": 0,
                "eager_fallbacks": 0,
            },
        },
        "totals": {
            "attempted_calls": 1,
            "cuda_calls": 1,
            "compiled_dispatches": 2,
            "graph_compiles": 1,
            "compile_failures": 0,
            "runtime_failures": 0,
            "eager_fallbacks": 0,
        },
    }
    proof["proof_sha256"] = contract._canonical_sha256(proof)
    with pytest.raises(
        contract.CBCompileContractError,
        match="did not complete every call",
    ):
        contract.validate_cb_compile_execution_proof(
            proof,
            require_live_calls=True,
            require_cuda_calls=True,
            allowed_helper_prefixes=("encode.",),
        )


def test_generic_score_min_keeps_compatibility_fallback(monkeypatch):
    monkeypatch.delenv(contract.CB_COMPILE_FAIL_CLOSED_ENV, raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_COMPILE", "1")

    def fail_compile():
        raise RuntimeError("generic compiler unavailable")

    monkeypatch.setattr(cb, "_score_min_compiled", fail_compile)
    a = torch.rand(4, 7)
    b = torch.randn(4, 7)
    scale = torch.rand(4, 1)
    torch.testing.assert_close(
        cb._score_min(a, b, scale),
        cb._score_min_eager(a, b, scale),
    )


def test_strict_score_min_propagates_compiled_callable_failure(monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_COMPILE", "1")

    def compiled_factory():
        def fail(*_args):
            raise RuntimeError("strict compiled callable failed")

        return fail

    monkeypatch.setattr(cb, "_score_min_compiled_strict", compiled_factory)
    with pytest.raises(RuntimeError, match="strict compiled callable failed"):
        cb._score_min(torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 1))


def test_strict_atom_gate_marks_cpu_eager_route_as_fallback(monkeypatch):
    _strict_env(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_CB_ATOM_COMPILE", "1")
    token = contract.begin_cb_compile_execution_proof()
    try:
        with pytest.raises(
            contract.CBCompileContractError,
            match="CUDA-only",
        ):
            atoms._fused_route(torch.ones(2, 4))
    finally:
        contract.abort_cb_compile_execution_proof(token)


def _write_checkpoint_manifest(tmp_path, *, compile_settings):
    qnames = ("model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj")
    arm = {"campaign": "strict"}
    model = {"schema": "test.streamed_model", "content_sha256": "a" * 64}
    extra_fields = {"global_plan_sha256": "b" * 64, "stripe_index": 0}
    extra = {
        **extra_fields,
        "compile_settings": dict(compile_settings),
        "production_anchor_renderer": {"arm_identity": arm},
        "streamed_model_identity": model,
    }
    identity = {
        "schema": campaign.AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "extra": extra,
    }
    units = []
    for qname in qnames:
        relative = (
            "units/" + hashlib.sha256(qname.encode("utf-8")).hexdigest() + ".pkl"
        )
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"identity-bound unit fixture")
        units.append({"qname": qname, "file": relative})
    manifest = {
        "schema": campaign.AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_sha256": campaign._canonical_sha256(identity),
        "identity": identity,
        "units": units,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8",
    )
    return qnames, arm, model, extra_fields


def test_restored_checkpoint_binding_closes_compile_arm_and_source(tmp_path):
    settings = {
        "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
        "PRISMAQUANT_CB_ATOM_COMPILE": "1",
        contract.CB_COMPILE_FAIL_CLOSED_ENV: "1",
    }
    qnames, arm, model, extra_fields = _write_checkpoint_manifest(
        tmp_path, compile_settings=settings,
    )
    binding = campaign.capture_aura_checkpoint_compile_binding(
        tmp_path,
        expected_qnames=qnames,
        expected_compile_settings=settings,
        expected_extra_fields=extra_fields,
        expected_arm_identity=arm,
        expected_model_identity=model,
    )
    assert binding["unit_count"] == 2
    assert binding["compile_settings_sha256"] == campaign._canonical_sha256(
        settings
    )


def test_restored_checkpoint_binding_refuses_compile_setting_relabel(tmp_path):
    settings = {
        "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
        "PRISMAQUANT_CB_ATOM_COMPILE": "1",
        contract.CB_COMPILE_FAIL_CLOSED_ENV: "1",
    }
    qnames, arm, model, extra_fields = _write_checkpoint_manifest(
        tmp_path, compile_settings=settings,
    )
    expected = dict(settings)
    expected[contract.CB_COMPILE_FAIL_CLOSED_ENV] = "0"
    with pytest.raises(
        campaign.RTX4090CBCompileProofError,
        match="compile settings differ",
    ):
        campaign.capture_aura_checkpoint_compile_binding(
            tmp_path,
            expected_qnames=qnames,
            expected_compile_settings=expected,
            expected_extra_fields=extra_fields,
            expected_arm_identity=arm,
            expected_model_identity=model,
        )
