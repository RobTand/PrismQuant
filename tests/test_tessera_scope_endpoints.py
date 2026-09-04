"""Explicit CLI targets reach Tessera admission with each unit's topology."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from types import SimpleNamespace

import pytest


IMAGE = "example/runtime@sha256:" + "a" * 64
DENSE = "model.layers.0.self_attn.o_proj"
EXPERT = "model.layers.0.mlp.experts.0.down_proj"
SHARED = "model.layers.0.mlp.shared_expert.down_proj"


def _scope():
    from prismaquant import tessera_serving_scope
    return tessera_serving_scope


def _args(**changed):
    values = dict(tessera_runtime_image=IMAGE, tessera_execution_mode="eager",
                  tessera_residency="resident", tessera_platform="sm_121")
    values.update(changed)
    return argparse.Namespace(**values)


def _profile():
    from prismaquant.model_profiles import Qwen3Profile
    return Qwen3Profile()


def _stats():
    return {
        DENSE: dict(router_path=None, expert_id=None),
        EXPERT: dict(router_path="model.layers.0.mlp.gate", expert_id="0"),
        SHARED: dict(router_path=None, expert_id=None),
    }


def test_scope_cli_has_no_implicit_runtime_defaults():
    parser = argparse.ArgumentParser()
    _scope().add_serving_scope_arguments(parser)
    assert _scope().serving_target_from_args(parser.parse_args([])) is None


@pytest.mark.parametrize("field", [
    "tessera_runtime_image", "tessera_execution_mode", "tessera_residency",
    "tessera_platform",
])
def test_partial_runtime_target_refuses_named_missing_field(field):
    with pytest.raises(ValueError, match=field.removeprefix("tessera_")):
        _scope().serving_target_from_args(_args(**{field: None}))


def test_platform_comes_from_selected_profile_not_probe_device():
    scope = _scope()
    target = scope.serving_target_from_args(
        _args(tessera_platform=None), target_platform="sm_121")
    assert target.platform == "sm_121"
    with pytest.raises(ValueError, match="platform.*conflict"):
        scope.serving_target_from_args(_args(), target_platform="sm_120")


def test_unit_topology_preserves_dense_shared_and_routed_split():
    scope = _scope()
    contexts = scope.context_by_unit_from_stats(
        scope.serving_target_from_args(_args()), _stats(), _profile())
    assert {name: ctx.structure for name, ctx in contexts.items()} == {
        DENSE: "dense", EXPERT: "routed_moe", SHARED: "dense"}
    assert all(ctx.runtime_image == IMAGE for ctx in contexts.values())


def test_packed_expert_and_grouped_attention_are_not_the_same_structure():
    scope = _scope()
    packed = "model.layers.0.mlp.experts.gate_up_proj"
    rows = {
        packed: dict(num_experts=2, _packed_experts_module="model.layers.0.mlp.experts",
                     router_path=None, expert_id=None),
        DENSE: dict(num_groups=2, router_path=None, expert_id=None),
    }
    contexts = scope.context_by_unit_from_stats(
        scope.serving_target_from_args(_args()), rows, _profile())
    assert contexts[packed].structure == "routed_moe"
    assert contexts[DENSE].structure == "dense"


@pytest.mark.parametrize("name,row", [
    (DENSE, {}),
    (EXPERT, dict(router_path=None, expert_id=None)),
    (DENSE, dict(router_path="router", expert_id=None)),
    (DENSE, dict(router_path=None, expert_id="0")),
])
def test_ambiguous_or_conflicting_unit_topology_is_not_guessed(name, row):
    scope = _scope()
    with pytest.raises(ValueError, match=name):
        scope.context_by_unit_from_stats(
            scope.serving_target_from_args(_args()), {name: row}, _profile())


def test_unscoped_legacy_call_does_not_require_new_topology_fields():
    assert _scope().context_by_unit_from_stats(None, {"old.row": {}}, None) is None


def _cli_scope():
    return ["--tessera-runtime-image", IMAGE, "--tessera-execution-mode", "eager",
            "--tessera-residency", "resident", "--tessera-platform", "sm_121"]


class EndpointReached(Exception):
    pass


def test_campaign_main_passes_live_model_topology_to_real_menu_boundary(tmp_path, monkeypatch):
    import torch
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_render as render
    from prismaquant import sensitivity_probe
    import transformers

    modules = {name: torch.nn.Linear(32, 32, bias=False) for name in (DENSE, EXPERT)}
    model = SimpleNamespace(named_modules=lambda: modules.items(), eval=lambda: None)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(render, "tessera_encoder_hessian_status", lambda: {"accepted": True})
    monkeypatch.setattr(campaign, "_calibration_tokens", lambda *a: ([torch.ones(1, 1, dtype=torch.int64)], "fixture"))
    monkeypatch.setattr(campaign, "_collect_activations", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(sensitivity_probe, "discover_moe_structure", lambda *a, **k: {
        EXPERT: ("model.layers.0.mlp.gate", "0")})
    seen = {}

    def menus(weights, targets, **kwargs):
        seen.update(kwargs)
        raise EndpointReached

    monkeypatch.setattr(campaign, "expand_menus_for_targets", menus)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    with pytest.raises(EndpointReached):
        campaign.main(["--model", str(model_dir), "--out", str(tmp_path / "cost.pkl"),
                       "--cache-dir", str(tmp_path / "cache"), "--hessian", "off", *_cli_scope()])
    assert seen["context_by_unit"][DENSE].structure == "dense"
    assert seen["context_by_unit"][EXPERT].structure == "routed_moe"


def _allocator_inputs(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    stats = _stats()
    for row in stats.values():
        row.update(h_trace=1.0, n_params=128 * 128, in_features=128, out_features=128)
    probe = tmp_path / "probe.pkl"
    costs = tmp_path / "cost.pkl"
    probe.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": str(model_dir)}}))
    costs.write_bytes(pickle.dumps({"costs": {
        name: {"BF16": {"predicted_dloss": 0.0}} for name in stats}, "formats": ["BF16"]}))
    return ["--probe", str(probe), "--costs", str(costs), "--formats", "BF16",
            "--allow-legacy-fisher-norm",
            "--target-profile", "tessera_research_sm121", "--target-bits", "16",
            "--pareto-targets", "16", "--bit-precision", "0.1",
            "--layer-config", str(tmp_path / "layer.json"),
            "--pareto-csv", str(tmp_path / "pareto.csv")]


def test_allocator_cli_carries_per_unit_context_into_candidate_builder(tmp_path, monkeypatch):
    from prismaquant import allocator
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        raise EndpointReached

    monkeypatch.setattr(allocator, "build_candidates", build)
    monkeypatch.setattr(sys, "argv", ["allocator", *_allocator_inputs(tmp_path), *_cli_scope()])
    with pytest.raises(EndpointReached):
        allocator.main()
    assert {name: ctx.structure for name, ctx in seen["context_by_unit"].items()} == {
        DENSE: "dense", EXPERT: "routed_moe", SHARED: "dense"}


def test_allocator_real_endpoint_records_each_selected_scope(tmp_path, monkeypatch):
    from prismaquant import allocator
    from prismaquant import allocator_candidates
    original = allocator_candidates.selection_serving_lane_provenance
    seen = {}

    def provenance(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(allocator, "selection_serving_lane_provenance", provenance)
    monkeypatch.setattr(sys, "argv", ["allocator", *_allocator_inputs(tmp_path),
                                    "--no-fused-aggregation", "--no-packed-aggregation", *_cli_scope()])
    allocator.main()
    assert seen["context_by_unit"][EXPERT].structure == "routed_moe"
    assert seen["context_by_unit"][DENSE].structure == "dense"
