"""Qwen3-30B-A3B producer-profile contract.

The fixture is header metadata from Qwen/Qwen3-30B-A3B revision
ad44e777bcd18fa416d9da3bd8f70d33ebb85d39, not model weights.  It pins the
real checkpoint names and dimensions that drive expert stacking, source-kind
masks, and vLLM scheme dispatch.
"""
from __future__ import annotations

import json
import re

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from prismaquant.allocator_candidates import (
    _scan_source_dtype_manifest,
    check_format_applicability,
)
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.model_profiles.registry import profile_from_config


@pytest.fixture
def census():
    return {
        "layers": 48,
        "experts": 128,
        "top_k": 8,
        "hidden": 2048,
        "expert_intermediate": 768,
        "shared_experts": 0,
        "dtype": "BF16",
        "real_names": (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.127.up_proj.weight",
            "model.layers.47.mlp.experts.127.down_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.47.self_attn.o_proj.weight",
        ),
    }


@pytest.fixture
def profile() -> Qwen3Profile:
    return Qwen3Profile()


def test_detects_real_checkpoint_config():
    resolved = profile_from_config({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
    })
    assert isinstance(resolved, Qwen3Profile)
    assert resolved.name == "qwen3"
    assert resolved.vllm_architecture_class() == "Qwen3MoeForCausalLM"
    # `nvfp4_cb` left this tuple on 2026-09-02 with the Gridbook lane
    # (archive/gridbook_lane_2026-09-02/); `tessera` joined it the next day,
    # the addition half of the same decision.
    #
    # Note what this config is: a ROUTED-MoE Qwen3 resolving to the profile
    # that declares the Tessera lane. Declaring the lane is not permission to
    # build one for this checkpoint -- the packaged contract declares
    # structures ["dense"] and carries no routed_moe cell, so
    # `prismaquant.tessera_export_lane` refuses this shape by reading that
    # table (tests/test_tessera_export_lane.py). Lane membership is a
    # vocabulary fact; buildability is a gate.
    assert resolved.supported_export_lanes() == (
        "compressed-tensors", "tessera")


def test_dense_qwen3_still_resolves_to_the_same_contract_profile():
    resolved = profile_from_config({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    })
    assert isinstance(resolved, Qwen3Profile)
    assert resolved.name == "qwen3"


def test_real_checkpoint_names_are_identity_round_trips(profile, census):
    for checkpoint_name in census["real_names"]:
        live_name = profile.checkpoint_to_live_name(checkpoint_name)
        assert live_name == checkpoint_name
        assert profile.source_tensor_name(live_name) == checkpoint_name
        assert profile.export_tensor_name(live_name) == checkpoint_name
        assert profile.to_vllm_internal_name(live_name) == checkpoint_name


def test_census_enumerates_all_experts_and_layers(profile, census):
    names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
        for layer in range(census["layers"])
        for expert in range(census["experts"])
        for projection in ("gate", "up", "down")
    ]
    regex = re.compile(profile.per_expert_moe_regex().removeprefix("re:"))
    assert len(names) == 48 * 128 * 3 == 18_432
    assert all(regex.fullmatch(name) for name in names)
    assert {int(name.split(".")[2]) for name in names} == set(range(48))
    assert {int(name.split(".")[5]) for name in names} == set(range(128))


def test_checkpoint_projection_shapes_pack_to_live_w13_w2(profile, census):
    hidden = census["hidden"]
    inter = census["expert_intermediate"]
    experts = census["experts"]

    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj", "down_proj",
    })
    assert profile.packed_expert_module_class_names() == frozenset({
        "Qwen3MoeExperts",
    })
    assert profile.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj", "up_proj",
    )
    assert profile.packed_expert_projection_names("down_proj") == (
        "down_proj",
    )
    assert (experts, 2 * inter, hidden) == (128, 1536, 2048)
    assert (experts, hidden, inter) == (128, 2048, 768)
    assert profile.split_packed_experts_for_format("BF16") is True
    assert profile.split_packed_experts_for_format("FP8_CB_K28") is True


def test_expert_stack_is_one_format_unit_per_layer(profile):
    packed_w13 = "model.layers.7.mlp.experts.gate_up_proj"
    packed_w2 = "model.layers.7.mlp.experts.down_proj"
    split = [
        f"model.layers.7.mlp.experts.{expert}.{projection}_proj"
        for expert in (0, 127)
        for projection in ("gate", "up", "down")
    ]
    packed_group = profile.packed_expert_format_group(packed_w13)
    assert packed_group == profile.packed_expert_format_group(packed_w2)
    split_groups = {profile.packed_expert_format_group(name) for name in split}
    assert len(split_groups) == 1
    assert packed_group not in split_groups


def test_body_classification_has_no_shared_expert(profile, census):
    assert census["shared_experts"] == 0
    assert profile.body_layer_prefix() == "model.layers"
    assert profile.lm_head_name() == "lm_head"
    assert profile.is_pinned_name("lm_head")
    assert not profile.is_pinned_name("model.layers.0.mlp.gate")
    assert profile.fused_sibling_group(
        "model.layers.0.self_attn.q_proj"
    ) == "model.layers.0.self_attn.qkv_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.gate_proj"
    ) == "model.layers.0.mlp.gate_up_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.shared_expert.gate_proj"
    ) is None


def test_small_census_fixture_folds_bf16_sources_onto_packed_recipe(
    tmp_path, profile,
):
    tensors = {}
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.zeros(
            (2, 3), dtype=torch.bfloat16)
        tensors[f"{prefix}.up_proj.weight"] = torch.zeros(
            (2, 3), dtype=torch.bfloat16)
        tensors[f"{prefix}.down_proj.weight"] = torch.zeros(
            (3, 2), dtype=torch.bfloat16)
    tensors["model.layers.0.mlp.gate.weight"] = torch.zeros(
        (2, 3), dtype=torch.bfloat16)

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {
            "total_size": sum(t.numel() * t.element_size()
                              for t in tensors.values()),
        },
        "weight_map": {name: shard for name in tensors},
    }))

    manifest = _scan_source_dtype_manifest(str(tmp_path), profile)
    assert manifest["model.layers.0.mlp.experts.gate_up_proj"] == "bf16"
    assert manifest["model.layers.0.mlp.experts.down_proj"] == "bf16"
    assert manifest[
        "model.layers.0.mlp.experts.1.gate_proj"
    ] == "bf16"
    assert manifest["model.layers.0.mlp.gate"] == "bf16"


def test_bf16_source_masks_only_true_passthroughs(census):
    packed_shape = (
        census["experts"],
        2 * census["expert_intermediate"],
        census["hidden"],
    )
    assert check_format_applicability(
        packed_shape, "BF16", source_kind="bf16",
        target_profile="vllm_packed_moe",
    ).legal
    for source_passthrough in ("FP8_SOURCE", "MXFP4_SOURCE"):
        verdict = check_format_applicability(
            packed_shape, source_passthrough, source_kind="bf16",
            target_profile="research",
        )
        assert not verdict.legal
        assert verdict.reason == "source_dtype_mismatch"


def test_reencode_formats_remain_legal_on_bf16_source(census):
    packed_shape = (
        census["experts"],
        2 * census["expert_intermediate"],
        census["hidden"],
    )
    # The third case was `("FP8_CB_K28", "nvfp4_cb")` until 2026-09-02. It is
    # deleted rather than re-pointed: no surviving serving profile offers a CB
    # rung (`nvfp4_cb.json` was the only one, now in
    # archive/gridbook_lane_2026-09-02/), and running the rung against
    # `vllm_packed_moe` -- which denies it -- would have inverted the
    # assertion's meaning. The CB FormatSpecs are still in the registry as
    # debt D34; what has no subject here is a *profile* that admits them.
    for fmt, target in (
        ("FP8_E4M3", "vllm_packed_moe"),
        ("MXFP8_E4M3", "vllm_packed_moe"),
    ):
        verdict = check_format_applicability(
            packed_shape, fmt, source_kind="bf16", target_profile=target,
        )
        assert verdict.legal, (fmt, verdict.reason, verdict.detail)


def test_census_fixture_exposes_probe_capture_points(profile):
    class TinyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.empty(2, 4, 3))
            self.down_proj = nn.Parameter(torch.empty(2, 3, 2))

    class TinyMoe(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(3, 2, bias=False)
            self.experts = TinyExperts()

    moe = TinyMoe()
    assert isinstance(moe.experts, nn.Module)
    assert tuple(moe.experts.gate_up_proj.shape) == (2, 4, 3)
    assert tuple(moe.experts.down_proj.shape) == (2, 3, 2)
    assert not hasattr(moe, "shared_expert")
    assert profile.packed_expert_role_group(
        "model.layers.0.mlp.experts.gate_up_proj") == "gate_up_proj"
    assert profile.packed_expert_role_group(
        "model.layers.0.mlp.experts.down_proj") == "down_proj"
