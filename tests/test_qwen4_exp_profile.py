"""Qwen3.8-Flash-Next / Qwen4-Exp producer-profile contract.

The census values and checkpoint names are from the official
``Qwen/Qwen3.8-Flash-Next`` config and safetensors index.  Runtime namespace
and HC packing assertions mirror vLLM PR #53896 at head ``d4d0f73``.
"""
from __future__ import annotations

import json
import re

import pytest

from prismaquant.export_native_compressed import build_quantization_config
from prismaquant.model_profiles import detect_profile, profile_from_config
from prismaquant.model_profiles.base import ModelProfile
from prismaquant.model_profiles.qwen4_exp import Qwen4ExpProfile
from prismaquant.model_profiles.structure import load_structure_spec
from prismaquant.serving_profiles import require_lane_supported


OFFICIAL_CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "language_model_only": False,
    "vision_config": {"depth": 27},
    "text_config": {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "ple_layer_ids": [2],
        "mtp_num_hidden_layers": 1,
    },
}


def test_official_config_detects_dedicated_profile(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(OFFICIAL_CONFIG))

    profile = detect_profile(str(tmp_path))

    assert isinstance(profile, Qwen4ExpProfile)
    assert profile.name == "qwen4_exp"
    assert profile.vllm_architecture_class() == (
        "Qwen4ExpForConditionalGeneration"
    )
    assert Qwen4ExpProfile.matches(
        "", ["Qwen4ExpForConditionalGeneration"]
    )
    assert not Qwen4ExpProfile.matches(
        "qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]
    )


def test_multimodal_source_recipe_and_vllm_names_round_trip():
    profile = profile_from_config(OFFICIAL_CONFIG)
    cases = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        ),
        "model.language_model.layers.1.ple.key_proj.weight": (
            "model.layers.1.ple.key_proj.weight",
            "language_model.model.layers.1.ple.key_proj.weight",
        ),
        "model.language_model.layers.47.mlp.experts.gate_up_proj": (
            "model.layers.47.mlp.experts.gate_up_proj",
            "language_model.model.layers.47.mlp.experts.gate_up_proj",
        ),
    }
    for source_name, (recipe_name, vllm_name) in cases.items():
        assert profile.checkpoint_to_live_name(source_name) == recipe_name
        assert profile.source_tensor_name(recipe_name) == source_name
        assert profile.export_tensor_name(recipe_name) == source_name
        assert profile.to_vllm_internal_name(recipe_name) == vllm_name

    assert profile.live_to_recipe_name(
        "model.language_model.layers.1.ple.value_proj"
    ) == "model.layers.1.ple.value_proj"
    assert profile.to_vllm_internal_name("lm_head") == (
        "language_model.lm_head"
    )
    assert profile.source_tensor_name("model.visual.blocks.0.attn.qkv.weight") == (
        "model.visual.blocks.0.attn.qkv.weight"
    )
    assert profile.to_vllm_internal_name(
        "model.visual.blocks.0.attn.qkv"
    ) == "visual.blocks.0.attn.qkv"


def test_text_only_variant_uses_text_runtime_namespace():
    profile = profile_from_config({
        "model_type": "qwen4_exp_text",
        "architectures": ["Qwen4ExpForCausalLM"],
    })

    name = "model.layers.0.self_attn.q_proj.weight"
    assert isinstance(profile, Qwen4ExpProfile)
    assert profile.vllm_architecture_class() == "Qwen4ExpForCausalLM"
    assert profile.live_to_recipe_name(name) == name
    assert profile.source_tensor_name(name) == name
    assert profile.to_vllm_internal_name(name) == name
    assert profile.to_vllm_internal_name("lm_head") == "lm_head"


@pytest.mark.parametrize(
    ("member", "target"),
    [
        ("self_attn.q_proj", "self_attn.qkv_proj"),
        ("self_attn.k_proj", "self_attn.qkv_proj"),
        ("self_attn.v_proj", "self_attn.qkv_proj"),
        ("mlp.shared_expert.gate_proj", "mlp.shared_expert.gate_up_proj"),
        ("mlp.shared_expert.up_proj", "mlp.shared_expert.gate_up_proj"),
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkvz"),
        ("linear_attn.in_proj_z", "linear_attn.in_proj_qkvz"),
        ("linear_attn.in_proj_b", "linear_attn.in_proj_ba"),
        ("linear_attn.in_proj_a", "linear_attn.in_proj_ba"),
        (
            "attn_hyper_connection.input_mix_weight_down",
            "attn_hyper_connection.input_mix_weight_down_block_inject",
        ),
        (
            "attn_hyper_connection.block_inject_weight",
            "attn_hyper_connection.input_mix_weight_down_block_inject",
        ),
        (
            "mlp_hyper_connection.input_mix_weight_down",
            "mlp_hyper_connection.input_mix_weight_down_block_inject",
        ),
        (
            "mlp_hyper_connection.block_inject_weight",
            "mlp_hyper_connection.input_mix_weight_down_block_inject",
        ),
    ],
)
def test_fused_groups_match_dedicated_vllm_runtime(member, target):
    profile = Qwen4ExpProfile()
    prefix = "model.layers.7."
    assert profile.fused_sibling_group(prefix + member) == prefix + target


def test_hyperconnection_up_projection_is_not_part_of_runtime_merge():
    profile = Qwen4ExpProfile()
    assert profile.fused_sibling_group(
        "model.layers.0.attn_hyper_connection.input_mix_weight_up"
    ) is None
    assert profile.fused_sibling_leaf_mapping()[
        "input_mix_weight_down_block_inject"
    ] == ("input_mix_weight_down", "block_inject_weight")


def test_bf16_hyperconnection_members_ignore_the_runtime_fused_module():
    profile = Qwen4ExpProfile()
    assignment = {
        "model.layers.0.attn_hyper_connection.input_mix_weight_down": "BF16",
        "model.layers.0.attn_hyper_connection.block_inject_weight": "BF16",
        "model.layers.0.linear_attn.out_proj": "NVFP4",
    }

    config = build_quantization_config(
        assignment, set(), profile=profile
    )

    assert (
        "language_model.model.layers.0.attn_hyper_connection."
        "input_mix_weight_down_block_inject"
    ) in config["ignore"]


def test_packed_experts_keep_source_decomposition_and_serving_coupling():
    profile = Qwen4ExpProfile()
    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj", "down_proj"
    })
    assert profile.packed_expert_module_class_names() == frozenset({
        "Qwen4ExpTextExperts"
    })
    assert profile.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj", "up_proj"
    )
    assert profile.packed_expert_projection_names("down_proj") == (
        "down_proj",
    )
    assert profile.split_packed_experts_for_format("NVFP4")
    assert profile.split_packed_experts_for_format("BF16")

    packed = "model.layers.0.mlp.experts.gate_up_proj"
    split = "model.layers.0.mlp.experts.511.gate_proj"
    assert profile.packed_expert_format_group(packed) == (
        "model.layers.0.mlp.experts::__packed_format__:"
        "gate_up_proj,down_proj"
    )
    assert profile.packed_expert_format_group(split) == (
        "model.layers.0.mlp.experts::__packed_format__:"
        "gate_proj,up_proj,down_proj"
    )
    regex = re.compile(profile.per_expert_moe_regex().removeprefix("re:"))
    assert regex.fullmatch(
        "language_model.model.layers.47.mlp.experts.511.down_proj"
    )
    assert regex.fullmatch("model.layers.0.mlp.experts.0.gate_proj")


def test_vision_ple_hyperconnection_and_mtp_are_preserved_safely():
    profile = Qwen4ExpProfile()
    assert profile.source_passthrough_prefixes() == ("model.visual.", "mtp.")
    assert profile.visual_config_key() == "vision_config"
    assert profile.visual_layer_prefix() == "model.visual.blocks"

    pinned = (
        "lm_head",
        "model.layers.1.ple.key_proj",
        "model.layers.1.ple.value_proj.weight",
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down",
        "model.layers.0.attn_hyper_connection.input_mix_weight_up",
        "model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "model.layers.0.mlp_hyper_connection.input_mix_weight_down",
        "model.hyper_connection_mixer.input_mix_weight_up",
    )
    assert all(profile.is_pinned_name(name) for name in pinned)
    assert not profile.is_pinned_name(
        "model.layers.0.linear_attn.out_proj.weight"
    )
    assert not profile.is_pinned_name(
        "model.layers.0.mlp.experts.gate_up_proj"
    )

    # The config advertises one MTP layer, but PrismaQuant intentionally does
    # not route it through Qwen3.5's incompatible synthetic module.
    assert profile.mtp_layer_count(OFFICIAL_CONFIG) == 1
    assert not profile.has_mtp()
    assert type(profile).build_mtp_module is ModelProfile.build_mtp_module
    assert profile.source_tensor_name("mtp.fc_hidden.weight") == (
        "mtp.fc_hidden.weight"
    )
    assert profile.checkpoint_to_live_name("mtp.fc_hidden.weight") is None


def test_staging_and_lane_declarations_fail_closed():
    profile = Qwen4ExpProfile()
    assert profile.stage_text_only_promote_inner_model_type()
    assert "vision_config" in profile.stage_text_only_strip_keys()
    assert profile.supported_export_lanes() == ("compressed-tensors",)
    assert profile.preferred_export_lane() == "compressed-tensors"
    assert require_lane_supported(profile, "compressed-tensors") == (
        "compressed-tensors"
    )
    with pytest.raises(SystemExit, match="qwen4_exp"):
        require_lane_supported(profile, "nvfp4_cb")
    with pytest.raises(SystemExit, match="qwen4_exp"):
        require_lane_supported(profile, "gguf")


def test_declarative_spec_matches_executable_profile():
    spec = load_structure_spec("qwen4_exp")
    assert spec is not None
    assert spec.id == "qwen4_exp"
    assert spec.priority == Qwen4ExpProfile.priority
    assert spec.match.claims(
        OFFICIAL_CONFIG["model_type"], OFFICIAL_CONFIG["architectures"]
    )
    assert spec.supported_lanes == ("compressed-tensors",)
    assert spec.packed_experts.param_names == ("gate_up_proj", "down_proj")
