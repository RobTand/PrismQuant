"""Qwen3.5-family MoE producer-profile contracts.

The first census is transcribed from the official Qwen3.6 wrapper.  The second
is transcribed from the official ``Qwen/Qwen3.8-2.4T-A95B`` config and
safetensors index and is deliberately only a namespace/shape-family proxy for
tomorrow's release: it does not assert that an unreleased 125B checkpoint has
the 2.4T model's dimensions or tensor census.
"""
from __future__ import annotations

import json
import re

import pytest

from prismaquant.model_profiles.registry import detect_profile, profile_from_config
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile


@pytest.fixture
def qwen36_35b_census():
    return {
        "repo_id": "Qwen/Qwen3.6-35B-A3B",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_model_type": "qwen3_5_moe_text",
        "layers": 40,
        "linear_attention_layers": 30,
        "full_attention_layers": 10,
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "expert_intermediate": 512,
        "shared_expert_intermediate": 512,
        "mtp_layers": 1,
        "real_checkpoint_names": (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.39.mlp.experts.down_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj",
            "model.language_model.layers.39.mlp.shared_expert.down_proj",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.38.linear_attn.out_proj",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.39.self_attn.o_proj",
        ),
    }


@pytest.fixture
def qwen38_2p4t_native_proxy():
    """Official released facts used only to prove native-causal intake."""
    return {
        "repo_id": "Qwen/Qwen3.8-2.4T-A95B",
        "proxy_only": True,
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "layers": 92,
        "experts": 512,
        "top_k": 10,
        "hidden": 8192,
        "expert_intermediate": 2048,
        "shared_expert_intermediate": 2048,
        "mtp_layers": 1,
        "real_checkpoint_names": (
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
            "model.layers.91.post_attention_layernorm.weight",
            "lm_head.weight",
        ),
    }


def test_qwen36_moe_profile_keeps_packed_expert_units():
    profile = profile_from_config({
        "model_type": "qwen3_6_moe_text",
        "architectures": ["Qwen3_6MoeForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5Profile)
    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert profile.to_vllm_internal_name(
        "model.layers.0.mlp.experts.gate_up_proj"
    ) == "model.layers.0.mlp.experts.gate_up_proj"
    assert profile.per_expert_moe_regex() is not None


def test_qwen38_2p4t_proxy_selects_native_causal_profile_and_class(
    qwen38_2p4t_native_proxy,
):
    c = qwen38_2p4t_native_proxy
    profile = profile_from_config({
        "model_type": c["model_type"],
        "architectures": c["architectures"],
    })

    assert c["proxy_only"] is True
    assert isinstance(profile, Qwen3_5Profile)
    assert profile.name == "qwen3_5"
    assert profile.vllm_architecture_class() == "Qwen3_5MoeForCausalLM"


def test_qwen38_2p4t_native_index_names_stay_direct_in_every_namespace(
    qwen38_2p4t_native_proxy,
):
    c = qwen38_2p4t_native_proxy
    profile = profile_from_config({
        "model_type": c["model_type"],
        "architectures": c["architectures"],
    })

    for checkpoint_name in c["real_checkpoint_names"]:
        live_name = profile.checkpoint_to_live_name(checkpoint_name)
        assert live_name == checkpoint_name
        assert profile.live_to_recipe_name(live_name) == checkpoint_name
        assert profile.source_tensor_name(live_name) == checkpoint_name
        assert profile.export_tensor_name(live_name) == checkpoint_name
        assert profile.to_vllm_internal_name(live_name) == checkpoint_name


def test_qwen38_2p4t_proxy_path_census_confirms_direct_source(
    tmp_path,
    qwen38_2p4t_native_proxy,
):
    c = qwen38_2p4t_native_proxy
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": c["model_type"],
        "architectures": c["architectures"],
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {name: "model-00001-of-00001.safetensors"
                       for name in c["real_checkpoint_names"]},
    }))

    profile = detect_profile(str(tmp_path))
    recipe = "model.layers.0.mlp.experts.gate_up_proj"
    assert profile.source_tensor_name(recipe) == recipe
    assert profile.to_vllm_internal_name(recipe) == recipe


def test_staged_causal_config_retains_wrapper_source_but_direct_vllm_namespace(
    tmp_path,
):
    """Architecture and source layout are independent for internal stages.

    ``stage_text_only`` changes the config entrypoint but symlinks the wrapper
    index. The source census must win for weight lookup; the causal entrypoint
    must still win for the module tree that vLLM builds.
    """
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                "model-00001-of-00001.safetensors"
            ),
        },
    }))

    profile = detect_profile(str(tmp_path))
    recipe = "model.layers.0.mlp.experts.gate_up_proj"
    assert profile.vllm_architecture_class() == "Qwen3_5MoeForCausalLM"
    assert profile.source_tensor_name(recipe) == (
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    assert profile.to_vllm_internal_name(recipe) == recipe


def test_causal_qwen_moe_mixed_source_index_fails_closed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {
            "model.layers.0.mlp.experts.gate_up_proj": "a.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj": (
                "b.safetensors"
            ),
        },
    }))

    profile = detect_profile(str(tmp_path))
    with pytest.raises(RuntimeError, match="contains both"):
        profile.source_tensor_name(
            "model.layers.0.mlp.experts.gate_up_proj"
        )


def test_qwen_moe_spec_specializes_source_and_vllm_maps_together():
    from prismaquant.model_profiles.structure import load_structure_spec

    spec = load_structure_spec("qwen3_5")
    assert spec is not None
    native = spec.for_config(
        "qwen3_5_moe_text", ["Qwen3_5MoeForCausalLM"]
    )
    wrapper = spec.for_config(
        "qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]
    )

    recipe = "model.layers.7.mlp.experts.gate_up_proj"
    assert native.rewrite_recipe_to_source(recipe) == recipe
    assert native.rewrite_recipe_to_vllm(recipe) == recipe
    assert wrapper.rewrite_recipe_to_source(recipe) == (
        "model.language_model.layers.7.mlp.experts.gate_up_proj"
    )
    assert wrapper.rewrite_recipe_to_vllm(recipe) == (
        "language_model.model.layers.7.mlp.experts.gate_up_proj"
    )


def test_qwen36_35b_official_config_resolves_to_gridbook_producer_id(
    qwen36_35b_census,
):
    profile = profile_from_config({
        "model_type": qwen36_35b_census["model_type"],
        "architectures": qwen36_35b_census["architectures"],
    })

    assert isinstance(profile, Qwen3_5Profile)
    assert profile.name == "qwen3_5"
    assert profile.vllm_architecture_class() == (
        "Qwen3_5MoeForConditionalGeneration"
    )
    assert profile.supported_export_lanes() == (
        "compressed-tensors",
        "nvfp4_cb",
    )


def test_qwen36_35b_real_census_names_round_trip(qwen36_35b_census):
    profile = Qwen3_5Profile()
    for checkpoint_name in qwen36_35b_census["real_checkpoint_names"]:
        live_name = profile.checkpoint_to_live_name(checkpoint_name)
        assert live_name == checkpoint_name.replace(
            "model.language_model.", "model.", 1
        )
        assert profile.source_tensor_name(live_name) == checkpoint_name
        assert profile.export_tensor_name(live_name) == checkpoint_name
        assert profile.to_vllm_internal_name(live_name) == (
            checkpoint_name.replace(
                "model.language_model.", "language_model.model.", 1
            )
        )


def test_qwen36_35b_census_pins_packed_and_shared_expert_shapes(
    qwen36_35b_census,
):
    c = qwen36_35b_census
    assert (c["experts"], 2 * c["expert_intermediate"], c["hidden"]) == (
        256,
        1024,
        2048,
    )
    assert (c["experts"], c["hidden"], c["expert_intermediate"]) == (
        256,
        2048,
        512,
    )
    assert (c["shared_expert_intermediate"], c["hidden"]) == (512, 2048)
    assert (c["hidden"], c["shared_expert_intermediate"]) == (2048, 512)

    profile = Qwen3_5Profile()
    assert profile.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert profile.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj",
        "up_proj",
    )
    assert profile.packed_expert_projection_names("down_proj") == (
        "down_proj",
    )


def test_qwen36_35b_per_expert_vllm_names_cover_the_full_census(
    qwen36_35b_census,
):
    c = qwen36_35b_census
    names = [
        "language_model.model.layers."
        f"{layer}.mlp.experts.{expert}.{projection}_proj"
        for layer in range(c["layers"])
        for expert in range(c["experts"])
        for projection in ("gate", "up", "down")
    ]
    regex = re.compile(
        Qwen3_5Profile().per_expert_moe_regex().removeprefix("re:")
    )
    assert len(names) == 40 * 256 * 3 == 30_720
    assert all(regex.fullmatch(name) for name in names)


@pytest.mark.parametrize("name", [
    "model.layers.0.mlp.experts.0.gate_proj",
    "model.layers.91.mlp.experts.511.up_proj",
    "language_model.model.layers.0.mlp.experts.0.down_proj",
    "language_model.model.layers.39.mlp.experts.255.gate_proj",
])
def test_qwen_moe_per_expert_regex_covers_native_and_wrapper_vllm_names(name):
    regex = re.compile(
        Qwen3_5Profile().per_expert_moe_regex().removeprefix("re:")
    )
    assert regex.fullmatch(name)


@pytest.mark.parametrize("name", [
    "model.language_model.layers.0.mlp.experts.0.gate_proj",
    "layers.0.mlp.experts.0.gate_proj",
    "model.layers.0.mlp.shared_expert.gate_proj",
])
def test_qwen_moe_per_expert_regex_rejects_non_vllm_or_shared_names(name):
    regex = re.compile(
        Qwen3_5Profile().per_expert_moe_regex().removeprefix("re:")
    )
    assert not regex.fullmatch(name)


@pytest.mark.parametrize("architectures", [
    ["Qwen3_8MoeForCausalLM"],
    ["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
    [],
])
def test_declared_unknown_or_ambiguous_qwen_moe_layout_fails_closed(
    architectures,
):
    profile = profile_from_config({
        "model_type": "qwen3_5_moe_text",
        "architectures": architectures,
    })

    with pytest.raises(RuntimeError, match="architecture|mixed|namespace"):
        profile.vllm_architecture_class()
    with pytest.raises(RuntimeError, match="architecture|mixed|namespace"):
        profile.source_tensor_name("model.layers.0.mlp.experts.gate_up_proj")


def test_qwen36_35b_fused_groups_match_real_layout():
    profile = Qwen3_5Profile()
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.shared_expert.gate_proj"
    ) == "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.mlp.shared_expert.up_proj"
    ) == "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_qkv"
    ) == "model.layers.0.linear_attn.in_proj_qkvz"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_z"
    ) == "model.layers.0.linear_attn.in_proj_qkvz"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_b"
    ) == "model.layers.0.linear_attn.in_proj_ba"
    assert profile.fused_sibling_group(
        "model.layers.0.linear_attn.in_proj_a"
    ) == "model.layers.0.linear_attn.in_proj_ba"


def test_qwen36_dense_profile_still_wins_for_non_moe_arch():
    profile = profile_from_config({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_6ForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5DenseProfile)
    assert profile.packed_expert_param_names() == frozenset()
