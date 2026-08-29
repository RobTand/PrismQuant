from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
from prismaquant.model_profiles.gemma4 import Gemma4Profile
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.qwen3_5_dense import Qwen3_5DenseProfile
from prismaquant.model_profiles.registry import profile_from_config
from prismaquant.model_profiles.structure import (
    ModelStructureSpec,
    build_model_graph,
    load_structure_spec,
)


class _QwenMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _QwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _QwenMlp()


class _QwenToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_QwenLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class _Qwen3Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 4)
        self.model.layers = nn.ModuleList([_QwenLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 4))


class _QwenMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedExperts()


class _QwenMoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _QwenMoeBlock()


class _Qwen3MoeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_QwenMoeLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class _GemmaMoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = nn.Module()
        self.moe.experts = _PackedExperts()


class _GemmaToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_GemmaMoeLayer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


class _Dsv4Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wkv = nn.Linear(4, 4, bias=False)


class _Dsv4Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 4))


class _Dsv4Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Dsv4Experts()


class _Dsv4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Dsv4Attn()
        self.mlp = _Dsv4Mlp()


class _Dsv4Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Dsv4Layer()])
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_qwen_structure_spec_matches_profile_naming():
    spec = load_structure_spec("qwen3_5")
    profile = Qwen3_5Profile()

    assert spec is not None
    live = "model.language_model.layers.0.mlp.gate_proj"
    recipe = "model.layers.0.mlp.gate_proj"
    assert spec.rewrite_live_to_recipe(live) == profile.live_to_recipe_name(live)
    assert spec.rewrite_live_to_recipe(live) == recipe
    assert profile.source_tensor_name(recipe) == (
        "model.language_model.layers.0.mlp.gate_proj"
    )
    assert profile.export_tensor_name(recipe) == (
        "model.language_model.layers.0.mlp.gate_proj"
    )
    assert profile.source_tensor_name(live) == live
    assert spec.rewrite_recipe_to_vllm(recipe) == (
        "language_model.model.layers.0.mlp.gate_proj"
    )
    assert spec.rewrite_recipe_to_vllm("model.visual.blocks.0.attn.proj") == (
        "visual.blocks.0.attn.proj"
    )
    assert spec.rewrite_recipe_to_vllm("mtp.layers.0.mlp.gate_proj") == (
        "mtp.layers.0.mlp.gate_proj"
    )
    # Both source layouts are real for this family: a staged text-only
    # checkpoint keeps its wrapper-named index (`language_model.model.`) even
    # when its config declares the native causal class (`model.`), so the
    # regex must match either. `specs/qwen3_5.json` `moe.per_expert_regex` is
    # the declared form; this literal is the independent check on it.
    assert profile.per_expert_moe_regex() == (
        r"re:^(?:model|language_model[.]model)[.]layers[.][0-9]+[.]mlp"
        r"[.]experts[.][0-9]+[.](gate|up|down)_proj$"
    )
    assert profile.per_expert_mtp_regex() == (
        r"re:^mtp[.]layers[.][0-9]+[.]mlp[.]experts[.][0-9]+"
        r"[.](gate|up|down)_proj$"
    )
    assert profile.pinned_names() == ("lm_head",)
    assert profile.is_pinned_name("lm_head")
    assert profile.is_pinned_name("language_model.lm_head")
    assert profile.stage_text_only_strip_keys() == (
        "vision_config",
        "audio_config",
        "speech_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    )
    assert profile.stage_text_only_promote_inner_model_type() is True
    assert profile.visual_config_key() == "vision_config"
    assert profile.visual_layer_prefix() == "model.visual.blocks"


def test_default_profile_common_fused_groups_are_profile_owned():
    profile = DefaultProfile()

    assert profile.fused_sibling_leaf_mapping()["qkv_proj"] == (
        "q_proj",
        "k_proj",
        "v_proj",
    )
    assert profile.fused_sibling_leaf_mapping()["in_proj_ba"] == (
        "in_proj_b",
        "in_proj_a",
    )
    assert (
        profile.fused_sibling_group("model.layers.0.self_attn.q_proj")
        == "model.layers.0.self_attn.qkv_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.mlp.gate_proj")
        == "model.layers.0.mlp.gate_up_proj"
    )
    assert (
        profile.fused_sibling_group("model.layers.0.linear_attn.in_proj_b")
        == "model.layers.0.linear_attn.in_proj_ba"
    )


def test_qwen_model_graph_records_recipe_group_and_pinned_head():
    profile = Qwen3_5Profile()
    graph = profile.build_model_graph(_QwenToy())
    by_recipe = graph.by_recipe_name()

    gate = by_recipe["model.layers.0.mlp.gate_proj.weight"]
    assert gate.live_name == "model.language_model.layers.0.mlp.gate_proj.weight"
    assert gate.vllm_name == "language_model.model.layers.0.mlp.gate_proj.weight"
    assert gate.block == "model.layers.0"
    assert gate.group == "model.layers.0.mlp.gate_up_proj"
    assert gate.quantizable is True
    assert "fused_sibling_format" in gate.constraints

    head = by_recipe["lm_head.weight"]
    assert head.pinned is True
    assert head.quantizable is False


def test_qwen_graph_exposes_fused_sibling_optimization_units():
    graph = Qwen3_5Profile().build_model_graph(_QwenToy())
    units = {unit.id: unit for unit in graph.optimization_units()}

    fused = units["fused:model.layers.0.mlp.gate_up_proj"]
    assert fused.scope == "fused_sibling_group"
    assert fused.members == (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    assert "fused_sibling_format" in fused.constraints

    down = units["tensor:model.layers.0.mlp.down_proj.weight"]
    assert down.scope == "tensor"


def test_qwen3_dense_and_moe_configs_share_contract_profile():
    dense = profile_from_config({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    })
    moe = profile_from_config({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
    })

    assert isinstance(dense, Qwen3Profile)
    assert isinstance(moe, Qwen3Profile)
    assert dense.structure_spec().id == "qwen3"
    assert moe.structure_spec().id == "qwen3"
    assert dense.serving_profile_id() == "vllm_packed_moe"
    assert moe.serving_profile_id() == "vllm_packed_moe"
    # Family-wide declarations are inert on dense modules, which have no 3-D
    # packed expert Parameters.
    assert dense.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert moe.packed_expert_param_names() == frozenset({
        "gate_up_proj",
        "down_proj",
    })
    assert moe.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj",
        "up_proj",
    )
    assert moe.packed_expert_parent_for_projection("up_proj") == "gate_up_proj"
    assert moe.split_packed_experts_for_format("BF16") is True
    assert moe.per_expert_moe_regex() == (
        r"re:^model[.]layers[.][0-9]+[.]mlp[.]experts[.][0-9]+"
        r"[.](gate|up|down)_proj$"
    )
    packed_group = moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.gate_up_proj"
    )
    assert packed_group == moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.down_proj"
    )
    split_group = moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.gate_proj"
    )
    assert split_group == moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.up_proj"
    )
    assert split_group == moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.down_proj"
    )
    assert split_group == moe.packed_expert_format_group(
        "model.layers.0.mlp.experts.9.down_proj"
    )
    assert split_group != packed_group


def test_qwen3_dense_graph_marks_linears_and_fused_groups():
    graph = Qwen3Profile().build_model_graph(_Qwen3Toy())
    by_recipe = graph.by_recipe_name()

    embed = by_recipe["model.embed_tokens.weight"]
    assert embed.role == "embedding_weight"
    assert embed.quantizable is False

    gate = by_recipe["model.layers.0.mlp.gate_proj.weight"]
    assert gate.live_name == "model.layers.0.mlp.gate_proj.weight"
    assert gate.vllm_name == "model.layers.0.mlp.gate_proj.weight"
    assert gate.group == "model.layers.0.mlp.gate_up_proj"
    assert gate.role == "linear_weight"
    assert gate.quantizable is True


def test_qwen3_moe_graph_marks_packed_experts_without_multimodal_rewrite():
    graph = Qwen3Profile().build_model_graph(_Qwen3MoeToy())
    by_recipe = graph.by_recipe_name()

    packed = by_recipe["model.layers.0.mlp.experts.gate_up_proj"]
    assert packed.live_name == "model.layers.0.mlp.experts.gate_up_proj"
    assert packed.vllm_name == "model.layers.0.mlp.experts.gate_up_proj"
    assert packed.role == "packed_expert_weight"
    assert packed.quantizable is True

    head = by_recipe["lm_head.weight"]
    assert head.pinned is True
    assert head.quantizable is False

    units = {unit.id: unit for unit in graph.optimization_units()}
    packed_unit = units["packed_expert:model.layers.0.mlp.experts"]
    assert packed_unit.scope == "packed_expert_group"
    assert packed_unit.members == (
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    )
    assert "packed_expert_decomposition" in packed_unit.constraints


def test_probe_packed_expert_detection_respects_profile_spec():
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    experts = _PackedExperts()

    assert _is_packed_experts_module(experts, Qwen3Profile()) is True
    assert _packed_experts_param_names(experts, Qwen3Profile()) == [
        "down_proj",
        "gate_up_proj",
    ]


def test_packed_expert_format_group_uses_declared_projection_splits():
    spec = ModelStructureSpec.from_dict({
        "schema": "prismaquant.model_structure.v1",
        "id": "custom",
        "packed_experts": {
            "param_names": ["w13", "w2"],
            "projection_splits": {
                "w13": ["w1_proj", "w3_proj"],
            },
            "format_groups": [
                ["w13", "w2"],
                ["w1_proj", "w3_proj", "w2"],
            ],
        },
    })

    packed_group = spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.w13"
    )
    assert packed_group == spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.w2"
    )
    split_group = spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.w1_proj"
    )
    assert split_group == spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.w3_proj"
    )
    assert split_group == spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.7.w2"
    )
    assert split_group == spec.packed_expert_format_group(
        "model.layers.0.mlp.experts.9.w2"
    )
    assert split_group != packed_group


def test_qwen35_dense_profile_uses_dense_structure_spec():
    profile = profile_from_config({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_6ForCausalLM"],
    })

    assert isinstance(profile, Qwen3_5DenseProfile)
    assert profile.structure_spec().id == "qwen3_5_dense"
    assert profile.serving_profile_id() == "vllm_packed_moe"
    assert profile.packed_expert_param_names() == frozenset()
    assert profile.per_expert_moe_regex() is None
    # `*ForCausalLM` is the TEXT-ONLY class, which vLLM builds under `model.`
    # — this assertion used to read `language_model.model.…` because the spec
    # had a single naming block written for the multimodal wrapper. The
    # declared architecture decides; see tests/test_qwen3_5_text_only_namespace.py.
    assert profile.to_vllm_internal_name("model.layers.0.mlp.gate_proj") == (
        "model.layers.0.mlp.gate_proj"
    )

    wrapper = profile_from_config({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
    })
    assert wrapper.structure_spec().id == "qwen3_5_dense"
    assert wrapper.to_vllm_internal_name("model.layers.0.mlp.gate_proj") == (
        "language_model.model.layers.0.mlp.gate_proj"
    )


def test_qwen36_model_type_aliases_route_dense_and_moe_profiles():
    # The subject here is model_type alias routing, which is orthogonal to
    # namespace selection -- so declare a real architecture. A *declared*
    # config with no architecture is refused on purpose (it is not evidence
    # for either namespace); that refusal is asserted separately below.
    dense = profile_from_config({
        "model_type": "qwen3_6",
        "architectures": ["Qwen3_6ForCausalLM"],
    })
    moe = profile_from_config({
        "model_type": "qwen3_6_moe",
        "architectures": ["Qwen3_6MoeForConditionalGeneration"],
    })

    assert isinstance(dense, Qwen3_5DenseProfile)
    assert dense.structure_spec().id == "qwen3_5_dense"
    assert isinstance(moe, Qwen3_5Profile)
    assert not isinstance(moe, Qwen3_5DenseProfile)
    assert moe.structure_spec().id == "qwen3_5"

    # Fail closed rather than inheriting the historical wrapper mapping: an
    # alias-routed MoE config that declares no architecture cannot say whether
    # its tensors are `model.` or `language_model.model.`.
    ambiguous = profile_from_config({
        "model_type": "qwen3_6_moe",
        "architectures": [],
    })
    with pytest.raises(RuntimeError, match="declares no architecture"):
        ambiguous.structure_spec()


def test_gemma_structure_collapses_live_moe_and_injects_vllm_moe_prefix():
    profile = Gemma4Profile()
    spec = profile.structure_spec()

    assert spec is not None
    live = "model.language_model.layers.0.moe.experts.gate_up_proj"
    recipe = "model.layers.0.experts.gate_up_proj"
    assert profile.live_to_recipe_name(live) == recipe
    assert spec.rewrite_live_to_recipe(live) == recipe
    assert profile.source_tensor_name(recipe) == (
        "model.language_model.layers.0.moe.experts.gate_up_proj"
    )
    assert profile.export_tensor_name(recipe) == recipe
    assert profile.source_tensor_name(live) == live
    assert profile.to_vllm_internal_name(recipe) == (
        "language_model.model.layers.0.moe.experts.gate_up_proj"
    )
    assert profile.serving_profile_id() == "vllm_packed_moe"
    assert profile.packed_expert_format_group(recipe) == (
        profile.packed_expert_format_group("model.layers.0.experts.down_proj")
    )
    assert profile.packed_expert_projection_names("gate_up_proj") == (
        "gate_proj",
        "up_proj",
    )
    assert profile.packed_expert_parent_for_projection("gate_proj") == (
        "gate_up_proj"
    )
    assert profile.packed_expert_format_group(
        "model.layers.0.experts.3.gate_proj"
    ) == profile.packed_expert_format_group(
        "model.layers.0.experts.3.down_proj"
    )
    assert profile.packed_expert_format_group(
        "model.layers.0.experts.3.gate_proj"
    ) == profile.packed_expert_format_group(
        "model.layers.0.experts.9.down_proj"
    )
    assert profile.source_passthrough_prefixes() == (
        "model.vision_tower.",
        "model.audio_tower.",
        "model.embed_vision.",
        "model.embed_audio.",
    )
    assert profile.stage_text_only_strip_keys() == (
        "vision_config",
        "audio_config",
        "speech_config",
        "image_token_id",
        "video_token_id",
        "audio_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    )
    assert profile.stage_text_only_promote_inner_model_type() is True
    assert profile.visual_config_key() == "vision_config"
    assert profile.visual_layer_prefix() == (
        "model.vision_tower.vision_model.encoder.layers"
    )


def test_gemma4_shared_kv_pass_state_uses_layer_indexes():
    profile = Gemma4Profile()
    key = torch.randn(1, 2, 3, requires_grad=True)
    value = torch.randn(1, 2, 3, requires_grad=True)

    captured = profile.capture_forward_pass_state({
        "shared_kv_states": {3: (key, value)},
    })

    assert set(captured) == {3}
    assert captured[3][0].device.type == "cpu"
    assert captured[3][1].device.type == "cpu"
    assert captured[3][0].requires_grad is False
    assert captured[3][1].requires_grad is False

    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            is_kv_shared_layer=True,
            kv_shared_layer_index=3,
            layer_type="full_attention",
        )
    )
    pass_state = profile.isolated_layer_pass_state(captured, layer)

    assert set(pass_state["shared_kv_states"]) == {3}
    assert pass_state["shared_kv_states"][3] == captured[3]


def test_gemma4_shared_kv_capture_rejects_malformed_entries():
    profile = Gemma4Profile()

    try:
        profile.capture_forward_pass_state({"shared_kv_states": {3: object()}})
    except RuntimeError as exc:
        assert "shared_kv_states[3]" in str(exc)
    else:  # pragma: no cover - assert path keeps compatibility without pytest.raises
        raise AssertionError("malformed shared_kv_states entry was accepted")


def test_gemma_graph_marks_packed_experts_under_recipe_experts():
    graph = Gemma4Profile().build_model_graph(_GemmaToy())
    by_recipe = graph.by_recipe_name()

    packed = by_recipe["model.layers.0.experts.gate_up_proj"]
    assert packed.live_name == (
        "model.language_model.layers.0.moe.experts.gate_up_proj"
    )
    assert packed.vllm_name == (
        "language_model.model.layers.0.moe.experts.gate_up_proj"
    )
    assert packed.source_name == (
        "model.language_model.layers.0.moe.experts.gate_up_proj"
    )
    assert packed.export_name == "model.layers.0.experts.gate_up_proj"
    assert packed.role == "packed_expert_weight"
    assert packed.quantizable is True


def test_deepseek_structure_spec_matches_profile_source_naming():
    spec = load_structure_spec("deepseek_v4")
    profile = DeepseekV4Profile()

    assert spec is not None
    assert profile.pinned_names() == ("lm_head", "head")
    assert profile.is_pinned_name("head")
    assert profile.is_pinned_name("model.head")
    cases = {
        "lm_head.weight": "head.weight",
        "model.embed_tokens.weight": "embed.weight",
        "model.layers.0.self_attn.wkv.weight": "layers.0.attn.wkv.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight": (
            "layers.0.ffn.shared_experts.w1.weight"
        ),
        "model.layers.5.ffn_hc.scale": "layers.5.hc_ffn_scale",
    }
    for live, source in cases.items():
        assert spec.rewrite_recipe_to_source(live) == profile.source_tensor_name(live)
        assert spec.rewrite_recipe_to_source(live) == source


def test_deepseek_model_graph_marks_packed_experts():
    graph = build_model_graph(_Dsv4Toy(), DeepseekV4Profile())
    by_recipe = graph.by_recipe_name()

    wkv = by_recipe["model.layers.0.self_attn.wkv.weight"]
    assert wkv.source_name == "layers.0.attn.wkv.weight"
    assert wkv.role == "linear_weight"
    assert wkv.quantizable is True

    packed = by_recipe["model.layers.0.mlp.experts.gate_up_proj"]
    assert packed.role == "packed_expert_weight"
    assert packed.source_name == "layers.0.ffn.experts.gate_up_proj"
    assert packed.quantizable is True
    assert "packed_expert_decomposition" in packed.constraints
    profile = DeepseekV4Profile()
    assert profile.packed_expert_format_group(
        "model.layers.0.mlp.experts.gate_up_proj"
    ) == profile.packed_expert_format_group(
        "model.layers.0.mlp.experts.down_proj"
    )
