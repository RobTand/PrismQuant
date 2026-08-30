"""Unit tests for the DeepseekV4Profile naming bridge.

DSv4 has the most complex live↔checkpoint name remapping of any profile
(self_attn↔attn, mlp↔ffn, gate_proj/up_proj/down_proj↔w1/w3/w2 for
shared experts, hyper-connection compaction). This file pins the
expected behavior so renames stay correct as the rest of the pipeline
evolves.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile


@pytest.fixture
def profile() -> DeepseekV4Profile:
    return DeepseekV4Profile()


def test_match_by_model_type(profile):
    assert DeepseekV4Profile.matches("deepseek_v4", [])
    assert DeepseekV4Profile.matches("deepseek-v4", [])
    assert not DeepseekV4Profile.matches("deepseek_v3", [])
    assert not DeepseekV4Profile.matches("minimax_m2", [])


def test_match_by_architecture(profile):
    assert DeepseekV4Profile.matches("", ["DeepseekV4ForCausalLM"])
    assert DeepseekV4Profile.matches("", ["DeepSeek-V4-Custom"])
    assert not DeepseekV4Profile.matches("", ["DeepseekV3ForCausalLM"])


def test_layer_prefixes(profile):
    assert profile.body_layer_prefix() == "layers"
    assert profile.lm_head_name() == "head"
    assert profile.mtp_layer_prefix() == "mtp"


def test_packed_expert_param_names(profile):
    assert profile.packed_expert_param_names() == frozenset({"gate_up_proj", "down_proj"})


def test_source_passthrough_prefixes_are_spec_backed(profile):
    # `mtp.` covers the nextn block. DSv4 takes the hy_v3 route until its
    # MTP is actually quantized: `has_mtp() -> False` (so probe/cost/
    # allocator never see it, and `checkpoint_to_live_name` drops the
    # keys) + verbatim passthrough here so vLLM's nextn spec decode still
    # loads. Audit R12, 2026-07-30.
    assert profile.source_passthrough_prefixes() == (
        "mtp.",
        "attn_sink",
        "hc_",
        "compressor.ape",
        "tid2eid",
        "kv_norm",
        "q_norm",
        "norm.",
    )


def test_probe_skip_and_grouped_class_declarations_are_spec_backed(profile):
    DeepseekV4GroupedLinear = type(
        "DeepseekV4GroupedLinear",
        (nn.Linear,),
        {},
    )

    # The grouped-BMM class LEFT the skip list when the grouped Fisher
    # accumulator landed: it is priced now, so holding it at source
    # precision would be a choice, not a debt. The declaration moved to
    # `grouped_module_class_names`, which routes the probe to the grouped
    # accumulator instead of the dense one.
    assert profile.structure_spec().probe_skip_module_class_names == ()
    assert profile.structure_spec().probe_grouped_module_class_names == (
        "DeepseekV4GroupedLinear",
    )
    assert profile.should_probe_linear("model.layers.0.self_attn.wq_a", nn.Linear(4, 4))
    assert profile.should_probe_linear(
        "model.layers.0.self_attn.wo_a",
        DeepseekV4GroupedLinear(4, 4),
    )


def test_split_packed_for_export(profile):
    # Mirror Qwen3.5: always split per-expert for export safety.
    assert profile.split_packed_experts_for_format("nvfp4") is True
    assert profile.split_packed_experts_for_format("fp8_source") is True
    assert profile.split_packed_experts_for_format("bf16") is True


def test_top_level_name_bridge(profile):
    """`lm_head` / `embed_tokens` / `norm` rename to DSv4's flat keys."""
    assert profile.source_tensor_name("lm_head.weight") == "head.weight"
    assert profile.source_tensor_name("model.embed_tokens.weight") == "embed.weight"
    assert profile.source_tensor_name("model.norm.weight") == "norm.weight"


def test_decoder_layer_attn_bridge(profile):
    """`self_attn` → `attn`, including for nested compressor + indexer."""
    cases = [
        ("model.layers.0.self_attn.wkv.weight",
         "layers.0.attn.wkv.weight"),
        ("model.layers.5.self_attn.wq_a.weight",
         "layers.5.attn.wq_a.weight"),
        ("model.layers.42.self_attn.wq_b.weight",
         "layers.42.attn.wq_b.weight"),
        ("model.layers.0.self_attn.q_norm.weight",
         "layers.0.attn.q_norm.weight"),
        ("model.layers.0.self_attn.compressor.wkv.weight",
         "layers.0.attn.compressor.wkv.weight"),
        ("model.layers.0.self_attn.compressor.wgate.weight",
         "layers.0.attn.compressor.wgate.weight"),
        ("model.layers.0.self_attn.compressor.indexer.weights_proj.weight",
         "layers.0.attn.compressor.indexer.weights_proj.weight"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source, (
            f"{live} ↦ {profile.source_tensor_name(live)}, expected {source}")


def test_decoder_layer_mlp_bridge(profile):
    """`mlp` → `ffn`, including routed/shared experts."""
    cases = [
        ("model.layers.0.mlp.gate.weight", "layers.0.ffn.gate.weight"),
        ("model.layers.0.mlp.experts.gate_up_proj",
         "layers.0.ffn.experts.gate_up_proj"),
        ("model.layers.0.mlp.experts.down_proj",
         "layers.0.ffn.experts.down_proj"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_shared_expert_leaf_rename(profile):
    """Shared experts: gate_proj/up_proj/down_proj → w1/w3/w2 in source."""
    cases = [
        ("model.layers.0.mlp.shared_experts.gate_proj.weight",
         "layers.0.ffn.shared_experts.w1.weight"),
        ("model.layers.0.mlp.shared_experts.up_proj.weight",
         "layers.0.ffn.shared_experts.w3.weight"),
        ("model.layers.0.mlp.shared_experts.down_proj.weight",
         "layers.0.ffn.shared_experts.w2.weight"),
        # Same with .scale suffix (FP8 block scales)
        ("model.layers.5.mlp.shared_experts.gate_proj.scale",
         "layers.5.ffn.shared_experts.w1.scale"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_hyper_connection_compaction(profile):
    """`attn_hc.X` / `ffn_hc.X` → `hc_attn_X` / `hc_ffn_X` (flat keys)."""
    cases = [
        ("model.layers.0.attn_hc.base", "layers.0.hc_attn_base"),
        ("model.layers.0.attn_hc.fn",   "layers.0.hc_attn_fn"),
        ("model.layers.0.attn_hc.scale", "layers.0.hc_attn_scale"),
        ("model.layers.5.ffn_hc.base", "layers.5.hc_ffn_base"),
        ("model.layers.5.ffn_hc.scale", "layers.5.hc_ffn_scale"),
    ]
    for live, source in cases:
        assert profile.source_tensor_name(live) == source


def test_layernorm_passthrough(profile):
    """input_layernorm / post_attention_layernorm pass through unchanged
    (they're at the layer level, no rename needed beyond model-prefix strip)."""
    assert (profile.source_tensor_name("model.layers.0.input_layernorm.weight")
            == "layers.0.input_layernorm.weight")
    assert (profile.source_tensor_name("model.layers.0.post_attention_layernorm.weight")
            == "layers.0.post_attention_layernorm.weight")


def test_mtp_layer_count(profile):
    assert profile.mtp_layer_count({"num_nextn_predict_layers": 1}) == 1
    assert profile.mtp_layer_count({"num_nextn_predict_layers": 2}) == 2
    # Backward-compat key
    assert profile.mtp_layer_count({"num_mtp_layers": 1}) == 1
    # Missing → 0
    assert profile.mtp_layer_count({}) == 0


def test_gridbook_dense_role_composites_are_not_allocator_fused(profile):
    """Gridbook may decode merged Linear roles from different formats.

    The consumer constructs one semantic composite for a merged vLLM Linear,
    decodes each role independently to the common FP8 execution type, and then
    concatenates the outputs.  A global producer fused group would incorrectly
    force the same codebook format onto those independent roles.
    """
    assert profile.fused_sibling_group("model.layers.0.self_attn.wq_a") is None
    assert profile.fused_sibling_group("model.layers.0.self_attn.wkv") is None
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        assert profile.fused_sibling_group(
            f"model.layers.0.mlp.shared_experts.{leaf}"
        ) is None


def test_gridbook_cb_export_lane_is_declared(profile):
    assert profile.supported_export_lanes() == (
        "compressed-tensors", "nvfp4_cb"
    )
    assert profile.preferred_export_lane() == "compressed-tensors"


def test_rope_axis_mapping_matches_the_vendored_definition(profile):
    """The profile must ANSWER from the model, not restate its rule.

    `DeepseekV4Model.forward` picks a layer's rotary table by attention
    schedule; PrismaQuant's streamed driver bypasses that forward and must
    reach the same answer. When the streamed path had its own copy of the rule
    it silently fed `main` rope to all 41 compressed V4-Flash layers -- base
    10000 with YaRN off instead of 160000 with YaRN -- and the BF16 teacher
    built that way scored perplexity 262. So this pins that there is exactly
    one definition and both callers reach it.
    """
    import importlib
    import inspect

    # Import through the REGISTERED name -- the vendored package's `__init__`
    # is Transformers-relative and only resolves after registration.
    profile.register_vendored_modeling()
    modeling = importlib.import_module(
        "transformers.models.deepseek_v4.modeling_deepseek_v4")
    DeepseekV4Model = modeling.DeepseekV4Model
    DeepseekV4RotaryEmbedding = modeling.DeepseekV4RotaryEmbedding

    axis_of = DeepseekV4RotaryEmbedding.rope_axis_for_layer_type
    assert axis_of("sliding_attention") == "main"
    assert axis_of("compressed_sparse_attention") == "compress"
    assert axis_of("heavily_compressed_attention") == "compress"
    # Every axis it can name must be a table the rotary actually builds.
    for layer_type in ("sliding_attention", "compressed_sparse_attention",
                       "heavily_compressed_attention"):
        assert axis_of(layer_type) in DeepseekV4RotaryEmbedding.layer_types
        assert profile.rope_axis_for_layer_type(layer_type) == axis_of(layer_type)

    # The model's own forward resolves through the same staticmethod rather
    # than an inline conditional, so the two cannot drift apart.
    source = inspect.getsource(DeepseekV4Model.forward)
    assert "rope_axis_for_layer_type" in source, (
        "DeepseekV4Model.forward no longer resolves the rope axis through the "
        "shared definition; the streamed driver can now silently disagree")
