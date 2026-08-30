"""The shared name-projection layer (``prismaquant/name_projection.py``).

These tests pin the contract the four R5 consumer migrations depend on:
one projection between the live/recipe/checkpoint/export/vLLM namespaces,
routed entirely through the model profile; fail-closed refusals with
structured fields; explicit many->one and one->many shapes for fused
siblings and packed experts; and round-trip identity wherever the
profile's mapping is total.

Fixtures use real profiles against real (toy) transformers models where a
forward can execute offline, and real profile declarations from their
shipped structure specs everywhere else. No checkpoint weights are used.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import prismaquant.name_projection as npx
from prismaquant.model_profiles.base import ModelProfile
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.model_walk import WalkResult, walk_model


# ---------------------------------------------------------------- fixtures


def _meta(build):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return build().eval()
    finally:
        torch.set_default_dtype(prev)


def _tiny_qwen3():
    from transformers import AutoConfig, AutoModelForCausalLM

    def build():
        cfg = AutoConfig.for_model(
            "qwen3",
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
            attention_dropout=0.0, tie_word_embeddings=False,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        return AutoModelForCausalLM.from_config(cfg)

    return _meta(build)


def _tiny_qwen3_moe():
    # transformers 5.x builds the PACKED expert tree (`mlp.experts.
    # gate_up_proj` / `.down_proj`, 3-D parameters) and its grouped-mm
    # forward only has a BF16 meta kernel, so this fixture constructs in
    # bfloat16. Root A (the node universe) is what these tests consume.
    from transformers import AutoConfig, AutoModelForCausalLM

    def build():
        cfg = AutoConfig.for_model(
            "qwen3_moe",
            vocab_size=64, hidden_size=32, intermediate_size=64,
            moe_intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2,
            max_position_embeddings=64, tie_word_embeddings=False,
        )
        return AutoModelForCausalLM.from_config(cfg)

    return _meta(build)


def _walked(model, profile):
    rules = profile.walk_claim_rules()
    result = walk_model(model, claim_rules=rules)
    result.raise_if_failed()
    return result


@pytest.fixture(scope="module")
def qwen3_result():
    return _walked(_tiny_qwen3(), Qwen3Profile())


@pytest.fixture(scope="module")
def qwen3_proj(qwen3_result):
    return npx.NameProjection.for_walk(qwen3_result, Qwen3Profile())


@pytest.fixture(scope="module")
def moe_result():
    return _walked(_tiny_qwen3_moe(), Qwen3Profile())


# ------------------------------------------------------- the probe/cost join


def test_recipe_unit_is_the_stats_assignment_key(qwen3_proj):
    unit = qwen3_proj.recipe_unit(
        "model.layers.0.self_attn.q_proj.weight")
    assert unit == "model.layers.0.self_attn.q_proj"
    # The leaf handling belongs to the layer, not to each consumer.
    assert not unit.endswith(".weight")


def test_live_to_recipe_is_total_over_the_walk_universe(
        qwen3_result, qwen3_proj):
    decided = [n for n, c in qwen3_result.claims.items()
               if c.disposition == "decide"]
    assert decided  # the fixture really walked a quantizable model
    for name in decided:
        recipe = qwen3_proj.live_to_recipe(name)
        assert isinstance(recipe, str) and recipe
        # Round trip over a total mapping is the identity.
        assert qwen3_proj.recipe_to_live(recipe) == name


def test_block_id_and_dispatch_agree_with_shared_helpers(qwen3_proj):
    assert qwen3_proj.block_id(
        "model.layers.1.self_attn.o_proj.weight") == "model.layers.1"


def test_project_dispatch_serves_declared_pairs_only(qwen3_proj):
    projected = qwen3_proj.project(
        "model.layers.0.self_attn.q_proj.weight", npx.LIVE, npx.RECIPE)
    assert isinstance(projected, npx.ProjectedName)
    assert projected.outcome == npx.MAPPED
    assert projected.target == "model.layers.0.self_attn.q_proj.weight"

    with pytest.raises(npx.NameProjectionError) as unknown_ns:
        qwen3_proj.project("x", "tensor_parallel_shard", npx.LIVE)
    assert unknown_ns.value.code == "unknown_namespace"

    with pytest.raises(npx.NameProjectionError) as bad_pair:
        qwen3_proj.project("x", npx.LIVE, npx.VLLM)
    assert bad_pair.value.code == "unsupported_pair"


# ------------------------------------------- fused siblings: many -> one


def test_fused_sibling_group_collapses_explicitly(qwen3_proj):
    qkv = {
        proj: qwen3_proj.fused_sibling_group(
            f"model.layers.0.self_attn.{proj}")
        for proj in ("q_proj", "k_proj", "v_proj")
    }
    assert len(set(qkv.values())) == 1
    group_key = next(iter(qkv.values()))
    assert group_key == "model.layers.0.self_attn.qkv_proj"
    # o_proj and down_proj are NOT fused siblings.
    assert qwen3_proj.fused_sibling_group(
        "model.layers.0.self_attn.o_proj") is None
    assert qwen3_proj.fused_sibling_group(
        "model.layers.0.mlp.down_proj") is None


def test_serving_group_reports_kind_members_namespace(qwen3_proj):
    group = qwen3_proj.serving_group("model.layers.1.mlp.gate_proj.weight")
    assert group.kind == npx.GROUP_FUSED_SIBLINGS
    assert group.key == "model.layers.1.mlp.gate_up_proj"
    assert group.members == (
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
    )
    singleton = qwen3_proj.serving_group(
        "model.layers.1.self_attn.o_proj.weight")
    assert singleton.kind == npx.GROUP_SINGLETON
    assert singleton.key == ""
    assert singleton.members == ("model.layers.1.self_attn.o_proj",)


def test_reverse_of_a_group_key_refuses_instead_of_guessing(qwen3_proj):
    # A fused-group KEY names a serving unit, not a parameter. Handing it
    # to a reverse lookup must refuse loudly — silently returning one of
    # the siblings would be exactly the ad hoc guessing this layer kills.
    with pytest.raises(npx.NameProjectionError) as err:
        qwen3_proj.unit_to_live("model.layers.0.self_attn.qkv_proj")
    assert err.value.code == "unmapped_in_universe"
    assert err.value.source_namespace == npx.RECIPE
    assert err.value.target_namespace == npx.LIVE
    assert err.value.tried  # what was attempted is part of the record


def test_require_serving_group_refuses_the_wrong_shape(qwen3_proj):
    with pytest.raises(npx.NameProjectionError) as wrong:
        qwen3_proj.require_serving_group(
            "model.layers.0.mlp.down_proj.weight",
            kinds=(npx.GROUP_PACKED_EXPERTS,))
    assert wrong.value.code == "wrong_group_kind"
    with pytest.raises(npx.NameProjectionError) as ungrouped:
        qwen3_proj.require_serving_group(
            "model.layers.0.mlp.down_proj.weight", kinds=())
    assert ungrouped.value.code == "ungrouped_unit"
    grouped = qwen3_proj.require_serving_group(
        "model.layers.0.mlp.gate_proj.weight",
        kinds=(npx.GROUP_FUSED_SIBLINGS, npx.GROUP_PACKED_EXPERTS))
    assert grouped.kind == npx.GROUP_FUSED_SIBLINGS


# ------------------------------------------------- packed experts (MoE tree)


def test_packed_expert_stack_groups_by_role(moe_result):
    profile = Qwen3Profile()
    proj = npx.NameProjection.for_walk(moe_result, profile)
    packed_node = "model.layers.1.mlp.experts.gate_up_proj"
    assert any(node.name == packed_node for node in moe_result.nodes), (
        "fixture expected the packed 3-D expert parameters of the "
        "transformers 5.x tree")
    key = proj.packed_format_group(packed_node)
    assert key == (
        "model.layers.1.mlp.experts::__packed_format__:gate_up_proj,down_proj"
    )
    group = proj.serving_group(packed_node)
    assert group.kind == npx.GROUP_PACKED_EXPERTS
    assert set(group.members) >= {packed_node,
                                  "model.layers.1.mlp.experts.down_proj"}


def test_split_checkpoint_spells_cover_the_packed_aggregate(moe_result):
    """One logical tensor, many export spellings: the coverage index must
    answer under EITHER naming scheme (footprint's dual-entry manifest
    convention), which is the one->many case made explicit."""
    profile = Qwen3Profile()
    split_keys = [
        f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
        for expert in range(4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    proj = npx.NameProjection.for_walk(
        moe_result, profile, checkpoint_keys=split_keys)
    gate_up = proj.checkpoint_keys_for(
        "model.layers.1.mlp.experts.gate_up_proj")
    assert gate_up == tuple(sorted(
        f"model.layers.1.mlp.experts.{e}.{p}.weight"
        for e in range(4) for p in ("gate_proj", "up_proj")))
    down = proj.checkpoint_keys_for("model.layers.1.mlp.experts.down_proj")
    assert down == tuple(sorted(
        f"model.layers.1.mlp.experts.{e}.down_proj.weight"
        for e in range(4)))
    # A per-expert spelling resolves too, to its own single span.
    own = proj.checkpoint_keys_for("model.layers.1.mlp.experts.2.gate_proj")
    assert own == ("model.layers.1.mlp.experts.2.gate_proj.weight",)


def test_coverage_queries_fail_closed(moe_result):
    profile = Qwen3Profile()
    # No checkpoint universe supplied: the query refuses rather than
    # answering an empty tuple that could be read as "uncovered".
    bare = npx.NameProjection.for_walk(moe_result, profile)
    with pytest.raises(npx.NameProjectionError) as missing:
        bare.checkpoint_keys_for("lm_head")
    assert missing.value.code == "no_universe_supplied"

    covered = npx.NameProjection.for_walk(
        moe_result, profile,
        checkpoint_keys=["model.layers.1.mlp.experts.gate_up_proj"])
    with pytest.raises(npx.NameProjectionError) as uncovered:
        covered.require_checkpoint_span("lm_head.weight")
    assert uncovered.value.code == "uncovered_unit"
    assert covered.require_checkpoint_span(
        "model.layers.1.mlp.experts.gate_up_proj") == (
        "model.layers.1.mlp.experts.gate_up_proj",)


def test_packed_precedence_over_fused_is_explicit():
    """An unpacked expert projection's LEAF can carry a fused-sibling
    name; routed experts must group by their stack. Packed wins."""

    class _BothGroupsProfile(ModelProfile):
        name = "both-groups-stub"

        @classmethod
        def matches(cls, model_type, architectures):
            return False

        def structure_spec(self):
            return None

        def packed_expert_param_names(self):
            return frozenset({"gate_up_proj", "down_proj"})

        def fused_sibling_group(self, linear_qname):
            # Aggressive stand-in for a vLLM-derived leaf matcher: fires
            # on every gate/up leaf anywhere in the tree.
            if linear_qname.rsplit(".", 1)[-1] in {"gate_proj", "up_proj"}:
                parent, _, _ = linear_qname.rpartition(".")
                return f"{parent}.gate_up_proj"
            return None

        def packed_expert_format_group(self, qname):
            if ".experts." in f"{qname}.":
                return f"{qname.split('.experts.')[0]}.experts::__stack__"
            return None

    profile = _BothGroupsProfile()
    proj = npx.NameProjection(profile, live_names=[
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ])
    expert_group = proj.serving_group(
        "model.layers.0.mlp.experts.3.gate_proj.weight")
    assert expert_group.kind == npx.GROUP_PACKED_EXPERTS
    assert expert_group.key.endswith("experts::__stack__")
    dense_group = proj.serving_group("model.layers.0.mlp.gate_proj.weight")
    assert dense_group.kind == npx.GROUP_FUSED_SIBLINGS
    assert dense_group.key == "model.layers.0.mlp.gate_up_proj"


# --------------------------------------------------- multimodal namespaces
#
# Qwen3.5's shipped spec declares non-identity rewrites in all three
# directions (specs/qwen3_5.json `naming`): live strips the
# `language_model.` infix into recipes, recipes re-add it for source
# spellings, and vLLM dispatch uses `language_model.model.`.


@pytest.fixture(scope="module")
def qwen35_multimodal_profile():
    from prismaquant.model_profiles.registry import profile_from_config

    return profile_from_config({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    })


def test_multimodal_live_to_recipe_strips_the_umbrella_infix(
        qwen35_multimodal_profile):
    proj = npx.NameProjection(qwen35_multimodal_profile, live_names=[
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "lm_head.weight",
    ])
    live = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert proj.live_to_recipe(live) == \
        "model.layers.0.self_attn.q_proj.weight"
    assert proj.recipe_unit(live) == "model.layers.0.self_attn.q_proj"
    # Round trip over a total mapping is the identity.
    assert proj.recipe_to_live("model.layers.0.self_attn.q_proj.weight") == \
        live


def test_multimodal_recipe_to_checkpoint_round_trip_identity(
        qwen35_multimodal_profile):
    proj = npx.NameProjection(qwen35_multimodal_profile)
    # recipe -> checkpoint -> recipe, requirement 4 verbatim: identity
    # wherever the profile's rules are total.
    for recipe in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        "model.layers.7.mlp.shared_expert.down_proj.weight",
    ):
        source = proj.live_to_source(recipe)  # live==recipe prefix here
        back = proj.project(source, npx.CHECKPOINT, npx.LIVE)
        assert back.outcome == npx.MAPPED
        # Live targets are UNIT spellings (module qnames), the form
        # allocator/assignment dicts key on.
        assert back.target == npx.strip_weight_leaf(recipe)


def test_vllm_internal_names_use_exact_rules_and_preserve_leaves(
        qwen35_multimodal_profile):
    proj = npx.NameProjection(qwen35_multimodal_profile)
    assert proj.to_vllm_internal(
        "model.layers.0.mlp.experts.gate_up_proj") == (
        "language_model.model.layers.0.mlp.experts.gate_up_proj")
    # The spec's EXACT rule targets the bare module qname `lm_head`;
    # applying the accessor at module granularity keeps the leaf intact.
    assert proj.to_vllm_internal("lm_head.weight") == \
        "language_model.lm_head.weight"
    assert proj.to_vllm_internal("lm_head") == "language_model.lm_head"


# ------------------------------------------------------------- DSv4 joins


DSV4_SAMPLES = [
    # (live parameter, flat checkpoint key) pairs attested by the
    # profile's own mapping (deepseek_v4.py checkpoint_to_live_name).
    ("model.embed_tokens.weight", "embed.weight"),
    ("lm_head.weight", "head.weight"),
    ("model.norm.weight", "norm.weight"),
    ("model.layers.0.self_attn.wo_a.weight", "layers.0.attn.wo_a.weight"),
    ("model.layers.2.mlp.gate_proj.weight", "layers.2.ffn.gate_proj.weight"),
    ("model.layers.2.mlp.experts.3.gate_proj.weight",
     "layers.2.ffn.experts.3.w1.weight"),
    ("model.layers.1.mlp.shared_experts.down_proj.weight",
     "layers.1.ffn.shared_experts.w2.weight"),
]


def test_dsv4_flat_checkpoint_join():
    proj = npx.NameProjection(DeepseekV4Profile())
    for live, ckpt in DSV4_SAMPLES:
        mapped = proj.checkpoint_to_live(ckpt)
        assert mapped.outcome == npx.MAPPED, ckpt
        assert mapped.target == live[:-len(".weight")] or \
            mapped.target == live
        assert mapped.via == "ModelProfile.checkpoint_to_live_name"


def test_dsv4_recipe_checkpoint_round_trip_is_identity_on_the_body():
    proj = npx.NameProjection(DeepseekV4Profile())
    for live, _ckpt in DSV4_SAMPLES:
        source = proj.live_to_source(live)
        back = proj.project(source, npx.CHECKPOINT, npx.LIVE)
        assert back.outcome == npx.MAPPED, (live, source)
        assert back.target == npx.strip_weight_leaf(live), (live, source)


def test_dsv4_declared_drops_are_data_not_errors():
    proj = npx.NameProjection(DeepseekV4Profile())
    # FP8 scale sibling, standalone scale, MTP sidecar: the profile
    # declines these BY CONTRACT and the projection reports it as an
    # outcome, never a silent None and never an exception.
    for declined in (
        "layers.2.attn.wkv.scale",
        "layers.2.ffn.experts.3.w1.scale",
        "mtp.2.hc_head_fn",
        "mtp.fc.weight",
    ):
        projected = proj.checkpoint_to_live(declined)
        assert projected.outcome == npx.DECLARED_OUT_OF_GRAPH, declined
        assert projected.target is None
    # The generic entry point carries the same structured outcome.
    via_project = proj.project("head.scale", npx.CHECKPOINT, npx.LIVE)
    assert via_project.outcome == npx.DECLARED_OUT_OF_GRAPH


def test_dsv4_hyperhead_inverse_gap_is_surfaced_not_guessed():
    """Recorded debt: DSv4's declared recipe->source rules generate
    `hc_head.hc_fn` while its checkpoint stores the flat `hc_head_fn`
    (real checkpoint keys verified 2026-08-22). The layer must surface
    the asymmetry — the generated spelling DECLINES on the way back —
    instead of papering over it with a guessed inverse. Fixing the spec
    is profile-side work; this test pins the honest behavior either way
    the fix lands (mapped-after-fix would need this test updated in the
    same commit)."""
    proj = npx.NameProjection(DeepseekV4Profile())
    forward = proj.live_to_source("model.hc_head.hc_fn.weight")
    assert forward == "hc_head.hc_fn.weight"
    back = proj.project(forward, npx.CHECKPOINT, npx.LIVE)
    assert back.outcome == npx.DECLARED_OUT_OF_GRAPH
    # The REAL flat key still maps, so the gap is one-sided.
    real = proj.checkpoint_to_live("hc_head_fn")
    assert real.outcome == npx.MAPPED
    assert real.target == "model.hc_head.hc_fn"


# ------------------------------------------------------------ refusals


class _CollapsingProfile(ModelProfile):
    """A profile whose live_to_recipe is genuinely many->one: two live
    tensors land on one recipe name. The inverse is ambiguous by
    construction and must be reported as such."""

    name = "collapsing-stub"

    @classmethod
    def matches(cls, model_type, architectures):
        return False

    def structure_spec(self):
        return None

    def live_to_recipe_name(self, live_qname):
        return "model.shared.unit.weight"


class _MalformedRecipeProfile(ModelProfile):
    name = "malformed-recipe-stub"

    @classmethod
    def matches(cls, model_type, architectures):
        return False

    def structure_spec(self):
        return None

    def live_to_recipe_name(self, live_qname):
        return None


class _RaisingFusedProfile(ModelProfile):
    name = "raising-fused-stub"

    @classmethod
    def matches(cls, model_type, architectures):
        return False

    def structure_spec(self):
        return None

    def fused_sibling_group(self, linear_qname):
        raise RuntimeError("spec exploded")

    def packed_expert_format_group(self, qname):
        return None


class _BadCheckpointProfile(ModelProfile):
    name = "bad-checkpoint-stub"

    @classmethod
    def matches(cls, model_type, architectures):
        return False

    def structure_spec(self):
        return None

    def checkpoint_to_live_name(self, k, *, multimodal=False):
        return 42


def test_ambiguous_inverse_raises_listing_both_candidates():
    proj = npx.NameProjection(_CollapsingProfile(), live_names=[
        "model.a.weight", "model.b.weight"])
    with pytest.raises(npx.NameProjectionError) as err:
        proj.recipe_to_live("model.shared.unit.weight")
    assert err.value.code == "ambiguous"
    assert "model.a.weight" in err.value.detail
    assert "model.b.weight" in err.value.detail


def test_malformed_accessor_results_are_structured_refusals():
    with pytest.raises(npx.NameProjectionError) as built:
        npx.NameProjection(
            _MalformedRecipeProfile(), live_names=["model.a.weight"])
    assert built.value.code == "malformed_profile_result"

    proj = npx.NameProjection(_BadCheckpointProfile())
    with pytest.raises(npx.NameProjectionError) as ckpt:
        proj.checkpoint_to_live("anything.weight")
    assert ckpt.value.code == "malformed_profile_result"
    assert ckpt.value.source_namespace == npx.CHECKPOINT
    assert ckpt.value.target_namespace == npx.LIVE

    # Construction eagerly validates the profile's group declarations
    # over the whole universe: an accessor that explodes is a declaration
    # bug and refuses at build time, not mid-query.
    with pytest.raises(npx.NameProjectionError) as fused:
        npx.NameProjection(
            _RaisingFusedProfile(),
            live_names=["model.layers.0.mlp.gate_proj"])
    assert fused.value.code == "profile_accessor_failed"
    assert any("fused_sibling_group" in t for t in fused.value.tried)


def test_construction_requires_a_profile():
    with pytest.raises(ValueError):
        npx.NameProjection(None)


def test_projection_carries_no_tp_degree():
    """TP seam guard: names are logical, degree-independent facts. No
    constructor kwarg, method argument, or field exposes a rank/shard/
    degree anywhere in this API — a future per-rank spelling is a
    SEPARATE decoration over these logical names."""
    proj = npx.NameProjection(Qwen3Profile(), live_names=["lm_head.weight"])
    with pytest.raises(TypeError):
        npx.NameProjection(Qwen3Profile(), tp_degree=8)
    with pytest.raises(TypeError):
        proj.recipe_unit("lm_head.weight", rank=0)  # type: ignore[call-arg]
    assert proj.recipe_unit("lm_head.weight") == "lm_head"


def test_strip_weight_leaf_is_the_one_leaf_rule():
    assert npx.strip_weight_leaf("a.b.weight") == "a.b"
    # Packed 3-D parameters carry NO leaf; they pass through unchanged.
    assert npx.strip_weight_leaf("model.layers.0.mlp.experts.gate_up_proj") \
        == "model.layers.0.mlp.experts.gate_up_proj"
    # Only the WEIGHT leaf is special; sidecars are footprint's business.
    assert npx.strip_weight_leaf("a.b.weight_scale") == "a.b.weight_scale"


def test_for_walk_rejects_non_results(qwen3_result):
    with pytest.raises(TypeError):
        npx.NameProjection.for_walk(["not", "a", "result"], Qwen3Profile())
    assert isinstance(qwen3_result, WalkResult)


def test_error_str_is_readable_but_fields_are_authoritative():
    proj = npx.NameProjection(Qwen3Profile(), live_names=["lm_head.weight"])
    with pytest.raises(npx.NameProjectionError) as err:
        proj.unit_to_live("nope")
    message = str(err.value)
    assert "nope" in message and "recipe" in message and "live" in message
    exc = err.value
    assert (exc.name, exc.code, exc.tried) == (
        "nope", "unmapped_in_universe", exc.tried)
