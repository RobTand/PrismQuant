"""R5 consumer migration: the probe's name derivation routes through the
shared name-projection layer.

Pins what changed when ``prismaquant/sensitivity_probe.py`` stopped
holding private name mappings (2026-08-22, ``walker/consumer-probe``):

1. The packed-expert shard-scope filter derives its block id from
   :meth:`NameProjection.block_id` (the shared
   ``decision_units.block_id_from_qname`` rule), not a positional
   ``[:3]`` slice of the qname. On every qname shape reachable in this
   repo today the two agree exactly — pinned here so the refactor is
   provably value-preserving on shipped families.
2. The Fisher skip set's embedding clause keys on the profile's DECLARED
   embedding name (:meth:`ModelProfile.embedding_name`), not a hardcoded
   ``"model.embed_tokens"`` spelling: rename the declaration and the
   skip follows it.
3. The accumulator builds its ``NameProjection`` once and refuses to run
   without a profile: the layer's constructor rejects ``None`` rather
   than degrading every mapping to identity, so the probe can never
   silently fall back to an ungoverned mapping.

No GPU, no checkpoint weights, no network.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from prismaquant.decision_units import block_id_from_qname
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.name_projection import NameProjection
from prismaquant.sensitivity_probe import FisherAccumulator


class _ToyPackedExperts(nn.Module):
    """Qwen3.5-style packed experts container: 3-D params named like the
    default profile declaration. The probe only needs install-time
    metadata to observe the block filter; nothing calls forward."""

    def __init__(self, num_experts: int = 2, hidden: int = 4, inter: int = 3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, inter))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4, 4)
        self.experts = _ToyPackedExperts()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


def _tiny_model() -> nn.Module:
    """`model.{embed_tokens,layers.{7,9}.mlp.*,lm_head}` — Linear-typed
    embedding/head stand-ins at the conventional live spellings, with
    packed-experts containers under two blocks so inclusion scoping is
    observable."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Linear(4, 4)
            self.layers = nn.ModuleDict({"7": _Block(), "9": _Block()})
            self.lm_head = nn.Linear(4, 4)

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    return _Root()


def _tracked_names(model: nn.Module) -> list[str]:
    return [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]


# --------------------------------------------------------------- block id


def test_packed_block_filter_matches_the_legacy_slice_on_reachable_shapes():
    """Value preservation: for every packed-experts qname this repo can
    produce today, NameProjection.block_id returns exactly what the old
    private `".".join(qname.split(".")[:3])` returned, so no stats row
    can gain or lose inclusion."""
    proj = NameProjection(Qwen3Profile())
    # NOTE: every live tree the accumulator can be handed today starts
    # with a two-component layer prefix (`model.layers`, `mtp.layers`),
    # so the `layers.<int>` pair sits at component depth 1 and the two
    # derivations coincide exactly. Deeper wrappers diverge (pinned as a
    # divergence below); a relative `layers.N...` root is a CHECKPOINT
    # spelling that never reaches this filter.
    for qname in (
        "model.layers.7.mlp.experts",          # body MoE block
        "model.layers.0.mlp.experts",          # first block
        "mtp.layers.0.ffn.experts",            # MTP shard wrapper
    ):
        assert proj.block_id(qname) == ".".join(qname.split(".")[:3])


def test_block_id_is_the_shared_rule_not_a_private_slice():
    """The accumulator's projection delegates to the one shared block-id
    rule, and diverges from the legacy slice exactly where the slice was
    only correct by accident (a layer prefix deeper than two components).
    The divergence is the layer's explicit rule winning over a heuristic;
    it is documented in the migration report, not smuggled in."""
    acc_proj = NameProjection(Qwen3Profile())
    assert acc_proj.block_id("model.layers.2.self_attn.q_proj") == (
        block_id_from_qname("model.layers.2.self_attn.q_proj"))
    deep = "model.language_model.layers.7.mlp.experts"
    assert acc_proj.block_id(deep) == "model.language_model.layers.7"
    assert ".".join(deep.split(".")[:3]) != acc_proj.block_id(deep)


def test_packed_entry_included_only_when_its_block_is_tracked():
    model = _tiny_model()
    # Only block 7 has a tracked dense Linear; block 9's experts container
    # must stay out of stats under both the old slice and the shared rule.
    tracked = ["model.layers.7.mlp.up"]
    acc = FisherAccumulator(model, tracked, {}, model_profile=Qwen3Profile())
    assert "model.layers.7.mlp.experts.gate_up_proj" in acc.stats
    assert "model.layers.7.mlp.experts.down_proj" in acc.stats
    assert "model.layers.9.mlp.experts.gate_up_proj" not in acc.stats
    assert "model.layers.9.mlp.experts.down_proj" not in acc.stats


# ------------------------------------------------------------ fisher skip


def test_fisher_skip_keys_on_pinned_and_declared_embedding_names():
    model = _tiny_model()
    acc = FisherAccumulator(
        model, _tracked_names(model), {}, model_profile=Qwen3Profile())
    # lm_head is skipped through the profile's pins (unchanged behavior);
    # the embedding is skipped because the profile DECLARES that name.
    assert "model.lm_head" in acc._fisher_skip
    assert "model.embed_tokens" in acc._fisher_skip
    assert "model.layers.7.mlp.up" not in acc._fisher_skip
    assert "model.layers.9.mlp.up" not in acc._fisher_skip


def test_embedding_skip_follows_a_renamed_declaration():
    class _RenamedEmbedProfile(Qwen3Profile):
        def embedding_name(self) -> str:
            return "model.model.embed_tokens"

    model = _tiny_model()
    acc = FisherAccumulator(
        model, _tracked_names(model), {},
        model_profile=_RenamedEmbedProfile())
    # Only the DECLARED spelling and the pins are skipped. In particular
    # `model.embed_tokens` stays measurable: the old private clause would
    # have substring-matched it regardless of any declaration.
    assert acc._fisher_skip == {"model.lm_head"}


# ------------------------------------------------- construction refusal


def test_projection_seam_refuses_a_none_profile():
    """Requirement: a refusal propagates instead of degrading. The probe
    builds its projection from a resolved profile; if it ever handed the
    layer a None profile, construction itself must fail loudly."""
    with pytest.raises(ValueError):
        NameProjection(None)


def test_accumulator_seam_carries_no_tp_degree():
    """TP invariant 1: names are logical whole-tensor facts; nothing in
    the migrated seam accepts a rank/shard/degree."""
    model = _tiny_model()
    acc = FisherAccumulator(model, _tracked_names(model), {},
                            model_profile=Qwen3Profile())
    assert not hasattr(acc._name_projection, "tp_degree")
    with pytest.raises(TypeError):
        NameProjection(Qwen3Profile(), tp_degree=8)  # type: ignore[call-arg]
