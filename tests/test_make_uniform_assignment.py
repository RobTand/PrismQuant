"""Unit tests for ``tools/make_uniform_assignment.py``.

The tool must produce an assignment that passes the SAME legality the
allocator's own output passes, on a purely synthetic architecture + profile —
i.e. nothing about it may be keyed to a specific model family.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from prismaquant.model_profiles.base import ModelProfile
from prismaquant.model_profiles.structure import build_model_graph
from tools.make_uniform_assignment import (
    assert_assignment_legal,
    build_uniform_assignment,
    layer_config_payload,
)

HIDDEN = 512
INTERMEDIATE = 1024
# 272 % 16 == 0 (legal for NVFP4's group-16) but 272 % 256 != 0 (illegal for
# FP8_CB_K36's group-256): the one shape that separates the two endpoints.
CB_ILLEGAL_IN = 272


# ---------------------------------------------------------------------------
# A synthetic architecture + profile
# ---------------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, complete: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        if complete:
            self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _Mlp(nn.Module):
    def __init__(self, up_in: int = HIDDEN):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(up_in, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class _Layer(nn.Module):
    def __init__(self, complete_attn: bool = True, up_in: int = HIDDEN):
        super().__init__()
        self.self_attn = _Attn(complete=complete_attn)
        self.mlp = _Mlp(up_in=up_in)
        self.input_layernorm = nn.LayerNorm(HIDDEN)


class _ToyModel(nn.Module):
    """layer 0: ordinary · layer 1: CB-illegal up_proj · layer 2: no v_proj."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, HIDDEN)
        self.model.layers = nn.ModuleList([
            _Layer(),
            _Layer(up_in=CB_ILLEGAL_IN),
            _Layer(complete_attn=False),
        ])
        self.lm_head = nn.Linear(HIDDEN, 64, bias=False)


class _ToyProfile(ModelProfile):
    """Synthetic profile — never registered, no vLLM class, no structure spec."""

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        return model_type == "toyformer"

    @property
    def name(self) -> str:
        return "toy_uniform_test"

    def vllm_architecture_class(self) -> str | None:
        return None

    def fused_sibling_leaf_mapping(self) -> dict[str, tuple[str, ...]]:
        return {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }

    def fused_sibling_group(self, linear_qname: str) -> str | None:
        stem, _, leaf = str(linear_qname).rpartition(".")
        for fused, members in self.fused_sibling_leaf_mapping().items():
            if leaf in members:
                return f"{stem}.{fused}" if stem else fused
        return None

    def pinned_names(self) -> tuple[str, ...]:
        return ("lm_head",)


@pytest.fixture(scope="module")
def graph():
    return build_model_graph(_ToyModel(), _ToyProfile())


@pytest.fixture(scope="module")
def profile():
    return _ToyProfile()


def _bf16_sources(graph) -> dict[str, str]:
    return {t.recipe_name.removesuffix(".weight"): "bf16" for t in graph.tensors}


# ---------------------------------------------------------------------------
# Coverage + the standard exceptions
# ---------------------------------------------------------------------------

def test_uniform_covers_every_quantizable_linear_and_never_the_pinned_head(
        graph, profile):
    result = build_uniform_assignment(
        graph, "NVFP4", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph))

    # layer 0 (7) + layer 1 (7) = 14 assigned; layer 2's incomplete qkv group
    # (q/k) is omitted, its o_proj/mlp are not.
    assert set(result.assignment.values()) == {"NVFP4"}
    assert "lm_head" not in result.assignment
    assert "model.embed_tokens" not in result.assignment
    assert result.assignment["model.layers.0.self_attn.q_proj"] == "NVFP4"
    assert result.assignment["model.layers.1.mlp.up_proj"] == "NVFP4"
    assert result.achieved_bits == pytest.approx(4.5)


def test_incomplete_fused_group_is_omitted_whole(graph, profile):
    result = build_uniform_assignment(
        graph, "NVFP4", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph))

    for leaf in ("q_proj", "k_proj"):
        qname = f"model.layers.2.self_attn.{leaf}"
        assert qname not in result.assignment
        assert result.excluded[qname] == "incomplete_fused_group"
    # The rest of that layer is untouched by the incomplete group.
    assert result.assignment["model.layers.2.self_attn.o_proj"] == "NVFP4"
    assert result.assignment["model.layers.2.mlp.down_proj"] == "NVFP4"


def test_tied_lm_head_names_are_omitted(graph, profile):
    result = build_uniform_assignment(
        graph, "NVFP4", profile=profile, target_profile="research",
        source_kinds=_bf16_sources(graph),
        tied_head_names=["model.layers.0.mlp.down_proj"])

    assert "model.layers.0.mlp.down_proj" not in result.assignment
    assert result.excluded["model.layers.0.mlp.down_proj"] == "tied_lm_head"


# ---------------------------------------------------------------------------
# Unit atomicity: a legality verdict may never split a fused group
# ---------------------------------------------------------------------------

def test_illegal_member_demotes_the_whole_fused_unit(graph, profile):
    result = build_uniform_assignment(
        graph, "FP8_CB_K36", profile=profile, target_profile="nvfp4_cb",
        source_kinds=_bf16_sources(graph))

    # up_proj is group-256-illegal; gate_proj alone would have been legal.
    assert result.assignment["model.layers.1.mlp.up_proj"] == "BF16"
    assert result.assignment["model.layers.1.mlp.gate_proj"] == "BF16"
    assert "fused:model.layers.1.mlp.gate_up_proj" in result.demoted_units
    # Nothing else demoted.
    assert result.assignment["model.layers.0.mlp.gate_proj"] == "FP8_CB_K36"
    assert result.assignment["model.layers.1.mlp.down_proj"] == "FP8_CB_K36"


def test_bf16_fallback_is_refused_on_a_non_bf16_source(graph, profile):
    """BF16 is passthrough-only: a demoted unit over an fp8 source is omitted,
    never synthesized (core principle 11)."""
    sources = _bf16_sources(graph)
    for leaf in ("gate_proj", "up_proj"):
        sources[f"model.layers.1.mlp.{leaf}"] = "fp8"

    result = build_uniform_assignment(
        graph, "FP8_CB_K36", profile=profile, target_profile="nvfp4_cb",
        source_kinds=sources)

    for leaf in ("gate_proj", "up_proj"):
        qname = f"model.layers.1.mlp.{leaf}"
        assert qname not in result.assignment
        assert result.excluded[qname].startswith("illegal_FP8_CB_K36_and_fallback")


def test_no_fallback_means_the_unit_is_omitted(graph, profile):
    result = build_uniform_assignment(
        graph, "FP8_CB_K36", profile=profile, target_profile="nvfp4_cb",
        source_kinds=_bf16_sources(graph), fallback_format=None)

    assert "model.layers.1.mlp.gate_proj" not in result.assignment
    assert "model.layers.1.mlp.up_proj" not in result.assignment
    assert result.demoted_units == ()


def test_reader_only_fp8_cb_rung_cannot_enter_a_uniform_assignment(
        graph, profile):
    with pytest.raises(ValueError, match="reader-only"):
        build_uniform_assignment(
            graph,
            "FP8_CB_K29",
            profile=profile,
            target_profile="research",
            source_kinds=_bf16_sources(graph),
        )

    with pytest.raises(AssertionError, match="reader-only"):
        assert_assignment_legal(
            {"model.layers.0.mlp.down_proj": "FP8_CB_K29"},
            graph,
            profile=profile,
            target_profile="research",
            source_kinds=_bf16_sources(graph),
        )


# ---------------------------------------------------------------------------
# The output must pass the allocator's legality gate + the production parser
# ---------------------------------------------------------------------------

def test_result_passes_the_legality_gate_and_round_trips(graph, profile):
    sources = _bf16_sources(graph)
    result = build_uniform_assignment(
        graph, "FP8_CB_K36", profile=profile, target_profile="nvfp4_cb",
        source_kinds=sources)

    assert_assignment_legal(result.assignment, graph, profile=profile,
                            target_profile="nvfp4_cb", source_kinds=sources)

    payload = layer_config_payload(result, meta={"schema": "test"})
    assert payload["__prismaquant__"] == {"schema": "test"}
    assert payload["model.layers.0.mlp.down_proj"]["data_type"] == "fp8_cb"
    assert payload["model.layers.0.mlp.down_proj"]["cb_k"] == 36


def test_legality_gate_rejects_a_mixed_fused_unit(graph, profile):
    sources = _bf16_sources(graph)
    result = build_uniform_assignment(
        graph, "NVFP4", profile=profile, target_profile="research",
        source_kinds=sources)

    broken = dict(result.assignment)
    broken["model.layers.0.self_attn.q_proj"] = "BF16"
    with pytest.raises(AssertionError, match="ONE format"):
        assert_assignment_legal(broken, graph, profile=profile,
                                target_profile="research", source_kinds=sources)


def test_legality_gate_rejects_an_illegal_shape(graph, profile):
    sources = _bf16_sources(graph)
    broken = {
        "model.layers.1.mlp.gate_proj": "FP8_CB_K36",
        "model.layers.1.mlp.up_proj": "FP8_CB_K36",
    }
    with pytest.raises(AssertionError, match="not legal"):
        assert_assignment_legal(broken, graph, profile=profile,
                                target_profile="nvfp4_cb", source_kinds=sources)


def test_legality_gate_rejects_a_synthesized_bf16_over_fp8_source(graph, profile):
    sources = _bf16_sources(graph)
    sources["model.layers.0.mlp.down_proj"] = "fp8"
    broken = {"model.layers.0.mlp.down_proj": "BF16"}
    with pytest.raises(AssertionError):
        assert_assignment_legal(broken, graph, profile=profile,
                                target_profile="research", source_kinds=sources)


def test_achieved_bits_counts_only_the_covered_quantizable_params(graph, profile):
    result = build_uniform_assignment(
        graph, "FP8_CB_K36", profile=profile, target_profile="nvfp4_cb",
        source_kinds=_bf16_sources(graph))

    # Mixed 4.5 (CB) + 16.0 (the demoted gate_up unit) over covered params only.
    covered = sum(result.params_by_format.values())
    assert covered == sum(
        int(t.shape[0]) * int(t.shape[1])
        for t in graph.quantizable_tensors()
        if t.recipe_name.removesuffix(".weight") in result.assignment
    )
    assert 4.5 < result.achieved_bits < 16.0
