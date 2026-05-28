"""Pluggable per-expert projection names for batched-Fisher MoE detection.

The batched-Fisher block detector keys on the per-expert module attribute names
returned by the model profile. The default is ('w1','w2','w3') so Qwen3/Qwen3.5
behavior is unchanged; a profile that exposes gate/up/down_proj experts must be
detected instead of silently no-opping the batched path (probe-speed only).
"""
from __future__ import annotations

import torch.nn as nn

from prismaquant.model_profiles.base import ModelProfile
from prismaquant.sensitivity_probe import FisherAccumulator


class _Profile(ModelProfile):
    name = "test-default"

    def __init__(self, proj_names):
        self._proj = tuple(proj_names)

    def matches(self, *a, **k):
        return False

    def structure_spec(self):
        return None

    def unpacked_expert_projection_names(self):
        return self._proj


class _Expert(nn.Module):
    def __init__(self, proj_names):
        super().__init__()
        for n in proj_names:
            setattr(self, n, nn.Linear(8, 8, bias=False))


class _MoEModel(nn.Module):
    def __init__(self, proj_names, n_experts=3):
        super().__init__()
        self.experts = nn.ModuleList(_Expert(proj_names) for _ in range(n_experts))


def _tracked_names(proj_names, n_experts=3):
    return [
        f"experts.{e}.{w}"
        for e in range(n_experts)
        for w in proj_names
    ]


def test_default_profile_projection_names_are_w1_w2_w3():
    prof = _Profile(("w1", "w2", "w3"))
    assert prof.unpacked_expert_projection_names() == ("w1", "w2", "w3")


def test_base_profile_default_unpacked_names():
    # The base ModelProfile (no override) defaults to the Qwen standard.
    class _Bare(ModelProfile):
        name = "bare"

        def matches(self, *a, **k):
            return False

        def structure_spec(self):
            return None

    assert _Bare().unpacked_expert_projection_names() == ("w1", "w2", "w3")


def test_batched_fisher_detects_w1_w2_w3_experts_with_default():
    proj = ("w1", "w2", "w3")
    model = _MoEModel(proj)
    acc = FisherAccumulator(
        model, tracked=_tracked_names(proj), expert_info={},
        model_profile=_Profile(proj))
    # The experts ModuleList is detected as a batched-Fisher MoE block.
    assert acc._moe_block_to_linears.get("experts")
    assert len(acc._moe_block_to_linears["experts"]) == 9  # 3 experts × 3 proj


def test_batched_fisher_detects_alternate_named_experts_via_profile():
    # Experts exposing gate/up/down_proj are detected ONLY because the profile
    # returns those names — the whole point of the pluggability fix.
    proj = ("gate_proj", "up_proj", "down_proj")
    model = _MoEModel(proj)
    acc = FisherAccumulator(
        model, tracked=_tracked_names(proj), expert_info={},
        model_profile=_Profile(proj))
    assert acc._moe_block_to_linears.get("experts")
    assert len(acc._moe_block_to_linears["experts"]) == 9


def test_alternate_named_experts_no_op_under_default_w1w2w3_profile():
    # Same gate/up/down_proj model, but a profile that still says w1/w2/w3:
    # detection must NOT fire (proving the names are what gates it). Per-Linear
    # Fisher hooks still cover these — this only loses the batched speedup.
    proj = ("gate_proj", "up_proj", "down_proj")
    model = _MoEModel(proj)
    acc = FisherAccumulator(
        model, tracked=_tracked_names(proj), expert_info={},
        model_profile=_Profile(("w1", "w2", "w3")))
    assert not acc._moe_block_to_linears.get("experts")
