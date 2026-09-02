"""Source-value contract for the exact-4.5 Stage-0 format screen."""
from __future__ import annotations

from pathlib import Path

import torch

from scripts import ab_nvfp4_vs_k36_dense as stage0


def test_stage0_resolves_the_active_checkout() -> None:
    expected = Path(stage0.__file__).resolve().parents[1]
    assert stage0.REPO_ROOT == expected


def test_stage0_loads_through_canonical_source_decoder() -> None:
    calls: list[str] = []

    class Source:
        def dequant_weight(self, name: str) -> torch.Tensor:
            calls.append(name)
            return torch.tensor([[1.25, -0.5]], dtype=torch.float32)

    got = stage0.load_source_weight(Source(), "layers.0.attn.q.weight", "cpu")

    assert calls == ["layers.0.attn.q.weight"]
    assert got.dtype is torch.bfloat16
    assert torch.equal(got.float(), torch.tensor([[1.25, -0.5]]))


def test_stage0_classifies_dsv4_shared_experts_separately() -> None:
    assert stage0.unit_role(
        "model.layers.7.mlp.shared_experts.down_proj"
    ) == "shared"


def test_stage0_indexes_profile_rewritten_checkpoint_names() -> None:
    class Profile:
        @staticmethod
        def checkpoint_to_live_name(name: str) -> str | None:
            if name == "layers.7.attn.q_proj.weight":
                return "model.layers.7.self_attn.q_proj.weight"
            return None

    headers = {"layers.7.attn.q_proj.weight": ({}, {})}
    index = stage0.build_source_index(headers, Profile())

    assert stage0.resolve_source_key(
        index, headers, "model.layers.7.self_attn.q_proj"
    ) == "layers.7.attn.q_proj.weight"
