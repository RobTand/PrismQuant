from __future__ import annotations

import pytest
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import aggregate_fused_siblings, build_candidates
from prismaquant.grouped_kl_cost import (
    discover_grouped_kl_units,
    synthesize_grouped_cost_payload,
)


class _FakeProfile:
    def fused_sibling_group(self, qname: str) -> str | None:
        if qname.endswith((".q_proj", ".k_proj", ".v_proj")):
            return qname.rsplit(".", 1)[0] + ".qkv_proj"
        if qname.endswith((".gate_proj", ".up_proj")):
            return qname.rsplit(".", 1)[0] + ".gate_up_proj"
        return None

    def is_pinned_name(self, _qname: str) -> bool:
        return False

    def live_to_recipe_name(self, qname: str) -> str:
        return qname


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(64, 32, bias=False)
        self.self_attn.k_proj = nn.Linear(64, 32, bias=False)
        self.self_attn.v_proj = nn.Linear(64, 32, bias=False)
        self.self_attn.o_proj = nn.Linear(32, 64, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(64, 64, bias=False)
        self.mlp.up_proj = nn.Linear(64, 64, bias=False)
        self.mlp.down_proj = nn.Linear(64, 64, bias=False)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock()])


def test_discover_grouped_kl_units_uses_fused_sibling_groups():
    units, diag = discover_grouped_kl_units(_TinyModel(), _FakeProfile())
    by_name = {unit.name: unit.members for unit in units}

    assert by_name["layers.0.self_attn.qkv_proj"] == (
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
    )
    assert by_name["layers.0.mlp.gate_up_proj"] == (
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    )
    assert by_name["layers.0.self_attn.o_proj"] == (
        "layers.0.self_attn.o_proj",
    )
    assert diag["unit_count"] == 4


def test_synthesize_grouped_cost_shares_group_kl_and_preserves_fallbacks():
    grouped = {
        "schema": "prismaquant.grouped_kl_cost.v1",
        "groups": {
            "layers.0.self_attn.qkv_proj": [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.v_proj",
            ],
        },
        "results": {
            "layers.0.self_attn.qkv_proj": {
                "NVFP4": 0.9,
                "MXFP8_E4M3": 0.3,
                "FP8_E4M3": 0.15,
                "BF16": 0.0,
            },
        },
        "kl_scope": "full_sequence",
    }
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16"],
        "costs": {
            "layers.0.self_attn.q_proj": {
                "NVFP4": {"predicted_dloss": 10.0},
                "MXFP8_E4M3": {"predicted_dloss": 1.0},
                "FP8_E4M3": {"predicted_dloss": 0.5},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.self_attn.k_proj": {
                "NVFP4": {"predicted_dloss": 20.0},
                "MXFP8_E4M3": {"predicted_dloss": 2.0},
                "FP8_E4M3": {"predicted_dloss": 1.0},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.self_attn.v_proj": {
                "NVFP4": {"predicted_dloss": 30.0},
                "MXFP8_E4M3": {"predicted_dloss": 3.0},
                "FP8_E4M3": {"predicted_dloss": 1.5},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.self_attn.o_proj": {
                "NVFP4": {"predicted_dloss": 40.0},
                "MXFP8_E4M3": {"predicted_dloss": 4.0},
                "FP8_E4M3": {"predicted_dloss": 2.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    cost = synthesize_grouped_cost_payload(grouped, baseline)

    q = cost["costs"]["layers.0.self_attn.q_proj"]
    assert q["NVFP4"]["predicted_dloss"] == 0.3
    assert abs(q["MXFP8_E4M3"]["predicted_dloss"] - 0.1) < 1e-12
    assert abs(q["FP8_E4M3"]["predicted_dloss"] - 0.05) < 1e-12
    assert q["NVFP4"]["cost_source"] == "grouped_kl_share"
    assert q["BF16"]["predicted_dloss"] == 0.0

    o = cost["costs"]["layers.0.self_attn.o_proj"]
    assert o["NVFP4"]["predicted_dloss"] == 40.0
    assert o["NVFP4"]["cost_source"] == "fallback_baseline"
    assert cost["meta"]["grouped_entries"] == 9


def test_grouped_kl_share_sums_back_to_group_cost_after_aggregation():
    grouped = {
        "schema": "prismaquant.grouped_kl_cost.v1",
        "groups": {
            "layers.0.self_attn.qkv_proj": [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.v_proj",
            ],
        },
        "results": {
            "layers.0.self_attn.qkv_proj": {
                "NVFP4": 0.9,
                "MXFP8_E4M3": 0.3,
                "BF16": 0.0,
            },
        },
        "kl_scope": "full_sequence",
    }
    members = grouped["groups"]["layers.0.self_attn.qkv_proj"]
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        "costs": {
            name: {
                "NVFP4": {"predicted_dloss": 10.0},
                "MXFP8_E4M3": {"predicted_dloss": 1.0},
                "BF16": {"predicted_dloss": 0.0},
            }
            for name in members
        },
    }
    cost = synthesize_grouped_cost_payload(grouped, baseline)
    stats = {
        name: {
            "h_trace": 1.0,
            "n_params": 128 * 128,
            "in_features": 128,
            "out_features": 128,
        }
        for name in members
    }
    specs = [fr.get_format(name) for name in baseline["formats"]]
    candidates = build_candidates(stats, cost["costs"], specs)

    _, _, aggregated = aggregate_fused_siblings(
        stats,
        cost["costs"],
        specs,
        candidates,
        _FakeProfile(),
    )

    super_name = next(name for name in aggregated if "qkv_proj" in name)
    by_fmt = {candidate.fmt: candidate for candidate in aggregated[super_name]}
    assert by_fmt["NVFP4"].predicted_dloss == pytest.approx(grouped["results"][
        "layers.0.self_attn.qkv_proj"
    ]["NVFP4"])
    assert by_fmt["MXFP8_E4M3"].predicted_dloss == pytest.approx(grouped["results"][
        "layers.0.self_attn.qkv_proj"
    ]["MXFP8_E4M3"])
    assert by_fmt["BF16"].predicted_dloss == 0.0
