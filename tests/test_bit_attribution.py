from __future__ import annotations

import json

from prismaquant.allocator import (
    _build_bit_attribution,
    _parse_role_from_qname,
    _write_bit_attribution_reports,
)
from prismaquant.allocator_solver import Candidate


def test_parse_role_distinguishes_dense_attn_mlp():
    assert _parse_role_from_qname("model.layers.3.self_attn.q_proj") == "q_proj"
    assert _parse_role_from_qname("model.layers.3.self_attn.o_proj") == "o_proj"
    assert _parse_role_from_qname("model.layers.3.mlp.gate_proj") == "gate_proj"
    assert _parse_role_from_qname("model.layers.3.mlp.down_proj") == "down_proj"


def test_parse_role_distinguishes_experts_from_dense():
    # Unpacked routed expert keeps an expert.* prefix so it never collapses
    # into the dense MLP bucket.
    assert _parse_role_from_qname(
        "model.layers.3.mlp.experts.7.down_proj") == "expert.down_proj"
    # Packed routed expert (3D param, no per-expert index).
    assert _parse_role_from_qname(
        "model.layers.3.mlp.experts.gate_up_proj") == "expert.gate_up_proj"
    # Shared expert.
    assert _parse_role_from_qname(
        "model.layers.3.mlp.shared_experts.up_proj") == "shared_expert.up_proj"


def test_parse_role_unknown_fallback():
    assert _parse_role_from_qname("some.weird.module") == "unknown"
    assert _parse_role_from_qname("lm_head") == "lm_head"


def test_build_bit_attribution_buckets_and_body_totals():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
        "model.layers.1.mlp.down_proj": "BF16",
        # Auxiliary entries must be excluded from the body buckets/totals.
        "mtp.0.self_attn.q_proj": "BF16",
        "model.visual.blocks.0.attn.qkv": "BF16",
    }
    candidates = {
        "model.layers.0.self_attn.q_proj": [Candidate("NVFP4", 4.0, 512, 0.01)],
        "model.layers.0.mlp.down_proj": [Candidate("FP8_E4M3", 8.0, 2048, 0.002)],
        "model.layers.1.mlp.down_proj": [Candidate("BF16", 16.0, 4096, 0.0)],
    }
    stats = {
        "model.layers.0.self_attn.q_proj": {"n_params": 1024, "h_trace": 0.5},
        "model.layers.0.mlp.down_proj": {"n_params": 2048, "h_trace": 9.0},
        "model.layers.1.mlp.down_proj": {"n_params": 2048, "h_trace": 1.0},
    }

    buckets, per_linear, totals = _build_bit_attribution(
        assignment, candidates, stats.get, format_specs={})

    # mtp + visual excluded.
    assert totals["n_body_linears"] == 3
    qnames = {r["qname"] for r in per_linear}
    assert not any("mtp." in q or "visual" in q for q in qnames)

    # bits = 8 * memory_bytes; bpp = bits / n_params.
    q = next(r for r in per_linear if r["role"] == "q_proj")
    assert q["bits"] == 8.0 * 512
    assert q["bpp"] == (8.0 * 512) / 1024
    assert q["predicted_dloss"] == 0.01
    assert q["h_trace"] == 0.5

    # one bucket per (block, role); down_proj appears in two different blocks.
    keys = {(b["block_id"], b["role"]) for b in buckets}
    assert ("model.layers.0", "down_proj") in keys
    assert ("model.layers.1", "down_proj") in keys

    # body totals only count the 3 body linears.
    assert totals["body_quantizable_params"] == 1024 + 2048 + 2048


def test_build_bit_attribution_null_dloss_when_no_candidate():
    # Expanded fused-sibling member: in the assignment but no scored candidate.
    assignment = {"model.layers.0.self_attn.k_proj": "NVFP4"}
    stats = {"model.layers.0.self_attn.k_proj": {
        "n_params": 1024, "h_trace": 0.3,
        "_memory_bytes_by_format": {"NVFP4": 512},
    }}
    buckets, per_linear, totals = _build_bit_attribution(
        assignment, candidates={}, stats_entry_for=stats.get, format_specs={})
    row = per_linear[0]
    # No fabricated dloss; bits still recovered from the memory map.
    assert row["predicted_dloss"] is None
    assert row["bits"] == 8.0 * 512
    assert buckets[0]["sum_predicted_dloss"] is None
    assert buckets[0]["predicted_dloss_coverage"] == "0/1"


def test_write_bit_attribution_reports_emit_files(tmp_path):
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
    }
    candidates = {
        "model.layers.0.self_attn.q_proj": [Candidate("NVFP4", 4.0, 512, 0.01)],
        "model.layers.0.mlp.down_proj": [Candidate("FP8_E4M3", 8.0, 2048, 0.002)],
    }
    stats = {
        "model.layers.0.self_attn.q_proj": {"n_params": 1024, "h_trace": 0.5},
        "model.layers.0.mlp.down_proj": {"n_params": 2048, "h_trace": 9.0},
    }
    jpath = tmp_path / "attr.json"
    cpath = tmp_path / "attr.csv"
    _write_bit_attribution_reports(
        str(jpath), str(cpath),
        target_bits=5.0, achieved_bits=4.9,
        assignment_expanded=assignment, candidates=candidates,
        stats_entry_for=stats.get, format_specs={})

    payload = json.loads(jpath.read_text())
    assert payload["schema"] == "prismaquant.allocator.bit_attribution.v1"
    assert payload["n_body_linears"] == 2
    assert len(payload["buckets"]) == 2
    csv_text = cpath.read_text()
    assert "qname,block_id,role,format,bits,bpp,n_params,h_trace,predicted_dloss" in csv_text
    assert "q_proj" in csv_text


def test_write_bit_attribution_noop_when_no_paths():
    # Should not raise and should write nothing when both paths are None.
    _write_bit_attribution_reports(
        None, None, target_bits=5.0, achieved_bits=4.9,
        assignment_expanded={}, candidates={}, stats_entry_for=lambda n: None,
        format_specs={})
