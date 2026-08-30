"""Pinning test for the pre-fix packed-Fisher probe guard."""
import os
import pickle
import re


import pytest
import torch

from prismaquant.measure_quant_cost import prepare_cost_context


def _write_probe(tmp_path, meta, stats):
    p = tmp_path / "probe.pkl"
    with open(p, "wb") as f:
        pickle.dump({"stats": stats, "meta": meta}, f)
    return str(p)


def _write_act_cache(tmp_path, stats):
    """Populate a minimally-valid probe activation cache for ``stats``.

    ``prepare_cost_context`` fails fast (ed4651b) when the act dir maps 0
    Linears, so the fixtures must write the same `<mangled-name>.pt` blobs the
    probe stage writes: for a packed-expert entry the activations are keyed by
    the experts MODULE qname, not the per-projection leaf.
    """
    act = tmp_path / "act"
    act.mkdir(exist_ok=True)
    for name, meta in stats.items():
        key = name
        if isinstance(meta, dict) and meta.get("_packed_experts_module"):
            key = meta["_packed_experts_module"]
        fname = re.sub(r"[^A-Za-z0-9_-]", "__", key) + ".pt"
        torch.save({"inputs": torch.zeros(2, 4)}, act / fname)
    return str(act)


PACKED = {"model.layers.0.mlp.experts.gate_up_proj": {
    "h_trace": 1.0, "_packed_experts_module": "model.layers.0.mlp.experts"}}
DENSE = {"model.layers.0.q_proj": {"h_trace": 1.0}}


def test_stale_packed_probe_refused(tmp_path):
    p = _write_probe(tmp_path, {}, PACKED)
    act = _write_act_cache(tmp_path, PACKED)
    with pytest.raises(SystemExit, match="sum-then-square"):
        prepare_cost_context(p, act, "NVFP4", True)


def test_stamped_packed_probe_accepted(tmp_path):
    p = _write_probe(tmp_path, {"packed_fisher_estimator": "per_token_v2"}, PACKED)
    act = _write_act_cache(tmp_path, PACKED)
    prepare_cost_context(p, act, "NVFP4", True)


def test_dense_probe_unaffected(tmp_path):
    p = _write_probe(tmp_path, {}, DENSE)
    act = _write_act_cache(tmp_path, DENSE)
    prepare_cost_context(p, act, "NVFP4", True)


def test_reader_only_fp8_cb_rung_is_refused_by_explicit_cost_menu(tmp_path):
    p = _write_probe(tmp_path, {}, DENSE)
    act = _write_act_cache(tmp_path, DENSE)
    with pytest.raises(SystemExit, match="reader-only"):
        prepare_cost_context(p, act, "FP8_CB_K29", True)


def test_escape_env_accepts_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER", "1")
    p = _write_probe(tmp_path, {}, PACKED)
    act = _write_act_cache(tmp_path, PACKED)
    prepare_cost_context(p, act, "NVFP4", True)
