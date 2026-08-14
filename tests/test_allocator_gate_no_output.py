"""Allocator must fail before any output mutation when scope != none with gate=0."""
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.production_weight_cache import build_production_cache_cb_render_identity, bind_cb_render_identity_source_weights
from prismaquant import format_registry as fr


def _build_minimal_fixture(tmp: Path):
    menu = ["NVFP4_CB_K12", "BF16"]
    specs = [fr.get_format(n) for n in menu]
    stats = {
        "model.layers.0.self_attn.o_proj": {"h_trace": 0.4, "n_params": 1024*2048, "in_features": 2048, "out_features": 1024},
        "model.layers.0.mlp.gate_proj": {"h_trace": 0.8, "n_params": 3072*1024, "in_features": 1024, "out_features": 3072},
        "model.layers.0.mlp.up_proj": {"h_trace": 0.6, "n_params": 3072*1024, "in_features": 1024, "out_features": 3072},
        "model.layers.0.mlp.down_proj": {"h_trace": 0.9, "n_params": 1024*3072, "in_features": 3072, "out_features": 1024},
    }
    def _cost_entry(d): return {"weight_mse": d, "predicted_dloss": d}
    costs = {n: {s.name: _cost_entry(0.01) for s in specs} for n in stats}
    col_weights = {n: torch.linspace(0.1, 1.0, int(v["in_features"])) for n, v in stats.items()}
    formats_by_qname = {n: [s.name for s in specs if s.name.startswith("NVFP4_CB")] for n in stats}
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4")
    render_identity = build_production_cache_cb_render_identity(
        formats_by_qname, cb_serialization_context=ctx, col_weights=col_weights, render_levers={"weighted_vq": True}, render_mechanism_plan=[]
    )
    source_weights = {n: torch.zeros((int(v["out_features"]), int(v["in_features"])), dtype=torch.bfloat16) for n, v in stats.items()}
    render_identity = bind_cb_render_identity_source_weights(render_identity, source_weights)
    provenance = {"cb_serialized_payload": render_identity["cb_serialized_payload"], "cb_render_identity": render_identity}
    probe_p = tmp / "probe.pkl"
    cost_p = tmp / "cost.pkl"
    cw_p = tmp / "col.pkl"
    lc = tmp / "layer_config.json"
    csv = tmp / "pareto.csv"
    probe_p.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": None}}))
    cost_p.write_bytes(pickle.dumps({"costs": costs, "formats": menu, "meta": {"formats": menu}, "provenance": provenance}))
    cw_p.write_bytes(pickle.dumps(col_weights))
    return probe_p, cost_p, cw_p, lc, csv, menu


def test_allocator_gate_0_with_scope_nvfp4_fails_before_write(tmp_path, monkeypatch):
    probe_p, cost_p, cw_p, lc, csv, menu = _build_minimal_fixture(tmp_path)
    # Ensure no prior file
    assert not lc.exists()
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "0")
    import prismaquant.allocator as alloc
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", ",".join(menu),
        "--target-bits", "4.0",
        "--pareto-targets", "4.0",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--target-profile", "nvfp4_cb",
        "--allow-default-profile",
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1",
        "--cb-ldlq-scope", "nvfp4",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(cw_p),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit, match="requires PRISMAQUANT_CB_LDLQ_GATE=1"):
            alloc.main()
    finally:
        sys.argv = old_argv
        monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "1")
    # Must not have written layer_config
    assert not lc.exists(), "allocator must not emit layer_config when gate=0 with scope nvfp4"
    assert not csv.exists() or csv.stat().st_size == 0 or not csv.is_file()  # pareto may not exist either


def test_allocator_gate_1_with_scope_nvfp4_succeeds(tmp_path, monkeypatch):
    probe_p, cost_p, cw_p, lc, csv, menu = _build_minimal_fixture(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "1")
    import prismaquant.allocator as alloc
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", ",".join(menu),
        "--target-bits", "4.0",
        "--pareto-targets", "4.0",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--target-profile", "nvfp4_cb",
        "--allow-default-profile",
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1",
        "--cb-ldlq-scope", "nvfp4",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(cw_p),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        alloc.main()
    finally:
        sys.argv = old_argv
    assert lc.exists(), "with gate=1, allocator must emit layer_config"


def test_allocator_gate_0_with_scope_none_succeeds(tmp_path, monkeypatch):
    # scope none should succeed even with gate 0
    probe_p, cost_p, cw_p, lc, csv, menu = _build_minimal_fixture(tmp_path)
    # For scope none, we need provenance with ldlq_scope none
    # Rebuild with none
    import pickle as _pk
    costs = _pk.loads(cost_p.read_bytes())["costs"]
    # Rebuild provenance with none
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext
    from prismaquant.production_weight_cache import build_production_cache_cb_render_identity, bind_cb_render_identity_source_weights
    import torch as _torch
    col_weights = _pk.loads(cw_p.read_bytes())
    stats = _pk.loads((tmp_path / "probe.pkl").read_bytes())["stats"]
    ctx_none = CBSerializationContext.production(ldlq_scope="none")
    formats_by_qname = {n: ["NVFP4_CB_K12"] for n in stats}
    render_identity = build_production_cache_cb_render_identity(
        formats_by_qname, cb_serialization_context=ctx_none, col_weights=col_weights, render_levers={"weighted_vq": True}, render_mechanism_plan=[]
    )
    source_weights = {n: _torch.zeros((int(v["out_features"]), int(v["in_features"])), dtype=_torch.bfloat16) for n, v in stats.items()}
    render_identity = bind_cb_render_identity_source_weights(render_identity, source_weights)
    provenance = {"cb_serialized_payload": render_identity["cb_serialized_payload"], "cb_render_identity": render_identity}
    cost_p.write_bytes(_pk.dumps({"costs": costs, "formats": menu, "meta": {"formats": menu}, "provenance": provenance}))
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "0")
    import prismaquant.allocator as alloc
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", ",".join(menu),
        "--target-bits", "4.0",
        "--pareto-targets", "4.0",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--target-profile", "nvfp4_cb",
        "--allow-default-profile",
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1",
        "--cb-ldlq-scope", "none",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(cw_p),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        alloc.main()
    finally:
        sys.argv = old_argv
        monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "1")
    assert lc.exists()


def test_allocator_env_scope_with_gate_0_fails(tmp_path, monkeypatch):
    # Env scope nvfp4 with gate 0 should also fail (legacy via env)
    probe_p, cost_p, cw_p, lc, csv, menu = _build_minimal_fixture(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_SCOPE", "nvfp4")
    import prismaquant.allocator as alloc
    # Use CLI without scope, relying on env
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", ",".join(menu),
        "--target-bits", "4.0",
        "--pareto-targets", "4.0",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--target-profile", "nvfp4_cb",
        "--allow-default-profile",
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-scale-sweep", "1",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(cw_p),
        # no --cb-ldlq-scope, should pick up from env
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit, match="requires PRISMAQUANT_CB_LDLQ_GATE=1"):
            alloc.main()
    finally:
        sys.argv = old_argv
        monkeypatch.delenv("PRISMAQUANT_CB_LDLQ_SCOPE", raising=False)
        monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "1")
    assert not lc.exists()
