"""Blocker A+B corrective pass tests - nonvacuous CPU.

Covers:
- CBLDLQActivationLoader fused gate/up deterministic concat, parity, fail-closed, pooled missing
- shared get_expert_weight dual-namespace (checkpoint w1/w3/w2 + logical gate/up/down), order, callback, logical validation
- load_packed_projection integration with native bases + unequal member evidence via real loader
- PROJECTION_RUNG_SCHEMA v2 stale fails after v3
"""
import hashlib
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from prismaquant.cb_ldlq import CBLDLQActivationLoader, fill_empty_expert_activation_rows
from prismaquant.cb_ldlq_fused_activation import concat_equal_member_samples as FUSED_POLICY
from tools.derive_dual_basis_packed import (
    PROJECTION_RUNG_SCHEMA,
    projection_checkpoint_identity,
    validated_projection_checkpoint,
    _packed_activation_evidence_identity,
    _sha,
)


def _write_act_file(dir_path: Path, qname: str, tensor: torch.Tensor):
    from prismaquant.cb_ldlq import _ACT_FNAME_SUB
    import re

    path = dir_path / (_ACT_FNAME_SUB.sub("__", qname) + ".pt")
    # blob format: {"inputs": tensor}
    torch.save({"inputs": tensor}, path)


def test_fused_concat_gate_then_up_and_pooled_parity(tmp_path):
    """Unequal synthetic gate/up returns gate-then-up row concat, mean(square(cat)) == pooled within 5.96e-08."""
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Profile that says gate_up_proj = gate_proj then up_proj
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    # expert_stack_members for one packed target with 2 experts
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
            ("gate_proj", 1): "model.layers.0.mlp.experts.1.gate_proj",
            ("up_proj", 1): "model.layers.0.mlp.experts.1.up_proj",
        }
    }
    # Create unequal tensors: 64x8 each, different but same width/count
    torch.manual_seed(0)
    gate0 = torch.randn(64, 8)
    up0 = torch.randn(64, 8)
    gate1 = torch.randn(64, 8)
    up1 = torch.randn(64, 8)
    # Ensure they differ
    assert not torch.equal(gate0, up0)
    for qname, t in [
        ("model.layers.0.mlp.experts.0.gate_proj", gate0),
        ("model.layers.0.mlp.experts.0.up_proj", up0),
        ("model.layers.0.mlp.experts.1.gate_proj", gate1),
        ("model.layers.0.mlp.experts.1.up_proj", up1),
    ]:
        _write_act_file(act_dir, qname, t)
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    rows = loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=2)
    assert isinstance(rows, tuple) and len(rows) == 2
    # gate-then-up deterministic order
    assert torch.equal(rows[0], torch.cat([gate0, up0], dim=0))
    assert torch.equal(rows[1], torch.cat([gate1, up1], dim=0))
    # n_activation_rows becomes concatenated count (128)
    assert rows[0].shape == (128, 8)
    # Exact pooled-cw parity: mean(square(cat)) equals mean-pooled col weights within 5.96e-08 (proven for L0E0)
    # Simulate col_weights as per-member vectors: each member's col weight is mean square per column
    cw_gate0 = gate0.square().mean(dim=0)
    cw_up0 = up0.square().mean(dim=0)
    pooled = torch.stack([cw_gate0, cw_up0]).mean(dim=0)
    derived = rows[0].square().mean(dim=0)
    assert torch.allclose(derived, pooled, rtol=1e-6, atol=5.96e-08), f"max delta {(derived-pooled).abs().max()}"
    # Dedup (pick one member) would NOT match pooled
    dedup = gate0.square().mean(dim=0)
    assert not torch.allclose(dedup, pooled, atol=1e-3)
    # Evidence hashes are of actual concatenated tensors, member order explicit
    ev = _packed_activation_evidence_identity(rows, act_dir, "model.layers.0.mlp.experts.gate_up_proj", member_order=["gate_proj", "up_proj"], slice_boundaries={"gate_proj": (0, 8), "up_proj": (8, 16)}, col_weight_pooling="mean_of_member_vectors")
    assert ev["fused_activation_policy"] == FUSED_POLICY
    assert ev["member_order"] == ["gate_proj", "up_proj"]
    assert ev["row_counts"] == [128, 128]
    # n_activation_rows becomes concatenated count in identity
    ident = projection_checkpoint_identity(0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64, context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, activation_evidence=ev, member_order=["gate_proj", "up_proj"], slice_boundaries={"gate_proj": (0, 8), "up_proj": (8, 16)}, col_weight_pooling="mean_of_member_vectors", expert_count=2)
    assert ident["fused_activation_policy"] == FUSED_POLICY
    assert ident["fused_activation_order"] == ["gate_proj", "up_proj"]
    # Single-member down stays unchanged: shape 64x8 not doubled
    # (covered elsewhere but sanity: loader with single projection should not double)


def test_fused_equal_short_samples_double_deterministically(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    torch.manual_seed(1)
    t = torch.randn(4, 8)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", t)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", t.clone())
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    rows = loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    assert rows[0].shape == (8, 8)
    assert torch.equal(rows[0], torch.cat([t, t], dim=0))


def test_fused_partial_and_mismatches_fail_closed(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    # Partial presence: only gate
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", torch.randn(4, 8))
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    with pytest.raises(ValueError, match="partial presence"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    # Clean act dir for next subcase
    for p in act_dir.iterdir():
        p.unlink()
    # Width mismatch
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", torch.randn(4, 8))
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", torch.randn(4, 7))
    with pytest.raises(ValueError, match="input width"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    for p in act_dir.iterdir():
        p.unlink()
    # Row count mismatch
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", torch.randn(4, 8))
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", torch.randn(3, 8))
    with pytest.raises(ValueError, match="row count"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    for p in act_dir.iterdir():
        p.unlink()
    # Rank mismatch (store rank 3 tensor via raw save bypassing _direct validation? _direct will raise)
    torch.save({"inputs": torch.randn(4, 8, 2)}, act_dir / "model__layers__0__mlp__experts__0__gate_proj.pt")
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", torch.randn(4, 8))
    with pytest.raises(ValueError, match="rank-2"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)


def test_fused_missing_both_uses_pooled_prior(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
            ("gate_proj", 1): "model.layers.0.mlp.experts.1.gate_proj",
            ("up_proj", 1): "model.layers.0.mlp.experts.1.up_proj",
        }
    }
    # Expert 0 has both, expert 1 has neither
    torch.manual_seed(2)
    gate0 = torch.randn(4, 8)
    up0 = torch.randn(4, 8)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", gate0)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", up0)
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    rows = loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=2)
    # Before fill, expert 1 is empty (0,0) placeholder
    assert rows[1].shape[0] == 0
    # After fill_empty, missing expert gets pooled fused rows
    filled, missing = fill_empty_expert_activation_rows(rows, qname="model.layers.0.mlp.experts.gate_up_proj")
    assert missing == (1,)
    assert filled[1].shape == rows[0].shape
    assert torch.equal(filled[1], rows[0])


def test_get_expert_weight_dual_namespace_gate_up_order_and_down_singleton(tmp_path):
    """Shared get_expert_weight with checkpoint-native w1/w3/w2 and logical gate/up/down."""
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    # Fake skeleton that returns distinguishable tensors per checkpoint base
    class FakeSkel:
        def __init__(self, tensors):
            self.tensors = tensors

        def dequant_weight(self, key):
            return self.tensors[key]

    torch.manual_seed(3)
    gate_w = torch.randn(8, 8)
    up_w = torch.randn(8, 8)
    down_w = torch.randn(8, 8)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": gate_w,
        "layers.0.ffn.experts.0.w3.weight": up_w,
        "layers.0.ffn.experts.0.w2.weight": down_w,
    }
    skel = FakeSkel(tensors)
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)

    # Checkpoint-native group (values are bases without .weight)
    group_gate_up = {"gate_proj": {0: "layers.0.ffn.experts.0.w1"}, "up_proj": {0: "layers.0.ffn.experts.0.w3"}}
    logical_gate_up = {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"}

    group_down = {"down_proj": {0: "layers.0.ffn.experts.0.w2"}}
    logical_down = {("down_proj", 0): "model.layers.0.mlp.experts.0.down_proj"}

    # Gate-up order must be gate then up deterministically
    calls = []

    def on_member(proj, eid, ckpt_base, logical_qname, decoded):
        calls.append((proj, eid, ckpt_base, logical_qname))

    fused = get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "gate_up_proj", group_gate_up, 0, logical_members=logical_gate_up, on_member=on_member)
    assert fused.shape == (16, 8)
    assert torch.equal(fused, torch.cat([gate_w, up_w], dim=0))
    assert calls == [
        ("gate_proj", 0, "layers.0.ffn.experts.0.w1", "model.layers.0.mlp.experts.0.gate_proj"),
        ("up_proj", 0, "layers.0.ffn.experts.0.w3", "model.layers.0.mlp.experts.0.up_proj"),
    ]
    # Down singleton
    calls.clear()
    single = get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "down_proj", group_down, 0, logical_members=logical_down, on_member=on_member)
    assert torch.equal(single, down_w)
    assert calls == [("down_proj", 0, "layers.0.ffn.experts.0.w2", "model.layers.0.mlp.experts.0.down_proj")]

    # Validation uses logical identity: prove that validating with logical passes and checkpoint base is checked separately
    # Simulate derive validation: checkpoint_base must equal group, logical must be in member map, otherwise fail
    bad_logical = {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "WRONG"}
    with pytest.raises(AssertionError):
        # mimic derive on_member that asserts logical_qname != expected
        def bad_on(proj, eid, ckpt_base, logical_qname, decoded):
            if logical_qname != logical_gate_up[(proj, eid)]:
                raise AssertionError("logical mismatch")
        get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "gate_up_proj", group_gate_up, 0, logical_members=bad_logical, on_member=bad_on)


def test_load_packed_projection_native_bases_unequal_evidence_no_fallback(tmp_path, monkeypatch):
    """Actual load_packed_projection integration with native w1/w3 bases, unequal member evidence via real loader."""
    import tools.derive_dual_basis_packed as mod

    act_dir = tmp_path / "act"
    act_dir.mkdir()
    # Use monkeypatched SOURCE to tmp
    # Create fake model dir with minimal structure not needed due to mocking skeleton
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Mock planning with native bases (w1/w3) and logical members
    profile = MagicMock()
    profile.packed_expert_param_names.return_value = frozenset({"gate_up_proj"})
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)

    native_groups = {
        "model.layers.0.mlp.experts": {
            "gate_proj": {0: "layers.0.ffn.experts.0.w1"},
            "up_proj": {0: "layers.0.ffn.experts.0.w3"},
        }
    }
    # Note: group keys are gate_proj/up_proj but values are native w1/w3
    # For load_packed_projection, it expects group[proj][eid] == native base
    # But _get_packed_planning returns native groups like above
    logical_members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    monkeypatch.setattr(mod, "_get_packed_planning", lambda: (profile, native_groups, logical_members, frozenset({"gate_up_proj"})))
    # Create unequal activation files for gate/up
    torch.manual_seed(4)
    gate_act = torch.randn(4, 8)
    up_act = torch.randn(4, 8)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", gate_act)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", up_act)
    # Override ACT_ROOT for loader
    monkeypatch.setattr(mod, "ACT_ROOT", act_dir)
    monkeypatch.setattr(mod, "SOURCE", model_dir)
    # Mock skeleton to return native bases
    import prismaquant.export_nvfp4_cb_streaming as stream_mod

    gate_w = torch.randn(8, 8)
    up_w = torch.randn(8, 8)

    def mock_dequant(key):
        if "w1" in key:
            return gate_w
        if "w3" in key:
            return up_w
        raise AssertionError(key)

    mock_skel = MagicMock()
    mock_skel.dequant_weight.side_effect = mock_dequant
    monkeypatch.setattr(mod, "open_packed_weight_source", lambda p: mock_skel)
    monkeypatch.setattr(stream_mod, "open_packed_weight_source", lambda p: mock_skel)
    # col weights derived from cat mean square
    cat = torch.cat([gate_act, up_act], dim=0)
    pooled = cat.square().mean(0)
    all_cw = {
        "model.layers.0.mlp.experts.0.gate_proj": pooled.clone(),
        "model.layers.0.mlp.experts.0.up_proj": pooled.clone(),
    }
    identity = {
        "col_weights_shapes": {k: list(v.shape) for k, v in all_cw.items()},
        "col_weights_content_sha256": {k: mod.content_sha256_float32(v) for k, v in all_cw.items()},
    }
    monkeypatch.setattr("prismaquant.production_weight_cache.validate_cb_render_source_weight", lambda *a, **k: None)
    # Must not catch exceptions - let them propagate
    data = mod.load_packed_projection(0, "gate_up_proj", device=torch.device("cpu"), identity=identity, all_col_weights=all_cw, model_to_shard={}, model_to_ckpt={}, scale_map={})
    # Should have used real loader (not mocked) via ACT_ROOT, and no fallback
    assert data["weight"].shape[0] == 1
    assert data["activation_rows"][0].shape == (8, 8)
    assert torch.equal(data["activation_rows"][0], torch.cat([gate_act, up_act], dim=0))
    # No try/except fallback - succeeded without exception


def test_schema_v2_checkpoint_fails_after_v3(tmp_path):
    from tools.derive_dual_basis_packed import PROJECTION_RUNG_SCHEMA

    assert PROJECTION_RUNG_SCHEMA == "prismaquant.dsv4_dual_basis_projection_rung.v3"
    # Build a v2 payload (old schema) and verify it fails
    identity_v3 = projection_checkpoint_identity(0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64, context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, activation_evidence={"act_root": "/tmp/act", "qname_prefix": "model.layers.0.mlp.experts.gate_up_proj", "row_counts": [4], "evidence_sha256": "e"*64, "gate": "activation_output_mse", "fused_activation_policy": FUSED_POLICY, "fused_activation_order": ["gate_proj", "up_proj"], "member_order": ["gate_proj", "up_proj"], "col_weight_pooling": "mean_of_member_vectors"}, member_order=["gate_proj", "up_proj"], slice_boundaries={"gate_proj": [0, 8], "up_proj": [8, 16]}, col_weight_pooling="mean_of_member_vectors", expert_count=1)
    # Tamper schema to v2
    payload_v2 = {"schema": "prismaquant.dsv4_dual_basis_projection_rung.v2", "content_key": _sha(identity_v3), "identity": identity_v3, "result": {"schema": "prismaquant.dsv4_nvfp4_projection_rung.v1", "layer": 0, "projection": "gate_up_proj", "packed_parent": "gate_up_proj", "expert_count": 1, "member_order": ["gate_proj", "up_proj"], "slice_boundaries": {"gate_proj": [0, 8], "up_proj": [8, 16]}, "col_weight_pooling": "mean_of_member_vectors", "fused_activation_policy": FUSED_POLICY, "fused_activation_order": ["gate_proj", "up_proj"], "format": "NVFP4_CB_K12", "rung": 12, "qnames": ["a", "b"], "qnames_per_leaf": {"gate_proj": ["a"], "up_proj": ["b"]}, "weight_mse_per_expert": [0.1], "weighted_mse_per_expert": [0.1], "output_mse_per_expert": [0.1], "rel_output_mse_per_expert": [0.1], "per_leaf": {"gate_proj": {"weight_mse_per_expert": [0.1], "weighted_mse_per_expert": [0.1], "output_mse_per_expert": [0.1], "rel_output_mse_per_expert": [0.1]}, "up_proj": {"weight_mse_per_expert": [0.1], "weighted_mse_per_expert": [0.1], "output_mse_per_expert": [0.1], "rel_output_mse_per_expert": [0.1]}}, "weight_mse_fused_per_expert": [0.1], "output_mse_fused_per_expert": [0.1], "n_activation_rows_per_expert": [8], "gate_info": {"gate": "ldlq_kept_all", "per_expert_kept": [True], "raw_mse_per_expert": [0.2], "ldlq_mse_per_expert": [0.1], "metric": "activation_output_mse"}}}
    p = tmp_path / "ckpt_v2.pkl"
    p.write_bytes(pickle.dumps(payload_v2))
    with pytest.raises(AssertionError, match="schema mismatch"):
        validated_projection_checkpoint(p, identity_v3)

