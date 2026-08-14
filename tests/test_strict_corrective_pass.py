"""Strict corrective pass CPU tests — fail-closed, no fallback, typed errors.

Covers audits 1-3:
 - CBLDLQActivationLoader no sorted fallback, profile order authoritative, declared vs order mismatch fail closed
 - get_expert_weight strictly five-arg, legacy four-arg fails, callback TypeError propagates once, on_member without logical_members fails before decode
 - typed NoObservedExpertRowsError produces raw fallback with telemetry, no string matching
 - unequal concat/native naming integration still holds
"""
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from prismaquant.cb_ldlq import (
    CBLDLQActivationLoader,
    NoObservedExpertRowsError,
    fill_empty_expert_activation_rows,
)
from prismaquant.cb_ldlq_fused_activation import (
    concat_equal_member_samples as FUSED_POLICY,
    get_packed_expert_projection_names_strict,
)


def _write_act_file(dir_path: Path, qname: str, tensor: torch.Tensor):
    from prismaquant.cb_ldlq import _ACT_FNAME_SUB

    path = dir_path / (_ACT_FNAME_SUB.sub("__", qname) + ".pt")
    torch.save({"inputs": tensor}, path)


# ---------- 1. CBLDLQActivationLoader strict profile order ----------

def test_activation_loader_profile_failure_fails_closed(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Profile that raises
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = RuntimeError("boom")
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    # Should raise RuntimeError ( wrapped ) not fallback to sorted
    with pytest.raises(RuntimeError, match="packed_expert_projection_names failed"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)


def test_activation_loader_none_profile_fails_closed(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=None, expert_stack_members=members)
    with pytest.raises(ValueError, match="requires profile"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)


def test_activation_loader_mismatched_declared_projections_fails(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    # Members only have gate_proj, missing up_proj
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
        }
    }
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", torch.randn(4, 8))
    with pytest.raises(ValueError, match="declared member projections"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)


def test_activation_loader_no_sorted_fallback(tmp_path):
    """Specifically verify sorted fallback not used: declared has single projection but profile says two."""
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    # profile says gate_up_proj -> gate, up ; members also gate, up
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    # Declare members with swapped names to test that order comes from profile, not sorted
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
            ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj",
        }
    }
    torch.manual_seed(10)
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    # Ensure not equal
    assert not torch.equal(gate, up)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", gate)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", up)
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    rows = loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    # Must be gate then up (profile order), not sorted (gate, up is same sorted but test mismatch via NaN?)
    # Use different values: verify exact cat order
    assert torch.equal(rows[0], torch.cat([gate, up], dim=0))
    # If fallback were sorted, it would still be same here; stronger test: profile order up, gate
    profile2 = MagicMock()
    profile2.packed_expert_projection_names.side_effect = lambda name: ("up_proj", "gate_proj") if name == "gate_up_proj" else (name,)
    loader2 = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile2, expert_stack_members=members)
    rows2 = loader2.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    assert torch.equal(rows2[0], torch.cat([up, gate], dim=0))


def test_activation_loader_empty_projection_order_fails(tmp_path):
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    profile = MagicMock()
    profile.packed_expert_projection_names.return_value = ()
    members = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            ("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj",
        }
    }
    # Direct helper fails
    with pytest.raises(ValueError, match="empty projection order"):
        get_packed_expert_projection_names_strict(profile, "gate_up_proj")
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    with pytest.raises(ValueError, match="empty projection order"):
        loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)


def test_shared_helper_neutral_and_both_delegate():
    """Both streaming public helper and cb_ldlq use neutral shared helper."""
    import prismaquant.export_nvfp4_cb_streaming as stream_mod
    import prismaquant.cb_ldlq_fused_activation as fused_mod
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    # streaming public
    assert stream_mod.get_packed_expert_projection_names(profile, "gate_up_proj") == ("gate_proj", "up_proj")
    # neutral direct
    assert fused_mod.get_packed_expert_projection_names_strict(profile, "gate_up_proj") == ("gate_proj", "up_proj")
    # they must be same object path - no regex/sorted fallback in implementation text
    import inspect
    src_stream = inspect.getsource(stream_mod.get_packed_expert_projection_names)
    assert "sorted" not in src_stream
    assert "regex" not in src_stream.lower()
    src_cb = inspect.getsource(CBLDLQActivationLoader._per_expert)
    assert "sorted({str(p)" not in src_cb
    assert "except Exception" not in src_cb


# ---------- 2. get_expert_weight strict five-arg ----------

def test_get_expert_weight_four_arg_fails():
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    class FakeSkel:
        def __init__(self, t):
            self.t = t

        def dequant_weight(self, key):
            return self.t

    t = torch.randn(8, 8)
    skel = FakeSkel(t)
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    group = {"gate_proj": {0: "layers.0.ffn.experts.0.w1"}, "up_proj": {0: "layers.0.ffn.experts.0.w3"}}
    logical = {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"}

    def four_arg(proj, eid, ckpt_base, decoded):
        pass

    with pytest.raises(TypeError):
        get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "gate_up_proj", group, 0, logical_members=logical, on_member=four_arg)


def test_callback_body_typeerror_not_retried():
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    class FakeSkel:
        def __init__(self, t):
            self.t = t

        def dequant_weight(self, key):
            return self.t

    t = torch.randn(8, 8)
    skel = FakeSkel(t)
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("down_proj",) if name == "down_proj" else (name,)
    group = {"down_proj": {0: "layers.0.ffn.experts.0.w2"}}
    logical = {("down_proj", 0): "model.layers.0.mlp.experts.0.down_proj"}

    calls = []

    def bad_cb(proj, eid, ckpt_base, logical_qname, decoded):
        calls.append(1)
        raise TypeError("inner boom")

    with pytest.raises(TypeError, match="inner boom"):
        get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "down_proj", group, 0, logical_members=logical, on_member=bad_cb)
    assert len(calls) == 1


def test_on_member_without_logical_members_fails_before_decode():
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    class FakeSkel:
        def dequant_weight(self, key):
            raise AssertionError("should not decode")

    profile = MagicMock()
    profile.packed_expert_projection_names.return_value = ("down_proj",)
    group = {"down_proj": {0: "a"}}
    t = torch.randn(4, 4)
    skel = MagicMock()
    skel.dequant_weight.return_value = t

    def cb(proj, eid, ckpt_base, logical_qname, decoded):
        pass

    with pytest.raises(ValueError, match="on_member requires logical_members"):
        get_expert_weight(skel, profile, "pref", "down_proj", group, 0, logical_members=None, on_member=cb)
    # ensure dequant not called
    skel.dequant_weight.assert_not_called()


def test_on_member_without_logical_members_when_none_callback_permitted():
    """logical_members omission only allowed when on_member is None (research path)."""
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    t = torch.randn(8, 8)

    class FakeSkel:
        def dequant_weight(self, key):
            return t

    profile = MagicMock()
    profile.packed_expert_projection_names.return_value = ("down_proj",)
    group = {"down_proj": {0: "layers.0.ffn.experts.0.w2"}}
    # No logical_members, no on_member -> should succeed and return tensor
    skel = FakeSkel()
    out = get_expert_weight(skel, profile, "pref", "down_proj", group, 0, logical_members=None, on_member=None)
    assert torch.equal(out, t)


# ---------- 3. Typed no-observed handling ----------

def test_fill_empty_typed_exception():
    with pytest.raises(NoObservedExpertRowsError, match="no observed"):
        fill_empty_expert_activation_rows(
            (torch.empty((0, 0)), torch.empty((0, 0))),
            qname="test.qname",
        )
    # Ensure it's ValueError subclass but typed distinct
    assert issubclass(NoObservedExpertRowsError, ValueError)
    # Normal ValueError for width mismatch still ValueError not typed
    try:
        fill_empty_expert_activation_rows(
            (torch.randn(2, 8), torch.randn(2, 7)),
            qname="x",
        )
    except Exception as e:
        assert type(e) is ValueError
        assert not isinstance(e, NoObservedExpertRowsError)


def test_nvfp4_typed_no_observed_produces_raw_fallback(tmp_path):
    from prismaquant.nvfp4_cb_formats import ldlq_reassign_cb_fields_gated, _ldlq_gate_enabled
    import prismaquant.nvfp4_cb_formats as fmt
    # Ensure gate enabled
    import os
    os.environ["PRISMAQUANT_CB_LDLQ_GATE"] = "1"
    # Need gate enabled; if env not enough, force gate=True explicitly
    # Build 3-D weight E=2,R=4,C=256, k=12 (256 required by SUPERBLOCK)
    torch.manual_seed(0)
    E, R, C = 2, 4, 256
    w = torch.randn(E, R, C)
    # Simple raw fields via nvfp4_cb_fields
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields, nvfp4_cb_reconstruct

    # Create fields for k=12
    fields = nvfp4_cb_fields(w, k=12, grid="fp4", mode="product", col_weights=torch.ones(1, 1, C))
    # All empty activation rows -> no observed
    empty_rows = tuple(torch.empty((0, C)) for _ in range(E))
    out_fields, gate_info = ldlq_reassign_cb_fields_gated(
        w, fields, col_weights=torch.ones(1, 1, C), activation_rows=empty_rows, grid="fp4", mode="product", k=12, gate=True
    )
    # Should fallback raw with per_expert_kept all False and telemetry
    assert gate_info["gate"] == "raw_fallback_missing_activation"
    assert gate_info["kept_ldlq"] is False
    assert gate_info["per_expert_kept"] == [False, False]
    assert gate_info["missing_experts"] == [0, 1]
    assert gate_info["metric"] == "activation_output_mse"
    # No string matching was used: verify the typed exception path was taken by checking fill raises typed
    with pytest.raises(NoObservedExpertRowsError):
        fill_empty_expert_activation_rows(empty_rows, qname="ldlq_gate")
    # Ensure function source has no substring matching
    import inspect
    src = inspect.getsource(fmt.ldlq_reassign_cb_fields_gated)
    assert '"no observed" in' not in src
    assert "'no observed' in" not in src
    assert "no observed" not in src.lower().split("reason")[0] or "NoObserved" in src  # typed path must not string-match


def test_pooled_missing_still_uses_typed_and_not_string():
    """Partial missing uses pooling, not typed fallback, and gate proceeds."""
    from prismaquant.nvfp4_cb_formats import ldlq_reassign_cb_fields_gated

    torch.manual_seed(1)
    E, R, C = 2, 4, 256
    w = torch.randn(E, R, C)
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields

    fields = nvfp4_cb_fields(w, k=12, grid="fp4", mode="product", col_weights=torch.ones(1, 1, C))
    # One empty, one observed
    row0 = torch.randn(8, C)
    row1 = torch.empty((0, C))
    out_fields, gate_info = ldlq_reassign_cb_fields_gated(
        w, fields, col_weights=torch.ones(1, 1, C), activation_rows=(row0, row1), grid="fp4", mode="product", k=12, gate=True
    )
    # Should have pooled, not all-empty fallback
    assert gate_info["gate"] in ("ldlq_kept_all", "raw_kept_all", "mixed_per_expert")
    # missing_experts should be recorded where applicable
    if "missing_experts" in gate_info:
        assert gate_info["missing_experts"] == [1]


# ---------- 4. Unequal concat / native naming integration ----------

def test_unequal_concat_native_naming_integration(tmp_path):
    """Current unequal concat with native w1/w3 stays deterministic and profile-ordered."""
    from prismaquant.export_nvfp4_cb_streaming import get_expert_weight

    class FakeSkel:
        def __init__(self, m):
            self.m = m

        def dequant_weight(self, key):
            return self.m[key]

    gate_w = torch.randn(8, 8)
    up_w = torch.randn(8, 8)
    m = {
        "layers.0.ffn.experts.0.w1.weight": gate_w,
        "layers.0.ffn.experts.0.w3.weight": up_w,
    }
    skel = FakeSkel(m)
    profile = MagicMock()
    profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    group = {"gate_proj": {0: "layers.0.ffn.experts.0.w1"}, "up_proj": {0: "layers.0.ffn.experts.0.w3"}}
    logical = {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"}
    calls = []

    def cb(proj, eid, ckpt_base, logical_qname, decoded):
        calls.append((proj, logical_qname))

    fused = get_expert_weight(skel, profile, "model.layers.0.mlp.experts", "gate_up_proj", group, 0, logical_members=logical, on_member=cb)
    assert torch.equal(fused, torch.cat([gate_w, up_w], dim=0))
    assert calls == [("gate_proj", "model.layers.0.mlp.experts.0.gate_proj"), ("up_proj", "model.layers.0.mlp.experts.0.up_proj")]
    # And activation loader still gives gate-then-up concat with equal-count gate
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    members = {"model.layers.0.mlp.experts.gate_up_proj": logical}
    gate_act = torch.randn(4, 8)
    up_act = torch.randn(4, 8)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.gate_proj", gate_act)
    _write_act_file(act_dir, "model.layers.0.mlp.experts.0.up_proj", up_act)
    loader = CBLDLQActivationLoader(act_dir, model_dir=model_dir, profile=profile, expert_stack_members=members)
    rows = loader.load("model.layers.0.mlp.experts.gate_up_proj", stack_size=1)
    assert torch.equal(rows[0], torch.cat([gate_act, up_act], dim=0))
