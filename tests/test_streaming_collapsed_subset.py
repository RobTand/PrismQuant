"""Validate streaming collapsed member-map permits subset but requires exact equality per emitted target."""
import pytest


def _validate_subset(pre_members, collapsed):
    """Replicate the validation logic in export_nvfp4_cb_streaming."""
    for qname, members in collapsed.items():
        expected = pre_members.get(qname)
        if expected is None:
            raise AssertionError(f"{qname}: collapsed stack not in authoritative planning {sorted(pre_members)}")
        if expected != members:
            raise AssertionError(f"{qname}: member map drift vs authoritative planning (expected {expected} got {members})")


def test_collapsed_valid_subset_permits():
    pre = {
        "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"},
        "model.layers.1.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.1.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.1.mlp.experts.0.up_proj"},
        "model.layers.0.mlp.experts.down_proj": {("down_proj", 0): "model.layers.0.mlp.experts.0.down_proj"},
    }
    collapsed_subset = {
        "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"},
    }
    # Should not raise — subset is allowed
    _validate_subset(pre, collapsed_subset)


def test_collapsed_mismatch_fails():
    pre = {
        "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj"},
    }
    # Drift: one member qname altered
    collapsed_drift = {
        "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("up_proj", 0): "WRONG.qname"},
    }
    with pytest.raises(AssertionError, match="member map drift"):
        _validate_subset(pre, collapsed_drift)


def test_collapsed_extra_stack_not_in_planning_fails():
    pre = {
        "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj"},
    }
    collapsed_extra = {
        "model.layers.1.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.1.mlp.experts.0.gate_proj"},
    }
    with pytest.raises(AssertionError, match="not in authoritative"):
        _validate_subset(pre, collapsed_extra)
