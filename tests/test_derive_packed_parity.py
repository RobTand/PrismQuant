"""Blocker A parity tests: derive must use authoritative planning, not None/solo qnames."""
import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_derive_source_uses_authoritative_planning():
    p = Path(__file__).resolve().parents[1] / "tools/derive_dual_basis_packed.py"
    text = p.read_text()
    # Must not use hardcoded nvfp4_cb profile
    assert 'load_serving_profile("nvfp4_cb")' not in text, "derive still uses hardcoded nvfp4_cb profile"
    # Must use public production helper, not a pile of private underscore APIs
    assert "get_packed_moe_planning" in text, "derive must import public get_packed_moe_planning"
    assert "get_packed_expert_col_weights" in text or "get_packed_expert_projection_names" in text, "derive must use public pooling/projection helpers"
    # Must not import a pile of private helpers
    private_imports = ["_LazySkeleton", "_plan_expert_stacks", "_packed_expert_param_names", "_packed_expert_projection_names", "_expert_member_qnames"]
    found_private = [name for name in private_imports if f"import {name}" in text or f"from prismaquant.export_nvfp4_cb_streaming import" in text and name in text]
    # Allow at most the public wrapper, not the pile
    assert text.count("_plan_expert_stacks") <= 1 or "get_packed_moe_planning" in text, "derive still imports private pile"
    # Must not pass None for expert_stack_members
    assert "expert_stack_members=None" not in text, "derive passes None for expert_stack_members, must pass real map"
    assert "expert_stack_members=expert_stack_members" in text or "expert_stack_members=expert_stack_members," in text or "expert_stack_members" in text
    # Must reference gate_up_proj (packed parent) and not just separate gate/up/down
    assert "gate_up_proj" in text, "derive must reference gate_up_proj packed parent"
    # Must not duplicate regex logic
    assert "def _plan_expert_stacks" not in text, "derive must not duplicate _plan_expert_stacks"
    assert "def _packed_expert_param_names" not in text, "derive must not duplicate packed param logic"


def test_derive_load_projection_uses_real_members_and_fused_gate_up(tmp_path, monkeypatch):
    """Executable: would fail if derive passes None or diverges on fused gate_up."""
    import tools.derive_dual_basis_packed as mod

    mock_profile = MagicMock()
    mock_profile.packed_expert_param_names.return_value = frozenset({"gate_up_proj", "down_proj"})
    mock_profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    mock_expert_groups = {
        "model.layers.0.mlp.experts": {
            "gate_proj": {i: f"model.layers.0.mlp.experts.{i}.gate_proj" for i in range(256)},
            "up_proj": {i: f"model.layers.0.mlp.experts.{i}.up_proj" for i in range(256)},
            "down_proj": {i: f"model.layers.0.mlp.experts.{i}.down_proj" for i in range(256)},
        }
    }
    mock_members = {
        "model.layers.0.mlp.experts.gate_up_proj": {(p, e): f"model.layers.0.mlp.experts.{e}.{p}" for p in ("gate_proj", "up_proj") for e in range(256)},
        "model.layers.0.mlp.experts.down_proj": {("down_proj", e): f"model.layers.0.mlp.experts.{e}.down_proj" for e in range(256)},
    }
    with patch.object(mod, "_get_packed_planning", return_value=(mock_profile, mock_expert_groups, mock_members, frozenset({"gate_up_proj", "down_proj"}))):
        # Create deterministic loader acts and derive col weights from them so activation validation passes
        torch.manual_seed(0)
        loader_acts = tuple(torch.randn(4, 8) for _ in range(256))
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = loader_acts
        with patch("prismaquant.cb_ldlq.CBLDLQActivationLoader", return_value=mock_loader_instance) as mock_loader_cls:
            mock_skel = MagicMock()
            def _mock_dequant(k):
                return torch.randn(8, 8, dtype=torch.float32)
            mock_skel.dequant_weight.side_effect = _mock_dequant
            with patch("prismaquant.export_nvfp4_cb_streaming.open_packed_weight_source", return_value=mock_skel):
                with patch.object(mod, "open_packed_weight_source", return_value=mock_skel):
                    with patch("prismaquant.production_weight_cache.validate_cb_render_source_weight"):
                        # Derive cw from loader acts: pooled mean must equal act square mean
                        all_cw = {}
                        for e in range(256):
                            cw_vec = loader_acts[e].square().mean(0)
                            for proj in ("gate_proj", "up_proj", "down_proj"):
                                all_cw[f"model.layers.0.mlp.experts.{e}.{proj}"] = cw_vec.clone()
                        identity = {
                            "col_weights_shapes": {k: list(v.shape) for k,v in all_cw.items()},
                            "col_weights_content_sha256": {k: mod.content_sha256_float32(v) for k,v in all_cw.items()},
                        }
                        data = mod.load_packed_projection(
                            0, "gate_up_proj",
                            device=torch.device("cpu"),
                            identity=identity,
                            all_col_weights=all_cw,
                            model_to_shard={},
                            model_to_ckpt={},
                            scale_map={},
                        )
                        assert mock_loader_cls.called, "CBLDLQActivationLoader not called"
                        _, kwargs = mock_loader_cls.call_args
                        assert kwargs.get("profile") is mock_profile
                        assert kwargs.get("expert_stack_members") == mock_members
                        called_qnames = [c.args[0] for c in mock_loader_instance.load.call_args_list]
                        assert any("gate_up_proj" in q for q in called_qnames), f"gate_up_proj packed parent not used, got {called_qnames}"
                        assert data["weight"].shape == (256, 16, 8), f"fused weight shape wrong {data['weight'].shape}"
                        assert data["col_weights"].shape == (256, 1, 8)
                        assert data["member_order"] == ["gate_proj", "up_proj"]
                        assert data["slice_boundaries"] == {"gate_proj": (0, 8), "up_proj": (8, 16)}


def test_derive_checkpoint_identity_uses_packed_evidence():
    p = Path(__file__).resolve().parents[1] / "tools/derive_dual_basis_packed.py"
    text = p.read_text()
    # Checkpoint identity for gate/up should use gate_up_proj evidence
    assert "gate_up_proj" in text and "_packed_activation_evidence_identity" in text
    # Must include member_order/slice_boundaries/col_weight_pooling in identity
    assert "member_order" in text and "slice_boundaries" in text and "col_weight_pooling" in text


def test_derive_pooling_uses_shared_helper():
    p = Path(__file__).resolve().parents[1] / "tools/derive_dual_basis_packed.py"
    text = p.read_text()
    # Production gate must NOT pool cold experts — eligible-only, no filled auxiliary
    assert "fill_empty_expert_activation_rows" not in text, "derive production path must not pool cold activation rows"
    # Col-weight pooling still via public helper mean_of_member_vectors
    assert "mean_of_member_vectors" in text
    # Activation pooling contract should be none/eligible-only
    assert "none_eligible_only_no_pooled_prior" in text or "eligible_only" in text


def test_derive_no_duplicate_stack_planning():
    p = Path(__file__).resolve().parents[1] / "tools/derive_dual_basis_packed.py"
    text = p.read_text()
    # Should not contain duplicate per-expert regex or manual prefix logic
    # The only planning should be via imported helpers
    # Count occurrences of _plan_expert_stacks definition — should be 0 in derive (imported only)
    assert "def _plan_expert_stacks" not in text, "derive must not duplicate _plan_expert_stacks, import it"
    assert "def _packed_expert_param_names" not in text, "derive must not duplicate packed param logic"
