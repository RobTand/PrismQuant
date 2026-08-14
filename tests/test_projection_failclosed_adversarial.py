"""Adversarial fail-closed behavior tests using actual production functions.

Proves:
- nonzero-start contiguous IDs fail in load_packed_projection (activation tuple indexing and export topology are zero-based)
- missing/partial/extra checkpoint data fail closed via validated_projection_checkpoint
- old schema (v1) fails
- projection_checkpoint_identity requires positive expert_count
"""
import pickle
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from tools.derive_dual_basis_packed import (
    PROJECTION_RUNG_SCHEMA,
    projection_checkpoint_identity,
    validated_projection_checkpoint,
    _sha,
)

from prismaquant.cb_ldlq_fused_activation import concat_equal_member_samples as FUSED_ACTIVATION_POLICY_V1


def _make_valid_identity(member_order=("gate_proj", "up_proj"), expert_count=4):
    return projection_checkpoint_identity(
        0, "gate_up_proj", 12,
        by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
        context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64,
        activation_evidence={
            "act_root": "/tmp/act",
            "qname_prefix": "model.layers.0.mlp.experts.gate_up_proj",
            "row_counts": [4]*expert_count,
            "evidence_sha256": "e"*64,
            "gate": "activation_output_mse",
            "fused_activation_policy": FUSED_ACTIVATION_POLICY_V1,
            "fused_activation_order": list(member_order),
            "member_order": list(member_order),
            "col_weight_pooling": "mean_of_member_vectors",
        },
        member_order=list(member_order), slice_boundaries={p: [0, 8] if i == 0 else [8, 16] for i, p in enumerate(member_order)}, col_weight_pooling="mean_of_member_vectors",
        expert_count=expert_count,
    )


def _make_valid_result(member_order=("gate_proj", "up_proj"), expert_count=4):
    per_leaf = {}
    qnames_per_leaf = {}
    for proj in member_order:
        per_leaf[proj] = {
            "weight_mse_per_expert": [0.1]*expert_count,
            "weighted_mse_per_expert": [0.1]*expert_count,
            "output_mse_per_expert": [0.2]*expert_count,
            "rel_output_mse_per_expert": [0.01]*expert_count,
        }
        qnames_per_leaf[proj] = [f"model.layers.0.mlp.experts.{i}.{proj}" for i in range(expert_count)]
    flat_qnames = []
    for proj in member_order:
        flat_qnames.extend(qnames_per_leaf[proj])
    return {
        "schema": "prismaquant.dsv4_nvfp4_projection_rung.v1",
        "layer": 0,
        "projection": "gate_up_proj",
        "packed_parent": "gate_up_proj",
        "expert_count": expert_count,
        "member_order": list(member_order),
        "slice_boundaries": {p: [0, 8] if i == 0 else [8, 16] for i, p in enumerate(member_order)},
        "col_weight_pooling": "mean_of_member_vectors",
        "format": "NVFP4_CB_K12",
        "rung": 12,
        "qnames": flat_qnames,
        "qnames_per_leaf": qnames_per_leaf,
        "weight_mse_per_expert": [0.1]*expert_count,
        "weighted_mse_per_expert": [0.1]*expert_count,
        "output_mse_per_expert": [0.2]*expert_count,
        "rel_output_mse_per_expert": [0.01]*expert_count,
        "per_leaf": per_leaf,
        "weight_mse_fused_per_expert": [0.1]*expert_count,
        "output_mse_fused_per_expert": [0.2]*expert_count,
        "n_activation_rows_per_expert": [4]*expert_count,
        "gate_info": {
            "gate": "ldlq_kept_all",
            "kept_ldlq": True,
            "per_expert_kept": [True]*expert_count,
            "raw_mse_per_expert": [0.2]*expert_count,
            "ldlq_mse_per_expert": [0.1]*expert_count,
            "metric": "activation_output_mse",
        },
        "warm_state_path": None,
    }


def _write_payload(tmp_path, identity, result, schema=PROJECTION_RUNG_SCHEMA):
    payload = {
        "schema": schema,
        "content_key": _sha(identity),
        "identity": dict(identity),
        "result": dict(result),
    }
    p = tmp_path / "ckpt.pkl"
    p.write_bytes(pickle.dumps(payload))
    return p


# ---------------------------------------------------------------------------
# 1. nonzero-start IDs must fail in load_packed_projection
# ---------------------------------------------------------------------------

def test_load_packed_projection_nonzero_start_ids_fail(tmp_path, monkeypatch):
    import tools.derive_dual_basis_packed as mod

    E = 4
    # Mock planning to return nonzero-start IDs: 1..E instead of 0..E-1
    mock_profile = MagicMock()
    mock_profile.packed_expert_param_names.return_value = frozenset({"gate_up_proj"})
    mock_profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    # expert_groups with nonzero-start ids
    mock_expert_groups = {
        "model.layers.0.mlp.experts": {
            "gate_proj": {i: f"model.layers.0.mlp.experts.{i}.gate_proj" for i in range(1, E+1)},
            "up_proj": {i: f"model.layers.0.mlp.experts.{i}.up_proj" for i in range(1, E+1)},
        }
    }
    mock_members = {
        "model.layers.0.mlp.experts.gate_up_proj": {(p, e): f"model.layers.0.mlp.experts.{e}.{p}" for p in ("gate_proj", "up_proj") for e in range(1, E+1)},
    }
    monkeypatch.setattr(mod, "_get_packed_planning", lambda: (mock_profile, mock_expert_groups, mock_members, frozenset({"gate_up_proj"})))
    identity = {"col_weights_shapes": {}, "col_weights_content_sha256": {}}
    all_cw = {}
    # col weights for 1..E
    for e in range(1, E+1):
        for proj in ("gate_proj", "up_proj"):
            qn = f"model.layers.0.mlp.experts.{e}.{proj}"
            cw = torch.ones(8)
            all_cw[qn] = cw
            identity["col_weights_shapes"][qn] = list(cw.shape)
            identity["col_weights_content_sha256"][qn] = mod.content_sha256_float32(cw)
    with pytest.raises(AssertionError, match="must be exactly list\\(range"):
        mod.load_packed_projection(
            0, "gate_up_proj", device=torch.device("cpu"), identity=identity,
            all_col_weights=all_cw, model_to_shard={}, model_to_ckpt={}, scale_map={},
        )


def test_load_packed_projection_zero_based_passes(tmp_path, monkeypatch):
    import tools.derive_dual_basis_packed as mod
    E = 2
    mock_profile = MagicMock()
    mock_profile.packed_expert_param_names.return_value = frozenset({"gate_up_proj"})
    mock_profile.packed_expert_projection_names.side_effect = lambda name: ("gate_proj", "up_proj") if name == "gate_up_proj" else (name,)
    mock_expert_groups = {
        "model.layers.0.mlp.experts": {
            "gate_proj": {i: f"model.layers.0.mlp.experts.{i}.gate_proj" for i in range(E)},
            "up_proj": {i: f"model.layers.0.mlp.experts.{i}.up_proj" for i in range(E)},
        }
    }
    mock_members = {
        "model.layers.0.mlp.experts.gate_up_proj": {(p, e): f"model.layers.0.mlp.experts.{e}.{p}" for p in ("gate_proj", "up_proj") for e in range(E)},
    }
    monkeypatch.setattr(mod, "_get_packed_planning", lambda: (mock_profile, mock_expert_groups, mock_members, frozenset({"gate_up_proj"})))
    torch.manual_seed(0)
    loader_acts = tuple(torch.randn(4, 8) for _ in range(E))
    mock_loader = MagicMock()
    mock_loader.load.return_value = loader_acts
    monkeypatch.setattr("prismaquant.cb_ldlq.CBLDLQActivationLoader", lambda *a, **k: mock_loader)
    mock_skel = MagicMock()
    mock_skel.dequant_weight.side_effect = lambda k: torch.randn(8, 8)
    monkeypatch.setattr("prismaquant.export_nvfp4_cb_streaming.open_packed_weight_source", lambda model_dir: mock_skel)
    monkeypatch.setattr(mod, "open_packed_weight_source", lambda model_dir: mock_skel)
    monkeypatch.setattr("prismaquant.production_weight_cache.validate_cb_render_source_weight", lambda *a, **k: None)
    all_cw = {}
    identity = {"col_weights_shapes": {}, "col_weights_content_sha256": {}}
    for e in range(E):
        cw_vec = loader_acts[e].square().mean(0)
        for proj in ("gate_proj", "up_proj"):
            qn = f"model.layers.0.mlp.experts.{e}.{proj}"
            all_cw[qn] = cw_vec.clone()
            identity["col_weights_shapes"][qn] = list(cw_vec.shape)
            identity["col_weights_content_sha256"][qn] = mod.content_sha256_float32(cw_vec)
    data = mod.load_packed_projection(0, "gate_up_proj", device=torch.device("cpu"), identity=identity, all_col_weights=all_cw, model_to_shard={}, model_to_ckpt={}, scale_map={})
    assert data["weight"].shape[0] == E
    assert data["member_order"] == ["gate_proj", "up_proj"]


# ---------------------------------------------------------------------------
# 2. old schema must fail
# ---------------------------------------------------------------------------

def test_validated_old_schema_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    p = _write_payload(tmp_path, ident, result, schema="prismaquant.dsv4_dual_basis_projection_rung.v1")
    with pytest.raises(AssertionError, match="schema mismatch"):
        validated_projection_checkpoint(p, ident)


# ---------------------------------------------------------------------------
# 3. missing/partial/extra checkpoint data fail closed
# ---------------------------------------------------------------------------

def test_validated_missing_per_leaf_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result.pop("per_leaf")
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="per_leaf missing or not mapping"):
        validated_projection_checkpoint(p, ident)


def test_validated_missing_qnames_per_leaf_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result.pop("qnames_per_leaf")
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="qnames_per_leaf missing or not mapping"):
        validated_projection_checkpoint(p, ident)


def test_validated_per_leaf_extra_key_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["per_leaf"]["extra_proj"] = {"weight_mse_per_expert": [0.1]*4, "weighted_mse_per_expert": [0.1]*4, "output_mse_per_expert": [0.2]*4, "rel_output_mse_per_expert": [0.01]*4}
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="per_leaf keys"):
        validated_projection_checkpoint(p, ident)


def test_validated_qnames_per_leaf_extra_key_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["qnames_per_leaf"]["extra_proj"] = ["a"]*4
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="qnames_per_leaf keys"):
        validated_projection_checkpoint(p, ident)


def test_validated_per_leaf_missing_metric_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    del result["per_leaf"]["gate_proj"]["output_mse_per_expert"]
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="per_leaf gate_proj missing required output"):
        validated_projection_checkpoint(p, ident)


def test_validated_per_leaf_not_list_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["per_leaf"]["gate_proj"]["weight_mse_per_expert"] = (0.1, 0.2, 0.3, 0.4)  # tuple not list
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="not list"):
        validated_projection_checkpoint(p, ident)


def test_validated_per_leaf_wrong_length_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["per_leaf"]["gate_proj"]["weight_mse_per_expert"] = [0.1]*3
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="length.*!= expert_count"):
        validated_projection_checkpoint(p, ident)


def test_validated_qnames_wrong_length_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["qnames_per_leaf"]["gate_proj"] = ["a"]*3
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="qnames_per_leaf gate_proj length"):
        validated_projection_checkpoint(p, ident)


def test_validated_qnames_total_mismatch_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    # keep per-leaf correct length but flat qnames wrong total
    result["qnames"] = result["qnames"][:-1]  # 7 instead of 8
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="qnames length"):
        validated_projection_checkpoint(p, ident)


def test_validated_result_missing_expert_count_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    del result["expert_count"]
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="result expert_count"):
        validated_projection_checkpoint(p, ident)


def test_validated_result_wrong_length_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["weight_mse_per_expert"] = [0.1]*3
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="weight_mse_per_expert length"):
        validated_projection_checkpoint(p, ident)


def test_validated_gate_vectors_wrong_length_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result["gate_info"]["per_expert_kept"] = [True]*3
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="gate_info per_expert_kept length"):
        validated_projection_checkpoint(p, ident)


def test_validated_missing_gate_info_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    result.pop("gate_info")
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="gate_info missing"):
        validated_projection_checkpoint(p, ident)


def test_validated_per_leaf_partial_missing_fails(tmp_path):
    ident = _make_valid_identity(expert_count=4)
    result = _make_valid_result(expert_count=4)
    # only one leaf present, missing up_proj
    result["per_leaf"] = {"gate_proj": result["per_leaf"]["gate_proj"]}
    result["qnames_per_leaf"] = {"gate_proj": result["qnames_per_leaf"]["gate_proj"]}
    # need to also adjust member_order mismatch: per_leaf keys != member_order triggers
    p = _write_payload(tmp_path, ident, result)
    with pytest.raises(AssertionError, match="per_leaf keys"):
        validated_projection_checkpoint(p, ident)


# ---------------------------------------------------------------------------
# 4. projection_checkpoint_identity requires positive expert_count
# ---------------------------------------------------------------------------

def test_projection_checkpoint_identity_requires_expert_count():
    with pytest.raises(TypeError):
        projection_checkpoint_identity(
            0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
            context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64,
        )
    with pytest.raises(ValueError, match="positive int"):
        projection_checkpoint_identity(
            0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
            context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, expert_count=0,
        )
    with pytest.raises(ValueError, match="positive int"):
        projection_checkpoint_identity(
            0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
            context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, expert_count=-1,
        )
    with pytest.raises(ValueError, match="positive int"):
        projection_checkpoint_identity(
            0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
            context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, expert_count=None,
        )
    ident = projection_checkpoint_identity(
        0, "gate_up_proj", 12, by_layer_sha256="a"*64, col_weights_sha256="b"*64, source_index_sha256="c"*64,
        context_stamp={"ldlq_scope": "nvfp4"}, tool_sha256="d"*64, expert_count=4,
    )
    assert ident["expert_count"] == 4
    assert ident["schema"] == PROJECTION_RUNG_SCHEMA
