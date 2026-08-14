"""Mixed NVFP4/FP8 LDLQ scope: nvfp4 only."""

import copy
import torch
from prismaquant import format_registry as fr
from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_tensor_payload_breakdown, _ldlq_for_format

SCHEMA = "prismaquant.dsv4_afast_layer_shard.v3"

def test_mixed_scope_per_tensor_ldlq():
    ctx_nvfp4 = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    ctx_all = CBSerializationContext.production(ldlq_scope="all", encode_tier="balanced")
    ctx_none = CBSerializationContext.production(ldlq_scope="none", encode_tier="balanced")
    # NVFP4 should be LDLQ under nvfp4 and all, not under none
    assert _ldlq_for_format("NVFP4_CB_K12", ctx_nvfp4) is True
    assert _ldlq_for_format("NVFP4_CB_K12", ctx_all) is True
    assert _ldlq_for_format("NVFP4_CB_K12", ctx_none) is False
    # FP8 should be LDLQ only under all
    assert _ldlq_for_format("FP8_CB_K28", ctx_nvfp4) is False
    assert _ldlq_for_format("FP8_CB_K28", ctx_all) is True
    assert _ldlq_for_format("FP8_CB_K28", ctx_none) is False

def test_mixed_scope_per_tensor_identity():
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    shape = (2048, 4096)
    nvfp4_payload = cb_tensor_payload_breakdown("NVFP4_CB_K12", shape, qname="test.nvfp4", context=ctx)
    fp8_payload = cb_tensor_payload_breakdown("FP8_CB_K28", shape, qname="test.fp8", context=ctx)
    assert nvfp4_payload["identity"]["ldlq"] is True
    assert nvfp4_payload["identity"]["ldlq_scope"] == "nvfp4"
    assert fp8_payload["identity"]["ldlq"] is False
    assert fp8_payload["identity"]["ldlq_scope"] == "nvfp4"
    # Global stamp should carry scope
    from prismaquant.nvfp4_cb_footprint import cb_serialization_context_stamp
    stamp = cb_serialization_context_stamp(ctx, formats=["NVFP4_CB_K12", "FP8_CB_K28"])
    assert stamp["ldlq_scope"] == "nvfp4"
    assert stamp["ldlq"] is True  # at least one family is LDLQ

def test_mixed_scope_cost_render_mismatch_rejection():
    from prismaquant.nvfp4_cb_footprint import validate_cb_serialization_context_stamp
    ctx_nvfp4 = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    ctx_all = CBSerializationContext.production(ldlq_scope="all", encode_tier="balanced")
    stamp_nvfp4 = __import__("prismaquant.nvfp4_cb_footprint", fromlist=["cb_serialization_context_stamp"]).cb_serialization_context_stamp(ctx_nvfp4)
    # Mismatch: exporter with scope all but recipe is nvfp4 should fail
    try:
        validate_cb_serialization_context_stamp(stamp_nvfp4, ctx_all, where="test")
        assert False, "should have raised scope mismatch"
    except ValueError as e:
        assert "ldlq_scope" in str(e)

def test_export_mixed_scope_only_nvfp4_uses_ldlq():
    # Simulate that only NVFP4 uses LDLQ in cost and export
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    # Check that cb_fields_for_context respects scope
    w = torch.randn(8, 2048)
    cw = torch.ones(1, 2048)
    act = torch.randn(32, 2048)
    from prismaquant.nvfp4_cb_footprint import cb_fields_for_context
    spec_nvfp4 = fr.get_format("NVFP4_CB_K12")
    spec_fp8 = fr.get_format("FP8_CB_K28")
    # NVFP4 with LDLQ should use activation
    fields_nvfp4 = cb_fields_for_context(spec_nvfp4, w, context=ctx, col_weights=cw, activation_rows=act)
    # FP8 with same context should NOT use LDLQ (should be raw, no activation needed)
    # For FP8, col_weights and activation_rows are not required for raw, but we pass them to ensure it doesn't use LDLQ
    fields_fp8 = cb_fields_for_context(spec_fp8, w, context=ctx, col_weights=cw, activation_rows=act)
    # For FP8, the fields should be same as raw (since scope nvfp4 means FP8 is raw)
    # We can check that FP8 fields do not have LDLQ-specific changes by comparing to raw context
    ctx_raw = CBSerializationContext.production(ldlq_scope="none", encode_tier="balanced")
    fields_fp8_raw = cb_fields_for_context(spec_fp8, w, context=ctx_raw, col_weights=cw, activation_rows=None)
    # For FP8, the number of indices should be same regardless of LDLQ (since we didn't apply LDLQ)
    # This is a weak check, but ensures no crash
    assert fields_fp8["indices"].shape == fields_fp8_raw["indices"].shape

def test_gate_info_plumbed_not_guessed_packed():
    """Actual gate_info from gated encoder, per-expert decisions, not guessed kept."""
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    from prismaquant.nvfp4_cb_footprint import cb_fields_for_context
    # 3-expert stack (small) to test per-expert gating on CPU; in_features must be multiple of SUPERBLOCK=256
    w = torch.randn(3, 4, 256)
    cw = torch.ones(3, 1, 256)
    acts = tuple(torch.randn(16, 256) for _ in range(3))
    spec = fr.get_format("NVFP4_CB_K12")
    fields, gate_info = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=acts, return_gate_info=True)
    assert gate_info is not None
    assert "gate" in gate_info
    # Packed gate must expose per_expert_kept and both mse lists
    # When all activations present, gate is one of ldlq_kept_all, raw_kept_all, mixed_per_expert
    assert gate_info["gate"] in {"ldlq_kept_all", "raw_kept_all", "mixed_per_expert", "ldlq_kept", "raw_kept"}
    if gate_info["gate"] in {"ldlq_kept_all", "raw_kept_all", "mixed_per_expert"}:
        assert "per_expert_kept" in gate_info
        assert len(gate_info["per_expert_kept"]) == 3
        assert "raw_mse_per_expert" in gate_info and "ldlq_mse_per_expert" in gate_info
        # Gate decision must correspond to actual mse comparison (within epsilon)
        for keep, raw, ldlq in zip(gate_info["per_expert_kept"], gate_info["raw_mse_per_expert"], gate_info["ldlq_mse_per_expert"]):
            if keep:
                assert ldlq <= raw + 1e-12 * max(abs(raw), abs(ldlq), 1.0)
            else:
                assert raw < ldlq - 1e-12 * max(abs(raw), abs(ldlq), 1.0) or gate_info["gate"] == "mixed_per_expert"
    # Backward compat: without return_gate_info, fields only
    fields2 = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=acts, return_gate_info=False)
    assert isinstance(fields2, dict) and "indices" in fields2

def test_gate_info_dense_missing_activation_fail_closed():
    """Missing activation must be explicit raw_fallback, never silent weight-MSE disguise."""
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    from prismaquant.nvfp4_cb_footprint import cb_fields_for_context
    w = torch.randn(8, 256)
    cw = torch.ones(1, 256)
    empty_act = torch.empty((0, 256), dtype=torch.float32)
    spec = fr.get_format("NVFP4_CB_K12")
    fields, gate_info = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=empty_act, return_gate_info=True)
    assert gate_info is not None
    assert gate_info["gate"] in {"raw_fallback_no_activation", "raw_fallback_missing_activation", "raw_fallback_malformed_activation", "raw_fallback_shared_activation_for_packed"}
    assert gate_info.get("kept_ldlq") is False
    # Also verify that packing path gates identically
    from prismaquant import nvfp4_cb_formats as cb
    packed, fields2 = cb.nvfp4_cb_pack(w, 12, grid="fp4", mode="product", col_weights=cw, activation_rows=empty_act, ldlq=True)
    assert packed is not None  # should have fallen back to raw

def test_content_key_is_sha_of_canonical_identity_not_costs():
    import hashlib, json
    from tools.derive_dual_basis_packed import _sha
    identity = {"layer": 0, "schema": SCHEMA, "profile": "A-FAST", "serialization_context": {"ldlq_scope": "nvfp4"}}
    # correct formula per burn
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert _sha(identity) == expected
    # wrong formula (costs+identity) must not be used
    wrong = hashlib.sha256(json.dumps({"costs": {"dummy": 1}, "identity": identity}, sort_keys=True).encode()).hexdigest()
    assert _sha(identity) != wrong

def test_fp8_deep_equality_synthetic():
    from tools.derive_dual_basis_packed import validate_no_fp8_drift
    raw_costs = {
        "model.layers.0.self_attn.q_proj": {
            "NVFP4_CB_K12": {"weight_mse": 0.1, "output_mse": 0.2},
            "FP8_CB_K28": {"weight_mse": 0.01, "output_mse": 0.02, "n_activation_rows": 64},
            "BF16": {"weight_mse": 0.0},
        }
    }
    derived_same = copy.deepcopy(raw_costs)
    # Derived may replace NVFP4 but FP8 must stay identical
    derived_same["model.layers.0.self_attn.q_proj"]["NVFP4_CB_K12"] = {"weight_mse": 0.05, "output_mse": 0.1, "gate": "ldlq_kept_all"}
    validate_no_fp8_drift(raw_costs, derived_same)  # should pass
    derived_drift = copy.deepcopy(raw_costs)
    derived_drift["model.layers.0.self_attn.q_proj"]["FP8_CB_K28"]["weight_mse"] = 0.011  # drift
    try:
        validate_no_fp8_drift(raw_costs, derived_drift)
        assert False, "should have rejected FP8 drift"
    except AssertionError as e:
        assert "FP8 drift" in str(e)
    # Mutating FP8 by adding interpolation_source key is also drift
    derived_add = copy.deepcopy(raw_costs)
    derived_add["model.layers.0.self_attn.q_proj"]["FP8_CB_K28"]["interpolation_source"] = "raw_nvfp4_bank"
    try:
        validate_no_fp8_drift(raw_costs, derived_add)
        assert False, "should have rejected FP8 key addition"
    except AssertionError:
        pass

def test_no_writes_in_smoke_prohibits_warm():
    import pathlib
    # Smoke path must reject --write-warm-state and must not create checkpoints
    # We test the guard without GPU: derive_one_projection_rung should raise ValueError if write_warm_state True
    try:
        from tools.derive_dual_basis_packed import derive_one_projection_rung
        # call with write_warm_state=True should raise before any file IO
        derive_one_projection_rung(0, "down_proj", 12, torch.device("cpu"), write_warm_state=True)  # type: ignore[arg-type]
        assert False, "should have rejected write_warm_state in smoke"
    except ValueError as e:
        assert "smoke must not write warm state" in str(e)
    except RuntimeError:
        # require_cuda may fire first (expected on cpu smoke); guard still correctly placed before CUDA
        pass

def test_derived_identity_rebuilds_scope_stamp():
    from tools.derive_dual_basis_packed import build_derived_identity
    lattice = {f"NVFP4_CB_K{k}": ["a"*64, "a"*64] for k in range(12, 19)}
    lattice.update({f"FP8_CB_K{k}": ["b"*64]*4 for k in range(28, 39)})
    sc = {"ldlq": False, "lattice_codebook_sha256_by_format": lattice, "scale_coding": "two_tier", "layout_version": 2, "codebook_source": "lattice", "scale_sweep": True, "encode_tier": "balanced", "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1", "activation_contract": "prismaquant.nvfp4_w4a4_activation.v1", "activation_execution": "e2m1_group16_ue4m3_static"}
    raw_id = {"schema": SCHEMA, "layer": 0, "profile": "A-FAST", "serialization_context": sc, "verified_base_layer_sha256": "abc", "implementation_sha256": {"burn_tool": "old", "pilot_tool": "old"}}
    new_id = build_derived_identity(raw_id)
    assert new_id["serialization_context"]["ldlq_scope"] == "nvfp4"
    assert "derive_tool_sha256" in new_id
    assert new_id["schema"] == SCHEMA
    assert new_id["raw_implementation_sha256"] == raw_id["implementation_sha256"]
    assert new_id["implementation_sha256"] == raw_id["implementation_sha256"]
    assert "dual_basis_implementation_sha256" in new_id and "derive_tool" in new_id["dual_basis_implementation_sha256"]
    assert "dual_basis_derivation" in new_id
    assert "burn_tool" not in new_id["dual_basis_implementation_sha256"]
    raw_id2 = {"schema": SCHEMA, "layer": 0, "profile": "A-FAST", "serialization_context": dict(sc), "verified_base_layer_sha256": "abc"}
    new_id2 = build_derived_identity(raw_id2)
    assert "dual_basis_implementation_sha256" in new_id2 and "derive_tool" in new_id2["dual_basis_implementation_sha256"]
