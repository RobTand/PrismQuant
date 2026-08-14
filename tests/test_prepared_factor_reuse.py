"""Prepared evidence factor-reuse and mutation regressions — exact inverse_hessian_cholesky counters."""
import torch
from unittest.mock import patch, MagicMock

from prismaquant.nvfp4_cb_formats import (
    clear_ldlq_factor_cache,
    prepare_ldlq_gate_evidence,
    ldlq_reassign_cb_fields_gated,
    nvfp4_cb_fields,
    _SPLIT_POLICY,
    _SPLIT_VERSION,
)
from prismaquant import format_registry as fr
from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_fields_for_context


def test_warm_plus_cold_reaches_gate_not_prewarm_error():
    clear_ldlq_factor_cache()
    torch.manual_seed(0)
    E, R, C = 2, 8, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    warm_rows = torch.randn(16, C)
    cold_rows = torch.empty((0, C), dtype=torch.float32)
    acts = [warm_rows, cold_rows]
    fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    # Without prepared, cold forced raw (no factor for cold)
    out_fields, gate = ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12)
    assert gate["per_expert_kept"][1] is False
    # With prepared, still forced raw, and cold fit remains empty (no pooling)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.warm_cold")
    assert prepared.cold_experts == (1,)
    assert prepared.eligible_experts == (0,)
    assert prepared.fit_rows[1].numel() == 0, "cold fit should remain empty (no pooling)"
    out2, gate2 = ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12, prepared=prepared)
    assert gate2["per_expert_kept"][1] is False
    # Cold rows must be bit-identical to raw fields
    assert torch.equal(out2["indices"][R:], fields["indices"][R:])
    if "signs" in fields:
        assert torch.equal(out2["signs"][R:], fields["signs"][R:])
    assert gate["metric"] == "activation_output_mse"
    assert gate2["metric"] == "activation_output_mse"


def _count_hessian_calls_via_patch(w, cw, acts, prepared, rungs):
    """Helper to count inverse_hessian_cholesky calls exactly."""
    import prismaquant.rotation_ldlq_pilot as pilot
    cnt = {"n": 0}
    orig = pilot.inverse_hessian_cholesky

    def counting(*a, **kw):
        cnt["n"] += 1
        return orig(*a, **kw)

    with patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting):
        clear_ldlq_factor_cache()
        for rung in rungs:
            fields = nvfp4_cb_fields(w, rung, grid="fp4", mode="product", col_weights=cw)
            # For packed, pass acts; for dense pass act tensor
            if isinstance(acts, torch.Tensor):
                ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=rung, prepared=prepared)
            else:
                ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=rung, prepared=prepared)
        # capture count before clearing
        n = cnt["n"]
    clear_ldlq_factor_cache()
    return n


def test_packed_factor_exact_counts_via_hessian_patch():
    # Case 1: E warm experts across seven rungs -> E calls
    torch.manual_seed(1)
    E, R, C = 4, 8, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    acts = [torch.randn(12, C) for _ in range(E)]
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.packed_factor")
    # Verify stable object identity across rungs: fit tensors same ids
    fit_ids = [id(t) for t in prepared.fit_rows]
    assert len(set(fit_ids)) == E
    # Count via patch
    n = _count_hessian_calls_via_patch(w, cw, acts, prepared, range(12, 19))
    assert n == E, f"expected {E} hessian builds for {E} warm across 7 rungs, got {n}"
    # Verify eligible subset does not create new activation tensors per rung: after 7 rungs, fit tensor ids unchanged
    assert [id(t) for t in prepared.fit_rows] == fit_ids

    # Case 2: one warm + one cold -> 1 call
    torch.manual_seed(2)
    E2, R2, C2 = 2, 8, 256
    w2 = torch.randn(E2, R2, C2)
    cw2 = torch.ones(E2, 1, C2)
    acts2 = [torch.randn(16, C2), torch.empty((0, C2), dtype=torch.float32)]
    prepared2 = prepare_ldlq_gate_evidence(acts2, qname="test.one_warm_one_cold")
    assert prepared2.eligible_experts == (0,)
    # Verify cold rows bit-identical to raw after gated
    fields2 = nvfp4_cb_fields(w2, 12, grid="fp4", mode="product", col_weights=cw2)
    out2, gate2 = ldlq_reassign_cb_fields_gated(w2, fields2, cw2, acts2, grid="fp4", mode="product", k=12, prepared=prepared2)
    assert gate2["per_expert_kept"][1] is False
    assert torch.equal(out2["indices"][R2:], fields2["indices"][R2:])
    n2 = _count_hessian_calls_via_patch(w2, cw2, acts2, prepared2, range(12, 19))
    assert n2 == 1, f"expected 1 hessian for 1 warm+1 cold across 7 rungs, got {n2}"

    # Case 3: all cold -> 0 calls
    torch.manual_seed(3)
    acts3 = [torch.empty((0, C2), dtype=torch.float32) for _ in range(E2)]
    prepared3 = prepare_ldlq_gate_evidence(acts3, qname="test.all_cold")
    assert prepared3.eligible_experts == ()
    n3 = _count_hessian_calls_via_patch(w2, cw2, acts3, prepared3, range(12, 19))
    assert n3 == 0, f"expected 0 hessian for all cold, got {n3}"
    # All-cold must return raw with zero factors and cold rows bit-identical
    fields3 = nvfp4_cb_fields(w2, 12, grid="fp4", mode="product", col_weights=cw2)
    out3, gate3 = ldlq_reassign_cb_fields_gated(w2, fields3, cw2, acts3, grid="fp4", mode="product", k=12, prepared=prepared3)
    assert torch.equal(out3["indices"], fields3["indices"])
    if "signs" in fields3:
        assert torch.equal(out3["signs"], fields3["signs"])
    assert gate3["gate"] == "raw_fallback_missing_activation"


def test_dense_factor_one_across_seven_rungs():
    torch.manual_seed(4)
    R, C = 8, 256
    w = torch.randn(R, C)
    cw = torch.ones(1, C)
    act = torch.randn(32, C)
    prepared = prepare_ldlq_gate_evidence(act, qname="test.dense_factor")
    n = _count_hessian_calls_via_patch(w, cw, act, prepared, range(12, 19))
    assert n == 1, f"expected 1 dense hessian across 7 rungs, got {n}"


def test_prepared_validation_fail_closed_on_mismatch():
    torch.manual_seed(5)
    E, R, C = 2, 8, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    acts = [torch.randn(16, C) for _ in range(E)]
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.validate")
    other_acts = [torch.randn(16, C) for _ in range(E)]
    fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    out_fields, gate = ldlq_reassign_cb_fields_gated(w, fields, cw, other_acts, grid="fp4", mode="product", k=12, prepared=prepared)
    assert gate["gate"] == "raw_fallback_malformed_activation"
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    spec = fr.get_format("NVFP4_CB_K12")
    fields2, gi2 = cb_fields_for_context(spec, w, context=ctx, col_weights=cw, activation_rows=other_acts, return_gate_info=True, prepared_evidence=prepared)
    assert gi2["gate"].startswith("raw_fallback")


def test_aliased_inplace_mutation_after_prepare_fails_closed():
    """Adversarial: same aliased caller object mutated in place after prepare — must fail closed."""
    torch.manual_seed(6)
    E, R, C = 2, 8, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    # Use same tensor objects for acts and prepared
    base = torch.randn(16, C)
    acts = [base, torch.randn(16, C)]
    # Keep second expert warm for contrast
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.aliased_mutation")
    # Mutate first expert's tensor in place via aliased reference (both prepared.original_rows[0] and acts[0] share same storage)
    # Adding 1.0 to first row changes content hash but alias means both see mutation
    acts[0][0, 0] += 10.0
    # Now prepared.original_rows[0] is mutated too (same object), but stored digest is pre-mutation
    fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    out_fields, gate = ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12, prepared=prepared)
    # Must detect mutation and fallback raw for all, with zero extra factors for mutated expert (fail closed)
    assert gate["gate"] == "raw_fallback_malformed_activation"
    assert "mutated" in gate.get("reason", "").lower() or "mismatch" in gate.get("reason", "").lower()
    # Also test stale fit/holdout snapshot: fit tensor mutated after prepare should also fail
    torch.manual_seed(7)
    acts2 = [torch.randn(16, C) for _ in range(2)]
    prepared2 = prepare_ldlq_gate_evidence(acts2, qname="test.stale_fit")
    # Mutate the stored fit tensor object in place (prepared.fit_rows is aliased to prepared object)
    # Since fit_rows are derived slices, we can mutate one element
    if prepared2.fit_rows[0].numel() > 0:
        prepared2.fit_rows[0][0, 0] += 10.0
        fields2 = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
        out2, gate2 = ldlq_reassign_cb_fields_gated(w, fields2, cw, acts2, grid="fp4", mode="product", k=12, prepared=prepared2)
        assert gate2["gate"] == "raw_fallback_malformed_activation"


def test_nested_split_info_mutation_fails_closed():
    torch.manual_seed(8)
    E, R, C = 2, 8, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    acts = [torch.randn(16, C) for _ in range(E)]
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.split_mutation")
    # Plain dicts are mutable (serializable), but mutation must fail closed via digest validation
    # Tamper and verify gate falls back raw
    tampered = prepared
    # Mutate copy to simulate caller tampering
    tampered.split_infos[0]["fit_digest"] = "tampered"
    fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    out_tamper, gate_tamper = ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12, prepared=tampered)
    assert gate_tamper["gate"] == "raw_fallback_malformed_activation"
    assert "mutated" in gate_tamper.get("reason", "").lower() or "digest" in gate_tamper.get("reason", "").lower()
    # Restore by re-preparing
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.split_mutation")
    for i, info in enumerate(prepared.split_infos):
        assert dict(info)["fit_digest"] == prepared.fit_digests[i]
        assert dict(info)["holdout_digest"] == prepared.holdout_digests[i]
    # Verify plain dicts are JSON and pickle serializable (no MappingProxyType)
    import json, pickle
    # Prepared itself picklable
    pickle.dumps(prepared)
    # split_infos JSON serializable
    json.dumps([dict(s) for s in prepared.split_infos], sort_keys=True)
    json.dumps(list(prepared.split_infos), sort_keys=True)
    # gate_info JSON/pickle serializable via actual gate
    fields2 = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    out2, gate2 = ldlq_reassign_cb_fields_gated(w, fields2, cw, acts, grid="fp4", mode="product", k=12, prepared=prepared)
    json.dumps(gate2, sort_keys=True)
    pickle.dumps(gate2)
    # Ensure reordered-but-equivalent caller rows still validate (multiset invariant)
    torch.manual_seed(9)
    acts_perm = [torch.randn(16, C) for _ in range(E)]
    # Create a permutation of rows within first expert
    perm = torch.randperm(acts_perm[0].shape[0])
    acts_perm_shuffled = [acts_perm[0][perm], acts_perm[1]]
    prepared_perm = prepare_ldlq_gate_evidence(acts_perm, qname="test.perm")
    fields_perm = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    # Shuffled order should still validate because digest is multiset sorted
    out_perm, gate_perm = ldlq_reassign_cb_fields_gated(w, fields_perm, cw, acts_perm_shuffled, grid="fp4", mode="product", k=12, prepared=prepared_perm)
    # If permutation is pure reorder, should not be malformed; may be kept raw or ldlq but not fallback malformed
    assert gate_perm["gate"] != "raw_fallback_malformed_activation" or "mismatch" not in gate_perm.get("reason", "")


def test_derive_packed_prepared_owner_integration_real_derive_layer_full(tmp_path, monkeypatch):
    """Real derive_layer_full integration via mocked files — proves owner reaches encode and cold stays raw."""
    import tools.derive_dual_basis_packed as mod
    import pickle, hashlib, json
    torch.manual_seed(10)
    E, R, C = 2, 8, 256
    # Use real temporary paths for derive
    orig_ckpt = mod.DERIVED_CHECKPOINTS
    orig_dense_ckpt = mod.DERIVED_DENSE_CHECKPOINTS
    orig_raw_plane = mod.DERIVED_RAW_PLANE
    orig_shards = mod.DERIVED_SHARDS
    orig_source = mod.SOURCE
    orig_by_layer = mod.BY_LAYER
    orig_raw_shards = mod.RAW_SHARDS
    orig_run_root = mod.RUN_ROOT
    orig_warm = mod.DERIVED_WARM
    try:
        mod.DERIVED_CHECKPOINTS = tmp_path / "ckpt"
        mod.DERIVED_DENSE_CHECKPOINTS = tmp_path / "dense_ckpt"
        mod.DERIVED_RAW_PLANE = tmp_path / "raw_plane"
        mod.DERIVED_SHARDS = tmp_path / "shards"
        mod.RAW_SHARDS = tmp_path / "raw_shards"
        mod.BY_LAYER = tmp_path / "by_layer"
        mod.SOURCE = tmp_path / "source"
        mod.RUN_ROOT = tmp_path / "run_root"
        mod.DERIVED_WARM = tmp_path / "warm"
        for p in [mod.DERIVED_CHECKPOINTS, mod.DERIVED_DENSE_CHECKPOINTS, mod.DERIVED_RAW_PLANE, mod.DERIVED_SHARDS, mod.RAW_SHARDS, mod.BY_LAYER, mod.SOURCE, mod.RUN_ROOT, mod.DERIVED_WARM]:
            p.mkdir(parents=True, exist_ok=True)
        # Create minimal source index file required by _cached_source_index_sha256
        (mod.SOURCE / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        # Create dummy by-layer probe identity
        by_layer_payload = {"stats": {}, "costs": {}, "meta": {}}
        # Need col_weights file
        col_weights_path = tmp_path / "col_weights.pkl"
        # Setup raw shard with one packed projection and one dense
        raw_identity = {
            "layer": 0,
            "col_weights_shapes": {},
            "col_weights_content_sha256": {},
            "col_weights_shapes": {},
            "formats": ["NVFP4_CB_K12", "NVFP4_CB_K15", "NVFP4_CB_K18"],
        }
        # We'll mock load_layer_identity and other heavy IO
        monkeypatch.setattr(mod, "require_cuda", lambda _: None)
        monkeypatch.setattr(mod, "require_ldlq_gate_enabled", lambda: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        # Mock planning to return gate_up_proj with 2 experts, one warm one cold
        from unittest.mock import MagicMock
        mock_profile = MagicMock()
        mock_profile.packed_expert_param_names.return_value = frozenset({"gate_up_proj"})
        mock_profile.packed_expert_projection_names.side_effect = lambda n: ("gate_proj", "up_proj") if n == "gate_up_proj" else (n,)
        # Provide minimal expert groups/members
        expert_groups = {
            "model.layers.0.mlp.experts": {
                "gate_proj": {0: "model.layers.0.mlp.experts.0.gate_proj", 1: "model.layers.0.mlp.experts.1.gate_proj"},
                "up_proj": {0: "model.layers.0.mlp.experts.0.up_proj", 1: "model.layers.0.mlp.experts.1.up_proj"},
            }
        }
        expert_stack_members = {
            "model.layers.0.mlp.experts.gate_up_proj": {("gate_proj", 0): "model.layers.0.mlp.experts.0.gate_proj", ("gate_proj", 1): "model.layers.0.mlp.experts.1.gate_proj", ("up_proj", 0): "model.layers.0.mlp.experts.0.up_proj", ("up_proj", 1): "model.layers.0.mlp.experts.1.up_proj"},
        }
        monkeypatch.setattr(mod, "_get_packed_planning", lambda: (mock_profile, expert_groups, expert_stack_members, frozenset({"gate_up_proj"})))
        # Mock load_packed_projection to return 1 warm + 1 cold
        torch.manual_seed(11)
        w_stack = torch.randn(E, R, C, dtype=torch.bfloat16)
        cw_stack = torch.ones(E, 1, C, dtype=torch.float32)
        warm_act = torch.randn(16, C)
        cold_act = torch.empty((0, C), dtype=torch.float32)
        acts_with_cold = (warm_act, cold_act)
        dummy_data = {
            "weight": w_stack,
            "col_weights": cw_stack,
            "activation_rows": acts_with_cold,
            "activation_rows_original": acts_with_cold,
            "activation_rows_filled_for_hessian": acts_with_cold,
            "cold_experts": [1],
            "projections": ("gate_proj", "up_proj"),
            "slice_boundaries": {"gate_proj": (0, R//2), "up_proj": (R//2, R)},
            "member_order": ["gate_proj", "up_proj"],
            "col_weight_pooling": "mean_of_member_vectors",
            "qnames_per_leaf": {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]},
            "leaf_out_dims": [R//2, R//2],
            "observed_activation_files": 1,
        }
        monkeypatch.setattr(mod, "load_packed_projection", lambda *a, **k: dummy_data)
        # Mock layer identity and col weights
        monkeypatch.setattr(mod, "load_layer_identity", lambda layer: ({"costs": {}}, {"identity": {"col_weights_shapes": {q: [8] for q in ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.gate_proj", "model.layers.0.mlp.experts.1.up_proj"]}, "col_weights_content_sha256": {}}, "sha256": "a"*64}))
        monkeypatch.setattr(mod, "_load_col_weights_cached", lambda: {q: torch.ones(8) for q in ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.gate_proj", "model.layers.0.mlp.experts.1.up_proj"]})
        monkeypatch.setattr("prismaquant.layer_streaming._build_weight_map", lambda p: ({}, {}))
        monkeypatch.setattr("prismaquant.layer_streaming._build_fp8_scale_inv_map", lambda p: {})
        monkeypatch.setattr(mod, "_cached_source_index_sha256", lambda: "b"*64)
        monkeypatch.setattr(mod, "_cached_col_weights_sha256", lambda: "c"*64)
        monkeypatch.setattr(mod, "_cached_tool_sha256", lambda: "d"*64)
        monkeypatch.setattr(mod, "sha256_file", lambda p: "e"*64)
        monkeypatch.setattr(mod, "atomic_bytes_copy", lambda src, dst: (dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(src.read_bytes()), "e"*64)[2])
        monkeypatch.setattr(mod, "content_sha256_float32", lambda t: "f"*64)
        # Warm state IO is heavy source boundary — mock to avoid real lattice/codebook hashing and FS
        class _DummyWarmStore:
            def __init__(self, *a, **k): pass
            def write(self, rec): return str(tmp_path / "warm_dummy")
        monkeypatch.setattr("prismaquant.cb_warm_state.CBWarmStateStore", _DummyWarmStore)
        monkeypatch.setattr("prismaquant.cb_warm_state.build_warm_record", lambda *a, **k: {"dummy": 1})
        # Prepare raw shard file
        import pickle as pkl
        def _sha(p): return hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # Present all declared K12-K18 raw rungs (7 rungs) for true owner fixture
        all_rungs = list(range(12, 19))
        all_formats = [f"NVFP4_CB_K{k}" for k in all_rungs]
        raw_costs_entry = {fmt: {"weight_mse": 0.1, "output_mse": 0.2} for fmt in all_formats}
        raw_payload = {
            "schema": mod.SCHEMA,
            "identity": {"test": 1, "serialization_context": {"lattice_codebook_sha256_by_format": {fmt: "x" for fmt in all_formats}}, "formats": all_formats},
            "costs": {
                "model.layers.0.mlp.experts.0.gate_proj": dict(raw_costs_entry),
                "model.layers.0.mlp.experts.1.gate_proj": dict(raw_costs_entry),
                "model.layers.0.mlp.experts.0.up_proj": dict(raw_costs_entry),
                "model.layers.0.mlp.experts.1.up_proj": dict(raw_costs_entry),
            },
            "formats": all_formats,
        }
        raw_payload["content_key"] = _sha(raw_payload["identity"])
        raw_path = mod.RAW_SHARDS / "layer_000.pkl"
        raw_path.write_bytes(pkl.dumps(raw_payload))
        # Mock to present all 7 rungs
        monkeypatch.setattr(mod, "_expected_nvfp4_formats_from_raw", lambda costs: list(all_formats))
        monkeypatch.setattr(mod, "_nvfp4_rungs_from_formats", lambda fmts: list(all_rungs))
        monkeypatch.setattr(mod, "NVFP4_RUNGS", tuple(all_rungs))
        monkeypatch.setattr(mod, "CBSerializationContext", MagicMock())
        monkeypatch.setattr(mod, "cb_serialization_context_stamp", lambda *a, **k: {"ldlq_scope": "nvfp4"})
        # Wrap real encode (spy) — count calls and verify cold stays raw, without replacing logic with direct gate call
        encode_calls = {"n": 0}
        orig_encode = mod.encode_nvfp4_rung_packed

        def counting_encode(layer, proj, rung, data, device, write_warm_state=True, prepared_evidence=None):
            encode_calls["n"] += 1
            assert prepared_evidence is not None
            assert 1 in prepared_evidence.cold_experts
            assert prepared_evidence.eligible_experts == (0,)
            res = orig_encode(layer, proj, rung, data, device, write_warm_state=write_warm_state, prepared_evidence=prepared_evidence)
            assert res["gate_info"]["per_expert_kept"][1] is False
            return res

        monkeypatch.setattr(mod, "encode_nvfp4_rung_packed", counting_encode)
        # Ensure checkpoints are missing initially
        monkeypatch.setattr(mod, "validated_projection_checkpoint", lambda p, ident: None)
        monkeypatch.setattr(mod, "validated_dense_checkpoint", lambda p, ident: None)
        monkeypatch.setattr(mod, "load_dense_tensor", lambda *a, **k: {"weight": torch.randn(4, 256), "col_weights": torch.ones(256), "activation_rows": torch.randn(4,256), "qname": "model.layers.0.self_attn.q_proj"})
        monkeypatch.setattr(mod, "measure_dense_all_rungs", lambda dense_loaded, rungs: {r: {"weight_mse": 0.1, "output_mse": 0.2, "rel_output_mse": 0.01, "n_activation_rows": 4, "cost_source": "raw_immutable_copy", "ldlq_scope": "nvfp4", "gate_info": {"gate": "ldlq_kept"}, "gate": "ldlq_kept"} for r in rungs})
        # Track factor cache and ensure cleanup
        from prismaquant.nvfp4_cb_formats import clear_ldlq_factor_cache, ldlq_factor_cache_size
        clear_ldlq_factor_cache()
        # Mock _cached to count hessian
        import prismaquant.rotation_ldlq_pilot as pilot
        hessian_calls = {"n": 0}
        orig_hess = pilot.inverse_hessian_cholesky

        def counting_hess(*a, **kw):
            hessian_calls["n"] += 1
            return orig_hess(*a, **kw)

        with patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting_hess):
            out_path = mod.derive_layer_full(0, torch.device("cpu"))
            assert out_path.is_file()
            # Should have reached encode for each missing rung (7 rungs K12-K18)
            assert encode_calls["n"] == 7
            # Should have built exactly 1 factor across 7 rungs for 1 warm (cached)
            assert hessian_calls["n"] == 1, f"expected 1 hessian for 1 warm across 7 rungs, got {hessian_calls['n']}"
            # Cold stayed raw proven via encode assertion
            assert ldlq_factor_cache_size() == 0, "factor cache must be cleared after derive"

        # All-resumed case: checkpoints exist, zero encode and zero factor builds
        encode_calls["n"] = 0
        hessian_calls["n"] = 0
        # Make validated return existing with full metrics
        monkeypatch.setattr(mod, "validated_projection_checkpoint", lambda p, ident: {"result": {"gate_info": {"gate": "ldlq_kept", "per_expert_kept": [True, False], "raw_mse_per_expert": [0.2, 0.2], "ldlq_mse_per_expert": [0.1, 0.2]}, "per_leaf": {"gate_proj": {"weight_mse_per_expert": [0.1, 0.1], "weighted_mse_per_expert": [0.1, 0.1], "output_mse_per_expert": [0.2, 0.2], "rel_output_mse_per_expert": [0.01, 0.01]}, "up_proj": {"weight_mse_per_expert": [0.1, 0.1], "weighted_mse_per_expert": [0.1, 0.1], "output_mse_per_expert": [0.2, 0.2], "rel_output_mse_per_expert": [0.01, 0.01]}}, "weight_mse_per_expert": [0.1,0.1], "weighted_mse_per_expert": [0.1,0.1], "output_mse_per_expert": [0.2,0.2], "rel_output_mse_per_expert": [0.01,0.01], "n_activation_rows_per_expert": [16,0], "expert_count": 2, "member_order": ["gate_proj", "up_proj"]}})

        def should_not_encode(*a, **k):
            raise AssertionError("encode should not be called when all checkpoints resumed")

        monkeypatch.setattr(mod, "encode_nvfp4_rung_packed", should_not_encode)
        with patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting_hess):
            out_path2 = mod.derive_layer_full(0, torch.device("cpu"))
            assert out_path2.is_file()
            assert hessian_calls["n"] == 0, f"all-resumed should build zero factors, got {hessian_calls['n']}"
        assert ldlq_factor_cache_size() == 0
    finally:
        mod.DERIVED_CHECKPOINTS = orig_ckpt
        mod.DERIVED_DENSE_CHECKPOINTS = orig_dense_ckpt
        mod.DERIVED_RAW_PLANE = orig_raw_plane
        mod.DERIVED_SHARDS = orig_shards
        mod.SOURCE = orig_source
        mod.BY_LAYER = orig_by_layer
        mod.RAW_SHARDS = orig_raw_shards
        mod.RUN_ROOT = orig_run_root
        mod.DERIVED_WARM = orig_warm
        clear_ldlq_factor_cache()
