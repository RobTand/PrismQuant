"""Adversarial tests for per-family LDLQ scope, empty-as-unset, shell args and gate telemetry."""

import os
import subprocess

import torch

from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_serialization_context_from_env, cb_serialization_context_from_stamp, cb_serialization_context_stamp
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_ldlq import fill_empty_expert_activation_rows


def test_scope_unset_maps_to_none():
    ctx = cb_serialization_context_from_env(
        environ={"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"},
        where="test",
    )
    assert ctx.ldlq is False
    assert ctx.ldlq_scope == "none"


def test_scope_empty_treated_as_unset():
    env = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ_SCOPE": "", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    ctx = cb_serialization_context_from_env(environ=env, where="test")
    # empty should be unset -> derive from legacy (default false -> none)
    assert ctx.ldlq_scope == "none"
    assert ctx.ldlq is False
    # empty with legacy 1 via bool? Actually legacy default false, but test empty + explicit 0
    env2 = dict(env, **{"PRISMAQUANT_CB_LDLQ": "1", "PRISMAQUANT_CB_LDLQ_SCOPE": "   "})
    ctx2 = cb_serialization_context_from_env(environ=env2, where="test")
    assert ctx2.ldlq_scope == "none" or ctx2.ldlq_scope == "all"  # whitespace empty treated as unset -> ldlq true -> all
    # When empty scope and legacy 1, scope should be all
    assert ctx2.ldlq_scope == "all"


def test_scope_only_nvfp4_normalizes_global_true():
    env = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    ctx = cb_serialization_context_from_env(environ=env, where="test")
    assert ctx.ldlq_scope == "nvfp4"
    assert ctx.ldlq is True  # any-LDLQ true for nvfp4
    # Also via CBSerializationContext direct
    ctx2 = CBSerializationContext.production(ldlq_scope="nvfp4")
    assert ctx2.ldlq is True
    assert ctx2.ldlq_scope == "nvfp4"


def test_conflicting_scope_and_legacy_rejected():
    # scope none with legacy 1 should error
    env = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "1", "PRISMAQUANT_CB_LDLQ_SCOPE": "none", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    try:
        cb_serialization_context_from_env(environ=env, where="test")
        assert False, "should have raised for conflicting none/1"
    except ValueError as exc:
        assert "inconsistent" in str(exc).lower()
    # scope all with legacy 0 should error
    env2 = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "0", "PRISMAQUANT_CB_LDLQ_SCOPE": "all", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    try:
        cb_serialization_context_from_env(environ=env2, where="test")
        assert False
    except ValueError:
        pass
    # scope nvfp4 with legacy 0 should error (since any true)
    env3 = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "0", "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    try:
        cb_serialization_context_from_env(environ=env3, where="test")
        assert False
    except ValueError:
        pass


def test_scope_nvfp4_with_legacy_1_allowed():
    env = {"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "1", "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}
    ctx = cb_serialization_context_from_env(environ=env, where="test")
    assert ctx.ldlq_scope == "nvfp4"
    assert ctx.ldlq is True


def test_old_stamp_without_scope_maps_all_or_none():
    # Old stamp with ldlq true -> all
    stamp_true = {"schema": "prismaquant.cb_serialized_payload.v3", "scale_coding": "two_tier", "layout_version": 2, "codebook_source": "lattice", "scale_sweep": True, "ldlq": True, "encode_tier": "balanced", "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1", "activation_contract": "prismaquant.nvfp4_w4a4_activation.v1", "activation_execution": "e2m1_group16_ue4m3_static"}
    ctx = cb_serialization_context_from_stamp(stamp_true, where="test")
    assert ctx.ldlq_scope == "all"
    assert ctx.ldlq is True
    # old with false -> none
    stamp_false = dict(stamp_true, **{"ldlq": False})
    ctx2 = cb_serialization_context_from_stamp(stamp_false, where="test")
    assert ctx2.ldlq_scope == "none"
    assert ctx2.ldlq is False
    # new stamp with nvfp4
    stamp_nvfp4 = dict(stamp_true, **{"ldlq_scope": "nvfp4"})
    ctx3 = cb_serialization_context_from_stamp(stamp_nvfp4, where="test")
    assert ctx3.ldlq_scope == "nvfp4"
    assert ctx3.ldlq is True


def test_allocator_scope_without_legacy_cli():
    # Simulate allocator CLI --cb-ldlq-scope nvfp4 without --cb-ldlq
    # Use direct CBSerializationContext via allocator logic: scope nvfp4 should normalize ldlq true
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4")
    assert ctx.ldlq is True
    # Also test empty string treated as unset
    ctx2 = CBSerializationContext(scale_coding="two_tier", codebook_source="lattice", layout_version=2, scale_sweep=True, ldlq=False, ldlq_scope="", encode_tier="balanced", renderer_abi="prismaquant.nvfp4_cb_renderer.v1", activation_contract="prismaquant.nvfp4_w4a4_activation.v1", activation_execution="e2m1_group16_ue4m3_static")
    assert ctx2.ldlq_scope == "none"


def test_shell_argument_shape_explicit_append():
    # Ensure run-pipeline.sh uses explicit array, not conditional expansion inside array
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "prismaquant/run-pipeline.sh"
    text = p.read_text()
    # Old buggy pattern: ${PRISMAQUANT_CB_LDLQ_SCOPE:+--cb-ldlq-scope inside ALLOCATOR_CB_ARGS
    assert '${PRISMAQUANT_CB_LDLQ_SCOPE:+--cb-ldlq-scope' not in text, "conditional expansion still inside ALLOCATOR_CB_ARGS"
    # Must pass --cb-ldlq-scope explicitly via array (either inline or via +=)
    assert '--cb-ldlq-scope' in text
    assert 'ALLOCATOR_CB_ARGS' in text and '--cb-ldlq-scope' in text
    # STAGE_SETTINGS_ENV must contain scope
    assert 'PRISMAQUANT_CB_LDLQ_SCOPE=' in text


def test_stage_fingerprints_include_scope():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS, stage_settings_projection

    # Every artifact that keys on _CB_SERIALIZATION_SETTINGS should include scope and gate
    for stage, keys in STAGE_SETTINGS_KEYS.items():
        sources = [src for _, src in keys]
        if "PRISMAQUANT_CB_LDLQ" in sources:
            assert "PRISMAQUANT_CB_LDLQ_SCOPE" in sources, f"{stage} missing LDLQ scope key"
            assert "PRISMAQUANT_CB_LDLQ_GATE" in sources, f"{stage} missing LDLQ gate key"
    # Projection should succeed when scope and gate provided
    proj, unresolved = stage_settings_projection("base-cost", {"MODEL_PATH": "x", "DATASET": "y", "NSAMPLES": "1", "SEQLEN": "1", "FORMATS": "z", "CB_SCALE_CODING": "a", "CB_CODEBOOK_SOURCE": "b", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "0", "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4", "PRISMAQUANT_CB_LDLQ_GATE": "1", "PRISMAQUANT_CB_MINCHAIN": "0", "PRISMAQUANT_CB_MINCHAIN_ANCHORS": "", "PRISMAQUANT_CB_MINCHAIN_HOLDBACKS": "", "PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED": "42", "PRISMAQUANT_CB_MINCHAIN_BACKSTOP": "0.25", "PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN": "0.05", "PRISMAQUANT_CB_MINCHAIN_AUDIT_P95": "0.15", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"})
    assert not unresolved


def test_gate_mixed_empty_nonempty_experts():
    # 4 experts, 2 missing, 2 observed; gate should produce per-expert telemetry
    torch.manual_seed(0)
    E, R, C = 4, 2, 256
    w = torch.randn(E, R, C)
    cw = torch.randn(E, 1, C).abs()
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    acts = [torch.randn(8, C) if i % 2 == 0 else torch.empty((0, C)) for i in range(E)]
    out_fields, info = cb.ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12)
    assert "per_expert_kept" in info
    assert len(info["per_expert_kept"]) == E
    assert out_fields["indices"].shape == fields["indices"].shape


def test_gate_all_empty_fails_closed_with_full_telemetry():
    torch.manual_seed(1)
    E, R, C = 4, 2, 256
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    acts = [torch.empty((0, C)) for _ in range(E)]
    out_fields, info = cb.ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12)
    assert info["gate"] == "raw_fallback_missing_activation"
    assert "per_expert_kept" in info
    assert len(info["per_expert_kept"]) == E
    assert all(v is False for v in info["per_expert_kept"])
    assert out_fields["indices"].equal(fields["indices"])


def test_pooled_missing_semantics_match_loader():
    torch.manual_seed(2)
    C = 256
    acts = [torch.randn(4, C), torch.empty((0, C)), torch.randn(5, C)]
    filled, missing = fill_empty_expert_activation_rows(tuple(acts), qname="test")
    assert missing == (1,)
    assert filled[1].shape[0] == acts[0].shape[0] + acts[2].shape[0]
    E, R = 3, 2
    w = torch.randn(E, R, C)
    cw = torch.ones(E, 1, C)
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    out_fields, info = cb.ldlq_reassign_cb_fields_gated(w, fields, cw, acts, grid="fp4", mode="product", k=12)
    assert "per_expert_kept" in info


def test_cost_and_export_same_gated_function():
    import inspect
    from pathlib import Path
    src = inspect.getsource(cb.ldlq_reassign_cb_fields_gated)
    assert "activation_output_mse" in src
    root = Path(__file__).resolve().parents[1]
    assert "ldlq_reassign_cb_fields_gated" in (root / "prismaquant/nvfp4_cb_footprint.py").read_text()
    assert "ldlq_reassign_cb_fields_gated" in (root / "prismaquant/nvfp4_cb_formats.py").read_text()
    # Both pack and fields should check gate
    assert "_ldlq_gate_enabled" in (root / "prismaquant/nvfp4_cb_formats.py").read_text()


def test_nvfp4_pack_returns_gate_telemetry():
    torch.manual_seed(3)
    w = torch.randn(4, 256)
    cw = torch.ones(1, 256)
    act = torch.randn(8, 256)
    packed, fields, gate_info = cb.nvfp4_cb_pack(w, 12, grid="fp4", mode="product", col_weights=cw, activation_rows=act, ldlq=True, return_gate_info=True)
    assert isinstance(gate_info, dict)
    assert "gate" in gate_info
    packed2, fields2 = cb.nvfp4_cb_pack(w, 12, grid="fp4", mode="product", col_weights=cw, activation_rows=act, ldlq=True, return_gate_info=False)
    assert isinstance(packed2, torch.Tensor)


def test_pipeline_truth_table_actual_bash():
    # Exercise the actual production normalization logic via the shared helper,
    # not a copied snippet. Sources prismaquant/cb_ldlq_normalize.sh and calls
    # normalize_cb_ldlq_vars — the same code run-pipeline.sh uses.
    import subprocess
    from pathlib import Path

    helper = Path(__file__).resolve().parents[1] / "prismaquant/cb_ldlq_normalize.sh"
    assert helper.is_file(), f"helper missing {helper}"
    snippet = f'''
        set -euo pipefail
        source "{helper}"
        _run() {{
          unset PRISMAQUANT_CB_LDLQ PRISMAQUANT_CB_LDLQ_SCOPE
          if [[ "$1" != "UNSET" ]]; then export PRISMAQUANT_CB_LDLQ="$1"; fi
          if [[ "$2" != "UNSET" ]]; then export PRISMAQUANT_CB_LDLQ_SCOPE="$2"; fi
          normalize_cb_ldlq_vars || return $?
          printf '%s %s\\n' "$PRISMAQUANT_CB_LDLQ" "$PRISMAQUANT_CB_LDLQ_SCOPE"
        }}
        _run UNSET UNSET
        _run 0 UNSET
        _run 1 UNSET
        _run UNSET nvfp4
        _run UNSET none
        _run UNSET ""
        if _run 0 nvfp4 2>/dev/null 1>/dev/null; then echo "should have failed" >&2; exit 1; fi
        if _run 1 none 2>/dev/null 1>/dev/null; then echo "should have failed" >&2; exit 1; fi
        echo "ok"
    '''
    result = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"bash truth table failed: {result.stdout} {result.stderr}"
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "0 none"  # neither set
    assert lines[1] == "0 none"  # legacy 0 -> scope none
    assert lines[2] == "1 all"   # legacy 1 -> scope all
    assert lines[3] == "1 nvfp4" # scope nvfp4 with legacy unset -> bool 1
    assert lines[4] == "0 none"  # scope none with legacy unset -> bool 0
    assert lines[5] == "0 none"  # empty scope -> unset -> 0 none
    assert lines[6] == "ok"
    # Also verify --normalize mode prints correctly
    r2 = subprocess.run(["bash", str(helper), "--normalize"], capture_output=True, text=True, timeout=5, env={**__import__("os").environ, "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4"})
    assert r2.returncode == 0 and "1 nvfp4" in r2.stdout


def test_allocator_scope_only_emits_provenance():
    """Real allocator main-path: --cb-ldlq-scope nvfp4 without --cb-ldlq emits allocation with nvfp4 scope."""
    import json
    import pickle
    import sys
    from pathlib import Path

    import torch
    import prismaquant.allocator as alloc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext
    from prismaquant.production_weight_cache import build_production_cache_cb_render_identity, bind_cb_render_identity_source_weights

    # Build a minimal 256-aligned dense fixture that survives the CB legality checks.
    # Use complete-layer shapes from _dense_model (all fused groups complete) to avoid pinned-away BF16.
    menu = ["NVFP4_CB_K12", "FP8_CB_K28", "BF16"]
    specs = [__import__("prismaquant.format_registry", fromlist=["get_format"]).get_format(n) for n in menu]
    stats = {
        "model.layers.0.self_attn.q_proj": {"h_trace": 0.5, "n_params": 2048 * 1024, "in_features": 1024, "out_features": 2048},
        "model.layers.0.self_attn.k_proj": {"h_trace": 0.3, "n_params": 256 * 1024, "in_features": 1024, "out_features": 256},
        "model.layers.0.self_attn.v_proj": {"h_trace": 0.7, "n_params": 256 * 1024, "in_features": 1024, "out_features": 256},
        "model.layers.0.self_attn.o_proj": {"h_trace": 0.4, "n_params": 1024 * 2048, "in_features": 2048, "out_features": 1024},
        "model.layers.0.mlp.gate_proj": {"h_trace": 0.8, "n_params": 3072 * 1024, "in_features": 1024, "out_features": 3072},
        "model.layers.0.mlp.up_proj": {"h_trace": 0.6, "n_params": 3072 * 1024, "in_features": 1024, "out_features": 3072},
        "model.layers.0.mlp.down_proj": {"h_trace": 0.9, "n_params": 1024 * 3072, "in_features": 3072, "out_features": 1024},
    }
    def _cost_entry(dloss: float) -> dict:
        return {"weight_mse": dloss, "predicted_dloss": dloss}
    def _costs_for(h):
        return {s.name: _cost_entry(0.02 * h / max(s.effective_bits, 1.0)) for s in specs}
    costs = {name: _costs_for(entry["h_trace"]) for name, entry in stats.items()}
    # CB context with scope nvfp4 — the allocator should stamp the same.
    context = CBSerializationContext.production(ldlq_scope="nvfp4")
    col_weights = {qname: torch.linspace(0.1, 1.0, int(entry["in_features"])) for qname, entry in stats.items()}
    formats_by_qname = {qname: [s.name for s in specs if s.name.startswith("NVFP4_CB") or s.name.startswith("FP8_CB")] for qname in stats}
    # Build a real render identity so provenance validates
    render_identity = build_production_cache_cb_render_identity(
        formats_by_qname, cb_serialization_context=context, col_weights=col_weights, render_levers={"weighted_vq": True}, render_mechanism_plan=[],
    )
    source_weights = {qname: torch.zeros((int(entry["out_features"]), int(entry["in_features"])), dtype=torch.bfloat16) for qname, entry in stats.items()}
    render_identity = bind_cb_render_identity_source_weights(render_identity, source_weights)
    provenance = {"cb_serialized_payload": render_identity["cb_serialized_payload"], "cb_render_identity": render_identity}

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        probe_p = td / "probe.pkl"
        cost_p = td / "cost.pkl"
        cw_p = td / "col_weights.pkl"
        lc = td / "layer_config.json"
        csv = td / "pareto.csv"
        probe_p.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": None}}))
        cost_p.write_bytes(pickle.dumps({"costs": costs, "formats": menu, "meta": {"formats": menu}, "provenance": provenance}))
        cw_p.write_bytes(pickle.dumps(col_weights))
        # Invoke the real allocator main-path with scope-only CLI (no --cb-ldlq)
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
        assert lc.is_file(), "allocator did not emit layer_config.json with scope-only CLI"
        emitted = json.loads(lc.read_text())
        # Provenance is in the reserved block
        cb_payload = emitted.get("__prismaquant__", {}).get("cb_serialized_payload") or emitted.get("cb_serialized_payload")
        assert cb_payload is not None, "emitted config missing cb_serialized_payload"
        assert cb_payload["ldlq_scope"] == "nvfp4", f"expected scope nvfp4, got {cb_payload.get('ldlq_scope')}"
        assert cb_payload["ldlq"] is True
        # Also check that NVFP4 entries would be LDLQ and FP8 not, via context helper
        from prismaquant.nvfp4_cb_footprint import _ldlq_for_format
        assert _ldlq_for_format("NVFP4_CB_K12", context) is True
        assert _ldlq_for_format("FP8_CB_K28", context) is False


def test_allocator_scope_only_parses():
    # Legacy empty-scope handling still maps to none
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_serialization_context_from_env
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4")
    assert ctx.ldlq is True and ctx.ldlq_scope == "nvfp4"
    ctx2 = CBSerializationContext(scale_coding="two_tier", codebook_source="lattice", layout_version=2, scale_sweep=True, ldlq=False, ldlq_scope="", encode_tier="balanced", renderer_abi="prismaquant.nvfp4_cb_renderer.v1", activation_contract="prismaquant.nvfp4_w4a4_activation.v1", activation_execution="e2m1_group16_ue4m3_static")
    assert ctx2.ldlq_scope == "none"
    try:
        cb_serialization_context_from_env(environ={"CB_SCALE_CODING": "two_tier", "CB_CODEBOOK_SOURCE": "lattice", "CB_SCALE_SWEEP": "1", "PRISMAQUANT_CB_LDLQ": "0", "PRISMAQUANT_CB_LDLQ_SCOPE": "nvfp4", "PRISMAQUANT_CB_ENCODE_TIER": "balanced"}, where="test")
        assert False
    except ValueError:
        pass


def test_provenance_validators_reject_malformed():
    """Real validators must reject malformed provenance fail-closed."""
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_serialization_context_stamp, validate_cb_cost_provenance, validate_cb_serialization_context_stamp
    from prismaquant.production_weight_cache import validate_cb_render_provenance

    ctx = CBSerializationContext.production(ldlq_scope="nvfp4")
    good_stamp = cb_serialization_context_stamp(ctx, formats=["NVFP4_CB_K12", "FP8_CB_K28"])
    good_payload = {"provenance": {"cb_serialized_payload": good_stamp, "cb_render_identity": {"cb_serialized_payload": good_stamp, "col_weights_shapes": {}, "col_weights_content_sha256": {}, "source_weights_shapes": {}, "source_weights_content_sha256": {}, "cb_formats_by_qname": {}, "render_contract": "test"}}, "formats": ["NVFP4_CB_K12", "FP8_CB_K28"]}
    # Good should pass cost provenance
    validate_cb_cost_provenance(good_payload, ["NVFP4_CB_K12"], context=ctx, where="test good")
    validate_cb_serialization_context_stamp(good_stamp, ctx, where="test good stamp")
    # Malformed: wrong ldlq_scope
    bad_stamp = dict(good_stamp)
    bad_stamp["ldlq_scope"] = "all"
    bad_payload = {"provenance": {"cb_serialized_payload": bad_stamp}, "formats": ["NVFP4_CB_K12"]}
    try:
        validate_cb_serialization_context_stamp(bad_stamp, ctx, where="test bad")
        assert False, "should have rejected scope mismatch"
    except ValueError:
        pass
    try:
        validate_cb_cost_provenance(bad_payload, ["NVFP4_CB_K12"], context=ctx, where="test bad cost")
        assert False
    except ValueError:
        pass
    # Render provenance with mismatched top vs identity should fail
    mismatched = {"provenance": {"cb_serialized_payload": good_stamp, "cb_render_identity": {"cb_serialized_payload": bad_stamp, "col_weights_shapes": {}, "col_weights_content_sha256": {}, "source_weights_shapes": {}, "source_weights_content_sha256": {}, "cb_formats_by_qname": {}, "render_contract": "test"}}, "formats": ["NVFP4_CB_K12"]}
    try:
        validate_cb_render_provenance(mismatched, expected_context=ctx, where="test render mismatch")
        assert False
    except (ValueError, AssertionError):
        pass


def test_campaign_mixing_fail_closed_by_default():
    """--run-root without opt-in while defaults pinned must fail closed; with opt-in it records."""
    import subprocess
    import sys
    # Use a temp dir as fake run-root; the derive tool will detect mixing before needing CUDA.
    r = subprocess.run([sys.executable, "-m", "tools.derive_dual_basis_packed", "--run-root", "/tmp/fake-mix-root", "--derived-root", "/tmp/fake-mix-derived"], capture_output=True, text=True, timeout=10)
    # Should exit non-zero due to mixing guard (SystemExit)
    assert r.returncode != 0, f"expected fail-closed, got {r.returncode} stdout={r.stdout} stderr={r.stderr}"
    assert "mix" in r.stderr.lower() or "mix" in r.stdout.lower()
    # With explicit opt-in it should pass the guard (no mixing error, falls through to no-op print)
    r2 = subprocess.run([sys.executable, "-m", "tools.derive_dual_basis_packed", "--run-root", "/tmp/fake-mix-root", "--derived-root", "/tmp/fake-mix-derived", "--allow-mixed-campaign-paths"], capture_output=True, text=True, timeout=10)
    assert r2.returncode == 0, f"opt-in should allow mixing, got {r2.returncode} stderr={r2.stderr} stdout={r2.stdout}"
    assert "full derive not executed" in r2.stdout.lower()
    # Env opt-in also works
    import os
    env = {**os.environ, "PQ_DERIVE_ALLOW_MIXED_CAMPAIGN": "1"}
    r3 = subprocess.run([sys.executable, "-m", "tools.derive_dual_basis_packed", "--run-root", "/tmp/fake-mix-root", "--derived-root", "/tmp/fake-mix-derived"], capture_output=True, text=True, timeout=10, env=env)
    assert r3.returncode == 0, f"env opt-in should allow mixing, got {r3.returncode} {r3.stderr}"
    # Canonical defaults without --run-root must still be usable (no mixing error)
    r4 = subprocess.run([sys.executable, "-m", "tools.derive_dual_basis_packed", "--derived-root", "/tmp/fake-mix-derived2"], capture_output=True, text=True, timeout=10)
    assert r4.returncode == 0, f"canonical defaults should be usable, got {r4.returncode} {r4.stderr}"
