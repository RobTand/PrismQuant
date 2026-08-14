"""Test nested provenance: cost and render identities grounded in real schema, preserving source/col-weight bindings."""

import torch
from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_serialization_context_stamp, validate_cb_cost_provenance
from prismaquant.production_weight_cache import (
    build_production_cache_cb_render_identity,
    bind_cb_render_identity_source_weights,
    validate_cb_render_provenance,
)


def _build_valid_payload():
    ctx = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
    fmt_nvfp4 = "NVFP4_CB_K12"
    fmt_fp8 = "FP8_CB_K28"
    qnames = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts.0.gate_proj"]
    col_weights = {q: torch.ones(256) for q in qnames}
    # Build render identity with col_weights
    identity = build_production_cache_cb_render_identity(
        {q: [fmt_nvfp4, fmt_fp8] for q in qnames},
        cb_serialization_context=ctx,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    # Bind source weights (preserve source binding)
    source_weights = {q: torch.randn(2, 256) for q in qnames}
    identity = bind_cb_render_identity_source_weights(identity, source_weights)
    stamp = cb_serialization_context_stamp(ctx, formats=[fmt_nvfp4, fmt_fp8, "BF16"])
    # Cost payload with both nested identities (compatible)
    payload = {
        "costs": {q: {fmt_nvfp4: {"weight_mse": 0.1}, fmt_fp8: {"weight_mse": 0.01}} for q in qnames},
        "formats": [fmt_nvfp4, fmt_fp8, "BF16"],
        "provenance": {
            "cb_serialized_payload": stamp,
            "cb_render_identity": identity,
        },
        "meta": {},
    }
    return ctx, payload, col_weights, qnames


def test_cost_and_render_provenance_both_pass():
    ctx, payload, col_weights, qnames = _build_valid_payload()
    # Cost provenance should pass (checks stamp vs context)
    validate_cb_cost_provenance(payload, payload["formats"], context=ctx, where="test cost")
    # Render provenance should pass and preserve bindings (checks nested identity)
    _ctx2, identity = validate_cb_render_provenance(payload, expected_context=ctx, col_weights=col_weights, where="test render")
    # Verify source and col-weight bindings preserved (deep equality)
    for q in qnames:
        assert q in identity["col_weights_shapes"]
        assert q in identity["source_weights_shapes"]


def test_render_provenance_top_level_must_match_nested():
    ctx, payload, col_weights, _ = _build_valid_payload()
    # Mutate top-level stamp to mismatch nested (should fail)
    bad_stamp = dict(payload["provenance"]["cb_serialized_payload"])
    bad_stamp["ldlq_scope"] = "all"  # mismatch vs nvfp4
    payload_bad = dict(payload)
    payload_bad["provenance"] = dict(payload["provenance"])
    payload_bad["provenance"]["cb_serialized_payload"] = bad_stamp
    try:
        validate_cb_render_provenance(payload_bad, expected_context=ctx, col_weights=col_weights, where="test mismatch")
        assert False, "should have raised on top-level/nested mismatch"
    except ValueError as exc:
        assert "top-level" in str(exc) or "scope" in str(exc).lower()


def test_cost_provenance_fails_on_wrong_lattice_digest():
    ctx, payload, _, _ = _build_valid_payload()
    # Corrupt lattice digest
    bad = dict(payload)
    bad["provenance"] = dict(payload["provenance"])
    bad_stamp = dict(payload["provenance"]["cb_serialized_payload"])
    # lattice digest is under lattice_codebook_sha256_by_format
    if "lattice_codebook_sha256_by_format" in bad_stamp:
        fmt = list(bad_stamp["lattice_codebook_sha256_by_format"].keys())[0]
        bad_stamp["lattice_codebook_sha256_by_format"] = dict(bad_stamp["lattice_codebook_sha256_by_format"])
        bad_stamp["lattice_codebook_sha256_by_format"][fmt] = ["0"*64]
    bad["provenance"]["cb_serialized_payload"] = bad_stamp
    try:
        validate_cb_cost_provenance(bad, payload["formats"], context=ctx, where="test bad lattice")
        assert False, "should have raised on lattice mismatch"
    except ValueError as exc:
        assert "lattice" in str(exc).lower()


def test_col_weight_binding_mismatch_fails():
    ctx, payload, _, _ = _build_valid_payload()
    # Provide wrong col_weights
    wrong_cw = {q: torch.ones(128) for q in payload["costs"]}
    try:
        validate_cb_render_provenance(payload, expected_context=ctx, col_weights=wrong_cw, where="test cw mismatch")
        assert False, "should have raised on col-weight shape mismatch"
    except ValueError as exc:
        assert "col" in str(exc).lower() or "shape" in str(exc).lower()
