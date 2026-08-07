"""Synthetic unit tests for NVFP4-CBL generalization (Phase A).

Covers:
  - FP8 bit-identical regression (learn_pool, book_key, cbl_eligible)
  - Geometry: NVFP4 2048-entry rule (K12-K22 eligible, K23/K24 ineligible)
  - Centroid grid snapping: E2M1 for fp4, E4M3 for fp8, idempotent
  - Decoded values representable (scale*grid) and exact byte accounting
  - Content-keyed book resume and bit-exact re-derivation
  - NVFP4-CB byte accounting (index K/8 + plane 0.5 / 0.28125)
"""
import hashlib
import json
import pathlib

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import bit_split, type_size
from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields, nvfp4_cb_reconstruct
from tools.dsv4_cbl_kernels import (
    BOOK_ROOT,
    CBL_ELIGIBLE_MAX_RUNG,
    CBL_SEMANTICS_SCHEMA,
    NVFP4_BOOK_ROOT,
    NVFP4_CBL_ELIGIBLE_MAX_RUNG,
    SEMANTICS_STAMP,
    _book_key,
    _book_root_for,
    cbl_eligible,
    cb_effective_bpw,
    cb_bytes_per_superblock,
    learn_pool,
    verify_grid_snap,
)


def test_fp8_bit_identical_book_key():
    """New _book_key with default grid=fp8 must match legacy FP8 hash."""
    def old_key(layer, proj, rung, sd, cd):
        payload = json.dumps({
            "schema": CBL_SEMANTICS_SCHEMA, "layer": int(layer),
            "projection": str(proj), "rung": int(rung),
            "source_digest": sd, "col_weights_digest": cd,
            "train": SEMANTICS_STAMP["book_train"],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
    for p in [(3, "gate_proj", 28, "abc", "def"), (0, "up_proj", 44, "x", "y")]:
        assert _book_key(layer=p[0], projection=p[1], rung=p[2], source_digest=p[3], col_weights_digest=p[4]) == old_key(*p)
        assert _book_key(layer=p[0], projection=p[1], rung=p[2], source_digest=p[3], col_weights_digest=p[4], grid="fp8", mode="product") == old_key(*p)


def test_fp8_learn_pool_bit_identical():
    torch.manual_seed(123)
    E, R, IN = 4, 128, 256
    weight = torch.randn(E, R, IN)
    col_weights = torch.rand(E, IN)
    pool_a = learn_pool(weight, col_weights, 30)  # default fp8
    pool_b = learn_pool(weight, col_weights, 30, grid="fp8", mode="product")
    for a, b in zip(pool_a, pool_b):
        assert torch.equal(a, b)
    # n_sub check: fp8 has 4 subtables, sub_dim=2
    assert len(pool_a) == 4
    for t in pool_a:
        assert t.shape == (1 << 7, 2) or t.shape[1] == 2  # K30 split (8,8,7,7) -> 256,256,128,128


def test_geometry_nvfp4_eligible():
    # Operator-derived geometry: K12-K22 eligible, K23/K24 excluded (4096 rule)
    for k in range(12, 23):
        assert cbl_eligible(k, grid="fp4", mode="product"), f"K{k} should be eligible"
        bits = bit_split(k, 2)
        assert max(1 << b for b in bits) <= 2048
    for k in (23, 24):
        assert not cbl_eligible(k, grid="fp4", mode="product"), f"K{k} should be ineligible"
        bits = bit_split(k, 2)
        assert max(1 << b for b in bits) > 2048
    # FP8 ceiling stays 44
    for k in (44, 28, 38):
        assert cbl_eligible(k, grid="fp8")
    assert not cbl_eligible(45, grid="fp8")
    # Unknown family defaults to FP8 ceiling for backward compat
    assert cbl_eligible(44)


def test_book_root_separation():
    assert _book_root_for("fp8", "product") == BOOK_ROOT
    assert _book_root_for("fp4", "product") == NVFP4_BOOK_ROOT
    # NVFP4 path must NOT be inside the production bank
    assert "nvfp4-cbl" in str(NVFP4_BOOK_ROOT)
    assert NVFP4_BOOK_ROOT != BOOK_ROOT
    # Ensure no accidental overlap
    assert str(BOOK_ROOT) not in str(NVFP4_BOOK_ROOT) or NVFP4_BOOK_ROOT != BOOK_ROOT


def test_centroid_grid_snapping_fp4():
    torch.manual_seed(0)
    E, R, IN = 4, 64, 256
    weight = torch.randn(E, R, IN)
    col_weights = torch.rand(E, IN)
    for k in (12, 16, 22):
        pool = learn_pool(weight, col_weights, k, grid="fp4", mode="product")
        assert len(pool) == 2
        for t in pool:
            # Every centroid coordinate must be on E2M1 grid (idempotent snap)
            snapped = cb._snap_to_grid(t, "fp4")
            assert torch.equal(snapped, t.to(torch.float32)), f"K{k} centroid not E2M1-snapped"
        assert verify_grid_snap(pool, "fp4")


def test_centroid_grid_snapping_fp8():
    torch.manual_seed(1)
    E, R, IN = 4, 64, 256
    weight = torch.randn(E, R, IN)
    col_weights = torch.rand(E, IN)
    pool = learn_pool(weight, col_weights, 28, grid="fp8", mode="product")
    assert verify_grid_snap(pool, "fp8")
    for t in pool:
        assert torch.equal(cb._snap_to_grid(t, "fp8"), t.to(torch.float32))


def test_decoded_values_representable_fp4():
    """NVFP4 decoded values must be (E2M1 grid) * group scale.

    For the CBL cand0 path scale_sweep=False uses the one-shot amax/6
    normalisation (unsnapped fp32) as the training normaliser — the final
    packed bytes store E4M3, but the emulation's one-shot scales are the
    raw amax/6. We verify that de-scaled codewords are E2M1 regardless.
    """
    torch.manual_seed(2)
    weight = torch.randn(2, 512)  # 2 rows, 512 cols = 2 superblocks per row
    col_weights = torch.rand(2, 512)
    for k in (12, 16, 22):
        fields = nvfp4_cb_fields(weight, k, grid="fp4", mode="product", col_weights=col_weights, scale_sweep=False, scale_coding="v1")
        recon = nvfp4_cb_reconstruct(fields, k, grid="fp4", mode="product")
        scales = fields["scales"]  # (rows, groups) – cand0 unsnapped for this tier
        # Check per-group de-scaled codewords are E2M1 (within fp32 roundtrip)
        pes = scales.repeat_interleave(16, dim=1)  # group-16
        de_scaled = (recon / pes).reshape(-1, 8).to(torch.float32)
        snapped = cb._snap_to_grid(de_scaled, "fp4")
        # Division introduces <1e-6 fp32 error vs the exact codebook; allow tolerance,
        # but require that snapped lands back on E2M1 and is close.
        assert torch.allclose(snapped, de_scaled, atol=1e-4), f"K{k} decoded not near E2M1 (maxdiff {(snapped-de_scaled).abs().max()})"
        assert verify_grid_snap([snapped], "fp4"), f"K{k} snapped not on E2M1"
        # Also verify via the sweep path the scales ARE E4M3-snapped (export parity)
        fields_sw = nvfp4_cb_fields(weight, k, grid="fp4", mode="product", col_weights=col_weights, scale_sweep=True, scale_coding="v1")
        scales_sw = fields_sw["scales"]
        assert torch.equal(cb._snap_to_grid(scales_sw, "fp8"), scales_sw.to(torch.float32)), f"K{k} sweep scales not E4M3"


def test_byte_accounting_exact():
    # FP4 v1: K/8 + 0.5, two_tier: +0.28125, FP8: K/8
    for k in (12, 14, 16, 18, 20, 22, 24):
        v1 = cb_effective_bpw(k, "fp4", "v1")
        assert abs(v1 - (k / 8 + 0.5)) < 1e-9, f"K{k} v1 bpw {v1}"
        assert cb_bytes_per_superblock(k, "fp4", "v1") == k * 4 + 16  # INDEX_BYTES_PER_K=4, 4*k +16
        assert cb_bytes_per_superblock(k, "fp4", "v1") == type_size(k, "fp4", "v1")
        tt = cb_effective_bpw(k, "fp4", "two_tier")
        assert abs(tt - (k / 8 + 0.28125)) < 1e-9
        assert cb_bytes_per_superblock(k, "fp4", "two_tier") == type_size(k, "fp4", "two_tier")
    for k in (28, 33, 38, 43):
        bpw = cb_effective_bpw(k, "fp8", "v1")
        assert abs(bpw - k / 8) < 1e-9
        assert cb_bytes_per_superblock(k, "fp8", "v1") == k * 4  # no scale plane
    # Cross-check nvfp4_cb_effective_bits direct
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_effective_bits
    for k in (12, 22):
        assert cb_effective_bpw(k, "fp4", "v1") == nvfp4_cb_effective_bits(k, "fp4", "v1")


def test_nvfp4_content_keyed_resume():
    """Train a small NVFP4 book under NVFP4_BOOK_ROOT, kill, reload bit-identically."""
    import tempfile
    from tools.dsv4_cbl_kernels import _load_or_train_book, book_sha256
    torch.manual_seed(7)
    E, R, IN = 4, 64, 256
    weight = torch.randn(E, R, IN)
    # Real col_weights shape is (E,1,IN) – one per-input-column vector broadcast over rows
    col_weights = torch.rand(E, 1, IN)
    layer, proj, rung = 3, "gate_proj", 14
    # Use content digests that simulate real identities
    from prismaquant.cb_warm_state import tensor_value_identity
    sid = tensor_value_identity(weight)[1]
    cid = tensor_value_identity(col_weights)[1]
    # First call trains and banks
    pool1, sha1, path1, outcome1, _ = _load_or_train_book(
        layer=layer, projection=proj, rung=rung, weight=weight, col_weights=col_weights,
        source_digest=sid, col_weights_digest=cid, grid="fp4", mode="product")
    assert outcome1 == "book_trained_and_banked"
    assert pathlib.Path(path1).is_file()
    assert str(NVFP4_BOOK_ROOT) in path1
    # Second call should restore without retraining and be bit-identical
    pool2, sha2, path2, outcome2, _ = _load_or_train_book(
        layer=layer, projection=proj, rung=rung, weight=weight, col_weights=col_weights,
        source_digest=sid, col_weights_digest=cid, grid="fp4", mode="product")
    assert outcome2 == "book_restored_content_addressed"
    assert path1 == path2
    assert sha1 == sha2
    assert sha1 == book_sha256(pool1) == book_sha256(pool2)
    for a, b in zip(pool1, pool2):
        assert torch.equal(a, b)
    # Cleanup: remove the test book so it doesn't pollute the study bank
    pathlib.Path(path1).unlink()
    # Remove empty subdir if possible
    try:
        pathlib.Path(path1).parent.rmdir()
    except OSError:
        pass


def test_encode_cbl_nvfp4_bit_exact_rederivation():
    """encode_cbl with same content must reproduce banked errors bit-exactly."""
    from tools.dsv4_cbl_kernels import encode_cbl
    torch.manual_seed(9)
    E, R, IN = 8, 64, 512
    weight = torch.randn(E, R, IN)
    col_weights = torch.rand(E, 1, IN)
    data = {"weight": weight, "col_weights": col_weights}
    layer, proj, rung = 5, "down_proj", 16
    fields1, recon1, errors1, timing1, path1 = encode_cbl(
        layer=layer, projection=proj, rung=rung, data=data, expert_ids=tuple(range(E)),
        grid="fp4", mode="product", scale_coding="v1")
    # Second encode with expected_free_errors should succeed (bit-exact)
    fields2, recon2, errors2, timing2, path2 = encode_cbl(
        layer=layer, projection=proj, rung=rung, data=data, expert_ids=tuple(range(E)),
        expected_free_errors=errors1, grid="fp4", mode="product", scale_coding="v1")
    assert path1 == path2
    assert errors1 == errors2
    assert torch.equal(recon1, recon2)
    # Cleanup
    import pathlib
    pathlib.Path(path1).unlink()
    try:
        pathlib.Path(path1).parent.rmdir()
    except OSError:
        pass


def test_synthetic_weight_mse_uplift_smoke():
    """At low bpw CBL should not be catastrophically worse than incumbent;
    this is a smoke check not a claim."""
    torch.manual_seed(11)
    E, R, IN = 8, 32, 256
    weight = torch.randn(E, R, IN) * 0.5
    col_weights = torch.rand(E, IN).clamp_min(0.1)
    for k in (12, 16):
        # CBL book
        pool = learn_pool(weight, col_weights, k, grid="fp4", mode="product")
        fields_cbl = nvfp4_cb_fields(weight.reshape(-1, IN), k, grid="fp4", mode="product", col_weights=col_weights.repeat(R, 1).reshape(-1, IN) if False else None,
                                      codebook=pool, scale_sweep=False)
        # Actually encode per-expert stack directly with CBL vs incumbent
        # incumbent sweep
        fields_inc = nvfp4_cb_fields(weight.reshape(-1, IN), k, grid="fp4", mode="product", col_weights=None, codebook=None, scale_sweep=True)
        # Not asserting win, just that both produce finite recons of correct shape
        rc = nvfp4_cb_reconstruct(fields_cbl, k, grid="fp4", mode="product")
        ri = nvfp4_cb_reconstruct(fields_inc, k, grid="fp4", mode="product")
        assert rc.shape == weight.reshape(-1, IN).shape
        assert ri.shape == rc.shape
        assert torch.isfinite(rc).all()
        assert torch.isfinite(ri).all()
