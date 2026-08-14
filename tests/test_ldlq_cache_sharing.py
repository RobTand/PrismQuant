"""Verify Hessian factor cache sharing: same activation objects hit cache and cached vs uncached fields bit-identical."""

import torch
from prismaquant.nvfp4_cb_formats import (
    _LDLQ_FACTOR_CACHE,
    _LDLQ_FACTOR_CACHE_LOCK,
    _ldlq_inverse_factor_cached,
    nvfp4_cb_fields,
    ldlq_reassign_cb_fields,
)


def test_same_activation_object_hits_cache():
    # Clear cache for isolation
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE.clear()
    torch.manual_seed(0)
    act = torch.randn(8, 16)
    device = torch.device("cpu")
    f1 = _ldlq_inverse_factor_cached(act, device=device, damping_fraction=0.01)
    # Cache should contain one entry keyed by id(act)
    with _LDLQ_FACTOR_CACHE_LOCK:
        assert len(_LDLQ_FACTOR_CACHE) == 1
        # Ensure key corresponds to this object
        keys = list(_LDLQ_FACTOR_CACHE.keys())
        assert keys[0][0] == id(act)
    f2 = _ldlq_inverse_factor_cached(act, device=device, damping_fraction=0.01)
    # Second call must return same tensor object (cache hit) and be equal
    assert f2 is f1 or torch.equal(f1, f2)
    with _LDLQ_FACTOR_CACHE_LOCK:
        assert len(_LDLQ_FACTOR_CACHE) == 1


def test_cached_vs_uncached_fields_bit_identical():
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE.clear()
    torch.manual_seed(1)
    # Weight: 2x256, group 16, superblock 256 -> compatible
    w = torch.randn(2, 256)
    cw = torch.ones(1, 256)
    act = torch.randn(8, 256)
    fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
    device = torch.device("cpu")
    # Uncached path: clear cache then compute
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE.clear()
    fields_uncached = ldlq_reassign_cb_fields(w, dict(fields), cw, act, grid="fp4", mode="product")
    # Cached path: prewarm then compute with same act object (should hit)
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE.clear()
    _ldlq_inverse_factor_cached(act, device=w.device, damping_fraction=0.01)
    with _LDLQ_FACTOR_CACHE_LOCK:
        assert len(_LDLQ_FACTOR_CACHE) == 1
    fields_cached = ldlq_reassign_cb_fields(w, dict(fields), cw, act, grid="fp4", mode="product")
    assert torch.equal(fields_uncached["indices"], fields_cached["indices"])
    assert torch.equal(fields_uncached["scales"], fields_cached["scales"])
    if "signs" in fields_uncached:
        assert torch.equal(fields_uncached["signs"], fields_cached["signs"])


def test_malformed_activation_aborts():
    # Empty activation must raise, not be swallowed
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE.clear()
    act_empty = torch.empty((0, 16))
    device = torch.device("cpu")
    try:
        _ldlq_inverse_factor_cached(act_empty, device=device, damping_fraction=0.01)
        assert False, "should have raised on empty activation"
    except Exception as exc:
        # Should be ValueError from inverse_hessian_cholesky or shape check
        assert isinstance(exc, (ValueError, RuntimeError))
    # Wrong width
    act_wrong = torch.randn(8, 8)
    try:
        _ldlq_inverse_factor_cached(act_wrong, device=device, damping_fraction=0.01)
        # This may not raise at factor level (depends on implementation), but LDLQ assignment should
        w = torch.randn(2, 256)
        cw = torch.ones(1, 256)
        fields = nvfp4_cb_fields(w, 12, grid="fp4", mode="product", col_weights=cw)
        ldlq_reassign_cb_fields(w, fields, cw, act_wrong, grid="fp4", mode="product")
        assert False, "should have raised on width mismatch"
    except ValueError:
        pass
