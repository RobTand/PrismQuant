"""WO-B B4: Trellis render mechanism — one encode, byte-identical wire.

The six tests mirror the WO-B deliverables exactly. They are principle 8 in
code: the bytes the exporter ships are the bytes whose decode the surrogate
priced and KL measured.
"""
from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn

from prismaquant.trellis_serialization import TrellisSerializationContext
from prismaquant.trellis_wire import decode_values_torch

pytest.importorskip("torch")

_ALPHABET = {2: (15, 13, 11, 9, 8, 2, 4, 7)}


def _make_trellis_ctx(
    *,
    columns: int = 256,
    backend: str = "eager",
    body_rate_q256: int = 512,
    layout: str = "fixed_quota_per_256",
) -> TrellisSerializationContext:
    return TrellisSerializationContext(
        family="TCQ_E2M1_R256",
        body_rate_q256=body_rate_q256,
        schedule=[2] * columns,
        layout=layout,
        alphabets=_ALPHABET,
        scale_rule="static_6",
        sb_chunk=64,
        determinism_mode="on",
        tailbite_candidates=4,
        backend=backend,
        point_route="full",
    )


def _weight_and_col(columns: int, rows: int, seed: int = 0, device: str = "cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    weight = torch.randn(rows, columns, generator=gen, device=device, dtype=torch.bfloat16)
    # col_weights must be positive, finite, with mass
    col_weights = torch.rand(columns, generator=gen, device=device) + 0.05
    return weight, col_weights


# ---------------------------------------------------------------------------
# 1. Rendering identity (load-bearing test)
# ---------------------------------------------------------------------------

def test_trellis_rendering_identity_bit_exact_via_retained_wire():
    """For at least three shapes including a non-multiple-of-256 row count:
    render once through ``render_production_weight``, retrieve the retained
    wire bytes from the cache, decode via ``trellis_wire.decode_values_torch``,
    and assert bit-exact equality with the returned tensor.
    """
    from prismaquant.production_weight_cache import fill_production_weight_cache

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            # 8, 7, 5 rows -> includes non-multiple-of-256
            self.a = nn.Linear(256, 8, bias=False)
            self.b = nn.Linear(256, 7, bias=False)
            self.c = nn.Linear(256, 5, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = torch.randn(input_ids.shape[0] * input_ids.shape[1], 256, device=self.a.weight.device)
            _ = self.a(x)
            _ = self.b(x)
            _ = self.c(x)
            return x

    ctx = _make_trellis_ctx(columns=256)

    # Use three different row counts, all with same column count so one context suffices
    col_weights = {
        "a": torch.rand(256) + 0.05,
        "b": torch.rand(256) + 0.05,
        "c": torch.rand(256) + 0.05,
    }
    model = Tiny()
    calib_ids = torch.randint(0, 10, (2, 4))
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["a", "b", "c"],
        formats=["TCQ_E2M1_R512"],
        levers={},
        col_weights=col_weights,
        trellis_serialization_context=ctx,
        progress=False,
    )

    for qname in ["a", "b", "c"]:
        tensor = cache.get(qname, "TCQ_E2M1_R512")
        assert tensor is not None, f"cache missing {qname}"
        wire = cache.get_trellis_wire_bytes(qname, "TCQ_E2M1_R512")
        assert wire is not None and len(wire) > 0, f"no wire retained for {qname}"
        decoded = decode_values_torch(wire, device=tensor.device, dtype=tensor.dtype)
        # Bit-exact at BF16 (the pipeline's stored dtype)
        assert torch.equal(tensor.to(torch.bfloat16), decoded.to(torch.bfloat16)), (
            f"{qname}: decoded wire differs from returned tensor at BF16 — "
            "principle 8 violated: the bytes priced are not the bytes shipped"
        )
        # Also check direct render path produces same identity
        # (one-encode invariant: direct render must also be wire-derived)
        from prismaquant.production_weight_cache import render_production_weight

        weight = model.get_submodule(qname).weight.detach()
        cw = col_weights[qname]
        direct = render_production_weight(
            weight,
            "TCQ_E2M1_R512",
            qname=qname,
            activations={},
            levers={},
            col_weights=cw,
            trellis_serialization_context=ctx,
        )
        # Direct render's wire should also decode to same as cache's wire's decode?
        # At minimum direct tensor should equal cache tensor at BF16 (same recipe)
        assert torch.equal(direct.to(torch.bfloat16), tensor.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

def test_trellis_determinism_byte_identical():
    from prismaquant.production_weight_cache import render_production_weight, _peek_last_trellis_artifact

    ctx = _make_trellis_ctx(columns=256, backend="eager")
    weight, col_weights = _weight_and_col(256, 16, seed=42)
    first = render_production_weight(
        weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_weights, trellis_serialization_context=ctx
    )
    art1 = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    second = render_production_weight(
        weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_weights, trellis_serialization_context=ctx
    )
    art2 = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    assert art1 is not None and art2 is not None
    assert art1.wire_bytes == art2.wire_bytes, "deterministic encode produced different wire bytes"
    assert art1.receipt["wire_identity_sha256"] == art2.receipt["wire_identity_sha256"]
    assert torch.equal(first.to(torch.bfloat16), second.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# 3. col_weights is value-bearing
# ---------------------------------------------------------------------------

def test_trellis_col_weights_is_value_bearing():
    from prismaquant.production_weight_cache import render_production_weight, _peek_last_trellis_artifact

    ctx = _make_trellis_ctx(columns=256)
    weight, cw1 = _weight_and_col(256, 8, seed=1)
    # cw2 is a different positive vector with same shape
    torch.manual_seed(99)
    cw2 = torch.rand(256) * 10.0 + 0.01

    render_production_weight(
        weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=cw1, trellis_serialization_context=ctx
    )
    art1 = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    render_production_weight(
        weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=cw2, trellis_serialization_context=ctx
    )
    art2 = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    assert art1.wire_bytes != art2.wire_bytes, (
        "different col_weights produced identical wire — weighted objective is being ignored"
    )
    # Also ensure uniform vs weighted differs (guard against vacuous pass)
    cw_uniform = torch.ones(256)
    render_production_weight(
        weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=cw_uniform, trellis_serialization_context=ctx
    )
    art_uniform = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    assert art1.wire_bytes != art_uniform.wire_bytes


# ---------------------------------------------------------------------------
# 4. Missing context refuses
# ---------------------------------------------------------------------------

def test_trellis_missing_context_refuses():
    from prismaquant.production_weight_cache import render_production_weight

    weight, col_weights = _weight_and_col(256, 8, seed=2)
    with pytest.raises(ValueError, match="TrellisSerializationContext"):
        render_production_weight(
            weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_weights, trellis_serialization_context=None
        )
    # Error must name the missing recipe fields
    try:
        render_production_weight(
            weight, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_weights, trellis_serialization_context=None
        )
    except ValueError as exc:
        msg = str(exc)
        assert "family" in msg
        assert "body_rate_q256" in msg
        assert "schedule" in msg
        assert "alphabets" in msg
        assert "scale_rule" in msg
        assert "sb_chunk" in msg
        assert "determinism_mode" in msg
        assert "tailbite_candidates" in msg
        assert "backend" in msg
        assert "point_route" in msg


# ---------------------------------------------------------------------------
# 5. Eager/Triton agreement
# ---------------------------------------------------------------------------

def test_trellis_eager_triton_agreement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — Triton path requires GB10")
    # Triton requires triton package
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton not installed")

    from prismaquant.production_weight_cache import render_production_weight, _peek_last_trellis_artifact

    ctx_eager = _make_trellis_ctx(columns=256, backend="eager")
    ctx_triton = _make_trellis_ctx(columns=256, backend="triton")

    weight_cpu, col_cpu = _weight_and_col(256, 8, seed=3, device="cpu")
    weight_cuda = weight_cpu.cuda()
    col_cuda = col_cpu.cuda()

    # Eager on CUDA (reference) — eager works on any device
    render_production_weight(
        weight_cuda, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_cuda, trellis_serialization_context=ctx_eager
    )
    art_eager = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    wire_eager = art_eager.wire_bytes

    render_production_weight(
        weight_cuda, "TCQ_E2M1_R512", qname="lin", activations={}, levers={}, col_weights=col_cuda, trellis_serialization_context=ctx_triton
    )
    art_triton = _peek_last_trellis_artifact("lin", "TCQ_E2M1_R512")
    wire_triton = art_triton.wire_bytes

    assert wire_eager == wire_triton, "CUDA Triton path and CPU eager path produced different wire bytes"
    # Also check decoded tensors equal at BF16
    assert torch.equal(art_eager.decoded_weight.to(torch.bfloat16), art_triton.decoded_weight.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# 6. Stale-ABI cache rebuilds loudly
# ---------------------------------------------------------------------------

def test_trellis_stale_abi_cache_rebuilds_loudly(tmp_path):
    """An old cache carrying the previous trellis ABI (or no trellis identity)
    must not silently serve a trellis-shaped miss; it must rebuild loudly.
    """
    from prismaquant.production_weight_cache import ProductionWeightCache, TRELLIS_RENDER_IDENTITY_METADATA_KEY, TRELLIS_RENDER_MECHANISM_ABI

    # Build a cache with a valid trellis identity
    ctx = _make_trellis_ctx(columns=256)

    # Simulate an old cache that has a trellis wire but an old ABI string in its
    # render contract. Mutate the ABI and ensure validation refuses.
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(256, 8, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = torch.randn(input_ids.shape[0] * input_ids.shape[1], 256, device=self.l1.weight.device)
            return self.l1(x)

    from prismaquant.production_weight_cache import fill_production_weight_cache

    col_weights = {"l1": torch.rand(256) + 0.05}
    calib_ids = torch.randint(0, 10, (2, 4))
    tiny = Tiny()
    cache = fill_production_weight_cache(
        tiny, calib_ids, qnames=["l1"], formats=["TCQ_E2M1_R512"], levers={}, col_weights=col_weights, trellis_serialization_context=ctx, progress=False
    )

    # Cache should be valid now
    assert cache.get("l1", "TCQ_E2M1_R512") is not None

    # Corrupt the ABI to simulate a stale cache
    corrupted = cache.metadata[TRELLIS_RENDER_IDENTITY_METADATA_KEY].copy()
    corrupted["render_contract"] = dict(corrupted["render_contract"])
    corrupted["render_contract"]["mechanism_abi"] = "prismaquant.trellis_render_mechanisms.v0"
    cache.metadata[TRELLIS_RENDER_IDENTITY_METADATA_KEY] = corrupted

    with pytest.raises(ValueError, match="unsupported trellis render mechanism ABI"):
        cache.get("l1", "TCQ_E2M1_R512")

    # Also test: old cache with no trellis identity at all but with a trellis wire entry
    # must be refused as legacy/missing (this is the stale-WORK_DIR rebuild case).
    legacy = ProductionWeightCache(
        weights={("l1", "TCQ_E2M1_R512"): torch.randn(8, 256)},
        levers={},
        metadata={},
        trellis_wires={("l1", "TCQ_E2M1_R512"): b"fake"},
        trellis_wire_identities={},
    )
    with pytest.raises(ValueError, match="trellis cache is missing versioned"):
        legacy.get("l1", "TCQ_E2M1_R512")

    # And a pickled cache from before WO-B (no trellis_wires attr) should also
    # refuse when unpickled and queried for a trellis format. Simulate by
    # constructing a cache with no trellis identity and then requesting trellis.
    import pickle
    blob = pickle.dumps(cache)
    reloaded = pickle.loads(blob)
    # Corrupt the reloaded cache's identity to old ABI as well and ensure it still refuses
    reloaded.metadata[TRELLIS_RENDER_IDENTITY_METADATA_KEY]["render_contract"]["mechanism_abi"] = "old"
    with pytest.raises(ValueError, match="unsupported trellis render mechanism ABI"):
        reloaded.get("l1", "TCQ_E2M1_R512")
