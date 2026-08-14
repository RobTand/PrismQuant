"""NVFP4-CB / FP8-CB vector-quantization codebook formats (emulation).

A codeword is a d=8 vector of grid values (E2M1 for the ``fp4`` family, E4M3
for the ``fp8`` family). A k-bit index per 8 weights gives ``k/8`` index bpw;
the fp4 family adds NVFP4-identical group-16 E4M3 scales (+0.5 bpw), the fp8
family a per-output-channel fp32 scale (negligible bpw). A decoded fp4 tile is
bit-compatible NVFP4 by construction (grid values on the E2M1 grid, NVFP4 group
scale), so it feeds the CUTLASS FP4 path unchanged.

This module is Milestone A: emulation only. One weighted-VQ field quantizer
feeds the emulation ``reconstruct`` (what cost measurement scores); the byte
packer and exporter land in Milestone B and must share this exact math path.

Three VQ modes:

* ``full`` — one ``2^k`` codebook over the 8-dim vector, exhaustive weighted
  argmin (chunked). Only feasible for k<=14; raises above without an explicit
  codebook.
* ``product`` (default) — the 8-dim vector splits into two 4-dim halves, each
  with its own ``2^(k/2)`` sub-codebook (ceil/floor bit split for odd k). Feasible
  for the whole NVFP4-CB ladder (k=12..24).
* ``signed`` — sign-magnitude factorization (the IQ-family move): 8 explicit
  sign bits + an ``m = k-8``-bit index into a MAGNITUDE codebook over the
  positive half-grid. A flat codebook burns most of its entries covering sign
  patterns (~2^8 per magnitude shape at d=8, leaving ~2^(k-8) effective
  magnitude shapes); factoring the signs out spends all 2^m entries on
  magnitude shapes (exp-1 diagnosis: this is why IQ2_S beat flat CB +66% at
  matched bytes). Encode is exactly separable under weighted L2 (see
  ``_fields_block``); tables are tiny (m=5..8 -> 32..256 entries).

The weighted objective is llama.cpp/imatrix style:
``sum_j w_j (x_j - c_j)^2`` per codeword, with per-input-column ``col_weights``
(the same plumbing as the GGUF lane).

Scale search: CB encode sweeps the per-group (fp4) / per-row (fp8) scale over a
grid of E4M3-legal candidates and picks the one minimizing weighted
reconstruction error in the ORIGINAL weight domain, then refines with 2
WLS-refit fixed-point iterations — rendering parity with the IQ lane's
27-candidate ``_grid_fields`` sweep. ``scale_sweep=True`` is the default for all
three modes and both grids. The amax/6 (fp4) / amax/448 (fp8) one-shot scale is
always in the candidate set, so the sweep is never worse than one-shot. **The
Phase-0 exp-1 0.6B results (pre-c3f8c6d) used one-shot scales for CB while the
IQ arms got their scale sweep — that rendering asymmetry is corrected here;
re-run before trusting any CB-vs-IQ delta.**

Scale coding (fp4 family): the v1 plane stores each group-16 scale as a bare
E4M3 byte — on real LLM weights ~90% of group scales sit in e4m3's SUBNORMAL
band, where the sweep's candidates collapse to ~1-2 distinct values
(two-tier-scale-spec.md §3). Layout v2 ("two_tier", opt-in until the serving
gates clear) stores a per-superblock E8M0 super `2^(E-127)` plus 16 4-bit sub
codes into an e4m3-exact multiplier table; the composition IS an E4M3 scale by
construction, the encoder explores every reachable value across the ideal-scale
window (~20+ distinct candidates where v1 had 1-2), and the scale plane shrinks
16 B -> 9 B per superblock (0.5 -> 0.28125 bpw). Every fp4 exp-1/1b number was
measured under v1 coding.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import types
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .cb_layout import (
    CODEWORDS_PER_SUPERBLOCK,
    FP4_GROUP,
    FP4_SCALE_GROUPS_PER_SUPERBLOCK,
    INDEX_BYTES_PER_K,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SCALE_CODINGS,
    SCALE_PLANE_BYTES,
    SUPERBLOCK,
    VEC_DIM,
    codebook_subtable_shapes,
    family_for,
    subtable_bit_widths,
    type_size as _serialized_type_size,
)

FP8_ELEMENT_MAX = 448.0
NVFP4_GRID_MAX = 6.0            # max(|E2M1|); amax/6 == no-clip one-shot scale
# Flat-table feasibility ceiling (encode-side exhaustive argmin + serve-side
# LUT). Above this a structured/learned codebook must be supplied explicitly.
MAX_FLAT_K = 14
# Slice stacked/huge tensors along the leading dim to bound VQ temporaries
# (mirrors gguf_slice_max_elems' 64M IQ threshold — UMA swap-kill guard).
_SLICE_MAX_ELEMS = 64 * 1024 * 1024
# Row-chunk bound for the (rows*nvec, K) distance sweep.
_SCORE_CHUNK_ELEMS = 1 << 26

# Scale search (rendering parity with the IQ lane's _grid_fields sweep):
# number of clipping-level candidates and fixed-point refit iterations. The
# fp4 grid sweeps amax/L for L spanning [6, 4] (grid max 6; the JSO {6,4}
# insight as a grid); fp8 spans amax/L for L in [448, 448*4/6]. L=grid-max is
# candidate 0 == the amax/grid-max one-shot, so the sweep is never worse.
_SCALE_SWEEP_CANDIDATES = 16
_SCALE_SWEEP_REFIT_ITERS = 2
_E4M3 = torch.float8_e4m3fn
# Smallest positive fp8_e4m3fn value (subnormal 2^-9), an E4M3-exact floor
# that keeps a chosen block scale strictly positive (never underflow to 0).
_E4M3_MIN_POS = 2.0 ** -9

# --- Two-tier scale coding (layout v2, fp4 family only) -----------------
# docs/lanes/nvfp4-cb/two-tier-scale-spec.md: per-256 E8M0 super (2^(E-127))
# x per-16 4-bit sub code into a fixed table of e4m3-exact multipliers;
# the composition lands exactly on E4M3 by construction (legality mask, no
# rounding anywhere). Scale plane 16 B -> 9 B per superblock (0.28125 bpw).
TWO_TIER_SUPER_BIAS = 127
# T4_2oct8m (spec §1.3): all 8 e4m3 mantissa steps x 2 octaves.
TWO_TIER_SUB_TABLE = (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
# Window margin around [min_ideal/T_max, 1.5*max_ideal] (spec §1.4 derives the
# E window from the ideal group scales; the conservatism note says a wider
# window can only improve, so pad one octave on each side).
_TWO_TIER_WINDOW_PAD = 1
_TWO_TIER_MAX_WINDOW = 14

# --- Tiered encoder (docs/lanes/nvfp4-cb/encode_tiers.md) -----------------
# PRISMAQUANT_CB_ENCODE_TIER in {fast, balanced, max}:
#   max      — the original exhaustive sweep, bit-identical
#              (regression-pinned);
#   balanced — analytic scale init (usage-calibrated second-moment match,
#              s0 = sqrt(sum q w^2 / (sum q * m2_used)); m2_used measured
#              from a pilot encode of the tensor's leading rows) + a
#              log-spaced micro-sweep of +-2 neighbors + the amax/grid-max
#              guarantee candidate, scored from per-(vector, entry) moments
#              (err = C - 2sB + s^2A; A, B scale-independent, built once
#              per chunk — the _grid_fields/_sweep_errs identity), then the
#              exact 2 WLS refits;
#   fast     — same, +-1 neighbors + 1 refit.
# Measured basis (encode_cost_4b.json + encode_tiers.md): the 4B FP8_CB
# winners spread across high-clip candidates (no JSO-style collapse; pruning
# alone <= 2.6x) and refits accept at 99.5%/95.2% (>=1 refit everywhere);
# the naive all-codeword m2 lands the wrong basin (2-3x err) so m2_used is
# calibrated from actually-USED assignments; a scalar RTN-snap proxy ranking
# was measured INVALID (+21% recon fp8) and dropped. The micro-sweep span
# covers the measured s0-vs-sweep ratio error (<=1.15).
_ENCODE_TIER_ENV = "PRISMAQUANT_CB_ENCODE_TIER"
_ENCODE_TIERS = ("fast", "balanced", "max")
_ENCODE_TIER_DEFAULT = "balanced"
_TIER_REFITS = {"fast": 1, "balanced": _SCALE_SWEEP_REFIT_ITERS,
                "max": _SCALE_SWEEP_REFIT_ITERS}
_TIER_MICRO_SPAN = {"fast": 1, "balanced": 2}     # +- steps around s0
_TIER_MICRO_RATIO = {"fast": 1.1, "balanced": 1.075}
# Per-group hill-climb extension after the micro-grid (measured: the reach,
# not granularity, sets quality — q_proj span curve hits max-parity at
# +-24% and BEATS max at +-34%; down_proj needs per-group adaptive reach).
_TIER_HILL_ITERS = {"fast": 2, "balanced": 4}
_PILOT_ROWS = 256
_ENCODE_COMPILE_ENV = "PRISMAQUANT_CB_ENCODE_COMPILE"

# --- Reconstruction chunking (production memory bound) -----------------
# The layer-0 fused gate_up_proj smoke (E=256) OOMed in nvfp4_cb_reconstruct
# with a 16 GiB torch.cat (67 GiB allocated + 23 GiB reserved, 90 GiB process).
# Chunking is over the authoritative outer expert dimension when the weight is
# 3-D packed, otherwise over rows. This keeps the hot path GPU-bound, releases
# temporaries promptly per chunk, and never moves data to CPU/NVMe.
# Default expert chunk 8 bounds the largest single decode to ~512 MiB on logical
# E=256,R=4096,C=4096 fused gate_up_proj (8*4096*4096*4 = 512 MiB) — physical MXFP4
# source halves the stored reduce dimension to 2048 I8, but decode is sized on logical
# 4096; the 512 MiB figure is the decode chunk alone (CPU/static bound, not a live
# allocation validation or guaranteed fit), not total process peak which also
# includes BF16 logical fused weight 8 GiB (E=256,R=4096,C=4096 *2 bytes; per leaf
# gate/up 2048 rows each), fields, activations and workspace and still requires live
# CUDA. Full FP32 reconstruction is 16 GiB; BF16 logical fused is 8 GiB. Physical
# w1/w3 I8 stores reduce width 2048 due to nibble packing. Larger chunks increase
# largest contiguous alloc with no quality gain, so 8 is the production default.
# Validated: chunk sizes 1..16 produce bit-identical results; bound validated by
# tests/test_cb_recon_chunking.py (test_chunk_env_validation, test_reconstruct_chunked_vs_monolithic_*).
_RECON_EXPERT_CHUNK_ENV = "PRISMAQUANT_CB_RECON_EXPERT_CHUNK"
_RECON_ROW_CHUNK_ENV = "PRISMAQUANT_CB_RECON_ROW_CHUNK"
_DEFAULT_RECON_EXPERT_CHUNK = 8
_DEFAULT_RECON_ROW_CHUNK = 4096


def _resolve_recon_expert_chunk() -> int:
    raw = os.environ.get(_RECON_EXPERT_CHUNK_ENV, str(_DEFAULT_RECON_EXPERT_CHUNK)).strip()
    if raw == "":
        raw = str(_DEFAULT_RECON_EXPERT_CHUNK)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_RECON_EXPERT_CHUNK_ENV} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_RECON_EXPERT_CHUNK_ENV} must be positive, got {value}")
    return value


def _resolve_recon_row_chunk() -> int:
    raw = os.environ.get(_RECON_ROW_CHUNK_ENV, str(_DEFAULT_RECON_ROW_CHUNK)).strip()
    if raw == "":
        raw = str(_DEFAULT_RECON_ROW_CHUNK)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_RECON_ROW_CHUNK_ENV} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_RECON_ROW_CHUNK_ENV} must be positive, got {value}")
    return value


_LDLQ_EXPERT_BATCH_ENV = "PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"
_DEFAULT_LDLQ_EXPERT_BATCH = 16


def _resolve_ldlq_expert_batch() -> int:
    """Strict resolver shared with batched LDLQ path — malformed or nonpositive raises."""
    raw = os.environ.get(_LDLQ_EXPERT_BATCH_ENV, str(_DEFAULT_LDLQ_EXPERT_BATCH)).strip()
    if raw == "":
        raw = str(_DEFAULT_LDLQ_EXPERT_BATCH)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_LDLQ_EXPERT_BATCH_ENV} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_LDLQ_EXPERT_BATCH_ENV} must be a positive integer, got {value}")
    return value


def _canonical_split_infos_digest(split_infos: Sequence[Mapping[str, Any]]) -> str:
    """Canonical helper for split_infos digest — no broad fallback. Serialization error is bug and must abort/fail closed."""
    return hashlib.sha256(
        json.dumps([dict(s) for s in split_infos], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _slice_fields_for_rows(fields: dict, start: int, end: int, *, new_shape: tuple[int, ...] | None = None) -> dict:
    """Return a shallow copy of ``fields`` sliced to rows [start:end)."""
    sliced: dict = {}
    # Indices / signs / scales are row-aligned; slice dim 0.
    for key in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if key in fields:
            value = fields[key]
            if isinstance(value, torch.Tensor):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
    # Preserve all global metadata and tuple product codebooks (not row-aligned).
    # Explicitly copy codebook/scale_coding and any other non-row-aligned keys.
    for key in ("codebook", "scale_coding"):
        if key in fields:
            sliced[key] = fields[key]
    # Preserve any other global metadata not in row-aligned set (e.g., lattice version, etc.)
    for key, value in fields.items():
        if key not in sliced and key not in ("shape",):
            # Already handled row-aligned and codebook/scale_coding; copy remaining globals
            if key not in ("indices", "signs", "scales", "scale_super", "scale_sub", "codebook", "scale_coding"):
                sliced[key] = value
    if new_shape is not None:
        sliced["shape"] = new_shape
    elif "shape" in fields:
        # Caller will override when needed; keep original for single-chunk path
        sliced["shape"] = fields["shape"]
    return sliced


def iter_nvfp4_cb_recon_chunks(
    fields: dict,
    k: int,
    *,
    grid: str = "fp4",
    mode: str = "product",
    codebook: torch.Tensor | tuple | None = None,
):
    """Public, validated reconstruction-chunk iterator over the outer expert dim.

    Validates that ``fields["shape"]`` is 3-D (E,R,C), that ``fields["indices"]``
    has ``E*R`` rows (fail-closed on mismatch), and that E is divisible only
    via the shape contract (no silent rounding). Yields authoritative expert
    ranges ``(first, last)`` and decoded ``(last-first, R, C)`` float32 chunks
    on the fields' device, preserving product tuple codebooks and two-tier
    scale metadata via ``_slice_fields_for_rows``. Single-ownership: each
    chunk is freshly allocated and the caller owns its lifetime; no full
    E*R*C buffer is ever materialized.

    Raises ValueError/TypeError fail-closed on malformed shape or non-divisible
    row alignment.
    """
    shape = fields.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError(
            f"iter_nvfp4_cb_recon_chunks: fields shape must be 3-D (E,R,C), got {shape!r}"
        )
    try:
        E, R, C = map(int, shape)
    except Exception as exc:
        raise ValueError(f"iter_nvfp4_cb_recon_chunks: shape not int-parseable {shape!r}") from exc
    if E <= 0 or R <= 0 or C <= 0:
        raise ValueError(f"iter_nvfp4_cb_recon_chunks: shape dimensions must be positive, got {(E,R,C)}")
    indices = fields.get("indices")
    if not isinstance(indices, torch.Tensor):
        raise ValueError("iter_nvfp4_cb_recon_chunks: fields missing Tensor indices")
    if indices.ndim < 1:
        raise ValueError(f"iter_nvfp4_cb_recon_chunks: indices must be at least 1-D, got ndim={indices.ndim}")
    rows = int(indices.shape[0])
    if rows != E * R:
        raise ValueError(
            f"iter_nvfp4_cb_recon_chunks: malformed 3-D shape {tuple(shape)} inconsistent with "
            f"indices rows {rows} (expected {E}*{R}={E*R}); fail-closed—refusing invalid reshape"
        )
    # Validate every present row-aligned field is a Tensor with first dimension exactly E*R
    for _key in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if _key in fields:
            _val = fields[_key]
            if not isinstance(_val, torch.Tensor):
                raise ValueError(f"iter_nvfp4_cb_recon_chunks: fields[{_key!r}] must be Tensor, got {type(_val).__name__}")
            if _val.ndim < 1:
                raise ValueError(f"iter_nvfp4_cb_recon_chunks: fields[{_key!r}] must be at least 1-D, got ndim={_val.ndim}")
            if int(_val.shape[0]) != E * R:
                raise ValueError(
                    f"iter_nvfp4_cb_recon_chunks: fields[{_key!r}] rows {int(_val.shape[0])} != E*R={E*R} (shape {tuple(_val.shape)} vs E={E} R={R}); fail-closed—refusing truncated/misaligned rows"
                )
            # Also ensure required dimensionality: indices must be at least 1-D, scales at least 2-D etc. already covered by ndim check
    # E must be integral; rows already ensures E*R alignment, no extra remainder.
    chunk_experts = _resolve_recon_expert_chunk()
    for first in range(0, E, chunk_experts):
        last = min(first + chunk_experts, E)
        row_start = first * R
        row_end = last * R
        chunk_shape = (last - first, R, C)
        chunk_fields = _slice_fields_for_rows(fields, row_start, row_end, new_shape=chunk_shape)
        chunk_recon = _nvfp4_cb_reconstruct_one(chunk_fields, k, grid=grid, mode=mode, codebook=codebook)
        yield (first, last), chunk_recon, chunk_fields


def _resolve_encode_tier(tier: str | None) -> str:
    t = tier if tier is not None else os.environ.get(
        _ENCODE_TIER_ENV, _ENCODE_TIER_DEFAULT)
    t = str(t).strip().lower()
    if t not in _ENCODE_TIERS:
        raise ValueError(
            f"unknown encode tier {t!r} (expected one of {_ENCODE_TIERS})")
    return t

_DATA = Path(__file__).resolve().parent / "data" / "nvfp4_cb_lattices.pt"
_LATTICE_SEED = 1234
_LATTICE_SAMPLES = 1 << 17
_LATTICE_ITERS = 12

# E2M1: {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@lru_cache(maxsize=None)
def _e2m1_grid(device: str) -> torch.Tensor:
    signed = {0.0}
    for v in _E2M1_VALUES[1:]:
        signed.add(v)
        signed.add(-v)
    return torch.tensor(sorted(signed), dtype=torch.float32,
                        device=torch.device(device))


def _snap_to_grid(t: torch.Tensor, grid: str,
                  positive: bool = False) -> torch.Tensor:
    """Project every coordinate onto the element grid (nearest).

    ``positive=True`` restricts to the non-negative half-grid (magnitude
    codebooks for signed mode): clamp to >=0 first — the nearest full-grid
    value of a non-negative input is itself non-negative, so a plain snap
    then lands on the half-grid."""
    if positive:
        t = t.clamp_min(0)
    if grid == "fp8":
        return (t.clamp(-FP8_ELEMENT_MAX, FP8_ELEMENT_MAX)
                .to(torch.float8_e4m3fn).to(torch.float32))
    if grid != "fp4":
        raise ValueError(f"unknown grid {grid!r} (expected 'fp4' or 'fp8')")
    cb = _e2m1_grid(str(t.device))
    x = t.to(torch.float32).contiguous()
    idx = torch.bucketize(x, cb)
    lo = cb[(idx - 1).clamp_min(0)]
    hi = cb[idx.clamp_max(cb.numel() - 1)]
    return torch.where((hi - x).abs() < (x - lo).abs(), hi, lo)


# ---------------------------------------------------------------------------
# Weighted VQ assignment (the imatrix-weighted exhaustive argmin).
# ---------------------------------------------------------------------------

def _vq_dist_argmin_eager(term2: torch.Tensor,
                          term1: torch.Tensor) -> torch.Tensor:
    """argmin_c of ``term2 - 2*term1`` over the trailing codeword axis."""
    return (term2 - 2.0 * term1).argmin(dim=-1)


@lru_cache(maxsize=None)
def _vq_dist_argmin_compiled():
    return torch.compile(_vq_dist_argmin_eager, dynamic=True)


def _vq_dist_argmin(term2: torch.Tensor, term1: torch.Tensor) -> torch.Tensor:
    """Fused distance + argmin.

    Eagerly this materializes a whole (m, K) fp32 distance plane, writes it,
    and reads it straight back for the reduction — three passes over the
    largest tensor in the encoder for one index per row. Compiled, the
    subtraction folds into the reduction and the plane never exists (measured
    2.9x on the production shapes). Same per-element arithmetic and the same
    first-occurrence tie rule, so the chosen codewords are unchanged."""
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _vq_dist_argmin_compiled()(term2, term1)
        except Exception:
            pass
    return _vq_dist_argmin_eager(term2, term1)


def _vq_assign(x: torch.Tensor, cb: torch.Tensor,
               wq: torch.Tensor | None,
               wq_period: int | None = None) -> torch.Tensor:
    """Argmin_c sum_j wq_j (x_j - cb[c]_j)^2 per row of ``x``.

    ``x`` is (m, d), ``cb`` is (K, d), ``wq`` is (m, d) or None. The additive
    sum_j wq_j x_j^2 term is constant per row and dropped (cancels in argmin).

    ``wq_period``: ``wq`` repeats every ``wq_period`` rows (per-column imatrix
    broadcast over rows), so the scale-independent ``wq @ cb_sq^T`` term is
    built once from the base block and broadcast instead of being materialized
    at (m, K) on every call. Same per-element values, same reduction axis.
    """
    m, K = x.shape[0], cb.shape[0]
    cb = cb.to(x.device, torch.float32)
    cb_sq = cb * cb
    cb_t = cb.t().contiguous()
    idx = torch.empty(m, dtype=torch.long, device=x.device)
    chunk = max(1, _SCORE_CHUNK_ELEMS // max(K, 1))
    if wq is None:
        cb_sqnorm = cb_sq.sum(dim=-1)
        for a in range(0, m, chunk):
            b = min(m, a + chunk)
            idx[a:b] = _vq_dist_argmin(cb_sqnorm, x[a:b] @ cb_t)
        return idx
    cb_sq_t = cb_sq.t().contiguous()
    P = wq_period
    if not (P and 0 < P < m and m % P == 0 and chunk % P == 0):
        P = None
    if P is not None:
        term2 = wq[:P] @ cb_sq_t                      # (P, K)
        for a in range(0, m, chunk):
            b = min(m, a + chunk)
            r = (b - a) // P
            term1 = (wq[a:b] * x[a:b]) @ cb_t
            idx[a:b] = _vq_dist_argmin(
                term2.reshape(1, P, K), term1.reshape(r, P, K)).reshape(-1)
        return idx
    for a in range(0, m, chunk):
        b = min(m, a + chunk)
        wc = wq[a:b]
        term1 = (wc * x[a:b]) @ cb_t
        term2 = wc @ cb_sq_t
        idx[a:b] = _vq_dist_argmin(term2, term1)
    return idx


# ---------------------------------------------------------------------------
# Fixed lattice + learned codebook (weighted Lloyd on the element grid).
# ---------------------------------------------------------------------------

def _lloyd(samples: torch.Tensor, init: torch.Tensor, grid: str,
           weights: torch.Tensor | None, iters: int, seed: int,
           positive: bool = False) -> torch.Tensor:
    """Grid-snapped weighted Lloyd. Every centroid coordinate is projected
    onto the element grid after each update, so codewords stay grid-valued
    (the positive half-grid for magnitude codebooks)."""
    cb = _snap_to_grid(init.to(torch.float32), grid, positive=positive)
    K, d = cb.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(iters)):
        assign = _vq_assign(samples, cb, weights)
        # Index-based accumulation: a dense (m, K) one-hot is ~51 GB fp32 at
        # 27B-Linear scale (m~3.1M, K=4096) and swap-kills a UMA box.
        counts = torch.bincount(assign, minlength=K).to(samples.dtype)
        if weights is None:
            summ = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            summ.index_add_(0, assign, samples)
            new = summ / counts.clamp_min(1.0).unsqueeze(-1)
        else:
            wsum = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            wsum.index_add_(0, assign, weights)
            summ = torch.zeros(K, d, dtype=samples.dtype,
                               device=samples.device)
            summ.index_add_(0, assign, weights * samples)
            new = summ / wsum.clamp_min(1e-12)
        empty = counts == 0
        if bool(empty.any()):
            n_empty = int(empty.sum())
            pick = torch.randint(0, samples.shape[0], (n_empty,),
                                 generator=gen).to(samples.device)
            new[empty] = samples[pick]
        cb = _snap_to_grid(new, grid, positive=positive)
    return cb


@lru_cache(maxsize=None)
def _lattice_file() -> dict[str, torch.Tensor]:
    if _DATA.exists():
        return torch.load(_DATA, map_location="cpu", weights_only=True)
    return {}


def _lattice_key(k: int, grid: str, d: int, positive: bool = False) -> str:
    return f"{grid}{'pos' if positive else ''}_d{d}_k{k}"


@lru_cache(maxsize=None)
def _fixed_lattice_cpu(k: int, grid: str, d: int,
                       positive: bool = False) -> torch.Tensor:
    if k > MAX_FLAT_K:
        raise ValueError(
            f"flat codebook infeasible at k={k} (2^{k} codewords > "
            f"2^{MAX_FLAT_K}); provide an explicit/structured codebook")
    cached = _lattice_file().get(_lattice_key(k, grid, d, positive))
    if cached is not None:
        return cached.to(torch.float32).contiguous()
    return _build_lattice(k, grid, d, positive=positive)


def _build_lattice(k: int, grid: str, d: int,
                   positive: bool = False) -> torch.Tensor:
    """Deterministic universal lattice: grid-snapped Lloyd on seeded samples
    drawn from the *post-normalization* distribution each grid's encoder
    actually produces. Regenerated on cache miss.

    Both families must train at the data scale or the codewords cluster far
    from the data and reconstruction collapses (2026-07-15: the original fp4
    path trained on standard N(0,1) while NVFP4 group-16 normalization yields
    normalized weights of std ~2.9 / absmax ~6, giving whole-model emulated
    KL ~15 / top1 ~0 — a measurement bug that would have falsely killed the
    family). Fixes:
      * fp8 — rows scale by amax/448, so scaled vectors live at
        ~sigma·448/amax_sigma; train at that scale (amax ~ 4 sigma).
      * fp4 — group-16 amax→6 normalization; train on genuinely NVFP4-
        normalized Gaussian weights via the encoder's own
        ``_scale_and_vectorize`` (no hand-tuned scale constant), so the
        lattice matches the exact distribution the encoder feeds it.
    """
    K = 1 << k
    gen = torch.Generator(device="cpu").manual_seed(_LATTICE_SEED + k * 131 + d)
    m = max(_LATTICE_SAMPLES, K * 16)
    if grid == "fp8":
        samples = torch.randn(m, d, generator=gen) * (FP8_ELEMENT_MAX / 4.0)
    else:  # fp4: normalized-weight samples at the true encoder scale.
        in_f = 512  # multiple of the group-16 scale window
        n8 = (m * d + VEC_DIM - 1) // VEC_DIM
        rows = (n8 * VEC_DIM + in_f - 1) // in_f
        w = torch.randn(rows, in_f, generator=gen)
        vectors, _, _ = _scale_and_vectorize(w, "fp4")   # (rows*64, 8), std~2.9
        samples = vectors.reshape(-1, d)[:m].contiguous()
    if positive:
        # Magnitude lattice: train on |x| of the same post-normalization
        # distribution (exactly what the signed-mode encoder searches over).
        samples = samples.abs()
    if torch.cuda.is_available():
        samples = samples.cuda()
    perm = torch.randperm(samples.shape[0], generator=gen).to(samples.device)[:K]
    init = samples[perm]
    return _lloyd(samples, init, grid, None, _LATTICE_ITERS, _LATTICE_SEED,
                  positive=positive).cpu()


def fixed_lattice(k: int, grid: str, d: int = 8,
                  positive: bool = False) -> torch.Tensor:
    """Universal (2^k, d) codebook of grid-valued codewords (positive
    half-grid magnitude codewords when ``positive=True``)."""
    return _fixed_lattice_cpu(int(k), str(grid), int(d), bool(positive))


def learn_codebook(vectors: torch.Tensor, k: int, *, grid: str,
                   col_weights: torch.Tensor | None = None,
                   init: torch.Tensor | None = None, iters: int = 4,
                   seed: int = 0, positive: bool = False) -> torch.Tensor:
    """Weighted Lloyd codebook on the element grid. Returns a (2^k, d)
    grid-valued tensor (positive half-grid when ``positive=True`` — pass
    ``|vectors|`` to learn a signed-mode magnitude codebook). Deterministic
    given ``seed`` + ``init`` on CPU; on CUDA the index_add_ float atomics
    can flip grid-snap ties across runs, so ship the resulting codebook
    rather than regenerating it."""
    vectors = vectors.to(torch.float32)
    d = vectors.shape[-1]
    vectors = vectors.reshape(-1, d)
    if init is None:
        init = fixed_lattice(k, grid, d, positive=positive).to(vectors.device)
    else:
        init = init.to(vectors.device, torch.float32)
    if (1 << int(k)) != init.shape[0]:
        raise ValueError(f"init has {init.shape[0]} entries, expected 2^{k}")
    weights = None
    if col_weights is not None:
        weights = torch.broadcast_to(
            col_weights.to(vectors.device, torch.float32), vectors.shape
        ).contiguous()
    return _lloyd(vectors, init, grid, weights, iters, seed,
                  positive=positive)


def _resolve_codebook(k: int, grid: str, mode: str,
                      codebook: torch.Tensor | tuple | None,
                      device: torch.device):
    if mode == "full":
        if codebook is None:
            cb = fixed_lattice(k, grid, VEC_DIM)
        else:
            cb = codebook
        return cb.to(device, torch.float32)
    if mode == "product":
        n_sub = family_for(grid, mode).n_sub
        expected_shapes = codebook_subtable_shapes(k, mode, n_sub)
        if codebook is None:
            tables = tuple(
                fixed_lattice(bits, grid, sub_dim)
                for bits, (_, sub_dim) in zip(
                    subtable_bit_widths(k, mode, n_sub), expected_shapes
                )
            )
        else:
            tables = tuple(codebook)
        actual_shapes = tuple(tuple(int(dim) for dim in table.shape)
                              for table in tables)
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"{grid} {mode} k={k} codebook shapes {actual_shapes} do "
                f"not match canonical serialized shapes {expected_shapes}"
            )
        return tuple(t.to(device, torch.float32) for t in tables)
    if mode == "signed":
        n_sub = family_for(grid, mode).n_sub
        (m,) = subtable_bit_widths(k, mode, n_sub)
        (expected_shape,) = codebook_subtable_shapes(k, mode, n_sub)
        if codebook is None:
            cb = fixed_lattice(m, grid, VEC_DIM, positive=True)
        else:
            cb = codebook
        cb = cb.to(device, torch.float32)
        actual_shape = tuple(int(dim) for dim in cb.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{grid} {mode} k={k} codebook shape {actual_shape} does "
                f"not match canonical serialized shape {expected_shape}"
            )
        if bool((cb < 0).any()):
            raise ValueError(
                "signed-mode magnitude codebook must be non-negative "
                "(sign-optimality requires codewords on the positive "
                "half-grid)")
        return cb
    raise ValueError(
        f"unknown mode {mode!r} (expected 'full', 'product' or 'signed')")


# ---------------------------------------------------------------------------
# Scale + vectorize.
# ---------------------------------------------------------------------------

def _fp4_group_scale(w2d: torch.Tensor) -> torch.Tensor:
    """NVFP4 group-16 effective scale (rows, in//16), via the export codec so
    resident emulation == served bytes."""
    from . import export_native_compressed as enc

    rows, in_f = w2d.shape
    grouped = w2d.to(torch.float32).reshape(rows, in_f // FP4_GROUP, FP4_GROUP)
    scale_real, global_real = enc._select_nvfp4_pack_scales_and_global(grouped)
    return enc._nvfp4_effective_scale_from_real(
        scale_real, global_real, quantize_fp8=True)


def _per_element_scale(scales: torch.Tensor, grid: str,
                       in_f: int) -> torch.Tensor:
    if grid == "fp4":
        return scales.repeat_interleave(FP4_GROUP, dim=1)
    return scales.expand(scales.shape[0], in_f)


def _scale_and_vectorize(w2d: torch.Tensor, grid: str):
    """Return (vectors (nvec, 8), scales, per_element_scale). ``scales`` is the
    stored scale plane: (rows, in//16) for fp4, (rows, 1) for fp8."""
    rows, in_f = w2d.shape
    wf = w2d.to(torch.float32)
    if grid == "fp4":
        scales = _fp4_group_scale(wf)
    elif grid == "fp8":
        scales = (wf.abs().amax(dim=-1, keepdim=True) / FP8_ELEMENT_MAX
                  ).clamp_min(1e-12)
    else:
        raise ValueError(f"unknown grid {grid!r}")
    pes = _per_element_scale(scales, grid, in_f)
    x = wf / pes
    vectors = x.reshape(rows * (in_f // VEC_DIM), VEC_DIM)
    return vectors, scales, pes


# ---------------------------------------------------------------------------
# Fields / reconstruct.
# ---------------------------------------------------------------------------

def _col_weight_vectors(cw2d: torch.Tensor) -> torch.Tensor:
    """Reshape a per-element (rows, in) weight to (nvec, 8) with a dead-vector
    guard (all-zero weight -> unweighted)."""
    wq = cw2d.reshape(-1, VEC_DIM)
    mass = wq.sum(dim=-1, keepdim=True)
    return torch.where(mass == 0, torch.ones_like(wq), wq)


def _mode_encode(vectors: torch.Tensor, mode: str, cb, wq,
                 wq_period: int | None = None) -> dict:
    """VQ-assign scaled ``vectors`` (nvec, 8) under one mode. Returns per-mode
    index fields ({"idx": (nvec,) or (nvec, n_sub)}, + "signs" for signed)."""
    if mode == "full":
        return {"idx": _vq_assign(vectors, cb, wq, wq_period)}
    if mode == "signed":
        # Exactly separable under weighted L2: for any magnitude codeword
        # c >= 0, sum_j w_j (x_j - s_j c_j)^2 is minimized over s_j in {+-1}
        # by s_j = sign(x_j) (the cross-term -2 w_j s_j x_j c_j is largest
        # when s_j x_j >= 0, independent of which codeword is chosen), and at
        # that sign the objective equals sum_j w_j (|x_j| - c_j)^2. So the
        # weighted argmin over |x| plus signs = sign(x) IS the joint optimum
        # — no sign x magnitude search needed. Zero-safe: sign(0) -> +1.
        return {"idx": _vq_assign(vectors.abs(), cb, wq, wq_period),
                "signs": torch.where(vectors < 0, -1.0, 1.0)}
    n_sub = len(cb)
    sub_dim = VEC_DIM // n_sub
    idxs = []
    for i, table in enumerate(cb):
        xs = vectors[:, i * sub_dim:(i + 1) * sub_dim]
        ws = wq[:, i * sub_dim:(i + 1) * sub_dim] if wq is not None else None
        idxs.append(_vq_assign(xs, table, ws, wq_period))
    return {"idx": torch.stack(idxs, dim=-1)}


def _mode_decode(enc: dict, mode: str, cb) -> torch.Tensor:
    """Scaled-domain grid reconstruction (nvec, 8) from index fields."""
    if mode == "full":
        return cb[enc["idx"]]
    if mode == "signed":
        return cb[enc["idx"]] * enc["signs"]
    parts = [table[enc["idx"][:, i]] for i, table in enumerate(cb)]
    return torch.cat(parts, dim=-1)


def _enc_to_fields(enc: dict, mode: str, cb, rows: int, in_f: int,
                   nvec_per_row: int) -> dict:
    if mode == "full":
        return {"indices": enc["idx"].reshape(rows, nvec_per_row)}
    if mode == "signed":
        return {"indices": enc["idx"].reshape(rows, nvec_per_row),
                "signs": enc["signs"].reshape(rows, in_f)}
    return {"indices": enc["idx"].reshape(rows, nvec_per_row, len(cb))}


def _group_amax(w2d: torch.Tensor, grid: str) -> torch.Tensor:
    rows, in_f = w2d.shape
    if grid == "fp4":
        return w2d.reshape(rows, in_f // FP4_GROUP, FP4_GROUP).abs().amax(-1)
    return w2d.abs().amax(-1, keepdim=True)


def _group_reduce(err: torch.Tensor, grid: str) -> torch.Tensor:
    rows, in_f = err.shape
    if grid == "fp4":
        return err.reshape(rows, in_f // FP4_GROUP, FP4_GROUP).sum(-1)
    return err.sum(-1, keepdim=True)


def _snap_scale(s: torch.Tensor, grid: str) -> torch.Tensor:
    """Project a per-group scale onto the legal grid: E4M3 (fp4 block scale,
    exactly like NVFP4) or fp32 (fp8 per-channel). Floored strictly positive."""
    if grid == "fp4":
        return _snap_to_grid(s, "fp8").clamp_min(_E4M3_MIN_POS)
    return s.clamp_min(1e-12)


def _candidate_scales(amax: torch.Tensor, grid: str, n: int) -> torch.Tensor:
    """(n, *amax.shape) legal candidate scales sweeping the clipping level.
    Candidate 0 is L=grid-max == the one-shot amax/grid-max scale, so an
    argmin over these is never worse than the one-shot."""
    if grid == "fp4":
        levels = torch.linspace(NVFP4_GRID_MAX, 4.0, n, device=amax.device)
    else:
        levels = torch.linspace(FP8_ELEMENT_MAX, FP8_ELEMENT_MAX * 4.0 / 6.0,
                                n, device=amax.device)
    shape = (n,) + (1,) * amax.dim()
    return _snap_scale(amax.unsqueeze(0) / levels.reshape(shape), grid)


def _eval_candidate(w2d: torch.Tensor, wq: torch.Tensor | None,
                    s: torch.Tensor, grid: str, mode: str, cb,
                    wq_period: int | None = None):
    """Encode ``w2d`` at per-group scale ``s`` and score the WEIGHTED
    reconstruction error in the ORIGINAL weight domain (so the scale choice
    is judged on real error, not scaled-domain error). Returns
    (err_group (rows, ngroups), enc, grid_decode (rows, in))."""
    rows, in_f = w2d.shape
    pes = _per_element_scale(s, grid, in_f)              # (rows, in)
    pes_vec = pes.reshape(-1, VEC_DIM)                   # (nvec, 8)
    wvec = w2d.reshape(-1, VEC_DIM)
    x = wvec / pes_vec
    enc = _mode_encode(x, mode, cb, wq, wq_period)
    dec = _mode_decode(enc, mode, cb)                    # (nvec, 8) grid
    recon = dec * pes_vec                                # original domain
    err = (recon - wvec).pow(2)
    if wq is not None:
        err = err * wq
    err_group = _group_reduce(err.reshape(rows, in_f), grid)
    return err_group, enc, dec.reshape(rows, in_f)


# ---------------------------------------------------------------------------
# Moment-scored sweep (fast/balanced tiers). The candidate error decomposes
# as err(s) = C - 2s*B + s^2*A with per-(vector, entry) moments A, B that do
# NOT depend on the scale (the _grid_fields/_sweep_errs identity), so all
# candidates score from moments built ONCE per chunk instead of recomputing
# a full VQ distance matrix per candidate. C is constant per vector across
# candidates and cancels in every argmin, so it is dropped. The selection
# matches the exact sweep up to fp-rounding ties; refits and the final
# assignment stay on the exact direct-eval path.
# ---------------------------------------------------------------------------

# --- Row-periodic A ---------------------------------------------------------
# A = sum_j w_j t_j^2 depends only on the COLUMN weights, and the production
# imatrix is one per-input-column vector broadcast over rows. wq therefore
# repeats every ``in_f // VEC_DIM`` vectors, so A repeats too: the (m, K)
# moment is ``m / P`` identical copies of a (P, K) base. Keeping only the base
# removes half the moment build and half the bytes every scoring pass has to
# stream (the scan is bandwidth-bound on exactly these two planes), and the
# base is small enough to stay resident in L2. Values are untouched — each
# output element is the same length-``sub_dim`` dot product — so every emitted
# byte is unchanged.


def _periodic_split(A: torch.Tensor, B: torch.Tensor):
    """(P, R) when A is a row-periodic base for B, else (None, None)."""
    if A.dim() != 2:
        return None, None
    P = A.shape[0]
    m = B.shape[0]
    if P == m or P <= 0 or m % P:
        return None, None
    return P, m // P


def _score_min_eager(A: torch.Tensor, B: torch.Tensor,
                     s: torch.Tensor) -> torch.Tensor:
    """min over entries of s^2*A - 2s*B; A (m,K), (P,K) or (K,), B (m,K),
    s (m,1)."""
    P, R = _periodic_split(A, B)
    if P is None:
        return ((s * s) * A - (2.0 * s) * B).min(dim=-1).values
    K = B.shape[-1]
    sv = s.reshape(R, P, 1)
    d = (sv * sv) * A.reshape(1, P, K) - (2.0 * sv) * B.reshape(R, P, K)
    return d.min(dim=-1).values.reshape(-1)


def _score_argmin_eager(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor):
    P, R = _periodic_split(A, B)
    if P is None:
        d = (s * s) * A - (2.0 * s) * B
        v, i = d.min(dim=-1)
        return v, i
    K = B.shape[-1]
    sv = s.reshape(R, P, 1)
    d = (sv * sv) * A.reshape(1, P, K) - (2.0 * sv) * B.reshape(R, P, K)
    v, i = d.min(dim=-1)
    return v.reshape(-1), i.reshape(-1)


@lru_cache(maxsize=None)
def _score_min_compiled():
    return torch.compile(_score_min_eager, dynamic=True)


@lru_cache(maxsize=None)
def _score_argmin_compiled():
    return torch.compile(_score_argmin_eager, dynamic=True)


def _encode_compile_on() -> bool:
    return os.environ.get(_ENCODE_COMPILE_ENV, "1").lower() not in (
        "0", "false", "no")


_ENCODE_RECOMPILE_LIMIT_RAISED = False


def _raise_encode_recompile_limit() -> None:
    """The moment-scoring kernels are compiled once (dynamic=True) but see a
    handful of distinct (K, S) specializations across grids/tiers/formats
    (fp8 K2048/S14, fp4 K256/S16, the S1 refit argmins, ...). The default
    dynamo recompile_limit=8 is exceeded across a mixed-format run, silently
    dropping the compiled path back to EAGER — which materializes the whole
    (m, K, S) intermediate and runs ~30x slower. Raise the limit so every
    specialization stays compiled (mirrors _make_rtn's defensive bump)."""
    global _ENCODE_RECOMPILE_LIMIT_RAISED
    if _ENCODE_RECOMPILE_LIMIT_RAISED:
        return
    try:
        torch._dynamo.config.recompile_limit = max(
            int(getattr(torch._dynamo.config, "recompile_limit", 8)), 256)
        torch._dynamo.config.accumulated_recompile_limit = max(
            int(getattr(torch._dynamo.config,
                        "accumulated_recompile_limit", 256)), 4096)
    except Exception:
        pass
    _ENCODE_RECOMPILE_LIMIT_RAISED = True


def _score_min(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_min_compiled()(A, B, s)
        except Exception:
            pass
    return _score_min_eager(A, B, s)


def _score_min_batched_eager(A: torch.Tensor, B: torch.Tensor,
                             s: torch.Tensor) -> torch.Tensor:
    """min over entries of s^2 A - 2s B for a BATCH of per-vector scales.

    ``A``/``B`` are (m, K); ``s`` is (m, S). Returns (m, S) — the per-vector
    min-over-K at each of the S scales. torch.compile fuses the elementwise
    scoring and the K-reduction so the (m, K) moments are read ONCE for all
    S scales (vs S separate reductions), the launch-bound/volume fix for the
    27B-scale sweep. ``A`` may also be a (P, K) row-periodic base."""
    P, R = _periodic_split(A, B)
    if P is None:
        s2 = (s * s).unsqueeze(1)                    # (m, 1, S)
        ts = (2.0 * s).unsqueeze(1)                  # (m, 1, S)
        d = s2 * A.unsqueeze(-1) - ts * B.unsqueeze(-1)   # (m, K, S)
        return d.min(dim=1).values                   # (m, S)
    K, S = B.shape[-1], s.shape[-1]
    sv = s.reshape(R, P, 1, S)
    d = (sv * sv) * A.reshape(1, P, K, 1) - (2.0 * sv) * B.reshape(R, P, K, 1)
    return d.min(dim=2).values.reshape(-1, S)


def _score_minargmin_batched_eager(A: torch.Tensor, B: torch.Tensor,
                                   s: torch.Tensor):
    """Batched min AND argmin over K for S per-vector scales. Returns
    (values (m, S), indices (m, S)). The argmin comes free with the min
    reduction, so scoring the scale grid ALSO yields the assignment at each
    candidate — folding the separate init-argmin pass into the scan."""
    P, R = _periodic_split(A, B)
    if P is None:
        s2 = (s * s).unsqueeze(1)
        ts = (2.0 * s).unsqueeze(1)
        d = s2 * A.unsqueeze(-1) - ts * B.unsqueeze(-1)   # (m, K, S)
        v, i = d.min(dim=1)
        return v, i
    K, S = B.shape[-1], s.shape[-1]
    sv = s.reshape(R, P, 1, S)
    d = (sv * sv) * A.reshape(1, P, K, 1) - (2.0 * sv) * B.reshape(R, P, K, 1)
    v, i = d.min(dim=2)
    return v.reshape(-1, S), i.reshape(-1, S)


@lru_cache(maxsize=None)
def _score_min_batched_compiled():
    return torch.compile(_score_min_batched_eager, dynamic=True)


@lru_cache(maxsize=None)
def _score_minargmin_batched_compiled():
    return torch.compile(_score_minargmin_batched_eager, dynamic=True)


def _score_minargmin_batched(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_minargmin_batched_compiled()(A, B, s)
        except Exception:
            pass
    vs, is_ = [], []
    for i in range(s.shape[-1]):
        v, ix = _score_argmin_eager(A, B, s[:, i:i + 1])
        vs.append(v)
        is_.append(ix)
    return torch.stack(vs, dim=-1), torch.stack(is_, dim=-1)


def _score_min_batched(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_min_batched_compiled()(A, B, s)
        except Exception:
            pass
    # Eager fallback: loop the S columns so the (m, K, S) intermediate is
    # never materialized (it would OOM at 27B scale — the fused compiled
    # path reads (m, K) once instead). Each column is one (m, K) reduction.
    return torch.stack(
        [_score_min_eager(A, B, s[:, i:i + 1]) for i in range(s.shape[-1])],
        dim=-1)


def _score_argmin(A, B, s):
    if _encode_compile_on():
        try:
            _raise_encode_recompile_limit()
            return _score_argmin_compiled()(A, B, s)
        except Exception:
            pass
    return _score_argmin_eager(A, B, s)


def _mode_streams(wvec: torch.Tensor, mode: str, cb, wq):
    """Per-sub scoring streams [(x, wq_sub, table)] in the ORIGINAL weight
    domain. signed scores on |w| (err (w - s*sign(w)*t)^2 == (|w| - s*t)^2
    for t >= 0); product splits sub-vectors (independent argmins)."""
    if mode == "full":
        return [(wvec, wq, cb)]
    if mode == "signed":
        return [(wvec.abs(), wq, cb)]
    n_sub = len(cb)
    sd = VEC_DIM // n_sub
    return [(wvec[:, i * sd:(i + 1) * sd],
             wq[:, i * sd:(i + 1) * sd] if wq is not None else None,
             cb[i]) for i in range(n_sub)]


def _stream_moments(x: torch.Tensor, ws: torch.Tensor | None,
                    table: torch.Tensor, ws_period: int | None = None):
    """A = sum_j w_j t_j^2 (per entry), B = sum_j w_j x_j t_j.

    When ``ws_period`` is given, ``ws`` is known to repeat every ``ws_period``
    rows (a per-column imatrix broadcast over rows), so A is built from the
    base block alone and returned as (ws_period, K); the scorers broadcast it.
    Each A entry is the same length-``sub_dim`` dot product either way."""
    t = table.to(x.device, torch.float32)
    tt = t.t().contiguous()
    B = ((ws * x) if ws is not None else x) @ tt
    if ws is not None:
        base = ws
        if (ws_period is not None and 0 < ws_period < ws.shape[0]
                and ws.shape[0] % ws_period == 0):
            base = ws[:ws_period]
        A = base @ (t * t).t().contiguous()
    else:
        A = (t * t).sum(dim=-1)
    return A, B


def _moment_rows_step(cb, vec_per_row: int) -> int:
    tables = cb if isinstance(cb, tuple) else (cb,)
    k_max = max(int(t.shape[0]) for t in tables)
    return max(1, (_SCORE_CHUNK_ELEMS // max(k_max, 1)) // max(vec_per_row, 1))


def _moment_err_groups(moms, s_g: torch.Tensor, grid: str, in_f: int,
                       vec_per_group: int) -> torch.Tensor:
    """Per-group error (minus the constant C term) at per-group scale
    ``s_g`` (rc, G), scored from cached moments."""
    rc, G = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    err_v = None
    for (A, B) in moms:
        v = _score_min(A, B, s_v)
        err_v = v if err_v is None else err_v + v
    return err_v.reshape(rc, G, vec_per_group).sum(dim=-1)


def _moment_err_groups_batched(moms, s_g: torch.Tensor, vec_per_group: int
                               ) -> torch.Tensor:
    """Per-group error for a BATCH of candidate scales in one fused pass.

    ``s_g`` is (rc, G, S). Returns (rc, G, S). The batched min reads each
    stream's (m, K) moments ONCE for all S candidates (vs S passes), so the
    whole scale sweep is a single volume pass instead of S — the fix for the
    launch/volume blowup on 27B-scale Linears."""
    rc, G, S = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, S)
    err_v = None
    for (A, B) in moms:
        v = _score_min_batched(A, B, s_v)                 # (m, S)
        err_v = v if err_v is None else err_v + v
    return err_v.reshape(rc, G, vec_per_group, S).sum(dim=2)


def _chunk_moments(wvec, wqc, mode, cb, wq_period: int | None = None):
    """Build the per-stream (A, B) moments for one row-chunk ONCE, reusable
    across the whole per-chunk sweep + argmin + refits (eliminates the 4x
    moment rebuild the old scan/eval/refit split incurred)."""
    return [_stream_moments(x, ws, t, ws_period=wq_period)
            for (x, ws, t) in _mode_streams(wvec, mode, cb, wqc)]


def _scan_and_assign(moms, wvec, grid_s_c, mode, cb, grid, in_f,
                     vec_per_group):
    """Batched exhaustive scale-grid scan that ALSO returns the assignment at
    the chosen (per-group) scale — one fused (m, K) pass does both the min
    (scale selection) and the argmin (codeword assignment), no separate
    init-argmin pass. ``grid_s_c`` is (rc, G, S). Returns
    (best_s (rc, G), err (rc, G), enc, dec (rc, in_f))."""
    rc, G, S = grid_s_c.shape
    s_v = grid_s_c.repeat_interleave(vec_per_group, dim=1).reshape(-1, S)
    err = None
    per_stream_idx = []
    for (A, B) in moms:
        v, i = _score_minargmin_batched(A, B, s_v)        # (m, S), (m, S)
        err = v if err is None else err + v
        per_stream_idx.append(i)
    err_g = err.reshape(rc, G, vec_per_group, S).sum(dim=2)   # (rc, G, S)
    best_col = err_g.argmin(dim=-1)                        # first-min on ties
    best_s = torch.gather(grid_s_c, -1, best_col.unsqueeze(-1)).squeeze(-1)
    err_c = torch.gather(err_g, -1, best_col.unsqueeze(-1)).squeeze(-1)
    col_v = best_col.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    idxs = [torch.gather(i, -1, col_v).squeeze(-1) for i in per_stream_idx]
    if mode == "full":
        enc = {"idx": idxs[0]}
    elif mode == "signed":
        enc = {"idx": idxs[0], "signs": torch.where(wvec < 0, -1.0, 1.0)}
    else:
        enc = {"idx": torch.stack(idxs, dim=-1)}
    dec = _mode_decode(enc, mode, cb).reshape(rc, in_f)
    return best_s, err_c, enc, dec


def _argmin_from_moments(moms, wvec, s_g, mode, cb, grid, in_f,
                         vec_per_group):
    """Assignment + decode at per-group scale ``s_g`` (rc, G) from RESIDENT
    moments (no rebuild). Returns (err (rc, G), enc, dec (rc, in_f))."""
    rc, G = s_g.shape
    s_v = s_g.repeat_interleave(vec_per_group, dim=1).reshape(-1, 1)
    err_v = None
    idxs = []
    for (A, B) in moms:
        v, i = _score_argmin(A, B, s_v)
        err_v = v if err_v is None else err_v + v
        idxs.append(i)
    if mode == "full":
        enc = {"idx": idxs[0]}
    elif mode == "signed":
        enc = {"idx": idxs[0], "signs": torch.where(wvec < 0, -1.0, 1.0)}
    else:
        enc = {"idx": torch.stack(idxs, dim=-1)}
    err = err_v.reshape(rc, G, vec_per_group).sum(dim=-1)
    dec = _mode_decode(enc, mode, cb).reshape(rc, in_f)
    return err, enc, dec


def _calibrate_m2_used(w2d, wq2d, grid, mode, cb,
                       wq_period=None) -> float:
    """Usage-calibrated mean-square codeword coordinate from a pilot encode
    of the tensor's leading rows: the pilot runs the full 16-candidate
    moment-scored sweep, then m2_used = sum(q * dec^2) / sum(q) over the
    pilot's ACTUAL assignments. (The naive all-codeword table m2 lands the
    wrong basin — 2-3x worse error that refits cannot escape; per-tensor
    pilots also absorb the measured per-role m2 variation.)"""
    p1 = min(w2d.shape[0], _PILOT_ROWS)
    pilot = w2d[:p1]
    wp2d = wq2d[:p1] if wq2d is not None else None
    in_f = pilot.shape[1]
    vec_per_group = (FP4_GROUP if grid == "fp4" else in_f) // VEC_DIM
    vec_per_row = in_f // VEC_DIM
    amax = _group_amax(pilot, grid)
    cands = _candidate_scales(amax, grid, _SCALE_SWEEP_CANDIDATES)   # (S,rc,G)
    grid_s = cands.permute(1, 2, 0).contiguous()                    # (rc,G,S)
    rows_step = _moment_rows_step(cb, vec_per_row)
    num_acc = 0.0
    den_acc = 0.0
    for r0 in range(0, p1, rows_step):
        r1 = min(p1, r0 + rows_step)
        wvec = pilot[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wp2d[r0:r1].reshape(-1, VEC_DIM)
               if wp2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)
        errs = _moment_err_groups_batched(moms, grid_s[r0:r1], vec_per_group)
        best_col = errs.argmin(dim=-1)
        best_s = torch.gather(grid_s[r0:r1], -1,
                              best_col.unsqueeze(-1)).squeeze(-1)
        _, _, dec = _argmin_from_moments(
            moms, wvec, best_s, mode, cb, grid, in_f, vec_per_group)
        wcol = (wp2d[r0:r1] if wp2d is not None
                else torch.ones_like(dec))
        num_acc += float((wcol * dec * dec).sum())
        den_acc += float(wcol.sum())
    m2 = num_acc / max(den_acc, 1e-30)
    return max(m2, 1e-30)


def _analytic_s0(w2d, wq2d, grid, m2_used: float) -> torch.Tensor:
    """Per-group second-moment-match scale s0 (rows, G)."""
    wcol = wq2d if wq2d is not None else torch.ones_like(w2d)
    num = _group_reduce(wcol * w2d * w2d, grid)
    den = _group_reduce(wcol, grid) * m2_used
    return (num / den.clamp_min(1e-30)).sqrt()


def _tier_scale_grid(s0, w2d, grid, tier):
    """Exhaustive per-group candidate scale grid (rows, G, S): s0*ratio^i for
    i spanning the FULL reach the micro-sweep + greedy hill-climb could visit
    (+-(span+hill_iters)), plus the amax/grid-max guarantee.

    A single global argmin over this grid is a strict SUPERSET of every scale
    the sequential greedy hill explores, so it is provably no worse than the
    old greedy result (equal for the unimodal RD error that is the actual
    case) while collapsing the ~14 sequential scored passes into ONE batched
    pass. Snapped legal; the guarantee keeps the never-worse-than-one-shot
    contract."""
    ratio = _TIER_MICRO_RATIO[tier]
    reach = _TIER_MICRO_SPAN[tier] + _TIER_HILL_ITERS[tier]
    mults = [ratio ** i for i in range(-reach, reach + 1)]
    cols = [_snap_scale(s0 * m, grid) for m in mults]
    amax = _group_amax(w2d, grid)
    cols.append(_candidate_scales(amax, grid, 1)[0])
    return torch.stack(cols, dim=-1)                       # (rows, G, S)


def _sweep_encode_moment(w2d: torch.Tensor, grid: str, mode: str, cb,
                         wq: torch.Tensor | None, tier: str,
                         wq_period: int | None = None):
    """fast/balanced v1 sweep. Unified per-chunk pipeline: build the (m, K)
    moments ONCE per chunk, batch-score the exhaustive scale grid in a single
    fused pass, then argmin + WLS refits all off the RESIDENT moments (no
    rebuild). The scale grid is a superset of the old greedy hill's reach, so
    encode choices are no worse (equal on unimodal groups)."""
    rows, in_f = w2d.shape
    wq2d = wq.reshape(rows, in_f) if wq is not None else None
    m2 = _calibrate_m2_used(w2d, wq2d, grid, mode, cb, wq_period)
    s0 = _analytic_s0(w2d, wq2d, grid, m2)
    grid_s = _tier_scale_grid(s0, w2d, grid, tier)         # (rows, G, S)
    vec_per_group = (FP4_GROUP if grid == "fp4" else in_f) // VEC_DIM
    vec_per_row = in_f // VEC_DIM
    refits = _TIER_REFITS[tier]
    best_s = torch.empty(rows, s0.shape[1], device=w2d.device)
    enc_parts: list[dict] = []
    rows_step = _moment_rows_step(cb, vec_per_row)
    for r0 in range(0, rows, rows_step):
        r1 = min(rows, r0 + rows_step)
        wvec = w2d[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wq2d[r0:r1].reshape(-1, VEC_DIM)
               if wq2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)
        # Exhaustive scale grid + assignment in ONE fused batched pass.
        s_c, err, enc, dec = _scan_and_assign(
            moms, wvec, grid_s[r0:r1], mode, cb, grid, in_f, vec_per_group)
        w_chunk = w2d[r0:r1]
        wcol = (wq2d[r0:r1] if wq2d is not None else torch.ones_like(w_chunk))
        for _ in range(refits):
            num = _group_reduce(wcol * dec * w_chunk, grid)
            den = _group_reduce(wcol * dec * dec, grid)
            s_star = _snap_scale(
                torch.where(den > 0, num / den.clamp_min(1e-30), s_c), grid)
            err_star, enc_star, dec_star = _argmin_from_moments(
                moms, wvec, s_star, mode, cb, grid, in_f, vec_per_group)
            better = err_star < err
            s_c = torch.where(better, s_star, s_c)
            err = torch.where(better, err_star, err)
            bvec = better.repeat_interleave(vec_per_group, dim=1).reshape(-1)
            for key in enc:
                cur, star = enc[key], enc_star[key]
                mask = bvec if cur.dim() == 1 else bvec.reshape(
                    (-1,) + (1,) * (cur.dim() - 1)).expand_as(cur)
                enc[key] = torch.where(mask, star, cur)
            belem = better.repeat_interleave(
                (FP4_GROUP if grid == "fp4" else in_f), dim=1)
            dec = torch.where(belem, dec_star, dec)
        best_s[r0:r1] = s_c
        enc_parts.append(enc)
    enc = {key: torch.cat([e[key] for e in enc_parts], dim=0)
           for key in enc_parts[0]}
    return best_s, enc


def _sweep_encode(w2d: torch.Tensor, grid: str, mode: str, cb,
                  wq: torch.Tensor | None, wq_period: int | None = None):
    """Joint scale sweep + WLS-refit fixed point (mirrors _grid_fields). Picks
    the per-group scale minimizing weighted real error over the E4M3-legal
    candidate grid, then refines with continuous WLS refits accepted per group
    only when strictly better. Returns (best_scales (rows, ng), enc)."""
    rows, in_f = w2d.shape
    amax = _group_amax(w2d, grid)                        # (rows, ng)
    cands = _candidate_scales(amax, grid, _SCALE_SWEEP_CANDIDATES)
    best_err, _, _ = _eval_candidate(w2d, wq, cands[0], grid, mode, cb,
                                     wq_period)
    best_s = cands[0]
    for si in range(1, cands.shape[0]):
        err, _, _ = _eval_candidate(w2d, wq, cands[si], grid, mode, cb,
                                    wq_period)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, cands[si], best_s)
    # WLS refit: optimal continuous scale s* = sum(w g v) / sum(w g^2) per
    # group at the current (fixed) assignment, snapped legal, accepted per
    # group only when it strictly lowers real error.
    for _ in range(_SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = _eval_candidate(w2d, wq, best_s, grid, mode, cb,
                                        wq_period)
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, grid)
        den = _group_reduce(wcol * g * g, grid)
        s_star = _snap_scale(torch.where(den > 0, num / den.clamp_min(1e-30),
                                         best_s), grid)
        err_star, _, _ = _eval_candidate(w2d, wq, s_star, grid, mode, cb,
                                         wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_star, best_s)
    _, enc, _ = _eval_candidate(w2d, wq, best_s, grid, mode, cb, wq_period)
    return best_s, enc


# ---------------------------------------------------------------------------
# Two-tier scale coding (layout v2): compose/legality tables + encoder.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _two_tier_tables(device: str):
    """Return (table (16,), compose (256, 16) fp32, legal (256, 16) bool).

    ``compose[E, c] = T[c] * 2^(E - 127)``; a pair is legal iff the composed
    value round-trips ``float8_e4m3fn`` bit-exactly and lies in (0, 448]
    (spec §1.2) — so every emitted scale is exact E4M3 by construction."""
    dev = torch.device(device)
    table = torch.tensor(TWO_TIER_SUB_TABLE, dtype=torch.float32, device=dev)
    snapped = table.to(_E4M3).to(torch.float32)
    if not torch.equal(snapped, table):
        raise AssertionError("TWO_TIER_SUB_TABLE entries must be e4m3-exact")
    exps = torch.arange(256, dtype=torch.float64, device=dev)
    compose64 = table.to(torch.float64) * torch.pow(
        2.0, exps - TWO_TIER_SUPER_BIAS).unsqueeze(-1)          # (256, 16)
    compose = compose64.to(torch.float32)
    finite = torch.isfinite(compose)
    rt = torch.where(finite, compose, torch.zeros_like(compose)).to(
        _E4M3).to(torch.float32)
    legal = (finite & (compose > 0) & (compose <= FP8_ELEMENT_MAX)
             & (rt == compose)
             & (compose.to(torch.float64) == compose64))
    return table, compose, legal


@lru_cache(maxsize=None)
def _two_tier_legal_e_range(device: str) -> tuple[int, int]:
    """(e_min, e_max): the first/last super-exponent with ANY legal sub entry.

    A pure function of the (device-independent) legality table, so it is
    resolved once per device instead of costing two ``nonzero`` launches plus
    two device->host syncs on every single encoded tensor. Same integers, so
    the window — and therefore every emitted byte — is unchanged."""
    _, _, legal = _two_tier_tables(device)
    any_legal = legal.any(dim=-1)
    nz = torch.nonzero(any_legal)
    return int(nz[0]), int(nz[-1])


def _two_tier_window(amax: torch.Tensor, ideal: torch.Tensor | None = None,
                     pad: int = _TWO_TIER_WINDOW_PAD):
    """Per-superblock E window (E_lo (rows, n_sb) int64, W int) from the ideal
    group scales (spec §1.4): E_lo so the table top reaches min_ideal, E_hi so
    the table bottom reaches 1.5*max_ideal, padded ``pad`` octaves each side.
    ``ideal`` defaults to amax/6 (the exact path); the analytic tiers pass the
    s0-calibrated ideals so the window centers where winners actually live."""
    rows, G = amax.shape
    n_sb = G * FP4_GROUP // SUPERBLOCK
    if ideal is None:
        ideal = amax / NVFP4_GRID_MAX
    ideal = ideal.clamp_min(_E4M3_MIN_POS)
    ideal_sb = ideal.reshape(rows, n_sb, SUPERBLOCK // FP4_GROUP)
    min_i = ideal_sb.amin(dim=-1)
    max_i = ideal_sb.amax(dim=-1)
    t_max = float(TWO_TIER_SUB_TABLE[-1])
    e_lo = (torch.ceil(torch.log2(min_i / t_max)) + TWO_TIER_SUPER_BIAS
            - pad)
    e_hi = (torch.floor(torch.log2(max_i * 1.5)) + TWO_TIER_SUPER_BIAS
            + pad)
    e_min, e_max = _two_tier_legal_e_range(str(amax.device))
    e_lo = e_lo.clamp(e_min, e_max).to(torch.int64)
    e_hi = e_hi.clamp(e_min, e_max).to(torch.int64)
    e_hi = torch.maximum(e_hi, e_lo)
    W = min(int((e_hi - e_lo).max()) + 1, _TWO_TIER_MAX_WINDOW)
    # When the cap truncates an extreme-spread superblock, keep the TOP of
    # the window: groups below the reachable floor snap UP with error bounded
    # by their (small) magnitude (spec §1.2 zero/degenerate rules), while
    # losing the top would cost ~amax^2 on the largest group.
    e_lo = torch.maximum(e_lo, e_hi - (W - 1))
    return e_lo, e_hi, W


def _sb_to_groups(t_sb: torch.Tensor) -> torch.Tensor:
    """(rows, n_sb) -> (rows, G): broadcast a per-superblock value to its 16
    group-16 slots."""
    rows, n_sb = t_sb.shape
    reps = SUPERBLOCK // FP4_GROUP
    return t_sb.unsqueeze(-1).expand(rows, n_sb, reps).reshape(rows, -1)


def _two_tier_eval_entry(w2d, wq, comp_sb, legal_sb, mode, cb,
                         wq_period=None):
    """Score one sub-table entry at per-superblock composed scale ``comp_sb``
    ((rows, n_sb), maybe illegal): weighted original-domain error per group,
    +inf where the (E, c) pair is illegal."""
    safe = torch.where(legal_sb, comp_sb, torch.ones_like(comp_sb))
    err_g, _, _ = _eval_candidate(w2d, wq, _sb_to_groups(safe), "fp4", mode,
                                  cb, wq_period)
    inf = torch.tensor(float("inf"), device=err_g.device)
    return torch.where(_sb_to_groups(legal_sb.to(torch.bool)), err_g, inf)


def _sweep_encode_two_tier(w2d: torch.Tensor, mode: str, cb,
                           wq: torch.Tensor | None,
                           wq_period: int | None = None):
    """Layout-v2 encoder (spec §1.4): the sweep machinery with the candidate
    set restricted to the two-tier reachable set.

    1. sweep E per superblock over the ideal-scale window; per (E, group) the
       best legal sub-table entry via the weighted original-domain eval;
    2. pick E per superblock by total weighted error;
    3. per-group entry argmin at the frozen E (strict <, so an all-zero group
       deterministically takes the FIRST legal entry — spec zero rule);
    4. WLS refits snapped to the frozen-E reachable set, accepted per group
       only when strictly better.

    Returns (scales (rows, G) composed e4m3-exact, enc, super_E (rows, n_sb),
    sub_codes (rows, G))."""
    rows, in_f = w2d.shape
    n_sb = in_f // SUPERBLOCK
    dev = str(w2d.device)
    _, compose, legal = _two_tier_tables(dev)
    amax = _group_amax(w2d, "fp4")                              # (rows, G)
    e_lo, e_hi, W = _two_tier_window(amax)                      # (rows, n_sb)

    inf = torch.tensor(float("inf"), device=w2d.device)
    best_tot = torch.full((rows, n_sb), float("inf"), device=w2d.device)
    best_e = e_lo.clone()
    for i in range(W):
        E = torch.minimum(e_lo + i, e_hi)
        valid_i = (e_lo + i) <= e_hi
        err_best_g = torch.full_like(amax, float("inf"))
        for c in range(len(TWO_TIER_SUB_TABLE)):
            err_g = _two_tier_eval_entry(
                w2d, wq, compose[E, c], legal[E, c], mode, cb, wq_period)
            err_best_g = torch.minimum(err_best_g, err_g)
        tot = err_best_g.reshape(rows, n_sb, -1).sum(-1)
        tot = torch.where(valid_i, tot, inf)
        better = tot < best_tot
        best_tot = torch.where(better, tot, best_tot)
        best_e = torch.where(better, E, best_e)

    # Per-group entry selection at the frozen per-superblock E. Strict < keeps
    # the FIRST legal entry on ties (all-zero groups -> deterministic bytes).
    best_err_g = torch.full_like(amax, float("inf"))
    best_s = torch.full_like(amax, _E4M3_MIN_POS)
    best_c = torch.zeros(rows, amax.shape[1], dtype=torch.int64,
                         device=w2d.device)
    for c in range(len(TWO_TIER_SUB_TABLE)):
        err_g = _two_tier_eval_entry(
            w2d, wq, compose[best_e, c], legal[best_e, c], mode, cb, wq_period)
        better = err_g < best_err_g
        best_err_g = torch.where(better, err_g, best_err_g)
        best_s = torch.where(better, _sb_to_groups(compose[best_e, c]), best_s)
        best_c = torch.where(better, torch.full_like(best_c, c), best_c)

    # WLS refit on the frozen-E reachable set (spec §1.4 step 3).
    reach = compose[best_e]                                     # (rows,n_sb,16)
    reach_legal = legal[best_e]
    reps = SUPERBLOCK // FP4_GROUP
    reach_g = reach.unsqueeze(2).expand(rows, n_sb, reps, -1)
    legal_g = reach_legal.unsqueeze(2).expand(rows, n_sb, reps, -1)
    for _ in range(_SCALE_SWEEP_REFIT_ITERS):
        err_cur, _, g = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb,
                                        wq_period)
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, "fp4")
        den = _group_reduce(wcol * g * g, "fp4")
        s_star = torch.where(den > 0, num / den.clamp_min(1e-30), best_s)
        dist = (s_star.reshape(rows, n_sb, reps, 1) - reach_g).abs()
        dist = torch.where(legal_g, dist, inf)
        c_star = dist.argmin(dim=-1)                            # (rows,n_sb,16)
        s_snap = torch.gather(reach_g, -1, c_star.unsqueeze(-1)).squeeze(-1)
        s_snap = s_snap.reshape(rows, -1)
        err_star, _, _ = _eval_candidate(w2d, wq, s_snap, "fp4", mode, cb,
                                         wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_snap, best_s)
        best_c = torch.where(better, c_star.reshape(rows, -1), best_c)

    _, enc, _ = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb, wq_period)
    return best_s, enc, best_e, best_c


def _sweep_encode_two_tier_moment(w2d: torch.Tensor, mode: str, cb,
                                  wq: torch.Tensor | None, tier: str,
                                  wq_period: int | None = None):
    """fast/balanced layout-v2 encoder: the windowed-E x entry search scored
    from cached moments (the W*16 combos reuse ONE moment build per chunk),
    with the E window centered on the s0-calibrated ideals (analytic init;
    pad 1 octave balanced / 0 fast), exact refits snapped to the frozen-E
    reachable set. Selection order and tie rules mirror the exact path
    (strict <, first-legal wins).

    NOTE: layout-v2 is the production writer default. Its W*16
    windowed-entry search does not batch cleanly (the launch-bound fix targets
    the v1/fp8-shaped search), so this keeps the original bit-preserving
    structure. Legacy-v1 remains an explicit read/reproduction mode."""
    rows, in_f = w2d.shape
    n_sb = in_f // SUPERBLOCK
    dev = str(w2d.device)
    _, compose, legal = _two_tier_tables(dev)
    amax = _group_amax(w2d, "fp4")
    wq2d_w = wq.reshape(rows, in_f) if wq is not None else None
    m2 = _calibrate_m2_used(w2d, wq2d_w, "fp4", mode, cb, wq_period)
    wcol = wq2d_w if wq2d_w is not None else torch.ones_like(w2d)
    s0 = (_group_reduce(wcol * w2d * w2d, "fp4")
          / (_group_reduce(wcol, "fp4") * m2).clamp_min(1e-30)).sqrt()
    refits = _TIER_REFITS[tier]
    e_lo, e_hi, W = _two_tier_window(
        amax, ideal=s0, pad=1 if tier == "balanced" else 0)
    G = amax.shape[1]
    gps = SUPERBLOCK // FP4_GROUP
    vec_per_row = in_f // VEC_DIM
    vec_per_group = FP4_GROUP // VEC_DIM
    n_ent = len(TWO_TIER_SUB_TABLE)
    inf = torch.tensor(float("inf"), device=w2d.device)

    best_e = e_lo.clone()
    best_s = torch.full((rows, G), _E4M3_MIN_POS, device=w2d.device)
    best_c = torch.zeros(rows, G, dtype=torch.int64, device=w2d.device)
    rows_step = _moment_rows_step(cb, vec_per_row)
    wq2d = wq.reshape(rows, in_f) if wq is not None else None
    for r0 in range(0, rows, rows_step):
        r1 = min(rows, r0 + rows_step)
        rc = r1 - r0
        wvec = w2d[r0:r1].reshape(-1, VEC_DIM)
        wqc = (wq2d[r0:r1].reshape(-1, VEC_DIM)
               if wq2d is not None else None)
        moms = _chunk_moments(wvec, wqc, mode, cb, wq_period)

        def entry_err_all(E):
            """All n_ent entries in ONE batched moment pass. Returns
            (err (rc, G, n_ent) inf-masked, s_g (rc, G, n_ent)). Values are
            the same per-(element, entry) arithmetic as the scalar path; the
            16-entry python loop was 2ms-kernel launch-bound (33.6s of a
            46.8s E=24 encode, 2026-07-19)."""
            comp = compose[E]                              # (rc, n_sb, n_ent)
            leg = legal[E]
            s_sb = torch.where(leg, comp, torch.ones_like(comp))
            s_g = (s_sb.unsqueeze(2).expand(rc, n_sb, gps, n_ent)
                   .reshape(rc, G, n_ent))
            err_g = _moment_err_groups_batched(moms, s_g, vec_per_group)
            leg_g = (leg.unsqueeze(2).expand(rc, n_sb, gps, n_ent)
                     .reshape(rc, G, n_ent))
            return torch.where(leg_g, err_g, inf), s_g

        # Phase 1 — E per superblock by total error (running strict-min in
        # window order, matching the exact path; min over entries is
        # order-free so the batched min is value-identical).
        lo, hi = e_lo[r0:r1], e_hi[r0:r1]
        best_tot = torch.full((rc, n_sb), float("inf"), device=w2d.device)
        for i in range(W):
            E = torch.minimum(lo + i, hi)
            err_best_g = entry_err_all(E)[0].min(dim=-1).values
            tot = err_best_g.reshape(rc, n_sb, gps).sum(-1)
            tot = torch.where((lo + i) <= hi, tot, inf)
            better = tot < best_tot
            best_tot = torch.where(better, tot, best_tot)
            best_e[r0:r1] = torch.where(better, E, best_e[r0:r1])

        # Phase 2 — per-group entry at the frozen E. torch.min's documented
        # first-occurrence tie rule IS the sequential strict-<, first-legal
        # rule (candidates scanned in c order from an inf init).
        Eb = best_e[r0:r1]
        err_all, s_all = entry_err_all(Eb)
        vals, idx = err_all.min(dim=-1)
        finite = vals < inf
        best_s[r0:r1] = torch.where(
            finite, torch.gather(s_all, -1, idx.unsqueeze(-1)).squeeze(-1),
            best_s[r0:r1])
        best_c[r0:r1] = torch.where(finite, idx, best_c[r0:r1])

    # Phase 3 — exact WLS refits on the frozen-E reachable set.
    #
    # State (err, enc, grid-decode) is CARRIED across refits instead of being
    # re-derived by re-evaluating at ``best_s`` each iteration. `_eval_candidate`
    # is group-local in every step — the per-element scale comes from the
    # element's own group, an 8-wide codeword never straddles a group-16
    # boundary, and the error reduction is per group — so evaluating at the
    # per-group-merged scale is EXACTLY the per-group selection between the two
    # evaluations already in hand. That drops the exact evaluations from
    # 2*refits+1 (5 at balanced) to refits+1 (3) with byte-identical output.
    reach = compose[best_e]
    reach_legal = legal[best_e]
    reps = SUPERBLOCK // FP4_GROUP
    reach_g = reach.unsqueeze(2).expand(rows, n_sb, reps, -1)
    legal_g = reach_legal.unsqueeze(2).expand(rows, n_sb, reps, -1)
    err_cur, enc, g = _eval_candidate(w2d, wq, best_s, "fp4", mode, cb,
                                      wq_period)
    for _ in range(int(refits)):
        wcol = wq.reshape(rows, in_f) if wq is not None else torch.ones_like(g)
        num = _group_reduce(wcol * g * w2d, "fp4")
        den = _group_reduce(wcol * g * g, "fp4")
        s_star = torch.where(den > 0, num / den.clamp_min(1e-30), best_s)
        dist = (s_star.reshape(rows, n_sb, reps, 1) - reach_g).abs()
        dist = torch.where(legal_g, dist, inf)
        c_star = dist.argmin(dim=-1)
        s_snap = torch.gather(reach_g, -1, c_star.unsqueeze(-1)).squeeze(-1)
        s_snap = s_snap.reshape(rows, -1)
        err_star, enc_star, g_star = _eval_candidate(
            w2d, wq, s_snap, "fp4", mode, cb, wq_period)
        better = err_star < err_cur
        best_s = torch.where(better, s_snap, best_s)
        best_c = torch.where(better, c_star.reshape(rows, -1), best_c)
        err_cur = torch.where(better, err_star, err_cur)
        g = torch.where(better.repeat_interleave(FP4_GROUP, dim=1), g_star, g)
        bvec = better.repeat_interleave(
            FP4_GROUP // VEC_DIM, dim=1).reshape(-1)
        for key in enc:
            cur, star = enc[key], enc_star[key]
            mask = bvec if cur.dim() == 1 else bvec.reshape(
                (-1,) + (1,) * (cur.dim() - 1)).expand_as(cur)
            enc[key] = torch.where(mask, star, cur)
    return best_s, enc, best_e, best_c


def _fields_block(w2d: torch.Tensor, k: int, grid: str, mode: str,
                  cb, cw2d: torch.Tensor | None, scale_sweep: bool,
                  scale_coding: str = SCALE_CODING_V1,
                  encode_tier: str = "max",
                  cw_row_broadcast: bool = False,
                  warm_scale_state: dict[str, torch.Tensor] | None = None,
                  ) -> dict:
    rows, in_f = w2d.shape
    nvec_per_row = in_f // VEC_DIM
    wq = _col_weight_vectors(cw2d) if cw2d is not None else None
    # A per-input-column imatrix repeats every ``nvec_per_row`` weight vectors,
    # so the scale-independent moments built from it do too. Only claim the
    # period when the caller proved every row of cw2d came from one vector.
    wq_period = (nvec_per_row
                 if (wq is not None and cw_row_broadcast and rows > 1)
                 else None)
    if warm_scale_state is not None:
        # Warm state is only the sweep argmin.  Assignment still runs through
        # the ordinary weighted VQ evaluator, and assembly still runs through
        # the ordinary packer; no codeword/index bytes are trusted or reused.
        scales = warm_scale_state["scales"].to(
            device=w2d.device, dtype=torch.float32
        )
        _, enc, _ = _eval_candidate(
            w2d, wq, scales, grid, mode, cb, wq_period
        )
        if scale_coding == SCALE_CODING_TWO_TIER:
            super_e = warm_scale_state["scale_super"].to(w2d.device)
            sub_c = warm_scale_state["scale_sub"].to(w2d.device)
    elif scale_coding == SCALE_CODING_TWO_TIER:
        if grid != "fp4":
            raise ValueError("two-tier scale coding is fp4-family only "
                             "(fp8 has no per-superblock scale plane)")
        if not scale_sweep:
            raise ValueError("two-tier scale coding IS the sweep encoder "
                             "(spec §1.4); scale_sweep=False is undefined")
        if encode_tier == "max":
            scales, enc, super_e, sub_c = _sweep_encode_two_tier(
                w2d, mode, cb, wq, wq_period)
        else:
            scales, enc, super_e, sub_c = _sweep_encode_two_tier_moment(
                w2d, mode, cb, wq, encode_tier, wq_period)
    elif scale_sweep:
        if encode_tier == "max":
            scales, enc = _sweep_encode(w2d, grid, mode, cb, wq, wq_period)
        else:
            scales, enc = _sweep_encode_moment(
                w2d, grid, mode, cb, wq, encode_tier, wq_period)
    else:
        vectors, scales, _ = _scale_and_vectorize(w2d, grid)
        enc = _mode_encode(vectors, mode, cb, wq, wq_period)
    out = _enc_to_fields(enc, mode, cb, rows, in_f, nvec_per_row)
    out["scales"] = scales
    if scale_coding == SCALE_CODING_TWO_TIER:
        out["scale_super"] = super_e.to(torch.uint8)
        out["scale_sub"] = sub_c
    return out


def nvfp4_cb_fields(w: torch.Tensor, k: int, *, grid: str = "fp4",
                    mode: str = "product",
                    col_weights: torch.Tensor | None = None,
                    codebook: torch.Tensor | tuple | None = None,
                    scale_sweep: bool = True,
                    scale_coding: str = SCALE_CODING_V1,
                    encode_tier: str | None = None,
                    warm_scale_state: dict[str, torch.Tensor] | None = None,
                    ) -> dict:
    """Quantize ``w`` (2-D or 3-D stacked experts) into VQ fields.

    ``scale_sweep`` (default True) jointly optimizes the per-group scale over
    the E4M3-legal candidate grid (IQ-rendering parity); set False for the
    one-shot amax/grid-max scale (A/B and the pre-c3f8c6d rendering).

    ``encode_tier``: fast / balanced / max speed-accuracy tier (None reads
    ``PRISMAQUANT_CB_ENCODE_TIER``, default balanced). max reproduces the
    original sweep bit-identically; see docs/lanes/nvfp4-cb/encode_tiers.md.

    ``scale_coding``: ``"v1"`` (default; bare e4m3 plane) or ``"two_tier"``
    (layout v2, fp4 only: per-superblock E8M0 super + 4-bit sub codes; the
    stored plane is still the composed E4M3-exact per-group scale). The low-
    level codec defaults to v1 for read compatibility; production callers bind
    an explicit serialization context and select v2.

    Returns at least {"indices", "scales"}; the resolved codebook is echoed
    back under "codebook" so reconstruct and the packer share one table.
    """
    in_f = int(w.shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    if scale_coding not in (SCALE_CODING_V1, SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    tier = _resolve_encode_tier(encode_tier)
    orig_shape = tuple(w.shape)
    w2d = w.reshape(-1, in_f)
    rows = w2d.shape[0]
    cb = _resolve_codebook(k, grid, mode, codebook, w2d.device)

    warm_state = None
    if warm_scale_state is not None:
        # File-level validation lives in cb_warm_state.  These shape checks
        # are the codec boundary's final defence for direct library callers.
        groups = in_f // FP4_GROUP if grid == "fp4" else 1
        expected = (rows, groups)
        scales = torch.as_tensor(warm_scale_state.get("scales"))
        if tuple(scales.shape) != expected:
            raise ValueError(
                f"warm scales shape {tuple(scales.shape)} != {expected}"
            )
        warm_state = {"scales": scales}
        if scale_coding == SCALE_CODING_TWO_TIER:
            n_sb = in_f // SUPERBLOCK
            super_e = torch.as_tensor(warm_scale_state.get("scale_super"))
            sub_c = torch.as_tensor(warm_scale_state.get("scale_sub"))
            if tuple(super_e.shape) != (rows, n_sb):
                raise ValueError("warm two-tier super-scale shape differs")
            if tuple(sub_c.shape) != expected:
                raise ValueError("warm two-tier sub-scale shape differs")
            warm_state.update(scale_super=super_e, scale_sub=sub_c)

    # col_weights stays a broadcast VIEW; blocks materialize only their rows
    # (a full-shape fp32 copy is another ~10GB on a Hy3 expert stack).
    cw_view = None
    cw_row_broadcast = False
    if col_weights is not None:
        cw_view = torch.broadcast_to(
            col_weights.to(w2d.device, torch.float32), orig_shape)
        # Every leading (row) dim stride 0 <=> one per-input-column vector
        # replicated across rows, which is what the production imatrix is.
        # Only then do the col-weight moments repeat per row (see
        # _stream_moments/_vq_assign ``wq_period``).
        cw_row_broadcast = all(
            cw_view.stride(d) == 0 for d in range(cw_view.dim() - 1))

    def _cw_rows(a: int, b: int) -> torch.Tensor | None:
        if cw_view is None:
            return None
        idx = torch.unravel_index(
            torch.arange(a, b, device=w2d.device), orig_shape[:-1])
        return cw_view[idx]                                  # (b-a, in_f)

    row_step = max(1, _SLICE_MAX_ELEMS // max(in_f, 1))

    def _warm_rows(a: int, b: int):
        if warm_state is None:
            return None
        return {key: value[a:b] for key, value in warm_state.items()}

    if rows <= row_step:
        out = _fields_block(w2d, k, grid, mode, cb, _cw_rows(0, rows),
                            scale_sweep, scale_coding, tier,
                            cw_row_broadcast, _warm_rows(0, rows))
    else:
        parts = []
        for a in range(0, rows, row_step):
            b = min(rows, a + row_step)
            parts.append(
                _fields_block(w2d[a:b], k, grid, mode, cb, _cw_rows(a, b),
                              scale_sweep, scale_coding, tier,
                              cw_row_broadcast, _warm_rows(a, b)))
        out = {key: torch.cat([p[key] for p in parts], dim=0)
               for key in parts[0]}
    out["shape"] = orig_shape
    out["codebook"] = cb
    if scale_coding == SCALE_CODING_TWO_TIER:
        out["scale_coding"] = SCALE_CODING_TWO_TIER
    return out


LDLQ_BLOCK_SIZE = 64
LDLQ_DAMPING_FRACTION = 0.01
# Production bound: one packed projection has at most 256 experts (DeepSeek-V4-Flash);
# caching one factor per expert is the useful across-rung reuse. 256 already holds
# ~16 GiB (256 * 64 MiB). 512 would retain 32 GiB across projections and is never
# needed within one projection. Cap at 256 and require explicit clear after each
# packed projection so torch.cuda.empty_cache can actually release.
_LDLQ_FACTOR_CACHE_MAX = 256
_LDLQ_FACTOR_CACHE: OrderedDict[tuple, tuple[weakref.ReferenceType, torch.Tensor]] = (
    OrderedDict()
)
_LDLQ_FACTOR_CACHE_LOCK = threading.Lock()


def clear_ldlq_factor_cache() -> int:
    """Clear the LDLQ inverse-factor cache (production-safe, locked).

    Returns the number of entries cleared. Useful after one packed projection
    (including exception paths) and before dense work so CUDA cache can be
    released. Thread-safe via _LDLQ_FACTOR_CACHE_LOCK.
    """
    with _LDLQ_FACTOR_CACHE_LOCK:
        n = len(_LDLQ_FACTOR_CACHE)
        _LDLQ_FACTOR_CACHE.clear()
        return n


def ldlq_factor_cache_size() -> int:
    """Return current LDLQ factor cache size (for tests/monitoring)."""
    with _LDLQ_FACTOR_CACHE_LOCK:
        return len(_LDLQ_FACTOR_CACHE)


def _ldlq_inverse_factor_cached(
    activation_rows: torch.Tensor,
    *,
    device: torch.device,
    damping_fraction: float,
) -> torch.Tensor:
    """Reuse the exact format-independent factor across adjacent CB rungs."""
    source = torch.as_tensor(activation_rows)
    key = (
        id(source),
        source.data_ptr(),
        source.storage_offset(),
        tuple(source.shape),
        tuple(source.stride()),
        source.device,
        source.dtype,
        device,
        float(damping_fraction),
    )
    with _LDLQ_FACTOR_CACHE_LOCK:
        cached = _LDLQ_FACTOR_CACHE.get(key)
        if cached is not None and cached[0]() is source:
            _LDLQ_FACTOR_CACHE.move_to_end(key)
            return cached[1]
        if cached is not None:
            del _LDLQ_FACTOR_CACHE[key]

    from .rotation_ldlq_pilot import inverse_hessian_cholesky

    x = source.to(device=device, dtype=torch.float32)
    factor = inverse_hessian_cholesky(
        x.T @ x,
        damping_fraction=float(damping_fraction),
    )[0]
    with _LDLQ_FACTOR_CACHE_LOCK:
        _LDLQ_FACTOR_CACHE[key] = (weakref.ref(source), factor)
        _LDLQ_FACTOR_CACHE.move_to_end(key)
        if len(_LDLQ_FACTOR_CACHE) > _LDLQ_FACTOR_CACHE_MAX:
            dead = [
                old_key for old_key, (reference, _value)
                in _LDLQ_FACTOR_CACHE.items() if reference() is None
            ]
            for old_key in dead:
                del _LDLQ_FACTOR_CACHE[old_key]
        while len(_LDLQ_FACTOR_CACHE) > _LDLQ_FACTOR_CACHE_MAX:
            _LDLQ_FACTOR_CACHE.popitem(last=False)
    return factor


def _ldlq_reassign_fields_2d(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
) -> dict:
    """Replace only fixed-codebook assignments using block Hessian feedback."""
    from .rotation_ldlq_pilot import block_error_feedback

    if weight.ndim != 2:
        raise ValueError(f"LDLQ weight must be 2-D, got {tuple(weight.shape)}")
    rows, columns = map(int, weight.shape)
    x = torch.as_tensor(activation_rows)
    if x.ndim != 2 or int(x.shape[1]) != columns:
        raise ValueError(
            "LDLQ activation rows must have shape (rows, in_features), got "
            f"{tuple(x.shape)} for weight {tuple(weight.shape)}"
        )
    if int(x.shape[0]) == 0:
        raise ValueError("LDLQ activation rows must be non-empty")
    if columns % int(block_size) or int(block_size) % FP4_GROUP:
        raise ValueError(
            f"LDLQ block_size={block_size} must divide in_features={columns} "
            f"and preserve group-{FP4_GROUP} scales"
        )

    upper = _ldlq_inverse_factor_cached(
        x,
        device=weight.device,
        damping_fraction=float(damping_fraction),
    )

    scales = fields["scales"].to(weight.device, torch.float32)
    codebook = fields["codebook"]
    if isinstance(codebook, tuple):
        codebook = tuple(table.to(weight.device, torch.float32) for table in codebook)
    else:
        codebook = codebook.to(weight.device, torch.float32)
    cw = torch.broadcast_to(
        torch.as_tensor(col_weights).to(weight.device, torch.float32),
        weight.shape,
    )
    assignment_parts: list[dict[str, torch.Tensor]] = []

    def quantize_block(block: torch.Tensor, start: int, end: int) -> torch.Tensor:
        width = end - start
        block_scales = (
            scales[:, start // FP4_GROUP:end // FP4_GROUP]
            if grid == "fp4"
            else scales
        )
        wq = _col_weight_vectors(cw[:, start:end])
        _err, enc, decoded = _eval_candidate(
            block.to(torch.float32),
            wq,
            block_scales,
            grid,
            mode,
            codebook,
        )
        assignment_parts.append(
            _enc_to_fields(enc, mode, codebook, rows, width, width // VEC_DIM)
        )
        return decoded * _per_element_scale(block_scales, grid, width)

    # The returned reconstruction is intentionally discarded: export needs
    # the assignments, and reconstructing those fields is the shared decoder.
    block_error_feedback(
        weight,
        upper,
        quantize_block,
        block_size=int(block_size),
    )
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=1
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=1
        )
    return updated


def _ldlq_reassign_fields_3d_batched(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
) -> dict:
    """Vectorize independent expert LDLQ solves over a batch dimension.

    Experts remain independent and the column-block loop retains the serial
    path's exact within-expert order.  Fixed-codebook assignment, triangular
    block solve, and feedback update are batched over the expert axis.  The
    inverse-Hessian factors remain the serial bit-identity anchors: CUDA's
    stacked Cholesky chooses a different numerical kernel and changes real
    expert indices at exact VQ boundaries.  Repeated cold-prior inputs share
    one exact factor, so identical work is still deduplicated.
    """
    if weight.ndim != 3:
        raise ValueError(
            f"batched LDLQ weight must be 3-D, got {tuple(weight.shape)}"
        )
    experts, rows, columns = map(int, weight.shape)
    activations = tuple(activation_rows)
    if len(activations) != experts:
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {experts}"
        )
    if columns % int(block_size) or int(block_size) % FP4_GROUP:
        raise ValueError(
            f"LDLQ block_size={block_size} must divide in_features={columns} "
            f"and preserve group-{FP4_GROUP} scales"
        )

    expert_batch = _resolve_ldlq_expert_batch()
    if experts > expert_batch:
        # A single E=256 launch makes the tall assignment and feedback GEMMs
        # slower on GB10 than several resident same-shape batches.  Chunking
        # changes only the independent expert batch dimension; column blocks
        # and every operation within one expert retain their original order.
        indices = fields["indices"].reshape(experts * rows, -1)
        signs = fields.get("signs")
        if signs is not None:
            signs = signs.reshape(experts * rows, -1)
        scales = fields["scales"].reshape(experts * rows, -1)
        chunk_ranges = [
            (first, min(first + expert_batch, experts))
            for first in range(0, experts, expert_batch)
        ]

        def encode_chunk(first: int, last: int) -> dict:
            last = min(first + expert_batch, experts)
            row_first, row_last = first * rows, last * rows
            chunk_fields = dict(fields)
            chunk_fields["indices"] = indices[row_first:row_last]
            chunk_fields["scales"] = scales[row_first:row_last]
            if signs is not None:
                chunk_fields["signs"] = signs[row_first:row_last]
            chunk_fields["shape"] = (last - first, rows, columns)
            return _ldlq_reassign_fields_3d_batched(
                weight[first:last],
                chunk_fields,
                torch.broadcast_to(
                    torch.as_tensor(col_weights), weight.shape
                )[first:last],
                activations[first:last],
                grid=grid,
                mode=mode,
                block_size=block_size,
                damping_fraction=damping_fraction,
            )

        raw_streams = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS", "1"
        ).strip()
        try:
            batch_streams = int(raw_streams)
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS must be a positive integer"
            ) from exc
        if batch_streams <= 0:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_STREAMS must be a positive integer"
            )
        chunk_results: list[dict | None] = [None] * len(chunk_ranges)
        if batch_streams > 1 and weight.device.type == "cuda":
            from concurrent.futures import ThreadPoolExecutor

            stream_count = min(batch_streams, len(chunk_ranges))
            streams = [
                torch.cuda.Stream(device=weight.device)
                for _ in range(stream_count)
            ]

            def encode_stream(stream_id: int) -> list[tuple[int, dict]]:
                encoded: list[tuple[int, dict]] = []
                with torch.cuda.device(weight.device), torch.cuda.stream(
                    streams[stream_id]
                ):
                    for chunk_id in range(
                        stream_id, len(chunk_ranges), stream_count
                    ):
                        first, last = chunk_ranges[chunk_id]
                        encoded.append(
                            (chunk_id, encode_chunk(first, last))
                        )
                return encoded

            with ThreadPoolExecutor(max_workers=stream_count) as pool:
                for encoded in pool.map(encode_stream, range(stream_count)):
                    for chunk_id, result in encoded:
                        chunk_results[chunk_id] = result
            current = torch.cuda.current_stream(weight.device)
            for stream in streams:
                current.wait_stream(stream)
        else:
            for chunk_id, (first, last) in enumerate(chunk_ranges):
                chunk_results[chunk_id] = encode_chunk(first, last)
        ready = [result for result in chunk_results if result is not None]
        if len(ready) != len(chunk_ranges):
            raise RuntimeError("batched LDLQ stream lost an expert chunk")
        updated = dict(fields)
        updated["indices"] = torch.cat(
            [result["indices"] for result in ready], dim=0
        )
        if signs is not None:
            updated["signs"] = torch.cat(
                [result["signs"] for result in ready], dim=0
            )
        return updated

    xs: list[torch.Tensor] = []
    for x in activations:
        x = torch.as_tensor(x)
        if x.ndim != 2 or int(x.shape[1]) != columns:
            raise ValueError(
                "LDLQ activation rows must have shape (rows, in_features), got "
                f"{tuple(x.shape)} for expert weight {(rows, columns)}"
            )
        if int(x.shape[0]) == 0:
            raise ValueError("LDLQ activation rows must be non-empty")
        xs.append(x)

    upper_parts = [
        _ldlq_inverse_factor_cached(
            x,
            device=weight.device,
            damping_fraction=float(damping_fraction),
        )
        for x in xs
    ]
    upper = torch.stack(upper_parts)
    del xs, upper_parts

    scales = fields["scales"].to(weight.device, torch.float32).reshape(
        experts, rows, -1
    )
    codebook = fields["codebook"]
    if isinstance(codebook, tuple):
        codebook = tuple(
            table.to(weight.device, torch.float32) for table in codebook
        )
    else:
        codebook = codebook.to(weight.device, torch.float32)
    cw = torch.broadcast_to(
        torch.as_tensor(col_weights).to(weight.device, torch.float32),
        weight.shape,
    )
    work = weight.to(torch.float32).clone()
    assignment_parts: list[dict[str, torch.Tensor]] = []

    for start in range(0, columns, int(block_size)):
        end = start + int(block_size)
        width = end - start
        block = work[:, :, start:end]
        block_scales = (
            scales[:, :, start // FP4_GROUP:end // FP4_GROUP]
            if grid == "fp4"
            else scales
        )
        flat_block = block.reshape(experts * rows, width)
        flat_scales = block_scales.reshape(experts * rows, -1)
        wq = _col_weight_vectors(
            cw[:, :, start:end].reshape(experts * rows, width)
        )
        _err, enc, decoded = _eval_candidate(
            flat_block,
            wq,
            flat_scales,
            grid,
            mode,
            codebook,
        )
        assignment_parts.append(
            _enc_to_fields(
                enc,
                mode,
                codebook,
                experts * rows,
                width,
                width // VEC_DIM,
            )
        )
        qblock = (
            decoded
            * _per_element_scale(flat_scales, grid, width)
        ).reshape(experts, rows, width)
        residual = block - qblock
        diagonal_block = upper[:, start:end, start:end]
        scaled_error = torch.linalg.solve_triangular(
            diagonal_block.transpose(-2, -1),
            residual.transpose(-2, -1),
            upper=False,
        ).transpose(-2, -1)
        work[:, :, start:] -= torch.bmm(
            scaled_error,
            upper[:, start:end, start:],
        )

    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=1
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=1
        )
    return updated


def _ldlq_reassign_fields_3d_threaded(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int,
    damping_fraction: float,
    workers: int,
) -> dict:
    """Feed exact per-expert LDLQ streams from multiple host threads.

    This is the secondary lever for rungs whose large fixed-codebook search
    does not benefit from flattening experts into one assignment batch.  One
    expert still executes the byte-pinned 2-D path, on one CUDA stream, in
    exactly the legacy order; threads only make independent units resident
    concurrently so a single Python core cannot starve the device.
    """
    from concurrent.futures import ThreadPoolExecutor

    if weight.device.type != "cuda":
        raise ValueError("threaded LDLQ feeder requires CUDA expert weights")
    experts, rows, columns = map(int, weight.shape)
    activations = tuple(activation_rows)
    if len(activations) != experts:
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {experts}"
        )
    workers = min(max(1, int(workers)), experts)
    indices = fields["indices"].reshape(experts * rows, -1)
    scales = fields["scales"].reshape(experts * rows, -1)
    signs = fields.get("signs")
    if signs is not None:
        signs = signs.reshape(experts * rows, -1)
    cw = torch.broadcast_to(torch.as_tensor(col_weights), weight.shape)
    streams = [torch.cuda.Stream(device=weight.device) for _ in range(workers)]

    def encode_worker(worker: int) -> list[tuple[int, dict]]:
        encoded: list[tuple[int, dict]] = []
        stream = streams[worker]
        with torch.cuda.device(weight.device), torch.cuda.stream(stream):
            for expert in range(worker, experts, workers):
                first, last = expert * rows, (expert + 1) * rows
                expert_fields = dict(fields)
                expert_fields["indices"] = indices[first:last]
                expert_fields["scales"] = scales[first:last]
                if signs is not None:
                    expert_fields["signs"] = signs[first:last]
                expert_fields["shape"] = (rows, columns)
                result = _ldlq_reassign_fields_2d(
                    weight[expert],
                    expert_fields,
                    cw[expert],
                    activations[expert],
                    grid=grid,
                    mode=mode,
                    block_size=block_size,
                    damping_fraction=damping_fraction,
                )
                encoded.append((expert, result))
        return encoded

    results: list[dict | None] = [None] * experts
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for encoded in pool.map(encode_worker, range(workers)):
            for expert, result in encoded:
                results[expert] = result
    current = torch.cuda.current_stream(weight.device)
    for stream in streams:
        current.wait_stream(stream)
    ready = [result for result in results if result is not None]
    if len(ready) != experts:
        raise RuntimeError("threaded LDLQ feeder lost an expert result")
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [result["indices"] for result in ready], dim=0
    )
    if signs is not None:
        updated["signs"] = torch.cat(
            [result["signs"] for result in ready], dim=0
        )
    return updated


LDLQ_GATE_ENV = "PRISMAQUANT_CB_LDLQ_GATE"
LDLQ_GATE_EPSILON = 1e-12


def _ldlq_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    raw = str(values.get(LDLQ_GATE_ENV, "1")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{LDLQ_GATE_ENV} must be a boolean 0/1 setting, got {raw!r}")


def _ldlq_weighted_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    col_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-row col-weighted MSE, matching the cost's activation-aware branch."""
    err = (weight.to(torch.float32) - reconstruction.to(torch.float32)).pow(2)
    if col_weights is not None:
        cw = torch.broadcast_to(
            torch.as_tensor(col_weights).to(weight.device, torch.float32),
            weight.shape,
        ).to(torch.float32)
        # Match _col_weight_vectors dead-vector guard: zero-mass -> unweighted.
        if err.dim() == 2:
            mass = cw.sum(dim=-1, keepdim=True)
            cw = torch.where(mass == 0, torch.ones_like(cw), cw)
        elif err.dim() == 3:
            mass = cw.sum(dim=-1, keepdim=True)
            cw = torch.where(mass == 0, torch.ones_like(cw), cw)
        err = err * cw
        denom = cw.sum(dim=-1).clamp_min(1e-30).mean().clamp_min(1e-30)
        # Use mean over all elements weighted by cw; denom above keeps scale
        # stable when some rows have zero mass. For per-expert gating we need
        # per-slice value, so caller handles slicing.
        return err.mean()
    return err.mean()


def _ldlq_per_expert_weighted_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    col_weights: torch.Tensor,
) -> list[float]:
    """Return per-expert col-weighted MSE for a 3-D stack or single Linear."""
    if weight.ndim == 2:
        return [_ldlq_weighted_mse(weight, reconstruction, col_weights).item()]
    # 3-D: (E, R, C)
    values: list[float] = []
    for idx in range(int(weight.shape[0])):
        w = weight[idx]
        r = reconstruction[idx] if reconstruction.ndim == 3 else reconstruction
        cw = col_weights[idx] if col_weights.ndim == 3 else col_weights
        # col_weights for experts is (E,1,C) broadcast; slice matches.
        if cw is not None and cw.ndim == 3:
            cw_slice = cw[idx] if cw.shape[0] == weight.shape[0] else cw
        else:
            cw_slice = cw
        err = (w.to(torch.float32) - r.to(torch.float32)).pow(2)
        if cw_slice is not None:
            cw_b = torch.broadcast_to(
                torch.as_tensor(cw_slice).to(w.device, torch.float32), w.shape
            ).to(torch.float32)
            mass = cw_b.sum(dim=-1, keepdim=True)
            cw_b = torch.where(mass == 0, torch.ones_like(cw_b), cw_b)
            err = err * cw_b
        values.append(float(err.mean().item()))
    return values


def _ldlq_activation_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor] | None,
) -> float | None:
    """Activation-weighted output MSE, or None if no activation rows.

    This is the declared gate metric ``activation_output_mse``.  It is fail-
    closed on malformed rows: a non-2-D tensor, a width mismatch, or any
    other structural error raises immediately so the caller can record an
    explicit fallback reason or abort, rather than silently falling back to
    a different metric.
    """
    if activation_rows is None:
        return None
    if isinstance(activation_rows, torch.Tensor):
        act = torch.as_tensor(activation_rows)
        if act.numel() == 0 or act.shape[0] == 0:
            return None
        if act.ndim != 2:
            raise ValueError(f"activation rows must be rank-2, got shape {tuple(act.shape)}")
        if int(act.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"activation width {act.shape[1]} != weight in_features {weight.shape[-1]}"
            )
        if weight.ndim == 3:
            total = 0.0
            for idx in range(int(weight.shape[0])):
                w = weight[idx].to(torch.float32)
                r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
                err = (act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item()
                total += float(err)
            return total / max(int(weight.shape[0]), 1)
        w = weight.to(torch.float32)
        r = reconstruction.to(torch.float32)
        return float((act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
    # Sequence per expert
    seq = tuple(activation_rows)
    if not seq:
        return None
    # Check for any empty or malformed entry; empty is not an error but a
    # missing-data signal that the caller must handle explicitly.
    has_data = False
    for act in seq:
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            continue
        has_data = True
        if act_t.ndim != 2:
            raise ValueError(f"per-expert activation rows must be rank-2, got {tuple(act_t.shape)}")
        if int(act_t.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"per-expert activation width {act_t.shape[1]} != weight in_features {weight.shape[-1]}"
            )
    if not has_data:
        return None
    if weight.ndim == 2:
        act = torch.as_tensor(seq[0])
        if act.numel() == 0:
            return None
        w = weight.to(torch.float32)
        r = reconstruction.to(torch.float32)
        return float((act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
    total = 0.0
    count = 0
    for idx, act in enumerate(seq):
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            continue
        w = weight[idx].to(torch.float32)
        r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
        total += float((act_t.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
        count += 1
    return total / max(count, 1) if count else None


def _ldlq_per_expert_activation_mse(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
) -> list[float] | None:
    if weight.ndim != 3:
        return None
    seq = tuple(activation_rows)
    if len(seq) != int(weight.shape[0]):
        raise ValueError(
            f"per-expert activation count {len(seq)} != stack size {weight.shape[0]}"
        )
    values: list[float] = []
    for idx, act in enumerate(seq):
        act_t = torch.as_tensor(act)
        if act_t.numel() == 0 or act_t.shape[0] == 0:
            # Missing rows for this expert is not silently inf; the caller
            # must decide to fallback per expert with an explicit reason.
            # We return inf as a sentinel that the caller will interpret as
            # missing, but we do not swallow malformed shapes.
            values.append(float("inf"))
            continue
        if act_t.ndim != 2:
            raise ValueError(f"per-expert activation rows must be rank-2, got {tuple(act_t.shape)} for expert {idx}")
        if int(act_t.shape[1]) != int(weight.shape[-1]):
            raise ValueError(
                f"per-expert activation width {act_t.shape[1]} != weight in_features {weight.shape[-1]} for expert {idx}"
            )
        w = weight[idx].to(torch.float32)
        r = reconstruction[idx].to(torch.float32) if reconstruction.ndim == 3 else reconstruction.to(torch.float32)
        err = float((act_t.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
        values.append(err)
    return values


def _per_expert_activation_mse_chunked_from_fields(
    weight: torch.Tensor,
    fields: dict,
    k: int,
    grid: str,
    mode: str,
    activation_rows: Sequence[torch.Tensor],
) -> list[float]:
    """Chunked per-expert activation_output_mse without materializing full recon.

    Remains GPU-bound: each expert chunk is decoded on GPU and its mse is
    computed immediately, releasing the chunk before the next. Result is
    bit-identical to the gather-then-reconstruct path because each expert's
    reconstruction is independent per row.
    """
    if weight.ndim != 3:
        raise ValueError("chunked per-expert mse requires 3-D weight")
    E, R, C = map(int, weight.shape)
    seq = tuple(activation_rows)
    if len(seq) != E:
        raise ValueError(f"per-expert activation count {len(seq)} != stack size {E}")
    values: list[float] = []
    for (first, last), chunk_recon, _chunk_fields in iter_nvfp4_cb_recon_chunks(
        fields, k, grid=grid, mode=mode
    ):
        # chunk_recon is (chunk_E, R, C) float32 on same device as fields
        for idx in range(first, last):
            local = idx - first
            act_t = torch.as_tensor(seq[idx])
            if act_t.numel() == 0 or act_t.shape[0] == 0:
                values.append(float("inf"))
                continue
            if act_t.ndim != 2:
                raise ValueError(f"per-expert activation rows must be rank-2, got {tuple(act_t.shape)} for expert {idx}")
            if int(act_t.shape[1]) != C:
                raise ValueError(f"per-expert activation width {act_t.shape[1]} != weight in_features {C} for expert {idx}")
            w = weight[idx].to(torch.float32)
            r = chunk_recon[local].to(torch.float32)
            err = float((act_t.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
            values.append(err)
        del chunk_recon, _chunk_fields
        # Prompt release of temporaries per chunk (GPU-bound, not CPU/NVMe)
        if weight.device.type == "cuda":
            # Do not empty_cache aggressively; let allocator reuse, but ensure chunk freed
            pass
    return values


def _dense_activation_mse_chunked_from_fields(
    weight: torch.Tensor,
    fields: dict,
    k: int,
    grid: str,
    mode: str,
    activation_rows: torch.Tensor,
) -> float | None:
    """Chunked activation_output_mse for 2-D dense weight.

    Splits the output-row dimension (R) into chunks bounded by
    PRISMAQUANT_CB_RECON_ROW_CHUNK so the (N, R) matmul never materializes
    the full 16 GiB intermediate. Aggregates mean exactly (sum/chunk counts).
    """
    if weight.ndim != 2:
        raise ValueError("dense chunked mse requires 2-D weight")
    act = torch.as_tensor(activation_rows)
    if act.numel() == 0 or act.shape[0] == 0:
        return None
    if act.ndim != 2:
        raise ValueError(f"activation rows must be rank-2, got shape {tuple(act.shape)}")
    if int(act.shape[1]) != int(weight.shape[-1]):
        raise ValueError(f"activation width {act.shape[1]} != weight in_features {weight.shape[-1]}")
    R, C = map(int, weight.shape)
    N = int(act.shape[0])
    rows = R
    chunk_rows = _resolve_recon_row_chunk()
    if rows <= chunk_rows:
        recon = _nvfp4_cb_reconstruct_one(fields, k, grid=grid, mode=mode)
        w = weight.to(torch.float32)
        r = recon.to(torch.float32)
        return float((act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item())
    # Chunked path: accumulate sum and count
    total_sum = 0.0
    total_elements = 0
    # Need to slice weight and fields consistently
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk_fields = _slice_fields_for_rows(fields, start, end, new_shape=(end - start, C))
        chunk_recon = _nvfp4_cb_reconstruct_one(chunk_fields, k, grid=grid, mode=mode)
        w_chunk = weight[start:end].to(torch.float32)
        r_chunk = chunk_recon.to(torch.float32)
        diff = w_chunk - r_chunk
        # act (N,C) @ diff.T (C, chunk_R) => (N, chunk_R)
        chunk_sum = float((act.to(w_chunk.device, torch.float32) @ diff.T).pow(2).sum().item())
        total_sum += chunk_sum
        total_elements += N * (end - start)
        del chunk_recon, chunk_fields, diff
    return total_sum / max(total_elements, 1)


# ---------------------------------------------------------------------------
# Canonical production-QDQ output MSE (measure_quant_cost semantics)
# Public API: measure_quant_cost, gate, packed/dense derive share this exact
# math without duplication or private imports. Preserves historical W/Q BF16
# rounding. Supports memory-bounded packed (chunked) and dense helpers.
# ---------------------------------------------------------------------------

def canonical_nvfp4_cb_single_output_mse(
    W: torch.Tensor,
    Q: torch.Tensor,
    X: torch.Tensor,
    spec,
) -> float:
    """Public canonical production output MSE for a single (R,C) slice.

    y_ref = X @ W.T, X_hat = spec.activation_quantize_dequantize(X.clone()),
    y_q = X_hat @ Q.T, mean((y_ref - y_q)^2). Preserves BF16 rounding for W/Q.
    """
    if X.numel() == 0 or X.shape[0] == 0:
        raise ValueError("canonical mse requires non-empty activation rows")
    if X.ndim != 2:
        raise ValueError(f"activation rows must be rank-2, got {tuple(X.shape)}")
    if int(X.shape[1]) != int(W.shape[-1]):
        raise ValueError(f"activation width {X.shape[1]} != weight in_features {W.shape[-1]}")
    if W.shape != Q.shape:
        raise ValueError(f"W shape {tuple(W.shape)} != Q shape {tuple(Q.shape)}")
    W_bf16 = W.to(torch.bfloat16) if W.dtype != torch.bfloat16 else W
    Q_bf16 = Q.to(torch.bfloat16) if Q.dtype != torch.bfloat16 else Q
    W_f32 = W_bf16.to(torch.float32)
    Q_f32 = Q_bf16.to(torch.float32)
    X_f32 = X.to(torch.float32)
    X_hat = spec.activation_quantize_dequantize(X_f32.clone())
    if not isinstance(X_hat, torch.Tensor):
        raise ValueError("spec.activation_quantize_dequantize must return Tensor")
    if X_hat.shape != X_f32.shape:
        raise ValueError(f"X_hat shape {tuple(X_hat.shape)} != X shape {tuple(X_f32.shape)}")
    X_hat_f32 = X_hat.to(torch.float32)
    y_ref = X_f32.to(W_f32.device) @ W_f32.T
    y_q = X_hat_f32.to(Q_f32.device) @ Q_f32.T
    return float((y_ref - y_q).pow(2).mean().item())


# Backward compat alias for private name used in tests
_canonical_single_output_mse = canonical_nvfp4_cb_single_output_mse


def canonical_nvfp4_cb_per_expert_mse_chunked(
    weight: torch.Tensor,
    fields: dict,
    k: int,
    grid: str,
    mode: str,
    activation_rows: Sequence[torch.Tensor],
    spec,
) -> list[float]:
    """Public chunked canonical per-expert output MSE without full recon."""
    if weight.ndim != 3:
        raise ValueError("canonical per-expert requires 3-D weight")
    E, R, C = map(int, weight.shape)
    seq = tuple(activation_rows)
    if len(seq) != E:
        raise ValueError(f"per-expert activation count {len(seq)} != stack size {E}")
    values: list[float] = []
    for (first, last), chunk_recon, _chunk_fields in iter_nvfp4_cb_recon_chunks(fields, k, grid=grid, mode=mode):
        for idx in range(first, last):
            local = idx - first
            act_t = torch.as_tensor(seq[idx])
            if act_t.numel() == 0 or act_t.shape[0] == 0:
                values.append(float("inf"))
                continue
            w = weight[idx]
            q = chunk_recon[local]
            values.append(canonical_nvfp4_cb_single_output_mse(w, q, act_t, spec))
        del chunk_recon, _chunk_fields
    return values


_canonical_per_expert_mse_chunked_from_fields = canonical_nvfp4_cb_per_expert_mse_chunked


def canonical_nvfp4_cb_dense_mse_chunked(
    weight: torch.Tensor,
    fields: dict,
    k: int,
    grid: str,
    mode: str,
    activation_rows: torch.Tensor,
    spec,
) -> float | None:
    """Public chunked canonical dense output MSE (splits output rows)."""
    if weight.ndim != 2:
        raise ValueError("canonical dense requires 2-D weight")
    act = torch.as_tensor(activation_rows)
    if act.numel() == 0 or act.shape[0] == 0:
        return None
    if act.ndim != 2:
        raise ValueError(f"activation rows must be rank-2, got {tuple(act.shape)}")
    if int(act.shape[1]) != int(weight.shape[-1]):
        raise ValueError(f"activation width {act.shape[1]} != weight in_features {weight.shape[-1]}")
    R, C = map(int, weight.shape)
    N = int(act.shape[0])
    chunk_rows = _resolve_recon_row_chunk()
    if R <= chunk_rows:
        recon = _nvfp4_cb_reconstruct_one(fields, k, grid=grid, mode=mode)
        return canonical_nvfp4_cb_single_output_mse(weight, recon, act, spec)
    total_sum = 0.0
    total_elements = 0
    for start in range(0, R, chunk_rows):
        end = min(R, start + chunk_rows)
        chunk_fields = _slice_fields_for_rows(fields, start, end, new_shape=(end - start, C))
        chunk_recon = _nvfp4_cb_reconstruct_one(chunk_fields, k, grid=grid, mode=mode)
        w_chunk = weight[start:end]
        q_chunk = chunk_recon
        mse_chunk = canonical_nvfp4_cb_single_output_mse(w_chunk, q_chunk, act, spec)
        total_sum += mse_chunk * (N * (end - start))
        total_elements += N * (end - start)
        del chunk_recon, chunk_fields
    return total_sum / max(total_elements, 1)


_canonical_dense_mse_chunked_from_fields = canonical_nvfp4_cb_dense_mse_chunked

# Deterministic per-expert fit/holdout split (content-hash grouped, no torch.randperm)
_SPLIT_POLICY = "v2_per_expert_deterministic_content_hash_50_50"
_SPLIT_VERSION = 2


def _row_content_hash(row: torch.Tensor) -> str:
    """Stable hash of a single row's float32 bytes."""
    b = row.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def _partition_multiset_digest(t: torch.Tensor) -> str:
    """Deterministic, permutation-invariant, multiplicity-sensitive multiset identity.

    Versioned framing plus sorted per-row normalized content digests, including
    multiplicity. Accounts for shape/dtype boundaries and avoids ambiguous
    concatenation via explicit delimiters and fixed framing. Device transfer
    stable (float32 bytes).
    """
    tt = torch.as_tensor(t).detach().cpu().to(torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(b"prismaquant.ldlq_split.v2|")
    h.update(f"policy={_SPLIT_POLICY}|version={_SPLIT_VERSION}|".encode())
    h.update(f"shape={tuple(tt.shape)}|dtype={str(tt.dtype)}|".encode())
    if tt.numel() == 0 or (tt.ndim == 2 and int(tt.shape[0]) == 0):
        h.update(b"empty|")
        return h.hexdigest()
    if tt.ndim == 2:
        n = int(tt.shape[0])
        row_hashes = [_row_content_hash(tt[i]) for i in range(n)]
        row_hashes.sort()
        h.update(f"n={n}|".encode())
        for rh in row_hashes:
            h.update(rh.encode())
            h.update(b"|")
        return h.hexdigest()
    # Fallback for non-2-D tensors (should not happen for split, but handle shape boundary)
    rh = hashlib.sha256(tt.numpy().tobytes()).hexdigest()
    h.update(b"n=1|")
    h.update(rh.encode())
    h.update(b"|")
    return h.hexdigest()


def deterministic_fit_holdout_split_per_expert(
    rows: Sequence[torch.Tensor],
    *,
    version: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    """Split each expert's original rows into fit/holdout deterministically.

    Groups identical row content so a duplicate never crosses the split.
    Input permutation does not change membership because groups are sorted by
    content hash. Fails raw (empty partitions) if two nonempty independent
    partitions cannot be formed.
    Returns (fit_rows, holdout_rows, split_info_per_expert).

    Digests (full/fit/holdout) are a deterministic, permutation-invariant,
    multiplicity-sensitive identity of the normalized row-content multiset in
    that partition (v2). They include versioned framing, shape/dtype, and
    sorted per-row normalized content digests with explicit delimiters. Duplicates
    remain grouped and affect identity via multiplicity. Device transfer is stable.
    """
    if version not in (1, 2):
        raise ValueError(f"unsupported split version {version}")
    # For v1, preserve legacy order-dependent digest for backward compat tests;
    # all production paths use v2 invariant digest.
    use_invariant = version == 2
    fit_rows: list[torch.Tensor] = []
    holdout_rows: list[torch.Tensor] = []
    infos: list[dict] = []
    for ei, t in enumerate(rows):
        tt = torch.as_tensor(t)
        if tt.numel() == 0 or tt.ndim != 2 or int(tt.shape[0]) < 2:
            fit_rows.append(torch.empty((0, int(tt.shape[1]) if tt.ndim == 2 else 0), dtype=torch.float32))
            holdout_rows.append(torch.empty((0, int(tt.shape[1]) if tt.ndim == 2 else 0), dtype=torch.float32))
            n_full = int(tt.shape[0]) if tt.ndim == 2 else 0
            # For v2, invariant digest of original even when insufficient
            if use_invariant:
                full_d = _partition_multiset_digest(tt if tt.ndim == 2 else torch.empty((0, 0), dtype=torch.float32))
                fit_empty = torch.empty((0, int(tt.shape[1]) if tt.ndim == 2 else 0), dtype=torch.float32)
                fit_d = _partition_multiset_digest(fit_empty)
                hold_d = _partition_multiset_digest(fit_empty)
                infos.append({"expert": ei, "n_full": n_full, "n_fit": 0, "n_holdout": 0, "insufficient": True, "policy": _SPLIT_POLICY, "version": _SPLIT_VERSION, "digest": fit_d, "full_digest": full_d, "fit_digest": fit_d, "holdout_digest": hold_d})
            else:
                infos.append({"expert": ei, "n_full": n_full, "n_fit": 0, "n_holdout": 0, "insufficient": True, "policy": "v1_per_expert_deterministic_content_hash_50_50", "version": 1, "digest": hashlib.sha256(b"empty").hexdigest() if n_full == 0 else _row_content_hash(tt) if n_full else "empty"})
            continue
        n, C = int(tt.shape[0]), int(tt.shape[1])
        # Build hash -> list of row indices
        hash_to_indices: dict[str, list[int]] = {}
        row_hashes: list[str] = []
        for idx in range(n):
            h = _row_content_hash(tt[idx])
            row_hashes.append(h)
            hash_to_indices.setdefault(h, []).append(idx)
        unique_hashes = sorted(hash_to_indices.keys())
        # Need at least 2 unique groups to split
        if len(unique_hashes) < 2:
            fit_rows.append(torch.empty((0, C), dtype=torch.float32))
            holdout_rows.append(torch.empty((0, C), dtype=torch.float32))
            if use_invariant:
                full_d = _partition_multiset_digest(tt)
                empty = torch.empty((0, C), dtype=torch.float32)
                fit_d = _partition_multiset_digest(empty)
                hold_d = _partition_multiset_digest(empty)
                infos.append({"expert": ei, "n_full": n, "n_fit": 0, "n_holdout": 0, "insufficient": True, "policy": _SPLIT_POLICY, "version": _SPLIT_VERSION, "digest": fit_d, "full_digest": full_d, "fit_digest": fit_d, "holdout_digest": hold_d})
            else:
                infos.append({"expert": ei, "n_full": n, "n_fit": 0, "n_holdout": 0, "insufficient": True, "reason": "single unique row content cannot form two partitions", "policy": "v1_per_expert_deterministic_content_hash_50_50", "version": 1, "unique_hashes": len(unique_hashes)})
            continue
        # Whole-group assignment to keep identical content together
        # Sort groups by hash, then assign groups alternately? Use prefix half by groups count to be permutation-invariant and content-grouped
        half_groups = len(unique_hashes) // 2
        fit_hashes = set(unique_hashes[:half_groups])
        holdout_hashes = set(unique_hashes[half_groups:])
        fit_indices: list[int] = []
        holdout_indices: list[int] = []
        for h in unique_hashes:
            idxs = hash_to_indices[h]
            if h in fit_hashes:
                fit_indices.extend(idxs)
            else:
                holdout_indices.extend(idxs)
        # Ensure both nonempty and no cross-group duplicate (by construction)
        if not fit_indices or not holdout_indices:
            fit_rows.append(torch.empty((0, C), dtype=torch.float32))
            holdout_rows.append(torch.empty((0, C), dtype=torch.float32))
            if use_invariant:
                full_d = _partition_multiset_digest(tt)
                empty = torch.empty((0, C), dtype=torch.float32)
                fit_d = _partition_multiset_digest(empty)
                hold_d = _partition_multiset_digest(empty)
                infos.append({"expert": ei, "n_full": n, "n_fit": 0, "n_holdout": 0, "insufficient": True, "policy": _SPLIT_POLICY, "version": _SPLIT_VERSION, "digest": fit_d, "full_digest": full_d, "fit_digest": fit_d, "holdout_digest": hold_d})
            else:
                infos.append({"expert": ei, "n_full": n, "n_fit": 0, "n_holdout": 0, "insufficient": True, "reason": "grouped split produced empty partition", "policy": "v1_per_expert_deterministic_content_hash_50_50", "version": 1})
            continue
        fit_t = tt[torch.tensor(sorted(fit_indices), dtype=torch.long)].contiguous()
        hold_t = tt[torch.tensor(sorted(holdout_indices), dtype=torch.long)].contiguous()
        # Digests for provenance: v2 invariant multiset, v1 legacy order-dependent
        if use_invariant:
            fit_digest = _partition_multiset_digest(fit_t)
            hold_digest = _partition_multiset_digest(hold_t)
            full_digest = _partition_multiset_digest(tt)
        else:
            fit_digest = hashlib.sha256(fit_t.numpy().tobytes()).hexdigest()
            hold_digest = hashlib.sha256(hold_t.numpy().tobytes()).hexdigest()
            full_digest = hashlib.sha256(tt.numpy().tobytes()).hexdigest()
            infos.append({"expert": ei, "n_full": n, "n_fit": int(fit_t.shape[0]), "n_holdout": int(hold_t.shape[0]), "insufficient": False, "policy": "v1_per_expert_deterministic_content_hash_50_50", "version": 1, "full_digest": full_digest, "fit_digest": fit_digest, "holdout_digest": hold_digest, "unique_hashes": len(unique_hashes)})
            fit_rows.append(fit_t)
            holdout_rows.append(hold_t)
            continue
        infos.append({"expert": ei, "n_full": n, "n_fit": int(fit_t.shape[0]), "n_holdout": int(hold_t.shape[0]), "insufficient": False, "policy": _SPLIT_POLICY, "version": _SPLIT_VERSION, "full_digest": full_digest, "fit_digest": fit_digest, "holdout_digest": hold_digest, "unique_hashes": len(unique_hashes)})
        fit_rows.append(fit_t)
        holdout_rows.append(hold_t)
    return fit_rows, holdout_rows, infos


from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedLDLQEvidence:
    """Immutable holder for prepared gate evidence — single truthful owner.

    Contains stable fit/holdout tensors, eligible subset, and content digests.
    Eligible = not cold and not insufficient (sufficient rows for a fit/holdout
    split). Cold/insufficient experts are never pooled or factored; their rows
    remain exactly raw. Validation is fail-closed against stored multiset
    digests for original/fit/holdout, expert count/width, and policy/version.
    Split_infos are deep-copied plain dicts and validated against stored
    fit/holdout digests before telemetry (mutation fails closed). No
    MappingProxyType enters activation_evidence/gate_info/checkpoint/JSON.
    """

    original_rows: tuple[torch.Tensor, ...]
    fit_rows: tuple[torch.Tensor, ...]
    holdout_rows: tuple[torch.Tensor, ...]
    fit_filled: tuple[torch.Tensor, ...]
    split_infos: tuple[dict, ...]
    cold_experts: tuple[int, ...]
    insufficient_experts: tuple[int, ...]
    eligible_experts: tuple[int, ...]
    fit_missing_pooled: tuple[int, ...]
    has_observed_fit: bool
    policy: str
    version: int
    original_digests: tuple[str, ...]
    fit_digests: tuple[str, ...]
    holdout_digests: tuple[str, ...]
    split_infos_digest: str


def prepare_ldlq_gate_evidence(
    activation_rows: torch.Tensor | Sequence[torch.Tensor],
    *,
    qname: str = "ldlq_gate",
) -> PreparedLDLQEvidence:
    """Prepare gate evidence once per packed parent / dense tensor.

    Deterministically splits, identifies cold/insufficient, and retains the
    same fit tensor objects for across-rung factor reuse. Cold/insufficient
    experts are NEVER pooled or factored; their FIT rows remain empty and they
    are excluded from the eligible set. Fail-closed on version/policy mismatch
    but tolerates all-empty fit (returns ineligible set, gate will fallback raw
    with zero factors).
    """
    # Normalize to tuple of Tensors; dense Tensor becomes single-entry tuple
    if isinstance(activation_rows, torch.Tensor):
        orig_seq: tuple[torch.Tensor, ...] = (torch.as_tensor(activation_rows),)
    else:
        orig_seq = tuple(torch.as_tensor(r) if not isinstance(r, torch.Tensor) else r for r in activation_rows)  # type: ignore[arg-type]
    # Also handle empty sequence edge
    if len(orig_seq) == 0:
        orig_seq = ()
    fit_rows, holdout_rows, split_infos = deterministic_fit_holdout_split_per_expert(
        list(orig_seq) if len(orig_seq) else [], version=_SPLIT_VERSION
    )
    # Cold: empty original rows
    cold = tuple(i for i, a in enumerate(orig_seq) if not torch.as_tensor(a).numel() or (torch.as_tensor(a).ndim == 2 and int(torch.as_tensor(a).shape[0]) == 0))
    # Insufficient: nonempty evidence that cannot form independent FIT/HOLDOUT (disjoint from cold)
    insufficient = tuple(i for i, info in enumerate(split_infos) if info.get("insufficient") and i not in set(cold))
    # Eligible: successful nonempty split (disjoint, covers range(E))
    eligible = tuple(i for i in range(len(orig_seq)) if i not in set(cold) and i not in set(insufficient))
    # Do NOT pool cold/insufficient FIT rows — they remain empty and receive no Hessian.
    fit_filled: tuple[torch.Tensor, ...] = tuple(fit_rows)
    fit_missing: tuple[int, ...] = tuple(sorted(set(cold) | set(insufficient)))
    has_observed = bool(eligible)
    # Immutable digests for validation (multiset, order-invariant within expert)
    original_digests = tuple(_partition_multiset_digest(torch.as_tensor(a)) for a in orig_seq) if len(orig_seq) else ()
    fit_digests = tuple(_partition_multiset_digest(torch.as_tensor(a)) for a in fit_rows) if len(orig_seq) else tuple()
    holdout_digests = tuple(_partition_multiset_digest(torch.as_tensor(a)) for a in holdout_rows) if len(orig_seq) else tuple()
    # Deep-copied plain dicts plus canonical digest (mutation fails closed, JSON/pickle serializable)
    frozen_infos = tuple(dict(info) for info in split_infos)
    # Canonical digest via single helper — serialization error is bug and must abort/fail closed
    _split_infos_digest = _canonical_split_infos_digest(frozen_infos)
    return PreparedLDLQEvidence(
        original_rows=orig_seq,
        fit_rows=tuple(fit_rows),
        holdout_rows=tuple(holdout_rows),
        fit_filled=fit_filled,
        split_infos=frozen_infos,
        cold_experts=cold,
        insufficient_experts=insufficient,
        eligible_experts=eligible,
        fit_missing_pooled=fit_missing,
        has_observed_fit=has_observed,
        policy=_SPLIT_POLICY,
        version=_SPLIT_VERSION,
        original_digests=original_digests,
        fit_digests=fit_digests,
        holdout_digests=holdout_digests,
        split_infos_digest=_split_infos_digest,
    )


# ---------------------------------------------------------------------------
# Validated packed evidence — one-time heavy validation + cheap fingerprint
# ---------------------------------------------------------------------------

def _canonical_json_bytes(obj: Any) -> bytes:
    """Versioned canonical JSON envelope (UTF-8, sorted keys, fixed separators, no NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")


def _tensor_fingerprint(t: torch.Tensor) -> tuple[Any, ...]:
    """Cheap tensor fingerprint: id, data_ptr, storage_offset, shape, stride, dtype, device, version."""
    try:
        data_ptr = t.data_ptr()
    except Exception:
        data_ptr = None
    try:
        storage_offset = t.storage_offset()  # type: ignore[attr-defined]
    except Exception:
        storage_offset = 0
    try:
        shape = tuple(t.shape)
    except Exception:
        shape = ()
    try:
        stride = tuple(t.stride())
    except Exception:
        stride = ()
    try:
        dtype_str = str(t.dtype)
    except Exception:
        dtype_str = ""
    try:
        device_str = str(t.device)
    except Exception:
        device_str = ""
    try:
        version = getattr(t, "_version", None)
    except Exception:
        version = None
    return (id(t), data_ptr, storage_offset, shape, stride, dtype_str, device_str, version)


def _canonical_le_bytes(t: torch.Tensor) -> bytes:
    """Centralized little-endian canonical bytes for float32/int64/uint8 payloads only."""
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"_canonical_le_bytes requires Tensor, got {type(t).__name__}")
    tt = t.detach().cpu().contiguous()
    import numpy as np  # local import
    if tt.dtype == torch.float32:
        return tt.numpy().astype(np.dtype('<f4'), copy=False).tobytes()
    elif tt.dtype == torch.int64:
        return tt.numpy().astype(np.dtype('<i8'), copy=False).tobytes()
    elif tt.dtype == torch.uint8:
        return tt.numpy().tobytes()
    else:
        raise ValueError(f"_canonical_le_bytes supports only torch.float32, torch.int64, torch.uint8; got {tt.dtype} (bfloat16 is metadata dtype='bfloat16', not a tensor payload)")


def _codebook_blob_bytes(codebook: torch.Tensor | tuple | list) -> bytes:
    """Deterministic codebook blob with per-table framing (ordinal/count/shape/dtype/byte-order/bytes).

    Framing binds table ordinal, total count, exact shape, exact dtype string,
    canonical little-endian byte order, and bytes. Changing dtype via .view(int32)
    preserves raw bytes but changes dtype framing, so digest differs. Never rely
    on raw concatenation alone.
    """
    tables: list[torch.Tensor]
    if isinstance(codebook, torch.Tensor):
        tables = [codebook]
    elif isinstance(codebook, (tuple, list)):
        tables = list(codebook)  # type: ignore[assignment]
    else:
        raise ValueError(f"codebook must be Tensor or tuple, got {type(codebook).__name__}")
    # Canonical little-endian for device-independent serialization
    blob = b""
    # Frame total count first
    blob += struct.pack(">Q", len(tables))
    for ordinal, t in enumerate(tables):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"codebook table not Tensor, got {type(t).__name__}")
        # Exact dtype string (e.g. torch.float32)
        dtype_str = str(t.dtype)
        shape = tuple(int(d) for d in t.shape)
        b = _canonical_le_bytes(t)
        # Framing: ordinal, shape len + dims, dtype len + bytes, byte_order, bytes len + bytes
        blob += struct.pack(">Q", int(ordinal))
        blob += struct.pack(">Q", len(shape))
        for dim in shape:
            blob += struct.pack(">Q", int(dim))
        dtype_b = dtype_str.encode("utf-8")
        blob += struct.pack(">Q", len(dtype_b)) + dtype_b
        bo_b = b"little"
        blob += struct.pack(">Q", len(bo_b)) + bo_b
        blob += struct.pack(">Q", len(b)) + b
    return blob


class ValidatedPackedLDLQEvidence:
    """Immutable capability for packed MoE leaf-local path.

    Constructor performs full digest/content/topology validation exactly once
    (heavy: partition_multiset digests, D2H copies) and snapshots
    policy/version/digest domain plus cheap tensor fingerprints.
    Helper and identity accept only this validated object; subsequent calls
    perform O(E) cheap fingerprint/metadata checks only, no row-content hashing.
    Clones input evidence once into private tensors to sever caller aliases;
    locks all attributes after construction; keeps private immutable canonical
    snapshots; does not expose raw private PreparedLDLQEvidence or mutable tensor
    aliases; exposes only immutable views or defensive copies. Preserves exact
    same private FIT tensor objects internally for factor-cache reuse across
    leaves/rungs. Ordinary internal in-place mutation fails via cheap fingerprints,
    while later mutation of the original caller-owned source does not affect the
    capability.
    """

    __slots__ = (
        "_priv_prepared",
        "_E",
        "_C",
        "_policy",
        "_version",
        "_original_digests",
        "_fit_digests",
        "_holdout_digests",
        "_split_infos_digest",
        "_split_infos_snapshot",
        "_cold_experts",
        "_insufficient_experts",
        "_eligible_experts",
        "_orig_fps",
        "_fit_fps",
        "_holdout_fps",
        "_fit_filled_fps",
        "_split_infos_id",
        "_split_dict_ids",
        "_split_dict_keys",
        "_frozen",
    )

    def __getattribute__(self, name: str):
        if name in ("_priv_prepared", "_orig_fps", "_fit_fps", "_holdout_fps", "_fit_filled_fps"):
            raise AttributeError(f"ValidatedPackedLDLQEvidence is immutable; private bundle {name!r} not accessible via ordinary attribute access (use object.__getattribute__ for module-internal access)")
        if name == "__dict__":
            raise AttributeError("ValidatedPackedLDLQEvidence has no __dict__ (slots only)")
        return object.__getattribute__(self, name)

    def __init__(self, prepared: "PreparedLDLQEvidence", E: int, C: int):
        # Allow construction-time attribute sets
        object.__setattr__(self, "_frozen", False)
        if type(E) is not int:
            raise ValueError(f"E must be exact int, got {type(E).__name__} {E!r}")
        if type(C) is not int:
            raise ValueError(f"C must be exact int, got {type(C).__name__} {C!r}")
        if E <= 0 or C <= 0:
            raise ValueError(f"E and C must be positive, got E={E} C={C}")
        if not isinstance(prepared, PreparedLDLQEvidence):
            raise ValueError(f"prepared must be PreparedLDLQEvidence, got {type(prepared).__name__}")
        if type(prepared.policy) is not str:
            raise ValueError(f"prepared.policy must be exact str (subclass rejected), got {type(prepared.policy).__name__} {prepared.policy!r}")
        if prepared.policy != _SPLIT_POLICY:
            raise ValueError(f"prepared policy {prepared.policy!r} != expected {_SPLIT_POLICY!r}")
        if type(prepared.version) is not int:
            raise ValueError(f"prepared.version must be exact int (bool/float/subclass rejected), got {type(prepared.version).__name__} {prepared.version!r}")
        if prepared.version != _SPLIT_VERSION:
            raise ValueError(f"prepared version {prepared.version!r} != expected {_SPLIT_VERSION}")
        # Exact built-in tuple containers for tensor/digest/category fields (defect 15)
        for _attr in ("original_rows", "fit_rows", "holdout_rows", "fit_filled", "original_digests", "fit_digests", "holdout_digests", "split_infos", "cold_experts", "insufficient_experts", "eligible_experts", "fit_missing_pooled"):
            _val = getattr(prepared, _attr)
            if type(_val) is not tuple:
                raise ValueError(f"prepared.{_attr} must be exact tuple (got {type(_val).__name__})")
        # Every stored outer digest must be exact lowercase SHA-256 str (defect 15)
        def _is_hex64(s: str) -> bool:
            return type(s) is str and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
        for _dig_list in (prepared.original_digests, prepared.fit_digests, prepared.holdout_digests):
            for _idx, _d in enumerate(_dig_list):
                if not _is_hex64(_d):
                    raise ValueError(f"prepared digest at index {_idx} must be exact lowercase SHA-256 hex64, got {_d!r} type {type(_d).__name__}")
        if not _is_hex64(prepared.split_infos_digest):
            raise ValueError(f"prepared.split_infos_digest must be exact lowercase SHA-256 hex64, got {prepared.split_infos_digest!r}")
        if type(prepared.has_observed_fit) is not bool:
            raise ValueError(f"prepared.has_observed_fit must be exact bool, got {type(prepared.has_observed_fit).__name__}")
        if len(prepared.original_rows) != E:
            raise ValueError(f"prepared expert count {len(prepared.original_rows)} != E {E}")
        if len(prepared.fit_rows) != E or len(prepared.holdout_rows) != E:
            raise ValueError("prepared fit/holdout length mismatch")
        if len(prepared.fit_filled) != E:
            raise ValueError("prepared fit_filled length mismatch")
        if len(prepared.original_digests) != E or len(prepared.fit_digests) != E or len(prepared.holdout_digests) != E:
            raise ValueError("prepared digest length mismatch")
        if len(prepared.split_infos) != E:
            raise ValueError(f"prepared split_infos length {len(prepared.split_infos)} != E {E}")
        # --- Topology validation (once, heavy) ---
        def _validate_index_tuple(name: str, tup: tuple) -> None:
            if not isinstance(tup, tuple):
                raise ValueError(f"{name} must be tuple, got {type(tup).__name__}")
            seen = set()
            for idx, eid in enumerate(tup):
                if type(eid) is not int:
                    raise ValueError(f"{name}[{idx}] must be exact int (bool rejected), got {type(eid).__name__} {eid!r}")
                if not (0 <= eid < E):
                    raise ValueError(f"{name}[{idx}]={eid} out of range [0,{E})")
                if eid in seen:
                    raise ValueError(f"{name} contains duplicate {eid}")
                seen.add(eid)
        _validate_index_tuple("cold_experts", prepared.cold_experts)
        _validate_index_tuple("insufficient_experts", prepared.insufficient_experts)
        _validate_index_tuple("eligible_experts", prepared.eligible_experts)
        _validate_index_tuple("fit_missing_pooled", prepared.fit_missing_pooled)
        # Canonical order required (defect 8): every category tuple must already be sorted increasing
        if tuple(prepared.cold_experts) != tuple(sorted(prepared.cold_experts)):
            raise ValueError(f"cold_experts {prepared.cold_experts!r} must be canonical increasing order")
        if tuple(prepared.insufficient_experts) != tuple(sorted(prepared.insufficient_experts)):
            raise ValueError(f"insufficient_experts {prepared.insufficient_experts!r} must be canonical increasing order")
        if tuple(prepared.eligible_experts) != tuple(sorted(prepared.eligible_experts)):
            raise ValueError(f"eligible_experts {prepared.eligible_experts!r} must be canonical increasing order (shuffled rejected)")
        if tuple(prepared.fit_missing_pooled) != tuple(sorted(prepared.fit_missing_pooled)):
            raise ValueError(f"fit_missing_pooled {prepared.fit_missing_pooled!r} must be canonical increasing order")
        cold_s = set(prepared.cold_experts)
        insuf_s = set(prepared.insufficient_experts)
        elig_s = set(prepared.eligible_experts)
        # Pairwise disjoint and cover range(E)
        if cold_s & insuf_s:
            raise ValueError(f"cold/insufficient overlap {cold_s & insuf_s} (must be disjoint)")
        if cold_s & elig_s:
            raise ValueError(f"cold/eligible overlap {cold_s & elig_s}")
        if insuf_s & elig_s:
            raise ValueError(f"insufficient/eligible overlap {insuf_s & elig_s}")
        if cold_s | insuf_s | elig_s != set(range(E)):
            raise ValueError(f"cold/insufficient/eligible must partition range(E) exactly, got cold={sorted(cold_s)} insuf={sorted(insuf_s)} elig={sorted(elig_s)}")
        expected_eligible = tuple(sorted(set(range(E)) - cold_s - insuf_s))
        if tuple(prepared.eligible_experts) != expected_eligible:
            raise ValueError(f"eligible_experts {tuple(prepared.eligible_experts)!r} != canonical complement {expected_eligible!r} (cold={sorted(cold_s)} insuf={sorted(insuf_s)}; shuffled rejected)")
        if tuple(prepared.fit_missing_pooled) != tuple(sorted(cold_s | insuf_s)):
            raise ValueError(f"fit_missing_pooled {tuple(prepared.fit_missing_pooled)!r} != sorted union cold+insufficient {tuple(sorted(cold_s | insuf_s))!r}")
        if bool(prepared.has_observed_fit) != bool(elig_s):
            raise ValueError(f"has_observed_fit {prepared.has_observed_fit} != bool(eligible) {bool(elig_s)}")
        if type(prepared.has_observed_fit) is not bool:
            raise ValueError(f"has_observed_fit must be exact bool, got {type(prepared.has_observed_fit).__name__}")
        # Validate every tensor has exact rank 2 and width C even when empty, finite, and row-count relationships
        for i in range(E):
            for name, tup in [("original_rows", prepared.original_rows), ("fit_rows", prepared.fit_rows), ("holdout_rows", prepared.holdout_rows), ("fit_filled", prepared.fit_filled)]:
                t = torch.as_tensor(tup[i])
                if t.ndim != 2:
                    raise ValueError(f"{name}[{i}] rank {t.ndim} != 2 (must be exactly 2 even when empty)")
                if int(t.shape[1]) != C:
                    raise ValueError(f"{name}[{i}] width {int(t.shape[1])} != C {C} (must be exactly C even when empty, got shape {tuple(t.shape)})")
                if t.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                    raise ValueError(f"{name}[{i}] dtype {t.dtype} must be floating (allowed float32/float64/float16/bfloat16), got {t.dtype}")
                if t.numel() > 0:
                    if not torch.isfinite(t).all():
                        raise ValueError(f"{name}[{i}] must be finite (NaN/Inf detected)")
                # Row-count relationships per category
                n = int(t.shape[0])
                # For fit_filled, validate against FIT (not just length)
                if name == "fit_filled":
                    fit_n = int(torch.as_tensor(prepared.fit_rows[i]).shape[0])
                    if n != fit_n:
                        raise ValueError(f"fit_filled[{i}] rows {n} != fit_rows rows {fit_n} (must equal FIT, not just length)")
                    # Content equality via digest
                    if _partition_multiset_digest(t) != prepared.fit_digests[i]:
                        raise ValueError(f"fit_filled[{i}] content digest mismatch vs fit_rows at {i}")
            # Category-specific row checks
            orig_n = int(torch.as_tensor(prepared.original_rows[i]).shape[0])
            fit_n = int(torch.as_tensor(prepared.fit_rows[i]).shape[0])
            hold_n = int(torch.as_tensor(prepared.holdout_rows[i]).shape[0])
            is_cold = i in cold_s
            is_insuf = i in insuf_s
            is_elig = i in elig_s
            if is_cold:
                if orig_n != 0:
                    raise ValueError(f"cold expert {i} original rows must be empty, got {orig_n}")
                if fit_n != 0 or hold_n != 0:
                    raise ValueError(f"cold expert {i} fit/holdout must be empty, got fit {fit_n} hold {hold_n}")
            elif is_insuf:
                if orig_n == 0:
                    raise ValueError(f"insufficient non-cold expert {i} original must be nonempty, got empty")
                if fit_n != 0 or hold_n != 0:
                    raise ValueError(f"insufficient expert {i} fit/holdout must be empty, got fit {fit_n} hold {hold_n}")
            else:  # eligible
                if fit_n == 0 or hold_n == 0:
                    raise ValueError(f"eligible expert {i} fit/holdout must be nonempty, got fit {fit_n} hold {hold_n}")
        # Split-info exact v2 branch schema, no extras/nested/container values
        def _is_lower_hex64(s: str) -> bool:
            return type(s) is str and len(s) == 64 and all(c in "0123456789abcdef" for c in s) and s == s.lower()
        for i, info in enumerate(prepared.split_infos):
            if not isinstance(info, dict):
                raise ValueError(f"split_infos[{i}] must be dict, got {type(info).__name__}")
            # No nested/container values, only scalar (int/str/bool) except we allow no list/dict
            for k, v in info.items():
                if isinstance(v, (dict, list, tuple, set)):
                    raise ValueError(f"split_infos[{i}][{k!r}] must be scalar, got container {type(v).__name__} {v!r}")
            # Common exact keys
            common_keys = {"expert", "n_full", "n_fit", "n_holdout", "insufficient", "policy", "version", "full_digest", "fit_digest", "holdout_digest"}
            is_insuf = info.get("insufficient") is True
            is_elig = info.get("insufficient") is False
            if is_insuf:
                expected_keys = common_keys | {"digest"}
            elif is_elig:
                expected_keys = common_keys | {"unique_hashes"}
            else:
                raise ValueError(f"split_infos[{i}] insufficient must be exact bool, got {info.get('insufficient')!r} type {type(info.get('insufficient')).__name__}")
            if set(info.keys()) != expected_keys:
                raise ValueError(f"split_infos[{i}] keys {sorted(info.keys())} != expected {sorted(expected_keys)} for {'insufficient' if is_insuf else 'eligible'} branch (no extras/nested)")
            # Exact types
            if type(info["expert"]) is not int:
                raise ValueError(f"split_infos[{i}][expert] must be exact int, got {type(info['expert']).__name__}")
            if int(info["expert"]) != i:
                raise ValueError(f"split_infos[{i}] expert field {info['expert']} != index {i}")
            for int_key in ("n_full", "n_fit", "n_holdout"):
                if type(info[int_key]) is not int:
                    raise ValueError(f"split_infos[{i}][{int_key}] must be exact int, got {type(info[int_key]).__name__}")
            if type(info["insufficient"]) is not bool:
                raise ValueError(f"split_infos[{i}][insufficient] must be exact bool, got {type(info['insufficient']).__name__}")
            if type(info["policy"]) is not str or info["policy"] != _SPLIT_POLICY:
                raise ValueError(f"split_infos[{i}] policy {info.get('policy')!r} != {_SPLIT_POLICY!r}")
            if type(info["version"]) is not int or info["version"] != _SPLIT_VERSION:
                raise ValueError(f"split_infos[{i}] version {info.get('version')!r} != {_SPLIT_VERSION}")
            for dk in ("full_digest", "fit_digest", "holdout_digest"):
                if not _is_lower_hex64(info[dk]):
                    raise ValueError(f"split_infos[{i}][{dk!r}] must be exact lowercase SHA-256 hex, got {info.get(dk)!r}")
            if is_insuf:
                if not _is_lower_hex64(info["digest"]):
                    raise ValueError(f"split_infos[{i}][digest] must be exact lowercase SHA-256, got {info.get('digest')!r}")
                if info["digest"] != info["fit_digest"]:
                    raise ValueError(f"split_infos[{i}][digest] {info['digest']!r} != fit_digest {info['fit_digest']!r} (insufficient branch must have digest == fit_digest)")
            else:
                if type(info["unique_hashes"]) is not int:
                    raise ValueError(f"split_infos[{i}][unique_hashes] must be exact int, got {type(info['unique_hashes']).__name__}")
                if int(info["unique_hashes"]) < 2:
                    raise ValueError(f"split_infos[{i}][unique_hashes] {info['unique_hashes']} <2 for eligible")
            # Validate counts match tensors
            orig_n = int(torch.as_tensor(prepared.original_rows[i]).shape[0])
            fit_n = int(torch.as_tensor(prepared.fit_rows[i]).shape[0])
            hold_n = int(torch.as_tensor(prepared.holdout_rows[i]).shape[0])
            if info["n_full"] != orig_n:
                raise ValueError(f"split_infos[{i}] n_full {info['n_full']} != actual {orig_n}")
            if info["n_fit"] != fit_n:
                raise ValueError(f"split_infos[{i}] n_fit {info['n_fit']} != actual fit rows {fit_n}")
            if info["n_holdout"] != hold_n:
                raise ValueError(f"split_infos[{i}] n_holdout {info['n_holdout']} != actual holdout {hold_n}")
            # Validate digests match stored/actual content
            if info.get("fit_digest") != prepared.fit_digests[i]:
                raise ValueError(f"split_infos[{i}] fit_digest {info.get('fit_digest')!r} != prepared.fit_digests[{i}]")
            if info.get("holdout_digest") != prepared.holdout_digests[i]:
                raise ValueError(f"split_infos[{i}] holdout_digest mismatch at {i}")
            if info.get("full_digest") != prepared.original_digests[i]:
                raise ValueError(f"split_infos[{i}] full_digest mismatch at {i}")
            if not _is_lower_hex64(info["full_digest"]) or not _is_lower_hex64(info["fit_digest"]) or not _is_lower_hex64(info["holdout_digest"]):
                raise ValueError(f"split_infos[{i}] digests must be lowercase hex64")
            # Additional heavy integrity: insufficient mapping, n_fit+n_holdout==n_full, unique_hashes exact, multiset union & disjoint
            expected_insuf = (i in cold_s) or (i in insuf_s)
            if bool(info["insufficient"]) != bool(expected_insuf):
                raise ValueError(f"split_infos[{i}] insufficient {info['insufficient']!r} != expected {expected_insuf} for expert {i} (cold={sorted(cold_s)} insuf={sorted(insuf_s)} elig={sorted(elig_s)})")
            if i in elig_s:
                if info["n_fit"] + info["n_holdout"] != info["n_full"]:
                    raise ValueError(f"split_infos[{i}] eligible n_fit {info['n_fit']} + n_holdout {info['n_holdout']} != n_full {info['n_full']}")
                # unique_hashes must equal actual number of unique original row-content hashes
                orig_t = torch.as_tensor(prepared.original_rows[i])
                actual_hashes = set(_row_content_hash(orig_t[r]) for r in range(int(orig_t.shape[0]))) if orig_t.numel() > 0 else set()
                if int(info["unique_hashes"]) != len(actual_hashes):
                    raise ValueError(f"split_infos[{i}] unique_hashes {info['unique_hashes']} != actual {len(actual_hashes)} (must equal exact unique original row hashes, not >=2)")
                # Multiset union of FIT and HOLDOUT must equal original exactly (multiplicity-sensitive)
                fit_t = torch.as_tensor(prepared.fit_rows[i])
                hold_t = torch.as_tensor(prepared.holdout_rows[i])
                if fit_t.shape[0] == 0 or hold_t.shape[0] == 0:
                    raise ValueError(f"eligible expert {i} FIT/HOLDOUT empty but should be nonempty")
                # Build union digest via sorted row hashes including multiplicity: compare digests
                union_t = torch.cat([fit_t, hold_t], dim=0) if fit_t.numel() and hold_t.numel() else torch.empty((0, int(orig_t.shape[1])), dtype=torch.float32)
                if _partition_multiset_digest(union_t) != _partition_multiset_digest(orig_t):
                    raise ValueError(f"eligible expert {i} FIT ∪ HOLDOUT multiset != original (content mismatch or missing/extra rows)")
                # Row-content hash sets disjoint so duplicate content never crosses partitions
                fit_hashes = set(_row_content_hash(fit_t[r]) for r in range(int(fit_t.shape[0])))
                hold_hashes = set(_row_content_hash(hold_t[r]) for r in range(int(hold_t.shape[0])))
                if fit_hashes & hold_hashes:
                    raise ValueError(f"eligible expert {i} duplicate row content crosses FIT/HOLDOUT {fit_hashes & hold_hashes}")
        # All stored metadata/policy/version/digests internally consistent already checked above
        # Heavy one-time content digest verification (D2H) — exactly once
        for i in range(E):
            if _partition_multiset_digest(torch.as_tensor(prepared.original_rows[i])) != prepared.original_digests[i]:
                raise ValueError(f"prepared original_rows content mismatch at expert {i} (mutation or corruption)")
            if _partition_multiset_digest(torch.as_tensor(prepared.fit_rows[i])) != prepared.fit_digests[i]:
                raise ValueError(f"prepared fit_rows content mismatch at expert {i}")
            if _partition_multiset_digest(torch.as_tensor(prepared.holdout_rows[i])) != prepared.holdout_digests[i]:
                raise ValueError(f"prepared holdout_rows content mismatch at expert {i}")
            # Validate fit_filled content against FIT (not just length) — already done above via digest, but double-check
            if _partition_multiset_digest(torch.as_tensor(prepared.fit_filled[i])) != prepared.fit_digests[i]:
                raise ValueError(f"prepared fit_filled content mismatch vs fit at expert {i}")
        if _canonical_split_infos_digest(prepared.split_infos) != prepared.split_infos_digest:
            raise ValueError("prepared split_infos digest mismatch (nested mutation)")
        # Canonical split enforcement (defect 14): recompute v2 deterministic split from original rows and require exact match (construction-only, D2H)
        _canon_fit, _canon_hold, _canon_infos = deterministic_fit_holdout_split_per_expert(list(prepared.original_rows), version=_SPLIT_VERSION)
        # Derive canonical categories
        _canon_cold = tuple(i for i, a in enumerate(prepared.original_rows) if not torch.as_tensor(a).numel() or (torch.as_tensor(a).ndim == 2 and int(torch.as_tensor(a).shape[0]) == 0))
        _canon_insufficient = tuple(i for i, info in enumerate(_canon_infos) if info.get("insufficient") and i not in set(_canon_cold))
        _canon_eligible = tuple(i for i in range(E) if i not in set(_canon_cold) and i not in set(_canon_insufficient))
        _canon_fit_missing = tuple(sorted(set(_canon_cold) | set(_canon_insufficient)))
        _canon_has_observed = bool(_canon_eligible)
        if tuple(prepared.cold_experts) != _canon_cold:
            raise ValueError(f"prepared cold_experts {tuple(prepared.cold_experts)!r} != canonical {_canon_cold!r} (recomputed from original rows)")
        if tuple(prepared.insufficient_experts) != _canon_insufficient:
            raise ValueError(f"prepared insufficient_experts {tuple(prepared.insufficient_experts)!r} != canonical {_canon_insufficient!r}")
        if tuple(prepared.eligible_experts) != _canon_eligible:
            raise ValueError(f"prepared eligible_experts {tuple(prepared.eligible_experts)!r} != canonical {_canon_eligible!r}")
        if tuple(prepared.fit_missing_pooled) != _canon_fit_missing:
            raise ValueError(f"prepared fit_missing_pooled {tuple(prepared.fit_missing_pooled)!r} != canonical {_canon_fit_missing!r}")
        if bool(prepared.has_observed_fit) != bool(_canon_has_observed):
            raise ValueError(f"prepared has_observed_fit {prepared.has_observed_fit!r} != canonical {_canon_has_observed!r}")
        for i in range(E):
            # Multiset digests must match canonical (where row order irrelevant, digest sufficient)
            if _partition_multiset_digest(torch.as_tensor(prepared.fit_rows[i])) != _partition_multiset_digest(torch.as_tensor(_canon_fit[i])):
                raise ValueError(f"prepared fit_rows[{i}] canonical multiset mismatch at expert {i} (alternate partition rejected)")
            if _partition_multiset_digest(torch.as_tensor(prepared.holdout_rows[i])) != _partition_multiset_digest(torch.as_tensor(_canon_hold[i])):
                raise ValueError(f"prepared holdout_rows[{i}] canonical multiset mismatch at expert {i}")
            # Split-info branch must match canonical exactly (policy/version/insufficient/digests/counts/unique_hashes)
            _exp_info = dict(prepared.split_infos[i])
            _canon_info = dict(_canon_infos[i])
            # Compare without relying on dict order; require exact keys and values (multiset digests already checked)
            if set(_exp_info.keys()) != set(_canon_info.keys()):
                raise ValueError(f"prepared split_infos[{i}] keys {sorted(_exp_info.keys())} != canonical {sorted(_canon_info.keys())}")
            for _k in _canon_info:
                if _exp_info[_k] != _canon_info[_k]:
                    raise ValueError(f"prepared split_infos[{i}][{_k!r}] { _exp_info[_k]!r} != canonical { _canon_info[_k]!r}")
        # --- Clone tensors once into private storage to sever caller aliases ---
        def _clone_tensor(t: torch.Tensor) -> torch.Tensor:
            tt = torch.as_tensor(t)
            # Preserve dtype/device, clone to sever alias
            return tt.clone().detach()
        priv_original = tuple(_clone_tensor(t) for t in prepared.original_rows)
        priv_fit = tuple(_clone_tensor(t) for t in prepared.fit_rows)
        priv_holdout = tuple(_clone_tensor(t) for t in prepared.holdout_rows)
        priv_filled = tuple(_clone_tensor(t) for t in prepared.fit_filled)
        # Build private Prepared snapshot with cloned tensors (for internal reuse)
        # Keep policy/version/digests from original (they were validated)
        priv_prepared = PreparedLDLQEvidence(
            original_rows=priv_original,
            fit_rows=priv_fit,
            holdout_rows=priv_holdout,
            fit_filled=priv_filled,
            split_infos=tuple(dict(s) for s in prepared.split_infos),
            cold_experts=tuple(prepared.cold_experts),
            insufficient_experts=tuple(prepared.insufficient_experts),
            eligible_experts=tuple(prepared.eligible_experts),
            fit_missing_pooled=tuple(prepared.fit_missing_pooled),
            has_observed_fit=bool(prepared.has_observed_fit),
            policy=str(prepared.policy),
            version=int(prepared.version),
            original_digests=tuple(prepared.original_digests),
            fit_digests=tuple(prepared.fit_digests),
            holdout_digests=tuple(prepared.holdout_digests),
            split_infos_digest=str(prepared.split_infos_digest),
        )
        # Snapshot immutable domain from private
        object.__setattr__(self, "_priv_prepared", priv_prepared)
        object.__setattr__(self, "_E", int(E))
        object.__setattr__(self, "_C", int(C))
        object.__setattr__(self, "_policy", str(priv_prepared.policy))
        object.__setattr__(self, "_version", int(priv_prepared.version))
        object.__setattr__(self, "_original_digests", tuple(priv_prepared.original_digests))
        object.__setattr__(self, "_fit_digests", tuple(priv_prepared.fit_digests))
        object.__setattr__(self, "_holdout_digests", tuple(priv_prepared.holdout_digests))
        object.__setattr__(self, "_split_infos_digest", str(priv_prepared.split_infos_digest))
        # Immutable snapshots: use MappingProxyType for dicts
        proxied = tuple(types.MappingProxyType(dict(s)) for s in priv_prepared.split_infos)
        object.__setattr__(self, "_split_infos_snapshot", proxied)
        object.__setattr__(self, "_cold_experts", tuple(priv_prepared.cold_experts))
        object.__setattr__(self, "_insufficient_experts", tuple(priv_prepared.insufficient_experts))
        object.__setattr__(self, "_eligible_experts", tuple(priv_prepared.eligible_experts))
        # Cheap fingerprints for each tensor list (private clones) — preserve same objects for factor-cache reuse
        object.__setattr__(self, "_orig_fps", tuple(_tensor_fingerprint(t) for t in priv_original))
        object.__setattr__(self, "_fit_fps", tuple(_tensor_fingerprint(t) for t in priv_fit))
        object.__setattr__(self, "_holdout_fps", tuple(_tensor_fingerprint(t) for t in priv_holdout))
        object.__setattr__(self, "_fit_filled_fps", tuple(_tensor_fingerprint(t) for t in priv_filled))
        object.__setattr__(self, "_split_infos_id", id(priv_prepared.split_infos))
        object.__setattr__(self, "_split_dict_ids", tuple(id(d) for d in priv_prepared.split_infos))
        object.__setattr__(self, "_split_dict_keys", tuple(tuple(sorted(d.keys())) for d in priv_prepared.split_infos))
        # Lock
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"ValidatedPackedLDLQEvidence is immutable; cannot set {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"ValidatedPackedLDLQEvidence is immutable; cannot delete {name!r}")
        object.__delattr__(self, name)

    @property
    def prepared(self) -> "PreparedLDLQEvidence":
        # Defensive copy: clone tensors so caller cannot mutate private via .data, and return new Prepared with proxies not needed
        # Return a fresh Prepared with cloned tensors (not same objects as private)
        priv = object.__getattribute__(self, "_priv_prepared")
        # Clone tensors for defensive copy
        def _clone(t): return torch.as_tensor(t).clone().detach()
        return PreparedLDLQEvidence(
            original_rows=tuple(_clone(t) for t in priv.original_rows),
            fit_rows=tuple(_clone(t) for t in priv.fit_rows),
            holdout_rows=tuple(_clone(t) for t in priv.holdout_rows),
            fit_filled=tuple(_clone(t) for t in priv.fit_filled),
            split_infos=tuple(dict(s) for s in priv.split_infos),
            cold_experts=tuple(priv.cold_experts),
            insufficient_experts=tuple(priv.insufficient_experts),
            eligible_experts=tuple(priv.eligible_experts),
            fit_missing_pooled=tuple(priv.fit_missing_pooled),
            has_observed_fit=bool(priv.has_observed_fit),
            policy=str(priv.policy),
            version=int(priv.version),
            original_digests=tuple(priv.original_digests),
            fit_digests=tuple(priv.fit_digests),
            holdout_digests=tuple(priv.holdout_digests),
            split_infos_digest=str(priv.split_infos_digest),
        )

    @property
    def E(self) -> int:
        return object.__getattribute__(self, "_E")

    @property
    def C(self) -> int:
        return object.__getattribute__(self, "_C")

    @property
    def policy(self) -> str:
        return object.__getattribute__(self, "_policy")

    @property
    def version(self) -> int:
        return object.__getattribute__(self, "_version")

    @property
    def original_digests(self) -> tuple[str, ...]:
        return object.__getattribute__(self, "_original_digests")

    @property
    def fit_digests(self) -> tuple[str, ...]:
        return object.__getattribute__(self, "_fit_digests")

    @property
    def holdout_digests(self) -> tuple[str, ...]:
        return object.__getattribute__(self, "_holdout_digests")

    @property
    def split_infos_digest(self) -> str:
        return object.__getattribute__(self, "_split_infos_digest")

    @property
    def split_infos(self) -> tuple[types.MappingProxyType, ...]:
        # Return immutable MappingProxyType view; caller cannot mutate
        return object.__getattribute__(self, "_split_infos_snapshot")

    @property
    def cold_experts(self) -> tuple[int, ...]:
        return object.__getattribute__(self, "_cold_experts")

    @property
    def insufficient_experts(self) -> tuple[int, ...]:
        return object.__getattribute__(self, "_insufficient_experts")

    @property
    def eligible_experts(self) -> tuple[int, ...]:
        return object.__getattribute__(self, "_eligible_experts")

    def _check_fingerprints(self) -> None:
        """O(E) cheap fingerprint check — no D2H or hashing. Fail closed on mutation."""
        priv = object.__getattribute__(self, "_priv_prepared")
        E = object.__getattribute__(self, "_E")
        # Check E/C still match private lengths (topology)
        if len(priv.original_rows) != E:
            raise ValueError(f"validated evidence expert count mutated {len(priv.original_rows)} != {E}")
        if len(priv.fit_rows) != E or len(priv.holdout_rows) != E:
            raise ValueError("validated evidence fit/holdout length mutated")
        # Cheap string domain checks (no hashing)
        if priv.policy != object.__getattribute__(self, "_policy") or priv.version != object.__getattribute__(self, "_version"):
            raise ValueError("validated evidence policy/version mutated")
        if tuple(priv.original_digests) != object.__getattribute__(self, "_original_digests"):
            raise ValueError("validated evidence original_digests mutated")
        if tuple(priv.fit_digests) != object.__getattribute__(self, "_fit_digests"):
            raise ValueError("validated evidence fit_digests mutated")
        if tuple(priv.holdout_digests) != object.__getattribute__(self, "_holdout_digests"):
            raise ValueError("validated evidence holdout_digests mutated")
        if str(priv.split_infos_digest) != object.__getattribute__(self, "_split_infos_digest"):
            raise ValueError("validated evidence split_infos_digest mutated")
        # Cheap fingerprint verification for private tensor objects (detect in-place .data mutation)
        cur_orig_fps = tuple(_tensor_fingerprint(t) for t in priv.original_rows)
        if cur_orig_fps != object.__getattribute__(self, "_orig_fps"):
            raise ValueError("validated evidence original_rows tensor fingerprint mismatch (in-place mutation or replacement)")
        cur_fit_fps = tuple(_tensor_fingerprint(t) for t in priv.fit_rows)
        if cur_fit_fps != object.__getattribute__(self, "_fit_fps"):
            raise ValueError("validated evidence fit_rows tensor fingerprint mismatch (in-place mutation or replacement)")
        cur_holdout_fps = tuple(_tensor_fingerprint(t) for t in priv.holdout_rows)
        if cur_holdout_fps != object.__getattribute__(self, "_holdout_fps"):
            raise ValueError("validated evidence holdout_rows tensor fingerprint mismatch (in-place mutation or replacement)")
        cur_filled_fps = tuple(_tensor_fingerprint(t) for t in priv.fit_filled)
        if cur_filled_fps != object.__getattribute__(self, "_fit_filled_fps"):
            raise ValueError("validated evidence fit_filled fingerprint mismatch")
        # Split_infos tuple/dict identity cheap check (private)
        if id(priv.split_infos) != object.__getattribute__(self, "_split_infos_id"):
            raise ValueError("validated evidence split_infos tuple object replaced")
        if tuple(id(d) for d in priv.split_infos) != object.__getattribute__(self, "_split_dict_ids"):
            raise ValueError("validated evidence split_infos dict object replaced")
        # Also verify keys unchanged (cheap)
        cur_keys = tuple(tuple(sorted(d.keys())) for d in priv.split_infos)
        if cur_keys != object.__getattribute__(self, "_split_dict_keys"):
            raise ValueError("validated evidence split_infos keys mutated")
        # Verify split_infos content digest still matches snapshot via cheap JSON canonical (small, no D2H)
        if _canonical_split_infos_digest(priv.split_infos) != object.__getattribute__(self, "_split_infos_digest"):
            raise ValueError("validated evidence split_infos content mutated")


def _validate_validated_evidence(validated: "ValidatedPackedLDLQEvidence", E: int, C: int) -> None:
    """Cheap O(E) validation of validated evidence against expected E,C."""
    if not isinstance(validated, ValidatedPackedLDLQEvidence):
        raise ValueError(f"prepared must be ValidatedPackedLDLQEvidence (got {type(validated).__name__}); raw PreparedLDLQEvidence rejected — construct ValidatedPackedLDLQEvidence first")
    if validated.E != int(E) or validated.C != int(C):
        raise ValueError(f"validated evidence E/C {validated.E}/{validated.C} != expected {E}/{C}")
    validated._check_fingerprints()


def ldlq_reassign_cb_fields(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    block_size: int = LDLQ_BLOCK_SIZE,
    damping_fraction: float = LDLQ_DAMPING_FRACTION,
    batch_experts: bool | None = None,
) -> dict:
    """Run deterministic fixed-scale/codebook LDLQ assignment.

    Scale and codebook fitting have already completed. For stacked experts a
    sequence supplies one activation matrix (and Hessian) per expert; a single
    matrix deliberately shares one Hessian across the stack.
    """
    if batch_experts is None:
        raw_batch = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS", "1"
        ).strip().lower()
        if raw_batch not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS must be 0 or 1"
            )
        batch_experts = raw_batch in {"1", "true", "yes", "on"}
    if weight.ndim == 2 or isinstance(activation_rows, torch.Tensor):
        flat = weight.reshape(-1, weight.shape[-1])
        flat_col_weights = torch.broadcast_to(
            torch.as_tensor(col_weights), weight.shape
        ).reshape_as(flat)
        return _ldlq_reassign_fields_2d(
            flat,
            fields,
            flat_col_weights,
            torch.as_tensor(activation_rows),
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        )
    if weight.ndim != 3:
        raise ValueError(
            "per-slice LDLQ activations require a 3-D expert stack, got "
            f"{tuple(weight.shape)}"
        )
    activations = tuple(activation_rows)
    if len(activations) != int(weight.shape[0]):
        raise ValueError(
            f"LDLQ expert activation count {len(activations)} != "
            f"stack size {weight.shape[0]}"
        )
    if batch_experts:
        raw_workers = os.environ.get(
            "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS", "0"
        ).strip()
        try:
            feeder_workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS must be a non-negative "
                "integer"
            ) from exc
        if feeder_workers < 0:
            raise ValueError(
                "PRISMAQUANT_CB_LDLQ_FEEDER_THREADS must be a non-negative "
                "integer"
            )
        if feeder_workers and weight.device.type == "cuda":
            return _ldlq_reassign_fields_3d_threaded(
                weight,
                fields,
                col_weights,
                activations,
                grid=grid,
                mode=mode,
                block_size=block_size,
                damping_fraction=damping_fraction,
                workers=feeder_workers,
            )
        return _ldlq_reassign_fields_3d_batched(
            weight,
            fields,
            col_weights,
            activations,
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        )

    # Retained as the bit-identity reference.  Production takes the batched
    # arm; tests and measurement gates exercise both on identical inputs.
    cw = torch.broadcast_to(torch.as_tensor(col_weights), weight.shape)
    rows_per_expert = int(weight.shape[1])
    assignment_parts: list[dict] = []
    slice_keys = {"indices", "signs", "scales", "scale_super", "scale_sub"}
    for expert, x in enumerate(activations):
        start = expert * rows_per_expert
        end = start + rows_per_expert
        local = {
            key: (value[start:end] if key in slice_keys else value)
            for key, value in fields.items()
        }
        local["shape"] = tuple(weight[expert].shape)
        assignment_parts.append(_ldlq_reassign_fields_2d(
            weight[expert],
            local,
            cw[expert],
            x,
            grid=grid,
            mode=mode,
            block_size=block_size,
            damping_fraction=damping_fraction,
        ))
    updated = dict(fields)
    updated["indices"] = torch.cat(
        [part["indices"] for part in assignment_parts], dim=0
    )
    if mode == "signed":
        updated["signs"] = torch.cat(
            [part["signs"] for part in assignment_parts], dim=0
        )
    return updated


def ldlq_reassign_cb_fields_gated(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor | Sequence[torch.Tensor],
    *,
    grid: str,
    mode: str,
    k: int,
    block_size: int = LDLQ_BLOCK_SIZE,
    damping_fraction: float = LDLQ_DAMPING_FRACTION,
    batch_experts: bool | None = None,
    gate: bool | None = None,
    prepared: "PreparedLDLQEvidence | None" = None,
) -> tuple[dict, dict]:
    """Fixed-codebook LDLQ with per-unit do-no-harm fallback.

    Returns ``(fields, gate_info)`` where ``gate_info`` records whether the
    LDLQ arm was kept per Linear / per expert slice.  The byte payload is
    identical in either arm, so this is a pure quality gate.  When ``gate``
    is False the raw LDLQ result is returned verbatim (no comparison).
    """
    if gate is None:
        gate = _ldlq_gate_enabled()
    if not gate:
        ldlq_fields = ldlq_reassign_cb_fields(
            weight, fields, col_weights, activation_rows,
            grid=grid, mode=mode, block_size=block_size,
            damping_fraction=damping_fraction, batch_experts=batch_experts,
        )
        return ldlq_fields, {"gate": "disabled", "kept_ldlq": True, "metric": "activation_output_mse"}
    # Fail-closed early: missing/empty activation must not invoke LDLQ assignment
    # which would raise non-empty assertion; return explicit fallback instead.
    # For 3-D packed experts, missing experts are filled with the pooled
    # observed rows (identical to CBLDLQActivationLoader) so LDLQ sees 256
    # valid Hessians and the gate can emit 256 decisions. All-empty has no
    # observed rows to pool and fails closed with full telemetry.
    # Prefer the existing helper to avoid drift.
    if weight.ndim == 3:
        if activation_rows is None:
            return fields, {
                "gate": "raw_fallback_no_activation",
                "kept_ldlq": False,
                "reason": "activation_rows is None but LDLQ gate requires activation_output_mse",
                "metric": "activation_output_mse",
            }
        if isinstance(activation_rows, torch.Tensor):
            return fields, {
                "gate": "raw_fallback_shared_activation_for_packed",
                "kept_ldlq": False,
                "reason": "3-D weight requires per-expert activation sequence, got single tensor",
                "metric": "activation_output_mse",
            }
        # Validate sequence length / rank / width fail-closed before pooling.
        _seq_check = tuple(activation_rows)  # type: ignore[arg-type]
        if len(_seq_check) != int(weight.shape[0]):
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": f"3-D weight expects {int(weight.shape[0])} activation rows, got {len(_seq_check)}",
                "metric": "activation_output_mse",
            }
        for _idx, _act in enumerate(_seq_check):
            _t = torch.as_tensor(_act) if not isinstance(_act, torch.Tensor) else _act
            if _t.numel() == 0:
                continue
            if _t.ndim != 2:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"activation_rows[{_idx}] rank { _t.ndim } != 2",
                    "metric": "activation_output_mse",
                }
            if int(_t.shape[1]) != int(weight.shape[-1]):
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"activation_rows[{_idx}] width { int(_t.shape[1]) } != weight width { int(weight.shape[-1]) }",
                    "metric": "activation_output_mse",
                }
            if int(_t.shape[0]) == 0:
                # empty already handled above, but keep
                continue
    else:
        if activation_rows is None:
            return fields, {
                "gate": "raw_fallback_no_activation",
                "kept_ldlq": False,
                "reason": "activation_rows is None but LDLQ gate requires activation_output_mse",
                "metric": "activation_output_mse",
            }
        if isinstance(activation_rows, torch.Tensor):
            _t2 = torch.as_tensor(activation_rows)
            if _t2.numel() == 0 or int(_t2.shape[0]) == 0:
                return fields, {
                    "gate": "raw_fallback_missing_activation",
                    "kept_ldlq": False,
                    "reason": "activation rows empty or missing for 2-D gate",
                    "metric": "activation_output_mse",
                }
            if _t2.ndim != 2:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"2-D activation rank { _t2.ndim } != 2",
                    "metric": "activation_output_mse",
                }
            if int(_t2.shape[1]) != int(weight.shape[-1]):
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"2-D activation width { int(_t2.shape[1]) } != weight width { int(weight.shape[-1]) }",
                    "metric": "activation_output_mse",
                }
        else:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "2-D weight requires single Tensor activation, got sequence",
                "metric": "activation_output_mse",
            }
    # 3-D path: eligible-only LDLQ. Cold/insufficient experts are NEVER pooled or factored.
    # Eligible = not cold and not insufficient. LDLQ runs only on the eligible subset
    # while retaining batched/chunked GPU execution: slice weight/colweights/flattened
    # fields to eligible subset, run existing batched LDLQ path using stable prepared FIT
    # tensor objects, then stitch only eligible assignment/sign rows into a raw-field copy.
    if weight.ndim == 3:
        _seq_orig: tuple[torch.Tensor, ...]
        _fit_rows: list[torch.Tensor]
        _holdout_rows: list[torch.Tensor]
        _split_infos: list[dict]
        _eligible: tuple[int, ...]
        _orig_cold: set[int]
        _insufficient_set: set[int]
        if prepared is not None:
            if not isinstance(prepared, PreparedLDLQEvidence):
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared evidence wrong type {type(prepared).__name__}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            if prepared.policy != _SPLIT_POLICY or prepared.version != _SPLIT_VERSION:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared evidence policy/version mismatch expected {_SPLIT_POLICY}/{_SPLIT_VERSION} got {prepared.policy}/{prepared.version}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            _seq_orig = tuple(prepared.original_rows)
            if len(_seq_orig) != int(weight.shape[0]):
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared evidence expert count {len(_seq_orig)} != weight stack {int(weight.shape[0])}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            # Validate caller multiset against stored original digest (order-invariant within expert)
            try:
                _caller_seq = tuple(activation_rows) if not isinstance(activation_rows, torch.Tensor) else (torch.as_tensor(activation_rows),)  # type: ignore[arg-type]
            except Exception as exc:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared validation: caller rows unreadable {exc}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            if len(_caller_seq) != len(_seq_orig):
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared evidence expert count {len(_seq_orig)} != caller rows {len(_caller_seq)}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            # Check 1: caller vs stored original digest (multiset)
            for _i in range(len(_seq_orig)):
                if _partition_multiset_digest(torch.as_tensor(_caller_seq[_i])) != prepared.original_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared evidence caller content mismatch at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
            # Check 2: current aliased original vs stored original digest (detect in-place mutation after prepare)
            for _i in range(len(_seq_orig)):
                if _partition_multiset_digest(torch.as_tensor(_seq_orig[_i])) != prepared.original_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared evidence original mutated at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
            # Check 3: stable fit/holdout tensors vs stored digests (detect stale snapshots)
            for _i in range(len(_seq_orig)):
                if _partition_multiset_digest(torch.as_tensor(prepared.fit_rows[_i])) != prepared.fit_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared evidence fit mutated at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
                if _partition_multiset_digest(torch.as_tensor(prepared.holdout_rows[_i])) != prepared.holdout_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared evidence holdout mutated at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
            # Check 4: expert count/width via shape in digest already, but also validate width
            for _i, a in enumerate(_seq_orig):
                at = torch.as_tensor(a)
                if at.numel() > 0 and int(at.shape[1]) != int(weight.shape[-1]):
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared evidence width mismatch at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
            # Check 5: split_infos frozen digests vs stored fit/holdout (detect nested dict mutation)
            for _i, info in enumerate(prepared.split_infos):
                exp_fit = dict(info).get("fit_digest")
                exp_hold = dict(info).get("holdout_digest")
                if exp_fit is not None and exp_fit != prepared.fit_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared split_info fit_digest mutated at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
                if exp_hold is not None and exp_hold != prepared.holdout_digests[_i]:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": f"prepared split_info holdout_digest mutated at expert {_i}",
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
            # Check 6: overall split_infos digest (detect any nested mutation) — canonical helper, require exact attribute
            _cur_split_digest = _canonical_split_infos_digest(prepared.split_infos)
            if _cur_split_digest != prepared.split_infos_digest:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": "prepared split_infos digest mismatch (nested mutation)",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            _fit_rows = list(prepared.fit_rows)
            _holdout_rows = list(prepared.holdout_rows)
            _split_infos = [dict(s) for s in prepared.split_infos]
            _eligible = tuple(prepared.eligible_experts)
            _orig_cold = set(prepared.cold_experts)
            _insufficient_set = set(prepared.insufficient_experts) | _orig_cold
            if not _eligible:
                _per_kept = [False] * len(_seq_orig)
                _inf = [float("inf")] * len(_seq_orig)
                return fields, {
                    "gate": "raw_fallback_missing_activation",
                    "kept_ldlq": False,
                    "per_expert_kept": _per_kept,
                    "missing_experts": sorted(_orig_cold | _insufficient_set),
                    "raw_mse_per_expert": _inf,
                    "ldlq_mse_per_expert": _inf,
                    "metric": "activation_output_mse",
                    "reason": "no eligible experts (all cold/insufficient)",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                    "split_counts": _split_infos,
                }
        else:
            _seq_orig = tuple(activation_rows)  # type: ignore[arg-type]
            _fit_rows, _holdout_rows, _split_infos = deterministic_fit_holdout_split_per_expert(_seq_orig, version=_SPLIT_VERSION)
            _orig_cold = {i for i, a in enumerate(_seq_orig) if torch.as_tensor(a).numel() == 0}
            _insufficient_set = {info["expert"] for info in _split_infos if info.get("insufficient")} | _orig_cold
            _eligible = tuple(i for i in range(len(_seq_orig)) if i not in _insufficient_set)
            if not _eligible:
                _per_kept = [False] * len(_seq_orig)
                _inf = [float("inf")] * len(_seq_orig)
                return fields, {
                    "gate": "raw_fallback_missing_activation",
                    "kept_ldlq": False,
                    "per_expert_kept": _per_kept,
                    "missing_experts": sorted(_insufficient_set),
                    "raw_mse_per_expert": _inf,
                    "ldlq_mse_per_expert": _inf,
                    "metric": "activation_output_mse",
                    "reason": "no eligible experts for LDLQ",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                    "split_counts": _split_infos,
                }
        # Eligible-only LDLQ: slice to eligible subset, retain batched/chunked GPU execution
        _E = int(weight.shape[0])
        _R = int(weight.shape[1])
        _C = int(weight.shape[-1])
        _elig_list = list(_eligible)
        # Strict resolver shared with batched LDLQ path — malformed/nonpositive raises
        _expert_batch = _resolve_ldlq_expert_batch()
        from prismaquant import format_registry as _fr_gate
        _gate_spec = _fr_gate.get_format(f"NVFP4_CB_K{k}")
        # Precompute raw per-expert MSE once (chunked, no weight copy)
        try:
            raw_per_all = canonical_nvfp4_cb_per_expert_mse_chunked(
                weight, fields, k, grid, mode, _holdout_rows, _gate_spec
            )
        except ValueError as exc:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": str(exc),
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        raw_per = [float(v) for v in raw_per_all]
        _missing_experts = sorted(_orig_cold | _insufficient_set)
        holdout_missing = [int(i) for i, h in enumerate(_holdout_rows) if torch.as_tensor(h).numel() == 0]
        ldlq_per = [float("inf")] * _E
        keep_mask: list[bool] = [False] * _E

        # All-eligible zero-copy fast path: pass original weight/fields/colweights and stable FIT tuple
        if len(_elig_list) == _E:
            fit_all = tuple(_fit_rows[i] for i in _elig_list)
            ldlq_fields_full = ldlq_reassign_cb_fields(
                weight, fields, col_weights, fit_all,
                grid=grid, mode=mode, block_size=block_size,
                damping_fraction=damping_fraction, batch_experts=batch_experts,
            )
            try:
                ldlq_per_all = canonical_nvfp4_cb_per_expert_mse_chunked(
                    weight, ldlq_fields_full, k, grid, mode, _holdout_rows, _gate_spec
                )
            except ValueError as exc:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": str(exc),
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            for elig in _elig_list:
                ldlq_per[elig] = float(ldlq_per_all[elig])
                raw_err = raw_per[elig]
                ldlq_err = ldlq_per[elig]
                if raw_err != float("inf") and ldlq_err != float("inf"):
                    keep = ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0)
                else:
                    keep = False
                keep_mask[elig] = bool(keep)
            if not any(keep_mask):
                _out2: dict = {
                    "gate": "raw_kept_all",
                    "kept_ldlq": False,
                    "per_expert_kept": keep_mask,
                    "raw_mse_per_expert": raw_per,
                    "ldlq_mse_per_expert": ldlq_per,
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                    "split_counts": [dict(s) for s in _split_infos],
                    "holdout_missing_experts": holdout_missing,
                }
                if _missing_experts:
                    _out2["missing_experts"] = list(_missing_experts)
                    _out2["cold_experts"] = sorted(_orig_cold)
                return fields, _out2
            # ldlq_kept_all only when every physical expert is eligible and kept (scalar True) — zero-copy
            if len(_elig_list) == _E and all(keep_mask[i] for i in _elig_list):
                _out: dict = {
                    "gate": "ldlq_kept_all",
                    "kept_ldlq": True,
                    "per_expert_kept": keep_mask,
                    "raw_mse_per_expert": raw_per,
                    "ldlq_mse_per_expert": ldlq_per,
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                    "split_counts": [dict(s) for s in _split_infos],
                    "holdout_missing_experts": holdout_missing,
                }
                if _missing_experts:
                    _out["missing_experts"] = list(_missing_experts)
                    _out["cold_experts"] = sorted(_orig_cold)
                return ldlq_fields_full, _out
            # Mixed: some eligible kept but not all (or all eligible but mixed due to missing? already handled)
            updated_mixed = dict(fields)
            for _k in ("indices", "signs"):
                if _k in fields and isinstance(fields[_k], torch.Tensor):
                    updated_mixed[_k] = fields[_k].clone()
            for elig in _elig_list:
                if not keep_mask[elig]:
                    continue
                src_start = elig * _R
                src_end = src_start + _R
                for _k in ("indices", "signs"):
                    if _k in ldlq_fields_full and isinstance(ldlq_fields_full[_k], torch.Tensor) and _k in updated_mixed:
                        updated_mixed[_k][src_start:src_end] = ldlq_fields_full[_k][src_start:src_end]
            _mixed_out: dict = {
                "gate": "mixed_per_expert",
                "kept_ldlq": keep_mask,
                "per_expert_kept": keep_mask,
                "raw_mse_per_expert": raw_per,
                "ldlq_mse_per_expert": ldlq_per,
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
                "split_counts": [dict(s) for s in _split_infos],
                "holdout_missing_experts": holdout_missing,
            }
            if _missing_experts:
                _mixed_out["missing_experts"] = list(_missing_experts)
                _mixed_out["cold_experts"] = sorted(_orig_cold)
            return updated_mixed, _mixed_out
        else:
            # Mixed eligible: bounded chunked processing no near-full copy
            _device_for_idx = fields["indices"].device if isinstance(fields.get("indices"), torch.Tensor) else weight.device
            cw_tensor = torch.as_tensor(col_weights) if not isinstance(col_weights, torch.Tensor) else col_weights
            updated = None  # lazy clone only after first kept
            # Process eligible experts in bounded chunks
            for _cs in range(0, len(_elig_list), _expert_batch):
                chunk = _elig_list[_cs:_cs + _expert_batch]
                is_contig = chunk == list(range(chunk[0], chunk[0] + len(chunk)))
                if is_contig:
                    c_start = chunk[0]
                    c_end = c_start + len(chunk)
                    weight_chunk = weight[c_start:c_end]
                    if isinstance(cw_tensor, torch.Tensor) and cw_tensor.ndim == 3 and int(cw_tensor.shape[0]) == _E:
                        cw_chunk = cw_tensor[c_start:c_end]
                    elif isinstance(cw_tensor, torch.Tensor) and cw_tensor.ndim == 2 and int(cw_tensor.shape[0]) == _E:
                        cw_chunk = cw_tensor[c_start:c_end]
                    else:
                        cw_chunk = col_weights
                    fields_chunk: dict = {}
                    for fk, fv in fields.items():
                        if fk in ("indices", "signs", "scales", "scale_super", "scale_sub") and isinstance(fv, torch.Tensor):
                            fields_chunk[fk] = fv[c_start * _R:c_end * _R]
                        else:
                            fields_chunk[fk] = fv
                    fields_chunk["shape"] = (len(chunk), _R, _C)
                else:
                    idx = torch.tensor(chunk, device=weight.device)
                    weight_chunk = weight[idx]
                    if isinstance(cw_tensor, torch.Tensor) and cw_tensor.ndim == 3 and int(cw_tensor.shape[0]) == _E:
                        cw_chunk = cw_tensor[idx]
                    elif isinstance(cw_tensor, torch.Tensor) and cw_tensor.ndim == 2 and int(cw_tensor.shape[0]) == _E:
                        cw_chunk = cw_tensor[idx]
                    else:
                        cw_chunk = col_weights
                    row_idx_parts = []
                    for e in chunk:
                        row_idx_parts.append(torch.arange(e * _R, (e + 1) * _R, device=_device_for_idx))
                    row_idx_chunk = torch.cat(row_idx_parts) if row_idx_parts else torch.empty((0,), dtype=torch.long, device=_device_for_idx)
                    fields_chunk = {}
                    for fk, fv in fields.items():
                        if fk in ("indices", "signs", "scales", "scale_super", "scale_sub") and isinstance(fv, torch.Tensor):
                            fields_chunk[fk] = fv[row_idx_chunk]
                        else:
                            fields_chunk[fk] = fv
                    fields_chunk["shape"] = (len(chunk), _R, _C)
                fit_chunk = tuple(_fit_rows[i] for i in chunk)
                holdout_chunk = tuple(_holdout_rows[i] for i in chunk)
                ldlq_fields_chunk = ldlq_reassign_cb_fields(
                    weight_chunk, fields_chunk, cw_chunk, fit_chunk,
                    grid=grid, mode=mode, block_size=block_size,
                    damping_fraction=damping_fraction, batch_experts=batch_experts,
                )
                try:
                    ldlq_per_chunk = canonical_nvfp4_cb_per_expert_mse_chunked(
                        weight_chunk, ldlq_fields_chunk, k, grid, mode, holdout_chunk, _gate_spec
                    )
                except ValueError as exc:
                    return fields, {
                        "gate": "raw_fallback_malformed_activation",
                        "kept_ldlq": False,
                        "reason": str(exc),
                        "metric": "activation_output_mse",
                        "split_policy": _SPLIT_POLICY,
                        "split_version": _SPLIT_VERSION,
                    }
                # Decide keep for this chunk and stitch
                for _idx_in_chunk, _elig in enumerate(chunk):
                    ldlq_val = float(ldlq_per_chunk[_idx_in_chunk])
                    ldlq_per[_elig] = ldlq_val
                    raw_val = raw_per[_elig]
                    if raw_val != float("inf") and ldlq_val != float("inf"):
                        keep = ldlq_val <= raw_val + LDLQ_GATE_EPSILON * max(abs(raw_val), abs(ldlq_val), 1.0)
                    else:
                        keep = False
                    keep_mask[_elig] = bool(keep)
                # Stitch kept rows from this chunk
                if any(keep_mask[e] for e in chunk):
                    if updated is None:
                        updated = dict(fields)
                        for _k in ("indices", "signs"):
                            if _k in fields and isinstance(fields[_k], torch.Tensor):
                                updated[_k] = fields[_k].clone()
                    for _idx_in_chunk, _elig in enumerate(chunk):
                        if not keep_mask[_elig]:
                            continue
                        src_start = _idx_in_chunk * _R
                        src_end = src_start + _R
                        dst_start = _elig * _R
                        dst_end = dst_start + _R
                        for _k in ("indices", "signs"):
                            if _k in ldlq_fields_chunk and isinstance(ldlq_fields_chunk[_k], torch.Tensor) and _k in updated:
                                updated[_k][dst_start:dst_end] = ldlq_fields_chunk[_k][src_start:src_end]
                # Bound temporary weight copy to one chunk — delete temporaries and let allocator reuse; no empty_cache in hot loop
                del weight_chunk, fields_chunk, ldlq_fields_chunk
            if not any(keep_mask):
                _out2: dict = {
                    "gate": "raw_kept_all",
                    "kept_ldlq": False,
                    "per_expert_kept": keep_mask,
                    "raw_mse_per_expert": raw_per,
                    "ldlq_mse_per_expert": ldlq_per,
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                    "split_counts": [dict(s) for s in _split_infos],
                    "holdout_missing_experts": holdout_missing,
                }
                if _missing_experts:
                    _out2["missing_experts"] = list(_missing_experts)
                    _out2["cold_experts"] = sorted(_orig_cold)
                return fields, _out2
            # ldlq_kept_all only when every physical expert eligible and kept — never for mixed with cold
            # For mixed path, len < E, so never ldlq_kept_all
            _mixed_out: dict = {
                "gate": "mixed_per_expert",
                "kept_ldlq": keep_mask,
                "per_expert_kept": keep_mask,
                "raw_mse_per_expert": raw_per,
                "ldlq_mse_per_expert": ldlq_per,
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
                "split_counts": [dict(s) for s in _split_infos],
                "holdout_missing_experts": holdout_missing,
            }
            if _missing_experts:
                _mixed_out["missing_experts"] = list(_missing_experts)
                _mixed_out["cold_experts"] = sorted(_orig_cold)
            assert updated is not None
            return updated, _mixed_out
    # 2-D path: deterministic content-hash split, single LDLQ on fit, gate on holdout
    if activation_rows is None:
        return fields, {
            "gate": "raw_fallback_no_activation",
            "kept_ldlq": False,
            "reason": "activation_rows is None but LDLQ gate requires activation_output_mse",
            "metric": "activation_output_mse",
            "split_policy": _SPLIT_POLICY,
            "split_version": _SPLIT_VERSION,
        }
    _act_2d = torch.as_tensor(activation_rows)
    if _act_2d.numel() == 0 or int(_act_2d.shape[0]) < 2:
        return fields, {
            "gate": "raw_fallback_missing_activation",
            "kept_ldlq": False,
            "reason": "dense rows <2 insufficient for fit+holdout",
            "metric": "activation_output_mse",
            "split_policy": _SPLIT_POLICY,
            "split_version": _SPLIT_VERSION,
        }
    # Deterministic split for dense: reuse per-expert helper with single entry
    # Use prepared evidence when supplied (single-owner, validated fail-closed)
    if prepared is not None:
        if not isinstance(prepared, PreparedLDLQEvidence):
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": f"prepared evidence wrong type {type(prepared).__name__}",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        if prepared.policy != _SPLIT_POLICY or prepared.version != _SPLIT_VERSION:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": f"prepared evidence policy/version mismatch expected {_SPLIT_POLICY}/{_SPLIT_VERSION} got {prepared.policy}/{prepared.version}",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        if len(prepared.original_rows) != 1:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": f"prepared evidence expert count {len(prepared.original_rows)} != 1 for dense",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        if _partition_multiset_digest(prepared.original_rows[0]) != _partition_multiset_digest(_act_2d):
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "prepared evidence content mismatch for dense",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        # Additional immutable checks for dense (detect in-place mutation)
        if _partition_multiset_digest(torch.as_tensor(prepared.original_rows[0])) != prepared.original_digests[0]:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "prepared evidence original mutated for dense",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        if len(prepared.fit_digests) and _partition_multiset_digest(torch.as_tensor(prepared.fit_rows[0])) != prepared.fit_digests[0]:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "prepared evidence fit mutated for dense",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        if len(prepared.holdout_digests) and _partition_multiset_digest(torch.as_tensor(prepared.holdout_rows[0])) != prepared.holdout_digests[0]:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "prepared evidence holdout mutated for dense",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        # Validate split_info digests
        for _ii, _info_chk in enumerate(prepared.split_infos):
            _d = dict(_info_chk)
            if _d.get("fit_digest") is not None and _d.get("fit_digest") != prepared.fit_digests[_ii]:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared split_info fit_digest mutated at dense expert {_ii}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
            if _d.get("holdout_digest") is not None and _d.get("holdout_digest") != prepared.holdout_digests[_ii]:
                return fields, {
                    "gate": "raw_fallback_malformed_activation",
                    "kept_ldlq": False,
                    "reason": f"prepared split_info holdout_digest mutated at dense expert {_ii}",
                    "metric": "activation_output_mse",
                    "split_policy": _SPLIT_POLICY,
                    "split_version": _SPLIT_VERSION,
                }
        _cur_dense_split = _canonical_split_infos_digest(prepared.split_infos)
        if _cur_dense_split != prepared.split_infos_digest:
            return fields, {
                "gate": "raw_fallback_malformed_activation",
                "kept_ldlq": False,
                "reason": "prepared split_infos digest mismatch (nested mutation)",
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
            }
        _fit_2d = prepared.fit_rows[0] if len(prepared.fit_rows) else torch.empty((0, int(_act_2d.shape[1])), dtype=torch.float32)
        _hold_2d = prepared.holdout_rows[0] if len(prepared.holdout_rows) else torch.empty((0, int(_act_2d.shape[1])), dtype=torch.float32)
        _infos2 = [dict(s) for s in prepared.split_infos]
        _info2 = _infos2[0] if _infos2 else {"insufficient": True}
        if _info2.get("insufficient") or not prepared.has_observed_fit:
            return fields, {
                "gate": "raw_fallback_missing_activation",
                "kept_ldlq": False,
                "reason": _info2.get("reason", "insufficient rows for split"),
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
                "split_counts": _infos2,
            }
    else:
        _fit_list, _hold_list, _infos2 = deterministic_fit_holdout_split_per_expert([_act_2d], version=_SPLIT_VERSION)
        _fit_2d, _hold_2d = _fit_list[0], _hold_list[0]
        _info2 = _infos2[0]
        if _info2.get("insufficient"):
            return fields, {
                "gate": "raw_fallback_missing_activation",
                "kept_ldlq": False,
                "reason": _info2.get("reason", "insufficient rows for split"),
                "metric": "activation_output_mse",
                "split_policy": _SPLIT_POLICY,
                "split_version": _SPLIT_VERSION,
                "split_counts": _infos2,
            }
    # LDLQ on fit only (ungated)
    ldlq_fields = ldlq_reassign_cb_fields(
        weight, fields, col_weights, _fit_2d,
        grid=grid, mode=mode, block_size=block_size,
        damping_fraction=damping_fraction, batch_experts=batch_experts,
    )
    from prismaquant import format_registry as _fr_gate2
    _gate_spec2 = _fr_gate2.get_format(f"NVFP4_CB_K{k}")
    try:
        raw_act = canonical_nvfp4_cb_dense_mse_chunked(weight, fields, k, grid, mode, _hold_2d, _gate_spec2)
        ldlq_act = canonical_nvfp4_cb_dense_mse_chunked(weight, ldlq_fields, k, grid, mode, _hold_2d, _gate_spec2)
    except ValueError as exc:
        return fields, {
            "gate": "raw_fallback_malformed_activation",
            "kept_ldlq": False,
            "reason": str(exc),
            "metric": "activation_output_mse",
            "split_policy": _SPLIT_POLICY,
            "split_version": _SPLIT_VERSION,
        }
    if raw_act is None or ldlq_act is None:
        return fields, {
            "gate": "raw_fallback_missing_activation",
            "kept_ldlq": False,
            "reason": "holdout empty",
            "metric": "activation_output_mse",
            "split_policy": _SPLIT_POLICY,
            "split_version": _SPLIT_VERSION,
        }
    raw_err, ldlq_err = float(raw_act), float(ldlq_act)
    if ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0):
        return ldlq_fields, {
            "gate": "ldlq_kept",
            "kept_ldlq": True,
            "raw_mse": raw_err,
            "ldlq_mse": ldlq_err,
            "metric": "activation_output_mse",
            "split_policy": _SPLIT_POLICY,
            "split_version": _SPLIT_VERSION,
            "split_counts": _infos2,
        }
    return fields, {
        "gate": "raw_kept",
        "kept_ldlq": False,
        "raw_mse": raw_err,
        "ldlq_mse": ldlq_err,
        "metric": "activation_output_mse",
        "split_policy": _SPLIT_POLICY,
        "split_version": _SPLIT_VERSION,
        "split_counts": _infos2,
    }


def _nvfp4_cb_reconstruct_one(fields: dict, k: int, *, grid: str = "fp4",
                              mode: str = "product",
                              codebook: torch.Tensor | tuple | None = None) -> torch.Tensor:
    """Monolithic decode for a single chunk — no internal chunking."""
    shape = fields.get("shape")
    indices = fields["indices"]
    scales = fields["scales"]
    rows = indices.shape[0]
    in_f = int(shape[-1]) if shape is not None else scales.shape[-1] * (
        FP4_GROUP if grid == "fp4" else 1)
    cb = fields.get("codebook")
    if cb is None:
        cb = _resolve_codebook(k, grid, mode, codebook, indices.device)
    if mode == "full":
        vecs = cb[indices]                                   # (rows, nvec, 8)
        recon = vecs.reshape(rows, in_f)
    elif mode == "signed":
        vecs = cb[indices]                                   # (rows, nvec, 8)
        recon = vecs.reshape(rows, in_f) * fields["signs"]
    else:
        parts = [table[indices[..., i]] for i, table in enumerate(cb)]
        recon = torch.cat(parts, dim=-1).reshape(rows, in_f)
    pes = _per_element_scale(scales.to(recon.dtype), grid, in_f)
    recon = recon * pes
    if shape is not None:
        recon = recon.reshape(shape)
    return recon


def nvfp4_cb_reconstruct(fields: dict, k: int, *, grid: str = "fp4",
                         mode: str = "product",
                         codebook: torch.Tensor | tuple | None = None
                         ) -> torch.Tensor:
    """Memory-bounded reconstruction: chunk over outer expert dimension.

    For 3-D packed weights (E, R, C) the authoritative chunk unit is the
    expert (outer dimension). For 2-D or shape-absent tensors the chunk unit
    is rows. Each chunk's ``torch.cat`` temporaries are therefore bounded to
    ``chunk_experts * R * in_f`` elements rather than the full ``E * R * in_f``
    (the 16 GiB cat on layer-0 fused gate_up_proj with E=256).

    The result is bit-identical to the monolithic decode; chunk boundaries
    recompute the same per-row dequant. Chunk size comes from
    ``PRISMAQUANT_CB_RECON_EXPERT_CHUNK`` (default 8, ~512 MiB on E=256,R=4096,C=4096) and
    ``PRISMAQUANT_CB_RECON_ROW_CHUNK`` (default 4096) and is validated.
    """
    shape = fields.get("shape")
    indices = fields["indices"]
    rows = int(indices.shape[0])
    # Packed 3-D: chunk over expert outer dimension via public validated iterator
    if shape is not None and len(shape) == 3:
        E, R, C = map(int, shape)
        if rows != E * R:
            raise ValueError(
                f"nvfp4_cb_reconstruct: malformed 3-D shape {tuple(shape)} inconsistent with "
                f"indices rows {rows} (expected {E}*{R}={E*R}); fail-closed—refusing invalid reshape"
            )
        chunk_experts = _resolve_recon_expert_chunk()
        if E <= chunk_experts:
            return _nvfp4_cb_reconstruct_one(fields, k, grid=grid, mode=mode, codebook=codebook)
        out = torch.empty(shape, device=indices.device, dtype=torch.float32)
        for (first, last), chunk_recon, _chunk_fields in iter_nvfp4_cb_recon_chunks(
            fields, k, grid=grid, mode=mode, codebook=codebook
        ):
            out[first:last].copy_(chunk_recon)
            del chunk_recon, _chunk_fields
        return out
    # 2-D or shape-absent: chunk over rows
    chunk_rows = _resolve_recon_row_chunk()
    if rows <= chunk_rows:
        return _nvfp4_cb_reconstruct_one(fields, k, grid=grid, mode=mode, codebook=codebook)
    in_f = int(shape[-1]) if shape is not None else int(fields["scales"].shape[-1]) * (FP4_GROUP if grid == "fp4" else 1)
    # Allocate final as (rows, in_f) then reshape to shape if needed
    flat_out = torch.empty((rows, in_f), device=indices.device, dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk_len = end - start
        chunk_shape_2d = (chunk_len, in_f)
        chunk_fields = _slice_fields_for_rows(fields, start, end, new_shape=chunk_shape_2d)
        chunk_recon = _nvfp4_cb_reconstruct_one(chunk_fields, k, grid=grid, mode=mode, codebook=codebook)
        flat_out[start:end].copy_(chunk_recon.view(chunk_len, in_f))
        del chunk_recon, chunk_fields
    if shape is not None:
        return flat_out.view(shape)
    return flat_out


def make_nvfp4_cb_qdq(k: int, grid: str = "fp4", mode: str = "product",
                      scale_sweep: bool = True,
                      scale_coding: str = SCALE_CODING_V1,
                      encode_tier: str | None = None):
    """Single-source emulation closure ``(w, col_weights=None) -> w_hat``
    used by both cost and (Milestone B) the packer. ``scale_sweep`` defaults
    True (joint scale search, IQ-rendering parity); ``scale_coding``
    selects the v1 e4m3 plane (default) or the layout-v2 two-tier coding;
    ``encode_tier`` the fast/balanced/max speed-accuracy tier (None ->
    PRISMAQUANT_CB_ENCODE_TIER, resolved per call)."""
    def f(w: torch.Tensor, col_weights: torch.Tensor | None = None
          ) -> torch.Tensor:
        fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                 col_weights=col_weights,
                                 scale_sweep=scale_sweep,
                                 scale_coding=scale_coding,
                                 encode_tier=encode_tier)
        return nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    return f


# ---------------------------------------------------------------------------
# RD-law ladder interpolation (cost path, opt-in): encode a few ANCHOR rungs,
# fit the one-parameter rate-distortion law D(k) = C * 2^(-k/4) per Linear
# (log2 D linear in k with fixed slope -1/4), predict weighted-recon cost at
# every other rung, and report a HOLDOUT check so the caller can fall back to
# full measurement where the fit is not trusted. This is "surrogate proposes,
# measurement verifies" — the anchors and the holdout ARE measurements; no
# unverified closed form ships (the analytical-damp graveyard rule).
# Wiring point: the local-cost path may consult this behind
# PRISMAQUANT_CB_LADDER_INTERP=1 (default OFF; cost-path wiring belongs to
# the menu-integration workstream — this module only provides the helper).
# ---------------------------------------------------------------------------

RD_LAW_SLOPE_BITS = 0.25          # log2 D drop per index bit (D ~ 2^(-k/4))


def _weighted_recon_cost(w: torch.Tensor, k: int, *, grid: str, mode: str,
                         col_weights: torch.Tensor | None,
                         scale_coding: str, encode_tier: str | None) -> float:
    qdq = make_nvfp4_cb_qdq(k, grid, mode, scale_coding=scale_coding,
                            encode_tier=encode_tier)
    w_hat = qdq(w, col_weights)
    err = (w.to(torch.float32) - w_hat.to(torch.float32)).pow(2)
    if col_weights is not None:
        cw = torch.broadcast_to(
            col_weights.to(w.device, torch.float32), w.shape)
        err = err * cw
    return float(err.sum())


def predict_cb_ladder_costs(w: torch.Tensor, ks: tuple[int, ...], *,
                            grid: str = "fp4", mode: str = "product",
                            col_weights: torch.Tensor | None = None,
                            anchors: tuple[int, ...] = (12, 18, 24),
                            holdout: int | None = None,
                            scale_coding: str = SCALE_CODING_V1,
                            encode_tier: str | None = None) -> dict:
    """Predict per-rung weighted-recon cost from a few measured anchors.

    Encodes ``anchors`` (and ``holdout`` if given) for real, fits
    ``log2 D(k) = log2 C - k/4`` (one parameter ``C``), and returns::

        {"measured": {k: D}, "predicted": {k: D_hat for k in ks},
         "log2_C": float, "holdout": {"k", "measured", "predicted",
                                      "rel_error"} | None}

    The caller trusts the interpolation only where the holdout relative
    error clears its noise floor (docs/lanes/nvfp4-cb/encode_tiers.md §B),
    falling back to full per-rung measurement elsewhere.
    """
    measured: dict[int, float] = {}
    for ak in anchors:
        measured[int(ak)] = _weighted_recon_cost(
            w, int(ak), grid=grid, mode=mode, col_weights=col_weights,
            scale_coding=scale_coding, encode_tier=encode_tier)
    logc = sum(
        (torch.log2(torch.tensor(d)).item() + RD_LAW_SLOPE_BITS * ak)
        for ak, d in measured.items()) / len(measured)
    predicted = {int(k): float(2.0 ** (logc - RD_LAW_SLOPE_BITS * int(k)))
                 for k in ks}
    hold = None
    if holdout is not None:
        h_meas = _weighted_recon_cost(
            w, int(holdout), grid=grid, mode=mode, col_weights=col_weights,
            scale_coding=scale_coding, encode_tier=encode_tier)
        h_pred = float(2.0 ** (logc - RD_LAW_SLOPE_BITS * int(holdout)))
        hold = {"k": int(holdout), "measured": h_meas, "predicted": h_pred,
                "rel_error": abs(h_pred - h_meas) / max(h_meas, 1e-30)}
    return {"measured": measured, "predicted": predicted,
            "log2_C": float(logc), "holdout": hold}


# ---------------------------------------------------------------------------
# Milestone B — byte packers (export path). Bit-exact on-disk layout:
# docs/lanes/nvfp4-cb/format-pipeline.md §1 / docs/lanes/nvfp4-cb/LAYOUT.md.
#
# The index body, scale-plane sizes, and total type size come only from
# ``cb_layout``. The packed tensor is 2-D uint8 (rows, bytes_per_row) — NEVER
# a flat 1-D buffer (the GGUF
# lesson: a flat store loses the logical row/superblock structure the reader
# and serving kernel index into). fp8 ships NO scale plane in the weight bytes;
# its per-output-channel fp32 scales are a separate ``<name>.weight_scale``.
# ---------------------------------------------------------------------------

def nvfp4_cb_type_size(k: int, grid: str = "fp4",
                       scale_coding: str = SCALE_CODING_V1) -> int:
    """Public: on-disk bytes per 256-weight superblock for a CB rung."""
    return _serialized_type_size(k, grid, scale_coding)


def nvfp4_cb_effective_bits(k: int, grid: str = "fp4",
                            scale_coding: str = SCALE_CODING_V1) -> float:
    """Version-keyed body bpw (spec §2): fp4 v1 ``k/8 + 0.5``, fp4 v2
    two-tier ``k/8 + 0.28125``, fp8 ``k/8`` (per-channel scale separate).
    Registered FormatSpec rates are nominal compatibility metadata; exact
    producer pricing is versioned by ``CBSerializationContext`` and asserted
    against the serialized payload."""
    return _serialized_type_size(k, grid, scale_coding) * 8.0 / SUPERBLOCK


def _vector_codes(fields: dict, k: int, grid: str, mode: str) -> torch.Tensor:
    """Per-8-weight-vector k-bit codeword (rows, nvec_per_row), int64.

    Bit layout inside the k-bit field (LSB-first), per §1.1 + the PLAN's
    product decomposition:
      * full   — the codebook index itself (k bits);
      * product— the n_sub sub-indices contiguous, sub0 in the low bits
                 (bit widths ``subtable_bit_widths(k, "product", n_sub)``,
                 ceil-first);
      * signed — 8 sign bits (bit j == coord j is negative) in the low byte,
                 then the (k-8)-bit magnitude index above them.
    """
    idx = fields["indices"]
    rows = idx.shape[0]
    if mode == "full":
        return idx.reshape(rows, -1).to(torch.int64)
    if mode == "signed":
        mag = idx.reshape(rows, -1).to(torch.int64)               # (rows, nvec)
        signs = fields["signs"].reshape(rows, -1, VEC_DIM)        # (rows,nvec,8)
        neg = (signs < 0).to(torch.int64)
        shifts = torch.arange(VEC_DIM, device=neg.device)
        sign_byte = (neg << shifts).sum(dim=-1)                   # (rows, nvec)
        return sign_byte | (mag << VEC_DIM)
    # product: idx is (rows, nvec, canonical family n_sub)
    n_sub = family_for(grid, mode).n_sub
    if idx.shape[-1] != n_sub:
        raise ValueError(
            f"{grid} {mode} index stream has {idx.shape[-1]} subtables; "
            f"serialized family requires {n_sub}"
        )
    bits = subtable_bit_widths(k, mode, n_sub)
    code = torch.zeros(idx.shape[:-1], dtype=torch.int64, device=idx.device)
    off = 0
    for i in range(n_sub):
        code = code | (idx[..., i].to(torch.int64) << off)
        off += bits[i]
    return code.reshape(rows, -1)


def _pack_codes_to_bytes(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Pack k-bit codewords into the canonical superblock index body.

    Each superblock is byte-aligned, so the codewords pack contiguously
    LSB-first (codeword c
    occupies stream bits [c*k, c*k+k), its own LSB first; bytes fill LSB-first).
    """
    rows, nvec = codes.shape
    if nvec % CODEWORDS_PER_SUPERBLOCK:
        raise ValueError(
            f"codeword count {nvec} is not divisible by "
            f"{CODEWORDS_PER_SUPERBLOCK}"
        )
    n_sb = nvec // CODEWORDS_PER_SUPERBLOCK
    shifts = torch.arange(k, device=codes.device)
    wt = 1 << torch.arange(8, device=codes.device)
    # Chunk over rows: the (chunk, nvec, k) int64 bitstream transient is
    # ~16x the packed bytes — unchunked it is ~155GB on a Hy3 192-expert
    # stack (three box-wide OOMs, 2026-07-19). Rows are independent, so
    # chunking is bit-identical.
    step = max(1, _SLICE_MAX_ELEMS // max(nvec * k, 1))
    index_bytes = INDEX_BYTES_PER_K * k
    out = torch.empty(rows, n_sb, index_bytes, dtype=torch.uint8,
                      device=codes.device)
    for a in range(0, rows, step):
        b = min(rows, a + step)
        bits = (codes[a:b].unsqueeze(-1) >> shifts) & 1      # (chunk, nvec, k)
        bits = bits.reshape(b - a, n_sb, index_bytes, 8)
        out[a:b] = (bits * wt).sum(dim=-1).to(torch.uint8)
    return out                                               # (rows,n_sb,4k)


def _unpack_bytes_to_codes(idx_bytes: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of :func:`_pack_codes_to_bytes` under ``cb_layout``."""
    rows, n_sb, _ = idx_bytes.shape
    bshift = torch.arange(8, device=idx_bytes.device)
    bits = (idx_bytes.to(torch.int64).unsqueeze(-1) >> bshift) & 1
    bits = bits.reshape(
        rows, n_sb, CODEWORDS_PER_SUPERBLOCK, k
    )                                                          # k bits/codeword
    kshift = torch.arange(k, device=idx_bytes.device)
    codes = (bits << kshift).sum(dim=-1)                        # (rows,n_sb,32)
    return codes.reshape(rows, n_sb * CODEWORDS_PER_SUPERBLOCK)


def _scale_plane_bytes(scales: torch.Tensor, n_sb: int) -> torch.Tensor:
    """Encode the canonical fp4-v1 scale plane. ``scales`` (rows, in//16)
    are already E4M3-exact (snapped by the encoder), so the E4M3 byte view is
    lossless."""
    rows = scales.shape[0]
    s = scales.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK
    ).to(_E4M3)
    return s.contiguous().view(torch.uint8)


def _two_tier_scale_bytes(super_e: torch.Tensor, sub_c: torch.Tensor,
                          n_sb: int) -> torch.Tensor:
    """Encode the canonical fp4-v2 two-tier scale plane.

    The first byte is E8M0; remaining bytes pack two 4-bit subscale codes,
    with the even group in the low nibble. Spec §5.1.
    """
    rows = super_e.shape[0]
    sup = super_e.reshape(rows, n_sb, 1).to(torch.uint8)
    c = sub_c.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK
    ).to(torch.int64)
    if bool((c < 0).any()) or bool((c > 15).any()):
        raise ValueError("two-tier sub codes must be 4-bit (0..15)")
    pairs = c.reshape(
        rows, n_sb, FP4_SCALE_GROUPS_PER_SUPERBLOCK // 2, 2
    )
    sub = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    return torch.cat([sup, sub], dim=-1)


def _two_tier_scale_unpack(sc_bytes: torch.Tensor):
    """Decode the canonical fp4-v2 scale plane to exact E4M3 scales."""
    expected = SCALE_PLANE_BYTES[("fp4", SCALE_CODING_TWO_TIER)]
    if sc_bytes.shape[-1] != expected:
        raise ValueError(
            f"two-tier scale plane has {sc_bytes.shape[-1]} bytes; "
            f"expected {expected}"
        )
    rows, n_sb, _ = sc_bytes.shape
    super_e = sc_bytes[..., 0].to(torch.int64)
    sub = sc_bytes[..., 1:].to(torch.int64)                     # (rows,n_sb,8)
    lo = sub & 0xF
    hi = (sub >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, n_sb, -1)
    _, compose, legal = _two_tier_tables(str(sc_bytes.device))
    e_exp = super_e.unsqueeze(-1).expand_as(codes)
    if not bool(legal[e_exp, codes].all()):
        raise ValueError(
            "two-tier scale bytes contain an illegal (super, sub) pair")
    scales = compose[e_exp, codes].reshape(
        rows, n_sb * FP4_SCALE_GROUPS_PER_SUPERBLOCK
    )
    return super_e, codes.reshape(rows, -1), scales


def nvfp4_cb_assemble_bytes(fields: dict, k: int, grid: str = "fp4",
                            mode: str = "product") -> torch.Tensor:
    """Bit-pack VQ ``fields`` into the §1 on-disk byte layout.

    Returns a 2-D uint8 tensor ``(rows, n_superblocks * type_size)`` on the
    fields' device. The scale coding is taken from the fields (a two-tier
    encode carries ``scale_super``/``scale_sub``); the size is resolved from
    :mod:`prismaquant.cb_layout` and asserted.
    """
    k = int(k)
    scale_coding = fields.get("scale_coding", SCALE_CODING_V1)
    codes = _vector_codes(fields, k, grid, mode)                # (rows, nvec)
    rows, nvec = codes.shape
    if nvec % CODEWORDS_PER_SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={nvec * VEC_DIM} is not a multiple of {SUPERBLOCK}")
    n_sb = nvec // CODEWORDS_PER_SUPERBLOCK
    idx_bytes = _pack_codes_to_bytes(codes, k)
    if grid == "fp4" and scale_coding == SCALE_CODING_TWO_TIER:
        sc_bytes = _two_tier_scale_bytes(
            fields["scale_super"], fields["scale_sub"], n_sb)
        block = torch.cat([idx_bytes, sc_bytes], dim=-1)
    elif grid == "fp4":
        sc_bytes = _scale_plane_bytes(fields["scales"], n_sb)
        block = torch.cat([idx_bytes, sc_bytes], dim=-1)
    elif grid == "fp8":
        block = idx_bytes
    else:
        raise ValueError(f"unknown grid {grid!r}")
    ts = _serialized_type_size(k, grid, scale_coding)
    assert block.shape[-1] == ts, (
        f"type_size mismatch: packed {block.shape[-1]} bytes/superblock, "
        f"expected {ts} for k={k} grid={grid} scale_coding={scale_coding}")
    return block.reshape(rows, n_sb * ts).contiguous()


def nvfp4_cb_unpack(packed: torch.Tensor, k: int, grid: str, mode: str,
                    shape: tuple[int, ...],
                    codebook: torch.Tensor | tuple | None = None,
                    scales: torch.Tensor | None = None,
                    scale_coding: str = SCALE_CODING_V1) -> dict:
    """Inverse of :func:`nvfp4_cb_assemble_bytes`: byte tensor -> VQ ``fields``
    ready for :func:`nvfp4_cb_reconstruct`.

    fp4 scales are recovered from the packed scale section — the v1 e4m3
    plane by default; pass ``scale_coding="two_tier"`` for layout-v2 bytes
    (absence of the scheme's ``scale_coding`` key means v1, so old artifacts
    decode unchanged, forever). fp8 has no scale plane on disk — pass the
    per-output-channel ``scales`` tensor (``<name>.weight_scale``)
    explicitly. ``codebook`` (the resolved learned / lattice table) is echoed
    into the fields so reconstruct uses the exact table the packer encoded
    against.
    """
    k = int(k)
    in_f = int(shape[-1])
    if in_f % SUPERBLOCK != 0:
        raise ValueError(
            f"in_features={in_f} must be a multiple of {SUPERBLOCK}")
    rows = int(packed.shape[0])
    n_sb = in_f // SUPERBLOCK
    ts = _serialized_type_size(k, grid, scale_coding)
    if tuple(packed.shape) != (rows, n_sb * ts):
        raise ValueError(
            f"packed shape {tuple(packed.shape)} != expected "
            f"{(rows, n_sb * ts)} for k={k} grid={grid} in_features={in_f} "
            f"scale_coding={scale_coding}")
    block = packed.reshape(rows, n_sb, ts)
    index_bytes = INDEX_BYTES_PER_K * k
    codes = _unpack_bytes_to_codes(block[..., :index_bytes], k)
    nvec = in_f // VEC_DIM

    if mode == "full":
        fields: dict = {"indices": codes.reshape(rows, nvec)}
    elif mode == "signed":
        sign_byte = codes & 0xFF
        mag = codes >> VEC_DIM
        shifts = torch.arange(VEC_DIM, device=codes.device)
        neg = ((sign_byte.unsqueeze(-1) >> shifts) & 1).bool()  # (rows,nvec,8)
        signs = torch.where(neg, -1.0, 1.0).reshape(rows, in_f)
        fields = {"indices": mag.reshape(rows, nvec), "signs": signs}
    elif mode == "product":
        n_sub = family_for(grid, "product").n_sub
        bits = subtable_bit_widths(k, "product", n_sub)
        subs, off = [], 0
        for i in range(n_sub):
            subs.append((codes >> off) & ((1 << bits[i]) - 1))
            off += bits[i]
        fields = {"indices": torch.stack(subs, dim=-1).reshape(
            rows, nvec, n_sub)}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if grid == "fp4" and scale_coding == SCALE_CODING_TWO_TIER:
        super_e, sub_c, composed = _two_tier_scale_unpack(
            block[..., index_bytes:
                  index_bytes + SCALE_PLANE_BYTES[
                      ("fp4", SCALE_CODING_TWO_TIER)
                  ]])
        fields["scales"] = composed
        fields["scale_super"] = super_e.to(torch.uint8)
        fields["scale_sub"] = sub_c
        fields["scale_coding"] = SCALE_CODING_TWO_TIER
    elif grid == "fp4":
        scale_plane_bytes = SCALE_PLANE_BYTES[("fp4", SCALE_CODING_V1)]
        sc = block[..., index_bytes:index_bytes + scale_plane_bytes].reshape(
            rows, n_sb * FP4_SCALE_GROUPS_PER_SUPERBLOCK
        )
        fields["scales"] = sc.contiguous().view(_E4M3).to(torch.float32)
    else:
        if scales is None:
            raise ValueError(
                "fp8 CB has no on-disk scale plane; pass the per-channel "
                "`scales` (<name>.weight_scale) to unpack")
        fields["scales"] = scales
    fields["shape"] = tuple(int(d) for d in shape)
    if codebook is not None:
        fields["codebook"] = codebook
    return fields


# ---------------------------------------------------------------------------
# Pure leaf-local encode/stitch API — single shared implementation for
# derive and streaming export. Side-effect-free, fail-closed, no IO/global.
# ---------------------------------------------------------------------------

_PAYLOAD_IDENTITY_VERSION = "canonical_nvfp4_cb_payload_identity.v4"
_PAYLOAD_IDENTITY_V1 = "canonical_nvfp4_cb_payload_identity.v1"
_PAYLOAD_IDENTITY_V2 = "canonical_nvfp4_cb_payload_identity.v2"
_PAYLOAD_IDENTITY_V3 = "canonical_nvfp4_cb_payload_identity.v3"

# Strict allowlists for leaf-local stitching
_ROW_FIELDS = frozenset({"indices", "signs", "scales", "scale_super", "scale_sub"})
_GLOBAL_FIELDS = frozenset({"codebook", "scale_coding", "shape"})


def _validate_nvfp4_format(k: int, grid: str, mode: str, *, scale_coding: str | None = None) -> str:
    """Validate NVFP4-only k/grid/mode/scale_coding against registry/menu.

    Production-shared core accepts ONLY actual cb_layout NVFP4 product
    (NVFP4_CB_K*) and signed (NVFP4_CB_S*) families. `full` is research-only
    and is explicitly rejected here. Rejects FP8, unknown k, or mismatched
    grid/mode/k. Requires exact types (no bool/str/float coercion) and exact
    values. For this 3-D packed MoE release path scale_coding must be
    two_tier (layout v2) when supplied.
    """
    # Exact type checks — bool is subclass of int, so type() must be int
    if type(k) is not int:
        raise ValueError(f"k must be exact int (bool/float/str rejected), got {type(k).__name__} {k!r}")
    if type(grid) is not str:
        raise ValueError(f"grid must be exact str, got {type(grid).__name__} {grid!r}")
    if type(mode) is not str:
        raise ValueError(f"mode must be exact str, got {type(mode).__name__} {mode!r}")
    if grid != "fp4":
        raise ValueError(f"leaf-local NVFP4-CB only supports grid='fp4', got {grid!r}")
    if mode not in ("product", "signed"):
        raise ValueError(f"mode must be 'product'/'signed' (full rejected in this production-shared core), got {mode!r}")
    from .cb_layout import FAMILY_BY_GRID_MODE, SCALE_CODINGS, NVFP4_PRODUCT_RUNGS, NVFP4_SIGNED_RUNGS

    if scale_coding is not None:
        if type(scale_coding) is not str:
            raise ValueError(f"scale_coding must be exact str, got {type(scale_coding).__name__} {scale_coding!r}")
        if str(scale_coding) not in SCALE_CODINGS:
            raise ValueError(f"scale_coding {scale_coding!r} not in {sorted(SCALE_CODINGS)}")
        # Enforce two_tier for this release path when supplied
        if str(scale_coding) != SCALE_CODING_TWO_TIER:
            raise ValueError(f"leaf-local packed MoE requires scale_coding='two_tier' (layout v2), got {scale_coding!r}")
    # Product / signed via family only — no full pseudo-family
    key = (grid, mode)
    try:
        family = FAMILY_BY_GRID_MODE[key]
    except KeyError as exc:
        raise ValueError(f"unknown CB grid/mode {key!r} for NVFP4 leaf-local") from exc
    if family.grid != "fp4":
        raise ValueError(f"leaf-local only supports NVFP4 (fp4) families, got {family.grid!r}")
    if int(k) not in family.rungs:
        raise ValueError(f"k={k} not in NVFP4 {mode} rung set {family.rungs} (format {family.prefix}{k} invalid)")
    try:
        from . import format_registry as _fr

        name = family.name(int(k))
        spec = _fr.get_format(name)
        if spec is None:
            raise ValueError(f"format {name!r} not registered")
    except Exception as exc:
        raise ValueError(f"NVFP4 format validation failed for k={k} grid={grid} mode={mode}: {exc}") from exc
    return family.name(int(k))


def _validate_packed_leaf_local_inputs(
    weight: torch.Tensor,
    k: int,
    grid: str,
    mode: str,
    member_order: Sequence[str],
    slice_boundaries: Mapping[str, Sequence[int]],
    leaf_col_weights: Mapping[str, torch.Tensor],
    *,
    scale_coding: str | None = None,
    encode_tier: str | None = None,
    codebook: torch.Tensor | tuple | None = None,
    scale_sweep: bool | None = None,
) -> tuple[int, int, int]:
    """Validate fused weight, member_order, slice_boundaries, leaf_col_weights.
    NVFP4-only, strict col-weight finite/nonnegative, strict registry k check.
    Returns (E,R_total,C). Fail-closed on any mismatch.
    Requires exact types for k/grid/mode/scale_coding/encode_tier/scale_sweep/codebook."""
    if not isinstance(weight, torch.Tensor):
        raise ValueError("weight must be Tensor")
    if weight.ndim != 3:
        raise ValueError(f"packed leaf-local weight must be 3-D (E,R_total,C), got {tuple(weight.shape)}")
    E, R_total, C = map(int, weight.shape)
    if E <= 0 or R_total <= 0 or C <= 0:
        raise ValueError(f"weight dimensions must be positive, got {(E,R_total,C)}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight dtype must be torch.bfloat16 (BF16 certified), got {weight.dtype} (production leaf-local certifies BF16 only)")
    # Exact type checks before any coercion
    if type(k) is not int:
        raise ValueError(f"k must be exact int (bool rejected), got {type(k).__name__} {k!r}")
    if type(grid) is not str:
        raise ValueError(f"grid must be exact str, got {type(grid).__name__} {grid!r}")
    if type(mode) is not str:
        raise ValueError(f"mode must be exact str, got {type(mode).__name__} {mode!r}")
    if scale_sweep is not None and type(scale_sweep) is not bool:
        raise ValueError(f"scale_sweep must be exact bool, got {type(scale_sweep).__name__} {scale_sweep!r}")
    if scale_coding is not None and type(scale_coding) is not str:
        raise ValueError(f"scale_coding must be exact str, got {type(scale_coding).__name__} {scale_coding!r}")
    if encode_tier is not None and type(encode_tier) is not str:
        raise ValueError(f"encode_tier must be exact str, got {type(encode_tier).__name__} {encode_tier!r}")
    # Require explicit resolved codebook (no default resolution)
    if codebook is None:
        raise ValueError("codebook must be explicit resolved Tensor/tuple (None rejected — no default resolution in production-shared helper)")
    # Validate codebook shapes strictly against family/rung
    # This also ensures grid/mode/k are NVFP4 product/signed only
    _validate_nvfp4_format(k, grid, mode, scale_coding=scale_coding)
    # Enforce two_tier for this packed MoE release path
    if scale_coding is not None and scale_coding != SCALE_CODING_TWO_TIER:
        raise ValueError(f"scale_coding must be 'two_tier' (layout v2) for packed MoE, got {scale_coding!r}")
    # Extra render-param validation (explicit, no silent defaults, no normalization)
    if scale_sweep is not None and type(scale_sweep) is not bool:
        raise ValueError(f"scale_sweep must be bool, got {type(scale_sweep).__name__} {scale_sweep!r}")
    if encode_tier is not None:
        if encode_tier not in _ENCODE_TIERS:
            raise ValueError(f"encode_tier {encode_tier!r} not in {_ENCODE_TIERS} (must be exact, whitespace/case rejected)")
    # Strict codebook shape/content validation
    # Use family to get expected shapes and compare
    from .cb_layout import codebook_subtable_shapes as _cb_shapes, subtable_bit_widths as _cb_bits, family_for as _fam
    try:
        fam = _fam(grid, mode)
        exp_shapes = _cb_shapes(k, mode, fam.n_sub)
        if mode == "product":
            if type(codebook) is not tuple:
                raise ValueError(f"product mode codebook must be exact tuple (list/subclass rejected), got {type(codebook).__name__}")
            for idx, t in enumerate(codebook):  # type: ignore[union-attr]
                if not isinstance(t, torch.Tensor):
                    raise ValueError(f"codebook[{idx}] must be Tensor, got {type(t).__name__}")
            actual_shapes = tuple(tuple(int(d) for d in t.shape) for t in codebook)  # type: ignore[union-attr]
            if actual_shapes != exp_shapes:
                raise ValueError(f"codebook shapes {actual_shapes} != expected {exp_shapes} for k={k} mode={mode}")
            # Validate each table dtype is float and not empty
            for idx, t in enumerate(codebook):  # type: ignore[union-attr]
                if t.ndim != 2 or t.shape[1] != (8 // fam.n_sub):
                    raise ValueError(f"codebook[{idx}] shape {tuple(t.shape)} invalid for sub_dim {8 // fam.n_sub}")
        elif mode == "signed":
            if not isinstance(codebook, torch.Tensor):
                raise ValueError(f"signed mode codebook must be Tensor, got {type(codebook).__name__}")
            actual_shape = tuple(int(d) for d in codebook.shape)
            if actual_shape != exp_shapes[0]:
                raise ValueError(f"codebook shape {actual_shape} != expected {exp_shapes[0]} for k={k} mode={mode}")
            if bool((codebook < 0).any()):
                raise ValueError("signed-mode magnitude codebook must be non-negative")
        else:
            raise ValueError(f"unsupported mode {mode!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"codebook validation failed for k={k} grid={grid} mode={mode}: {exc}") from exc
    # Enforce fp4 only
    if grid != "fp4":
        raise ValueError(f"leaf-local NVFP4-CB requires grid='fp4', got {grid!r}")
    if not isinstance(member_order, (list, tuple)) or not member_order:
        raise ValueError(f"member_order must be non-empty sequence, got {member_order!r}")
    for _leaf in member_order:
        if type(_leaf) is not str:
            raise ValueError(f"member_order element {_leaf!r} must be exact str, got {type(_leaf).__name__}")
        if _leaf == "":
            raise ValueError(f"member_order element must be nonempty str, got {_leaf!r}")
    if len(set(member_order)) != len(member_order):
        raise ValueError(f"member_order must have unique leaves, got {member_order!r}")
    if tuple(member_order) not in (("gate_proj", "up_proj"), ("down_proj",)):
        raise ValueError(f"member_order {tuple(member_order)!r} must be exact routed-expert topology ('gate_proj','up_proj') for fused gate_up or ('down_proj',) for down parent")
    if not isinstance(slice_boundaries, Mapping):
        raise ValueError(f"slice_boundaries must be mapping, got {type(slice_boundaries).__name__}")
    if set(slice_boundaries.keys()) != set(member_order):
        raise ValueError(
            f"slice_boundaries keys {sorted(slice_boundaries.keys())} != member_order {sorted(member_order)}"
        )
    if not isinstance(leaf_col_weights, Mapping):
        raise ValueError(f"leaf_col_weights must be mapping, got {type(leaf_col_weights).__name__}")
    if set(leaf_col_weights.keys()) != set(member_order):
        raise ValueError(
            f"leaf_col_weights keys {sorted(leaf_col_weights.keys())} != member_order {sorted(member_order)}"
        )
    # Validate slices are contiguous, non-overlapping, covering [0,R_total) in member_order order, non-empty
    off = 0
    for leaf in member_order:
        bounds = slice_boundaries[leaf]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"slice_boundaries[{leaf!r}] must be (start,end), got {bounds!r}")
        if type(bounds[0]) is not int or type(bounds[1]) is not int:
            raise ValueError(f"slice_boundaries[{leaf!r}] must be (int, int) exact, got {bounds!r} types {[type(x).__name__ for x in bounds]}")
        s, e = bounds[0], bounds[1]
        if s != off:
            raise ValueError(
                f"slice_boundaries[{leaf!r}] start {s} != expected {off} (gap/overlap/out-of-order)"
            )
        if e <= s:
            raise ValueError(f"slice_boundaries[{leaf!r}] empty or inverted {(s,e)}")
        if e > R_total:
            raise ValueError(f"slice_boundaries[{leaf!r}] end {e} > R_total {R_total}")
        R_leaf = e - s
        if R_leaf <= 0:
            raise ValueError(f"leaf {leaf!r} has non-positive rows {R_leaf}")
        off = e
        # Validate leaf col weights: exact (E,1,C), finite, nonnegative
        cw = leaf_col_weights[leaf]
        if not isinstance(cw, torch.Tensor):
            raise ValueError(f"leaf_col_weights[{leaf!r}] must be Tensor, got {type(cw).__name__}")
        if tuple(cw.shape) != (E, 1, C):
            raise ValueError(
                f"leaf_col_weights[{leaf!r}] shape {tuple(cw.shape)} != {(E,1,C)}"
            )
        if cw.ndim != 3:
            raise ValueError(f"leaf_col_weights[{leaf!r}] must be 3-D, got ndim={cw.ndim}")
        if not torch.isfinite(cw).all():
            raise ValueError(f"leaf_col_weights[{leaf!r}] must be finite")
        if bool((cw < 0).any()):
            raise ValueError(f"leaf_col_weights[{leaf!r}] must be nonnegative")
    if off != R_total:
        raise ValueError(f"slice_boundaries cover {off} rows != R_total {R_total} (gapped)")
    return E, R_total, C


def _validate_prepared_for_packed(
    prepared: "PreparedLDLQEvidence",
    E: int,
    C: int,
) -> None:
    """Fail-closed validation of PreparedLDLQEvidence against shape."""
    if not isinstance(prepared, PreparedLDLQEvidence):
        raise ValueError(f"prepared must be PreparedLDLQEvidence, got {type(prepared).__name__}")
    if prepared.policy != _SPLIT_POLICY or prepared.version != _SPLIT_VERSION:
        raise ValueError(
            f"prepared policy/version {prepared.policy}/{prepared.version} != expected {_SPLIT_POLICY}/{_SPLIT_VERSION}"
        )
    if len(prepared.original_rows) != E:
        raise ValueError(f"prepared expert count {len(prepared.original_rows)} != E {E}")
    if len(prepared.fit_rows) != E or len(prepared.holdout_rows) != E:
        raise ValueError("prepared fit/holdout length mismatch")
    if len(prepared.original_digests) != E or len(prepared.fit_digests) != E or len(prepared.holdout_digests) != E:
        raise ValueError("prepared digest length mismatch")
    # Check per-expert digests still match (detect mutation)
    for i in range(E):
        if _partition_multiset_digest(torch.as_tensor(prepared.original_rows[i])) != prepared.original_digests[i]:
            raise ValueError(f"prepared original_rows mutated at expert {i}")
        if _partition_multiset_digest(torch.as_tensor(prepared.fit_rows[i])) != prepared.fit_digests[i]:
            raise ValueError(f"prepared fit_rows mutated at expert {i}")
        if _partition_multiset_digest(torch.as_tensor(prepared.holdout_rows[i])) != prepared.holdout_digests[i]:
            raise ValueError(f"prepared holdout_rows mutated at expert {i}")
        # Width check for non-empty
        orig = torch.as_tensor(prepared.original_rows[i])
        if orig.numel() > 0 and orig.ndim == 2 and int(orig.shape[1]) != C:
            raise ValueError(f"prepared width mismatch at expert {i}: {orig.shape[1]} != {C}")
    # Overall split_infos digest
    if _canonical_split_infos_digest(prepared.split_infos) != prepared.split_infos_digest:
        raise ValueError("prepared split_infos digest mismatch (nested mutation)")


def encode_packed_parent_leaf_local(
    weight: torch.Tensor,
    k: int,
    *,
    grid: str,
    mode: str,
    codebook: torch.Tensor | tuple,
    scale_sweep: bool,
    scale_coding: str,
    encode_tier: str,
    member_order: Sequence[str],
    slice_boundaries: Mapping[str, Sequence[int]],
    leaf_col_weights: Mapping[str, torch.Tensor],
    prepared: "ValidatedPackedLDLQEvidence",
    warm_state: None = None,
    block_size: int = LDLQ_BLOCK_SIZE,
    damping_fraction: float = LDLQ_DAMPING_FRACTION,
) -> tuple[dict, dict]:
    """Pure leaf-local NVFP4-CB encode/stitch — single shared side-effect-free implementation.

    NVFP4 only: rejects FP8/grid mismatch and validates k/format against the
    actual NVFP4 registry/menu (not merely k>0). All byte-affecting render
    inputs (exact codebook, scale_sweep, scale_coding, encode_tier, mode) are
    required explicitly — no silent default into a different layout. Warm
    state is rejected (cold/no-warm only) — fail closed.

    Validates exact topology (unique member_order, contiguous ordered full
    coverage of (E,R_total,C), exact leaf map keys, exact (E,1,C) finite
    nonnegative leaf column weights). Processes one leaf at a time — at peak
    retains only that leaf's raw/candidate/final temporaries plus the
    preallocated final parent fields and small metric telemetry. Reuses the
    same validated prepared FIT/HOLDOUT objects across all leaves and adjacent
    rungs so the factor cache hits (deep validation once per prepared parent;
    subsequent cheap invariant checks still reject mutation/topology mismatches).

    Gates independently per (leaf,expert) on leaf-local held-out
    activation-output MSE, with cold/insufficient experts raw. Zero LDLQ error
    is valid; invalid/nonfinite evidence fails to raw for that arm. Preserves
    exact raw bytes for rejected arms. Stitches every and only known
    row-bearing fields in expert-major order (indices, signs, scales,
    scale_super, scale_sub) and strict global allowlist (codebook,
    scale_coding, shape) — rejects unknown fields or global mismatch.
    Output shape is exactly (E,R_total,C). Preserves exact single-leaf byte
    behavior for same explicit render inputs. Returns plain serializable
    complete gate telemetry with exact per-leaf masks and raw/LDLQ values;
    does not label diagnostic mean as authoritative fused cost.
    """
    # Fail closed on any warm leaf-local state (composable warm not provably safe)
    if warm_state is not None:
        raise ValueError(
            f"leaf-local warm_state not supported (got {warm_state!r}); "
            "production re-encodes cold — pass warm_state=None"
        )
    # Exact type checks for production-shared API (no coercion)
    if type(k) is not int:
        raise ValueError(f"k must be exact int (bool rejected), got {type(k).__name__} {k!r}")
    if type(grid) is not str:
        raise ValueError(f"grid must be exact str, got {type(grid).__name__} {grid!r}")
    if type(mode) is not str:
        raise ValueError(f"mode must be exact str, got {type(mode).__name__} {mode!r}")
    if type(scale_sweep) is not bool:
        raise ValueError(f"scale_sweep must be exact bool, got {type(scale_sweep).__name__} {scale_sweep!r}")
    if type(scale_coding) is not str:
        raise ValueError(f"scale_coding must be exact str, got {type(scale_coding).__name__} {scale_coding!r}")
    if scale_coding != SCALE_CODING_TWO_TIER:
        raise ValueError(f"scale_coding must be 'two_tier' for packed MoE (layout v2), got {scale_coding!r}")
    if type(encode_tier) is not str:
        raise ValueError(f"encode_tier must be exact str, got {type(encode_tier).__name__} {encode_tier!r}")
    if encode_tier not in _ENCODE_TIERS:
        raise ValueError(f"encode_tier {encode_tier!r} not in {_ENCODE_TIERS} (must be exact, whitespace/case rejected)")
    if codebook is None:
        raise ValueError("codebook must be explicit resolved Tensor/tuple (None rejected)")
    # NVFP4-only strict topology and format validation (k/grid/mode/scale_coding)
    E, R_total, C = _validate_packed_leaf_local_inputs(
        weight, k, grid, mode, member_order, slice_boundaries, leaf_col_weights,
        scale_coding=scale_coding, encode_tier=encode_tier, codebook=codebook, scale_sweep=scale_sweep,
    )
    _validate_nvfp4_format(k, grid, mode, scale_coding=scale_coding)
    # Validate prepared — must be ValidatedPackedLDLQEvidence, cheap fingerprint only
    if not isinstance(prepared, ValidatedPackedLDLQEvidence):
        raise ValueError(f"prepared must be ValidatedPackedLDLQEvidence (got {type(prepared).__name__}); raw PreparedLDLQEvidence rejected — construct ValidatedPackedLDLQEvidence(prepared, E, C) first")
    _validate_validated_evidence(prepared, E, C)
    # Use private cloned tensors to preserve factor-cache reuse; defensive copy would miss cache
    _priv = object.__getattribute__(prepared, "_priv_prepared")
    holdout_rows: tuple[torch.Tensor, ...] = tuple(_priv.holdout_rows)
    fit_rows: tuple[torch.Tensor, ...] = tuple(_priv.fit_rows)
    # Ineligible union for internal raw-fallback (defect 20: preserve union only internally, telemetry separates)
    _actual_cold_set = set(_priv.cold_experts)
    _actual_insufficient_set = set(_priv.insufficient_experts)
    ineligible_set = _actual_cold_set | _actual_insufficient_set
    cold_set = ineligible_set  # internal alias used for raw-fallback decision; telemetry below separates
    eligible = tuple(_priv.eligible_experts)
    # Resolve format spec for canonical held-out metric (NVFP4 only)
    from prismaquant import format_registry as _fr_local
    # Use NVFP4 format name derived via strict validation
    _format_name = _validate_nvfp4_format(int(k), str(grid), str(mode), scale_coding=scale_coding)
    _spec = _fr_local.get_format(_format_name)
    if _spec is None:
        raise ValueError(f"validated NVFP4 format {_format_name!r} not in registry")
    # Small metric telemetry (per leaf per expert)
    raw_leaf_mse: dict[str, list[float]] = {leaf: [float("inf")] * E for leaf in member_order}
    ldlq_leaf_mse: dict[str, list[float]] = {leaf: [float("inf")] * E for leaf in member_order}
    per_leaf_kept: dict[str, list[bool]] = {leaf: [False] * E for leaf in member_order}
    weight_dtype = weight.dtype
    # Cross-leaf codebook authority snapshot (defect 12): D2H-free fingerprint before leaf processing
    def _cb_fingerprint_for_snapshot(cb: Any):
        def _fp(t: torch.Tensor):
            try:
                dp = t.data_ptr()
            except Exception:
                dp = None
            try:
                so = t.storage_offset()  # type: ignore[attr-defined]
            except Exception:
                so = 0
            shape = tuple(int(d) for d in t.shape)
            try:
                stride = tuple(int(s) for s in t.stride())
            except Exception:
                stride = ()
            dtype_s = str(t.dtype)
            device_s = str(t.device)
            ver = getattr(t, "_version", None)
            return (dp, so, shape, stride, dtype_s, device_s, ver)
        if isinstance(cb, tuple):
            return tuple(_fp(tt) for tt in cb)
        elif isinstance(cb, torch.Tensor):
            return (_fp(cb),)
        else:
            # Should have been validated earlier; capture as is
            return None
    _codebook_snapshot_fps = _cb_fingerprint_for_snapshot(codebook)
    if _codebook_snapshot_fps is None:
        raise ValueError(f"codebook snapshot failed: unexpected codebook type {type(codebook).__name__}")
    # Preallocate parent fields on first leaf (peek trail shapes), then stitch one leaf at a time
    parent_fields: dict[str, Any] = {}
    global_codebook: Any = None
    global_scale_coding: str | None = None
    first_row_fields: set[str] | None = None
    # --- One-leaf-at-a-time isolation: nested function guarantees scope release before next leaf ---
    def _one_leaf_work(leaf: str, leaf_idx: int, s: int, e: int):
        """Isolated per-leaf encode/gate/stitch — all temporaries are local to this call frame."""
        nonlocal global_codebook, global_scale_coding, first_row_fields, parent_fields
        R_leaf = int(e - s)
        weight_leaf = weight[:, s:e, :].contiguous()
        cw_leaf = leaf_col_weights[leaf]
        raw_leaf = nvfp4_cb_fields(
            weight_leaf, k, grid=grid, mode=mode, col_weights=cw_leaf,
            codebook=codebook, scale_sweep=scale_sweep, scale_coding=scale_coding, encode_tier=encode_tier,
        )
        _allowed = _ROW_FIELDS | _GLOBAL_FIELDS
        for kk in raw_leaf:
            if kk not in _allowed:
                raise ValueError(f"leaf {leaf!r} raw fields contain unknown key {kk!r} not in allowlist {sorted(_allowed)}")
        cur_cb = raw_leaf.get("codebook")
        cur_sc = raw_leaf.get("scale_coding", SCALE_CODING_V1)
        # D2H-free fingerprint compare against immutable snapshot (defect 12)
        def _cur_fps(cb: Any):
            def _fp(t: torch.Tensor):
                try:
                    dp = t.data_ptr()
                except Exception:
                    dp = None
                try:
                    so = t.storage_offset()  # type: ignore[attr-defined]
                except Exception:
                    so = 0
                shape = tuple(int(d) for d in t.shape)
                try:
                    stride = tuple(int(s) for s in t.stride())
                except Exception:
                    stride = ()
                dtype_s = str(t.dtype)
                device_s = str(t.device)
                ver = getattr(t, "_version", None)
                return (dp, so, shape, stride, dtype_s, device_s, ver)
            if isinstance(cb, tuple):
                return tuple(_fp(tt) for tt in cb)
            elif isinstance(cb, torch.Tensor):
                return (_fp(cb),)
            else:
                return None
        _cur_fps_val = _cur_fps(cur_cb)
        if _cur_fps_val != _codebook_snapshot_fps:
            raise ValueError(f"global codebook fingerprint mismatch at leaf {leaf!r} (data_ptr/storage_offset/shape/stride/dtype/device/_version differ from snapshot; same-pointer different-stride/view or in-place version increment rejected)")
        if leaf_idx == 0:
            global_codebook = cur_cb
            global_scale_coding = cur_sc
            first_row_fields = {kk for kk in raw_leaf if kk in _ROW_FIELDS}
            if "indices" not in raw_leaf:
                raise ValueError("first leaf missing required row field 'indices'")
            for tkey in first_row_fields:
                tval = raw_leaf[tkey]
                if not isinstance(tval, torch.Tensor):
                    raise ValueError(f"row field {tkey!r} not Tensor in leaf {leaf!r}")
                trail = tuple(tval.shape[1:])
                parent_shape = (E * R_total,) + trail
                parent_fields[tkey] = torch.empty(parent_shape, dtype=tval.dtype, device=tval.device)
            if global_codebook is not None:
                parent_fields["codebook"] = global_codebook
            if global_scale_coding is not None and global_scale_coding != SCALE_CODING_V1:
                parent_fields["scale_coding"] = global_scale_coding
        else:
            if global_scale_coding != cur_sc:
                raise ValueError(f"global scale_coding {global_scale_coding!r} != {cur_sc!r} at leaf {leaf!r}")
            cur_row_fields = {kk for kk in raw_leaf if kk in _ROW_FIELDS}
            if cur_row_fields != first_row_fields:
                raise ValueError(f"row field set mismatch leaf {leaf!r} {sorted(cur_row_fields)} != first {sorted(first_row_fields)}")
            for tkey in cur_row_fields:
                tval = raw_leaf[tkey]
                ref = parent_fields[tkey]
                if tuple(tval.shape[1:]) != tuple(ref.shape[1:]):
                    raise ValueError(f"field {tkey!r} trail shape mismatch leaf {leaf!r} {tuple(tval.shape[1:])} != {tuple(ref.shape[1:])}")
                if tval.dtype != ref.dtype:
                    raise ValueError(f"field {tkey!r} dtype mismatch leaf {leaf!r} {tval.dtype} != {ref.dtype}")
                exp_rows = E * R_leaf
                if int(tval.shape[0]) != exp_rows:
                    raise ValueError(f"field {tkey!r} leaf {leaf!r} rows {int(tval.shape[0])} != E*R_leaf {exp_rows} (non-row-separable)")
        if leaf_idx == 0:
            for tkey in first_row_fields:  # type: ignore[union-attr]
                tval = raw_leaf[tkey]
                exp_rows = E * R_leaf
                if int(tval.shape[0]) != exp_rows:
                    raise ValueError(f"field {tkey!r} leaf {leaf!r} rows {int(tval.shape[0])} != E*R_leaf {exp_rows}")
        if not eligible:
            ldlq_leaf = raw_leaf
        elif len(eligible) == E:
            fit_all = tuple(fit_rows[i] for i in eligible)
            ldlq_leaf = ldlq_reassign_cb_fields(
                weight_leaf, raw_leaf, cw_leaf, fit_all,
                grid=grid, mode=mode, block_size=block_size, damping_fraction=damping_fraction,
            )
            for kk in ldlq_leaf:
                if kk not in _allowed:
                    raise ValueError(f"ldlq leaf {leaf!r} unknown key {kk!r}")
        else:
            _expert_batch = _resolve_ldlq_expert_batch()
            cand = dict(raw_leaf)
            for _k in ("indices", "signs"):
                if _k in raw_leaf and isinstance(raw_leaf[_k], torch.Tensor):
                    cand[_k] = raw_leaf[_k].clone()
            _elig_list = list(eligible)
            cw_tensor = cw_leaf
            for _cs in range(0, len(_elig_list), _expert_batch):
                chunk = _elig_list[_cs:_cs + _expert_batch]
                is_contig = chunk == list(range(chunk[0], chunk[0] + len(chunk)))
                if is_contig:
                    c_start = chunk[0]
                    c_end = c_start + len(chunk)
                    weight_chunk = weight_leaf[c_start:c_end]
                    cw_chunk = cw_tensor[c_start:c_end] if isinstance(cw_tensor, torch.Tensor) and cw_tensor.ndim == 3 and int(cw_tensor.shape[0]) == E else cw_tensor
                    fields_chunk: dict = {}
                    for fk, fv in raw_leaf.items():
                        if fk in _ROW_FIELDS and isinstance(fv, torch.Tensor):
                            fields_chunk[fk] = fv[c_start * R_leaf:c_end * R_leaf]
                        else:
                            fields_chunk[fk] = fv
                    fields_chunk["shape"] = (len(chunk), R_leaf, C)
                else:
                    idx = torch.tensor(chunk, device=weight_leaf.device)
                    weight_chunk = weight_leaf[idx]
                    cw_chunk = cw_tensor[idx] if isinstance(cw_tensor, torch.Tensor) and int(cw_tensor.shape[0]) == E else cw_leaf
                    row_idx_parts = []
                    for ex in chunk:
                        row_idx_parts.append(torch.arange(ex * R_leaf, (ex + 1) * R_leaf, device=raw_leaf["indices"].device))
                    row_idx_chunk = torch.cat(row_idx_parts) if row_idx_parts else torch.empty((0,), dtype=torch.long, device=raw_leaf["indices"].device)
                    fields_chunk = {}
                    for fk, fv in raw_leaf.items():
                        if fk in _ROW_FIELDS and isinstance(fv, torch.Tensor):
                            fields_chunk[fk] = fv[row_idx_chunk]
                        else:
                            fields_chunk[fk] = fv
                    fields_chunk["shape"] = (len(chunk), R_leaf, C)
                fit_chunk = tuple(fit_rows[i] for i in chunk)
                ldlq_fields_chunk = ldlq_reassign_cb_fields(
                    weight_chunk, fields_chunk, cw_chunk, fit_chunk,
                    grid=grid, mode=mode, block_size=block_size, damping_fraction=damping_fraction,
                )
                for _idx_in_chunk, _elig in enumerate(chunk):
                    src_start = _idx_in_chunk * R_leaf
                    src_end = src_start + R_leaf
                    dst_start = _elig * R_leaf
                    dst_end = dst_start + R_leaf
                    for _k in ("indices", "signs"):
                        if _k in ldlq_fields_chunk and isinstance(ldlq_fields_chunk[_k], torch.Tensor) and _k in cand:
                            cand[_k][dst_start:dst_end] = ldlq_fields_chunk[_k][src_start:src_end]
                del weight_chunk, fields_chunk, ldlq_fields_chunk
            ldlq_leaf = cand
            for kk in ldlq_leaf:
                if kk not in _allowed:
                    raise ValueError(f"ldlq leaf {leaf!r} unknown key {kk!r}")
        cur_raw_mse = [float("inf")] * E
        cur_ldlq_mse = [float("inf")] * E
        def _compute_mse_for_fields(fields_dict: dict, out_list: list[float]) -> None:
            for (first, last), chunk_recon, _ in iter_nvfp4_cb_recon_chunks(fields_dict, k, grid=grid, mode=mode):
                chunk_bf16 = chunk_recon.to(weight_dtype)
                del chunk_recon
                for local in range(last - first):
                    gidx = first + local
                    w_leaf_bf16 = weight[gidx, s:e, :].contiguous().to(weight_dtype)
                    r_leaf_bf16 = chunk_bf16[local]
                    act = holdout_rows[gidx]
                    if act is None or (isinstance(act, torch.Tensor) and (act.numel() == 0 or int(act.shape[0]) == 0)):
                        out_list[gidx] = float("inf")
                    else:
                        act_f = torch.as_tensor(act).to(w_leaf_bf16.device, torch.float32)
                        if not torch.isfinite(act_f).all():
                            out_list[gidx] = float("inf")
                        else:
                            try:
                                mse = canonical_nvfp4_cb_single_output_mse(w_leaf_bf16, r_leaf_bf16, act_f, _spec)
                                if mse != mse or mse == float("inf") or mse == float("-inf") or not (abs(mse) < float("inf")):
                                    out_list[gidx] = float("inf")
                                else:
                                    out_list[gidx] = float(mse)
                            except ValueError:
                                out_list[gidx] = float("inf")
                del chunk_bf16, _
        _compute_mse_for_fields(raw_leaf, cur_raw_mse)
        if ldlq_leaf is raw_leaf:
            cur_ldlq_mse = list(cur_raw_mse)
        else:
            _compute_mse_for_fields(ldlq_leaf, cur_ldlq_mse)
        cur_kept = [False] * E
        for e_idx in range(E):
            if e_idx in cold_set:
                cur_kept[e_idx] = False
            else:
                raw_err = cur_raw_mse[e_idx]
                ldlq_err = cur_ldlq_mse[e_idx]
                if raw_err != raw_err or ldlq_err != ldlq_err or raw_err == float("inf") or ldlq_err == float("inf") or raw_err == float("-inf") or ldlq_err == float("-inf"):
                    cur_kept[e_idx] = False
                else:
                    cur_kept[e_idx] = bool(ldlq_err <= raw_err + LDLQ_GATE_EPSILON * max(abs(raw_err), abs(ldlq_err), 1.0))
        # Stitch into parent (still inside isolated frame but touches preallocated parent)
        for tkey in first_row_fields:  # type: ignore[union-attr]
            parent_t = parent_fields[tkey]
            leaf_offset_in_parent = sum(int(slice_boundaries[ml][1] - slice_boundaries[ml][0]) for ml in member_order[:leaf_idx])
            for e_idx in range(E):
                dst_off = e_idx * R_total + leaf_offset_in_parent
                dst_start = dst_off
                dst_end = dst_start + R_leaf
                src_start = e_idx * R_leaf
                src_end = src_start + R_leaf
                src_tensor = ldlq_leaf[tkey] if cur_kept[e_idx] else raw_leaf[tkey]
                parent_t[dst_start:dst_end] = src_tensor[src_start:src_end]
        # Return telemetry; locals (raw_leaf, ldlq_leaf, weight_leaf, etc.) go out of scope on return
        return cur_raw_mse, cur_ldlq_mse, cur_kept

    for leaf_idx, leaf in enumerate(member_order):
        s, e = int(slice_boundaries[leaf][0]), int(slice_boundaries[leaf][1])
        cur_raw_mse, cur_ldlq_mse, cur_kept = _one_leaf_work(leaf, leaf_idx, s, e)
        raw_leaf_mse[leaf] = cur_raw_mse
        ldlq_leaf_mse[leaf] = cur_ldlq_mse
        per_leaf_kept[leaf] = cur_kept
        # Explicitly release any lingering references before next leaf (scope already freed, but ensure GC)
        # No raw_leaf/ldlq_leaf/weight_leaf locals remain here
    # Re-check original codebook fingerprint before return (detect in-place version increment between leaves)
    if _cb_fingerprint_for_snapshot(codebook) != _codebook_snapshot_fps:
        raise ValueError("global codebook mutated between leaves (version/stride/storage_offset changed after snapshot; in-place mutation rejected)")
    # Finalize shape and validate coverage
    R_total_check = sum(int(slice_boundaries[leaf][1] - slice_boundaries[leaf][0]) for leaf in member_order)
    if R_total_check != R_total:
        raise ValueError(f"R_total mismatch {R_total_check} != {R_total}")
    parent_fields["shape"] = (E, R_total, C)
    # Gate telemetry (plain serializable, no diagnostic mean labeled as authoritative)
    has_any_kept = any(any(v) for v in per_leaf_kept.values())
    all_eligible_arms = sum(1 for leaf in member_order for e in eligible)
    kept_arms = sum(1 for leaf in member_order for e in eligible if per_leaf_kept[leaf][e])
    all_kept = (kept_arms == all_eligible_arms) and (len(ineligible_set) == 0) and bool(eligible)
    if not has_any_kept:
        gate_label = "raw_fallback_missing_activation" if not eligible else "raw_kept_all"
    else:
        gate_label = "ldlq_kept_all" if all_kept else "mixed_per_expert"
    per_expert_kept = [all(per_leaf_kept[leaf][e] for leaf in member_order) if e not in ineligible_set else False for e in range(E)]
    scalar_kept = gate_label == "ldlq_kept_all"
    # Diagnostic mean explicitly labeled as diagnostic, not authoritative
    raw_diag = [float(sum(raw_leaf_mse[leaf][e] for leaf in member_order) / len(member_order)) if all(v != float("inf") for v in [raw_leaf_mse[leaf][e] for leaf in member_order]) else float("inf") for e in range(E)]
    ldlq_diag = [float(sum(ldlq_leaf_mse[leaf][e] for leaf in member_order) / len(member_order)) if all(v != float("inf") for v in [ldlq_leaf_mse[leaf][e] for leaf in member_order]) else float("inf") for e in range(E)]
    gate_info: dict = {
        "gate": gate_label,
        "kept_ldlq": per_expert_kept if gate_label == "mixed_per_expert" else scalar_kept,
        "per_expert_kept": per_expert_kept,
        "per_leaf_kept": {k: list(map(bool, v)) for k, v in per_leaf_kept.items()},
        "raw_mse_per_expert_per_leaf": {k: list(map(float, v)) for k, v in raw_leaf_mse.items()},
        "ldlq_mse_per_expert_per_leaf": {k: list(map(float, v)) for k, v in ldlq_leaf_mse.items()},
        "raw_mse_per_expert_diagnostic_mean": list(raw_diag),
        "ldlq_mse_per_expert_diagnostic_mean": list(ldlq_diag),
        "metric": "activation_output_mse",
        "split_policy": _priv.policy,
        "split_version": _priv.version,
        "expert_count": int(E),
        "member_order": list(member_order),
        "slice_boundaries": {k: list(v) for k, v in slice_boundaries.items()},
    }
    # Telemetry must separate cold vs insufficient (defect 20); missing_experts is cold only
    if _actual_cold_set or _actual_insufficient_set:
        gate_info["cold_experts"] = sorted(_actual_cold_set)
        gate_info["insufficient_experts"] = sorted(_actual_insufficient_set)
        gate_info["ineligible_experts"] = sorted(ineligible_set)
        if _actual_cold_set:
            gate_info["missing_experts"] = sorted(_actual_cold_set)
    # Do not materialize full reconstruction; no empty_cache in hot loop
    return parent_fields, gate_info


# ---------------------------------------------------------------------------
# Pure canonical payload identity — fail-closed, full byte hash with ABI framing
# ABI v2: NVFP4 only, strict allowlists, binds physical target, expert_ids,
# qnames_per_leaf, evidence/split digests, per_leaf_kept bool strict.
# ---------------------------------------------------------------------------

def _qname_has_expert_token(qname: str, expert_id: int) -> bool:
    """Strict component check: qname must contain exactly one '.experts.<id>.' component, no substring fallback."""
    parts = qname.split(".")
    if parts.count("experts") != 1:
        return False
    for i, p in enumerate(parts):
        if p == "experts" and i + 1 < len(parts) and parts[i + 1] == str(expert_id):
            if i + 2 < len(parts):
                return True
    return False


def _validate_fields_physically(
    fields: Mapping[str, Any],
    k: int,
    grid: str,
    mode: str,
    scale_coding: str,
    E: int,
    R_total: int,
    C: int,
) -> None:
    """Validate fields physically match declared rung before hashing.

    Strict, no fail-open pass branches: exact field set, tensor types, exact dtype,
    exact rank/shape, device consistency, contiguity, finiteness, signs exactly +/-1,
    index/code range, scale-sub range, and exact resolved-codebook family/rung schema.
    Derived from codec/decoder authorities.
    """
    from .cb_layout import codebook_subtable_shapes as _cb_shapes, family_for as _fam, subtable_bit_widths as _cb_bits
    fam = _fam(grid, mode)
    exp_shapes = _cb_shapes(k, mode, fam.n_sub)
    bits = _cb_bits(k, mode, fam.n_sub)
    # Exact field set: row fields per mode + globals, no extra/missing
    if mode == "signed":
        expected_row = {"indices", "signs", "scales", "scale_super", "scale_sub"}
    else:
        expected_row = {"indices", "scales", "scale_super", "scale_sub"}
    expected_global = {"codebook", "scale_coding", "shape"}
    expected_keys = expected_row | expected_global
    actual_keys = set(fields.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"fields keys mismatch missing {missing} extra {extra} (expected exactly {sorted(expected_keys)})")
    # Codebook exact dtype float32, exact rank/shape, device-independent, finiteness, nonnegative for signed
    cb = fields.get("codebook")
    if cb is None:
        raise ValueError("fields missing codebook (NVFP4-CB requires resolved codebook for attestation)")
    if mode == "product":
        if type(cb) is not tuple:
            raise ValueError(f"product mode codebook must be exact tuple (list/subclass rejected), got {type(cb).__name__}")
        if len(cb) != fam.n_sub:
            raise ValueError(f"product codebook n_sub {len(cb)} != expected {fam.n_sub}")
        for idx, t in enumerate(cb):  # type: ignore[union-attr]
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"codebook[{idx}] must be Tensor, got {type(t).__name__} (non-tensor rejected before shape access)")
        actual_shapes = tuple(tuple(int(d) for d in t.shape) for t in cb)  # type: ignore[union-attr]
        if actual_shapes != exp_shapes:
            raise ValueError(f"codebook shapes {actual_shapes} != expected {exp_shapes} for k={k} mode={mode} (mislabeled rung)")
        for idx, t in enumerate(cb):  # type: ignore[union-attr]
            if t.dtype != torch.float32:
                raise ValueError(f"codebook[{idx}] dtype {t.dtype} != torch.float32 (exact)")
            if t.ndim != 2:
                raise ValueError(f"codebook[{idx}] ndim {t.ndim} != 2")
            if tuple(int(d) for d in t.shape) != exp_shapes[idx]:
                raise ValueError(f"codebook[{idx}] shape {tuple(t.shape)} != expected {exp_shapes[idx]}")
            if not t.is_contiguous():
                raise ValueError(f"codebook[{idx}] must be contiguous")
            if not torch.isfinite(t).all():
                raise ValueError(f"codebook[{idx}] must be finite (NaN/Inf detected)")
    elif mode == "signed":
        if not isinstance(cb, torch.Tensor):
            raise ValueError(f"signed mode codebook must be Tensor, got {type(cb).__name__}")
        if cb.dtype != torch.float32:
            raise ValueError(f"signed codebook dtype {cb.dtype} != torch.float32")
        if cb.ndim != 2:
            raise ValueError(f"signed codebook ndim {cb.ndim} != 2")
        actual_shape = tuple(int(d) for d in cb.shape)
        if actual_shape != exp_shapes[0]:
            raise ValueError(f"codebook shape {actual_shape} != expected {exp_shapes[0]} for k={k} mode={mode} (mislabeled rung)")
        if not cb.is_contiguous():
            raise ValueError("signed codebook must be contiguous")
        if not torch.isfinite(cb).all():
            raise ValueError("signed codebook must be finite (NaN/Inf)")
        if bool((cb < 0).any()):
            raise ValueError("signed-mode magnitude codebook must be non-negative")
    else:
        raise ValueError(f"unsupported mode {mode!r} for physical validation")
    # Exact scale_coding
    if scale_coding != SCALE_CODING_TWO_TIER:
        raise ValueError(f"scale_coding must be 'two_tier' for attestation (v2 layout), got {scale_coding!r}")
    if fields.get("scale_coding") != SCALE_CODING_TWO_TIER:
        raise ValueError(f"fields scale_coding {fields.get('scale_coding')!r} != 'two_tier'")
    # Shape global exact
    shape = fields.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"fields shape must be 3-tuple/list, got {shape!r}")
    if tuple(int(d) for d in shape) != (int(E), int(R_total), int(C)):
        raise ValueError(f"fields shape {tuple(shape)} != (E,R_total,C) {(E,R_total,C)}")
    # Device consistency: all row tensors same device
    row_devices = []
    for kk in expected_row:
        v = fields[kk]
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"fields[{kk!r}] must be Tensor, got {type(v).__name__}")
        row_devices.append(str(v.device))
    if len(set(row_devices)) != 1:
        raise ValueError(f"row fields devices inconsistent {row_devices}")
    # Rank/shape/dtype/contiguity/finiteness/range per field
    ER = E * R_total
    nvec = C // VEC_DIM
    ng = C // FP4_GROUP
    n_sb = C // SUPERBLOCK
    if C % VEC_DIM != 0 or C % FP4_GROUP != 0 or C % SUPERBLOCK != 0:
        raise ValueError(f"C={C} must be multiple of VEC_DIM {VEC_DIM}, FP4_GROUP {FP4_GROUP}, SUPERBLOCK {SUPERBLOCK}")
    # indices
    v = fields["indices"]
    if v.dtype != torch.int64:
        raise ValueError(f"indices dtype {v.dtype} != torch.int64 (exact)")
    if not v.is_contiguous():
        raise ValueError("indices must be contiguous")
    if mode == "product":
        if v.ndim != 3:
            raise ValueError(f"indices ndim {v.ndim} != 3 for product")
        if tuple(int(d) for d in v.shape) != (ER, nvec, fam.n_sub):
            raise ValueError(f"indices shape {tuple(v.shape)} != expected {(ER, nvec, fam.n_sub)} for product")
        # range per subtable
        for si in range(fam.n_sub):
            max_val = (1 << bits[si]) - 1
            col = v[..., si]
            if bool((col < 0).any()) or bool((col > max_val).any()):
                raise ValueError(f"indices subtable {si} out of range [0,{max_val}]")
    else:  # signed or full
        if v.ndim != 2:
            raise ValueError(f"indices ndim {v.ndim} != 2 for {mode}")
        if tuple(int(d) for d in v.shape) != (ER, nvec):
            raise ValueError(f"indices shape {tuple(v.shape)} != expected {(ER, nvec)} for {mode}")
        max_val = (1 << bits[0]) - 1
        if bool((v < 0).any()) or bool((v > max_val).any()):
            raise ValueError(f"indices out of range [0,{max_val}] for {mode}")
    # signs
    if mode == "signed":
        sv = fields["signs"]
        if sv.dtype != torch.float32:
            raise ValueError(f"signs dtype {sv.dtype} != torch.float32")
        if not sv.is_contiguous():
            raise ValueError("signs must be contiguous")
        if sv.ndim != 2 or tuple(int(d) for d in sv.shape) != (ER, C):
            raise ValueError(f"signs shape {tuple(sv.shape)} != {(ER, C)}")
        if not torch.isfinite(sv).all():
            raise ValueError("signs must be finite")
        # exactly +/-1
        if not bool(((sv == 1.0) | (sv == -1.0)).all()):
            raise ValueError("signs values must be exactly +/-1")
    # scales
    sc = fields["scales"]
    if sc.dtype != torch.float32:
        raise ValueError(f"scales dtype {sc.dtype} != torch.float32")
    if not sc.is_contiguous():
        raise ValueError("scales must be contiguous")
    if sc.ndim != 2 or tuple(int(d) for d in sc.shape) != (ER, ng):
        raise ValueError(f"scales shape {tuple(sc.shape)} != {(ER, ng)}")
    if not torch.isfinite(sc).all():
        raise ValueError("scales must be finite")
    # scale_super
    sup = fields["scale_super"]
    if sup.dtype != torch.uint8:
        raise ValueError(f"scale_super dtype {sup.dtype} != torch.uint8")
    if not sup.is_contiguous():
        raise ValueError("scale_super must be contiguous")
    if sup.ndim != 2 or tuple(int(d) for d in sup.shape) != (ER, n_sb):
        raise ValueError(f"scale_super shape {tuple(sup.shape)} != {(ER, n_sb)}")
    # scale_sub
    sub = fields["scale_sub"]
    if sub.dtype != torch.int64:
        raise ValueError(f"scale_sub dtype {sub.dtype} != torch.int64 (exact)")
    if not sub.is_contiguous():
        raise ValueError("scale_sub must be contiguous")
    if sub.ndim != 2 or tuple(int(d) for d in sub.shape) != (ER, ng):
        raise ValueError(f"scale_sub shape {tuple(sub.shape)} != {(ER, ng)}")
    if bool((sub < 0).any()) or bool((sub > 15).any()):
        raise ValueError("scale_sub values must be in [0,15]")
    # Two-tier scale coherence: round-trip through _two_tier_scale_bytes + _two_tier_scale_unpack
    try:
        _sc_bytes = _two_tier_scale_bytes(sup, sub, n_sb)
        _sup2, _sub2, _scales_composed = _two_tier_scale_unpack(_sc_bytes)
    except Exception as exc:
        raise ValueError(f"two-tier scale coherence failed (illegal super/sub pair): {exc}") from exc
    if not torch.equal(_sup2, sup):
        raise ValueError("scale_super round-trip mismatch: unpacked super != input super")
    if not torch.equal(_sub2, sub):
        raise ValueError("scale_sub round-trip mismatch: unpacked sub != input sub")
    if not torch.equal(_scales_composed, sc):
        raise ValueError("scales != composed scale from super/sub (perturbed/negative/arbitrary scale)")
    # Assembled row byte width against declared layout
    try:
        expected_type_size = _serialized_type_size(k, grid, scale_coding)
    except Exception as exc:
        raise ValueError(f"failed to resolve type_size for k={k} grid={grid} scale_coding={scale_coding}: {exc}") from exc
    if C % SUPERBLOCK != 0:
        raise ValueError(f"C={C} must be multiple of SUPERBLOCK {SUPERBLOCK} for byte width validation")
    if expected_type_size <= 0:
        raise ValueError(f"invalid type_size {expected_type_size}")
    # Row byte width will be validated after assembly (packed.shape[1] must equal expected)


def canonical_nvfp4_cb_payload_identity(
    fields: Mapping[str, Any],
    k: int,
    *,
    grid: str,
    mode: str,
    format_name: str,
    physical_target: str,
    member_order: Sequence[str],
    slice_boundaries: Mapping[str, Sequence[int]],
    expert_ids: Sequence[int],
    qnames_per_leaf: Mapping[str, Sequence[str]],
    prepared: "ValidatedPackedLDLQEvidence",
    per_leaf_kept: Mapping[str, Sequence[bool]],
    scale_sweep: bool,
    encode_tier: str,
    dtype: str = "bfloat16",
    byte_order: str = "little",
) -> dict:
    """Hash the FULL assembled payload bytes with explicit ABI v4 framing.

    NVFP4 only: validates format<->k/grid/mode via registry, rejects FP8 and
    full. Requires and binds physical packed target name, exact member_order,
    ordered slices, exact nonnegative unique integer expert_ids, exact ordered
    qnames_per_leaf aligned to expert IDs via strict '.experts.<id>.' token
    and terminal leaf name, and exact evidence/split identity domain (policy/version and full
    original/FIT/HOLDOUT/split digests) snapshotted in Validated evidence.
    Requires per_leaf_kept to have exact leaf set, each value to be real bool
    (not truthy), each mask length exactly E. Validates exact resolved
    codebook/subtable shapes for family+rung, signs presence/absence, exact
    scale-plane field set, row shapes/dtypes, and assembled row byte width
    against declared layout before hashing. Binds mandatory scale_sweep=True and
    explicit encode_tier (fast/balanced/max) in full and row metadata, and exact
    per-row field schemas/dtypes/shapes plus codebook digest/schema. Hashes entire assembled payload
    bytes, entire codebook bytes, scale_coding, dtype/shape/byte_order, all
    exact topology/evidence/masks, with versioned canonical JSON envelope
    (allow_nan=False, sorted keys, fixed separators, UTF-8) and explicit
    length-prefix framing for metadata/codebook/packed bytes. Row hashes from
    exact assembled rows [local_ordinal*R_total+s : local_ordinal*R_total+t]
    bound to physical qname/expert id (local ordinal for storage), never
    source-weight bytes, and do not bind parent E/neighbor count/whole-parent shape/local expert ordinal. Strict field/global allowlists, no caught
    hashing/assembly exceptions, truncation, stringify fallback, or sidecar.
    Returns JSON-native lists/maps/scalars only (shape/lists not tuples).
    """
    # Strict type/shape validation, no silent coercion — exact types
    if not isinstance(fields, Mapping):
        raise ValueError("fields must be mapping")
    if type(k) is not int:
        raise ValueError(f"k must be exact int (bool rejected), got {type(k).__name__} {k!r}")
    if type(grid) is not str:
        raise ValueError(f"grid must be exact str, got {type(grid).__name__} {grid!r}")
    if type(mode) is not str:
        raise ValueError(f"mode must be exact str, got {type(mode).__name__} {mode!r}")
    if type(format_name) is not str or not format_name:
        raise ValueError(f"format_name must be non-empty exact str, got {format_name!r}")
    if type(physical_target) is not str or not physical_target:
        raise ValueError(f"physical_target must be non-empty exact str, got {physical_target!r}")
    if type(dtype) is not str or dtype != "bfloat16":
        raise ValueError(f"dtype must be exact string 'bfloat16', got {dtype!r} (only BF16 certified in this ABI)")
    if type(byte_order) is not str or byte_order != "little":
        raise ValueError(f"byte_order must be exact 'little', got {byte_order!r} (only little-endian certified)")
    if type(scale_sweep) is not bool:
        raise ValueError(f"scale_sweep must be exact bool, got {type(scale_sweep).__name__} {scale_sweep!r}")
    if scale_sweep is not True:
        raise ValueError(f"scale_sweep must be True (only True certified for this NVFP4-CB path), got {scale_sweep!r}")
    if type(encode_tier) is not str:
        raise ValueError(f"encode_tier must be exact str, got {type(encode_tier).__name__} {encode_tier!r}")
    if encode_tier not in _ENCODE_TIERS:
        raise ValueError(f"encode_tier {encode_tier!r} not in {_ENCODE_TIERS} (must be exact member, whitespace/case aliases rejected)")
    # NVFP4-only format validation (exact types, product/signed only, two_tier)
    expected_name = _validate_nvfp4_format(k, grid, mode, scale_coding=SCALE_CODING_TWO_TIER)
    if format_name != expected_name:
        raise ValueError(f"format_name {format_name!r} != expected NVFP4 name {expected_name!r} for k={k} grid={grid} mode={mode}")
    if not isinstance(member_order, (list, tuple)) or not member_order:
        raise ValueError(f"member_order must be non-empty sequence, got {member_order!r}")
    for _leaf in member_order:
        if type(_leaf) is not str:
            raise ValueError(f"member_order element {_leaf!r} must be exact str, got {type(_leaf).__name__}")
        if _leaf == "":
            raise ValueError(f"member_order element must be nonempty str, got {_leaf!r}")
    if len(set(member_order)) != len(member_order):
        raise ValueError(f"member_order must have unique leaves, got {member_order!r}")
    if tuple(member_order) not in (("gate_proj", "up_proj"), ("down_proj",)):
        raise ValueError(f"member_order {tuple(member_order)!r} must be exact routed-expert topology ('gate_proj','up_proj') or ('down_proj',)")
    if not isinstance(slice_boundaries, Mapping):
        raise ValueError(f"slice_boundaries must be mapping, got {type(slice_boundaries).__name__}")
    if set(slice_boundaries.keys()) != set(member_order):
        raise ValueError(f"slice_boundaries keys {sorted(slice_boundaries.keys())} != member_order {sorted(member_order)}")
    if not isinstance(expert_ids, (list, tuple)):
        raise ValueError(f"expert_ids must be sequence, got {type(expert_ids).__name__}")
    if len(expert_ids) == 0:
        raise ValueError("expert_ids must be non-empty")
    # expert_ids must be exact ints, nonnegative, unique
    for idx, eid in enumerate(expert_ids):
        if type(eid) is not int:
            raise ValueError(f"expert_ids[{idx}] must be exact int (bool rejected), got {type(eid).__name__} {eid!r}")
        if eid < 0:
            raise ValueError(f"expert_ids[{idx}] must be nonnegative, got {eid}")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError(f"expert_ids must be unique, got {expert_ids!r}")
    # per_leaf_kept strict bool validation
    if not isinstance(per_leaf_kept, Mapping):
        raise ValueError(f"per_leaf_kept must be mapping, got {type(per_leaf_kept).__name__}")
    if set(per_leaf_kept.keys()) != set(member_order):
        raise ValueError(f"per_leaf_kept keys {sorted(per_leaf_kept.keys())} != member_order {sorted(member_order)}")
    if not isinstance(qnames_per_leaf, Mapping):
        raise ValueError(f"qnames_per_leaf must be mapping, got {type(qnames_per_leaf).__name__}")
    if set(qnames_per_leaf.keys()) != set(member_order):
        raise ValueError(f"qnames_per_leaf keys {sorted(qnames_per_leaf.keys())} != member_order {sorted(member_order)}")
    # Validate prepared evidence domain — must be ValidatedPackedLDLQEvidence only
    if not isinstance(prepared, ValidatedPackedLDLQEvidence):
        raise ValueError(f"prepared must be ValidatedPackedLDLQEvidence (got {type(prepared).__name__}); raw PreparedLDLQEvidence rejected — construct ValidatedPackedLDLQEvidence first")
    # Shape must be list/tuple of exact ints (JSON-native: list on round-trip)
    shape = fields.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError(f"fields shape must be 3-D, got {shape!r}")
    # Allow both list and tuple on input but require exact int elements (no bool/float)
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if type(dim) is not int:
            raise ValueError(f"shape[{idx}] must be exact int, got {type(dim).__name__} {dim!r}")
        if dim <= 0:
            raise ValueError(f"shape[{idx}] must be positive, got {dim}")
    E, R_total, C = shape_list  # already ints
    if len(expert_ids) != E:
        raise ValueError(f"expert_ids length {len(expert_ids)} != E {E}")
    # Validate slices cover exactly R_total contiguously in member_order order — exact int types
    off = 0
    for leaf in member_order:
        bounds = slice_boundaries[leaf]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"slice_boundaries[{leaf!r}] must be (start,end) list, got {bounds!r}")
        if type(bounds[0]) is not int or type(bounds[1]) is not int:
            raise ValueError(f"slice_boundaries[{leaf!r}] must be (int,int) exact, got {bounds!r}")
        s, e = bounds[0], bounds[1]
        if s != off:
            raise ValueError(f"slice_boundaries[{leaf!r}] start {s} != expected {off}")
        if e <= s or e > R_total:
            raise ValueError(f"slice_boundaries[{leaf!r}] invalid {(s,e)} for R_total {R_total}")
        off = e
    if off != R_total:
        raise ValueError(f"slice_boundaries cover {off} != R_total {R_total}")
    # Physical target topology gate (defect 18): exactly one .experts. component, suffix matches member_order, prefix binding
    _target_parts = physical_target.split(".")
    if _target_parts.count("experts") != 1:
        raise ValueError(f"physical_target {physical_target!r} must contain exactly one 'experts' component, got {_target_parts.count('experts')}")
    _exp_idx = _target_parts.index("experts")
    # Routed-expert suffix must match member_order
    if tuple(member_order) == ("gate_proj", "up_proj"):
        _expected_tgt = "gate_up_proj"
    elif tuple(member_order) == ("down_proj",):
        _expected_tgt = "down_proj"
    else:
        raise ValueError(f"member_order {tuple(member_order)!r} has no physical_target suffix mapping")
    if _target_parts[-1] != _expected_tgt:
        raise ValueError(f"physical_target {physical_target!r} terminal {_target_parts[-1]!r} != expected {_expected_tgt!r} for member_order {tuple(member_order)!r}")
    if _exp_idx != len(_target_parts) - 2:
        raise ValueError(f"physical_target {physical_target!r} must be exactly '<prefix>.experts.{_expected_tgt}' with no intervening components after 'experts'")
    _target_prefix = ".".join(_target_parts[:_exp_idx])
    if not _target_prefix:
        raise ValueError(f"physical_target {physical_target!r} prefix before 'experts' must be non-empty")
    # Validate per_leaf_kept: each mask length exactly E, each element is real bool
    for leaf in member_order:
        mask = per_leaf_kept[leaf]
        if not isinstance(mask, (list, tuple)):
            raise ValueError(f"per_leaf_kept[{leaf!r}] must be list/tuple, got {type(mask).__name__}")
        if len(mask) != E:
            raise ValueError(f"per_leaf_kept[{leaf!r}] length {len(mask)} != E {E}")
        for idx, val in enumerate(mask):
            if type(val) is not bool:
                raise ValueError(f"per_leaf_kept[{leaf!r}][{idx}] must be real bool, got {type(val).__name__} {val!r} (truthy int/string rejected)")
    # Validate qnames_per_leaf: each leaf list length E, ordered by expert_ids, strict token, exactly one experts, prefix, uniqueness
    for leaf in member_order:
        qnames = qnames_per_leaf[leaf]
        if not isinstance(qnames, (list, tuple)):
            raise ValueError(f"qnames_per_leaf[{leaf!r}] must be list/tuple, got {type(qnames).__name__}")
        if len(qnames) != E:
            raise ValueError(f"qnames_per_leaf[{leaf!r}] length {len(qnames)} != E {E}")
        # Unique within each leaf (defect 10)
        if len(set(qnames)) != len(qnames):
            raise ValueError(f"qnames_per_leaf[{leaf!r}] must have unique qnames, got duplicates {qnames!r}")
        for idx, qn in enumerate(qnames):
            if type(qn) is not str:
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] must be exact str, got {type(qn).__name__} {qn!r}")
            if qn == "":
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] must be nonempty str, got {qn!r}")
            exp_id = expert_ids[idx]
            # Strict exact-one-experts check (defect 10): require exactly one 'experts' component
            _qn_parts = qn.split(".")
            if _qn_parts.count("experts") != 1:
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] {qn!r} must contain exactly one 'experts' component, got {_qn_parts.count('experts')}")
            _q_exp_idx = _qn_parts.index("experts")
            if _q_exp_idx + 1 >= len(_qn_parts) or _qn_parts[_q_exp_idx + 1] != str(exp_id):
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] {qn!r} does not contain exact component '.experts.{exp_id}.' for expert {exp_id}")
            if _qn_parts[-1] != leaf:
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] {qn!r} terminal {_qn_parts[-1]!r} != expected leaf {leaf!r} (must end with .{leaf})")
            # Prefix must equal physical_target prefix (no alternate layer/prefix, no intervening)
            _qn_prefix = ".".join(_qn_parts[:_q_exp_idx])
            if _qn_prefix != _target_prefix:
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] {qn!r} prefix {_qn_prefix!r} != physical_target prefix {_target_prefix!r} (alternate layer/prefix rejected)")
            # Exact form must be '<prefix>.experts.<id>.<leaf>' with no intervening components between id and leaf
            if _q_exp_idx + 2 != len(_qn_parts) - 1 or _qn_parts[_q_exp_idx + 2] != leaf:
                raise ValueError(f"qnames_per_leaf[{leaf!r}][{idx}] {qn!r} must be exactly '<prefix>.experts.{exp_id}.{leaf}' with no intervening components")
    # Validate prepared evidence matches E, C, policy/version, and holds full digests — cheap fingerprint only
    _validate_validated_evidence(prepared, E, C)
    # Strict field allowlist: fields must contain only _ROW_FIELDS ∪ _GLOBAL_FIELDS
    for kk in fields:
        if kk not in (_ROW_FIELDS | _GLOBAL_FIELDS):
            raise ValueError(f"fields contains unknown key {kk!r} not in allowlist {sorted(_ROW_FIELDS | _GLOBAL_FIELDS)}")
    if "indices" not in fields:
        raise ValueError("fields missing required row field 'indices'")
    # Validate field row alignment: every present row field must have first dim == E*R_total
    for kk in _ROW_FIELDS:
        if kk in fields:
            v = fields[kk]
            if not isinstance(v, torch.Tensor):
                raise ValueError(f"fields[{kk!r}] must be Tensor, got {type(v).__name__}")
            if int(v.shape[0]) != E * R_total:
                raise ValueError(f"fields[{kk!r}] rows {int(v.shape[0])} != E*R_total {E*R_total}")
    # Physical validation: exact codebook/subtable shapes, signs, scale plane, row dtypes, byte width
    scale_coding = fields.get("scale_coding", SCALE_CODING_TWO_TIER)
    if scale_coding != SCALE_CODING_TWO_TIER:
        raise ValueError(f"fields scale_coding {scale_coding!r} != 'two_tier' (legacy v1 not certified)")
    _validate_fields_physically(fields, k, grid, mode, scale_coding, E, R_total, C)
    # Assembler is authoritative — never caught, never truncated, no stringify fallback
    # Validate assembled byte width against declared layout before hashing
    packed = nvfp4_cb_assemble_bytes(dict(fields), k, grid=grid, mode=mode)
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError(f"assembled payload must be uint8 Tensor, got {type(packed).__name__}/{getattr(packed, 'dtype', None)}")
    if int(packed.shape[0]) != E * R_total:
        raise ValueError(f"assembled rows {packed.shape[0]} != E*R_total {E*R_total}")
    # Validate byte width: packed.shape[1] must equal n_sb * type_size for declared layout
    n_sb_check = C // SUPERBLOCK
    if C % SUPERBLOCK != 0:
        raise ValueError(f"C={C} must be multiple of SUPERBLOCK {SUPERBLOCK}")
    expected_type_size = _serialized_type_size(k, grid, SCALE_CODING_TWO_TIER)
    expected_bytes_per_row = n_sb_check * expected_type_size
    if int(packed.shape[1]) != expected_bytes_per_row:
        raise ValueError(f"assembled bytes per row {packed.shape[1]} != expected {expected_bytes_per_row} (n_sb={n_sb_check} * type_size={expected_type_size} for k={k} grid={grid} scale_coding=two_tier)")
    # Codebook digest: framed blob binds ordinal/count/shape/dtype/byte_order/bytes
    codebook = fields.get("codebook")
    if codebook is None:
        raise ValueError("fields missing codebook (NVFP4-CB requires resolved codebook for attestation)")
    codebook_blob = _codebook_blob_bytes(codebook)
    codebook_digest = hashlib.sha256(codebook_blob).hexdigest()
    scale_coding = fields.get("scale_coding", SCALE_CODING_TWO_TIER)
    if scale_coding != SCALE_CODING_TWO_TIER:
        raise ValueError(f"scale_coding {scale_coding!r} != 'two_tier' (legacy v1 not certified at attestation)")
    # Build codebook schema for row hash (exact shapes)
    from .cb_layout import codebook_subtable_shapes as _cb_shapes2, family_for as _fam2
    fam2 = _fam2(grid, mode)
    exp_shapes2 = _cb_shapes2(k, mode, fam2.n_sub)
    codebook_schema = [list(s) for s in exp_shapes2] if mode == "product" else [list(exp_shapes2[0])]
    packed_bytes = _canonical_le_bytes(packed)
    packed_digest = hashlib.sha256(packed_bytes).hexdigest()
    # Per-field framed blobs for full hash (name/order/shape/dtype/byte_order/bytes)
    field_order = ["indices", "signs", "scales", "scale_super", "scale_sub"]
    field_framed_blobs: list[bytes] = []
    for fname in field_order:
        if fname in fields:
            t = fields[fname]
            name_b = fname.encode("utf-8")
            shape = tuple(int(d) for d in t.shape)
            dtype_b = str(t.dtype).encode("utf-8")
            bo_b = b"little"
            b = _canonical_le_bytes(t)
            fb = b""
            fb += struct.pack(">Q", len(name_b)) + name_b
            fb += struct.pack(">Q", len(shape))
            for dim in shape:
                fb += struct.pack(">Q", int(dim))
            fb += struct.pack(">Q", len(dtype_b)) + dtype_b
            fb += struct.pack(">Q", len(bo_b)) + bo_b
            fb += struct.pack(">Q", len(b)) + b
            # Outer framing for this field blob
            field_framed_blobs.append(struct.pack(">Q", len(fb)) + fb)
    # Build per-field schemas for binding (exact name/order/shape/dtype/byte_order)
    _field_schemas = {}
    for fname in sorted(field_order):
        if fname in fields:
            t = fields[fname]
            _field_schemas[fname] = {"shape": list(int(d) for d in t.shape), "dtype": str(t.dtype), "byte_order": "little"}
    # Metadata dict for envelope — all binding fields, JSON-native, includes per-field meta for full parent
    full_metadata = {
        "abi": _PAYLOAD_IDENTITY_VERSION,
        "byte_order": str(byte_order),
        "codebook_digest": str(codebook_digest),
        "codebook_schema": codebook_schema,
        "dtype": str(dtype),
        "encode_tier": str(encode_tier),
        "evidence_fit_digests": list(prepared.fit_digests),
        "evidence_holdout_digests": list(prepared.holdout_digests),
        "evidence_original_digests": list(prepared.original_digests),
        "evidence_policy": str(prepared.policy),
        "evidence_split_infos": [dict(s) for s in prepared.split_infos],
        "evidence_split_infos_digest": str(prepared.split_infos_digest),
        "evidence_version": int(prepared.version),
        "expert_ids": list(int(x) for x in expert_ids),
        "field_schemas": _field_schemas,
        "format": str(format_name),
        "grid": str(grid),
        "k": int(k),
        "member_order": list(member_order),
        "mode": str(mode),
        "packed_digest": str(packed_digest),
        "packed_shape": list(int(x) for x in packed.shape),
        "per_leaf_kept": {kk: list(map(bool, v)) for kk, v in per_leaf_kept.items()},
        "physical_target": str(physical_target),
        "qnames_per_leaf": {kk: list(v) for kk, v in qnames_per_leaf.items()},
        "scale_coding": str(scale_coding),
        "scale_sweep": bool(scale_sweep),
        "shape": list(int(x) for x in shape_list),
        "slice_boundaries": {kk: list(v) for kk, v in slice_boundaries.items()},
    }
    metadata_bytes = _canonical_json_bytes(full_metadata)
    h_full = hashlib.sha256()
    h_full.update(struct.pack(">Q", len(metadata_bytes)))
    h_full.update(metadata_bytes)
    for fb in field_framed_blobs:
        h_full.update(fb)
    h_full.update(struct.pack(">Q", len(codebook_blob)))
    h_full.update(codebook_blob)
    h_full.update(struct.pack(">Q", len(packed_bytes)))
    h_full.update(packed_bytes)
    full_hash = h_full.hexdigest()
    # Per-leaf/per-expert row hashes — composable across E subsets, binds decode authorities
    # Row identity over local leaf block + all semantic decode authorities, not neighbor count
    # Build leaf-local per-row field schema: shape [R_leaf] + trailing_shape for every present row field
    row_hashes: dict[str, list[str]] = {}
    for leaf in member_order:
        s, e = slice_boundaries[leaf][0], slice_boundaries[leaf][1]  # exact ints
        hashes: list[str] = []
        qnames_leaf = qnames_per_leaf[leaf]
        R_leaf = int(e - s)
        # Leaf-local per-row field schemas: bind only this leaf's rows, not parent E or E*R_total
        leaf_field_schemas: dict[str, dict] = {}
        for fname in sorted(field_order):
            if fname in fields:
                t = fields[fname]
                trailing = tuple(int(d) for d in t.shape[1:])
                leaf_shape = [int(R_leaf)] + list(trailing)
                leaf_field_schemas[fname] = {"shape": leaf_shape, "dtype": str(t.dtype), "byte_order": "little"}
        for idx, expert_id in enumerate(expert_ids):
            start = idx * R_total + s
            end = idx * R_total + e
            leaf_block = packed[start:end]
            leaf_block_bytes = _canonical_le_bytes(leaf_block)
            qname = qnames_leaf[idx]
            # Per-expert evidence and mask for this row
            ev_orig = prepared.original_digests[idx] if idx < len(prepared.original_digests) else ""
            ev_fit = prepared.fit_digests[idx] if idx < len(prepared.fit_digests) else ""
            ev_hold = prepared.holdout_digests[idx] if idx < len(prepared.holdout_digests) else ""
            # Find split info for this expert, excluding local expert ordinal (row must not bind it)
            ev_split_info_raw = dict(prepared.split_infos[idx]) if idx < len(prepared.split_infos) else {}
            ev_split_info = {k: v for k, v in ev_split_info_raw.items() if k != "expert"}
            leaf_kept_val = bool(per_leaf_kept[leaf][idx]) if leaf in per_leaf_kept and idx < len(per_leaf_kept[leaf]) else False
            row_meta = {
                "abi": _PAYLOAD_IDENTITY_VERSION,
                "byte_order": str(byte_order),
                "codebook_digest": str(codebook_digest),
                "codebook_schema": codebook_schema,
                "dtype": str(dtype),
                "encode_tier": str(encode_tier),
                "evidence_fit_digest": str(ev_fit),
                "evidence_holdout_digest": str(ev_hold),
                "evidence_original_digest": str(ev_orig),
                "evidence_policy": str(prepared.policy),
                "evidence_split_info": dict(ev_split_info),
                "evidence_version": int(prepared.version),
                "expert_id": int(expert_id),
                "field_schemas": leaf_field_schemas,
                "format": str(format_name),
                "grid": str(grid),
                "k": int(k),
                "leaf": str(leaf),
                "leaf_block_shape": [int(R_leaf), int(C)],
                "member_order": list(member_order),
                "mode": str(mode),
                "per_leaf_kept": bool(leaf_kept_val),
                "physical_target": str(physical_target),
                "qname": str(qname),
                "scale_coding": str(scale_coding),
                "scale_sweep": bool(scale_sweep),
                "slice": [int(s), int(e)],
            }
            row_meta_bytes = _canonical_json_bytes(row_meta)
            h_row = hashlib.sha256()
            h_row.update(struct.pack(">Q", len(row_meta_bytes)))
            h_row.update(row_meta_bytes)
            h_row.update(struct.pack(">Q", len(leaf_block_bytes)))
            h_row.update(leaf_block_bytes)
            hashes.append(h_row.hexdigest())
        row_hashes[leaf] = hashes
    return {
        "abi": _PAYLOAD_IDENTITY_VERSION,
        "format": str(format_name),
        "k": int(k),
        "grid": str(grid),
        "mode": str(mode),
        "physical_target": str(physical_target),
        "dtype": str(dtype),
        "shape": list(int(x) for x in shape_list),
        "byte_order": str(byte_order),
        "member_order": list(member_order),
        "slice_boundaries": {kk: list(v) for kk, v in slice_boundaries.items()},
        "expert_ids": list(int(x) for x in expert_ids),
        "qnames_per_leaf": {kk: list(v) for kk, v in qnames_per_leaf.items()},
        "per_leaf_kept": {kk: list(map(bool, v)) for kk, v in per_leaf_kept.items()},
        "evidence_policy": str(prepared.policy),
        "evidence_version": int(prepared.version),
        "evidence_original_digests": list(prepared.original_digests),
        "evidence_fit_digests": list(prepared.fit_digests),
        "evidence_holdout_digests": list(prepared.holdout_digests),
        "evidence_split_infos_digest": str(prepared.split_infos_digest),
        "evidence_split_infos": [dict(s) for s in prepared.split_infos],
        "scale_coding": str(scale_coding),
        "scale_sweep": bool(scale_sweep),
        "encode_tier": str(encode_tier),
        "field_schemas": _field_schemas,
        "codebook_digest": str(codebook_digest),
        "codebook_schema": codebook_schema,
        "packed_shape": list(int(x) for x in packed.shape),
        "packed_digest": str(packed_digest),
        "full_hash": str(full_hash),
        "row_hashes": {kk: list(v) for kk, v in row_hashes.items()},
    }


def _strict_json_validate(obj: Any, path: str = "$") -> None:
    """Validate JSON-native schema recursively, exact scalar types, no NaN."""
    if obj is None:
        return
    if type(obj) is bool:
        return
    if type(obj) is int:
        return
    if type(obj) is float:
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            raise ValueError(f"JSON schema at {path}: non-finite float {obj!r}")
        # JSON disallows NaN/Inf, already checked, and allow_nan=False will catch
        return
    if type(obj) is str:
        return
    if type(obj) is list:
        for i, v in enumerate(obj):
            _strict_json_validate(v, f"{path}[{i}]")
        return
    if type(obj) is tuple:
        raise ValueError(f"JSON schema at {path}: tuple not allowed, use list (got {type(obj).__name__})")
    if type(obj) is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise ValueError(f"JSON schema at {path}: dict key must be exact str, got {type(k).__name__} {k!r}")
            _strict_json_validate(v, f"{path}.{k}")
        return
    if isinstance(obj, Mapping):
        raise ValueError(f"JSON schema at {path}: Mapping subclass not allowed, use exact dict (got {type(obj).__name__})")
    # Catch list subclasses as well
    if isinstance(obj, list):
        raise ValueError(f"JSON schema at {path}: list subclass not allowed, use exact list (got {type(obj).__name__})")
    raise ValueError(f"JSON schema at {path}: unsupported type {type(obj).__name__} {obj!r}")


def _strict_recursive_equal(a: Any, b: Any, path: str = "$") -> None:
    """Strict recursive typed equality, never Python bool/int conflation, missing/extra keys fail."""
    if type(a) is dict and type(b) is dict:
        if set(a.keys()) != set(b.keys()):
            missing = sorted(set(a.keys()) - set(b.keys()))
            extra = sorted(set(b.keys()) - set(a.keys()))
            raise ValueError(f"strict mismatch at {path}: missing {missing} extra {extra}")
        for k in a:
            _strict_recursive_equal(a[k], b[k], f"{path}.{k}")
        return
    if type(a) is list and type(b) is list:
        if len(a) != len(b):
            raise ValueError(f"strict mismatch at {path}: list len {len(a)} != {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b)):
            _strict_recursive_equal(av, bv, f"{path}[{i}]")
        return
    if isinstance(a, list) or isinstance(b, list):
        raise ValueError(f"strict mismatch at {path}: list type must be exact list (got {type(a).__name__}/{type(b).__name__})")
    # Scalars: exact type and value
    if type(a) is not type(b):
        raise ValueError(f"strict mismatch at {path}: type {type(a).__name__} {a!r} != {type(b).__name__} {b!r}")
    if a != b:
        raise ValueError(f"strict mismatch at {path}: {a!r} != {b!r}")


def verify_canonical_payload_identity(
    fields: Mapping[str, Any],
    k: int,
    *,
    grid: str,
    mode: str,
    format_name: str,
    physical_target: str,
    member_order: Sequence[str],
    slice_boundaries: Mapping[str, Sequence[int]],
    expert_ids: Sequence[int],
    qnames_per_leaf: Mapping[str, Sequence[str]],
    prepared: "ValidatedPackedLDLQEvidence",
    per_leaf_kept: Mapping[str, Sequence[bool]],
    scale_sweep: bool,
    encode_tier: str,
    dtype: str = "bfloat16",
    byte_order: str = "little",
    expected: Mapping[str, Any],
) -> dict:
    """Verify complete canonical identity.

    Requires a complete expected identity of the exact ABI/schema (v4) and
    compares via strict JSON-native canonical bytes (allow_nan=False) and recursive
    typed validator, never Python bool/int equality. Rejects strings, partial
    mappings, missing/extra/nested-extra keys, malformed hashes, mismatched topology/
    masks/evidence, and legacy ABI (v1/v2/v3). Complete expected mapping mandatory.
    """
    if type(expected) is not dict:
        raise ValueError(f"expected must be exact dict (Mapping/list subclass rejected), got {type(expected).__name__} {expected!r} (strings rejected)")
    # Reject legacy ABI early — v1, v2, v3 are legacy when current is v4
    exp_abi = expected.get("abi")
    if exp_abi != _PAYLOAD_IDENTITY_VERSION:
        raise ValueError(f"expected abi {exp_abi!r} != required {_PAYLOAD_IDENTITY_VERSION!r} (legacy {exp_abi!r} rejected)")
    # Also explicitly reject v1/v2/v3 if somehow equal to current (defensive)
    if exp_abi in (_PAYLOAD_IDENTITY_V1, _PAYLOAD_IDENTITY_V2, _PAYLOAD_IDENTITY_V3):
        raise ValueError(f"legacy ABI {exp_abi!r} rejected")
    # Validate expected is JSON-native strictly before comparison
    _strict_json_validate(expected, "$expected")
    # Compute actual identity (no caught exceptions, no truncation)
    actual = canonical_nvfp4_cb_payload_identity(
        fields, k, grid=grid, mode=mode, format_name=format_name, physical_target=physical_target,
        member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids,
        qnames_per_leaf=qnames_per_leaf, prepared=prepared, per_leaf_kept=per_leaf_kept,
        scale_sweep=scale_sweep, encode_tier=encode_tier, dtype=dtype, byte_order=byte_order,
    )
    _strict_json_validate(actual, "$actual")
    # Require complete expected identity — exact key set, no missing/extra at top level
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        missing = sorted(actual_keys - expected_keys)
        extra = sorted(expected_keys - actual_keys)
        raise ValueError(f"expected identity missing keys {missing} extra keys {extra} (complete identity required)")
    # Compare canonical JSON bytes (allow_nan=False) — strict scalar types, no bool/int conflation
    try:
        actual_bytes = _canonical_json_bytes(actual)
        expected_bytes = _canonical_json_bytes(expected)
    except ValueError as exc:
        raise ValueError(f"JSON canonicalization failed (non-JSON-native or NaN): {exc}") from exc
    if actual_bytes != expected_bytes:
        # Provide detailed strict recursive diff for diagnostics (still fail closed)
        _strict_recursive_equal(actual, expected, "$")
        # Fallback if recursive didn't raise but bytes differ (should not happen)
        raise ValueError(f"payload identity mismatch: canonical JSON bytes differ")
    # Additional malformed hash check
    for hk in ("full_hash", "packed_digest", "codebook_digest", "evidence_split_infos_digest"):
        hv = expected.get(hk)
        if not isinstance(hv, str) or len(hv) != 64 or any(c not in "0123456789abcdef" for c in hv.lower()):
            raise ValueError(f"payload identity {hk!r} malformed hash {hv!r}")
    return actual


def nvfp4_cb_pack(w: torch.Tensor, k: int, *, grid: str = "fp4",
                  mode: str = "product",
                  col_weights: torch.Tensor | None = None,
                  codebook: torch.Tensor | tuple | None = None,
                  scale_sweep: bool = True,
                  scale_coding: str = SCALE_CODING_V1,
                  encode_tier: str | None = None,
                  warm_scale_state: dict[str, torch.Tensor] | None = None,
                  ldlq: bool = False,
                  activation_rows: torch.Tensor | Sequence[torch.Tensor] | None = None,
                  return_gate_info: bool = False,
                  ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, dict | None]:
    """Quantize + bit-pack a weight in one call (mirrors ``gguf_pack``).

    Returns ``(packed uint8 (rows, bytes_per_row), fields)``; ``fields``
    carries ``scales`` (per-channel fp8 scale plane the exporter ships
    separately) and the resolved ``codebook``. When ``return_gate_info`` is
    True, also returns the exact gate telemetry dict from the shared
    ``ldlq_reassign_cb_fields_gated`` path (or ``None`` when LDLQ not
    applied).
    """
    fields = nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                             col_weights=col_weights, codebook=codebook,
                             scale_sweep=scale_sweep,
                             scale_coding=scale_coding,
                             encode_tier=encode_tier,
                             warm_scale_state=warm_scale_state)
    gate_info: dict | None = None
    if ldlq:
        if grid != "fp4":
            raise ValueError(f"LDLQ CB packing requires grid='fp4' (NVFP4 only), got {grid!r} — FP8+LDLQ rejected fail-closed")
        if col_weights is None:
            raise ValueError("LDLQ CB packing requires activation-weighted col_weights")
        if activation_rows is None:
            raise ValueError("LDLQ CB packing requires calibration activation rows")
        if _ldlq_gate_enabled():
            # Same gate as cb_fields_for_context: do-no-harm on activation_output_mse.
            fields, gate_info = ldlq_reassign_cb_fields_gated(
                w,
                fields,
                col_weights,
                activation_rows,
                grid=grid,
                mode=mode,
                k=k,
            )
            # Preserve observed gate dict (copy) for telemetry.
            gate_info = dict(gate_info) if isinstance(gate_info, dict) else gate_info
        else:
            fields = ldlq_reassign_cb_fields(
                w,
                fields,
                col_weights,
                activation_rows,
                grid=grid,
                mode=mode,
            )
            gate_info = {"gate": "disabled", "kept_ldlq": True, "metric": "activation_output_mse"}
    packed = nvfp4_cb_assemble_bytes(fields, k, grid=grid, mode=mode)
    if return_gate_info:
        return packed, fields, gate_info
    return packed, fields
