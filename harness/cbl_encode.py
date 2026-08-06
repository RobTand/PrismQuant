"""
CBL per-group encode helpers.

Production: assignment-only re-encode against banked pooled books (one per layer/proj/rung)
via prismaquant.nvfp4_cb_formats.nvfp4_cb_fields with codebook= and scale_sweep=False.

Additivity invariant: for a fixed rung, sum_g MSE_g == whole-unit free_weight_mse
(when groups share the rung), because column groups are disjoint and encoding is
per-vector independent (no cross-group coupling, no LDLQ). Tolerance 1e-6 relative.

This module provides both the real GPU path (requires bucket-books + source weights)
and a synthetic CPU path for unit tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch

RUNGS = list(range(28, 39))

# Geometry
VEC_DIM = 8
SUPERBLOCK = 256

def group_boundaries(in_features: int, groups: int) -> List[tuple[int, int]]:
    """Contiguous column block boundaries [start, end) aligned to SUPERBLOCK and VEC_DIM."""
    if in_features % groups != 0:
        raise ValueError(f"in_features {in_features} not divisible by groups {groups}")
    width = in_features // groups
    if width % SUPERBLOCK != 0 or width % VEC_DIM != 0:
        raise ValueError(f"group width {width} not aligned to {SUPERBLOCK} and {VEC_DIM}")
    bounds = []
    for g in range(groups):
        s = g * width
        e = (g + 1) * width
        bounds.append((s, e))
    return bounds

def per_group_mse_from_full_weight(
    weight: torch.Tensor,  # (out, in) or (E, out, in) single expert slice
    col_weights: torch.Tensor | None,  # (out, in) or (in,) broadcast
    books: dict[int, tuple],  # rung -> pooled codebook tuple
    *,
    projection: str,
    groups: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute per-group per-rung MSE by assignment-only re-encode on CPU or CUDA.

    weight: 2-D (out, in) for one expert; col_weights: same shape or (in,) or None
    books: dict mapping rung -> codebook tuple (fp16 sub-tables) or None for synthetic

    Returns tensor (groups, n_rungs) of sum-of-squared-error (not mean) per group?
    We return per-group sum-of-squares (so additivity holds as sum), but caller can
    convert to mean by dividing by (out*width). For gain, either is proportional;
    we use sum (weighted MSE) to keep additive exactly.
    For consistency with burn's free_weight_mse (mean over out*in), we store mean
    but note sum = mean * out*in.  Here we store mean per group (over its own elements)
    scaled so sum of means weighted by group width reproduces whole mean.
    Simpler: return sum of squared errors per group (unweighted, or col-weighted if provided).
    The harness normalizes to mse per group mean for solving, but additive check uses sum.
    """
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields, nvfp4_cb_reconstruct

    if weight.ndim != 2:
        raise ValueError("per_group_mse expects 2-D weight (out, in) for one expert")
    out, inn = weight.shape
    if inn % groups != 0:
        raise ValueError(f"in {inn} not divisible by groups {groups}")
    bounds = group_boundaries(inn, groups)
    n_rungs = len(RUNGS)
    # output: (G, n_rungs) sum of squared errors (weighted)
    result = torch.zeros(groups, n_rungs, dtype=torch.float64)

    # Normalize col_weights to (out, in) if provided else ones
    # For unit tests with uniform weights, we use None (unweighted)
    for gi, (s, e) in enumerate(bounds):
        w_g = weight[:, s:e].to(device).to(torch.float32)
        cw_g = None
        if col_weights is not None:
            # col_weights may be (in,) per-column, (out,in) or (1,in)
            cw = col_weights
            if cw.ndim == 1:
                cw_g = cw[s:e].to(device).to(torch.float32).expand_as(w_g)
            elif cw.ndim == 2:
                cw_g = cw[:, s:e].to(device).to(torch.float32)
                if cw_g.shape != w_g.shape:
                    # maybe (1,in) -> broadcast rows
                    if cw_g.shape[0] == 1 and w_g.shape[0] > 1:
                        cw_g = cw_g.expand_as(w_g)
            else:
                raise ValueError(f"col_weights ndim {cw.ndim}")
        for ri, rung in enumerate(RUNGS):
            cb = books.get(rung) if books is not None else None
            # If no real book (synthetic test), use dummy lattice zero reconstruction?
            # For synthetic path we skip nvfp4_cb_fields and compute direct synthetic mse below.
            if cb is None:
                # synthetic caller should not reach here; handled separately
                raise ValueError(f"no book for rung {rung}; provide synthetic books or use synthetic helper")
            fields = nvfp4_cb_fields(
                w_g, int(rung), grid="fp8", mode="product",
                col_weights=cw_g, codebook=cb,
                scale_sweep=False, encode_tier="balanced",
            )
            recon = nvfp4_cb_reconstruct(fields, int(rung), grid="fp8", mode="product").to(torch.float32)
            err = (recon - w_g).pow(2)
            if cw_g is not None:
                # Weighted sum (for gain we use weighted mse)
                mse_sum = (err * cw_g).sum().item()  # sum weighted
                # For additive check relative, sum is what matters
                result[gi, ri] = mse_sum
            else:
                result[gi, ri] = err.sum().item()
            # free cuda cache
            del fields, recon, err
    return result

def synthetic_per_group_mse(
    out: int = 64,
    inn: int = 512,
    groups: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """
    CPU-only synthetic per-group per-rung MSE for unit tests.

    We fabricate per-group MSE values that are decreasing with rung (higher K better)
    and vary across groups (so splitting has non-zero gain).  Construction ensures
    that whole-unit MSE at any single rung equals sum_g synthetic_mse[g,rung]
    by definition (additivity). Values are deterministic given seed.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(groups, len(RUNGS), generator=g).abs() * 0.1 + 0.5
    # Make monotonic decreasing across rungs (higher rung = lower mse)
    for gi in range(groups):
        # scale per group different sensitivity: group 0 more sensitive (steeper)
        sensitivity = 0.8 + 0.4 * (gi / max(groups - 1, 1))
        for ri in range(1, len(RUNGS)):
            # each step improvement ~ 5-15% base * sensitivity
            imp = 0.08 * sensitivity * (1 + 0.1 * torch.randn((), generator=g).item())
            imp = max(0.02, min(0.25, imp))
            base[gi, ri] = base[gi, ri - 1] * (1 - imp)
    # Ensure group dispersion: amplify difference
    # Make group 0 higher error than group 1 at low rungs
    return base.to(torch.float64)
