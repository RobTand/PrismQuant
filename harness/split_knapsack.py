"""
Split knapsack for within-linear mixed-rung assignment.

Given per-group per-rung MSE table shape (G, n_rungs) where rungs = [K28..K38]
and per-group bytes per rung (linear in K), find minimum-total-MSE split with
total bytes == budget (equal-bytes vs uniform).

We assume shared row-scale (4*out once), so bytes per group = out*sb_g*4*k + 0 incremental.
Total index bytes = out*sb_g*4*sum(ks).  Equality to uniform budget K_budget:
sum(ks) == G * K_budget   (exact integer).  This is an integer partition knapsack.

For per-group-scale model (G*4*out), sum(ks) == G*K_budget - (G-1)*?  Quantified
separately; search handles arbitrary per-rung byte cost via DP.

Solver: DP over groups and byte levels.  G<=8, n_rungs=11, byte domain small:
out*sb_g*4*k range ~ 0.5M-3M, sum range ~ 1M-25M but only 11^G combos.  DP via
dict of achievable total bytes -> (mse, choice) per prefix is exact and fast.
If multiple combos tie on MSE, smallest lexicographic ks wins (deterministic).
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

import torch

# FP8_CB_K rungs in study menu
RUNGS = list(range(28, 39))  # K28..K38 inclusive
RUNG_TO_IDX = {k: i for i, k in enumerate(RUNGS)}

def bytes_per_group_for_rung(projection: str, groups: int, rung: int, *, per_group_scale: bool = False) -> int:
    from .byte_accounting import PER_PROJ_SHAPE, SUPERBLOCK, INDEX_BYTES_PER_K, FP32_BYTES
    out, inn = PER_PROJ_SHAPE[projection]
    in_g = inn // groups
    sb_g = in_g // SUPERBLOCK
    idx = out * sb_g * INDEX_BYTES_PER_K * int(rung)
    # In shared-scale model, scale not per group; account 0 here, add once at top-level
    # For DP we want per-group additive bytes including its share of scale if per_group_scale.
    if per_group_scale:
        # Distribute scale as 4*out per group (already in bytes_uniform); but for
        # split we can add per group so total is G*scale.  DP will sum.
        return int(idx + FP32_BYTES * out)
    else:
        return int(idx)

def total_bytes_for_ks(projection: str, groups: int, ks: List[int], *, per_group_scale: bool = False) -> int:
    from .byte_accounting import PER_PROJ_SHAPE, SUPERBLOCK, INDEX_BYTES_PER_K, FP32_BYTES
    out, inn = PER_PROJ_SHAPE[projection]
    in_g = inn // groups
    sb_g = in_g // SUPERBLOCK
    total_idx = sum(out * sb_g * INDEX_BYTES_PER_K * int(k) for k in ks)
    if per_group_scale:
        total = total_idx + FP32_BYTES * out * groups
    else:
        total = total_idx + FP32_BYTES * out
    return int(total)

def solve_split(
    per_group_mse: torch.Tensor,  # shape (G, n_rungs)
    projection: str,
    groups: int,
    budget_bytes: int,
    *,
    per_group_scale: bool = False,
) -> Tuple[List[int], float]:
    """
    Returns (best_ks, best_mse) with total_bytes == budget_bytes minimizing sum_g MSE_g[k_g].
    If no exact byte match, returns (None, inf) -> caller should treat as no feasible split.
    per_group_mse is (G, 11) ordered by RUNGS (K28..K38).
    """
    G = int(groups)
    assert per_group_mse.shape[0] == G and per_group_mse.shape[1] == len(RUNGS)
    # Precompute bytes per group per rung
    bytes_table = torch.tensor(
        [bytes_per_group_for_rung(projection, G, k, per_group_scale=per_group_scale) for k in RUNGS],
        dtype=torch.int64,
    )
    # For shared-scale model, bytes_table excludes global scale; add once to budget comparison.
    # The per-group bytes above is 0-scale for shared model, so each group contributes only idx.
    # To compare total vs budget, we need to account that budget includes 1 scale.
    # Approach: DP sum of per-group idx bytes, then compare sum + scale == budget.
    # For per_group_scale, bytes_table already includes scale per group, DP sum directly vs budget.

    if not per_group_scale:
        from .byte_accounting import FP32_BYTES, PER_PROJ_SHAPE
        out = PER_PROJ_SHAPE[projection][0]
        scale_once = FP32_BYTES * out
        # DP sums idx only; target idx sum = budget - scale_once
        target_idx = budget_bytes - scale_once
        # DP over groups
        # dp maps total_idx -> (mse, ks_tuple)
        dp: dict[int, tuple[float, tuple]] = {0: (0.0, ())}
        for g in range(G):
            ndp: dict[int, tuple[float, tuple]] = {}
            for total_idx_so_far, (mse_so_far, ks_so_far) in dp.items():
                for ri, k in enumerate(RUNGS):
                    nmse = mse_so_far + float(per_group_mse[g, ri].item())
                    nidx = total_idx_so_far + int(bytes_table[ri].item())  # bytes_table is idx only here
                    # keep best mse per nidx (tie: lexicographically smallest ks)
                    prev = ndp.get(nidx)
                    cand_ks = ks_so_far + (k,)
                    if prev is None or nmse < prev[0] - 1e-12 or (abs(nmse - prev[0]) < 1e-12 and cand_ks < prev[1]):
                        ndp[nidx] = (nmse, cand_ks)
            # prune to keep dp size bounded (at most 11*G distinct sums, actually combos = G*10+1)
            dp = ndp
        # lookup target
        entry = dp.get(int(target_idx))
        if entry is None:
            return None, float("inf")
        best_mse, best_ks_t = entry
        return list(best_ks_t), float(best_mse)
    else:
        # per_group_scale: dp sum directly -> budget
        dp2: dict[int, tuple[float, tuple]] = {0: (0.0, ())}
        for g in range(G):
            ndp: dict[int, tuple[float, tuple]] = {}
            for total, (mse_so_far, ks_so_far) in dp2.items():
                for ri, k in enumerate(RUNGS):
                    nmse = mse_so_far + float(per_group_mse[g, ri].item())
                    ntotal = total + int(bytes_table[ri].item())
                    cand_ks = ks_so_far + (k,)
                    prev = ndp.get(ntotal)
                    if prev is None or nmse < prev[0] - 1e-12 or (abs(nmse - prev[0]) < 1e-12 and cand_ks < prev[1]):
                        ndp[ntotal] = (nmse, cand_ks)
            dp2 = ndp
        entry = dp2.get(int(budget_bytes))
        if entry is None:
            return None, float("inf")
        best_mse, best_ks_t = entry
        return list(best_ks_t), float(best_mse)

def uniform_mse(per_group_mse: torch.Tensor, uniform_k: int) -> float:
    """Sum_g MSE_g[uniform_k] (whole-unit uniform assignment)."""
    if uniform_k not in RUNG_TO_IDX:
        raise ValueError(f"uniform_k {uniform_k} not in RUNGS")
    ri = RUNG_TO_IDX[uniform_k]
    return float(per_group_mse[:, ri].sum().item())

def gain_vs_uniform(per_group_mse: torch.Tensor, projection: str, groups: int, budget_k: int, *, per_group_scale: bool = False) -> dict:
    """Compute equal-bytes gain for budget equivalent to uniform budget_k.

    Returns dict with uniform_mse, split_ks, split_mse, relative_gain.
    relative_gain = (uniform - split)/uniform  (positive = split wins)
    """
    from .byte_accounting import bytes_for_budget
    # Budget bytes = uniform bytes for K_budget
    budget_bytes = bytes_for_budget(projection, budget_k, per_group_scale=per_group_scale, groups=groups if per_group_scale else 1)
    # uniform covers G groups at same k
    u_mse = uniform_mse(per_group_mse, budget_k)
    best_ks, s_mse = solve_split(per_group_mse, projection, groups, budget_bytes, per_group_scale=per_group_scale)
    if best_ks is None:
        return {
            "budget_k": budget_k,
            "budget_bytes": budget_bytes,
            "uniform_k": budget_k,
            "uniform_mse": u_mse,
            "split_ks": None,
            "split_mse": None,
            "gain": 0.0,
            "feasible": False,
        }
    gain = (u_mse - s_mse) / max(u_mse, 1e-30)
    return {
        "budget_k": budget_k,
        "budget_bytes": budget_bytes,
        "uniform_k": budget_k,
        "uniform_mse": float(u_mse),
        "split_ks": best_ks,
        "split_mse": float(s_mse),
        "gain": float(gain),
        "feasible": True,
    }

def sweep_budgets(per_group_mse: torch.Tensor, projection: str, groups: int, budget_ks: List[int] = None, *, per_group_scale: bool = False) -> List[dict]:
    if budget_ks is None:
        budget_ks = [29, 30, 31, 32, 33, 34, 35, 36]
    out = []
    for bk in budget_ks:
        if bk not in RUNGS:
            continue
        out.append(gain_vs_uniform(per_group_mse, projection, groups, bk, per_group_scale=per_group_scale))
    return out
