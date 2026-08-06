# CAMPAIGN_V2_PROTOCOL: measurement architecture for next campaign

**Date:** 2026-08-06 22:05 UTC  
**Replaces:** 5-anchor `pchip` in `x(K)` (FIX_PROPOSAL) and 3-anchor PCHIP legacy (28,33,38).  
**Scope:** priced domain K28–38 only (K39–48 demand-extension, never interpolated blind).  
**Implementer needs no other context** beyond this file + `harness/law_*.py` + `prismaquant/cb_layout.py`.

## 1. Law

```
G(K) = exp(a0 + a1·K + φ_{K%4})                         (Level-1 global)
  a0,a1 = log-scale/tilt, φ0=0, φ1–3 = family offsets (gate/up/down per-projection)
  ρ = exp(a1)  (≈0.851–0.853, see ERROR_LAW.md §1.1)

L(K) = G(K)·exp(a + b·K)                                 (Level-2 per-layer)
  (a,b) fitted from n anchors' medians in log space
  prediction for any K in 28..38, any expert's median: (5) in law_model.py
```

Per-expert residual handled by existing backstop (`v2s-full-layer` fallback, per-expert `winning_arm` free vs embed, `epsilon 1e-12`); law only predicts medians.

Global coefficients (fit on L14 medians K28–38, log-LS, 95% CI ±0.04/±0.0012/±0.01):

```
gate: [-6.8249, -0.15897, 0.03594, 0.05849, 0.06637]
up:   [-6.7873, -0.15964, 0.04179, 0.06700, 0.06039]
down: [-6.4734, -0.16092, 0.04078, 0.06231, 0.05702]
```

Code: `harness/law_model.py:global_G`, `log_G`, `fit_level2`, `predict`.

## 2. Which anchors at which rungs (principled, family coverage)

Residue families `r=K%4` (0..3), staircase period 4 (DIAGNOSIS Table).  Law's `φ_r` already encodes family, but coverage still matters for `b` tilt estimation.

| n | anchors (K) | families r | max gap | rationale vs search | expected median max (gate L0) | cost vs legacy 3-anch |
|---|-------------|------------|---------|---------------------|-------------------------------|----------------------|
|2| **28, 38** | 0,2 |10| maximizes span, determines `a,b` exactly, minimal measurement; 9 holdouts; family via G | 3.5% | 0.66× (save 33%) |
|3| **28, 33, 38** | 0,1,2 |5| minimal covering 3 families, matches legacy 3-anch for apples-to-apples, 1 DoF leftover for validation; missing r3 inferred via φ3 | 3.0% | 1.00× (same) |
|4| **28, 33, 35, 38** | **0,1,3,2** |5| **minimal full-family** (only 8 of C(11,4)=330 cover all 4 with max gap≤5); recommended production; overdetermined (2 DoF) → jitter averaging | 3.3% | 1.33× vs 3-anch, 0.80× vs 5-anch geometry, 0.66× vs 6-rung scout |

Legacy 3-anch (28,33,38) is retained as n=3 option; n=4 is the cost of one extra K vs legacy but still 0.67× the FIX_PROPOSAL 5-anch cost and 0.66× the current 6-rung scout (28,31,35,36,37,38).

**Exhaustive search confirms:** among `C(11,5)=462` 5-anch sets with 28,38 inclusive, 0 pass both L14 and L0 gate with global `x(K)` PCHIP (GATE_DIAGNOSIS §5); with law, n=4 full-family set passes.

## 3. Fitting procedure (per layer×projection, CPU <1µs)

Given `n` measured anchor medians `{y_k, k∈anchors}` (median over 256 experts, `free_weight_mse` for CBL, `selected_weight_mse` for pure-incumbent where `free==selected`):

```
rhs_k = log y_k - log G(K)                # vector length n
[a,b] = lsq( [1 K] , rhs )                # 2×n LS, cond≈1
# n=2: exact; n>2: overdetermined, closed form via numpy.linalg.lstsq
```

Predict any `Kq` in 28..38:

```
ŷ_q = G(Kq)·exp(a + b·Kq)
```

For per-expert backstop decision (existing), compute per-expert `a_i,b_i` similarly if per-expert law desired, but default is median law + backstop: if per-expert `|ŷ - y_true|>tolerance` (backstop 25% for `v2s`), that slice ships measured row (see `dsv4_afast_burn.py:_acceptance`).

No new kernels, no book retraining, no export change.  `G(K)` is pure table lookup `exp(a0+a1K+φ_r)` (<1µs, no GPU).

**Pseudocode for burn:**

```python
from harness.law_model import global_G, fit_level2, predict
# Level-1 already fixed per projection (import)
anchors = (28,33,35,38)  # or per table
anchor_medians = {k: np.median(free_weight_mse[k]) for k in anchors}  # 256 array -> median
a,b = fit_level2(anchors, anchor_medians, proj)  # proj="gate_proj"
for Kq in range(28,39):
    if Kq in anchors: y = anchor_medians[Kq]
    else: y = predict(proj, Kq, a,b)  # median prediction
```

## 4. Detection rule for measure-don't-model layers (fail-closed)

Prior on Level-2 from pilot (L14, 256 experts, bootstrap):

```
a ∼ 0.0 ±0.15 (σ),  b ∼ -0.001 ±0.002
```

Flag if:

```
|a| > 0.50   (≈3.3σ)  OR  |b| > 0.015 (≈7σ)
OR  audit K35 (optional, 1 extra encode) residue >5% vs law prediction
```

**Action:** full-measure that layer×projection (21 rungs K28–48) or at minimum add K35 as extra anchor and refit (cost +1 rung, 2.3% overhead for that layer, 0.8% overall for 1 proj).  In backlog, L0 gate has `a=1.22` → flagged (would have been flagged even though law already passes median 3.5%; conservative).  L0 down K36 marginal 6–7% also flagged via audit.

A detection is an **acceptable answer**; silent 10% median failure is not.  If no flag, use law.

## 5. Expected measurement cost vs today

| scheme | anchors per layer×proj | encodes per 43-layer model (3 projs) | wall time (pilot timing 5.1s free +4.6s arms per K, L21 free 25.99s+22.87s per 5K) | vs legacy 3-anch (98k encodes) | vs FIX 5-anch (164k) | vs current 6-rung scout (197k) |
|--------|------------------------|--------------------------------------|------------------------------------------------|----------------|----------------|----------------|
| legacy 3-anch PCHIP | 3 (28,33,38) | 98k | baseline | 1.00× | 0.60× | 0.50× |
| law n=2 | 2 (28,38) | 66k | –33% | 0.67× | 0.40× | 0.34× |
| law n=3 | 3 (28,33,38) | 98k | same as legacy | 1.00× | 0.60× | 0.50× |
| **law n=4 (recommended)** | 4 (28,33,35,38) | 132k | +34% vs legacy (+19.4s/layer×43≈14min) | 1.34× | 0.80× | **0.67×** |
| FIX 5-anch x(K) |5 (28,31,36,37,38)|164k|+67%|1.66×|1.00×|0.83×|
| 6-anch scout (today) |6 (28,31,35,36,37,38)|197k|+100%|2.00×|1.20×|1.00×|

`TARGET_BITS=4.75` parity: 6-anch scout is current burn; law n=4 saves 65k encodes (33%) vs scout and 32k vs FIX, while passing L0 gate (scout fails K36 median 6–10% with PCHIP).  If detection flags 1/43 layers for full measure (+21 vs 4), overhead is +17 encodes (0.13×) still <6-anch.

No new GPU kernels, no `gpu.lock` (CPU harness), no change to export/serving.

## 6. Validation & ship gates (what operator re-derives)

Before live burn, run:

```bash
PYTHONPATH=/w python3 interp-diagnosis/harness/law_validate.py
PYTHONPATH=/w/interp-diagnosis/harness pytest law_validate.py law_weight_stats.py -v
```

Must show on ALL banked full ladders (L14 both free/selected, L21, L0-gate CBL) for K28..38 leave-out:

- n=2 median max ≤6.5% (≤6% typical), n=3,4 median max ≤5.5% (down L0 K36 allowed 8% with detection)
- per-expert p95 ≤15% for L0 (homogeneous), L14/L21 per-expert 60% is heterogeneity floor (not a failure; backstop handles)
- conditioning 1% anchor jitter → K35 spread <2.5% (n=2) <1.8% (n=4)

If any median >5.5% (or >6.5% for n=2) without detection, add anchor (n→n+1) or flag for full measure.

## 7. Rollback

If law fails pilot re-derive, fallback is FIX Proposal 5-anchor `x(K)` or 6-anchor scout (both already validated CPU).  Law does not change live burn until `ARCHITECTURE.md` and provenance block are re-stamped per AGENTS.md:10 and tests pass.

## 8. References

- `prismaquant/cb_layout.py:121` `bit_split`, `143` `subtable_bit_widths`
- `tools/dsv4_afast_campaign.py:277` `pchip_monotone` (legacy)
- `harness/law_model.py`, `law_validate.py`, `law_weight_stats.py`
- `ERROR_LAW.md` (full tables, weight-stats R²/LOO, jitter)
- `DIAGNOSIS.md`, `GATE_DIAGNOSIS.md`, `OMEGA_FIX_EVAL.md`

