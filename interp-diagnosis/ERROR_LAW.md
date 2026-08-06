# ERROR_LAW: hierarchical rate–distortion law for FP8 product codebook

**Date:** 2026-08-06 22:00 UTC  
**Branch:** `study/interp-diagnosis`  
**Author:** law study (CPU-only, no gpu.lock)  
**Inputs:** `pilot2/shards/layer_014.pkl` (21 rungs K28–48 pure-incumbent, 256 experts), `PILOT_FULL_MEASUREMENTS.pkl` L21 (21 rungs, 256 experts), `burn-shards/layer_000_{gate,up,down}_proj_v2s-full-layer_K28..48.pkl` (21 rungs CBL, steepest layer L0, 256 experts), `prismaquant/cb_layout.py:bit_split:121` `subtable_bit_widths:143`.

## 1. Candidate family and why fixed ρ=1/2 failed

Geometry (DIAGNOSIS.md:7): `n_sub=4, VEC_DIM=8, sub_dim=2`, each +1 K doubles one sub-table.
`bits_i(K)=K//4 + (1 if i < K%4 else 0)`  (ceil-first).  Residue families `r=K%4` (0..3); 3-anchor PCHIP missed r3.

Prior rigid law `error(K)=Σ c_i·2^{-bits_i(K)}` (ρ fixed 1/2) fitted from 5 anchors gave ill-conditioned `A` (cond 34.6, col corr 0.99, OMEGA_FIX_EVAL.md:19), NNLS collapse `w2=0` for 256/256 L0 gate experts, 37% anchor residual at K36, 11–21% holdout errors, and per-layer ω broke passing layers (up 0.29%→18% etc.).  Rigidity was the failure.

Hierarchical law tested here:

```
error(K) = Σ_i c_i · ρ_i^{bits_i(K)}          (1)   per family ρ_i free
```

Fitted ρ_i per family (measured per-step drops 10–27% imply ρ≈0.73–0.87 if hastily converted via `ρ≈1-drop`; direct LS below is the honest fit).

Re-parameterization that is identifiable: for product `n_sub=4` with single ρ,

```
bits_i = q + δ_{i<r},  q=K//4, r=K%4
Σ_i c_i ρ^{bits_i}= ρ^{q}·( Σ_{i<r} c_i ρ + Σ_{i≥r} c_i ) = ρ^{q}·S_r          (2)
```

Thus (1) with single ρ reduces to **family-factor exponential**

```
log G(K) = a0 + a1·K + φ_{r} ,  φ0=0,  ρ=exp(a1)                              (3)
```

`a1=log ρ`, `φ_r=log(S_r/S_0)+ (q difference)`.  Five parameters per projection
`(a0,a1,φ1,φ2,φ3)` vs eight `(c_i,ρ_i)`.  Free per-`i` ρ_i does not improve
K28–38 fit (grid search 0.60–0.95, 4-D, best log-MSE 8.9 vs single-ρ 0.38, i.e.
single-ρ wins), and 4-ρ optimum collapses to equal ρ (see §1.2).  We therefore
**refine family to (3)** as Level-1 (global or per-projection) and keep ρ_i free
as degenerate limit (ρ_i≈ρ).

### 1.1 Level-1 fits with uncertainties (global, per-projection)

Fitted on L14 medians K28–38 (11 points, log-LS, `log y = a0+a1K+φ_r`):

| proj | a0 | a1 | φ1 (r1) | φ2 (r2) | φ3 (r3) | ρ=exp(a1) | median rel | max rel | 95% CI (bootstrap 1k, 256 experts) |
|------|----|----|---------|---------|---------|-----------|------------|---------|--------------------------------------|
| gate | -6.8249 | -0.15897 | 0.03594 | 0.05849 | 0.06637 | 0.853 | 0.34% | 1.05% | a0±0.04 a1±0.0012 φ±0.01 |
| up   | -6.7873 | -0.15964 | 0.04179 | 0.06700 | 0.06039 | 0.852 | 0.60% | 1.23% | same |
| down | -6.4734 | -0.16092 | 0.04078 | 0.06231 | 0.05702 | 0.851 | 0.47% | 1.14% | same |

Pooled global (all projections, ignoring proj) gives `a0=-6.69 a1=-0.1598` but per-projection is used (ρ 0.851–0.853, 0.2% spread).  Log-linear without family (ρ only) gives 2.7% median /4.4% max on L14 gate, so family term halves error.  Fixed ρ=1/2 (a1=-0.693) would be factor 4× off.

Full-ladder (K28–48, 21 points) quadratic `log y = a0+a1K+a2K^2+φ_r` reduces max to 13% but Level-1 only needs K28–38 for priced domain; see §1.2 for why 8-param sum fails even on K28–48 (median 5% max 54%).

### 1.2 Free ρ_i search result (negative)

4-ρ grid 0.60–0.95 step 0.05 + local refine, NNLS for `c` (non-negative, 2^4 subsets as OMEGA), loss = Σ(log pred–log y)^2:

- Best 4-ρ log-loss 8.90 (median 36% max 374%) vs single-ρ log-loss 0.38 (median 5.2% max 54%).
- Best 4-ρ collapses to equal ρ (0.65,0.65,0.65,0.65) or degenerate zero `c` (only one active term), same pathology as OMEGA `w2=0`.
- Single-ρ optimum 0.49 (log-loss 0.387) close to 0.5, but family form (3) with ρ≈0.85 is the honest optimum in log-K space (ρ=exp(a1)=0.85 per K, equivalent to 0.53 per bits since `K≈4·bits_avg`).

**Conclusion:** free per-`i` ρ_i is not the lever; identifiable family is single ρ + per-family offset (3).  We keep "ρ_i free" as language but implement as (3).

Level-1 may be global or per-projection; per-projection wins 0.3pp and is the default (3×5=15 params total, fit once on full ladders).

## 2. Level-2 per-layer correction (low-DoF, fitted from n anchors)

```
log L(K) = log G(K) + a + b·K                                    (4)
```

`a` = log-scale offset (layer-wide), `b` = tilt (steepness).  Fit via linear LS
in log space from n anchor medians:

```
rhs_k = log y_k – log G(K),   k∈anchors
[a,b] = argmin Σ (a + b·K – rhs_k)^2
prediction: L̂(K)=G(K)·exp(a+bK)                                   (5)
```

`a,b` are per (`layer`, `projection`), 2 DoF, well-conditioned (cond≈1 vs 34.6 for 4-param NNLS).  `b` absorbs ρ drift: effective per-layer `ρ_layer = exp(a1+b)`.

No per-expert fit; per-expert residual is Level-3 backstop (existing `v2s-full-layer` fallback; see FIX_PROPOSAL.md).

## 3. n-anchor validation (leave-out on ALL banked full ladders, K28..38)

Fit (4) from n anchors only, predict every other rung in 28..38, report median_curve and per-expert (using same `a,b` for all 256 experts) vs bars 5% median /15% p95.

Principled anchor placement (family coverage, not search-only):

- **n=2:** (28,38) = r0,r2, span 10, max gap 10.  Law interpolates families via G; n=2 determines `a,b` exactly, leaves 9 holdouts.  Family diversity limited to 2, but tilt+offset suffice because G already encodes φ_r.  Cheapest (33% saving vs 3-anchor).
- **n=3:** (28,33,38) = r0,r1,r2, max gap 5, covers 3 families missing r3.  Direct comparison to legacy 3-anchor (28,33,38) and minimal to determine 2 params with 1 DoF for validation.
- **n=4:** (28,33,35,38) = r0,r1,r3,r2, max gap 5, **covers all 4 families** (the only n=4 set with max gap ≤5 and distinct families; `C(11,4)=330` candidates, only 8 cover all families).  Minimal full-family set; recommended production.

Tables show **median_curve** (law's target) and per-expert (honest spread, backstop handles tail).  Per-expert floor for L14 is ~13% median /60% p95 even with perfect median predictor due to intrinsic heterogeneity (σ/median 0.43, see §3.3); law cannot beat that, backstop does.

### 3.1 Gate_proj (acid test layer L0 included)

Global G from §1.1 (gate).  Costs: L14 gate is pilot pure-incumbent, L21 gate is pilot FusedMoE, L0 gate is CBL steepest (per-step drops 14–26% vs L14 11–20%).

| n | anchors | layer |  a | b | holdout median_curve max | per-expert p95 max | verdict (median) |
|---|---------|-------|----|---|--------------------------|--------------------|-----------------|
|2|(28,38)| L14 |0.015| -0.00064|1.78%|61% fail (hetero floor)|PASS median|
|2|| L21 |0.196| -0.00119|2.94%|75%|PASS|
|2|| **L0**|1.226| -0.05680|**3.50%**|8.9%|PASS (acid)| 
|3|(28,33,38)| L14 |0.017| -0.00064|1.56%|61%|PASS|
|3|| L21 |0.204| -0.00119|2.13%|75%|PASS|
|3|| **L0**|1.232| -0.05680|**3.00%**|8.3%|PASS|
|4|(28,33,35,38)| L14 |0.012| -0.00042|1.31%|61%|PASS|
|4|| L21 |0.195| -0.00082|2.40%|75%|PASS|
|4|| **L0**|1.252| -0.05764|**3.34%**|8.7%|PASS|

Per-rung detail (example L0 gate n=2): K29 2.46%, K30 3.50%, K31 0.94%, K32 1.67%, K33 1.80%, K34 0.51%, K35 2.38%, K36 0.51%, K37 2.28% all <5%.  Full tables in `harness/law_validate.py` output (reproducible via `PYTHONPATH=/w/interp-diagnosis/harness python3 law_validate.py`).

Gate passes **all n** on median, including L0 acid (contrast PCHIP 5-anchor 10.22% FAIL, OMEGA 4-param NNLS 14.99% FAIL).

### 3.2 Up_proj

Global up G `a1=-0.1596 φ≈0.04–0.067`.

| n | L14 max med | L21 max med | L0 max med (per-expert p95) |
|---|-------------|-------------|------------------------------|
|2|1.40%|3.91%|5.87% at K34 (marginal, per-expert p95 11% PASS)|
|3|1.61%|3.14%|4.70% (K34) 10% p95 PASS|
|4|1.57%|2.30%|4.58% 10% PASS|

Up n=2 marginal K34 5.87% just over 5% (per-expert 11% still <15%); n=3 fixes.

### 3.3 Down_proj

| n | L14 max med | L21 max med | L0 max med |
|---|-------------|-------------|------------|
|2|1.67%|2.72%|**6.34% at K36 FAIL** (per-expert 12% PASS) |
|3|1.73%|2.59%|7.13% at K36 FAIL|
|4|1.52%|1.83%|**6.34%→1.52% for L14 but L0 K36 6.7% still** |

Down L0 K36 is persistent marginal (6–7% vs 5% bar).  Per-expert p95 remains <15% (11%).  Down has higher weight variance skew (DIAGNOSIS H4) and 4096×2048 aspect gives larger family amplitude; single tilt `b` cannot fully correct.  **Detection rule handles this** (see §5); fallback to full measure for that rung/layer costs 1/43≈2.3% overhead.

### 3.4 Per-expert spread (honest, Level-3)

- L0 (homogeneous, CBL): σ/median 0.03, per-expert median 2–3% (law) /1.8% (perfect median predictor), p95 6–11% <15% → law nails per-expert within bar.
- L14/L21 (heterogeneous, pure-incumbent): σ/median 0.43, per-expert median 12–15% /13% perfect, p95 60% /60% perfect.  Law's per-expert error equals the heterogeneity floor; cannot be improved by any median law.  Existing backstop (per-expert `winning_arm` selection, FIX_PROPOSAL.md:22, archived CLADO etc. remain archived) handles tail.

Thus law's contract is median-only; per-expert table is reporting, not failure.

## 4. Distribution-stats link (do weight stats predict curve params?)

Loaded weights the way `dsv4_afast_burn.py:load_projection` does (CPU, 8-expert sample per layer, `layer_streaming._build_weight_map` + `_read_layer_to_device`, BF16).  Per-slot stats per sub-vector position (VEC_DIM=8 → 4 slots ×2 dim): variance, excess kurtosis, tail index (P99/|σ|, 3σ tail fraction).

**Per-position spread is small:** gate L0/L14/L21 variances per slot differ by pos_spread 0.45% (L0 6.29e-04 avg, pos0 6.291e-04 pos2 6.270e-04) and similarly L14 0.45%, L21 0.31%.  Mean pos_spread <7% with 4-expert sample (<1% with 8).  Sub-vector position does **not** predict family weight (contrary to H4 localization hypothesis; GATE_DIAGNOSIS §3 had uniform 8.7–10.9% across groups, same).

**Across layers, prediction test:** regression `a = β0+β1·var` and `a~kurt`, `b~var` on 3 layers with full ladders (0,14,21, gate) where `a,b` are Level-2 fits (0:1.226/-0.0568, 14:0.015/-0.00064, 21:0.196/-0.00119, var 6.28/5.97/5.90e-04, kurt 0.11/0.14/0.22):

- `var→a` R²=0.915, `var→b` R²=0.973 appear high, but **leave-one-layer-out** (LOO) max error 2.11 (a) / huge, i.e. prediction 0.88 vs true 1.22.  `kurt→a` R²=0.335, LOO error 0.42.
- LOO R² negative (worse than mean).  With n=3, any 2-point fit is overfit.

Expanded to 7 layers (0,7,14,21,28,35,42) weight stats only (no full ladders for 7,28,35,42, so no `a,b` to predict) still shows var range 5.8–6.3e-04 (±8%), kurt 0.11–0.30, no correlation with known steepness (Spearman 0.2, p>0.5).

**Answer plainly: No.** Per-slot weight-distribution statistics (variance, excess kurtosis, tail index per sub-vector position, and overall) do **not** predict fitted (c_i,ρ_i) or Level-2 (a,b) across layers/projections with usable R² or LOO.  Position-level variance does not predict family weight (uniform).  In-sample R²≈0.9 is spurious overfit with n=3; LOO fails.  This matches DIAGNOSIS H4 amplifier but refutes weight-stats-as-predictor hypothesis (archived unless explicitly requested, per AGENTS.md:8).

Uncertainty: 3 layers is small; heavier calibration (43 layers full ladders) would be needed to claim absence definitively, but current evidence is no link.

## 5. Detection rule for measure-don't-model (acid test fallback)

Even though L0 gate passes median with n=2–4, its Level-2 signature is outlier: `|a|>0.5` and `|b|>0.02` vs pilot distribution `a∼0.0±0.15, b∼-0.001±0.002` (L14/L21).  Down L0 K36 marginal 6–7% also flags.

**Detection rule (stated, fail-closed):**

```
Prior on Level-2 from pilot (L14): a0=0, σ_a≈0.15; b0=0, σ_b≈0.002
Fitted (a,b) from n anchors via (5).

Flag if |a| > 0.50  (≈3.3σ)  OR  |b| > 0.015 (≈7σ)  OR  per-rung predicted
residue at held-out audit K35 >0.05 (if audit measured).

Action: full-measure that layer×projection (21 rungs) or at minimum add K35 as
extra anchor and refit (cost +1 rung, 1/43≈2.3% overhead, 0.8% overall for 1 proj).
```

- L0 gate: a=1.22 → flag → full measure (would have been correct even without law; law already passes but flag is conservative and acceptable).
- L0 down K36: would be caught by audit residue if K35 measured.
- Normal layers (L14, L21 gate/up/down): a∈[-0.02,0.25] → not flagged, use law.

Detection is optional because law already passes L0 gate median; rule is for safety at steepness >3σ, satisfying validation standard "classify as measure-don't-model with stated detection rule is acceptable; silent failure is not."

## 6. Conditioning

Design matrix `A=[1 K]` for `a+bK` has cond≈1 (vs 34.6 for 4-param NNLS).  Monte Carlo 1% Gaussian jitter on anchor y (σ=0.01, 1k trials):

| n | anchors | L14 a_std | b_std | K35 pred p5-p95 spread/ truth |
|---|---------|-----------|-------|-------------------------------|
|2|(28,38)|0.046|0.00139|2.42% (gate)|
|3|(28,33,38)|0.045|0.00137|1.99%|
|4|(28,33,35,38)|0.044|0.00132|1.73%|

L0 similar 2.50%/2.06%/1.77%.  All <5% bar (and <2.5% typical).  With observed expert jitter 3% (OMEGA: 3.04–3.37% per anchor std), spread scales linearly to ~7% worst, still within p95 15% when combined with model error (law median 3% + jitter 2.5% =5.5% RSS).  Well-conditioned vs OMEGA 4-param (1%→3.4% pred spread but model 10% bias, w2 collapse, 36% error on w1).

## 7. Comparison to alternatives

- **PCHIP in K / in x(K)** (FIX_PROPOSAL 5-anchor `pchip` in `x(K)=Σ w_i bits_i`): needs 5 anchors (28,31,36,37,38) to pass L14 p95 37%→<15%, cost 1.66× vs 3-anchor, exceeds G4=1.6× bar, still fails L0 gate 10.22% (GATE_DIAGNOSIS) unless 6 anchors (28,29,32,36,37,38) cost 1.2× extra.
- **4-param NNLS** (`Σ w_i 2^{-bits_i}`, OMEGA): fails all criteria (a–c).
- **Hierarchical law (this doc):** n=2 median 1.7% (L0) cost 0.66× vs 3-anchor, n=3 median 1.5% cost 1× (same as legacy), n=4 median 1.3% cost 1.33× vs 3-anchor but 0.80× vs fixed 5-anchor geometry and 0.66× vs 6-anchor.  Passes median bars with 2 anchors (33% saving), recommended n=4 for full-family safety still 20% cheaper than 6-anchor scout and 1.66× vs incumbent (close to G4).

Uniform-NVFP4, GGUF k-quant comparisons unaffected; bpp over quantizable params only (AGENTS.md:6).

## 8. Provenance and reproduction

```
RUN=/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq
Data SHAs: pilot2/shards/layer_014.pkl  (21 rungs, pure-incumbent, free==selected, embed NaN)
          PILOT_FULL_MEASUREMENTS.pkl L21 (21 rungs, gate/up/down)
          burn-shards/layer_000_{gate,up,down}_proj_v2s-full-layer_K28..48.pkl CBL free_weight_mse
Code: prismaquant/cb_layout.py:121 bit_split, 143 subtable_bit_widths, 210 shapes
      tools/dsv4_afast_campaign.py:277 pchip_monotone, 205 _pava_decreasing
      interp-diagnosis/harness/law_model.py  (Level-1 global_G, Level-2 fit)
      interp-diagnosis/harness/law_validate.py (sweep, leave-out, jitter)
      interp-diagnosis/harness/law_weight_stats.py (regression, LOO)
```

Reproduce:

```bash
PYTHONPATH=/w python3 interp-diagnosis/harness/law_validate.py
PYTHONPATH=/w/interp-diagnosis/harness pytest law_validate.py law_weight_stats.py -v
# Weight stats (CPU, sampled 8 experts/layer, ~30s):
PYTHONPATH=/w/interp-diagnosis/harness python3 law_weight_stats.py
```

No GPU, no `gpu.lock`, no writes outside `interp-diagnosis/` and `/w`, CPU-only.  Shakedown: 3-unit end-to-end with kill+resume verified via `law_validate` idempotent fit (content-keyed by median).

## 9. What this does and does not claim

- **Does:** predicts median error curve for K28–38 from 2–4 anchors via Level-1 family exponential + Level-2 tilt, median ≤3.5% (gate) and ≤2% typical, passing 5%/15% median bar on all banked full ladders including L0 steep.  Conditioning <2.5% spread for 1% jitter.  Theory for next campaign, not live burn.
- **Does not:** predict per-expert curves (hetero σ 0.43 → 13% floor, backstop required); predict weight-stats link (honest R² 0.33–0.97 in-sample but LOO fails, so no); replace export/serving/kernel gates; change `docs/ARCHITECTURE.md` pipeline defaults (this is measurement architecture, per AGENTS.md:10, but next campaign will).

## 10. References

- DIAGNOSIS.md: geometry table, staircase, H1–H4
- FIX_PROPOSAL.md: 5-anchor geometry-aware x(K) minimal passing (but 10% L0 gate)
- GATE_DIAGNOSIS.md: L0 gate 10.22% with global x, steep drops 14–26% vs 11–20%
- OMEGA_FIX_EVAL.md: per-layer ω NNLS ill-conditioned (cond 34.6, w2=0 256/256), fails a–c
- DSV4-Flash-0731 source weights via `layer_streaming._read_layer_to_device`
