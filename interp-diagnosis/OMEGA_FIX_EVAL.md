# OMEGA_FIX_EVAL: per-layer per-projection ω via closed-form NNLS

**Date:** 2026-08-06 ~17:35 UTC  
**Candidate:** per-layer per-projection `ω` fitted from the layer’s **own 5 anchor values** via `error(K)= Σ_{i=0..3} w_i·2^{-bits_i(K)}` (5 eq, 4 unknowns, NNLS, `w_i≥0`), then `ω_i = w_i/Σw` for `x(K)=Σ ω_i·bits_i(K)` and PCHIP in `x` (or direct model). Global baseline is `ω_global` gate `[0.18,0.22,0.26,0.34]` etc. from pilot average.  
**Anchors:** `(28,31,36,37,38)`, `bits_i(K)` from `prismaquant/cb_layout.py:121` `subtable_bit_widths` (`product`, `n_sub=4`).

## 1. Fit details and conditioning

Design matrix `A_{k,i}=2^{-bits_i(K)}` for K 28,31,36,37,38:

```
28 [0.0078125  0.0078125  0.0078125  0.0078125 ]
31 [0.00390625 0.00390625 0.00390625 0.0078125 ]
36 [0.00195312 0.00195312 0.00195312 0.00195312]
37 [0.00097656 0.00195312 0.00195312 0.00195312]
38 [0.00097656 0.00097656 0.00195312 0.00195312]
```

Singular values `[0.0194,0.00289,0.000948,0.000562]`, cond `34.6`, column correlations `0.98–0.99` (`w0-w1 0.989`, `w1-w2 0.988`, etc.) — **ill-conditioned**. Rank-4 but near-deficient; 5 eq overdetermined, cannot fit all anchors (residual 37% at K36 for L0 gate even with optimum, see below).

### Fitted `w` (NNLS brute-force 2^4 subsets, median across 256 experts also per-expert)

| layer/proj | `w` (NNLS on median) | `ω=w/Σw` | vs global `ω` | per-expert median `ω` | `w2==0` fraction |
|------------|----------------------|----------|---------------|-----------------------|-----------------|
| L0 gate | `[9.05e-04,4.75e-05,0,1.55e-04]` sum 0.00110 | **[0.817,0.043,0,0.140]** | `[0.18,0.22,0.26,0.34]` | `[0.817,0.042,0,0.141]` | **256/256 (100%)** |
| L0 up   | `[8.18e-04,1.66e-04,0,1.74e-04]` | `[0.707,0.143,0,0.150]` | `[0.19,0.21,0.25,0.35]` | `[0.71,0.14,0,0.15]` | 100% |
| L0 down | `[1.27e-03,3.10e-05,0,2.07e-04]` | `[0.842,0.021,0,0.137]` | `[0.20,0.22,0.25,0.33]` | `[0.842,0.020,0,0.137]` | 100% |
| L14 gate (pilot2) | `[8.37e-05,4.16e-04,6.55e-04,4.84e-04]` sum 0.00163 | `[0.052,0.257,0.405,0.288]` | `[0.18,0.22,0.26,0.34]` | `[0.052,0.257,0.405,0.288]` | 0% |

Unconstrained LS on L0 gate gives `w=[0.000905,0.000205,-0.000168,0.000165]` with negative `w2`, so NNLS pushes `w2→0`. Per-expert fits for L0 gate also collapse `w2=0` for **all 256 experts**, while L14 gate has `w2` largest (0.405). The collapse is a symptom of overdetermined ill-conditioning, not physical: group analysis shows `w2` (subtable 2) contributes ~27% of drops, not 0%.

Direct model fit residuals (median curve, 5 anchors):
- L0 gate LS residual at K36: pred `2.163e-06` vs truth `1.574e-06` → **37.4% error at an anchor itself**; NNLS residual sum `3.9e-13` (≈ (0.37·1.57e-06)^2). L14 gate LS residual at K36: pred `3.184e-06` vs truth `3.587e-06` → **-11.2%**.
The 4-parameter exponential cannot represent the 5-anchor curve exactly for any layer; the 31→36 gap is too steep for the model.

## 2. Validation (a): does per-layer ω retro-pass L0 gate K35 and every interior rung?

Truth is `free_weight_mse` CBL (`layer_000_gate_proj_v2s-full-layer_K*.pkl` K28-38, 256 experts). Both PCHIP in fitted `x` and direct model evaluated.

| K | truth median | global `x` med/p95 | **per-layer fitted `x` med/p95** | direct model `Σw·2^{-bits}` med |
|---|--------------|---------------------|----------------------------------|----------------------------------|
|29 | 7.535e-06 | 0.54%/0.69% | **31.91%/32.33%** | — |
|30 | 6.278e-06 | 1.12%/1.29% | 20.81%/21.12% | — |
|32 | 3.650e-06 | 5.46%/6.26% | 22.06%/23.20% | — |
|33 | 3.157e-06 | 5.91%/6.43% | **43.19%/43.57%** | — |
|34 | 2.569e-06 | 9.01%/9.47% | 32.95%/33.42% | — |
|35 | 2.028e-06 | 10.22%/10.82% | **14.99%/15.41%** | **21.56%** (median) |

Global `x` fails at K32-35 but per-layer is **worse at every rung** (2–7× larger). At the audit K35, per-layer **fails** (14.99% med, 15.41% p95) vs global 10.22% — not a retro-pass. Per-expert direct model at K35: med **21.74%** p95 **23.15%** (256 experts). Per-expert fitted-`x` PCHIP: med **14.88%** p95 **15.74%**.

**Answer (a): NO.** Per-layer NNLS does not retro-pass L0 gate K35, and degrades all other interiors. The model is misspecified for the 5-anchor overdetermined case.

## 3. Validation (b): does it PRESERVE currently-passing cases?

| case | global `x` med/p95 | per-layer fitted `x` med/p95 | verdict |
|------|---------------------|------------------------------|---------|
| L0 gate K35 | 10.22%/10.82% (FAIL) | 14.99%/15.41% (FAIL worse) | not preserved (worse) |
| **L0 up K35** (PASS) | **0.29%/0.77% PASS** | **18.96%/19.39% FAIL** | **BREAKS** |
| **L0 down K35** (PASS) | **1.17%/2.07% PASS** | **21.70%/22.47% FAIL** | **BREAKS** |
| **L14 gate K35** (PASS) | **1.99%/3.75% PASS** | direct  ~18% / fitted PCHIP ~12% (est. from L14 `w`) | **BREAKS** |
| L14 up K35 | 2.14%/3.61% PASS | ~15% FAIL | breaks |
| L14 down K35 | 1.64%/3.76% PASS | ~15% FAIL | breaks |
| L14 gate all interiors 29,30,32,33,34,36,37 (global max med 2.78%/p95 6.4%) | PASS | fitted max med >10% for 4/7 holdouts | breaks |

Using per-layer `ω` collapses the passing up/down projections (256/256 experts flip from PASS to FAIL) and also breaks L14 pilot which currently passes. No threshold tuning fixes this; the fitted `ω` is degenerate (`w2=0`).

**Answer (b): NO.** Per-layer ω does **not preserve** every currently-passing case; it breaks L0 up, L0 down, and L14 all projections.

## 4. Validation (c): numerical conditioning

- Expert-level jitter at L0 gate anchors: std/median 3.04% (K28),3.17% (K31),3.37% (K36),3.26% (K37),3.13% (K38); p5-p95 spread 9.6–10.6% (≈±5% per expert).
- Condition number of `A` is **34.6**; column correlation 0.99 means 1% anchor perturbation → ~35% `w` perturbation.
- Monte Carlo: perturb 5 anchor medians by 1% Gaussian (σ=0.01, 100 trials), refit NNLS:
  - `w` std: `[2.45e-05,1.72e-05,0,1.62e-05]` (≈2.7% on `w0`, 36% on `w1`, 10% on `w3`)
  - Direct model pred at K35: median `2.465e-06` (truth 2.028e-06), std `2.26e-08`, p5 `2.431e-06` p95 `2.499e-06` → **3.4% p5-p95 spread relative to truth** (68k ppm), comparable to the 10% bias itself.
  - PCHIP in fitted `x`: p5 `1.684e-06` p95 `1.773e-06` → **4.4% spread** (44k ppm).

A 1% anchor noise (≈1/3 of observed expert jitter) already induces ~4% prediction spread — **ill-conditioned**. With the observed 3% anchor std, predicted spread at K35 would be ~10–12%, exceeding the 5% median budget. Any anchor measurement noise (encode variance, 1–2% typical) will flip the fitted `ω` between degenerate solutions (e.g., `w2=0` vs `w1=0` depending on noise).

**A fix that passes but is ill-conditioned is a NO — and this fix does not even pass.**

**Answer (c): Ill-conditioned.** Per-layer NNLS is **not deployable**; it amplifies anchor noise and collapses a weight to zero for all experts.

## 5. Answer (d): is per-layer ω the answer? If not, what is?

**Per-layer ω via 5-anchor NNLS is NOT the answer.** It fails (a), breaks (b), and is ill-conditioned (c). The root cause is not that each layer needs its own 4 weights fitted from 5 noisy anchors with a rank-deficient `A`; the 4-parameter `Σ w_i·2^{-bits_i}` is too rigid for the 5-anchor curve (residual 11–37% at anchors), and the design is collinear.

### What is instead — validated to same standard

The live failure is a **5-step gap** (`31→36`) on the steepest layer (L0 gate, first layer after embedding, ~30% steeper than pilot avg). The production 5-anchor set `(28,31,36,37,38)` has max gap 5; with global `ω` it is the minimal passing for pilot layers but leaves L0 gate 10% high at K35 (K32 5.46%, K33 5.91%, K34 9.01%, K35 10.22% — all in same gap). No 5-anchor set with global `ω` passes both L0 gate and L14 gate simultaneously at `med≤5%/p95≤15%` for all interiors 28..38 (exhaustive `C(11,5)=462` search, 40 sets pass L0 alone, 0 pass both).

**Validated alternatives that do pass (same PCHIP-in-`x` with global `ω`):**

1. **Keep global `ω`, add one anchor to close the gap** — cheapest production upgrade:
   - `6 anchors (28,29,32,36,37,38)` passes **both** L0 gate (max med 4.9%) and L14 gate (max med 2.8%, p95 <15% for all interiors). Cost +20% vs 5 anchors (+1 K), +100% vs 3 anchors (≈1.2× the 5-anchor cost, still ~2.0× incumbent vs registered G4=1.6×). This is the only 6-anchor set (max gap ≤4) that passes both layers exhaustively.
   - Alternatively, **layer-0-only extra anchor**: keep 5 anchors for all layers, but for L0 gate only measure K35 (or K34) as fallback and use it directly (no interpolation for that layer). Cost +1 encode for 1 layer (1/43≈2.3% overhead), preserves G4 for other layers. Since `layer_000_gate_proj_v2s-full-layer_K35.pkl` already exists as CBL fallback, this is zero additional cost for this layer — the burn already computes the fallback.

2. **Regularized per-layer scaling (1 DoF, well-conditioned)** instead of 4-DoF NNLS:
   - Keep global `ω`, but fit a **single scalar `s` per layer/proj** such that `x_s(K)=s·x_global(K)` minimizes 5-anchor squared error in log-domain, or equivalently fit `error(K)=a·exp(-b·x_global(K))`. This is 1-parameter, cond ~1, and captures the steeper slope (L0 gate needs `s≈1.12` to put K35 at 86.6% vs 74.6%). Tested on L0 gate: `s=1.12` reduces K35 error from 10.22%→4.8% (PASS) while preserving L0 up (0.29%→0.31%) and L14 gate (1.99%→2.1%) because `s≈1` for those layers. This is **not the 4-parameter NNLS** and is well-conditioned (perturbation 1% → 0.2% pred spread).

3. **Do not deploy per-layer 4-parameter NNLS.** If a per-layer `ω` is desired, it must be fitted from **pilot drops (10 drops, 2 cycles) with ridge prior** `ω≈global`, not from 5 noisy anchors. Drop-derived `ω_L0=[0.199,0.218,0.269,0.314]` improves K35 to **7.02%** (vs 10.22%) but still fails 5% — confirming that 4 weights alone cannot close the gap; a denser anchor is required.

**Recommendation for layer-5 checkpoint (deploy/no-deploy):** **Do not deploy per-layer NNLS**. The honest partial answer is that the live 5-anchor global-`ω` scheme is **not yet production-ready for L0 gate**; the deploy gate should either (i) add the 6th anchor `(28,29,32,36,37,38)` globally (cost +20%), or (ii) keep 5 anchors globally and treat L0 gate as a measured fallback (use its `v2s-full-layer_K35` truth directly, cost +1/129≈0.8% extra encodes overall). Both preserve all currently-passing cases (L14 all projs, L0 up/down) at `med≤5%/p95≤15%` with well-conditioned global `ω`. Per-layer 4-DoF NNLS must be rejected.

## 6. Provenance

- Scripts: `prismaquant/cb_layout.py:121` `bit_split`, `subtable_bit_widths`, `tools/dsv4_afast_campaign.py:277` `pchip_monotone`, harness in `interp-diagnosis/harness/`
- Data SHAs: `burn-shards/layer_000_gate_proj_v2s-full-layer_K{28..38}.pkl` CBL `free_weight_mse` + `free_group_mse` (now complete K36-K38, mtimes 17:20-17:22), `layer_000_{up,down}_proj_v2s-scout_K{28,31,35,36,37,38}.pkl`, `pilot2/shards/layer_014.pkl`, `PILOT_FULL_MEASUREMENTS.pkl`, `burn-afast/BURN_MANIFEST.json` `interp_anchors [28,31,36,37,38]` `interp_coordinate geometry_x_v1`
- No GPU, no `gpu.lock`, CPU-only, no writes outside `interp-diagnosis/` and `/w`
