# GATE_DIAGNOSIS: why L0 gate_proj fails geometry-aware interpolation

**Date:** 2026-08-06 ~17:30 UTC  
**Anchors live:** `(28,31,36,37,38)` PCHIP in `x(K)=Σ ω_i·bits_i(K)` with **global** `ω` from pilot-average drops (gate `[0.18,0.22,0.26,0.34]`, up `[0.19,0.21,0.25,0.35]`, down `[0.20,0.22,0.25,0.33]`) — `prismaquant/cb_layout.py:121` `bit_split`, `tools/dsv4_afast_campaign.py:277` `pchip_monotone`.  
**Inputs:** `burn-shards/layer_000_gate_proj_v2s-full-layer_K{28..38}.pkl` (now complete K28-K38 CBL, `cbl_poolb` `free_weight_mse`, `free_group_mse` {2,4,8}), `burn-shards/layer_000_{up,down}_proj_v2s-scout_K{28,31,35,36,37,38}.pkl` (CBL), `pilot2/shards/layer_014.pkl` (21-rung pure-incumbent), `PILOT_FULL_MEASUREMENTS.pkl` L21, `BURN_MANIFEST.json` `interp_coordinate=geometry_x_v1`.

## 1. Live result reproduced (audit K35, 256 experts, `free_weight_mse`)

PCHIP in global `x` through 5 anchors, predict K35 vs measured CBL truth:

| proj | truth median | pred median | med err | p95 err | sign pred>truth | verdict |
|------|--------------|-------------|---------|---------|-----------------|---------|
| L0 gate | 2.028e-06 | 2.237e-06 | **10.22%** | **10.82%** | 256/256 (100%) tight one-sided | **FAIL** |
| L0 up   | 2.220e-06 | 2.224e-06 | 0.29% | 0.77% | 256/256 | PASS |
| L0 down | 2.732e-06 | 2.703e-06 | 1.17% | 2.07% | pred<truth 100% | PASS |
| L14 gate (pilot2, same anchors/x) | — | — | 1.99% | 3.75% | — | PASS |

Reproduced exactly the operator report (`CBL_AUDIT_L00_*`). L0 down passes with opposite sign (pred low), L0 up passes essentially exact, L14 gate passes. Deviation is **layer-local to L0 gate**.

All 256 L0 gate experts share the signature (8.7% min, 11.3% max), same class as the original 3-anchor bug but now isolated to one projection — staircase geometry with global weights misaligned.

## 2. Full CBL ladder (L0 gate, median across 256 experts)

`burn-shards/layer_000_gate_proj_v2s-full-layer_K*.pkl` `free_weight_mse` (K36-K38 now landing, scout until then identical):

| K | median | per-step drop (median) | doubled subtable | bits |
|---|--------|------------------------|------------------|------|
|28 | 8.798e-06 | — | — | 7,7,7,7 |
|29 | 7.535e-06 | **14.35%** | 0 | 8,7,7,7 |
|30 | 6.278e-06 | 16.68% | 1 | 8,8,7,7 |
|31 | 4.968e-06 | 20.86% | 2 | 8,8,8,7 |
|32 | 3.650e-06 | **26.52%** | 3 | 8,8,8,8 |
|33 | 3.157e-06 | 13.51% | 0 | 9,8,8,8 |
|34 | 2.569e-06 | 18.64% | 1 | 9,9,8,8 |
|35 | 2.028e-06 | 21.06% | 2 | 9,9,9,8 |
|36 | 1.574e-06 | 22.39% | 3 | 9,9,9,9 |
|37 | 1.279e-06 | 18.74% | 0 |10,9,9,9 |
|38 | 1.078e-06 | 15.69% | 1 |10,10,9,9 |

Per-expert median drops are tight (p90-p10 <0.5pp per step), monotone decreasing for all 256 experts (K34 worse than K33 for 0/256).

**Comparison L14 gate (pilot2) same steps:** 11.1%,13.2%,15.9%,19.7%,11.1%,12.8%,14.9%,19.3%,11.3%,13.2% — same 4-cycle ordering (3 largest) but **uniformly smaller** (avg 15% vs L0 avg 19%). Pilot-average drops (PILOT_FULL_MEASUREMENTS L21) are 10.8%,12.8%,15.5%,19.7%,10.5%,12.3%,14.2%,18.3%,15.8%,13.5% — also smaller than L0. L0 gate is **~30% steeper per step** than pilot average.

Drop-derived `ω` (avg drop per family normalized):
- L0 gate: `[0.199,0.218,0.269,0.314]` (w3=0.314 vs global 0.34, w0=0.199 vs 0.18)
- L14 gate: `[0.189,0.221,0.260,0.330]` ≈ global `[0.18,0.22,0.26,0.34]`
- Pilot avg: `[0.185,0.220,0.255,0.340]` ≈ global.

L0’s family weight is not wildly different, but its **absolute drops are larger**, meaning the `x` coordinate stretched by global `ω` underestimates the total decay between K31 and K36.

Total decay 31→36: L0 `(4.968-1.574)/4.968=68.3%`, L14 `(8.38-3.58)/8.38=57%`, pilot avg ~50%. PCHIP in global `x` puts K35 at ` (x35-x31)/(x36-x31)= (8.66-7.66)/(9.0-7.66)=74.6%` of the interval, but L0 truth has completed ` (4.968-2.028)/(4.968-1.574)=86.6%` by K35 — pred lags truth, hence pred high by 10%.

## 3. Is the anomaly localized to column groups? (free_group_mse)

`free_group_mse` {2:[256,2],4:[256,4],8:[256,8]} per-expert per-block MSE, mean across groups equals overall `free_weight_mse` exactly (max diff 2.8e-13, 0.00%). Per-group PCHIP error at K35 (global `x`, 256 experts):

- ng=2: g0 10.09%/10.97%, g1 10.31%/11.03%
- ng=4: g0 9.65%/10.82%, g1 10.60%/11.70%, g2 10.47%/11.40%, g3 10.16%/11.15%
- ng=8: g0 8.71%/10.55%, g1 10.59%/12.05%, g2 10.14%/11.63%, g3 10.94%/12.51%, g4 10.36%/11.76%, g5 10.59%/11.93%, g6 9.88%/11.30%, g7 10.36%/11.83%

All groups show **8.7–10.9% median** (spread 2.2pp) and 100% pred>truth one-sided, same as overall 10.22%. Per-step group drops (ng=8, median) are also uniform: e.g. 34→35: 20.4–22.6% across 8 groups, 35→36: 21.9–23.6% across groups. No group is outlier.

**Conclusion:** `ω` deviation is **uniform across column groups**, not concentrated in a sub-vector family. Sub-vector geometry maps groups to sub-table usage, but all groups share the same steeper-than-pilot decay. This refutes the “family-weight anomaly localized to specific column groups” rival as the driver for L0 gate (the rival predicted a spike in one group).

## 4. Rival hypotheses (1 hour budget)

### H0 (working): global ω misfits L0 gate — CONFIRMED directionally but not explanatory via localized family
Per-family drop ratios for L0 (`0.199/0.218/0.269/0.314`) vs global (`0.18/0.22/0.26/0.34`) differ by ≤2pp per family; using L0’s own drop-derived ω improves K35 error from 10.22%→7.02% but still fails 5% (see OMEGA_FIX_EVAL). The misfit is not a gross per-family weight error but a **layer-wide steeper slope** (all families larger) that the 4-parameter discrete geometry model `error(K)=Σ w_i·2^{-bits_i(K)}` cannot capture with 5 anchors (residual 37% at K36 even when fitting, see §5 of OMEGA). Uniform across groups suggests L0 (layer 0, immediately after embedding) has different weight statistics (higher variance, more quant-sensitive) rather than a sub-table-specific book defect.

### H1: book-quality anomaly at specific rungs — REJECTED for gate
- Gate K28-K38 books: `cbl_poolb` `book_restored_content_addressed` or `book_trained_and_banked`, `scale_policy one_shot_cand0`, `ldlq False`, monotone decreasing for 256/256 experts, entry utilization >98% per harness. No non-monotone step (gate K34 worse than K33 for 0/256 vs up/down incumbent shards where K34 worse for 256/256). Scout vs primary vs full-layer K35 identical (2.028e-06) — no book divergence. L14 gate also CBL monotone. Book lens does not explain L0-only gate failure.

### H2: chain-min arm structure unique to L0 gate — REJECTED
`winning_arm` for L0 gate at all 5 anchors and K35 is `free` for 256/256 experts (`selected_weight_mse == free_weight_mse`, diff 0). `embed_weight_mse` not selected. No chain-min flat-tread (contrast L0 up/down `v2s-full-layer` incumbent fallback shards at K29/K34 where `embed` wins for 256/256 and free is 45% worse — those shards are **not CBL**, excluded from this gate analysis). Pure-incumbent L14 also shows same staircase without chain. Chain adds flatness only for mixed-basis fallback, not for gate CBL.

### H3: column-group localized ω anomaly — REJECTED (see §3)
Uniform 10% error across all 2/4/8 partitions refutes concentration. If localized, one of 8 groups would show >>10% vs others ~0%; observed spread is 2.2pp, all groups fail.

**Evidence is not ambiguous:** the 10.22% bias is systematic, one-sided, uniform across experts (min 8.74% max 11.30%) and groups, with K32 5.46%, K33 5.91%, K34 9.01% also failing in the 31→36 gap — exactly the interval with max gap 5. The next layer (L14) same anchors/x passes with 1.99% at K35, so the effect is L0-specific.

## 5. What the live fix gets right and where it breaks

The 5-anchor geometry-aware `x(K)` fixes the original 3-anchor r3-missing-family bug for pilot layers (L14 gate p95 22.5%→3.75% at K35). For L0 gate, the same `x` corrects the family staircase (global 3-anchor would be ~22% at K35, now 10%) but leaves a **residual layer offset**: L0’s steeper slope requires `x` spacing ~12% larger between 31–36 than pilot average. With global `ω`, PCHIP under-decays. The fix is therefore necessary but not sufficient for the steepest layer.

No single 5-anchor set with global `ω` passes both L0 gate and L14 gate simultaneously (exhaustive search over `C(11,5)=462` sets with 28,38 inclusive, max gap ≤5: 40 sets pass L0 gate alone, 0 pass both L0 and L14 gate at med≤5%/p95≤15% for all interiors 28..38). The production set `(28,31,36,37,38)` is the minimal passing for L14; L0 gate needs a set that shortens the 31→36 gap (e.g. `(28,31,35,37,38)` passes L0 gate with 4.96% max but fails L14 gate at K36 p95 37.9%). This confirms a **per-layer slope difference**, not a universally bad anchor.

## 6. Provenance

- Scripts: `harness/harness.py`, `prismaquant/cb_layout.py:121` `bit_split`, `codebook_subtable_shapes:210`, `tools/dsv4_afast_campaign.py:277` `pchip_monotone` + `_pava_decreasing`
- Data: `/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/burn-shards/layer_000_gate_proj_v2s-full-layer_K{28..38}.pkl` (CBL `free_weight_mse`, `free_group_mse`), `layer_000_{up,down}_proj_v2s-scout_K*.pkl`, `pilot2/shards/layer_014.pkl`, `PILOT_FULL_MEASUREMENTS.pkl`, `burn-afast/BURN_MANIFEST.json` `interp_anchors [28,31,36,37,38]` `interp_coordinate geometry_x_v1` `book_sha256` per K
- No GPU, no writes outside `interp-diagnosis/` and `/w`, CPU-only
