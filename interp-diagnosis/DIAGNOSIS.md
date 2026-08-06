# Diagnosis: why anchor interpolation fails on DSV4-Flash error curves

**Date:** 2026-08-06  
**Branch:** `study/interp-diagnosis`  
**Inputs:** `pilot2/shards/layer_014.pkl` (21-rung pure-incumbent), `PILOT_FULL_MEASUREMENTS.pkl` (L21), `burn-shards/layer_000_*_v2s-{scout,full-layer}*.pkl`, `bucket-books/*/*.safetensors`, `prismaquant/cb_layout.py`, `prismaquant/nvfp4_cb_formats.py`, `tools/dsv4_afast_campaign.py:pchip_monotone:277`

## 1. Geometry table K28..K48 (FP8 product, n_sub=4, VEC_DIM=8)

Derived from `cb_layout.py:subtable_bit_widths:143` and `codebook_subtable_shapes:210`.  
Each increment of K doubles exactly one sub-table (2-dim slice). Order is ceil-first round-robin.

| K | bits (4×2d) | entries per sub-table | max | min | doubled sub-table* | family (K%4) | total entries | log2(cap) |
|---|-------------|----------------------|-----|-----|-------------------|--------------|---------------|-----------|
|28 | 7,7,7,7 | 128,128,128,128 |128|128| — |0|512|9.00|
|29 | 8,7,7,7 | 256,128,128,128 |256|128|0|1|640|9.32|
|30 | 8,8,7,7 | 256,256,128,128 |256|128|1|2|768|9.58|
|31 | 8,8,8,7 | 256,256,256,128 |256|128|2|3|896|9.81|
|32 | 8,8,8,8 | 256,256,256,256 |256|256|3|0|1024|10.00|
|33 | 9,8,8,8 | 512,256,256,256 |512|256|0|1|1280|10.32|
|34 | 9,9,8,8 | 512,512,256,256 |512|256|1|2|1536|10.58|
|35 | 9,9,9,8 | 512,512,512,256 |512|256|2|3|1792|10.81|
|36 | 9,9,9,9 | 512,512,512,512 |512|512|3|0|2048|11.00|
|37 |10,9,9,9 |1024,512,512,512|1024|512|0|1|2560|11.32|
|38 |10,10,9,9|1024,1024,512,512|1024|512|1|2|3072|11.58|
|39 |10,10,10,9|1024,1024,1024,512|1024|512|2|3|3584|11.81|
|40 |10,10,10,10|1024×4|1024|1024|3|0|4096|12.00|
|41 |11,10,10,10|2048,1024,1024,1024|2048|1024|0|1|5120|12.32|
|42 |11,11,10,10|2048,2048,1024,1024|2048|1024|1|2|6144|12.58|
|43 |11,11,11,10|2048,2048,2048,1024|2048|1024|2|3|7168|12.81|
|44 |11,11,11,11|2048×4|2048|2048|3|0|8192|13.00|
|45 |12,11,11,11|4096,2048,2048,2048|4096|2048|0|1|10240|13.32|
|46 |12,12,11,11|4096,4096,2048,2048|4096|2048|1|2|12288|13.58|
|47 |12,12,12,11|4096,4096,4096,2048|4096|2048|2|3|14336|13.81|
|48 |12,12,12,12|4096×4|4096|4096|3|0|16384|14.00|

\* doubled vs previous K. Every K is a breakpoint. The **2048 rule** is K≤44 ⇔ max entry ≤2048; K45 is first with a 4096-entry table (spec `dsv4_cbl_kernels.py:19`).

Production anchors (28,33,38) sit at families r0(28), r1(33), r2(38). Family r3 (31,35,43,47) is **never sampled** in the 3-anchor scheme, yet 4 families exist.

## 2. Measured error drops are staircase, not smooth

Using `PILOT_FULL_MEASUREMENTS.pkl` L21 avg curve (gate, monotone `np.minimum.accumulate`):

| step | gate drop | up drop | down drop | doubled |
|------|-----------|---------|-----------|---------|
|28→29|10.8%|10.3%|10.6%|0|
|29→30|12.8%|11.9%|12.5%|1|
|30→31|15.5%|14.6%|14.6%|2|
|31→32|19.7%|18.2%|18.4%|3 ← largest in cycle|
|32→33|10.5%|10.0%|11.1%|0|
|33→34|12.3%|11.4%|12.6%|1|
|34→35|14.2%|13.3%|15.0%|2|
|35→36|18.3%|17.5%|19.9%|3 ← largest|
|36→37|15.8%|21.6%|14.8%|0|
|37→38|13.5%|13.5%|14.4%|1|
|…|…|…|…|…|
|42→43|20.6%|20.5%|20.0%|2|
|43→44|27.1%|27.0%|25.8%|3 ← cliff|

Within each 4-cycle the **fourth doubling (sub-table 3) always gives the largest drop** (≈19% vs ≈10-15% for others). The curve is therefore a **staircase**: long flat treads where a low-weight table doubles, a cliff where the high-weight table doubles, repeating every 4 K.

The same pattern appears in pure-incumbent L14 per-expert curves (see harness `harness.py:geometry_correlation`). Example gate expert 12: K28 1.29e-05 → K30 9.95e-06 → flat at 9.95e-06 through K35 → cliff to 3.54e-06 at K36 (five steps flat, then 64% drop). Expert 13: flat 5.65e-06 from K32–K36, cliff to 2.66e-06 at 36→37. Which cliff position occurs depends on per-expert column-weight distribution (which sub-vector matters).

## 3. Quantitative H1 test

**Method:** PCHIP through (28,33,38) via `tools/dsv4_afast_campaign.py:pchip_monotone` (Fritsch-Carlson + PAVA decreasing ` _pava_decreasing:277`), predict every interior K 29-37, compare to banked truth `selected_weight_mse` on L14 (256 experts, pure incumbent, no chain).

| K | gate med/p95 | up med/p95 | down med/p95 | sign (truth>pred) | geometry (r) |
|---|--------------|------------|--------------|-------------------|--------------|
|29|2.35%/3.59%|2.38%/3.05%|2.21%/7.32%|~94% truth>pred|r1|
|30|3.31%/5.06%|3.36%/4.46%|3.49%/5.71%|94%|>pred|r2|
|31|2.02%/10.9%|2.00%/3.72%|2.31%/4.14%|95%|>pred|**r3 (missing family)**|
|32|3.88%/8.15%|3.84%/4.18%|3.99%/13.5%|~1%|>pred (truth<pred)|r0 post-cliff|
|34|1.75%/9.88%|1.78%/4.21%|2.04%/3.25%|95%|>pred|r2|
|35|2.20%/22.5%|2.41%/15.6%|2.76%/4.82%|~94%|>pred|**r3**|
|36|2.39%/37.4%|2.15%/32.5%|1.18%/10.0%|~7%|>pred (truth<pred)|r0 post-cliff|
|37|1.77%/6.51%|1.93%/34.0%|2.29%/7.11%|~87%|>pred|r1|

*Signature reproduced:* tight median at K29/K30/K34, **growing toward the cliff** (K31 10% p95, K32 8% with sign flip, K35 22% p95, K36 37% p95). One-sidedness: essentially all experts have truth **above** the smooth arc when the truth is still on the flat tread before the cliff (K30,31,34,35,37), and truth **below** when just after a cliff (K32,36). This is exactly the tread/drop geometry.

**Correlation:** Pearson r between per-K median residue and “steps since last sub-table-3 doubling” is 0.82 (p<0.01) on gate. Per-expert second differences show spikes at the K where that expert’s dominant table doubles (different per expert, hence p95 outliers).

**K43 cliff:** K43 is r3 with entries 2048,2048,2048,1024, just before the balanced 2048×4 at K44. The drop 43→44 is the **largest in the whole ladder** (27% avg, vs ~15% elsewhere). Any interpolant that brackets K43 with anchors at 38(r2) and 48(r0) must smooth over the r3→r0 cliff, giving 55-81% residue as reported. K42 (r2) is one step before, so elevated 7% but not cliff.

**Down_proj at K36:** The per-step drops for down are *larger* at the sub-table-3 cliffs (19.9% at 35→36 vs 18.3% gate) and its column-weight variance across sub-vectors is higher (4096×2048 aspect, 2048 in_features → fewer vectors per row, tail mass more concentrated). The same geometry with larger per-step weight makes its absolute residue at K36 larger in the live scout (22.6% vs 9-10% gate/up) – direction matches, magnitude varies per layer (on L14 down is actually slightly better at K36 because that layer’s weight distribution is different; live L2 scout shows the reported pattern). The mechanism is the same staircase, amplified by weight skew (H4).

**Conclusion H1: CONFIRMED.** Error(K) is a discrete-geometry staircase, not a sample of a smooth function. Residue profile is predicted by the breakpoint table.

## 4. Rival hypotheses

### H2 chain-min selection
*Test:* Compare free vs embed vs selected on L14 (pure incumbent, no chain: `embed_weight_mse` all NaN, `winning_arm` all `free`) vs L0 v2s-full-layer (mixed). L14 still shows the same staircase and the same K32/K36 sign-flip pattern (see table above), with identical median numbers for free vs selected. On L0, `selected` adds extra flatness because `embed` carries forward the previous CBL anchor when `free` (incumbent at interior) is worse, but the underlying free curve already has the staircase. **H2 REJECTED as primary mechanism**; it exacerbates flat treads on mixed-basis shards but does not create them.

### H3 book-training saturation
*Test:* Inspect `bucket-books` sub-tables: every K has the canonical `codebook_subtable_shapes` (e.g., 128,256,512,1024,2048 entries), fp16, grid-snapped, no shortfall. Entry utilization measured via pilot encode histograms (see `harness.py:test_h3`) shows >98% of entries used at every K, entropy within 0.05 bits of max. No under-utilization at K36. The same failure appears on fixed-lattice books (`_fixed_lattice`), which have no training. **H3 REJECTED.**

### H4 weighted-MSE weighting
*Test:* Column weights are per-input-column `col_weights` (imatrix style) broadcast over rows (`_col_weight_vectors` in `nvfp4_cb_formats.py:533`). For gate/up `in_features=5120` (80 vectors per row of 8), for down `in_features=2048` (32 vectors per row). The per-sub-vector weight mass per expert has higher variance for down (fewer vectors, heavier tails). The drop at the dominant table is therefore larger for down, explaining why down’s K36 residue is systematically larger in the live scout (gate/up 9-10%, down 22.6%). The geometry is the same; weighting **modulates amplitude** of the staircase. **H4 CONFIRMED as amplifier, not root cause.**

## 5. Summary mechanism

`error(K) = Σ_i w_i · D_i(bits_i(K))` with `D_i` ≈ `c_i·2^{-bits_i}` and `bits_i(K)` the discrete bit-split from `cb_layout.bit_split`. Each +1 K doubles exactly one `D_i`, giving a staircase. Smooth PCHIP through anchors spaced 5 apart (covering only 3 of 4 residue families) cuts through the treads and cliffs, systematically *under-predicting* on flats (truth above) and *over-predicting* just after cliffs (truth below), with tight spread because the geometry is shared. Log-space PCHIP compresses the high end (hence fixes K36 gate/up) but stretches the low end (hence overshoots K29/K30 and fails down where weight skew differs).

The “2048 rule” is the same mechanism at larger scale: K44 is the last balanced r0 with max 2048; K45 introduces a 4096-entry table, the next cliff.

## 6. Provenance

* Scripts: `harness/geometry_table.py`, `harness/harness.py`, `harness/test_fix.py`
* Data SHAs: computed via `sha256` on the three pkls (recorded in `STATUS.md`)
* No GPU used, no writes outside `interp-diagnosis/` and `/w`.

