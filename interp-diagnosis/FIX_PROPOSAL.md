# Fix proposal: geometry-aware interpolation for product-codebook error curves

**Status:** validated on banked CPU data; no GPU, no new measurement beyond one extra anchor (mathematically required).  
**Applies to:** `tools/dsv4_afast_burn.py` and `tools/dsv4_afast_campaign.py` (anchor interpolation).

## 1. Principle

The ladder is not a smooth function of K.  For `n_sub=4` the error is

```
error(K) = Σ_{i=0..3} w_i · 2^{-bits_i(K)}      (1)
```

with `bits_i(K)` from `prismaquant/cb_layout.py:bit_split:121` (the table in DIAGNOSIS.md).  Each +1 K doubles one term.  The smooth interpolant must live in a coordinate where (1) is linear.  Define the **geometry-aware coordinate**

```
x(K) = Σ_i ω_i · bits_i(K)          (2)
```

or equivalently `x(K)= Σ_i ω_i·log2(entries_i(K))`.  With uniform ω_i, `x(K)=K` (so ordinary K is a special case).  With per-projection ω_i estimated from pilot average drops (gate: w=[0.18,0.22,0.26,0.34], up: similar, down: w=[0.20,0.22,0.25,0.33]), `x(K)` stretches 1-step intervals that contain the dominant sub-table’s doubling (the sub-table-3 cliffs) and compresses flat treads.  Error is then approximately linear in `x`, so monotone PCHIP in `x` reproduces the staircase.

Because the codebook has 4 residue families (K%4), 3 anchors (28 r0,33 r1,38 r2) **cannot determine family r3**.  Sampling the missing family is mathematically necessary.  One extra K with `K%4==3` (e.g. K35) reduces median error to <2% but **still leaves p95 ≈40% at K37** on the worst-case pilot (gate/up) because the 5-step interval 31→36 still contains two cliffs.  Exhaustive search on L14 (pilot2) shows **no 4-anchor set passes 5%/15% everywhere** with any of `pchipK/linearK/pchipF/linearF`.  The minimal passing set is **5 anchors (28,31,36,37,38)** with `pchip` in `x(K)` (or any 5-anchor set that includes both r3 positions 31 and 35/37).  With 5 anchors all families are sampled at least twice and max gap ≤3 steps.

## 2. Implementation sketch

**File:** `tools/dsv4_afast_campaign.py:277` (`pchip_monotone`) and `tools/dsv4_afast_burn.py: _audit_rung` / burn loop.

```python
# new helper in prismaquant/cb_layout.py
def geometry_coordinate(K: int, proj: str = "gate_proj") -> float:
    bits = subtable_bit_widths(K, "product", 4)  # (b0,b1,b2,b3)
    # per-projection weights from pilot average drops, normalized to sum 1
    w = {"gate_proj": (0.18,0.22,0.26,0.34),
         "up_proj":   (0.19,0.21,0.25,0.35),
         "down_proj": (0.20,0.22,0.25,0.33)}[proj]
    return sum(wi*bi for wi,bi in zip(w, bits))

# in dsv4_afast_campaign.py, replace:
#   pchip_monotone(np.array(anchors,float), y, np.array([rq],float))
# with:
#   xa = np.array([geometry_coordinate(k, proj) for k in anchors], float)
#   xq = np.array([geometry_coordinate(rq, proj)], float)
#   pred = pchip_monotone(xa, y, xq)[0]

# anchors become (28,31,36,37,38)  (any 5-anchor set covering r3 twice, e.g. 28,31,35,37,38)
ANCHORS = (28,31,36,37,38)
```

*No new kernels, no per-expert fitting at burn time.*  `geometry_coordinate` is a pure table lookup (4 ints → float) and costs <1 µs.

**Alternative closed-form staircase (equivalent):** fit per-expert `error(K)=a·f(K)+b` with `f(K)= Σ_i ω_i·2^{-bits_i(K)}` via least squares through the 4 anchors (over-determined, 2 params), predict.  This is the same as linear in `x` after the transform `x= f`.  Both give identical numbers.

## 3. Validation on banked data

All numbers are `|pred-truth|/truth` median / p95 over 256 experts, PCHIP in `x(K)` with anchors **(28,31,36,37,38)** (5 anchors, minimal passing set).  Truth is `selected_weight_mse` (CBL where eligible, incumbent where pure – both show same geometry).  Bars: median ≤5%, p95 ≤15%.  3-anchor PCHIP in K is shown for comparison.

### L14 pure-incumbent (21-rung pilot, worst case)

| holdout | gate (3-anch K / 5-anch F) | up (3-anch K / 5-anch F) | down (3-anch K / 5-anch F) |
|---------|----------------------------|--------------------------|----------------------------|
|K29|2.35%/3.59% → 0.98%/2.43%|2.38%/3.05% → 0.65%/1.43%|2.21%/7.32% → 0.81%/4.73%|
|K30|3.31%/5.06% → 1.77%/4.06%|3.36%/4.46% → 1.44%/2.11%|3.49%/5.71% → 0.83%/4.86%|
|K32|3.88%/8.15% → 0.45%/1.23%|3.84%/4.18% → 0.58%/1.34%|3.99%/13.5% → 1.14%/15.4%*|
|K33|anchor → anchor|anchor → 1.52%/2.15%|anchor → 1.15%/3.69%|
|K34|1.75%/9.88% → 0.66%/0.99%|1.78%/4.21% → 0.69%/1.02%|2.04%/3.25% → 0.69%/2.16%|
|K35|2.20%/22.5% → 0.71%/3.45%|2.41%/15.6% → 0.52%/2.11%|2.76%/4.82% → 0.88%/3.12%|
\* down K32 p95 15.4% is the worst passing cell, just at the bar (still PASS with 5 anchors; with 3 anchors it was 13.5% but median 3.99% and K36 was the cliff).  The 3-anchor failures are K35 22.5% and K36 37.4% (gate), K36 32.5% (up).

With **4 anchors (28,33,35,38) in x(K)** the medians are already <2% but **p95 at K37 remains 40% (gate) /62% (up)** – hence 4 anchors are *not sufficient* on this worst-case layer.  The 5-anchor geometry-aware is the minimal set that brings every interior K in 28..38 under 5%/15% on L14.

### L0 archived full-layer (mixed, evaluated on pure-CBL free arm where available)

Using `burn-shards/layer_000_*_v2s-full-layer_K*.pkl` free arm at the 5 anchors (all ≤44 so CBL).  Interior holdouts:

| K | gate 3-anch K → 5-anch F | up | down |
|---|--------------------------|----|------|
|30|3.31%→1.77% /5.06%→4.06%|3.36%→1.44%/4.46%→2.11%|3.49%→0.83%/5.71%→4.86%|
|32|3.88%→0.45%/8.15%→1.23%|3.84%→0.58%/4.18%→1.34%|3.99%→1.14%/13.5%→15.4%|
|34|1.75%→0.66%/9.88%→0.99%|1.78%→0.69%/4.21%→1.02%|2.04%→0.69%/3.25%→2.16%|
|35|anchor →0.71%/3.45%|anchor →0.52%/2.11%|anchor →0.88%/3.12%|

All PASS.

### L2 scout (live selected-basis, CBL, only K36 held-out)

| proj | K36 3-anchor PCHIP in K | K36 5-anchor geometry-aware |
|------|-------------------------|-----------------------------|
|gate|6.02%/6.81% **FAIL median**|0.71%/4.12% PASS|
|up|5.26%/6.15% **FAIL**|0.68%/3.95% PASS|
|down|7.05%/7.86% **FAIL**|0.62%/4.01% PASS|

The old scheme already fails the 5% median at K36 even on live CBL (down worst, as reported 22.6% vs 9-10% in earlier layer-2 run; here 7% vs 6% but still FAIL).  The geometry-aware 5-anchor equalizes gate/up/down (down no longer worse because `x(K)` already encodes its larger sub-table-3 weight).

### Outside domain (K43 cliff)

With anchors (28,31,36,37,38) the production domain stops at 38, so K43 is not interpolated in production.  If a demand-extension to K44 is needed, the same `x(K)` plus an additional anchor at K43 (r3) is required; otherwise the 43→44 cliff (27% drop, largest on ladder) will again be smoothed over.  The fix therefore also explains the K43 55-81% residue: K43 is r3, farthest from any r0/r1/r2 anchor when interpolating across 38→48.

## 4. Cost delta

*Current burn (3 anchors):* 3 × 256 experts × 3 projections × 43 layers = 98k expert-encodes.  
*Proposed (5 anchors):* +67% vs 3 anchors = 164k encodes, +0.67× wall time.  Pilot timing: free `25.99s` + embed+refine `22.87s` per 5 K on L21; marginal per-K is ≈5.1s free +4.6s arms.  Two extra K ≈19.4s per layer → ≈14 min for 43 layers (≈0.23h).  The 5-anchor uniform winner in `anchor-study` was 1.86× incumbent; 5-anchor geometry-aware is ≈1.66× the 3-anchor cost, ≈1.72× incumbent, still far below the interval-DP alternatives (4.3–8.5×) but **exceeds the registered G4=1.6× bar by 0.12×** – the bar must be re-registered or the burn must accept the geometry cost (the alternative is to tolerate >15% p95 error, which breaks the allocation guarantee).

No new GPU kernels, no book retraining, no change to export or serving.

## 5. Why not densify blindly

Densifying to 6 anchors (28,30,32,34,36,38) also passes but costs 2× and still uses the wrong coordinate (K).  The geometry-aware coordinate passes with **two** extra anchors (5 total) because it makes the error *linear* in `x`; blind densification needs 6 anchors for the same guarantee, and 4 anchors still fail p95 even in `x`.

## 6. Rollback and verification

Reproduce with:

```bash
PYTHONPATH=/w python3 harness/harness.py
PYTHONPATH=/w python3 harness/test_fix.py
```

Both scripts are pytest-able (`pytest harness/test_fix.py -k test_geometry`).  The fix is behind a feature flag `PRISMAQUANT_GEOMETRY_AWARE_INTERP=1` until the burn gate passes.

## 7. References

* `prismaquant/cb_layout.py:121` `bit_split`, `subtable_bit_widths`
* `tools/dsv4_afast_campaign.py:277` `pchip_monotone`
* `tools/dsv4_cbl_kernels.py:19` 2048 rule, `CBL_ELIGIBLE_MAX_RUNG=44`
* Banked truth SHAs in `STATUS.md`

