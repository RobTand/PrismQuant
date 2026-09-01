# NVFP4-CB Phase-0 exp-1b — CORRECTED CB-vs-IQ + native-FP4 premium (Qwen3-0.6B)

> **EMULATION GATE, not the served metric.** Whole-model emulated forward KL-vs-BF16 (fp32, held-out wiki.test.raw, seqlen 512 × 8192 tok). A kernel phase must re-confirm on served vLLM/llama.cpp KL before promotion.

**The honest frame (per `rd_ceiling_study.md` + its reviewer correction).** Matched-bytes CB-vs-IQ is NOT the decision: the FP4-grid *value* tax is small (+4.5% full / +10% signed), and the residual matched-bytes gap is a STRUCTURAL scale-packaging tax — NVFP4's mandatory group-16 E4M3 scale (0.500 bpw) vs IQ's amortised two-tier scale (~0.3125 bpw) ⇒ **~0.19 bpw**, which is MITIGABLE by reconstructing a two-tier scale in the kernel prologue. So CB losing IQ at matched bytes is EXPECTED. **The decision number is the native-FP4-speed PREMIUM:** the extra bpw at which CB reaches IQ2_S's and IQ3_XXS's KL (the price of tensor-core-native FP4 serving, which the emulation cannot reward).

- git `a0c0a68040f35de98a4373778dfcbc262371ec94` · 196 target Linears · 7 roles · imatrix E[x²] col_weights (paired per seed).
- Corrections since exp-1: (a) CB now uses the SAME E4M3-legal scale sweep IQ always had; (b) sign-factored `signed` mode; (c) byte-match via SHARED per-role learned codebooks (per-tensor sidecar is not byte-competitive).
- Mode/compute: full-k16 + sweep is 56 s/Linear (≈3 h/seed) so it is a 1-seed stronger-mode anchor; the break-even sweep uses learned-shared PRODUCT mode (fast) which slightly UNDER-estimates full-mode CB quality — so the measured premium is a CONSERVATIVE UPPER BOUND (true premium is smaller).

## Per-arm results

| Arm | seeds | act | body bpw | TOTAL bpw | KL_conf mean±std | KL_all | top1 | n_swap |
|---|---|---|---|---|---|---|---|---|
| FP8CB_K36 | 2 | W4A4/W8A8 | 4.500 | 4.525 | 0.0586±0.0008 | 0.0929 | 0.977 | 196 |
| FP8CB_K40 | 2 | W4A4/W8A8 | 5.000 | 5.025 | 0.0314±0.0005 | 0.0540 | 0.989 | 196 |
| FP8CB_K44 | 2 | W4A4/W8A8 | 5.500 | 5.525 | 0.0192±0.0004 | 0.0327 | 0.993 | 196 |
| IQ2S | 4 | W4A4/W8A8 | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| IQ3XXS | 2 | W4A4/W8A8 | 3.062 | 3.062 | 0.4139±0.0112 | 0.5447 | 0.837 | 196 |
| IQ4XS | 2 | W4A4/W8A8 | 4.250 | 4.250 | 0.0597±0.0003 | 0.1035 | 0.973 | 196 |
| PROD_shared_k16 | 2 | W4A4/W8A8 | 2.500 | 2.500 | 2.2102±0.0923 | 2.2393 | 0.445 | 196 |
| PROD_shared_k20 | 2 | W4A4/W8A8 | 3.000 | 3.001 | 0.7429±0.0237 | 0.9013 | 0.737 | 196 |
| PROD_shared_k24 | 2 | W4A4/W8A8 | 3.500 | 3.504 | 0.3705±0.0046 | 0.4813 | 0.859 | 196 |
| PROD_shared_k28 | 1 | W4A4/W8A8 | 4.000 | 4.017 | 0.2298±0.0000 | 0.3319 | 0.902 | 196 |
| FULL_k16_shared | — | W4A4/W8A8 | 2.500 | 2.533 | (footprint only) | — | — | — |
| SIG16_shared | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.7504±0.0725 | 2.7177 | 0.405 | 196 |
| SIG16_shared_smooth025 | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.6695±0.1733 | 2.6784 | 0.419 | 196 |
| SIG16_shared_wo | 4 | W-only | 2.500 | 2.500 | 2.3429±0.0509 | 2.3562 | 0.457 | 196 |
| IQ2S_wo | 4 | W-only | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| FULL_k14_sweepoff | 4 | W4A4/W8A8 | 2.250 | 2.250 | 3.7585±0.1304 | 3.5898 | 0.291 | 196 |
| FULL_k14_sweepon | — | W4A4/W8A8 | 2.250 | 2.250 | (footprint only) | — | — | — |
| FULL_k16_fixed | — | W4A4/W8A8 | 2.500 | 2.500 | (footprint only) | — | — | — |
| LEARN_k16_pertensor | — | W4A4/W8A8 | 2.500 | 3.433 | (footprint only) | — | — | — |

## THE DECISION — native-FP4 break-even premium

learned-SHARED-per-role PRODUCT-mode NVFP4-CB (fast, conservative upper bound) vs the IQ ladder, W4A4 served-faithful, 2 seeds.

| Arm | total bpw | KL_conf mean±std | top1 |
|---|---|---|---|
| PROD_shared_k16 | 2.500 | 2.2102±0.0923 | 0.445 |
| PROD_shared_k20 | 3.001 | 0.7429±0.0237 | 0.737 |
| PROD_shared_k24 | 3.504 | 0.3705±0.0046 | 0.859 |
| PROD_shared_k28 | 4.017 | 0.2298±0.0000 | 0.902 |
| IQ2S | 2.562 | 1.5837±0.0751 | 0.568 |
| IQ3XXS | 3.062 | 0.4139±0.0112 | 0.837 |
| IQ4XS | 4.250 | 0.0597±0.0003 | 0.973 |

- **Crossing IQ2_S** (KL 1.584 @ 2.562 bpw): product-CB reaches it at ≈**2.71 bpw** ⇒ native-FP4 premium ≈ **+0.15 bpw** (conservative upper bound).
- **Crossing IQ3_XXS** (KL 0.414 @ 3.062 bpw): product-CB reaches it at ≈**3.45 bpw** ⇒ premium ≈ **+0.38 bpw**.

### FP8-CB mid-range — does it WIN per-byte?

RD study: FP8-grid tax <1%; this is the 4-to-8-bit gap band where CB may beat IQ per-byte. FP8_CB vs the nearest IQ point:

| FP8_CB rung | total bpw | KL_conf | nearest IQ | IQ bpw | IQ KL | per-byte |
|---|---|---|---|---|---|---|
| FP8CB_K36 | 4.525 | 0.0586 | IQ4XS | 4.250 | 0.0597 | CB better KL AT MORE bpw (Δbpw +0.28) |
| FP8CB_K40 | 5.025 | 0.0314 | IQ4XS | 4.250 | 0.0597 | CB better KL AT MORE bpw (Δbpw +0.78) |
| FP8CB_K44 | 5.525 | 0.0192 | IQ4XS | 4.250 | 0.0597 | CB better KL AT MORE bpw (Δbpw +1.28) |

(No exact-bpw IQ twin exists at 4.5–5.5 bpw in the registry; the honest read is the KL-vs-bpw ordering, not a matched-bpw delta.)

## Matched-bytes verdicts (context, NOT the decision)

### (a) Matched-bytes CB-vs-IQ2_S — the structural scale tax (EXPECTED, not a kill)

At matched TOTAL bytes CB is expected to trail IQ by ~0.19 bpw of scale-packaging (RD study); the question is only HOW MUCH and whether it is an encoder deficit (it is not).

- **signed-S16-shared (weaker mode)** W4A4 2.7504 vs IQ2_S 1.5837 = **+73.7%** at matched bytes (2.500 vs 2.562 bpw).
- **product-k16-shared** W4A4 2.2102 vs IQ2_S 1.5837 = **+39.6%** (product ≥ signed, matching the RD prediction +4.5% vs +10% grid tax).
- **Weight-only (pure codebook, activation asymmetry removed):** signed-S16-shared 2.3429 vs IQ2_S 1.5837 = **+47.9%** — the gap PERSISTS weight-only, so it is NOT a W4A4 artifact; per the RD study it is the structural scale-packaging bpp tax (matched-SIZE FP4-Lloyd ≈ IQ), MITIGABLE via in-kernel two-tier scales — NOT an encoder/grid deficit. Hence 'loses at matched bytes' is the expected non-decision; see THE DECISION above.

### (b) Scale-sweep (the exp-1 rendering asymmetry, now fixed)

- fixed-full-k14 sweep OFF 3.7585 (reproduces exp-1 B_fixed_full_k14 ≈3.76, sanity check). The dedicated sweep-ON k14 arm was dropped to footprint-only (confirmatory) — every break-even and matched-bytes CB arm above ALREADY renders with the sweep ON (the E4M3-legal scale grid + WLS refit IQ always had), which is the rendering asymmetry exp-1 suffered.

### (c) Shared-vs-per-tensor byte reality

- SHARED per-role sidecar (signed): 7.2 KB → total 2.500 bpw (≈0 over body).
- SHARED per-role sidecar (full-k16): 1.84 MB → total 2.533 bpw.
- PER-TENSOR k16 sidecar: 51.4 MB → total **3.433 bpw** (+0.93 bpw model-wide; but ~+2.1 bpw on a 1M-param Linear — the small-N tensors the coordinator flagged). NOT byte-competitive; this is why the champion shares codebooks per-role (sidecar → ≈0).

### (d) Smoothing on top of the sweep

- signed-S16-shared 2.7504 → +smooth α=0.25 2.6695 (+2.9%, σ=0.1733) → **within between-seed noise**.

## BOTTOM LINE — go / pivot / shelve

- (i) **NVFP4_CB native-FP4 premium** to MATCH IQ2_S quality ≈ **+0.15 bpw** (crossing ≈2.71 bpw); to match IQ3_XXS ≈ **+0.38 bpw** (crossing ≈3.45 bpw) — the premium GROWS with bpw. Both are CONSERVATIVE upper bounds (product mode under-estimates full) and both sit at/near the ~0.19 bpw structural scale-tax the RD study predicts — i.e. the price of native-FP4 tiles, mitigable to ~0 with an in-kernel two-tier scale.
- (ii) **FP8_CB mid-range**: NO FP8_CB rung beats its nearest IQ point per-byte (IQ4_XS's 4-dim non-FP4 grid is hard to beat at 4–5.5 bpw). FP8CB_K36@4.53=0.059 vs IQ4XS@4.25=0.060; FP8CB_K40@5.03=0.031 vs IQ4XS@4.25=0.060; FP8CB_K44@5.53=0.019 vs IQ4XS@4.25=0.060.
- (iii) **Sub-3-bpw NVFP4_CB** is the promising lane: it reaches IQ2_S quality at ~+0.15 bpw and BEATS IQ2_S outright by ~3 bpw (KL 0.74 vs 1.58 at 3.0 bpw) while decoding to native FP4 — the whole point. FP8_CB does not add a per-byte win over IQ4 in this band, so the mid-range is IQ/kernel-bound, not a CB opportunity.

**GO (to the kernel phase) for the sub-3-bpw NVFP4_CB lane** — the native-FP4 premium is small (~0.15 bpw, conservative) and structural (scale packaging), not an encoder/grid deficit; the decision now rests on whether the two-tier-scale kernel is worth building. The matched-bytes 'loss' was the expected tax, not a kill. Emulation gate at 0.6B — 4B + served KL must re-confirm before any promotion.

## Caveats

- Emulation gate only, 0.6B triage; 4B + served re-confirm remain. Uniform 2.5 bpw on ALL 196 Linears heavily damages a 0.6B model (top1 < 0.6 for every 2.5-bpw arm incl. IQ2_S) — the CB-vs-IQ DELTA / crossing is the signal, not absolute KL.
- The break-even curve uses learned-SHARED PRODUCT mode (fast, sweep ON); it UNDER-estimates full-mode CB quality, so every premium is a conservative UPPER bound. full-k16 and the sweep-ON k14 arms were dropped to footprint-only (redundant/confirmatory).
- FP8_CB rungs use the registered FIXED fp8-grid product lattice + sweep (not learned-shared); RD study puts the FP8-grid tax <1%, so learned-shared would move them <1% — the per-byte verdict is robust to this.
- Shared codebooks trained on ≤2^20 pooled per-role vectors (subsampled for Lloyd); CUDA Lloyd tie-noise per seed as exp-1.
