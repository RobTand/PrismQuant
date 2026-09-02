# NVFP4-CB rate-distortion CEILING study

> Self-contained numerical RD analysis (numpy/torch, held-out eval). **Not** a served metric — it isolates the codebook/grid coding limit from encoder/rendering and bit-allocation effects, to decide whether exp-1's +66% CB-vs-IQ2_S gap is a fixable encoder/bpp deficit or a fundamental FP4-grid ceiling.

- Sources scaled through the format's own `_scale_and_vectorize(grid='fp4')` (post NVFP4 group-16 domain, std ~2.9, absmax <= 6).
- Held-out: codebooks trained on one split, scored on a disjoint split.
- Weighted MSE weight = per-lane variance imatrix stand-in (normalised mean 1); Lloyd 20 iters, seed 1234, N_synth=1048576 vec/split.

## Part 1 — FP4-grid TAX (full signed 8-dim codebooks, format's full/product mode)

Weighted MSE (unweighted in parens). **tax = MSE(fp4-grid B) / MSE(unconstrained A)** — the pure cost of forcing centroids onto E2M1. C = fp8-grid ceiling. Codeword encodes direction+magnitude (no extra per-vector scale), exactly as full mode ships.

| source | m (2^m cb) | A unconstrained | B fp4-grid | C fp8-grid | **FP4 tax B/A** | fp8 tax C/A |
|---|---|---|---|---|---|---|
| Gaussian N(0,1) | 5 | 4.4283 (4.4283) | 4.5156 (4.5156) | 4.4407 (4.4406) | **+2.0%** | +0.3% |
| Gaussian N(0,1) | 6 | 3.7587 (3.7587) | 3.8708 (3.8708) | 3.7738 (3.7738) | **+3.0%** | +0.4% |
| Gaussian N(0,1) | 7 | 3.1786 (3.1786) | 3.3062 (3.3062) | 3.1967 (3.1967) | **+4.0%** | +0.6% |
| Gaussian N(0,1) | 8 | 2.6883 (2.6883) | 2.8157 (2.8157) | 2.7064 (2.7064) | **+4.7%** | +0.7% |
| Student-t (df=4) | 5 | 3.3830 (3.3831) | 3.5134 (3.5135) | 3.3892 (3.3892) | **+3.9%** | +0.2% |
| Student-t (df=4) | 6 | 2.9356 (2.9356) | 3.0376 (3.0376) | 2.9436 (2.9436) | **+3.5%** | +0.3% |
| Student-t (df=4) | 7 | 2.4987 (2.4988) | 2.5988 (2.5988) | 2.5113 (2.5113) | **+4.0%** | +0.5% |
| Student-t (df=4) | 8 | 2.1104 (2.1104) | 2.2345 (2.2345) | 2.1262 (2.1262) | **+5.9%** | +0.7% |
| Qwen3-0.6B (real weights) | 5 | 4.3406 (4.3407) | 4.4362 (4.4362) | 4.3527 (4.3528) | **+2.2%** | +0.3% |
| Qwen3-0.6B (real weights) | 6 | 3.6862 (3.6863) | 3.7982 (3.7981) | 3.6996 (3.6998) | **+3.0%** | +0.4% |
| Qwen3-0.6B (real weights) | 7 | 3.1129 (3.1130) | 3.2326 (3.2326) | 3.1320 (3.1320) | **+3.8%** | +0.6% |
| Qwen3-0.6B (real weights) | 8 | 2.6388 (2.6389) | 2.7588 (2.7587) | 2.6537 (2.6537) | **+4.5%** | +0.6% |
| Laplace | 5 | 3.1523 (3.1522) | 3.3514 (3.3513) | 3.1635 (3.1635) | **+6.3%** | +0.4% |
| Laplace | 6 | 2.7202 (2.7203) | 2.8501 (2.8501) | 2.7387 (2.7387) | **+4.8%** | +0.7% |
| Laplace | 7 | 2.2635 (2.2635) | 2.4429 (2.4429) | 2.2774 (2.2774) | **+7.9%** | +0.6% |
| Laplace | 8 | 1.9274 (1.9274) | 2.0826 (2.0826) | 1.9390 (1.9390) | **+8.1%** | +0.6% |

Scalar NVFP4 RTN per-coordinate (d=1, "no vector coding", 4 bit/coord) weighted MSE: Gaussian N(0,1) 0.0800, Student-t (df=4) 0.0609, Qwen3-0.6B (real weights) 0.0778, Laplace 0.0588.

## Part 2 — CB-vs-IQ CODEBOOK gap (signed mode, per-vector optimal scale, matched codebook SIZE)

Magnitude codebooks (8 explicit signs + m-bit magnitude table — the format's signed mode, and what IQ2 does). Scored with the per-vector WLS-optimal scalar scale (**scale-invariant**: compares pure codebook SHAPE, moots IQ's foreign scale normalisation). Weighted MSE.

| source | m | A_mag unconstr | B_mag fp4-grid | IQ2 (native) | IQ bpw | **tax B/A** | **CB-vs-IQ B/IQ** | A/IQ |
|---|---|---|---|---|---|---|---|---|
| Gaussian N(0,1) | 5 | 1.0707 | 1.1560 | — | — | **+8.0%** | — | — |
| Gaussian N(0,1) | 6 | 0.8598 | 0.9514 | — | — | **+10.6%** | — | — |
| Gaussian N(0,1) | 7 | 0.7006 | 0.7775 | — | — | **+11.0%** | — | — |
| Gaussian N(0,1) | 8 | 0.5750 | 0.6324 | 0.6217 (IQ2_XXS) | 2.062 | **+10.0%** | **+1.7%** | -7.5% |
| Gaussian N(0,1) | 9 | 0.4729 | 0.5157 | 0.4777 (IQ2_XS) | 2.312 | **+9.0%** | **+8.0%** | -1.0% |
| Gaussian N(0,1) | 10 | 0.3882 | 0.4234 | 0.3829 (IQ2_S) | 2.562 | **+9.1%** | **+10.6%** | +1.4% |
| Student-t (df=4) | 5 | 0.8655 | 0.9498 | — | — | **+9.7%** | — | — |
| Student-t (df=4) | 6 | 0.6946 | 0.7694 | — | — | **+10.8%** | — | — |
| Student-t (df=4) | 7 | 0.5679 | 0.6297 | — | — | **+10.9%** | — | — |
| Student-t (df=4) | 8 | 0.4656 | 0.5121 | 0.4965 (IQ2_XXS) | 2.062 | **+10.0%** | **+3.1%** | -6.2% |
| Student-t (df=4) | 9 | 0.3802 | 0.4177 | 0.3898 (IQ2_XS) | 2.312 | **+9.9%** | **+7.1%** | -2.5% |
| Student-t (df=4) | 10 | 0.3125 | 0.3396 | 0.3290 (IQ2_S) | 2.562 | **+8.7%** | **+3.2%** | -5.0% |
| Qwen3-0.6B (real weights) | 5 | 1.0455 | 1.1359 | — | — | **+8.6%** | — | — |
| Qwen3-0.6B (real weights) | 6 | 0.8439 | 0.9302 | — | — | **+10.2%** | — | — |
| Qwen3-0.6B (real weights) | 7 | 0.6913 | 0.7664 | — | — | **+10.9%** | — | — |
| Qwen3-0.6B (real weights) | 8 | 0.5680 | 0.6205 | 0.6074 (IQ2_XXS) | 2.062 | **+9.2%** | **+2.1%** | -6.5% |
| Qwen3-0.6B (real weights) | 9 | 0.4663 | 0.5082 | 0.4679 (IQ2_XS) | 2.312 | **+9.0%** | **+8.6%** | -0.3% |
| Qwen3-0.6B (real weights) | 10 | 0.3826 | 0.4149 | 0.3767 (IQ2_S) | 2.562 | **+8.4%** | **+10.1%** | +1.6% |
| Laplace | 5 | 0.8520 | 0.9658 | — | — | **+13.4%** | — | — |
| Laplace | 6 | 0.6921 | 0.7779 | — | — | **+12.4%** | — | — |
| Laplace | 7 | 0.5591 | 0.6191 | — | — | **+10.7%** | — | — |
| Laplace | 8 | 0.4574 | 0.5037 | 0.4966 (IQ2_XXS) | 2.062 | **+10.1%** | **+1.4%** | -7.9% |
| Laplace | 9 | 0.3707 | 0.4081 | 0.3884 (IQ2_XS) | 2.312 | **+10.1%** | **+5.1%** | -4.6% |
| Laplace | 10 | 0.3024 | 0.3303 | 0.3341 (IQ2_S) | 2.562 | **+9.2%** | **-1.1%** | -9.5% |

FP4-CB signed-mode bpw at matched codebook size = (8 signs + m index)/8 + 0.5 (NVFP4 group scale). m=8 -> 2.50, m=9 -> 2.625, m=10 -> 2.75 bpw, vs IQ2_XXS 2.063 / IQ2_XS 2.313 / IQ2_S 2.563 — the format spends ~0.2-0.45 bpw MORE per matched-size codebook (explicit signs + the +0.5 NVFP4 scale vs IQ's shared 7-bit ksigns / super-scale packing).

## Part 3 — IQ3 (4-dim grids) on 4-dim sub-vectors

| source | IQ3 fmt (m) | IQ3 | B_mag4 fp4-grid | A_mag4 unconstr | B/IQ | A/IQ |
|---|---|---|---|---|---|---|
| Gaussian N(0,1) | IQ3_XXS (8) | 0.0804 | 0.0873 | 0.0799 | +8.5% | -0.7% |
| Gaussian N(0,1) | IQ3_S (9) | 0.0481 | 0.0558 | 0.0509 | +16.0% | +5.8% |
| Student-t (df=4) | IQ3_XXS (8) | 0.0677 | 0.0697 | 0.0660 | +2.9% | -2.5% |
| Student-t (df=4) | IQ3_S (9) | 0.0388 | 0.0461 | 0.0424 | +18.9% | +9.2% |
| Qwen3-0.6B (real weights) | IQ3_XXS (8) | 0.0789 | 0.0860 | 0.0812 | +9.1% | +2.9% |
| Qwen3-0.6B (real weights) | IQ3_S (9) | 0.0470 | 0.0573 | 0.0505 | +21.9% | +7.6% |
| Laplace | IQ3_XXS (8) | 0.0678 | 0.0670 | 0.0664 | -1.2% | -2.1% |
| Laplace | IQ3_S (9) | 0.0380 | 0.0435 | 0.0418 | +14.6% | +10.0% |

## VERDICT

Mean FP4-grid tax (full mode, all sources/m) = **+4.5%**; (signed/magnitude mode) = **+10.0%**. Mean CB-vs-IQ2 at matched size B_mag/IQ = **+5.0%**; unconstrained A_mag/IQ = **-4.0%**.

1. **Is the FP4-grid tax small or large?** SMALL (<15%) — the grid constraint is NOT the ceiling; the exp-1 gap is bit-budget / encoder, which is fixable.
2. **Does IQ's codebook beat FP4-grid-Lloyd at matched size?** Yes, by +5.0% (matched codebook size). a LEARNED magnitude codebook beats IQ's fixed grid at matched size (A_mag/IQ -4.0%), so IQ's exp-1 edge was mostly its larger BIT BUDGET, not a codebook-design moat.
3. **Does the empirical source agree with synthetic?** Empirical FP4 tax at m=8 (full) = +4.5%; the tax and CB-vs-IQ columns track the synthetic sources across the table — the real-weight source corroborates, so this is not a Gaussian-only artifact.
4. **What does this predict for the corrected exp-1 rerun (signed mode + learned codebook, matched BYTES)?** The grid and codebook-design axes are cheap, but the NVFP4-tile PACKAGING is not free: at matched codebook SIZE, FP4-CB signed mode costs ~0.19-0.44 bpw MORE than its IQ2 twin (8 explicit sign bits so decoded tiles are literally NVFP4, + the mandatory 0.5-bpw group-16 E4M3 scale the FP4 tensor core consumes, vs IQ's amortised per-256 super-scale). To hit IQ2_S's 2.562 bpw a signed FP4-CB can only afford m~8.5 magnitude bits (vs IQ2_S's 10) — ~4x fewer shapes. Extrapolated at matched BYTES, a PERFECT-encoder FP4-CB still trails IQ2_S by **~+44% weighted MSE** (per source: gaussian +50%, t4 +41%, empirical +50%, laplace +36%).

**Bottom line.** exp-1's +66% is NOT an FP4-grid ceiling and NOT a codebook-design deficit: at matched size FP4-grid-Lloyd ~= IQ2 and both ~= the unconstrained optimum. Correcting the encoder (signed mode + learned codebook) will NARROW the gap, but a structural **bpp** ceiling remains — the price of NVFP4-tensor-core-compatible tiles (explicit signs + the 0.5-bpw group-16 scale) is ~0.2-0.45 bpw, which at ~2.5 bpp targets forces a ~4x smaller codebook and leaves ~40-55% residual MSE vs IQ2_S at matched bytes. The format is viable ONLY if free FP4 serving is judged worth ~0.5 bpw; it will not match IQ at matched bytes. Caveat: MSE is the coding-theoretic distortion proxy, not served KL — the per-vector-scale shape metric is symmetric across FP4/IQ but absolute values are optimistic vs the real per-16 scale; re-confirm the matched-bytes prediction on the corrected exp-1 served/emulated KL.

---

## Reviewer correction (Opus main-loop, session 2) — the packaging tax is ~0.19 bpw, not 0.35–0.5, and it is MITIGABLE

The verdict's load-bearing number (the "~0.2–0.45 bpw" NVFP4-tile packaging tax
driving the ~+44% matched-bytes residual) overstates the scale component.
Verified from the actual IQ2_S block layout (`block_iq2_s` = 2 B fp16 super-`d`
+ 64 B `qs` + 8 B `qh` + 8 B `scales` = 82 B/256 = 2.5625 bpw):

- **NVFP4-CB scale overhead**: 8-bit E4M3 per 16 = **0.500 bpw** (mandatory for
  the CUTLASS tensor-core contract).
- **IQ2_S scale overhead**: fp16 super-`d` (0.0625) + per-32 `scales` (0.250) =
  **0.3125 bpw**.
- **Real scale-packaging gap = 0.500 − 0.3125 ≈ 0.19 bpw**, NOT 0.35–0.5. (The
  sign-bit framing double-counts: IQ *also* stores signs — in `qh`/ksigns — so
  signs are not a net CB disadvantage; the honest delta is scales only.)

So the matched-bytes residual is materially smaller than +44% (the empirical
rerun is the arbiter). More important: **the 0.5-bpw group-16 E4M3 scale is the
SIMPLE choice, not an immutable one.** The mitigation already sketched in
PLAN.md applies directly here — ship a two-tier scale (fp16 super per 256 + a
cheap 4-bit sub per 16, IQ's own trick) and reconstruct the E4M3-per-16 plane in
the kernel prologue/at load. Scales are tiny, so this does NOT hit the INV-1
weight-expansion trap. With IQ-parity scale packaging + the cheap FP4-grid value
tax (+4.5%), NVFP4-CB could **match IQ at matched bytes AND serve native FP4** —
the "viable only if free FP4 serving is worth ~0.5 bpw" conclusion is too
pessimistic by roughly the mitigation's worth. Net: the format is more viable
than this study concluded; the decision hinges on the empirical rerun's real-KL
matched-bytes number and on whether the two-tier scale is worth its kernel cost.
On 295B, 0.19 bpw ≈ +7 GB (not +18) — much less likely to break single-Spark Hy3.
