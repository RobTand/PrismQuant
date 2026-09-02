# NVFP4-CB Phase-0 Experiment 1 (+2) — Qwen3-0.6B results

> **This is the EMULATION gate, not the served metric.** Whole-model emulated forward KL-vs-BF16 (fp32, held-out WikiText, seqlen 512 × 8192 tokens, W4A4/W8A8 activation buckets emulated). A kernel phase MUST re-confirm the winner on true served vLLM/llama.cpp KL before any promotion past Candidate.

- Model: `/home/rob/models/Qwen3-0.6B` · git `6c697bcbb9d28c03e9e1631ee02a662346b5b0e0`
- Calibration: `diverse-v1.jsonl` (4 draws, seeds 0–3, 32×1024 tok/draw) · eval: held-out `wiki.test.raw`
- Targets: 196 Linears (in_features%256==0). Excluded: 0 for in_features%256≠0, 1 lm_head.
- imatrix col_weights = E[x²] per column (llama.cpp convention); all arms in a seed share one paired draw.

## Per-arm results (kl_confident primary)

| Arm | mode/k | body bpw | total bpw | KL_conf mean±std | KL_all mean | top1 | n_swap |
|---|---|---|---|---|---|---|---|
| A_fixed_prod_k12 | 4sd | 2.000 | 2.000 | 11.1252±0.4953 | 10.1804 | 0.006 | 196 |
| A_fixed_prod_k13 | 4sd | 2.125 | 2.125 | 8.0096±0.6877 | 7.3533 | 0.042 | 196 |
| A_fixed_prod_k14 | 4sd | 2.250 | 2.250 | 5.2428±0.1836 | 4.8669 | 0.158 | 196 |
| B_fixed_full_k12 | 4sd | 2.000 | 2.000 | 8.3667±0.7578 | 7.7114 | 0.052 | 196 |
| B_fixed_full_k14 | 4sd | 2.250 | 2.250 | 3.7585±0.1304 | 3.5898 | 0.291 | 196 |
| C_learned_full_k12 | 4sd | 2.000 | 2.058 | 6.7569±0.5617 | 6.1790 | 0.092 | 196 |
| C_learned_full_k14 | 4sd | 2.250 | 2.483 | 2.6342±0.3688 | 2.5213 | 0.436 | 196 |
| D_iq2_s | 4sd | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| D_iq3_xxs | 4sd | 3.062 | 3.062 | 0.4085±0.0101 | 0.5354 | 0.838 | 196 |
| E_smooth_a0.25_k12 | 4sd | 2.000 | 2.000 | 10.3109±0.3033 | 9.3774 | 0.019 | 196 |
| E_smooth_a0.25_k14 | 4sd | 2.250 | 2.250 | 4.1173±0.1252 | 3.8615 | 0.250 | 196 |
| E_smooth_a0.5_k12 | 4sd | 2.000 | 2.000 | 11.4700±0.4465 | 10.4545 | 0.005 | 196 |
| E_smooth_a0.5_k14 | 4sd | 2.250 | 2.250 | 5.0906±0.3512 | 4.7609 | 0.179 | 196 |
| F_nvfp4 | 1sd | 4.500 | 4.500 | 0.2222±0.0000 | 0.3084 | 0.913 | 196 |
| F_fp8cb_k40 | 1sd | 5.000 | 5.025 | 0.1305±0.0000 | 0.1818 | 0.927 | 196 |

## Decision-gate verdicts

### Product-vs-full penalty (fixed lattice)

- k12: product 11.1252 vs full 8.3667 (Δ=+2.7585, +33.0%; max σ=0.7578) → **full better beyond noise**.
- k14: product 5.2428 vs full 3.7585 (Δ=+1.4844, +39.5%; max σ=0.1836) → **full better beyond noise**.

### Learned-vs-fixed (match-k, full mode)

- k12: fixed 8.3667 vs learned 6.7569 (learned Δ=+1.6097, +19.2%; max σ=0.7578) → **learned beats fixed beyond noise**.
- k14: fixed 3.7585 vs learned 2.6342 (learned Δ=+1.1243, +29.9%; max σ=0.3688) → **learned beats fixed beyond noise**.

**Match-bytes (learned sidecar as analytic curve over N).** Sidecar = 2^k·32/N bpw; shrinks ~50× from 0.6B→27B class:

| k | 0.6B (1e6) | 4B (6e6) | 27B (25e6) | 300B (1e8) |
|---|---|---|---|---|
| 12 | +0.131 | +0.022 | +0.005 | +0.0013 |
| 14 | +0.524 | +0.087 | +0.021 | +0.0052 |

At 0.6B the learned sidecar is a real byte penalty (+0.131 bpw at k12, +0.524 at k14 — see the total-bpw column); at 27B+ it is negligible (per-deployment gate). Note the near-matched-bytes reading this enables at 0.6B: learned-k14 TOTAL bpw is 2.483 vs IQ2_S 2.5625 (Δ −0.08 bpw) — the closest matched-bytes comparison in this experiment, and CB still loses it (see CB-vs-IQ below).

### CB-vs-IQ (native-FP4 thesis / >15% kill test)

- CB k14 (2.250 bpw) vs D_iq2_s (2.562 bpw), Δbpw=-0.312: KL_conf 2.6342 vs 1.5837 (+66.3%, σ=0.3688) → **CB worse by >15% at NEAREST bpw — kill-test FLAG, but the formal kill gate requires MATCHED bpw on BOTH models; not triggerable from this comparison alone**.
- CB k14 (2.250 bpw) vs D_iq3_xxs (3.062 bpw), Δbpw=-0.812: KL_conf 2.6342 vs 0.4085 (+544.9%, σ=0.3688) → **CB worse by >15% at NEAREST bpw — kill-test FLAG, but the formal kill gate requires MATCHED bpw on BOTH models; not triggerable from this comparison alone**.

**Near-matched-BYTES reading (0.6B, sidecar included):** learned-k14 at 2.483 total bpw vs IQ2_S at 2.562 (Δ −0.08 bpw): KL_conf 2.6342 vs 1.5837 (+66.3%). At 0.6B, CB loses the closest available matched-bytes comparison by well over 15% — a kill-test flag on ONE model; the formal kill gate additionally requires the 4B check (and at 27B-class tensor sizes the sidecar shrinks ~25×, moving learned-k14 to ~2.27 total bpw, where no IQ twin exists).

Honest confounds in this comparison: (1) index-only bpw is NOT matched — the flat-table CB ladder tops out at k14 = 2.25 bpw while IQ2_S is 2.5625 (Δ −0.3125 bpw in CB's favor) and IQ3_XXS is 3.0625 (no CB twin); (2) CB arms are measured in their served W4A4 activation bucket while IQ arms are weight-only — deliberate (each format is measured with its served activation behavior, per plan), but it means the KL gap is not a pure weight-codebook comparison (the decomposition diagnostic below bounds this at ~10% of CB KL).

### Smoothing sub-arm (α × k, gated on whole-model KL)

- k12 α=0.25: base 11.1252 → smoothed 10.3109 (Δ=+0.8142, +7.3%; σ=0.4953) → **smoothing helps beyond noise**.
- k12 α=0.5: base 11.1252 → smoothed 11.4700 (Δ=-0.3449, -3.1%; σ=0.4953) → **within between-seed noise**.
- k14 α=0.25: base 5.2428 → smoothed 4.1173 (Δ=+1.1255, +21.5%; σ=0.1836) → **smoothing helps beyond noise**.
- k14 α=0.5: base 5.2428 → smoothed 5.0906 (Δ=+0.1522, +2.9%; σ=0.3512) → **within between-seed noise**.

### Baseline anchors (single seed)

- F_nvfp4: body 4.500 bpw, KL_conf 0.2222, top1 0.913 (sanity anchor).
- F_fp8cb_k40: body 5.000 bpw, KL_conf 0.1305, top1 0.927 (sanity anchor).

## Exp-2 — index entropy (largest 8 Linears)

Redundancy k−H in bits per 8-weight vector; per-weight bpw recoverable = (k−H)/8. Gate: >0.25 bpw recoverable at k∈{12,14} on both models opens an entropy-coding investigation; else close the question.

| k | arm-A product Σ(k_sub−H) bits/vec | arm-C learned k−H bits/vec | recoverable bpw (max) | product cond-gain |
|---|---|---|---|---|
| 12 | 0.181 | 0.071 | 0.0226 | 0.017 |
| 14 | 0.230 | 0.075 | 0.0287 | 0.064 |

Exp-2 verdict: max recoverable rate 0.0287 bpw → **≪0.25 bpw recoverable — CLOSE the question (fixed-rate indexing is optimal; the expected result)**. Even reading the plan's gate as k−H in raw bits (0.23 max) it stays below 0.25.

The learned-arm first-order conditional gain (5–9 bits) is a small-sample ARTIFACT, not real serial correlation: with 2^k symbols the per-tensor pair histogram (~4×10^5 consecutive pairs over up to 2.7×10^8 cells) is massively undersampled, so H(idx_t|idx_{t-1}) is underestimated toward 0. The well-sampled product sub-streams (128–256 symbols) show the true serial correlation: 0.02–0.06 bits — negligible.

### Decomposition diagnostic (weight vs activation share)

B_fixed_full_k14 seed0 measured weight-only (act_emulation=False): KL_conf 3.2903 vs 3.7585 with the served W4A4 bucket → the activation bucket contributes ~10% of the CB KL at this rate; the weight codebook dominates. The CB-vs-IQ gap is therefore mostly real codebook/rate deficit, not the act-emulation asymmetry.

## Caveats

- **Measurement bug found & fixed during this experiment** (nvfp4_cb_formats._build_lattice): the fp4 fixed lattice was trained on standard N(0,1) samples while NVFP4 group-16 normalization yields normalized weights of std≈2.9/absmax≈6 — the mis-scaled codebook gave whole-model KL≈15 / top1≈0 and would have falsely killed the family. Fixed by training the lattice on genuinely NVFP4-normalized samples via the encoder's own _scale_and_vectorize (no hand-tuned constant); data/nvfp4_cb_lattices.pt regenerated. All numbers here are post-fix.
- Emulation gate only; 0.6B triage — a 4B scale-check and served re-confirm remain (the GGUF lane repeatedly saw 0.6B wins fail at 4B).
- Learned codebooks use CUDA weighted-Lloyd (float atomics can flip grid-snap ties across runs; per-seed noise, acceptable at Phase-0).
- CB-vs-IQ compares at nearest bpw; deltas are not exact-bpw matched (K13=2.125 has no exact IQ twin).
