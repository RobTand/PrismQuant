# Eliminating the gridbook serving tax (12% decode / 30% prefill)

2026-07-22. Measured baseline (27B, matched 23 GB, same protocol, FDO):
AURA conventional 1,737 prefill / 10.4 decode vs CB 1,220 / 9.2 — the tax
Robert flags as untenable for many users. Decomposition and the levers.

## 1. Where the decode 12% actually goes

At matched bytes both artifacts read identical weight bytes/token, so the
gap is pure per-weight COMPUTE in the GEMV inner loop. Per weight today:
unpack (shift/mask) → LUT gather → **multiply by scale → round to bf16**
→ FMA. The Hy3 ncu data already convicted this chain: SM 71% vs memory 44%
— the decode is compute-bound under a bandwidth-bound workload, and the
bit-exact decode contract (w = bf16(val × scale)) is what freezes it.

**The key algebra: the scale factors OUT of the dot product.**
Σ_j bf16(val_j·sc)·x_j is the only reason sc lives in the hot loop; without
the per-weight round, Σ_j (val_j·sc)·x_j = sc·Σ_j val_j·x_j — one multiply
per output row (fp8-CB, per-channel) or per 16-weight group (fp4-CB,
two-tier) instead of per weight. The entire scale chain and the round
leave the inner loop; what remains is unpack + gather + FMA, which is the
same work-per-byte class as a conventional dequant kernel.

**The contract is already asymmetric.** The prefill path (expand e4m3
values → cutlass W8A8) applies the per-channel scale in the f32 epilogue
and never bakes the per-weight bf16 round — GEMV and prefill differ at the
ulp level TODAY. "Decode contract v2" (no per-weight round anywhere; scale
in the epilogue everywhere) therefore makes the format MORE internally
consistent, not less, while unlocking the hoist.

- Change class: format-spec revision (v2 decode contract), version-flagged
  in quant_config; old artifacts decode bit-identically under v1 readers.
- Expected recovery: most of the 12%. The remaining unpack+gather cost is
  small and partially hidden under the memory latency the loop already
  waits on.
- Risk: numerics shift ~1 bf16 ulp/weight vs v1 → gate with the standard
  served logprob A/B + KL re-verify on an existing artifact re-served
  under v2 kernels (weights unchanged — only the decode math relaxes).
- Effort: days, not weeks — the kernels get SIMPLER (delete ops from the
  hot loop), and cb_fma_x/expand paths already share structure.

## 2. Where the prefill 30% goes, and the two answers

The tax is the transient expand: write the [N,K] tile to HBM, GEMM reads
it back. Two independent fixes:

1. **Persistent-N fused kernel (§4b, GO recorded)** — decode B once per
   N-tile inside the CUTLASS mainloop; the expand write and the GEMM's
   B-read both disappear. This is the designed on-Spark answer; 2–4 GPU
   days; the mid-M niche (17..128) is already wired opt-in.
2. **The 5090 calculus inverts on its own.** The 30% is a 273 GB/s
   unified-memory number: on the Spark, large-M GEMM is partially
   memory-bound, so the expander has no spare bandwidth to hide in (the
   measured 0.74–0.79× overlap verdict). A 5090 has ~1.8 TB/s and a
   compute-bound large-M GEMM — the same expand traffic fits in the
   GEMM's bandwidth shadow, and stream-overlapped expand (layer i+1
   expands while layer i GEMMs) should hide most of the tax with NO new
   CUTLASS work. Unmeasured (no 5090 on-box) — flag on the card, verify
   with the first community 5090 report or a cloud instance.

## 3. The framing that matters for "untenable"

The tax is quoted iso-SIZE. Users choose iso-QUALITY: the 19.93 GB CB
artifact already beats the 23 GB conventional one on KL (0.0082 vs 0.0130)
while reading 13% fewer bytes per token — at iso-quality, CB decode is
speed-NEUTRAL-or-better TODAY, and only the prefill tax genuinely remains.
Cards and the launch post should present both framings; iso-size-only
understates the format.

## 4. Recommended sequence

1. Decode contract v2 (scale-epilogue hoist, round removal) — biggest
   win-per-effort, kernels get simpler, format becomes self-consistent.
   Gate: logprob A/B + KL re-verify + the full GPU battery re-anchored to
   a v2 reconstruct reference.
2. Persistent-N §4b in the next dedicated GPU window — closes prefill on
   Spark-class bandwidth.
3. Stream-overlap expand as the cheap consumer-Blackwell mitigation and
   the measurement that sizes how much of §4b consumer cards even need.
4. Re-run the matched-size perf table after (1); update cards with both
   iso-size and iso-quality rows.

> **OUTCOME 2026-07-26 — items 1 and 2 were run and both came back non-positive;
> the tax fell to different levers.**
>
> | Item | Verdict |
> |---|---|
> | 1. Decode contract v2 (scale-epilogue hoist) | **MEASURED NULL** on the served 27B: 10.10 vs 10.13 tok/s, quality-neutral. Decode is bandwidth-bound at per-byte parity, so there is nothing for the hoist to recover. Default stays v1; v2 supported via `PRISMAQUANT_CB_DECODE_CONTRACT=v2` (`STANDARDS.md:61-64`, commit `d924d76`). |
> | 2. Persistent-N §4b | **MEASURED NEGATIVE** for dense: 2–5.7× slower than expand+fork at 27B shapes once the CUDA expander had shrunk the dense expand tax to ~10%. Quarantined behind `PRISMAQUANT_ENABLE_PTC=1` (`STANDARDS.md:55`; see `persistent-n-prefill.md` §7 OUTCOME). |
> | 3. Stream-overlap expand | Previously rejected on the dense path: N-chunked expand+GEMM overlap measured 0.46× and is **not** bit-exact (`cutlass_scaled_mm` reconfigures on narrow N) — `linear.py:53-59`. |
>
> What actually removed the tax was elsewhere: the mid-M CUTLASS fused
> decode-in-prologue (default since 2026-07-26, 1.40× in-niche), the measured
> per-layer MoE prefill selector (`auto`), and the R6 smem-resident codebook LUT.
> Laguna-class MoE prefill went 293 → 2,186 tok/s over that campaign. Item 4's
> iso-size/iso-quality framing (§3) still stands and is the right one for cards.
