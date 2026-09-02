# Persistent-N large-M CB prefill (Task 9)

Status: **design + INV-1 reference kernel + opportunity-sizing bench** (round 3,
write-only). The tensor-core (INV-2) endgame kernel is specified here and gated
on the opportunity-sizing measurement below. `cb_prefill_persistent_n_fp8`
(plain-CUDA reference) is env-gated off and exists to validate the schedule and
INV-1 compliance, **not** to beat cuBLAS.

## 1. Why this exists — the two shipping prefill paths and what's wrong

The decode GEMV (`cb_gemv.cu`) owns M ≤ 16 (decode). Prefill (large M) has two
implemented answers, both flawed:

- **Fused decode-in-prologue** (`cb_fused_gemm.cu`, `cb_fused_prefill_mm`): one
  CUTLASS CTA per output tile decodes the B superblocks it needs, in smem, then
  runs the FP4/FP8 tensor-core MMA. Honors INV-1 (no `[N,K]` in HBM) and INV-2
  (tensor cores), bit-exact — but **every M-tile CTA re-decodes the same B
  tiles**. At M=1400 that is ⌈M/128⌉ ≈ 11× redundant decode; measured 0.22× of
  serial (36 vs 8.1 ms). Its honest niche is mid-M (17…128, one M-tile → no
  redundancy), where it wins 1.04–1.45×.

- **Transient-expand + cuBLAS/fork GEMM** (the current large-M shipping answer):
  `cb_expand_fp8` decodes the whole `[N,K]` to an HBM e4m3 tile, then a dense
  fp8 GEMM consumes it. Decode is paid once, so it beats the fused kernel at
  large M — **but it materializes `[N,K]` in HBM, violating INV-1**, and pays a
  full HBM round-trip on the decoded bytes (write `N·K` in the expander, read
  `N·K` in the GEMM). On the 273 GB/s unified-memory part that round-trip is not
  free; the chunked-overlap attempt (expand on a side stream) only reached
  0.74–0.79× of serial because the M=1400 GEMM is already partly memory-bound
  and left the expander no spare bandwidth to hide in.

**Persistent-N is the schedule that removes both flaws:** decode each B N-tile
**once**, keep it in on-chip staging (smem, or a bounded L2-resident scratch),
and **stream the whole M dimension through it**. Decode amortizes over M like
transient-expand, but the decoded bytes never fully materialize in HBM (INV-1),
and the GEMM reads B from smem/L2 instead of from an HBM `[N,K]` tile.

## 2. The scheduling tension (why this is a kernel-layer restructure)

A weight-stationary GEMM wants, per N-tile: decode B once, reuse across all M.
Three ways to arrange the loops, each with a cost:

| loop nest (per N-tile) | decode | accumulator | B-resident need |
|---|---|---|---|
| **M-outer, K-inner, decode-in-K** (the fused kernel) | ⌈M/TileM⌉× redundant | `TileM×TileN` regs | one `TileN×TileK` smem tile |
| **K-outer, M-inner** (decode once per K-tile) | 1× ✓ | **`M×TileN` in HBM, RMW per K-tile** ✗ | one `TileN×TileK` smem tile |
| **M-outer, full-K B resident** (this design) | 1× ✓ | `TileM×TileN` regs ✓ | **`TileN×K` resident** |

The first re-decodes; the second's HBM accumulator RMW (n_ktiles × M × TileN ×
8 B) dwarfs the decode saving at large N. The third is the only one with both
decode-once **and** a register accumulator — its price is holding **all of K for
the N-tile** in staging:

- `B[TileN, K]` as **e4m3 bytes** (the decoded CB values are on the e4m3 grid) =
  `TileN·K` bytes. For K=4096: TileN=16 → 64 KB (no opt-in); TileN=32 → 128 KB
  (needs `cudaFuncAttributeMaxDynamicSharedMemorySize`, ~1 CTA/SM). Skinny TileN
  is forced by the smem budget.
- **Or** decode `B[TileN, :]` into a bounded per-CTA **HBM scratch** (`TileN·K`
  bytes, e.g. 512 KB for TileN=128) that stays L2-resident for the M-sweep — a
  larger TileN (better MMA-N efficiency) at the cost of L2 pressure across
  concurrent CTAs. This is the TileN-vs-L2 knob the endgame kernel sweeps.

Skinny TileN (16/32) trades MMA-N efficiency for smem residency; the L2-scratch
variant trades L2 pressure for a fat TileN. Which wins is exactly what §5 sizes.

## 3. Data-movement accounting (the ceiling)

Let the packed stream be `p·N·K` bytes (p = k_bits/8/… ; ≈0.69 B/weight at
k44 = 5.5 bits) and the decoded e4m3 be `1·N·K` bytes. HBM traffic:

- **transient-expand:** read packed `p·N·K` (expander) + write decoded `N·K`
  (expander) + read decoded `N·K` (GEMM B) + read A `M·K` + write D `M·N`
  = `(p + 2)·N·K + M·K + M·N`.
- **persistent-N (smem-resident B):** read packed `p·N·K` (once) + read A `M·K`
  + write D `M·N` = `p·N·K + M·K + M·N`.

Persistent-N removes **`2·N·K`** bytes of HBM traffic (the decoded write + read).
That is the *ceiling* of what the schedule buys. Whether it beats transient
end-to-end depends on (a) how much of transient's time is that `2·N·K` vs the
tensor-core GEMM, and (b) whether skinny-TileN MMA keeps tensor-core efficiency.
**§5's opportunity bench measures (a) directly with existing kernels — that is
the go/no-go, and it can be a clean negative without building the endgame.**

## 4. Implementations

### 4a. INV-1 reference — `cb_prefill_persistent_n_fp8` (this round, plain CUDA)
- Persistent grid (`gridDim.x` = SM count × k), grid-stride over N-tiles of
  `TILE_N` (=16, smem-resident full-K).
- **Phase 1:** the block cooperatively decodes packed `B[n0:n0+TILE_N, :K]` →
  `smem sB[TILE_N][K]` e4m3 bytes (identical codeword extraction + LUT gather to
  `cb_expand_fp8`, so bit-exact decode) and loads `scale[TILE_N]`. Once.
- **Phase 2:** stream M in `TILE_M` tiles; a register-blocked FMA GEMM reads A
  from gmem and B from smem, applies the per-column scale, writes D.
- **Honors INV-1** (B never in HBM). **Does NOT honor INV-2** (FMA, not FP4/FP8
  tensor cores) — so it is a *schedule + correctness reference and a decode-
  amortization demonstrator*, not a cuBLAS competitor. Parity gate: `≤1 bf16
  ULP + norm backstop` vs `cb_expand_fp8 → F.linear` (the decode is bit-exact;
  the FMA reassociates vs cuBLAS). Env-gated: only built/run via the test+bench.

### 4b. INV-2 endgame — the CUTLASS persistent-N kernel (plan, not this round)
Reuse the **passthrough collective** (`sm120_cb_mma_tma.hpp`, the bit-exact
fork64 reference that already runs the FP8 tensor-core MMA on dense e4m3 B).
Restructure the kernel layer (not just the tile scheduler):
1. Persistent CTA owns an N-tile; a static tile scheduler grid-strides N.
2. **Phase 1** decodes `B[TileN, :]` → either smem (skinny TileN) or a bounded
   per-CTA HBM scratch (fat TileN, L2-resident), reusing the fused kernel's
   `decode_stage` (bit-exact).
3. **Phase 2** drives the passthrough collective's `mma()` over the M-tiles
   against the resident decoded B — decode is hoisted out of the M-loop, so it
   is paid once per N-tile. Because Phase 2 is the *unchanged* passthrough MMA,
   the result is bit-exact vs `fork64` per N-panel (the existing gate).
This is the multi-iteration piece that needs a GPU-enabled session (TMA
descriptors, warp-specialized pipeline, f8f6f4 fragment plumbing, NamedBarrier
ordering across the decode/GEMM phases). **Build it only if §5 says the ceiling
is worth it.**

## 5. Opportunity-sizing — the measurement that decides (runnable now)

`bench_prefill_opportunity.py` decomposes the transient path with the **existing
proven kernels** at real 27B prefill shapes (N,K) × M ∈ {256,512,1024,1400,2048}:
- `t_expand` = time of `cb_expand_fp8` alone (the `p·N·K` read + `N·K` write).
- `t_gemm`   = time of `sm120_fp8_mm_fork(A, W_dense)` alone (the tensor-core GEMM
  reading the decoded `N·K` + A, writing D).
- `t_serial = t_expand + t_gemm` (the shipping large-M answer).

Decision criterion:
- The persistent-N **ceiling** removes the expander's `N·K` write and the GEMM's
  `N·K` decoded-B read. Estimate the removable fraction as
  `f = (t_expand + t_gemm_Bread) / t_serial`, where `t_gemm_Bread ≈ (N·K)/(N·K +
  M·K + M·N) · t_gemm` (the B-read share of the GEMM's HBM traffic).
- **If `f` (the ceiling speedup headroom) is small** (say < ~15% at the target
  M), the fused tensor-core kernel cannot beat transient by enough to justify a
  CUTLASS restructure → **clean negative, keep transient-expand as final.**
- **If `f` is large**, build §4b; the reference kernel (§4a) already confirms the
  schedule + INV-1 compliance, and its decode-amortization (`t_expand` folded
  into the GEMM, paid once) is the mechanism §4b makes tensor-core-fast.

The reference kernel's own bench (`bench_persistent_n.py`) reports (i) parity
PASS/FAIL and (ii) its weight-bytes/s to confirm decode is paid once (packed
stream read `p·N·K`, no decoded HBM materialization) — a schedule sanity, not a
perf claim.

## 6. GPU-window run order
1. `bench_prefill_opportunity.py` → the go/no-go number `f` per shape×M.
2. `test_persistent_prefill.py` → parity of the reference kernel (schedule + INV-1
   correct).
3. `bench_persistent_n.py` → reference kernel decode-amortization sanity.
4. **If `f` says go:** implement §4b in a GPU-enabled session and A/B vs `t_serial`.

## 7. §4b concrete implementation plan (2026-07-21 — the remaining kernel)

Status: §5's opportunity bench said **GO** (recorded 2026-07-20,
prod_hy3_results.md "persistent-N = GO-as-roadmap"); §4a's reference kernel
is parity-green. What remains is the tensor-core §4b build, a bounded
CUTLASS project for the next dedicated GPU window:

1. **Scheduler**: CUTLASS 3.x persistent tile scheduler, specialized to
   N-MAJOR visit order — each persistent CTA takes a fixed N-tile and
   iterates ALL M-tiles before advancing (the opposite of the default
   M-major raster). Decode-in-prologue then fires once per N-tile visit:
   the existing sm120_cb_fused_mma.hpp prologue is reused verbatim; only
   the "is this the first M-tile of my N-tile" predicate gates it, and the
   decoded B smem allocation persists across the CTA's M-loop (no
   per-tile re-stage).
2. **Bit-exactness gates** (all exist): fork64 passthrough parity per
   N-panel; test_fused_prefill's synth/ragged/real-artifact equality; the
   served logprob A/B before any default flip.
3. **Dispatch**: extends the mid-M fused path (PRISMAQUANT_CB_FUSED_MIDM)
   upward — M > 128 routes to the persistent variant, mid-M keeps the
   single-tile kernel, M <= 16 stays on the GEMV. Same rung coverage
   (KBits template list) as the mid-M kernel; widen both together if the
   allocator ever picks non-step-4 fp8 rungs hot.
4. **Estimate**: 2-4 focused GPU days (TMA descriptors, warp-specialized
   pipeline, NamedBarrier ordering across the persistent M-loop). The
   ceiling is bounded by §5's f (expander write + GEMM B-read removal);
   re-run bench_prefill_opportunity.py at the CURRENT serve shapes first
   — if f moved below ~15%, record the clean negative instead.

Everything else in the kernel surface is DONE as of 2026-07-21: all-integer
product rungs (ceil-first uneven splits, encoder-anchored), signed S-rung
decode (CUDA + Triton, GPU battery chained), the M-branch-hoist dispatch
ops, the mid-M fused niche (opt-in), w2 round-2 schedule (rowpack measured
negative), and the capture-safe stock-MoE prefill (opt-in).

> **OUTCOME 2026-07-26 — §4b was built and MEASURED NEGATIVE for the dense
> lane.** `csrc/cb_persistent_tc.cu` is parity-green but **2–5.7× slower** than
> expand+fork at 27B shapes: the CUDA expander (landed after this plan was
> written) had already cut the dense expand tax to ~10%, so the `f` this section
> tells you to re-check had collapsed — exactly the "record the clean negative"
> branch of step 4. Recorded in `STANDARDS.md:55`, commit `d924d76`. The kernel
> is **quarantined, not deleted**: `PRISMAQUANT_ENABLE_PTC=1` +
> `PRISMAQUANT_CB_PREFILL_DENSE=persistent` (`linear.py:450-476`,
> `cuda_ext.py:165`), kept as the schedule reference. Item 3's dispatch note is
> also stale — mid-M fused is ON by default (`linear.py:439`).
>
> The **MoE** persistent/grouped decode-in-mainloop target is untouched by this
> negative and remains the fat one (`STANDARDS.md:56`: expand ≈ 35% of a Laguna
> MoE layer). Do not re-derive the dense case from this page's `f`.
