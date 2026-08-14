# NVFP4-CB fused fp4-MMA prefill (dense + MoE) — landed OPT-IN, 2026-07-31

Status: **LANDED, OPT-IN, DEFAULT OFF.** Nothing here is promoted; promotion
requires a served A/B per `STANDARDS.md` (and doubly so here, because this
path changes the served activation bucket — see §4). Env gates:
`PRISMAQUANT_CB_FUSED_FP4` (dense: `1` = all prefill M, `midm` = 17–128),
`PRISMAQUANT_CB_FUSED_FP4_MOE` (MoE grouped: `1`/`128`, or `256` for the
TileM=256 arm). Unset ⇒ the shipping fp4 dispatch is byte-identical.

## 1. What was missing, and what this is

The fp4-CB ladder `NVFP4_CB_K12..K24` (2.0–3.28 bpw — the ladder every large
artifact ships on; Hy3-295B is 2.9 bpw and MoE) had **no tensor-core prefill
path at any M**: dense fp4 prefill was Triton bf16-expand → cuBLAS bf16, MoE
fp4 prefill was the per-expert loop. This lane adds the fp4 counterparts of
the fp8 fused campaign:

* `gridbook/csrc/cutlass_fork/sm120_cb_fused_fp4_mma.hpp` — a fork of the
  vendored CUTLASS 4.3.4 **block-scaled** sm120 mainloop (pristine copy:
  `sm120_blockscaled_mma_tma_orig.hpp`). The A side (packed-e2m1 activations
  + swizzled ue4m3 SFA, TMA-pipelined) is pristine; the entire B side is
  replaced by decode-in-prologue over the packed CB stream.
* `gridbook/csrc/cb_fused_fp4_gemm.cu` — entries `cb_fused_fp4_prefill_mm_scaled`
  (dense), `cb_fused_fp4_moe_grouped` (tile-indexed MoE grouping, TileM
  128/256), and `sm120_nvf4_mm_scaled` (the STOCK block-scaled collective at
  the identical config — the bit-exactness reference, i.e. the `fork64` role).
* Dispatch: `gridbook/linear.py` (`_try_fused_fp4`) and `gridbook/moe.py`
  (`_apply_prefill_grouped_fused_fp4`), both additive and eligibility-cached,
  falling through silently to the shipping paths on any miss.
* LUT/packers: `gridbook/codec.py` (`build_fp4_value_lut`, `build_compose_u8`,
  `swizzle_sf_plane`, `nvfp4_act_quant_ref`, e2m1 code helpers).

## 2. The SASS gate — the load-bearing verdict

On this chip the fp4 rate is only real through `kind::mxf4nvf4.block_scale`
(**OMMA.SF.16864**, k=64). The trap: `kind::f8f6f4` also accepts e2m1
operands but issues **QMMA k=32** — fp4 data at the fp8 rate, invisible to
every parity test. The gate is therefore disassembly, and it is a pinned test
(`tests/test_fused_fp4_prefill.py::test_sass_omma_16864_and_no_qmma`), not a
one-time check.

Verdict on the built module (all kernels: dense fused, MoE grouped ×2 tiles,
stock reference), `cuobjdump -sass`, CUDA 13.0, `sm_121a`:

```
256 × OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
  0 × QMMA / HMMA / IMMA
```

The same verdict holds for the object compiled inside `vllm-node:latest`
against its bundled CUTLASS 4.3.4 (the serving toolchain).

## 3. Architecture notes (deltas vs the fp8 fused fork)

* **No TMA for the packed stream — producer-warp manual staging.** fp4 rows
  have an ODD `type_size` (4k+9 v2 / 4k+16 v1), so the packed row stride can
  never satisfy TMA's 16-byte stride rule (the fp8 fork's `4k` sizes could).
  The producer warp stages, per K-tile, each row's half-superblock index
  window + 16 scale-plane bytes (aligned-u32 window loads over the misaligned
  source) into a 72 B/row smem stage. Ordering into the TMA mbarrier: plain
  stores → `__syncwarp` → the leader's `mbarrier.arrive.expect_tx`
  (release.cta) issued by the A/SFA TMA copies → `consumer_wait`
  (acquire.cta). This also makes MoE grouping trivial: the expert indirection
  is one pointer offset in the producer (`expert_ids[m_tile]`), nothing in
  the consumer.
* **Decode writes both MMA operands.** Values: the codebook is E2M1-grid-
  valued by construction (`nvfp4_cb_formats._snap_to_grid`), so decode is
  LUT-gather → raw nibbles into the swizzled `SmemLayoutB`. Scales: two-tier
  v2 compose is **exact e4m3 by construction** (two-tier-scale-spec §1.2), so
  an e4m3-byte compose table (4 KiB smem) yields the ue4m3 SFB operand
  directly; v1 plane bytes are already e4m3. The weight side of this kernel
  is therefore **lossless** — bit-identical to `nvfp4_cb_reconstruct`.
* **Runtime rung parameters.** Unlike the fp8 fork's 6-way `KBits` template
  dispatch, `k_bits/n_sub/type_size/scale-coding` are runtime: the packed
  stream never touches a TMA descriptor or a k-sized smem layout here. ONE
  kernel instantiation serves product K12–K24, signed S13–S16, v1 and v2.
* **R6 discipline inherited:** the value LUT (≤16 KiB even at k24) + compose
  table are smem-resident, staged once per CTA. `AssertSmemFits` gates every
  instantiation (fused 76,800 B; grouped-256 95,232 B; ceiling 101,376 B).
* **Two sub-byte landmines found by the smem debug dump** (kept, `ptr_debug`,
  test-only): (a) `TiledMma::ValTypeB` is `integer_subbyte<4,false>`, so
  assigning a `float_e2m1_t` NUMERICALLY converts (1.0→1, 6.0→6, negatives→0)
  — nibbles must be stored as raw subbyte integers; (b) the UMMA sub-byte
  atoms' `Swizzle<2,4,3>` operates on **byte** addresses (the
  `smem_ptr_flag_bits<4>` convention the LDSM read path honors), while
  element-wise writes through the position-independent tensor swizzle nibble
  offsets — a row-dependent 16/32-nibble block permutation. The decode
  therefore addresses manually: plain layout → nibble offset → byte offset →
  swizzle functor in byte space, full-byte stores, no sub-byte RMW.

## 4. Numerics — the honest part

* **Weight side: exact** (e2m1 values × e4m3 scales are the MMA operands).
* **Activation side: the bucket CHANGES.** The hardware SF operand is ue4m3;
  the Triton/transient fp4 paths' emulation bucket (`fp4_group16_act_qdq`)
  uses **fp32** group-16 scales, which are unrepresentable in the MMA. The
  fused paths quantize natively (per-tensor fp32 global × per-group-16 ue4m3
  SF, vLLM `scaled_fp4_quant`), and the fp32 residual is applied in the same
  fp32-EVT epilogue as the fp8 fused kernel
  (`bf16_rn(b_scale·(a_scale·acc))`).
* Measured on random Linears (k16, N=320, K=1536, M=128): fused-vs-Triton
  relative delta **≈ 7.5e-2** — the e4m3-SF snap, not a kernel defect (the
  same-bucket fp32 emulation agrees with the kernel to ≈1.6e-3, and the
  weight side is bit-exact). This is exactly the class of change that KL
  screens can misjudge: **the served KL A/B is the promotion arbiter**, which
  is why both gates default OFF.

## 5. Parity evidence (all pinned in `tests/test_fused_fp4_prefill.py`)

Run 2026-07-31 on GB10 (sm_121), host venv + CUTLASS 4.4.0 headers, and the
kernels also compile against the serving container's CUTLASS 4.3.4 (the two
files' upstream diff is cosmetic). 25 passed, 1 skipped (the vLLM-layout
cross-check needs in-container vLLM; run it before any served A/B).

* fused == stock NVF4 collective **bit-exact** (uint16 view equality):
  product k ∈ {12,13,16,18,20,24} v2, signed k ∈ {13,16} v2, product
  k ∈ {16,20} v1; M ∈ {32,64,128,512,2048}; ragged N/K (320/1536, 192/1024,
  104/768, 520/2560); padded-row-stride views.
* fused vs same-bucket fp32 emulation ≤ 1e-2 (measured ≈1.6e-3) everywhere.
* MoE grouped == per-expert dense **bit-exact** at TileM=128 and TileM=256,
  including multi-tile experts.
* Constraints made loud: `N % 8 == 0` (bf16-TMA epilogue), `K % 256 == 0`,
  ≥12 B storage tail slack on packed stacks (`codec.pad_qweight` provides it;
  the MoE dispatch re-points the stacked params at a slack-padded buffer once
  — no second resident copy, the issue-#1 discipline).

## 6. Benchmarks (release gate — measured 2026-07-31)

GB10 sm_121, box idle (only the untouched :8000 serve resident, 0% GPU util
verified before each run), **45 s sustained warmup**, SM clock read per block
(2.24–2.51 GHz across all blocks; bf16-GEMM proxy 83–86 TF steady — cold the
same GPU reads 4.7 TF, i.e. the cold-number trap is ~17×), cuda-event timing,
median-of-30 (15 at M=2048) × 3 blocks, spread = max−min of block medians
(reported per line in `scratch/fp4mma/bench_fp4_final.py` output; ≤3% and
usually ≤1%). Synthetic packed weights (decode cost is data-independent;
numerics separately pinned bit-exact by the test suite).

### 6.1 Dense: fused vs the two paths it replaces (kernel-path ms)

Baselines consume the same pre-QDQ'd activations the shipping dispatch feeds
them; the fused column consumes prepacked e2m1+SFA (same accounting; act-prep
measured separately below). k12/k16/k24 shown; k14/18/20 sit between.

| N×K | M | transient (ships, v2 M>16) | Triton decode | fused | × vs transient | × vs Triton |
|---|---|---|---|---|---|---|
| 4096×4096 | 32 | 0.384–0.448 | 0.222–0.415 | **0.072–0.080** | 5.2–5.6 | 2.9–5.2 |
| 4096×4096 | 128 | 0.392–0.465 | 0.342–0.685 | **0.072–0.078** | 5.4–5.9 | 4.7–8.8 |
| 4096×4096 | 512 | 0.482–0.597 | 1.12–2.60 | **0.193–0.210** | 2.5–2.9 | 5.8–12.4 |
| 4096×4096 | 2048 | 1.06–1.13 | 4.33–10.3 | **0.724–0.867** | 1.3–1.5 | 6.0–11.9 |
| 8192×4096 | 32 | 0.756–0.845 | 0.337–0.598 | **0.136–0.148** | 5.4–5.7 | 2.5–4.0 |
| 8192×4096 | 128 | 0.795–0.884 | 0.618–1.19 | **0.132–0.144** | 5.8–6.1 | 4.7–8.3 |
| 8192×4096 | 512 | 1.01–1.10 | 2.20–4.40 | **0.371–0.423** | 2.6–2.7 | 5.9–10.4 |
| 8192×4096 | 2048 | 2.22–2.33 | 8.60–17.3 | **1.53–1.72** | 1.3–1.5 | 5.6–10.1 |

**The fused kernel wins at every (rung, M, shape) measured — there is no
losing region in the prefill range.** Rung cost is nearly flat (k12→k24 adds
~8% — the smem LUT removes the k-scaling that makes the Triton path 2.4×
slower at k24 than k12). Act-prep (additive to every path, M=2048/K=4096):
baseline `fp4_group16_act_qdq` 6.5 ms; the torch reference NVFP4 quant
stand-in 13.8 ms — serving uses vLLM's fused CUDA `scaled_fp4_quant` instead,
so the fused path's serving-side act cost is far below either.

### 6.2 Dense: native ratio (the number Robert asked to minimise)

`sm120_nvf4_mm_scaled` — the SAME collective/TiledMma/epilogue fed dense
pre-packed NVFP4 — is the speed reference, so the entire ratio is the
decode-in-prologue cost, by construction:

| N×K | M | native | fused k12 / k16 / k24 | ratio |
|---|---|---|---|---|
| 4096×4096 | 32 | 0.044 | 0.072 / 0.076 / 0.078 | 1.65 / 1.75 / 1.79 |
| 4096×4096 | 128 | 0.027 | 0.070 / 0.074 / 0.078 | 2.6 / 2.7 / 2.8 |
| 4096×4096 | 512 | 0.063 | 0.191 / 0.201 / 0.210 | 3.0 / 3.2 / 3.3 |
| 4096×4096 | 2048 | 0.247 | 0.729 / 0.790 / 0.866 | 3.0 / 3.2 / 3.5 |
| 8192×4096 | 2048 | 0.483 | 1.53 / 1.60 / 1.72 | 3.2 / 3.3 / 3.6 |

(the 2.28–3.53 bpw stream is also 1.4–2.2× SMALLER than native 4.5 bpw — at
equal bytes-served the effective gap is smaller than the raw ratio.)

### 6.3 MoE grouped: vs the shipping loop and the native ceiling

E=128, hidden=4096, inter=1536, top-k 8. Loop = the shipping per-expert path
as it serves (its own act-QDQ included, hit-lists pre-computed — favors the
baseline). FusedK = routing + gathers + 2 grouped GEMMs + activation +
combine, with act-quant EXCLUDED from both fused and native-ceiling lines
(the torch quant stand-in is a host-harness artifact; serving uses vLLM's
CUDA quant). Native ceiling = the same row count through the stock dense
NVF4 kernel (ideal grouped bound, no expert indirection).

| T | tile_m | Mp | loop (ships) | fusedK | native ceil | × vs loop | native ratio |
|---|---|---|---|---|---|---|---|
| 512 | 128 | 16384 | 69.4 | **12.3** | 5.95 | **5.6** | 2.07 |
| 512 | 256 | 32768 | 69.4 | 18.1 | 11.8 | 3.8 | 1.54 |
| 2048 | 128 | 25088 | 94.2 | **17.2** | 9.19 | **5.5** | 1.87 |
| 2048 | 256 | 32768 | 94.2 | 18.2 | 11.9 | 5.2 | 1.52 |

tile_m=128 wins in absolute time at these expert sizes (256's lower ratio is
against its own larger padded Mp). Dispatch default: 128.

### 6.4 The tuning campaign (attempts, with measured effect)

1. **Packed byte-writes via `recast<uint8_t>` of the position-independent
   sub-byte tensor — LOST (wrong results).** The UMMA 4-bit atoms' swizzle is
   byte-address-based; the recast view permuted odd rows. Replaced by manual
   byte-space swizzle addressing (also faster than per-nibble RMW).
2. **Producer-warp smem staging of packed rows (the fp8-fork pattern) —
   MEASURED 7–38× off native.** One 32-thread warp staging ~9 KB/K-tile
   serialized the whole pipeline (0.373 ms at k16/M=128 where native is
   0.027). Kept only as a 16-byte per-stage descriptor publish.
3. **Consumer gmem-direct decode (landed):** the 256 MMA threads gather
   packed bytes straight from gmem (L1/L2-hot, the decode-GEMV pattern);
   producer stages only {base ptr, n_base}. **5× kernel-level** (0.373 →
   0.074 ms at k16/M=128); native ratio 13.6× → 2.7×; MoE fusedK 58.5 →
   12.3 ms. Also removed the tail-slack requirement (all gmem windows stay
   in-superblock) and 17 KB of smem (76,800 → 59,392 B).
4. **NOT ATTEMPTED — decode-ahead double buffering.** The remaining 1.6–3.6×
   to native is decode latency serialized with the MMA per K-tile (the same
   "v1 tax" the fp8 fork documents). A second decoded B/SFB buffer
   (+9 KB, fits) with decode(t+1) issued before MMA(t) would overlap the
   gather latency; it needs a barrier-choreography rework of the mainloop
   and was deliberately not rushed into a bit-exact-passing kernel at the
   end of this session. It is the next lever, and the measured ratio table
   above says exactly where it pays (large M, high rungs).

### 6.5 fp8 non-regression and default-path evidence

* Every fp8 kernel source is **byte-identical to HEAD** (`git diff` empty on
  `cb_fused_gemm.cu`, `cb_gemv.cu`, `cb_persistent_tc.cu`, all fp8
  `cutlass_fork/*.hpp`, `kernels.py`, `expand.py`) — the compiled fp8
  binaries cannot differ. The fp4 mainloop is a NEW file beside them.
* Shared-file changes are additive: `codec.py` new helpers only —
  HEAD-vs-current byte-identity harness: `fp4_group16_act_qdq`,
  `fp8_dynamic_act_qdq`, `pad_qweight`, `build_compose_table` all
  `torch.equal` on random inputs, timings within noise (≤1% at M=2048).
  `linear.py`/`moe.py` additions are unreachable with the env gates unset
  (fp8 layers short-circuit on `self.is_fp4`).
* Full venv kernel suites green post-change (test_cb_kernels,
  test_two_tier_v2: 42 passed; test_fused_fp4_prefill: 25 passed —
  includes the SASS gate).

### 6.6 Dispatch recommendation

* **Dense:** fused for ALL prefill M (it wins every measured point;
  `PRISMAQUANT_CB_FUSED_FP4=1`). Decode (M≤16) stays on the CUDA GEMV,
  untouched. `midm` remains available for a conservative staged A/B.
* **MoE:** grouped fused at tile_m=128 (`PRISMAQUANT_CB_FUSED_FP4_MOE=1`);
  256 available for large-expert models pending per-layer auto-selection.
* Both remain **opt-in until the served KL A/B** (§4: the activation bucket
  changes; speed evidence alone cannot promote them under STANDARDS.md).

### 6.7 Explicitly NOT measured

* Served tok/s and served KL (needs a serve window + the in-container vLLM
  quant op; the harness quant stand-in under-sells the fused end-to-end).
* The fp4 arms inside `moe_autotune`'s per-layer auto — add after the served
  numbers exist, gated the same two-model way as fp8's auto.
