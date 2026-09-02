# CUTLASS serving-kernel — grounding map (2026-07-18)

> **FUSED-PREFILL VERDICT (2026-07-19, commits 80e6414/8936a01).** The
> decode-in-prologue collective EXISTS and is **bit-exact**
> (`csrc/cutlass_fork/sm120_cb_fused_mma.hpp`, KBits-templated, packed-B TMA +
> consumer-side smem decode, 128×64×128 Stages=2, 74.7–76.8 KB smem;
> pinned by `tests/test_fused_prefill.py` incl. the real 0.6B layer). But at
> M=1400 it measures **0.22×** vs the serial transient — structural, not
> implementation: every M-tile CTA re-decodes the same B tiles, so decode work
> scales ×ceil(M/128) while the transient expands once (predicted 35 ms ≈
> measured 36 ms). Chunked expand+GEMM overlap with our fixed-config fork GEMM
> (bit-safe, unlike `cutlass_scaled_mm`) is also dead: 0.74–0.79× — the
> M=1400 GEMM is already partially memory-bound on GB10's ~273 GB/s, no spare
> bandwidth to hide the expander. **The fused kernel's honest niche is
> M∈(16,128]: 1.04×/1.26×/1.45× at M=32/64/128.** *(Correction 2026-07-30: the
> "dispatch intentionally not wired" note is stale — mid-M dispatch is WIRED
> and ON BY DEFAULT since the 2026-07-26 promotion:
> [Gridbook `linear.py`](https://github.com/RobTand/gridbook/blob/master/gridbook/linear.py), gated only by
> `PRISMAQUANT_CB_FUSED_MIDM` defaulting to `"1"` at `:439`, over rungs
> k ∈ {28,32,36,40,44,48}. Promotion evidence in `STANDARDS.md:54` — 1.40×
> in-niche, conf-KL-vs-teacher gate preserved.)* Large-M parity (the remaining 0.33 s
> TTFT) requires a **weight-stationary / persistent-N schedule** — decode each
> B tile once, loop M inside the CTA — a kernel-layer restructure beyond the
> collective fork. Until then the serial transient (1.075 s TTFT) is default.
> Incidental: side-stream drivers must `es.wait_stream(main)` before consuming
> main-enqueued tensors (IMA repro otherwise).

> **STATUS UPDATE (2026-07-19 kernel session).** Build steps 1 and 3 of the
> sequence below are DONE and served; step 2 (fused prefill) is the open
> piece.
> - **Decode: SOLVED at native parity.** CUDA dequant-GEMV
>   (`csrc/cb_gemv.cu`, fused act-QDQ, E4M3-byte LUT, warp-per-superblock):
>   served 27B decode 4.20 → **10.28 tok/s** (AURA 10.26), 250–355 GB/s
>   effective. M-gated: CUDA for M≤8, Triton 9–16, transient expand above.
> - **Prefill: 1.622 → 1.075 s** (fp8-direct expand + CUDA expander at 2× the
>   Triton one). Remaining 0.33 s vs AURA's 0.746 is the transient
>   write+read traffic — the fused prologue below is the only remover.
>   N-chunked overlap REJECTED (0.46×, not bit-exact: `cutlass_scaled_mm`
>   reconfigures on narrow N).
> - **Baseline-parity gate PASSED** (`csrc/sm120_fp8_gemm.cu`): sm120
>   CollectiveBuilder fp8 GEMM from the vendored headers at 0.91–0.99× of
>   vLLM's `cutlass_scaled_mm` on 27B shapes. Fork target for FP8_CB =
>   `sm120_mma_tma.hpp` (copied to `csrc/cutlass_fork/sm120_mma_tma_orig.hpp`);
>   B-tile packed bytes are a CONTIGUOUS per-row slice for 128/256-wide
>   K-tiles (codewords are ordered LSB-first), so the packed tile is
>   TMA-loadable; decode goes producer-side into the SmemLayoutB staging.
> - **Measurement landmine found:** loading ANY extra CUDA extension into the
>   serving process shifts allocator addresses → alignment-sensitive dispatch
>   drift elsewhere → conf-KL evaluation sensitivity ±17% on the 27B (both
>   readings −45%+ vs AURA). This is the mechanism of the documented
>   cross-session KL drift. Compare arms with identical extension residency.

Concrete starting map for the CB serving kernels, after the 27B served verdict
proved quality (−58% KL at matched bpp) but exposed the speed gap (2.2× prefill,
2.4× decode vs native AURA). This is the working brief for the multi-session
kernel build. Read alongside `serving-kernel.md` (the original plan).

## Why fusion is mandatory (evidence, not assertion)
The transient-expand prefill (`expand_cb_to_value` → `cutlass_scaled_mm`)
**materializes the full [N,K] weight tile to HBM, then the GEMM reads it back**.
That is inherently ≥2× the memory traffic of a resident-weight GEMM — and worse
today because `expand_cb_to_value` writes a **bf16** tile (2 B/elt = 32 GB for
the 27B body) then casts to fp8 (another 16 GB), vs AURA's single 16 GB fp8
read. Measured prefill is 2.2× slower — consistent with the traffic doubling.
No amount of tuning the expand removes this; only **decode-in-prologue fusion
(never materialize the tile)** does. This retires the "transient path may be
enough" hope from the old plan — it is not, at 27B.

Cheap partial win available NOW without CUTLASS (do first, it de-risks + helps):
- Make `expand_cb_to_value` write **fp8 directly** (codebook values are on the
  e4m3 grid), skipping the 32 GB bf16 intermediate → halves expand-side traffic.
- Still ≥2× vs resident; a stopgap, not the fix.

## CUDA-graph is NOT the decode fix (falsified 2026-07-18)
Serving OURS without `--enforce-eager` made decode WORSE (4.20 → 1.21 tok/s):
vLLM pads captured decode batches above `PREFILL_M_THRESHOLD=16`, so every
graphed step takes the expand path per token. Keep `--enforce-eager`. The
decode bottleneck is the `cb_gemm` kernel's own throughput (below even BF16),
not launch overhead. The M-gated dispatch (host branch on M) is cudagraph-
hostile — a production kernel must not branch on a padded batch size.

## Environment (vllm-node:latest, confirmed)
- GB10 **sm_121** (compute_cap 12.1); it is the **sm_120 family** (NOT sm_100a /
  tcgen05 — that's datacenter Blackwell). Target the sm_120 block-scaled `mma`.
- nvcc **13.0**, torch cuda 13.0 → can build a CUDA extension (GGUF-plugin
  `setup.py` CUDAExtension model).
- **CUTLASS C++ headers 4.3.4** vendored in vLLM:
  `.../vllm/third_party/fmha_sm100/cutlass/include` (also under `deep_gemm/`;
  the `cutlass` python pkg is 4.5.2 — use the vendored C++ headers for building).
- **Toolchain gate PASSED (2026-07-18):** `csrc/toolchain_probe.cu` compiles with
  `nvcc -std=c++17 -arch=sm_121a -I<cutlass/include>` and runs on the GB10 —
  `cutlass::float_e2m1_t(1.5f)` round-trips correctly on-device. So custom
  CUDA+CUTLASS+FP4 kernels build+run for sm_121; the build path is proven before
  any mainloop work.
- Native reference GEMM AURA uses: `vllm._custom_ops.cutlass_scaled_fp4_mm`
  (+ `scaled_fp4_quant`, `cutlass_scaled_mm_supports_fp4`,
  `flashinfer_quant_nvfp4_8x4_sf_layout` for the SF swizzle,
  `cutlass_fp4_moe_mm` for MoE).

## Fork targets (the files to base the kernel on)
- **Collective mainloop:** `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`
  — its global→shared **A producer** is where we inject: load k-bit CB indices →
  codebook lookup → 8 FP4 codes into the smem staging tile in the exact nibble
  layout the block-scaled MMA consumes. The group-16 E4M3 scale plane is
  **unchanged** (that is the whole point of matching NVFP4's scale envelope) —
  reuse it verbatim.
- **Array/grouped variant (MoE, prototype iv):**
  `sm120_blockscaled_mma_array_tma.hpp`.
- **Kernel driver:** `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`.
- **Types/layout:** `float_subbyte.h` (FP4/E2M1), `detail/sm100_blockscaled_layout.hpp`.

## Build sequence (revised, evidence-driven)
1. **Baseline parity:** compile a plain sm_120 block-scaled NVFP4 GEMM from the
   CUTLASS collective as a CUDA extension; match `cutlass_scaled_fp4_mm`
   numerically + on speed. Proves the toolchain + our layout understanding
   before touching the mainloop. (This is the de-risking gate.)
2. **Small-k LUT fused prefill (k≤13):** fork the sm120 A-producer to do
   flat-LUT codebook lookup in smem (GB10: 99 KB opt-in smem; k=13 LUT = 32 KB).
   FP8_CB first (the 27B rung set), then NVFP4_CB.
3. **Decode GEMV (parallel, tractable):** a fixed-shape bandwidth-bound dequant-
   GEMV that beats AURA on HBM traffic (fewer bytes). Must not branch on padded
   batch; obey INV-1 (no full-weight materialize).
4. **Structured-codebook variant (k≥14)** + **MoE grouped** (iv) later.

## Native NVFP4 GEMM contract (RESOLVED 2026-07-18 from vLLM source)
`vllm._custom_ops.cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b,
alpha, out_dtype)`:
- `a`,`b`: fp4-packed (2 codes/byte) activation + weight.
- `block_scale_a/b`: the group-16 **E4M3** block scales, run through
  `swizzle_blockscale` (`quantization/utils/nvfp4_utils.py`): pad M→128, K→4,
  reshape `(B, M/128, 4, 32, K/4, 4)`, permute `(0,1,4,3,2,5)`. This IS the SF
  interleave the sm120 block-scaled MMA consumes — the fused kernel must emit the
  decoded scale plane in exactly this layout (or pre-swizzle at load).
- `alpha`: per-tensor scalar = input_global_scale × weight_global_scale (both
  stored inverted, `compressed_tensors_w4a4_nvfp4.py`). Our CB per-channel/E4M3
  scales already live in NVFP4's envelope; the extra per-tensor global is the
  only new scalar to thread.

### Two kernel routes, now both concrete
1. **Expand-to-native + native GEMM (easy, no mainloop fork, still ~2× traffic):**
   a Triton kernel decodes CB indices → fp4-packed tile + swizzled E4M3 scales,
   then call `cutlass_scaled_fp4_mm`. Reaches FP4 tensor cores (INV-2) for
   *fp4-CB* artifacts (Hy3 ultra-low-bpp) without touching CUTLASS internals.
   Does NOT help the 27B (fp8-CB already uses native fp8 `cutlass_scaled_mm`; its
   gap is the 2× traffic, not the MMA). Good stepping stone + Hy3-relevant.
2. **Fused decode-in-prologue (the real 1× fix, hard):** fork
   `sm120_blockscaled_mma_tma.hpp` A-producer; decode in smem, never materialize.
   Required to beat AURA on prefill for BOTH fp8-CB and fp4-CB.

Remaining open: exact FP4 nibble/interleave the sm120 MMA smem tile wants (match
`_pack_fp4_indices`, `nvfp4_fused.py:32`) — resolve at baseline-parity step 1.
