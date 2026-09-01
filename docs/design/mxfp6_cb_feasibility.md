# MXFP6-grid codebooks: feasibility (performance-gated)

> **2026-09-01 — MXFP6 is removed from the tree.** Rob: *"You can drop mxfp6.
> It's unsupported everywhere."* `MXFP6_E3M2` / `MXFP6_E2M3` are no longer
> registered formats, and MXFP6 appears in no menu, profile, lane spec or test.
> **This file is kept, not archived**, because it is the *evidence* for that
> decision (Q1: FP6 rides Blackwell's 8-bit datapath — one `kind::mxf8f6f4`,
> no separate fp6 rate, operands byte-padded to 1 byte per value) and because
> its subset-dominance result is cited as the anchor by
> `format_kernel_inventory.md`, `mxfp4_cb_feasibility.md` and
> `strix_halo_format_plan.md`. Read it as a settled record, not as a live
> candidate.

2026-07-30. Commissioned by Robert: "explore the feasibility of using MXFP6 to
support codebook formats… FIRST decide if it could yield any performance
improvement." Architecture/hardware-capability study only; no GPU work run.

## Verdict: NO-GO

**An MXFP6-grid CB rung is strictly dominated by the existing FP8-CB rung at the
same k — identical stored bytes, identical compute rate, equal-or-worse quality.**
Robert's hypothesis (MXFP6 computes on the 8-bit hardware) is confirmed, and the
CB architecture makes the case *stronger* than the hypothesis: there is not even
a storage win to forfeit. In a codebook format the stored weight stream is the
k-bit **index** stream; the element grid appears only in the codebook's *values*.
So "MXFP6-CB at k bits" stores byte-for-byte what FP8-CB at k bits stores — and
since both MXFP6 element grids are **exact subsets of the e4m3 grid** (verified
numerically: all 63 distinct E2M3 values and all 63 E3M2 values round-trip
`float8_e4m3fn` exactly; E2M3 = 4-bit significands over 2^-3..2^2, E3M2 = 3-bit
significands over 2^-4..2^4, both inside e4m3's 4-bit/2^-6..2^8), an MXFP6-grid
codebook **is** an FP8-CB codebook whose entries are handicapped to a subset.
The free-e4m3 VQ learner can always represent the MXFP6 solution and can only do
better. Nothing is left to measure: this is a subset relation, not a tradeoff.

## Q1 — Is there any native 6-bit matrix path?

**NVIDIA Blackwell (the target, GB10/sm_121): no.** FP6 rides the 8-bit datapath:

- PTX/CUTLASS expose one mixed kind, `kind::mxf8f6f4` (A,B ∈ {f4,f6,f8}), rated
  "2x Hopper Fp8 Tensor Core" on SM100 with **no separate fp6 rate**; only the
  fp4-only kinds (`mxf4`, `mxf4nvf4`) are faster ("4x Hopper Fp8")
  ([CUTLASS Blackwell functionality](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)).
- In the f8f6f4 kind, fp6/fp4 operands are **"padded so as to take up 1 byte per
  value"** in smem (TMA types `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` unpack gmem
  fp6 into byte containers) — fp6 does not even save operand-pipe bytes
  ([Colfax sub-byte GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)).
  Our own in-tree CUTLASS fork carries the same fact as code:
  `float_e2m3_unpacksmem_t` / `float_e3m2_unpacksmem_t` in
  external Gridbook `gridbook/csrc/cutlass_fork/sm120_cb_mma_tma.hpp`, and
  the fused mainloop's `IsF8F6F4` static-assert (`sm120_cb_fused_mma.hpp:214-216`).
- B200 datasheets group FP8/FP6 at one PFLOPS number. One microbench claims FP6 >
  FP8 on B200 (`tcgen05.mma` m64n8k16, 2567 vs 1925 TFLOPS,
  [arXiv:2512.02189 Table VI](https://arxiv.org/html/2512.02189v2)) — noted
  honestly: it is a datacenter `tcgen05` tiny-shape microbench, at odds with
  NVIDIA's own grouped FP8/FP6 spec, and GB10 uses the sm_120-class `mma` path,
  where fp6 is byte-container'd with no documented separate rate. Not a lever here.
- **AMD CDNA4 (MI355X) is the one genuine 6-bit-advantaged path**: MXFP6 runs at
  the FP4 rate — ~10 PFLOPS vs ~5 for MXFP8, i.e. 2× fp8
  ([ROCm occupancy guide](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html),
  [MI355X datasheet](https://www.nec.com/en/global/solutions/hpc/lx/images/AMD/amd-instinct-mi355x-gpu-datasheet.pdf)).
  ROCm/HIP hardware, not this box. RDNA4 has FP8 WMMA, no FP6. No Intel XMX FP6 path.
- **vLLM**: MXFP6 upstream is dequant-*emulation* for unsupported devices
  ([RFC #34331](https://github.com/vllm-project/vllm/issues/34331)); the native
  MXFP6 serving kernels that exist are AMD Quark/ROCm on MI355X
  ([W_MXFP4_A_MXFP6 blog](https://rocm.blogs.amd.com/artificial-intelligence/w4a6-quant-mm/README.html)).
  ARCHITECTURE §5.1's "no vLLM kernel" note for `MXFP6_E3M2/E2M3`
  (`format_registry.py:702-719`) is therefore still correct for the CUDA target.

## Q2 — What would an MXFP6-CB rung compute on in gridbook?

The grid sets the GEMM dtype through decode/expand, and every fp8-ladder path is
e4m3 end to end (external Gridbook `gridbook/linear.py`): decode-regime
CUDA GEMV gathers the **E4M3-byte** codebook (`csrc/cb_gemv.cu:266-296`); prefill
expands to a `[N,K]` `float8_e4m3fn` transient and calls stock `cutlass_scaled_mm`
W8A8 (`expand.py:312-352`, "an expanded FP8_CB weight IS a plain fp8 checkpoint");
mid-M fused decode-in-prologue static-asserts `ElementB == float_e4m3_t`
(`sm120_cb_fused_mma.hpp:214-216`). Because MXFP6 grids ⊂ e4m3 (above), an
MXFP6-grid codebook decodes to values that already **are** e4m3: same byte
gather, same transient, same W8A8 GEMM, same rate. A "true fp6" tile could not
run faster anyway — Q1: byte containers, fp8 rate. Compute equality is exact.

## Q3 — Where could a win possibly come from? (each sized)

- **(a) Index stream — the crux: no effect.** FP8-CB stores k bits per d=8
  vector + per-channel fp32 scale + a codebook sidecar (`format_registry.py:954`,
  `nvfp4_cb_footprint.py`). The element grid never appears in the weight bytes.
  MXFP6-CB is not a new storage rate; it is the same rung. Win: **zero**.
- **(b) Expand transient.** The fp8 expander already writes 1 B/elt once
  (`expand.py:131-135`). A packed 0.75 B/elt fp6 transient cannot feed any sm_121
  GEMM (byte containers required, Q1), so it would be re-padded to 1 B/elt.
  Best case: −25% on the expand-*store* leg only, a fraction of a prefill tax
  already ~10% dense total (`docs/lanes/nvfp4-cb/format-speed-policy.md`). Win:
  **negative after the unpack**.
- **(c) Capacity.** The codebook sidecar is `8 << CbSubW` bytes per role — 1 KiB
  (k28) to 32 KiB (k48) (`sm120_cb_fused_mma.hpp:167-172`) vs multi-GB index
  streams; 6-bit packing saves ≤25% of KiB. On a 128 GB box: **nil**.
- **(d) smem LUT (R6).** LUT entries are e4m3 bytes. Storing fp6 values in byte
  containers is footprint-identical (zero win). Bit-packing k48's table to 24 KiB
  would land exactly on the 24,576 B TileM=128 headroom with 0 B margin — a
  configuration class the code deliberately refuses ("EXACTLY 101,376 with zero
  margin … deliberately not taken", `sm120_cb_fused_mma.hpp:174-186`) — and
  reintroduces per-gather shift/mask ALU, precisely the 9.1× ALU term R6 removed
  (`1ede688`). The GEMV's 32-bit paired-value `__ldg` (`cb_gemv.cu:13`) breaks
  too. Win: **negative**.
- **(e) Per-element decode.** Decode is index-driven; element width surfaces only
  as LUT entry width — covered by (d). A6 activations on f8f6f4 would run at the
  fp8 rate while losing precision: strictly worse.

## Q4 — Against the incumbent ladder

The burden was "beat `FP8_CB_K<lower>`", but the comparison cannot even be
formed: there is no k at which MXFP6-CB stores fewer bytes than FP8-CB at that
same k, and the ladder already has every integer rung 28–48 (0.125 bpw steps,
3.5–6.0). At equal k, MXFP6-CB is the same bytes with a subset-restricted
codebook → equal-or-worse KL at identical speed. Strict dominance; no
measurement could invert a subset relation.

## What would change this answer

1. **A serving target with a >fp8-rate 6-bit matrix path.** CDNA4/MI355X already
   is one (MXFP6 at 2× fp8): if the HIP lane ever targets CDNA4 seriously, an
   fp6-grid codebook (expand → MXFP6 GEMM) computes 2× fp8 at unchanged stored
   bpw — a real GO case *on that hardware*, worth revisiting then.
2. NVIDIA landing a distinct fp6 rate on a served sm (e.g. the arXiv:2512.02189
   tcgen05 anomaly confirmed and reaching the sm_12x/GB10 class), with packed-fp6
   (not byte-container) operands.
3. A non-CB question — shipping *native* MXFP6 checkpoints — is out of this
   verdict's scope, but is independently blocked today: no CUDA vLLM kernel, fp8
   compute rate, and the E8M0 √2-binade waste that de-menued MXFP8 (§5.1).

**Graveyard lesson (for ARCHITECTURE §11):** in a codebook format the element
grid is not a storage dial — the index stream is; a narrower grid that is a
subset of the compute grid can only handicap the codebook, never shrink the
artifact or speed the GEMM.
