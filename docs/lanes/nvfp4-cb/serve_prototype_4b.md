# NVFP4-CB / FP8-CB — served prototype (ii+): transient-expansion prefill + 4B scale

> **Historical evidence.** Gridbook now solely owns the runtime and kernel
> implementation. PrismaQuant consumes its immutable external pin from
> `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; there is no in-repo runtime copy.

> **Prototype (ii+) of `docs/lanes/nvfp4-cb/serving-kernel.md`.** Builds on the
> prototype-(i) correctness serve (`serve_prototype_0p6b.md`): adds a **tuned
> decode kernel** and a **transient-expansion prefill path** that reaches vLLM's
> **stock native fp8 W8A8 GEMM**, and takes the whole thing to **Qwen3-4B** — the
> scale where prefill compute starts to matter. INV-1 honored throughout (no
> resident dense weight); INV-2 still waived for decode (bf16 `tl.dot`), but the
> **prefill path now hits fp8 tensor cores** via the transient trick.

- Runtime: the historical Gridbook prototype (now external). Serve stack:
  `vllm-node:latest`
  (vLLM 0.23.1rc1), GB10 / sm_121, `--enforce-eager --max-model-len 4096`, tp=1,
  `gpu_memory_utilization 0.5`. KL: served top-20 `prompt_logprobs` (kl_tool),
  held-out `wiki.test.raw`, 8192 tok × seqlen 512, CB and BF16 same session.
- Artifacts: 0.6B `fp8cb_k44` / `nvfp4cb_k16`; **4B `fp8cb_k44_4b`** (uniform
  FP8_CB_K44, product, shared-per-role learned codebooks, scale_sweep, imatrix
  seed-0 draw; 252 CB Linears / 7 roles / 36 layers; 3.31 GB).

## 0. Bottom line

The **transient-expansion prefill trick works and scales**: routing M>16 FP8_CB
prefill through vLLM's stock fp8 W8A8 GEMM (via a bounded per-layer e4m3 tile,
INV-1 intact) cuts prefill **6.3× at 0.6B / 8.5× at 4B** vs the Triton re-decode,
landing FP8_CB prefill at **1.18× BF16 (0.6B) / 1.35× BF16 (4B)** — BF16/IQ4_XS-
class — **without** the 15–25-day CUTLASS fused-expand mainloop. Decode is tuned
(+34% NVFP4) and **beats BF16 at 4B** (25.1 vs 22.6 tok/s, the 5.5-bpw bandwidth
win). Quality holds: 4B FP8_CB served conf-KL **0.0202** (emu 0.0181, **1.12×**
agreement — same faithfulness as 0.6B), PPL +1.0% vs BF16. Encode cost is real
(sweep = 96% of the 187.7-min 4B pack) and **not** JSO-prunable to ~10× (refits
fire 99.5%; candidates spread over 6 high-clip levels → ~2.6× at most).

## 1. The transient-expansion prefill path (the cheap native-GEMM trick)

M-gated dispatch (GGUF's `mmvq_safe` pattern): **M ≤ 16 (decode)** → the tuned
Triton decode-GEMM (bf16 MMA, INV-2 waived); **M > 16 (prefill), FP8_CB** →
transiently expand THIS layer's packed weight to a native fp8 tile and call
vLLM's stock per-channel W8A8 fp8 GEMM, then free it.

**Why it's clean:** an expanded FP8_CB weight *is* a standard per-channel fp8
checkpoint — the codebook values already live on the e4m3 grid (‖·‖≤448) and
`weight_scale` is per-output-channel. So `expand_cb_to_value` (the decode kernel
minus the matmul minus the scale) produces a bf16 tile whose `.to(e4m3)` is
**provably lossless** (verified: max cast-error 0.0), and it drops straight into
`ops.scaled_fp8_quant` + `ops.cutlass_scaled_mm` (`scale_b=[N,1]`).

**INV-1 nuance (honored precisely).** The retired low-bit lane died from a RESIDENT, model-wide
dense expansion (92.9 GB → 115.7 GiB, OOM). Here the expansion is a per-LAYER
TRANSIENT: expand one layer → GEMM → free before the next. Verified: peak
transient **9.4 MiB** for the 0.6B down_proj, and allocation returns to baseline
across 20 forwards (**0.0 MiB growth**). The resident weight stays the packed
k-bit stream + tiny codebook + per-channel fp32 scale.

Correctness (`tests/test_transient_fp8.py`, both verified independently):
`expand_cb_to_value` == `nvfp4_cb_reconstruct / weight_scale` (rel ~2e-8, e4m3
cast lossless); the transient fp8 GEMM matches a fp32 dequant reference at rel
**1.7e-3** (≤ the 2e-2 gate). NVFP4_CB stays on the Triton decode path (a
transient FP4 tile still needs the FP4-MMA to be worth it — prototype (iii)).

## 2. Decode-kernel tuning

The decode kernel now loads each of the 32 distinct codewords once per
superblock (not once per column — an ~8× byte-load reduction) and, for fp4,
loads the 16 group-16 scales once (~16× reduction), broadcasting each.
Bit-exact (25/25 tests unchanged). Same-session decode tok/s (0.6B):

| artifact | decode before | decode after | Δ |
|---|---|---|---|
| FP8_CB_K44 | 77.2 | 80.3 | +4.1% |
| NVFP4_CB_K16 | 33.8 | 45.2 | +33.7% |

(NVFP4 wins big — it benefits from both the codeword and the scale-plane
reduction; FP8 has no in-weight scale plane so it gets only the codeword cut.)

## 3. 0.6B v2 speed suite — transient vs old-Triton prefill (the isolated lever)

Same box/flags, 3× each, steady = mean of samples 2–3. Prefill A/B toggled with
`PRISMAQUANT_PREFILL_M_THRESHOLD` (∞ forces the old Triton path at prefill).

| 0.6B path | TTFT(1400) s | vs BF16 | decode tok/s | vs BF16 |
|---|---|---|---|---|
| BF16 (vLLM native) | 0.035 | 1.0× | 133.2 | 1.0× |
| **FP8_CB — transient fp8 prefill** | **0.042** | **1.18× slower** | 83.6 | 0.63× |
| FP8_CB — old Triton prefill (A/B) | 0.263 | 7.5× slower | 83.6 | 0.63× |
| NVFP4_CB (decode-tuned, Triton) | 0.275 | 7.9× slower | 45.2 | 0.34× |
| IQ4_XS GGUF (v1 session, native) | 0.039 | 1.1× slower | 95.8 | 0.72× |

**The transient lever, isolated (same session):** FP8_CB prefill 0.263 → **0.042 s
(6.3× faster)** just by routing M>16 through the stock fp8 GEMM instead of the
Triton re-decode — landing at **1.18× BF16**, i.e. prefill is now essentially
BF16-class. Decode improved 77→84 tok/s from the codeword/scale-load tuning.
Coherence smoke (transient prefill on fused qkv/gate_up): *"…Paris, and the
capital of Italy is Rome. The capital of Spain is Madrid…"* — coherent, so the
fused-transient path (per-role codebook offsets) is correct end-to-end. FP8_CB
served conf-KL **0.0215** / PPL 34.86 (vs v1 0.0208 / 34.98) — quality preserved
through the transient path (the KL dump's 512-token prefills exercise it).

## 4. 4B — the scale point that decides the native-GEMM story

### 4a. Quality (first 4B datum — 1 seed; emu gate + served)

| 4B FP8_CB_K44 | emu conf-KL | served conf-KL | served/emu | top1 (conf) | PPL | BF16 PPL |
|---|---|---|---|---|---|---|
| whole-model (252/252 CB) | **0.0181** | **0.0202** | **1.12×** | 98.6% | 21.94 | 21.72 |

**First 4B quality datum (1 seed, v1 scale-plane).** The emulation gate predicts
the served KL to **1.12×** at 4B — the same emu↔served faithfulness proven at
0.6B (1.02–1.09×), now holding on a 6.7× larger model and through the transient
native-fp8 prefill path. FP8_CB_K44 is near-lossless at 4B: served conf-KL
0.0202, top-1 agreement 98.6%, PPL +1.0% over BF16. (Served KL is top-20; the
emu number is full-vocab — the ~12% gap is partly that. v1 scale plane = a
known-understating baseline for the fp4 family, but FP8_CB has no fp4 scale
plane so it is unaffected; see caveats.)

### 4b. Speed (TTFT where prefill compute matters)

| 4B path | TTFT(1400) s | vs BF16 | decode tok/s | vs BF16 |
|---|---|---|---|---|
| BF16 (vLLM native) | 0.166 | 1.0× | 22.6 | 1.0× |
| **FP8_CB — transient fp8 prefill** | **0.223** | **1.35× slower** | **25.1** | **1.11× (faster)** |
| FP8_CB — old Triton prefill (A/B) | 1.897 | 11.4× slower | 25.1 | 1.11× |
| IQ4_XS GGUF (native) | 0.214 | 1.29× slower | 21.9 | 0.97× |

**The transient lever at 4B (the decision number).** Prefill compute matters at
4B, and the story holds: transient fp8 prefill **0.223 s vs the old Triton
re-decode 1.897 s = 8.5× faster**, landing at **1.35× BF16** — on par with the
mature IQ4_XS-GGUF native path (0.214 s). The old-Triton path costs 11.4× BF16;
the transient trick is what closes that gap without the 15–25-day CUTLASS
fused-expand mainloop. **Decode: FP8_CB 25.1 tok/s beats BF16 (22.6) and IQ4_XS
(21.9)** — the 5.5-bpw bandwidth win shows at 4B (decode reads ~3× fewer weight
bytes than BF16). (Clean methodology: old-Triton TTFT re-measured with a warmup
call + prefix-caching disabled + distinct prompts — the naive measure.py run had
a prefix-cache artifact.)

## 5. Encode cost at scale (Robert's "is CB quantize hopelessly slow?" question)

4B export wall-clock: **imatrix collection ≈ 19 s** (one calibration forward
pass — negligible); **codebook Lloyd + per-Linear scale-sweep+assign+pack
dominates — 187.7 min** for 252 Linears (~45 s/Linear). So the encode cost is
the per-Linear weighted-VQ scale sweep, not calibration; naively that is
~3.1 h @ 4B → ~O(days) at 100B+ scale, which is exactly why the histogram below
matters.

**Stage split:** the per-Linear sweep+assign dominates. Mean sweep **42.8 s/Linear**
across a 7-role sample (down/gate/up ~74 s, q/o ~31 s, k/v ~8 s — cost ∝ N·K),
extrapolating to **~180 min for 252 Linears ≈ 96 % of the 187.7 min packing**.
Codebook Lloyd is the small remainder (~4 %); imatrix is ~19 s. **So essentially
all CB-encode cost is the scale sweep.** Full JSON:
`/home/rob/dq-runs/nvfp4-cb-phase0/serve/encode_cost_4b.json`.

**Chosen-scale-candidate histogram** (JSO-collapse test — does the 16-candidate
sweep collapse to a few winners like JSO's 7-level grid → {6,4}? N = 30 720
FP8 per-channel groups):

| candidate (0 = amax/6 one-shot … 15 = max clip) | 8 | 9 | 10 | 11 | 12 | 13 | 14 | **15** |
|---|---|---|---|---|---|---|---|---|
| % of groups winning | 2.1 | 3.2 | 4.9 | 7.1 | 10.5 | 15.7 | 17.5 | **32.9** |

(candidates 0–7 each < 1.5 %.) **It does NOT collapse** — the winners spread
across the *high-clip* end (candidates 11–15 = **83.8 %**), i.e. the sweep is
genuinely finding value in clipping, not defaulting to the one-shot scale. This
is the opposite of the JSO collapse and (separately) of the v1 *fp4* defect
where E4M3-subnormal snapping degenerates the candidates to ~4 values — an
**fp8**-per-channel sweep has no such snapping and explores meaningfully.

**Refit strict-improvement rate:** iter1 = **99.5 %**, iter2 = **95.2 %** — the
WLS refits fire almost every group, so they earn their cost and are **not**
prunable.

**Verdict.** The naive ~10× encode speedup (16×3 → 3×1) is **not** supported for
FP8_CB: the refits are load-bearing and the candidate mass sits on ~6 high-clip
levels, not ~3. A defensible prune is 16 → the ~6 winning high-clip candidates
(captures 83.8 % exactly, ≈ **2.6×** faster) with refits kept — a real but
modest lever, *measured not assumed*. (Not implemented; this measurement decides
it. NVFP4_CB's fp4 sweep is a separate question, gated on the layout-v2 fix.)

## 6. Caveats (read before citing)

- **Correctness prototype, not production.** Decode is bf16 MMA (INV-2 waived);
  prefill now reaches fp8 tensor cores via the transient trick, but the CUTLASS
  fused-expand (prototype iii, no transient buffer at all) is still the endgame.
- **CUDA-graph capture: pre-existing silent divergence** (the original prototype-
  (i) kernel degenerates under graph replay too — not introduced by the decode
  tuning). Likely Triton JIT-during-capture; the prototype serves `--enforce-eager`
  per spec, so measured numbers are unaffected. Flagged for a future fix.
- **Transient path is FP8_CB-only.** NVFP4_CB prefill stays on the Triton path.
- **4B is 1 seed, top-20 KL** (full-vocab not exposed over the API); PPL reported
  alongside. tp=1.
- **Transient buffer peak** is one layer's `[N,K]` (bf16 + e4m3 ≈ 3·N·K bytes),
  freed each forward — bounded, not resident (INV-1). Measured 9.4 MiB for the
  0.6B down_proj; the largest 4B layer (fused gate_up, N=19456×K=2560) bounds it
  at ≈ 150 MiB — vs a resident dense expansion of the whole 4B model (~19 GiB
  bf16), the resident-expansion trap this avoids.
- **v1 scale-plane caveat.** These artifacts use the v1 group-16 E4M3 scale
  plane; a confirmed v1 defect (fp4 sweep candidates collapse to ~4 distinct
  values under subnormal-E4M3 snapping) is being fixed as layout v2 in parallel.
  It is **fp4-specific** (the FP8_CB family has no in-weight scale plane — its
  scales are per-output-channel fp32), so the FP8_CB speed / emu↔served / memory
  results here are unaffected; the **4B FP8_CB KL is a v1 baseline** and the
  NVFP4_CB numbers are known-understating. Quality re-measurement on v2 is
  sequenced after this pipeline.
