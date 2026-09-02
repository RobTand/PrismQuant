# 27B served A/B — NVFP4-CB/FP8-CB lane (K2 gold metric)

**Model:** Qwen3.6-27B (hybrid VLM + Gated-DeltaNet), BF16 source
`/home/rob/.cache/huggingface/qwen36-27b-bf16`.
**OURS:** `EXPORT_CONTAINER=nvfp4_cb`, menu `NVFP4,FP8_DYNAMIC,BF16,FP8_CB_{K36,K40,K44,K48}`,
`TARGET_BITS=5.5`, `COST_MODE=local`, `CB_SCALE_CODING=v1`, lattice codebooks,
`NSAMPLES×SEQLEN = 8×1024`. Allocator body bpp **achieved = 5.501**.
**Baseline:** shipped `rdtand/Qwen3.6-27B-PrismaAURA-5.5bit`.
**Metric:** exact vLLM top-20 KL-vs-BF16 on held-out WikiText (8176 positions) +
direct PPL (actual-token NLL) + TTFT(1400)/decode, each a fresh `vllm-node`
`--enforce-eager` container, same BF16 reference dump. Served 2026-07-18.

## Allocation composition (thesis at allocation level)
The allocator, given the CB menu at real (local) cost, put the **entire
quantizable body** on codebook formats — **386 Linears FP8_CB**, **0** stock
NVFP4/FP8:

| format | Linears |
|---|---|
| FP8_CB_K36 | 136 |
| FP8_CB_K40 | 30 |
| FP8_CB_K44 | 77 |
| FP8_CB_K48 | 143 |
| BF16 (floor: visual + embed + lm_head + norms + conv1d + GDN in_proj_a/b + MTP) | 813 copied |

## Matched-bpp check (honest denominator)
Quantized-body bytes over the identical architecture, from the safetensors
headers (non-floor tensors):

| | total | **body (non-floor)** | floor |
|---|---|---|---|
| OURS (CB) | 23.62 GB | **16.713 GB** | 6.909 GB |
| AURA-5.5  | 23.61 GB | **16.707 GB** | 6.898 GB |

The two artifacts spend the **same ~16.71 GB** on the quantized body (Δ 0.04%).
This is a genuine matched-bpp comparison — the same byte budget, allocated to
codebook formats vs uniform NVFP4/FP8.

## Gold-metric verdict

| | conf-KL | ALL-KL | conf top1 | ALL top1 | PPL | NLL |
|---|---|---|---|---|---|---|
| BF16 (ref) | — | — | — | — | 9.123 | 2.2108 |
| AURA-5.5 | 0.02407 | 0.0321 | 98.77% | 92.3% | 9.251 | 2.2247 |
| **OURS (CB)** | **0.01134** | **0.0134** | **99.56%** | **95.4%** | **9.166** | **2.2155** |

**Quality — CB wins decisively, at matched bpp:**
- ALL-KL **−58.3%** (0.0134 vs 0.0321), confident-KL **−52.9%** (0.01134 vs 0.02407).
- top-1 agreement higher on both slices (95.4% vs 92.3%; 99.56% vs 98.77%).
- PPL gap to BF16 **3× smaller** (OURS +0.043 vs AURA +0.128).

The KL delta (−58%) is far larger than the calibration-draw noise band (~±10–40%
at single-seed 8×512) and corroborated by PPL, so the win is robust. The 0.04%
body-byte difference cannot explain it (≈2^−bpp ⇒ ~0.1% KL). **The same bytes,
spent on codebook formats, buy materially more quality.**

## Speed — the known prototype-(i) limitation

| | TTFT(1400 tok) | decode tok/s |
|---|---|---|
| BF16 | 1.269 s | 4.59 |
| AURA-5.5 (native NVFP4/FP8) | **0.746 s** | **10.26** |
| OURS (CB, Triton prototype-i) | 1.622 s | 4.20 |

**OURS is 2.2× slower prefill, 2.4× slower decode than AURA** — it does NOT yet
meet the goal's "no prefill degradation" bar. This is expected and documented:
the plugin is *"correctness-first Triton serving (INV-2 waived), NOT
production-eligible."* Prefill uses the transient CB→native-tile expand + a
bf16-MMA/`cutlass_scaled_mm` path (per-layer expand overhead dominates at these
M); decode uses the Triton `cb_gemm` (per-M-tile decode), both far off the native
tensor-core GEMM AURA reaches. **Closing this is the CUTLASS fused-kernel
workstream** (goal amendment: prove validity first, then build CUTLASS kernels).

## Serving-load bugs fixed to reach this (commit 198a1b9)
The 0.6B prototype (dense, standard-attention, text-only) never exercised these;
serving a hybrid VLM + GDN on metal surfaced all three:
1. **`apply_vllm_mapper`** did not remap the CB `target_scheme` through vLLM's
   weight mapper — for a VLM the `model.language_model.` → `language_model.model.`
   nesting is remapped at load, so every CB target fell through to unquantized
   and the `cb_qweight` load failed.
2. **Merged-role `cb_row_offset` truncation:** vLLM merges GDN `in_proj_qkvz`
   (4 logical widths) from two CB roles (`in_proj_qkv`=q,k,v + `in_proj_z`=z);
   the offset table was built from `widths[0]` only → illegal memory access.
   Per-role spans now recovered from the checkpoint's separate `.cb_qweight`
   tensors, with a hard assert that the table covers every output row.
3. **Missing VLM sidecars:** the exporter omitted `preprocessor_config.json` /
   `video_preprocessor_config.json` / `chat_template.jinja`, which vLLM's
   `qwen3_vl` input processor requires. Exporter now copies them (in-memory +
   streaming).

## Status
- **Quality thesis: PROVEN on the 27B gold metric at matched bpp.** ✅
- **Speed thesis: NOT met by the Triton prototype; CUTLASS fused kernel is the
  fix.** ⏳
- Coherent greedy generation confirmed ("The capital of France is" → " Paris.";
  "2+2 equals" → " 4.").
- Next model classes per the goal: 35B MoE, then Hy3/DSv4 ultra-low-bpp.

## Addendum — CUDA kernel session (2026-07-18/19): decode at native parity

The kernel workstream (CUDA decode-GEMV + fused act-QDQ + CUDA transient
expander + fp8-direct expand; commits c5741ad..) closed the decode gap and
~40% of the prefill gap. Same artifact, same harness, `--enforce-eager`:

| | TTFT(1400) | decode tok/s |
|---|---|---|
| AURA-5.5 (native) | **0.746 s** | 10.26 |
| OURS Triton prototype-i (the old row) | 1.622 s | 4.20 |
| OURS + fp8-direct expand | 1.222 s | 4.23 |
| **OURS + CUDA GEMV + CUDA expander (now)** | **1.075 s** | **10.27–10.30** |

- **Decode 4.20 → 10.28 (2.45×): AT/ABOVE native AURA** (10.26). The GEMV is
  bandwidth-bound (250–355 GB/s effective per layer); at matched body bytes
  (16.71 GB both) parity IS the ceiling — reached.
- **Prefill 1.622 → 1.075 s**: fp8-direct expand (−25%) + CUDA expander 2×
  (61–86 → 123–132 GB/s; expand now ~34% of serial prefill). The remaining
  0.33 s vs AURA is the transient's write+read traffic — only the fused
  decode-in-prologue CUTLASS kernel removes it (baseline-parity gate for that
  fork PASSED: sm120 CollectiveBuilder fp8 GEMM from vendored headers runs at
  0.91–0.99× of vLLM's `cutlass_scaled_mm`).
- N-chunked expand+GEMM overlap was tried and REJECTED: 0.46× and not
  bit-exact (`cutlass_scaled_mm` changes config on narrow N).

**KL preserved — with a measurement-arithmetic caveat worth recording.** The
dump reproduces conf-KL 0.01134 / ALL 0.0134 / PPL 9.166 bit-for-bit across
sessions when the serving process matches the original arithmetic state, and
reads 0.01328 / 0.0142 / 9.189 when the CUDA extension is resident during the
dump. The cause is NOT the CB kernels (both prefill paths are bit-identical
offline, pinned by tests): loading the extension shifts allocator addresses →
alignment-sensitive dispatch elsewhere in the model → global reassociation-
level drift. This is the concrete mechanism of the known cross-session KL
drift; conf-KL on this artifact has ±17% evaluation sensitivity. Under either
reading the verdict is unchanged: −45% to −53% conf-KL vs AURA (0.02407),
ALL-KL −56 to −58%, PPL gap to BF16 2–3× smaller.

## Addendum — CUDA-graph decode experiment (2026-07-18)
Tested serving OURS WITHOUT `--enforce-eager` (let vLLM capture decode graphs):

| | TTFT(1400) | decode tok/s |
|---|---|---|
| OURS eager | 1.622 | **4.20** |
| OURS cudagraph | 1.372 | 1.21 |
| AURA (native) | 0.746 | 10.26 |

**CUDA graphs HURT decode (4.20→1.21).** Root cause: vLLM pads captured decode
batches above `PREFILL_M_THRESHOLD=16`, so every graphed decode step takes the
**expand path** (full [N,K] materialize + GEMM per token) instead of the cheap
`cb_gemm`. Launch-overhead is NOT the decode bottleneck — the `cb_gemm` decode
kernel's own throughput is (it's below even BF16). Conclusions: (a) keep
`--enforce-eager` for the current plugin; (b) the decode fix is a faster
bandwidth-bound dequant-GEMV (plan §1b / prototype ii), NOT cudagraph; (c) the
M-gated dispatch is cudagraph-hostile — a production decode kernel must key off
the real token count, not the padded batch. Prefill still needs the CUTLASS
fused-expand kernel (§1a / iii).
