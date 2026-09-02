# NVFP4-CB / FP8-CB — served prototype (i), Qwen3-0.6B

> **Historical evidence.** The prototype runtime described below was later
> moved to the external [Gridbook repository](https://github.com/RobTand/gridbook).
> PrismaQuant no longer contains or synchronizes a runtime source tree; current
> runs use the immutable pin in `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`.

> **Prototype (i) of `docs/lanes/nvfp4-cb/serving-kernel.md`: a CORRECT-but-slow
> Triton vLLM plugin.** First served KL-vs-BF16 and first speed reading for the
> CB codebook formats. **INV-1 honored** (no dense `[N,K]` in HBM — per-tile
> decode in registers); **INV-2 WAIVED** (bf16 `tl.dot`, not the Blackwell FP4
> MMA). This kernel is *not* production-eligible; it exists to (a) measure CB
> *quality* on the real served stack, and (b) get a first, honest speed reading.
> 0.6B only.

- Runtime: the historical Gridbook prototype (then in-repo; now external and
  immutable-pinned). Artifacts:
  `/home/rob/dq-runs/nvfp4-cb-phase0/serve/{fp8cb_k44,nvfp4cb_k16}`.
- Both artifacts are **uniform** (every target Linear one CB format), product
  mode, shared-per-role learned codebooks, scale_sweep, imatrix col_weights from
  one calibration draw (seed 0) of `diverse-v1.jsonl`. Non-target Linears
  (`lm_head`, `embed_tokens`) stay BF16. 196 CB Linears / 7 roles / 28 layers.
- Serve stack: `vllm-node:latest` (vLLM 0.23.1rc1), GB10 / sm_121,
  `--enforce-eager --max-model-len 4096`, tp=1, `gpu_memory_utilization 0.5`.
- KL convention: served top-20 `prompt_logprobs` (kl_tool), held-out
  `wiki.test.raw`, 8192 tokens × seqlen 512, BF16 and CB served in the same
  session on the identical corpus. Full-vocab is not exposed over the API, so
  this is **top-20 KL** (labelled), plus direct WikiText PPL.

## 1. Served KL-vs-BF16 vs the emulation gate (THE number)

The point is **agreement between the emulation gate's prediction and the real
served KL** — not the absolute value. NVFP4_CB_K16-product is a deliberately
harsh uniform 2.5-bpw artifact (emu ≈ 2.21); we expect matching badness.

| artifact | body bpw | emu conf-KL (predicted) | **served conf-KL** | served/emu | served all-KL | top1 (conf) | PPL |
|---|---|---|---|---|---|---|---|
| FP8_CB_K44 (product, W8A8) | 5.50 | 0.019 | **0.0208** | 1.09× | 0.0373 | 99.40% | 34.98 |
| NVFP4_CB_K16 (product, W4A4) | 2.50 | ≈2.21 | **2.246** | 1.02× | 2.087 | 42.72% | 291.66 |
| BF16 baseline | 16.0 | — | — | — | — | — | 34.32 |

**Served KL tracks the emulation prediction to within ~9% (FP8_CB) / ~2%
(NVFP4_CB).** The emulation gate is a faithful predictor of served behaviour for
the CB family — including reproducing the harsh 2.5-bpw artifact's collapse
(served PPL 291.7, top1 43%) almost exactly. FP8_CB_K44 is near-lossless (KL
0.0208, PPL +1.9% vs BF16, top1 99.4% on BF16-confident positions).

Coherent-generation smoke (greedy, both artifacts loaded + generated):
- FP8_CB_K44: *"The capital of France is Paris, and the capital of Italy is
  Rome. The capital of Spain is Madrid…"* — coherent.
- NVFP4_CB_K16: *"the country of France, and the country of France…"* —
  degenerate, consistent with its emu KL ≈ 2.2 (a stress-test artifact, not a
  shipping candidate).

## 2. Speed (the decision number)

Same box / flags, 3× each. Prefill TTFT on a ~1400-token cold prompt (varied, no
prefix-cache reuse); decode = 128 greedy tokens. The Triton path JIT-compiles on
first call, so sample 1 is warmup-inflated; **"steady" = mean of samples 2–3.**

| path | TTFT(1400) steady s | TTFT ratio vs BF16 | decode tok/s steady | decode ratio vs BF16 |
|---|---|---|---|---|
| (iv) BF16 (vLLM native) | 0.035 | 1.0× | 132.2 | 1.0× |
| (i) FP8_CB_K44 (our Triton plugin) | 0.355 | 10.1× slower | 76.7 | 0.58× |
| (ii) NVFP4_CB_K16 (our Triton plugin) | 0.418 | 11.9× slower | 34.3 | 0.26× |
| (iii) IQ4_XS GGUF (vllm-gguf-plugin, Triton) | 0.039 | 1.1× slower | 95.8 | 0.72× |

Raw samples (s / tok/s): FP8_CB TTFT [0.69, 0.355, 0.355] decode [61.9, 76.4,
77.1]; NVFP4_CB TTFT [0.88, 0.423, 0.413] decode [30.8, 34.3, 34.3]; IQ4_XS GGUF
TTFT [0.37, 0.039, 0.038] decode [95.0, 96.4, 95.9]; BF16 TTFT [0.036, 0.035,
0.035] decode [130.1, 133.2, 133.2]. (IQ4_XS GGUF direct PPL 39.18 — 4.25 bpw,
no imatrix; a speed reference, not a matched-bpw quality point.)

**Honest readout.** Our CB prefill is ~10–12× slower than native BF16 — expected
and *not* the shippable number: this Triton kernel re-decodes each weight tile
once per M-tile (no cross-M reuse) and cannot reach the Blackwell FP4 MMA (INV-2
waived). The **IQ4_XS-GGUF row is the load-bearing contrast**: a *mature* Triton
dequant path (vllm-gguf-plugin) does prefill at ~1.1× BF16 and decode at 0.72×
BF16 — i.e. the ~10× gap is **kernel immaturity in our prototype, not anything
intrinsic to the CB format or to codebook decode**. The production prefill is
prototype (iii) (CUTLASS/CuTe fused-expand): a decoded CB tile is bit-identical
NVFP4 and feeds the *existing* block-scaled FP4 mainloop, so its prefill should
approach plain-NVFP4, i.e. **~1× BF16-class** (better than IQ4_XS, which pays a
BF16-MMA-after-dequant tax). Decode is the representative regime for this
prototype (bandwidth-bound); even un-tuned FP8_CB is 0.58× BF16, and prototype
(ii) (tuned decode kernel + CUDA-graph capture) targets decode parity/advantage
(fewer resident bytes ⇒ less HBM traffic).

## 3. Caveats (read before citing)

- **Triton prototype ≠ production kernel.** INV-2 waived (bf16 `tl.dot`, no FP4
  tensor cores); the kernel re-decodes per M-tile; per-call JIT. It is a
  correctness + quality-measurement tool, disqualified at the perf gate by
  construction (serving-kernel.md §1a, §4).
- **0.6B only, tp=1.** No scale check at 4B/27B; no tensor parallelism (the
  packed-byte input dim is not TP-shardable here — fine at tp=1).
- **Top-20 KL, not full-vocab.** The served API exposes only top-20
  `prompt_logprobs`; the emu gate is full-vocab. On BF16-confident positions
  (top-1 mass > 0.5) top-20 covers the mass, so the comparison is tight, but the
  absolute KL is a top-20 figure. PPL (direct, actual-token NLL) is reported
  alongside.
- **Activations are emulated in bf16.** The plugin RTN-quantizes activations to
  the served bucket (fp4 group-16 / fp8 dynamic per-token) then runs bf16 MMA,
  exactly matching what the emulation gate measured — that is why served≈emu. A
  production kernel would send fp4 codes to the FP4 MMA (INV-2).
- **Codebook packaging deviation (documented).** The exporter's shipped layout
  puts `cb_codebook.*` in `model.safetensors`; for zero-vLLM-core-patch serving
  the driver splits them into a `cb_codebooks.pqcb` sidecar (still safetensors
  bytes, non-globbed extension) and inlines the full quant config into
  `config.json`. The plugin loads the sidecar once via
  `get_current_vllm_config()`. No functional effect on KL/PPL/speed.
- **NVFP4_CB_K16 is a stress artifact**, not a candidate — a uniform 2.5-bpw
  allocation is far below the CB knee; it is here to test emu↔served fidelity in
  the high-KL regime (which it passes: 2.21 → 2.246).

## 4. Correctness gate (passed)

The corresponding Gridbook kernel suite — 25/25 on the **real
exported** tensors: (a) the kernel's byte-window codeword extraction is
bit-identical to `nvfp4_cb_unpack`; (b) the decode-GEMM matches
`nvfp4_cb_reconstruct @ x` to ≤1e-2 rel (bf16 accumulation), M∈{1,17}, both
grids; (c) the fused qkv/gate_up per-row codebook-offset path is correct.
