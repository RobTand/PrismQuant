# PrismaQuant

**Mixed-precision LLM quantization that chooses the right format for every weight matrix, selected on real end-to-end KL — shipped as artifacts that stock inference engines serve with no forked runtime and no custom kernels.**

PrismaQuant's allocator is **AURA** (*Production-Faithful KL–Fisher Allocation*): a per-Linear cost model built from KL-Fisher probes of the full model multiplied against the *production-rendered* weight error — the exact bytes that ship — solved as a multi-choice knapsack, and gated by real KL measured on a held-out split before anything is published. Two output containers:

- **`compressed-tensors`** — vanilla vLLM serves it natively (`vllm serve $WORK_DIR/exported`). NVFP4 / FP8 / BF16 per Linear, CUTLASS kernels on Blackwell.
- **GGUF** — llama.cpp *and* vLLM (via the GGUF plugin) serve the same file. Full k-quant + IQ menu, per-tensor mixed, imatrix-weighted. This is how a 295B MoE fits on one 128 GB box.

---

## Headlines

All quality numbers below are **served-artifact measurements** (exact vLLM KL-vs-BF16 on held-out text, or deterministic benchmark runs on the served endpoint) — never local screens.

**Qwen3.6-27B PrismaAURA 5.5 bpp — tool-use fidelity above full precision.**

| Artifact | ToolEvalBench (hardmode, temp 0, same seed) |
|---|---:|
| **Qwen3.6-27B PrismaAURA 5.5 (23 GB)** | **91 / 100** |
| Qwen3.6-27B BF16 (full precision) | 86 / 100 |
| Qwen3.6-27B prior flagship (5.31 bpp) | 85 / 100 |

Served KL-vs-BF16 0.0342 — a −40.9% reduction over the prior AURA build at the same bpp, from adding the FP8 middle rung to the menu plus render/export fidelity fixes. [`rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm)

**Ornith-1.0-35B-A3B PrismaAURA 4.75 bpp — 70 GB → 23 GB, KL 0.0143.**

Served confident KL-vs-BF16 **0.0143** (top-1 agreement 98.6%), with a grafted MTP head for speculative decoding (91.3% acceptance at position 0). [`rdtand/Ornith-1.0-35B-PrismaAURA-4.75bit-vllm-MTP`](https://huggingface.co/rdtand/Ornith-1.0-35B-PrismaAURA-4.75bit-vllm-MTP)

**Qwen3.6-35B-A3B 4.75 bpp — wins 8 of 9 zero-shot metrics vs uniform NVFP4.**

Against RedHatAI's uniform NVFP4 quantization (342 hand-picked BF16 ignores), the measured allocation ships **2 GB smaller with ~90 fewer Linears in BF16** and lands ~4× closer to BF16 on mean zero-shot delta (−0.56 pp vs −2.21 pp). Uniform precision collapses the ~5% of genuinely sensitive Linears; measurement finds them. [`rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`](https://huggingface.co/rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm)

**Tencent Hy3 295B-A21B → 103.7 GB GGUF — serves on a single DGX Spark.**

A 295B/21B-active MoE quantized from the BF16 source to 2.80 bpp by measured allocation over the GGUF k-quant + IQ menu (streaming probe — the model never fits in memory; byte-budget selection targets the box, not a curve heuristic). **No quality claims** — models at this scale can't be KL-validated against their BF16 teacher on the target hardware; validation is load + coherent-generation smokes and bit-exact packing. [`rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm`](https://huggingface.co/rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm), plus a [5.3-bit `compressed-tensors` variant with MTP for two Sparks](https://huggingface.co/rdtand/Hy3-295B-A21B-PrismaQuant-5.3bit-2xSpark-vllm).

The artifact family is at **~400k downloads** on Hugging Face ([`rdtand`](https://huggingface.co/rdtand)).

---

## Quick start

**compressed-tensors lane (vLLM):**

```bash
export MODEL_PATH=/path/to/model
export WORK_DIR=./dq-runs/mymodel
export FORMATS=NVFP4,FP8_DYNAMIC,BF16   # the production menu — FP8 belongs in every recipe
export TARGET_BITS=4.75
export COST_MODE=aura                    # AURA cost (default: production-render-score)
export SELECTION_MODE=validated-surrogate  # real-KL frontier selection (default: surrogate)

./prismaquant/run-pipeline.sh
vllm serve $WORK_DIR/exported --quantization compressed-tensors
```

**GGUF lane (llama.cpp + vLLM):**

```bash
export EXPORT_CONTAINER=gguf TARGET_PROFILE=gguf COST_MODE=local
export PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0
export FORMATS=IQ2_XS,IQ3_XXS,IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0
export TARGET_BITS=2.9

./prismaquant/run-pipeline.sh
```

Every stage is also runnable on its own; the allocator takes the same menu as a
flag, so a re-solve against existing probe/cost artifacts needs no re-measurement:

```bash
python -m prismaquant.allocator --formats NVFP4,FP8_DYNAMIC,BF16 \
  --probe      $WORK_DIR/artifacts/probe.pkl \
  --costs      $WORK_DIR/artifacts/cost.pkl \
  --layer-config $WORK_DIR/artifacts/layer_config.json \
  --pareto-csv $WORK_DIR/artifacts/pareto.csv \
  --target-bits 4.75
```

The pipeline runs probe → cost → allocator → export → serve-smoke end-to-end, with fail-fast gates on any configuration that would break the measurement contract (it will tell you exactly why and what to set). Models too large for memory (200B+ MoE) run the streaming layer-by-layer path — peak memory is bounded by ~1 layer + a tunable cache.

---

## How AURA works

Mixed-precision quantization splits into two orthogonal questions:

- **Local (well-studied):** given a fixed format, how do you round this one Linear best? GPTQ, scale search, activation ordering — the per-tensor toolkit runs *under* whatever format is chosen.
- **Global (PrismaQuant's contribution):** how many bits should each Linear get, and in which hardware format? A per-Linear allocation of a total bit budget across the format menu.

AURA answers the global question with three commitments:

**1. A KL-faithful per-Linear cost.** The classical additive cost is `½ · H_trace · MSE_W` — a Fisher-diagonal trace times weight round-trip error. AURA replaces both halves: the sensitivity comes from KL-Fisher probes of the full model (the adjoint of the end-to-end KL objective, not a layer-local proxy), and the error term is measured on the **production-rendered** weights — GPTQ, joint scale optimization, activation ordering, the exact render that ships — not a raw round-trip. On served A/Bs at matched bpp this cost beats the strong `h_trace × output_mse` baseline by **−38% KL on a 4B model and −17.9% on 27B** (replicated across two calibration corpora).

**2. Production-faithfulness as an invariant.** The cost the allocator measures, the KL the validator measures, and the bytes the exporter ships all come from **one weight cache** — no rendering confound anywhere in the loop. On the GGUF lane the same contract holds through the imatrix: the calibration weighting used to measure a format's cost is byte-identical to the weighting used to pack it (enforced bit-exact against gguf-py, the llama.cpp reference decoder).

**3. Surrogates generate, real KL selects.** The additive surrogate is a candidate generator, never the ship decision. The allocator renders a Pareto sweep of candidates, each is measured with **real end-to-end KL on a held-out split** (disjoint from everything the cost stage saw), and the shipping point is picked on the measured `(bpp, KL)` frontier. Outside the surrogate's trust region, bpp order ≠ KL order: on the 27B, the surrogate's own knee picks 5.86 bpp / KL 0.056 while validated selection picks 5.31 bpp / KL 0.015 — smaller *and* 3.7× better, which is what end-to-end measurement buys over trusting the model of the cost.

**MoE hybrid.** Smooth per-Linear costs are structurally blind to router flips: quantizing a routed expert can change *which* experts fire, and no local error term sees that. AURA therefore allocates non-expert weights from the smooth cost and packed routed experts from **measured empirical unit-KL** (each expert-tier choice scored end-to-end), merged into one knapsack.

**What we measured and dropped.** The cross-layer-interaction literature (CLADO's pairwise IQP, HAWQ-V3's second-order ILP, cooperative-game formulations) models the coupling between Linears. We built the full L2 perturbed-activation cascade and ran the head-to-head on the served metric: cross-layer modeling bought **−1.5%**; AURA's better per-Linear cost bought **−38.5%** on the same A/B. Per-Linear cost fidelity was the lever; interaction modeling was not. That experiment — and the rest of the rejected-methods catalog (pairwise QUBO, Lagrangian λ-bisection as a selector, top-K Hessian covering, polish DPs that regress under the real-KL gate) — is documented in the paper. Negative results are recorded with their durable lesson; we don't re-litigate them silently.

Full derivations, the additivity/cancellation analysis, and the served evidence: [`paper/main.pdf`](paper/main.pdf).

---

## Pipeline

```
incremental_probe ──────► probe.pkl        (KL-Fisher sensitivity per Linear; streaming for 200B+)
        │
cost stage ─────────────► cost.pkl         (COST_MODE=aura: KL-Fisher × production-rendered dW;
        │                                   MoE hybrid merges measured expert unit-KL)
allocator ──────────────► layer_config.json + Pareto candidates
        │                                  (multi-choice knapsack; fused-sibling & packed-MoE
        │                                   format coherence via union-find promotion)
validate_assignments_kl ► held-out real-KL per candidate
select_validated_frontier► the shipping point, picked on measured (bpp, KL)
        │
export ─────────────────► exported/        (compressed-tensors  OR  GGUF via export_gguf)
        │
validate_native_export ─► engine load + greedy-decode smoke (eager and graph mode)
validate_quantized_model► PPL / p99 per-prompt NLL / MTP-acceptance ship gate
```

Every hot path is GPU-bound by design; the pipeline refuses to run on CPU.

---

## Formats

**compressed-tensors container (vLLM-native):**

| Format | Role |
|---|---|
| NVFP4 | W4A4, group-16 FP8 block scales; CUTLASS on Blackwell — the 4-bit workhorse |
| FP8_DYNAMIC / FP8_E4M3 | the 8-bit rung — on the menu in every production recipe; it's what bends the rate-distortion curve into a knee |
| BF16 | passthrough only (never synthesized from a quantized source) |
| FP8_SOURCE | byte-exact passthrough of natively-FP8 checkpoints (lossless at ~8 bpp) |
| MXFP4/6/8, INT4/INT8, NVFP4A16 | registry-supported research rungs, excluded from defaults where a served kernel or a Pareto case is missing |

**GGUF container (llama.cpp + vLLM GGUF plugin):**

| Format | bpw |
|---|---|
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 | 2.625 / 3.44 / 4.5 / 5.5 / 6.56 / 8.5 |
| IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL | 2.06 – 4.5 (E8-lattice codebooks; the sub-Q2_K regime) |

GGUF quantizers are PrismaQuant's own GPU implementations (imatrix-weighted grid/least-squares search, exhaustive codeword search for IQ), validated bit-exact against gguf-py. GPTQ-under-frozen-scales is available for the k-quants as a research lever. The allocator is serving-profile-aware in both containers: it never picks a format the target engine can't serve at that tensor's shape.

---

## Architectures

First-class profiles in-tree:

- **Qwen3 / 3.5 / 3.6** — dense + packed-3D MoE + MTP heads, Gated-DeltaNet hybrids
- **Tencent Hy3** (`hy_v3`) — 295B/21B-active, 192-expert MoE + MTP sidecar
- **DeepSeek-V4-Flash** (vendored transformer + profile) — 671B-class, per-expert-Linear MoE
- **MiniMax M2 / M2.7** — nested per-expert MoE, native-FP8 source
- **Gemma4** — multi-layer-type rope, KV sharing, visual/text profiles
- **LFM2.5** — per-expert MoE + short-conv layers

A new architecture is a `model_profiles/` registration (structure spec JSON + a small profile class); most of the fused-group and naming plumbing is auto-derived from the vLLM model class, so ports land in ~30–200 LoC.

---

## Published artifacts

| Model | Artifact |
|---|---|
| Qwen3.6-27B | [PrismaAURA 5.5 bpp](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm) · [5.31 bpp prior flagship](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm) (DOI 10.57967/hf/8656) · [v1 5.5 bpp](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm) |
| Qwen3.6-35B-A3B | [4.75 bpp](https://huggingface.co/rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm) |
| Ornith-1.0-35B | [PrismaAURA 4.75 bpp + MTP](https://huggingface.co/rdtand/Ornith-1.0-35B-PrismaAURA-4.75bit-vllm-MTP) |
| Tencent Hy3 295B-A21B | [2.8 bpp GGUF, single Spark](https://huggingface.co/rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm) · [5.3 bpp CT + MTP, 2× Spark](https://huggingface.co/rdtand/Hy3-295B-A21B-PrismaQuant-5.3bit-2xSpark-vllm) |
| Qwen3.5-122B-A10B | [4.75 bpp](https://huggingface.co/rdtand/Qwen3.5-122B-A10B-PrismaQuant-4.75bit-vllm) |
| Mistral-Medium-3.5-128B | [4.75 bpp](https://huggingface.co/rdtand/Mistral-Medium-3.5-128B-PrismaQuant-4.75-vllm) |
| MiniMax-M2.7 | [3.20 bpp](https://huggingface.co/rdtand/MiniMax-M2.7-PrismaQuant-3.20bit-vllm) |
| Gemma4-31B-IT | [6 bpp](https://huggingface.co/rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm) · [5.5 bpp](https://huggingface.co/rdtand/Gemma4-31B-IT-PrismaQuant-5.5bit-vllm) |
| LFM2.5-8B-A1B | [6.5 bpp](https://huggingface.co/rdtand/LFM2.5-8B-A1B-PrismaQuant-6.5bit-vllm) |

---

## Method discipline

The rules this project holds itself to, because low-bit quantization results are unusually easy to overstate:

- **Promote on the serving metric, not the screen.** A win counts only when it holds on exact full-vocab vLLM KL-vs-BF16 and direct perplexity *on the served artifact* at matched bpp. We have watched local-PPL "wins" invert on the served A/B more than once; those methods are archived with the lesson, not quietly dropped.
- **KL screens, it doesn't ship alone.** Lower mean KL can hide a heavier tail; candidates also clear direct PPL, p99 per-prompt NLL, and tool-use benchmarks before publication.
- **Held-out means held-out.** Selection KL uses text the cost stage never saw.
- **No hand-tuned bans.** If the allocator picks something that breaks, the cost model is wrong — fix the measurement, don't constrain the optimizer. The only constants allowed are ones derived from a dtype's numerical precision.
- **No quality claims we can't back.** The 295B artifact ships with an explicit "no quality claims" section because its teacher can't be measured on the target hardware. Provenance (git commit, calibration hash, assignment hash) is baked into every artifact's metadata.

---

## Citation

```bibtex
@software{prismaquant2026,
  title  = {PrismaQuant: Mixed-Precision LLM Quantization Selected on Real End-to-End KL},
  author = {Tand, Robert},
  year   = {2026},
  url    = {https://github.com/RobTand/prismaquant},
}
```

The AURA paper (*AURA: Production-Faithful KL–Fisher Allocation*) is at [`paper/main.pdf`](paper/main.pdf) (source: [`paper/main.tex`](paper/main.tex)). Contact: robert.tand@icloud.com.

## Acknowledgements

PrismaQuant is assembled on top of a decade of quantization research. Key influences: CLADO (Deng et al. 2023) for the cross-layer decision-unit framing (whose interaction-modeling half our head-to-head then retired); HAWQ-V1–V3 (Dong et al. 2019–2021), CoopQ, and AMQ on mixed-precision allocation; GPTQ (Frantar et al. 2022) and its Babai-nearest-plane interpretation (Chen et al. 2026) for rounding; the llama.cpp/ggml project for the GGUF formats and reference implementations; Kneedle (Satopää et al. 2011); and Cover & Thomas ch. 13 on rate-distortion bit allocation. Full bibliography in [`paper/main.tex`](paper/main.tex).
