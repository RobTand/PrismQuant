# PrismaQuant Architecture

As of: 2026-08-03 · branch `release/prismaquant-0.8.0` · verified against implementation
baseline commit `7183d21`, with the external Gridbook runtime pinned to
`9011a19228ddb96b8a49e11a20ac75c99c83998e` (v0.8.0). This branch ports the dated
2026-08-01 DeepSeek-V4-Flash-0731 92 GB study record (§9.2) forward from its 0.5.1
working tree; the study's Gridbook-candidate claims were **not** carried over, because
the candidate they described has since been reviewed, cut, and pinned as Gridbook 0.6.0.

This revision retains the four 2026-07-30 architecture re-vet waves documented in
`docs/audits/architecture_re-vet_2026-07-30.md` and closes the runtime-ownership debt: the
vendored Gridbook tree and sync path are gone, producer ABI/menu/config facts have one owner,
and required CI checks the independent producer and consumer at one immutable commit. The 0.6.0
release advances that boundary to Gridbook 0.6.0 and lands the producer half of the cross-repo
performance ultraplan (P5a–P5d, K0.2): candidates are priced and described differently and gain
a second hard constraint axis, while the producer ABI, format menu, export defaults, and
quality-promotion status are unchanged and DSv4 remains gated. The three behavioural facts a
returning reader must know are that **`COST_MODE` defaults to `aura`** (§3.3), Gridbook serving
is native CUDA/CUTLASS-only and fails closed (§9.2), and fused native-NVFP4 remains default-off
after its teacher-backed quality gate (§9.2).

**Prime directive:** the code is the authority. Where this document and the tree disagree, the
document is wrong — fix it, or record the divergence in §12; never propagate it.

---

## Contents

[0 Maintenance contract](#0-maintenance-contract) ·
[1 What PrismaQuant is](#1-what-prismaquant-is) ·
[2 Methodological spine](#2-methodological-spine) ·
[3 The quantization pipeline](#3-the-quantization-pipeline) ·
[4 Cost models & allocation](#4-cost-models--allocation) ·
[5 Formats & render](#5-formats--render) ·
[6 Export & serving invariants](#6-export--serving-invariants) ·
[7 Validation & ship gates](#7-validation--ship-gates) ·
[8 Model support: the plugin architecture](#8-model-support-the-plugin-architecture) ·
[9 Serving lanes](#9-serving-lanes) ·
[10 Hardware & environment](#10-hardware--environment) ·
[11 History](#11-history--what-was-tried-and-rejected) ·
[12 Known gaps and debt register](#12-known-gaps-and-debt-register)

## 0. Maintenance contract

This file is the master document. `docs/README.md` is the index and carries a status tag
(CURRENT / HISTORICAL / ARCHIVED) per doc; everything else is a rule set this file points at,
a lane record, or history.

**The rule.** A commit that changes any of the following must update this file in the same
commit: (1) a `prismaquant/run-pipeline.sh` default, gate, or stage order (§3); (2) the format
menu, a scale rule, or a render lever (§5); (3) an export codec, a `config_groups` emission
rule, or a serving invariant (§6); (4) a ship-gate threshold or what the pipeline runs versus
echoes (§7); (5) the plugin contract — profile accessors, registry order, serving-profile
schema, gridbook per-arch wiring (§8); (6) a serving-lane default or a promoted/reverted kernel
lever (§9). If topology changed, the affected mermaid diagram changes with it. The provenance
block at the top must be re-stamped (date, commit, branch) on every substantive edit.

**Corollaries.** Handovers (`docs/handovers/`, gitignored) and dated results (`docs/results/`,
`docs/lanes/*/`) are append-only history: they record what was true on a date and never
substitute for updating this file. Every normative claim here carries a `file:line` or a commit
hash — a claim without one is a lead, not a fact. Staleness discovered in this document is a
bug: fix it, or if the fix is larger than the edit, add it to §12 with a severity. Do not
silently leave a wrong line here for the next reader.

## 1. What PrismaQuant is

Mixed-precision LLM quantization that chooses a serving format **per Linear**, selects the
assignment on **real end-to-end KL-vs-BF16**, and ships the result as an artifact a stock or
plugin-extended vLLM serves. No forked runtime on the native lane. Allocation is a
multiple-choice knapsack over a per-(Linear, format) cost (§4); the winning candidate is decided
by measurement, not by the cost model (§2).

### 1.1 The three artifact containers

| Lane | Container | Runtime | Formats | Status |
|---|---|---|---|---|
| Native | `compressed-tensors` | vanilla vLLM, Blackwell CUTLASS | NVFP4, FP8_DYNAMIC/E4M3, FP8_SOURCE, BF16 | production default |
| CB ("gridbook") | `nvfp4_cb` codebook checkpoint | vLLM + the separately versioned `gridbook` package (native CUDA/CUTLASS-only, fail-closed), installed from the exact commit in `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` | FP4-CB / FP8-CB rungs plus the native menu | production only for architectures declared by Gridbook's packaged runtime contract; DSv4 remains gated |
| GGUF | single `.gguf` | llama.cpp; vLLM via `vllm-gguf-plugin` | Q2_K…Q8_0 k-quants + IQ family + BF16 | enabled end-to-end; the only 2–3 bpw path |

Lane detail, defaults and proven results: §9. Export codecs: §6. Pipeline defaults: §3.3.

### 1.2 Shipped artifact family

bpp is over **quantizable** parameters only (excludes `lm_head`, MTP/visual sidecars, pinned
Linears) and labels are **not** comparable across accounting eras (§12). conf-KL =
confident-position KL-vs-BF16; ALL-KL = all positions. Comparative lane deltas belong to §9;
the numbers below are each artifact's own readout.

| Artifact | Lane | bpp | Quality readout | Provenance |
|---|---|---|---|---|
| Qwen3.6-27B `prismaquant-cb-5.5bit-vllm` | CB | 5.501 | ALL-KL **0.0134** / conf-KL 0.0113; PPL 9.166 vs BF16 9.123 | `docs/lanes/nvfp4-cb/prod_27b_results.md` |
| Qwen3.6-27B `PrismaAURA-5.5bit` | native | 5.5 | ALL-KL 0.0321 / conf-KL 0.0241; TEB 91 (BF16 86) | same A/B table; TEB from memory, unverified vs code |
| Qwen3.6-27B PrismaSCOUT 5.31 (DOI `10.57967/hf/8656`) | native | 5.31 (≈4.76 under current accounting) | held-out KL 0.0151, 20.17 GB | superseded by the two rows above |
| Ornith-1.0-35B (CB) | CB | 4.758 | conf-KL **0.01706** / ALL-KL 0.0278; PPL 9.542 (+1.1%) | `docs/lanes/nvfp4-cb/prod_35b_results.md` |
| Ornith-1.0-35B PrismaAURA | native | 4.748 | conf-KL 0.03625 re-measured; the older **0.0143** figure is a different protocol and is *not* comparable | `prod_35b_results.md` |
| Hy3-295B-A21B `prismaquant-cb-2.9bit-vllm` | CB | 2.902 | no quality claim possible (no 295B BF16 teacher on one box); TEB 87/100; serves on one Spark | `docs/lanes/nvfp4-cb/prod_hy3_results.md` |
| Hy3-295B-A21B `PrismaQuant-2.8bit-gguf-vllm` | GGUF | 2.799 (103.686 GB) | TEB 87/100 (IQ) vs 86 (k-quant) | `docs/lanes/gguf.md` |
| Hy3-295B-A21B `PrismaQuant-5.3bit-2xSpark-vllm` | GGUF | 5.3 (190 GB) | two-Spark target | memory, unverified vs code |
| Laguna-S-2.1 117B | CB | 6.0 (84 GB) | no BF16 teacher at 117B; serves 256k ctx | memory `laguna_s21_lane`, unverified vs code |
| Gemma4-31B-IT | native | 6.0 | −24% conf-KL vs the shipped 5.5, +5.9 pp top-1 | memory, unverified vs code |
| LFM2.5-8B-A1B | native | ~6.58 (labelled 6.5) | ToolEvalBench = BF16 parity | memory, unverified vs code |
| Qwen3.5-122B-A10B · Mistral-Medium-3.5-128B · Qwen3.6-35B-A3B | native | 4.75 | — (the 35B-A3B predates 4 allocator/export fixes; do not re-export without an orthogonal reason) | memory, unverified vs code |
| MiniMax-M2.7 | native | 3.2 | — | memory, unverified vs code |

The two CB rows carry the load-bearing result: at matched body bytes, codebook formats buy
materially more quality. The magnitudes, and the speed side of the trade, are §9.2.

Author: Robert Tand, independent researcher; public attribution uses
`robert.tand@icloud.com`. Paper: `paper/main.tex` (AURA spine; the PrismaSCOUT spine was
retired 2026-06-05 and archived at `paper/archive/prismascout_paper_2026-06-05.tex`).

## 2. Methodological spine

### 2.1 Two axes

- **Local** — *given a fixed format, how do you round this Linear best?* Well studied: GPTQ,
  AutoRound, rotations, scale rules. The render toolkit; it runs *under* whatever format is
  chosen (§5).
- **Global** — *how many bits does each Linear get, and in which hardware format?* Allocation,
  and the contribution (§4). Sensitivity is wildly unequal across a transformer's Linears, so a
  heterogeneous assignment extracts quality no single-format method structurally can.

### 2.2 Surrogates generate, real KL selects

The governing sentence: *an allocator does not need a perfect cost model if every candidate it
proposes can be cheaply re-scored end-to-end on a held-out split.* Cross-layer interaction
therefore stops being a quantity you must **model** and becomes one you **observe**.

The modelling branch of the literature (CLADO's pairwise IQP, HAWQ-V3's second-order ILP,
CoopQ's Shapley allocator) is not reproduced here: measured pairwise interaction is noise at
the bit-widths that ship (3/1180 pairs significant; pair-term ρ = −0.10), and the apparent
non-additivity is largely a bf16 KL-differencing floor — in fp32 the per-Linear unary KLs are
near-additive (`paper/main.tex` §additivity; §11).

**There is one cost level, not three.** The three-level cascade (L1 additive Fisher → L2
perturbed-X fixed point → L3 propagated end-KL) was **retired from the spine on 2026-07-30**
and its code walled at `archive/l3_propagated_2026-07-30/` (re-vet R4). What ships is *one
faithful unary cost* — **AURA by default since 2026-07-30** (re-vet R2),
`production-render-score` as the explicit/legacy spelling, plus measured empirical unit-KL for
packed experts — and *real held-out KL* to select among
the candidates that cost proposes. The retiring evidence is measured, not argued: the L2
fixed point beat additive L1 by **−1.5%** while AURA beat L1 by **−38.5%** on the same
baseline (`aura_cascade_headtohead`); a better single cost was worth 25× more than another
level. Status, citations and what survives: §4.4; wall and lesson: §11.

### 2.3 Metric authority

Highest first. A claim is worth exactly the rung it was measured on.

| # | Metric | Contract | Where |
|---|---|---|---|
| 1 | Exact full-vocab vLLM KL-vs-BF16 on the served artifact, matched bpp | n=8 × seqlen=512 | `tools/measure_vllm_full_kl.py:461-462` — invoked **manually**, never by the pipeline |
| 2 | Direct WikiText PPL on the served artifact | 8192 tokens, seqlen 512 | `tools/measure_vllm_wikitext_ppl.py:78-79` — manual |
| 3 | Mean NLL alongside PPL; KL-vs-BF16 (`/home/rob/dq-runs/kl_tool.py`) for IT/BOS-sensitive models where raw PPL is meaningless | — | §7.5 |
| 4 | Downstream suite on materialized artifacts: GSM8K, IFEval, MMLU, **ToolEvalBench** (`--no-think --hardmode --parallel 1`) | — | tool-use fidelity is the deep reason KL matters: a small probability shift at a decision point flips a tool call |
| 5 | Cheap last-token "hook KL" screens | — | **triage only**; never a selection or promotion metric |

Rung 2 can veto a rung-1 win — a lower *mean* KL can hide a heavier tail. A candidate that
improves calibration KL but regresses held-out PPL/NLL or a downstream task stays
research-only unless Robert explicitly accepts the trade. (The selector has no tail term
today; §12 D1.)

**Held-out discipline.** The selection split must be disjoint from the text that generated any
cost — an audit found "validation" KL had been in-sample; the house rule and the
token-disjoint construction are documented at `validate_assignments_kl.py:513,581`. Small-scale
levers are validated on Qwen3-0.6B *and* 4B with `--calib-repeats ≥ 4`; single-seed n=8/T=512
is dangerously noisy (+10% can flip to −5.2% across repeats).

**Reproducibility is a gate.** Git commit, calibration hash, assignment hash and cache
hit/miss/RTN-fallback counts are baked into output JSON; an irreproducible number is
quarantined, not trusted. KL is bit-identical within a docker session and drifts across them —
mechanism, magnitude and the resulting A/B rule are §7.4.

### 2.4 Promotion ladder

| Stage | Bar |
|---|---|
| Research | opt-in, documented, excluded from defaults |
| Candidate | small-model GPU + vLLM smokes, plus a measurement plan on a real target |
| Production recipe | wins or preserves KL/bpp/runtime on the target stack; serving suite green; tests |
| Default-on | cleared on the target **and** one more representative model/shape |

Regression or inconclusive → demote back to Research. The numeric ship gate that guards
materialization is separate, automated and thresholded (§7.2).

### 2.5 Honest accounting

Retraction is routine and is itself a deliverable. The grouped-KL surrogate's "−3.52% PPL win"
was a local/HF screen that **inverted** on the vLLM A/B; the "17 promotions / 0.0056 KL" polish
headline and the "4× lower KL" framing were withdrawn the moment the comparisons were found
non-rigorous; the staged-render last-token-KL win regressed direct PPL; `current_only`
extrapolation won its hook screen and lost full-vocab KL; the damp sweep's "+137.5% if
disabled" was a hook screen that inverted on the gold lane. Hence the rule: **never sell a
screen as a result.** Expect most pipeline "improvements" to be <5% deltas — the cost surrogate
is itself mis-ranked against PPL at the margin (5.5 bpp beats 6.0 bpp on Qwen3-4B WikiText
PPL). Negative results are recorded with the durable lesson (§11); the paper publishes the
graveyard.

## 3. The quantization pipeline

The orchestrator is `prismaquant/run-pipeline.sh` — **not** the repo root; several older docs
imply a root-level copy and there is none. One bash script, four numbered phases (probe → cost
→ allocate → cache+export), each phase file-artifact-coupled and skip-if-exists.
`prismaquant/pipeline.py` is a *declarative* spec layer invoked once at the top; it executes
nothing (§3.6).

**DIAGRAM-1 — Pipeline dataflow:** source checkpoint to three artifact containers, with the
four `COST_MODE`s, the opt-in validated-frontier loop, and the manual (echoed-only) ship gate.

```mermaid
flowchart TD
  SRC["source checkpoint<br/>HF safetensors"]
  PROBE["[1/4] incremental_probe -- run-pipeline.sh:544-560<br/>per-Linear empirical Fisher h_trace<br/>artifacts/probe.pkl"]
  ACT["activation cache<br/>WORK_DIR/act"]
  BASE["[2/4] incremental_measure_quant_cost -- :645-658<br/>RTN per-Linear-per-format error<br/>cost.pkl (local) or cost_baseline.pkl"]

  SRC --> PROBE
  PROBE --> ACT
  PROBE --> BASE
  ACT --> BASE

  subgraph COST["cost stage -- one of three COST_MODEs, dispatched in the COST_MODE case"]
    PRS["production-render-score -- explicit/legacy<br/>build_production_cache --render-scope format-menu<br/>then production_render_cost -> cost.pkl"]
    LOC["local<br/>the RTN base cost IS the allocator cost<br/>the CB/GGUF lanes shipping recipe"]
    AUR["aura -- DEFAULT since 2026-07-30<br/>aura_cost -> cost_aura.pkl<br/>then expert_empirical_cost --merge-base -> cost.pkl<br/>then the [3c] additivity report"]
    CBH["CB sub-stage (:966-1035)<br/>cb_col_weights.pkl imatrix harvest, then<br/>expert_empirical_cost --replace-experts"]
  end

  BASE --> PRS
  BASE --> LOC
  BASE --> AUR
  LOC --> CBH

  ALLOC["[3/4] allocator + allocator_solver -- :1076-1090<br/>multi-choice knapsack DP over Linear x format<br/>union-find serving-unit promotion<br/>artifacts/layer_config.json + pareto.csv"]

  PRS --> ALLOC
  LOC --> ALLOC
  AUR --> ALLOC
  CBH --> ALLOC

  subgraph VS["SELECTION_MODE=validated-surrogate -- OPT-IN; default is surrogate (:250)"]
    FR["A. build_production_cache --render-packed-experts<br/>production_weight_cache_frontier_raw.pkl"]
    VAK["B. validate_assignments_kl -- :1243-1277<br/>measured held-out KL per Pareto point<br/>validated_frontier_kl.json"]
    SVF["C. select_validated_frontier -- :1281-1288<br/>kneedle -> rewrites layer_config.json"]
  end

  ALLOC --> FR
  FR --> VAK
  VAK --> SVF

  PCACHE["[4/4] D. build_production_cache / production_recache<br/>ProductionWeightCache -- the one rendered-weight store<br/>levers: gptq, static_act_order, joint_scale_opt"]

  SVF --> PCACHE
  ALLOC -->|"SELECTION_MODE=surrogate"| PCACHE

  EXPCT["export_native_compressed -- :1665-1699"]
  EXPCB["export_nvfp4_cb or export_nvfp4_cb_streaming<br/>auto-switch above 80 GB source (:1585-1641)"]
  EXPGG["convert_hf_to_gguf.py skeleton -> export_gguf<br/>(:1461-1493)"]

  PCACHE --> EXPCT
  ALLOC -->|"EXPORT_CONTAINER=nvfp4_cb, PRODUCTION_CACHE=0"| EXPCB
  ALLOC -->|"EXPORT_CONTAINER=gguf, PRODUCTION_CACHE=0"| EXPGG

  OUTCT["compressed-tensors checkpoint<br/>WORK_DIR/exported"]
  OUTCB["CB checkpoint + quant_config.json + cb_codebooks.pqcb<br/>WORK_DIR/exported_nvfp4_cb"]
  OUTGG["single-file GGUF<br/>WORK_DIR/exported.gguf"]

  EXPCT --> OUTCT
  EXPCB --> OUTCB
  EXPGG --> OUTGG

  GGSMOKE["llama-completion greedy smoke<br/>in-lane, :1500-1516"]
  NOSMOKE["no in-lane serving smoke<br/>gate set declared in lane_specs/nvfp4_cb.json (R16)"]
  OUTGG --> GGSMOKE
  OUTCB --> NOSMOKE

  subgraph GATE["ship gate -- NOT executed by the pipeline"]
    VNE["validate_native_export<br/>vLLM eager+graph load + greedy smoke<br/>echoed at :1704-1705"]
    VQM["validate_quantized_model<br/>PPL 25 / mean-NLL 3 / worst-NLL 6 / MTP p0 0.60<br/>validate_quantized_model.py:116-120 -- never echoed"]
    GOLD["gold lane, invoked by hand<br/>tools/measure_vllm_full_kl.py:461-462 -- n=8 x 512 full-vocab KL<br/>tools/measure_vllm_wikitext_ppl.py:78-79 -- 8192-token PPL"]
  end

  OUTCT --> VNE
  VNE --> VQM
  VQM --> GOLD
  NOSMOKE --> GOLD
  GGSMOKE --> GOLD

  classDef optin stroke:#c07800,stroke-width:2px,stroke-dasharray:4
  classDef manual stroke:#c0392b,stroke-width:2px
  class AUR,CBH,FR,VAK,SVF optin
  class VNE,VQM,GOLD,NOSMOKE manual
```

### 3.1 Pre-flight gates

In order, all failing `exit 2`: required `MODEL_PATH`/`WORK_DIR` (`43-44`); GGUF lane
consistency (`97-110`); CB lane consistency (`119-132`); GPU-or-bust — both `DEVICE` and
`EXPORT_DEVICE` must match `cuda*` and an inline `python3` asserts `torch.cuda.is_available()`
(`134-145`); the archived-lever gates of §3.5 (`233-248`, `337-340`, `387-406`); `COST_MODE`
dispatch, unknown mode rejected (`314-385`); work-dir creation (`408`); `SELECTION_MODE`
legality (`410-416`); `MSE_PROMOTION` legality — requires validated-surrogate and a production
cache (`417-429`); spec write/validate (`462-481`).

The two lane gates encode one contract, and since re-vet **R3** they say so directly (§4.7):
the GGUF and CB exporters requantize the bf16 skeleton with **imatrix-weighted** renders, so
the render that produces the allocator's cost must be imatrix-weighted too — a
**render-faithfulness assertion**, keyed off the format family, not a `COST_MODE` whitelist.
`PRODUCTION_CACHE=0` + the matching `TARGET_PROFILE` remain mandatory for their own reasons
(those exporters never read the export cache; the exporter hard-fails on out-of-profile
formats). The lanes themselves are §9.

### 3.2 Stage table (execution order)

Line refs are `run-pipeline.sh` unless stated. Artifact paths are relative to `$WORK_DIR`.
**The line numbers in this table are indicative, not a contract** — the four 2026-07-30 re-vet
waves moved them repeatedly, and §3.3 already dropped them for exactly that reason. The
non-decaying anchor is the bracketed **stage label the script echoes** (`grep '\[2d-CB\]'`),
which is what the rows touched since are keyed on.

| # | Stage | Script | Artifact(s) | Reuse guard | Mode/lane gate |
|---|---|---|---|---|---|
| **1/4** | Sensitivity probe — per-Linear empirical Fisher `h_trace`, body + MTP in one pass; tied heads materialized and excluded, KV-sharing cotangents grafted (§7.5) | `prismaquant.incremental_probe` (`544-560`) | `artifacts/probe.pkl`; activations → `act/`; shards → `work/`; `logs/probe.log` | settings-hash `probe` (`703`); reuse also re-checks stored `calibration_modality` | — |
| **2/4** | Baseline per-(Linear,format) RTN cost | `prismaquant.incremental_measure_quant_cost` (`645-658`) | `artifacts/cost.pkl` (`COST_MODE=local`) or `artifacts/cost_baseline.pkl` (`314-380`); `logs/cost.log` | settings-hash `base-cost` (`768`) + cost-mode provenance when it IS the allocator table (`769-777`) | — |
| **2a-CB** | imatrix column-weight harvest | `harvest_cb_col_weights` — ONE shell function, four call sites (`[2/4] pre-cost`, `[2b/4] cost-cache`, `[2d-CB]`, `[4/4]`) → `export_gguf.build_imatrix_from_act_cache` + `moe_imatrix.synthesize_packed_expert_col_weights` | `artifacts/cb_col_weights.pkl` | settings-hash `cb-col-weights` | CB lane; called by whichever stage needs the vector first |
| **2b/4** | Format-menu production render for allocator cost | `build_production_cache --render-scope format-menu` (`672-686`) | `artifacts/production_render_score_cache.pkl` + `…_weight_cache/` | settings-hash `render-cost-cache` (`837`) | `production-render-score` |
| **2c/4** | Synthesize allocator cost from render scores | `prismaquant.production_render_cost` (`704-711`) | `artifacts/cost.pkl` | settings-hash `render-cost` (`858`) + cost-mode provenance (`859`) | `production-render-score` |
| **2b/4** | Format-menu cache for AURA dW | `build_production_cache … --render-scope format-menu` (`857-871`) | frontier cache under validated-surrogate, else `production_render_score_cache.pkl` (`366-378`) | settings-hash `aura-dw-cache` (`913`) | `aura`; `exit 2` if the menu is BF16-only |
| **2c/4** | AURA downstream-KL-adjoint cost | `prismaquant.aura_cost` (`881-900`) | `artifacts/cost_aura.pkl` | settings-hash `aura-cost` (`939`) | `aura` |
| **2d/4** | Hybrid finalize: empirical packed-expert unit-KL + sidecar backfill | `prismaquant.expert_empirical_cost --merge-base --backfill-base` (`920-929`) or inline backfill (`932-952`) | `artifacts/cost.pkl` | settings-hash `aura-hybrid-cost` (`971`) + cost-mode provenance (`972`) | `aura` |
| **2d-CB** | CB hybrid: replace packed-expert rows with empirical unit-KL | `harvest_cb_col_weights "[2d-CB]"` → `expert_empirical_cost --replace-experts --col-weights` | `artifacts/cost_local_raw.pkl`, `artifacts/cost.pkl`, `cb_col_weights.pkl` | settings-hash `cb-hybrid-cost` + the in-payload merge probe; col-weights `cb-col-weights` | CB lane, `CB_EXPERT_EMPIRICAL=1` |
| **2b/4 cw** | Cost-cache col-weights (weighted lanes only) | `harvest_cb_col_weights "[2b/4] cost-cache"` → `build_production_cache --col-weights` | `artifacts/cb_col_weights.pkl` | settings-hash `cb-col-weights` | `COST_RENDER=cached-menu` on a CB/GGUF lane (§4.7) |
| **3/4** | Allocator — multi-choice knapsack over per-Linear formats (§4) | `prismaquant.allocator` (`1076-1090`) | `artifacts/layer_config.json`, `artifacts/pareto.csv`, `artifacts/pareto_assignments/` (validated-surrogate only, `1056-1061`); `logs/allocator.log` | **none — always runs** | — |
| **4/4 A** | Frontier format-menu cache | `build_production_cache … --render-scope format-menu --render-packed-experts` | `artifacts/production_weight_cache_frontier_raw.pkl` + `…_frontier/` | settings-hash `frontier-cache` (`1206`) | validated-surrogate; `exit 2` if `PRODUCTION_CACHE=0` |
| **4/4 B** | Measured held-out KL per Pareto point | `prismaquant.validate_assignments_kl` (`1243-1248` per-point, `1272-1277` batched) | `artifacts/validated_frontier_kl.json` + `…_parts/*.json` (merged `1250-1269`) | settings-hash `frontier-kl-point` per point (`1294`) | validated-surrogate |
| **4/4 C** | Frontier point selection | `prismaquant.select_validated_frontier` (`1281-1288`) | overwrites `artifacts/layer_config.json`; `layer_config_validated_assignment.json`; `validated_frontier_selection.json` | none | validated-surrogate |
| **4/4 D** | Production cache build / recache for the selected assignment | `production_recache` (`1331-1346`, `1399-1414`) or `build_production_cache --recache-layer-config` (`1379-1396`, `1423-1438`) | `production_weight_cache_frontier_<digest>_recached.pkl` (`1328`), `production_weight_cache_recached.pkl` / `…_raw.pkl` (`1102-1103`) | settings-hash `production-cache-recached` (`1404`), `frontier-recache` (`1360`), `production-cache-raw` (`1452`) | `PRODUCTION_CACHE=1` |
| **3c** | AURA additivity report — `residual = measured_end_KL − Σ predicted_dloss`, stamped into `cost.pkl` `provenance["additivity"]` (§4.3) | `prismaquant.aura_additivity_gate` (+ optional `validate_assignments_kl` under `AURA_ADDITIVITY_GATE=measure`) | `artifacts/aura_additivity.json`, `aura_additivity_kl.json` (measure only); `logs/aura_additivity*.log` | none — non-blocking report, skip-if-exists on the KL half | `COST_MODE=aura`, `AURA_ADDITIVITY_GATE≠0` |
| **4/4 E-gguf** | GGUF skeleton + export + llama.cpp smoke | `convert_hf_to_gguf.py` (`1461-1464`), `prismaquant.export_gguf` (`1469-1493`), `llama-completion` (`1500-1516`) | `artifacts/skeleton.gguf`, `exported.gguf` | settings-hash `gguf-skeleton` (`1488`); export always runs | GGUF lane; **exits 0** |
| **4/4 E-cb** | CB col-weights + codebook export | `harvest_cb_col_weights "[4/4]"`, `export_nvfp4_cb[_streaming]` | `exported_nvfp4_cb/` | settings-hash `cb-col-weights`; export always runs | CB lane; no in-lane serving smoke; **exits 0** |
| **4/4 E** | compressed-tensors export (§6) | `prismaquant.export_native_compressed` (`1665-1699`) | `exported/`; `logs/export.log` | **none — always runs** | default lane |

**Nothing in the pipeline validates the artifact** — that is a physical lane boundary (`vllm`
is not importable in the build venv), not laziness. What the script now does instead of
echoing a suggested command is **print the open ship record**: the closing block runs
`python -m prismaquant.shipcard_cli show <exported>/shipcard.json` and names every slot still UNFILLED
(re-vet R13 + the deferred wave-1 item). The build lane opens the record; the serve lane must
close it. Both the numeric ship gate and the gold-lane KL/PPL contracts remain manual; §7
owns that.

### 3.3 Defaults at HEAD (`8f14400` + the 2026-07-30 re-vet waves)

This table is the single source of truth for pipeline defaults; other sections reference it
rather than restate it. `tests/test_architecture_doc.py` pins the enumerable half against
`run-pipeline.sh`, so a default change that skips this table fails the suite. (Line numbers
were dropped from this block: the re-vet waves shifted them, and a stale `file:line` is worse
than none — `grep ': "${NAME:='` is exact and never decays.)

```
FORMATS=NVFP4,FP8_DYNAMIC,BF16   TARGET_BITS=4.75
PARETO_TARGETS=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25
NSAMPLES=32  SEQLEN=1024  DATASET=…/calibration/diverse-v1.jsonl
EXPERT_GATE_DATASET=…/calibration/xdom-gate-v1.jsonl (cross-domain)
ACTIVATION_ROWS_LIMIT=1024 on the GGUF/CB lanes else 256
COST_MODE=aura (R2 flip 2026-07-30; = COST_RENDER=cached-menu x
                COST_OBJECTIVE=aura-adjoint) PRODUCTION_CACHE_PREFETCH=require
PRODUCTION_RENDER_COST_SCORE_FIELD=weight_mse (M6, §4.2)
TARGET_DISK_GB=<unset>  EXPORT_CONTAINER=compressed-tensors
TARGET_PROFILE=<unset, spec-resolved>  TARGET_PROFILE_DEFAULT=vllm_packed_moe
SELECTION_MODE=surrogate, or validated-surrogate under a TARGET_DISK_GB card
EXPORT_PRODUCTION_CACHE_PREFETCH=require (native lane, D8)
MTP_FORMAT=BF16  PRODUCTION_CACHE=1  PRODUCTION_RECACHE=1
PRODUCTION_CACHE_LEVERS=gptq,static_act_order,joint_scale_opt
PRODUCTION_CACHE_RENDER_SCOPE=assignment  …_CACHE_PREFETCH=require
VALIDATED_SOURCE_PREFETCH=require   VALIDATED_FRONTIER_PICK=kneedle,
                                    or `budget` under a TARGET_DISK_GB card
VALIDATED_FRONTIER_SKIP_CALIB=$NSAMPLES (held-out disjointness, ON)
CB_EXPERT_EMPIRICAL=0  CB_SCALE_CODING=two_tier  (D15: shipped values)
AURA_ADDITIVITY_GATE=measure
PRISMAQUANT_GGUF_IMATRIX=1  DEVICE=cuda  EXPORT_DEVICE=cuda
```

`EXPORT_CONTAINER` ∈ {`compressed-tensors`, `gguf`, `nvfp4_cb`} selects the lane, and the
preflight now **refuses a lane the architecture has not declared** (`supported_lanes`,
re-vet R6) — an undeclared lane does not fail at serve time, it serves uninitialised expert
memory and generates coherent-looking garbage.

**`COST_MODE=aura` is the default since 2026-07-30 (re-vet R2).** Both flagship artifacts
(regen-27B, 35B arm-E) were produced with it and its served margin over the previous default
is −38%/−39.5% confident-KL at the 4B knee across two calibrations and −17.9% at 27B (§4.3).
`production-render-score` remains fully supported and is the explicit/legacy spelling —
historical artifacts reproduce by setting it. The flip was gated on the two preconditions R2
named, both landed: the `provenance["cost_mode"]` stamp (§3.4), so a `WORK_DIR` built under
the old default **rebuilds its cost table loudly** instead of silently allocating on the other
estimator; and the wired additivity report (§4.3, stage `[3c]`), so every AURA artifact carries
its own trust-region number. The three CB/GGUF lanes resolve their own render and objective
through the render-faithfulness assertion (§4.7) rather than through this default.

**`TARGET_PROFILE` is deliberately unset** (re-vet R11 / D4). `resolve_target_profile` gives an
explicit request precedence, so a shell default silently beat every architecture's
`spec.default_serving_profile` — measured cost 2026-07-11: 226 dense FP8 Linears coerced to
BF16 on the Hy3 export. Unset, the spec wins; `TARGET_PROFILE_DEFAULT=vllm_packed_moe` is the
fallback for architectures that declare nothing (never `research`, whose menu is unbounded);
an explicit `TARGET_PROFILE` still wins, so every in-tree launch script is bit-identical. The
resolved profile is stamped into `layer_config.json`'s reserved `__prismaquant__` block and
read back by the exporter, so allocator and export cannot disagree.

**`TARGET_DISK_GB` makes the card the constraint** (re-vet R1 / D12). When set it overrides
`TARGET_BITS` (the allocator re-emits at the bpp whose exact footprint fits), narrows the
Pareto sweep to the ~3 rungs that can ship, flips `SELECTION_MODE` to `validated-surrogate`
and `VALIDATED_FRONTIER_PICK` to `budget` — min measured KL among the allocations that fit.
An explicit `SELECTION_MODE`/`VALIDATED_FRONTIER_PICK` still wins. §4.6 owns the selection
semantics; §4 owns the cost-mode semantics; §5 owns the lever semantics.

### 3.4 Reuse guards and the silent-reuse class

**The key set is `pipeline.py`'s job; the values are the shell's.** `STAGE_SETTINGS_KEYS`
(`pipeline.py`) declares, per artifact, which settings that artifact's identity depends on.
`run-pipeline.sh` passes every value once (`STAGE_SETTINGS_ENV`, `596`+), `pipeline.py
--write-stage-settings` projects them onto each declared key set and emits
`artifacts/stage_settings.json`, and `require_stage_settings <artifact> <stage> [LATE=v …]`
(`684`) reads that projection instead of re-deciding the key set at the call site. Late-computed
values (`AURA_CACHE_FORMATS`, `CACHE_FORMATS`, `ASSIGNMENT_DIGEST`) are passed as overrides;
a declared key nobody supplies is a hard stop, not a silent gap. This is re-vet **R5**, and it
closes **D6** by mechanism rather than by enumeration — the twelfth stage cannot arrive without
a guard, because adding one is a table entry, not a bespoke argument list.

Contract:

| state | outcome |
|---|---|
| artifact absent | record this stage's projection, build |
| recorded projection matches | reuse |
| recorded projection differs | **`exit 2`**, naming every differing key and the stale file |
| no record for this stage (pre-guard artifact) | **WARN**, record, guard from then on |

The manifest is `<artifact>.settings.json`, keyed by stage, so two stages can legitimately own
one path — under `COST_MODE=aura` + `validated-surrogate` the AURA dW cache and the frontier
cache **are the same file** (principle 8's one-render identity), and both key sets coexist.
Pre-R5 flat manifests are read as a `legacy` block and still guard the stage whose key set they
match, so no live `WORK_DIR` is invalidated by the upgrade.

**Coverage is now every skip-if-exists artifact** — **15 call sites over 15 declared
artifacts**: `probe`, `base-cost`, `render-cost-cache`, `render-cost`, `aura-dw-cache`,
`aura-cost`, `aura-hybrid-cost`, `cb-col-weights`, `cb-hybrid-cost`, `frontier-cache`,
`frontier-kl-point`, `frontier-recache`, `production-cache-recached`, `production-cache-raw`,
`gguf-skeleton`. (Wave 3 reported 16 sites because `cb-col-weights` was guarded at three
near-copies of the harvest; wave 4's `harvest_cb_col_weights` collapsed those into one function
with four callers, so the guard is now stated once and the artifact-to-site map is 1:1.) Render-affecting env is captured in `RENDER_ENV_SETTINGS` (`585`:
`PRISMAQUANT_NVFP4_SCALE_RULE`, `PRISMAQUANT_GPTQ_DAMP_SWEEP` default `0`,
`PRISMAQUANT_GPTQ_DAMP`, `PRISMAQUANT_ACT_CLIP_QUANTILE` default `0.999`,
`PRODUCTION_CACHE_LEVERS`, `PRODUCTION_CACHE_DISABLE_LEVERS`) and spliced into every artifact
that stores rendered weights.

**Over-keying is the risk the table is written against.** Declaring a key an artifact does not
depend on forces a spurious rebuild, and some of these are 90 GB. The rule applied: key an
artifact on the inputs that change its *bytes*, and key expensive artifacts conservatively (the
probe is keyed on model/corpus/windows/modality and **not** on `FORMATS` — it is format-blind;
`cb_col_weights.pkl` is keyed generously because it rebuilds in minutes). Historical manifest
key names (`NS`, `SL`, `SEED`) are preserved where they existed, so artifacts built before R5
compare equal instead of rebuilding.

**Cost tables get a second, orthogonal gate.** `cost.pkl` is the same path under *every*
`COST_MODE`, so a settings match is not sufficient — the file could be the previous mode's
estimator. Every producer (`incremental_measure_quant_cost`, `production_render_cost`,
`aura_cost`, `expert_empirical_cost`, and the inline sidecar-backfill finalize) now stamps
`provenance["cost_mode"]` from `--cost-mode`, and `cost_table_reusable()` (`669`) makes reuse of
the *allocator's* table conditional on it matching. A mismatch **rebuilds** with a loud line
naming both modes; an unstamped (pre-R2) table warns and is reused, never invalidated. Under
`COST_MODE=local` the baseline *is* the allocator table so it carries the same gate; under the
other modes `cost_baseline.pkl` is mode-agnostic on purpose and is shared across mode changes.
This is re-vet **R2 precondition (i)** — the prerequisite to flipping the `COST_MODE` default,
which is *not* done here.

### 3.5 Archived modes — the eleven `exit 2` gates

| Trigger | Lines | Archive |
|---|---|---|
| `COST_MODE=grouped-kl` | `308-312` | `archive/grouped_kl_2026-05-28` |
| `COST_MODE=production-render-staged` \| `-tail` | `318-322` | `archive/production_render_staged_2026-07-30` |
| `FISHER_WEIGHTED_GPTQ` truthy | `205-213` | `archive/fisher_2026-05-15` |
| `FISHER_OUTPUT_MSE_ALLOCATOR` truthy | `205-213` | `archive/fisher_2026-05-15` |
| `PRODUCTION_CACHE_LEVERS` ∋ `fisher_gptq` | `215-220` | `archive/fisher_2026-05-15` |
| `HADAMARD_DUQUANT` truthy | `354-360` | `archive/hdq_2026-05-14` |
| `PRODUCTION_CACHE_LEVERS` ∋ `hadamard_duquant` | `361-366` | same |
| `MULTI_SHOT_PASSES` ∉ {unset, 1} | `367-373` | `archive/multi_shot_2026-05-19` |
| `ALLOC_PROPAGATED_SENSITIVITY_REPORT` non-empty | `374-380` | `archive/l3_propagated_2026-07-30` |
| `PRODUCTION_CACHE_UNION` truthy | `381-387` | `archive/union_cache_2026-07-30` |
| `MSE_PROMOTION` truthy | `388-394` | `archive/mse_promotion_2026-07-30` |

Four of these landed with the 2026-07-30 re-vet (R17, R4, R18 ×2). Each error string carries
**the measurement that killed the lever**, so the refusal teaches rather than merely blocks.
The archive directory names are load-bearing for the orchestrator: moving or renaming one
breaks its gate. Lessons: §11.

A twelfth refusal lives outside `run-pipeline.sh`: `export_native_compressed.main()` calls
`_refuse_archived_block_output_match()`, which `SystemExit`s if
`PRISMAQUANT_BLOCK_OUTPUT_MATCH` is set truthy (`archive/block_output_match_2026-07-30`,
re-vet R25). It is a `SystemExit` rather than a shell gate because the lever was an exporter
env var no pipeline stage ever set.

### 3.6 `pipeline.py` — what the contract layer actually is

**It has exactly one load-bearing job, and §3.4 is it.** `pipeline.py` owns
`STAGE_SETTINGS_KEYS` — the per-artifact declaration of which settings each build artifact's
identity is keyed on — and the `--check-stage-settings` guard the orchestrator calls at every
skip-if-exists site. That is the one thing the shell provably got wrong (ten artifacts with no
guard at all, six more each holding their own opinion of their key set), and it is the one
thing `pipeline.py` was already positioned to fix: it receives the settings and it is the only
place where "what does this artifact depend on?" can be reviewed as a table rather than
rediscovered per call site. Re-vet **R5**, adjudicated in favour of Lens 2's narrow promotion.

Everything else in the file remains **descriptive**: it also writes and `--validate`s a spec
JSON declaring 14 artifacts, 3 gates and 9 base stages plus render-mechanism stages generated
from `render_score.resolve_render_mechanism_order`. Nothing downstream reads that JSON back,
and its `validate()` is tautological in the production path — the spec it validates is the one
`default_production_pipeline_spec()` just generated from its own hardcoded `ResourceContract`s,
and `run-pipeline.sh` never passes `--input`. Treat the *spec* half as documentation with a
linter. Coverage stays partial in both directions by choice (re-vet: modelling the ten
executed-but-unmodelled stages would be fiction-surface without teeth): `validate.vllm_smoke` is
always stripped and `validate.kl` is stripped whenever `SELECTION_MODE=surrogate`.

`APPROVED_RESOURCE_OWNERS` is now honest (D10): `rendered_weights → ProductionWeightCache`,
`perturbed_activations → PerturbedActivationCache`, `streaming_model_weights → LayerCache`
(`layer_streaming.py`). The two placeholder names that existed nowhere in the tree
(`StreamingActivationCache`, `StreamingModelPrefetch`) are deleted, and a test asserts every
approved owner has a class behind it. `kl_measurement.QuantWeightCache`, the other candidate
owner, went to the archive wall with L3 (§4.4) and is no longer a live holder.

The one-cache rule (§5.4) is still enforced by the runtime strict-cache gates, not by this file.

### 3.7 `WORK_DIR` layout

Created at `408`:

```
artifacts/  probe.pkl, cost*.pkl, layer_config*.json, pareto.csv,
            pareto_assignments/, production_*_cache.pkl + shard dirs,
            validated_frontier_kl*.json, cb_col_weights.pkl, skeleton.gguf,
            pipeline_spec.json, stage_settings.json, *.settings.json
act/        probe activation cache        work/  streaming layer shards
logs/       probe|cost|allocator|export   exported/  compressed-tensors ckpt
```

Plus `exported.gguf` (GGUF lane) and `exported_nvfp4_cb/` (CB lane) directly under `$WORK_DIR`.
Sizing discipline — a 27B cache is ~90 GB — is §10.

## 4. Cost models & allocation

The allocator needs one number per `(Linear, format)`: `predicted_dloss`, the estimated
end-loss damage of that rendering. Below, the machinery that produces it and spends a bit
budget against it. Paths are repo-root-relative; the orchestrator is
`prismaquant/run-pipeline.sh`.

### 4.1 Stages that always run

| Stage | Module | Produces |
|---|---|---|
| L1 Fisher probe | `incremental_probe.py` (`run-pipeline.sh:539-560`) | `artifacts/probe.pkl` (per-Linear `h_trace`, `n_params`, shapes) **and** `WORK_DIR/act`, the activation cache every later stage reads |
| Base RTN cost | `incremental_measure_quant_cost.py` (`:606-661`) | per-`(Linear, format)` measured RTN error; under `aura` demoted to sidecar-backfill source (`:928-952`) |

The probe is streamed shard-by-shard through `layer_streaming` — head resident, body paged,
MTP a built-in shard kind (`incremental_probe.py:2-17`); a modality guard aborts on
probe/`CALIBRATION_MODALITY` mismatch (`:562-599`).

`h_trace` is the empirical CE Fisher diagonal trace. Additive model:
`0.5 · h_trace · weight_mse · gain` (`allocator_solver.py:60-63`, derivation
`allocator.py:13-52`).

**One denominator: the global calibration token count** (PR #14, `f53945f`). Every row —
dense trunk and per-expert alike — is `h_trace_raw / (nsamples × seqlen)`
(`finalize_fisher_stats`, `sensitivity_probe.py:496-534`; the incremental backend calls the
same function, `incremental_probe.py:2644`, stamping `meta["fisher_norm_tokens"]` at `:2754`).
Both backends share it, and `h_detail` blobs use the identical count (`h_detail_version: 4`,
`sensitivity_probe.py:488`).

This **reverses** the earlier per-routed-token convention that this document and `CLAUDE.md`
previously described. Dividing an expert row by its own routed count inflates it by
(global / routed) — the same `1/p_e` overweighting audit M4 set out to remove, merely implicit,
and exactly inverted importance weighting (the least-used experts look the most sensitive).
Typical inflation is ~`n_experts/top_k` (≈32× on a 256-expert top-8 model); the degenerate
1-routed-token case reaches ~33,000× at a 32k-token calibration. `n_tokens_seen` and
`route_prob` both survive as metadata only.

**Legacy probes hard-fail.** `renormalize_probe_fisher` (`allocator.py:1066-1163`, called
`:1455`) recomputes every row from the stored raw accumulators. Per-row
`h_trace_norm_tokens` stamps win over the meta count — a merged multimodal visual pass was
finalized at its own token count, so honouring the stamp keeps the recompute idempotent. A row
that carries raw accumulators but has neither a stamp nor a usable meta count is a `SystemExit`
naming the remedy (re-probe); `--allow-legacy-fisher-norm` (`:1190-1195`) downgrades it to a
warning for reproducing historical allocations (`612fc38`). Re-solves of the shipped
Qwen3.6-27B and Qwen3.5-35B-A3B probe/cost pairs at `TARGET_BITS` were unchanged by the fix.

### 4.2 `production-render-score` — the explicit/legacy cost mode

The default until 2026-07-30, now the explicit spelling that reproduces every pre-flip
artifact (§3.3; the flip is re-vet R2). It builds a format-menu render-score cache and derives cost from the scores
the render itself recorded (`run-pipeline.sh:665-715`; staged/tail variant `:717-823`).
Contract at `production_render_cost.py:1-16`: the rendered score is the damage of the weights
export will actually ship, so rows set `output_mse_measured=False` and the allocator consumes
`predicted_dloss` directly instead of re-applying the Fisher proxy.

**M6 — the score field is `weight_mse`, not `h_trace × output_mse`.** The legacy product
carries activation energy `E‖x‖²` twice, since `h_trace` is already a weight-space Fisher
trace. Served A/B at matched 4.75 bpp: Qwen3-4B KL −50.8% / PPL −15.1%; Qwen3-0.6B KL −58.5% /
PPL −24.4% (`run-pipeline.sh:191-200`). 27B-class confirmation is ladder debt.

The stratified per-expert subsample (`PRISMAQUANT_EXPERT_COST_SAMPLE`) is applied on this path
too since `79964de` — `_measure_production_render_dense` had the `resolve_cost_target_name` fix
but not the subsample, so under the *default* cost mode the lever silently did nothing on
exactly the models that need it (DSv4: 256 experts × 3 projections × 43 layers). Split at
`incremental_measure_quant_cost.py:291`, extrapolated at `:421`, filled before the `render_path`
stamp so extrapolated rows still carry production provenance. Export still quantizes every
expert.

### 4.3 AURA — the default cost mode (`COST_MODE=aura`)

`aura_cost.py`. Cost is the KL-adjoint inner product with the production-rendered weight error
(`:5-14`, impl `:695-725`):

```
predicted_dloss[i,f] = 0.5 · mean_k ( <gW_i^(k), dW_{i,f}> )²
gW_i^(k) = ∂/∂W_i fisher_probe_scalar(logits; seed=k)   # KL/GN Fisher, rademacher
dW_{i,f} = Q_f(W_i) − W_i                               # production-rendered
```

The probe is `kl_fisher.fisher_probe_scalar` (`kl_fisher.py:77-131`). `dW` provenance is
recorded per row as `rendered` vs `rtn` (`aura_cost.py:195-234`) — immaterial at fp4, decisive
at fp8 (+36% served KL under RTN dW); `--require-production-cache` makes a missing rendered row
fatal and the pipeline always passes it (`run-pipeline.sh:886`). Passthroughs are zero-cost by
construction (`aura_cost.py:83`). Packed-MoE experts are hard-excluded (`:315-337`); the
pipeline passes `--allow-packed-expert-omission` (`run-pipeline.sh:899`) and covers them in
`[2d]`. Three sub-stages: `[2b]` format-menu cache for dW (`:825-874` — under
`validated-surrogate` this *is* the frontier cache, per the one-cache principle), `[2c]`
`aura_cost` (`:879-903`), `[2d]` hybrid finalize (`:905-956`).

**Empirical expert costs** (`expert_empirical_cost.py`) exist because AURA's smooth cost is
route-flip-blind on routed experts (Spearman 0.45→0.35 under faithful dW; predicted NVFP4/FP8
ratios 2–49× vs measured 1.1–1.5×, `:1-28`). The unit is all packed expert tensors of one MoE
module (vLLM FusedMoE must share one format); unit cost is end-to-end mean-token
`KL(BF16 ‖ unit-quantized)` split across members ∝ `n_params` (`:481+`, `_unit_kl :318+`). FP8
stays in the expert menu by standing decision — no hardcoded ban; the DP plus real KL rejects
it (`:19-23`). CB families render the whole stack in one qdq call (`:53-55`), with opt-in
holdout-gated RD-law ladder interpolation `D(k)=C·2^(−k/4)` (`:57-66`).

**UCB — two of them, both default-off, neither set by the pipeline.** Cost-side
`PRISMAQUANT_COST_UCB_Z` adds `z·stderr` before the DP (`allocator_candidates.py:357-370`);
`z=0` is bit-identical to no-UCB and it only bites on the `predicted_dloss` branch (AURA /
expert-empirical), not `output_mse`/`weight_mse`. Selection-side `--kl-ucb-z` yields
`kl_ucb = mean + z·stderr` over calib repeats (`validate_assignments_kl.py:640-660`), consumed
by `select_validated_frontier --metric ucb`. Both are **research-only** as of R28
(`docs/design/runtime_flags.md` §1): the one measured win (`z=2`, −8.0% on the 27B old-vs-new
AURA A/B) is a thin-calibration result, and at production calibration the hedge moves 6/252
rows to served parity — hence default-off, no driver, and the standing decision to keep `z=0`.

**The additivity report — stage `[3c]`, wired 2026-07-30** (re-vet R2 precondition (ii); the
R2-vs-R19 disagreement was resolved as *wire it*, not *wall it*). AURA's one structural
assumption is that per-Linear KL contributions add. `aura_additivity_gate.py` records
`residual = measured_end_KL(assignment) − Σᵢ predicted_dlossᵢ` with an honest stderr — exact
per-probe when the rows carry `x2_per_probe` (probe-aligned raw samples, so the correlated sum
is computed rather than approximated), else the independence lower bound, and it says which.
It is a **report**: never blocking, never touching an allocation. It runs after the assignment
is final (that is why it is `[3c]` and not `[2d]` — no assignment exists at `[2d]`), and its
result is stamped into `cost.pkl`'s `provenance["additivity"]`, so every artifact derived from
that table carries the trust-region number. **`AURA_ADDITIVITY_GATE=measure` is the default
since 2026-07-30** (Robert's ruling on the R2 residue): it reports from a measured KL the run
already produced when there is one (validated-surrogate's frontier JSON, free) and otherwise
runs **one bounded KL eval** of the final assignment against the same format-menu dW cache AURA
costed on — so every AURA-default run performs the measurement and **every artifact carries a
real residual**. The wiring's one weak spot was that under `SELECTION_MODE=surrogate` an
artifact carried a prediction and no residual, leaving AURA's structural assumption a two-model
memory instead of a per-artifact number; the ruling closes it, at the price of one bounded GPU
eval per run. `auto` (the pre-ruling behaviour) stays selectable and is zero-added-GPU: it
reports only from a measurement the run already made, otherwise recording the predicted sum with
`measured_kl: null` and a status naming the reason. `0` disables. Either way it is a report —
non-blocking, never touching an allocation.

### 4.4 L2 and L3 — retired 2026-07-30 (`archive/l3_propagated_2026-07-30/`)

Both levels of the old cascade are **walled**, not merely off. Re-vet **R4**; the wall's
README carries the full lesson.

**L2 perturbed-X was never a cost stage.** No `COST_MODE` ever ran a re-measure/re-solve
loop; the accepted set is `local | production-render-score | aura` (`run-pipeline.sh`
`COST_MODE` case). `perturbed_x_cache.py` **stays live** — it is the activation-cache /
model-loading utility that `validate_assignments_kl`, `kl_measurement` and
`production_recache` depend on, and it always was.

**L3 propagated end-KL is walled.** `kl_sensitivity_probe.py` (3,678 L),
`propagated_sensitivity_costs.py`, `sensitivity_response.py`, the five
`--propagated-sensitivity-*` allocator arguments and the L3 half of `kl_measurement.py`
(97 top-level symbols, ~4.3k lines, now
`archive/l3_propagated_2026-07-30/prismaquant/kl_measurement_l3.py` — still importable
against the live tree) all moved. Setting `ALLOC_PROPAGATED_SENSITIVITY_REPORT` is `exit 2`
(§3.5).

**Why, in three measurements** — none of them an argument:

| Evidence | Result |
|---|---|
| `aura_cascade_headtohead` | L2 fixed point beats additive L1 by **−1.5%**; AURA beats L1 by **−38.5%** |
| `xlayer_sensitivity_2026_06_09` + `cross_layer_additivity_fp32` | pairwise residual **+5–12% and diffuse**, **3/1180** pairs significant; the apparent non-additivity is a **bf16 differencing artifact** — per-Linear KLs add in fp32 |
| §11 / `prismaclade_l3_non_additivity` | L3 costs measured under an L2 context do **not** sum when many flip at once, so L3's expensive measurement could not be composed anyway |

L3's only consumer (DP/coordinate-descent polish) was already archived
(`archive/polish_2026-05-15/`), and `kl_sensitivity_probe` had zero references in
`run-pipeline.sh` at wall time.

**What survives in `kl_measurement.py`** (1,206 lines, down from 5,731): whole-assignment
`measure_assignment_kl`, the per-sequence tail machinery (`sequence_token_nll`,
`summarize_per_sequence_kl`, `return_per_sequence`), `assignment_bit_total` /
`assignment_hash`, `l2_cost_value`, and the `CUDAGraphRegistry`. `validate_assignments_kl`
and `validation_harness` are unchanged.

### 4.5 Solver

`allocator_solver.py`. Multi-choice knapsack DP over average-bits-per-parameter bins,
numpy-vectorized (`solve_allocation :427-520`); the baseline per Linear is its cheapest
candidate, bins = `(target − min_bits)/bit_precision + 2`, and backtrack mirrors the forward
charge exactly. `_charged_bins` (`:409-424`) charges any strictly positive Δbits at least one
bin, so sub-half-bin upgrades are never free.

**The DP unit is the serving-atomic unit, not the Linear** (#17, `f719d93`). A packed-MoE
serving group is atomic at serve time — vLLM's FusedMoE loads every projection of every routed
expert in a layer under ONE scheme — so "upgrade one expert row" is not a real option, and
pricing it per-row while `promote_serving_units` charges the whole group is a ~1000× price
mismatch: mispriced expert rows top the per-bin ranking, feasibility tightening over-corrects,
and cheap dense rows starve. `aggregate_packed_serving_groups`
(`allocator_candidates.py:993-1174`) pre-aggregates each group into one multi-choice item whose
per-format cost and byte cost are the exact sums of its members, over the **intersection** of
member-legal formats; `expand_packed_group_assignment` (`:1176-1189`) broadcasts the decision
back for emission. Post-DP MoE promotion becomes a validated no-op. A group with no common
legal format falls back to individual rows and is then **not allocatable** — `compute_achieved`
raises rather than score the unpriced member at zero Δloss (which would make the illegal state
look cheapest to the min-Δloss ratchet). `--no-packed-aggregation` (`allocator.py:1281-1288`)
restores per-row pricing for back-compat experiments only.

**Serving-unit promotion is union-find, and legality-aware** (#28, `9b4347f`).
`promote_serving_units` (`:302-327`) unions fused-sibling and packed-MoE groups in one
order-independent pass; `_promote_group_components` (`:234-299`) chooses the component's
format via `_choose_group_format` (`:192-231`) from per-row legal sets derived from the
candidate lists (`legal_formats_from_candidates :103-116`) — the **cheapest legal-for-all**
format at or above the max rank, falling back to the highest legal-for-all only when nothing
above is common, and raising with every member's legal set when the intersection is empty
(`_serving_group_menu_error :152-189`). Before this, promotion took only `assignment`,
`format_rank` and `groups`, so it wrote the max-rank format blind to whether the rest of the
unit could carry it — members do not share a shape, and often it could not. The legality
argument is optional by design: omit it and the legacy two lines run verbatim, so hand-built
and auxiliary MTP/visual assignments cannot acquire a new failure. `promote_fused` (`:362-406`)
still hard-asserts post-promotion coherence. Non-regression: re-solving the shipped 27B and 35B
at `TARGET_BITS` changed 0 of 614 and 0 of 500 assignments.

**Termination is feasible-only, and solves are memoized** (#16, `8d3d0dc`).
`solve_with_promotion` (`:606-851`) contracts that the returned assignment is always feasible
(`achieved ≤ target + overshoot_tolerance`) and, among feasible iterates, the one with
**minimum total predicted Δloss** (ties → larger achieved bits). Δloss is the objective; density
is not a proxy for it — 5.5 bpp has beaten 6.0 bpp on served PPL. Three silent fallbacks are
gone: a `solve_allocation` returning None no longer yields the previous over-target iterate,
an arbitrarily deep undershoot is no longer accepted, and the stall exit no longer returns an
iterate above target. When no iterate is feasible within `max_iters=40` the rung is INFEASIBLE
and `(None, nan)` is returned so callers drop it from the Pareto curve. Search is damped
descent to the first feasible iterate, then bracket bisection with a min-Δloss ratchet
(promotion is a coarse step function, so `achieved(tightened)` is locally non-monotone). A
`diagnostics` dict is filled in place on every return path — `min_bits`, `evals`,
`closest_achieved_bits`, `floor_achieved_bits` — which is the only thing that makes an
INFEASIBLE verdict actionable. `PRISMAQUANT_SOLVER_TRACE` (`:37`) prints per-eval timing.
`allocator.py` memoizes the solve per target (`:1959-1982`): it is a pure function of the
target given fixed stats/candidates, and the byte-budget grid plus ratchet bisection re-visit
targets the Pareto sweep already solved. Callers get a **copy** of the assignment dict —
fused-sibling expansion mutates it — and per-target diagnostics are kept beside the memo so a
cache hit never loses them.

**Bit-exact re-encodes price at zero, but only on an identity activation path** (#20,
`5028fff`). `cost_entry_is_bit_exact` (`allocator_candidates.py:233-286`) short-circuits a
measured `weight_mse == 0.0` to `predicted_dloss = 0.0` — genuinely optimal when the format
stores the source weights verbatim (MXFP8 over an FP8 128-block source; MXFP4/6/8 over an
MXFP4-packed QAT source). But `W' == W` silences only the *weight* side: for W·A· formats the
cost pipeline quantizes activations before measuring, so a weight-lossless MXFP4 re-encode of
an MXFP4 source would price at dloss 0.0 — the unbeatable global minimum at any budget — while
serving 4-bit activations. The gate is therefore the dtype-level predicate
`FormatSpec.act_quant_changes_input` (`format_registry.py:75-106`: `act_bits` absent or ≥ 16),
not a heuristic; unregistered formats never short-circuit, and an entry declaring an explicit
`cost_source` (the production-render pipeline, whose `weight_mse` is a placeholder) is never
treated as bit-exact.

**Opt-in gate_up/down role split** (#21, `237a029`). `--packed-role-split`
(`allocator.py:1289-1300`) keys each packed expert group as two per-layer serving units
(gate+up, down) by wrapping the profile view (`packed_role_split_profile`,
`allocator_candidates.py:1243+`), so DP aggregation and serving promotion stay consistent. It
**hard-errors** unless the resolved serving profile declares
`supports_per_role_expert_schemes` (`serving_profiles.py:399-405`, gate
`require_per_role_expert_scheme_support :636-674`). GGUF declares it — expert tensors are
stacked per projection, each carrying its own ggml type. vLLM's compressed-tensors packed-MoE
path does not: `CompressedTensorsMoEMethod` selects one scheme per FusedMoE layer, so a
role-split checkpoint is unloadable. Default off.

Candidate legality, passthrough integrity, cost-source precedence and fused-sibling aggregation
also live in `allocator_candidates.py`; the invariants they enforce are §6.4's.

### 4.6 Selection

`SELECTION_MODE` defaults to `surrogate` (§3.3): `layer_config.json` straight from the DP at
`--target-bits`, no real KL. The knee is the post-cliff log-error kneedle
(`allocator.py:212-220`); raw-linear and global-log knees are diagnostics.
`_rd_curve_diagnostic` (`:285-338`) fits `log10(Δloss)` vs bpp and at `R² ≥ 0.99` prints that
there is no intrinsic knee and ship bpp should come from a byte budget or measured saturation.

`SELECTION_MODE=validated-surrogate` (`run-pipeline.sh:1056-1288`, requires
`PRODUCTION_CACHE=1`) is the real-KL path: Pareto assignments → one format-menu frontier cache
→ `validate_assignments_kl` per point (`--calib-skip-first $NSAMPLES` is the held-out
mechanism; `--kl-scope full_sequence` since M26) → `select_validated_frontier` → optional
`MSE_PROMOTION` rewrite → `production_recache` re-fits activation scales for the selected
assignment.

`select_validated_frontier.py` builds an **η-dominance** envelope: rows sorted by (bpp, kl), a
point enters only if it beats the running best by more than `--kl-noise-floor`
(`_frontier_from_rows`). Picks: `kneedle` (default without a card), `budget` (**the default
under `TARGET_DISK_GB`**), `best-kl`, `lowest-bpp`, `practical-knee`, `saturation`.
Diagnostics emitted with the pick: surrogate-vs-KL Spearman, `worst_rank_inversion`,
leave-one-out kneedle stability.

**Tail veto** (D1, 2026-07-30) — **DEFAULT-ON, contract statistic `kl_max`** (ruled by Robert
2026-07-30). §2.3 rule: KL is a *screening* metric and a lower mean can hide a heavier tail —
the shipped 27B PrismaSCOUT has a worse max-prompt NLL than the artifact it beat on mean KL.
`--tail-veto {none,kl_p99,kl_max,nll_p99}` adds a second admission condition to the same single
pass: a row that improves mean KL enters only when
`row[tail] <= incumbent[tail] * (1 + --tail-eta)`, the incumbent being the last *admitted*
frontier point. Columns come from `validate_assignments_kl`'s per-sequence emission (§7.1) and
carry the gold lane's key names, so a selection row and a served row read the same.

- **The contract statistic is `kl_max`** — the worst sequence. It is the statistic that would
  have caught the broken 27B that passed on the *mean* while 80% of its prompts were bad, which
  is the same reason §7.2's ship gate guards p99 per-prompt NLL. `nll_p99` continues to be
  recorded on every row (both are free), so switching the contract later is a flag, not a
  re-measurement.
- **Default-on is safe because the failure is one-sided.** A spurious veto only makes the pick
  **more conservative** — it refuses a lower-bpp/lower-mean point and keeps a higher-bpp one
  with a smaller tail — and it is never silent: every refusal is printed and retained in the
  summary under `vetoed_rows` with its `veto_reason` (`tail_regression`, or `tail_missing` when
  a row predates the emission). The `--tail-veto` help text says exactly this.
- **`--tail-eta` defaults to `auto`: the slack is derived, not chosen** (house rule 2). `auto` =
  the incumbent row's **relative stderr of the tail statistic across calibration repeats**
  (`std/√n ÷ mean` over `<column>_repeats`, floored at 0) — i.e. how much that tail moves when
  nothing about the assignment changes, so a candidate inside its own measurement noise is
  admitted and one outside it is a real regression. `validate_assignments_kl` emits the
  per-repeat tails from the same forwards the mean already paid for. With a **single** repeat
  there is no spread: `auto` degrades to a strict `0` **and prints a warning**, because
  single-seed tails are noisy (§2.5: a +10% reading has flipped to −5.2% across repeats) — run
  validation with `--calib-repeats ≥ 4` to get a real slack. An explicit numeric `--tail-eta`
  always wins. Derivation documented at `select_validated_frontier.tail_eta_auto`.
- **A pre-R9 validation JSON carries no tail column at all.** Vetoing every row would turn a
  stale input into an empty frontier, so the veto goes **inert** with a loud warning and the
  run reproduces the mean-only envelope (`tail_veto_inert_reason`, recorded in the summary).

`--tail-veto none` restores the pre-R9 envelope byte-for-byte (pinned by a regression test).
The veto also applies inside the leave-one-out rebuild, so the stability diagnostic reflects
the same envelope the pick came from. There is **no second eval pass** — the tail was already
being computed and discarded.

**Byte budget = constraint, measured KL = objective** (re-vet **R1**, closes D12). Two disjoint
ship selectors used to exist: the allocator's `--target-disk-gb` picked by *predicted* Δloss
among the allocations that fit, and `select_validated_frontier` picked by *measured* KL but was
byte-blind (`grep -c bytes` → 0). The selector that owned the ship decision ran on the
surrogate; the one that measured could not see the card — §2.2 inverted exactly where it is
load-bearing, and the surrogate-knee failure is on record (27B: surrogate 5.857/0.056 vs
validated 5.31/0.015). They are now one stage:

* `TARGET_DISK_GB` is plumbed through `run-pipeline.sh` into the allocator. When set it
  **overrides `TARGET_BITS`** (the allocator re-emits at the chosen bpp) — the CLI semantics,
  now the pipeline's.
* The allocator prices **every Pareto candidate** with `footprint.assignment_artifact_bytes`
  — the same accounting its own byte-budget selector uses, so the two can never disagree — and
  stamps `artifact_bytes` into each Pareto assignment payload and the manifest.
* Under a card it then **narrows the Pareto set to the bracket around the byte-feasible bpp**
  (the largest fitting rung ±1, ~3 of 11), with a log line naming the largest fitting rung. A
  computed narrowing, not a hardcoded rung count; skipped loudly if any candidate is unpriced.
  This is what makes byte-budget selection ~3 KL evals rather than 11, and it is why
  `validated-surrogate` defaults **on** under a card and stays opt-in without one.
* `select_validated_frontier --mode budget --target-disk-gb` picks **min measured KL among the
  rows whose exact footprint fits**. `measured_rows` gains the `artifact_bytes` column, read
  from the row or from the allocator payload at `row["path"]`. Bytes are monotone in bpp and
  the frontier is the KL lower envelope, so the min-KL fitting frontier row is the min-KL
  fitting row overall. Unpriced rows or an infeasible card are hard errors, never a silent
  fallback to another pick.

`footprint.py` reproduces real `metadata.total_size` to 0.00% on three 27B artifacts (`GB = 1e9`);
`saturation_select.select_under_byte_budget` grids the rungs and the ratchet bisects the
memoized DP for an exact fit. Kneedle stays available and stays what `_rd_curve_diagnostic`
already calls it: a diagnostic on a log-linear RD curve.

**The floor is a per-tensor manifest, and it cannot go negative** (#15, `bb974a0`; `0a9dc00`).
The identity is `artifact_bytes = floor + Σ_reencoded memory_bytes_for_shape`, with
`floor = source_total − Σ_reencoded source_bytes`. Charging that second term at a *regime-wide*
per-param rate breaks on a mixed source — a DSv4-Flash checkpoint (I8-nibble MXFP4 experts +
E8M0 scales, F8 attention, BF16 floor) charged at the FP8_SOURCE 1 B/param layout removed more
bytes than the checkpoint holds and drove the floor to −113 GB, letting an artifact more than
twice the budget "fit". `source_tensor_bytes_manifest` (`footprint.py:229-329`) now sums each
re-encoded Linear's **actual safetensors header byte span** (weight + scale siblings), resolving
per-expert-on-disk names to the packed live names the allocator uses via the profile's
`packed_expert_parent_for_projection` — the same bridge layer-streaming uses — and keeping
tensors the live-name map declines rather than dropping them (`0a9dc00`). Three failure modes
are **hard errors raised before any selection number is consumed**
(`resolve_reencoded_source_bytes :331-423`, `check_floor_non_negative :462-495`): a re-encoded
name the manifest cannot resolve (its source bytes stay in the floor while its quantized bytes
are still added — on a packed-MoE model that is the whole expert mass, after which every rung
reads "below the floor"), two names resolving to the same source span (bytes removed twice, so
an over-budget artifact reads as fitting), and a negative floor, which is reported with a
per-tensor-class byte breakdown so the offending class is named rather than rationalized. The
byte-budget selector calls the shared `assignment_artifact_bytes` rather than an inlined copy
(`allocator.py:2309-2345`) — the inlined copy was the one path the exactness tests never
covered.

### 4.7 The two cost axes, and what the lane gates actually assert (R3)

`COST_MODE` silently decided **two independent things**: which *render* produces the
per-`(Linear, format)` error, and which *objective* maps it to `predicted_dloss`. Since
2026-07-30 they are named — `COST_RENDER ∈ {inline, cached-menu}` × `COST_OBJECTIVE ∈
{weight-recon, render-score, aura-adjoint}` — and `COST_MODE` is the documented **spelling**
over them, with its three values unchanged in meaning:

| `COST_MODE` | `COST_RENDER` | `COST_OBJECTIVE` |
|---|---|---|
| `local` | `inline` | `weight-recon` |
| `production-render-score` | `cached-menu` | `render-score` |
| `aura` (default) | `cached-menu` | `aura-adjoint` |

Setting the axes directly is equivalent; setting both spellings at once is `exit 2`, and the
two unimplemented pairs stop with the reason (`inline × aura-adjoint`: the adjoint consumes
production-rendered dW from a format-menu cache by definition).

**Why the split mattered.** The CB and GGUF lanes hard-`exit 2`'d unless `COST_MODE=local`,
justified as "production-render-score scores UNWEIGHTED registry renders" — a property of the
**render**, not of the objective, which is why an objective change was being blocked by a
render argument. The right key already existed: `measure_quant_cost._cost_render_uses_imatrix`
decides per **format family** (CB always weighted, `gguf` tracking `PRISMAQUANT_GGUF_IMATRIX`),
which is why `local` was never one thing either — under it the CB lane already measured the
exporter-faithful weighted encode while the compressed-tensors lane measured unweighted RTN.

The gates are now one **render-faithfulness assertion**: the render that produces the
allocator's cost must be the render the exporter ships. `inline` satisfies it by construction;
`cached-menu` satisfies it iff the `ProductionWeightCache` render applies the same imatrix,
which **CB Milestone C** made possible — `render_production_weight` /
`build_production_cache` take `col_weights`, applied to the weighted families only, with every
other format's bytes bit-identical (`tests/test_col_weights_render_identity.py`). The pipeline
harvests the vector once (`harvest_cb_col_weights`, one definition, four call sites) and passes
`--col-weights` to the cost cache. `PRODUCTION_CACHE=0` stays required on those lanes for the
unchanged reason: their **exporters** requantize the bf16 skeleton and never read a production
cache, so building the *export* cache burns hours on bytes that never ship.

**What this deliberately does NOT do.** The CB lane can now run `render-score` or
`aura-adjoint`; neither is its default and neither is recommended. AURA's −38%/−17.9% margins
are native-lane results, and CB's error surface (VQ quantization plus the expert route-flip
floor) is a different animal. The pipeline prints that the combination is opt-in when it is
selected. The A/B that would justify a CB default has not been run — and is **deferred by
Robert (2026-07-30)** behind the NVFP4-vs-CB-FP8@4.5 criteria work; CB stays `weight-recon`
until it runs.

**The trust-region readout that rides with the default objective.** `AURA_ADDITIVITY_GATE`
defaults to **`measure`** (ruled 2026-07-30), so every `COST_MODE=aura` run performs one
bounded end-KL eval and stamps a real `residual` into `cost.pkl`'s `provenance["additivity"]`
— stage `[3c]`, §4.3. `auto` (report only from a measurement the run already made) and `0`
remain selectable.

## 5. Formats & render

### 5.1 The menu

`format_registry.py`. For non-CB formats, `FormatSpec` byte accounting is
shape-exact rather than a nominal scalar:
`scale_count_for_shape` (`:123-155`) handles `scale_block_shape`, per-channel `group_size==0`,
and 3-D packed-expert stacks; `memory_bytes_for_shape` / `effective_bits_for_shape`
(`:157-168`) are what the DP, `footprint.py`, and the Pareto table consume for
those formats. Aliases:
`FP8`/`FP8_DYNAMIC` → `FP8_E4M3`, `MXFP8` → `MXFP8_E4M3` (`:170-188`).
`act_quant_changes_input` (`:75-106`) is the **single** predicate for "does the serving kernel
consume quantized activations" (`act_bits` absent or ≥ 16 ⇒ no): the allocator's bit-exact
short-circuit (§4.5), the KL validator's activation-quant assignment, `layer_state_cache` and
`perturbed_x_cache` all key off it, so a format's activation semantics cannot drift between
pricing and emulation. Registry-vs-callable consistency is pinned by
`tests/test_bit_exact_cost_pricing.py`.

| Format | line | w-bits / group / scale | eff. bpp (2-D) | Status |
|---|---|---|---|---|
| `NVFP4` | `:667-677` | 4 / 16 / fp8 e4m3, A4 g16 | 4.5 | Production, in the default menu (§3.3) |
| `FP8_E4M3` | `:750-762` | 8 / per-channel / fp32, A8 per-token | 8 + 32/in_f | Production (`FP8_DYNAMIC`), in the default menu |
| `BF16` | `:798-808` | 16 / — | 16 | Production, **passthrough only**, in the default menu |
| `FP8_SOURCE` | `:825-837` | 8 / block (128,128) / fp32 | 8.00195 | Production, **passthrough only**, verbatim copy |
| `MXFP8_E4M3` | `:720-728` | 8 / 32 / e8m0 | 8.25 | Registered, profile-allowed, **de-menued** |
| `NVFP4A16`, `MXFP4`, `MXFP6_E3M2/E2M3`, `MXFP8A16`, `MXFP8_E5M2`, `FP8_E5M2`, `INT8_W8A16`, `INT4_W4A16_g128` | `:678-795` | — | — | Research / registry-only |
| GGUF k-quants + IQ | `:884-902` | `_make_gguf_spec :864` | 2.0625–8.5 | GGUF lane (§9.3) |
| `NVFP4_CB_K12..K24` / `FP8_CB_K28..K48` | `:913`, `:954` | product-VQ codebook, g256 | 1.78125–3.28125 serialized body / 3.5–6.0 index stream plus row scales | production gridbook CB menu (§9.2) |
| `NVFP4_CB_S13..S16` | `:932` | signed codebook, g256 | legacy/research layout | decoder/export compatible, production-menu denied (§9.2) |

MXFP8 is de-menued rather than denied — `vllm_packed_moe` still allows `MXFP8_E4M3` — because
its E8M0 pow2 scale wastes ~√2 of a binade and exact-scale FP8 Pareto-dominates it; offered
both, the allocator never picks it.

`FP8_SOURCE`'s `quantize_dequantize` is identity (`:835`): the bf16 view *is* the lossless
dequant of the source E4M3, so cost is exactly zero. Legal on dense Linears under
`vllm_packed_moe`, **illegal on packed experts** (absent from the expert allow-list — §6.4).
CB candidates deliberately use a different byte authority: the versioned
`CBSerializationContext` in `nvfp4_cb_footprint.py` prices the exact FP4 layout,
FP8 per-row FP32 scales, and deduplicated sidecar identity. Candidate
construction, assignment accounting, reports, and exporter assertions share
that context; exact post-export inventory remains the final artifact check.

NVFP4 *weight* RTN routes through the export codec (`_nvfp4_export_aligned_rtn` `:636-663`) so
emulation and shipped bytes share one rendering; *activations* do not — per-group dynamic RTN,
because the codec's per-tensor global scale would make emulation batch-dependent while serving
uses a static `input_global_scale` (`:674-676`). The `torch.compile` RTN hot path is
MSE-identical but not bit-identical to eager (~0.036% of elements flip at codebook midpoints,
`:445-458`).

### 5.2 Scale rules and JSO

NVFP4 scale rules live in the *exporter*, not the registry
(`export_native_compressed.py:111-132`): `static_6` (default), `four_over_six_mse`,
`joint_mse`. `joint_scale_opt` / `joint_scale_optimization` / `codebook_mse` are all aliases of
`joint_mse` — three names, one rule.

- **JSO = `joint_mse` evaluated inside the GPTQ loop**, per-group levels default `(6.0, 4.0)`
  (`_parse_joint_scale_levels :292-310`). The full 7-level grid collapses to {6,4} for 99.998%
  of groups at +0.009% aggregate weight-MSE, and the trim is monotone: a genuinely hurt Linear
  can only be *promoted* to FP8/BF16, never silently degraded. Override
  `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`.
- `static_6` is the `PRISMAQUANT_NVFP4_SCALE_RULE` env default, governing non-JSO RTN renders
  only; `four_over_six_mse` is a separate, non-JSO rule. Do not conflate the three. Single
  selection point for RTN / GPTQ / scale-sweep / packed / export: `_select_nvfp4_group_scales`
  (`:316-349`).

### 5.3 GPTQ damp

Sweep **OFF** since 2026-06-12 (`gptq_damp_sweep_enabled` `export_native_compressed.py:2543`,
env default `"0"` at `:2555`), fixed damp **1.0** (`_resolve_gptq_fixed_damp :2558-2577`). The
sweep's evaluator was in-sample; the V1 served A/B had fixed damp winning every gold-lane
readout across calibration draws at ~4.4× less render time.
`PRISMAQUANT_GPTQ_DAMP_SWEEP=1` reproduces historical artifacts, `PRISMAQUANT_GPTQ_DAMP`
overrides the constant, per-role overrides at `:2586-2640`. The second reader that used to
default the same variable to `"1"` (`kl_sensitivity_probe.py`) was a forked copy of the lever
defaulting; it now delegates to `production_weight_cache._resolve_production_render_levers`
(`kl_sensitivity_probe.py:272-285`), so there is one default. §12 D5, FIXED 2026-07-30.

### 5.4 The single rendered-weight store

`ProductionWeightCache` (`production_weight_cache.py:137`) is the only store for rendered
weights and `render_production_weight` (`:1785`) the only producer. Not tidiness: the
surrogate, the KL validation, and the exported bytes must be the *same* rendering, or every A/B
carries a rendering confound. Levers are recorded on the cache (`:165`, `:835-858`), which is
what makes M19 (§6.1) possible.

Render mechanisms are a registry with declared ordering semantics, not a lever string parsed in
spelling order (`render_score.py:188-260`): each `RenderMechanismSpec` declares `operation`,
`scope`, `phase`, `gate_metric` and optional `before`/`after`, and
`resolve_render_mechanism_order` resolves them topologically. Built-ins (`:322-380`):
`four_over_six` (40); then `joint_scale_opt` → `static_act_order` → `gptq` — both levers sit
at phase 50 with `before=("gptq",)` and no relation to each other, and
`resolve_render_mechanism_order` resolves that to `[joint_scale_opt, static_act_order, gptq]`
(matching `pipeline.py`'s own stage list); `fisher_gptq` (50, archived); `scale_sweep` (60,
after gptq). The production lever set is §3.3.

### 5.5 Named invariants

| Name | One line | Detail |
|---|---|---|
| **M6** | Allocator cost scores `weight_mse`, not `h_trace × output_mse` | §4.2 |
| **M19** | Export re-derives NVFP4 codes under the render's *recorded* scale rule, not the export-entry `static_6`. Default ON | §6.1 |
| **M26** | Frontier KL is scored `full_sequence`, not last-token | §4.6, §7.1 |

### 5.6 `input_global_scale` is a free post-export knob

The NVFP4 activation global scale can be patched in place after export and re-measured — no
re-render. `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` selects the compressed-tensors
`generate_gparam` convention `FP8_MAX·FP4_MAX/amax` over the legacy `FP4_MAX/amax`
(`export_native_compressed.py:874-910`); it rescues blocks far below calibration amax from FP8
subnormals at the cost of clipping any serve block above it. Served A/Bs 2026-07-02, weights
byte-identical: 35B-A3B MoE frontier −14.1% KL (win), LFM2.5 +5.8% (loss), 27B regen dense
+37.5% (loss). Strongly artifact-dependent, so the default stays legacy (`0`) and any change
requires a per-artifact served A/B.

## 6. Export & serving invariants

`prismaquant/export_native_compressed.py` (9,130 lines) turns a `layer_config.json` recipe plus
(normally) a `ProductionWeightCache` into a `compressed-tensors` checkpoint. §5 owns the
render; this section owns the bytes and the metadata that make vLLM accept them. Bare `:N`
refs are that file.

### 6.1 Codec map

`_quantize_2d` `:4740-5225`, dispatching on `_canonical_export_format` `:676-680`.

| Format | codec | emitted tensors |
|---|---|---|
| NVFP4 | `quantize_dequantize_nvfp4` `:3147`, packer `pack_fp4_indices` `:851` | `weight_packed`, `weight_scale` (fp8 e4m3, g16), `weight_global_scale`, `input_global_scale` (`:4971`) |
| MXFP4 | `quantize_dequantize_mxfp4` `:3476` | packed fp4 + uint8 E8M0 scales (g32) |
| MXFP8_E4M3 / _E5M2 | `quantize_dequantize_mxfp8` `:3549` | fp8 weight + uint8 E8M0 scales (g32) |
| FP8_E4M3 / _E5M2 | `quantize_dequantize_fp8_dynamic` `:3696` | fp8 weight + per-row fp32 `weight_scale` |
| BF16 | `_passthrough_tensor` `:5727` | verbatim |
| FP8_SOURCE | verbatim copy (§6.3) | source `weight` + `weight_scale_inv` |
| 3-D packed experts | `_quantize_3d_packed` `:5228` + `_split_packed_expert_tensor` `:4540` | per-expert per-projection tensors |

Activation-aware passes compose inside `_quantize_2d` (`:4818-4832`): `gptq`, `scale_sweep`,
`static_act_order`, `joint_scale_opt`, the latter two forced to require `gptq`
(`:4830-4831`). `input_global_scale` follows the compressed-tensors `generate_gparam`
convention `FP8_MAX·FP4_MAX/max_abs` (`_nvfp4_input_global_scale_from_max_abs :895-910`).

**Export refuses what it cannot emit** (#27, `29f3cff`). `EXPORTABLE_FORMATS` `:7517` is
*derived* from `FORMAT_SCHEME` plus the container passthrough, never hand-listed, and the vLLM
lane spec reads its menu from that constant. A format with no emit path used to be rewritten
to BF16 behind a `print` — a Linear allocated at ~4.25 bpp shipped at 16, blowing the byte
budget it was selected under and leaving the artifact's real bpp disagreeing with its own
`layer_config.json`. It is now a hard error naming the Linear, the format and the resolved
profile (`:1548`, `:1574-1589`), with the wrong-container cases (`nvfp4_cb`, GGUF) called out
by name. The *legitimate* coercion is deliberately kept: a format the exporter can emit but
which is shape-illegal or profile-denied still falls back to BF16 and is still audited.

**M19 — export honours the render's scale rule.** `_export_match_render_scale_rule`
`:2130-2147` reads the cache's `levers["nvfp4_scale_rule"]` and re-derives NVFP4 codes under
*that* rule rather than the entry default `static_6`, making the re-quant of the cache's bf16
dequant near-idempotent; `PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE` **defaults ON**
(`:2143`). Packed companion `_packed_expert_render_scale_rule` `:2150-2177` — without it,
joint_mse-rendered experts re-derived under `static_6` flipped 43% of packed bytes. Residual
gap: joint scale *levels* are not in the lever dict, so a non-default
`PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` must match between cache-build and export.
`_pack_production_cached_2d` `:2180-2279` re-packs only — no re-run of GPTQ/scale-sweep, which
would measure a different artifact.

**Block-output match** (`prismaquant/block_output_match.py`), `PRISMAQUANT_BLOCK_OUTPUT_MATCH`
**default `"1"` = ON** (`:6168-6169`, `:6321`, `:6467`): for NVFP4 dense block Linears
(q/k/v/o, gate/up/down) it defers the pack, greedily refines per-Linear group scales against an
FP16 block-reference forward, then finalises (`:6555+`). Its own ~0.05–0.10 PPL estimate
predates JSO and has never been re-measured on the gold lane (§12).

### 6.2 `config_groups` / `ignore` / packed-MoE emission

`build_quantization_config` `:7589-8005` emits explicit per-name targets grouped by format,
remapped to vLLM-internal names via `profile.to_vllm_internal_name`; the catch-all default
group is the format with the most non-BF16 members (`:7951`). Schemes are hand-authored
constants — `NVFP4_SCHEME` `:7247`, `MXFP8_SCHEME` `:7264`, `MXFP4_SCHEME` `:7282`,
`FP8_SOURCE_SCHEME` `:7305`, `FP8_E4M3_SCHEME` `:7325`.

BF16 plus `bf16_passthrough` plus `extra_ignore` go to `ignore` (`:7614-7636`); BF16 **packed**
experts additionally need a per-layer regex over every (expert, projection) because vLLM
scheme-dispatches on per-expert Linear qnames (`_bf16_packed_expert_ignore_regex` `:7386-7481`,
used `:7632`). Fused siblings present in the serving model but absent from the probe (Gemma4
`k_eq_v` with no `v_proj`) are back-filled into `ignore` from `packed_modules_mapping` / the
structure spec (`:7643-7700`). `compute_extra_ignore` `:8055-8095` must *not* add per-expert
source keys when the packed parent is quantized — that marks the FusedMoE un-quantized, the
NVFP4 scale params never register, and weight-load KeyErrors follow (`:8089-8091`).
`_preflight_quantization_config` `:8008-8025` builds the entire config before any GPU render or
shard write (called `:8477`), so metadata violations fail in seconds rather than hours.

**Packed-MoE 3-D.** vLLM's `get_moe_method` probes three *synthetic* names
`<block>.experts.0.{gate_proj,up_proj,down_proj}`, not the on-disk packed qnames
(`experts.gate_up_proj`). Each packed recipe entry is therefore replaced by **one per-layer
regex pinned to that layer index** (`_constrain_per_expert_projection_regex` `:7350-7383`,
`_pin_regex_to_layer` `:7339-7347`, emission `:7779-7800`), with on-disk leaves translated
through `_vllm_moe_scheme_projection_names` `:4503-4525` so LFM2.5's `w1/w3/w2` are advertised
as `gate_proj/up_proj/down_proj` (`:7980-7994`).

### 6.3 FP8_SOURCE verbatim, MTP, audits

`_build_fp8_source_map` `:5747-5854`: a tensor qualifies when `<base>.weight` has a sibling
`<base>.weight_scale_inv` in the index (the 128×128 block convention of MiniMax-M2 /
DeepSeek-V3 / NVIDIA FP8 releases). Bytes are copied unchanged; only the suffix is renamed
(`weight_scale_inv` ≡ compressed-tensors `weight_scale`). A non-FP8 source returns `{}`, which
makes FP8_SOURCE inert — the allocator's passthrough-integrity filter then drops it everywhere
(`:5766-5769`). Overlay `_fp8_source_config_overlay` `:5885-5929`.

**Passthrough integrity now uses the allocator's own vocabulary** (#29, `b6ec9cb`). The
coercion never passed `source_kind`, so `check_format_applicability` judged **every**
FP8_SOURCE Linear illegal and rewrote it to BF16. That was inert in the bytes — materialization
copies the source fp8 verbatim and the config overlay restores the scheme — but it filled every
DSv4 / Hy3 / MiniMax `runtime_coercions` with demotions that never happened, hiding any real
one, and it forced a passthrough exemption in the group-escalation path. `source_kind` now
comes from `_scan_source_dtype_manifest`, the same recipe-keyed map that gates the allocator's
passthrough candidates, scanned lazily so a BF16-source export does no extra header IO
(`:1530-1545`). Bogus rows went 4/4 → 0 on a synthetic fp8 checkpoint; the exemption is deleted,
so a genuine passthrough mismatch inside a serving unit escalates like any other illegality.

transformers v5 does not instantiate MTP for Qwen3.5/3.6 MoE, so `_materialize_mtp_tensors`
`:8939-9002` rebuilds a standalone MTP module under a parent named `mtp` and materialises it in
memory, keeping checkpoint-convention names. **Since 2026-07-30 (R12) it goes through the
profile**: `profile.build_mtp_module(text_config)` builds it, `profile.read_mtp_source_state_dict()`
pulls the source tensors keyed on `profile.mtp_source_prefix()` (default `"mtp."`), and
`profile.load_mtp_state_dict()` loads them, folding per-expert checkpoint keys into the packed
3D expert Parameters. `build_mtp_module`'s contract is that the returned module's names, once
wrapped in that `mtp` parent, equal the allocator's recipe names (`mtp.fc.*`, `mtp.layers.0.*`) —
which is why the recipe filter here stays `mtp.` regardless of the source prefix.
`validate_mtp_assignment_coverage` `:9195-9222` **hard-fails** when the source has tensors under
`mtp_source_prefix()`, the profile `has_mtp()`, and the recipe has no `mtp.*` entries.

`_bf16_upgrade_audit` `:1965-2087` (emitted `:8622`) classifies each BF16 Linear as
passthrough/immutable, runtime-coerced, or a genuine budget choice — a manifest, not a policy;
serving-unit coercions are reported as such (`serving_group` key), because a whole FusedMoE
shipping unquantized is a different and louder fact than one Linear whose own shape was illegal.
`_coerce_runtime_legal_assignment` `:1452-1756` is the defensive legality re-check for stale or
hand-written recipes; it resolves whole serving-atomic components (§6.4).
`_unify_input_global_scales_across_fused_siblings` `:4624-4685` +
`_compute_nvfp4_joint_global` `:4688-4734` force one global scale per fused group (vLLM warns
and degrades otherwise). `_production_cache_fingerprint` `:1125-1182` /
`_production_cache_expected_keys` `:1088-1122` gate cache↔assignment coverage.

### 6.4 Hard serving invariants

Violating any of these yields a checkpoint that crashes vLLM at load or — worse — loads and
silently corrupts.

| Invariant | Enforced at | Failure mode |
|---|---|---|
| Fused siblings (q/k/v, gate/up) share **one** format | DP aggregation over the intersection of member candidates; legality-aware union-find `promote_serving_units` `allocator_solver.py:302-327` + `_choose_group_format` `:192-231`; hard assert `promote_fused` `:362-406`; export re-check `:7896-7944` | ≥2 quantized schemes → load crash (merged-column scale-shape assert). Quantized + BF16 → **loads and silently corrupts**: measured 4.3× worse served KL on Qwen3.x DeltaNet `in_proj_ba` (0.106 vs 0.025 at matched bpp) |
| Packed MoE experts uniform per FusedMoE (mix across layers, never within) | pre-DP `aggregate_packed_serving_groups` (§4.5) + the same union-find pass via `profile.packed_expert_format_group`; export raise `:7780-7792` | unservable; the raise names the usual root cause — allocation produced under `DefaultProfile` because the probe lacked `meta['model']` |
| A serving-atomic unit is never left **mixed** by promotion or by export coercion | promotion picks the cheapest legal-for-all format ≥ max rank and writes **every** member unconditionally (`allocator_solver.py:192-299`); export coercion resolves whole unioned components, raising when a quantized format is legal for all and coercing the *whole* unit to BF16 only when none is (`:1452-1756`) | previously reachable via the un-aggregated solve path and, silently, via Pareto seed-JSON promotion (which `compute_achieved` never prices); the fused-coherence gate reported it only at the very END of export, and as a wrong-model-profile problem it is not |
| Incomplete fused groups → BF16 + `ignore` | `allocator.py:1482`; ignore back-fill `:7643-7700` | the fused loader expects all siblings; a missing `v_proj` breaks the merged Linear |
| Packed `config_groups` use vLLM **canonical** scheme names | `:7980-7994` | no scheme binds to FusedMoE; `w2_input_global_scale` never registers; `load_weights` KeyError |
| Multi-format menu must not resolve to `DefaultProfile` | `validate_default_profile_format_menu` `allocator.py:961-988`, called `:1550-1554` | silently produces the fused-coherence bug class above |
| Final serving promotion is a no-op | `validate_final_serving_promotion_noop` `allocator.py:1046-1063`, called `:2669` | a late promotion means the DP priced an assignment that is not the one shipped |
| Passthrough integrity (BF16/FP8_SOURCE only if the source already is) | `allocator_candidates.py:24-27`, `:112-120`; export judges it against the *same* `_scan_source_dtype_manifest` vocabulary (§6.3) | synthesising BF16 from a dequantised FP8 source burns 8 bpp for nothing |
| Every format in the assignment must have an emit path | `EXPORTABLE_FORMATS` `:7517`, checked `:1548`; the serving profile's `export_lane.codec_formats_from` bounds the allocator's menu by that same constant (`serving_profiles.py:252-330`) | a format with no `config_groups` scheme used to be silently rewritten to BF16 at 16 bpp, blowing the selected byte budget (#27) |
| Registry ↔ served metadata agree on bits/group | **not enforced** — `FormatSpec` (`format_registry.py:44-168`) and the export `*_SCHEME` constants (`:7247-7336`) are independent sources of truth with no reconciling test | a divergence mis-prices bpp or mis-declares the served scheme; §12 D17 |

## 7. Validation & ship gates

### 7.1 What runs where

| Stage | Tool | Run by the pipeline? | Verdict? |
|---|---|---|---|
| Candidate real-KL (selection) | `validate_assignments_kl.py` | yes, only under `SELECTION_MODE=validated-surrogate` (`run-pipeline.sh:1223-1278`) | ranks, does not gate |
| Artifact survey (PPL/MMLU/end-KL) | `validation_harness.py` | no | **no thresholds at all** |
| vLLM load + greedy smoke | `validate_native_export.py` | **echoed only** (`run-pipeline.sh:1704-1705`) | binary |
| Numeric ship gate | `validate_quantized_model.py` | **never run, never echoed** | yes, exit 0/1 |
| Gold lane | `tools/measure_vllm_full_kl.py`, `tools/measure_vllm_wikitext_ppl.py` | never | manual, authoritative |
| Ship record | `exported/shipcard.json` (opened by the exporter) → `python -m prismaquant.shipcard_cli verify` | opened by every export | **refuses** until every serve-lane slot is closed |
| **Publication** | `tools/publish_artifact.py` | no — operator-run | **BLOCKING**: refuses to upload (or even print the upload command) unless `shipcard.verify` passes |

Nothing in the pipeline blocks on a quality number — and it should not: `vllm` is not
importable in the build venv, so embedding a serve inside `run-pipeline.sh` would make the
build tool own the serving stack. The boundary is physical, so the contract is a **record**,
not CI.

**The bar is defined once, per lane** (`prismaquant/lane_specs/*.json` + `lane_spec.py`, re-vet
**R16**). Each lane declares its `{serve command/scripts, endpoint, gate set, KL evaluator}`,
and every gate names the shipcard slot its record closes — so `LaneSpec` is the *runner's*
description and the shipcard is the *refusal*. Two things this made visible rather than
assumed: the CB half was pure wiring (native and CB declare the **same** ship-gate runner on
the **same** endpoint kind, because `validate_quantized_model.py` is endpoint-agnostic), and
GGUF's missing frontier evaluator is a thin adapter over its own harness
(`gguf_kl_evaluator.py`, §9.3). **Gates stay advisory in the pipeline; the blocking point is PUBLICATION** (Robert's ruling on
R16, 2026-07-30). Nothing in `run-pipeline.sh` blocks on a quality number and nothing should —
the build/serve boundary is physical. But an artifact only becomes a claim when it goes public,
so that is where the record is enforced: `tools/publish_artifact.py <artifact_dir> --repo-id
rdtand/<name>` calls `prismaquant.shipcard.verify` as a **library call** (never a subprocess —
the ambient python may not have the package) and **refuses before it uploads anything or even
prints the command it would run**, listing every unfilled or failing slot; a refusal that still
hands over a copy-pasteable command is not a refusal. `huggingface_hub` is imported lazily, so
in the build venv the tool verifies and then prints the exact `hf upload …` line to run
elsewhere. The escape hatch is deliberately expensive: `--force-unverified` requires the
operator to **re-type the artifact directory's basename** (interactively, or `--confirm-name`
for scripts) and stamps `forced_unverified: true` plus the overridden problems into the
shipcard, so the artifact itself carries the record that it shipped ungated. Tests:
`tests/test_publish_artifact.py`.

**The ship record (`exported/shipcard.json`).** `export_native_compressed._write_shipcard`
(`:8111`, called after `mixed_native_manifest.json`) opens a card carrying the build-lane
facts it already holds — git commit, `assignment_hash`, `layer_config_sha`, achieved bpp *with
its provenance named* (read from the allocator's `pareto.knees.json`, never recomputed under a
different accounting convention), exact `artifact_bytes`, format histogram, the render-lever
echo (`_render_lever_provenance()`, shared with the export cache's fingerprint so the two
cannot drift), and the `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` / `PRISMAQUANT_KV_COTANGENT` state
so an allocation that rode an unvalidated Fisher correction is visible on the artifact rather
than only in a probe log (D24) — plus five **empty, required** serve-lane slots:
`native_export.eager`, `native_export.graph`, `ship_gate`, `gold.kl`, `gold.ppl`.

`python -m prismaquant.shipcard_cli verify <card> --model-dir <dir>` exits non-zero unless every slot holds a
*passing* record whose `model_sha` matches the artifact on disk (config sha + per-shard byte
sizes — cheap enough to run on a 90 GB artifact), and unless both `gold.*` records report
`spec_decode_detected: false`. `show` prints the remaining unfilled slots. The validators fill
their own slots via `--shipcard`; `fill --slot gold.kl --record <json>` closes the gold slots
from the measurement JSON. This turns "the numeric ship gate was never run" (the row above)
from a silent omission into an explicit refusal. `verify` is not yet wired into
`run-pipeline.sh`'s closing echo — that is a follow-up wave.

**`validate_assignments_kl.py`** — the pipeline passes `--kl-scope full_sequence`
(`run-pipeline.sh:269`, `:1208`; option `:832`), `--n-calib-samples 32`, `--calib-seqlen 1024`,
and `--calib-skip-first $NSAMPLES` for held-out disjointness (`:1194-1219`); the CLI's own
defaults (2 × 128, `:767-925`) are not what ships. `_kl_repeat_summary` emits
`kl_mean/kl_std/kl_stderr/kl_ucb`. GPU-only via `gpu_guard.require_cuda_hot_path`.

*Key rename, 2026-07-30 (R28):* the mean is now `kl_mean` — it was `last_token_kl` under **both**
scopes, which had already misled a doc. `last_token_kl` is still emitted as a **deprecated alias
for one cycle**, and `select_validated_frontier._row_metric` resolves either, so pre-rename
result JSONs select identically.

*Per-sequence tail, 2026-07-30 (R9):* both measurement paths — `_measure_inplace_assignment_kl`
and `measure_assignment_kl(..., return_per_sequence=True)` — return `(mean, per_seq, stats)`
instead of discarding the per-sequence values they already accumulate. Each row therefore also
carries `kl_per_sample`, `kl_p95`, `kl_p99`, `kl_max` (**the same key names
`tools/measure_vllm_full_kl.py` emits**, so a selection row and a served row are comparable for
the first time; `kl_tail_domain: "sequence"` records the one honest difference — the sample unit
is a sequence, not a position) and the rung-2 term `nll_mean`/`nll_p99`, from one `gather` +
`logsumexp` over student logits already in hand (`kl_measurement.sequence_token_nll`, chunked so
the fp32 upcast stays bounded; `None` under the last-token scope, which has no next-token
label). **Zero extra forwards.** §4.6 is the consumer.

*Held-out disjointness is mechanized, 2026-07-30 (R14):* cost/probe artifacts stamp the
canonical `perturbed_x_cache.calibration_data_hash` into their output meta
(`incremental_probe` per shard, unioned as `calib_hashes` at merge; `aura_cost.provenance`;
`build_production_cache` onto `cache.metadata`, inherited by `production_render_cost`), and
`validate_assignments_kl` **hard-errors** when its own `calib_repeat_hashes` intersect the
probe's or the cost table's. Pre-R14 artifacts stamp nothing and the check stays inert on them
rather than guessing. Relatedly, `--calib-skip-first` on the wikitext branch was a **silent
no-op** (computed, never applied) and now raises — the mechanism that guarantees the held-out
split cannot quietly do nothing.

**`validation_harness.py`** — `validate_artifact` `:77-153` records `{ppl_wikitext, end_kl,
ppl_mmlu_acc, model_sha, layer_config_sha, eval_split, metric_era}` into `artifact_registry`
(`:18`); defaults 65,536 wikitext tokens, 200 MMLU questions, calib 8 × 512 on split `test`
(`:84-89`). Raises on non-finite metrics (`:156`), otherwise passes everything: measurement and
provenance, not a gate. `metric_era` matters — records lacking `eval_split` were measured on
wikitext **train** and are not face-value comparable (`:147-152`).

**`validate_native_export.py`** — does vLLM accept the checkpoint and emit tokens. Defaults
`--max-new-tokens 16`, `--gpu-memory-utilization 0.55`, `--max-model-len 2048` (`:206-209`);
eager by default, `--no-enforce-eager` `:226` is the graph-mode arm, and **`--both-arms`
`:229` runs both in one invocation** — the run-both-arms rule used to live only in the CLI
help text with nothing in code enforcing the second arm; it is now two named shipcard slots
(`_run_arm` `:112`, `_record_arm` `:174`, `--shipcard` `:234`), and each arm tears its engine
down before the next loads. A failed arm exits 1 instead of raising. Flashinfer pinned from
the profile's `runtime_package("flashinfer")` (`:30-71`); `--speculative-config` exercises MTP
(and marks the record `spec_decode_detected`).

### 7.2 `validate_quantized_model.py` — the numeric ship gate

Check order `:12-25`: serve → generation sanity → perplexity/NLL → MTP acceptance. Fixed
12-prompt PPL suite `:87-100`, 4-prompt generation suite `:105-110`. Thresholds `:116-120`,
CLI-overridable `:513-518`:

| Constant | Value | Rationale |
|---|---|---|
| `DEFAULT_MAX_PPL` | 25.0 | catastrophic-breakage bound only (BF16 ~3–5, 4-bit ~4–8) |
| `DEFAULT_MAX_P99_NLL` | 6.0 | ~2σ above BF16 mean; implemented as the **worst per-prompt** NLL guard (legacy flag name), true p99 reported separately (`:20-23`, `:65-69`, `:275-278`). Added after a broken 27B passed on the mean while 80% of prompts were broken — a mean cannot see a tail |
| `DEFAULT_MAX_MEAN_NLL` | 3.0 | mean NLL |
| `DEFAULT_MIN_GEN_LEN` | 30 chars | per completion |
| `DEFAULT_MIN_MTP_ACCEPT_P0` | 0.60 | position-0 draft acceptance |

**Spec-decode refusal.** `_spec_decode_on` `:171-189` scrapes `/metrics` for
`vllm:spec_decode`; if present the perplexity check **refuses a verdict** rather than return
draft-model NLL (`:292-302`). MTP artifacts need the two-serve workflow (`:37-54`): serve
without `--speculative-config` for the PPL verdict, re-serve with it for MTP acceptance;
ship-ready requires both. The same refusal now also guards the gold lane (§7.3) — it used to
exist only here.

`--shipcard` (`:594`) appends this run's whole verdict block (per-check pass/fail, metrics,
thresholds, `base_url`, served model name, detected spec-decode state) to the `ship_gate` slot;
`--artifact-dir` (`:598`) names the local directory the `model_sha` is computed from, since the
validator drives an HTTP endpoint and cannot otherwise know what the server loaded
(`_fill_shipcard` `:516`, `_resolve_artifact_dir` `:502`).

### 7.3 The gold lane (manual)

**Exact full-vocab vLLM KL-vs-BF16** — `tools/measure_vllm_full_kl.py`: `--n-samples 8`
(`:504`), `--seqlen 512` (`:505`), teacher/student two-pass, `--max-logprobs 248320` (`:509`),
`--score-positions final|all` (`:511`), `--prompt-top-k 1024` (`:515`). **The "n=8 × 512"
contract lives here** — not in the pipeline, not in `CLAUDE.md`.
**Direct WikiText PPL** — `tools/measure_vllm_wikitext_ppl.py`: `--split test` (`:118`),
`--n-tokens 8192` (`:119`), `--seqlen 512` (`:120`). Promotion authority is §2.4; these two are
its instruments.

Both tools build their own in-process `LLM`, so the measuring process **is** the server. Two
guards ride on that:

* **Spec-decode refusal** (`tools/spec_decode_guard.py`). Rung-1 authority had no spec-decode
  guard at all until R13 — the refusal existed only in §7.2. `_load_llm` now inspects the live
  engine's `speculative_config` and raises with the draft-NLL diagnostic (`--allow-spec-decode`
  overrides, and the shipcard then refuses the record). Every result dict carries
  `spec_decode_detected`; `None` means "could not inspect" and is refused too — an unverified
  negative is what the original trap looked like.
* **Serve fingerprint + `git_commit`** (§7.4). Neither tool recorded any provenance before —
  gold-lane numbers were *less* provenanced than the surrogate KL JSONs. Each result dict now
  carries `git_commit`, `serve_fingerprint` and the full `serve_manifest`
  (`_provenance`, `measure_vllm_full_kl.py:35` / `measure_vllm_wikitext_ppl.py:30`).

### 7.4 Reproducibility contract

KL is **bit-identical within one docker session** and drifts 4–8× **across** sessions, so
provenance is baked into every KL output JSON: `_git_provenance`
`validate_assignments_kl.py:280`, `_calibration_provenance` `:307` (calib sha256),
`assignment_hash` `:1344`/`:1380`, cache `cache_hit_count` / `rtn_fallback_count` `:371-373`.
An output without these is quarantined, not compared.

**Mechanism of the cross-session drift (2026-07-19).** Loading *any* CUDA extension into the
serving process shifts allocator addresses → activations get different pointer alignments →
alignment-sensitive cuBLAS/CUTLASS heuristic selection elsewhere → ULP-level logit drift. On
the 27B this reads as two bit-reproducible states, conf-KL 0.01134 vs 0.01328 (**±17%**), keyed
purely on whether the gridbook extension `.so` was resident during the dump; ~97% of positions
drift uniformly, so it is global, not path-local.

**Rule:** A/B arms must have identical extension residency and ideally identical
pre-measurement traffic. conf-KL deltas below ~±20% across differing serving stacks are not
evidence either way and should be quoted as a range.

**Mechanized (R15).** The rule is no longer prose an author has to remember.

* **`serve_manifest.json`, written server-side.** Each `scripts/serve_*.sh`, once READY, calls
  `write_serve_manifest` (`scripts/lib/serve_manifest.sh`), which `docker exec`s
  `tools/serve_fingerprint.py write` **inside the container**: launch argv, image tag,
  `vllm`/`torch`/driver versions (`importlib.metadata` + NVML only — the writer never imports
  torch or touches CUDA, so it cannot add a context to a 121 GiB pool), GPU name,
  `enforce_eager`, `--quantization`, `PRISMAQUANT_*` env, and the resident-extension basenames
  read from the **server's** `/proc/<pid>/maps` (`gridbook|prismaquant|flashinfer|causal_conv1d|fla`,
  unioned over the API-server *and* EngineCore processes — it is the engine that holds the
  kernels). Client-side is not an option: the measuring client cannot see the server's address
  space — reading a root-owned container process's maps from the host is *denied*, and the
  denial is indistinguishable from "nothing is resident", which is exactly why the ±17% stayed
  invisible. The manifest therefore records `residency_readable` and folds it into the
  fingerprint, so an unverified scan can never match a verified empty one. Never fatal — a
  serve that came up is not torn down over a JSON.
* **`serve_fingerprint` = sha256(canonical JSON of the manifest minus argv paths).** Path
  elision is load-bearing: arm A and arm B name different artifact directories and must still
  share a fingerprint, while a changed image, extension set, `enforce_eager`, quantization,
  version or GPU must not. In-process gold-lane runs fingerprint themselves from
  `/proc/self/maps` (`self_manifest`).
* **`tools/kl_ab.py A.json B.json` refuses to cross one.** Same fingerprint → a delta. Different
  → exit 3 with **no delta quoted** and the differing manifest keys named;
  `--allow-cross-fingerprint` downgrades the output to a **range** that prints the ±20% band and
  says plainly whether the difference clears it. Legacy JSONs with no fingerprint compare as
  before, with a printed warning.

### 7.5 Validation landmines

| Landmine | Symptom | Handling |
|---|---|---|
| Spec-decode poisons PPL | `/v1/completions` echo+logprobs returns the **draft** model's NLL under `--speculative-config` | detected and refused (§7.2); run PPL on a no-spec serve |
| Gemma / instruct BOS | raw PPL ≈ ln(vocab) garbage when BOS is dropped | use KL-vs-BF16 (`/home/rob/dq-runs/kl_tool.py`); raw PPL cannot separate quantizations of instruct models anyway |
| Activation CPU-residency | tensors from `_LazyActivationCache.get()` are CPU-resident; the matmul silently runs on CPU — no error, no speedup | `.to(device, float32)` explicitly in every batched/sweep path; recurs across export work |
| In-sample "validation" | selection KL measured on text the surrogates saw | `--calib-skip-first $NSAMPLES` (`run-pipeline.sh:1194-1219`); an audit found this had regressed once already |
| Metric-era mixing | old harness records measured on wikitext **train** | check `eval_split`/`metric_era` (`validation_harness.py:147-152`) before comparing |
| Tied embeddings (`tie_word_embeddings`) | the cost stage died on the `lm_head` shard with `NotImplementedError: Cannot copy out of meta tensor` — the checkpoint ships no `lm_head` tensor at all, so the head is a meta alias of `embed_tokens` | `prismaquant/tied_embeddings.py` (landed `d058267`). The head is **materialized** — phase-2's CE backward runs through it, so meta is never acceptable — via transformers' own `get_output_embeddings()`/`get_input_embeddings()`, and **excluded from probe/cost/DP**: a tie means one Parameter, so quantizing the head quantizes the embedding, and probe/cost measure only the head's *output* MSE while the identical perturbation enters every token embedding and thus layer 0 for the whole forward — a cost no surrogate, not even L2 perturbed-X, can observe. There is also nothing to re-encode (no `lm_head.weight` bytes), so `footprint` would either fail to resolve the name or subtract the embedding from the floor while it still ships verbatim. Detection = config declaration AND a source index with no head tensor, never a name guess; a meta head with no declared tie raises immediately. The allocator exclusion (`allocator.py:1010-1043`, called `:1465`) also covers probes built before the fix. It ignores `--allow-pinned lm_head` by design — the tie is a property of the checkpoint, not of the serving profile. Gemma4-31B completed probe → cost → allocate → export for the first time on this fix (**enablement, not a quality claim** — unserved, no KL/PPL) |
| KV-sharing layers (`num_kv_shared_layers > 0`) | phase-3 forwards each layer in isolation and handed the consumer a **detached** K/V, so the storing layer's `k_proj`/`v_proj` Fisher never saw any consumer's contribution — and phase-3 chains each layer's input gradient downward, so the truncation was inherited by every layer *below* the producer too | The KV-cotangent path (`b6ec9cb`): consumers get grad-enabled leaf clones whose `.grad` is the cotangent they contribute, accumulated per storing layer and used to seed that layer's backward alongside its own output cotangent, in one reverse pass (`sensitivity_probe.py:1269-1299`, `:3185-3222`; `incremental_probe.py:1943-2409`). Verified by **exact equivalence** on an fp64 synthetic model — h_trace bit-identical to one end-to-end autograd backward (rel err 0.00e+00) — where the pre-fix protocol under-counts `k_proj` 85.1% and `v_proj` 38.5%. Guard semantics were **inverted, not deleted**: `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` no longer gates KV-sharing models generally; the probe hard-errors only when the path is turned *off* (`PRISMAQUANT_KV_COTANGENT=0`) on a model that needs it, and `PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` still reproduces a pre-fix probe (`incremental_probe.py:1035-1060`). Models without KV sharing are bit-for-bit unaffected either way. **Honest limit:** no real `num_kv_shared_layers > 0` checkpoint has been probed; those percentages are a toy correctness demonstration, not a quality claim |

## 8. Model support: the plugin architecture

Adding an architecture is a registration exercise, not a fork. Three registries hold everything
a model needs; the allocator, solver, caches, exporter and `pipeline.py` contain zero
architecture conditionals. Re-verified 2026-07-30 by AST scan: string literals naming an
architecture that reach **control flow** (a comparison, `startswith`/`endswith`, a dict lookup)
anywhere under `prismaquant/` outside `model_profiles/` and `vendored/` numbered exactly
**three** — the MiniMax hardcodes of §8.5 L4 — and are now **zero**: R27 routed both through
profile accessors (`bypass_hf_fp8_module_rewrite()`,
`packed_expert_module_class_names()`), declared in `specs/minimax_m2.json`.
Gridbook is a separate repository and therefore outside this AST scan. Its supported producer
profiles and serving aliases are imported as data from the installed package's
`runtime_contract.json`; PrismaQuant carries no copy of those tables. An earlier, laxer count
("5 and 2") could not be reproduced and is withdrawn; the
remaining arch-named literals in the core stack are argparse help, log/error text, and
`vendored/`'s registration machinery, which is arch-specific by design — the cosmetic list at
the end of §8.5 is the audited set.

**DIAGRAM-3 — Plugin registries:** the three registries plus the gridbook per-arch loader
chain, what auto-derives from the vLLM class, and the four places production bypasses a
declared extension point.

```mermaid
flowchart TD
  subgraph R1["registry 1 -- model structure"]
    VLLMCLS["vLLM model class<br/>packed_modules_mapping, hf_to_vllm_mapper"]
    DERIVE["auto-derivation -- model_profiles/vllm_registry.py:25-195<br/>fused_sibling_group, fused_sibling_leaf_mapping,<br/>to_vllm_internal_name (prefix mappers only)"]
    SPEC["structure spec JSON<br/>model_profiles/specs/ARCH.json<br/>schema prismaquant.model_structure.v1<br/>match, priority, naming, fused_groups, packed_experts,<br/>pinned_names, passthrough_prefixes, default_serving_profile,<br/>supported_lanes / preferred_lane"]
    PROF["ModelProfile subclass -- model_profiles/ARCH.py<br/>only matches() and name are abstract (base.py:57-66)<br/>Python-only: MTP, streaming adapters, forward state"]
    REGY["model_profiles/registry.py _REGISTERED + detection_order()<br/>ordered by ModelProfile.priority (lower first, ties keep list order);<br/>SpecMatchProfile per unclaimed spec; DefaultProfile terminal fallback"]
  end

  VLLMCLS --> DERIVE
  DERIVE -->|"tier 1"| PROF
  SPEC -->|"tier 2"| PROF
  SPEC -->|"match + priority"| REGY
  PROF --> REGY

  CONSUMERS["consumers -- ~30 detect_profile call sites across 22 modules<br/>probe, cost, cache, allocator, exporters, validators"]
  REGY --> CONSUMERS

  subgraph R2["registry 2 -- serving profiles"]
    SPROF["serving_profile_specs/ID.json<br/>research, vllm_packed_moe, gguf, nvfp4_cb<br/>allow/deny formats, shape rules, runtime validators"]
    RESOLVE["resolve_target_profile -- serving_profiles.py:611-633<br/>explicit request wins first (:623-624)"]
  end

  SPEC -->|"default_serving_profile"| RESOLVE
  SPROF --> RESOLVE
  RESOLVE --> ALLOCGATE["allocator candidate legality<br/>allocator_candidates.py + allocator.py:1661"]

  subgraph R3["registry 3 -- pipeline contract"]
    PIPE["pipeline.py -- declarative, not executive<br/>APPROVED_RESOURCE_OWNERS (:19-26), 14 artifacts, 9 stages<br/>validation is tautological in the production path"]
  end
  CONSUMERS --> PIPE

  subgraph GB["external Gridbook repository -- sole serving/runtime owner"]
    GBPLUG["packaged runtime_contract.json<br/>consumer aliases, format ABI, supported producer profiles"]
    GBSCAN["Gridbook-owned architecture loader registry<br/>version-robust, inert for non-CB checkpoints"]
    GBINST["Gridbook-owned top-level expert loader + fill guard<br/>missing coverage fails closed before execution"]
    GBCFG["Gridbook-owned quantization config<br/>native CUDA/CUTLASS-only, fail-closed<br/>CB / ignore / delegated stock CT / embedding / routed experts"]
  end

  GBPLUG --> GBSCAN
  GBSCAN --> GBINST
  GBPLUG --> GBINST
  GBINST --> GBCFG

  L1["LEAK 1 -- run-pipeline.sh:91<br/>TARGET_PROFILE hardcoded to vllm_packed_moe and passed<br/>unconditionally (:471, :1081); spec.default_serving_profile<br/>can never win. hy_v3 declares gguf, laguna declares nvfp4_cb.<br/>MEASURED 2026-07-11: 226 dense FP8 Linears silently -> BF16<br/>on the Hy3 CT export. PRISMAQUANT_TARGET_PROFILE is the audit<br/>escape hatch and run-pipeline.sh does not set it."]
  L2["LEAK 2 -- FIXED 2026-07-30 (R12)<br/>MTP now routed through profile.build_mtp_module /<br/>read_mtp_source_state_dict / load_mtp_state_dict at all three sites.<br/>mtp_module.py deleted; DSv4 takes the hy_v3 passthrough route."]
  L3["LEAK 3 -- FIXED + OWNERSHIP MOVED 2026-08-01<br/>Gridbook alone owns loader wiring and the fill guard; missing coverage<br/>raises before execution. PrismaQuant carries no loader table and CI compares<br/>its eligible producer profiles with the exact pinned consumer contract."]
  L4["LEAK 4 -- FIXED 2026-07-30 (R27)<br/>streaming_model FP8-rewrite bypass -> profile.bypass_hf_fp8_module_rewrite()<br/>(spec staging.bypass_hf_fp8_module_rewrite); incremental_probe expert<br/>container -> profile.packed_expert_module_class_names().<br/>Zero arch literals in core-stack control flow."]

  L1 -.->|"leak"| RESOLVE
  L2 -.->|"leak"| PROF
  L3 -.->|"leak"| GBPLUG
  L4 -.->|"was leak"| CONSUMERS

  classDef leak stroke:#c0392b,stroke-width:2px
  class L1,L3 leak
```

### 8.1 The three registries

| Registry | Where | Holds |
|---|---|---|
| Model structure | `model_profiles/<arch>.py` (`ModelProfile` subclass) + `model_profiles/specs/<name>.json` (`ModelStructureSpec`, schema `prismaquant.model_structure.v1`, `structure.py:20`) | detection (`match`, `priority`), naming across five name spaces, fused groups, packed-expert layout, pinned/passthrough names, staging, shard regexes, probe skips, `default_serving_profile`, `supported_lanes`/`preferred_lane` |
| Serving constraints | `serving_profiles.py` + `serving_profile_specs/<id>.json` (schema `prismaquant.serving_profile.v1`) | per-format allow/deny rules with name conditions, shape rules, runtime shape validators, runtime package requirements; `extends` composition (`serving_profiles.py:557-609`) |
| Pipeline contract | `pipeline.py` | almost nothing — `target_profile` as a kwarg (`:644`), run metadata (`:688`), CLI passthrough (`:1115`, `:1151`), one `model.structure_graph` stage spec (`:877-884`). Zero architecture names, which is correct: the contract layer should not know models (§3.6) |

Detection is **priority-ordered, not list-ordered** (R8, 2026-07-30). Subset profiles must
still precede supersets — `Qwen3_5DenseProfile` before `Qwen3_5Profile`, `Qwen3MoeProfile`
before `Qwen3Profile` — but that used to be encoded in `_REGISTERED`'s literal order plus two
comments. It is now a `priority` int on each profile (**lower is consulted first**, like a sort
rank), declared both on the Python class and in its spec, so the ordering survives the Python
body being deleted. Built-ins take 100–190 in the historical order; `ModelProfile.priority`
defaults to **0**, which is what keeps `register_profile`'s documented insert-at-front override
true for third parties. `detect_profile` keys on `config.json` `model_type` + `architectures`
and dispatches through `_resolve`, which walks `detection_order()`; unmatched models fall to
`DefaultProfile(architectures=archs)`. `tests/test_spec_match_profile.py` asserts that priority
order still reproduces the list literal exactly.

`detection_order()` folds in a second kind of candidate: a **`SpecMatchProfile`**
(`model_profiles/spec_profile.py`) per `specs/<id>.json` whose `id` no registered Python
profile claims, matched by its declarative `match` block. All ten shipped specs are claimed by
a Python profile, so today the live order contains none — landing the reader changed detection
for exactly zero shipped models, which is the point (see §8.3 Tier A).

`_resolve` also **refuses to hand back a profile whose vendored-modelling override is known
dead** (`_refuse_dead_vendored_override`, added by #19 / `29f3cff`). Its `except Exception:
pass` around `register_vendored_modeling()` is right for keeping *detection* alive, but the old
comment assumed "the eventual model load error" would surface a failure — true only for a
failure that raises. The failure it actually hid is the opposite: `register_qwen3()` returned
cleanly on transformers ≥ 5.13.0 and did nothing, after which the probe ran **upstream** Qwen3
modelling code — on the family behind most shipped artifacts — with no exception anywhere. Root
cause is upstream: `_LazyAutoMapping.register` returns early when the config key's `__module__`
starts with `transformers.`, so no override of a natively-supported `model_type` can land that
way. The fix registers a PrismaQuant-owned subclass of the native config through
`AutoConfig.register` (public API, no internals patched), engages only when the direct route is
verified dead, and verifies every registration by a config-only resolution before setting the
"done" flag. Boundary measured, not assumed: healthy through 5.12.1, broken from 5.13.0.

The `DefaultProfile` fallback is *guarded, not silent*: `allocator.py:1550-1554` calls
`validate_default_profile_format_menu(...)` (`:961-988`), which refuses a multi-format menu
under `DefaultProfile` unless `--allow-default-profile`, on the grounds that fused-sibling
coherence and packed-expert uniformity (§6.4) cannot be enforced without arch knowledge.

### 8.2 Resolution precedence and vLLM auto-derivation

Every `ModelProfile` accessor resolves in one fixed order:

```
vLLM class metadata  →  declarative JSON spec  →  generic hardcoded default
```

Only `matches()` (`base.py:57-61`) and `name` (`:63-66`) are abstract — and `matches()` is now
also spec-expressible, via `SpecMatchProfile` (§8.1). The `match` vocabulary is deliberately
tiny, because nine of the ten in-tree predicates were already pure `(model_type ∈ set, arch
startswith prefix)` tests:

| key | form | why it exists |
|---|---|---|
| `model_type` | exact strings | the common case |
| `architectures` | `fnmatch` globs (a bare class name is a valid exact glob) | `Qwen3Moe*` prefixes, and `qwen3.json`'s exact `Qwen3ForCausalLM` — the one predicate that is *not* a prefix |
| `architectures_exclude` | globs; any hit **vetoes** the whole match | `qwen3_5_dense`'s `not any("Moe" in arch)` |
| `priority` (top level) | int, lower first | replaces the comment-encoded `_REGISTERED` order |

Unknown keys raise at parse time. That matters: `match` was declared in all nine spec files and
parsed since day one with **no reader**, and `qwen3_5_dense.json` had silently drifted out of
agreement with its Python (it was missing the Moe exclusion) — dead config decays.
`tests/test_spec_match_profile.py` is the standing gate: for every registered profile, on every
representative config in the family, the spec verdict must equal the Python verdict. Only after
that is green for a release does a `matches()` body get deleted, one architecture at a time.

What `base.py` reads off the vLLM class named by `vllm_architecture_class()` (`:68-74`, resolved
lazily at `:76-84`, `None` permitted):

| Derived | vLLM attribute | base.py | Spec fallback |
|---|---|---|---|
| `fused_sibling_group()` | `packed_modules_mapping` | `:89-118` | `spec.fused_groups` `:110-115` |
| `fused_sibling_leaf_mapping()` | `packed_modules_mapping` | `:120-164` | same |
| `to_vllm_internal_name()` | `hf_to_vllm_mapper.orig_to_new_prefix` | `:290-319` | `spec.recipe_to_vllm` rules take **precedence** `:314-318` |

The adapter is `model_profiles/vllm_registry.py`: `vllm_class_for_architecture` (`:25-102`)
tries four registry APIs plus internal-table fallbacks and degrades to `None` when vLLM is
absent. It consumes **prefix-substitution mappers only** (`:123-125`) — regex/substring mappers
are skipped, which is why LFM2.5 (`lfm2_moe.py:115-141`), MiniMax (`minimax_m2.py:110-131`) and
HyV3 (`hy_v3.py:75-89`) still hand-override `to_vllm_internal_name`. Spec `regex` rewrite rules
can now express those; `lfm2_moe.json` already does.

Roughly 25 further accessors are pure spec reads (packed-expert names/classes, pinned names,
per-expert regexes, source/recipe/live name mapping, format groups, passthrough prefixes,
staging, layer prefixes, lm_head, probe skips, export-lane eligibility,
`bypass_hf_fp8_module_rewrite`), `base.py:169-820`. Deliberately Python-only,
because they are forward-pass *behaviour* rather than naming: MTP (`:248-272`),
streaming-probe adapters (`:823-947` — `checkpoint_to_live_name`, `fp8_scale_pairs`,
`head_resident_extra_prefixes`, `init_rotaries`, `expand_hidden_for_layers`,
`extra_layer_kwargs`, …), cross-layer forward state for Gemma4 KV sharing (`:949+`, which the
KV-cotangent path now grafts through — §7.5), `register_vendored_modeling()` (`:974-979`).
`vllm_fused_moe_scheme_projection_names` (`:443-468`) is intentionally hardcoded to vLLM's
canonical names — §6.2.

`structure.py`'s `build_model_graph` (five parallel name spaces per tensor) is a declared
contract, not an executor — `base.py:999-1008`, "intentionally not called from hot paths yet";
production reads the accessors.

### 8.3 Adding a model, end-to-end, as it stands today

**Tier A — pure JSON.** Now *possible*, still never done. The obstacle was `matches()` being
abstract; `SpecMatchProfile` (§8.1–§8.2) removes it, so a spec file with a `match` block, a
`priority`, and declared `fused_groups`/`naming` resolves on its own with no Python. Tier A does
**not** get vLLM tier-1 auto-derivation — that is keyed on a Python
`vllm_architecture_class()` — so a spec-only architecture must declare its fused groups
outright. Every one of the ten shipped specs is still claimed by a Python profile, which is
deliberate: the R8 mitigation lands the reader alongside the Python and deletes `matches()`
bodies one architecture at a time, only after a release of green equivalence.

**Tier B — the realistic minimum (5 items).** (1) `model_profiles/<arch>.py` — subclass with
`matches()`, `name`, `vllm_architecture_class()` (may return `None`); 34–172 LoC in practice.
(2) `model_profiles/specs/<name>.json` — the declarative contract (§8.1). (3)
`registry.py:46-57` — import + one line, **in the right order**. (4) Serving profile — reuse
`vllm_packed_moe`, or add `serving_profile_specs/<id>.json` (`extends` supported). (5)
`TARGET_PROFILE=<id>` on the run invocation — leak L1 means the spec field alone does not take
effect.

**Tier C — commonly also needed.** MTP (today: edit `mtp_module.py` itself, see L2; or the
hy_v3 route — `has_mtp → False` plus `passthrough_prefixes` and out-of-band CB encoding
scripts); streaming overrides (`checkpoint_to_live_name` for flat naming, `init_rotaries` for
multi-layer-type rope, `head_resident_extra_prefixes`); cross-layer forward state; vendored
modeling. Then run the conformance validator (§8.6), which nothing else does.

**Tier D — the gridbook CB lane (§9.2) adds per-arch work.** (6) `default_serving_profile:
"nvfp4_cb"` in the spec **and** `TARGET_PROFILE=nvfp4_cb` (gated `run-pipeline.sh:124-125`),
plus `"nvfp4_cb"` in the spec's `supported_lanes` — the lane declaration is what
`require_lane_supported` (`serving_profiles.py`) checks, and it must be added *with* the loader
wiring of (7), never ahead of it.
(7) **Read the architecture's vLLM `load_weights`** and implement any required
top-level expert-loader hook in the external Gridbook repository. Gridbook owns
that registry and its fail-closed fill guard; its packaged consumer contract is
the only architecture-capability list PrismaQuant checks. Do not copy module
paths here. (8) A CB-quantized MTP/drafter needs corresponding Gridbook loader
coverage and a contract update before the PrismaQuant lane declaration lands.

Serving-side registry keys are owned by the pinned Gridbook consumer contract:
canonical `"gridbook"`, with `"prismaquant"` retained only as a read alias for
artifacts exported before the rename.

### 8.4 Conformance matrix

| Arch | profile | prio | structure spec | `default_serving_profile` | `supported_lanes` (preferred) | gridbook opt-in | MTP |
|---|---|---|---|---|---|---|---|
| qwen3 (dense) | `qwen3.py` | 130 | ✅ | `vllm_packed_moe` | CT, **nvfp4_cb** (CT) | no special loader hook | none |
| qwen3_moe | `qwen3_moe.py` | 120 | ✅ | `vllm_packed_moe` | CT | ⚠ none | none |
| qwen3_5 / 3.6 MoE | `qwen3_5.py` | 110 | ✅ | `vllm_packed_moe` | CT, **nvfp4_cb** (CT) | declared by pinned Gridbook contract | `build_mtp_module` → `MtpModule` (live; R12) |
| qwen3_5_dense | `qwen3_5_dense.py` | 100 | ✅ | `vllm_packed_moe` | CT, **nvfp4_cb** (CT) | no expert-loader hook | inherits `Qwen3_5Profile.build_mtp_module` (dead copy removed, R12) |
| gemma4 | `gemma4.py` | 140 | ✅ | `vllm_packed_moe` | CT | ⚠ none | none |
| lfm2_moe (LFM2.5) | `lfm2_moe.py` | 150 | ✅ | `vllm_packed_moe` | CT | ⚠ none | `has_mtp → False` |
| minimax_m2 | `minimax_m2.py` | 160 | ✅ **added R22** — all 8 overrides declared | `vllm_packed_moe` **(added R22)** | CT | ⚠ none | `has_mtp → False` |
| deepseek_v4 | `deepseek_v4.py` | 170 | ✅ | `vllm_packed_moe` **(added R22)** | CT | ❌ absent from the pinned Gridbook contract; loader/delegation unimplemented and lane gated | `has_mtp → False` + `mtp.` passthrough (hy_v3 route, R12) |
| hy_v3 | `hy_v3.py` | 180 | ✅ | `gguf` (overridden, L1) | CT, nvfp4_cb, **gguf** (gguf) | declared by pinned Gridbook contract | `has_mtp → False`; MTP passthrough + out-of-band CB scripts |
| laguna (poolside S/XS 2.x) | `laguna.py` | 190 | ✅ | `nvfp4_cb` (overridden, L1) | CT, **nvfp4_cb** (nvfp4_cb) | declared by pinned Gridbook contract; drafter still separate | `has_mtp → False` |
| default | `default.py` | — (terminal) | n/a by design | — | CT (default) | n/a | none |

`prio` = detection priority, lower first (§8.1); the same number is declared on the Python class
and in the spec, and a test asserts they agree. **CT** = `compressed-tensors`. The lane column is
the *declared* set (R6, spec `supported_lanes`/`preferred_lane`), and required CI compares the
five CB producer profiles with Gridbook's packaged contract; GGUF has one. Over-declaring is the exact
failure the field exists to prevent: an undeclared lane does not fail loudly, it serves
uninitialised expert memory. `require_lane_supported(profile, EXPORT_CONTAINER)`
(`serving_profiles.py`) runs in `run-pipeline.sh` before profile resolution and export.

Gaps beyond the four leaks. **minimax_m2's missing spec is closed** (R22, 2026-07-30): all
eight overrides (`:69,:86,:91,:101,:104,:110,:133,:137`) are now declared in
`specs/minimax_m2.json` — `fused_groups`, `packed_experts` (+`projection_splits`,
`format_groups`), `moe.per_expert_regex`, and three `naming.recipe_to_vllm` regex rules for the
`block_sparse_moe.experts.N.w{1,2,3}` → `mlp.experts.N.{gate,down,up}_proj` rename. The Python
overrides stay for now; `tests/test_minimax_m2_spec.py` compares a spec-only profile against
the pre-spec Python behaviour accessor by accessor, and that gate must hold for a release
before the Python comes out. It closed a latent bug on the way: without a spec,
`packed_expert_format_group` fell through to the legacy fallback, whose first group is
`(gate_up_proj, down_proj)`, so MiniMax's `down_proj` got a *different* coupling key than its
`gate_proj`/`up_proj` — one expert bank in two format groups, which violates §6.4 and would
have been unservable. **`deepseek_v4.json` now declares `default_serving_profile:
vllm_packed_moe`** (R22) — the conservative, provably-tighter choice while its lane is
undecided; `research` carries no format allow-list at all. The
spec did gain `_verified_source_layout` (`2b5b937`, closing #26): the real
DeepSeek-V4-Flash-Base headers say routed experts are I8 nibble-packed MXFP4 with F8_E8M0
scales while **shared experts are block-FP8 E4M3, not fp4** — settled against the checkpoint
and the authors' `convert.py`, not inferred. **`serving_profile_specs/vllm_qwen3_5_packed_moe.json`
is an empty `extends: [vllm_packed_moe]` alias whose own description says not to use it** — but it
is NOT unreferenced: `.github/scripts/check_installed.py`, `tests/test_allocator_packed_group_units.py`,
and `tests/test_serving_profiles.py` all name it, so deletion means retiring those references and
checking shipped artifact metadata first.
**Mistral-Medium-3.5-128B is in the shipped family table (§1.2) with no profile** — no Mistral
profile class or spec exists (the sole textual mention is a comment at
`model_profiles/default.py:6`), so it ran under
`DefaultProfile`; the `allocator.py:1550-1554` gate would refuse that menu today. Finally, the
never-declared `unpacked_expert_projection_names` (`base.py:470-495`) **is now declared** — by
`specs/minimax_m2.json` (R27), which also required adding it as a real
`ModelStructureSpec` field; `base.py` had been reading it off the spec with `getattr`, so no
declaration could ever have taken effect. Other architectures still ride the `('w1','w2','w3')`
default, which is correct for them.

### 8.5 Known contract leaks

These four are the canonical statement; §12 references them rather than restating them. **All four are now FIXED** (L1/R11, L2/R12, L3/R10, L4/R27, all 2026-07-30) and are kept in the table with their fix, so each leak and its resolution stay in one place.

| # | Leak | Severity |
|---|---|---|
| L1 | **FIXED 2026-07-30 (R11).** Was: `run-pipeline.sh` hardcoded `TARGET_PROFILE:=vllm_packed_moe` and passed it unconditionally, and `resolve_target_profile` gives an explicit request precedence (`serving_profiles.py`), so `spec.default_serving_profile` was **never consulted through the production orchestrator** — `hy_v3.json` (`gguf`) and `laguna.json` (`nvfp4_cb`) silently overridden. **The leak had a measured cost:** because export re-resolved the profile it judges legality under, on 2026-07-11 **226 dense FP8 Linears were silently demoted to BF16** on the Hy3 compressed-tensors export. **Mechanism of the fix:** (i) the shell default is now empty and `--target-profile` is passed to the allocator **only when non-empty**, with a new `--target-profile-default vllm_packed_moe` supplying the fallback for architectures that declare nothing — never `research`, whose menu is unbounded. (ii) The allocator stamps its **resolved** profile into `layer_config.json`'s reserved `__prismaquant__` metadata block (`layer_config.LAYER_CONFIG_META_KEY`, skipped by every assignment parser and by the schema), and `export_native_compressed._allocator_target_profile_for_audit` reads it, with `PRISMAQUANT_TARGET_PROFILE` kept as the operator override for direct exporter invocations. `select_validated_frontier` carries the block forward when it overwrites the layer config, so the validated path keeps it too. Allocator and export can no longer disagree, and the channel travels **with** the artifact. **Non-regression:** re-solving the shipped 27B and 35B from their stored probe/cost artifacts changed **0 of 614** and **0 of 500** assignments vs the same code without the change (the 35B differs from its *shipped* config by 32/500 for an unrelated, pre-existing reason — the Fisher renormalization fix that landed after that artifact shipped). Every in-tree launch script sets `TARGET_PROFILE` explicitly, so all eight are bit-identical. | ~~high~~ FIXED |
| L2 | **FIXED 2026-07-30 (R12).** MTP construction bypassed the profile: `prismaquant/mtp_module.py` was Qwen3.5-specific yet imported **directly** by `incremental_probe.py`, `incremental_measure_quant_cost.py` and `export_native_compressed.py`, gated only on the arch-agnostic `profile.has_mtp()`, so `deepseek_v4` (`has_mtp → True`, `build_mtp_module → None`) would have been handed a Qwen3.5 decoder layer. **Mechanism of the fix:** a fourth accessor `ModelProfile.mtp_source_prefix()` (`base.py:255-272`, spec-expressible as `shard_regexes.mtp_source_prefix`, default `"mtp."`) plus a generic `read_mtp_source_state_dict()` (`:290-326`) and a packed-expert-aware `load_mtp_state_dict()` (`:329-396`, absorbed from the deleted `_load_into_mtp`); `build_mtp_module`'s docstring now states the naming contract (names under an `mtp` parent must equal the recipe names). The Qwen body moved verbatim into `model_profiles/qwen3_5.py:124` (`MtpModule`) and the dead near-copies in `qwen3_5.py` and `qwen3_5_dense.py` were reconciled into it; all three call sites now go through the profile and hard-fail with a named error if `has_mtp()` and `build_mtp_module()` disagree. `prismaquant/mtp_module.py` is **deleted**. DSv4 takes the hy_v3 route (`has_mtp → False` + `"mtp."` in `passthrough_prefixes`) until its nextn block is actually quantized. Gates: `tests/test_mtp_module_arch.py` pins parameter-name-set equality against the pre-move layout for both the dense and MoE profile; `tests/test_model_profile_conformance.py::test_has_mtp_implies_a_buildable_mtp_module` is the standing ratchet. | ~~high~~ FIXED |
| L3 | **FIXED 2026-07-30 (R10), ownership boundary hardened 2026-08-01.** Was: a hand-maintained `try/except ImportError` opt-in chain whose missing-line failure mode was **coherent-looking garbage generation**. Gridbook now owns the module-path registry, fill guard, and tests in its sole canonical repository. PrismaQuant owns no loader or runtime copy; its required CI job installs the exact pinned Gridbook commit, validates PEP 610 provenance, and compares the producer profile set with Gridbook's packaged `runtime_contract.json`. The runtime stamps CB expert parameters unfilled, stamps them after either loader path, and fails closed before execution if any local registered expert remains unfilled. **No env bypass.** | ~~high~~ closed |
| L4 | **FIXED 2026-07-30 (R27).** Both MiniMax hardcodes now go through profile accessors. `streaming_model.py`'s FP8-rewrite bypass was already half config-derived (`quant_method == "fp8"` and `weight_block_size`); the architecture half is a static property, so it became `staging.bypass_hf_fp8_module_rewrite` in the spec behind `profile.bypass_hf_fp8_module_rewrite()` (`base.py`), leaving the per-checkpoint half a config read where it belongs. `incremental_probe.py`'s `type(module).__name__ == "MiniMaxM2Experts"` became `profile.packed_expert_module_class_names()` (`base.py:182-192`) — the accessor that already existed for exactly this lookup — plus the structural shape test; the declared class stays **required**, because the replacement forward implements one specific expert-loop signature and applying it to a lookalike container would silently change a forward pass. `specs/minimax_m2.json` declares both, and `unpacked_expert_projection_names` with them. | closed |

Cosmetic, listed so they are not re-discovered as leaks:
`export_native_compressed.py:94,151-152` imports `Qwen3_5Profile` for `_COMPAT_QWEN_PROFILE`
(verified test-only back-compat); `_fast_kernel_guard.py:86-90`'s Qwen substring list is a
labelled fallback for remote HF IDs with no local `config.json`; `layer_streaming.py:1914-1920`
imports an upstream transformers Gemma3 masking helper under config-driven selection;
`gridbook/config.py:174-194` shared-prefix aliasing is HunYuan-motivated but written
structurally.

### 8.6 The conformance validator

`python -m prismaquant.model_profiles.validate --model <path>` implements 8 conformance checks
(docstring `validate.py:17-53`): profile claim `:136`, vLLM class resolvable `:153`,
fused-sibling self-consistency against vLLM's own sibling lists `:191`, name-remap fixed points
`:219`, MTP module construction `:246`, source-passthrough prefixes matching ≥1 real tensor
`:270`, packed-expert param names `:355`, serving profile exists and its validator callables
import `:462`. Exit 0/1, CI-shaped.

**It now has callers** (2026-07-30). `tests/test_model_profile_conformance.py` runs the
CPU-safe part over every registered profile — checks 1, 6 (against synthetic index fixtures)
and 8, plus four structural invariants (spec presence, fused-sibling source, registry order,
name uniqueness); the vLLM-registry checks 2/3/4 sit behind an `integration` marker (their
answer is vLLM-version-dependent) and the real-checkpoint index checks 6/7 behind `slow`.
Check 5 (MTP) is deliberately absent: `build_mtp_module()` materialises a full decoder layer,
a multi-GB CPU allocation — use the manual CLI for it. Its cheap declarative half IS automated
since 2026-07-30 (R12): `test_has_mtp_implies_a_buildable_mtp_module` fails any profile that
answers `has_mtp()` without a real `build_mtp_module` override or `mtp_source_prefix()`, which
is the L2/D2 defect class. Known gaps (`minimax_m2` has no spec; `deepseek_v4` returns `None` from
`vllm_architecture_class()`) are encoded as *ratchets*: each asserts the gap is still real and
only then xfails, so closing one turns the test red with an instruction to shrink the list.
And there is CI to run it — `.github/workflows/ci.yml` (#18, `1cc7b90`) executes the suite on
every push and PR, on py3.11 and 3.12 with CPU torch. §12 D11.

## 9. Serving lanes

Three artifact containers, one allocator. `EXPORT_CONTAINER` picks the lane (§3.3) and
constrains the whole run: both non-default lanes hard-gate `PRODUCTION_CACHE=0` and a matching
`TARGET_PROFILE` (`exit 2`), and must pass the render-faithfulness assertion (§4.7) — their
exporters ship imatrix-weighted bytes, so the cost render must be weighted too. That is no
longer the same as requiring `COST_MODE=local`: since CB Milestone C the production cache can
render those families weighted (`--col-weights`), so a cached-menu render is admissible on
these lanes, opt-in and non-default. Each lane's serve command, endpoint, gate set and KL
evaluator are declared in `prismaquant/lane_specs/*.json` (re-vet **R16**); the gates are
advisory and the shipcard is what refuses — at **publication**, via `tools/publish_artifact.py`
(§7.1).

**DIAGRAM-2 — Serving lanes:** the three artifact containers, the runtime each requires, and
the one box any of it has been proven on.

```mermaid
flowchart LR
  subgraph CONT["artifact containers"]
    A1["compressed-tensors<br/>NVFP4 / FP8_DYNAMIC / FP8_SOURCE / BF16<br/>export_native_compressed.py"]
    A2["codebook (CB)<br/>production: NVFP4_CB_K12-K24 + FP8_CB_K28-K48<br/>legacy/research decoder: signed S13-S16<br/>plus stock rungs -- deliberately a mixed container<br/>export_nvfp4_cb.py, cb_layout.py"]
    A3["GGUF<br/>Q2_K..Q8_0 + IQ family<br/>export_gguf.py"]
  end

  subgraph RT["runtimes"]
    R1["vanilla vLLM<br/>no plugin, no forked runtime, no custom kernels<br/>CUTLASS NVFP4 path on Blackwell"]
    R2["vLLM + Gridbook plugin<br/>exact commit from gridbook_runtime_pin.json<br/>entry point vllm.general_plugins; runtime details owned by Gridbook"]
    R3["llama.cpp"]
    R4["vLLM GGUF path<br/>in-tree up to vLLM 0.19; official vllm-gguf-plugin after"]
  end

  subgraph HW["hardware"]
    H1["NVIDIA GB10 DGX Spark<br/>Blackwell sm_121, 128 GB unified memory<br/>~121 GB usable serving budget"]
    H2["Strix Halo<br/>CANCELED / UNSUPPORTED<br/>prototype removed after hardware access was lost;<br/>no qualified Gridbook backend"]
  end

  A1 -->|"serving profile vllm_packed_moe"| R1
  A2 -->|"serving profile nvfp4_cb"| R2
  A3 -->|"serving profile gguf"| R3
  A3 -->|"serving profile gguf"| R4

  R1 -->|"Spark-proven -- shipped rdtand artifacts"| H1
  R2 -->|"Spark-proven -- 295B-class at ~2.9 bpp on ONE Spark"| H1
  R3 -->|"Spark-proven -- 295B-class at 2.8 bpp; the KL harness for this lane"| H1
  R4 -->|"smoke-verified on the 0.19.2 venv only, never KL-measured"| H1

  R3 -.->|"no qualified deployment"| H2
  R4 -.->|"no qualified deployment"| H2

  classDef proven stroke:#2d7a2d,stroke-width:2px
  classDef unsupported stroke:#c0392b,stroke-width:2px,stroke-dasharray:4
  class H1 proven
  class H2 unsupported
```

PrismaQuant paths below are repo-root-relative. Gridbook source paths refer to the separately
versioned repository at the exact commit recorded in `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`.

### 9.1 Native compressed-tensors — the default lane

`export_native_compressed.py` writes a stock checkpoint: no forked runtime, no plugin, no custom
kernel — the only lane whose correctness depends on nothing we maintain. All of §6 belongs to
it; §7 owns its gates. Validation runs in-process (`validate_native_export.py:171` constructs
`LLM(...)`), so it needs a venv or container carrying vLLM (§10).

### 9.2 gridbook — codebook (CB) serving

**Package, registration, and ownership.** Gridbook is one independent package with one source
repository, one test suite, and one release history. It registers through vLLM's general-plugin
entry point and owns all serving code: configuration parsing, architecture loader hooks, CUDA
sources, kernel dispatch, telemetry, and runtime tests. PrismaQuant owns the inverse boundary:
model analysis, allocation, artifact encoding, and exporter metadata. There is no runtime source
under this repository and no sync operation between repositories.

The complete integration is one immutable record, `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`.
Every serving script resolves that record through `prismaquant/gridbook_runtime/gridbook_runtime.sh`, accepts only
an exact clean commit checkout, mounts it read-only, and independently re-reads the tracked pin
inside the container before re-attesting and installing the checkout. Branch names, moving tags,
dirty trees, wheels, and editable installs are rejected. Serve fingerprints include the resolved
Gridbook commit, so an A/B cannot silently compare different runtime code.

**One machine-readable contract, not parallel tables.** Gridbook packages
`gridbook/runtime_contract.json`; it is authoritative for the runtime's quantization aliases,
CB rung ranges, serialized packing/type-size rules, and supported producer-profile ids.
PrismaQuant's required `gridbook-contract` CI job VCS-installs the exact pinned commit, verifies
its PEP 610 provenance and package version, and compares the producer's declarations against
that contract. That job is also where the **K0.2** stage attestation runs end to end in a single
process: `tests/test_gridbook_attestation_interop.py` emits a routed-MoE record with the real
producer and feeds it to the pinned runtime's parser, payload validator, stage verifier, and
artifact-level K0.2 verdict. It exists because a stage entry must declare *exactly* its five
attested fields while every digest in the contract is framed over those same five by name — so a
field ADDED on the producer side moves no hex, leaves both repositories' suites green, and fails
for the first time at vLLM model load. Adding an architecture or changing a runtime ABI therefore
starts in Gridbook, then advances this repository's single pin only after the contract test
passes. PrismaQuant does not import Gridbook while exporting and carries no parallel runtime
alias or loader table. Its producer codec remains an intentionally independent implementation of
the artifact ABI; CI
compares every packing/layout field and every rung so incompatibility fails at the boundary.

At runtime `register()` registers `"gridbook"` plus the legacy artifact alias `"prismaquant"`
and installs the per-architecture loader hooks. It does not patch vLLM core. Gridbook 0.7.0
resolves and attests every serving-reachable extension, optional-kernel mode, ABI, device, and
shape contract during model load. Decode, expansion, activation QDQ, and routing support are
native CUDA; GEMM and grouped GEMM are native CUTLASS. A missing or ineligible required native
operation raises instead of selecting Triton, a fallback-capable vLLM helper, or another serving
implementation. The container may mix CB groups, ignored BF16 prefixes, and stock NVFP4/FP8
groups delegated to vLLM. Gridbook's own FP8 transient paths call vLLM's registered native
CUDA quantizer and CUTLASS scaled-matmul operators directly after attestation. Fused dense and
grouped native-NVFP4 paths remain explicit opt-ins: the 2026-08-01 teacher-backed LFM gate
rejected default enablement even though operator arithmetic passed. The canceled gfx1151/ROCm
prototype was removed rather than maintained as an unqualified second backend.

**Storage format.** Product vector quantization onto a codebook whose every entry lies exactly
on a hardware grid, so a decoded tile *is* a bit-standard NVFP4/FP8 tensor and dequantization
is a gather rather than arithmetic. A weight vector is d=8 wide; a k-bit index selects a
codeword; 32 codewords plus scales form a 256-weight superblock (external Gridbook
`gridbook/codec.py`). Two
ladders, every integer rung: `NVFP4_CB_K12–K24` (E2M1 grid, 1.78125–3.28125
serialized body bpw under production layout v2) and `FP8_CB_K28–K48`
(E4M3 grid, 3.5–6.0 bpw) — `prismaquant/cb_layout.py`, sourced into
`serving_profile_specs/nvfp4_cb.json`. A third, signed-codebook ladder
`NVFP4_CB_S13–S16` remains codec/export/decoder compatible for legacy and
explicit research use, but is **excluded from new production allocations**.
The 2026-07-22 Qwen3.5-0.8B matched-rung screen found product K lower in
609/776 weight-MSE comparisons (78.48%); only six signed units survived the
2.6-bpp allocation. The reproducible command, comparison definition, source
identity, and artifact checksums are in
`docs/results/qwen35_0p8b_s_rung_headtohead_2026-07-22.md`. Keeping decoder
support preserves already exported artifacts without advertising a losing rung
to production solves (`format_registry.py:932-983`,
`export_nvfp4_cb.py:277-281`). Storage rate and compute
precision are independent dials: `FP8_CB_K32` *stores*
4.0 bpw and *computes* in fp8 — why CB beats native NVFP4 at matched bpw (fp8 rungs run A8
activations where NVFP4 runs A4). Codebooks live in a `.pqcb` safetensors sidecar pointed at
from `config.json` (external Gridbook `gridbook/config.py`,
`export_nvfp4_cb.py:630-639`); the non-globbed extension
keeps vLLM's weight loader off it.

**Runtime defaults and kernel provenance live only in Gridbook.** The old table
here was removed after it drifted from the runtime it described. The current pin is Gridbook
0.8.0 at `9011a19228ddb96b8a49e11a20ac75c99c83998e`; resolve it from
`prismaquant/gridbook_runtime/gridbook_runtime_pin.json`, then consult that source's
`docs/PLUGIN.md`, `docs/KERNELS.md`, and dated audits. The cross-project policy
is only this: a numerics-changing path cannot be promoted by kernel arithmetic
or speed alone. The latest teacher-backed LFM gate rejected both fused-NVFP4
defaults, so dense and grouped paths remain explicit opt-ins.

**DSv4-Flash-0731 exact-shape native A/B (dated record, measured 2026-08-01).** This
paragraph is ported verbatim from the 0.5.1-era study working tree and is kept under its
measurement date. It was taken against the then-uncommitted native-only Gridbook candidate
(base `4e7c1bc6` plus a dirty tree), which has since been cut and pinned as Gridbook 0.6.0
at `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1`; the numbers are therefore candidate-era
evidence for that pin, not a re-measurement of it.

The exact-shape native A/B used the seven ordinary Linear calls repeated across all 43 DSV4
body blocks (301 calls total), flushing 256 MiB before each timed call. K36 was 1.083x faster
at `M=1` and 1.064x at `M=2`; native NVFP4 crossed over by `M=4` (1.120x), reached 1.589x at
`M=8`, and was 1.34–2.50x faster over measured `M=9..1400`. Replacing the retired
Triton-containing `M=16` route reduced K36's weighted block subtotal from about 1.168 ms to
0.763 ms (1.53x). All 12 fused quality gates passed: worst relative L2 `2.146e-5`, max absolute
`0.015625`, and at least 99.9983% BF16 output equality. Evidence:
`/home/rob/dq-runs/dsv4-flash-0731/synthetic-bench-native-quality/README.md`; harness SHA-256
`e38338b1e07469560ecc359af9e35af2ceb35476f72e79edcef0450acae765cb`, wide-result SHA-256
`b63fec8a08930b5d1091dc17577261642d2df84786cfb8925c561375ad85a8ca`, decode-result SHA-256
`a93cf8957c531ba3108ea27671de50f9b0fda2db26a710c85d4fbc0fb6b7cad5`. These are
candidate-kernel measurements, not served tokens/s.

**Per-arch wiring — the no-longer-silent no-load trap** (R10, 2026-07-30). Archs whose vLLM
loader maps experts at the top level never call the per-layer `FusedMoE.load_weights`, so
Gridbook installs its top-level wrapper from the packaged runtime contract. The authoritative
module list is deliberately not repeated here. DSv4 remains absent and gated; Gridbook 0.5.0's
warm synthetic DSV4-shape grouped-bridge timing is a kernel microbenchmark, not artifact
qualification or a production promotion.
Over-installing is harmless and a missing module is a no-op.
An unwired arch used to load no stacked-CB expert tensors at all while the FusedMoE served
uninitialised memory — garbage generation, not a crash (confirmed, `9a79963`: Laguna, 93% of
params). That is now a hard serve-time failure: `create_weights` stamps `_pq_cb_filled = False`,
both fill paths stamp `True`, and `process_weights_after_loading` raises through
`cb_fill_guard.assert_cb_experts_filled` naming the model class and the module path to add —
no env bypass, scoped to the params the local rank registered (EP/PP-absent and zero-expert
shards skipped). New-arch checklist: §8.3 Tier D; leak record: §8.5 L3 (closed).

**Serving.** Every live CB launcher resolves the same exact external Gridbook pin through
`prismaquant/gridbook_runtime/gridbook_runtime.sh`, mounts the exact checkout read-only, re-verifies the tracked
pin inside the container, and records the resolved runtime in the serve fingerprint. The OOM discipline lives in the launcher
code (`serve_laguna_smoke.sh:64-90`):
poll `/v1/models`, sleep 10 s for the allocator to settle, fail the serve and `exit 3` if
`MemAvailable < MIN_FREE_GIB` (8), then arm a detached watchdog that kills the container below
`WATCHDOG_GIB` (4). Rationale inline at `serve_hy3_teb.sh:76-84`; see §10.

**Encode path.** Stage 4/4 (`run-pipeline.sh:1526-1640`): harvest per-column imatrix weights
from the same activation cache the cost stage used (`:1549-1582`; REQUIRED for every CB target
— no silent RTN), then `export_nvfp4_cb`, or `export_nvfp4_cb_streaming` when the source
exceeds `EXPORT_STREAMING_THRESHOLD_GB` (80) — non-streaming goes resident and OOMs the box on
200–300B sources. Encoder tiers `PRISMAQUANT_CB_ENCODE_TIER ∈ {fast, balanced, max}`, default
`balanced` (`nvfp4_cb_formats.py:128-141`); `max` is bit-identical to the pre-tier encoder and
is the regression anchor. There is no in-lane serving smoke — CB artifacts serve only through
the out-of-tree plugin — but the gate set is now *declared* (`prismaquant/lane_specs/nvfp4_cb.json`,
re-vet **R16**): same OpenAI endpoint, same endpoint-agnostic `validate_quantized_model`, plus
the lane-specific prefill perf gate (INV-2). Gates are advisory; the shipcard refuses, and
`tools/publish_artifact.py` is where that refusal binds (§7.1).

**Milestone C is closed (2026-07-30, re-vet R3).** `render_production_weight` /
`build_production_cache` take `col_weights`, so a `ProductionWeightCache` render of a CB rung
is the exporter's imatrix-weighted render and the lane's cost, KL and shipped bytes can come
from ONE render (§4.7). The `COST_MODE=local` restriction is gone; render-score / AURA
objectives are reachable here but **opt-in and non-default**, because the accuracy case for
AURA is native-lane evidence and no served CB objective A/B exists. Lane defaults now match
shipping practice (§12 D15 closed).

**Proven results.** These measurements remain tied to their recorded runtime commits; they are
not relabelled as Gridbook 0.7.0 native-only measurements.

| artifact | result |
|---|---|
| Qwen3.6-27B @5.5 bpp | vs shipped PrismaAURA-5.5bit on the same BF16 dump: conf-KL −45…−53%, ALL-KL −56/−58%; PPL gap to BF16 3× smaller (9.166 vs 9.251, BF16 9.123). **Matched by construction, not by luck**: 16.713 vs 16.707 GB of quantized body (Δ 0.04%), 23.62 vs 23.61 GB total — the same byte budget spent on codebook vs uniform NVFP4/FP8. All 386 quantizable body Linears chose CB rungs (K36 136 / K40 30 / K44 77 / K48 143), zero stock NVFP4/FP8. (The "19.93 vs 23 GB, 0.0082 vs 0.0130" pair quoted in `serving-tax-elimination.md:63-64` is the *iso-quality* framing of a different, lower-bpp artifact — do not mix it into this row, as an earlier draft of this table did.) |
| Ornith-1.0-35B MoE @4.75 bpp | conf-KL 0.01706 vs 0.03625 (−53%), ALL-KL −43%, PPL gap to BF16 −30%, decode ~33 tok/s vs BF16 28.4 |
| Hy3-295B-A21B @2.9 bpp, **one Spark** | 105.73 GB resident; prefill 89 → 108.7–115 tok/s across the kernel campaign vs the shipped GGUF-IQ's 42 (2.1× → 2.6×) — the lane's thesis (tensor-core CB removes the IQ dequant tax) proven at 300B class; decode 13.1 base / 16.1 prose with the K44 MTP draft; TEB 88 vs GGUF-IQ 87 / k-quant 86. **No quality claims** — a 295B cannot be KL-validated on this box |
| Laguna-S-2.1 @6.0 bpp / 84 GB | MoE prefill 293 → 1,821 → 2,063 → 2,186 tok/s under `auto`; native grouped-CUTLASS 3,603 — remaining gap 1.65× |

**RESOLVED (2026-07-30, R28 / §12 D21).** The canonical public repo id for the Hy3 CB
artifact is **`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`** — cite that one.
`prod_hy3_results.md` records two older ids from the ship sequence,
`rdtand/Hy3-295B-A21B-PrismaQuant-2.9bit-nvfp4cb-vllm` (2026-07-20 ship ledger, `:248`) and
`rdtand/Hy3-295B-A21B-gridbook-2.9bit-vllm` (2026-07-21 joint-menu re-ship, `:313`); both were
renamed rather than deleted, so both **307-redirect** to the canonical id (verified against the
Hub 2026-07-30). The two historical citations are annotated in place rather than rewritten —
they are the ship ledger, and the ledger records what was posted on the day.

**Standing native-parity policy** (`docs/lanes/nvfp4-cb/format-speed-policy.md`): minimize
predicted quality loss subject to exact whole-artifact bytes, separate p95 TTFT and p95 ITL /
p05 TPS SLOs, device-residency limits, and backend/shape/TP/serving-unit legality. There is no
`quality + lambda*time` objective and no blended `serve_ms`. Per-layer timings may propose an
assignment, but same-session served KL/PPL/tasks plus end-to-end timing decide it. Exact-4.5
Stage 0 favors production-faithful K36 weight error (493/496 units at 27B, 252/252 at 4B), but
that is a stop-only surrogate, not a served promotion. Plain M=1 decode evidence does not imply
batched/speculative parity, and the Hy3 zero-NVFP4 allocation is circular under its
accuracy-only objective. The DSv4 screen makes the same conditional rule concrete: K36 won
301/301 activation-fitted ordinary-Linears at `0.390x` NVFP4's sensitivity-weighted error and
adds only `8,019,608` bytes to the proposed 92 GB artifact, so it is the quality-first,
low-concurrency arm; native NVFP4 is the TTFT/batching/speculative-throughput arm because its
kernel crossover occurs before `M=4`. Whole-model served A/B remains authoritative.

**Implementation status: SHIPPED (ultraplan P5c, `b052255`).** The constrained
Pareto formulation above is current allocator behaviour, not future work.
`prismaquant/serve_constraints.py` + `prismaquant/serve_dispatch_table.py` are
wired into the allocator (`allocator.py:158-163` imports, `:1979-2009` context
construction and the ACTIVE/INACTIVE banner, `:2951` the per-assignment
evaluation). The constraints are **hard and enforced at assignment level** — in
the exact-payload / byte-budget ratchet, on the EXPANDED promoted assignment
that actually ships, not inside `solve_allocation`'s bits-DP, whose
unconstrained semantics are unchanged and pinned by test. An assignment that
misses p95 TTFT, p95 ITL, p05 TPS or `resident + KV + peak_scratch` is
INFEASIBLE and leaves the candidate set; it is never merely re-ranked. There is
**no λ**, no phase-weighted `serve_ms` and no default workload mix: the
objective is still exactly minimum predicted Δloss, with the existing ratchet
tie-break. Fail-closed — an assignment the dispatch table cannot price is
infeasible, not passed. Supplying no table and no SLOs leaves every code path
**byte-identical** to the pre-P5c allocator, with a stamp recording that
constraints were absent. Profile legality masks (including signed-rung
exclusion) are enforced as before. What has *not* changed is the promotion
rule: table-driven latency is proposal data, and the same-session served
NATIVE-PARITY protocol is what promotes an assignment. A format allow-list
still does not prove a backend, activation contract, or promotion state; those
live in the structured native-parity benchmark record. Normative policy and the
full shipped/gating split: `docs/lanes/nvfp4-cb/format-speed-policy.md:42-98`
("What is implemented, and what still gates promotion"); design and the eight
named assumptions: `docs/design/constrained_pareto_allocation.md`.

For the DSv4
92 GB study, the quality arm is 35 routed-expert layers at K15, eight provisional layers
`23,24,25,27,28,31,32,33` at K14, and 301 ordinary Linears at K36: tensor payload
`91,724,116,088` bytes, or `91,992,551,544` bytes with the 256 MiB reserve. Replacing those
301 ordinary Linears with native NVFP4 yields `91,716,096,480` payload bytes. Neither arm is
release-eligible until DSv4 body/MTP/DSpark loader coverage, production expert calibration,
export/load/generation, and whole-model quality/speed gates pass; the current Gridbook pin
still declares DSv4 unsupported (§8.4).

### 9.3 GGUF

A single `.gguf` that llama.cpp serves natively and vLLM through the official
`vllm-gguf-plugin`. No custom kernels anywhere; the only lane reaching 2–3 bpw, where NVFP4 is
the compressed-tensors floor. Menu: k-quants Q2_K–Q6_K/Q8_0 plus the IQ family
(IQ2_XXS…IQ4_NL), all with `gguf-py dequantize(pack(w))` pinned **bit-identical** to the
registry emulation, so measured cost and shipped bytes cannot diverge (`docs/lanes/gguf.md`).
Container correctness is delegated: we requantize llama.cpp's own
`convert_hf_to_gguf --outtype bf16` skeleton and own only tensor bytes.

Three measured facts carry the lane. **imatrix is the dominant lever at ~3 bpw** — 0.6B KLD
2.728 → 0.913 from activation weighting alone, applied in lockstep to the batched cost path and
the exporter under one flag (`PRISMAQUANT_GGUF_IMATRIX`, default on, §3.3).
**GPTQ-into-k-quant** freezes the two-tier scales from the weighted search and re-decides only
`q` under full-Hessian OBS: 0.6B at matched 347 MB, KLD 0.890 / 56.9% top-1 vs llama.cpp's best
stack at 0.913 / 55.6% — the first arm to beat them on their own harness. **The 4B scale check
is honest about the gap**: byte-matched, the fully consistent stack lands at 0.510 vs their
0.461 (+10.6%) = ~+7.7% residual render (Hessian rank — 1024 activation rows is 10.5% rank at
4B) plus ~+2.6% allocation. Deep-bpw surrogate mis-ranking is the known regime failure;
validated-frontier selection is the house answer, and since re-vet **R16** the lane HAS its
evaluator — `gguf_kl_evaluator.measure_assignment_kl` wraps `llama-perplexity
--kl-divergence-base` behind the `validate_assignments_kl` interface and returns
`(mean, per_sequence, stats)` under the gold lane's key names, with `per_sequence` empty and
`kl_tail_domain="aggregate"` because llama.cpp reports token-domain quantiles. The frontier
loop is not wired to it yet, so `SELECTION_MODE=surrogate` is still what this lane runs
(§12 D26).

Shipped: `rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` — 103.686 GB at 2.799 bpp from the
prod `tencent/Hy3` base, measured allocation with IQ rungs displacing Q2_K/Q4_K entirely,
single Spark, vLLM smoke only (no quality claims). IQ vs k-quant at matched bytes: decode 17.8
vs 18.7 tok/s (−5%), TEB 87 vs 86 (churn at one plateau), and **prefill 42 tok/s is the whole
IQ tax** — k-quants have CUDA MMQ, IQ falls to MMVQ/Triton. That number is what the CB lane
exists to remove (§9.2). Open work: MoE expert stacking in cost/export;
**running** the ship gate on this lane — no analog is needed, `validate_quantized_model` is
endpoint-agnostic and llama-server speaks OpenAI, which is what `lane_specs/gguf.json` declares
(the pipeline smoke still proves load+generate only); embedding/head format as a measured
decision rather than operator policy. Strix Halo enters this lane first, serving-only
(re-vet R7): `docs/lanes/gguf.md` carries the dated tracking table.

## 10. Hardware & environment

One NVIDIA GB10 / DGX Spark ("sparky"), Blackwell sm_121, **128 GB unified memory** shared
physically between CPU and GPU, ~121 GB usable, 1.8 TB NVMe. Two consequences that catch every
newcomer: "move it to CPU to spare the GPU" is a **no-op** for memory pressure, and a
production run gets the box — concurrent heavy agents or downloads starve the launch-bound cost
loop. Every production hot path must be GPU-bound; `prismaquant/gpu_guard.py:7`
(`require_cuda_hot_path`) refuses to run otherwise (though seven stages never call it — §12 D9).

**OOM discipline.** The pool has no evictable slack, so an allocation that would merely swap on
a discrete-GPU box kills the machine instead. Rules, all learned from kills: serve at util
**0.90 or below** for spec-decode + compiled configs (0.94/0.95 died under long-prefill
activation spikes with a drafter resident, `prod_hy3_results.md`); arm the slack gate and
watchdog (§9.2); never bench a new kernel while a serve holds the pool. An idle serve is not
safe — one killed the box ~1.75 h after going quiet.

| environment | use | note |
|---|---|---|
| `/home/rob/dq-runs/venvs/prismaquant-cu130` | build / probe / cost / export / PPL | torch 2.11+cu130; `PYTHONPATH=.` for tests; the host `.venv` has no torch |
| `/home/rob/dq-runs/venvs/prismaquant-hy3` | Hy3 (`hy_v3`) chain | transformers 5.13; the cu130 venv lacks `hy_v3` |
| `/home/rob/dq-runs/venvs/prismaquant-vllm-kl-20260521` | vLLM 0.19.2 in-tree GGUF | the working local GGUF-serving venv |
| `vllm-node:latest` | all four CB serve scripts; the Hy3 GGUF stack | native HYV3; the only serving image the current scripts reference |
| `~/.cache/prismaquant-cb-ext` (or `PRISMAQUANT_CB_EXT_DIR`) | Gridbook JIT build cache | never `/tmp` (external Gridbook `gridbook/cuda_ext.py`) |

`transformers` pins are model-specific and have cost hours: MiniMax requires 4.57.5,
Qwen3.5/3.6 need ≥5.5 (4.57.5 raises `KeyError` on the model type). Older launchers and
`CLAUDE.md` name images (`vllm-fresh-b12x`, `vllm-node-tf5-cu132-lfm`) that are **not present on
the box today** — treat those references as historical.

**Disk.** Keep ≥10% of the 1.8 TB free (224 GB at time of writing). A 27B production cache is
~90 GB and a multi-arm matrix is bounded by peak, not final state: `df -h /home/rob` before
launching, build → measure → delete before the next arm. **Never write to `/tmp`** — an OOM
cleared it in 2026-04 and took the MiniMax artifacts with it. Set `TMPDIR` explicitly for any
tool reaching for `mkdtemp()`.

**Strix Halo / ROCm — CANCELED 2026-07-31; no supported backend.** Access to the only gfx1151
machine was lost before build ABI, dispatch, fallback, graph, wheel-install, vLLM, or served
quality gates could be completed. The prototype sources and dispatch hook were deleted from the
canonical Gridbook tree; PrismaQuant contains no copy. ROCm is therefore unsupported and must
fail through the ordinary absence of a qualified backend. Reintroduction requires new hardware,
hard architecture attestation, installed-wheel tests, and the full served promotion ladder.

The remainder of this subsection is a **frozen historical measurement record**, not an active
implementation description, support claim, or build plan. Paths named below belonged to the
now-deleted prototype.

**Historical Strix snapshot.** Robert funded kernel
authoring on 2026-07-30 ("build fp8 vllm kernels that target strix halo and support codebook
formats"), which **supersedes re-vet R7's serving-only framing** (R7 accepted GGUF-first with
"no ROCm build stack in scope" and left option B — a HIP port of the CB kernels — unfunded; the
outcome note in `docs/audits/architecture_re-vet_2026-07-30.md` §R7 records the change). A box
arrived the same day: AMD Ryzen AI MAX+ 395 / Radeon 8060S, **gfx1151**, RDNA 3.5, Fedora 44,
ROCm 7.1.1, torch 2.9.1 with HIP, **58 GB** unified memory (not 128 — size test shapes
accordingly), 919 GB free.

The deleted prototype's `csrc_hip/` held the result: a wave32 decode GEMV (fp8-CB **and**
NVFP4_CB two-tier v2), a bf16 **WMMA** decode-in-prologue prefill GEMM, the transient expander,
and the fused fp8 activation QDQ, sharing one format header with the CUDA lane. It **compiles,
runs, and passes parity at a 1-bf16-ULP gate against an fp64 torch reference** across the odd
rungs, every register-tile M boundary, ragged edge tiles and the fused multi-role codebook case.

**Benchmark discipline on this box is itself a finding:** gfx1151 idles at ~1.2 GHz and needs
~45 s of sustained load to reach ~2.6 GHz, so a conventional 5-iteration warmup measures the
idle clock. Every number taken that way was discarded and re-taken behind a `sustain_clock()`
that prints the achieved clock. The related correction matters more: there is **no large
bandwidth headroom to reclaim** — stock bf16 GEMV already measures 201–233 GB/s against a
~210 GB/s copy ceiling. **The decode GEMV's justification is the index stream, not out-coding
hipBLAS**: a CB rung reads `k/8` bits per weight (4.5 bpw at K36) where bf16 reads 16, and the
right question is how much of that byte advantage survives the decode. Measured against a
perfectly bandwidth-bound bf16 GEMV of the same logical matrix, at N=K=4096: **K36 = 1.55×,
K40 = 1.40×, then a cliff to a flat 0.80–0.89× from K42 through K48**. Flat-while-bytes-rise
means the top rungs are **decode-bound, not bandwidth-bound**, and the cliff coincides exactly
with the codebook LUT ceasing to be LDS-resident. Actionable consequence: **on Strix prefer the
K≤40 rungs** — which is also where a 58 GB box wants to be. The WMMA prefill GEMM reaches
**11.9 TFLOP/s** (K44, M=128), ~22% of a spec-derived peak.

**The load-bearing measured fact: RDNA 3.5 has no fp8 matrix instruction.** A device-pass
`__has_builtin` probe (`csrc_hip/wmma_probe.hip`) shows bf16/f16/**iu8** WMMA present and every
fp8 variant absent — those are gfx12/RDNA4. So fp8-CB here is a **storage** format decoded to
bf16 fragments; since e4m3's 3 mantissa bits fit bf16's 7 the decode is lossless and the
arithmetic is *better* than an fp8-MMA path, not worse. The same probe established the wave32
fragment layout on device (`D[2*i + lane/16][lane%16]`), which is what lets the GEMM decode B
straight into registers with no LDS tile, and measured that the WMMA unit is **not** exactly
rounded (~0.5 f32 ULP), which is why every parity gate here is relative rather than bit-exact.

**The codebook grid is a free choice on this platform, and the kernels are built for that.**
Since an FP8_CB codeword must be materialised as bf16 for WMMA regardless, a bf16-grid codebook
is same-bytes, same-speed and a strict grid superset of the e4m3 one — so the likely artifact
design is one index stream plus a per-grid codebook (~0.02% of artifact bytes), Blackwell
reading e4m3 and Strix bf16. The HIP kernels are therefore **dtype-agnostic at materialisation**:
the LDS LUT is always filled as bf16 and any e4m3→bf16 conversion happens **once at fill time,
never per gather** (the ALU term R6 removed on Blackwell stays removed), pinned by a test that
asserts a bf16 sidecar is **bit-identical** to the e4m3 one. The cost is LDS footprint —
a materialised LUT is 2 B/element, so K48 and NVFP4_CB K24 need the full 64 KiB. Those still
*fit* here, because A and B are register-resident and the LUT is the only LDS consumer; the top
rungs use the global-gather arm because it is measurably faster, not because the LUT overflows.
`iu8`/`iu4` WMMA are present, and measured on this box at **1.56× bf16 LDS-fed but only 1.06×
register-resident** (iu8) and 2.86× (iu4) — iu8's gain is halved LDS traffic, not faster math,
which buys little in a GEMM whose operands are already in registers. Both need integer
activations and an accuracy gate is running; **neither is started**.

**Not yet true, and the docs must keep saying so:** no vLLM-ROCm serve has been attempted, so
there is **no serving-metric claim** — no KL, no PPL, no tok/s under a real engine; the
`linear_hip.py` dispatch is authored but never exercised inside a live serve; MoE/grouped-expert
HIP kernels do not exist, so an MoE artifact falls back to Triton on ROCm; and the signed S-rung
path is compiled but untested. Quantization stays on the Spark either way — R7's load-bearing
half is unchanged, since probe/cost/render/export are CUDA and need zero ROCm work. GGUF remains
the zero-code serving lane (§9.3). Full status, the LDS budget table per rung, the measured
LDS-vs-global LUT policy and its shape-dependence, seven Fedora/ROCm bring-up landmines and the
deferred-work list was kept in the deleted prototype's `csrc_hip/README.md`; the dated audit is
the surviving record.

## 11. History — what was tried and rejected

Two conventions. (a) Every rejection gets a **dated wall**: `archive/<name>_YYYY-MM-DD/` with a
top-level `README.md` banner stating the kill order and the lesson. (b) Four of those walls are
**load-bearing for the orchestrator** — `run-pipeline.sh` fail-fast messages name them by path,
so `archive/` cannot be moved or renamed without editing the `exit 2` gates of §3.5. Doc-only
walls live under `docs/archive/`; code walls under repo-root `archive/`.

| Method | Why it lost (the lesson) | Wall / gate |
|---|---|---|
| grouped-KL cost surrogate | "−3.52% PPL" was a local screen; lost the vLLM A/B. Promote on the serving metric. | `archive/grouped_kl_2026-05-28/` · gate §3.5 |
| Fisher-weighted GPTQ / Fisher output-MSE allocator | Killed by order; no demonstrated utility on a production model. | `archive/fisher_2026-05-15/` · gate §3.5 |
| Hadamard-DuQuant (HDQ) | Fold-only preconditioner, no served win. | `archive/hdq_2026-05-14/` · gate §3.5 |
| Multi-shot recalibration | Double-negative: ΔKL=0 at production calib, −153% on a small calib. | `archive/multi_shot_2026-05-19/` · gate §3.5 |
| CLADO full IQP solver | O(N²) per-pair measurement; the O(N) cascade matched it to 1–2%. Framing kept (`decision_units.py`), solver dropped. | `archive/cross_layer_2026-05-09/` · docs `docs/archive/block_clado/` |
| Sparse pairwise QUBO / SMRF | 8-of-~500-Linear coverage is homeopathic; too local to fix global non-additivity. | same wall |
| Top-K Hessian covering | Blind to the propagation graph; misses small-eigenvalue Linears with long downstream paths. | same wall |
| L3-polish-of-many DP | Per-Linear L3 costs measured under L2 context do not sum when many units flip at once. | `archive/polish_2026-05-15/` |
| Top-down / ceiling-start polish | Spends its budget on cheap ~12-bit flips, never reaches the knee bpp range. | same wall |
| Coordinate-descent polish (as a shipped stage) | Overfits at n=8 (train→val sign flip); provable only under its own polish-time evaluator. | same wall |
| HALO / Hadamard-Fisher rotations | Worked once on Qwen3.5 dense, never on a production model; cut in the 2026-05-15 consolidation. ParoQuant (`2511.10645`) is the tracked replacement. | `archive/halo_2026-05-15/` |
| ReSpinQuant / layer-wise rotations | Needs a residual-transition adapter (a custom kernel) at serve time — forbidden in the vanilla-vLLM container. | `archive/respinquant_2026-05-13/` |
| Fold-scale / OrthoG, DuQuant++ fold | Preconditioner family, no served win at matched bpp. | `archive/foldscale_orthog_2026-05-13/`, `archive/duquant_dqpp_2026-05-13/` |
| PrismaClip / PrismaFisherClip | Subsumed by JSO's per-block scale grid — clipping is another way of asking what the right scale is. | `archive/prismaclip_2026-05-14/` |
| `scale_sweep` as a default lever | +77.5% KL on 4B: re-picks block scales *after* GPTQ, mis-calibrating its error compensation. Still reachable via `--enable scale_sweep` for ablations. | no wall (menu-only) |
| SAO (column permutation) | Failed on its own objective; redundant with GPTQ's full-Hessian propagation. | `archive/sao_2026-05-15/` |
| REAP / expert pruning | Cost model under-counts token redistribution and misrouting. Hit size via format/factorization, not pruning. | `archive/reap_2026-05-15/` |
| Entmoot expert-merge | Never wired into the runtime. | `archive/entmoot_2026-05-03/` |
| Analytical / closed-form GPTQ damp | +100–161% KL vs the discrete sweep; the fit's 2.4× per-Linear error compounds. Then the sweep itself fell (below). | `docs/design/unified_render_theory.md` |
| GPTQ damp sweep (as default) | Its evaluator was in-sample; held-out basins invert 31/31; served A/B null per role. Fixed damp 1.0 (§5.3). | flag-only, `PRISMAQUANT_GPTQ_DAMP_SWEEP=1` reproduces |
| Surrogate-only knee | On 27B the surrogate knee picks 5.857/0.056, validated picks 5.31/0.015. Outside the additive trust region, bpp order ≠ KL order. | superseded by `SELECTION_MODE=validated-surrogate` |
| Kneedle as the ship rule | Axis-dependent and LOO-unstable (fp32 4B: elbow at 5.00 in 454/1000 bootstraps). Byte budget + saturation B* replaced it; `allocator.py:1247-1252` says so in the CLI itself. | demoted, not removed |
| Lagrangian λ-bisection (as selector) | The discrete frontier has non-convex pockets no λ selects. Kept as a candidate *generator*. | demoted |
| The three-level cost cascade (L1→L2→L3) | **Retired from the spine 2026-07-30 (R4).** L2 beat additive L1 by −1.5% while AURA beat L1 by −38.5%; pairwise residuals are +5–12% diffuse with 3/1180 pairs significant and the apparent non-additivity is a bf16 artifact; L3-polish-of-many does not compose. One faithful cost + real-KL selection replaced all three (§2.2, §4.4). | `archive/l3_propagated_2026-07-30/` |
| `COST_MODE=production-render-staged` | Rendered NVFP4 first, promoted only the top-30% error tail — so every Linear outside the tail carried an `unavailable` cost the DP could not consider. On 27B its last-token-KL screen improved (0.0232 vs 0.0280) while **direct WikiText PPL regressed 10.83 vs 8.33**; the result doc says "Do not ship". The canonical screen-vs-gold inversion. | `archive/production_render_staged_2026-07-30/` |
| `MSE_PROMOTION` post-frontier rewrite | Re-ranked the *already measured-KL-selected* point by local `output_mse_per_bit`. On 35B it beat the strategic baseline but lost to both the shipped 4.75 and the 5.16 kneedle. A post-allocator rewrite cannot beat a better cost inside the DP (AURA). No shipped run carries `layer_config_before_mse_promotion.json`. | `archive/mse_promotion_2026-07-30/` |
| `PRODUCTION_CACHE_UNION` smart-union cache | Saved ~40% of the frontier render by offering an FP8 rung only above a percentile of the NVFP4 `output_mse` surrogate — a render-budget heuristic deciding the allocator's candidate set (principle 1). Never used by a shipped artifact. | `archive/union_cache_2026-07-30/` |
| Block-output match (quality lever #12) | **Unreachable, not unmeasured.** The production-cache pack `continue`s first, so on the shipping recipe no dense NVFP4 Linear ever reached the branch (0 hits in two real export logs). Had it run it would have re-derived NVFP4 scales outside `_export_match_render_scale_rule` and discarded the render's `joint_mse` scales (the −6.6% M19 defect). Its `{0.95,1.0,1.05}` gain search is subsumed by JSO; its "~0.05–0.10 PPL" was a pre-JSO expectation. | `archive/block_output_match_2026-07-30/` |
| Orphan modules and tools (4 + 6) | Zero references tree-wide; three belong to threads with recorded verdicts (damp-sweep OFF-final, xlayer null, export-config collapse subsumed). `_fast_kernel_guard` is the exception — an orphan that is a **missing caller**, booked as debt in §12 rather than declared dead. | `archive/orphans_2026-07-30/` |
| MXFP8 in the default menu | E8M0 pow2 scale wastes ~√2 of a binade (+13.8% output MSE over 410 Gemma Linears); exact-scale FP8 Pareto-dominates. Registry entry retained. | de-menued (§5.1) |
| NVINT2 / NVINT3 Triton kernels | Standalone vector kernels are memory-latency-bound (~6 ms/call floor on GB10); never vLLM-served. Removed from the tree. | git history only |
| MXFP4-grid codebooks | Shares NVFP4's element grid exactly and differs only in the scale plane — but E8M0's pow2-only scale costs **~25× at the 4-bit grid what it costs at 8-bit**, and the 8-bit figure (+13.8%) is what de-menued MXFP8. The cross-platform premise is also false as stated: gfx1151 has no fp4 matrix path at all, so an MXFP4 grid buys nothing on the only non-Blackwell box we own. Revisit only with hardware that runs MXFP4 natively (RDNA4/CDNA4/Intel). | `docs/design/mxfp4_cb_feasibility.md` |
| MXFP6-grid codebooks | **In a codebook the grid is not a storage dial.** The stored stream is the k-bit index stream, so an MXFP6-grid rung stores byte-for-byte what FP8-CB stores at the same k; both MXFP6 grids are *exact subsets* of e4m3 (63/63 values round-trip, verified), so the codebook is an FP8-CB codebook handicapped to a subset, decoding to the same e4m3 tile and the same GEMM. Strict dominance, not a tradeoff — no measurement warranted. Would change only on a part with a genuine 6-bit matrix rate (CDNA4/MI355X runs MXFP6 at 2× fp8). | `docs/design/mxfp6_cb_feasibility.md` |
| CB persistent-N dense prefill; decode contract v2; w2 `rowpack`; chunked expand/GEMM overlap | Parity-green, 0.74–5.7× slower. Quarantined behind flags, kept as measured negatives. | `docs/lanes/nvfp4-cb/STANDARDS.md` |
| CB `l2_pipeline` MoE prefill | Wedged live serving three times; DIAGNOSTIC-ONLY, excluded from `auto` in external Gridbook (`gridbook/moe.py`; evidence commit `afc64ec`). | same |

Derivations and the additivity/cancellation analysis behind the CLADO/QUBO rejections belong to
`paper/main.tex` §`sec:additivity`; the retired PrismaSCOUT paper (cascade spine, monotone
polish, full rejected-methods catalog) is at
`paper/archive/prismascout_paper_2026-06-05.tex`. Dated measurement records are under
`docs/results/`; superseded narrative docs under `docs/archive/`.

## 12. Known gaps and debt register

Honest register, code-cited, as of 2026-08-03 (`release/prismaquant-0.8.0`, implementation
baseline commit `7183d21`; external Gridbook pin
`9011a19228ddb96b8a49e11a20ac75c99c83998e`, v0.8.0). The DSv4 study's working tree carried a
proposed **D29** ("the native-only Gridbook candidate is measured but not yet an attested
runtime"). It is deliberately **not** ported: the 0.6.0 merge cut and pinned that candidate,
so re-adding the row would assert a stale pin (`59cebf9f…`, v0.4.1) that no longer exists in
this tree. The study's measurement half survives as the dated §9.2 record.
Severity is operational risk, not effort. Plugin-contract leaks are stated in §8.5 and only
referenced here. Entries closed on 2026-07-30 are kept, marked, for one cycle so a reader
returning with a stale copy sees the resolution rather than silence.

**State after all four re-vet waves.** Closed today: D2, D3, D4, D5, D6, D7, D8, D9, D10, D12,
D13, **D14**, **D15**, D16, D19, D20, D21, D27 — and D1 is implemented but shipped default-off,
so its *default* remains a decision for Robert rather than debt.
Still open, unchanged by the waves: **D11** (no profile-validator preflight for the actual
`MODEL_PATH`), **D17** (registry vs export-scheme metadata unreconciled), **D18** (two dead
PrismaQuant flag tokens not yet deleted), **D23** (no
accounting-era stamp), **D24** (KV-cotangent path never run on a real KV-sharing checkpoint),
**D25** (Gemma4 tied-embeddings result is enablement, not quality), **D28** (fast-kernel guard
has no caller) — and **D26**, whose measurement half closed (a GGUF KL evaluator exists) while
its plumbing half did not (the frontier loop is not wired to it; `PACKED_ROLE_SPLIT` is still
unplumbed).

| # | Item | Evidence | Sev | Suggested action |
|---|---|---|---|---|
| D1 | **FIXED 2026-07-30 (R9).** Tail-veto was unimplemented since 2026-06-05 — and it had stalled on an assumed cost (a second eval pass) that does not exist. **Mechanism:** every KL site already accumulated per-sequence values and discarded them at the return; both selection paths now return `(mean, per_seq, stats)`, so `kl_p95/kl_p99/kl_max` and the rung-2 `nll_mean/nll_p99` (one `gather` + `logsumexp` on logits already in hand) cost **zero extra forwards**. `_frontier_from_rows` gained a second admission condition — `row[tail] <= incumbent[tail] * (1 + tail_eta)` — behind `--tail-veto {none,kl_p99,kl_max,nll_p99}` / `--tail-eta`, with vetoed rows retained under `vetoed_rows` + `veto_reason` so a refusal is visible. **DEFAULT-ON since 2026-07-30** with `kl_max` (the worst sequence) as the contract statistic — Robert's ruling; `--tail-veto none` still reproduces the pre-R9 envelope byte-for-byte (pinned by a frontier-identity regression test). The slack is **derived, not chosen**: `--tail-eta auto` (default) is the incumbent's between-repeat relative stderr of the tail statistic, degrading to a strict 0 **with a printed warning** on a single repeat. A pre-R9 validation JSON (no tail column on any row) makes the veto go inert with a warning rather than empty the frontier. §4.6, §7.1. | `select_validated_frontier.py` `_frontier_from_rows`, `measured_rows`, `tail_eta_auto`, `tail_veto_inert_reason`, `TAIL_VETO_COLUMNS`/`TAIL_REPEAT_COLUMNS`; `validate_assignments_kl._kl_repeat_summary` (per-repeat tails); `kl_measurement.sequence_token_nll` / `summarize_per_sequence_kl`; `tests/test_select_validated_frontier.py`, `tests/test_kl_per_sequence_tail.py` | — | CLOSED — default-on, `kl_max`, repeat-derived eta (ruled 2026-07-30). |
| D2 | **FIXED 2026-07-30 (R12).** MTP construction bypassed the profile — §8.5 L2. All three import sites now call `profile.build_mtp_module()` / `read_mtp_source_state_dict()` / `load_mtp_state_dict()`, keyed on the new `mtp_source_prefix()` accessor; `prismaquant/mtp_module.py` is deleted and DSv4 declares `has_mtp → False` + `"mtp."` passthrough. | §8.5 L2 | ~~HIGH~~ CLOSED | — |
| D3 | **CLOSED 2026-08-01 (R10 ownership follow-through)** — was: Gridbook per-arch CB expert opt-in as a hand-maintained list inside PrismaQuant, with a missing line failing silently as coherent garbage generation. Gridbook now solely owns its loader registry and unbypassable fill guard; PrismaQuant carries no runtime copy and required CI checks its one exact pin, PEP 610 provenance, producer-profile set, and emitted artifacts against Gridbook's packaged contract. | §8.5 L3; `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; `tests/test_gridbook_runtime_contract.py` | ~~HIGH~~ closed | — |
| ~~D4~~ | **CLOSED 2026-07-30 (re-vet R11).** `TARGET_PROFILE` has no shell default; `--target-profile` reaches the allocator only when requested, with `--target-profile-default vllm_packed_moe` as the fallback; the allocator stamps its resolved profile into `layer_config.json`'s reserved `__prismaquant__` block and the exporter reads it (env override kept). Non-regression 0/614 and 0/500 on the shipped 27B/35B. See §8.5 L1. | §8.5 L1 | ~~HIGH~~ | closed |
| D5 | **RESOLVED 2026-07-30.** `PRISMAQUANT_GPTQ_DAMP_SWEEP` had two readers with opposite defaults — `"0"` in the exporter, `"1"` in a forked lever-defaulting copy inside the KL sensitivity probe (stale from `9c91d62`, missed by the sweep-OFF policy in `f2363e2`), so any A/B touching both compared different renders. `_normalized_production_cache_levers` now delegates to `production_weight_cache._resolve_production_render_levers` — one contract, and the probe's stamped provenance can no longer disagree with the render that produced it. | `archive/l3_propagated_2026-07-30/prismaquant/kl_sensitivity_probe.py:272-285`; `tests/test_production_weight_cache.py` | — | Done, and fully closed later the same day: R4 walled the probe itself, so the forked reader no longer exists in the live tree. The follow-up landed too — the delegation contract is pinned by `tests/test_production_weight_cache.py`, which carries it as a local shim and keeps every assertion (sweep OFF by default; sweep-off renders must record their fixed damp). |
| ~~D6~~ | **CLOSED 2026-07-30 (re-vet R5).** Closed by *mechanism*, not enumeration: `pipeline.STAGE_SETTINGS_KEYS` declares each artifact's key set, `run-pipeline.sh` supplies values once, and the guard now covers every skip-if-exists artifact (16 call sites / 15 artifacts). `cost.pkl` additionally carries a `provenance["cost_mode"]` stamp so a mode change cannot silently reuse the other estimator's table (R2 precondition (i)). See §3.4. | §3.4; `pipeline.py` `STAGE_SETTINGS_KEYS` | ~~HIGH~~ | closed |
| D7 | **RESOLVED 2026-07-30 — and the original diagnosis was wrong.** The register previously read "`pyproject.toml` on `main` is `0.1.0` while PyPI serves `0.4.1` from a tag that is not an ancestor of `main`", implying the release had been cut off-trunk. It had not: `origin/main` *was* the release source all along (`v0.2.0` `4745887` → `v0.2.1` → `v0.3.x` → `v0.4.1` `d058267`, each an ancestor of `origin/main`), and the **local** `main` ref was simply 54 commits behind. Merging `origin/main` into this branch (`8f14400`) brings the whole release stack: `pyproject.toml:7` is `0.4.1`, `requires-python = ">=3.11"` (`:14`), plus the tag-driven PyPI pipeline, packaging gates and `docs/RELEASING.md`. `git merge-base --is-ancestor v0.4.1 HEAD` → true. Lesson: verify a divergence claim against the **remote** ref before filing it as debt. | `pyproject.toml:6-14`; `.github/workflows/release.yml`; `git merge-base --is-ancestor v0.4.1 HEAD` | — | Done. Follow-up: fast-forward the local `main` ref so the next reader's `git log main` is not 54 commits stale. |
| ~~D8~~ | **CLOSED 2026-07-30 (re-vet R24).** `_production_cache_prefetch_assignment` gained a `require` mode mirroring `production_weight_cache.prefetch_assignment(require=…)`, exposed as `--production-cache-prefetch {require,warn}`; `run-pipeline.sh` passes `require` on the native lane (matching `VALIDATED_SOURCE_PREFETCH=require`), and the CB/GGUF lanes read no production cache at all. A total miss is now a named failure instead of a silent NVMe-bound export. | `export_native_compressed._production_cache_prefetch_assignment` | ~~MED~~ | closed |
| ~~D9~~ | **CLOSED 2026-07-30 (re-vet R24).** The guard is at `main()` entry (not import time) in all seven — `incremental_probe`, `incremental_measure_quant_cost`, `aura_cost`, `production_render_cost`, `export_nvfp4_cb[_streaming]`, `export_gguf`, `select_validated_frontier` — verified against every CPU-only test import first, and a parametrized test pins all twelve callers so a refactor cannot drop one. | `gpu_guard.py` | ~~MED~~ | closed |
| ~~D10~~ | **CLOSED 2026-07-30 (re-vet R5).** `pipeline.py` now has one real job — settings-hash authority (§3.4) — and the bookkeeping is honest: the two owner names that existed nowhere in the tree are deleted, `streaming_model_weights` names `layer_streaming.LayerCache`, and a test asserts every approved owner has a class behind it. `QuantWeightCache` went to the archive wall with L3, so it is no longer an unmodelled holder. The *spec* half stays explicitly descriptive (§3.6); modelling the ten executed-but-unmodelled stages was refused as fiction-surface. | §3.6; `pipeline.py` | ~~MED~~ | closed |
| D11 | **MOSTLY FIXED 2026-07-30.** `model_profiles/validate.py`'s 8 conformance checks had zero callers and there were no workflow files in the tree. Both halves closed: `.github/workflows/ci.yml` (#18, `1cc7b90`) runs the suite on every push and PR (py3.11/3.12, CPU torch), and `tests/test_model_profile_conformance.py` drives the CPU-safe checks (1, 6, 8 + four structural invariants) over every registered profile, with 2/3/4 behind `integration` and 6/7 behind `slow`, and known gaps encoded as ratchets rather than bare xfails. **Residual (2026-07-30, R12): the check-5 half is now covered** — `test_has_mtp_implies_a_buildable_mtp_module` asserts `build_mtp_module` is a real override (and `mtp_source_prefix()` non-empty) whenever `has_mtp()`, which is the declarative part of the check that would catch L2/D2; check 5 proper still materialises a decoder layer and stays out of CI. Remaining: nothing invokes the validator as a `run-pipeline.sh` preflight for the actual `MODEL_PATH`. | `.github/workflows/ci.yml`; `tests/test_model_profile_conformance.py:9-31,223-249` | LOW (was MED) | Add a preflight invocation for `MODEL_PATH`. |
| ~~D12~~ | **CLOSED 2026-07-30 (re-vet R1).** `TARGET_DISK_GB` is plumbed through `run-pipeline.sh`: it overrides `TARGET_BITS`, narrows the Pareto sweep to the byte-feasible bracket, flips `SELECTION_MODE` to `validated-surrogate` and the frontier pick to `budget` = min measured KL among the rows that fit. Kneedle stays the default without a card and stays a diagnostic. See §4.6. | §4.6; `select_validated_frontier --mode budget` | ~~MED~~ | closed |
| D13 | **FIXED 2026-07-30 (R22 + R27).** The two hardcoded MiniMax arch tests now route through `profile.bypass_hf_fp8_module_rewrite()` and `profile.packed_expert_module_class_names()`; `specs/minimax_m2.json` exists and declares all eight of that profile's overrides; `deepseek_v4.json` declares `default_serving_profile: vllm_packed_moe`. Core-stack arch literals in control flow: **0**. Residual (not debt, sequencing): the MiniMax Python overrides stay until the equivalence gate `tests/test_minimax_m2_spec.py` has held for a release. | §8.4, §8.5 L4 | closed | — |
| ~~D14~~ | **CLOSED 2026-08-01.** Runtime documentation now lives with the sole canonical Gridbook package. PrismaQuant documents only its producer/export contract and points to the pinned package's machine-readable runtime contract; the former in-tree README was deleted with the vendored runtime. | external Gridbook `README.md`; `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` | ~~MED~~ | closed |
| ~~D15~~ | **CLOSED 2026-07-30 (wave 4, R28/R3).** The approved option was taken — **flip the defaults to the shipped values**, not defend the conservative one: `CB_SCALE_CODING=two_tier` (layout-v2 shipped in the Hy3 295B and Laguna-S-2.1 artifacts, and `STANDARDS.md` calls it the production fp4 scale coding with v1 legacy read-compat only — the old "serve gates pending, do NOT ship" comment predated its own ship; the knob is inert on fp8-CB-only menus, which is why two 27B/35B drivers set `v1` without contradicting it) and `CB_EXPERT_EMPIRICAL=0` (every shipped MoE CB driver sets it). Both are pinned by `tests/test_architecture_doc.py::test_cb_defaults_match_the_shipped_drivers`, which asserts the shell default against the drivers themselves so the two cannot drift apart again. No shipped run changes: every driver sets both explicitly. | `run-pipeline.sh`; `tests/test_architecture_doc.py` | ~~MED~~ | closed |
| D16 | **RESOLVED 2026-07-30 (R25) — as *unreachable*, not unmeasured, and the A/B was never needed.** The register asked for a gold-lane A/B on a 27B-class artifact; reading the emit-loop dispatch order answered it for free: the production-cache pack fires first and `continue`s, so with `PRODUCTION_CACHE=1` (the shipping default) **no dense NVFP4 Linear ever reached the branch** — confirmed by `grep -c "block-output-match"` → 0 on two real production export logs. Two further findings made keeping it indefensible: had it run, `_finalize_compute_only` would have re-derived per-group scales **outside** `_export_match_render_scale_rule`, discarding the render's `joint_mse` scales (the −6.6% KL defect M19 fixed everywhere else); and its `{0.95, 1.0, 1.05}` per-tensor gain re-search is subsumed by JSO, wrapped in `except Exception → WARN` so failure was invisible. Walled with `_finalize_compute_only` and the three export branches; `main()` now hard-`SystemExit`s if the flag is set truthy (§3.5). **Lesson: before funding a measurement, check the code under test executes on the recipe you ship.** | `archive/block_output_match_2026-07-30/README.md`; `export_native_compressed.py::_refuse_archived_block_output_match` | ~~MED~~ closed | — |
| D17 | **Registry and export metadata are unreconciled sources of truth** for bits/group per format — `FormatSpec` vs the `*_SCHEME` constants, with no test comparing them (§6.4, last row). | `format_registry.py:44-168`; `export_native_compressed.py:7247-7336` | MED | Add a parametrized test asserting scheme ↔ spec agreement per production format. |
| D18 | **PARTIALLY FIXED 2026-08-01.** The Gridbook-documentation half is closed: Gridbook owns and publishes its runtime flags, and PrismaQuant no longer mirrors that external catalog. The only remaining debt is producer-local cleanup: the dead `PRISMAQUANT_L2_CUDA_GRAPHS` and `PRISMAQUANT_DO_NO_HARM_MIN_GAIN` tokens remain in the historical/dead section even though the former's sole code occurrence is a comment at `perturbed_x_cache.py:1225` and the latter has no code occurrence. The live producer analogue remains `PRISMAQUANT_RENDER_GATE_MIN_GAIN`. | `docs/design/runtime_flags.md`; external pinned Gridbook runtime contract | LOW | Delete the two dead PrismaQuant entries once no reader is chasing them. |
| D19 | **FIXED 2026-07-30.** The count was low: **14** launchers under `examples/launchers/`, not 8, invoke `python -m prismaquant.<module>` for a module that no longer exists (`iterate_block_clado`, `measure_block_clado`, `block_clado`, `validate_block_clado`, `measure_output_fisher`, `dense_cone`, `polish_from_assignment`, `coord_descent_polish`, `measure_adjoint_l3`, `adjoint_l3_frontier`). Walled at `archive/launchers_2026-07-30/` with a banner README enumerating each file and its dead invocation, per the dated-wall convention of §11. | `archive/launchers_2026-07-30/README.md`; `examples/launchers/README.md` | — | Done. |
| D20 | **RESOLVED 2026-07-30.** Two archive walls had no banner README (`archive/prismaclip_2026-05-14/`, `archive/reap_2026-05-15/`) — the latter walls off live-adjacent code (`expert_prune.py`, `allocator_prune.py`, `observers/`, 5 tests) and encodes a policy the code still enforces. Two more walls violated the dated-directory convention. Banners written; `archive/entmoot/` → `archive/entmoot_2026-05-03/` (date from `193f313`) and `archive/minimax_m2p7/` → `archive/minimax_m2p7_2026-04-24/` (date from its own banner). Neither renamed wall is cited by a `run-pipeline.sh` `exit 2` message. | `ls archive/*/README.md` | — | Done. Follow-up: a test asserting every `archive/*/` carries a `README.md`. |
| D21 | **RESOLVED 2026-07-30 (R28).** Three ids appeared across the docs for one Hy3 artifact; the premise "at most one is live" was wrong — they are *renames*, not rivals, so the older ids **307-redirect** rather than 404. Canonical id (verified against the Hub 2026-07-30): **`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`** — the one to cite in all new material. The two `prod_hy3_results.md` citations are the dated ship ledger and were **annotated in place** ("now redirects to …"), not rewritten: a ledger records what was posted on the day. §9.2's unresolved paragraph now carries the resolution. | `docs/lanes/nvfp4-cb/prod_hy3_results.md:248-251,313-320`; §9.2 | ~~LOW~~ closed | Done. Follow-up: `scratch/gridbook-launch-post.md:24,179` still carries a third variant (`…-prismaquant-codebook-2.9bit-vllm`) — `scratch/` is out of the doc contract's scope, so it is left as-is; do not cite from it. |
| D23 | **bpp labels are not comparable across accounting eras.** The public "5.31" artifact's body bpp is ~4.76 under current accounting (§1.2); nothing in the tree records which era an artifact's label came from. | §1.2 | LOW | Stamp an accounting-era field into exported artifact metadata. |

New with the 2026-07-30 merge:

| # | Item | Evidence | Sev | Suggested action |
|---|---|---|---|---|
| D24 | **The KV-cotangent path has never touched a real KV-sharing checkpoint.** Its correctness is established by exact fp64 equivalence on a synthetic model (rel err 0.00e+00 vs one end-to-end autograd backward; the pre-fix protocol under-counts `k_proj` 85.1% / `v_proj` 38.5%) — a demonstration, not a measurement. No `num_kv_shared_layers > 0` model has been probed, so the magnitude of the correction on a shipping architecture is unknown, and the guard it replaced (`PRISMAQUANT_ALLOW_KV_SHARED_FISHER`) was the only thing previously stopping such a probe. | §7.5; `tests/test_kv_cotangent_path.py`; commit `b6ec9cb` | MED | Probe one real KV-sharing checkpoint (Gemma4-class) with the path on and off, and record the h_trace delta before any allocation claim rides on it. |
| D25 | **Gemma4-31B tied-embeddings result is enablement, not quality.** The first end-to-end probe → cost → allocate → export on a tied model (244 NVFP4 / 119 FP8 / 27 BF16 at achieved 6.000 bpp, 27.18 GB, `tie_word_embeddings` preserved and no duplicated `lm_head` bytes) ran at **2 samples × seqlen 512** to reach failures fast. The artifact has not been served and no KL/PPL exists for it. Nothing in §1.2 should cite it. | §7.5; commit `d058267` | MED | Re-run at production calibration and take it through the §7 gates before the family table gains a row. |
| D26 | **MEASUREMENT HALF CLOSED 2026-07-30 (wave 4, R16); the plumbing half is open.** The lane now has a KL evaluator: `prismaquant/gguf_kl_evaluator.py:measure_assignment_kl` wraps `llama-perplexity --kl-divergence-base` behind the `validate_assignments_kl` interface and returns `(mean, per_sequence, stats)` under the gold lane's key names — with the honest caveat that `per_sequence` is empty and `kl_tail_domain="aggregate"` (llama.cpp reports token-domain quantiles). Parsing is pinned against canned output in both shipped spellings; the live path is integration and unrun. **Still open:** `run-pipeline.sh`'s frontier loop is not wired to it (GGUF selection is still `surrogate`), and there is still no `PACKED_ROLE_SPLIT` plumbing, so every use of the split is a manual `allocator.py` invocation. | `prismaquant/gguf_kl_evaluator.py`; `prismaquant/lane_specs/gguf.json`; `grep -c PACKED_ROLE_SPLIT prismaquant/run-pipeline.sh` → 0 | LOW | Wire the frontier loop to the adapter, and plumb `PACKED_ROLE_SPLIT`. |
| D27 | **CLOSED 2026-08-01.** The version skew was not benign enough to preserve: the entire vendored Gridbook package, mirror script, and sync test were deleted. Gridbook now has one source tree and one version. PrismaQuant consumes one full-commit pin, verifies installed PEP 610 provenance and package version in CI, and fingerprints the resolved runtime for every serve. | `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; `prismaquant/gridbook_runtime/gridbook_runtime.sh`; `tests/test_gridbook_runtime_boundary.py`; `tests/test_gridbook_runtime_contract.py` | ~~LOW~~ closed | — |
| D28 | **Serve-time fast-kernel enforcement has no caller.** `require_fast_kernels(model)` — which reads the model profile's kernel requirements and hard-fails at startup when a required fast kernel (`causal-conv1d`, `flash-linear-attention`, …) is not importable — lost its only caller when `polish_from_assignment` was archived on **2026-05-15**, and was itself walled 2026-07-30 (R19) as an orphan. It is the only mechanized piece of **core principle 9's** "routed to a *performant* kernel (not a slow fallback)" gate, so that gate is **manual today**: nothing in the build or serve path refuses a checkpoint whose arch would silently fall back to the slow PyTorch implementation. The mechanism is written and tested — only the call site is missing. | `archive/orphans_2026-07-30/prismaquant/_fast_kernel_guard.py` + `tests/test_fast_kernel_guard.py`; sole historical caller `archive/polish_2026-05-15/prismaquant/polish_from_assignment.py:202` | LOW | Move the guard back and call it from `validate_native_export` / the serve launcher, keyed on the resolved profile — or, if serve-time enforcement belongs to the lane scripts, say so in §7 and delete the row. |

**Open items carried from session handovers.** Of the 41 items the handover census could not
map to a verified closure, the prior FP4-CB fast-expander/Triton item is now closed by the
exact pinned Gridbook 0.7.0 runtime: FP4-v2 prepares its native expander at model load, decode
uses native CUDA GEMV, M>8 uses native BF16 expansion plus Gridbook's owned CUTLASS grouped
bridge, and a missing operation fails closed. The remaining re-verified items are folded in
above: tail-veto (D1), `TARGET_DISK_GB` (D12), the DSv4 CB lane (D3), and the shipped
Mistral-Medium-3.5-128B artifact with no profile or spec (§8.4). Two are standing
research questions rather than debt: deriving the GPTQ damp constant from the weights, and the
XLAYER Q4 LFM2.5 routing-channel measurement. The remaining ~34 — mostly PrismaSCOUT-era items
that died with their subsystem — are enumerated with verdicts in
`scratch/doc-consolidation-2026-07-30/census_handovers.md` §POSSIBLY-STILL-OPEN.
