# DeepSeek-V4-Flash — CB-lane readiness audit

**Date:** 2026-07-17 · **Branch:** `claude/nvfp4-cb` · **Mode:** read-only audit
(no GPU, no downloads, no edits to mid-change files). Every claim cites
`file:line` against the tree at HEAD (`a1d11a0`). Skeptical by mandate: this
records gaps at full strength; nothing here is softened.

> **Size correction (2026-08-02, addendum — findings below left as recorded).**
> This audit's working assumptions of "~295 GB, 671B-class" predate local
> inspection of the released checkpoint. The actual DeepSeek-V4-Flash-0731
> release measures 166.9 GB on disk with ~285 B total parameters —
> 281,263,734,784 quantizable across 33,325 probeable Linears (probe
> inventory, 16x512) — and serves TP=1 on a single GB10. 671 B is the
> DeepSeek-V3-family headline and describes a different model. The scale
> reasoning in Q2 below is therefore conservative by ~2.4x on parameter
> count; its streaming conclusions hold a fortiori.

## Verdict (top line)

**A DSv4-Flash CB run is NOT attemptable today, and the blocker is not one
thing — it is the entire fp8-source ingestion + streaming + MoE-serve chain.**
The 0.6B/4B/27B-dense evidence does not transfer: those are bf16-source, dense,
single-GPU, TP=1. DSv4-Flash is fp8-native (~295 GB, block-scaled), 671B-class
MoE (256 routed experts/layer stored per-expert), and its own serving recipe is
`cluster_only: true`, `tensor_parallel: 2`, MTP spec-decode on
(`recipes/deepseek-v4-flash.yaml`). The CB *cost* path can plausibly consume it
(it streams and inherits the CT fp8 dequant); the CB **exporter** and **plugin**
cannot, by construction. 9 gaps below, dependency-ordered. Realistic critical
path before a *first probe→cost→export* is even runnable: gaps 1–4 (~450–700
LoC). Before it can *serve*: gaps 5–7 (the MoE plugin + TP, the hard ones).

---

## Q1 — FP8-source ingestion

**The CB exporter reads a whole bf16 skeleton and has no dequant. DSv4 has no
bf16 source.**

- `export_nvfp4_cb._load_skeleton` (`export_nvfp4_cb.py:78-93`) loads *every*
  shard and `tensors.update(load_file(...))` them into **one in-memory dict** —
  a full-model materialization (see Q2 for the scale consequence).
- It then feeds each `.weight` straight to `_vecs_and_wq` →
  `cb._scale_and_vectorize(w2d, grid)` with `w.reshape(...).to(torch.float32)`
  (`export_nvfp4_cb.py:96-111`). For an fp8-native source the tensor is
  `float8_e4m3fn` **codes**; `.to(float32)` casts the raw codes with **no block
  scale applied** → garbage reconstruction. There is no dequant anywhere in the
  exporter.
- The **precedent that solves this lives in two places the CB exporter does not
  use:**
  - GGUF lane: `export_gguf_direct._ShardIndex.dequant` (`export_gguf_direct.py:120-134`)
    applies an fp8 scale — but **only a per-tensor scalar** (`scale.numel() != 1`
    → raises, `:126-133`). DSv4 uses **per-block** `.scale` siblings, so this
    precedent is necessary but **insufficient** as-is.
  - CT lane: `layer_streaming._build_fp8_scale_inv_map` (`layer_streaming.py:253-322`)
    dispatches through `profile.fp8_scale_pairs(model_path)` (`:280`) and
    `_dequant_fp8_block_weight` (`layer_streaming.py:345+`) applies a rectangular
    block scale. **This is the real fp8-block-dequant engine**, and DSv4 is
    already wired into it: `deepseek_v4.fp8_scale_pairs` (`deepseek_v4.py:200-234`)
    builds `{live_weight_qname: (shard, .scale key)}` from DSv4's `.scale`
    siblings. So the *cost/probe* streaming path can dequant DSv4; the *exporter*
    is blind to all of it.
- **Verify, do not assume:** `deepseek_v4.py:200` calls the scale dtype
  `F8_E8M0`. `_dequant_fp8_block_weight` assumes a castable block scale and
  reads the grid from `quantization_config.weight_block_size`
  (`layer_streaming.py:188-233`). Whether DSv4's grid is 128×128 (MiniMax
  convention) or MX-style 1×32 with an E8M0 exponent scale is **unconfirmed on a
  real shard** (no DSv4 config or shard is on disk — see source note). This must
  be validated before trusting the dequant; `FP8_SOURCE`'s own spec hardcodes
  `scale_block_shape=(128,128)` (`format_registry.py:788-800`), which is *not*
  guaranteed to match DSv4.

**Missing:** a streaming, fp8-block-dequant-aware weight source for the CB
exporter (and its provider hooked to `fp8_scale_pairs` + `_dequant_fp8_block_weight`).

## Q2 — Scale (671B through 128 GB)

- **Probe:** streams (per-layer, on-demand); designed for this class. **Not
  validated on DSv4** — no probe artifact exists on disk
  (`find … -iname probe.pkl -path '*dsv4*'` empty).
- **Cost:** `incremental_measure_quant_cost` "streamed shard-by-shard on top of
  `layer_streaming`" (`incremental_measure_quant_cost.py:3-11`); the CB branch
  is `_CB_COST_FAMILIES` in `_batched_quantize` (`measure_quant_cost.py:1198,1226`),
  which sees a **bf16 tensor after** `layer_streaming` has dequant'd it. So the
  CB cost path *inherits* the DSv4 fp8 handling for free — the one stage that is
  structurally ready. (Still unexercised on DSv4.)
- **Export: this is where it breaks at scale.** `_load_skeleton`
  (`export_nvfp4_cb.py:78-93`) holds the entire model resident. 295 GB into
  ~121 GB usable unified memory = **immediate OOM**, before any dequant question.
  Contrast `export_gguf_direct`: `_ShardIndex` opens shards lazily and reads one
  tensor at a time (`export_gguf_direct.py:102-118`); Pass-1 writes only tensor
  metadata, Pass-2 streams tensor-by-tensor with `del w, data`
  (`export_gguf_direct.py:240-352`) so "neither the source nor the artifact is
  ever memory-resident" (`:16-18`). The CB exporter must be refactored to that
  streaming shape.
- **Second scale break — per-expert stacking.** DSv4 stores 256 experts
  **separately** per layer (`layers.N.ffn.experts.{0..255}.{w1,w3,w2}`,
  `deepseek_v4.py:20-21,154-158`). The CB exporter matches assignment keys by
  `qname + ".weight"` **against the raw skeleton** (`export_nvfp4_cb.py:250-274,
  354-356`) and has **no** checkpoint→live remap and **no** per-expert→packed
  stacking. `detect_profile` is imported but used only for NVFP4 fused-sibling
  global-scale sharing (`export_nvfp4_cb.py:287`), not naming. The allocator
  keys stacked experts by the **live packed** name
  (`…experts.gate_up_proj/down_proj`); those names are **absent** from DSv4's raw
  checkpoint → the coverage gate raises "no weight tensor" (`:252-255`). GGUF
  solves both with `_plan_tensors`/`_EXPERT_RE` stacking
  (`export_gguf_direct.py:137-170`) + `profile.checkpoint_to_live_name`
  (`:200-207`). The CB exporter has neither.

## Q3 — Profile / pipeline

- `run-pipeline.sh` **does** have a real `EXPORT_CONTAINER=nvfp4_cb` lane
  (`run-pipeline.sh:119-129, 1395-1489`) with the right gates
  (`COST_MODE=local`, `TARGET_PROFILE=nvfp4_cb`, `PRODUCTION_CACHE=0`) and a
  col-weights harvest from the probe act cache (`:1427-1442`). But it passes
  `--model-dir "$MODEL_PATH"` (`:1450`) — i.e. hands the fp8 295 GB source
  straight into the broken `_load_skeleton`. The lane is dense/bf16-shaped
  end-to-end.
- `DeepseekV4Profile` (`deepseek_v4.py`) is well-built for **probe** (name bridge
  `:100-198`, `fp8_scale_pairs` `:200-234`, hyper-connection stream expand/collapse
  `:280-293`, rotary `:246-278`, per-expert probe swap
  `vendored/dsv4_probe_experts.py`) but `vllm_architecture_class()` returns
  **None** (`:63-70`, "until vllm-fresh-b12x is rebuilt"), so fused-sibling /
  packed-MoE autoderivation is off — the CB serving-unit promotion the MoE
  design relies on (`moe_cb_design.md §1`) has no vLLM class to derive from.
- **No serving profile spec for DSv4** (`serving_profile_specs/` has no
  deepseek entry). The generic `nvfp4_cb.json` profile is model-agnostic and
  its shape rule (`in_features % 256`) applies, so an allocation *can* be gated,
  but there is no DSv4-specific serving-constraint spec.
- `pipeline.py` has **no** `nvfp4_cb` or `deepseek_v4` contract entries (grep
  empty); the CB lane `exit 0`s (`run-pipeline.sh:1488`) before the declarative
  contract runs, so this is a documentation/validation gap, not a runtime block.

## Q4 — Menu design for ultra-low-bpp

At ~2.5–3 bpp on a 671B, the load-bearing rungs are fp4-CB v2 (two-tier) for the
bulk, FP8_CB for the sensitive tail, and **FP8_SOURCE passthrough** where an
fp8-native layer should ship verbatim. Registry/legality reality:

- **fp4-CB v2 two-tier** (the sub-3bpp win, `exp1c_v2_premium.md`: K18-v2 beats
  IQ2_S at fewer bytes): exporter supports it (`--scale-coding=two_tier`,
  `export_nvfp4_cb.py:450-459, 498-501`) **and the plugin now decodes it**
  (`expand.py:105-188 _cb_expand_weight_v2_kernel` + `expand_fp4_v2_to_weight`;
  `linear.py:58-69` v2 dispatch). This rung is the most ready.
- **FP8_CB** (`FP8_CB_K36/40/44/48`): fully in the profile allow-list
  (`nvfp4_cb.json`) and exporter. Escapes the scale-packaging tax (per-channel
  fp32 scale). Ready.
- **FP8_SOURCE — a real hole for an fp8-native model.** It is a registered
  format (`format_registry.py:789`, a verbatim `w.clone()` passthrough) and the
  natural rung for DSv4's most sensitive layers (the source is *already* fp8, so
  passthrough is lossless at ~8.002 bpp). But:
  1. **Not in the `nvfp4_cb` profile allow-list** (`nvfp4_cb.json` lists only
     NVFP4 / FP8_DYNAMIC / BF16 + CB rungs) → under `TARGET_PROFILE=nvfp4_cb`
     the legality gate never lets the allocator **propose** it.
  2. **The CB exporter cannot carry it.** `_STOCK_CT_SCHEMES = {"NVFP4",
     "FP8_E4M3"}` (`export_nvfp4_cb.py:224`); the coverage gate raises on any
     other non-CB/non-BF16 format (`:232-248`). FP8_SOURCE would hard-fail.
  3. Even the **stock** NVFP4/FP8_DYNAMIC rungs in the CB exporter re-quantize
     via `_ct_quantize_2d(tensor.to(device), …)` (`export_nvfp4_cb.py:387-388`),
     which assumes a bf16 input — so on an fp8 source they break for the same
     root cause as Q1. The whole mixed container is bf16-input-shaped.
- **Expected composition** (extrapolating the 0.6B full-menu result,
  `PLAN.md:534-568`, all-CB at 3.0/3.5 bpp): NVFP4_CB_K16-v2 bulk + FP8_CB tail
  + a few BF16/FP8_SOURCE spends on the most sensitive Linears (attn out,
  router-adjacent, shared expert). But without FP8_SOURCE in the menu, the only
  high-fidelity spend is BF16 at 16 bpp — wasteful on a model whose source is
  8-bpp fp8 (violates the "never synthesize BF16 from dequant'd fp8 — *shame*"
  rule if a dequant'd fp8 layer is re-upcast to BF16). This is a design gap, not
  just a plumbing one.

## Q5 — Serving

- **The plugin is dense-only. DSv4 is MoE. This is the hardest gap.**
  `PrismaQuantConfig.get_quant_method` handles only `LinearBase` and
  `VocabParallelEmbedding` (`config.py:180-206`); there is **no `FusedMoE`
  branch** anywhere in the package (grep for `FusedMoE`/`experts` outside tests
  is empty). A CB MoE artifact exports (once Q1–2 land) but **cannot serve** —
  the precise missing contract is `moe_cb_design.md §4` (a FusedMoE-analog method
  loading `cb_qweight (E, out, bytes)` into w13/w2, splitting the fused
  gate_up stack by canonical scheme names, staying INV-1/CUDA-graph clean).
  Even plain-NVFP4 experts would not serve: the plugin never routes a FusedMoE
  module to the delegated CT config.
- **Tensor parallelism — unproven for byte-packed CB weights, and DSv4 forces
  it.** `recipes/deepseek-v4-flash.yaml` is `cluster_only: true`,
  `tensor_parallel: 2`. The plugin's `create_weights` takes
  `input_size_per_partition` / `output_partition_sizes` (`linear.py:72-83`) so it
  is TP-*aware* at the API, but `cb_qweight` is a byte-packed 256-weight
  superblock stream **along in_features** (LAYOUT.md §1). A RowParallel split
  (down_proj / o_proj) cuts in_features; unless every shard boundary lands on a
  256-superblock (and 8-wide codeword) boundary the byte layout is invalid.
  This has **never been tested** (all CB serving was TP=1, single 0.6B/4B). High
  risk; needs a sharding-legality proof or a per-shard-256 export constraint.
- **Image / arch mismatch.** CB serving was validated in `vllm-fresh-b12x`;
  DSv4 serving needs the `vllm-node` image with a custom
  `--tokenizer-mode/--reasoning-parser/--tool-call-parser deepseek_v4` and vLLM's
  native DSv4 arch (`recipes/deepseek-v4-flash.yaml`). The out-of-tree plugin has
  never been installed into `vllm-node` alongside DSv4 support.
- **MTP / spec-decode.** DSv4 has 1 nextn-predict block (`deepseek_v4.py:72-91`,
  `has_mtp=True`, `build_mtp_module` returns None = **deferred/unbuilt**), and the
  recipe serves with `--speculative-config method=mtp` on. Per the standing
  landmine, spec-decode **poisons the PPL gate** — the gold-metric served
  KL/PPL A/B must run on a **no-spec-decode** serve, and `validate_quantized_model`
  refuses a verdict if it sees `vllm:spec_decode_*`. The MTP head itself needs a
  format decision (CB or passthrough) and is currently unwired.

---

## Numbered gap list (owner-file · est. LoC · blocks)

| # | Gap | Owner-file | LoC | Blocks |
|---|-----|-----------|-----|--------|
| 1 | **Streaming skeleton for the CB exporter** — replace `_load_skeleton`'s whole-model `load_file` with a lazy `_ShardIndex`-style reader (Pass-1 metadata, Pass-2 tensor-by-tensor with `del`). | `export_nvfp4_cb.py` (mid-change; **not this agent**) | ~120–180 | 2,3 |
| 2 | **fp8-block dequant-on-read** — provide bf16 weights to the VQ search via `profile.fp8_scale_pairs` + `layer_streaming._dequant_fp8_block_weight`; handle DSv4 `.scale`/E8M0 grid. Port from `layer_streaming.py:253-395`. | `export_nvfp4_cb.py` + a shared helper | ~60–100 | 3 |
| 3 | **checkpoint→live remap + per-expert→packed stacking** in the exporter (DSv4's 256 per-expert `w1/w3/w2` → live `experts.gate_up_proj/down_proj`). Port `_plan_tensors`/`_EXPERT_RE` + `checkpoint_to_live_name`. | `export_nvfp4_cb.py` | ~120–200 | serve |
| 4 | **Verify DSv4 fp8 scale layout on a real shard** (E8M0 vs fp32, block grid vs MX 1×32) before trusting dequant; add a shape/dtype assertion. Requires the source on disk (source note). | audit/test | ~20–40 | 1,2 |
| 5 | **Gridbook FusedMoE CB method** — the `moe_cb_design.md §4` contract: load `cb_qweight (E,out,bytes)`, split fused gate_up by canonical names, INV-1/CUDA-graph clean. Runtime code belongs only in external Gridbook. | external Gridbook runtime | ~250–400 | serve |
| 6 | **TP-sharded CB byte-stream correctness** — prove/enforce 256-superblock-aligned RowParallel splits, or export a per-shard-legal layout; validate at TP=2. | external Gridbook runtime + PrismaQuant export | ~80–150 | serve |
| 7 | **Plugin in `vllm-node` + DSv4 arch + no-spec-decode PPL serve** — install path, custom-parser coexistence, MTP-off gold-metric harness. | serving env + harness | ~60–120 | serve |
| 8 | **FP8_SOURCE as a first-class CB-container rung** — add to `nvfp4_cb.json` allow-list, teach the CB exporter to copy fp8+scale verbatim (a 4th `_STOCK`-class path), teach the plugin to delegate/serve it. | `nvfp4_cb.json` + `export_nvfp4_cb.py` + plugin | ~80–140 | menu quality |
| 9 | **DSv4 expert empirical cost under the CB lane** — route-flip-blind smooth cost needs `EXPERT_EMPIRICAL` (`moe_cb_design.md §3`, ~90–130 LoC there) extended to 256-expert/layer scale + zero-expert-calibration guard (Ornith lesson). | `expert_empirical_cost.py` + lane gate | ~90–150 | alloc quality |

**LoC total (rough): 900–1500**, dominated by gaps 3, 5, 6.

## Dependency-ordered plan (what must land before a DSv4 CB run is attemptable)

1. **Source on disk + shape verification (gap 4)** — cannot design the dequant
   or stacking without confirming DSv4's real scale/name layout. Gated on the
   download (source note).
2. **Streaming + fp8 dequant + expert stacking exporter (gaps 1→2→3)** — this
   trio makes probe→cost→**export** runnable at all. Until it lands the exporter
   OOMs on load and mis-keys every expert. *This is the minimum for a first
   emulation-gate (offline KL) DSv4 CB artifact.*
3. **Expert empirical cost (gap 9)** — needed for a *good* allocation (smooth
   CB cost is route-flip-blind; `moe_cb_design.md §2`), but a first *smoke* run
   can use `COST_MODE=local` without it.
4. **Plugin MoE method + TP + image (gaps 5→6→7)** — the serve half. None of the
   served gold-metric evidence exists until all three land; TP=2 correctness for
   byte-packed weights is the single highest-risk item.
5. **FP8_SOURCE rung (gap 8)** — quality lever, parallel to the above; not a
   blocker for a first run but load-bearing for the ultra-low-bpp menu on an
   fp8-native model.

**Attemptable-run milestones:**
- *First offline (emulation-gate) DSv4 CB export:* gaps 1–4 (+9 optional).
- *First served DSv4 CB verdict:* additionally gaps 5–7.

## Source-acquisition note

- **Source is NOT on disk.** No DSv4 safetensors/config found under
  `/home/rob` or `/models`; only two stale April download *logs* and the serving
  recipe. It was cleaned in the disk sweeps. **~295 GB fp8-native** re-download
  required (`deepseek-ai/DeepSeek-V4-Flash-Base`, the fp8 source per house
  memory; the recipe points at the non-Base instruct variant — confirm which
  before pulling).
- **Disk today: 547 GB free of 1.8 TB (69% used).** A **Hy3 re-download is in
  flight** to `/home/rob/dq-runs/hy3-prod/source` (`snapshot_download('tencent/Hy3',
  max_workers=8)`, live pid confirmed; ~557 GB target). Hy3 **takes priority**
  (per tasking). 547 − 295 ≈ 252 GB would remain *if DSv4 landed alone*, but with
  Hy3's ~557 GB also inbound the two together breach the ≥10% (~180 GB) headroom
  rule. **Do not co-download.** Sequence: let Hy3 finish (and its intermediate
  bulk be consumed/cleared) *before* pulling DSv4, or the box wedges on disk.
- Even once downloaded, gaps 1–4 must land before the source is anything but
  dead weight — do not pull it ahead of exporter readiness.

---

*Audit method: read-only against HEAD `a1d11a0`; no GPU, no downloads, no edits
to mid-change files (`nvfp4_cb_formats.py`, `export_nvfp4_cb.py`, `plugins/`,
`measure_quant_cost.py`). Findings are code-anchored; where the code could not be
exercised (no DSv4 shard on disk) the claim is marked "verify".*
