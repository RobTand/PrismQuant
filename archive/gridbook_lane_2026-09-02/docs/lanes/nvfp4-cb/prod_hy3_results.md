# Hy3 295B ultra-low-bpp — NVFP4-CB lane (running log)

> Historical run log. Resume observations below describe the July 2026
> implementation and are not a current capability: as of 2026-07-31 the
> producer rejects unbound resume/delta reuse and requires a fresh output.

**Model:** tencent Hy3 295B-A21B (hy_v3, 80 layers, 192 experts top-8,
router `mlp.router.gate.weight` + `mlp.expert_bias`), BF16 source 557 GB
(non-preview release, verified). **NO QUALITY CLAIMS** (standing rule: a
295B cannot be KL-validated against its BF16 teacher on this box).
Validation = loads + coherent generation + bit-exact packing + speed vs
the shipped GGUF Hy3 2.8bpp (prefill 42 tok/s = the IQ tax the CB lane
exists to remove).

**Driver:** `scripts/run_hy3_prod_nvfp4cb.sh` — COST_MODE=local,
CB_EXPERT_EMPIRICAL=0 (+sample 16, ladder interp), two_tier (v2) coding,
TARGET_BITS=2.9, streaming export.

## Chain (all stages GPU, one box)
- Probe: 10×8-layer shards + tail (~1.7 h after LAYERS_PER_SHARD=8; auto
  had picked 1/shard ≈ 16 h).
- Cost: ~3 h, both rung-family floor-law ladders live.
- Allocation @2.9 (achieved 2.902): mixed-family body — experts
  fp4-CB (K16×36, K18×38, K20×30, K14×18) + fp8-CB K28×36; dense/attn
  fp8-CB (K32×327, K44×57, K36×33, K28×26) + fp4 tail; BF16 floor 57.
- Streaming export: ~105 GB expected, resumable (see ledger 5).

## First-contact bug ledger (2026-07-19, all committed)
1. **Streaming export box-OOM ×3** (global kernel OOM, exporter CPU RSS ~0,
   torch alloc 0 — invisible GPU-side): root cause
   `_pack_codes_to_bytes` materialized the (rows, nvec, k) int64 bitstream
   twice = **~155 GB** on a (192,3072,4096) expert stack. Fix: row-chunked
   bit-pack, bit-identity verified k=12..48 + roundtrip (`b4a8b3f`).
   Synthetic E=192 pack peak 173 → 27 GB. Diagnosed OFFLINE at 1/32 scale
   under `set_per_process_memory_fraction` after the live-relaunch pattern
   burned two extra runs — repro-before-relaunch is the rule.
2. Full broadcast col_weights copy (~10 GB/stack) → per-block gather,
   identity-verified (same commit).
3. Exporter now self-caps at 75% of the unified pool: future runaways
   raise a clean torch OOM naming the tensor, never a box-wide kill.
4. Unbounded open shard mmaps (233-shard source → ~1 TB VM) → LRU-4
   (`06e653a`). Not the OOM cause, but hygiene worth keeping.
5. **Resume support** in the streaming writer (`be300dd`): analytic
   offsets + deterministic producers → partial file resumes at the first
   incomplete entry (header must match bit-for-bit; sibling state groups
   backed up together). Byte-identity tested at 5 cut points. Salvaged the
   12 GB partial (resumed 763/2271).
6. **v2 encode pace**: the W×16 two-tier entry loop was launch-bound
   (33.6 s of 46.8 s at E=24 in 16,128×2 ms kernels) → batched via
   `_moment_err_groups_batched`, **2.9×, bit-identical across 32 configs**
   (torch.min first-occurrence == strict-<-first-legal). Expert stack
   ~15 → ~3 min; export ETA ~30 h → ~7-10 h (same commit).
7. Ops: export runs detached (`setsid`) — two of the four kills were
   session-side SIGKILLs, not the box. Monitor deadman keys on ARTIFACT
   mtime, not log mtime (20-entry log cadence stretches >30 min across
   expert stacks); pgrep patterns bracket-escaped (`pro[d]`) against
   self-match.

## Footprint (exact, read from the pre-written streaming header mid-export)
- **model.safetensors = 110.3 GB** (102.7 GiB): U8 cb_qweight 99.68 GB +
  BF16 sidecars 10.52 GB (57 BF16 layers) + F32 scales 0.10 GB. 2271
  tensors, no double-ship. Codebook sidecar (lattice, shared per role)
  negligible. This is **above the ~105 GB estimate** — 2.902 bpp over
  quantizable params understates the on-disk total because the BF16 floor
  is bpp-excluded but disk-resident.
- **Single-Spark fit: YES, context-capped.** 110.3 GB weights + ~2 GB
  framework on the ~121 GB usable pool. CB decode is transient (no load
  expansion), so resident weight footprint == disk. KV headroom (GQA 8
  kv-heads × 128 × 80 layers × fp16 = 0.328 MB/token):
  - 8k ctx: 113.0 GB resident (~8 GB spare) — comfortable
  - 16k ctx: 115.7 GB resident (~5 GB spare) — OK
  - 32k ctx: 121.0 GB — at the ceiling, no room for activations
  Native 262k context does NOT fit (true of any ~110 GB weight class on
  this box). **Serve with --max-model-len 8192–16384.** If practice shows
  the headroom too thin, 2.7 bpp is the PARETO_TARGETS fallback rung.

## Serve first-contact (2026-07-20)
- Artifact integrity: safetensors COMPLETE (110.3 GB, header end ==
  file size), config HYV3ForCausalLM + layout_version 2 + two_tier,
  cb_codebooks.pqcb loads via safetensors (24 shared lattice codebooks,
  sub-structure correct per rung). Codebooks are NOT torch.load/pickle —
  they are a safetensors file with a .pqcb extension (probe trap).
- hy_v3 serving arch: HYV3ForCausalLM + HYV3MTPModel are NATIVE in
  `vllm-node:latest` (vLLM 0.23.1-dev, tf 5.13) — the run-script's
  pre-launch gate (d) wrongly credited the GGUF work (that was
  llama.cpp, no vLLM adapter); the image covers it regardless.
- **Bug 9 (serving loader, hy_v3): stacked CB expert tensors KeyError at
  load.** `HYV3ForCausalLM.load_weights` loads experts at the TOP-LEVEL
  model via `expert_params_mapping` (per-expert names) and never calls the
  per-layer `FusedMoE.load_weights` the plugin wraps — so that wrap is dead
  code here (it works for Qwen3.5-MoE/35B, which delegates per-layer). Our
  stacked `experts.{gate_up_proj,down_proj}.cb_qweight` match no per-expert
  mapping → final `params_dict[name]` KeyError vs registered
  `experts.w13/w2_cb_qweight`. Fix: plugin-installed model-level
  load_weights wrap mapping stacked CB expert names → fused params (plain
  copy), delegating everything else. This is the top-level-loader analog of
  moe.py's per-layer wrap; generalizes to DSv4 (same convention). Serve
  memory params for the smoke: --enforce-eager --max-model-len 8192
  --gpu-memory-utilization 0.95 (110 GB weights leave ~5 GB for KV).

## Serve-bringup bug chain (2026-07-20, first serve of a top-level-loader MoE)
Each is a distinct hy_v3-specific first-contact bug; the artifact bytes are
correct (proven on disk), these are all serving-adapter gaps.
- **Bug 9a — top-level expert loader (FIXED, committed f202841 then
  refined).** hy_v3 loads experts at the top model via expert_params_mapping,
  bypassing the per-layer FusedMoE.load_weights the plugin wraps. Plugin-
  installed model-level load_weights wrap. The wrap install runs in the
  EngineCore process (verified via marker).
- **Bug 9b — routed_experts nesting (FIXED).** hy_v3's SharedFusedMoE nests
  the routed FusedMoE one level deeper: params are at
  `…mlp.experts.routed_experts.w13/w2_cb_qweight`, not `…mlp.experts.w13…`.
  A fixed-string suffix rewrite missed. Fix: `resolve_cb_expert_param`
  resolves the target by (`…mlp.experts.` prefix, leaf suffix) against the
  ACTUAL named_parameters — robust to this nesting and any future one.
  Confirmed working: both routed_experts params load.
- **Bug 9c — shared expert built unquantized (FIX IN FLIGHT).** hy_v3 passes
  `shared_experts=self.shared_mlp` into FusedMoE; vLLM builds the shared-MLP
  Linears as PLAIN BF16 `.weight` and NEVER calls the quant config's
  get_quant_method for them (proven: instrumented print never fired). But the
  export quantized shared_mlp to CB (190 tensors, 0.58 GB packed) → KeyError.
  Fix: decode the CB shared_mlp tensors to bf16 at load and populate the
  `.weight` params (the plugin already decodes CB at prefill; ~+1.7 GB
  resident). Unfused checkpoint gate/up → cat into vLLM's fused
  gate_up_proj.weight. NOTE for the clean re-export path: a hy_v3 serving
  profile should force shared_mlp→BF16 (like Gemma incomplete-fused groups),
  making this decode-at-load unnecessary — but that needs a re-export; the
  decode serves the existing artifact now.

- **Bug 9d — the shared-expert NON-bug (investigated + reverted).** A
  defensive warning claimed apply() might drop the shared expert. A fix to
  compute it inside apply() CRASHED (vLLM's `shared_experts` is a
  `SharedExperts` stream-overlap wrapper: `forward(hidden_states, order)`,
  not a plain module). Reading vLLM v0.23 `moe_runner.py` settled it:
  `_apply_quant_method` runs the SharedExperts wrapper SEPARATELY
  (`_maybe_apply_shared_experts` → `SharedExperts._layer(input)` = the
  shared_mlp with our bf16-decoded weights) and `_maybe_combine` adds it;
  apply() returning routed-only IS the contract. Reverted; corrected the
  comment. The shared expert was computed correctly all along.

## SERVE VERDICT — 295B on ONE Spark (2026-07-20)
**Loads, serves, generates coherently and correctly.** Single DGX Spark,
vllm-node (vLLM 0.23), plugin, --enforce-eager --max-model-len 4096
--gpu-memory-utilization 0.95. Load 77 s (100.7 GiB), KV 44,272 tokens
(10.8× concurrency @4k). Validation bar (NO QUALITY CLAIMS on 295B) MET:
- Coherent + factually correct: Tokyo/Paris/Berlin chain; correct recursive
  fibonacci; **arithmetic correct** (17×24=408, 60mi/1.5h=40mph,
  60mi/2gal=30mpg); RGB primaries.
- **Speed — the thesis: prefill 89 tok/s vs the shipped GGUF Hy3 2.8bpp's
  42 tok/s IQ tax = 2.1× faster prefill on native tensor-core CB.** Decode
  ~9-10 tok/s (fp4-v2 grouped kernel + Triton dense fp4). This is the CB
  lane's raison d'être proven at 300B class: native-format serving removes
  IQ's prefill tax on a single Spark.

## Ship sprint (2026-07-20) — quality gate + native-decode arc
Goal: exceed the shipped GGUF Hy3 in quality, native perf in prefill AND decode.

**TEB quality gate (exact GGUF protocol: 12288 ctx, kv fp8, hy_v3 parsers,
seed 1234, --no-think --hardmode --parallel 1): 87/100 (129/148) — an exact
TIE with the shipped GGUF IQ 2.8bpp (87, 129/148); beats k-quant (86).**
Zero errors, 74/74 scenarios. Failures are the known cross-artifact family
fails (TC-34/42/43/60) + TC-58; scenario churn at the plateau, same as the
IQ-vs-k comparison. Honest verdict: at matched body bytes, tool-use quality
is base-model-dominated — PARITY, not exceed, on the TEB axis. The
exceeds-the-ship case = quality parity + the 7.5 GB BF16 MTP head the GGUF
cannot carry + native-kernel speed. Report:
tooleval/2026/07/2026-07-20T13-28-37Z_994c74.md.

**Decode time budget (torch profiler, eager, per step):** wall 158 ms =
GPU-busy 85.5 + host/launch gap 72.7 (4,400+ launches/step — norms, rotary,
MoE sort/gather). GPU-busy split: cb_moe_gemv_fp4_v2 37.6 ms at ~25-30% of
bandwidth (the wall), dense fp8 GEMV 20.0 @ ~55%, cuBLAS (lm_head + router)
10.9, MoE fp8 8.7, norms 6.2, attention 0.7.

**Perf levers landed (decode 9.9 → 13.1 tok/s so far, prefill 98 → 115):**
- **shared_mlp CB-direct (config-only fix).** vLLM builds hy_v3's shared MLP
  with the PARENT ``…mlp`` prefix (prefix=f"{prefix}" in HYV3MoEFused) — the
  real reason bug-9c's instrumented print never fired. Collapsed-dispatch
  aliases in PrismaQuantConfig let the CB linear method own the shared expert
  natively; the decode-at-load path goes dead automatically (structural
  detection); module paths / checkpoint names unchanged. −1.9 GiB resident
  (KV 13.6→14.4 GiB at eager), −1.8 GB active bytes/token (ceiling 24→28
  tok/s). Every quantized Linear now runs through CB kernels.
- **Dense fp4-v2 CUDA GEMV** (was Triton): 13/13 bit-match tests vs Triton +
  expand reference; covers 27 dense + 33 shared fp4 units.
- **CUDA graphs, FULL_DECODE_ONLY + mode NONE** (no torch.compile over the
  plugin): 2 decode graphs, 0.18 GiB, +24% decode. Flushed out a REAL bug:
  codec.fp4_group16_act_qdq built its E2M1 grid on CPU and H2D-copied it
  EVERY call (hidden sync in eager, hard error under capture) — now cached
  per-device. Prefill stays eager (the per-expert loop's one host sync).
- **MTP spec decode: REJECTED on one Spark (box OOM 2026-07-20).** The bf16
  draft (+7.5 GB) on top of ~102 GiB weights leaves no margin for vLLM's
  memory-profiling peak on the unified pool (~5 GiB system floor). Do NOT
  retry at these sizes. v2 roadmap: quantize the MTP module to CB (~2 GB) —
  then spec decode fits. The artifact still carries MTP for bigger boxes.
- **fp4-v2 grouped MoE kernel pass: HONEST NEGATIVE on the bandwidth
  premise.** Bit-identical 1.06–1.19× per (proj, rung) landed (160/160
  outputs EXACT vs the pre-edit kernel, 58 suite tests): codebook gather
  8→2 aligned u64 loads, wide 8-aligned staging with off8 carry,
  predicated third extraction word. But ncu shows the kernel is
  COMPUTE/LATENCY-bound (SM 71% vs Mem 44%, L1TEX scoreboard stalls) —
  the fp4 unpack + two-tier compose + per-weight bf16-round chain is
  frozen by the bit-identity contract, so ≥55% of bandwidth is not
  reachable without changing the decode numerics. Rejected variants
  (decode+FMA fusion, packed bf16x2 — bit-identical but slower) recorded
  in /home/rob/dq-runs/cb-bench-scratch/. Base-decode ceiling at these
  rungs ≈ 13–14 tok/s.

## MTP ships (2026-07-20, Robert: "always include MTP when it's available")
- **CB-quantized MTP module**: layers.80 encoded to the body's modal rungs
  (experts NVFP4_CB_K18, shared+attn FP8_CB_K32, glue BF16; uniform
  col-weights — a draft cannot change outputs, only acceptance), 1.19 GB vs
  7.5 GB BF16 (which OOM'd the box next to 102 GiB of weights — do NOT
  retry bf16-MTP on one Spark). Same lattice codebooks (bit-compared at
  merge). Exporter gained opt-in --subset-prefix; the toplevel loader
  gained spec-layer rename support (HYV3MTP renames layers.80.* →
  .mtp_block.*; param targets resolve on renamed names, schemes on
  original names).
- **Ship artifact = ship_mtp/**: 104.06 GB, 5 HF shards + index (dropped
  7.44 GB bf16 MTP, added 1.19 GB CB MTP; net −6.2 GB vs the single-file
  body artifact).
- **Serving findings (all measured):**
  - vLLM auto-wires hy_v3 → hy_v3_mtp draft from num_nextn_predict_layers.
  - **Spec decode + CUDA graphs needs TRITON_ATTN on BOTH models**
    (--attention-backend + attention_backend inside --speculative-config):
    FlashInfer is UNIFORM_SINGLE_TOKEN_DECODE-only → vLLM silently sets
    cudagraph_mode=NONE and the whole serve runs eager (spec decode then
    LOSES: 9-10 tok/s vs 13.1 base).
  - **Acceptance at the K18 draft rung: 78–93% per-position on natural
    text** (~55% on random-word benches). Draft fidelity is NOT the
    binding constraint.
  - **k-sweep (prose tok/s): k=1 13.1 · k=2 10.7 · k=3 8.8** — the drafter
    runs UNCAPTURED (no drafter cudagraph support in vLLM 0.23 for this
    method), costing ~50 ms/draft-token of eager host overhead that scales
    with k. **k=1 is the throughput optimum today**: neutral on prose
    (13.1 = 13.1 vs spec-off), positive on structured/code, negative on
    adversarial text. The MTP throughput question ("how much to quantize
    the draft") is answered structurally: d is dominated by the SHARED
    bf16 lm_head + eager overhead, so the block's rung barely moves speed —
    acceptance is king, quantize only as memory forces, and the rung sweep
    only matters once vLLM captures drafter graphs (upstream).

## FINAL SPEED (ship config: graphs FULL_DECODE_ONLY, TRITON_ATTN, spec k=1)
- **Prefill 109–115 tok/s** (1.46k-token prompt) — **2.6× GGUF IQ's 42**.
- **Decode (batch 1, prose) 13.1 tok/s** vs GGUF IQ 17.8 / k-quant 18.7 —
  base decode TRAILS the GGUF CUDA-core MMVQ path at these bytes; bounded
  by (a) the compute-bound fp4-CB decode chain (bit-identity-frozen) and
  (b) the eager drafter. Decode arc this session: 9.9 → 13.1 (+32%):
  shared-CB-direct + dense fp4-v2 CUDA GEMV + CUDA graphs + kernel pass.
- Weights resident ~100 GiB + CB MTP; KV 12.7 GiB at 12288/fp8.

## SHIP LEDGER (2026-07-20, final)
- **Artifact PUBLIC on HF:** rdtand/Hy3-295B-A21B-PrismaQuant-2.9bit-nvfp4cb-
  vllm *(historical id as shipped on 2026-07-20; now 307-redirects to the
  canonical `rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm` — verified
  against the Hub 2026-07-30, R28/D21)* — 104.06 GB, 5 shards, CB MTP included, serving kit (plugin snapshot +
  serve.sh), allocation provenance, per-format allocation table on the card,
  Apache 2.0, NO quality claims. Standalone formats repo:
  github.com/RobTand/cbq (PRIVATE, working title — Robert's rename/public
  call pending).
- **TEB ledger (same bytes, three serving configs, seed 1234, hardmode):**
  interim eager/bf16-shared/no-MTP config = **87** (129/148); ship config
  (CB-direct shared, graphs, TRITON_ATTN, MTP spec k=1) = **85** (126/148).
  Per-scenario diff: only 4/74 flipped (1 up, 3 down; TC-68 "called tools
  when none were needed" = a restraint decision-point flip). Fewer flips
  than the GGUF IQ-vs-k churn (12/74) — serving-numerics churn at the
  plateau, net −3 within binomial noise (P≈0.31). **Honest read: this
  artifact's TEB band is 85–87, GGUF's is 86–87 — same capability plateau,
  NO directional quality claim either way at matched bytes.** The "exceeds
  the shipped GGUF" goal is met on capabilities (MTP) and prefill (2.6×),
  NOT on TEB quality (parity band) and NOT on batch-1 decode (13.1 vs
  17.8).

## JOINT-MENU REGENERATION + gridbook ship (2026-07-20/21)
Robert's orders executed in sequence: pull the HF artifact down; regenerate
with vanilla NVFP4 + FP8_DYNAMIC on the menu; finish the kernel backlog;
canonize MTP rung selection; rename everything **gridbook** (repo
github.com/RobTand/gridbook; python package + registry key "gridbook",
legacy "prismaquant" key still accepted).

**Chain (2.7 h total, was ~10 h):** probe/act reused; cost fresh across the
11-format joint menu; allocation; **delta-export copied 633 targets and
re-encoded 38** (sampled byte-verification gated every copy). Joint
allocation verdict: **36 dense/shared Linears -> vanilla FP8_DYNAMIC; 0 ->
vanilla NVFP4** (offered and never chosen — the A/B's dominance held in the
full solve; the outlier-row units preferred full fp8); layer-21 experts
reflowed fp4-K20 -> fp8-K28. MTP re-encoded at **FP8-CB K44** (2.56 GB) per
the canon selector's degenerate branch.

**First-contact bugs (both fixed + committed):**
- Merge config-groups rebuild emitted stock groups as scheme:null CB — the
  36 vanilla units served UNQUANTIZED bf16. Merge now carries group dicts
  verbatim; shipped config hot-patched; loader also taught that a
  weight_scale with its own param is stock-CT, never an orphaned CB scale.
- Batched-prefill path crashed the serve at 1.4k-token prefill (chunk
  transients ~1.6 GB vs the loop's ~56 MB) — default reverted to loop,
  batched opt-in pending a memory-bounded scale gate.

**Two box OOMs (exit 137) closed by POLICY, not another shave: serve at
util 0.90 on this box** (0.94/0.95 die under long-prefill activation spikes
with the drafter resident; ~6.5 GiB unclaimed pool is the required spike
buffer). serve script default updated; ship serve.sh documents it.

**Ship config verified (util 0.90, graphs, TRITON_ATTN, spec k=1 K44):**
loads, correct, prefill 108.7 tok/s, decode base 14.6 / **prose 16.1 tok/s
with the K44 draft** (K18 draft was throughput-neutral; the selector's pick
turned spec decode positive — acceptance 68% mixed). ship_gridbook =
105.73 GB, 5 shards, serving kit + allocation provenance incl.
mtp_rung_selection.json. Kernel finalization: w2 rowpack = measured
negative (round-2 default stands); persistent-N = GO-as-roadmap
(expand_frac 0.23–0.38, ceiling 1.3–1.6×, reference kernel 6/6);
batched-prefill 27/27 small-scale, at-scale gate pending.

**FINAL SHIP TEB (2026-07-21, ship config, seed 1234): 88/100 (130/148)** —
the highest Hy3 score on this box (GGUF IQ 87/129, k-quant 86/128, prior
body band 85–87). Strictly above the shipped GGUF on the same protocol;
recorded at the top of the plateau band, NOT as a directional quality
claim (single-seed churn ±2–3 established) — though the joint artifact
genuinely carries more fidelity in sensitive places. UPLOADED PUBLIC:
huggingface.co/rdtand/Hy3-295B-A21B-gridbook-2.9bit-vllm (gridbook-branded
id for the public post; card = no quality claims + kit + provenance).
*(Historical id as posted; now 307-redirects to the canonical
`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm` — verified against
the Hub 2026-07-30, R28/D21. Cite the canonical id in new material.)*

## Remaining (perf, not correctness)
- MoE prefill still per-expert-loop (batched-expert expand + grouped GEMM =
  task 15) — prefill already 2.6× GGUF without it.
- Drafter CUDA-graph capture (upstream vLLM) — unlocks spec decode as a
  true multiplier (measured acceptance already 78–93%).
- fp4-CB decode chain cost is a FORMAT-level insight: at GEMV shapes the
  unpack+compose+round work, not bandwidth, is the wall. A future rung or
  decode-contract revision (e.g. fp32-accum without the intermediate
  bf16 round) trades bit-compat for speed — research, not a patch.
- Larger context: 8k–16k fits (footprint §).
