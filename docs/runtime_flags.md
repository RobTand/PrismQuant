# prismaquant runtime flags

All performance-critical paths can be tuned at runtime via env vars.
Most proven probe/cost/export flags default ON and exist mostly for opt-out /
debugging. CUDA graph flags are different: L3 and coord-descent graph capture
defaults to `auto` because one-shot candidate batches do not amortize capture
cost. Set the env var to `"1"` to force a graph path for benchmarking, or
`"0"` (also `"false"`, `"no"`, etc.) to disable it.

## Probe + cost flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_DEFERRED_FISHER_SYNC` | **on** | `_run_body_streaming_shard` accumulates h_trace / h_w2_sum on the device as 0-D tensors and batches the host transfer to one .cpu().tolist() per layer. Without it, every Linear's backward hook does two .item() syncs (~94k stalls per phase-3 sweep). Math identical, only timing differs. |
| `PRISMAQUANT_DEFERRED_FISHER_COMPUTE` | **on** | Defers the per-Linear Fisher matmul itself out of the autograd engine's per-Linear callback path. The bwd hook just queues `(name, x, gy, mod_ref)`; after `out.backward()` returns, a tight Python loop drains the queue. SM utilization rises from ~13% to ~50-80% on MoE-heavy phase-3. Math identical. |
| `PRISMAQUANT_ACT_CACHE_ASYNC` | **on** | Activation-cache writes (per-Linear `.pt` files) submit to a small thread pool instead of blocking the main thread. Drains at end of shard so the cost step sees a fully-flushed cache. |
| `PRISMAQUANT_ACT_CACHE_WORKERS` | `4` | Pool size for the above. Higher = more parallel disk writes, but contends with the CPU readers in cost step. |
| `PRISMAQUANT_DIRECT_CUDA_LOAD` | **on** | Pass `device=cuda:N` to `safetensors.safe_open` so layer tensors land on the GPU directly instead of going through a host stage. ~10-30 ms saved per layer load. Falls back transparently if safetensors complains. |
| `PRISMAQUANT_COST_PREFETCH_ACT` | **on** | `measure_batched_gpu` prefetches chunk N+1's activation files on a thread pool while chunk N runs on the GPU. Hides ~30-40% of the cost step's wall on big models. |
| `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` | `0` | Probe guard override for KV-sharing architectures (e.g. Gemma4 `k_eq_v` layers): by default the probe fails fast when KV-shared Linears would receive aliased Fisher mass; set `1` to probe anyway, accepting the shared-gradient double-count. |
| `PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER` | `0` | Probe guard override for packed-MoE experts whose compute is NOT a per-expert `F.linear(x, packed[e])` (e.g. bmm/grouped-mm): the per-token Fisher interception cannot capture them, and by default the probe fails fast rather than fall back to squaring the token-summed weight gradient (the sum-then-square estimator, audit M3: 5-50× cross-token-covariance inflation). Set `1` to accept the biased legacy estimator. Also accepted by `prepare_cost_context` to reuse a PRE-FIX probe.pkl whose packed-expert entries lack the `packed_fisher_estimator=per_token_v2` meta stamp (stale pickles are otherwise refused). |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` | **archived** | Historical Fisher row-weighted allocator objective. The production pipeline rejects it; archive context lives under `archive/fisher_2026-05-15/`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP` | archived companion | Historical cap for Fisher output-MSE allocation; not used by the production pipeline. |

## Cost-estimation subsampling

| Flag | Default | What it does |
|---|---|---|
| `PRISMAQUANT_EXPERT_COST_SAMPLE` | `0` (off) | Stratified expert subsample for COST entries on stacked-expert (3D) weights: prices `S` of `E` experts and takes the mean, at `E/S` less quantize work. The allocator prices the whole stack as one unit, so this estimates exactly the quantity it consumes. **Export always quantizes every expert** — this only touches the DP's cost estimates. Added because an exhaustive IQ grid search on a 3.5G-element stack ran ~25 min/layer at full `E`, and a full-stack NVFP4/FP8 registry quantize swap-kills a unified-memory box the same way (2026-07-11). |
| `PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE` | `0` (off) | Legacy name for the above, read as a fallback when `PRISMAQUANT_EXPERT_COST_SAMPLE` is unset. The mechanism is family-agnostic despite the name; prefer the unprefixed flag. |

## Export flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | Routes NVFP4 same-shape Linears through the batched GPTQ / optional scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block. `joint_mse` is the production JSO scale rule selected by the `joint_scale_opt` lever: it chooses from `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` under the served FP8-snapped scale objective. All preserve the compressed-tensors NVFP4 schema and vLLM kernel. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` | `6,4` | Candidate max-to-levels for NVFP4 joint scale optimization. Extend only for explicit JSO ablations; production defaults use the validated `{6,4}` grid. |
| `PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING` | `0` | Research lever: score NVFP4 scale candidates under the FP8-snapped effective scale with a per-tensor global fixed point. More serve-faithful in principle but changes shipped NVFP4 bytes for `joint_mse`/`four_over_six` — default OFF pending a served gold-metric A/B. Recorded in the export fingerprint. |
| `PRISMAQUANT_COST_UCB_Z` | `0` | Risk-aware allocation: charge `z·predicted_dloss_stderr` on top of each AURA cost row (upper-confidence-bound). `0` = bit-identical legacy behavior. The 27B A/B found point-KL parity with z=2; UCB's case is allocation stability across probe seeds (−15% seed-Hamming at 4B), pending the production-calib seed-stability decision. |
| `PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN` | `0` | Research/A-B escape hatch: allows non-BF16 packed-MoE experts to skip the production-cache GPTQ render and export RTN bytes. Never use for a production artifact. |
| `PRISMAQUANT_ALLOW_UNSCALED_FP8` | `0` | Streaming-load guard override: by default a float8-dtyped checkpoint tensor with no entry in the fp8 scale-inv map fails fast (loading raw FP8 codes as if they were weights is the historical ±448-range corruption). Set `1` to permit the raw cast anyway (debug only). |
| `PRISMAQUANT_EXPERT_LAZY_FILL` | **on** | M4 frontier expert selection: `validate_assignments_kl` lazily renders a Pareto point's missing packed-expert entries (e.g. FP8) into the shared frontier cache just before scoring, on the BUILD/render calib split, then re-pickles the cache so recache/export ship the same bytes real KL selected. The format-menu build eager-renders only the NVFP4 rung. `0` restores the legacy hard-fail on expert cache misses. |
| `PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE` | **on** | M19: NVFP4 export re-derives block scales from the cached production render using the SAME scale rule the render chose (`joint_mse` under JSO), instead of re-quantizing the bf16 dequant with `static_6`. Served-validated −6.6% KL / −3.3% PPL on the 4B paired A/B. Since the 2026-07-02 audit (M2) this also covers packed-expert re-pack and the fused joint-global pre-passes (rule = the cache's recorded `nvfp4_scale_rule` lever; env default when nothing is recorded). `0` reproduces pre-M19 artifacts. |
| `PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES` | `0` | Perturbed-X emulation hooks: `1` quantizes NVFP4 activations with the SERVE-faithful two-level semantics (static per-tensor `input_global_scale` derived from the calibrated max_abs — honoring `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` — plus FP8 snap of each 16-group block scale, including block-zeroing and above-calibration-amax clipping) instead of the dynamic exact-fp32-scale RTN. Closes the audit M18-residual/C1 measurement gap; default off pending a served correlation study. |
| `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` | `0` | C1 (2026-07-02 audit): `1` switches NVFP4 `input_global_scale` to the compressed-tensors/vLLM `generate_gparam` convention `448·6/amax`, placing serve-time FP8-stored activation block scales in (0,448] instead of the legacy (0,1] — rescues blocks ≫64× below calibration amax from FP8 subnormals, but CLIPS any serve block whose amax exceeds calibration amax. Served A/Bs (byte-identical weights, only this scalar ×448, 2 window draws each): 35B MoE frontier **−14.1% KL (win)**; 27B regen dense **+37.5% (loss)**; LFM2.5 thin-calib smoke +5.8% (loss). Strongly artifact-dependent → default stays legacy; opt in per artifact behind a served A/B. The scale is a free post-export knob (in-place patch, no re-render) — see `/home/rob/dq-runs/c1-igs-ab-20260702/patch_igs.py`. |
| `PRISMAQUANT_GPTQ_DAMP_ROLES` | unset | Per-role GPTQ damp override, e.g. `qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0`. Default-off research lever (the 2026-06-22 per-role served A/B was NULL; fixed damp 1.0 is final). Unlisted roles keep the fixed damp. |
| `PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE` | **on** | Export/build coverage guard for assignment-required production-cache entries. Missing non-BF16 renders fail early instead of falling through to RTN or late materialization errors. |
| `PRISMAQUANT_STRICT_PRODUCTION_CACHE` | **on** | KL/activation-cache residency guard. Missing required production-cache weights fail fast by default; set `0` only for explicit legacy/non-production fallback runs. |
| `PRISMAQUANT_DO_NO_HARM` | **on** | Enables export-time GPTQ-vs-RTN do-no-harm gates where supported. Failures and reverts are counted in export provenance. |
| `PRISMAQUANT_DO_NO_HARM_MIN_GAIN` | `0.0` | Optional relative-gain floor for accepting a do-no-harm candidate. Keep at `0.0` for normal production runs. |
| `PRISMAQUANT_RENDER_PROGRESSIVE_GATES` | **on** | Production-cache render gate for local mechanisms. All formats use the same progressive order, while unsupported mechanisms are format-gated off. NVFP4 can score FourOverSix, static activation ordering, GPTQ, joint-scale optimization, and optional scale_sweep candidate packages; FP8_DYNAMIC/FP8_E4M3 can score GPTQ with damp sweep, and can additionally score explicit scale_sweep when enabled. MXFP8 remains explicit opt-in and can score static activation ordering plus GPTQ using the canonical E8M0 scale rule. Regressive candidates keep the prior accepted render. Cache metadata records decisions in `render_gates`; FourOverSix has a compact `four_over_six` summary. |
| `PRISMAQUANT_RENDER_GATE_MIN_GAIN` | `0.0` | Optional minimum relative gain required by the progressive render gate. Keep at `0.0` for normal runs so tiny local improvements can accumulate; raise only for ablations. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **archived** | Fisher-weighted GPTQ is archived under `archive/fisher_2026-05-15/` and rejected by the production pipeline. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | archived companion | Historical Fisher-weighted GPTQ row-weight cap; not used by the production pipeline. |
| `PRISMAQUANT_GPTQ_BLOCK_SIZE` | `128` | Column block size for the FP-Quant-style GPTQ OBS update across NVFP4, FP8_DYNAMIC/FP8_E4M3, and explicit MX research formats. Quantizer scales are fixed before the solve; each column is quantized and its error is propagated through the current GPTQ block and later blocks. `PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE` remains accepted as a backward-compatible alias when the new flag is unset. |
| `PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS` | ignored | Legacy candidate E8M0 exponent shifts for the removed MXFP8 joint-scale search. Production MXFP8 no longer consumes `joint_scale_opt`; it uses the canonical E8M0 scale rule inside GPTQ. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | Explicit-ablation candidate E8M0 exponent shifts for MXFP8_E4M3 activation-weighted scale search. The default is a no-op; nonzero shifts are experimental and refine the current accepted render under the same progressive gate. |

### Export-time overrides

| Flag | Default | What it does |
|---|---|---|
| `PRISMAQUANT_TARGET_PROFILE` | unset | Audit export legality under the **same** serving profile the allocator solved with, instead of the architecture's spec default. Without it, an arch whose default differs (Hy3 defaults to `gguf`) coerces every format that default cannot serve — measured 2026-07-11: 226 dense FP8 Linears silently demoted to BF16 on the Hy3 compressed-tensors export. |
| `PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ` | `0` (off) | Run per-expert GPTQ inline during export, passing a transient rendered stack rather than materialising one. Opt-in; the default export path renders from the production cache. |

## Pipeline production-cache flags

These are `run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags.

Research levers outside the current production recipe live under `archive/`.
The production pipeline fails fast when archived Fisher levers are requested.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
| `PRODUCTION_CACHE_LEVERS` | `gptq,static_act_order,joint_scale_opt` | V1 production render levers: GPTQ with damp sweep plus static activation ordering and joint scale optimization where the format supports them. FP8_DYNAMIC/FP8_E4M3 uses GPTQ damp sweep without static ordering or JSO because the served representation is per-row scaled FP8 dynamic. `static_act_order` applies to production microscaling GPTQ formats: NVFP4, MXFP4, and explicit MXFP8. `joint_scale_opt` applies only to NVFP4. MXFP4/MXFP8 use the canonical E8M0 scale rule when explicitly requested. `scale_sweep` remains available for explicit ablations but is not a default. Runtime activation scores use their served activation quantizers; NVFP4 is the only current score path that applies the calibrated activation-max clip. |
| `FISHER_WEIGHTED_GPTQ` | archived | Any truthy value is rejected; archive context lives under `archive/fisher_2026-05-15/`. |
| `FISHER_OUTPUT_MSE_ALLOCATOR` | archived | Any truthy value is rejected; V1 allocation uses the non-Fisher objective plus measured frontier validation. |
| `COST_MODE` | `production-render-score` | `production-render-score` (default) renders the full `FORMATS` menu for every Linear through `ProductionWeightCache` and writes an allocator-compatible cost.pkl from the recorded production render scores. `production-render-staged` first builds the normal baseline cost, then renders NVFP4 for every eligible Linear, ranks Linears by post-render local forward error, renders FP8_DYNAMIC candidates only for the high-error tail, marks unmeasured promotions unavailable, and lets BF16 compete only in the staged tail. `local` keeps the legacy `h_trace × output_mse` objective. `aura` runs the AURA downstream-KL-adjoint cost (`aura_cost.py`, served −38% KL @4B / −17.9% @27B vs the h_trace×output_mse baseline) against a production-rendered dW cache; on packed-MoE models the route-flip-blind smooth cost is replaced for experts by measured empirical unit-KL (`expert_empirical_cost.py`, FP8 kept in the menu) merged into one hybrid payload, with MTP/visual sidecar rows backfilled from the baseline cost (recorded in provenance). Under `SELECTION_MODE=validated-surrogate` the AURA dW cache IS the frontier cache (one cache: cost dW = measured bytes = exported bytes). The default production menu is `NVFP4,FP8_DYNAMIC,BF16`; MXFP8/E5M2 remain explicit research/legacy formats. `grouped-kl` is **archived** (`archive/grouped_kl_2026-05-28/`) — it lost the shipped vLLM A/B on Qwen3.6-27B and now fails fast if requested. |
| `AURA_COST_NPROBES` / `AURA_COST_NSAMPLES` / `AURA_COST_SEQLEN` / `AURA_COST_CALIB_SEED` | `32` / `8` / `128` / `42` | COST_MODE=aura probe/calibration volume (defaults = the regen-27b recipe). Also `AURA_COST_LINEAR_CHUNKS` (8), `AURA_COST_PROBE_MICROBATCH` (8), `AURA_COST_MIN_FREE_GIB` (18). |
| `AURA_COST_DTYPE` | `auto` | Resident model dtype for the aura_cost stage. `auto` sizes the checkpoint (fp8 sidecars counted at 1 byte/param) and picks `float32` (additivity-preferred, the 27B regen regime) only when params×4 bytes + `AURA_COST_MIN_FREE_GIB` fits in MemAvailable, else `bfloat16` (35B-class — fp32 is ~140 GiB against the 121 GiB pool and OOM-kills the box). |
| `AURA_EXPERT_NSAMPLES` / `AURA_EXPERT_SEQLEN` | `16` / `512` | COST_MODE=aura empirical packed-expert unit-KL stage calibration volume (the 35B arm-E recipe). |
| `VALIDATED_FRONTIER_MATERIALIZATION` | `hooks` | How the validated frontier materializes each Pareto point. `hooks` = all points in one process (fast; needs model + full render set co-resident — OOMs 35B-class MoE on the 128 GB pool). `inplace` = one `validate_assignments_kl` process per point, per-point JSONs merged for selection (the memory-fit 35B path). |
| `PRODUCTION_RENDER_COST_NSAMPLES` / `PRODUCTION_RENDER_COST_SEQLEN` / `PRODUCTION_RENDER_COST_SEED` | `8` / `1024` / `42` | Calibration contract for `COST_MODE=production-render-staged` and `production-render-score`, using the production cache scorer. |
| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `weight_mse` | Render-score field used as allocator cost. `weight_mse` (default since 2026-07-02, audit M6) routes through the dimensionally-consistent `h_trace × weight_mse` path — the legacy `output_mse` product double-counted activation energy E‖x‖² (h_trace already contains it), a per-Linear bias ∝ in_features·x_rms². Served two-arm A/B at matched 4.75 bpp: Qwen3-4B KL −50.8% / 32k-PPL −15.1%; Qwen3-0.6B KL −58.5% / 32k-PPL −24.4% (5 window draws each, same pipeline seeds). `output_mse` reproduces the historical objective; `score_sum`/`score` are ablation fields. 27B-class confirmation = ladder debt. |
| `PRODUCTION_RENDER_COST_PROMOTE_FRACTION` | `0.30` | Fraction of Linears, ranked by NVFP4 post-render error, that receive measured FP8_DYNAMIC promotion candidates plus BF16 fallback in `COST_MODE=production-render-staged`. |
| `SELECTION_MODE` | `surrogate` | `surrogate` preserves the normal allocator-selected `TARGET_BITS` assignment. `validated-surrogate` writes allocator Pareto assignments, builds a format-menu production cache, measures real assignment KL for each Pareto point, selects the measured KL/bpp kneedle with `prismaquant.select_validated_frontier`, then recaches and exports the selected assignment. |
| `VALIDATED_FRONTIER_NSAMPLES` / `VALIDATED_FRONTIER_SEQLEN` | `$NSAMPLES` / `$SEQLEN` | Calibration size for measured-frontier KL selection. Keep these at the artifact validation contract for 27B decisions; lower values are smoke-only. |
| `VALIDATED_FRONTIER_PICK` | `kneedle` | Selection rule for `SELECTION_MODE=validated-surrogate`: `kneedle`, `best-kl`, or `lowest-bpp`. Production selection should use `kneedle` unless the run is explicitly an ablation. |
| `PRODUCTION_CACHE_LRU_GB` | `64.0` | Resident tensor budget for disk-backed production-cache use in recache and export. The 27B n=8 recache smoke needed `45.32 GiB` for the selected assignment. |
| `PRODUCTION_CACHE_PREFETCH` | `require` | Standalone recache prefetch policy. `require` fails fast when assignment-required weights cannot fit resident, preventing silent NVMe-bound replay. |
| `PRODUCTION_CACHE_PREFETCH_WORKERS` | `4` | Thread count for eager production-cache prefetch. |

## CUDA / system flags

| env var | recommended | what it does |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Required on UMA hardware (DGX Spark) to keep the CUDA caching allocator from hoarding freed blocks. Set automatically by `incremental_probe.py` at module load. |
| `PRISMAQUANT_L3_CUDA_GRAPHS` | `auto` | Graphs decoder-tail L3 propagation only when the same graph key has enough repeated calibration calls. Default threshold: `8`, override with `PRISMAQUANT_L3_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS` | `auto` | Graphs lane-batched coord flip evaluation only when repeated calls can amortize capture. Replay-cache coord batches are one-shot, so auto leaves them eager. Default thresholds: `8` for replay mode, `16` for full-forward mode; override with `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_KL_CUDA_GRAPHS` | `auto` | Graphs assignment-KL validation only for larger calibration batches. Default threshold: `16`, override with `PRISMAQUANT_KL_CUDA_GRAPHS_MIN_CALLS`. |
| `PRISMAQUANT_COORD_REPLAY_CACHE` | `off` | Opt-in LayerHiddenStateCache for coord descent. It reduces tail layer forwards but currently copies too much baseline model state on large Qwen runs, so the fast default is lane-batched eager evaluation. |
| `PRISMAQUANT_L3_MIN_HOST_MEM_GB` | unset | Optional host-memory floor for L3 pair/scout diagnostics. When set, L3 raises `GPUMemoryBudgetExceeded` between paired-override chunks if `/proc/meminfo` `MemAvailable` drops below this many GiB, giving long runs a chance to stop before system OOM. |

## Live PRISMAQUANT flag index

The tables above explain the production-facing knobs. This index is the
complete live `PRISMAQUANT_*` vocabulary found in `prismaquant/`, including
low-level debug, cache, validation, and archived-research switches:

```text
PRISMAQUANT_ACT_CACHE_ASYNC
PRISMAQUANT_ACT_CACHE_FP32
PRISMAQUANT_ACT_CACHE_WORKERS
PRISMAQUANT_ACT_CLIP_QUANTILE
PRISMAQUANT_ALLOW_KV_SHARED_FISHER
PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN
PRISMAQUANT_ALLOW_PYTORCH_FALLBACK
PRISMAQUANT_ALLOW_UNSCALED_FP8
PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE
PRISMAQUANT_BATCHED_NVFP4_EXPORT
PRISMAQUANT_BLOCK_OUTPUT_MATCH
PRISMAQUANT_CHANNEL_SENTINEL
PRISMAQUANT_COORD_LANE_CUDA_GRAPHS
PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_COORD_REPLAY_CACHE
PRISMAQUANT_COST_PREFETCH_ACT
PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH
PRISMAQUANT_DAMP_ANALYTICAL
PRISMAQUANT_DAMP_ANALYTICAL_C
PRISMAQUANT_DAMP_SWEEP_LOG
PRISMAQUANT_DEFERRED_FISHER_COMPUTE
PRISMAQUANT_DEFERRED_FISHER_SYNC
PRISMAQUANT_DETERMINISTIC
PRISMAQUANT_DIRECT_CUDA_LOAD
PRISMAQUANT_DISABLE_RTN_COMPILE
PRISMAQUANT_DO_NO_HARM
PRISMAQUANT_DO_NO_HARM_VERBOSE
PRISMAQUANT_EMPTY_CACHE_EACH_REPLAY_BATCH
PRISMAQUANT_EXPERT_COST_SAMPLE
PRISMAQUANT_EXPERT_LAZY_FILL
PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ
PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT
PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP
PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR
PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP
PRISMAQUANT_FISHER_WEIGHTED_GPTQ
PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE
PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION
PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB
PRISMAQUANT_FULL_SENTINEL
PRISMAQUANT_FULL_SEQUENCE_KL
PRISMAQUANT_FUSED_KERNEL_NVFP4
PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE
PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE
PRISMAQUANT_GPTQ_BLOCK_SIZE
PRISMAQUANT_GPTQ_DAMP
PRISMAQUANT_GPTQ_DAMP_ROLES
PRISMAQUANT_GPTQ_DAMP_SWEEP
PRISMAQUANT_GPTQ_STATIC_ACT_ORDER
PRISMAQUANT_GPU_MEM_RESERVE_FRACTION
PRISMAQUANT_GPU_MEM_RESERVE_GB
PRISMAQUANT_GRAPH_AUDIT
PRISMAQUANT_GRAPH_OUTPUT_CLONE
PRISMAQUANT_GRAPH_POOL
PRISMAQUANT_GRAPH_SHARED_POOL
PRISMAQUANT_HOST_MEM_RESERVE_FRACTION
PRISMAQUANT_HOST_MEM_RESERVE_GB
PRISMAQUANT_IQ_COMPILE_SWEEP
PRISMAQUANT_KL_CUDA_GRAPHS
PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE
PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_L2_CUDA_GRAPHS
PRISMAQUANT_L3_CUDA_GRAPHS
PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE
PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_FRACTION
PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB
PRISMAQUANT_L3_MIN_HOST_MEM_GB
PRISMAQUANT_L3_PREQUANT_CACHE
PRISMAQUANT_L3_PREQUANT_CACHE_PEAK_MULTIPLIER
PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_FRACTION
PRISMAQUANT_L3_PREQUANT_CACHE_RESERVE_GB
PRISMAQUANT_MASK_CUDA_DURING_META_INIT
PRISMAQUANT_MAX_GPU_MEM_GB
PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS
PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS
PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES
PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE
PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP
PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI
PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO
PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS
PRISMAQUANT_NVFP4_JOINT_SCALE_OPT
PRISMAQUANT_NVFP4_SCALE_RULE
PRISMAQUANT_PATCH_SENTINEL
PRISMAQUANT_PROBE_CTX_CACHE
PRISMAQUANT_PROBE_DOMAIN
PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK
PRISMAQUANT_PROD_ACT_SCALES
PRISMAQUANT_RENDER_GATE_MIN_GAIN
PRISMAQUANT_RENDER_PROGRESSIVE_GATES
PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE
PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE
PRISMAQUANT_STRICT_PRODUCTION_CACHE
PRISMAQUANT_TARGET_PROFILE
PRISMAQUANT_TMPDIR
PRISMAQUANT_UMA_MEMORY_INFO
PRISMAQUANT_VALIDATION_CUDA_GRAPHS
PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE
PRISMAQUANT_VALIDATION_FAKE_METRICS
PRISMAQUANT_VALIDATION_PROD_CACHE
PRISMAQUANT_VALIDATION_PROD_CACHE_DIR
PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB
PRISMAQUANT_VALIDATION_SKIP_END_KL
PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE
```

## Disabling for debugging

To revert a single flag for A/B comparison:

```bash
PRISMAQUANT_DEFERRED_FISHER_COMPUTE=0 \
  python -m prismaquant.incremental_probe ...
```

To revert ALL perf flags (legacy v20 behavior):

```bash
for f in DEFERRED_FISHER_SYNC DEFERRED_FISHER_COMPUTE ACT_CACHE_ASYNC \
         DIRECT_CUDA_LOAD COST_PREFETCH_ACT BATCHED_NVFP4_EXPORT; do
    export "PRISMAQUANT_$f=0"
done
```

## GGUF lane (2026-07-06, `docs/gguf_lane.md`)

| Flag | Default | What it does |
|---|---|---|
| `PRISMAQUANT_IQ_COMPILE_SWEEP` | `1` (on) | `torch.compile` the IQ full-grid sweep kernels. Set to `0`/`false`/`no` to force the eager path — the compiled and eager sweeps compute the same errors, so this is a debug/compat switch, not a quality one. |


| Flag | Default | Meaning |
|---|---|---|
| `EXPORT_CONTAINER` | `compressed-tensors` | `gguf` switches stage 4 to skeleton-build + `export_gguf` (llama.cpp/vLLM-GGUF artifact); requires `TARGET_PROFILE=gguf` at allocation |
| `PRISMAQUANT_GGUF_IMATRIX` | `1` | activation-weighted (imatrix) k-quant scale selection in BOTH the batched cost path and the pipeline's export call — keep the two in lockstep or the A/B has a rendering confound |
| `LLAMA_CPP_DIR` | `/home/rob/dq-runs/llama.cpp` | source of `convert_hf_to_gguf.py` for the skeleton |
| `GGUF_SKELETON` | `WORK_DIR/artifacts/skeleton.gguf` | bf16 skeleton path (built if missing) |
| `GGUF_TOKEN_EMBEDDING_FORMAT` / `GGUF_OUTPUT_FORMAT` | keep skeleton precision | quantize `token_embd` / `output` (llama.cpp presets use Q2_K / Q6_K) |
