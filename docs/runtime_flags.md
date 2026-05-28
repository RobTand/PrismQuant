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
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` | **archived** | Historical Fisher row-weighted allocator objective. The production pipeline rejects it; archive context lives under `archive/fisher_2026-05-15/`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP` | archived companion | Historical cap for Fisher output-MSE allocation; not used by the production pipeline. |

## Export flags

| env var | default | what it does |
|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | Routes NVFP4 same-shape Linears through the batched GPTQ / optional scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block and keeps the lower block-MSE scale while preserving the same compressed-tensors NVFP4 schema and vLLM kernel. Experimental until .8B/4B/27B KL and vLLM smokes land. |
| `PRISMAQUANT_RENDER_PROGRESSIVE_GATES` | **on** | Production-cache render gate for local mechanisms. All formats use the same progressive order, while unsupported mechanisms are format-gated off. NVFP4 can score FourOverSix, static activation ordering, GPTQ, joint-scale optimization, and optional scale_sweep candidate packages; FP8_DYNAMIC/FP8_E4M3 can score GPTQ with damp sweep, and can additionally score explicit scale_sweep when enabled. MXFP8 remains explicit opt-in and can score static activation ordering plus GPTQ using the canonical E8M0 scale rule. Regressive candidates keep the prior accepted render. Cache metadata records decisions in `render_gates`; FourOverSix has a compact `four_over_six` summary. |
| `PRISMAQUANT_RENDER_GATE_MIN_GAIN` | `0.0` | Optional minimum relative gain required by the progressive render gate. Keep at `0.0` for normal runs so tiny local improvements can accumulate; raise only for ablations. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **archived** | Fisher-weighted GPTQ is archived under `archive/fisher_2026-05-15/` and rejected by the production pipeline. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | archived companion | Historical Fisher-weighted GPTQ row-weight cap; not used by the production pipeline. |
| `PRISMAQUANT_GPTQ_BLOCK_SIZE` | `128` | Column block size for the FP-Quant-style GPTQ OBS update across NVFP4, FP8_DYNAMIC/FP8_E4M3, and explicit MX research formats. Quantizer scales are fixed before the solve; each column is quantized and its error is propagated through the current GPTQ block and later blocks. `PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE` remains accepted as a backward-compatible alias when the new flag is unset. |
| `PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS` | ignored | Legacy candidate E8M0 exponent shifts for the removed MXFP8 joint-scale search. Production MXFP8 no longer consumes `joint_scale_opt`; it uses the canonical E8M0 scale rule inside GPTQ. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | Explicit-ablation candidate E8M0 exponent shifts for MXFP8_E4M3 activation-weighted scale search. The default is a no-op; nonzero shifts are experimental and refine the current accepted render under the same progressive gate. |

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
| `COST_MODE` | `production-render-score` | `production-render-score` (default) renders the full `FORMATS` menu for every Linear through `ProductionWeightCache` and writes an allocator-compatible cost.pkl from the recorded production render scores. `production-render-staged` first builds the normal baseline cost, then renders NVFP4 for every eligible Linear, ranks Linears by post-render local forward error, renders FP8_DYNAMIC candidates only for the high-error tail, marks unmeasured promotions unavailable, and lets BF16 compete only in the staged tail. `local` keeps the legacy `h_trace × output_mse` objective. The default production menu is `NVFP4,FP8_DYNAMIC,BF16`; MXFP8/E5M2 remain explicit research/legacy formats. `grouped-kl` is **archived** (`archive/grouped_kl_2026-05-28/`) — it lost the shipped vLLM A/B on Qwen3.6-27B and now fails fast if requested. |
| `PRODUCTION_RENDER_COST_NSAMPLES` / `PRODUCTION_RENDER_COST_SEQLEN` / `PRODUCTION_RENDER_COST_SEED` | `8` / `1024` / `42` | Calibration contract for `COST_MODE=production-render-staged` and `production-render-score`, using the production cache scorer. |
| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `score_sum` | Render-score field used as allocator `predicted_dloss`. `score_sum` is the GPTQ-style summed reconstruction objective; `score` is the per-element mean used by local render gates and is mainly for ablations. |
| `PRODUCTION_RENDER_COST_PROMOTE_FRACTION` | `0.30` | Fraction of Linears, ranked by NVFP4 post-render error, that receive measured FP8_DYNAMIC promotion candidates plus BF16 fallback in `COST_MODE=production-render-staged`. |
| `H_DETAIL_DIR` | `$WORK_DIR/h_detail` | Historical h-detail location for archived Fisher levers; not required for V1 production defaults. |
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
