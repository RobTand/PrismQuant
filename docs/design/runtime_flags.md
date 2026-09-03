# prismaquant runtime flags

*Reconciled against code 2026-07-30 (branch `claude/docs-consolidation`);
citations and live/dead status re-verified 2026-08-02 against `main` @ v0.6.0.*
The 2026-08-02 pass retired the L2/L3 knobs the 2026-07-30 wall had left in the
live tables (§9.1) and re-resolved every `file.py:NNN` in this document that
pointed past its file's EOF — the split moved the L3 half of
`kl_measurement.py` into `archive/l3_propagated_2026-07-30/` and the citations
were never re-anchored.
*2026-08-14: one row added, `PRISMAQUANT_PROBE_MARGINALS` (§1) — a new
probe-module default, not a `run-pipeline.sh` variable. No other row re-verified
on that pass.*
Method: AST + literal sweep of `os.environ` / `os.getenv` / `_env_flag` /
`_env_int` / `_env_flag_enabled` / registry `*_env=` parameters / `pq_env_*`
across `prismaquant/`, `tools/`, `scripts/`, `pipeline.py`, and the separately
versioned serving-runtime source at its pinned commit (through 2026-09-02
that was Gridbook; that lane is retired — §5, §7)
(excluding `archive/`, `fp8/`, `scratch/`, `tests/`, worktrees). Every row cites
its reading `file:line`; when a flag has several readers the row cites the one
that decides behaviour and notes the others.

## 0. Calibration corpus (`DATASET`, `PRISMAQUANT_CALIBRATION_DIR`)

Calibration corpora are **built, not vendored** — the repo ships the builder,
not the bytes, because the corpora are large and the mix is a methodological
choice a user should make deliberately.

| var | default | meaning |
|---|---|---|
| `PRISMAQUANT_CALIBRATION_DIR` | `<repo>/calibration` | where corpora live |
| `DATASET` | `$PRISMAQUANT_CALIBRATION_DIR/diverse-v1.jsonl` | probe/cost/render calibration |
| `EXPERT_GATE_DATASET` | `$PRISMAQUANT_CALIBRATION_DIR/xdom-gate-v1.jsonl` | MoE-only cross-domain GPTQ-vs-RTN gate corpus; must be DISJOINT from `DATASET`; set empty for the historical same-corpus gate |
| `VALIDATED_FRONTIER_DATASET` | `$DATASET` | held-out KL selection corpus |

`DATASET` accepts three things (`sensitivity_probe.load_calibration`): a local
`.jsonl` (rows of `{"text": ...}` or `{"messages": [...]}`), a local `.txt`
(one sample per line), or a HuggingFace dataset id.

Build the default corpus with:

```bash
python tools/build_diverse_calibration.py \
  --tokenizer <model dir or HF id> \
  --output "$PRISMAQUANT_CALIBRATION_DIR/diverse-v1.jsonl"
```

It is a 40/20/20/20 prose/code/math/multilingual mix drawn from public HF
datasets at pinned revisions, with a manifest as its first row, so it
regenerates reproducibly on any machine. `--tokenizer` is required: rows are
built to a token budget, so a corpus is only reusable across models whose
tokenizers agree.

**A local path that does not exist now fails immediately** and says so. It
previously fell through to the HuggingFace loader, which reported
`Dataset '/home/.../diverse-v1.jsonl' doesn't exist on the Hub or cannot be
accessed` — an error naming a filesystem path as a dataset id, which reads as
"you failed to download something" when in fact nothing is downloadable.
There is no `xdom-gate-v1` builder in the repo yet; on a dense model that
corpus is unused, and on MoE set `EXPERT_GATE_DATASET` to your own disjoint
corpus (or empty, accepting the weaker same-corpus gate).

All performance-critical paths can be tuned at runtime via env vars.
Most proven probe/cost/export flags default ON and exist mostly for opt-out /
debugging. CUDA graph flags are different: assignment-KL graph capture defaults
to `auto` because small one-shot calibration batches do not amortize capture
cost, so it graphs only when the same key is expected to run at least
`*_MIN_CALLS` times. Set the env var to `"1"` to force a graph path for
benchmarking, or `"0"` (also `"false"`, `"no"`, etc.) to disable it. (The L3
and coord-descent graph selectors that used to be described here retired with
their subsystem — §9.1.)

Three consumer families share the `PRISMAQUANT_*` namespace and are separated
below: the **build pipeline** (probe/cost/render/export, §1–§4), the **CB build
render plumbing** (§5 — its lane was retired 2026-09-02), and the **serving
plugin** flags (§7), which run inside vLLM
and never sees a build flag.

## 1. Probe + cost flags

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_DEFERRED_FISHER_SYNC` | **on** | `incremental_probe.py:2044` | `_run_body_streaming_shard` accumulates h_trace / h_w2_sum on the device as 0-D tensors and batches the host transfer to one `.cpu().tolist()` per layer. Without it, every Linear's backward hook does two `.item()` syncs (~94k stalls per phase-3 sweep). Math identical, only timing differs. |
| `PRISMAQUANT_DEFERRED_FISHER_COMPUTE` | **on** | `incremental_probe.py:2073` | Defers the per-Linear Fisher matmul itself out of the autograd engine's per-Linear callback path. The bwd hook queues `(name, x, gy, mod_ref)`; after `out.backward()` returns, a tight Python loop drains the queue. SM utilization rises from ~13% to ~50-80% on MoE-heavy phase-3. Math identical. |
| `PRISMAQUANT_ACT_CACHE_ASYNC` | **on** | `incremental_probe.py:1738` | Activation-cache writes (per-Linear `.pt` files) submit to a small thread pool instead of blocking the main thread. Drains at end of shard so the cost step sees a fully-flushed cache. |
| `PRISMAQUANT_ACT_CACHE_WORKERS` | `4` | `incremental_probe.py:1744` | Pool size for the above. Higher = more parallel disk writes, but contends with the CPU readers in cost step. |
| `PRISMAQUANT_ACT_CACHE_FP32` | `1` | `incremental_probe.py:1770` / `:2561` | Store cached activations in fp32 rather than the model dtype. |
| `PRISMAQUANT_DIRECT_CUDA_LOAD` | **on** | `layer_streaming.py:103` | Pass `device=cuda:N` to `safetensors.safe_open` so layer tensors land on the GPU directly instead of going through a host stage. ~10-30 ms saved per layer load. Falls back transparently if safetensors complains. |
| `PRISMAQUANT_LAYER_READ_THREADS` | auto (`min(8, cpu//2)`) | `layer_streaming.py` (`layer_read_threads`, `_read_layer_to_device`) | Worker count for the intra-layer gather: one streamed layer of a large MoE checkpoint is thousands of small tensors (GLM-5.3-Flash: 1759 tensors / ~6.8 GB of FP8 source per body layer), and reading them one at a time is a single mmap page-fault stream. Measured cold on the GB10 NVMe over 6 GLM body layers: serial **2.80 GB/s**, 8-thread gather **4.76 GB/s** (1.70x), against a **5.1 GB/s** raw 12-stream `O_DIRECT` device ceiling — i.e. the gather now runs at ~93% of what the disk can give. Layers with fewer than 16 tensors keep the serial path (dense models see no change). `1` restores the byte-identical serial read. The pool is shared process-wide so `prefetch workers x gather threads` cannot multiply into disk thrash. Same tensors, same dtype cast, same contiguity, same deterministic key order — only the page-fault order changes. |
| `PRISMAQUANT_PREFETCH_DELIVERY` | **on** | `streaming_model.py` (`_prefetch_delivery_enabled`) | The streamed-loader admission contract: **admission is decided before the read, delivery is guaranteed after it**. A completed speculative layer read is handed to its consumer even when `LayerCache.put(force=False)` declines to *retain* it. `schedule_prefetch` reserves only unfinished reads against `affordable_prefetch_slots()` (MemAvailable minus the pressure floor, in layers): completed futures remain claimable delivery aliases, but their tensor storage is already reflected in MemAvailable and must not be double-reserved. `install()` re-derives the achievable lookahead window each step for a +-1 walk so one refused enqueue no longer drops the rest of the sweep to cold reads. `0` reproduces the pre-2026-08-26 discard-on-refusal loader for a controlled A/B — production leaves it on. Tests: `tests/test_streamed_prefetch_scheduling.py`. |
| `PRISMAQUANT_COST_PREFETCH_ACT` | **on** | `measure_quant_cost.py:1518` | `measure_batched_gpu` prefetches chunk N+1's activation files on a thread pool while chunk N runs on the GPU. Hides ~30-40% of the cost step's wall on big models. |
| `PRISMAQUANT_PROBE_MARGINALS` | **on** | `incremental_probe.py:94` (`_marginals_enabled`); CLI twin `--emit-marginals` / `--no-emit-marginals` publishes into this env var at `:3486` | Emit five per-channel vectors per Linear into `probe.pkl` alongside the scalars: `fisher_row [out]`, `fisher_col [in]` (the two marginals of the per-element weight Fisher `H`, which is itself unstorable at ~800 GB), plus `g_sq_sum [out]`, `act_sq_sum [in]`, `act_absmax [in]`. Cost is `(2·out + 3·in)·4` bytes per Linear and **no extra matmul** — every reduction is taken off the `chunk_h` each accumulation site already forms. They are what the Sensitivity Card prices an arbitrary format menu from (ARCHITECTURE.md §4.8); nothing in the production allocator reads them yet. `0` / `--no-emit-marginals` restores byte-identical legacy output. `run-pipeline.sh` sets nothing, so this is a **module** default, not a shell one. Sums add elementwise across chunks and shards; `act_absmax` is a bound and merges by elementwise **maximum** (one `merge_marginals` shared by the per-layer flush and the cross-shard merge). Because the flag decides WHICH KEYS a stats entry carries rather than how work is grouped, it is part of the precompute-cache fingerprint, `_expected_probe_shard_meta`, and `_CONTENT_META_KEYS` (`5639a8b`): a flag-off shard is rebuilt rather than pooled into a flag-on run, including shards written before the key existed. Tests: `tests/test_probe_marginals.py`. |
| `PRISMAQUANT_PROBE_DOMAIN` | unset | `incremental_probe.py:970` | Calibration-domain tag stamped into probe provenance. |
| `PRISMAQUANT_PROBE_CTX_CACHE` | unset | `incremental_probe.py:3126` | Reuse the cross-chunk probe context cache. |
| `PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK` | unset | `incremental_probe.py:3143` | Retain cross-chunk probe state instead of dropping it between chunks. |
| `PRISMAQUANT_ALLOW_KV_SHARED_FISHER` | `0` | `incremental_probe.py:1022` | Probe guard override for KV-sharing architectures (MINOR-M33). Only reachable with `PRISMAQUANT_KV_COTANGENT=0`: severing the shared-consumer cotangent *under*-counts the storing layer's `k_proj`/`v_proj` `h_trace`, so the probe fails fast; set `1` to probe anyway, accepting the under-count. (Earlier revisions called this an aliased-Fisher *double*-count — wrong direction; the missing edge only ever removes gradient.) |
| `PRISMAQUANT_KV_COTANGENT` | on | `sensitivity_probe.py:1254` | The KV-cotangent path. On cross-layer KV-sharing architectures (Gemma4 `num_kv_shared_layers>0`) the phase-3 reverse sweep grafts grad-enabled leaves over borrowed K/V, sums each consumer's `leaf.grad` per source, and drives the storing layer's backward with that sum alongside its own output cotangent. Without it a sharing layer's backward stops at the detached capture and the storing layer's `h_trace` is under-counted. `0` restores the pre-fix severed cotangent for an A/B and re-arms `PRISMAQUANT_ALLOW_KV_SHARED_FISHER`. Verified against an end-to-end backward in `tests/test_kv_cotangent_path.py`. |
| `PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER` | `0` (off) | `incremental_probe.py` | Restores the v22 "Fix E1" phase-1 activation transfer: hold all L+1 layer activations device-resident, then stack for a single device→host copy. Default (off) streams each layer's activation to host inside the forward loop, bounding device residency to one activation — a doubling DSv4's multi-stream hidden can't afford. Exists to A/B the unmeasured transfer-time cost; both paths report true copy time as `host transfer`. Batched mode requires uniform activation shapes. |
| `PRISMAQUANT_SOLVER_TRACE` | `0` (off) | `allocator_solver.py` | Per-evaluation trace for `solve_with_promotion`: each tightened-target DP eval with achieved bits, predicted Δloss, wall time, plus DP-infeasible probes. Read once at module import — set before launching the allocator. Observability only. |
| `PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER` | `0` | `sensitivity_probe.py:673` (const `:451`), `measure_quant_cost.py:1824` | Probe guard override for packed-MoE experts whose compute is NOT a per-expert `F.linear(x, packed[e])` (e.g. bmm/grouped-mm): the per-token Fisher interception cannot capture them, and by default the probe fails fast rather than fall back to squaring the token-summed weight gradient (the sum-then-square estimator, audit M3: 5-50× cross-token-covariance inflation). Set `1` to accept the biased legacy estimator. Also accepted by `prepare_cost_context` to reuse a PRE-FIX probe.pkl whose packed-expert entries lack the `packed_fisher_estimator=per_token_v2` meta stamp (stale pickles are otherwise refused). |
| `PRISMAQUANT_COST_UCB_Z` | `0` — **RESEARCH-ONLY** | `allocator_candidates.py:450-453` (`_cost_ucb_z`) | Risk-aware allocation: charge `z·predicted_dloss_stderr` on top of each AURA cost row (upper-confidence-bound). `0` = bit-identical legacy behavior, and it only bites on the `predicted_dloss` branch (AURA / expert-empirical), never `output_mse`/`weight_mse`. **Research-only, and no driver sets it** — `run-pipeline.sh` never exports it; the only setters in the tree are tests. Its one measured win is confined to the **thin-calibration regime**: on the 27B old-vs-new AURA A/B at thin calib, `z=2` won −8.0%. At *production* calibration the stderr collapses and the hedge buys nothing — 6/252 rows of assignment churn, served parity — so the production-calib decision is **keep at `0`**. Turn it on only when deliberately allocating off a thin/noisy cost run, and re-measure on the serving metric before shipping anything it picked. |
| `--kl-ucb-z` (CLI, not env) | `0` — **RESEARCH-ONLY** | `validate_assignments_kl.py:859` → `:1359`, stamped at `:680` | Selection-side twin of the above: reports `kl_ucb = mean + z·stderr` over `--calib-repeats` alongside `kl_mean/std/stderr`, for `select_validated_frontier --metric ucb` to select on. Same status — **no driver passes it**, `run-pipeline.sh`'s validated-surrogate arm selects on the mean. Same regime caveat: a UCB frontier point is only meaningfully different from the mean point when the repeat stderr is large, i.e. thin calib. |
| `PRISMAQUANT_FISHER_CAP_MULTIPLIER` | unset (off) — **RESEARCH lever** | `allocator.py:1187-1280` (`clip_probe_fisher_outliers`), called from `main` at `:1579`, right after `renormalize_probe_fisher` | Robust Fisher clip: cap each row's finalized `h_trace` at `K × median(h_trace)` over its **role** bucket, rescaling `h_w2_sum` by the same ratio so the derived cost stays consistent (raw accumulators untouched). Unset or empty = byte-identical no-op; a non-finite or `≤ 0` value is a hard error, not a silent skip. Motivation: `predicted_dloss = ½·h_trace·MSE` is *linear* in `h_trace`, so a few heavy-tailed rows can capture the whole DP budget. Role buckets are deliberately the reference tool's grouping (`/home/rob/dq-runs/robust_fisher_clip.py`) — the regex `layers\.<N>\.<one container>\.<role>$`, i.e. dense attention/MLP leaves only; packed/unpacked MoE experts, shared experts and sidecars are skipped, since that is the grouping the result was measured under. **Status: research.** `K=3` measured ~5% better WikiText PPL at 6.0 bpp on Qwen3-4B (2026-05-19); never carried to a served A/B, so promote nothing on it without one. Tests: `tests/test_fisher_normalization.py`. |
| `PRISMAQUANT_FISHER_COL_WEIGHTS` | `0` | `aura_cost.py:853` | Opt-in: `aura_cost` also emits a per-Linear per-column KL-Fisher energy vector (`stats[name]['fisher_col']`, length `in_features`, sums to `h_trace`) alongside the scalar cost. Strictly additive — the rest of the cost payload is bit-identical when off. Feeds `fisher_col_weights.py`. Equivalent to `aura_cost --collect-col-energy`. |
| `PRISMAQUANT_EXPERT_COST_SAMPLE` | falls back to `PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE`, then `0` | `measure_quant_cost.py:426`, fallback `measure_quant_cost.py:427` (both in `_expert_cost_sample_n`, `:421`) | Stratified expert subsample per packed-expert unit in the cost stage; `0` = full stacks. The fallback chain exists so the GGUF lane's older name keeps working. |
| `PRISMAQUANT_SKIP_PACKED_EXPERT_COST` | `0` | `measure_quant_cost.py:1240` | `1` skips the local packed-expert cost measurement entirely — the single most expensive part of the local cost stage. **The pipeline sets it itself** (`run-pipeline.sh:631`) whenever `EXPORT_CONTAINER=nvfp4_cb` and `CB_EXPERT_EMPIRICAL=1`, because stage `[2d-CB]` replaces every packed-expert row wholesale. Do not set by hand unless that replacement is guaranteed to run. |
| `PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE` | `0` | `measure_quant_cost.py:427` | GGUF-lane name for the above subsample; consulted only as the fallback. |
| `PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE` | unset | `measure_quant_cost.py:1614`; set by `scripts/run_dsv4_flash_92gb.sh:288` | Path to the col-weights provenance sidecar written by `synthesize_unrouted_expert_col_weights`. When set, the cost stage emits weight-only rows (`cost_source="unrouted_expert_weight_only"`, `output_mse_measured=False`) for EXACTLY the never-routed experts the sidecar declares — the declared class narrows the `allocator.py:2427` no-silent-holes refusal without weakening it: a missing row for a ROUTED expert still refuses. Unset → no emission, gate unchanged. |
| `PRISMAQUANT_COST_MAX_ACT_ROWS` | `0` (all available rows) | `measure_quant_cost.py` (row-bucketed batched mode) | Optional cap on activation rows used per Linear in the cost stage's `output_mse` measurement. `0` uses every cached row (up to the collection cap). Introduced with the per-row-count bucketing that replaced the chunk-minimum truncation; every cost row records its `n_activation_rows`. |
| `PRISMAQUANT_COST_FAIL_FAST` | **`1` (on)** | `measure_quant_cost.py` (`_measure_spec_into_accum`) | On a measurement exception: print a `[cost] FATAL:` line with traceback, stamp the rows `cost_measurement_failed`, and abort the shard; the shard merge additionally refuses stamped rows (`SystemExit(2)`). `0` restores the old swallow-and-continue behaviour for triage only — the merged-table gate still refuses the stamped rows. |
| `PRISMAQUANT_ACTIVATION_FAIR_PRICING` | **`1` (on)** | `activation_fair_pricing.py` (`env_enabled`), calibrated once per run in `allocator.py` before `build_candidates`; pipeline knob `ACTIVATION_FAIR_PRICING` (`run-pipeline.sh`) | Ultraplan P5a. The cost precedence prices the W4A4-vs-W8A8 activation contract **only** on the measured `output_mse` branch, and the two rows above are what remove that branch from most of a production run: `PRISMAQUANT_EXPERT_COST_SAMPLE` makes `measure_quant_cost` stamp `output_mse_measured=False` on every packed-expert row, and `PRISMAQUANT_CB_LADDER_INTERP=1` fills interpolated rungs with `output_mse_measured=False` too. On those rows NVFP4-CB is credited with its cheaper index stream and charged none of its A-side cost (measured in the retired Gridbook lane's `docs/audits/ultraplan_perf_2026-08-01.md` §6, asymmetry 1; the finding is about weight-only pricing and outlives the lane). The allocator now fits ONE per-format-family correction per run — the geometric mean of `measured_dloss / weight_only_dloss` over the rows that carry BOTH estimators — and multiplies it into the weight-only-priced rows of that family. Multiplicative, so it cannot lift an exactly-0.0 price off the DP's global minimum (the `activation_cost_unmeasured` candidate removal keeps full strength). **It CAN reorder rungs inside a family, and did (2026-08-07):** the constant is uniform only where the family's rungs all take one pricing branch, and a CB ladder mixing measured rungs with band-interpolated ones does not — on the DSv4-Flash `nvfp4_cb` menu that mispriced K13/K14/K17 ~12x high on down_proj and ~1.6x low on gate/up_proj, because the family constant (112.5) averages a per-projection `output_mse/weight_mse` ratio spanning 9.4–320. Fixed by pricing a band-interpolated row from its own banked `output_mse` — the tensor's holdout-gated ladder fit, already in output space — instead of `weight_mse x` the constant. Such a row still declares `cost_source: band_interpolated` / `output_mse_measured: false`, is still barred from the calibration sample (its output_mse is derived; admitting it would be circular), and is stamped with its own branch label `interpolated_output_mse` so the artifact can still say which selected prices were predictions. A `band_interpolated` row whose `output_mse` is 0.0 (the packed-expert ladder path, which fits in weight space only) keeps the weight-only branch. **Fail-closed:** if one family calibrates while another activation-quantizing family still has uncorrected weight-only rows, the run refuses (`AssertionError` naming this flag) rather than ship a half-corrected menu; if NO family has measured rows, nothing is corrected, the verdict is printed and stamped, and pricing is unchanged. `0` reproduces pre-0.5.3 pricing bit-for-bit and also suppresses the refusal — for bisecting an allocation change, not for tuning. Fit, calibration sample (bounded + sha256 of the full list), residual band and per-rung dependence are stamped into `format_applicability.json` and `selection.json`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` | **archived** | `allocator_candidates.py:253` | Historical Fisher row-weighted allocator objective. The production pipeline rejects it; archive context lives under `archive/fisher_2026-05-15/`. |
| `PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP` | falls back to `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP`, then `64` | `measure_quant_cost.py:248`, fallback `measure_quant_cost.py:249` | Historical cap for Fisher output-MSE allocation; not used by the production pipeline. |

## 2. Export flags

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_BATCHED_NVFP4_EXPORT` | **on** (when act-aware passes fire and an activation cache is supplied) | `export_native_compressed.py:5605`; fingerprint `:5472` | Routes NVFP4 same-shape Linears through the batched GPTQ / optional scale_sweep path (`export_batched_gptq.py`). Stacks per-layer experts into `(E, out, in)` tensors and runs Cholesky / column update batched across E. |
| `PRISMAQUANT_BLOCK_OUTPUT_MATCH` | **ARCHIVED 2026-07-30** — setting it truthy is a hard `SystemExit` | gate `export_native_compressed.py::_refuse_archived_block_output_match`; fingerprint records the constant `"archived_2026-07-30"` | Quality lever #12 (`block_output_match.py`), walled under `archive/block_output_match_2026-07-30/` by re-vet **R25** (closes D16 as *unreachable*, not unmeasured). It ran a greedy `{0.95, 1.0, 1.05}` per-Linear gain search against an FP32 reference **block** output. Three reasons: (1) **it never executed** — the production-cache pack `continue`s first, so with `PRODUCTION_CACHE=1` no dense NVFP4 Linear reached the branch (0 hits in two real production export logs); (2) had it run it would have re-derived NVFP4 group scales outside `_export_match_render_scale_rule`, discarding the render's `joint_mse` scales — the −6.6% KL defect **M19** fixed everywhere else; (3) a per-tensor gain re-search *after* JSO already solved the scale, wrapped in `except Exception → WARN` so failures were invisible. Its "~0.05–0.10 PPL" docstring was a pre-JSO expectation, never a measurement. **`0` and unset both pass** (they asked for what now always happens); any other value refuses so an old launcher fails loudly rather than exporting differently in silence. |
| `PRISMAQUANT_NVFP4_SCALE_RULE` | `static_6` | `export_native_compressed.py:190` (const `:107`); `incremental_measure_quant_cost.py:287` | NVFP4 local block-scale rule. `static_6` is standard NVFP4 max-to-6 scaling. `four_over_six_mse` tries max-to-6 and max-to-4 per 16-value block. `joint_mse` is the production JSO scale rule selected by the `joint_scale_opt` lever: it chooses from `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` under the served FP8-snapped scale objective. All preserve the compressed-tensors NVFP4 schema and vLLM kernel. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS` | `6,4` | `export_native_compressed.py:298` | Candidate max-to-levels for NVFP4 joint scale optimization. Extend only for explicit JSO ablations; production defaults use the validated `{6,4}` grid. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_OPT` | `0` | `export_native_compressed.py`; `production_weight_cache.py:2162` | Direct JSO switch for callers that bypass `PRODUCTION_CACHE_LEVERS`. |
| `PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID` / `PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO` / `PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI` | `5` (clamped to 1…33) / `0.75` / `1.25` | `export_native_compressed.py:591` / `:599` / `:603` | Per-tensor global-scale search grid for the JSO fused joint pre-pass: `GRID` is the number of candidate globals scored after FP8 realization of the group scales (`≤1` short-circuits to the base global), `SPAN_LO`/`SPAN_HI` are the multiplicative bracket around it. Non-finite or non-positive `SPAN_LO` falls back to `0.75`; a `SPAN_HI` below `SPAN_LO` falls back to `max(SPAN_LO, 1.25)`. |
| `PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING` | `0` | `export_native_compressed.py:359`; fingerprint `:5466` | Research lever: score NVFP4 scale candidates under the FP8-snapped effective scale with a per-tensor global fixed point. More serve-faithful in principle but changes shipped NVFP4 bytes for `joint_mse`/`four_over_six` — default OFF pending a served gold-metric A/B. Recorded in the export fingerprint. |
| `PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE` | **on** | `export_native_compressed.py:1445` | M19: NVFP4 export re-derives block scales from the cached production render using the SAME scale rule the render chose (`joint_mse` under JSO), instead of re-quantizing the bf16 dequant with `static_6`. Served-validated −6.6% KL / −3.3% PPL on the 4B paired A/B. Since the 2026-07-02 audit (M2) this also covers packed-expert re-pack and the fused joint-global pre-passes (rule = the cache's recorded `nvfp4_scale_rule` lever; env default when nothing is recorded). `0` reproduces pre-M19 artifacts. |
| `PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES` | `0` | `perturbed_x_cache.py:108` | Perturbed-X emulation hooks: `1` quantizes NVFP4 activations with the SERVE-faithful two-level semantics (static per-tensor `input_global_scale` derived from the calibrated max_abs — honoring `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` — plus FP8 snap of each 16-group block scale, including block-zeroing and above-calibration-amax clipping) instead of the dynamic exact-fp32-scale RTN. Closes the audit M18-residual/C1 measurement gap; default off pending a served correlation study. |
| `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE` | `0` | `export_native_compressed.py:888` | C1 (2026-07-02 audit): `1` switches NVFP4 `input_global_scale` to the compressed-tensors/vLLM `generate_gparam` convention `448·6/amax`, placing serve-time FP8-stored activation block scales in (0,448] instead of the legacy (0,1] — rescues blocks ≫64× below calibration amax from FP8 subnormals, but CLIPS any serve block whose amax exceeds calibration amax. Served A/Bs (byte-identical weights, only this scalar ×448, 2 window draws each): 35B MoE frontier **−14.1% KL (win)**; 27B regen dense **+37.5% (loss)**; LFM2.5 thin-calib smoke +5.8% (loss). Strongly artifact-dependent → default stays legacy; opt in per artifact behind a served A/B. The scale is a free post-export knob (in-place patch, no re-render) — see `/home/rob/dq-runs/c1-igs-ab-20260702/patch_igs.py`. |
| `PRISMAQUANT_GPTQ_BLOCK_SIZE` | `128`, via `PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE` fallback | `export_native_compressed.py:1793`, fallback `export_native_compressed.py:1794` | Column block size for the FP-Quant-style GPTQ OBS update across NVFP4, FP8_DYNAMIC/FP8_E4M3, and explicit MX research formats. Quantizer scales are fixed before the solve; each column is quantized and its error is propagated through the current GPTQ block and later blocks. |
| `PRISMAQUANT_GPTQ_DAMP` | `""` (fixed 1.0) | `export_native_compressed.py:1872`; fingerprint `:5464` | Overrides the fixed GPTQ damping constant. |
| `PRISMAQUANT_GPTQ_DAMP_SWEEP` | **`0`** (one reader) | `export_native_compressed.py:1857`, fingerprint | Damp sweep is OFF for production render/export (fixed damp 1.0, 2026-06-12); `1` reproduces historical artifacts. **D5 is fully closed:** the second reader with the opposite default lived in `kl_sensitivity_probe`, was made a delegation to `production_weight_cache._resolve_production_render_levers` on 2026-07-30, and the file itself was walled the same day with the L3 cascade (`archive/l3_propagated_2026-07-30/`, re-vet R4). One reader, one default; the contract is pinned by `tests/test_production_weight_cache.py`. |
| `PRISMAQUANT_GPTQ_DAMP_ROLES` | unset | `export_native_compressed.py:1939` | Per-role GPTQ damp override, e.g. `qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0`. Default-off research lever (the 2026-06-22 per-role served A/B was NULL; fixed damp 1.0 is final). Unlisted roles keep the fixed damp. |
| `PRISMAQUANT_GPTQ_STATIC_ACT_ORDER` | `0` | `export_native_compressed.py`; `production_weight_cache.py:2158` | Direct static-act-order switch for callers that bypass `PRODUCTION_CACHE_LEVERS`. |
| `PRISMAQUANT_DAMP_ANALYTICAL` / `PRISMAQUANT_DAMP_ANALYTICAL_C` | `""` (off) / `1.784e-5` | `export_native_compressed.py:3057` / `:3061` | Archived closed-form damp (refuted: +100–161% KL vs the discrete sweep). `PRISMAQUANT_DAMP_ANALYTICAL` accepts `1`/`true`/`yes`/`on`/`kappa_target`; `_C` is the fitted constant in `damp = C · λ_max / mean(diag H)` (fitted on Qwen3-4B's 450 logged damp-sweep winners) and is only read when the outer flag is on. Kept for reproduction only. |
| `PRISMAQUANT_DAMP_SWEEP_LOG` | unset | `export_native_compressed.py:2178` | Per-Linear damp-sweep decision log. |
| `PRISMAQUANT_ACT_CLIP_QUANTILE` | `0.999` | `export_native_compressed.py:681`; fingerprint `:5468` | Calibrated activation-max clip quantile. |
| `PRISMAQUANT_FP8_SCALE_SWEEP_FACTORS` | `0.25 … 2.0` (9 log-spaced) | `export_native_compressed.py:3543` | Candidate FP8 scale multipliers for the explicit `scale_sweep` ablation. |
| `PRISMAQUANT_ALLOW_PACKED_EXPERT_RTN` | `0` | `export_native_compressed.py:1663`, enforced `:6113-6127` | Research/A-B escape hatch: allows non-BF16 packed-MoE experts to skip the production-cache GPTQ render and export RTN bytes. Never use for a production artifact. |
| `PRISMAQUANT_EXPORT_INLINE_EXPERT_GPTQ` | `0` | `export_native_compressed.py:1674` | 295B-class alternative when no dequant cache can exist: run expert GPTQ inline during export. |
| `PRISMAQUANT_ALLOW_UNSCALED_FP8` | `0` | `layer_streaming.py:417` | Streaming-load guard override: by default a float8-dtyped checkpoint tensor with no entry in the fp8 scale-inv map fails fast (loading raw FP8 codes as if they were weights is the historical ±448-range corruption). Set `1` to permit the raw cast anyway (debug only). |
| `PRISMAQUANT_EXPERT_LAZY_FILL` | **on** | `validate_assignments_kl.py:337` | M4 frontier expert selection: `validate_assignments_kl` lazily renders a Pareto point's missing packed-expert entries (e.g. FP8) into the shared frontier cache just before scoring, on the BUILD/render calib split, then re-pickles the cache so recache/export ship the same bytes real KL selected. The format-menu build eager-renders only the NVFP4 rung. `0` restores the legacy hard-fail on expert cache misses. |
| `PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE` | **conditional** — defaults ON when a production weight cache is supplied *or* `PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT` is set | `kl_measurement.py:1112`, default computed `kl_measurement.py:1104-1110` | Coverage guard for assignment-required cache entries in the **KL hook path** (not, despite the older wording here, the exporter). Missing non-BF16 renders fail early instead of falling through to RTN. |
| `PRISMAQUANT_STRICT_PRODUCTION_CACHE` | **on** | `perturbed_x_cache.py:579` (decides; also `:844` / `:1011`), `layer_state_cache.py:489`, `weight_session.py:82`, `validate_assignments_kl.py:478` (helper default `True`, `memory_management.py:18`) | KL/activation-cache residency guard. Missing required production-cache weights fail fast by default; set `0` only for explicit legacy/non-production fallback runs. (No longer read in `kl_measurement.py`: the guard moved out with the 2026-07-30 L3 split.) |
| `PRISMAQUANT_DO_NO_HARM` | **on** | `export_native_compressed.py:4209`, `:4350`, `:4492`, `:4797`; fingerprint `:5460` | Enables export-time GPTQ-vs-RTN do-no-harm gates where supported. Failures and reverts are counted in export provenance. |
| `PRISMAQUANT_DO_NO_HARM_VERBOSE` | unset | `export_native_compressed.py:4231`, `:4372`, `:4514`, `:4831` | Per-Linear do-no-harm decision log. |
| `PRISMAQUANT_RENDER_PROGRESSIVE_GATES` | **on** | `production_weight_cache.py:1831` | Production-cache render gate for local mechanisms. All formats use the same progressive order, while unsupported mechanisms are format-gated off. NVFP4 can score FourOverSix, static activation ordering, GPTQ, joint-scale optimization, and optional scale_sweep candidate packages; FP8_DYNAMIC/FP8_E4M3 can score GPTQ with damp sweep, and can additionally score explicit scale_sweep when enabled. MXFP8 remains explicit opt-in and can score static activation ordering plus GPTQ using the canonical E8M0 scale rule. Regressive candidates keep the prior accepted render. Cache metadata records decisions in `render_gates`; FourOverSix has a compact `four_over_six` summary. |
| `PRISMAQUANT_RENDER_GATE_MIN_GAIN` | `0.0` | `production_weight_cache.py:1495` / `:1905` | Minimum relative gain required by the progressive render gate (reason string `below_min_gain`, `render_score.py:122`). Keep at `0.0` for normal runs so tiny local improvements can accumulate; raise only for ablations. |
| `PRISMAQUANT_TARGET_PROFILE` | unset | `export_native_compressed._allocator_target_profile_for_audit` | Serving-profile **override** for direct exporter invocations. Since re-vet R11 the exporter's normal channel is the allocator's stamp in `layer_config.json`'s reserved `__prismaquant__` block, so the pipeline sets nothing; this env var still wins when set. |
| `PQ_EXPORT_VECTOR_CHUNK` | `auto` (cap 128) | `export_native_compressed.py:4878` | Upper bound on the grouped-export vectorization chunk. **Note the `PQ_` prefix** — the only non-`PRISMAQUANT_` flag in the exporter. |
| `PRISMAQUANT_FISHER_WEIGHTED_GPTQ` | **archived** | `production_weight_cache.py:2169` | Fisher-weighted GPTQ is archived under `archive/fisher_2026-05-15/` and rejected by the production pipeline. |
| `PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP` | `64` | `export_native_compressed.py:784`; `render_score.py:38` / `:55` | Historical Fisher-weighted GPTQ row-weight cap; not used by the production pipeline. |
| `PRISMAQUANT_MXFP8_JOINT_SCALE_SHIFTS` | `-1,0` | `export_native_compressed.py:3081` | Candidate E8M0 exponent shifts for the MXFP8 JSO block search. Live code, but only reached when MXFP8 is rendered with `joint_scale_opt=True`; `joint_scale_opt` is registered as an `nvfp4_scale_optimizer` mechanism (`render_score.py:349-358`) and is not offered to MXFP8 under any production lever set, so in practice it never fires. |
| `PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS` | `0` | `export_native_compressed.py:2917` | Explicit-ablation candidate E8M0 exponent shifts for MXFP8_E4M3 activation-weighted scale search. The default is a no-op; nonzero shifts are experimental and refine the current accepted render under the same progressive gate. |

## 3. Pipeline production-cache flags

These are `prismaquant/run-pipeline.sh` environment variables rather than
`PRISMAQUANT_*` flags. Research levers outside the current production recipe
live under `archive/`; the pipeline fails fast (`exit 2`) when archived Fisher
levers or cost modes are requested.

| env var | default | what it does |
|---|---|---|
| `PRODUCTION_CACHE` | `1` | Build and use a `ProductionWeightCache` so export packs the same rendered weights that KL/polish measured. |
| `PRODUCTION_RECACHE` | `1` | Replay calibration with production weights installed and re-fit `activation_max_abs` before export. |
| `PRODUCTION_CACHE_LEVERS` | `gptq,static_act_order,joint_scale_opt` | V1 production render levers. FP8_DYNAMIC/FP8_E4M3 uses GPTQ without static ordering or JSO because the served representation is per-row scaled FP8 dynamic. `static_act_order` applies to production microscaling GPTQ formats: NVFP4, MXFP4, and explicit MXFP8. `joint_scale_opt` applies only to NVFP4. MXFP4/MXFP8 use the canonical E8M0 scale rule when explicitly requested. `scale_sweep` remains available for explicit ablations but is not a default. Runtime activation scores use their served activation quantizers; NVFP4 is the only current score path that applies the calibrated activation-max clip. |
| `FORMATS` | `NVFP4,FP8_DYNAMIC,BF16` (`run-pipeline.sh:45`) | Allocator format menu. MXFP8 is de-menued for inference — exact-scale FP8 Pareto-dominates it. |
| `TARGET_BITS` | `4.75` | Allocator bit budget over quantizable parameters. |
| `TARGET_PROFILE` | **unset** (re-vet R11) | Serving profile. Left unset so the architecture's own `spec.default_serving_profile` wins — a shell default silently overrode every spec (`resolve_target_profile` gives explicit requests precedence), which cost 226 Hy3 FP8 Linears silently coerced to BF16 on 2026-07-11. An explicit value still wins, so every in-tree launch script is unchanged. The resolved profile is stamped into `layer_config.json` and read by the exporter. |
| `ALLOW_PINNED` | unset | Comma-separated qname substrings forwarded as `allocator --allow-pinned`, lifting `ModelProfile.is_pinned_name` so the DP places those units by budget-value instead of force-excluding them at source dtype. Empty = historical behaviour, byte-identical. The allocator enforces the preconditions (cost row + probe `n_params`) and refuses rather than pricing an unpriced unit at zero. Matters at card scale: a BF16 `lm_head` is 2.543 GB on Qwen3.8-27B, 20% of a 13.0 GB budget. A quantized pinned name also needs render/pack/serve support — `lm_head` on the native lane. `embed_tokens` was quantizable only on the retired Gridbook lane (via its `quantized_embedding`), so as of 2026-09-02 no lane can quantize it at all. |
| `TARGET_PROFILE_DEFAULT` | `vllm_packed_moe` | Fallback passed to the allocator as `--target-profile-default` for architectures whose spec declares no serving profile. Never `research` (unbounded format menu). |
| `TARGET_DISK_GB` | unset | Byte budget in decimal GB (re-vet R1, closes D12). When set: **overrides `TARGET_BITS`**, narrows the Pareto sweep to the ~3 byte-feasible rungs, and flips `SELECTION_MODE` to `validated-surrogate` + `VALIDATED_FRONTIER_PICK` to `budget` (explicit values still win). The card is the constraint; measured KL is the objective. |
| `FISHER_WEIGHTED_GPTQ` | archived | Any truthy value is rejected; archive context lives under `archive/fisher_2026-05-15/`. |
| `FISHER_OUTPUT_MSE_ALLOCATOR` | archived | Any truthy value is rejected; V1 allocation uses the non-Fisher objective plus measured frontier validation. |
| `COST_MODE` | **`aura`** (flipped 2026-07-30, re-vet R2) | The documented **spelling** over the two axes `COST_RENDER` × `COST_OBJECTIVE` (re-vet R3, §4.7 of ARCHITECTURE.md); the three values keep their exact meanings. `aura` (default) runs the AURA downstream-KL-adjoint cost (`aura_cost.py`, served −38%/−39.5% confident-KL @4B across two calibrations, −17.9% @27B, both flagships) against a production-rendered dW cache; on packed-MoE models the route-flip-blind smooth cost is replaced for experts by measured empirical unit-KL (`expert_empirical_cost.py`, FP8 kept in the menu) merged into one hybrid payload, with MTP/visual sidecar rows backfilled from the baseline cost. `production-render-score` renders the full `FORMATS` menu through `ProductionWeightCache` and writes an allocator cost from the recorded render scores — the **explicit/legacy** spelling that reproduces every pre-flip artifact. `local` keeps the inline weight-recon measurement. The CB/GGUF lanes are **no longer restricted to `local`**: the old gate named a render property to block an objective, and is replaced by the render-faithfulness assertion (a cached-menu render on those lanes is built `--col-weights`, CB Milestone C). Non-`local` objectives there are reachable but OPT-IN and not recommended pending a served CB A/B. Every producer stamps `provenance["cost_mode"]` (`--cost-mode`, R2 precondition (i)) and reuse of `cost.pkl` is conditional on it matching: `cost.pkl` is the same path under every mode, so a mode change used to silently allocate on the previous estimator. A mismatch rebuilds loudly; an unstamped (pre-R2) table warns and is reused. `grouped-kl`, `production-render-staged`, `fisher`, `hdq` and `multi-shot` are **archived** and `exit 2`. `production-render-staged` was walled 2026-07-30 (`archive/production_render_staged_2026-07-30/`, re-vet R17): it rendered NVFP4 first and offered promotion formats only to the top-30% error tail, so on 27B its last-token-KL screen improved (0.0232 vs 0.0280) while direct WikiText PPL regressed (10.83 vs 8.33) — "Do not ship". |
| `COST_RENDER` / `COST_OBJECTIVE` | derived from `COST_MODE` | The mechanism under the spelling (re-vet R3): `COST_RENDER ∈ {inline, cached-menu}` is WHICH render produces the per-`(Linear, format)` error; `COST_OBJECTIVE ∈ {weight-recon, render-score, aura-adjoint}` is how that error becomes `predicted_dloss`. Setting the pair directly is equivalent to the alias; setting both spellings at once is `exit 2`; the two unimplemented pairs (`inline × aura-adjoint`, `cached-menu × weight-recon`) stop with the reason. |
| `AURA_ADDITIVITY_GATE` | **`measure`** (flipped 2026-07-30 by Robert's ruling on the R2 residue) | Stage `[3c]`, the AURA trust-region report (R2 precondition (ii)): `residual = measured_end_KL − Σ predicted_dloss`, stamped into `cost.pkl` `provenance["additivity"]`. `measure` (default) reports from a measured KL the run already produced when there is one (validated-surrogate's frontier JSON, free) and otherwise runs **one bounded KL eval** of the final assignment against the same dW cache AURA costed on (`AURA_COST_NSAMPLES × AURA_COST_SEQLEN`) — so every AURA-default run performs the measurement and every artifact carries a **real residual** rather than a prediction. `auto` is the pre-ruling, zero-added-GPU behaviour: report only from a measurement the run already made, else record the predicted sum with `measured_kl: null` and a status naming why. `0` disables. Non-blocking in every mode — it never changes an allocation. |
| `AURA_COST_NPROBES` / `AURA_COST_NSAMPLES` / `AURA_COST_SEQLEN` / `AURA_COST_CALIB_SEED` | `32` / `8` / `128` / `42` | `COST_MODE=aura` probe/calibration volume (defaults = the regen-27b recipe). Also `AURA_COST_LINEAR_CHUNKS` (8), `AURA_COST_PROBE_MICROBATCH` (8), `AURA_COST_MIN_FREE_GIB` (18). |
| `AURA_COST_DTYPE` | `auto` | Resident model dtype for the aura_cost stage. `auto` sizes the checkpoint (fp8 sidecars counted at 1 byte/param) and picks `float32` (additivity-preferred, the 27B regen regime) only when params×4 bytes + `AURA_COST_MIN_FREE_GIB` fits in MemAvailable, else `bfloat16` (35B-class — fp32 is ~140 GiB against the 121 GiB pool and OOM-kills the box). |
| `AURA_COST_STREAMING` / `AURA_COST_CHECKPOINT_DIR` | `0` / unset | Stream and durably resume the AURA path. Enabling requires one absolute checkpoint directory and passes `--streaming --checkpoint-dir … --resume` to both the smooth adjoint and its empirical routed-expert tail. The expert stage owns the deterministic `expert-empirical-cost/` child so its manifest cannot collide with the adjoint manifest. Both existing value-bearing source/cache identity gates remain fail-closed. |
| `AURA_EXPERT_NSAMPLES` / `AURA_EXPERT_SEQLEN` | `16` / `512` | `COST_MODE=aura` empirical packed-expert unit-KL stage calibration volume (the 35B arm-E recipe). |
| `VALIDATED_FRONTIER_MATERIALIZATION` | `hooks` | How the validated frontier materializes each Pareto point. `hooks` = all points in one process (fast), but the pipeline admits it only when safetensors headers prove the checkpoint is dense and below 35B parameters; MoE, ≥35B, and unclassifiable checkpoints fail closed before GPU work. `inplace` = one `validate_assignments_kl` process per point, per-point JSONs merged for selection, and is required for those large/MoE campaigns. |
| `PRODUCTION_RENDER_COST_NSAMPLES` / `_SEQLEN` / `_SEED` | `8` / `1024` / `42` | Calibration contract for `COST_MODE=production-render-score`, using the production cache scorer. |
| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `weight_mse` | Render-score field used as allocator cost. `weight_mse` (default since 2026-07-02, audit M6) routes through the dimensionally-consistent `h_trace × weight_mse` path — the legacy `output_mse` product double-counted activation energy E‖x‖² (h_trace already contains it), a per-Linear bias ∝ in_features·x_rms². Served two-arm A/B at matched 4.75 bpp: Qwen3-4B KL −50.8% / 32k-PPL −15.1%; Qwen3-0.6B KL −58.5% / 32k-PPL −24.4% (5 window draws each, same pipeline seeds). `output_mse` reproduces the historical objective; `score_sum`/`score` are ablation fields. 27B-class confirmation = ladder debt. |
| `SELECTION_MODE` | `surrogate`, or `validated-surrogate` when `TARGET_DISK_GB` is set | `surrogate` preserves the normal allocator-selected `TARGET_BITS` assignment. `validated-surrogate` writes allocator Pareto assignments, builds a format-menu production cache, measures real assignment KL for each Pareto point, selects the measured KL/bpp kneedle with `prismaquant.select_validated_frontier`, then recaches and exports the selected assignment. The flagship artifacts used `validated-surrogate`; it is not the default. |
| `VALIDATED_FRONTIER_NSAMPLES` / `_SEQLEN` | `$NSAMPLES` / `$SEQLEN` | Calibration size for measured-frontier KL selection. Keep these at the artifact validation contract for 27B decisions; lower values are smoke-only. |
| `VALIDATED_FRONTIER_PICK` | `kneedle`, or `budget` when `TARGET_DISK_GB` is set | Selection rule for `SELECTION_MODE=validated-surrogate`: `budget` (min measured KL among rows whose tensor payload plus operator-supplied non-tensor reserve fits `--target-disk-gb`, followed by the exporter's exact recursive-file hard gate; re-vet R1), `kneedle`, `best-kl`, `lowest-bpp`, `practical-knee`, `saturation`. Kneedle is axis-dependent on a log-linear RD curve — under a card, `budget` is the ship rule and kneedle is the diagnostic. |
| `select_validated_frontier --tail-veto` | **`kl_max`** (DEFAULT-ON since 2026-07-30; CLI flag, no env) | D1 tail veto (re-vet R9), ruled by Robert with `kl_max` — the worst sequence — as the **contract statistic**: a row that improves mean KL enters the frontier only when `row[kl_max] <= incumbent[kl_max] * (1 + tail_eta)`. `kl_max` is the statistic that would have caught the broken 27B that passed on the mean while 80% of its prompts were bad; `nll_p99` is still recorded on every row, so changing the contract is a flag, not a re-measurement. Choices `{none,kl_p99,kl_max,nll_p99}`; `none` restores the pre-R9 envelope byte-for-byte. Safe on by default because the failure is one-sided — a spurious veto only makes the pick MORE conservative — and never silent: refusals are printed and kept under `vetoed_rows` with a `veto_reason`. A pre-R9 validation JSON carries no tail column, so the veto goes **inert with a warning** instead of emptying the frontier. |
| `select_validated_frontier --tail-eta` | **`auto`** (derived; CLI flag, no env) | Slack on the tail veto. `auto` derives it (house rule 2) as the incumbent row's **relative stderr of the tail statistic across calibration repeats** (`std/√n ÷ mean` over the `<column>_repeats` emitted by `validate_assignments_kl` at zero extra forward cost), floored at 0 — a candidate inside the tail's own measurement noise is admitted, one outside it is a real regression. A **single** repeat has no spread: `auto` degrades to a strict `0` and prints a warning (single-seed tails are noisy; a +10% reading has flipped to −5.2% across repeats — run validation with `--calib-repeats ≥ 4`). An explicit number always wins. Derivation documented at `select_validated_frontier.tail_eta_auto`. |
| `VALIDATED_SOURCE_PREFETCH` | `require` (`run-pipeline.sh:282`) | Source-checkpoint residency gate for the validated-frontier stages (`source_prefetch.py`). `require` fails fast rather than silently becoming NVMe-bound. |
| `PRODUCTION_CACHE_LRU_GB` | `64.0` | Resident tensor budget for disk-backed production-cache use in recache and export. The 27B n=8 recache smoke needed `45.32 GiB` for the selected assignment. |
| `PRODUCTION_CACHE_PREFETCH` | `require` | Standalone recache prefetch policy. `require` fails fast when assignment-required weights cannot fit resident. |
| `EXPORT_PRODUCTION_CACHE_PREFETCH` | `require` (native lane) | Export-side assignment prefetch policy (re-vet R24, closes D8) → `export_native_compressed --production-cache-prefetch {require,warn}`. `require` fails the export when the cache cannot supply a layer's assignment instead of silently degrading to per-tensor NVMe reads; the bare-CLI default stays `warn`. The CB/GGUF lanes read no production cache. |
| `PRODUCTION_CACHE_PREFETCH_WORKERS` | `4` | Thread count for eager production-cache prefetch. |
| `EXPORT_CONTAINER` | `compressed-tensors` | `gguf` and `tessera` switch stage 4 to their own exporters and impose gates (see §8, §8a). The vocabulary is the sanctioned three; `nvfp4_cb` was the fourth until 2026-09-02 and now `exit 2`s (§5). |

## 4. CUDA / system flags

| env var | recommended | read at | what it does |
|---|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | set by `incremental_probe.py` at module load | Required on UMA hardware (DGX Spark) to keep the CUDA caching allocator from hoarding freed blocks. |
| `PRISMAQUANT_KL_CUDA_GRAPHS` | `auto` | `kl_measurement.py:1138` | Graphs assignment-KL validation only for larger calibration batches. Default threshold `16`; override with `PRISMAQUANT_KL_CUDA_GRAPHS_MIN_CALLS` (composed as `f"{name}_MIN_CALLS"`, `kl_measurement.py:107`). |
| `PRISMAQUANT_VALIDATION_CUDA_GRAPHS` | **on** | `validation_harness.py:53` | Graph capture in the validation harness. |
| `PRISMAQUANT_KL_CUDA_GRAPH_CACHE_SIZE` / `PRISMAQUANT_VALIDATION_CUDA_GRAPH_CACHE_SIZE` | `4` each | `kl_measurement.py:871`; `validation_harness.py:45` | Per-registry graph-cache capacity (registry `max_entries_env` parameter). |
| `PRISMAQUANT_CUDA_GRAPH_MAX_ENTRIES_PER_PATH` | `4` | `kl_measurement.py:465` | Global fallback capacity when a registry has no dedicated cache-size env set. |
| `PRISMAQUANT_KL_CUDA_GRAPHS_VERBOSE` | unset | `kl_measurement.py:872` | Per-capture logging for the assignment-KL registry. |
| `PRISMAQUANT_GRAPH_SHARED_POOL` / `PRISMAQUANT_GRAPH_OUTPUT_CLONE` / `PRISMAQUANT_GRAPH_AUDIT` | on / on / unset | `kl_measurement.py:341` / `:337`; `memory_management.py:242` | Graph memory-pool sharing, output cloning, and capture auditing. Note the coupling enforced at `kl_measurement.py:337-351`: with the shared pool on, `PRISMAQUANT_GRAPH_OUTPUT_CLONE=0` is unsafe and is ignored with a warning. |
| `PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES` / `PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION` | `400` / `0.05` | `perturbed_x_cache.py:1034`; `kl_measurement.py:898` | Frozen-weight cache capacity, and the fraction of the GPU budget that must stay free before `_maybe_disable_frozen_weight_cache_for_memory` turns whole-assignment frozen-weight caching off. (`PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB` was listed here with no reader anywhere in the tree — see §9.) |
| `PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE` | **on** | `kl_measurement.py:1084` (written to `0` by `validate_assignments_kl.py:1194` under `--disable-frozen-weight-cache`) | Enable the frozen-weight cache in assignment-KL measurement. |
| `PRISMAQUANT_GPU_MEM_RESERVE_GB` / `_FRACTION`, `PRISMAQUANT_HOST_MEM_RESERVE_GB` / `_FRACTION`, `PRISMAQUANT_MAX_GPU_MEM_GB` | unset | `memory_management.py:144` / `:147` / `:156` / `:159` / `:180` | Memory-budget reserves used by the resident-fit calculations. |
| `PRISMAQUANT_UMA_MEMORY_INFO` | `auto` | `memory_management.py:98` | Treat GPU+host as one physical pool (DGX Spark). |
| `PRISMAQUANT_FULL_SEQUENCE_KL` | `0` | `kl_measurement.py:75` (`resolve_kl_scope`) | Score all positions instead of the last-token hook screen. Legacy env override only: an explicit `kl_scope=` argument always wins. |
| `PRISMAQUANT_DETERMINISTIC` | `0` | `build_production_cache.py:505` | Deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG` for the render path. |
| `PRISMAQUANT_MASK_CUDA_DURING_META_INIT` | `1` | `streaming_model.py:114` | Hide CUDA during meta-device model construction. |
| `PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT` | unset | `perturbed_x_cache.py:799` / `:1099` | Declares that weights are staged by an external owner (`weight_session`); also flips the `STRICT_ASSIGNMENT_COVERAGE` default to ON. (Its other reader, `kl_sensitivity_probe`, was walled 2026-07-30 — `archive/l3_propagated_2026-07-30/`.) |
| `PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE` | unset | `perturbed_x_cache.py:470` / `:821` | Share one rendered-format cache across perturbed-X passes. |
| `PRISMAQUANT_PROD_ACT_SCALES` | unset | `perturbed_x_cache.py:91`; `production_weight_cache.py:1286` | Use production activation scales in the emulation hooks. |
| `PRISMAQUANT_FUSED_KERNEL_NVFP4` / `_OVER_PROD_CACHE` | unset | `perturbed_x_cache.py:865` / `:875` | Route emulation through the fused NVFP4 kernel, optionally in preference to the production cache. |
| `PRISMAQUANT_NVFP4_FUSED_JIT_WARMUP` | unset | `kernels/nvfp4_fused.py:365` | Pre-JIT the fused NVFP4 kernel. |
| `PRISMAQUANT_DISABLE_RTN_COMPILE` | `""` | `format_registry.py:479` | Pin the RTN quantize/dequantize hot path to eager instead of `torch.compile`. |
| `PRISMAQUANT_IQ_COMPILE_SWEEP` | `1` | `gguf_iq_formats.py:181` | `torch.compile` the GGUF IQ-quant sweep kernels. |
| `PRISMAQUANT_TMPDIR` | unset | `sensitivity_probe.py:88` (then `TMPDIR`, `sensitivity_probe.py:89`) | Scratch directory for probe temporaries. **Never point either at `/tmp`** — see the operational landmines. |
| `PRISMAQUANT_VALIDATION_PROD_CACHE` / `_DIR` / `_LRU_GB` | unset / unset / `16` | `validation_harness.py:303` / `:308` / `:311` | Production cache wiring for the validation harness. |
| `PRISMAQUANT_VALIDATION_SKIP_END_KL` | `0` | `validation_harness.py:248` | Skip end-KL in the harness (smoke only). |
| `PRISMAQUANT_VALIDATION_WIKITEXT_STRIDE` | unset | `validation_harness.py:392` | WikiText stride for the harness PPL screen. |
| `PRISMAQUANT_VALIDATION_FAKE_METRICS` | unset | `validation_harness.py:166` | Test-only metric injection. Never set in a run whose numbers will be cited. |
| `PRISMAQUANT_SMOKE_MODEL` / `_SAMPLES` / `_SEQLEN` / `_SEED` / `_DETERMINISM` | unset / `2` / `32` / `12345` / `0` | `tools/smoke_graph_memory.py:97` / `:138` / `:136` / `:211` (defaulted at `:75`) / `:25` | Graph-memory smoke harness knobs. `_MODEL` names a **local** path or cached HF repo id (the smoke never downloads; unset probes three small Qwen ids with `local_files_only=True`). Caveat: this harness also `setdefault`s the retired L2/L3 graph selectors at `:66-74`, so those lines are not evidence that any of them is still read — see §9. |

## 5. NVFP4-CB / FP8-CB render flags — the LANE IS RETIRED (2026-09-02)

> **The build lane these flags served no longer exists.** Robert, 2026-09-02:
> *"put Tessera in PrismaQuant and remove Gridbook."* `EXPORT_CONTAINER=nvfp4_cb`
> now `exit 2`s, the exporter and the pin are archived at
> `archive/gridbook_lane_2026-09-02/`, and **no shell knob below has a
> `run-pipeline.sh` default any more** — every `CB_*` shell variable was read
> only inside an `EXPORT_CONTAINER=nvfp4_cb` block. What survives is the
> *render and cost plumbing* (`nvfp4_cb_formats.py`, `nvfp4_cb_footprint.py`,
> `cb_ldlq*.py`, `cb_minchain.py`, `cb_warm_state.py`, …), recorded as debt
> **D34** in `docs/ARCHITECTURE.md` §12. So the `PRISMAQUANT_CB_*` rows below
> are still **accurate about the code** and are kept for that reason — but
> nothing sets them, nothing reaches them from the pipeline, and a rung they
> price cannot be exported or served. Read this section as documentation of
> live-but-orphaned machinery, never as a lane you can run.

Historically: the lane was enabled by `EXPORT_CONTAINER=nvfp4_cb`, which the
pipeline gated to `TARGET_PROFILE=nvfp4_cb`, `PRODUCTION_CACHE=0` and
`PRODUCTION_RECACHE=0` (the CB exporter requantized the bf16 skeleton and never
read the export cache). The `COST_MODE=local` requirement was dropped
2026-07-30 (re-vet R3) in favour of render faithfulness via `--col-weights`
(CB Milestone C). Lane record:
`archive/gridbook_lane_2026-09-02/docs/lanes/nvfp4-cb/PLAN.md`. One class of `PRISMAQUANT_CB_LDLQ_*` env names is
deliberately absent from the table below: the canonical packed LDLQ route is
**ABI-fixed** — batched experts required, `expert_batch = LDLQ_PACKED_EXPERT_BATCH
= 16`, `feeder_threads = 0` (`nvfp4_cb_formats.py:1624`, refusals `:1693-1701`,
`:1707-1711`, `:1712-1718`) — so those names exist to be validated and stamped, not
tuned, and §10 classifies refusal/compat checks as non-knobs.

| env var | default | read at | what it does |
|---|---|---|---|
| `PRISMAQUANT_CB_ENCODE_TIER` | `balanced` | `nvfp4_cb_formats.py:156` (`_resolve_encode_tier` at `:154`; env const `:139`, default const `:141`); provenance stamp `expert_empirical_cost.py:1251` | Encoder speed-accuracy tier: `fast` / `balanced` / `max`. `max` is the original exhaustive scale sweep, bit-identical (regression-pinned); `balanced`/`fast` use the analytic s0 init + moment-scored micro-sweep + hill climb (measured ×3.9/×5.9 mean, `docs/lanes/nvfp4-cb/encode_tiers.md`). **`max` is NOT a quality upper bound** — it is the exhaustive *sweep*, not an exhaustive *search*. Measured 2026-08-29 over 24 corpus tensors: on FP8-CB at K48 `balanced` **beats** `max` on **24/24** (min +0.517, median +0.998, max +1.686 dB), because `max`'s `_candidate_scales` sweeps only `[1.0, 1.5]×amax/448` — a window anchored on the fixed lattice's grid-max — and that window is binding, while the s0-centred moment grid reaches past it (`encode_tiers.md` §Tier semantics span-curve note). At K32/K40 the two tiers are a wash in both directions, and NVFP4-CB is untouched (0/24 wins, deltas ±0.0000 dB). Consequence for study design: an FP8-CB number rendered at `max` **understates** what the shipped `balanced` default produces, so a comparison that pins one arm to `max` is arm-asymmetric, not conservative. |
| `PRISMAQUANT_CB_ENCODE_COMPILE` | **on** (`"1"`) | `nvfp4_cb_formats.py:624` (env const `:151`) | `torch.compile` the CB moment-scoring inner kernels (fast/balanced tiers only; max never compiles). Set `0` to pin eager — the compiled-vs-eager tie-flip caveat applies within a tier. |
| `PRISMAQUANT_CB_COMPILE_FAIL_CLOSED` | `0` (generic compatibility) | `cb_compile_contract.py`; strict RTX4090 producer requires exact `1` | When set, CB compile helpers use Inductor with `fullgraph=True`, `dynamic=True`, and `torch._dynamo.config.suppress_errors=False`; compile creation/runtime failure, a CPU eager route for an index-producing helper, or a return without exactly one backend dispatch refuses. The strict burn records live CUDA dispatches separately from AURA units restored under the same identity-bound checkpoint manifest. This does not change generic fallback behavior unless explicitly enabled. |
| `PRISMAQUANT_CB_LDLQ` | `0` (off) | serialization context `nvfp4_cb_footprint.py`; assignment `nvfp4_cb_formats.py`; pipeline/export/cost paths | `1` runs deterministic 64-column Hessian-feedback reassignment after the ordinary codebook and scale fit, using the same activation rows and activation-weighted metric as the cost render. It is a required CB serialization-context and render-identity dimension: cost/export stamp drift is refused and opposite-mode warm records cold-fallback. Assignment-time only; emitted fields remain grid-native and serving has no added runtime cost. Validation and timing history: `docs/lanes/nvfp4-cb/encode_tiers.md` §LDLQ feedback assignment. |
| `PRISMAQUANT_CB_LDLQ_SCOPE` | unset (→ derived from `PRISMAQUANT_CB_LDLQ`: `all` when it is true, `none` when it is false or absent) | `nvfp4_cb_footprint.py:683`, resolved in `cb_serialization_context_from_env` (`:660`; validation `:717-720`, consistency `:726-737`, derivation `:739-745`) | Per-family LDLQ scope, legal `none` / `nvfp4` / `all`; anything else raises. **Authoritative over the legacy bool `PRISMAQUANT_CB_LDLQ`** — when the scope is set, a legacy bool that disagrees with `scope != "none"` is refused, with exactly one back-compat exemption (legacy `true` + `scope=nvfp4`), because the bool cannot express a mixed scope. `nvfp4` is the **dual-basis production recipe**: the LDLQ NVFP4_CB plane feeds cost/allocator/export while the raw NVFP4 bank stays the immutable FP8_CB interpolation basis (`docs/ARCHITECTURE.md` §6.5.1). Under `require_explicit` at least one of the pair must be present (`:691-701`). |
| `PRISMAQUANT_CB_LDLQ_GATE` | `1` → `holdout` | `nvfp4_cb_formats.py:2254` (env const), resolved by `_ldlq_gate_mode` `:2268` | `holdout` (default since 2026-08-08) certifies the LDLQ reassignment on rows the Hessian fit never saw, keeping raw unless LDLQ strictly wins; `in_sample` is the legacy scoring that fits and scores on the same rows and therefore cannot fail — reproduction of pre-2026-08-08 artifacts only, never for a new one; `0`/`false`/`off` disables the gate; anything else raises. Byte-neutral either way: only the `k`-bit indices move, the codebook and scales are fixed. Tensors under `LDLQ_GATE_MIN_ROWS = 16` (`:2261`) are uncertifiable and keep raw. |
| `PRISMAQUANT_CB_MINCHAIN` | `0` (off) | `cb_minchain.py`; packed-expert cost path; CB serialization/export gates | `1` selects the minimum of the unchanged free fit and predecessor embed independently per expert slice, in ascending rung order. Selection is weight MSE with `a <= b + 1e-12*max(abs(a),abs(b))`; ties choose free. The mode/version and per-cell arm/solution/predecessor digests are render identity, so export refuses mixed chain/non-chain artifacts. Serving bytes remain ordinary flat CB payloads. |
| `PRISMAQUANT_CB_MINCHAIN_ANCHORS` / `_HOLDBACKS` | automatic five anchors / anchors 2 and 4 | `cb_minchain.MinChainInterpolationConfig` | Optional comma-separated rung numbers or format names for amendment-v2 monotone PCHIP. Exactly five ascending anchors and two holdbacks drawn from them are required. FP8 K28–K48 defaults resolve to anchors K28/K33/K38/K43/K48 and holdbacks K33/K43. |
| `PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED` | `42` | `cb_minchain.MinChainInterpolationConfig.audit_rung` | Base for the deterministic per-layer non-anchor audit draw: `random.Random(seed + layer)`. |
| `PRISMAQUANT_CB_MINCHAIN_BACKSTOP` | `0.25` | `cb_minchain.interpolation_acceptance_v2` | Gross-outlier backstop. A slice exceeding this relative error on either independent four-anchor holdback fit measures its remaining missing rungs; all other slices are accepted. |
| `PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN` / `_AUDIT_P95` | `0.05` / `0.15` | `cb_minchain.interpolation_acceptance_v2` | Per-projection audit thresholds. If either gate fails, amendment v2 requires full measurement of every projection in that layer. |
| `PRISMAQUANT_CB_WARM_STATE_DIR` | unset (off) | cost writer `measure_quant_cost.py:75`; streaming-export CLI default `export_nvfp4_cb_streaming.py:3286` | Opt-in directory for content-keyed CB encoder warm records. Cost measurement atomically stores each measured unit/rung's selected scale state plus source/imatrix identities, encoder-initializer identity, format, and complete serialization context. The streaming exporter accepts only exact matches; absent, corrupt, or mismatched records take the ordinary full-search path. See `encode_tiers.md` §Encoder warm start. |
| `PRISMAQUANT_CB_LADDER_INTERP` | `0` | `measure_quant_cost.py` (`_cb_ladder_plan`, dense and packed-stack paths); `expert_empirical_cost.py` (empirical expert path) | `1` enables per-`(family,mode)` RD-law ladder interpolation: anchors + holdout are measured normally and predicted rungs use holdout-gated fits with measured fallback (`encode_tiers.md` §B/§C). Dense Linears fit per tensor. Physically packed expert stacks fit and gate each expert slice independently; only rejected slices are encoded for missing rungs, and mixed rows carry `cost_source=mixed` plus the slice vector `cost_source_per_expert`. The empirical expert path retains its existing unit granularity. All paths run the shared law (`expert_empirical_cost._cb_ladder_law`: floored-linear in the exact ceil-first rate factor `R(k)` → smooth floor law → log-linear) and log their holdout accept/reject rates. One shell knob drives the pipeline wiring: `CB_LADDER_INTERP` (default `0`), which exports `PRISMAQUANT_CB_LADDER_INTERP=1` for the cost stage and gates the empirical expert stage's `--cb-ladder-interp`. |
| `PRISMAQUANT_CB_LADDER_ANCHORS` | unset | `expert_empirical_cost.py` (`_cb_ladder_split`); provenance `measure_quant_cost.py` (`cost_payload_provenance`) | Optional comma-separated explicit CB rungs to measure as interpolation anchors. It must be paired with `PRISMAQUANT_CB_LADDER_HOLDOUT`, stay within one CB family, contain at least two distinct menu rungs, and exclude the holdout. The exact plan is stamped into monolithic and incremental cost provenance, so changing a campaign plan invalidates shard identity. |
| `PRISMAQUANT_CB_LADDER_HOLDOUT` | unset | `expert_empirical_cost.py` (`_cb_ladder_split`); provenance `measure_quant_cost.py` (`cost_payload_provenance`) | Optional explicit held-out rung for the plan above. It must be present in the measured menu and separate from every anchor. A failed dense-tensor gate measures that tensor's predicted rungs; a failed packed-expert slice gate measures only that slice's missing rungs. |
| `PRISMAQUANT_CB_LADDER_TOL` | `0.10` | `measure_quant_cost.py` (module-level constant `_CB_LADDER_TOL`, read at import) | Holdout-gate relative-error tolerance for the dense and physical packed-stack cost ladders; matches the expert stage's `--ladder-holdout-tol` default. **This is the FALLBACK value, not the rule** (R20): per `encode_tiers.md` §B the gate must trust a fit only where the holdout error clears the *between-seed cost noise*, so `_cb_ladder_holdout_tol` (`expert_empirical_cost.py`) derives the tolerance from the paired per-calibration-window spread of the measured rungs — free on the expert path, which already measures every unit KL window by window. The constant stands only where that datum is absent or degenerate: local cost measurement takes one draw per `(tensor or packed slice, format)`, so it has no between-draw spread and uses this unchanged value. |
| `PRISMAQUANT_CB_COL_WEIGHTS` | unset | `measure_quant_cost.py:1718` | Path to the shared CB col-weights (imatrix) pickle; exported by the pipeline at `run-pipeline.sh:1077` from the `CB_COL_WEIGHTS` shell knob (default `run-pipeline.sh:542`). This is the lockstep contract: measured CB cost and the exporter's weighted-VQ render must use the same weights, including the synthesized per-expert down_proj replay entries the inline module-input pool cannot provide. |
| `PRISMAQUANT_EXPERT_CALIB_BATCH` | `1` | `expert_empirical_cost.py:92` (`_CALIB_BATCH_ENV`), used in `_calib_batch()` | Calibration sequences per forward in the empirical expert unit-KL stage — the dominant-wall knob of `[2d]` / `[2d-CB]`. `1` preserves the historical per-sequence numerics exactly; `>1` batches independent windows (both arms always use the same batching, so the KL comparison stays internally consistent). |
| `PRISMAQUANT_EXPORT_REUSE_PRIOR` | **quarantined** | `export_nvfp4_cb_streaming.py` | Reserved legacy alias. Current HEAD fails closed: prior artifacts are not bound to an exact source/imatrix/codebook/exporter ABI, so no tensor may be reused. `EXPORT_REUSE_PRIOR` is rejected by `run-pipeline.sh`. |
| `PRISMAQUANT_EXPORT_REUSE_VERIFY` | inactive | `export_nvfp4_cb_streaming.py` | Reserved with the quarantined reuse interface; sampling is not proof of whole-artifact identity. |
| `PRISMAQUANT_EXPORT_PIPELINE` | `0` (off) | `export_nvfp4_cb_streaming._StreamWriter.write` | `1` overlaps dense-source prefetch, the unchanged ordered encode stream, and bounded ordered writing. This is execution strategy only: it is deliberately absent from the CB serialization context and every render-identity stamp, and on/off artifacts must be byte-identical. Default stays off pending the skip-marked real-GPU identity gate. |
| `PRISMAQUANT_EXPORT_PREFETCH_DEPTH` | `1` | `export_nvfp4_cb_streaming._StreamWriter._write_pipeline` | Maximum prefetched source tensors ahead of encode; output suffix entries do not consume the depth. Dense encode sources are pinned when CUDA is available; 10GB-class expert stacks keep the existing one-device-buffer producer rather than duplicating a whole stack in pinned host memory on the unified pool. |
| `PRISMAQUANT_EXPORT_WRITE_QUEUE_BYTES` | `2147483648` (2 GiB) | `export_nvfp4_cb_streaming._StreamWriter._write_pipeline` | Hard reservation budget for encoded outputs in flight or waiting for the canonical writer. Reservation occurs before encode. A single larger tensor runs exclusively; later outputs backpressure until it is committed. |

**Shell-side CB knobs — REMOVED from `run-pipeline.sh` on 2026-09-02** with the
lane. The list below is what they were; none of them has a default now, and
`CB_LADDER_INTERP`, `CB_EXPERT_NSAMPLES` / `_SEQLEN` / `_SAMPLE` and
`CB_COL_WEIGHTS` are the only survivors, kept solely because the GGUF lane and
the generic imatrix harvest read them. Historically: `CB_LADDER_INTERP` (`0`),
**`CB_EXPERT_EMPIRICAL` (`0` since 2026-07-30 — D15: the default is now the
value every shipped MoE CB driver sets; `1` re-enables the `[2d-CB]` empirical
unit-KL replacement)**, `CB_EXPERT_NSAMPLES` / `CB_EXPERT_SEQLEN` (`16` /
`512`), `CB_EXPERT_SAMPLE` (`0`), `CB_COL_WEIGHTS`
(`$WORK_DIR/artifacts/cb_col_weights.pkl`; harvested by the shared
`harvest_cb_col_weights` helper, one definition and four call sites), `CB_OUT`
(`$WORK_DIR/exported_nvfp4_cb`), `CB_CODEBOOK_SOURCE` (`lattice`),
`CB_CODEBOOK_ITERS` (`4`), `CB_CODEBOOK_SEED` (`0`), `CB_SCALE_SWEEP` (`1`;
`0` is the one-shot amax/grid-max ablation only), **`CB_SCALE_CODING`
(`two_tier` since 2026-07-30 — D15: layout-v2 shipped in the Hy3 295B and
Laguna-S-2.1 artifacts and `STANDARDS.md` calls it the production fp4 scale
coding, with v1 legacy read-compat only; the flag is inert on fp8-CB-only
menus, which have no group-16 scale plane)**,
`PRISMAQUANT_CB_LDLQ` (`0`; opt-in post-fit feedback assignment),
`PRISMAQUANT_CB_MINCHAIN` (`0`; opt-in monotone min-chain),
`EXPORT_REUSE_PRIOR` / `EXPORT_REUSE_VERIFY` are reserved and rejected as of
2026-07-31; every CB export uses a fresh output.

The streaming exporter also exposes `--warm-state-dir` (defaulting to
`PRISMAQUANT_CB_WARM_STATE_DIR`) and `--warm-verify-sample N` (default `32`).
The latter re-runs full scale search for a deterministic random sample of
matched records and aborts before publication unless both selected scales and
rendered bytes are exactly identical.

## 6. Non-`PRISMAQUANT_` shell vars read directly by Python

A coupling worth knowing: several `run-pipeline.sh` variables are read straight
out of `os.environ` by library code, so they act on library defaults even when
the corresponding CLI flag is absent.

| env var | default | read at |
|---|---|---|
| `NSAMPLES` | `32` | `autoscale.py:403`, `streaming_model.py:881` |
| `SEQLEN` | `1024` | `autoscale.py:404`, `streaming_model.py:882` |
| `LAYERS_PER_SHARD` | `1` | `autoscale.py:406`, `streaming_model.py:879-880` |
| `CACHE_HEADROOM_GB` | — | `autoscale.py:407`, `streaming_model.py:871` |
| `PREFETCH_WORKERS` | `auto` | `incremental_probe.py:2959`; `streaming_model.py:284` |
| `PREFETCH_MIN_AVAILABLE_GB` | `auto` | `incremental_probe.py:2963`, `streaming_model.py:301` |
| `PREFETCH_LOOKAHEAD` | `auto` | `incremental_probe.py:2954` |
| `ACTIVATION_ROWS_LIMIT` | `256` | `incremental_probe.py:2979` |
| `TMPDIR` | — | `sensitivity_probe.py:89`, `validate_assignments_kl.py:963` |
| `VLLM_URL` | `http://localhost:8000` | `validate_quantized_model.py:507` |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | set by the allocator | `allocator.py:1215-1216` |
| `CUBLAS_WORKSPACE_CONFIG` | set under `PRISMAQUANT_DETERMINISTIC` | `build_production_cache.py:519` |
| `PQ_SERVE_IMAGE` | none — **fails closed** | `tools/measure_vllm_full_kl.py`, `tools/measure_vllm_wikitext_ppl.py` |

`PQ_SERVE_IMAGE` (equivalently `--serve-image`) is new on 2026-09-02 and is the
one operator knob the Gridbook retirement *added* rather than removed. Both
measurement tools used to take the serving container image from the Gridbook
pin; with the pin archived (`archive/gridbook_lane_2026-09-02/`) the image has
no attested source, so the tools refuse immediately after `parse_args` — before
any model load — when neither the flag nor the env var is set. Defaulting to a
tag would be exactly the unattested runtime claim principle 14 forbids: the
image is what the measurement is *of*.

## 7. Plugin serving flags — RETIRED for Gridbook (2026-09-02), and the rule that outlives it

The Gridbook serving plugin is no longer a PrismaQuant lane
(`archive/gridbook_lane_2026-09-02/`), so its flags are not documented anywhere
in this repository and its pin is gone.

**The rule this section existed to state still holds, and now applies to
Tessera.** A serving runtime owns its flags, their defaults, their validation
and their dispatch semantics. Keeping a second table here already caused
factual drift once (the dense CUDA GEMV crossover and fused-FP4 promotion
status), which is why there is no table. Consult the documentation shipped by
the exact pinned commit — for Tessera,
`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`, whose only
operator knob is `TESSERA_SERVE_MODE=resident|streamed`. A serve record must
fingerprint that commit and its actual environment; a PrismaQuant document is
never authority for a runtime default (principle 14).

## 7a. Tessera menu, pin and ablation flags

Producer-side only; none of these reach a serving runtime. See
`docs/ARCHITECTURE.md` §4.10 and
`docs/measurements/tessera-continuous-menu-2026-09-02.md` §10.

| flag | default | what it does |
|---|---|---|
| `PRISMAQUANT_TESSERA_MENU` | `attested` | `attested` offers only rungs a pinned runtime contract attests; `research` offers the whole realisable rate set (~3000 rungs per unit) and is what every measurement on this branch used. `research` is **not** shippable: nothing attests those routes. The mode also decides what the `FORMATS=TESSERA` **token** expands to: a cost table priced under `research` and read back under `attested` expands to the priced-and-attested intersection, and the allocator prints how much it narrowed (`[alloc] Tessera menu: N of M priced rungs are attested`). An empty intersection refuses; an explicitly named unattested rung still refuses regardless of mode. |
| `PRISMAQUANT_TESSERA_DEV_PIN` | unset | The development override for Tessera admission, which is otherwise fail-closed until a Tessera RELEASE tag exists. Any non-empty value opts in and is recorded verbatim; the gate is the installed `tessera/serving/runtime_contract.json`'s **answer** — every value a gate reads, canonicalised by `tessera_runtime_contract.contract_answer()` — which must equal `TESSERA_DEV_PIN_ANSWER` or the read **raises** with a field-level diff. Prose, `contract_version` and `plugin_version` are not in the answer, so a Tessera commit that moves none of the values a gate reads does not re-stale the pin (issue #38); adding a family or a cell does. Unset means no Tessera contract is read at all and every rung stays `unattested`; there is no third state, so a stale pin can never degrade into a silently empty menu. Stamped into provenance as `tessera_dev_pin`, with the reviewed commit and sha alongside the bytes actually read. |
| `SELECTION_MODE` (pipeline, not a `PRISMAQUANT_*` flag) | `surrogate` | **Required to be `validated-surrogate` for any Tessera menu run that is promoted.** Measured 2026-09-02: this menu's own allocations, served against a byte-matched uniform arm, lose 2.00x in KL at 4.0 bpp (2.33x / 2.88x at 3.0 / 5.0) with bytes exact to the bit (`docs/measurements/tessera-allocated-served-2026-09-02.md`). The allocator prints a warning and stamps `tessera_menu.selection_caveat` on every surrogate-selected Tessera run. Validated selection is necessary and not sufficient -- it re-scores only the allocator's own Pareto points -- so the recipe also requires a byte-matched uniform arm served beside the candidate. |
| `PRISMAQUANT_TESSERA_GROUP_KNAPSACK` | `1` | Debug/ablation only. `0` disables the fused-group Minkowski fold, so a fused group carries one candidate per format NAME and every member of the group must take the **same rung**. That is the constraint the allocator used to impose by accident; the lever exists so its cost can be measured on the same cost table rather than across two campaigns. A run with it off logs no `tessera group knapsack` line and stamps `__ablation__` into the group report. Never set it for a shipping build. |

## 8a. Tessera lane (`EXPORT_CONTAINER=tessera`, ARCHITECTURE.md §9.4)

The arm NAMES Tessera's own plan translator and exporter rather than vendoring
either, so the one path knob is where that checkout lives. It refuses on the
PENDING release pin today (RobTand/tessera#17) before it reads any of the rest.

| env var | default | what it does |
|---|---|---|
| `EXPORT_CONTAINER` | `compressed-tensors` | `tessera` switches stage 4 to `plan_from_layer_config.py` + `export_tessera_serving.py` under `TESSERA_REPO`, behind `prismaquant.tessera_export_lane`'s three gates: the checkpoint's structure against the packaged contract's declared `structures`, the lane spec's `executes` against the contract's `formats[]` rows (principle 14), and the release pin. Only `qwen3` declares the lane, so `require_lane_supported` refuses every other architecture first. |
| `TESSERA_REPO` | `/home/rob/tessera` | Checkout of the pinned Tessera release. The arm refuses up front if it does not hold both named tools. Never a place to point at a working tree with local edits during a build you intend to ship. |
| `TESSERA_SERVE_MODE` | `resident` | The plugin's single operator knob, `resident` or `streamed`. Declared rather than defaulted silently because it changes the artifact's footprint and is folded into vLLM's compile-cache key; both residencies are receipted by the contract and both must be exercised, since they decode the same bytes by different paths. Any third value refuses. |
| `TESSERA_PLAN_COVER` | `as-allocated` | How a partial allocation becomes a whole-model plan. `as-allocated` plans exactly the units the allocation names and spells every other body Linear BF16 **explicitly** — silence would otherwise become the exporter's 4-bit default rung. `broadcast-by-role` applies the per-role assignment at every depth; it is an EXTRAPOLATION, stamps itself as one in the plan's sidecar, and is refused unless the allocation is single-layer with matching shapes. Part of the `tessera-plan` stage's settings hash, so a plan built under the other mode is not silently reused. Any third value refuses in the same up-front gate block as `TESSERA_SERVE_MODE`, not at the translator's `argparse` `choices` — the translator does not run until stage 4, and a typo must not cost a whole probe/cost/render run first. |

## 8. GGUF lane (`docs/lanes/gguf.md`)

| env var | default | what it does |
|---|---|---|
| `EXPORT_CONTAINER` | `compressed-tensors` | `gguf` switches stage 4 to skeleton-build + `export_gguf`; the pipeline requires `TARGET_PROFILE=gguf` and `PRODUCTION_CACHE=0`. The `COST_MODE=local` requirement was replaced by the render-faithfulness assertion (re-vet R3): what matters is that the cost render applies the imatrix exactly when the export does, which `PRISMAQUANT_GGUF_IMATRIX` already keys. |
| `PRISMAQUANT_GGUF_IMATRIX` | `1` (`measure_quant_cost.py:1211`) | Activation-weighted (imatrix) k-quant scale selection in BOTH the batched cost path and the pipeline's export call — keep the two in lockstep or the A/B has a rendering confound |
| `LLAMA_CPP_DIR` | `/home/rob/dq-runs/llama.cpp` | Source of `convert_hf_to_gguf.py` for the skeleton |
| `GGUF_SKELETON` | `WORK_DIR/artifacts/skeleton.gguf` | bf16 skeleton path (built if missing) |
| `GGUF_TOKEN_EMBEDDING_FORMAT` / `GGUF_OUTPUT_FORMAT` | keep skeleton precision | Quantize `token_embd` / `output` (llama.cpp presets use Q2_K / Q6_K) |

## 9. Retired entries, dead tokens, and non-flags

Previously listed in the live tables or here, now removed from the index. Kept
as a record so they are not re-added by a future scrape — the dated-wall
convention of `docs/ARCHITECTURE.md` §11 applied to a flag catalogue.

### 9.1 Retired 2026-08-02 — the L2/L3 graph and cache knobs

`docs/ARCHITECTURE.md` §4.4 walled both levels of the old cascade on
2026-07-30 (`archive/l3_propagated_2026-07-30/`, re-vet **R4**), moving the L3
half of `kl_measurement.py` to
`archive/l3_propagated_2026-07-30/prismaquant/kl_measurement_l3.py`. §4 kept
documenting that walled subsystem's knobs as if they were live, with
pre-split line numbers: `kl_measurement.py` is **1,246 lines** today and the
rows cited lines up to `:5516`. Every token below was re-grepped over
`prismaquant/`, `tools/`, `scripts/` and `pipeline.py` on 2026-08-02 and has
**zero readers outside `archive/`**.

Note the trap that kept these alive: `tools/smoke_graph_memory.py:63-75`
`setdefault`s most of them. That is a **writer**, not a reader — the smoke
drives the walled L2/L3 stack, so nothing consumes the values.

| token | last reader | verdict |
|---|---|---|
| `PRISMAQUANT_L3_CUDA_GRAPHS` | `kl_measurement_l3.py:3969`, `:4403` | RETIRED — walled. Still written by `tools/smoke_graph_memory.py:67`. |
| `PRISMAQUANT_COORD_LANE_CUDA_GRAPHS` | `kl_measurement_l3.py:2872`, `:3051`, `:3453`, `:3617` | RETIRED — walled. Still written by `tools/smoke_graph_memory.py:69`. |
| `PRISMAQUANT_COORD_LANE_CUDA_GRAPH_CACHE_SIZE` | `kl_measurement_l3.py:1361` | RETIRED — walled. Still written by `tools/smoke_graph_memory.py:72`. |
| `PRISMAQUANT_COORD_LANE_BATCH` | none, ever | **Set but never read** — written by `tools/smoke_graph_memory.py:68` (not `:72`, as this table previously said), consumed by nothing. |
| `PRISMAQUANT_COORD_REPLAY_CACHE` | `kl_measurement_l3.py:2857`, `:3437` | RETIRED — walled. Still written by `tools/smoke_graph_memory.py:70`. |
| `PRISMAQUANT_L3_PREQUANT_CACHE` / `_RESERVE_GB` / `_RESERVE_FRACTION` / `_PEAK_MULTIPLIER` | `kl_measurement_l3.py:2766`, `:3336`, `:3823`, `:4274`; reserves `:312`, `:316`, `:322` | RETIRED — walled. |
| `PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE` | `kl_measurement_l3.py:2772`, `:3342`, `:3830`, `:4280` | RETIRED — walled. |
| `PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB` / `_FRACTION` | `kl_measurement_l3.py:2614`, `:2658` | RETIRED — walled. |
| `PRISMAQUANT_L3_MIN_HOST_MEM_GB` | `kl_measurement_l3.py:355`, `:364`, `:377` | RETIRED — walled. The `GPUMemoryBudgetExceeded` host-memory floor it named is gone with the L3 pair/scout path; `PRISMAQUANT_HOST_MEM_RESERVE_GB` / `_FRACTION` (§4) are the live analogues. |
| `PRISMAQUANT_EMPTY_CACHE_EACH_REPLAY_BATCH` | `kl_measurement_l3.py:152` | RETIRED — walled. |
| `PRISMAQUANT_ALLOW_PYTORCH_FALLBACK` | `archive/orphans_2026-07-30/prismaquant/_fast_kernel_guard.py:42` | RETIRED — walled 2026-07-30 (re-vet **R19**). The guard that read it (`require_fast_kernels`) lost its only caller when `polish_from_assignment` was archived 2026-05-15. **Principle 9's kernel-performance gate is manual today** — `docs/ARCHITECTURE.md` §12 (LOW). |
| `PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_GB` | none, ever | **DEAD** — no occurrence anywhere in the tree; it was only ever a slash-suffix in the §4 row beside two real flags. The live free-memory floor is `PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_FRACTION`. |
| `PRISMAQUANT_DO_NO_HARM_MIN_GAIN` | none, ever | **DEAD** — no occurrence anywhere in the tree; the documented default `0.0` was fiction. The live analogue is `PRISMAQUANT_RENDER_GATE_MIN_GAIN` (a different gate). This is the token `docs/ARCHITECTURE.md` **D18** asks to delete outright; it is kept here as a ledger row instead, because a scrape that re-adds a fictional flag is the failure the row exists to prevent. |

### 9.2 Dead tokens and non-flags

| token | verdict |
|---|---|
| `PRISMAQUANT_L2_CUDA_GRAPHS` | **DEAD** — never read. Sole occurrence is the comment at `perturbed_x_cache.py:1243` ("intentionally not applied here"). |
| `PRISMAQUANT_PATCH_SENTINEL` / `PRISMAQUANT_CHANNEL_SENTINEL` / `PRISMAQUANT_FULL_SENTINEL` | **Not env vars.** Python module-attribute name constants `_PRISMAQUANT_*_SENTINEL` (`sensitivity_probe.py:853-855`), used with `setattr`/`getattr`. |
| `PRISMAQUANT_GRAPH_POOL` | **Not an env var.** Module global `_PRISMAQUANT_GRAPH_POOL` (`kl_measurement.py:111`). |

## 10. Discovering live flags

There is deliberately no hand-maintained exhaustive flag list. The previous
snapshot mixed producer, archived, and externally owned runtime selectors and
drifted as soon as the runtime moved repositories. Sections 1–9 document the
supported PrismaQuant policy knobs by subsystem. For a mechanical source audit,
search live producer code directly, for example:

```bash
rg -o 'PRISMAQUANT_[A-Z0-9_]+' prismaquant scripts tools \
  | cut -d: -f2 | sort -u
```

That inventory is diagnostic rather than a stability promise: a token may be a
compatibility check, refusal, or research switch rather than a supported knob.
Serving-runtime flags are defined and documented only by the exact pinned
commit (§7). `PQ_EXPORT_VECTOR_CHUNK` remains the
one live producer flag outside the `PRISMAQUANT_` namespace.

## 11. Disabling for debugging

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

Two traps when doing this:

- `PRISMAQUANT_GPTQ_DAMP_SWEEP` used to have **different defaults in the
  exporter and in `kl_sensitivity_probe`**; D5 closed 2026-07-30 and the probe
  is walled (§2). There is now one reader and one default (`0`), but pinning it
  explicitly in an A/B is still the cheap habit.
- Extension residency shifts allocator addresses and moves confident-KL by up
  to ±17% between arms (measured on the retired Gridbook lane, but the effect
  is a property of loading a plugin extension, not of that plugin). For a served
  plugin-lane comparison, use the same-process A/B harness and residency
  controls documented by the exact pinned runtime commit; runtime selector names and defaults are intentionally not repeated
  here.
