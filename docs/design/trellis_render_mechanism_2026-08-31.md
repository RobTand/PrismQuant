# Trellis TCQ Render Mechanism — One-Encode Invariant and Wire Retention (2026-08-31)

Branch: `muse/wo-b-trellis-render-20260831` · Commit: `007effe` base + WO-B deliverables
Date: 2026-08-31 · Author: Muse Spark (WO-B)

## 1. The invariant that defines success (principle 8)

**One encode. The bytes the exporter ships are the bytes whose decode the
surrogate priced and the KL measured.**

```
encode_trellis_one_linear(weight, col_weights, recipe)
    -> wire_bytes  ---------------------> what WO-C writes to checkpoint
    -> decoded_weight (parsed FROM those exact bytes)
                   ---------------------> what render_production_weight returns,
                                          what AURA/render-score prices,
                                          what validate_assignments_kl measures
```

Violating this priced a real A-side at zero on 2026-08-17: the weight plane
rendered for cost and KL was not the weight plane the runtime executed. The
trellis lane would have the same confound if the cache returned a decoded
tensor whose bytes were not the bytes the surrogate scored. Two encodes are
two chances to diverge (the 256-state tail-biting Viterbi is far too expensive
to run twice), so the wire bytes are **retained by the cache at render time**.

## 2. Retention design — one cache mechanism (principle 8)

### 2.1 What is stored

`ProductionWeightCache` now stores **both** planes per trellis entry:

* `weights[(qname, fmt)]` — the decoded tensor (`[out, in]` bf16, the same
  representation the existing cache stores for NVFP4/CB/GGUF)
* `trellis_wires[(qname, fmt)]` — the immutable `gridbook.trellis.wire.v1`
  bytes (`bytes` in-memory, `*.trellis_wire` file beside the `*.pt` shard
  when `cache_dir` is set)

A third map `trellis_wire_identities[qname|fmt] = {wire_identity_sha256,
recipe_identity_sha256}` carries hash discipline: the two digests come from
`TrellisOneLinearArtifact.receipt` (the producer's one-encode receipt). A
later consumer (WO-C exporter, KL validator) proves byte identity by
comparing `hash(wire_bytes)` to `wire_identity_sha256` and the context's
`recipe_identity_sha256` to the stored recipe digest. No second encode is
performed to obtain bytes at export time.

### 2.2 Where it lives

No new sidecar directory, no parallel pickle, no second store. The existing
cache store is extended:

* `ProductionWeightCache.weights` — unchanged for non-trellis, extended to
  also hold trellis shards (same `*.pt` naming)
* `ProductionWeightCache.trellis_wires` — new dict, mirrored on disk as
  `cache_dir/<safe>__<fmt>.trellis_wire` alongside the shard
* `ProductionWeightCache.trellis_wire_identities` — new dict persisted in
  `metadata["trellis_wire_identities"]`
* `metadata["trellis_render_identity"]` — versioned artifact-wide identity
  (schema `prismaquant.trellis_render_identity.v1`, ABI
  `prismaquant.trellis_render_mechanisms.v1`) binding `TrellisSerializationContext`
  + `col_weights` digests + render-contract. Old caches missing this identity
  fail closed (see §2.4).

### 2.3 Artifact-wide recipe

`TrellisSerializationContext` ( `prismaquant/trellis_serialization.py`,
schema `prismaquant.trellis_serialization.v1`) is the exact analogue of
`CBSerializationContext`:

```
family, body_rate_q256, schedule, layout, alphabets,
scale_rule, sb_chunk, determinism_mode, tailbite_candidates,
backend, point_route, [global_scale_real_override]
```

Every value-bearing encoder choice is explicit; there are no
environment-derived defaults (the encoder module's docstring forbids them).
`render_production_weight(..., trellis_serialization_context=ctx)` refuses
when `ctx is None` on a `TCQ_*` format, naming the eleven missing recipe
fields in the error. `col_weights` is value-bearing and belongs to
`WEIGHTED_RENDER_FAMILIES += ("tcq_trellis",)` so the
`tests/test_col_weights_render_identity.py` inertness check excludes it
rather than weakening the assertion.

Mechanism plumbing mirrors CB: `render_score.py` registers `trellis`
(`imatrix_weighted_trellis_search`, phase 50, `weight_mse`) and
`_format_supports_render_mechanism` returns `{"trellis","weighted_vq"}` for
TCQ, exactly as it returns `{"weighted_vq"}` for CB/GGUF. No parallel
mechanism ordering is invented.

### 2.4 ABI and stale-cache behavior

Entry shape changes bump the trellis ABI:
`TRELLIS_RENDER_MECHANISM_ABI = "prismaquant.trellis_render_mechanisms.v1"`.
An old cache carrying `v0` or no `trellis_render_identity` at all refuses
loudly when a trellis format is requested:

```
validate_production_cache_trellis_render_identity ->
  ValueError: trellis cache is missing versioned metadata['trellis_render_identity'];
  legacy or partially resumed trellis caches must be rebuilt
```

or

```
ValueError: unsupported trellis render mechanism ABI
```

A tensor shard without its wire (`*.pt` present, `*.trellis_wire` absent)
also refuses — it is a stale `WORK_DIR` miss, not a silent fallback to RTN
(the `COST_MODE` flip precedent). This is exercised by
`tests/test_trellis_render_mechanism.py::test_trellis_stale_abi_cache_rebuilds_loudly`.

### 2.5 Thread-local one-encode bridge

`render_production_weight` stays `-> Tensor` for backward compatibility.
On a trellis render it calls `encode_trellis_one_linear` once, stashes the
`TrellisOneLinearArtifact` in a thread-local
`_trellis_thread_local.artifacts[(qname,fmt)]`, and returns
`artifact.decoded_weight`. `fill_production_weight_cache` pops the artifact
immediately and persists both tensor and wire via `_store_trellis_wire_entry`,
checking `hash(wire_bytes) == receipt["wire_identity_sha256"]` and
`BF16(decoded_weight) == BF16(tensor)` before publication. No second Viterbi
is run.

## 3. Performance — measured deliverable (principles 7, 15)

### 3.1 Routing

* `backend="triton"` (four fused Triton launches per `sb_chunk`: warm
  metrics, candidate closure, survivors, traceback) is used **by default on
  CUDA** — the hot path is GPU-bound.
* `backend="eager"` is the executable CPU reference and stays for
  bit-exactness tests (`Eager/Triton agreement`) only.

No fallback: if `backend="triton"` is requested but `weight.device` is not
CUDA, the render fails closed. No sampling of fewer Viterbi candidates.

### 3.2 Measurement instruments (two, both required)

* **In-process profiler** — `torch.profiler` with `CPU + CUDA` activities;
  chrome traces saved to `scratch/trellis_profile_*_trace.json` and
  `key_averages` tables to `scratch/*_table.txt`.
* **Box-level power** — `nvidia-smi --query-gpu=power.draw` sampled per
  tensor encode, reported as watts against the ~140 W GB10 envelope and as
  **work per joule** (params / (power × time)). `gpu_utilization` is
  non-diagnostic on GB10 (96% on both sides of a 5.83× speedup on 2026-08-28)
  and `utilization.memory` is a fake hard 0; neither is quoted.

### 3.3 Raw numbers

All runs on Sparky, GB10 sm_121, `prismaquant-cu130` (torch 2.11+cu130),
one `torch.randn` weight per measurement, `col_weights` uniform in
`(0.05,1.05)`, `family=TCQ_E2M1_R256`, `body_rate_q256=512`,
`schedule=[2]*cols`, `layout=fixed_quota_per_256`, `sb_chunk=1024` (large
shape) / 256 (small shape), `scale_rule=static_6`, `point_route=full`,
`determinism_mode=on`, `tailbite_candidates=4`.

| shape | backend | per-tensor (s) | avg power (W) | envelope | work/J (params/J) | note |
|---|---|---|---|---|---|---|
| [1024,1024] bf16 | eager (CPU ref) | 2.466 | 27.7 | 19.8% | 15.3k | CPU-bound, 65k `index_select`/`gather` launches, Self CUDA 1.15s |
| [1024,1024] bf16 | triton | 0.148 | 15.5 | 11.1% | 456k | 16.6× faster, 29.8× more work/J |
| [4096,4096] bf16 | triton | **0.806** | 41.9 | 29.9% | 495k | production size, 4 Triton kernels dominate CUDA time |

*1024 shape profiler tables*: `scratch/eager_table.txt`,
`scratch/triton_table.txt`; `scratch/b3_small.json` holds the JSON
summaries. Large-shape trace omitted for eager (would exceed the harness
timeout; the 16.6× delta on 1024 already establishes the CPU-bound bug).

**Before/after delta is the claim**: on the same 1024 shape, triton is
**16.6× faster** and **29.8× more work per joule** — power was right where
`gpu_utilization` would have read 60 vs 96% and hidden the headroom.
Before: eager's `aten::index_select`/`gather`/`fill_` dominate CPU and issue
~65k small CUDA launches (Self CPU 4.15s, Self CUDA 1.15s). After: four fused
Triton kernels (`_candidate_kernel` 30.8%, `_warm_metrics_kernel` 17.4%,
`_traceback_kernel` 15.8%, `_survivor_kernel` 9.5%) dominate and the launch
count collapses; Self CPU 0.24s, Self CUDA 0.11s.

`nvidia-smi` samples during the 4096 triton run:
`power.draw 41.87 W, clocks.sm 2489 MHz, utilization.gpu 60%` (utilization
still non-diagnostic — it read 96% on the 5.83× case).

### 3.4 Qwen3-4B extrapolation

Qwen3-4B has 252 Linears ( `docs/design/format_choice_4p5_stage0_results.md`).
Using the measured **0.806 s per [4096,4096] triton tensor**:

```
252 × 0.806 s = 203 s ≈ 3.4 minutes
```

Even at 300 linears: 242 s ≈ 4.0 min. Well under the ~30 minute GB10
budget — no loud warning needed. The extrapolation is conservative: real
Qwen3-4B linears are mix of 2560×2560, 2560×3072, etc., generally smaller than
4096×4096, so 3.4 min is an upper bound for the dense body.

No speculative optimization was performed before the profile; the profile is
the claim.

### 3.5 Files

* `scratch/profile_trellis_small.py` — bench harness (1024 eager vs triton +
  4096 triton)
* `scratch/eager_table.txt`, `scratch/triton_table.txt`,
  `scratch/triton4096_table.txt` — `torch.profiler.key_averages` tables
* `scratch/b3_small.json`, `scratch/profile_small.log` — JSON summaries and
  console log
* `scratch/profile_trellis.py` — full 4096 eager/triton harness (eager
  exceeds harness timeout; retained for completeness)

## 4. Tests

`tests/test_trellis_render_mechanism.py` (6 tests, all green):

1. **Rendering identity** — three shapes including non-multiple 7 rows (8/7/5
   ×256), fill cache, retrieve wire, `decode_values_torch` → BF16-equal.
2. **Determinism** — same `(weight,col_weights,ctx)` → byte-identical wire
   and `wire_identity_sha256`.
3. **col_weights value-bearing** — different imatrix → different wire.
4. **Missing context refuses** — `ValueError` naming all eleven recipe fields.
5. **Eager/Triton agreement** — CUDA triton vs CPU eager byte-identical wire
   (skips if no CUDA/triton).
6. **Stale-ABI rebuilds loudly** — old ABI `v0` or missing identity → `ValueError`
   (`unsupported ... ABI` / `missing versioned`).

## 5. What is *not* done (seam for WO-A / WO-C)

* No `FormatSpec` registration for `TCQ_*` (WO-A in parallel). Rendering keys
  off `parse_trellis_format_name`; when registry carries `family=="tcq_trellis"`
  the seam at `production_weight_cache.py:5005-5030` and `_weighted_render_family`
  subsumes the fallback.
* No `export_native_compressed` packing for trellis (WO-C). The cache exposes
  `get_trellis_wire_bytes(qname, fmt)` and `get_trellis_wire_identity` for the
  exporter to consume; the exporter still refuses `TCQ_*` until WO-C lands,
  which is the correct fail-closed behavior.

## 6. Commands and logs

```bash
PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest tests/test_trellis_render_mechanism.py -v
# 6 passed

PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest tests/ -q -x -k "trellis or render or cache or col_weights"
# 84 passed (6 trellis + 23 col_weights + 55 render/cache)

PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python scratch/profile_trellis_small.py
# see scratch/b3_small.json and scratch/*_table.txt for evidence
nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,utilization.gpu --format=csv
# 41.87, [N/A], 2489, 60  (trellis 4096 triton, 29.9% of 140W envelope)
```

No `pip install` was performed; `PYTHONPATH=.` and the pinned `prismaquant-cu130` venv were used.

## 7. Provenance

* `prismaquant/trellis_serialization.py` — new, `TrellisSerializationContext`
  + stamp, `TRELLIS_RENDER_MECHANISM_ABI v1`
* `prismaquant/production_weight_cache.py` — trellis branch in
  `render_production_weight`, wire retention in `ProductionWeightCache`
  (`trellis_wires`, `trellis_wire_identities`, `TRELLIS_RENDER_IDENTITY_*`,
  `WEIGHTED_RENDER_FAMILIES += ("tcq_trellis",)`), ABI bump handling,
  `scratch/`-based profiling
* `prismaquant/render_score.py` — register `trellis` mechanism
* `tests/test_trellis_render_mechanism.py` — new, 6 tests
* `docs/ARCHITECTURE.md` — re-stamped 2026-08-31 `muse/wo-b-trellis-render-20260831`
