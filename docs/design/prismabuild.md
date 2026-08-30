# PrismaBuild — distributed campaign execution

**Status: DETERMINISTIC CORE + SLURM ADAPTER BUILT / NOT LIVE-DEPLOYED.** The
dependency-free action-key, immutable-CAS, and local-worker core lives in
`prismaquant/prismabuild.py`; the fail-closed SLURM resource transport lives in
`prismaquant/prismabuild_slurm.py`. The adapter submits a canonical immutable
action request with exact argv, `--export=NONE`, and explicit resources, then
accepts only a scope-correct CAS receipt as success. A SLURM `COMPLETED` state
without that receipt is a failed action. `tools/prismabuild_worker.py` is the
direct batch-script entry point. The SLURM daemons, Dagster layer, and
observability stack in the chosen design below are not deployed, and nothing
in the live quantization pipeline depends on PrismaBuild yet.

## Problem

Campaign work (screens, per-point KL fan-outs, per-tensor encodes, A/Bs)
serializes behind one coordinator's attention while GPUs and CPUs idle.
Utilization is bursty; dispatch is manual (ssh + systemd-run). We want
independent work to run the moment its inputs exist, across a heterogeneous
fleet, without hand dispatch — and with strong observability.

## Fleet inventory (2026-08)

| host class | machines | role |
|---|---|---|
| `gb10` | sparky, sparklina (GB10, 128 GB unified, sm_121) | gold path: probes, validated KL, ship gates, big renders. sparky default-**reserved** for interactive/campaign use (reservation, NOT cluster exclusion). |
| `rocm-16g` | Rob's + son's 9800X3D/9070 XT desktops | 0.6B screen tier; brute-force search/encode (trellis Viterbi, permutation/gauge searches, CB training) |
| `strix-32g` | son's AI Max laptop (32 GB unified, opportunistic) | 4B screen tier (the size 16 GB cards can't hold) |
| `cpu-x86-large` | dl380g10 (80 cores, 300 GB, NFS server) | page-cache pre-warm (vmtouch), data-gravity work (hashing, repacking, shard merges), fp64 references, bootstraps, CPU encode farms. Batch niced/cgroup-capped: storage QoS outranks batch. |
| — | M5 Mac mini | below the value line; not a tier |

Data plane: /mnt/shared (NFS, dl380, 38 T, ~1 GB/s). Code plane: git SHA
checkout per job + per-arch venvs (envs cannot be shared across
aarch64-CUDA / x86-ROCm / Strix). Trust plane: munge-authenticated SLURM =
trusted-cluster model; joining a machine puts it inside the boundary.

## Chosen stack (industry-standard per layer; no roll-your-own queue)

1. **SLURM** — resource layer. Partitions = host classes; GRES = GPU slots;
   QOS/priority = gold-path-never-waits; standing reservation on sparky;
   slurmdbd accounting. Handles nodes joining/leaving (laptops).
2. **Dagster** — DAG + memoization layer. Chosen over Snakemake because two
   hard requirements point at it: (a) native asset memoization keyed by
   `code_version` + upstream input versions — exactly the cache model below;
   (b) best-in-class live observability (run timelines, per-step logs, asset
   lineage/staleness UI). Known seam we own: Dagster→sbatch run-launcher
   glue is community-grade (~100 LoC).
3. **CAS on /mnt/shared** — content-addressed store; payload path = key
   hash. A naming convention + hashing helper, not a system.
4. **Prometheus + Grafana + Loki + Alertmanager** on dl380 — node_exporter,
   dcgm-exporter (GB10), AMD SMI exporter, slurm-exporter; job logs via
   promtail; **our receipts pushed as metrics** so campaign progress (KL per
   point, stage durations, gate outcomes) is graphable, not just machine
   health. Orchestrator-independent; build once.

## Cache/action-key semantics (the Bazel steal)

Result address = hash(input artifacts, **code closure**, params, env-that-
matters). Rules:
- **Code closure, not repo SHA** — per-task declared file lists (stage-7's
  contract-pinned dependency list is the house precedent). Bias to
  over-declare: over-invalidation wastes compute; under-invalidation serves
  stale results.
- **Generation vs measurement tasks**: generation (encodes, permutation/
  gauge searches, CB books — discrete outputs re-scored later) EXCLUDES
  host from the key → any box's result is valid ("surrogates generate,
  real KL selects" applied to hardware). Measurement (KL, PPL, probe)
  INCLUDES host-class + toolchain — numerics don't transfer across
  architectures; gold path pinned to `gb10`.
- **Deterministic vs stochastic** task classes: deterministic entries may be
  verified by recompute; stochastic (probe backward is recorded
  non-bit-reproducible) get run-once / first-result-wins.
- Re-enqueue of an existing key = cache hit = no-op. Speculative enqueueing
  is therefore safe: superseded keys are simply never requested again.

**Restart economics + provenance (Rob, 2026-08-26).** Reruns become replays:
after a failure or a code fix, re-enqueueing the whole campaign returns
cached results for every task whose key is unchanged and recomputes only
what the edit actually invalidated. The stage-7 trellis chain is the
motivating counter-example: its contract binds ONE closure over the whole
chain, so each of the eight 2026-08 re-arms re-ran plan + preflight +
calibration (~10 min each) even when the edit touched only the spotcheck
gate — the calibration was recomputed byte-identical to v2 four consecutive
times, empirical proof that per-task closures would have made those re-arms
near-instant. The key is also the provenance: hash(inputs, code closure,
params, env) is machine-checkable identity, stronger than prose receipts,
and deterministic-class entries can be audited by recompute-and-compare.
Honest caveats: stochastic tasks (probe backward is recorded
non-bit-reproducible) get run-once/first-result-wins — their entry is the
*canonical* result, pinned but not re-derivable; and a cached measurement is
valid only under its host-class key (a gb10 KL never answers an x86 query).

## Speculative tier (pre-probe parallelism)

The probe is the true DAG barrier — everything decision-relevant consumes
its outputs. Input-complete before it, enqueue-able the moment a model's
tensor inventory exists:
- RTN-tier renders + weight-space error tables (importance-independent;
  real pipeline inputs: legality/fallback/statistics).
- Candidate GENERATION under weight-only scores (doctrine-legal proposals).
- Staging, hashing, FP8 source-map verification, census metadata,
  page-cache pre-warm.
Marked `speculative: true`, routed only to idle non-gold hardware, governed
by a disk budget (≥10 % free is non-negotiable).

## Memory-pressure corollary (Rob, 2026-08-26)

SLURM's cgroup enforcement (`--mem` per job) eliminates the **inter-job** OOM
class: work that doesn't fit is never placed, and a job exceeding its declared
budget is killed by its own cgroup instead of the kernel OOM-killing a random
victim (or, worse, /tmp). Lowering worker counts/capacity per node is exactly
that knob. This is allocation-time enforcement, not a reactive monitor — the
opposite of the recorded Ray landmine, where a userspace memory monitor on
unified memory killed healthy ranks. On GB10, `--mem` must be sized for GPU
allocations too (one physical pool).

It does NOT retire the **intra-job** LRU: layer streaming exists because one
task's working set (a 328 GB model through a 128 GB box) exceeds physical
memory, and no scheduler shrinks a model. What sharding does buy: per-layer /
per-tensor tasks have few-GB working sets, so as heavy stages shard, the OS
page cache + dl380's 300 GB NFS backing absorb re-reads and the LRU's *role*
shrinks. The floor that remains: order-dependent monolithic forwards (the
sequential probe on a 314B teacher) keep streaming regardless.

## Boundaries that do not move

- **Certification stays PrismaQuant's.** Shipcards, fail-closed gates,
  receipts, provenance stamps run inside jobs. The orchestrator schedules
  and remembers; it never certifies.
- run-pipeline.sh survives as the per-run executor inside jobs (v0: one
  task = one pipeline run; later versions shard heavy stages: per-point KL,
  per-tensor encodes, per-expert measurements, parallel coord-descent).

## Rejected alternatives (with reasons)

- **Airflow / k8s**: ops weight, time-oriented, poor measured-KL branching.
- **Ray**: recorded unified-memory landmine (OOM monitor kills ranks on
  GB10); runtime-env sync across three architectures.
- **Bazel directly**: right cache semantics, wrong job model — no honest
  representation of long exclusive-GPU jobs; hermeticity dies on 328 GB NFS
  inputs; BUILD-file loop taxes research-pace code churn; cache presumes
  determinism our probe lacks. We take its action-key discipline, not the
  tool. ("Bazel's cache discipline on SLURM's job model.")
- **Snakemake** (as the DAG layer): file-native and simple, but weak live
  observability and mtime/param triggers rather than content keys; loses to
  Dagster on the two requirements Rob weighted hardest. Remains the
  fallback if the Dagster–SLURM seam proves painful.
- **Roll-your-own queue dir**: explicitly declined by Rob 2026-08-26.

## Sequencing

1. (May precede GLM v1, CPU-side only) Minimal SLURM: controller on dl380,
   slurmd on both Sparks, `interactive` reservation on sparky; drive
   existing scripts via sbatch unchanged.
2. After GLM v1: observability stack; Dagster pilot on the speculative tier
   (GLM RTN render sweep = shakedown asset); family nodes join as
   `rocm-16g`/`strix-32g`.
3. Then: shard heavy stages; GLM/Qwen validation fan-outs as the first
   production campaign on the full stack.
