# PrismaBuild — distributed campaign execution

**Status: DETERMINISTIC CORE + SLURM + OPTIONAL DAGSTER LAYER BUILT / NOT
LIVE-DEPLOYED.** The
dependency-free action-key, immutable-CAS, and local-worker core lives in
`prismaquant/prismabuild.py`; the fail-closed SLURM resource transport lives in
`prismaquant/prismabuild_slurm.py`; and the optional asset/DAG adapter lives in
`prismaquant/prismabuild_dagster.py`. The SLURM adapter submits a canonical
immutable action request with exact argv, `--export=NONE`, and explicit
resources, then accepts only a scope-correct CAS receipt as success. A SLURM
`COMPLETED` state without that receipt is a failed action.
`tools/prismabuild_worker.py` is the direct batch-script entry point. The
Dagster adapter constructs deterministic assets from sealed action keys, binds
each edge to an expected CAS output digest, and materializes only after
re-reading that receipt and payload from the CAS. The SLURM daemons, Dagster
service, and observability stack in the chosen design below are not deployed,
and nothing in the live quantization pipeline depends on PrismaBuild yet.

## Problem

Campaign work (screens, per-point KL fan-outs, per-tensor encodes, A/Bs)
serializes behind one coordinator's attention while GPUs and CPUs idle.
Utilization is bursty; dispatch is manual (ssh + systemd-run). We want
independent work to run the moment its inputs exist, across a heterogeneous
fleet, without hand dispatch — and with strong observability.

## Target fleet inventory (design only; not deployed, 2026-08)

This table is a proposed PrismaBuild placement inventory, not discovered or
enforced cluster state. In particular, PrismaBuild has not installed a SLURM
controller or node daemon, created the named partitions/reservations, or
attested these machines through a live allocation.

| host class | machines | role |
|---|---|---|
| `gb10` | sparky, sparklina (GB10, 128 GB unified, sm_121) | proposed gold path: probes, validated KL, ship gates, big renders. The design reserves sparky for interactive/campaign use; no PrismaBuild reservation is live. |
| `rocm-16g` | Rob's + son's 9800X3D/9070 XT desktops | 0.6B screen tier; brute-force search/encode (trellis Viterbi, permutation/gauge searches, CB training) |
| `strix-32g` | son's AI Max laptop (32 GB unified, opportunistic) | 4B screen tier (the size 16 GB cards can't hold) |
| `cpu-x86-large` | dl380g10 (80 cores, 300 GB, NFS server) | page-cache pre-warm (vmtouch), data-gravity work (hashing, repacking, shard merges), fp64 references, bootstraps, CPU encode farms. Batch niced/cgroup-capped: storage QoS outranks batch. |
| — | M5 Mac mini | below the value line; not a tier |

The intended data plane is `/mnt/shared` (NFS, dl380, 38 T, ~1 GB/s); it is not
a PrismaBuild-deployed shared CAS today. The intended code plane is a git-SHA
checkout per job plus per-architecture venvs (envs cannot be shared across
aarch64-CUDA / x86-ROCm / Strix). The proposed trust plane is
munge-authenticated SLURM: joining a machine would put it inside that trusted
cluster boundary.

## Target stack (design only; no services installed)

The components below are the selected deployment design. The repository
implements and tests the PrismaBuild core, SLURM command adapter, and optional
Dagster definitions, but it does not install or operate SLURM, `slurmdbd`, a
Dagster daemon/webserver, a shared PrismaBuild CAS, or the listed telemetry
services.

1. **SLURM** — resource layer. The deployment would use partitions as host
   classes, GRES as GPU slots, QOS/priority for the gold path, a standing
   reservation on sparky, and `slurmdbd` accounting. It would handle nodes
   joining and leaving (laptops).
2. **Dagster** — DAG + memoization layer. Selected over Snakemake because two
   hard requirements point at it: (a) native asset memoization keyed by
   `code_version` + upstream input versions — exactly the cache model below;
   (b) best-in-class live observability (run timelines, per-step logs, asset
   lineage/staleness UI). Known seam we own: Dagster→sbatch run-launcher
   glue is community-grade (~100 LoC).
3. **CAS on /mnt/shared** — intended content-addressed store; payload path =
   key hash. A naming convention + hashing helper, not a deployed service.
4. **Prometheus + Grafana + Loki + Alertmanager** on dl380 — the proposed
   stack would use node_exporter, dcgm-exporter (GB10), AMD SMI exporter, and
   slurm-exporter, with job logs via promtail. Receipts would be pushed as
   metrics so campaign progress (KL per point, stage durations, gate outcomes)
   is graphable, not just machine health. It would remain orchestrator-
   independent.

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
- Re-enqueue of an existing verified key is a tested cache-hit no-op. A future
  speculative policy could build on that property, but no such enqueueing or
  superseded-key scheduler exists yet.

### Worker preflight and execution attestation

Scheduler placement is intent, not producer identity. `run-local` accepts no
`--worker-id`, `--platform-key`, or `--host-class` arguments. Before a cache
miss executes, `prismaquant.prismabuild.preflight_action` emits and validates a
`prismaquant.prismabuild.worker_attestation.v1` record bound to the action key:

- `platform_key` is derived from the live lower-case OS and machine plus the
  single visible NVIDIA compute capability, when present (for example,
  `linux-aarch64-sm121`). Heterogeneous visible capabilities are ambiguous and
  refuse.
- `worker_id` is the live hostname locally or SLURM's node name inside an
  allocation. A `host_class_keyed` action is SLURM-only: its class must equal
  the job partition or an exact constraint token, and the claimed numeric job
  must occur in `/proc/self/cgroup`. Merely setting `SLURM_*` variables is not
  attestation.
- The resolved regular file behind `argv[0]` is hashed before execution and
  checked again before publication. Nonportable actions must bind that digest
  and byte count as `environment.toolchain.{argv0.sha256,argv0.bytes}`, plus
  the exact system, machine, and libc ABI fields. Their
  toolchain may contain only preflight-backed fields (`python`, `torch`,
  `transformers`, `vllm`, `gridbook`, OS/machine/libc, CUDA capability, NVIDIA
  driver, and the executable identity); every declared field must verify.
  NVIDIA workers additionally require the CUDA capability and driver fields.
- Every nonportable `action.inputs` digest must already exist and verify in the
  PrismaBuild CAS before argv starts. Portable actions preserve the existing
  external-input contract: CAS-resident inputs and recognized toolchain fields
  are verified when possible, while unresolved inputs and descriptive
  toolchain fields remain permitted and are visibly absent from the
  attestation's verified subsets.

Two limits are explicit. For portable actions the observed executable hash is
receipt provenance, not a newly required action-key field; callers that need
the executable to participate in cache identity must use a nonportable scope
and the `argv0.*` toolchain fields. Input preflight proves that the declared
CAS bytes exist and match before execution, but the sealed argv/code remains
responsible for resolving and consuming those bytes; process provenance is not
an OS-level proof of every file read.

The attestation becomes `producer` in the self-hashed
`prismaquant.prismabuild.cas_receipt.v2` receipt. CAS lookup replays its action,
scope, platform derivation, host-class evidence, executable identity, verified
toolchain, verified-input subset, and self-digest before accepting the result.
The `preflight` CLI prints the same machine-readable record without executing
the action. This is process/platform provenance, not a cryptographic quote. In
the target deployment, the trust boundary would be the munge-authenticated,
cgroup-enforced cluster and its shared CAS; that boundary is not live today.

**Intended restart economics + provenance (Rob, 2026-08-26).** Unit-tested
local/CAS semantics are designed so a rerun can become a replay: after a
failure or code fix, re-enqueueing a campaign should return cached results for
unchanged keys and recompute only what the edit invalidated. No end-to-end
SLURM/Dagster campaign replay has run, so this is not a measured deployment
claim. The stage-7 trellis chain is a motivating counterexample from the
pre-PrismaBuild workflow: its contract bound one closure over the whole chain,
so each of the eight 2026-08 re-arms re-ran plan + preflight + calibration
(~10 min each) even when the edit touched only the spotcheck gate. Four
calibrations were byte-identical to v2; that supports the value of finer task
closures but does not prove the timing or reliability of a deployed
PrismaBuild replay. The key is also intended as provenance: hash(inputs, code
closure, params, env) is machine-checkable identity, and deterministic-class
entries can be audited by recompute-and-compare.
Honest caveats: stochastic tasks (probe backward is recorded
non-bit-reproducible) get run-once/first-result-wins — their entry is the
*canonical* result, pinned but not re-derivable; and a cached measurement is
valid only under its host-class key (a gb10 KL never answers an x86 query).

### Optional Dagster orchestration (implemented, not deployed)

`prismaquant.prismabuild_dagster` is an optional-import adapter over the core
and SLURM resource layer; importing PrismaQuant still does not import or require
Dagster. `ActionSpec` binds the sealed action, checkout, exact SLURM resources
and placement, bounded same-job requeue policy, and content-addressed upstream
dependencies. An edge is the tuple `(upstream action key, downstream input id,
result sha256, result bytes)`. Graph construction refuses an edge unless that
tuple is also present exactly in the downstream action's sealed `inputs`, and
uses a key-sorted topological order.

Native definitions use one asset key per action key and set `code_version` to
that full key. Dagster-level retries are disabled: the bounded retry path uses
`SlurmAdapter.requeue` on the original allocation and sealed action within one
live runner invocation, avoiding a second allocation on that in-memory retry
path. The returned SLURM job id is not durably mapped to the action key. If the
orchestrator dies after `sbatch` and before a CAS receipt, a fresh invocation
cannot prove whether an exact-key allocation is still live and may submit
again. Crash recovery therefore remains unimplemented and must fail closed or
gain an externally durable/adoptable submission identity before deployment. A
cache hit, upstream dependency, or successful SLURM resolution is accepted only
after an independent `PrismaBuildCAS.lookup()` verifies the exact receipt,
producer scope, and blob bytes. Therefore Dagster run state and materialization state
are views of CAS truth, never certification themselves. The optional package
extra is `prismaquant[prismabuild]` (supported `>=1.13,<2`, checked against
1.13.20); no daemon, webserver, workspace, or scheduler installation is
performed by the repository.

## Proposed speculative tier (not implemented)

There is no `speculative` field, idle-hardware router, or disk-budget policy in
the current action schema or adapters. The following is target behavior for a
future scheduler policy, not a capability of the tested implementation.

The probe is the true DAG barrier — everything decision-relevant consumes
its outputs. Input-complete before it, enqueue-able the moment a model's
tensor inventory exists:
- RTN-tier renders + weight-space error tables (importance-independent;
  real pipeline inputs: legality/fallback/statistics).
- Candidate GENERATION under weight-only scores (doctrine-legal proposals).
- Staging, hashing, FP8 source-map verification, census metadata,
  page-cache pre-warm.
Such actions would be marked explicitly, routed only to idle non-gold hardware,
and governed by a disk budget (≥10 % free is non-negotiable). The spelling
`speculative: true` is illustrative, not a currently accepted schema field.

## Memory-pressure hypothesis (not live-validated; Rob, 2026-08-26)

The adapter emits SLURM `--mem`, but this repository has not validated a live
controller/cgroup configuration or GB10 unified-memory accounting. With
correctly requested limits and a correctly configured cluster, cgroups should
isolate an over-budget job instead of letting the kernel OOM-kill an unrelated
victim. The current code and mocked tests do **not** establish that work which
does not fit is never placed, that requested limits are correctly sized, or
that GPU allocations in GB10's unified physical pool are isolated. Those
claims require a live allocation plus cgroup and Netdata evidence. Lowering
worker counts/capacity per node is the intended allocation-time knob, not a
reactive userspace monitor like the recorded Ray landmine.

Even a validated scheduler limit would not retire the **intra-job** LRU: layer
streaming exists because one task's working set (a 328 GB model through a
128 GB box) exceeds physical memory, and no scheduler shrinks a model. What
sharding may buy: per-layer/per-tensor tasks have few-GB working sets, so as
heavy stages shard, the OS page cache plus dl380's 300 GB NFS backing may
absorb re-reads and shrink the LRU's role. The floor that remains:
order-dependent monolithic forwards (the sequential probe on a 314B teacher)
keep streaming regardless.

## Target boundaries that do not move

- **Certification stays PrismaQuant's.** When deployed, shipcards, fail-closed
  gates, receipts, and provenance stamps would run inside jobs. The
  orchestrator would schedule and remember; it would never certify.
- `run-pipeline.sh` remains the intended per-run executor when PrismaBuild is
  deployed (v0: one task = one pipeline run; later versions may shard heavy
  stages: per-point KL, per-tensor encodes, per-expert measurements, parallel
  coord-descent). No live pipeline run currently executes inside a PrismaBuild
  SLURM or Dagster job.

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
