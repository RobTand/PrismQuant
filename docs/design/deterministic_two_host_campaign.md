# Deterministic Two-Host Campaigns

## Purpose

PrismaQuant owns the scheduling of ordinary multi-host quantization work. An
agent may author or review a campaign manifest, diagnose a refusal, or improve
the implementation, but it is never a worker dispatcher or a barrier. Once a
sealed manifest is launched, machine-readable state alone determines every
legal transition.

The first production consumer is the two-host Qwen3.8-27B BF16 probe and strict
RTX 4090 FP8 Gridbook cost burn. The orchestration layer is deliberately
transport-neutral so later NVFP4 campaigns can reuse it without copying the
quantization, cache, or scheduling machinery.

## Closed Contract

One versioned manifest declares:

- exactly two named hosts and one stable partition/stripe assignment per host;
- one coordinator host, while the control process may run on either host;
- local or SSH transport, with SSH credentials left to OpenSSH configuration;
- expected hostname, architecture, non-root UID/GID, GPU count/name/UUID, and
  driver constraints;
- host-local absolute roots for the model, dataset, run tree, private worker
  state, and immutable runtime snapshots;
- the full producer commit/tree/closure and registry RepoDigest;
- an artifact target separate from execution hardware: RTX 4090/sm89,
  validation-only, BF16 source, context-first 18,000,000,000-byte ceiling,
  physical FP8-CB K4/K16/K48 plus FP8_E4M3, and a BF16 terminal;
- calibration, probe, cache, compile, retry, telemetry, and output policy; and
- minimum free disk and host memory required before admission.

The manifest contains no shell fragments or arbitrary commands. PrismaQuant
maps a typed stage name to fixed argument, mount, and environment builders.
Paths are absolute, normalized, confined to their declared roots, and rejected
if they contain traversal or alias another declared mutable root. A canonical
manifest digest binds all state and receipts.

## State Graph

The campaign is a fixed directed acyclic graph:

```text
host admission + immutable snapshot/image verification
                         |
        calibration -> coordinator source identity
                         |
                 run contract -> cover
                         |
              host-local source identities
                         |
               +---------+---------+
               |                   |
             CE[0]               CE[1]
               +---------+---------+
                         |
                   global CE barrier
                         |
               +---------+---------+
               |                   |
          Fisher[0]             Fisher[1]
               +---------+---------+
                         |
        probe/activation merge -> column weights
                         |
            execution attestation -> burn plan
                         |
               +---------+---------+
               |                   |
           burn[0]              burn[1]
               +---------+---------+
                         |
                    burn merge
                         |
                     allocation
```

Only the CE, Fisher, and burn map pairs may run concurrently. A reducer cannot
start until both exact inputs are present and independently validated. Worker
source caches, compiler state, probe work shards, and burn checkpoints stay
host-private and are never transferred.

## Execution and Transfer Law

The public `prismaquant.rtx4090_two_host_campaign` `run`, `resume`, and
`verify` commands construct the live application directly. The application
owns host admission, transports, barriers, scheduling, monitoring, recovery,
and final lease release. The documented shell workflow is explanatory and
diagnostic only; invoking it stage by stage is not an operational mode. These
live commands must enter through the manifest's local immutable
`prismaquant_source_bootstrap.py`: the application refuses unless its own
module and snapshot verifier resolve inside that root and the full snapshot
revalidates against the manifest's commit, tree, and closure digest.

Every numerical stage enters through the immutable snapshot-owned bootstrap
inside the pinned Docker image. Before creating even a campaign directory or
installing the remote helper, the runtime runs a fixed read-only probe on both
endpoints and requires the manifest's exact hostname, UID/GID, GPU name/UUID,
compute capability, and device count. The launcher uses fixed argv, canonical
container paths, read-only source/model/dataset mounts, the invoking non-root
UID/GID, and private writable cache/state mounts. CPU-only stages are not given
GPU devices and explicitly set `NVIDIA_VISIBLE_DEVICES=void`. Both the host admission and the container re-verify the snapshot
closure. The image is named by its full registry RepoDigest, never a mutable
tag or backend-local image ID.

Local and SSH transports have identical typed operations. OpenSSH is invoked
with `shell=False`, forwarding disabled, and batch mode enabled. Because an SSH
server still passes its remote command through a shell, variable request data
is carried only as canonical JSON to a fixed Python helper; it is never
interpolated into the remote command. Long jobs publish durable start and exit
receipts so a dropped SSH connection is not interpreted as success or failure.
Detached workers are identified by boot ID plus process start ticks, not PID
alone; zombie/dead states are terminal, not live. Every fixed stage has a
finite timeout. Each container receives a private deterministic name and CID
file outside its mounts; an inner supervisor stops, kills, and removes only
the exact manifest/host/work-labeled container before the outer transport
timeout expires. The helper and durable job
state live in a campaign-addressed sibling control tree outside every
container-writable bind mount; the workload's `worker_state_root` can never
replace helper code or forge a transport receipt. `status` reads only the
controller state tree; `verify` may inspect sealed local and remote artifacts
but does not create directories, reinstall helpers, or launch work.

Derived helper and transfer paths are proven disjoint from every declared host
root before the endpoint identity probe or any mutation. Controller state is
also validated before runtime construction: every existing ancestor must be a
real directory, and its lexical and resolved paths must remain disjoint from
all local declared and derived control/transfer roots.

Barrier artifacts cross hosts only through a content-addressed transfer:

1. reject symlinks, special files, duplicate paths, and traversal;
2. hash a sorted source member ledger;
3. copy into a unique temporary destination;
4. reconstruct and compare the ledger on the destination host; and
5. fsync and publish without clobbering an existing destination.

The content-addressed staging directory is a sibling of its destination root
on the same filesystem, so the final no-clobber rename cannot cross devices
and a workload container cannot mutate transfer control state. Run artifacts
and immutable snapshots use separate staging roots. A durable
barrier receipt is reused by validating that exact stored receipt and both
current endpoint manifests; a fresh timestamped transfer receipt is not
fabricated during verification.

An existing exact destination is a validated resume. An existing divergent or
unverifiable destination is a refusal, never an overwrite.

## Receipts, Resume, and Failure

Admission is deliberately split. Before any campaign work, each host proves
its exact hostname/UID/GID/GPU, an idle device, the dataset digest, real model
and snapshot roots, resource minima, and ownership of a durable host-wide GPU
lease. Model-content admission follows the two fixed
`worker_source_identity` assignments: each host validates its live checkpoint
and publishes the same path-neutral portable content digest from its own
freshly created or exactly revalidated cache. This avoids requiring a cache that the campaign has not produced
yet without weakening the model identity gate.

Each transition durably records the manifest digest, stage and assignment,
dependency receipt digests, fixed command digest, host identity, start/end
times, exit status, log and telemetry hashes, and output member ledger. A
campaign lock prevents concurrent coordinators. On restart, PrismaQuant checks
the receipts and existing stage-specific artifacts and selects the unique next
legal transition. It does not infer completion from a PID, pathname, log tail,
or SSH exit alone.

Pre-admission, model-admission, and cross-host barrier receipt digests are
also process preconditions. They enter the `RunRequest` identity and the
execution receipt, and are recomputed from the stored exact receipts during
verification. Immediately before each new container start, the application
adopts the campaign lease, proves the exact GPU is idle, proves no prior
labeled campaign container is still running, and persists a request-bound
guard journal. The fixed bare-host launch wrapper repeats those checks while
holding the GPU lease lock and carries that lock into the foreground Docker
client. Thus a lost detached worker cannot cause a retry to overlap its
surviving client or daemon-side container. A restarted controller may adopt an
already-created transport job only when that exact request already has a
valid durable guard; it does not rerun an idleness check after the process has
started. The CID-bound supervisor uses a shorter inner deadline and proves the
exact labeled container absent before timeout cleanup returns. CPU stages set
`NVIDIA_VISIBLE_DEVICES=void`, so neither the image nor a default NVIDIA host
runtime can silently expose a GPU. Leases are released automatically only after the coordinator result,
all outputs, all preconditions, and all guard journals verify. Any failure
retains the lease so `resume` has one unambiguous owner.

Lease release publishes and fsyncs an exact hard-linked lease tombstone before
unlinking the active lease. The application accepts `already_absent` only from
a retry carrying that exact lease and only when the prior exact release request
has durable terminal-failure evidence; an unproven fresh absence is refused.
The tombstone also makes one sealed campaign generation single-shot: it cannot
reacquire the same host/GPU after successful release.

Retries are bounded by the manifest and reuse only stage implementations with
an exact validation or resume contract. Host actions use indexed physical job
IDs: exact terminal failures remain immutable evidence, while a later `resume`
can spend a new bounded ID and finish the missing side of a partially completed
two-host admission. Failed numerical attempts likewise retain their valid
guard journals; final verification requires every successful guard and accepts
extras only when they are exact legal attempt IDs from the sealed retry
catalog. Conflicting receipts, a lost process
without a durable exit receipt, identity drift, a busy unexpected GPU, missing
barrier input, or ambiguous output state fails closed. Recovery consists of a
code fix or a newly reviewed explicit input, not an agent choosing which file
to delete or which stage to skip.

## Utilization Evidence

GPU telemetry is part of the result rather than an informal observation.
Sampling begins before each GPU-bearing child and ends after its durable exit.
Raw samples bind monotonic time, GPU name/UUID, utilization, memory-controller
utilization when available, power, temperature, and clocks. GB10 has unified
memory, so the sampler also records host `MemAvailable`; an unavailable VRAM
field is represented explicitly rather than converted to zero.

The canonical summary reports per host and stage:

- sample count and coverage duration;
- mean, median, and p95 GPU utilization;
- fractions of samples above 0%, 50%, and 90%;
- minimum host-available memory; and
- whether live CUDA work occurred.

One narrow exception is provenance-bearing rather than performance-bearing:
an exact host-local source-identity cache may validate and return before CUDA.
Its byte-exact machine receipt must say `validated_reuse`, bind the expected
cache path and portable model digest, and is recorded as an explicit activity
waiver. A fresh cache build and every numerical GPU stage still require at
least one positive utilization sample.

Campaign reporting separates GPU-active stages from calibration transfers,
barrier waits, startup, and serial reductions. The first reviewed physical
trace establishes a performance baseline; a future numeric ship threshold must
be declared before the comparison run rather than invented afterward.

## Campaign-Specific Boundaries

The initial graph reuses the existing `sample_parallel_probe`,
`incremental_probe`, and `rtx4090_fp8_burn` producers. It adds no alternate
weight cache, activation cache, numerical reducer, renderer, or allocator. The
burn ends at a layer assignment; export and serving qualification remain later
explicit stages. The same orchestration contract can schedule an NVFP4 burn
once that producer exists, without an agent or a new cluster control plane.
