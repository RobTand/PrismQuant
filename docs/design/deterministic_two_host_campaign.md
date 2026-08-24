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

Every numerical stage enters through the immutable snapshot-owned bootstrap
inside the pinned Docker image. The launcher uses fixed argv, canonical
container paths, read-only source/model/dataset mounts, the invoking non-root
UID/GID, and private writable cache/state mounts. Both the host admission and
the container re-verify the snapshot closure. The image is named by its full
registry RepoDigest, never a mutable tag or backend-local image ID.

Local and SSH transports have identical typed operations. OpenSSH is invoked
with `shell=False`, forwarding disabled, and batch mode enabled. Because an SSH
server still passes its remote command through a shell, variable request data
is carried only as canonical JSON to a fixed Python helper; it is never
interpolated into the remote command. Long jobs publish durable start and exit
receipts so a dropped SSH connection is not interpreted as success or failure.

Barrier artifacts cross hosts only through a content-addressed transfer:

1. reject symlinks, special files, duplicate paths, and traversal;
2. hash a sorted source member ledger;
3. copy into a unique temporary destination;
4. reconstruct and compare the ledger on the destination host; and
5. fsync and publish without clobbering an existing destination.

An existing exact destination is a validated resume. An existing divergent or
unverifiable destination is a refusal, never an overwrite.

## Receipts, Resume, and Failure

Each transition durably records the manifest digest, stage and assignment,
dependency receipt digests, fixed command digest, host identity, start/end
times, exit status, log and telemetry hashes, and output member ledger. A
campaign lock prevents concurrent coordinators. On restart, PrismaQuant checks
the receipts and existing stage-specific artifacts and selects the unique next
legal transition. It does not infer completion from a PID, pathname, log tail,
or SSH exit alone.

Retries are bounded by the manifest and reuse only stage implementations with
an exact validation or resume contract. Conflicting receipts, a lost process
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
