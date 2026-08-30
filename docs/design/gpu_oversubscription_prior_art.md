# GPU oversubscription prior art for PrismaQuant on DGX Spark

**Status:** research survey and design recommendation, 2026-08-30.  This
document critiques the governor in [`tools/pqwork.py`](../../tools/pqwork.py)
as documented by [`work_queue.md`](work_queue.md); it does not change that
implementation.

## Executive decision

The useful prior art is not a replacement cluster scheduler.  It is three
policies to incorporate into the existing receipt-gated queue: Linux PSI as a
second system signal, progress-aware preemption, and a stateful speculation
guard.  MIG is unavailable on this GB10, generic GPU-sharing stacks do not
isolate the one physical memory pool, and cgroup memory limits do not account
the CUDA portion of that pool on the measured software stack.

The four requested decisions are:

| Question | One recommendation | Adoption cost |
|---|---|---|
| **A. Eviction trigger** | **Combine the signals.** Keep the MemAvailable trajectory as the anticipatory OOM guard, add global memory PSI, stop admissions on sustained `some`, and evict on calibrated `full` pressure **or** a projected hard-reserve crossing. PSI is the better signal of harmful reclaim; it is not the better sole predictor of a fast allocation cliff. | **Medium:** one `/proc/pressure/memory` reader, hysteresis/state, shadow telemetry on both boxes, and local threshold calibration under the same workloads. No new daemon or scheduler. |
| **B. Victim** | **Extend, do not discard, priority-then-newest.** Prefer restartable work, then lowest priority; protect at least one incumbent so the node makes progress; within a priority class minimize uncheckpointed work per expected GiB released and select only enough victims to recover the target. Use newest only when no checkpoint/progress estimate exists. | **Medium to high:** checkpoint/progress fields and a bounded checkpoint hook are straightforward; trustworthy per-job freed-memory estimates are harder because CUDA pages are not memcg-accounted here. |
| **C. `memory.high`** | **Do not adopt it as a GPU throttle.** On this GB10 stack a touched 2 GiB CUDA allocation crossed neither `memory.high` nor `memory.max`; both limits saw only about 378 MiB while system MemAvailable fell by about 2.30 GiB. Keep cgroups for host-charged memory and job containment, but keep kill-and-requeue as the unified-pool actuator. | **Zero for rejection; low for honesty:** retain the existing cgroup plumbing, label its guarantee host-side only, and keep a small driver-upgrade conformance test. |
| **D. More aggressive admission** | **Use one bounded speculation token per node plus a resident-generation no-good record.** Split today's single threshold into a soft admission watermark and a lower, non-borrowable hard reserve. Protect one incumbent, admit at most one speculative job below the soft watermark when PSI is quiet, and, after an eviction, forbid the same job/resident combination until an incumbent exits or the observed capacity changes. This turns a failed overlap into learned state instead of a timer loop. | **Medium:** a few durable item/worker state fields, serialized speculative starts, generation invalidation, and deterministic transition tests. Application checkpoint support raises the cost but reduces wasted compute substantially. |

Priority order is therefore: (1) stop treating cgroup limits as a total
GB10-memory boundary; (2) add PSI beside, not instead of, MemAvailable; (3)
make speculative admission remember failed co-residencies.  Victim refinement
is useful, but the present priority/newest ordering is not the main defect.

## Scope, evidence, and platform constraints

This survey uses three evidence labels:

* **Documented** means a project manual or the original research paper says
  it.  Every such claim has an inline link to the primary source.
* **Measured here** means a read-only or bounded local experiment on `sparky`
  on 2026-08-30.  It is evidence for this kernel/driver/runtime combination,
  not a promise about every Linux or CUDA release.
* **Inference** means the cited mechanism is being mapped to PrismaQuant's
  workload.  The inference is stated separately from what its source claims.

The hardware premise is unusually important.  NVIDIA describes DGX Spark as
an integrated CPU/GPU with 128 GB of unified LPDDR5x and a 140 W GB10 SoC
envelope ([hardware specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)).
Its porting guide is explicit that CPU and iGPU share the same physical memory
without a fixed carve-out
([UMA architecture](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)).
NVIDIA also documents `nvidia-smi` framebuffer memory as unsupported on this
iGPU and warns that `cudaMemGetInfo` is not a complete allocatable-memory
measure on UMA because system reclaim and swap affect it
([DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).
Consequently, “move it to CPU” may change which processor accesses an object,
but it is not a spill tier that returns bytes to a separate VRAM pool.  Any
scheduler whose safety argument depends on host offload freeing dedicated GPU
memory is inapplicable here; this is an inference from NVIDIA's documented
single physical pool, not a claim about discrete GPUs.
Thus the local observations supplied with this task—MemAvailable is the useful
whole-pool ceiling, GPU memory telemetry is absent or misleading, and GPU
utilization cannot distinguish stalled from productive work—are consistent
with the platform design.  The 5.83x throughput delta at an unchanged 96%
utilization is local evidence; the survey does not generalize that number to
other GPUs.

The target workload is also unlike the one most cluster schedulers optimize:
two interchangeable nodes, mainly one long PyTorch process per job, no tenant
fairness requirement, and receipt-gated idempotence.  Killing is logically
safe, but without an application checkpoint it discards all compute since the
job began.  “Restartable” and “cheap to preempt” are therefore not synonyms.

## Current governor: what is sound and what is missing

The current design has several sound choices:

* It uses the physical-pool signal (`MemAvailable`) instead of inventing a
  discrete-VRAM model.
* It projects a rate rather than waiting for a nearly exhausted level.
* It chooses a victim itself before the kernel does, sends a bounded graceful
  termination first, and requeues through the same receipt/lease state
  machine.
* It makes priority dominate recency.  Recency is a reasonable proxy for sunk
  work when jobs have no checkpoints.
* It bounds repeated eviction: a job eventually fails rather than retrying
  forever.

There are nevertheless four structural gaps.

First, a 30-second endpoint slope cannot distinguish productive allocation
from reclaim contention or a transient cache swing.  It answers “where will
available bytes go if this line continues?” but not “is the kernel already
spending material time trying to recover pages?”  PSI answers the latter.

Second, the same 8 GB reserve is both the admission boundary and eviction
boundary.  There is no soft/hard band: a job cannot intentionally borrow below
the reserve, because that state is immediately an eviction state.  Conversely,
one loop iteration may launch several undeclared jobs against one stale memory
sample; undeclared jobs contribute no “unrealized” commitment during the
settle window.  That is uncontrolled simultaneity, not bounded speculation.

Third, exponential time backoff forgets the cause.  If the same incumbents are
still resident when 60, 120, 240, or 480 seconds expires, the exact failed
combination can recur.  The nominal 30-minute cap is never reached before the
fifth-eviction failure.  Five evictions bound the loop, but they do not ensure
useful progress or avoid hours of repeated lost work.

Fourth, the code and current work-queue design describe `MemoryMax` as a
per-job allocation boundary.  The local test in this survey shows that this is
true for memory charged by the Linux memory controller, but false for the bulk
of a CUDA tensor allocation on this GB10 stack.  This distinction affects both
admission guarantees and victim attribution.

## 1. GPU-sharing schedulers

### MPS, time slicing, and MIG

**NVIDIA MPS — documented.**  MPS is a CUDA service that lets work from
different client processes overlap and reduces context-switch/context-storage
overhead
([MPS overview](https://docs.nvidia.com/deploy/mps/latest/index.html)).  Its
active-thread percentage is a ceiling, not a reservation, and NVIDIA documents
a pinned-device-memory limit that makes a client's CUDA allocation fail after
the configured amount.  NVIDIA also requires active clients to be terminated
through the MPS control daemon before an ordinary process signal is safe;
otherwise the server and peer clients can be left in an undefined state
([MPS provisioning and limits](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)).

**Inference for PrismaQuant.**  MPS attacks compute underfill between several
processes.  It does not make CPU and GPU memory physically separate, and its
documented memory-management examples assume the CUDA device-memory view that
is incomplete on Spark UMA.  The jobs here are mostly single-process and
minutes to hours long; enabling an MPS daemon would also require changing
`pqwork`'s TERM/KILL eviction path.  That adds lifecycle and fault-domain
complexity without replacing the node-wide memory governor.  Do not adopt MPS
for oversubscription safety.  If a future profile shows two individually
underfilled processes can improve work per joule, validate the MPS memory limit
on sm_121 and evaluate MPS as a compute-throughput A/B, independently of node
admission.  **Adoption cost: medium to high.**

**NVIDIA/Kubernetes time slicing — documented.**  NVIDIA's device plugin can
advertise several replicas of one GPU and interleave their work.  NVIDIA states
that these replicas have no memory or fault isolation and that requesting more
replicas does not buy proportional compute
([GPU time-slicing documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)).

**Inference for PrismaQuant.**  Multiple `pqwork` CUDA processes already cause
the driver to multiplex contexts.  Installing Kubernetes merely to advertise
replicas would add a second admission layer without a memory boundary.  Do not
adopt it.  **Adoption cost: high; relevant safety benefit: none.**

**MIG — documented and measured here.**  MIG partitions supported products
into isolated compute and memory instances.  NVIDIA's current supported-product
table includes GB200, B200, Blackwell RTX PRO products, and a *Thor iGPU GB10B*;
it does not list DGX Spark's NVIDIA GB10
([MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)).
On `sparky`, `nvidia-smi -q` reported `MIG Mode Current: N/A, Pending: N/A` for
`NVIDIA GB10` on 2026-08-30.

**Inference for PrismaQuant.**  Thor GB10B is not evidence that this GB10 is
MIG-capable.  MIG is unavailable on the actual target, and a survey that treats
it as an option would be misleading.  **Adoption cost: not applicable.**

### Kubernetes GPU batch schedulers

**KAI Scheduler / NVIDIA Run:ai — documented.**  KAI accepts fractional GPU or
GPU-memory requests, but its own GPU-sharing guide says it does not enforce
memory limits or isolate processes by default; an interception mechanism such
as HAMi is required
([KAI GPU sharing](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/gpu-sharing/README.md)).
KAI's scheduling cycle has allocate, consolidate, reclaim, preempt, and stale
gang-eviction phases; priority governs lower-priority preemption
([KAI scheduling deep dive](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/scheduling-deep-dive/README.md)).
Run:ai's dynamic-fractions contract is closer to this task: a workload gets a
GPU-memory/compute **Request**, may borrow up to a larger **Limit**, and an
extendable GPU OOM killer reclaims the loan when necessary
([Run:ai dynamic GPU fractions](https://run-ai-docs.nvidia.com/self-hosted/platform-management/runai-scheduler/resource-optimization/dynamic-fractions)).

**Inference for PrismaQuant.**  Run:ai's request/limit split is valuable prior
art for a soft watermark plus a non-borrowable reserve.  Its implementation is
not portable to Spark as documented: it reasons about a fraction or byte count
of one GPU's memory, while this GB10 reports no dedicated framebuffer capacity
and competes with the host in one pool.  KAI's optional CUDA-interposition
enforcement likewise needs explicit sm_121/UMA validation; its own default has
no isolation.  Borrow the burstable-loan policy, not either Kubernetes stack.
**Adoption cost for the policy: medium; for KAI/Run:ai: very high.**

**Volcano — documented.**  Volcano's gang plugin uses all-or-nothing
`minAvailable` admission so a distributed job does not strand a partial set of
pods, and its configurable cycle can combine allocate, preempt, reclaim, and
backfill
([gang plugin](https://volcano.sh/docs/scheduler/plugins/gang/),
[scheduler workflow](https://volcano.sh/docs/scheduler/overview/)).

**Inference for PrismaQuant.**  Gang admission matters when several ranks must
start together.  It does not help a mostly single-process workload decide
whether two CUDA processes fit one UMA pool.  Receipt dependencies already
cover the local all-or-nothing relationships that exist.  Do not adopt
Volcano; revisit only for real multi-rank campaigns.  **Adoption cost: high.**

**Kueue — documented.**  Kueue is a Kubernetes-native quota/admission manager,
not a replacement pod scheduler
([Kueue overview](https://kueue.sigs.k8s.io/docs/overview/)).  Its classic
preemption candidates are ordered by borrowing status, lowest priority, and
most recent admission; it then tries to remove unnecessary victims from the
selected set
([Kueue preemption](https://kueue.sigs.k8s.io/docs/concepts/preemption/)).
For gang workloads, Kueue can serialize admissions with `blockAdmission` and
evict/requeue a workload with backoff if all pods do not become ready in time
([Kueue all-or-nothing scheduling](https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/)).

**Inference for PrismaQuant.**  This is strong evidence that “lowest priority,
then newest” is not aberrant.  The missing parts in `pqwork` are a reason to
preempt, a minimal sufficient victim set, and forward-progress protection—not
a wholesale replacement of the tie-breaker.  Borrow Kueue's policy; do not
install Kueue.  Its serialized-admission precedent also supports testing one
unknown overlap at a time, although Kueue itself is solving pod readiness, not
UMA fit.  **Adoption cost for the policy: medium; for Kubernetes: high.**

### Slurm and Ray

**Slurm — documented.**  Slurm's ordinary `OverSubscribe` shares nodes,
sockets, cores, or CPUs; the configuration explicitly excludes GRES.  Slurm
warns that avoiding memory oversubscription is important and recommends
tracking memory as a consumable resource
([`OverSubscribe` configuration](https://slurm.schedmd.com/slurm.conf.html),
[consumable-resource sharing](https://slurm.schedmd.com/cons_tres_share.html)).
GPU sharing is a separate GRES configuration using MPS or generic shards; the
MPS count becomes a percentage and several jobs may share the underlying GPU
([Slurm GRES/MPS](https://slurm.schedmd.com/gres.html)).

**Inference for PrismaQuant.**  Slurm's default lesson is conservative
accounting from declared memory—the opposite of “start it and find out.”  Its
GPU-sharing path inherits MPS's limits and still lacks a Spark-specific unified
pool actuator.  The repository has already replaced its earlier Slurm design
with `pqwork`; restoring Slurm would add daemons and duplicate receipt logic
without solving the signal problem.  Do not adopt.  **Adoption cost: high.**

**Ray — documented.**  Ray memory requests are logical admission resources,
not enforced limits; by default Ray cannot estimate a task's future memory use
([memory-aware scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html)).
Ray's documented memory monitor is a whole-node level trigger.  The policy
documented for Ray 2.56 and later prefers idle workers, then retriable active
work, then the newest lease; it selects enough workers to restore a kill
buffer.  The legacy policy instead grouped work by caller, killed the newest
task in the largest group, and failed rather than retrying the last task
forever
([Ray OOM prevention, including the 2.56 policy](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html)).

**Inference for PrismaQuant.**  Ray is the closest prior art to the current
governor: restartability first, recency as a sunk-work proxy, a sufficient
victim set, and an explicit anti-livelock rule.  Its trigger is still a memory
level rather than reclaim harm, and its object-store/runtime machinery is
unnecessary.  More importantly, the local project history records a Ray
monitor killing healthy ranks on this UMA platform.  Borrow its victim-policy
ideas; do not adopt Ray as the worker runtime.  **Adoption cost for the rules:
low to medium; for Ray: high.**

## 2. Memory-pressure detection and eviction

### Kubernetes soft and hard eviction

**Documented.**  Kubelet supports a soft memory-available threshold that must
remain crossed for a configured grace period and a hard threshold that evicts
immediately with no grace.  It also supports a minimum amount to reclaim and a
pressure-transition period to prevent state flapping
([node-pressure eviction](https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/),
[kubelet configuration](https://kubernetes.io/docs/reference/config-api/kubelet-config.v1beta1/)).
For memory pressure, kubelet first considers whether use exceeds the request,
then pod priority, then use relative to the request
([pod selection order](https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/#pod-selection-for-kubelet-eviction)).

**Inference.**  The valuable idea is a two-band controller, not Kubernetes.
PrismaQuant needs a soft admission brake (PSI `some`), a hard eviction path
(PSI `full` or projected reserve crossing), hysteresis, and a reclaim target.
It cannot copy usage-versus-request ranking until per-job CUDA memory is
measurable.  **Policy adoption cost: medium.**

### PSI

**Documented.**  Linux PSI reports the fraction of time in which at least one
task is stalled (`some`) and all non-idle tasks are stalled (`full`).  Extended
memory `full` indicates thrashing and wasted CPU cycles.  The kernel explicitly
lists load shedding and killing restartable batch jobs as uses, exposes
10/60/300-second averages and cumulative stall time, and supports `poll()`
triggers over 0.5–10 second windows
([Linux PSI documentation](https://www.kernel.org/doc/html/latest/accounting/psi.html)).

**Inference.**  PSI is the most important missing signal.  It observes the
kernel's actual reclaim-induced loss even when a GPU-specific memory counter is
unusable.  But it is reactive: a fast CUDA allocation can consume capacity
before a ten-second average rises, and GPU execution can be unproductive
without every Linux task entering a memory stall.  Therefore:

1. use system-wide `/proc/pressure/memory`, not per-job `memory.pressure`, for
   the physical-pool trigger;
2. treat sustained `some` as “do not admit more”;
3. treat calibrated `full` as an eviction signal; and
4. retain the MemAvailable projection as an independent hard guard.

The threshold must be learned in shadow mode on both boxes.  The kernel's
example values and systemd's defaults are examples, not GB10 calibration.
Record `some/full` total deltas and averages beside MemAvailable, victim,
throughput, and whole-box power; select the smallest threshold that precedes
known throughput collapse without firing during healthy allocation.  This
follows PrismaQuant's requirement that performance policy be justified by
before/after in-process and Netdata evidence.  Power against the
[roughly 140 W envelope](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
is the productivity validation signal, not an eviction trigger;
neither it nor the known-bad GPU utilization readings replace a memory-safety
sensor.  **Adoption cost: medium.**

### cgroup v2 `memory.high`, `memory.max`, and `memory.pressure`

**Documented.**  The cgroup v2 memory controller describes `memory.high` as a
throttle: tasks above it enter heavy reclaim, it does not invoke OOM, and it
may be exceeded.  `memory.max` is a hard boundary that invokes the cgroup OOM
killer when charged memory cannot be reduced.  The kernel also says that
overcommitting `memory.high` values and using pressure to manage the result is
viable
([cgroup v2 memory controller](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)).
The documented charged categories are user anonymous/page-cache memory,
selected kernel structures, and socket buffers; the list does not promise
CUDA device allocations.  Cgroup-local PSI has the same format as global PSI
([PSI cgroup interface](https://www.kernel.org/doc/html/latest/accounting/psi.html#cgroup2-interface)).

**Measured here: CUDA allocations escape the job memory cgroup.**  On `sparky`
(Linux `6.17.0-1031-nvidia`, NVIDIA driver `595.84`, PyTorch
`2.11.0+cu130`), a transient user service was run with
`MemoryHigh=448M`, `MemoryMax=512M`, and `MemorySwapMax=0`.  It allocated and
touched one 2 GiB `torch.uint8` CUDA tensor, then synchronized.

| Observation | Value |
|---|---:|
| `torch.cuda.memory_allocated()` | 2,147,483,648 B |
| `torch.cuda.memory_reserved()` | 2,147,483,648 B |
| `/proc/meminfo` MemAvailable before → after | 61,247,492 → 58,833,892 kB |
| cgroup `memory.current` | 377,737,216 B |
| cgroup `memory.peak` | 378,171,392 B |
| `memory.events` `high/max/oom/oom_kill` | 0 / 0 / 0 / 0 |

A CUDA-initialized control process without the tensor used about 305 MB in
`memory.current`.  In other words, the 2 GiB CUDA tensor consumed the physical
pool—as shown by the roughly 2.30 GiB MemAvailable fall—but added only a small
host-side amount to the cgroup.  The same allocation succeeded beyond both the
448 MiB soft and 512 MiB hard settings.  The service was stopped after the
measurement; no repository or `/tmp` output was created.

The reproducible command was a transient user unit (line wrapping added):

```bash
systemd-run --user --unit=pq-priorart-cgroup-max \
  --property=MemoryHigh=448M --property=MemoryMax=512M \
  --property=MemorySwapMax=0 \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -c \
  'import time, torch
ma = lambda: next(int(x.split()[1]) for x in open("/proc/meminfo")
                  if x.startswith("MemAvailable:"))
before = ma()
t = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
t.zero_(); torch.cuda.synchronize()
print(before, ma(), torch.cuda.memory_allocated(),
      torch.cuda.memory_reserved(), flush=True)
time.sleep(12)'
```

While it slept, the observer read `memory.current`, `memory.peak`, and
`memory.events` below the unit path returned by
`systemctl --user show -p ControlGroup pq-priorart-cgroup-max.service`.  Raw
service stdout was captured in the user journal under
`pq-priorart-cgroup-max.service`; retrieve it with
`journalctl --user -u pq-priorart-cgroup-max.service`.  The table above is the
durable result record; there was deliberately no ad-hoc output file.

**Conclusion for C.**  `memory.high` does not throttle the material CUDA
allocation on this stack, and `memory.max` is not a total `--mem-gb` guarantee
for a GPU job.  It may still protect the box from a job's Python heap, page
cache, subprocesses, and other memcg-charged allocations.  Keep that limited
benefit, but do not let either setting justify more CUDA admission.  Retest
this small conformance cell after kernel/driver changes; do not infer future
behavior from today's result.  **Adoption cost of `memory.high` as requested:
reject (zero); cost to maintain the conformance test: low.**

### systemd-oomd, earlyoom, and nohang

**systemd-oomd — documented.**  systemd-oomd uses cgroup v2 and PSI, monitors
opted-in units, and sends `SIGKILL` to an eligible descendant cgroup when its
policy fires.  Its documentation highly recommends swap
([systemd-oomd service](https://man7.org/linux/man-pages/man8/systemd-oomd.service.8.html)).
The documented default memory-pressure policy is `full avg10` above 60% for
30 seconds, then selection by reclaim activity
([oomd configuration](https://github.com/systemd/systemd/blob/main/man/oomd.conf.xml)).

**Inference.**  A generic cgroup killer is slower and less informed than
`pqwork` for this workload: the default duration can exceed the current
20-second projection horizon, its attribution misses CUDA pages, it does not
know receipt/retry priority, and it uses immediate group `SIGKILL` rather than
the queue's bounded graceful path.  Running it as a second killer also creates
two competing state machines.  Do not delegate PrismaQuant job eviction to
systemd-oomd.  Use its PSI design as precedent.  **Adoption cost: medium;
benefit over an in-process PSI reader: negative.**

**earlyoom — documented.**  earlyoom acts on MemAvailable and swap levels,
sends SIGTERM then SIGKILL at lower thresholds, and allows prefer/avoid process
patterns
([earlyoom README](https://github.com/rfjakob/earlyoom/blob/master/README.md),
[threshold semantics](https://github.com/rfjakob/earlyoom/blob/master/MANPAGE.md)).

**nohang — documented.**  nohang adds PSI support, configurable victim/action
matching, and graceful escalation; its own README warns that no universal
settings prevent all unwanted kills
([nohang README](https://github.com/hakavlad/nohang)).

**Inference.**  Both are general host-responsiveness daemons.  Neither knows
leases, receipts, eviction counts, job priority, or the resident-generation
that caused a failed overlap.  `pqwork` already has better actuation and needs
only the missing PSI sensor.  Do not install either.  **Adoption cost: low to
medium, but it would duplicate policy and weaken victim correctness.**

## 3. Preemption and checkpoint/restart research

### Gang scheduling

**Documented.**  Gang scheduling admits all required workers together so
partial distributed jobs do not occupy resources while waiting for missing
ranks; Volcano's `minAvailable` contract is a concrete implementation
([Volcano gang scheduling](https://volcano.sh/docs/scheduler/plugins/gang/)).
Production-cluster analysis identifies gang and locality constraints as major
DL queuing factors
([Microsoft GPU-cluster study](https://www.usenix.org/conference/atc19/presentation/jeon)).

**Inference.**  Gang scheduling is irrelevant to ordinary single-process
PrismaQuant items.  For a future multi-rank item, represent the whole rank set
as one queue allocation and receipt group; do not let independent workers
claim ranks piecemeal.  **Adoption cost if needed later: high.**

### Tiresias

**Documented.**  Tiresias schedules distributed DL jobs with a
two-dimensional least-attained-service or Gittins-index policy when durations
are unknown, targeting completion time while avoiding starvation
([Tiresias paper](https://www.usenix.org/conference/nsdi19/presentation/gu)).
Its implementation preempts by invoking framework checkpoints rather than
preserving the full GPU/main-memory image; the paper reports hundreds of
preemptions and nontrivial aggregate switching overhead
([paper PDF](https://www.usenix.org/system/files/nsdi19-gu.pdf)).

**Inference.**  Tiresias supports two changes here: track attained useful work,
not just process start time, and make checkpointability a first-class victim
property.  Its distributed placement system is unnecessary.  **Policy cost:
medium; application checkpoint integration: high for jobs that lack it.**

### Gavel

**Documented.**  Gavel converts fairness/JCT policies into
heterogeneity-aware allocations using measured effective throughput, then
enforces them in rounds
([Gavel paper](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak)).
Its artifact uses six-minute rounds and explicit checkpoint directories, and
provides a benchmark for preemption overhead
([Gavel artifact](https://github.com/stanford-futuredata/gavel/blob/master/EXPERIMENTS.md)).

**Inference.**  The two Sparks are currently homogeneous, so Gavel's optimizer
is not useful.  Its lease lesson is useful: preemption cadence must be long
relative to measured save/reload cost, and the scheduler must know that cost.
The fixed 60-second first backoff in `pqwork` is unrelated to checkpoint cost
or useful work.  **Adoption cost for checkpoint-aware leases: medium.**

### Pollux

**Documented.**  Pollux co-optimizes resource count, batch size, learning rate,
system throughput, and statistical efficiency (“goodput”), dynamically
reallocating distributed training jobs
([Pollux paper](https://www.usenix.org/conference/osdi21/presentation/qiao)).
Its evaluation reports jobs reallocated about every seven minutes and an
average 8% runtime overhead from checkpoint-restarts
([Pollux proceedings, system overheads](https://www.usenix.org/system/files/osdi21_full_proceedings_interior.pdf)).
AdaptDL exposes application state save/load specifically for
checkpoint-restart elasticity
([AdaptDL checkpoint API](https://adaptdl.readthedocs.io/en/latest/api/adaptdl.checkpoint.html)).

**Inference.**  Pollux is designed for elastic distributed training whose
batch and learning rate can change.  Quantization probes and exports are not
that workload, and an 8% restart tax is not automatically acceptable.
Borrow only its principle that scheduling should optimize *useful progress*,
not GPU occupancy.  On GB10, PrismaQuant's corresponding metric is completed
work per joule and receipt progress, not `gpu_utilization`.  **Adoption cost of
Pollux/AdaptDL: very high.**

### What checkpointing changes locally

Receipt gating makes a complete retry correct; it does not preserve partial
work.  A bounded, deterministic preemption contract would add:

1. an item-declared checkpoint signal or command;
2. a checkpoint receipt that is atomically published by the job;
3. a maximum checkpoint grace, after which ordinary TERM/KILL proceeds;
4. resume arguments bound to that checkpoint's identity; and
5. `last_checkpoint_at`/`progress_units` for victim scoring.

This does not require a general DL scheduler.  It makes the existing state
machine honest about preemption cost and allows the victim policy to minimize
lost useful work.  Jobs without the contract remain kill-and-restart jobs.

## 4. PyTorch allocator levers on unified memory

### Allocated versus reserved

**Documented.**  PyTorch's CUDA caching allocator keeps freed blocks for fast
reuse.  `memory_allocated()` reports bytes occupied by live tensors, while
`memory_reserved()` reports all bytes managed by the caching allocator;
`empty_cache()` releases unused cached blocks but cannot free live tensor
storage
([PyTorch CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)).

**Inference.**  `reserved - allocated` is useful *inside one PyTorch process*
to distinguish live working set from cache slack.  It is not a node admission
signal: it omits host consumers, other jobs, and CUDA allocations outside the
PyTorch allocator.  Export it as job telemetry and correlate it with
MemAvailable/PSI; never subtract it from a fabricated VRAM total.

### `expandable_segments`

**Documented.**  `PYTORCH_ALLOC_CONF` controls the allocator;
`PYTORCH_CUDA_ALLOC_CONF` is now a backward-compatible alias.  Experimental
`expandable_segments=True` grows one segment per stream and targets slivers
caused by slightly changing allocation sizes such as variable batches
([allocator options](https://docs.pytorch.org/docs/main/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf),
[environment-variable alias](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)).

**Measured here.**  With installed PyTorch `2.11.0+cu130`, three touched CUDA
allocations of 64, 65, and 66 MiB under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` produced one snapshot
segment with `is_expandable=True` on sm_121.  Thus the lever is technically
available on this target; unified memory does not make the allocator itself
inapplicable.

**Inference.**  It can reduce fragmentation for changing shapes, not reduce
the bytes held by live tensors or isolate a process.  Enable it only for a
workload whose snapshots show slivers/inactive splits, then compare peak
MemAvailable, runtime, and work per joule before and after.  Do not make it an
oversubscription safety requirement.  **Adoption cost: low to test, medium to
promote across production jobs.**

### `garbage_collection_threshold`

**Documented.**  With the native allocator,
`garbage_collection_threshold` starts reclaiming old unused CUDA blocks after
allocator use crosses a fraction of GPU capacity, avoiding a later global
sync-and-release; it is ignored by the `cudaMallocAsync` backend
([PyTorch allocator options](https://docs.pytorch.org/docs/main/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)).

**Inference.**  It is relevant to cache hoarding, but the denominator is the
CUDA device-capacity view, not current whole-system MemAvailable.  NVIDIA
already warns that CUDA's free-memory view is incomplete on Spark UMA
([UMA memory reporting](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)).
It can therefore release unused cache earlier, but cannot serve as a node OOM
trigger and cannot reclaim live tensors.  Trial it only when
`reserved - allocated` is materially large during co-residency.  **Adoption
cost: low to configure, medium to profile safely.**

### PyTorch allocator caps

**Documented.**  `torch.cuda.set_per_process_memory_fraction()` limits the
caching allocator to a fraction of total visible memory and raises an
allocator OOM above it; PyTorch cautions that free memory is generally less
than total capacity
([PyTorch API](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.set_per_process_memory_fraction.html)).

**Measured here.**  A 0.004 fraction (about 498 MiB allowed on this device)
made a requested 768 MiB PyTorch tensor fail immediately with a catchable
`torch.OutOfMemoryError`.  This is stronger for PyTorch-managed CUDA blocks
than the cgroup limit measured above.

**Inference.**  This is a useful *eligibility-specific* guard for known
PyTorch jobs, but not a universal node boundary: custom CUDA libraries and
host allocations can sit outside that allocator, while the fraction is based
on total device capacity rather than dynamic UMA availability.  It could be
one layer of bounded speculation only if a job explicitly opts into the shim
and allocator-OOM is a terminal “does not fit this cap” transition, not a blind
retry.  **Adoption cost: medium.**

## Detailed answers to A–D

### A. Combine PSI and MemAvailable trajectory

**Recommendation:** keep the current projected-reserve test and add a PSI
state machine.  PSI is plainly the better *pressure* trigger because it
measures time tasks actually lose to memory contention, whereas a byte slope
can fall harmlessly during planned allocation.  PSI is not the better sole
*OOM* trigger because it can lag a fast allocation and does not promise to see
GPU-side unproductivity.  On a platform with no trustworthy dedicated memory
telemetry, these two independent global signals are stronger together than
either alone.

The initial design should be one policy, not a menu:

* **Normal:** admit according to the speculation policy while `some` is below
  its calibrated threshold and the hard projection is safe.
* **Pressure:** after sustained `some`, freeze new admissions but do not kill;
  this is the analogue of a Kubernetes soft threshold.
* **Emergency:** evict when calibrated `full` persists for the short local
  grace **or** the MemAvailable projection crosses the hard reserve.  The
  projection path has no extra grace.
* **Recovery:** require both PSI and the projected headroom to recover for a
  hysteresis interval before reopening admission.

Do not copy systemd's documented
[60%-for-30-seconds default](https://github.com/systemd/systemd/blob/main/man/oomd.conf.xml):
it is designed for generic cgroups and may be later than the present 20-second
horizon.  Shadow-log first, including Netdata memory/power on both boxes and
an in-process profile for the workload, then fix versioned thresholds from
measured healthy and failing runs.  **Adoption cost: medium.**

### B. Newest-lowest-priority is defensible but incomplete

**Answer:** no, it is not intrinsically wrong.  Kueue documents candidate
ordering by lowest priority then most recent admission
([Kueue preemption](https://kueue.sigs.k8s.io/docs/concepts/preemption/)), and
Ray's policy documented for 2.56 and later prefers retriable active work, then
the newest lease, and chooses enough workers to restore a buffer
([Ray OOM prevention](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html)).
The current order minimizes sunk runtime in a checkpoint-free system.  Real
schedulers add facts PrismaQuant currently lacks: whether a task is retriable,
whether a tenant/job can still make progress, whether use exceeds its request,
how many victims are sufficient, and where the last checkpoint lies.

**Recommendation:** use one lexicographic progress-aware policy:

1. only explicitly evictable/restartable jobs;
2. preserve one incumbent progress anchor per node;
3. lowest declared priority;
4. lowest uncheckpointed-work cost per expected GiB released; and
5. newest start as the fallback tie-breaker.

Choose a minimal sufficient victim set rather than always exactly one victim.
Until checkpoint age and freed bytes are available, retain the current order;
do not replace real fields with guessed precision.  The anchor is protected
from *speculative* churn, not from the hard safety boundary: if it alone cannot
fit, terminate it and fail it closed as “does not fit” rather than risking a
kernel OOM.  **Adoption cost: medium to high.**

### C. Do not use `memory.high` to replace kill-and-requeue

**Answer:** no.  The kernel documents `memory.high` as a reclaim throttle
([cgroup v2 memory controller](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)),
but it does not dominate here because the material CUDA allocation was not
charged to the job cgroup in the local GB10 test.  `memory.high` would throttle
Python heap or page-cache growth while the CUDA tensor could continue consuming
the same physical pool.  `memory.max` has the same accounting blind spot and
must not be described as enforcement of total GPU-job `mem_gb` on this stack.

**Recommendation:** keep `MemoryMax`/`MemorySwapMax` only as host-side blast
containment, do not add `MemoryHigh` as the oversubscription actuator, and keep
proactive queue-directed termination.  Rerun the conformance test after each
kernel/driver change; change this conclusion only if the CUDA bytes appear in
`memory.current` and both high/max events behave under pressure.  **Adoption
cost: zero; verification maintenance: low.**

### D. Bounded speculation with learned no-good states

**Recommendation:** replace purely time-based speculative retry with a
node-local speculation epoch:

1. Designate one running job as the protected progress anchor (normally the
   highest-priority/oldest incumbent).  Its protection applies while another
   job is being tested, never when the anchor itself violates the hard guard.
2. Split the present threshold into a soft admission watermark and a lower,
   non-borrowable hard reserve.  Issue exactly one speculation token.  It may
   admit one additional job into measured, bounded negative **soft** headroom
   when PSI is quiet and the hard-reserve trajectory remains safe.  Never
   start a herd from one stale sample.
3. Snapshot the resident generation (job IDs plus start generations) when the
   speculative job starts.
4. If pressure evicts it or another victim, persist a no-good record binding
   the failed job to that resident generation and the observed headroom.
5. Do not retry that combination when a wall-clock timer expires.  It becomes
   eligible only after an incumbent completes/is evicted, the host reservation
   changes, or a materially larger measured capacity is available.
6. On success, record the observed peak envelope and use it as evidence for
   later placement, never as an unbounded hard promise.

This permits admission below today's `headroom > 0` boundary without
recreating the same losing state.  The anchor is an inference from the same
forward-progress problem addressed by Ray's legacy “do not retry the caller's
last task forever” policy; the no-good record converts an eviction into
scheduler knowledge; the one-token limit makes the unknown appetite observable
before another unknown starts
([Ray OOM prevention](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html)).
Exponential backoff can remain for transient errors, but it is no longer the
memory-fit mechanism.

The extra aggression is borrowing the **soft** band, not the physical safety
reserve.  No scheduler can safely lend the hard reserve to an unbounded
allocator; that requires an enforceable application/allocator cap, and the
GB10 measurement shows that `memory.high` is not such a cap on this stack.

For jobs that explicitly support it, a PyTorch allocator fraction and a
checkpoint receipt can reduce the damage further.  Neither is required for
the state-machine safety invariant, and neither should silently broaden the
contract for non-PyTorch jobs.  **Adoption cost: medium for admission state;
high when retrofitting application checkpoints.**

## Acceptance evidence for any implementation

This is a research recommendation, not a request to change `pqwork.py` in this
PR.  A later implementation should not be promoted from shadow mode until it
has all of the following:

* A deterministic trace test showing that the same failed resident generation
  cannot be readmitted, while a changed generation can.
* A test showing one progress anchor survives repeated pressure and the queue
  either completes something or fails closed.
* PSI parser and hysteresis tests, including missing/unsupported PSI behavior.
* A GB10 conformance test proving whether CUDA bytes are charged to cgroup
  `memory.current`; current expected result is **not charged**.
* Before/after in-process profiles and Netdata series from both boxes.  Report
  throughput and work per joule against the documented
  [approximately 140 W SoC envelope](https://docs.nvidia.com/dgx/dgx-spark/hardware.html);
  do not use GPU utilization as the causal metric.
* A replay of known fast-allocation and slow-reclaim incidents showing the
  combined trigger fires early enough, does not kill during a healthy cache
  fill, and recovers with hysteresis.
* For allocator changes, `memory_allocated`, `memory_reserved`, allocator
  snapshots, MemAvailable, PSI, runtime, and power from matched runs.

## Bottom line

The best available design is intentionally small.  Keep the queue's existing
receipt/lease machinery and MemAvailable projection.  Add the kernel's global
stall signal, add persistent knowledge about failed overlaps, and make useful
progress/checkpoint age part of preemption.  Do not import a multi-tenant
cluster stack, do not list MIG as a GB10 option, and—most importantly—do not
expect `memory.high` or the existing `MemoryMax` setting to police CUDA tensors
until a GB10 conformance test proves that the driver charges those pages to the
job cgroup.
