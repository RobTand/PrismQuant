# PrismaBuild — distributed campaign execution

**Status: DETERMINISTIC CORE + DURABLE SLURM ADOPTION + OPTIONAL DAGSTER LAYER
BUILT / NOT LIVE-DEPLOYED.** The
dependency-free action-key, immutable-CAS, and local-worker core lives in
`prismaquant/prismabuild.py`; the fail-closed SLURM resource transport lives in
`prismaquant/prismabuild_slurm.py`; and the optional asset/DAG adapter lives in
`prismaquant/prismabuild_dagster.py`. Before `sbatch`, the SLURM adapter
first-writer-publishes a sealed submission identity, bounded retry policy, and
self-hashed runtime identity for the loaded adapter module plus configured
worker-launcher bytes;
after acceptance it binds the returned job id, and after an orchestrator
restart it adopts only the unique scheduler allocation carrying that exact
identity. Poll and scheduler-mutation claims are append-only durable state. The
adapter submits a canonical immutable action request to one sealed cluster
with an exact `sbatch` argv and a sealed, POSIX-quoted `--wrap=exec` worker argv,
`--export=NIL`, and explicit resources, then accepts only a scope-correct CAS
receipt as success. A SLURM `COMPLETED` state without that receipt is a failed
action.
`tools/prismabuild_worker.py` is the direct batch-script entry point. The
Dagster adapter constructs deterministic assets from sealed action keys, binds
each edge to an expected CAS output digest, and materializes only after
re-reading that receipt and payload from the CAS. The SLURM daemons, Dagster
service, and observability stack in the chosen design below are not deployed,
and nothing in the live quantization pipeline depends on PrismaBuild yet.

Local task output is now crash-recoverable without accepting unowned bytes.
Before argv, the worker publishes an immutable claim for the exact action,
resolved checkout, working directory, and declared result. Under the same
output lock, a retry may discard and recompute only a regular contained result
with that exact claim; it never adopts the old bytes under a new producer
attestation. `repair-local-result` performs only that checked cleanup. One
subprocess SIGKILL fault test covers death after result/blob and receipt-temp
staging but before canonical receipt publication. This is process-fault
coverage, not power-loss or deployed cross-host-lock evidence.

`run-local` also has one qualification-only, opt-in causal hook:
`--initial-miss-rendezvous /absolute/manifest.json`. It proves that two exact
worker processes both observed an initial miss against the same configured CAS
root before either can
reach the output lock and publish. It is inert when unset, is not entered on an
initial cache hit, and is incompatible with `--recompute`. This proves worker/
miss contention; it makes no task-argv timing claim. The output lock serializes
only workers sharing the same live checkout/output-lock identity. Workers in
distinct validated checkouts may execute task argv concurrently and converge
through ordinary CAS publication.

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
- **Generation vs measurement tasks**: ordinary generation (encodes,
  permutation/gauge searches — discrete outputs re-scored later) may exclude
  host from the key → any box's result is valid ("surrogates generate, real KL
  selects" applied to hardware). Measurement (KL, PPL, probe)
  INCLUDES host-class + toolchain — numerics don't transfer across
  architectures; gold path pinned to `gb10`. Codebook generation is also
  nonportable because D29 records cross-architecture row-scale byte drift.
- **Artifact family is explicit** — action schema
  `prismaquant.prismabuild.action.v2` requires the closed
  `task.artifact_family` value `generic` or `codebook`. `artifact_kind` remains
  a descriptive identifier and never drives portability by substring. V1 is
  not reinterpreted: callers must redeclare the family and reseal the action.
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
`prismaquant.prismabuild.worker_attestation.v2` record bound to the action key:

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
- The worker implementation is a separate closed
  `prismaquant.prismabuild.worker_runtime.v1` object. It binds the exact
  `prismaquant/prismabuild.py` source snapshot taken once while that module
  initializes. Canonical JSON and SHA-256 are implemented in that same file,
  so the receipt-digest implementation does not escape into an unrecorded
  repository import. The live core file must still match the load-time
  snapshot at preflight, after task execution, and at publication. For the
  SLURM path, `tools/prismabuild_worker.py` snapshots its own source at the
  earliest executed wrapper code, before importing the core, and passes that
  identity into preflight. The launcher is checked there and at the same two
  later boundaries. Direct Python API calls record the explicit `in_process`
  mode and a null launcher rather than inventing a script identity.
- Every nonportable `action.inputs` digest must already exist and verify in the
  PrismaBuild CAS before argv starts. Portable actions preserve the existing
  external-input contract: CAS-resident inputs and recognized toolchain fields
  are verified when possible, while unresolved inputs and descriptive
  toolchain fields remain permitted and are visibly absent from the
  attestation's verified subsets.

The supported preparation boundary is `PrismaBuildCAS.ingest_input()` or the
dependency-free `ingest-input` CLI. It takes a stable regular-file snapshot,
derives the canonical SHA-256 and byte count, optionally checks both against
caller-supplied expectations, publishes through a read-only first-writer-wins
hard link, and fsyncs the blob shard. A winning publisher reopens the canonical
name and proves that it is the exact private, read-only staging inode whose
bytes it just hashed and fsynced; it does not hash that same inode again. A
loser never trusts the other writer's inode and hashes the canonical blob in
full. `input_path()`, `verify-input`, and every public cache lookup retain the
schema, size, mode, and full-content check. A conflicting, malformed,
symlinked, truncated, writable, or changed object refuses.
This closes the code-level input-ingress gap. One narrow cross-host pilot was
run on 2026-08-30 from repository commit `5bd2d2c`: Sparky and Sparklina used
their direct stdlib launchers concurrently to ingest the same 2,601-byte
`pyproject.toml` into the fresh NFS4 CAS
`/mnt/shared/prismaquant-prismabuild-validation/5bd2d2c/input-cas-race4-direct`
on the same export with `local_lock=none`. The source SHA-256 was
`2a872eb7dfbe734920ec90e997a91460a33b725a8ab19372340e68d11f39a495`;
Sparky returned `published` in about 3.2 seconds, Sparklina returned
`already_present` in about 3.3 seconds, and both exited zero. That historical
pilot validated only the small-file concurrent input hard-link/readback case;
the larger evidence and its remaining limits are recorded below.

#### Scheduler-free live-NFS qualification (2026-08-31)

A larger wall-clock-overlap run at exact commit `568eeb4` found a real CAS
false refusal before qualification. In simultaneous identical 128 MiB input
and 8 MiB deterministic-result publication, one losing reader raised
`CASTamperError` even though the canonical bytes were correct. The retained
trace held device, inode, size, mode, uid, gid, mtime, and `nlink == 2`
constant while NFS reported ctime moving backward from
`1788148069334890999` to `1788148069099740663`. The before-run, timed reports,
and trace are retained under the `timed-overlap` and `ctime-trace` directories
of `/mnt/shared/prismaquant-prismabuild-validation/568eeb4/run-20260831T034200Z-codex-live-nfs-v2`.

Commit `7acf3ad` closes that defect without a ctime exception. A CAS read has at
most `_STABLE_FILE_READ_ATTEMPTS = 3` attempts. Each attempt resolves the path
again through held no-follow directory descriptors, opens a fresh leaf FD, and
replays the entire read or SHA-256 from byte zero. PrismaBuild accepts only an
attempt whose complete identity (device, inode, size, mode, uid, gid, mtime,
nlink, and ctime) is identical before and after the read and whose expected
byte count and content address match. A ctime/nlink-only within-read mismatch
discards that attempt and retries; a substantive identity change, wrong
content address or byte count, non-regular/writable object, changed path, or
symlink hop refuses immediately. Three unstable reads refuse. Deterministic
regressions cover transient `2 -> 1` publication-link cleanup, transient
`1 -> 1` link/unlink ctime churn followed by a stable pass, perpetual `1 -> 1`
churn, and immediate content/mode/mtime/owner refusal. The focused core,
Slurm-adapter, and Dagster-adapter suite passed `174 passed, 1 skipped`.
The Slurm durable-state adapter now calls that same core primitive with
read-only enforcement and a 16 MiB bound instead of maintaining a divergent
single-pass copy. Adapter regressions separately pin transient ctime/nlink
replay from a fresh FD and immediate substantive mode/mtime refusal.

The exact-fix rerun is retained without overwrite at
`/mnt/shared/prismaquant-prismabuild-validation/7acf3ad/run-20260831T035517Z-codex-live-nfs-fix`.
Both clean checkouts reported exact
commit `7acf3adec56a44cb909297938fd6a860e0c1a78b` and identical core SHA-256
`733b1515af957e08bcb9ff2f51dba5c4e338e3cbda7330d5936e238f55acbe69`.
Sparky and Sparklina mounted the same NFSv4.2 export with `local_lock=none` and
reported NTP synchronized. Sequential clock samples placed the remote sample
0.30--0.56 seconds after the local one (including SSH latency), so races used
absolute wall-clock starts rather than NFS marker visibility.

The retained verifier (`facts/verification.json`, SHA-256
`f1e3d40eb6072244aa8817f3bdcaf31611710b1424c3d89fdf448eb9fb0324d7`)
establishes the following scheduler-free cases:

- Both 128 MiB identical-input operations overlapped, returned successfully,
  and observed exactly one winner at content address
  `254bcc3fc4f27172636df4bf32de9f107f620d559b20d760197e452b97453917`.
  Both 8 MiB identical-result operations also overlapped and returned one
  winner plus one verified hit with receipt
  `60051d82481570da096e91c643a19dd170a0c3548e4ef6385599fed360d4a302`.
- Conflicting deterministic writers overlapped and produced one result plus
  one `CASConflictError`; independent reads on both hosts agreed on the
  canonical receipt, payload SHA-256, inode, and read-only mode.
- Both continuous readers saw misses (`592` and `559`) before publication,
  then each completed 100 fully verified hits with no hit-to-miss or identity
  regression. Both parent rename-to-symlink traps raised `CASTamperError`, and
  the outside directory remained empty.
- With retained fake executables only, concurrent Slurm adapters produced one
  `submitted` and one `adopted` result for the same job, unique poll ordinals 1
  and 2, and one successful cancel claim plus one fail-closed concurrent
  refusal. The command log contains exactly one `sbatch`, one `sacct`, two
  `squeue`, and one `scancel`; no real scheduler was contacted. Both durable
  readbacks were identical except for the client-local NFS device number.

The final 173-file manifests read independently on both hosts are byte-identical
at SHA-256
`ba8f70879b8f198ed989335172399a51cfbc4c0189be4444d9b1df2425a29f06`.
This qualifies the exercised scheduler-free CAS and durable-state races on the
current NFS mount. It does not qualify host/power loss at a durability
boundary, ACL/WORM retention, a production-scale result, real Slurm services
or allocation identity, a Dagster daemon, GPU execution, or deployment.

The dependency-free launcher for a bare host is direct script invocation, for
example `/usr/bin/python3 /path/to/prismaquant/prismabuild.py ingest-input ...`.
That form ran on both pilot hosts. `python -m prismaquant.prismabuild` first
executes `prismaquant/__init__.py` and therefore requires the installed
PrismaQuant environment; on bare Sparklina system Python it failed on the
package's `compressed_tensors` dependency before reaching the stdlib-only
core. The module form works in the `pq-cu130` environment, but must not be
advertised as the dependency-free launcher.

Two limits are explicit. For portable actions the observed executable hash is
receipt provenance, not a newly required action-key field; callers that need
the executable to participate in cache identity must use a nonportable scope
and the `argv0.*` toolchain fields. Input preflight proves that the declared
CAS bytes exist and match before execution, but the sealed argv/code remains
responsible for resolving and consuming those bytes; process provenance is not
an OS-level proof of every file read. Worker core/launcher identity is likewise
receipt provenance, not action-key identity: a cache hit retains the producer
revision that created its canonical result. Separately, Slurm submission intent
v2 seals a self-hashed runtime object containing the exact loaded adapter-module
bytes and configured worker-launcher bytes. Both are rehashed after the durable
intent readback and immediately before `sbatch`; path or byte drift refuses
without invoking the scheduler. This does not put transport code into the
reusable action key or action request. The scheduler cannot retain the submit-
host FD while a job is queued: the started wrapper therefore still records and
rechecks its own earliest live snapshot, and that receipt provenance may
visibly differ from the submission-time planned launcher if deployment changed
after `sbatch`. Python does not expose the already-started script's parser input
buffer, so even that early snapshot is not a cryptographic proof of the exact
bytes the interpreter parsed.

The attestation becomes `producer` in the self-hashed
`prismaquant.prismabuild.cas_receipt.v3` receipt. CAS lookup replays its action,
scope, platform derivation, host-class evidence, worker core/launcher identity,
task-executable identity, verified toolchain, verified-input subset, and self-
digest before accepting the result. V3 receipts use
`actions/v3/<prefix>/<action-key>.json`. Legacy v2 receipts retain their
immutable unversioned `actions/<prefix>/<action-key>.json` addresses: v3 lookup
does not parse them as v3, overwrite them, delete them, or silently migrate
them. A v2-only key is a v3 cache miss and must be recomputed under the new
producer contract.

Receipt publication fsyncs the candidate, runs the potentially longer
action-closure and executable callback first, then rehashes the core and
launcher as the final userspace check before the first-writer-wins hard link.
There remains an unavoidable sequential interval between that final check
returning and the `os.link` syscall; this design minimizes that interval rather
than claiming a zero-gap filesystem snapshot.
Result-blob publication uses the same consumed-inode rule as input ingestion.
The successful publisher already computed SHA-256 while copying into a
private read-only staging inode. After the canonical hard link is durable, it
reopens that name and requires exact device/inode and substantive metadata
identity. Receipt readback can then validate the canonical receipt without a
second or third payload hash. If another blob or stochastic receipt won, every
unconsumed winning blob is hashed normally. Returning a path immediately from
that successful publication reuses this proof; later `lookup()` and
`result_path()` calls always consume and hash the canonical payload anew.
The before/after 2 GiB and 256 MiB NFS measurements and their limits are in
`docs/results/prismabuild_publish_io_2026-08-31.md`.
CAS staging, blob, request, and receipt directories are walked or created only
through held `dir_fd` values with `O_NOFOLLOW`; new components use `mkdirat`
semantics and are fsynced after their final mode is applied. Hard links,
readback, hashing, and cleanup are relative to those held descriptors. Before
accepting a read or completed publication, PrismaBuild reopens the configured
parent path and canonical leaf and verifies their device/inode identities.
Thus an ancestor rename-to-symlink race fails closed and never redirects a CAS
read, write, or unlink outside the configured root. The Slurm worker likewise
accepts only the canonical `requests/<prefix>/<action-key>.json` address and
reads it through this anchored path after its restart guard. These guarantees
depend on Linux `openat`/`O_NOFOLLOW` and `/proc/self/fd`; a returned payload
`Path` is only evidence of the just-verified name, not a file descriptor held
open for an arbitrary later consumer.

The live-checkout output path has a separate recovery contract. A canonical
`prismaquant.prismabuild.local_result_claim.v1` record below
`local-results/v1/` binds the action key, full action-manifest digest, resolved
checkout, normalized working directory, and normalized result path. It is
durable before argv starts. A matching retry first holds the existing
checkout/output lock, validates the claim byte-for-byte, rejects symlinks and
non-regular paths, unlinks only the claimed leaf, removes at most 64 permitted
same-UID files from the claim-private result-staging directory, and reruns argv
to produce a new attestation. An unclaimed dirty result remains a hard error.
A valid CAS receipt also blocks explicit repair; repair repeats that lookup
after acquiring the output lock so a receipt published while it waited wins.
Claims are retained as immutable recovery authority; they are not success
records and cannot satisfy `lookup()`.

The checkout/output `flock` is also an action-lifetime lease. The worker passes
the exact locked open-file description into task argv, so abrupt worker death
does not release exclusion while that direct task process can still write its
declared result. A retry waits; after the orphan exits it removes only the
exact claimed result and recomputes. If a handled worker exception such as
`SIGINT` unwinds Python, the worker terminates and reaps the task's complete
new-session process group before its context manager closes the worker's lock
descriptor. It never explicitly unlocks the shared open-file description: if
the task cannot be reaped, its inherited descriptor retains exclusion.
Regressions kill the worker both during argv and after result staging. This is
not a kernel-enforced sandbox: task code that deliberately closes inherited
descriptors or escapes its process group violates the local-action contract,
and an indefinitely uninterruptible task can retain the lock indefinitely.
The proposed Slurm deployment's cgroup and sealed time limit remain required
external containment; PrismaBuild has no durable local holder lease or
lock-acquisition timeout today.

#### Opt-in two-phase initial-miss rendezvous

The qualification hook is called immediately after the ordinary first
`PrismaBuildCAS.lookup(action)` returns `None` and before checkout/output path
resolution or `_local_output_lock`. Its immutable, canonical, self-hashed
`prismaquant.prismabuild.initial_miss_rendezvous_manifest.v1` file binds one
normalized non-root absolute rendezvous namespace, the exact normalized
non-root absolute CAS root used by both workers, a 128-bit lowercase-hex run
nonce, the exact action key, exactly two sorted unique lowercase hostnames, and
a positive finite local-monotonic timeout no greater than one day. The manifest
must be a read-only regular inode with exactly one link and is read through the
existing bounded no-follow stable-file primitive. The namespace has exactly
the `arrivals/` and `ready/` directories; each phase admits only the two exact
`<hostname>.json` leaves (plus a bounded transient private publication name).

For participant `i`, let `M_i` be its initial verified miss. It derives its
hostname from `socket.gethostname().lower()`, and a self-hashed process identity
from hostname, PID, Linux `/proc/<pid>/stat` start tick, a fresh invocation
nonce, and the existing exact loaded-core plus optional launcher runtime
identity. It no-clobber publishes an immutable
`initial_miss_rendezvous_arrival.v1` record only after `M_i`. Both workers wait
for and validate the exact complete arrival set while refusing a CAS receipt.
Each then makes a fresh CAS-absence observation and publishes an immutable
`initial_miss_rendezvous_ready.v1` record that binds its process/arrival and the
canonical digest of that complete arrival set. The same absence/runtime check
is the last userspace callback before the ready hard link. Only the exact
complete ready set releases either worker to the output lock. Thus, for both
participants, `M_i < arrival_i < ready_i < release < any in-protocol result
publication`; no cross-host wall clock or realtime timestamp participates in
the proof. A worker stopped after release may resume later and return the
ordinary post-lock cache hit, carrying the same proof.

Arrival, ready, manifest, process, and returned
`initial_miss_rendezvous_receipt.v1` objects have closed schemas and canonical
self-digests. Wrong action/run/manifest/host/runtime bindings, corrupt digests,
duplicate/replayed participants, missing or extra entries, writable files,
symlinks, special files, persistent hard links, source drift, and a CAS receipt
visible during either worker's pre-release checks fail closed. The protocol
does not claim atomic exclusion against an out-of-protocol receipt linked in
the interval between the last ready pre-link callback and the ready link;
workers released by that link may legitimately return proof-bearing hits.
Polling and directory cardinality are bounded and use
`time.monotonic()`; an unavailable peer times out without task argv. The second
ready link is the logical release event. If the CAS receipt becomes visible
while NFS still returns an incomplete ready-directory view, the worker keeps
performing bounded exact scans through the original monotonic deadline; this
does not misclassify a legitimate peer publication after release.

This is an integrity protocol inside PrismaBuild's existing cooperative
filesystem trust boundary, not authentication or a Byzantine quorum. SHA-256
self-digests detect corruption and cross-contract mismatch but are unkeyed. A
principal able to write arbitrary correctly formed files as another hostname,
or to remove namespace entries, can fabricate or suppress the evidence.
Qualification therefore requires the same isolated worker principal and
ACL/WORM/retention controls already required for CAS and Slurm state. The hook
has CPU-only hostile tests but no shared-NFS, host/power-loss, live Slurm, or
deployment qualification yet.

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

### Durable SLURM submission, polling, and cancellation (implemented, not live-validated)

Scheduler identity is shared CAS state, separate from result truth. For each
action the adapter owns one immutable lineage:

```text
submissions/v2/<action-prefix>/<action-key>/
  intent.json
  job.json
  transitions/polls/00000000-<ordinal>.json
  mutations/<ordinal>.json
```

`intent.json` uses `prismaquant.prismabuild.slurm_submission_intent.v2` and
contains a `prismaquant.prismabuild.slurm_submit_spec.v2`. It
seals the action key; one cluster; CAS, log, checkout, worker, and SLURM
executable paths; the closed submit environment; resources and placement; and
the exact `max_polls`, `poll_interval_seconds`, and zero-`max_requeues` policy.
It also seals the complete canonical worker argv.
The submit spec also carries a self-hashed
`prismaquant.prismabuild.slurm_runtime.v1` record for the load-time adapter
source and configured worker-launcher source. The runtime launcher's declared
path must exactly equal the sealed worker path, and both source identities and
the runtime digest are strictly validated. Existing `submissions/v1` records
remain immutable history: the v2 adapter neither parses nor migrates them.
SLURM recompute is also sealed false. Its canonical submit-spec digest
derives both the full `pqb-<digest>` job name and
`prismabuild:<digest>` comment. The read-only first-writer object is published
and re-read before `sbatch`; a changed resource, path, environment, or retry
limit on replay conflicts rather than creating another lineage.

The worker is deliberately not passed as `sbatch`'s positional batch script:
Slurm copies such a script into its spool, so the executed launcher's
`__file__` becomes a spool path and its checkout-relative core import fails.
Instead the adapter supplies one `--wrap=exec <command>` option. The command is
derived only from the sealed worker argv with `shlex.join`; tests round-trip
spaces, quotes, dollar signs, and semicolons through `shlex.split` and assert
there is no positional script. This is a controlled POSIX-shell encoding at
the Slurm boundary, while the submitting process still uses `shell=False` and
the task's own argv remains inside the immutable JSON request rather than shell
text. A real Slurm launch of this path remains part of deployment validation.

Submission uses `--no-requeue`, and positive same-job retry is deliberately
unavailable. This is not a temporary command-line combination: current Slurm
uses the job's single Requeue eligibility flag for both explicit
`scontrol requeue` and automatic/site/admin restart. `--no-requeue` therefore
also makes explicit requeue ineligible; changing it to `--requeue` would admit
restarts that have no durable PrismaBuild authorization. The adapter and
Dagster `ActionSpec` reject `max_requeues > 0`, and `SlurmAdapter.requeue()`
never sends `scontrol`. The batch argv adds `--require-slurm-initial-start`;
before `run-local` can reach task argv, the worker requires a real numeric
`SLURM_JOB_ID` and absent-or-zero `SLURM_RESTART_COUNT`. A malformed or nonzero
count refuses even if a site administrator overrides the submission policy.
Positive retry can return only with a new protocol that binds Slurm's actual
restart counter/`Restarts` state to an authorized durable mutation claim.
The v2 submit spec still seals the configured absolute `scontrol` path even
though this zero-requeue protocol never executes it. That field is inert and
over-broad provenance, not a hidden retry path. Removing it would change the
canonical submit-spec digest and therefore requires an honest later schema/
namespace boundary; D9 deliberately does not reinterpret v2.

It also uses `--export=NIL`, not `NONE`: current Slurm defines `NONE` to invoke
the implicit `--get-user-env` path, whereas `NIL` passes only scheduler/SPANK
variables to the already-required absolute worker path.

`sbatch --clusters=<sealed-cluster>` restricts submission to one cluster rather
than creating federation siblings. After `sbatch --parsable` returns,
`job.json` binds that intent to its exact cluster-qualified job id; clusterless
output is normalized only because the submission already selected exactly that
one cluster, and a different returned cluster refuses. A restart first reuses a
valid binding. If the process died after scheduler acceptance but before
binding, recovery queries `sacct --clusters=<sealed-cluster>` from the Unix
epoch by the sealed name, requests widths large enough not to truncate the
name/comment, and binds only one allocation row whose name, comment, and
cluster all match. Both adoption and bound accounting queries request
`--duplicates`: Slurm documents duplicate records after requeue, federation,
resize, or JobID rollover, so hiding all but the newest row could hide an
ambiguous allocation lineage. Zero rows are ambiguous between a pre-`sbatch`
death, accounting lag, or retention loss; multiple rows, malformed rows,
identity drift, and unknown states also refuse without another `sbatch`.
Bound state queries are cluster-qualified; the helper's only clusterless query
form forces `--local`, preventing federation display defaults from silently
widening it.

The poll budget and cadence survive process loss. Each canonical self-digested
poll record includes its append wall-clock nanoseconds. Ordinals must be a
contiguous prefix, the count may not exceed sealed `max_polls`, and a restarted
adapter waits out the remaining sealed interval before winning the next
first-writer claim. The interval is positive, finite, and capped at one day;
the poll and filename counters are bounded to their eight-digit durable
representation. Clock rollback refuses. A crash after a poll claim
conservatively consumes it. This wall-clock protocol still requires deployed
hosts to have bounded synchronized time; that is a live gate below.

Poll replay is accelerated only by a process-local, non-authoritative snapshot.
The first access in an adapter process (and therefore every adapter restart)
replays and validates the complete append-only retry journal. An uncontended
subsequent claim revalidates the exact canonical durable tail, preserves its
timestamp for pacing, and attempts the next first-writer publication in O(1)
journal work. A read-only progress query also probes the one expected successor;
if another writer advanced it, the adapter invalidates the snapshot and derives
a new one by full replay. A lost publication race follows the same invalidation
and full-replay path before it refuses the loser. Striped process-local `RLock`s
serialize threads sharing one adapter for the same action without globally
serializing distinct actions, but are not—and are not presented as—cross-host
filesystem locks. Cross-host serialization remains the no-clobber append
itself. No durable record is compacted, rewritten, or deleted.

That optimization is deliberately unavailable as terminal authority. Poll
budget exhaustion, scheduler terminal resolution, cancellation, and success
of an action with a durable Slurm intent all discard the cached view and audit
the complete prefix before returning. Thus deletion/corruption hidden behind a
still-valid tail can allow another nonterminal observation, but cannot license
success, failure, cancellation, or budget exhaustion. Latest-step deletion or
replacement refuses immediately in the hot loop. Complete replay remains the
crash/restart recovery mechanism; the cache contains no state that must survive
a process loss.

The Slurm module contains no bare Python `assert` invariants. Durable-schema,
runtime-identity, retry-policy, attempt-bound, cache-reconstruction, placement,
worker-argv, and anchored-path assumptions all raise explicit contract,
tamper, protocol, or local-action exceptions. An AST regression pins that
property, so `python -O` cannot erase a refusal check.

`SlurmAdapter.resolve()` itself remains an unbudgeted scheduler-observation API:
it does **not** consume a poll claim. The native `DagsterActionRunner` enforces
the intended pairing by calling `claim_poll()` immediately before each
`resolve()`. A caller that invokes `resolve()` directly can issue scheduler RPCs
without consuming the sealed `max_polls`; PrismaBuild does not yet implement a
claim token or exact claim-to-observation consumption contract for that API.
Accordingly the current bound applies to the native orchestrated loop, not to
arbitrary direct `resolve()` calls. This is an explicit remaining contract gap,
not a throughput claim.

The exact merged-source CPU profile at 4,000 retained poll records is immutable
under `/home/rob/dq-runs/prismabuild-d9-poll-cache-merged-20260831` (manifest
SHA-256
`189e7ddc8d9c56ff546954c5ab7a09312c5b492137ce5a10c1706cdeee416037`,
comparison SHA-256
`265db8e2e3871b5d274ba009ed39c447fdd31c93d1ce1b27b54770e47766a65d`).
Against exact pre-change commit `a71680c`, eight claims after one warm replay
on merged commit `825120c` fell from 13.003521138 s to 0.053521276 s
(242.96x), `/proc/self/io` read syscalls from 128,226 to 162 (791.52x fewer),
and `rchar` from 27,068,601 to 67,465 bytes (401.22x less). First complete
replay remained 1.597760866 s versus 1.603223845 s, as required for
restart/audit semantics. The before/after hot cProfile artifacts have SHA-256
`5650df4b43aa8bf0698dd35b5ce6deb4ffb4dacc37a78818a3cbe9d569ff3d76`
and `1ce2aadc0fdbc8715d5b5141b8cda93fc784124444832f37b3dc2255bb09de6e`.
Raw `system.cpu`, `system.io`, `system.net`, and `nfs.rpc` Netdata windows from
both active `gx10-6b77` and Sparky are included and individually hashed by the
manifest. This is a local-filesystem CPU microprofile with host-context
telemetry; it used no GPU or live Slurm service and does not qualify shared-NFS
latency, a daemon, an allocation, or deployment.
The final core/Slurm/Dagster/docs-focused suite passed `215 passed, 1 skipped`;
the skip is the existing optional-dependency boundary.

Scheduler mutations use one append-only ordinal journal. The active protocol
emits only `cancel`: after proving the exact bound allocation is active, the
adapter first-writer-claims the next mutation ordinal, rechecks receipt and
scheduler identity/state, and issues at most one cluster-qualified `scancel`.
Cancel must be the unique final mutation. Concurrent contenders have one
winner. A crash, timeout, or command error after the claim is deliberately
ambiguous; a restarted caller sees the final claim and never replays the RPC.
The schema admits a future `requeue` kind so cancel and retry cannot race in
separate journals, but the current zero-requeue policy rejects any such record.
SLURM state directories are created component-by-component relative to held
directory descriptors (`mkdirat` semantics), and reads, listings, and
first-writer publication use `O_NOFOLLOW`-anchored directory descriptors.
New directory entries and final read-only file modes are fsynced before use.
Symlinked directory hops, noncanonical/writable files, gaps, wrong ordinals,
excess counters, wrong identities, and self-digest mismatches refuse. This is
a Linux/procfs implementation contract: atomic temporary publication uses the
held parent through `/proc/self/fd` rather than reopening its pathname.

This remains one allocation lineage per action. `recompute=True` is de-menued
in the SLURM adapter: launching a second allocation after the canonical receipt
exists would make resolve/cancel short-circuit on that old receipt and strand
the new allocation. The adapter never silently creates a fresh job or retries
a terminal one. Repair after an irrecoverably ambiguous or exhausted lineage
is an explicit operator/schema action. The verified CAS receipt remains the
sole result authority throughout.

### Optional Dagster orchestration (implemented, not deployed)

`prismaquant.prismabuild_dagster` is an optional-import adapter over the core
and SLURM resource layer; importing PrismaQuant still does not import or require
Dagster. `ActionSpec` binds the sealed action, checkout, exact SLURM resources
and placement, zero-requeue/durable-poll policy, and content-addressed upstream
dependencies. An edge is the tuple `(upstream action key, downstream input id,
result sha256, result bytes)`. Graph construction refuses an edge unless that
tuple is also present exactly in the downstream action's sealed `inputs`, and
uses a key-sorted topological order.

Native definitions use one asset key per action key and set `code_version` to
that full key. The resource requires the single cluster name in addition to
CAS/log/worker paths. Dagster-level retries and same-job requeues are disabled.
`DagsterActionRunner` passes the action's poll maximum and interval into the
sealed pre-submit identity, accepts either a new submission or adoption, loads
durable progress, and lets the adapter pace and atomically claim every poll. A
fresh runner therefore continues the remaining limit instead of resetting a
Python counter. On poll exhaustion it cancels only the bound allocation and
then re-reads the CAS, so a receipt published concurrently with a no-op or
completed cancellation wins the race. A cache hit, upstream
dependency, or successful SLURM resolution is accepted only after an independent
`PrismaBuildCAS.lookup()` verifies the exact receipt, producer scope, and blob
bytes. Dagster run and materialization state are views of CAS truth, never
certification themselves. The optional package extra is
`prismaquant[prismabuild]` (supported `>=1.13,<2`, checked against 1.13.20); no
daemon, webserver, workspace, or scheduler installation is performed by the
repository. A native-import package-level run against Dagster 1.13.20 passed
all 18 adapter tests on 2026-08-31 (14 Torch deprecation warnings):

```bash
PYTHONPATH=/home/rob/venvs/pq-cu130/lib/python3.12/site-packages /tmp/pq-prismabuild-dagster-1.13.20/bin/python -m pytest -q tests/test_prismabuild_dagster.py
```

This is adapter compatibility evidence, not a daemon, workspace,
materialization, or restart pilot.

### Remaining live-deployment gates

The state machine above is covered by mocked crash/restart/corruption tests and
the scheduler-free two-host NFS race above; it has not submitted, adopted,
polled, or cancelled a live SLURM job.
Deployment still requires all of the following evidence:

- The fake-command race above covers shared-CAS `intent.json`, `job.json`, poll
  claims, and the unified mutation journal. Host/power-loss injection around
  every file, link, directory, and scheduler-RPC boundary remains required.
- Live `slurmctld`/`slurmd`/`slurmdbd` behavior with accounting configured to
  retain job comments (`AccountingStoreFlags=job_comment`) for longer than the
  adoption horizon. Exact `JobName`, `Comment`, `Cluster`, allocation-only
  filtering, state widths, single-cluster selection, and permissions must be
  verified on the deployed version. Purged accounting leaves an
  intent-without-binding deliberately ambiguous; mutable/absent comments fail
  adoption. Slurm permits an authorized user to mutate stored job comments,
  including after completion, so the deployment must isolate the submission
  principal or independently audit `Comment`/`JobName` mutations; these
  scheduler fields are discovery evidence, not a cryptographic identity.
- Crash injection immediately before/after `sbatch`, binding publication,
  cancel-journal publication, and `scancel`, including accounting lag, stale
  terminal jobs, numeric job-id reuse, command timeout, and concurrent
  orchestrators. No live SLURM validation is claimed.
- A live forced/admin restart must demonstrate that `SLURM_RESTART_COUNT` is
  present and nonzero before the worker reaches task argv and that
  `--no-requeue` blocks ordinary restart. Positive same-job retry remains
  unavailable; a future implementation additionally needs trustworthy
  scheduler `Restarts` reconciliation and exact claim-to-worker binding.
- NTP/chrony monitoring must bound wall-clock offset between orchestrator hosts
  because durable poll pacing uses append timestamps and refuses rollback.
- A real Dagster daemon/webserver restart and concurrent-run pilot proving that
  Dagster-level retry settings cannot escape through a second submission and
  that the durable poll budget and pacing interval are preserved.
- Storage retention and access control for the active submission namespace.
  Read-only hard-linked files plus strict semantic/canonical validation reject
  malformed, conflicting, writable, symlinked, and gapped state, but this is
  not a keyed or WORM log. Individual Slurm state records are capped at 16 MiB,
  and history/temp entry counts are bounded before loading their contents. An
  authorized directory owner can still unlink the final member of an
  append-only prefix; preventing or auditing that tail deletion requires the
  deployed filesystem/ACL/backup policy.
- The orchestrator hosts must expose Linux `openat`/`O_NOFOLLOW` semantics and
  `/proc/self/fd`, and the worker/Slurm executable path ancestors must be
  immutable to the submitting principal. The adapter seals and rehashes the
  worker leaf immediately before submit, but a later scheduler launch cannot
  retain that submit-host file descriptor across the allocation boundary.
- Production-scale large-result NFS publication, host-loss directory
  durability, munge/cgroup worker attestation, launcher deployment, and the
  production-run Netdata/Prometheus evidence on both boxes remain separate
  gates. The D9 CPU microprofile's two-host context windows are not that live
  scheduler/production telemetry qualification.
- The local SIGKILL test proves deterministic cleanup while `flock` exclusion
  is available in one filesystem/process environment. Deployment must still
  prove that the shared checkout mount enforces that lock across hosts and
  inject host/power loss at the claim, unlink, staging-reap, and recompute
  fsync boundaries.

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
