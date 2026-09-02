# Sample-parallel incremental probe

> **STATUS 2026-09-02: UNAVAILABLE — the lane it was built for is retired.**
>
> This document describes the design accurately and is kept for that reason,
> but the lane does not run today. `prepare-run-contract` and the per-worker
> source census below were both built on
> `prismaquant.rtx4090_artifact_census` — the strict-Ada FP8-CB campaign's
> closed Qwen3.8-27B layout — which went to
> `archive/gridbook_lane_2026-09-02/` when Rob retired the Gridbook codebook
> lane. Nothing can mint a run contract, and
> `incremental_probe.py --global-calibration-tensor` refuses up front rather
> than admitting a pre-retirement contract with one leg of its identity replay
> missing. Reviving sample parallelism means giving the census a
> lane-independent source of truth. Recorded as debt D34 in
> `docs/ARCHITECTURE.md`.

Status: opt-in RTX 4090 producer lane, 2026-08-24. This is not a
`run-pipeline.sh` default. The contract is implemented and CPU-tested; no
campaign performance or quality claim is made until the reviewed GPU run.

This lane partitions one immutable tokenized calibration tensor by sample.
Every worker processes the complete text probe census. The merger accepts the
workers only when their partition contracts form one exact, disjoint cover and
their raw sufficient statistics have the same complete, source-derived qname
plan.

## Closed execution contract

Version 1 has no caller-supplied shared metadata or qname lists. Before workers
run, `prepare-run-contract` scans the validated Qwen 3.8 27B source checkpoint
and publishes a digest-bound bundle containing:

- the source tensor/qname census and checkpoint-content identity;
- the full probe manifest (dense body, `lm_head`, and MTP);
- the dense-body activation manifest;
- the terminal BF16 stats-only manifest (`lm_head` and all MTP Linears); and
- one closed execution identity for model, dataset, seed, dtype, importance,
  marginals, estimator/math contracts, text-only scope, 1024 activation rows,
  the immutable PrismaQuant source-snapshot closure/commit/tree, and the
  pinned producer container's `sha256:` image digest.

The source model identity binds checkpoint contents, not just a model path.
Every host must provide its own
`PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE`; workers derive the strict source
census locally and compare its stable projection (config, tensor and
weight-map hashes, deterministic model-content digest and shard/tensor counts,
and exact Linear/qname manifests) before admitting CE, precompute, or shard
reuse. The existing streamed identity v1 keeps its path-bearing local digest.
Its additive cross-host projection strips only recursive `_name_or_path` and
`transformers_version` config provenance and replaces absolute shard paths with
unique basename/size/SHA-256 records; all checkpoint maps remain bound.
Host-dependent derivation and inode fingerprints are not part of the cross-host
comparison. The cheap local fingerprint/config/census check is
repeated immediately before worker publication to close model-source TOCTOU.
The identity cache is host-local and must not be copied between Sparky and
Sparklina.
`prepare-run-contract` also requires a coordinator-local complete cache and
stores its upstream portable streamed-model digest inside the portable source
identity. A contract built by directly hashing indexed shards without that
upstream bridge is refused before any worker starts. The RTX4090 burn later
validates each cache again against the live source fingerprints and semantic
config and requires that portable upstream digest to match; path-bearing v1
digests remain local cache provenance only.

The producer itself must execute from a mounted, read-only runtime snapshot.
The host executes the verifier from the pinned Git object, checks the candidate
snapshot non-writable, and re-hashes its complete tracked-file ledger before
every launch; the container's snapshot-owned verifier repeats that check as
supplemental defense, and publication performs the producer's independent
ledger replay. Container identity has a separate trust boundary: Python inside
a container cannot authoritatively
inspect the host image that launched it, so a trusted launcher must inspect the
immutable image, pass its exact `sha256:<64 hex>` digest, and ensure the
read-only snapshot and model mounts are the ones declared. The run contract
binds both values; the worker compares the launcher-supplied image digest but
does not claim to derive it internally.

The v1 schedule is fixed: complete body range, `--unified-sweep`,
`--no-include-visual`, `--include-mtp`, `--include-lm-head`, no h-detail, and
text-only calibration. Routed, packed, and per-expert statistics or markers
fail closed, including in MTP. All workers must carry the identical complete
execution identity.

The census dispositions are:

| Namespace | Probe statistics | Activation payload | Raw token rows per sample |
| --- | --- | --- | --- |
| Dense body | required | required | `seqlen` |
| `lm_head` | required, terminal BF16 | omitted | `seqlen - 1` |
| MTP | required, terminal BF16 | omitted | `seqlen - 2` |
| Visual | excluded in text-only v1 | omitted | n/a |

All raw statistics must be finite and nonnegative, and every qname must report
the exact local dense token count above. Raw sufficient statistics are added
across workers and Fisher values are finalized exactly once using the existing
global denominator `global_nsamples * seqlen`. Each worker and the final sum
must also satisfy `sum(fisher_row) ≈ sum(fisher_col) ≈ h_trace_raw` at the
repository's qualified `rtol=1e-3`; finite but incorrectly wired marginals are
not accepted.

## Exactness boundary

“Exact” means the same mathematical sample cover, global CE normalization,
raw-stat addition, and one final global Fisher normalization, subject to normal
floating-point tolerance. It does not mean bitwise equality with a monolithic
run: changing GEMM M from the global batch to a partition batch can change
low-order floating-point bits.

Importance-weighted probing is a two-stage barrier. Its v2 receipts bind not
only calibration/partition identity but the complete execution identity,
stable source projection, producer snapshot closure/commit/tree, dtype, and
container image digest:

1. Each worker runs phase 1 and writes its raw shifted-token CE sum/count.
2. `merge-importance` proves the exact sample/count cover and publishes one
   digest-bound global CE receipt, including every local receipt and the full
   unprojected global sum, count, and mean.
3. Each worker reruns phase 1, validates and carries that complete receipt, uses
   the global mean before body backward, and then completes phase 2/3. MTP
   retains its existing per-sample CE mean, which is partition invariant.

Version 1 intentionally uses a duplicate phase-1 forward across the barrier.
Receipts bind the calibration artifact SHA and stamp
`phase1_reused_across_barrier=false` and
`expected_apply_overhead=one_additional_phase1_forward`. No timing estimate is
claimed until a reviewed GPU run measures it.

Activation-cache exactness is dense-body-only, FP32, and requires h-detail
off. The exact priority schema is
`blake2b64-keyed-fmix32x2-prp-global-row-fused-group-v2`. For fused group UTF-8
bytes `g`, key material is
`b"prismaquant.sample-parallel.activation-priority.v2\0" ||
bytes.fromhex(cal_hash) || LE_u32(len(g)) || g`; an eight-byte BLAKE2b digest
is decoded as little-endian uint32 `k0,k1`, and priority is
`fmix32(fmix32(global_row XOR k0) XOR k1)` with Murmur finalizer constants
`0x85EBCA6B` and `0xC2B2AE35`. Exact 16-bit-limb `mullo32` keeps every Torch
intermediate in safe signed int64 range. It is a uint32 permutation, so
priorities are unique and exact top-K is associative without a tie-breaking
rule. Runs fail unless `0 < global_samples * seqlen <= 2^32`.

Global rows, priorities, and top-R reservoirs stay on the input device in
Linear hooks. One selection plan is cached per fused group, so q/k/v and
gate/up siblings reuse the same device indices. The only device-to-host
transfer is the completed layer flush. Worker blobs store shard-local
flat-token indices; the merger independently reconstructs priorities with a
separate scalar/NumPy implementation, proves every local top-R and the exact
global union/top-1024, and publishes global indices. `lm_head` and MTP keep
their probe statistics but intentionally have no activation payload. The
actual `lm_head` Linear call is restricted to shifted-token rows, so its raw
count and activation marginals cover exactly `N*(T-1)` rather than including
the unscored final token.

Publication claims are not trusted on replay. The public validator independently
recomputes the priority of every retained global row, the canonical top-R over
the complete `0..N*T-1` domain, exact cardinality/order, row uniqueness/range,
and identical rows for every fused priority group. A self-consistently re-sealed
manifest cannot authorize reversed, duplicated, out-of-domain, truncated, or
fused-misaligned rows.

Final merge publication is one required output bundle, never two independently
visible paths. `probe.pkl`, `activation_cache/`, and final `commit.json` are
fully written and fsynced below one same-filesystem staging parent; only then
is that directory atomically renamed with no replacement. A crash before the
rename exposes no output pair and an exact retry succeeds. A committed bundle
is never clobbered. Calibration creation is likewise restartable if the tensor
was durably published before its manifest. Worker-private shard and activation
files remain resumable work caches; only completed outputs are publication
artifacts.

The resumable phase-1/2 precompute uses the existing `precomputed.pt` cache,
not another residency mechanism. Its atomic, versioned payload is validated as
a closed object before reuse. On this dense Qwen lane it must contain exactly
the raw `lm_head` resident Fisher cover with source geometry,
`N_i*(T-1)` tokens, FP32 marginals, FP32 `h_full[out,in]`, FP32 per-token
gradient-square values, mutually consistent traces, and no routed, resident
activation, or shared-pass state. Decoder activations must cover every layer
boundary and all cached floating-point values must be finite. A malformed or
stale cache is a miss and is recomputed through the existing streamed GPU path.

Every worker also receives the digest-bound `cover.json`. Before any model or
GPU setup, it strictly decodes the cover and run-contract JSON (duplicate
object members are rejected), replays both closed schemas and digests, and
requires the cover's execution identity and qname census to equal the run
contract. Its selected partition must equal the corresponding cover member
exactly, which binds the current calibration artifact SHA-256 and both the
local and global calibration hashes. Replacing `calibration.pt` after
`build-cover` therefore fails during CPU preflight rather than after GPU work.

## Commands

The examples below use two workers. All paths under `worker-$I` must be private
to that worker. They deliberately do not execute `python -m` from a live
checkout. First commit the reviewed producer, require a clean tree, materialize
and host-verify its tracked snapshot, inspect the pinned image on the host, and
launch that exact digest-qualified image reference with read-only
source/model/dataset mounts:

```bash
set -euo pipefail

SP_TRUSTED_REPO_HOST=/home/rob/prismaquant
SP_COMMIT="$(git -C "$SP_TRUSTED_REPO_HOST" rev-parse --verify 'HEAD^{commit}')"
test -z "$(git -C "$SP_TRUSTED_REPO_HOST" status --porcelain --untracked-files=all)"
sp_trusted_snapshot_tool() {
  git -C "$SP_TRUSTED_REPO_HOST" \
    show "$SP_COMMIT:tools/prismaquant_runtime_snapshot.py" |
    env -u PYTHONPATH \
      PYTHONSAFEPATH=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONNOUSERSITE=1 \
      python3 -P -B -s - "$@"
}
SP_SNAPSHOT_JSON="$(
  sp_trusted_snapshot_tool materialize \
    --source-root "$SP_TRUSTED_REPO_HOST" \
    --cache-root /home/rob/dq-runs/runtime-snapshots \
    --commit "$SP_COMMIT"
)"
sp_snapshot_field() {
  env -u PYTHONPATH \
    PYTHONSAFEPATH=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    python3 -P -B -s -c \
      'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' \
      "$1" <<<"$SP_SNAPSHOT_JSON"
}
SP_SNAPSHOT="$(sp_snapshot_field snapshot)"
SP_TREE="$(sp_snapshot_field tree)"
SP_CLOSURE="$(sp_snapshot_field closure_sha256)"
find -P "$SP_SNAPSHOT" \( -type f -o -type d \) -exec chmod a-w {} +
test -z "$(find -P "$SP_SNAPSHOT" -perm /222 -print -quit)"
sp_trusted_snapshot_tool verify \
  --snapshot "$SP_SNAPSHOT" \
  --expected-commit "$SP_COMMIT" \
  --expected-tree "$SP_TREE" \
  --expected-closure-sha256 "$SP_CLOSURE"

SP_IMAGE_REF='eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869'
docker pull "$SP_IMAGE_REF"
SP_IMAGE_DIGEST="${SP_IMAGE_REF##*@}"
[[ "$SP_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
  "$SP_IMAGE_REF" | grep -Fx -- "$SP_IMAGE_REF" >/dev/null
export PRISMAQUANT_PRODUCER_IMAGE_DIGEST="$SP_IMAGE_DIGEST"
export PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT=/pq
```

The campaign receipt records this complete reviewed RepoDigest and the
host-inspected immutable registry RepoDigest. Docker's backend-dependent local
`.Id` is not a cross-host identity and is never placed in the contract. The
verifier above comes directly from the
pinned Git object, not from the candidate snapshot it judges; the snapshot is
then made and checked non-writable. Repeat that verification on every host.

Host layouts may differ, but all value-bearing commands use the same canonical
container paths. On each host set the four `_HOST` inputs below, while retaining
the literal container names. The calibration loader content-sniffs the
extensionless regular file mounted at canonical `/dataset`, so this bind cannot
fall through to a Hugging Face dataset lookup. `SP_RUN_HOST` must be one shared tree, or its
published barrier artifacts must be transferred byte-for-byte to the other host
before the next stage. Worker state remains private:

```bash
SP_MODEL_HOST=/absolute/host/path/to/model
SP_DATASET_HOST=/absolute/host/path/to/diverse-v1.jsonl
SP_RUN_HOST=/absolute/host/path/to/campaign
SP_WORKER_STATE_HOST=/absolute/host/path/to/private-worker-state

SP_MODEL=/model
SP_DATASET=/dataset
SP_RUN=/run
SP_WORKER_STATE=/worker-state
SP_HOST_UID="$(id -u)"
SP_HOST_GID="$(id -g)"
test "$SP_HOST_UID" -ne 0
test -r /dev/nvidia0 && test -w /dev/nvidia0

mkdir -p "$SP_RUN_HOST/coordinator" "$SP_RUN_HOST/burn" \
  "$SP_WORKER_STATE_HOST/home" "$SP_WORKER_STATE_HOST/cache" \
  "$SP_WORKER_STATE_HOST/tmp" "$SP_WORKER_STATE_HOST/torchinductor" \
  "$SP_WORKER_STATE_HOST/triton"
```

Define the launcher on each host. It replays the trusted host verifier before
launch, starts by the immutable RepoDigest-qualified reference, and re-verifies
the read-only snapshot inside the same container as a supplemental check. It
removes `PYTHONPATH`, enables Python safe-
path/no-user-site/no-bytecode modes, requires both Gridbook compilation switches,
and enters only through the snapshot-owned source bootstrap:

```bash
sp_module() {
  local module="$1"
  shift
  sp_trusted_snapshot_tool verify \
    --snapshot "$SP_SNAPSHOT" \
    --expected-commit "$SP_COMMIT" \
    --expected-tree "$SP_TREE" \
    --expected-closure-sha256 "$SP_CLOSURE" >/dev/null || return 1
  docker run --rm --gpus all --ipc=host \
    --user "$SP_HOST_UID:$SP_HOST_GID" \
    --workdir /worker-state/tmp \
    --mount "type=bind,src=$SP_SNAPSHOT,dst=/pq,readonly" \
    --mount "type=bind,src=$SP_MODEL_HOST,dst=$SP_MODEL,readonly" \
    --mount "type=bind,src=$SP_DATASET_HOST,dst=$SP_DATASET,readonly" \
    --mount "type=bind,src=$SP_RUN_HOST,dst=$SP_RUN" \
    --mount "type=bind,src=$SP_WORKER_STATE_HOST,dst=$SP_WORKER_STATE" \
    --env "SP_MODULE=$module" \
    --env "SP_COMMIT=$SP_COMMIT" \
    --env "SP_TREE=$SP_TREE" \
    --env "SP_CLOSURE=$SP_CLOSURE" \
    --env "SP_HOST_UID=$SP_HOST_UID" \
    --env "SP_HOST_GID=$SP_HOST_GID" \
    --env HOME=/worker-state/home \
    --env XDG_CACHE_HOME=/worker-state/cache \
    --env TMPDIR=/worker-state/tmp \
    --env TORCHINDUCTOR_CACHE_DIR=/worker-state/torchinductor \
    --env TRITON_CACHE_DIR=/worker-state/triton \
    --env "PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT=/pq" \
    --env "PRISMAQUANT_PRODUCER_IMAGE_DIGEST=$SP_IMAGE_DIGEST" \
    --env "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE=${PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE:-}" \
    --env PRISMAQUANT_CB_ENCODE_COMPILE=1 \
    --env PRISMAQUANT_CB_ATOM_COMPILE=1 \
    --env PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1 \
    --env PYTHONSAFEPATH=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONNOUSERSITE=1 \
    --entrypoint /bin/bash "$SP_IMAGE_REF" -c '
      set -euo pipefail
      test "$(id -u)" = "$SP_HOST_UID"
      test "$(id -g)" = "$SP_HOST_GID"
      python3 -P -B -s /pq/tools/prismaquant_runtime_snapshot.py verify \
        --snapshot /pq \
        --expected-commit "$SP_COMMIT" \
        --expected-tree "$SP_TREE" \
        --expected-closure-sha256 "$SP_CLOSURE" >/dev/null
      exec env -u PYTHONPATH \
        PQ_RUNTIME_PRISMAQUANT_ROOT=/pq \
        python3 -P -B -s /pq/tools/prismaquant_source_bootstrap.py \
          run-module --source-root /pq "$SP_MODULE" "$@"
    ' bash "$@"
}
```

The trusted operator must also verify that both hosts report the exact
`SP_IMAGE_REF` in `RepoDigests` and use the common `SP_IMAGE_DIGEST` before
recording the execution attestation. Writable mounts are limited to the run and
worker-state trees; the model, dataset, and runtime source are read-only. The
reviewed Spark launcher runs as the invoking host UID/GID, not the image's
default root, and keeps writable compiler/cache/home state under the private
worker tree. On both campaign hosts the NVIDIA device nodes are user-readable
and writable; a host with different device permissions must add its reviewed
GPU group rather than reverting the producer to root.

Prepare the immutable global tensor once on a trusted filesystem:

```bash
sp_module prismaquant.sample_parallel_probe prepare-calibration \
  --model "$SP_MODEL" \
  --dataset "$SP_DATASET" \
  --nsamples 32 \
  --seqlen 1024 \
  --calib-seed 42 \
  --partitions 2 \
  --output "$SP_RUN/calibration.pt" \
  --manifest-output "$SP_RUN/calibration.json"
```

On the coordinator, create and validate the complete streamed identity through
the repository's existing streamed-model/cache path. This is a GPU-bearing
initialization step and must run inside the pinned producer environment. Keep
the JSON written to stdout as its operation receipt:

```bash
sp_module prismaquant.sample_parallel_probe prepare-worker-source-cache \
  --model "$SP_MODEL" \
  --output "$SP_RUN/coordinator/source-identity-cache.json" \
  --offload-folder "$SP_RUN/coordinator/source-identity-offload" \
  | tee "$SP_RUN_HOST/coordinator/source-identity-receipt.json"
```

Derive the authoritative source census and closed execution identity. The
activation row limit is explicitly fixed at 1024 for this campaign. These
values come from the trusted launcher, not an unverified in-container guess:

```bash
sp_module prismaquant.sample_parallel_probe prepare-run-contract \
  --model "$SP_MODEL" \
  --dataset "$SP_DATASET" \
  --calib-seed 42 \
  --dtype bf16 \
  --importance-weighting \
  --emit-marginals \
  --activation-rows-limit 1024 \
  --producer-snapshot-root "$PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT" \
  --container-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --source-identity-cache "$SP_RUN/coordinator/source-identity-cache.json" \
  --output "$SP_RUN/run-contract.json"
```

Bind that contract to the exact calibration partition cover:

```bash
sp_module prismaquant.sample_parallel_probe build-cover \
  --calibration-manifest "$SP_RUN/calibration.json" \
  --run-contract "$SP_RUN/run-contract.json" \
  --output "$SP_RUN/cover.json"
```

On each worker, independently create its host-local cache (never copy the
coordinator's or the other worker's file), then export that exact path:

```bash
sp_module prismaquant.sample_parallel_probe prepare-worker-source-cache \
  --model "$SP_MODEL" \
  --output "$SP_WORKER_STATE/source-identity-cache.json" \
  --offload-folder "$SP_WORKER_STATE/source-identity-offload" \
  | tee "$SP_WORKER_STATE_HOST/source-identity-receipt.json"
export PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE=/worker-state/source-identity-cache.json
```

On worker `I` (`0` or `1`), collect the CE scalar. Although stage 1 exits
before the Fisher probe, `incremental_probe` still requires its normal private
output/cache/work arguments:

```bash
sp_module prismaquant.incremental_probe \
  --model "$SP_MODEL" \
  --dataset "$SP_DATASET" \
  --seqlen 1024 \
  --calib-seed 42 \
  --dtype bf16 \
  --global-calibration-tensor "$SP_RUN/calibration.pt" \
  --sample-partition-index "$I" \
  --sample-run-contract "$SP_RUN/run-contract.json" \
  --sample-cover "$SP_RUN/cover.json" \
  --producer-snapshot-root "$PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT" \
  --container-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --sample-importance-stats-output "$SP_RUN/worker-$I/ce.json" \
  --importance-weighting --emit-marginals \
  --include-mtp --include-lm-head --no-include-visual \
  --unified-sweep --activation-rows-limit 1024 \
  --activation-cache-dir "$SP_RUN/worker-$I/act" \
  --work-dir "$SP_RUN/worker-$I/work" \
  --output "$SP_RUN/worker-$I/probe.pkl"
```

Merge the local CE receipts at the barrier:

```bash
sp_module prismaquant.sample_parallel_probe merge-importance \
  --local-stats "$SP_RUN/worker-0/ce.json" "$SP_RUN/worker-1/ce.json" \
  --output "$SP_RUN/global-ce.json"
```

Then rerun each worker with the complete global receipt:

```bash
sp_module prismaquant.incremental_probe \
  --model "$SP_MODEL" \
  --dataset "$SP_DATASET" \
  --seqlen 1024 \
  --calib-seed 42 \
  --dtype bf16 \
  --global-calibration-tensor "$SP_RUN/calibration.pt" \
  --sample-partition-index "$I" \
  --sample-run-contract "$SP_RUN/run-contract.json" \
  --sample-cover "$SP_RUN/cover.json" \
  --producer-snapshot-root "$PRISMAQUANT_PRODUCER_SNAPSHOT_ROOT" \
  --container-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --sample-global-importance-receipt "$SP_RUN/global-ce.json" \
  --importance-weighting --emit-marginals \
  --include-mtp --include-lm-head --no-include-visual \
  --unified-sweep --activation-rows-limit 1024 \
  --activation-cache-dir "$SP_RUN/worker-$I/act" \
  --work-dir "$SP_RUN/worker-$I/work" \
  --output "$SP_RUN/worker-$I/probe.pkl"
```

The strict merge gets both authoritative qname manifests from the cover. It
does not accept caller-provided qname lists:

```bash
sp_module prismaquant.sample_parallel_probe merge \
  --cover "$SP_RUN/cover.json" \
  --probe-shards "$SP_RUN/worker-0/probe.pkl" "$SP_RUN/worker-1/probe.pkl" \
  --activation-cache "0=$SP_RUN/worker-0/act" "1=$SP_RUN/worker-1/act" \
  --output-bundle "$SP_RUN/merged" \
  --max-rows 1024
```

The strict RTX4090 burn consumes the merged bundle directly. First derive the
column weights from the exact captured probe object; do not use the generic
pipeline harvester, which does not enter through this bundle contract:

```bash
sp_module prismaquant.rtx4090_fp8_burn derive-col-weights \
  --sample-merge-bundle "$SP_RUN/merged" \
  --output "$SP_RUN/cb_col_weights.pkl"
```

Create one execution attestation from the already closed sample run contract,
the manifest of the mounted producer snapshot, and the registry RepoDigest
verified by the host launcher. Then prepare the path-independent two-stripe
plan on the coordinator using its locally validated source cache:

```bash
sp_module prismaquant.rtx4090_fp8_burn attest-execution \
  --sample-run-contract "$SP_RUN/run-contract.json" \
  --producer-snapshot /pq/.prismaquant-runtime-snapshot.json \
  --launcher-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --output "$SP_RUN/burn/execution-attestation.json"

sp_module prismaquant.rtx4090_fp8_burn prepare \
  --model "$SP_MODEL" \
  --probe "$SP_RUN/merged/probe.pkl" \
  --col-weights "$SP_RUN/cb_col_weights.pkl" \
  --dataset "$SP_DATASET" \
  --source-identity "$SP_RUN/coordinator/source-identity-cache.json" \
  --producer-snapshot /pq/.prismaquant-runtime-snapshot.json \
  --execution-attestation "$SP_RUN/burn/execution-attestation.json" \
  --launcher-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --sample-merge-commit "$SP_RUN/merged/commit.json" \
  --activation-cache-dir "$SP_RUN/merged/activation_cache" \
  --output-dir "$SP_RUN/burn/plan" \
  --n-calib-samples 32 --calib-seqlen 1024 --calib-seed 42
```

Publish the merged bundle, column weights, attestation, and complete plan to
both hosts byte-for-byte. On host `I` set `I=0` or `I=1`, retain its canonical
`PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE=/worker-state/source-identity-cache.json`,
and measure exactly its assigned whole-layer stripe. Checkpoint/offload state is
host-private; the receipt-bearing final stripe is common campaign state:

```bash
sp_module prismaquant.rtx4090_fp8_burn measure \
  --model "$SP_MODEL" \
  --probe "$SP_RUN/merged/probe.pkl" \
  --col-weights "$SP_RUN/cb_col_weights.pkl" \
  --dataset "$SP_DATASET" \
  --source-identity "$SP_WORKER_STATE/source-identity-cache.json" \
  --producer-snapshot /pq/.prismaquant-runtime-snapshot.json \
  --execution-attestation "$SP_RUN/burn/execution-attestation.json" \
  --launcher-image-digest "$PRISMAQUANT_PRODUCER_IMAGE_DIGEST" \
  --sample-merge-commit "$SP_RUN/merged/commit.json" \
  --activation-cache-dir "$SP_RUN/merged/activation_cache" \
  --plan "$SP_RUN/burn/plan/campaign-plan.json" \
  --stripe "$I" \
  --checkpoint-dir "$SP_WORKER_STATE/burn-stripe-$I" \
  --output "$SP_RUN/burn/stripe-$I.pkl"
```

The three compile switches are one closed producer input.  For every live CB
render, the scorer uses `torch.compile` with Inductor, `fullgraph=True`,
`dynamic=True`, and `suppress_errors=False`; each call must enter exactly one
compiled backend dispatch or the stripe aborts.  A fully or partially resumed
stripe may cover restored units transitively because the existing AURA loader
has already checksummed and identity-validated each unit envelope against a
manifest that binds these settings, the producer arm/snapshot, and the streamed
source model.  The receipt separates live and restored unit/cell counts.  It
marks the atom route `not_applicable` because this lattice campaign has
`ldlq=false`/`ldlq_scope=none`; `PRISMAQUANT_CB_ATOM_COMPILE=1` is an
identity-bound setting, not a claim that an atom kernel executed.

After both receipt-bearing stripes are present on the coordinator, merge once
and run the single exact-byte allocator. These stages revalidate the snapshot,
plan, imatrix, source census, and disjoint stripe receipts; they do not render
the nine imputed rungs:

```bash
sp_module prismaquant.rtx4090_fp8_burn merge \
  --plan "$SP_RUN/burn/plan/campaign-plan.json" \
  --producer-snapshot /pq/.prismaquant-runtime-snapshot.json \
  --col-weights "$SP_RUN/cb_col_weights.pkl" \
  --shards "$SP_RUN/burn/stripe-0.pkl" "$SP_RUN/burn/stripe-1.pkl" \
  --output "$SP_RUN/burn/aura-merged.pkl"

sp_module prismaquant.rtx4090_fp8_burn allocate \
  --plan "$SP_RUN/burn/plan/campaign-plan.json" \
  --producer-snapshot /pq/.prismaquant-runtime-snapshot.json \
  --model "$SP_MODEL" \
  --probe "$SP_RUN/merged/probe.pkl" \
  --sample-merge-commit "$SP_RUN/merged/commit.json" \
  --activation-cache-dir "$SP_RUN/merged/activation_cache" \
  --col-weights "$SP_RUN/cb_col_weights.pkl" \
  --merged "$SP_RUN/burn/aura-merged.pkl" \
  --cost-output "$SP_RUN/burn/allocator-cost.pkl" \
  --output-dir "$SP_RUN/burn/allocation" \
  --threads 16
```

The committed outputs are `$SP_RUN/merged/probe.pkl`,
`$SP_RUN/merged/activation_cache/`, and `$SP_RUN/merged/commit.json`.
Downstream consumers enter through `validate_sample_parallel_merge_bundle`,
which replays exact directory topology, commit/member hashes, closed merged
probe rows, the activation manifest, and every activation tensor. Probe hashing
and deserialization use the same `O_NOFOLLOW` descriptor; the RTX4090 burn
consumes that captured object instead of reopening the pathname. Its lazy
activation index holds the validated directory descriptor and compares every
opened tensor/stamp with the captured committed manifest at use time. The RTX4090
burn then joins the bundle's portable source-census digest to the additive
path-neutral streamed-model content digest derived from each freshly validated
host-local v1 identity cache before accepting it. Host-local path-bearing v1
digests remain cache-validation provenance and are not compared across hosts.

All model execution remains on the existing streamed model, layer cache,
prefetch, and activation-cache paths. This lane adds no second residency or
weight-cache mechanism and changes no serving-runtime, vLLM graph, or compiler
gate.
