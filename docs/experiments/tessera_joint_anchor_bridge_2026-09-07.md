# Completed Tessera anchors to full-model joint AURA

Issue #322 adds a checked bridge from a completed Tessera campaign into existing
streamed joint AURA. The scalar campaign establishes which exact wires and BF16
decoded renders exist; its MSE and interpolated prices never become joint costs.
Preparation verifies actual source weights, original prefetched Hessians and
encoding settings against every wire receipt, decodes that wire, and requires
exact equality with its original PWC render. Pricing uses precisely those
per-Linear formats plus source BF16. This research bridge changes no wire format,
menu default, allocator default, export lane or serving gate.

No full-model joint result has been produced by this bridge at this writing.
The first-model invocation uses all 2,142 census Linears in all 38 groups from
`full-model-anchors-02`, with the original 512 × 512 token artifact and capture.
Settings are all-token joint activation pricing, complete-sequence microbatch 1,
four global-row Rademacher probes, seed 7000, temperature 1, source BF16, FP32
weight deltas and optional production activation clipping disabled. Measured
E2M1 retains its unified fused-module static scale. The plan and prepared record
bind source/backend/calibration and implementation identities. Resume requires
the exact existing checkpoint identity; subset draws and mixed checkpoints fail.

The plan schema is `prismaquant.tessera_joint_aura.plan.v1`. Its `inputs` contains
independently bound `{path, sha256}` entries for campaign plan, census, complete
fleet receipts, merged cost and merged checkpoint, plus `required_source_units`
and `required_campaign_groups`. Model, token input, canonical capture, execution
settings, output root, PWC/GPU budgets, memory floor and profiler are explicit.
The concrete first-model plan sealer and provisional calculation are retained at
`/mnt/shared/tessera-measurements/first-model-20260907/full-model-joint-aura/`.
They cannot seal a plan until required merged artifacts and receipts exist.

After independently verifying the merged campaign and importer qualification,
use these dependent PB actions. Supply the actual plan SHA256; independently
rehash the completed preparation record after checking its fleet terminal,
logs, CAS receipt and contents. Neither action uses a latest-file pointer.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-joint-anchor-bridge --tag gb10 \
  --cpus 4 --gpu --demand mem_gb=96 --gpu-memory-gb 40 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --detach -- bash experiments/pq322_joint_anchor_run.sh prepare \
  --plan "$JOINT_PLAN" --plan-sha256 "$PLAN_SHA256"

python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-joint-anchor-bridge --tag gb10 \
  --cpus 4 --gpu --demand mem_gb=96 --gpu-memory-gb 40 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --detach -- bash experiments/pq322_joint_anchor_run.sh run \
  --plan "$JOINT_PLAN" --plan-sha256 "$PLAN_SHA256" \
  --prepared "$PREPARED_RECORD" --prepared-sha256 "$PREPARED_SHA256"
```

The pinned known-good container preserves PB affinity and ownership. PB owns
placement; preparation and pricing have an actual data dependency. Existing
streamed pass/calibration/probe semantics do not expose independent model-layer
jobs, so the bridge adds no dispatcher. Existing completed-layer checkpoints
retain their restart semantics and are additionally bound to the input plan.

Each action writes full-duration `profile.pstats`, readable `profile.txt`,
`/proc/self/io` counters, CUDA allocation peaks, source identity, phase epochs and
`results.json`. Collect both boxes' Netdata series with the existing
`/mnt/shared/tessera-measurements/pq300-batch-20260907/harness/collect_netdata.py`
against each result, retaining raw series and any coverage errors. Shared-host
profiles describe this run and establish no isolated speedup. Check terminal
status and cleanup, output roster/hashes, CAS payload and receipt before consuming
`prepare/prepared.json` or `run/joint-cost.pkl`.

The provisional reservation is CPU 4/native 1, 96 GiB aggregate and a 40 GiB GPU
subset on GB10. The earlier 96-unit, one-render full-calibration run measured
20,751,726,080 CUDA allocated bytes and 23,603,445,760 reserved bytes, with
26,843,545,600 CPU boundary bytes. Four rolling BF16 cotangent planes add about
4 GiB. The largest census decoder layer has 362,807,296 quantizable parameters
in 100 Linears. At seven measured candidates each, simultaneously resident FP32
deltas need 10,158,604,288 bytes and CPU BF16 PWC renders need 5,079,302,144 bytes.
The explicit 6 GiB PWC cap covers only these renders. Source residency, deltas,
boundary/cotangent planes, activation/backward temporaries, Hessian qualification
buffers and allocator slack remain inside the aggregate/GPU reservation.
The measured GPU baseline includes source residency and one-candidate deltas;
adding the difference to the largest seven-candidate layer suggests roughly
30 GB allocated before slack. This is an estimate, not an observed full-model
peak. Recompute from the complete actual roster before submission; partial
telemetry is no basis for shrinking the reservation.
