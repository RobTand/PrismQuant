# Bounded PrismaQuant #183 LFM measurement plan — 2026-09-05

Status: prepared, **not measured**. This is an opt-in experiment driver, not a
production default, shipcard gate completion, allocator optimum, or quality
promotion. The coordinator owns PrismaBuild admission and final receipts.

`experiments/pq183_lfm_bound.py` runs a real calibration/campaign through the
existing capture, Hessian, and `ProductionWeightCache` path. It then selects the
predeclared measured E4M3/q1024 recipe, carries the allocator's existing expert
projection, Hessian, static-scale and serving-scope metadata, calls the same
preflight/translator/cached-wire exporter handoff as `run-pipeline.sh`, and
compares actual exported safetensors payload bytes with the priced receipts.
The experiment does not manufacture Fisher statistics or unmeasured cost rows.

## Inputs and scope

* PrismaQuant base: `cda074a8c063a9e9ebd59bb549271c6160a91796`, plus the
  separately reviewed campaign priced-population coverage fix and this driver.
  Record the final snapshot/CAS identity, not only that base commit.
* Tessera: `ba582d476a3b6db9057ebd1385dc52926f171451`, supplied as a sealed
  external dependency snapshot. Production code is not vendored.
* Source: `/mnt/shared/models/LFM2.5-8B-A1B-BF16`. The driver refuses a topology
  other than 24 layers, first two dense, 32 experts, top four, hidden 2048,
  expert intermediate 1792. The producer seals the actual checkpoint bytes.
* Producer image ID:
  `sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88`
  (`tessera/lfm25-teacher:base-61fc8a-cconv-245e314e`, torch 2.13/CUDA 13.0,
  transformers 5.15.1; verify versions from the actual container).
* Serving target: SM121, TP1, eager, resident, exact
  `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
  The producer environment and serving environment differ; their timings are
  not a same-image comparison.
* One real campaign: attested menu, layer stride 12, one anchor/round/budget,
  eight Wikitext train calibration samples, length 512, seed zero, maximum
  512 activation rows per unit, required Hessian. The campaign records its
  actual draw and per-unit row counts; expert routed support is not presumed
  to equal the dense 4096-token draw.
* The projected population is exactly the 96 expert matrices in
  `model.layers.12.feed_forward.experts.{0..31}.{w1,w3,w2}`. Only measured
  H-aware E4M3/q1024 rows can be selected. The layer-0 dense matrices have no
  admitted Tessera cell on this serving image and remain explicitly BF16.
  Every other source body matrix is explicitly BF16 as well. The campaign's
  population record is carried verbatim; `recipe.json` separately names every
  selected and BF16 assignment. Embeddings and `lm_head` are not in the bpp
  denominator. The reported selected-wire bpp excludes BF16 and fused-container
  framing and is **not whole-model bpp**.

## Resource and environment envelope

Use one exclusive GB10 GPU, four physical CPU cores, and a 64 GiB aggregate
memory reservation/cap. Bind native OMP/MKL/OpenBLAS thread counts to one and
preserve PB-assigned CPU affinity in producer/census. The unchanged upstream
paired smoke launcher has no native-thread injection option; PB's four-CPU
cpuset bounds all its threads, and the runner checks/records the actual Docker
cpuset and environment. Its default thread count is not claimed to be one.
Expected selected expert H storage is about
1.45 GiB, capped raw expert activations about 0.36 GiB, selected expert weights
about 0.67 GiB, plus the dense capture and roughly 16 GiB full BF16 model.
Loader copies, CUDA context, factorization and temporary tensors must fit the
remaining envelope; these estimates are admission planning, not measurements.
Record the actual memory high-water mark and any failure. Do not lower a
reservation to force an otherwise invalid admission.

Run campaign/build sequentially in the producer container. Stop and verify that
container's exact ID before each separate census/serve phase. These stages may
share their admitted action but must not overlap another GPU measurement on
the same device. Both GB10 hosts can execute independently admitted work when
images, snapshots, and local mount dependencies are satisfied.

Keep outputs at a fresh task-owned absolute local directory under
`/home/rob/tessera-runs/measurement-208-183-2026-09-05/`; mount it at the same
absolute path in the producer and serving containers. NFS parents with mode
0700 can prevent Docker root traversal, so archive only after safe cleanup.
Use PB's Docker shim throughout. Do not set ownership/admission variables by
hand. Source snapshots must resolve the external Tessera dependency explicitly;
scope any removal/restoration of absolute calibration symlinks to PB sealing.
The outer wrapper must verify the actual Tessera source snapshot against its
sealed file manifest from the full pinned commit before each container launch.
The driver's exact `--tessera-commit` equality rejects a wrong declaration but
does not establish the identity of an arbitrary source directory by itself.
The generated census and both smoke arms explicitly use GPU-memory fraction
0.35, inside the 64 GiB aggregate envelope; no image default is relied on.

## Stage commands

The `host` stage is the deterministic entry point for the admitted GPU action.
It executes producer campaign/build, before/after artifact binding, census,
census replay, the existing paired smoke, and final producer verification in
that fixed order. It verifies the external Tessera source before every phase,
records both-host Netdata/power/CPU/memory, checks each container's PB ownership
label and CPU/memory envelope, and cleans up exact container IDs on every exit.
Timeouts and any failed phase leave an inconclusive receipt and a nonzero exit.

```bash
"$PY" "$PQ/experiments/pq183_lfm_bound.py" host \
  --model /mnt/shared/models/LFM2.5-8B-A1B-BF16 --out "$OUT" \
  --tessera-repo "$TS" \
  --tessera-commit ba582d476a3b6db9057ebd1385dc52926f171451 \
  --tessera-source-manifest "$TS_MANIFEST" \
  --tessera-source-manifest-sha256 "$TS_MANIFEST_SHA256" \
  --producer-image sha256:79cb5c9a8cd696f30cb0d8b5803d67d65906de4df91741c9811f3de088a13846 \
  --seconds 7200 --netdata-url http://sparky:19999 --netdata-url http://sparklina:19999
```

That image ID is the transferred image's local ID on sparklina. The runner
requires an explicit immutable local ID and verifies the inspected canonical
JSON hashes of `Config` and `RootFS` against, respectively,
`83d0dcabcd3b6d259e9dea48bb67b5bf36108e22d03a7abb2209d73a2adc9e53` and
`d97ec6de925255c82642f99bc250a3e5a554002583b276aa8eacfd15166c7592`.
This handles the older Docker engine reporting a config-image ID after
save/load while sparky reports an OCI-index identity, without substituting a
different numerical environment. Both complete image inspections are retained.

`TS_MANIFEST` is a PB-sealed input outside the Tessera directory, with exactly
`schema: "prismaquant.pq183-tessera-source.v1"`, `commit` equal to the full pin,
and `files` mapping every relative source file to SHA-256. Build that manifest
from the clean pinned checkout. The runner verifies its supplied SHA-256 and
the complete actual source file roster (excluding `.git` only); symlinks,
changed bytes, extra files and missing files refuse before launch. Source is
mounted read-only, at its existing absolute path. The source seal is checked
again before every subsequent phase.

Set PB's action timeout at least 120 seconds beyond `--seconds` for cleanup.
The driver is a child inside admission and never resubmits itself. Its host
interpreter must provide `requests`, `tokenizers` and `numpy`; that preflight
runs before GPU containers. The producer campaign retains an in-process
`cProfile` artifact at `campaign.pstats` for Python/capture/encode attribution;
this is not CUDA kernel timing or a performance comparison.

The following are **commands inside admitted actions**, not permission to run
on a coordinator GPU. Set `PQ`, `TS`, and `OUT` to the mounted sealed snapshots
and the new local attempt; set `PY` to the producer container's Python.

```bash
"$PY" "$PQ/experiments/pq183_lfm_bound.py" campaign \
  --model /mnt/shared/models/LFM2.5-8B-A1B-BF16 --out "$OUT" \
  --tessera-repo "$TS" \
  --tessera-commit ba582d476a3b6db9057ebd1385dc52926f171451
"$PY" "$PQ/experiments/pq183_lfm_bound.py" build \
  --model /mnt/shared/models/LFM2.5-8B-A1B-BF16 --out "$OUT" \
  --tessera-repo "$TS" \
  --tessera-commit ba582d476a3b6db9057ebd1385dc52926f171451
```

The `commands` stage with those same arguments prints argv arrays for the
existing producer's `tessera_plugin_run.sh` route census, its
`ts5_census_check.py --require-attested`, and `moe_greedy_smoke_pair.sh` on the
BF16 source and **this** exported artifact/seal. Run that stage using the host
interpreter chosen for the serve instruments, because it records that exact
interpreter in the generated commands. It emits required directories and the
three exact owned container names; use a distinct `--container-prefix` and
available `--port` per attempt. Refuse preexisting names before either launcher
can remove one. Capture container IDs immediately after launch, and verify
cleanup by exact ID on every exit, including timeout and signal paths.

The older `ts5_lfm_served_bound.py` has hardcoded campaign paths and is not used
here. No teacher/source/artifact identity is borrowed from an unrelated run.
The smoke script uses its packaged prompt corpus for both arms. This smoke
does not produce KL or establish held-out quality.

After census and smoke finish, run the driver's `check` stage in the producer
environment with the same arguments. It repeats the actual-byte audit,
compares the complete artifact identity to the seal, requires an attested
census, validates both served build identities, and re-derives the smoke status
through the pinned producer's contract. An observed negative smoke result is
written to `artifact-after.json` before failing; do not promote it to success.

## Acceptance and measurement evidence

1. Campaign exit zero; 96 named real measured expert rows and exact actual
   population/omission records. Missing expert activation support or rows is a
   failure, not a synthetic fill.
2. Export preflight and translator/exporter exit zero; `build.json` names the
   cached-expert manifest; all 96 exported fused members match their priced
   cache bytes and SHA-256 receipts. Extra, duplicate, missing, wrongly framed,
   or changed wires fail. Sidecar census alone is insufficient for this check.
3. Seal the actual full exported checkpoint before serving; compare that seal
   before and after each serving phase. Run the exact-image resident/eager
   census and replay its declared routes with `--require-attested`.
4. Both source and selected artifact complete the producer's paired greedy
   smoke on the same corpus. Retain raw outputs, build identities, logs and
   the re-derived status; `recorded` is only this bounded smoke observation.
5. Collect both-host Netdata over the entire interval, GPU power, host CPU,
   memory and resident-work evidence. Obtain an in-process profile where it
   answers a timing question. This experiment makes no speedup or GPU
   saturation claim, so no unmeasured before/after delta is reported. Any later
   performance claim needs the same workload/environment and before/after
   profiler plus host telemetry; GPU utilization alone is non-diagnostic.
6. Retain actual PB terminal records, action keys, exit statuses, logs and CAS
   receipts, final source identities and exact-container cleanup evidence.
   A stage wrapper or submission acknowledgement is not acceptance.

The initial stage driver passed stdlib syntax, CLI, argv generation, explicit
GPU-memory fraction, artifact-seal path and argv-roundtrip checks through PB
action `0af1c6b9423d029a53f55f80e422b6b34de745d013c9baceffdd01b99d562a30`
(sparky, exit 0, no skips, no GPU launch). The exact-pin rejection update passed
portable PB action
`19fc6aec105296ff5dcc1fd6ce061a4f56dcbc5eedea8d6d8105c95e6b080cc4`
(dl380g10, exit 0, no skips, no GPU launch). Terminal records and CAS payloads
were inspected. Final host-runner CPU checks and the real GPU observation are
recorded separately when completed; neither initial check establishes them.

The completed host-runner CPU check is PB action
`df564370757104cc3da15981a6ef1be2c70f1d8e6bc3cb3fe476e991f52f9883`:
dl380g10, exit 0, 1.18 seconds, 50,421,760-byte peak, no skips and no GPU
launch. It checks syntax/host CLI, the fixed serving argv and resource flags,
short/wrong pin refusals, source content/roster/missing-file/symlink/manifest
mutations, and changed producer Config/local-ID refusal. Its terminal record,
stdout and actual CAS payload were inspected. Receipt SHA-256 is
`4b5f0c4ed621bdad1382446c6f95149278ea8230f81a9f5526d0db9c478f0c7e`;
payload is
`/mnt/shared/prismabuild-fleet/cas/blobs/25/25a69c17288f1b136bb3bbf565e34fbfc8fdbdea8823da651ed5b79a1442548d`.
The three absolute calibration symlinks were removed only while each PB input
was sealed and restored immediately afterward. GPU execution is still pending
the coordinator's combined-source admission.
