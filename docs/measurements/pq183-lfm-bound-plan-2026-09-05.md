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

## Producer dependency preflight after the first actual attempt

The first actual attempt reached model construction before discovering that
the producer image lacked `datasets`: the import lives inside the campaign's
`_calibration_tokens`, after model loading. The opt-in driver now checks the
actual calibration/campaign/export dependency APIs before calling the campaign
entry point. Its `producer-dependencies.json` records module source paths,
available versions, required APIs and every failure. Missing dependencies or
APIs raise before checkpoint construction; no alternate calibration source,
synthetic tokens or numerical fallback is introduced.

`producer-preflight` is also an independent CLI stage with the usual model,
output and pinned Tessera arguments. It imports and checks dependencies only;
it does not load a checkpoint or sample the corpus. Use a fresh output directory
for that check. The real campaign always performs it again, rather than trusting
an earlier image's receipt. A successful check does not prove corpus download
or tokenization; the derivative-image qualification separately runs the existing
`_calibration_tokens` with the same 8 samples, sequence length 512 and seed zero.

The ordering regression passed via CPU PB action
`2117a64c17012a82085cd35b364d4000df89fa91bfb9487d145c0199f7d19360`
(sparky, exit 0, 0.50 seconds, no skips, no GPU). Dependency fixtures demonstrate
that missing `datasets` and a missing `load_dataset` API both prevent the
campaign/model stage, while the available-dependency fixture reaches that stage
only after every preflight import. Fixtures are explicitly not installed-package
qualification. Terminal stdout and actual CAS payload
`/mnt/shared/prismabuild-fleet/cas/blobs/cd/cdd06c484b6da83bca2713341f99d23ec2682825b8756fcb4d01d1bba281ddcf`
were inspected (receipt
`6b68923318bccfdb6a8d4811cfe660bda9bcc1e022453f0473ee100cf8eb9411`).
The existing producer image identity constants remain unchanged in this fix;
any derivative image must supply separately measured identity evidence before
the next GPU admission.

The final source syntax and successful standalone preflight return status also
passed CPU PB action
`cf9c2a2d70521c3011c58c312cc7dd2a18b004cd39571807a3f944fc54429093`
(sparky, exit 0, no skips, dependency fixtures, no GPU). Its terminal/CAS payload
was inspected at
`/mnt/shared/prismabuild-fleet/cas/blobs/5a/5a76cd9d7c45e6e695f2f3d624dbb41bb5bfaf93ef4fc9793026f00c930823cc`.

## Qualified execution plan after dependency repair

The producer image values in the initial plan above are superseded for the next
actual attempt by the qualified derivative
`sha256:47dd0e9aaa4e7a6575d21cfc661d96a47c0e35e87c64e850631e210bdf04ebc0`.
Pass that exact value to `--producer-image`. Its canonical Config hash is
`fda47b55fb7105c93e8a0bf99cd633191c198e4033719957734d065a635de31e`;
RootFS hash is
`df0f8207331bd466df86322a178e14501f707f7b765e820a60e7ce9f28d51d71`.
The host runner verifies these values before launching the producer.

The image contains the calibrated corpus cache at `/opt/pq183-hf-cache`. Both
producer phases explicitly set that `HF_HOME`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`; they keep fresh task output for weights, activations,
compiler products, and receipts. The existing calibration implementation and
draw are unchanged. Offline Docker qualification with networking disabled
matched the online corpus SHA and all eight token-sample SHAs exactly.

Relative to the original image, existing effective package versions changed
only for fsspec (2026.7.0 to 2026.6.0) and huggingface-hub (1.28.0 to 1.10.2).
Added packages are datasets5.0.1, multiprocess0.70.19, pandas3.0.5, pyarrow25.0.1,
and xxhash4.0.1. Torch2.13.0+cu130, Transformers5.15.1, NumPy2.2.6,
safetensors0.8.0, accelerate1.14.0, and the existing CUDA/native packages are
unchanged. The coordinator independently compared effective inventories,
image components, online/offline sample hashes, and the actual successful
PrismaBuild CAS result.

See `pq183-lfm-bound-results-2026-09-05.md` for the failed first GPU attempt,
negative environment qualifications, final image recipe, and exact receipts.
The next source combines merged main90f37e00 (including #230/#242/#243/#245)
with this opt-in runner and dependency preflight; its actual full commit and
PB snapshot will be recorded at admission. Complete GPU campaign/export/serve
acceptance remains pending until those outputs exist.

## Explicit calibration coverage revision for attempt 04

Attempt 03 completed real calibration forwards with the repaired parent-owned
router bias, but expert 2 received no rows in the eight-sequence draw. Its
three projections therefore had no empirical Hessian, and the campaign refused
before producing costs. The full negative observation is retained in the results
document. This is calibration coverage setup; no eight-sample quality, cost, or
serving observation exists to select candidates against.

The next attempt explicitly uses **32** WikiText `wikitext-2-raw-v1` train
sequences, length 512, seed zero, with the existing maximum 512 stored activation
rows per unit. It reuses the same complete corpus in the qualified offline image
and the existing sampling implementation; the first eight draws are unchanged.
All 96 layer-12 expert projections remain in scope, all other declared BF16
dispositions remain explicit, and the no-routed-rows refusal remains enforced.
This is a revised calibration contract, not a claim that the eight-sample
campaign passed. It uses a fresh production cache and output directory,
`run183-04`, with the same fixed E4M3/q1024 candidate and serving topology.

## Frozen scope amendment for attempt 05

The 32-sequence layer-12 attempt still lacked support for expert 2. An actual
forward-hook diagnostic under the same draw observed all 22 sparse layers,
recording only selected-expert counts and actual parent bias vectors. Before
inspecting its results, the scope criterion was fixed to the lowest layer index
at least 13 for which every expert received a calibration row. Eligible layers
were 13, 17, 20, 22, and 23, so **layer 13** is selected before any quantization.

The campaign now uses stride 13 and exactly
`model.layers.13.feed_forward.experts.{0..31}.{w1,w3,w2}`: still one complete
96-projection stack plus dense layer 0. All 32 samples, corpus/train split,
512-token length, seed zero, 512 stored-row cap, required Hessian, fixed
E4M3/q1024 candidate, explicit BF16 dispositions, image and serving topology
remain unchanged. The recipe metadata binds the diagnostic histogram SHA-256
`d8ab6d0816a53a715596c0f4ff2ab28cf2895270668822aa93143d96819c1fc3`.
The minimum expert support is 13 routed rows; nonzero support for all experts
does not mean full-rank Hessians or certified quantization quality. Existing
no-row and Hessian gates remain unchanged. Attempt 05 uses fresh `run183-05`
output and production cache; all earlier observations remain in the results doc.

## Frozen campaign continuation after attempt 05

Attempt 05 completed the declared 96-unit campaign and wrote its fixed
assignment before export preflight refused the shared string-format parser.
The opt-in continuation preserves those exact cost, assignment, recipe, Hessian
and priced wire bytes. It performs no campaign, allocation or quantization.
The parser repair must be integrated into the fresh source snapshot first.

The `seal-campaign` stage, executed through PB as offline input preparation,
writes an externally pinned SHA-256 manifest outside the original run. Its
106-file consumed roster comprises cost/assignment/recipe, five prior execution
and image receipts, the Hessian capture and provenance, and 96 priced wires.
It requires the successful campaign receipt, complete measured scope, unchanged
calibration/runtime identities and safe prior container cleanup. Unused static
input scales and rendered-cache copies are excluded; the E4M3 exporter does
not consume them. The original run remains mounted read-only at its original
absolute path, preserving paths already embedded in the priced receipts.

Add `--campaign-input /absolute/run183-05`,
`--campaign-input-manifest /absolute/run183-05-inputs.json` and
`--campaign-input-manifest-sha256 <sealed-sha256>` to the existing `host`
invocation, with a fresh `--out`. The deterministic remaining phase graph is
shared assignment/Hessian/source preflight, cached export, artifact seal,
census and its gate, paired serving smoke, and final byte/output check. Existing
deadline, Netdata, runtime/source identity and exact-container cleanup remain
in force; PB retains sole responsibility for admission and placement.

Preflight uses the unchanged copied assignment and reads original H/wires.
Only after that gate passes, the continuation copies the validated 96 wire
blobs into a fresh transport bundle and calls the existing cached-unit writer.
Build serialization matches the shared CLI's sorted, indented JSON plus newline;
its digest comes from the bytes written by this action. The original cost and
assignment are copied byte-for-byte; old campaign and new continuation snapshot
heads are recorded together. Every phase checks the frozen input seal before
and after execution. Fresh outputs retain failures without overwriting the
original run or requiring an agent between phases.

`tests/test_pq183_continuation.py` supplies portable, offline stdlib fixtures for
this transport and refusal contract. Fixture success does not qualify actual
Hessian numerics, exporter output, or serving behavior; those gates run in the
admitted continuation with the actual image, model and sealed campaign.

Attempt 06 passed the actual shared preflight and constructed the validated
96-wire bundle, then failed while querying Git inside the producer image.
The continuation now consumes the existing host source receipt: the host
resolves the admitted snapshot before container launch, and the producer
requires its schema, full snapshot ID, prior campaign snapshot and phase input
manifest binding before any copied output or export. Missing or malformed
identity refuses; the producer image and numerical environment stay unchanged.
