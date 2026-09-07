# Runtime provenance relation checks — 2026-09-07

Scope: CPU contract and retained-artifact intake checks for PrismaQuant #323.
No GPU timings, full-model fixed resources, allocation table, served result or
performance improvement is claimed.

The first portable submission used `/home/rob/venvs/pb-cpu/bin/python` directly.
On Sparklina that interpreter lacks Torch: actions `2c9929899fa7` and
`f1469b99db32` ended with collection errors (2 and 4 files respectively). These
are not test passes. The portable helper now selects the existing CPU container
on ARM and the scoped Transformers/dependency environment with CPU Torch on
x86, following the existing anchor-bridge helper. No tests bypassed PrismaBuild.

Action `fbe94f5eaa16` exercised 167 tests successfully but reported 10 fixture
setup errors: the new synthetic native-intake fixture omitted the existing
native execution fields. Restoring that original execution contract fixed the
fixture. Action `f311a6e18544` then passed all 179 selected contracts on Sparklina
in CPU-only mode, 4 reserved CPUs and 12 GiB host memory. Its terminal record is
`/mnt/shared/prismabuild-fleet/pb-queue/done/f311a6e18544dfd2389f939f814e357e1c8c2ed17cb6d9f2f95dbfdec96a082a.json`.

After adding archive-backed package identity checks and the documentation,
action `08e7dd440898` passed 203 tests on DL380 G10: Python 3.14.4,
Torch 2.11.0+cpu, Transformers 5.16.1, pytest 9.0.2; four xdist processes,
12 GiB reservation, native thread limits of one. There were no skips and 56
upstream Torch deprecation warnings. Terminal record:
`/mnt/shared/prismabuild-fleet/pb-queue/done/08e7dd440898fcaeb6dd8888e6718fda9b53a46d49a6880470bff2c2bdc8aebc.json`.
CAS receipt SHA-256:
`f73cba83c5a28e2a88476231d1d59e1e2e4ed368c6610393467d3037ac4d73f4`.

The raw image manifest was read through `docker buildx imagetools inspect --raw`
without launching a container. Its bytes hash exactly to the pinned image
manifest digest and link to the other host's config ID. Original image and
container inspections remain unchanged. The declared native and engine source,
binary and build artifacts are retained beneath
`/mnt/shared/tessera-native376-resource/runtime-provenance-323/`; the engine
worker source was recovered from its exact PB checkout bundle `5ae7e9f0...`,
commit `9b5e51ef...`, and matches observed source SHA-256 `dbbf2f88...`.

The actual prepare-05/engine-r4 pair is deliberately unadmitted: different GPU
UUIDs and a missing exact generated-library dependency relation. The fixed
resource admission gate continues to reject every available raw/incomplete
producer ledger. See the [contract and retained evidence](../design/runtime_provenance_relation.md).

After rebasing onto main `21288248` and completing the wire-source,
GPU-capability and graph-mode checks, action `2981e3fe7b26` passed **208 tests**
with no skips on DL380 G10 (same CPU/dependency environment, 56 upstream Torch
warnings). This includes the actual archive and individual-runtime checks and
the expected cross-device refusal. Terminal record:
`/mnt/shared/prismabuild-fleet/pb-queue/done/2981e3fe7b2601fd3010e09b100989468e96eb87129314bcbc94e58e2ed9214b.json`.
CAS receipt SHA-256:
`9851d2c37671de8f4a868c1de2c0986da19c08aabfd240798856d18feb0a7c0b`.
Result blob SHA-256:
`dadd9abdd68e78f2a5e969bec1eca069d07cbecee70e2e9bff4af78fd7fccc9f`.
The final action reuses the now-integrated `experiments/pq322_cpu_checks.sh`;
the temporary duplicate helper was removed before submission.

Command (portable CPU placement, with PB preserving its assigned affinity):

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-runtime-provenance --anywhere --cpus 4 \
  --demand mem_gb=12 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --detach -- \
  bash experiments/pq322_cpu_checks.sh -n 4 \
  tests/test_runtime_provenance.py tests/test_measured_runtime_prices.py \
  tests/test_native_operator_panel.py tests/test_native_moe_panel.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py \
  experiments/pq323_provenance_artifact_checks.py -q -p no:cacheprovider
```

Terminal exit status, actual logs, CAS receipt and result byte digests were
inspected; the retained index is
`/mnt/shared/tessera-native376-resource/runtime-provenance-323/verified-cpu-actions.json`.

The four touched Python modules/tests also passed an explicit PB `compileall`
action `ec3a9ac8009c` (DL380 G10, one CPU, 1 GiB, exit 0). Terminal:
`/mnt/shared/prismabuild-fleet/pb-queue/done/ec3a9ac8009c597db9a590599519943c62565e4f275b51991936adfd4b7705a8.json`.
Comparing final Git content against the passing suite's PB snapshot
`407c91cc27114601e360ebebfbab6aadcabb3264` showed identical implementation/tests;
only this appended evidence report and PB's own generated closure record differ.
