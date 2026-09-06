# Tessera campaign population coverage, 2026-09-05

PrismaQuant #183's actual-model preparation found that campaign targets were
reported as priced even when no admitted menu or successful anchor produced
a cost row. At PrismaQuant `729972255a71a8cba4b99a7213e8af29e7e311e5`,
`tessera_campaign.main()` passed its target lists directly into
`provenance.population.priced` before constructing the emitted cost table.

Population v2 now separates enumerated targets, units with emitted prices,
in-scope unpriced units and stride/profile omissions. A packed stack counts
as completely priced only when every projected member has rows. Allocation
requires explicit BF16 retention for unpriced units, verifies emitted-row
agreement and disjoint dispositions, and refuses an enumerated unit whose
disposition disappeared. Legacy v1 receipts remain readable.

## CPU regression evidence

All execution used PrismaBuild on `dl380g10`, Python 3.14.4, Torch
2.11.0+cpu, two pytest workers, and OMP/MKL/OpenBLAS threads set to one.
CUDA visibility was empty. A test-only snapshot supplied Tessera
`ba582d476a3b6db9057ebd1385dc52926f171451`; action-scoped dependencies used
Transformers 5.6.0, Accelerate 1.12.0 and the known-good pure-Python
compressed-tensors 0.15.1a20260414 sources. Dependency inventories and
per-file hashes are retained with the submission inputs.

| PrismaBuild action | Result | Scope |
|---|---|---|
| `110a3a962e3bbbfb35707e9bb3d693c6941bc135e999ea9dc216f7e2df7a74d9` | 7 expected failures, exit 1 | New mixed-main empty-menu/failed-anchor/failed-expert cases and missing/quantized/BF16-retention cases against the original source modules |
| `6ad031429a0183f0236f737d54e9dc718c2e13c81b3a95ec7dc14eaf0d376a23` | 137 passed, 5 skipped, 7 failures | Campaign, packed campaign, resume, allocator projection, export projection and architecture/docs files |
| `c2b36821a29d15820be4b1aaefabd03b6b5b12417ab11fca158440a7d782c690` | 61 passed, exit 0 | Final allocator/export/docs files, touched-module compile checks and the two remaining failed resume/dependency cases |

The broader run's seven failures were resolved: five export CLI cases had
an incomplete synthetic fixture that did not isolate the producer-repository
pin gate when `TESSERA_REPO` was set; one resume fixture discarded the
payload's provenance; one subprocess deliberately reset `PYTHONPATH` and
therefore lost the staged compressed-tensors dependency. The pin isolation
is a separate test-only fix. The dependency was made visible through the
action-scoped environment's site path. Production pin gates are unchanged.
The five skips explicitly require CUDA; this CPU screen does not qualify
GPU scoring, export or serving.

Red peak memory was 9,960,148,992 bytes under a 24-GiB reservation;
the broader screen peaked at 10,112,835,584 bytes. The final follow-up
reserved 8 GiB and peaked at 1,480,130,560 bytes. An earlier 6-GiB attempt
(`d9fafc6ae5529100d8a63768fd0a0229d44476e835bbc7b59dde587d6bff4c8b`)
hit its exact cgroup cap and exited 137; it is an undersized attempt,
not regression evidence. Earlier collection/setup failures are likewise
not counted as red tests.

The final CAS receipt is
`da8fd946d3d10e49b3a7162543ab6f554e8ed110267e8781ae0fdde57a339862`.
Its 11,948-byte payload was read and independently SHA-256 checked as
`e522618fd2e5b99804178f627fd3745299088f0b3081754740a455e4ca63513e`.
Terminal records, full logs and receipt JSON are retained under
`/home/rob/tessera-runs/measurement-208-183-2026-09-05/pq183-evidence/`.
The admitted command is recorded in each terminal's source request; its
common submission shape was:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd CHECKOUT --anywhere --cpus 2 --demand mem_gb=8 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --timeout-s 600 --detach -- \
  bash pb183-inputs/cpu-tests.sh consumers
```

## Actual-model boundary

The pinned contract admits routed E4M3/q1024 on the EUGR image and dense
cells on a different stock vLLM image. A bounded LFM2.5 campaign may
therefore price its selected routed stack while retaining unpriced dense
units explicitly at BF16. The shared model and reference ts5 directories
were inspected read-only; source-derived scope for layer stride 12 is three
dense FFN units in layer 0 and 96 expert projections in layer 12. A real
producer projection must verify that population during the admitted run.
No actual-model, GPU or served measurement is claimed by this correction;
#183 acceptance item 4 remains a separate measurement obligation.

## Integration on the delivered main branch

The final code was integrated with merged PRs #230 and #242 (main
`015b67b19f21db2497f2fedc13c2777a4ac4676f`). PrismaBuild action
`97858e8ae052447622994737c1b5f53506040d945bb86019e533aee42f4d7ce1` tested
source `379e2f21c99e265771b3c6febf9ad72ae80f1ff5` on sparklina with CUDA
hidden, four pytest workers, native threads bounded to one, and a 24 GiB
reservation. The six files covering campaign population, allocator projection,
export projection, activation policy, architecture, and documentation staleness
passed: **95 passed, zero skips**, 100.69 seconds. Measured action memory peak
was 16,034,717,696 bytes; there were no OOM events.

The coordinator inspected the terminal exit status and independently hashed the
actual 743-byte result payload
`9faf232c5b24153dba979b2aceffe73e0af3a0695d8e68321c55fce5f96f1f96`;
receipt SHA-256 is
`1034b506db5e8e6a4dd3609284de0c425ab7aa8a41fa33cfe45da636b14f94fd`.
Submission, terminal, receipt, and result are retained under
`/home/rob/tessera-runs/measurement-208-183-2026-09-05/coverage-integration-*`.
Only this evidence paragraph was added after that code validation. Actual GPU
campaign/export/serve measurement for #183 remains separate and pending.
