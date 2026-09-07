# Verified single-read anchor preparation

Preparation previously scanned every wire and render before layer residency,
loaded each render again through PWC, then rehashed it before consumption. On
the original fourteen-cell fixture, the complete-path comparison measured
**70 → 28 target-file opens** and **1,439,737,470 → 514,309,808 bytes read**
through the file handles, a 64.3% reduction. Warm median time fell from
3.331945 to 3.102984 seconds (1.0738× throughput; 6.87% less wall time).
All fourteen verification records remained exactly equal to the previously
qualified original oracle; eight corruption cases refused.

This is a bounded preparation result. It does not establish a full-model
speedup, change cost/resume behavior, or require discarding an already qualified
prepared artifact. The original 2,142-unit, seven-candidate census represents
103.03125 GiB of rendered tensor payload, exceeding its 6 GiB layer PWC budget;
a global scan therefore cannot guarantee later layer residency.

## Contract and implementation

The strict importer still hashes payloads by default. Only preparation requests
metadata-only intake: all complete roster, journal, producer receipt, path and
size checks remain. The existing PWC loader reads each serialized donor once,
hashes those bytes, and passes exactly that buffer to `torch.load` with
`weights_only=True`. Read allocation uses the individual observed file size;
preparation refuses a global file maximum above the declared PWC budget before
any layer prefetch. Temporary buffers are bounded per admitted loader worker.

The PWC receipt is valid only for its exact resident tensor object, storage,
version, shape, stride, dtype, device and unchanged donor signature. Eviction
and compaction discard it. No additional rendered-weight cache is introduced.
The existing verifier still checks original wire SHA, source/H/encoding
settings and exact decoded-render equality. Prepared completion waits for the
complete verified roster. Cost/resume retain their strict payload scans.

## Complete-path comparison

Both arms use the same bound source/H identity and the same reader source
`14df443217e2a6a1bc4857532f0b5ead7fa7f5755dbd61e96605659940a8a1bc`.
Source weights and the canonical 512×512 capture are resident on CUDA in both
arms. Each timed arm includes actual metadata intake, payload validation,
a fresh PWC and prefetch, bound identity creation, transfers, verification and
compaction. Original row0000 producer records remain unchanged inside an
explicitly retained bounded metadata projection; its original fleet action is
bound and every imported anchor/record was also checked against the original
journal in a separate CPU action.

| Arm | Render opens / bytes | Wire opens / bytes |
|---|---:|---:|
| Original | 42 / 1,233,260,490 | 28 / 206,476,980 |
| Single read | 14 / 411,071,318 | 14 / 103,238,490 |

A scoped observer counts actual `read`/`readinto` calls made through both
Path/hashlib and torch serialization file adapters; it retains no bytes.
Per-process `/proc` counters are separate: warm `rchar` fell from
1,495,094,870 to 525,585,716. Every warm arm recorded zero physical `read_bytes`.
These measurements therefore establish reduced logical reads and warm timing,
not a cold-storage throughput improvement.

| Pair | Original seconds | Single-read seconds |
|---|---:|---:|
| First parse, retained separately | 12.613666 | 3.143335 |
| Measured 0, AB | 3.332286 | 3.091577 |
| Measured 1, BA | 3.328836 | 3.102984 |
| Measured 2, AB | 3.331945 | 3.106612 |

The first pair includes asymmetric startup/cache state and is not used for a
speedup claim. All ten phases match the original fourteen-record oracle.
The profiler uses the same 50 Hz py-spy wrapper throughout; separate torch
CPU/CUDA traces follow timing. The sampler produced 3,543 samples, zero errors.
Across all optimized-arm phases, including first parse and explicit profiling,
582 of 757 samples include `read_unit_artifact`; remaining decoding work is the
Tessera owner's separate optimization, not part of this change.

PB measurement isolation selected Sparky's GB10 with CPU 4, 12 GiB aggregate
memory and an 8 GiB GPU subset, preserving assigned affinity and one native
thread per worker. The known production image is
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
Raw Netdata from both boxes shows roughly 7–9% aggregate CPU activity and
15–18 W interpolated GPU power in warm phases. Ten-second power sampling cannot
resolve three-second arms, so no work-per-joule ranking is claimed. The scope
reached its 12 GiB memory cap without OOM; the first pair is not a memory- or
storage-isolated timing result.

## Validation and evidence

Test-first PB `62a8600fbba8` exposed the missing receipt/deferred-intake API.
The main gate `7e6c2d7b0b8d` passed 65 tests; `1f5f79f61b62` then demonstrated
an oversized read request inherited from a global bound. The corrected guard,
receipt lifetime and actual torch-read observer gate `6b282fda8f0a` passed 11
tests. Dedicated changed-module compile action `28753a7807a9` passed, and
actual bounded metadata action `c011f178160a` preserved all fourteen original
records. There were no skipped cases. CPU evidence and both red attempts are
retained; successful terminals, cleanup, CAS bytes and receipts were checked.

All evidence below is under
`/mnt/shared/tessera-measurements/first-model-20260907/full-model-joint-aura/`:

- Plan `qualify-file-io-ab-plan-01.json`, SHA
  `2f5587c2de18a200ddc41e9f7fc1fe9c1ff7db8158f4fd000e0b084625636456`.
- `qualify-file-io-ab-01/results.json`, SHA
  `783172f73ecd63109c27db3476711c651bef0e39084e3644a8ccb3c837e14b40`.
- `qualify-file-io-ab-01/verified-evidence.json`, SHA
  `df03fa2fe55f056901e6ba84b07931d87fc0a8f7a4dc770f7e348d9e850ee814`,
  seals results, the bounded metadata projection, profiles and both-host Netdata.
- `qualify-file-io-ab-01/profile-00-{before,after}.{trace.json,operators.txt}`,
  `profile-observations.json`, and `netdata-both-hosts.json`.
- `qualify-file-io-ab-sampler-01/` holds raw stacks and independent child/sampler
  exit receipts; both exit statuses were zero.
- `prepare-single-read-cpu-receipts.json` and
  `prepare-single-read-intake-cpu-receipt.json` retain CPU evidence.

GPU action `53fca49df8d345b8ffec4816ad9028cff6e922972a4d6eee9439cde4e7579cb2`
finished with exit zero and complete cleanup. Its 80,496-byte CAS payload SHA is
`3aa8b8edac611bb6ade094d9715df140fcabfc92aafa4f01f171f22ecd1b9694`, and receipt SHA is
`97e8f96070b0b9b35a51e33bf923cfdc635623a4a97b2f8ad2f10c52b0f91f39`.
Both were independently rehashed.

Reproduce from source `3ac1d094`, with a fresh output directory in a newly
hashed plan, through PB on the measuring GB10:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd CHECKOUT --cpus 4 --demand mem_gb=12 --gpu --gpu-memory-gb 8 \
  --measurement --detach -- bash experiments/pq322_anchor_file_io_ab.sh \
  SAMPLER_OUTPUT --plan PLAN --plan-sha256 PLAN_SHA256
```
