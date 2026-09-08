# Joint-AURA allocator identity reuse — 2026-09-08

The same 114-budget allocation completed in **770.654 seconds before and
148.395 seconds after**: **5.193× faster, 80.744% less wall time under cProfile**.
The probe was not repeated. All assignments, prices, bpp, candidate metadata,
frontier CSV and knee results matched. Forty output files were byte-identical;
the terminal layer config differed only in its 114 measured `solver_seconds`
values. Excluding exactly those fields, all 41 files matched.

## Workload and implementation

LFM2.5-8B-A1B; the completed 2,142-unit joint table, seven measured Tessera
formats plus BF16; the original 114 budgets, 0.0001 bit precision and cost
settings. Input SHA256:
`5d08904df98d07d91fa29cffe9d5bb44080e63f8149037b1f29153b6457bd451`.
Baseline source: `3264f85828f90f2cb4321bc47e0bc1a5604f7e84`.
Measured implementation: `77d3938075481a84f615b052d0182aaa946bc9d3`.

The allocator owns immutable copies of the loaded table's shared probe
identities. Each distinct source identity is validated when copied. Its digest
and validated model identity can then be reused, while every row consumption
still validates operator identity, digest binding, samples, scalar values and
currency. Nested values are returned as detached copies. Pickle restores normal
dictionaries, requiring fresh validation on admission. The original source
object is checked before JSON normalization, preserving strict type rejection.
No format, calibration, numerical objective, artifact schema or serving gate
changes. This extends CPU post-probe allocation; it adds no GPU or weight cache.

## Profile evidence

Both arms ran consecutively in one PrismaBuild measurement on DL380, with one
preferred physical CPU, 8 GiB reserved, Python 3.12, native threads bounded to
one, the original producer package and identical input/argv/environment apart
from output locations. Both arms used cProfile with local profile output before
archiving to shared storage. This is one profiled pair, not an estimate of
unprofiled throughput or a multi-run confidence interval.

| Work | Before calls | After calls | Before cumulative s | After cumulative s |
|---|---:|---:|---:|---:|
| Joint row validation | 79,554 | 79,554 | 656.323 | 21.173 |
| Streamed model identity validation | 79,554 | 1,261 | 495.838 | 8.105 |
| Owned identity preparation | — | 1 | — | 14.138 |
| Artifact-size calculation | 114 | 114 | 57.915 | 57.977 |
| Core allocation solves | 114 | 114 | 2.485 | 2.373 |

Cumulative rows overlap. Total in-process profile time fell from 769.374 to
147.102 seconds. The new identity preparation cost includes its model checks.
Artifact-size calculation is now the largest remaining component; it was not
changed by this fix.

Live Netdata CPU/RAM/I/O/load and Spark GPU-power samples cover both arms on all
three hosts: 155 samples per host before and 30 after, with zero sampling errors.
DL380 whole-host CPU busy averaged 3.869% before and 4.749% after. PB's combined
scope recorded 921.058 CPU-seconds, peak memory 2,971,553,792 bytes and complete
cleanup. Spark telemetry includes unrelated work; this CPU-only experiment makes
no GPU saturation, energy-efficiency or serving-speed claim.

## Validation and receipts

The initial six feature regressions failed before implementation. A further
source-type regression reproduced normalization bypass, then passed after the
constructor preserved the original admission check. The integrated targeted
suite passed **112 tests, zero skips**, across seven PB shards. Touched Python
modules and the measurement helper passed compile checks through PB. The final
source was integrated with main in `be92f57b31`; the allocation and row-validator
changes match the measured implementation.

Measurement action:
`8eed82abb3dad3bc65dcd79b974c1dc0af4df7dd38e76eceef63a809a6e56bbf`.
Both allocator children exited zero. The original wrapper exited one because
it incorrectly required measured solver timings to match. That failed terminal
record is retained. No timing run was repeated to repair the comparator.

A separate admitted audit checked the original artifact hashes, required both
successful children and complete telemetry, and excluded only the exact 114
`__prismaquant__.solve_diagnostics.<target>.solver_seconds` fields:
`50c7d6c0ea3144a64417346ced3d7b0bddd0cff318119f7861e60c9d4710ccf9`.
It exited zero with complete cleanup. Its actual CAS payload binds audit SHA256
`4a271921c291fa12b74adb803406691560e14b1b13d27750b14ff941e199b4c0`.
Root independently checked all changed leaves, canonical successful receipts,
actual CAS payload hashes, source bundles against their Git parents, and every
artifact hash named by the final audit.

The first cross-host measurement submission failed preflight because it carried
the Spark's accelerator identity to DL380; no payload ran. A shallow submission
clone was also refused before submission. The final pair was submitted from
DL380 using a complete clone and the correct measurement identity.

Evidence root:
`/mnt/shared/tessera-measurements/first-model-20260907/allocation-preparation-01/validation-pair-01/`.
It contains both profiles/logs/invocations/telemetry, `comparison.json` (original
failed comparison), `difference-leaves.json`, `comparison-audited.json`,
`performance-summary.json`, and `root-final-audit.json` with exact receipt and
source hashes. Integrated test manifest:
`/mnt/shared/tessera-measurements/joint-validation-integrated-tests-20260908.json`.

Reproduction uses the checked-in `tools/benchmarks/joint_validation_pair.py`
through PB `--measurement --cpus 1 --demand mem_gb=8`, on the same DL380 runtime,
with `--invocation` pointing to the sibling `frontier-invocation-01.json` and a
new `--output` directory. `--audit-existing` audits retained arms without running
allocation again.

The existing non-BF16 terminal assignment and conservative fused-group fallback
are unchanged and unresolved. These outputs remain research evidence, not
shipping candidates; this result does not establish GLM capture or quant
shipping completion.
