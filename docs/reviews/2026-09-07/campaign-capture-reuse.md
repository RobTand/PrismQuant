# Canonical campaign capture reuse — 2026-09-07

Issue: https://github.com/RobTand/prismaquant/issues/305

The campaign can capture the whole census once, store exact float32 scoring
prefixes and uncapped Hessians through the existing activation-cache writer,
and reuse selected units in each anchor or materialization quantum. The
existing cost-stage journal seals per-unit file receipts. The manifest is
`prismaquant.tessera_calibration_cache.v2`, with a complete unit roster and
identity bound to canonical checkpoint initialization, actual attention
backend, Torch/CUDA/Transformers runtime, census, complete source checkpoint
bytes, exact calibration draw and per-unit geometry.

The public flow is `dispatch_tessera_campaign capture`, followed by
`plan --calibration-cache <capture_manifest.json>`. Each row carries the exact
manifest SHA256 and verifies selected artifacts before making X/H resident.
Missing, stale or incomplete captures fail closed. Selected-wire materialization
derives its cache and hash from the priced cost provenance and loads the same
explicit attention backend. Merging requires every row to agree on the capture.
The optional packed-collector boundary consumer preserves original positional
and keyword arguments before dtype conversion or sampling, with original
sample/token coordinates from the calibration loop. It does not add a second
forward or alter default reservoir sampling.

CPU and compile validation passed through PrismaBuild action
`b2b5bac37cc59636019d06897ae99f6b945d3a467535f74b80e8d71e89c8beee` on dl380g10:
**176 passed, no skips**, exit 0, 70.58 seconds. Eight pytest workers used the
scoped `pq-cpu312` environment with pinned Tessera producer imports and one
OMP/MKL/OpenBLAS thread each; aggregate demand was eight CPUs and 20 GiB.
Compile checks covered all touched production modules and measurement harnesses.
The tests cover exact X/H precision, selective prefetch, source/draw/scope/
geometry/artifact drift, legacy initialization refusal, complete-journal
resume, CLI reuse without a second forward, automatic materializer binding,
raw argument identity, keyword routes, calibration coordinates, unchanged
reservoir sampling, and native input contracts. Actual inspected CAS payload:
`/mnt/shared/prismabuild-fleet/cas/blobs/46/4605bb918e52c1ed53800a40813b9b852d2500fabba8ee48c2a6726c9e2bd119`.

The canonical full-model census is
`/mnt/shared/tessera-measurements/first-model-20260907/canonical-census-512/census.json`,
SHA256 `62d41825f84edd280de46bbb89893676fae8203563834dc95d8ad29c05bad04d`.
The draw is 512 samples of 512 tokens; its int32 token digest is
`ed77a9890e02ae0b6bac2cbaa8fab4dd27617d7b57635976b689badaf8d5a8e4`.
The prior no-op-initializer capture remains quarantined under
`capture-reuse/baseline/INVALID_FOR_CALIBRATION_REUSE.json`; no quality or
reuse claim derives from those X/H bytes. Canonical full capture and profiled
ABBA reuse measurements are tracked below as separate evidence, not inferred
from the CPU suite.

Initial validation attempts `a8a78fab56ed`, `c30ca448df8a`, and `791a51c9c833`
used the PB infrastructure Python instead of the scoped project Python and
failed on missing Torch or compressed-tensors. The corrected project-runtime
run `3f5a504cf1d3` passed 133 tests and exposed two synthetic producer-hash
fixtures. Those fixtures now declare their actual source hashes; the final
176-test run above passed. No runtime input guard was weakened.

The first canonical GPU attempt `35ab901ece41` stopped before writing a routed
boundary: the native helper called the LFM profile's `name` property as a
method. The owner is fixing the three affected sites with a real-profile
regression. Failed-attempt artifacts are retained as bounded diagnosis under
`capture-reuse/canonical-helper-failed-35ab901ece41`; the worker exit was 1 and
the broker's separate termination receipt confirms its scope stopped. The
current fleet terminal JSON omits that cleanup field, so the separate receipt
was inspected. This attempt provides no completed calibration or reuse result.
