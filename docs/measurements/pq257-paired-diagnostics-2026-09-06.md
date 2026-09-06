# Paired AURA diagnostic validation, 2026-09-06

Scope: #257, diagnostic groundwork for #237. These are CPU scalar and
small-module regression results, not model-quality or serving measurements.
The additive unary objective and quadratic of summed signed projections
remain explicitly distinct. Probe SE is conditional on fixed calibration;
held-out sequence uncertainty is not inferred from it.

Implementation started from main `6f3044019a7f64c027d7a3f15bee53d20b28db80`
in an isolated no-local clone. All execution used the published PrismaBuild
tools. PB selected placement and file shards; no manual host test jobs ran.
The successful CPU actions used the existing
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python` dependency environment,
tag `gb10`, CUDA hidden, one CPU and 4 GiB per test shard, native threads
bounded to one. The first x86 attempt failed collection because its `pb-cpu`
environment lacked `compressed_tensors`; that was not counted as a regression.

The snapshotter initially refused three unused tracked calibration symlinks
whose targets escape the repository. They were temporarily moved aside in
the isolated clone for all test snapshots, then restored before commit.
Synthetic inputs only were used. No calibration data, production cache,
allocator assignment, or serving artifact was generated or changed.

| Phase | Actual result | PB action(s) |
|---|---|---|
| Initial RED | 37 failed: five existing gate regressions and 32 absent-API cases; pytest exit 1 | `ff4b4d8d6a68` |
| First focused GREEN | 100 passed, no skips, six exit-0 shards | `7466c541786d`, `f00a731729b4`, `ac82b5f6109e`, `bbe0c94bf877`, `27da068611b7`, `ec8d51c615e9` |
| Review RED | Six intended failures: canonical JSON identity and small residual cancellation; 33 deselected, pytest exit 1 | `9c8673e2ede7` |
| Final affected-file GREEN | 66 passed, no skips, three exit-0 shards | `f0458ad5006e`, `0cfb8e52415e`, `ca9e8a9350d8` |
| Final compile | Both source modules compiled, exit 0 | `3ae6862321db` |

The first green covered `test_aura_additivity_identity.py`,
`test_joint_aura_assignment_diagnostics.py`, `test_aura_cost.py`,
`test_joint_aura_streamed.py`, `test_architecture_doc.py` and
`test_docs_staleness.py`. The final green repeated the two new diagnostic
files and streamed integration file after the review fixes; the other 40
checks were unaffected. Runs emitted existing torch JIT deprecation warnings.
Terminal records, actual stdout, resource cleanup and CAS result hashes were
checked; a submission acknowledgement was not counted as a pass.

Exact commands, logs, full action keys and CAS receipts are retained at
`/home/rob/tessera-runs/pq237-paired-diagnostics-receipts-2026-09-06/` in
`commands.md`, `red.log`, `red-receipt.json`, `review-red.log`,
`green-results.json` and `final-green-results.json`. PB terminal records are
under `/mnt/shared/prismabuild-fleet/pb-queue/{done,failed}/` by full action key.

The mathematical tests exercise one-unit reduction, common-probe covariance,
cross-unit cancellation, unchanged-background cross terms, large cancelling
terms, full rosters, and mismatched probe/source/calibration/currency/operator
identities. The report CLI consumes complete rows and preserves separate
probe and supplied sequence standard errors. Historical bare-list screens
remain unverified by this identity contract. These results do not qualify
background-dependent unary ordering, sparse-anchor generalization, runtime
admission, or positive quality gains; #237 remains open.

## Integration review, 2026-09-06

Rebased onto main `3cb0d35933c614c61df79be25d478ebf7d43e859`, retaining
the preparation and row-count provenance stamps. Review found the existing
gate accepted NaN/infinite measured KL and negative/nonfinite supplied standard
errors. Five new cases failed before the guard; the finite negative sample-KL
control passed. RED action
`3849fc643867792bc014b89de2b248bf773404e5136c73c7ab3a4db28fba226c`
returned 1: five failed, one passed, 13 deselected.

After finite-domain validation and overflow-resistant `math.hypot` combination,
`tests/test_aura_additivity_identity.py` and `tests/test_aura_cost.py` passed
40 tests, no skips, in 6.52 seconds using two xdist workers, two CPUs and
4 GiB total on PB-selected Sparky, CUDA hidden and native threads bounded to one.
GREEN action
`3b99832b56d056196d0d01f109f9f3072a9dc0e4af686d6e0d20eaf2b76146a5`
returned 0 with complete scope cleanup and no OOM. Receipt SHA256
`402bbfe73506fbed0aa0cbf5abbda56079f0a6f60fccbf68988589c90cae83c7`
binds the actual 651-byte test log, independently read and rehashed as
`35a2267abef3a135aa893f629b640c69e7b78a905e51ba26c781566adbe6da87`.
The final three agent-produced GREEN payloads above were also independently
read and rehashed during integration. Original failure records are retained.
