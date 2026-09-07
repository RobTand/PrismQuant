# Export resume identity review — 2026-09-07

The standalone compressed-tensors exporter binds resumable layer payloads to
source shard/config/index/root-Python content, requested parameter dtype,
persistent-buffer dtype declarations, render policy and assignment. Matching
inputs retain replay; changed inputs discard old layer payloads through the
existing admission path. This completes the source identity work in issue #340.

Independent review found one remaining admission hole at author head
`ecb0e292e91da127c92c6a92a1a06c9c77a79bc9`: deleting `manifest.json` left
`layer_*.pt` files eligible for replay after writing a new fingerprint. A changed
source could therefore emit the previous source's weights. The same-source
orphan also lacked any bound identity. Two real streaming-export regressions
failed before the correction, action
`daa537459dba50c00440525dbb30fc4db6a1c1dd1be56fa2f6fce349c41db2ac`.
They inspect emitted tensors and spy on actual cached-layer loads; only the
small CPU model skeleton is substituted.

The correction sends missing-manifest caches containing layer payloads through
the existing refusal path. Fresh empty caches retain initialization behavior.
The original author's commits are retained, with the review correction and
prose scope changes in separate commits on a separate integration branch.
The author checkout and measurement checkouts remain untouched.

After the correction, PrismaBuild action
`adc5e166123d01a67fd9a5acb4ce870ce52cf330ed9c0bb98e06dcc56181eb9a`
passed 239 tests and 163 subtests in 9.21 seconds. Final integration with main
`4f201a6470b1ea8b2837a039f7807ed94e867a31` passed the same 239 tests and
163 subtests in 9.47 seconds, action
`3ce3152c3d69e8f16def072a2c515010b856fa8c0c558e4557816e963ab1a2b0`.
There were no skips or missing collections; both touched producer modules
compiled in the final action. Both ran on DL380 CPU, four workers/8 GiB total,
native threads one, using the scoped Python 3.12 CPU environment and recorded
Tessera producer source dependency. Exact source snapshots, actual terminal
exits, logs, CAS receipts/payloads and complete resource cleanup were checked.
The final tested source differs from `d7212ed230` only in the generated PB
closure file; the dated report added afterward changes no implementation.

Test selection: `tests/test_pr348_review_regressions.py`,
`tests/test_export_resume_source_identity.py`,
`tests/test_prismaquant_export_native_compressed.py`,
`tests/test_export_buffer_precision.py`, `tests/test_export_output_safety.py`,
`tests/test_architecture_doc.py`, and `tests/test_docs_staleness.py`, with
`pytest -q -n4 --dist worksteal -p no:cacheprovider` through the published PB
client. Exact commands and audits are retained in
`/home/rob/tmp/pr348-root-final-command.json`,
`/home/rob/tmp/pr348-orphan-root-green-audit.json`, and
`/home/rob/tmp/pr348-root-final-cas-audit.json`. The earlier independent
review packet is `/mnt/shared/tessera-measurements/pq-pr348-review-20260907/REVIEW.md`.

No full export identity/admission runtime was benchmarked. Cached shard
digests avoid rereading unchanged shard contents; source discovery, metadata
hashing, digest-cache JSON handling and identity construction still execute.
Primitive hash/stat measurements do not establish total resume latency.
No GPU, vLLM serving, format change or artifact ship qualification is claimed.
