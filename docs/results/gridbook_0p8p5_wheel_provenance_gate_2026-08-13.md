# Gridbook 0.8.5 release-wheel provenance gate — 2026-08-13

## Purpose

Validate PrismaQuant 0.12.1's digest-pinned wheel path against the exact
DSv4Flash serving image after 0.12.0 rejected its truthful PEP 610
`archive_info` as though every supported install had to be a VCS install.

## Immutable inputs

- Image: `sha256:15da66d6a6c516a54e5490be30b1ab0780e7ab5e1403656082cd251f9eddbaf9`
- Gridbook version: `0.8.5`
- Gridbook commit: `e992e5980c96333a48149f96392d6cff56ae9e3f`
- Wheel SHA-256: `51122fab1533d538230836b103cef9f438dbea015a75c671437e52392cf90d4d`
- PrismaQuant baseline: `v0.12.0` (`3c23cf076c69ca44fcd59a0d7d1acef5166c8a97`)

## Command

The image was run without a GPU, with this worktree mounted read-only at
`/repo`. A Python safe-path process called
`tools.serve_fingerprint.gridbook_distribution_provenance` with the exact
repository, commit, version, and wheel digest above. The focused replay suite
was then run with:

```bash
python3 -m pytest -q \
  tests/test_serve_fingerprint_descendants.py \
  tests/test_kl_ab.py \
  tests/test_shipcard_gold_replay.py \
  tests/test_validate_cb_endpoint.py \
  tests/test_validate_cb_performance.py
```

## Result

The real image passed. Its PEP 610 record named
`file:///opt/gridbook-install/gridbook-0.8.5-py3-none-any.whl` and carried the
exact expected SHA-256 in both `archive_info.hash` and
`archive_info.hashes.sha256`. PrismaQuant verified 52 installed Gridbook
Python/CUDA/package-data files against RECORD; their canonical identity was
`73600a168e707dc02b3972e060f4826d698150eb6c2b73d19ee7554266f843c4`.
The imported module resolved to
`/usr/local/lib/python3.12/dist-packages/gridbook/__init__.py`.

The initial focused suite passed 180 tests. After adding endpoint replay tests
for a valid release wheel and a mismatched digest, the expanded provenance,
endpoint, performance, shipcard, runtime-contract, and architecture-doc suite
passed 200 tests with six expected skips. No GPU work was required for this
provenance repair.
