# Tessera consumer integration — 2026-09-04

Source `a0c8cf25` integrates main's pinned CI dependency, the reviewed
boundary request fixes from PR #177, and PR #179's v4 consumer. The only
merge conflict was the architecture-document preamble; both records remain.
No release pin, shipping threshold, or MoE eligibility is advanced here.

## PrismaBuild receipt

- Action: `446880313ebc79c732e6170f97849247bd2dd5fef35974d4e8b7583b87bf9026`.
- Receipt: `04edf74ae248f59b3f6323aedc43cf0171150a668aa1851c067b07e76d80155c`.
- Result SHA256: `e216c67fa411d63a959e2823409f6ea5e7966d7b950f86a90fb26dbb4fe20826`.
- Worker: Sparky, one CPU, 4 GiB reservation, CUDA hidden; serial pytest.
- Result: **196 passed, 0 failed, 0 skipped, 0 collection errors**, 135.93 s.
  There are no skip reasons to report. This was CPU-only and does not cover
  the GPU surface or the entire repository.
- Compilation checks passed for `lane_eligibility.py`,
  `tessera_runtime_contract.py`, and `resolve_tessera_dev_pin.py`.

The run used the verified Tessera `1221d2a` source archive,
`/mnt/shared/pq-v4-source.o2Tc3O/tessera-1221d2a-src.tar`, SHA256
`b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8`.
The exact command is retained in PrismaBuild's action request.

Targeted files: `test_lane_eligibility_v4.py`, `test_tessera_contract_v4.py`,
`test_tessera_dev_predicates.py`, `test_tessera_lane_admission.py`,
`test_tessera_export_lane.py`, `test_tessera_substitute_decoder.py`,
`test_tessera_serve_fingerprint.py`, `test_tessera_serving_contract_path.py`,
`test_tessera_menu.py`, `test_tessera_menu_real_table.py`,
`test_docs_staleness.py`, `test_architecture_doc.py`, and
`test_ci_tessera_install.py`.

## Snapshot boundary and remaining work

PrismaBuild correctly refused the first submission because the repository
contains absolute external calibration symlinks. For this targeted test
snapshot only, `calibration/diverse-v1.jsonl`, `calibration/xdom-fit-v1.jsonl`,
and `calibration/xdom-gate-v1.jsonl` were omitted. All three links were restored
to their exact HEAD values immediately after queueing; no target data was
removed or changed. These tests do not use those datasets. Consequently this
is a source-and-test integration receipt, not a claim of a byte-identical
whole-checkout snapshot or a calibration measurement.

The separate #87 paired-instrument work and its GPU A/B remain pending.
Runtime-scoped v5 consumption is a separate reviewed change. Hosted CI's
private Tessera checkout remains #175, explicitly deferred until publication;
the local receipt does not claim hosted CI is green.
