# Explicit joint AURA source transition — 2026-09-07

The first-model run completed 1,260 of 2,142 units before its 7,200-second
execution ceiling. Attempt `4bd590c68638629d6051e83ea690ea2016bd4d0dda9a0fa6584cb361eb8fd51f`
ran 7,200.128 seconds, ended with launcher exit 143, had no OOM, and completed
cleanup. Thirteen reverse layers had completed. The killed sampling process did
not flush its stacks, so this attempt has no completed full-cost profile.

Resume attempt `1ace36719ce9e613a93f6943524499dba2438e2ce7db5998bf4a2b62c4e9b7eb`
ran 528.679 seconds and exited 1 when a fully completed reverse layer created an
empty joint projection lease. The empty lease selected CPU, while the explicitly
qualified fused backend required its prewarmed CUDA device. There was no OOM;
cleanup completed. The sampler retained 32,383 samples and one error. Its
`run-03/run-attempt-02-empty-lease/sampling/stacks.raw` SHA256 is
`29ffccc448796f083ee9720b1c7b006d1a76f5d91d743ad290b954667924ba73`.
No extra checkpoint unit was written. All original unit bytes and the manifest
were rehashed unchanged, and both hosts' Netdata evidence was retained. Neither
attempt establishes completed full-model joint costs or end-to-end speed.

The empty-lease fix is independently documented in
[joint_resume_empty_lease_2026-09-07.md](joint_resume_empty_lease_2026-09-07.md).
It retains reverse cotangent traversal through completed layers, because earlier
pending layers still depend on it. CPU regression and exact signed-cost evidence
support this arithmetic-neutral change. They do not permit pretending that the
new producer source is the old source.

`prismaquant.joint_aura_source_transition` defines one closed version,
`empty_joint_lease_v1`. Its original source, Git, plan, prepared cache, manifest,
inspection receipt and 1,260-unit roster are code-owned content bindings. There
is no caller-selected expected source hash or output pathname exemption.
`source_proof` reverses exact reviewed snippets in `aura_cost.py` and
`tessera_joint_aura.py`, omits only the newly introduced transition module, and
hashes every remaining durable package byte with the existing complete-package
algorithm. Any additional source change fails. The independent transition
receipt also binds the complete current package, including the verifier module,
and the actual clean Git commit. The legacy Git identity environment override is
refused. A future source change requires a separately reviewed contract.

The receipt is created with exclusive creation and cannot overwrite an existing
receipt. Creation reads original artifacts and writes only this new receipt.
Admission rechecks every content binding, all original envelopes and payloads,
and the actual running source before calibration/model CUDA work. Runtime APIs
accept only a factory-issued immutable verified object. The old probe/source
identity remains the measurement identity after the exact source proof; strict
checkpoint comparison substitutes only top-level Git and producer source fields
and rejects every changed non-source field. The old checkpoint manifest and
prepared cache are never rewritten. Existing render/source/calibration/backend
checks still run.

Each newly completed unit adds an `execution_provenance` field outside its
signed rows, binding the current Git/package/verifier hashes and transition
receipt SHA256. Subsequent resumes require that provenance on all units outside
the preserved original roster. Final output lists both the preserved envelope
and payload hashes and the newly completed envelope and payload hashes. It keeps
actual execution provenance separate from the original measurement identity.
A transition receipt made under one PB snapshot cannot admit another Git HEAD.
Create a new receipt inside each cost action's actual snapshot, binding the
previous receipt with `--predecessor` and `--predecessor-sha256` after an
interruption. The receipt chain must have identical complete producer/verifier
and reconstructed-source hashes and original input bindings. The new receipt
captures every existing post-original unit's exact envelope/payload hashes and
its predecessor execution provenance. Earlier receipts and their unit bytes
remain unchanged. Cycles, missing predecessor bindings, changed inherited unit
hashes and mismatched execution source fail closed. Do not set `PRISMAQUANT_IDENTITY_GIT_COMMIT`.

Create the receipt with `python -m prismaquant.joint_aura_source_transition`,
supplying independently SHA256-bound `--plan`, `--prepared`, `--inspection`,
`--checkpoint-dir` and a new `--output`. Resume through the existing
`python -m prismaquant.tessera_joint_aura run --resume` command, adding
`--source-transition` and `--source-transition-sha256`. Both actions must run
through PrismaBuild. Receipt creation and CPU qualification do not authorize
GPU success claims; inspect the actual resumed action's result and output.

## Validation

PB `3ea902107496222681ca0842364337247ede2223b48660824ef771329ae955ad`
ran 188 CPU tests: all passed, zero skips or missing collection, with 56
upstream Torch/Python deprecation warnings. The suite includes source-proof
algorithm mutations, changed original inputs, forged capability rejection,
changed non-source identity, exact producer resume costs, two interrupted
resumes with different PB-style Git HEADs, immutable prior unit bytes,
predecessor-chain tampering, backend/packed/microbatch/checkpoint regressions,
and architecture/staleness gates.

```text
bash experiments/pq322_cpu_checks.sh \
  tests/test_joint_aura_source_transition.py tests/test_tessera_joint_aura.py \
  tests/test_joint_aura_streamed.py tests/test_joint_aura_microbatch.py \
  tests/test_joint_aura_packed.py tests/test_joint_aura_projection.py \
  tests/test_joint_projection_backend.py tests/test_streamed_cost_checkpoints.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py \
  -q -n 4 --dist worksteal -p no:cacheprovider
```

It was a portable priority -10 action with four CPUs/eight GiB, executed on
dl380g10 with native threads bounded to one. Runtime: Python 3.14.4,
Torch 2.11.0+cpu, Transformers 5.16.1, pytest 9.0.2; CUDA disabled.
CAS payload `06ea47126056e09934170497bee50c986b49e271396e5980dea863c58966bdd2`;
receipt `649898a0982dde227be025385f56e096e2fa9007a4b43af13ae9756c49cb7a44`.

PB `1f442dc649c5b79b24f0e65a0704d8ded6541b6d3817e555fe4818a44de1afde`
ran `experiments/verify_joint_source_transition.py` on the original first-model
artifacts using the existing x86 CPU runtime (one CPU/four GiB; priority -10).
It verified all 1,260 original envelope/payload hashes, the unchanged original
manifest/preparation/cache and plan bindings, receipt admission, exact
non-source identity comparison, and a rejected seed mutation. Its complete
new package SHA256 is
`f19fc208d0ed101d849351976224f86d471ba76b9d9fadcba8d61c4a86b306a8`;
reconstructing only the approved source change gives the exact original
`735a5712fc43153709c1f9e463fe1fff649cb50ac8c28c5c305ad6369c3043ba`.
The verifier module SHA256 is
`499ab81af0304405ddd76b49b6af28265c650fce8ba72c14be8aea13122a1aef`.
This actual-package gate is explicit acceptance evidence; ordinary CI tests
verify the algorithm on portable byte-tree fixtures so future unrelated source
changes remain fail-closed without breaking normal CI.

Both terminal exit codes were zero. Cleanup completed with released scopes and
no OOM. Actual stdout/stderr, CAS payload bytes and canonical receipt hashes
were independently checked. Evidence root:
`/mnt/shared/tessera-measurements/joint-source-transition-20260907/`;
`cpu-verified-01.json`, `artifact-verified-01.json`, and
`artifact-gate-01/result.json` retain the checks. The artifact gate wrote only a
new receipt/result under its evidence directory; frozen sources and checkpoint
bytes were unchanged. No GPU cost, serving, throughput or quality claim is made.

A first CPU submission `9bfffc7c0a2c` inherited PB's local-worker tag and was
withdrawn from the ready queue with zero tokens held before coordinated NFS
maintenance. It ran no tests and is superseded by the portable green action.


PB `2b13aee3a14fc8eb651754b9ba88a7c36d2b23c93828d62a76a4330aa9ab99a0`
compiled all five touched Python modules on dl380g10 (portable priority -10,
one CPU/one GiB), exit zero and released scope. Its actual CAS output was
`5 touched Python modules compile`. `compile-verified-01.json` records the
terminal, snapshot, cleanup, payload and verified canonical receipt.


## Integration into current main

The integration branch starts from `8281ace2f2`, where #344 already supplies
the empty-lease guard. Only the explicit transition implementation, its tests,
acceptance tool and evidence documentation are added. The execution checkout
at `9e05ab56bd` remains frozen; its original artifact-gate receipt continues to
attest that exact source. Current main contains unrelated producer changes and
therefore intentionally cannot pass the old source reconstruction contract.
Integration does not relax or update the approved original package hash.

Fresh current-main integration validation is recorded below after it completes.
