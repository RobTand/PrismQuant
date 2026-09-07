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
`run-03/run/sampling/stacks.raw` SHA256 is
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

Validation is recorded below after admitted checks complete.
