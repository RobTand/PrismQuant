# #87 paired quantized validation plan

Prepared before serving; no result is claimed. The policy implementation is
recorded in `docs/results/pq87_paired_policy_2026-09-04.md`.

The committed `experiments/pq87_validation_manifest.json` selects Qwen3-0.6B:
shared BF16 source from the #113 input freeze and the existing native NVFP4
reference under `tessera-runs/bf16/refs`. Read-only PrismaBuild inventory
`74cbc51063720a960483cd2be87cedea20bf39d197500d1f9b1eb298bb51aaf8`
found 1,503,300,328 BF16 safetensors bytes and 870,290,032 candidate bytes,
identical `tokenizer.json` and `tokenizer_config.json` hashes, and a candidate
build block naming the original Qwen3-0.6B with 196 NVFP4 assignments. The candidate's shipping slots
are all unfilled: this is a materialized native candidate, not a ship-certified
artifact. Exact weight content will be frozen and hashed before startup.

One exclusive PrismaBuild action will run two sequential servers on the same
box/image/source closure: BF16, then native compressed-tensors NVFP4. Each
uses the instrument's explicit one sequence, 4096-token context, 1 GiB KV and
eager settings. Docker has 28 GiB memory/swap and two CPU limits; the fleet
reservation must be the entire physical GPU capacity plus two CPUs/32 GiB.
The manifest's 2400-second wall deadline is an inconclusive backstop, not a
predicted completion time. An enclosing `timeout` must retain cleanup grace.
No GPU action may be submitted without the coordinator allocating that box.

Resource correction for #208 (2026-09-05): both paired servers explicitly
pass `--gpu-memory-utilization 0.2`. This is about 24.3 GiB on the 121.63 GiB
GB10, within the existing 28 GiB container envelope. The pinned image's
inherited 0.92 startup check requested 111.9 GiB despite the explicit 1 GiB
KV allocation and refused with 111.87 GiB free before any model observation.
The KV, sequence, prompt, seed, cap, PPL and candidate-decision contracts
remain unchanged. Use a fresh task-owned local output directory for the
Docker bind mounts, then archive receipts to shared storage after safe
cleanup: the first #208 action's new NFS parent was mode 0700 and Docker
could not traverse it. Retain both infrastructure-inconclusive attempts.

The client now retains `<arm>-serve-pre.json` before any pairing refusal,
so the observed stack is available even when no completed arm receipt is
written. A bounded `--stack-preflight` diagnostic reuses the full controller's
freeze, sequential servers, warmups and cleanup to capture both raw stack
manifests. It skips all policy and PPL clients and ends with
`status: preflight_completed`; this is never a completed quantized campaign.
Keep the full fingerprints and all their fields when diagnosing a mismatch.

The BF16 serve derives an uncensored schedule separately for the existing
30-pair screen and a disjoint 30-pair heldout set. Both prompt sets and seeds
are committed before either model is observed; no candidate outcome selects
a cap or prompt. The candidate runs the exact matched BF16 schedules. A
refused candidate is recorded and the second population still runs. No raw
endpoint A/B is repeated. A new source/campaign cannot silently reuse the old
BF16-only receipt as its own control.

Both live servers also run `validate_quantized_model.check_perplexity` under
its existing bounds and prompt roster. This is the small numeric screening
check, not gold WikiText PPL, full-vocabulary served KL, or a tool benchmark.
It has its own pre/post serve/content bindings. Every HTTP response, cap step,
server log, process I/O window and both-box Netdata series is retained. GPU
power is recorded separately from unsupported GB10 memory-utilization metrics.
No speed or work-per-joule claim is planned.
An existing PPL `CheckResult(passed=False)` can mean either a real threshold
failure or missing data. The controller retains the raw result but requires
finite metrics for the full existing prompt roster, scored tokens and an
explicit non-speculative observation before counting the phase as completed.
Empty, skipped or incomparable results are inconclusive; a measured numeric
threshold refusal remains a completed observation, not a pass.

The controller refuses overlap between screen and heldout prompts, mutable
image tags, ambiguous model identity, missing telemetry URLs and an unbounded
manifest. Original sources are never edited; copied model bytes are mounted
read-only through both container aliases. Shutdown must leave no serving
process before the next arm. Docker publishes a fresh CID file; cleanup
inspects that exact ID and verifies both the pool action label and a new
campaign nonce before any stop/remove. Output basenames never identify a
cleanup target. Prelaunch failures clean nothing, and a name collision
refuses without touching the existing container. Exact-owned-container
cleanup is recorded even after timeout or refusal, and the coordinator checks
actual worker completion before releasing the GPU to another task.
The proposed enclosing command is `timeout --signal=TERM --kill-after=90s
2400s python -P experiments/pq87_paired_validation.py --manifest
experiments/pq87_validation_manifest.json --out <fresh-absolute-output>` inside
a deployed-pool action with `--timeout-s 2550`. GNU timeout is required: the
outer TERM/KILL bound covers synchronous source hashing/copying too, while
the internal deadline bounds requests and subprocesses. This is a proposed
launch contract, not authorization or a measurement receipt.

In deployed runtime `aa6d3cfa2f77`, `pool.py:2059` calls
`cleanup_action_containers` before capacity release; its incomplete path at
2060–2071 retains the claim/tokens. The owner cleanup at 1593–1597 removes
only action-labelled containers and verifies the label query is empty.
Consequently a controller's advisory rc0 is not itself evidence of capacity
release. The coordinator still verifies the original worker's final state
and owner cleanup before handoff; no scheduler/runtime change is made here.

Remaining after this first campaign: an additional representative model/shape
population with its own BF16 control; actual heldout/gold/downstream evidence
sufficient for any proposed shipping-default promotion. Existing Gridbook-only
reproduction claims remain third-party/historical and are not substituted by
Qwen measurements. Nothing here reinstates the retired Gridbook lane or
waives that difference in scope. #87 stays open until the agreed shipping fix
and required evidence are complete.

## Controller verification, not served validation

All controller checks used deployed PrismaBuild `aa6d3cfa2f77`, Sparky with
one CPU/4 GiB, `CUDA_VISIBLE_DEVICES=''`, and all three BLAS/OMP thread limits
set to one. The serial eight-file population collected every selected module,
with zero skips or collection errors. No master baseline or full suite ran.
As for the policy receipt, the three unused absolute calibration symlinks
were excluded only during snapshot sealing, then immediately restored.

Before implementing the controller, action
`220734d546405ba7811ab1b61a683c24e7a9421979e773fc41be0bd15caf9b60`
was **8 failed** with `ModuleNotFoundError: No module named
'experiments.pq87_paired_validation'`. Before adding physical-admission and
PPL binding guards, action
`41e866101a125f3087b962ea99b0f49f6d9bd88a8f7a9223cff82a7ceff37150`
was **3 failed, 8 deselected**: `Failed: CPU action reached physical
preparation`, and two missing-`require_ppl_binding` `AttributeError`s.
Both red wrappers returned success only for pytest status 1; neither is a
green test claim. A later submission named a nonexistent test file and
collected nothing; that command error supplies no test evidence.

Final action
`28a6956f4ef3bc1e9a5203535b0e22e787a67d1c23f9f623f4f4fc1a56321681`
observed **116 passed, 0 failed, 0 skipped**, plus successful compilation of
`boundary_control.py`, `measure_boundary_control.py` and this controller.
Receipt SHA-256:
`a0220f3c9e8ffb3c217e7e9254108251616b61fa08148e34a30e4067fc6ff99d`.
Result blob:
`4b7b4b008403cac3dda1a2341bfa0d985441a79289c6c198f8915acd124420d2`.

Selected files: `test_boundary_policy.py`, `test_boundary_control.py`,
`test_measure_boundary_control_identity.py`, `test_pq87_physical_ab.py`,
`test_pq87_paired_validation.py`, `test_ship_boundary_behavior.py`,
`test_docs_staleness.py`, and `test_architecture_doc.py`. The command was
`python -m pytest -q -ra --tb=short -p no:cacheprovider` with those eight
`tests/` paths, followed by `python -m py_compile` on the three modules,
inside the bounded CPU-only pool action. These fixtures verify the policy,
manifest and safety guards; they do not prove a Docker lifecycle or a served
quantized population.

Independent source review then found the old basename cleanup could touch an
unrelated container after prelaunch failure. The owned-container correction
has separate red/green evidence: action
`d953742a3fea0d650b250951169123b2cac8d4ebdbf7ddcdc46f90a67e2476a3`
was **5 failed, 11 deselected**, including `Failed: cleaned uncreated container
pq87-shared-basename` and absent ownership helpers. Action
`2383a1f48370038e91a8e2384b4e7e89b81bf909e973e75baa3bd3ea24f105a5`
was **121 passed, 0 failed, 0 skipped**, all eight files collected and the
same three compile checks successful, on the CPU-only population above.
Receipt: `4eb9490f3acac4d30aa2f0dad6270df16ad9f96525aefa5bb567c02970a7b547`;
result: `be4aeb24f9ae8fd7491f3fef903146989165206df6705d80e26c7c40a408fc5f`.

The review also found missing PPL results could count as completed. The
separate PPL correction first failed eight result-shape fixtures under action
`1bbd62be5ec432ba07ec8a60b9aa91a15640ede68fe1588aea43bed62edd5c64`
(`AttributeError` for the absent `require_ppl_measurement` guard). Two actual
`client_phase` regressions then failed with the old unguarded PPL completion
path: **2 failed, 24 deselected**, `Failed: DID NOT RAISE <class 'ValueError'>`
under action
`3a5855f81e6cd37eb602ce8218917e462ff8a0c61fa2c256c25bba2aeafb5cdb`.
They supplied the real empty and skipped `CheckResult` shapes and observed
the previous success return; no served endpoint was contacted.

Final action
`1226efdf25566273bfb14e7d48121a2825e3fe4421a006d757c56a8bc8b56629`
was **131 passed, 0 failed, 0 skipped**, all eight files collected and the
same three compile checks successful, on the CPU-only population above.
Receipt: `d504f29de11609a0852914fa518c92e73b6191f2c7c0dca26a66f0f48184a448`;
result: `0f425ec14cd0018d78a88eef72722993445a4ed2a4f2e8fd9c89825e43356d15`.
