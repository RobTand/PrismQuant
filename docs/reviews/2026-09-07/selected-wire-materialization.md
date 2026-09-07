# Selected expert-wire materialization — PrismaQuant #301

The sampled allocator can now write an explicit non-exportable selected-wire
request. Its canonical packed decisions and original cost digest remain fixed.
The new module turns only census stacks with missing selected wires into PB
rows; complete selected stacks reuse their original evidence. Each row uses
campaign calibration and the shared scalar/batch encoder, ProductionWeightCache,
producer input identities and cost-stage journal. It encodes only missing
selected unit/rung pairs and verifies existing receipts before reuse.

Finalization verifies the original source files, producer source/recipe, exact
census identity, every selected wire and the union of captures. Overlapping
Hessians and counts must agree exactly. The original sampled prices are not
rewritten or promoted to census measurements. Final allocation publication uses
the existing strict expert-receipt and priced-input gates. The latter now
expands the explicit packed-member map before checking each source unit's
static scalar, so unequal expert scalars are checked individually.

## Validation

All execution used PrismaBuild with GPU visibility disabled for CPU tests,
OMP/MKL/OpenBLAS threads bounded to one, and aggregate CPU/memory reservations.
No full-model GPU materialization, serving quality, throughput or residency
claim is made by these tests. The production path still pays the existing
whole-model forward cost once per selected census stack. No before/after
performance comparison was made.

- Pre-fix PB `5cdfb77dd2e2`: the new unequal-source-scale case failed because
  preflight searched packed keys. The two defective-scale variants also failed
  at that earlier wrong boundary. All three now exercise the source-key gate:
  valid unequal values pass, missing and altered values refuse.
- PB `707ecc774ed770da8838da52433486a20c6bc500fab8cea2069f63c92f3f2f8e`:
  **127 passed, 2 skipped**, rc 0, CPU on sparklina. The skips are the legacy
  absent-Hessian-seal case (the producer publishes a seal) and the separate
  CUDA-only batch CLI test. This used the frozen first-model producer
  `tessera-382a1a97`, including the real producer encoder/receipt/translator.
  Receipt: `/mnt/shared/prismabuild-fleet/cas/actions/v3/70/707ecc774ed770da8838da52433486a20c6bc500fab8cea2069f63c92f3f2f8e.json`.
  Actual pytest output: `/mnt/shared/prismabuild-fleet/cas/blobs/28/28f995706e71969c1c9dbb6174b80b832040e680f2eedf1c31f33b34080f53e0`.
- Final materializer PB `0bcf755f5d17d4549877a4d6305146f28f4f2110b0f77855b40f24b9e9976985`:
  **24 passed, no skips**, rc 0. This includes the strengthened packed
  two-of-three native producer fixture, source/census geometry checks,
  unbound-producer refusal and original-capture count checks added after the
  wider run. It used the same command/environment, restricted to
  `tests/test_tessera_materialization.py`.
  Receipt: `/mnt/shared/prismabuild-fleet/cas/actions/v3/0b/0bcf755f5d17d4549877a4d6305146f28f4f2110b0f77855b40f24b9e9976985.json`.
  Actual pytest output: `/mnt/shared/prismabuild-fleet/cas/blobs/54/54541db33208cf6266aece113f35e32c7a7e275b0ff8df59b275084a6b26182d`.
- PB `d40318e6d6ac7af10f7bc3cfd0dd33470f70afe92329d54632f73fad8dea8e9e`:
  compileall of the four touched modules, rc 0. Receipt:
  `/mnt/shared/prismabuild-fleet/cas/actions/v3/d4/d40318e6d6ac7af10f7bc3cfd0dd33470f70afe92329d54632f73fad8dea8e9e.json`.

The final materialization fixture additionally checks a packed 2-of-3 expert
sample using real producer bytes: only the third expert's three wires are
encoded, its original reference receipt is reproduced exactly, the packed
allocation stays packed, and the existing source-view handoff reaches the
producer's actual translator. It uses synthetic routed activations and omits
served-route scoring; it is a protocol and byte-integrity test, not a served
measurement. The sample needs at least two non-certainty experts: an attempted
one-of-two fixture correctly hit the estimator's variance refusal and was
replaced with the valid two-of-three frame.

A wider intermediate run against the old host producer also found three batch
test stubs that assumed `encode_linears` existed. Their owner fixed the test
portability in `87af2f94`; production materialization still refuses an explicitly
requested batch when the actual producer has no batch API.

## Commands

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-selected-wire-materialization \
  --cpus 2 --demand mem_gb=6 --tag gb10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --env TESSERA_REPO=/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97 \
  --env PYTHONPATH=/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97/src \
  -- /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q -rs -n 2 \
  tests/test_tessera_materialization.py tests/test_tessera_packed_export_scope.py \
  tests/test_tessera_expert_projection.py tests/test_tessera_priced_export_inputs.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py \
  tests/test_tessera_packed_plan_handoff.py tests/test_tessera_stack_driver_integration.py \
  tests/test_tessera_campaign_batch.py --tb=short
```

For the application workflow, add `--tessera-materialization-plan REQUEST` to
the ordinary allocator invocation while retaining its intended `--layer-config`
output. Run the new module's `plan` command with the same cost-derived census
and existing campaign spec, optionally `--anchor-batch-size N`, submit
`WORKSPACE/manifest.json` with the published `pbcampaign.py`, and run `finalize`
inside an admitted action. `WORKSPACE/final/result.json` names the exact final
allocation, Hessian capture, scale file and wire directory. This opt-in step
requires the dispatcher/container adapter and shared batch adapter commits;
no format default or serving gate changes.
