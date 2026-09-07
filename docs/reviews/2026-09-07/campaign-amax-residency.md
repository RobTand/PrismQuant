# Campaign maxima residency — 2026-09-07

The campaign previously called `.item()` for each observed unit in each
calibration batch. The full-capture profile at
`/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/baseline/collector.speedscope.json`
identified this scalar-read site as a hotspot. That earlier capture used the
subsequently rejected checkpoint initializer; its artifacts remain invalid for
calibration reuse or quality claims. It is bounded profiling evidence only.

The collector now keeps each maximum as a device float32 scalar, combines
batches with `torch.fmax`, and converts once per unit after all forwards.
`fmax(previous, NaN)` retains the previous value, matching the existing Python
`max` policy. Full-row coverage, score-prefix selection and Hessians are
unchanged. No GPU speedup or energy delta is claimed here; canonical matched
profiling is a separate measurement prerequisite.

Validation ran through PrismaBuild on the x86 CPU worker with
`/home/rob/venvs/pq-cpu312/bin/python`, the pinned Tessera source on PYTHONPATH,
and OMP/MKL/OpenBLAS threads bounded to one per worker:

- Before: `pytest -q tests/test_campaign_amax_residency.py --tb=short`, action
  `74706a628738f7065b1b8708c70ae70e14ba62167396fab6082d868b9ebda40b`, expected
  exit 1. The two-unit, three-nonempty-batch fixture observed six scalar reads,
  while maximum values and counts already matched.
- After: eight-worker pytest over that regression, `test_tessera_campaign.py`,
  `test_tessera_campaign_packed.py`, and `test_architecture_doc.py`, action
  `5048a5c104a7751674f9fd99fb07945ef1ab2cfca2c450860db8a81d06e95ac9`, exit 0,
  **76 passed, 5 CUDA-required skips**, 171.39 seconds. Actual CAS payload:
  `/mnt/shared/prismabuild-fleet/cas/blobs/5f/5fc9ee47f6a06cff5dbd78a971d4015091ce1574a2959a7ef30ec0c9aa8432df`.

Initial submissions `be339e093292` and `4c1c3cdc5cd3` omitted the required
Tessera import path and failed collection/imports. They are not behavioral
results; the corrected invocations above supply that dependency explicitly.
