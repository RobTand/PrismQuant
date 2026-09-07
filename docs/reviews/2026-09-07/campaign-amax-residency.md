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

## Completed isolated canonical comparison

The earlier profiling prerequisite is now satisfied by the seven-arm canonical
measurement documented in [campaign-capture-reuse.md](campaign-capture-reuse.md).
PB `0c4b585e8f43472a085435628bff68955d8bd6b546da2fe39a583dcb8d6ed821`
exited 0; actual CAS payload
`84b67475c6831a43c7b03f404e1827e81e051539ff87f0186d9ece2d46e7d349`
and the broker's successful termination receipt were inspected. All arms
preserve the finalized-buffer change, canonical initializer, eager attention,
source checkpoint, 512 samples of 512 tokens, and the same 96 selected layer-10
expert projections. The scalar-sync collector differs from the current
collector only in the maxima hunk. Source bytes and `amax-only.diff` are
retained under
`/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/canonical-compare/frozen-source/`.
No cached arm is included in the isolated maxima statistic.

The ABBA order is device-max forward 3, scalar-sync forwards 4/5, device-max
forward 6. Arm 3 is the shared endpoint with the separate reuse ABBA. Profiled
scalar-sync times are **102.406 / 102.715 seconds**, versus device-max
**101.423 / 100.154 seconds**. Medians are 102.561 and 100.789 seconds, a
**1.73% observed reduction**. This is a small effect across only two repetitions
per method; it does not establish a robust or general speedup, and it does not
explain the much larger gain from skipping repeated forwards through reuse.
Every arm exactly matches all 96 canonical X/H/count/max records.

The before profiles have 1,938/2,023 samples; 195 samples in each fall at the
per-batch scalar-read line. The after profiles have 2,076/1,976 samples, and
the per-batch scalar-read site is absent by construction. Grouped-matmul frames
remain the largest sampled Python location on both sides. These are sampled
call locations, not CUDA event timing or a claim that every sampled scalar-read
second becomes saved wall time.

Both-host Netdata raw series and per-phase CPU/RAM/I/O/power summaries are
retained alongside all four in-process profiles. Interpolated median GPU
energy is **4,057.215 J scalar-sync versus 3,463.417 J device-max**, or 0.02366
versus 0.02772 selected units per GPU joule (**1.17× observed work/J ratio**).
Power uses ten-second sampling, with device-max means 31.821/36.938 W and
scalar-sync 40.227/38.894 W; repeat variability and coarse telemetry limit the
claim. This is GPU energy only, not whole-machine energy. Power is far below
the approximate 140 W envelope, so these measurements do not establish GPU
saturation. `derived-metrics.json` retains unrounded figures and limitations.
