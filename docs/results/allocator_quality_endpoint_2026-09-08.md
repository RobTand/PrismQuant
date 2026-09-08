# Preserve the feasible allocation quality endpoint — 2026-09-08

The completed LFM2.5-8B-A1B joint probe now produces the exact zero-predicted-loss
16-bpp endpoint: **2,142 BF16 units**, replacing the previous 2,141 BF16 units
plus `model.layers.10.self_attn.out_proj@TESSERA_BF16_K1_R1024`.
The remaining **113 budget rows and 35 distinct lower-budget assignment files
are unchanged**. The original probe and its bytes were not modified or repeated.

## Cause and correction

The actual 38 aggregated solver units had a 3.543686139861996-bpp minimum.
At 0.0001-bit precision, the globally rounded 16-bpp capacity was 124,564 bins.
Rounding the BF16 upgrade separately for each unit charged 124,570 bins,
although its exact candidate payload was 16.0 bpp and its predicted loss zero.
The solver therefore retained a lossy candidate, producing 15.993667894108281
bpp and predicted loss 0.00029748423740553554.

`solve_with_promotion` now checks the unconstrained per-unit minimum measured
loss assignment before the binned proposal. Local equal-loss ties prefer larger
payloads, consistent with the existing feasible-iterate tie rule. The point is
accepted only when the existing serving-group promotion preserves every unit's
minimum and the unrounded candidate bytes fit the declared target and tolerance.
Otherwise the old search runs. Each unit's loss is compared independently, so a
large unchanged background term cannot hide a smaller promotion-induced increase.

This is not a BF16 default: another measured format can be a unit's minimum.
`solve_allocation` and its bin charges are unchanged. The caller's exact shared
sidecar/artifact price and budget filter remain mandatory, and the full allocator
still makes no global artifact-optimality claim. No calibration, format menu,
wire, source checkpoint or serving contract changes.

## Evidence and validation

Source before the correction: `56c51cc28bce3a7dba09b1d9fa55ba731bc00fce`.
Validated implementation: `7280605d5` (full commit in the PB snapshot receipt).
Original allocation input SHA256:
`5d08904df98d07d91fa29cffe9d5bb44080e63f8149037b1f29153b6457bd451`.
The actual diagnostic's 38-unit numeric inputs are retained with source/probe
hashes in `tests/fixtures/lfm_quality_endpoint.json`; the test isolates the
already aggregated units and the full CLI run separately covers real grouping.

- Diagnostic PB `8adeb39b22f1e6996ec4a40fb6af8a8e3a0f9efcb2cc8a79763ae289e310e207`
  captured the first real solver invocation after normal input admission and
  intentionally stopped there. Exit0 and complete cleanup; it was not a full
  allocator run.
- Regression PB `1e9ea8538cc77b193349a04de9df573d21580592b341d89f5c61b52105b949f1`
  failed the expected all-BF16 assertion before the fix (one failure, two passes).
- **67 targeted CPU tests passed, zero skips**, across eight PB shards. Coverage
  includes bin rounding, strict budgets, format-independent minima, real grouping,
  promotion loss beside a 1e16 background term, the extracted LFM input,
  main enforcement, sidecar/byte budgets, serving constraints and architecture.
- Full-frontier PB `738178c019b6f11820bef45ae58bc7152aae147ca9ee219c2cb9de9a2ca88006`
  passed compile checks and all 114 budgets. Only the 16-bpp CSV row changed;
  all 35 other assignment payloads were byte-identical. The final serialized
  control contains exactly 2,142 W16A16 float entries. Input SHA256 was unchanged.

The full validation was a CPU-only admitted measurement on DL380, one preferred
physical CPU, 8 GiB reserved, Python3.12 and native threads1, with the original
producer package and calibration-bound table. It took 140.891 wall-seconds under
cProfile; the profile total was 139.722 seconds. There were 114 promoted solver
calls and 113 binned solves (2.178 cumulative seconds in the latter). Live Netdata
covered DL380 and both Sparks, 29 samples each, zero errors. Mean whole-host CPU
busy was 2.067% on DL380, 0.773% on Sparky and 0.906% on Sparklina. PB recorded
143.355 scope CPU-seconds, peak memory 2,815,180,800 bytes and complete cleanup.

The retained before profile is `validation-pair-01/after/allocator.pstats`
(147.102 profile-seconds), with its contemporaneous all-three-host telemetry.
These instruments document the hot-path change; this correctness result makes
no additional speedup, GPU-saturation, energy or serving-quality claim.

Root verified the ten successful canonical PB receipts, actual CAS payload
hashes, source bundles, all 48 full-validation artifact hashes and the final
serialized BF16 control. The full measurement source differs from the reviewed
commit only by PB-generated closure metadata. Test snapshots have the same
production code, tests, fixture and architecture; the later measurement helper
is the only additional source file in the committed tree.

Evidence under
`/mnt/shared/tessera-measurements/first-model-20260907/allocation-preparation-01/`:
`endpoint-diagnostic-01/solver-inputs.json` and
`endpoint-validation-01/{result.json,root-audit.json,allocator.pstats,profile-top.txt,frontier/}`,
plus the three host telemetry files. The test manifest is
`/mnt/shared/tessera-measurements/allocator-endpoint-tests-20260908.json`.
Reproduce through PB using the checked-in
`experiments/allocator_endpoint_validation.py` with a fresh output location;
its frozen input/reference paths identify this retained experiment.

The earlier fused-group producer-contract fallback remains unresolved. This
allocator correction does not complete GLM capture or qualify a shipping quant.
