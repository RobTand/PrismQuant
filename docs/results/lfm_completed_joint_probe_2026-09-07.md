# LFM completed joint probe — 2026-09-07

The fixed LFM2.5-8B-A1B calibration produced all 2,142 quantizable units across
24 layers: 14,994 measured format cells and 2,142 explicit BF16 zero controls.
The final cost file is 326,004,962 bytes, SHA-256
`e976ed5fbfae0a86bbad75b5d21a9bf36333f42cbaacafa47229070f45d769c0`.
This is a reusable cost table, not a shipped artifact or a held-out quality result.

CPU validation through PrismaBuild `156235f31bfb27cce3e5f3589d2a5ed7c4b6833df4d001708d797bcd64e96888`
checked every row with the production joint validator, including signed/squared
samples, conditional standard errors, operator coordinates and shared probe
identity. All 17,136 rows exactly equal the checkpoint rows. The original 1,260
unit files and payloads retain their pre-interruption hashes; all 882 new unit
files carry the exact approved source-transition execution. Plan, prepared PWC,
original checkpoint manifest, inspection and transition receipt hashes passed.
The validator, source bundle, terminal exit, CAS receipt/payload, complete resource
cleanup and all 14 final-run artifacts were independently checked.

Inputs remain the original WT2 train seed-0 512×512 draw, B=1 source arithmetic,
four probes 7000–7003, temperature 1 and all-token KL-Fisher normalization.
Measurement identity is
`35097589e2bd3ec5385b5977058edf1ce60dd855dc1e346883d6d9af75757660`.
The final producer snapshot is `5ad5facd9a226cb04ec028eeab2dd102abffeb4e`;
the source-transition receipt is
`54a0c3a8e2a7d25b01334c69f4698cf84f6c63e763aee34b734cc32534dfaeb8`.
No completed unit was remeasured. Resume repeats forward boundaries and
cotangents needed for the unfinished reverse sweep.

## Execution and limits

The initial full run reached its declared 7,200-second deadline after 1,260
units. Its partial evidence and checkpoints were retained. The final successful
resume used the explicit reviewed source transition after a second bounded
attempt exposed the empty joint-lease defect; those failed attempts remain
negative evidence. The final action
`47b48e156cd82f1fb2cdcf12b9681f480453f6fcbb24f12bbd0d4d3589961f8d`
finished with exit 0 and complete cleanup on Sparklina in 5,261.414 seconds.
It used eight assigned CPU cores, native threads bounded to one, the unchanged
96 GiB host / 88 GiB GPU reservations, and the pinned EUGR container
`0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
The runtime reports PyTorch 2.13.0+cu130 and CUDA 13.0. The loaded temporary NFS
module identity and original measurement source are preserved in the run files.

The run reports peak CUDA allocation 29,550,637,056 bytes and peak reservation
30,823,940,096 bytes. The broker reports no OOM event and a host-memory peak at
its 96 GiB cap; this is not evidence of large physical headroom. These are
different accounting planes and must not be added as independent residency.

The complete py-spy recording has 381,652 samples at nominal 50 Hz, with no
reported sampler errors. Sample counts include threads and subprocesses, so
summing them is not elapsed time. Observed leaf locations include the CPU
cotangent copy at `aura_cost.py:3003` (23,949 samples), projection reduction at
`joint_projection_backend.py:149` (19,267), backward engine (14,587), and served
NVFP4 activation QDQ (8,207). They identify observed call sites, not measured
GPU kernel durations or an established bottleneck. The initial analysis failed
to include relative source paths in its optional inclusive table; analysis-02
corrects that parser and supersedes that table without repeating GPU work.

Both Sparks' Netdata series cover the final attempt. Sparklina device power
averaged 33.582 W over the recorded window, about 24% of the approximate 140 W
envelope. The native power chart updates every 10 seconds; the queried 1-second
rows are not independent power observations. Coarse trapezoidal integration is
176,262 device joules, or approximately 0.00500 new units/J and 0.03503 new
nonzero cells/J for this entire resume, including repeated setup/cotangents.
This is gross device energy, with observer/context overhead, and has no paired
baseline ranking. Sparky was external concurrent-work context. Neither GPU
utilization nor this single-run wall time establishes saturation or a speedup.

## Evidence and next stage

All relative evidence paths below are beneath
`/mnt/shared/tessera-measurements/first-model-20260907/full-model-joint-aura/run-03/`.
The original report is retained there as `final-cost-report-01.md`:

- `run/joint-cost.pkl`, `run/results.json`, `run/source-transition.json`.
- `final-cost-root-validation-01.json` and `final-cost-analysis-verified-01.json`.
- `run/sampling/stacks.raw`, `final-cost-profile-analysis-02.json`, and
  `cost-final-attempt-netdata-01/index.json` with both hosts' native chart metadata.
- CPU analysis actions `4a5bb3b062a8…` (retained first table) and `439754654aaa…`
  (relative-path correction), with actual CAS/source/cleanup verification.
- Root audit files: `/home/rob/tmp/lfm-cost-final-root-cas-audit.json`,
  `/home/rob/tmp/lfm-final-validation-profile-root-cas-audit.json`,
  `/home/rob/tmp/lfm-final-profile02-root-cas-audit.json`, and
  `/home/rob/tmp/lfm-final-cost-artifacts-root-audit.json`.

Allocation/export needs a checked metadata handoff from original measured
anchors and prepared PWC; the joint table itself lacks the wire/population and
priced-input metadata. The next stage must retain every joint row and statistic,
produce the coherent measured-cost frontier, and reuse original wire bytes.
A qualified single runtime covering the selected dense and expert cells,
held-out accuracy, matched-byte uniform control and full artifact serving
remain separate gates. The 8 bpp research menu ceiling is not a selected final
budget. No format, ship gate, pin, calibration or cost identity is changed here.
