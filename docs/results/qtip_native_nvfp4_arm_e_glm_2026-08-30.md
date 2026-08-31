# QTIP-derived native-NVFP4 Arm E: GLM census

Date: 2026-08-30
Scope: research-only producer quality and feasibility; no serving claim

## Outcome

The implemented Arm E rotation plus structured BlockLDL path is physically
valid on every selected GLM shape, including `[4096, 12288]`, but it fails the
predeclared quality gate on both GLM populations.  It must remain a research
ablation and must not replace Arm C or change the production format menu.

This is a result about the current Arm E recipe, not about the base trellis
family.  It also is not an EXL3 comparison: EXL3 was not executed, its wire was
not imported, and no EXL3 quality or speed row enters this campaign.

## Bound inputs and execution

- Source tree: `e48e88b` (`Add exact structured Arm E GLM path`).
- Container: `sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9`.
- Finalized corpus:
  `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/`.
- Corpus manifest SHA-256:
  `a66f800827b92383985ce205004cd2d70b63bcc5e19cada6b05a8162401ee5b0`.
- Corpus artifact SHA-256:
  `0d3c08aed48e8d0b540d0705c305cc3197f77c250b07dd7a07e55345f5ddd94e`.
- Importance value identity:
  `dad7818dd11ea8f853bd1869f41189ca3de4a2d10deda52cfef563f63496a9dd`.
- Calibration identity:
  `45d126bca6b3a8ce75f03f29f85cc4d891a7af3c923ee26372a81e04b91bf5bf`
  (8 samples, 512 tokens each, seed 42).
- Full-census manifest SHA-256:
  `990d0bb57cce612b181bd6ab1fc16978372ff03edfe676a361ca6eae80320af3`.
- Full receipt SHA-256:
  `84987783448da1db3aa6d0dcf8a409f7ab383de45fb2329f1bc667a58dd87b32`.
- Execution window: epoch `[1788143617, 1788144269]` on Sparky.

The census contains the predeclared 9 dense and 24 routed tensors.  The two
populations are never pooled.  Arm A and Arm C happen to be byte- and
metric-identical for this diagonal-importance campaign; Arm E is compared to
both independently in the receipt.

## Quality

All entries below are paired Arm E minus stock Arm C SNR in dB.  A positive
number favors Arm E.

| Population | Metric | Median delta | Wins | Range |
|---|---|---:|---:|---:|
| dense (9) | raw importance-weighted | **-0.442704** | 2/9 | -1.711408 to +1.263259 |
| dense (9) | regularized-H proxy | **-1.350010** | 0/9 | -1.839072 to -0.718361 |
| dense (9) | unweighted weight | **-1.951331** | 0/9 | -2.136491 to -1.842462 |
| routed (24) | raw importance-weighted | **-1.803932** | 3/24 | -2.134158 to +3.303166 |
| routed (24) | regularized-H proxy | **-1.944914** | 0/24 | -2.209659 to -0.235071 |
| routed (24) | unweighted weight | **-2.109500** | 0/24 | -2.349015 to -1.697331 |

For the primary metric, the one-sided paired-bootstrap 95% lower bounds are
`-1.673316 dB` dense and `-2.021070 dB` routed.  The minimum gate required
positive medians and at least 7/9 dense and 18/24 routed wins.  Both population
gates therefore fail decisively.

The corpus contains retained activation second moments, not activation rows.
The receipt correctly marks activation-output measurement unavailable and
forbids a model-quality claim.  The synthetic algebra witnesses prove the
transform/decode seam only; they are not quality samples.

## Feasibility and profiling

The largest structured tensor ran with three ordered 4096-column factor
groups.  No global `12288 x 12288` Hessian was materialized.  Across all 33
tensors:

- maximum `torch.cuda.max_memory_allocated`: `6,686,095,360` bytes;
- sum of measured phase times: `507.615 s` under `cProfile`;
- slowest tensor measured phases: `40.534 s`;
- full profiled wall window: `652 s`.

`cProfile` is deliberately treated as hotspot evidence, not an unbiased
throughput measurement.  It recorded 3.72 billion calls and 651.420 s total.
The dominant stack was:

| Function | Calls | Cumulative time |
|---|---:|---:|
| `TrellisWire.validate` | 3,756 | 563.345 s |
| E2M1 scale generator | 736,104,108 | 524.918 s |
| `native_code_value` | 736,430,907 | 430.871 s |
| `get_trellis_family` | 736,454,689 | 149.745 s |

Profile SHA-256:
`ee56a80632389fb691e486d667390cd23040f0b5644ea43dbb7feb7bd20a1a9d`.

A separate `[4096, 12288]` pilot used an in-container Torch CUDA profiler and
is the valid CUDA feasibility trace.  It measured 6,014,437,376 peak bytes,
16.127 s of measured phases, 14.449 s in encode, 2.760 s self CUDA time, and
4.891 s self CPU time.  Its canonical wire was 4.4697747 bpw and passed the
same-byte check.  The earlier host-side `nsys profile docker run` pilot emitted
no CUDA events and is explicitly invalid as a CUDA profile; it must not be
cited as performance evidence.

## Box telemetry

Both boxes were captured over the full-census epoch interval.  The Netdata
manifest SHA-256 is
`5860fc600cbfaf7d15a2e2ce21a1bd03e1ce928f80323625f150763ef73441ef`.

- Sparky Netdata: 653 rows, mean 13.7098 W, p95 19 W, max 28 W,
  sample-sum estimate 8,952.5 J, or 9.79% of the 140 W envelope.
- Sparky pqteld: 1,304 rows, mean 14.4090 W, max 26.4 W,
  trapezoidal energy 9,390.02 J, or 10.29% of the envelope.
- Sparklina Netdata context: mean 3.9541 W and max 4 W.

These values show that the campaign was not a saturated serving workload.
GPU utilization is not used diagnostically, and no throughput or
work-per-joule ranking follows from this quality run.

## Validation-loop repair

The scale validator performed one Python family lookup and E4M3 decode for
every scale byte.  The equivalent closed check now constructs the set of
distinct scale bytes in C and tests it against the module-level set of all 126
finite positive E4M3FN codes.  The accepted language is unchanged: positive
subnormals through 448 are accepted; `+0`, `-0`, all negative values, and both
NaN encodings are rejected.

On the retained 28,121,391-byte GLM wire
`cfcd39e60a9cb16cd1c4900dcba161bc5fc096bf611c1ae2f2cd7cf856dad89a`,
the actual 3,145,728-byte scale plane contains 30 distinct codes.  A matched
single-check benchmark measured:

| Check | Plain time | Profiled time | Profile calls |
|---|---:|---:|---:|
| per-byte Python decode | 0.611835 s | 2.369589 s | 15,728,644 |
| distinct-byte subset | 0.019107 s | 0.019107 s | 3 |

The plain scale-check speedup is **32.02x**.  This is a same-input hotspot
claim only, not a projected whole-campaign speedup.  Benchmark JSON SHA-256:
`02a0e310d799de96c5f49e7e3feddee56c8cd5d6cf00aad63d3cb9ee95fca82b`;
old/new profile SHA-256 values are
`a450f1e60adb323786b23e3cfb8662cd5c2b636a097b564b9bee3925d418b43a`
and
`036e1d1517d588a0678148c82994c111031a2c45ff87ffcbf412b1b79249aa59`.
Both-host Netdata/CPU context is bound by manifest
`58963a893359a60075a845aeb7814e1472da11d6c93439b1320e9e023e0fc201`.

## Disposition

1. Preserve the native NVFP4 trellis lane and its measured low/mid-rate coding
   gain.  This census does not refute it.
2. Keep the current Arm E transform/BlockLDL recipe research-only; do not
   promote it, register it, or change Gridbook's immutable runtime pin.
3. Continue with the zero-bpw, same-wire scale-grid experiment only through a
   realized-objective no-regression gate.  A raw RTN group selector is not a
   proof for shaped or feedback-coupled trellis groups.
4. Use in-container Torch profiling for the next CUDA campaign; reject an
   empty external `nsys` trace at the evidence boundary.
