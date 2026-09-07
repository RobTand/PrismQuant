# Shared packed-input collector qualification — 2026-09-07

The default-off `_collect_activations(..., shared_packed_inputs=True)` experiment
shares internal state only for packed projections receiving the same derived
input from the same module and expert. It retains the existing FP32 batch order,
full-row H/count/max calibration and capped prefix order. It uses the existing
store for private device prefixes, then drains each unique group before making
independent compact CPU outputs for its qnames. The physical guard checks each
transfer and sibling clone. Normal campaigns retain the legacy default.

The existing GLM target-layer-4 prefix attempt completed 512 original forwards
but refused final H materialization under the physical guard. This is a capacity
failure, not a successful collection baseline. Its source result is
`/mnt/shared/tessera-measurements/glm-streaming-source-20260907/workspace-prefix-04/result.json`,
SHA256 `7a6405943fd2e9ef5a69d24aad1b6e17aab565ce166d2dbd5a8381fe127f4048`.
The detailed cold/copy/steady-window analysis and proposal remain at
`/mnt/shared/tessera-measurements/glm-prefix-profile-review-20260907/proposal-01.md`.
The 18 GiB duplicate-H reduction for all 288 GLM gate/up pairs is an arithmetic
storage bound, not a measured saving; final independent CPU output size is
unchanged. There is no GPU speed, energy, full-fit or served-quality claim here.

## CPU qualification

Before changing the collector, PB
`4ce32b0277b0928ec730344a63da3bbf517f97b64d8bf00352a8d3a01ef9cbe0`
passed the actual routed call-count proof: six qnames over two batches compute
12 Grams although they have only four unique input groups per batch. The new
option computes eight and returns identical tensor bytes and serialized
per-qname X/H/count/max payloads. The first attempted bare CPU interpreter
collected no tests because its scoped dependencies were absent; subsequent
checks used the existing portable CPU/container wrapper with pinned tooling.

Final PB
`23cad6f0d274b38736e9ea1376a7483465748f75c55adaf1f8fcebe9c49a1b05`
passed **109 tests, zero skips and two subtests** on DL380 in 66.57 seconds,
using CPU Torch 2.11.0, Python 3.14.4, eight xdist workers and one native thread
per worker. Explicit compilation of the collector, workspace/replay harness and
both new test modules passed. Cases cover FP32/BF16, zero-row batches,
unobserved-group refusal, partial siblings, dense controls, cap crossings,
NaNs, signed zero, private prefix ownership, independent output mutation,
transfer/clone guards, forward failure, retained-traceback cleanup, partial
fixture-copy refusal, input/row replay, storage identity and telemetry energy
resolution. A real CPU-profiler exercise produced all twelve nonempty traces:
cold, row-filled and CPU-return windows for each of four arms.

The terminal exit, scope cleanup/release, actual payload, canonical receipt,
whole 26 MB checkout bundle and current source bytes were verified:

- Source snapshot: `45bb0cbbf48b396d17a739b75c9c954790003f16`.
- CAS payload: `3b273d9762fb1791a9a0855a3e7df91fee6c57097284cf4ccb3833a16ca0b36e`.
- Canonical receipt: `3a4ac7ff23ca8b429c979cec835872f6dbf442f6c0a65222e0f3a1a54309230f`.
- Exact packet: `/mnt/shared/tessera-measurements/glm-shared-input-capture-20260907/cpu-final-01.json`.

The separate ARM test-wrapper environment-forwarding correction is inherited
from driver commit `be82b2f3` (local equivalent `d2b2e7f8`); it is kept in its own
commit. It enables complete producer-fixture coverage on eligible ARM workers
without enabling GPU visibility.

## Reviewed GPU experiment, not yet a result

`--qualify-shared-inputs --qualification-expert-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15`
uses one original 512×512, B=1, seed-0 source-prefix traversal to freeze bounded
actual derived gate/up/down tensors for those explicit experts. The 2 GiB
fixture is an action-local qualification input, not a production cache. It then
feeds identical original tensors, shapes and batch order through the real
legacy/shared/shared/legacy collectors, including complete selected H/X CPU
return. Only input delivery is replaced during replay; source-forward and
routing/SwiGLU costs are excluded from those timings.

The first legacy arm has no retained CPU reference. Later arms retain that
same reference and record its byte footprint; memory comparisons must use
matched later contexts. Every arm checks source-row parity, independent compact
CPU storage and exact tensor/serialized payloads. Both-host bounded Netdata
records are retained. Energy estimates identify the actual GPU UUID, use
reported power rather than utilization, and refuse a work/J value when updates
are too sparse or stale. Any reported estimate is gross device energy including
profiler/observer and external load.

The exact reviewed command is retained in
`/mnt/shared/tessera-measurements/glm-shared-input-capture-20260907/bounded-replay-command-01.json`.
It binds the existing original token/source inputs and immutable producer
container, preserves the 104 GiB physical reservation, 102 GiB conservative
trip threshold, 8 GiB host-available floor and 92 GiB GPU subset. Root approved
one such PB measurement after the CPU/source review. No GPU result is recorded
by this commit. A separate full 512-sample candidate fit requires review of
that bounded A/B; the known failed full legacy H return is not repeated.

## Integration qualification, 2026-09-07

The candidate was integrated into PR #341 as `1ec2c07cbe`, on top of the
current-main streaming branch `be82b2f3`. The collector, workspace CLI, replay,
both new test files and ARM CPU wrapper are byte-identical to the previously
tested `45bb0cbbf48b396d17a739b75c9c954790003f16` snapshot. Architecture
integration preserves the complete prior stamp/body suffix and adds the shared
collector stamp without a conflict.

PB action `0dc6b54c9f0dae5d1a7ca6b68cb0592ed64071a3796a1bf8f4aad0b35ba34f53`
ran on DL380 with eight physical CPU cores, 12 GiB reserved memory, one native
thread per worker and explicit pinned `TESSERA_REPO`. Candidate, guard, row
counts, bounded telemetry, GLM streaming, source-transition and documentation
checks passed: **147 tests, two subtests, zero skips**, in 86.92 seconds.
Explicit compilation of the collector, workspace/replay and both new tests
also passed. This is CPU integration evidence, not a GPU performance result.

The terminal exit is zero, scope cleanup and release completed, and the
result/snapshot blob hashes and canonical receipt were independently checked.
Snapshot: `7904c94727da6fc4f5f7fc34a04b4590b93a7078`.
Result SHA256: `3766c057e5422d2a74ea7d8e029b3e2eebcab5079c82644d46e85b3dbf71e7cd`.
Receipt SHA256: `d430561502eab657892a86ed68563a64759a5c0c742d2293116529fc7de7129b`.
The complete command, environment, source hashes and admission/cleanup evidence
are in `/mnt/shared/tessera-measurements/glm-streaming-source-20260907/review-02/shared-input-integration-cpu.json`.
An initial implicit-host-pinned submission was withdrawn from ready before
execution and superseded by this portable action. The active bounded GPU
replay checkout was untouched; no full-prefix candidate run was submitted.


# Bounded real GLM collector replay — 2026-09-07

The shared-input collector is bitwise and serialization-exact on the 48 selected
real GLM projections. The recorded isolated collector runs take 13.069–13.205 s
with sharing versus 23.211–23.339 s for the legacy collector. Raw GPU profiles
show the duplicate wide Grams and hot-loop scoring-row transfers disappearing.
These timings include profiling and complete selected CPU H/X return; they
exclude original source forwards and routing/SwiGLU derivation. They are not a
full-prefix fit, full-model throughput, quality or serving result. Shared-arm
energy is unresolved at the monitor's 10-second cadence, so no work/J improvement
or energy ranking is claimed.

## Exact workload and completion

Reviewed source `d4a21a882799048c92259d5c5adba0b20d6aae25` executed as sealed PB
snapshot `fa7bb428f3b2dbb1f14e5df034b52b3db94b3283`. All seven checked
collector/harness/test/architecture/wrapper files are byte-identical to CPU
qualification snapshot `45bb0cbbf48b396d17a739b75c9c954790003f16`; only the
provenance document and PB closure metadata differ between whole trees.

PB `41c0afedbdd0c6b7966b312c31797b468ddeba58f0c10054ab5b5aa6f14e0621`
ran on Sparky's GB10 UUID `e76c7efc-c157-b1f4-1348-83e4eb5092f4`, Torch
2.13.0+cu130/CUDA 13.0, driver 595.84, kernel 6.17.0-1032-nvidia. The reviewed
immutable producer container, patched NFS module, original 512×512/B=1/seed-0
input binding and source fingerprints were checked before submission and by
the run. PB admitted six preferred CPUs, 104 GiB physical memory and a 92 GiB
GPU subset as a measurement with no prior GPU member. The 102 GiB conservative
guard and 8 GiB host-available floor remained active. Exit was 0, cleanup and
scope release completed, source binding was unchanged and no telemetry or
memory-guard failure occurred.

One original prefix traversal reached target layer 4 and completed all 512
target source forwards. It froze the actual BF16 gate/up/down tensors for
explicit experts 0–15 of `model.language_model.layers.4.mlp.experts`: 4096-wide
gate/up inputs and 2048-wide down inputs. The bounded fixture occupied
1,448,472,576 bytes below its 2 GiB cap. All selected scoring prefixes reached
512 rows by batch 91. Every arm consumed all original 512 input records in order
and returned 2,751,463,424 bytes of independent compact CPU H/X storage.
Source counts, maxima, all H/X tensor bytes and per-qname serialized payloads
matched across all four arms. This is selection-scoped equality over every
observed row; the H/count/max population was never capped at 512.

## Recorded timing and profiler evidence

| Arm | Wall seconds, including profiles and CPU return | Median completed-batch interval | Retained CPU reference at start |
|---|---:|---:|---:|
| Legacy 0 | 23.3394 | 43.105 ms | 0 |
| Shared 1 | 13.2050 | 23.934 ms | 2,751,463,424 B |
| Shared 2 | 13.0692 | 23.885 ms | 2,751,463,424 B |
| Legacy 3 | 23.2109 | 43.158 ms | 2,751,463,424 B |

The matched later traces show the mechanism directly:

- In the two row-filled batches, legacy executes 64 width-4096 Grams and 32
  width-2048 Grams; shared executes 32 of each. Actual GPU kernels fall from
  672 to 448, with kernel interval union 86.141 ms versus 47.726 ms.
- Across the cold two batches, `cudaMemcpyAsync` calls fall from 98 to 2:
  96 synchronous scoring-prefix copies leave the forward loop. The remaining
  two are the original token-input delivery. Matched host copy API interval
  union falls from 54.037 ms to 0.166 ms; physical GPU copy time is a separate
  quantity and is not equated with those host waits.
- All twelve cold, row-filled and full CPU-return traces are nonempty. Actual
  kernel categories are counted separately from GPU ProfilerStep annotations.
  Async kernel tails can outlive CPU step intervals; inclusive host calls and
  GPU intervals overlap and are never added into a fabricated total.

Peak CUDA allocated memory falls by exactly 832 MiB on this selection: 1 GiB
of duplicate H removed minus 192 MiB of device scoring prefixes. CUDA reserved
peaks remain identical at 57,438,896,128 bytes. With the same retained CPU
reference, sampled conservative cgroup-plus-CUDA-reserved peaks are 77.096 GB
and 77.162 GB for shared versus 77.218 GB for the later legacy arm. These do not
establish a material physical-memory saving. The first legacy arm lacks the
CPU reference and is unsuitable as a memory baseline. A full 288-expert fit
must still be measured through complete independent CPU output materialization.

Both-host telemetry is retained. Mean Sparky CPU non-idle is 13.20% and 11.78%
for shared versus 10.66% for the matched later legacy arm; Lina's corresponding
1.72%, 1.73% and 2.10% are external-load context. The legacy arms have two
in-arm power updates each and coarse gross estimates of 964.595 J and
843.099 J. Each 13-second shared arm has only one in-arm update, so the declared
energy rule correctly emits null. This practical limit prevents an energy
ranking; it does not invalidate exact arithmetic/ownership checks. No GPU
utilization percentage is used.

## Sealed evidence and remaining gate

All artifacts live under
`/mnt/shared/tessera-measurements/glm-shared-input-capture-20260907/`:

- `cpu-final-01.json`: 109 CPU tests, zero skips, two subtests, explicit compile,
  exact tested source/bundle/CAS verification.
- `bounded-replay-command-01.json`, `gpu-preflight-01.json`,
  `gpu-submission-01.json`, `gpu-admission-01.json`: exact reviewed command,
  source/runtime checks and actual admission.
- `gpu-terminal-verified-01.json`: exit/cleanup, whole GPU-source bundle,
  CPU-to-GPU file equality, small output hashes and CAS receipt. GPU payload
  SHA256 `44526036cb5d874a8c3c2d2a48b07a17647837215419e70dce2c83af5c6284ff`;
  canonical receipt `b01f3894f00e6964a4ee244b0c50376729e9e0d8e9cbe3fed004bdc412e96aa1`.
- `real-routed-replay-01/`: actual result, input manifest, four-arm report,
  twelve traces, bounded both-host Netdata and continuous physical-guard data.
- `replay-analysis-01.json` and `replay-analysis-verified-01.json`: raw trace
  categories/shapes/interval unions, matched sampled memory and host context,
  with every trace and telemetry file sealed. PB CPU analysis
  `52b813773d76b18444d6536c045e3cb0023d4c9ee8a83f056a26006387da8085` completed
  on DL380 with exit 0 and cleanup verified; twelve independent parsers used
  the admitted CPU action. No giant trace was parsed on the GPU host during
  measurement.

The option remains default-off. Root review of this bounded result is required
before the separate full 512-sample candidate fit. No repeat of the failed full
legacy H return and no full-prefix-05 run were submitted by this work.
