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
