# Campaign batching: input and journal qualification

PrismaQuant #300 consumes Tessera #386 (`382a1a97`) for the #275 campaign.
`--anchor-batch-size` is opt-in; the scalar default remains one. The campaign
keeps its adaptive pending-anchor schedule and groups compatible expert units
inside each admitted PrismaBuild action. It uses the existing resident source
weights, captured scoring rows, ActivationSource memo, production scorer and
ProductionWeightCache. Tessera owns batch encoding. No serving release pin or
wire recipe changes.

The scalar and batch adapters share one input guard. Batch calls carry separate
per-unit Hessian mappings, retain the scalar artifact name, and decode each
result through the existing wire reader. The campaign scores each unit with its
own served activation scale, then writes its normal cache/wire entries and
journal receipt. Batch width is an execution setting excluded from checkpoint
input identity; batch timing is explicitly apportioned among its units. Missing
batch API support refuses before model load. A width-one run keeps the previous
ten-anchor journal flush cadence.

## Correctness evidence

All execution used PrismaBuild. The initial three adapter/grouping regressions
failed on main `70e5f7a8` because the batch seam did not exist. The final new suite
passed all six tests on CUDA in the known producer image
`prismaquant-qwen38-producer:20260827-tf516-hf128`, Python 3.12 / Torch 2.13+cu130:

- Action `a25b803d46b5b611292a00d724bd7c7353d374f1b6b98f6daf2f50069ced6599`,
  Sparky, exit 0, six passes, no skips, 21.63 seconds.
- CAS receipt `0b811bd4d82ed8ce2a800101d92c10a951a5d8f8ab71eed9ffb66b632bdcb10e`.
- Actual `campaign.main()` on a routed fixture encodes seven source units with
  each unit's own Hessian. Batches of four and two reach the real producer.
  Wire bytes and numerical cost rows equal the scalar run. Resuming with width
  two performs no encode, and a tampered wire refuses resume.
- Separate tests retain distinct static activation scales, refuse missing H,
  split incompatible shapes/rungs, and refuse an old producer before loading.

The relevant distinct coverage totals 150 tests: the existing campaign suite
(46 CPU plus five CUDA), new batch suite (six CUDA), packed campaign (16),
fanout (33), journal merge (10), architecture (13), document staleness (six),
and stack sample costs (15). CPU runs explicitly skipped CUDA cases; all those
cases were subsequently executed on CUDA. The first combined GPU run's sole
failure was a new fixture replacing every registry row; restricting the fixture
to its target format corrected it. The initial real CLI fixture used 64-wide
weights, which the normal pricing contract correctly refused; it was widened
to 256 before qualification. An expensive CPU encode rerun was withdrawn through
PB and moved to the GPU. None of those attempts is performance evidence.

Receipts/logs for the final GPU gate are retained under
`/mnt/shared/tessera-measurements/pq300-batch-20260907/`.
The CPU reports are `/home/rob/tmp/pq-batch-contracts.json` and
`/home/rob/tmp/pq-batch-finalcpu.json`; their CAS receipts remain authoritative.

## Calibration preparation and its limit

Retained input: `/mnt/shared/tessera-measurements/tessera385-2026-09-06/units/`,
WikiText-2 train, 32 samples of length 512, seed zero. Both exact corpus and token
hashes matched before the new forward:

- text: `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`
- IDs: `809883cb91e1359d0b7e7829f4a7ab3ddeacd7eee75a66dd6866aa671a4f935d`

The current container forward routed no rows to layer-18 expert 6; the collector
refused, as it must (action `ba0a368758de697b`, exit 1). This is not a complete
campaign calibration. An explicitly bounded validation recapture excluded that
one unit and retained the other 31 expert w1 units plus dense layer-0 w1, using
the same collector and exact tokens. It saved 512 scoring rows per unit where
available, the existing export H/scales files, the token tensor and corpus text.

Action `539fa6315350c4182922d091f2faa33afca5a3b58f25973c87a0a99ee80d77d4`
completed on Sparklina with exit 0; receipt
`0134db4fd8b9574554164c8b84f0d3beef9f398896a991df4c9a7fe5b9ddc952`.
Files are in `/mnt/shared/tessera-measurements/pq300-batch-20260907/capture/`.
Dense H equals the retained H exactly. All 31 expert H tensors differ, despite
identical tokens and source weights; the largest absolute H-entry difference is
777.96484375. This establishes a forward difference, not its cause. Every count
and comparison is in `capture.json`. No old expert cost is relabelled as a
measurement on this new capture.

The original capture inherited the source identity's `fit_tokens_min=1`.
Its observed selected-scope minimum is 84; individual counts were recorded
correctly. `capture-scope-correction-v2.json` records that correction without
mutating files already consumed by measurements. The retained harness now
computes this field from fresh counts. This narrow scope does not demonstrate
full-model routing coverage.

## Real cost-row BF16 A/B

On Sparklina GB10, producer `382a1a97`, the same known Docker image,
Python 3.12.3 / Torch 2.13.0+cu130 / Triton 3.7.1, four PB-assigned CPU cores
and 16 GiB aggregate reservation, eight `[1792,2048]` positive-route expert
w1 units were measured at `TESSERA_BF16_K1_R1792`. Each arm used exactly the
same source weight, freshly captured Hessian and scored activation rows.
Inputs and activation factorizations were resident before timing. Each phase
executes real encode, decode, production cost scoring, ProductionWeightCache
write and wire persistence; calibration, model loading and journal publication
are outside this timing. CLI/journal behavior is separately covered above.

Both arms warmed up, followed by scalar / batch-eight / batch-eight / scalar.
All wire SHA-256 values, measured dloss, memory costs, scales and H-applied flags
were identical in every phase. Batch phases reached one eight-unit producer
call. This is a layer-local cost-row comparison, not an end-to-end KL or served
benchmark, and no serving gate is qualified by it.

| Arm/order | Seconds for 8 rows | Rows/s | GPU joules | Rows/J |
|---|---:|---:|---:|---:|
| Scalar / 1 | 31.54958 | 0.253569 | 2501.66 | 0.003198 |
| Batch 8 / 2 | 31.33940 | 0.255270 | 1979.42 | 0.004042 |
| Batch 8 / 3 | 31.32566 | 0.255382 | 2483.03 | 0.003222 |
| Scalar / 4 | 31.53244 | 0.253707 | 2569.75 | 0.003113 |

Throughput is essentially flat: approximately 0.6% separates these two-arm
means. GPU energy is integrated from Netdata's ten-second power samples,
linearly interpolated at phase boundaries. The series brackets each interval
without gaps, but each 31-second phase spans few power samples and batch-arm
energy itself varies substantially. The aggregate work/J ordering favors
batching in this run; this does **not** establish a repeatable energy gain.
Power averaged 63.2–81.5 W against GB10's approximately 140 W envelope.
Sparklina CPU activity averaged about 6.0% machine-wide, with 0.7–0.9% iowait.
Both boxes' CPU, memory, I/O and GPU power series are retained; the other box's
independent work is not charged to this action. GPU utilization is not used.

The measured action is
`3b12d6fa3f4c42d72d301e511661b87bd104845204b3bdb1254a6af3d384d7de`,
exit zero; CAS receipt
`fbcf392528ce1cfac35878e5d8d12481bc293eb3b5dee21126442f78fd317b6d`.
`BF16/results.json` contains every phase, environment, source hashes and wire
fingerprint; `BF16/netdata-both-hosts.json` contains raw series and integrals.

Initial paired cProfile passes are retained as a diagnostic limitation:
scalar accounts for 32.57 seconds of a 32.86-second wall interval, while batch
accounts for only 3.57 seconds of a 32.64-second wall interval. The incomplete
batch accounting cannot explain total wall time. Full native sampled profiles were collected separately (below); no
acceleration claim follows from the incomplete cProfile accounting.

All artifact-relative paths above are beneath
`/mnt/shared/tessera-measurements/pq300-batch-20260907/`. The `harness/` directory
retains exact reproduction helpers and their SHA-256 manifest; PB receipts
also identify the actual submitted checkout snapshot. `campaign.json` holds
the two isolated measurement actions and `native-profile-campaign.json` the
follow-up sampled-profile actions. Submit with the published `pbcampaign.py`
from the measurement host; use new output directories because the harness
refuses to overwrite an existing result.


## Real cost-row E2M1 A/B

The same eight-unit input, resident preparation, environment, four-core
reservation and cost-row timing scope were then measured at
`TESSERA_E2M1_K2_R896`. Wire bytes and all numerical cost fields again matched
exactly in both warmups, all four measured phases and both cProfile phases.
The first call warmed scalar for 102.83 seconds and batch-eight for 24.53
seconds; these are excluded from the table.

| Arm/order | Seconds for 8 rows | Rows/s | GPU joules | Rows/J |
|---|---:|---:|---:|---:|
| Scalar / 1 | 98.83112 | 0.080946 | 1973.32 | 0.004054 |
| Batch 8 / 2 | 23.77231 | 0.336526 | 467.66 | 0.017106 |
| Batch 8 / 3 | 23.74366 | 0.336932 | 536.65 | 0.014907 |
| Scalar / 4 | 98.83196 | 0.080945 | 1680.14 | 0.004761 |

The aggregate comparison is **4.1599× rows/s and 3.6378× rows/J** on this
workload. Scalar and batch arms each measured 16 total rows: 197.6631 versus
47.5160 seconds, and 3653.46 versus 1004.31 GPU joules. Energy uses the same
bracket-checked ten-second Netdata interpolation as BF16 and includes device
baseline power. It is GPU energy, not wall-plug system energy. The short batch
phases limit its precision; both batch observations nevertheless exceed both
scalar observations in work/J in this run.

Average GPU power was only 17.0–22.6 W; machine-wide CPU activity was about
6.0–6.1%, with 0.7–0.8% iowait. These measurements establish a family-specific
latency and energy benefit and substantial unused GPU power headroom. They do
not establish GPU saturation or identify its cause. The default remains one.

Action `c4c08a329ba4cbb86d299ee2d91382e467c724d60001159b5af332e8afafb7b9`
completed on Sparklina, exit zero, 509.72 seconds; CAS receipt
`e0ec6b724209b7744909cdc57d21269f99700314041c7f48e77747b01357fe5c`.
The raw measurements, paired cProfile files and both-host telemetry are under
`E2M1/`. Sampled native profiles remain a separate pair of admitted actions.


## Full-call native profile follow-up

A separate admitted profile action warmed both complete eight-unit arms and
sampled one scalar and one batch call at 100 Hz using py-spy native/Python
stacks. These profiled wall times are excluded from the throughput comparison.
Each profile receipt verifies its binary, output and log SHA-256 values and
positive sample count. This bounds memory without retaining an unbounded CUDA
event table. Native categories are inclusive and may overlap; they are sampled
host stacks, not GPU kernel durations. Per-thread coverage is explicit in
`profile-summary.json`.

BF16 produced 3,463 scalar and 3,242 batch main-thread samples, representing
34.63 and 32.42 sampled thread-seconds; timed cost-row walls were 34.19 and
32.66 seconds. The small window difference includes sampling/phase bookkeeping
and sampling resolution. Inclusive CUDA stream synchronization occupied 22.26
versus 21.25 sampled seconds. The dominant source callsite is
Tessera `window_viterbi.py:771`, `sse += float(final.sum())`, with 22.19 versus
21.16 sampled seconds. Inclusive `viterbi_window_fused` occupies 30.77 versus
29.01 sampled seconds. This broadly unchanged full-call profile supports the
flat throughput finding and replaces the incomplete batch cProfile account.
A synchronization wait includes outstanding GPU work; its duration alone is
not a promise of removable latency.

BF16 native action
`d9cacad92f402405ba1cdd0413e92bed4fc8eab0eaa0785710267e457df59534`
completed on Sparklina, exit zero, 145.57 seconds; CAS receipt
`551d095978f5d2a55c5eed1849df6b8770251487d2c30733c9dc188f7d5cb349`.
Raw profiles, verified profile receipts and both-host Netdata series are under
`BF16-native-profile/`.

The E2M1 cProfile already accounts for 100.40 of 100.54 scalar wall seconds
and 25.00 of 25.34 batch wall seconds. `viterbi_columns` calls fall from
2,048 to 256, with cumulative time 89.56 to 14.46 seconds. Its
`_TCQPlan.sse` host scalar read (`encode.py:573`) accounts for 89.10 to
14.38 seconds; unit-local refit work remains near 6.9 seconds. The producer's
`_run_joined` discards the returned scalar SSE. That is an attributable
follow-up seam for Tessera #385, reported to its owner; this integration does
not change the producer, and further speedup from suppressing the readback is
unmeasured. The full native E2M1 pair below confirms the reduction.


E2M1 native sampling produced 10,608 scalar and 2,814 batch main-thread
samples, representing 106.08 and 28.14 sampled seconds; profiled walls were
105.45 and 27.85 seconds. Inclusive `sse` time is 90.43 versus 14.98 sampled
seconds. CUDA stream synchronization totals 71.58 versus 16.38 sampled seconds;
the `sse` readback at `encode.py:574` is responsible for 66.24 versus 11.20
of those seconds. The next largest synchronization site, `_fit_lut:1522`,
remains 4.68 versus 4.63 seconds. This localizes the observed batching gain to
the joined trellis/readback path while showing that independent refit waits
persist. It does not measure the effect of a future asynchronous SSE change.

E2M1 native action
`192ab33b80faa3a251dab65fad7a5fda92834340bf308025c1ea31b9f2c867a9`
completed on Sparklina, exit zero, 272.18 seconds; CAS receipt
`2626f7830802671e6351d240eff23988543f94f75e5124f4bfa1e7834a1ac922`.
Raw profiles, verified profile receipts and both-host Netdata series are under
`E2M1-native-profile/`. Both hosts' telemetry brackets every measured and
profiled interval in both formats, without missing-series or gap errors.

All four performance actions and the CUDA correctness gate have terminal
exit-zero records and CAS receipts. The 48-file `evidence-manifest.json` binds
the capture correction, reproduction helpers, action receipts, results,
profiles and telemetry by SHA-256. Temporary checkout helpers were retired
after byte-identical copies were retained in `harness/`; the production branch
has no untracked helper files. The original zero-route failed capture remains
bounded negative evidence and does not qualify full-model calibration.

Evidence manifest SHA-256: `8b144abda1b4a9933070146e9a54a1c5dcbd62c0159bae7428184a9bafdd11e5`.

## Source-reference correction — 2026-09-07 05:25 UTC

A later full-vocabulary LFM parity diagnosis found that PrismaQuant's global
HF initialization no-op also suppresses Transformers 5.16.1's initialization
of nonpersistent RoPE buffers during `from_pretrained`. The capture above used
that ordinary loading path. Its activation rows, Hessians, route counts and
maxima therefore cannot qualify canonical source-model quality or a new native
operator panel. They must be regenerated after the shared loader correction.
Token IDs and source checkpoint weight bytes are unaffected.

The matched scalar/batch measurements remain evidence for the exact retained
inputs, encoded-byte equality and measured throughput/energy deltas described
above. They are not a canonical-model quality comparison. Original artifacts
and their hashes are retained unchanged.

Diagnosis action `c999ff211ab7` completed through PrismaBuild on GB10 using the
same producer image. Its actual full-vocabulary first-sequence comparison is
bit-exact between HF loaded with genuine initialization and the streamed model
with FP32 routing buffers and eager attention. Injecting the old reference's
invalid RoPE into that streamed path reproduces the old reference exactly;
matching its attention mask alone does not. Detailed evidence is retained at
`/mnt/shared/tessera-measurements/lfm-streamed-parity-20260907/diagnose-02/diagnosis.json`.
This bounded diagnosis establishes the loading defect; it does not replace
fresh full-calibration capture or final served quality validation.
