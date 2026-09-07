# LFM packed joint AURA screen, 2026-09-07

Issue [#313](https://github.com/RobTand/prismaquant/issues/313). Streamed joint
AURA now observes the original packed expert source operators and emits the
existing per-Linear currency for all 96 w1/w3/w2 projections in LFM layer 2.
This measurement covers **only canonical calibration row 0, 512 tokens, four
Rademacher probes**. It does not establish full-calibration quality, measured
KL, serving quality, or a performance improvement.

## Source and projection contracts

The implementation extends `PackedExpertProjection`, the existing streamed
runner, and `SignedJointProjectionLease`. It refreshes 2D views after each
source install and unload, harvests one gradient per physical packed leaf,
and observes original `F.linear` slices or `F.grouped_mm` transposes and expert
row offsets. It preserves the source model, expert dispatch, routing, and
forward outputs. Unsupported boundaries fail closed. Output cotangents split
both expert rows and fused gate/up columns before applying each logical
Linear's weight, activation, and mixed residual terms. Repeated invocations
sum while signed within that Linear before squaring. Different Linears remain
separate unary prices under the existing additive objective.

A separate lifetime fix releases the completed joint lease and delta tensors
before the next layer install. A separate identity fix binds each module's
resolved attention/expert backend, including private Transformers selectors
omitted by `config.to_dict()`. Backend changes reject checkpoint resume before
source installation; changes during measurement reject the layer before its
checkpoint rows are written.

## Frozen inputs

- BF16 source: `/mnt/shared/models/LFM2.5-8B-A1B-BF16`, 24 layers, 32 experts,
  top-k 4. Checkpoint SHA256:
  `c9b9e3c4b3be50b576e6da8c02de1b4223614ffe131d812abf92bb84421f6217`.
- Parent calibration: canonical int64 `[512,512]` artifact at
  `/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/canonical/calibration_tokens.safetensors`,
  SHA256 `a38312e3b1eeecc2a4363d2a91739ba535388036d4910bffa815d451dcb9a940`.
  WT2 train corpus SHA256:
  `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`.
- Actual operator subset: sole int64 `[1,512]` row 0 at
  `/mnt/shared/tessera-measurements/pq313-packed-joint-20260907/operator-subset/calibration_tokens.safetensors`,
  file SHA256 `ba4482b67fa9256ce48579182504a94cfc5d29f8d0057337b22ca303373c7cc7`;
  raw int64 bytes SHA256
  `24b78ebefe086cc05dd11849766a543db3e23569a679cd201fa55a16221b3017`.
  The subset metadata retains parent corpus/source/seed and explicitly sets
  `nsamples=1`, `fit_tokens=512`, parent file/hash/row and screen scope.
  Its int32 fit-token hash is
  `b1622ee77efe13444af2d8f0dff5d156176ce8395a954fabb65a1334bc28f4af`.
- Original decoded production cache:
  `/mnt/shared/tessera-measurements/first-model-20260907/native-moe-wires-r1024/production.pkl`,
  SHA256 `6a07cb3431476dd41a2a12a55a999cef866afddec9b1e69d3d9664eaacec9127`.
  These are the 96 existing projected E4M3 K1 R1024 wire renders fitted using
  the full canonical Hessian contract. This screen does not re-encode them.
- Probe seeds 7000–7003, all-token scope, temperature 1, Rademacher distribution,
  global KL Fisher normalization. Attention is eager; experts remain the
  canonical Transformers `grouped_mm` backend. `PRISMAQUANT_PROD_ACT_SCALES=0`
  matches the explicit native comparison protocol.

## Actual results

The ordinary Hugging Face reference and streamed source agree bit for bit on
all `[1,512,128000]` BF16 logits. For each of four probes, both full packed
leaf gradients agree bit for bit: gate/up `[32,3584,2048]` and down
`[32,2048,1792]`; all eight comparisons have maximum absolute error zero.
The shared `derive_per_expert_activations` down inputs also agree bit for bit
with actual grouped source inputs after aligning token/top-k-slot order:
`[2048,1792]`, zero differing elements, SHA256
`29050e797f37e4aca7c43ef50ee110dc13d8e17f517ed1018c34bcd95e8046bb`.
This last comparison is scoped to the actual row-0 layer-2 operator inputs.

The final cost payload contains 96 E4M3 rows and 96 BF16 zero controls, all with
four aligned signed samples and source/render/activation/probe identities.
Expert 19 receives no routes in this sequence; its three source rows correctly
retain aligned zeros. The other 93 E4M3 rows have nonzero prices. Their additive
predicted-loss sum is `0.001974112346254766`. This is the sum of the per-Linear
screen prices, not a whole-block coherent quality scalar or measured KL.
Per-row uncertainty is conditional on these fixed calibration tokens.

Final cost artifact:
`/mnt/shared/tessera-measurements/pq313-packed-joint-20260907/cost-02/joint-cost.json`,
SHA256 `fcf48abdb0e0b1d8bc324c5c938b9b68ebb821f1c5b0591aefffb4b7c3c45462`.
Its producer package SHA256 is
`7265e47475ded17f10b542874c289b054a1cd8c004b30d247fb510b9eeeb6405`.
The cost consumer and wire producer have separate exact source identities;
actual cached render tensors and source weights are independently hash-bound.

## Validation and retained evidence

All tests and GPU work ran through PrismaBuild. GPU runs used the existing
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
container with Transformers 5.16.1 and PyTorch 2.13.0+cu130 on GB10. GPU actions
reserved four CPUs, 40 GiB aggregate memory and 32 GiB GPU memory; native
threads were bounded to one and PB affinity was preserved. CPU checks used
2–4 workers. The final selected suite passed 63 tests, followed by two targeted
checks for grouped QDQ failure cleanup and backend mutation before publication.
The latter overlaps one test from the 63-test suite; there are 64 distinct
final checks, with no skips. Scoped pytest-xdist 3.8.0 and execnet 2.1.2 supplied
parallel CPU execution in the existing image.

| Evidence | PB action | Outcome |
|---|---|---|
| Lifetime and packed support regression | `567ebd4a94a94c404a4df913a57448582c14dc2e2d664a03ba22e0c4d3b9d7ec` | Three expected pre-fix failures |
| Lifetime fix and existing joint checks | `910381a7f145ef54f93b10e3c8d421272e8f24b9630dd89f033f8963d3ee72fe` | 15 passed |
| Packed views, resume and failure cleanup | `6c48859f61f99ec212dee915980837435d024a6f008041bb7de8a68c514503b9` | 24 passed |
| Grouped-boundary regression | `1b8d78e28352812432a6f6db8e7d81bc44983a4d5713e204f52894890af0e757` | Expected pre-fix refusal |
| Canonical source forward/reverse/down-input proof | `9c91cb5cf200df4b3f63a4b7fbe7e1464c532a330f180523b70c101ca3963a14` | GPU rc0, all exact comparisons |
| Backend resume regression | `8f08b0728520b4ee0df023e07740b09cb2fb8ca681a766bc6d2d89bb26e171b0` | Two expected pre-fix failures |
| Final selected source/joint/doc checks | `584254592eb3833a8bdb09466b2559406d31c5d4fe6adaefb27e629adc3b117e` | 63 passed |
| Backend mutation and grouped QDQ cleanup | `92d4c7353b5af7b0394fb56dc06988b04fb577394dda9d0b641e036bdd981d01` | 2 passed |
| Final original-wire joint costs | `3398c8e2b6880ec84783bb9a02ddd6c2a0417662140aa6cbbe0497a0800d1259` | GPU rc0, 192 valid rows |

The final GPU CAS receipt is
`d2e8823057de794c124a12e6684c51f8cf315aa53df013c069962c4418679a37`.
The evidence directory retains terminal summaries, logs, verified CAS result
receipts, snapshot provenance, harnesses, cProfile files and both hosts' Netdata
series. The checked-in companion JSON hashes those artifacts. Profiling here
records correctness-run activity; the runs were not isolated performance arms
and establish no speedup or work-per-joule comparison.

The first actual source attempt (`39c37...`) stopped at the unsupported grouped
boundary before gradient collection. An explicitly eager expert diagnostic
(`f45f...`) passed but is superseded and never joined to the canonical grouped
capture. `cost-01` passed numerically; `cost-02` supersedes it with explicit
backend identities, and every signed sample is exactly unchanged. Two early
parallel CPU attempts (`d74f...`, `4f66...`) collected no tests because xdist was
missing; the scoped install and later successful runs supersede them. An earlier
mistyped test filename was corrected and the actual namespace/doc suite passed.
The first metadata-free subset file was superseded by the metadata-complete
`operator-subset` artifact above. All superseded data remain bounded evidence.

## Remaining scope

The full 512-sequence draw has not been priced by this screen. Full-draw
execution needs deterministic calibration microbatching within the shared
runner/AURA checkpoint path, with global probe indexing/normalization and
per-Linear signed accumulation across partitions before squaring. Whole-draw
logits and Fisher tensors must not be materialized at once. Native whole-MoE
runtime measurements bind the 96-row roster separately; this cost result alone
does not qualify a serving lane or change any ship gate.

## Independent retained-boundary qualification

A subsequent source-only run qualifies the retained first canonical MoE boundary
without changing or recapturing it. PB action
`e46ba7cf0b4fd2f38ed6a127bee4a3fb78010b73b223f54756eec66b1626e1fe`
finished rc0. The proof at
`/mnt/shared/tessera-measurements/pq313-packed-joint-20260907/source-04/results.json`
has SHA256 `f308e7f6fbf563884d9625721523f4247322e0bfa95876e04798438553ce8558`.
It binds the original raw boundary artifact SHA256
`1290dd0dcb4ebecd09aee9fd2427aae62cb9c09974a466c05dcbe1ff08c6874d`,
retains that artifact's original metadata, and independently compares new source
input activations, top-k IDs, top-k weights, FP32 expert bias, and token
coordinates. Every tensor is bit-exact with matching dtype and hash. The
reference descriptor is captured before its first forward; all 49 resolved
module backend descriptors match the streamed model. The run also repeats and
passes full-vocabulary forward, all eight packed gradient comparisons, and
the shared-derived versus actual grouped down-input comparison.

The retained cProfile files have an attribution limitation discovered during
separate inspection: some call edges contradict the source call graph, including
an apparent recursive `assignment_keys` edge. These files must not support
function-level timing or bottleneck claims. The failure mechanism is not
established. Future performance comparisons require reliable per-thread
instrumentation or an independent sampler alongside both hosts' telemetry.
The tensor comparisons, terminal outcomes, CAS results and Netdata artifacts
remain separately attributable; this screen makes no speed claim.
