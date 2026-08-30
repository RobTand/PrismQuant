# QTIP-informed optimizer, native NVFP4 output

Status: **implemented and measured Arm C research-only one-Linear isolate,
plus a physical transformed-basis PrismaQuant/Gridbook trellis producer; no
production registration, QTIP runtime, or new serving lane.** Arm C tests
QTIP-derived BlockLDLQ with a stock native-NVFP4 terminal. The combined
producer emits canonical `gridbook.trellis.wire.v1` bytes and validates the
online sign/Hadamard algebra against an external, unpinned Gridbook research
reference outside the current Gridbook 0.9.1 contract. Schema v2 requires a
fresh no-clobber result root and a receipt that binds both source trees,
inputs, calibration, container, host, GPU/driver, command, and quality levers.

Pinned reference: official QTIP checkout
`/home/rob/dq-runs/qtip-reference-20260830` at
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`. The receipt pins SHA-256 digests
for `math_utils.py`, `ldlq.py`, `finetune.py`, and `bitshift.py` as well.

## What transfers

- QTIP's block-unit-lower factor, reverse BlockLDLQ schedule, and later-block
  error feedback are offline optimizer choices. The harness transfers that
  recurrence. It deliberately uses PrismaQuant's activation preprocessing and
  `damp=1.0`, not QTIP's default `sigma_reg=0.01`, so all controls share the
  same local objective.
- QTIP's random signs, input/output Hadamards, and `SU`/`SV` factors require
  inverse transforms during execution. They cannot decorate one native-NVFP4
  Linear without runtime sidecars. PrismaQuant now has a strict research
  producer for Gridbook's `gridbook.qtip-online-hadamard.v1` sidecar and its
  existing trellis wire; the stock-native Arm C isolate intentionally remains
  unrotated so its mechanism stays attributable.
- QTIP's tail-biting Viterbi codebook correlates choices across a 256-element
  tile and serializes trellis state. Native NVFP4 independently decodes each
  E2M1 nibble under one E4M3 scale per group of 16, so the QTIP trellis is not
  representable as standard native bytes. That QTIP wire is excluded and is
  not decoded by Gridbook. The combined research producer retains
  PrismaQuant/Gridbook's existing `TCQ_E2M1_R256` trellis wire while borrowing
  QTIP-derived optimizer and rotation ideas; it would not emit QTIP bytes.

Quartet II is literature motivation only: no Quartet II source, implementation,
or result is used here. EXL3 was source-reading context only and is neither a
dependency nor an implemented or measured arm.

## The NVFP4 scale byte

The group-scale byte is useful optimization capacity but not spare storage.
The stock decoder reconstructs every value as
`global_real * e4m3_group_scale * e2m1(nibble)`. Arbitrary payload bits in the
scale byte therefore alter all 16 reconstructed weights; there is no decoder-
invisible scalar channel. It already costs exactly 0.5 bits/weight. The legal
way to use more of that capacity is to search more E4M3 scale choices.

The harness reuses the exporter's existing search rather than creating a new
codec. Arm C uses the production FourOverSix candidates `{6, 4}`. Arm C2 uses
seven existing max-to-E2M1-level heuristics
`{6, 4, 3, 2, 1.5, 1, 0.5}`. This is **not** an exhaustive search over positive
E4M3 byte values, a Quartet II implementation, or a Quartet II reproduction,
and snapped-scale scoring remains disabled to match the production recipe. C2
therefore tests only whether those five extra existing PrismaQuant heuristic
candidates help the QTIP-derived recurrence at the same native payload rate.

## Matched arms

1. native NVFP4 RTN + production JSO, where JSO is only the final native
   group/tensor scale search (there is no GPTQ optimizer in this arm);
2. current PrismaQuant GPTQ + static activation order + JSO;
3. QTIP BlockLDLQ with a stock NVFP4 terminal at every reverse group-16 step;
4. C2, the same recurrence with the seven-level scale heuristic, not an
   exhaustive E4M3 scalar-channel search.

All arms emit only `weight_packed`, `weight_scale`, `weight_global_scale`, and
`input_global_scale`. Exact payload bpw includes both float32 scalars. The
metric is a BF16/FP32 activation-Hessian weight isolate, not a served W4A4 or
full QTIP model-quality claim.

The Arm-C receipt labels the ordinary activation-output proxy and the damped proxy
as **untransformed original-Linear space**. No transformed-Hessian number is
reported because this stock-native isolate has no online rotation. The
separate combined producer transforms both weight and Hessian consistently
and proves the post-decode serve algebra. Its runtime transform exists only in
an external, unpinned Gridbook research branch and is not a production
contract.

```bash
python -m research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq \
  --weight weight.safetensors --weight-key model.layers.0.self_attn.q_proj.weight \
  --activations q_proj_inputs.pt --activations-key inputs --device cuda \
  --output /run/qtip-native-v2/receipt.json \
  --artifacts-dir /run/qtip-native-v2/native-fields \
  --profile-dir /run/qtip-native-v2/profile \
  --publication-root /run/qtip-native-v2 \
  --durable-root-uri sparky:/absolute/host/path/qtip-native-v2 \
  --host sparky --container-identity image@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --model-id Qwen/Qwen3-0.6B --calibration-manifest calibration.json \
  --prismaquant-checkout /code --prismaquant-commit 0123456789abcdef0123456789abcdef01234567 \
  --qtip-checkout /path/to/qtip
```

The calibration manifest must use
`prismaquant.calibration_identity.v1`, carry non-empty `dataset`,
`capture_precision`, and `calibration_hash` fields, integer `nsamples`,
`seqlen`, and `seed`, and include `identity_sha256`, the canonical SHA-256 of
all other fields. The code and QTIP Git metadata must be readable inside the
container. Every output must resolve beneath `publication-root`. Files publish
individually without clobber; the receipt publishes last and is the commit
marker. A directory containing fields or a trace but no receipt is incomplete
and must be abandoned rather than resumed in place.

## Accepted schema-v2 one-Linear result (Sparky, 2026-08-30)

The accepted r3 run uses Qwen3-0.6B BF16
`model.layers.0.self_attn.q_proj.weight`, shape `[2048, 1024]`, and the same
256 FP32 UltraChat activation rows as the historical isolate. The publication
binds PrismaQuant commit `64679a7`, QTIP commit
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`, the calibration identity,
container, command, device/driver, exact quality environment, field hashes,
Torch trace, py-spy profile, and both-host Netdata series. Every arm is exactly
4.500030517578125 bpw over 2,097,152 quantizable weights.

| Arm | activation-output NSSE | regularized-H NSSE | weight SNR | bpw |
|---|---:|---:|---:|---:|
| A native RTN + JSO | 0.002579497 | 0.003893172 | 21.1268 dB | 4.5000305 |
| B current GPTQ + static order + JSO | 0.002579497 | 0.003893172 | 21.1268 dB | 4.5000305 |
| C QTIP BlockLDLQ + native terminal | **0.001043210** | **0.002936581** | 20.7342 dB | 4.5000305 |
| C2 C + seven-level scale heuristic | **0.001043210** | **0.002936581** | 20.7342 dB | 4.5000305 |

Against A/B, C reduces activation-output NSSE by **59.56%** (+3.9316 dB)
and the regularized-H proxy by **24.57%** (+1.2246 dB), despite losing 0.3925
dB of weight-only SNR. The result therefore isolates the value of error
direction under a non-diagonal activation metric. C2 is byte-identical to C:
the five additional max-to-level scale heuristics add no value on this
Linear. This is not an exhaustive legal E4M3 scale search.

The raw receipt is
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7/receipt.v2.json`
(SHA-256
`77912b7def1a0951dbeb067c50d0d617bd2458945ffd8170938d480828d1a54b`).
The independent acceptance receipt is
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/acceptance.v2.json`
(SHA-256
`98ad857d0addac04d22c318bbd0dffff3fb62cebacf83b35dda356b2208cd8ca`).
Its Netdata manifest SHA-256 is
`3026abf638434b34492c0ea55626959a74289f770b2de15a3a408bad4bc9aabb`.
The approximately 23-second isolate is quality evidence only; it is not a
throughput or work-per-joule benchmark.

## Gridbook online-transform reference

The separate Gridbook research branch at commit
`84a78c745e53676f87397e937456d7f2fc6ddd3f` implements strict
`gridbook.qtip-online-hadamard.v1` metadata and reference transforms without a
core-vLLM patch or PrismaQuant import. At M=32, K=N=4096, block size 256, and
20,000 iterations on Sparky, the input transform measured 0.150208 ms median
and the inverse output transform 0.153952 ms median; the pair adds 0.304160 ms
in this launch-heavy Torch reference. Pqteld measured 21.104 W mean, 26.33 W
maximum, 166.57 J, 240.13 transforms/J, and 31.475 million values/J. This is a
transform-only reference—not a fused kernel, graph, GEMM, or serving result.
The authoritative directory is
`/mnt/shared/dq-runs/gridbook-qtip-hadamard-ref-84a78c7-20260830T1747EDT-r5`.

The PrismaQuant combined producer uses that exact metadata contract, emits
canonical `gridbook.trellis.wire.v1` bytes, reparses and decodes those same
bytes, and proves original-basis serve algebra. It remains one-Linear,
unregistered, unpinned, and production-ineligible.

## Preliminary v1 one-Linear result (Sparky, 2026-08-30; superseded)

The following run discovered a promising seam, but its v1 receipt did not bind
the PrismaQuant source commit, full calibration identity, container/device
identity, or all quality-affecting configuration. It is retained as historical
evidence only and is **not an accepted v2 result**; r3 above supersedes it.

Source was Qwen3-0.6B BF16
`model.layers.0.self_attn.q_proj.weight`, shape `[2048, 1024]`. The 256 FP32
activation rows came from the existing UltraChat calibration capture
(`nsamples=8`, `seqlen=512`, BF16 capture, calibration hash
`41d25e036f3aaea2d3dafcf37831018a`). All four artifacts contain 2,097,152
quantized weights and exactly 1,179,656 payload bytes: 4.5000305 bpw including
the two float32 scalars.

| Arm | output NSSE on X | damped H proxy NSSE | weight SNR dB | payload bpw |
|---|---:|---:|---:|---:|
| A RTN + final JSO | 0.002579497 | 0.003893172 | 21.1268 | 4.5000305 |
| B GPTQ + static order + JSO | 0.002579497 | 0.003893172 | 21.1268 | 4.5000305 |
| C QTIP BlockLDLQ + native terminal | 0.001043210 | 0.002936581 | 20.7342 | 4.5000305 |
| C2 C + seven-level scale heuristic | 0.001043210 | 0.002936581 | 20.7342 | 4.5000305 |

On this calibration isolate, C reduced activation-output NSSE by **59.56%**
(+3.93 dB) and the exact regularized-H proxy by **24.57%** (+1.22 dB) versus
A/B, while weight-only SNR fell 0.39 dB. B's final bytes equal A's; v1 did not
record a pre-gate candidate or a gate counter, so it cannot prove why. C2's
final field digest equals C's, so the five additional max-to-level heuristics
found no gain here; this says nothing about exhaustive legal E4M3 scale-byte
search. This is preliminary evidence for the QTIP feedback seam, not held-out
KL/PPL or a served-quality claim.

Preliminary v1 receipt, superseded by the v2 evidence contract above:
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-20260830/qwen3_0p6b_l0_qproj.json`
(SHA-256 `2cb266f637e6bcf2cd60cbdd63490a62c00d8d746bc59b1d08eac663464edba4`).
The original stock-field safetensors are beside it under `fields/`, but were
published by a root container as mode `0600`; v2 normalizes artifacts to
`0644` and records relative plus durable paths and file hashes. The original
in-process trace is `profile/one_linear_trace.json`. Queue log:
`/mnt/shared/pq-queue/logs/qtip-native-nvfp4-qwen06-l0q-v4.log`. No timing or
energy claim is made from this run.
