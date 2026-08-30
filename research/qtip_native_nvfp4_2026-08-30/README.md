# QTIP-informed optimizer, native NVFP4 output

Status: **implemented Arm C research-only one-Linear isolate; no production
registration, QTIP runtime, or new serving lane.** It tests QTIP-derived
BlockLDLQ with a stock native-NVFP4 terminal. It is not the combined rotated
PrismaQuant/Gridbook trellis producer, which does not exist. An external,
unpinned Gridbook research reference implements the online-transform runtime
seam, but it is outside the current Gridbook 0.9.1 contract. Schema v2 requires
a fresh no-clobber result root and a receipt that binds both source trees,
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
  Linear without runtime sidecars. A separately proved model-wide fold could
  be tested only as a Gridbook arm; this isolate intentionally does not claim
  or implement one in PrismaQuant.
- QTIP's tail-biting Viterbi codebook correlates choices across a 256-element
  tile and serializes trellis state. Native NVFP4 independently decodes each
  E2M1 nibble under one E4M3 scale per group of 16, so the QTIP trellis is not
  representable as standard native bytes. That QTIP wire is excluded and is
  not decoded by Gridbook. The intended combined follow-on would retain
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

The receipt labels the ordinary activation-output proxy and the damped proxy
as **untransformed original-Linear space**. No transformed-Hessian number is
reported because this stock-native isolate has no online rotation. Any later
rotation arm must transform both weight and Hessian consistently. The runtime
transform reference exists only in an external, unpinned Gridbook research
branch; the corresponding combined PrismaQuant producer and pinned contract
do not exist.

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

## Preliminary v1 one-Linear result (Sparky, 2026-08-30; superseded)

The following run discovered a promising seam, but its v1 receipt did not bind
the PrismaQuant source commit, full calibration identity, container/device
identity, or all quality-affecting configuration. It is retained as historical
evidence only and is **not an accepted v2 result**. A fresh run must use a new
no-clobber publication root after the active Sparky probe completes.

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
