# QTIP-informed optimizer, native NVFP4 output

Status: **research-only one-Linear isolate; no production registration, QTIP
runtime, or new serving lane.** It tests which QTIP mechanisms survive when
the final artifact must remain vanilla native NVFP4.

Pinned reference: official QTIP checkout
`/home/rob/dq-runs/qtip-reference-20260830` at
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`. The receipt pins SHA-256 digests
for `math_utils.py`, `ldlq.py`, `finetune.py`, and `bitshift.py` as well.

## What transfers

- QTIP's Hessian regularization, block-unit-lower factor, reverse BlockLDLQ
  schedule, and later-block error feedback are offline optimizer choices. The
  harness transfers that recurrence, while using PrismaQuant's existing
  activation preprocessing and damping so all control arms share one contract.
- QTIP's random signs, input/output Hadamards, and `SU`/`SV` factors require
  inverse transforms during execution. They cannot decorate one native-NVFP4
  Linear without runtime sidecars. A separately proved model-wide fold could
  be tested only as a Gridbook arm; this isolate intentionally does not claim
  or implement one in PrismaQuant.
- QTIP's tail-biting Viterbi codebook correlates choices across a 256-element
  tile and serializes trellis state. Native NVFP4 independently decodes each
  E2M1 nibble under one E4M3 scale per group of 16, so the QTIP trellis is not
  representable as standard native bytes. Serving that terminal would require
  the existing Gridbook codebook runtime, not this stock-native lane.

## The NVFP4 scale byte

The group-scale byte is useful optimization capacity but not spare storage.
The stock decoder reconstructs every value as
`global_real * e4m3_group_scale * e2m1(nibble)`. Arbitrary payload bits in the
scale byte therefore alter all 16 reconstructed weights; there is no decoder-
invisible scalar channel. It already costs exactly 0.5 bits/weight. The legal
way to use more of that capacity is to search more E4M3 scale choices.

The harness reuses the exporter's existing search rather than creating a new
codec. Arm C uses the production FourOverSix candidates `{6, 4}`. Arm C2 uses
the exporter's existing full grid `{6, 4, 3, 2, 1.5, 1, 0.5}`. Both serialize
the same 4.5-ish bpw native fields, so D directly tests whether fuller use of
the semantic scale byte helps the QTIP-derived recurrence.

## Matched arms

1. native NVFP4 RTN + production JSO, where JSO is only the final native
   group/tensor scale search (there is no GPTQ optimizer in this arm);
2. current PrismaQuant GPTQ + static activation order + JSO;
3. QTIP BlockLDLQ with a stock NVFP4 terminal at every reverse group-16 step;
4. C2, the same QTIP-derived recurrence with the existing full JSO scale grid.

All arms emit only `weight_packed`, `weight_scale`, `weight_global_scale`, and
`input_global_scale`. Exact payload bpw includes both float32 scalars. The
metric is a BF16/FP32 activation-Hessian weight isolate, not a served W4A4 or
full QTIP model-quality claim.

The receipt labels the ordinary activation-output proxy and the damped proxy
as **untransformed original-Linear space**. No transformed-Hessian number is
reported because this stock-native isolate has no online rotation. Any later
rotation arm must transform both weight and Hessian consistently and belongs
to the explicit Gridbook follow-on.

```bash
python -m research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq \
  --weight weight.safetensors --weight-key model.layers.0.self_attn.q_proj.weight \
  --activations q_proj_inputs.pt --device cuda --output receipt.json \
  --artifacts-dir native-fields --profile-dir profile \
  --qtip-checkout /path/to/qtip
```

## First matched one-Linear result (Sparky, 2026-08-30)

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
| C2 C + existing full JSO grid | 0.001043210 | 0.002936581 | 20.7342 | 4.5000305 |

On this calibration isolate, C reduced activation-output NSSE by **59.56%**
(+3.93 dB) and the exact regularized-H proxy by **24.57%** (+1.22 dB) versus
A/B, while weight-only SNR fell 0.39 dB. B's final bytes equal A's because the
existing do-no-harm gate rejected its candidate on this Linear. C2's final
field digest equals C's, so searching the wider legal scale grid found no
additional scale-plane gain here. This is encouraging evidence for the QTIP
feedback seam, not held-out KL/PPL or a served-quality claim.

Immutable receipt:
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-20260830/qwen3_0p6b_l0_qproj.json`
(SHA-256 `2cb266f637e6bcf2cd60cbdd63490a62c00d8d746bc59b1d08eac663464edba4`).
The stock-field safetensors are beside it under `fields/`; the in-process
trace is `profile/one_linear_trace.json`. Queue log:
`/mnt/shared/pq-queue/logs/qtip-native-nvfp4-qwen06-l0q-v4.log`. No timing or
energy claim is made from this run.
