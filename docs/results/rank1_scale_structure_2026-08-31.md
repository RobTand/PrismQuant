# Where the scale bits should live: rank-1 diagonals vs a block plane

Qwen3-0.6B, 21 tensors (layers 4/13/22), `q256=768` (3.0 body bits, **zero
bypass**, so the bypass penalty found earlier cannot contaminate anything),
Hadamard block 128 where used, unweighted encode, importance-weighted SNR
scored in the original basis. Driver `scratchpad/rank1.py`, log
`dq-runs/trellis-serve-smoke-20260831/rank1.log`.

## The question

Robert: *"could we fuse many nvfp4 blocks together to allow us to do something
similar? My paramount concern is leveraging the native 4-bit hardware on the
blackwell chips"*, and then: *"because we're taking the block scalar out of
nvfp4, we could support other hardware much more easily."*

EXL3 carries **no weight-side scale plane at all**. It divides magnitude out
into two rank-1 diagonals and applies them around the GEMM:

```python
# exl3_lib/quantize.py:1208-1212
in_channel_scales = block_rms(weight, dim=1, keepdim=True)
su = (su * in_channel_scales / (-codebook_scale)).float()
weight /= su
```

`suh` folds into the input transform, `svh` into the epilogue
(`exl3_gemm_kernel.cuh:25,66`), and its Hadamard is 128 wide
(`hadamard_inner.cuh`: *"Hadamard transform 128-element vector across one
warp"*). Cost for a 1024x1024 Linear: `k+n = 2048` scalars, ~0.03 bpw, against
NVFP4's `n*k/16 = 65,536` E4M3 codes at 0.5 bpw.

## The measurement

| arm | scale bpw | total bpw | wSNR | vs base |
|---|---|---|---|---|
| fine plane, no rank-1 (today) | 0.5000 | 3.5000 | 16.67 | +0.00 |
| flat plane, no rank-1 | 0.0000 | 3.0000 | 7.14 | −9.53 |
| flat plane + rank-1 | 0.0234 | 3.0234 | 7.88 | −8.79 |
| flat plane + rank-1 + rotation | 0.0234 | 3.0234 | 12.11 | −4.56 |
| **fuse-16 plane + rank-1** | 0.0547 | **3.0547** | 15.87 | **−0.80** |
| **fine plane + rank-1** | 0.5234 | 3.5234 | **17.53** | **+0.87** |

## Three findings

**1. Rotation and the block plane are substitutes, and now it is measured, not
argued.** With the fine plane, rotation costs −0.51 dB. With the plane removed,
rotation is worth **+4.23 dB** (7.88 → 12.11). They do the same job; whichever
is present makes the other redundant. This is the mechanism behind every
rotation number in `qtip_rotation_weight_side_2026-08-31.md`.

**2. Rank-1 diagonals are the best-value bits in the format, and they are
additive.** `fine + rank-1` gains **+0.87 dB for +0.023 bpw** — about **37 dB
per bit**, against ~17.6 dB/bit for the group-16 plane and ~5 dB/bit for body
rate. They are worth adding to the current format **independently of any
portability decision**, and the operands already exist in spirit: the E2M1 lane
applies `global_scale_real` in the epilogue and `input_global_scale` on the A
side, both scalars today. Promoting those two to per-channel vectors *is*
`su`/`sv`.

**3. Rank-1 is what makes coarsening viable.** Fuse-16 alone cost −1.94 dB
(`qtip_rotation_weight_side_...` §3-bis companion sweep); fuse-16 **with**
rank-1 costs only **−0.80 dB** while saving **0.445 bpw**. Spending that on
shaped rate at ~5 dB/bit returns ~+2.2 dB, so **fuse-16 + rank-1 is a net win
of roughly +1.4 dB at matched bpw**. Rank-1 recovers 1.14 dB of the coarsening
loss for 0.023 bpw because channel-wise magnitude structure is exactly what
piecewise-constant spans along a row cannot express, and it is where LLM
outliers live.

## Why this bears on hardware portability

A group-16 block scale is a **hardware operand**: it needs a GEMM that consumes
a per-16-element E4M3 scale next to packed fp4, i.e. `block_scaled_ue4m3xe2m1`,
Blackwell sm120+. A flat plane plus rank-1 diagonals needs none of that — the
weight operand is pure packed fp4 and the diagonals are elementwise vector ops
in the input transform and epilogue. That runs on anything with a 4-bit dot
product.

And a flat plane is **still a legal NVFP4 operand** (set every group scale
equal), so one artifact can take two paths: feed the constant plane to the
Blackwell mainloop for native fp4 tensor cores, or strip it on hardware without
block-scale support. That is native 4-bit Blackwell support without being
NVFP4-*shaped*.

The price today is the −4.56 dB of the fully flat arm. How much of that is
scale structure and how much is the fixed E2M1 grid mis-scaled for a Gaussian
source is a separate, testable question and is not settled here.

## Scope

Weight-space only; prices W\*A16, not the attested W4A4 contract. One rung, one
model, 21 tensors, three layers. Rank-1 factors come from 3 rounds of
alternating RMS normalisation — EXL3 takes one `block_rms` pass per side, and
the choice of 3 is mine and is not tuned. No LDLQ. The ~5 dB/bit shaped-rate
figure used in the net-win arithmetic is recorded, not re-measured here, so the
"+1.4 dB net" is an indication rather than a measured matched-bpw result.
