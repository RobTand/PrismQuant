# Incoherence processing on the weight side: a cost, monotone in block size

Qwen3-0.6B, 21 tensors from layers 4/13/22, `q256=896` (3.5 body bits),
weight-space importance-weighted SNR scored in the **original basis**. Driver
`scratchpad/rot.py` / `blk.py`; logs under
`dq-runs/trellis-serve-smoke-20260831/`.

## 1. Rotation costs quality on the weight side

All arms unweighted, because **a per-column importance vector cannot survive
rotation**: under `R_in` a diagonal importance becomes `R_in diag(w) R_in^T`,
which is dense, not a vector. Comparing a weighted unrotated arm against an
unrotated one would credit rotation with the loss of the objective itself.

| q256 | body b/w | trellis weighted | trellis unweighted | trellis rotated | rotation | NVFP4 | NVFP4 rotated | rotation |
|---|---|---|---|---|---|---|---|---|
| 768 | 3.00 | 17.93 | 16.67 | 16.06 | **−0.61** | 20.11 | 18.87 | **−1.24** |
| 896 | 3.50 | 18.85 | 18.25 | 17.19 | **−1.06** | 20.11 | 18.87 | **−1.24** |

Importance weighting is worth **+1.26 dB at 3.0** and **+0.60 dB at 3.5**, so
adopting rotation means giving that up unless something replaces it.

## 2. The damage is monotone in Hadamard block size

`q256=896`, unweighted, same tensors. Baselines: trellis 18.25 dB, NVFP4
20.11 dB.

| block | trellis Δ | NVFP4 Δ |
|---|---|---|
| 16 | −0.77 | −0.72 |
| 32 | −0.84 | −0.65 |
| 64 | −0.81 | −0.86 |
| **128** | **−0.93** | **−1.04** |
| 256 | −1.06 | −1.24 |
| 512 | −1.13 | −1.36 |

## 3. The mechanism, and why it is specific to a native-NVFP4 target

**Incoherence processing assumes a quantizer with a coarse scale. NVFP4 has a
per-group-16 scale, which is itself an adaptive mechanism.** Unrotated weights
inside a 16-lane group are typically smooth and correlated, so the group has low
internal dynamic range and its scale exploits that; a lone outlier in one group
is handled perfectly and costs its neighbours nothing. A *large* rotation block
redistributes outliers across many groups, equalising every group amax and
throwing that advantage away. A block of 16 mixes only within one scale group.

The monotone table is that prediction confirmed. It also means the two
mechanisms are in direct competition on a native NVFP4 target in a way they
would not be on a per-tensor-scaled format — which is exactly the setting QTIP
was designed for.

**A tension to carry forward:** the native warp kernel dispatches **only** at
`block_size=128` (`docs/QTIP-NATIVE-NVFP4-RESEARCH.md`), where the weight-side
cost is ~1 dB. If the activation side prefers larger blocks, block size is a
tuned parameter with opposing gradients, not a constant.

## 3-bis. The damage tracks BYPASS fraction, not bit rate

I predicted the delta would shrink monotonically toward low rate and reverse to
a gain somewhere near 2 bits, on the reasoning that low-rate quantizers care
most about outlier control. **That prediction was wrong**, and the real pattern
is more useful. Same tensors, unweighted, delta = rotated - unrotated:

| q256 | body bits | bypass b/w | blk 16 | blk 128 |
|---|---|---|---|---|
| 512 | 2.00 | 0.0 | −0.39 | −0.69 |
| 768 | 3.00 | 0.0 | **−0.27** | −0.51 |
| 896 | 3.50 | **0.5** | −0.77 | −0.93 |
| scalar NVFP4 | 4.0 | (all scalar) | −0.72 | −1.04 |

Not monotone in rate. The least-damaged rung is 3.0 -- fully shaped, zero
bypass. The most-damaged trellis rung is 3.5, the only one carrying uncoded
bypass, and pure scalar NVFP4 is worst of all.

**Rotation hurts scalar quantization, not trellis shaping.** Bypass positions
are raw E2M1 codes, so they take the full scalar penalty; genuinely shaped codes
barely notice, at −0.27 to −0.39 dB. That matters because the band where the
trellis is defensible at all (≤3 bits, zero bypass -- see the shaped-rate
ceiling result) is exactly the band where rotation is nearly free.

Block 16 beats block 128 at every rung, and the native warp kernel dispatches
only at 128, so the current kernel geometry carries a standing ~0.2–0.3 dB tax.

## 3-ter. Why EXL3 needs rotation and we may not

From EXL3's own kernel contract (`exl3_gemm.cu:26-31`):

```
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- suh: packed input scales/flips,  shape (k//16), dtype float16
- svh: packed output scales/flips, shape (n//16), dtype float16
```

plus a **computed** codebook (`mul_const_u32<0x83DCD12D>`,
`decode_mul1_product_2` in `codebook.cuh`) -- QTIP's multiply-hash Gaussian
construction, not a table of native format values.

So EXL3's entire scale structure is **two rank-1 vectors, `k/16 + n/16`
scalars**: 128 scalars for a 1024x1024 Linear, effectively 0 bpw. NVFP4's is one
FP8 scale per 16 contiguous weights **within each row** -- a 2-D grid of
`rows x cols/16` = **65,536 scalars, 0.5 bpw**. Five hundred times more scale
parameters.

**Incoherence processing is EXL3's substitute for a fine-grained scale plane,
not an addition to one.** Its codebook is fixed and Gaussian-matched, so a
Gaussian source is a *precondition*, not an optimization. We adapt locally with
the group-16 plane instead, and pay 0.5 bpw for the privilege -- which at 2 bits
is a quarter of the entire budget, spent on scales that EXL3 spends on payload.

The tension this creates is structural rather than incidental: **native NVFP4
mandates the group-16 scale plane**, because that is what the
`block_scaled_ue4m3xe2m1` mainloop consumes. The container requires the very
mechanism that makes rotation redundant on scalar content. What the bypass
finding above adds is that the redundancy is confined to the scalar parts, so
the shaped band escapes most of it.

## 4. What this does NOT measure, which is the larger half

**The activation side.** The transform's stated purpose in the runtime doc is
*"an input transform before native FP4 activation quantization"*. Our lane is
**W4A4**; making activations Gaussian before a 4-bit activation quantizer is
where incoherence processing earns its keep, and weight-space wSNR is
structurally blind to it. Rotation costing 0.6–1.2 dB on weights is a price,
not a verdict.

**LDLQ, and this is likely decisive.** A diagonal importance vector offers LDLQ
no cross-column structure — but
`iter_transformed_diagonal_block_ldl_factors` exists precisely because
**rotating a diagonal Hessian makes it dense within each block**. So the
transform *creates* the structure BlockLDLQ feeds on. Rotation and LDLQ are not
independent levers: rotation alone discards the vector objective and gets
nothing back, which is exactly the loss measured here. Measuring rotation
without its partner may be measuring the wrong thing, and the next experiment is
the 2x2 {no-rot, rot} x {no-LDLQ, LDLQ}.

Other scope: one seed pair (11/29), 21 tensors, three layers, one small model,
`static_6` scale rule with no scale grid.
