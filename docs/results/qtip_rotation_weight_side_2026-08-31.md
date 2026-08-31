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
