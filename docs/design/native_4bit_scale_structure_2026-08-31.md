# Where the scale bits live: a design note for native 4-bit Blackwell support

**Status: research, measured, not yet a proposal.** Weight-space only; nothing
here has touched an activation contract, a wire, an ABI or a kernel. Every
number is Qwen3-0.6B, 21 tensors from layers 4/13/22, `q256=768` (3.0 body
bits, **zero bypass**), importance-weighted SNR scored in the original basis.
Logs under `dq-runs/trellis-serve-smoke-20260831/`.

Framing per Robert: the target is **native 4-bit Blackwell support**, of which
NVFP4 is one convention. The `block_scaled_ue4m3xe2m1` mainloop is the hardware
capability; how we drive it is ours to choose.

## 1. The three mechanisms are substitutes, not additions

A 4-bit format needs *some* way to handle magnitude structure. There are three
on the table, and the measurements say they compete rather than compose:

| mechanism | cost | what it does |
|---|---|---|
| group-16 block scale plane | **0.500 bpw** | local dynamic range, per 16 contiguous weights per row |
| rank-1 diagonals `su ⊗ sv` | **0.023 bpw** | per-input-channel and per-output-channel magnitude |
| Hadamard rotation | 0 bpw (runtime) | flattens dynamic range by mixing |

**Rotation and the block plane are substitutes, measured.** With the fine plane
present, rotation *costs* −0.51 dB. Remove the plane and rotation *gains*
**+4.23 dB** (7.88 → 12.11). Whichever is present makes the other redundant.

**Rank-1 is complementary to both** and is the cheapest by an order of
magnitude: `fine plane + rank-1` gains **+0.87 dB for 0.023 bpw** — about
**37 dB/bit**, against ~17.6 dB/bit for the block plane and ~5 dB/bit for body
rate. It captures channel-wise structure that neither a per-row block plane nor
a rotation expresses, and that is where LLM outliers live.

## 2. Block/Hadamard alignment does not matter

Tested because MXFP4's native block is 32 and the literature reports 16-element
rotations as weak: does aligning the Hadamard block to the scale block help?

| scale block | scale bpw | no rot | had 16 | had 32 | had 64 | had 128 |
|---|---|---|---|---|---|---|
| 16 | 0.500 | **17.53** | 16.40\* | 16.28 | 16.27 | 16.15 |
| 32 | 0.250 | **16.98** | 15.92 | 15.80\* | 15.76 | 15.65 |
| 64 | 0.125 | **16.58** | 15.49 | 15.45 | 15.40\* | 15.23 |

`\*` = Hadamard block equals scale block. **No alignment effect at any row** —
the aligned cell is never the best rotated cell, and at scale-32 it is worse
than Hadamard-16. The only pattern is monotone: smaller Hadamard, less damage;
no rotation, least damage. Consistent with §1 — with a block plane present,
rotation has nothing left to do.

**But block-32 is a good trade on its own merits**: 0.55 dB for 0.25 bpw
(17.53 → 16.98), and at ~5 dB/bit for shaped rate that 0.25 bpw is worth
+1.25 dB, i.e. **net +0.7 dB at matched bpw**. Coarsening pays; rotation is not
why. *Caveat: this uses an exact E4M3 scale per 32. Real MXFP4 uses E8M0
power-of-two, measured in this repo at +13.8% output MSE over 410 Gemma
Linears, so the row above is an upper bound on MXFP4.*

## 3. Once you rotate, you must re-clip — and it is most of the gap

The flat-plane arm looked hopeless at −4.56 dB. Most of that was the grid being
mis-scaled, not the structure: after rank-1 + rotation the source is Gaussian
with RMS 1, and amax scaling spends the E2M1 range on ~5σ tails no weight
occupies.

| flat + rank-1 + rot, scale at | wSNR | vs fine plane |
|---|---|---|
| amax | 12.11 | −4.56 |
| 4.0σ | 13.65 | −3.02 |
| 3.0σ | 14.67 | −2.00 |
| 2.5σ | **14.92** | **−1.75** |

**+2.81 dB recovered by clipping alone**, and still improving at 2.5σ. This is
not a new mechanism: it is exactly what JSO's `joint_mse` scale search already
does for NVFP4 (levels {6,4}), pointed at the global scale instead of the
group scale.

*(An earlier version of this sweep produced eight identical rows. The cause was
mine: `scale_plane_override` sets the codes, but `encode_trellis_planes`
derives `global_scale` from the natural amax unless
`global_scale_real_override` is also supplied, so the requested scale was
silently discarded. Both must be pinned together.)*

## 4. What this implies for portability

A group-16 block scale is a **hardware operand**: it requires a GEMM consuming
a per-16 E4M3 scale beside packed fp4, i.e. `block_scaled_ue4m3xe2m1`,
Blackwell sm120+. A flat plane plus rank-1 diagonals requires none — the weight
operand is pure packed fp4 and the diagonals are elementwise vector ops in the
input transform and epilogue. That runs on anything with a 4-bit dot product.

And a flat plane **is still a legal NVFP4 operand** (set every group scale
equal), so one artifact takes two paths: feed the constant plane to the
Blackwell mainloop for native fp4 tensor cores, or strip it entirely on
hardware without block-scale support.

Standing at 3.0 body bits:

| design | scale bpw | total bpw | wSNR | needs block-scale HW? |
|---|---|---|---|---|
| fine plane + rank-1 | 0.523 | 3.523 | **17.53** | yes |
| block-32 + rank-1 | 0.273 | 3.273 | 16.98 | yes |
| flat + rank-1 + rot + clip | 0.023 | 3.023 | 14.92 | **no** |

The flat design is 2.61 dB behind at **0.500 bpw less**. At ~5 dB/bit that
0.500 bpw is worth ~2.5 dB, so the portable design is roughly **break-even**
while dropping a hardware requirement — and its clipping optimum has not been
found yet.

## 5. What is not established

- **Weight-space only.** Prices W\*A16; says nothing about the attested W4A4
  activation contract, which is where incoherence processing is supposed to earn
  most of its keep and which none of this measures.
- **No LDLQ**, on any arm. Rotation destroys the vector importance objective and
  BlockLDLQ is what replaces it; the flat arm is the one most likely to move.
- The "~5 dB/bit for shaped rate" used in every matched-bpw comparison here is
  **recorded, not re-measured**, so all the net-win arithmetic is an indication
  rather than a measured matched-bpw pair. Producing those pairs needs the
  intermediate rungs, which need a rate-1/2 alphabet convention the tree does
  not expose.
- One rung, one model, 21 tensors, three layers, one seed pair. Rank-1 factors
  are 3 rounds of alternating RMS; EXL3 takes one `block_rms` pass per side and
  the choice of 3 is untuned.
