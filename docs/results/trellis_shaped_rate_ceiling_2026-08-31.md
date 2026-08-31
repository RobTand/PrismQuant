# The TCQ_E2M1 shaped-rate ceiling, measured

**Why this exists.** Robert: *"the issue with old trellis and new qtip is that
old trellis couldn't exceed 3 bits of descriptivity."* This is that claim
measured on real weights, because it is the reason the trellis lane was taken
in a different direction and it deserves a number rather than a recollection.

## The structure

`TrellisFamily(family='TCQ_E2M1_R256', shaped_max_rate=3, bypass_rate=4, ...)`.

Only three bits per weight can be **shaped** — carried by the convolutional
code and chosen by the Viterbi search. Rate above three goes through **bypass**:
raw uncoded E2M1 codes with no trellis gain. Fable's costing (2026-08-31
handover §5) puts the structural ceiling at `R = 3.96875`, with bypass fraction
at least `R - 3`, and states that **rotation cannot change that schedule
ceiling**. Extending past `shaped_max_rate=3` was cancelled at "+0.00 dB for an
ABI break".

## The measurement

Qwen3-0.6B, 21 tensors from layers 4/13/22, importance-weighted SNR with
`stage3_importance_qwen3-0.6b_n32_s512.pt` as `col_weights`. Both trellis rungs
use the one non-learned alphabet
`arm_e_quality_campaign.canonical_highrate_alphabets` blesses (shaped rates
exactly `{3}`), so the slope between them carries no alphabet confound.

| | body b/w | bypass b/w | wire bpw | wSNR |
|---|---|---|---|---|
| TCQ `q256=768` | 3.000 | 0.000 | 3.5042 | 17.93 dB |
| TCQ `q256=896` | 3.500 | 0.500 | 4.0042 | 18.85 dB |
| scalar NVFP4, no trellis | 4.000 | — | 4.5 | **20.11 dB** |

**Above the cap the trellis buys +1.85 dB per body bit.** A scalar bit is worth
roughly 6 dB, and the family's own low-rate ladder ran ~5.4 dB/bit. So a bypass
bit is worth under a third of an ordinary bit — which is what "uncoded" means,
now with a number on it.

## The consequence, which is worse than saturation

At **4.0042 wire bpw the trellis reaches 18.85 dB while plain scalar NVFP4
reaches 20.11 dB.** In the four-bit region the construction is 1.26 dB *behind*
the scalar format it is meant to improve on. Carrying the measured +1.85 dB/bit
forward, it would need about **4.2 body bits** to catch scalar NVFP4 — beyond
its own structural ceiling of 3.96875. **There is no rung at which this family
catches a 4-bit scalar grid**, let alone EXL3's trellis, which shapes its full
rate at every bitrate.

This is the quantitative form of "old trellis could never match EXL3", and it
is a stronger statement than the incoherence-processing story: an online
Hadamard lifts *both* arms and, per Fable, does not move the schedule ceiling.
The ceiling is not something rotation rescues.

## Scope

- Weight-space wSNR only; it prices the W\*A16 shape, not the attested W4A4
  activation contract.
- One alphabet convention, no scale grid, no rotation, no LDLQ.
- 21 tensors from three layers of one small model.
- **It measures the above-cap slope, not the whole curve.** The sub-3-bit rungs
  (`q256` 384/512/640) need a rate-1/rate-2 set-partitioning alphabet
  convention that the tree does not currently expose and that this measurement
  declined to invent. The trellis is expected to look good down there; that is
  not in dispute and is not what the ceiling claim is about.
