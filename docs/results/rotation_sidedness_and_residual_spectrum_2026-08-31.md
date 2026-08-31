# Rotation sidedness, and the post-trellis residual spectrum

The artifact for two results that were reported in review before this file
existed, and were correctly downgraded to `[review-reported]` in the EN4/EN8
fold's ledger for exactly that reason. Both are now citable.

Qwen3-0.6B, 21 tensors from layers 4/13/22, Hadamard block 128, seeds 11/29,
transform via `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`.
Drivers `scratchpad/pred.py` and `pred2.py`; logs
`dq-runs/trellis-serve-smoke-20260831/pred.log`, `pred2.log`.

## 1. `sv` is not rotation-invariant — rotation homogenizes rows *more* than columns

Dispersion of row and column RMS norms (std of log; lower = more homogenized):

| | row dispersion | column dispersion |
|---|---|---|
| unrotated | 0.2133 | 0.1491 |
| rotated (two-sided) | **0.1049** | **0.0850** |
| ratio | **2.03×** | 1.75× |

Row norms of `R_out·W·R_inᵀ` equal `‖(R_out·W)_i‖` because `R_in` is
orthogonal — but **`R_out` mixes rows**. Row-norm invariance holds under
`R_in` alone, not under the two-sided transform. So a per-output-channel `sv`
diagonal loses at least as much of its structure to rotation as a
per-input-channel `su` does, not less.

**The design option this opens:** apply `R_in` only. One-sided rotation halves
the transform cost, preserves the per-output-channel structure `sv` captures,
and still delivers A-side incoherence — which is the side the transform exists
for. Whether one-sided homogenization suffices for a 4-bit activation
quantizer is **unmeasured** and is the open question.

## 2. The post-trellis residual is white — in both rotation states

Two matrices were measured, and the distinction matters.

**First, on `log|W|`** — the object a *multiplicative* rank-1 fit approximates:

| | s₁/s₂ | s₁ share | top-4 share | post-rank-1 residual top-4 |
|---|---|---|---|---|
| unrotated | 38.72 | 0.1159 | 0.1224 | 0.0270 |
| rotated | 65.36 | 0.1140 | 0.1183 | 0.0243 |

**This was the wrong matrix.** A residual *wire* encodes an **additive**
residual, so the spectrum must be taken on `W − Ŵ` after a real encode
(trellis, `q256=768`):

| | s₁/s₂ | s₁ share | top-4 share | top-16 share | effective rank |
|---|---|---|---|---|---|
| unrotated | **1.013** | 0.00246 | 0.00880 | 0.03092 | **865.7** |
| rotated | **1.006** | 0.00185 | 0.00733 | 0.02869 | **883.2** |

(effective rank = participation ratio `(Σs)²/Σs²`; equals `min(n,k)` for a
perfectly flat spectrum.)

`s₁/s₂ ≈ 1.0` with an effective rank of ~870 out of ~1024: **the residual is
white, in both rotation states.** A rank-4 term captures 0.88% of its energy
unrotated and 0.73% rotated.

**Mechanism.** This is stronger than a rotated-vs-unrotated differential: a
low-rank compensation term has nothing to capture in *either* state. The
low-rank error-compensation line (ZeroQuant-V2 LRD, LQ-LoRA, CALDERA) operates
on **RTN/GPTQ** residuals, which retain row/column structure. A shaped trellis
has already extracted that structure — its coding gain *is* that extraction.
So a prediction that rotation destroys the low-rank structure fails not because
rotation fails to homogenize, but because **the trellis got there first**. The
negative is evidence the trellis is doing its job.

## 3. Scope

21 tensors, three layers, one 0.6B model, one seed pair, one rung
(`q256=768`), weight-space only, unweighted encode. Neither test touches the
activation side. Replication at other rungs and on GLM5.3-flash tensors is
owed before either is treated as settled.
