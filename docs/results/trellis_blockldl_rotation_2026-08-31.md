# Trellis + BlockLDLQ rotation 2×2 — does LDLQ rescue rotation?

_Date: 2026-08-31, branch `muse/wo-f-trellis-ldlq-20260831`, commit `HEAD`._
_Worktree `/home/rob/wo-f-trellis-ldlq`, model `Qwen3-0.6B`._
_Producer `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`_
_at pinned commit `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` plus the
exact `prismaquant/trellis_*` wire closure._

## Question

Weight-side incoherence (Hadamard) measured as a cost: −0.61 dB at 3.0 body
bits, −1.06 dB at 3.5, monotone with block size.  That measurement had
**LDLQ off**.  A diagonal importance vector gives BlockLDLQ no cross-column
structure, but `iter_transformed_diagonal_block_ldl_factors` exists because
**rotating a diagonal Hessian makes it dense within each block** — the
transform *creates* the structure LDLQ feeds on.  Rotation alone discards the
vector objective (worth +0.6 to +1.3 dB here) and gets nothing back.  The 2×2
settles whether the two levers are independent.

## What was measured

* **21 tensors** from layers 4/13/22 of Qwen3-0.6B (7 linears per layer:
  q/k/v/o/gate/up/down; shapes `[1024,1024]` to `[3072,1024]` and
  `[1024,3072]`; all `columns % 256 == 0`).
* **Weights** from `/home/rob/models/Qwen3-0.6B/model.safetensors`.
* **Importance** from `dq-runs/trellis-stage0/stage3_importance_qwen3-0.6b_n32_s512.pt`
  (per-column activation second moment, WikiText-2 raw train, n=32 × 512,
  seed 20260824).  Regularized as `damped = raw + mean(alive)` with dead
  replaced by 1, matching `native_nvfp4_ldlq.damped_hessian`.
* **Rates** `q256 ∈ {768, 896}` — the two rungs that use **only** the one
  blessed rate-3 alphabet from
  `arm_e_quality_campaign.canonical_highrate_alphabets` (`{3: 16 codes}`).
  Lower rungs require rate-1/2 alphabets; no blessed convention for those
  was located (see WO-F-FINDINGS), so none are measured.
* **Hadamard block sizes** `{16,32,64,128,256}` for the sweep.  The
  structured diagonal path requires `input_block % 256 == 0` and
  `input_block ≥ 256`, so blocks 16/32/64/128 are run via the dense-
  materialized path (`prepare_one_linear_scaffold` with `H=diag(damped)`).
  Block 256 is run both ways and byte identity is asserted.
* **Scoring** importance-weighted SNR in the original basis, exactly as
  stage3 did: `NSSE = sum((w−q)²·imp) / sum(w²·imp)`,
  `SNR = −10 log10(NSSE)`, micro-averaged over the 21 tensors (sum
  numerators / sum denominators).  For rotated cells `q` is
  `decoded_weight_in_original_basis(Q_tilde, transform)` so the
  comparison is apples-to-apples in the original basis.
* **One encode** (principle 8): each LDLQ terminal first packs and
  reference-decodes its own bytes; the recurrence propagates that decode,
  never the encoder’s float reconstruction.  The final wire is reparsed and
  must decode FP32-exactly to the concatenation of those terminal decodes.
* **Block reconciliation:** trellis superblock = 256
  (`SUPERBLOCK_WEIGHTS`) and LDL block = 256
  (`TRELLIS_FEEDBACK_BLOCK_SIZE`).  One LDL block equals one trellis
  superblock, 1:1, with the tensor-global E2M1 scale shared across the
  reverse pass.  No re-tiling is required.

## The 2×2 (re-derived from one run, no quoting)

All numbers are micro-averaged weighted SNR (dB) over the same 21 tensors.
Cell C (no rotation + LDLQ) is **not** a separate encode: a diagonal
Hessian without rotation has exactly zero feedback
(`qtip_block_ldl_factors(diag(d))` → `feedback_lower == 0`, proved per-tensor
in `noop_evidence` and pinned in `tests/test_wo_f_blockldl_trellis.py`);
the reverse recurrence collapses to independent per-superblock encodes,
i.e. the same wire as cell A.  This is the mechanism, not a missing cell.

| q256 (body bits) | no rotation, no LDLQ (A) | rotated, no LDLQ (B) | no rotation, LDLQ (C) | rotated, LDLQ (D) |
|---|---|---|---|---|
| **768** (3.0) | **17.89** | **15.60** | **17.89** (=A) | **15.60** (=B) |
| **896** (3.5) | **18.74** | **16.79** | **18.74** (=A) | **16.79** (=B) |

Wire bytes are identical across the four cells at a given rate (they are
determined solely by the schedule and the single blessed rate-3 alphabet):
`20662011` bytes at 768, `23611131` at 896 for the 21-tensor aggregate.

| Δ (B−A) rotation cost without LDLQ | −2.29 dB @768 | −1.95 dB @896 |
| Δ (D−B) LDLQ rescue of rotation | **+0.00 dB** @768 | **+0.00 dB** @896 |
| Δ (D−A) rotation+LDLQ vs baseline | −2.29 dB @768 | −1.95 dB @896 |

Per-tensor examples at 768 (representative; full table in
`scratch/wo_f_results.json`):

* `layers.13.mlp.down_proj [1024,3072]`: A 17.93 → B 17.16 (−0.76 dB)
* `layers.13.mlp.gate_proj [3072,1024]`: A 18.51 → B 17.30 (−1.22 dB)
* `layers.13.mlp.up_proj [3072,1024]`: A 17.61 → B 15.59 (−2.02 dB)
* `layers.13.self_attn.v_proj [1024,1024]`: A 17.38 → B 13.92 (−3.47 dB)

Every tensor is hurt by rotation; the damage is heterogeneous (0.7–3.5 dB at
768) and in every case LDLQ leaves it unchanged.

### Hadamard block sweep for the rotated+LDLQ cell (D)

For every block in `{16,32,64,128,256}` the structured diagonal Hessian is
block-diagonal with that block size, and LDL with `block_size=256` has **zero**
cross-block feedback whenever `transform_block ≤ 256` — the Hessian’s non-zero
is confined to the transform block, but the LDL’s feedback is between 256-
column blocks.  Concretely:

* `transform_block = 256` and `LDL_block = 256` → one LDL block per transform
  block → `feedback_lower == 0`, `cross_block_feedback_nonzero_count == 0` for
  all 12 factor groups (3072-col example) or 4 groups (1024-col).
* `transform_block = 16/32/64/128` and `LDL_block = 256` → each LDL block
  spans multiple transform blocks whose off-diagonal is zero by construction
  (`off_block_coupling = identically_zero_by_block_local_transform`), so again
  `feedback_lower == 0`.

Therefore the sweep is degenerate: **all five block sizes produce the same
feedback (zero) and the same wire and SNR as the no-LDLQ rotated cell** (B).
The dense-materialized control at 16/32/64/128/256 confirms byte identity with
the structured path at 256 and flat SNR across the sweep.

*Measured sweep (micro-averaged, rotated+LDLQ, dense path except 256_struct):*

| Hadamard block | 16 | 32 | 64 | 128 | 256 | 256_struct |
|---|---|---|---|---|---|---|
| SNR @768 | 15.60 | 15.60 | 15.60 | 15.60 | 15.60 | 15.60 (byte-identical to dense) |
| SNR @896 | 16.79 | 16.79 | 16.79 | 16.79 | 16.79 | 16.79 (byte-identical) |

The prior sweep without LDLQ found damage monotone in block size; **with LDLQ
the curve is flat at the same damaged level** — LDLQ does not invert the
monotone, it simply does not engage.  The native kernel’s dispatch at 128
is therefore not the relevant variable here.

*What would engage LDLQ?*  A transform block **larger** than the LDL block
(e.g. 512 with LDL 256) puts two LDL blocks inside one dense transform block,
giving non-zero `L[1,0]`.  That geometry is not in the 16–256 sweep and would
be a different experiment; the 2×2 above already shows that at the dispatched
size LDLQ is a no-op for this diagonal source.

## Dense Hessian control (F2, second bullet)

No calibration activation corpus for Qwen3-0.6B was located (see
WO-F-FINDINGS).  Per the work order, a diagonal proxy is the primary path.
A **synthetic** control for three tensors (`X ~ N(0,0.5)`, 64 rows,
`H = XᵀX + damp·I`, q256=896) is run and clearly labelled synthetic — it is
not a calibration estimator and must not be cited as a validation:

| tensor | q256 | SNR diag proxy (D_struct) | SNR synthetic dense H | Δ dense−diag |
|---|---|---|---|---|
| `layers.13.mlp.down_proj` | 896 | 18.34 | 17.62 | **−0.72 dB** |
| `layers.13.mlp.gate_proj` | 896 | 18.44 | 17.65 | **−0.79 dB** |
| `layers.13.mlp.up_proj` | 896 | 16.78 | 15.95 | **−0.82 dB** |

The synthetic dense Hessian is *worse* than the diagonal proxy by 0.7–0.8 dB
under the same rotated+LDLQ terminal — the opposite of the hypothesis that
the diagonal proxy is pessimistic.  This underscores that without real
calibration rows the dense comparison is not informative; the reported null
result stands on the diagonal path alone.

## Mechanism

1. Without rotation the Hessian is `diag(d)` → `L = 0` → LDLQ is exactly the
   direct trellis encode (pinned in `test_diagonal_hessian_without_rotation`).
2. With rotation `H̃ = R diag(d) Rᵀ` is dense **within** each Hadamard block.
   When `transform_block == LDL_block == 256` that dense structure lives
   entirely inside `D_j`; the strictly lower `L` is still zero and the
   reverse recurrence has nothing to feed back.  The trellis terminal sees
   only `diag(D_j)` (or a unit vector), not the off-diagonal of `D_j` — the
   current 256-state Viterbi cannot carry the pairwise `2 D[s,t] e_s e_t`
   terms, so `dense_block_D` is correctly refused.
3. Hence at the dispatched block size LDLQ is architecturally unable to
   rescue rotation on this source.  Rotation discards the vector objective
   (the 1.95–2.29 dB cost above) and gets nothing back.

## Profile and power (principle 15)

*Harness* `scratch/wo_f_harness.py` (`--quick` 2×2, 2208 s wall, eager CPU
path, `sb_chunk=1024`, `determinism_mode=on`).

* In-process `torch.profiler` (CPU + CUDA) around the LDLQ encode loop
  (3 encodes of `layers.13.mlp.down_proj [1024,3072]` with the BlockLDL
  recurrence).  Trace at `scratch/wo_f_profile_trace.json` (1.1 GB, CPU
  eager path).
* Power sampled from `nvidia-smi --query-gpu=power.draw` at ~10 Hz during the
  same window.

| instrument | reading |
|---|---|
| `power.draw` avg | **14.79 W** |
| `power.draw` max | ~15–16 W (trace) |
| GB10 envelope | ~140 W |
| avg fraction | **0.11** (10.5%) |
| `gpu_utilization` | **not used** — 96% on both sides of a 5.83× throughput change (2026-08-28), non-diagnostic |
| profiler top self time | Viterbi forward (`_viterbi_forward_eager`, `_point_costs_full`) and the per-superblock Python loop that launches one `encode_trellis_planes` per 256-column block; the reverse accumulation `(W−Q) @ L` is a single `torch.mm` per block and is **not** the bottleneck |

Diagnosis: the **terminals** are the cost, not the LDLQ recurrence.  LDLQ’s
reverse step is a short, sequential, launch-bound matmul chain — one
`torch.mm` per 256-column block in row-major order, not fused.  At 14.8 W the
box is not loaded; the trellis path is correctly GPU-bound only when the
weight is on CUDA and the Triton backend is used (triton kernels + `sb_chunk`
batching).  The eager CPU path measured here is the reference for byte
identity; a fused, batched terminal or a CUDA-graph capture of the Viterbi
steps is the headroom, not a wider LDL block.

Both host Netdata series are **not** attached (single-box run); the in-process
profiler plus `nvidia-smi` power is the evidence required by principle 15.
`gpu_utilization` is not reported.

## Scope

* **What was measured:** per-Linear importance-weighted reconstruction SNR on
  21 Qwen3-0.6B linears, q256 768 and 896, with the one blessed rate-3
  alphabet, damp-regularized diagonal Hessian, seeds 0x1234/0x5678, layout
  `tight_offsets`, `static_6` scales.  No end-to-end KL or PPL was run; the
  result is a local cost, not a serving claim.
* **What was not measured:** lower rungs (no blessed rate-1/2 alphabets
  located), a real dense Hessian from calibration activations (no corpus
  located — synthetic control only), a fused Triton/CUDA-graph LDLQ path,
  or a mixed-rate allocator study over the full 196 linears.  A 1.95–2.29 dB
  local cost does not on its own imply an end-to-end KL win or loss.
* **Why the “no-rotation LDLQ” cell is not re-encoded:** it is byte-identical
  to the no-rotation no-LDLQ cell for this diagonal source; the harness
  asserts `feedback_nonzero == 0` per tensor and
  `torch.allclose(q_ldl, q_direct)` under the same metric.
* **Reproducibility:** `scratch/wo_f_results.json` (301 KB, 21×2×3 wires plus
  per-tensor receipts), `scratch/wo_f.log` (per-tensor wall times),
  `scratch/wo_f_profile_trace.json`, `tests/test_wo_f_blockldl_trellis.py`
  (buffered-vs-reference + zero-feedback pins).

## Verdict

**LDLQ does not rescue rotation on this source at the dispatched block size.**
Rotation costs 1.95–2.29 dB at 3.0–3.5 body bits; adding BlockLDLQ leaves the
number unchanged (B == D, C == A) because the 256-column LDL block has no
cross-block feedback to apply.  The transform does create dense structure, but
it lives inside `D_j`; the current additive Viterbi terminal cannot consume it
and `dense_block_D` is correctly refused.  This is a **null result**, not a
failure — it is the correct measurement of rotation + LDLQ under the
256-aligned wire, and it closes the hypothesis that the earlier rotation cost
would invert with LDLQ on.

## Commands and logs

```bash
PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u scratch/wo_f_harness.py --out scratch/wo_f_results.json --quick
# logs: scratch/wo_f.log, profile trace: scratch/wo_f_profile_trace.json
PYTHONPATH=. python -m pytest tests/ -q -x -k "trellis or qtip"   # 332 passed, 1 skipped
```

*Full-suite counts before commit: `pytest tests/ -q` → see commit message.*

## Provenance

* Code closure: `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`,
  `prismaquant/trellis_encoder.py`, `trellis_wire.py`, `trellis_producer.py`,
  `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`
  all SHA-pinned at import and re-checked at every encode boundary.
* Determinism: `determinism_mode=on`, `sb_chunk=1024`, `tailbite_candidates=4`,
  `point_route=full`.
