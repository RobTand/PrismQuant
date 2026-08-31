# WO-F — Trellis + BlockLDLQ, and the 2×2 that settles rotation

Worktree `/home/rob/wo-f-trellis-ldlq`, branch `muse/wo-f-trellis-ldlq-20260831`.
Python `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`, `PYTHONPATH=.`
from the worktree root. Never `pip install`. Scratch files go in
`<worktree>/scratch/`, never `/tmp`.

## Read first

1. `AGENTS.md`, `CLAUDE.md`. Binding here: **1** (fix the measurement, not the
   optimizer), **2** (no constants from intuition), **3** (a screen is not a
   result), **8** (one rendering), **15** (measurement is first-class; on GB10
   read **power against the ~140 W envelope**, `gpu_utilization` is
   non-diagnostic).
2. `docs/results/qtip_rotation_weight_side_2026-08-31.md` — **the finding this
   work order exists to explain.** Read it before writing code.
3. `docs/results/trellis_shaped_rate_ceiling_2026-08-31.md`.
4. `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`
   — especially `qtip_block_ldl_factors`,
   `iter_transformed_diagonal_block_ldl_factors`,
   `reverse_block_feedback_reference`, `reverse_block_feedback_buffered`,
   `build_online_transform`, `transform_weight`,
   `decoded_weight_in_original_basis`, `transform_weight_and_hessian`.
5. `prismaquant/trellis_producer.py`, `prismaquant/trellis_encoder.py`.

## The question

Incoherence processing measured as a **cost** on the weight side: −0.61 dB at
3.0 body bits, −1.06 at 3.5, and monotonically worse with Hadamard block size.
That was measured with **LDLQ off**, and there is a specific reason to think
that makes it the wrong measurement:

> A diagonal importance vector gives BlockLDLQ no cross-column structure. But
> `iter_transformed_diagonal_block_ldl_factors` exists precisely because
> **rotating a diagonal Hessian makes it dense within each block.** The
> transform *creates* the structure LDLQ feeds on. Rotation alone discards the
> vector objective (worth +0.6 to +1.3 dB here) and gets nothing back.

So rotation and LDLQ may not be independent levers. Settle it with a 2×2.

## Deliverables

### F1. Trellis encode under BlockLDLQ feedback

`reverse_block_feedback_reference(weight, feedback_lower, terminal, block_size)`
already implements the reverse recurrence and takes `terminal(j, block)` as the
callback that quantizes block `j`. **That is the seam.** Supply a terminal that
trellis-encodes the block through the existing producer path, so the quantized
result the recurrence propagates is the real wire's decode and not a stand-in.

Constraints that are not negotiable:

- **One encode.** The bytes whose decode the recurrence consumes must be the
  bytes a later export would ship (principle 8). Do not quantize twice, and do
  not let the recurrence see a value the wire cannot reproduce.
- The trellis encodes over **256-column superblocks**; the LDL recurrence works
  in blocks of `block_size`. Reconcile those two block structures **explicitly**
  and write down the reconciliation. If they cannot be reconciled without
  changing the wire, say so in findings rather than fudging it — that is a real
  finding, not a failure.
- Use `reverse_block_feedback_buffered` for the production path and
  `reverse_block_feedback_reference` as the oracle; pin their agreement in a
  test.

### F2. The Hessian source

Two paths, both wanted:

- **Diagonal + transform**, via `iter_transformed_diagonal_block_ldl_factors`,
  fed by the existing importance vectors
  (`dq-runs/trellis-stage0/stage3_importance_qwen3-0.6b_n32_s512.pt`). This is
  the cheap path and the one that tests the hypothesis directly.
- **Dense `H = E[xx^T]`** from calibration activations via
  `qtip_block_ldl_factors`, for at least a few tensors, as the control that
  says how much the diagonal proxy costs.

Do not invent a Hessian estimator. If no calibration activation source is
readily available for Qwen3-0.6B, report that in findings and run the diagonal
path only.

### F3. The 2×2 measurement

Qwen3-0.6B, the same 21 tensors from layers 4/13/22 the prior results used, so
the numbers are directly comparable. `q256 ∈ {768, 896}` at minimum (both take
the one blessed rate-3 alphabet from
`arm_e_quality_campaign.canonical_highrate_alphabets`); add lower rungs only if
you can source a rate-1/2 alphabet convention from the tree — **do not invent
one**, and say so if you cannot.

| | no rotation | rotated |
|---|---|---|
| **no LDLQ** | have it (18.25 dB @896 unweighted) | have it (17.19 dB @896) |
| **LDLQ** | ? | ? |

Score **importance-weighted SNR in the original basis**, exactly as the prior
results did, and re-derive the two existing cells rather than quoting mine, so
all four come from one run.

Also sweep Hadamard block size ∈ {16, 32, 64, 128, 256} for the rotated+LDLQ
cell. The prior sweep found rotation damage monotone in block size **without**
LDLQ; whether LDLQ inverts that is the interesting part, because the native
kernel only dispatches at **128**.

### F4. Report honestly

A results doc under `docs/results/`, in the voice of the two named above:
what was measured, the table, the mechanism if one is visible, and an explicit
scope section. If LDLQ does **not** rescue rotation, say so plainly — a
negative result here is worth as much as a positive one and the tree records
negative results with their lesson. Do not bury a null.

Include a profile and **power against the ~140 W envelope** for the LDLQ encode
path (principle 15). BlockLDLQ is sequential over blocks and could easily be
launch-bound; if it is, that is a finding.

## Rules

- Fail closed. No fallback to an unweighted encode when a Hessian is missing.
- No new constants from intuition; block sizes come from the sweep or the
  runtime contract, not from taste.
- Match the surrounding comment density; this tree writes long *why* comments.
- Run `PYTHONPATH=. python -m pytest tests/ -q -x -k "trellis or qtip"` green
  before committing, then report full-suite counts.
- Commit and push on your branch.
- Contradictions, blocked paths, and anything you had to guess go in
  `WO-F-FINDINGS.md` at the worktree root. **A wrong assumption in this work
  order is expected; propagating one silently is not.**
