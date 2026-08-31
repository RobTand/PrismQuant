# WO-F Findings — Trellis + BlockLDLQ rotation 2x2

_Date: 2026-08-31, branch `muse/wo-f-trellis-ldlq-20260831`_

This file records contradictions, blocked paths, and guesses as required by
WO-F.  A wrong assumption in the work order is expected; propagating one
silently is not.

## 1. Read-first documents named in WO-F were not present

WO-F lists as mandatory reads:

* `docs/results/qtip_rotation_weight_side_2026-08-31.md`
* `docs/results/trellis_shaped_rate_ceiling_2026-08-31.md`

Neither file exists at that path on this worktree (`ls docs/results` lists 30
files, the closest is `qtip_trellis_online_hadamard_producer_status_2026-08-30.md`
and `qtip_native_nvfp4_*.md`).  The hypothesis quoted in WO-F (“diagonal
importance vector gives BlockLDLQ no cross-column structure, but
`iter_transformed_diagonal_block_ldl_factors` exists because rotating a
diagonal Hessian makes it dense within each block”) was taken as the authority
and the measurement was built around it.  If the two named result files become
available they should be reconciled against the table below.

## 2. Block reconciliation (F1) — explicit

* Trellis: `SUPERBLOCK_WEIGHTS = 256`, `prismaquant/trellis_encoder.py:965`
  loops ``range(0, columns, 256)`` and `prismaquant/trellis_wire.py` packs
  per-256-block offsets.  The wire is valid only when `columns % 256 == 0`.
* BlockLDL: `TRELLIS_FEEDBACK_BLOCK_SIZE = 256` in
  `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py:68`
  and both `reverse_block_feedback_*` default to `block_size=256`.
* Reconciliation: **1:1, no re-tiling.**  One LDL block is exactly one trellis
  superblock.  The tensor-global E2M1 scale is selected once from the complete
  (transformed) weight before the reverse pass and frozen for every block
  terminal.  The per-block `[rows,16]` E4M3 planes are concatenated in row-major
  order and packed once into one wire (`compose_wire` in the producer).  Each
  feedback step first packs and reference-decodes its terminal bytes; that same-
  byte decode (never the encoder's in-memory float reconstruction) feeds earlier
  blocks.  The final wire must decode FP32-exactly to the concatenation of those
  terminal decodes.  This satisfies F1’s “one encode” and principle 8.

If the two block sizes could *not* be reconciled without changing the wire, the
finding would be reported as a failure; they can, so no fudge was needed.

## 3. F1 terminal seam — what was supplied

`reverse_block_feedback_reference(weight, feedback_lower, terminal, block_size)`
was supplied a `terminal(block_index, target)` that:

1. slices the 256-column schedule and selects only the alphabets whose rates
   appear in that slice (the producer does the same);
2. takes `diag(D_j)` as the column objective when `terminal_metric_mode ==
   diag_block_D` and a unit vector otherwise, exactly as
   `require_blockldl_trellis_wire_round_trip` does;
3. calls `encode_trellis_one_linear` / `encode_trellis_planes` with
   `global_scale_real_override = shared_global`, packs with `pack_planes`,
   reparses via `TrellisWire.from_bytes`, decodes via `decode_values_torch`,
   and asserts BF16 equality — then returns that decode to the recurrence.

The production path uses `reverse_block_feedback_buffered` and the oracle uses
`reverse_block_feedback_reference`; their agreement is pinned in
`tests/test_wo_f_blockldl_trellis.py` (one-encode + buffered-vs-reference).

## 4. Hessian source (F2) — diagonal + transform is the primary path

* Chosen source: `dq-runs/trellis-stage0/stage3_importance_qwen3-0.6b_n32_s512.pt`
  (also present as `importance_qwen3-0.6b_L196_n32_s512.pt`), 196 tensors,
  per-column activation second moment on WikiText-2 raw train, n=32 × seqlen 512,
  seed 20260824.  Regularized as `damped = raw + mean(alive)` with dead
  replaced by 1 before adding the mean, matching `native_nvfp4_ldlq.damped_hessian`
  (damp_fraction 1.0).  The measurement therefore uses the same objective the
  stage3 RD curves used, scored in the original basis as WO-F requires.

* Structured diagonal path:
  `prepare_one_linear_diagonal_hessian_scaffold` + `iter_transformed_diagonal_block_ldl_factors`
  + `require_blockldl_trellis_wire_round_trip`.  This is the cheap, exact path
  that never materializes a dense K×K matrix.  It requires
  `input_block_size % 256 == 0` and `input_block_size >= 256`; see §6 for the
  sweep consequence.

* Dense `H = E[xx^T]` control (F2, second bullet): **no calibration activation
  corpus for Qwen3-0.6B was located.**  Searches under `/home/rob/dq-runs`,
  `/home/rob/models/Qwen3-0.6B`, and the trellis-stage0 directory found only
  diagonal second moments and a small set of synthetic/test fixtures, no
  `[tokens, hidden]` activation rows or a calibration manifest that could be
  loaded as `H`.  `native_nvfp4_ldlq` expects `activations: [*, in]` and a
  `damped_hessian` helper, but no such artifact is pinned for Qwen3-0.6B on this
  worktree.  Per WO-F rule 3 under F2, the finding is recorded here and the
  diagonal path is the primary measurement; a **synthetic** dense control is
  run for three tensors (`X ~ N(0,0.5)`, 64 rows) and clearly labelled as
  synthetic — it is not a calibration estimator and must not be cited as a
  validation of the proxy cost.

## 5. Lower rungs — not added

WO-F asks for `q256 ∈ {768, 896}` at minimum (both use the one blessed
rate-3 alphabet from `arm_e_quality_campaign.canonical_highrate_alphabets`);
lower rungs only if a rate-1/2 alphabet convention can be sourced from the
tree — do not invent one.

* `canonical_highrate_alphabets` validates that `shaped == {3}` and returns
  `{3: 16 codes}` ordered by `(value, code)`.  Any rung whose schedule
  contains rate 1 or 2 is refused.  `q256 = 768` yields pure rate-3 across all
  256-column blocks; `q256 = 896` yields half 3 / half 4 (bypass).  Both pass.
* Lower rungs in the family’s `quality_candidate_q256` are 384,512,640.
  `q256 = 640` would require rates 2 and 3, `q256 = 512` requires 2, etc.,
  each needing an 8-code (rate-2) or 4-code (rate-1) alphabet.  No blessed
  convention for those alphabets was located: `prismaquant/trellis_formats.py`
  validates alphabet size `{rate: 1<<(rate+1)}` but does not ship a canonical
  ordering beyond the rate-3 case; `research/qtip_native_nvfp4_2026-08-30/`
  tests contain a local `_e2_alphabet` helper that is marked test-only and
  explicitly not a producer convention; `stage3_mixed_rate.py` documents a
  per-rate SSE-optimal 8-subset for rate-2 but also states alphabets are
  per-rate and not nested, with no pinned 16×256 vector blessed for export.
  Therefore no lower rung is measured; the 2×2 uses 768 and 896 only, as
  required at minimum.

## 6. Hadamard block sweep — scope and a contract refusal

WO-F asks for sweep `{16,32,64,128,256}` for the rotated+LDLQ cell.

* The structured diagonal path (`prepare_one_linear_diagonal_hessian_scaffold`)
  refuses `input_block_size < 256` or `input_block_size % 256 != 0`
  (`trellis_online_hadamard_producer.py:1127`).  Hence blocks 16/32/64/128
  are not admissible on that exact, allocation-free path.
* All five blocks are admissible on the dense-materialized path
  (`prepare_one_linear_scaffold` with `H = diag(damped)`).  The sweep therefore
  runs dense for 16/32/64/128/256 and additionally runs structured for 256,
  asserting byte identity between the two on that rung (they agree when the
  source Hessian is diagonal).  The table reports the dense series and notes
  the structured equality.  The interesting question — whether LDLQ inverts the
  monotone damage in block size seen without LDLQ — is answered on the dense
  series.  The native kernel’s dispatch at 128 is therefore measurable.

## 7. No-rotation LDLQ collapses (F3 bridge)

A diagonal Hessian without rotation has exactly zero strictly-lower BlockLDL
feedback (`qtip_block_ldl_factors(diag(d))` yields `feedback_lower == 0`,
proved in `test_wo_f_blockldl_trellis.py` and recorded per-tensor in
`noop_evidence` of the results JSON).  The reverse recurrence therefore
degenerates to independent per-superblock encodes, i.e. the identical wire
to the no-rotation/no-LDLQ cell (up to the shared global scale, which is also
identical because the pre-feedback weight is the same).  The harness asserts
`torch.allclose(q_ldl, direct.decoded_weight)` under the same column metric.

Consequences:

* For a diagonal Hessian source, the 2×2’s “no rotation / LDLQ” cell is the
  same measurement as “no rotation / no LDLQ” — LDLQ adds nothing because
  rotation is what *creates* the dense structure LDLQ feeds on.  This is the
  mechanism quoted in WO-F, now measured.
* A dense Hessian (synthetic control) does have non-zero feedback without
  rotation (393k nonzeros for a 1024-width example), so the collapse is a
  diagonal-proxy fact, not a general LDLQ fact.

The results doc reports the collapsed cell explicitly rather than re-measuring
an identical wire.

## 8. Measurement contract — what was and was not invoked

* Scored in the original basis after the inverse transform
  (`decoded_weight_in_original_basis`), exactly as prior stage3 did.  Weighted
  NSSE uses the *raw* importance vector (the same one the encoder’s column
  objective is derived from), not the damped diagonal.  Micro-averaged over
  the 21 tensors (sum numerators / sum denominators → single NSSE → SNR).
* All four (really three distinct) cells are re-derived from one run; no prior
  table is quoted.
* No calibration text was used for KL or PPL; this is a per-Linear weighted-
  reconstruction measurement, not an end-to-end KL, and the scope section says
  so explicitly.  A mixed-rate allocator study over the full 196 Linears would
  be required to claim an end-to-end rate-distortion win.

## 9. Profile / power (principle 15) — what was measured and what it says

* In-process `torch.profiler` (CPU + CUDA) around the LDLQ encode loop and
  `nvidia-smi --query-gpu=power.draw` sampled at ~10 Hz during the same window.
* The trellis encoder hot path is correctly GPU-bound *only when the tensors
  are on CUDA and the Triton backend is used.*  The harness runs with
  `backend="eager"` on CPU tensors for determinism (the reference for byte
  identity is the eager path).  Power on this GB10 under that CPU-only path is
  ~5–15 W of the ~140 W envelope (fraction ~0.04–0.11) — i.e. **not** loaded,
  which `gpu_utilization` would have misreported as 96% saturated.  The profile
  shows the time in `torch.profiler` dominated by the tail-biting Viterbi
  forward/backward and the per-superblock Python loop that launches one encode
  per 256-column block.  BlockLDL itself (the reverse accumulation
  `(W-Q) @ L`) is a small, sequential, launch-bound matmul chain — one
  `torch.mm` per block in row-major order, not fused — so the LDLQ *recurrence*
  is not the bottleneck; the *terminals* are.  A fused, batched terminal or a
  CUDA-graph capture of the Viterbi steps is the headroom, not a wider LDL
  block.  The result is not a serving claim.

## 10. Environment and reproducibility

* Weights: `/home/rob/models/Qwen3-0.6B/model.safetensors` (pinned, 1.5 GB).
* Importance: `/home/rob/dq-runs/trellis-stage0/stage3_importance_qwen3-0.6b_n32_s512.pt`
  (and the identical `stage3_importance_qwen3-0.6b_n32_s512.pt` prefixed variant).
* Code: `research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`
  plus `prismaquant/trellis_encoder.py`, `trellis_wire.py`, `trellis_producer.py`,
  `trellis_rate_surface.py`, `trellis_formats.py` — all SHA-pinned at import
  and re-checked at every encode boundary (the producer refuses a mid-run edit).
* Determinism: `determinism_mode="on"`, `sb_chunk=1024`, `tailbite_candidates=4`,
  `point_route="full"`, `layout="tight_offsets"`, `scale_rule="static_6"`,
  seeds `0x1234 / 0x5678`, `buffer_blocks=1`.

## 11. Guess that was made and how it was discharged

A guess was required for `q256 → schedule → alphabets` at 896: `uniform_column_schedule`
spreads the extra `remainder = round((896/256)*cols) - base*cols` promoted
columns via Bresenham, yielding half rate-3 / half rate-4 (bypass) for
`cols=1024`.  This is not a task invention; it is the repo’s canonical
schedule helper, and the alphabets come from the one blessed rate-3 ordering.
No rate-1/2 alphabet was invented.

## 12. What was run, commands, and log paths

* Harness: `PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python scratch/wo_f_harness.py --out scratch/wo_f_results.json` (+ `--quick` for the
  2×2-only pre-pass).  Logs at `scratch/wo_f.log`, trace at
  `scratch/wo_f_profile_trace.json` (profiler) and power samples in the results
  JSON itself.
* Tests: `PYTHONPATH=. python -m pytest tests/test_wo_f_blockldl_trellis.py -q -v`
  and the full `tests/ -q -x -k "trellis or qtip"` gate passed before commit
  (full-suite counts recorded in the results doc and commit message).
