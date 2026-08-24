# Native-parity and production format-selection policy

This is the normative policy for balancing Gridbook quality against native
execution performance. It supersedes the earlier `quality + lambda*time`
proposal, the blanket decode-neutrality claim, and the retracted 503/503
screen. Historical experiments remain useful evidence, but they do not
override these acceptance rules.

## 1. Optimize quality under hard deployment constraints

For an assignment `a`, production selection is:

```text
minimize    predicted_quality_loss(a)

subject to  exact_whole_artifact_bytes(a) <= B
            p95_TTFT(a, workload)          <= SLO_prefill
            p95_ITL(a, workload)           <= SLO_decode_itl
            p05_TPS(a, workload)           >= SLO_decode_tps
            resident + KV + peak_scratch   <= device_budget
            backend, shape, TP, fallback and serving-unit coupling are legal
```

Latency is not blended into the objective. There is no `lambda`, no single
phase-weighted `serve_ms`, and no default workload mix hidden in the allocator.
Prefill and decode are separate constraints because a format can move them in
opposite directions. Operators choose explicit SLOs; the allocator minimizes
quality loss within them.

Per-layer or per-operator timing tables may generate candidate assignments.
They are not final evidence. A selected assignment still requires same-session
end-to-end timing plus served KL/PPL/tasks, because routing, scheduler batching,
graphs, scratch pressure, and fallback can change the global rank.

Two denominators have different jobs and must not be conflated:

- An allocator study that claims a *relative speed tax* against the best
  attainable assignment uses the fastest **globally feasible assignment under
  the same whole-artifact byte budget**, memory limits, legality rules, and
  serving-unit coupling. Summing independently fastest units is invalid, and a
  bounded probe set cannot claim global optimality.
- A production release must meet served-speed parity with the exact container
  it displaces (repository rule 4). The manifest names that prior artifact,
  why it is the displaced deployment, its complete content identity and byte
  budget, and benchmarks it afresh in the same image/runtime/hardware/workload
  session. This is a release-comparison claim, not a claim that the displaced
  artifact is globally fastest.

Both comparisons remain phase-specific for prefill and decode; neither blends
the phases. A proof that an all-native assignment exceeds the byte budget only
rules out that class. It neither proves global optimality nor chooses the
release comparator.

### What is implemented, and what still gates promotion

The constrained Pareto solver described here **is implemented** as proposal
machinery as of the ultraplan P5c producer work (gridbook
`docs/audits/ultraplan_perf_2026-08-01.md` §6). Concretely:

- **The constraints are hard and separate.** `prismaquant/serve_constraints.py`
  evaluates p95 TTFT, p95 ITL, p05 TPS and `resident + KV + peak_scratch`
  against operator-supplied SLOs. An assignment that misses any of them is
  INFEASIBLE — removed from the candidate set, never re-ranked. There is no
  `lambda`, no phase-weighted `serve_ms`, and no default workload mix: the
  objective is still exactly minimum predicted Δloss, and among feasible
  assignments the existing ratchet tie-break (min Δloss, ties toward the
  larger footprint) is unchanged.
- **Where it is enforced.** At assignment level, in the allocator's
  exact-payload / byte-budget ratchet — the same point the exact byte filter
  sits — and not inside `solve_allocation`'s bits-DP, whose semantics for the
  unconstrained case are unchanged and pinned by test. The filter sees the
  EXPANDED, promoted assignment, which is the object that actually ships.
- **What it consumes.** A declarative measured dispatch table
  (`prismaquant/serve_dispatch_table.py`, schema
  `prismaquant.serve_dispatch_table.v1`): per (format family, phase, M-regime,
  serving lane) relative serving cost, with mandatory per-row provenance —
  source document, date, GPU identity, measured quantity, units, and the
  derivation from the published number to the ratio. **A row without a source
  is a load error.** Each `(phase, M-regime)` arena names exactly one
  reference route, so ratios measured against different denominators are never
  silently composed. Lane resolution uses the §2 serving-lane metadata: a rung
  whose fused mid-M lane the pinned Gridbook version does not instantiate is
  priced with its FALLBACK route's row, never the fused lane's.
- **The aggregation model is explicit and stamped**: an additive layer-time
  model, parameter-share weighted, with eight named assumptions (additivity,
  parameter-share weighting, route locality, regime uniformity, baseline
  transfer, resident bytes, the single-stream `p05_TPS = 1000 / p95_ITL_ms`
  identity, and statistic transfer). It is stamped as
  `additive_layer_time__param_share_weighted__table_driven_proposal` beside
  the existing `additive_candidate_proposal_then_exact_assignment_filter`
  honesty stamp, and it claims no global optimality: the filter guarantees
  every ACCEPTED assignment is feasible on both axes, not that the feasible
  set was enumerated.
- **Fail-closed.** An assignment the table cannot price is infeasible, not
  "passed": a missing row, an arena with no absolute reference, and an
  isolated-operator microbenchmark arena all refuse to certify. `selection.json`
  records which constraints were active, which probed assignments were
  rejected and for which SLO, and which constraint binds at the optimum.

What has **not** changed is the promotion rule. Table-driven latency remains
**proposal data**, exactly as the paragraphs above say: per-layer and
per-operator timing tables generate candidate assignments; the same-session
served protocol (streaming TTFT/ITL/TPS percentiles plus served KL/PPL/tasks,
NATIVE-PARITY) is what promotes one. Supplying no table and no SLOs leaves
every code path byte-identical to the pre-P5c allocator, with a stamp
recording that constraints were absent — so an artifact never implies a
latency claim it did not make.

See `docs/design/constrained_pareto_allocation.md` for the design and the
assumptions in full.

## 2. Bytes are a serialized-artifact constraint

Candidate construction, allocation, reports, and exporter assertions use the
versioned `CBSerializationContext` payload accountant. The final authority is
the exact exported artifact: model shards and metadata plus every served
sidecar, with shared sidecars charged once by serialized identity. Target bpp
and index-body bytes are diagnostics, not the acceptance gate.

The relevant serialized contracts are:

- production NVFP4-CB K12..K24 uses two-tier layout v2: `4k + 9` bytes per
  256-weight superblock, about 1.78125..3.28125 bpw before shared sidecars;
- FP8-CB uses a `4k`-byte index body per superblock **plus** one FP32
  `weight_scale` per output row; and
- product-VQ FP16 subtable sidecars are serialized once per codebook
  reference/format identity. A lattice reference does not imply a free
  sidecar.

The full execution contract must be selection and benchmark identity. A format
name alone is not a byte or execution identity.

`Candidate` encodes format, predicted loss, payload bytes, serialized layout,
and sidecar identity through `CBSerializationContext`. Since the ultraplan P5
producer work (gridbook `docs/audits/ultraplan_perf_2026-08-01.md` §6) it also
encodes:

- **the concrete serving-lane route** (`Candidate.serving_lane`), resolved
  from the target serving profile's declarative `serving_lanes` block: the
  served activation contract (`w8a8-dynamic-e4m3` for FP8-CB,
  `w4-bf16-bridge` for the default FP4-CB quality path), whether the
  consumer's fused mid-M kernel actually instantiates **this rung**, and the
  fallback route it takes when it does not. The backed rung set is spec data
  keyed by the pinned Gridbook version in
  `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; a pinned version
  the spec does not declare backs nothing. The legacy 0.8.11 contract names
  fused mid-M K ∈ {28,32,36,40,44,48}; new production obeys the generalized
  K%4 ladder K4..K48. Every historical off-law K28..K48 wire id remains a
  reader input only and can no longer enter allocator or exporter menus. Low
  rungs and Ada remain unshippable: the current Gridbook v10 candidate carries
  only `compile_only` exact-sm89 dense cells. Strict publication requires
  `device_qualified` decode and batch cells for the complete twelve-rung
  producer ladder, even when one assignment selects fewer rungs, plus the
  separate physical 4090 graph/device receipt. The strict artifact is
  lattice-only, sets `CB_ACTIVATION_SCOPE=none`, and contains no NVFP4 family.
- **which estimator priced its activation contract**
  (`Candidate.activation_pricing`): the measured `output_mse` branch, a
  weight-only branch corrected by the per-family activation calibration, or a
  weight-only branch left uncorrected. Weight-only rows — packed experts
  under `PRISMAQUANT_EXPERT_COST_SAMPLE`, interpolated rungs under
  `CB_LADDER_INTERP` — no longer read as if their A side had been measured.

The serving profile additionally models gridbook's real load gates, not just
the `in_features % 256` superblock rule: `out_features % 8` for the fp4-CB
families and `out_features % 16` for fp8-CB, declared per grid from the
`cb_layout` family table.

`selection.json` records, for the shipped assignment, which selected rungs
ride a backed fused lane versus the expand+GEMM fallback, the activation
contracts in play, the per-family activation calibration (fit, sample digest,
residual band), and the cross-family CB-ladder symmetry verdict. Since P5c it
also records the hard serving constraints that were active, the probed
assignments the SLO axis rejected and the limit that rejected each, and the
constraint that binds at the shipped optimum (§1). Serving-unit identity
remains a feasibility input and benchmark provenance rather than a property of
the candidate object.

## 3. Benchmark the execution contract

Every result records and pins:

- format and rung;
- serialized layout version and scale coding;
- activation quantization (`W4A4`, `W8A8`, or the observed fallback contract);
- concrete kernel/backend and fallback state;
- model, tokenizer, Gridbook, vLLM/runtime, GPU/driver, and image commits;
- tensor-parallel size and scheduler/graph configuration; and
- exact whole-artifact bytes and budget.

Native NVFP4 is W4A4 while FP8-CB K36 is W8A8. A pure endpoint comparison
therefore measures the complete format-plus-activation execution contract, not
weight encoding alone. In delegated NVFP4 MoE, vLLM `auto` can select Marlin,
drop activation scales, and execute W4A16. Such a run is not a W4A4 baseline;
the server backend trace must prove the declared contract.

Release evidence covers the workload rather than one convenient shape:

- a prompt-length distribution and concurrency ladder;
- chunked prefill on and off;
- plain M=1 decode;
- batched and speculative decode at the shipped M, with acceptance recorded;
- MoE routed-token histograms and expert imbalance; and
- whole grouped-MoE operators, not isolated tensor or summed per-expert times.

Plain low-M decode cannot be extrapolated into tensor-core, batched, or
speculative regimes. Final timing uses streaming TTFT/ITL/TPS percentiles and
same-session arm ordering; offline whole-request latency is directional only.

## 4. Same-rate comparisons and production capability

At 4.5 bpp, compare native NVFP4 against FP8-CB K36 only after exact
whole-artifact accounting. Below native NVFP4's 4.5-bpp floor, “same average
rate” is an assignment-level comparison: every NVFP4 promotion must be funded
by lower CB rungs elsewhere. Evaluate the byte-neutral bundle's net quality
loss and phase-specific latency gain; never compare an isolated promoted layer
against an unfunded baseline.

NVFP4-CB v2 is the capacity backbone in the approximately
1.78125..3.28125-bpw band, but it does not yet have teacher-backed same-rate
quality validation. Keep its fused native-FP4 prefill paths explicit opt-ins
until served KL/PPL and routing/shape gates pass; the path changes activation
scales from the fp32-emulated bucket to native ue4m3 factors.

Packed expert stacks currently deny stock NVFP4/FP8 in the Gridbook producer
profile. The native-versus-CB mixed frontier is therefore feasible only for
dense and shared units. Packed experts require a lane-level native/CB A/B and
native expert delegation before the allocator may claim that frontier.

Signed S13..S16 remain a legacy Gridbook reader concern and are excluded from
PrismaQuant's producer registry. Every current product rung remains in the
production menu: NVFP4-CB K12..K24 and FP8-CB K4..K48 with K%4.

## 5. Evidence boundaries

The current record supports only the following statements:

- Strong BF16-teacher-backed wins are FP8-CB at Qwen3.6-27B/5.5 and
  Ornith-35B/4.75. They do not establish low-bit FP4-CB quality.
- Exact-4.5 Stage 0 strongly favors production-faithful K36 weight error:
  493/496 units at 27B and 252/252 at 4B. This is a stop-only surrogate screen,
  not served KL/PPL and not a promotion.
- Hy3-295B/2.9 has fit, serve, and TEB evidence but no BF16-teacher quality
  claim. Zero selected NVFP4 units are circular evidence because that allocator
  optimized accuracy only.
- Published “native” reference artifacts are generally mixed NVFP4/FP8, not
  pure NVFP4. Name their actual assignment and activation/backend contract.
- The rapid 0.6B endpoint pair is approximate performance evidence only:
  native is 870,290,032 bytes; FP8-CB model plus sidecar is 871,628,664 bytes,
  a 1,338,632-byte (+0.154%) excess that misses the <=0.1% formal target. The
  arms are also W4A4 versus W8A8.

Raw standalone kernel timing is never served evidence. A result advances only
after the exact production dispatch, quantization, fallback, and end-to-end
workload contract is measured.

## 6. Approved W4A16 support backlog

W4A16 is an approved Gridbook support addition, not merely an external
comparison. Gridbook will own exact symmetric packing, serialized scale and
metadata accounting, serving-profile declaration, loader/delegation, and
validation. The first execution implementation should reuse upstream vLLM's
`RDNAHybridW4A16` backend; it must not create a duplicate custom W4A16 kernel.

The initial experiment covers BF16, TP1, symmetric/no-`g_idx` W4A16 at group
sizes 128/64/32 (nominal 4.125/4.25/4.5 bpw), with exact serialized bytes
including scales and metadata. Report two explicitly different views:

1. served W4A16-A16 versus production CB-A8; and
2. weight-isolated W4A16 versus an explicitly named CB-A16 contract.

This work is paused until suitable validation hardware is available. It is
unimplemented and unvalidated; `INT4_W4A16_g128` therefore remains research
only and denied by every production Gridbook scope (dense, shared, and packed).
No production profile, exporter, loader, or performance claim may imply support
before the full integration and served-quality gates pass.

## 7. Implementation order

1. Use unified serialized byte accounting and candidate execution identity.
2. Keep fused FP4 opt-in until the served quality/routing gate passes.
3. Rebuild exact-byte 0.6B/4B/27B endpoints and optimized menus over the full
   workload matrix.
4. Check whether per-layer timing tables predict end-to-end ranks. **Still
   open, and it is now the gating question** — the constrained solver of §1
   consumes a measured dispatch table, so whether that table's per-layer
   ratios rank whole-request outcomes correctly is what decides if the
   constraint axis is predictive or merely self-consistent. The
   `d03_exact_rate` harness (§4, gridbook ROADMAP D0.3) prepares the offline
   half of the evidence; the served protocol supplies the other half.
5. Implement the constrained Pareto allocator above. **Done** (ultraplan P5c):
   hard p95 TTFT / p95 ITL / p05 TPS / device-memory constraints as a second
   axis in the byte-budget ratchet, priced from a provenance-gated measured
   dispatch table, with no λ anywhere and the min-Δloss objective and its
   tie-break untouched. Its output is proposal data; the served protocol is
   still what promotes. See §1 and
   `docs/design/constrained_pareto_allocation.md`.
6. Resume the Gridbook-owned W4A16 packing/delegation feature when validation
   hardware and the upstream backend are available.
