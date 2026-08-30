# Requirement discovery by traversal

**Status: normative for new-architecture intake and for every export, once the
walker lands (campaign R5). Adopted 2026-08-21 after the `wo_a` finding;
reframed the same day from closure ledgers to a discovery traversal (Rob: "I
don't believe in checklists. I believe in processes that discover requirements
like a tree traversal.").**

**The walker landed 2026-08-21: `prismaquant/model_walk.py` +
`ModelProfile.walk_claim_rules()` (see `docs/ARCHITECTURE.md` §8.8). Intake
walks are usable now; wiring the walk as an export gate and migrating
probe/cost/footprint/read-traffic onto its edge list is still open.**

**Update 2026-08-22 (`walker/woa-grouped-fisher`): the motivating instance is
priced.** The grouped Fisher accumulator landed (`ARCHITECTURE.md` §8.9) —
exact per-group reductions on the same estimator, flat-plane marginals, the
one global-token normalization — and `DeepseekV4GroupedLinear` left the
probe's skip list for the new `probe.grouped_module_class_names`. DSv4's
`wo_a` walk claim moved from `pin(probe cannot price grouped operands yet)`
to `decide`, and cost cells flow from probe keys with an honestly unmeasured
output-MSE screen. The edge-list migration above remains open: this change
prices `wo_a` through the EXISTING module-boundary hook mechanism; it does not
yet derive the probe's inventory from the walk's edge list.

## The failure class this exists to kill

On DeepSeek-V4, `attn.wo_a` — 17.9% of all decode read traffic — was never an
allocator decision. The probe walks module types, `wo_a` is a parameter
consumed by a grouped einsum rather than an `nn.Linear`, so no cost cell was
ever created, the tensor shipped as 8-bit passthrough by omission, and every
coverage statistic still read as complete. Nothing was wrong with any
individual stage. The defect is structural: **the pipeline's decision universe
is defined by the pipeline's own enumeration.** When the probe enumerates the
units and cost, allocation, bpp, and "coverage" are all computed over that
same enumeration, an omission is invisible by construction — numerator and
denominator come from the same code path.

This is a class, not an event. Instances already in the institutional record:

- AQUA priced 5.5% of an MoE's mass because packed experts were outside the
  A-side's enumeration (fixed in `d61bddf`).
- `units_on_fallback_route = 0` published as clean while no unit had a
  declared lane — a zero over an empty denominator.
- The completeness gate read fused-group claims from a dense-only source, so
  routed claims were invisible to it.
- The byte-budget audit caught overshoot but not undershoot — a one-sided
  check on a two-sided invariant.
- Silent-zero cost lookups ranked broken arms first.
- The vLLM post-load sweep matched by `isinstance`, so a non-subclass was
  skipped silently.
- 73.7% of a built body rode an `Sm80` fallback that its own `selection.json`
  recorded and nothing consumed.
- The test-suite sibling: `importorskip` turned our own import errors into
  green all-skip runs, twice.

## The frame: discover, don't audit

A checklist is a second enumeration, curated by hand, and this repo already
knows what happens to curated prose — the prime directive, and "currency is
not truth." The durable mechanism is generative, not reconciliatory: a
traversal that discovers the requirements from the object itself, the way a
garbage collector discovers liveness from the roots rather than from a list of
allocations someone remembered to record.

**Walk what runs, not what's declared.** The traversal has one root pair:

1. **The loaded model tree** — every module, parameter, and buffer.
2. **One traced forward** — every matmul-family op the model executes, with
   the parameters that feed it, captured at dispatch level. A parameter that
   feeds a matmul is a weight this pipeline must disposition, regardless of
   the class of the module that owns it. This is the edge a module-type walk
   cannot see, and it is exactly where `wo_a` lived.

Every node the walk discovers must be **claimed** at discovery time by exactly
one disposition:

- **decide** — the node enters the allocator's domain: probed, priced,
  allocated, rendered, gated.
- **pin(reason)** — held at source precision on purpose, with a reason string
  (`runtime cannot serve a CB lm_head`).
- **exclude(reason)** — outside the artifact's scope, with a reason string
  (`MTP sidecar, spec-decode off`).

An unclaimed node fails the walk. The walk runs at new-architecture intake —
profile-plugin time, before any campaign spends GPU hours — and again at
export as a gate. Dispositions with reasons are first-class output: they land
on the shipcard and the model card, so the honest state is visible instead of
silent. `wo_a` claimed as `pin(probe cannot price grouped operands yet)` on
day one is a known debt with a name; `wo_a` absent is what bit us. (That
particular debt retired 2026-08-22 when the accumulator landed — see the
update note above — but the pin mechanism stays for the next class no
accumulator covers.)

## One enumeration, every consumer derives

The walker's output — the node list and the parameter→op edge list — is the
single universe. The probe attaches marginals per edge. Cost cells, footprint
bytes, read-bytes-per-token, and the route map are all computed over the same
edge list. No stage re-enumerates. This kills the class at the root: there is
no second universe to disagree with the first, so a coverage claim cannot be
vacuously true.

The practical consequence on DSv4: discovering the `wo_a` parameter→einsum
edge is the same machinery the probe needs to hook that einsum and collect
marginals for it. The walker is not process overhead on top of the `wo_a`
enablement — it is the `wo_a` enablement, generalized so the next
architecture's `wo_a` is discovered mechanically.

## The stamped views

The traversal output projects into four per-artifact tables — views over one
walk, not independent checks:

| view | quantity | catches |
|---|---|---|
| V1 | Bytes on disk, by disposition | Unclassified tensors; budget undershoot and overshoot |
| V2 | Read bytes per decode token (dense p=1, routed p=topk/E), by disposition | `wo_a`-class omissions, visible on the shipcard as labeled traffic |
| V3 | Executed matmuls → serving routes, attested from the runtime's published contract (principle 14) | Fallback kernels, lane-ineligible layers, decode-vs-batch regime splits |
| V4 | Cost-model mass over V1/V2 totals | "Priced 5.5% of the model" — the unpriced remainder is itemized, not averaged away |

V1's and V2's totals reconcile against the checkpoint's safetensors header —
the one enumeration even the walker cannot get wrong, because it is the
artifact itself.

## Honest limits

The traversal discovers everything that executes in the traced forward. A
path the trace does not exercise (a conditional branch, a spec-decode-only
module) is discovered only by the model-tree half and dispositioned there —
trace coverage is a real parameter of the walk and gets recorded with it.
Quantities outside both roots (KV-cache traffic, activation traffic) are
named exclusions of the views, not of the walk. And a genuinely unconceived
dimension still needs the outside-report channel — a user with a profiler —
which is part of the system and worked this time. The walker's job is to make
sure the same lesson is never paid for twice.
