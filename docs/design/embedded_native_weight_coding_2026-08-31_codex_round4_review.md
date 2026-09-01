# Codex round-4 review: embedded-native weight coding

**Date:** 2026-08-31

**Reviewed document:** `embedded_native_weight_coding_2026-08-31.md`, revision 3

**Review scope:** design correctness, format topology, exact-rate claims,
artifact/serving semantics, measurement provenance, and readiness to implement.

## Verdict

**Changes are still required before EN encoder, schema, allocator, menu, export,
or serving implementation begins.** Revision 3 correctly accepts the seven
round-3 findings and materially improves the proposal. In particular, it fixes
the root-rate claim, Blackwell/CDNA4/RDNA4 targeting, arbitrary-alphabet issue,
side-information accounting policy, encoded-versus-resident traffic distinction,
A-side/rotation contract status, and the misuse of `allocation_regret`.

The remaining blockers are not requests for more polish. The document still
lacks a self-consistent refinement grammar, and its canonical artifact does not
contain all dimensions that the encoder says change the encoded bytes. The
compressed-tensors terminal also cannot preserve the `su` operator as currently
specified. These must be resolved in the folding revision.

Severity used below:

- **P0:** the current format cannot be encoded/decoded or selected as claimed.
- **P1:** a serving branch or load-bearing quality claim is not yet valid.
- **P2:** evidence or document authority is insufficient for the stated status.

## Findings

### P0-1 — Segment 1 still has no self-consistent code grammar or rate budget

The in-place table caps segment 1 at one bit per weight
(`embedded_native_weight_coding_2026-08-31.md:324-335`). Revision 3 instead says
that a scheduled rate-`R` alphabet needs `(3-R)` completion bits per column
(`:1245-1258`). That is two bits for an `R=1` column and an average 1.5 bits for
an `r0=1.5` root, so the discrepancy is payload-sized, not provisional header
or alignment overhead.

The mapping is also under-specified. One partner bijection doubles an alphabet.
It can complete an eight-code `R=2` alphabet to 16 codes, but it cannot complete
a four-code `R=1` alphabet; that needs a four-way descendant map or two explicit
nested pairing levels. At `R=3`, the alphabet already contains 16 values but
the trellis still constrains sequences. Reaching the scalar/RTN rate-4 ceiling
requires the separate constraint-release/bypass bit described by old sub-mode B.
Revision 3 neither generalizes that stage to lower-rate roots after completion
nor includes it in the segment capacity, while the summary still promises each
root a continuous path toward the 4-bit-class terminal (`:73-89`).

Consequently, the following are not derivable from the current grammar:

- which legal prefixes exist for each integer scheduled column rate;
- the maximum source rate reachable from each fractional root schedule;
- whether “full 16-code grid” means per-position reachability, a rate-3
  constrained sequence, or independent scalar rate-4 selection;
- the T-portable/T-nvfp4 endpoint rates; and
- the source-byte and fused-read traffic numbers.

**Required correction:** specify a decoder-level refinement tree for each
scheduled `R in {1,2,3}`, separately name alphabet-completion and
constraint-release stages, state their exact per-column capacities, and derive
fractional-root totals from the actual schedule. Add tiny exhaustive tests that
prove nesting, unique decode, legal truncation, and expected code cardinality at
every prefix. Only then update the terminal and traffic tables through the
exact-byte authority. “Provisional arithmetic” is appropriate for headers and
padding, but cannot stand in for missing body bits.

### P0-2 — The accepted root forest is still smaller than the encoder's real branch graph

Revision 3 defines one stream per root rate (`:1200-1215`) and the canonical
bundle as one forest per unit (`:1288-1298`). Elsewhere, however, the base wire,
diagonals, and rotation are jointly fit (`:432-449`); the `su` optimum is re-fit
per A4/A8/A16 serving lane (`:888-908`); and rotation is a three-state allocator
dimension (`:1356-1377`). Rotation changes the source tensor itself, so its base
bytes cannot be shared. A lane-specific joint objective may change the base bytes
as well.

The allocator therefore cannot choose A-side and rotation state after selecting
one of five stored root streams unless those candidate encodes actually exist.
At minimum the format is a `root x rotation-state` branch graph. The A-side must
either be another stored branch dimension or use one explicitly declared
multi-lane objective whose regret against lane-specific encodes is measured.
Per-group rotation constraints for fused siblings and packed experts then couple
those branch choices.

This also changes storage accounting. A five-root source bundle already stores
multiple independent encodes; multiplying it by rotation and possibly A-side
states is not a sub-4-bit canonical checkpoint. Selected-prefix serving bpp may
still be sub-4, but canonical-bundle bpp, derived-artifact bpp, encoded-resident
bpp, and expanded-resident bpp are four different quantities and must be named
and charged separately.

**Required correction:** define the complete branch DAG and immutable branch
identity before defining `prismaquant.en-stream.v1`. State which planes/bytes
may be shared, which candidates are physically present, and what exact object
the production DP selects. Extend exact-byte accounting to the forest directory,
branch offsets, duplicated or shared scalar planes, and the whole canonical
bundle. A solver must fail closed rather than select a priced state with no
corresponding bytes.

### P1-3 — T-nvfp4 is not a semantics-preserving compressed-tensors terminal when `su` is present

The decoded EN operator is `sv[i] * s_b * q * su[j]` (`:230-258`), and T-nvfp4
comes after segment 2a, so it includes the channel diagonals (`:341-349`). The
artifact correction then says vanilla compressed-tensors consumes a materialized
byte-legal NVFP4 tensor (`:1288-1298`). Those are not generally the same
operator.

PrismaQuant's native export has group-16 weight scales plus one scalar
`input_global_scale`; the latter is explicitly per-tensor and emitted as a
one-element tensor (`prismaquant/export_native_compressed.py:980-988` and
`:2395-2413`). An arbitrary `su[j]` varies inside a 16-weight block and cannot be
represented by that contract. For routed experts it also cannot generally be
folded into a shared producer. Materializing `Dv * Q * Du` and quantizing it
again to NVFP4 is legal, but it is a second lossy quantization and is not the EN
terminal whose diagonal gain was measured. GGUF has the same distinction.

The AMD wording has a related serving gap. The current Gridbook lane is
CUDA/CUTLASS-only (`docs/ARCHITECTURE.md:1120-1126`, `:6567-6576`). AITER's
standard MXFP4 path on gfx950 does not by itself consume EN bytes or apply EN
diagonals. Thus T-portable is representable in native operand types on CDNA4,
but is not yet a served EN lane there.

**Required correction:** make CT/GGUF branches explicitly unrotated and either
diagonal-free, lawfully producer-folded, or lossy requantized derived artifacts.
The lossy option needs its own render, exact bytes, probe/KL row, and serving
gate. Do not call it the same T-nvfp4 terminal. Likewise, say “native-operand
representable on CDNA4” until a released consumer contract and served-speed gate
exist for EN bytes plus side operators.

### P1-4 — The canonical prefix order contradicts both marginal ordering and clip tracking

Revision 3 fixes the canonical order as body -> refinement -> diagonals -> scale
mantissa -> residual (`:1200-1210`). The encoder section simultaneously calls
the diagonals the cheapest first edges, at roughly 37 dB/bit (`:432-440`). A
literal prefix cannot spend the approximately 0.02-bpp diagonal until all prior
refinement bits have been bought, so it cannot produce the rate-distortion order
the document claims.

The same order does not implement revision 2's clip-path answer. Segment 2b is
said to walk the optimal clip as effective resolution grows (`:945-980`), but
all payload refinement occurs before segment 2b. Intermediate segment-1 prefixes
therefore use neither the later scale bits nor, under the stated order, a
prefix-specific exponent. A manifest exponent can repair a declared deployment
terminal, but supporting every advertised internal prefix requires defining
which exponent belongs to each legal prefix and charging or deterministically
deriving it.

**Required correction:** either define one global decode-deterministic event
order that may interleave diagonal, payload, and scale events by marginal value,
or admit that the representation is a DAG with a finite set of legal terminals
rather than one ordered stream. Then state exactly which internal truncations
are supported. Include the diagonal-present and diagonal-absent base prefixes in
the multi-prefix objective if both remain legal.

### P1-5 — “Multi-prefix Viterbi” is an objective, not yet an executable Viterbi algorithm

The new objective is the correct direction (`:1245-1267`), but no recurrence is
given. Partial-prefix distortion depends on the refinement placement order, and
that order is itself a sort over already-decoded anchors in a superblock
(`:1269-1279`). Fixed quotas and order statistics couple multiple emissions, so
the cost is not automatically additive over the existing trellis state. A
standard endpoint Viterbi cannot optimize it merely by replacing one scalar
metric with `sum_p w_p D_p`.

Also, “at minimum the base and terminal” does not protect the internal prefixes
that carry the product claim. If only those endpoints are optimized, middle
prefixes can regress arbitrarily while the objective improves.

**Required correction:** provide the state, recurrence, and complexity for the
exact algorithm, or label the encoder as an alternating/beam/greedy heuristic.
Declare the complete supported prefix set and its weights. Validate tiny cases
against brute force, then profile the implementation at representative shapes
before claiming it remains GPU-bound at 33k-expert scale.

### P1-6 — The current document is not yet an authoritative implementation spec

Revision 3 explicitly schedules a folding revision (`:1428-1444`), and that
fold is required. The body still says round 2 is pending (`:3-5`), calls the
format one stream in places (`:98-101`, `:432-440`), says the po2 plane is native
on RDNA4 (`:293-308`), describes the three containers as truncation profiles
(`:398-414`), and retains measurement arms later superseded by sections 17 and
19. The reader contract says the only load-bearing ledger is section 12
(`:65-67`), but newer claims live in sections 17.6 and 19.10.

These are not harmless historical notes because they disagree on wire order,
vendor execution, artifact semantics, and gates. Pointer-based supersession
forces an implementer to synthesize a fourth design and makes deterministic
review impossible.

**Required correction:** fold sections 17-19 into the normative body, retain a
short revision history, and leave exactly one format grammar, terminal table,
artifact model, arm matrix, claim ledger, build order, and attack surface. Run a
second review against that folded text before implementation.

### P2-7 — The new round-3 measurements are not auditable from the named artifacts

Section 18 names `scratchpad/pred.py`, `pred2.py`, and
`dq-runs/trellis-serve-smoke-20260831/` as the drivers/logs (`:1085-1091`), then
promotes the rotation and residual-spectrum rows to measured status
(`:1411-1420`). None of those drivers or logs is present in this checkout,
`/home/rob/dq-runs`, or the named integration branch. The four round-1 result
documents do exist on
`origin/claude/prismabuild-trellis-integration-20260831`, consistent with the
document's branch-scoped caveat; the section-18 artifacts do not.

The numerical conclusions may be correct, but the current evidence state is
`review-reported`, not repository-measured. This matters because the results
delete the rotation-based rank-eligibility rule and place a negative prior on
segment 3a.

**Required correction:** commit the exact drivers, immutable inputs or input
digests, commands, environment/container identity, raw per-tensor rows, and
summary logs. Until then, downgrade the section-18 and section-19 ledger rows to
`review-reported`. Preserve the stated small-model/unweighted/one-rung scope
after the artifacts land.

## Answer to section 19.12(5): permitted measurement

Yes. Re-running the existing section-18 weight-space diagnostics at additional
existing TCQ rungs and on GLM5.3-flash tensors counts as review measurement, not
EN encoder implementation, provided it uses the existing encoder unchanged.
It is permitted under the prior verdict.

The result must remain scoped as an unweighted residual-spectrum/dispersion
diagnostic. The current GLM corpus has no importance vectors, so it cannot be
used to infer AURA allocation quality, activation-aware loss, or production KL.
Record tensor/input digests, q256, exact rotation sidedness, seeds, commands,
container/environment identity, raw rows, and logs. This permission does **not**
extend to implementing partner completion, multi-prefix encoding, the EN stream
schema, DP/menu wiring, export, or runtime kernels.

Arm 1 and the scale-penalty-only portion of arm 5 also remain permitted as the
proposal states. Any performance claim still requires before/after in-process
profiling plus Netdata/power evidence and work-per-joule ranking.

## Exit gate for the next review

The next review should receive one folded document that includes:

1. an executable refinement grammar with exact payload accounting;
2. the full stored branch DAG and separate bundle/selected/resident byte metrics;
3. honest CT, GGUF, Gridbook-CUDA, and CDNA4 terminal semantics;
4. one legal-prefix/event-order definition;
5. an implementable multi-prefix optimization algorithm;
6. one guideline-bar measurement matrix; and
7. attached artifacts or correctly downgraded evidence labels.

Until those are present, measurement-only work above is appropriate; EN
implementation and pipeline wiring are not.
