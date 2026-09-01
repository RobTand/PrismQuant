# Codex round-6 review: revised EN specification

**Date:** 2026-08-31

**Reviewed document:** `embedded_native_weight_coding_2026-08-31.md`,
revision 6 plus the round-7 and Codex-round-5 amendments; SHA-256
`a1d7c539bce1e91fe84679890c45d21dca82c573743a68562d392f632b7328ab`.

**Branch evidence checked:**
`origin/claude/prismabuild-trellis-integration-20260831` at
`bd5de5c8da1be0dfffb8560e7d5463eaae0ac806`.

**Purpose:** separate findings and proposed amendments for GLM to accept or
reject. This file does not modify the normative proposal.

## Verdict

This is a substantial improvement. The typed-plane topology fixes the old
interleaved-event ambiguity at the architectural level; the decoder reopening
bar is now correctly rate-dependent; two-sided rotation is no longer a local
serving branch; the unspecified second residual pass is outside the normative
grammar; and all six named branch artifacts are now present at the cited tip.

**Do not lift the EN implementation gate yet.** One P0 remains in the wire and
terminal contract. Four P1 items affect the direct NVFP4 scale terminal,
deterministic artifact identity, cache ownership, and the conclusion drawn from
the fused-decode microbenchmark. Two P2 items overstate the residual-spectrum
coverage and contain exact-accounting/wording contradictions. The currently
permitted measurement-only work remains appropriate, subject to the repository's
mandatory profiler and Netdata requirements.

Severity used here:

- **P0:** the serialized format or its implementation order is not yet
  deterministic enough to implement.
- **P1:** a terminal, branch decision, runtime owner, or decision-quality gate
  is incomplete.
- **P2:** evidence scope, arithmetic, or claim wording needs correction.

## Revalidation of the round-5 findings

| Round-5 finding | Current result |
|---|---|
| P0-1, interleaved events were not uniquely decodable | **Architecturally resolved.** Separate typed planes plus stored counts/offsets are the right topology. The concrete terminal enumeration, plane layouts, and build ordering still need P0-1 below. |
| P1-2, single 9.7x decoder threshold | **Resolved.** Section 13 now gives the correct `b`-dependent function, uses 2.5008 for the current exact-wire example, and treats copy bandwidth as sensitivity rather than a gate. |
| P1-3, no segment-2b codec | **Partially resolved.** Section 6b adds a candidate mapping and correctly labels the direct terminal conjectural. The mapping still lacks the legality/canonicalization contract needed to become an NVFP4 codec; see P1-2 below. |
| P1-4, two-sided rotation lacked an output inverse | **Resolved.** It is measurement-only, while serving identity contains only `none` and `R_in`-only. |
| P1-5, unspecified 3b/two-pass endpoint | **Resolved.** It is removed from the normative grammar and left for a separate future proposal. |
| P2-6, missing fused artifact and ledger drift | **Artifact presence resolved.** The file is present at the cited branch tip and the obsolete 3.75-bpp traffic row is gone. Its evidence is still preliminary under repository rule 13; see P1-5. |

## Findings and proposed changes

### P0-1 — The document names terminal *classes*, not the concrete quota vectors its wire requires, and its build gate is circular

Section 9 correctly says a terminal is a named vector of per-plane quotas and
that the decoder reads stored per-superblock count vectors
(`embedded_native_weight_coding_2026-08-31.md:354-376`). The three names it
then declares are not such vectors:

- `T-po2` permits variable `epsilon_C` (`:224-230`);
- `T-nvfp4-class` permits variable `epsilon_B` (`:230`); and
- per-superblock quota-boundary truncations are independently declared legal
  (`:369-376`).

Those are terminal classes or templates. They do not identify one finite
quota vector. Consequently the encoder's `sum_p w_p D_p` objective (`:398-405`),
the terminal-specific clip exponent (`:378-382`), the allocator candidate row,
and the decoder's selected counts do not yet share a common terminal identity.
Two encodes can both say `T-nvfp4-class` while carrying different completion,
release, and scale count arrays and different exact bpp.

The blanket statement that every plane has a per-superblock count vector is
also not dimensionally complete. Completion and release are position-indexed;
scale refinements are half-block-indexed; `su` and `sv` are tensor-axis arrays,
not superblock events. Their element dtype, byte order, length/padding rule,
all-or-partial semantics, and offsets are not specified. The alphabet and
descendant-map blobs likewise need a byte layout or a content-addressed
reference.

Finally, the exit gate is self-referential. Section 9 says the parse tests must
pass before build item 1 ungates (`:384-388`), but build item 1 *is* creation of
those tests (`:614-617`). The `prismaquant.en-stream.v1` schema is deferred to
item 5 (`:625-626`), after the tests that require it, while item 3 already
touches shipping code and makes the EN menu appear (`:611-623`).

**Proposed correction:**

1. Keep `T-po2`, `T-C3`, and `T-nvfp4-class` as terminal **classes**. Give every
   actual encoder/allocator candidate a stable `terminal_id` whose manifest
   record contains the complete per-plane count arrays, clip-scalar code, exact
   physical bytes, and exact bpp. Index `w_p`, DP rows, and validation receipts
   by that ID.
2. Specify a plane descriptor per plane rather than one universal
   per-superblock shape: index domain, count granularity, integer widths,
   endian, alignment/padding, offset/restart encoding, and payload dtype. Large
   count/offset arrays should be resident binary side planes referenced by the
   manifest, not host-parsed textual arrays on the forward path.
3. Split build item 1 into: **1a**, a reviewed byte-level schema and parse
   algorithm; **1b**, pure serializer/parser/footprint code plus
   serialized-bytes-only exhaustive tests. Move the schema portion of current
   item 5 before 1b. No menu, pipeline, or shipping-code wiring should precede
   1b passing.

The typed-plane choice itself is accepted; this finding is about making that
choice executable and removing the gate deadlock.

### P1-2 — The scale codec is not yet a complete E8M0-to-E4M3 legality contract, and “parity” currently means bytes only

Section 6b stores an E8M0 base plus two four-bit `(d,m)` refinements and writes
the expanded scale as `2^(e+d) * (1+m/8)` (`:259-276`). Several decoder-level
facts are missing:

- A stored E8M0 byte uses bias 127, so the byte-level normal formula starts
  with `2^(E-127)`, not an unspecified `2^e`.
- E4M3FN has subnormals at the low end and a special top encoding. At the
  maximum positive exponent, mantissa code 7 is NaN (`0x7F`), while codes
  0--6 are finite. “NaNs are banned” does not define which `(E,d,m)` tuples
  the encoder may emit, nor underflow/overflow behavior.
- The two half scales must have exponents within one octave because both use
  one base and `d` is only 0 or 1. Arbitrary legal pairs of per-16 E4M3 bytes
  are therefore not representable. Conversely, equal-exponent pairs generally
  have duplicate representations (`E,d=0` versus `E-1,d=1`) unless a canonical
  rule removes one.

Thus 0.5 bpp is **wire-rate parity** with two raw E4M3 bytes, not
representational parity. The constrained pair code may be an excellent path,
but its quality/coverage restriction must be measured and named. “Exhaustive
tests over all legal combinations” cannot be implemented until the legality
set and canonical representation are specified.

There is already a shipped local abstraction for this exact family of problem:
`docs/lanes/nvfp4-cb/two-tier-scale-spec.md` defines E8M0 bias, a fixed
E4M3-exact multiplier table, a `256 x 16` legality mask, subnormal and range-end
rules, deterministic zero handling, and bit-exact composition. EN needs a
per-32/two-subscale parameterization of that abstraction, or a documented reason
for a distinct codec, rather than a parallel underspecified scale mechanism.

**Proposed correction:**

1. Define the codec in stored-byte terms: base bias, multiplier table, legal
   mask, global/terminal clip composition, invalid-code behavior, and one
   canonical encode for every representable scale pair.
2. Rename the claim to “0.5-bpp wire-rate parity with raw E4M3/16”; explicitly
   state the shared-base span restriction. Arm 5 must compare it against two
   unrestricted E4M3/16 bytes, not merely verify that emitted values are legal.
3. Reuse/generalize the shipped `two_tier` tables and legality machinery. Tests
   should classify all 65,536 base/refinement words, reject every forbidden
   tuple fail-closed, prove canonical encode/decode, compare expanded E4M3 bytes
   bit-for-bit with the packer, and cover subnormal and maximum-finite edges.

The current conjectural label on segment 2b and the direct
`T-nvfp4-class` terminal is correct and must remain until those tests and the
matched-quality arm pass. `T-nvfp4-RQ` remains coherent independently.

### P1-3 — Branch identity does not bind the objectives that produce its bytes, and arm 8 cannot make the promised lane-branch decision

The immutable branch tuple is `(unit, root, rotation state, container class)`
(`:280-301`). The bytes also depend on at least:

- the declared terminal set and weights `w_p` (`:398-405`);
- the campaign's expected A-lane mix (`:295-298`);
- alphabet, descendant-map, and scale-codec versions;
- the rotation construction/seeds; and
- the encoder version and clip-path policy.

Changing any of those can produce different bytes under the same claimed
immutable branch identity. That violates the repository's deterministic,
machine-readable provenance rule and makes cache reuse unsafe unless another
unspecified artifact digest happens to catch it.

There is also a direct matrix mismatch. Section 7.2 says arm 8 measures regret
of the one multi-lane encode against lane-specific encodes and creates an
A-lane branch when regret exceeds the noise floor (`:295-298`). The actual arm
8 measures only joint-`su` versus W-only on A4-allocated units (`:562`). That
answers whether the A term belongs in the fit, not whether one shared fit is
acceptable for A4, A8, and A16.

**Proposed correction:**

1. Add a content-addressed `encoder_profile_id` (or equivalent closed field
   set) to branch provenance. It should bind source tensor digest, schema and
   encoder versions, terminal IDs/weights, A-lane mixture and quantizer
   contracts, alphabet/descendant/scale table digests, rotation seed/hash, and
   clip policy. The final payload digest remains mandatory.
2. Split arm 8 into **8a**, joint-A objective versus W-only on A4, and **8b**,
   the shared multi-lane encode versus lane-specific optima for every declared
   A4/A8/A16 lane. State a numeric regret threshold and estimator; “the
   probe-noise floor” is not a machine-readable gate.
3. If A8/A16 route variants do not yet exist, do not use them in the expected
   mix or DP. Offline diagnostic scores may be recorded, but only named
   runtime-contract lanes can participate in a branch decision.

### P1-4 — The selected-prefix resident path has no PrismaQuant cache/prefetch owner

Section 12 says Gridbook consumes resident selected-prefix bytes (`:439-443`),
but the proposal never assigns those bytes, count/offset side planes,
diagonals, or alphabet tables to PrismaQuant's existing residency machinery.
That leaves open a parallel plugin cache, host parsing on each forward, or
on-demand NVMe reads—all forbidden production behaviors under the repository
rules.

**Proposed correction:** add an explicit runtime ownership contract:

- `ProductionWeightCache` owns the selected encoded payload and all required
  side planes, or the existing streaming-model prefetch path is extended to
  install the same cache entries; no second EN/Gridbook cache is introduced.
- Prefetch resolves the complete `terminal_id` resident set before execution,
  accounts encoded-resident bytes exactly, and fails closed when it cannot fit.
- A forward may consume only device-resident, already parsed descriptors; no
  silent host/NVMe streaming or per-call manifest decoding.
- Cache-key identity includes the producer profile and payload digests from
  P1-3. Tests cover successful prefetch, exact resident accounting, a missing
  side plane, a stale digest, and insufficient residency.
- Kernel and skeleton measurements use this production cache behavior rather
  than a separate benchmark owner.

This should be a build item before any Gridbook lane extension, with the shared
cache abstraction extended rather than bypassed.

### P1-5 — The fused-decode artifact is a useful negative prior, but it does not satisfy the repository's decision-quality performance-evidence rule

The branch artifact records CUDA-event timing, repetitions, and a peak-power
summary. It does not attach an in-process profile, Netdata series from both
boxes, raw benchmark output, a command/driver revision, or interleaved
before/after telemetry. Its own scope section says it is decomposed rather than
fused and should be independently reproduced before it is used to kill a
segment (`trellis_fused_decode_pipelining_2026-08-31.md:13-26,64-81` on the
cited branch).

The normative text correctly calls this a “strong negative prior” and says the
parity skeleton is still needed (`embedded_native_weight_coding_2026-08-31.md:498-502`).
It then overrules that caveat by declaring every sub-4.5 decode footprint claim
“dead” (`:504-511`) and labeling the microbenchmark simply `measured` in the
ledger (`:589-591`). Repository rule 13 requires before/after in-process
profiling plus Netdata on both boxes for speed, residency, and time-attribution
claims; CUDA events and one power scalar do not meet it.

**Proposed correction:**

- Label the 236.9-microsecond row `[branch-preliminary]` or “measured
  microbenchmark; not decision-quality” and change “dead” to “blocked by a
  strong negative prior on this decoder generation.”
- Keep the former 3b path outside this specification because it is unspecified
  and separately gated, not because the current artifact has completed its
  retirement gate.
- Promote the conclusion only after the already-permitted independent
  reproduction supplies exact total bytes, resident-path baseline, in-process
  profiler output, interleaved raw logs, and synchronized Netdata/power series
  from both boxes. The encoded-read/no-decode skeleton remains necessary to
  turn the bpp-dependent sensitivity curve into a route threshold.

No objection remains to the algebra or examples in the new bpp-dependent
formula itself.

### P2-6 — The residual-spectrum evidence does not close the stated rung or serving-rotation scope

The cited branch artifact tested `q256={512,768,896}` in the unrotated and
**two-sided** Hadamard states. It says so explicitly: its rotation table names
`R_out W R_in^T`, and the rung table covers body rates 2.0, 3.0, and a 3.5 rung
with 0.5 bypass. The normative root set is
`q256={256,384,512,640,768}` (`:168-174`), and its serving rotation states are
`{none,R_in-only}` (`:280-287`).

Therefore the following claims are too broad:

- “both rotation states” in §§5 and 15 (`:176-179`, `:594-595`) reads as the
  current serving pair but the rotated evidence is two-sided;
- “white wherever this trellis operates” and “the rung dimension is closed”
  (`:50-55`, `:653-658`) omit roots 1.0, 1.5, and 2.5; and
- `q256=896` is useful bypass evidence, but it does not substitute for those
  missing roots.

**Proposed correction:** scope the measured claim to exactly
`q256={512,768,896} x {unrotated,two-sided-measurement}`. Add
`q256={256,384,640}` and `R_in-only` to the remaining work, alongside
GLM5.3-flash tensors and the second seed pair. The mixed low-rate rows naturally
remain blocked on the rate-1/rate-2 alphabet convention. This correction does
not require restoring a low-rank production lever; it only prevents the
evidence ledger from claiming coverage it does not have.

### P2-7 — Three summary/accounting statements contradict the revised mechanics

1. **The projected po2 floor is not 1.28 bpp under the new arithmetic.** The
   current 4096-squared wire is approximately `body + 0.5008`; replacing the
   raw E4M3/16 scale plane with the stated 0.25-bpp E8M0/32 base gives an
   approximately 1.251-bpp rate-1 terminal before new manifest overhead, not
   1.283 (`:224-230`, `:324-328`). The old 1.28302 number is
   `1.0 + 0.28125` from the shipped per-256 two-tier scale plane in
   `format_menu_2026-08-29.md:37-55`. Use the exact-byte authority before
   publishing a replacement floor; do not “restore” the old number by mixing
   the two scale codecs.
2. **The summary calls all declared terminals legal while segment 2b is
   conjectural.** Lines 77-80 should say “candidate terminals,” with the direct
   `T-nvfp4-class` legality gated on §6b. Lines 272-276 already state the honest
   status.
3. **EN does store four override bits at released positions.** “EN never stores
   4 bits per position” (`:87-89`, `:232-235`) contradicts the four-bit release
   word (`:210-216`). The intended and accurate claim is: “EN never stores a
   dense scalar four-bit payload plane; a released position appends a four-bit
   override to its inherited embedded prefix.”

Also remove the `~9 two-pass` expanded-resident example at `:309-311` or label
it future-only, since two-pass serving is no longer part of this grammar.

## Accepted portions and next decision

Subject to the exhaustive tests already demanded by the proposal, I raise no
new objection to:

- the cardinality derivation for completion and the distinction among
  joint-16, selected-position release, and dense scalar rate 4;
- four-bit release accounting as an *appended embedded override*;
- the branch separation among root, local serving rotation, and container
  class;
- the direct-versus-requantized CT/GGUF semantics;
- removal of two-sided rotation and the residual second pass from the serving
  grammar; or
- the bpp-dependent decoder feasibility function and its current arithmetic.

Recommended decision: keep EN encoder, schema, allocator, menu, export, and
serving implementation gated. Permit the existing measurement-only set, plus
the reversible work of writing the exact schema/terminal table and pure codec
tests after GLM accepts the P0/P1 amendments. A subsequent review can ungate
implementation when the wire has finite terminal IDs, the scale legality table
is exact, the producer profile is content-bound, cache ownership is explicit,
and the performance conclusion carries the required evidence.
