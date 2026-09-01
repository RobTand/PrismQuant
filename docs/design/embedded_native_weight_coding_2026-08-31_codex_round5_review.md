# Codex round-5 review: revision-6 folded EN specification

**Date:** 2026-08-31

**Reviewed document:** `embedded_native_weight_coding_2026-08-31.md`, revision 6

**Purpose:** separate review/amendment record for GLM to accept or reject. This
file does not modify the normative proposal.

## Verdict

Revision 6 resolves the seven round-4 findings in the intended direction. The
fold is now genuinely authoritative; the abstract completion grammar closes;
the stored topology is a branch DAG; bundle, selected, encoded-resident, and
expanded-resident bytes are separated; CT/GGUF diagonal semantics are honest;
the encoder is correctly labeled a heuristic; and unavailable evidence is
downgraded.

**Changes are nevertheless still required before EN implementation.** One P0
remains: the proposed interleaved event stream is not uniquely decodable from
the information specified. Five narrower issues affect the kernel threshold,
the scale terminal, two-sided rotation, the residual suffix, and evidence
status. Measurement-only work remains appropriate.

Severity:

- **P0:** the current format cannot be serialized/decoded deterministically.
- **P1:** a terminal, branch, or decision gate is not valid as stated.
- **P2:** evidence labels or measurement-matrix text need correction.

## Resolved from round 4

No further objection is raised to these revisions, subject to their stated
measurements and exhaustive tests:

1. The descendant-set construction in §6 is cardinally coherent. At C-full,
   `|A_R| * |D(a)| = 2^(R+1) * 2^(3-R) = 16`; fractional roots also total
   exactly three body bits when averaged over their integer-rate schedule.
   Distinguishing joint-16, selected-position release, and scalar rate-4 fixes
   the old ambiguity.
2. The branch identity now acknowledges root, rotation, and container class,
   and it correctly separates whole-bundle bytes from the selected artifact.
   The multi-lane A objective is an acceptable staged choice if its lane mix is
   versioned in encode provenance and arm 8 promotes A-side to a stored branch
   before DP use when regret exceeds the declared floor.
3. CT/GGUF branches now exclude, fold, or explicitly requantize `su`; CDNA4 is
   correctly described as operand-representable rather than served.
4. The document is folded, the claim ledger is singular, the encoder is called
   an alternating heuristic, and the round-3 artifactless rows are labeled
   `review-reported`.

## Findings and proposed changes

### P0-1 — Anchor order does not make the interleaved event stream uniquely decodable

Section 9 says completion, release, and scale-mantissa events are interleaved
and ordered by descending decoded anchor magnitude, so placement costs zero
bits (`embedded_native_weight_coding_2026-08-31.md:297-307`). That key identifies
a weight position. It does not identify:

- the event type at that position;
- its width (completion is one bit per level, release is a four-bit code, and a
  scale event is block-level rather than position-level);
- how many events of each type a superblock owns; or
- the offsets at which variable-width per-superblock streams restart.

The encoder then performs lambda-greedy type/placement selection
(`:326-331`). Unless that choice is itself a deterministic function of decoded
base bytes, the decoder cannot reconstruct it. If per-superblock quotas vary,
the quota vector is side information even when the first `k` positions within a
block use a free canonical order. For a single event family, an arbitrary
`k in [0,16]` already has 17 states; canonical placement removes the mask, not
the count. Interleaving several families also needs a type schedule. The
explicit-mask comparison at `:305-307` therefore does not establish zero total
location/schedule cost.

This also leaves scale ordering dimensionally undefined: a per-16 scale event
has no single weight anchor from which to obtain the stated positional key.

**Proposed correction:** choose and specify one of these wire contracts:

1. A deterministic event catalogue derived from base bytes, including a fixed
   event-type priority/tie-break and a block-level key for scale events. A legal
   terminal selects a deterministic prefix of that complete catalogue; or
2. Separate typed planes, with per-superblock counts, offsets/restart tables,
   and any merge schedule stored and charged by the exact-byte authority.

In either case, define the bitstream parse algorithm before the encoder
objective. Add exhaustive tests that start only from serialized bytes and prove
unique parse, unique decode, truncation at every declared quota boundary, and
exact agreement between physical bytes and the footprint accountant. Until
then, build items 1, 2, and 5 remain gated; a decoder cannot be tested against a
wire that does not specify its event types and counts.

### P1-2 — The ~9.7x decoder reopening bar uses body bits, is rate-specific, and has an unsupported caveat

Section 13 compares a 4.5-bpp expanded tile with a “2.0-bpp wire” to derive
24.4 microseconds of saved traffic and a 9.7x decoder target (`:409-419`). The
same folded document records that q256=512 is **2.501 total wire bpp**
(`:269-271`, `:485`), and the branch artifact explicitly warns that every NVFP4
comparison must use the 2.51-style footprint, never the 2.0 body rate
(`trellis_first_artifact_plan_2026-08-31.md:223-243` on the integration branch).

Using the document's own 214.5 GB/s proxy on a 4096-squared tile gives:

| selected encoded bpp | bytes saved vs 4.5 | allowed decode time | speedup from 236.9 us |
|---:|---:|---:|---:|
| 2.0000 body-only | 5.243 MB | 24.44 us | 9.69x |
| 2.2500 projected body+po2, before side overhead | 4.719 MB | 22.00 us | 10.77x |
| 2.5008 current exact wire | 4.193 MB | 19.55 us | 12.12x |
| 3.7500 claimed EN traffic point | 1.573 MB | 7.33 us | 32.31x |

The general proxy is

`required_speedup(b) = T_decode / (((4.5 - b) * N * K / 8) / B_effective)`.

It diverges as selected bpp approaches 4.5. A single ~10x number therefore
cannot gate the proposal's 3-4.25-bpp quality band. It is only the body-only
q256=512 example, and even that example omits the current scale plane.

The caveat at `:418-419` also has the algebraic direction backwards if both
streams are bandwidth-limited: lowering effective bandwidth increases the time
saved by removing a fixed number of bytes and therefore lowers the required
decoder speedup. If launch or compute hides those bytes, however, the saving
can instead shrink toward zero. Copy bandwidth alone cannot decide which case
applies.

**Proposed correction:** replace the single threshold with the bpp-dependent
formula, using exact selected-prefix bytes including scales, side planes,
quotas, and offsets. Treat copy bandwidth as a sensitivity input, not a gate.
The actual parity quantity is

`T_resident_path - T_same_fused_path_with_encoded_reads_but_no_decode`,

which needs a directly measured skeleton/prologue on the target route. Until
that exists, retain the current result as a strong negative prior, not a precise
reopening threshold. Delete the ledger claim “traffic 3.75 < 4.25 < 4.5” until
3.75 is derived by the exact-byte authority and tied to a named terminal.

### P1-3 — Segment 2b still lacks a decoder-level codec, so the direct NVFP4-class terminal is not specified

The folded text gives an exact grammar for C and B, but segment 2b is only
described as “po2/32 -> E4M3/16, <=0.25 bpp” and interleaved along a clip path
(`:263-295`, `:297-316`). It does not define:

- how one E8M0/32 base plus at most eight refinement bits per 32 weights maps
  to the two legal E4M3/16 scale bytes;
- the nested meaning of each partial scale prefix;
- exponent-delta range, exceptional/forbidden codes, and global-scale
  interaction;
- the block-level event key and byte layout; or
- how the 2-3-bit terminal clip exponent composes with those scale values.

Without that mapping, a decoder cannot reconstruct the scale plane and the
non-requantized T-nvfp4-class branch is not yet demonstrably byte-legal NVFP4.
The separate T-nvfp4-RQ branch remains coherent because it openly performs a
new quantization and carries its own quality gate.

**Proposed correction:** add a scale-codec subsection parallel to §6, with an
exact encode/decode table or algorithm, prefix semantics, physical layout, and
exhaustive round-trip tests over all legal base/refinement combinations. Add it
to build item 1 and the exact-byte authority. Until it exists, describe 2b and
the direct T-nvfp4-class terminal as conjectural; do not call them legal by
construction.

### P1-4 — A two-sided rotation is not a per-unit serving branch without an output-basis contract

The branch DAG contains `{none, R_in-only, two-sided}` (`:227-236`), but the
serving contract mentions only the input operation `R^T x` (`:351-365`). For a
two-sided encoded weight

`W_q approximately R_out * W * R_in^T`,

feeding `R_in * x` produces output in the `R_out` basis. Correct model semantics
require `R_out^T` after the operator or a proved propagation of that basis into
every consumer. Per-unit selection makes that especially nonlocal around
residual additions, norms, fused siblings, and routed experts. Coupling sibling
branch choices is not itself an inverse-transform contract.

**Proposed correction:** either remove `two-sided` from serveable branch
identity and keep it as a weight-space measurement state, or specify the exact
output inverse/propagation location, fusion mechanism, storage/runtime cost,
group constraints, and eager/graph/MTP correctness gates. `R_in`-only is the
only rotation state that is algebraically local under the contract currently
written.

### P1-5 — Segment 3b is claimed to remain, but it is absent from the normative grammar

The segment table has 0, C, 2a, 2b, and B. It then says a second coded residual
pass “remains in the grammar” (`:137-151`), while §6 defines no residual wire,
§9 declares no residual terminal, and the branch/artifact sections define no
dependency or bytes for it. Section 13 and arm 4 nevertheless retain
dual-accumulate claims (`:421-427`, `:462-463`).

**Proposed correction:** given the measured kernel negative, remove 3b and
dual-accumulate from the normative EN grammar and keep them as a separately
gated future proposal. If GLM retains 3b, add its base dependency, code/scale
grammar, terminal, event ordering, exact bytes, activation transform semantics,
and two-pass serving contract to this document. “Remains” is not sufficient in
a folded single-authority specification.

### P2-6 — The fused result is not on the branch named by `[branch]`, and three matrix/ledger rows drifted

The provenance block defines `[branch]` as auditable on
`origin/claude/prismabuild-trellis-integration-20260831` (`:9-16`). The cited
`docs/results/trellis_fused_decode_pipelining_2026-08-31.md` is not present in
this checkout, that remote branch, the WO-F branch, or any currently visible git
ref. The in-tree handover summarizes the result, but does not supply the named
driver, raw rows, profiler trace, or Netdata series. Under the document's own
evidence policy and repository principle 13, the fused timing must be
`review-reported` until those artifacts land. “Reproduction owed” does not make
an unavailable artifact measured.

Additional mechanical corrections:

- `E4M4/16` at `:444` should be `E4M3/16`.
- Arm 11 says “both rotation states” at `:472`, while the branch DAG has three;
  enumerate the intended states.
- The ledger's “traffic 3.75 < 4.25 < 4.5” at `:508` has no named terminal or
  exact-byte derivation in the folded text and conflicts with §6's explicit
  statement that terminal/traffic tables are still owed. Remove it for now.
- Expand the `[branch]` definition to permit an explicitly named branch, or use
  a separate tag for WO-F; the current definition names only the integration
  branch while the BlockLDLQ row cites `muse/wo-f-trellis-ldlq-20260831`.

## Measurement boundary

The §14 permitted boundary is otherwise correct. Arm 1, the scale-penalty
sub-arm, and existing-encoder weight-space replications remain permitted with
the stated scope and artifacts.

One addition is warranted: **independent reproduction of the decomposed fused
decode benchmark is permitted now** because it exercises the existing released
decoder and native MMA without implementing EN or a new kernel. It must use the
known-good container/pin, exact total wire bytes, an in-process profile plus
Netdata/power evidence, interleaved repetitions, and committed raw logs. This
permission does not include fused-kernel implementation.

Grammar/codec implementation, schema/DP/menu wiring, and runtime work remain
gated on P0-1 and the relevant P1 corrections.

## Exit gate for the next revision

The next text is ready for implementation review when it contains:

1. a byte-level, uniquely parseable typed-event stream with charged quotas and
   offsets;
2. an exact scale-mantissa codec;
3. a bpp-dependent kernel feasibility function using exact selected bytes and
   an honest measured parity delta;
4. a local correctness contract or removal for two-sided rotation;
5. either a complete residual-wire grammar or no normative 3b claims; and
6. corrected evidence tags and measurement-matrix rows.

No edits to the normative proposal were made by this review.
