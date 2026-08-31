# Trellis E4M3 scale-grid gate

Status: **research-only, unregistered, and ineligible for production or menu
promotion**. The implementation is numerical scaffolding for measuring whether
the existing fixed-size E4M3 group-16 scale plane can improve the native E2M1
trellis reconstruction. It changes no Gridbook wire ABI, runtime pin, format
registry entry, cache path, or serving claim.

## Why the old selector was retired

The former `select_e2m1_scale_grid` minimized an independent 15-level RTN
surrogate per group of 16. That is exact only for pure bypass positions. At a
shaped rate, the reachable point is determined by the 256-state path and the
tail-biting closure over the whole 256-column superblock. Changing one group
scale can therefore change the path and reconstruction in another group. A
per-group RTN win is not evidence that the realized trellis wire wins.

The old entry point now refuses. `propose_e2m1_scale_plane` retains the RTN
search only as a proposal generator; none of its scores can authorize an
output.

## Diagonal two-arm construction

`encode_e2m1_scale_grid_two_arm` takes the complete explicit encoder recipe.
It performs this closed protocol:

1. Run the unchanged identity encode and canonically pack, reparse, and decode
   its wire.
2. Freeze that identity arm's tensor-global scale. Candidate zero is exactly
   multiplier `1.0`, and its snapped `uint8` scale codes must byte-equal the
   identity plane.
3. Generate a legal candidate plane. Every selected scale code must decode to
   a finite, strictly positive E4M3 value. Illegal non-identity proposals are
   masked; an illegal identity cell refuses.
4. Run a second complete encode with the candidate plane and the same frozen
   global. No Viterbi metrics, survivor state, tail-bite start, or traceback
   state is shared between arms. Canonically pack, reparse, and decode it.
5. Score both realized byte-derived reconstructions in fp64 using the same
   diagonal objective as the encoder (including its `1e-12` importance floor).
   A candidate wins only on strict `<`; ties select identity.
6. Splice every value-bearing plane only at `(output row, 256-column
   superblock)`: path bits, point indices, bypass codes, all 16 group-scale
   bytes, and reconstruction columns.
7. Pack, reparse, and decode the splice, then require bit-exact per-tile
   `Cf == min(C0, C1)`, `Cf <= C0`, and identical wire length. If no tile wins,
   the final bytes must equal the identity bytes exactly.

`scale_plane_override` is the narrow encoder seam. `None` preserves the old
encoder path byte-for-byte. An override must already be a same-device `uint8`
tensor of the exact group-plane shape, and every code must be legal. The
tensor-global scale is not reoptimized.

The scale field remains one byte per 16 weights, exactly `128 q256` or 0.5 bpw.
The body rate remains the declared `body_rate_q256`. Identity, candidate, and
final wires must have the same byte length, so `wire_byte_delta = 0` and
`delta_bpw_q256 = 0`; no padding or side channel is credited as capacity.

## BlockLDL / Arm E scope

Arm E cannot use the diagonal superblock splice inside feedback. An earlier
block's target depends on every later decoded block in the same LDL factor
group, so mixing independently chosen block trajectories would not be a valid
recurrence.

The opt-in Arm E arguments are `scale_grid_multipliers=(1.0, ...)` and
`scale_grid_selection_scope="row_factor_group"`. The producer runs the entire
reverse feedback recurrence twice for every factor group:

- identity scales at every terminal; and
- an in-loop candidate proposal and full candidate encode at every terminal,
  using that trajectory's own feedback-adjusted target and the same
  precommitted global.

It scores the two complete byte-decoded trajectories in fp64 with the factor
group's transformed Hessian and chooses strictly per `(row, factor_group)`.
The same row mask is applied to every 256-column block in that factor group.
The selected trajectory is checked again against the unbuffered recurrence,
the LDL factor reconstruction, and the direct-versus-decomposed quadratic
proxy. Dense-H runs therefore gate per `(row, whole K)`; structured diagonal-H
runs may gate per transform-derived factor group only because cross-group
feedback is exactly zero by construction. Requests for `row_superblock`,
`row_group16`, or any other finer scope refuse before encoding.

The finalized GLM BF16 corpus v2 has hash-bound importance vectors. Those
vectors remain part of the campaign input identity and supply the structured
diagonal-H objective; the implementation does not rely on the earlier stale
assumption that GLM importance was absent. Enabling the new scale grid in a
campaign still requires an explicit recipe/manifest change so static and
grid-gated measurements cannot share a curve identity.

## Evidence and non-claims

Receipts bind the ordered multiplier menu, the complete canonical render
recipe, import-time selector and encoder source identities, immutable global,
both arms, winner scope, canonical wire hashes, fp64 objective bit patterns,
exact-byte accounting, and the non-regression assertions. Both loaded source
identities are also checked against the live files around execution, so a
cached implementation cannot be labelled with bytes written after its import.
Every recipe scalar must be a plain JSON-compatible type; schedule and
alphabet containers are copied once into an immutable validated snapshot used
by both execution and receipt construction, and each parsed wire is compared
back to that snapshot. Stateful containers and string subclasses therefore
cannot make the receipt name a recipe different from the executed one.
The mapping-only `validate_scale_grid_receipt` checks the closed schema and
self-consistency; it cannot authenticate an opaque hash without the artifact
it names and is not a replay authority. Authoritative replay uses
`require_scale_grid_selection_replay` for the diagonal gate or
`require_blockldl_trellis_artifact_replay` for Arm E. Those paths revalidate
caller-owned tensors and the full explicit recipe, rerun both arms or complete
feedback trajectories, and require every canonical byte, decoded tensor,
score, proposal, winner, transform field, and receipt semantic to match.
Well-formed receipt forgeries with recomputed self-digests, identity-code
drift, illegal scale bytes, source drift after import, global drift, a wrong
splice, scorer disagreement, and a finer Arm E scope all refuse in focused
tests.

This construction proves only that the selected research wire is no worse
than its realized identity arm under the bound objective. It does not prove a
quality lift on a model, a KL/PPL result, a performant GPU implementation, a
Gridbook runtime ABI, or serving parity. Any promotion still requires matched
corpus measurements, before/after profiling, Netdata and power/work-per-joule
evidence on Sparky and Sparklina, and the ordinary Gridbook ship gates.
