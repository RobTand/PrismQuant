# Continuous Trellis rate surface (research only)

## Status and boundary

This is an offline allocator-research surface. It creates no `FormatSpec`, no
producer format, no exporter branch, no pipeline flag, and no compiled-kernel
ladder. A legal q256 value is a parameter of one Trellis family implementation,
not a public format ID. Every record carries `producer_eligible: false`.

## Interpolated measured curves: sealed v2 contract

`prismaquant.trellis_rate_surface` may densify sparse measurements for exactly
two purposes:

1. campaign planning, to choose the next real rung to encode and measure; and
2. retrospective allocator-cost validation, only after a measured holdout and
   full-truth allocation-regret gate have validated the exact candidate menu.

It is mechanically ineligible for a menu verdict, family verdict, publication
claim, producer format, export, or serving decision. Those spellings are not
aliases for a permissive analysis mode: the closed use vocabulary rejects them.
The compatibility fields named `licensed_use` / `licensed_uses` encode only
that closed vocabulary; they are not statistical, population, or
consumer-allocator licenses.
The legacy `prismaquant.trellis_surface_manifest.v1` path cannot enter either
the direct research menu or the production seam. V1 has useful read-only
campaign provenance, but it lacks the identity and decision records below.
Those facts may not be inferred. An identity-complete manifest-v2 loader does
not exist yet and v2 input is explicitly refused rather than downgraded to v1.

### Frozen curve identity

Every measured anchor, matched scalar control, fitted surface, holdout, and
regret record binds the same canonical curve identity:

```
{
  wire_family,
  selector,
  alphabet_policy,
  alphabet_fitting_scope,
  scale_plane,
  scale_coding,
  encode_tier,
  schedule_policy,
  render_path,              # rtn | gptq | ldlq | rotated
  corpus_id,
  importance_id,
  population,               # dense | routed
  render_recipe_identity_sha256,
  codec_closure_sha256
}
```

`render_recipe_identity_sha256` is the canonical digest of the complete
value-level render recipe, including every applicable knob such as LDL damp,
terminal mode and block geometry, rotation construction and seeds, scale-grid
mode/menu, tail-biting, and superblock chunking. `render_path` remains the
broad categorical label; the codec closure says what code was available, and
neither substitutes for the exact recipe digest.

The fitted surface additionally binds the allocator unit, exact tensor shape,
layout, target serving profile, objective currency, fit response, and the
identities of every measured anchor. The currency is already sealed into each
measured trellis anchor and scalar control; fitting cannot relabel the same
numbers as a different objective. Shape is part of the surface context because
it changes both the per-column schedule and the exact serialized-byte
function. Changing any
identity field, mixing wire families or layouts, changing shape or schedule
policy, or supplying a scalar control from another identity is a refusal, not
a new implicit curve. Every curve requires its own identity-bound measurements;
QTIP selectors and LDLQ or rotated render paths likewise cannot borrow a
scalar, RTN, or zero-gain anchor from another curve. One
`allocation_regret` call additionally requires every unit to share the same
curve identity (unit name and tensor shape remain separate surface fields), so
its result cannot be read as evidence pooled across corpora, importance
definitions, populations, or render recipes.

This research identity does not contain a model-checkpoint digest or a hash of
the source tensor values. `unit_name` is therefore meaningful only inside the
same trusted campaign/model context; it is not a cross-model content address.
Persisting or transferring a surface beyond that context requires a future
sealed model and unit-universe identity. The refusing manifest/production seams
mean this omission cannot currently authorize a shipped assignment.

### Fit response

The allocator-capable response is gain over a measured, context-matched scalar
subgrid. For anchor `i`, with trellis loss `D_i` and scalar-control loss `S_i`,
the fitted value is

```
G_i = 10 log10(S_i / D_i).
```

`G` is interpolated linearly in integer q256 between the two measured anchors,
then converted back with the independently measured scalar point at the target
rung:

```
D_hat(q) = S(q) * 10 ** (-G_hat(q) / 10).
```

Every scalar point carries its own closure digest and an explicit
`context_parity_verified` bit under the same curve identity. Missing controls,
false parity, or a different target rate fail closed. Supplying scalar controls
to the raw path is also refused; a caller cannot silently throw away the
better-matched response.

Piecewise-linear interpolation in `(q256, log2 raw_dloss)` remains available
only as an explicit campaign-planning fallback. It cannot provide allocator
costs. Both responses require at least two identity-bound measured anchors,
strictly increasing integer q256 and strictly decreasing positive loss.
Interpolation never extrapolates and a prediction that escapes the loss values
of its bracketing anchors is rejected. Nonmonotone measurements are re-anchor
work, not values to smooth.

### Sealed holdout and retrospective regret validation

Retrospective allocator validation is a two-phase protocol:

1. Fit without at least one strictly interior rung and seal its prediction,
   surface identity, curve identity, target q256, and matched scalar-point
   identity before the rung's trellis loss is supplied. The seal also commits
   the exact pre-render recipe digest; grading a different schedule/alphabet
   recipe at the same nominal q256 is refused.
2. Measure that exact rung under the same identity and scalar control, then
   grade the pre-existing seal. A fitted anchor cannot double as a holdout.
3. Densify the allocator menu through the real schedule builder and wire
   pricer. Every truth value must be an identity-bound measured anchor whose
   pre-render recipe digest exactly matches that densified rung; unbound float
   claims are refused. Grade allocation twice at one sealed byte budget: once
   deciding on interpolated losses and once deciding on measured truth,
   scoring both assignments with truth.
4. Require explicit scale-price bracket agreement and a configured maximum
   allocation-regret percentage. Every surface needs a measured holdout grade.
5. Pass the resulting immutable gate to the research menu materializer. The
   gate binds both
   each fitted-surface digest, each full densified-menu digest, and the exact
   measured-truth anchor set, so it cannot authorize different rungs, recipes,
   predicted losses, shapes, byte prices, or validation truth. A failed gate,
   planning-only surface, missing grade, changed menu, different unit set, or
   solver budget other than the one graded is refused before candidates are
   returned. Materializing those candidates does not transfer the measured
   regret result to the repository's `allocator_solver` or to any other
   consuming algorithm.

Leave-one-anchor-out residuals remain diagnostics only. The acceptance number
is decision regret within the internal greedy comparison: the measured-truth
cost of using interpolation in that algorithm. Bracket disagreement cannot be
hidden by averaging or by an apparently small regret on a different menu.

The seal and grade are canonical self-hashed in-memory evidence. They check
that the supplied grade is arithmetically consistent with the bound surface,
prediction, measured anchor, scalar control, and wire recipe. They are not a
trusted timestamp, signature, append-only log, or proof that the caller did not
seal several rungs and report only the favorable one. A campaign that needs a
chronology or anti-shopping claim must externally precommit its holdout set,
budget, regret threshold, and bracket procedure before measurement. No such
population/campaign attestation is implemented here.

The other canonical SHA-256 fields have the same boundary: they detect an
accidental mutation only when checked against the objects they bind. They are
not signatures or authentication, and a trusted Python caller can construct a
new object and recompute a new self-consistent hash (or call a lower-level menu
helper directly). The high-level API now replays the derivable arithmetic and
surface/menu bindings before materialization, but it is not a hostile-caller
security boundary. Both production manifest paths still refuse.

### Operational scope: zero allocator-use encodes saved

The full-truth gate is intentionally strict and operationally vacuous as a GPU
measurement reducer. For every unit, `allocation_regret` requires the measured
truth rates, scalar-backbone rates, and exact densified candidate rates to be
the same nonempty set. Each truth row must be the identity- and recipe-bound
measurement of that exact candidate. The solver-menu materializer then accepts
only the densified-menu digest covered by that gate. Therefore every candidate
it can return has already been trellis-encoded and measured: allocator-use
savings are zero encodes, with positive validation overhead.

The resulting regret number is still useful as a retrospective validation of
the internal greedy marginal allocator. That allocator is not the consuming
`allocator_solver`, and the gate neither certifies a different algorithm nor an
exact global optimum on non-convex per-unit curves. Campaign planning is the
only current path that can reduce measurements: it may interpolate between
sparse anchors to choose the next rung to measure, but cannot enter the
allocator bridge.

This module has no sampled-cohort or population-transfer mechanism. On the
current full-truth path, a non-monotone measured curve is refused and sent for
re-anchoring; no unit is omitted from a statistic. If a sampled population path
is ever added, a sampled non-monotone curve must count as a violation rather
than being dropped by reusing this refusal behavior. Current evidence does not
authorize such a path.

### q256 and exact-byte pricing

Interpolation happens only on exact integer `body_rate_q256`. Every proposed
rung is rendered into a legal schedule and priced with
`trellis_tensor_payload_breakdown`; bytes are never interpolated or estimated
from nominal bpp. Alignment makes the map from q256 to bytes non-injective. The
regret allocator therefore collapses an equal-byte plateau to its lowest-loss
rung before computing marginal gains, preserving a free quality improvement
that a naive positive-byte-delta loop would skip.

The sparse `quality_candidate_q256` values in `trellis_formats.py` remain the
reviewed seed measurements. They are not the mathematical rate domain. An
experiment must explicitly request either `all_legal`, a deterministic `dense`
grid, or an `adaptive` surface derived from a measured per-tensor frontier.
Nothing expands the sparse seed implicitly.

## Address and exact bytes

One research candidate is addressed by:

```
(family, body_rate_q256, layout, pre_render_recipe_identity_sha256)
```

The ephemeral solver spelling includes the complete digest as
`...:layout=<layout>:recipe=<64-hex-sha256>`. A descriptive `variant_label` is
receipt metadata, not an address disambiguator. Two measurements of the same
pre-render recipe for one allocator unit are rejected and must be
aggregated upstream; two schedules with the same rate and byte count remain
distinct because their full recipe digests differ.

`body_rate_q256` is an integer body-bit quota per 256 weights. Both
`fixed_quota_per_256` and `tight_offsets` retain their existing validation.
`trellis_tensor_payload_breakdown` remains the byte authority and charges the
body, row padding, family scale plane, schedule, offset table, alphabet blob,
mandatory wire header, and any explicit sidecar header. The solver sees
`total_bytes` and `8 * total_bytes / n_params`, never nominal body bpw.
`structural_side_information_bytes` is the non-scale wire subtotal;
`wire_side_information_bytes` matches Gridbook `account().side_bytes` by adding
the scale plane; `side_information_bytes` adds any explicit exporter sidecar.
Thus `total_bytes = body_bytes + side_information_bytes` without changing the
previous exact total.
Rows, columns, schedule count, alphabet blob size, and scale-plane size obey
the actual Gridbook v1 header widths; every uint32 field must be representable
before a footprint can become a candidate.

The pre-render recipe identity binds the canonical schedule and native-code
alphabet values, layout, shape, scale contract, and all byte counts. It does
not bind encoded body indices, scale-plane contents, or E2M1
`global_scale_real`; candidate construction does not possess those rendered
values. It is therefore never described as a physical wire identity, and
`rendered_wire_identity_sha256` remains `null` until a future renderer hashes
actual wire bytes. Trellis side planes are tensor-local, not a separately
deduplicated physical codebook, so the solver bridge leaves
`serialized_sidecar_identity` unset.

## Lazy deterministic surfaces

`trellis_rate_surface(..., mode="all_legal")` represents the complete inclusive
integer range using only its bounds. `mode="dense"` stores bounds, a positive
q256 step, and explicit anchors; the exact stop is always included. The
re-iterable `rates_q256` view generates values in sorted order without storing a
rate list. One family surface can be shared by every tensor; callers must not
construct a `units × all-rates` table merely because the range is iterable.
Positive and negative slices return explicit tuples, while the stored view
remains lazy and O(1)-state.
Constructor-supplied bounds, anchors, and proposals are copied into tuples
before validation, so caller mutation cannot change a surface identity.

The intended fast path is tensor-dependent:

1. Start each tensor from a small deterministic dense grid or reviewed anchors.
2. Measure only those wires and retain exact bytes, mean predicted loss, its
   standard error, and the target-profile decision.
3. Build the tensor's objective Pareto frontier and compressed lower-convex RD
   hull.
4. Use a global marginal price to choose one point per tensor, then bisect the
   byte monotone global result.
5. Refine only lower-hull q256 intervals whose exact rational
   marginal-loss-per-byte threshold is closest to the requested binary64 alpha
   (uncertainty intervals bracket alpha first).
6. Hand an explicit Pareto neighborhood to a later integer-repair method.
   `complete_pareto=True` widens the menu to the complete measured Pareto
   surface; it does not certify the resulting allocation.

No stage invents a tensor-independent family cutoff. E2M1 Trellis, E4M3
Trellis, native formats, and CB formats can be placed in the same ordinary
`Candidate` menu by a later ablation. The Trellis bridge does not register or
promote any of them.

Adaptive refinement first subdivides a hull interval at every already
observed interior q256, including Pareto points that lie above the lower
convex hull and dominated measurements. It then bisects the widest eligible
unobserved subinterval with deterministic endpoint and identity tie-breaks.
If `max_new_points` permits, selected subintervals are subdivided again in the
same call. Thus a previously measured midpoint cannot stall refinement: each
selection adds a new legal q256, and refinement stops only at the explicit cap
or when no legal interior remains. One adaptive curve must have one tensor
shape, target profile, Trellis family, layout, and a q256-increasing exact-byte
hull. General exact-byte menus and
RD hulls still permit different Trellis families/layouts to compete; q256
adaptive interpolation does not mix those distinct curve domains.

Candidate footprints and adaptive bracket metadata are recursively immutable.
Serialization creates recursive mutable copies, so mutating a report payload
cannot alter a candidate, proposal, or content identity.
Adaptive rows carry only an inclusive pair of indices into the surface's one
`anchor_q256` tuple for their parent hull interval; they never copy the full
parent interior list into each subinterval.

Candidate construction re-copies and validates the complete canonical
footprint schema, recomputes the pre-render recipe digest with only the digest
field excluded, and independently checks all byte-plane arithmetic. A stale
or self-resigned but arithmetically inconsistent footprint is refused.
Every sequence nested in a resolved serving lane is likewise copied to an
immutable tuple before it can affect candidate identity.

## Marginal price, uncertainty, and uncertified repair menus

`build_trellis_allocator_candidate` applies the existing target profile's
format and shape gates, plus the Trellis family's declared capability floor.
The `research` profile admits emulation but has no export lane. Shipping
profiles currently reject the ephemeral Trellis wire name. A legal research
decision is recorded separately from producer eligibility and never changes
the latter.

The solver objective is the supplied measured or already-shrunk point estimate:

```
objective_loss = predicted_dloss
```

The bridge never adds a second configurable UCB penalty: doing so would
double-count uncertainty when the upstream estimate is already shrunk or
hedged. Standard error remains a separate reported field.

`trellis_pareto_frontier` reports exact-byte marginal point-estimate loss
reduction per byte, with an uncertainty interval. `trellis_rate_distortion_hull`
removes Pareto points that no scalar marginal price supports. For a hull with
`H` points, every binary64 loss is interpreted as its exact dyadic rational.
Hull orientation and breakpoint ordering use reduced integer ratios and
integer cross-products. The immutable hull stores those exact rational
breakpoints once. `choice_at_lambda` converts the supplied finite binary64
lambda to its exact ratio and bisects against those cached thresholds, with a
deterministic cheaper-on-exact-equality rule. A lookup never rebuilds an `O(H)`
tuple. Float slope values are explicitly rounded diagnostics and make no hull
or lambda choice; an exact positive subnormal-per-byte threshold may therefore
have a diagnostic value of zero without changing the lambda-zero argmin.

Any non-finite derived uncertainty sum, product, radius, or bound is refused.
Confidence-overlap widening is evaluated around the selected point using
`loss_delta + lambda * byte_delta`, not the overflow-prone absolute
`loss + lambda * bytes`; every `lambda * byte_delta`, `z * stderr`, radius sum,
and centered score sum is checked before use.

This module is a candidate generator, not a replacement for
`solve_allocation`: discrete non-convex pockets can contain a Pareto point no
lambda selects. The frontier retains those points, and
`trellis_local_repair_solver_menu` returns an explicit local window—or the
complete measured Pareto menu. It does **not** return a feasible allocation or
an optimality certificate. The default `allocator_solver` may quantize its
internal budget state; exact candidate byte fields do not make that state an
exact-byte proof. A later consumer must filter any assignment against the exact
integer sum of selected `memory_bytes` and attach a method-appropriate
feasibility/optimality certificate. No such solver is integrated here. The
hull separately names its cheapest profile-legal measured floor.

Alpha is derived only from measured loss reduction divided by exact serialized
byte increase. There is no Trellis family bonus, equivalent-bit exchange rate,
or banked credit in `Candidate.bits_per_param`, `memory_bytes`, the lambda
breakpoints, or the DP budget. A scalar-equivalent-rate view may be computed by
an analysis notebook as a diagnostic, but it is not an allocator input and
cannot fund another tensor. Capability, route backedness, latency evidence,
family inclusion, and rate-count/refinement caps are likewise independent
gates or experiment controls; none is folded into alpha or storage bytes.

## Complexity contract

Let `A` be explicit anchors, `P` the measured points for one tensor, `H` its
compressed hull size, `K` the adaptive point cap, and `W` the local
repair-window width.

- A full or dense surface stores `O(A)` state, not `O(number of legal q256)`;
  iteration is `O(number yielded + A)` and is performed only for a tensor an
  experiment chooses to measure.
- Per-tensor Pareto construction is `O(P log P)`, convex-hull compression is
  `O(P)`, and the stored RD hull is `O(H)`.
- For `U` tensors, one deterministic global marginal-price evaluation is
  `O(U log U + sum_tensors(log H_tensor))`: the first term is sorting mapping
  keys for stable output, while each hull lookup uses its precomputed
  breakpoints and is `O(log H_tensor)`. It stores one choice per tensor.
- Adaptive proposal construction sorts the `P` observed q256 values into the
  `H - 1` hull intervals and maintains only unresolved subintervals. With
  `K = max_new_points`, time is
  `O((P + H + K) log(P + H + K))`, state is `O(P + H + K)`, and at most `K`
  unobserved rates are emitted.
- Local repair-menu construction materializes `O(W + C)` candidates per tensor,
  where `C` is the explicit confidence-overlap set. The
  complete measured-surface menu is explicit and bounded by `P`; neither path
  materializes every mathematical q256 for every tensor. These bounds say
  nothing about a later solver's certification complexity.

These are deterministic structural bounds, not performance claims. GPU
measurement/rendering remains a separate future research implementation; this
contract and its tests are CPU-only and do not add a production hot path.

### CPU microbenchmark snapshot

On 2026-08-25, Python 3.12.3 on one aarch64 Cortex-X925, median of seven
repeats with setup outside the timed region:

- Constructing the lazy 1,785-rate E4M3 all-legal surface took 3.894 us;
  iterating it into a tuple took 94.732 us.
- Cached exact-rational lambda lookup took 1.247 us for `H=3` and 1.376 us
  for `H=33`, with maximum observed comparison counts 2 and 6.
- The 271/276/279-byte equal-rounded-slope regression lookup took 1.309 us and
  selected the exact 276-byte argmin.
- One deterministic sorted lookup over `U=128` three-point hulls took
  165.449 us.
- Building a real `P=33, H=33` exact hull took 5.517 ms. Adaptive proposal
  construction took 8.580 ms for `P=33, H=33, K=8` and 10.894 ms for
  `P=129, H=2, K=0`.
- The `P=129, H=2, K=0` proposal contained 128 bracket rows, 256 parent-index
  references, no copied interior lists, and 154,271 canonical compact-JSON
  bytes. Exact rational metadata makes this larger than a rounded-only report
  while preserving the linear structural bound.
- Building and validating one ordinary candidate took 98.829 us; revalidating
  an already-built candidate at the dataclass boundary took 35.617 us.

These are local diagnostic timings, not an SLA or serving measurement. Exact
integer cross-products intentionally cost more than the rejected rounded-slope
lookup, and no GPU or production path was exercised.
