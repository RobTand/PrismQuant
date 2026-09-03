# Continuous Trellis rate surface (research only)

> **ARCHIVED 2026-09-02 (RobTand/prismaquant#118).** The modules this document
> describes — the five `prismaquant/trellis_*.py`,
> `serving_profile_specs/trellis_research_sm121.json` and the four
> `tests/test_trellis_*.py` — now live at `archive/trellis_wire_2026-09-02/`,
> whose README is their obituary. They priced `gridbook.trellis.wire.v1`, and
> Robert retired the Gridbook lane on 2026-09-02, so no sanctioned runtime
> reads those bytes. Tessera's wire is `prismaquant.tessera.v1`, a different
> plane set and deliberately not a port. `PRISMAQUANT_TRELLIS_SURFACE` is still
> refused, loudly, by `allocator_candidates.refuse_retired_trellis_surface`.
> Everything below is history. See `docs/ARCHITECTURE.md` §4.9.

## Status and boundary

This is an offline allocator-research surface. It creates no `FormatSpec`, no
producer format, no exporter branch, no pipeline flag, and no compiled-kernel
ladder. A legal q256 value is a parameter of one Trellis family implementation,
not a public format ID. Every record carries `producer_eligible: false`.

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
