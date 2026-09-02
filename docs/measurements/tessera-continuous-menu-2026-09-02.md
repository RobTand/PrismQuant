# Tessera's continuous rate axis on the production allocator — 2026-09-02

**Branch** `tessera/continuous-menu`. The allocations ran at
`cb64eba`; `prismaquant/tessera_campaign.py` is **byte-identical from
`4d2d9b2` through `cb64eba`** (`git diff 4d2d9b2 HEAD --stat --
prismaquant/tessera_campaign.py` is empty), so the campaign's own code is the
same object across every commit on this branch and the restart mid-branch
changed nothing about how an anchor is priced. **Tessera** — every campaign and allocation below ran at
`3d419e7b4d6ffd0259fcb7c36ea179bddc1d7ce9`; the runtime contract, the shard
granularity and the `ActivationSource` seam (§10, §11) are read at
`f3e7d0ae78e64fcc1a13d5b9553a95fe4006bef4`, which is what the dev pin names.
`src/tessera/serving/` is never imported — another worker owns it; its packaged
`runtime_contract.json` is read as a data file by path arithmetic. ·
**Boxes** sparky (host tests) and sparklina (probe, campaign, allocations) ·
**Python** `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`.

**No artifact was served and no KL was measured in this work.** There is no
Tessera export leg in this tree. Everything below is an *allocator* result:
what the DP may consider, what it costs, and what it picks. Nothing here is a
quality claim about a served model, and §9's export gate has not been
exercised.

## 1. What was built

`FORMATS` accepts the token `TESSERA`. `tessera_menu.expand_menu_tokens`
replaces it with the Tessera columns the run's own cost table holds **that the
pinned runtime attests**, so the menu is the widest one the DP could honestly
consider, never wider, and never a set the producer-eligibility guard would
then refuse (§10).
Per-Linear the menu is the whole realisable rate set of every family the
hardware bases admit, subject to three gates and nothing else — shape legality,
route legality read from the attested contract table, and W(n≤A) by
construction. Architecture: `docs/ARCHITECTURE.md` §4.10.

The single seam for the runtime's own table is `tessera_menu.route_admission`.
The single seam for tensor-parallel legality is
`tessera_menu.tessera_shard_granularity`, which asks
`tessera.layout.shard_granularity` (Tessera `f3e7d0a`). Both are one function
each, by design, so the incoming work swapped in without touching the menu.

## 2. Commands

Probe and campaign (sparklina):

```
rsync -a --delete --exclude .git --exclude __pycache__ \
  /home/rob/pq-wt/tessera-continuous/ sparklina:/home/rob/tmp/wt-pq-continuous/
rsync -a --delete --exclude __pycache__ \
  /home/rob/tessera/src/ sparklina:/home/rob/tmp/tessera-3d419e7/src/

PYTHONPATH=/home/rob/tmp/tessera-3d419e7/src:/home/rob/tmp/wt-pq-continuous \
PRISMAQUANT_TESSERA_MENU=research \
python -m prismaquant.tessera_campaign \
  --model /home/rob/models/Qwen3-0.6B \
  --out   $ROOT/cost.pkl --cache-dir $ROOT/cache \
  --nsamples 4 --seqlen 512 --max-act-rows 256 \
  --layer-stride 28 --anchors 3 --max-rounds 3 --loo-gate 0.25 \
  --max-artifact-bpp 5.5 --deadline-seconds 9000 --hessian off
```

> **`--hessian off`, and what that means for every number below.** The Tessera
> encoder's shipping default is **H-aware**: given a unit's `H = XᵀX` it applies
> LDLQ (sigma 1.0, block 32) plus an exact full-H row-scale refit. That branch
> is **not merged into the pinned Tessera `3d419e7`** — no encoder entry point
> there takes an `H` — so no anchor in this receipt was priced with one, and
> `--hessian require` (the default) refuses on this build rather than pretend.
> Weights-only encodes are reported by the encoder's author to be byte-identical
> under the new default, so these bytes are the bytes a weights-only encode
> still produces; they are **not** the bytes that ship. Their author's served
> figure for the change — Qwen3-0.6B KL 0.1512 → 0.1046 at byte-identical wire
> bpp — is **their measurement, cited, not reproduced here**. Read every rate
> surface, every LOO number and every allocation below as a *weights-only*
> price. §7 carries this as an open item.

Allocations — three targets × five arms
(`/home/rob/tmp/pq-continuous/run_allocations.sh` for `full`/`meas`/`nofuse`/
`free`, `/home/rob/tmp/pq-continuous/run_meas_nofuse.sh` for `measnofuse`):

```
python -m prismaquant.allocator \
  --probe $ROOT/probe.pkl --costs $ROOT/cost.pkl \
  --formats TESSERA --target-bits {3.0,4.0,5.0} \
  --target-profile tessera_research_sm121 \
  --layer-config $ROOT/alloc/lc_<arm>_<T>.json \
  --pareto-csv   $ROOT/alloc/pareto_<arm>_<T>.csv
```

* `full` — the interpolated rate surface, default (pre-aggregated) DP.
* `meas` — measured anchors only, built by
  `/home/rob/tmp/pq-continuous/measured_only_cost.py`; the sparse-menu baseline.
* `measnofuse` — measured anchors only **and** `--no-fused-aggregation`; the
  only baseline that isolates menu density, because `meas` vs `full` also
  differs in what fused-sibling intersection leaves on the menu (§8.3).
* `nofuse` — interpolated surface with `--no-fused-aggregation`. **Retired as
  an arm** (§4): it drops the family constraint as well as the rate one, so
  its assignments are not legal serving configurations. Kept below as a
  *bound* -- the best a group could do if nothing coupled its members -- and
  as the matched partner of `measnofuse` in the density comparison.
* `mink` — the group knapsack: one family per fused group, a rate per member.
  This is the arm that measures the real constraint (§4).

## 3. Tests

```
cd /home/rob/pq-wt/tessera-continuous
PYTHONPATH=/home/rob/tessera/src:. python -m pytest \
  tests/test_tessera_menu.py tests/test_tessera_campaign.py \
  tests/test_tessera_formats.py tests/test_docs_staleness.py \
  tests/test_architecture_doc.py -q
```

**138 passed, 0 skipped** on sparky (read from that command's own output).
Two earlier revisions of this file are superseded: 114, which was summed from
two older runs and counted a skip as a pass, and 118 passed / 1 skipped, which
was true before the Hessian seam landed. There is no skip left --
`test_the_hessian_kwarg_pin_matches_the_pinned_encoder` had nothing to check
while `TESSERA_HESSIAN_KWARG` was `None`, and it became a real assertion when
the seam became `ActivationSource` (§11). The `CUDA` marker here is a `skipif`,
not a deselect, so sparky's GPU means every CUDA campaign test ran. The
load-bearing ones:

* `test_the_same_weight_rate_costs_differently_on_the_two_routes` — the A-leg
  pricing test the acceptance asks for. It scores one rendered rung twice, once
  under the route's own activation quantiser and once under the identity, for
  both routes, and asserts the **ratio** differs between them. Asserting only
  that each route differs from identity would pass on a harness that applied
  one contract to both, which is exactly the bug it is there to catch.
* `test_the_priced_render_is_the_decoded_wire` — the cache stores the wire
  beside the dequantised render, and the priced render is the decode of those
  bytes (principle 8).

The full sweep, including every allocator suite this branch could regress:

```
PYTHONPATH=/home/rob/tessera/src:. python -m pytest \
  tests/test_tessera_menu.py tests/test_tessera_campaign.py \
  tests/test_tessera_formats.py tests/test_tessera_footprint.py \
  tests/test_tessera_shape_dependent_recipe.py \
  tests/test_docs_staleness.py \
  tests/test_architecture_doc.py tests/test_allocator_sibling_aggregation.py \
  tests/test_allocator_promotion_legality.py \
  tests/test_allocator_packed_group_units.py \
  tests/test_allocator_main_enforcement.py \
  tests/test_allocator_pareto_seed_export.py \
  tests/test_allocator_byte_budget_selection.py \
  tests/test_allocator_solver_bins.py \
  tests/test_interpolated_output_mse_pricing.py \
  tests/test_production_weight_cache.py \
  tests/test_col_weights_render_identity.py tests/test_render_score.py \
  tests/test_weight_session.py tests/test_perturbed_x_cache.py \
  tests/test_aura_cost.py tests/test_format_registry.py -q
```

**436 passed** (150 s, sparky), read from that command's own output -- 433
before the three menu-token tests of §10 were added. An earlier
revision of this file said **349 passed, 1 skipped** over 19 suites; that number
is stale twice over -- three suites were added (`test_tessera_footprint`,
`test_tessera_shape_dependent_recipe`, `test_allocator_solver_bins`) and the
one skip became a real assertion when the Hessian seam stopped being a pin and
became `ActivationSource` (§11). The last seven suites were added when
`render_production_weight` gained the Tessera interception and the three
cache-miss fallbacks gained their refusal (§7): a change inside those functions
has to be shown not to move any *other* format's render.

**One pre-existing failure, not in the sweep and not this branch's.**
`tests/test_allocator_serve_constraints.py::test_no_constraints_selection_is_identical_to_feature_present_but_unused`
compares two `selection.json` dicts for equality, and those dicts carry
`solve_diagnostics[*].solver_seconds` -- a wall-clock float that is never equal
across two solves. It fails identically with this branch's changes stashed
(`git stash && pytest ...`), so it is recorded here rather than fixed: the fix
is to compare the selection numbers without the timing field, which is someone
else's file to change.

## 4. The decision-unit constraint, and what the wrong one cost

**Superseded, and the supersession is the result.** This section used to
explain why the family relaxation was unreachable on the default path: the DP
saw one super item per fused group with one candidate per format **name**, so
it could only ever return a single shared rung, and `_promote_group_components`
had nothing to do. That was recorded as a design, not a defect. It was a
defect, and an expensive one.

The one-rung reading is not the serving constraint. What a fused group's
members cannot disagree about is the **decoder** the runtime dispatches on --
for a Tessera rung, the family (grid and arity); the rate is a point on that
family's continuous axis and is not a dispatch property. `q_proj`
(2048x1024) and `k_proj` (1024x1024) are different tensors with different
sensitivities. And `--no-fused-aggregation`, the arm that used to stand in for
the relaxation, is the opposite error: it drops the family constraint too, so
its assignments are **not legal serving configurations** and its numbers are a
bound, not an arm.

**The exact constraint, implemented exactly.** For each fused group and each
family F, the group's option set is the **Minkowski sum** of its members'
`(bytes, cost)` menus restricted to F -- the group's own multi-choice knapsack
-- kept as a Pareto set under **dominance** pruning. Never a hull: the budget
is discrete, so an option strictly inside the hull can still be the unique
optimum at one particular remaining capacity, and hull pruning drops exactly
those. The outer DP then chooses among those per-family group option sets plus
the stock per-NAME options, exactly as it chooses among rungs of a single unit.
`tessera_group_composites` builds it; `expand_fused_sibling_assignment` reads
the per-member rung map back out; `_promote_group_components` was already
family-aware and is a verified no-op on the result.

Exactness is not a claim here, it is enforced:

* the fold is a full cross product at each step, Pareto-pruned by dominance,
  and pinned against **brute force** over every member combination on a menu
  built with a non-convex pocket
  (`test_group_knapsack_equals_brute_force_including_a_nonconvex_pocket`);
* a uniform-rung option exists in both constructions, and the aggregation
  **refuses** if the per-NAME path and the group knapsack disagree about its
  bytes or its cost;
* summing member costs is exact only while the UCB hedge is linear. The group
  hedge is `z*sqrt(sum stderr^2)`, which is not additive, so the fold refuses
  at `z > 0` rather than pricing it wrong. Both flagship runs and these use
  `z = 0` and a table whose 16893 rows all carry `stderr = 0`.

**Group menu sizes and DP cost** (Qwen3-0.6B layer 0, from
`__prismaquant__.tessera_group_knapsack`):

| group | family | member menus | fold frontier | options |
|---|---|---|---|---|
| qkv_proj | `TESSERA_E2M1_K2` | 471 / 80 / 92 | 471 -> 629 -> 720 | 720 |
| qkv_proj | `TESSERA_E4M3_K1` | 593 / 1133 / 1117 | 593 -> 2638 -> 3754 | 3754 |
| gate_up_proj | `TESSERA_E2M1_K2` | 76 / 76 | 76 -> 151 | 151 |
| gate_up_proj | `TESSERA_E4M3_K1` | 1138 / 1138 | 1138 -> 2275 | 2275 |

The fold stays small because each member's menu is already a monotone
dominance-pruned frontier; the guard that refuses a fold over 8M intermediate
pairs is never approached here. The aggregated menu into the DP goes from
**4315 rungs (one-rung aggregation) to 11215 -> 9328 after dominance**, and a
full three-target sweep costs **64-67 s wall** per target end-to-end
(allocator process, `/usr/bin/time`), against ~2.3 s of DP solve time on the
old menu. The asymmetry in the qkv member menus (471 / 80 / 92) is the
per-unit anchor placement described in §7, and §4a is the arm that fixes it.

**What the wrong constraint cost.** All three arms priced by one script
(`price_mink.py`) on the same full interpolated table:

| target | one-rung `full` | group knapsack | `--no-fused-aggregation` (illegal) |
|---|---|---|---|
| 3.0 | 34.1677 @ 3.00026 bpp, 4 rungs | **27.3986 @ 3.00026 bpp, 7 rungs** | 27.1762 @ 3.00729 bpp |
| 4.0 | 11.9033 @ 4.00026 bpp, 4 rungs | **9.19888 @ 4.00026 bpp, 7 rungs** | 9.19860 @ 4.00026 bpp |
| 5.0 | 4.23201 @ 5.00026 bpp, 4 rungs | **3.77304 @ 5.00000 bpp, 6 rungs** | 3.77304 @ 5.00000 bpp |

* **The one-rung constraint cost 1.247x / 1.294x / 1.122x** in Δloss. That is
  the honest replacement for the "pre-DP aggregation costs 11-23%" line, which
  was measured against an illegal relaxation.
* **The family constraint costs 1.008x / 1.000x / 1.000x** against that
  illegal relaxation -- and at 3.0 the comparison is not matched: the
  `--no-fused-aggregation` point is 0.0070 bpp **fatter**, so at matched bpp
  its 0.8% advantage is not resolved by these runs. At 4.0 and 5.0 both land
  on the same bpp and agree to five significant figures. **Requiring one
  family per fused group is, on this table, very nearly free; requiring one
  rate was not.**
* **At 4.0 the "1.000x" is 0.99999795x, and the residual is the DP's bins,
  not the menu.** Both arms land on exactly 62,918,656 bits, and the group
  arm's Δloss (9.198883829) is 0.0031% *worse* than the unconstrained one
  (9.198602389). The menu is not what is missing: the qkv fold **contains the
  unconstrained arm's exact rung triple** (`R692`/`R916`/`R1372`, 1,937,408
  bytes, 2.61999845), and the gate_up fold does not contain its pair only
  because dominance dropped it in favour of `R1150`/`R1190` at the **same**
  3,639,296 bytes and a lower cost. Substituting that dominating pair into the
  unconstrained assignment gives **9.198583530 at the identical byte total** --
  strictly better than either arm. So the constrained menu attains a lower cost
  than the unconstrained arm at matched bytes, and the group arm's DP answer is
  0.0033% off the best point its own menu holds there. That gap is the outer
  DP's charged-bin quantisation at `--bit-precision 1e-4`: a group is one item
  whose aggregate delta is rounded once, where three separate items round three
  deltas whose errors partly cancel. Read the family constraint at 4.0 as
  **free**, and the 0.003% as bin resolution with a named mechanism --
  named by elimination and **not separately measured**: `solve_allocation`
  is exact over bins, so the only way it can miss a point its own menu
  contains at identical true bytes and lower cost is that the point's
  *binned* charge exceeds the budget. Charging `_charged_bins` for the
  three assignments by hand would turn that deduction into a
  measurement; it is not done here.
* The DP picks a mixed-rung qkv at every target -- at 4.0, `q_proj` R694,
  `k_proj` R920, `v_proj` R1372, all `TESSERA_E4M3_K1`.

**The serving premise is still unattested.** Free per-member rates require
that the runtime decodes a fused group as per-member wires it concatenates:
`bresenham_rate_schedule(root, n_columns)` is a per-COLUMN quota shared by
every row of ONE unit, so siblings concatenated along rows into a single unit
could not hold different rates. That is a fact about a runtime and no pinned
release attests it. This section states what the **allocator** may consider;
principle 9's export gate decides what ships, and there is no Tessera export
leg at all (§7).

## 4b. The same question, asked on a table built for the group

§4's numbers were measured on a cost table whose anchors were placed **per
unit**: each Linear got its own adaptive rung grid. The DP item is the fused
group, so the fold in §4 was a fold over three grids that were never chosen
together, and the qkv asymmetry (471 / 80 / 92) is that, not a property of the
tensors. The campaign now places anchors **per fused group** -- every member of
a group is measured at the same rung set, always, and the placement is driven
by whichever member's surface is worst -- and a second campaign was run that
way from scratch, so the constraint question could be asked once on a table
built for it.

`/mnt/shared/tessera-runs/pq-continuous/qwen06b_group/cost.pkl` -- Tessera
`3d419e7`, same model, same calibration draw, same gates
(`--anchors 3 --anchor-budget 12 --loo-gate 0.25 --max-artifact-bpp 5.5
--hessian off`): **9 rounds, 126 anchors, 2752 s of encode-and-score, 2758 s
wall, 7 units, 14659 priced rungs.**

**Per surface: anchors / encode seconds / leave-one-out max |log2| error.**

| unit | `E2M1_K1` | `E2M1_K2` | `E4M3_K1` |
|---|---|---|---|
| `self_attn.q_proj` | 5 / 197 s / 0.205 | 5 / 250 s / 0.182 | 8 / 40 s / **refused** |
| `self_attn.k_proj` | 5 / 194 s / 0.232 | 5 / 221 s / 0.123 | 8 / 20 s / **refused** |
| `self_attn.v_proj` | 5 / 68 s / 0.192 | 5 / 127 s / 0.149 | 8 / 20 s / 0.146 |
| `self_attn.o_proj` | 6 / 165 s / 0.221 | 5 / 163 s / 0.180 | 6 / 27 s / 0.238 |
| `mlp.gate_proj` | 4 / 89 s / 0.222 | 10 / 256 s / 0.176 | 5 / 39 s / 0.109 |
| `mlp.up_proj` | 4 / 81 s / 0.249 | 10 / 311 s / 0.079 | 5 / 37 s / 0.207 |
| `mlp.down_proj` | 6 / 195 s / 0.247 | 5 / 210 s / 0.144 | 6 / 42 s / 0.196 |

**19 of the 21 surfaces closed the 0.25 gate; none hit the 12-anchor budget;
two were refused.** `q_proj` and `k_proj` on `E4M3_K1` produced anchor sets
that are not monotone in rate -- at q256 604 -> 639 the measured dloss *rises*
(k: 0.003110 -> 0.003289; q: 0.0020754 -> 0.0020784) -- and `TesseraRateSurface`
refuses to interpolate through that rather than launder a measurement problem
into a cost. Those two families are therefore priced **only at their eight
measured anchors**; everything else on those units is unchanged.

> **A bookkeeping bug, fixed in this commit, is visible in that pickle.** The
> per-surface record read `max_abs_log2_error` with a `0.0` default, so a
> refused surface -- which has no such key -- was written down as
> `loo 0.0000, gate_closed true`: the one surface that does not interpolate at
> all, recorded as the one that interpolates perfectly. The payload's own
> `non_interpolable` list was right the whole time and is what the table above
> reads. `_surface_loo` now returns `None` and `gate_closed: False` for a
> refusal, stamps `non_interpolable: true` and `stopped_by:
> "non_interpolable"`, and the adaptive loop skips refused members when it
> takes the worst-LOO maximum instead of letting their absent error be a zero.
> **The anchor placement in this run was not affected**: a refusal contributed
> `0.0` to a `max`, which never raises it, and `v_proj` (0.146) drove the qkv
> group's E4M3 rounds either way. The numbers below stand; only their label
> was wrong.

**Group menu sizes on this table** (`__prismaquant__.tessera_group_knapsack`,
members in `k, q, v` order):

| group | family | member menus | fold frontier | options |
|---|---|---|---|---|
| qkv_proj | `TESSERA_E2M1_K2` | 539 / 435 / 92 | 539 -> 1775 -> 1866 | 1866 |
| qkv_proj | `TESSERA_E4M3_K1` | 2 / 7 / 1117 | 2 -> 10 -> 2372 | 2372 |
| gate_up_proj | `TESSERA_E2M1_K2` | 76 / 76 | 76 -> 151 | 151 |
| gate_up_proj | `TESSERA_E4M3_K1` | 1138 / 1138 | 1138 -> 2275 | 2275 |

`gate_up` is now **exactly symmetric** in both families, which is the readout
that says the per-group placement did what it was for: two units of identical
shape, measured at identical rungs, produce identical menus. `qkv` is still
asymmetric, and now for a stated reason rather than an unstated one -- q and k
carry only their 8 measured E4M3 anchors, and the menu reduction (dominance,
then bin collapse at the DP's own `--bit-precision`, applied across all three
families of a unit at once) takes those 8 to 7 and 2 respectively, because in
most of those bins an `E2M1_K2` rung is both no fatter and cheaper. Reproduced
directly from the table: `k_proj` 1290 raw rows -> 717, of which 2 are E4M3;
`q_proj` 1290 -> 442, of which 7 are.

**The three allocations on this table**, all `TESSERA_E4M3_K1` with mixed rungs
per fused group, 66-67 s wall each:

| target | achieved bpp | Δloss | one-rung (`PRISMAQUANT_TESSERA_GROUP_KNAPSACK=0`) | wall |
|---|---|---|---|---|
| 3.0 | 3.0000000 | **27.1196363** | 33.5475388 @ 3.0000000 | 67 / 63 s |
| 4.0 | 4.0000000 | **9.04154973** | 14.0901939 @ 4.0000000 | 67 / 62 s |
| 5.0 | 5.0000000 | **3.76838927** | 4.19272081 @ 5.0000000 | 67 / 60 s |

**The one-rung constraint costs 1.237x / 1.558x / 1.113x on this table**, both
arms solved against the same cost table, at the same bpp to seven digits, by
one pricing script. The lever that produces the second column is an explicit
ablation switch that disables the group fold and nothing else; with it set the
allocation logs no `tessera group knapsack` line at all, which is how the arm
is checked to be the arm.

Read the 1.558x at 4.0 with the refusals in mind: the one-rung arm has to find
a single rate that suits q, k and v, and on this table two of those three carry
an 8-rung E4M3 menu, so it is choosing under a coarser constraint than the
group fold is. The 3.0 and 5.0 ratios (1.237x, 1.113x) are close to §4's
per-unit-table ones (1.247x, 1.122x); 4.0 is the one that moved.

**Rows 1 and 2 are two rows, not one.** They were built from different anchor
placements and are priced on different tables. Nothing in this file compares a
Δloss from §4 against a Δloss from §4b, and neither should a reader: the only
comparisons made are within a table.

## 5. Two things checked rather than assumed

* **`pipeline.py` needs no entry for the token.** The declarative contract
  names `FORMATS` only as a cache-key ingredient (`"FORMATS<-COST_FORMATS"`
  and friends). It does not enumerate legal format names anywhere, so the
  `TESSERA` token is carried by `run-pipeline.sh` and `allocator.py` alone.
* **`drop_interpolated_candidates_dominated_by_measured` gates nothing.** It
  exists and is tested, but `grep` finds **no live call site** in this tree —
  only tests and docstrings reference it. It therefore does not guard the
  campaign's interpolated rows, or anyone else's. Said here rather than
  quietly relied on. What does gate the surface is the campaign's own
  leave-one-anchor-out error (`--loo-gate`, adaptive rounds) plus the
  menu-density arms below.

## 6. Scope — read the denominator before the numbers

* **The priced subset is Qwen3-0.6B decoder layer 0 only: 7 Linears**
  (`--layer-stride 28`). Every bpp figure below is over *those* quantizable
  parameters, not the model. The first run of this campaign covered layers
  0/7/14/21 at three anchors each and would have been truncated by its
  deadline mid-way through layer 7, leaving several units with an unusable
  anchor count; it was restarted narrower so that **every** priced unit gets
  the adaptive rounds that close the LOO gate. Widening the subset is a
  wall-clock decision, not a capability one — the campaign resumes from its
  own checkpoint, so `--layer-stride 7` continues rather than restarts.
* With `--formats TESSERA` the other 190 Linears get no candidate and do not
  appear in `layer_config.json`. The unit count in each allocation says which
  denominator applies.
* **The default (attested) menu is empty.** These runs set
  `PRISMAQUANT_TESSERA_MENU=research`. Under the pinned Gridbook release the
  attested contract table backs no Tessera route, so `route_admission` admits
  nothing and the DP sees no Tessera candidate at all. That is the design
  (principle 14: a route is attested, never asserted) and it means **nothing
  in this receipt is on by default**.
* **`parallel_kind` in the campaign is `PARALLEL_NONE`.** TP legality is a
  menu gate and is tested, but the campaign prices at TP=1, so no priced rung
  here exercises a sharded granularity.
* **`TESSERA_E4M3_K2` does not exist**: Tessera's own `build_forest` refuses
  arity 2 on the E4M3 grid, and `menu_families()` takes that refusal as the
  answer rather than carrying a family the encoder will not build.

## 7. Not done, and not claimed

* **No export leg, so nothing was served and no KL was measured.** The wire is
  produced and stored beside every priced render, but nothing writes a
  checkpoint and nothing loads one. §9's export gate has not been exercised by
  any of this.
* **No BF16 family.** The three families here are what
  `tessera_menu.menu_families()` derives from Tessera's own hardware bases
  today: `TESSERA_E2M1_K1`, `TESSERA_E2M1_K2`, `TESSERA_E4M3_K1`. The family
  set and the per-family activation contract are read from one table and the
  anchor rule is per family (`family_anchor_rule`), so the incoming 16-bit
  grid is a `_HARDWARE_BASES` entry plus a contract row, not a menu rewrite.
  Until it lands, `TESSERA_BF16` is absent — not deferred, absent.
* **The relaxation's serving premise is unattested** (§4). Nothing here says a
  runtime can serve a fused group at mixed rungs of one family.
* **No anchor was priced with a Hessian**, so no number here is a price of
  the bytes that ship. The encoder's shipping default consumes a per-unit
  `H = XᵀX` (LDLQ sigma 1.0 block 32, plus an exact full-H row-scale refit);
  the pinned Tessera `3d419e7` has no encoder parameter to pass one through.
  **The seam is built and gated, not the pricing**: one function owns the
  encoder call (`tessera_render.encode_tessera_unit`), the campaign computes
  `XᵀX` over *every* calibration row (not the 256 the render score keeps) and
  hard-fails on a qname miss, `--hessian require` is the default and refuses
  on this build, `--hessian off` stamps `hessian.supplied=false` on every row
  and the payload, and the allocator refuses a cost table that mixes two
  Hessian identities. When the encoder branch is pinned, the change is the
  single constant `TESSERA_HESSIAN_KWARG` — and a test fails if that constant
  ever names a parameter the pinned encoder does not have. **Blocked on the
  encoder pin, not on PrismaQuant.**
* **`render_production_weight` now owns Tessera too, and it did not before.**
  Until this branch, a `TESSERA_*` unit in a `layer_config` fell through that
  function's format cascade to the registry's synthesized
  `quantize_dequantize` — a weights-only *reconstruction*, not the decoded
  wire. So an allocator-chosen Tessera unit fed to `build_production_cache` or
  `validate_assignments_kl` would have been cached and KL-scored on bytes
  nobody encoded, silently on both counts: a principle-8 break with no error.
  It is now intercepted ahead of the cascade (`render_tessera_production`),
  forms `H = XᵀX` from `activations[qname]`, hard-fails on a missing key or a
  column-count mismatch, and returns the decoded wire. **No cache or KL run
  has exercised it** — the tests do, `test_render_production_weight_does_not_
  fall_to_the_registry_for_tessera` and
  `test_weights_only_on_the_production_seam_is_a_stamped_lever`.
  The consequence for §8.2's three allocations is concrete: feeding any of
  those `layer_config.json` files to `build_production_cache` **refuses today**
  unless `tessera_weights_only` is set, which is the same refusal
  `--hessian require` gives and for the same reason.
* **All three cache-miss RTN fallbacks refuse Tessera as well.**
  `weight_session._format_weight` and `perturbed_x_cache` fall back to the
  registry `quantize_dequantize` when `PRISMAQUANT_STRICT_PRODUCTION_CACHE=0`;
  `aura_cost._delta_w` does the same when `require_production_cache` is off,
  **which is its default** — and AURA is the default `COST_MODE`. Same
  weights-only reconstruction, one function further on, same silence. All
  three raise now, as they already did for CB. The orchestrator could not
  reach the AURA one with a Tessera name (`run-pipeline.sh:69` refuses the
  `TESSERA` token before that stage) but a direct call could, and
  "unreachable through one entry point" is not closed. Pinned by
  `test_a_production_cache_miss_does_not_fall_back_to_the_registry`,
  `test_the_perturbed_x_cache_fallback_refuses_tessera` and
  `test_the_aura_dw_fallback_refuses_tessera`. Two of them refuse *before*
  `get_format`, whose failure path in `weight_session` is `return None` — a
  refusal placed after it would have become a silent None on any box without
  the `tessera` package.
* **Asking "is this mine?" does not import Tessera.** All four sites are on
  the hot path of every *non*-Tessera format, and both `tessera_formats` and
  `tessera_render` require the `tessera` package at import — so routing the
  predicate through either would have made Tessera a hard dependency of the
  shipping NVFP4 pipeline. The predicate is
  `format_registry.is_tessera_format_name`, a prefix test on the family's own
  name grammar, which is the line `get_format` already drew for the same
  reason; `tessera_render` is imported inside the Tessera branch only. The
  sweep cannot see this (it always has Tessera on the path), so it is pinned
  by a subprocess test that blocks the import:
  `test_a_non_tessera_render_does_not_import_the_tessera_package`. On this box
  the venv carries an editable Tessera install, so nothing here would have
  failed locally — which is precisely why it is a test and not an observation.
* **The opt-out is reachable from the pipeline, not just from the API.**
  `build_production_cache` builds `levers` from `--enable` with no whitelist
  and `run-pipeline.sh` passes `PRODUCTION_CACHE_LEVERS` straight into it, so
  `PRODUCTION_CACHE_LEVERS=gptq,...,tessera_weights_only` lands as a truthy
  key. Nothing sets it by default, and whoever sets it owns the stamp.
* **Anchors are placed per `(unit, family)`, not per fused group.** The
  campaign solves each unit's top anchor independently against the
  `--max-artifact-bpp` wire cap, and because the wire→body map is
  shape-dependent (§8.2) that puts fused siblings on *different* body grids
  even when they share `in_features` and therefore share the realisable set —
  `q_proj` tops out at `R1388` where `k/v_proj` top out at `R1372`, and every
  bisected anchor below inherits the offset. On a measured-only table that
  makes the qkv E4M3 intersection collapse to one rung (§8.3). The fix is on
  the campaign side and does not need interpolation: pin a fused group's top
  anchor at the **group-minimum** body cap and bisect once for the group.
  Not implemented, and it means §8.3's aggregated `meas` arm is a weaker
  baseline than it could be.
* **The two Hessian sources are not tied to each other.** The campaign forms
  `H` from its own `_calibration_tokens(seed, nsamples, seqlen)` draw and
  stamps a `text_sha`; `render_tessera_production` forms `H` from whatever
  `activations[qname]` the production-cache build collected — a different draw,
  possibly a different row count. Nothing compares the cache build's draw to
  the cost table's `tessera_hessian` stamp, so the identity that the allocator
  refuses on is not checked by the one stage that renders the bytes. Moot
  today (no `H` reaches the encoder at this pin) and load-bearing the day it
  does: priced bytes and cached bytes would diverge again, with the identity
  sitting unread in `layer_config`. `build_production_cache` should refuse a
  calibration draw whose `text_sha` differs from the cost table's. Wired on
  the cost side, not the cache side.
* **Per-unit anchor budgets are not tuned.** Three round-one anchors, adaptive
  rounds to a LOO gate of 0.25 in |log2|, capped at three rounds. Whether that
  is the right budget per family is unmeasured; what is measured is what the
  budget bought.

## 8. The campaign

```
7 units, 16893 priced rungs, 2423 distinct formats
```

**Menu width per unit** (`cost.pkl` `menu_sizes`) — the legal rate set the three
gates admit, before any pricing:

| unit | rungs |
|---|---|
| `self_attn.q_proj` | 3055 |
| `self_attn.k_proj` | 3039 |
| `self_attn.v_proj` | 3039 |
| `self_attn.o_proj` | 3057 |
| `mlp.gate_proj` | 3060 |
| `mlp.up_proj` | 3060 |
| `mlp.down_proj` | 3063 |

That is the answer to "does the menu collapse to five rungs": it does not. The
axis is ~3000 rungs wide per unit across three families, and it is continuous
at the wire's own 1/256-bpp quantum.

**Anchor counts per family per unit.** Three round-one anchors, then adaptive
rounds placed by `next_anchor_rate` where the surface's own LOO is worst,
capped at three rounds:

| unit | `E2M1_K1` | `E2M1_K2` | `E4M3_K1` |
|---|---|---|---|
| `self_attn.q_proj` | 5 | 5 | 5 |
| `self_attn.k_proj` | 5 | 5 | 5 |
| `self_attn.v_proj` | 5 | 5 | 5 |
| `self_attn.o_proj` | 5 | 5 | 5 |
| `mlp.gate_proj` | 4 | 5 | 4 |
| `mlp.up_proj` | 5 | 4 | 5 |
| `mlp.down_proj` | 5 | 5 | 5 |

**102 anchors total, 2534 s of encode-and-score** across both runs of the
campaign (the sum of the per-anchor `seconds` in `cost.anchors.json`; run 2's
`provenance.wall_seconds` is 1566 s and excludes the 31 anchors it resumed).
**102 measured rungs became 16893 priced ones** — a 166× expansion, which is
the whole reason the campaign exists rather than an enumeration.
`non_interpolable` is empty: no family on any unit was refused a surface.

### 8.1 Leave-one-anchor-out — and it does not close on E4M3

Gate: 0.25 in |log₂| error. **15 of 21 (unit × family) surfaces are at or under
it; 6 are not**, and the round budget ran out before they closed.

| family | LOO max abs log₂ error, by unit |
|---|---|
| `E2M1_K2` | k 0.118 · down 0.145 · v 0.150 · q 0.182 · o 0.182 · up 0.237 · gate 0.259 |
| `E2M1_K1` | v 0.189 · q 0.202 · k 0.222 · gate 0.226 · up 0.243 · o 0.332 · down 0.343 |
| `E4M3_K1` | up 0.205 · q 0.235 · gate 0.243 · down 0.314 · o 0.376 · v 0.410 · **k 0.571** |

Median 0.235, worst 0.571 (`k_proj`, `E4M3_K1`).

**The acceptance item asked specifically for E4M3 interpolation gated by
LOO/regret with numbers. The number says E4M3 is the least reliable of the
three families**: it holds the four worst surfaces and misses the gate on four
of seven units (k, v, o, down), where `E2M1_K2` misses on none and `E2M1_K1` on
two. A worst-case 0.571 in log₂ is a 1.49× error in Δloss at the worst
interpolated rung of the worst unit. It is not fatal — §8.3 measures what it
actually costs at the rungs the allocator chose — but it is not a closed gate,
and the fix is more anchors on E4M3, not a wider claim.

### 8.2 The three allocations

Denominator: `body_assignment_quantizable_params = 15,728,640` (the seven
layer-0 Linears; `lm_head` is profile-pinned and excluded, per principle 12).

**`full` — the interpolated surface, default DP.** Menu into the DP:
**16893 rungs over 7 units → 8487 after dominance → 8487 after bin collapse**
at `--bit-precision 1e-4`; then, after fused-sibling aggregation into 4 DP
items (`qkv_proj`, `gate_up_proj`, `o_proj`, `down_proj`), **4315 → 4315 →
4315**. Bin collapse dropped nothing at this precision and this unit size —
worth stating, because it means the coarseness of the result is not the bin
grid's doing.

| target | achieved | q / k / v | o | gate / up | down |
|---|---|---|---|---|---|
| 3.0 | **3.00026** | `E4M3_K1_R814` | `R785` | `R824` | `R493` |
| 4.0 | **4.00026** | `E4M3_K1_R1083` | `R934` | `R1107` | `R749` |
| 5.0 | **5.00026** | `E4M3_K1_R1366` | `R1217` | `R1384` | `R909` |

Four distinct rungs per allocation, twelve distinct rungs across the three, and
every one of them is a different point on the rate axis — not a rung from a
five-entry ladder. The achieved bpp lands within **3×10⁻⁴ bpp of the target at
all three budgets**, which is the property a continuous axis exists to provide.

Four rungs, not seven, because the DP has four items: `q/k/v` are one fused
super item and `gate/up` another (§4). The rates differ *between* units and are
pinned *within* a fused group.

**Why E4M3 everywhere — and what the rate label actually means.** The
allocator is not choosing 8-bit weights, but the rung label is a *body* rate
and the budget is charged on the *wire*, so the two must not be read as the
same number. `R814` is 814/256 = **3.1797 bits/weight of body**; what the DP
is charged is the body plus that unit's scale plane and header, taken from the
format spec **for the unit's own shape** rather than derived from the label:
`R814` costs **3.2578 bpp on `q_proj` [2048x1024]** and **3.3203 bpp on
`k_proj` [1024x1024]**, because the CHANNEL plane amortises over twice as many
rows on `q`. Over the whole 3.0 assignment the param-weighted body is **2.9294
bpp** and the param-weighted charged wire is **3.000260 bpp** — which is
exactly the `achieved_bits` the allocator reports. The planes are 0.071 bpp of
the budget; the accounting is the wire's.

Why E4M3 sweeps the board has **two** candidate explanations and this receipt
separates neither:

* every candidate is priced **as it would serve**, and the `E2M1` families
  carry NVFP4's W4A4 activation leg while `E4M3_K1` carries per-token FP8, so
  the same weight rate is a different cost on the two routes — which is what
  `test_the_same_weight_rate_costs_differently_on_the_two_routes` asserts, here
  deciding an allocation rather than a unit test;
* the E4M3 WINDOW/CHANNEL body may simply be the better weight-space quantiser
  at these rates on these shapes, independently of the A leg.

Both are consistent with the assignments above. Telling them apart needs the
same allocation re-run with the A legs equalised, which was not done. Read this
as **"E4M3 wins under as-served pricing"**, not as "the A leg is why".

**DP cost.** 11 solves for the shipped target on the `full` 5.0 run, **2.32 s
total, mean 0.211 s, max 0.356 s** over a 4315-rung aggregated menu. A
continuous menu is not what makes this allocator slow.

### 8.3 What interpolation bought — and the confound in the obvious comparison

The `meas` arm solves the same three targets against **measured anchors only**.
The obvious comparison — `full` vs `meas` — is **confounded**, and the confound
is worth more than the comparison was.

**The confound, and its actual mechanism.** `aggregate_fused_siblings` builds a
super item whose menu is the **intersection** of its members' menus by format
name. A Tessera rung's name carries its *body* rate, and the campaign's anchors
are placed by bisecting between a floor and a **cap that is a wire bpp** —
`--max-artifact-bpp 5.5`. The wire→body map is shape-dependent through the
CHANNEL plane's amortisation (§8.2), so the *same* wire cap is a *different*
body rate on every shape:

| unit | shape | top E4M3 anchor | body | charged wire | next rung up |
|---|---|---|---|---|---|
| `k_proj`, `v_proj` | 1024x1024 | `R1372` | 5.3594 | **5.5000** | 5.5039 (over) |
| `q_proj` | 2048x1024 | `R1388` | 5.4219 | **5.5000** | 5.5039 (over) |
| `o_proj` | 1024x2048 | `R1390` | 5.4297 | **5.5000** | 5.5039 (over) |
| `gate/up_proj` | 3072x1024 | `R1393` | 5.4414 | **5.4987** | 5.5026 (over) |
| `down_proj` | 1024x3072 | `R1396` | 5.4531 | **5.5000** | 5.5039 (over) |

Column count is *not* the mechanism — `q_proj` and `k_proj` have the same 1024
columns and therefore the same realisable body set, yet their top anchors are
`R1388` and `R1372`. Rows are, because rows are what the plane amortises over.
Every later anchor then **inherits** that endpoint through the bisection:
(256+1372)/2 = 814 on k/v, (256+1388)/2 = 822 on q, then (256+814)/2 = 535 and
(256+822)/2 = 539. The whole grid is offset by the cap.

`E2M1`'s top anchors are the **families' own** caps, not the artifact cap —
`E2M1_K1_R768` is 3.7500 bpp and `E2M1_K2_R896` is exactly 4.0000 bpp **on
every shape**, because their s6b plane is a fixed cost per parameter. Both are
below 5.5, so the artifact cap never binds and the grids never diverge. That is
the whole difference between the families' intersections:

| family | `q_proj` measured rungs | `k_proj` | `v_proj` | **intersection** |
|---|---|---|---|---|
| `E2M1_K1` | 256,384,448,512,768 | same | same | **5 rungs** |
| `E2M1_K2` | 128,320,416,512,896 | same | same | **5 rungs** |
| `E4M3_K1` | 256,539,680,822,1388 | 256,535,674,814,1372 | 256,535,674,814,1372 | **{256}** |

**E4M3's qkv intersection collapses to the single lowest rung, R256 (1.07
bpp)** — R256 survives only because it is the shared *floor*. That is why the
aggregated `meas` arm parks q/k/v on `E2M1_K2_R512` at all three targets
including 5.0, where it has 0.36 bpp of slack: it is not "the DP reaching
across routes to fill bins", as an earlier draft of this section claimed — the
super item simply **had no usable E4M3 candidate on its menu**. The aggregated
`meas` ratios (1.76x / 3.78x / 9.84x, §8.4) are therefore mostly an
intersection artifact, and are kept below only as a footnote.

**But this is a wound the baseline inflicted on itself, and it is fixable
without interpolation.** `q`, `k` and `v` share `in_features`, so they share the
realisable body set *exactly*; only the anchor *placement* diverged, and only
because each unit's cap was solved independently against a wire budget. A
campaign that placed a fused group's anchors on **one shared body grid** — pin
the group's top anchor at the group-minimum body cap (`R1372` here) and bisect
from there — would produce identical grids on all three members and no collapse
at all, at a fraction of the interpolated surface's encode cost. **That is not
implemented** (see §7); the campaign anchors per `(unit, family)`. So this
section must not be read as "the continuous axis is needed because discrete
menus collapse": the collapse is a property of *this* anchor placement, not of
discrete menus. The interpolated surface does dissolve it — being defined on
the whole q256 grid, its intersection is the whole grid — but it is not the
only thing that would.

**The aggregation-matched comparison.** Holding aggregation fixed
(`--no-fused-aggregation` on both arms) so the only difference is menu density:

| target | `nofuse` (interpolated) | `measnofuse` (anchors only) | Δloss ratio | budget unreached |
|---|---|---|---|---|
| 3.0 | 27.176 @ 3.00729 | 29.718 @ 3.00469 | **1.094x** | — |
| 4.0 | 9.199 @ 4.00026 | 12.970 @ 3.94870 | **1.410x** | 0.052 bpp |
| 5.0 | 3.773 @ 5.00000 | 4.541 @ 4.90469 | **1.204x** | 0.095 bpp |

> **Both arms in this table run `--no-fused-aggregation`**, which §4 retires
> as a serving configuration. They are compared to each other, so the ratio is
> a clean read on **menu density** and is unaffected; it is not a read on any
> shippable assignment. Re-running the pair under the group knapsack is listed
> in §7 as not done.

**Interpolation is worth 1.09x–1.41x in Δloss at matched aggregation** — not
the 1.76x–9.84x the confounded pair suggested. Part of even that is reach
rather than ranking: the measured-only menu is **0.052 and 0.095 bpp short of
the budget** at 4.0 and 5.0 because its rungs do not land there, and unspent
budget is unspent quality. At 3.0 it does land on the budget and the gap is
correspondingly small (1.094x).

So the honest statement of what the continuous axis buys, on this subset, is:
**a 1.09–1.41x Δloss improvement at matched aggregation, plus the ability to
land on the byte budget instead of 0.05–0.10 bpp short of it.** The third
effect an earlier draft claimed — surviving fused aggregation — is real but is
not evidence for interpolation, because a shared per-group anchor grid would
buy it more cheaply (above). §8.1's LOO is the price of the reach, and it does
not close on E4M3.

### 8.4 What the fused invariant costs, and whether the relaxation binds

All Δloss figures below are recomputed by one function over the **same** cost
table (`½·h_trace·output_mse` from the full interpolated `cost.pkl`), so arms
solved against different tables are still compared on one scale.

| target | arm | aggregation | achieved | distinct rungs | Δloss | vs `full` |
|---|---|---|---|---|---|---|
| 3.0 | `full` | on | 3.00026 | 4 | 34.168 | — |
| 3.0 | `meas` | on | 2.8977 | 4 | 59.991 | 1.756× † |
| 3.0 | `nofuse` | off | 3.00729 | 7 | 27.176 | **0.795×** |
| 3.0 | `measnofuse` | off | 3.00469 | 6 | 29.718 | 0.870× |
| 4.0 | `full` | on | 4.00026 | 4 | 11.903 | — |
| 4.0 | `meas` | on | 3.9937 | 4 | 44.984 | 3.779× † |
| 4.0 | `nofuse` | off | 4.00026 | 7 | 9.199 | **0.773×** |
| 4.0 | `measnofuse` | off | 3.94870 | 6 | 12.970 | 1.090× |
| 4.0 | `free` | off | 4.00026 | 7 | 9.199 | 0.773× |
| 5.0 | `full` | on | 5.00026 | 4 | 4.232 | — |
| 5.0 | `meas` | on | 4.6391 | 4 | 41.641 | 9.840× † |
| 5.0 | `nofuse` | off | 5.00000 | 6 | 3.773 | **0.891×** |
| 5.0 | `measnofuse` | off | 4.90469 | 5 | 4.541 | 1.073× |
| 5.0 | `free` | off | 5.00000 | 6 | 3.773 | 0.891× |

† **The `meas` rows are a footnote, not a result.** With aggregation on, the
measured-only menu is intersected across fused siblings and E4M3's qkv
intersection collapses to one rung (§8.3). Those three ratios measure that
collapse, not interpolation. The comparison that holds aggregation fixed is
`nofuse` vs `measnofuse`: **1.094× / 1.410× / 1.204×**.

> **Superseded by §4.** The comparison below measures the one-rung constraint
> against `--no-fused-aggregation`, which drops the family constraint too and
> is therefore not a legal serving configuration. The correct comparison is
> the group knapsack -- one family, a rate per member -- and it puts the
> one-rung constraint's cost at **1.247x / 1.294x / 1.122x**, with the family
> constraint itself costing 1.008x / 1.000x / 1.000x against this same bound.
> The 11–23% figure below is kept as history and is not the number to cite.

**Forcing fused siblings to one rate costs 11–23% in Δloss at matched bpp.**
That is the price of the pre-DP aggregation, measured, and it is what free
per-sibling rates would buy *if* a runtime can serve them. It is a number, not
a proposal; nothing here attests that serving.

**`free` == `nofuse` exactly at 4.0 and 5.0, and `free` refused to emit at
3.0.** That pair is the direct evidence that the family relaxation is doing
work and doing it correctly:

* At 3.0 the DP's unconstrained pick put `k_proj` on `TESSERA_E2M1_K1_R512`
  while its siblings `q`/`v` were on `E4M3_K1` — two different **families** in
  one fused group, which no runtime can dispatch. The allocator's own staleness
  guard caught it and refused rather than emitting stale accounting:
  `ERROR: final serving-unit promotion changed the emitted assignment after
  achieved_bits/Delta-loss were computed … Changed 1 entries, sample=
  [('…k_proj', 'TESSERA_E2M1_K1_R512', 'TESSERA_E4M3_K1_R780')]`. So the family
  invariant genuinely binds; it is not a formality.
* With promotion on (`nofuse`), the same solve produces a legal assignment in
  which `q`/`k`/`v` hold **three different rungs of one family** — at 3.0,
  `R473` / `R780` / `R1154`. Uniform promotion would have had to lift `q` and
  `k` to `R1154`, a strict byte increase the target ratchet then pays for
  elsewhere. The relaxation is what turns "one family" into a constraint the
  DP's answer already satisfies.
* At 4.0 and 5.0 the unconstrained pick was already family-consistent, so the
  relaxation cost exactly nothing — `free` and `nofuse` agree to the last digit
  in both bpp and Δloss.

**Read `free` as an unconstrained bound, not a shippable assignment.** Without
promotion the members of a fused group can land in different families, which is
precisely what happened at 3.0.

### 8.5 Trusted-cost error at the chosen rungs

This is **not** regret in the decision-theoretic sense — regret is
true(chosen) − true(optimum under true costs), and computing that would need
the true cost of every rung, not just the chosen ones. An earlier draft called
it regret; it is not. What it measures is **the error of the trusted cost at
the rungs the DP actually selected**, which is the quantity that decides
whether the interpolated surface misled the allocator. (The `meas` vs
`measnofuse` pair in §8.3 is at least a *bound* on regret, since it compares
two feasible assignments on one table.) LOO measures the surface at a held-out
anchor; §8.3 measures what interpolation buys in reach. Neither answers: *at
the rungs this DP selected, how wrong was the cost it trusted?* So every rung
the `full` arm chose was re-encoded and re-scored through the campaign's own
path — same activations, same route contract, same render — and compared with
what the surface predicted. **21 distinct (unit, rung) picks across the three
targets; 17 had never been measured** (`verify_chosen.py`, results in
`regret_full.json`).

Per-target, in the DP's own currency over the selected rungs:

| target | predicted Δloss | true Δloss | true / predicted |
|---|---|---|---|
| 3.0 | 34.168 | 34.140 | **0.999** |
| 4.0 | 11.903 | 9.806 | **0.824** |
| 5.0 | 4.232 | 4.096 | **0.968** |

Per-unit the ratio spans **0.676 to 1.273** — the surface is 48% pessimistic at
its worst (`gate_proj` `R1107`) and 27% optimistic at its worst
(`k_proj` `R1366`, the same unit-family pair that holds the worst LOO). But the
errors are signed and largely cancel: **the aggregate the DP optimises is
within 18% at every target, and within 3% at two of three**, and it errs
*conservative* (predicted ≥ true) at all three.

That is the honest gate on interpolation. Per-rung it is not tight — §8.1's LOO
already said so and this confirms it at the chosen rungs rather than at held-out
anchors. In aggregate, at the quantity the allocator ranks by, it is close
enough that a 166× menu expansion is a good trade for it. What would make this
a stronger claim is the same measurement on a second model and a wider layer
subset; what would make it a *ship* claim is a served KL, which does not exist.

## 10. The dev pin, the attested menu, and what tensor parallelism costs

Everything above ran with `PRISMAQUANT_TESSERA_MENU=research`, which is the
menu of everything the wire can *write*. This section is about what a runtime
says it will *execute*, read from that runtime's own table (principle 14).

**The pin.** PrismaQuant's Tessera admission is fail-closed until a Tessera
RELEASE tag exists, and none has been cut. `tessera_runtime_contract.py` is the
development override, and it has exactly two states and no third:

| condition | result |
|---|---|
| `PRISMAQUANT_TESSERA_DEV_PIN` unset | no Tessera contract is read; every rung is `unattested`; the attested menu is empty |
| set to `f3e7d0ae78e64fcc1a13d5b9553a95fe4006bef4`, contract sha256 `dff4fef7...` | the packaged contract is read and its cells admit rungs |
| set to any other commit | `TesseraContractError` |
| commit right, contract bytes different | `TesseraContractError` |

A mismatch never degrades to "unattested" -- that would turn a stale pin into a
silently empty menu, which is the failure this file exists to prevent. **The
sha is the leg that attests**; the commit is declared. A worktree rsync'd to a
second box is not a git checkout and cannot be asked its HEAD, so the bytes the
reader consumed are what travels. (Tessera main has since moved to `7d86cb4`;
`git diff f3e7d0a..7d86cb4 -- src/tessera` touches only `kernel_window_gemv.py`,
`csrc/window_gemv.cu` and two files under `serving/`, and
`runtime_contract.json` is byte-identical, so the pin still verifies against
the current tree and the *encoder* is the same object at both commits.)

The contract's identity travels into every allocation's provenance as
`tessera_dev_pin` -- commit, contract sha256, path, schema, contract version,
plugin version, `attested_on`, and a `note` saying in words that this run was
admitted by a development override rather than a pinned release -- so a
shipcard records which table admitted its units. The allocator prints it too:
`[alloc] tessera dev pin: commit=... contract_sha=... plugin=0.1.0 contract_v1`.

Reading the contract does **not** import `tessera.serving`: the path is
computed by arithmetic on `tessera.__file__`, never
`importlib.resources.files("tessera.serving")`, because importing that package
registers the vLLM plugin and a producer-side contract read must not have that
side effect. There is a test that asserts the module stays out of `sys.modules`
across the read.

**What the contract actually publishes** (`tessera.runtime-contract.v1`,
lane schema `tessera.lane-eligibility.v3`, `quant_method: tessera`,
contract version 1, plugin 0.1.0):

| family | candidate rungs (q256) | cells | activation contract | route status | serve flags | max world size |
|---|---|---|---|---|---|---|
| `TESSERA_E2M1_K2` | **896** | sm_121 dense decode + batch | `e2m1_group16_ue4m3_static` | `backed_with_serve_flag` | `TESSERA_SERVE_MODE=resident|streamed` | **1** |
| `TESSERA_E4M3_K1` | **1024** | sm_121 dense decode + batch | `fp8_per_token_dynamic` | `backed_with_serve_flag` | `TESSERA_SERVE_MODE=resident|streamed` | **1** |

Four cells, two families, **one rung each**.

**So the attested menu has no rate axis.** On any shape it is exactly two
rungs: `TESSERA_E2M1_K2_R896` at 4.0000 bpp and `TESSERA_E4M3_K1_R1024` at
4.0573 bpp on a `[3072, 1024]` Linear (4.1406 on `[1024, 1024]` -- the CHANNEL
row field is charged per unit, so its bpp is shape-dependent). They are not two
points on one axis: they are two *routes*, W4A4 and W8A8, that happen to land
within 0.06-0.14 bpp of each other in weight bytes.

**This is the honest limit of "allocate continuously."** The allocator can
address ~3060 rungs per unit and the DP will use them; the runtime currently
attests two. Widening that is a change to `candidate_rungs_q256` on Tessera's
side, not a change here, and PrismaQuant should not be the place that decides
a rung is servable. Every number in §4, §4b and §8 is a **research-menu**
number and says so.

**What the default path allocates today -- measured, not described.** Every
allocation in this file until now ran under `PRISMAQUANT_TESSERA_MENU=research`,
so the dev pin's stamp and the attested menu had only unit tests behind them.
Running the same group cost table with `PRISMAQUANT_TESSERA_DEV_PIN` set and
`PRISMAQUANT_TESSERA_MENU` **unset** found a real hole first: the `TESSERA`
menu token expanded to every Tessera column the cost table held, and
`require_producer_formats` then refused the *whole run* -- the whole menu, not
the unattested part of it -- so the default path could not allocate at all and
the two backed rungs never reached the DP. The token now expands to
priced-and-attested, using the same predicate the guard refuses on, and prints
the narrowing; an explicitly named unattested rung still refuses, so only the
token narrows (`tests/test_tessera_menu.py` gate 12, 3 tests). When the
intersection is empty the run refuses with the reason and names both fixes
(widen `candidate_rungs_q256`, or price under the attested menu).

With that, on `qwen06b_group/cost.pkl` (2423 priced Tessera columns, 7 Linears
of layer 0), the default path reports
`Tessera menu: 2 of 2423 priced rungs are attested by the pinned runtime` and
allocates in ~4 s per target:

| `--target-bits` | achieved | assignment | note |
|---|---|---|---|
| 3.0 | *infeasible* | -- | "the format floor (cheapest legal format everywhere) is 4.000 bpp" |
| 4.0 | 4.000 | 7 x `E2M1_K2_R896` | the floor |
| 4.1 | 4.042 | 4 x `E4M3_K1_R1024`, 3 x `E2M1_K2_R896` | the ceiling |
| 5.0 | 4.042 | same | menu exhausted 0.96 bpp below target |

The three E2M1 units are exactly the `qkv` group, and the reason is the
refusal in §8.1 rather than a preference: `q_proj` and `k_proj` never priced an
E4M3 rung at R1024 (their E4M3 surfaces were refused as non-interpolable), so
the group's E4M3 Minkowski fold does not exist and the family constraint puts
all three on E2M1. The whole attested range is **4.000-4.042 bpp on this
model** -- 0.042 bpp wide, two routes, no axis -- and `layer_config.json`
carries the `tessera_dev_pin` block naming commit `f3e7d0a`, contract sha
`dff4fef7...`, plugin 0.1.0, and the development-override note.

**Tensor parallelism, on Tessera's own granularity.** `tessera_shard_granularity`
no longer derives the shard period locally; it builds the unit geometry from
`tessera_wire_recipe` plus the family's column schedule and hands it to
`tessera.layout.shard_granularity(geometry, 256, arity)`. The local derivation
was wrong in a way that mattered: a **mixed** Bresenham schedule
(`len(set(rates)) > 1`) raises the column period to the full 256-column
superblock, because a per-column rate quota is only meaningful over the
superblock it was computed for. Measured on the research menu, `[3072, 1024]`:

| shard kind | tp=1 | tp=2 | tp=4 | tp=8 |
|---|---|---|---|---|
| column-parallel (`out_features` split) | 3060 | 3060 | 3060 | 3060 |
| row-parallel (`in_features` split) | 3060 | 3060 | 3060 | **17** |

At tp=8 a row-parallel `[3072, 1024]` Linear gives each rank 128 input columns,
which is half a superblock, so every mixed-schedule rung is illegal and only
the 17 uniform-schedule rungs survive. That collapse is Tessera's answer, not
PrismaQuant's arithmetic, and it is read through **one** function so there is
one place to look when it changes.

The TP gate has two legs and they are separate on purpose: `can_shard` asks
whether the *wire* survives the cut, and `tessera_tp_world_attested` asks
whether the *runtime* claims that world size. In the attested menu the second
leg is decisive -- both families publish `max_world_size: 1` under a
`closed_world` policy, so **the attested menu is empty at every tp > 1**,
measured, on both shard kinds. At tp=1 the attestation leg is not consulted at
all: a whole unit makes no tensor-parallel claim.

## 11. The Hessian seam

At `f3e7d0a` Tessera's encoder takes activation statistics through a frozen
`tessera.export.ActivationSource` -- `hessians` keyed by tensor name minus one
`.weight`, `provenance`, `ldlq_sigma=1.0`, `ldlq_block=32`,
`refit_objective="hessian"`, `refit_reach_floor=False` -- and
`ActivationSource.for_unit(name, in_features, device)` emits the per-unit
encoder kwargs. A missing key raises rather than falling back. PrismaQuant now
forms that object in exactly one place.

`prismaquant/tessera_hessian.py` is that place: one formation
(`hessian_from_rows`: fp32, `reshape(-1, in_features)`, `X^T X`,
un-normalised), one identity (`calibration_identity`, whose required fields are
read from `tessera.export.HESSIAN_IDENTITY` rather than restated --
`text_sha256`, `fit_tokens`, `fit_ids_sha256`), and two callers: the anchor
campaign and the production render. A test asserts both callers reach the same
function and that the two paths' identities are the same draw; another asserts
a production render **refuses** rows whose `hessian_identity` names no draw.

**The applicability is a measured platform fact, and it is not uniform.**
Passing activation kwargs to a non-CHANNEL rung raises `GrammarError` at
`f3e7d0a`: `ldl` is documented "LDLQ is implemented for the CHANNEL scale
plane" and `refit_metric` "read only by the CHANNEL plane's refit". Of the
three families this work prices, `TESSERA_E4M3_K1` is CHANNEL throughout and
**both E2M1 families are LUT16 throughout**, so:

* `rung_accepts_hessian(format_name)` answers by asking the wire recipe for the
  rung's scale plane -- not by a hardcoded family list;
* a block-plane rung is priced **weights-only, not refused** -- the campaign
  stamps `hessian_applied: false` on the row and the row is still a legal cost;
* activation kwargs handed to a block-plane rung **are** refused, as is a
  weights-only encode that was handed kwargs ("applied and unrecorded"), and so
  is any kwarg name `ActivationSource.for_unit` does not produce;
* per-row `hessian_identity` carries `applied`, so a table can hold both kinds
  of row and a reader can tell them apart. `applied` is deliberately *not* part
  of the uniform-identity key -- one table has one calibration draw, and rows
  differ in whether that draw reached the encoder, which is a property of the
  rung, not of the draw.

**Producer-side blocker, reproduced.**
`export_checkpoint(weights, plan, dir, grid=E2M1x2, activation=ActivationSource(...))`
raises `GrammarError` on any E2M1 unit at `f3e7d0a`. A mixed-family Tessera
artifact therefore cannot be exported with one `ActivationSource` today. The
render seam handles this per rung; the exporter does not. There is no Tessera
export leg in this tree (§7), so nothing here is blocked by it -- but the first
export that mixes families will be.

**Row 3 -- the H A/B -- is in flight, not reported here.** Two campaigns on the
same 0.6B layer at Tessera `f3e7d0a`, identical in every argument except
`--hessian require` vs `--hessian off`
(`sparklina:/home/rob/tmp/run_campaign_row3.sh`, log
`sparklina:/home/rob/tmp/row3.log`, outputs
`/mnt/shared/tessera-runs/pq-continuous/qwen06b_h_{on,off}/`). What is
established so far is the fail-fast leg: the H-aware path *runs*. Round 1 of
the `require` arm reached
`r1 40/63 model.layers.0.self_attn.q_proj TESSERA_E4M3_K1_R814 mse=0.000348269`
with zero `FAILED` anchors, so `for_unit` -> `block_ldl` -> `encode_linear`
with all four kwargs encodes rather than raising, on real calibration
activations (2048 rows per Linear, `in_features` 1024-3072).

**When it lands, compare it per shared anchor, not per allocation.** Both arms
are adaptive, so after round 1 they place different anchors and an
allocation-level comparison mixes the H effect with placement -- the same
confound §4b exists to warn about. Round-1 anchors are deterministic, so every
surface shares at least three rungs. On the intersection of
`(unit, family, rung)` from `cost.anchors.json`: the E2M1 anchors must be
**bit-identical** in dloss (weights-only in both arms), and the E4M3 anchors
must **differ** (H applied in one, not the other). E4M3-identical would mean
the kwargs are accepted but inert, which no unit test with a synthetic H
catches. Row 2 vs the `off` arm on the same intersection gives the
`3d419e7` -> `f3e7d0a` encoder drift as its own number.

**Scope caveat to record with it:** `down_proj` has `in_features=3072` against
2048 calibration rows, so its X^T X is **rank-deficient by construction**.
`ldlq_sigma=1.0` is what keeps `block_ldl` defined there; a `down_proj` E4M3
result is a regularised-H result and should not be read as "the H is worth
this much" on a unit whose H is not full rank.

## 9. Where the acceptance criteria landed

| asked | result |
|---|---|
| Tests pass, listed with the command | §3. **436 passed** on the full sweep over 22 suites, including every allocator suite this branch could regress and the seven render/cache/cost/registry suites the Tessera interception and the fallback refusals could. One pre-existing failure outside the sweep, reproduced with this branch stashed, is named in §3. |
| Three 0.6B allocations spanning more than a few distinct rates | §8.2. Menu 3039–3063 rungs **per unit**; 16893 priced rungs over 7 units. Assignments hit 3.00026 / 4.00026 / 5.00026 bpp with 4 distinct rungs each (4 DP items), and 6–7 distinct rungs each without pre-aggregation. Not a five-rung ladder. |
| Anchor counts per family per unit | §8 (per-unit placement): 4–5 per family per unit, 21 surfaces, 102 anchors, 2534 s. §4b (per-**group** placement, the correct one): 4–10 per surface, 21 surfaces, **126 anchors, 2752 encode-seconds**, per-surface anchors/seconds/LOO tabulated. 19 of 21 surfaces closed the 0.25 gate; **none hit the 12-anchor budget**; two E4M3 surfaces were refused as non-interpolable. |
| Anchors placed per fused group, adaptive until the gate closes or the budget is hit | §4b. Placement is per group unconditionally; the loop ran 9 rounds and stopped on the gate, never on the budget. |
| The fused-group constraint as an exact Minkowski sum, Pareto-pruned, never hull-only | §4. Pinned against brute force on a menu with a non-convex pocket; the per-NAME and group constructions must agree on the shared uniform-rung option or the aggregation refuses. §4 also shows the fold **containing** the unconstrained arm's own rung triple at 4.0. |
| Group menu sizes and DP time | §4 (per-unit table) and §4b (per-group table). 151–3754 options per (group, family); 60–67 s wall per target end-to-end. |
| Three allocations re-run under the correct constraint | §4b. 27.1196363 / 9.04154973 / 3.76838927 at exactly 3.0 / 4.0 / 5.0 bpp; the one-rung constraint costs **1.237× / 1.558× / 1.113×** on that same table. |
| `nofuse` retired as an arm | §4. It drops the family constraint too and is not a legal serving configuration; it survives only as a bound, and §4 shows the constrained menu beating that bound at matched bytes. |
| Hessian kwarg constant set; render seam passes H through `ActivationSource` | §11. The constant is gone -- there is no kwarg name to pin. `tessera_hessian.py` forms one `ActivationSource` from PrismaQuant's own calibration activations and both callers use it. Applicability is CHANNEL-plane-only, measured, and stamped per row. |
| Default menu attested from the packaged contract, not `MENU=research` | §10. `PRISMAQUANT_TESSERA_DEV_PIN` reads `tessera/serving/runtime_contract.json` at the named commit and sha. **The attested menu is two rungs, one per family, and empty at every tp > 1.** |
| The default path actually allocates from the attested menu | §10. Run measured, not asserted: `2 of 2423 priced rungs are attested`, floor 4.000 bpp, ceiling 4.042 bpp, `--target-bits 3.0` infeasible, and `layer_config.json` carries the `tessera_dev_pin` block. Finding it required a fix: the `TESSERA` token used to expand to the unattested columns too, which made `require_producer_formats` refuse the whole run. |
| TP gate on the real `shard_granularity` | §10. `tessera.layout.shard_granularity` through one function; row-parallel `[3072,1024]` collapses 3060 → 17 at tp=8, and `max_world_size: 1` closes the attested menu above tp=1. |
| The two Hessian sources are one draw | §11, `test_the_campaign_and_the_production_render_form_one_hessian` and `test_one_identity_function_answers_for_both_paths`. |
| A-leg pricing test exists | §3, `test_the_same_weight_rate_costs_differently_on_the_two_routes`, and §8.2 shows it deciding a real allocation. |
| E4M3 interpolation gated by LOO/regret, with numbers | §8.1 (LOO: **E4M3 is the worst family**, 4 of 7 units miss the 0.25 gate, worst 0.571), §8.3 (what interpolation buys at matched aggregation: **1.094× / 1.410× / 1.204×**, plus 0.052 / 0.095 bpp of budget the sparse menu cannot reach), §8.5 (trusted-cost error at the chosen rungs: 0.999 / 0.824 / 0.968, conservative at every target). **The LOO gate does not close on E4M3** — stated, not smoothed. |
| `ARCHITECTURE.md` updated | §4.10, in the same commits as the code. |
| Commits on `tessera/continuous-menu` | Yes; no pushes. |
| Menu size per unit and DP time on 0.6B | §8.2. DP: 11 solves, 2.32 s total, mean 0.211 s, max 0.356 s over a 4315-rung menu (0.72 s max on the 8487-rung unaggregated menu). |
| Served KL | **None. Not measured, not claimed** (§7). |

**Fable consultations: none.** No sub-question in this task was judged to need
the top rung; the two that were close — whether the fused-group collapse was a
promotion defect, and whether "regret" meant density or trusted-cost error —
were settled by reading the code and by measuring, which is cheaper and is
evidence rather than opinion.

**Files.** Worktree `/home/rob/pq-wt/tessera-continuous` (branch
`tessera/continuous-menu`). Run outputs, all on the shared mount:
`/mnt/shared/tessera-runs/pq-continuous/qwen06b/` — `probe.pkl`, `cost.pkl`,
`cost.anchors.json`, `cache/wire/` (one wire blob per priced anchor),
`alloc/lc_{full,meas,measnofuse,nofuse,free}_{3.0,4.0,5.0}.json`,
`alloc/log_*.txt`, `measnofuse.log`,
`regret_full.json`, `campaign2.log`, `alloc_suite.log`, `verify.log`.
Checked in beside this receipt: `data/tessera-continuous-regret_full-2026-09-02.json` (the trusted-cost
error of §8.5) and `data/tessera-continuous-menu-density-2026-09-02.json`
(the §8.3 intersection table, all five arms' assignments, and the
aggregation-matched ratios recomputed on one cost table).
The per-group campaign of §4b:
`/mnt/shared/tessera-runs/pq-continuous/qwen06b_group/` — `cost.pkl`,
`cost.anchors.json`, `cache/wire/`; driver
`sparklina:/home/rob/tmp/run_campaign_g.sh`.
The default-path (attested) allocations of §10:
`/home/rob/tmp/pq-continuous/attested/lc_att_{4.0,4.1,5.0}.json` with
`attested.log`; driver `run_attested.sh`.
Row 3 (the Hessian A/B, Tessera `f3e7d0a`):
`sparklina:/home/rob/tmp/row3.log`, driver
`sparklina:/home/rob/tmp/run_campaign_row3.sh`, outputs
`/mnt/shared/tessera-runs/pq-continuous/qwen06b_h_{on,off}/`.
Scratch drivers: `/home/rob/tmp/pq-continuous/{run_allocations.sh,
run_mink.sh,run_mink_group.sh,run_onerung_group.sh,run_a1.sh,run_attested.sh,
price_mink.py,price_group.py,price_onerung.py,
measured_only_cost.py,verify_chosen.py,summarise.py,checkpoint_to_cost.py}`,
with logs beside them (`mink_group.log`, `onerung_group.log`, `sweep_b5.log`).
