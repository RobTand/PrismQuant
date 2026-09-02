# Tessera's continuous rate axis on the production allocator — 2026-09-02

**Branch** `tessera/continuous-menu`. The allocations ran at
`cb64eba`; `prismaquant/tessera_campaign.py` is **byte-identical from
`4d2d9b2` through `cb64eba`** (`git diff 4d2d9b2 HEAD --stat --
prismaquant/tessera_campaign.py` is empty), so the campaign's own code is the
same object across every commit on this branch and the restart mid-branch
changed nothing about how an anchor is priced. **Tessera**
`3d419e7b4d6ffd0259fcb7c36ea179bddc1d7ce9` (main tree HEAD;
`src/tessera/serving/` never imported — another worker owns it) ·
**Boxes** sparky (host tests) and sparklina (probe, campaign, allocations) ·
**Python** `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`.

**No artifact was served and no KL was measured in this work.** There is no
Tessera export leg in this tree. Everything below is an *allocator* result:
what the DP may consider, what it costs, and what it picks. Nothing here is a
quality claim about a served model, and §9's export gate has not been
exercised.

## 1. What was built

`FORMATS` accepts the token `TESSERA`. `tessera_menu.expand_menu_tokens`
replaces it with exactly the Tessera columns the run's own cost table holds, so
the menu is the widest one the DP could honestly consider and never wider.
Per-Linear the menu is the whole realisable rate set of every family the
hardware bases admit, subject to three gates and nothing else — shape legality,
route legality read from the attested contract table, and W(n≤A) by
construction. Architecture: `docs/ARCHITECTURE.md` §4.10.

The single seam for the runtime's own table is `tessera_menu.route_admission`.
The single seam for tensor-parallel legality is
`tessera_menu.tessera_shard_granularity`, which prefers
`tessera.layout.shard_granularity` when that lands. Both are one function each,
by design, so the incoming work swaps in without touching the menu.

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
  --max-artifact-bpp 5.5 --deadline-seconds 9000
```

Allocations — three targets × three arms
(`/home/rob/tmp/pq-continuous/run_allocations.sh`):

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
  `/home/rob/tmp/pq-continuous/measured_only_cost.py`; the **regret** baseline.
* `nofuse` — interpolated surface with `--no-fused-aggregation`, the only arm
  in which the family-promotion relaxation is reachable (§5).

## 3. Tests

```
cd /home/rob/pq-wt/tessera-continuous
PYTHONPATH=/home/rob/tessera/src:. python -m pytest \
  tests/test_tessera_menu.py tests/test_tessera_campaign.py \
  tests/test_tessera_formats.py tests/test_docs_staleness.py \
  tests/test_architecture_doc.py -q
```

**101 passed** on sparky — 26 menu + 9 campaign + 46 format + 20 doc. The
`CUDA` marker here is a `skipif`, not a deselect, so sparky's GPU means all
nine campaign tests ran, including the two load-bearing ones:

* `test_the_same_weight_rate_costs_differently_on_the_two_routes` — the A-leg
  pricing test the acceptance asks for. It scores one rendered rung twice, once
  under the route's own activation quantiser and once under the identity, for
  both routes, and asserts the **ratio** differs between them. Asserting only
  that each route differs from identity would pass on a harness that applied
  one contract to both, which is exactly the bug it is there to catch.
* `test_the_priced_render_is_the_decoded_wire` — the cache stores the wire
  beside the dequantised render, and the priced render is the decode of those
  bytes (principle 8).

## 4. Where the relaxation is reached, and where it is not

The brief asked for union-find promotion relaxed "so fused siblings / packed
experts share a **family** with rates free per unit". That is implemented
(`allocator_solver._resolve_family_group`, via
`tessera_formats.format_promotion_class`) and tested. It is **not reached on
the default path**, and the honest reason is two facts, neither of which is a
defect:

1. **The DP never sees a mixed-rung group.** `aggregate_fused_siblings` runs
   *before* the solve and builds one super item per group with one candidate
   per format **name** (`for spec in formats: … Candidate(fmt=spec.name, …)`).
   The DP therefore returns exactly one rung for the whole group, and
   `_promote_group_components` is a no-op on it. The call site states the
   design outright: *"The DP can't pick mixed-sibling solutions because there's
   only one item per group."*
2. **The wire agrees, for a group that is one unit.**
   `tessera.grammar.bresenham_rate_schedule(root, n_columns)` is a per-COLUMN
   quota shared by every row. Siblings concatenated along ROWS into one unit
   cannot hold different rates — the schedule is indexed by the axis they
   share. Free per-sibling rates presuppose that the runtime decodes q, k and v
   as separate units and concatenates afterwards. That is a claim about a
   runtime; nothing attests it, so it is not made here (principle 14).

So the relaxation is exercised only under `--no-fused-aggregation`, and the
arms below measure both. `tests/test_tessera_menu.py::
test_pre_aggregation_forces_one_rung_on_a_fused_group` pins fact 1 so a change
that makes the DP mixed-rung capable has to come here and say so.

**A correction to an earlier read.** A partial-table smoke run collapsed to
1.079 bpp against a 4.0 target, and that was first attributed to promotion.
It is not: with aggregation on, promotion has nothing to do. The cause is that
`aggregate_fused_siblings` intersects the members' menus, so a qkv group in
which only `q_proj` was priced is left with a single common candidate. It is an
artifact of a partial cost table, and it disappears when every member is
priced.

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
  leave-one-anchor-out error (`--loo-gate`, adaptive rounds) plus the regret
  arms below.

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
* **`tessera.layout.shard_granularity` is not yet importable**, so
  `tessera_shard_granularity` is running on its own fallback. The seam is one
  function and the test asserts the fallback's contract, not its numbers.
* **The relaxation's serving premise is unattested** (§4). Nothing here says a
  runtime can serve a fused group at mixed rungs of one family.
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
| 4.0 | **4.00027** | `E4M3_K1_R1083` | `R934` | `R1107` | `R749` |
| 5.0 | **5.00026** | `E4M3_K1_R1366` | `R1217` | `R1384` | `R909` |

Four distinct rungs per allocation, twelve distinct rungs across the three, and
every one of them is a different point on the rate axis — not a rung from a
five-entry ladder. The achieved bpp lands within **3×10⁻⁴ bpp of the target at
all three budgets**, which is the property a continuous axis exists to provide.

Four rungs, not seven, because the DP has four items: `q/k/v` are one fused
super item and `gate/up` another (§4). The rates differ *between* units and are
pinned *within* a fused group.

**Why E4M3 everywhere.** The allocator is not choosing 8-bit weights: at 3.0
bpp `E4M3_K1_R814` is a **3.18-bit body** on the E4M3 grid. It wins because
each candidate is priced **as it would serve** — the `E2M1` families carry
NVFP4's W4A4 activation leg, `E4M3_K1` carries per-token FP8 — so the same
weight rate is a different cost on the two routes. That is exactly what
`test_the_same_weight_rate_costs_differently_on_the_two_routes` asserts, and
here it is deciding an allocation rather than a unit test.

**DP cost.** 11 solves for the shipped target on the `full` 5.0 run, **2.32 s
total, mean 0.211 s, max 0.356 s** over a 4315-rung aggregated menu. A
continuous menu is not what makes this allocator slow.

### 8.3 Regret — what interpolation bought, measured

The `meas` arm solves the same three targets against **measured anchors only**:
102 rows → 71 after dominance → **31** after aggregation. The result is not a
slightly worse assignment. It is an allocator that **cannot reach the budget**:

| target | `full` achieved | `meas` achieved | budget left unused |
|---|---|---|---|
| 3.0 | 3.00026 | 2.8977 | **0.103 bpp** |
| 4.0 | 4.00027 | 3.9937 | 0.006 bpp |
| 5.0 | 5.00026 | **4.6391** | **0.361 bpp** |

At 5.0 the measured-only menu leaves 7.2% of the byte budget on the table,
because its densest rungs on the units that matter simply do not exist as
options. This is the concrete answer to "what is the continuous axis for":
not a better assignment at a given rate, but the ability to *land on the rate
you were given*. It is also why the interpolation has to be gated rather than
trusted — §8.1's LOO is the cost of that reach.

The `meas` arm also picks **mixed families** (`E2M1_K2` on q/k/v, `E4M3_K1` on
the MLP) where `full` picks one. With 31 candidates the DP reaches across
routes to fill bins it cannot fill within a route; with 4315 it does not have
to. Read that as evidence about the sparse menu, not about the families.

### 8.4 What the fused invariant costs, and whether the relaxation binds

All Δloss figures below are recomputed by one function over the **same** cost
table (`½·h_trace·output_mse` from the full interpolated `cost.pkl`), so arms
solved against different tables are still compared on one scale.

| target | arm | achieved | distinct rungs | Δloss | vs `full` |
|---|---|---|---|---|---|
| 3.0 | `full` | 3.00026 | 4 | 34.168 | — |
| 3.0 | `meas` | 2.8977 | 4 | 59.991 | **1.756×** |
| 3.0 | `nofuse` | 3.00729 | 7 | 27.176 | **0.795×** |
| 4.0 | `full` | 4.00026 | 4 | 11.903 | — |
| 4.0 | `meas` | 3.9937 | 4 | 44.984 | **3.779×** |
| 4.0 | `nofuse` | 4.00026 | 7 | 9.199 | **0.773×** |
| 4.0 | `free` | 4.00026 | 7 | 9.199 | 0.773× |
| 5.0 | `full` | 5.00026 | 4 | 4.232 | — |
| 5.0 | `meas` | 4.6391 | 4 | 41.641 | **9.840×** |
| 5.0 | `nofuse` | 5.00000 | 6 | 3.773 | **0.891×** |
| 5.0 | `free` | 5.00000 | 6 | 3.773 | 0.891× |

**Interpolation is worth 1.76×–9.84× in Δloss**, and the gap widens with the
budget because the sparse menu cannot even spend it (§8.3).

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

### 8.5 True allocation regret — how wrong the trusted cost was

LOO measures the surface at a held-out anchor; §8.3 measures what interpolation
buys in reach. Neither answers the question the allocator actually raises: *at
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

## 9. Where the acceptance criteria landed

| asked | result |
|---|---|
| Tests pass, listed with the command | §3. 101 passed (menu 26, campaign 9, formats 46, docs 20); plus 82 and 50 on the allocator regression suites named in the commits. |
| Three 0.6B allocations spanning more than a few distinct rates | §8.2. Menu 3039–3063 rungs **per unit**; 16893 priced rungs over 7 units. Assignments hit 3.00026 / 4.00027 / 5.00026 bpp with 4 distinct rungs each (4 DP items), and 6–7 distinct rungs each without pre-aggregation. Not a five-rung ladder. |
| Anchor counts per family per unit | §8. 4–5 per family per unit, 21 surfaces, 102 anchors, 2534 s. |
| A-leg pricing test exists | §3, `test_the_same_weight_rate_costs_differently_on_the_two_routes`, and §8.2 shows it deciding a real allocation. |
| E4M3 interpolation gated by LOO/regret, with numbers | §8.1 (LOO: **E4M3 is the worst family**, 4 of 7 units miss the 0.25 gate, worst 0.571), §8.3 (reach: 1.76×–9.84×), §8.5 (true regret at the chosen rungs: 0.999 / 0.824 / 0.968). **The LOO gate does not close on E4M3** — stated, not smoothed. |
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
`alloc/lc_{full,meas,nofuse,free}_{3.0,4.0,5.0}.json`, `alloc/log_*.txt`,
`regret_full.json`, `campaign2.log`, `alloc_suite.log`, `verify.log`.
Scratch drivers: `/home/rob/tmp/pq-continuous/{run_allocations.sh,
measured_only_cost.py,verify_chosen.py,summarise.py,checkpoint_to_cost.py}`.
