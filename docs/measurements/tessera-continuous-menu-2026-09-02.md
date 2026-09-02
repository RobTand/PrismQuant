# Tessera's continuous rate axis on the production allocator — 2026-09-02

**Branch** `tessera/continuous-menu` · **PrismaQuant** `90b5876` (+ this
receipt) · **Tessera** `3d419e7b4d6ffd0259fcb7c36ea179bddc1d7ce9` (main tree
HEAD; `src/tessera/serving/` never imported — another worker owns it) ·
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
