# Packed campaign driver integration — 2026-09-06

Issue: #293, connecting #282 and #290. This is CPU correctness evidence. No
GPU throughput, served quality, or GLM streaming qualification is claimed.

The original `c488b592` planner fails on the checked-in attributable LFM
layer-18 packed probe fixture: it asks for per-expert keys that the original
probe does not carry. PB action
`fb9883239d0645040627064141cef0c1eaea945727f9c4eb1ecccf8979a05879`
recorded both driver regressions failing with “no h_trace for 32 of 32 experts”.
Its terminal log is under `/mnt/shared/prismabuild-fleet/pb-queue/failed/`.
The first attempt (`6a6021e7ba79`) instead failed setup because the `pb-cpu`
environment lacked torch; it is not evidence of the regression.

With the integration, planner/selection, existing fanout, and architecture
checks passed: **57 passed, no skips**, action
`0ca7ff015ce761752dea5c1e1d1d73fadfe6c4fa6cce94b9d6d80d6ee9341c34`,
receipt `9050e44344e94bc2a26eee85180731dbb7c7ecfa449ae0bccb4de0bb734632d4`.

The driver and population bridge then passed together: **13 passed, no skips**,
action `7e4cb4fc6bd79c0bbb61984aad3788014a0b6afb903106a6930ea5de64d6aef6`,
receipt `a31a4948079497101b96961e2394cbaa7915098b22ca5d217f53e18a5da967ed`.
This includes the actual campaign CLI using the existing CPU model fixture,
real producer projection/encoding/capture/wire receipts, persisted sampling
selection and packed population output; and scoped `allocator.main()` with a
packed census and twelve source receipts. Missing unsampled wires refuse,
while an explicitly BF16 stack needs no Tessera wires. The producer fixture
replaces served-route scoring; it is not a serving smoke.

Commands used the published `pbrun.py --tag gb10 --cpus 2 --demand mem_gb=6`
(no GPU demand, GPU visibility disabled) and
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q -rs`
against `tests/test_tessera_stack_driver_integration.py` and
`tests/test_tessera_stack_population_bridge.py`, with OMP/MKL/OpenBLAS native
threads bounded to one. The successful producer run set
`TESSERA_REPO=/mnt/shared/prismaquant-validation/stack-producer-ba582d` and
`PYTHONPATH=.:/mnt/shared/prismaquant-validation/stack-producer-ba582d/src`.
That shared source tree is `git archive` of Tessera
`ba582d476a3b6db9057ebd1385dc52926f171451`, matching this PrismaQuant runtime pin.
An earlier run without `TESSERA_REPO` reported 12 passes and one explicit
missing-tool skip; the pinned rerun closed that skip.

The old unlanded allocator test used 32-column shapes that Tessera's serialized
size contract rejects. Its initial failure occurred before topology handling;
using supported 256-column shapes makes the intended scoped allocation test
meaningful without changing allocator admission or size accounting.

The final allocator assertions additionally check the selected format and
`routed_moe` route in emitted metadata, and checkpoint assertions use the
journal's canonical identity digest. The two changed assertions were rerun:
**2 passed, 4 deselected**, action
`4287303842fa8418b0efe1fe4ae466e2173c9a32ae5cc0432d48038c9d70f068`,
receipt `eb7160fb8a25d2b2dcc4f0852463308c0eee8e5bef77bead7caded0df2d230bc`.

## Producer translator handoff

A subsequent end-to-end review found a second name boundary: full-census
scoped preflight passed, but the pipeline passed the original packed
allocation to Tessera's translator. The pinned producer refused
`experts.down_proj` because the source names per-expert `w1/w3/w2` tensors.
The actual producer `build()` regression failed through PB action
`aebcba1ccd587500bd22e708a2662c519a865ddf4034015adfb99eae09555512`.

Preflight now derives a separate source-unit assignment from the already
verified population mapping, preserving the original allocation and metadata.
The build anchor records the derived file's path and digest; the shell hands
that file to the producer translator and checks the digest before and after.
Tessera still owns translation to its wire plan. The packed allocation remains
the artifact's original allocation identity.

A shell regression also established that merely passing a new cache-setting
name does not bind it: `pipeline.STAGE_SETTINGS_KEYS` ignored the undeclared
`PLAN_ASSIGNMENT_DIGEST` and reused an old plan. The fix declares that optional
key in the existing settings projection. The regression now refuses that
cached plan, as well as a derived file changed after preflight.

Final relevant coverage: **92 tests passed without skips** across eight test
files: real pinned producer handoff (1), shell plan binding (11), packed export
scope (7), expert projection (20), stage settings (17), pipeline contracts
(17), architecture (13), and documentation staleness (6). These were targeted
PB fanout runs; successful suites were not repeated after unrelated test-only
corrections. Result manifests are `/home/rob/tmp/pq-stack-plan-tests.json`,
`pq-stack-plan-contract-tests.json`, and `pq-stack-plan-final-tests.json`.
The producer handoff's receipt is
`e10493d23a79c21ce4cc26432011ae280c91e39c2674a98bd8d8f4e1708b15e0`.
Full shell syntax and Python compilation passed in action
`95257284c29ed9acfeba6027ce8950cace52710ec69535354cd308fa3b1b1036`,
receipt `01889714277e009664e2236a3c6f02fb9f4d0da9aa8918118c409d11686212fe`.

The fanout used `pbtest.py --tag gb10`, CPU only, with one native thread per
worker for final parallel checks. Its temporary interpreter wrapper supplied
the same immutable `ba582d` producer archive described above; those helper
bytes are retained by the PB checkout snapshots. The actual translator test
uses fixture source shapes and verified fake wire receipts at the preflight
boundary; it exercises naming, identity and plan content without encoding or
serving an artifact. Missing unsampled wires and static scales remain refusals.
