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
