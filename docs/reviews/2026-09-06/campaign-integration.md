# Campaign integration review — 2026-09-06 EDT

This review covers the day's pushed main changes through PrismaQuant
`c488b592`, Tessera `f8b87ce4`, and PrismaBuild `645f440`, plus the open primary
campaign issues/PRs and the follow-up fixes produced during review. The sibling
reports are source/evidence ledgers at their stated cutoffs; later integration
and merge status belongs here. The original working checkouts and historical
model artifacts were preserved.

## Repaired campaign boundaries

- The planner now persists the original packed Fisher probe and replayable
  expert draw. Campaign execution binds that receipt in its checkpoint and
  computes packed decisions using shared gate/up projections.
- Population metadata explicitly maps each packed decision to its full source
  population and measured subset. Allocation verifies full source receipts
  without double-pricing packed and expanded members. Full census wires admit
  measured rungs; interpolation cannot invent exportable wire bytes.
- Export scope shares allocation's decision expansion, retaining whole-stack,
  source geometry, wire-byte and static-scale refusals. A partial sample with
  missing unsampled wires remains insufficient for export.
- Merge reuses the journal's checksum/identity reader before rewriting state,
  validates rosters and canonical paths, preserves full sampling metadata,
  rebuilds packed population, unions audit/empty-menu evidence and requires
  shared rate-band/checkpoint inputs to agree.
- Sampling rejects nonfinite/negative Fisher values, invalid probabilities and
  omitted certainty units. Repeated draws using the production PPS design and
  attributable LFM probe validate point estimates; the existing HR approximation
  is not an exact systematic-design standard error and does not gate allocation.
- Common-family projection retries match family names across differing menus.
  This repairs the existing bounded request sequence without introducing
  per-stack checkpoint hashing or a Cartesian search.

## Frozen fanout equality

`tools/pq282_equality/recheck_refusal_order.py` re-merges the saved v7 rows using
current code. It first proves that historical refusal content and multiplicity
match, canonicalizes only refusal ordering in a separate monolith comparison
copy, and runs the existing equality tool across prices, population, captures
and journal identity. No encodes occur and historical files remain unchanged.

Final recheck action:
`1414446fd468e25fbe04ae42a0930b1d14dd6aeeef05307eaf4b5d6a50b8531e`.
CPU-only, sparklina, 2 CPU/8 GiB, native threads1, exit0. Actual output is
**EQUAL**: 3 units, 10,728 priced rows, 3,578 formats, 37 wire blobs.
CAS receipt `ff4c90a604ea9d31c7535453c91ed9865621eaeeadc5cb62ee03a75d7683c497`;
result `72f2398819ba32296f738add5b45ce9777d100df9de0ebaa1e25af0da70633ba`.
Output directory:
`/mnt/shared/tessera-measurements/astra-review-20260906/pq282-recheck-03`.
This repairs equality of the existing bounded artifact; it is not an unsampled
full-model quality or serving qualification. The first recheck attempt failed
at import before comparison; corrected PYTHONPATH/source-root setup was used
for the successful runs.

## Offpath disposition

Per Rob's explicit ownership split, the following findings were filed for the
background workers: PQ#296 source-manifest closure; Tessera#387 empty implicit
export and #388 uniform-control cost guard; PB#301 ambiguous claim recovery,
#302 incomplete cleanup exception boundary, #303 invalid telemetry coverage,
#304 duplicate Netdata jobs, and #306 plugin executable/protocol mismatch.
Background workers acknowledged Astra as final reviewer/merger. PB#307 was
already merged when that worker acknowledged; root subsequently read its
production diff and actual 95-pass/four-subtest CAS output. Runtime/exporter
adoption remains a distinct acceptance step for #303.

One separate prose fix in PQ#297 repairs the launch example's readlink call
following deletion of the historical calibration symlinks. Sampling prose was
also corrected: an unweighted reconstruction screen does not establish bias
of the uniform Horvitz–Thompson estimator.

## Remaining primary acceptance

The existing mixed-LFM result supports an eager TP=1 resident sanity claim:
22 routed owners covered, 52 dense owners unqualified, production admission
refused. Its complete checkpoint and archives rehash cleanly. Tessera release
pin and serving admission were not relaxed by this review.

The recovered Tessera#376 native-operator snapshot remains unfinished
(39 CPU passes/10 failures, no delivered GPU pricing table); PQ#267 must not
invent missing resource fields. PQ#237/#275 still require informative
held-out/interpolated cost and served-quality evidence. Current sampled
campaign prices do not supply missing unsampled wires/scales by themselves.
The older #266 GPU/parity actions cited in checked-in prose were withdrawn or
failed before test execution; their absence is recorded, not called a numerical
regression. Dedicated RDMA mount timing cannot be inferred from earlier LAN
storage measurements.
