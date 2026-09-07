# Canonical campaign capture reuse — 2026-09-07

Issue: https://github.com/RobTand/prismaquant/issues/305

The campaign can capture the whole census once, store exact float32 scoring
prefixes and uncapped Hessians through the existing activation-cache writer,
and reuse selected units in each anchor or materialization quantum. The
existing cost-stage journal seals per-unit file receipts. The manifest is
`prismaquant.tessera_calibration_cache.v2`, with a complete unit roster and
identity bound to canonical checkpoint initialization, actual attention
backend, Torch/CUDA/Transformers runtime, census, complete source checkpoint
bytes, exact calibration draw and per-unit geometry.

The public flow is `dispatch_tessera_campaign capture`, followed by
`plan --calibration-cache <capture_manifest.json>`. Each row carries the exact
manifest SHA256 and verifies selected artifacts before making X/H resident.
Missing, stale or incomplete captures fail closed. Selected-wire materialization
derives its cache and hash from the priced cost provenance and loads the same
explicit attention backend. Merging requires every row to agree on the capture.
The optional packed-collector boundary consumer preserves original positional
and keyword arguments before dtype conversion or sampling, with original
sample/token coordinates from the calibration loop. It does not add a second
forward or alter default reservoir sampling.

CPU and compile validation passed through PrismaBuild action
`b2b5bac37cc59636019d06897ae99f6b945d3a467535f74b80e8d71e89c8beee` on dl380g10:
**176 passed, no skips**, exit 0, 70.58 seconds. Eight pytest workers used the
scoped `pq-cpu312` environment with pinned Tessera producer imports and one
OMP/MKL/OpenBLAS thread each; aggregate demand was eight CPUs and 20 GiB.
Compile checks covered all touched production modules and measurement harnesses.
The tests cover exact X/H precision, selective prefetch, source/draw/scope/
geometry/artifact drift, legacy initialization refusal, complete-journal
resume, CLI reuse without a second forward, automatic materializer binding,
raw argument identity, keyword routes, calibration coordinates, unchanged
reservoir sampling, and native input contracts. Actual inspected CAS payload:
`/mnt/shared/prismabuild-fleet/cas/blobs/46/4605bb918e52c1ed53800a40813b9b852d2500fabba8ee48c2a6726c9e2bd119`.

The canonical full-model census is
`/mnt/shared/tessera-measurements/first-model-20260907/canonical-census-512/census.json`,
SHA256 `62d41825f84edd280de46bbb89893676fae8203563834dc95d8ad29c05bad04d`.
The draw is 512 samples of 512 tokens; its int32 token digest is
`ed77a9890e02ae0b6bac2cbaa8fab4dd27617d7b57635976b689badaf8d5a8e4`.
The prior no-op-initializer capture remains quarantined under
`capture-reuse/baseline/INVALID_FOR_CALIBRATION_REUSE.json`; no quality or
reuse claim derives from those X/H bytes. Canonical full capture and profiled
ABBA reuse measurements are tracked below as separate evidence, not inferred
from the CPU suite.

Initial validation attempts `a8a78fab56ed`, `c30ca448df8a`, and `791a51c9c833`
used the PB infrastructure Python instead of the scoped project Python and
failed on missing Torch or compressed-tensors. The corrected project-runtime
run `3f5a504cf1d3` passed 133 tests and exposed two synthetic producer-hash
fixtures. Those fixtures now declare their actual source hashes; the final
176-test run above passed. No runtime input guard was weakened.

The first canonical GPU attempt `35ab901ece41` stopped before writing a routed
boundary: the native helper called the LFM profile's `name` property as a
method. The owner is fixing the three affected sites with a real-profile
regression. Failed-attempt artifacts are retained as bounded diagnosis under
`capture-reuse/canonical-helper-failed-35ab901ece41`; the worker exit was 1 and
the broker's separate termination receipt confirms its scope stopped. The
current fleet terminal JSON omits that cleanup field, so the separate receipt
was inspected. This attempt provides no completed calibration or reuse result.

## Completed canonical capture

The native helper owner fixed the profile-property error, and replacement PB
`13a85577915ce5349749e66bbab9b94e3d896f976e2689442a1e459d96eea4a6`
completed on Sparklina with four assigned CPUs, 104 GiB aggregate memory and
GPU measurement admission with no existing admitted members. Exit status, CAS stdout payload
`ffd4a8a11b2ce6981c4456410b9573a79cc114ced07505bebac1bb1ad38038ed`, and the
separate broker termination receipt were inspected. The known-good image was
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`,
with Torch 2.13.0+cu130, CUDA 13.0, canonical checkpoint initialization and
explicit eager attention. All 2,142 units (30 dense projections and 2,112 expert
projections) were persisted and sealed; the files total 41,938,210,454 bytes.
The producer verified every entry; independent inspection checked the complete
roster, file presence, three representative artifact hashes and the boundary
hash. Each reuse quantum additionally rehashes its selected files.

Evidence root:
`/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/canonical/`.
The completed `capture_manifest.json` SHA256 is
`db3cd996ee8a3ac82d62c6e7e2f23cdb995874b831adcf9360acdae682654823`.
`verified-receipt.json` links the terminal, CAS, token, profile and boundary
receipts. The token safetensors file SHA256 is
`a38312e3b1eeecc2a4363d2a91739ba535388036d4910bffa815d451dcb9a940`.
The original layer-2 routed boundary is retained at
`native-boundary/model__layers__2__feed_forward__experts.pt`, SHA256
`1290dd0dcb4ebecd09aee9fd2427aae62cb9c09974a466c05dcbe1ff08c6874d`;
it preserves BF16 inputs/weights, int64 route IDs and sample/token coordinates,
and the actual float32 expert bias. Its separate consumer qualification is
outside this capture-reuse change.

The full collector took **553.835 seconds under py-spy**, with 10,289 samples;
its profile SHA256 is
`63f6e8a13ddf818819407e7baeebca8275e562e7ede42cc238233622c45df999`.
Peak CUDA allocated/reserved bytes were 50,762,741,760 / 52,204,404,736;
process peak RSS was 48,187,324 KiB. Sampled Python leaves included grouped
matmul (4,446 samples) and per-expert route derivation (1,363); these are sample
locations, not CUDA kernel timings. Setup, persistence (197.662 seconds) and
sealing (46.793 seconds) have wall timers only, despite the harness's generic
`kind=profile` phase label. No in-process profile is claimed for those phases.

`netdata-both-hosts.json` retains raw series from both machines. During the
collector, Sparklina sampled GPU power averaged 33.980 W, with a linearly
interpolated integral of 18,819.128 J at 10-second sampling. Host CPU user/system
means were 4.810% / 1.728%; CPU and RAM series bracket the phase without gaps.
This power is about 24% of the GB10's approximate 140 W envelope; the run does
not establish GPU saturation. Sparky averaged 4.163 W, but its CPU/RAM
integrals were refused because the series had gaps. These are absolute capture
measurements, not a before/after delta against the rejected initializer.

## Clean delivery validation

The four implementation commits were replayed onto current main after its
canonical initializer fix, preserving separate finalization, maxima, census
and reuse changes. PB
`ace64b6c34994bebe6f08e95eae145ef3801430e9add66de1b0cafd8900c7e76`
checked the clean delivery head with compile checks and **137 passing tests,
no skips**, exit 0, 77.26 seconds. Inspected CAS payload:
`a304b329bb7eb58200672bb26a9826ff5bbdb0e16b3992ce023fabd7919d0ce5`.
The five touched production modules match the earlier integration-tested
source byte for byte. The native helper is a separate committed dependency of
the full-capture measurement harness and is not included in this PR.

## Matched profiled reuse comparison

PB `0c4b585e8f43472a085435628bff68955d8bd6b546da2fe39a583dcb8d6ed821`
completed all seven arms with exit 0. The actual CAS payload was rehashed and
read at
`/mnt/shared/prismabuild-fleet/cas/blobs/84/84b67475c6831a43c7b03f404e1827e81e051539ff87f0186d9ece2d46e7d349`;
the broker's separate termination receipt reports `result.ok=true`.
The evidence root is
`/mnt/shared/tessera-measurements/first-model-20260907/capture-reuse/canonical-compare/`.
`verified-receipt.json` binds terminal/CAS, all seven profiles and derived
results; `receipt.json`, `derived-metrics.json`, `netdata-both-hosts.json` and
`frozen-source/` retain timings, raw telemetry, analysis code, exact harness
sources, hashes and the maxima-only source diff. The PB campaign input is
`frozen-source/canonical-compare-campaign.json`; its admitted command is
`python3 -m tools.run_campaign_capture_baseline --module tools.campaign_capture_compare`.

The workload selects the 96 projections of
`model.layers.10.feed_forward.experts` from the same canonical 512-by-512 draw
and full-model capture above. Source model, explicit eager attention, tokens,
unit roster, full Hessians, scoring-prefix policy, device and container match.
The model and tokens load once before all arms. Every forward arm completes the
same full-model draw, collecting only those 96 units. Every cached arm rehashes
source identity and selected capture artifacts, validates them and makes X/H
resident. Both methods stop at the same resident-ready boundary; separate
comparison/encoding and shared initial model/token setup are excluded.
All seven arms include py-spy at 20 Hz. PB assigned Sparklina CPUs 5–8,
50 GiB aggregate memory and measurement admission with no existing admitted
GPU members. Native thread limits remained four. This is a preparation-stage
comparison, not an end-to-end encoding or served-throughput benchmark.

| Arm | Profiled seconds | Mean sampled GPU W | Interpolated GPU J |
| --- | ---: | ---: | ---: |
| 0 forward, device maxima | 103.144 | 31.480 | 3,247.026 |
| 1 cached | 12.348 | 16.624 | 205.282 |
| 2 cached | 12.315 | 15.747 | 193.915 |
| 3 forward, device maxima | 101.423 | 31.821 | 3,227.401 |
| 4 forward, scalar-sync maxima | 102.406 | 40.227 | 4,119.456 |
| 5 forward, scalar-sync maxima | 102.715 | 38.894 | 3,994.973 |
| 6 forward, device maxima | 100.154 | 36.938 | 3,699.434 |

Reuse uses arms 0/1/2/3 in ABBA order. Median preparation time changes from
**102.284 to 12.331 seconds: 8.295× throughput, 87.94% less elapsed time** on
this selected scope. Median interpolated GPU energy changes from 3,237.213 to
199.598 J, corresponding to 0.02966 versus 0.48097 selected units per GPU
joule, a **16.22× sampled work-per-joule ratio**. GPU power is sampled every ten
seconds, and each cached arm lasts only about twelve seconds; these integrals
are coarse telemetry estimates, not precision energy measurements or whole-
machine energy. Two repetitions on one scope/model/device do not establish a
confidence interval or a general speedup across workloads. Persistence and the
one-time full capture cost must also be amortized across reuse quanta.

The in-process profiles explain the change. Forward arms 0/3 have 2,065/2,076
samples; deepest Python grouped-matmul frames account for 1,353/1,382. Cached
arms have 248/252 samples, of which 219/229 fall in `hashlib.file_digest`.
The cached path has eliminated the repeated model forward, but its remaining
preparation cost is dominated by content hashing and validation; GPU saturation
is not established. This identifies remaining CPU preparation overhead without
weakening source-content checks or inferring a further unmeasured speedup.

Both hosts' CPU, RAM, I/O and GPU power series bracket all comparison phases
without refused integrals. Sparklina host CPU user means range 4.568–4.936%,
system 1.336–1.501%, and process peak RSS stays at 17,500,120 KiB. Peak CUDA
allocated memory stays below 19.0 billion bytes and reserved below 19.1 billion
bytes. Sparky's simultaneous sampled GPU power means range 4.0–13.011 W;
its activity is preserved as external context and excluded from this GPU-energy
ratio. Sparklina's 15.747–40.227 W range is about 11–29% of the approximate
140 W GB10 envelope. GPU utilization percentages are not used to infer
saturation or rank implementations.

Every arm passed exact equality of all 96 scoring-input tensors, full Hessians,
counts and maxima against the canonical per-unit files. Forward arm 0 and
cached arm 1 also encoded expert 0's w1/w2/w3 using
`TESSERA_E4M3_K1_R1024`; all three wire hashes, sizes and scored `dloss` values
match exactly. The six output wire files were independently rehashed.
The dloss values are 0.00011898282848830734, 0.0000005687553539246437 and
0.00009622986960623945 respectively. This validates identical bytes and scored
loss for those encodes; it adds no served-runtime, model-level KL or format-
quality claim.

The first comparison attempt `f3c07fbaa5e4` completed one 99.708-second forward
and exact tensor comparisons, then failed because its measurement harness had
not created the PWC output directory before encoding. The corrected harness
creates that directory and records failed phase status explicitly. The failed
attempt remains under `canonical-compare-harness-failed-f3c07fbaa5e4`; none of
its timings enter the completed ABBA statistics.
