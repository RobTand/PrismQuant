# E2M1 trellis: the high-rate band, and the coding gain against an honest scalar

Status: **numeric scaffold complete and measured on the immutable GLM BF16
corpus; research evidence only, with no serving verdict.** Originated
2026-08-30; authoritative numeric closure updated 2026-08-31.

These drivers live in the repo; the corpora and result caches they read live
under `/home/rob/dq-runs/` and are not checked in. The historical control
scripts bind the frozen Stage-6 snapshot by its absolute path. The active
publication drivers instead take their manifest and output paths on the
command line and bind the repository, Git common directory, corpus, command,
container launch, and live execution identity.

## Why this exists

`hull_sweep.py` hardcodes `TCQ_TWO_TIER_RATES = (1.0 … 2.25)` with no comment
justifying a ceiling, so the published E2M1 ladder stopped at 2.5334 bpw. The
family's own bound is `mathematical_q256_bounds == (256, 1016)` -> body rate
**3.96875**, because a block may spend `bypass_rate` (4) at up to 248 of 256
positions so long as `MIN_TRELLIS_STEPS` (8) stay coded. The band from 2.25 to
the ceiling was unmeasured, and it is the band where a W4A4 trellis would have
to beat scalar NVFP4 (4.0 body + 0.5 plane = 4.5 bpw).

## What was measured, and what it overturned

**1. The band was less unmeasured than believed.** Rates 2.5/2.75/3.0 already
existed in `hull_results.menuext-20260829.json` (reproduction gate 24/24 pass).
The genuinely unmeasured band starts at 3.0.

**2. Above body rate 3.0 the schedule stops shaping, and it is forced**
(`bypass_census.py` -> `results/e2m1_bypass_census.json`). The mean rate of
shaped columns pins at exactly 3.0000 from R=3.25 up -- that is
`shaped_max_rate=3` in the family contract. Once every shaped column saturates,
the only way to spend another bit is to promote a whole column to bypass, so
the bypass fraction tracks the budget mechanically (R=3.5 -> 50.27% of columns,
R=3.75 -> 75.03%, i.e. exactly `R-3`). A bypass column carries no alphabet
(`stage6_worker.used_alphabets` drops rate 4) and IS scalar RTN E2M1. So the
high band is a forced two-point blend of "trellis at rate 3" and "RTN NVFP4 at
rate 4". Not a scheduler defect; a wire/ABI limit.

**3. Below 3.0 bypass is negligible** -- 3.45% of the bit budget at R=3.0 --
so it cannot explain anything about the low band.

**4. The "6.02 dB/bit line through NVFP4" was an unfair baseline, and the
measured scalar ladder replaces it** (`scalar_subgrid_ladder.py`). RTN onto the
exhaustively-optimal `2^R`-level subset of the E2M1 grid, same plane, same
importance, same scorer -- `P.best_level_subset` is the routine the trellis
uses to choose its own alphabet, so the two sides differ in the coder alone.
The 4-bit row reproduces scalar NVFP4 to the digit on both aggregates, which is
the control that the ladder and the incumbent are the same object at 4 bits.

| bits | DSv4 (MXFP4 src) | bf16 (Qwen3-4B dense) |
|---|---|---|
| 1 | 4.199 | 3.945 |
| 2 | 9.491 | 9.418 |
| 3 | 15.538 | 15.430 |
| 4 = NVFP4 | **23.744** | **20.525** |
| dB/bit 1->2 | +5.292 | +5.473 |
| dB/bit 2->3 | +6.047 | +6.012 |
| **dB/bit 3->4** | **+8.205** | **+5.095** |

The 1/2/3-bit rungs agree across two models and two dtypes to within 0.2 dB.
**Only the last bit diverges, by 3.1 dB.** The DSv4 corpus has 21-27 distinct
source values (MXFP4 already quantized it), so the 15-level grid nearly
interpolates it. **The 23.744 dB NVFP4 bar is corpus-inflated by ~3.2 dB and
does not transfer to a bf16-source model.**

**5. The trellis does have real coding gain; the "zero gain" reading was the
unfair baseline.** Against the measured same-rate, same-grid, same-plane
scalar, 24/24 tensors positive at every rate:

| body rate | DSv4 gain | bf16 gain (historical preliminary, 18/24) |
|---|---|---|
| 1.0 | +1.339 dB | +1.833 dB |
| 2.0 | +1.880 dB | +2.446 dB |
| 3.0 | +2.175 dB | not yet measured |

`shaped_scalar_control.py` removes the rate-shaping confound by giving the
scalar side the SAME reverse-water-fill schedule: shaping is worth only ~+0.16
dB at R=2.0, so the gain is the code. Gain RISES with rate, and is ~0.5 dB
LARGER on bf16 than on DSv4 -- a continuous density gives a trellis more to
exploit than a 27-value discrete source does.

## Scope limits

Weight-only importance-weighted SSE (W*A16). RTN render on both sides, so the
NVFP4 incumbent is a lower bound on production NVFP4 (which renders under
GPTQ+JSO). No served A/B or activation-side quality gate was run. The early
BF16 rows are Qwen3-4B dense MLP; the completed immutable corpus is GLM-5.3 Flash
dense plus selected routed-expert Linears and reports those populations
separately. The two-tier scale plane is research pricing and does not exist on
the production wire. Profiler and power records characterize the offline
drivers, not a serving kernel.

## Completed GLM campaigns

The original `/home/rob/dq-runs/glm-corpus-20260830/` remains deliberately
unloadable as `trellis.glm_corpus.v0-INCOMPLETE`. The completed campaign reads
only the immutable `trellis.bf16_corpus.v2` artifact at
`final-bf16-pread-1469b9b-v2/manifest.json`: 33 BF16 tensors, split into 9
dense and 24 routed tensors, with FP32 importance vectors. The safetensors
artifact SHA-256 is
`0d3c08aed48e8d0b540d0705c305cc3197f77c250b07dd7a07e55345f5ddd94e`;
the manifest-file SHA-256 is
`a66f800827b92383985ce205004cd2d70b63bcc5e19cada6b05a8162401ee5b0`.
Finalization copies through `os.pread`; the numeric drivers isolate their live
corpus reader from the deliberately frozen historical encoder modules.

### Legacy result identity boundary

The three superseded GLM result files in the finalized-corpus directory
predate the hardened publication contract: both E2M1 files carry
`trellis.e2m1_highrate.v2`, and the learned-FP8 file carries
`trellis.glm_fp8_learned_balanced.v1`. In substance their status is
`execution_identity_attested=false`: their retained bytes establish the
reported numeric values and hashes, but do not bind the active driver,
isolated corpus loader, integration checkout, or transitive frozen-codec
closure that actually executed. No post-hoc receipt can recover that missing
fact, so none is fabricated and the result files remain untouched. The result
sections below instead cite new, prospectively attested E2M1-v3 and
learned-FP8-v2 publications from exact clean commit
`2f955fa7c073799e494110ff81029027955ee85d`. All three bind manifest SHA-256
`a66f800827b92383985ce205004cd2d70b63bcc5e19cada6b05a8162401ee5b0`,
corpus artifact SHA-256
`0d3c08aed48e8d0b540d0705c305cc3197f77c250b07dd7a07e55345f5ddd94e`,
importance SHA-256
`dad7818dd11ea8f853bd1869f41189ca3de4a2d10deda52cfef563f63496a9dd`,
9 dense tensors, and 24 routed tensors.

New E2M1 v3 and learned-FP8 v2 outputs bind the corpus manifest, artifact
and importance hashes; active driver, isolated loader, corpus reader and Git
identity; and the frozen hull source/tree closure. The BF16 lane additionally
binds the imported `bf16_ladder.py`, `B.MANIFEST`, `B.INPUT`, and its control
result; DSv4 binds its manifest, input, and control; GLM binds its finalized
manifest, artifact, importance identity, and source commit. They use an
identity-bound exclusive claim which also reserves the result's `.partial`
namespace, closed-schema and semantic exact-prefix resume validation,
final-time identity replay, and hard-link no-replace publication with the
completed result published last. The checkpoint SHA is only a checksum; it is
not accepted as evidence that cells, metrics, footprints, or arm coverage are
valid. A structurally valid completed cell is never skipped: both drivers
regenerate it from the bound corpus through the currently pinned codec and
require exact equality of every claim, normalizing only explicitly non-claim
wall timing. An unreachable declaration is not a free-form substitute for an arm:
only the mathematical 3.96875 ceiling may carry the exact scheduler guard
refusal, and both lanes must refuse together; every lower lane/rate is required
to contain a measured arm. Resume also binds every tensor name to the logical
shape and population derived from the bound corpus, requires the complete
published-control metric domain, and re-derives wire-v1 body, alignment,
offset, alphabet, scale, byte, bpw, metric, subset, and final-summary
identities. The GLM reader opens one artifact file description for each
weight/importance pair and rechecks both tensor hashes at the point of use, so
an artifact path replacement cannot mix or silently substitute corpus bytes.
Manifest and control JSON identity is computed from the same pinned file
description whose bytes are parsed; a path swap between a parse and a later
hash cannot bind one document while using another. The scalar-NVFP4 comparator
now calls the frozen export codec directly, and the receipt binds both that
module and its NVFP4 activation-contract dependency by exact canary digest;
the mutable checkout's `format_registry.py` is not in this claim path.
Learned-FP8 v2 independently recomputes its body, row-scale, learned-book,
total-bit/byte/bpw, population-summary, and performance-gate fields. For cells
produced in the live process its closed validator requires the exact generated
reconstruction-hash domain for every arm and the exact generated-book domain
for every learned arm; missing caller evidence is a refusal, never an optional
check. It also compares the recorded reconstruction and book reports with the
encoder-returned objects. A resumed checkpoint contains
only their digests and summaries, not the reconstruction or tables themselves;
those persisted hashes alone are therefore integrity labels, not independent
proof that GPU execution occurred. Resume closes that boundary by regenerating
the reconstruction and tables and comparing the complete claim-bearing cell
before it may be reused or published.

### Same-grid coding gain

The completed matched-rate result covers every tensor. Each median below is
trellis SNR minus the exhaustive same-rate E2M1 scalar-subgrid SNR. It is a
code gain, not a comparison with full scalar NVFP4.

| population | R=1 | R=2 | R=3 |
|---|---:|---:|---:|
| dense (9) | +2.263 dB | +1.455 dB | +2.104 dB |
| routed (24) | +1.369 dB | +1.473 dB | +2.123 dB |

At R=3, the median trellis points are 17.833 dB at 3.50036 bpw for dense and
17.796 dB at 3.50213 bpw for routed weights. The corresponding full four-bit
scalar-NVFP4 medians are 20.595 dB and 20.678 dB, so the same-grid code gain
does not erase the missing body bit.

The authoritative run directory is
`/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/e2m1-scaffold-v1/`.
Its final result SHA-256 is
`d7dd0478d5e3ef3b55946f42e161f46ab0c62c1fff2fd57babb5cb11704ee332`;
the exactly recomputed coding-gain table SHA-256 is
`e4a2af3d734fa7266cc8eb790148be5a3d86b80979a14d47bf0e0fbea78d7d60`.
A no-replace V2 recomputation receipt, verified by a tracked recomputation
tool,
`analysis/coding-gain.receipt-v2.json`, has file SHA-256
`3fc1e6650367bdcb33a5170c431a815050c00d629080b00343d9ea0904c08bfc`
and internal self-digest
`4fbf36e4f4ab0636d99c668feaf6c3924c86a3e850bf131a5c490df3fb955f43`.
It binds `coding_gain` to the `scaffold` plan, source result, analysis, and
verifier closure. The earlier V1 derived receipt is retained but superseded;
it omitted this kind-to-plan binding and its numeric values are unchanged.

The in-container py-spy trace SHA-256 is
`c12cedcc550011649c76c125005a41ebedc7b9f988c55a98768bc8bf1fcd9175`.
Over the 629-second telemetry envelope, Sparky Netdata measured 24.6024 W mean,
50 W maximum, and 15.491 kJ (17.57% of the approximately 140 W envelope);
pqteld measured 25.4434 W mean, 47.96 W maximum, and 15.999 kJ. Only
`telemetry-v3/` is admissible (manifest SHA-256
`318505d2684bf64251207db8b9d80404f265e9dd662cca8f16e32898a1805772`).
The retained `telemetry/` directory is forbidden: a cross-date glob applied an
older pqteld schema header to current rows and shifted `power_draw_w` onto
temperature. `telemetry-v2/` is also forbidden: capture aborted on an older
source containing nonnumeric rows and produced no manifest. These measurements
characterize the offline driver only; they do not establish served-kernel
performance.

### The near-four-bit boundary

The explicit high plan measures body rates 3.25, 3.5, 3.75, 3.9375, and
3.96875. The best common rung is R=3.9375. The dense population median point
is 20.177 dB at 4.43785 bpw, versus a 20.595 dB population median for scalar
NVFP4; routed weights reach 20.408 dB at 4.43962 bpw, versus 20.678 dB. The
median of the per-tensor paired deficits is 0.236 dB dense and 0.299 dB routed,
with zero wins in either population. All 33 tensors reach that rung.
R=3.96875 regresses to 19.897/20.285 dB and one dense tensor is unreachable.

This is the measured form of the fixed-support boundary. Above shaped rate 3,
the current wire can only mix rate-3 trellis columns with rate-4 scalar bypass
columns. It approaches the scalar endpoint but does not create new E2M1
reconstruction support. The authoritative run is
`/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/e2m1-high-v1/`.
Its result SHA-256 is
`af0b0893401dc22c771ada019d181b2f4a09b64ea8797484543d45e877f21ac0`;
the exact near-four summary SHA-256 is
`08ae60d76d90a712cc8751ca76f13800a49834dee844a512fa7bf4eda85b0a48`.
The no-replace V2 recomputation receipt, verified by a tracked recomputation
tool,
`analysis/near-four-summary.receipt-v2.json`, has file SHA-256
`d17f8869dbeda7597148b8d60f2382edb4906ddb82444a2ee8a0af60d475a59e`
and internal self-digest
`bfe8214596bc685230f7ac657db400747614e657fe4bbf61b109611503632344`;
it binds `near_four` to the `high` plan. Its V1 predecessor is retained but
superseded for the same binding reason, without a numeric change.

The py-spy trace SHA-256 is
`865e74d0e8a34d628acd0688ff3b737f58d7b336ea40ce3024716e95fc4fa065`.
The 498-second `telemetry-v1/` envelope has manifest SHA-256
`37fac51b2d7874393e8f73d85de5e66903a339003b3008636d8c78a7961bc89c`.
Sparky Netdata measured 22.9359 W mean, 55 W maximum, and 11.436 kJ;
pqteld measured 21.0886 W mean, 39.28 W maximum, and 10.498 kJ. No other
Sparky GPU workload was observed. The concurrent learned-FP8 work on
Sparklina is peer-box context only. This clean window supports offline-driver
energy characterization, never a serving or cross-implementation speed claim.

### Learned FP8-CB scale control

The provenance-locked K32/K40/K48 fixed-vs-learned campaign completed for all
33 tensors at the production `balanced` tier. Learned-minus-fixed median SNR
gains were +0.061/+0.037/+0.051 dB for dense and +0.083/+0.073/+0.125 dB for
routed weights, with every tensor positive at every K. The authoritative run
is
`/home/rob/dq-runs/numeric-authoritative-2f955fa/sparklina/fp8-learned-v1/`.
Its final result SHA-256 is
`6f717f53dbc3401c72c6fac978b2f36a68368fe76ede13f26efb24b6e2971d6a`
and its 1,168-second py-spy trace SHA-256 is
`e6521fd7466a0894476d9d02aa27697a322ce95069c3acd9aa79f84f437cd9a3`.
The 1,174-second `telemetry-v1/` manifest SHA-256 is
`8f17d9523e45ce642cd1bfa6a8d0f74a7821b349a1a5f05101020697edb96933`.
On the active host, Sparklina Netdata measured 45.2677 W mean, 71 W maximum,
and 53.182 kJ (32.33% of the approximately 140 W envelope); pqteld measured
45.7926 W mean, 67.47 W maximum, and 53.755 kJ. No other Sparklina GPU workload
was observed; Sparky's E2M1 campaigns are peer-box context only. This is a
small, unanimous offline numeric gain, not a serving verdict.

### Completed GLM FP8-CB vs E4M3-TCQ exact-byte closure

The research-only campaign completed on Sparky for all 33 finalized GLM
tensors: 9 dense and 24 routed, kept separate with no pooled result. It tested
fixed and per-tensor learned FP8-CB K32/K40 at the production `balanced` encode
tier against TCQ_E4M3 R4/R5, both `production_row_fp32` and research-only
`two_tier` scale brackets, both frozen `lloyd` and `exact_dp` TCQ alphabets,
and both honest learned-book accounting endpoints (`wire8` and the declared
FP16 production sidecar).

| population | body cell | FP8-CB rung | strict family-coverage tests | final cell |
|---|---:|---:|---:|---|
| dense (9) | 4 | K32 | 36/36 refuse either-family coverage | `NO_VERDICT` |
| dense (9) | 5 | K40 | 36/36 refuse either-family coverage | `NO_VERDICT` |
| routed (24) | 4 | K32 | 96/96 refuse either-family coverage | `NO_VERDICT` |
| routed (24) | 5 | K40 | 96/96 refuse either-family coverage | `NO_VERDICT` |

Each count is tensor × two scale brackets × two book-price endpoints.
All 264 strict coverage tests therefore refuse a family verdict. The raw
artifact's `NO_VERDICT_exact_byte_frontiers_cross` spelling names that
predicate outcome; it is not a literal geometric description. Independent
envelope reconstruction found zero true crossings: TCQ has strictly higher
weighted SNR at every shared byte budget, while fixed FP8-CB retains the
cheapest point in a budget sliver below TCQ's minimum footprint. This is still
not TCQ family Pareto domination. The descriptive median best-quality
FP8-CB-minus-TCQ SNR deltas and median minimum-footprint offsets are shown
together below; negative `CB−TCQ bpw` means FP8-CB is cheaper.

| population | body cell | production row-FP32 (`ΔSNR`; `CB−TCQ bpw`) | research two-tier (`ΔSNR`; `CB−TCQ bpw`) |
|---|---:|---:|---:|
| dense | 4 | -1.509895 dB; -0.000432 bpw | -3.477842 dB; -0.273869 bpw |
| dense | 5 | -1.948798 dB; -0.000429 bpw | -2.858964 dB; -0.273866 bpw |
| routed | 4 | -1.283757 dB; -0.002217 bpw | -3.408104 dB; -0.275655 bpw |
| routed | 5 | -1.787727 dB; -0.002324 bpw | -2.785978 dB; -0.275762 bpw |

Across individual tensors the production-bracket sliver is 0.0002–0.0023 bpw
and is only 277 bytes at its narrowest; the research two-tier bracket makes it
roughly 0.25–0.28 bpw. In this result, book price changes bytes but not those
quality deltas or any verdict. The strict
frontier theorem was independently audited in
`/home/rob/dq-runs/deliberation/numeric_closure_fable_2026-08-31.md`
(SHA-256
`36cffef74cea2a89f2784ba9bd94c8e1c37e502b6bc437500aad333491cce719`).
The report's contemporaneous campaign-unrun and branch-status prose is
superseded by the completed artifact and receipt above; its theorem-level
frontier analysis remains the cited result.
Its valid conclusion, now applied to the completed artifact, is **no
objective-free family replacement on this artifact-exact single draw**. That
does not mean equality, absence of a useful budget-specific choice, or a
model-general result. Learned-book endpoint conjunction is uniform over
prices between the endpoints; the two scale brackets establish only
two-doctrine robustness. Strict quality comparisons remain one-ULP sensitive.

The authoritative directory is
`/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/fp8-cb-tcq-v1/`.
The receipt and result are no-replace evidence files, not git-tracked files;
their verifier is tracked. The complete 31-file directory is also mirrored
byte-for-byte on Sparklina at the same absolute path. Its sorted file-hash
manifest has SHA-256
`4ac1df35ecccaa8cb203e92bf583b8cefec940a104df9df1d637231efa16525c`,
stored beside the mirror as
`fp8-cb-tcq-v1.sparklina-mirror-tree.sha256`.
The result is final, nonpartial, `tensors_done=33`, and has status
`measurement_complete_no_serving_verdict`; its SHA-256 is
`3e37ebeb24f575d802461c14c8d844fe5ba0fd7029a0a3f0b2bfe4d2a9befe18`;
its checkpoint self-digest is
`c1d6699de67bcac88b084cd10b0dc3cc9b418aa0568febf8612e4c7c034a86ff`
and settings identity is
`913e070ff6bd74506bc0156f9105ed453fa2af86cb0bdbc39d6efcd675fd5712`.
The no-replace
`analysis/fp8-cb-tcq.analysis-receipt-v1.json` has file SHA-256
`f8d58799363bbb36f0b57d8c637a162931511015bf6604f87d352f6254890ec2`
and internal self-digest
`804bd192bdd41672c82fda3d772e88bbc480543cd4f8b545c0562b40cc530dc4`.
Its tracked verifier is commit
`511385b5c916ee12f06771a68fb302d60e5e161a`, file SHA-256
`3b28992cdf91893cc6de54d73ac9492b3935c002297debf760ccb8fa59cece2f`.
It independently pins the exact result, corpus, sources, runtime versions,
container/attestation, canonical integer costs, metric identities, and all
frontiers before rebuilding the receipt.

The execution used exact producer commit
`2f955fa7c073799e494110ff81029027955ee85d` in container
`8d5a2b79f98d24f10635bbe6fa7c339a8361377f18926f281ad4e2503d4bac89`.
Docker and the container both exited 0, with `OOMKilled=false`; final Docker
inspect SHA-256 is
`bfa6e728e8805d2aa6efdef2e1710100b8dc85fa14db2d7b90507652e84dbc95`.
The canonical launch-attestation file SHA-256 is
`f495263965cba1a6b63d05c86fb328b8f488b3337d78a055090f6af9a84c6ee1`
and its internal identity is
`1cfad16122829dd0f3554a77d5b630a5005af78a57b706eca886bd756d8c510d`.
The algorithm interval was 2,181.061 seconds. The valid py-spy speedscope
profile SHA-256 is
`b13df57960f0042f2232f1c36ded7c5e6c2ba060be7d5d3bec674d8b07ef3c3f`;
it contains 250,583 samples, including 219,697 main-thread samples over
2,196.97 seconds.

The exact 2,208-second dual-host envelope is published only as
`telemetry-v1/`, whose manifest SHA-256 is
`a9c1e01bd12c11bf990c46db9fcd8c392be1edcaffdf188652ae86f0438644aa`.
Sparky Netdata measured 23.0511 W mean, 48.9 W p95, 74 W maximum, and
50.914 kJ (16.47% of the approximately 140 W envelope); schema-aware pqteld
measured 22.8822 W mean, 48.39 W p95, 72.97 W maximum, and 50.519 kJ. The
midnight-spanning pqteld capture parsed each source under its own 42-column
header and required exact cross-source schema equality. Sparklina's peer-box
series measured 6.4611 W mean/14.268 kJ in Netdata and 6.8662 W
mean/15.158 kJ in pqteld; it is context only and is not attributed to this
campaign. No failed or schema-invalid CB/TCQ telemetry directory is
admissible. These are offline box measurements, not process-exclusive energy,
served-kernel throughput, saturation, work per joule, or a
cross-implementation performance comparison.

#### Campaign design and exact-byte verdict contract

The driver imports the existing frozen `fp8_ladder.py` arms and accountants;
it does not reimplement either codec. It additionally pins the exact-DP source
in the frozen Stage-0 snapshot. Every completed checkpoint cell is regenerated
from the hash-checked corpus and compared in full before reuse; only wall
timing is excluded from replay. A persistent identity-bound claim reserves
both final and partial names, publication is no-replace, and the corpus,
active sources, frozen codec closure, exact command, live CUDA identity,
expected host, and immutable container digest are rechecked before the result
publishes. Every checkpoint carries an append-only
`trellis.numeric_execution_segment.v1` history.

The result constructs exact-byte Pareto frontiers. A cell is awarded only if
the same family strictly covers the opposing frontier on every tensor under
all four scale-plane × learned-book-price combinations. A crossing frontier,
mixed tensors, or bracket disagreement produces structured `NO_VERDICT`.
This is deliberately stricter than comparing round nominal labels.

#### Exact numeric launch and attestation contract

`--expected-host` and `--container-identity` are not driver arguments. The
host creates a stopped container, then
`numeric_execution_contract.py --container ... --physical-host ... --output
...` inspects it before it may start. The helper resolves a full immutable
container ID, performs two matching Docker-inspect and rootfs-diff reads, and
creates a canonical `trellis.numeric_launch_attestation.v1` JSON file with
exclusive no-clobber creation and file-plus-directory fsync. The attestation
parent is mounted read-only into the container at the identical absolute path.
At publication, the process rereads those exact canonical bytes, joins them to
the live container-ID hostname, UID/GID/groups, UTS hostname, GPU UUID/name,
command, and environment, then rehashes every tracked HEAD blob and rejects
untracked or ignored executable/import code. The durable stable projection is
`trellis.numeric_execution.v2`; the per-container linkage is the execution
segment described above. This is Docker-daemon evidence on a trusted host,
not a cryptographic defense against a hostile root operator or Docker daemon.

The image reference in Docker `Config.Image` is exactly:

```text
eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
```

`HULL_CONTAINER_IMAGE` is the digest portion of that reference, not Docker's
local image-object ID. The latter is checked separately and is host-specific:

| physical-host key | required UTS hostname | required GPU UUID | allowed local Docker image ID |
|---|---|---|---|
| `sparky` | `sparky` | `GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4` | `sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869` |
| `sparklina` | `gx10-6b77` | `GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705` | `sha256:ac631d27c1514ec3f838299d424c98892a0ba854fa642002df4c8f576bbfe9fa` |

This is the reference fresh-container CB/TCQ preflight. Run it on the selected
host from a clean committed checkout for publication parity. All source and
destination mount paths must be identical. Use a new container name,
attestation path, and result path for every attempt; the contracts deliberately
do not overwrite an old claim.

```bash
HOST=sparky                         # or sparklina, on that physical host
REPO=/absolute/path/to/clean-committed-prismaquant-worktree
STUDY="$REPO/research/trellis_e2m1_highrate_2026-08-30"
GIT_COMMON=$(/usr/bin/git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
STORAGE=/home/rob/dq-runs
MANIFEST="$STORAGE/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/manifest.json"
RUN_ID="preflight-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEST_DIR="$STORAGE/numeric-launches/$RUN_ID"
ATTEST="$ATTEST_DIR/launch-attestation.json"
OUT="$STORAGE/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/$RUN_ID.json"
NAME="pq-fp8-cb-tcq-$HOST-$RUN_ID"
IMAGE_REF='eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869'
IMAGE_DIGEST='sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869'

/usr/bin/install -d -m 0700 "$ATTEST_DIR"
CID=$(/usr/bin/docker create --pull=never --name "$NAME" \
  --user 1000:1000 --uts host --network none --ipc private \
  --cgroupns private --cap-drop ALL \
  --security-opt no-new-privileges:true --read-only --gpus all \
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g,uid=1000,gid=1000,mode=0700 \
  --workdir "$REPO" \
  --env "HULL_PHYSICAL_HOST=$HOST" \
  --env "HULL_CONTAINER_IMAGE=$IMAGE_DIGEST" \
  --env "HULL_LAUNCH_ATTESTATION=$ATTEST" \
  --env "HULL_REPO_ROOT=$REPO" \
  --env "HULL_GIT_COMMON_DIR=$GIT_COMMON" \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env CUDA_CACHE_PATH=/tmp/cuda-cache \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache --env TMPDIR=/tmp \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache \
  --env TRITON_CACHE_DIR=/tmp/triton-cache \
  --env XDG_CACHE_HOME=/tmp/cache \
  --env HF_DATASETS_OFFLINE=1 \
  --env PRISMAQUANT_NVFP4_SCALE_RULE=static_6 \
  --env PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING=0 \
  --mount "type=bind,src=$ATTEST_DIR,dst=$ATTEST_DIR,readonly" \
  --mount "type=bind,src=$REPO,dst=$REPO,readonly" \
  --mount "type=bind,src=$GIT_COMMON,dst=$GIT_COMMON,readonly" \
  --mount "type=bind,src=$STORAGE,dst=$STORAGE" \
  "$IMAGE_REF" \
  /usr/bin/python3 -B "$STUDY/fp8_cb_tcq_glm.py" \
    --manifest "$MANIFEST" --out "$OUT" --preflight-only)

/usr/bin/python3 -B "$STUDY/numeric_execution_contract.py" \
  --container "$CID" --physical-host "$HOST" --output "$ATTEST"
/usr/bin/docker start --attach "$CID"
```

The host helper accepts only that closed launcher shape: UID/GID
`1000:1000`, host UTS, private PID/cgroup/IPC, no network, all capabilities
dropped, no-new-privileges, a read-only rootfs, the exact `/tmp` tmpfs, the
exact four binds above, the pinned NVIDIA entrypoint, and the pinned image's
complete environment plus only the five `HULL_*` and twelve fixed
Python/scratch/study fields shown. The frozen ladder imports otherwise add or
overwrite the five newly explicit cache/study fields; pinning them before
Python starts prevents an import from expanding or changing the attested
environment, and both writable caches remain on the ephemeral `/tmp` tmpfs.
Extra mounts, devices, groups, capabilities, security modes, or environment
keys are refusals. In particular, additional source-code, `/usr/bin`, CUDA,
site-packages, and dynamic-loader overlays cannot be added.

Direct `/usr/bin/python3 -B` is accepted only for
`fp8_cb_tcq_glm.py --preflight-only` and `fp8_learned_glm.py --dry-run`.
Every publication-capable invocation of `e2m1_highrate.py`,
`fp8_learned_glm.py`, or `fp8_cb_tcq_glm.py` must replace the direct command
above with this exact tracked in-container supervisor and omit the preflight
flag:

```bash
/usr/bin/python3 -B "$STUDY/numeric_profiled_launcher.py" \
  --profile <fresh-path-below-/home/rob/dq-runs> -- \
  /usr/bin/python3 -B <absolute-approved-driver> \
  <that-driver's-arguments>
```

The supervisor starts the pinned in-image `py-spy record` command itself. It
refuses a pre-existing or symlinked profile/result path and reports success
only when `py-spy` succeeds and both the new nonempty speedscope profile and
the driver's new nonempty `--out` commit marker exist. This is load-bearing:
the image's `py-spy` can return zero after a child failure, whereas a numeric
driver publishes `--out` only after its own final validation.

Do not wrap `docker run` or `docker start` in host-side `nsys profile`: that
profiles the short-lived Docker client rather than the daemon-owned CUDA
process and can yield an empty CUDA trace. The current accepted path is
in-container py-spy plus time-aligned Netdata and pqteld evidence. Adding nsys
later requires an explicit image/command-contract revision and an in-container
launch; it is not an unrecorded wrapper around this schema.

#### Verified no-workload preflight evidence

On 2026-08-30 a newly created container passed the finalized host inspection
and the CB/TCQ direct preflight on each machine:

| host | inspected local image ID | helper / container result |
|---|---|---|
| Sparky | `sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869` | attestation accepted; exit 0 |
| Sparklina | `sha256:ac631d27c1514ec3f838299d424c98892a0ba854fa642002df4c8f576bbfe9fa` | attestation accepted; exit 0 |

Both driver payloads reported schema
`trellis.glm_fp8_cb_tcq_two_bracket.v1`, 9 dense and 24 routed tensors,
`status=validated_no_gpu_no_write`, `publication_capable=false`, and
`publication_receipt=null`. Sparklina mounted the current integration
worktree read-only. Sparky mounted the clean immutable checkout
`e48e88b5fd2fd2a6c94cb544b3760ffc1b19d0c5`; the finalized host-helper bytes
were supplied to that host for the pre-start inspection. These checks validate
the closed Docker configuration and the nonpublishing parser/import path.
They did not execute the encoder, enter the CUDA publication contract, or
establish any quality, speed, power, or serving result themselves. The
subsequent completed campaign documented above supersedes only the former
unrun study status; it does not turn either preflight into measurement.

Even a later completed result remains W\*A16 weighted-SSE research evidence:
it cannot qualify W8A8 activation behavior, Gridbook load, KL/PPL, serving
speed, residency, or work per joule.

## Known reachability limit

Body rate **3.96875 is not universally reachable**. At 1016 q256 every 256-block
must keep exactly 8 coded positions, leaving the global reverse-water-fill zero
slack to move a bit between blocks; on `layers.12.ffn.experts.82.w2.weight` the
arm-D schedule refuses with `cannot rebalance trellis-length guard`. **3.9375
(q256 1008) builds on all 24 tensors on both arms** and is the honest top rung.
Both the census and the driver record a refusal per (tensor, rate) rather than
crashing.

## Reproduce

Run repo drivers from the repository, not from the frozen snapshot directory;
the control scripts import that snapshot themselves. These are the complete
current CPU/offline argument vectors:

```bash
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
REPO=/absolute/path/to/prismaquant-checkout
STUDY="$REPO/research/trellis_e2m1_highrate_2026-08-30"
cd "$REPO"

"$PY" "$STUDY/bypass_census.py"
"$PY" "$STUDY/scalar_subgrid_ladder.py" --corpus dsv4
"$PY" "$STUDY/scalar_subgrid_ladder.py" --corpus bf16
"$PY" "$STUDY/shaped_scalar_control.py"
"$PY" "$STUDY/fp8_learned_glm.py" \
  --manifest <final-v2.json> --out <glm-fp8-learned.json> --dry-run
"$PY" "$STUDY/fp8_cb_tcq_glm.py" \
  --manifest <final-v2.json> --out <glm-fp8-cb-tcq.json> --preflight-only
"$PY" "$STUDY/coding_gain_table.py" --rows <rows.json> \
  --out <coding-gain-table.json>
```

The first four commands and `coding_gain_table.py` are offline derivations;
the two GLM validation modes explicitly do no GPU work and write no result.
For a publication run, place one of the following complete Python command
suffixes after the tracked supervisor's `--` in the attested container recipe
above:

```bash
/usr/bin/python3 -B "$STUDY/e2m1_highrate.py" \
  --corpus dsv4 --out <rows.json>
/usr/bin/python3 -B "$STUDY/e2m1_highrate.py" \
  --corpus glm --glm-manifest <final-v2.json> --out <glm-rows.json>
/usr/bin/python3 -B "$STUDY/e2m1_highrate.py" \
  --corpus glm --glm-manifest <final-v2.json> --glm-rate-plan high \
  --out <glm-high-rate-rows.json>
/usr/bin/python3 -B "$STUDY/fp8_learned_glm.py" \
  --manifest <final-v2.json> --out <glm-fp8-learned.json>
/usr/bin/python3 -B "$STUDY/fp8_cb_tcq_glm.py" \
  --manifest <final-v2.json> --out <glm-fp8-cb-tcq.json>
```

Those lines show driver argv only; invoking them directly is not a valid
publication launch. All three GPU-capable drivers require the host attestation
and tracked in-container profiling supervisor. `e2m1_highrate.py` additionally
reproduces published control rungs before trusting new rows and refuses on a
mismatch.
