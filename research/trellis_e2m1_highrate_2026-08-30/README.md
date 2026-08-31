# E2M1 trellis: the high-rate band, and the coding gain against an honest scalar

Status: **numeric scaffold complete and measured on the immutable GLM BF16
corpus; research evidence only, with no serving verdict.** 2026-08-30.

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

The three immutable GLM result files cited below predate the hardened
publication contract: both E2M1 files carry
`trellis.e2m1_highrate.v2`, and the learned-FP8 file carries
`trellis.glm_fp8_learned_balanced.v1`. In substance their status is
`execution_identity_attested=false`: their retained bytes establish the
reported numeric values and hashes, but do not bind the active driver,
isolated corpus loader, integration checkout, or transitive frozen-codec
closure that actually executed. No post-hoc receipt can recover that missing
fact, so none is fabricated and the result files remain untouched.

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

The result receipt is `glm-e2m1-highrate.json` (SHA-256
`097b1863ff6c1ba27cecc0fb897e825bc360edb8d09e9b3596951c9bc5730bcd`),
and its derived table is `glm-e2m1-coding-gain.json` (SHA-256
`8b873d2bcc16a8da311591eb1f0b340ed8f4addc65acd3275283b2db299e3c79`).
The 258.4-second run has an in-process py-spy speedscope trace (SHA-256
`edc6bc486132703ca47e9da9f374fb82986144350623d291d9043addde165450`)
and aligned Netdata evidence from both boxes. Sparky's Netdata power mean was
37.85 W, 27.0% of the approximately 140 W envelope, with a 63 W maximum;
its independent pqteld series measured 39.87 W mean and 11.00 kJ. These
measurements characterize the offline driver only and do not establish served
kernel performance.

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
reconstruction support. The result is `glm-e2m1-near4.json` (SHA-256
`d9356eee75f94c07fe11cfdab4f70a72357e2092a7e6846677aec0668211e3cd`)
with py-spy trace SHA-256
`f4c879e1ba040adf364c9c0cfd03cecdf5793d9dcf8d89fa64bc487bb38ff645`.
Its first roughly 20 seconds overlapped the rejected QTIP r2 attempt because
the work queue admitted both. Quality is deterministic, but this window must
not be used for a performance or energy claim.

### Learned FP8-CB scale control

The provenance-locked K32/K40/K48 fixed-vs-learned campaign completed for all
33 tensors at the production `balanced` tier. Learned-minus-fixed median SNR
gains were +0.061/+0.037/+0.051 dB for dense and +0.083/+0.073/+0.125 dB for
routed weights, with every tensor positive at every K. The result is
`glm-fp8-learned-balanced.json` (SHA-256
`3437be1473e1a880d5a6d338bc93f85902ea661be982d040681834426baf1a94`)
and the 1,125.8-second algorithm trace has SHA-256
`f7ef868f00d55794b7f16f34d09880f682f5bd63991a3e6880764172820662ad`.
Sparky Netdata measured 55.93 W mean and 82 W maximum (40.0% of the 140 W
envelope); pqteld measured 57.21 W mean and 65.73 kJ. This is a small,
unanimous offline numeric gain, not a serving verdict.

### Hardened GLM FP8-CB vs E4M3-TCQ closure scaffold

`fp8_cb_tcq_glm.py` is the unrun, research-only campaign that closes the
handover's exact remaining 4.0/5.0 numeric question without combining old
results across corpora or receipts.  One immutable result must cover all 33
finalized GLM tensors and contains, for nominal body cells 4 and 5:

- fixed and per-tensor learned FP8-CB K32/K40 at the production `balanced`
  encode tier;
- TCQ_E4M3 R4/R5 under both `production_row_fp32` and `two_tier` scale-plane
  brackets;
- both frozen `lloyd` and `exact_dp` alphabet selectors; and
- both honest learned-book prices: one legal E4M3 byte per element and the
  FP16 sidecar the production footprint currently declares.

The driver imports the existing frozen `fp8_ladder.py` arms and accountants;
it does not reimplement either codec. It additionally pins the exact-DP
source in the frozen Stage-0 snapshot, because that source is a separate file
and is not present beside `fp8_ladder.py`.  Every completed checkpoint cell is
regenerated from the hash-checked corpus and compared in full before reuse.
Only wall timing is excluded from replay.  A persistent identity-bound claim
reserves both final and partial names, the final receipt is no-replace, and
the corpus, active sources, frozen codec closure, exact command, live CUDA
identity, expected host, and declared immutable container digest are rechecked
before the result publishes. Every completed partial and final carries an
append-only `trellis.numeric_execution_segment.v1` history. A restart in a
fresh, correctly attested container on the same host appends a segment without
making the container ID part of the stable checkpoint identity; a host or
host-specific image-ID change alters that stable identity and is refused.

The result derives two separate population tables and has no pooled field.
For each population/cell it constructs exact-byte Pareto frontiers rather than
calling a higher-SNR point at more bytes a win.  A cell is awarded only if the
same family strictly covers the opposing frontier on every tensor under all
four scale-plane × learned-book-price combinations.  A crossing frontier,
mixed tensors, or bracket disagreement produces a structured `NO_VERDICT`.
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
  --env CUDA_CACHE_PATH=/tmp/cuda-cache --env TMPDIR=/tmp \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
  --env TRITON_CACHE_DIR=/tmp/triton-cache \
  --env XDG_CACHE_HOME=/tmp/cache \
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
complete environment plus only the five `HULL_*` and seven Python/scratch
fields shown. Extra mounts, devices, groups, capabilities, security modes, or
environment keys are refusals. In particular, additional source-code,
`/usr/bin`, CUDA, site-packages, and dynamic-loader overlays cannot be added.

Direct `/usr/bin/python3 -B` is accepted only for
`fp8_cb_tcq_glm.py --preflight-only` and `fp8_learned_glm.py --dry-run`.
Every publication-capable invocation of `e2m1_highrate.py`,
`fp8_learned_glm.py`, or `fp8_cb_tcq_glm.py` must replace the direct command
above with this exact in-container profiler prefix and omit the preflight flag:

```bash
/usr/local/bin/py-spy record --output <path-below-/home/rob/dq-runs> \
  --format speedscope -- /usr/bin/python3 -B <absolute-approved-driver> \
  <that-driver's-arguments>
```

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
They do not execute the encoder, enter the CUDA publication contract, prove a
clean future integration commit, or establish any quality, speed, power, or
serving result. No numeric GPU campaign was run.

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
suffixes after py-spy's `--` in the attested container recipe above:

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
and in-container py-spy command. `e2m1_highrate.py` additionally reproduces
published control rungs before trusting new rows and refuses on a mismatch.
