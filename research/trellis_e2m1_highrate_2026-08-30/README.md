# E2M1 trellis: the high-rate band, and the coding gain against an honest scalar

Status: **numeric scaffold complete and measured on the immutable GLM BF16
corpus; research evidence only, with no serving verdict.** 2026-08-30.

These drivers live in the repo; the corpora and result caches they read live
under `/home/rob/dq-runs/` and are not checked in. Paths are absolute inside
each script because the study directory is a sibling of the frozen Stage-6
encoder snapshot they must bind to.

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

Future E2M1 v3 and learned-FP8 v2 outputs bind the corpus manifest, artifact
and importance hashes; active driver, isolated loader, corpus reader and Git
identity; and the frozen hull source/tree closure. They use an identity-bound
exclusive claim, self-digested exact-prefix resume, final-time identity replay,
and hard-link no-replace publication with the completed result published last.

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

## Known reachability limit

Body rate **3.96875 is not universally reachable**. At 1016 q256 every 256-block
must keep exactly 8 coded positions, leaving the global reverse-water-fill zero
slack to move a bit between blocks; on `layers.12.ffn.experts.82.w2.weight` the
arm-D schedule refuses with `cannot rebalance trellis-length guard`. **3.9375
(q256 1008) builds on all 24 tensors on both arms** and is the honest top rung.
Both the census and the driver record a refusal per (tensor, rate) rather than
crashing.

## Reproduce

    PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    cd /home/rob/dq-runs/trellis-hull-20260828     # scripts bind to the snapshot here
    $PY bypass_census.py                            # CPU only
    $PY scalar_subgrid_ladder.py --corpus dsv4      # CPU only
    $PY scalar_subgrid_ladder.py --corpus bf16      # CPU only
    $PY shaped_scalar_control.py                    # CPU only
    $PY e2m1_highrate.py --corpus dsv4 --out <rows.json>   # GPU
    $PY e2m1_highrate.py --corpus glm --glm-manifest <final-v2.json> \
      --out <glm-rows.json>                                # GPU
    $PY e2m1_highrate.py --corpus glm --glm-manifest <final-v2.json> \
      --glm-rate-plan high --out <glm-high-rate-rows.json> # GPU
    $PY fp8_learned_glm.py --manifest <final-v2.json> \
      --out <glm-fp8-learned.json> --dry-run               # CPU validation
    $PY coding_gain_table.py --rows <rows.json>

The four CPU drivers need no GPU and no trellis encode. `e2m1_highrate.py` is
the only one that funds GPU work; it reproduces published control rungs before
its new rows are trusted and refuses on a mismatch.
