# E2M1 trellis: the high-rate band, and the coding gain against an honest scalar

Status: **drivers preserved, three runs STAGED BUT UNRUN.** 2026-08-30.

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

| body rate | DSv4 gain | bf16 gain (preliminary, 18/24) |
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

Weight-only corpus SSE (W*A16). RTN render on both sides, so the NVFP4
incumbent is a lower bound on production NVFP4 (which renders under GPTQ+JSO).
No served A/B, no kernel measurement, no activation side. The bf16 corpus is
Qwen3-4B **dense** MLP -- not MoE, not GLM. The two-tier scale plane is a
research pricing and does not exist on the wire.

## STAGED BUT UNRUN

Nothing below was launched; the box was owned by `bf16-w4a4-24t` all session
and the session closed before it cleared.

1. `e2m1_highrate.py --corpus bf16` -- arms C and D at R in {1.0, 2.0, 2.5,
   3.0}. Would replace the preliminary bf16 gains above with class-restricted
   ones and add R=2.5/3.0.
2. `e2m1_highrate.py --corpus dsv4` -- the high band, 3.25/3.5/3.75/3.9375,
   plus 3.96875 where reachable.
3. `coding_gain_table.py --rows <each rows file>` -- the single matched-rate
   table over both corpora.
4. A GLM arm was specified and then countermanded. `--corpus glm` is NOT
   implemented. `/home/rob/dq-runs/glm-corpus-20260830/` exists (33 tensors,
   3192-4197 distinct source values, routed + dense). NOTE for whoever wires
   it: its manifest is a `tensors` DICT keyed by name with a `role` field, not
   the `entries` LIST the bf16 loader expects, and **it carries no importance
   vectors at all** -- every other corpus here supplies one, and the metric
   `enc_w = imp * pes^2` depends on it. That has to be resolved before a GLM
   arm means anything. Read it with `glm-corpus-20260830/extract.py`'s
   pread range reader, never `safe_open`: mmap over NFS wedged sparky into an
   unclean reboot on 2026-08-30 with 66 GB free and no OOM kill.

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
    $PY coding_gain_table.py --rows <rows.json>

The four CPU drivers need no GPU and no trellis encode. `e2m1_highrate.py` is
the only one that funds GPU work; it reproduces published control rungs before
its new rows are trusted and refuses on a mismatch.
