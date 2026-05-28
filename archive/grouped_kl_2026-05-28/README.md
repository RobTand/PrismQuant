# Grouped-KL (fusion-matched) Cost Surrogate Archive

This archive preserves the implementation, tests, and validation artifacts for
the **grouped-KL** allocator cost surrogate: a `COST_MODE=grouped-kl` path that
replaced the per-Linear `h_trace × output_mse` Δloss with a fusion-matched
*grouped* KL measurement. It measured full-model KL for each (fusion-sibling
group, format) override and distributed `group_KL / N_members` as the
per-Linear `predicted_dloss`. The motivation was that per-Linear additive cost
over-attributes attention QKVO groups (~0.67× damping vs sum-of-individuals),
which produced the wrong promotions at high budgets and a 5.5→6.0 bpp PPL
non-monotonicity in the local allocator.

**The technique did not survive the production serving contract and is not in
the production path.** It looked like a large win on a local/HF screen but
LOST the apples-to-apples vLLM A/B against the shipped artifact:

- On a **local allocator + HF-PPL screen** (the original 2026-05-20 result),
  grouped-KL beat the per-Linear baseline at all budgets and fixed the
  non-monotonicity, with a headline **−3.52% PPL at 6.0 bpp** on Qwen3.6-27B.
- On the **production vLLM serving contract**, the same grouped 5.5 assignment
  **regressed**: worse exact full-vocab vLLM KL (grouped last-token KL ≈ 0.075
  vs the local baseline's ≈ 0.028 at matched bpp) and worse direct WikiText PPL
  than the **shipped 5.5 artifact**. The recorded decision was *"do not ship
  grouped-KL 5.5 as a replacement for the published 5.5 artifact"*
  (see `docs/grouped_kl_allocator_results_2026-05-20.md`, the "Decision after
  shipped comparison" section).

The KL/HF-PPL screen and the vLLM serving result **disagree in sign**. Per the
measurement-discipline rule (KL is a screen, not a promotion metric), a
candidate that improves a screen but regresses the shipped serving metric stays
research-only. Grouped-KL is therefore walled off.

The production pipeline (`prismaquant/run-pipeline.sh`) now **errors fast** when
`COST_MODE=grouped-kl` is requested, pointing back to this directory. The
default `COST_MODE` remains `production-render-score`; `production-render-staged`
and `local` are also available.

## Why it lost under serving

The grouped objective spent its budget very differently from the local
baseline at matched bpp. The recorded 27B 5.5 mixes:

| assignment        | bpp   | format counts            |
|-------------------|-------|--------------------------|
| local baseline 5.5| 5.499 | `BF16=157`, `NVFP4=347`  |
| grouped 5.5       | 5.497 | `BF16=283`, `NVFP4=331`  |

Grouped promoted nearly 2× as many Linears to BF16 (whole fusion groups at a
time, on the strength of grouped damping). That reduced the *local* per-group
KL the surrogate measured, but produced a served model that is worse on exact
vLLM KL and PPL than the more NVFP4-heavy baseline. The damping signal the
surrogate captured did not transfer to the served numerics.

## Contents

```
prismaquant/grouped_kl_cost.py   — the grouped-KL probe + cost-synthesis module + CLI
tests/test_grouped_kl_cost.py    — unit tests (3 tests: unit discovery, share
                                   synthesis + fallback preservation, round-trip
                                   share-sums-back-to-group-cost after aggregation)
docs/grouped_kl_allocator_results_2026-05-20.md
                                 — full 27B validation report incl. the
                                   "do not ship" shipped-comparison decision
```

## MoE caveat (never validated)

Grouped-KL was only ever validated on **dense** models (Qwen3-4B + Qwen3.6-27B).
On MoE, the module **skips packed expert tensors and falls back to per-Linear
baseline cost for them** (`grouped_kl_cost.py`, `synthesize_grouped_cost_payload`
fallback path), so an MoE run would silently mix grouped cost on non-experts
with per-Linear cost on experts — an unvalidated, inhomogeneous objective. The
fail-fast guard removes this footgun entirely; do not re-enable for MoE without
designing a per-expert grouping first.

## To revive

Copy the contents back into the production tree and remove the `grouped-kl`
fail-fast arm in `prismaquant/run-pipeline.sh`. Then **re-validate under the
production vLLM serving contract** (exact full-vocab KL + direct WikiText PPL
against the shipped artifact at matched bpp), not the HF-PPL/local screen that
inverted here. Do not re-promote on a KL/HF-PPL improvement alone — that is
exactly the signal that misled the 2026-05-20 result.
