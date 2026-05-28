# PrismaQuant Design Document

> *Mixed-precision LLM quantization, selected on real end-to-end KL.*
> A principled account of the numerical methods, design choices, and
> rejected alternatives that produced the current production pipeline.

This document captures the **why** behind every load-bearing decision in
the codebase, so a reader can trace any module back to the principle it
serves and recognise what alternatives were considered and rejected. It
is the companion to `AGENTS.md` (rules) and
`docs/design_guidelines.md` (gates); together they constitute the
methodology of the project.

It describes the current production method, the superseded methods
that explain why it looks this way, and the limitations that still
matter.

---

## Table of Contents

1.  [Product Promise and Core Principle](#1-product-promise-and-core-principle)
2.  [Numerical Foundations](#2-numerical-foundations)
3.  [Cost-Surrogate Hierarchy](#3-cost-surrogate-hierarchy)
4.  [Allocation: Knapsack DP and Knee Detection](#4-allocation-knapsack-dp-and-knee-detection)
5.  [Render Pipeline](#5-render-pipeline)
6.  [Production Weight Cache](#6-production-weight-cache)
7.  [Export](#7-export)
8.  [Validation](#8-validation)
9.  [Plugin Architecture](#9-plugin-architecture)
10. [MoE-Specific Design](#10-moe-specific-design)
11. [Alternatives Considered and Rejected](#11-alternatives-considered-and-rejected)
12. [Known Limitations and Maintenance Debt](#12-known-limitations-and-maintenance-debt)
13. [Open Questions and Roadmap](#13-open-questions-and-roadmap)

---

## 1. Product Promise and Core Principle

### Promise

> *Choose the right quantization for the right layer.*

PrismaQuant exists to allocate a **bit budget** across a model so each
Linear (and each MoE expert) gets the format that maximises the
per-bit reduction in expected loss. The output is a standard
`compressed-tensors` checkpoint that vLLM serves natively. No forked
runtime, no custom kernels in the published artifact.

### Core Principle: each extra bit goes where it best improves accuracy

This single sentence drives every load-bearing decision below. Concretely:

-   **Per-Linear granularity** is non-negotiable. Model-wide defaults
    are caps and fallbacks, never substitutes for measured per-Linear
    choice (`docs/design_guidelines.md` "Per-Linear Decisions").
-   **Surrogates generate, real KL selects.** Cheap surrogates rank
    candidates; the shipping artifact is selected by measured
    end-to-end KL on a held-out calibration split.
-   **vLLM and kernel reality gate every format.** A format is
    production-eligible only when it loads in vLLM, routes to a
    performant kernel on representative shapes, and survives MTP
    sidecar artifacts.
-   **One cache mechanism.** Rendered weights live in
    `ProductionWeightCache`; activation replay in
    `PerturbedActivationCache` or the streaming activation cache.
    Anything else is a parallel cache and must be rejected.
-   **Measurement discipline.** Every promotion claim is paired with a
    matched-calibration A vs A+lever comparison, KL screening, and a
    downstream serving-suite check (PPL, MMLU log-likelihood,
    ToolEvalBench).

### Why MSE matters even when benchmarks don't move

A 5.0 bpp artifact whose calibration-KL collapses by 30-40× over a
4.75 bpp ship can register near-identical zero-shot scores. This is
not a contradiction: KL captures **distributional fidelity across all
tokens**; zero-shot benchmarks capture **a small set of
greedy/argmax answers**. The Pareto position has actually moved; the
benchmark suite simply lacks sensitivity at this resolution. The
correct response is not to give up the .25 bits; it is to add a
serving-suite metric (e.g. mean NLL on a long-form held-out corpus,
ToolEvalBench, or tail-quality probes like "does the model still
solve a 6-step chain-of-thought") that *does* register the change.

This is the rationale behind `docs/design_guidelines.md` "Measurement
Discipline": *KL is a screening metric, not a standalone promotion
metric*. The 5.0 bpp Qwen3.5-122B example is exactly the case the
gate was designed to handle: ship it on KL+MSE evidence, declare the
benchmark-suite null result explicitly, document that the next
quantization step needs a more sensitive task suite.

---

## 2. Numerical Foundations

### 2.1 Second-order loss expansion

For a model parameterised by `W`, the expected calibration loss `L`
around the unperturbed weights satisfies, to second order in the
perturbation `ΔW = W_q - W`,

$$
\mathbb{E}\bigl[L(W + ΔW)\bigr] - L(W)
\;\approx\;
\nabla L^{\top} ΔW
\;+\;
\tfrac{1}{2}\,ΔW^{\top}\,H\,ΔW.
$$

For a converged base model the gradient term `∇L` vanishes in
expectation, and the Hessian `H` is approximated by the **empirical
Fisher diagonal** `diag(F) ≈ diag(E[gg^T])`. Per Linear, summing along
the parameter dimension:

$$
\boxed{\;\;\Delta L \;\approx\; \tfrac{1}{2}\,H_{\text{trace}}\cdot \mathrm{MSE}_W\;\;}
$$

where `H_trace` is the empirical Fisher diagonal trace (computed once,
on a calibration pass through `sensitivity_probe.py`) and `MSE_W` is
the measured per-format round-trip weight error
(`measure_quant_cost.py`).

This is the **L1 cost surrogate** at the bottom of the hierarchy. It
is additive, cheap, and provably consistent under the second-order
expansion when only one Linear is perturbed at a time. Its failure
mode is also clear: when many Linears are perturbed jointly, the
quadratic cross-terms (the off-diagonal of `H`) become
non-negligible, and the additive sum overshoots the measured KL by
30-50% across budgets
(`archive/grouped_kl_2026-05-28/docs/grouped_kl_allocator_results_2026-05-20.md`).

That bias is what drives the surrogate hierarchy in §3.

### 2.2 Fisher diagonal computation

`sensitivity_probe.py:1027+` defines `FisherAccumulator`, which
installs a full backward hook on every quantizable Linear, accumulates
`||∇_W L||_F^2 / n_tokens` into a per-tensor scalar `gpu_h_trace`,
and (optionally) a per-row/per-output-channel diagonal `h_full`.

For MoE packed experts, the same principle applies — squared
gradient per weight — but the storage layout is `[E, M, N]` instead
of `[out, in]`. The implementation in
`sensitivity_probe._GradNormCapture` (line 413) chunks along the
expert dimension to bound peak memory and uses the
`RouterTracker` (line 920+) to **divide each expert's Fisher by its
observed routing probability**. Without that scaling, sparse experts
look artificially less sensitive than densely routed ones. With it,
expert Fisher values are calibrated as "loss change *per actually
used token*" and become comparable across the expert bank and across
dense Linears.

### 2.3 Why empirical Fisher and not true Hessian

True Hessian-vector products on a 27B model cost more than the entire
quantization run and are unstable when the loss surface is locally
flat (i.e. exactly where it matters least). The empirical Fisher
diagonal is the same trace you get from a single calibration backward
pass; it is what every modern mixed-precision allocator uses
(HAWQ-V2/V3, AMQ, CoopQ) and the lower-bound on quality from this
choice is captured in the L2/L3 lifts below.

Alternative considered: **Block-Hessian** (full per-block `H_block`,
captured by SqueezeLLM-style outlier-aware methods). Rejected because
the storage cost on a 671B MoE puts it at >100 GB just for
per-Linear blocks, and the L2 "perturbed-X" fixed-point recovers
most of the benefit at negligible cost; see §3.2.

---

## 3. Cost-Surrogate Hierarchy

PrismaQuant runs a **multi-level cost cascade**, from cheap-and-biased
to expensive-and-faithful. Each level is generated from the level
below or alongside it, and the validated-frontier kneedle (§4)
selects the level that wins on the calibration-anchored measurement
contract.

| Level | Surrogate                    | Storage     | Time on 27B | Drives                  |
|-------|------------------------------|-------------|-------------|-------------------------|
| L1    | `½·H_trace·MSE_W`            | KB          | seconds     | First DP solve          |
| L2    | Perturbed-X output MSE       | MB          | ~5 min      | Fixed-point DP re-solve |
| L3    | Paired BF16 end-KL           | small       | ~1 hr       | Bounded neighborhood DP |

Archived research, not used by the current production path:

| Level | Surrogate                    | Storage     | Time on 27B | Drives                  |
|-------|------------------------------|-------------|-------------|-------------------------|
| L2b   | Grouped-KL (fusion-matched)  | MB          | ~30 min     | DP cost replacement (ARCHIVED 2026-05-28 — lost shipped vLLM A/B; `archive/grouped_kl_2026-05-28/`) |
| Lswap | Empirical budget-neutral swap | small      | ~1 hr/swap  | Post-DP refinement      |

The cascade is **monotone in cost and quality**; each level can
demote to the previous if its measurement gate fails (`docs/design_guidelines.md`
"Progressive Local Gates").

### 3.1 L1 — additive Fisher

Computed by:

- `sensitivity_probe.py` (or `incremental_probe.py` for models that
  don't fit in RAM): one calibration pass, one Fisher diagonal trace.
- `measure_quant_cost.py` (or `incremental_measure_quant_cost.py`):
  for every `(Linear, format)` pair, render the weight via the
  format's `quantize_dequantize` closure and compute `MSE_W`
  and (when activation captures are available) `MSE_out`.

The DP allocator (§4) consumes `predicted_dloss = ½·H_trace·MSE` as
the cost edge weight, plus the bit cost of each format (`fr.bpp(fmt,
shape, ...)`). The selector picks measured `output_mse` when present
and falls back to `predicted_dloss` or `weight_mse`
(`allocator_candidates.py:237-265`). The fallback chain is now logged
at candidate-build time via allocator `cost-source usage` summaries,
and cost payload validation accepts and checks an explicit
`cost_source` field. Older artifacts that lack the field still load,
but rewritten costs such as grouped-KL or production-render-score rows
carry provenance.

### 3.2 L2 — perturbed-X fixed point

Once an L1 assignment exists, the activations downstream of every
Linear have shifted. Re-measuring `MSE_out` under the *perturbed*
activation distribution gives a better-conditioned cost. The fixed
point is reached in roughly three sweeps under weighted-Hamming
convergence (`docs/prismascout_overview.md`,
`docs/prismascout_handover-2026-05-03.md`).

The activation distribution is captured by `PerturbedActivationCache`
(`prismaquant/perturbed_x_cache.py`), the **canonical activation
cache**. Production runs forbid parallel activation stores. The L2
fixed point lifts measured KL by 5-15% over L1 on 4B-27B models.

### 3.3 L2b — grouped-KL (fusion-matched) — ARCHIVED 2026-05-28

> **ARCHIVED, not in the production path.** Grouped-KL looked like a win on
> the local-allocator / HF-PPL screen below, but **lost the apples-to-apples
> vLLM A/B** against the shipped Qwen3.6-27B 5.5 artifact (worse exact
> full-vocab vLLM KL and worse direct WikiText PPL at matched bpp). Per the
> measurement-discipline rule (KL/HF-PPL is a screen, not a promotion metric),
> it is walled off: `COST_MODE=grouped-kl` now fails fast. The implementation,
> tests, and the full validation record (including the "do not ship" decision)
> live under `archive/grouped_kl_2026-05-28/`. The analysis below is retained
> as design history. **Treat the PPL table as a screen-only result that the
> serving contract reversed.**

It was proposed 2026-05-20 on Qwen3.6-27B
(`archive/grouped_kl_2026-05-28/docs/grouped_kl_allocator_results_2026-05-20.md`).
It replaces per-Linear additive `½·H_trace·MSE` with a measurement of
**group KL** taken over the model's fused-sibling decision units
({qkv}, {o_proj}, {gate_up}, {down_proj}, plus
{in_proj_qkvz}, {in_proj_ab}, {linear_out_proj} for linear-attention),
distributing `group_KL / N_members` back to each Linear as its
predicted Δloss.

The mechanism it captures: attention QKVO groups exhibit ~0.67×
damping versus the sum of individual `½·H_trace·MSE` contributions.
The additive surrogate over-attributes attention, and the wrong
promotions land at high budgets — empirically as the **non-monotone
5.5→6.0 bpp PPL regression** that the per-Linear cost surrogate
exhibits (Qwen3-4B quality survey, 2026-05-19).

On the local-allocator / HF-PPL screen, grouped-KL **fixed the
non-monotonicity** and beat per-Linear cost at all measured budgets on 27B
(`archive/grouped_kl_2026-05-28/docs/grouped_kl_allocator_results_2026-05-20.md`)
— but this screen was **reversed by the vLLM serving contract** (see banner above):

| Budget | per-Linear PPL | Grouped PPL | Δ        |
|--------|---------------:|------------:|---------:|
| 5.0    | 7.235          | 7.167       | −0.94%   |
| 5.5    | 7.137          | 7.131       | −0.10%   |
| 6.0    | 7.237          | 6.982       | **−3.52%** |

It now lives at `archive/grouped_kl_2026-05-28/prismaquant/grouped_kl_cost.py`.
Schema: `"prismaquant.grouped_kl_cost.v1"`. It never measured MoE expert
groups — experts route per token and the group-vs-sum damping observed on
attention QKVO has no obvious MoE analogue — and on MoE it silently fell back
to per-Linear baseline cost for experts (an inhomogeneous objective), which is
one more reason it is walled off. The queued 35B-A3B A/B was never run.

The grouped-KL share distribution (`group_KL / N_members`) is now pinned by
a unit test: `group_KL → distribute → aggregate_fused_siblings → group_KL`
must round-trip within float tolerance. This guards against accidental
double-counting when grouped-KL shares flow through the normal DP aggregator.

### 3.4 L3 — propagated end-KL

For the **bounded neighborhood** of Linears the cascade still flags
as uncertain after L2/L2b, L3 measures **paired BF16 versus candidate
end-to-end KL** through the actual model on a small calibration
batch. Each candidate's loss is its measured KL contribution
*conditional on the rest of the assignment being at the candidate
formats*. The L3 DP then re-solves over the bounded set.

Implementation:
- `kl_measurement.py:select_l3_neighborhood`, `select_formats_for_l3`,
  `build_l3_candidates`, `solve_frozen_l3_neighborhood`.
- Paired baseline construction: each candidate row pairs against
  the same-target-BF16 lane so layer-to-layer interference is held
  constant across rows.
- An experimental drift-relative mode pairs candidates against the
  *current assignment* lane (`baseline_mode="assignment"`). That
  remains research-only until the swap selector is validated on target models.

The L3 cost is end-to-end; running it on a full 27B is feasible only
because of the `PerturbedActivationCache` + frozen-weight + replay
optimisations in `kl_measurement.py`.

Memory governors at `kl_measurement.py:206-330` automatically disable
the L3 prequant cache or shrink lane count when host memory drops
below configured floors. These are gated by
`PRISMAQUANT_L3_MIN_HOST_MEM_GB` and
`PRISMAQUANT_FROZEN_WEIGHT_CACHE_MIN_FREE_*`.

### 3.5 Lswap — empirical budget-neutral swaps (archived research)

Where the propagated-sensitivity path answers *"where do extra bits
help most?"*, the budget-swap module answers the conjugate question:
*"which low-risk demotions can pay for those promotions under the
same bit budget?"*

The research implementation consists of
`prismaquant/budget_swaps.py`,
`tools/build_budget_neutral_swaps.py`,
`tools/measure_budget_neutral_swaps.py`, and
`tools/select_measured_budget_swaps.py`:

1.  **Build candidates.** For each promotion candidate (high
    propagated-KL-per-added-bit), pair it with one or more demotions
    (low `output_mse_per_saved_bit` or `predicted_dloss_per_saved_bit`)
    such that `bits_added - bits_saved ≤ 0` and no two
    candidates touch the same qname.
2.  **Measure.** Each candidate swap is paired-KL-measured against a
    chosen baseline (`baseline_mode="target_bf16"` for absolute, or
    `baseline_mode="assignment"` for drift-relative).
3.  **Select.** A greedy, non-conflicting, budget-respecting
    selector keeps only swaps that improve BF16-relative KL by
    `min_kl_improvement` or more, with optional drift cap.

The 35B smoke (`docs/qwen36_35b_budget_neutral_swap_smoke_2026-05-25.md`)
demonstrated:

-   **n=4 smoke:** 2/4 measured swaps improved real KL; selector
    accepted one non-conflicting improver at net bits delta of
    -27.5M params.
-   **n=8 top-32:** 0/32 candidates were negative-delta. The
    surrogate proposed plausible swaps that empirical KL refused.

This is the right outcome. The cascade is doing its job: cheap
surrogates *propose*, real measurement *decides*. The result is not
"swaps fail"; it is "the surrogate's swap proposals are not yet
trustworthy at n=8/seqlen=512, and the policy correctly refused to
ship them". Budget-neutral swaps remain archived research and are not
a production dependency.

---

## 4. Allocation: Knapsack DP and Knee Detection

### 4.1 Multi-choice knapsack DP

Once every `(Linear, format)` candidate has a cost
(`predicted_dloss`) and a size (`memory_bytes` → bits), the
assignment problem is the canonical **multi-choice knapsack**:
choose exactly one format per Linear, minimise total predicted
Δloss, subject to a total-bits budget.

`allocator_solver.py:solve_allocation` discretises bits with a
configurable granularity (default `0.001` bits/param;
`allocator.py:476`), builds a DP table `dp[bin] → min Δloss`, and
backtracks from the budget bin to recover the assignment. The
single solver is the only DP path in production. There are **no
greedy, branch-and-bound, or relaxation fallbacks**.

**Why DP and not Lagrangian λ-bisection?** Lagrangian methods give
a continuous optimum for a fixed multiplier but produce a
*non-integer* assignment that must then be rounded; on small
budgets, rounding-induced infeasibilities matter and the DP is
within milliseconds of the relaxation while always being feasible.
PrismaSCOUT considered λ-bisection (paper §3) and rejected it on the
following grounds: the DP is fast enough at 0.001 bits/param
granularity, it composes cleanly with promotion (see §4.2), and the
optimal assignment is rarely sensitive to bit granularity below
~0.001 — measured at <10% Δloss improvement going from 0.001 to
0.0001 on 27B.

### 4.2 Fused-sibling pre-aggregation and MoE pair promotion

The DP cannot pick mixed-sibling assignments because there is only
one item per group, so fused-sibling groups (`qkv_proj`,
`gate_up_proj`, `down_proj`) are pre-aggregated into single DP
items (`allocator_candidates.py:412-529`). After the DP solves,
post-promotion handles two cases:

- **`promote_fused`** (`allocator_solver.py:98-128`): if the DP
  picked a format whose fused-sibling promotion would fit within
  the overshoot tolerance, promote the group.
- **`promote_moe_pair`** (`allocator_solver.py:62-95`): MoE
  `gate_up`/`down` per expert must share one serving format. If the
  DP gave them different formats, promote the cheaper to the
  more-expensive partner's format when budget allows.

These two constraints can overlap for nested per-expert layouts
(`gate_proj`/`up_proj` are a fused pair and also part of the
`gate/up/down` expert serving unit). The allocator now handles this
with `promote_serving_units`: it builds connected components across
both fused-sibling and packed-MoE group edges, then promotes each
component to the highest-ranked selected format in one pass. This
replaces order-dependent sequential promotion in the solve/finalize
paths while keeping `promote_fused` and `promote_moe_pair` as
compatibility wrappers.

### 4.3 Kneedle on log-error

After running the DP at a sweep of budgets, the validated-frontier
kneedle picks the **elbow** on `(bpp, KL)`. The historical
implementation (`kneedle_raw_linear`, still available for
diagnostics) ran on raw error. The default is now
`kneedle_log_error` (`allocator.py:158`):

```
floor = max(min(finite_positive) × 1e-6, 1e-300)
y' = log10(max(y, floor))
```

**Why log-error?** Allocator `predicted_dloss` spans 3-5 orders of
magnitude across the per-Linear menu. On the raw scale, the largest
point dominates the normalised curve and the kneedle picks the
first big absolute drop — typically far from the human-intuitive
elbow. On log-error, the curve is well-conditioned and the kneedle
selects the bpp where *relative* improvement starts to flatten.

This is exactly the right reframing for a quantity that is
fundamentally rate-distortion in nature: information-theoretic
distortion measures should be on the log scale before any
elbow-detector touches them.

The allocator writes both indices to the `.knees.json` sidecar, so the
diagnostic output records the raw and log-error knee points side by side.

**Hard-coded Pareto anchors (`allocator.py:439-440`):** the Pareto-sweep
defaults `4.5, 4.75, 5.5, 8.25` are workload-anchored, not
principle-anchored. They should remain as defaults but ideally be
derived from the `bpp` range observed in the cost pickle, with a
fallback to these constants only when no signal is available.

---

## 5. Render Pipeline

### 5.1 Progressive gates

A render mechanism is any per-Linear (or per-fused-group)
numerical transform applied during weight rendering: FourOverSix,
joint_scale_opt, GPTQ, scale_sweep. The progressive-render
contract (`docs/progressive_render_pipeline.md`) says:

1.  Render the current baseline.
2.  Render the candidate after one mechanism (or a candidate
    package for initialiser-class mechanisms).
3.  Score both on the same activation rows (Fisher-weighted output
    MSE when h-detail is available, otherwise activation output
    MSE).
4.  Accept the candidate only when the score improves by
    `PRISMAQUANT_RENDER_GATE_MIN_GAIN`. Reject = keep the previous
    baseline and continue.

Order is **declared by operation type** (`scale-rule`,
`scale-optimiser`, `rounding-solver`, `codebook-scale-refine`),
not by environment-variable text order. Current V1 order:

```
FourOverSix (NVFP4 only)   →  joint_scale_opt (NVFP4 only)  →
static_act_order (NVFP4, MXFP4, MXFP8)  →  GPTQ  →
scale_sweep (explicit ablations only)
```

### 5.2 Plugin contract

The shared scorer is `render_score.py:RenderMechanismSpec` (line
190): `name`, `operation`, `scope`, `phase`, `gate_metric`,
`after`/`before`/`exclusive_group`. Registered via
`register_render_mechanism()` and ordered by
`resolve_render_mechanism_order()`.

Uniformity is still partial. `gptq` and `scale_sweep` honour the
shared ordering; `FourOverSix`, `joint_scale_opt`,
`static_act_order`, and `fisher_gptq` have **inline special-case logic** in
`production_weight_cache.py:_render_nvfp4_progressively` (lines
1524-1617). `joint_scale_opt` is not even an independent mechanism
— it is a parameter passed to GPTQ at line 1849.

This is the most visible **principled-design violation** in the
implementation: every mechanism that participates in scoring should
follow the same registry contract, or the abstraction is leaky.

**Recommended fix:** elevate `joint_scale_opt` and `FourOverSix` to
first-class registered mechanisms with `phase` slots adjacent to
GPTQ. The inline branching becomes one shared `for mechanism in
ordered_plan: ...` loop. This is a ~200-line refactor inside
`production_weight_cache.py` with no behavioural change when gates
agree.

### 5.3 `block_output_match.py`

Block-level joint scale refinement (`MVGreedy` on grouped block
scales). Currently lives **outside** the progressive gates,
integrated via env flag `PRISMAQUANT_BLOCK_OUTPUT_MATCH=1`. It is a
post-scale_sweep refinement, not a pre-GPTQ initialiser, so its
correct phase slot is `phase=65` (between scale_sweep and export).
It should be registered as a normal mechanism.

---

## 6. Production Weight Cache

### 6.1 Single resident store

`prismaquant/production_weight_cache.py` is the **only** rendered-
weight cache in the production codebase. It implements:

-   `weights: dict[(qname, fmt), Tensor|Path]` — either resident or
    on disk (LRU evicted under `_lru_max_bytes`).
-   `activation_max_abs: dict[qname, float]` — calibrated clip per
    Linear, written from probe and re-fit by
    `production_recache.py`.
-   `metadata["render_gates"]` — per-Linear per-mechanism
    accept/reject decision log.
-   `metadata["render_scores"]` — per-Linear per-format scored MSE.
-   `metadata["four_over_six"]` — compact summary for the
    first-class FoS plugin.

The cache is **GPU-bound by contract** (`AGENTS.md` rule 1).
Production runs must never silently stream from NVMe; the
`PRISMAQUANT_STRICT_PRODUCTION_CACHE` env flag promotes residency
failures to errors. The prefetch path
(`production_weight_cache.prefetch_assignment_file_pages`) hooks
into `source_prefetch.py` to page-warm before validation.

### 6.2 `build_production_cache.py` vs `build_rtn_cache.py`

There are two cache-building CLIs, but only one production cache
mechanism:

-   `build_production_cache.py` is the canonical CLI that fills the
    production cache with **activation-aware, GPTQ + joint_scale_opt
    + scale_sweep + calibrated-clip** rendered weights.
-   `build_rtn_cache.py` is a **research baseline** that produces
    naive RTN safetensors shards for ablation/baseline KL runs. It
    is *not* consumed by the production export. It is not a parallel
    cache mechanism; it is a research artifact builder.

The right disposition is either to rename `build_rtn_cache.py`
something like `tools/build_rtn_baseline.py` (it is a research tool)
or to add a header comment marking it explicitly as research-only.
Both live in `prismaquant/` proper, which is misleading.

### 6.3 Packed-MoE exclusion is intentional, not a parallel cache

`production_weight_cache.py:97-111` defines
`_UNCACHED_PACKED_EXPERT_RE` and `is_uncached_packed_expert_qname`,
and the residency check explicitly skips qnames matching it. The
docstring justifies this: the cache renders 2D `nn.Linear` weights;
packed 3D expert tensors are quantized by `_quantize_3d_packed` in
the exporter. The cost pass now replays routed packed-MoE forwards
when activation caches are available, so the allocator can price real
`output_mse` for packed experts. The remaining gap is render-side:
GPTQ, joint_scale_opt, scale_sweep, and calibrated activation clip are
still not applied inside `_quantize_3d_packed`.

---

## 7. Export

### 7.1 Format codec primitives are unified

The exporter keeps NVFP4/FP8/MXFP8/MXFP4 codec math behind single
shared primitives:

| Codec primitive | File:line | Callers |
|-----------------|-----------|---------|
| `_nvfp4_quantize_grouped_codec` | `export_native_compressed.py:349` | 3 |
| `_mxfp4_grouped_codec`          | `:2093`  | 3 |
| `_mxfp8_grouped_codec`          | `:2061`  | 7 |
| `_fp8_codec`                    | `:1953`  | 3 |
| `_fp8_dynamic_codec`            | `:1977`  | 4 |

Each codec primitive is the **sole call site** for its algorithm.
This prevents a class of bugs where renderer math and served metadata
math drift. One concrete example was **MXFP4 E8M0 scale encoding**:
it disagreed with `compressed-tensors`'s `generate_mx_scales` near
power-of-two rounding boundaries. The fix added shared
`_mx_rounded_amax_power2` / `_mx_base_exponent_from_amax` and
re-routed both MXFP8 and MXFP4 packers through it.

The reconciliation test at
`tests/test_prismaquant_export_native_compressed.py:447`
(`test_registry_render_dequant_matches_served_metadata`) now
forces every registered exportable format to be **either reconciled
with vLLM served metadata math or marked as an explicit research
gap**. Explicit gaps: `MXFP6_E3M2`, `MXFP6_E2M3`, `INT8_W8A16`,
`INT4_W4A16_g128`.

This pattern — *one codec primitive, one reconciliation test* —
is the right shape for the codebase and should be the template
for any future format that joins the menu.

### 7.2 Dense (2D) vs packed-MoE (3D) paths

`_quantize_2d` (line 3151, 19 format branches) and
`_quantize_3d_packed` (line 3494, 5 format branches) are the two
entry points. The divergence is **legitimate** at the codec layer
(3D vs 2D have different memory layouts) but **lazy** at the
optimisation-stack layer: packed experts still have no render-side
GPTQ, no JSO, and no scale_sweep, even though the same math applies
and `export_batched_gptq.py` already implements batched same-shape
NVFP4 GPTQ proven to be bit-exact-equivalent to the per-Linear path.
The measurement half is closed: `measure_quant_cost.py` can replay
routed packed-MoE forward passes and record measured `output_mse`.

Closing this gap is one of two highest-impact follow-ups (§13).

### 7.3 MTP passthrough deduplication

`_filter_source_passthrough_against_materialized` prevents the source
passthrough does not duplicate keys that the materialiser has
already emitted in synthesized aggregate form. The vLLM loader
warns on the duplicate; the fix is purely an export-discipline
cleanup. It is profile-driven and uses `_per_expert_parent` to detect
packed-expert children that correspond to a materialised packed tensor.

### 7.4 What the exporter does and does not own

The exporter owns:
- shape coercion to the serving profile (every `(name, format)`
  must satisfy the serving profile's shape rules; otherwise the
  format is coerced to a legal alternative);
- BF16 upgrade report (an immutable region — `lm_head`, pinned
  Linears, MTP sidecars when configured — never enters the
  optimisable budget);
- collapsed `quantization_config` emission so vLLM does not pay
  O(N²) per-expert target walk on init
  (`collapse_config_groups.py` is the runtime + retro-fix tool);
- writer-coherent global scale unification for sibling-coherent
  NVFP4.

The exporter does **not** own kernel selection. That belongs to
vLLM at runtime. `kernels/nvfp4_fused.py` is used by
`PerturbedActivationCache` for validation/replay acceleration and is
covered by kernel tests, but it is not serialized into exported
artifacts and therefore does not violate the "vanilla vLLM shipped
artifact" rule.

---

## 8. Validation

There are four validators with **four distinct responsibilities**:

| Module                          | Owns                                                       |
|---------------------------------|------------------------------------------------------------|
| `validate_assignments_kl.py`    | Measure real last-token KL for an assignment JSON vs BF16. |
| `validate_native_export.py`     | vLLM smoke: can it load + forward + greedy decode?         |
| `validate_quantized_model.py`   | Production quality gate: PPL, MMLU, MTP acceptance.        |
| `validation_harness.py`         | CI/CD aggregation; artifact registry bookkeeping.          |

The chain is:

```
allocator.py
  → validate_assignments_kl.py    (KL ✓ before export commits)
  → export_native_compressed.py
  → validate_native_export.py     (vLLM load ✓)
  → validate_quantized_model.py   (downstream metrics ✓)
  → validation_harness.py         (record in artifact registry)
```

The 27B production discipline (the env-var hooks
`PRISMAQUANT_VALIDATION_PROD_CACHE`,
`PRISMAQUANT_VALIDATION_PROD_CACHE_DIR`,
`PRISMAQUANT_VALIDATION_PROD_CACHE_LRU_GB`,
`PRISMAQUANT_VALIDATION_SKIP_END_KL`) is what makes 27B PPL fit on
121 GB UMA hardware (`handover 2026-05-20`). They should remain
opt-in, because they trade fidelity (RTN render in
`PerturbedActivationCache` is the on-the-fly fallback) for
host-memory headroom. The right way to think of them is as
**hardware-class enablement flags**, not as quality knobs.

---

## 9. Plugin Architecture

### 9.1 Three concerns, three registries

The plugin contract (`docs/pluggable_refactor.md`) separates:

1.  **Model structure** — `prismaquant/model_profiles/specs/*.json`
    + the `ModelProfile` subclass. Owns naming
    (`source ↔ recipe ↔ vllm`), fused groups, packed-expert
    decomposition, MTP sidecars, lm_head, fast-kernel deps,
    default serving profile.
2.  **Serving constraints** — `prismaquant/serving_profile_specs/*.json`
    + `prismaquant/serving_profiles.py`. Owns backend format
    menus, kernel shape rules (NVFP4 in_features%16, MXFP8 32×128),
    optional runtime validators (FlashInfer problem-size check),
    pinned package versions.
3.  **Pipeline contract** — `prismaquant/pipeline.py`. Owns
    `PipelineSpec`: stages, typed artifacts, metric gates, cache
    ownership.

Profile resolution
(`prismaquant/model_profiles/registry.py:_REGISTERED`) is by
`(model_type, architectures)` from the source `config.json`. No
path-string sniffing in live code; the only remaining substring
match is the fallback in `_fast_kernel_guard.py:69-92` when
`config.json` is unreachable. That fallback is acceptable as a
fast-kernel guard, but it should not grow into the rest of the
codebase.

### 9.2 What the JSON spec already owns

The JSON spec is sufficient for **most** of the contract:

| Aspect                       | JSON | Python | Required? |
|------------------------------|:---:|:------:|:---------:|
| Profile registration         |  -  |   ✓    |  yes      |
| Match criteria               |  ✓  |  fallback | yes    |
| Fused-sibling groups         |  ✓  | vLLM auto-derive | no |
| Name rewrites (3 directions) |  ✓  | vLLM auto-derive | no |
| MoE packing structure        |  ✓  | vLLM auto-derive | no |
| Per-expert / per-MTP regex   |  ✓  | optional override | no |
| MTP module build             |  -  |   ✓    | when MTP  |
| Streaming-probe adapters     |  -  |   ✓    | when custom naming |
| Shard regexes                |  ✓  | extend in code | no |
| Fast-kernel deps             |  ✓  | extend in code | no |
| Default serving profile      |  ✓  |   -    | no        |
| Passthrough prefixes         |  ✓  |   -    | no        |

A new architecture can be added **without Python code** when it has
no MTP, no custom streaming-probe naming, and standard fused-sibling
+ packed-expert layout known to vLLM. Real-world architectures
almost always need at least one Python hook (typically
`build_mtp_module`).

### 9.3 Where the contract leaks

Three observed leaks:

1.  **MTP module construction is Python-only.**
    `Qwen3_5Profile.build_mtp_module` (lines 78-126) and
    `Qwen3_5DenseProfile.build_mtp_module` (lines 55-107) are 95%
    identical. JSON specs cannot declare MTP forward logic. This is
    the **most copy-paste-prone** seam in the model-profile layer.

2.  **Streaming-probe adapters (refactor #32) are all Python.**
    `checkpoint_to_live_name`, `init_rotaries`,
    `expand_hidden_for_layers` are Python methods on the profile
    class. No JSON equivalent.

3.  **vLLM class dependency is tight.** If vLLM has not yet shipped
    an architecture class, the profile must hand-author
    `fused_sibling_group()` and `to_vllm_internal_name()` instead of
    auto-deriving from vLLM's `packed_modules_mapping`. This is
    fine in practice — most production architectures are vLLM-blessed
    — but it ties the live-tree quantization workflow to upstream
    vLLM's release calendar.

None of these are blockers; they are seams to factor when the
inheritance burden becomes painful.

### 9.4 `decision_units.py` is profile-driven

The atomic flip targets for post-DP polish are constructed by
`discover_units(model, profile, ...)`. Fused-sibling grouping uses
`profile.fused_sibling_group(qname)`; packed-expert format groups
use `spec.packed_expert_format_group(qname)`. **Zero architecture-
specific branches** in `decision_units.py` itself. This is the
shape we want every other allocator-adjacent module to converge
toward.

---

## 10. MoE-Specific Design

MoE and dense paths diverge in eleven places. Some divergences are
required by serving semantics; others are implementation gaps:

| # | Divergence                              | Classification | Status                |
|---|------------------------------------------|----------------|-----------------------|
| 1 | Production cache excludes packed experts | necessary      | by design (2D vs 3D)  |
| 2 | GPTQ not applied to packed experts       | **lazy**       | open render gap       |
| 3 | joint_scale_opt not applied to experts   | **lazy**       | open render gap       |
| 4 | Fused-sibling skipped for experts        | necessary      | experts route per-token, never fuse |
| 5 | Grouped-KL skipped for experts           | necessary      | per-token routing; per-expert KL ≡ per-Linear |
| 6 | Activation collection not used for experts | **lazy/bug** | implemented           |
| 7 | Format restrictions: experts vs dense    | necessary      | vLLM packed-MoE kernel support |
| 8 | mse_promotion applies uniformly          | consistent     | no action             |
| 9 | propagated-sensitivity applies uniformly | consistent     | no action             |
|10 | decision-unit grouping different for MoE | necessary      | by serving model      |
|11 | MTP handling                              | orthogonal     | independent feature   |

### 10.1 The necessary divergences

**Per-token routing changes what "fused" means.** Dense `q/k/v` are
always fused at serving time and always see the same input tokens,
so a single format choice for the group is principled. MoE experts
see **disjoint token subsets** at routing time, so each expert's
allocation is independent. The fused-sibling pre-aggregation
correctly skips packed experts, and the post-DP `promote_moe_pair`
correctly handles the constraint that gate_up and down within one
expert must match.

**Per-token routing also explains why grouped-KL doesn't trivially
extend.** Grouped-KL measures damping when a whole group is
quantized together. The damping pattern observed on dense
attention QKVO comes from inter-projection cancellation. For
MoE experts, there is no inter-expert cancellation; each expert
operates on its own token subset. The right MoE analogue would be
**grouped over routing decisions**, not over experts, and that is a
research direction, not a default lever.

### 10.2 Packed-MoE activation-aware measurement

**Packed-MoE activation-aware measurement** was the largest
unexploited MoE lever. The
`measure_quant_cost._packed_experts_forward_with_weights` and
`_measure_packed_experts` (with `act_cache` parameter) replay
the **routed** packed-MoE forward path with the actual router's
top-k weights, allowing:

-   real per-expert `output_mse` measurement (not the placeholder
    zero older costs recorded);
-   correct activation-aware output-MSE pricing for the available
    packed-expert render callbacks;
-   downstream allocator cost edges that reflect *real* output
    error, not weight-only proxy.

The architecture is router-aware replay with input/intermediate
quantize callbacks and gate/down weight slots, implemented in
`prismaquant/measure_quant_cost.py`.

The render half remains open:
**activation-aware GPTQ + scale_sweep on `_quantize_3d_packed`**.
The batched-GPTQ infrastructure (`export_batched_gptq.py`) already proved
bit-exact equivalence to per-Linear NVFP4 GPTQ; wiring it into the 3D export
is a localised refactor.

### 10.3 MTP budgeting

MTP and visual Linears are auxiliary serving decisions, not part of the
language-model bit budget. The allocator stamps them into the exported
`layer_config` through `--mtp-format` and `--visual-format`, but the default
Pareto frontier, kneedle, bpp, and predicted Δloss report only the body
Linears that are actually competing for the budget.

When MTP is explicitly quantized, measured candidates are still required so
the run can record auxiliary Δloss and catch unsupported serving formats
early. Those auxiliary numbers are written beside the Pareto rows as
`aux_fixed_*` and `total_*_with_aux` fields, but they do not move the body
kneedle or consume the target bpp. This keeps MTP/speculative-decode speed
and quality decisions independent from the core language-model allocation.

---

## 11. Alternatives Considered and Rejected

This section is the institutional memory of *what we tried, what
worked, and what didn't*. Each row is a research path that lives
under `archive/` with a dated suffix, with the corresponding
notes and run logs preserved.

### 11.1 Cross-layer integer quadratic programming (CLADO)

**`archive/cross_layer_2026-05-09/`**.

CLADO (Deng et al. 2023, [arXiv:2307.05657](https://arxiv.org/abs/2307.05657))
measures the residual pairwise quantization-error coupling between
Linears on a small data subset and solves the resulting integer
quadratic program directly. PrismaQuant prototyped this end-to-end
and uses its **decision-unit framing** in the polish path (Block-
CLADO decision units), but the **full IQP solver was rejected**
because:

1.  Per-pair measurement scales `O(N²)` in Linears (≈3.5M Linears
    for 27B). Even with sparse subsampling the measurement budget
    dominated total wall time.
2.  The DP + propagated-sensitivity cascade (§3) recovers most of
    the cross-layer benefit at `O(N)` measurement cost.
3.  The IQP optimum was within 1-2% of the cascade output on the
    measured 27B and 4B comparisons.

PrismaSCOUT (§3) is what we ship instead. The CLADO framing of
decision units survives in `prismaquant/decision_units.py`; the
solver does not.

### 11.2 PrismaSCOUT L3 polish (validated then archived)

**`archive/polish_2026-05-15/`**.

L3-polish was the original production-faithful polish over a chosen
assignment, using `WeightSession` for in-place delta-quantize trials.
It validated on 4B (KL improvement +34%) but archived 2026-05-15
because:

1.  Per-Linear costs measured under the *L2* context don't sum to
    true end-KL when DP changes many Linears simultaneously
    (`prismaclade_l3_non_additivity.md`).
2.  The same machinery was extracted into `decision_units.py` and
    `WeightSession` (still live in `prismaquant/weight_session.py`),
    so the *infrastructure* is preserved but the polish-of-many
    *policy* is research-only.
3.  The later propagated-sensitivity and budget-swap experiments
    explored the same intent (post-DP local improvement gated on
    real KL), but the budget-neutral swap policy remains archived
    research and is not part of the live production path.

The top-level production-path comment now reflects the live flow and no
longer mentions archived `polish_from_assignment.py`.

### 11.3 Layer-wise ReSpinQuant rotations

**`archive/respinquant_2026-05-13/`**.

Layer-wise residual-basis rotations are mathematically attractive
— rotate the activation basis so outliers align with axes, then
quantize. ReSpinQuant prototyped this. The blocker is **not
algorithmic**: rotations that change the residual-stream basis
between layers require a residual-transition adapter at vLLM
serving time. That adapter is a custom kernel; `docs/design_guidelines.md`
"Rotation Transforms" forbids it without an explicit kernel-support
decision.

Archived. Microscale block-diagonal rotation (BlockOrtho-G)
remained viable for a while because it folds into NVFP4 group
boundaries and *does not* change the residual basis. See §11.6.

### 11.4 HALO and Fisher-weighted rotations

**`archive/halo_2026-05-15/`, `archive/fisher_2026-05-15/`**.

HALO + Hadamard-style rotations did work on Qwen3.5 dense after
the 2026-05-09 fixes (γ-fold + linear_attn detection,
`halo_qwen35_norm_bug.md`). They were archived 2026-05-15 along
with the rest of the rotation family when the consolidation
(`consolidation_2026_05_15.md`) cut the production stack down to
`gptq + joint_scale_opt + PrismaQuant solver`. HALO is paper-cited
as a candidate replacement and there is a tracked memory
(`paroquant_candidate.md`) flagging ParoQuant (arXiv:2511.10645)
as a future rotation candidate to evaluate.

### 11.5 PrismaClip and PrismaFisherClip

**`archive/prismaclip_2026-05-14/`**.

Per-Linear weight clipping was a candidate lever but the clipping
behaviour ended up being implicitly captured by JSO's per-block
scale grid (`jso_is_implicit_clipping.md`). When JSO's 7-level
scale-grid collapsed to {6, 4} on >99.98% of blocks on Qwen3.5
0.8B, it became clear that the clip was just another way of asking
"what's the right scale?" — and JSO already answers that.

### 11.6 BlockOrtho-G

**`archive/foldscale_orthog_2026-05-13/`**.

Microscale (16-element) block-diagonal rotations that fold into
the NVFP4 group boundaries. The math is correct, and the runtime
is compressed-tensors-compatible via `transforms_config`. It remains
archived because the production cache + exporter wiring was
incomplete and the measured KL improvement was less than 1% over JSO
at the same bpp. Kept on the shelf as a future rotation option.

### 11.7 Multi-shot recalibration (LLM-Surgeon §3.5)

**`archive/multi_shot_2026-05-19/`**.

A cheap variant of cross-layer interaction correction failed
cross-cal validation on Qwen3-4B
(`multi_shot_qwen3_4b_validation_2026_05_18.md`) showed
**double-negative**: at production calibration, 5/5 runs gave
ΔKL=0 (no-op); under calibration-efficiency cross-check, one
budget actively regressed by -153%. It remains archived.

### 11.8 scale_sweep on small models

**Archived as default** (`scale_sweep_4b_regression.md`).

Direct scale_sweep over Qwen3-4B at production cal regressed KL
+77.5% despite local act-weighted MSE −2.97%. The local MSE proxy
disagreed with end-to-end KL because scale_sweep at production
cal trades a small local gain for a larger inter-layer coupling
cost. Still available as `--enable scale_sweep` for explicit
ablations on dense 27B+ models where the trade-off favours
the optimisation, but not part of the default recipe.

### 11.9 Closed-form analytical damp

**`damp_analytical_rejected.md`**.

A closed-form `κ-target` damping schedule (`c · λ_max / μ`)
regressed KL +100-161% versus the 5-candidate damp sweep on
Qwen3-4B. The model's per-Linear prediction error compounded
catastrophically end-to-end. The damp sweep remains essential
on small models (`damp_sweep_4b_essential.md`).

### 11.10 JSO remains default

**`jso_archived_2026_05_20.md`**.

A 4B A/B suggested JSO regressed PPL by 0.8-1.6%, but that evidence
is insufficient to remove it from the default recipe:

1.  The A/B had a cost-surrogate confound (each arm measured its
    own rendering).
2.  Shipped Qwen3.6-27B with JSO has 60k HF downloads — strong
    field evidence that JSO is fine at scale.
3.  Grouped-KL cost was newly introduced in the same session;
    disentangling JSO from grouped cost is a separate experiment.

A queued **27B isolation A/B** (`qwen36_27b_jso_isolation.sh`)
holds cost surrogate constant and varies only the renderer. The bar
for removing JSO is reproducibility under matched cost surrogates at
the actual ship target.

### 11.11 REAP / expert pruning

**`archive/reap_2026-05-15/`**.

Expert pruning was empirically bad on DSv4 v1
(`reap_pruning_disabled.md`). Size reduction should come from expert
factorisation, rotations, sub-NVFP4 formats, or larger host budgets,
not by dropping experts.

### 11.12 Sparse pairwise QUBO, top-K Hessian covering, surrogate-only knee, etc.

Detailed in `paper/main.pdf`. The short version: each represents a
different point on the expressiveness-vs-cost frontier, each was
prototyped, and each was rejected for one of three reasons:
prohibitive per-pair measurement cost (QUBO, top-K Hessian),
no improvement over the cascade at matched calibration (surrogate-
only knee), or no path to vLLM-served compressed-tensors output
(adapter-based rotations).

---

## 12. Known Limitations and Maintenance Debt

### 12.1 Render mechanism plugin uniformity

Only `gptq` and `scale_sweep` fully honour the
`resolve_render_mechanism_order()` plan. `FourOverSix`,
`joint_scale_opt`, `static_act_order`, and `fisher_gptq` still have
inline special-case logic in `production_weight_cache.py`.
`joint_scale_opt` is a parameter to GPTQ rather than an independent
mechanism, and `block_output_match.py` is integrated via env flag
rather than the registry.

The target shape is one loop over ordered `RenderMechanismSpec`
instances, with every mechanism declaring its phase, metric, and
compatibility gates in the same registry.

### 12.2 Packed-MoE render parity

Packed-MoE experts can now be measured with router-aware
activation-output MSE, but `_quantize_3d_packed` still lacks the
full dense render stack: GPTQ, scale-sweep, Fisher row weights,
calibrated activation clipping, and NVFP4 joint-scale optimization.

The exporter should wire `export_batched_gptq.py` into the 3D path
and thread the same activation metadata used by dense Linears.
Format-specific JSO should stay disabled where the measurement does
not justify it.

### 12.3 Cost provenance

Cost payloads can record `cost_source`, `output_mse_measured`, and
`fisher_output_mse`, and allocator candidate construction prints a
`cost-source usage` summary. The remaining provenance gap is that
`PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR` is still an env-var switch.
If Fisher-output allocation returns to production use, it should be
a normal CLI flag and should be copied into the emitted artifact
metadata.

### 12.4 Critical-path test coverage

The most concerning remaining test gaps:

| Module | Issue |
|--------|-------|
| `allocator_solver.py`     | Promotion order is covered; DP recurrence still needs edge-case tests. |
| `kl_measurement.py` (5555 lines) | One small test (override-cache). |
| `sensitivity_probe.py`    | No direct test; covered indirectly. |
| `pipeline.py` (1155 lines) | Stage/component registration untested. |
| `streaming_model.py`      | No direct test. |
| `production_recache.py`   | No direct test. |
| `build_production_cache.py` | No direct test. |
| `INT4_W4A16_g128` format  | Not exercised by any test. |

Highest-value additions: `allocator_solver.solve_allocation`
edge cases, a tiny `_packed_experts_forward_with_weights` fixture,
and a small end-to-end pipeline-contract test that runs probe →
cost → allocator → export → vLLM smoke on a fixture model.

### 12.5 Duplicate utilities and large modules

Several small utilities are reimplemented across modules:

-   `_env_flag`, `_env_int`, `_env_float` defined inline in
    `production_weight_cache.py:953-973` instead of imported from
    `memory_management.py:20-52`.
-   `_canonical()` wrapper duplicated in `mse_promotion.py` and
    `sensitivity_response.py`.
-   `_shape_from_stats` duplicated in `allocator_solver.py`,
    `allocator_candidates.py`, `mse_promotion.py`,
    `decision_units.py`.

These should be centralised in `prismaquant.schemas` or a small
allocator utility module when the surrounding code is next touched.

### 12.6 Sizing of large modules

These are the largest modules and likely candidates for careful
decomposition (size alone is not a flaw; coupling is):

| Module | Lines | Notes |
|--------|------:|-------|
| `export_native_compressed.py` | 6845 | 19-branch `_quantize_2d` dispatch could delegate to `format_registry.quantize_dequantize` for RTN-only formats. |
| `kl_measurement.py` | 5555 | CUDA-graph utilities + tail-layer replay + lane batching could each be a module. |
| `kl_sensitivity_probe.py` | 3708 | Production-cache integration + L3 neighborhood + frontier solver are three concerns. |
| `incremental_probe.py` | 3492 | Streaming shard scheduler + Fisher accumulator wrapper + multimodal probe. |
| `production_weight_cache.py` | 2703 | Activation collector + render scoring + LRU + prefetch — cohesive but large. |

None of these are emergencies. They are debt to retire incrementally
as the surrounding code gets touched.

---

## 13. Open Questions and Roadmap

### 13.1 The 122B 5.0 bpp result and the benchmark sensitivity gap

The 5.0 bpp Qwen3.5-122B artifact has MSE scores that **collapse**
relative to the shipped 4.75 bpp baseline — in some Linears by
41× — yet downstream zero-shot benchmarks register comparable
numbers. This is **principled** for two reasons.

First, the rate-distortion principle. A 0.25 bpp lift on a 122B
model is ~3.8 GB of extra precision, allocated where the allocator
priced its marginal Δloss highest. The MSE collapse on those
specific Linears is *expected*; that is exactly what the allocator
was minimising. The benchmark scores not moving means that those
particular Linears were not on the critical path of the
greedy-decoded answers in the benchmark suite. Both can be true
simultaneously.

Second, the measurement-discipline rule in
`docs/design_guidelines.md`: *"A candidate that improves
calibration KL but regresses held-out PPL/mean NLL or downstream
log-likelihood checks should remain research-only unless there is
a documented reason to prefer the KL tradeoff and the user
explicitly accepts it."* For the 122B 5.0 bpp case, calibration
KL improved, MSE improved 1-41×, downstream zero-shot did not
regress, and bpp went up 0.25. That meets the gate.

**Open question:** at what bpp does the marginal MSE improvement
stop showing up in *any* held-out task suite? The right way to
answer this empirically is to add **task-suite probes with known
sensitivity to micro-quality** to the validator. Candidates:

-   Long-form perplexity on out-of-domain text (≥4k context).
-   Counting/arithmetic chain-of-thought (sensitive to
    output-distribution sharpness).
-   ToolEvalBench (already used for materialized artifacts; expand
    coverage).
-   Calibration of pass@k vs greedy on coding tasks.

This is the right thing to add to the artifact registry's
quality manifest.

### 13.2 Productionizing grouped-KL for MoE — SUPERSEDED (grouped-KL archived 2026-05-28)

> **Moot.** Grouped-KL was walled off after it lost the shipped vLLM A/B on
> dense 27B (see §3.3 and `archive/grouped_kl_2026-05-28/`), so there is no
> dense win to extend to MoE. The dense screen never survived the serving
> contract; do not pursue the MoE extension below until grouped-KL is first
> re-validated under vLLM serving on a dense model. The sketch is retained
> only as a record of the original (now-defunct) plan.

The grouped-KL screen result was on dense 27B and 4B. MoE expert
fusion structure differs; the obvious analogue (per-expert group
KL) is meaningless because experts route per-token. The (defunct) candidate
formulation was **routing-conditional** grouped-KL:

> Given the routing pattern observed on calibration, for each set
> of {layer-routed-experts}, measure end-KL when that whole
> set is quantized to a candidate format. Distribute back per
> expert weighted by routing probability.

Whether the inter-expert "damping" that justifies grouped-KL on
dense attention QKVO has any analogue under routing is an open
empirical question. The natural first experiment is a 35B-A3B A/B
holding cost surrogate constant and varying only grouped-vs-per-Linear.

### 13.3 Budget-neutral swaps

Budget-neutral swaps are research-only, not part of the production
allocator. The n=8 smoke showed that the selector correctly rejected
the surrogate's proposals, but that is not yet evidence that the
proposal generator is useful.

**Next step:** improve the swap candidate generator and prove it on a
fixed held-out calibration contract before reintroducing any
budget-swap policy to production.

### 13.4 Activation-aware MoE export

Closing the **render half** of the MoE measurement gap (the
**measurement half** is handled by `_packed_experts_forward_with_weights`):

-   Wire `export_batched_gptq.py` into `_quantize_3d_packed` so
    packed-MoE experts use GPTQ + scale_sweep when activations are
    cached.
-   Thread `fisher_row_weights` and calibrated `activation_max_abs`
    through the 3D path.
-   Add `joint_scale_opt` support for packed NVFP4.

This is the largest known unexploited quality lever in the
codebase. The math is identical to the dense path; the wiring is
a localised refactor inside the exporter.

### 13.5 Test pyramid

Critical-path tests to add, in priority order:

1.  `allocator_solver.solve_allocation` happy-path + edge cases
    (single-Linear, identical-cost ties, infeasible budget,
    promotion-overshoots).
2.  `_packed_experts_forward_with_weights` correctness on a tiny
    synthetic 2-expert MoE.
3.  End-to-end pipeline integration test on a tiny fixture model:
    probe → cost → allocator → export → vLLM smoke.
4.  Promote `production_weight_cache.py` LRU semantics from
    smoke-only to assertion-based.

### 13.6 Plugin contract hardening

Two small refactors would strengthen the seam:

1.  Factor `_MtpModule` between Qwen3.5 MoE and dense profiles into
    a shared mixin or builder. The 95% duplication is just waiting
    to drift.
2.  Document the `build_mtp_module` contract: input is a text-only
    config, output is a `nn.Module` whose forward signature matches
    the live model's MTP layer. Add a test that instantiates each
    profile's MTP module from a synthetic config.

---

## Closing Note

Every load-bearing decision in PrismaQuant has a principle behind it.
Some principles are abstract (rate-distortion, second-order Taylor,
empirical Fisher); others are practical (single cache mechanism,
GPU-bound hot path, vLLM-served output). Where the code drifts from
these principles, the fix is usually a clarification at the existing
shared layer, not a re-architecture.

The 5.0 bpp 122B artifact is the result of these principles working
correctly: each extra bit was allocated where it most reduced
predicted loss, the measurement gate confirmed real KL fell, and
the benchmark suite at this resolution registered no regression.
That the MSE improvement is **invisible to greedy zero-shot
benchmarks** is not a refutation of the approach; it is a
limitation of the benchmark suite, and the right response is to
add measurements sensitive enough to register the change.

Each extra bit goes where it best improves accuracy. The rest of
this document is the technical rationale for how we know that.
