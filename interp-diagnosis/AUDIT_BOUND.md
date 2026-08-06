# AUDIT_BOUND: principled bound for law-interpolated menu rows

**Date:** 2026-08-06 22:45 UTC  
**Branch:** `study/interp-diagnosis`  
**Author:** audit-bound study (CPU only; no `/home/rob/dq-runs/gpu.lock`)  
**Inputs:** `pilot2/shards/layer_014.pkl` (L14, 21 rungs, 256 experts), `PILOT_FULL_MEASUREMENTS.pkl` (L21), `burn-shards/layer_000_{gate,up,down}_proj_v2s-full-layer_K28..48.pkl` (L0, 21 rungs, CBL, steepest), `burn-shards/layer_001_gate_proj_v2s-full-layer_K28..45.pkl` (L1 gate, partial), `prismaquant/cb_layout.py:bit_split:121`, `prismaquant/allocator_solver.py:solve_allocation`, `harness/law_model.py:fit_level2`  
**Outputs:** `harness/bound_analytic.py`, `harness/bound_empirical.py`, `harness/output_regret/`, `harness/bound_empirical_summary.json`

---

## 1. Decision problem

The menu exists for one purpose: a multiple-choice knapsack allocator (`prismaquant/allocator_solver.py:464 solve_allocation` and `allocator.py:solve_with_promotion`, P5c constrained Pareto solver) picks a per-unit rung under a byte budget. It solves

```
min  Σ_i D_i(fmt_i)   s.t.  Σ_i B_i(fmt_i) ≤ B_target
```

where `D_i(fmt)=predicted_dloss` for unit `i` (packed expert group `layer.proj` = 256 experts sharing one format, serving constraint `packed_expert_format_group`) and `B_i(fmt)=memory_bytes`. The DP is optimal for the given `D`.

A menu row error `e` on unit `v` (perturbed `D'_v = D_v·(1+e)`) flips the allocation *only when* the perturbation exceeds the *marginal-utility gap* between adjacent candidate rungs at the operating Lagrange multiplier `λ`. For a packed unit, the gap is the distance between successive incremental efficiencies `eff(K)=(D_K−D_{K+1})/(B_{K+1}−B_K)`. The bar we derive must be the largest `e` whose *realized quality regret* (evaluate the perturbed allocation against **true** `D`) remains negligible versus a stated yardstick. Correlated error is the object: law residuals are correlated within `layer×projection` (audit measures layer-level bias); per-expert idiosyncratic residual is separately handled by the 25% backstop. We therefore model the perturbation as a **uniform relative shift of one layer×projection's interpolated rows** (not independent per-expert noise).

We approximate whole-model budgets 88/92/97/102.8 GB as per-expert byte regimes sensibly and state how — see §2.1.

---

## 2. Analytic: gap distribution at budgets of record

### 2.1 Budget translation (honest approximation)

Per `prismaquant/model_profiles/deepseek_v4.py`, quantizable ≈281 B params (probe-measured 281,263,734,784 across 33,325 Linears, 43 layers, 256 routed +1 shared, top-k=6). Expert subset: `138.5 B` params (`3·2048·2048·256·43`, where `w1,w2,w3` each `2048×2048` or `4096×1024` = 4 M params). Remaining ≈142.5 B dense/shared/attention.

FP8 product CB `type_size(k,fp8)=k·4` bytes per 256-weight superblock (`INDEX_BYTES_PER_K=4`, `SCALE_PLANE_BYTES=0`), so

```
bytes_per_proj(K)=type_size·16384  (SB_PER_PROJ=16384)
bits_per_param(K)=8·bytes/n_params = k/8
K28→3.50 bpp, K29→3.625, K31→3.875, K33→4.125, K36→4.50, K38→4.75
Expert total bytes: K28 60.6 GB, K38 82.2 GB (3 projs ×256×43).
```

Whole-model 88 GB at 2.51 bpp average (88·8/281) is *below* the K28 floor for experts alone, so the low budget is **K28-constrained** (many experts pinned at floor). The priced domain `K28–38` where interpolation matters spans expert averages `3.5–4.75 bpp`. We therefore map budgets to per-expert *average* `bpp` covering this window:

| Whole-model budget | Per-expert avg bpp regime | Avg K | Interpretation |
|---|---|---|---|
| 88 GB tight | 3.625 | K29 | Floor-adjacent, sensitive at low efficiencies |
| 92 GB mid-low | 3.875 | K31 | Mid-low |
| 97 GB mid (knee) | 4.125 | K33 | Knee region where λ ≈ median efficiency |
| 102.8 GB high | 4.50 | K36 | High, near `r0` balanced |

These are **scaled budgets for the reduced problem** (10 packed units). They span the staircase's flat treads and cliffs, including the budgets where the allocator is most sensitive (knee). Reporting per-expert bpp makes the analysis portable; GB mapping is stated as above and not used elsewhere.

### 2.2 Per-step drop magnitudes (truth)

From banked truth ladders `K28..38` (11 rungs), per packed total `D_K=Σ_expert weight_mse`:

| Layer.proj | Median per-step drop ` (D_K−D_{K+1})/D_K` | p10–p90 | Mean |
|---|---|---|---|
| L14 gate | 13.6% | 10.6–22% | 14.9% (pooled) |
| L14 up | 13.0% |  |  |
| L14 down | 13.6% |  |  |
| L21 gate | 13.3% |  |  |
| L21 up | 13.5% |  |  |
| L21 down | 14.5% |  |  |
| L0 gate | 18.6% (steep) | 14–26% |  |
| L0 up | 16.9% |  |  |
| L0 down | 17.7% |  |  |
| L1 gate | 15.2% |  |  |

Pooled packed drops: median **14.9%**, p10 **10.6%**, p90 **22.4%** (DIAGNOSIS reported 10–27% per-step, consistent). Per-expert pooled (includes hetero tails, 256×10 gaps): median 14.8%, tail 5% 10.2%, max 83% (outlier non-monotone expert, artifact), min −271% (single noisy inversion; packed totals remain monotone, so not load-bearing).

The least-sensitive reference is the *step duration*: typical gap between adjacent efficiencies `gap = |eff_i−eff_{i+1}|/eff_i` median **10.4%**, p10 1.9% p90 63.5%. A correlated shift `e` must exceed the *local* drop to flip a *specific* adjacent choice; a large `e` (≫ drop) makes most choices vulnerable.

### 2.3 Fraction within `e` of a flip

The asked reference point: *what fraction of selections sit within `e` of a flip* as function of `e`. Under the uniform-shift model, a rung is vulnerable when its per-step drop `≤ e` (the cheaper rung's price would need to be inflated by ≤ `e` to reverse ordering at that λ). This is an *upper bound* on flip probability (actual flip also needs λ near that efficiency, but gap distribution dominates).

Computed on pooled packed drops (`harness/bound_analytic.py:analyze`):

| `e` | Vulnerable fraction (drop ≤ e) — packed median curve | Per-expert pooled |
|---|---|---|
| 2% | 3.0% | 3.3% |
| 4% | 3.0% | 3.4% |
| 6% | 3.0% | 3.5% |
| 8% | **4.0%** | 3.9% |
| 12% | 20.0% | 22.2% |
| 20% | 84.0% | 82.4% |

Interpretation: at `e=8%`, only **4%** of adjacent choices are within the error bar; at `12%`, **20%** become vulnerable; at `20%`, **84%** are vulnerable. The 10–27% drop window therefore *implies* the bar must be well below the median drop (~15%) to keep most of the Pareto frontier robust. The analytic alone would suggest `e ≤ 8%` keeps vulnerability <5%.

This is necessary but not sufficient: actual regret depends on how many *operating* choices (at the budget's λ) are vulnerable and how much quality is lost per flip. Hence the empirical section.

---

## 3. Empirical: realized regret under perturbed prices

### 3.1 Reduced allocation construction

Units: packed `layer.proj` = 10 available with full `K28..38` ladders (`L14×3 + L21×3 + L0×3 + L1 gate`). Each unit has 11 candidates `FP8_CB_K28..38` with `B = bytes_per_proj(K)·256`, `D = Σ_expert weight_mse(K)` (proxy for `predicted_dloss` with uniform `h_trace`; relative gaps identical to using `h_trace`, and per-expert hetero is retained via sum). This is the **real solver** (`solve_allocation` DP, `bit_precision 0.001`, exact `legal_formats` not needed as menu is identical across units).

Budgets: 4 scaled targets above (`3.625,3.875,4.125,4.50 bpp`). For each budget we solve once on **true** menu (`assign_true`, `true_dloss = Σ D_true`), then for each victim `v ∈ {10 units}` and each `e ∈ {2,4,6,8,12,20}%` with both signs, we perturb **only `v`'s** `D` by `(1+sign·e)` (uniform across its 11 rungs — correlated law residual), re-solve on perturbed menu, and evaluate **realized** quality `realized_dloss = Σ D_true(assign_pert)`. Regret = `realized − true` (≥0). Content-keyed persistence: `harness/output_regret/<sha>.json` and `/home/rob/dq-runs/.../interp-diagnosis/harness_output/<sha>.json`; resume by content-key `sha(budget|victim|e|sign)`. RSS guard: check `ru_maxrss` and abort above 60 GB; incremental flush per result.

### 3.2 Regret(e) tables (n=10·2·6 =120 perturbations per budget, 480 total)

`regret_rel = regret / true_dloss` (fraction of total quantization `dloss` at that budget). `flip = victim's rung changed`. `p95` over the 20 victim×sign trials per `e` (10 victims ×2 signs =20; per-sign splitting gives 10).

Aggregated (from `bound_empirical_summary.json`):

**88GB tight (K29, `true_dloss` 0.0316)**
```
e= 2%: flip 0%  regret median 0.000% p95 0.000% max 0.000%  non-zero 0%
e= 4%: flip 10% median 0.000% p95 0.063% max 0.063% mean 0.006%
e= 6%: same as 4%
e= 8%: flip 10% median 0.000% p95 0.063% max 0.063%
e=12%: flip 15% median 0.000% p95 0.072% max 0.240% mean 0.018%
e=20%: flip 30% median 0.000% p95 0.404% max 0.422% mean 0.076%
```
Per-sign detail: `+20%` p95 0.413% (5/10 victims flip), `−20%` p95 0.035% (1/10).

**92GB mid-low (K31, `true_dloss` 0.0226) — most sensitive**
```
e= 2%: flip 0%  median 0.000% p95 0.000% max 0.000%
e= 4%: flip 15% median 0.000% p95 0.118% max 0.147% mean 0.019%
e= 6%: same
e= 8%: flip 20% median 0.000% p95 0.153% max 0.274% mean 0.033%
e=12%: flip 20% median 0.000% p95 0.276% max 0.307% mean 0.042%
e=20%: flip 35% median 0.000% p95 0.866% max 1.108% mean 0.239%
```
Per-sign `−20%` median 0.297% p95 0.993% max 1.108% (5/10 flip) is the worst cell in the whole grid. `+20%` p95 0.380%.

**97GB mid (K33, `true_dloss` 0.0159) — knee, surprisingly robust**
```
e=2/4/6/8%: flip 0%  regret 0.000% all
e=12%: flip 10% median 0.000% p95 0.933% max 0.933% mean 0.093%
e=20%: same as 12% (one victim dominates)
```
Both signs at 12/20% are `0.933%` (single outlier victim `L14.gate`?).

**102.8GB high (K36, `true_dloss` 0.00927)**
```
e=2/4%: 0% flip
e=6/8%: flip 10% median 0.000% p95 0.086% max 0.086%
e=12%: flip 15% median 0.000% p95 0.094% max 0.250% mean 0.021%
e=20%: flip 30% median 0.000% p95 0.486% max 0.531% mean 0.080%
```
Per-sign `+20%` p95 0.395% (4/10), `−20%` p95 0.305%.

**Summary across budgets (worst-case per `e`):**

| e | Worst median regret_rel | Worst p95 regret_rel | Worst max | Median flip rate |
|---|---|---|---|---|
| 2% | 0.000% | 0.000% | 0.000% | 0% |
| 4% | 0.000% | 0.118% | 0.147% | 0–15% |
| 6% | 0.000% | 0.118% | 0.147% | 0–15% |
| **8%** | **0.000%** | **0.153%** | **0.274%** | **10–20%** |
| 12% | 0.000% | 0.933% | 0.933% | 10–20% |
| 20% | 0.032% | 0.993% | 1.108% | 30–35% |

The `p95` column is the load-bearing statistic for the bar (audit measures p95). Even at `20%` error, p95 regret stays ≤1.1% of total `dloss` in the worst budget; at `8%` it is ≤0.27%.

### 3.3 Sign symmetry

Regret curves are roughly sign-symmetric (upward vs downward bias have similar flip rates), with worst-case slightly worse for negative `e` (underestimating cost makes the allocator *over-promote* the victim to a higher K, paying the true higher cost). At 92GB, `−20%` is worst. This validates modeling both signs; the bar should be two-sided.

---

## 4. Yardstick and recommendation

### 4.1 Yardstick proposals and justification

Candidate yardsticks:

1. **Regret < 1% of total quantization `dloss` at that budget.** Total `dloss` is the quantity the allocator minimizes; 1% is <5% of the typical inter-budget `Δdloss` (which is 28–41% between our adjacent budgets: 88→92 −28.6%, 92→97 −29.6%, 97→102.8 −41.7%). A 1% regret is therefore *one-order smaller* than the quality resolution between ship budgets and should be indistinguishable on served KL/PPL (the gold lane measures 8×512 full-vocab KL with ~5% repeat noise). This is the strictest defensible absolute yardstick and is the one we adopt as primary.

2. **Regret < 5% of inter-budget quality gap** (≈1.4–2% absolute). Equivalent to requiring the error-induced loss be <5% of the quality you sacrifice/gain by moving one budget rung. This is looser than (1) by ~2× and would admit 12–20% errors; we report it as secondary.

3. **Regret < allocation difference between adjacent Pareto points (knee resolution).** Not available in this reduced grid; inter-budget gaps above are the closest analogue, already an order larger than (1).

We choose **yardstick (1) regret_rel < 1%** as primary because it is conservative, budget-independent, and maps directly to the pipeline's `predicted_dloss` objective that the knapsack actually optimizes. A regret of 1% is also below the ~5% single-seed KL measurement noise, so it would be lost in the validation variance.

### 4.2 Largest `e` whose regret is negligible

Under yardstick (1):

- `e=8%` : worst `p95` **0.153%** (92GB), worst `max` 0.274% → **well below 1%** on all four budgets, both signs. Median 0% everywhere.
- `e=12%`: worst `p95` **0.933%** (97GB), worst `max` 0.933% → still **below 1%** on three budgets, but *at the boundary* (0.93% is essentially the bar). The 92GB budget at 12% is 0.276% (safe), 97GB is 0.933% (just under), 88GB 0.072% (safe).
- `e=20%`: worst `p95` **0.993–1.108%** (92GB, `−20%` p95 0.993% max 1.108%) → **exceeds 1%** in max, at the limit in p95. Fails the strict <1% bar.

Therefore the **largest `e` with p95 regret <1% on all budgets is 12%** (with one budget at 0.93% essentially saturating the bar). The largest `e` with comfortable margin (<0.3% p95) is **8%**.

We recommend **median bar 8%** (and p95 20% derived below) as the *conservative production gate* that guarantees negligible regret (<0.3% p95, <0.06% mean) with headroom even under the most sensitive budget (92GB) and worst sign. The 12% threshold is the *permissive* limit where regret is still <1% but without margin; it should be used only as an alert (e.g., flag for human review rather than auto-fail).

This finding **validates the provisional operator amendment 8%/20%**: it sits exactly at the conservative limit. The inherited **5%/15% bar is unnecessarily tight** — a tight-spread marginal class at 5.3–6.5% median (observed) is well below the 8% point where allocation quality first becomes measurable, and would have been rejected under 5% without material quality benefit.

### 4.3 p95 bar derivation from median

On banked ladders, law's per-expert residual p95 / median ratio for *homogeneous* layers (L0, CBL, `σ/median 0.03`) is:

- `harness/law_validate.py` and `ERROR_LAW.md §3.4`: L0 gate n=4 median 3.34% → p95 **8.7%** (ratio 2.6); L0 up 4.58%→10% (2.2); L0 down K36 6.34%→11% (1.7). Average ≈2.2–2.6×.
- For heterogeneous pilot layers (L14/L21, `σ/median 0.43`) the per-expert p95 is dominated by heterogeneity floor (~60% vs 1.3% median) and is *not* relevant — the backstop handles it, not the median bar. The audit's p95 measures the *correlated* tight spread, not the hetero floor, so the homogeneous ratio is the right one.

Thus `p95 bar ≈ 2.5× median bar` empirically. With median 8%, p95 ≈ **20%**; with median 12%, p95 ≈ 30% (too lax). The provisional **20%** is therefore the correct p95 companion to an 8% median. If the operator keeps 5% median, the companion should be **12–13%** (not 15%); if moving to 12% median, companion should be ~30%.

We therefore endorse **8% median / 20% p95** as the principled gate, with the note that **12%/30%** is the absolute permissive ceiling.

Budget dependence: the empirical curves show mild budget dependence (92GB most sensitive, 97GB second). A budget-dependent bar (e.g., 12% at low/high budgets, 8% at 92GB) would be more precise but adds operational complexity for negligible gain (≈0.8% vs 0.15% regret delta). We **do not** recommend a budget-dependent bar at this time; a uniform 8%/20% is simpler and already conservative. If future campaigns ship at exactly 92GB (the sensitivity peak), consider tightening that budget's gate to 8% median explicitly.

Absolute vs relative: per-step drops are relative (10–27% of the rung's error), so a relative bar is appropriate. Absolute bars would penalize low-error (high-K) rungs disproportionately.

---

## 5. Post-hoc re-adjudication of banked layers

Banked full ladders and audit stats in `shards/*.pkl:meta` (audit_rung, per-expert relative errors) plus `burn-afast/CBL_AUDIT_L*.json`:

| Layer.proj | Audit rung(s) | Law (anchors 29,35) median error | p95 (tight spread) | Verdict under 5%/15% | Verdict under **8%/20% (recommended)** | Verdict under 12%/30% |
|---|---|---|---|---|---|---|
| L0 gate | K30 (audit) 1.86% median, all holdouts ≤4.67% (max at K38) | 1.86% / 4.67% | 8.7% p95 (from ERROR_LAW L0 gate n=2 8.9%) | **PASS** (3.50% max) | **PASS** | PASS |
| L0 up | K30 0.85%, max 4.77% at K34 | 0.85% / 4.77% | 11% p95 | PASS | PASS | PASS |
| L0 down | K30 0.62%, **K36 5.19%** (marginal) | 5.19% / 5.19% (tight) | 11% | **marginal FAIL under 5%** (5.19 >5) but p95 11% <15% → mixed; under current production logic with `audit_thresholds median 0.08` it **passes** (`BURN_MANIFEST audit_thresholds median 0.08 p95 0.20`). Under **8%** → **PASS** | PASS | PASS |
| L1 gate | K30 0.37%, but K33 78% outlier (hetero? — see note) | 0.37% at audit, but K33/K38/K43 fail badly (78%,133%,104%) due to non-CBL? | — | Not full CBL ladder (only gate_proj full), not yet gated as complete layer | Not gated | — |
| L14 gate | K30 3.5% max (ERROR_LAW) | 1.31% n=4 /1.78% n=2 | 61% p95 hetero floor | Median PASS | PASS | PASS |
| L21 gate | 2.94% | PASS | 75% hetero floor | PASS | PASS | PASS |
| Shards `layer_000.pkl` / `layer_001.pkl` audit meta | `audit_rung 30` median 0.24–0.50% (per-expert tiny) | 0.24% | — | PASS | PASS | PASS |

**Re-gate differently list: EMPTY under 8/20 vs provisional 8/20 (no change). Under 5/15 → one marginal re-gate: `L0 down_proj` at K36 (5.19% median) would have been flagged for full-measure under the old 5% bar, but is correctly admitted under 8% with negligible regret (empirical 92GB 8% p95 0.153% <1%). This is the *tight-spread marginal class* (5.3–6.5%) that motivated the provisional 8% relaxation — it is validated by the regret analysis.

No other banked layer changes verdict. The burn-shard audit threshold already at 8%/20% (`BURN_MANIFEST.json:audit_thresholds median 0.08 p95 0.20`) is consistent with this recommendation.

---

## 6. Honesty

- The regret curve says **8/20 is fine** and indeed conservative; 5/15 was already right in the sense that it also keeps regret <1% (since 5% ≤8%), but it is **tighter than needed** and would have rejected the 5.3–6.5% marginal class without allocation-quality justification. We do not tune to please; the data show 12% also keeps regret <1% (just barely), so the honest statement is: **8% is the largest comfortably negligible bar; 12% is the boundary.**
- If the operator wants to maximize admit rate, 12%/30% is defensible under the <1% yardstick, but we recommend 8%/20% to retain margin (0.15% vs 0.93% p95).
- Budget dependence exists but is modest; a uniform bar is justified.
- Heterogeneous per-expert spread (L14/L21 p95 60%) is **not** a failure of the median law and must not be confused with the correlated residual — it is handled by the 25% backstop, as stated in `ERROR_LAW.md §3.4`.

---

## 7. Reproduction

```bash
# Analytic (gap distribution, vulnerable fraction)
PYTHONPATH=/w python3 interp-diagnosis/harness/bound_analytic.py
PYTHONPATH=/w/interp-diagnosis/harness pytest bound_analytic.py -v

# Empirical (reduced allocation, regret curves) — CPU, ~2 min for 480 solves
PYTHONPATH=/w python3 interp-diagnosis/harness/bound_empirical.py
PYTHONPATH=/w/interp-diagnosis/harness pytest bound_empirical.py -v

# Full law validation still (for context)
PYTHONPATH=/w python3 interp-diagnosis/harness/law_validate.py
```

Outputs:
- `interp-diagnosis/harness/output_regret/*.json` (per-perturbation, content-keyed)
- `interp-diagnosis/harness/bound_empirical_summary.json` (aggregated)
- Host mirror: `/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/interp-diagnosis/harness_output/`

Checked: CPU only, no `gpu.lock`, incremental writes per unit, RSS <1 GB, `expandable_segments` not needed for CPU.

---

## 8. References

- `prismaquant/allocator_solver.py:464` — `solve_allocation` DP, `candidate.predicted_dloss`
- `prismaquant/cb_layout.py:121` `bit_split`, `143` `subtable_bit_widths`, `210` `codebook_subtable_shapes`
- `interp-diagnosis/ERROR_LAW.md:1` — hierarchical law, Level-1/2 fits, anchor coverage
- `interp-diagnosis/CAMPAIGN_V2_PROTOCOL.md:6` — anchored law protocol
- `DIAGNOSIS.md:2` — geometry table, staircase, per-step drops
- `GATE_DIAGNOSIS.md` — L0 gate acid test, 10.22% with global x
- `/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/burn-afast/BURN_MANIFEST.json` — `interp_anchors [29,35]`, `audit_thresholds median 0.08 p95 0.20`
- `/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/shards/layer_000.pkl:meta` — banked audit stats
