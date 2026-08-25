# Math re-underwrite — 2026-08-21

Date: 2026-08-21
Branch: `audit/math-reunderwrite-2026-08-21` (at `merge/proven-rescues` @ `56f0a8c`; gridbook companion at `perf/kernel-eval-2026-08-21` @ `ac39366`)
Status: COMPLETE (numeric verification + derivations + new-math proofs landed)
Scope: every load-bearing mathematical artifact in this repository, re-derived from first principles and cross-checked against the implementations that ship it.

Method. Each artifact received: a first-principles derivation, an implementation cross-check (file:line at the branch point), an independent numeric verification executed by scratch scripts outside the repo (reports and scripts were written under `/tmp/opencode/{reports,matheval}/`, a volatile path; the 2026-08-21 22:30 snapshot lives at `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/{reports,matheval}/`), and a pinning-test status. Verdict taxonomy: **VERIFIED** (derivation and implementation agree; numeric check passes), **CORRECTED** (a defect was found and fixed in this branch, failing-test-first), **ASSUMPTION-GAP** (math is correct given stated assumptions; the assumption's domain is now documented and where feasible bounded), **PARTIAL** (verified with a scoped finding).

Companion documents: gridbook-side math review and new proofs live in the gridbook branch (`docs/audits/math_review_2026-08-21.md`); paper edits are applied to `paper/main.tex` in this branch with `tests/test_paper_claims.py` updated constraints respected.

---

## §1 The cost chain

### 1.1 KL-Fisher probe construction — VERIFIED

Module contract (`kl_fisher.py:3-13`): forward KL locally is `0.5·dzᵀF dz`, `F = diag(p) − ppᵀ` at `p = softmax(z/T)`.

- Probe direction (`kl_fisher.py:105-135`): `r = √p⊙ε − p·⟨√p, ε⟩` with Rademacher or Gaussian ε.
- Covariance identity: `E[εᵥε_w] = δᵥw` gives `E[r rᵀ] = diag(p̃) − p̃p̃ᵀ` exactly (centering term cancels one of two cross terms; verified algebraically and by MC over 4M vectorized probes).
- **Temperature accounting**: gradients are taken w.r.t. unscaled logits while the scalar uses `scaled = z/T`, so autograd delivers a `1/T` per factor: `T²·E[rrᵀ] = diag(p̃) − p̃p̃ᵀ`. At T=1 the module-docstring form holds as written; the general law is the T²-scaled one. Pinned in `tests/test_math_reunderwrite_pins.py`.
- Token normalization: the `1/√token_count` division (`kl_fisher.py:134`) makes `E[rrᵀ]` the MEAN over selected positions of per-position Fishers (√N-sum alternative excluded ~300× numerically); the micro-batch override path inflates by exactly `√n_microbatches` if misused, matching the docstring warning (`kl_fisher.py:94-99`; override inflation ratio 3.000 confirmed for n_mb=9).
- `fisher_quadratic_form` equals the matrix form `0.5·dz̃ᵀ(diag(p̃)−p̃p̃ᵀ)dz̃` to ≤~6 ulp float32 on non-cancelling draws (median <1 ulp; the identity is exact in real arithmetic, and under near-cancellation inputs the two fp32 evaluation orders diverge arbitrarily — consensus review 2026-08-21, `verify_d_ulp.py`); the earlier "exact at dyadic T only" observation is not resolvable above the fp32 accumulation floor and is treated as noise, not structure.

### 1.2 The ½·H_trace·MSE_W collapse — VERIFIED, ASSUMPTION-GAP BOUNDED

Derivation chain (`allocator.py:13-48`): second-order expansion → diagonal-Fisher replacement (HAWQ-V1 style) → uncorrelatedness collapse to product form. The derivation as written is honest about each step; no spurious `d_out` factor remains (:44-48 documents its removal).

Numeric characterization (cost-chain verification report):
- i.i.d. quantization error: estimator unbiased, ratio 0.986–0.992 across error scales.
- Fisher-correlated error at equal MSE_W: mispricing **17.6× / 7.9× / 16.0×** (∝F̂ concentration / top-decile / off-diagonal eigenvector directions).
- **Fisher-convention trap** (documented for posterity): the collapse is exact only with the *base-point* Fisher delivered by kl_fisher probes; a squared-gradient "empirical Fisher" evaluated at perturbed points scales as ‖ΔW‖² and errs by ~10⁴×. Any future "cheaper probe" proposal must respect this.
- New closed-form envelope (see §6/NM-3): rearrangement inequality gives the sharp instance band for the ratio `E[ΔL] / (½·H·MSE_W)`, plus iid variance `sd = √(γ+2)·cv(F)/√N`.

### 1.3 AURA estimator — VERIFIED

`aura_cost.py:8-22`: `predicted_dloss = ½·(1/K)Σₖ⟨gW⁽ᵏ⁾, ΔW⟩²`; unbiasedness is Prop 4 of the paper (algebra checked line-by-line; correct). Stderr uses sample variance `(s4 − K·mean²)/(K−1)`; population form understates by exactly √(K/(K−1)) = 3.28% @K=16, 1.60% @K=32 (verified to 2e-15). Zero-cost passthrough formats guarded (`aura_cost.py:419-438`). dW provenance stamped rendered-vs-RTN; RTN-vs-rendered moved FP8 allocations +36% served KL historically (`aura_cost.py:620-622`).

### 1.4 P5a activation-fair pricing — VERIFIED

Geometric-mean family penalty `penalty = 2^(mean log₂ ratio)` (`activation_fair_pricing.py:27-32`): gain-invariant (≤1 ulp), invariant to common scaling of both branches within a family, cannot reorder rungs within a family, kill switch restores exactly 1.0 with branch label `weight_only_kill_switch`. Known recorded bias (A-side rung-independent vs W-side shrinking) surfaced via `rung_dependence_log2_range` (:114-123). Application precedence prevents double-counting on aggregated rows (`per_row_pricing.py:3-9`).

### 1.5 AQUA activation term — VERIFIED

Diagonal expansion `dLoss_a ≈ 0.5·Σₒ g_sq[ₒ]·(W[ₒ,:]²·var_dx)` (`format_cost_protocol.py:48-57, 300-340`) correctly uses `g_sq_sum` (output-space Fisher diagonal), never `h_trace` (input-energy double-count guard; pinned by `test_activation_dloss_uses_g_sq_sum_not_fisher_row`). Packed-expert variant normalizes g_sq by GLOBAL tokens but var by ROUTED tokens (`format_cost_protocol.py:360-366`) — the asymmetry is deliberate and correct (rare-expert double-discount avoidance).

Block-scaled exact model (`format_cost_protocol.py:581-647`): `E[M²]` computed by Gauss-Legendre evaluation of the tail integral `∫2u(1−∏erf(u/(σⱼ√2)))du`; the erf table already embeds the 1/√2 ("callers must NOT divide again" — inflating E[M²] by exactly 2×); the `sqrt(2 ln 2G)` shortcut is wrong by +52%/+46% against simulation (module-documented; consistent with our independent reading). Faithfulness ladder measured→synthetic→analytic with recorded calibration ratios.

### 1.6 Expert empirical cost & RD-law ladder — VERIFIED

Unit KL = mean-token forward KL(BF16‖unit-quantized) with everything else at source precision (`expert_empirical_cost.py:398-417`); split ∝ n_params reassembles exactly (`:1413-1419`). Ladder machinery: planted `D(k)=F+C·2^(−bk)` recovered ≤2e-13 rel by `_fit_floor_law`; `_cb_ladder_rate_factor` hand-values bitwise (incl. `R(4m)=4·2^(−m)`); holdout tolerance reproduced exactly; all four fallback branches fire correctly; negative floors refused / origin-refit engaged. Opt-in + holdout-gated status unchanged.

### 1.7 Anchored cost extrapolation — VERIFIED

Log-log shape fit with per-unit centering; price = anchor level × within-segment ratio (`anchored_cost.py:1043-1095, 1287-1289`). Full receipt→fit→price pipeline recovers linear truth to 9.3e-15 and prices ≡ anchor·10^(ΣβΔf) bitwise. Curvature bias is exactly linear-in-γ in log space (ratios 2.000000/3.000000 constructed); deep-rung extrapolation reaches −81%/+425% at γ=±0.03 — the reason anchor-distance reporting exists (:1429-1459). Double-Fisher-application receipts refused at construction and table assertion.

## §2 Encoders

### 2.1 CB weighted-VQ — VERIFIED

Objective `Σⱼ wⱼ(xⱼ−cⱼ)²` per codeword, argmin via dropped-constant identity `term2 − 2·term1` (`nvfp4_cb_formats.py:227-296`); moment identity err(s)=C−2sB+s²A exact to 4e-15 rel float64; C-drop preserves argmin 1536/1536 draws. WLS refit `s* = Σwgv/Σwg²` is the exact per-group minimizer (calculus residual ≤1.9e-16); monotone strict-< acceptance makes the iteration non-increasing. Scale sweep candidate 0 ≡ snapped one-shot for both grids ⇒ never-worse guarantee holds **for the sweep's internal baseline**; note the export codec's effective group scale can differ from candidate 0 by up to 8.3% (its own e4m3 ratio quantization) — the guarantee does not span that boundary, and the module never claims it does.

### 2.2 Ceil-first sub-index split — VERIFIED

`cb_layout.bit_split` exhaustive k∈[12..48] × n_sub∈{1,2,4}: parts sum to k, non-increasing; pins (13,2)=(7,6), (29,4)=(8,7,7,7), (33,4)=(9,8,8,8). Serialization packs sub0 at offset 0 (`nvfp4_cb_formats.py:3405-3411`); consumer-side oracle and CUDA SubSplit agree bit-for-bit.

### 2.3 Two-tier scale coding — VERIFIED + STRUCTURE THEOREM (new math)

Legality = finite ∧ (0,448] ∧ e4m3-round-trip ∧ fp64-exact (`nvfp4_cb_formats.py:1141-1163`). Independent enumeration: legal mask 252/4096 pairs covering ALL 126 positive finite E4M3 values (119 normals + 7 subnormals), each value via EXACTLY TWO codes ((E,c≥8) ≡ (E+1,c−8)) because the sub-table spans two octaves. New-math deliverable NM-4 proves coverage, the exact 2-to-1 structure, and that the exception set at range edges is provably EMPTY (legal-E window strictly interior). Encoder determinism despite double coverage holds by strict-< + first-legal-wins policy (`:1273-1277`) — canonical bytes are policy-relative, now documented as such.

### 2.4 UE8M0 shared exponent — VERIFIED

frexp integer arithmetic (`mx_formats.py:186-193`): smallest-nonclipping inequalities hold over all adversarial sweeps; naive `ceil(log₂(amax/448))` diverges when fp32 log2 collapses onto an integer just above powers of two (constructed case amax=7168.00048828125 saturates under naive, frexp returns naive+1); divergence rate magnitude-dependent, consistent with the module's ~1/4e5 claim under log-uniform draw. Block128-source losslessness pinned exhaustively elsewhere (`tests/test_mxfp8_ue8m0.py`).

### 2.5 GGUF ports — VERIFIED

`_round_half_away` = half-away-from-zero confirmed against independent reference bit-for-bit; constructive tie block flips 1022/2048 bytes under half-even — the rounding choice is load-bearing, not cosmetic. `_fp16r` super-scale round-trip changes final bytes on constructed values (q=11 vs 12) — pins why the fp16 storage rounding must stay in the encoder. imatrix weighting `qw·√(σ²+x²)` matches ggml conventions; Q8_0 ignores imatrix by reference semantics.

### 2.6 IQ grid encoders — VERIFIED

Exhaustive GPU argmin vs llama.cpp neighbor heuristic; WLS refit `db←Σwyg/Σwg²` is what makes re-quantization a fixed point (bitwise idempotence 40+/40+ runs; skipping the refit drifts IQ3_S on 19/20 seeds). Grid tables lifted from decoded gguf-py tables; pack≡emulation pinned.

### 2.7 GPTQ/LDLQ atoms — VERIFIED (structural)

Shared Cholesky-inverse convention `UᵀU = H_damped⁻¹` across all four GPTQ implementations (CB atoms :158-214, NVFP4 lane :2974-2978, batched twin, GGUF lane :91-93); dead-channel policy identical verbatim everywhere (detect before damp, exclude from damp mean, identity entry, weights never zeroed). LDLQ atom recurrence is GPTQ's block-form update lifted to atom granularity (`cb_ldlq_atoms.py:751-784`); col_weights deliberately excluded from the LDLQ metric (no double-imatrix); holdout gate replaces the anti-correlated in-sample gate.

## §3 Selection & accounting

### 3.1 Knapsack DP — VERIFIED optimality; CONTRACT CORRECTED (documentation)

DP == brute-force optimum on 69/69 instances incl. degenerates (max Δ 3.6e-15). **Finding F1**: raw `solve_allocation` returns assignments whose achieved bits exceed target (13/69 random instances; max overshoot 0.036 avg-bpp). Investigation verdict: *docstring-only* defect — the function bounds CHARGED BINS, feasibility deliberately lives upstream in `solve_with_promotion`'s overshoot ratchet (`allocator_solver.py:944`) and the byte-budget filter; the only live raw caller (`allocate_from_card`, `sensitivity_card_allocate.py`) documents the None/infeasible contract and relies on the same downstream feasibility enforcement. Fixed in this branch: true contract documented with the PROVEN overshoot bound `achieved − target ≤ bit_precision·(n_units+3)/2` (derivation: per-unit slack |ρ|≤½bp except min-1-bin regime; total charge ≤ ⌊E/bp⌉+1 columns ⇒ n·½bp + 1.5bp; empirically never violated in 400 fuzz instances, near-worst construction reaches ~97% of bound). Pinned in tests.

### 3.2 Charged-bin discretization — VERIFIED, COST MEASURED (consensus review)

Conservative clamp: any strictly positive delta ⇒ ≥1 bin (overcharge ≤1 bin, worst observed 0.99983); `(0.0005, 0.001) → 1` via the clamp riding on banker's round(0.5)=0 — pinned by value so the interaction cannot silently change. Backtrack mirroring necessity upgraded to sufficiency theorem (NM-5): any consistent decrement schedule reconstructs AN optimal solution; zero-decrement reconstruction shown infeasible on a constructed instance.

**Finding F8 (consensus review 2026-08-21, `pqcode`):** the clamp is load-bearing for shipped bytes. On the real cards at production `bit_precision = 1e-4`, sub-half-bin candidates are 672 of 6 080 (Qwen3.8-27B, 48 units) and 2 021 of 4 626 (DSv4 92 GB, 301 units — 43.7% of the menu, because aggregated expert super-items have small share-scaled deltas); each such candidate is charged 0.55–0.97 of a bin more than its true delta. Re-running both recorded allocator invocations with `_charged_bins` replaced by plain `int(round(d/bp))` (baseline runs reproduce both shipped `selection.json` files bit-exactly first) moves 97/496 Qwen buckets and 47/301 DSv4 buckets and lands **−0.9 MiB / −33.3 MiB** of achieved footprint at **lower** predicted dloss (−8.7e-5 / −2.3e-4) — the no-clamp trajectories Pareto-dominate the shipped points. The under-fill direction does not materialise (byte-budget bisection compensates; both runs stay feasible); the cost is a distortion of the DP's per-unit economics. Scripts: `/home/rob/dq-runs/review-watch-2026-08-21/ox/pqcode/`. Status: research follow-up (a uniformly conservative `ceil` charge, or a finer bin, is the candidate replacement); no default changes until a served A/B.

### 3.3 Dual intervals — VERIFIED

Weak-Lagrangian support condition `loss(s)+λB_s ≤ loss(c)+λB_c`; interval endpoints match hand computation ([0.5,1.5] anchors); dominated-selected emptiness ([0,−0.5]); inside-holds/outside-fails across 50 units; empty intervals legitimate off-Lagrangian-hull (documented in-module at :507-514). Downstream-consumption gap noted: no test ties a budget decision to interval boundaries (registered as follow-up, not a defect).

### 3.4 Footprint / bpp — VERIFIED

Floor identity `artifact = floor + body`, `floor = source_total − reencoded_spans` hand-exact (7820 = 2048 + 5772 synthetic); double-charged spans, assignable-prefix exclusions, empty partitions all raise. NVFP4 global sidecars +4/+4 per Linear additive; packed experts ×E×n_proj. CB payload via `nvfp4_cb_footprint` schemas v2/v3/v4 with type_size law 4k+16 | 4k+9 | 4k (§4 cross-check).

### 3.5 Read traffic — VERIFIED

Model `read_bytes/token = Σ stored×p`, p(topk/E) exact under per-layer-uniform routing (skew redistributes identity, not expectation); dense/held_fixed 1.0; excluded classes 0.0; unknown names land held_fixed (over-count direction deliberate and safe for a bandwidth figure). Hand ledger byte-exact; reconciliation against footprint before probabilities applied.

### 3.6 MTP rung selection — CORRECTED (real bug, fixed)

Acceptance chain `a(b)=a_inf−β√E(b)`, Pinsker scoping, throughput objective verified as documented (`docs/design/mtp_rung_selection.md`); discrete argmax matches brute force. **Finding F2**: the Lambert-W continuous optimum silently self-disabled via float overflow in `exp(g_over_c)` for `(t+d0)/c ≳ 1022.6` — ~41% of the parameter range the audit treated as plausible (an illustrative share — the threshold itself, `(t+d0)/c ≈ 1023`, is exact) INCLUDING the recorded Hy3 constants (1260) — indistinguishable from "no scipy"/"no real solution", masked by fixed-point fallback. Fixed failing-test-first: algebraic rescale to log-space (`log_M = ln((1+a_inf)/β) − g_over_c`), direct scipy for representable args, exact Newton continuation of W₋₁ on `s − ln s = L` below float64 range, and a loud `continuous_bstar_lambertw_status` provenance field recording which solver answered. 6 new tests incl. overflow-regime survival and status handover.

### 3.7 Saturation / frontier selection — VERIFIED + ADMISSIBILITY CHARACTERIZED

Band rule `kl_m ≤ kl_hi + z·hypot(se_m,se_hi)` dense-exact 60/60 noisy curves. Dense vs bisection asymmetry formalized (NM set §2): dense is exact relative to its rule by construction; bisection hides earlier saturated points on non-monotone dips (measured 3-vs-5 evaluations hiding bpp=1) and diverged on 17% of merely-noisy monotone-mean curves; admissibility certificate = thresholdness of measured margins — monotone means are NOT sufficient. Pipeline default remains dense/auto-conservative.

### 3.8 Serve constraints — VERIFIED

Additive model `phase_ms = reference·Σ share·relcost` hand-exact (440.58/24.36); unpriced phase refuses feasibility; `p05_TPS = 1000/p95_ITL_ms` same-float; mix sums enforced to 1e-6; no-λ constraint intact. Assumptions A1-A8 remain honestly declared; dispatch-table provenance mandatory-or-refuse unchanged.

## §4 Gridbook companion (summary; full document in gridbook branch)

Byte-law triple agreement (producer cb_layout ↔ packaged contracts md5-identical ↔ gridbook runtime_contract validator): VEC_DIM/SUPERBLOCK/CODEWORDS/INDEX_BYTES, layout versions, scale planes {16,9,0}, type_size laws, format families/rungs/n_sub — all fields match. Compose table bitwise-equal across three implementations. Window extraction oracle agreement at k∈{13,17,20,24}. Kernel structural asymmetries (dense-unconditional third word :839/:866 vs grouped-predicated :1383; scalar codebook loads :766-778 vs uint2 :1415; byte staging :830/:860 vs u64 burst :1354-1362) line-confirmed — feeding optimization B1 (landed in gridbook branch, wins on all 32 measured points, bit-exact).

ρ-threshold lemma: padding lemma + tile identity B(128)=2B(256)−q verified on 20k histograms (with the premise correction that q counts residues in [1,128]); analytic family `ρ > 128(1+256/x)` proved TIGHT (not merely sufficient); shipped 512 certifies x̂ ≥ 85.3; advantage profile NON-MONOTONE (uniform routing loses again over ρ∈(~261,~373) after first winning ~133) — any routed calibration sweep must sweep past first crossings. Empty-expert cost bounded at ~0.59 ns vs ms-scale cells. Staging vectorization byte-neutrality theorem (Theorem 15) preconditions are stated in-source by the proposed B2 — which was REJECTED on measurement (whole-operator +7…+12% at the shipped DSv4-dominant k=12 in two independent A/B series; the theorem stands on its own).

## §5 Paper consistency

Props 1 & 4 verified correct as written (ε-counting; E[⟨g,dW⟩²] step exact given deterministic dW) — no edits. Applied in this branch: Prop 5 full four-move proof (replaces sketch); Prop 6 restated over the segment LP with full proof, empirical clauses moved to signposted prose; §5 reconciliation splice citing the validated log-linear fit (log₁₀KL = −0.1133 − 0.2357·bpp, R²=0.9948, 4.50–8.25 bpp, kneedle axis-dependence 4.75-vs-6.00); single provenance footnote covering the dispersion/fidelity constant cluster. All `test_paper_claims.py` literals preserved (suite green); pdflatex compiles clean.

## §6 Findings and actions

| # | Finding | Severity | Evidence | Action |
|---|---|---|---|---|
| F1 | solve_allocation overshoot undocumented | LOW (contract only) | verifier 13/69; investigator trace | FIXED: docstring + proven bound + pin test |
| F2 | MTP Lambert-W silent overflow disable | MED (silent fallback) | overflow site exp(g_over_c); Hy3 ratio 1260 dead | FIXED: log-space rescale + Newton continuation + loud status |
| F3 | Two-tier scale codes 2-to-1 per value | INFO (structure) | 252 pairs / 126 values | PROVEN (NM-4); canonicity policy-relative documented |
| F4 | CS-bound suggestion itself unsound as stated | INFO (audit-method) | violated on 10.7% of ∝F̂ draws | REPLACED: two-term diagonal+off-diagonal bound (NM-3) |
| F5 | Fisher-convention trap (base-point vs perturbed empirical) | INFO (documented hazard) | ~10⁴× discrepancy | DOCUMENTED §1.2 |
| F6 | ρ>512 threshold: tight bound derived; non-monotone profile | INFO (calibration guidance) | gridbook NM Thms 4-7 | Documented; routed-sweep acceptance test specified |
| F7 | Dual-interval consumers untested downstream | LOW (gap) | pin-mapper gap list | Registered follow-up |
| NM-1..5 | New theorems (envelope bounds, backtrack sufficiency, saturation admissibility, two-tier structure, window/spill characterization) | — | `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/reports/new_math_*.md` (snapshot) | LaTeX-ready; paper-appendix candidates flagged |

Verification artifacts: `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/reports/{encoder,selection,cost_chain}_verification.md`, `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/reports/gridbook_math_verification.md` (snapshots of the volatile `/tmp/opencode/reports/` originals), in-repo: `docs/audits/math_reunderwrite_2026-08-21_proofs.md` (NM-1..5 full LaTeX-ready proofs); companion proofs on the gridbook branch (`docs/audits/math_review_2026-08-21_proofs.md`); working reports in the same snapshot directory.
