# MTP draft-rung selection — canon

**Status: normative (Robert, 2026-07-20).** Every MTP/spec-decode draft module
prismaquant quantizes is assigned its rung by the throughput-optimal selector
below — not by hand, not by copying body rungs. Implementation:
`prismaquant/mtp_rung_selection.py`; the Hy3 driver
(`scripts/build_hy3_mtp_cb_inputs.py --rung-select auto`) is the reference
integration. A draft can never change outputs (rejection sampling reproduces
the target distribution exactly), so this selector optimizes **throughput
only**; no quality gate applies to the draft itself.

## 1. Objective

Per spec-decode engine cycle with `k` speculative tokens:

```
T(b) = E[tokens/cycle] / cycle = (1 + Σ_{i=1..k} Π_{j<=i} a_j(b)) / (t + k·d(b))
```

`b` = draft bits/weight, `t` = target verify-step time, `d(b)` = one drafter
forward. For `k = 1`: `T(b) = (1 + a(b)) / (t + d(b))`.

## 2. The two curves

**Cost side (exact, measurable):** `d(b) = d0 + c·b`, where
`c = active_draft_params / (8 · BW)` and `d0` is everything the rung cannot
touch — the shared (unquantized) lm_head read, attention, KV, and host/launch
overhead. On an eager drafter `d0` is host-dominated; with drafter CUDA graphs
`d0` collapses to the lm_head+glue bytes.

**Acceptance side (modeled, calibrated):** for standard speculative sampling
the identity `acceptance = 1 − TV(draft, target)` is exact. Pinsker gives
`TV ≤ sqrt(ΔKL/2)`, and the draft's quantization-induced self-KL is what the
Fisher expansion predicts: `ΔKL(b) ≈ ½ Σ_i h_i · MSE_i(b)` over the draft's
Linears — the same additive cost model the allocator runs on the body. Hence
the canonical shape

```
a(b) = a_inf − β · sqrt(E(b)),    E(b) = Σ_i h_i · MSE_i(b)
```

with `MSE_i(b)` measured per rung (never the idealized `2^{-2b}` law when real
measurements exist) and `h_i` from an MTP-aware probe when available, else
uniform weights with the `h_source` provenance field set accordingly. `a_inf`
and `β` are **fit from served acceptance measurements** (≥2 rungs; with one
point, `a_inf` is taken from the highest measured rung and the fit is flagged
single-point). Pinsker is an upper bound and greedy acceptance is
argmax-match, not `1 − TV` — both scale like `sqrt(ΔKL)` empirically, and the
fit absorbs the constants; that is why calibration is mandatory, not optional.

## 3. The selector

1. Compute `E(b)` for every rung in the draft menu (per-role rung families map
   to a common bits level; experts and dense may sit on different families at
   the same tier).
2. Fit `(a_inf, β)` from the available served acceptance points.
3. Measure `t`, `d0`, `c` (one serve at a reference rung + the footprint math).
4. **Discrete argmax** of `T(b)` over the menu — the menu is discrete, so the
   selector solves it directly. (The continuous optimum is Lambert-W:
   `2^{-b} β [ln2·(t+d0+c·b) + c] = c(1+a_inf)`; it lives here as theory and
   as a sanity cross-check, not as the implementation.)
5. **Memory gate:** discard rungs whose resident draft bytes break
   `weights + draft + profiling-peak + 3 GiB margin ≤ usable pool`. The gate
   is a hard clamp applied before the argmax.
6. **Degenerate-regime branch (first-class):** if `c·(b_max − b_min)` is under
   1% of the cycle — true whenever `d0` dominates, e.g. today's eager drafter
   (~50 ms host) or any draft whose bytes are small next to the shared lm_head
   — the argmax provably lands on the acceptance-max rung, so the selector
   picks **the highest-fidelity rung that passes the memory gate** and records
   `regime: degenerate`. This is the formula's own verdict, not a shortcut
   around it.
7. Emit the rung choice + a provenance JSON: constants, fit points, regime,
   menu, per-rung `T(b)` estimates. The continuous cross-check records WHICH
   solver answered in `continuous_bstar_lambertw_status`:
   `"scipy_lambertw"` when scipy evaluated `W₋₁` directly,
   `"log_space_continuation"` when the Newton continuation answered below
   float64 range, or one of the refusals `"scipy_absent"`,
   `"invalid_fit_constants"`, `"no_real_solution"`,
   `"continuation_did_not_converge"`; the separate `continuous_method` field
   is `"fixed_point"` when the iteration-only cross-check produced a value,
   else null. Since the 2026-08-21 re-underwrite the closed form is computed
   in log space (`log_M = ln((1+a_inf)/β) − g_over_c`) so `exp` can no longer
   overflow silently on large `(t+d0)/c` — the pre-fix code lost scipy for
   ratios ≳1022 (Hy3's recorded constants sit at ~1260) and masked it with
   the fixed-point answer; a Newton continuation of `W₋₁` on `s − ln s = L`
   covers sub-representable arguments exactly.

## 4. `k > 1` and the future regime

Acceptance enters as a product across positions, so sensitivity to draft
fidelity **compounds** with `k` — larger `k` pushes `b*` up. Today `k = 1` is
forced anyway (vLLM runs the drafter uncaptured; its ~50 ms/draft-token host
cost scales with `k`, measured k=1 13.1 / k=2 10.7 / k=3 8.8 tok/s on Hy3).
When upstream drafter capture lands, `d0` drops ~10×, the interior optimum
activates, and both `k` and `b*` should be re-solved with the same machinery.

## 5. Calibration procedure (per artifact)

- Encode the draft at two rungs spanning the menu (e.g. fp4 K18 and fp8 K44 —
  minutes each with the subset exporter).
- Serve each; read `vllm:spec_decode_num_{draft,accepted}_tokens_total` on a
  fixed natural-text generation set; record per-position acceptance.
- Fit, select, ship the winner; keep both measurements in the provenance.

Hy3 2026-07-20 data points: K18-mix draft acceptance 0.78–0.93 per-position on
natural text (~0.66 mixed incl. adversarial benches); `t ≈ 76 ms`,
`c ≈ 0.1 ms/bit`, eager `d0 ≈ 50 ms` → degenerate regime → highest rung
fitting memory.
