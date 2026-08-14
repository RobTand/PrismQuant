# MTP draft-rung selection — canon

**Status: normative (Robert, 2026-07-20; block-parallel extension 2026-08-13).**
Every MTP/spec-decode draft PrismaQuant quantizes is selected by an explicit
throughput-and-memory objective — not by hand and not by copying body rungs.
Implementation: `prismaquant/mtp_rung_selection.py`.

There are two execution models, and confusing them changes the answer:

- `select_rung` is the original sequential-drafter approximation. The Hy3
  driver (`scripts/build_hy3_mtp_cb_inputs.py --rung-select auto`) is its
  reference integration.
- `select_measured_configuration` is the ship authority for block-parallel
  drafters such as DeepSeek DSpark. It selects a joint `(quantization
  assignment, k)` candidate from served cycle time and exact per-position
  acceptance counters.

Under exact rejection sampling a draft does not change the target
distribution, so draft fidelity affects acceptance and therefore speed rather
than target quality. Coherence and exact-output checks remain mandatory runtime
gates; this document does not use that property to excuse a broken draft.

## 1. Objective

Per spec-decode engine cycle with `k` speculative tokens:

```
T(b) = E[tokens/cycle] / cycle = (1 + Σ_{i=1..k} Π_{j<=i} a_j(b)) / (t + k·d(b))
```

`b` = draft bits/weight, `t` = target verify-step time, `d(b)` = one drafter
forward. For `k = 1`: `T(b) = (1 + a(b)) / (t + d(b))`.

### 1.1 Block-parallel objective

DSpark produces all `k` positions in one parallel draft-backbone call. Its
candidate `c = (assignment, k)` is therefore scored directly:

```
p[c,i]   = served cumulative survival at position i
u[c]     = 1 + Σ_i p[c,i]                 expected output tokens/cycle
τ[c]     = measured complete decode-cycle wall time
T[c]     = 1000 · u[c] / τ[c]             steady-state tokens/second
```

`p[c,i]` is `accepted_at_position_i / draft_cycles`, not conditional
acceptance and not the aggregate `accepted / proposed` ratio. The sequence must
be non-increasing. The denominator is measured once for the whole block; using
`t + k·d` here would count DSpark's parallel backbone `k` times.

Startup time can be priced without mixing units. For an expected service
horizon of `H` decode cycles and measured draft load time `L[c]`:

```
T_H[c] = 1000 · H · u[c] / (L[c] + H · τ[c])
```

Absent a declared `H`, steady-state `T[c]` is primary and load time is a
deterministic tie-break. The selector also reports the Pareto frontier over
steady throughput, resident bytes, and load time.

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
   menu, per-rung `T(b)` estimates.

### 3.1 Direct measured selector for block-parallel drafts

For every candidate, run the same concurrency-1 workload and record:

- complete cycle wall time `τ[c]`;
- cumulative acceptance `p[c,0..k-1]` from counter deltas;
- steady resident draft bytes and candidate-specific peak scratch bytes;
- load time when startup amortization matters;
- the same workload identity for every row.

Then solve the finite problem exactly:

```
maximize_c  T[c]                     (or T_H[c] when H is declared)

subject to
  fixed_runtime
  + target_KV
  + draft_KV
  + profiling_peak
  + safety_margin
  + resident[c]
  + peak_scratch[c]  <= usable_pool

  k[c] >= artifact_k_floor
```

No monotone-in-bits assumption is needed. Kernel route changes, expert-union
effects, and block-parallel execution are already present in the measurement.
Candidates from different workload identities are refused. Ties prefer higher
steady throughput, then smaller residency, faster load, lower measured error,
then lexical name.

The complete memory ledger is part of the selection record. A deliberately
relaxed experiment may set `admission_mode="test-only-relaxed"` and reduce the
safety term, but that result is not publishable evidence for a production
memory gate. Shipping requires re-selection under the production ledger.

## 4. `k > 1` and the future regime

For a sequential drafter, acceptance enters as a product across positions, so
sensitivity to fidelity compounds with `k`; larger `k` can push `b*` up. The
historical Hy3 result was k=1 because its uncaptured per-token host cost scaled
with `k` (measured k=1 13.1 / k=2 10.7 / k=3 8.8 tok/s).

For DSpark, `k` belongs directly in the discrete candidate. The released
DeepSeek-V4-Flash draft has `dspark_block_size=5`; vLLM refuses `k < 5` because
that layout produces invalid output. Start with `k=5`. Treat `k>5` as a
separate measured candidate, never as a free extrapolation from the sequential
formula.

## 5. Calibration procedure (per artifact)

- Encode the draft at two rungs spanning the menu (e.g. fp4 K18 and fp8 K44 —
  minutes each with the subset exporter).
- Serve each; read `vllm:spec_decode_num_{draft,accepted}_tokens_total` on a
  fixed natural-text generation set; record per-position acceptance.
- Fit, select, ship the winner; keep both measurements in the provenance.

For DSpark/block-parallel candidates, skip the `d0+c·b` fit and measure each
feasible candidate end to end. At concurrency 1, snapshot metrics around the
same fixed prompt block. `spec_decode_num_drafts_total` supplies the cycle
denominator; accepted-per-position counters supply `p[c,i]`; decode-time delta
divided by cycles supplies `τ[c]`. Use `(drafts + accepted) / decode_seconds`
as an independent aggregate cross-check. Prometheus does not isolate draft-only
GPU time; CUDA-event instrumentation may diagnose the split, but the selection
objective remains the served complete cycle.

Hy3 2026-07-20 data points: K18-mix draft acceptance 0.78–0.93 per-position on
natural text (~0.66 mixed incl. adversarial benches); `t ≈ 76 ms`,
`c ≈ 0.1 ms/bit`, eager `d0 ≈ 50 ms` → degenerate regime → highest rung
fitting memory.
