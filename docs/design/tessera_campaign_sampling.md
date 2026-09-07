# Tessera campaign sampling design (2026-09-06)

**Status:** design, grounded in measurements on `campaign-01`
(`/mnt/shared/tessera-measurements/pq275-2026-09-06/campaign-01`, LFM2.5-8B-A1B,
layers 0 and 18, Tessera `a9eb572`, contract v22, 501 anchor rows at the time of
writing). Nothing here is a served result; the validation section says what
must hold before a GLM-5.3 Flash row is encoded under this design.

## 1. The problem

`prismaquant/tessera_campaign.py` prices every `(unit, family)` surface with
three to twelve anchors under `--menu-mode research`, for every routed expert,
one unit per encode. At LFM scale that is a 3 h round on one box. Extrapolated
to GLM-5.3 Flash (45 layers, 42 MoE layers × 288 routed experts × 3 Linears of
4096×2048, 305 B routed parameters) it is about 250 box-days. Three measured
causes, in order of size:

1. **Two thirds of the rows cannot be served.** The pinned contract publishes
   three families and their decoder ranges (`formats[].reader_rate_range_q256`):
   `TESSERA_E2M1_K2` only at q256 896, `TESSERA_E4M3_K1` on [256, 2048],
   `TESSERA_BF16_K1` on [256, 4096]. There is no `TESSERA_E2M1_K1` reader.
   campaign-01 spent 168 rows × 27 s on E2M1_K1 and two of its three E2M1_K2
   anchors sit at unreadable rates. Fix: #284 (`readable` menu mode).
2. **Every expert is priced although the stack is the decision unit.** vLLM
   packs a layer's routed experts into one fused module; the serving invariant
   (experts uniform per layer) makes the packed stack the allocation unit, so
   the campaign needs the stack's price, not 288 prices.
3. **The encoder runs at 16 W of a 140 W envelope.** `perf` on the campaign
   process shows the CPU parked in `cuStreamSynchronize` behind Python
   `float()` calls; under LDLQ each trellis launch is `rows × 32` columns wide
   (`DEFAULT_LDLQ_BLOCK=32`), 64–128 launches per pass, four passes per rung.
   Fix: Tessera batched multi-unit LDLQ encode (Tessera issue filed
   2026-09-06, "Encoder: batch many same-shape units").

## 2. Measured inputs (campaign-01)

Encode cost per row, 3.7 M-parameter expert Linears, LDLQ on:

| family | rate (bits) | wire | median s |
|---|---|---|---|
| BF16_K1 (window) | 1.0 / 4.46 / 7.92 | 0.5 / 2.1 / 3.7 MB | 2.5 / 3.1 / 7.2 |
| E4M3_K1 (window) | 1.0 / 4.43 / 7.87 (dense 4.8 M) | 0.5 / 2.3 / 4.1 MB | 2.8 / 3.6 / 8.3 |
| E2M1_K2 (TCQ + LUT) | 3.5 (dense 4.8 M) | 2.1 MB | 15.3 |
| E2M1_K1 (forest) | 1.0 | 0.8 MB | 26.0 |

Encode time scales with wire bytes: ~2 s/MB for the window body, ~7 s/MB for
the TCQ/LUT body, ~30 s/MB for the forest.

Rate law, per expert unit (log₂ Δloss vs body bits):

- Slopes are tight within a role: BF16_K1 −1.85 (per-unit sd 0.10, n = 96);
  E2M1_K1 −1.57 (sd 0.17, n = 77).
- The surface is **not** one line over 1–8 bits. One anchor at q256 1142 plus
  the pooled slope misses the 256 and 2027 anchors by median 0.29 / p90 0.74
  log₂ (gate 0.25) with the same sign at both ends: curvature, not noise.
  Inside a one-bit bracket (E2M1_K1, 256 → 512) the same construction lands
  at median 0.08 / p90 0.29.
- Cross-family: E4M3_K1 and BF16_K1 coincide at 1 bit (within 0.02 log₂) and
  diverge toward the E4M3 grid floor (−15.8 … −17.2 log₂ at 8 bits); an
  additive-floor model of E4M3 from the BF16 law is 0.26 log₂ off at 4.4
  bits, so E4M3 keeps its own anchors. E2M1_K2@896 sits 0.45–1.12 log₂ above
  the window law at 3.5 bits (sd 0.25 across roles): it is measured, never
  derived.

Cross-expert spread inside one stack at a fixed rung: CV 0.33–0.55 by role;
one rarely routed expert (`experts.6`) prices 10× below the median at every
rung. Uniform sampling of the 32-expert stack sum: n = 8 gives a median
relative error of 8–11 % (p90 17–28 %); n = 16 gives 4–7 % (p90 10–16 %).

## 3. Design

- **D1 Menu.** `--menu-mode readable` (#284): the campaign never encodes a
  rung the pinned decoder cannot read. Rows already encoded outside the
  readable set stay in the checkpoint as evidence and are never priced.
- **D2 Decision unit and sampling.** For routed experts the stack
  (layer × role) is the unit. Each stack is priced from an importance sample
  of `n_s` experts drawn with probability proportional to the probe's
  `h_trace` for that expert, stratified so the top decile is always
  represented; the **same subset at every rung** of every family, so the
  between-rung marginal carries only within-expert slope noise (sd 0.10 log₂)
  while the level carries the sampling error. Estimator: Horvitz–Thompson on
  `h_e · mse_e`; its standard error is stamped as `dloss_stderr`, which the
  allocator already consumes. Default `n_s = 16`; a `--stack-se` target picks
  `n_s` per stack from the probe's `h_trace` dispersion.
- **D3 Anchors per sampled Linear.** `E2M1_K2@896` once, directly (its wire is
  the shipping bytes when that rung is chosen). Each window family gets two
  anchors at the ends of the artifact band, with linear log₂ interpolation
  inside the bracket and no extrapolation (the existing surface rule). The
  band is derived from the byte budget of the artifacts the run serves: a
  one-Spark GLM artifact sits near 2–4 body bits, a two-Spark one near 4–7. A
  third anchor is placed only on an audit subsample (one expert in ten) where
  its three-anchor LOO error exceeds the 0.25 gate, using the existing
  adaptive loop; a stack whose audit fails is declared non-interpolable for
  that family and priced at its anchors only.
- **D4 Dense units.** Attention, dense MLP and shared-expert Linears are priced
  in full with the same anchor rule; they are ~7 B of GLM's 312 B parameters.
- **D5 Fan-out.** Every `(stack, family, rung)` is one PrismaBuild quantum
  (#282); the checkpoint merge is keyed by `(unit, family, rung)` so a quantum
  that dies is re-run, never re-designed.
- **D6 Encoder.** The sampled experts of a stack share shape, family and rung:
  one batched LDLQ call per `(stack, family, rung)` is the "same grammar"
  lever, and it serves the export as well as the campaign.

## 4. Projection for GLM-5.3 Flash

Rows per sampled expert Linear at GLM size (8.4 M parameters, 2.3× the LFM
units, current encoder speed): E2M1_K2@896 ~30 s, E4M3_K1 at two anchors
~20 s, BF16_K1 at two anchors ~25 s: about 75 s.

| design | routed experts | dense/attention | total, two boxes |
|---|---|---|---|
| as-is (research menu, all experts, 3+ anchors) | ~250 box-days | — | months |
| readable menu, all experts, bracketed anchors | 42 × 288 × 3 × 75 s = 31 box-days | 0.6 box-days | 16 days |
| **this design, n_s = 16** | 42 × 3 × 16 × 75 s = 42 h | ~14 h | **~28 h** |
| this design, n_s = 8 | 21 h | ~14 h | ~18 h |
| this design + batched encoder at 5× | 8 h | 3 h | ~6 h |

The export of the chosen allocation (312 B parameters at one rung each) is a
separate cost: 3–12 box-days at today's LDLQ row rate (0.3–1.2 M params/s), or
6.4 h at the LDLQ-free rate the 2026-09-01 GLM export achieved (13.5 M
params/s). The batched encoder or the LDLQ-free path is required for the
export; which one is a quality question answered by the #283 factor screen
(hessian require/off × ldl_block × scale_refit, with `dloss` per cell).

CPU (dl380g10, 2× Xeon Gold 6230, AVX-512) does not encode: Tessera PR #383's
reference Viterbi took 981 s for a 128×512 fixture over seven rungs, three
orders of magnitude off the trellis. It takes the LOO fits, the checkpoint
merge, the allocations and the regret analysis below.

## 5. Validation before any GLM row

- **V1 Regret on the LFM table.** campaign-01's full table (3+ anchors, all 32
  experts, layer 18) is the ground truth. Simulate D2 + D3 offline (n_s ∈
  {4, 8, 16}, two bands) and allocate under the group knapsack at three byte
  budgets; report the Δloss regret of each arm's allocation evaluated with
  the full table's prices, and the byte-equivalent of that regret. The screen
  is informative, not decisive.
- **V2 Served KL.** One allocation per budget from the sampled design and one
  from the full table, exported and served; the #275 real-KL gate decides.
  A sampled-design artifact that is within the between-session KL noise of
  the full-table artifact at matched bytes passes.
- **V3 Encoder identity.** The batched encoder is bit-identical per unit to
  the unbatched one on a two-unit fixture at one rung per family, LDLQ on,
  before the campaign uses it; the checkpoint stays identity-bound.

## 6. What this does not do

It does not change the currency (`output_mse_under_route_activation_contract`),
the allocator, the export gate or the attested set. Rungs admitted under
`readable` are still `unattested` on every stamp until a served-KL receipt
moves them into the contract's `attested_rungs_q256`.
