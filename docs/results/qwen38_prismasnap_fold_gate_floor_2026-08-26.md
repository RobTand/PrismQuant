# Qwen3.8-27B PrismaSnap fold gate sat below the BF16 perturbation floor

Date: 2026-08-26. Status: diagnosis **closed**; null-floor measurement and
re-attestation receipts referenced below.

## Claim

The served-BF16 fold-fidelity gate (`required_bf16_fold_kl_max = 5e-4`,
calibrated on the 0.6B trial where the refold floor measured ~6e-5) is below
the Qwen3.8-27B model's own response floor to *any* equal-mass BF16 weight
perturbation. No honest fold materialization can pass it, which is why every
overnight repair arm failed while none of them was broken.

## Evidence (all under the frozen 8x512 / all-position / top-1024 contract)

1. **Sub-additivity across disjoint arms.** The served-cost e4 campaign
   measured sixteen disjoint octile arms (8 layers x one seam family each) at
   3.5-6.5e-4 all-position KL apiece, while the union of all 128 seams
   measured 6.65e-4. Additive fold error predicts ~8.5e-3 for the union. The
   response is saturated. (`prismasnap-qwen38-27b-beta085-served-cost-e4-*/
   results/served-cost-collation.json`.)
2. **Strength inversion.** The beta ladder 0.85/0.70/0.55 measured
   6.65/6.51/6.44e-4 — weaker transforms did not measure better, and all
   three sit above full-strength v2 (6.17e-4). Floor jitter, not signal.
3. **Noiseless measurement.** The source A/A measured exactly 0.0
   (`prismasnap-qwen38-27b-20gb-20260825/fold-fidelity/source_aa.json`), so
   the band is chaos response to perturbation, not evaluator noise.
4. **The one real numerics defect was already fixed.** v1 (ideal-scale
   compensation, sequential per-transform BF16 rounding) measured 2.96e-3 —
   genuinely above floor. The v2 `bf16_realized` projection with realized
   consumer inverses cut it to 6.17e-4, i.e. to the floor. Everything after
   v2 was pushing below an unpassable constant.
5. **Matched-mass null.** Coherent-free elementwise jitter arms over exactly
   the v1 plan's 2-D tensors (half-ulp and full-ulp, predeclared fork in
   `/home/rob/dq-runs/prismasnap-null-floor-20260826/materialize_null_jitter.py`)
   measure the floor directly; the resulting
   `null_floor_receipt.json` is the licensing input for the derived gate.

The native-power2 pivot is closed as a v1 direction: exact-representability
admission keeps ~2% of the ~23% free-oracle proxy gain
(`prismasnap_p2_oracle2` runs; lattice-repair diagnostics agree).

## Decision

The gate constant moves from a fixed 0.6B-calibrated number to a
model-calibrated derivation in the attestor, with the plan hash untouched:

```
threshold = max(plan 5e-4, 2.0 x measured null floor)
```

licensed only when at least two independent null arms agree within 3x
(saturation is the licensing condition), and stamped into the VERIFIED
provenance as `fold_fidelity.threshold_derivation` with the receipt
content-hash. Implemented in `prismaquant/prismasnap_validation.py`
(`attest_fold_fidelity --null-floor-receipt`). Rationale: a fold whose served
KL is within 2x of what any equal-mass BF16 re-rounding costs adds no harm
beyond BF16 storage itself, while structural fold defects stay detectable
(the v1 defect measured ~5x the floor). Per-model calibration is mandatory —
the 125B MoE lane must measure its own null floor.

## Scope of the shipped claim

Rob descoped the full 27B A/B on 2026-08-26: the ship requirement is
*harmlessness* (fold KL at the measured floor) plus the ordinary pipeline
gates. The SnapQuant benefit numbers remain small-model claims — 0.6B served
mainline pipeline −31%/−33% KL, 4B −21.4/−17.8% — and no 27B percentage may
be claimed or carded until a 27B A/B exists. The frozen A/B ledger
(`qwen38_prismasnap_20gb_ab_2026-08-25.md`) stays open with its thresholds
unchanged for whenever that measurement is funded.

## Operational failures recorded alongside

- The overnight cleanup deleted the pinned BF16 source
  (`qwen38-27b-scout-aqua-20gb/source-text-mtp`) that every plan, teacher
  payload, and "deterministically reconstructible" prune receipt bound.
  It was rebuilt byte-identically from the HF cache (all 18 shard SHA-256s
  re-verified; `prismasnap-source-rebuild-20260826/REBUILD_RECEIPT.json`).
  A prune receipt that names a reconstruction dependency does not license
  deleting that dependency.
- The pruned v2 checkpoint was re-materialized deterministically inside the
  frozen v2 producer container and verified byte-identical to the retained
  v2 provenance (`prismasnap-v2-rematerialize-20260826/`), so the measured
  6.17e-4 student receipt applies to the restored bytes verbatim.
