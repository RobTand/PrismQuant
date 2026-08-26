# Qwen3.8-27B PrismaSnap 20 GB A/B — 2026-08-25

Status: **PAUSED — descoped 2026-08-26** (Rob): the full 27B A/B is not funded
now; the ship requirement is fold harmlessness plus the ordinary pipeline
gates, and the fixed fold gate itself was found to sit below the 27B BF16
perturbation floor — see
`qwen38_prismasnap_fold_gate_floor_2026-08-26.md` for the diagnosis, the
derived-threshold decision, and the null-floor receipts.  The thresholds
below stay frozen unchanged for whenever this measurement is funded.  Do not
cite the small-model SnapQuant percentages as this run's result.

## Fixed control

- Control artifact: `rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB`.
- BF16 text+MTP/no-vision source revision:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Source portable content identity:
  `6a7d9f66d85062bdb9990b556950a197044776142ccabb43e30b0f7756d846cb`;
  source identity file SHA-256
  `5f85249f4d2855d1948674594cfee927166937a3709bf165422aa878a702a7dc`.
- Probe: diverse-v1, 16×512, calibration hash
  `cfb1fc0073d2dd250a7351608dd85aa4`, file SHA-256
  `dbe7ee001e857a284c66f13831aae5809bbe5551f147cfbf91f6391b38097169`.
- Frozen assignment SHA-256:
  `656af501472c2be7c06ee5903c62ab3019eac4e11109b7f6382ec524ce9c4a1c`,
  505 Linears, body bpp 5.159680.
- Gold teacher: 8×512, all positions, top-1024 plus tail; payload SHA-256
  `3edc8c77960af0459998778fa15c0426fd19539e0e780ff03caaa7797e8fab01`.
- Control exported tensor payload: 19,950,598,312 bytes; complete Hub tree:
  19,974,334,328 bytes.
- Control all-position KL 0.04022016687631274; confident-position KL
  0.023954083969016368; WikiText PPL 9.7312 (BF16 9.3662).

The requested 20% thresholds are therefore all-position KL ≤0.03217613350105
and confident-position KL ≤0.01916326717521.  The final artifact must remain
below the strict decimal 20,000,000,000-byte card using the same accounting
semantics; the control has only 25,665,672 bytes of complete-tree headroom.

## Candidate protocol

1. Run the exact fast `stage,polish` PrismaSnap v1 plan across Sparky and
   Sparklina using a sealed `cluster_campaign` manifest.
2. Exact-union layer plans, materialize disjoint original shard sets, transfer
   the remote part by content hash, and exact-union the checkpoint parts.
3. Serve the ordinary snapped BF16 checkpoint against the fixed unsnapped gold
   teacher.  Require fold KL ≤5e-4, then attest the result.
4. Recapture activation rows from the snapped source on the identical
   calibration IDs.  Keep assignment semantics frozen for the primary causal
   A/B; a fresh allocator arm may be reported separately.
5. Rebuild the ordinary production cache/recache/export at the strict decimal
   20 GB target and run eager+graph load, gold 8×512 KL, WikiText PPL, artifact
   census, and served runtime comparison.

## Evidence paths

Run root: `/home/rob/dq-runs/prismasnap-qwen38-27b-20gb-20260825/`.
Commands, immutable manifest, state, logs, per-stage receipts, exact digests,
GPU telemetry, and final metrics are appended here only after they exist.

## Result

Pending.  No promotion claim is made by this document yet.
