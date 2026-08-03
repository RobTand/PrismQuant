#!/usr/bin/env bash
# ============================================================================
# run_dsv4_mxfp4_dual_alloc.sh — the two DP variants the mid-rung question needs
# ============================================================================
# Both run against ONE merged cost table: the K12-K18 probe columns (experts,
# holdout-validated interpolation) merged over the v2 measured table (which
# supplies FP8_CB_K36 for the body). The merge always PREFERS a measured value
# over a fitted one, and reports the overlap validation before merging.
#
#   (a) shippable-today   menu WITHOUT FP8_BLOCK_UE8M0_SOURCE.
#                         The body passthrough route is measured-blocked on
#                         sm121, so this is the honest answer against the
#                         5812.11 baseline.
#
#   (b) LIKELY SHIP PICK  menu WITH it. Gridbook 0.8.0 is released with a
#                         dense MXFP8 lane serving these bytes, so the route
#                         is real and route_status is `backed`. It remains
#                         OPT-IN pending its native-parity timing bench, which
#                         makes GRIDBOOK_MXFP8_DENSE=1 a load-bearing serve
#                         flag alongside marlin -- so (a) stays worth
#                         computing as the conservative fallback if that flag
#                         cannot be set at deploy.
#
# The single --formats list applies to experts AND body, so the body units in
# both variants already carry the probe's K12-K18 columns. That matters for
# (b): with passthrough legal the DP can MIX -- passthrough for sensitive body
# units, cheap CB for flat ones -- and the body split is the reportable.
#
# Neither variant writes to artifacts/, artifacts-mxfp4-sm121/, or
# cost_full.pkl. Both are CPU-only -- they must not be run while a GPU tenant
# needs the box, but they do not take the flock.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
WORK="${WORK:-${RUN_ROOT}/prod-cal-0p6-v2}"
ART="${ART:-${WORK}/artifacts-mxfp4}"
PROBE="${PROBE:-${ART}/probe-k12k18}"
IMAGE="${IMAGE:-gridbook:test}"

EXPERT_MENU="NVFP4_CB_K12,NVFP4_CB_K13,NVFP4_CB_K14,NVFP4_CB_K15,NVFP4_CB_K16,NVFP4_CB_K17,NVFP4_CB_K18"
MENU_A="${EXPERT_MENU},FP8_CB_K36,MXFP4_SOURCE"
MENU_B="${MENU_A},FP8_BLOCK_UE8M0_SOURCE"
PARETO="${PARETO:-1.80,1.90,2.00,2.04,2.08,2.12,2.15,2.17,2.20,2.25,2.30,2.40,2.50,2.75,3.00,3.50,4.00,4.25,4.50}"

# --- 1. merge + overlap validation ----------------------------------------
# K14/K17/K13 are FITTED in the probe and K14 is MEASURED in v2, so the merge
# report carries a real out-of-sample error band for the interpolator.
echo "[dual] merging probe over v2 (measured always wins)"
python3 "${REPO}/scripts/merge_dsv4_mxfp4_cost.py" \
  --base "${WORK}/artifacts/cost_full.pkl" \
  --new  "${PROBE}/cost_k12k18.pkl" \
  --out  "${PROBE}/cost_merged.pkl" \
  --report "${PROBE}/merge_report.json"

run_variant() {
  local tag="$1" menu="$2" budget_b="${3:-92000000000}" extra_b="${4:-0}"
  local eff_b=$(( budget_b + extra_b ))
  local disk_gb; disk_gb=$(python3 -c "print(${eff_b}/1e9)")
  local out="${PROBE}/alloc-${tag}-$(( budget_b / 1000000000 ))"
  mkdir -p "$out"
  ln -sfn "${ART}/probe.pkl" "$out/probe.pkl"
  ln -sfn "${ART}/cb_col_weights.pkl" "$out/cb_col_weights.pkl"
  echo "[dual] variant ${tag}: ${menu}"
  docker run --rm --name "pq-mxfp4-alloc-${tag}" --gpus all --ipc=host \
    -v "${RUN_ROOT}:${RUN_ROOT}" -v "${REPO}:/pq" -w /pq \
    -e PYTHONPATH=/pq \
    -e PRISMAQUANT_CB_EXT_DIR="${RUN_ROOT}/ext" \
    -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier \
    -e CB_SCALE_SWEEP=1 -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.allocator \
  --probe $out/probe.pkl \
  --costs ${PROBE}/cost_merged.pkl \
  --model ${RUN_ROOT}/source \
  --formats '${menu}' \
  --target-bits 2.17 --pareto-targets '${PARETO}' \
  --target-disk-gb ${disk_gb} --artifact-overhead-reserve-bytes 268435456 \
  --target-profile nvfp4_cb \
  --cb-scale-coding two_tier --cb-codebook-source lattice \
  --cb-scale-sweep 1 --cb-encode-tier balanced \
  --cb-col-weights $out/cb_col_weights.pkl \
  --layer-config $out/layer_config.json \
  --bit-attribution-json $out/bit_attribution.json \
  --pareto-csv $out/pareto.csv" > "${WORK}/logs-mxfp4/alloc-${tag}.log" 2>&1
  echo "[dual] variant ${tag} done -> $out/selection.json"
}

# --- (c) the no-MTP counterfactual --------------------------------------
# Dropping the MTP block from the artifact is ARITHMETICALLY IDENTICAL to
# raising the budget by its size, because MTP is never re-encoded and so sits
# wholly in the immutable floor:
#
#     floor_without_mtp + body <= B    <=>    floor_with_mtp + body <= B + MTP
#
# So (c) runs the SAME table and menu as (b) at budget 92e9 + 10,862,838,300,
# and its selection is exactly the selection a genuinely MTP-less artifact
# would get. Only the REPORTED total needs correcting back down by the MTP
# bytes, which the summariser does. Measured independently against the
# checkpoint headers: mtp.* = 10,862,838,300 B, of which 10,267,656,192 B is
# routed experts.
#
# Purpose is to price the QUALITY side of a speed-vs-quality exchange:
# (c) vs (b) Delta-loss IS the quality cost of keeping the DSpark draft heads.
# The speed side is measured post-export on the serve arm.
MTP_BYTES="${MTP_BYTES:-10862838300}"
# --- BUDGET AXIS ----------------------------------------------------------
# The serving target moved to ~131k context, so weight bytes now trade against
# KV-pool headroom: ~29.7 KB/token => ~3.97 GB per full 131k stream. 92 GB of
# weights supports ~4 concurrent streams, 88 GB supports ~5.
#
# THE STREAM COUNT IS INTEGRAL, and that dominates the choice: every budget
# between the 5-stream threshold and 92 GB buys exactly ZERO extra streams
# while still costing quality. So the interesting point is not "as low as
# possible" but "the LARGEST budget that still clears 5 streams" -- anything
# below that is quality paid for nothing. 90e9 is added for (b) to test
# whether the 5th stream is already reachable at roughly half the shave, which
# is the one extra budget point the latitude allows. If 90 clears 5 streams it
# strictly dominates 88; if it does not, 88 is vindicated and the extra run
# cost ~4 CPU-minutes to prove it.
run_variant a "$MENU_A" 92000000000
run_variant b "$MENU_B" 92000000000
run_variant b "$MENU_B" 90000000000
run_variant b "$MENU_B" 88000000000
run_variant c "$MENU_B" 92000000000 "$MTP_BYTES"
run_variant c "$MENU_B" 88000000000 "$MTP_BYTES"
echo "[dual] budget grid complete; summarise with scripts/summarise_dual_alloc.py"
