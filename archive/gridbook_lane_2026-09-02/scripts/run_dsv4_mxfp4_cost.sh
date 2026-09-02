#!/usr/bin/env bash
# ============================================================================
# run_dsv4_mxfp4_cost.sh — extend the DSv4-Flash cost table to the full
#                          <= 4.25 bpw expert menu, resumably.
# ============================================================================
# The shipped v2 table measured three CB rungs (NVFP4_CB_K14/K15, FP8_CB_K36).
# The source-MXFP4 passthrough puts a ZERO-ERROR rung at 4.25 bpw on the expert
# menu, which dominates every costlier rung — so the useful menu is everything
# at or under 4.25 bpw, and the old 2.3 -> 4.5 bpw void (which forced the
# binary K15-vs-K36 jump at layers 39/40) gets filled with graded rungs the DP
# can use to express partial protection.
#
# MEASUREMENT ECONOMY. 19 rungs x 33,325 Linears is not affordable and is not
# necessary: PRISMAQUANT_CB_LADDER_INTERP=1 measures per-family ANCHORS plus a
# HOLDOUT and fits the rest from the shared RD law. For this menu the splitter
# picks
#     fp4:  anchors K12/K18/K24, holdout K19  -> predicts 9 rungs
#     fp8:  anchors K28/K31/K33, holdout K30  -> predicts 2 rungs
# i.e. 8 measured columns instead of 19. Both families' MENU ENDPOINTS are
# anchors, so the law only ever interpolates, never extrapolates. The fit is
# gated PER TENSOR: a tensor whose law misses its holdout by more than
# PRISMAQUANT_CB_LADDER_TOL has its predicted rungs MEASURED instead, so no
# tensor ever receives an unvalidated price. Fitted rows are stamped
# cost_source="band_interpolated" and stay distinguishable in the artifact.
#
# The run also re-prices K14/K15, which the v2 table already MEASURED. That
# overlap is deliberate: it is the only out-of-sample check on the
# interpolator that uses this model's real production rows, and
# scripts/merge_dsv4_mxfp4_cost.py reports it before merging (merge always
# prefers the measured value).
#
# RESUMABILITY. cost_shard_is_reusable fails closed on CB shards, so a single
# re-invocation would re-measure every completed layer. This driver therefore
# runs ONE LAYER PER CONTAINER into its own output pickle and skips layers
# whose pickle already exists — an interrupt costs at most the layer in flight.
# The per-layer skeleton reload is ~3 s against ~20 min of encode.
#
# Cost measurement is correctness-only (no timing claims), so it needs no
# exclusive-GPU flock.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/source}"
WORK_DIR="${WORK_DIR:-${RUN_ROOT}/prod-cal-0p6-v2}"
ART="${ART:-${WORK_DIR}/artifacts-mxfp4}"
SHARDS="${SHARDS:-${ART}/shards}"
LOGS="${LOGS:-${WORK_DIR}/logs-mxfp4}"
IMAGE="${IMAGE:-gridbook:test}"
CALIB_DIR="${CALIB_DIR:-/home/rob/dq-runs/calibration}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-43}"

# Every nvfp4_cb-carriable rung at or under 4.25 bpw. MXFP4_SOURCE itself is
# NOT here: it is a byte-copy contract the allocator synthesizes at zero cost,
# so measuring it would burn GPU hours reproducing a zero.
MENU="${MENU:-NVFP4_CB_K12,NVFP4_CB_K13,NVFP4_CB_K14,NVFP4_CB_K15,NVFP4_CB_K16,NVFP4_CB_K17,NVFP4_CB_K18,NVFP4_CB_K19,NVFP4_CB_K20,NVFP4_CB_K21,NVFP4_CB_K22,NVFP4_CB_K23,NVFP4_CB_K24,FP8_CB_K28,FP8_CB_K29,FP8_CB_K30,FP8_CB_K31,FP8_CB_K32,FP8_CB_K33}"

mkdir -p "$SHARDS" "$LOGS"

for L in $(seq "$START_LAYER" $(( END_LAYER - 1 ))); do
  OUT="${SHARDS}/cost_L$(printf '%03d' "$L").pkl"
  if [ -s "$OUT" ]; then
    echo "[mxfp4-cost] layer ${L}: already done ($(stat -c%s "$OUT") B) — skipping"
    continue
  fi
  echo "[mxfp4-cost] layer ${L}: measuring -> ${OUT}"
  docker run --rm --name "pq-mxfp4-cost-L${L}" --gpus all --ipc=host \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${CALIB_DIR}:${CALIB_DIR}:ro" \
    -v "${REPO}:/pq" -w /pq \
    -e PYTHONPATH=/pq \
    -e PRISMAQUANT_CB_EXT_DIR="${RUN_ROOT}/ext" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    -e PRISMAQUANT_CB_LADDER_INTERP=1 \
    -e CB_CODEBOOK_SOURCE=lattice \
    -e CB_SCALE_CODING=two_tier \
    -e CB_SCALE_SWEEP=1 \
    -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
    -e PRISMAQUANT_CB_COL_WEIGHTS="${ART}/cb_col_weights.pkl" \
    -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE="${ART}/cb_col_weights.pkl.provenance.json" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model ${MODEL_PATH} --cost-mode local \
  --probe ${ART}/probe.pkl \
  --activation-cache-dir ${WORK_DIR}/act \
  --formats '${MENU}' \
  --output ${OUT} \
  --work-dir ${WORK_DIR}/work-mxfp4-L${L} \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer ${L} --end-layer $(( L + 1 )) \
  --skip-missing-activations --no-include-lm-head" \
    >> "${LOGS}/cost-L$(printf '%03d' "$L").log" 2>&1
  echo "[mxfp4-cost] layer ${L}: done at $(date -Is)"
done

echo "[mxfp4-cost] all layers ${START_LAYER}..$(( END_LAYER - 1 )) present in ${SHARDS}"
