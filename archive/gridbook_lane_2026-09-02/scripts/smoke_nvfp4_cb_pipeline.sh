#!/usr/bin/env bash
# ============================================================================
# smoke_nvfp4_cb_pipeline.sh — 0.6B end-to-end smoke for the NVFP4-CB lane
# ============================================================================
#
#   *** NOT AUTO-RUN. REQUIRES AN IDLE GPU. ***
#
# This drives the FULL EXPORT_CONTAINER=nvfp4_cb lane end-to-end on Qwen3-0.6B:
#
#     probe  ->  LOCAL cost  ->  allocate @ ~3.0 bpp (mixed CB menu)  ->
#     harvest imatrix col-weights from the act cache  ->  export_nvfp4_cb
#
# It is a CORRECTNESS smoke (does the lane run, allocate a mixed CB assignment,
# and produce a loadable custom-quant-config checkpoint), NOT a quality gate.
# The served gold-metric KL/PPL gate runs separately, in the out-of-tree
# gridbook_plugin serving env (docs/lanes/nvfp4-cb/serve_prototype_0p6b.md).
#
# GPU-or-bust: run-pipeline.sh + gpu_guard refuse to run on CPU. The
# orchestrator schedules this when the GPU frees. Do NOT launch it while
# another agent owns the GPU.
#
# Usage (when the GPU is free):
#     bash scripts/smoke_nvfp4_cb_pipeline.sh
# Override any variable inline, e.g.:
#     TARGET_BITS=3.0 CB_CODEBOOK_SOURCE=learned bash scripts/smoke_nvfp4_cb_pipeline.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- inputs ----------------------------------------------------------------
# NEVER write artifacts to /tmp (cleared on OOM 2026-04-23, wiped MiniMax
# artifacts). Keep the work dir under /home/rob.
: "${MODEL_PATH:=/home/rob/models/Qwen3-0.6B}"
: "${WORK_DIR:=/home/rob/dq-runs/smoke-nvfp4-cb-0p6b}"

# --- the CB lane contract (all gated by run-pipeline.sh) -------------------
export EXPORT_CONTAINER=nvfp4_cb   # select the codebook container lane
export TARGET_PROFILE=nvfp4_cb     # allocator gates every candidate via nvfp4_cb.json
export COST_MODE=local             # weighted skeleton-requantize render == shipped bytes
export PRODUCTION_CACHE=0          # exporter requantizes the bf16 skeleton; no cache read
export PRODUCTION_RECACHE=0

# --- the mixed CB menu @ ~3.0 bpp (the "FP8 in every recipe" thesis) -------
# fp4-CB rungs bracket the 3.0 target (K16=2.5, K20=3.0, K24=3.5), plus the
# mid-range FP8-CB rung (K44=5.5), plus the native carriers NVFP4 (4.5),
# FP8_DYNAMIC (8.0) and BF16. Well-spaced (>=0.5 bpw apart) so the
# family-coherence gate does not even warn.
: "${FORMATS:=NVFP4_CB_K16,NVFP4_CB_K20,NVFP4_CB_K24,FP8_CB_K44,NVFP4,FP8_DYNAMIC,BF16}"
: "${TARGET_BITS:=3.0}"
: "${PARETO_TARGETS:=2.75,3.0,3.25}"

# --- calibration (small = fast smoke; ACTIVATION_ROWS_LIMIT auto-defaults to
# 1024 for this lane, the higher-rank imatrix the weighted VQ search wants) --
: "${NSAMPLES:=16}"
: "${SEQLEN:=512}"

# --- CB exporter knobs (see the lane in run-pipeline.sh) -------------------
# lattice = deterministic fixed FP4/FP8 lattice (no sidecar, fastest smoke).
# Switch to `learned` for the shared-per-role byte-competitive champion.
: "${CB_CODEBOOK_SOURCE:=lattice}"
: "${CB_SCALE_SWEEP:=1}"            # default-on joint E4M3-legal scale sweep
: "${CB_SCALE_CODING:=v1}"         # v1 e4m3 plane (served); two_tier is serve-gated
export FORMATS TARGET_BITS PARETO_TARGETS NSAMPLES SEQLEN
export CB_CODEBOOK_SOURCE CB_SCALE_SWEEP CB_SCALE_CODING MODEL_PATH WORK_DIR

echo "============================================================================"
echo "NVFP4-CB 0.6B SMOKE — configuration (echo for the record)"
echo "----------------------------------------------------------------------------"
echo "  MODEL_PATH        = $MODEL_PATH"
echo "  WORK_DIR          = $WORK_DIR"
echo "  EXPORT_CONTAINER  = $EXPORT_CONTAINER"
echo "  TARGET_PROFILE    = $TARGET_PROFILE"
echo "  COST_MODE         = $COST_MODE"
echo "  PRODUCTION_CACHE  = $PRODUCTION_CACHE / RECACHE=$PRODUCTION_RECACHE"
echo "  FORMATS           = $FORMATS"
echo "  TARGET_BITS       = $TARGET_BITS   (PARETO_TARGETS=$PARETO_TARGETS)"
echo "  NSAMPLES x SEQLEN = ${NSAMPLES} x ${SEQLEN}"
echo "  CB_CODEBOOK_SOURCE= $CB_CODEBOOK_SOURCE"
echo "  CB_SCALE_SWEEP    = $CB_SCALE_SWEEP   CB_SCALE_CODING=$CB_SCALE_CODING"
echo "============================================================================"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH=$MODEL_PATH not found. Set MODEL_PATH to a bf16 Qwen3-0.6B dir." >&2
  exit 1
fi
mkdir -p "$WORK_DIR"

# Run the real orchestrator. The CB lane fail-fast gates (COST_MODE/
# TARGET_PROFILE/PRODUCTION_CACHE) validate the config before any GPU work.
bash "${REPO_ROOT}/prismaquant/run-pipeline.sh"

echo
echo "############################################################################"
echo "##  REVIEW OUTPUTS  ########################################################"
echo "############################################################################"
echo "##  Checkpoint : ${WORK_DIR}/exported_nvfp4_cb"
echo "##  Assignment : ${WORK_DIR}/artifacts/layer_config.json  (confirm a MIXED"
echo "##               CB allocation: NVFP4_CB_* / FP8_CB_* + NVFP4/FP8/BF16)"
echo "##  Col-weights: ${WORK_DIR}/artifacts/cb_col_weights.pkl  (imatrix lockstep)"
echo "##  Config     : ${WORK_DIR}/exported_nvfp4_cb/quant_config.json"
echo "##               (quant_method=gridbook; LAYOUT.md byte contract)"
echo "##"
echo "##  NEXT (separate, plugin serving env — NOT this script):"
echo "##   1. Load + generate in vLLM with the pinned gridbook package installed."
echo "##   2. Served KL-vs-BF16 + WikiText PPL vs the emulation-gate predictions"
echo "##      (docs/lanes/nvfp4-cb/serve_prototype_0p6b.md). No rung is production-"
echo "##      eligible until it clears that gold-metric gate AND the prefill perf"
echo "##      gate (INV-2, no Triton masquerade)."
echo "############################################################################"
