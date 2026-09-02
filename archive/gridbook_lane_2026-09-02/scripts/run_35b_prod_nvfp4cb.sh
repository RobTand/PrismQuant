#!/usr/bin/env bash
# ============================================================================
# run_35b_prod_nvfp4cb.sh — 35B MoE PRODUCTION A/B driver (research; NO HF publish)
# ============================================================================
# The goal's SECOND model class (35B MoE). Ornith-1.0-35B (Qwen3.6-35B-A3B-class
# packed MoE + MTP) through the EXPORT_CONTAINER=nvfp4_cb lane @4.75 bpp, then
# the gold-metric A/B vs the shipped rdtand/Ornith-1.0-35B-PrismaAURA-4.75bit
# (served, same BF16 session). GPU-or-bust. Single-box: launch only after the
# 27B run frees the GPU AND its verdict is reviewed (the approach must clear on
# 27B first).
#
# MoE ROUTE-FLIP CORRECTNESS (verified wired, run-pipeline.sh:917-973): the CB
# lane's COST_MODE=local is route-flip-blind on routed experts, so CB_EXPERT_
# EMPIRICAL=1 (default ON) auto-invokes prismaquant.expert_empirical_cost to
# MEASURE per-expert-unit KL end-to-end and merge it into the local CB payload
# (the M4-hybrid, 6b0a2f6). Non-expert Linears keep the local CB cost.
#
# MENU: stock NVFP4/FP8_DYNAMIC/BF16 + FP8_CB_K36..K48 (matches the 27B run;
# fp4-CB rungs need plugin v2-compose served-validation before inclusion).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- VERIFY-BEFORE-LAUNCH (flagged; menu agent confirms post-27B-verdict) ---
# (a) 35B base source: the shipped comparison is rdtand/Ornith-1.0-35B-PrismaAURA
#     -4.75bit, whose base is deepreinforce-ai/Ornith-1.0-35B. The prior AURA run
#     used that snapshot. Confirm it's on disk (the earlier ornith-35b-aura run
#     path) or download the base before launch.
export MODEL_PATH="${MODEL_PATH:-/home/rob/dq-runs/ornith-35b-base}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-35b-nvfp4cb-4p75}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

# --- CB lane contract ---
export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
# 2026-07-19 recipe (Robert-authorized): for the CB menu the empirical
# unit-KL is a perturbation floor (S=1 of 256 experts ≈ full-stack KL, memory
# cb_expert_unitkl_floor) and cannot rank CB rungs — expert stacks use the
# LOCAL weighted-MSE cost (per-expert imatrix incl. the down_proj replay
# synthesis) with MSE-unbiased stratified sampling (÷16 on the 1h45/shard
# encode wall) + the per-family floor-law rung ladder. Set
# CB_EXPERT_EMPIRICAL=1 only for menus of coarse formats whose unit KLs sit
# far above the floor (the original M4 regime).
export CB_EXPERT_EMPIRICAL=0
export PRISMAQUANT_EXPERT_COST_SAMPLE=16
export CB_LADDER_INTERP=1

# --- menu + budget (matched to the shipped PrismaAURA-4.75) ---
# Widened 6-rung FP8_CB ladder (K28/K32 added 2026-07-19: 3.5/4.0 bpw,
# served by the existing plugin kernels; the 0.6B smoke allocated across all
# six rungs at 4.5 bpp).
export FORMATS="NVFP4,FP8_DYNAMIC,BF16,FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48"
export TARGET_BITS=4.75
export PARETO_TARGETS="4.5,4.75,5.0"

# --- calibration: 35B > 27B, so calibrate CONSERVATIVELY to avoid the 128GB
#     unified-memory OOM the 27B hit at 32x1024. Start 8x512 (the shipped-27B
#     calibration size); bump only if headroom allows. Confirm at launch. ---
export NSAMPLES=8
export SEQLEN=512
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_SCALE_CODING=v1
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced   # the b9ca9e6 fast encode (~22s/big-Linear)
export VISUAL_FORMAT=BF16
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
# Resident export OOM-killed at 35B (66 GB source + accumulated outputs +
# GPU encode workspace share the 121 GB unified pool — the auto threshold's
# 80 GB source-size heuristic under-counts on MoE). The streaming exporter
# carries its own per-expert->stacked bridge (one expert resident at a time).
export EXPORT_STREAMING=1

echo "============================================================================"
echo "35B MoE PRODUCTION A/B — nvfp4_cb @${TARGET_BITS} bpp (M4-hybrid expert cost)"
echo "  MODEL=$MODEL_PATH"
echo "  FORMATS=$FORMATS   NSAMPLES x SEQLEN=${NSAMPLES}x${SEQLEN}"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NOTE: serve A/B needs DeltaNet --max-num-batched-tokens>=2096 + MTP handling"
echo "        (spec-decode OFF for the PPL gate); compare vs rdtand/Ornith-1.0-35B-"
echo "        PrismaAURA-4.75bit-vllm (on disk)."
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "##  35B EXPORT DONE — REVIEW: ${WORK_DIR}/exported_nvfp4_cb"
echo "##  Next: footprint verify, then served A/B (OURS vs Ornith-PrismaAURA-4.75 vs BF16)."
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
