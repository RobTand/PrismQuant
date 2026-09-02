#!/usr/bin/env bash
# ============================================================================
# run_hy3_prod_nvfp4cb.sh — Hy3 295B-A21B ULTRA-LOW-BPP driver (research)
# ============================================================================
# The goal's THIRD model class (200-300B ultra-low-bpp). Tencent Hy3
# (295B-A21B MoE, hy_v3 profile, 192 experts) from BF16 source through the
# STREAMING CB exporter @ ~2.9 bpp — the flagship raison d'être of the CB lane:
# a native-tensor-core-servable ultra-low-bit artifact on a single DGX Spark,
# WITHOUT IQ's prefill tax.
#
# GATED (launch only after): (a) 27B + 35B verdicts clear the approach; (b) the
# GPU is free; (c) v2-compose fp4-CB SERVING is validated end-to-end (serving
# agent built it — commit for the plugin v2-compose; the first-served-v2 smoke
# must pass before an fp4-CB-dominated artifact can be SERVED, not just exported);
# (d) hy_v3 vLLM adapter present (from the earlier GGUF Hy3 serving work).
#
# NO QUALITY CLAIMS (Robert's standing instruction: no one can KL-validate a
# 295B against its BF16 teacher on this hardware). Validation = loads + coherent
# generation + bit-exact packing, per the shipped GGUF Hy3.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/dq-runs/hy3-prod/source}"   # bf16, 557GB, on disk
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

# --- CB lane + STREAMING (Hy3 never fits resident: 557GB >> 121GB pool) ---
export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
# 2026-07-19 recipe (per the 35B-proven corrections): the empirical unit-KL
# is a perturbation floor at CB fidelity (memory cb_expert_unitkl_floor) and
# cannot rank CB rungs — and at 295B it would run for days. Expert stacks use
# the LOCAL weighted-MSE cost (per-expert imatrix incl the down_proj routed
# replay) with MSE-unbiased sampling (192 experts -> ÷12) + the per-family
# floor-law rung ladder.
export CB_EXPERT_EMPIRICAL=0
export PRISMAQUANT_EXPERT_COST_SAMPLE=16
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto            # >=80GB source auto-streams (export_nvfp4_cb_streaming, 1ff7185)

# --- ultra-low-bpp menu: fp4-CB v2 (two-tier, 2.0-3.5bpp) dominant + fp8-CB for
#     the sensitive tail + BF16 sidecars. TWO-TIER v2 for the byte win at this
#     band (0.28 vs 0.5 bpw scale). fp4-CB SERVING needs v2-compose (gate c).
#     fp8 tail widened to the 4-rung K28..K44 ladder (K28/K32 added 2026-07-19;
#     3.5/4.0 bpw steps for the sensitive tail, and 4 rungs make the fp8
#     family ladder-interpolable). ---
export FORMATS="NVFP4_CB_K14,NVFP4_CB_K16,NVFP4_CB_K18,NVFP4_CB_K20,FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K44,BF16"
export CB_SCALE_CODING=two_tier        # v2 (premium-flip; exp-1c GO)
export TARGET_BITS=2.9                 # single-Spark fit (matches shipped GGUF Hy3 2.8bpp)
export PARETO_TARGETS="2.7,2.9,3.1"

# --- streaming probe/cost (bounded residency); conservative calibration ---
export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45   # shard-1 OOM 2026-07-19: 84.9GB cache budget + bwd workspace overran the pool
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PRISMAQUANT_PROBE_PREFETCH_LOOKAHEAD=2   # Hy3 probe OOM guard (2026-07 lesson)
# Streaming export OOM 2026-07-19 (global OOM 2GB into the write, tiny CPU
# RSS): 10GB-class per-target pack transients fragment the caching allocator
# on the 121GB unified pool. Expandable segments + per-tensor empty_cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISMAQUANT_PROBE_MIN_AVAILABLE_GB=40
# auto picked 1 layer/shard at 295B -> 80 shards x ~12 min reverse sweeps
# (~16 h). The sweep cost is per SHARD (all-layer reload), so 8 layers/shard
# = 10 sweeps (~2.5 h) at unchanged residency (the pressure is the layer
# cache, not tracked-Linear state; 35B ran 14/shard).
export LAYERS_PER_SHARD=8

echo "============================================================================"
echo "Hy3 295B-A21B ULTRA-LOW-BPP — nvfp4_cb STREAMING @${TARGET_BITS} bpp (v2 two-tier)"
echo "  MODEL=$MODEL_PATH  (bf16 source, streaming export)"
echo "  FORMATS=$FORMATS"
echo "  NO QUALITY CLAIMS. Validation = load + coherent-gen + bit-exact packing."
echo "  PRE-LAUNCH GATES: 27B+35B verdicts clear; GPU free; v2-compose fp4-CB"
echo "    SERVING validated; hy_v3 vLLM adapter present. exclude layers.80 (MTP)."
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "##  Hy3 EXPORT DONE — REVIEW: ${WORK_DIR}/exported_nvfp4_cb"
echo "##  Next: footprint (single-Spark fit?), load+coherent-gen smoke, TTFT/decode vs the"
echo "##  shipped GGUF Hy3 2.8bpp (the native-vs-IQ-prefill comparison — the whole thesis)."
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
