#!/usr/bin/env bash
# ============================================================================
# run_laguna_s21_prod.sh — poolside Laguna-S-2.1 on the full gridbook standard
# ============================================================================
# Robert 2026-07-22: "go forward with Laguna. Leave enough room for 256k
# cache." The KV budget SETS the artifact size: GQA 8 kv-heads x 128 dim x
# 48 layers = ~96 KiB/token at fp8 KV -> 256k ctx ~= 24 GiB. Pool per Robert
# (2026-07-22: "128 GB, leave 3-4 for the OS; 110 is too conservative"):
# 121 GiB reported - 3.5 OS = 117.5 GiB; - 24 KV - ~9 (activations/graphs/
# host engine/DFlash) => ~84.5 GiB ~= 90 GB weight ceiling. Body ~116.7B
# params => ~6.0 bpp — the TOP of the CB ladder (K47/K48 band + FP8/BF16
# escapes); serve at util ~0.93 under the slack-gate + watchdog discipline.
#
# Menu: the full all-integer standard (STANDARDS.md) — 34 CB rungs + NVFP4 +
# FP8_DYNAMIC + BF16. 256-expert top-10 MoE, per-expert on disk (the
# Qwen3.5/Ornith bridge), shared expert per layer, sigmoid routing.
# Embeddings/lm_head BF16 (untied, 100k vocab = only ~1.2 GB tax).
#
# Drafter: poolside's DFlash (separate checkpoint, vLLM-native class) — NOT
# an in-body MTP; attach at serve time; rung/precision via the canon
# selector once the body footprint is known. NO QUALITY CLAIMS at 117B
# without a served teacher A/B protocol — gates are load + coherent gen +
# packing checks + speed + the uniform-quant comparisons.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/dq-runs/laguna-s21/source}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/laguna-s21/prod}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto
export CB_EXPERT_EMPIRICAL=0
export PRISMAQUANT_EXPERT_COST_SAMPLE=16

export FORMATS="$(PYTHONPATH="$REPO" python3 \
  "$REPO/scripts/print_cb_format_menu.py" NVFP4 FP8_DYNAMIC BF16)"
export CB_SCALE_CODING=two_tier
export TARGET_BITS="${TARGET_BITS:-6.0}"
export PARETO_TARGETS="${PARETO_TARGETS:-3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.5,5.75,6.0,6.25}"
# Robert 2026-07-22: see the size/quality/speed CURVE before the final
# size decision — dense Pareto ladder; the export that follows runs at
# TARGET_BITS as a default and is cheap to interrupt/re-point.

export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISMAQUANT_PROBE_MIN_AVAILABLE_GB=40
export LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"

echo "============================================================================"
echo "Laguna-S-2.1 FULL-STANDARD CB @ ${TARGET_BITS} bpp (ship pick: footprint <= 90 GB"
echo "  so 256k fp8 KV [~24 GiB] + DFlash drafter fit the Spark pool)"
echo "  FORMATS=$FORMATS"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

cd "${REPO}"
PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  PYTHONPATH="${REPO}" \
  bash "${REPO}/prismaquant/run-pipeline.sh"
echo "## LAGUNA EXPORT DONE — next: DFlash drafter (canon selector), serve smoke"
echo "## at 256k ctx (util per slack gate), speed, uniform-quant A/B."
