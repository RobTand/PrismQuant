#!/usr/bin/env bash
# ============================================================================
# run_27b_cb_20gb.sh — Qwen3.6-27B on the FULL gridbook stack, 20 GB target
# ============================================================================
# Robert 2026-07-21: "create a Qwen 3.6 27B using the new stack and serve with
# the new kernels; target 20 GB so it runs on a 24 GB 4090 or 5090."
#
# MENU: the full producer CB ladder (fp4 K1-K25 two-tier + fp8 step-four,
# 0.125-bpw steps — landed 2026-07-21, 116-test gate) + vanilla NVFP4 +
# FP8_DYNAMIC + BF16. Robert 2026-07-21: "Blackwell only is fine" + "Put
# nvfp4 in the menu for completeness sake" — the allocator decides per-Linear
# (35B dense-tier A/B: 42 outlier-row attention units favored NVFP4's
# act-weighted group-16 scales; on Hy3's MoE-dominated joint menu it won
# zero — measurement decides here). Target cards: RTX 5090 (sm_120) / GB10
# (sm_121).
#
# VISION TOWER: NVFP4 (Robert: "Make the vision tower nvfp4") — was BF16
# passthrough in the prior run. VISUAL_FORMAT accepts any registry name the
# target profile allows; text-only calibration means visual Linears render
# RTN (no visual activations), which is the standard vision-tower treatment.
# The serve smoke must include an IMAGE request (vision path load+generate).
#
# SIZE: 20 GB total artifact. Prior 5.5-bpp CB export = 23 GB with ~5.1 GB
# BF16 embed+lm_head (248k vocab, untied) => body ~25B params. 20 GB total
# - 5.1 embeds - ~0.7 visual/norms - ~0.7 CB MTP => body ~13.5 GB ~ 4.3-4.6
# bpp. TARGET_BITS=4.5 with a 4.25/4.75 Pareto bracket; the ship pick is by
# EXACT footprint <= 19.3 GB pre-MTP (nvfp4_cb_footprint is authoritative),
# NOT by the bpp label.
#
# REUSE from prod-27b-nvfp4cb-5p5 (menu-independent): probe.pkl (+settings),
# cb_col_weights.pkl. Cost is measured FRESH across the new menu (ladder
# interpolation keeps it anchor-priced). Act cache regenerates (prior was
# cleaned).
#
# MTP: mtp.* (15 keys) is encoded post-body via the canon throughput selector
# (prismaquant/mtp_rung_selection.py) and merged — "always include MTP".
# Visual tower: BF16 passthrough -> ignore list (unchanged from prior run).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/qwen36-27b-bf16}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-27b-cb-20gb}"
export TMPDIR="${WORK_DIR}/tmp"
PRIOR="/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/artifacts"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs" "$WORK_DIR/artifacts"

# --- probe + col-weights reuse (menu-independent; settings travel along) ---
for f in cb_col_weights.pkl; do  # probe NOT reusable without its act/ cache (the probe stage writes activations)
  if [ -f "$PRIOR/$f" ] && [ ! -f "$WORK_DIR/artifacts/$f" ]; then
    cp -v "$PRIOR/$f" "$WORK_DIR/artifacts/$f"
  fi
done

# --- CB lane contract ---
export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_LADDER_INTERP=1
export EXPORT_STREAMING=auto

# --- FULL ladder, no vanilla NVFP4 (see header) ---
export FORMATS="$(PYTHONPATH="$REPO" python3 \
  "$REPO/scripts/print_cb_format_menu.py" NVFP4 FP8_DYNAMIC BF16)"
export CB_SCALE_CODING=two_tier
# Overridable for additional SKUs from the SAME probe/cost (allocation +
# export only). 16 GB SKU (5080/5070-Ti): TARGET_BITS=2.4 MTP_FORMAT=CB.
export TARGET_BITS="${TARGET_BITS:-4.5}"
export PARETO_TARGETS="${PARETO_TARGETS:-4.25,4.5,4.75}"
export MTP_FORMAT="${MTP_FORMAT:-BF16}"

# --- calibration (box-forced 8x1024, same as the prior 27B CB run) ---
export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45
export ACTIVATION_ROWS_LIMIT=1024
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export VISUAL_FORMAT=NVFP4
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"

echo "============================================================================"
echo "Qwen3.6-27B FULL-LADDER CB @ ${TARGET_BITS} bpp (ship pick: footprint <= 19.3 GB)"
echo "  WORK_DIR=$WORK_DIR (probe/col-weights reused from prod-27b-nvfp4cb-5p5)"
echo "  FORMATS=$FORMATS"
echo "  Blackwell target (5090/GB10): CB ladder + NVFP4 + FP8_DYNAMIC + BF16; vision tower NVFP4."
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "##  BODY EXPORT DONE — next: MTP canon encode (mtp.*) + merge, footprint"
echo "##  check vs 20 GB, serve smoke (Spark), gold-lane KL-vs-BF16 + TEB."
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
