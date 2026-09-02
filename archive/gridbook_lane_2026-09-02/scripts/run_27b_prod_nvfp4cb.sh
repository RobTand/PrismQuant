#!/usr/bin/env bash
# ============================================================================
# run_27b_prod_nvfp4cb.sh — 27B PRODUCTION A/B driver (research; NO HF publish)
# ============================================================================
# Qwen3.6-27B through the EXPORT_CONTAINER=nvfp4_cb lane @5.5 bpp, then the
# gold-metric A/B vs rdtand/Qwen3.6-27B-PrismaAURA-5.5bit (served, same BF16
# session). Overnight-scale. GPU-or-bust.
#
# MENU (deliberate): stock NVFP4/FP8_DYNAMIC/BF16 + FP8_CB_K36..K48. NO fp4-CB
# rungs — at 5.5 bpp the 4.5-6 bpw band is load-bearing and fp4-CB serving
# needs plugin v2-compose (separate workstream, serving agent).
#
# codebook-source=lattice: the COST_MODE=local cost path renders CB via the
# registry qdq (fixed lattice), so lattice export keeps the imatrix lockstep
# exact (cost == shipped bytes). learned would ship bytes the cost never saw.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/qwen36-27b-bf16}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/prod-27b-nvfp4cb-5p5}"
# NEVER /tmp (OOM-cleared before): keep scratch under the work dir.
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

# --- CB lane contract (gated by run-pipeline.sh) ---
export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0

# --- the deliberate 5.5 bpp mixed menu ---
export FORMATS="NVFP4,FP8_DYNAMIC,BF16,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48"
export TARGET_BITS=5.5
export PARETO_TARGETS="5.25,5.5,5.75"

# --- production calibration + CB lane knobs ---
# NOTE: the dispatch asked for 32x1024, but the 27B probe at 32x1024 (32768
# tokens) OOM-kills on 128 GB unified memory (per-layer activations x64 layers +
# the ~47 GB streaming weight cache spike past the pool at the phase-1->phase-3
# transition — deadman-caught 2026-07-17). The SHIPPED 27B artifacts used 8x512
# ("27b-repolish-8x512"); 8x1024 here honors SEQLEN=1024, gives 8192 tokens
# (2x the shipped calibration), and fits with comfortable margin. Documented in
# prod_27b_results.md as a deviation forced by the box, not a choice.
export NSAMPLES=8
export SEQLEN=1024
export CACHE_HEADROOM_GB=45            # extra activation headroom vs the OOM'd autoscale
export ACTIVATION_ROWS_LIMIT=1024      # CB lane auto-default; explicit for the record
export CB_SCALE_CODING=v1              # fp8-CB is v1-only anyway (no fp4-CB in menu)
export CB_CODEBOOK_SOURCE=lattice      # lockstep-exact (see header)
export PRISMAQUANT_CB_ENCODE_TIER=balanced  # explicit (it is the default); the
# at-scale FP8_CB per-slice encode is ~130 min/shard on a QUIET box (7 layers x
# ~54 Linears x 4 fp8-CB rungs) — the first clean at-scale encode-speed datum.
# Cost is skip-if-shard-exists: a resumed run picks up at the first missing shard.
export VISUAL_FORMAT=BF16              # VLM: visual Linears passthrough (-> ignore)
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda

echo "============================================================================"
echo "27B PRODUCTION A/B — nvfp4_cb @${TARGET_BITS} bpp"
echo "  MODEL=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS"
echo "  NSAMPLES x SEQLEN = ${NSAMPLES} x ${SEQLEN}   ACT_ROWS=${ACTIVATION_ROWS_LIMIT}"
echo "  CB: source=${CB_CODEBOOK_SOURCE} coding=${CB_SCALE_CODING}"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

# Per-stage wall-clock: run-pipeline.sh tees each stage to WORK_DIR/logs; the
# artifact mtimes (probe.pkl / cost.pkl / layer_config.json / exported_nvfp4_cb)
# give exact per-stage timing for the results doc.
PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"

echo
echo "############################################################################"
echo "##  EXPORT DONE — REVIEW: ${WORK_DIR}/exported_nvfp4_cb                     ##"
echo "##  Next: footprint verify, then serve+measure A/B (measure.py/kl_tool).   ##"
echo "############################################################################"
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
