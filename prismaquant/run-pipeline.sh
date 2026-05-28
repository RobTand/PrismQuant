#!/usr/bin/env bash
# run-pipeline.sh — end-to-end PrismaQuant pipeline: probe → cost →
# allocator → native compressed-tensors export → vLLM validate.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
#   WORK_DIR=./dq-runs/qwen36 \
#   FORMATS=NVFP4,FP8_DYNAMIC,BF16 \
#   TARGET_BITS=4.75 \
#   VISUAL_FORMAT=BF16 \
#   CALIBRATION_MODALITY=text-only \
#   ./prismaquant/run-pipeline.sh
#
# VISUAL_FORMAT accepts any format registry name allowed by the target
# serving profile and applies to visual-encoder Linears on multimodal models.
# In text-only calibration mode it's the Phase 1 uniform override for every
# visual Linear. In multimodal mode it's the fallback applied to un-probed
# visual Linears the allocator's Fisher-driven DP didn't touch (plus the
# graceful OOM fallback on 122B-scale VLMs that can't fit the whole model in
# RAM for the multimodal pass).
#
# CALIBRATION_MODALITY (text-only | multimodal):
#   - text-only (default): body-only streaming probe + cost. Visual
#     shards emit empty pickles; allocator stamps all visual Linears
#     with --visual-format uniformly.
#   - multimodal: also runs a non-streaming second probe pass with
#     image+text calibration (synthetic stub by default; set MM_DATASET
#     to a HuggingFace dataset id to use real images). The allocator
#     treats visual Linears as regular DP candidates when real Fisher
#     stats are present. Requires ~full-model RAM; falls back to
#     text-only behavior automatically on OOM.
#
# Memory note: probe + cost peak around 90 GB on a 35B model under
# BF16 calibration. The watchdog in incremental_measure_quant_cost
# aborts cleanly on swap pressure rather than OOM-killing the host.
#
# MTP is folded into the incremental probe + cost as a built-in shard;
# mtp.* tensors are measured in the same pass as the body and land in
# the same probe/cost pickles. No separate MTP stages.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the source HF model directory}"
: "${WORK_DIR:?Set WORK_DIR to a writable directory for artifacts}"
: "${FORMATS:=NVFP4,FP8_DYNAMIC,BF16}"
: "${TARGET_BITS:=4.75}"
: "${PARETO_TARGETS:=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"
# Calibration defaults. 4x256 was the historical minimum for correctness
# validation; 32x1024 (N=32, T=1024 = 32768 tokens/sample, 32 samples)
# produces ~7% lower PPL on the resulting quantized artifact at a
# linear time cost in probe wall-time. Override for faster iteration.
: "${NSAMPLES:=32}"
: "${SEQLEN:=1024}"
# LAYERS_PER_SHARD controls how many decoder layers share one reverse
# sweep for Fisher accumulation. Larger = fewer sweeps = faster probe,
# but each shard needs more gradient + retained-activation memory.
# Default `auto` asks prismaquant.autoscale to pick from available RAM +
# model size. Set to an int (e.g. `2`) to override.
: "${LAYERS_PER_SHARD:=auto}"
# CACHE_HEADROOM_GB controls the streaming layer-cache budget
# (cache = free_RAM - headroom). Default `auto` picks from model +
# LAYERS_PER_SHARD. Set to a float to override.
: "${CACHE_HEADROOM_GB:=auto}"
: "${PREFETCH_LOOKAHEAD:=auto}"
: "${PREFETCH_WORKERS:=auto}"
: "${PREFETCH_MIN_AVAILABLE_GB:=auto}"
: "${ACTIVATION_ROWS_LIMIT:=256}"
: "${DATASET:=/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
: "${DEVICE:=cuda}"
: "${EXPORT_DEVICE:=cuda}"   # CUDA ~10× faster than CPU on NVFP4 packing
: "${TARGET_PROFILE:=vllm_packed_moe}"

if [[ "$DEVICE" != cuda* || "$EXPORT_DEVICE" != cuda* ]]; then
  echo "[pipeline] ERROR: PrismaQuant production pipeline is GPU-or-bust; DEVICE and EXPORT_DEVICE must be cuda*" >&2
  exit 2
fi
python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[pipeline] ERROR: CUDA is not available; refusing CPU quantization", file=sys.stderr)
    raise SystemExit(2)
PY
# Visual encoder format: fallback for visual Linears. See header docstring
# for the full text-only vs multimodal semantics. BF16 (default) is
# passthrough; non-BF16 registry formats uniformly quantize
# non-Fisher-allocated visual Linears via the existing RTN math at export time.
: "${VISUAL_FORMAT:=BF16}"
# Calibration modality. `text-only` (default) is the streaming body probe
# alone; `multimodal` adds a second non-streaming pass over the full
# model with image+text calibration so visual Linears get real Fisher
# stats. See header docstring.
: "${CALIBRATION_MODALITY:=text-only}"
# Multimodal dataset. `synthetic` is the offline stub that exercises the
# code path without network. Set to an HF dataset id (e.g.
# `HuggingFaceM4/COCO`) for real image calibration when
# CALIBRATION_MODALITY=multimodal.
: "${MM_DATASET:=synthetic}"
: "${MTP_FORMAT:=BF16}"
# Production-cache export path. Enabled by default so export packs the same
# rendered weights that KL/polish paths measure. Re-cache is enabled by default
# after the Qwen3.5-0.8B and Qwen3-4B smoke ladder cleared vLLM eager/graph
# serving; set PRODUCTION_RECACHE=0 for an explicit no-recache ablation.
: "${PRODUCTION_CACHE:=1}"
: "${PRODUCTION_RECACHE:=1}"
: "${PRODUCTION_CACHE_MAX_ACT_ROWS:=512}"
: "${PRODUCTION_CACHE_LRU_GB:=64.0}"
: "${PRODUCTION_CACHE_PREFETCH:=require}"
: "${PRODUCTION_CACHE_PREFETCH_WORKERS:=4}"
: "${PRODUCTION_RECACHE_MICROBATCH:=1}"
: "${PRODUCTION_CACHE_FORMATS:=auto}"
# assignment: render only concrete non-BF16 entries from layer_config.json.
# format-menu: render every requested format for every quantizable Linear,
# useful when intentionally building a reusable cache for reallocations.
: "${PRODUCTION_CACHE_RENDER_SCOPE:=assignment}"
: "${PRODUCTION_CACHE_LEVERS:=gptq,static_act_order,joint_scale_opt}"
: "${PRODUCTION_CACHE_DISABLE_LEVERS:=}"
# PRODUCTION_CACHE_UNION=1 switches the validated-surrogate frontier path
# from full format-menu rendering (every Linear × every quantized format)
# to a smart-union render that only adds FP8_DYNAMIC fallback entries for
# Linears whose NVFP4 output_mse exceeds the band threshold. It deliberately
# does not render MXFP8 or E5M2 fallbacks in the production path.
: "${PRODUCTION_CACHE_UNION:=0}"
: "${PRODUCTION_CACHE_UNION_P_FP8:=0.75}"
: "${COST_MODE:=production-render-score}"
: "${PRODUCTION_RENDER_COST_NSAMPLES:=8}"
: "${PRODUCTION_RENDER_COST_SEQLEN:=1024}"
: "${PRODUCTION_RENDER_COST_SEED:=42}"
: "${PRODUCTION_RENDER_COST_SCORE_FIELD:=output_mse}"
: "${PRODUCTION_RENDER_COST_REQUIRE_SCORES:=0}"
: "${PRODUCTION_RENDER_COST_REQUIRE_OUTPUT:=1}"
: "${PRODUCTION_RENDER_COST_PROMOTE_FRACTION:=0.30}"
: "${PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE:=}"
: "${PRODUCTION_RENDER_COST_MAX_PROMOTIONS:=}"
# Optional post-frontier MSE promotion rewrite. This is disabled by default
# and only promotes existing quantized assignments to a higher target format
# (BF16 by default), so it does not require additional production-cache
# renders. It is intended as an explicit recipe arm for testing whether local
# output-MSE can protect structurally sensitive regions without KL replay.
: "${MSE_PROMOTION:=0}"
: "${MSE_PROMOTION_CATEGORIES:=linear_attn,self_attn}"
: "${MSE_PROMOTION_TARGET_FORMAT:=BF16}"
: "${MSE_PROMOTION_MAX_BPP_DELTA:=}"
: "${MSE_PROMOTION_TARGET_BPP:=}"
: "${MSE_PROMOTION_GROUP_BY:=serving_unit}"
: "${MSE_PROMOTION_METRIC:=output_mse_per_bit}"
: "${ALLOC_PROPAGATED_SENSITIVITY_REPORT:=}"
: "${ALLOC_PROPAGATED_SENSITIVITY_SCALE:=1.0}"
: "${ALLOC_PROPAGATED_SENSITIVITY_SCORE_FIELD:=propagated_kl}"
: "${ALLOC_PROPAGATED_SENSITIVITY_FORMAT_EXTRAPOLATION:=local_mse_ratio}"
: "${ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT:=}"
: "${EXPORT_GPTQ:=auto}"
: "${EXPORT_SCALE_SWEEP:=auto}"
: "${PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json}"
: "${PIPELINE_SPEC_VALIDATE:=1}"
# Fisher-weighted GPTQ + Fisher output-MSE allocator are archived under
# archive/fisher_2026-05-15/ (the row-weighting + allocator-objective code
# paths remain in the live tree but are unreachable from the production
# pipeline). Any explicit override here fails fast.
: "${FISHER_WEIGHTED_GPTQ:=0}"
: "${FISHER_OUTPUT_MSE_ALLOCATOR:=0}"
for _legacy_fisher_var in FISHER_WEIGHTED_GPTQ FISHER_OUTPUT_MSE_ALLOCATOR; do
  case "${!_legacy_fisher_var}" in
    0|false|False|FALSE|no|No|NO|"") ;;
    *)
      echo "[pipeline] ERROR: $_legacy_fisher_var=${!_legacy_fisher_var} — Fisher levers are archived under archive/fisher_2026-05-15." >&2
      exit 2
      ;;
  esac
done
unset _legacy_fisher_var
case ",$PRODUCTION_CACHE_LEVERS," in
  *,fisher_gptq,*)
    echo "[pipeline] ERROR: PRODUCTION_CACHE_LEVERS includes fisher_gptq — Fisher levers are archived under archive/fisher_2026-05-15." >&2
    exit 2
    ;;
esac
export PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=0
: "${H_DETAIL_DIR:=${WORK_DIR}/h_detail}"
: "${SELECTION_MODE:=surrogate}"
: "${VALIDATED_FRONTIER_NSAMPLES:=$NSAMPLES}"
: "${VALIDATED_FRONTIER_SEQLEN:=$SEQLEN}"
: "${VALIDATED_FRONTIER_KL_SCOPE:=last_token}"
: "${VALIDATED_FRONTIER_PICK:=kneedle}"
: "${VALIDATED_SOURCE_PREFETCH:=require}"
: "${VALIDATED_SOURCE_PREFETCH_MAX_GB:=0}"
: "${VALIDATED_SOURCE_PREFETCH_HEADROOM_GB:=16}"
: "${VALIDATED_SOURCE_PREFETCH_WORKERS:=2}"
: "${VALIDATED_FRONTIER_KL_CUDA_GRAPHS:=auto}"
: "${VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE:=0}"

PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"
case "$COST_MODE" in
  local)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    COST_PATH="${BASE_COST_PATH}"
    PRODUCTION_RENDER_COST_CACHE_PATH=""
    PRODUCTION_RENDER_COST_CACHE_DIR=""
    PRODUCTION_RENDER_COST_TAIL_QNAMES=""
    ;;
  grouped-kl)
    echo "[pipeline] ERROR: COST_MODE=grouped-kl — the grouped-KL (fusion-matched) cost surrogate is archived under archive/grouped_kl_2026-05-28. It fixed a local allocator non-monotonicity but LOST the shipped vLLM A/B on Qwen3.6-27B (worse exact vLLM KL and direct WikiText PPL than the shipped 5.5 artifact); see archive/grouped_kl_2026-05-28/README.md. Use production-render-score (default), production-render-staged, or local." >&2
    exit 2
    ;;
  production-render-score|production-render)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_cache.pkl"
    PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_weight_cache"
    PRODUCTION_RENDER_COST_TAIL_QNAMES=""
    ;;
  production-render-staged|production-render-tail)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_staged_cache.pkl"
    PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_staged_weight_cache"
    PRODUCTION_RENDER_COST_TAIL_QNAMES="${WORK_DIR}/artifacts/production_render_score_tail_qnames.txt"
    ;;
  *)
    echo "[pipeline] ERROR: COST_MODE must be local, production-render-score, or production-render-staged" >&2
    exit 2
    ;;
esac

PROBE_H_DETAIL_ARGS=()
COST_H_DETAIL_ARGS=()
PROD_H_DETAIL_ARGS=()
case "${HADAMARD_DUQUANT:-}" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14 and is not available on the production path." >&2
    exit 2
    ;;
esac
case ",$PRODUCTION_CACHE_LEVERS," in
  *,hadamard_duquant,*)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14." >&2
    exit 2
    ;;
esac
case "${MULTI_SHOT_PASSES:-1}" in
  ""|1) ;;
  *)
    echo "[pipeline] ERROR: MULTI_SHOT_PASSES=${MULTI_SHOT_PASSES} — multi-shot recalibration is archived under archive/multi_shot_2026-05-19 (Qwen3-4B production-cal showed ΔKL=0 at 5/5 budgets; small-cal cross-eval showed mean -42.5% gap-closed with one budget regressing -154%). Not a production lever." >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_DIR}"/{artifacts,act,work,logs,exported}

case "$SELECTION_MODE" in
  surrogate|validated-surrogate) ;;
  *)
    echo "[pipeline] ERROR: SELECTION_MODE must be surrogate or validated-surrogate" >&2
    exit 2
    ;;
esac

echo "[pipeline] config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS  TARGET_BITS=$TARGET_BITS"
echo "  NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN LAYERS_PER_SHARD=$LAYERS_PER_SHARD"
echo "  PREFETCH_LOOKAHEAD=$PREFETCH_LOOKAHEAD PREFETCH_WORKERS=$PREFETCH_WORKERS"
echo "  ACTIVATION_ROWS_LIMIT=$ACTIVATION_ROWS_LIMIT"
echo "  VISUAL_FORMAT=$VISUAL_FORMAT"
echo "  MTP_FORMAT=$MTP_FORMAT"
echo "  CALIBRATION_MODALITY=$CALIBRATION_MODALITY  MM_DATASET=$MM_DATASET"
echo "  PRODUCTION_CACHE=$PRODUCTION_CACHE PRODUCTION_RECACHE=$PRODUCTION_RECACHE"
echo "  PRODUCTION_CACHE_FORMATS=$PRODUCTION_CACHE_FORMATS"
echo "  PRODUCTION_CACHE_RENDER_SCOPE=$PRODUCTION_CACHE_RENDER_SCOPE"
echo "  PRODUCTION_CACHE_LEVERS=$PRODUCTION_CACHE_LEVERS"
echo "  PRODUCTION_CACHE_DISABLE_LEVERS=$PRODUCTION_CACHE_DISABLE_LEVERS"
echo "  COST_MODE=$COST_MODE"
if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" || "$COST_MODE" == "production-render-staged" || "$COST_MODE" == "production-render-tail" ]]; then
  echo "  PRODUCTION_RENDER_COST_NSAMPLES=$PRODUCTION_RENDER_COST_NSAMPLES PRODUCTION_RENDER_COST_SEQLEN=$PRODUCTION_RENDER_COST_SEQLEN PRODUCTION_RENDER_COST_SEED=$PRODUCTION_RENDER_COST_SEED SCORE_FIELD=$PRODUCTION_RENDER_COST_SCORE_FIELD"
fi
echo "  EXPORT_GPTQ=$EXPORT_GPTQ EXPORT_SCALE_SWEEP=$EXPORT_SCALE_SWEEP"
echo "  PIPELINE_SPEC_PATH=$PIPELINE_SPEC_PATH"
echo "  PRISMAQUANT_NVFP4_SCALE_RULE=${PRISMAQUANT_NVFP4_SCALE_RULE:-static_6}"
echo "  H_DETAIL_DIR=$H_DETAIL_DIR"
echo "  PRODUCTION_CACHE_LRU_GB=$PRODUCTION_CACHE_LRU_GB PRODUCTION_CACHE_PREFETCH=$PRODUCTION_CACHE_PREFETCH"
echo "  SELECTION_MODE=$SELECTION_MODE VALIDATED_FRONTIER_NSAMPLES=$VALIDATED_FRONTIER_NSAMPLES VALIDATED_FRONTIER_SEQLEN=$VALIDATED_FRONTIER_SEQLEN VALIDATED_FRONTIER_PICK=$VALIDATED_FRONTIER_PICK"
echo

PIPELINE_SPEC_ARGS=(
  python3 -m prismaquant.pipeline
  --write-default-production "$PIPELINE_SPEC_PATH"
  --render-mechanisms "$PRODUCTION_CACHE_LEVERS"
  --disable-render-mechanisms "$PRODUCTION_CACHE_DISABLE_LEVERS"
  --model-path "$MODEL_PATH"
  --work-dir "$WORK_DIR"
  --formats "$FORMATS"
  --target-bits "$TARGET_BITS"
  --target-profile "$TARGET_PROFILE"
  --calibration-modality "$CALIBRATION_MODALITY"
  --selection-mode "$SELECTION_MODE"
  --production-cache "$PRODUCTION_CACHE"
  --production-recache "$PRODUCTION_RECACHE"
)
case "$PIPELINE_SPEC_VALIDATE" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *) PIPELINE_SPEC_ARGS+=(--validate) ;;
esac
"${PIPELINE_SPEC_ARGS[@]}"

# -----------------------------------------------------------------------
# 1. Sensitivity probe (per-Linear empirical Fisher diagonal trace,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
if [[ ! -f "${PROBE_PATH}" ]]; then
  echo "[pipeline] [1/4] running sensitivity probe ..."
  python3 -m prismaquant.incremental_probe \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
    --device "$DEVICE" --dtype bf16 \
    --output "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --work-dir "${WORK_DIR}/work" \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --prefetch-lookahead "$PREFETCH_LOOKAHEAD" \
    --prefetch-workers "$PREFETCH_WORKERS" \
    --prefetch-min-available-gb "$PREFETCH_MIN_AVAILABLE_GB" \
    --activation-rows-limit "$ACTIVATION_ROWS_LIMIT" \
    --calibration-modality "$CALIBRATION_MODALITY" \
    --mm-dataset "$MM_DATASET" \
    --mm-nsamples 8 --mm-max-text-len 128 \
    "${PROBE_H_DETAIL_ARGS[@]}" \
    2>&1 | tee "${WORK_DIR}/logs/probe.log"
else
  # Reuse guard: make sure the pre-existing probe.pkl matches the
  # currently-requested calibration modality. Silently reusing a
  # text-only probe under multimodal (or vice versa) would produce
  # an assignment calibrated for the wrong activation distribution
  # — visible later as bad PPL that's hard to root-cause. Fail loud
  # and point the user at the file to delete.
  probe_modality=$(python3 -c "
import pickle, sys
try:
    with open(sys.argv[1], 'rb') as f:
        blob = pickle.load(f)
    meta = blob.get('meta', {}) if isinstance(blob, dict) else {}
    m = meta.get('calibration_modality') or meta.get('modality') or 'text-only'
    print(m)
except Exception as e:
    print(f'__error__:{e}', file=sys.stderr)
    sys.exit(2)
" "${PROBE_PATH}" 2>/dev/null || echo "__unknown__")
  if [[ "${probe_modality}" == "__unknown__" ]]; then
    echo "[pipeline] [1/4] probe.pkl exists but its calibration_modality"
    echo "             could not be read. Aborting to avoid mixing probes."
    echo "             Delete it explicitly to regenerate:"
    echo "               rm ${PROBE_PATH}"
    exit 2
  fi
  if [[ "${probe_modality}" != "${CALIBRATION_MODALITY}" ]]; then
    echo "[pipeline] [1/4] ABORT: probe.pkl was calibrated for"
    echo "             modality='${probe_modality}' but this run requests"
    echo "             CALIBRATION_MODALITY='${CALIBRATION_MODALITY}'."
    echo "             Reusing the probe would silently produce an"
    echo "             assignment calibrated on the wrong activations."
    echo ""
    echo "             Delete the stale probe to regenerate:"
    echo "               rm ${PROBE_PATH}"
    echo "             Or unset CALIBRATION_MODALITY to match the probe."
    exit 2
  fi
  if [[ "${#PROBE_H_DETAIL_ARGS[@]}" -gt 0 && ! -d "$H_DETAIL_DIR" ]]; then
    echo "[pipeline] [1/4] ABORT: Fisher h-detail was requested but"
    echo "             h-detail dir is missing: $H_DETAIL_DIR"
    echo "             Delete ${PROBE_PATH} to regenerate probe+h-detail."
    exit 2
  fi
  echo "[pipeline] [1/4] probe.pkl exists (modality=${probe_modality}), skipping"
fi

# -----------------------------------------------------------------------
# 2. Cost measurement (per-(Linear, format) measured RTN error,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
if [[ ! -f "${BASE_COST_PATH}" ]]; then
  echo "[pipeline] [2/4] measuring per-(layer, format) cost ..."
  python3 -m prismaquant.incremental_measure_quant_cost \
    --model "$MODEL_PATH" \
    --probe "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --formats "$FORMATS" \
    --output "${BASE_COST_PATH}" \
    --work-dir "${WORK_DIR}/work" \
    --device "$DEVICE" --dtype bf16 \
    --mode batched --chunk-size 256 \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --skip-missing-activations \
    --no-include-lm-head \
    --swap-grow-limit-mb "${SWAP_GROW_LIMIT_MB:-2048}" \
    "${COST_H_DETAIL_ARGS[@]}" \
    2>&1 | tee "${WORK_DIR}/logs/cost.log"
else
  if [[ "${#COST_H_DETAIL_ARGS[@]}" -gt 0 ]]; then
    cost_h_detail_status=$(python3 - "$BASE_COST_PATH" "$H_DETAIL_DIR" <<'PY'
import pickle
import sys
from pathlib import Path

with open(sys.argv[1], "rb") as f:
    blob = pickle.load(f)
meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
expected = str(Path(sys.argv[2]))
actual = meta.get("h_detail_dir")
if actual is None:
    shards = meta.get("shards") or []
    vals = {
        (
            s.get("h_detail_dir")
            or (
                s.get("incremental_shard", {}).get("h_detail_dir")
                if isinstance(s.get("incremental_shard"), dict)
                else None
            )
        )
        for s in shards
        if isinstance(s, dict)
    }
    vals.discard(None)
    if len(vals) == 1:
        actual = next(iter(vals))
    elif len(vals) > 1:
        actual = "__mixed__"
print("ok" if actual == expected else f"bad:{actual}")
PY
)
    if [[ "$cost_h_detail_status" != "ok" ]]; then
      echo "[pipeline] [2/4] ABORT: cost.pkl was not measured with requested h-detail dir."
      echo "             expected: $H_DETAIL_DIR"
      echo "             actual:   ${cost_h_detail_status#bad:}"
      echo "             Delete stale cost artifacts to regenerate:"
      echo "               rm ${BASE_COST_PATH}"
      echo "               rm -rf ${WORK_DIR}/work/shards"
      exit 2
    fi
  fi
  echo "[pipeline] [2/4] baseline cost exists, skipping"
fi
# Fisher output-MSE allocator cost-status check removed; the archive guard
# at the top of this file errors out before reaching here if the var is set.

if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" ]]; then
  if [[ ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2b/4] rendering production weights for allocator cost ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$FORMATS" \
      --render-scope format-menu \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_cache.log"
  else
    echo "[pipeline] [2b/4] production-render cost cache exists, skipping"
  fi
  if [[ ! -f "$COST_PATH" ]]; then
    echo "[pipeline] [2c/4] synthesizing production-render allocator cost ..."
    PROD_RENDER_COST_ARGS=()
    case "$PRODUCTION_RENDER_COST_REQUIRE_SCORES" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        PROD_RENDER_COST_ARGS+=(--require-render-scores)
        ;;
    esac
    case "$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
      *)
        PROD_RENDER_COST_ARGS+=(--require-output-metric)
        ;;
    esac
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --baseline-cost "$BASE_COST_PATH" \
      --output "$COST_PATH" \
      --formats "$FORMATS" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      "${PROD_RENDER_COST_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_cost.log"
  else
    echo "[pipeline] [2c/4] production-render allocator cost exists, skipping"
  fi
fi

if [[ "$COST_MODE" == "production-render-staged" || "$COST_MODE" == "production-render-tail" ]]; then
  PRODUCTION_RENDER_COST_NVFP4_CACHE="${WORK_DIR}/artifacts/production_render_score_staged_nvfp4_cache.pkl"
  PRODUCTION_RENDER_COST_TAIL_SUMMARY="${WORK_DIR}/artifacts/production_render_score_tail_summary.json"
  if [[ ! -f "$PRODUCTION_RENDER_COST_NVFP4_CACHE" ]]; then
    echo "[pipeline] [2b/4] rendering NVFP4 production weights for staged allocator cost ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_NVFP4_CACHE" \
      --formats NVFP4 \
      --render-scope format-menu \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_nvfp4_cache.log"
  else
    echo "[pipeline] [2b/4] staged NVFP4 production-render cache exists, skipping"
  fi
  if [[ ! -f "$PRODUCTION_RENDER_COST_TAIL_QNAMES" ]]; then
    echo "[pipeline] [2c/4] selecting high-error NVFP4 tail for staged promotions ..."
    SELECT_TAIL_ARGS=()
    if [[ -n "$PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE" ]]; then
      SELECT_TAIL_ARGS+=(--select-tail-min-score "$PRODUCTION_RENDER_COST_MIN_PROMOTE_SCORE")
    fi
    if [[ -n "$PRODUCTION_RENDER_COST_MAX_PROMOTIONS" ]]; then
      SELECT_TAIL_ARGS+=(--select-tail-max-count "$PRODUCTION_RENDER_COST_MAX_PROMOTIONS")
    fi
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_NVFP4_CACHE" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      --select-tail-output "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --select-tail-summary "$PRODUCTION_RENDER_COST_TAIL_SUMMARY" \
      --select-tail-format NVFP4 \
      --select-tail-top-fraction "$PRODUCTION_RENDER_COST_PROMOTE_FRACTION" \
      "${SELECT_TAIL_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_tail.log"
  else
    echo "[pipeline] [2c/4] staged promotion tail exists, skipping"
  fi
  PRODUCTION_RENDER_COST_HIGH_FORMATS="$(python3 - "$FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon not in {"BF16", "NVFP4"} and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
  if [[ -n "$PRODUCTION_RENDER_COST_HIGH_FORMATS" && ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2d/4] rendering staged promotion formats (${PRODUCTION_RENDER_COST_HIGH_FORMATS}) for high-error tail ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$PRODUCTION_RENDER_COST_HIGH_FORMATS" \
      --render-scope format-menu \
      --include-qnames-file "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_tail_cache.log"
  elif [[ -z "$PRODUCTION_RENDER_COST_HIGH_FORMATS" ]]; then
    PRODUCTION_RENDER_COST_CACHE_PATH="$PRODUCTION_RENDER_COST_NVFP4_CACHE"
    echo "[pipeline] [2d/4] no staged high formats requested"
  else
    echo "[pipeline] [2d/4] staged promotion-format cache exists, skipping"
  fi
  if [[ ! -f "$COST_PATH" ]]; then
    echo "[pipeline] [2e/4] synthesizing staged production-render allocator cost ..."
    PROD_RENDER_COST_ARGS=()
    case "$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
      *)
        PROD_RENDER_COST_ARGS+=(--require-output-metric)
        ;;
    esac
    python3 -m prismaquant.production_render_cost \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --baseline-cost "$BASE_COST_PATH" \
      --output "$COST_PATH" \
      --formats "$FORMATS" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      --missing-render-score-policy unavailable \
      --promotion-qnames-file "$PRODUCTION_RENDER_COST_TAIL_QNAMES" \
      --bf16-policy promotion-set \
      "${PROD_RENDER_COST_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_cost.log"
  else
    echo "[pipeline] [2e/4] staged production-render allocator cost exists, skipping"
  fi
fi

# -----------------------------------------------------------------------
# 3. Allocator (multi-choice knapsack over per-layer formats)
# -----------------------------------------------------------------------
echo "[pipeline] [3/4] running allocator at target=${TARGET_BITS} bpp ..."
# Choose visual-sensitivity mode from calibration modality:
#   text-only → uniform (Phase 1 --visual-format path, as before)
#   multimodal → fisher (Phase 2: DP places visual Linears from real
#                        multimodal Fisher; --visual-format acts as a
#                        fallback for un-probed visual Linears only)
if [[ "$CALIBRATION_MODALITY" == "multimodal" ]]; then
  VISUAL_SENSITIVITY=fisher
else
  VISUAL_SENSITIVITY=uniform
fi
ALLOCATOR_PARETO_ARGS=()
if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
  ALLOCATOR_PARETO_DIR="${WORK_DIR}/artifacts/pareto_assignments"
  ALLOCATOR_PARETO_ARGS=(--pareto-output-dir "$ALLOCATOR_PARETO_DIR")
else
  ALLOCATOR_PARETO_DIR=""
fi
ALLOCATOR_PROPAGATED_ARGS=()
if [[ -n "$ALLOC_PROPAGATED_SENSITIVITY_REPORT" ]]; then
  ALLOCATOR_PROPAGATED_ARGS+=(
    --propagated-sensitivity-report "$ALLOC_PROPAGATED_SENSITIVITY_REPORT"
    --propagated-sensitivity-scale "$ALLOC_PROPAGATED_SENSITIVITY_SCALE"
    --propagated-sensitivity-score-field "$ALLOC_PROPAGATED_SENSITIVITY_SCORE_FIELD"
    --propagated-sensitivity-format-extrapolation "$ALLOC_PROPAGATED_SENSITIVITY_FORMAT_EXTRAPOLATION"
  )
  if [[ -n "$ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT" ]]; then
    ALLOCATOR_PROPAGATED_ARGS+=(
      --propagated-sensitivity-target-format "$ALLOC_PROPAGATED_SENSITIVITY_TARGET_FORMAT"
    )
  fi
fi
python3 -m prismaquant.allocator \
  --probe "${PROBE_PATH}" \
  --costs "${COST_PATH}" \
  --target-bits "$TARGET_BITS" \
  --formats "$FORMATS" \
  --target-profile "$TARGET_PROFILE" \
  --pareto-targets "$PARETO_TARGETS" \
  --visual-format "$VISUAL_FORMAT" \
  --visual-sensitivity "$VISUAL_SENSITIVITY" \
  --mtp-format "$MTP_FORMAT" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --pareto-csv "${WORK_DIR}/artifacts/pareto.csv" \
  "${ALLOCATOR_PARETO_ARGS[@]}" \
  "${ALLOCATOR_PROPAGATED_ARGS[@]}" \
  2>&1 | tee "${WORK_DIR}/logs/allocator.log"

# -----------------------------------------------------------------------
# 4. Production cache + native compressed-tensors export
# -----------------------------------------------------------------------
PRODUCTION_CACHE_PATH=""
if [[ "$SELECTION_MODE" == "validated-surrogate" ]] && [[ "$PRODUCTION_CACHE" == "0" || "$PRODUCTION_CACHE" == "false" || "$PRODUCTION_CACHE" == "False" ]]; then
  echo "[pipeline] ERROR: SELECTION_MODE=validated-surrogate requires PRODUCTION_CACHE=1 so KL validates production-rendered weights." >&2
  exit 2
fi
if [[ "$PRODUCTION_CACHE" != "0" && "$PRODUCTION_CACHE" != "false" && "$PRODUCTION_CACHE" != "False" ]]; then
  LEVER_CACHE_TAG=""
  case "$FISHER_WEIGHTED_GPTQ" in
    0|false|False|FALSE|no|No|NO|"") ;;
    *) LEVER_CACHE_TAG="${LEVER_CACHE_TAG}_fisher" ;;
  esac
  PROD_CACHE_DIR="${WORK_DIR}/artifacts/production_weight_cache${LEVER_CACHE_TAG}"
  PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache${LEVER_CACHE_TAG}_raw.pkl"
  PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache${LEVER_CACHE_TAG}_recached.pkl"
  CACHE_FORMATS="$PRODUCTION_CACHE_FORMATS"
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    if [[ -z "$ALLOCATOR_PARETO_DIR" || ! -f "$ALLOCATOR_PARETO_DIR/manifest.json" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection requires allocator pareto assignments at $ALLOCATOR_PARETO_DIR" >&2
      exit 2
    fi
    if [[ "$CACHE_FORMATS" == "auto" ]]; then
      CACHE_FORMATS="$(python3 - "$FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon != "BF16" and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
      echo "[pipeline] frontier production cache formats selected from FORMATS: ${CACHE_FORMATS:-none}"
    fi
    if [[ -z "$CACHE_FORMATS" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection has no non-BF16 cache formats" >&2
      exit 2
    fi
    PROD_CACHE_DIR="${PROD_CACHE_DIR}_frontier"
    PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache${LEVER_CACHE_TAG}_frontier_raw.pkl"
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      if [[ "${PRODUCTION_CACHE_UNION:-0}" == "1" ]]; then
        # Smart-union render: only render FP8_DYNAMIC fallbacks for Linears
        # whose NVFP4 output_mse is above the threshold percentile. Same
        # cache layout as format-menu, just fewer entries.
        echo "[pipeline] [4/4] building smart-union production cache for validated frontier ..."
        python3 -m tools.build_union_cache \
          --model "$MODEL_PATH" \
          --cost-pkl "$COST_PATH" \
          --input-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --output-cache-dir "$PROD_CACHE_DIR" \
          --output-pkl "$PROD_CACHE_RAW" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --levers "$PRODUCTION_CACHE_LEVERS" \
          --p-fp8 "${PRODUCTION_CACHE_UNION_P_FP8:-0.75}" \
          2>&1 | tee "${WORK_DIR}/logs/production_cache_frontier.log"
      else
        echo "[pipeline] [4/4] building format-menu production cache for validated frontier ..."
        python3 -m prismaquant.build_production_cache \
          --model "$MODEL_PATH" \
          --output "$PROD_CACHE_RAW" \
          --formats "$CACHE_FORMATS" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
          --enable "$PRODUCTION_CACHE_LEVERS" \
          --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
          --cache-dir "$PROD_CACHE_DIR" \
          --render-scope format-menu \
          "${PROD_H_DETAIL_ARGS[@]}" \
          2>&1 | tee "${WORK_DIR}/logs/production_cache_frontier.log"
      fi
    else
      echo "[pipeline] [4/4] frontier production cache exists, skipping"
    fi

    VALIDATED_ASSIGNMENT_ARGS=()
    while IFS=$'\t' read -r label path; do
      [[ -n "$label" && -n "$path" ]] || continue
      VALIDATED_ASSIGNMENT_ARGS+=(--assignment "${label}=${path}")
    done < <(python3 - "$ALLOCATOR_PARETO_DIR/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for row in payload.get("candidates", []):
    print(f"{row['label']}\t{row['path']}")
PY
)
    if [[ "${#VALIDATED_ASSIGNMENT_ARGS[@]}" -eq 0 ]]; then
      echo "[pipeline] ERROR: no Pareto assignments found for validated frontier" >&2
      exit 2
    fi

    VALIDATION_JSON="${WORK_DIR}/artifacts/validated_frontier_kl.json"
    VALIDATED_ASSIGNMENT_COUNT=$(( ${#VALIDATED_ASSIGNMENT_ARGS[@]} / 2 ))
    echo "[pipeline] [4/4] measuring real KL for ${VALIDATED_ASSIGNMENT_COUNT} Pareto assignments ..."
    python3 -m prismaquant.validate_assignments_kl \
      --model "$MODEL_PATH" \
      --probe "$PROBE_PATH" \
      --costs "$COST_PATH" \
      --base-assignment "${WORK_DIR}/artifacts/layer_config.json" \
      "${VALIDATED_ASSIGNMENT_ARGS[@]}" \
      --output "$VALIDATION_JSON" \
      --formats "$FORMATS" \
      --dataset "$DATASET" \
      --n-calib-samples "$VALIDATED_FRONTIER_NSAMPLES" \
      --calib-seqlen "$VALIDATED_FRONTIER_SEQLEN" \
      --dtype bf16 \
      --device "$DEVICE" \
      --kl-scope "$VALIDATED_FRONTIER_KL_SCOPE" \
      --kl-cuda-graphs "$VALIDATED_FRONTIER_KL_CUDA_GRAPHS" \
      --assignment-materialization hooks \
      --source-prefetch "$VALIDATED_SOURCE_PREFETCH" \
      --source-prefetch-max-gb "$VALIDATED_SOURCE_PREFETCH_MAX_GB" \
      --source-prefetch-headroom-gb "$VALIDATED_SOURCE_PREFETCH_HEADROOM_GB" \
      --source-prefetch-workers "$VALIDATED_SOURCE_PREFETCH_WORKERS" \
      $(if [[ "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "0" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "false" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "False" ]]; then echo "--disable-frozen-weight-cache"; fi) \
      --production-weight-cache "$PROD_CACHE_RAW" \
      --production-cache-dir-override "$PROD_CACHE_DIR" \
      --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
      --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
      --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
      2>&1 | tee "${WORK_DIR}/logs/validated_frontier_kl.log"

    echo "[pipeline] [4/4] selecting measured frontier point ..."
    python3 -m prismaquant.select_validated_frontier \
      --validation-json "$VALIDATION_JSON" \
      --mode "$VALIDATED_FRONTIER_PICK" \
      --output-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
      --output-assignment "${WORK_DIR}/artifacts/layer_config_validated_assignment.json" \
      --output-summary "${WORK_DIR}/artifacts/validated_frontier_selection.json" \
      2>&1 | tee "${WORK_DIR}/logs/validated_frontier_select.log"

    if [[ "$MSE_PROMOTION" != "0" && "$MSE_PROMOTION" != "false" && "$MSE_PROMOTION" != "False" ]]; then
      MSE_PROMOTION_ARGS=()
      if [[ -n "$MSE_PROMOTION_TARGET_BPP" ]]; then
        MSE_PROMOTION_ARGS+=(--target-bpp "$MSE_PROMOTION_TARGET_BPP")
      elif [[ -n "$MSE_PROMOTION_MAX_BPP_DELTA" ]]; then
        MSE_PROMOTION_ARGS+=(--max-bpp-delta "$MSE_PROMOTION_MAX_BPP_DELTA")
      else
        echo "[pipeline] ERROR: MSE_PROMOTION requires MSE_PROMOTION_TARGET_BPP or MSE_PROMOTION_MAX_BPP_DELTA" >&2
        exit 2
      fi
      echo "[pipeline] applying MSE-driven promotion rewrite ..."
      python3 tools/build_mse_promotion_assignment.py \
        --base-assignment "${WORK_DIR}/artifacts/layer_config.json" \
        --model "$MODEL_PATH" \
        --costs "$COST_PATH" \
        --probe "$PROBE_PATH" \
        --output-layer-config "${WORK_DIR}/artifacts/layer_config_mse_promoted.json" \
        --output-report "${WORK_DIR}/artifacts/mse_promotion_report.json" \
        --categories "$MSE_PROMOTION_CATEGORIES" \
        --target-format "$MSE_PROMOTION_TARGET_FORMAT" \
        --group-by "$MSE_PROMOTION_GROUP_BY" \
        --metric "$MSE_PROMOTION_METRIC" \
        "${MSE_PROMOTION_ARGS[@]}" \
        2>&1 | tee "${WORK_DIR}/logs/mse_promotion.log"
      cp "${WORK_DIR}/artifacts/layer_config.json" \
         "${WORK_DIR}/artifacts/layer_config_before_mse_promotion.json"
      mv "${WORK_DIR}/artifacts/layer_config_mse_promoted.json" \
         "${WORK_DIR}/artifacts/layer_config.json"
    fi

    if [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
      SELECTED_DIGEST="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()[:12])
PY
)"
      PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache${LEVER_CACHE_TAG}_frontier_${SELECTED_DIGEST}_recached.pkl"
      if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
        echo "[pipeline] [4/4] re-fitting activation scales for selected measured-${VALIDATED_FRONTIER_PICK} assignment ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      else
        echo "[pipeline] [4/4] selected-assignment recached production cache exists, skipping"
      fi
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
    else
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
    fi
  elif [[ "$CACHE_FORMATS" == "auto" ]]; then
    CACHE_FORMATS="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import sys
from prismaquant.production_recache import _load_assignment

assignment = _load_assignment(sys.argv[1])
formats = sorted({fmt for fmt in assignment.values() if fmt.upper() != "BF16"})
print(",".join(formats))
PY
)"
    echo "[pipeline] production cache formats selected from assignment: ${CACHE_FORMATS:-none}"
  fi
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    :
  elif [[ -z "$CACHE_FORMATS" ]]; then
    echo "[pipeline] production cache requested but no non-BF16 formats are in FORMATS; skipping cache"
  elif [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
    if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
      if [[ ! -f "$PROD_CACHE_RAW" ]]; then
        echo "[pipeline] [4/4] building production cache + re-fitting activation scales ..."
        python3 -m prismaquant.build_production_cache \
          --model "$MODEL_PATH" \
          --output "$PROD_CACHE_RECACHED" \
          --formats "$CACHE_FORMATS" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
          --enable "$PRODUCTION_CACHE_LEVERS" \
          --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
          --cache-dir "$PROD_CACHE_DIR" \
          --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
          --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          "${PROD_H_DETAIL_ARGS[@]}" \
          2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
      else
        echo "[pipeline] [4/4] re-fitting production activation scales ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      fi
    else
      echo "[pipeline] [4/4] recached production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
  else
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      echo "[pipeline] [4/4] building production cache ..."
      python3 -m prismaquant.build_production_cache \
        --model "$MODEL_PATH" \
        --output "$PROD_CACHE_RAW" \
        --formats "$CACHE_FORMATS" \
        --dataset "$DATASET" \
        --n-calib-samples "$NSAMPLES" \
        --calib-seqlen "$SEQLEN" \
        --dtype bf16 \
        --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
        --enable "$PRODUCTION_CACHE_LEVERS" \
        --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
        --cache-dir "$PROD_CACHE_DIR" \
        --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
        --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
        "${PROD_H_DETAIL_ARGS[@]}" \
        2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
    else
      echo "[pipeline] [4/4] production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
  fi
fi

echo "[pipeline] [4/4] exporting to compressed-tensors ..."
EXPORT_ARGS=(
  python3 -m prismaquant.export_native_compressed
  --model "$MODEL_PATH"
  --layer-config "${WORK_DIR}/artifacts/layer_config.json"
  --output "${WORK_DIR}/exported"
  --device "$EXPORT_DEVICE"
  --activation-cache-dir "${WORK_DIR}/act"
)
case "$EXPORT_GPTQ" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-gptq) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--gptq) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_GPTQ must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
case "$EXPORT_SCALE_SWEEP" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-scale-sweep) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--scale-sweep) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_SCALE_SWEEP must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$PRODUCTION_CACHE_PATH" ]]; then
	  EXPORT_ARGS+=(
	    --production-weight-cache "$PRODUCTION_CACHE_PATH"
	    --production-cache-dir-override "$PROD_CACHE_DIR"
	    --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB"
	    --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS"
	  )
fi
"${EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

echo
echo "[pipeline] done."
echo "  Artifact: ${WORK_DIR}/exported"
echo "  Validate: python3 -m prismaquant.validate_native_export \\"
echo "              --model ${WORK_DIR}/exported"
echo "  Serve:    vllm serve ${WORK_DIR}/exported \\"
echo "              --quantization compressed-tensors --trust-remote-code"
