#!/usr/bin/env bash
# GLM-5.3-Flash stock-vLLM anchored cost lane.
#
#   inventory  Report existing/build/block items and return success. CPU only.
#   preflight  Same checks, but exit 2 on any BLOCK. CPU only.
#
# There is deliberately no `launch` action here. The anchor measurement is a
# batched KL-adjoint harvest (ANCHOR_MEASUREMENT_CONTRACT in
# prismaquant/glm53_stock_reprice.py); its wired launcher is
# tools/run_glm53_stock_harvest.sh (2026-08-27), which runs the GPU harvest
# on sparklina and then the CPU `campaign` pricing action. This script stays
# the CPU-only census/inventory half.
#
# The probe for this campaign lives on sparklina. Regenerate its census with:
#
#   ssh sparklina 'PYTHONPATH=/home/rob/prismaquant CUDA_VISIBLE_DEVICES="" \
#     /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
#     -m prismaquant.glm53_stock_reprice ...' # or the inline extractor
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/glm53-flash}"
CAMPAIGN="${CAMPAIGN:-${RUN_ROOT}/stock_anchored}"
MODEL="${MODEL:-/mnt/shared/models/GLM-5.3-Flash}"

ACTION="${1:-}"
shift || true

usage() {
  cat <<'USAGE'
usage: run_glm53_stock_reprice.sh {inventory|preflight|checkpoint-census} [args...]

  inventory          print every admission and blocker; always exits 0
  preflight          same checks; exits 2 if anything BLOCKs
  checkpoint-census  rebuild the source-precision census from the index

Environment: PYTHON_BIN RUN_ROOT CAMPAIGN MODEL
USAGE
}

if [[ ! "$ACTION" =~ ^(inventory|preflight|checkpoint-census)$ ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$CAMPAIGN"

if [[ "$ACTION" == "checkpoint-census" ]]; then
  exec env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    -m prismaquant.glm53_stock_reprice checkpoint-census \
    --model "$MODEL" --output "${CAMPAIGN}/checkpoint_census.json" "$@"
fi

args=(
  --model "$MODEL"
  --probe-census "${CAMPAIGN}/probe_census.json"
  --checkpoint-census "${CAMPAIGN}/checkpoint_census.json"
  --expert-empirical "${RUN_ROOT}/work/artifacts/cost_expert_empirical.pkl"
  --checkpoint-dir "${CAMPAIGN}/ckpt"
  --repo "$REPO_ROOT"
)
if [[ "$ACTION" == "preflight" ]]; then
  args+=(--strict)
fi

exec env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  -m prismaquant.glm53_stock_reprice inventory "${args[@]}" "$@"
