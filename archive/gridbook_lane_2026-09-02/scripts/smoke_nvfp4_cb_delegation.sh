#!/usr/bin/env bash
# ============================================================================
# smoke_nvfp4_cb_delegation.sh — MIXED-container 0.6B delegation smoke
# ============================================================================
#
# Exports the FULL mixed menu (CB rungs + STOCK NVFP4/FP8_DYNAMIC + BF16) at
# ~3.5 bpp, then a SHORT vLLM load+greedy-generate confirming that stock-rung
# Linears route through vLLM's CompressedTensors delegation and CB Linears
# through the plugin's CB decode. This is the 27B production menu's shape,
# validated on 0.6B first.
#
# Part 2 installs the immutable external Gridbook revision pinned by
# prismaquant/gridbook_runtime/gridbook_runtime_pin.json. Stock compressed-tensors delegation is
# implemented; the serve must exercise both routes and fail loudly if either
# the external package contract or the artifact is incompatible.
#
# GPU-or-bust for Part 1. Part 2 needs the vllm-node container. Track exact
# container IDs; NEVER pattern-kill (house norm).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO/prismaquant/gridbook_runtime/gridbook_runtime.sh"

: "${MODEL_PATH:=/home/rob/models/Qwen3-0.6B}"
: "${WORK_DIR:=/home/rob/dq-runs/smoke-nvfp4-cb-mixed-3p5}"
: "${ARTIFACT:=${WORK_DIR}/exported_nvfp4_cb}"
: "${SERVE_IMAGE:=vllm-node:latest}"
: "${PORT:=8000}"
: "${RUN_SERVE:=1}"   # set 0 to run only the export (Part 1)
if [[ "$RUN_SERVE" == "1" ]]; then
  # Resolve the runtime before the expensive export so a missing/wrong pin
  # cannot waste the producer half of a serve-enabled smoke.
  gridbook_runtime_prepare || exit $?
fi
# The full mixed menu (mirrors the 27B production menu, exp1c GO).
: "${FORMATS:=NVFP4,FP8_DYNAMIC,BF16,NVFP4_CB_K18,NVFP4_CB_K20,NVFP4_CB_K22,NVFP4_CB_K24,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48}"

echo "============================================================================"
echo "MIXED-CONTAINER DELEGATION SMOKE — export @3.5 bpp then vLLM load+generate"
echo "  MODEL=$MODEL_PATH  WORK_DIR=$WORK_DIR  FORMATS=$FORMATS"
echo "============================================================================"

# --- Part 1: export the full mixed menu (stock rungs win some Linears) -------
MODEL_PATH="$MODEL_PATH" WORK_DIR="$WORK_DIR" FORMATS="$FORMATS" \
  TARGET_BITS=3.5 PARETO_TARGETS="3.25,3.5,3.75" \
  CB_SCALE_CODING=v1 CB_CODEBOOK_SOURCE=lattice NSAMPLES=16 SEQLEN=512 \
  PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/scripts/smoke_nvfp4_cb_pipeline.sh"

echo "[deleg-smoke] export done. Composition (confirm STOCK rungs appear):"
python3 - "$WORK_DIR/artifacts/layer_config.json" <<'PY' || true
import json, sys
from collections import Counter
from prismaquant import layer_config as lc
asg = json.load(open(sys.argv[1])).get("assignment", {})
print("  ", dict(Counter(lc.canonicalize_format(v) for v in asg.values())))
PY

if [[ "${RUN_SERVE}" != "1" ]]; then
  echo "[deleg-smoke] RUN_SERVE=0 — skipping Part 2 (vLLM). Export validated."
  exit 0
fi

# --- Part 2: vLLM load + greedy-generate ------------------------------------
echo "[deleg-smoke] launching vLLM ($SERVE_IMAGE) with the plugin ..."
CID=$(docker run -d --gpus all -p "${PORT}:${PORT}" \
  -v "${REPO}:/repo:ro" -v "${ARTIFACT}:/model:ro" \
  "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
  --entrypoint bash "${SERVE_IMAGE}" -c \
  "set -euo pipefail; \
   bash \"\${PQ_GRIDBOOK_RUNTIME_HELPER:?}\" install-container && \
   vllm serve /model --host 0.0.0.0 --port ${PORT} --trust-remote-code \
     --quantization gridbook --enforce-eager")
echo "[deleg-smoke] container CID=${CID} — stop by this EXACT id, never pattern-kill"
trap 'docker stop "${CID}" >/dev/null 2>&1 || true; docker rm "${CID}" >/dev/null 2>&1 || true' EXIT

# Poll /health; fail fast and dump logs if the container exits early.
for _ in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { READY=1; break; }
  if [[ -z "$(docker ps -q --no-trunc --filter id=${CID})" ]]; then
    echo "[deleg-smoke] server exited before ready. Last logs:"
    docker logs --tail 60 "${CID}" 2>&1 | tail -60
    exit 1
  fi
  sleep 5
done

RESP=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/model","prompt":"The capital of France is","max_tokens":16,"temperature":0}')
echo "[deleg-smoke] greedy completion: ${RESP}"

echo "[deleg-smoke] routing evidence (stock via CT, CB via plugin):"
docker logs "${CID}" 2>&1 | grep -iE \
  "CompressedTensors|PrismaQuantCB|nvfp4-pack|float-quantized|delegat" | head -20

echo "############################################################################"
echo "##  REVIEW: coherent generation above + both routes exercised.            ##"
echo "############################################################################"
