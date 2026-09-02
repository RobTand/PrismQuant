#!/usr/bin/env bash
# ============================================================================
# serve_hy3_smoke.sh — first serve of the Hy3 295B NVFP4-CB artifact (110 GB)
# on ONE DGX Spark. Load + coherent-gen smoke ONLY (NO QUALITY CLAIMS: a 295B
# cannot be KL-validated vs its BF16 teacher on this box). Then a TTFT/decode
# read for the native-vs-GGUF-IQ-prefill comparison (the thesis number).
#
# Memory: 110.3 GB weights on the ~121 GB unified pool -> util 0.95, context
# CAPPED (native 262k will NOT fit; see prod_hy3_results.md footprint). Hy3 is
# standard GQA (kv-heads 8), NOT DeltaNet — no --max-num-batched-tokens floor.
# HYV3ForCausalLM is native in vllm-node (verified), no trust-remote-code.
# Exact container name; never pattern-kill.
# ============================================================================
set -u
# R15 serve fingerprint (docs/ARCHITECTURE.md §7.4): write_serve_manifest.
PQ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PQ_SCRIPT_DIR/lib/serve_manifest.sh" ] && . "$PQ_SCRIPT_DIR/lib/serve_manifest.sh"
PQ_REPO_ROOT="$(cd "$PQ_SCRIPT_DIR/.." && pwd)"
. "$PQ_REPO_ROOT/prismaquant/gridbook_runtime/gridbook_runtime.sh"
gridbook_runtime_prepare || exit $?
NAME=pq_hy3_cb
MODEL=/dqruns/prod-hy3-nvfp4cb-2p9/exported_nvfp4_cb
MAXLEN="${MAXLEN:-8192}"
UTIL="${UTIL:-0.95}"
LOG=/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/logs/serve_smoke.log

docker rm -f "$NAME" >/dev/null 2>&1
echo "[serve] launching $NAME (max-model-len $MAXLEN, util $UTIL) $(date '+%H:%M:%S')"
docker run -d --rm --gpus all -p 8000:8000 --name "$NAME" \
  -v "$PQ_REPO_ROOT":/repo:ro \
  -v /home/rob/dq-runs:/dqruns \
  -v /home/rob/.cache/huggingface:/hf \
  "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PQ_MODEL="$MODEL" -e PQ_MAXLEN="$MAXLEN" -e PQ_UTIL="$UTIL" \
  --entrypoint bash vllm-node:latest -c '
    set -euo pipefail
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    exec vllm serve "$PQ_MODEL" --host 0.0.0.0 --port 8000 --enforce-eager \
      --max-model-len "$PQ_MAXLEN" --served-model-name m \
      --gpu-memory-utilization "$PQ_UTIL"' > "$LOG" 2>&1

# Long load window: 110 GB weight read + plugin JIT build + KV profile.
for i in $(seq 1 240); do   # up to 20 min
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[serve] READY after $((i*5))s $(date '+%H:%M:%S')"
    write_serve_manifest "$NAME" "$MODEL"
    exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] FAILED (container exited)"; docker logs "$NAME" 2>&1 | tail -40; exit 1; fi
  sleep 5
done
echo "[serve] TIMEOUT after 20min"; docker logs "$NAME" 2>&1 | tail -40; exit 1
