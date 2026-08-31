#!/usr/bin/env bash
# ============================================================================
# serve_gridbook_trellis.sh — serve the Gridbook trellis (TCQ) artifact on sm_121.
# Modelled on scripts/serve_qwen27b_smoke.sh (gridbook_runtime_prepare,
# GRIDBOOK_RUNTIME_DOCKER_ARGS, in-container install-container, then vllm serve).
# Plus the required trellis env flags passed through, and --host 0.0.0.0 --port 8000
# with -p 8000:8000 (loopback bind is unreachable from LAN host).
# ============================================================================
set -u
# R15 serve fingerprint (docs/ARCHITECTURE.md §7.4): write_serve_manifest.
PQ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PQ_SCRIPT_DIR/lib/serve_manifest.sh" ] && . "$PQ_SCRIPT_DIR/lib/serve_manifest.sh"
PQ_REPO_ROOT="$(cd "$PQ_SCRIPT_DIR/.." && pwd)"
. "$PQ_REPO_ROOT/prismaquant/gridbook_runtime/gridbook_runtime.sh"
gridbook_runtime_prepare || exit $?
NAME=pq_gridbook_trellis
MODEL="${MODEL:-/dqruns/trellis-artifact}"
MAXLEN="${MAXLEN:-32768}"
[ "${SHORTLEN:-0}" = "1" ] && MAXLEN=16384
UTIL="${UTIL:-0.80}"
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"
LOG=/home/rob/dq-runs/trellis/logs/serve_trellis.log
mkdir -p /home/rob/dq-runs/trellis/logs

docker rm -f "$NAME" >/dev/null 2>&1
HOST_MODEL="${MODEL/\/dqruns/\/home\/rob\/dq-runs}"
python3 - "$HOST_MODEL" <<'PYFLUSH' 2>/dev/null || true
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PYFLUSH
echo "[serve] launching $NAME (trellis len $MAXLEN, util $UTIL, extra: $EXTRA_ARGS) $(date '+%H:%M:%S')"
docker run -d --gpus all --ipc=host -p 8000:8000 --name "$NAME" \
  -v "$PQ_REPO_ROOT":/repo:ro \
  -v /home/rob/dq-runs:/dqruns \
  "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_PROFILER_DIR=/dqruns/trellis/profiles \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PQ_MODEL="$MODEL" -e PQ_MAXLEN="$MAXLEN" -e PQ_UTIL="$UTIL" \
  -e PQ_EXTRA="$EXTRA_ARGS" \
  -e PQ_SPEC="${SPEC_CONFIG:-}" \
  -e GRIDBOOK_TRELLIS_E2M1="${GRIDBOOK_TRELLIS_E2M1:-}" \
  -e GRIDBOOK_TRELLIS_E2M1_MODE="${GRIDBOOK_TRELLIS_E2M1_MODE:-}" \
  -e GRIDBOOK_TRELLIS_E4M3="${GRIDBOOK_TRELLIS_E4M3:-}" \
  -e GRIDBOOK_TRELLIS_E4M3_MODE="${GRIDBOOK_TRELLIS_E4M3_MODE:-}" \
  --entrypoint bash vllm-node:latest -c '
    set -euo pipefail
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    exec vllm serve "$PQ_MODEL" --host 0.0.0.0 --port 8000 \
      --served-model-name trellis \
      --max-model-len "$PQ_MAXLEN" --max-num-seqs 2 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization "$PQ_UTIL" \
      ${PQ_SPEC:+--speculative-config "$PQ_SPEC"} \
      $PQ_EXTRA' > "$LOG" 2>&1

for i in $(seq 1 240); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[serve] READY after $((i*5))s $(date '+%H:%M:%S')"
    MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
    WATCHDOG_GIB="${WATCHDOG_GIB:-4}"
    sleep 10
    avail_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
      echo "[serve] SLACK GATE FAILED: MemAvailable ${avail_gib} GiB < ${MIN_FREE_GIB} GiB at UTIL=$UTIL / len $MAXLEN."
      docker rm -f "$NAME" >/dev/null 2>&1
      exit 3
    fi
    echo "[serve] slack gate OK: MemAvailable ${avail_gib} GiB (floor ${MIN_FREE_GIB})"
    nohup bash -c '
      while docker ps --format "{{.Names}}" | grep -q "^'"$NAME"'$"; do
        a=$(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)
        if [ "$a" -lt '"$WATCHDOG_GIB"' ]; then
          echo "$(date "+%F %T") WATCHDOG: MemAvailable ${a} GiB — stopping '"$NAME"'" >> '"$LOG"'
          docker rm -f '"$NAME"' >/dev/null 2>&1
          exit 0
        fi
        sleep 20
      done' >/dev/null 2>&1 &
    disown
    echo "[serve] memory watchdog armed (stop below ${WATCHDOG_GIB} GiB free)"
    write_serve_manifest "$NAME" "$MODEL"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] FAILED (container exited)"; docker logs "$NAME" 2>&1 | tail -40; exit 1; fi
  sleep 5
done
echo "[serve] TIMEOUT after 20min"; docker logs "$NAME" 2>&1 | tail -40; exit 1
