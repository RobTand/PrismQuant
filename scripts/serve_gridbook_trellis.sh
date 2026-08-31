#!/usr/bin/env bash
# ============================================================================
# serve_gridbook_trellis.sh — serve the Gridbook trellis (TCQ) artifact on sm_121.
# Uses the SERVING runtime helper, not the producer one. That distinction is
# load-bearing and was established empirically on 2026-08-31: the producer
# helper (gridbook_runtime.sh) attests a git CHECKOUT, and the vLLM images have
# no git, so it dies with
#   gridbook-runtime: ERROR: git is required to attest a Gridbook checkout
# gridbook_serving_runtime.sh binds the released wheel and its published
# SHA-256 instead -- the stronger attestation anyway. Supply the wheel with
# GRIDBOOK_SERVING_RUNTIME_WHEEL, and note the pin matches the **dist-ci**
# build; dist-final is a later non-reproducible rebuild with a different digest.
#
# The trellis lanes JIT-build trellis_r256.cu, which includes
# ATen/cuda/CUDAContext.h and therefore needs cusparse.h and cusolverDn.h. The
# vLLM images ship neither in /usr/local/cuda/include, but both live under
# nvidia/cu13/include in site-packages. Link ONLY the headers the toolkit dir
# lacks: putting that whole directory on CPATH makes crt/host_runtime.h come
# from the pip package too and nvcc then fails with
#   error: macro "__cudaLaunch" passed 2 arguments, but takes just 1
# The build also needs the `ninja` BINARY on PATH, which the ninja Python
# package installs into the venv's bin/ -- an absolute-path interpreter
# invocation is not enough, and the resulting failure misreports itself as a
# missing CUDA toolchain.
#
# Required trellis env flags are passed through, and --host 0.0.0.0 --port 8000
# with -p 8000:8000 (a loopback bind is unreachable from the LAN host).
# ============================================================================
set -u
# R15 serve fingerprint (docs/ARCHITECTURE.md §7.4): write_serve_manifest.
PQ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PQ_SCRIPT_DIR/lib/serve_manifest.sh" ] && . "$PQ_SCRIPT_DIR/lib/serve_manifest.sh"
PQ_REPO_ROOT="$(cd "$PQ_SCRIPT_DIR/.." && pwd)"
: "${GRIDBOOK_SERVING_RUNTIME_WHEEL:?set to the pinned dist-ci gridbook wheel}"
export GRIDBOOK_SERVING_RUNTIME_WHEEL
. "$PQ_REPO_ROOT/prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
gridbook_serving_runtime_prepare || exit $?
NAME=pq_gridbook_trellis
MODEL="${MODEL:-/dqruns/trellis-artifact}"
MAXLEN="${MAXLEN:-32768}"
[ "${SHORTLEN:-0}" = "1" ] && MAXLEN=16384
UTIL="${UTIL:-0.80}"
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"
LOG=/home/rob/dq-runs/trellis/logs/serve_trellis.log
mkdir -p /home/rob/dq-runs/trellis/logs /home/rob/dq-runs/trellis/extcache

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
  "${GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_PROFILER_DIR=/dqruns/trellis/profiles \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_HOME=/usr/local/cuda \
  -e PRISMAQUANT_CB_EXT_DIR="${PRISMAQUANT_CB_EXT_DIR:-/dqruns/trellis/extcache}" \
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
    NVINC=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
    for f in "$NVINC"/*.h; do
      b=$(basename "$f")
      [ -e "/usr/local/cuda/include/$b" ] || ln -s "$f" "/usr/local/cuda/include/$b"
    done
    export PATH=/usr/local/cuda/bin:$PATH
    python3 -c "import ninja" >/dev/null 2>&1 || pip install --quiet ninja
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
