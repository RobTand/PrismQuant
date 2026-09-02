#!/usr/bin/env bash
# ============================================================================
# serve_hy3_teb.sh — serve the Hy3 295B NVFP4-CB artifact under the EXACT
# ToolEvalBench protocol the shipped GGUF artifacts used (July-08/13 runs,
# hy3/bench/serving/serve.sh): 12288 ctx, kv fp8, max-num-seqs 2, hy_v3
# tool+reasoning parsers, eager. Only the artifact + plugin differ, so the
# TEB score is protocol-comparable to GGUF IQ 87 / k-quant 86.
#
# VLLM_TORCH_PROFILER_DIR enables POST /start_profile → /stop_profile for the
# decode time-budget trace (no overhead unless started).
# EXTRA_ARGS lets perf experiments (cudagraph configs) reuse this script.
# ============================================================================
set -u
# R15 serve fingerprint (docs/ARCHITECTURE.md §7.4): write_serve_manifest.
PQ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PQ_SCRIPT_DIR/lib/serve_manifest.sh" ] && . "$PQ_SCRIPT_DIR/lib/serve_manifest.sh"
PQ_REPO_ROOT="$(cd "$PQ_SCRIPT_DIR/.." && pwd)"
. "$PQ_REPO_ROOT/prismaquant/gridbook_runtime/gridbook_runtime.sh"
gridbook_runtime_prepare || exit $?
NAME=pq_hy3_cb
MODEL="${MODEL:-/dqruns/prod-hy3-nvfp4cb-2p9/exported_nvfp4_cb}"
MAXLEN="${MAXLEN:-12288}"
UTIL="${UTIL:-0.90}"
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"
LOG=/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/logs/serve_teb.log
PROF=/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9/profiles
mkdir -p "$PROF"

docker rm -f "$NAME" >/dev/null 2>&1
# Drop the checkpoint's page cache from any PREVIOUS boot (unprivileged:
# posix_fadvise DONTNEED). Stale file cache from a dead serve shrinks the
# free unified pool vLLM's memory profiler sees at the next boot — measured
# 12.7 -> 1.07 GiB KV across three relaunches of the same model (2026-07-21).
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
echo "[serve] launching $NAME (TEB protocol: len $MAXLEN, util $UTIL, extra: $EXTRA_ARGS) $(date '+%H:%M:%S')"
# EXTRA_ARGS reaches the container via env + a SINGLE-quoted -c script: the
# container shell expands it with word-splitting but WITHOUT quote removal, so
# embedded JSON (--compilation-config '{"…"}') survives intact. Host-side
# interpolation into a double-quoted -c string strips the JSON's quotes.
docker run -d --gpus all --ipc=host -p 8000:8000 --name "$NAME" \
  -v "$PQ_REPO_ROOT":/repo:ro \
  -v /home/rob/dq-runs:/dqruns \
  "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_TORCH_PROFILER_DIR=/dqruns/prod-hy3-nvfp4cb-2p9/profiles \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PQ_MODEL="$MODEL" -e PQ_MAXLEN="$MAXLEN" -e PQ_UTIL="$UTIL" \
  -e PQ_EXTRA="$EXTRA_ARGS" \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}" \
  -e PRISMAQUANT_CB_W2_SCHED="${PRISMAQUANT_CB_W2_SCHED:-}" \
  -e PRISMAQUANT_CB_PREFILL="${PRISMAQUANT_CB_PREFILL:-}" \
  -e PRISMAQUANT_CB_DISPATCH="${PRISMAQUANT_CB_DISPATCH:-}" \
  --entrypoint bash vllm-node:latest -c '
    set -euo pipefail
    # The helper re-attests and snapshots the externally pinned source before
    # installation, so host edits cannot race a JIT build.
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    exec vllm serve "$PQ_MODEL" --host 0.0.0.0 --port 8000 \
      --served-model-name hy3 \
      --max-model-len "$PQ_MAXLEN" --max-num-seqs 2 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization "$PQ_UTIL" \
      --enable-auto-tool-choice --tool-call-parser hy_v3 \
      --reasoning-parser hy_v3 \
      $PQ_EXTRA' > "$LOG" 2>&1

for i in $(seq 1 240); do   # up to 20 min (110 GB load + plugin JIT)
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[serve] READY after $((i*5))s $(date '+%H:%M:%S')"
    # --- OOM guards (2026-07-21 box-kill postmortem) ---------------------
    # The spec+compile serve at util 0.90 left the unified pool with no
    # evictable slack; ~1.75h of IDLE later, a trivial kernel allocation
    # tripped a global OOM cascade (EngineCore 143.5 GiB rss incl. mmap'd
    # weight pages + CUDA pool) and the box needed a power cycle. Guard 1:
    # fail-fast — if true slack is under MIN_FREE_GIB right after READY,
    # this config does not fit; stop the serve and say so (a dead serve
    # beats a dead box). Guard 2: a detached watchdog stops the container
    # if MemAvailable ever dips below WATCHDOG_GIB while it runs.
    MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
    WATCHDOG_GIB="${WATCHDOG_GIB:-4}"
    sleep 10   # let allocator settle post-READY before judging slack
    avail_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
      echo "[serve] SLACK GATE FAILED: MemAvailable ${avail_gib} GiB < ${MIN_FREE_GIB} GiB — this config does not fit the box at UTIL=$UTIL. Stopping the serve (lower UTIL or trim the config)."
      docker rm -f "$NAME" >/dev/null 2>&1
      exit 3
    fi
    echo "[serve] slack gate OK: MemAvailable ${avail_gib} GiB (floor ${MIN_FREE_GIB})"
    nohup bash -c '
      while docker ps --format "{{.Names}}" | grep -q "^'"$NAME"'$"; do
        a=$(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)
        if [ "$a" -lt '"$WATCHDOG_GIB"' ]; then
          echo "$(date "+%F %T") WATCHDOG: MemAvailable ${a} GiB < '"$WATCHDOG_GIB"' — stopping '"$NAME"' to save the box" >> '"$LOG"'
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
