#!/usr/bin/env bash
# ============================================================================
# serve_qwen38_cb_a_smoke.sh — serve the Qwen3.8-27B AQUA-AURA CB artifact "A"
# (the ~13 GB consumer-card build).
#
# WHY THIS IS NOT serve_qwen27b_smoke.sh
# --------------------------------------
# That script sources gridbook_runtime.sh, which is the *build* pin: contract
# v4, Gridbook 0.8.11 since 2026-08-21 (0.8.5/v3 when this note was written),
# installed into vllm-node:latest from a source checkout.
# Artifact A's recipe assigns `model.embed_tokens` to NVFP4, and the quantized
# embedding mechanism does not exist before Gridbook 0.8.7 -- on 0.8.5 the load
# dies on an unrecognised `embed_tokens.cb_*`/NVFP4 embedding parameter. So
# this lane must cross the *serving* pin (gridbook_serving_runtime.sh, contract
# v4, 0.8.11) instead.  0.8.8+ is required, not merely current: 0.8.7 shipped
# the quantized-embedding METHOD but Qwen3.5/3.6 never calls get_quant_method
# for its lookup table, so on 0.8.7 this artifact does not load at all; 0.8.11
# is the pinned release (0.8.9 defaults the qualified CB kernels on, 0.8.10
# fixes a split-bank MoE load regression, 0.8.11 makes the routed grouped
# lanes and the MXFP8 A-side capture-safe -- dense-only artifacts like this one
# see no route change from any of them, and every explicit spelling keeps its
# 0.8.8 semantics). The two pins are held in lockstep by
# tests/test_gridbook_runtime_boundary.py since 2026-08-21, but they remain
# distinct boundaries (only the serving pin binds a reviewed wheel digest);
# picking the wrong one is a load-time failure, not a warning.
#
# The base image already carries the pinned wheel, so `install-container` is a
# reinstall-and-verify no-op here rather than a first install -- kept anyway
# because it is the gate that proves the running interpreter imports the exact
# reviewed archive (PEP 610 digest + runtime contract + ABI closure), which is
# the whole point of the serving pin.
#
# First-boot smoke: NO drafter (the MTP heads are stripped from this artifact),
# fp8 KV, eager. UTIL starts at 0.80 with the slack gate + watchdog per the
# 2026-07-21 OOM discipline. GRAPH=1 re-runs with capture enabled, which is the
# second half of the principle-9 eager+graph load gate.
# ============================================================================
set -u
# R15 serve fingerprint (docs/ARCHITECTURE.md §7.4): write_serve_manifest.
PQ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PQ_SCRIPT_DIR/lib/serve_manifest.sh" ] && . "$PQ_SCRIPT_DIR/lib/serve_manifest.sh"
PQ_REPO_ROOT="$(cd "$PQ_SCRIPT_DIR/.." && pwd)"
. "$PQ_REPO_ROOT/prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
gridbook_serving_runtime_prepare || exit $?

# The image built from the published gridbook archive the serving pin names;
# its installed distribution is the archive whose sha256 the pin asserts.
BASE_IMAGE="${BASE_IMAGE:-gridbook:0.8.11-clean-187c721}"
NAME=pq_qwen38_cb_a
WORK=/home/rob/dq-runs/qwen38-27b-cb-a
MODEL="${MODEL:-/dqruns/qwen38-27b-cb-a/exported_nvfp4_cb}"
MAXLEN="${MAXLEN:-16384}"
UTIL="${UTIL:-0.80}"
if [ "${GRAPH:-0}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS:-}"
else
  EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"
fi
LOG="$WORK/logs/serve_smoke${GRAPH:+_graph}.log"
mkdir -p "$WORK/logs"

docker rm -f "$NAME" >/dev/null 2>&1
HOST_MODEL="${MODEL/\/dqruns/\/home\/rob\/dq-runs}"
if [ ! -f "$HOST_MODEL/config.json" ]; then
  echo "[serve] no artifact at $HOST_MODEL (config.json missing)"; exit 2
fi
# Drop the page cache for the artifact: on unified memory a warm page cache is
# not free headroom, it is the KV pool's competitor.
python3 - "$HOST_MODEL" <<'PYFLUSH' 2>/dev/null || true
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PYFLUSH

echo "[serve] launching $NAME on $BASE_IMAGE (len $MAXLEN, util $UTIL, extra: ${EXTRA_ARGS:-<graph>}) $(date '+%H:%M:%S')"
docker run -d --gpus all --ipc=host -p 8000:8000 --name "$NAME" \
  -v "$PQ_REPO_ROOT":/repo:ro \
  -v /home/rob/dq-runs:/dqruns \
  "${GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PQ_MODEL="$MODEL" -e PQ_MAXLEN="$MAXLEN" -e PQ_UTIL="$UTIL" \
  -e PQ_EXTRA="$EXTRA_ARGS" \
  -e PRISMAQUANT_CB_DISPATCH="${PRISMAQUANT_CB_DISPATCH:-}" \
  -e PRISMAQUANT_CB_DECODE_CONTRACT="${PRISMAQUANT_CB_DECODE_CONTRACT:-}" \
  -e PRISMAQUANT_CB_PREFILL="${PRISMAQUANT_CB_PREFILL:-}" \
  -e PRISMAQUANT_CB_FUSED_MIDM="${PRISMAQUANT_CB_FUSED_MIDM:-}" \
  --entrypoint bash "$BASE_IMAGE" -c '
    set -euo pipefail
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    exec vllm serve "$PQ_MODEL" --host 0.0.0.0 --port 8000 \
      --served-model-name qwen \
      --max-model-len "$PQ_MAXLEN" --max-num-seqs 2 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization "$PQ_UTIL" \
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
