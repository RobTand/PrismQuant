#!/usr/bin/env bash
# ============================================================================
# canary_ladder.sh — staged GPU stability isolation after the 2026-07-23
# wedges (4 force reboots; 595.84 driver regression rolled back, one wedge
# on the restored 1026+595.71.05 pair 4 min after a bench container exited).
#
# Four escalating steps, each followed by a SOAK_MIN-minute soak with a
# kernel-journal watch. Stops at the first anomaly. A step that wedges the
# box identifies the trigger at that step's workload class.
#
#   1. plain torch CUDA matmul (no gridbook code)
#   2. proven gridbook kernels (decode GEMV + expander battery)
#   3. serve boot + one 8k prefill  (doubles as the W2-overlap measurement)
#   4. hardened §4b battery (PRISMAQUANT_ENABLE_PTC=1; v1+v2 parity+bench)
#
# Usage: bash scripts/canary_ladder.sh [start_step] [SOAK_MIN]
# ============================================================================
set -euo pipefail
STEP="${1:-1}"
SOAK_MIN="${2:-10}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO/prismaquant/gridbook_runtime/gridbook_runtime.sh"
gridbook_runtime_prepare || exit $?
LOG=/home/rob/dq-runs/laguna-s21/logs/canary_ladder.log
mkdir -p "$(dirname "$LOG")"

say() { echo "[ladder $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

soak() {
  say "soaking ${SOAK_MIN}m (journal watch)..."
  local end=$(( $(date +%s) + SOAK_MIN * 60 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    if journalctl -k --since "15 minutes ago" --no-pager 2>/dev/null \
        | grep -qE "hung_task|blocked for more|Xid|NVRM: Xid"; then
      say "ANOMALY IN JOURNAL — stopping ladder"; exit 1
    fi
    sleep 30
  done
  say "soak clean"
}

if [ "$STEP" -le 1 ]; then
  say "step 1: plain torch matmul"
  docker run --rm --gpus all --ipc=host --entrypoint python3 vllm-node:latest \
    -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); b=a@a; torch.cuda.synchronize(); print('matmul ok', float(b.sum()))" \
    2>&1 | tail -1 | tee -a "$LOG"
  soak
fi

if [ "$STEP" -le 2 ]; then
  say "step 2: proven gridbook kernels"
  docker run --rm --gpus all --ipc=host -v "$REPO":/repo:ro \
    "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" --entrypoint bash \
    vllm-node:latest -c '
      set -euo pipefail
      . "${PQ_GRIDBOOK_RUNTIME_HELPER:?}"
      gridbook_runtime_install_container
      pip install pytest -q 2>/dev/null
      cd "$(gridbook_runtime_container_install_target)"
      python3 -m pytest tests/test_cuda_gemv.py -x -q 2>&1 | tail -1' \
    2>&1 | tail -1 | tee -a "$LOG"
  soak
fi

if [ "$STEP" -le 3 ]; then
  say "step 3: serve boot + prefill (W2-overlap measurement)"
  UTIL=0.85 SHORTLEN=1 EXTRA_ARGS="--enforce-eager --max-num-batched-tokens 16384" \
    bash "$REPO/scripts/serve_laguna_smoke.sh" 2>&1 | tail -2 | tee -a "$LOG"
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, time, urllib.request
def req(p, mt=1, extra=None):
    body = {"model": "laguna", "prompt": p, "max_tokens": mt, "temperature": 0}
    r = urllib.request.Request('http://localhost:8000/v1/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as f:
        out = json.load(f)
    return time.time() - t0, out
req("Hello", 8)
dt, out = req("Ladder overlap measurement corpus. " * 1100)
print(f"prefill 8k (overlap ON): {out['usage']['prompt_tokens']/dt:.0f} tok/s")
dt2, out2 = req("Q: 17*23? Steps.\nA:", mt=48)
print("coherent:", "391" in out2['choices'][0]['text'])
PYEOF
  docker rm -f pq_laguna >/dev/null 2>&1
  say "serve stopped for overlap-OFF arm"
  PRISMAQUANT_CB_PREFILL_OVERLAP=0 UTIL=0.85 SHORTLEN=1 \
    EXTRA_ARGS="--enforce-eager --max-num-batched-tokens 16384" \
    bash "$REPO/scripts/serve_laguna_smoke.sh" 2>&1 | tail -1 | tee -a "$LOG"
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, time, urllib.request
body = {"model":"laguna","prompt":"Ladder overlap-off corpus, distinct text. " * 1000,
        "max_tokens":1,"temperature":0}
r = urllib.request.Request('http://localhost:8000/v1/completions',
    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
import time as T; t0=T.time()
import urllib.request as U
with U.urlopen(r, timeout=900) as f: out=json.load(f)
dt=T.time()-t0
print(f"prefill 8k (overlap OFF): {out['usage']['prompt_tokens']/dt:.0f} tok/s")
PYEOF
  docker rm -f pq_laguna >/dev/null 2>&1
  soak
fi

if [ "$STEP" -le 4 ]; then
  say "step 4: hardened persistent-TC battery (quarantine opt-in)"
  docker run --rm --gpus all --ipc=host -v "$REPO":/repo:ro \
    "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" \
    -e PRISMAQUANT_ENABLE_PTC=1 --entrypoint bash vllm-node:latest -c '
      set -euo pipefail
      . "${PQ_GRIDBOOK_RUNTIME_HELPER:?}"
      gridbook_runtime_install_container
      pip install pytest -q 2>/dev/null
      cd "$(gridbook_runtime_container_install_target)"
      python3 -m pytest tests/test_persistent_tc.py -q 2>&1 | tail -1
      python3 tests/test_persistent_tc.py bench 2>&1 | tail -12' \
    2>&1 | tail -13 | tee -a "$LOG"
  soak
fi

say "LADDER COMPLETE — all steps clean"
