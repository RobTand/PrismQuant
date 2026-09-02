#!/usr/bin/env bash
# ============================================================================
# measure_35b_ab.sh — 35B MoE gold-metric A/B: OURS (nvfp4_cb, CB experts) vs
# BF16 (and PrismaAURA-4.75 if present), served via the vllm-node container.
# First served test of the CB MoE-EXPERT path (PrismaQuantCBMoEMethod) — expect
# possible new serving-load bugs like the 27B dense run surfaced.
#
# 35B = Qwen3.6-35B-A3B-class hybrid MoE + Gated-DeltaNet. DeltaNet needs
# --max-num-batched-tokens >= 2096 (memory aura_moe_proven_35b_arme); spec-decode
# OFF for the PPL gate. Serve the exported dir via the /dqruns mount (no copies).
# Held-out wiki.test, 8192 tok x 512: conf-KL/all-KL/top1/PPL vs the SAME BF16
# session + 3x TTFT(1400)/decode. Exact container names; never pattern-kill.
# ============================================================================
set -u
SERVE=/home/rob/dq-runs/nvfp4-cb-phase0/serve/serve_one.sh   # mounts /dqruns
MEASURE=/home/rob/dq-runs/nvfp4-cb-phase0/serve/measure.py
WORK=/home/rob/dq-runs/prod-35b-nvfp4cb-4p75
OUT="$WORK/ab"; mkdir -p "$OUT"
TOK=/home/rob/dq-runs/ornith-35b-base                        # host tokenizer path
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python

BF16_CT=/dqruns/ornith-35b-base
OURS_CT=/dqruns/prod-35b-nvfp4cb-4p75/exported_nvfp4_cb
AURA_DIR=/home/rob/dq-runs/ornith-35b-aura-dl                # optional baseline
AURA_CT=/dqruns/ornith-35b-aura-dl

DELTANET="--max-num-batched-tokens 2096"

serve_dump_speed () {
  local name="$1" model_ct="$2" label="$3"; shift 3; local extra="$*"
  echo "=== [$label] serving $model_ct (extra: $extra) ==="
  if [ "$(bash "$SERVE" "$name" "$model_ct" $extra)" != "READY" ]; then
    echo "[$label] SERVE FAILED"; docker logs "$name" 2>&1 | tail -30; return 1; fi
  local smoke; smoke=$(curl -s http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
    | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null)
  echo "[$label] smoke: '$smoke'"
  "$PY" "$MEASURE" dump m "$TOK" "$OUT/$label.json"
  "$PY" "$MEASURE" speed m "$TOK" | tee "$OUT/${label}_speed.txt"
  docker stop "$name" >/dev/null 2>&1
  echo "[$label] done, container stopped."
}

serve_dump_speed pq35b_bf16 "$BF16_CT" bf16 $DELTANET --gpu-memory-utilization 0.90
serve_dump_speed pq35b_ours "$OURS_CT" ours $DELTANET --gpu-memory-utilization 0.85
if [ -d "$AURA_DIR" ]; then
  serve_dump_speed pq35b_aura "$AURA_CT" aura $DELTANET --gpu-memory-utilization 0.85
fi

echo "############################################################################"
echo "## 35B A/B RESULTS (vs the SAME bf16 session)"
echo "== OURS vs BF16 =="; "$PY" "$MEASURE" compare "$OUT/bf16.json" "$OUT/ours.json"
echo "  PPL(ours):"; "$PY" "$MEASURE" ppl "$OUT/ours.json"
echo "  PPL(bf16):"; "$PY" "$MEASURE" ppl "$OUT/bf16.json"
if [ -f "$OUT/aura.json" ]; then
  echo "== PrismaAURA-4.75 vs BF16 =="; "$PY" "$MEASURE" compare "$OUT/bf16.json" "$OUT/aura.json"
  echo "  PPL(aura):"; "$PY" "$MEASURE" ppl "$OUT/aura.json"
fi
echo "## speed: $OUT/{bf16,ours,aura}_speed.txt"
echo "############################################################################"
