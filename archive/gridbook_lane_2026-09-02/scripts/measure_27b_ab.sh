#!/usr/bin/env bash
# ============================================================================
# measure_27b_ab.sh — 27B gold-metric A/B: OURS (nvfp4_cb) vs PrismaAURA-5.5 vs
# BF16, one serving session each (vllm-node container), reusing the serving
# agent's serve_one.sh + measure.py + kl_tool conventions (do NOT reinvent).
#
# Held-out wiki.test.raw, 8192 tok x seqlen 512: conf-KL + all-KL + top1 + PPL
# for both quant artifacts vs the SAME BF16 session; + 3x TTFT(1400)+decode for
# all three (no-prefill-degradation proof at 27B). Exact container IDs; never
# pattern-kill.
# ============================================================================
set -u
SERVE=/home/rob/dq-runs/nvfp4-cb-phase0/serve/serve_one.sh
MEASURE=/home/rob/dq-runs/nvfp4-cb-phase0/serve/measure.py
HFCACHE=/home/rob/.cache/huggingface
WORK=/home/rob/dq-runs/prod-27b-nvfp4cb-5p5
OUT="$WORK/ab"; mkdir -p "$OUT"
TOK="$HFCACHE/qwen36-27b-bf16"                 # host tokenizer path for measure.py
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python

# --- artifact placement: serve the EXISTING dirs via the /dqruns mount (no
#     23GB x2 copies — that breaches the 10% disk floor; serve_one.sh now mounts
#     /home/rob/dq-runs:/dqruns). The export dir already holds cb_codebooks.pqcb. ---
BF16_CT=/hf/qwen36-27b-bf16
OURS_CT=/dqruns/prod-27b-nvfp4cb-5p5/exported_nvfp4_cb
AURA_CT=/dqruns/prod-27b-aura-dl

# serve -> coherent smoke -> dump top20 -> speed -> stop (exact container name).
serve_dump_speed () {
  local name="$1" model_ct="$2" label="$3"; shift 3; local extra="$*"
  echo "=== [$label] serving $model_ct (extra: $extra) ==="
  if [ "$(bash "$SERVE" "$name" "$model_ct" $extra)" != "READY" ]; then
    echo "[$label] SERVE FAILED"; docker logs "$name" 2>&1 | tail -20; return 1; fi
  # COHERENT-GENERATION SMOKE before any measurement.
  local smoke; smoke=$(curl -s http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
    | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null)
  echo "[$label] smoke: '$smoke'"
  "$PY" "$MEASURE" dump m "$TOK" "$OUT/$label.json"
  "$PY" "$MEASURE" speed m "$TOK" | tee "$OUT/${label}_speed.txt"
  docker stop "$name" >/dev/null 2>&1     # EXACT name, never pattern-kill
  echo "[$label] done, container stopped."
}

# BF16 reference (needs high gpu-mem-util for 27B; last --gpu-memory-utilization wins).
serve_dump_speed pq27b_bf16 "$BF16_CT" bf16 --gpu-memory-utilization 0.90
# OURS — Gridbook engages (quant_method=gridbook; CB + stock-CT delegation).
serve_dump_speed pq27b_ours "$OURS_CT" ours --gpu-memory-utilization 0.85
# PrismaAURA-5.5 — stock compressed-tensors (vLLM native; plugin idle).
serve_dump_speed pq27b_aura "$AURA_CT" aura --gpu-memory-utilization 0.85

echo "############################################################################"
echo "## A/B RESULTS (vs the SAME bf16 session)"
echo "== OURS vs BF16 =="; "$PY" "$MEASURE" compare "$OUT/bf16.json" "$OUT/ours.json"
echo "  PPL(ours):"; "$PY" "$MEASURE" ppl "$OUT/ours.json"
echo "== PrismaAURA-5.5 vs BF16 =="; "$PY" "$MEASURE" compare "$OUT/bf16.json" "$OUT/aura.json"
echo "  PPL(aura):"; "$PY" "$MEASURE" ppl "$OUT/aura.json"
echo "  PPL(bf16):"; "$PY" "$MEASURE" ppl "$OUT/bf16.json"
echo "## speed tables: $OUT/{bf16,ours,aura}_speed.txt"
echo "############################################################################"
