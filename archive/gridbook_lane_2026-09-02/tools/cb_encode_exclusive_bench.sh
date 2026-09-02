#!/usr/bin/env bash
# Exclusive-GPU old-vs-new validation for the CB encode optimization + the
# activation-row fix, then (on PASS only) land the branch and relaunch the
# full cost run as v2.
#
# LAYER RANGE: --end-layer is EXCLUSIVE. A single layer 0 is
# "--start-layer 0 --end-layer 1"; passing 0 asks for zero layers and the
# driver exits in ~3s with "empty layer range" having written nothing.
#
# EXIT CODE: a partial-range run trips the driver's own merged-table coverage
# gate (the probe covers all 43 body layers, this measures one), so a non-zero
# rc is EXPECTED. The shard is written before that gate, so the identity gate
# keys on the shard and distinguishes "no output" from "different output".
set -uo pipefail

PROD=pq-dsv4-cost-prod
SIDE=pq-perf-side
LOCK=/tmp/claude-1000/gpu-bench.lock
RUN=/home/rob/dq-runs/dsv4-flash-0731
OUT=$RUN/encoder-profile/exclusive
SCRATCH=$RUN/perf-bench-scratch
SHIPPED=$RUN/prod-cal-0p6/work-prod/shards/cost_shard_000.pkl
mkdir -p "$OUT" "$SCRATCH"
: > "$OUT/bench.log"
: > "$OUT/layer_seconds.txt"
rm -f "$OUT/gate.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bench.log"; }

while docker ps --format '{{.Names}}' | grep -q "^${PROD}\$"; do
  log "waiting for $PROD to exit..."; sleep 60
done
# Pre-flight: refuse to benchmark against a dirty GPU. A stray container from
# an earlier `timeout N docker run --rm` (timeout kills the CLIENT, not the
# container) silently halves every number here -- it contaminated the first
# rerun before this check existed.
cleanup() { for c in "$@"; do docker rm -f "$c" >/dev/null 2>&1; done; }
trap 'cleanup pq-perf-layer-base pq-perf-layer-new' EXIT INT TERM
cleanup pq-perf-layer-base pq-perf-layer-new
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  stray=$(docker ps --format '{{.Names}}' | grep -v -E "^${SIDE}\$" | tr '\n' ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if [ -z "$stray" ] && [ "$apps" -eq 0 ] && [ "$util" -lt 15 ]; then break; fi
  log "GPU not idle (containers='$stray' compute_apps=$apps util=${util}%); waiting 30s"
  sleep 30
done
if [ -n "$stray" ] || [ "$apps" -ne 0 ]; then
  log "REFUSING to benchmark a contended GPU: containers='$stray' apps=$apps"
  echo "CONTENDED_GPU" > "$OUT/gate.txt"; exit 1
fi
log "GPU verified idle (util=${util}%, compute_apps=0); settling 15s"
sleep 15

dmon_start() {
  nvidia-smi dmon -s pucm -d 1 -o T > "$OUT/dmon_$1.txt" 2>&1 &
  echo $!
}

# --- A: per-Linear bench, 3 warm reps, both arms -------------------------
for arm in base new; do
  pp=/pq; [ "$arm" = new ] && pp=/wt
  log "=== per-Linear bench [$arm] (PYTHONPATH=$pp) ==="
  d=$(dmon_start "perlinear_$arm")
  flock "$LOCK" docker exec -e PYTHONPATH=$pp -e PYTHONDONTWRITEBYTECODE=1 \
    "$SIDE" python3 /wt/tools/cb_encode_profile.py --reps 3 \
    --outdir "$OUT/$arm" > "$OUT/perlinear_$arm.txt" 2>&1
  sleep 2; kill "$d" 2>/dev/null
  grep -E '^\[clean\]' "$OUT/perlinear_$arm.txt" | tee -a "$OUT/bench.log"
done

# --- B: real end-to-end single-layer driver run, both arms ---------------
for arm in base new; do
  pp=/pq; [ "$arm" = new ] && pp=/wt
  wd="$SCRATCH/work-$arm"; rm -rf "$wd"; mkdir -p "$wd"
  log "=== end-to-end layer 0 [$arm] ==="
  d=$(dmon_start "layer_$arm")
  t0=$(date +%s)
  docker rm -f "pq-perf-layer-$arm" >/dev/null 2>&1
  flock "$LOCK" docker run --rm --name "pq-perf-layer-$arm" \
    --gpus all --ipc=host --entrypoint bash \
    -v "$RUN":"$RUN" -v /home/rob/pq-perf-wt:/wt \
    -v /home/rob/prismaquant-ultraplan:/pq:ro \
    -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier \
    -e CB_SCALE_SWEEP=1 -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
    -e PYTHONPATH=$pp -e PYTHONDONTWRITEBYTECODE=1 \
    -e PRISMAQUANT_CB_EXT_DIR=$RUN/ext \
    -e PRISMAQUANT_CB_COL_WEIGHTS=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl \
    -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl.provenance.json \
    gridbook:test -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model $RUN/source --cost-mode local \
  --probe $RUN/prod-cal-0p6/artifacts/probe.pkl \
  --activation-cache-dir $RUN/prod-cal-0p6/act \
  --formats 'NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,BF16' \
  --output $wd/cost_full.pkl --work-dir $wd \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer 0 --end-layer 1 \
  --skip-missing-activations --no-include-lm-head
" > "$OUT/layer_$arm.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  sleep 2; kill "$d" 2>/dev/null
  log "end-to-end layer 0 [$arm]: $((t1 - t0)) s (driver rc=$rc; non-zero is expected here)"
  echo "$arm seconds=$((t1 - t0)) rc=$rc" >> "$OUT/layer_seconds.txt"
  ls -la "$wd/shards/" 2>&1 | tail -3 | tee -a "$OUT/bench.log"
  grep -E "activation-row coverage|FATAL|empty layer range" \
    "$OUT/layer_$arm.txt" | head -4 | tee -a "$OUT/bench.log"
  cleanup "pq-perf-layer-$arm"
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$a" -ne 0 ] && log "WARNING: $a compute app(s) still resident after [$arm]"
done

# --- C: identity gate ----------------------------------------------------
log "=== identity gate: new arm layer 0 vs shipped v1 shard ==="
flock "$LOCK" docker exec -e PYTHONPATH=/wt -e PYTHONDONTWRITEBYTECODE=1 \
  "$SIDE" python3 /wt/tools/cost_layer0_identity_gate.py \
  --shipped "$SHIPPED" \
  --new "$SCRATCH/work-new/shards/cost_shard_000.pkl" \
  --base "$SCRATCH/work-base/shards/cost_shard_000.pkl" \
  --verdict "$OUT/gate.txt" >> "$OUT/bench.log" 2>&1
tail -26 "$OUT/bench.log"

log "=== dmon summaries (whole window AND busy-only) ==="
for f in "$OUT"/dmon_*.txt; do
  python3 - "$f" <<'PY' >> "$OUT/bench.log" 2>&1
import sys, statistics as st
rows = []
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) < 8 or not p[1].isdigit():
        continue
    try:
        rows.append((float(p[2]), float(p[5]), float(p[6])))
    except ValueError:
        pass
tag = sys.argv[1].split('/')[-1]
if not rows:
    print(f"{tag}: no samples")
else:
    pw = [r[0] for r in rows]; sm = [r[1] for r in rows]
    busy = [r for r in rows if r[1] >= 50]
    s = (f"{tag}: n={len(rows)} pw med={st.median(pw):.1f} "
         f"max={max(pw):.1f} sm med={st.median(sm):.0f}")
    if busy:
        s += (f" | BUSY(sm>=50) n={len(busy)} "
              f"pw med={st.median([b[0] for b in busy]):.1f} "
              f"sm med={st.median([b[1] for b in busy]):.0f}")
    else:
        s += " | BUSY(sm>=50) n=0  <-- GPU never engaged"
    print(s)
PY
done
tail -8 "$OUT/bench.log"

# --- D: land the branch and relaunch the full run as v2 -------------------
verdict="$(cat "$OUT/gate.txt" 2>/dev/null || echo MISSING)"
if [ "$verdict" != "PASS" ]; then
  log "GATE=$verdict -- not merging, not relaunching."
  log "DONE (gate=$verdict) -> $OUT"; exit 1
fi
log "identity gate PASS; fast-forwarding dsv4/flash-0731-92gb"
git -C /home/rob/prismaquant-ultraplan merge --ff-only perf/cb-encode-gpu \
  >> "$OUT/bench.log" 2>&1 || { log "MERGE FAILED"; exit 1; }
git -C /home/rob/prismaquant-ultraplan log --oneline -5 >> "$OUT/bench.log"

V2=$RUN/prod-cal-0p6-v2
mkdir -p "$V2/artifacts" "$V2/work-prod" "$V2/logs"
log "relaunching full 43-layer cost run as pq-dsv4-cost-prod2 -> $V2"
docker run -d --name pq-dsv4-cost-prod2 --gpus all --ipc=host --entrypoint bash \
  -v "$RUN":"$RUN" -v /home/rob/prismaquant-ultraplan:/pq \
  -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier \
  -e CB_SCALE_SWEEP=1 -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
  -e PYTHONPATH=/pq -e PRISMAQUANT_CB_EXT_DIR=$RUN/ext \
  -e PRISMAQUANT_CB_COL_WEIGHTS=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl \
  -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE=$RUN/prod-cal-0p6/artifacts/cb_col_weights.pkl.provenance.json \
  gridbook:test -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model $RUN/source --cost-mode local \
  --probe $RUN/prod-cal-0p6/artifacts/probe.pkl \
  --activation-cache-dir $RUN/prod-cal-0p6/act \
  --formats 'NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,BF16' \
  --output $V2/artifacts/cost_full.pkl --work-dir $V2/work-prod \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer 0 --end-layer 43 \
  --skip-missing-activations --no-include-lm-head \
  > $V2/logs/cost_prod2.log 2>&1
" >> "$OUT/bench.log" 2>&1
sleep 25
docker ps --filter name=pq-dsv4-cost-prod2 --format '{{.Names}} {{.Status}}' \
  | tee -a "$OUT/bench.log"
log "prod2 launched; log = $V2/logs/cost_prod2.log"
log "DONE -> $OUT"
