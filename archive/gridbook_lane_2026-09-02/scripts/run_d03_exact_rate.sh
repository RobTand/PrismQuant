#!/usr/bin/env bash
# ============================================================================
# run_d03_exact_rate.sh — gridbook ROADMAP D0.3 exact-rate experiments (P5d)
# ============================================================================
#
# Runs the two experiments gridbook ROADMAP D0.3 names, against a model's
# EXISTING probe/cost artifacts:
#
#   (i)  FP8_CB_K36 vs vanilla NVFP4 on dense units at matched EXACT
#        whole-artifact bytes; and
#   (ii) below 4.5 bpw, byte-neutral sweeps where vanilla-NVFP4 promotions on
#        chosen layers are FUNDED by cheaper CB rungs elsewhere.
#
# *** THIS PREPARES RELEASE-GATE EVIDENCE. IT DOES NOT CONSTITUTE IT. ***
#
# D0.3 is an empirical release gate and the gate is the served NATIVE-PARITY
# protocol (streaming TTFT/ITL/TPS percentiles + served KL/PPL/tasks over the
# representative workload matrix, with format/rung, layout, activation
# quantization, concrete backend, GPU/runtime identity, TP and fallback state
# recorded). Everything this script emits is labelled PROPOSAL DATA: exact
# offline byte accounting plus predicted Delta-loss under the P5a
# activation-fair pricing, with the P5c constraint/provenance stamps.
#
# CPU-only and deterministic: no probe, no render, no GPU. It reads pickles
# and writes one JSON report, so it is safe to run while the GPU is busy.
#
# Usage:
#     bash scripts/run_d03_exact_rate.sh
# Override any variable inline, e.g.:
#     WORK_DIR=/home/rob/dq-runs/prod-27b-cb \
#     BASELINE_CB_FORMAT=FP8_CB_K28 \
#     bash scripts/run_d03_exact_rate.sh
#
# Scope note, repeated in the report: PACKED-EXPERT VANILLA NVFP4 IS EXCLUDED
# from the contest. The producer profile denies stock NVFP4/FP8 on packed
# expert stacks because no stock-compressed-tensors packed-expert emit path
# exists in the container, and building one is out of scope under the
# one-payload / no-new-packer rule. Unlocking it is gridbook D0.2.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- inputs ----------------------------------------------------------------
# NEVER write artifacts to /tmp. Keep the work dir under /home/rob.
: "${WORK_DIR:=/home/rob/dq-runs/d03-exact-rate}"
: "${PROBE_PATH:=${WORK_DIR}/artifacts/probe.pkl}"
: "${COST_PATH:=${WORK_DIR}/artifacts/cost.pkl}"
: "${OUT_JSON:=${WORK_DIR}/artifacts/d03_exact_rate.json}"

# --- the contest ------------------------------------------------------------
# The menu must contain BOTH contenders plus the CB ladder that funds the
# byte-neutral sweep. NVFP4 is 4.500 bpw effective and FP8_CB_K36 is 4.508, so
# (i) is the near-matched-rate pair; K28..K35 are the cheaper rungs (ii) draws
# funding from.
: "${FORMATS:=FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,NVFP4,BF16}"
: "${CB_FORMAT:=FP8_CB_K36}"
: "${NATIVE_FORMAT:=NVFP4}"
# Baseline rung of the byte-neutral sweep. Set this BELOW 4.5 bpw for the
# experiment D0.3 actually asks for: K28 = 3.508 bpw, K32 = 4.008 bpw.
: "${BASELINE_CB_FORMAT:=FP8_CB_K32}"
: "${CB_GRID:=fp8}"
: "${PROMOTE_COUNTS:=1,2,4,8,16}"
# Format pinned on units outside the contest so the two arms of (i) differ
# only on contested rows. Empty = leave them out of both arms.
: "${BASELINE_FORMAT:=}"
# Packed experts excluded BY DEFAULT (see the scope note above).
: "${EXCLUDE_MARKERS:=.experts.,.expert_}"

# --- the serving profile ----------------------------------------------------
# The profile supplies BOTH the legality gate (which units may take which
# format) and the P5b serving lanes, so the report says which selected rungs
# ride a backed fused mid-M lane and which take the expand+GEMM fallback.
: "${TARGET_PROFILE:=nvfp4_cb}"

# --- CB serialization contract (exact bytes depend on it) -------------------
: "${CB_SCALE_CODING:=two_tier}"
: "${CB_CODEBOOK_SOURCE:=lattice}"
: "${CB_SCALE_SWEEP:=1}"
: "${CB_ENCODE_TIER:=balanced}"

# --- optional: the P5c hard serving constraints -----------------------------
# Leave SERVE_DISPATCH_TABLE empty and no constraint is evaluated; the report
# stamps that constraints were absent and claims nothing about latency.
# The shipped example table is populated ONLY from published Gridbook
# measurements and is PROPOSAL DATA, not a qualified serving model:
#   SERVE_DISPATCH_TABLE=prismaquant/serve_dispatch_tables/gridbook_gb10_2026-08-01.example.json
: "${SERVE_DISPATCH_TABLE:=}"
: "${SERVE_WORKLOAD_MIX:=}"
: "${SLO_PREFILL_P95_TTFT_MS:=}"
: "${SLO_DECODE_P95_ITL_MS:=}"
: "${SLO_DECODE_P05_TPS:=}"
: "${SERVE_DEVICE_BUDGET_BYTES:=}"
: "${SERVE_KV_BYTES:=0}"
: "${SERVE_PEAK_SCRATCH_BYTES:=0}"

# --- activation-fair pricing (P5a) -----------------------------------------
# Defaults ON. `0` reproduces pre-P5a pricing bit-for-bit and is for bisecting
# an allocation change, not for tuning. A D0.3 contest run with it OFF is not
# activation-fair and the report says so in its own provenance block.
: "${ACTIVATION_FAIR_PRICING:=1}"
export PRISMAQUANT_ACTIVATION_FAIR_PRICING="${ACTIVATION_FAIR_PRICING}"

for path in "$PROBE_PATH" "$COST_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "[d03] ERROR: missing input artifact: $path" >&2
    echo "[d03]        run the probe/cost stages first (run-pipeline.sh), or" >&2
    echo "[d03]        point PROBE_PATH / COST_PATH at an existing run." >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$OUT_JSON")"

D03_ARGS=(
  --probe "$PROBE_PATH"
  --costs "$COST_PATH"
  --out "$OUT_JSON"
  --formats "$FORMATS"
  --target-profile "$TARGET_PROFILE"
  --cb-format "$CB_FORMAT"
  --native-format "$NATIVE_FORMAT"
  --baseline-cb-format "$BASELINE_CB_FORMAT"
  --cb-grid "$CB_GRID"
  --promote-counts "$PROMOTE_COUNTS"
  --exclude "$EXCLUDE_MARKERS"
  --cb-scale-coding "$CB_SCALE_CODING"
  --cb-codebook-source "$CB_CODEBOOK_SOURCE"
  --cb-scale-sweep "$CB_SCALE_SWEEP"
  --cb-encode-tier "$CB_ENCODE_TIER"
)
[[ -n "$BASELINE_FORMAT" ]] && D03_ARGS+=(--baseline-format "$BASELINE_FORMAT")
[[ -n "$SERVE_DISPATCH_TABLE" ]] && D03_ARGS+=(--serve-dispatch-table "$SERVE_DISPATCH_TABLE")
[[ -n "$SERVE_WORKLOAD_MIX" ]] && D03_ARGS+=(--serve-workload-mix "$SERVE_WORKLOAD_MIX")
[[ -n "$SLO_PREFILL_P95_TTFT_MS" ]] && D03_ARGS+=(--slo-prefill-p95-ttft-ms "$SLO_PREFILL_P95_TTFT_MS")
[[ -n "$SLO_DECODE_P95_ITL_MS" ]] && D03_ARGS+=(--slo-decode-p95-itl-ms "$SLO_DECODE_P95_ITL_MS")
[[ -n "$SLO_DECODE_P05_TPS" ]] && D03_ARGS+=(--slo-decode-p05-tps "$SLO_DECODE_P05_TPS")
[[ -n "$SERVE_DEVICE_BUDGET_BYTES" ]] && D03_ARGS+=(
  --serve-device-budget-bytes "$SERVE_DEVICE_BUDGET_BYTES"
  --serve-kv-bytes "$SERVE_KV_BYTES"
  --serve-peak-scratch-bytes "$SERVE_PEAK_SCRATCH_BYTES"
)

echo "[d03] gridbook ROADMAP D0.3 exact-rate experiments (ultraplan P5d)"
echo "[d03]   probe=$PROBE_PATH"
echo "[d03]   costs=$COST_PATH"
echo "[d03]   menu=$FORMATS  profile=$TARGET_PROFILE"
echo "[d03]   (i)  $CB_FORMAT vs $NATIVE_FORMAT at matched exact whole-artifact bytes"
echo "[d03]   (ii) byte-neutral sweep from $BASELINE_CB_FORMAT, promotions=$PROMOTE_COUNTS"
echo "[d03]   packed experts EXCLUDED (gridbook D0.2); markers=$EXCLUDE_MARKERS"
if [[ -n "$SERVE_DISPATCH_TABLE" ]]; then
  echo "[d03]   serving constraints from $SERVE_DISPATCH_TABLE (PROPOSAL DATA)"
else
  echo "[d03]   serving constraints: none supplied; no latency claim is made"
fi

cd "$REPO_ROOT"
PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}" CUDA_VISIBLE_DEVICES="" \
  python3 -m prismaquant.d03_exact_rate "${D03_ARGS[@]}"

echo "[d03] done -> $OUT_JSON"
echo "[d03] REMINDER: this is proposal data. Promotion requires the served"
echo "[d03] NATIVE-PARITY protocol on the exported artifacts, not this report."
