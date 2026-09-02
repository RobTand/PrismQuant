#!/usr/bin/env bash
# ============================================================================
# run_dsv4_flash_92gb.sh — DeepSeek-V4-Flash-0731 production calibration for
#                          the 92 decimal GB CB artifact
# ============================================================================
# Produces the production K14/K15 boundary decision the 2026-08-01 study could
# only make provisionally. The study's REFERENCE allocation was 35 expert layers
# at NVFP4_CB_K15, eight at NVFP4_CB_K14 (provisionally 23,24,25,27,28,31,32,33
# from a 35.8%-coverage probe), and 301 ordinary body Linears at FP8_CB_K36 —
# but the alloc stage below is NOT constrained to that split: the DP decides all
# 344 serving units under the exact 92 GB byte budget, may move layers between
# K14/K15 and body units onto fp4 rungs, and its selection.json is the verdict.
# A selection that deviates from the study's 35/8 split is a RESULT, not a bug
# (see the alloc-stage comment below). The study's split is 35 K15 + 8 K14 + 301
# body Linears at FP8_CB_K36. Study record: docs/ARCHITECTURE.md §9.2 and
# /home/rob/dq-runs/dsv4-flash-0731/README.md.
#
# ---------------------------------------------------------------------------
# WHY THIS RE-PROBES INSTEAD OF REUSING screen-1x128/
# ---------------------------------------------------------------------------
# The study's cost attempt failed closed, and the cause is ROUTE COVERAGE in
# the probe, not the cost sampler. The CB render is imatrix-weighted, so every
# measured CB row needs an exact col-weights vector
# (measure_quant_cost.cb_render_provenance_for_results). col-weights can only
# exist for a Linear that has an activation-cache entry
# (export_gguf.build_imatrix_from_act_cache enumerates act_dir/*.pt), and the
# study's ONE 128-token sample routed to only 3,944 of 11,008 experts — 36%,
# and just 73-95 of 256 in each of the eight boundary layers. Measured:
#
#   experts routed per layer (of 256): min 52  max 226  mean 91.7
#
# Neither sampling mode escapes that. With PRISMAQUANT_EXPERT_COST_SAMPLE set,
# _extrapolate_expert_costs fabricates a row for every skipped expert, those
# rows enter the provenance scope with no col-weights, and
# measure_quant_cost.py:124 raises "measured CB rows are missing exact
# col_weights" — which is exactly what killed screen-1x128/cost-k14-k15/
# (it left only an empty work-dir skeleton, dead ~3 s after creation).
# Unsampled, the inactive experts get no row at all and allocator.py:2427 (guard at :2422)
# refuses instead: "production CB cost coverage is incomplete ... missing/error
# rows cannot be silently pruned from the menu". The fix is more calibration
# tokens, then a re-harvest, then the cost stage.
#
# ---------------------------------------------------------------------------
# CALIBRATION DESIGN DECISIONS
# ---------------------------------------------------------------------------
# SAMPLING — PRISMAQUANT_EXPERT_COST_SAMPLE is deliberately UNSET (exact).
#   It is never exported by run-pipeline.sh; only the older dense-model drivers
#   set it. Sampled mode stamps output_mse_measured=False on packed-expert rows
#   and fabricates extrapolated rows on the per-expert path, which both defeats
#   P5a pricing and trips the col-weights gate. Exact mode is also what the
#   allocator's own remedy text asks for.
#
# LADDER SCOPE — {K14, K15} only, plus FP8_CB_K36 for the 301 body Linears and
#   BF16 for menu completeness. The frozen decision is per-layer K14 vs K15;
#   measuring the whole fp4 ladder multiplies encode volume across 43 layers x
#   256 experts x 3 projections for rungs no one can select.
#   CORRECTION (verified 2026-08-02): BF16 CANNOT act as the identity-activation
#   escape rung on this checkpoint. DSv4-Flash is FP8-source (manifest scan:
#   33,393 fp8 / 238 bf16), and PASSTHROUGH_SOURCE_REQUIREMENTS masks BF16
#   model-wide with `source_dtype_mismatch`. An uncovered Linear therefore gets
#   ZERO candidates, not a BF16 fallback — which is why the unrouted-expert rule
#   below is mandatory rather than a convenience.
#
# CB_LADDER_INTERP — left at 0, and it is INERT here regardless: _cb_ladder_split
#   requires at least 4 rungs of one CB (family, mode) before it will anchor and
#   interpolate. With two fp4 rungs every rung is measured either way. Setting it
#   would matter only for a full-ladder run, where it also fills the predicted
#   rungs WEIGHT-ONLY and so removes them from P5a's measured branch.
#
# P5a ACTIVATION-FAIR PRICING — left ON (the default; it is documented as "NOT a
#   knob to tune"). Its calibration sample is harvested from this run's own cost
#   pickle by collect_activation_calibration_rows; there is no separate stage or
#   artifact. A row qualifies only if it carries BOTH a measured output_mse and a
#   weight-only estimator, both > 0. Exact mode is what produces those rows.
#   NOTE the live hazard: nvfp4_cb (act_bits=4) and fp8_cb (act_bits=8) are BOTH
#   activation-quantizing, and activation_fair_pricing.calibrate raises if one
#   family ends up calibrated while the other still has uncorrected weight-only
#   rows. With a single menu measured exactly in one pass, both families get
#   measured rows and the mixed-scale assertion stays quiet. If it ever fires,
#   the answer is to close the measurement gap, not to set the kill switch.
#
# ---------------------------------------------------------------------------
# RECONCILED AGAINST THE GRIDBOOK D0.1 CONTRACT (2026-08-02)
# ---------------------------------------------------------------------------
# D0.1 rejects CB on attn.wo_a (its forward bypasses apply() and reads .weight
# directly, so a CB assignment there would serve silently wrong results), the
# ffn.gate router, both compressor.fused_wkv_wgate, indexer.weights_proj,
# lm_head and embed_tokens. VERIFIED against the probe inventory: the 33,325
# selectable Linears are 33,024 routed-expert projections plus exactly seven
# leaves per layer —
#     self_attn.{wq_a, wq_b, wkv, wo_b} + mlp.shared_experts.{gate,up,down}_proj
# self_attn.wo_a, mlp.gate, the compressor and indexer leaves, lm_head and
# embed_tokens are NOT probeable and therefore cannot receive a CB rung; the
# exporter keeps their source layout and charges them to the immutable floor.
# The study's 92 GB arithmetic already assumes exactly that (its own note about
# the 86 wrongly-included wo_a/mlp.gate tensors), so NO byte re-check is needed.
# wq_a and wkv are the two shards gridbook fuses into fused_wqa_wkv — both are
# in D0.1's native-CB set. indexer.wq_b is legal for CB under D0.1 but is not
# in the probe inventory, so it stays source-format; that is safe (under-
# quantifying), and is the one place this artifact leaves CB bytes on the table.
#
# Export/serve follow-ups from D0.1, NOT handled here: gridbook accepts both
# `model.layers....` and `layers....` spellings but the exporter picks per
# tensor via a skeleton check, so the first export's manifest must be diffed
# against the contract; and the serve smoke needs --kv-cache-dtype fp8 (the
# SM120 MLA backend asserts it).
#
# ---------------------------------------------------------------------------
# STAGES (each is independently resumable; run with STAGE=<name>)
# ---------------------------------------------------------------------------
#   probe   — production-calibration probe + activation cache   [GPU, hours]
#   colw    — harvest cb_col_weights.pkl from act/              [CPU/GPU]
#   cost    — exact per-(Linear, rung) cost, K14/K15/K36/BF16   [GPU, hours]
#   alloc   — byte-budget allocation -> selection.json          [CPU]
#   export  — stream the 92 GB CB artifact from layer_config    [GPU, hours]
#
# The export stage collapses the allocator's 768 per-expert entries per layer
# into the two packed stacks gridbook's deepseek_v4 contract names
# (`ffn.experts.{gate_up_proj,down_proj}`), refusing any layer whose experts do
# not agree on one format.
#
# The cost stage shards by layer and reuses completed shards, so it can be
# resumed or split with --start-layer/--end-layer. CB shards are deliberately
# never reused across runs (cost_shard_is_reusable fails closed on CB formats),
# so a resumed cost run re-measures the layers of any shard it did not finish.
#
# ---------------------------------------------------------------------------
# ALLOC STAGE PRE-FLIGHT (2026-08-02, verified against layer 0's REAL rows)
# ---------------------------------------------------------------------------
# Two defects that would each have killed the alloc stage AFTER the ~12-hour
# cost run finished are fixed below; both were reproduced on real production
# rows, not on a fixture.
#
# 1. Required CB producer flags were MISSING. allocator.main hard-exits within
#    seconds on any CB menu without all five CB context flags, including the
#    now-explicit `--cb-ldlq 0` assignment identity ("refusing implicit render
#    defaults") — the render the cost stage
#    measured and the render the exporter ships must be provably the same
#    one, so the allocator will not infer them from the cost pickle.
# 2. --pareto-targets was left at the default. See PARETO_TARGETS below.
#
# LEGALITY, verified on the real shard: NVFP4_CB_K14, NVFP4_CB_K15 and
# FP8_CB_K36 are legal for BOTH expert shapes — w13-class (out 2048, in 4096)
# and w2/down_proj-class (out 4096, in 2048) clear the CB in_features % 256
# superblock rule and both fp4 (%8) and fp8 (%16) out_features load gates. The
# ONLY masked format is BF16, model-wide, source_dtype_mismatch, exactly as
# the CORRECTION note above predicts. All 775 real layer-0 rows carry the
# full {K14, K15, K36} menu; no CB rung is masked on any shape.
#
# DP GRANULARITY, also verified: aggregate_packed_serving_groups collapses a
# layer's 768 routed projections (256 experts x gate/up/down) into ONE group,
# and fused-sibling aggregation finds ZERO groups here — so the DP decides 43
# expert-layer units plus 301 body Linears, exactly the 344 units the per-layer
# K14-vs-K15 question assumes. It is NOT constrained to the study's 35/8 split
# or to body-at-K36: the min-Δloss DP may put body Linears on fp4 rungs and
# spread the expert layers across all three rungs. The 92 GB budget is the
# constraint; the format map is the allocator's verdict, not the study's.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
MODEL_PATH="${MODEL_PATH:-${RUN_ROOT}/source}"
WORK_DIR="${WORK_DIR:-${RUN_ROOT}/prod-cal-0p6}"
IMAGE="${IMAGE:-gridbook:test}"
STAGE="${STAGE:-all}"

# Production calibration. The study's 1x128 screen reached 36% expert
# coverage; 16x512 = 8,192 tokens gives ~49k routed assignments per layer
# against 256 experts. Verify coverage after the probe (see `colw` stage).
NSAMPLES="${NSAMPLES:-16}"
SEQLEN="${SEQLEN:-512}"
ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT:-64}"

# The diverse-v1 corpus (prose 0.4 / code 0.2 / math 0.2 / multilingual 0.2,
# docs/design/calibration_diverse_v1.md) rather than the wikitext default: a
# mixed-domain corpus fires more of the routing table per token, which is the
# binding constraint here, and it is a local .jsonl so the probe does not need
# the `datasets` package — gridbook:test does not ship it.
CALIB_DIR="${CALIB_DIR:-/home/rob/dq-runs/calibration}"
DATASET="${DATASET:-${CALIB_DIR}/diverse-v1.jsonl}"

FORMATS="${FORMATS:-NVFP4_CB_K14,NVFP4_CB_K15,FP8_CB_K36,BF16}"

# 92 decimal GB with the 256 MiB non-tensor reserve (study README).
TARGET_DISK_GB="${TARGET_DISK_GB:-92}"
ARTIFACT_OVERHEAD_RESERVE_BYTES="${ARTIFACT_OVERHEAD_RESERVE_BYTES:-268435456}"

# PARETO GRID — the allocator's default (4.5..8.25) was written for a 4-bit/
# 8-bit menu and does not touch this one. This menu's cheapest allocation is
# 2.0313 bpp (all NVFP4_CB_K14 = 87.153 GB whole-artifact) and the 92 GB plan
# sits at 2.169 bpp; every default rung is more than twice as dense, so the
# byte-budget selector saw nothing below 172.422 GB and reported "92.000 is
# below the floor" — verified end to end on layer 0's REAL production rows
# (dq-runs/dsv4-flash-0731/alloc-feasibility-check/). allocator.main now
# probes below the sweep before declaring a budget infeasible, so this list is
# belt-and-braces: it also puts the low rungs on the Pareto CSV and gives the
# ratchet a tighter bracket around the budget (measured: 91.987 GB shipped
# with this grid vs 91.890 GB from the repair path alone).
PARETO_TARGETS="${PARETO_TARGETS:-2.04,2.06,2.08,2.10,2.12,2.14,2.15,2.16,2.17,2.18,2.20,2.25,2.35,2.50,3.00,3.50,4.00,4.50}"

mkdir -p "$WORK_DIR"/{artifacts,act,work,logs}

# The study's containers mounted the run root at its own host path, so the
# absolute paths in probe.log resolve identically inside and out. Keep that.
DOCKER_COMMON=(
  --rm --gpus all --ipc=host
  -v "${RUN_ROOT}:${RUN_ROOT}"
  -v "${CALIB_DIR}:${CALIB_DIR}:ro"
  -v "${REPO}:/pq" -w /pq
  -e PYTHONPATH=/pq
  -e PRISMAQUANT_CB_EXT_DIR="${RUN_ROOT}/ext"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1
  # Explicit CB producer identity. cb_serialization_context_from_env refuses
  # to guess: measured cost and shipped bytes must be the same render.
  -e CB_CODEBOOK_SOURCE=lattice
  -e CB_SCALE_CODING=two_tier
  -e CB_SCALE_SWEEP=1
  -e PRISMAQUANT_CB_ENCODE_TIER=balanced
  # PRISMAQUANT_EXPERT_COST_SAMPLE intentionally NOT set — see header.
  #
  # VALUE-LESS forwards: `-e NAME` passes the HOST's value through when the
  # variable is set and passes nothing at all when it is not. That is the
  # property wanted here — the launch wrapper decides, per invocation, and the
  # script needs no per-run edit and carries no default of its own. Writing
  # `-e NAME=` instead would define them as empty INSIDE the container, which
  # is a different statement and, for an acknowledgement flag, the wrong one.
  #
  #   PQ_ALLOW_ROUTE_PENDING=1        acknowledge shipping a route-pending
  #                                   passthrough (the exporter refuses
  #                                   otherwise, and records the
  #                                   acknowledgement in artifact provenance)
  #   PQ_EXPORT_EXCLUDE_NAMESPACES    comma-separated tensor-name prefixes to
  #                                   OMIT from the artifact entirely
  #                                   (e.g. `mtp.`); empty/unset changes nothing
  -e PQ_ALLOW_ROUTE_PENDING
  -e PQ_EXPORT_EXCLUDE_NAMESPACES
  # LDLQ render identity (value-less forwards, same contract as above):
  # the launch wrapper decides per invocation. A cost/alloc/export chain
  # must run under ONE consistent setting of these — the allocator's
  # explicit --cb-ldlq assignment identity and the exporter's stamp
  # re-validation refuse a mismatch. PRISMAQUANT_CB_ENCODE_COMPILE is
  # part of the byte identity (compiled != eager, each internally
  # deterministic — GB10 canaries 2026-08-08); pin it explicitly.
  -e PRISMAQUANT_CB_LDLQ
  # Stratified experts-per-(layer,projection) subsample for the cost
  # stage (value-less forward). The 2026-08-02 header objection
  # (fabricated rows lacking col-weights) applied to a 36%-coverage
  # capture; with full 33,325-entry col-weight coverage the sampled
  # rows price cleanly, and the DP decides whole expert-layer units,
  # so an unbiased per-layer mean matches decision granularity. The
  # exact-vs-sampled choice is the launch wrapper's, per invocation.
  -e PRISMAQUANT_EXPERT_COST_SAMPLE
  # RD-ladder interpolation plan (anchors+holdout measured, remaining
  # rungs fitted per tensor, holdout-gated with measured fallback) —
  # value-less forwards; the launch wrapper opts in per invocation.
  -e PRISMAQUANT_CB_LADDER_INTERP
  -e PRISMAQUANT_CB_LADDER_ANCHORS
  -e PRISMAQUANT_CB_LADDER_HOLDOUT
  -e PRISMAQUANT_CB_LDLQ_SCOPE
  -e PRISMAQUANT_CB_LDLQ_GATE
  -e PRISMAQUANT_CB_MINCHAIN
  -e PRISMAQUANT_CB_ENCODE_COMPILE
)

run_probe() {
  echo "[dsv4] probe: ${NSAMPLES}x${SEQLEN} production calibration -> ${WORK_DIR}/act"
  docker run -d --name pq-dsv4-probe "${DOCKER_COMMON[@]}" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.incremental_probe \
  --model ${MODEL_PATH} \
  --dataset ${DATASET} \
  --nsamples ${NSAMPLES} --seqlen ${SEQLEN} \
  --device cuda --dtype bf16 \
  --output ${WORK_DIR}/artifacts/probe.pkl \
  --activation-cache-dir ${WORK_DIR}/act \
  --work-dir ${WORK_DIR}/work \
  --layers-per-shard auto --unified-sweep \
  --prefetch-lookahead 2 --prefetch-workers 2 \
  --prefetch-min-available-gb 40 \
  --activation-rows-limit ${ACTIVATION_ROWS_LIMIT} \
  --calibration-modality text-only \
  > ${WORK_DIR}/logs/probe.log 2>&1"
}

run_colw() {
  echo "[dsv4] col-weights: harvesting imatrix from ${WORK_DIR}/act"
  # Coverage gate: the cost stage needs an exact col-weights vector for every
  # CB row it measures. 33,325 is full coverage of the probe inventory.
  docker run --name pq-dsv4-colw "${DOCKER_COMMON[@]}" \
    -e CB_ACT_DIR="${WORK_DIR}/act" \
    -e CB_COL_WEIGHTS="${WORK_DIR}/artifacts/cb_col_weights.pkl" \
    -e MODEL_PATH="${MODEL_PATH}" \
    -e COLW_LOG="${WORK_DIR}/logs/colw.log" \
    -e PROBE_PKL="${WORK_DIR}/artifacts/probe.pkl" \
    --entrypoint bash "$IMAGE" -c '
python3 - <<\PYEOF 2>&1 | tee "$COLW_LOG"
import os, pickle, torch
from pathlib import Path
# Inlined from export_gguf.build_imatrix_from_act_cache: llama.cpp imatrix
# semantics (mean squared activation per input column). Inlined because
# export_gguf imports `gguf` at module scope and gridbook:test does not ship
# it; the arithmetic here is identical.
act = Path(os.environ["CB_ACT_DIR"]); out = os.environ["CB_COL_WEIGHTS"]
cw = {}
for p in sorted(act.glob("*.pt")):
    blob = torch.load(p, map_location="cpu", weights_only=False)
    inputs = blob.get("inputs") if isinstance(blob, dict) else None
    if inputs is None or inputs.ndim != 2:
        continue
    name = (blob.get("name") if isinstance(blob, dict) else None) or p.stem.replace("__", ".")
    cw[name] = inputs.float().pow(2).mean(dim=0)
if not cw:
    raise SystemExit(f"no activation cache under {act!r}")
# synthesize_packed_expert_col_weights is a no-op on DSv4: it only fills
# packed 3-D stack entries, and DSv4-Flash exposes routed experts as per-expert
# nn.Linears. Those need the declared unrouted-expert rule instead.
import pickle as _p
from prismaquant.moe_imatrix import synthesize_unrouted_expert_col_weights
stats = _p.load(open(os.environ["PROBE_PKL"], "rb"))["stats"]
report = synthesize_unrouted_expert_col_weights(stats, cw)
n_syn = len(report["names"])
print("neutral-prior entries for never-routed experts: "
      + str(n_syn) + " (rule=" + report["rule"] + ", basis=" + report["basis"] + ")")
import json as _j
_j.dump(report, open(os.environ["CB_COL_WEIGHTS"] + ".provenance.json", "w"), indent=1)
with open(out, "wb") as fh:
    pickle.dump(cw, fh)
print(f"wrote {out}: {len(cw)} entries (33,325 = full CB coverage)")
PYEOF'
}

run_cost() {
  echo "[dsv4] cost: exact per-(Linear, rung) over ${FORMATS}"
  docker run -d --name pq-dsv4-cost "${DOCKER_COMMON[@]}" \
    -e PRISMAQUANT_CB_COL_WEIGHTS="${WORK_DIR}/artifacts/cb_col_weights.pkl" \
    -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE="${WORK_DIR}/artifacts/cb_col_weights.pkl.provenance.json" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model ${MODEL_PATH} \
  --cost-mode local \
  --probe ${WORK_DIR}/artifacts/probe.pkl \
  --activation-cache-dir ${WORK_DIR}/act \
  --formats '${FORMATS}' \
  --output ${WORK_DIR}/artifacts/cost.pkl \
  --work-dir ${WORK_DIR}/work \
  --device cuda --dtype bf16 \
  --mode batched --chunk-size 256 \
  --layers-per-shard 1 \
  --start-layer ${START_LAYER:-0} --end-layer ${END_LAYER:-43} \
  --skip-missing-activations --no-include-lm-head \
  > ${WORK_DIR}/logs/cost.log 2>&1"
}

run_alloc() {
  echo "[dsv4] allocate: ${TARGET_DISK_GB} GB byte budget, exact accounting"
  docker run --name pq-dsv4-alloc "${DOCKER_COMMON[@]}" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.allocator \
  --probe ${WORK_DIR}/artifacts/probe.pkl \
  --costs ${WORK_DIR}/artifacts/cost.pkl \
  --formats '${FORMATS}' \
  --target-bits 2.17 \
  --pareto-targets '${PARETO_TARGETS}' \
  --target-disk-gb ${TARGET_DISK_GB} \
  --artifact-overhead-reserve-bytes ${ARTIFACT_OVERHEAD_RESERVE_BYTES} \
  --target-profile nvfp4_cb \
  --cb-scale-coding two_tier \
  --cb-codebook-source lattice \
  --cb-scale-sweep 1 \
  --cb-ldlq 0 \
  --cb-encode-tier balanced \
  --cb-col-weights ${WORK_DIR}/artifacts/cb_col_weights.pkl \
  --layer-config ${WORK_DIR}/artifacts/layer_config.json \
  --pareto-csv ${WORK_DIR}/artifacts/pareto.csv \
  > ${WORK_DIR}/logs/alloc.log 2>&1
# selection.json lands beside pareto.csv; the eight K14 layers are its verdict.
python3 - <<'PY'
import json
sel = json.load(open('${WORK_DIR}/artifacts/selection.json'))
print(json.dumps(sel, indent=2)[:4000])
PY"
}

run_export() {
  echo "[dsv4] export: ${RUN_ROOT}/artifact-92gb from ${WORK_DIR}/artifacts/layer_config.json"
  # Runs under the SAME DOCKER_COMMON producer identity as cost/alloc
  # (CB_CODEBOOK_SOURCE / CB_SCALE_CODING / CB_SCALE_SWEEP /
  # PRISMAQUANT_CB_ENCODE_TIER), so the render the allocator priced and the
  # render this ships are provably the same one — the exporter re-validates
  # that stamp against the layer_config and refuses a mismatch.
  #
  # --activation-cache-dir is MANDATORY, not optional: the layer_config carries
  # the static W4A4 activation contract, and without the cache the export
  # refuses rather than shipping uncalibrated fused W4A4.
  #
  # Deliberately absent, and they must stay absent for a production artifact:
  #   --allow-unstamped-research  (would ship CB bytes with no render identity)
  #   --reuse-prior               (DELTA-EXPORT is disabled; fails closed anyway)
  docker run --name pq-dsv4-export "${DOCKER_COMMON[@]}" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.export_nvfp4_cb_streaming \
  --model-dir ${MODEL_PATH} \
  --layer-config ${WORK_DIR}/artifacts/layer_config.json \
  --out ${RUN_ROOT}/artifact-92gb \
  --col-weights ${WORK_DIR}/artifacts/cb_col_weights.pkl \
  --activation-cache-dir ${WORK_DIR}/act \
  --codebook-source lattice \
  --scale-coding two_tier \
  > ${WORK_DIR}/logs/export.log 2>&1"
}

case "$STAGE" in
  probe)  run_probe  ;;
  colw)   run_colw   ;;
  cost)   run_cost   ;;
  alloc)  run_alloc  ;;
  export) run_export ;;
  all)   run_probe; echo "[dsv4] probe detached; run STAGE=colw once it exits" ;;
  *) echo "unknown STAGE=$STAGE (probe|colw|cost|alloc|export|all)" >&2; exit 2 ;;
esac
