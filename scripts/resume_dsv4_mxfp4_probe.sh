#!/usr/bin/env bash
# ============================================================================
# resume_dsv4_mxfp4_probe.sh — measure whatever layers are still missing,
#                              without ever colliding with completed work.
# ============================================================================
# THE BUG THIS EXISTS TO PREVENT (hit for real, 2026-08-03):
# ``incremental_measure_quant_cost`` numbers its shards by INDEX WITHIN THE
# RUN, not by absolute layer -- a run started at --start-layer 27 writes its
# first layer to cost_shard_000.pkl. Resuming into the SAME work-dir therefore
# silently overwrites completed shards, one per layer. The previous version of
# this script resumed in place and destroyed layers 0-2 (the log line reads
# "stale shard 3: recomputing ... include='layers\.30\.'") before it was
# caught.
#
# The original guard checked that shard FILES were contiguous (000..NNN, no
# gaps). That invariant held perfectly while the data underneath was being
# replaced, because it asserted file NUMBERING rather than LAYER COVERAGE --
# the property actually being relied on. A guard must assert the thing you
# care about, not a thing that usually correlates with it.
#
# TWO STRUCTURAL FIXES, so this cannot recur rather than merely being noticed:
#   1. every segment runs in its OWN work-dir, so shard indices from different
#      runs can never address the same file;
#   2. results are promoted into a by-layer store keyed by the layer the shard
#      actually CONTAINS (read back from the data), and promotion REFUSES to
#      overwrite an existing layer.
# Coverage is then derived from that store's content, never from filenames.
# ============================================================================
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
WORK="${WORK:-${RUN_ROOT}/prod-cal-0p6-v2}"
ART="${ART:-${WORK}/artifacts-mxfp4}"
PROBE="${PROBE:-${ART}/probe-k12k18}"
REPO="${REPO:-/home/rob/pq-mxfp4-wt}"
IMAGE="${IMAGE:-gridbook:test}"
N_LAYERS="${N_LAYERS:-43}"
MENU="${MENU:-NVFP4_CB_K12,NVFP4_CB_K13,NVFP4_CB_K14,NVFP4_CB_K15,NVFP4_CB_K16,NVFP4_CB_K17,NVFP4_CB_K18}"

# ===========================================================================
# HARD GATE — this script produces CORRECT DATA with an UNUSABLE PROVENANCE
# CHAIN, and that is the worst failure shape available: it succeeds, and the
# failure surfaces hours later at the allocator instead of here.
#
# WHY. A CB cost table carries a render identity attesting what a run actually
# measured, and `cost_shard_is_reusable` refuses ANY measured-CB shard by
# design ("would need its exact decoded source tensors reloaded and re-hashed
# before acceptance"). So the tool cannot merge segments, and a hand-assembled
# table is refused in turn by four independent guards: missing serialized-
# payload identity, lattice bytes not covering the format menu, render identity
# not binding those bytes for its scope, and format scope not exact/canonical/
# qname-complete. Those guards are correct. An attestation a cost run is
# supposed to EARN must not be writable by hand, so there is no fix on this
# side -- only a contiguous single run produces a mergeable, validated table.
#
# The DATA from a segmented run is still fine for direct table reads (hull
# analysis, coverage, interpolator validation all worked on it). It is only
# unusable as an allocator cost cache. If that is genuinely what you want,
# say so explicitly.
# ===========================================================================
if [ "${I_UNDERSTAND_UNMERGEABLE:-}" != "1" ]; then
  cat >&2 <<'MSG'
refusing: segmented resume produces a cost table the allocator will REJECT.

  Shards from a resumed segment cannot be merged into a validated cost cache:
  cost_shard_is_reusable fails closed on every measured-CB shard, and a
  hand-assembled render identity is refused by the CB provenance guards. You
  will get 43 correct layers and an allocator that will not load them.

  For an ALLOCATION, run all layers in one contiguous container instead.
  For direct table analysis (hull / coverage / interpolator validation), the
  segmented data is fine -- set I_UNDERSTAND_UNMERGEABLE=1 to proceed.
MSG
  exit 64
fi

if [ -n "$(docker ps -q -f name=pq-mxfp4-probe)" ]; then
  echo "refusing: pq-mxfp4-probe is still RUNNING; stop it first" >&2
  exit 1
fi
mkdir -p "${PROBE}/by-layer"

promote() {
  python3 - "$PROBE" <<'PY'
import glob, os, pickle, re, sys
probe = sys.argv[1]
promoted, refused = [], []
for f in sorted(glob.glob(f"{probe}/work-seg*/shards/cost_shard_*.pkl")):
    try:
        d = pickle.load(open(f, "rb"))
    except Exception:
        continue
    layers = {int(m.group(1)) for n in d.get("costs", {})
              if (m := re.match(r"model\.layers\.(\d+)\.", n))}
    if len(layers) != 1 or len(d.get("formats", [])) != 7:
        continue
    layer = layers.pop()
    out = f"{probe}/by-layer/layer_{layer:03d}.pkl"
    if os.path.exists(out):
        refused.append(layer)          # never clobber a completed layer
        continue
    pickle.dump(d, open(out, "wb"))
    promoted.append(layer)
if promoted:
    print(f"[promote] stored layers {sorted(promoted)}")
if refused:
    print(f"[promote] already present, left alone: {sorted(refused)}")
PY
}
promote

readarray -t SEGS < <(python3 - "$PROBE" "$N_LAYERS" <<'PY'
import glob, re, sys
probe, n = sys.argv[1], int(sys.argv[2])
have = {int(m.group(1)) for f in glob.glob(f"{probe}/by-layer/layer_*.pkl")
        if (m := re.search(r"layer_(\d+)\.pkl$", f))}
missing = sorted(set(range(n)) - have)
segs = []
for layer in missing:
    if segs and layer == segs[-1][1] + 1:
        segs[-1][1] = layer
    else:
        segs.append([layer, layer])
print(f"# have {len(have)}/{n}; missing {len(missing)}: {missing}", file=sys.stderr)
for a, b in segs:
    print(f"{a} {b + 1}")
PY
)
[ ${#SEGS[@]} -eq 0 ] && { echo "all ${N_LAYERS} layers present in by-layer store"; exit 0; }

for seg in "${SEGS[@]}"; do
  read -r START END <<<"$seg"
  WD="${PROBE}/work-seg${START}"
  echo "[resume] segment layers ${START}..$((END - 1)) -> ${WD}"
  rm -rf "$WD"; mkdir -p "$WD"
  docker rm -f pq-mxfp4-probe 2>/dev/null || true
  docker run -d --name pq-mxfp4-probe --gpus all --ipc=host \
    -v "${RUN_ROOT}:${RUN_ROOT}" -v "${REPO}:/pq" -w /pq \
    -e PYTHONPATH=/pq \
    -e PRISMAQUANT_CB_EXT_DIR="${RUN_ROOT}/ext" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    -e PRISMAQUANT_CB_LADDER_INTERP=1 \
    -e CB_CODEBOOK_SOURCE=lattice -e CB_SCALE_CODING=two_tier -e CB_SCALE_SWEEP=1 \
    -e PRISMAQUANT_CB_ENCODE_TIER=balanced \
    -e PRISMAQUANT_CB_COL_WEIGHTS="${ART}/cb_col_weights.pkl" \
    -e PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE="${ART}/cb_col_weights.pkl.provenance.json" \
    --entrypoint bash "$IMAGE" -c "
python3 -m prismaquant.incremental_measure_quant_cost \
  --model ${RUN_ROOT}/source --cost-mode local \
  --probe ${ART}/probe.pkl \
  --activation-cache-dir ${WORK}/act \
  --formats '${MENU}' \
  --output ${WD}/cost_seg.pkl \
  --work-dir ${WD} \
  --device cuda --dtype bf16 --mode batched --chunk-size 256 \
  --layers-per-shard 1 --start-layer ${START} --end-layer ${END} \
  --skip-missing-activations --no-include-lm-head \
  >> ${PROBE}/cost.log 2>&1"
  echo "[resume] launched layers ${START}..$((END - 1)) at $(date -Is)"
  # One GPU tenant at a time is this box's discipline, so segments are serial.
  while [ -n "$(docker ps -q -f name=pq-mxfp4-probe)" ]; do sleep 60; done
  promote
done
echo "[resume] all segments complete"
