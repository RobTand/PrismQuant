#!/usr/bin/env bash
# =============================================================================
# GLM-5.3-Flash — dense-half production-anchor AURA harvest + campaign pricing
# =============================================================================
#
# WHAT THIS IS
#   The GPU seam named by glm53_stock_reprice.ANCHOR_MEASUREMENT_CONTRACT,
#   now wired: prismaquant.glm53_stock_harvest runs ONE batched streamed
#   KL-adjoint over the exact dense plan (135 units x NVFP4, production
#   render: gptq + static_act_order + joint_scale_opt), then
#   `glm53_stock_reprice campaign` prices/merges it with the measured
#   packed-expert unit-KLs and the pinned source terminals into the allocator
#   cost payload. Unblocked 2026-08-27: the streamed checkpoint guard accepts
#   the production-anchor renderer identity on a non-CB menu
#   (tests/test_streamed_cost_checkpoints.py), and the anchored lane renders
#   per-layer transiently so no retained 306B menu cache exists to require.
#
# SCALE AUDIT (re-derive-at-scale-boundaries doctrine — receipt beside
# .cost_feasibility.json, written BEFORE launch)
#   innermost ops x trip counts under the streamed substrate:
#     capture pass: 1 body stream (~306 GiB) — one forward, boundaries kept.
#     reverse pass: 1 body stream — each layer installed once; n_probes=32
#       multiplies RESIDENT-LAYER compute (backward per probe), not streams;
#       per-layer GPTQ/JSO renders are resident-layer compute.
#   => ~2 body streams ≈ 600 GiB total read, same O(1)-streams class as the
#   forked expert driver. No per-unit model restream exists in this path.
#
# SMOKE POLICY (deliberate deviation from "separate 1-layer smoke first")
#   The reverse sweep is structurally full-model: every layer's backward must
#   run to roll cotangents down, so a --unit-filter smoke still pays both
#   body streams (~70% of the full run) and verifies nothing the full run's
#   FIRST rendered layer does not. The reverse pass renders layer 44 first
#   and per-unit checkpoints land immediately, so the full launch IS the
#   smoke: watch boundary-capture + the first reverse-layer line, verify the
#   first unit checkpoints, kill+fix+--resume if wrong. Nothing is re-paid on
#   resume (identity-validated per-unit shards).
#
# ARGPARSE / IDENTITY PINS
#   --dataset REQUIRED by the CLI itself (loader default is a Hub stream).
#   --n-calib-samples 8 --calib-seqlen 512 --calib-seed 42: the PROBE's
#     calibration (probe_census.json meta), asserted by hash before any GPU
#     work — the activation rows were captured under it.
#   --n-probes 32: pipeline AURA default (run-pipeline.sh).
#   --max-act-rows 256: the probe's --activation-rows-limit.
#   MODEL is sparklina-LOCAL (/home/rob/models/GLM-5.3-Flash), never the NFS
#     mirror: the harvest streams the body twice (P7).
#
# ENVELOPE (proven by the 08-26 expert run on this box)
#   CACHE_HEADROOM_GB=90, PREFETCH_WORKERS=1, TMPDIR under the campaign dir,
#   HF/TRANSFORMERS offline, py-spy record --rate 5 --idle (blocking; 100 Hz
#   fell behind, --nonblocking recorded nothing), systemd-run --user unit.
# =============================================================================
set -euo pipefail

REMOTE="sparklina"
CAMPAIGN="/home/rob/dq-runs/glm53-flash"
REMOTE_REPO="/home/rob/prismaquant-glm53-gate"
REMOTE_LIB="${CAMPAIGN}/lib"
VENV="/home/rob/dq-runs/venvs/prismaquant-tf516"
MODEL="/home/rob/models/GLM-5.3-Flash"
DATASET="/home/rob/dq-runs/calibration/diverse-v1.jsonl"
ARTIFACTS="${CAMPAIGN}/work/artifacts"
ACT_DIR="${CAMPAIGN}/act"
STOCK_DIR="${CAMPAIGN}/stock_anchored"
PROBE_CENSUS="${STOCK_DIR}/probe_census.json"
CHECKPOINT_CENSUS="${STOCK_DIR}/checkpoint_census.json"
EXPERT_PKL="${ARTIFACTS}/cost_expert_empirical.pkl"
HARVEST_PKL="${ARTIFACTS}/harvest_stock_anchor.pkl"
COST_PKL="${ARTIFACTS}/cost_stock_anchored.pkl"
CKPT_DIR="${CAMPAIGN}/work/ckpt-aura/stock-anchor-harvest"
LOG="${CAMPAIGN}/stock_harvest_sparklina.log"
PROFILE_OUT="${CAMPAIGN}/stock_harvest_profile.speedscope"
UNIT="glm53-stock-harvest"
EXPERT_UNIT="glm53-expert-cost"

N_CALIB_SAMPLES="8"
CALIB_SEQLEN="512"
CALIB_SEED="42"
N_PROBES="32"
MAX_ACT_ROWS="256"

PREFLIGHT_ONLY=0
[[ "${1:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=1

say() { echo; echo "=== $*"; }

# ---------------------------------------------------------------- local
say "local preconditions"
cd /home/rob/prismaquant
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain | wc -l)"
echo "source ${BRANCH} ${COMMIT} (${DIRTY} dirty entries)"
for f in "$PROBE_CENSUS" "$CHECKPOINT_CENSUS"; do
  [[ -f "$f" ]] || { echo "missing census: $f" >&2; exit 3; }
done

# ---------------------------------------------------------------- sync
say "sync -> ${REMOTE}"
for u in "$UNIT" "$EXPERT_UNIT" glm53-probe.service; do
  if ssh "$REMOTE" "systemctl --user is-active --quiet '$u'"; then
    echo "$u is ACTIVE on ${REMOTE}; refusing to mutate its PYTHONPATH tree." >&2
    exit 3
  fi
done
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.venv' --exclude 'work' --exclude 'exported' \
  /home/rob/prismaquant/ "${REMOTE}:${REMOTE_REPO}/"
ssh "$REMOTE" "mkdir -p '${STOCK_DIR}'"
rsync -a "$PROBE_CENSUS" "$CHECKPOINT_CENSUS" "${REMOTE}:${STOCK_DIR}/"

# ---------------------------------------------------------------- preflight
say "CLI preflight on ${REMOTE} (import + parse + plan + skeleton gate, no GPU)"
ssh "$REMOTE" "CUDA_VISIBLE_DEVICES='' PYTHONPATH='${REMOTE_REPO}' \
  LD_LIBRARY_PATH='${REMOTE_LIB}' TMPDIR='${CAMPAIGN}/tmp' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  '${VENV}/bin/python' - <<'PY'
from pathlib import Path

import prismaquant.glm53_stock_harvest as h
from prismaquant.glm53_stock_reprice import (
    PROBE_CENSUS_SCHEMA, CHECKPOINT_CENSUS_SCHEMA, load_census,
)

print(f\"  module                 {h.__file__}\")
for label, path in (
    ('model', '${MODEL}'),
    ('calibration', '${DATASET}'),
    ('activation cache', '${ACT_DIR}'),
    ('probe census', '${PROBE_CENSUS}'),
    ('checkpoint census', '${CHECKPOINT_CENSUS}'),
    ('expert pkl', '${EXPERT_PKL}'),
):
    p = Path(path)
    assert p.exists(), f'missing {label}: {path}'
    print(f\"  {label:22s} present\")

argv = [
    '--model', '${MODEL}',
    '--probe-census', '${PROBE_CENSUS}',
    '--checkpoint-census', '${CHECKPOINT_CENSUS}',
    '--activation-cache-dir', '${ACT_DIR}',
    '--dataset', '${DATASET}',
    '--n-calib-samples', '${N_CALIB_SAMPLES}',
    '--calib-seqlen', '${CALIB_SEQLEN}',
    '--calib-seed', '${CALIB_SEED}',
    '--n-probes', '${N_PROBES}',
    '--max-act-rows', '${MAX_ACT_ROWS}',
    '--checkpoint-dir', '${CKPT_DIR}',
    '--resume',
    '--output', '${HARVEST_PKL}',
]
args = h._build_parser().parse_args(argv)
print('  parsed OK')

pc = load_census(args.probe_census, schema=PROBE_CENSUS_SCHEMA)
cc = load_census(args.checkpoint_census, schema=CHECKPOINT_CENSUS_SCHEMA)
planned = h.build_dense_plan(pc, cc)
assert planned['plan_scope'] == 'full'
assert len(planned['plan']) == 135, len(planned['plan'])
assert len(planned['packed_expert_units_excluded']) == 84
h._precheck_activation_files(Path(args.activation_cache_dir), sorted(planned['plan']))
print(f\"  plan                   {len(planned['plan'])} dense units, act files present\")
print(f\"  probe calib_hash       {pc['meta']['calib_hash']}\")

# Skeleton-construction gate (blocker-3 class, fix-agnostic).
from transformers import AutoConfig, AutoModelForCausalLM
from prismaquant.build_rtn_cache import stage_multimodal as _stage
from prismaquant.streaming_model import _skeleton_config_and_class

_staged, _ = _stage('${MODEL}')
_cfg = AutoConfig.from_pretrained(_staged, trust_remote_code=True)
_c, _cls = _skeleton_config_and_class(_cfg, multimodal=True)
print(f\"  staged path            {'RAW (passed through)' if _staged == '${MODEL}' else _staged}\")
print(f\"  skeleton model_cls     {getattr(_cls, '__name__', _cls)}\")
if _cls is AutoModelForCausalLM:
    raise SystemExit('  BLOCKED: skeleton falls back to AutoModelForCausalLM')
print('PREFLIGHT OK')
PY"

if (( PREFLIGHT_ONLY )); then
  say "preflight only: stopping before launch"
  exit 0
fi

# ---------------------------------------------------------------- gates
say "launch gates"
if ssh "$REMOTE" "test -f '${COST_PKL}'"; then
  echo "  ${COST_PKL} exists; nothing to do (delete it to re-price)."
  exit 0
fi
ssh "$REMOTE" "mkdir -p '${ARTIFACTS}' '${CKPT_DIR}' '${CAMPAIGN}/tmp'"
echo "  df: $(ssh "$REMOTE" "df -h /home/rob | tail -1")"

# ---------------------------------------------------------------- payload
say "writing payload"
PAYLOAD_LOCAL="${CAMPAIGN}/stock_harvest_payload.sh"
PAYLOAD_REMOTE="${CAMPAIGN}/stock_harvest_payload.sh"
mkdir -p "${CAMPAIGN}"
cat > "$PAYLOAD_LOCAL" <<EOF
#!/usr/bin/env bash
# GENERATED by run_glm53_stock_harvest.sh — do not edit by hand.
set -euo pipefail

echo "[stock-harvest] start \$(date -Is)"
# GPU-free gate: the harvest owns the box for hours; refuse to stack.
GPU_DEADLINE=\$(( \$(date +%s) + 1800 ))
while true; do
  BUSY="\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  if [[ "\${BUSY}" == "0" ]]; then
    echo "[stock-harvest] GPU free at \$(date -Is)"
    break
  fi
  if (( \$(date +%s) >= GPU_DEADLINE )); then
    echo "[stock-harvest] GPU still busy after 1800s; REFUSING" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 3
  fi
  echo "[stock-harvest] GPU busy (procs=\${BUSY}); waiting"
  sleep 60
done

echo "[stock-harvest] df: \$(df -h /home/rob | tail -1)"
echo "[stock-harvest] harvest under py-spy (rate 5, blocking)"
set -x
'${VENV}/bin/py-spy' record -f speedscope -o '${PROFILE_OUT}' \\
  --rate 5 --idle -- \\
  '${VENV}/bin/python' -u -m prismaquant.glm53_stock_harvest \\
    --model '${MODEL}' \\
    --probe-census '${PROBE_CENSUS}' \\
    --checkpoint-census '${CHECKPOINT_CENSUS}' \\
    --activation-cache-dir '${ACT_DIR}' \\
    --dataset '${DATASET}' \\
    --n-calib-samples '${N_CALIB_SAMPLES}' \\
    --calib-seqlen '${CALIB_SEQLEN}' \\
    --calib-seed '${CALIB_SEED}' \\
    --n-probes '${N_PROBES}' \\
    --max-act-rows '${MAX_ACT_ROWS}' \\
    --checkpoint-dir '${CKPT_DIR}' \\
    --resume \\
    --output '${HARVEST_PKL}' \\
    --device cuda
set +x

# py-spy 0.4.2 exits 0 even when the profiled child fails, so set -e does not
# stop a dead harvest from reaching the pricing step (observed 2026-08-27:
# the renderer crash fell through to a FileNotFoundError here). The artifact
# is the gate.
if [[ ! -f '${HARVEST_PKL}' ]]; then
  echo "[stock-harvest] FAILED: harvest wrapper ${HARVEST_PKL} was not written" >&2
  exit 4
fi

echo "[stock-harvest] campaign pricing (CPU)"
set -x
CUDA_VISIBLE_DEVICES='' '${VENV}/bin/python' -u -m prismaquant.glm53_stock_reprice campaign \\
  --probe-census '${PROBE_CENSUS}' \\
  --checkpoint-census '${CHECKPOINT_CENSUS}' \\
  --harvest '${HARVEST_PKL}' \\
  --expert-empirical '${EXPERT_PKL}' \\
  --output '${COST_PKL}'
set +x
echo "[stock-harvest] done \$(date -Is)"
EOF
bash -n "$PAYLOAD_LOCAL"
rsync -a "$PAYLOAD_LOCAL" "${REMOTE}:${PAYLOAD_REMOTE}"
ssh "$REMOTE" "chmod +x '${PAYLOAD_REMOTE}'"
echo "payload -> ${REMOTE}:${PAYLOAD_REMOTE}"

# ---------------------------------------------------------------- launch
say "launching ${UNIT} on ${REMOTE}"
# A prior FAILED transient unit holds the name (no --collect, so Result stays
# readable); clear it now that its state has been read.
ssh "$REMOTE" "systemctl --user reset-failed '${UNIT}' 2>/dev/null || true"
ssh "$REMOTE" "systemd-run --user --unit='${UNIT}' \
  --setenv=PYTHONPATH='${REMOTE_REPO}' \
  --setenv=LD_LIBRARY_PATH='${REMOTE_LIB}' \
  --setenv=TMPDIR='${CAMPAIGN}/tmp' \
  --setenv=HF_HUB_OFFLINE=1 \
  --setenv=TRANSFORMERS_OFFLINE=1 \
  --setenv=CACHE_HEADROOM_GB=90 \
  --setenv=PREFETCH_WORKERS=1 \
  --setenv=PRISMAQUANT_IDENTITY_GIT_COMMIT='${COMMIT}' \
  --working-directory='${REMOTE_REPO}' \
  bash -c \"set -o pipefail; '${PAYLOAD_REMOTE}' 2>&1 | tee '${LOG}'\""

sleep 5
ssh "$REMOTE" "systemctl --user show '${UNIT}' -p ActiveState -p SubState -p ExecMainPID --value | tr '\n' ' '; echo"
echo
echo "log:     ${REMOTE}:${LOG}"
echo "profile: ${REMOTE}:${PROFILE_OUT}"
echo "harvest: ${REMOTE}:${HARVEST_PKL}"
echo "cost:    ${REMOTE}:${COST_PKL}"
