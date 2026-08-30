#!/usr/bin/env bash
# Exact-pin, one-Spark Gridbook CB load/generation gate.
#
# The filename is historical: this launcher was written for DSv4 and is now
# lane-generic.  The serving LANE is derived from the artifact's own
# config.json (`prismaquant.shipcard.cb_serving_lane`) and selects exactly
# three things -- the container image, the tokenizer mode, and the
# served-model brand.  Every other pin here (vLLM version/commit, GPU, the
# 8192/1-seq/1 GiB-KV/0.90-util gate parameters, the environment profile) was
# verified lane-invariant and stays shared.
#
# Run each arm separately; each invocation creates a fresh container and
# installs Gridbook from the immutable PrismaQuant runtime pin inside it:
#
#   MODEL=/abs/path/to/artifact scripts/serve_dsv4_cb_validate.sh eager
#   MODEL=/abs/path/to/artifact scripts/serve_dsv4_cb_validate.sh graph
#
# Each arm closes its matching native_export slot.  The eager arm also runs the
# numeric catastrophic-quality gate against that same live nonce-bound server
# and closes ship_gate.  Neither arm claims gold quality or throughput, and the
# launcher deliberately exposes no speculative-decode option.
set -euo pipefail

ARM=${1:-}
case "$ARM" in
  eager|graph) ;;
  *) echo "usage: MODEL=/absolute/artifact $0 {eager|graph}" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
MODEL=${MODEL:-}
if [[ -z "$MODEL" || "$MODEL" != /* || ! -d "$MODEL" ]]; then
  echo "REFUSE: MODEL must be an existing absolute artifact directory" >&2
  exit 2
fi
MODEL=$(cd -- "$MODEL" && pwd -P)
SHIPCARD=${SHIPCARD:-$MODEL/shipcard.json}
if [[ ! -f "$MODEL/config.json" || ! -f "$MODEL/quant_config.json" \
      || ! -f "$SHIPCARD" ]]; then
  echo "REFUSE: MODEL must contain config.json, quant_config.json, and the requested shipcard" >&2
  exit 2
fi
if ! compgen -G "$MODEL/*.safetensors" >/dev/null; then
  echo "REFUSE: MODEL has no safetensors checkpoint" >&2
  exit 2
fi

# The artifact's build receipt selects the PrismaQuant source commit the SERVED
# STACK runs at; the live checkout selects the commit the GATE runs at.  Those
# were one commit until 2026-08-16, and holding them equal made a validator bug
# structurally incurable -- the only remedy the gate admitted for a wrong
# verdict was rebuilding bytes that were never wrong.  They are now split:
#
#   runtime  = artifact build commit.  Mounted at /repo, sources the Gridbook
#              serving runtime, executes every in-container step.  The stack
#              that decodes the bytes is still exactly the stack they were made
#              for; nothing about the serve is relaxed.
#   judge    = the live checkout's HEAD.  Runs this launcher and every
#              host-side verdict, so a gate fix reaches artifacts already on
#              disk.  It must be CLEAN and a DESCENDANT of the build commit --
#              forward only, never judging with code older than the producer.
#
# The split is not taken on trust.  Both snapshots are materialized and
# verified, and `judge-divergence` proves every closure path that differs
# between them is judge-only.  A divergence anywhere in the serve path refuses,
# and re-export is the honest answer there.
#
# A live checkout is only a bootstrap: this script re-executes from a complete
# content-addressed snapshot before sourcing helpers or importing PrismaQuant.
PQ_SERVE_SNAPSHOT_REEXEC=${PQ_SERVE_SNAPSHOT_REEXEC:-0}
PQ_SNAPSHOT_CACHE=${PQ_SNAPSHOT_CACHE:-$(dirname -- "$MODEL")/runtime-source-cache}
if [[ "$PQ_SERVE_SNAPSHOT_REEXEC" != 1 ]]; then
  PQ_SNAPSHOT_CACHE=$(realpath -m -- "$PQ_SNAPSHOT_CACHE")
  case "$PQ_SNAPSHOT_CACHE/" in
    "$MODEL/"*)
      echo "REFUSE: PQ_SNAPSHOT_CACHE must be outside the immutable artifact" >&2
      exit 2
      ;;
  esac
  readarray -t build_git < <(env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    PYTHONSAFEPATH=1 python3 -P - "$SHIPCARD" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    card = json.load(handle)
git = (card.get("build") or {}).get("git") or {}
commit = git.get("commit")
dirty = git.get("dirty")
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit("shipcard build.git.commit is not one full Git commit")
if dirty is not False:
    raise SystemExit("shipcard build.git.dirty is not false")
print(commit)
PY
  )
  ARTIFACT_BUILD_COMMIT=${build_git[0]:-}
  if [[ ! "$ARTIFACT_BUILD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "REFUSE: artifact lacks one clean full PrismaQuant build commit" >&2
    exit 2
  fi
  REPO_HEAD=$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true)
  REPO_DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=all)
  if [[ ! "$REPO_HEAD" =~ ^[0-9a-f]{40}$ || -n "$REPO_DIRTY" ]]; then
    echo "REFUSE: serve checkout must be a clean Git checkout" >&2
    exit 2
  fi
  # Forward only.  A descendant contains the producer's own code, so the judge
  # can never be older than the artifact it is judging.
  if ! git -C "$REPO" merge-base --is-ancestor \
       "$ARTIFACT_BUILD_COMMIT" "$REPO_HEAD" 2>/dev/null; then
    echo "REFUSE: serve checkout $REPO_HEAD is not a descendant of the artifact build commit $ARTIFACT_BUILD_COMMIT" >&2
    exit 2
  fi
  pq_materialize_snapshot() {  # $1=commit; emits snapshot, tree, closure
    local snapshot_json
    snapshot_json=$(env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
      python3 -P "$REPO/tools/prismaquant_runtime_snapshot.py" \
      materialize --source-root "$REPO" --cache-root "$PQ_SNAPSHOT_CACHE" \
      --commit "$1")
    env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 python3 -P -c '
import json, sys
value = json.load(sys.stdin)
for key in ("snapshot", "tree", "closure_sha256"):
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise SystemExit(f"runtime snapshot lacks {key}")
    print(item)
' <<<"$snapshot_json"
  }
  readarray -t snapshot_fields < <(pq_materialize_snapshot "$ARTIFACT_BUILD_COMMIT")
  PQ_RUNTIME_SNAPSHOT=${snapshot_fields[0]:-}
  PQ_RUNTIME_TREE=${snapshot_fields[1]:-}
  PQ_RUNTIME_CLOSURE_SHA256=${snapshot_fields[2]:-}
  readarray -t judge_fields < <(pq_materialize_snapshot "$REPO_HEAD")
  PQ_JUDGE_SNAPSHOT=${judge_fields[0]:-}
  PQ_JUDGE_TREE=${judge_fields[1]:-}
  PQ_JUDGE_CLOSURE_SHA256=${judge_fields[2]:-}
  PQ_JUDGE_COMMIT=$REPO_HEAD
  if [[ ! -d "$PQ_RUNTIME_SNAPSHOT" \
        || ! "$PQ_RUNTIME_TREE" =~ ^[0-9a-f]{40}$ \
        || ! "$PQ_RUNTIME_CLOSURE_SHA256" =~ ^[0-9a-f]{64}$ \
        || ! -d "$PQ_JUDGE_SNAPSHOT" \
        || ! "$PQ_JUDGE_TREE" =~ ^[0-9a-f]{40}$ \
        || ! "$PQ_JUDGE_CLOSURE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "REFUSE: reviewed PrismaQuant snapshot identity is malformed" >&2
    exit 2
  fi
  env -u PYTHONPATH python3 -P \
    "$PQ_RUNTIME_SNAPSHOT/tools/prismaquant_runtime_snapshot.py" verify \
    --snapshot "$PQ_RUNTIME_SNAPSHOT" \
    --expected-commit "$ARTIFACT_BUILD_COMMIT" \
    --expected-tree "$PQ_RUNTIME_TREE" \
    --expected-closure-sha256 "$PQ_RUNTIME_CLOSURE_SHA256" >/dev/null
  env -u PYTHONPATH python3 -P \
    "$PQ_JUDGE_SNAPSHOT/tools/prismaquant_runtime_snapshot.py" verify \
    --snapshot "$PQ_JUDGE_SNAPSHOT" \
    --expected-commit "$PQ_JUDGE_COMMIT" \
    --expected-tree "$PQ_JUDGE_TREE" \
    --expected-closure-sha256 "$PQ_JUDGE_CLOSURE_SHA256" >/dev/null
  # The split's proof obligation, run from the JUDGE snapshot: the newer
  # revision states which of its own divergences it claims are judge-only, and
  # anything in the serve path refuses here rather than at the receipt.
  env -u PYTHONPATH python3 -P \
    "$PQ_JUDGE_SNAPSHOT/tools/prismaquant_runtime_snapshot.py" judge-divergence \
    --runtime-snapshot "$PQ_RUNTIME_SNAPSHOT" \
    --judge-snapshot "$PQ_JUDGE_SNAPSHOT" >/dev/null
  if [[ $(git -C "$REPO" rev-parse --verify 'HEAD^{commit}') != "$PQ_JUDGE_COMMIT" \
        || -n $(git -C "$REPO" status --porcelain --untracked-files=all) ]]; then
    echo "REFUSE: serve checkout changed while snapshotting" >&2
    exit 2
  fi
  export PQ_SERVE_SNAPSHOT_REEXEC=1
  export PQ_RUNTIME_SNAPSHOT ARTIFACT_BUILD_COMMIT PQ_RUNTIME_TREE
  export PQ_RUNTIME_CLOSURE_SHA256
  export PQ_JUDGE_SNAPSHOT PQ_JUDGE_COMMIT PQ_JUDGE_TREE
  export PQ_JUDGE_CLOSURE_SHA256
  # The source-bootstrap contract: this names the root HOST-SIDE Python must
  # import from, and `activate_prismaquant_source` refuses unless it equals the
  # directory of the bootstrap tool being run.  Host-side runs are the judge, so
  # this is the judge root.  The container gets its own explicit `/repo`, which
  # is the runtime snapshot -- see the `docker create` environment below.
  export PQ_RUNTIME_PRISMAQUANT_ROOT=$PQ_JUDGE_SNAPSHOT
  # The artifact's identity is still the artifact's build commit -- the judge
  # running newer does not restamp what produced these bytes.
  export PRISMAQUANT_IDENTITY_GIT_COMMIT=$ARTIFACT_BUILD_COMMIT
  export PRISMAQUANT_IDENTITY_GIT_DIRTY=0
  export PYTHONSAFEPATH=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  exec bash "$PQ_JUDGE_SNAPSHOT/scripts/serve_dsv4_cb_validate.sh" "$ARM"
fi

if [[ ! "$ARTIFACT_BUILD_COMMIT" =~ ^[0-9a-f]{40}$ \
      || ! "$PQ_RUNTIME_TREE" =~ ^[0-9a-f]{40}$ \
      || ! "$PQ_RUNTIME_CLOSURE_SHA256" =~ ^[0-9a-f]{64}$ \
      || ! "${PQ_JUDGE_COMMIT:-}" =~ ^[0-9a-f]{40}$ \
      || ! "${PQ_JUDGE_TREE:-}" =~ ^[0-9a-f]{40}$ \
      || ! "${PQ_JUDGE_CLOSURE_SHA256:-}" =~ ^[0-9a-f]{64}$ \
      || "$REPO" != "${PQ_JUDGE_SNAPSHOT:-}" \
      || ! -d "$PQ_RUNTIME_SNAPSHOT" \
      || "${PQ_RUNTIME_PRISMAQUANT_ROOT:-}" != "$REPO" \
      || "${PYTHONSAFEPATH:-}" != 1 \
      || "${PYTHONDONTWRITEBYTECODE:-}" != 1 \
      || "${PYTHONNOUSERSITE:-}" != 1 \
      || -n "${PYTHONPATH+x}" ]]; then
  echo "REFUSE: serve launcher is not executing from its attested snapshot" >&2
  exit 2
fi
# Both halves are re-verified at every checkpoint, and so is the claim that
# lets them differ.  Re-proving the divergence each time is deliberate: the
# judge-only list is the whole justification for the split, so it may not be
# checked once at bootstrap and then trusted for the rest of a long serve.
verify_runtime_snapshot() {
  env -u PYTHONPATH python3 -P \
    "$PQ_RUNTIME_SNAPSHOT/tools/prismaquant_runtime_snapshot.py" verify \
    --snapshot "$PQ_RUNTIME_SNAPSHOT" \
    --expected-commit "$ARTIFACT_BUILD_COMMIT" \
    --expected-tree "$PQ_RUNTIME_TREE" \
    --expected-closure-sha256 "$PQ_RUNTIME_CLOSURE_SHA256" >/dev/null
  env -u PYTHONPATH python3 -P \
    "$REPO/tools/prismaquant_runtime_snapshot.py" verify \
    --snapshot "$REPO" \
    --expected-commit "$PQ_JUDGE_COMMIT" \
    --expected-tree "$PQ_JUDGE_TREE" \
    --expected-closure-sha256 "$PQ_JUDGE_CLOSURE_SHA256" >/dev/null
  if [[ "$PQ_JUDGE_SNAPSHOT" != "$PQ_RUNTIME_SNAPSHOT" ]]; then
    env -u PYTHONPATH python3 -P \
      "$REPO/tools/prismaquant_runtime_snapshot.py" judge-divergence \
      --runtime-snapshot "$PQ_RUNTIME_SNAPSHOT" \
      --judge-snapshot "$REPO" >/dev/null
  fi
}
pq_run_module() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -P \
    "$REPO/tools/prismaquant_source_bootstrap.py" run-module "$@"
}
verify_runtime_snapshot
readarray -t inner_build_git < <(env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  python3 -P - "$SHIPCARD" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    git = (json.load(handle).get("build") or {}).get("git") or {}
print(git.get("commit", ""))
print("false" if git.get("dirty") is False else "not-false")
PY
)
if [[ "${inner_build_git[0]:-}" != "$ARTIFACT_BUILD_COMMIT" \
      || "${inner_build_git[1]:-}" != false ]]; then
  echo "REFUSE: artifact build identity changed across snapshot re-exec" >&2
  exit 2
fi
# Serving runtime, so it comes from the artifact's build commit -- this decides
# the Gridbook wheel the container installs.
. "$PQ_RUNTIME_SNAPSHOT/prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
gridbook_serving_runtime_prepare

# The lane, its image, its tokenizer mode and its served-model brand come from
# the SAME table the validator replays against (`CB_SERVING_LANE_SPECS`), so a
# serve this launcher can produce is exactly a serve the gate can accept.
# Reading them here rather than restating them is what stopped the two sides
# from drifting apart in the first place.
readarray -t lane_fields < <(env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 python3 -P - "$MODEL" "$REPO" <<'PY'
import sys

# The attested snapshot, never the cwd: there is more than one PrismaQuant
# checkout on this box and safe-path mode deliberately drops the implicit one.
sys.path.insert(0, sys.argv[2])
from prismaquant.shipcard import cb_serving_lane
from prismaquant.validate_cb_endpoint import cb_lane_spec

lane = cb_serving_lane(sys.argv[1])
spec = cb_lane_spec(lane)
print(lane)
print(spec["image"])
print(spec["tokenizer_mode"])
print(spec["served_model_prefix"])
PY
)
SERVE_LANE=${lane_fields[0]:-}
BASE_IMAGE=${lane_fields[1]:-}
TOKENIZER_MODE=${lane_fields[2]:-}
SERVED_MODEL_PREFIX=${lane_fields[3]:-}
if [[ -z "$SERVE_LANE" || -z "$BASE_IMAGE" || -z "$TOKENIZER_MODE" \
      || -z "$SERVED_MODEL_PREFIX" ]]; then
  echo "REFUSE: could not resolve the artifact's CB serving lane" >&2
  exit 2
fi
# Lane-invariant: the Gridbook image is built FROM the eugr Spark base and
# reports this identical vLLM version/commit.
VLLM_VERSION=0.26.1rc1.dev693+g7f7a32cfe.d20260812
VLLM_COMMIT=7f7a32cfec0f1bc5b73c37200b86631523a1ea8f
GRAPH_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
if [[ -n "${SERVED_MODEL:-}" ]]; then
  echo "REFUSE: the exact CB gate owns the per-run served-model nonce" >&2
  exit 2
fi
if command -v openssl >/dev/null 2>&1; then
  SERVE_NONCE=$(openssl rand -hex 16)
else
  SERVE_NONCE=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
fi
if [[ ! "$SERVE_NONCE" =~ ^[0-9a-f]{32}$ ]]; then
  echo "REFUSE: could not generate a 128-bit served-model session nonce" >&2
  exit 2
fi
SERVED_MODEL=${SERVED_MODEL_PREFIX}${SERVE_NONCE}
PORT=${PORT:-8000}
EXPECTED_GPU_LOCK=/home/rob/dq-runs/gpu.lock
LOCK=${GPU_LOCK:-$EXPECTED_GPU_LOCK}
START_FLOOR_GIB=${START_FLOOR_GIB:-110}
READY_FLOOR_GIB=${READY_FLOOR_GIB:-8}
WATCHDOG_GIB=${WATCHDOG_GIB:-4}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-1800}
RUN_ROOT=${RUN_ROOT:-$(dirname -- "$MODEL")/cb-serve-validation}
EVIDENCE=${EVIDENCE:-$RUN_ROOT/$ARM}
EXT_CACHE_ROOT=${EXT_CACHE_ROOT:-/home/rob/dq-runs/gridbook-ext-cache}
EXT_CACHE=$EXT_CACHE_ROOT/${GRIDBOOK_RUNTIME_COMMIT}-eugr-${VLLM_COMMIT}-${ARM}
NAME=${NAME:-pq-dsv4-cb-${ARM}}

mkdir -p -- "$EVIDENCE" "$EXT_CACHE" "$(dirname -- "$LOCK")"
EVIDENCE=$(cd -- "$EVIDENCE" && pwd -P)
EXT_CACHE=$(cd -- "$EXT_CACHE" && pwd -P)
case "$EVIDENCE/" in
  "$MODEL/"*) echo "REFUSE: EVIDENCE must be outside the immutable MODEL tree" >&2; exit 2 ;;
esac
case "$EXT_CACHE/" in
  "$MODEL/"*) echo "REFUSE: EXT_CACHE must be outside the immutable MODEL tree" >&2; exit 2 ;;
esac
WATCHDOG_LOG=$EVIDENCE/memory-watchdog.log
SERVE_LOG=$EVIDENCE/serve.log
CAPTURE_LOG=$EVIDENCE/capture-evidence.log
MANIFEST=$EVIDENCE/serve_manifest.json
RESULT=$EVIDENCE/validation.json
SHIP_GATE_REPORT=$EVIDENCE/ship-gate-report.md

for value in "$START_FLOOR_GIB" "$READY_FLOOR_GIB" "$WATCHDOG_GIB"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "REFUSE: memory floors must be whole GiB values" >&2
    exit 2
  fi
done
if (( START_FLOOR_GIB < 110 || READY_FLOOR_GIB < 8 || WATCHDOG_GIB < 4 \
      || READY_FLOOR_GIB <= WATCHDOG_GIB )); then
  echo "REFUSE: safety floors require start>=110, ready>=8, watchdog>=4, ready>watchdog GiB" >&2
  exit 2
fi
if [[ "$LOCK" != "$EXPECTED_GPU_LOCK" ]]; then
  echo "REFUSE: exact DSv4 gate requires GPU_LOCK=$EXPECTED_GPU_LOCK" >&2
  exit 2
fi

# Which code produced these bytes and which code judged them are now two
# answers, so the evidence carries both rather than leaving a reader to assume
# they are one.  Written before the GPU is reserved: a receipt that cannot say
# who rendered its verdict is not a receipt.
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 python3 -P - \
    "$REPO" "$PQ_RUNTIME_SNAPSHOT" "$EVIDENCE/judge_split.json" <<'PY'
import json
import sys
from pathlib import Path

judge_root, runtime_root, out = sys.argv[1:]
sys.path.insert(0, str(Path(judge_root) / "tools"))
from prismaquant_runtime_snapshot import (
    MANIFEST,
    _load_manifest,
    _validate_manifest_shape,
    judge_divergence,
)

# Read the ledgers rather than re-verify. `verify_runtime_snapshot` re-hashes
# both closures at the top of this pass and again in the preflight just below,
# so another full pass over every file here buys nothing but wall-clock.
def _identity(root):
    return _validate_manifest_shape(_load_manifest(Path(root) / MANIFEST))

runtime = _identity(runtime_root)
judge = _identity(judge_root)
payload = {
    "schema": "prismaquant.serve_gate_judge_split.v1",
    "runtime": {
        "commit": runtime["commit"],
        "tree": runtime["tree"],
        "closure_sha256": runtime["closure_sha256"],
        "role": "served stack; mounted at /repo; the artifact's build commit",
    },
    "judge": {
        "commit": judge["commit"],
        "tree": judge["tree"],
        "closure_sha256": judge["closure_sha256"],
        "role": "host-side launcher and verdict",
    },
    "split": (
        judge_divergence(Path(runtime_root), Path(judge_root))
        if runtime["commit"] != judge["commit"]
        else {"schema": "prismaquant.judge_split_divergence.v1",
              "divergent_paths": [], "divergent_count": 0,
              "judge_only_paths": []}
    ),
}
Path(out).write_text(
    json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY
test -s "$EVIDENCE/judge_split.json"

# Header-only and sidecar-only preflight.  Refuse an incomplete decoder map or
# a missing/mismatched released DSpark overlay before reserving the single GPU.
verify_runtime_snapshot
env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -P - "$REPO" "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv.pop(1)) / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

from prismaquant.validate_cb_endpoint import (
    validate_cb_artifact,
    validate_cb_artifact_decode_contract,
)

artifact = sys.argv[1]
quant_config = validate_cb_artifact(artifact)
evidence = validate_cb_artifact_decode_contract(artifact, quant_config)
print(json.dumps(evidence, sort_keys=True))
PY
verify_runtime_snapshot

exec 9>"$LOCK"
if ! flock -n -x 9; then
  echo "REFUSE: GPU lock is held: $LOCK" >&2
  exit 75
fi
: > "$WATCHDOG_LOG"
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "REFUSE: container name already exists: $NAME" >&2
  exit 2
fi
if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "REFUSE: TCP port $PORT is already listening" >&2
  exit 2
fi
docker image inspect "$BASE_IMAGE" >/dev/null

available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
start_floor_kib=$((START_FLOOR_GIB * 1048576))
if (( available_kib < start_floor_kib )); then
  echo "REFUSE: MemAvailable ${available_kib} KiB < start floor ${start_floor_kib} KiB" >&2
  exit 3
fi

{
  date --iso-8601=seconds
  echo "ARM=$ARM"
  echo "MODEL=$MODEL"
  echo "ARTIFACT_BUILD_COMMIT=$ARTIFACT_BUILD_COMMIT"
  echo "PQ_RUNTIME_CLOSURE_SHA256=$PQ_RUNTIME_CLOSURE_SHA256"
  echo "PQ_JUDGE_COMMIT=$PQ_JUDGE_COMMIT"
  echo "PQ_JUDGE_CLOSURE_SHA256=$PQ_JUDGE_CLOSURE_SHA256"
  echo "BASE_IMAGE=$BASE_IMAGE"
  echo "VLLM_VERSION=$VLLM_VERSION"
  echo "VLLM_COMMIT=$VLLM_COMMIT"
  echo "GRIDBOOK_RUNTIME_COMMIT=$GRIDBOOK_RUNTIME_COMMIT"
  echo "GRIDBOOK_RUNTIME_VERSION=$GRIDBOOK_RUNTIME_VERSION"
  echo "GRIDBOOK_RUNTIME_WHEEL_SHA256=$GRIDBOOK_RUNTIME_WHEEL_SHA256"
  echo "MEM_AVAILABLE_KIB=$available_kib"
  docker image inspect "$BASE_IMAGE"
  nvidia-smi
  free -h
} > "$EVIDENCE/prelaunch.log" 2>&1

# A fresh ephemeral container is load-bearing here.  Reusing one container for
# both arms can make graph pass on extensions or allocator state loaded by the
# eager arm, and no longer proves either exact install independently.
CID=$(docker create --pull=never --name "$NAME" --gpus all --ipc=host \
  --user 0:0 --oom-score-adj=1000 \
  --log-driver local --log-opt max-size=100m --log-opt max-file=3 \
  -p "127.0.0.1:$PORT:8000" \
  -v "$PQ_RUNTIME_SNAPSHOT:/repo:ro" \
  -v "$MODEL:/model:ro" \
  -v "$EVIDENCE:/evidence:rw" \
  -v "$EXT_CACHE:/opt/gridbook/ext-cache:rw" \
  "${GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS[@]}" \
  -e "PQ_RUNTIME_PRISMAQUANT_ROOT=/repo" \
  -e "PQ_RUNTIME_PRISMAQUANT_GIT_COMMIT=$ARTIFACT_BUILD_COMMIT" \
  -e "PQ_RUNTIME_PRISMAQUANT_TREE=$PQ_RUNTIME_TREE" \
  -e "PQ_RUNTIME_PRISMAQUANT_CLOSURE_SHA256=$PQ_RUNTIME_CLOSURE_SHA256" \
  -e "PRISMAQUANT_IDENTITY_GIT_COMMIT=$ARTIFACT_BUILD_COMMIT" \
  -e "PRISMAQUANT_IDENTITY_GIT_DIRTY=0" \
  -e "PYTHONDONTWRITEBYTECODE=1" \
  -e "PYTHONNOUSERSITE=1" \
  -e "PQ_ARM=$ARM" \
  -e "PQ_SERVED_MODEL=$SERVED_MODEL" \
  -e "PQ_TOKENIZER_MODE=$TOKENIZER_MODE" \
  -e "PQ_VLLM_VERSION=$VLLM_VERSION" \
  -e "PQ_GRAPH_COMPILATION_CONFIG=$GRAPH_COMPILATION_CONFIG" \
  -e XDG_CACHE_HOME=/opt/gridbook/ext-cache/xdg \
  -e PRISMAQUANT_CB_EXT_DIR=/opt/gridbook/ext-cache \
  -e PRISMAQUANT_CB_GEMV=inherited \
  -e PRISMAQUANT_CB_FP8_GEMV_V2=0 \
  -e PRISMAQUANT_CB_BF16_SM120=0 \
  -e PRISMAQUANT_CB_FP4_FUSED_MIDM=0 \
  -e PRISMAQUANT_CB_FP4V2_DENSE_R2=0 \
  -e PRISMAQUANT_CB_MOE_PERSISTENT_B=0 \
  -e PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG=0 \
  -e PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R=0 \
  -e PRISMAQUANT_CB_FUSED_MIDM=1 \
  -e PRISMAQUANT_CB_GROUPED_TRIM=1 \
  -e PRISMAQUANT_CB_PREFILL_CHUNK_BYTES=1073741824 \
  -e PRISMAQUANT_CB_DECODE_CONTRACT=v1 \
  -e PRISMAQUANT_SKIP_CB_CAST_CHECK=0 \
  -e PRISMAQUANT_PRELOAD_FUSED=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$BASE_IMAGE" -lc '
    set -euo pipefail
    # Absence is part of the released environment contract.  In particular,
    # literal 0 is invalid for both fused-FP4 selectors and expert chunking.
    unset CUDACXX CXX GRIDBOOK_MXFP8_DENSE \
      PRISMAQUANT_CB_DECODE PRISMAQUANT_CB_EXPAND PRISMAQUANT_CB_PREFILL \
      PRISMAQUANT_CB_FUSED_FP4 PRISMAQUANT_CB_FUSED_FP4_MOE \
      PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK \
      PRISMAQUANT_CB_FP8_SCHED PRISMAQUANT_CB_FP4V2_SCHED \
      PRISMAQUANT_CB_W2_SCHED PRISMAQUANT_CB_W2_ROWS \
      PRISMAQUANT_CB_W2_WARPS PRISMAQUANT_CUTLASS_INCLUDE \
      PRISMAQUANT_DEBUG_PREFIXES
    bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
    test "$(python3 -c '\''from importlib.metadata import version; print(version("gridbook"))'\'')" = "$PQ_GRIDBOOK_RUNTIME_VERSION"
    test "$(python3 -c '\''from importlib.metadata import version; print(version("vllm"))'\'')" = "$PQ_VLLM_VERSION"
    arm_args=()
    if [[ "$PQ_ARM" == eager ]]; then
      arm_args+=(--enforce-eager)
    else
      arm_args+=(--compilation-config "$PQ_GRAPH_COMPILATION_CONFIG")
    fi
    extra_route_args=()
    if python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("/model/quant_config.json").read_text())

def contains_mxfp4_wire(value):
    if isinstance(value, dict):
        return any(contains_mxfp4_wire(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_mxfp4_wire(item) for item in value)
    return value == "mxfp4_e2m1_ue8m0_g32"

raise SystemExit(0 if contains_mxfp4_wire(config) else 1)
PY
    then
      extra_route_args+=(--moe-backend marlin)
    fi
    exec /usr/local/bin/vllm serve /model \
      --served-model-name "$PQ_SERVED_MODEL" \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --tokenizer-mode "$PQ_TOKENIZER_MODE" \
      --generation-config vllm \
      --quantization gridbook \
      --tensor-parallel-size 1 \
      --kv-cache-dtype fp8 \
      --kv-cache-memory-bytes 1073741824 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 512 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.90 \
      "${arm_args[@]}" "${extra_route_args[@]}"
  ')
echo "$CID" > "$EVIDENCE/container-id.txt"

watchdog_pid=
cleanup() {
  set +e
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" >/dev/null 2>&1 || true
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  docker logs "$CID" > "$SERVE_LOG" 2>&1 || true
  docker inspect "$CID" > "$EVIDENCE/container-final-inspect.json" 2>&1 || true
  if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) == true ]]; then
    docker stop -t 30 "$CID" >/dev/null 2>&1 || docker kill "$CID" >/dev/null 2>&1 || true
  fi
  docker rm -f "$CID" >/dev/null 2>&1 || true
  free -h > "$EVIDENCE/memory-after-exit.txt" 2>&1 || true
}
on_signal() {
  code=$1
  trap - EXIT INT TERM HUP
  cleanup
  exit "$code"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP

docker start "$CID" >/dev/null
(
  while [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) == true ]]; do
    timestamp=$(date --iso-8601=seconds)
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    printf '%s mem_available_kib=%s\n' "$timestamp" "$available_kib" >> "$WATCHDOG_LOG"
    if (( available_kib < WATCHDOG_GIB * 1048576 )); then
      printf '%s WATCHDOG: MemAvailable %s KiB below %s GiB; stopping %s\n' \
        "$timestamp" "$available_kib" "$WATCHDOG_GIB" "$CID" >> "$WATCHDOG_LOG"
      : > "$EVIDENCE/watchdog-tripped"
      docker stop -t 5 "$CID" >/dev/null 2>&1 || docker kill "$CID" >/dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
) &
watchdog_pid=$!

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/v1/models" \
      > "$EVIDENCE/models.json"; then
    break
  fi
  if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) != true ]]; then
    echo "REFUSE: $ARM serving container exited before READY" >&2
    exit 1
  fi
  sleep 5
done
if ! curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "REFUSE: $ARM serving container did not become ready" >&2
  exit 1
fi

sleep 10
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: post-READY MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi
if [[ -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: memory watchdog tripped" >&2
  exit 3
fi

# Exercise one inference before fingerprinting so lazily loaded Gridbook/JIT
# extensions are part of the server-side residency scan.
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0.0,\"top_p\":1.0,\"seed\":0,\"n\":1,\"stream\":false}" \
  "http://127.0.0.1:$PORT/v1/completions" > "$EVIDENCE/warmup.json"
python3 - "$EVIDENCE/warmup.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
text = payload["choices"][0]["text"]
if not isinstance(text, str) or not text.strip():
    raise SystemExit("warmup completion was empty")
PY

# Capture evidence is an immutable pre-validation snapshot. cleanup writes a
# separate final log, so the digest recorded on the graph receipt cannot be
# invalidated by the validator's own requests or container shutdown messages.
docker logs "$CID" > "$CAPTURE_LOG" 2>&1

# Fingerprinting is fatal for this gate.  The older serve helper is advisory;
# this ship receipt cannot be, because the stack identity is part of the proof.
verify_runtime_snapshot
docker exec "$CID" env -u PYTHONPATH python3 -P \
  /repo/tools/prismaquant_runtime_snapshot.py verify \
  --snapshot /repo \
  --expected-commit "$ARTIFACT_BUILD_COMMIT" \
  --expected-tree "$PQ_RUNTIME_TREE" \
  --expected-closure-sha256 "$PQ_RUNTIME_CLOSURE_SHA256" >/dev/null
docker exec --workdir / "$CID" env -u PYTHONPATH python3 -P \
  /repo/tools/prismaquant_source_bootstrap.py run-tool --source-root /repo \
  serve-fingerprint write \
  --out /evidence/serve_manifest.json --image "$BASE_IMAGE" \
  --artifact-dir /model --base-url http://127.0.0.1:8000
test -s "$MANIFEST"
verify_runtime_snapshot

graph_evidence_args=()
if [[ "$ARM" == graph ]]; then
  graph_evidence_args+=(--serve-log "$CAPTURE_LOG")
fi
pq_run_module prismaquant.validate_cb_endpoint \
  --arm "$ARM" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model-dir "$MODEL" \
  --model-name "$SERVED_MODEL" \
  --serve-manifest "$MANIFEST" \
  --shipcard "$SHIPCARD" \
  --output-json "$RESULT" \
  --defer-shipcard-fill \
  "${graph_evidence_args[@]}"
verify_runtime_snapshot

# The eager server is already identity-bound by validate_cb_endpoint above.
# Reuse that exact live session for the numeric catastrophic-quality gate; a
# second serve would lose the nonce/process/artifact binding we just proved.
# validate_quantized_model deliberately treats shipcard write errors as
# advisory, so this release driver must explicitly replay and verify only the
# ship_gate slot before it is allowed to continue.
if [[ "$ARM" == eager ]]; then
  ship_gate_base_url="http://127.0.0.1:$PORT"
  pq_run_module prismaquant.validate_quantized_model \
    --base-url "$ship_gate_base_url" \
    --model-name "$SERVED_MODEL" \
    --max-ppl 25.0 \
    --max-mean-nll 3.0 \
    --max-p99-nll 6.0 \
    --min-gen-len 30 \
    --min-mtp-accept-p0 0.60 \
    --shipcard "$SHIPCARD" \
    --artifact-dir "$MODEL" \
    --report "$SHIP_GATE_REPORT"
  verify_runtime_snapshot
  env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -P - \
      "$REPO" "$SHIPCARD" "$MODEL" "$SERVED_MODEL" "$ship_gate_base_url" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv.pop(1)) / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

from prismaquant.shipcard import load_shipcard, verify

shipcard_path, model_dir, served_model, base_url = sys.argv[1:]
card = load_shipcard(shipcard_path)
problems = verify(card, model_dir=model_dir, required=("ship_gate",))
record = (card.get("slots") or {}).get("ship_gate")
if isinstance(record, dict):
    if record.get("served_model_name") != served_model:
        problems.append(
            "ship_gate: served-model nonce differs from the current eager session"
        )
    if record.get("base_url") != base_url:
        problems.append("ship_gate: base URL differs from the current eager session")
    if record.get("model_sha_source") != model_dir:
        problems.append("ship_gate: artifact path differs from the mounted model")
if problems:
    raise SystemExit("REFUSE: current eager ship_gate did not verify:\n- " + "\n- ".join(problems))
print("[shipcard] verified ship_gate from current nonce-bound eager session")
PY
  verify_runtime_snapshot
fi

if [[ $(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null) != true \
      || -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: container or memory watchdog failed during validation" >&2
  exit 3
fi
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: final MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi

# End the measured serve before mutating the receipt. This removes the final
# check/write race: no watchdog or server transition can occur after the
# terminal proof and leave a stale PASS behind.
if ! kill -0 "$watchdog_pid" 2>/dev/null; then
  echo "REFUSE: memory watchdog exited before planned server shutdown" >&2
  exit 3
fi
docker stop -t 30 "$CID" >/dev/null
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=
if [[ -e "$EVIDENCE/watchdog-tripped" ]]; then
  echo "REFUSE: memory watchdog tripped before clean server shutdown" >&2
  exit 3
fi
container_exit=$(docker inspect -f '{{.State.ExitCode}}' "$CID")
container_oom=$(docker inspect -f '{{.State.OOMKilled}}' "$CID")
if [[ "$container_oom" != false ]]; then
  echo "REFUSE: serving container was OOM-killed" >&2
  exit 3
fi
if [[ "$container_exit" != 0 && "$container_exit" != 137 && "$container_exit" != 143 ]]; then
  echo "REFUSE: serving container exited unexpectedly with code $container_exit" >&2
  exit 3
fi
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < READY_FLOOR_GIB * 1048576 )); then
  echo "REFUSE: post-shutdown MemAvailable is below ${READY_FLOOR_GIB} GiB" >&2
  exit 3
fi

# This is intentionally the terminal mutation. A PASS cannot reach the card
# until endpoint, clean server shutdown, watchdog, and memory checks all pass.
verify_runtime_snapshot
env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -P - \
    "$REPO" "$RESULT" "$SHIPCARD" "$MODEL" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv.pop(1)) / "tools"))
from prismaquant_source_bootstrap import activate_prismaquant_source
activate_prismaquant_source()

from prismaquant.validate_cb_endpoint import commit_deferred_result

commit_deferred_result(sys.argv[1], sys.argv[2], sys.argv[3])
PY
verify_runtime_snapshot
if [[ "$ARM" == eager ]]; then
  echo "PASS: native_export.eager and ship_gate filled from exact-pinned CB serve"
else
  echo "PASS: native_export.graph filled from exact-pinned CB serve"
fi
