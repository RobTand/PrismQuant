#!/usr/bin/env bash
# Persistent-B served A/B on the built DSv4 92 GB body.
#
# Two arms, identical in every dimension except PRISMAQUANT_CB_MOE_PERSISTENT_B:
#   A (baseline)  routed batch prefill on the default expand+Sm80 bridge
#   B (candidate) 32 FP4-CB routed layers on cb_moe_persistent_b_prefill
#                 (the 11 FP8-CB routed layers keep the bridge in BOTH arms --
#                 moe.py:662 `if persistent_b_on and self.is_fp4`)
#
# Stages per arm: dispatch-proof probe -> full-vocab KL vs the digest-validated
# corrected-rope teacher -> WikiText PPL. Same eugr image digest, same gridbook
# 0.8.8 wheel pin, same reviewed PrismaQuant snapshot, same GPU mutex and
# memory watchdog as run_gold_92gb.sh, which this driver is cut down from.
# It fills NO gold slots and refuses to touch gold-results-92gb outputs.
#
# The KL/PPL identity claim rides the reassociation-class surface only: decode
# is bit-identical to cb_expand_v2 (tested upstream with torch.equal); the FP32
# GEMM reduction order is what differs. Expect |dKL| within between-arm
# arithmetic noise; a large delta FAILS the candidate.
set -euo pipefail

RUN=/home/rob/dq-runs/dsv4-flash-0731
REPO=${REPO:-/home/rob/pq-dsv4flash-release}
MODEL=${MODEL:-$RUN/artifact-aura-cb-92gb}
GOLD=$RUN/gold-results-92gb
AB=${AB:-$RUN/pb-validation/gold-ab}
WIKITEXT_INPUTS=${WIKITEXT_INPUTS:-$RUN/gold-inputs/dsv4-wikitext-inputs-v1.json}
WIKITEXT_FILE_SHA256=1176ba1214a640e086f353ff485a339bf2069ad5dd964f990569af30bb38dd5b
GPU_LOCK=/home/rob/dq-runs/gpu.lock
WATCHDOG_FLOOR_GIB=4
PQ_SNAPSHOT_CACHE=$RUN/runtime-source-cache
IMAGE=eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
IMAGE_ID=sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
EXPECTED_GRIDBOOK_COMMIT=064a4cb093da10d7c35be03435bb6a525280a45f
EXT_CACHE=$RUN/gridbook-ext-cache/pb-ab-064a4cb0-eugr-58862b38
PROBE=$RUN/pb-validation/dump_routes_probe.py

TEACHER=$GOLD/dsv4_bf16_teacher_topk8192.pt
TEACHER_META=$GOLD/dsv4_bf16_teacher_topk8192.meta.json

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 2; }

[[ -f "$TEACHER" && -f "$TEACHER_META" ]] \
  || refuse "corrected-rope teacher pair is missing under $GOLD"
[[ -f "$MODEL/model.safetensors" ]] \
  || refuse "artifact has no safetensors checkpoint: $MODEL"
[[ -f "$WIKITEXT_INPUTS" ]] \
  || refuse "offline WikiText payload is missing: $WIKITEXT_INPUTS"
observed_inputs_sha=$(sha256sum -- "$WIKITEXT_INPUTS" | awk '{print $1}')
[[ "$observed_inputs_sha" == "$WIKITEXT_FILE_SHA256" ]] \
  || refuse "offline WikiText payload bytes differ: $observed_inputs_sha"
[[ -f "$PROBE" ]] || refuse "route probe is missing: $PROBE"
[[ -e "$GPU_LOCK" ]] || refuse "host GPU mutex is missing: $GPU_LOCK"

HEAD=$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true)
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] \
  || refuse "cannot resolve the release checkout's HEAD"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] \
  || refuse "release checkout is dirty"

observed_image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)
[[ "$observed_image_id" == "$IMAGE_ID" ]] \
  || refuse "exact eugr image is absent or resolves to $observed_image_id"

PQ_SNAPSHOT_JSON=$(python3 "$REPO/tools/prismaquant_runtime_snapshot.py" \
  materialize --source-root "$REPO" --cache-root "$PQ_SNAPSHOT_CACHE" \
  --commit "$HEAD")
readarray -t pq_snapshot_fields < <(python3 -c \
  'import json,sys; p=json.load(sys.stdin); print(p["snapshot"]); print(p["tree"]); print(p["closure_sha256"])' \
  <<<"$PQ_SNAPSHOT_JSON")
PQ_RUNTIME_SNAPSHOT=${pq_snapshot_fields[0]:-}
PQ_RUNTIME_TREE=${pq_snapshot_fields[1]:-}
PQ_RUNTIME_CLOSURE_SHA256=${pq_snapshot_fields[2]:-}
[[ -d "$PQ_RUNTIME_SNAPSHOT" ]] \
  || refuse "could not materialize the reviewed PrismaQuant snapshot"
PQ_SOURCE_SHA256=$(python3 \
  "$PQ_RUNTIME_SNAPSHOT/tools/container_runtime_identity.py" \
  source-sha256 --source-root "$PQ_RUNTIME_SNAPSHOT")

# shellcheck source=/dev/null
. "$PQ_RUNTIME_SNAPSHOT/prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
gridbook_serving_runtime_prepare
[[ "${GRIDBOOK_RUNTIME_COMMIT:-}" == "$EXPECTED_GRIDBOOK_COMMIT" ]] \
  || refuse "Gridbook runtime is ${GRIDBOOK_RUNTIME_COMMIT:-unset}, not the serving pin"

mkdir -p -- "$AB/logs" "$AB/container-home" "$EXT_CACHE/xdg"

current_container=
cleanup() {
  [[ -n "$current_container" ]] \
    && docker rm -f "$current_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

check_memory_floor() {
  local available_kib
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  (( available_kib >= WATCHDOG_FLOOR_GIB * 1048576 )) \
    || refuse "MemAvailable below ${WATCHDOG_FLOOR_GIB} GiB before launch"
}

# run_gpu_python <name> <log> <done-artifact> <extra-env "K=V" or -> <script + args...>
run_gpu_python() {
  local name=$1 log=$2 done_artifact=$3 extra_env=$4 docker_pid watchdog_pid rc oom
  shift 4
  if [[ -s "$done_artifact" ]]; then
    printf '[pb-ab] %s SKIP (artifact exists: %s)\n' "$name" "$done_artifact"
    return 0
  fi
  docker inspect "$name" >/dev/null 2>&1 \
    && refuse "container name already exists: $name"
  # A killed container can orphan a torch-cpp_extension FileBaton `lock` in the
  # persistent ext caches; the NEXT flag-on serve then polls it silently
  # forever (observed 2026-08-17: run 3's docker-kill landed during a no-op
  # ninja recheck, and run 4's probe-b sat 26 min at 0% CPU). No pb-ab
  # container is live at this point in the driver, so any baton here is stale.
  if ! docker ps --format '{{.Names}}' | grep -q '^pb-ab-'; then
    find "$EXT_CACHE" "$AB/container-home/.cache/prismaquant-cb-ext" \
        -name lock -type f 2>/dev/null | while read -r stale; do
      printf '[pb-ab] %s: removing stale ext-build baton %s\n' "$name" "$stale"
      docker run --rm -v "$(dirname "$stale"):/lockdir" "$IMAGE" \
        rm -f /lockdir/lock 2>/dev/null || rm -f "$stale" || true
    done
  fi
  check_memory_floor
  local -a env_args=()
  [[ "$extra_env" != "-" ]] && env_args+=(-e "$extra_env")
  current_container=$name
  docker run --pull=never --name "$name" \
    --network none --gpus all --ipc=host --user 0:0 --oom-score-adj=1000 \
    --log-driver local --log-opt max-size=100m --log-opt max-file=3 \
    -v "$PQ_RUNTIME_SNAPSHOT:$PQ_RUNTIME_SNAPSHOT:ro" \
    -v "$RUN/pb-validation:$RUN/pb-validation:ro" \
    -v "$GOLD:$GOLD:ro" \
    -v "$WIKITEXT_INPUTS:$WIKITEXT_INPUTS:ro" \
    -v "$MODEL:$MODEL:ro" \
    -v "$AB:$AB:rw" \
    -v "$EXT_CACHE:$EXT_CACHE:rw" \
    "${GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS[@]}" \
    "${env_args[@]}" \
    -e "PRISMAQUANT_IDENTITY_GIT_COMMIT=$HEAD" \
    -e PRISMAQUANT_IDENTITY_GIT_DIRTY=0 \
    -e "PQ_RUNTIME_PRISMAQUANT_ROOT=$PQ_RUNTIME_SNAPSHOT" \
    -e PYTHONSAFEPATH=1 \
    -e PYTHONNOUSERSITE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e "HOME=$AB/container-home" \
    -e "XDG_CACHE_HOME=$EXT_CACHE/xdg" \
    --entrypoint bash "$IMAGE_ID" -ceu '
bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container
python3 "$@"
' bash "$@" >"$log" 2>&1 &
  docker_pid=$!
  (
    while kill -0 "$docker_pid" 2>/dev/null; do
      available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
      if (( available_kib < WATCHDOG_FLOOR_GIB * 1048576 )); then
        printf 'WATCHDOG: below %s GiB; stopping %s\n' \
          "$WATCHDOG_FLOOR_GIB" "$name"
        docker stop --time 30 "$name" >/dev/null 2>&1 || true
        exit 1
      fi
      sleep 2
    done
  ) >>"$AB/logs/${name}-watchdog.log" 2>&1 &
  watchdog_pid=$!
  set +e; wait "$docker_pid"; rc=$?; set -e
  kill "$watchdog_pid" >/dev/null 2>&1 || true
  wait "$watchdog_pid" 2>/dev/null || true
  oom=$(docker inspect -f '{{.State.OOMKilled}}' "$name" 2>/dev/null || printf unknown)
  docker rm "$name" >/dev/null 2>&1 || true
  current_container=
  printf '[pb-ab] %s rc=%s oom=%s\n' "$name" "$rc" "$oom"
  (( rc == 0 )) || refuse "stage $name failed (rc=$rc oom=$oom); read $log"
}

PB_FLAG="PRISMAQUANT_CB_MOE_PERSISTENT_B=1"

exec 9>"$GPU_LOCK"
flock 9

# ---- arm A: dispatch proof, KL, PPL -----------------------------------------
run_gpu_python pb-ab-probe-a "$AB/logs/probe-a.log" "$AB/routes-arm-a.json" - \
  "$PROBE" "$MODEL" "$AB/routes-arm-a.json"

run_gpu_python pb-ab-kl-a "$AB/logs/kl-a.log" "$AB/kl-arm-a.json" - \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_full_kl.py" \
    --mode student --model "$MODEL" \
    --output "$AB/kl-arm-a.json" \
    --teacher-payload "$TEACHER" --teacher-meta "$TEACHER_META" \
    --score-positions all --dsv4-gridbook-contract

run_gpu_python pb-ab-ppl-a "$AB/logs/ppl-a.log" "$AB/ppl-arm-a.json" - \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_wikitext_ppl.py" \
    --model "$MODEL" --output "$AB/ppl-arm-a.json" \
    --wikitext-inputs "$WIKITEXT_INPUTS" --dsv4-gridbook-contract

# ---- arm B: identical, plus the ONE DECLARED deviation ----------------------
# The gold contract is CLOSED: exact_llm_contract pops every allowlisted
# PRISMAQUANT_* var and installs the canonical state (PERSISTENT_B=0), so the
# container -e alone is silently reverted before model load (run 3 proved it:
# arm B probed flag='0' and served the bridge).  The -e stays as the INTENT
# channel (the probe reads it before activation); pb_flag_shim.py is what
# APPLIES it — canonical install first, then the flag, then the receipt
# rewritten to the observed truth, so the two arms' serve fingerprints differ
# in exactly and only the declared flag.
run_gpu_python pb-ab-probe-b "$AB/logs/probe-b.log" "$AB/routes-arm-b.json" "$PB_FLAG" \
  "$PROBE" "$MODEL" "$AB/routes-arm-b.json"

run_gpu_python pb-ab-kl-b "$AB/logs/kl-b.log" "$AB/kl-arm-b.json" "$PB_FLAG" \
  "$RUN/pb-validation/pb_flag_shim.py" \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_full_kl.py" \
    --mode student --model "$MODEL" \
    --output "$AB/kl-arm-b.json" \
    --teacher-payload "$TEACHER" --teacher-meta "$TEACHER_META" \
    --score-positions all --dsv4-gridbook-contract

run_gpu_python pb-ab-ppl-b "$AB/logs/ppl-b.log" "$AB/ppl-arm-b.json" "$PB_FLAG" \
  "$RUN/pb-validation/pb_flag_shim.py" \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_wikitext_ppl.py" \
    --model "$MODEL" --output "$AB/ppl-arm-b.json" \
    --wikitext-inputs "$WIKITEXT_INPUTS" --dsv4-gridbook-contract

python3 - "$AB" <<'EOF'
import json, sys
ab = sys.argv[1]
def load(p):
    with open(f"{ab}/{p}") as f: return json.load(f)
ka, kb = load("kl-arm-a.json"), load("kl-arm-b.json")
pa, pb = load("ppl-arm-a.json"), load("ppl-arm-b.json")
print("\n=== PERSISTENT-B SERVED A/B (built 92GB body, corrected-rope teacher) ===")
for name, a, b in (("kl_mean", ka, kb), ("kl_p99", ka, kb), ("kl_max", ka, kb)):
    va, vb = a[name], b[name]
    print(f"{name:8s}  A={va:.6f}  B={vb:.6f}  d={vb-va:+.2e}  rel={0 if va==0 else (vb-va)/va:+.3%}")
for name in ("ppl", "mean_nll", "max_chunk_mean_nll"):
    print(f"{name:18s}  A={pa[name]:.6f}  B={pb[name]:.6f}  d={pb[name]-pa[name]:+.2e}")
EOF

printf '\n[pb-ab] complete: %s\n' "$AB"
