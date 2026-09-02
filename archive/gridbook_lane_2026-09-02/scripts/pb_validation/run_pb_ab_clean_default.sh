#!/usr/bin/env bash
# 0.8.9 DEFAULT-STATE serve on the SHIPPED clean 87 GB body — the served leg
# the default-flip promotion needs.
#
# r1-r4 tried PRISMAQUANT_CB_MOE_PERSISTENT_B=1 (require) and learned the
# real lesson at load: every FP8-CB expert layer of this body carries
# per-role split books, which the FP8 arm cannot consume, and the explicit
# flag fails the load BY DESIGN. The candidate tri-state (commit 599a15d)
# exists for exactly this: unset means AUTO — engage the family arm where it
# attests, keep the bridge where it does not, with route telemetry naming
# which. This driver measures that composite default state:
#   * persistent-B auto: 32 FP4-CB routed layers on the lane, 11 FP8-CB
#     routed layers on the bridge (announced fallback);
#   * CB_GEMV auto: predicate-gated — expected inherited on these widths;
#   * CB_FP8_GEMV_V2 auto: whole-row-v2 on the qualified K28 cells.
# Arm A is the existing gold measurement (gold-results-92gb-clean: KL 1.2221 /
# PPL 20.95): same image digest, same 0.8.8 wheel as entry-point provider,
# same teacher, same inputs. The candidate gridbook is overlaid from a
# DETACHED WORKTREE at 599a15d; the default-state shim applies the 0.8.9
# unset-means-auto semantics through the closed gold contract and rewrites
# the receipt to the observed truth.
#
# Comparison is CROSS-SESSION vs gold (±0.7% KL envelope observed on repeats
# of identical bytes) — the same-session exact pairing for the FP4 arm
# already exists (gold-ab: kl_mean −0.051%).
set -euo pipefail

RUN=/home/rob/dq-runs/dsv4-flash-0731
REPO=${REPO:-/home/rob/pq-dsv4flash-release}
MODEL=$RUN/artifact-aura-cb-92gb-clean
GOLD=$RUN/gold-results-92gb-clean
AB=$RUN/pb-validation/gold-ab-clean-default
GB_WORKTREE=$RUN/pb-validation/gridbook-599a15d
WIKITEXT_INPUTS=$RUN/gold-inputs/dsv4-wikitext-inputs-v1.json
WIKITEXT_FILE_SHA256=1176ba1214a640e086f353ff485a339bf2069ad5dd964f990569af30bb38dd5b
GPU_LOCK=/home/rob/dq-runs/gpu.lock
WATCHDOG_FLOOR_GIB=4
PQ_SNAPSHOT_CACHE=$RUN/runtime-source-cache
IMAGE=eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
IMAGE_ID=sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
EXPECTED_GRIDBOOK_COMMIT=064a4cb093da10d7c35be03435bb6a525280a45f
CANDIDATE_COMMIT=599a15d2a216e10f15533674afff74cba1d11114
EXT_CACHE=$RUN/gridbook-ext-cache/pb-ab-clean-d9dfe53-eugr-58862b38
PROBE=$RUN/pb-validation/dump_routes_probe_default_state.py

TEACHER=$GOLD/dsv4_bf16_teacher_topk8192.pt
TEACHER_META=$GOLD/dsv4_bf16_teacher_topk8192.meta.json

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 2; }

[[ -f "$TEACHER" && -f "$TEACHER_META" ]] \
  || refuse "clean gold teacher pair is missing under $GOLD"
[[ -f "$MODEL/model.safetensors" ]] || refuse "no checkpoint: $MODEL"
[[ -f "$WIKITEXT_INPUTS" ]] || refuse "WikiText payload missing"
observed_inputs_sha=$(sha256sum -- "$WIKITEXT_INPUTS" | awk '{print $1}')
[[ "$observed_inputs_sha" == "$WIKITEXT_FILE_SHA256" ]] \
  || refuse "WikiText payload bytes differ: $observed_inputs_sha"
[[ -f "$PROBE" ]] || refuse "route probe missing: $PROBE"
[[ -e "$GPU_LOCK" ]] || refuse "GPU mutex missing"
observed_wt=$(git -C "$GB_WORKTREE" rev-parse HEAD 2>/dev/null || true)
[[ "$observed_wt" == "$CANDIDATE_COMMIT" ]] \
  || refuse "gridbook worktree is $observed_wt, not the candidate $CANDIDATE_COMMIT"
[[ -z "$(git -C "$GB_WORKTREE" status --porcelain=v1)" ]] \
  || refuse "gridbook worktree is dirty"

HEAD=$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true)
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] || refuse "cannot resolve release checkout HEAD"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] \
  || refuse "release checkout is dirty"

observed_image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)
[[ "$observed_image_id" == "$IMAGE_ID" ]] || refuse "eugr image absent/mismatched"

PQ_SNAPSHOT_JSON=$(python3 "$REPO/tools/prismaquant_runtime_snapshot.py" \
  materialize --source-root "$REPO" --cache-root "$PQ_SNAPSHOT_CACHE" \
  --commit "$HEAD")
readarray -t pq_snapshot_fields < <(python3 -c \
  'import json,sys; p=json.load(sys.stdin); print(p["snapshot"]); print(p["tree"]); print(p["closure_sha256"])' \
  <<<"$PQ_SNAPSHOT_JSON")
PQ_RUNTIME_SNAPSHOT=${pq_snapshot_fields[0]:-}
[[ -d "$PQ_RUNTIME_SNAPSHOT" ]] || refuse "could not materialize PQ snapshot"

# shellcheck source=/dev/null
. "$PQ_RUNTIME_SNAPSHOT/prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
gridbook_serving_runtime_prepare
[[ "${GRIDBOOK_RUNTIME_COMMIT:-}" == "$EXPECTED_GRIDBOOK_COMMIT" ]] \
  || refuse "pinned wheel is ${GRIDBOOK_RUNTIME_COMMIT:-unset}, not 0.8.8"

mkdir -p -- "$AB/logs" "$AB/container-home" "$EXT_CACHE/xdg"
printf '{"candidate_gridbook_commit": "%s", "wheel_entrypoint_commit": "%s", "arm": "default-state (unset means auto: persistent-B + both GEMV selectors)"}\n' \
  "$CANDIDATE_COMMIT" "$EXPECTED_GRIDBOOK_COMMIT" > "$AB/candidate-identity.json"

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

run_gpu_python() {
  local name=$1 log=$2 done_artifact=$3 extra_env=$4 docker_pid watchdog_pid rc oom
  shift 4
  if [[ -s "$done_artifact" ]]; then
    printf '[pb-ab-default] %s SKIP (artifact exists)\n' "$name"
    return 0
  fi
  docker inspect "$name" >/dev/null 2>&1 && refuse "container exists: $name"
  if ! docker ps --format '{{.Names}}' | grep -q '^pb-ab-'; then
    { find "$EXT_CACHE" "$AB/container-home/.cache/prismaquant-cb-ext" \
        -name lock -type f 2>/dev/null || true; } | while read -r stale; do
      printf '[pb-ab-default] %s: removing stale ext baton %s\n' "$name" "$stale"
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
    -v "$GB_WORKTREE:$GB_WORKTREE:ro" \
    "${GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS[@]}" \
    "${env_args[@]}" \
    -e "GB_CANDIDATE_TREE=$GB_WORKTREE" \
    -e "GB_CANDIDATE_COMMIT=$CANDIDATE_COMMIT" \
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
GB_DST=$(python3 -c "import gridbook, os; print(os.path.dirname(gridbook.__file__))")
rm -rf "$GB_DST"
cp -a "${GB_CANDIDATE_TREE:?}/gridbook" "$GB_DST"
python3 - <<PYEOF
import gridbook, pathlib
got = pathlib.Path(gridbook.__file__).parent
probe = (got / "moe_persistent_b_lane.py").read_text()
assert "def supports_fp8" in probe, \
    "candidate overlay did not land: FP8 arm symbols absent"
assert "latched_choice" in probe, \
    "candidate overlay did not land: tri-state selector absent"
print(f"[pb-ab-default] candidate gridbook overlaid at {got}")
PYEOF
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
  printf '[pb-ab-default] %s rc=%s oom=%s\n' "$name" "$rc" "$oom"
  (( rc == 0 )) || refuse "stage $name failed (rc=$rc oom=$oom); read $log"
}

# No container-level flag: the default-state shim/probe apply the 0.8.9
# unset-means-auto deviations in-process, AFTER the gold contract's
# canonical activation, and rewrite the receipt to the observed truth.

exec 9>"$GPU_LOCK"
flock 9

run_gpu_python pb-ab-default-probe-b "$AB/logs/probe-b.log" "$AB/routes-arm-b.json" - \
  "$PROBE" "$MODEL" "$AB/routes-arm-b.json"

run_gpu_python pb-ab-default-kl-b "$AB/logs/kl-b.log" "$AB/kl-arm-b.json" - \
  "$RUN/pb-validation/pb_default_state_shim.py" \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_full_kl.py" \
    --mode student --model "$MODEL" \
    --output "$AB/kl-arm-b.json" \
    --teacher-payload "$TEACHER" --teacher-meta "$TEACHER_META" \
    --score-positions all --dsv4-gridbook-contract

run_gpu_python pb-ab-default-ppl-b "$AB/logs/ppl-b.log" "$AB/ppl-arm-b.json" - \
  "$RUN/pb-validation/pb_default_state_shim.py" \
  "$PQ_RUNTIME_SNAPSHOT/tools/measure_vllm_wikitext_ppl.py" \
    --model "$MODEL" --output "$AB/ppl-arm-b.json" \
    --wikitext-inputs "$WIKITEXT_INPUTS" --dsv4-gridbook-contract

python3 - "$AB" "$GOLD" <<'EOF'
import json, sys
ab, gold = sys.argv[1], sys.argv[2]
kb = json.load(open(f"{ab}/kl-arm-b.json"))
pb = json.load(open(f"{ab}/ppl-arm-b.json"))
ka = json.load(open(f"{gold}/dsv4_gridbook_kl.json"))
pa = json.load(open(f"{gold}/dsv4_gridbook_ppl.json"))
print("\n=== 0.8.9 DEFAULT-STATE arm-B vs GOLD, clean 87GB body ===")
print("(cross-session compare; ±0.7% KL envelope on identical bytes)")
for name in ("kl_mean", "kl_p99", "kl_max"):
    va, vb = ka[name], kb[name]
    print(f"{name:8s}  gold={va:.6f}  pb={vb:.6f}  d={vb-va:+.2e}  rel={(vb-va)/va:+.3%}")
for name in ("ppl", "mean_nll", "max_chunk_mean_nll"):
    print(f"{name:18s}  gold={pa[name]:.6f}  pb={pb[name]:.6f}  d={pb[name]-pa[name]:+.2e}")
EOF

printf '\n[pb-ab-default] complete: %s\n' "$AB"
