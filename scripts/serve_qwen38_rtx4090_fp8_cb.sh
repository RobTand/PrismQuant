#!/usr/bin/env bash
# Launch and attest one strict Qwen3.8-27B FP8-CB endpoint arm. Run once with
# SERVE_ARM=eager and once with the default SERVE_ARM=graph; the graph arm is
# mandatory and additionally fills the physical RTX4090 shipcard slot.
# This release launcher requires exactly one physical RTX 4090 (SM89).
# A Spark or another GPU may be useful for compile-only structural diagnosis,
# but it must not run this script or produce an RTX 4090 release receipt.

set -euo pipefail

PQ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PQ_REPO_ROOT"

: "${MODEL_DIR:?Set MODEL_DIR to the strict exported artifact}"
: "${SERVE_IMAGE:?Set SERVE_IMAGE to immutable name@sha256:<digest>}"
: "${GRIDBOOK_RUNTIME_PIN:?Set GRIDBOOK_RUNTIME_PIN to the candidate release pin JSON}"
: "${VLLM_RUNTIME_PIN:?Set VLLM_RUNTIME_PIN to the official vLLM VCS/RECORD pin JSON}"
: "${GRIDBOOK_RUNTIME_CONTRACT:?Set GRIDBOOK_RUNTIME_CONTRACT to its Gridbook v10 runtime_contract.json}"
: "${EVIDENCE_DIR:?Set EVIDENCE_DIR to a fresh dedicated absolute directory}"
: "${SERVE_ARM:=graph}"

case "$SERVE_ARM" in
  eager|graph) ;;
  *)
    echo "[rtx4090-serve] REFUSED: SERVE_ARM must be eager or graph" >&2
    exit 2
    ;;
esac

for path_name in MODEL_DIR GRIDBOOK_RUNTIME_PIN VLLM_RUNTIME_PIN GRIDBOOK_RUNTIME_CONTRACT EVIDENCE_DIR; do
  path_value="${!path_name}"
  if [[ "$path_value" != /* || "$path_value" == / ]]; then
    echo "[rtx4090-serve] REFUSED: $path_name must be a dedicated absolute path" >&2
    exit 2
  fi
done
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/quant_config.json" ]]; then
  echo "[rtx4090-serve] REFUSED: MODEL_DIR is not a strict exported artifact" >&2
  exit 2
fi
if [[ ! -f "$GRIDBOOK_RUNTIME_PIN" || -L "$GRIDBOOK_RUNTIME_PIN" \
   || ! -f "$VLLM_RUNTIME_PIN" || -L "$VLLM_RUNTIME_PIN" \
   || ! -f "$GRIDBOOK_RUNTIME_CONTRACT" || -L "$GRIDBOOK_RUNTIME_CONTRACT" ]]; then
  echo "[rtx4090-serve] REFUSED: runtime pins and contract must be ordinary files" >&2
  exit 2
fi
if [[ ! "$SERVE_IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "[rtx4090-serve] REFUSED: SERVE_IMAGE must be immutable name@sha256:<digest>" >&2
  exit 2
fi
if [[ -e "$EVIDENCE_DIR" ]]; then
  echo "[rtx4090-serve] REFUSED: EVIDENCE_DIR must not already exist" >&2
  exit 2
fi

mapfile -t PQ_GPU_ROWS < <(
  nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader,nounits
)
if (( ${#PQ_GPU_ROWS[@]} != 1 )); then
  echo "[rtx4090-serve] REFUSED: exactly one visible physical GPU is required" >&2
  exit 2
fi
IFS=',' read -r PQ_GPU_NAME PQ_GPU_UUID PQ_GPU_CC <<<"${PQ_GPU_ROWS[0]}"
PQ_GPU_NAME="${PQ_GPU_NAME#${PQ_GPU_NAME%%[![:space:]]*}}"
PQ_GPU_NAME="${PQ_GPU_NAME%${PQ_GPU_NAME##*[![:space:]]}}"
PQ_GPU_UUID="${PQ_GPU_UUID#${PQ_GPU_UUID%%[![:space:]]*}}"
PQ_GPU_UUID="${PQ_GPU_UUID%${PQ_GPU_UUID##*[![:space:]]}}"
PQ_GPU_CC="${PQ_GPU_CC#${PQ_GPU_CC%%[![:space:]]*}}"
PQ_GPU_CC="${PQ_GPU_CC%${PQ_GPU_CC##*[![:space:]]}}"
if [[ "$PQ_GPU_NAME" != "NVIDIA GeForce RTX 4090" || "$PQ_GPU_CC" != "8.9" ]]; then
  echo "[rtx4090-serve] REFUSED: observed '$PQ_GPU_NAME' CC '$PQ_GPU_CC'; a physical RTX 4090/SM89 is mandatory" >&2
  exit 2
fi

docker image inspect "$SERVE_IMAGE" >/dev/null

PQ_SHIPCARD="$MODEL_DIR/shipcard.json"
if [[ ! -f "$PQ_SHIPCARD" ]]; then
  echo "[rtx4090-serve] REFUSED: strict artifact has no shipcard.json" >&2
  exit 2
fi

# Validate the candidate pin, the v10 contract, every selected K%4 route, the
# dense model identity, and the recursive 18 GB inventory before CUDA startup.
PQ_MODEL_DIR="$MODEL_DIR" \
PQ_RUNTIME_PIN="$GRIDBOOK_RUNTIME_PIN" \
PQ_VLLM_RUNTIME_PIN="$VLLM_RUNTIME_PIN" \
PQ_RUNTIME_CONTRACT="$GRIDBOOK_RUNTIME_CONTRACT" \
python3 - <<'PY'
import json
import os
from pathlib import Path

from prismaquant.validate_rtx4090_fp8_cb import (
    artifact_runtime_attestation,
    validate_candidate_runtime_pin,
    validate_candidate_vllm_runtime_pin,
    validate_rtx4090_artifact_metadata,
)

pin = json.loads(Path(os.environ["PQ_RUNTIME_PIN"]).read_text(encoding="utf-8"))
contract = json.loads(
    Path(os.environ["PQ_RUNTIME_CONTRACT"]).read_text(encoding="utf-8")
)
validate_candidate_runtime_pin(pin, runtime_contract=contract)
vllm_pin = json.loads(
    Path(os.environ["PQ_VLLM_RUNTIME_PIN"]).read_text(encoding="utf-8")
)
validate_candidate_vllm_runtime_pin(vllm_pin)
validate_rtx4090_artifact_metadata(
    os.environ["PQ_MODEL_DIR"], runtime_contract=contract
)
artifact_runtime_attestation(os.environ["PQ_MODEL_DIR"])
PY

# Resolve the exact weight set from the finalized content/index ledgers without
# opening any shard. Each shard is then mounted individually over /model so its
# verified inode cannot be retargeted between the preflight process and vLLM.
if [[ "$MODEL_DIR" == *:* || "$MODEL_DIR" == *$'\n'* ]]; then
  echo "[rtx4090-serve] REFUSED: MODEL_DIR contains an unsafe mount character" >&2
  exit 2
fi
if ! PQ_WEIGHT_FILE_TEXT="$(
  PQ_MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import os
from prismaquant.validate_rtx4090_fp8_cb import (
    rtx4090_artifact_content_expectations,
)

expectations = rtx4090_artifact_content_expectations(
    os.environ["PQ_MODEL_DIR"]
)
print(*expectations["weight_manifest"]["files"], sep="\n")
PY
)"; then
  echo "[rtx4090-serve] REFUSED: could not resolve strict weight mounts" >&2
  exit 2
fi
mapfile -t PQ_WEIGHT_FILES <<<"$PQ_WEIGHT_FILE_TEXT"
PQ_WEIGHT_MOUNTS=()
for name in "${PQ_WEIGHT_FILES[@]}"; do
  if [[ ! "$name" =~ ^[A-Za-z0-9._-]+[.]safetensors$ \
     || ! -f "$MODEL_DIR/$name" || -L "$MODEL_DIR/$name" ]]; then
    echo "[rtx4090-serve] REFUSED: unsafe strict weight shard '$name'" >&2
    exit 2
  fi
  PQ_WEIGHT_MOUNTS+=( -v "$MODEL_DIR/$name:/model/$name:ro" )
done
if (( ${#PQ_WEIGHT_MOUNTS[@]} == 0 )); then
  echo "[rtx4090-serve] REFUSED: strict artifact has no weight shards" >&2
  exit 2
fi

mkdir -p "$EVIDENCE_DIR"
PQ_SESSION_NONCE="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"
PQ_MANIFEST="$EVIDENCE_DIR/serve_manifest.json"
PQ_SERVE_LOG="$EVIDENCE_DIR/serve.log"
PQ_ARTIFACT_CONTENT_RECEIPT_CONTAINER=/run/prismaquant-rtx4090-content-receipt.json
PQ_COMPILE_CACHE_CONTAINER=/compile-cache
PQ_GRAPH_CONFIG=""
PQ_COMPILE_MOUNT=()
PQ_VALIDATOR_GRAPH_ARGS=()
if [[ "$SERVE_ARM" == graph ]]; then
  PQ_COMPILE_CACHE_HOST="$EVIDENCE_DIR/torch-compile-cache"
  PQ_COMPILE_CACHE_PREFLIGHT="$EVIDENCE_DIR/compile_cache_preflight.json"
  PQ_CACHE_HOST="$PQ_COMPILE_CACHE_HOST" \
  PQ_CACHE_RECEIPT="$PQ_COMPILE_CACHE_PREFLIGHT" \
  PQ_CACHE_CONTAINER="$PQ_COMPILE_CACHE_CONTAINER" \
  PQ_SESSION_NONCE="$PQ_SESSION_NONCE" python3 - <<'PY'
import os
from prismaquant.rtx4090_graph_contract import create_compile_cache_preflight

create_compile_cache_preflight(
    os.environ["PQ_CACHE_HOST"],
    os.environ["PQ_CACHE_RECEIPT"],
    configured_container_root=os.environ["PQ_CACHE_CONTAINER"],
    session_nonce=os.environ["PQ_SESSION_NONCE"],
)
PY
  PQ_COMPILE_MOUNT=(
    -v "$PQ_COMPILE_CACHE_HOST:$PQ_COMPILE_CACHE_CONTAINER:rw"
  )
  PQ_VALIDATOR_GRAPH_ARGS=(
    --serve-log "$PQ_SERVE_LOG"
    --compile-cache "$PQ_COMPILE_CACHE_CONTAINER"
    --compile-cache-host-root "$PQ_COMPILE_CACHE_HOST"
    --compile-cache-preflight "$PQ_COMPILE_CACHE_PREFLIGHT"
    --session-nonce "$PQ_SESSION_NONCE"
  )
fi

PQ_MODEL_SHA="$(
  PQ_MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import os
from prismaquant.shipcard import compute_model_sha
print(compute_model_sha(os.environ["PQ_MODEL_DIR"]))
PY
)"
PQ_SERVED_MODEL="qwen38-rtx4090-${PQ_MODEL_SHA:0:32}-${PQ_SESSION_NONCE}"
PQ_CONTAINER_NAME="${CONTAINER_NAME:-pq-qwen38-rtx4090-fp8-cb-$SERVE_ARM}"
if docker container inspect "$PQ_CONTAINER_NAME" >/dev/null 2>&1; then
  echo "[rtx4090-serve] REFUSED: container name already exists: $PQ_CONTAINER_NAME" >&2
  exit 2
fi

if [[ "$SERVE_ARM" == graph ]]; then
  PQ_GRAPH_CONFIG="$(
    PQ_CACHE="$PQ_COMPILE_CACHE_CONTAINER" python3 - <<'PY'
import os
from prismaquant.rtx4090_graph_contract import compilation_config_json
print(compilation_config_json(cache_dir=os.environ["PQ_CACHE"]))
PY
  )"
fi

# Build the server process's exact closed Gridbook environment from the same
# policy functions used by the validator.  The entrypoint first unsets every
# allowlisted name (including PYTHONPATH) and then installs only these values.
if ! PQ_ALLOWLIST_TEXT="$(python3 - <<'PY'
from prismaquant.validate_rtx4090_fp8_cb import rtx4090_serve_environment_allowlist
print(*rtx4090_serve_environment_allowlist(), sep="\n")
PY
)"; then
  echo "[rtx4090-serve] REFUSED: could not resolve the closed server environment allowlist" >&2
  exit 2
fi
mapfile -t PQ_ENV_ALLOWLIST <<<"$PQ_ALLOWLIST_TEXT"
if ! PQ_EXPECTED_ENV_TEXT="$(
  PQ_RUNTIME_PIN="$GRIDBOOK_RUNTIME_PIN" python3 - <<'PY'
import json
import os
from pathlib import Path
from prismaquant.validate_rtx4090_fp8_cb import rtx4090_serve_environment

pin = json.loads(Path(os.environ["PQ_RUNTIME_PIN"]).read_text(encoding="utf-8"))
for name, value in rtx4090_serve_environment(pin).items():
    if "\n" in value:
        raise SystemExit(f"unsafe newline in environment value for {name}")
    print(f"{name}={value}")
PY
)"; then
  echo "[rtx4090-serve] REFUSED: could not resolve the exact pinned server environment" >&2
  exit 2
fi
mapfile -t PQ_EXPECTED_ENV <<<"$PQ_EXPECTED_ENV_TEXT"
if (( ${#PQ_ENV_ALLOWLIST[@]} == 0 || ${#PQ_EXPECTED_ENV[@]} == 0 )); then
  echo "[rtx4090-serve] REFUSED: resolved server environment is empty" >&2
  exit 2
fi
PQ_ALLOWLIST_CSV="$(IFS=,; echo "${PQ_ENV_ALLOWLIST[*]}")"
PQ_EXPECTED_NAMES=()
PQ_DOCKER_ENV=(
  -e "PQ_RTX_ALLOWLIST=$PQ_ALLOWLIST_CSV"
  -e "PQ_SERVE_ARM=$SERVE_ARM"
  -e "PQ_GRAPH_CONFIG=$PQ_GRAPH_CONFIG"
  -e "PQ_SERVED_MODEL=$PQ_SERVED_MODEL"
  -e "PYTHONDONTWRITEBYTECODE=1"
  -e "PYTHONNOUSERSITE=1"
)
for row in "${PQ_EXPECTED_ENV[@]}"; do
  name="${row%%=*}"
  value="${row#*=}"
  PQ_EXPECTED_NAMES+=("$name")
  PQ_DOCKER_ENV+=( -e "PQ_RTX_VALUE_${name}=$value" )
done
PQ_DOCKER_ENV+=( -e "PQ_RTX_EXPECTED_NAMES=$(IFS=,; echo "${PQ_EXPECTED_NAMES[*]}")" )

PQ_CONTAINER_ID="$(docker create --pull=never \
  --name "$PQ_CONTAINER_NAME" \
  --gpus "device=$PQ_GPU_UUID" \
  --ipc=host \
  --user 0:0 \
  --log-driver local --log-opt max-size=100m --log-opt max-file=3 \
  -p 127.0.0.1:8000:8000 \
  -v "$PQ_REPO_ROOT:/repo:ro" \
  -v "$MODEL_DIR:/model:ro" \
  "${PQ_WEIGHT_MOUNTS[@]}" \
  -v "$VLLM_RUNTIME_PIN:/vllm-runtime-pin.json:ro" \
  -v "$GRIDBOOK_RUNTIME_CONTRACT:/gridbook-runtime-contract.json:ro" \
  -v "$EVIDENCE_DIR:/evidence:rw" \
  "${PQ_COMPILE_MOUNT[@]}" \
  "${PQ_DOCKER_ENV[@]}" \
  --entrypoint bash "$SERVE_IMAGE" -c '
    set -euo pipefail
    IFS=, read -r -a allowlist <<<"$PQ_RTX_ALLOWLIST"
    for name in "${allowlist[@]}"; do
      unset "$name"
    done
    IFS=, read -r -a expected_names <<<"$PQ_RTX_EXPECTED_NAMES"
    for name in "${expected_names[@]}"; do
      source_name="PQ_RTX_VALUE_${name}"
      export "$name=${!source_name}"
      unset "$source_name"
    done
    serve_arm="$PQ_SERVE_ARM"
    served_model="$PQ_SERVED_MODEL"
    graph_config="$PQ_GRAPH_CONFIG"
    unset PQ_RTX_ALLOWLIST PQ_RTX_EXPECTED_NAMES allowlist expected_names name source_name
    unset PQ_SERVE_ARM PQ_SERVED_MODEL PQ_GRAPH_CONFIG
    arm_args=()
    if [[ "$serve_arm" == graph ]]; then
      arm_args=(--compilation-config "$graph_config")
    else
      arm_args=(--enforce-eager)
    fi
    PQ_RUNTIME_PRISMAQUANT_ROOT=/repo python3 -P \
      /repo/tools/prismaquant_source_bootstrap.py run-tool --source-root /repo \
      serve-fingerprint rtx4090-artifact-preflight \
      --model-dir /model \
      --runtime-contract /gridbook-runtime-contract.json \
      --out /run/prismaquant-rtx4090-content-receipt.json
    exec /usr/local/bin/vllm serve /model \
      --served-model-name "$served_model" \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --tokenizer-mode auto \
      --generation-config vllm \
      --quantization gridbook \
      --tensor-parallel-size 1 \
      --kv-cache-dtype fp8 \
      --kv-cache-memory-bytes 4294967296 \
      --max-model-len 32768 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 32768 \
      --no-enable-prefix-caching \
      --enable-chunked-prefill \
      --gpu-memory-utilization 0.95 \
      "${arm_args[@]}"
  ')"

PQ_KEEP_CONTAINER=0
cleanup() {
  if [[ "$PQ_KEEP_CONTAINER" == 0 ]]; then
    docker logs "$PQ_CONTAINER_ID" > "$PQ_SERVE_LOG" 2>&1 || true
    docker rm -f "$PQ_CONTAINER_ID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM HUP

docker start "$PQ_CONTAINER_ID" >/dev/null
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models \
      > "$EVIDENCE_DIR/models.json"; then
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$PQ_CONTAINER_ID" 2>/dev/null)" != true ]]; then
    echo "[rtx4090-serve] REFUSED: server exited before READY" >&2
    exit 1
  fi
  sleep 5
done
if ! curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
  echo "[rtx4090-serve] REFUSED: server did not become ready in 30 minutes" >&2
  exit 1
fi

# Force lazy native extensions resident before the post-run process snapshot.
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$PQ_SERVED_MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0.0,\"top_p\":1.0,\"seed\":0,\"n\":1,\"stream\":false}" \
  http://127.0.0.1:8000/v1/completions > "$EVIDENCE_DIR/warmup.json"

docker logs "$PQ_CONTAINER_ID" > "$PQ_SERVE_LOG" 2>&1
docker exec --workdir / "$PQ_CONTAINER_ID" env -u PYTHONPATH \
  PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PQ_RUNTIME_PRISMAQUANT_ROOT=/repo python3 -P \
  /repo/tools/prismaquant_source_bootstrap.py run-tool --source-root /repo \
  serve-fingerprint write \
  --out /evidence/serve_manifest.json \
  --image "$SERVE_IMAGE" \
  --artifact-dir /model \
  --base-url http://127.0.0.1:8000/v1 \
  --server-environment-profile rtx4090_fp8_cb \
  --vllm-runtime-pin /vllm-runtime-pin.json \
  --artifact-content-receipt "$PQ_ARTIFACT_CONTENT_RECEIPT_CONTAINER" \
  --rtx4090-runtime-contract /gridbook-runtime-contract.json \
  --attestation-phase post

python3 -m prismaquant.validate_rtx4090_fp8_cb \
  --model-dir "$MODEL_DIR" \
  --shipcard "$PQ_SHIPCARD" \
  --manifest "$PQ_MANIFEST" \
  --runtime-pin "$GRIDBOOK_RUNTIME_PIN" \
  --vllm-runtime-pin "$VLLM_RUNTIME_PIN" \
  --runtime-contract "$GRIDBOOK_RUNTIME_CONTRACT" \
  --expected-image "$SERVE_IMAGE" \
  --served-model "$PQ_SERVED_MODEL" \
  --arm "$SERVE_ARM" \
  "${PQ_VALIDATOR_GRAPH_ARGS[@]}" \
  --base-url http://127.0.0.1:8000/v1 \
  > "$EVIDENCE_DIR/rtx4090_endpoint_contract.json"

PQ_KEEP_CONTAINER=1
trap - EXIT INT TERM HUP
echo "[rtx4090-serve] READY and strict $SERVE_ARM receipt verified"
echo "[rtx4090-serve] endpoint=http://127.0.0.1:8000/v1 model=$PQ_SERVED_MODEL"
echo "[rtx4090-serve] container=$PQ_CONTAINER_NAME evidence=$EVIDENCE_DIR"
