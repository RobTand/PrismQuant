#!/usr/bin/env bash
# Direct post-allocation handoff: selected assignment -> unreleasable artifact.
# This intentionally bypasses the stock pipeline and retained-menu rebuild.

set -euo pipefail

PQ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PQ_REPO_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the original dense Qwen3.8 checkpoint}"
: "${LAYER_CONFIG:?Set LAYER_CONFIG to the completed allocator layer_config.json}"
: "${CB_COL_WEIGHTS:?Set CB_COL_WEIGHTS to the exact render-identity-bound pickle}"
: "${GRIDBOOK_PRODUCER_RUNTIME_CONTRACT:?Set GRIDBOOK_PRODUCER_RUNTIME_CONTRACT to the Gridbook v11 compile_only contract}"
: "${CB_OUT:?Set CB_OUT to a fresh validation-only output directory}"

exec python3 -m prismaquant.rtx4090_validation_export \
  --model-dir "$MODEL_PATH" \
  --layer-config "$LAYER_CONFIG" \
  --col-weights "$CB_COL_WEIGHTS" \
  --runtime-contract "$GRIDBOOK_PRODUCER_RUNTIME_CONTRACT" \
  --out "$CB_OUT" \
  --shard-bytes "${EXPORT_SHARD_BYTES:-1073741824}"
