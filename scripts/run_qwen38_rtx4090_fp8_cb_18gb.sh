#!/usr/bin/env bash
# Build the strict dense Qwen3.8-27B context-first artifact.  A known-good
# CUDA build container (including a Spark) may run this producer workflow,
# but a Spark is compile-only evidence for the serving lane: it cannot create
# the physical RTX 4090/SM89 endpoint or graph receipt required for release.

set -euo pipefail

PQ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PQ_REPO_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the dense Qwen3.8-27B BF16 checkpoint}"
: "${WORK_DIR:?Set WORK_DIR to a fresh output directory}"
: "${GRIDBOOK_PRODUCER_RUNTIME_CONTRACT:?Set GRIDBOOK_PRODUCER_RUNTIME_CONTRACT to the qualified Gridbook v11 JSON}"
: "${RTX4090_BUILD_DISPOSITION:=strict}"
if [[ "$RTX4090_BUILD_DISPOSITION" != strict \
   && "$RTX4090_BUILD_DISPOSITION" != validation_only ]]; then
  echo "[rtx4090-build] REFUSED: RTX4090_BUILD_DISPOSITION must be strict or validation_only" >&2
  exit 2
fi

if [[ "$MODEL_PATH" != /* || ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[rtx4090-build] REFUSED: MODEL_PATH must be an absolute HF directory with config.json" >&2
  exit 2
fi
if [[ "$WORK_DIR" != /* || "$WORK_DIR" == / ]]; then
  echo "[rtx4090-build] REFUSED: WORK_DIR must be a dedicated absolute directory" >&2
  exit 2
fi
if [[ "$GRIDBOOK_PRODUCER_RUNTIME_CONTRACT" != /* \
   || ! -f "$GRIDBOOK_PRODUCER_RUNTIME_CONTRACT" ]]; then
  echo "[rtx4090-build] REFUSED: the Gridbook producer contract must be an existing absolute file" >&2
  exit 2
fi

# This is the complete strict producer menu: the K%4 ladder and only the two
# policy terminals.  Keep it literal so a launch record is independently
# readable and fail if it ever diverges from the policy module.
RTX4090_FORMATS="FP8_CB_K4,FP8_CB_K8,FP8_CB_K12,FP8_CB_K16,FP8_CB_K20,FP8_CB_K24,FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48,FP8_E4M3,BF16"
PQ_MODEL_PATH="$MODEL_PATH" \
PQ_GRIDBOOK_CONTRACT="$GRIDBOOK_PRODUCER_RUNTIME_CONTRACT" \
PQ_FORMATS="$RTX4090_FORMATS" \
PQ_BUILD_DISPOSITION="$RTX4090_BUILD_DISPOSITION" \
python3 - <<'PY'
import json
import os
from pathlib import Path

from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_QWEN38_FORMAT_MENU,
    require_rtx4090_compile_only_runtime_contract,
    require_rtx4090_runtime_contract,
    validate_qwen38_dense_config,
    validate_rtx4090_format_menu,
)

model = Path(os.environ["PQ_MODEL_PATH"])
contract_path = Path(os.environ["PQ_GRIDBOOK_CONTRACT"])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
menu = validate_rtx4090_format_menu(os.environ["PQ_FORMATS"].split(","))
if menu != RTX4090_QWEN38_FORMAT_MENU:
    raise SystemExit(
        "REFUSED: launcher menu differs from the exact strict RTX4090 policy"
    )
validate_qwen38_dense_config(config, where="RTX4090 build source")
resolver = (
    require_rtx4090_compile_only_runtime_contract
    if os.environ["PQ_BUILD_DISPOSITION"] == "validation_only"
    else require_rtx4090_runtime_contract
)
resolver(contract, menu, where="RTX4090 build Gridbook contract")
PY

export CB_OUT="$WORK_DIR/exported_fp8_cb"
if [[ -e "$CB_OUT" ]]; then
  echo "[rtx4090-build] REFUSED: strict export output already exists: $CB_OUT" >&2
  exit 2
fi

mkdir -p "$WORK_DIR" "$WORK_DIR/logs" "$WORK_DIR/tmp"

export MODEL_PATH WORK_DIR GRIDBOOK_PRODUCER_RUNTIME_CONTRACT
export TMPDIR="$WORK_DIR/tmp"
export EXPORT_CONTAINER=nvfp4_cb
if [[ "$RTX4090_BUILD_DISPOSITION" == validation_only ]]; then
  export TARGET_PROFILE=qwen38_rtx4090_fp8_cb_validation_only
else
  export TARGET_PROFILE=qwen38_rtx4090_fp8_cb
fi
export FORMATS="$RTX4090_FORMATS"

# The allocator prices every Linear independently under an exact decimal-byte
# constraint.  The reserve covers non-tensor files during selection; the
# strict exporter remains authoritative and refuses a recursive directory
# inventory above exactly 18,000,000,000 bytes.
export TARGET_DISK_GB=18
export ARTIFACT_OVERHEAD_RESERVE_BYTES=268435456
export TARGET_BITS=5.5
# Under a byte ceiling, measured held-out KL chooses among the feasible
# Pareto assignments.  Do not override the pipeline back to surrogate-only.
export SELECTION_MODE=validated-surrogate

# Real frontier KL consumes the imatrix-weighted format-menu renders through
# the existing ProductionWeightCache with required resident prefetch.  The CB
# serializer then re-encodes the selected assignment through the same bound
# codec; selected-assignment recache remains off because export cannot consume
# it.  There is no parallel cache or silent export-time NVMe fallback.
export COST_MODE=aura
export AURA_COST_STREAMING=1
export AURA_COST_DTYPE=bfloat16
export AURA_COST_CHECKPOINT_DIR="$WORK_DIR/artifacts/aura_checkpoints"
export PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE="${AURA_COST_CHECKPOINT_DIR}/streamed_model_identity.json"
export PRODUCTION_CACHE=1
export PRODUCTION_RECACHE=0
export PRODUCTION_CACHE_PREFETCH=require
export PRODUCTION_CACHE_PREFETCH_WORKERS=4
export PREFETCH_LOOKAHEAD=auto
export PREFETCH_WORKERS=auto
export PREFETCH_MIN_AVAILABLE_GB=auto

# Fixed calibration/render identity.  Column weights are harvested from the
# same activation cache used by the cost renderer and final weighted export.
export NSAMPLES=32
export SEQLEN=1024
export ACTIVATION_ROWS_LIMIT=1024
export CALIBRATION_MODALITY=text-only
# Cost, frontier-KL renders, and export all use the probe's full-corpus
# act_sq_sum/n_tokens_seen imatrix rather than recomputing E[x^2] from the
# row-capped activation replay cache.
export CB_IMATRIX_SOURCE=probe
export VISUAL_FORMAT=BF16
# The output head is an immutable BF16 auxiliary region: it is counted in the
# exact 18GB artifact inventory but never diluted into quantizable-body bpp.
export LM_HEAD_FORMAT=BF16
export MTP_FORMAT=BF16
export CB_CODEBOOK_SOURCE_SCOPE=none
export CB_CODEBOOK_SOURCE=lattice
export CB_CODEBOOK_ITERS=4
export CB_CODEBOOK_SEED=0
export CB_SCALE_SWEEP=1
export CB_SCALE_SWEEP_SCOPE=fp8
# FP8-CB is W8A8 and has no NVFP4 static-activation scalar.  This setting is
# consumed by every cost/cache/export process and changes the serialized
# context to the existing no-activation contract.
export CB_ACTIVATION_SCOPE=none
export CB_SCALE_CODING=v1
export CB_LADDER_INTERP=0
export PRISMAQUANT_CB_LDLQ=0
export PRISMAQUANT_CB_MINCHAIN=0
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export EXPORT_STREAMING=auto
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[rtx4090-build] dense Qwen3.8-27B ${RTX4090_BUILD_DISPOSITION} FP8-CB build"
echo "[rtx4090-build] menu=$FORMATS"
echo "[rtx4090-build] whole-artifact ceiling=18000000000 bytes"
echo "[rtx4090-build] work_dir=$WORK_DIR"

bash "$PQ_REPO_ROOT/prismaquant/run-pipeline.sh"

PQ_MODEL_DIR="$CB_OUT" \
PQ_GRIDBOOK_CONTRACT="$GRIDBOOK_PRODUCER_RUNTIME_CONTRACT" \
PQ_BUILD_DISPOSITION="$RTX4090_BUILD_DISPOSITION" \
python3 - <<'PY'
import json
import os
from pathlib import Path

contract = json.loads(
    Path(os.environ["PQ_GRIDBOOK_CONTRACT"]).read_text(encoding="utf-8")
)
if os.environ["PQ_BUILD_DISPOSITION"] == "validation_only":
    from prismaquant.validate_rtx4090_fp8_cb_validation_only import (
        validate_rtx4090_validation_only_artifact,
    )
    binding = validate_rtx4090_validation_only_artifact(
        os.environ["PQ_MODEL_DIR"], runtime_contract=contract
    )
else:
    from prismaquant.validate_rtx4090_fp8_cb import (
        validate_rtx4090_artifact_metadata,
    )
    binding = validate_rtx4090_artifact_metadata(
        os.environ["PQ_MODEL_DIR"], runtime_contract=contract
    )
print(
    f"[rtx4090-build] {os.environ['PQ_BUILD_DISPOSITION']} artifact verified: "
    f"{binding['artifact_bytes']} bytes, model_sha={binding['model_sha']}"
)
PY
