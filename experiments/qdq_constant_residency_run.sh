#!/usr/bin/env bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
entry=experiments.qdq_constant_residency
if [[ "${1:-}" == --source-capture ]]; then
  entry=experiments.qdq_projection_source_capture
  shift
fi
exec docker run --rm --gpus all --ipc=host --network=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PRISMABUILD_CONTAINER_OWNER -e PRISMAQUANT_PROD_ACT_SCALES=0 \
  -e XDG_CACHE_HOME=/tmp/qdq-cache -e HF_HOME=/tmp/qdq-hf -e HF_MODULES_CACHE=/tmp/qdq-modules \
  -e "QDQ_ENTRY_MODULE=$entry" \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -c 'python3 -m "$QDQ_ENTRY_MODULE" "$@"' bash "$@"
