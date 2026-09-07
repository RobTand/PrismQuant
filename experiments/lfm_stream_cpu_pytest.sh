#!/usr/bin/env bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
exec docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES -e NVIDIA_VISIBLE_DEVICES=void \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/pq-stream-cpu-cache -e HF_HOME=/tmp/pq-stream-cpu-hf \
  -e PRISMABUILD_CONTAINER_OWNER -e TESSERA_REPO \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps:/mnt/shared/tessera-measurements/pq313-packed-joint-20260907/pytest-deps:/mnt/shared/tessera-measurements/lfm-streamed-parity-20260907/pytest-deps" \
  "$image" -m pytest "$@"
