#!/usr/bin/env bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
run_root=/mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection
mode=$1
shift
gpu_args=()
if [[ "$mode" != build && "$mode" != policy_tests ]]; then
  gpu_args=(--gpus all)
fi
exec docker run --rm "${gpu_args[@]}" --ipc=host --network=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e MAX_JOBS=4 -e TORCH_CUDA_ARCH_LIST=12.1 -e "TORCH_EXTENSIONS_DIR=$run_root/extensions" \
  -e XDG_CACHE_HOME=/tmp/pq-reduction-cache -e HF_HOME=/tmp/pq-reduction-hf \
  -e PRISMABUILD_CONTAINER_OWNER -e PRISMAQUANT_PROD_ACT_SCALES=0 \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -m "experiments.joint_projection_reduce_$mode" "$@"
