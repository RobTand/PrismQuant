#!/usr/bin/env bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
exec docker run --cap-add SYS_PTRACE --security-opt seccomp=unconfined --rm --gpus all --ipc=host --network=host --user "$(id -u):$(id -g)" \
  -v /home/rob/venvs/pq-release/bin/py-spy:/py-spy:ro -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/pq-anchor-qualification-cache -e TORCH_EXTENSIONS_DIR=/tmp/pq-anchor-qualification-ext \
  -e HF_HOME=/tmp/pq-anchor-qualification-hf -e HF_MODULES_CACHE=/tmp/pq-anchor-qualification-modules \
  -e PRISMABUILD_CONTAINER_OWNER -e PRISMAQUANT_PROD_ACT_SCALES=0 \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -m experiments.profile_command --output "$1" --profiler /py-spy --rate 50 -- python3 -m experiments.reader_unpack_ab "${@:2}"
