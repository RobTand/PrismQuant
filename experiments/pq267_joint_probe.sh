#!/usr/bin/env bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
prisma_snapshot_commit=$(git rev-parse HEAD)
docker run --rm --gpus all --ipc=host --network=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/pq267-cache -e TORCH_EXTENSIONS_DIR=/tmp/pq267-extensions \
  -e HF_HOME=/tmp/pq267-hf -e HF_MODULES_CACHE=/tmp/pq267-modules \
  -e "PRISMAQUANT_IDENTITY_GIT_COMMIT=$prisma_snapshot_commit" \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -m experiments.pq267_native_panel probe \
  --inputs /mnt/shared/tessera-measurements/pq267-native-20260907/dense-r1792-01/inputs.v2.json \
  --inputs-sha256 c9d9f5a886a2ca3ff4e919349d005356b46d231eee04640af3abb3dfb862a756 \
  --out /mnt/shared/tessera-measurements/pq267-native-20260907/joint-02
