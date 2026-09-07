#!/usr/bin/env bash
set -euo pipefail
image=sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
# This image is currently present only on Sparky; PB submission declares --here.
docker run --rm --gpus all --ipc=host --network=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared \
  -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/pq267-cache -e TORCH_EXTENSIONS_DIR=/tmp/pq267-extensions \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -m experiments.pq267_native_panel prepare \
  --plan experiments/protocols/pq267-lfm-dense-20260907.json \
  --plan-sha256 5884af5c07be34ae37d8fef42aa580f098aa427dcadf261c6b9c2a92df0f6ace \
  --out /mnt/shared/tessera-measurements/pq267-native-20260907/dense-r1792-01
