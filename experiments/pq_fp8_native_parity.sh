#!/usr/bin/env bash
set -euo pipefail
# Replay PQ arithmetic in the immutable original source-reference environment.
image=sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f
root=/mnt/shared/tessera-measurements/first-model-20260907
module=${PQ_FP8_DIAGNOSTIC_MODULE:-experiments.pq_fp8_native_parity}
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace \
  --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e "PYTHONPATH=.:$root/inputs/tessera-382a1a97/src:$root/inputs/container-compatible-deps" \
  "$image" -m "$module" "$@"
