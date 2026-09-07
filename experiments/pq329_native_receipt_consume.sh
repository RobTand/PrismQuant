#!/usr/bin/env bash
set -euo pipefail
# The existing qualified CPU container and frozen producer dependencies.
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
input_root=/mnt/shared/tessera-measurements/first-model-20260907/inputs
exec docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES -e NVIDIA_VISIBLE_DEVICES=void \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e XDG_CACHE_HOME=/tmp/pq-receipt-cpu-cache -e HF_HOME=/tmp/pq-receipt-cpu-hf \
  -e PRISMABUILD_CONTAINER_OWNER \
  -e "PYTHONPATH=/workspace:$input_root/tessera-382a1a97/src:$input_root/container-compatible-deps" \
  "$image" -c 'import pathlib, runpy; [compile(pathlib.Path(p).read_text(), p, "exec") for p in ("prismaquant/native_moe_panel.py", "tests/test_native_moe_panel.py")]; print("Touched Python modules compile", flush=True); runpy.run_module("experiments.pq309_native_moe_panel", run_name="__main__")' consume "$@"
