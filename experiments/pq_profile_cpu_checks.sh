#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -m)" != x86_64 ] && [ "${PQ_PROFILE_CONTAINER:-0}" != 1 ]; then
  image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
  exec docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint bash \
    -e PQ_PROFILE_CONTAINER=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e CUDA_VISIBLE_DEVICES -e NVIDIA_VISIBLE_DEVICES=void \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e PRISMABUILD_CONTAINER_OWNER \
    "$image" experiments/pq_profile_cpu_checks.sh
fi
pq_profile_deps=$(mktemp -d /tmp/pq-profile-checks.XXXXXX)
trap 'rm -rf "$pq_profile_deps"' EXIT
pq_profile_python=/home/rob/venvs/pb-cpu/bin/python
if [ "${PQ_PROFILE_CONTAINER:-0}" = 1 ]; then pq_profile_python=python3; fi
"$pq_profile_python" -m pip install --disable-pip-version-check --quiet --target "$pq_profile_deps" \
  'py-spy==0.4.2' 'pytest==9.0.2' 'pytest-xdist==3.8.0'
export PYTHONPATH="$PWD:$pq_profile_deps:/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97/src${PYTHONPATH:+:$PYTHONPATH}"
export PQ_TEST_PY_SPY="$pq_profile_deps/bin/py-spy"
"$pq_profile_python" -m pytest -n 2 tests/test_profile_command.py tests/test_tessera_joint_aura.py tests/test_tessera_reader_namespace.py -q -p no:cacheprovider
"$pq_profile_python" -X "pycache_prefix=$pq_profile_deps/pycache" -m compileall -q experiments/profile_command.py experiments/pq_joint_profile_entry.py prismaquant/tessera_joint_aura.py tests/test_profile_command.py tests/test_tessera_joint_aura.py
