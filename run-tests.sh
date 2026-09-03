#!/bin/bash
# Run the repo's tests. Use: ./run-tests.sh tests/test_foo.py [tests/test_bar.py ...]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
export TMPDIR="$PWD/.claude/tmp" TRITON_CACHE_DIR="$PWD/.claude/triton" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
export PYTHONPATH=.
for i in 0 1 2 3 4 5; do
  exec 9>>/home/rob/tmp/arb/fleetlocks/slot.$i
  flock -n 9 && break
done
flock 9
exec taskset -c 5-9,15-19 /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q --no-header -p no:cacheprovider "$@"
