#!/bin/bash
# Run the repo's tests. Use: ./run-tests.sh tests/test_foo.py [tests/test_bar.py ...]
export TMPDIR="$PWD/.claude/tmp" TRITON_CACHE_DIR="$PWD/.claude/triton" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
export PYTHONPATH=.
exec /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q --no-header -p no:cacheprovider "$@"
