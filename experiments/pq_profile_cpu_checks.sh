#!/usr/bin/env bash
set -euo pipefail
pq_profile_deps=$(mktemp -d /tmp/pq-profile-checks.XXXXXX)
trap 'rm -rf "$pq_profile_deps"' EXIT
pq_profile_python=/home/rob/venvs/pb-cpu/bin/python
"$pq_profile_python" -m pip install --disable-pip-version-check --quiet --target "$pq_profile_deps" \
  'py-spy==0.4.2' 'pytest==9.0.2' 'pytest-xdist==3.8.0'
export PYTHONPATH="$PWD:$pq_profile_deps${PYTHONPATH:+:$PYTHONPATH}"
export PQ_TEST_PY_SPY="$pq_profile_deps/bin/py-spy"
"$pq_profile_python" -m pytest -n 2 tests/test_profile_command.py -q -p no:cacheprovider
"$pq_profile_python" -m compileall -q experiments/profile_command.py tests/test_profile_command.py
