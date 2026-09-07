#!/usr/bin/env bash
set -euo pipefail
# CPU control-flow checks may use the fleet's CPU torch. GPU qualification uses
# the separately pinned production container and original calibration runtime.
if [ "$(uname -m)" != x86_64 ]; then
  exec bash experiments/lfm_stream_cpu_pytest.sh "$@"
fi
pq_python=/home/rob/venvs/pb-cpu/bin/python
pq_tag="$($pq_python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
pq_target="/home/rob/.cache/prismaquant/pq322-cpu-$pq_tag"
"$pq_python" -m pip install --disable-pip-version-check --target "$pq_target" \
  'transformers==5.16.1' 'pytest==9.0.2' 'pytest-xdist==3.8.0' \
  'pydantic==2.12.5' 'loguru==0.7.3' 'psutil==7.2.2'
# Keep the fleet's CPU torch; the remaining runtime dependencies are above.
"$pq_python" -m pip install --disable-pip-version-check --target "$pq_target" \
  --no-deps 'compressed-tensors==0.18.0'
export PYTHONPATH="$PWD:$pq_target:/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$pq_python" -c 'import sys, torch, transformers, pytest; print({"python":sys.version, "torch":torch.__version__, "transformers":transformers.__version__, "pytest":pytest.__version__}, flush=True)'
"$pq_python" -c 'import ast, pathlib; paths=["prismaquant/tessera_joint_aura.py", "experiments/pq322_anchor_wire_qualification.py", "tests/test_tessera_joint_aura.py"]; [compile(pathlib.Path(p).read_text(),p,"exec") for p in paths]; print("Touched Python modules compile",flush=True)'
exec "$pq_python" -m pytest "$@"
