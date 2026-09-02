#!/usr/bin/env bash
# GPU qualification for the gridbook 0.8.9 release (branch pb-fp8): the
# default flip of persistent-B + both GEMV selectors. Extends the d9dfe53
# FP8 qual with the selector/dispatch CPU suites and the GEMV CUDA suites,
# since 0.8.9 changes what an UNSET flag serves.
#
# Phases:
#   1. JIT build of prismaquant_cb_ext at ABI schema 2 (the schema bump forces
#      a rebuild; a stale ABI-1 ext is refused by cuda_ext, not silently used).
#   2. tests/test_cb_moe_persistent_b_fp8.py  — bitwise decode identity vs two
#      independent references, incl. the all-256-e4m3-bytes gate (NaN payload
#      equality adjudicated empirically if it trips), DSv4-shape smoke,
#      cfg-eligibility-vs-reality at k=48, graph capture, rejections.
#   3. tests/test_cb_moe_persistent_b.py      — FP4 regression: the extractor
#      refactor + pick_cfg extraction must stay bit-identical.
#   4. tests/test_moe_persistent_b_lane.py    — lane predicates (CPU, cheap).
#   5. scripts/bench_moe_persistent_b_fp8.py  — DSv4 shape E=256 h4096 i2048,
#      k=28 (the built body's FP8-CB rung), topk 6, correctness tol then
#      timing vs the expand+bridge path it replaces.
#
# Files are run individually (the full suite has order-dependent failures —
# see memory suite_has_order_dependent_failures).
set -uo pipefail

QUAL=/home/rob/dq-runs/gridbook-fp8-qual
GB=/home/rob/gridbook
VENV=/home/rob/dq-runs/venvs/prismaquant-cu130
LOG="$QUAL/qual-089-$(date +%Y%m%d-%H%M%S).log"
VERDICT="$QUAL/VERDICT-089"
GPU_LOCK=/home/rob/dq-runs/gpu.lock

exec > >(tee -a "$LOG") 2>&1
echo "[qual] start $(date -Is)  log=$LOG"

# ---- refuse to contend with the served A/B ---------------------------------
live=$(docker ps --format '{{.Names}}' | grep -E '^pb-ab-' || true)
[[ -z "$live" ]] || { echo "[qual] REFUSE: A/B containers live: $live"; exit 1; }
if pgrep -f 'run_pb_ab_92gb.sh|run_pb_ab_clean' >/dev/null 2>&1; then
  echo "[qual] REFUSE: a pb-ab driver is still running"; exit 1
fi
# Hold the host-wide GPU mutex for the whole run so nothing else launches.
exec 9<>"$GPU_LOCK"
flock -n -x 9 || { echo "[qual] REFUSE: GPU lock held"; exit 1; }

# ---- environment (memory: host-nvcc-cuda-home-broken) ----------------------
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export PRISMAQUANT_CB_EXT_DIR=/home/rob/dq-runs/.cb-ext-cache
# The bench BASELINE (expand+bridge) JIT-builds the grouped-BF16 CUTLASS
# module; this venv has no vLLM wheel to source headers from, so point the
# documented override at the local CUTLASS checkout (fails closed if wrong).
export PRISMAQUANT_CUTLASS_INCLUDE=/home/rob/cutlass/include
export MAX_JOBS=6
# user-site stays visible: pytest lives ONLY in ~/.local on this box (the venv
# ships no pip); the no-user-site discipline is publication tooling, not tests.
unset PYTHONPATH || true
cd "$GB"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "pb-fp8" ]] \
  || { echo "[qual] REFUSE: gridbook not on pb-fp8"; exit 1; }
git rev-parse HEAD
dirty=$(git status --porcelain --untracked-files=all)
[[ -z "$dirty" ]] || { echo "[qual] REFUSE: gridbook dirty:"; echo "$dirty"; exit 1; }

PY="$VENV/bin/python"
fail=0
phase() { echo; echo "=== [$(date -Is)] $* ==="; }

phase "1/6 JIT build (ABI schema 2)"
"$PY" - <<'EOF' || fail=1
import time
t0 = time.time()
from gridbook import cuda_ext
# require_* fails closed and already asserts the full ABI-2 symbol list
# (incl. the three FP8 entries) against the freshly built module.
ext = cuda_ext.require_moe_persistent_b_ext("fp8 qualification")
print(f"build+load {time.time()-t0:.1f}s; abi={getattr(ext, '__gridbook_jit_abi_schema__', '?')}")
for sym in ("cb_moe_persistent_b_prefill_fp8",
            "cb_moe_persistent_b_decode_fp8",
            "cb_moe_persistent_b_fp8_cfg_eligible"):
    assert hasattr(ext, sym), f"missing symbol {sym}"
    print(f"symbol OK: {sym}")
EOF
[[ $fail -eq 0 ]] || { echo "BUILD FAILED" > "$VERDICT"; exit 1; }

phase "2/6 FP8 CUDA tests"
"$PY" -m pytest tests/test_cb_moe_persistent_b_fp8.py -q -x || fail=2

phase "3/6 FP4 regression tests"
"$PY" -m pytest tests/test_cb_moe_persistent_b.py -q || fail=3

phase "4/6 lane + selector + dispatch-policy tests"
for t in test_moe_persistent_b_lane test_moe_persistent_b_d2r_lane \
         test_cb_gemv_select test_moe_native_policy; do
  "$PY" -m pytest "tests/$t.py" -q || fail=4
done

phase "5/6 GEMV CUDA suites (v2 dictionary + main incl. FP8 whole-row)"
# These suites use PrismaQuant's real two-tier encoder as the reference
# (test_cb_gemv_v2_cuda importorskips prismaquant.nvfp4_cb_formats, and 14
# test_cuda_gemv groups do the same) -- without the release checkout on the
# path they "pass" as an all-skip that validates nothing. Provide it, and
# refuse any file that ends with zero passed tests.
for t in test_cb_gemv_v2_cuda test_cuda_gemv test_routed_fp8_v2_cuda; do
  out=$(PYTHONPATH=/home/rob/pq-dsv4flash-release "$PY" -m pytest "tests/$t.py" -q 2>&1); rc=$?
  echo "$out" | tail -4
  [[ $rc -eq 0 ]] || fail=5
  grep -qE '[0-9]+ passed' <<<"$out" || { echo "[qual] $t: ZERO tests passed (all-skip) -> FAIL"; fail=5; }
done

phase "6/6 whole-operator bench (DSv4 shape, k=28, topk=6) — in the pinned serving container"
# The whole-operator arms use vLLM's compiled activation op and the baseline
# builds against vLLM's bundled CUTLASS — neither exists in the host venv, so
# the bench runs where they do: the pinned serving image (which also carries
# nvcc; it built the fused ext earlier tonight). PYTHONPATH puts the pb-fp8
# checkout ahead of the image's installed 0.8.8 wheel; PYTHONSAFEPATH keeps
# sys.path[0] (the scripts dir) from shadowing anything. Fresh ext cache so
# the ABI-2 kernel and the baseline both JIT inside the container.
SERVING_IMAGE=eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869
if [[ $fail -eq 0 ]]; then
  mkdir -p "$QUAL/bench-ext-cache/home"
  docker run --rm --pull=never --name gb-fp8-bench \
    --network none --gpus all --ipc=host --user 0:0 \
    -v /home/rob/gridbook:/home/rob/gridbook:ro \
    -v "$QUAL/bench-ext-cache:/bench-ext:rw" \
    -e PYTHONPATH=/home/rob/gridbook \
    -e PYTHONSAFEPATH=1 -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PRISMAQUANT_CB_EXT_DIR=/bench-ext \
    -e HOME=/bench-ext/home \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint python3 "$SERVING_IMAGE" \
    /home/rob/gridbook/scripts/bench_moe_persistent_b_fp8.py || fail=6
else
  echo "[qual] SKIP bench: earlier phase failed ($fail)"
fi

phase "verdict"
if [[ $fail -eq 0 ]]; then
  echo "PASS $(git rev-parse HEAD) $(date -Is)" > "$VERDICT"
  echo "[qual] PASS"
else
  echo "FAIL phase=$fail $(git rev-parse HEAD) $(date -Is)" > "$VERDICT"
  echo "[qual] FAIL at phase $fail"
fi
exit $fail
