#!/usr/bin/env bash
# Sealed BF16 GLM-5.3 importance probe through the explicit pread backend.
set -euo pipefail
set -o pipefail
umask 002

RUN_ROOT="${PQ_GLM_PREAD_RUN_ROOT:?set PQ_GLM_PREAD_RUN_ROOT}"
MODEL="${PQ_GLM_PREAD_MODEL:-/mnt/shared/models/GLM-5.3-Flash-BF16}"
DATASET="${PQ_GLM_PREAD_DATASET:-/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
REPO="${PQ_GLM_PREAD_REPO:?set PQ_GLM_PREAD_REPO to the immutable worktree}"
HELPERS="${PQ_GLM_PREAD_HELPERS:-/home/rob/dq-runs/glm53-flash}"
VENV="${PQ_GLM_PREAD_VENV:-/home/rob/dq-runs/venvs/prismaquant-tf516}"
LIBDIR="${PQ_GLM_PREAD_LIBDIR:-/usr/lib/aarch64-linux-gnu/nvshmem/13}"
GPU_LOCK="${PQ_GLM_PREAD_GPU_LOCK:-/home/rob/dq-runs/gpu.lock}"
EXPECTED_COMMIT="${PQ_GLM_PREAD_COMMIT:?set PQ_GLM_PREAD_COMMIT}"

ARTIFACTS="$RUN_ROOT/artifacts"
WORK="$RUN_ROOT/work"
ACT="$RUN_ROOT/act"
TELEMETRY="$RUN_ROOT/telemetry"
NETDATA="$TELEMETRY/netdata"
TMPROOT="$RUN_ROOT/tmp"
CACHE="$RUN_ROOT/cache"
PROBE="$ARTIFACTS/probe.pkl"
HEALTH="$RUN_ROOT/probe_health.json"
LOG="$RUN_ROOT/probe.log"
PROFILE="$RUN_ROOT/profile_pread.speedscope.json"
PROC_IO="$TELEMETRY/proc_io.csv"
WINDOW="$TELEMETRY/probe_window.json"
RECEIPT="$RUN_ROOT/receipt.json"

for path in "$MODEL" "$MODEL/config.json" \
  "$MODEL/model.safetensors.index.json" "$DATASET" "$REPO" \
  "$HELPERS/probe_telemetry.py" "$HELPERS/timestamp_stream.py" \
  "$HELPERS/capture_probe_netdata.py" "$HELPERS/probe_health_gate.py" \
  "$VENV/bin/python" "$VENV/bin/py-spy" "$LIBDIR/libnvshmem_host.so.3" \
  "$GPU_LOCK"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "repo commit does not match PQ_GLM_PREAD_COMMIT" >&2
  exit 2
}
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "repo worktree is dirty: $REPO" >&2
  exit 2
}
for path in "$PROBE" "$HEALTH" "$LOG" "$PROFILE" "$PROC_IO" \
  "$WINDOW" "$RECEIPT"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite: $path" >&2; exit 4; }
done

mkdir -p "$ARTIFACTS" "$WORK" "$ACT" "$TELEMETRY" "$NETDATA" \
  "$TMPROOT" "$CACHE/torchinductor" "$CACHE/triton" "$CACHE/pycache" \
  "$CACHE/hf"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$LIBDIR"
export TMPDIR="$TMPROOT"
export XDG_CACHE_HOME="$CACHE"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache"
export HF_HOME="$CACHE/hf"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PRISMAQUANT_SAFETENSORS_BACKEND=pread
export PRISMAQUANT_PREFETCH_DELIVERY=1
export PRISMAQUANT_LAYER_READ_THREADS="${PRISMAQUANT_LAYER_READ_THREADS:-4}"

exec 9<>"$GPU_LOCK"
flock -n -x 9 || {
  echo "GPU mutex is held; refusing to overlap another campaign" >&2
  exit 3
}
mapfile -t GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d'
)
if (( ${#GPU_PIDS[@]} )); then
  echo "GPU has foreign compute clients: ${GPU_PIDS[*]}" >&2
  exit 3
fi
MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( MEM_AVAILABLE_KIB >= 100 * 1024 * 1024 )) || {
  echo "need at least 100 GiB MemAvailable; found $MEM_AVAILABLE_KIB KiB" >&2
  exit 3
}

"$VENV/bin/python" "$HELPERS/probe_telemetry.py" \
  --output "$PROC_IO" --parent-pid "$$" \
  --match prismaquant.incremental_probe --interval 0.5 &
TELEMETRY_PID=$!
TELEMETRY_STOPPED=0
stop_telemetry() {
  if [[ "$TELEMETRY_STOPPED" == 0 ]]; then
    kill -INT "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_STOPPED=1
  fi
}
trap stop_telemetry EXIT INT TERM

START_EPOCH="$(date +%s)"
START_ISO="$(date -Ins)"
set +e
"$VENV/bin/py-spy" record --format speedscope --output "$PROFILE" \
  --rate 2 --threads --idle --full-filenames -- \
  "$VENV/bin/python" -u -m prismaquant.incremental_probe \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --nsamples 8 --seqlen 512 \
    --device cuda --dtype bf16 \
    --output "$PROBE" \
    --activation-cache-dir "$ACT" \
    --work-dir "$WORK" \
    --layers-per-shard auto \
    --prefetch-lookahead auto \
    --prefetch-workers 1 \
    --prefetch-min-available-gb auto \
    --activation-rows-limit 256 \
    --calibration-modality text-only \
    --emit-marginals \
    --unified-sweep \
    2>&1 | "$VENV/bin/python" -u "$HELPERS/timestamp_stream.py" | tee "$LOG"
PIPE_STATUSES=("${PIPESTATUS[@]}")
set -e
END_EPOCH="$(date +%s)"
END_ISO="$(date -Ins)"
stop_telemetry
trap - EXIT INT TERM

"$VENV/bin/python" - "$WINDOW" "$START_EPOCH" "$END_EPOCH" \
  "$START_ISO" "$END_ISO" "${PIPE_STATUSES[*]}" <<'PY'
import json, os, pathlib, sys
path, start, end, start_iso, end_iso, statuses = sys.argv[1:]
tmp = pathlib.Path(path + f".tmp-{os.getpid()}")
tmp.write_text(json.dumps({
    "start_epoch": int(start), "end_epoch": int(end),
    "start_iso": start_iso, "end_iso": end_iso,
    "pipeline_statuses": [int(x) for x in statuses.split()],
}, indent=2) + "\n")
os.replace(tmp, path)
PY
[[ "${PIPE_STATUSES[*]}" == "0 0 0" ]] || {
  echo "probe pipeline failed: ${PIPE_STATUSES[*]}" >&2
  exit 5
}
[[ -s "$PROBE" ]] || { echo "probe was not published: $PROBE" >&2; exit 5; }

"$VENV/bin/python" "$HELPERS/probe_health_gate.py" \
  "$PROBE" --model "$MODEL" --json "$HEALTH"
"$VENV/bin/python" "$HELPERS/capture_probe_netdata.py" \
  --after "$START_EPOCH" --before "$END_EPOCH" --output-dir "$NETDATA"

"$VENV/bin/python" - "$RECEIPT" "$RUN_ROOT" "$MODEL" "$DATASET" \
  "$REPO" "$HELPERS" "$EXPECTED_COMMIT" "$PROBE" "$HEALTH" "$LOG" \
  "$PROFILE" "$PROC_IO" "$WINDOW" "$NETDATA/netdata_manifest.json" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
(receipt, root, model, dataset, repo, helpers, commit, probe, health, log,
 profile, proc_io, window, netdata) = map(pathlib.Path, sys.argv[1:])
commit = str(commit)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def item(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path)}
payload = {
  "schema": "prismaquant.glm53-bf16-pread-probe.v1",
  "status": "PASS",
  "backend": "pread",
  "body_schedule": "unified_full_body_sweep",
  "emit_marginals": True,
  "calibration": {"dataset": item(dataset), "nsamples": 8, "seqlen": 512,
                  "modality": "text-only"},
  "model": {"path": str(model), "config": item(model / "config.json"),
            "index": item(model / "model.safetensors.index.json")},
  "source": {"repo": str(repo), "commit": commit,
             "tree": subprocess.check_output(
                 ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
                 text=True).strip(),
             "dirty": bool(subprocess.check_output(
                 ["git", "-C", str(repo), "status", "--porcelain"],
                 text=True).strip())},
  "artifacts": {name: item(path) for name, path in {
      "probe": probe, "health": health, "log": log, "profile": profile,
      "proc_io": proc_io, "window": window, "netdata": netdata}.items()},
  "helpers": {p.name: item(p) for p in [
      helpers / "probe_telemetry.py", helpers / "timestamp_stream.py",
      helpers / "capture_probe_netdata.py", helpers / "probe_health_gate.py"]},
  "run_root": str(root),
  "throughput_claim": False,
}
if payload["source"]["dirty"]:
    raise SystemExit("source became dirty during probe")
tmp = receipt.with_name(f".{receipt.name}.tmp-{os.getpid()}")
with tmp.open("x") as f:
    json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, receipt)
PY

echo "RECEIPT $RECEIPT"
