#!/usr/bin/env bash
# Build an explicitly UNRELEASABLE structural artifact on DGX Spark GB10.
# The delegated launcher consumes only Gridbook v10 compile_only SM89 cells;
# its output can never satisfy shipcard, publication, or RTX4090 serving gates.

set -euo pipefail

PQ_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PQ_REPO_ROOT"

python3 - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        "REFUSED: validation-only launcher requires exactly one visible GB10 GPU"
    )
capability = tuple(torch.cuda.get_device_capability(0))
name = torch.cuda.get_device_name(0)
if capability != (12, 1) or "GB10" not in name.upper():
    raise SystemExit(
        f"REFUSED: expected DGX Spark GB10 compute capability 12.1, got "
        f"{name!r} capability={capability}"
    )
print(f"[rtx4090-validation-only] build host={name}, capability={capability}")
PY

export RTX4090_BUILD_DISPOSITION=validation_only
exec "$PQ_REPO_ROOT/scripts/run_qwen38_rtx4090_fp8_cb_18gb.sh"
