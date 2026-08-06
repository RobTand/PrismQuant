"""
Content-keyed persistence helpers for tier-3 units.

Each unit is keyed by (layer, projection, expert, source_digest, col_weights_digest,
book_shas for K28..K38, groups).  Filename is sha256 of canonical JSON of key (hex)
with human readable prefix.  Resume skips if target JSON exists and validates count.

RSS guard and incremental flush are handled by callers; helpers just give paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

def unit_content_key(
    *,
    layer: int,
    projection: str,
    expert: int,
    source_digest: str,
    col_weights_digest: str,
    book_shas: dict[int, str],  # rung -> sha256
    groups: int,
) -> str:
    payload = {
        "layer": int(layer),
        "projection": str(projection),
        "expert": int(expert),
        "source_digest": str(source_digest),
        "col_weights_digest": str(col_weights_digest),
        "book_shas": {str(k): str(v) for k, v in sorted(book_shas.items())},
        "groups": int(groups),
        "schema": "prismaquant.tier3.unit_key.v1",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def unit_path(
    results_root: Path,
    *,
    layer: int,
    projection: str,
    expert: int,
    groups: int,
    key: str,
) -> Path:
    # Human prefix + hash for debug; resume uses existence check on same path
    fname = f"L{layer:02d}_{projection}_E{expert:03d}_G{groups}_{key[:16]}.json"
    return results_root / fname

def atomic_write_json(path: Path, payload: Any) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)

def rss_bytes() -> int:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except Exception:
        try:
            import psutil
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0

def host_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        return 0
