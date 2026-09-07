"""Source-checkpoint prefetch helpers for GPU-bound validation paths."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time


DEFAULT_HEADROOM_GB = 16.0


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _unique_safetensor_shards(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors index: {index_path}")
        names = sorted({str(name) for name in weight_map.values()})
        return [Path(root / name).resolve() for name in names]
    single = root / "model.safetensors"
    if single.exists():
        return [single.resolve()]
    return []


def _read_file_to_page_cache(path: Path, *, chunk_bytes: int) -> int:
    total = 0
    buf = bytearray(int(chunk_bytes))
    view = memoryview(buf)
    with path.open("rb", buffering=0) as fh:
        while True:
            n = fh.readinto(view)
            if not n:
                break
            total += int(n)
    return total


def prefetch_files_to_page_cache(
    paths: list[str | Path],
    *,
    mode: str = "require",
    max_resident_bytes: int | None = None,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    workers: int = 2,
    chunk_mb: int = 64,
    progress: bool = True,
    log_prefix: str = "[file-prefetch]",
    label: str = "files",
) -> dict[str, object]:
    """Read local files once so later faults hit the OS page cache.

    This is deliberately not a rendered-weight or activation cache. It is a
    fail-fast residency gate for production validation paths: if required local
    files cannot be made resident within the requested budget, validation should
    stop instead of silently becoming NVMe-bound.
    """
    mode = str(mode or "off").lower()
    if mode not in {"off", "auto", "require"}:
        raise ValueError(f"unsupported file prefetch mode {mode!r}")
    stats: dict[str, object] = {
        "mode": mode,
        "label": label,
        "files": 0,
        "bytes": 0,
        "max_resident_bytes": 0,
        "available_bytes": None,
        "prefetched_bytes": 0,
        "elapsed_seconds": 0.0,
        "skipped": False,
    }
    if mode == "off":
        stats["skipped"] = True
        stats["reason"] = "mode=off"
        return stats

    unique_paths = sorted({Path(path).resolve() for path in paths})
    stats["files"] = len(unique_paths)
    if not unique_paths:
        stats["skipped"] = True
        stats["reason"] = f"no local {label}"
        if mode == "require":
            raise RuntimeError(f"{log_prefix} no local {label}")
        return stats

    missing = [str(path) for path in unique_paths if not path.is_file()]
    if missing:
        stats["missing"] = missing[:8]
        if mode == "require":
            raise FileNotFoundError(
                f"{log_prefix} missing {label}: {missing[:8]}"
            )
        stats["skipped"] = True
        stats["reason"] = f"missing {label}"
        return stats

    total_bytes = sum(path.stat().st_size for path in unique_paths)
    stats["bytes"] = int(total_bytes)
    available = _available_memory_bytes()
    stats["available_bytes"] = available
    if max_resident_bytes is None or int(max_resident_bytes) <= 0:
        if available is not None:
            max_resident_bytes = max(
                0,
                int(available - float(headroom_gb) * 1024**3),
            )
        else:
            max_resident_bytes = 0
    stats["max_resident_bytes"] = int(max_resident_bytes or 0)
    # An exhausted or unknown automatic budget is zero, never unlimited.
    if total_bytes > int(max_resident_bytes):
        stats["skipped"] = True
        stats["reason"] = (
            f"{label} require {total_bytes / 1024**3:.2f} GiB, "
            f"budget is {int(max_resident_bytes) / 1024**3:.2f} GiB"
        )
        if mode == "require":
            raise RuntimeError(f"{log_prefix} {stats['reason']}")
        if progress:
            print(f"{log_prefix} WARNING: {stats['reason']}; skipping", flush=True)
        return stats

    if progress:
        budget_text = (
            f", budget={int(max_resident_bytes) / 1024**3:.2f} GiB"
            if max_resident_bytes else ""
        )
        print(
            f"{log_prefix} prefetching {len(unique_paths)} {label}, "
            f"{total_bytes / 1024**3:.2f} GiB{budget_text}",
            flush=True,
        )
    start = time.time()
    chunk_bytes = max(1, int(chunk_mb)) * 1024**2
    workers = max(1, int(workers))
    if workers == 1 or len(unique_paths) == 1:
        prefetched = sum(
            _read_file_to_page_cache(path, chunk_bytes=chunk_bytes)
            for path in unique_paths
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prefetched = sum(
                pool.map(
                    lambda path: _read_file_to_page_cache(
                        path, chunk_bytes=chunk_bytes
                    ),
                    unique_paths,
                )
            )
    elapsed = time.time() - start
    stats["prefetched_bytes"] = int(prefetched)
    stats["elapsed_seconds"] = float(elapsed)
    if prefetched != total_bytes:
        stats["incomplete"] = True
        msg = f"prefetched {prefetched} of {total_bytes} bytes"
        if mode == "require":
            raise RuntimeError(f"{log_prefix} incomplete source prefetch: {msg}")
        if progress:
            print(f"{log_prefix} WARNING: {msg}", flush=True)
    if progress:
        gib_s = (prefetched / 1024**3 / elapsed) if elapsed > 0 else 0.0
        print(
            f"{log_prefix} prefetched {prefetched / 1024**3:.2f} GiB "
            f"in {elapsed:.1f}s ({gib_s:.2f} GiB/s)",
            flush=True,
        )
    return stats


def prefetch_safetensors_checkpoint(
    model_path: str | Path,
    *,
    mode: str = "require",
    max_resident_bytes: int | None = None,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    workers: int = 2,
    chunk_mb: int = 64,
    progress: bool = True,
    log_prefix: str = "[source-prefetch]",
) -> dict[str, object]:
    """Read local safetensors shards once so later mmap faults hit RAM."""
    mode = str(mode or "off").lower()
    shards = _unique_safetensor_shards(model_path)
    stats = prefetch_files_to_page_cache(
        shards,
        mode=mode,
        max_resident_bytes=max_resident_bytes,
        headroom_gb=headroom_gb,
        workers=workers,
        chunk_mb=chunk_mb,
        progress=progress,
        log_prefix=log_prefix,
        label="safetensors shards",
    )
    stats["model_path"] = str(model_path)
    stats["shards"] = stats.get("files", 0)
    if (
        stats.get("skipped")
        and stats.get("reason") == "no local safetensors shards"
        and mode == "require"
    ):
        raise RuntimeError(
            f"{log_prefix} no local safetensors shards under {model_path}"
        )
    return stats
