#!/usr/bin/python3
"""SLURM batch-script entry point for the dependency-free PrismaBuild worker."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType


def _capture_launched_source(path: Path) -> dict[str, object]:
    """Snapshot this entry point before importing the worker core."""

    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"worker launcher is not a regular file: {resolved}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"worker launcher changed while read: {resolved}")
    finally:
        os.close(descriptor)
    return {
        "path": str(resolved),
        "resolved_path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


_WORKER_LAUNCHER_IDENTITY = _capture_launched_source(Path(__file__))


# Import the dependency-free worker without executing prismaquant/__init__.py,
# whose public convenience imports intentionally pull in the numeric stack.
# Batch nodes need only the standard library to verify and dispatch an action;
# the action's own absolute argv selects its pinned per-architecture venv.
repository_root = Path(__file__).resolve().parents[1]
package = ModuleType("prismaquant")
package.__path__ = [str(repository_root / "prismaquant")]  # type: ignore[attr-defined]
package.__package__ = "prismaquant"
sys.modules.setdefault("prismaquant", package)

from prismaquant.prismabuild import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(worker_launcher_identity=_WORKER_LAUNCHER_IDENTITY))
