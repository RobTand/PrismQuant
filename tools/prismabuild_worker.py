#!/usr/bin/python3
"""SLURM batch-script entry point for the dependency-free PrismaBuild worker."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType


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
    raise SystemExit(main())
