#!/usr/bin/env python3
"""CLI shim for :mod:`prismaquant.cluster_campaign`."""

from pathlib import Path
import sys


# Campaign stages deliberately run from an explicit cwd with a closed env.
# Keep this checked-out tool usable from such a cwd without relying on an
# editable install or ambient PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_campaign import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
