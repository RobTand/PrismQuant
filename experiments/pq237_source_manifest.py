#!/usr/bin/env python3
"""Freeze a host Git source map for the GPU image, which need not carry Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Importing this module must not leave a __pycache__ under the root it is
# about to seal: the reader refuses in-tree bytecode caches it would read.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.pq237_source_closure import (  # noqa: E402
    SCHEMA, find_bytecode_cache_directories, iter_source_closure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv], text=True)
    # Not filtered on exists(): that follows symlinks, so a tracked but
    # dangling link would be reported below as untracked.
    tracked = git("ls-files").splitlines()
    # Seal the union of the tracked roster the commit describes and the shared
    # import closure. The reader enumerates the same closure and refuses
    # anything unlisted, so writer and reader cannot disagree about what an
    # "exact file closure" covers.
    kinds = {p: "symlink" if (root / p).is_symlink() else "file"
             for p in tracked if (root / p).is_symlink() or (root / p).is_file()}
    closure = dict(iter_source_closure(root))
    kinds.update(closure)
    receipt = {
        "schema": SCHEMA,
        "commit": git("rev-parse", "HEAD").strip(),
        "status": git("status", "--porcelain"),
        # Named individually instead of being left inside the opaque `status`
        # text: the commit above does not describe these, so a reader can see
        # exactly what it trusts beyond the tree.
        "untracked_closure": sorted(set(closure) - set(tracked)),
        # Information for the operator: the reader refuses these unless its
        # interpreter reads its bytecode cache from outside the root.
        "bytecode_cache_directories": find_bytecode_cache_directories(root),
        "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                  for p, kind in kinds.items() if kind == "file"},
        "symlinks": {p: os.readlink(root / p)
                     for p, kind in kinds.items() if kind == "symlink"},
    }
    Path(args.out).write_text(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
