#!/usr/bin/env python3
"""Freeze a host Git source map for the GPU image, which need not carry Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv], text=True)
    files = git("ls-files").splitlines()
    receipt = {
        "commit": git("rev-parse", "HEAD").strip(),
        "status": git("status", "--porcelain"),
        "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                  for p in files if (root / p).is_file() and not (root / p).is_symlink()},
        "symlinks": {p: os.readlink(root / p) for p in files if (root / p).is_symlink()},
    }
    Path(args.out).write_text(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
