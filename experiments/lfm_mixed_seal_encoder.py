#!/usr/bin/env python3
"""Seal one Tessera commit as the wrapper's encoder input (#253).

``experiments/lfm_mixed_serving.py`` refuses an encoder tree whose file closure
differs from ``<prefix>-source-manifest.json`` and whose archive bytes differ
from the manifest, so the trio (archive, archive-input record, source
manifest) is what an observation is bound to.  The archive is
``git archive --format=tar <commit>``: bit-reproducible from the commit alone
(mtime is the commit time, ownership root/0), which is what makes the seal a
statement about a commit rather than about a checkout.  The published
``tessera-7018fa22...`` trio under ``joint-inputs/`` is reproduced by this
script byte for byte; ``--check`` proves that against an existing trio.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ARCHIVE_SCHEMA = "prismaquant.pq237.archive_preflight_source.v1"
SOURCE_SCHEMA = "prismaquant.pq237.tessera_source.v1"


def _git(repo: Path, *argv: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *argv])


def seal(repo: Path, commit: str) -> tuple[bytes, dict, dict]:
    commit = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    archive = _git(repo, "archive", "--format=tar", commit)
    files, symlinks = {}, {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if member.issym():
                symlinks[member.name] = member.linkname
            elif member.isfile():
                files[member.name] = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            elif not member.isdir():
                raise SystemExit(f"{commit}: archive entry {member.name} is neither file, dir nor symlink")
    digest = hashlib.sha256(archive).hexdigest()
    record = {"archive_bytes": len(archive), "archive_sha256": digest, "commit": commit,
              "excluded_symlinks": dict(symlinks), "files": files, "schema": ARCHIVE_SCHEMA}
    manifest = {"archive_bytes": len(archive), "archive_sha256": digest, "commit": commit,
                "files": files, "git_tree": tree, "schema": SOURCE_SCHEMA, "symlinks": dict(symlinks)}
    return archive, record, manifest


def _dump(value: dict) -> str:
    return json.dumps(value, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tessera", type=Path, required=True, help="a Tessera checkout holding the commit")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory the trio is written into")
    parser.add_argument("--check", action="store_true",
                        help="compare against an existing trio under --out instead of writing")
    args = parser.parse_args()
    archive, record, manifest = seal(args.tessera, args.commit)
    prefix = args.out / f"tessera-{record['commit']}"
    paths = {"archive": Path(f"{prefix}-archive-input.tar"),
             "record": Path(f"{prefix}-archive-input.json"),
             "manifest": Path(f"{prefix}-source-manifest.json")}
    if args.check:
        for name, path, expected in (("archive", paths["archive"], archive),
                                     ("record", paths["record"], _dump(record).encode()),
                                     ("manifest", paths["manifest"], _dump(manifest).encode())):
            actual = path.read_bytes()
            if name == "archive":
                same = actual == expected
            else:
                same = json.loads(actual) == json.loads(expected)
            print(f"{name}: {'identical' if same else 'DIFFERS'} {path}")
            if not same:
                return 1
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    paths["archive"].write_bytes(archive)
    paths["record"].write_text(_dump(record))
    paths["manifest"].write_text(_dump(manifest))
    print(json.dumps({"commit": record["commit"], "git_tree": manifest["git_tree"],
                      "files": len(record["files"]), "archive_bytes": record["archive_bytes"],
                      "archive_sha256": record["archive_sha256"],
                      "manifest_sha256": hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
                      **{k: str(v) for k, v in paths.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
