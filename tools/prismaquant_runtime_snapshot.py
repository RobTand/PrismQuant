#!/usr/bin/env python3
"""Materialize and verify an immutable PrismaQuant runtime source snapshot.

Containerized release jobs must not execute from a live worktree for hours.
This stdlib-only helper archives one exact Git commit into a content-addressed
cache, records every tracked regular file/symlink, and re-hashes that complete
closure before it is mounted and again inside the container.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "prismaquant.runtime_source_snapshot.v1"
MANIFEST = ".prismaquant-runtime-snapshot.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TREE = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SnapshotError(ValueError):
    """The runtime snapshot is absent, unsafe, or differs from its identity."""


def _run_git(source: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise SnapshotError(f"Git identity lookup failed: {detail.strip()}") from exc
    return result.stdout.strip()


def _resolve_identity(source: Path, commit: str) -> tuple[str, str]:
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise SnapshotError(f"source root is not a directory: {root}")
    if _COMMIT.fullmatch(commit) is None:
        raise SnapshotError("commit must be a full lowercase 40-hex Git identity")
    resolved = _run_git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise SnapshotError(
            f"requested commit {commit} resolves to a different object {resolved}"
        )
    tree = _run_git(root, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if _TREE.fullmatch(tree) is None:
        raise SnapshotError(f"Git returned an invalid tree identity: {tree!r}")
    return resolved, tree


def _reject_duplicate_members(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError(f"duplicate manifest member {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotError(f"non-finite JSON value {value}")
            ),
        )
    except SnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"snapshot manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot manifest must be a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SnapshotError(f"cannot hash runtime source file {path}") from exc
    return digest.hexdigest()


def _snapshot_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISDIR(info.st_mode):
                kept_directories.append(name)
            else:
                raise SnapshotError(
                    f"unsupported runtime source filesystem entry: {path}"
                )
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISREG(info.st_mode):
                entries.append({
                    "path": relative,
                    "type": "file",
                    "bytes": int(info.st_size),
                    "executable": bool(info.st_mode & stat.S_IXUSR),
                    "sha256": _file_sha256(path),
                })
            else:
                raise SnapshotError(
                    f"unsupported runtime source filesystem entry: {path}"
                )
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _closure_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(entries),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "commit", "tree", "closure_sha256", "entry_count", "entries"
    }
    if set(payload) != required:
        raise SnapshotError(
            f"snapshot manifest expected exactly {sorted(required)}, "
            f"got {sorted(payload)}"
        )
    if payload.get("schema") != SCHEMA:
        raise SnapshotError(f"unsupported snapshot schema {payload.get('schema')!r}")
    if not isinstance(payload.get("commit"), str) or _COMMIT.fullmatch(
        str(payload["commit"])
    ) is None:
        raise SnapshotError("snapshot commit is invalid")
    if not isinstance(payload.get("tree"), str) or _TREE.fullmatch(
        str(payload["tree"])
    ) is None:
        raise SnapshotError("snapshot tree is invalid")
    if not isinstance(payload.get("closure_sha256"), str) or _SHA256.fullmatch(
        str(payload["closure_sha256"])
    ) is None:
        raise SnapshotError("snapshot closure hash is invalid")
    entries = payload.get("entries")
    count = payload.get("entry_count")
    if (
        not isinstance(entries, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or len(entries) != count
    ):
        raise SnapshotError("snapshot entry ledger is malformed")
    return dict(payload)


def verify_snapshot(
    snapshot: Path,
    *,
    expected_commit: str,
    expected_tree: str | None = None,
    expected_closure_sha256: str | None = None,
) -> dict[str, Any]:
    if snapshot.is_symlink():
        raise SnapshotError(f"snapshot path must not be a symlink: {snapshot}")
    root = snapshot.resolve(strict=True)
    if not root.is_dir():
        raise SnapshotError(f"snapshot is not one real directory: {root}")
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SnapshotError("snapshot manifest must be one real regular file")
    payload = _validate_manifest_shape(_load_manifest(manifest_path))
    if payload["commit"] != expected_commit:
        raise SnapshotError(
            f"snapshot commit {payload['commit']} differs from {expected_commit}"
        )
    if expected_tree is not None and payload["tree"] != expected_tree:
        raise SnapshotError(
            f"snapshot tree {payload['tree']} differs from {expected_tree}"
        )
    observed_entries = _snapshot_entries(root)
    if observed_entries != payload["entries"]:
        raise SnapshotError("snapshot files differ from the exact tracked ledger")
    observed_closure = _closure_sha256(observed_entries)
    if observed_closure != payload["closure_sha256"]:
        raise SnapshotError("snapshot closure hash differs from its ledger")
    if (
        expected_closure_sha256 is not None
        and observed_closure != expected_closure_sha256
    ):
        raise SnapshotError(
            "snapshot closure hash differs from the caller-attested identity"
        )
    for directory in (root / "prismaquant", root / "tools"):
        if directory.is_symlink() or not directory.is_dir():
            raise SnapshotError(
                f"snapshot runtime directory is absent or unsafe: {directory}"
            )
    for required in (
        root / "prismaquant" / "__init__.py",
        root / "tools" / "container_runtime_identity.py",
        root / "tools" / "prismaquant_runtime_snapshot.py",
    ):
        if not required.is_file() or required.is_symlink():
            raise SnapshotError(f"snapshot runtime entry is absent or unsafe: {required}")
    return {
        "schema": SCHEMA,
        "snapshot": str(root),
        "commit": payload["commit"],
        "tree": payload["tree"],
        "closure_sha256": observed_closure,
        "entry_count": len(observed_entries),
    }


def materialize_snapshot(
    source: Path, cache_root: Path, *, commit: str
) -> dict[str, Any]:
    source_root = source.resolve(strict=True)
    resolved_commit, tree = _resolve_identity(source_root, commit)
    cache = cache_root.resolve(strict=False)
    if (
        str(cache) in {"", "/"}
        or cache == source_root
        or cache in source_root.parents
        or source_root in cache.parents
    ):
        raise SnapshotError(f"unsafe runtime snapshot cache root: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    cache = cache.resolve(strict=True)
    destination = cache / f"{resolved_commit}-{tree[:12]}"
    lock_path = cache / f".{resolved_commit}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            return verify_snapshot(
                destination,
                expected_commit=resolved_commit,
                expected_tree=tree,
            )
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{resolved_commit[:12]}.tmp-", dir=cache
        ))
        try:
            archive = temporary / ".runtime-source.tar"
            try:
                subprocess.run(
                    [
                        "git", "-C", str(source_root), "archive", "--format=tar",
                        f"--output={archive}", resolved_commit,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise SnapshotError(
                    f"could not archive reviewed runtime commit: {detail.strip()}"
                ) from exc
            try:
                with tarfile.open(archive, mode="r:") as handle:
                    handle.extractall(temporary, filter="data")
            except (OSError, tarfile.TarError) as exc:
                raise SnapshotError("could not safely extract runtime archive") from exc
            archive.unlink()
            entries = _snapshot_entries(temporary)
            payload = {
                "schema": SCHEMA,
                "commit": resolved_commit,
                "tree": tree,
                "closure_sha256": _closure_sha256(entries),
                "entry_count": len(entries),
                "entries": entries,
            }
            (temporary / MANIFEST).write_text(
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            verify_snapshot(
                temporary,
                expected_commit=resolved_commit,
                expected_tree=tree,
                expected_closure_sha256=payload["closure_sha256"],
            )
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return verify_snapshot(
        destination,
        expected_commit=resolved_commit,
        expected_tree=tree,
    )


#: Paths whose HEAD revision is used ONLY by the host-side gate, never by the
#: served stack. The container is mounted the *runtime* snapshot, so it always
#: executes the artifact's own build-commit copy of every file; this list is not
#: about what the container runs. It is the answer to a narrower question: when
#: the judge runs newer than the runtime it is judging, which divergences can be
#: known harmless? Only those in code that renders a verdict. A judge that also
#: expects something NEW from the serve would be a divergence in the serve path,
#: and every such path is deliberately absent from this list, so it refuses.
#:
#: Trailing "/" means the whole subtree. Keep this list short and justified;
#: growing it is a contract decision, not a convenience.
JUDGE_ONLY_PATHS: tuple[str, ...] = (
    # Documentation and tests are not executable stack under any entry point.
    "docs/",
    "tests/",
    # Three lane-owned entries (the CB endpoint gate, the W8A16 export-handoff
    # closure and the CB validate launcher) left this list when the Gridbook
    # lane retired 2026-09-02; see archive/gridbook_lane_2026-09-02/.
    #
    # Classifies a checkpoint's units for the cover proof. Host-side only in the
    # gate's path; the exporter imports it too, but export already happened.
    "prismaquant/artifact_completeness.py",
    # This file. It is the one entry here that the CONTAINER also executes -- as
    # `/repo/tools/prismaquant_runtime_snapshot.py verify`, from the runtime
    # snapshot, so at the build commit. That is safe for a specific reason
    # rather than by assertion: the launcher runs the runtime snapshot's copy
    # host-side, with the identical `verify --snapshot --expected-commit
    # --expected-tree --expected-closure-sha256` surface, at every checkpoint
    # and before any container starts. A newer judge that broke that CLI would
    # therefore fail on the host first, loudly, with no GPU reserved.
    "tools/prismaquant_runtime_snapshot.py",
)


def _manifest_paths(snapshot: Path) -> dict[str, tuple[Any, ...]]:
    payload = _validate_manifest_shape(_load_manifest(snapshot / MANIFEST))
    return {
        str(entry["path"]): (
            entry.get("type"),
            entry.get("sha256"),
            entry.get("target"),
            entry.get("executable"),
        )
        for entry in payload["entries"]
    }


def _is_judge_only(path: str) -> bool:
    for allowed in JUDGE_ONLY_PATHS:
        if allowed.endswith("/"):
            if path.startswith(allowed):
                return True
        elif path == allowed:
            return True
    return False


def judge_divergence(runtime: Path, judge: Path) -> dict[str, Any]:
    """Prove a judge snapshot may validate an artifact built at *runtime*.

    The serve gate binds the container to the artifact's build commit so the
    stack that decodes the bytes is the stack they were made for. Binding the
    *judge* to it as well made a validator bug structurally incurable: the only
    remedy the gate admitted was rebuilding bytes that were never wrong. The two
    roles are now split, and this is the proof obligation that keeps the split
    honest -- every closure path that differs between the two revisions must be
    judge-only, or the gate refuses and re-export is the answer.
    """

    runtime_paths = _manifest_paths(runtime)
    judge_paths = _manifest_paths(judge)
    divergent = sorted(
        path
        for path in runtime_paths.keys() | judge_paths.keys()
        if runtime_paths.get(path) != judge_paths.get(path)
    )
    blocking = [path for path in divergent if not _is_judge_only(path)]
    if blocking:
        raise SnapshotError(
            "judge snapshot diverges from the artifact's runtime outside the "
            "judge-only set, so it may not validate this artifact: "
            + ", ".join(blocking[:8])
            + (f" (+{len(blocking) - 8} more)" if len(blocking) > 8 else "")
        )
    return {
        "schema": "prismaquant.judge_split_divergence.v1",
        "divergent_paths": divergent,
        "divergent_count": len(divergent),
        "judge_only_paths": list(JUDGE_ONLY_PATHS),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--cache-root", type=Path, required=True)
    materialize.add_argument("--commit", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--expected-tree")
    verify.add_argument("--expected-closure-sha256")
    divergence = subparsers.add_parser("judge-divergence")
    divergence.add_argument("--runtime-snapshot", type=Path, required=True)
    divergence.add_argument("--judge-snapshot", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            payload = materialize_snapshot(
                args.source_root, args.cache_root, commit=args.commit
            )
        elif args.command == "judge-divergence":
            payload = judge_divergence(args.runtime_snapshot, args.judge_snapshot)
        else:
            payload = verify_snapshot(
                args.snapshot,
                expected_commit=args.expected_commit,
                expected_tree=args.expected_tree,
                expected_closure_sha256=args.expected_closure_sha256,
            )
    except (OSError, SnapshotError) as exc:
        print(f"runtime-snapshot: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
