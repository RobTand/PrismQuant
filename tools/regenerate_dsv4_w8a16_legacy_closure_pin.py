#!/usr/bin/env python3
"""Regenerate or check the complete DSv4 W8A16 PrismaQuant source pin."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cluster_transport import canonical_json_bytes
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant import dsv4_w8a16_export_handoff as handoff


def derive_pin(repo_root: Path) -> dict[str, object]:
    """Derive the pin with the handoff verifier's exact runtime inventory."""

    root = repo_root.resolve(strict=True)
    package = (root / "prismaquant").resolve(strict=True)
    if package.parent != root:
        raise ValueError("prismaquant package resolves outside repo root")
    pin_path = package / handoff._SOURCE_CLOSURE_PIN_NAME
    pin_digest = (
        hashlib.sha256(pin_path.read_bytes()).hexdigest()
        if pin_path.is_file()
        else ""
    )
    package_fd = os.open(
        package,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        files = handoff._runtime_inventory(
            package_fd,
            pin_payload_sha256=pin_digest,
        )
    finally:
        os.close(package_fd)
    body: dict[str, object] = {
        "schema": handoff.DSV4_W8A16_EXPORT_SOURCE_CLOSURE_PIN_SCHEMA,
        "files_sha256": files,
    }
    return {
        **body,
        "identity_sha256": canonical_json_sha256(
            body,
            where="DSv4 W8A16 source-closure pin identity",
        ),
    }


def _pin_bytes(pin: dict[str, object]) -> bytes:
    return canonical_json_bytes(pin) + b"\n"


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while regenerating closure pin")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve(strict=True).parent.parent,
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    path = (
        repo_root
        / "prismaquant"
        / handoff._SOURCE_CLOSURE_PIN_NAME
    )
    pin = derive_pin(repo_root)
    expected = _pin_bytes(pin)
    if args.check:
        try:
            observed = path.read_bytes()
        except FileNotFoundError:
            observed = b""
        if observed != expected:
            raise SystemExit(
                "DSv4 W8A16 source-closure pin differs; run with --write"
            )
    else:
        _write_atomic(path, expected)
    print(json.dumps({
        "path": str(path),
        "file_count": len(pin["files_sha256"]),
        "identity_sha256": pin["identity_sha256"],
        "status": "matched" if args.check else "written",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
