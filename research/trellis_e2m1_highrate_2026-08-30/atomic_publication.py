"""Crash-releasable claims and durable no-replace result publication.

This module is deliberately local to the numeric-driver directory.  Importing
it cannot rebind the canonical ``prismaquant`` package that the frozen hull
snapshot owns.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


CLAIM_SCHEMA = "trellis.numeric_publication_claim.v1"


class PublicationError(RuntimeError):
    """A numeric checkpoint or result cannot be published unambiguously."""


def canonical_json_bytes(value: object, *, indent: int | None = None) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=indent,
                sort_keys=True,
                separators=(",", ":") if indent is None else None,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationError("publication payload is not canonical JSON") from exc


def identity_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_destination(path: Path) -> Path:
    absolute = path.absolute()
    parent = absolute.parent.resolve(strict=True)
    return parent / absolute.name


@contextmanager
def exclusive_publication_claim(
    destination: Path,
    *,
    identity: Mapping[str, object],
) -> Iterator[Path]:
    """Exclude concurrent producers while allowing same-identity crash resume.

    A persistent claim binds the output namespace to one deterministic
    identity.  A nonblocking ``flock`` supplies live ownership and is released
    by the kernel if the producer dies.  A different identity must select a
    fresh output path.
    """

    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _canonical_destination(destination)
    claim = destination.with_name(f".{destination.name}.partial-claim")
    expected = canonical_json_bytes({
        "schema": CLAIM_SCHEMA,
        "destination": destination.name,
        "identity_sha256": identity_sha256(identity),
    })
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(claim, flags, 0o600)
    except OSError as exc:
        raise PublicationError(f"cannot open publication claim {claim}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublicationError(
                f"competing producer owns publication claim: {claim}"
            ) from exc
        size = os.fstat(descriptor).st_size
        if size:
            current = os.pread(descriptor, size, 0)
            if current != expected:
                raise PublicationError(
                    "publication claim identity differs; use a fresh --out path"
                )
        else:
            written = os.pwrite(descriptor, expected, 0)
            if written != len(expected):
                raise PublicationError("short publication-claim write")
            os.fsync(descriptor)
            _fsync_directory(destination.parent)
        yield claim
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_checkpoint_json(path: Path, value: Mapping[str, object]) -> None:
    """Replace one owner-held checkpoint durably.

    The caller must hold ``exclusive_publication_claim`` for the corresponding
    final path.  Replacement is safe only under that live ownership.
    """

    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _canonical_destination(path)
    payload = canonical_json_bytes(value, indent=1)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.write-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file_no_replace(staged: Path, destination: Path) -> None:
    """Hard-link one complete staged file into its immutable final name."""

    staged = staged.resolve(strict=True)
    destination = _canonical_destination(destination)
    if staged.parent != destination.parent:
        raise PublicationError("staged result and destination must be siblings")
    with staged.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(staged, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise PublicationError(
            f"final output appeared concurrently; refusing to replace: {destination}"
        ) from exc
    except OSError as exc:
        raise PublicationError(
            f"atomic no-replace publication failed for {destination}: {exc}"
        ) from exc
    _fsync_directory(destination.parent)
    staged.unlink()
    _fsync_directory(destination.parent)


__all__ = [
    "PublicationError",
    "atomic_checkpoint_json",
    "canonical_json_bytes",
    "exclusive_publication_claim",
    "file_sha256",
    "identity_sha256",
    "publish_file_no_replace",
]
