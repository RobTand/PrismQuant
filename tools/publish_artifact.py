#!/usr/bin/env python3
"""Publish an exported artifact to the Hub through the blocking ship gate.

Model publication is deliberately stricter than a convenience ``hf upload``:

* the artifact's canonical ``shipcard.json`` must be a readable, regular,
  non-symlink file (this structural rule is not force-overrideable);
* the complete local file set is frozen before the authoritative shipcard
  replay. Small files are captured as bytes. Large files stay zero-copy, but
  are held by no-follow file descriptors and content-addressed blockwise; the
  upload reader refuses to expose a block that differs from that snapshot;
* the remote file set is enumerated at one exact branch head and replaced in
  one ``create_commit`` call with ``revision`` plus ``parent_commit`` CAS.
  ``.gitattributes`` is the one intentional retained Hub-managed file;
* model artifacts publish only at repository root and never through include or
  exclude filters.

``--force-unverified`` overrides failed quality/evidence slots only. It still
requires a valid canonical card, requires the artifact basename to be retyped,
and stamps the override into the bytes that are frozen and uploaded.

There is intentionally no raw ``hf upload`` fallback: that command cannot
preserve this frozen-byte and remote-CAS contract. If ``huggingface_hub`` is not
available, install it and rerun this publisher from the beginning.

Exit codes: 0 published (or local dry-run complete) · 1 refused/runtime failure
· 2 usage/structural error.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.shipcard import (  # noqa: E402
    SCHEMA as SHIPCARD_SCHEMA,
    SHIPCARD_FILENAME,
    compute_model_sha,
    load_shipcard,
    required_slots,
    unfilled_slots,
    verify,
    write_shipcard,
)
REPO_TYPES = ("model", "dataset", "space")
SNAPSHOT_BLOCK_BYTES = 8 * 1024 * 1024
SNAPSHOT_INLINE_BYTES = 16 * 1024 * 1024
SNAPSHOT_PREFIX = ".prismaquant-publish-snapshot-"
_FULL_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class FrozenSnapshotError(RuntimeError):
    """The local tree could not be frozen or changed after it was frozen."""


@dataclasses.dataclass(frozen=True)
class _HubBindings:
    api_type: type
    add_type: type
    delete_type: type


def _load_hub_bindings() -> _HubBindings:
    """Import the exact low-level primitives lazily.

    ``upload_folder`` is intentionally not imported: it re-enumerates paths and
    cannot guarantee that the bytes it opens are the bytes we verified.
    """
    from huggingface_hub import (  # noqa: PLC0415
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
    )

    return _HubBindings(
        api_type=HfApi,
        add_type=CommitOperationAdd,
        delete_type=CommitOperationDelete,
    )


def _shipcard_path(artifact_dir: Path, explicit: str | None) -> Path:
    return Path(explicit) if explicit else artifact_dir / SHIPCARD_FILENAME


def _canonical_shipcard_problem(
    artifact_dir: Path,
    shipcard_path: Path,
) -> str | None:
    """Require the one ordinary in-tree publication receipt."""
    try:
        root = artifact_dir.resolve(strict=True)
    except OSError as exc:
        return f"artifact directory is not canonical/readable: {exc}"
    expected = root / SHIPCARD_FILENAME
    if shipcard_path.is_symlink() or expected.is_symlink():
        return f"shipcard must not be a symlink: {shipcard_path}"
    try:
        actual = shipcard_path.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
        mode = expected.lstat().st_mode
    except OSError as exc:
        return f"canonical shipcard {expected} is missing or unreadable: {exc}"
    if actual != expected_resolved:
        return (
            f"shipcard must be the canonical in-tree {expected}; got "
            f"{shipcard_path}"
        )
    if not stat.S_ISREG(mode):
        return f"canonical shipcard {expected} is not a regular file"
    return None


def check_shipcard(
    artifact_dir: Path,
    shipcard_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return ``(card, problems)``; malformed input is not forceable."""
    if not shipcard_path.is_file():
        return None, [
            f"no shipcard at {shipcard_path} — an artifact with no ship record "
            "has never been gated; re-export at a commit that writes one"
        ]
    try:
        card = load_shipcard(shipcard_path)
        structural = _shipcard_structure_problems(card, artifact_dir)
        if structural:
            return None, structural
        problems = verify(card, model_dir=artifact_dir)
    except Exception as exc:
        return None, [f"{shipcard_path} is not a readable shipcard: {exc!r}"]
    return card, problems


# The strict RTX4090 FP8-CB publication gates (policy/serving-profile
# detection, the validation-only refusal, and the whole-artifact size ceiling)
# lived here. They gated one lane; Gridbook lane retired 2026-09-02, see
# archive/gridbook_lane_2026-09-02/. Nothing else reached
# them, so a publication is now judged only by its shipcard.
def _shipcard_structure_problems(
    card: Mapping[str, Any],
    artifact_dir: Path,
) -> list[str]:
    """Separate malformed receipts from forceable evidence failures."""
    problems: list[str] = []
    if card.get("schema") != SHIPCARD_SCHEMA:
        problems.append(
            f"shipcard schema must be {SHIPCARD_SCHEMA!r}; got "
            f"{card.get('schema')!r}"
        )
    for key in ("created", "model_dir"):
        value = card.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"shipcard {key} must be a non-empty string")
    model_sha = card.get("model_sha")
    if not isinstance(model_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", model_sha,
    ) is None:
        problems.append("shipcard model_sha must be one lowercase SHA-256")
    for key in ("artifact_bytes", "reserved_file_bytes"):
        value = card.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if key == "reserved_file_bytes" else 0)
        ):
            problems.append(f"shipcard {key} is not a valid integer")
    if not isinstance(card.get("build"), Mapping):
        problems.append("shipcard build must be an object")
    slots = card.get("slots")
    if not isinstance(slots, Mapping):
        problems.append("shipcard slots must be an object")
        return problems
    if any(not isinstance(slot, str) for slot in slots):
        problems.append("shipcard slot names must be strings")
    malformed_records = sorted(
        str(slot)
        for slot, record in slots.items()
        if record is not None and not isinstance(record, Mapping)
    )
    if malformed_records:
        problems.append(
            "shipcard has non-object slot record(s): "
            + ", ".join(malformed_records)
        )
    try:
        missing = [
            slot
            for slot in required_slots(card, model_dir=artifact_dir)
            if slot not in slots
        ]
    except Exception as exc:
        problems.append(f"shipcard required-slot contract is malformed: {exc!r}")
    else:
        if missing:
            problems.append(
                "shipcard omits required slot key(s): " + ", ".join(missing)
            )
    return problems


def _confirm_forced(artifact_dir: Path, confirm_name: str | None) -> bool:
    """Re-typing the basename is the confirmation. Deliberate, not a y/n."""
    expected = artifact_dir.resolve().name
    typed = confirm_name
    if typed is None:
        if not sys.stdin.isatty():
            print(
                "[publish] REFUSED: --force-unverified needs the artifact "
                f"directory basename re-typed ({expected!r}); no tty, so pass "
                "--confirm-name",
                file=sys.stderr,
            )
            return False
        typed = input(
            f"[publish] Type the artifact directory name to publish it "
            f"UNVERIFIED ({expected}): "
        )
    if str(typed).strip() != expected:
        print(
            f"[publish] REFUSED: typed {str(typed).strip()!r} != {expected!r}; "
            "the confirmation must match the artifact directory basename",
            file=sys.stderr,
        )
        return False
    return True


def stamp_forced(
    shipcard_path: Path,
    card: dict[str, Any],
    artifact_dir: Path,
    problems: list[str],
    repo_id: str,
) -> None:
    """Record on the artifact that it was published without a closed card."""
    card["forced_unverified"] = True
    history = list(card.get("forced_unverified_history") or [])
    history.append({
        "repo_id": repo_id,
        "model_sha": compute_model_sha(artifact_dir),
        "problems": list(problems),
        "unfilled_slots": unfilled_slots(card, model_dir=artifact_dir),
    })
    card["forced_unverified_history"] = history
    write_shipcard(shipcard_path, card)
    print(f"[publish] stamped forced_unverified=true into {shipcard_path}")


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


# The per-file content stat and its payload spelling served only the
# safetensors content receipt, which had exactly one consumer: the strict
# RTX4090 FP8-CB publication replay.  That lane retired 2026-09-02 (see
# archive/gridbook_lane_2026-09-02/), leaving the receipt with no verifier, so
# capture and receipt both go rather than accumulate unread provenance.


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            raise FrozenSnapshotError(
                f"file shortened while freezing at byte {cursor}"
            )
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclasses.dataclass
class _FrozenEntry:
    relative_path: str
    size: int
    sha256: str
    sample: bytes
    block_sha256: tuple[str, ...]
    content: bytes | None
    fd: int | None

    def close(self) -> None:
        if self.fd is not None:
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = None


class _VerifiedBlockReader(io.BufferedIOBase):
    """Seekable reader that exposes only content matching the frozen blocks."""

    def __init__(self, entry: _FrozenEntry):
        super().__init__()
        if entry.fd is None or entry.content is not None:
            raise ValueError("verified reader requires a file-backed entry")
        self._entry = entry
        self._position = 0
        self._cached_index = -1
        self._cached_block = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._entry.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = int(position)
        return self._position

    def _block(self, index: int) -> bytes:
        if index == self._cached_index:
            return self._cached_block
        fd = self._entry.fd
        if fd is None:
            raise FrozenSnapshotError(
                f"frozen descriptor closed for {self._entry.relative_path}"
            )
        offset = index * SNAPSHOT_BLOCK_BYTES
        expected_size = min(
            SNAPSHOT_BLOCK_BYTES,
            self._entry.size - offset,
        )
        try:
            data = _pread_exact(fd, expected_size, offset)
        except (OSError, FrozenSnapshotError) as exc:
            raise FrozenSnapshotError(
                f"frozen source became unreadable: "
                f"{self._entry.relative_path}: {exc}"
            ) from exc
        observed = hashlib.sha256(data).hexdigest()
        expected = self._entry.block_sha256[index]
        if observed != expected:
            raise FrozenSnapshotError(
                "frozen source mutated after verification: "
                f"{self._entry.relative_path} block {index} "
                f"({observed[:12]} != {expected[:12]})"
            )
        self._cached_index = index
        self._cached_block = data
        return data

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("read of closed file")
        if self._position >= self._entry.size:
            return b""
        remaining_file = self._entry.size - self._position
        # Hub's streaming transports use bounded reads. Keep an accidental
        # unbounded read bounded to one block instead of materializing a
        # 112-GB object in RAM, while honoring every positive-size request.
        target = (
            min(remaining_file, SNAPSHOT_BLOCK_BYTES)
            if size is None or size < 0
            else min(remaining_file, size)
        )
        chunks: list[bytes] = []
        remaining = target
        while remaining:
            index, inside = divmod(self._position, SNAPSHOT_BLOCK_BYTES)
            block = self._block(index)
            take = min(len(block) - inside, remaining)
            if take <= 0:
                raise FrozenSnapshotError(
                    f"invalid frozen block geometry: {self._entry.relative_path}"
                )
            chunks.append(block[inside:inside + take])
            self._position += take
            remaining -= take
        return b"".join(chunks)


@dataclasses.dataclass
class _FrozenSnapshot:
    source_root: Path
    source_root_identity: tuple[int, int]
    root: Path
    root_identity: tuple[int, int]
    entries: list[_FrozenEntry]
    manifest_sha256: str
    total_bytes: int
    _readers: list[BinaryIO] = dataclasses.field(default_factory=list)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.relative_path for entry in self.entries)

    def reader_for(self, entry: _FrozenEntry) -> BinaryIO:
        reader = _VerifiedBlockReader(entry)
        self._readers.append(reader)
        return reader

    def close(self) -> None:
        for reader in self._readers:
            with contextlib.suppress(Exception):
                reader.close()
        self._readers.clear()
        for entry in self.entries:
            entry.close()
        # The target is a tool-created sibling with a randomized basename.
        # Validate that invariant before removing it.
        if (
            self.root.parent == self.source_root.parent
            and self.root.name.startswith(SNAPSHOT_PREFIX)
        ):
            try:
                current = self.root.lstat()
            except OSError:
                return
            if (
                stat.S_ISDIR(current.st_mode)
                and not self.root.is_symlink()
                and (int(current.st_dev), int(current.st_ino))
                == self.root_identity
            ):
                shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "_FrozenSnapshot":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class _ScannedFile:
    content: bytes | None
    sha256: str
    sample: bytes
    block_sha256: tuple[str, ...]
    read_calls: int


def _scan_file(fd: int, size: int) -> _ScannedFile:
    """Hash one file once, whole and in fixed blocks.

    The only retained data are inline files small enough to live in the
    snapshot tree.  A per-tensor safetensors span hash used to be computed from
    the same bytes; it fed the content receipt whose only consumer was the
    strict RTX4090 FP8-CB publication replay, retired with the Gridbook lane on
    2026-09-02 (see archive/gridbook_lane_2026-09-02/).
    """

    whole = hashlib.sha256()
    blocks: list[str] = []
    sample = b""
    inline_chunks: list[bytes] | None = [] if size <= SNAPSHOT_INLINE_BYTES else None
    read_calls = 0

    for offset in range(0, size, SNAPSHOT_BLOCK_BYTES):
        length = min(SNAPSHOT_BLOCK_BYTES, size - offset)
        data = _pread_exact(fd, length, offset)
        read_calls += 1
        if offset == 0:
            sample = data[:512]
        whole.update(data)
        blocks.append(hashlib.sha256(data).hexdigest())
        if inline_chunks is not None:
            inline_chunks.append(data)

    content = b"".join(inline_chunks) if inline_chunks is not None else None
    return _ScannedFile(
        content=content,
        sha256=whole.hexdigest(),
        sample=sample,
        block_sha256=tuple(blocks),
        read_calls=read_calls,
    )


def _write_snapshot_bytes(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        view = memoryview(content)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(f"short write while freezing {path}")
            written += count
    finally:
        os.close(fd)


def _freeze_artifact(
    artifact_dir: Path,
    *,
    expected_shipcard_sha256: str,
) -> _FrozenSnapshot:
    """Freeze one stable, regular-file-only artifact tree without data copying.

    Large files are represented inside the verifier's private tree by
    ``/proc/self/fd`` links, which do not change source inode ctime (hardlinks
    would invalidate the shipcard's stat attestation). Upload consumes those
    links as paths -- each resolves through this process's held descriptor,
    which cannot be retargeted -- because the Hub's Xet transport (the only
    one without a 50 GB per-file cap) cannot read a Python file object; the
    declared digests are replayed across the same descriptors after the
    commit. ``_VerifiedBlockReader`` remains the accessor for local replay.
    """
    root = artifact_dir.absolute()
    if root.is_symlink():
        raise FrozenSnapshotError("artifact directory itself must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise FrozenSnapshotError(f"artifact root is unreadable: {exc}") from exc

    snapshot_root = Path(tempfile.mkdtemp(
        prefix=SNAPSHOT_PREFIX,
        dir=str(root.parent),
    ))
    snapshot_root_info = snapshot_root.lstat()
    snapshot_root_identity = (
        int(snapshot_root_info.st_dev),
        int(snapshot_root_info.st_ino),
    )
    entries: list[_FrozenEntry] = []
    directory_records: list[tuple[int, tuple[int, int, int, int, int, int], str]] = []
    directory_fds: list[int] = []
    retained_file_fds: list[int] = []

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW

    def walk(source_fd: int, relative_dir: PurePosixPath) -> None:
        directory_records.append((
            source_fd,
            _file_state(os.fstat(source_fd)),
            relative_dir.as_posix(),
        ))
        try:
            names = sorted(os.listdir(source_fd))
        except OSError as exc:
            raise FrozenSnapshotError(
                f"cannot enumerate {relative_dir.as_posix()}: {exc}"
            ) from exc
        for name in names:
            if not name or name in {".", ".."} or "/" in name:
                raise FrozenSnapshotError(f"unsafe artifact entry name: {name!r}")
            relative = relative_dir / name
            relative_text = relative.as_posix()
            try:
                before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            except OSError as exc:
                raise FrozenSnapshotError(
                    f"cannot stat artifact entry {relative_text}: {exc}"
                ) from exc
            destination = snapshot_root.joinpath(*relative.parts)
            if stat.S_ISDIR(before.st_mode):
                destination.mkdir(mode=0o700)
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=source_fd)
                except OSError as exc:
                    raise FrozenSnapshotError(
                        f"artifact directory changed during freeze: "
                        f"{relative_text}: {exc}"
                    ) from exc
                directory_fds.append(child_fd)
                if _file_state(os.fstat(child_fd)) != _file_state(before):
                    raise FrozenSnapshotError(
                        f"artifact directory changed during freeze: {relative_text}"
                    )
                walk(child_fd, relative)
                continue
            if stat.S_ISLNK(before.st_mode):
                raise FrozenSnapshotError(
                    f"artifact contains a symlink; publication requires regular "
                    f"files only: {relative_text}"
                )
            if not stat.S_ISREG(before.st_mode):
                raise FrozenSnapshotError(
                    f"artifact contains a non-regular file: {relative_text}"
                )
            try:
                fd = os.open(name, file_flags, dir_fd=source_fd)
            except OSError as exc:
                raise FrozenSnapshotError(
                    f"artifact file changed during freeze: {relative_text}: {exc}"
                ) from exc
            try:
                opened = os.fstat(fd)
                if _file_state(opened) != _file_state(before):
                    raise FrozenSnapshotError(
                        f"artifact file changed during freeze: {relative_text}"
                    )
                scanned = _scan_file(fd, int(opened.st_size))
                after = os.fstat(fd)
                if _file_state(after) != _file_state(opened):
                    raise FrozenSnapshotError(
                        f"artifact file changed while hashing: {relative_text}"
                    )
                entry = _FrozenEntry(
                    relative_path=relative_text,
                    size=int(opened.st_size),
                    sha256=scanned.sha256,
                    sample=scanned.sample,
                    block_sha256=scanned.block_sha256,
                    content=scanned.content,
                    fd=None if scanned.content is not None else fd,
                )
                entries.append(entry)
                if scanned.content is not None:
                    _write_snapshot_bytes(destination, scanned.content)
                else:
                    os.symlink(f"/proc/self/fd/{fd}", destination)
                    retained_file_fds.append(fd)
                    fd = -1
            finally:
                if fd >= 0:
                    os.close(fd)

    try:
        root_fd = os.open(root, directory_flags)
        directory_fds.append(root_fd)
        source_root_stat = os.fstat(root_fd)
        source_root_identity = (
            int(source_root_stat.st_dev),
            int(source_root_stat.st_ino),
        )
        walk(root_fd, PurePosixPath("."))

        # A mutation in a directory already walked changes that directory's
        # ctime/mtime. Recheck every still-open descriptor at one freeze point.
        for directory_fd, expected, relative in directory_records:
            if _file_state(os.fstat(directory_fd)) != expected:
                raise FrozenSnapshotError(
                    f"artifact file set changed while freezing: {relative}"
                )

        by_path = {entry.relative_path: entry for entry in entries}
        if len(by_path) != len(entries):
            raise FrozenSnapshotError("artifact snapshot contains duplicate paths")
        shipcard_entry = by_path.get(SHIPCARD_FILENAME)
        if shipcard_entry is None:
            raise FrozenSnapshotError("canonical shipcard disappeared during freeze")
        if shipcard_entry.sha256 != expected_shipcard_sha256:
            raise FrozenSnapshotError(
                "canonical shipcard changed between preflight and snapshot"
            )

        manifest_rows = [
            {
                "path": entry.relative_path,
                "bytes": entry.size,
                "sha256": entry.sha256,
            }
            for entry in sorted(entries, key=lambda item: item.relative_path)
        ]
        manifest_sha = hashlib.sha256(json.dumps(
            manifest_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        snapshot = _FrozenSnapshot(
            source_root=root,
            source_root_identity=source_root_identity,
            root=snapshot_root,
            root_identity=snapshot_root_identity,
            entries=sorted(entries, key=lambda item: item.relative_path),
            manifest_sha256=manifest_sha,
            total_bytes=sum(entry.size for entry in entries),
        )
        retained_file_fds.clear()  # ownership transferred to entries/snapshot
        return snapshot
    except Exception:
        for entry in entries:
            entry.close()
        for fd in retained_file_fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise
    finally:
        for fd in directory_fds:
            with contextlib.suppress(OSError):
                os.close(fd)


def _verify_declared_weight_hashes(snapshot: _FrozenSnapshot) -> None:
    """Bind the one-pass frozen hashes to a CB exporter content manifest."""
    quant_path = snapshot.root / "quant_config.json"
    if not quant_path.is_file():
        return
    try:
        quant = json.loads(quant_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FrozenSnapshotError(
            f"frozen quant_config.json is unreadable: {exc}"
        ) from exc
    provenance = quant.get("provenance") if isinstance(quant, Mapping) else None
    manifest = provenance.get("weight_content_manifest") if isinstance(
        provenance, Mapping
    ) else None
    if manifest is None:
        return
    declared = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(declared, Mapping):
        raise FrozenSnapshotError("weight content manifest has no file map")
    observed = {
        entry.relative_path: entry
        for entry in snapshot.entries
        if PurePosixPath(entry.relative_path).parent == PurePosixPath(".")
        and entry.relative_path.endswith(".safetensors")
    }
    if set(declared) != set(observed):
        raise FrozenSnapshotError(
            "frozen safetensors set differs from weight content manifest"
        )
    for name, row in declared.items():
        entry = observed[str(name)]
        if (
            not isinstance(row, Mapping)
            or row.get("bytes") != entry.size
            or row.get("sha256") != entry.sha256
        ):
            raise FrozenSnapshotError(
                f"frozen bytes differ from declared weight hash: {name}"
            )


def _replay_frozen_digests(snapshot: _FrozenSnapshot) -> None:
    """Re-read every held descriptor and compare against the frozen digests.

    The Xet transport consumes the frozen link paths rather than the
    streaming verified reader, so nothing re-checks bytes DURING upload;
    replaying the digests across the same descriptors after the commit
    detects any same-inode mutation across the upload window.  (An unlink
    or path swap cannot reach these descriptors at all.)
    """
    for entry in snapshot.entries:
        if entry.fd is None:
            continue
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.pread(entry.fd, SNAPSHOT_BLOCK_BYTES, size)
            if not block:
                break
            digest.update(block)
            size += len(block)
        if size != entry.size or digest.hexdigest() != entry.sha256:
            raise FrozenSnapshotError(
                f"held bytes diverged during upload: {entry.relative_path}"
            )


def _repo_prefix(path_in_repo: str) -> str:
    raw = str(path_in_repo).strip()
    if raw in {"", "."}:
        return ""
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe --path-in-repo: {path_in_repo!r}")
    normalized = pure.as_posix().strip("/")
    if not normalized or normalized == ".":
        return ""
    return normalized


def _remote_path(prefix: str, relative_path: str) -> str:
    return f"{prefix}/{relative_path}" if prefix else relative_path


def _make_additions(
    snapshot: _FrozenSnapshot,
    *,
    prefix: str,
    bindings: _HubBindings,
) -> list[Any]:
    additions: list[Any] = []
    for entry in snapshot.entries:
        remote_path = _remote_path(prefix, entry.relative_path)
        if entry.content is not None:
            operation = bindings.add_type(
                path_in_repo=remote_path,
                path_or_fileobj=entry.content,
            )
        else:
            # Construct a validated operation cheaply, then replace its empty
            # UploadInfo with the digest computed by the frozen scan. This
            # avoids a second 112 GB local read before the actual upload.
            operation = bindings.add_type(
                path_in_repo=remote_path,
                path_or_fileobj=b"",
            )
            info_type = type(operation.upload_info)
            operation.upload_info = info_type(
                sha256=bytes.fromhex(entry.sha256),
                size=entry.size,
                sample=entry.sample,
            )
            # A BufferedIOBase operation suppresses the Hub's Xet transport
            # (huggingface_hub offers "xet" only for path/bytes operations),
            # forcing basic LFS and its 50 GB per-file cap -- which no
            # single-file CB body above 50 GB can pass.  Hand the frozen
            # view's link path over instead: it resolves through this
            # process's held descriptor (`/proc/self/fd/N`), so it cannot be
            # retargeted to other content, and the declared UploadInfo stays
            # the frozen scan's digest.  The reader's per-block streaming
            # re-check is traded for a full post-commit re-verification of
            # the held descriptors (see `_publish_snapshot`), which detects
            # any local mutation across the upload window.
            operation.path_or_fileobj = str(
                snapshot.root / entry.relative_path
            )
        additions.append(operation)
    return additions


def _scope_contains(prefix: str, remote_path: str) -> bool:
    return not prefix or remote_path.startswith(prefix + "/")


def _is_cas_conflict(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    text = str(exc).lower()
    return status in {409, 412} or (
        "parent" in text
        and any(word in text for word in ("commit", "conflict", "changed"))
    )


def _publish_snapshot(
    args: argparse.Namespace,
    snapshot: _FrozenSnapshot,
    bindings: _HubBindings,
) -> int:
    try:
        prefix = _repo_prefix(args.path_in_repo)
    except ValueError as exc:
        print(f"[publish] ERROR: {exc}", file=sys.stderr)
        return 2
    if args.repo_type == "model" and prefix:
        print(
            "[publish] ERROR: model releases must replace the repository root; "
            "--path-in-repo must be '.'",
            file=sys.stderr,
        )
        return 2

    api = bindings.api_type()
    try:
        info = api.repo_info(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
        )
        parent_commit = str(getattr(info, "sha", "") or "").lower()
        if _FULL_COMMIT_RE.fullmatch(parent_commit) is None:
            raise RuntimeError(
                "remote revision did not resolve to one full 40/64-hex commit"
            )
        if args.private and getattr(info, "private", None) is not True:
            raise RuntimeError(
                "--private is an assertion only; configure the existing Hub "
                "repository as private before publishing"
            )
        remote_files = set(api.list_repo_files(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=parent_commit,
        ))
    except Exception as exc:
        print(
            "[publish] REFUSED: could not resolve and enumerate one exact "
            f"remote parent for {args.revision!r}: {exc!r}",
            file=sys.stderr,
        )
        return 1

    additions = _make_additions(snapshot, prefix=prefix, bindings=bindings)
    local_remote_paths = {operation.path_in_repo for operation in additions}
    if len(local_remote_paths) != len(additions):
        print("[publish] ERROR: duplicate remote artifact paths", file=sys.stderr)
        return 2

    # Enumerate deletions against the exact parent, not a moving branch. The
    # Hub intentionally manages root .gitattributes; retain it unless this
    # artifact explicitly supplies its own replacement.
    stale_paths = sorted(
        path
        for path in remote_files
        if _scope_contains(prefix, path)
        and path not in local_remote_paths
        and path != ".gitattributes"
    )
    deletions = [
        bindings.delete_type(path_in_repo=path, is_folder=False)
        for path in stale_paths
    ]

    try:
        # Empty gitignore content is deliberate. Otherwise a remote .gitignore
        # can silently suppress a member of the verified local artifact.
        api.preupload_lfs_files(
            repo_id=args.repo_id,
            additions=additions,
            repo_type=args.repo_type,
            revision=parent_commit,
            free_memory=False,
            gitignore_content="",
        )
        for operation in additions:
            if getattr(operation, "_should_ignore", False):
                raise FrozenSnapshotError(
                    f"Hub attempted to ignore verified file "
                    f"{operation.path_in_repo!r}"
                )
        for entry, operation in zip(snapshot.entries, additions, strict=True):
            if entry.content is None and getattr(
                operation, "_upload_mode", None
            ) != "lfs":
                raise FrozenSnapshotError(
                    f"large frozen file was not classified as LFS: "
                    f"{entry.relative_path}"
                )
    except FrozenSnapshotError as exc:
        print(
            f"[publish] REFUSED: {exc}; no Hub commit was created",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            "[publish] REFUSED: frozen upload preparation failed before the "
            f"Hub commit: {exc!r}",
            file=sys.stderr,
        )
        return 1

    # huggingface_hub normally removes identical additions before building its
    # commit payload. In the all-identical case that early return bypasses
    # parent_commit entirely. Clear only the comparison cache (not upload mode,
    # OID, or uploaded state) so one explicit CAS commit is always attempted.
    for operation in additions:
        if not hasattr(operation, "_remote_oid"):
            print(
                "[publish] REFUSED: installed huggingface_hub lacks the "
                "required no-op/CAS operation state; upgrade it and rerun",
                file=sys.stderr,
            )
            return 1
        operation._remote_oid = None

    commit_message = args.commit_message or (
        f"Publish verified artifact {snapshot.source_root.name}"
    )
    commit_description = (
        f"PrismaQuant frozen snapshot sha256: {snapshot.manifest_sha256}\n"
        f"Verified files: {len(snapshot.entries)}; bytes: "
        f"{snapshot.total_bytes}; stale files deleted: {len(stale_paths)}.\n"
        "Root .gitattributes is intentionally retained when Hub-managed."
    )
    try:
        result = api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            parent_commit=parent_commit,
            operations=[*deletions, *additions],
            commit_message=commit_message,
            commit_description=commit_description,
        )
    except Exception as exc:
        if _is_cas_conflict(exc):
            print(
                "[publish] REFUSED: remote head changed under the "
                "parent_commit CAS. No replacement commit was accepted. "
                "Rerun this publisher from the beginning so it repeats local "
                "freeze/verification and remote enumeration.",
                file=sys.stderr,
            )
        else:
            print(
                "[publish] ERROR: Hub commit outcome was not confirmed under "
                f"the parent_commit CAS: {exc!r}. Inspect the repository "
                "before retrying this publisher.",
                file=sys.stderr,
            )
        return 1

    # The upload read the held descriptors through the frozen link paths (the
    # Xet transport cannot consume the verifying reader), so replay the
    # frozen digests once more across the same descriptors: a divergence
    # here means bytes mutated during the upload window and the commit above
    # must not be announced as verified.
    try:
        _replay_frozen_digests(snapshot)
    except FrozenSnapshotError as exc:
        print(
            "[publish] ERROR: post-commit re-verification of the held "
            f"descriptors failed: {exc}. The Hub commit above may carry "
            "bytes that diverged from the verified snapshot during upload — "
            "inspect the repository and re-publish before announcing.",
            file=sys.stderr,
        )
        return 1

    commit_oid = str(getattr(result, "oid", "") or "").lower()
    if _FULL_COMMIT_RE.fullmatch(commit_oid) is None:
        print(
            "[publish] ERROR: Hub returned no full commit identity after "
            "publication; inspect the repository before retrying",
            file=sys.stderr,
        )
        return 1

    expected_remote = (remote_files - set(stale_paths)) | local_remote_paths
    try:
        observed_remote = set(api.list_repo_files(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=commit_oid,
        ))
    except Exception as exc:
        print(
            "[publish] ERROR: commit exists but its final file set could not "
            f"be audited: {exc!r}; commit={commit_oid}",
            file=sys.stderr,
        )
        return 1
    if observed_remote != expected_remote:
        missing = sorted(expected_remote - observed_remote)
        extra = sorted(observed_remote - expected_remote)
        print(
            "[publish] ERROR: committed remote file set differs from the "
            f"CAS replacement (missing={missing}, extra={extra}); "
            f"commit={commit_oid}",
            file=sys.stderr,
        )
        return 1

    url = getattr(result, "commit_url", None) or commit_oid
    print(
        f"[publish] done: {url} (parent={parent_commit}, "
        f"snapshot_sha256={snapshot.manifest_sha256}, "
        f"stale_deleted={len(stale_paths)}, "
        ".gitattributes retained when Hub-managed)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("artifact_dir", help="exported/ directory to publish")
    ap.add_argument("--repo-id", required=True, help="e.g. rdtand/<name>")
    ap.add_argument("--repo-type", default="model", choices=REPO_TYPES)
    ap.add_argument("--revision", default="main",
                    help="explicit destination branch (default: main)")
    ap.add_argument("--path-in-repo", default=".")
    ap.add_argument(
        "--private",
        action="store_true",
        help="assert that the existing Hub repository is already private",
    )
    ap.add_argument("--commit-message", default=None)
    ap.add_argument("--allow-patterns", nargs="*", default=None)
    ap.add_argument("--ignore-patterns", nargs="*", default=None)
    ap.add_argument(
        "--shipcard",
        default=None,
        help=f"must resolve to <artifact_dir>/{SHIPCARD_FILENAME}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze and replay the local artifact only; make no Hub request",
    )
    ap.add_argument(
        "--force-unverified",
        action="store_true",
        help="publish despite failed evidence slots; requires basename "
             "confirmation and stamps the canonical card",
    )
    ap.add_argument(
        "--confirm-name",
        default=None,
        help="re-typed basename for non-interactive --force-unverified",
    )
    args = ap.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_dir():
        print(
            f"[publish] ERROR: {artifact_dir} is not a directory",
            file=sys.stderr,
        )
        return 2
    try:
        prefix = _repo_prefix(args.path_in_repo)
    except ValueError as exc:
        print(f"[publish] ERROR: {exc}", file=sys.stderr)
        return 2
    if args.repo_type == "model" and prefix:
        print(
            "[publish] ERROR: model releases must replace the repository root; "
            "--path-in-repo must be '.'",
            file=sys.stderr,
        )
        return 2
    if args.allow_patterns is not None or args.ignore_patterns is not None:
        print(
            "[publish] ERROR: publication must snapshot the complete artifact "
            "directory; --allow-patterns/--ignore-patterns are forbidden",
            file=sys.stderr,
        )
        return 2

    shipcard_path = _shipcard_path(artifact_dir, args.shipcard)
    canonical_problem = _canonical_shipcard_problem(
        artifact_dir,
        shipcard_path,
    )
    if canonical_problem is not None:
        print(f"[publish] ERROR: {canonical_problem}", file=sys.stderr)
        return 2
    card, problems = check_shipcard(artifact_dir, shipcard_path)
    if card is None:
        print(
            "[publish] ERROR: the canonical shipcard is malformed; this "
            "structural failure is never overrideable:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    forced_override_confirmed = False
    if problems:
        print(
            f"[publish] REFUSED — {len(problems)} problem(s) with "
            f"{shipcard_path}:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if not args.force_unverified:
            print(
                "[publish] nothing was uploaded and no upload command was "
                "printed. Close the slots or re-run with --force-unverified.",
                file=sys.stderr,
            )
            return 1
        if not _confirm_forced(artifact_dir, args.confirm_name):
            return 2
        forced_override_confirmed = True
        print(
            "[publish] WARNING: publishing an UNVERIFIED artifact by "
            "explicit override.",
            file=sys.stderr,
        )
        try:
            stamp_forced(
                shipcard_path,
                card,
                artifact_dir,
                problems,
                args.repo_id,
            )
        except Exception as exc:
            print(
                f"[publish] ERROR: could not stamp the canonical shipcard: "
                f"{exc!r}",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            "[publish] shipcard preflight OK for "
            f"{json.dumps(str(artifact_dir.resolve()))}"
        )

    # Actual publication requires the low-level API. Import it before hashing a
    # 112 GB artifact, but only after all cheap structural/card refusals.
    bindings: _HubBindings | None = None
    if not args.dry_run:
        try:
            bindings = _load_hub_bindings()
        except Exception as exc:
            print(
                "[publish] REFUSED: huggingface_hub with HfApi.create_commit "
                f"is required ({exc!r}). Install it and rerun this publisher; "
                "no raw hf CLI command is safe for this release contract.",
                file=sys.stderr,
            )
            return 2

    try:
        canonical_bytes = shipcard_path.read_bytes()
        expected_shipcard_sha = hashlib.sha256(canonical_bytes).hexdigest()
    except OSError as exc:
        print(
            f"[publish] REFUSED: canonical shipcard became unreadable: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        # The one-pass safetensors content capture existed for the strict
        # RTX4090 FP8-CB publication replay (Gridbook lane retired 2026-09-02,
        # archive/gridbook_lane_2026-09-02/). No publication asks
        # for it now; the freeze keeps the capability for a future lane.
        snapshot_cm = _freeze_artifact(
            artifact_dir,
            expected_shipcard_sha256=expected_shipcard_sha,
        )
    except Exception as exc:
        print(
            f"[publish] REFUSED: artifact freeze failed: {exc}",
            file=sys.stderr,
        )
        return 1

    with snapshot_cm as snapshot:
        frozen_card_path = snapshot.root / SHIPCARD_FILENAME
        frozen_card, frozen_problems = check_shipcard(
            snapshot.root,
            frozen_card_path,
        )
        if frozen_card is None:
            print(
                "[publish] ERROR: frozen canonical shipcard is malformed; "
                "nothing was uploaded",
                file=sys.stderr,
            )
            return 2
        # The non-forceable strict-snapshot replay and the whole-artifact byte
        # ceiling applied only to the RTX4090 FP8-CB publication, whose
        # Gridbook lane retired 2026-09-02 (see
        # archive/gridbook_lane_2026-09-02/). Every publication now takes the
        # declared-weight-hash replay that used to be the non-strict branch.
        try:
            _verify_declared_weight_hashes(snapshot)
        except FrozenSnapshotError as exc:
            print(f"[publish] REFUSED: {exc}", file=sys.stderr)
            return 1
        if frozen_problems and not forced_override_confirmed:
            print(
                "[publish] REFUSED: frozen snapshot failed authoritative "
                "shipcard replay; nothing was uploaded:",
                file=sys.stderr,
            )
            for problem in frozen_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        if frozen_problems:
            if frozen_card.get("forced_unverified") is not True:
                print(
                    "[publish] ERROR: frozen forced publication lacks its "
                    "forced_unverified stamp",
                    file=sys.stderr,
                )
                return 2
            print(
                "[publish] frozen shipcard remains UNVERIFIED under the "
                "confirmed, recorded override",
                file=sys.stderr,
            )
        else:
            print(
                "[publish] frozen snapshot VERIFIED — "
                f"files={len(snapshot.entries)} bytes={snapshot.total_bytes} "
                f"sha256={snapshot.manifest_sha256}"
            )

        if args.dry_run:
            print(
                "[publish] --dry-run complete; no Hub request was made and "
                "no raw upload command was emitted"
            )
            return 0
        assert bindings is not None
        return _publish_snapshot(args, snapshot, bindings)


if __name__ == "__main__":
    raise SystemExit(main())
