"""Deterministic, shell-free primitives for multi-host campaign execution.

The transport boundary is deliberately small.  A :class:`RunRequest` is
canonical JSON data, never a command string.  ``LocalTransport`` passes its
``argv`` directly to ``subprocess`` with ``shell=False``.  ``SSHTransport``
does the same locally and sends the request as base64-encoded canonical JSON
on stdin to one fixed remote helper command; request fields are never
interpolated into either the local or remote command line.

Completed and detached jobs leave durable request/receipt files.  Tree copies
are content-addressed, reject links and unsafe paths, verify the staged copy,
and use Linux ``renameat2(RENAME_NOREPLACE)`` for atomic no-clobber publish.
The module has no third-party dependencies so the same file can serve as the
remote helper on a minimally provisioned host.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
import ctypes
from dataclasses import dataclass, replace
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal, Protocol, TypeAlias


RUN_REQUEST_SCHEMA = "prismaquant.cluster_transport.run_request.v1"
JOB_RECEIPT_SCHEMA = "prismaquant.cluster_transport.job_receipt.v1"
TREE_MANIFEST_SCHEMA = "prismaquant.cluster_transport.tree_manifest.v1"
TRANSFER_RECEIPT_SCHEMA = "prismaquant.cluster_transport.transfer_receipt.v1"
HELPER_ENVELOPE_SCHEMA = "prismaquant.cluster_transport.helper_envelope.v1"
HELPER_RESPONSE_SCHEMA = "prismaquant.cluster_transport.helper_response.v1"
HELPER_ERROR_SCHEMA = "prismaquant.cluster_transport.helper_error.v1"

_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SSH_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,254}")
_SAFE_REMOTE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+")
_FINAL_STATES = frozenset({"succeeded", "failed", "timed_out", "transport_error"})
_NVIDIA_QUERY_FIELDS = (
    "timestamp",
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)

# Fixed, stdlib-only launcher for every SSH job action and detached worker.
# The mutable helper pathname is never handed to Python as a script.  This
# launcher executes the held bytes as a registered module, injects one reserved
# immutable launch context, and replaces historical helpers' direct-path
# worker spawn before calling their entry point.  The same launcher is then
# used recursively for ``--local-worker`` so neither current nor sealed source
# bytes can escape the digest gate.
_SSH_HELPER_LAUNCHER_PROGRAM = r'''
import base64, hashlib, os, re, stat, sys, types

SAFE = re.compile(r"[A-Za-z0-9._-]+")
HEX = re.compile(r"[0-9a-f]{64}")

def open_held_helper(raw):
    if (not isinstance(raw, str) or not raw.startswith("/")
            or "\x00" in raw):
        raise ValueError("unsafe remote helper path")
    parts = raw.split("/")[1:]
    if (not parts or any(not part or part in (".", "..")
                         or SAFE.fullmatch(part) is None for part in parts)):
        raise ValueError("unsafe remote helper path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not directory:
        raise ValueError("descriptor-safe remote helper traversal unavailable")
    parent_fd = os.open("/", os.O_RDONLY | directory | nofollow | cloexec)
    try:
        for component in parts[:-1]:
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("remote helper ancestry is not a real directory")
            child_fd = os.open(
                component,
                os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=parent_fd,
            )
            opened = os.fstat(child_fd)
            if (not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)):
                os.close(child_fd)
                raise ValueError("remote helper ancestry changed while opening")
            os.close(parent_fd)
            parent_fd = child_fd
        before = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("remote helper is not a regular file")
        helper_fd = os.open(
            parts[-1],
            os.O_RDONLY | nofollow | cloexec | nonblock,
            dir_fd=parent_fd,
        )
        opened = os.fstat(helper_fd)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            os.close(helper_fd)
            raise ValueError("remote helper identity changed while opening")
        return helper_fd, opened
    finally:
        os.close(parent_fd)

try:
    if len(sys.argv) < 6 or HEX.fullmatch(sys.argv[2]) is None:
        raise ValueError("remote helper launcher arguments are invalid")
    helper_path = sys.argv[1]
    expected_digest = sys.argv[2]
    launcher_source = base64.b64decode(sys.argv[3].encode("ascii"), validate=True)
    if (HEX.fullmatch(sys.argv[4]) is None
            or hashlib.sha256(launcher_source).hexdigest() != sys.argv[4]):
        raise ValueError("remote helper launcher identity is invalid")
    helper_args = sys.argv[5:]
    remote_mode = helper_args == ["--remote-helper"] or (
            len(helper_args) == 3
            and helper_args[0] == "--remote-helper"
            and helper_args[1] == "--remote-helper-state-root")
    worker_mode = len(helper_args) == 4 and helper_args[0] == "--local-worker"
    verify_mode = helper_args == ["--verify-held-helper"]
    if not (remote_mode or worker_mode or verify_mode):
        raise ValueError("remote helper action argv is invalid")
    descriptor, opened = open_held_helper(helper_path)
    try:
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mtime_ns, opened.st_ctime_ns):
            raise ValueError("remote helper changed while reading")
        if len(source) != opened.st_size:
            raise ValueError("remote helper size changed while reading")
        if hashlib.sha256(source).hexdigest() != expected_digest:
            raise ValueError("remote helper digest differs from bootstrap plan")
        if verify_mode:
            raise SystemExit(0)
        sys.argv = [helper_path, *helper_args]
        module_name = "_prismaquant_held_remote_helper"
        module = types.ModuleType(module_name)
        module.__file__ = helper_path
        module.__package__ = None
        module.__cached__ = None
        module.__dict__["_PRISMAQUANT_HELD_HELPER_CONTEXT"] = (
            "prismaquant.cluster_transport.held_helper.v1",
            helper_path,
            expected_digest,
            launcher_source.decode("utf-8"),
        )
        sys.modules[module_name] = module
        exec(compile(source, helper_path, "exec", dont_inherit=True), module.__dict__)
        transport_type = getattr(module, "LocalTransport", None)
        entry_point = getattr(module, "_main", None)
        if transport_type is None or not callable(entry_point):
            raise ValueError("remote helper lacks the pinned execution surface")
        original_start = transport_type.start

        def held_start(instance, request):
            original_popen = instance._popen_impl

            def held_popen(argv, *args, **kwargs):
                candidate = list(argv)
                if (len(candidate) == 9 and candidate[1:4] == ["-P", "-B", "-s"]
                        and candidate[4] == helper_path
                        and candidate[5] == "--local-worker"):
                    candidate = [
                        candidate[0], "-P", "-B", "-s", "-c",
                        launcher_source.decode("utf-8"), helper_path,
                        expected_digest,
                        base64.b64encode(launcher_source).decode("ascii"),
                        hashlib.sha256(launcher_source).hexdigest(),
                        *candidate[5:],
                    ]
                expected_prefix = [
                    candidate[0], "-P", "-B", "-s", "-c",
                    launcher_source.decode("utf-8"), helper_path,
                    expected_digest,
                    base64.b64encode(launcher_source).decode("ascii"),
                    hashlib.sha256(launcher_source).hexdigest(),
                    "--local-worker",
                ]
                if len(candidate) != 14 or candidate[:11] != expected_prefix:
                    raise ValueError("detached helper worker argv escaped held launcher")
                return original_popen(candidate, *args, **kwargs)

            instance._popen_impl = held_popen
            try:
                return original_start(instance, request)
            finally:
                instance._popen_impl = original_popen

        transport_type.start = held_start
        try:
            raise SystemExit(entry_point(sys.argv[1:]))
        finally:
            sys.modules.pop(module_name, None)
    finally:
        os.close(descriptor)
except BaseException as exc:
    if isinstance(exc, SystemExit):
        raise
    sys.stderr.write(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(126)
'''.strip()

JobState: TypeAlias = Literal[
    "running", "succeeded", "failed", "timed_out", "transport_error"
]


class ClusterTransportError(RuntimeError):
    """Base error for a refused or failed transport operation."""


class TelemetryUnavailableError(ClusterTransportError):
    """A telemetry management query was unavailable without losing job state."""


class TelemetryIntegrityError(ClusterTransportError):
    """Telemetry bytes or protocol state were present but invalid."""


class _SSHInvocationUnavailableError(ClusterTransportError):
    """The SSH helper process could not be reached or did not complete."""


class JobNotFoundError(ClusterTransportError):
    """The requested job ID has no durable transport claim."""


class JobConflictError(ClusterTransportError):
    """A job ID already exists and therefore cannot be overwritten."""


class ManifestError(ClusterTransportError):
    """A file/tree manifest is unsafe, noncanonical, or fails verification."""


class _CompletedProcess(Protocol):
    returncode: int
    stdout: bytes | str | None
    stderr: bytes | str | None


RunImplementation: TypeAlias = Callable[..., _CompletedProcess]
PopenImplementation: TypeAlias = Callable[..., Any]


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_process_identity(pid: int) -> Mapping[str, object]:
    """Return a boot-scoped PID identity that rejects PID reuse."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process identity PID must be positive")
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii",
        ).strip()
        stat_payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise ProcessLookupError(pid) from exc
    closing = stat_payload.rfind(")")
    if closing < 0:
        raise ValueError("process stat has no command terminator")
    fields = stat_payload[closing + 2:].split()
    if len(fields) <= 19:
        raise ValueError("process stat is incomplete")
    # ``kill(pid, 0)`` succeeds for a zombie until its parent reaps it.  Such
    # a worker can never replace the durable running receipt, so treating it
    # as live would strand the job forever.  Linux documents ``Z`` as zombie
    # and ``X``/``x`` as dead (the latter spellings are rare but possible).
    if fields[0] in {"Z", "X", "x"}:
        raise ProcessLookupError(pid)
    start_ticks = int(fields[19])
    if start_ticks <= 0 or not re.fullmatch(
        r"[0-9a-fA-F-]{36}", boot_id,
    ):
        raise ValueError("process identity fields are invalid")
    return {
        "schema": "prismaquant.cluster_transport.process_identity.v1",
        "pid": pid,
        "boot_id": boot_id.lower(),
        "start_ticks": start_ticks,
    }


def _reject_duplicate_members(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: object, *, where: str = "value") -> bytes:
    """Return the one canonical UTF-8 encoding accepted by this protocol."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc


def _strict_json_bytes(payload: bytes, *, where: str) -> object:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_members)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{where} is not strict JSON") from exc
    if canonical_json_bytes(value, where=where) != payload:
        raise ValueError(f"{where} is not canonically encoded JSON")
    return value


def read_regular_file_nofollow(
    path: str | Path, *, where: str = "stored file",
) -> bytes:
    """Read one stable regular file without traversing any symlink.

    Verification inputs are attacker-controlled durable state, so checking
    ``Path.is_symlink()`` before ``Path.read_bytes()`` is not sufficient: an
    ancestor can be a link and the final component can race between those two
    operations.  Walk the directory ancestry with ``openat``-style held file
    descriptors, open the final component with ``O_NOFOLLOW``, and bind the
    bytes to matching lstat/fstat identities before returning them.
    """

    candidate = Path(path)
    if "\0" in str(candidate) or ".." in candidate.parts:
        raise ClusterTransportError(f"{where} path is unsafe: {candidate}")
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if absolute == Path(absolute.anchor):
        raise ClusterTransportError(f"{where} is not a regular file: {candidate}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ClusterTransportError(
            f"{where} cannot be read safely: O_NOFOLLOW is unavailable"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | nofollow
    )

    descriptor: int | None = None
    file_descriptor: int | None = None
    try:
        advertised = absolute.lstat()
        if not stat.S_ISREG(advertised.st_mode):
            raise ClusterTransportError(
                f"{where} is not a regular file: {candidate}"
            )
        descriptor = os.open(absolute.anchor, directory_flags)
        for component in absolute.parts[1:-1]:
            before = os.stat(
                component, dir_fd=descriptor, follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise ClusterTransportError(
                    f"{where} directory ancestry is not real: {candidate}"
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                os.close(child)
                raise ClusterTransportError(
                    f"{where} directory ancestry changed: {candidate}"
                )
            os.close(descriptor)
            descriptor = child

        filename = absolute.parts[-1]
        before = os.stat(
            filename, dir_fd=descriptor, follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise ClusterTransportError(
                f"{where} is not a regular file: {candidate}"
            )
        file_descriptor = os.open(
            filename, file_flags, dir_fd=descriptor,
        )
        opened = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
            or (opened.st_dev, opened.st_ino)
            != (advertised.st_dev, advertised.st_ino)
        ):
            raise ClusterTransportError(
                f"{where} changed while being opened: {candidate}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(file_descriptor)
        current = os.stat(
            filename, dir_fd=descriptor, follow_symlinks=False,
        )
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(opened, key) != getattr(after, key) for key in stable_fields)
            or (current.st_dev, current.st_ino)
            != (opened.st_dev, opened.st_ino)
            or len(payload) != after.st_size
        ):
            raise ClusterTransportError(
                f"{where} changed while being read: {candidate}"
            )
        return payload
    except ClusterTransportError:
        raise
    except OSError as exc:
        raise ClusterTransportError(
            f"cannot read {where} without following links: {candidate}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if descriptor is not None:
            os.close(descriptor)


def list_regular_file_children_nofollow(
    directory: str | Path,
    *,
    where: str,
    suffix: str = ".json",
    allow_missing: bool = False,
) -> tuple[Path, ...]:
    """List matching regular children of one real directory, without links."""

    candidate = Path(directory)
    if "\0" in str(candidate) or ".." in candidate.parts:
        raise ClusterTransportError(f"{where} path is unsafe: {candidate}")
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise ClusterTransportError(
            f"{where} cannot be inspected safely: O_NOFOLLOW is unavailable"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            try:
                before = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False,
                )
            except FileNotFoundError:
                if allow_missing:
                    return ()
                raise
            if not stat.S_ISDIR(before.st_mode):
                raise ClusterTransportError(
                    f"{where} is not a real directory: {candidate}"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                os.close(child)
                raise ClusterTransportError(
                    f"{where} directory changed: {candidate}"
                )
            os.close(descriptor)
            descriptor = child
        paths: list[Path] = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ClusterTransportError(
                        f"{where} contains a non-regular child: {entry.name}"
                    )
                if suffix and not entry.name.endswith(suffix):
                    continue
                paths.append(candidate / entry.name)
        return tuple(sorted(paths, key=lambda item: item.name))
    except ClusterTransportError:
        raise
    except OSError as exc:
        raise ClusterTransportError(
            f"cannot inspect {where} without following links: {candidate}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def canonical_json_sha256(value: object, *, where: str = "value") -> str:
    return hashlib.sha256(canonical_json_bytes(value, where=where)).hexdigest()


def write_exact_bytes_no_clobber(path: str | Path, payload: bytes) -> None:
    """Durably publish bytes, accepting only an identical regular-file prior."""

    destination = Path(path)
    if not isinstance(payload, bytes):
        raise ClusterTransportError("no-clobber payload must be bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def existing_matches() -> bool:
        try:
            before = destination.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(before.st_mode):
            raise ClusterTransportError(
                f"existing no-clobber destination is not a regular file: "
                f"{destination}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
            ):
                raise ClusterTransportError(
                    f"existing no-clobber destination changed: {destination}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks) == payload
        finally:
            os.close(descriptor)

    if destination.exists():
        if existing_matches():
            return
        raise ClusterTransportError(
            f"existing no-clobber destination differs: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            if not existing_matches():
                raise ClusterTransportError(
                    f"racing no-clobber destination differs: {destination}"
                ) from exc
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _require_job_id(value: object, *, where: str = "job_id") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    job_id = value
    if _JOB_ID.fullmatch(job_id) is None:
        raise ValueError(
            f"{where} must match {_JOB_ID.pattern!r}; got {job_id!r}"
        )
    return job_id


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    digest = value
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return digest


def _bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _decode_b64(value: object, *, where: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{where} is not valid base64") from exc


def _encode_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


@dataclass(frozen=True)
class RunRequest:
    """A canonical argv-based process request.

    ``env`` is stored as a sorted tuple so mapping insertion order cannot
    affect identity.  With ``inherit_env=True`` those values are overrides of
    the destination host's environment; otherwise they are the complete
    child environment.
    """

    job_id: str
    argv: tuple[str, ...]
    cwd: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float | None = None
    stdin: bytes = b""
    inherit_env: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_job_id(self.job_id))
        if not isinstance(self.argv, (tuple, list)) or any(
            not isinstance(part, str) for part in self.argv
        ):
            raise TypeError("argv must be a sequence of strings")
        argv = tuple(self.argv)
        if not argv or any(not part or "\0" in part for part in argv):
            raise ValueError("argv must contain nonempty, NUL-free strings")
        object.__setattr__(self, "argv", argv)

        if self.cwd is not None:
            if not isinstance(self.cwd, str):
                raise TypeError("cwd must be a string or None")
            cwd = self.cwd
            if "\0" in cwd or not Path(cwd).is_absolute():
                raise ValueError("cwd must be a NUL-free absolute path")
            object.__setattr__(self, "cwd", cwd)

        raw_env: Iterable[tuple[object, object]]
        if isinstance(self.env, Mapping):
            raw_env = self.env.items()
        else:
            raw_env = self.env
        normalized: dict[str, str] = {}
        for raw_name, raw_value in raw_env:
            if not isinstance(raw_name, str) or not isinstance(raw_value, str):
                raise TypeError("environment names and values must be strings")
            name, value = raw_name, raw_value
            if _ENV_NAME.fullmatch(name) is None or "\0" in value:
                raise ValueError(f"invalid environment entry {name!r}")
            if name in normalized:
                raise ValueError(f"duplicate environment name {name!r}")
            normalized[name] = value
        object.__setattr__(self, "env", tuple(sorted(normalized.items())))

        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise TypeError("timeout_seconds must be a number or None")
            timeout = float(self.timeout_seconds)
            if not math.isfinite(timeout) or timeout <= 0.0:
                raise ValueError("timeout_seconds must be finite and positive")
            object.__setattr__(self, "timeout_seconds", timeout)
        if not isinstance(self.stdin, bytes):
            raise TypeError("stdin must be bytes")
        if not isinstance(self.inherit_env, bool):
            raise TypeError("inherit_env must be bool")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": RUN_REQUEST_SCHEMA,
            "job_id": self.job_id,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": [[name, value] for name, value in self.env],
            "timeout_seconds": self.timeout_seconds,
            "stdin_b64": _encode_b64(self.stdin),
            "inherit_env": self.inherit_env,
        }

    def as_dict(self) -> dict[str, object]:
        """Compatibility spelling for manifest/coordinator consumers."""
        return self.to_payload()

    @classmethod
    def from_payload(cls, payload: object) -> RunRequest:
        if not isinstance(payload, Mapping) or payload.get("schema") != RUN_REQUEST_SCHEMA:
            raise ValueError("run request has the wrong schema")
        if set(payload) != {
            "schema",
            "job_id",
            "argv",
            "cwd",
            "env",
            "timeout_seconds",
            "stdin_b64",
            "inherit_env",
        }:
            raise ValueError("run request has unknown or missing fields")
        argv = payload["argv"]
        env = payload["env"]
        if not isinstance(argv, list) or not isinstance(env, list):
            raise ValueError("run request argv/env must be arrays")
        env_pairs: list[tuple[str, str]] = []
        for pair in env:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("run request env entries must be pairs")
            env_pairs.append((pair[0], pair[1]))  # type: ignore[arg-type]
        return cls(
            job_id=payload["job_id"],  # type: ignore[arg-type]
            argv=tuple(argv),
            cwd=payload["cwd"],  # type: ignore[arg-type]
            env=tuple(env_pairs),
            timeout_seconds=payload["timeout_seconds"],  # type: ignore[arg-type]
            stdin=_decode_b64(payload["stdin_b64"], where="run request stdin"),
            inherit_env=payload["inherit_env"],
        )

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_payload(), where="run request")

    @property
    def request_sha256(self) -> str:
        return self.identity_sha256


@dataclass(frozen=True)
class JobReceipt:
    """Durable process state and byte-exact captured output."""

    job_id: str
    request_sha256: str
    state: JobState
    started_ns: int
    finished_ns: int | None
    returncode: int | None
    pid: int | None
    stdout: bytes
    stderr: bytes
    transport: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_job_id(self.job_id))
        object.__setattr__(
            self,
            "request_sha256",
            _require_sha256(self.request_sha256, where="request_sha256"),
        )
        if self.state not in {"running", *_FINAL_STATES}:
            raise ValueError(f"invalid job state {self.state!r}")
        if (
            isinstance(self.started_ns, bool)
            or not isinstance(self.started_ns, int)
            or self.started_ns <= 0
        ):
            raise ValueError("started_ns must be a positive integer")
        if self.state == "running":
            if self.finished_ns is not None or self.returncode is not None:
                raise ValueError("running receipt cannot be finished")
        else:
            if (
                isinstance(self.finished_ns, bool)
                or not isinstance(self.finished_ns, int)
                or self.finished_ns < self.started_ns
            ):
                raise ValueError("final receipt needs finished_ns >= started_ns")
        if self.state == "succeeded" and (
            isinstance(self.returncode, bool)
            or not isinstance(self.returncode, int)
            or self.returncode != 0
        ):
            raise ValueError("succeeded receipt must have returncode 0")
        if self.state == "failed" and (
            isinstance(self.returncode, bool)
            or not isinstance(self.returncode, int)
            or self.returncode == 0
        ):
            raise ValueError("failed receipt must have a nonzero returncode")
        if self.state in {"timed_out", "transport_error"} and self.returncode is not None:
            raise ValueError(f"{self.state} receipt cannot have a returncode")
        if self.pid is not None and (
            isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0
        ):
            raise ValueError("pid must be a positive integer or None")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("stdout/stderr must be bytes")
        if (
            not isinstance(self.transport, str)
            or not self.transport
            or "\0" in self.transport
        ):
            raise ValueError("transport must be a nonempty NUL-free string")

    @property
    def is_final(self) -> bool:
        return self.state in _FINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == "succeeded"

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": JOB_RECEIPT_SCHEMA,
            "job_id": self.job_id,
            "request_sha256": self.request_sha256,
            "state": self.state,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "returncode": self.returncode,
            "pid": self.pid,
            "stdout_b64": _encode_b64(self.stdout),
            "stdout_sha256": hashlib.sha256(self.stdout).hexdigest(),
            "stderr_b64": _encode_b64(self.stderr),
            "stderr_sha256": hashlib.sha256(self.stderr).hexdigest(),
            "transport": self.transport,
        }

    def as_dict(self) -> dict[str, object]:
        """Compatibility spelling for manifest/coordinator consumers."""
        return self.to_payload()

    @property
    def receipt_sha256(self) -> str:
        return canonical_json_sha256(self.to_payload(), where="job receipt")

    @classmethod
    def from_payload(cls, payload: object) -> JobReceipt:
        expected = {
            "schema",
            "job_id",
            "request_sha256",
            "state",
            "started_ns",
            "finished_ns",
            "returncode",
            "pid",
            "stdout_b64",
            "stdout_sha256",
            "stderr_b64",
            "stderr_sha256",
            "transport",
        }
        if not isinstance(payload, Mapping) or payload.get("schema") != JOB_RECEIPT_SCHEMA:
            raise ValueError("job receipt has the wrong schema")
        if set(payload) != expected:
            raise ValueError("job receipt has unknown or missing fields")
        stdout = _decode_b64(payload["stdout_b64"], where="receipt stdout")
        stderr = _decode_b64(payload["stderr_b64"], where="receipt stderr")
        if hashlib.sha256(stdout).hexdigest() != _require_sha256(
            payload["stdout_sha256"], where="stdout_sha256"
        ):
            raise ValueError("receipt stdout digest mismatch")
        if hashlib.sha256(stderr).hexdigest() != _require_sha256(
            payload["stderr_sha256"], where="stderr_sha256"
        ):
            raise ValueError("receipt stderr digest mismatch")
        return cls(
            job_id=payload["job_id"],  # type: ignore[arg-type]
            request_sha256=payload["request_sha256"],  # type: ignore[arg-type]
            state=payload["state"],  # type: ignore[arg-type]
            started_ns=payload["started_ns"],  # type: ignore[arg-type]
            finished_ns=payload["finished_ns"],  # type: ignore[arg-type]
            returncode=payload["returncode"],  # type: ignore[arg-type]
            pid=payload["pid"],  # type: ignore[arg-type]
            stdout=stdout,
            stderr=stderr,
            transport=payload["transport"],  # type: ignore[arg-type]
        )


def _safe_relative_posix(raw: object, *, where: str) -> str:
    if not isinstance(raw, str) or not raw or "\0" in raw or "\\" in raw:
        raise ManifestError(f"{where} must be a nonempty safe POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"{where} contains traversal or is absolute: {raw!r}")
    return path.as_posix()


def _require_no_symlink_components(path: Path, *, where: str) -> None:
    if ".." in path.parts:
        raise ManifestError(f"{where} cannot contain traversal: {path}")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ManifestError(f"{where} component is unavailable: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ManifestError(f"{where} cannot traverse a symlink: {current}")


@dataclass(frozen=True, order=True)
class ManifestEntry:
    path: str
    kind: Literal["directory", "file"]
    size_bytes: int
    sha256: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative_posix(self.path, where="entry.path"))
        if self.kind not in {"directory", "file"}:
            raise ManifestError(f"invalid manifest entry kind {self.kind!r}")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ManifestError("entry size_bytes must be a nonnegative integer")
        if self.kind == "directory":
            if self.size_bytes != 0 or self.sha256 is not None:
                raise ManifestError("directory entries cannot carry bytes or a digest")
        else:
            object.__setattr__(
                self,
                "sha256",
                _require_sha256(self.sha256, where=f"entry[{self.path}].sha256"),
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TreeManifest:
    root_kind: Literal["directory", "file"]
    entries: tuple[ManifestEntry, ...]

    def __post_init__(self) -> None:
        if self.root_kind not in {"directory", "file"}:
            raise ManifestError(f"invalid root_kind {self.root_kind!r}")
        entries = tuple(self.entries)
        if entries != tuple(sorted(entries, key=lambda entry: entry.path)):
            raise ManifestError("manifest entries must be sorted by path")
        paths = [entry.path for entry in entries]
        if len(paths) != len(set(paths)):
            raise ManifestError("manifest contains duplicate paths")
        if self.root_kind == "file":
            if len(entries) != 1 or entries[0].path != "payload" or entries[0].kind != "file":
                raise ManifestError("file manifest must contain one 'payload' file")
        object.__setattr__(self, "entries", entries)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries if entry.kind == "file")

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_payload(), where="tree manifest")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": TREE_MANIFEST_SCHEMA,
            "root_kind": self.root_kind,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload(), where="tree manifest")

    @classmethod
    def from_payload(cls, payload: object) -> TreeManifest:
        if not isinstance(payload, Mapping) or payload.get("schema") != TREE_MANIFEST_SCHEMA:
            raise ManifestError("tree manifest has the wrong schema")
        if set(payload) != {"schema", "root_kind", "entries"}:
            raise ManifestError("tree manifest has unknown or missing fields")
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list):
            raise ManifestError("tree manifest entries must be an array")
        entries: list[ManifestEntry] = []
        for index, row in enumerate(raw_entries):
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "kind",
                "size_bytes",
                "sha256",
            }:
                raise ManifestError(f"manifest entry {index} has invalid fields")
            entries.append(
                ManifestEntry(
                    path=row["path"],  # type: ignore[arg-type]
                    kind=row["kind"],  # type: ignore[arg-type]
                    size_bytes=row["size_bytes"],  # type: ignore[arg-type]
                    sha256=row["sha256"],  # type: ignore[arg-type]
                )
            )
        return cls(
            root_kind=payload["root_kind"],  # type: ignore[arg-type]
            entries=tuple(entries),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> TreeManifest:
        return cls.from_payload(_strict_json_bytes(payload, where="tree manifest"))


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot open regular file without following links: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise ManifestError(f"manifest source is not a regular file: {path}")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def build_tree_manifest(source: str | Path) -> TreeManifest:
    """Hash a regular file or directory without following any symlink."""
    root = Path(source)
    _require_no_symlink_components(root, where="manifest source")
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ManifestError(f"manifest source is unavailable: {root}") from exc
    if stat.S_ISLNK(root_mode):
        raise ManifestError(f"manifest source cannot be a symlink: {root}")
    if stat.S_ISREG(root_mode):
        size, digest = _hash_regular_file(root)
        return TreeManifest(
            root_kind="file",
            entries=(ManifestEntry("payload", "file", size, digest),),
        )
    if not stat.S_ISDIR(root_mode):
        raise ManifestError(f"manifest source must be a regular file or directory: {root}")

    entries: list[ManifestEntry] = []

    def visit(directory: Path, relative: PurePosixPath | None) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ManifestError(f"cannot enumerate manifest directory: {directory}") from exc
        for child in children:
            if "/" in child.name or "\0" in child.name:
                raise ManifestError(f"unsafe directory member {child.name!r}")
            rel = PurePosixPath(child.name) if relative is None else relative / child.name
            rel_text = _safe_relative_posix(rel.as_posix(), where="source member")
            try:
                mode = child.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ManifestError(f"cannot stat manifest member: {child.path}") from exc
            if stat.S_ISLNK(mode):
                raise ManifestError(f"manifest trees cannot contain symlinks: {child.path}")
            child_path = Path(child.path)
            if stat.S_ISDIR(mode):
                entries.append(ManifestEntry(rel_text, "directory", 0, None))
                visit(child_path, rel)
            elif stat.S_ISREG(mode):
                size, digest = _hash_regular_file(child_path)
                entries.append(ManifestEntry(rel_text, "file", size, digest))
            else:
                raise ManifestError(f"unsupported manifest member type: {child.path}")

    visit(root, None)
    entries.sort(key=lambda entry: entry.path)
    return TreeManifest(root_kind="directory", entries=tuple(entries))


def verify_tree_manifest(source: str | Path, manifest: TreeManifest) -> None:
    actual = build_tree_manifest(source)
    if actual != manifest:
        raise ManifestError(
            "tree content differs from manifest: "
            f"expected={manifest.identity_sha256} actual={actual.identity_sha256}"
        )


@dataclass(frozen=True)
class TransferReceipt:
    destination: str
    manifest_sha256: str
    total_bytes: int
    entry_count: int
    completed_ns: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": TRANSFER_RECEIPT_SCHEMA,
            "destination": self.destination,
            "manifest_sha256": self.manifest_sha256,
            "total_bytes": self.total_bytes,
            "entry_count": self.entry_count,
            "completed_ns": self.completed_ns,
        }


@dataclass(frozen=True)
class HelperInstallSpec:
    """Byte-verifiable source needed to provision an SSH helper once.

    SSHTransport intentionally does not pipe source code to an ad-hoc shell.
    A coordinator can transfer these exact bytes with its ordinary verified,
    no-clobber artifact path, then construct the transport against
    ``remote_path``.  Until that happens, SSH fails closed at process exit.
    """

    remote_path: str
    source_sha256: str
    size_bytes: int
    source: bytes

    def __post_init__(self) -> None:
        _validate_remote_executable(self.remote_path, where="remote_path")
        _require_sha256(self.source_sha256, where="source_sha256")
        if self.size_bytes != len(self.source):
            raise ValueError("helper install size does not match source bytes")
        if hashlib.sha256(self.source).hexdigest() != self.source_sha256:
            raise ValueError("helper install digest does not match source bytes")

    def to_payload(self) -> dict[str, object]:
        return {
            "remote_path": self.remote_path,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
            "source_b64": _encode_b64(self.source),
        }


def canonical_helper_source() -> bytes:
    """Return the exact standalone stdlib-only helper implementation."""
    return Path(__file__).read_bytes()


def _copy_file_verified(source: Path, destination: Path, entry: ManifestEntry) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(source, flags)
    destination_fd: int | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise ManifestError(f"copy source is not a regular file: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while True:
            block = os.read(source_fd, 8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise ManifestError(f"source changed while copying {source}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without replacing any destination (Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if source.is_file():
            try:
                os.link(source, destination, follow_symlinks=False)
                source.unlink()
                return
            except FileExistsError:
                raise
            except OSError as exc:
                raise ClusterTransportError(
                    "atomic no-clobber file publish is unavailable"
                ) from exc
        raise ClusterTransportError(
            "atomic no-clobber directory publish requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


@dataclass(frozen=True)
class GpuSample:
    timestamp: str
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    gpu_utilization_pct: float | None
    memory_utilization_pct: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    temperature_c: float | None
    power_w: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("GPU index must be a nonnegative integer")
        if any(
            not isinstance(value, str) or not value or "\0" in value
            for value in (self.timestamp, self.name, self.uuid, self.pci_bus_id)
        ):
            raise ValueError("GPU identity fields are incomplete")
        for name in ("gpu_utilization_pct", "memory_utilization_pct"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 100.0
            ):
                raise ValueError(f"{name} must be between 0 and 100")
        for name in ("memory_used_mib", "memory_total_mib"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be nonnegative")
        for name in ("temperature_c", "power_w"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def to_payload(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "memory_utilization_pct": self.memory_utilization_pct,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
        }

    @classmethod
    def from_payload(cls, payload: object) -> GpuSample:
        if not isinstance(payload, Mapping):
            raise ValueError("GPU sample must be an object")
        expected = {
            "timestamp",
            "index",
            "name",
            "uuid",
            "pci_bus_id",
            "gpu_utilization_pct",
            "memory_utilization_pct",
            "memory_used_mib",
            "memory_total_mib",
            "temperature_c",
            "power_w",
        }
        if set(payload) != expected:
            raise ValueError("GPU sample has unknown or missing fields")

        def optional_float(name: str) -> float | None:
            value = payload[name]
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"GPU sample {name} must be numeric or null")
            return float(value)

        if isinstance(payload["index"], bool) or not isinstance(payload["index"], int):
            raise ValueError("GPU sample index must be an integer")

        return cls(
            timestamp=payload["timestamp"],  # type: ignore[arg-type]
            index=payload["index"],
            name=payload["name"],  # type: ignore[arg-type]
            uuid=payload["uuid"],  # type: ignore[arg-type]
            pci_bus_id=payload["pci_bus_id"],  # type: ignore[arg-type]
            gpu_utilization_pct=optional_float("gpu_utilization_pct"),
            memory_utilization_pct=optional_float("memory_utilization_pct"),
            memory_used_mib=optional_float("memory_used_mib"),
            memory_total_mib=optional_float("memory_total_mib"),
            temperature_c=optional_float("temperature_c"),
            power_w=optional_float("power_w"),
        )


@dataclass(frozen=True)
class TelemetrySnapshot:
    captured_ns: int
    host_mem_available_bytes: int
    gpus: tuple[GpuSample, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.captured_ns, bool)
            or not isinstance(self.captured_ns, int)
            or self.captured_ns <= 0
        ):
            raise ValueError("captured_ns must be a positive integer")
        if (
            isinstance(self.host_mem_available_bytes, bool)
            or not isinstance(self.host_mem_available_bytes, int)
            or self.host_mem_available_bytes < 0
        ):
            raise ValueError("host_mem_available_bytes must be nonnegative")
        gpus = tuple(self.gpus)
        indexes = [gpu.index for gpu in gpus]
        uuids = [gpu.uuid for gpu in gpus]
        if len(indexes) != len(set(indexes)) or len(uuids) != len(set(uuids)):
            raise ValueError("telemetry snapshot contains duplicate GPUs")
        object.__setattr__(self, "gpus", gpus)

    def to_payload(self) -> dict[str, object]:
        return {
            "captured_ns": self.captured_ns,
            "host_mem_available_bytes": self.host_mem_available_bytes,
            "gpus": [gpu.to_payload() for gpu in self.gpus],
        }

    @classmethod
    def from_payload(cls, payload: object) -> TelemetrySnapshot:
        if not isinstance(payload, Mapping) or set(payload) != {
            "captured_ns",
            "host_mem_available_bytes",
            "gpus",
        }:
            raise ValueError("telemetry snapshot has invalid fields")
        gpus = payload["gpus"]
        if not isinstance(gpus, list):
            raise ValueError("telemetry snapshot gpus must be an array")
        return cls(
            captured_ns=payload["captured_ns"],  # type: ignore[arg-type]
            host_mem_available_bytes=payload["host_mem_available_bytes"],  # type: ignore[arg-type]
            gpus=tuple(GpuSample.from_payload(row) for row in gpus),
        )


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class UtilizationMetricSummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    active_fraction: float | None
    active_fraction_gt_0: float | None
    active_fraction_gt_50: float | None
    active_fraction_gt_90: float | None

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "active_fraction": self.active_fraction,
            "active_fraction_gt_0": self.active_fraction_gt_0,
            "active_fraction_gt_50": self.active_fraction_gt_50,
            "active_fraction_gt_90": self.active_fraction_gt_90,
        }


@dataclass(frozen=True)
class UtilizationSummary:
    snapshot_count: int
    gpu_sample_count: int
    gpu_count: int
    active_threshold_pct: float
    gpu_utilization: UtilizationMetricSummary
    memory_utilization: UtilizationMetricSummary
    host_mem_available_bytes: DistributionSummary

    def to_payload(self) -> dict[str, object]:
        return {
            "snapshot_count": self.snapshot_count,
            "gpu_sample_count": self.gpu_sample_count,
            "gpu_count": self.gpu_count,
            "active_threshold_pct": self.active_threshold_pct,
            "gpu_utilization": self.gpu_utilization.to_payload(),
            "memory_utilization": self.memory_utilization.to_payload(),
            "host_mem_available_bytes": self.host_mem_available_bytes.to_payload(),
        }


def _parse_optional_number(raw: str, *, where: str) -> float | None:
    value = raw.strip()
    if value.lower() in {"", "n/a", "[n/a]", "not supported"}:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{where} is not numeric: {raw!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def parse_nvidia_smi_csv(payload: str | bytes) -> tuple[GpuSample, ...]:
    """Parse the fixed ``--query-gpu``/``nounits`` schema used below."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    rows = csv.reader(text.splitlines(), skipinitialspace=True)
    samples: list[GpuSample] = []
    for line_number, row in enumerate(rows, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(_NVIDIA_QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi row {line_number} has {len(row)} fields; "
                f"expected {len(_NVIDIA_QUERY_FIELDS)}"
            )
        cells = [cell.strip() for cell in row]
        try:
            index = int(cells[1])
        except ValueError as exc:
            raise ValueError(f"nvidia-smi row {line_number} has invalid index") from exc
        samples.append(
            GpuSample(
                timestamp=cells[0],
                index=index,
                name=cells[2],
                uuid=cells[3],
                pci_bus_id=cells[4],
                gpu_utilization_pct=_parse_optional_number(
                    cells[5], where=f"row {line_number} utilization.gpu"
                ),
                memory_utilization_pct=_parse_optional_number(
                    cells[6], where=f"row {line_number} utilization.memory"
                ),
                memory_used_mib=_parse_optional_number(
                    cells[7], where=f"row {line_number} memory.used"
                ),
                memory_total_mib=_parse_optional_number(
                    cells[8], where=f"row {line_number} memory.total"
                ),
                temperature_c=_parse_optional_number(
                    cells[9], where=f"row {line_number} temperature.gpu"
                ),
                power_w=_parse_optional_number(
                    cells[10], where=f"row {line_number} power.draw"
                ),
            )
        )
    indexes = [sample.index for sample in samples]
    uuids = [sample.uuid for sample in samples]
    if len(indexes) != len(set(indexes)) or len(uuids) != len(set(uuids)):
        raise ValueError("nvidia-smi CSV contains duplicate GPU identities")
    return tuple(samples)


def parse_mem_available(meminfo: str | bytes) -> int:
    text = meminfo.decode("utf-8") if isinstance(meminfo, bytes) else meminfo
    values: list[int] = []
    pattern = re.compile(r"^MemAvailable:\s*([0-9]+)\s*(kB|MB|B)\s*$")
    multipliers = {"B": 1, "kB": 1024, "MB": 1024 * 1024}
    for line in text.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            values.append(int(match.group(1)) * multipliers[match.group(2)])
    if len(values) != 1:
        raise ValueError("/proc/meminfo must contain exactly one valid MemAvailable")
    return values[0]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> DistributionSummary:
    if not values:
        return DistributionSummary(0, None, None, None, None, None)
    return DistributionSummary(
        count=len(values),
        mean=math.fsum(values) / len(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        minimum=min(values),
        maximum=max(values),
    )


def _utilization(
    values: Sequence[float], *, active_threshold_pct: float
) -> UtilizationMetricSummary:
    distribution = _distribution(values)
    return UtilizationMetricSummary(
        count=distribution.count,
        mean=distribution.mean,
        p50=distribution.p50,
        p95=distribution.p95,
        active_fraction=(
            None
            if not values
            else sum(value > active_threshold_pct for value in values) / len(values)
        ),
        active_fraction_gt_0=(
            None if not values else sum(value > 0.0 for value in values) / len(values)
        ),
        active_fraction_gt_50=(
            None if not values else sum(value > 50.0 for value in values) / len(values)
        ),
        active_fraction_gt_90=(
            None if not values else sum(value > 90.0 for value in values) / len(values)
        ),
    )


def summarize_utilization(
    snapshots: Sequence[TelemetrySnapshot],
    *,
    active_threshold_pct: float = 1.0,
) -> UtilizationSummary:
    """Summarize samples with a specified linear-percentile convention."""
    threshold = float(active_threshold_pct)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise ValueError("active_threshold_pct must be finite and between 0 and 100")
    gpu_values = [
        gpu.gpu_utilization_pct
        for snapshot in snapshots
        for gpu in snapshot.gpus
        if gpu.gpu_utilization_pct is not None
    ]
    memory_values = [
        gpu.memory_utilization_pct
        for snapshot in snapshots
        for gpu in snapshot.gpus
        if gpu.memory_utilization_pct is not None
    ]
    host_values = [float(snapshot.host_mem_available_bytes) for snapshot in snapshots]
    identities = {gpu.uuid for snapshot in snapshots for gpu in snapshot.gpus}
    return UtilizationSummary(
        snapshot_count=len(snapshots),
        gpu_sample_count=sum(len(snapshot.gpus) for snapshot in snapshots),
        gpu_count=len(identities),
        active_threshold_pct=threshold,
        gpu_utilization=_utilization(gpu_values, active_threshold_pct=threshold),
        memory_utilization=_utilization(memory_values, active_threshold_pct=threshold),
        host_mem_available_bytes=_distribution(host_values),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_canonical(path: Path, *, where: str) -> object:
    try:
        payload = read_regular_file_nofollow(path, where=where)
    except ClusterTransportError:
        raise
    try:
        return _strict_json_bytes(payload, where=where)
    except ValueError as exc:
        raise ClusterTransportError(f"invalid {where}: {path}") from exc


class LocalTransport:
    """Shell-free local execution, durable receipts, and verified copies."""

    def __init__(
        self,
        state_root: str | Path,
        *,
        run_impl: RunImplementation = subprocess.run,
        popen_impl: PopenImplementation = subprocess.Popen,
        meminfo_reader: Callable[[], str | bytes] | None = None,
        nvidia_smi_binary: str = "nvidia-smi",
        transport_name: str = "local",
        pid_alive: Callable[[int], bool] = _process_is_alive,
        process_identity_reader: Callable[[int], Mapping[str, object]] = (
            _linux_process_identity
        ),
    ) -> None:
        root = Path(state_root)
        if not root.is_absolute():
            raise ValueError("state_root must be absolute")
        if "\0" in str(root):
            raise ValueError("state_root cannot contain NUL")
        self.state_root = root
        self._run_impl = run_impl
        self._popen_impl = popen_impl
        self._meminfo_reader = meminfo_reader or (
            lambda: Path("/proc/meminfo").read_bytes()
        )
        if not nvidia_smi_binary or "\0" in nvidia_smi_binary:
            raise ValueError("nvidia_smi_binary must be nonempty and NUL-free")
        self._nvidia_smi_binary = nvidia_smi_binary
        self.transport_name = transport_name
        self._pid_alive = pid_alive
        if not callable(process_identity_reader):
            raise TypeError("process_identity_reader must be callable")
        self._process_identity_reader = process_identity_reader

    @property
    def _jobs_root(self) -> Path:
        return self.state_root / "jobs"

    def _job_root(self, job_id: str) -> Path:
        return self._jobs_root / _require_job_id(job_id)

    def _claim(self, request: RunRequest) -> tuple[Path, JobReceipt]:
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        job_root = self._job_root(request.job_id)
        try:
            job_root.mkdir()
        except FileExistsError as exc:
            raise JobConflictError(
                f"job {request.job_id!r} already exists; refusing overwrite"
            ) from exc
        _atomic_write(
            job_root / "request.json",
            canonical_json_bytes(request.to_payload(), where="run request"),
        )
        running = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.identity_sha256,
            state="running",
            started_ns=time.time_ns(),
            finished_ns=None,
            returncode=None,
            pid=None,
            stdout=b"",
            stderr=b"",
            transport=self.transport_name,
        )
        self._write_receipt(job_root, running)
        return job_root, running

    @staticmethod
    def _write_receipt(job_root: Path, receipt: JobReceipt) -> None:
        _atomic_write(
            job_root / "receipt.json",
            canonical_json_bytes(receipt.to_payload(), where="job receipt"),
        )

    def _execute(self, request: RunRequest, running: JobReceipt) -> JobReceipt:
        child_env: dict[str, str]
        if request.inherit_env:
            child_env = dict(os.environ)
            child_env.update(request.env)
        else:
            child_env = dict(request.env)
        try:
            completed = self._run_impl(
                list(request.argv),
                cwd=request.cwd,
                env=child_env,
                input=request.stdin,
                capture_output=True,
                check=False,
                timeout=request.timeout_seconds,
                shell=False,
            )
            returncode = int(completed.returncode)
            return JobReceipt(
                job_id=request.job_id,
                request_sha256=request.identity_sha256,
                state="succeeded" if returncode == 0 else "failed",
                started_ns=running.started_ns,
                finished_ns=time.time_ns(),
                returncode=returncode,
                pid=running.pid,
                stdout=_bytes(completed.stdout),
                stderr=_bytes(completed.stderr),
                transport=self.transport_name,
            )
        except subprocess.TimeoutExpired as exc:
            return JobReceipt(
                job_id=request.job_id,
                request_sha256=request.identity_sha256,
                state="timed_out",
                started_ns=running.started_ns,
                finished_ns=time.time_ns(),
                returncode=None,
                pid=running.pid,
                stdout=_bytes(exc.stdout),
                stderr=_bytes(exc.stderr),
                transport=self.transport_name,
            )
        except Exception as exc:
            return JobReceipt(
                job_id=request.job_id,
                request_sha256=request.identity_sha256,
                state="transport_error",
                started_ns=running.started_ns,
                finished_ns=time.time_ns(),
                returncode=None,
                pid=running.pid,
                stdout=b"",
                stderr=f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace"),
                transport=self.transport_name,
            )

    def run(self, request: RunRequest) -> JobReceipt:
        job_root, running = self._claim(request)
        receipt = self._execute(request, running)
        self._write_receipt(job_root, receipt)
        return receipt

    def start(self, request: RunRequest) -> JobReceipt:
        """Start a detached worker which will atomically replace the receipt."""
        job_root, running = self._claim(request)
        held_context = globals().get("_PRISMAQUANT_HELD_HELPER_CONTEXT")
        if held_context is None:
            worker_argv = [
                sys.executable,
                "-P",
                "-B",
                "-s",
                str(Path(__file__).resolve()),
                "--local-worker",
                str(self.state_root),
                request.job_id,
                self.transport_name,
            ]
        else:
            if (
                not isinstance(held_context, tuple)
                or len(held_context) != 4
                or held_context[0]
                != "prismaquant.cluster_transport.held_helper.v1"
                or held_context[1] != str(Path(__file__))
                or not isinstance(held_context[2], str)
                or _SHA256.fullmatch(held_context[2]) is None
                or not isinstance(held_context[3], str)
                or not held_context[3]
            ):
                raise ClusterTransportError(
                    "held remote-helper launch context is invalid"
                )
            launcher = held_context[3]
            launcher_bytes = launcher.encode("utf-8")
            worker_argv = [
                sys.executable,
                "-P",
                "-B",
                "-s",
                "-c",
                launcher,
                held_context[1],
                held_context[2],
                base64.b64encode(launcher_bytes).decode("ascii"),
                hashlib.sha256(launcher_bytes).hexdigest(),
                "--local-worker",
                str(self.state_root),
                request.job_id,
                self.transport_name,
            ]
        try:
            worker = self._popen_impl(
                worker_argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
            launched = replace(running, pid=int(worker.pid))
            process_identity = dict(
                self._process_identity_reader(launched.pid)
            )
            if (
                process_identity.get("pid") != launched.pid
                or process_identity.get("schema")
                != "prismaquant.cluster_transport.process_identity.v1"
            ):
                raise ValueError("detached worker process identity differs")
            _atomic_write(
                job_root / "worker-identity.json",
                canonical_json_bytes(
                    process_identity, where="worker process identity",
                ),
            )
            self._write_receipt(job_root, launched)
            _atomic_write(job_root / "launched", b"ready\n")
        except Exception as exc:
            receipt = replace(
                running,
                state="transport_error",
                finished_ns=time.time_ns(),
                stderr=f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace"),
            )
            self._write_receipt(job_root, receipt)
            return receipt
        return launched

    def _inspect_job(
        self, job_id: str,
    ) -> tuple[Path, RunRequest, JobReceipt]:
        validated_job_id = _require_job_id(job_id)
        job_root = self._job_root(validated_job_id)
        try:
            root_stat = job_root.lstat()
        except FileNotFoundError as exc:
            raise JobNotFoundError(
                f"job {validated_job_id!r} does not exist"
            ) from exc
        except OSError as exc:
            raise ClusterTransportError(
                f"cannot inspect job {validated_job_id!r} durable state"
            ) from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ClusterTransportError(
                f"job {validated_job_id!r} root is not a real directory"
            )
        list_regular_file_children_nofollow(
            job_root,
            where=f"job {validated_job_id!r} root",
            suffix="",
        )
        try:
            request = RunRequest.from_payload(
                _read_canonical(
                    job_root / "request.json",
                    where=f"job {validated_job_id!r} request",
                )
            )
            receipt = JobReceipt.from_payload(
                _read_canonical(
                    job_root / "receipt.json",
                    where=f"job {validated_job_id!r} receipt",
                )
            )
        except (TypeError, ValueError) as exc:
            raise ClusterTransportError(
                f"job {validated_job_id!r} has invalid durable state"
            ) from exc
        if (
            receipt.job_id != request.job_id
            or receipt.job_id != validated_job_id
            or receipt.request_sha256 != request.request_sha256
        ):
            raise ClusterTransportError(
                f"job {validated_job_id!r} request/receipt identity mismatch"
            )
        return job_root, request, receipt

    def inspect(self, job_id: str) -> JobReceipt:
        """Return only already-durable state and never reconcile it.

        Verification uses this path so observing a stale ``running`` receipt
        cannot rewrite it into ``transport_error``.  Run/resume uses
        :meth:`status`, whose explicit job-management contract still performs
        dead-worker reconciliation.
        """

        _job_root, _request, receipt = self._inspect_job(job_id)
        return receipt

    def status(self, job_id: str) -> JobReceipt:
        job_root, request, receipt = self._inspect_job(job_id)
        validated_job_id = request.job_id
        if receipt.state != "running":
            return receipt
        if receipt.pid is not None and self._pid_alive(receipt.pid):
            try:
                expected_process = _read_canonical(
                    job_root / "worker-identity.json",
                    where=f"job {validated_job_id!r} worker identity",
                )
                live_process = dict(self._process_identity_reader(receipt.pid))
            except (
                ClusterTransportError,
                OSError,
                ProcessLookupError,
                TypeError,
                ValueError,
            ):
                pass
            else:
                if expected_process == live_process:
                    return receipt

        # A worker writes its final receipt before it exits.  Re-read after the
        # liveness check so a just-completed worker wins the race.  A receipt
        # still marked running with no live PID is ambiguous and is converted
        # to an explicit terminal failure rather than reported forever.
        refreshed = JobReceipt.from_payload(
            _read_canonical(
                job_root / "receipt.json",
                where=f"job {validated_job_id!r} refreshed receipt",
            )
        )
        if refreshed.state != "running":
            if refreshed.request_sha256 != request.request_sha256:
                raise ClusterTransportError(
                    f"job {validated_job_id!r} refreshed receipt identity mismatch"
                )
            return refreshed
        detail = (
            "detached worker PID is missing"
            if refreshed.pid is None
            else f"detached worker PID {refreshed.pid} identity is not alive"
        )
        failed = replace(
            refreshed,
            state="transport_error",
            finished_ns=time.time_ns(),
            stderr=(detail + "; refusing ambiguous running state").encode("utf-8"),
        )
        self._write_receipt(job_root, failed)
        return failed

    def sample_telemetry(self) -> TelemetrySnapshot:
        command = [
            self._nvidia_smi_binary,
            f"--query-gpu={','.join(_NVIDIA_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = self._run_impl(
                command,
                capture_output=True,
                check=False,
                shell=False,
                timeout=15.0,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise TelemetryUnavailableError(
                "nvidia-smi telemetry timed out"
            ) from exc
        except OSError as exc:
            raise TelemetryUnavailableError(
                f"nvidia-smi telemetry query was unavailable: {exc}"
            ) from exc
        if int(completed.returncode) != 0:
            raise TelemetryUnavailableError(
                "nvidia-smi telemetry failed: "
                + _bytes(completed.stderr).decode("utf-8", errors="replace")
            )
        try:
            gpus = parse_nvidia_smi_csv(_bytes(completed.stdout))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TelemetryIntegrityError(
                f"nvidia-smi returned malformed telemetry: {exc}"
            ) from exc
        try:
            meminfo = self._meminfo_reader()
        except (TimeoutError, OSError) as exc:
            raise TelemetryUnavailableError(
                f"host memory telemetry query was unavailable: {exc}"
            ) from exc
        if not isinstance(meminfo, (str, bytes)):
            raise TelemetryIntegrityError(
                "host memory telemetry returned a non-text payload"
            )
        try:
            host_mem_available_bytes = parse_mem_available(meminfo)
        except (TypeError, ValueError) as exc:
            raise TelemetryIntegrityError(
                f"host memory telemetry was malformed: {exc}"
            ) from exc
        return TelemetrySnapshot(
            captured_ns=time.time_ns(),
            host_mem_available_bytes=host_mem_available_bytes,
            gpus=gpus,
        )

    def copy_verified(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        expected_manifest: TreeManifest | None = None,
    ) -> TransferReceipt:
        """Copy and atomically publish a verified file/tree without clobber."""
        source_path = Path(source)
        destination_path = Path(destination)
        if not destination_path.is_absolute():
            raise ValueError("destination must be absolute")
        parent = destination_path.parent
        if ".." in destination_path.parts:
            raise ManifestError("destination cannot contain traversal")
        _require_no_symlink_components(parent, where="destination parent")
        try:
            parent_mode = parent.lstat().st_mode
        except OSError as exc:
            raise ManifestError(f"destination parent is unavailable: {parent}") from exc
        if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
            raise ManifestError("destination parent must be a real directory, not a symlink")
        try:
            destination_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                errno.EEXIST,
                "destination exists; refusing overwrite",
                destination_path,
            )

        manifest = build_tree_manifest(source_path)
        if expected_manifest is not None and manifest != expected_manifest:
            raise ManifestError(
                "source manifest differs from expected manifest: "
                f"expected={expected_manifest.identity_sha256} "
                f"actual={manifest.identity_sha256}"
            )
        stage_root = Path(
            tempfile.mkdtemp(prefix=f".{destination_path.name}.stage-", dir=parent)
        )
        stage_payload = stage_root / "payload"
        try:
            if manifest.root_kind == "file":
                _copy_file_verified(source_path, stage_payload, manifest.entries[0])
            else:
                stage_payload.mkdir()
                for entry in manifest.entries:
                    relative = Path(*PurePosixPath(entry.path).parts)
                    target = stage_payload / relative
                    if entry.kind == "directory":
                        target.mkdir()
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _copy_file_verified(source_path / relative, target, entry)
            verify_tree_manifest(stage_payload, manifest)
            _rename_noreplace(stage_payload, destination_path)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        return TransferReceipt(
            destination=str(destination_path),
            manifest_sha256=manifest.identity_sha256,
            total_bytes=manifest.total_bytes,
            entry_count=len(manifest.entries),
            completed_ns=time.time_ns(),
        )


def _validate_remote_executable(raw: str, *, where: str) -> str:
    path = PurePosixPath(raw)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} or _SAFE_REMOTE_COMPONENT.fullmatch(part) is None
        for part in path.parts[1:]
    ):
        raise ValueError(f"{where} must be an absolute path with safe components")
    return path.as_posix()


_HELPER_ERROR_KINDS = frozenset({
    "job_conflict",
    "job_not_found",
    "operation_error",
    "telemetry_integrity",
    "telemetry_unavailable",
})
_HELPER_TYPED_ERROR_NAMES = {
    "job_conflict": JobConflictError.__name__,
    "job_not_found": JobNotFoundError.__name__,
    "telemetry_unavailable": TelemetryUnavailableError.__name__,
}


def _helper_error_payload(
    exc: Exception,
    *,
    action: str | None,
) -> dict[str, object]:
    if isinstance(exc, JobNotFoundError):
        error_kind = "job_not_found"
    elif isinstance(exc, JobConflictError):
        error_kind = "job_conflict"
    elif isinstance(exc, TelemetryUnavailableError):
        error_kind = "telemetry_unavailable"
    elif action == "telemetry":
        error_kind = "telemetry_integrity"
    else:
        error_kind = "operation_error"
    return {
        "schema": HELPER_ERROR_SCHEMA,
        "error_kind": error_kind,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }


def _parse_helper_error_payload(payload: object) -> tuple[str, str, str]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "error_kind",
        "exception_type",
        "message",
    }:
        raise ClusterTransportError("SSH helper error payload has invalid fields")
    if payload["schema"] != HELPER_ERROR_SCHEMA:
        raise ClusterTransportError("SSH helper error payload has the wrong schema")
    error_kind = payload["error_kind"]
    exception_type = payload["exception_type"]
    message = payload["message"]
    if not isinstance(error_kind, str) or error_kind not in _HELPER_ERROR_KINDS:
        raise ClusterTransportError("SSH helper error payload has an invalid error kind")
    if (
        not isinstance(exception_type, str)
        or len(exception_type) > 128
        or _ENV_NAME.fullmatch(exception_type) is None
    ):
        raise ClusterTransportError(
            "SSH helper error payload has an invalid exception type"
        )
    if not isinstance(message, str):
        raise ClusterTransportError("SSH helper error payload has a non-text message")
    expected_exception_type = _HELPER_TYPED_ERROR_NAMES.get(error_kind)
    names_typed_error = exception_type in _HELPER_TYPED_ERROR_NAMES.values()
    if (
        expected_exception_type is not None
        and exception_type != expected_exception_type
    ) or (expected_exception_type is None and names_typed_error):
        raise ClusterTransportError(
            "SSH helper error payload has contradictory typed classification"
        )
    return error_kind, exception_type, message


class SSHTransport:
    """OpenSSH backend with a fixed helper and stdin-only request payloads."""

    def __init__(
        self,
        host: str,
        *,
        remote_helper_path: str,
        helper_source: bytes | None = None,
        remote_python_path: str = "/usr/bin/python3",
        ssh_binary: str = "ssh",
        run_impl: RunImplementation = subprocess.run,
        connect_timeout_seconds: int = 5,
    ) -> None:
        if _SSH_HOST.fullmatch(host) is None or host.startswith("-"):
            raise ValueError("host must be a safe SSH host or configured alias")
        if not ssh_binary or "\0" in ssh_binary:
            raise ValueError("ssh_binary must be nonempty and NUL-free")
        if not isinstance(connect_timeout_seconds, int) or connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be a positive integer")
        self.host = host
        self.remote_helper_path = _validate_remote_executable(
            remote_helper_path, where="remote_helper_path"
        )
        if helper_source is not None and (
            not isinstance(helper_source, bytes) or not helper_source
        ):
            raise ValueError("helper_source must be nonempty bytes when supplied")
        # Bind one immutable helper identity for bootstrap and every later
        # action.  Re-reading ``__file__`` per call would allow a local source
        # replacement to make bootstrap and invocation silently disagree.
        self._helper_source = (
            canonical_helper_source() if helper_source is None else helper_source
        )
        self._helper_source_sha256 = hashlib.sha256(
            self._helper_source
        ).hexdigest()
        self.remote_python_path = _validate_remote_executable(
            remote_python_path, where="remote_python_path"
        )
        self.ssh_binary = ssh_binary
        self._run_impl = run_impl
        self.connect_timeout_seconds = connect_timeout_seconds

    @property
    def command_argv(self) -> tuple[str, ...]:
        # The final argument contains only constructor-validated paths, the
        # immutable helper digest, and the fixed launcher above.  OpenSSH runs
        # it through the login shell; request/manifest fields remain stdin-only.
        launcher_source = _SSH_HELPER_LAUNCHER_PROGRAM.encode("utf-8")
        launcher_source_b64 = base64.b64encode(launcher_source).decode("ascii")
        launcher_source_sha256 = hashlib.sha256(launcher_source).hexdigest()
        remote_command = (
            f"exec {self.remote_python_path} -P -B -s -c "
            f"{shlex.quote(_SSH_HELPER_LAUNCHER_PROGRAM)} "
            f"{shlex.quote(self.remote_helper_path)} "
            f"{self._helper_source_sha256} {launcher_source_b64} "
            f"{launcher_source_sha256} --remote-helper"
        )
        return (
            self.ssh_binary,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_seconds}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            "-o",
            "ClearAllForwardings=yes",
            "--",
            self.host,
            remote_command,
        )

    def helper_install_spec(self) -> HelperInstallSpec:
        return HelperInstallSpec(
            remote_path=self.remote_helper_path,
            source_sha256=self._helper_source_sha256,
            size_bytes=len(self._helper_source),
            source=self._helper_source,
        )

    def _invoke(self, envelope: Mapping[str, object], *, timeout: float) -> tuple[str, object]:
        canonical = canonical_json_bytes(envelope, where="helper envelope")
        wire = base64.b64encode(canonical) + b"\n"
        try:
            completed = self._run_impl(
                list(self.command_argv),
                input=wire,
                capture_output=True,
                check=False,
                shell=False,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise _SSHInvocationUnavailableError(
                "SSH helper invocation timed out"
            ) from exc
        except OSError as exc:
            raise _SSHInvocationUnavailableError(
                f"SSH helper invocation was unavailable: {exc}"
            ) from exc
        stdout, stderr = _bytes(completed.stdout).strip(), _bytes(completed.stderr)
        if int(completed.returncode) != 0:
            raise _SSHInvocationUnavailableError(
                f"SSH helper exited {completed.returncode}: "
                + stderr.decode("utf-8", errors="replace")
            )
        try:
            response = _strict_json_bytes(stdout, where="SSH helper response")
        except ValueError as exc:
            raise ClusterTransportError("SSH helper returned invalid canonical JSON") from exc
        if not isinstance(response, Mapping) or response.get("schema") != HELPER_RESPONSE_SCHEMA:
            raise ClusterTransportError("SSH helper returned the wrong response schema")
        if set(response) != {"schema", "kind", "payload"}:
            raise ClusterTransportError("SSH helper response has invalid fields")
        kind = response["kind"]
        if not isinstance(kind, str):
            raise ClusterTransportError("SSH helper response kind must be text")
        if kind == "error":
            error_kind, exception_type, message = _parse_helper_error_payload(
                response["payload"]
            )
            detail = f"{exception_type}: {message}"
            if error_kind == "job_not_found":
                raise JobNotFoundError(detail)
            if error_kind == "job_conflict":
                raise JobConflictError(detail)
            if error_kind == "telemetry_unavailable":
                raise TelemetryUnavailableError(detail)
            if error_kind == "telemetry_integrity":
                raise TelemetryIntegrityError(detail)
            raise ClusterTransportError(detail)
        return kind, response["payload"]

    @staticmethod
    def _envelope(action: str, payload: object) -> dict[str, object]:
        return {
            "schema": HELPER_ENVELOPE_SCHEMA,
            "action": action,
            "payload": payload,
        }

    def _job_action(self, action: str, payload: object, *, timeout: float) -> JobReceipt:
        kind, response_payload = self._invoke(
            self._envelope(action, payload), timeout=timeout
        )
        if kind != "job_receipt":
            raise ClusterTransportError(f"SSH helper returned {kind!r}, expected job_receipt")
        receipt = JobReceipt.from_payload(response_payload)
        return replace(receipt, transport=f"ssh:{self.host}")

    def run(self, request: RunRequest) -> JobReceipt:
        timeout = 60.0 + (request.timeout_seconds or 0.0)
        return self._job_action("run", request.to_payload(), timeout=timeout)

    def start(self, request: RunRequest) -> JobReceipt:
        return self._job_action("start", request.to_payload(), timeout=30.0)

    def status(self, job_id: str) -> JobReceipt:
        return self._job_action(
            "status", {"job_id": _require_job_id(job_id)}, timeout=30.0
        )

    def inspect(self, job_id: str) -> JobReceipt:
        return self._job_action(
            "inspect", {"job_id": _require_job_id(job_id)}, timeout=30.0
        )

    def sample_telemetry(self) -> TelemetrySnapshot:
        try:
            kind, payload = self._invoke(
                self._envelope("telemetry", {}), timeout=30.0,
            )
        except _SSHInvocationUnavailableError as exc:
            raise TelemetryUnavailableError(
                f"SSH telemetry query was unavailable: {exc}"
            ) from exc
        except (TelemetryUnavailableError, TelemetryIntegrityError):
            raise
        except ClusterTransportError as exc:
            raise TelemetryIntegrityError(
                f"SSH telemetry helper response violated the protocol: {exc}"
            ) from exc
        if kind != "telemetry":
            raise TelemetryIntegrityError(
                f"SSH helper returned {kind!r}, expected telemetry"
            )
        try:
            return TelemetrySnapshot.from_payload(payload)
        except (TypeError, ValueError) as exc:
            raise TelemetryIntegrityError(
                f"SSH helper returned malformed telemetry: {exc}"
            ) from exc


def _default_helper_state_root() -> Path:
    return Path.home() / ".cache" / "prismaquant" / "cluster_transport"


def _helper_response(kind: str, payload: object) -> bytes:
    return canonical_json_bytes(
        {"schema": HELPER_RESPONSE_SCHEMA, "kind": kind, "payload": payload},
        where="helper response",
    )


def _remote_helper(state_root: str | None = None) -> int:
    action: str | None = None
    try:
        encoded = sys.stdin.buffer.read().strip()
        canonical = base64.b64decode(encoded, validate=True)
        envelope = _strict_json_bytes(canonical, where="helper envelope")
        if not isinstance(envelope, Mapping) or envelope.get("schema") != HELPER_ENVELOPE_SCHEMA:
            raise ValueError("helper envelope has wrong schema")
        if set(envelope) != {"schema", "action", "payload"}:
            raise ValueError("helper envelope has invalid fields")
        action, payload = str(envelope["action"]), envelope["payload"]
        helper_state_root = (
            _default_helper_state_root()
            if state_root is None
            else Path(_validate_remote_executable(
                state_root, where="remote helper state root",
            ))
        )
        transport = LocalTransport(helper_state_root, transport_name="remote")
        if action in {"run", "start"}:
            request = RunRequest.from_payload(payload)
            receipt = transport.run(request) if action == "run" else transport.start(request)
            response = _helper_response("job_receipt", receipt.to_payload())
        elif action in {"status", "inspect"}:
            if not isinstance(payload, Mapping) or set(payload) != {"job_id"}:
                raise ValueError(f"{action} payload must contain only job_id")
            reader = transport.status if action == "status" else transport.inspect
            receipt = reader(_require_job_id(payload["job_id"]))
            response = _helper_response("job_receipt", receipt.to_payload())
        elif action == "telemetry":
            if payload != {}:
                raise ValueError("telemetry payload must be empty")
            response = _helper_response("telemetry", transport.sample_telemetry().to_payload())
        else:
            raise ValueError(f"unknown helper action {action!r}")
    except Exception as exc:
        response = _helper_response(
            "error",
            _helper_error_payload(exc, action=action),
        )
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()
    return 0


def _local_worker(state_root: str, job_id: str, transport_name: str) -> int:
    transport = LocalTransport(Path(state_root), transport_name=transport_name)
    job_root = transport._job_root(_require_job_id(job_id))
    launch_deadline = time.monotonic() + 30.0
    while not (job_root / "launched").is_file():
        if time.monotonic() >= launch_deadline:
            running = JobReceipt.from_payload(
                _read_canonical(job_root / "receipt.json", where="worker receipt")
            )
            failed = replace(
                running,
                state="transport_error",
                finished_ns=time.time_ns(),
                stderr=b"controller did not publish detached-worker launch marker",
            )
            transport._write_receipt(job_root, failed)
            return 1
        time.sleep(0.01)
    request = RunRequest.from_payload(
        _read_canonical(job_root / "request.json", where="worker request")
    )
    running = transport.status(job_id)
    if running.state != "running" or running.request_sha256 != request.identity_sha256:
        raise ClusterTransportError("worker request/receipt identity mismatch")
    receipt = transport._execute(request, running)
    transport._write_receipt(job_root, receipt)
    return 0 if receipt.succeeded else 1


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--remote-helper", action="store_true")
    parser.add_argument("--remote-helper-state-root")
    parser.add_argument("--local-worker", nargs=3, metavar=("ROOT", "JOB_ID", "TRANSPORT"))
    args = parser.parse_args(argv)
    if args.remote_helper:
        if args.local_worker is not None:
            raise SystemExit("helper modes are mutually exclusive")
        return _remote_helper(args.remote_helper_state_root)
    if args.local_worker is not None:
        if args.remote_helper_state_root is not None:
            raise SystemExit("remote helper state root requires remote-helper mode")
        return _local_worker(*args.local_worker)
    raise SystemExit("one helper mode is required")


if __name__ == "__main__":
    raise SystemExit(_main())
