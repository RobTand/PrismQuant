"""Live monitoring, helper provisioning, and verified SSH transfers.

This module composes :mod:`prismaquant.cluster_transport`; it does not add a
second execution or manifest contract.  All local subprocess calls use argv
vectors with ``shell=False``.  The two SSH control programs are fixed source
strings.  Their variable requests (including helper source bytes and paths)
travel only as base64-encoded canonical JSON on stdin.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
from typing import Any, Literal, Protocol, TypeAlias

from prismaquant.cluster_transport import (
    JobReceipt,
    RunRequest,
    SSHTransport,
    TelemetrySnapshot,
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
    verify_tree_manifest,
    _rename_noreplace,
    _require_no_symlink_components,
)


BOOTSTRAP_REQUEST_SCHEMA = "prismaquant.cluster_live_transport.bootstrap_request.v1"
BOOTSTRAP_RECEIPT_SCHEMA = "prismaquant.cluster_live_transport.bootstrap_receipt.v1"
REMOTE_OP_REQUEST_SCHEMA = "prismaquant.cluster_live_transport.remote_op_request.v1"
REMOTE_OP_RESPONSE_SCHEMA = "prismaquant.cluster_live_transport.remote_op_response.v1"
RSYNC_TRANSFER_RECEIPT_SCHEMA = "prismaquant.cluster_live_transport.rsync_transfer_receipt.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_REMOTE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+")


class ClusterLiveTransportError(RuntimeError):
    """A live monitor, bootstrap, or verified transfer failed closed."""


class MonitoredTransport(Protocol):
    def start(self, request: RunRequest) -> JobReceipt:
        ...

    def status(self, job_id: str) -> JobReceipt:
        ...

    def sample_telemetry(self) -> TelemetrySnapshot:
        ...


RunImplementation: TypeAlias = Callable[..., Any]


def _duplicate_refusing_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _strict_canonical_json(payload: bytes, *, where: str) -> object:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_refusing_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClusterLiveTransportError(f"{where} is not strict JSON") from exc
    if canonical_json_bytes(value, where=where) != payload:
        raise ClusterLiveTransportError(f"{where} is not canonical JSON")
    return value


def _wire_payload(value: object) -> bytes:
    return base64.b64encode(canonical_json_bytes(value)) + b"\n"


def _decode_response(
    completed: Any,
    *,
    schema: str,
    where: str,
) -> Mapping[str, object]:
    stdout = completed.stdout
    stderr = completed.stderr
    stdout_bytes = stdout if isinstance(stdout, bytes) else str(stdout or "").encode()
    stderr_bytes = stderr if isinstance(stderr, bytes) else str(stderr or "").encode()
    if int(completed.returncode) != 0:
        raise ClusterLiveTransportError(
            f"{where} exited {completed.returncode}: "
            + stderr_bytes.decode("utf-8", errors="replace")
        )
    value = _strict_canonical_json(stdout_bytes.strip(), where=where)
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise ClusterLiveTransportError(f"{where} returned the wrong schema")
    return value


def _validated_receipt(value: object, request: RunRequest) -> JobReceipt:
    try:
        receipt = (
            value
            if isinstance(value, JobReceipt)
            else JobReceipt.from_payload(value)
        )
    except (TypeError, ValueError) as exc:
        raise ClusterLiveTransportError("transport returned an invalid job receipt") from exc
    if (
        receipt.job_id != request.job_id
        or receipt.request_sha256 != request.request_sha256
    ):
        raise ClusterLiveTransportError("job receipt differs from the monitored request")
    return receipt


def _validated_snapshot(value: object) -> TelemetrySnapshot:
    try:
        return (
            value
            if isinstance(value, TelemetrySnapshot)
            else TelemetrySnapshot.from_payload(value)
        )
    except (TypeError, ValueError) as exc:
        raise ClusterLiveTransportError("transport returned invalid telemetry") from exc


class TelemetryJobAdapter:
    """Combine durable start/status with cadence-controlled telemetry.

    A normal monitored run must be observed in ``running`` state and yield at
    least one valid sample.  An immediately terminal start therefore fails
    closed when ``require_samples`` is true: a post-hoc idle sample is not job
    utilization evidence.
    """

    def __init__(
        self,
        transport: MonitoredTransport,
        *,
        cadence_seconds: float = 1.0,
        require_samples: bool = True,
        monitor_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cadence = float(cadence_seconds)
        if not math.isfinite(cadence) or cadence <= 0.0:
            raise ValueError("cadence_seconds must be finite and positive")
        if monitor_timeout_seconds is not None:
            monitor_timeout = float(monitor_timeout_seconds)
            if not math.isfinite(monitor_timeout) or monitor_timeout <= 0.0:
                raise ValueError(
                    "monitor_timeout_seconds must be finite and positive"
                )
        else:
            monitor_timeout = None
        self.transport = transport
        self.cadence_seconds = cadence
        self.require_samples = bool(require_samples)
        self.monitor_timeout_seconds = monitor_timeout
        self._clock = clock
        self._sleep = sleep

    def start(self, request: RunRequest) -> JobReceipt:
        """Expose the durable primitive for coordinators that adopt attempts."""
        return _validated_receipt(self.transport.start(request), request)

    def status(self, job_id: str) -> JobReceipt:
        """Expose status without hiding the underlying durable job identity."""
        value = self.transport.status(job_id)
        try:
            result = (
                value
                if isinstance(value, JobReceipt)
                else JobReceipt.from_payload(value)
            )
        except (TypeError, ValueError) as exc:
            raise ClusterLiveTransportError(
                "transport returned an invalid status receipt"
            ) from exc
        if result.job_id != job_id:
            raise ClusterLiveTransportError("status receipt job id differs")
        return result

    def sample_telemetry(self) -> TelemetrySnapshot:
        """Expose one validated sample for coordinator-owned polling loops."""
        return _validated_snapshot(self.transport.sample_telemetry())

    def _now(self, *, previous: float | None = None) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ClusterLiveTransportError("monitor clock returned a non-finite value")
        if previous is not None and value < previous:
            raise ClusterLiveTransportError("monitor clock moved backwards")
        return value

    def run_with_telemetry(
        self,
        request: RunRequest,
    ) -> tuple[JobReceipt, tuple[TelemetrySnapshot, ...]]:
        started_at = self._now()
        receipt = self.start(request)
        if receipt.state != "running":
            if self.require_samples:
                raise ClusterLiveTransportError(
                    "job became terminal before any live telemetry sample"
                )
            return receipt, ()

        samples: list[TelemetrySnapshot] = []
        last_captured_ns = 0
        next_sample_at = started_at
        last_clock = started_at
        while receipt.state == "running":
            now = self._now(previous=last_clock)
            last_clock = now
            if self.monitor_timeout_seconds is not None and (
                now - started_at > self.monitor_timeout_seconds
            ):
                raise ClusterLiveTransportError(
                    f"job {request.job_id!r} exceeded monitor timeout"
                )
            delay = next_sample_at - now
            if delay > 0.0:
                self._sleep(delay)
                now = self._now(previous=last_clock)
                last_clock = now
            try:
                sample = self.sample_telemetry()
            except Exception as exc:
                if isinstance(exc, ClusterLiveTransportError):
                    raise
                raise ClusterLiveTransportError(
                    "telemetry sampling failed; refusing an unobserved job"
                ) from exc
            if sample.captured_ns <= last_captured_ns:
                raise ClusterLiveTransportError(
                    "telemetry captured_ns values are not strictly increasing"
                )
            samples.append(sample)
            last_captured_ns = sample.captured_ns
            next_sample_at = now + self.cadence_seconds
            receipt = _validated_receipt(
                self.status(request.job_id), request
            )

        if self.require_samples and not samples:
            raise ClusterLiveTransportError(
                "monitored job produced no valid telemetry samples"
            )
        return receipt, tuple(samples)


_BOOTSTRAP_PROGRAM = r'''
import base64, ctypes, errno, hashlib, json, os, re, stat, sys

SCHEMA = "prismaquant.cluster_live_transport.bootstrap_request.v1"
OUT = "prismaquant.cluster_live_transport.bootstrap_receipt.v1"
SAFE = re.compile(r"[A-Za-z0-9._-]+")

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON member")
        out[key] = value
    return out

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

def open_parent(raw):
    if (not isinstance(raw, str) or not raw.startswith("/") or "\x00" in raw):
        raise ValueError("unsafe remote_path")
    parts = raw.split("/")[1:]
    if (not parts or any(not part or part in (".", "..")
                         or SAFE.fullmatch(part) is None for part in parts)):
        raise ValueError("unsafe remote_path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ValueError("descriptor-safe bootstrap is unavailable")
    parent = os.open(
        "/", os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for component in parts[:-1]:
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("remote_path ancestry is not a real directory")
            child = os.open(
                component,
                os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            opened = os.fstat(child)
            if (not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)):
                os.close(child)
                raise ValueError("remote_path ancestry changed while opening")
            os.close(parent)
            parent = child
        return parent, parts[-1]
    except BaseException:
        os.close(parent)
        raise

def open_regular(parent, name):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("bootstrap path is not a regular file")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent,
    )
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            raise ValueError("bootstrap path changed while opening")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mtime_ns, opened.st_ctime_ns)
                or len(payload) != after.st_size):
            raise ValueError("bootstrap path changed while reading")
        return descriptor, after, payload
    except BaseException:
        os.close(descriptor)
        raise

def read_regular(parent, name):
    descriptor, _opened, payload = open_regular(parent, name)
    os.close(descriptor)
    return payload

def read_held(descriptor):
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise ValueError("bootstrap staging descriptor is not regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)
    if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns) != (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns)
            or len(payload) != after.st_size):
        raise ValueError("bootstrap staging descriptor changed while reading")
    return payload

def link_held(descriptor, parent, destination):
    # Linux linkat(AT_EMPTY_PATH) publishes the verified held inode rather
    # than re-opening a mutable staging pathname.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.linkat(
        ctypes.c_int(descriptor), ctypes.c_char_p(b""),
        ctypes.c_int(parent), ctypes.c_char_p(destination.encode("ascii")),
        ctypes.c_int(0x1000),
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), destination)

try:
    encoded = sys.stdin.buffer.read().strip()
    raw = base64.b64decode(encoded, validate=True)
    request = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if canonical(request) != raw:
        raise ValueError("request is not canonical JSON")
    if set(request) != {"schema", "remote_path", "source_b64",
                       "source_sha256", "size_bytes"} or request["schema"] != SCHEMA:
        raise ValueError("invalid bootstrap request")
    source = base64.b64decode(request["source_b64"].encode("ascii"), validate=True)
    digest = hashlib.sha256(source).hexdigest()
    if digest != request["source_sha256"] or len(source) != request["size_bytes"]:
        raise ValueError("bootstrap source identity mismatch")
    parent, destination = open_parent(request["remote_path"])
    try:
        already_present = False
        try:
            existing = read_regular(parent, destination)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if hashlib.sha256(existing).hexdigest() != digest:
                raise FileExistsError("remote helper exists with different bytes")
            already_present = True
        else:
            temporary = "." + destination + "." + digest + ".tmp"
            try:
                fd = os.open(
                    temporary,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent,
                )
            except FileExistsError:
                fd, _opened, staged = open_regular(parent, temporary)
                if hashlib.sha256(staged).hexdigest() != digest:
                    os.close(fd)
                    raise FileExistsError("bootstrap staging path has different bytes")
            else:
                try:
                    view = memoryview(source)
                    while view:
                        count = os.write(fd, view)
                        view = view[count:]
                    os.fsync(fd)
                    if hashlib.sha256(read_held(fd)).hexdigest() != digest:
                        raise ValueError("bootstrap staging write differs")
                except BaseException:
                    os.close(fd)
                    raise
            try:
                try:
                    link_held(fd, parent, destination)
                except FileExistsError:
                    if hashlib.sha256(read_regular(parent, destination)).hexdigest() != digest:
                        raise FileExistsError("remote helper raced with different bytes")
                    already_present = True
                else:
                    published_fd, published, published_bytes = open_regular(
                        parent, destination,
                    )
                    try:
                        staged = os.fstat(fd)
                        if ((published.st_dev, published.st_ino)
                                != (staged.st_dev, staged.st_ino)
                                or hashlib.sha256(published_bytes).hexdigest() != digest):
                            raise ValueError(
                                "published helper differs from held staging inode"
                            )
                    finally:
                        os.close(published_fd)
            finally:
                os.close(fd)
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.fsync(parent)
    finally:
        os.close(parent)
    response = {"schema": OUT, "remote_path": request["remote_path"],
                "source_sha256": digest, "size_bytes": len(source),
                "already_present": already_present}
    sys.stdout.buffer.write(canonical(response))
except Exception as exc:
    sys.stderr.write(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
'''.strip()


@dataclass(frozen=True)
class HelperBootstrapReceipt:
    remote_path: str
    source_sha256: str
    size_bytes: int
    already_present: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> HelperBootstrapReceipt:
        expected = {
            "schema",
            "remote_path",
            "source_sha256",
            "size_bytes",
            "already_present",
        }
        if set(payload) != expected or payload.get("schema") != BOOTSTRAP_RECEIPT_SCHEMA:
            raise ClusterLiveTransportError("bootstrap receipt has invalid fields")
        digest = payload["source_sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ClusterLiveTransportError("bootstrap receipt has invalid digest")
        size = payload["size_bytes"]
        present = payload["already_present"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ClusterLiveTransportError("bootstrap receipt has invalid size")
        if not isinstance(present, bool) or not isinstance(payload["remote_path"], str):
            raise ClusterLiveTransportError("bootstrap receipt has invalid types")
        return cls(str(payload["remote_path"]), digest, size, present)


def _fixed_python_command(
    ssh: SSHTransport,
    program: str,
) -> tuple[str, ...]:
    command = list(ssh.command_argv)
    command[-1] = (
        f"exec {ssh.remote_python_path} -P -B -s -c {shlex.quote(program)}"
    )
    return tuple(command)


def bootstrap_ssh_helper(
    ssh: SSHTransport,
    *,
    run_impl: RunImplementation | None = None,
) -> HelperBootstrapReceipt:
    """Install or verify the exact helper source without shell interpolation."""
    spec = ssh.helper_install_spec()
    request = {
        "schema": BOOTSTRAP_REQUEST_SCHEMA,
        "remote_path": spec.remote_path,
        "source_b64": base64.b64encode(spec.source).decode("ascii"),
        "source_sha256": spec.source_sha256,
        "size_bytes": spec.size_bytes,
    }
    runner = run_impl or ssh._run_impl
    completed = runner(
        list(_fixed_python_command(ssh, _BOOTSTRAP_PROGRAM)),
        input=_wire_payload(request),
        capture_output=True,
        check=False,
        shell=False,
        timeout=60.0,
    )
    payload = _decode_response(
        completed,
        schema=BOOTSTRAP_RECEIPT_SCHEMA,
        where="SSH helper bootstrap",
    )
    receipt = HelperBootstrapReceipt.from_payload(payload)
    if (
        receipt.remote_path != spec.remote_path
        or receipt.source_sha256 != spec.source_sha256
        or receipt.size_bytes != spec.size_bytes
    ):
        raise ClusterLiveTransportError("bootstrap receipt differs from requested source")
    return receipt


_REMOTE_FILE_OP_PROGRAM = r'''
import base64, hashlib, json, os, pathlib, re, stat, sys, types

SCHEMA = "prismaquant.cluster_live_transport.remote_op_request.v1"
OUT = "prismaquant.cluster_live_transport.remote_op_response.v1"
SAFE = re.compile(r"[A-Za-z0-9._-]+")
HEX = re.compile(r"[0-9a-f]{64}")

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON member")
        out[key] = value
    return out

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

def safe_path(raw):
    if not isinstance(raw, str):
        raise ValueError("path must be a string")
    pure = pathlib.PurePosixPath(raw)
    if not pure.is_absolute() or any(
        part in ("", ".", "..") or SAFE.fullmatch(part) is None
        for part in pure.parts[1:]
    ):
        raise ValueError("unsafe remote path")
    return pathlib.Path(raw)

def open_held_helper(path):
    path = safe_path(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not directory:
        raise ValueError("descriptor-safe remote helper traversal unavailable")
    parts = path.parts[1:]
    parent = os.open("/", os.O_RDONLY | directory | nofollow | cloexec)
    try:
        for component in parts[:-1]:
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("remote helper ancestry is not a real directory")
            child = os.open(
                component,
                os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=parent,
            )
            opened = os.fstat(child)
            if (not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)):
                os.close(child)
                raise ValueError("remote helper ancestry changed while opening")
            os.close(parent)
            parent = child
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("remote helper is not a regular file")
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | nofollow | cloexec | nonblock,
            dir_fd=parent,
        )
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            os.close(descriptor)
            raise ValueError("remote helper identity changed while opening")
        return path, descriptor, opened
    finally:
        os.close(parent)

def load_helper(path, expected):
    path, descriptor, opened = open_held_helper(path)
    try:
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if ((after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_size != len(source)
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or hashlib.sha256(source).hexdigest() != expected):
        raise ValueError("remote helper identity mismatch")
    name = "_pq_cluster_transport_helper"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module

try:
    encoded = sys.stdin.buffer.read().strip()
    raw = base64.b64decode(encoded, validate=True)
    request = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if canonical(request) != raw:
        raise ValueError("request is not canonical JSON")
    if set(request) != {"schema", "action", "helper_path", "helper_sha256", "payload"}:
        raise ValueError("invalid remote operation request fields")
    if request["schema"] != SCHEMA or request["action"] not in {
        "manifest", "prepare_upload", "publish_upload"
    }:
        raise ValueError("invalid remote operation request")
    if not isinstance(request["helper_sha256"], str) or not HEX.fullmatch(request["helper_sha256"]):
        raise ValueError("invalid helper digest")
    module = load_helper(request["helper_path"], request["helper_sha256"])
    action = request["action"]
    payload = request["payload"]
    if not isinstance(payload, dict):
        raise ValueError("remote operation payload must be an object")
    if action == "manifest":
        if set(payload) != {"source"}:
            raise ValueError("manifest payload fields differ")
        source = safe_path(payload["source"])
        manifest = module.build_tree_manifest(source)
        result = {"manifest": manifest.to_payload(),
                  "manifest_sha256": manifest.identity_sha256}
    else:
        expected_keys = {"remote_stage_root", "manifest", "manifest_sha256"}
        if action == "publish_upload":
            expected_keys.add("destination")
        if set(payload) != expected_keys:
            raise ValueError("upload payload fields differ")
        expected = module.TreeManifest.from_payload(payload["manifest"])
        if expected.identity_sha256 != payload["manifest_sha256"]:
            raise ValueError("upload manifest identity mismatch")
        root = safe_path(payload["remote_stage_root"])
        module._require_no_symlink_components(root, where="remote stage root")
        if not root.is_dir():
            raise ValueError("remote stage root is not a directory")
        stage = root / expected.identity_sha256
        staged_payload = stage / "payload"
        if action == "prepare_upload":
            reused = False
            try:
                stage_mode = stage.lstat().st_mode
            except FileNotFoundError:
                stage.mkdir(mode=0o700)
                if expected.root_kind == "directory":
                    staged_payload.mkdir(mode=0o700)
            else:
                if not stat.S_ISDIR(stage_mode):
                    raise ValueError("content stage is not a directory")
                try:
                    module.verify_tree_manifest(staged_payload, expected)
                except Exception as exc:
                    raise FileExistsError("content stage exists but is not verified") from exc
                reused = True
            result = {"stage_path": str(staged_payload), "reused": reused}
        else:
            module.verify_tree_manifest(staged_payload, expected)
            destination = safe_path(payload["destination"])
            module._require_no_symlink_components(
                destination.parent, where="upload destination parent"
            )
            already_present = False
            try:
                destination.lstat()
            except FileNotFoundError:
                module._rename_noreplace(staged_payload, destination)
                stage.rmdir()
            else:
                module.verify_tree_manifest(destination, expected)
                already_present = True
            verified = module.build_tree_manifest(destination)
            result = {"destination": str(destination),
                      "already_present": already_present,
                      "manifest": verified.to_payload(),
                      "manifest_sha256": verified.identity_sha256}
    response = {"schema": OUT, "kind": action, "payload": result}
    sys.stdout.buffer.write(canonical(response))
except Exception as exc:
    sys.stderr.write(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
'''.strip()


def _safe_remote_path(raw: str, *, where: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{where} must be a string")
    path = PurePosixPath(raw)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} or _SAFE_REMOTE_COMPONENT.fullmatch(part) is None
        for part in path.parts[1:]
    ):
        raise ValueError(f"{where} must be an absolute path with safe components")
    return path.as_posix()


@dataclass(frozen=True)
class RsyncTransferReceipt:
    direction: Literal["download", "upload"]
    source: str
    destination: str
    manifest_sha256: str
    total_bytes: int
    entry_count: int
    content_stage: str
    already_present: bool
    completed_ns: int

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": RSYNC_TRANSFER_RECEIPT_SCHEMA,
            "direction": self.direction,
            "source": self.source,
            "destination": self.destination,
            "manifest_sha256": self.manifest_sha256,
            "total_bytes": self.total_bytes,
            "entry_count": self.entry_count,
            "content_stage": self.content_stage,
            "already_present": self.already_present,
            "completed_ns": self.completed_ns,
        }


class VerifiedRsyncSSHTransfer:
    """Content-addressed, manifest-verified local/SSH transfer backend."""

    def __init__(
        self,
        ssh: SSHTransport,
        *,
        remote_stage_root: str,
        local_stage_root: str | Path,
        ssh_run_impl: RunImplementation | None = None,
        rsync_run_impl: RunImplementation = subprocess.run,
        rsync_binary: str = "rsync",
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.ssh = ssh
        self.remote_stage_root = _safe_remote_path(
            remote_stage_root, where="remote_stage_root"
        )
        local_root = Path(local_stage_root)
        if not local_root.is_absolute() or ".." in local_root.parts:
            raise ValueError("local_stage_root must be an absolute traversal-free path")
        self.local_stage_root = local_root
        self._ssh_run_impl = ssh_run_impl or ssh._run_impl
        self._rsync_run_impl = rsync_run_impl
        if not rsync_binary or "\0" in rsync_binary:
            raise ValueError("rsync_binary must be nonempty and NUL-free")
        self.rsync_binary = rsync_binary
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = timeout

    def _remote_action(
        self,
        action: Literal["manifest", "prepare_upload", "publish_upload"],
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        helper = self.ssh.helper_install_spec()
        request = {
            "schema": REMOTE_OP_REQUEST_SCHEMA,
            "action": action,
            "helper_path": helper.remote_path,
            "helper_sha256": helper.source_sha256,
            "payload": dict(payload),
        }
        completed = self._ssh_run_impl(
            list(_fixed_python_command(self.ssh, _REMOTE_FILE_OP_PROGRAM)),
            input=_wire_payload(request),
            capture_output=True,
            check=False,
            shell=False,
            timeout=60.0,
        )
        response = _decode_response(
            completed,
            schema=REMOTE_OP_RESPONSE_SCHEMA,
            where=f"SSH remote {action}",
        )
        if set(response) != {"schema", "kind", "payload"} or response["kind"] != action:
            raise ClusterLiveTransportError("remote operation response differs from request")
        result = response["payload"]
        if not isinstance(result, Mapping):
            raise ClusterLiveTransportError("remote operation payload is not an object")
        return result

    def _rsync_rsh(self) -> str:
        command = list(self.ssh.command_argv)
        separator = command.index("--")
        fixed = command[:separator]
        if any(any(character.isspace() for character in part) for part in fixed):
            raise ClusterLiveTransportError("SSH executable/options cannot form fixed rsync rsh")
        return shlex.join(fixed)

    def _run_rsync(self, source: str, destination: str) -> None:
        argv = [
            self.rsync_binary,
            "--recursive",
            "--no-links",
            "--checksum",
            "--protect-args",
            "--no-owner",
            "--no-group",
            "--no-perms",
            "--no-times",
            "--omit-dir-times",
            "--rsh",
            self._rsync_rsh(),
            "--",
            source,
            destination,
        ]
        completed = self._rsync_run_impl(
            argv,
            capture_output=True,
            check=False,
            shell=False,
            timeout=self.timeout_seconds,
        )
        if int(completed.returncode) != 0:
            stderr = completed.stderr
            detail = stderr if isinstance(stderr, bytes) else str(stderr or "").encode()
            raise ClusterLiveTransportError(
                f"rsync exited {completed.returncode}: "
                + detail.decode("utf-8", errors="replace")
            )

    @staticmethod
    def _remote_manifest_result(result: Mapping[str, object]) -> TreeManifest:
        if set(result) != {"manifest", "manifest_sha256"}:
            raise ClusterLiveTransportError("remote manifest response has invalid fields")
        try:
            manifest = TreeManifest.from_payload(result["manifest"])
        except (TypeError, ValueError) as exc:
            raise ClusterLiveTransportError("remote manifest is invalid") from exc
        if result["manifest_sha256"] != manifest.identity_sha256:
            raise ClusterLiveTransportError("remote manifest digest differs")
        return manifest

    def upload(
        self,
        source: str | Path,
        remote_destination: str,
    ) -> RsyncTransferReceipt:
        source_path = Path(source)
        destination = _safe_remote_path(
            remote_destination, where="remote_destination"
        )
        manifest = build_tree_manifest(source_path)
        common = {
            "remote_stage_root": self.remote_stage_root,
            "manifest": manifest.to_payload(),
            "manifest_sha256": manifest.identity_sha256,
        }
        prepared = self._remote_action("prepare_upload", common)
        expected_stage = (
            f"{self.remote_stage_root}/{manifest.identity_sha256}/payload"
        )
        if set(prepared) != {"stage_path", "reused"} or (
            prepared["stage_path"] != expected_stage
            or not isinstance(prepared["reused"], bool)
        ):
            raise ClusterLiveTransportError("remote stage response differs")
        if not prepared["reused"]:
            local_source = str(source_path)
            remote_stage = f"{self.ssh.host}:{expected_stage}"
            if manifest.root_kind == "directory":
                local_source = local_source.rstrip("/") + "/"
                remote_stage += "/"
            self._run_rsync(local_source, remote_stage)
        published = self._remote_action(
            "publish_upload", {**common, "destination": destination}
        )
        if set(published) != {
            "destination",
            "already_present",
            "manifest",
            "manifest_sha256",
        }:
            raise ClusterLiveTransportError("remote publish response has invalid fields")
        verified = self._remote_manifest_result(
            {
                "manifest": published["manifest"],
                "manifest_sha256": published["manifest_sha256"],
            }
        )
        if (
            published["destination"] != destination
            or not isinstance(published["already_present"], bool)
            or verified != manifest
        ):
            raise ClusterLiveTransportError("remote destination verification differs")
        return RsyncTransferReceipt(
            direction="upload",
            source=str(source_path),
            destination=destination,
            manifest_sha256=manifest.identity_sha256,
            total_bytes=manifest.total_bytes,
            entry_count=len(manifest.entries),
            content_stage=expected_stage,
            already_present=published["already_present"],
            completed_ns=time.time_ns(),
        )

    def inspect_artifact(self, absolute_host_path: str) -> TreeManifest:
        """Return the destination host's canonical manifest for one artifact."""
        source = _safe_remote_path(
            absolute_host_path, where="absolute_host_path"
        )
        return self._remote_manifest_result(
            self._remote_action("manifest", {"source": source})
        )

    def download(
        self,
        remote_source: str,
        destination: str | Path,
    ) -> RsyncTransferReceipt:
        source = _safe_remote_path(remote_source, where="remote_source")
        destination_path = Path(destination)
        if not destination_path.is_absolute() or ".." in destination_path.parts:
            raise ValueError("destination must be an absolute traversal-free path")
        _require_no_symlink_components(
            destination_path.parent, where="download destination parent"
        )
        if not destination_path.parent.is_dir():
            raise ValueError("download destination parent must be a directory")
        manifest = self.inspect_artifact(source)
        try:
            destination_path.lstat()
        except FileNotFoundError:
            pass
        else:
            try:
                verify_tree_manifest(destination_path, manifest)
            except Exception as exc:
                raise ClusterLiveTransportError(
                    "download destination exists but differs from remote manifest"
                ) from exc
            return RsyncTransferReceipt(
                direction="download",
                source=source,
                destination=str(destination_path),
                manifest_sha256=manifest.identity_sha256,
                total_bytes=manifest.total_bytes,
                entry_count=len(manifest.entries),
                content_stage=str(
                    self.local_stage_root / manifest.identity_sha256 / "payload"
                ),
                already_present=True,
                completed_ns=time.time_ns(),
            )
        self.local_stage_root.mkdir(parents=True, exist_ok=True)
        # Building an empty manifest is also the symlink/traversal check for
        # every existing component of the local staging root.
        build_tree_manifest(self.local_stage_root)
        stage = self.local_stage_root / manifest.identity_sha256
        staged_payload = stage / "payload"
        reused = False
        if stage.exists():
            try:
                verify_tree_manifest(staged_payload, manifest)
            except Exception as exc:
                raise ClusterLiveTransportError(
                    "local content-addressed stage exists but is not verified"
                ) from exc
            reused = True
        else:
            stage.mkdir()
            if manifest.root_kind == "directory":
                staged_payload.mkdir()
        if not reused:
            remote = f"{self.ssh.host}:{source}"
            local = str(staged_payload)
            if manifest.root_kind == "directory":
                remote += "/"
                local += "/"
            self._run_rsync(remote, local)
        verify_tree_manifest(staged_payload, manifest)
        already_present = False
        try:
            _rename_noreplace(staged_payload, destination_path)
        except FileExistsError:
            # A concurrent exact publisher is safe to adopt; any differing
            # destination remains a hard no-clobber failure.  The verified
            # content stage is retained in this rare race for explicit reuse.
            try:
                verify_tree_manifest(destination_path, manifest)
            except Exception as exc:
                raise ClusterLiveTransportError(
                    "download destination raced with different content"
                ) from exc
            already_present = True
        else:
            stage.rmdir()
            stage_parent_fd = os.open(self.local_stage_root, os.O_RDONLY)
            try:
                os.fsync(stage_parent_fd)
            finally:
                os.close(stage_parent_fd)
        destination_parent_fd = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(destination_parent_fd)
        finally:
            os.close(destination_parent_fd)
        return RsyncTransferReceipt(
            direction="download",
            source=source,
            destination=str(destination_path),
            manifest_sha256=manifest.identity_sha256,
            total_bytes=manifest.total_bytes,
            entry_count=len(manifest.entries),
            content_stage=str(staged_payload),
            already_present=already_present,
            completed_ns=time.time_ns(),
        )
