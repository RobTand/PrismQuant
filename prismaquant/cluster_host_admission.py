"""Deterministic host admission and durable GPU leases for cluster campaigns.

This module is deliberately separate from the campaign scheduler.  It turns a
sealed host record into one fixed, transportable admission request and owns a
host-wide per-GPU lease outside campaign output trees.  Lease ownership is a
durable canonical record, not a controller PID or an advisory lock, so the
same sealed campaign can explicitly adopt it after a controller restart.

``inputs.model_content_sha256`` is interpreted as the path-neutral
``portable_content_sha256`` derived by :mod:`prismaquant.cost_streaming` from
an independently built and live-validated complete streamed-model identity
cache.  Admission never substitutes a pathname, mtime, index file, or model
name for that value-bearing identity.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
from typing import Literal, Protocol

from prismaquant.cluster_campaign_contract import (
    canonical_sha256,
    parse_campaign_manifest,
    validate_campaign_manifest,
)
from prismaquant.cluster_transport import (
    RunRequest,
    canonical_json_bytes,
    parse_mem_available,
)


HOST_ADMISSION_RECEIPT_SCHEMA = "prismaquant.cluster_host_admission.receipt.v1"
HOST_PRE_ADMISSION_RECEIPT_SCHEMA = (
    "prismaquant.cluster_host_admission.pre_admission.v1"
)
GPU_START_GUARD_RECEIPT_SCHEMA = (
    "prismaquant.cluster_host_admission.gpu_start_guard.v1"
)
MODEL_IDENTITY_RECEIPT_SCHEMA = (
    "prismaquant.cluster_host_admission.model_identity.v1"
)
GPU_LEASE_SCHEMA = "prismaquant.cluster_host_admission.gpu_lease.v1"
GPU_LEASE_OPERATION_SCHEMA = (
    "prismaquant.cluster_host_admission.gpu_lease_operation.v1"
)
DEFAULT_GPU_LEASE_ROOT = Path("/var/tmp/prismaquant-gpu-leases")
SOURCE_IDENTITY_CACHE_NAME = "source-identity-cache.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HOST_ENV = (
    ("LANG", "C.UTF-8"),
    ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONNOUSERSITE", "1"),
    ("PYTHONSAFEPATH", "1"),
)
_GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,uuid,compute_cap",
    "--format=csv,noheader,nounits",
)
_COMPUTE_APPS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)
_CAMPAIGN_CONTAINER_LABEL = "io.prismaquant.campaign"
_HOST_CONTAINER_LABEL = "io.prismaquant.host"
_WORK_CONTAINER_LABEL = "io.prismaquant.work"
_CONTAINER_TIMEOUT_EXIT = 124
_DOCKER_CONTROL_TIMEOUT_SECONDS = 45.0
_LEASE_BODY_KEYS = frozenset({
    "schema",
    "campaign_id",
    "campaign_identity_sha256",
    "host_id",
    "host_identity_sha256",
    "gpu_uuid",
})
_LEASE_KEYS = _LEASE_BODY_KEYS | {"identity_sha256"}


class ClusterHostAdmissionError(RuntimeError):
    """A host, content identity, resource check, or GPU lease was refused."""


class _CompletedProcess(Protocol):
    returncode: int
    stdout: bytes | str | None
    stderr: bytes | str | None


class _DiskUsage(Protocol):
    free: int


CommandRunner = Callable[[tuple[str, ...]], _CompletedProcess]
DiskUsageReader = Callable[[str], _DiskUsage]
ModelIdentityReader = Callable[[Path, Path], Mapping[str, object]]


def _run_command(argv: tuple[str, ...]) -> _CompletedProcess:
    return subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        timeout=30.0,
    )


def _read_meminfo() -> bytes:
    return Path("/proc/meminfo").read_bytes()


@dataclass(frozen=True)
class HostAdmissionRuntime:
    """Injectable, side-effect boundary for deterministic admission tests."""

    command_runner: CommandRunner = _run_command
    disk_usage_reader: DiskUsageReader = shutil.disk_usage
    meminfo_reader: Callable[[], str | bytes] = _read_meminfo
    hostname_reader: Callable[[], str] = socket.gethostname
    uid_reader: Callable[[], int] = os.getuid
    gid_reader: Callable[[], int] = os.getgid
    model_identity_reader: ModelIdentityReader | None = None


def _host_by_id(
    manifest: Mapping[str, object], host_id: str,
) -> Mapping[str, object]:
    for host in manifest["hosts"]:  # type: ignore[index]
        if isinstance(host, Mapping) and host.get("id") == host_id:
            return host
    raise ClusterHostAdmissionError(f"sealed manifest has no host {host_id!r}")


def _sealed(body: Mapping[str, object]) -> dict[str, object]:
    return {**body, "identity_sha256": canonical_sha256(body)}


def _strict_json(payload: bytes, *, where: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ClusterHostAdmissionError(
                    f"{where} contains duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {value}")
            ),
        )
    except ClusterHostAdmissionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClusterHostAdmissionError(f"{where} is not strict JSON") from exc
    if canonical_json_bytes(decoded, where=where) != payload:
        raise ClusterHostAdmissionError(f"{where} is not canonically encoded")
    return decoded


def _stable_regular_file_sha256(path: Path, *, where: str) -> dict[str, object]:
    """Hash one held, no-follow regular-file descriptor with race checks."""

    if not path.is_absolute():
        raise ClusterHostAdmissionError(f"{where} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ClusterHostAdmissionError(f"{where} is unreadable") from exc
    if resolved != path or path.is_symlink():
        raise ClusterHostAdmissionError(f"{where} must be a real non-symlink path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClusterHostAdmissionError(f"{where} cannot be safely opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ClusterHostAdmissionError(f"{where} is not a regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_before != fingerprint_after or total != before.st_size:
        raise ClusterHostAdmissionError(f"{where} changed while hashing")
    return {
        "sha256": digest.hexdigest(),
        "bytes": total,
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
    }


def _trusted_model_identity(model_root: Path, cache_path: Path) -> dict[str, object]:
    """Validate the existing complete cache and derive its path-neutral ID."""

    from prismaquant.cost_streaming import (
        compact_streamed_model_identity,
        portable_streamed_model_content_identity,
        validate_cached_streamed_model_identity,
    )

    try:
        identity = validate_cached_streamed_model_identity(
            model_root,
            cache_path,
            require_complete_checkpoint=True,
        )
        compact = compact_streamed_model_identity(
            identity, where="cluster host-admission source model",
        )
        portable = portable_streamed_model_content_identity(
            identity, where="cluster host-admission portable source model",
        )
    except Exception as exc:
        raise ClusterHostAdmissionError(
            f"trusted streamed-model identity validation failed: {exc}"
        ) from exc
    return {
        "schema": MODEL_IDENTITY_RECEIPT_SCHEMA,
        "identity": compact,
        "portable": portable,
    }


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return value


def _fixed_command(
    runtime: HostAdmissionRuntime,
    argv: tuple[str, ...],
    *,
    where: str,
) -> str:
    try:
        completed = runtime.command_runner(argv)
        stdout = _decode_output(completed.stdout)
        stderr = _decode_output(completed.stderr)
    except Exception as exc:
        raise ClusterHostAdmissionError(f"{where} could not execute") from exc
    if int(completed.returncode) != 0:
        detail = stderr.strip()[:300]
        raise ClusterHostAdmissionError(
            f"{where} failed with return code {completed.returncode}: {detail}"
        )
    return stdout


def _live_gpu_identity(
    expected_gpu: Mapping[str, object], runtime: HostAdmissionRuntime,
) -> dict[str, object]:
    output = _fixed_command(runtime, _GPU_QUERY, where="nvidia-smi GPU identity query")
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(output.splitlines(), skipinitialspace=True)
        if row and any(cell.strip() for cell in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ClusterHostAdmissionError(
            "host admission requires exactly one visible GPU identity row"
        )
    raw_index, name, uuid, capability = rows[0]
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ClusterHostAdmissionError("nvidia-smi GPU index is malformed") from exc
    expected_capability = ".".join(
        str(value) for value in expected_gpu["compute_capability"]  # type: ignore[index]
    )
    if (
        index < 0
        or name != expected_gpu["name"]
        or uuid != expected_gpu["uuid"]
        or capability != expected_capability
        or expected_gpu["device_count"] != 1
    ):
        raise ClusterHostAdmissionError(
            "live GPU identity/count differs from the sealed host record"
        )
    return {
        "index": index,
        "name": name,
        "uuid": uuid,
        "compute_capability": capability,
        "device_count": 1,
    }


def _require_idle_gpu(runtime: HostAdmissionRuntime) -> dict[str, object]:
    output = _fixed_command(
        runtime, _COMPUTE_APPS_QUERY, where="nvidia-smi compute-apps query",
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if rows:
        raise ClusterHostAdmissionError(
            "nvidia-smi query-compute-apps is nonempty; GPU is busy"
        )
    return {
        "query": list(_COMPUTE_APPS_QUERY),
        "rows": 0,
        "stdout_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _real_directory(path: Path, *, where: str, writable: bool = False) -> Path:
    if not path.is_absolute():
        raise ClusterHostAdmissionError(f"{where} must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise ClusterHostAdmissionError(f"{where} is absent or unreadable") from exc
    if resolved != path or path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ClusterHostAdmissionError(
            f"{where} must be a real non-symlink directory"
        )
    if writable and not os.access(path, os.W_OK | os.X_OK):
        raise ClusterHostAdmissionError(f"{where} is not writable/searchable")
    return path


def _resource_receipt(
    manifest: Mapping[str, object],
    host_id: str,
    host: Mapping[str, object],
    runtime: HostAdmissionRuntime,
) -> dict[str, object]:
    policy = manifest["policy"]
    assert isinstance(policy, Mapping)
    resources = policy["resources"]
    assert isinstance(resources, Mapping)
    minimum = int(
        resources[
            "coordinator_min_free_bytes"
            if host_id == manifest["coordinator"]
            else "worker_min_free_bytes"
        ]
    )
    roots = host["roots"]
    assert isinstance(roots, Mapping)
    disks: list[dict[str, object]] = []
    for field in ("run_root", "worker_state_root"):
        path = _real_directory(
            Path(str(roots[field])), where=f"host {field}", writable=True,
        )
        try:
            free = int(runtime.disk_usage_reader(str(path)).free)
        except Exception as exc:
            raise ClusterHostAdmissionError(
                f"cannot inspect free disk for host {field}"
            ) from exc
        if free < minimum:
            raise ClusterHostAdmissionError(
                f"host {field} free bytes {free} are below required {minimum}"
            )
        disks.append({
            "root": field,
            "path": str(path),
            "free_bytes": free,
            "required_free_bytes": minimum,
        })
    try:
        mem_available = parse_mem_available(runtime.meminfo_reader())
    except Exception as exc:
        raise ClusterHostAdmissionError("cannot read exact MemAvailable") from exc
    min_memory = int(resources["min_mem_available_bytes"])
    if mem_available < min_memory:
        raise ClusterHostAdmissionError(
            f"MemAvailable {mem_available} is below required {min_memory}"
        )
    return {
        "role": "coordinator" if host_id == manifest["coordinator"] else "worker",
        "disks": disks,
        "mem_available_bytes": mem_available,
        "required_mem_available_bytes": min_memory,
    }


def _lease_body(
    manifest: Mapping[str, object], host: Mapping[str, object],
) -> dict[str, object]:
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    return {
        "schema": GPU_LEASE_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_id": host["id"],
        "host_identity_sha256": canonical_sha256(host),
        "gpu_uuid": gpu["uuid"],
    }


def _expected_lease(
    manifest: Mapping[str, object], host: Mapping[str, object],
) -> dict[str, object]:
    return _sealed(_lease_body(manifest, host))


def _lease_key(gpu_uuid: str) -> str:
    return hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()


def _lease_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise ClusterHostAdmissionError(
            "GPU lease root must be a normalized non-root absolute path"
        )
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ClusterHostAdmissionError("GPU lease root is unavailable") from exc
    if (
        resolved != root
        or root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ClusterHostAdmissionError(
            "GPU lease root must be a private real directory owned by this UID"
        )
    return root


@contextmanager
def _lease_guard(root: Path, gpu_uuid: str):
    guard = root / f"{_lease_key(gpu_uuid)}.guard"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(guard, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise ClusterHostAdmissionError("GPU lease guard is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ClusterHostAdmissionError(
            "another GPU lease operation is in progress"
        ) from exc
    except ClusterHostAdmissionError:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ClusterHostAdmissionError("cannot lock GPU lease state") from exc
    try:
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _lease_path(root: Path, gpu_uuid: str) -> Path:
    return root / f"{_lease_key(gpu_uuid)}.json"


def _lease_release_marker_path(
    root: Path,
    gpu_uuid: str,
    campaign_identity_sha256: str,
) -> Path:
    if _SHA256.fullmatch(campaign_identity_sha256) is None:
        raise ClusterHostAdmissionError(
            "release marker campaign identity is malformed"
        )
    return root / (
        f"{_lease_key(gpu_uuid)}.released-"
        f"{campaign_identity_sha256}.json"
    )


def _publish_lease_release_marker(
    lease_path: Path, release_marker: Path,
) -> None:
    """Atomically retain the already-fsynced lease before unlinking it."""

    try:
        os.link(lease_path, release_marker, follow_symlinks=False)
        _fsync_directory(release_marker.parent)
    except FileExistsError as exc:
        raise ClusterHostAdmissionError(
            "GPU release marker appeared concurrently"
        ) from exc
    except OSError as exc:
        raise ClusterHostAdmissionError(
            "cannot publish durable GPU release marker"
        ) from exc


def _refuse_released_lease_generation(
    root: Path,
    *,
    gpu_uuid: str,
    expected: Mapping[str, object],
) -> None:
    marker = _lease_release_marker_path(
        root, gpu_uuid, str(expected["campaign_identity_sha256"]),
    )
    if not marker.exists() and not marker.is_symlink():
        return
    released = _read_lease(marker, gpu_uuid=gpu_uuid)
    if released != dict(expected):
        raise ClusterHostAdmissionError(
            "GPU release marker differs from this sealed campaign/host"
        )
    raise ClusterHostAdmissionError(
        "sealed campaign GPU lease was already released; refusing reacquisition"
    )


def _validate_lease(raw: object, *, gpu_uuid: str) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != _LEASE_KEYS:
        raise ClusterHostAdmissionError("GPU lease fields differ")
    lease = dict(raw)
    if lease.get("schema") != GPU_LEASE_SCHEMA or lease.get("gpu_uuid") != gpu_uuid:
        raise ClusterHostAdmissionError("GPU lease identity differs from its key")
    for field in (
        "campaign_identity_sha256", "host_identity_sha256", "identity_sha256",
    ):
        if _SHA256.fullmatch(str(lease.get(field, ""))) is None:
            raise ClusterHostAdmissionError(f"GPU lease {field} is malformed")
    body = {key: lease[key] for key in _LEASE_BODY_KEYS}
    if lease["identity_sha256"] != canonical_sha256(body):
        raise ClusterHostAdmissionError("GPU lease digest differs")
    return lease


def _read_lease(path: Path, *, gpu_uuid: str) -> dict[str, object]:
    if path.is_symlink():
        raise ClusterHostAdmissionError("GPU lease is a symlink")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ClusterHostAdmissionError("GPU lease is unreadable") from exc
    return _validate_lease(
        _strict_json(payload, where="GPU lease"), gpu_uuid=gpu_uuid,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_lease(path: Path, lease: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(lease, where="GPU lease")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _lease_operation(
    *, action: str, disposition: str, lease: Mapping[str, object] | None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": GPU_LEASE_OPERATION_SCHEMA,
        "action": action,
        "disposition": disposition,
        "lease": dict(lease) if lease is not None else None,
    }
    return _sealed(body)


def inspect_gpu_lease(
    gpu_uuid: str, *, lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
) -> dict[str, object]:
    root = _lease_root(lease_root)
    path = _lease_path(root, gpu_uuid)
    with _lease_guard(root, gpu_uuid):
        if not path.exists():
            return _lease_operation(
                action="inspect", disposition="absent", lease=None,
            )
        lease = _read_lease(path, gpu_uuid=gpu_uuid)
        return _lease_operation(
            action="inspect", disposition="held", lease=lease,
        )


def acquire_gpu_lease(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
) -> dict[str, object]:
    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = _expected_lease(normalized, host)
    gpu_uuid = str(expected["gpu_uuid"])
    root = _lease_root(lease_root)
    path = _lease_path(root, gpu_uuid)
    with _lease_guard(root, gpu_uuid):
        if path.exists():
            existing = _read_lease(path, gpu_uuid=gpu_uuid)
            if existing != expected:
                raise ClusterHostAdmissionError(
                    "GPU is leased by a different sealed campaign/host"
                )
            return _lease_operation(
                action="acquire", disposition="adopted", lease=existing,
            )
        _refuse_released_lease_generation(
            root, gpu_uuid=gpu_uuid, expected=expected,
        )
        try:
            _publish_lease(path, expected)
        except FileExistsError as exc:
            raise ClusterHostAdmissionError(
                "GPU lease appeared concurrently; refusing ambiguous acquire"
            ) from exc
        published = _read_lease(path, gpu_uuid=gpu_uuid)
        if published != expected:
            raise ClusterHostAdmissionError("published GPU lease differs")
        return _lease_operation(
            action="acquire", disposition="acquired", lease=published,
        )


def adopt_gpu_lease(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
) -> dict[str, object]:
    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = _expected_lease(normalized, host)
    gpu_uuid = str(expected["gpu_uuid"])
    root = _lease_root(lease_root)
    path = _lease_path(root, gpu_uuid)
    with _lease_guard(root, gpu_uuid):
        if not path.exists():
            raise ClusterHostAdmissionError("GPU lease is absent; cannot adopt")
        existing = _read_lease(path, gpu_uuid=gpu_uuid)
        if existing != expected:
            raise ClusterHostAdmissionError(
                "GPU lease belongs to a different sealed campaign/host"
            )
        return _lease_operation(
            action="adopt", disposition="adopted", lease=existing,
        )


def release_gpu_lease(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
) -> dict[str, object]:
    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = _expected_lease(normalized, host)
    gpu_uuid = str(expected["gpu_uuid"])
    root = _lease_root(lease_root)
    path = _lease_path(root, gpu_uuid)
    release_marker = _lease_release_marker_path(
        root, gpu_uuid, str(normalized["identity_sha256"]),
    )
    with _lease_guard(root, gpu_uuid):
        if not path.exists():
            # Absence alone is not evidence that this campaign released the
            # lease.  A retry can distinguish the post-unlink/pre-receipt
            # crash window only through the durable exact-lease marker that
            # was published before unlinking.
            if release_marker.exists():
                released = _read_lease(
                    release_marker, gpu_uuid=gpu_uuid,
                )
                if released != expected:
                    raise ClusterHostAdmissionError(
                        "GPU release marker differs from this sealed campaign/host"
                    )
                return _lease_operation(
                    action="release",
                    disposition="already_absent",
                    lease=released,
                )
            return _lease_operation(
                action="release", disposition="already_absent", lease=None,
            )
        existing = _read_lease(path, gpu_uuid=gpu_uuid)
        if existing != expected:
            raise ClusterHostAdmissionError(
                "refusing to release another sealed campaign's GPU lease"
            )
        try:
            if release_marker.exists():
                marked = _read_lease(release_marker, gpu_uuid=gpu_uuid)
                if marked != existing:
                    raise ClusterHostAdmissionError(
                        "GPU release marker differs from the held lease"
                    )
            else:
                _publish_lease_release_marker(path, release_marker)
            path.unlink()
            _fsync_directory(root)
        except ClusterHostAdmissionError:
            raise
        except OSError as exc:
            raise ClusterHostAdmissionError("cannot release GPU lease") from exc
        return _lease_operation(
            action="release", disposition="released", lease=existing,
        )


def pre_admit_host(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
    runtime: HostAdmissionRuntime | None = None,
) -> dict[str, object]:
    """Admit immutable host/input resources before any campaign GPU work.

    Model-content validation is intentionally deferred until the fixed
    ``worker_source_identity`` stage has produced a fresh, independently
    validated cache on each host.  Everything needed to launch that stage is
    checked here: exact host/GPU identity, an idle device, disk/RAM minima,
    immutable dataset bytes, real model/snapshot roots, and the durable
    per-GPU campaign lease.
    """

    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = host["expected"]
    roots = host["roots"]
    inputs = normalized["inputs"]
    assert isinstance(expected, Mapping)
    assert isinstance(roots, Mapping)
    assert isinstance(inputs, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    if gpu.get("device_count") != 1:
        raise ClusterHostAdmissionError(
            "sealed host must declare exactly one GPU"
        )
    active_runtime = runtime or HostAdmissionRuntime()
    if active_runtime.hostname_reader() != expected["hostname"]:
        raise ClusterHostAdmissionError("live hostname differs from sealed host")
    if (
        active_runtime.uid_reader() != expected["uid"]
        or active_runtime.gid_reader() != expected["gid"]
    ):
        raise ClusterHostAdmissionError("live UID/GID differs from sealed host")

    live_gpu = _live_gpu_identity(gpu, active_runtime)
    compute_apps = _require_idle_gpu(active_runtime)
    resources = _resource_receipt(normalized, host_id, host, active_runtime)
    model_root = _real_directory(
        Path(str(roots["model_root"])), where="campaign model root",
    )
    snapshot_root = _real_directory(
        Path(str(roots["snapshot_root"])), where="campaign snapshot root",
    )
    dataset = _stable_regular_file_sha256(
        Path(str(roots["dataset_path"])), where="campaign dataset",
    )
    if dataset["sha256"] != inputs["dataset_sha256"]:
        raise ClusterHostAdmissionError(
            "campaign dataset SHA-256 differs from sealed manifest"
        )
    lease = acquire_gpu_lease(normalized, host_id, lease_root=lease_root)
    body: dict[str, object] = {
        "schema": HOST_PRE_ADMISSION_RECEIPT_SCHEMA,
        "campaign_identity_sha256": normalized["identity_sha256"],
        "host_id": host_id,
        "host_identity_sha256": canonical_sha256(host),
        "hostname": expected["hostname"],
        "uid": expected["uid"],
        "gid": expected["gid"],
        "gpu": live_gpu,
        "compute_apps": compute_apps,
        "resources": resources,
        "dataset": {
            "path": str(roots["dataset_path"]),
            "sha256": dataset["sha256"],
            "bytes": dataset["bytes"],
        },
        "model_root": str(model_root),
        "snapshot_root": str(snapshot_root),
        "lease": lease,
    }
    return _sealed(body)


def guard_gpu_start(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
    runtime: HostAdmissionRuntime | None = None,
) -> dict[str, object]:
    """Recheck lease ownership and GPU idleness immediately before start."""

    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    active_runtime = runtime or HostAdmissionRuntime()
    if active_runtime.hostname_reader() != expected["hostname"]:
        raise ClusterHostAdmissionError("live hostname differs from sealed host")
    if (
        active_runtime.uid_reader() != expected["uid"]
        or active_runtime.gid_reader() != expected["gid"]
    ):
        raise ClusterHostAdmissionError("live UID/GID differs from sealed host")
    root = _lease_root(lease_root)
    gpu_uuid = str(gpu["uuid"])
    expected_lease = _expected_lease(normalized, host)
    with _lease_guard(root, gpu_uuid):
        lease_path = _lease_path(root, gpu_uuid)
        if not lease_path.exists():
            raise ClusterHostAdmissionError(
                "GPU lease is absent; cannot guard a campaign launch"
            )
        existing_lease = _read_lease(lease_path, gpu_uuid=gpu_uuid)
        if existing_lease != expected_lease:
            raise ClusterHostAdmissionError(
                "GPU lease belongs to a different sealed campaign/host"
            )
        live_gpu = _live_gpu_identity(gpu, active_runtime)
        compute_apps = _require_idle_gpu(active_runtime)
        campaign_containers = _require_no_campaign_containers(
            normalized, active_runtime,
        )
        lease = _lease_operation(
            action="adopt", disposition="adopted", lease=existing_lease,
        )
    body: dict[str, object] = {
        "schema": GPU_START_GUARD_RECEIPT_SCHEMA,
        "campaign_identity_sha256": normalized["identity_sha256"],
        "host_id": host_id,
        "host_identity_sha256": canonical_sha256(host),
        "gpu": live_gpu,
        "compute_apps": compute_apps,
        "campaign_containers": campaign_containers,
        "lease": lease,
    }
    return _sealed(body)


def _require_no_campaign_containers(
    manifest: Mapping[str, object], runtime: HostAdmissionRuntime,
) -> list[str]:
    label = f"{_CAMPAIGN_CONTAINER_LABEL}={manifest['identity_sha256']}"
    completed = runtime.command_runner((
        "docker", "ps", "--filter", f"label={label}",
        "--format", "{{.ID}}",
    ))
    if int(completed.returncode) != 0:
        raise ClusterHostAdmissionError(
            "cannot inspect live containers for this campaign"
        )
    payload = completed.stdout or b""
    text = (
        payload.decode("utf-8")
        if isinstance(payload, bytes)
        else str(payload)
    )
    containers = [line.strip() for line in text.splitlines() if line.strip()]
    if any(re.fullmatch(r"[0-9a-f]{12,64}", item) is None for item in containers):
        raise ClusterHostAdmissionError(
            "docker returned a malformed campaign container identity"
        )
    if containers:
        raise ClusterHostAdmissionError(
            "a prior campaign container is still running; refusing overlap"
        )
    return containers


def _private_supervision_directory(
    lease_root: Path, campaign_identity: str, host_id: str,
) -> Path:
    """Return a private, non-symlink directory for Docker CID ownership."""

    result = lease_root / "container-supervision" / campaign_identity / host_id
    current = lease_root
    for component in result.relative_to(lease_root).parts:
        current /= component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ClusterHostAdmissionError(
                "container supervision directory is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ClusterHostAdmissionError(
                "container supervision directory is not private"
            )
    return result


def _docker_control(
    argv: Sequence[str], *, run_impl: Callable[..., _CompletedProcess],
) -> _CompletedProcess:
    try:
        return run_impl(
            list(argv),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=_DOCKER_CONTROL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClusterHostAdmissionError(
            "bounded Docker supervision command failed"
        ) from exc


def _inspect_container(
    reference: str,
    *,
    run_impl: Callable[..., _CompletedProcess],
) -> Mapping[str, object] | None:
    completed = _docker_control(
        ("docker", "inspect", "--type", "container", reference),
        run_impl=run_impl,
    )
    if int(completed.returncode) != 0:
        # Distinguish a genuinely absent container from a daemon/permission
        # failure with a second fixed listing operation.
        listing = _docker_control(
            (
                "docker", "ps", "-a", "--no-trunc", "--format",
                "{{.ID}}\t{{.Names}}",
            ),
            run_impl=run_impl,
        )
        if int(listing.returncode) != 0:
            raise ClusterHostAdmissionError(
                "cannot prove supervised container absence"
            )
        raw = listing.stdout or b""
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        matches = []
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if (
                len(fields) != 2
                or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", fields[1])
                is None
            ):
                raise ClusterHostAdmissionError(
                    "Docker container listing is malformed"
                )
            if reference in fields:
                matches.append(tuple(fields))
        if matches:
            raise ClusterHostAdmissionError(
                "supervised container exists but cannot be inspected"
            )
        return None
    raw = completed.stdout or b""
    payload = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ClusterHostAdmissionError(
            "Docker inspect returned invalid JSON"
        ) from exc
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        raise ClusterHostAdmissionError(
            "Docker inspect returned an ambiguous container"
        )
    return value[0]


def _require_owned_container(
    inspected: Mapping[str, object],
    *,
    campaign_identity: str,
    host_id: str,
    work_id: str,
) -> bool:
    config = inspected.get("Config")
    state = inspected.get("State")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ClusterHostAdmissionError(
            "supervised container metadata is incomplete"
        )
    labels = config.get("Labels")
    if not isinstance(labels, Mapping) or (
        labels.get(_CAMPAIGN_CONTAINER_LABEL) != campaign_identity
        or labels.get(_HOST_CONTAINER_LABEL) != host_id
        or labels.get(_WORK_CONTAINER_LABEL) != work_id
    ):
        raise ClusterHostAdmissionError(
            "refusing to control a container without exact campaign ownership"
        )
    running = state.get("Running")
    if type(running) is not bool:
        raise ClusterHostAdmissionError(
            "supervised container running state is invalid"
        )
    return running


def _read_cid(path: Path) -> str:
    try:
        metadata = path.lstat()
        payload = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ClusterHostAdmissionError(
            "cannot read supervised Docker CID"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or re.fullmatch(r"[0-9a-f]{64}", payload) is None
    ):
        raise ClusterHostAdmissionError("supervised Docker CID is unsafe")
    return payload


def _remove_cid(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise ClusterHostAdmissionError(
            "cannot retire supervised Docker CID"
        ) from exc


def _stop_owned_container(
    reference: str,
    *,
    campaign_identity: str,
    host_id: str,
    work_id: str,
    run_impl: Callable[..., _CompletedProcess],
) -> None:
    inspected = _inspect_container(reference, run_impl=run_impl)
    if inspected is None:
        return
    running = _require_owned_container(
        inspected,
        campaign_identity=campaign_identity,
        host_id=host_id,
        work_id=work_id,
    )
    if running:
        _docker_control(
            ("docker", "stop", "--time", "30", reference),
            run_impl=run_impl,
        )
    inspected = _inspect_container(reference, run_impl=run_impl)
    if inspected is not None and _require_owned_container(
        inspected,
        campaign_identity=campaign_identity,
        host_id=host_id,
        work_id=work_id,
    ):
        _docker_control(("docker", "kill", reference), run_impl=run_impl)
    inspected = _inspect_container(reference, run_impl=run_impl)
    if inspected is not None:
        _require_owned_container(
            inspected,
            campaign_identity=campaign_identity,
            host_id=host_id,
            work_id=work_id,
        )
        _docker_control(
            ("docker", "rm", "--force", reference), run_impl=run_impl,
        )
    if _inspect_container(reference, run_impl=run_impl) is not None:
        raise ClusterHostAdmissionError(
            "supervised container survived bounded stop/kill cleanup"
        )


def _terminate_docker_client(process: object) -> None:
    pid = int(getattr(process, "pid"))
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)  # type: ignore[attr-defined]
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)  # type: ignore[attr-defined]
    except subprocess.TimeoutExpired as exc:
        raise ClusterHostAdmissionError(
            "Docker client survived bounded process-group termination"
        ) from exc


def guarded_container_launch(
    manifest: Mapping[str, object],
    host_id: str,
    command: Sequence[str],
    *,
    work_id: str,
    timeout_seconds: float,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
    runtime: HostAdmissionRuntime | None = None,
    popen_impl: Callable[..., object] = subprocess.Popen,
    run_impl: Callable[..., _CompletedProcess] = subprocess.run,
) -> int:
    """Atomically recheck the cooperative lease and run one fixed container.

    This wrapper retains the lease lock while waiting and also passes the lock
    descriptor into the Docker client.  If either the detached transport
    worker or this wrapper dies, another campaign launch therefore cannot pass
    its gate while that client is alive; the campaign-label check also rejects
    a daemon-side orphan after the client exits.
    """

    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    producer = normalized["producer"]
    assert isinstance(producer, Mapping)
    argv = tuple(command)
    if (
        not isinstance(work_id, str)
        or re.fullmatch(r"[a-z0-9][A-Za-z0-9_.:-]{0,255}", work_id) is None
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0.0 < float(timeout_seconds) <= 604800.0
    ):
        raise ClusterHostAdmissionError(
            "guarded launch work identity or timeout is invalid"
        )
    campaign_label = (
        f"{_CAMPAIGN_CONTAINER_LABEL}={normalized['identity_sha256']}"
    )
    host_label = f"{_HOST_CONTAINER_LABEL}={host_id}"
    work_labels = tuple(
        value for value in argv
        if value.startswith(f"{_WORK_CONTAINER_LABEL}=")
    )
    expected_image = f"eugr/spark-vllm@{producer['image_digest']}"
    image_indexes = tuple(
        index for index, value in enumerate(argv) if value == expected_image
    )
    docker_options = (
        argv[3:image_indexes[0]] if len(image_indexes) == 1 else ()
    )
    if (
        len(argv) < 7
        or argv[:3] != ("docker", "run", "--rm")
        or argv[3:9] != (
            "--label", campaign_label,
            "--label", host_label,
            "--label", f"{_WORK_CONTAINER_LABEL}={work_id}",
        )
        or argv.count("--label") != 3
        or any(value.startswith("--label=") for value in argv)
        or len(image_indexes) != 1
        or any(
            value == "-l"
            or value.startswith("-l=")
            or value == "--label-file"
            or value.startswith("--label-file=")
            or value == "--detach"
            or value.startswith("--detach=")
            or (value.startswith("-") and not value.startswith("--"))
            for value in docker_options
        )
        or len(work_labels) != 1
        or work_labels[0] != f"{_WORK_CONTAINER_LABEL}={work_id}"
        or re.fullmatch(
            r"io\.prismaquant\.work=[a-z0-9][A-Za-z0-9_.:-]{0,255}",
            work_labels[0],
        ) is None
        or any(not isinstance(value, str) or not value or "\0" in value for value in argv)
        or "--detach" in argv
        or "-d" in argv
        or "--cidfile" in argv
        or any(value.startswith("--cidfile=") for value in argv)
        or "--name" in argv
        or any(value.startswith("--name=") for value in argv)
    ):
        raise ClusterHostAdmissionError(
            "guarded launch command differs from the fixed foreground container shape"
        )
    active_runtime = runtime or HostAdmissionRuntime()
    if active_runtime.hostname_reader() != expected["hostname"]:
        raise ClusterHostAdmissionError("live hostname differs from sealed host")
    if (
        active_runtime.uid_reader() != expected["uid"]
        or active_runtime.gid_reader() != expected["gid"]
    ):
        raise ClusterHostAdmissionError("live UID/GID differs from sealed host")
    root = _lease_root(lease_root)
    supervision = _private_supervision_directory(
        root, str(normalized["identity_sha256"]), host_id,
    )
    work_key = hashlib.sha256(work_id.encode("utf-8")).hexdigest()
    cid_path = supervision / f"{work_key}.cid"
    container_name = (
        f"pq-{str(normalized['identity_sha256'])[:16]}-{work_key[:24]}"
    )
    gpu_uuid = str(gpu["uuid"])
    expected_lease = _expected_lease(normalized, host)
    with _lease_guard(root, gpu_uuid) as descriptor:
        path = _lease_path(root, gpu_uuid)
        if not path.exists() or _read_lease(path, gpu_uuid=gpu_uuid) != expected_lease:
            raise ClusterHostAdmissionError(
                "exact GPU lease is absent at guarded container launch"
            )
        _live_gpu_identity(gpu, active_runtime)
        # Reconcile only this exact work item's privately named/CID-bound
        # container before the broad idle/orphan gate.  A wrapper crash may
        # leave that owned container active; checking GPU idleness first would
        # make deterministic cleanup and resume impossible.  Unknown or
        # mismatched containers are never controlled here.
        if cid_path.exists() or cid_path.is_symlink():
            stale_cid = _read_cid(cid_path)
            _stop_owned_container(
                stale_cid,
                campaign_identity=str(normalized["identity_sha256"]),
                host_id=host_id,
                work_id=work_id,
                run_impl=run_impl,
            )
            _remove_cid(cid_path)
        named = _inspect_container(container_name, run_impl=run_impl)
        if named is not None:
            _require_owned_container(
                named,
                campaign_identity=str(normalized["identity_sha256"]),
                host_id=host_id,
                work_id=work_id,
            )
            _stop_owned_container(
                container_name,
                campaign_identity=str(normalized["identity_sha256"]),
                host_id=host_id,
                work_id=work_id,
                run_impl=run_impl,
            )
        _require_idle_gpu(active_runtime)
        _require_no_campaign_containers(normalized, active_runtime)
        launch_argv = (
            *argv[:3], "--cidfile", str(cid_path), "--name", container_name,
            *argv[3:],
        )
        os.set_inheritable(descriptor, True)
        process = popen_impl(
            list(launch_argv),
            shell=False,
            close_fds=True,
            pass_fds=(descriptor,),
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = int(process.wait(timeout=float(timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _CONTAINER_TIMEOUT_EXIT
        finally:
            if timed_out:
                try:
                    _terminate_docker_client(process)
                finally:
                    reference = (
                        _read_cid(cid_path)
                        if cid_path.exists() else container_name
                    )
                    _stop_owned_container(
                        reference,
                        campaign_identity=str(normalized["identity_sha256"]),
                        host_id=host_id,
                        work_id=work_id,
                        run_impl=run_impl,
                    )
                    if cid_path.exists():
                        _remove_cid(cid_path)
            else:
                reference = (
                    _read_cid(cid_path)
                    if cid_path.exists() else container_name
                )
                _stop_owned_container(
                    reference,
                    campaign_identity=str(normalized["identity_sha256"]),
                    host_id=host_id,
                    work_id=work_id,
                    run_impl=run_impl,
                )
                if cid_path.exists():
                    _remove_cid(cid_path)
        return returncode


def admit_host(
    manifest: Mapping[str, object],
    host_id: str,
    *,
    lease_root: str | Path = DEFAULT_GPU_LEASE_ROOT,
    runtime: HostAdmissionRuntime | None = None,
) -> dict[str, object]:
    """Validate one exact host and acquire/adopt its durable GPU lease."""

    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    expected = host["expected"]
    roots = host["roots"]
    inputs = normalized["inputs"]
    assert isinstance(expected, Mapping)
    assert isinstance(roots, Mapping)
    assert isinstance(inputs, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    if gpu.get("device_count") != 1:
        raise ClusterHostAdmissionError(
            "sealed host must declare exactly one GPU"
        )
    active_runtime = runtime or HostAdmissionRuntime()
    if active_runtime.hostname_reader() != expected["hostname"]:
        raise ClusterHostAdmissionError("live hostname differs from sealed host")
    if (
        active_runtime.uid_reader() != expected["uid"]
        or active_runtime.gid_reader() != expected["gid"]
    ):
        raise ClusterHostAdmissionError("live UID/GID differs from sealed host")

    live_gpu = _live_gpu_identity(gpu, active_runtime)
    compute_apps = _require_idle_gpu(active_runtime)
    resources = _resource_receipt(normalized, host_id, host, active_runtime)

    dataset = _stable_regular_file_sha256(
        Path(str(roots["dataset_path"])), where="campaign dataset",
    )
    if dataset["sha256"] != inputs["dataset_sha256"]:
        raise ClusterHostAdmissionError(
            "campaign dataset SHA-256 differs from sealed manifest"
        )

    model_root = _real_directory(
        Path(str(roots["model_root"])), where="campaign model root",
    )
    cache_path = Path(str(roots["worker_state_root"])) / SOURCE_IDENTITY_CACHE_NAME
    cache_before = _stable_regular_file_sha256(
        cache_path, where="streamed-model identity cache",
    )
    reader = active_runtime.model_identity_reader or _trusted_model_identity
    try:
        model = dict(reader(model_root, cache_path))
    except ClusterHostAdmissionError:
        raise
    except Exception as exc:
        raise ClusterHostAdmissionError(
            "trusted streamed-model identity reader failed"
        ) from exc
    if set(model) != {"schema", "identity", "portable"} or model.get(
        "schema"
    ) != MODEL_IDENTITY_RECEIPT_SCHEMA:
        raise ClusterHostAdmissionError(
            "trusted streamed-model identity receipt fields differ"
        )
    compact = model.get("identity")
    cache_after = _stable_regular_file_sha256(
        cache_path, where="streamed-model identity cache",
    )
    if cache_before != cache_after:
        raise ClusterHostAdmissionError(
            "streamed-model identity cache changed during admission"
        )
    portable = model.get("portable")
    if (
        not isinstance(compact, Mapping)
        or compact.get("schema") != "prismaquant.streamed_model.identity.v1"
        or _SHA256.fullmatch(str(compact.get("content_sha256", ""))) is None
        or type(compact.get("checkpoint_shards")) is not int
        or int(compact["checkpoint_shards"]) < 1
        or type(compact.get("checkpoint_tensors")) is not int
        or int(compact["checkpoint_tensors"]) < 1
        or not isinstance(portable, Mapping)
        or portable.get("schema")
        != "prismaquant.streamed_model.portable_content.v1"
        or type(portable.get("checkpoint_shards")) is not int
        or int(portable["checkpoint_shards"]) < 1
        or type(portable.get("checkpoint_tensors")) is not int
        or int(portable["checkpoint_tensors"]) < 1
    ):
        raise ClusterHostAdmissionError(
            "trusted streamed-model identity receipt is incomplete"
        )
    portable_sha = (
        portable.get("portable_content_sha256")
        if isinstance(portable, Mapping)
        else None
    )
    if portable_sha != inputs["model_content_sha256"]:
        raise ClusterHostAdmissionError(
            "portable streamed-model content SHA-256 differs from sealed manifest"
        )
    model_receipt = {
        **model,
        "cache": str(cache_path),
        "cache_sha256": cache_after["sha256"],
        "cache_bytes": cache_after["bytes"],
    }

    lease = acquire_gpu_lease(
        normalized, host_id, lease_root=lease_root,
    )
    body: dict[str, object] = {
        "schema": HOST_ADMISSION_RECEIPT_SCHEMA,
        "campaign_identity_sha256": normalized["identity_sha256"],
        "host_id": host_id,
        "host_identity_sha256": canonical_sha256(host),
        "hostname": expected["hostname"],
        "uid": expected["uid"],
        "gid": expected["gid"],
        "gpu": live_gpu,
        "compute_apps": compute_apps,
        "resources": resources,
        "dataset": {
            "path": str(roots["dataset_path"]),
            "sha256": dataset["sha256"],
            "bytes": dataset["bytes"],
        },
        "model": model_receipt,
        "lease": lease,
    }
    return _sealed(body)


HostAction = Literal[
    "pre-admit", "admit", "guard", "inspect", "adopt", "release"
]


def build_host_action_request(
    action: HostAction,
    manifest: Mapping[str, object],
    host_id: str,
    *,
    operation_index: int = 0,
    operation_token: str | None = None,
) -> RunRequest:
    """Build one fixed request suitable for LocalTransport or SSHTransport."""

    if action not in {
        "pre-admit", "admit", "guard", "inspect", "adopt", "release",
    }:
        raise ClusterHostAdmissionError(f"unsupported host action {action!r}")
    if type(operation_index) is not int or not 0 <= operation_index <= 999:
        raise ClusterHostAdmissionError("operation_index must be in [0, 999]")
    if operation_token is not None and (
        not isinstance(operation_token, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,31}", operation_token) is None
    ):
        raise ClusterHostAdmissionError("operation_token is invalid")
    normalized = validate_campaign_manifest(manifest)
    host = _host_by_id(normalized, host_id)
    roots = host["roots"]
    assert isinstance(roots, Mapping)
    snapshot = str(roots["snapshot_root"])
    argv = (
        "python3", "-P", "-B", "-s",
        f"{snapshot}/tools/prismaquant_source_bootstrap.py",
        "run-module", "--source-root", snapshot,
        "prismaquant.cluster_host_admission", action,
        "--host-id", host_id,
    )
    digest_prefix = str(normalized["identity_sha256"])[:16]
    operation = operation_token or f"{operation_index:03d}"
    job_id = f"pq-host-{action}-{digest_prefix}-{host_id}-{operation}"
    if len(job_id) > 128:
        host_token = "host-" + hashlib.sha256(
            host_id.encode("utf-8")
        ).hexdigest()[:16]
        job_id = f"pq-host-{action}-{digest_prefix}-{host_token}-{operation}"
    return RunRequest(
        job_id=job_id,
        argv=argv,
        cwd="/",
        env=_HOST_ENV,
        timeout_seconds=300.0,
        stdin=canonical_json_bytes(normalized, where="campaign manifest"),
        inherit_env=False,
    )


def _manifest_from_stdin() -> dict[str, object]:
    try:
        text = sys.stdin.buffer.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ClusterHostAdmissionError("stdin manifest is not UTF-8") from exc
    try:
        return parse_campaign_manifest(text)
    except Exception as exc:
        raise ClusterHostAdmissionError("stdin manifest is invalid") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in (
        "pre-admit", "admit", "guard", "inspect", "adopt", "release",
    ):
        command = sub.add_parser(action)
        command.add_argument("--host-id", required=True)
    launch = sub.add_parser("guarded-launch")
    launch.add_argument("--host-id", required=True)
    launch.add_argument("--work-id", required=True)
    launch.add_argument("--timeout-seconds", required=True, type=float)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _manifest_from_stdin()
    host = _host_by_id(manifest, args.host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    if args.action == "guarded-launch":
        command = tuple(args.command)
        if command[:1] == ("--",):
            command = command[1:]
        return guarded_container_launch(
            manifest,
            args.host_id,
            command,
            work_id=args.work_id,
            timeout_seconds=args.timeout_seconds,
        )
    if args.action == "pre-admit":
        result = pre_admit_host(manifest, args.host_id)
    elif args.action == "admit":
        result = admit_host(manifest, args.host_id)
    elif args.action == "guard":
        result = guard_gpu_start(manifest, args.host_id)
    elif args.action == "inspect":
        result = inspect_gpu_lease(str(gpu["uuid"]))
    elif args.action == "adopt":
        result = adopt_gpu_lease(manifest, args.host_id)
    else:
        result = release_gpu_lease(manifest, args.host_id)
    sys.stdout.buffer.write(canonical_json_bytes(result, where="host action receipt") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClusterHostAdmissionError as exc:
        print(f"cluster-host-admission: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "ClusterHostAdmissionError",
    "DEFAULT_GPU_LEASE_ROOT",
    "GPU_LEASE_OPERATION_SCHEMA",
    "GPU_LEASE_SCHEMA",
    "GPU_START_GUARD_RECEIPT_SCHEMA",
    "HOST_ADMISSION_RECEIPT_SCHEMA",
    "HOST_PRE_ADMISSION_RECEIPT_SCHEMA",
    "HostAdmissionRuntime",
    "MODEL_IDENTITY_RECEIPT_SCHEMA",
    "SOURCE_IDENTITY_CACHE_NAME",
    "acquire_gpu_lease",
    "admit_host",
    "adopt_gpu_lease",
    "build_host_action_request",
    "guard_gpu_start",
    "guarded_container_launch",
    "inspect_gpu_lease",
    "main",
    "pre_admit_host",
    "release_gpu_lease",
]
