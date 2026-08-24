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
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _lease_path(root: Path, gpu_uuid: str) -> Path:
    return root / f"{_lease_key(gpu_uuid)}.json"


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
    with _lease_guard(root, gpu_uuid):
        if not path.exists():
            return _lease_operation(
                action="release", disposition="already_absent", lease=None,
            )
        existing = _read_lease(path, gpu_uuid=gpu_uuid)
        if existing != expected:
            raise ClusterHostAdmissionError(
                "refusing to release another sealed campaign's GPU lease"
            )
        try:
            path.unlink()
            _fsync_directory(root)
        except OSError as exc:
            raise ClusterHostAdmissionError("cannot release GPU lease") from exc
        return _lease_operation(
            action="release", disposition="released", lease=existing,
        )


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


HostAction = Literal["admit", "inspect", "adopt", "release"]


def build_host_action_request(
    action: HostAction,
    manifest: Mapping[str, object],
    host_id: str,
    *,
    operation_index: int = 0,
) -> RunRequest:
    """Build one fixed request suitable for LocalTransport or SSHTransport."""

    if action not in {"admit", "inspect", "adopt", "release"}:
        raise ClusterHostAdmissionError(f"unsupported host action {action!r}")
    if type(operation_index) is not int or not 0 <= operation_index <= 999:
        raise ClusterHostAdmissionError("operation_index must be in [0, 999]")
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
    return RunRequest(
        job_id=(
            f"pq-host-{action}-{digest_prefix}-{host_id}-{operation_index:03d}"
        ),
        argv=argv,
        cwd="/",
        env=_HOST_ENV,
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
    for action in ("admit", "inspect", "adopt", "release"):
        command = sub.add_parser(action)
        command.add_argument("--host-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _manifest_from_stdin()
    host = _host_by_id(manifest, args.host_id)
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    gpu = expected["gpu"]
    assert isinstance(gpu, Mapping)
    if args.action == "admit":
        result = admit_host(manifest, args.host_id)
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
    "HOST_ADMISSION_RECEIPT_SCHEMA",
    "HostAdmissionRuntime",
    "MODEL_IDENTITY_RECEIPT_SCHEMA",
    "SOURCE_IDENTITY_CACHE_NAME",
    "acquire_gpu_lease",
    "admit_host",
    "adopt_gpu_lease",
    "build_host_action_request",
    "inspect_gpu_lease",
    "main",
    "release_gpu_lease",
]
