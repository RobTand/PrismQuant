"""Deterministic live wiring for the sealed two-host campaign.

The pure campaign contract and coordinator intentionally do not know how a
local control process reaches the two campaign hosts.  This module is the
small, imperative boundary that constructs that topology and moves the fixed
stage artifacts.  It accepts no commands or transfer paths from the manifest:
only validated host connection fields and roots are consumed, and all remote
Python programs are constant source with canonical requests on stdin.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Protocol

from prismaquant.cluster_campaign_contract import (
    STAGE_DAG,
    validate_campaign_manifest,
)
from prismaquant.cluster_live_transport import (
    HelperBootstrapReceipt,
    RsyncTransferReceipt,
    TelemetryJobAdapter,
    VerifiedRsyncSSHTransfer,
    bootstrap_ssh_helper,
)
from prismaquant.cluster_transport import (
    LocalTransport,
    SSHTransport,
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
)


DIRECTORY_SKELETON_RECEIPT_SCHEMA = (
    "prismaquant.cluster_live_runtime.directory_skeleton_receipt.v1"
)
REMOTE_DIRECTORY_REQUEST_SCHEMA = (
    "prismaquant.cluster_live_runtime.remote_directory_request.v1"
)
REMOTE_DIRECTORY_RESPONSE_SCHEMA = (
    "prismaquant.cluster_live_runtime.remote_directory_response.v1"
)
ENDPOINT_IDENTITY_RECEIPT_SCHEMA = (
    "prismaquant.cluster_live_runtime.endpoint_identity_receipt.v1"
)
ARTIFACT_ROUTE_SCHEMA = "prismaquant.cluster_live_runtime.artifact_route.v1"
ARTIFACT_TRANSFER_RECEIPT_SCHEMA = (
    "prismaquant.cluster_live_runtime.artifact_transfer_receipt.v1"
)
ARTIFACT_BARRIER_RECEIPT_SCHEMA = (
    "prismaquant.cluster_live_runtime.artifact_barrier_receipt.v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")
_STAGES = frozenset(item.stage for item in STAGE_DAG)


class ClusterLiveRuntimeError(RuntimeError):
    """Live topology construction or a fixed artifact barrier failed closed."""


class ArtifactInspector(Protocol):
    """The artifact-inspection surface consumed by CampaignCoordinator."""

    def inspect_artifact(self, absolute_host_path: str) -> TreeManifest:
        ...


class LocalArtifactInspector:
    """Build canonical manifests for artifacts on the control-process host."""

    def inspect_artifact(self, absolute_host_path: str) -> TreeManifest:
        path = Path(absolute_host_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must be absolute and traversal-free")
        return build_tree_manifest(path)


@dataclass(frozen=True, slots=True)
class DirectorySkeletonReceipt:
    """Deterministic postcondition for one host's run/state directories."""

    campaign_identity_sha256: str
    host_id: str
    directories: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.campaign_identity_sha256) is None:
            raise ValueError("campaign identity must be a SHA-256 digest")
        if _SAFE_ID.fullmatch(self.host_id) is None:
            raise ValueError("host_id is invalid")
        directories = tuple(self.directories)
        if directories != tuple(sorted(set(directories))) or not directories:
            raise ValueError("directories must be a nonempty sorted unique tuple")
        for raw in directories:
            _safe_absolute_path(raw, where="directory")
        object.__setattr__(self, "directories", directories)

    def _body(self) -> dict[str, object]:
        return {
            "schema": DIRECTORY_SKELETON_RECEIPT_SCHEMA,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "host_id": self.host_id,
            "directories": list(self.directories),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self._body())

    def to_payload(self) -> dict[str, object]:
        body = self._body()
        return {**body, "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class ArtifactRouteSpec:
    """One immutable, manifest-derived cross-host artifact route."""

    name: str
    barrier_stage: str
    source_host_id: str
    source_path: str
    destination_host_id: str
    destination_path: str

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.name) is None:
            raise ValueError("route name is invalid")
        if self.barrier_stage not in _STAGES:
            raise ValueError("route barrier_stage is invalid")
        if (
            _SAFE_ID.fullmatch(self.source_host_id) is None
            or _SAFE_ID.fullmatch(self.destination_host_id) is None
            or self.source_host_id == self.destination_host_id
        ):
            raise ValueError("route must join two distinct valid hosts")
        _safe_absolute_path(self.source_path, where="route source")
        _safe_absolute_path(self.destination_path, where="route destination")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": ARTIFACT_ROUTE_SCHEMA,
            "name": self.name,
            "barrier_stage": self.barrier_stage,
            "source_host_id": self.source_host_id,
            "source_path": self.source_path,
            "destination_host_id": self.destination_host_id,
            "destination_path": self.destination_path,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class ArtifactTransferReceipt:
    """A route-bound receipt wrapping one verified rsync publication."""

    campaign_identity_sha256: str
    route: ArtifactRouteSpec
    transfer: RsyncTransferReceipt

    def _body(self) -> dict[str, object]:
        return {
            "schema": ARTIFACT_TRANSFER_RECEIPT_SCHEMA,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "route": self.route.to_payload(),
            "route_identity_sha256": self.route.identity_sha256,
            "manifest_sha256": self.transfer.manifest_sha256,
            "total_bytes": self.transfer.total_bytes,
            "entry_count": self.transfer.entry_count,
            "direction": self.transfer.direction,
            "source": self.transfer.source,
            "destination": self.transfer.destination,
            "content_stage": self.transfer.content_stage,
            "already_present": self.transfer.already_present,
            "completed_ns": self.transfer.completed_ns,
            "transfer_identity_sha256": self.transfer.identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self._body())

    def to_payload(self) -> dict[str, object]:
        body = self._body()
        return {**body, "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class ArtifactBarrierReceipt:
    """Returned only after every route in one pre-stage barrier verifies."""

    campaign_identity_sha256: str
    stage: str
    transfers: tuple[ArtifactTransferReceipt, ...]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.campaign_identity_sha256) is None:
            raise ValueError("campaign identity must be a SHA-256 digest")
        if self.stage not in _STAGES:
            raise ValueError("barrier stage is invalid")
        transfers = tuple(self.transfers)
        if not transfers:
            raise ValueError("an artifact barrier must contain transfers")
        if any(item.route.barrier_stage != self.stage for item in transfers):
            raise ValueError("barrier contains a route for another stage")
        object.__setattr__(self, "transfers", transfers)

    def _body(self) -> dict[str, object]:
        return {
            "schema": ARTIFACT_BARRIER_RECEIPT_SCHEMA,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "stage": self.stage,
            "transfers": [item.to_payload() for item in self.transfers],
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self._body())

    def to_payload(self) -> dict[str, object]:
        body = self._body()
        return {**body, "identity_sha256": self.identity_sha256}


def _safe_absolute_path(raw: str, *, where: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{where} must be a string")
    path = PurePosixPath(raw)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} or _SAFE_COMPONENT.fullmatch(part) is None
        for part in path.parts[1:]
    ):
        raise ValueError(
            f"{where} must be an absolute path with safe components"
        )
    return path.as_posix()


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    """Return whether either normalized absolute path contains the other."""

    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _derived_control_paths(
    host: Mapping[str, object], campaign_identity: str,
) -> tuple[PurePosixPath, PurePosixPath, PurePosixPath]:
    """Derive isolated runtime/transfer roots and reject namespace collisions.

    The manifest permits arbitrary safe absolute root names. Merely choosing
    dot-prefixed sibling names is therefore insufficient: a legal declared
    root can itself use one of those names. Prove that every application-owned
    control path is disjoint from every container-visible/input root before
    either endpoint is mutated.
    """

    roots = host["roots"]
    if not isinstance(roots, Mapping):
        raise ClusterLiveRuntimeError("host roots are not an object")
    declared = {
        field: PurePosixPath(str(roots[field]))
        for field in (
            "model_root", "dataset_path", "snapshot_root", "run_root",
            "worker_state_root",
        )
    }
    state_parent = declared["worker_state_root"].parent
    run_parent = declared["run_root"].parent
    snapshot_parent = declared["snapshot_root"].parent
    if any(
        path.as_posix() == "/"
        for path in (state_parent, run_parent, snapshot_parent)
    ):
        raise ClusterLiveRuntimeError(
            "campaign roots need non-root parents for isolated control staging"
        )
    derived = (
        state_parent / ".prismaquant-cluster-control" / campaign_identity,
        run_parent / ".prismaquant-transfer-stage" / campaign_identity / "run",
        snapshot_parent / ".prismaquant-transfer-stage" / campaign_identity
        / "snapshot",
    )
    labels = ("runtime control", "run transfer", "snapshot transfer")
    for label, candidate in zip(labels, derived, strict=True):
        for field, root in declared.items():
            if _paths_overlap(candidate, root):
                raise ClusterLiveRuntimeError(
                    f"derived {label} root overlaps declared {field}"
                )
    for index, candidate in enumerate(derived):
        for other_index in range(index):
            if _paths_overlap(candidate, derived[other_index]):
                raise ClusterLiveRuntimeError(
                    f"derived {labels[index]} and {labels[other_index]} roots overlap"
                )
    return derived


def _mkdir_no_symlinks(path: Path) -> None:
    """Create one absolute directory without traversing a symlink component."""

    if not path.is_absolute():
        raise ValueError("directory must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        if _SAFE_COMPONENT.fullmatch(component) is None:
            raise ValueError("directory contains an unsafe component")
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                mode = current.lstat().st_mode
                if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                    raise ClusterLiveRuntimeError(
                        f"directory component raced with a non-directory: {current}"
                    )
            else:
                mode = current.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ClusterLiveRuntimeError(
                f"directory component is not a real directory: {current}"
            )


def _create_local_directory_skeleton(
    receipt: DirectorySkeletonReceipt,
) -> DirectorySkeletonReceipt:
    for raw in receipt.directories:
        _mkdir_no_symlinks(Path(raw))
    return receipt


_REMOTE_DIRECTORY_PROGRAM = r'''
import base64, json, os, pathlib, re, stat, sys

IN = "prismaquant.cluster_live_runtime.remote_directory_request.v1"
OUT = "prismaquant.cluster_live_runtime.remote_directory_response.v1"
SAFE = re.compile(r"[A-Za-z0-9._-]+")
HOST = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
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
        raise ValueError("directory must be a string")
    path = pathlib.PurePosixPath(raw)
    if not path.is_absolute() or any(
        part in ("", ".", "..") or SAFE.fullmatch(part) is None
        for part in path.parts[1:]
    ):
        raise ValueError("unsafe directory")
    return pathlib.Path(raw)

def ensure(path):
    current = pathlib.Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            mode = current.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ValueError("directory component is not a real directory")

try:
    encoded = sys.stdin.buffer.read().strip()
    raw = base64.b64decode(encoded, validate=True)
    request = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if canonical(request) != raw:
        raise ValueError("request is not canonical JSON")
    if set(request) != {"schema", "campaign_identity_sha256", "host_id", "directories"}:
        raise ValueError("request fields differ")
    if request["schema"] != IN:
        raise ValueError("request schema differs")
    if not isinstance(request["campaign_identity_sha256"], str) or not HEX.fullmatch(request["campaign_identity_sha256"]):
        raise ValueError("invalid campaign identity")
    if not isinstance(request["host_id"], str) or not HOST.fullmatch(request["host_id"]):
        raise ValueError("invalid host id")
    directories = request["directories"]
    if (not isinstance(directories, list) or not directories or
            directories != sorted(set(directories))):
        raise ValueError("directories must be sorted and unique")
    paths = [safe_path(item) for item in directories]
    for path in paths:
        ensure(path)
    response = {
        "schema": OUT,
        "campaign_identity_sha256": request["campaign_identity_sha256"],
        "host_id": request["host_id"],
        "directories": directories,
    }
    sys.stdout.buffer.write(canonical(response))
except Exception as exc:
    sys.stderr.write(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
'''.strip()


_ENDPOINT_IDENTITY_PROGRAM = r'''
import csv, json, os, socket, subprocess, sys

OUT = "prismaquant.cluster_live_runtime.endpoint_identity_probe.v1"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

try:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap",
         "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=False, timeout=30.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi identity query failed")
    rows = []
    for row in csv.reader(completed.stdout.decode("utf-8").splitlines(),
                          skipinitialspace=True):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != 4:
            raise ValueError("nvidia-smi identity row has wrong width")
        rows.append({"index": int(row[0].strip()), "name": row[1].strip(),
                     "uuid": row[2].strip(),
                     "compute_capability": row[3].strip()})
    response = {"schema": OUT, "hostname": socket.gethostname(),
                "uid": os.getuid(), "gid": os.getgid(), "gpus": rows}
    sys.stdout.buffer.write(canonical(response))
except Exception as exc:
    sys.stderr.write(type(exc).__name__ + ": " + str(exc))
    raise SystemExit(2)
'''.strip()


def _duplicate_refusing_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _fixed_remote_program_argv(
    ssh: SSHTransport,
    program: str,
) -> tuple[str, ...]:
    command = list(ssh.command_argv)
    # The program is a module constant.  No manifest value enters the one
    # remote-shell argument; all variable data travels on stdin below.
    command[-1] = (
        f"exec {ssh.remote_python_path} -P -B -s -c {shlex.quote(program)}"
    )
    return tuple(command)


def _endpoint_probe_receipt(
    completed: Any, host: Mapping[str, object],
) -> dict[str, object]:
    stdout = (
        completed.stdout
        if isinstance(completed.stdout, bytes)
        else str(completed.stdout or "").encode()
    )
    stderr = (
        completed.stderr
        if isinstance(completed.stderr, bytes)
        else str(completed.stderr or "").encode()
    )
    if int(completed.returncode) != 0:
        raise ClusterLiveRuntimeError(
            "read-only endpoint identity probe failed: "
            + stderr.decode("utf-8", errors="replace")
        )
    payload = stdout.strip()
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_refusing_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClusterLiveRuntimeError(
            "endpoint identity probe returned invalid JSON"
        ) from exc
    if canonical_json_bytes(value) != payload or not isinstance(value, Mapping):
        raise ClusterLiveRuntimeError(
            "endpoint identity probe response is not canonical"
        )
    expected = host["expected"]
    assert isinstance(expected, Mapping)
    expected_gpu = expected["gpu"]
    assert isinstance(expected_gpu, Mapping)
    gpus = value.get("gpus")
    expected_capability = ".".join(
        str(item) for item in expected_gpu["compute_capability"]
    )
    exact_gpu = (
        len(gpus) == 1
        and isinstance(gpus[0], Mapping)
        and set(gpus[0]) == {
            "index", "name", "uuid", "compute_capability",
        }
        and type(gpus[0].get("index")) is int
        and int(gpus[0]["index"]) >= 0
        and gpus[0].get("name") == expected_gpu["name"]
        and gpus[0].get("uuid") == expected_gpu["uuid"]
        and gpus[0].get("compute_capability") == expected_capability
    ) if isinstance(gpus, list) else False
    if (
        set(value) != {"schema", "hostname", "uid", "gid", "gpus"}
        or value.get("schema")
        != "prismaquant.cluster_live_runtime.endpoint_identity_probe.v1"
        or value.get("hostname") != expected["hostname"]
        or value.get("uid") != expected["uid"]
        or value.get("gid") != expected["gid"]
        or not isinstance(gpus, list)
        or len(gpus) != expected_gpu["device_count"]
        or not exact_gpu
    ):
        raise ClusterLiveRuntimeError(
            f"read-only endpoint identity differs for host {host['id']!r}"
        )
    body: dict[str, object] = {
        "schema": ENDPOINT_IDENTITY_RECEIPT_SCHEMA,
        "host_id": host["id"],
        "host_identity": dict(value),
    }
    return {**body, "identity_sha256": canonical_json_sha256(body)}


def _verify_endpoint_identities(
    local_host: Mapping[str, object],
    ssh_host: Mapping[str, object],
    ssh_transport: SSHTransport,
    *,
    local_run_impl: Callable[..., Any] = subprocess.run,
    ssh_run_impl: Callable[..., Any] = subprocess.run,
) -> Mapping[str, Mapping[str, object]]:
    """Verify both manifest endpoints read-only before creating any path."""

    local = local_run_impl(
        [sys.executable, "-P", "-B", "-s", "-c", _ENDPOINT_IDENTITY_PROGRAM],
        capture_output=True,
        check=False,
        shell=False,
        timeout=60.0,
    )
    remote = ssh_run_impl(
        list(_fixed_remote_program_argv(
            ssh_transport, _ENDPOINT_IDENTITY_PROGRAM,
        )),
        capture_output=True,
        check=False,
        shell=False,
        timeout=60.0,
    )
    return {
        str(local_host["id"]): _endpoint_probe_receipt(local, local_host),
        str(ssh_host["id"]): _endpoint_probe_receipt(remote, ssh_host),
    }


def _initialize_remote_directory_skeleton(
    ssh: SSHTransport,
    receipt: DirectorySkeletonReceipt,
    *,
    run_impl: Callable[..., Any] = subprocess.run,
) -> DirectorySkeletonReceipt:
    request = {
        "schema": REMOTE_DIRECTORY_REQUEST_SCHEMA,
        "campaign_identity_sha256": receipt.campaign_identity_sha256,
        "host_id": receipt.host_id,
        "directories": list(receipt.directories),
    }
    wire = base64.b64encode(canonical_json_bytes(request)) + b"\n"
    completed = run_impl(
        list(_fixed_remote_program_argv(ssh, _REMOTE_DIRECTORY_PROGRAM)),
        input=wire,
        capture_output=True,
        check=False,
        shell=False,
        timeout=60.0,
    )
    stdout = (
        completed.stdout
        if isinstance(completed.stdout, bytes)
        else str(completed.stdout or "").encode()
    )
    stderr = (
        completed.stderr
        if isinstance(completed.stderr, bytes)
        else str(completed.stderr or "").encode()
    )
    if int(completed.returncode) != 0:
        raise ClusterLiveRuntimeError(
            f"remote directory initializer exited {completed.returncode}: "
            + stderr.decode("utf-8", errors="replace")
        )
    payload = stdout.strip()
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_refusing_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClusterLiveRuntimeError(
            "remote directory initializer returned invalid JSON"
        ) from exc
    if canonical_json_bytes(value) != payload or not isinstance(value, Mapping):
        raise ClusterLiveRuntimeError(
            "remote directory initializer response is not canonical"
        )
    expected = {
        "schema": REMOTE_DIRECTORY_RESPONSE_SCHEMA,
        "campaign_identity_sha256": receipt.campaign_identity_sha256,
        "host_id": receipt.host_id,
        "directories": list(receipt.directories),
    }
    if value != expected:
        raise ClusterLiveRuntimeError(
            "remote directory initializer response differs from request"
        )
    return receipt


class _ManifestSSHTransport(SSHTransport):
    """SSHTransport that preserves the manifest's explicit TCP port."""

    def __init__(
        self,
        host: str,
        *,
        port: int,
        remote_helper_path: str,
        remote_state_root: str | None = None,
        remote_python_path: str = "/usr/bin/python3",
        ssh_binary: str = "ssh",
        run_impl: Callable[..., Any] = subprocess.run,
        connect_timeout_seconds: int = 5,
    ) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("SSH port must be an integer from 1 through 65535")
        self.port = port
        self.remote_state_root = (
            None
            if remote_state_root is None
            else _safe_absolute_path(
                remote_state_root, where="remote transport state root",
            )
        )
        super().__init__(
            host,
            remote_helper_path=remote_helper_path,
            remote_python_path=remote_python_path,
            ssh_binary=ssh_binary,
            run_impl=run_impl,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    @property
    def command_argv(self) -> tuple[str, ...]:
        command = list(super().command_argv)
        if self.remote_state_root is not None:
            command[-1] += (
                " --remote-helper-state-root "
                + shlex.quote(self.remote_state_root)
            )
        separator = command.index("--")
        command[separator:separator] = ["-p", str(self.port)]
        return tuple(command)


def _directory_receipt(
    host: Mapping[str, object],
    *,
    campaign_identity_sha256: str,
) -> DirectorySkeletonReceipt:
    roots = host["roots"]
    if not isinstance(roots, Mapping):  # defensive after contract validation
        raise ClusterLiveRuntimeError("host roots are not an object")
    run_root = PurePosixPath(str(roots["run_root"]))
    state_root = PurePosixPath(str(roots["worker_state_root"]))
    snapshot_parent = PurePosixPath(str(roots["snapshot_root"])).parent
    runtime_root, run_transfer_root, snapshot_transfer_root = (
        _derived_control_paths(host, campaign_identity_sha256)
    )
    directories = {
        run_root.as_posix(),
        (run_root / "coordinator").as_posix(),
        (run_root / "worker-0").as_posix(),
        (run_root / "worker-1").as_posix(),
        (run_root / "burn").as_posix(),
        state_root.as_posix(),
        (state_root / "home").as_posix(),
        (state_root / "cache").as_posix(),
        (state_root / "tmp").as_posix(),
        (state_root / "torchinductor").as_posix(),
        (state_root / "triton").as_posix(),
        runtime_root.as_posix(),
        (runtime_root / "transport").as_posix(),
        (runtime_root / "helper").as_posix(),
        run_transfer_root.as_posix(),
        snapshot_transfer_root.as_posix(),
    }
    directories.add(snapshot_parent.as_posix())
    return DirectorySkeletonReceipt(
        campaign_identity_sha256=campaign_identity_sha256,
        host_id=str(host["id"]),
        directories=tuple(sorted(directories)),
    )


def _runtime_root(host: Mapping[str, object], campaign_identity: str) -> PurePosixPath:
    return _derived_control_paths(host, campaign_identity)[0]


def _run_transfer_root(
    host: Mapping[str, object], campaign_identity: str,
) -> PurePosixPath:
    return _derived_control_paths(host, campaign_identity)[1]


def _run_artifact(host: Mapping[str, object], relative: str) -> str:
    roots = host["roots"]
    assert isinstance(roots, Mapping)
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ClusterLiveRuntimeError("internal artifact path is invalid")
    return (PurePosixPath(str(roots["run_root"])) / relative_path).as_posix()


def _build_routes(
    manifest: Mapping[str, object],
) -> Mapping[str, tuple[ArtifactRouteSpec, ...]]:
    hosts_value = manifest["hosts"]
    assert isinstance(hosts_value, list)
    hosts = tuple(hosts_value)
    by_id = {str(host["id"]): host for host in hosts}
    local = next(host for host in hosts if host["transport"]["kind"] == "local")  # type: ignore[index]
    remote = next(host for host in hosts if host["transport"]["kind"] == "ssh")  # type: ignore[index]
    coordinator = by_id[str(manifest["coordinator"])]
    peer = next(host for host in hosts if host is not coordinator)
    worker_index = {str(host["id"]): index for index, host in enumerate(hosts)}
    peer_index = worker_index[str(peer["id"])]

    def route(
        name: str,
        stage: str,
        source: Mapping[str, object],
        source_relative: str,
        destination: Mapping[str, object],
        destination_relative: str | None = None,
    ) -> ArtifactRouteSpec:
        return ArtifactRouteSpec(
            name=name,
            barrier_stage=stage,
            source_host_id=str(source["id"]),
            source_path=_run_artifact(source, source_relative),
            destination_host_id=str(destination["id"]),
            destination_path=_run_artifact(
                destination,
                destination_relative or source_relative,
            ),
        )

    local_roots = local["roots"]
    remote_roots = remote["roots"]
    assert isinstance(local_roots, Mapping) and isinstance(remote_roots, Mapping)
    snapshot = ArtifactRouteSpec(
        name="immutable_snapshot",
        barrier_stage="host_preflight",
        source_host_id=str(local["id"]),
        source_path=str(local_roots["snapshot_root"]),
        destination_host_id=str(remote["id"]),
        destination_path=str(remote_roots["snapshot_root"]),
    )
    routes = {
        "host_preflight": (snapshot,),
        "sample_ce": (
            route("calibration_tensor", "sample_ce", coordinator, "calibration.pt", peer),
            route("calibration_contract", "sample_ce", coordinator, "calibration.json", peer),
            route("run_contract", "sample_ce", coordinator, "run-contract.json", peer),
            route("sample_cover", "sample_ce", coordinator, "cover.json", peer),
        ),
        "merge_importance": (
            route(
                f"worker_{peer_index}_ce",
                "merge_importance",
                peer,
                f"worker-{peer_index}/ce.json",
                coordinator,
            ),
        ),
        "sample_fisher": (
            route("global_ce", "sample_fisher", coordinator, "global-ce.json", peer),
        ),
        "merge_sample_probe": (
            route(
                f"worker_{peer_index}_probe",
                "merge_sample_probe",
                peer,
                f"worker-{peer_index}/probe.pkl",
                coordinator,
            ),
            route(
                f"worker_{peer_index}_act",
                "merge_sample_probe",
                peer,
                f"worker-{peer_index}/act",
                coordinator,
            ),
        ),
        "measure_burn": (
            route("merged_bundle", "measure_burn", coordinator, "merged", peer),
            route("column_weights", "measure_burn", coordinator, "cb_col_weights.pkl", peer),
            route(
                "execution_attestation",
                "measure_burn",
                coordinator,
                "burn/execution-attestation.json",
                peer,
            ),
            route("burn_plan", "measure_burn", coordinator, "burn/plan", peer),
        ),
        "merge_burn": (
            route(
                f"worker_{peer_index}_burn_stripe",
                "merge_burn",
                peer,
                f"burn/stripe-{peer_index}.pkl",
                coordinator,
            ),
        ),
    }
    return MappingProxyType(routes)


class LiveCampaignRuntime:
    """Constructed two-host transports plus fixed pre-stage artifact barriers."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, object],
        local_transport: LocalTransport,
        ssh_transport: SSHTransport,
        transports: Mapping[str, TelemetryJobAdapter],
        artifact_inspectors: Mapping[str, ArtifactInspector],
        transfer: VerifiedRsyncSSHTransfer,
        snapshot_transfer: VerifiedRsyncSSHTransfer,
        helper_bootstrap_receipt: HelperBootstrapReceipt | None,
        directory_receipts: Mapping[str, DirectorySkeletonReceipt],
        endpoint_identity_receipts: Mapping[str, Mapping[str, object]],
    ) -> None:
        normalized = validate_campaign_manifest(manifest)
        self.manifest = MappingProxyType(normalized)
        self.local_transport = local_transport
        self.ssh_transport = ssh_transport
        self.transports = MappingProxyType(dict(transports))
        self.artifact_inspectors = MappingProxyType(dict(artifact_inspectors))
        self.transfer = transfer
        self.snapshot_transfer = snapshot_transfer
        self.helper_bootstrap_receipt = helper_bootstrap_receipt
        self.directory_receipts = MappingProxyType(dict(directory_receipts))
        self.endpoint_identity_receipts = MappingProxyType({
            key: MappingProxyType(dict(value))
            for key, value in endpoint_identity_receipts.items()
        })
        self._routes = _build_routes(normalized)

        host_ids = {str(host["id"]) for host in normalized["hosts"]}  # type: ignore[index]
        if (
            set(self.transports) != host_ids
            or set(self.artifact_inspectors) != host_ids
            or set(self.directory_receipts) != host_ids
            or (
                self.endpoint_identity_receipts
                and set(self.endpoint_identity_receipts) != host_ids
            )
        ):
            raise ClusterLiveRuntimeError(
                "runtime maps must contain exactly the two manifest hosts"
            )

    @property
    def campaign_identity_sha256(self) -> str:
        return str(self.manifest["identity_sha256"])

    def route_specs_for_stage(self, stage: str) -> tuple[ArtifactRouteSpec, ...]:
        """Return the sealed routes that must complete before ``stage``."""

        if stage not in _STAGES:
            raise ClusterLiveRuntimeError(f"unknown campaign stage {stage!r}")
        return self._routes.get(stage, ())

    @property
    def barrier_stages(self) -> tuple[str, ...]:
        """Return routed stages in fixed DAG order without exposing internals."""

        return tuple(
            spec.stage for spec in STAGE_DAG if spec.stage in self._routes
        )

    def _manifest_for(self, host_id: str, path: str) -> TreeManifest:
        try:
            inspector = self.artifact_inspectors[host_id]
            manifest = inspector.inspect_artifact(path)
        except Exception as exc:
            raise ClusterLiveRuntimeError(
                f"cannot inspect artifact {path!r} on host {host_id!r}"
            ) from exc
        if not isinstance(manifest, TreeManifest):
            raise ClusterLiveRuntimeError("artifact inspector returned an invalid manifest")
        return manifest

    def _transfer_route(self, route: ArtifactRouteSpec) -> ArtifactTransferReceipt:
        host_kind = {
            str(host["id"]): str(host["transport"]["kind"])  # type: ignore[index]
            for host in self.manifest["hosts"]  # type: ignore[union-attr]
        }
        source_kind = host_kind[route.source_host_id]
        destination_kind = host_kind[route.destination_host_id]
        before = self._manifest_for(route.source_host_id, route.source_path)
        backend = (
            self.snapshot_transfer
            if route.name == "immutable_snapshot"
            else self.transfer
        )
        try:
            if (source_kind, destination_kind) == ("local", "ssh"):
                transfer = backend.upload(
                    route.source_path,
                    route.destination_path,
                )
                expected_direction = "upload"
            elif (source_kind, destination_kind) == ("ssh", "local"):
                transfer = backend.download(
                    route.source_path,
                    route.destination_path,
                )
                expected_direction = "download"
            else:  # impossible for a valid two-host manifest
                raise ClusterLiveRuntimeError(
                    "artifact route is not local-to-SSH or SSH-to-local"
                )
        except Exception as exc:
            if isinstance(exc, ClusterLiveRuntimeError):
                raise
            raise ClusterLiveRuntimeError(
                f"artifact transfer {route.name!r} failed"
            ) from exc
        if not isinstance(transfer, RsyncTransferReceipt):
            raise ClusterLiveRuntimeError("transfer backend returned an invalid receipt")

        source_after = self._manifest_for(route.source_host_id, route.source_path)
        destination_after = self._manifest_for(
            route.destination_host_id,
            route.destination_path,
        )
        if before != source_after or before != destination_after:
            raise ClusterLiveRuntimeError(
                f"artifact {route.name!r} changed or failed destination verification"
            )
        if (
            transfer.direction != expected_direction
            or transfer.source != route.source_path
            or transfer.destination != route.destination_path
            or transfer.manifest_sha256 != before.identity_sha256
            or transfer.total_bytes != before.total_bytes
            or transfer.entry_count != len(before.entries)
            or type(transfer.already_present) is not bool
            or isinstance(transfer.completed_ns, bool)
            or not isinstance(transfer.completed_ns, int)
            or transfer.completed_ns <= 0
        ):
            raise ClusterLiveRuntimeError(
                f"artifact transfer receipt differs from route {route.name!r}"
            )
        return ArtifactTransferReceipt(
            campaign_identity_sha256=self.campaign_identity_sha256,
            route=route,
            transfer=transfer,
        )

    def synchronize_before_stage(
        self,
        stage: str,
    ) -> ArtifactBarrierReceipt | None:
        """Execute and verify the fixed artifact barrier before one DAG stage.

        Stages without cross-host inputs return ``None``.  A receipt is never
        returned for a partial barrier; retrying safely adopts exact existing
        destinations through :class:`VerifiedRsyncSSHTransfer`.
        """

        routes = self.route_specs_for_stage(stage)
        if not routes:
            return None
        transfers = tuple(self._transfer_route(route) for route in routes)
        return ArtifactBarrierReceipt(
            campaign_identity_sha256=self.campaign_identity_sha256,
            stage=stage,
            transfers=transfers,
        )

    def validate_barrier_receipt(
        self,
        stage: str,
        raw: Mapping[str, object],
    ) -> ArtifactBarrierReceipt:
        """Validate a durable barrier receipt and both current endpoint trees."""

        routes = self.route_specs_for_stage(stage)
        if not routes:
            raise ClusterLiveRuntimeError(
                f"stage {stage!r} has no cross-host artifact barrier"
            )
        expected_top = {
            "schema", "campaign_identity_sha256", "stage", "transfers",
            "identity_sha256",
        }
        if set(raw) != expected_top:
            raise ClusterLiveRuntimeError("artifact barrier receipt fields differ")
        rows = raw.get("transfers")
        if (
            raw.get("schema") != ARTIFACT_BARRIER_RECEIPT_SCHEMA
            or raw.get("campaign_identity_sha256")
            != self.campaign_identity_sha256
            or raw.get("stage") != stage
            or not isinstance(rows, list)
            or len(rows) != len(routes)
        ):
            raise ClusterLiveRuntimeError("artifact barrier receipt binding differs")
        transfer_keys = {
            "schema", "campaign_identity_sha256", "route",
            "route_identity_sha256", "manifest_sha256", "total_bytes",
            "entry_count", "direction", "source", "destination",
            "content_stage", "already_present", "completed_ns",
            "transfer_identity_sha256", "identity_sha256",
        }
        transfers: list[ArtifactTransferReceipt] = []
        for route, row in zip(routes, rows, strict=True):
            if not isinstance(row, Mapping) or set(row) != transfer_keys:
                raise ClusterLiveRuntimeError(
                    f"artifact transfer receipt fields differ for {route.name!r}"
                )
            direction = (
                "upload"
                if self.manifest_host_kind(route.source_host_id) == "local"
                else "download"
            )
            if (
                _SHA256.fullmatch(str(row.get("manifest_sha256", ""))) is None
                or type(row.get("total_bytes")) is not int
                or int(row["total_bytes"]) < 0
                or type(row.get("entry_count")) is not int
                or int(row["entry_count"]) < 1
                or type(row.get("completed_ns")) is not int
                or int(row["completed_ns"]) <= 0
                or type(row.get("already_present")) is not bool
                or row.get("direction") != direction
                or row.get("source") != route.source_path
                or row.get("destination") != route.destination_path
                or not isinstance(row.get("content_stage"), str)
            ):
                raise ClusterLiveRuntimeError(
                    f"artifact transfer receipt types differ for {route.name!r}"
                )
            try:
                content_stage = _safe_absolute_path(
                    str(row["content_stage"]), where="transfer content stage",
                )
                transfer = RsyncTransferReceipt(
                    direction=direction,
                    source=route.source_path,
                    destination=route.destination_path,
                    manifest_sha256=str(row["manifest_sha256"]),
                    total_bytes=int(row["total_bytes"]),
                    entry_count=int(row["entry_count"]),
                    content_stage=content_stage,
                    already_present=row["already_present"],  # type: ignore[arg-type]
                    completed_ns=int(row["completed_ns"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ClusterLiveRuntimeError(
                    f"artifact transfer receipt is malformed for {route.name!r}"
                ) from exc
            receipt = ArtifactTransferReceipt(
                campaign_identity_sha256=self.campaign_identity_sha256,
                route=route,
                transfer=transfer,
            )
            if dict(row) != receipt.to_payload():
                raise ClusterLiveRuntimeError(
                    f"artifact transfer receipt differs for {route.name!r}"
                )
            source = self._manifest_for(route.source_host_id, route.source_path)
            destination = self._manifest_for(
                route.destination_host_id, route.destination_path,
            )
            if (
                source != destination
                or source.identity_sha256 != transfer.manifest_sha256
                or source.total_bytes != transfer.total_bytes
                or len(source.entries) != transfer.entry_count
            ):
                raise ClusterLiveRuntimeError(
                    f"artifact barrier content drifted for {route.name!r}"
                )
            transfers.append(receipt)
        receipt = ArtifactBarrierReceipt(
            campaign_identity_sha256=self.campaign_identity_sha256,
            stage=stage,
            transfers=tuple(transfers),
        )
        if dict(raw) != receipt.to_payload():
            raise ClusterLiveRuntimeError("artifact barrier receipt digest differs")
        return receipt

    def manifest_host_kind(self, host_id: str) -> str:
        for host in self.manifest["hosts"]:  # type: ignore[index]
            if isinstance(host, Mapping) and host.get("id") == host_id:
                transport = host["transport"]
                assert isinstance(transport, Mapping)
                return str(transport["kind"])
        raise ClusterLiveRuntimeError(f"manifest has no host {host_id!r}")


def build_live_campaign_runtime(
    manifest: Mapping[str, object],
    *,
    local_run_impl: Callable[..., Any] = subprocess.run,
    local_popen_impl: Callable[..., Any] = subprocess.Popen,
    ssh_run_impl: Callable[..., Any] = subprocess.run,
    rsync_run_impl: Callable[..., Any] = subprocess.run,
    ssh_binary: str = "ssh",
    rsync_binary: str = "rsync",
    remote_python_path: str = "/usr/bin/python3",
    connect_timeout_seconds: int = 5,
    transfer_timeout_seconds: float = 3600.0,
    initialize: bool = True,
    endpoint_verifier: Callable[..., Mapping[str, Mapping[str, object]]] | None = None,
) -> LiveCampaignRuntime:
    """Build exactly one local and one SSH live transport from ``manifest``.

    ``initialize=False`` is the read-only open path used by ``verify``.  It
    constructs deterministic endpoints and inspectors but does not create
    directories or install the remote helper; subsequent inspection still
    verifies the helper's exact source digest before trusting it.
    """

    normalized = validate_campaign_manifest(manifest)
    campaign_identity = str(normalized["identity_sha256"])
    hosts = normalized["hosts"]
    assert isinstance(hosts, list)
    local_host = next(
        host for host in hosts if host["transport"]["kind"] == "local"  # type: ignore[index]
    )
    ssh_host = next(
        host for host in hosts if host["transport"]["kind"] == "ssh"  # type: ignore[index]
    )

    receipts = {
        str(host["id"]): _directory_receipt(
            host,
            campaign_identity_sha256=campaign_identity,
        )
        for host in hosts
    }
    if type(initialize) is not bool:
        raise TypeError("initialize must be a boolean")
    local_runtime_root = _runtime_root(local_host, campaign_identity)
    remote_runtime_root = _runtime_root(ssh_host, campaign_identity)
    local_transport = LocalTransport(
        local_runtime_root / "transport",
        run_impl=local_run_impl,
        popen_impl=local_popen_impl,
        transport_name=f"local:{local_host['id']}",
    )

    ssh_config = ssh_host["transport"]
    assert isinstance(ssh_config, Mapping)
    target = f"{ssh_config['user']}@{ssh_config['host']}"
    ssh_transport = _ManifestSSHTransport(
        target,
        port=int(ssh_config["port"]),
        remote_helper_path=(remote_runtime_root / "helper" / "cluster_transport.py").as_posix(),
        remote_state_root=(remote_runtime_root / "transport").as_posix(),
        remote_python_path=remote_python_path,
        ssh_binary=ssh_binary,
        run_impl=ssh_run_impl,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    endpoint_receipts: Mapping[str, Mapping[str, object]] = {}
    if initialize:
        active_endpoint_verifier = endpoint_verifier or _verify_endpoint_identities
        endpoint_receipts = active_endpoint_verifier(
            local_host,
            ssh_host,
            ssh_transport,
            local_run_impl=local_run_impl,
            ssh_run_impl=ssh_run_impl,
        )
        if set(endpoint_receipts) != {
            str(local_host["id"]), str(ssh_host["id"]),
        }:
            raise ClusterLiveRuntimeError(
                "endpoint verifier omitted a manifest host"
            )
        _create_local_directory_skeleton(receipts[str(local_host["id"])])
    helper_receipt = None
    if initialize:
        _initialize_remote_directory_skeleton(
            ssh_transport,
            receipts[str(ssh_host["id"])],
            run_impl=ssh_run_impl,
        )
        helper_receipt = bootstrap_ssh_helper(
            ssh_transport,
            run_impl=ssh_run_impl,
        )

    local_roots = local_host["roots"]
    remote_roots = ssh_host["roots"]
    assert isinstance(local_roots, Mapping)
    assert isinstance(remote_roots, Mapping)
    transfer = VerifiedRsyncSSHTransfer(
        ssh_transport,
        remote_stage_root=_run_transfer_root(
            ssh_host, campaign_identity,
        ).as_posix(),
        local_stage_root=Path(_run_transfer_root(
            local_host, campaign_identity,
        ).as_posix()),
        ssh_run_impl=ssh_run_impl,
        rsync_run_impl=rsync_run_impl,
        rsync_binary=rsync_binary,
        timeout_seconds=transfer_timeout_seconds,
    )
    local_snapshot_stage = _derived_control_paths(
        local_host, campaign_identity,
    )[2]
    remote_snapshot_stage = _derived_control_paths(
        ssh_host, campaign_identity,
    )[2]
    snapshot_transfer = VerifiedRsyncSSHTransfer(
        ssh_transport,
        remote_stage_root=remote_snapshot_stage.as_posix(),
        local_stage_root=Path(local_snapshot_stage.as_posix()),
        ssh_run_impl=ssh_run_impl,
        rsync_run_impl=rsync_run_impl,
        rsync_binary=rsync_binary,
        timeout_seconds=transfer_timeout_seconds,
    )
    cadence_seconds = (
        int(normalized["policy"]["telemetry"]["interval_milliseconds"])  # type: ignore[index]
        / 1000.0
    )
    adapters = {
        str(local_host["id"]): TelemetryJobAdapter(
            local_transport,
            cadence_seconds=cadence_seconds,
            require_samples=False,
        ),
        str(ssh_host["id"]): TelemetryJobAdapter(
            ssh_transport,
            cadence_seconds=cadence_seconds,
            require_samples=False,
        ),
    }
    inspectors: dict[str, ArtifactInspector] = {
        str(local_host["id"]): LocalArtifactInspector(),
        str(ssh_host["id"]): transfer,
    }
    return LiveCampaignRuntime(
        manifest=normalized,
        local_transport=local_transport,
        ssh_transport=ssh_transport,
        transports=adapters,
        artifact_inspectors=inspectors,
        transfer=transfer,
        snapshot_transfer=snapshot_transfer,
        helper_bootstrap_receipt=helper_receipt,
        directory_receipts=receipts,
        endpoint_identity_receipts=endpoint_receipts,
    )


__all__ = [
    "ARTIFACT_BARRIER_RECEIPT_SCHEMA",
    "ARTIFACT_ROUTE_SCHEMA",
    "ARTIFACT_TRANSFER_RECEIPT_SCHEMA",
    "DIRECTORY_SKELETON_RECEIPT_SCHEMA",
    "ENDPOINT_IDENTITY_RECEIPT_SCHEMA",
    "ArtifactBarrierReceipt",
    "ArtifactInspector",
    "ArtifactRouteSpec",
    "ArtifactTransferReceipt",
    "ClusterLiveRuntimeError",
    "DirectorySkeletonReceipt",
    "LiveCampaignRuntime",
    "LocalArtifactInspector",
    "build_live_campaign_runtime",
]
