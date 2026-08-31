"""Fail-closed execution identity for the GLM numeric campaign drivers.

Publication-capable runs use one pinned image and a closed physical-host/GPU
map. A host-side helper inspects the *created, not yet started* Docker
container and writes a no-clobber launch record into a directory mounted
read-only in the container. The publisher joins that external Docker-inspect
evidence to live UTS, container-id, GPU-UUID, commit, and clean-tree checks.

This is not cryptographic image self-attestation: an ordinary process inside
a container cannot independently recover the Docker daemon's image identity.
The durable receipt states that limit explicitly. The host record is a launch
gate, not a signature or a trust boundary against a hostile host operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path, PurePosixPath
from typing import Mapping, NoReturn


class NumericExecutionContractError(RuntimeError):
    """The live process cannot prove its numeric execution identity."""


CAMPAIGN_IMAGE_REFERENCE = (
    "eugr/spark-vllm@sha256:"
    "58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
CAMPAIGN_IMAGE_DIGEST = (
    "sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
CAMPAIGN_IMAGE_IDS = frozenset({CAMPAIGN_IMAGE_DIGEST})
PHYSICAL_HOSTS = {
    "sparky": {
        "uts_hostname": "sparky",
        "gpu_uuid": "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4",
        "gpu_name": "NVIDIA GB10",
    },
    "sparklina": {
        "uts_hostname": "gx10-6b77",
        "gpu_uuid": "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705",
        "gpu_name": "NVIDIA GB10",
    },
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BINARY = "/usr/bin/git"
_DOCKER_BINARY = "/usr/bin/docker"
_NVIDIA_SMI_BINARY = "/usr/bin/nvidia-smi"
_ENV_KEYS = frozenset({
    "torch", "python", "host", "device", "triton", "container_image",
})
_ATTESTATION_KEYS = frozenset({
    "schema",
    "verification_scope",
    "physical_host",
    "uts_hostname",
    "gpu_uuid",
    "container_id",
    "container_hostname",
    "container_state",
    "container_user",
    "image_reference",
    "image_digest",
    "image_id",
    "uts_mode",
    "network_mode",
    "ipc_mode",
    "gpu_request",
    "launch_attestation_container_path",
    "repo_root",
    "git_common_dir",
    "repo_mount_readonly",
    "git_mount_readonly",
    "attestation_sha256",
})
EXECUTION_RECORD_KEYS = frozenset({
    "schema",
    "physical_host",
    "uts_hostname",
    "gpu_uuid",
    "container_image_reference",
    "container_image_digest",
    "container_image_id",
    "container_image_evidence",
    "container_image_in_process_verification",
    "container_user",
    "ipc_mode",
    "repo_root",
    "source_mount_evidence",
    "repo_git_commit",
    "repo_tree_clean",
    "python",
    "torch",
    "triton",
    "device",
})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json(raw: str) -> object:
    def reject_constant(token: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant {token!r}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            [_GIT_BINARY, "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=_git_process_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NumericExecutionContractError(
            f"cannot attest repository with git: {detail.strip()}"
        ) from exc
    return result.stdout.strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            [_GIT_BINARY, "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            env=_git_process_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise NumericExecutionContractError(
            f"cannot attest repository bytes with git: {detail or exc}"
        ) from exc
    return result.stdout


def _git_process_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update({
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
    })
    return environment


def require_repo_commit(repo_root: Path) -> str:
    """Return a full live HEAD or refuse; ``None`` is never provenance."""

    commit = _git(repo_root, "rev-parse", "HEAD")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise NumericExecutionContractError(
            f"repository HEAD is not a full lowercase commit: {commit!r}"
        )
    return commit


def require_repo_tree_matches_head(repo_root: Path) -> None:
    """Hash every HEAD blob from worktree bytes, ignoring index hint flags.

    ``git status`` may intentionally trust ``assume-unchanged`` or
    ``skip-worktree`` bits.  Publication cannot: each tracked regular file and
    symlink is independently re-hashed using Git's blob framing.
    """

    listing = _git_bytes(repo_root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    for encoded in listing.split(b"\0"):
        if not encoded:
            continue
        metadata, separator, path_bytes = encoded.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise NumericExecutionContractError("Git HEAD tree listing is malformed")
        mode, object_type, expected_oid = fields
        if object_type != b"blob" or mode not in (b"100644", b"100755", b"120000"):
            raise NumericExecutionContractError(
                "numeric publication refuses non-blob or unsupported tracked entries"
            )
        path = repo_root / os.fsdecode(path_bytes)
        try:
            if mode == b"120000":
                if not path.is_symlink():
                    raise NumericExecutionContractError(
                        f"tracked symlink is unavailable or changed: {path}"
                    )
                payload = os.fsencode(os.readlink(path))
            else:
                if path.is_symlink() or not path.is_file():
                    raise NumericExecutionContractError(
                        f"tracked file is unavailable or changed: {path}"
                    )
                executable = bool(path.stat().st_mode & 0o111)
                if executable != (mode == b"100755"):
                    raise NumericExecutionContractError(
                        f"tracked executable mode differs from HEAD: {path}"
                    )
                payload = path.read_bytes()
        except OSError as exc:
            raise NumericExecutionContractError(
                f"cannot hash tracked source {path}: {exc}"
            ) from exc
        if len(expected_oid) == 40:
            digest = hashlib.sha1(usedforsecurity=False)
        elif len(expected_oid) == 64:
            digest = hashlib.sha256()
        else:
            raise NumericExecutionContractError("Git object ID width is unsupported")
        digest.update(f"blob {len(payload)}\0".encode("ascii"))
        digest.update(payload)
        if digest.hexdigest().encode("ascii") != expected_oid:
            raise NumericExecutionContractError(
                f"tracked source bytes differ from HEAD: {path}"
            )
    untracked = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    if any(untracked.split(b"\0")):
        first = next(path for path in untracked.split(b"\0") if path)
        raise NumericExecutionContractError(
            f"untracked worktree path is forbidden: {os.fsdecode(first)}"
        )
    ignored = _git_bytes(
        repo_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    executable_suffixes = (b".py", b".pyc", b".pyo", b".so", b".pth", b".egg-link")
    for path_bytes in ignored.split(b"\0"):
        lowered = path_bytes.lower()
        if lowered.endswith(executable_suffixes):
            raise NumericExecutionContractError(
                "ignored executable source/code is forbidden in a numeric worktree: "
                f"{os.fsdecode(path_bytes)}"
            )


def _docker_inspect(container_id: str) -> Mapping[str, object]:
    try:
        result = subprocess.run(
            [_DOCKER_BINARY, "inspect", "--type", "container", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
        decoded = _strict_json(result.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NumericExecutionContractError(
            f"cannot obtain strict Docker inspect evidence: {detail.strip()}"
        ) from exc
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise NumericExecutionContractError(
            "Docker inspect must return exactly one container object"
        )
    inspected = decoded[0]
    if not isinstance(inspected, dict):
        raise NumericExecutionContractError("Docker inspect object is malformed")
    return inspected


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise NumericExecutionContractError(f"Docker inspect {field} is malformed")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise NumericExecutionContractError(f"Docker inspect {field} is malformed")
    return value


def _inspect_environment(config: Mapping[str, object]) -> dict[str, str]:
    entries = _string_list(config.get("Env"), "Config.Env")
    result: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key or key in result:
            raise NumericExecutionContractError(
                "Docker inspect Config.Env is ambiguous or malformed"
            )
        result[key] = value
    return result


def _has_gpu_request(host_config: Mapping[str, object]) -> bool:
    requests = host_config.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1:
        return False
    request = requests[0]
    if not isinstance(request, dict) or request.get("Driver") not in ("", "nvidia"):
        return False
    count = request.get("Count")
    device_ids = request.get("DeviceIDs")
    requests_one_or_all = count in (1, -1) or (
        isinstance(device_ids, list) and len(device_ids) == 1
    )
    capabilities = request.get("Capabilities")
    return bool(
        requests_one_or_all
        and isinstance(capabilities, list)
        and any(
            isinstance(group, list) and "gpu" in group
            for group in capabilities
        )
    )


def _require_exact_readonly_mount(
    mounts: object, destination: str, *, description: str,
) -> None:
    if not isinstance(mounts, list):
        raise NumericExecutionContractError("Docker inspect Mounts is malformed")
    matching = [
        mount for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == destination
    ]
    if len(matching) != 1 or matching[0].get("RW") is not False:
        raise NumericExecutionContractError(
            f"{description} must be one exact read-only Docker mount"
        )


def _reject_overlapping_writable_mounts(
    mounts: object, protected_roots: tuple[str, ...],
) -> None:
    if not isinstance(mounts, list):
        raise NumericExecutionContractError("Docker inspect Mounts is malformed")
    protected = tuple(PurePosixPath(root) for root in protected_roots)
    for mount in mounts:
        if not isinstance(mount, dict):
            raise NumericExecutionContractError("Docker inspect Mounts is malformed")
        if mount.get("RW") is not True:
            continue
        destination = mount.get("Destination")
        if not isinstance(destination, str) or not destination.startswith("/"):
            raise NumericExecutionContractError(
                "Docker writable mount destination is malformed"
            )
        candidate = PurePosixPath(destination)
        if any(
            candidate == root
            or candidate in root.parents
            or root in candidate.parents
            for root in protected
        ):
            raise NumericExecutionContractError(
                "Docker writable mount overlaps the protected repository/Git roots"
            )


def build_launch_attestation(
    inspected: Mapping[str, object], physical_host: str,
) -> dict[str, object]:
    """Validate a Docker inspect object and return its closed launch record."""

    host_spec = PHYSICAL_HOSTS.get(physical_host)
    if host_spec is None:
        raise NumericExecutionContractError(
            f"physical host {physical_host!r} is outside the numeric campaign map"
        )
    config = _mapping(inspected.get("Config"), "Config")
    host_config = _mapping(inspected.get("HostConfig"), "HostConfig")
    state = _mapping(inspected.get("State"), "State")
    container_id = inspected.get("Id")
    image_id = inspected.get("Image")
    if not isinstance(container_id, str) or _CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise NumericExecutionContractError("Docker inspect container ID is not full sha256")
    if image_id not in CAMPAIGN_IMAGE_IDS:
        raise NumericExecutionContractError(
            "Docker inspect image ID is outside the allowed campaign image-ID set"
        )
    if state.get("Status") != "created":
        raise NumericExecutionContractError(
            "launch attestation must inspect a created, not-yet-started container"
        )
    if config.get("Image") != CAMPAIGN_IMAGE_REFERENCE:
        raise NumericExecutionContractError(
            "Docker Config.Image is not the allowed campaign image reference"
        )
    if config.get("User") != "1000:1000":
        raise NumericExecutionContractError(
            "Docker launch must use the approved non-root --user 1000:1000"
        )
    container_hostname = config.get("Hostname")
    if container_hostname != container_id[:12]:
        raise NumericExecutionContractError(
            "Docker Config.Hostname must be the default container-ID prefix"
        )
    if host_config.get("UTSMode") != "host":
        raise NumericExecutionContractError("Docker launch must use --uts=host")
    if host_config.get("NetworkMode") != "none":
        raise NumericExecutionContractError("Docker launch must use --network=none")
    if host_config.get("IpcMode") != "private":
        raise NumericExecutionContractError("Docker launch must use --ipc=private")
    if not _has_gpu_request(host_config):
        raise NumericExecutionContractError(
            "Docker launch must request exactly one GPU or all GPUs"
        )

    environment = _inspect_environment(config)
    if "HOSTNAME" in environment:
        raise NumericExecutionContractError(
            "Docker launch must not override the container-ID HOSTNAME variable"
        )
    if environment.get("HULL_PHYSICAL_HOST") != physical_host:
        raise NumericExecutionContractError(
            "Docker launch HULL_PHYSICAL_HOST does not match the inspected host"
        )
    if environment.get("HULL_CONTAINER_IMAGE") != CAMPAIGN_IMAGE_DIGEST:
        raise NumericExecutionContractError(
            "Docker launch HULL_CONTAINER_IMAGE is not the campaign digest"
        )
    attestation_path = environment.get("HULL_LAUNCH_ATTESTATION", "")
    if not attestation_path.startswith("/") or Path(attestation_path).name in ("", ".", ".."):
        raise NumericExecutionContractError(
            "Docker launch needs an absolute HULL_LAUNCH_ATTESTATION file path"
        )
    repo_root = environment.get("HULL_REPO_ROOT", "")
    git_common_dir = environment.get("HULL_GIT_COMMON_DIR", "")
    if not repo_root.startswith("/") or repo_root == "/":
        raise NumericExecutionContractError(
            "Docker launch needs a scoped absolute HULL_REPO_ROOT"
        )
    if not git_common_dir.startswith("/") or git_common_dir == "/":
        raise NumericExecutionContractError(
            "Docker launch needs a scoped absolute HULL_GIT_COMMON_DIR"
        )

    mounts = inspected.get("Mounts")
    parent = str(Path(attestation_path).parent)
    _require_exact_readonly_mount(
        mounts, parent, description="launch-attestation parent directory"
    )
    _require_exact_readonly_mount(
        mounts, repo_root, description="numeric repository root"
    )
    _require_exact_readonly_mount(
        mounts, git_common_dir, description="numeric Git common directory"
    )
    _reject_overlapping_writable_mounts(mounts, (repo_root, git_common_dir))

    unsigned: dict[str, object] = {
        "schema": "trellis.numeric_launch_attestation.v1",
        "verification_scope": "host_docker_daemon_inspect_before_start",
        "physical_host": physical_host,
        "uts_hostname": host_spec["uts_hostname"],
        "gpu_uuid": host_spec["gpu_uuid"],
        "container_id": container_id,
        "container_hostname": container_hostname,
        "container_state": "created",
        "container_user": "1000:1000",
        "image_reference": CAMPAIGN_IMAGE_REFERENCE,
        "image_digest": CAMPAIGN_IMAGE_DIGEST,
        "image_id": image_id,
        "uts_mode": "host",
        "network_mode": "none",
        "ipc_mode": "private",
        "gpu_request": "one_or_all_gpu_device_request",
        "launch_attestation_container_path": attestation_path,
        "repo_root": repo_root,
        "git_common_dir": git_common_dir,
        "repo_mount_readonly": True,
        "git_mount_readonly": True,
    }
    return {**unsigned, "attestation_sha256": _identity_sha256(unsigned)}


def write_launch_attestation(
    container_id: str, physical_host: str, output_path: Path,
) -> dict[str, object]:
    """Inspect once and atomically create a no-clobber host-side record."""

    record = build_launch_attestation(_docker_inspect(container_id), physical_host)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(record) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(output_path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise NumericExecutionContractError(
            f"cannot create launch attestation without clobbering: {exc}"
        ) from exc
    return record


def _read_launch_attestation(path_text: str) -> Mapping[str, object]:
    if not path_text.startswith("/"):
        raise NumericExecutionContractError(
            "HULL_LAUNCH_ATTESTATION must be an absolute file path"
        )
    path = Path(path_text)
    try:
        if path.is_symlink() or not path.is_file():
            raise NumericExecutionContractError(
                "HULL_LAUNCH_ATTESTATION must name a regular non-symlink file"
            )
        decoded = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise NumericExecutionContractError(
            f"cannot read strict launch attestation: {exc}"
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != _ATTESTATION_KEYS:
        raise NumericExecutionContractError(
            "launch attestation field set differs from the pinned contract"
        )
    digest = decoded.get("attestation_sha256")
    unsigned = {key: value for key, value in decoded.items() if key != "attestation_sha256"}
    if not isinstance(digest, str) or digest != _identity_sha256(unsigned):
        raise NumericExecutionContractError("launch attestation identity digest mismatch")
    return decoded


def _live_gpu_identity() -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                _NVIDIA_SMI_BINARY,
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NumericExecutionContractError(
            f"cannot query live CUDA GPU UUID: {detail.strip()}"
        ) from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise NumericExecutionContractError(
            "numeric publication requires exactly one nvidia-smi-visible GPU"
        )
    gpu_uuid, separator, gpu_name = lines[0].partition(",")
    if not separator or not gpu_uuid.strip() or not gpu_name.strip():
        raise NumericExecutionContractError("nvidia-smi GPU identity is malformed")
    return gpu_uuid.strip(), gpu_name.strip()


def validate_numeric_execution_record(record: object) -> None:
    """Validate the closed durable projection of a live launch."""

    if not isinstance(record, dict) or set(record) != EXECUTION_RECORD_KEYS:
        raise NumericExecutionContractError(
            "numeric execution record field set differs from the pinned contract"
        )
    if record.get("schema") != "trellis.numeric_execution.v2":
        raise NumericExecutionContractError("unsupported numeric execution schema")
    physical_host = record.get("physical_host")
    host_spec = PHYSICAL_HOSTS.get(physical_host) if isinstance(physical_host, str) else None
    if host_spec is None:
        raise NumericExecutionContractError("numeric execution host is outside the campaign map")
    if record.get("uts_hostname") != host_spec["uts_hostname"]:
        raise NumericExecutionContractError("numeric execution UTS hostname mismatch")
    if record.get("gpu_uuid") != host_spec["gpu_uuid"]:
        raise NumericExecutionContractError("numeric execution GPU UUID mismatch")
    if record.get("device") != host_spec["gpu_name"]:
        raise NumericExecutionContractError("numeric execution GPU name mismatch")
    if record.get("container_image_reference") != CAMPAIGN_IMAGE_REFERENCE:
        raise NumericExecutionContractError("numeric execution image reference mismatch")
    if record.get("container_image_digest") != CAMPAIGN_IMAGE_DIGEST:
        raise NumericExecutionContractError("numeric execution image digest mismatch")
    if record.get("container_image_id") not in CAMPAIGN_IMAGE_IDS:
        raise NumericExecutionContractError("numeric execution image ID is not allowed")
    if record.get("container_image_evidence") != "host_docker_daemon_inspect_before_start":
        raise NumericExecutionContractError("numeric execution image evidence is unsupported")
    if record.get("container_image_in_process_verification") != "not_available":
        raise NumericExecutionContractError(
            "numeric execution must not overclaim in-process image verification"
        )
    if record.get("container_user") != "1000:1000":
        raise NumericExecutionContractError("numeric execution container user differs")
    if record.get("ipc_mode") != "private":
        raise NumericExecutionContractError("numeric execution IPC mode differs")
    repo_root = record.get("repo_root")
    if not isinstance(repo_root, str) or not repo_root.startswith("/") or repo_root == "/":
        raise NumericExecutionContractError("numeric execution repository root is invalid")
    if record.get("source_mount_evidence") != (
        "host_docker_daemon_inspect_readonly_repo_and_git"
    ):
        raise NumericExecutionContractError("numeric execution source-mount evidence differs")
    if not isinstance(record.get("repo_git_commit"), str) or _COMMIT_RE.fullmatch(
        record["repo_git_commit"]
    ) is None:
        raise NumericExecutionContractError("numeric execution commit is malformed")
    if record.get("repo_tree_clean") is not True:
        raise NumericExecutionContractError("numeric execution repository was not clean")
    for field in ("python", "torch", "triton"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise NumericExecutionContractError(
                f"numeric execution {field} identity is unavailable"
            )


def require_numeric_execution_environment(
    repo_root: Path,
    current_environment: Mapping[str, object],
    process_environment: Mapping[str, str],
    *,
    require_cuda: bool,
) -> dict[str, object]:
    """Join external launch evidence to live state and return a receipt.

    This publication contract is intentionally CUDA-only. GPU-optional dry
    runs must return before calling it and are consequently incapable of
    constructing a publication receipt.
    """

    if not require_cuda:
        raise NumericExecutionContractError(
            "GPU-optional preflight must not call the publication execution contract"
        )
    if set(current_environment) != _ENV_KEYS:
        raise NumericExecutionContractError(
            "numeric environment field set differs from the pinned contract"
        )
    if os.getuid() != 1000 or os.getgid() != 1000:
        raise NumericExecutionContractError(
            "live numeric process must run as approved non-root UID/GID 1000:1000"
        )
    physical_host = process_environment.get("HULL_PHYSICAL_HOST", "")
    host_spec = PHYSICAL_HOSTS.get(physical_host)
    if host_spec is None:
        raise NumericExecutionContractError(
            "HULL_PHYSICAL_HOST is outside the closed physical-host/GPU map"
        )
    live_host = socket.gethostname()
    if live_host != host_spec["uts_hostname"]:
        raise NumericExecutionContractError(
            f"live UTS host {live_host!r} does not equal mapped physical host "
            f"{host_spec['uts_hostname']!r}; use --uts=host"
        )
    if current_environment.get("host") != live_host:
        raise NumericExecutionContractError(
            "numeric environment did not retain the live UTS hostname"
        )

    image = process_environment.get("HULL_CONTAINER_IMAGE", "")
    if image != CAMPAIGN_IMAGE_DIGEST:
        raise NumericExecutionContractError(
            "HULL_CONTAINER_IMAGE is not the allowed campaign image digest"
        )
    if current_environment.get("container_image") != image:
        raise NumericExecutionContractError(
            "live numeric environment did not retain the campaign image digest"
        )

    attestation_path = process_environment.get("HULL_LAUNCH_ATTESTATION", "")
    attestation = _read_launch_attestation(attestation_path)
    try:
        live_repo_root = str(repo_root.resolve(strict=True))
    except OSError as exc:
        raise NumericExecutionContractError(
            f"numeric repository root is unavailable: {exc}"
        ) from exc
    declared_repo_root = process_environment.get("HULL_REPO_ROOT", "")
    declared_git_common_dir = process_environment.get("HULL_GIT_COMMON_DIR", "")
    if declared_repo_root != live_repo_root:
        raise NumericExecutionContractError(
            "HULL_REPO_ROOT does not equal the live numeric repository root"
        )
    if not declared_git_common_dir.startswith("/") or declared_git_common_dir == "/":
        raise NumericExecutionContractError("HULL_GIT_COMMON_DIR is invalid")
    exact_attestation = {
        "schema": "trellis.numeric_launch_attestation.v1",
        "verification_scope": "host_docker_daemon_inspect_before_start",
        "physical_host": physical_host,
        "uts_hostname": host_spec["uts_hostname"],
        "gpu_uuid": host_spec["gpu_uuid"],
        "container_hostname": attestation["container_id"][:12]
        if isinstance(attestation.get("container_id"), str) else None,
        "container_state": "created",
        "container_user": "1000:1000",
        "image_reference": CAMPAIGN_IMAGE_REFERENCE,
        "image_digest": CAMPAIGN_IMAGE_DIGEST,
        "uts_mode": "host",
        "network_mode": "none",
        "ipc_mode": "private",
        "gpu_request": "one_or_all_gpu_device_request",
        "launch_attestation_container_path": attestation_path,
        "repo_root": declared_repo_root,
        "git_common_dir": declared_git_common_dir,
        "repo_mount_readonly": True,
        "git_mount_readonly": True,
    }
    for field, expected in exact_attestation.items():
        if attestation.get(field) != expected:
            raise NumericExecutionContractError(
                f"launch attestation {field} does not match the live campaign contract"
            )
    container_id = attestation.get("container_id")
    image_id = attestation.get("image_id")
    if not isinstance(container_id, str) or _CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise NumericExecutionContractError("launch attestation container ID is malformed")
    if image_id not in CAMPAIGN_IMAGE_IDS:
        raise NumericExecutionContractError("launch attestation image ID is not allowed")
    if process_environment.get("HOSTNAME") != container_id[:12]:
        raise NumericExecutionContractError(
            "live container-ID HOSTNAME does not match host-side Docker inspection"
        )

    gpu_uuid, gpu_name = _live_gpu_identity()
    if gpu_uuid != host_spec["gpu_uuid"] or gpu_name != host_spec["gpu_name"]:
        raise NumericExecutionContractError(
            "live nvidia-smi GPU UUID/name does not match the physical-host map"
        )
    device = current_environment.get("device")
    if device != gpu_name:
        raise NumericExecutionContractError(
            "CUDA device identity does not match live nvidia-smi"
        )
    for field in ("python", "torch", "triton"):
        value = current_environment.get(field)
        if not isinstance(value, str) or not value:
            raise NumericExecutionContractError(
                f"numeric environment {field} identity is unavailable"
            )

    commit = require_repo_commit(repo_root)
    live_git_common_dir = _git(
        repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    try:
        normalized_git_common_dir = str(Path(live_git_common_dir).resolve(strict=True))
    except OSError as exc:
        raise NumericExecutionContractError(
            f"live Git common directory is unavailable: {exc}"
        ) from exc
    if normalized_git_common_dir != declared_git_common_dir:
        raise NumericExecutionContractError(
            "live Git common directory differs from the read-only launch mount"
        )
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise NumericExecutionContractError(
            "numeric publication requires a clean repository worktree"
        )
    require_repo_tree_matches_head(repo_root)

    record: dict[str, object] = {
        "schema": "trellis.numeric_execution.v2",
        "physical_host": physical_host,
        "uts_hostname": host_spec["uts_hostname"],
        "gpu_uuid": gpu_uuid,
        "container_image_reference": CAMPAIGN_IMAGE_REFERENCE,
        "container_image_digest": CAMPAIGN_IMAGE_DIGEST,
        "container_image_id": image_id,
        "container_image_evidence": "host_docker_daemon_inspect_before_start",
        "container_image_in_process_verification": "not_available",
        "container_user": "1000:1000",
        "ipc_mode": "private",
        "repo_root": live_repo_root,
        "source_mount_evidence": (
            "host_docker_daemon_inspect_readonly_repo_and_git"
        ),
        "repo_git_commit": commit,
        "repo_tree_clean": True,
        "python": current_environment["python"],
        "torch": current_environment["torch"],
        "triton": current_environment["triton"],
        "device": device,
    }
    validate_numeric_execution_record(record)
    return record


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create host-side Docker-inspect evidence for a numeric launch."
    )
    parser.add_argument("--container", required=True)
    parser.add_argument("--physical-host", choices=sorted(PHYSICAL_HOSTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = write_launch_attestation(args.container, args.physical_host, args.output)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CAMPAIGN_IMAGE_DIGEST",
    "CAMPAIGN_IMAGE_IDS",
    "CAMPAIGN_IMAGE_REFERENCE",
    "EXECUTION_RECORD_KEYS",
    "NumericExecutionContractError",
    "PHYSICAL_HOSTS",
    "build_launch_attestation",
    "require_numeric_execution_environment",
    "require_repo_commit",
    "require_repo_tree_matches_head",
    "validate_numeric_execution_record",
    "write_launch_attestation",
]
