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
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, NoReturn, Sequence


class NumericExecutionContractError(RuntimeError):
    """The live process cannot prove its numeric execution identity."""


CAMPAIGN_IMAGE_REFERENCE = (
    "eugr/spark-vllm@sha256:"
    "58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
CAMPAIGN_IMAGE_DIGEST = (
    "sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
PHYSICAL_HOSTS = {
    "sparky": {
        "uts_hostname": "sparky",
        "gpu_uuid": "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4",
        "gpu_name": "NVIDIA GB10",
        "image_id": CAMPAIGN_IMAGE_DIGEST,
    },
    "sparklina": {
        "uts_hostname": "gx10-6b77",
        "gpu_uuid": "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705",
        "gpu_name": "NVIDIA GB10",
        # Re-verified from a newly created container on 2026-08-30.  Both
        # hosts resolve the same digest-qualified Config.Image and identical
        # RootFS diff IDs; Docker's local image-object ID differs on Sparklina.
        "image_id": (
            "sha256:ac631d27c1514ec3f838299d424c98892a0ba854fa642002df4c8f576bbfe9fa"
        ),
    },
}
CAMPAIGN_IMAGE_IDS_BY_HOST = {
    host: frozenset({str(spec["image_id"])})
    for host, spec in PHYSICAL_HOSTS.items()
}
CAMPAIGN_IMAGE_IDS = frozenset().union(*CAMPAIGN_IMAGE_IDS_BY_HOST.values())
CAMPAIGN_STORAGE_ROOT = "/home/rob/dq-runs"
CAMPAIGN_ENTRYPOINT = ["/opt/nvidia/nvidia_entrypoint.sh"]
CAMPAIGN_TMPFS = {
    "/tmp": "rw,nosuid,nodev,exec,size=8g,uid=1000,gid=1000,mode=0700",
}
CAMPAIGN_SCRATCH_ENVIRONMENT = {
    "CUDA_CACHE_PATH": "/tmp/cuda-cache",
    "TMPDIR": "/tmp",
    "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions",
    "TRITON_CACHE_DIR": "/tmp/triton-cache",
    "XDG_CACHE_HOME": "/tmp/cache",
}
# Canonical JSON digest of the complete 53-key Config.Env mapping embedded in
# the pinned image.  Dynamic HULL fields and the seven reviewed Python/scratch
# additions are removed before comparison.  This makes every other launch
# knob closed: an unreviewed CUDA/CUBLAS/Torch/Triton/NCCL/OMP variable, or an
# override of an image value, changes the digest and is refused.
CAMPAIGN_IMAGE_ENVIRONMENT_SHA256 = (
    "347c08481fb3c7a5207a04f2e402add7a180ed5d28f9529ed1f306924a408e04"
)
CAMPAIGN_PYSPY_PREFIX = (
    "/usr/local/bin/py-spy", "record", "--output", "<profile>",
    "--format", "speedscope", "--", "/usr/bin/python3", "-B",
)
CAMPAIGN_DRIVER_NAMES = frozenset({
    "e2m1_highrate.py", "fp8_learned_glm.py", "fp8_cb_tcq_glm.py",
})
_FORBIDDEN_ENVIRONMENT = frozenset({
    "BASH_ENV", "ENV", "GCONV_PATH", "LD_AUDIT", "LD_PRELOAD",
    "PYTHONHOME", "PYTHONINSPECT", "PYTHONPATH", "PYTHONSTARTUP",
    "PYTHONUSERBASE", "VIRTUAL_ENV",
})
_ALLOWED_PYTHON_ENVIRONMENT = frozenset({
    "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
})
_DYNAMIC_LAUNCH_ENVIRONMENT = frozenset({
    "HULL_CONTAINER_IMAGE", "HULL_GIT_COMMON_DIR",
    "HULL_LAUNCH_ATTESTATION", "HULL_PHYSICAL_HOST", "HULL_REPO_ROOT",
})

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
    "container_rootfs_changes",
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
    "storage_mount_readwrite",
    "rootfs_readonly",
    "runtime_isolation",
    "launch_environment",
    "launch_command",
    "launch_command_sha256",
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
EXECUTION_SEGMENT_KEYS = frozenset({
    "schema", "physical_host", "container_id", "image_id", "gpu_uuid",
    "launch_attestation_path", "launch_attestation_sha256",
    "launch_command_sha256",
    "segment_sha256",
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
    executable_suffixes = (
        b".py", b".pyc", b".pyo", b".so", b".pth", b".egg-link",
        b".zip", b".egg", b".whl",
    )
    for path_bytes in ignored.split(b"\0"):
        lowered = path_bytes.lower()
        ignored_path = repo_root / os.fsdecode(path_bytes)
        if ignored_path.is_symlink() or lowered.endswith(executable_suffixes):
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


def _docker_diff(container_id: str) -> tuple[str, ...]:
    """Return the created-container writable-layer changes, strictly parsed."""

    try:
        result = subprocess.run(
            [_DOCKER_BINARY, "diff", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NumericExecutionContractError(
            f"cannot inspect created-container rootfs diff: {detail.strip()}"
        ) from exc
    changes = tuple(line for line in result.stdout.splitlines() if line)
    if any(re.fullmatch(r"[ACD] /.+", line) is None for line in changes):
        raise NumericExecutionContractError(
            "Docker rootfs-diff evidence is malformed"
        )
    return changes


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


def _require_safe_environment(environment: Mapping[str, str], *, live: bool) -> None:
    scope = "live numeric process" if live else "Docker launch"
    for key in environment:
        if key in _FORBIDDEN_ENVIRONMENT:
            raise NumericExecutionContractError(
                f"{scope} forbids external import/runtime environment {key}"
            )
        if key.startswith("PYTHON") and key not in _ALLOWED_PYTHON_ENVIRONMENT:
            raise NumericExecutionContractError(
                f"{scope} forbids unreviewed Python environment {key}"
            )
        if key.startswith("LD_") and key != "LD_LIBRARY_PATH":
            raise NumericExecutionContractError(
                f"{scope} forbids dynamic-loader environment {key}"
            )
    required = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        **CAMPAIGN_SCRATCH_ENVIRONMENT,
    }
    for key, expected in required.items():
        if environment.get(key) != expected:
            raise NumericExecutionContractError(
                f"{scope} requires {key}={expected}"
            )
    exact_image_paths = {
        "PATH": (
            "/workspace/vllm:/usr/local/nvidia/bin:/usr/local/cuda/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "LD_LIBRARY_PATH": (
            "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
            "/usr/local/cuda/lib64"
        ),
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }
    for key, expected in exact_image_paths.items():
        actual = environment.get(key)
        if live and key == "NVIDIA_VISIBLE_DEVICES" and actual == "void":
            # Docker's NVIDIA CDI hook replaces the image's declarative `all`
            # with `void` after injecting the requested GPU/device libraries.
            # The created-container DeviceRequest is independently exact.
            continue
        if actual != expected:
            raise NumericExecutionContractError(
                f"{scope} {key} differs from the pinned image launch"
            )

    normalized = dict(environment)
    if live:
        runtime_exact = {
            "HOME": "/home/ubuntu",
            "LC_CTYPE": "C.UTF-8",
            "NVIDIA_CTK_LIBCUDA_DIR": "/usr/lib/aarch64-linux-gnu",
            "SHLVL": "0",
        }
        for key, expected in runtime_exact.items():
            if normalized.pop(key, None) != expected:
                raise NumericExecutionContractError(
                    f"{scope} runtime-generated {key} differs"
                )
        hostname = normalized.pop("HOSTNAME", None)
        if not isinstance(hostname, str) or re.fullmatch(r"[0-9a-f]{12}", hostname) is None:
            raise NumericExecutionContractError(
                f"{scope} runtime-generated HOSTNAME is malformed"
            )
        if normalized.pop("PWD", None) != environment.get("HULL_REPO_ROOT"):
            raise NumericExecutionContractError(
                f"{scope} runtime-generated PWD differs from HULL_REPO_ROOT"
            )
        if normalized.get("NVIDIA_VISIBLE_DEVICES") == "void":
            normalized["NVIDIA_VISIBLE_DEVICES"] = "all"
    for key in (
        *_DYNAMIC_LAUNCH_ENVIRONMENT,
        *_ALLOWED_PYTHON_ENVIRONMENT,
        *CAMPAIGN_SCRATCH_ENVIRONMENT,
    ):
        normalized.pop(key, None)
    if hashlib.sha256(_canonical_json(normalized)).hexdigest() != (
        CAMPAIGN_IMAGE_ENVIRONMENT_SHA256
    ):
        raise NumericExecutionContractError(
            f"{scope} environment schema differs from the exact pinned-image baseline"
        )


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


def _require_closed_mounts(
    mounts: object,
    *,
    attestation_parent: str,
    host_attestation_parent: str,
    repo_root: str,
    git_common_dir: str,
) -> None:
    """Accept exactly the four bind mounts in the reviewed campaign launcher.

    The broad campaign-storage bind is the only writable mount.  The launch
    record lives in a nested read-only bind, while the immutable repository
    and its worktree-external Git common directory are separate read-only
    binds.  Rejecting *every* other mount prevents a digest-pinned image from
    being silently overlaid at /usr/bin, site-packages, CUDA, or ld.so paths.
    NVIDIA's device request is validated separately; daemon/runtime device
    injection does not appear as an extra created-container bind mount.
    """

    if not isinstance(mounts, list):
        raise NumericExecutionContractError("Docker inspect Mounts is malformed")
    expected = {
        attestation_parent: (host_attestation_parent, False),
        repo_root: (repo_root, False),
        git_common_dir: (git_common_dir, False),
        CAMPAIGN_STORAGE_ROOT: (CAMPAIGN_STORAGE_ROOT, True),
    }
    if len(expected) != 4:
        raise NumericExecutionContractError(
            "campaign mount destinations must be four distinct paths"
        )
    observed: dict[str, tuple[str, bool]] = {}
    for mount in mounts:
        if not isinstance(mount, dict):
            raise NumericExecutionContractError("Docker inspect Mounts is malformed")
        destination = mount.get("Destination")
        source = mount.get("Source")
        rw = mount.get("RW")
        if (
            mount.get("Type") != "bind"
            or mount.get("Mode") != ""
            or mount.get("Propagation") != "rprivate"
            or not isinstance(destination, str)
            or not destination.startswith("/")
            or not isinstance(source, str)
            or not source.startswith("/")
            or not isinstance(rw, bool)
            or destination in observed
        ):
            raise NumericExecutionContractError(
                "Docker launch mounts must be unique default-propagation bind mounts"
            )
        observed[destination] = (source, rw)
    if observed != expected:
        raise NumericExecutionContractError(
            "Docker launch mount set differs from the closed campaign allowlist"
        )

    storage = PurePosixPath(CAMPAIGN_STORAGE_ROOT)
    attestation = PurePosixPath(attestation_parent)
    repository = PurePosixPath(repo_root)
    git = PurePosixPath(git_common_dir)
    if not attestation.is_relative_to(storage) or attestation == storage:
        raise NumericExecutionContractError(
            "launch-attestation mount must be strictly nested below campaign storage"
        )
    for protected, label in ((repository, "repository"), (git, "Git common")):
        if protected.is_relative_to(storage) or storage.is_relative_to(protected):
            raise NumericExecutionContractError(
                f"{label} mount must not overlap writable campaign storage"
            )
    if repository.is_relative_to(git) or git.is_relative_to(repository):
        raise NumericExecutionContractError(
            "repository and Git-common mounts must not overlap"
        )


def _require_rootfs_mount_scaffolding(
    changes: Sequence[str], *, mount_destinations: Sequence[str],
) -> list[str]:
    """Allow only directory scaffolding Docker creates for bind targets.

    Docker materializes absent bind-mount target ancestors in the container's
    writable layer at create time, even for a read-only rootfs. `docker diff`
    therefore is not generally empty. Every reported path must be a target or
    an ancestor of one of the four closed mounts; deletes and duplicates fail.
    Any copied/changed image file (for example /usr/bin/python3) is outside
    this set and is rejected before the container can start.
    """

    allowed: set[str] = set()
    for destination in mount_destinations:
        path = PurePosixPath(destination)
        if not path.is_absolute():
            raise NumericExecutionContractError(
                "rootfs-diff mount destination is not absolute"
            )
        while path != PurePosixPath("/"):
            allowed.add(str(path))
            path = path.parent
    result: list[str] = []
    seen: set[str] = set()
    for change in changes:
        if not isinstance(change, str):
            raise NumericExecutionContractError("Docker rootfs-diff evidence is malformed")
        operation, separator, path_text = change.partition(" ")
        if (
            not separator
            or operation not in {"A", "C"}
            or path_text not in allowed
            or change in seen
        ):
            raise NumericExecutionContractError(
                "created container writable-layer differs beyond bind-mount scaffolding"
            )
        seen.add(change)
        result.append(change)
    return result


def _require_safe_host_config(host_config: Mapping[str, object]) -> None:
    exact = {
        "Privileged": False,
        "CapAdd": None,
        "CapDrop": ["ALL"],
        "GroupAdd": None,
        "Devices": [],
        "PidMode": "",
        "UsernsMode": "",
        "CgroupnsMode": "private",
        "SecurityOpt": ["no-new-privileges:true"],
        "ReadonlyRootfs": True,
        "Runtime": "runc",
        "Tmpfs": CAMPAIGN_TMPFS,
        "Binds": None,
        "VolumesFrom": None,
        "Cgroup": "",
        "CgroupParent": "",
        "DeviceCgroupRules": None,
        "Isolation": "",
        "Init": None,
        "Sysctls": None,
    }
    for field, expected in exact.items():
        if host_config.get(field) != expected:
            raise NumericExecutionContractError(
                f"Docker HostConfig.{field} differs from the isolated campaign launch"
            )
    requests = host_config.get("DeviceRequests")
    if requests != [{
        "Driver": "",
        "Count": -1,
        "DeviceIDs": None,
        "Capabilities": [["gpu"]],
        "Options": {},
    }]:
        raise NumericExecutionContractError(
            "Docker launch must use the exact --gpus all device request"
        )


def _require_launch_command(
    config: Mapping[str, object], repo_root: str,
) -> list[str]:
    if config.get("Entrypoint") != CAMPAIGN_ENTRYPOINT:
        raise NumericExecutionContractError(
            "Docker entrypoint differs from the pinned NVIDIA image entrypoint"
        )
    if config.get("WorkingDir") != repo_root:
        raise NumericExecutionContractError(
            "Docker working directory must be the immutable numeric repository"
        )
    command = _string_list(config.get("Cmd"), "Config.Cmd")
    profiled = command[:2] == ["/usr/local/bin/py-spy", "record"]
    if profiled:
        if len(command) < 11 or command[2:3] != ["--output"] or command[4:7] != [
            "--format", "speedscope", "--",
        ]:
            raise NumericExecutionContractError(
                "Docker py-spy command differs from the pinned profiler contract"
            )
        profile = PurePosixPath(command[3])
        storage = PurePosixPath(CAMPAIGN_STORAGE_ROOT)
        if not profile.is_absolute() or not profile.is_relative_to(storage):
            raise NumericExecutionContractError(
                "Docker py-spy output must live below the campaign storage root"
            )
        python_index = 7
    else:
        python_index = 0
    if (
        len(command) <= python_index + 2
        or command[python_index:python_index + 2] != ["/usr/bin/python3", "-B"]
    ):
        raise NumericExecutionContractError(
            "Docker command must use py-spy or direct /usr/bin/python3 -B"
        )
    driver = PurePosixPath(command[python_index + 2])
    repo = PurePosixPath(repo_root)
    if (
        not driver.is_absolute()
        or not driver.is_relative_to(repo)
        or driver.name not in CAMPAIGN_DRIVER_NAMES
        or driver.parent != repo / "research/trellis_e2m1_highrate_2026-08-30"
    ):
        raise NumericExecutionContractError(
            "Docker command does not name one approved numeric driver"
        )
    if not profiled:
        preflight_flag = {
            "fp8_cb_tcq_glm.py": "--preflight-only",
            "fp8_learned_glm.py": "--dry-run",
        }.get(driver.name)
        if preflight_flag is None or command[-1:] != [preflight_flag]:
            raise NumericExecutionContractError(
                "direct Python launch is permitted only for a nonpublishing preflight"
            )
    return command


def build_launch_attestation(
    inspected: Mapping[str, object], physical_host: str, *, host_output_path: Path,
    rootfs_changes: Sequence[str],
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
    if image_id not in CAMPAIGN_IMAGE_IDS_BY_HOST[physical_host]:
        raise NumericExecutionContractError(
            "Docker inspect image ID is outside the physical host's allowed image-ID set"
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
    _require_safe_host_config(host_config)

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
    _require_safe_environment(environment, live=False)
    if config.get("Healthcheck") is not None:
        raise NumericExecutionContractError(
            "Docker launch must not add a concurrent healthcheck command"
        )
    if config.get("Shell") is not None:
        raise NumericExecutionContractError(
            "Docker launch must retain the pinned null Config.Shell"
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

    parent = str(Path(attestation_path).parent)
    try:
        resolved_output = host_output_path.resolve(strict=False)
        resolved_parent = resolved_output.parent.resolve(strict=True)
    except OSError as exc:
        raise NumericExecutionContractError(
            f"host launch-attestation output parent is unavailable: {exc}"
        ) from exc
    if resolved_output.name in ("", ".", ".."):
        raise NumericExecutionContractError(
            "host launch-attestation output path has no file name"
        )
    storage = Path(CAMPAIGN_STORAGE_ROOT)
    if not resolved_parent.is_relative_to(storage) or resolved_parent == storage:
        raise NumericExecutionContractError(
            "launch-attestation output parent must be scoped below campaign storage"
        )
    if parent != str(resolved_parent) or attestation_path != str(resolved_output):
        raise NumericExecutionContractError(
            "container launch-attestation path must equal the host output path"
        )
    _require_closed_mounts(
        inspected.get("Mounts"),
        attestation_parent=parent,
        host_attestation_parent=str(resolved_parent),
        repo_root=repo_root,
        git_common_dir=git_common_dir,
    )
    validated_rootfs_changes = _require_rootfs_mount_scaffolding(
        rootfs_changes,
        mount_destinations=(
            parent, repo_root, git_common_dir, CAMPAIGN_STORAGE_ROOT,
        ),
    )
    command = _require_launch_command(config, repo_root)

    unsigned: dict[str, object] = {
        "schema": "trellis.numeric_launch_attestation.v1",
        "verification_scope": "host_docker_daemon_inspect_before_start",
        "physical_host": physical_host,
        "uts_hostname": host_spec["uts_hostname"],
        "gpu_uuid": host_spec["gpu_uuid"],
        "container_id": container_id,
        "container_hostname": container_hostname,
        "container_state": "created",
        "container_rootfs_changes": validated_rootfs_changes,
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
        "storage_mount_readwrite": True,
        "rootfs_readonly": True,
        "runtime_isolation": (
            "private_pid_cgroup_ipc;network_none;cap_drop_all;"
            "no_new_privileges;closed_bind_mounts;ephemeral_tmp_scratch"
        ),
        "launch_environment": dict(sorted(environment.items())),
        "launch_command": command,
        "launch_command_sha256": hashlib.sha256(
            _canonical_json(command)
        ).hexdigest(),
    }
    return {**unsigned, "attestation_sha256": _identity_sha256(unsigned)}


def write_launch_attestation(
    container_id: str, physical_host: str, output_path: Path,
) -> dict[str, object]:
    """Inspect once and atomically create a no-clobber host-side record."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_inspection = _docker_inspect(container_id)
    resolved_container_id = first_inspection.get("Id")
    if (
        not isinstance(resolved_container_id, str)
        or _CONTAINER_ID_RE.fullmatch(resolved_container_id) is None
    ):
        raise NumericExecutionContractError(
            "Docker inspect did not resolve a full immutable container ID"
        )
    first_diff = _docker_diff(resolved_container_id)
    record = build_launch_attestation(
        first_inspection, physical_host,
        host_output_path=output_path,
        rootfs_changes=first_diff,
    )
    # Recheck immediately before O_EXCL publication. This closes accidental
    # inspect/write races; a hostile host/daemon remains outside the stated
    # trust boundary and can always mutate state after this helper returns.
    if _docker_inspect(resolved_container_id) != first_inspection:
        raise NumericExecutionContractError(
            "Docker container changed during launch-attestation inspection"
        )
    if _docker_diff(resolved_container_id) != first_diff:
        raise NumericExecutionContractError(
            "Docker container rootfs changed during launch-attestation inspection"
        )
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
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise NumericExecutionContractError(
                "HULL_LAUNCH_ATTESTATION must name a regular non-symlink file"
            )
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read((1 << 20) + 1)
        if len(raw) > (1 << 20):
            raise NumericExecutionContractError(
                "HULL_LAUNCH_ATTESTATION exceeds the closed size limit"
            )
        decoded = _strict_json(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise NumericExecutionContractError(
            f"cannot read strict launch attestation: {exc}"
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != _ATTESTATION_KEYS:
        raise NumericExecutionContractError(
            "launch attestation field set differs from the pinned contract"
        )
    if raw != _canonical_json(decoded) + b"\n":
        raise NumericExecutionContractError(
            "launch attestation bytes are not the canonical JSON encoding"
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
    if record.get("container_image_id") not in CAMPAIGN_IMAGE_IDS_BY_HOST[
        str(physical_host)
    ]:
        raise NumericExecutionContractError(
            "numeric execution image ID is not allowed on this physical host"
        )
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


def validate_numeric_execution_segment(
    segment: object, *, environment: Mapping[str, object],
) -> None:
    """Validate one immutable launch segment against its stable environment.

    Checkpoint identity intentionally uses the stable environment so a crashed
    run can resume in a fresh correct container.  Every partial/final also
    retains the append-only segment list, preserving which inspected
    containers performed or replayed work without making container ID a cache
    key.
    """

    validate_numeric_execution_record(environment)
    if not isinstance(segment, dict) or set(segment) != EXECUTION_SEGMENT_KEYS:
        raise NumericExecutionContractError(
            "numeric execution segment field set differs from the pinned contract"
        )
    if segment.get("schema") != "trellis.numeric_execution_segment.v1":
        raise NumericExecutionContractError("unsupported numeric execution segment")
    host = environment["physical_host"]
    if (
        segment.get("physical_host") != host
        or segment.get("image_id") != environment["container_image_id"]
        or segment.get("gpu_uuid") != environment["gpu_uuid"]
    ):
        raise NumericExecutionContractError(
            "numeric execution segment differs from its stable environment"
        )
    container_id = segment.get("container_id")
    if not isinstance(container_id, str) or _CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise NumericExecutionContractError("numeric execution segment container ID is invalid")
    attestation_path = segment.get("launch_attestation_path")
    storage_path = PurePosixPath(CAMPAIGN_STORAGE_ROOT)
    if (
        not isinstance(attestation_path, str)
        or not PurePosixPath(attestation_path).is_absolute()
        or not PurePosixPath(attestation_path).is_relative_to(storage_path)
        or PurePosixPath(attestation_path).parent == storage_path
        or PurePosixPath(attestation_path).name in {"", ".", ".."}
    ):
        raise NumericExecutionContractError(
            "numeric execution segment attestation path is invalid"
        )
    for field in (
        "launch_attestation_sha256", "launch_command_sha256", "segment_sha256",
    ):
        value = segment.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise NumericExecutionContractError(
                f"numeric execution segment {field} is invalid"
            )
    unsigned = {key: value for key, value in segment.items() if key != "segment_sha256"}
    if segment["segment_sha256"] != _identity_sha256(unsigned):
        raise NumericExecutionContractError("numeric execution segment digest mismatch")


def require_numeric_execution_environment(
    repo_root: Path,
    current_environment: Mapping[str, object],
    process_environment: Mapping[str, str],
    *,
    require_cuda: bool,
    driver_path: Path,
    driver_argv: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
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
    _require_safe_environment(process_environment, live=True)
    if not sys.dont_write_bytecode:
        raise NumericExecutionContractError(
            "live Python did not honor the no-bytecode launch contract"
        )
    if (
        os.getuid() != 1000
        or os.getgid() != 1000
        or os.getgroups() != [1000]
    ):
        raise NumericExecutionContractError(
            "live numeric process must run only as non-root UID/GID/groups 1000:1000:[1000]"
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
        "storage_mount_readwrite": True,
        "rootfs_readonly": True,
        "runtime_isolation": (
            "private_pid_cgroup_ipc;network_none;cap_drop_all;"
            "no_new_privileges;closed_bind_mounts;ephemeral_tmp_scratch"
        ),
    }
    rootfs_changes = attestation.get("container_rootfs_changes")
    if not isinstance(rootfs_changes, list):
        raise NumericExecutionContractError(
            "launch attestation rootfs-diff evidence is malformed"
        )
    exact_attestation["container_rootfs_changes"] = _require_rootfs_mount_scaffolding(
        rootfs_changes,
        mount_destinations=(
            str(Path(attestation_path).parent), declared_repo_root,
            declared_git_common_dir, CAMPAIGN_STORAGE_ROOT,
        ),
    )
    launch_environment = attestation.get("launch_environment")
    if (
        not isinstance(launch_environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in launch_environment.items()
        )
    ):
        raise NumericExecutionContractError(
            "launch attestation environment is malformed"
        )
    _require_safe_environment(launch_environment, live=False)
    for key, value in launch_environment.items():
        actual = process_environment.get(key)
        if key == "NVIDIA_VISIBLE_DEVICES" and value == "all" and actual == "void":
            continue
        if actual != value:
            raise NumericExecutionContractError(
                f"live numeric process environment {key} differs from Docker inspection"
            )
    exact_attestation["launch_environment"] = launch_environment
    command = attestation.get("launch_command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise NumericExecutionContractError("launch attestation command is malformed")
    exact_attestation["launch_command"] = command
    command_sha256 = hashlib.sha256(_canonical_json(command)).hexdigest()
    exact_attestation["launch_command_sha256"] = command_sha256
    for field, expected in exact_attestation.items():
        if attestation.get(field) != expected:
            raise NumericExecutionContractError(
                f"launch attestation {field} does not match the live campaign contract"
            )
    container_id = attestation.get("container_id")
    image_id = attestation.get("image_id")
    if not isinstance(container_id, str) or _CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise NumericExecutionContractError("launch attestation container ID is malformed")
    if image_id not in CAMPAIGN_IMAGE_IDS_BY_HOST[physical_host]:
        raise NumericExecutionContractError(
            "launch attestation image ID is not allowed on this physical host"
        )
    if process_environment.get("HOSTNAME") != container_id[:12]:
        raise NumericExecutionContractError(
            "live container-ID HOSTNAME does not match host-side Docker inspection"
        )
    try:
        live_driver = str(driver_path.resolve(strict=True))
    except OSError as exc:
        raise NumericExecutionContractError(
            f"live numeric driver is unavailable: {exc}"
        ) from exc
    if not driver_argv or not isinstance(driver_argv[0], str):
        raise NumericExecutionContractError("live numeric driver argv is malformed")
    expected_driver_command = [
        "/usr/bin/python3", "-B", live_driver, *list(driver_argv[1:]),
    ]
    python_index = 7 if command[:2] == [
        "/usr/local/bin/py-spy", "record"
    ] else 0
    if command[python_index:] != expected_driver_command:
        raise NumericExecutionContractError(
            "live numeric driver argv differs from the inspected Docker command"
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
    segment_unsigned: dict[str, object] = {
        "schema": "trellis.numeric_execution_segment.v1",
        "physical_host": physical_host,
        "container_id": container_id,
        "image_id": image_id,
        "gpu_uuid": gpu_uuid,
        "launch_attestation_path": attestation_path,
        "launch_attestation_sha256": attestation["attestation_sha256"],
        "launch_command_sha256": command_sha256,
    }
    segment = {
        **segment_unsigned,
        "segment_sha256": _identity_sha256(segment_unsigned),
    }
    validate_numeric_execution_segment(segment, environment=record)
    return record, segment


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
    "CAMPAIGN_IMAGE_IDS_BY_HOST",
    "CAMPAIGN_IMAGE_REFERENCE",
    "CAMPAIGN_STORAGE_ROOT",
    "CAMPAIGN_TMPFS",
    "EXECUTION_RECORD_KEYS",
    "EXECUTION_SEGMENT_KEYS",
    "NumericExecutionContractError",
    "PHYSICAL_HOSTS",
    "build_launch_attestation",
    "require_numeric_execution_environment",
    "require_repo_commit",
    "require_repo_tree_matches_head",
    "validate_numeric_execution_record",
    "validate_numeric_execution_segment",
    "write_launch_attestation",
]
