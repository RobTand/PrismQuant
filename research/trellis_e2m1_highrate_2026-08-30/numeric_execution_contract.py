"""Fail-closed execution identity for the GLM numeric campaign drivers.

The numeric result schemas bind source bytes, but source hashes alone do not
identify the physical host or immutable container that executed them.  This
module is deliberately stdlib-only and requires those facts before a
publication claim can be acquired.  Container runs must use the host UTS
namespace and expose ``git`` read-only so every asserted value is checked
inside the process that publishes the result.
"""
from __future__ import annotations

import re
import socket
import subprocess
from pathlib import Path
from typing import Mapping


class NumericExecutionContractError(RuntimeError):
    """The live process cannot prove its declared numeric execution identity."""


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,253}[A-Za-z0-9])?$")
_ENV_KEYS = frozenset({
    "torch", "python", "host", "device", "triton", "container_image",
})


def _git(
    repo_root: Path, *args: str, check_output: bool = True,
) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NumericExecutionContractError(
            f"cannot attest repository with git: {detail.strip()}"
        ) from exc
    return result.stdout.strip() if check_output else ""


def require_repo_commit(repo_root: Path) -> str:
    """Return a full live HEAD or refuse; ``None`` is never provenance."""

    commit = _git(repo_root, "rev-parse", "HEAD")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise NumericExecutionContractError(
            f"repository HEAD is not a full lowercase commit: {commit!r}"
        )
    return commit


def require_numeric_execution_environment(
    repo_root: Path,
    current_environment: Mapping[str, object],
    process_environment: Mapping[str, str],
    *,
    require_cuda: bool,
) -> dict[str, object]:
    """Validate and return the closed execution record used by a receipt.

    ``HULL_PHYSICAL_HOST`` must equal the live UTS hostname (Docker therefore
    uses ``--uts=host``), and ``HULL_CONTAINER_IMAGE`` must be an immutable
    digest.  The repository must be clean at the exact HEAD recorded here.
    """

    if set(current_environment) != _ENV_KEYS:
        raise NumericExecutionContractError(
            "numeric environment field set differs from the pinned contract"
        )
    physical_host = process_environment.get("HULL_PHYSICAL_HOST", "")
    if _HOST_RE.fullmatch(physical_host) is None:
        raise NumericExecutionContractError(
            "HULL_PHYSICAL_HOST must be an explicit valid physical hostname"
        )
    live_host = socket.gethostname()
    if live_host != physical_host or current_environment.get("host") != physical_host:
        raise NumericExecutionContractError(
            f"live UTS host {live_host!r} does not equal declared physical "
            f"host {physical_host!r}; use the host UTS namespace"
        )

    image = process_environment.get("HULL_CONTAINER_IMAGE", "")
    if _IMAGE_RE.fullmatch(image) is None:
        raise NumericExecutionContractError(
            "HULL_CONTAINER_IMAGE must be an immutable sha256 digest"
        )
    if current_environment.get("container_image") != image:
        raise NumericExecutionContractError(
            "live numeric environment did not retain the declared image digest"
        )

    for field in ("python", "torch", "triton"):
        value = current_environment.get(field)
        if not isinstance(value, str) or not value:
            raise NumericExecutionContractError(
                f"numeric environment {field} identity is unavailable"
            )
    device = current_environment.get("device")
    if require_cuda and (not isinstance(device, str) or not device):
        raise NumericExecutionContractError(
            "CUDA device identity is required for a GPU numeric publication"
        )
    if device is not None and (not isinstance(device, str) or not device):
        raise NumericExecutionContractError("numeric device identity is malformed")

    commit = require_repo_commit(repo_root)
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise NumericExecutionContractError(
            "numeric publication requires a clean repository worktree"
        )

    return {
        "schema": "trellis.numeric_execution.v1",
        "physical_host": physical_host,
        "container_image_digest": image,
        "repo_git_commit": commit,
        "repo_tree_clean": True,
        "python": current_environment["python"],
        "torch": current_environment["torch"],
        "triton": current_environment["triton"],
        "device": device,
    }


__all__ = [
    "NumericExecutionContractError",
    "require_numeric_execution_environment",
    "require_repo_commit",
]
