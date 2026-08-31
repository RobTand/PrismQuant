"""Deterministic PrismaBuild action keys, local execution, and immutable CAS.

This is the dependency-free core beneath a future Dagster/SLURM deployment.
It deliberately does not contain a queue or a weight cache.  An action is an
exact, self-hashed contract; a local worker executes its argv with
``shell=False`` and a closed environment; and successful file results are
published to a content-addressed store with NFS-safe, first-writer-wins hard
links.

Portable actions omit machine identity from their key.  Platform- and
host-class-keyed actions bind the corresponding explicit execution scope.
Measurement actions are never portable.  FP8-CB generation is also never
portable because D29 records cross-architecture row-scale byte drift.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import socket
import stat
import subprocess
import tempfile

ACTION_SCHEMA_V1 = "prismaquant.prismabuild.action.v1"
CODE_CLOSURE_SCHEMA_V1 = "prismaquant.prismabuild.code_closure.v1"
CAS_RECEIPT_SCHEMA_V3 = "prismaquant.prismabuild.cas_receipt.v3"
WORKER_ATTESTATION_SCHEMA_V2 = "prismaquant.prismabuild.worker_attestation.v2"
WORKER_RUNTIME_SCHEMA_V1 = "prismaquant.prismabuild.worker_runtime.v1"
_CAS_RECEIPT_NAMESPACE = CAS_RECEIPT_SCHEMA_V3.rsplit(".", 1)[-1]

_ACTION_BODY_KEYS = frozenset(
    {
        "schema",
        "task",
        "inputs",
        "code_closure",
        "params",
        "environment",
        "execution_scope",
    }
)
_ACTION_KEYS = _ACTION_BODY_KEYS | {"action_key"}
_TASK_KEYS = frozenset(
    {
        "definition_id",
        "definition_version",
        "task_class",
        "determinism",
        "artifact_kind",
        "argv",
        "working_directory",
        "result_path",
    }
)
_INPUT_KEYS = frozenset({"id", "sha256", "bytes"})
_CLOSURE_KEYS = frozenset({"schema", "files", "closure_sha256"})
_CLOSURE_FILE_KEYS = frozenset({"path", "sha256", "bytes"})
_ENVIRONMENT_KEYS = frozenset({"variables", "toolchain"})
_SCOPE_KEYS = frozenset({"portability", "platform_key", "host_class"})
_RECEIPT_BODY_KEYS = frozenset(
    {"schema", "action_key", "action_manifest_sha256", "result", "producer"}
)
_RECEIPT_KEYS = _RECEIPT_BODY_KEYS | {"receipt_sha256"}
_RESULT_KEYS = frozenset({"sha256", "bytes"})
_PRODUCER_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "worker_id",
        "platform_key",
        "host_class",
        "evidence",
        "runtime",
        "executable",
        "toolchain",
        "inputs",
        "attestation_sha256",
    }
)
_EVIDENCE_KEYS = frozenset(
    {"source", "hostname", "system", "machine", "libc", "accelerators", "slurm"}
)
_ACCELERATOR_KEYS = frozenset(
    {"kind", "compute_capability", "driver_version"}
)
_SLURM_EVIDENCE_KEYS = frozenset(
    {"job_id", "node_name", "partition", "constraints", "cgroup"}
)
_EXECUTABLE_KEYS = frozenset({"path", "resolved_path", "sha256", "bytes"})
_RUNTIME_KEYS = frozenset(
    {"schema", "launch_kind", "core", "launcher", "runtime_sha256"}
)
_TOOLCHAIN_ATTESTATION_KEYS = frozenset({"declared", "verified"})

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,255}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}\Z")
_SCOPE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/-]{0,255}\Z")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_PORTABILITY = frozenset({"portable", "platform_keyed", "host_class_keyed"})
_TASK_CLASSES = frozenset({"generation", "measurement"})
_DETERMINISM = frozenset({"deterministic", "stochastic"})
_PROCESS_GROUP_GRACE_SECONDS = 5.0
_ATTESTABLE_TOOLCHAIN_KEYS = frozenset(
    {
        "argv0.sha256",
        "argv0.bytes",
        "python",
        "torch",
        "transformers",
        "vllm",
        "gridbook",
        "system",
        "machine",
        "libc",
        "cuda_compute_capability",
        "nvidia_driver",
    }
)
_PYTHON_DISTRIBUTIONS = frozenset({"torch", "transformers", "vllm", "gridbook"})


class PrismaBuildError(RuntimeError):
    """Base class for PrismaBuild core failures."""


class ActionContractError(PrismaBuildError, ValueError):
    """An action, closure, receipt, or worker identity is not exact."""


class CASTamperError(PrismaBuildError):
    """An existing content-addressed entry failed verification."""


class CASUnavailableError(PrismaBuildError):
    """The content-addressed store could not be read reliably."""


class CASConflictError(PrismaBuildError):
    """A deterministic recomputation disagreed with the canonical result."""


class LocalActionError(PrismaBuildError):
    """A local action could not execute or did not produce its declared file."""


def _fail(message: str) -> None:
    raise ActionContractError(message)


def _exact_mapping(
    value: object, *, keys: frozenset[str], where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        _fail(f"{where} keys must be strings")
    actual = set(value)
    if actual != set(keys):
        _fail(
            f"{where} fields differ: missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )
    return value


def _text(
    value: object,
    *,
    where: str,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or (not value and not allow_empty):
        _fail(f"{where} must be a {'string' if allow_empty else 'non-empty string'}")
    if "\x00" in value or any(ord(char) < 32 for char in value):
        _fail(f"{where} contains a NUL or control character")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(f"{where} has an invalid value")
    return value


def _optional_token(value: object, *, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where=where, pattern=_SCOPE_TOKEN_RE)


def _nonnegative_integer(value: object, *, where: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{where} must be a non-negative integer")
    return value


def _sha256(value: object, *, where: str) -> str:
    return _text(value, where=where, pattern=_SHA256_RE)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActionContractError("value is not finite canonical JSON data") from exc


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON without importing another repository module."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_relative_path(value: object, *, where: str, dot_ok: bool) -> str:
    raw = _text(value, where=where)
    if "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        _fail(f"{where} must be a normalized relative POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".."} for part in raw.split("/")):
        _fail(f"{where} must be a normalized relative POSIX path")
    if raw == ".":
        if dot_ok:
            return raw
        _fail(f"{where} must name a file, not '.'")
    if any(part == "." for part in raw.split("/")) or str(path) != raw:
        _fail(f"{where} must be a normalized relative POSIX path")
    return raw


def _normalize_json_value(value: object, *, where: str) -> object:
    """Validate JSON recursively without Python's silent key coercions."""

    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{where} contains a non-finite number")
        return value
    if type(value) is str:
        return _text(value, where=where, allow_empty=True)
    if type(value) is list:
        return [
            _normalize_json_value(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _text(raw_key, where=f"{where} key", allow_empty=True)
            normalized[key] = _normalize_json_value(
                raw_value, where=f"{where}.{key}"
            )
        return normalized
    _fail(f"{where} contains a value that is not JSON data")


def _is_codebook_artifact_kind(value: str) -> bool:
    """Recognize codebook spelling variants conservatively for D29."""

    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    has_quant_family = "fp8" in compact or "nvfp4" in compact
    has_codebook = "codebook" in compact or "cb" in compact
    return has_quant_family and has_codebook


def _normalize_argv(value: object) -> list[str]:
    if type(value) is not list or not value:
        _fail("action.task.argv must be a non-empty array")
    argv = [
        _text(item, where=f"action.task.argv[{index}]", allow_empty=True)
        for index, item in enumerate(value)
    ]
    if not PurePosixPath(argv[0]).is_absolute():
        _fail("action.task.argv[0] must be an absolute executable path")
    return argv


def _normalize_string_mapping(
    value: object,
    *,
    where: str,
    key_pattern: re.Pattern[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, where=f"{where} key", pattern=key_pattern)
        normalized[key] = _text(
            raw_value, where=f"{where}.{key}", allow_empty=True
        )
    return dict(sorted(normalized.items()))


def _decode_strict_json(raw: bytes, *, where: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ActionContractError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ActionContractError(f"{where} contains non-finite number {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except ActionContractError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        # ``json.loads`` can raise a plain ValueError when an integer exceeds
        # Python's configured digit limit.  Durable hostile input must stay in
        # the fail-closed PrismaBuild error vocabulary instead of escaping as
        # an implementation-specific parser exception.
        raise ActionContractError(f"{where} is not strict UTF-8 JSON") from exc


def _read_regular_file(path: Path, *, where: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ActionContractError(f"cannot open {where} as a regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ActionContractError(f"{where} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ActionContractError(f"{where} changed while it was read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_identity(path: Path, *, where: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ActionContractError(
            f"cannot open {where} as a regular file: {path}"
        ) from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ActionContractError(f"{where} is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ActionContractError(f"{where} changed while it was read: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def identify_executable(path: str | Path) -> dict[str, object]:
    """Return the exact regular-file identity behind an absolute argv[0]."""

    declared = Path(path)
    if not declared.is_absolute():
        _fail("executable path must be absolute")
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise ActionContractError(
            f"cannot resolve executable path: {declared}"
        ) from exc
    digest, size = _file_identity(resolved, where="action executable")
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ActionContractError(
            f"cannot inspect executable path: {resolved}"
        ) from exc
    if mode & 0o111 == 0:
        _fail(f"action executable is not executable: {resolved}")
    return {
        "path": str(declared),
        "resolved_path": str(resolved),
        "sha256": digest,
        "bytes": size,
    }


def _identify_runtime_source(path: str | Path, *, where: str) -> dict[str, object]:
    """Return the exact regular-file identity of worker implementation source."""

    declared = Path(path)
    if not declared.is_absolute():
        _fail(f"{where} path must be absolute")
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise ActionContractError(f"cannot resolve {where}: {declared}") from exc
    digest, size = _file_identity(resolved, where=where)
    return {
        "path": str(declared),
        "resolved_path": str(resolved),
        "sha256": digest,
        "bytes": size,
    }


# This snapshot is part of module initialization, not worker preflight.  It
# therefore precedes action loading and execution and represents the source
# implementation this worker process loaded as closely as Python exposes it.
_LOADED_WORKER_CORE_IDENTITY = _identify_runtime_source(
    Path(__file__).resolve(), where="PrismaBuild worker core"
)


def _normalize_runtime_source(value: object, *, where: str) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_EXECUTABLE_KEYS, where=where)
    path = _text(raw["path"], where=f"{where}.path")
    resolved_path = _text(
        raw["resolved_path"], where=f"{where}.resolved_path"
    )
    if not Path(path).is_absolute() or not Path(resolved_path).is_absolute():
        _fail(f"{where} paths must be absolute")
    return {
        "path": path,
        "resolved_path": resolved_path,
        "sha256": _sha256(raw["sha256"], where=f"{where}.sha256"),
        "bytes": _nonnegative_integer(raw["bytes"], where=f"{where}.bytes"),
    }


def _verify_runtime_source_unchanged(
    expected: Mapping[str, object], *, where: str, changed_message: str
) -> None:
    try:
        observed = _identify_runtime_source(str(expected["path"]), where=where)
    except ActionContractError as exc:
        raise LocalActionError(f"{changed_message}: {exc}") from exc
    if observed != expected:
        raise LocalActionError(changed_message)


def _worker_runtime_identity(
    worker_launcher_identity: object | None,
) -> dict[str, object]:
    """Bind the load-time core and optional entry-point launcher snapshot."""

    core = _normalize_runtime_source(
        _LOADED_WORKER_CORE_IDENTITY,
        where="loaded PrismaBuild worker core",
    )
    _verify_runtime_source_unchanged(
        core,
        where="PrismaBuild worker core",
        changed_message="PrismaBuild worker core changed after module import",
    )
    launcher = (
        None
        if worker_launcher_identity is None
        else _normalize_runtime_source(
            worker_launcher_identity,
            where="captured PrismaBuild worker launcher",
        )
    )
    if launcher is not None:
        _verify_runtime_source_unchanged(
            launcher,
            where="PrismaBuild worker launcher",
            changed_message=(
                "PrismaBuild worker launcher changed after entry-point capture"
            ),
        )
    body: dict[str, object] = {
        "schema": WORKER_RUNTIME_SCHEMA_V1,
        "launch_kind": "in_process" if launcher is None else "script",
        "core": core,
        "launcher": launcher,
    }
    return {**body, "runtime_sha256": canonical_sha256(body)}


def _validate_worker_runtime(value: object) -> dict[str, object]:
    raw = _exact_mapping(
        value, keys=_RUNTIME_KEYS, where="worker attestation.runtime"
    )
    if raw["schema"] != WORKER_RUNTIME_SCHEMA_V1:
        _fail(
            "worker attestation.runtime.schema must be "
            f"{WORKER_RUNTIME_SCHEMA_V1!r}"
        )
    launch_kind = _text(
        raw["launch_kind"], where="worker attestation.runtime.launch_kind"
    )
    if launch_kind not in {"in_process", "script"}:
        _fail("worker attestation.runtime.launch_kind is unsupported")
    core = _normalize_runtime_source(
        raw["core"], where="worker attestation.runtime.core"
    )
    launcher_raw = raw["launcher"]
    launcher = (
        None
        if launcher_raw is None
        else _normalize_runtime_source(
            launcher_raw, where="worker attestation.runtime.launcher"
        )
    )
    if (launch_kind == "in_process") != (launcher is None):
        _fail("worker attestation runtime launch_kind and launcher disagree")
    body: dict[str, object] = {
        "schema": WORKER_RUNTIME_SCHEMA_V1,
        "launch_kind": launch_kind,
        "core": core,
        "launcher": launcher,
    }
    recorded = _sha256(
        raw["runtime_sha256"],
        where="worker attestation.runtime.runtime_sha256",
    )
    if recorded != canonical_sha256(body):
        _fail("worker attestation runtime digest does not match its body")
    return {**body, "runtime_sha256": recorded}


def _verify_worker_runtime_unchanged(runtime: object) -> None:
    expected = _validate_worker_runtime(runtime)
    core = expected["core"]
    assert isinstance(core, Mapping)
    loaded_core = _normalize_runtime_source(
        _LOADED_WORKER_CORE_IDENTITY,
        where="loaded PrismaBuild worker core",
    )
    if core != loaded_core:
        raise LocalActionError(
            "attested PrismaBuild worker core differs from module import identity"
        )
    _verify_runtime_source_unchanged(
        core,
        where="PrismaBuild worker core",
        changed_message="PrismaBuild worker core changed after module import",
    )
    launcher = expected["launcher"]
    if isinstance(launcher, Mapping):
        _verify_runtime_source_unchanged(
            launcher,
            where="PrismaBuild worker launcher",
            changed_message=(
                "PrismaBuild worker launcher changed after entry-point capture"
            ),
        )


def executable_toolchain_contract(path: str | Path) -> dict[str, str]:
    """Build the action-key fields that bind argv[0] to exact file bytes."""

    identity = identify_executable(path)
    return {
        "argv0.sha256": str(identity["sha256"]),
        "argv0.bytes": str(identity["bytes"]),
    }


def _probe_nvidia_accelerators() -> list[dict[str, str]]:
    """Read live NVIDIA compute/driver facts without importing the task stack."""

    executable = Path("/usr/bin/nvidia-smi")
    if not executable.is_file():
        return []
    argv = [
        str(executable),
        "--query-gpu=compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env={"LANG": "C", "LC_ALL": "C"},
            timeout=10.0,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    facts: set[tuple[str, str]] = set()
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 2:
            raise ActionContractError("nvidia-smi returned malformed accelerator facts")
        capability, driver = fields
        if re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None:
            raise ActionContractError("nvidia-smi returned an invalid compute capability")
        if _VERSION_RE.fullmatch(driver) is None:
            raise ActionContractError("nvidia-smi returned an invalid driver version")
        facts.add((capability, driver))
    return [
        {
            "kind": "nvidia",
            "compute_capability": capability,
            "driver_version": driver,
        }
        for capability, driver in sorted(facts)
    ]


def _constraint_tokens(value: str) -> list[str]:
    return sorted(
        {
            token
            for token in re.split(r"[^A-Za-z0-9._+:/-]+", value)
            if token
        }
    )


def _verify_slurm_process_membership(job_id: str) -> str:
    """Bind SLURM environment claims to this process's kernel-owned cgroup."""

    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ActionContractError("SLURM_JOB_ID must be a positive numeric job id")
    raw = _read_regular_file(Path("/proc/self/cgroup"), where="worker cgroup")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ActionContractError("worker cgroup is not UTF-8") from exc
    pattern = re.compile(rf"(?:^|/)job_{re.escape(job_id)}(?:[./]|$)")
    matches: set[str] = set()
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and pattern.search(fields[2]):
            matches.add(fields[2])
    if len(matches) != 1:
        raise ActionContractError(
            "SLURM environment is not attested by this process's cgroup membership"
        )
    return _text(next(iter(matches)), where="worker SLURM cgroup")


def _collect_worker_evidence(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Collect facts from the live worker and SLURM-owned job environment."""

    env = os.environ if environment is None else environment
    system = platform.system().lower()
    machine = platform.machine().lower()
    libc_name, libc_version = platform.libc_ver()
    libc = f"{libc_name.lower()}-{libc_version}" if libc_name else "unknown"
    hostname = socket.gethostname().lower()
    slurm_values = {
        "job_id": env.get("SLURM_JOB_ID"),
        "node_name": env.get("SLURMD_NODENAME"),
        "partition": env.get("SLURM_JOB_PARTITION"),
    }
    present = [value is not None for value in slurm_values.values()]
    if any(present) and not all(present):
        raise ActionContractError(
            "partial SLURM worker evidence is ambiguous; job, node, and partition are required"
        )
    slurm: dict[str, object] | None = None
    source = "local"
    if all(present):
        source = "slurm"
        job_id = _text(
            slurm_values["job_id"], where="SLURM_JOB_ID", pattern=_SCOPE_TOKEN_RE
        )
        node_name = str(slurm_values["node_name"]).lower()
        partition = str(slurm_values["partition"])
        slurm = {
            "job_id": job_id,
            "node_name": _text(
                node_name, where="SLURMD_NODENAME", pattern=_ID_RE
            ),
            "partition": _text(
                partition, where="SLURM_JOB_PARTITION", pattern=_SCOPE_TOKEN_RE
            ),
            "constraints": _constraint_tokens(env.get("SLURM_JOB_CONSTRAINTS", "")),
            "cgroup": _verify_slurm_process_membership(job_id),
        }
    return {
        "source": source,
        "hostname": _text(hostname, where="worker hostname", pattern=_ID_RE),
        "system": _text(system, where="worker system", pattern=_SCOPE_TOKEN_RE),
        "machine": _text(machine, where="worker machine", pattern=_SCOPE_TOKEN_RE),
        "libc": _text(libc, where="worker libc", pattern=_SCOPE_TOKEN_RE),
        "accelerators": _probe_nvidia_accelerators(),
        "slurm": slurm,
    }


def _platform_key_from_evidence(evidence: Mapping[str, object]) -> str:
    system = str(evidence["system"])
    machine = str(evidence["machine"])
    accelerators = evidence["accelerators"]
    assert isinstance(accelerators, list)
    capabilities = {
        str(accelerator["compute_capability"])
        for accelerator in accelerators
        if isinstance(accelerator, Mapping) and accelerator.get("kind") == "nvidia"
    }
    if len(capabilities) > 1:
        raise ActionContractError(
            "worker exposes heterogeneous NVIDIA compute capabilities; platform is ambiguous"
        )
    suffix = ""
    if capabilities:
        capability = next(iter(capabilities))
        suffix = f"-sm{capability.replace('.', '')}"
    return _text(
        f"{system}-{machine}{suffix}",
        where="derived worker platform_key",
        pattern=_SCOPE_TOKEN_RE,
    )


def _worker_identity_from_evidence(evidence: Mapping[str, object]) -> str:
    slurm = evidence["slurm"]
    if isinstance(slurm, Mapping):
        return str(slurm["node_name"])
    return str(evidence["hostname"])


def _host_class_from_evidence(
    evidence: Mapping[str, object], *, expected: str | None
) -> str | None:
    # Partition and constraint are relevant only to a host-class-keyed action.
    # Do not turn scheduler metadata into an ambient host-class assertion for
    # portable or platform-keyed work.
    if expected is None:
        return None
    slurm = evidence["slurm"]
    if not isinstance(slurm, Mapping):
        if expected is not None:
            raise ActionContractError(
                "host_class_keyed actions require complete SLURM job evidence"
            )
        return None
    partition = str(slurm["partition"])
    constraints = slurm["constraints"]
    assert isinstance(constraints, list)
    if expected != partition and expected not in constraints:
        raise ActionContractError(
            "SLURM partition/constraints do not attest the action host_class"
        )
    return expected


def _probe_python_toolchain(executable: Path) -> dict[str, str]:
    script = "\n".join(
        (
            "import importlib.metadata as metadata",
            "import json",
            "import platform",
            "out = {'python': platform.python_version()}",
            "for name in ('torch', 'transformers', 'vllm', 'gridbook'):",
            "    try:",
            "        out[name] = metadata.version(name)",
            "    except metadata.PackageNotFoundError:",
            "        pass",
            "print(json.dumps(out, sort_keys=True, separators=(',', ':')))",
        )
    )
    try:
        completed = subprocess.run(
            [str(executable), "-I", "-c", script],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            timeout=30.0,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise ActionContractError(
            f"cannot probe declared Python toolchain through {executable}"
        ) from exc
    if completed.returncode != 0 or "\n" in completed.stdout.strip():
        raise ActionContractError(
            f"declared Python toolchain probe failed through {executable}"
        )
    value = _decode_strict_json(
        completed.stdout.strip().encode("utf-8"), where="Python toolchain probe"
    )
    if not isinstance(value, Mapping) or any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise ActionContractError("Python toolchain probe returned malformed data")
    return dict(value)


def _toolchain_value_matches(declared: str, observed: str) -> bool:
    if declared == observed:
        return True
    short = re.fullmatch(r"([0-9]+\.[0-9]+)(?:\+([A-Za-z0-9._-]+))?", declared)
    full = re.fullmatch(
        r"([0-9]+\.[0-9]+)(?:\.[0-9]+)?(?:\+([A-Za-z0-9._-]+))?",
        observed,
    )
    return bool(
        short
        and full
        and short.group(1) == full.group(1)
        and short.group(2) == full.group(2)
    )


def build_code_closure(root: str | Path, files: Sequence[str]) -> dict[str, object]:
    """Hash a declared, path-independent code closure below ``root``.

    Only declared regular files participate.  Symlinks and traversal are
    refused, and the logical relative paths (not the checkout root) are bound
    into the closure identity.
    """

    checkout_root = Path(root)
    if not checkout_root.is_absolute():
        _fail("code closure root must be absolute")
    normalized_paths = [
        _normalize_relative_path(path, where=f"code closure file[{index}]", dot_ok=False)
        for index, path in enumerate(files)
    ]
    if not normalized_paths:
        _fail("code closure must contain at least one file")
    if len(set(normalized_paths)) != len(normalized_paths):
        _fail("code closure file paths must be unique")
    entries: list[dict[str, object]] = []
    for relative in sorted(normalized_paths):
        digest, size = _file_identity(
            checkout_root / relative, where=f"code closure file {relative!r}"
        )
        entries.append({"path": relative, "sha256": digest, "bytes": size})
    body: dict[str, object] = {
        "schema": CODE_CLOSURE_SCHEMA_V1,
        "files": entries,
    }
    return {**body, "closure_sha256": canonical_sha256(body)}


def validate_code_closure(value: object) -> dict[str, object]:
    closure = _exact_mapping(value, keys=_CLOSURE_KEYS, where="action.code_closure")
    if closure["schema"] != CODE_CLOSURE_SCHEMA_V1:
        _fail(f"action.code_closure.schema must be {CODE_CLOSURE_SCHEMA_V1!r}")
    raw_files = closure["files"]
    if type(raw_files) is not list or not raw_files:
        _fail("action.code_closure.files must be a non-empty array")
    files: list[dict[str, object]] = []
    previous: str | None = None
    for index, raw_entry in enumerate(raw_files):
        entry = _exact_mapping(
            raw_entry,
            keys=_CLOSURE_FILE_KEYS,
            where=f"action.code_closure.files[{index}]",
        )
        path = _normalize_relative_path(
            entry["path"],
            where=f"action.code_closure.files[{index}].path",
            dot_ok=False,
        )
        if previous is not None and path <= previous:
            _fail("action.code_closure.files must be unique and sorted by path")
        previous = path
        files.append(
            {
                "path": path,
                "sha256": _sha256(
                    entry["sha256"],
                    where=f"action.code_closure.files[{index}].sha256",
                ),
                "bytes": _nonnegative_integer(
                    entry["bytes"],
                    where=f"action.code_closure.files[{index}].bytes",
                ),
            }
        )
    body = {"schema": CODE_CLOSURE_SCHEMA_V1, "files": files}
    expected = canonical_sha256(body)
    recorded = _sha256(
        closure["closure_sha256"], where="action.code_closure.closure_sha256"
    )
    if recorded != expected:
        _fail("action.code_closure.closure_sha256 does not match its files")
    return {**body, "closure_sha256": recorded}


def verify_code_closure(value: object, root: str | Path) -> dict[str, object]:
    """Re-hash every closure member from a live checkout."""

    expected = validate_code_closure(value)
    live = build_code_closure(
        root, [str(entry["path"]) for entry in expected["files"]]  # type: ignore[index]
    )
    if live != expected:
        raise ActionContractError(
            "live code closure differs from the action-pinned closure"
        )
    return expected


def _normalize_inputs(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        _fail("action.inputs must be an array")
    inputs: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(value):
        entry = _exact_mapping(
            raw_entry, keys=_INPUT_KEYS, where=f"action.inputs[{index}]"
        )
        identity = _text(
            entry["id"], where=f"action.inputs[{index}].id", pattern=_ID_RE
        )
        if identity in seen:
            _fail("action.inputs ids must be unique")
        seen.add(identity)
        inputs.append(
            {
                "id": identity,
                "sha256": _sha256(
                    entry["sha256"], where=f"action.inputs[{index}].sha256"
                ),
                "bytes": _nonnegative_integer(
                    entry["bytes"], where=f"action.inputs[{index}].bytes"
                ),
            }
        )
    return sorted(inputs, key=lambda entry: str(entry["id"]))


def validate_input_contract(value: object) -> dict[str, object]:
    """Normalize one content-addressed input row used by an action.

    The returned object is exactly the ``{id, sha256, bytes}`` shape accepted
    in ``action.inputs``.  Keeping this validation public lets input ingestion
    and later lookup share the action schema instead of inventing a second
    descriptor vocabulary.
    """

    return _normalize_inputs([value])[0]


def _normalize_task(value: object) -> dict[str, object]:
    task = _exact_mapping(value, keys=_TASK_KEYS, where="action.task")
    task_class = _text(task["task_class"], where="action.task.task_class")
    if task_class not in _TASK_CLASSES:
        _fail(f"action.task.task_class must be one of {sorted(_TASK_CLASSES)}")
    determinism = _text(task["determinism"], where="action.task.determinism")
    if determinism not in _DETERMINISM:
        _fail(f"action.task.determinism must be one of {sorted(_DETERMINISM)}")
    return {
        "definition_id": _text(
            task["definition_id"],
            where="action.task.definition_id",
            pattern=_ID_RE,
        ),
        "definition_version": _text(
            task["definition_version"],
            where="action.task.definition_version",
            pattern=_VERSION_RE,
        ),
        "task_class": task_class,
        "determinism": determinism,
        "artifact_kind": _text(
            task["artifact_kind"],
            where="action.task.artifact_kind",
            pattern=_ID_RE,
        ),
        "argv": _normalize_argv(task["argv"]),
        "working_directory": _normalize_relative_path(
            task["working_directory"],
            where="action.task.working_directory",
            dot_ok=True,
        ),
        "result_path": _normalize_relative_path(
            task["result_path"], where="action.task.result_path", dot_ok=False
        ),
    }


def _normalize_environment(value: object) -> dict[str, object]:
    environment = _exact_mapping(
        value, keys=_ENVIRONMENT_KEYS, where="action.environment"
    )
    return {
        "variables": _normalize_string_mapping(
            environment["variables"],
            where="action.environment.variables",
            key_pattern=_ENV_RE,
        ),
        "toolchain": _normalize_string_mapping(
            environment["toolchain"], where="action.environment.toolchain"
        ),
    }


def _normalize_scope(value: object) -> dict[str, object]:
    scope = _exact_mapping(value, keys=_SCOPE_KEYS, where="action.execution_scope")
    portability = _text(
        scope["portability"], where="action.execution_scope.portability"
    )
    if portability not in _PORTABILITY:
        _fail(
            "action.execution_scope.portability must be one of "
            f"{sorted(_PORTABILITY)}"
        )
    platform_key = _optional_token(
        scope["platform_key"], where="action.execution_scope.platform_key"
    )
    host_class = _optional_token(
        scope["host_class"], where="action.execution_scope.host_class"
    )
    if portability == "portable" and (platform_key is not None or host_class is not None):
        _fail("portable actions must set platform_key and host_class to null")
    if portability == "platform_keyed" and (
        platform_key is None or host_class is not None
    ):
        _fail("platform_keyed actions require platform_key and null host_class")
    if portability == "host_class_keyed" and (
        host_class is None or platform_key is not None
    ):
        _fail("host_class_keyed actions require host_class and null platform_key")
    return {
        "portability": portability,
        "platform_key": platform_key,
        "host_class": host_class,
    }


def _normalize_action_body(value: object) -> dict[str, object]:
    body = _exact_mapping(value, keys=_ACTION_BODY_KEYS, where="action body")
    if body["schema"] != ACTION_SCHEMA_V1:
        _fail(f"action.schema must be {ACTION_SCHEMA_V1!r}")
    task = _normalize_task(body["task"])
    scope = _normalize_scope(body["execution_scope"])
    if task["task_class"] == "measurement" and scope["portability"] == "portable":
        _fail("measurement actions must be platform_keyed or host_class_keyed")
    if (
        _is_codebook_artifact_kind(str(task["artifact_kind"]))
        and scope["portability"] == "portable"
    ):
        _fail(
            "FP8-CB actions cannot be portable: D29 records cross-architecture "
            "row-scale byte drift"
        )
    params = body["params"]
    if not isinstance(params, Mapping):
        _fail("action.params must be an object with string keys")
    normalized_params = _normalize_json_value(params, where="action.params")
    assert isinstance(normalized_params, Mapping)
    # Detach the sealed value from caller-owned mutable containers and replay
    # the strict decoder used for persisted manifests.
    normalized_params = _decode_strict_json(
        _canonical_bytes(normalized_params), where="action.params"
    )
    environment = _normalize_environment(body["environment"])
    toolchain = environment["toolchain"]
    assert isinstance(toolchain, Mapping)
    if "argv0.sha256" in toolchain:
        _sha256(
            toolchain["argv0.sha256"],
            where="action.environment.toolchain.argv0.sha256",
        )
    if "argv0.bytes" in toolchain:
        raw_bytes = toolchain["argv0.bytes"]
        if (
            re.fullmatch(r"0|[1-9][0-9]*", str(raw_bytes)) is None
            or str(int(str(raw_bytes))) != raw_bytes
        ):
            _fail(
                "action.environment.toolchain.argv0.bytes must be a canonical "
                "integer string"
            )
    if "cuda_compute_capability" in toolchain and re.fullmatch(
        r"[0-9]+\.[0-9]+", str(toolchain["cuda_compute_capability"])
    ) is None:
        _fail("action.environment.toolchain.cuda_compute_capability is malformed")
    if scope["portability"] != "portable":
        unknown = set(toolchain) - _ATTESTABLE_TOOLCHAIN_KEYS
        if unknown:
            _fail(
                "nonportable action toolchain contains fields with no worker "
                f"preflight: {sorted(unknown)}"
            )
        required = {
            "argv0.sha256",
            "argv0.bytes",
            "system",
            "machine",
            "libc",
        }
        if not required <= set(toolchain):
            _fail(
                "nonportable actions must bind argv[0] and platform ABI with "
                "toolchain fields argv0.sha256, argv0.bytes, system, machine, "
                "and libc"
            )
    return {
        "schema": ACTION_SCHEMA_V1,
        "task": task,
        "inputs": _normalize_inputs(body["inputs"]),
        "code_closure": validate_code_closure(body["code_closure"]),
        "params": normalized_params,
        "environment": environment,
        "execution_scope": scope,
    }


def seal_action(value: object) -> dict[str, object]:
    """Normalize an action body and attach its action key."""

    body = _normalize_action_body(value)
    return {**body, "action_key": canonical_sha256(body)}


def validate_action(value: object) -> dict[str, object]:
    action = _exact_mapping(value, keys=_ACTION_KEYS, where="action")
    raw_body = {key: action[key] for key in _ACTION_BODY_KEYS}
    body = _normalize_action_body(raw_body)
    if raw_body != body:
        _fail("action body is valid but not in normalized contract form")
    recorded = _sha256(action["action_key"], where="action.action_key")
    expected = canonical_sha256(body)
    if recorded != expected:
        _fail("action.action_key does not match the canonical action body")
    return {**body, "action_key": recorded}


def validate_worker_scope(
    action: object,
    *,
    attestation: object,
) -> None:
    """Validate scope from live-derived, self-hashed worker evidence.

    Scheduler placement strings are intentionally not accepted here: they are
    intent, not evidence that the allocated machine has the requested scope.
    """

    validate_worker_attestation(attestation, action=action)


def _validate_scope_labels(
    action: Mapping[str, object], *, platform_key: str | None, host_class: str | None
) -> None:
    actual_platform = _optional_token(platform_key, where="worker platform_key")
    actual_host = _optional_token(host_class, where="worker host_class")
    normalized = validate_action(action)
    scope = normalized["execution_scope"]
    assert isinstance(scope, Mapping)
    if (
        scope["portability"] == "platform_keyed"
        and actual_platform != scope["platform_key"]
    ):
        raise ActionContractError(
            "worker platform_key does not match the action execution scope"
        )
    if (
        scope["portability"] == "host_class_keyed"
        and actual_host != scope["host_class"]
    ):
        raise ActionContractError(
            "worker host_class does not match the action execution scope"
        )


def _atomic_publish(
    path: Path,
    raw: bytes,
    *,
    prelink_verify: Callable[[], None] | None = None,
) -> bool:
    """Publish immutable bytes relative to a held no-follow parent FD."""

    path = _absolute_nofollow_path(path, where="publication path")
    directory_fd = _open_directory_nofollow(
        path.parent, where="publication directory", create=True
    )
    temporary_name: str | None = None
    try:
        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=f"/proc/self/fd/{directory_fd}",
        )
        temporary_name = Path(temporary_raw).name
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
            temporary_identity = os.fstat(handle.fileno())
        if prelink_verify is not None:
            prelink_verify()
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            won = True
        except FileExistsError:
            won = False
        os.fsync(directory_fd)
        if won:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            try:
                published_fd = os.open(path.name, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise CASTamperError(
                    f"published file changed before readback: {path}"
                ) from exc
            try:
                published_identity = os.fstat(published_fd)
                if (
                    temporary_identity.st_dev,
                    temporary_identity.st_ino,
                ) != (
                    published_identity.st_dev,
                    published_identity.st_ino,
                ):
                    raise CASTamperError(
                        f"published file changed before readback: {path}"
                    )
            finally:
                os.close(published_fd)
        _assert_directory_identity(
            directory_fd, path.parent, where="publication directory"
        )
        return won
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _absolute_nofollow_path(path: Path, *, where: str) -> Path:
    """Return an absolute lexical path that cannot walk through ``..``."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ActionContractError(f"{where} must not contain parent traversal")
    return candidate


def _open_directory_nofollow(
    path: Path,
    *,
    where: str,
    create: bool = False,
    mode: int = 0o755,
) -> int:
    """Open/create an absolute directory using only mkdirat/openat operations."""

    path = _absolute_nofollow_path(path, where=where)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise CASUnavailableError(f"cannot open {where} filesystem root: {exc}") from exc
    try:
        for part in path.parts[1:]:
            created = False
            if create:
                try:
                    os.mkdir(part, mode=mode, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise CASUnavailableError(
                        f"cannot create {where} component {part!r}: {exc}"
                    ) from exc
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                raise
            except OSError as exc:
                if exc.errno in {errno.ENOTDIR, errno.ELOOP}:
                    raise CASTamperError(
                        f"{where} ancestor is not a real directory: {path}"
                    ) from exc
                raise CASUnavailableError(
                    f"cannot open {where} component {part!r}: {exc}"
                ) from exc
            try:
                if created:
                    os.fchmod(child, mode)
                    os.fsync(child)
                    os.fsync(descriptor)
            except OSError as exc:
                os.close(child)
                raise CASUnavailableError(
                    f"cannot durably create {where} component {part!r}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_identity(descriptor: int, path: Path, *, where: str) -> None:
    """Fail if the held directory is no longer the configured pathname."""

    try:
        current = _open_directory_nofollow(path, where=where)
    except FileNotFoundError as exc:
        raise CASTamperError(f"{where} disappeared during operation: {path}") from exc
    try:
        held = os.fstat(descriptor)
        observed = os.fstat(current)
        if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            raise CASTamperError(f"{where} changed during operation: {path}")
    finally:
        os.close(current)


def _open_regular_nofollow(path: Path, *, where: str) -> tuple[int, int]:
    """Open a regular-file candidate relative to its held no-follow parent."""

    path = _absolute_nofollow_path(path, where=where)
    parent_fd = _open_directory_nofollow(path.parent, where=f"{where} parent")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        os.close(parent_fd)
        raise
    except OSError as exc:
        os.close(parent_fd)
        if exc.errno in {errno.ENOTDIR, errno.ELOOP, errno.EISDIR}:
            raise CASTamperError(
                f"cannot open {where} as a real regular file: {path}"
            ) from exc
        raise CASUnavailableError(f"cannot open {where}: {path}: {exc}") from exc
    return descriptor, parent_fd


def _assert_regular_identity(
    descriptor: int, parent_fd: int, path: Path, *, where: str
) -> None:
    """Fail if a held regular inode is no longer its canonical leaf name."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        current = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CASTamperError(f"{where} changed during operation: {path}") from exc
    try:
        held = os.fstat(descriptor)
        observed = os.fstat(current)
        if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            raise CASTamperError(f"{where} changed during operation: {path}")
    finally:
        os.close(current)


def _read_regular_file_nofollow(
    path: Path, *, where: str, require_readonly: bool = False
) -> bytes:
    """Read one stable inode without following any pathname component."""

    descriptor, parent_fd = _open_regular_nofollow(path, where=where)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CASTamperError(f"{where} is not a regular file: {path}")
        if require_readonly and before.st_mode & 0o222:
            raise CASTamperError(f"{where} is writable: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CASTamperError(f"{where} changed while it was read: {path}")
        _assert_regular_identity(
            descriptor, parent_fd, path, where=where
        )
        _assert_directory_identity(parent_fd, path.parent, where=f"{where} parent")
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _file_identity_nofollow(path: Path, *, where: str) -> tuple[str, int, int]:
    """Hash one stable inode reached through a held no-follow parent."""

    descriptor, parent_fd = _open_regular_nofollow(path, where=where)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CASTamperError(f"{where} is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CASTamperError(f"{where} changed while it was read: {path}")
        _assert_regular_identity(
            descriptor, parent_fd, path, where=where
        )
        _assert_directory_identity(parent_fd, path.parent, where=f"{where} parent")
        return digest.hexdigest(), size, before.st_mode
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _unlink_nofollow(path: Path, *, where: str) -> None:
    """Unlink a leaf only through its current no-follow parent directory."""

    try:
        directory_fd = _open_directory_nofollow(path.parent, where=f"{where} parent")
    except FileNotFoundError:
        return
    try:
        try:
            os.unlink(path.name, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        os.fsync(directory_fd)
        _assert_directory_identity(
            directory_fd, path.parent, where=f"{where} parent"
        )
    finally:
        os.close(directory_fd)


def _copy_to_staging(source: Path, staging_directory: Path) -> tuple[Path, str, int]:
    """Take a stable regular-file snapshot into the CAS filesystem."""

    staging_directory = _absolute_nofollow_path(
        staging_directory, where="CAS staging directory"
    )
    staging_fd = _open_directory_nofollow(
        staging_directory, where="CAS staging directory", create=True
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        os.close(staging_fd)
        raise LocalActionError(f"result is not a readable regular file: {source}") from exc
    try:
        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=".payload.", suffix=".tmp", dir=f"/proc/self/fd/{staging_fd}"
        )
    except BaseException:
        os.close(source_fd)
        os.close(staging_fd)
        raise
    temporary_name = Path(temporary_raw).name
    temporary = staging_directory / temporary_name
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise LocalActionError(f"result is not a regular file: {source}")
        with os.fdopen(descriptor, "wb") as destination:
            while True:
                chunk = os.read(source_fd, 4 * 1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            destination.flush()
            os.fchmod(destination.fileno(), 0o444)
            os.fsync(destination.fileno())
        after = os.fstat(source_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise LocalActionError(f"result changed while it was copied: {source}")
        _assert_directory_identity(
            staging_fd, staging_directory, where="CAS staging directory"
        )
        return temporary, digest.hexdigest(), size
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=staging_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_fd)
        os.close(staging_fd)


def _normalize_worker_evidence(value: object) -> dict[str, object]:
    evidence = _exact_mapping(
        value, keys=_EVIDENCE_KEYS, where="worker attestation.evidence"
    )
    source = _text(evidence["source"], where="worker evidence.source")
    if source not in {"local", "slurm"}:
        _fail("worker evidence.source must be 'local' or 'slurm'")
    accelerators_raw = evidence["accelerators"]
    if type(accelerators_raw) is not list:
        _fail("worker evidence.accelerators must be an array")
    accelerators: list[dict[str, str]] = []
    for index, raw in enumerate(accelerators_raw):
        accelerator = _exact_mapping(
            raw,
            keys=_ACCELERATOR_KEYS,
            where=f"worker evidence.accelerators[{index}]",
        )
        kind = _text(
            accelerator["kind"],
            where=f"worker evidence.accelerators[{index}].kind",
            pattern=_SCOPE_TOKEN_RE,
        )
        if kind != "nvidia":
            _fail("worker evidence contains an unsupported accelerator kind")
        capability = _text(
            accelerator["compute_capability"],
            where=f"worker evidence.accelerators[{index}].compute_capability",
        )
        if re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None:
            _fail("worker evidence compute capability is malformed")
        accelerators.append(
            {
                "kind": kind,
                "compute_capability": capability,
                "driver_version": _text(
                    accelerator["driver_version"],
                    where=f"worker evidence.accelerators[{index}].driver_version",
                    pattern=_VERSION_RE,
                ),
            }
        )
    accelerators.sort(
        key=lambda row: (row["kind"], row["compute_capability"], row["driver_version"])
    )
    if len({_canonical_bytes(row) for row in accelerators}) != len(accelerators):
        _fail("worker evidence accelerator rows must be unique")
    raw_slurm = evidence["slurm"]
    slurm: dict[str, object] | None
    if raw_slurm is None:
        slurm = None
    else:
        slurm_mapping = _exact_mapping(
            raw_slurm, keys=_SLURM_EVIDENCE_KEYS, where="worker evidence.slurm"
        )
        constraints_raw = slurm_mapping["constraints"]
        if type(constraints_raw) is not list:
            _fail("worker evidence.slurm.constraints must be an array")
        constraints = [
            _text(
                item,
                where=f"worker evidence.slurm.constraints[{index}]",
                pattern=_SCOPE_TOKEN_RE,
            )
            for index, item in enumerate(constraints_raw)
        ]
        if constraints != sorted(set(constraints)):
            _fail("worker evidence.slurm.constraints must be unique and sorted")
        cgroup = _text(
            slurm_mapping["cgroup"], where="worker evidence.slurm.cgroup"
        )
        if not cgroup.startswith("/") or ".." in PurePosixPath(cgroup).parts:
            _fail("worker evidence.slurm.cgroup must be an absolute cgroup path")
        job_id = _text(
            slurm_mapping["job_id"],
            where="worker evidence.slurm.job_id",
            pattern=_SCOPE_TOKEN_RE,
        )
        if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
            _fail("worker evidence.slurm.job_id must be a positive numeric job id")
        slurm = {
            "job_id": job_id,
            "node_name": _text(
                slurm_mapping["node_name"],
                where="worker evidence.slurm.node_name",
                pattern=_ID_RE,
            ),
            "partition": _text(
                slurm_mapping["partition"],
                where="worker evidence.slurm.partition",
                pattern=_SCOPE_TOKEN_RE,
            ),
            "constraints": constraints,
            "cgroup": cgroup,
        }
        job_pattern = re.compile(
            rf"(?:^|/)job_{re.escape(str(slurm['job_id']))}(?:[./]|$)"
        )
        if job_pattern.search(str(slurm["cgroup"])) is None:
            _fail("worker evidence SLURM cgroup does not bind its job id")
    if (source == "slurm") != (slurm is not None):
        _fail("worker evidence source and SLURM evidence disagree")
    return {
        "source": source,
        "hostname": _text(
            evidence["hostname"], where="worker evidence.hostname", pattern=_ID_RE
        ),
        "system": _text(
            evidence["system"], where="worker evidence.system", pattern=_SCOPE_TOKEN_RE
        ),
        "machine": _text(
            evidence["machine"], where="worker evidence.machine", pattern=_SCOPE_TOKEN_RE
        ),
        "libc": _text(
            evidence["libc"], where="worker evidence.libc", pattern=_SCOPE_TOKEN_RE
        ),
        "accelerators": accelerators,
        "slurm": slurm,
    }


def validate_worker_attestation(
    value: object, *, action: object
) -> dict[str, object]:
    """Validate a persisted preflight against the exact sealed action."""

    normalized_action = validate_action(action)
    raw = _exact_mapping(value, keys=_PRODUCER_KEYS, where="worker attestation")
    if raw["schema"] != WORKER_ATTESTATION_SCHEMA_V2:
        _fail(
            f"worker attestation.schema must be {WORKER_ATTESTATION_SCHEMA_V2!r}"
        )
    action_key = _sha256(raw["action_key"], where="worker attestation.action_key")
    if action_key != normalized_action["action_key"]:
        _fail("worker attestation is bound to a different action")
    evidence = _normalize_worker_evidence(raw["evidence"])
    worker_id = _text(
        raw["worker_id"], where="worker attestation.worker_id", pattern=_ID_RE
    )
    if worker_id != _worker_identity_from_evidence(evidence):
        _fail("worker_id is not derived from the worker evidence")
    platform_key = _optional_token(
        raw["platform_key"], where="worker attestation.platform_key"
    )
    derived_platform = _platform_key_from_evidence(evidence)
    if platform_key != derived_platform:
        _fail("platform_key is not derived from the worker evidence")
    host_class = _optional_token(
        raw["host_class"], where="worker attestation.host_class"
    )
    scope = normalized_action["execution_scope"]
    assert isinstance(scope, Mapping)
    expected_host = (
        str(scope["host_class"])
        if scope["portability"] == "host_class_keyed"
        else None
    )
    derived_host = _host_class_from_evidence(evidence, expected=expected_host)
    if host_class != derived_host:
        _fail("host_class is not derived from SLURM worker evidence")
    runtime = _validate_worker_runtime(raw["runtime"])

    executable_raw = _exact_mapping(
        raw["executable"], keys=_EXECUTABLE_KEYS, where="worker attestation.executable"
    )
    task = normalized_action["task"]
    assert isinstance(task, Mapping)
    argv = task["argv"]
    assert isinstance(argv, list)
    executable = {
        "path": _text(
            executable_raw["path"], where="worker attestation.executable.path"
        ),
        "resolved_path": _text(
            executable_raw["resolved_path"],
            where="worker attestation.executable.resolved_path",
        ),
        "sha256": _sha256(
            executable_raw["sha256"], where="worker attestation.executable.sha256"
        ),
        "bytes": _nonnegative_integer(
            executable_raw["bytes"], where="worker attestation.executable.bytes"
        ),
    }
    if executable["path"] != argv[0] or not Path(
        str(executable["resolved_path"])
    ).is_absolute():
        _fail("worker attestation executable differs from action.task.argv[0]")

    toolchain_raw = _exact_mapping(
        raw["toolchain"],
        keys=_TOOLCHAIN_ATTESTATION_KEYS,
        where="worker attestation.toolchain",
    )
    declared = _normalize_string_mapping(
        toolchain_raw["declared"], where="worker attestation.toolchain.declared"
    )
    environment = normalized_action["environment"]
    assert isinstance(environment, Mapping)
    if declared != environment["toolchain"]:
        _fail("worker attestation toolchain differs from the action")
    verified = _normalize_string_mapping(
        toolchain_raw["verified"], where="worker attestation.toolchain.verified"
    )
    if not set(verified) <= set(declared):
        _fail("worker attestation verifies undeclared toolchain fields")
    for key, observed in verified.items():
        if not _toolchain_value_matches(declared[key], observed):
            _fail(f"worker toolchain field {key!r} differs from the action")
    if (
        declared.get("argv0.sha256") is not None
        and declared["argv0.sha256"] != executable["sha256"]
    ):
        _fail("worker executable digest differs from toolchain argv0.sha256")
    if (
        declared.get("argv0.bytes") is not None
        and declared["argv0.bytes"] != str(executable["bytes"])
    ):
        _fail("worker executable size differs from toolchain argv0.bytes")

    inputs = _normalize_inputs(raw["inputs"])
    expected_inputs = {
        str(entry["id"]): entry
        for entry in normalized_action["inputs"]  # type: ignore[union-attr]
    }
    if any(expected_inputs.get(str(entry["id"])) != entry for entry in inputs):
        _fail("worker attestation contains an input not bound by the action")

    nonportable = scope["portability"] != "portable"
    if nonportable:
        if evidence["libc"] == "unknown":
            _fail("nonportable action cannot attest an unknown libc ABI")
        if set(verified) != set(declared):
            _fail("nonportable action has unverified toolchain fields")
        if len(inputs) != len(expected_inputs):
            _fail("nonportable action has unresolved input artifacts")
        accelerators = evidence["accelerators"]
        assert isinstance(accelerators, list)
        if accelerators and not {
            "cuda_compute_capability",
            "nvidia_driver",
        } <= set(verified):
            _fail(
                "nonportable NVIDIA action must bind cuda_compute_capability "
                "and nvidia_driver in its toolchain"
            )

    _validate_scope_labels(
        normalized_action, platform_key=platform_key, host_class=host_class
    )
    body = {
        "schema": WORKER_ATTESTATION_SCHEMA_V2,
        "action_key": action_key,
        "worker_id": worker_id,
        "platform_key": platform_key,
        "host_class": host_class,
        "evidence": evidence,
        "runtime": runtime,
        "executable": executable,
        "toolchain": {"declared": declared, "verified": verified},
        "inputs": inputs,
    }
    recorded = _sha256(
        raw["attestation_sha256"], where="worker attestation.attestation_sha256"
    )
    if recorded != canonical_sha256(body):
        _fail("worker attestation digest does not match its body")
    return {**body, "attestation_sha256": recorded}


def _verified_toolchain(
    declared: Mapping[str, str],
    *,
    executable: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, str]:
    observed: dict[str, str] = {
        "argv0.sha256": str(executable["sha256"]),
        "argv0.bytes": str(executable["bytes"]),
        "system": str(evidence["system"]),
        "machine": str(evidence["machine"]),
        "libc": str(evidence["libc"]),
    }
    accelerators = evidence["accelerators"]
    assert isinstance(accelerators, list)
    if accelerators:
        capabilities = {str(row["compute_capability"]) for row in accelerators}
        drivers = {str(row["driver_version"]) for row in accelerators}
        if len(capabilities) == 1:
            observed["cuda_compute_capability"] = next(iter(capabilities))
        if len(drivers) == 1:
            observed["nvidia_driver"] = next(iter(drivers))
    python_keys = set(declared) & ({"python"} | _PYTHON_DISTRIBUTIONS)
    if python_keys:
        observed.update(
            _probe_python_toolchain(Path(str(executable["path"])))
        )
    verified: dict[str, str] = {}
    for key, expected in declared.items():
        actual = observed.get(key)
        if actual is None:
            continue
        if not _toolchain_value_matches(expected, actual):
            raise ActionContractError(
                f"worker toolchain field {key!r} differs: "
                f"declared={expected!r}, observed={actual!r}"
            )
        verified[key] = actual
    return dict(sorted(verified.items()))


def _verified_input_contracts(
    action: Mapping[str, object], cas: "PrismaBuildCAS"
) -> list[dict[str, object]]:
    scope = action["execution_scope"]
    assert isinstance(scope, Mapping)
    required = scope["portability"] != "portable"
    verified: list[dict[str, object]] = []
    for entry in action["inputs"]:  # type: ignore[union-attr]
        assert isinstance(entry, Mapping)
        path = cas._blob_path(str(entry["sha256"]))
        try:
            path.lstat()
        except FileNotFoundError:
            if required:
                raise ActionContractError(
                    f"nonportable action input is absent from the CAS: {entry['id']}"
                )
            continue
        except OSError as exc:
            raise CASUnavailableError(
                f"cannot inspect action input in the CAS: {entry['id']}: {exc}"
            ) from exc
        cas.input_path(entry)
        verified.append(dict(entry))
    return sorted(verified, key=lambda entry: str(entry["id"]))


def preflight_action(
    action: object,
    *,
    cas_root: str | Path,
    checkout_root: str | Path,
    worker_launcher_identity: object | None = None,
) -> dict[str, object]:
    """Derive and verify the execution facts required before launching argv."""

    normalized = validate_action(action)
    root = Path(checkout_root)
    if not root.is_absolute():
        _fail("checkout_root must be absolute")
    verify_code_closure(normalized["code_closure"], root)
    evidence = _collect_worker_evidence()
    scope = normalized["execution_scope"]
    assert isinstance(scope, Mapping)
    expected_host = (
        str(scope["host_class"])
        if scope["portability"] == "host_class_keyed"
        else None
    )
    host_class = _host_class_from_evidence(evidence, expected=expected_host)
    executable = identify_executable(
        normalized["task"]["argv"][0]  # type: ignore[index]
    )
    environment = normalized["environment"]
    assert isinstance(environment, Mapping)
    declared = environment["toolchain"]
    assert isinstance(declared, Mapping)
    verified = _verified_toolchain(
        declared, executable=executable, evidence=evidence  # type: ignore[arg-type]
    )
    inputs = _verified_input_contracts(normalized, PrismaBuildCAS(cas_root))
    body: dict[str, object] = {
        "schema": WORKER_ATTESTATION_SCHEMA_V2,
        "action_key": normalized["action_key"],
        "worker_id": _worker_identity_from_evidence(evidence),
        "platform_key": _platform_key_from_evidence(evidence),
        "host_class": host_class,
        "evidence": evidence,
        "runtime": _worker_runtime_identity(worker_launcher_identity),
        "executable": executable,
        "toolchain": {"declared": dict(declared), "verified": verified},
        "inputs": inputs,
    }
    attestation = {**body, "attestation_sha256": canonical_sha256(body)}
    return validate_worker_attestation(attestation, action=normalized)


def _verify_attested_executable_unchanged(
    attestation: Mapping[str, object], action: Mapping[str, object]
) -> None:
    validated = validate_worker_attestation(attestation, action=action)
    expected = validated["executable"]
    assert isinstance(expected, Mapping)
    observed = identify_executable(str(expected["path"]))
    if observed != expected:
        raise LocalActionError("action executable changed after worker preflight")


def _verify_attested_worker_runtime_unchanged(
    attestation: Mapping[str, object], action: Mapping[str, object]
) -> None:
    validated = validate_worker_attestation(attestation, action=action)
    _verify_worker_runtime_unchanged(validated["runtime"])


class PrismaBuildCAS:
    """Immutable file-result CAS with verified first-result-wins receipts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ActionContractError("CAS root must be a non-root absolute path")
        if ".." in self.root.parts:
            raise ActionContractError("CAS root must not contain parent traversal")

    def _receipt_path(self, action_key: str) -> Path:
        # Receipt schemas are immutable interpretation domains.  v2 occupied
        # the unversioned path below ``actions/``; v3 must coexist with it and
        # must never parse, overwrite, or delete those historical bytes.
        return (
            self.root
            / "actions"
            / _CAS_RECEIPT_NAMESPACE
            / action_key[:2]
            / f"{action_key}.json"
        )

    def _legacy_v2_receipt_path(self, action_key: str) -> Path:
        return self.root / "actions" / action_key[:2] / f"{action_key}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def _open_input_blob_shard(self, digest: str) -> tuple[Path, int]:
        """Open a real CAS shard directory without following CAS symlinks."""

        shard = self.root / "blobs" / digest[:2]
        descriptor = _open_directory_nofollow(
            shard,
            where="CAS input directory",
            create=True,
        )
        return shard / digest, descriptor

    def _publish_staged_input_blob(
        self, staging: Path, contract: Mapping[str, object]
    ) -> tuple[Path, bool]:
        """Link a staged input into the CAS and verify the canonical name."""

        digest = str(contract["sha256"])
        blob_path, directory_fd = self._open_input_blob_shard(digest)
        try:
            source_fd = _open_directory_nofollow(
                staging.parent, where="CAS staging directory"
            )
        except BaseException:
            os.close(directory_fd)
            raise
        try:
            try:
                os.link(
                    staging.name,
                    digest,
                    src_dir_fd=source_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                won = True
            except FileExistsError:
                won = False
            if won:
                os.fsync(directory_fd)
            _assert_directory_identity(
                source_fd, staging.parent, where="CAS staging directory"
            )
            _assert_directory_identity(
                directory_fd, blob_path.parent, where="CAS blob directory"
            )
        finally:
            os.close(source_fd)
            os.close(directory_fd)
        # A successful hard link is not sufficient evidence: re-open and hash
        # the canonical name after its directory entry has been synchronized.
        # The same verification rejects a malformed or conflicting race winner.
        return self._verify_input_blob(contract), won

    def ingest_input(
        self,
        source_path: str | Path,
        *,
        input_id: str,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> tuple[dict[str, object], bool]:
        """Publish one stable file snapshot as an immutable action input.

        The digest and byte count are derived from the staged bytes.  Optional
        expectations fail before publication when the source differs.  The
        returned row can be inserted directly into ``action.inputs``; ``won``
        is false only when an independently verified identical blob already
        occupied the content address.
        """

        identity = _text(input_id, where="input id", pattern=_ID_RE)
        expected_digest = (
            _sha256(expected_sha256, where="expected input sha256")
            if expected_sha256 is not None
            else None
        )
        expected_size = (
            _nonnegative_integer(expected_bytes, where="expected input bytes")
            if expected_bytes is not None
            else None
        )
        staging, digest, size = _copy_to_staging(
            Path(source_path), self.root / ".staging"
        )
        try:
            if expected_digest is not None and digest != expected_digest:
                raise ActionContractError(
                    "ingested input sha256 differs from the expected digest"
                )
            if expected_size is not None and size != expected_size:
                raise ActionContractError(
                    "ingested input byte count differs from the expected size"
                )
            entry = validate_input_contract(
                {"id": identity, "sha256": digest, "bytes": size}
            )
            contract = {"sha256": digest, "bytes": size}
            _, won = self._publish_staged_input_blob(staging, contract)
            return entry, won
        finally:
            _unlink_nofollow(staging, where="CAS staging file")

    def input_path(self, input_contract: object) -> Path:
        """Return an action input's CAS path after full content verification."""

        entry = validate_input_contract(input_contract)
        return self._verify_input_blob(
            {"sha256": entry["sha256"], "bytes": entry["bytes"]}
        )

    def publish_action_request(self, action: object) -> Path:
        """Publish the canonical immutable request consumed by remote workers."""

        normalized = validate_action(action)
        key = str(normalized["action_key"])
        path = self.root / "requests" / key[:2] / f"{key}.json"
        raw = _canonical_file_bytes(normalized)
        won = _atomic_publish(path, raw)
        if not won:
            observed = _read_regular_file_nofollow(
                path,
                where="PrismaBuild action request",
                require_readonly=True,
            )
            if observed != raw:
                raise CASTamperError(
                    "existing PrismaBuild action request differs from its action key"
                )
        # Re-read after either side of a publication race.  A scheduler must
        # never send a path whose bytes the submitter merely assumes.
        verified = _read_regular_file_nofollow(
            path,
            where="PrismaBuild action request",
            require_readonly=True,
        )
        if verified != raw:
            raise CASTamperError(
                "published PrismaBuild action request failed canonical readback"
            )
        return path

    def _load_receipt_bytes(self, path: Path) -> bytes:
        try:
            return _read_regular_file_nofollow(
                path, where="CAS receipt", require_readonly=True
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(path) from exc

    def _validate_receipt(
        self, value: object, *, action: Mapping[str, object]
    ) -> dict[str, object]:
        try:
            receipt = _exact_mapping(value, keys=_RECEIPT_KEYS, where="CAS receipt")
            if receipt["schema"] != CAS_RECEIPT_SCHEMA_V3:
                _fail(f"CAS receipt schema must be {CAS_RECEIPT_SCHEMA_V3!r}")
            action_key = _sha256(receipt["action_key"], where="CAS receipt.action_key")
            if action_key != action["action_key"]:
                _fail("CAS receipt action_key differs from the requested action")
            manifest_sha = _sha256(
                receipt["action_manifest_sha256"],
                where="CAS receipt.action_manifest_sha256",
            )
            if manifest_sha != canonical_sha256(action):
                _fail("CAS receipt action manifest digest differs from the action")
            raw_result = _exact_mapping(
                receipt["result"], keys=_RESULT_KEYS, where="CAS receipt.result"
            )
            result = {
                "sha256": _sha256(
                    raw_result["sha256"], where="CAS receipt.result.sha256"
                ),
                "bytes": _nonnegative_integer(
                    raw_result["bytes"], where="CAS receipt.result.bytes"
                ),
            }
            raw_producer = _exact_mapping(
                receipt["producer"],
                keys=_PRODUCER_KEYS,
                where="CAS receipt.producer",
            )
            # Publication runs this preflight live.  Lookup independently
            # replays its derivations and action binding from persisted data.
            producer = validate_worker_attestation(raw_producer, action=action)
            body = {
                "schema": CAS_RECEIPT_SCHEMA_V3,
                "action_key": action_key,
                "action_manifest_sha256": manifest_sha,
                "result": result,
                "producer": producer,
            }
            recorded = _sha256(
                receipt["receipt_sha256"], where="CAS receipt.receipt_sha256"
            )
            if recorded != canonical_sha256(body):
                _fail("CAS receipt.receipt_sha256 does not match its body")
            return {**body, "receipt_sha256": recorded}
        except ActionContractError as exc:
            raise CASTamperError(str(exc)) from exc

    def _verify_blob(
        self,
        result: Mapping[str, object],
        *,
        writable_label: str = "CAS payload",
    ) -> Path:
        digest = str(result["sha256"])
        path = self._blob_path(digest)
        try:
            observed_digest, observed_size, observed_mode = _file_identity_nofollow(
                path, where="CAS payload"
            )
        except FileNotFoundError as exc:
            raise CASTamperError(f"CAS payload is missing: {path}") from exc
        if observed_size != result["bytes"]:
            raise CASTamperError(
                f"CAS payload content differs from receipt (size mismatch): {path}"
            )
        if observed_digest != digest or observed_size != result["bytes"]:
            raise CASTamperError(
                f"CAS payload content differs from receipt: {path}"
            )
        if observed_mode & 0o222:
            raise CASTamperError(f"{writable_label} is writable: {path}")
        return path

    def _verify_input_blob(self, contract: Mapping[str, object]) -> Path:
        return self._verify_blob(contract, writable_label="CAS input payload")

    def lookup(self, action: object) -> dict[str, object] | None:
        """Return a verified v3 receipt, or ``None`` on a v3 miss.

        An unversioned legacy-v2 receipt is deliberately neither migrated nor
        interpreted here.  It remains immutable history while v3 recomputes
        into its disjoint namespace.
        """

        normalized = validate_action(action)
        path = self._receipt_path(str(normalized["action_key"]))
        try:
            raw = self._load_receipt_bytes(path)
        except FileNotFoundError:
            return None
        try:
            value = _decode_strict_json(raw, where="CAS receipt")
            receipt = self._validate_receipt(value, action=normalized)
        except ActionContractError as exc:
            raise CASTamperError(str(exc)) from exc
        if raw != _canonical_file_bytes(receipt):
            raise CASTamperError("CAS receipt bytes are not canonical JSON")
        self._verify_blob(receipt["result"])  # type: ignore[arg-type]
        return receipt

    def _verified_receipt_result_path(self, receipt: Mapping[str, object]) -> Path:
        """Return the blob path after ``lookup`` has already verified it."""

        result = receipt["result"]
        assert isinstance(result, Mapping)
        return self._blob_path(str(result["sha256"]))

    def result_path(self, receipt: object, action: object) -> Path:
        normalized = validate_action(action)
        validated = self._validate_receipt(receipt, action=normalized)
        return self._verify_blob(validated["result"])  # type: ignore[arg-type]

    def publish_result(
        self,
        action: object,
        result_path: str | Path,
        *,
        attestation: object,
        precommit_verify: Callable[[], None] | None = None,
    ) -> tuple[dict[str, object], bool]:
        """Publish a result and return ``(canonical_receipt, won_publication)``.

        A losing deterministic producer must reproduce the winner byte-for-byte;
        a stochastic producer accepts the already-published canonical result.
        ``precommit_verify`` runs after the payload copy and temporary receipt
        fsync. Local workers use it for their action-specific closure checks;
        the worker core and optional script launcher are rechecked after that
        callback as the final userspace operation before the no-clobber link.
        This cannot make mutable source an immutable snapshot; the remaining
        verification-to-link syscall interval is deliberately small.
        """

        normalized = validate_action(action)
        producer = validate_worker_attestation(attestation, action=normalized)
        staging, digest, size = _copy_to_staging(
            Path(result_path), self.root / ".staging"
        )
        result = {"sha256": digest, "bytes": size}
        try:
            _, blob_won = self._publish_staged_input_blob(staging, result)
            body: dict[str, object] = {
                "schema": CAS_RECEIPT_SCHEMA_V3,
                "action_key": normalized["action_key"],
                "action_manifest_sha256": canonical_sha256(normalized),
                "result": result,
                "producer": producer,
            }
            candidate = {**body, "receipt_sha256": canonical_sha256(body)}
            receipt_path = self._receipt_path(str(normalized["action_key"]))

            def verify_publication_provenance() -> None:
                # Action-specific closure/executable checks may be relatively
                # slow.  Run them first so the runtime recheck is the final
                # userspace operation before _atomic_publish calls os.link.
                if precommit_verify is not None:
                    precommit_verify()
                _verify_worker_runtime_unchanged(producer["runtime"])

            won = _atomic_publish(
                receipt_path,
                _canonical_file_bytes(candidate),
                prelink_verify=verify_publication_provenance,
            )
            canonical = self.lookup(normalized)
            if canonical is None:
                raise CASTamperError("CAS receipt vanished after publication race")
            if won:
                if canonical != candidate:
                    raise CASTamperError(
                        "published CAS receipt failed canonical readback"
                    )
                return canonical, True
            task = normalized["task"]
            assert isinstance(task, Mapping)
            canonical_result = canonical["result"]
            assert isinstance(canonical_result, Mapping)
            if task["determinism"] == "deterministic" and canonical_result != result:
                raise CASConflictError(
                    "deterministic recomputation differs from the canonical CAS result"
                )
            return canonical, False
        finally:
            _unlink_nofollow(staging, where="CAS staging file")


def _validate_execution_paths(
    checkout_root: Path, working_directory: str, result_path: str
) -> tuple[Path, Path]:
    """Resolve an action cwd and refuse traversal out of the checkout."""

    if checkout_root.is_symlink():
        raise LocalActionError(f"checkout root is a symlink: {checkout_root}")
    try:
        resolved_root = checkout_root.resolve(strict=True)
    except OSError as exc:
        raise LocalActionError(f"checkout root is unavailable: {checkout_root}") from exc
    cwd = (
        checkout_root
        if working_directory == "."
        else checkout_root / working_directory
    )
    try:
        resolved_cwd = cwd.resolve(strict=True)
        resolved_cwd.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise LocalActionError(
            f"declared working directory escapes or is unavailable: {cwd}"
        ) from exc
    if cwd.is_symlink() or not resolved_cwd.is_dir():
        raise LocalActionError(
            f"declared working directory is not a real directory: {cwd}"
        )
    return resolved_cwd, resolved_cwd / result_path


def _validate_result_containment(output: Path, cwd: Path) -> None:
    """Refuse result files reached through a symlinked parent or final link."""

    try:
        resolved = output.resolve(strict=True)
        resolved.relative_to(cwd)
    except (OSError, ValueError) as exc:
        raise LocalActionError(
            f"declared result escapes or is unavailable: {output}"
        ) from exc
    relative = output.relative_to(cwd)
    cursor = cwd
    for component in relative.parts:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise LocalActionError(
                f"cannot inspect declared result path: {cursor}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise LocalActionError(
                f"declared result path traverses a symlink: {cursor}"
            )
    if not stat.S_ISREG(output.lstat().st_mode):
        raise LocalActionError(f"declared result is not a regular file: {output}")


def _refuse_existing_result_symlink_prefix(output: Path, cwd: Path) -> None:
    """Reject a pre-existing symlink in the declared output path."""

    relative = output.relative_to(cwd)
    cursor = cwd
    for component in relative.parts:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise LocalActionError(
                f"cannot inspect declared result path: {cursor}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise LocalActionError(
                f"declared result path traverses a symlink: {cursor}"
            )


@contextmanager
def _local_output_lock(cas: PrismaBuildCAS, checkout: Path, output: Path):
    """Serialize actions sharing one live-checkout result path."""

    identity = hashlib.sha256(
        f"{checkout.resolve(strict=True)}\0{output}".encode("utf-8")
    ).hexdigest()
    directory = cas.root / ".worker-locks"
    path = directory / f"{identity}.lock"
    directory_fd = _open_directory_nofollow(
        directory, where="local action lock directory", create=True
    )
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        os.close(directory_fd)
        raise LocalActionError(f"cannot open local action lock: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LocalActionError(
                f"local action lock is not a regular file: {path}"
            )
        _assert_directory_identity(
            directory_fd, directory, where="local action lock directory"
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            os.close(directory_fd)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def run_local_action(
    action: object,
    *,
    cas_root: str | Path,
    checkout_root: str | Path,
    timeout_seconds: float | None = None,
    recompute: bool = False,
    worker_launcher_identity: object | None = None,
) -> dict[str, object]:
    """Execute one action locally and publish its declared file result."""

    normalized = validate_action(action)
    root = Path(checkout_root)
    if not root.is_absolute():
        raise ActionContractError("checkout_root must be absolute")
    cas = PrismaBuildCAS(cas_root)
    if not recompute:
        cached = cas.lookup(normalized)
        if cached is not None:
            verify_code_closure(normalized["code_closure"], root)
            return {
                "status": "cache_hit",
                "receipt": cached,
                "payload_path": str(cas._verified_receipt_result_path(cached)),
            }
    task = normalized["task"]
    environment = normalized["environment"]
    assert isinstance(task, Mapping)
    assert isinstance(environment, Mapping)
    cwd, output = _validate_execution_paths(
        root, str(task["working_directory"]), str(task["result_path"])
    )
    variables = environment["variables"]
    assert isinstance(variables, Mapping)
    with _local_output_lock(cas, root, output):
        # A concurrent producer may have filled the cache while this worker
        # waited for the checkout/output lock.
        if not recompute:
            cached = cas.lookup(normalized)
            if cached is not None:
                verify_code_closure(normalized["code_closure"], root)
                return {
                    "status": "cache_hit",
                    "receipt": cached,
                    "payload_path": str(cas._verified_receipt_result_path(cached)),
                }
        attestation = preflight_action(
            normalized,
            cas_root=cas_root,
            checkout_root=root,
            worker_launcher_identity=worker_launcher_identity,
        )
        if output.exists() or output.is_symlink():
            raise LocalActionError(
                f"declared result path must be absent before execution: {output}"
            )
        _refuse_existing_result_symlink_prefix(output, cwd)
        try:
            process = subprocess.Popen(
                list(task["argv"]),
                cwd=cwd,
                env={str(key): str(value) for key, value in variables.items()},
                shell=False,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                raise LocalActionError(f"action execution timed out: {exc}") from exc
        except OSError as exc:
            raise LocalActionError(f"action execution failed: {exc}") from exc
        if returncode != 0:
            raise LocalActionError(
                f"action argv exited with status {returncode}"
            )
        if not output.exists() and not output.is_symlink():
            raise LocalActionError(
                f"action succeeded without its declared result file: {output}"
            )
        # Preflight alone is insufficient: an action (or another checkout
        # writer) can change a closure member while argv is running.  Check
        # once before an expensive result copy, then again at the CAS receipt
        # commit point so copying a large artifact cannot reopen that gap.
        verify_code_closure(normalized["code_closure"], root)
        _verify_attested_executable_unchanged(attestation, normalized)
        _verify_attested_worker_runtime_unchanged(attestation, normalized)
        _validate_result_containment(output, cwd)

        def verify_publication_provenance() -> None:
            verify_code_closure(normalized["code_closure"], root)
            _verify_attested_executable_unchanged(attestation, normalized)

        receipt, won = cas.publish_result(
            normalized,
            output,
            attestation=attestation,
            precommit_verify=verify_publication_provenance,
        )
    return {
        "status": "published" if won else "canonical_result_reused",
        "receipt": receipt,
        "payload_path": str(cas.result_path(receipt, normalized)),
    }


def _read_json_mapping(path: Path, *, where: str) -> Mapping[str, object]:
    raw = _read_regular_file(path, where=where)
    value = _decode_strict_json(raw, where=where)
    if not isinstance(value, Mapping):
        raise ActionContractError(f"{where} root must be an object")
    return value


def _atomic_write_new_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ActionContractError(f"output already exists: {path}")
    if not _atomic_publish(path, json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"):
        raise ActionContractError(f"output appeared concurrently: {path}")


def _require_slurm_initial_start(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Refuse a requeued/admin-restarted batch process before task argv runs.

    PrismaBuild's Slurm adapter currently seals ``max_requeues=0`` and submits
    with ``--no-requeue``. Slurm nevertheless exposes the actual restart count
    to the launched process; this worker-side gate is the final defense if an
    administrator or site policy restarts the allocation anyway.
    """

    env = os.environ if environment is None else environment
    job_id = env.get("SLURM_JOB_ID")
    if type(job_id) is not str or re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None:
        raise ActionContractError(
            "Slurm worker requires a positive numeric SLURM_JOB_ID"
        )
    restart_count = env.get("SLURM_RESTART_COUNT", "0")
    if type(restart_count) is not str or re.fullmatch(
        r"(?:0|[1-9][0-9]{0,8})", restart_count
    ) is None:
        raise ActionContractError("SLURM_RESTART_COUNT is malformed")
    if int(restart_count) != 0:
        raise ActionContractError(
            "restarted Slurm allocation is not authorized to execute task argv"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    closure = commands.add_parser("seal-closure")
    closure.add_argument("--root", required=True, type=Path)
    closure.add_argument("--file", action="append", required=True)
    closure.add_argument("--output", required=True, type=Path)
    seal = commands.add_parser("seal-action")
    seal.add_argument("--body", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    key = commands.add_parser("key")
    key.add_argument("--action", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--action", required=True, type=Path)
    verify.add_argument("--cas-root", required=True, type=Path)
    ingest = commands.add_parser("ingest-input")
    ingest.add_argument("--source", required=True, type=Path)
    ingest.add_argument("--cas-root", required=True, type=Path)
    ingest.add_argument("--input-id", required=True)
    ingest.add_argument("--expected-sha256")
    ingest.add_argument("--expected-bytes", type=int)
    verify_input = commands.add_parser("verify-input")
    verify_input.add_argument("--input-contract", required=True, type=Path)
    verify_input.add_argument("--cas-root", required=True, type=Path)
    run = commands.add_parser("run-local")
    run.add_argument("--action", required=True, type=Path)
    run.add_argument("--cas-root", required=True, type=Path)
    run.add_argument("--checkout-root", required=True, type=Path)
    run.add_argument("--timeout-seconds", type=float)
    run.add_argument("--recompute", action="store_true")
    run.add_argument("--require-slurm-initial-start", action="store_true")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--action", required=True, type=Path)
    preflight.add_argument("--cas-root", required=True, type=Path)
    preflight.add_argument("--checkout-root", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    worker_launcher_identity: object | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run-local" and args.require_slurm_initial_start:
        if args.recompute:
            raise ActionContractError(
                "Slurm worker protocol does not permit recompute"
            )
        # This is deliberately before reading even the immutable action
        # request: a forced/admin-restarted allocation must reach no worker or
        # task-side input processing beyond argument parsing.
        _require_slurm_initial_start()
    if args.command == "seal-closure":
        closure = build_code_closure(args.root, args.file)
        _atomic_write_new_json(args.output, closure)
        print(args.output)
        return 0
    if args.command == "ingest-input":
        cas = PrismaBuildCAS(args.cas_root)
        entry, won = cas.ingest_input(
            args.source,
            input_id=args.input_id,
            expected_sha256=args.expected_sha256,
            expected_bytes=args.expected_bytes,
        )
        print(
            json.dumps(
                {
                    "status": "published" if won else "already_present",
                    "input": entry,
                    "payload_path": str(cas.input_path(entry)),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify-input":
        contract = _read_json_mapping(args.input_contract, where="input contract")
        cas = PrismaBuildCAS(args.cas_root)
        normalized = validate_input_contract(contract)
        print(
            json.dumps(
                {
                    "input": normalized,
                    "payload_path": str(cas.input_path(normalized)),
                },
                sort_keys=True,
            )
        )
        return 0
    action_path = args.action if hasattr(args, "action") else args.body
    if args.command == "run-local" and args.require_slurm_initial_start:
        raw_action = _read_regular_file_nofollow(
            action_path, where="action request", require_readonly=True
        )
        action_data_value = _decode_strict_json(raw_action, where="action request")
        if not isinstance(action_data_value, Mapping):
            raise ActionContractError("action request root must be an object")
        action_data = action_data_value
    else:
        action_data = _read_json_mapping(action_path, where="action")
    if args.command == "seal-action":
        sealed = seal_action(action_data)
        _atomic_write_new_json(args.output, sealed)
        print(args.output)
        return 0
    action = validate_action(action_data)
    if args.command == "run-local" and args.require_slurm_initial_start:
        cas = PrismaBuildCAS(args.cas_root)
        expected_request = (
            cas.root
            / "requests"
            / str(action["action_key"])[:2]
            / f"{action['action_key']}.json"
        )
        if action_path != expected_request:
            raise ActionContractError(
                "Slurm worker action path differs from its canonical CAS request"
            )
    if args.command == "key":
        print(action["action_key"])
        return 0
    if args.command == "preflight":
        result = preflight_action(
            action,
            cas_root=args.cas_root,
            checkout_root=args.checkout_root,
            worker_launcher_identity=worker_launcher_identity,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    cas = PrismaBuildCAS(args.cas_root)
    if args.command == "verify":
        receipt = cas.lookup(action)
        if receipt is None:
            raise PrismaBuildError("action is not present in the CAS")
        print(json.dumps(receipt, sort_keys=True))
        return 0
    result = run_local_action(
        action,
        cas_root=args.cas_root,
        checkout_root=args.checkout_root,
        timeout_seconds=args.timeout_seconds,
        recompute=args.recompute,
        worker_launcher_identity=worker_launcher_identity,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "ACTION_SCHEMA_V1",
    "CAS_RECEIPT_SCHEMA_V3",
    "CODE_CLOSURE_SCHEMA_V1",
    "WORKER_ATTESTATION_SCHEMA_V2",
    "WORKER_RUNTIME_SCHEMA_V1",
    "ActionContractError",
    "CASConflictError",
    "CASTamperError",
    "CASUnavailableError",
    "LocalActionError",
    "PrismaBuildCAS",
    "PrismaBuildError",
    "build_code_closure",
    "executable_toolchain_contract",
    "identify_executable",
    "main",
    "preflight_action",
    "run_local_action",
    "seal_action",
    "validate_action",
    "validate_code_closure",
    "validate_input_contract",
    "validate_worker_scope",
    "validate_worker_attestation",
    "verify_code_closure",
]


if __name__ == "__main__":
    raise SystemExit(
        main(worker_launcher_identity=dict(_LOADED_WORKER_CORE_IDENTITY))
    )
