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
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile

from .cluster_campaign import canonical_sha256


ACTION_SCHEMA_V1 = "prismaquant.prismabuild.action.v1"
CODE_CLOSURE_SCHEMA_V1 = "prismaquant.prismabuild.code_closure.v1"
CAS_RECEIPT_SCHEMA_V1 = "prismaquant.prismabuild.cas_receipt.v1"

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
_PRODUCER_KEYS = frozenset({"worker_id", "platform_key", "host_class"})

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,255}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}\Z")
_SCOPE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/-]{0,255}\Z")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_PORTABILITY = frozenset({"portable", "platform_keyed", "host_class_keyed"})
_TASK_CLASSES = frozenset({"generation", "measurement"})
_DETERMINISM = frozenset({"deterministic", "stochastic"})
_FP8_CB_ARTIFACT_KINDS = frozenset(
    {"fp8_cb", "fp8-cb", "nvfp4_cb", "nvfp4-cb"}
)


class PrismaBuildError(RuntimeError):
    """Base class for PrismaBuild core failures."""


class ActionContractError(PrismaBuildError, ValueError):
    """An action, closure, receipt, or worker identity is not exact."""


class CASTamperError(PrismaBuildError):
    """An existing content-addressed entry failed verification."""


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


def _normalize_relative_path(value: object, *, where: str, dot_ok: bool) -> str:
    raw = _text(value, where=where)
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
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
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ActionContractError(f"{where} changed while it was read: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


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
        task["artifact_kind"] in _FP8_CB_ARTIFACT_KINDS
        and scope["portability"] == "portable"
    ):
        _fail(
            "FP8-CB actions cannot be portable: D29 records cross-architecture "
            "row-scale byte drift"
        )
    params = body["params"]
    if not isinstance(params, Mapping) or any(type(key) is not str for key in params):
        _fail("action.params must be an object with string keys")
    # Round-trip through canonical JSON to reject unsupported and non-finite data
    # and to detach the sealed value from caller-owned mutable containers.
    normalized_params = _decode_strict_json(
        _canonical_bytes(params), where="action.params"
    )
    return {
        "schema": ACTION_SCHEMA_V1,
        "task": task,
        "inputs": _normalize_inputs(body["inputs"]),
        "code_closure": validate_code_closure(body["code_closure"]),
        "params": normalized_params,
        "environment": _normalize_environment(body["environment"]),
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
    platform_key: str | None,
    host_class: str | None,
) -> None:
    normalized = validate_action(action)
    actual_platform = _optional_token(platform_key, where="worker platform_key")
    actual_host = _optional_token(host_class, where="worker host_class")
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


def _atomic_publish(path: Path, raw: bytes) -> bool:
    """Publish immutable bytes with an NFS-safe first-writer-wins hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        try:
            os.link(temporary, path)
            won = True
        except FileExistsError:
            won = False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return won
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _copy_to_staging(source: Path, staging_directory: Path) -> tuple[Path, str, int]:
    """Take a stable regular-file snapshot into the CAS filesystem."""

    staging_directory.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise LocalActionError(f"result is not a readable regular file: {source}") from exc
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=".payload.", suffix=".tmp", dir=staging_directory
    )
    temporary = Path(temporary_raw)
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
            os.fsync(destination.fileno())
            os.fchmod(destination.fileno(), 0o444)
        after = os.fstat(source_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise LocalActionError(f"result changed while it was copied: {source}")
        return temporary, digest.hexdigest(), size
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_fd)


def _validate_producer(
    *, worker_id: object, platform_key: object, host_class: object
) -> dict[str, object]:
    return {
        "worker_id": _text(worker_id, where="producer.worker_id", pattern=_ID_RE),
        "platform_key": _optional_token(
            platform_key, where="producer.platform_key"
        ),
        "host_class": _optional_token(host_class, where="producer.host_class"),
    }


class PrismaBuildCAS:
    """Immutable file-result CAS with verified first-result-wins receipts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ActionContractError("CAS root must be a non-root absolute path")

    def _receipt_path(self, action_key: str) -> Path:
        return self.root / "actions" / action_key[:2] / f"{action_key}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def _load_receipt_bytes(self, path: Path) -> bytes:
        try:
            return _read_regular_file(path, where="CAS receipt")
        except ActionContractError as exc:
            raise CASTamperError(str(exc)) from exc

    def _validate_receipt(
        self, value: object, *, action: Mapping[str, object]
    ) -> dict[str, object]:
        try:
            receipt = _exact_mapping(value, keys=_RECEIPT_KEYS, where="CAS receipt")
            if receipt["schema"] != CAS_RECEIPT_SCHEMA_V1:
                _fail(f"CAS receipt schema must be {CAS_RECEIPT_SCHEMA_V1!r}")
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
            producer = _validate_producer(
                worker_id=raw_producer["worker_id"],
                platform_key=raw_producer["platform_key"],
                host_class=raw_producer["host_class"],
            )
            body = {
                "schema": CAS_RECEIPT_SCHEMA_V1,
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

    def _verify_blob(self, result: Mapping[str, object]) -> Path:
        digest = str(result["sha256"])
        path = self._blob_path(digest)
        try:
            observed_digest, observed_size = _file_identity(path, where="CAS payload")
        except ActionContractError as exc:
            raise CASTamperError(str(exc)) from exc
        if observed_digest != digest or observed_size != result["bytes"]:
            raise CASTamperError(
                f"CAS payload content differs from receipt: {path}"
            )
        return path

    def lookup(self, action: object) -> dict[str, object] | None:
        """Return a fully revalidated receipt, or ``None`` on a clean miss."""

        normalized = validate_action(action)
        path = self._receipt_path(str(normalized["action_key"]))
        try:
            raw = self._load_receipt_bytes(path)
        except CASTamperError:
            if not path.exists() and not path.is_symlink():
                return None
            raise
        try:
            value = _decode_strict_json(raw, where="CAS receipt")
            receipt = self._validate_receipt(value, action=normalized)
        except ActionContractError as exc:
            raise CASTamperError(str(exc)) from exc
        if raw != _canonical_file_bytes(receipt):
            raise CASTamperError("CAS receipt bytes are not canonical JSON")
        self._verify_blob(receipt["result"])  # type: ignore[arg-type]
        return receipt

    def result_path(self, receipt: object, action: object) -> Path:
        normalized = validate_action(action)
        validated = self._validate_receipt(receipt, action=normalized)
        return self._verify_blob(validated["result"])  # type: ignore[arg-type]

    def publish_result(
        self,
        action: object,
        result_path: str | Path,
        *,
        worker_id: str,
        platform_key: str | None,
        host_class: str | None,
    ) -> tuple[dict[str, object], bool]:
        """Publish a result and return ``(canonical_receipt, won_publication)``.

        A losing deterministic producer must reproduce the winner byte-for-byte;
        a stochastic producer accepts the already-published canonical result.
        """

        normalized = validate_action(action)
        producer = _validate_producer(
            worker_id=worker_id,
            platform_key=platform_key,
            host_class=host_class,
        )
        validate_worker_scope(
            normalized, platform_key=platform_key, host_class=host_class
        )
        staging, digest, size = _copy_to_staging(
            Path(result_path), self.root / ".staging"
        )
        result = {"sha256": digest, "bytes": size}
        try:
            blob_path = self._blob_path(digest)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging, blob_path)
                blob_won = True
            except FileExistsError:
                blob_won = False
            if not blob_won:
                self._verify_blob(result)
            else:
                directory_fd = os.open(blob_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            body: dict[str, object] = {
                "schema": CAS_RECEIPT_SCHEMA_V1,
                "action_key": normalized["action_key"],
                "action_manifest_sha256": canonical_sha256(normalized),
                "result": result,
                "producer": producer,
            }
            candidate = {**body, "receipt_sha256": canonical_sha256(body)}
            receipt_path = self._receipt_path(str(normalized["action_key"]))
            won = _atomic_publish(receipt_path, _canonical_file_bytes(candidate))
            if won:
                return candidate, True
            canonical = self.lookup(normalized)
            if canonical is None:  # pragma: no cover - impossible absent deletion/tamper
                raise CASTamperError("CAS receipt vanished after losing publication race")
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
            try:
                staging.unlink()
            except FileNotFoundError:
                pass


def run_local_action(
    action: object,
    *,
    cas_root: str | Path,
    checkout_root: str | Path,
    worker_id: str,
    platform_key: str | None,
    host_class: str | None,
    timeout_seconds: float | None = None,
    recompute: bool = False,
) -> dict[str, object]:
    """Execute one action locally and publish its declared file result."""

    normalized = validate_action(action)
    validate_worker_scope(
        normalized, platform_key=platform_key, host_class=host_class
    )
    root = Path(checkout_root)
    if not root.is_absolute():
        raise ActionContractError("checkout_root must be absolute")
    verify_code_closure(normalized["code_closure"], root)
    cas = PrismaBuildCAS(cas_root)
    if not recompute:
        cached = cas.lookup(normalized)
        if cached is not None:
            return {
                "status": "cache_hit",
                "receipt": cached,
                "payload_path": str(cas.result_path(cached, normalized)),
            }
    task = normalized["task"]
    environment = normalized["environment"]
    assert isinstance(task, Mapping)
    assert isinstance(environment, Mapping)
    cwd = root if task["working_directory"] == "." else root / str(
        task["working_directory"]
    )
    if not cwd.is_dir():
        raise LocalActionError(f"declared working directory does not exist: {cwd}")
    output = cwd / str(task["result_path"])
    if output.exists() or output.is_symlink():
        raise LocalActionError(
            f"declared result path must be absent before execution: {output}"
        )
    variables = environment["variables"]
    assert isinstance(variables, Mapping)
    try:
        completed = subprocess.run(
            list(task["argv"]),
            cwd=cwd,
            env={str(key): str(value) for key, value in variables.items()},
            shell=False,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalActionError(f"action execution failed: {exc}") from exc
    if completed.returncode != 0:
        raise LocalActionError(
            f"action argv exited with status {completed.returncode}"
        )
    if not output.exists() and not output.is_symlink():
        raise LocalActionError(
            f"action succeeded without its declared result file: {output}"
        )
    receipt, won = cas.publish_result(
        normalized,
        output,
        worker_id=worker_id,
        platform_key=platform_key,
        host_class=host_class,
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
    run = commands.add_parser("run-local")
    run.add_argument("--action", required=True, type=Path)
    run.add_argument("--cas-root", required=True, type=Path)
    run.add_argument("--checkout-root", required=True, type=Path)
    run.add_argument("--worker-id", required=True)
    run.add_argument("--platform-key")
    run.add_argument("--host-class")
    run.add_argument("--timeout-seconds", type=float)
    run.add_argument("--recompute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "seal-closure":
        closure = build_code_closure(args.root, args.file)
        _atomic_write_new_json(args.output, closure)
        print(args.output)
        return 0
    action_data = _read_json_mapping(args.action if hasattr(args, "action") else args.body, where="action")
    if args.command == "seal-action":
        sealed = seal_action(action_data)
        _atomic_write_new_json(args.output, sealed)
        print(args.output)
        return 0
    action = validate_action(action_data)
    if args.command == "key":
        print(action["action_key"])
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
        worker_id=args.worker_id,
        platform_key=args.platform_key,
        host_class=args.host_class,
        timeout_seconds=args.timeout_seconds,
        recompute=args.recompute,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "ACTION_SCHEMA_V1",
    "CAS_RECEIPT_SCHEMA_V1",
    "CODE_CLOSURE_SCHEMA_V1",
    "ActionContractError",
    "CASConflictError",
    "CASTamperError",
    "LocalActionError",
    "PrismaBuildCAS",
    "PrismaBuildError",
    "build_code_closure",
    "main",
    "run_local_action",
    "seal_action",
    "validate_action",
    "validate_code_closure",
    "validate_worker_scope",
    "verify_code_closure",
]


if __name__ == "__main__":
    raise SystemExit(main())
