"""Deterministic, receipt-gated multi-host campaign runner.

Version 1 of PrismaQuant's cluster campaign contract is intentionally a pure,
fixed RTX 4090 state machine.  This module is a separate version-2 contract
and executor.  It does not reinterpret or mutate the v1 schemas.

Every stage is an explicit argv array executed with ``shell=False``.  Local
and SSH stages use the same small worker protocol: the worker takes an exact
JSON request on stdin, acquires a host-local stage lock, verifies any existing
receipt files, runs the argv, and accepts success only when every declared
receipt is a regular file with the manifest-pinned SHA-256.  Transfer and
collation are ordinary stages whose reviewed argv publishes such a receipt;
the coordinator contains no transport-specific artifact semantics.

For commands whose outputs cannot be hashed before a run, ``sealed-stage``
executes an explicit nested argv and atomically publishes a fixed receipt only
after a zero exit.  Its receipt binds both a reviewed token and the nested argv.

The coordinator state is self-hashed, written with fsync + atomic replace,
guarded by flock, and updated by compare-and-swap against the prior state
identity.  Attempts are bounded.  A resumed coordinator monitors only a PID
whose start time and owner token match the durable attempt record; ambiguous
ownership fails closed and is never killed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


CAMPAIGN_MANIFEST_SCHEMA_V2 = "prismaquant.cluster_campaign.manifest.v2"
CAMPAIGN_STATE_SCHEMA_V2 = "prismaquant.cluster_campaign.state.v2"
WORKER_REQUEST_SCHEMA_V2 = "prismaquant.cluster_campaign.worker_request.v2"
SEALED_STAGE_RECEIPT_SCHEMA_V1 = (
    "prismaquant.cluster_campaign.sealed_stage_receipt.v1"
)

_MANIFEST_BODY_KEYS = frozenset(
    {"schema", "campaign_id", "coordinator", "max_parallel", "hosts", "stages"}
)
_MANIFEST_KEYS = _MANIFEST_BODY_KEYS | {"identity_sha256"}
_HOST_KEYS = frozenset({"id", "transport", "work_root"})
_LOCAL_TRANSPORT_KEYS = frozenset({"kind"})
_SSH_TRANSPORT_KEYS = frozenset(
    {
        "kind",
        "host",
        "port",
        "user",
        "known_hosts",
        "ssh_executable",
        "remote_helper_argv",
        "connect_timeout_seconds",
    }
)
_STAGE_KEYS = frozenset(
    {
        "id",
        "host_id",
        "dependencies",
        "argv",
        "cwd",
        "env",
        "receipts",
        "max_attempts",
        "timeout_seconds",
    }
)
_RECEIPT_KEYS = frozenset({"path", "sha256"})
_STATE_BODY_KEYS = frozenset(
    {"schema", "campaign_identity_sha256", "revision", "stages"}
)
_STATE_KEYS = _STATE_BODY_KEYS | {"identity_sha256"}
_STATE_STAGE_KEYS = frozenset({"status", "attempts", "receipts"})
_ATTEMPT_KEYS = frozenset(
    {
        "number",
        "owner_token",
        "status",
        "pid",
        "pid_start_ticks",
        "started_unix_ns",
        "finished_unix_ns",
        "log_path",
        "detail",
    }
)
_WORKER_REQUEST_KEYS = frozenset(
    {
        "schema",
        "owner_token",
        "work_root",
        "stage_id",
        "argv",
        "cwd",
        "env",
        "receipts",
        "lock_path",
        "timeout_seconds",
        "verify_only",
    }
)

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,252}\Z")
_USER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# OpenSSH joins remote command arguments into one command string.  The helper
# bootstrap is therefore deliberately narrower than stage argv: no whitespace
# or shell metacharacters.  The actual arbitrary argv travels as JSON on stdin
# and is executed by the remote Python helper with shell=False.
_REMOTE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:=,+@%-]+\Z")
_SEALED_STAGE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:=,+@%-]{0,255}\Z")
_RESERVED_OWNER_ENV = "PRISMAQUANT_CLUSTER_OWNER"
_RESERVED_LOCK_FD_ENV = "PRISMAQUANT_CLUSTER_LOCK_FD"
_RESERVED_STAGE_ENV = frozenset({_RESERVED_OWNER_ENV, _RESERVED_LOCK_FD_ENV})

_STAGE_STATUSES = frozenset(
    {"pending", "running", "retryable_failed", "succeeded", "terminal_failed"}
)
_ATTEMPT_STATUSES = frozenset({"running", "failed", "succeeded"})

EXIT_REQUEST_INVALID = 70
EXIT_RECEIPT_MISMATCH = 73
EXIT_RECEIPT_MISSING = 74
EXIT_STAGE_FAILED = 75
EXIT_RECEIPT_ABSENT = 76

_TERMINAL_WORKER_CODES = frozenset({EXIT_REQUEST_INVALID, EXIT_RECEIPT_MISMATCH})
_RECOVERY_KILL_GRACE_SECONDS = 5.0


class ClusterCampaignV2Error(RuntimeError):
    """Base error for a v2 campaign contract or execution failure."""


class CampaignContractError(ClusterCampaignV2Error, ValueError):
    """The v2 manifest or state is not exact and self-consistent."""


class CampaignStateConflict(ClusterCampaignV2Error):
    """The durable state changed since the caller's last validated read."""


class CampaignLockBusy(ClusterCampaignV2Error):
    """Another coordinator currently owns the campaign state lock."""


class CampaignTerminalFailure(ClusterCampaignV2Error):
    """One or more stages exhausted retries or failed closed."""


def _fail(message: str) -> None:
    raise CampaignContractError(message)


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    where: str,
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
        _fail(f"{where} must be {'a string' if allow_empty else 'a non-empty string'}")
    if "\x00" in value or any(ord(char) < 32 and char not in "\t\n" for char in value):
        _fail(f"{where} contains a NUL or control character")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(f"{where} has an invalid value")
    return value


def _integer(
    value: object,
    *,
    where: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{where} must be an integer in [{minimum}, {maximum}]")
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
        raise CampaignContractError("value is not finite canonical JSON data") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_absolute_path(value: object, *, where: str) -> str:
    raw = _text(value, where=where)
    path = PurePosixPath(raw)
    if not raw.startswith("/") or raw == "/":
        _fail(f"{where} must be a non-root absolute POSIX path")
    if any(part in {"", ".", ".."} for part in raw.split("/")[1:]):
        _fail(f"{where} must be normalized and traversal-free")
    if str(path) != raw:
        _fail(f"{where} must be normalized and traversal-free")
    return raw


def _normalize_string_array(
    value: object,
    *,
    where: str,
    nonempty: bool,
    token_pattern: re.Pattern[str] | None = None,
) -> list[str]:
    if isinstance(value, (str, bytes)) or type(value) is not list:
        _fail(f"{where} must be an argv-style string array")
    if nonempty and not value:
        _fail(f"{where} must not be empty")
    return [
        _text(item, where=f"{where}[{index}]", pattern=token_pattern)
        for index, item in enumerate(value)
    ]


def _normalize_receipts(value: object, *, where: str) -> list[dict[str, str]]:
    if type(value) is not list or not value:
        _fail(f"{where} must be a non-empty array")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        raw = _exact_mapping(
            item, keys=_RECEIPT_KEYS, where=f"{where}[{index}]"
        )
        rows.append(
            {
                "path": _safe_absolute_path(
                    raw["path"], where=f"{where}[{index}].path"
                ),
                "sha256": _sha256(
                    raw["sha256"], where=f"{where}[{index}].sha256"
                ),
            }
        )
    rows.sort(key=lambda row: row["path"])
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        _fail(f"{where} contains duplicate paths")
    return rows


def _normalize_env(value: object, *, where: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, where=f"{where} key", pattern=_ENV_RE)
        if key in _RESERVED_STAGE_ENV:
            _fail(f"{where} cannot override reserved {key}")
        result[key] = _text(
            raw_value, where=f"{where}.{key}", allow_empty=True
        )
    return dict(sorted(result.items()))


def _normalize_transport(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    kind = value.get("kind")
    if kind == "local":
        _exact_mapping(value, keys=_LOCAL_TRANSPORT_KEYS, where=where)
        return {"kind": "local"}
    if kind != "ssh":
        _fail(f"{where}.kind must be exactly 'local' or 'ssh'")
    raw = _exact_mapping(value, keys=_SSH_TRANSPORT_KEYS, where=where)
    helper = _normalize_string_array(
        raw["remote_helper_argv"],
        where=f"{where}.remote_helper_argv",
        nonempty=True,
        token_pattern=_REMOTE_TOKEN_RE,
    )
    return {
        "kind": "ssh",
        "host": _text(raw["host"], where=f"{where}.host", pattern=_HOST_RE),
        "port": _integer(
            raw["port"], where=f"{where}.port", minimum=1, maximum=65535
        ),
        "user": _text(raw["user"], where=f"{where}.user", pattern=_USER_RE),
        "known_hosts": _safe_absolute_path(
            raw["known_hosts"], where=f"{where}.known_hosts"
        ),
        "ssh_executable": _safe_absolute_path(
            raw["ssh_executable"], where=f"{where}.ssh_executable"
        ),
        "remote_helper_argv": helper,
        "connect_timeout_seconds": _integer(
            raw["connect_timeout_seconds"],
            where=f"{where}.connect_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
    }


def _normalize_host(value: object, *, index: int) -> dict[str, object]:
    where = f"hosts[{index}]"
    raw = _exact_mapping(value, keys=_HOST_KEYS, where=where)
    return {
        "id": _text(raw["id"], where=f"{where}.id", pattern=_ID_RE),
        "transport": _normalize_transport(
            raw["transport"], where=f"{where}.transport"
        ),
        "work_root": _safe_absolute_path(
            raw["work_root"], where=f"{where}.work_root"
        ),
    }


def _normalize_stage(value: object, *, index: int) -> dict[str, object]:
    where = f"stages[{index}]"
    raw = _exact_mapping(value, keys=_STAGE_KEYS, where=where)
    stage_id = _text(raw["id"], where=f"{where}.id", pattern=_ID_RE)
    dependencies = _normalize_string_array(
        raw["dependencies"], where=f"{where}.dependencies", nonempty=False
    )
    for dependency in dependencies:
        if _ID_RE.fullmatch(dependency) is None:
            _fail(f"{where}.dependencies contains an invalid stage id")
    dependencies = sorted(dependencies)
    if len(dependencies) != len(set(dependencies)):
        _fail(f"{where}.dependencies contains duplicates")
    argv = _normalize_string_array(
        raw["argv"], where=f"{where}.argv", nonempty=True
    )
    env = _normalize_env(raw["env"], where=f"{where}.env")
    if not PurePosixPath(argv[0]).is_absolute() and "PATH" not in env:
        _fail(
            f"{where}.argv[0] must be absolute unless the closed stage env "
            "declares PATH"
        )
    if "PATH" in env:
        components = env["PATH"].split(":")
        if not components or any(
            not item or not PurePosixPath(item).is_absolute() for item in components
        ):
            _fail(f"{where}.env.PATH must contain only absolute directories")
    return {
        "id": stage_id,
        "host_id": _text(
            raw["host_id"], where=f"{where}.host_id", pattern=_ID_RE
        ),
        "dependencies": dependencies,
        "argv": argv,
        "cwd": _safe_absolute_path(raw["cwd"], where=f"{where}.cwd"),
        "env": env,
        "receipts": _normalize_receipts(
            raw["receipts"], where=f"{where}.receipts"
        ),
        "max_attempts": _integer(
            raw["max_attempts"],
            where=f"{where}.max_attempts",
            minimum=1,
            maximum=10,
        ),
        "timeout_seconds": _integer(
            raw["timeout_seconds"],
            where=f"{where}.timeout_seconds",
            minimum=1,
            maximum=7 * 24 * 60 * 60,
        ),
    }


def _validate_dag(stages: Sequence[Mapping[str, object]]) -> None:
    by_id = {str(stage["id"]): stage for stage in stages}
    for stage_id, stage in by_id.items():
        dependencies = [str(item) for item in stage["dependencies"]]  # type: ignore[index]
        if stage_id in dependencies:
            _fail(f"stage {stage_id!r} depends on itself")
        unknown = sorted(set(dependencies) - set(by_id))
        if unknown:
            _fail(f"stage {stage_id!r} has unknown dependencies {unknown}")

    remaining = {
        stage_id: set(str(item) for item in stage["dependencies"])  # type: ignore[index]
        for stage_id, stage in by_id.items()
    }
    closed: set[str] = set()
    while remaining:
        frontier = sorted(
            stage_id
            for stage_id, dependencies in remaining.items()
            if dependencies <= closed
        )
        if not frontier:
            _fail(f"campaign stage DAG contains a cycle among {sorted(remaining)}")
        for stage_id in frontier:
            remaining.pop(stage_id)
            closed.add(stage_id)


def _normalize_manifest_body(value: object) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_MANIFEST_BODY_KEYS, where="manifest body")
    if raw["schema"] != CAMPAIGN_MANIFEST_SCHEMA_V2:
        _fail("campaign manifest v2 schema is unsupported")
    raw_hosts = raw["hosts"]
    if type(raw_hosts) is not list or not raw_hosts:
        _fail("hosts must be a non-empty array")
    hosts = [
        _normalize_host(host, index=index) for index, host in enumerate(raw_hosts)
    ]
    hosts.sort(key=lambda row: str(row["id"]))
    host_ids = [str(host["id"]) for host in hosts]
    if len(host_ids) != len(set(host_ids)):
        _fail("hosts contain duplicate ids")
    local_ids = [
        str(host["id"])
        for host in hosts
        if host["transport"]["kind"] == "local"  # type: ignore[index]
    ]
    if len(local_ids) != 1:
        _fail("campaign v2 requires exactly one local coordinator host")
    coordinator = _text(
        raw["coordinator"], where="coordinator", pattern=_ID_RE
    )
    if coordinator != local_ids[0]:
        _fail("coordinator must identify the sole local host")

    raw_stages = raw["stages"]
    if type(raw_stages) is not list or not raw_stages:
        _fail("stages must be a non-empty array")
    stages = [
        _normalize_stage(stage, index=index)
        for index, stage in enumerate(raw_stages)
    ]
    stages.sort(key=lambda row: str(row["id"]))
    stage_ids = [str(stage["id"]) for stage in stages]
    if len(stage_ids) != len(set(stage_ids)):
        _fail("stages contain duplicate ids")
    unknown_hosts = sorted(
        {str(stage["host_id"]) for stage in stages} - set(host_ids)
    )
    if unknown_hosts:
        _fail(f"stages reference unknown hosts {unknown_hosts}")
    all_receipt_paths = [
        str(receipt["path"])
        for stage in stages
        for receipt in stage["receipts"]  # type: ignore[index]
    ]
    if len(all_receipt_paths) != len(set(all_receipt_paths)):
        _fail("receipt paths must be owned by exactly one stage")
    _validate_dag(stages)

    max_parallel = _integer(
        raw["max_parallel"],
        where="max_parallel",
        minimum=1,
        maximum=len(hosts),
    )
    return {
        "schema": CAMPAIGN_MANIFEST_SCHEMA_V2,
        "campaign_id": _text(
            raw["campaign_id"], where="campaign_id", pattern=_ID_RE
        ),
        "coordinator": coordinator,
        "max_parallel": max_parallel,
        "hosts": hosts,
        "stages": stages,
    }


def seal_campaign_manifest_v2(body: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_manifest_body(body)
    return {**normalized, "identity_sha256": canonical_sha256(normalized)}


def validate_campaign_manifest_v2(
    value: Mapping[str, object],
) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_MANIFEST_KEYS, where="campaign manifest")
    body = _normalize_manifest_body(
        {key: raw[key] for key in _MANIFEST_BODY_KEYS}
    )
    identity = _sha256(raw["identity_sha256"], where="manifest.identity_sha256")
    expected = canonical_sha256(body)
    if identity != expected:
        _fail("manifest identity_sha256 differs from its canonical body")
    return {**body, "identity_sha256": identity}


def _decode_strict_json(text: str, *, where: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{where} contains duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        _fail(f"{where} contains non-JSON constant {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except CampaignContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CampaignContractError(f"{where} is not strict JSON") from exc


def load_campaign_manifest_v2(path: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(path)
    try:
        decoded = _decode_strict_json(
            source.read_text(encoding="utf-8"), where=f"manifest {source}"
        )
    except OSError as exc:
        raise CampaignContractError(f"cannot read manifest {source}") from exc
    if not isinstance(decoded, Mapping):
        _fail("campaign manifest root must be an object")
    return validate_campaign_manifest_v2(decoded)


def _owner_token(campaign_identity: str, stage_id: str, attempt: int) -> str:
    encoded = (
        f"{campaign_identity}\0{stage_id}\0{int(attempt)}"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_body(
    *,
    campaign_identity_sha256: str,
    revision: int,
    stages: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": CAMPAIGN_STATE_SCHEMA_V2,
        "campaign_identity_sha256": campaign_identity_sha256,
        "revision": int(revision),
        "stages": dict(sorted(stages.items())),
    }


def _seal_state(body: Mapping[str, object]) -> dict[str, object]:
    canonical = copy.deepcopy(dict(body))
    return {**canonical, "identity_sha256": canonical_sha256(canonical)}


def initial_campaign_state_v2(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    normalized = validate_campaign_manifest_v2(manifest)
    stages = {
        str(stage["id"]): {
            "status": "pending",
            "attempts": [],
            "receipts": [],
        }
        for stage in normalized["stages"]  # type: ignore[index]
    }
    return _seal_state(
        _state_body(
            campaign_identity_sha256=str(normalized["identity_sha256"]),
            revision=0,
            stages=stages,
        )
    )


def _normalize_attempt(
    value: object,
    *,
    where: str,
    campaign_identity: str,
    stage_id: str,
    expected_number: int,
) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_ATTEMPT_KEYS, where=where)
    number = _integer(
        raw["number"], where=f"{where}.number", minimum=1, maximum=10
    )
    if number != expected_number:
        _fail(f"{where}.number is not sequential")
    owner = _sha256(raw["owner_token"], where=f"{where}.owner_token")
    expected_owner = _owner_token(campaign_identity, stage_id, number)
    if owner != expected_owner:
        _fail(f"{where}.owner_token differs from its deterministic owner")
    status_value = _text(raw["status"], where=f"{where}.status")
    if status_value not in _ATTEMPT_STATUSES:
        _fail(f"{where}.status is unsupported")
    pid_raw = raw["pid"]
    pid = None
    if pid_raw is not None:
        pid = _integer(pid_raw, where=f"{where}.pid", minimum=1, maximum=2**31 - 1)
    ticks_raw = raw["pid_start_ticks"]
    ticks = None
    if ticks_raw is not None:
        ticks = _integer(
            ticks_raw,
            where=f"{where}.pid_start_ticks",
            minimum=1,
            maximum=2**63 - 1,
        )
    if (pid is None) != (ticks is None):
        _fail(f"{where}.pid and pid_start_ticks must both be null or integers")
    started = _integer(
        raw["started_unix_ns"],
        where=f"{where}.started_unix_ns",
        minimum=1,
        maximum=2**63 - 1,
    )
    finished_raw = raw["finished_unix_ns"]
    finished = None
    if finished_raw is not None:
        finished = _integer(
            finished_raw,
            where=f"{where}.finished_unix_ns",
            minimum=started,
            maximum=2**63 - 1,
        )
    if status_value == "running" and finished is not None:
        _fail(f"{where}.running attempt cannot have finished_unix_ns")
    if status_value != "running" and finished is None:
        _fail(f"{where}.{status_value} attempt requires finished_unix_ns")
    return {
        "number": number,
        "owner_token": owner,
        "status": status_value,
        "pid": pid,
        "pid_start_ticks": ticks,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "log_path": _safe_absolute_path(
            raw["log_path"], where=f"{where}.log_path"
        ),
        "detail": _text(
            raw["detail"], where=f"{where}.detail", allow_empty=True
        ),
    }


def validate_campaign_state_v2(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    normalized_manifest = validate_campaign_manifest_v2(manifest)
    raw = _exact_mapping(state, keys=_STATE_KEYS, where="campaign state")
    body_raw = {key: raw[key] for key in _STATE_BODY_KEYS}
    if body_raw["schema"] != CAMPAIGN_STATE_SCHEMA_V2:
        _fail("campaign state v2 schema is unsupported")
    campaign_identity = _sha256(
        body_raw["campaign_identity_sha256"],
        where="state.campaign_identity_sha256",
    )
    if campaign_identity != normalized_manifest["identity_sha256"]:
        _fail("campaign state belongs to a different manifest")
    revision = _integer(
        body_raw["revision"],
        where="state.revision",
        minimum=0,
        maximum=2**63 - 1,
    )
    raw_stages = body_raw["stages"]
    if not isinstance(raw_stages, Mapping):
        _fail("state.stages must be an object")
    stage_specs = {
        str(stage["id"]): stage
        for stage in normalized_manifest["stages"]  # type: ignore[index]
    }
    if set(raw_stages) != set(stage_specs):
        _fail(
            "state stage set differs from manifest: "
            f"missing={sorted(set(stage_specs) - set(raw_stages))}, "
            f"extra={sorted(set(raw_stages) - set(stage_specs))}"
        )
    stages: dict[str, object] = {}
    for stage_id in sorted(stage_specs):
        where = f"state.stages.{stage_id}"
        stage_raw = _exact_mapping(
            raw_stages[stage_id], keys=_STATE_STAGE_KEYS, where=where
        )
        status_value = _text(stage_raw["status"], where=f"{where}.status")
        if status_value not in _STAGE_STATUSES:
            _fail(f"{where}.status is unsupported")
        raw_attempts = stage_raw["attempts"]
        if type(raw_attempts) is not list:
            _fail(f"{where}.attempts must be an array")
        max_attempts = int(stage_specs[stage_id]["max_attempts"])
        if len(raw_attempts) > max_attempts:
            _fail(f"{where}.attempts exceeds the manifest bound")
        attempts = [
            _normalize_attempt(
                item,
                where=f"{where}.attempts[{index}]",
                campaign_identity=campaign_identity,
                stage_id=stage_id,
                expected_number=index + 1,
            )
            for index, item in enumerate(raw_attempts)
        ]
        running_attempts = [item for item in attempts if item["status"] == "running"]
        if running_attempts and (
            len(running_attempts) != 1 or running_attempts[0] is not attempts[-1]
        ):
            _fail(f"{where} has a non-terminal running attempt")
        receipts_raw = stage_raw["receipts"]
        receipts = (
            _normalize_receipts(receipts_raw, where=f"{where}.receipts")
            if receipts_raw
            else []
        )
        expected_receipts = stage_specs[stage_id]["receipts"]
        if status_value == "pending":
            if attempts or receipts:
                _fail(f"{where}.pending stage cannot have attempts or receipts")
        elif status_value == "running":
            if not attempts or attempts[-1]["status"] != "running" or receipts:
                _fail(f"{where}.running state is inconsistent")
        elif status_value == "retryable_failed":
            if (
                not attempts
                or attempts[-1]["status"] != "failed"
                or len(attempts) >= max_attempts
                or receipts
            ):
                _fail(f"{where}.retryable_failed state is inconsistent")
        elif status_value == "terminal_failed":
            if not attempts or attempts[-1]["status"] != "failed" or receipts:
                _fail(f"{where}.terminal_failed state is inconsistent")
        elif status_value == "succeeded":
            if (
                not attempts
                or attempts[-1]["status"] != "succeeded"
                or receipts != expected_receipts
            ):
                _fail(f"{where}.succeeded state is inconsistent")
        stages[stage_id] = {
            "status": status_value,
            "attempts": attempts,
            "receipts": receipts,
        }

    # A succeeded stage may only exist after all of its dependencies succeeded.
    for stage_id, spec in stage_specs.items():
        if stages[stage_id]["status"] == "pending":  # type: ignore[index]
            continue
        for dependency in spec["dependencies"]:  # type: ignore[index]
            dep_status = stages[str(dependency)]["status"]  # type: ignore[index]
            if dep_status != "succeeded":
                _fail(
                    f"state advances {stage_id!r} before dependency "
                    f"{dependency!r} succeeded"
                )

    body = _state_body(
        campaign_identity_sha256=campaign_identity,
        revision=revision,
        stages=stages,
    )
    identity = _sha256(raw["identity_sha256"], where="state.identity_sha256")
    if identity != canonical_sha256(body):
        _fail("state identity_sha256 differs from its canonical body")
    return {**body, "identity_sha256": identity}


class _StateStore:
    def __init__(self, path: Path) -> None:
        # ``resolve`` would follow a pre-existing state symlink before the
        # no-follow checks below could see it.  Make the path absolute without
        # resolving any component so an aliased final path still fails closed.
        self.path = Path(os.path.abspath(os.fspath(path)))
        if not self.path.is_absolute() or self.path == Path("/"):
            raise CampaignContractError("state path must be a non-root absolute path")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock_fd: int | None = None

    def __enter__(self) -> "_StateStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise CampaignLockBusy(
                f"cannot open coordinator lock {self.lock_path}: {exc}"
            ) from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise CampaignLockBusy(
                    f"coordinator lock is not a regular file: {self.lock_path}"
                )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignLockBusy(
                    f"another coordinator holds {self.lock_path}"
                ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._lock_fd = fd
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def _read_raw(self) -> Mapping[str, object] | None:
        if self.path.is_symlink():
            raise CampaignContractError(f"state path is a symlink: {self.path}")
        if not self.path.exists():
            return None
        try:
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise CampaignContractError(
                    f"state path is not a regular file: {self.path}"
                )
            decoded = _decode_strict_json(
                self.path.read_text(encoding="utf-8"),
                where=f"campaign state {self.path}",
            )
        except OSError as exc:
            raise CampaignContractError(f"cannot read state {self.path}") from exc
        if not isinstance(decoded, Mapping):
            raise CampaignContractError("campaign state root must be an object")
        return decoded

    def load(
        self, manifest: Mapping[str, object]
    ) -> dict[str, object] | None:
        raw = self._read_raw()
        return None if raw is None else validate_campaign_state_v2(raw, manifest)

    def write(
        self,
        state: Mapping[str, object],
        manifest: Mapping[str, object],
        *,
        expected_identity: str | None,
    ) -> dict[str, object]:
        validated = validate_campaign_state_v2(state, manifest)
        current_raw = self._read_raw()
        if current_raw is None:
            current_identity = None
        else:
            current = validate_campaign_state_v2(current_raw, manifest)
            current_identity = str(current["identity_sha256"])
        if current_identity != expected_identity:
            raise CampaignStateConflict(
                "campaign state compare-and-swap failed: "
                f"expected={expected_identity!r}, observed={current_identity!r}"
            )
        encoded = json.dumps(
            validated,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_raw)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        observed = self.load(manifest)
        if observed != validated:
            raise CampaignStateConflict("published campaign state differs after write")
        return validated


def _advance_state(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    store: _StateStore,
    mutator,
) -> dict[str, object]:
    current = validate_campaign_state_v2(state, manifest)
    body = {
        key: copy.deepcopy(current[key]) for key in _STATE_BODY_KEYS
    }
    mutator(body)
    body["revision"] = int(body["revision"]) + 1
    sealed = _seal_state(body)
    return store.write(
        sealed,
        manifest,
        expected_identity=str(current["identity_sha256"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _receipt_observation(
    receipts: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    for receipt in receipts:
        path = Path(str(receipt["path"]))
        try:
            info = path.lstat()
        except FileNotFoundError:
            missing.append(str(path))
            continue
        except OSError:
            mismatched.append(str(path))
            continue
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            mismatched.append(str(path))
            continue
        try:
            observed = _file_sha256(path)
        except OSError:
            mismatched.append(str(path))
            continue
        if observed != str(receipt["sha256"]):
            mismatched.append(str(path))
    return missing, mismatched


def _normalize_worker_request(value: object) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_WORKER_REQUEST_KEYS, where="worker request")
    if raw["schema"] != WORKER_REQUEST_SCHEMA_V2:
        _fail("worker request schema is unsupported")
    owner = _sha256(raw["owner_token"], where="worker.owner_token")
    stage_id = _text(raw["stage_id"], where="worker.stage_id", pattern=_ID_RE)
    argv = _normalize_string_array(
        raw["argv"], where="worker.argv", nonempty=True
    )
    env = _normalize_env(raw["env"], where="worker.env")
    if not PurePosixPath(argv[0]).is_absolute() and "PATH" not in env:
        _fail("worker argv[0] is relative but the closed env has no PATH")
    work_root = _safe_absolute_path(raw["work_root"], where="worker.work_root")
    lock_path = _safe_absolute_path(raw["lock_path"], where="worker.lock_path")
    try:
        PurePosixPath(lock_path).relative_to(PurePosixPath(work_root))
    except ValueError as exc:
        raise CampaignContractError(
            "worker lock path is outside the host work root"
        ) from exc
    return {
        "schema": WORKER_REQUEST_SCHEMA_V2,
        "owner_token": owner,
        "work_root": work_root,
        "stage_id": stage_id,
        "argv": argv,
        "cwd": _safe_absolute_path(raw["cwd"], where="worker.cwd"),
        "env": env,
        "receipts": _normalize_receipts(raw["receipts"], where="worker.receipts"),
        "lock_path": lock_path,
        "timeout_seconds": _integer(
            raw["timeout_seconds"],
            where="worker.timeout_seconds",
            minimum=1,
            maximum=7 * 24 * 60 * 60,
        ),
        "verify_only": (
            raw["verify_only"]
            if type(raw["verify_only"]) is bool
            else _fail("worker.verify_only must be boolean")
        ),
    }


def _worker_lock(path: Path, work_root: Path, *, timeout_seconds: int) -> int:
    if work_root.is_symlink():
        raise CampaignContractError(f"worker work root is a symlink: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    resolved_root = work_root.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise CampaignContractError("worker lock parent escapes work root") from exc
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise CampaignContractError("worker lock is not a regular file")
    deadline = time.monotonic() + int(timeout_seconds)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(f"timed out acquiring worker lock: {path}")
            time.sleep(0.05)
    return fd


def _terminate_child_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_RECOVERY_KILL_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_RECOVERY_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def execute_worker_request(value: Mapping[str, object]) -> int:
    """Execute one already-validated host request and return a stable exit code."""

    request = _normalize_worker_request(value)
    work_root = Path(str(request["work_root"]))
    lock_path = Path(str(request["lock_path"]))
    try:
        lock_fd = _worker_lock(
            lock_path,
            work_root,
            timeout_seconds=int(request["timeout_seconds"]),
        )
    except TimeoutError as exc:
        print(f"[cluster-worker] {exc}", file=sys.stderr, flush=True)
        return EXIT_STAGE_FAILED
    process: subprocess.Popen[Any] | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum, _frame) -> None:
        del signum
        if process is not None:
            _terminate_child_process_group(process)
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

        missing, mismatched = _receipt_observation(request["receipts"])  # type: ignore[arg-type]
        if mismatched:
            print(
                "[cluster-worker] receipt mismatch; refusing execution: "
                f"{mismatched[:8]}",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_RECEIPT_MISMATCH
        if not missing:
            print(
                f"[cluster-worker] {request['stage_id']}: receipts already exact",
                flush=True,
            )
            return 0
        if request["verify_only"]:
            return EXIT_RECEIPT_ABSENT

        cwd = Path(str(request["cwd"]))
        if cwd.is_symlink() or not cwd.is_dir():
            print(
                f"[cluster-worker] invalid cwd: {cwd}",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_REQUEST_INVALID
        child_env = dict(request["env"])  # type: ignore[arg-type]
        child_env[_RESERVED_OWNER_ENV] = str(request["owner_token"])
        child_env[_RESERVED_LOCK_FD_ENV] = str(lock_fd)
        print(
            f"[cluster-worker] {request['stage_id']}: exec argv={request['argv']!r}",
            flush=True,
        )
        process = subprocess.Popen(
            list(request["argv"]),  # type: ignore[arg-type]
            cwd=str(cwd),
            env=child_env,
            shell=False,
            start_new_session=True,
            # If the worker is killed abruptly, the stage process keeps the
            # host-local lock.  Recovery probes/retries therefore cannot run a
            # duplicate argv while an orphaned child is still materializing.
            pass_fds=(lock_fd,),
        )
        try:
            returncode = process.wait(timeout=int(request["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            print(
                f"[cluster-worker] {request['stage_id']}: timeout",
                file=sys.stderr,
                flush=True,
            )
            _terminate_child_process_group(process)
            return EXIT_STAGE_FAILED
        if returncode != 0:
            print(
                f"[cluster-worker] {request['stage_id']}: argv exited {returncode}",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_STAGE_FAILED
        missing, mismatched = _receipt_observation(request["receipts"])  # type: ignore[arg-type]
        if mismatched:
            print(
                "[cluster-worker] postflight receipt mismatch: "
                f"{mismatched[:8]}",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_RECEIPT_MISMATCH
        if missing:
            print(
                f"[cluster-worker] postflight receipts missing: {missing[:8]}",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_RECEIPT_MISSING
        return 0
    except KeyboardInterrupt:
        return EXIT_STAGE_FAILED
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _worker_request(
    manifest: Mapping[str, object],
    host: Mapping[str, object],
    stage: Mapping[str, object],
    *,
    owner_token: str,
    verify_only: bool,
) -> dict[str, object]:
    work_root = str(host["work_root"])
    lock_path = (
        PurePosixPath(work_root)
        / ".prismaquant-cluster"
        / str(manifest["identity_sha256"])
        / "locks"
        / f"{stage['id']}.lock"
    )
    request = {
        "schema": WORKER_REQUEST_SCHEMA_V2,
        "owner_token": owner_token,
        "work_root": work_root,
        "stage_id": str(stage["id"]),
        "argv": list(stage["argv"]),
        "cwd": str(stage["cwd"]),
        "env": dict(stage["env"]),
        "receipts": copy.deepcopy(stage["receipts"]),
        "lock_path": str(lock_path),
        "timeout_seconds": int(stage["timeout_seconds"]),
        "verify_only": bool(verify_only),
    }
    return _normalize_worker_request(request)


def _real_regular_file(path: Path, *, executable: bool = False) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CampaignContractError(f"required file is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise CampaignContractError(f"required file is not one regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise CampaignContractError(f"required executable is not executable: {path}")


def _helper_argv(host: Mapping[str, object]) -> list[str]:
    transport = host["transport"]
    if transport["kind"] == "local":  # type: ignore[index]
        # Execute this self-contained module by absolute path.  Importing it
        # through ``prismaquant.__init__`` would initialize Transformers and
        # Torch for every orchestration transition even though the worker has
        # no ML dependency.
        return [sys.executable, str(Path(__file__).resolve()), "_exec-request"]
    ssh = transport  # type: ignore[assignment]
    executable = Path(str(ssh["ssh_executable"]))
    known_hosts = Path(str(ssh["known_hosts"]))
    _real_regular_file(executable, executable=True)
    _real_regular_file(known_hosts)
    target = f"{ssh['user']}@{ssh['host']}"
    return [
        str(executable),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        f"ConnectTimeout={int(ssh['connect_timeout_seconds'])}",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(int(ssh["port"])),
        "--",
        target,
        *list(ssh["remote_helper_argv"]),
    ]


def _proc_snapshot(pid: int, *, strict: bool = False) -> tuple[int, str] | None:
    """Read start time/state; strict checks distinguish missing from unreadable."""
    stat_path = Path(f"/proc/{int(pid)}/stat")
    try:
        text = stat_path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError:
        if strict:
            raise
        return None
    right = text.rfind(")")
    if right < 0:
        if strict:
            raise ValueError("malformed process stat")
        return None
    fields = text[right + 2 :].split()
    if len(fields) <= 19:
        if strict:
            raise ValueError("incomplete process stat")
        return None
    state = fields[0]
    try:
        ticks = int(fields[19])
    except ValueError:
        if strict:
            raise
        return None
    return ticks, state


def _proc_has_owner(pid: int, owner_token: str) -> bool:
    try:
        data = Path(f"/proc/{int(pid)}/environ").read_bytes()
    except OSError:
        return False
    expected = f"{_RESERVED_OWNER_ENV}={owner_token}".encode("utf-8")
    return expected in data.split(b"\0")


def _owned_process_state(pid: int, ticks: int, owner_token: str) -> str:
    try:
        snapshot = _proc_snapshot(pid, strict=True)
        if snapshot is None:
            return "gone"
        if snapshot[0] != ticks:
            return "mismatch"
        if snapshot[1] == "Z":
            return "gone"
        if _proc_has_owner(pid, owner_token):
            return "owned"
        # Exit can empty/remove environ after stat still showed a live helper.
        # Only disappearance or a zombie with the same start time proves this
        # attempt ended. PID reuse and unreadable state remain ambiguous.
        snapshot = _proc_snapshot(pid, strict=True)
    except (OSError, ValueError):
        return "mismatch"
    if snapshot is None or snapshot == (ticks, "Z"):
        return "gone"
    return "mismatch"


def _terminate_recorded_owned_process(pid: int, ticks: int, owner_token: str) -> bool:
    if _owned_process_state(pid, ticks, owner_token) != "owned":
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + _RECOVERY_KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _owned_process_state(pid, ticks, owner_token) == "gone":
            return True
        time.sleep(0.05)
    if _owned_process_state(pid, ticks, owner_token) != "owned":
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return True


def _attempt_log_path(state_path: Path, stage_id: str, attempt: int) -> Path:
    return (
        state_path.parent
        / f"{state_path.name}.logs"
        / stage_id
        / f"attempt-{attempt:04d}.log"
    ).resolve(strict=False)


class _Launch:
    def __init__(
        self,
        *,
        stage_id: str,
        process: subprocess.Popen[Any],
        owner_token: str,
    ) -> None:
        self.stage_id = stage_id
        self.process = process
        self.owner_token = owner_token


def _start_attempt(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    store: _StateStore,
    state_path: Path,
    stage: Mapping[str, object],
    host: Mapping[str, object],
) -> tuple[dict[str, object], _Launch | None]:
    stage_id = str(stage["id"])
    current_stage = state["stages"][stage_id]  # type: ignore[index]
    attempt_number = len(current_stage["attempts"]) + 1  # type: ignore[index]
    owner = _owner_token(
        str(manifest["identity_sha256"]), stage_id, attempt_number
    )
    log_path = _attempt_log_path(state_path, stage_id, attempt_number)

    def add_intent(body: dict[str, object]) -> None:
        record = body["stages"][stage_id]  # type: ignore[index]
        record["status"] = "running"
        record["attempts"].append(  # type: ignore[index]
            {
                "number": attempt_number,
                "owner_token": owner,
                "status": "running",
                "pid": None,
                "pid_start_ticks": None,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "log_path": str(log_path),
                "detail": "",
            }
        )

    state = _advance_state(state, manifest, store, add_intent)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("xb")
    except Exception as exc:
        return _finish_attempt(
            state,
            manifest,
            store,
            stage,
            success=False,
            terminal=True,
            detail=f"cannot create unique attempt log: {exc}",
        ), None

    request = _worker_request(
        manifest, host, stage, owner_token=owner, verify_only=False
    )
    process_env = dict(os.environ)
    process_env[_RESERVED_OWNER_ENV] = owner
    try:
        process = subprocess.Popen(
            _helper_argv(host),
            stdin=subprocess.PIPE,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=process_env,
            shell=False,
            start_new_session=True,
        )
    except Exception as exc:
        log_handle.close()
        return _finish_attempt(
            state,
            manifest,
            store,
            stage,
            success=False,
            terminal=False,
            detail=f"helper launch failed: {exc}",
        ), None
    finally:
        if not log_handle.closed:
            log_handle.close()

    snapshot = _proc_snapshot(process.pid)
    if snapshot is None:
        # This is a child we just created, so it is safe to stop even though the
        # durable PID/start-time tuple could not be captured.  Never leave an
        # unrecorded helper blocked on its stdin pipe.
        _terminate_child_process_group(process)
        return _finish_attempt(
            state,
            manifest,
            store,
            stage,
            success=False,
            terminal=False,
            detail="helper exited before ownership could be recorded",
        ), None
    start_ticks = snapshot[0]

    def bind_process(body: dict[str, object]) -> None:
        attempt = body["stages"][stage_id]["attempts"][-1]  # type: ignore[index]
        attempt["pid"] = int(process.pid)
        attempt["pid_start_ticks"] = int(start_ticks)

    try:
        state = _advance_state(state, manifest, store, bind_process)
    except BaseException:
        _terminate_recorded_owned_process(process.pid, start_ticks, owner)
        raise
    try:
        assert process.stdin is not None
        process.stdin.write(_canonical_bytes(request) + b"\n")
        process.stdin.flush()
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    return state, _Launch(stage_id=stage_id, process=process, owner_token=owner)


def _finish_attempt(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    store: _StateStore,
    stage: Mapping[str, object],
    *,
    success: bool,
    terminal: bool,
    detail: str,
) -> dict[str, object]:
    stage_id = str(stage["id"])

    def mutate(body: dict[str, object]) -> None:
        record = body["stages"][stage_id]  # type: ignore[index]
        attempt = record["attempts"][-1]
        if attempt["status"] != "running":
            raise CampaignContractError(
                f"cannot finish non-running attempt for {stage_id}"
            )
        attempt["status"] = "succeeded" if success else "failed"
        attempt["finished_unix_ns"] = max(
            time.time_ns(), int(attempt["started_unix_ns"])
        )
        attempt["detail"] = str(detail)
        if success:
            record["status"] = "succeeded"
            record["receipts"] = copy.deepcopy(stage["receipts"])
        else:
            exhausted = len(record["attempts"]) >= int(stage["max_attempts"])
            record["status"] = (
                "terminal_failed" if terminal or exhausted else "retryable_failed"
            )
            record["receipts"] = []

    return _advance_state(state, manifest, store, mutate)


def _probe_stage_receipts(
    manifest: Mapping[str, object],
    host: Mapping[str, object],
    stage: Mapping[str, object],
    *,
    owner_token: str,
    log_path: Path,
) -> int:
    probe_owner = hashlib.sha256((owner_token + "\0probe").encode("utf-8")).hexdigest()
    request = _worker_request(
        manifest, host, stage, owner_token=probe_owner, verify_only=True
    )
    process_env = dict(os.environ)
    process_env[_RESERVED_OWNER_ENV] = probe_owner
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                _helper_argv(host),
                stdin=subprocess.PIPE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=process_env,
                shell=False,
                start_new_session=True,
            )
            try:
                encoded = _canonical_bytes(request) + b"\n"
                stdout, _ = process.communicate(
                    encoded,
                    timeout=int(stage["timeout_seconds"])
                    + int(_RECOVERY_KILL_GRACE_SECONDS),
                )
                del stdout
            except subprocess.TimeoutExpired:
                _terminate_child_process_group(process)
                return EXIT_STAGE_FAILED
            return int(process.returncode)
    except OSError:
        return EXIT_STAGE_FAILED


def _recover_running_stages(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    store: _StateStore,
    *,
    poll_interval: float,
) -> dict[str, object]:
    stage_specs = {
        str(stage["id"]): stage
        for stage in manifest["stages"]  # type: ignore[index]
    }
    hosts = {
        str(host["id"]): host for host in manifest["hosts"]  # type: ignore[index]
    }
    for stage_id in sorted(stage_specs):
        record = state["stages"][stage_id]  # type: ignore[index]
        if record["status"] != "running":  # type: ignore[index]
            continue
        stage = stage_specs[stage_id]
        host = hosts[str(stage["host_id"])]
        attempt = record["attempts"][-1]  # type: ignore[index]
        pid = attempt["pid"]
        ticks = attempt["pid_start_ticks"]
        owner = str(attempt["owner_token"])
        detail = "coordinator recovery"
        terminal = False
        if pid is not None and ticks is not None:
            process_state = _owned_process_state(int(pid), int(ticks), owner)
            if process_state == "mismatch":
                state = _finish_attempt(
                    state,
                    manifest,
                    store,
                    stage,
                    success=False,
                    terminal=True,
                    detail="recorded PID ownership is ambiguous; process was not killed",
                )
                continue
            deadline_ns = int(attempt["started_unix_ns"]) + int(
                stage["timeout_seconds"] + _RECOVERY_KILL_GRACE_SECONDS
            ) * 1_000_000_000
            while process_state == "owned" and time.time_ns() < deadline_ns:
                time.sleep(max(0.01, float(poll_interval)))
                process_state = _owned_process_state(int(pid), int(ticks), owner)
            if process_state == "owned":
                killed = _terminate_recorded_owned_process(
                    int(pid), int(ticks), owner
                )
                if not killed:
                    state = _finish_attempt(
                        state,
                        manifest,
                        store,
                        stage,
                        success=False,
                        terminal=True,
                        detail="timed-out process ownership changed; process was not killed",
                    )
                    continue
                detail = "timed-out recorded owned helper was terminated during recovery"
        else:
            detail = "coordinator stopped before helper ownership was recorded"

        log_path = Path(str(attempt["log_path"]))
        probe_code = _probe_stage_receipts(
            manifest,
            host,
            stage,
            owner_token=owner,
            log_path=log_path,
        )
        if probe_code == 0:
            state = _finish_attempt(
                state,
                manifest,
                store,
                stage,
                success=True,
                terminal=False,
                detail=detail + "; exact receipts recovered",
            )
        else:
            terminal = probe_code in _TERMINAL_WORKER_CODES
            state = _finish_attempt(
                state,
                manifest,
                store,
                stage,
                success=False,
                terminal=terminal,
                detail=detail + f"; receipt probe exited {probe_code}",
            )
    return state


def _verify_succeeded_stage_receipts(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    stage_specs = {
        str(stage["id"]): stage
        for stage in manifest["stages"]  # type: ignore[index]
    }
    hosts = {
        str(host["id"]): host for host in manifest["hosts"]  # type: ignore[index]
    }
    failures: list[str] = []
    for stage_id in sorted(stage_specs):
        record = state["stages"][stage_id]  # type: ignore[index]
        if record["status"] != "succeeded":  # type: ignore[index]
            continue
        stage = stage_specs[stage_id]
        attempt = record["attempts"][-1]  # type: ignore[index]
        result = _probe_stage_receipts(
            manifest,
            hosts[str(stage["host_id"])],
            stage,
            owner_token=str(attempt["owner_token"]),
            log_path=Path(str(attempt["log_path"])),
        )
        if result != 0:
            failures.append(f"{stage_id}:{result}")
    if failures:
        raise CampaignTerminalFailure(
            "durable succeeded-stage receipts failed revalidation: "
            + ", ".join(failures)
        )


def _ready_stages(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    occupied_hosts: set[str],
    limit: int,
) -> list[Mapping[str, object]]:
    records = state["stages"]  # type: ignore[assignment]
    ready: list[Mapping[str, object]] = []
    used_hosts = set(occupied_hosts)
    if limit <= 0:
        return ready
    for stage in manifest["stages"]:  # type: ignore[index]
        stage_id = str(stage["id"])
        status_value = records[stage_id]["status"]  # type: ignore[index]
        if status_value not in {"pending", "retryable_failed"}:
            continue
        if not all(
            records[str(dependency)]["status"] == "succeeded"  # type: ignore[index]
            for dependency in stage["dependencies"]  # type: ignore[index]
        ):
            continue
        host_id = str(stage["host_id"])
        if host_id in used_hosts:
            continue
        ready.append(stage)
        used_hosts.add(host_id)
        if len(ready) >= limit:
            break
    return ready


def _terminal_stage_ids(state: Mapping[str, object]) -> list[str]:
    return sorted(
        str(stage_id)
        for stage_id, record in state["stages"].items()  # type: ignore[union-attr]
        if record["status"] == "terminal_failed"
    )


def _stop_active_after_terminal_failure(
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    store: _StateStore,
    active: dict[str, _Launch],
    stage_specs: Mapping[str, Mapping[str, object]],
    hosts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    for stage_id in sorted(list(active)):
        launch = active.pop(stage_id)
        stage = stage_specs[stage_id]
        host = hosts[str(stage["host_id"])]
        _terminate_child_process_group(launch.process)
        attempt = state["stages"][stage_id]["attempts"][-1]  # type: ignore[index]
        probe_code = _probe_stage_receipts(
            manifest,
            host,
            stage,
            owner_token=launch.owner_token,
            log_path=Path(str(attempt["log_path"])),
        )
        if probe_code == 0:
            state = _finish_attempt(
                state,
                manifest,
                store,
                stage,
                success=True,
                terminal=False,
                detail="campaign failed elsewhere; exact receipts recovered",
            )
        else:
            state = _finish_attempt(
                state,
                manifest,
                store,
                stage,
                success=False,
                terminal=True,
                detail=(
                    "campaign failed elsewhere; owned helper stopped; "
                    f"receipt probe exited {probe_code}"
                ),
            )
    return state


def run_campaign_v2(
    manifest: Mapping[str, object],
    state_path: str | os.PathLike[str],
    *,
    poll_interval: float = 0.2,
) -> dict[str, object]:
    """Run or resume a sealed v2 campaign until success or terminal failure."""

    normalized = validate_campaign_manifest_v2(manifest)
    path = Path(os.path.abspath(os.fspath(state_path)))
    hosts = {
        str(host["id"]): host for host in normalized["hosts"]  # type: ignore[index]
    }
    stage_specs = {
        str(stage["id"]): stage
        for stage in normalized["stages"]  # type: ignore[index]
    }
    with _StateStore(path) as store:
        state = store.load(normalized)
        if state is None:
            initial = initial_campaign_state_v2(normalized)
            state = store.write(
                initial, normalized, expected_identity=None
            )
        state = _recover_running_stages(
            state,
            normalized,
            store,
            poll_interval=poll_interval,
        )
        # A state hash proves what the coordinator observed, not that remote
        # receipt files still exist.  Revalidate prior successes before any
        # dependent work is allowed to start.
        _verify_succeeded_stage_receipts(state, normalized)
        active: dict[str, _Launch] = {}
        while True:
            terminal = _terminal_stage_ids(state)
            if terminal:
                if active:
                    state = _stop_active_after_terminal_failure(
                        state,
                        normalized,
                        store,
                        active,
                        stage_specs,
                        hosts,
                    )
                    terminal = _terminal_stage_ids(state)
                raise CampaignTerminalFailure(
                    f"campaign has terminal failed stages: {terminal}"
                )
            if all(
                record["status"] == "succeeded"
                for record in state["stages"].values()  # type: ignore[union-attr]
            ):
                _verify_succeeded_stage_receipts(state, normalized)
                return validate_campaign_state_v2(state, normalized)

            occupied_hosts = {
                str(stage_specs[stage_id]["host_id"]) for stage_id in active
            }
            ready = _ready_stages(
                state,
                normalized,
                occupied_hosts=occupied_hosts,
                limit=int(normalized["max_parallel"]) - len(active),
            )
            for stage in ready:
                host = hosts[str(stage["host_id"])]
                state, launch = _start_attempt(
                    state, normalized, store, path, stage, host
                )
                if launch is not None:
                    active[launch.stage_id] = launch
            if not active:
                if ready:
                    # Every launch failed before it produced an active helper;
                    # the updated retry state determines the next transition.
                    continue
                raise CampaignTerminalFailure(
                    "campaign has no ready or active stages; state is deadlocked"
                )

            completed: list[tuple[str, int, bool]] = []
            while not completed:
                for stage_id, launch in sorted(active.items()):
                    returncode = launch.process.poll()
                    if returncode is not None:
                        completed.append((stage_id, int(returncode), False))
                        continue
                    attempt = state["stages"][stage_id]["attempts"][-1]  # type: ignore[index]
                    stage = stage_specs[stage_id]
                    deadline_ns = int(attempt["started_unix_ns"]) + (
                        int(stage["timeout_seconds"])
                        + int(_RECOVERY_KILL_GRACE_SECONDS)
                    ) * 1_000_000_000
                    if time.time_ns() >= deadline_ns:
                        # ``Popen.poll`` still owns this unreaped child PID, so
                        # it cannot have been recycled.  Stop the exact helper
                        # group; the worker's signal handler stops its stage
                        # child group before releasing the host-local lock.
                        _terminate_child_process_group(launch.process)
                        returncode = launch.process.poll()
                        completed.append(
                            (
                                stage_id,
                                EXIT_STAGE_FAILED
                                if returncode is None
                                else int(returncode),
                                True,
                            )
                        )
                if not completed:
                    time.sleep(max(0.01, float(poll_interval)))
            for stage_id, returncode, coordinator_timeout in completed:
                launch = active.pop(stage_id)
                stage = stage_specs[stage_id]
                host = hosts[str(stage["host_id"])]
                if returncode == 0:
                    state = _finish_attempt(
                        state,
                        normalized,
                        store,
                        stage,
                        success=True,
                        terminal=False,
                        detail="worker exited successfully with exact receipts",
                    )
                    continue
                attempt = state["stages"][stage_id]["attempts"][-1]  # type: ignore[index]
                probe_code = _probe_stage_receipts(
                    normalized,
                    host,
                    stage,
                    owner_token=launch.owner_token,
                    log_path=Path(str(attempt["log_path"])),
                )
                if probe_code == 0:
                    state = _finish_attempt(
                        state,
                        normalized,
                        store,
                        stage,
                        success=True,
                        terminal=False,
                        detail=(
                            (
                                "coordinator watchdog expired; "
                                if coordinator_timeout
                                else ""
                            )
                            + f"worker exited {returncode}; exact receipts recovered"
                        ),
                    )
                else:
                    terminal_failure = (
                        returncode in _TERMINAL_WORKER_CODES
                        or probe_code in _TERMINAL_WORKER_CODES
                    )
                    state = _finish_attempt(
                        state,
                        normalized,
                        store,
                        stage,
                        success=False,
                        terminal=terminal_failure,
                        detail=(
                            (
                                "coordinator watchdog expired; "
                                if coordinator_timeout
                                else ""
                            )
                            + f"worker exited {returncode}; receipt probe exited "
                            + f"{probe_code}"
                        ),
                    )


def _atomic_write_new_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CampaignContractError(f"output already exists: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CampaignContractError(f"output appeared concurrently: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_new_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_new_bytes(path, encoded)


def _normalize_sealed_stage(
    token: object,
    child_argv: object,
) -> tuple[str, list[str]]:
    normalized_token = _text(
        token,
        where="sealed stage token",
        pattern=_SEALED_STAGE_TOKEN_RE,
    )
    if isinstance(child_argv, (str, bytes)) or not isinstance(
        child_argv, Sequence
    ):
        _fail("sealed stage child argv must be a string sequence")
    normalized_argv = _normalize_string_array(
        list(child_argv),
        where="sealed stage child argv",
        nonempty=True,
    )
    if not PurePosixPath(normalized_argv[0]).is_absolute():
        _fail("sealed stage child argv[0] must be absolute")
    return normalized_token, normalized_argv


def sealed_stage_receipt_bytes(token: str, child_argv: Sequence[str]) -> bytes:
    """Return the fixed receipt bytes for one explicit nested argv.

    The child argv is bound into the receipt so changing a command requires a
    newly reviewed manifest receipt hash even when an operator accidentally
    reuses the human-readable token.
    """

    normalized_token, normalized_argv = _normalize_sealed_stage(token, child_argv)
    receipt = {
        "schema": SEALED_STAGE_RECEIPT_SCHEMA_V1,
        "token": normalized_token,
        "child_argv_sha256": canonical_sha256(normalized_argv),
    }
    return _canonical_bytes(receipt) + b"\n"


def sealed_stage_receipt_sha256(token: str, child_argv: Sequence[str]) -> str:
    """Return the manifest-known SHA-256 for a sealed-stage receipt."""

    return hashlib.sha256(sealed_stage_receipt_bytes(token, child_argv)).hexdigest()


def run_sealed_stage(
    *,
    receipt_path: str | os.PathLike[str],
    token: str,
    child_argv: Sequence[str],
) -> int:
    """Run an argv without a shell and publish its fixed receipt on success.

    This wrapper is for stages whose materialized products do not have known
    hashes before execution.  The nested command remains responsible for
    validating and durably publishing those products before returning zero.
    A nonzero child exit never publishes the completion receipt.
    """

    normalized_token, normalized_argv = _normalize_sealed_stage(token, child_argv)
    path = Path(
        _safe_absolute_path(str(receipt_path), where="sealed stage receipt path")
    )
    expected = sealed_stage_receipt_bytes(normalized_token, normalized_argv)
    try:
        info = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise CampaignContractError(
            f"cannot inspect sealed stage receipt: {path}"
        ) from exc
    else:
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise CampaignContractError(
                f"sealed stage receipt is not one regular file: {path}"
            )
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise CampaignContractError(
                f"cannot read sealed stage receipt: {path}"
            ) from exc
        if existing != expected:
            raise CampaignContractError(
                f"sealed stage receipt already exists with wrong content: {path}"
            )
        return 0

    # Do not create a session here.  When launched by the campaign worker the
    # wrapper is already the leader of an owned process group, and the nested
    # child must remain in that group for bounded timeout and recovery kills.
    inherited_lock_fds: tuple[int, ...] = ()
    inherited_lock_raw = os.environ.get(_RESERVED_LOCK_FD_ENV)
    if inherited_lock_raw is not None:
        try:
            inherited_lock_fd = int(inherited_lock_raw)
            if inherited_lock_fd < 3:
                raise ValueError
            os.fstat(inherited_lock_fd)
        except (OSError, ValueError) as exc:
            raise CampaignContractError(
                "sealed stage inherited an invalid campaign lock fd"
            ) from exc
        inherited_lock_fds = (inherited_lock_fd,)
    completed = subprocess.run(
        normalized_argv,
        shell=False,
        check=False,
        pass_fds=inherited_lock_fds,
    )
    if completed.returncode != 0:
        print(
            f"[sealed-stage] child argv exited {completed.returncode}; "
            "completion receipt withheld",
            file=sys.stderr,
            flush=True,
        )
        return int(completed.returncode) if completed.returncode > 0 else 1
    _atomic_write_new_bytes(path, expected)
    return 0


def _read_json_mapping(path: Path, *, where: str) -> Mapping[str, object]:
    try:
        value = _decode_strict_json(path.read_text(encoding="utf-8"), where=where)
    except OSError as exc:
        raise CampaignContractError(f"cannot read {where}: {path}") from exc
    if not isinstance(value, Mapping):
        raise CampaignContractError(f"{where} root must be an object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-manifest")
    seal.add_argument("--body", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--state", required=True, type=Path)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--manifest", required=True, type=Path)
    status_parser.add_argument("--state", required=True, type=Path)
    sealed = subparsers.add_parser(
        "sealed-stage",
        help="run a nested argv and atomically publish its fixed receipt",
    )
    sealed.add_argument("--receipt", required=True, type=Path)
    sealed.add_argument("--token", required=True)
    sealed.add_argument("child_argv", nargs=argparse.REMAINDER)
    sealed_hash = subparsers.add_parser(
        "sealed-stage-receipt-sha256",
        help="print the fixed receipt SHA-256 for an explicit nested argv",
    )
    sealed_hash.add_argument("--token", required=True)
    sealed_hash.add_argument("child_argv", nargs=argparse.REMAINDER)
    subparsers.add_parser("_exec-request", help=argparse.SUPPRESS)
    return parser


def _nested_argv(value: Sequence[str]) -> list[str]:
    nested = list(value)
    if nested and nested[0] == "--":
        nested.pop(0)
    if not nested:
        raise CampaignContractError("sealed stage requires a nested argv after --")
    return nested


def _worker_main() -> int:
    try:
        text = sys.stdin.read()
        value = _decode_strict_json(text, where="worker request stdin")
        if not isinstance(value, Mapping):
            raise CampaignContractError("worker request root must be an object")
        return execute_worker_request(value)
    except CampaignContractError as exc:
        print(f"[cluster-worker] invalid request: {exc}", file=sys.stderr, flush=True)
        return EXIT_REQUEST_INVALID
    except BaseException as exc:
        print(f"[cluster-worker] fatal: {exc}", file=sys.stderr, flush=True)
        return EXIT_STAGE_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "_exec-request":
        return _worker_main()
    if args.command == "sealed-stage":
        return run_sealed_stage(
            receipt_path=args.receipt,
            token=args.token,
            child_argv=_nested_argv(args.child_argv),
        )
    if args.command == "sealed-stage-receipt-sha256":
        print(
            sealed_stage_receipt_sha256(
                args.token,
                _nested_argv(args.child_argv),
            )
        )
        return 0
    if args.command == "seal-manifest":
        body = _read_json_mapping(args.body, where="manifest body")
        sealed = seal_campaign_manifest_v2(body)
        _atomic_write_new_json(args.output, sealed)
        print(args.output)
        return 0
    manifest = load_campaign_manifest_v2(args.manifest)
    if args.command == "run":
        state = run_campaign_v2(manifest, args.state)
        print(json.dumps(state, sort_keys=True))
        return 0
    with _StateStore(args.state) as store:
        state = store.load(manifest)
    if state is None:
        raise CampaignContractError(f"campaign state does not exist: {args.state}")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CAMPAIGN_MANIFEST_SCHEMA_V2",
    "CAMPAIGN_STATE_SCHEMA_V2",
    "SEALED_STAGE_RECEIPT_SCHEMA_V1",
    "WORKER_REQUEST_SCHEMA_V2",
    "CampaignContractError",
    "CampaignLockBusy",
    "CampaignStateConflict",
    "CampaignTerminalFailure",
    "ClusterCampaignV2Error",
    "canonical_sha256",
    "execute_worker_request",
    "initial_campaign_state_v2",
    "load_campaign_manifest_v2",
    "main",
    "run_campaign_v2",
    "run_sealed_stage",
    "sealed_stage_receipt_bytes",
    "sealed_stage_receipt_sha256",
    "seal_campaign_manifest_v2",
    "validate_campaign_manifest_v2",
    "validate_campaign_state_v2",
]


if __name__ == "__main__":
    raise SystemExit(main())
