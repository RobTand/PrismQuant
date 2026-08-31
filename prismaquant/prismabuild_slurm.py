"""Fail-closed SLURM transport for immutable PrismaBuild actions.

SLURM is only the resource layer.  It may report that an allocation completed,
but a PrismaBuild action is successful only after :class:`PrismaBuildCAS`
returns a fully verified receipt for that exact action.  This module therefore
does not maintain a second result database or infer success from scheduler
state.

The adapter intentionally uses a local argv array, ``shell=False``,
``--export=NIL``, and a small closed submit environment.  Slurm's positional
batch-script copy would relocate the Python launcher, so the sealed worker argv
is POSIX-quoted into one ``--wrap=exec`` argument.  The remote worker receives
one content-addressed, immutable action request and invokes the ordinary
``prismabuild run-local`` command; task argv never enters shell text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
import time
from typing import Literal

from . import prismabuild as pb


_SLURM_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_SCOPE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/-]{0,255}\Z")
_CLUSTER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TIME_RE = re.compile(r"(?:(?:[0-9]+)-)?(?:[0-9]{1,2}):[0-5][0-9]:[0-5][0-9]\Z")
_JOB_ID_RE = re.compile(
    r"(?P<number>[1-9][0-9]{0,19})(?:;(?P<cluster>[A-Za-z0-9][A-Za-z0-9._-]{0,63}))?\Z"
)
_CANCELLED_RE = re.compile(r"CANCELLED(?: by [0-9]+)?\Z")

SLURM_SUBMISSION_INTENT_SCHEMA_V1 = (
    "prismaquant.prismabuild.slurm_submission_intent.v1"
)
SLURM_SUBMISSION_INTENT_SCHEMA_V2 = (
    "prismaquant.prismabuild.slurm_submission_intent.v2"
)
SLURM_JOB_BINDING_SCHEMA_V1 = "prismaquant.prismabuild.slurm_job_binding.v1"
SLURM_RETRY_TRANSITION_SCHEMA_V1 = (
    "prismaquant.prismabuild.slurm_retry_transition.v1"
)
SLURM_REQUEUE_OBSERVED_SCHEMA_V1 = (
    "prismaquant.prismabuild.slurm_requeue_observed.v1"
)
SLURM_MUTATION_SCHEMA_V1 = "prismaquant.prismabuild.slurm_mutation.v1"
_SLURM_SUBMIT_SPEC_SCHEMA_V2 = "prismaquant.prismabuild.slurm_submit_spec.v2"
_SLURM_RUNTIME_SCHEMA_V1 = "prismaquant.prismabuild.slurm_runtime.v1"
_SUBMISSION_NAMESPACE = "v2"
_SUBMIT_SPEC_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "cluster",
        "cas_root",
        "log_root",
        "worker_script",
        "worker_argv",
        "checkout_root",
        "commands",
        "resources",
        "placement",
        "retry_policy",
        "recompute",
        "submit_environment",
        "runtime",
    }
)
_SLURM_RUNTIME_KEYS = frozenset(
    {"schema", "adapter", "worker_launcher", "runtime_sha256"}
)
_COMMAND_KEYS = frozenset({"sbatch", "squeue", "sacct", "scancel", "scontrol"})
_RESOURCE_KEYS = frozenset(
    {
        "cpus",
        "memory_mib",
        "gpus",
        "constraint",
        "partition",
        "account",
        "qos",
        "time_limit",
    }
)
_PLACEMENT_KEYS = frozenset({"platform_key", "host_class"})
_RETRY_POLICY_KEYS = frozenset(
    {"max_polls", "max_requeues", "poll_interval_seconds"}
)
_INTENT_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "submission_key",
        "job_name",
        "comment",
        "submit_spec",
        "intent_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "submission_key",
        "job_id",
        "binding_sha256",
    }
)
_RETRY_TRANSITION_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "submission_key",
        "job_id",
        "kind",
        "attempt",
        "ordinal",
        "claimed_unix_ns",
        "transition_sha256",
    }
)
_REQUEUE_OBSERVED_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "submission_key",
        "job_id",
        "ordinal",
        "observed_sha256",
    }
)
_MUTATION_KEYS = frozenset(
    {
        "schema",
        "action_key",
        "submission_key",
        "job_id",
        "kind",
        "ordinal",
        "mutation_sha256",
    }
)
_COUNTER_FILENAME_RE = re.compile(r"(?P<ordinal>[0-9]{8})\.json\Z")
_POLL_FILENAME_RE = re.compile(
    r"(?P<attempt>[0-9]{8})-(?P<ordinal>[0-9]{8})\.json\Z"
)
_SACCT_ADOPTION_FORMAT = (
    "JobIDRaw%64,Cluster%64,JobName%128,Comment%256,State%64"
)
_SACCT_STATE_FORMAT = "JobIDRaw%64,State%64,JobName%128,Comment%256"
_MAX_DURABLE_COUNTER = 99_999_999
_MAX_POLL_INTERVAL_SECONDS = 86_400.0
_MAX_STALE_TEMP_ENTRIES = 64
_MAX_UNIX_NS = (1 << 63) - 1
_MAX_STATE_BYTES = 16 * 1024 * 1024

# Capture the adapter implementation Python loaded, before any adapter method
# accepts an action or inspects durable scheduler state. The sealed submit spec
# carries this identity alongside the configured worker launcher identity.
_LOADED_SLURM_ADAPTER_IDENTITY = pb._identify_runtime_source(
    Path(__file__).resolve(), where="PrismaBuild SLURM adapter"
)

_PENDING_STATES = frozenset(
    {
        "CONFIGURING",
        "EXPEDITING",
        "PENDING",
        "POWER_UP_NODE",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
        "UPDATE_DB",
    }
)
_RUNNING_STATES = frozenset(
    {
        "COMPLETING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
)
_FAILED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "LAUNCH_FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "RECONFIG_FAIL",
        "REVOKED",
        "TIMEOUT",
    }
)


class SlurmAdapterError(pb.PrismaBuildError):
    """Base class for SLURM transport failures."""


class SlurmUnavailableError(SlurmAdapterError):
    """A configured SLURM executable is absent or cannot be invoked."""


class SlurmCommandError(SlurmAdapterError):
    """A SLURM command returned a non-zero exit status."""


class SlurmProtocolError(SlurmAdapterError):
    """SLURM returned unknown, malformed, or ambiguous protocol output."""


class SlurmAdoptionError(SlurmProtocolError):
    """A durable submission cannot yet be uniquely and safely adopted."""


def _absolute_path(value: str | Path, *, where: str, root_ok: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute() or (not root_ok and path == Path("/")):
        raise pb.ActionContractError(f"{where} must be a non-root absolute path")
    if ".." in path.parts:
        raise pb.ActionContractError(f"{where} must not contain parent traversal")
    return path


def _token(value: object, *, where: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise pb.ActionContractError(f"{where} has an invalid value")
    return value


def _positive_integer(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise pb.ActionContractError(f"{where} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise pb.ActionContractError(f"{where} must be a non-negative integer")
    return value


def _positive_finite(value: object, *, where: str) -> float:
    if type(value) not in {int, float}:
        raise pb.ActionContractError(f"{where} must be a positive finite number")
    normalized = float(value)
    if normalized <= 0 or not math.isfinite(normalized):
        raise pb.ActionContractError(f"{where} must be a positive finite number")
    return normalized


def _poll_interval(value: object, *, where: str) -> float:
    normalized = _positive_finite(value, where=where)
    if normalized > _MAX_POLL_INTERVAL_SECONDS:
        raise pb.ActionContractError(
            f"{where} must not exceed {_MAX_POLL_INTERVAL_SECONDS:g} seconds"
        )
    return normalized


def _bounded_counter(value: object, *, where: str, positive: bool) -> int:
    normalized = (
        _positive_integer(value, where=where)
        if positive
        else _nonnegative_integer(value, where=where)
    )
    if normalized > _MAX_DURABLE_COUNTER:
        raise pb.ActionContractError(
            f"{where} exceeds the durable eight-digit counter bound"
        )
    return normalized


@dataclass(frozen=True)
class SlurmResources:
    """Every placement-affecting resource is explicit and argv-safe."""

    cpus: int
    memory_mib: int
    gpus: int
    constraint: str
    partition: str
    account: str
    qos: str
    time_limit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpus", _positive_integer(self.cpus, where="cpus"))
        object.__setattr__(
            self,
            "memory_mib",
            _positive_integer(self.memory_mib, where="memory_mib"),
        )
        object.__setattr__(self, "gpus", _nonnegative_integer(self.gpus, where="gpus"))
        for name in ("constraint", "partition", "account", "qos"):
            object.__setattr__(
                self,
                name,
                _token(getattr(self, name), where=name, pattern=_SLURM_TOKEN_RE),
            )
        object.__setattr__(
            self,
            "time_limit",
            _token(self.time_limit, where="time_limit", pattern=_TIME_RE),
        )


@dataclass(frozen=True)
class SlurmPlacement:
    """Scheduler placement intent; never accepted as worker attestation."""

    platform_key: str | None
    host_class: str | None

    def __post_init__(self) -> None:
        for name in ("platform_key", "host_class"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _token(value, where=name, pattern=_SCOPE_TOKEN_RE),
                )


@dataclass(frozen=True)
class SlurmJobId:
    number: int
    cluster: str | None = None

    def __post_init__(self) -> None:
        if type(self.number) is not int or self.number <= 0:
            raise pb.ActionContractError("SLURM job number must be a positive integer")
        if self.cluster is not None:
            object.__setattr__(
                self,
                "cluster",
                _token(self.cluster, where="SLURM cluster", pattern=_CLUSTER_RE),
            )

    def __str__(self) -> str:
        suffix = f";{self.cluster}" if self.cluster is not None else ""
        return f"{self.number}{suffix}"


@dataclass(frozen=True)
class SlurmSubmission:
    status: Literal["cache_hit", "submitted", "adopted"]
    action_key: str
    job_id: SlurmJobId | None
    receipt: Mapping[str, object] | None
    submission_key: str | None = None


@dataclass(frozen=True)
class SlurmResolution:
    status: Literal["succeeded", "pending", "running", "cancelled", "failed"]
    action_key: str
    job_id: SlurmJobId
    slurm_state: str | None
    reason: str
    receipt: Mapping[str, object] | None
    payload_path: Path | None


@dataclass(frozen=True)
class SlurmRetryProgress:
    """Durable retry-budget state for one action/allocation binding."""

    polls: int
    requeues: int
    latest_requeue_observed: bool


def parse_job_id(value: str) -> SlurmJobId:
    """Parse the exact output of ``sbatch --parsable``."""

    match = _JOB_ID_RE.fullmatch(value)
    if match is None:
        raise SlurmProtocolError(f"malformed sbatch job id: {value!r}")
    cluster = match.group("cluster")
    return SlurmJobId(number=int(match.group("number")), cluster=cluster)


def _exact_mapping(
    value: object, *, keys: frozenset[str], where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise pb.ActionContractError(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        raise pb.ActionContractError(f"{where} keys must be strings")
    actual = set(value)
    if actual != set(keys):
        raise pb.ActionContractError(
            f"{where} fields differ: missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )
    return value


def _sha256(value: object, *, where: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise pb.ActionContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _path_string(value: object, *, where: str, root_ok: bool = False) -> str:
    if type(value) is not str:
        raise pb.ActionContractError(f"{where} must be an absolute path string")
    return str(_absolute_path(value, where=where, root_ok=root_ok))


def _validate_slurm_runtime(value: object) -> dict[str, object]:
    raw = _exact_mapping(
        value, keys=_SLURM_RUNTIME_KEYS, where="SLURM submit spec.runtime"
    )
    if raw["schema"] != _SLURM_RUNTIME_SCHEMA_V1:
        raise pb.ActionContractError(
            f"SLURM runtime schema must be {_SLURM_RUNTIME_SCHEMA_V1!r}"
        )
    adapter = pb._normalize_runtime_source(
        raw["adapter"], where="SLURM runtime.adapter"
    )
    worker_launcher = pb._normalize_runtime_source(
        raw["worker_launcher"], where="SLURM runtime.worker_launcher"
    )
    body: dict[str, object] = {
        "schema": _SLURM_RUNTIME_SCHEMA_V1,
        "adapter": adapter,
        "worker_launcher": worker_launcher,
    }
    digest = _sha256(
        raw["runtime_sha256"], where="SLURM runtime.runtime_sha256"
    )
    if digest != pb.canonical_sha256(body):
        raise pb.ActionContractError(
            "SLURM runtime digest does not match its source identities"
        )
    return {**body, "runtime_sha256": digest}


def _validate_submit_spec(value: object) -> dict[str, object]:
    raw = _exact_mapping(value, keys=_SUBMIT_SPEC_KEYS, where="SLURM submit spec")
    if raw["schema"] != _SLURM_SUBMIT_SPEC_SCHEMA_V2:
        raise pb.ActionContractError(
            f"SLURM submit spec schema must be {_SLURM_SUBMIT_SPEC_SCHEMA_V2!r}"
        )
    action_key = _sha256(raw["action_key"], where="SLURM submit spec.action_key")
    cluster = _token(
        raw["cluster"], where="SLURM submit spec.cluster", pattern=_CLUSTER_RE
    )
    commands_raw = _exact_mapping(
        raw["commands"], keys=_COMMAND_KEYS, where="SLURM submit spec.commands"
    )
    commands = {
        name: _path_string(
            commands_raw[name], where=f"SLURM submit spec.commands.{name}", root_ok=True
        )
        for name in sorted(_COMMAND_KEYS)
    }
    resources_raw = _exact_mapping(
        raw["resources"], keys=_RESOURCE_KEYS, where="SLURM submit spec.resources"
    )
    resources = SlurmResources(**resources_raw)  # type: ignore[arg-type]
    placement_raw = _exact_mapping(
        raw["placement"], keys=_PLACEMENT_KEYS, where="SLURM submit spec.placement"
    )
    placement = SlurmPlacement(**placement_raw)  # type: ignore[arg-type]
    retry_policy_raw = _exact_mapping(
        raw["retry_policy"],
        keys=_RETRY_POLICY_KEYS,
        where="SLURM submit spec.retry_policy",
    )
    max_polls = _bounded_counter(
        retry_policy_raw["max_polls"],
        where="SLURM submit spec.retry_policy.max_polls",
        positive=True,
    )
    max_requeues = _bounded_counter(
        retry_policy_raw["max_requeues"],
        where="SLURM submit spec.retry_policy.max_requeues",
        positive=False,
    )
    if max_requeues != 0:
        # Slurm's Requeue flag controls explicit ``scontrol requeue`` and
        # automatic/admin restart together.  Until a worker can bind Restarts
        # to an authorized durable claim, a positive budget is unsafe.
        raise pb.ActionContractError(
            "SLURM same-job requeue is disabled; max_requeues must be zero"
        )
    poll_interval_seconds = _poll_interval(
        retry_policy_raw["poll_interval_seconds"],
        where="SLURM submit spec.retry_policy.poll_interval_seconds",
    )
    if type(raw["recompute"]) is not bool:
        raise pb.ActionContractError("SLURM submit spec.recompute must be a boolean")
    if raw["recompute"]:
        raise pb.ActionContractError(
            "SLURM recompute is disabled because receipts have one canonical lineage"
        )
    environment_raw = raw["submit_environment"]
    if not isinstance(environment_raw, Mapping) or any(
        type(key) is not str for key in environment_raw
    ):
        raise pb.ActionContractError(
            "SLURM submit spec.submit_environment must be an object"
        )
    environment: dict[str, str] = {}
    for raw_name, raw_value in environment_raw.items():
        name = _token(
            raw_name, where="SLURM submit environment name", pattern=_ENV_NAME_RE
        )
        if type(raw_value) is not str or "\x00" in raw_value:
            raise pb.ActionContractError(
                f"SLURM submit environment value for {name!r} is invalid"
            )
        environment[name] = raw_value
    worker_script = _path_string(
        raw["worker_script"],
        where="SLURM submit spec.worker_script",
        root_ok=True,
    )
    cas_root = _path_string(raw["cas_root"], where="SLURM submit spec.cas_root")
    checkout_root = _path_string(
        raw["checkout_root"], where="SLURM submit spec.checkout_root"
    )
    expected_worker_argv = [
        worker_script,
        "run-local",
        "--require-slurm-initial-start",
        "--action",
        str(
            Path(cas_root)
            / "requests"
            / action_key[:2]
            / f"{action_key}.json"
        ),
        "--cas-root",
        cas_root,
        "--checkout-root",
        checkout_root,
    ]
    if raw["worker_argv"] != expected_worker_argv:
        raise pb.ActionContractError(
            "SLURM submit spec.worker_argv is not the exact canonical worker launch"
        )
    runtime = _validate_slurm_runtime(raw["runtime"])
    worker_launcher = runtime["worker_launcher"]
    assert isinstance(worker_launcher, Mapping)
    if worker_launcher["path"] != worker_script:
        raise pb.ActionContractError(
            "SLURM runtime worker launcher differs from the configured worker script"
        )
    return {
        "schema": _SLURM_SUBMIT_SPEC_SCHEMA_V2,
        "action_key": action_key,
        "cluster": cluster,
        "cas_root": cas_root,
        "log_root": _path_string(raw["log_root"], where="SLURM submit spec.log_root"),
        "worker_script": worker_script,
        "worker_argv": expected_worker_argv,
        "checkout_root": checkout_root,
        "commands": commands,
        "resources": {
            "cpus": resources.cpus,
            "memory_mib": resources.memory_mib,
            "gpus": resources.gpus,
            "constraint": resources.constraint,
            "partition": resources.partition,
            "account": resources.account,
            "qos": resources.qos,
            "time_limit": resources.time_limit,
        },
        "placement": {
            "platform_key": placement.platform_key,
            "host_class": placement.host_class,
        },
        "retry_policy": {
            "max_polls": max_polls,
            "max_requeues": max_requeues,
            "poll_interval_seconds": poll_interval_seconds,
        },
        "recompute": raw["recompute"],
        "submit_environment": dict(sorted(environment.items())),
        "runtime": runtime,
    }


def _validate_submission_intent(
    value: object, *, expected_action_key: str | None = None
) -> dict[str, object]:
    try:
        raw = _exact_mapping(value, keys=_INTENT_KEYS, where="SLURM submission intent")
        if raw["schema"] != SLURM_SUBMISSION_INTENT_SCHEMA_V2:
            raise pb.ActionContractError(
                "SLURM submission intent has an unsupported schema"
            )
        action_key = _sha256(
            raw["action_key"], where="SLURM submission intent.action_key"
        )
        if expected_action_key is not None and action_key != expected_action_key:
            raise pb.ActionContractError(
                "SLURM submission intent action key differs from its CAS address"
            )
        submit_spec = _validate_submit_spec(raw["submit_spec"])
        if submit_spec["action_key"] != action_key:
            raise pb.ActionContractError(
                "SLURM submission intent submit spec names a different action"
            )
        submission_key = _sha256(
            raw["submission_key"], where="SLURM submission intent.submission_key"
        )
        if submission_key != pb.canonical_sha256(submit_spec):
            raise pb.ActionContractError(
                "SLURM submission key differs from the sealed submit spec"
            )
        job_name = _token(
            raw["job_name"],
            where="SLURM submission intent.job_name",
            pattern=_SLURM_TOKEN_RE,
        )
        comment = _token(
            raw["comment"],
            where="SLURM submission intent.comment",
            pattern=_SCOPE_TOKEN_RE,
        )
        if job_name != f"pqb-{submission_key}":
            raise pb.ActionContractError(
                "SLURM submission intent job name is not derived from its key"
            )
        if comment != f"prismabuild:{submission_key}":
            raise pb.ActionContractError(
                "SLURM submission intent comment is not derived from its key"
            )
        body: dict[str, object] = {
            "schema": SLURM_SUBMISSION_INTENT_SCHEMA_V2,
            "action_key": action_key,
            "submission_key": submission_key,
            "job_name": job_name,
            "comment": comment,
            "submit_spec": submit_spec,
        }
        digest = _sha256(
            raw["intent_sha256"], where="SLURM submission intent.intent_sha256"
        )
        if digest != pb.canonical_sha256(body):
            raise pb.ActionContractError(
                "SLURM submission intent digest does not match its body"
            )
        return {**body, "intent_sha256": digest}
    except pb.ActionContractError as exc:
        raise pb.CASTamperError(str(exc)) from exc


def _validate_job_binding(
    value: object, *, intent: Mapping[str, object]
) -> dict[str, object]:
    try:
        raw = _exact_mapping(value, keys=_BINDING_KEYS, where="SLURM job binding")
        if raw["schema"] != SLURM_JOB_BINDING_SCHEMA_V1:
            raise pb.ActionContractError("SLURM job binding has an unsupported schema")
        action_key = _sha256(raw["action_key"], where="SLURM job binding.action_key")
        submission_key = _sha256(
            raw["submission_key"], where="SLURM job binding.submission_key"
        )
        if action_key != intent["action_key"] or submission_key != intent["submission_key"]:
            raise pb.ActionContractError(
                "SLURM job binding differs from its durable submission intent"
            )
        if type(raw["job_id"]) is not str:
            raise pb.ActionContractError("SLURM job binding.job_id must be a string")
        parsed_job_id = parse_job_id(raw["job_id"])
        submit_spec = intent["submit_spec"]
        assert isinstance(submit_spec, Mapping)
        if parsed_job_id.cluster != submit_spec["cluster"]:
            raise pb.ActionContractError(
                "SLURM job binding differs from the sealed cluster"
            )
        job_id = str(parsed_job_id)
        body: dict[str, object] = {
            "schema": SLURM_JOB_BINDING_SCHEMA_V1,
            "action_key": action_key,
            "submission_key": submission_key,
            "job_id": job_id,
        }
        digest = _sha256(
            raw["binding_sha256"], where="SLURM job binding.binding_sha256"
        )
        if digest != pb.canonical_sha256(body):
            raise pb.ActionContractError("SLURM job binding digest does not match its body")
        return {**body, "binding_sha256": digest}
    except (pb.ActionContractError, SlurmProtocolError) as exc:
        raise pb.CASTamperError(str(exc)) from exc


def _validate_retry_transition(
    value: object,
    *,
    intent: Mapping[str, object],
    binding: Mapping[str, object],
    expected_kind: Literal["poll", "requeue"],
    expected_attempt: int,
    expected_ordinal: int,
) -> dict[str, object]:
    try:
        raw = _exact_mapping(
            value, keys=_RETRY_TRANSITION_KEYS, where="SLURM retry transition"
        )
        if raw["schema"] != SLURM_RETRY_TRANSITION_SCHEMA_V1:
            raise pb.ActionContractError(
                "SLURM retry transition has an unsupported schema"
            )
        action_key = _sha256(
            raw["action_key"], where="SLURM retry transition.action_key"
        )
        submission_key = _sha256(
            raw["submission_key"], where="SLURM retry transition.submission_key"
        )
        if (
            action_key != intent["action_key"]
            or submission_key != intent["submission_key"]
        ):
            raise pb.ActionContractError(
                "SLURM retry transition differs from its submission intent"
            )
        if type(raw["job_id"]) is not str:
            raise pb.ActionContractError(
                "SLURM retry transition.job_id must be a string"
            )
        job_id = str(parse_job_id(raw["job_id"]))
        if job_id != binding["job_id"]:
            raise pb.ActionContractError(
                "SLURM retry transition differs from its job binding"
            )
        kind = raw["kind"]
        if kind != expected_kind:
            raise pb.ActionContractError(
                "SLURM retry transition kind differs from its counter directory"
            )
        attempt = _nonnegative_integer(
            raw["attempt"], where="SLURM retry transition.attempt"
        )
        if attempt != expected_attempt:
            raise pb.ActionContractError(
                "SLURM retry transition attempt differs from its CAS filename"
            )
        ordinal = _positive_integer(
            raw["ordinal"], where="SLURM retry transition.ordinal"
        )
        if ordinal != expected_ordinal:
            raise pb.ActionContractError(
                "SLURM retry transition ordinal differs from its CAS filename"
            )
        claimed_unix_ns = _nonnegative_integer(
            raw["claimed_unix_ns"],
            where="SLURM retry transition.claimed_unix_ns",
        )
        if claimed_unix_ns > _MAX_UNIX_NS:
            raise pb.ActionContractError(
                "SLURM retry transition.claimed_unix_ns exceeds the signed "
                "64-bit durable timestamp bound"
            )
        body: dict[str, object] = {
            "schema": SLURM_RETRY_TRANSITION_SCHEMA_V1,
            "action_key": action_key,
            "submission_key": submission_key,
            "job_id": job_id,
            "kind": kind,
            "attempt": attempt,
            "ordinal": ordinal,
            "claimed_unix_ns": claimed_unix_ns,
        }
        digest = _sha256(
            raw["transition_sha256"],
            where="SLURM retry transition.transition_sha256",
        )
        if digest != pb.canonical_sha256(body):
            raise pb.ActionContractError(
                "SLURM retry transition digest does not match its body"
            )
        return {**body, "transition_sha256": digest}
    except (pb.ActionContractError, SlurmProtocolError) as exc:
        raise pb.CASTamperError(str(exc)) from exc


def _validate_requeue_observed(
    value: object,
    *,
    intent: Mapping[str, object],
    binding: Mapping[str, object],
    expected_ordinal: int,
) -> dict[str, object]:
    try:
        raw = _exact_mapping(
            value, keys=_REQUEUE_OBSERVED_KEYS, where="SLURM requeue observation"
        )
        if raw["schema"] != SLURM_REQUEUE_OBSERVED_SCHEMA_V1:
            raise pb.ActionContractError(
                "SLURM requeue observation has an unsupported schema"
            )
        action_key = _sha256(
            raw["action_key"], where="SLURM requeue observation.action_key"
        )
        submission_key = _sha256(
            raw["submission_key"],
            where="SLURM requeue observation.submission_key",
        )
        if (
            action_key != intent["action_key"]
            or submission_key != intent["submission_key"]
        ):
            raise pb.ActionContractError(
                "SLURM requeue observation differs from its submission intent"
            )
        if type(raw["job_id"]) is not str:
            raise pb.ActionContractError(
                "SLURM requeue observation.job_id must be a string"
            )
        job_id = str(parse_job_id(raw["job_id"]))
        if job_id != binding["job_id"]:
            raise pb.ActionContractError(
                "SLURM requeue observation differs from its job binding"
            )
        ordinal = _positive_integer(
            raw["ordinal"], where="SLURM requeue observation.ordinal"
        )
        if ordinal != expected_ordinal:
            raise pb.ActionContractError(
                "SLURM requeue observation ordinal differs from its CAS filename"
            )
        body: dict[str, object] = {
            "schema": SLURM_REQUEUE_OBSERVED_SCHEMA_V1,
            "action_key": action_key,
            "submission_key": submission_key,
            "job_id": job_id,
            "ordinal": ordinal,
        }
        digest = _sha256(
            raw["observed_sha256"],
            where="SLURM requeue observation.observed_sha256",
        )
        if digest != pb.canonical_sha256(body):
            raise pb.ActionContractError(
                "SLURM requeue observation digest does not match its body"
            )
        return {**body, "observed_sha256": digest}
    except (pb.ActionContractError, SlurmProtocolError) as exc:
        raise pb.CASTamperError(str(exc)) from exc


def _validate_mutation(
    value: object,
    *,
    intent: Mapping[str, object],
    binding: Mapping[str, object],
    expected_ordinal: int,
) -> dict[str, object]:
    """Validate one append-only scheduler-mutation authorization claim."""

    try:
        raw = _exact_mapping(value, keys=_MUTATION_KEYS, where="SLURM mutation")
        if raw["schema"] != SLURM_MUTATION_SCHEMA_V1:
            raise pb.ActionContractError("SLURM mutation has an unsupported schema")
        action_key = _sha256(raw["action_key"], where="SLURM mutation.action_key")
        submission_key = _sha256(
            raw["submission_key"], where="SLURM mutation.submission_key"
        )
        if (
            action_key != intent["action_key"]
            or submission_key != intent["submission_key"]
        ):
            raise pb.ActionContractError(
                "SLURM mutation differs from its submission intent"
            )
        if type(raw["job_id"]) is not str:
            raise pb.ActionContractError("SLURM mutation.job_id must be a string")
        job_id = str(parse_job_id(raw["job_id"]))
        if job_id != binding["job_id"]:
            raise pb.ActionContractError("SLURM mutation differs from its job binding")
        kind = raw["kind"]
        if kind not in {"cancel", "requeue"}:
            raise pb.ActionContractError("SLURM mutation.kind is unsupported")
        ordinal = _positive_integer(raw["ordinal"], where="SLURM mutation.ordinal")
        if ordinal != expected_ordinal:
            raise pb.ActionContractError(
                "SLURM mutation ordinal differs from its CAS filename"
            )
        body: dict[str, object] = {
            "schema": SLURM_MUTATION_SCHEMA_V1,
            "action_key": action_key,
            "submission_key": submission_key,
            "job_id": job_id,
            "kind": kind,
            "ordinal": ordinal,
        }
        digest = _sha256(
            raw["mutation_sha256"], where="SLURM mutation.mutation_sha256"
        )
        if digest != pb.canonical_sha256(body):
            raise pb.ActionContractError("SLURM mutation digest does not match its body")
        return {**body, "mutation_sha256": digest}
    except (pb.ActionContractError, SlurmProtocolError) as exc:
        raise pb.CASTamperError(str(exc)) from exc


def _validated_receipt(
    cas: pb.PrismaBuildCAS, action: Mapping[str, object]
) -> Mapping[str, object] | None:
    return cas.lookup(action)


def validate_placement_scope(
    action: Mapping[str, object],
    *,
    placement: SlurmPlacement,
    resources: SlurmResources,
) -> None:
    scope = action["execution_scope"]
    assert isinstance(scope, Mapping)
    portability = scope["portability"]
    if (
        portability == "platform_keyed"
        and placement.platform_key != scope["platform_key"]
    ):
        raise pb.ActionContractError(
            "scheduler platform_key does not match the action execution scope"
        )
    if portability == "host_class_keyed":
        expected = scope["host_class"]
        if placement.host_class != expected:
            raise pb.ActionContractError(
                "scheduler host_class does not match the action execution scope"
            )
        if expected not in {resources.partition, resources.constraint}:
            raise pb.ActionContractError(
                "host_class_keyed action must select a matching SLURM partition "
                "or constraint"
            )


def publish_action_request(action: object, *, cas_root: str | Path) -> Path:
    """Publish and verify the canonical immutable request used by the worker."""

    root = _absolute_path(cas_root, where="CAS root")
    normalized = pb.validate_action(action)
    action_key = str(normalized["action_key"])
    _ensure_real_directory(
        root / "requests" / action_key[:2],
        root=root,
        where="PrismaBuild action request",
    )
    request_path = root / "requests" / action_key[:2] / f"{action_key}.json"
    raw = pb._canonical_file_bytes(normalized)
    _atomic_publish_nofollow(
        request_path,
        raw,
        where="PrismaBuild action request",
    )
    observed = SlurmAdapter._read_state_bytes(
        request_path, where="PrismaBuild action request"
    )
    if observed != raw:
        raise pb.CASTamperError(
            "published PrismaBuild action request failed canonical readback"
        )
    return request_path


def _normalize_state(
    raw: str,
) -> tuple[
    str, Literal["pending", "running", "completed", "cancelled", "failed"]
]:
    value = raw.strip().removesuffix("+")
    if value in _PENDING_STATES:
        return value, "pending"
    if value in _RUNNING_STATES:
        return value, "running"
    if value == "COMPLETED":
        return value, "completed"
    if _CANCELLED_RE.fullmatch(value) is not None:
        return "CANCELLED", "cancelled"
    if value in _FAILED_STATES:
        return value, "failed"
    raise SlurmProtocolError(f"unknown SLURM state: {raw!r}")


def _real_directory_chain_exists(path: Path, *, where: str) -> bool:
    """Check every existing component with lstat; never accept a symlink hop."""

    if not path.is_absolute():
        raise AssertionError("directory-chain validation requires an absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot inspect {where} ancestor {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise pb.CASTamperError(
                f"{where} ancestor is not a real directory: {current}"
            )
    return True


def _ensure_real_directory(path: Path, *, root: Path, where: str) -> None:
    """Create descendants of ``root`` through held no-follow directory FDs.

    A pathname check followed by ``mkdir(path)`` is not sufficient: another
    process can replace an already-checked ancestor with a symlink between the
    two operations.  Keep each parent open and make the next component with
    ``mkdirat`` semantics instead.  Newly created directory entries and their
    final modes are synced before advancing to the next component.
    """

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AssertionError(f"{where} escaped its configured root") from exc
    if ".." in root.parts or ".." in relative.parts:
        raise pb.CASTamperError(f"{where} path contains parent traversal")

    descriptor = _open_directory_nofollow(root.parent, where=where)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        for part in (root.name, *relative.parts):
            created = False
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise pb.CASUnavailableError(
                    f"cannot create {where} directory component {part!r}: {exc}"
                ) from exc

            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                raise pb.CASUnavailableError(
                    f"{where} directory component disappeared: {part!r}"
                ) from exc
            except OSError as exc:
                raise pb.CASTamperError(
                    f"{where} directory component cannot be opened without "
                    f"following links: {part!r}"
                ) from exc

            try:
                if created:
                    # Apply the final mode before syncing the new inode, then
                    # sync the held parent so its directory entry is durable.
                    os.fchmod(child, 0o755)
                    os.fsync(child)
                    os.fsync(descriptor)
            except OSError as exc:
                os.close(child)
                raise pb.CASUnavailableError(
                    f"cannot durably create {where} directory component "
                    f"{part!r}: {exc}"
                ) from exc

            os.close(descriptor)
            descriptor = child
        _assert_directory_identity(
            descriptor, path, where=where
        )
    finally:
        os.close(descriptor)


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    """Open an absolute directory through anchored no-follow components."""

    if not path.is_absolute() or ".." in path.parts:
        raise pb.CASTamperError(
            f"{where} directory must be absolute and contain no parent traversal"
        )
    # PrismaBuild's SLURM adapter is Linux-only.  Do not silently weaken the
    # no-follow contract on a platform lacking these openat flags.
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise pb.CASUnavailableError(
            f"cannot open {where} filesystem root: {exc}"
        ) from exc
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                raise pb.CASUnavailableError(
                    f"{where} directory disappeared: {path}"
                ) from exc
            except OSError as exc:
                raise pb.CASTamperError(
                    f"{where} ancestor cannot be opened without following links: {path}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_identity(descriptor: int, path: Path, *, where: str) -> None:
    """Fail if a held directory is no longer the configured pathname."""

    try:
        current = _open_directory_nofollow(path, where=where)
    except pb.CASUnavailableError as exc:
        raise pb.CASTamperError(
            f"{where} directory disappeared during operation: {path}"
        ) from exc
    try:
        held = os.fstat(descriptor)
        observed = os.fstat(current)
        if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            raise pb.CASTamperError(
                f"{where} directory changed during operation: {path}"
            )
    finally:
        os.close(current)


def _atomic_publish_nofollow(path: Path, raw: bytes, *, where: str) -> bool:
    """First-writer-publish relative to a held, no-follow parent directory FD."""

    directory_fd = _open_directory_nofollow(path.parent, where=where)
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
        _assert_directory_identity(directory_fd, path.parent, where=where)
        return won
    except OSError as exc:
        raise pb.CASUnavailableError(f"cannot publish {where}: {path}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


class SlurmAdapter:
    """Submit and resolve PrismaBuild actions through a SLURM installation."""

    def __init__(
        self,
        *,
        cas_root: str | Path,
        log_root: str | Path,
        worker_script: str | Path,
        cluster: str,
        sbatch: str | Path = "/usr/bin/sbatch",
        squeue: str | Path = "/usr/bin/squeue",
        sacct: str | Path = "/usr/bin/sacct",
        scancel: str | Path = "/usr/bin/scancel",
        scontrol: str | Path = "/usr/bin/scontrol",
        submit_environment: Mapping[str, str] | None = None,
        command_timeout_seconds: float = 30.0,
    ):
        self.cas_root = _absolute_path(cas_root, where="CAS root")
        self.log_root = _absolute_path(log_root, where="log root")
        self.worker_script = _absolute_path(
            worker_script, where="worker script", root_ok=True
        )
        self.cluster = _token(cluster, where="SLURM cluster", pattern=_CLUSTER_RE)
        self.sbatch = _absolute_path(sbatch, where="sbatch", root_ok=True)
        self.squeue = _absolute_path(squeue, where="squeue", root_ok=True)
        self.sacct = _absolute_path(sacct, where="sacct", root_ok=True)
        self.scancel = _absolute_path(scancel, where="scancel", root_ok=True)
        self.scontrol = _absolute_path(scontrol, where="scontrol", root_ok=True)
        if type(command_timeout_seconds) not in {int, float} or command_timeout_seconds <= 0:
            raise pb.ActionContractError(
                "command_timeout_seconds must be a positive finite number"
            )
        self.command_timeout_seconds = float(command_timeout_seconds)
        if not (self.command_timeout_seconds < float("inf")):
            raise pb.ActionContractError(
                "command_timeout_seconds must be a positive finite number"
            )
        raw_environment = submit_environment or {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        environment: dict[str, str] = {}
        for key, value in raw_environment.items():
            name = _token(key, where="submit environment name", pattern=_ENV_NAME_RE)
            if type(value) is not str or "\x00" in value:
                raise pb.ActionContractError(
                    f"submit environment value for {name!r} must be a NUL-free string"
                )
            environment[name] = value
        self.submit_environment = dict(sorted(environment.items()))

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(argv),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                env=self.submit_environment,
                timeout=self.command_timeout_seconds,
            )
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as exc:
            raise SlurmUnavailableError(
                f"cannot execute configured SLURM command {argv[0]!r}: {exc}"
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise SlurmProtocolError(
                f"SLURM command {argv[0]!r} could not return strict UTF-8 output"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "..."
            raise SlurmCommandError(
                f"SLURM command {argv[0]!r} exited {completed.returncode}: {stderr}"
            )
        return completed

    def _check_worker_script(self) -> None:
        try:
            info = os.lstat(self.worker_script)
        except OSError as exc:
            raise pb.ActionContractError(
                f"worker script is unavailable: {self.worker_script}"
            ) from exc
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise pb.ActionContractError(
                f"worker script must be a regular non-symlink file: {self.worker_script}"
            )
        if info.st_mode & 0o111 == 0:
            raise pb.ActionContractError(
                f"worker script must be executable: {self.worker_script}"
            )

    def _capture_submission_runtime(self) -> dict[str, object]:
        """Seal the loaded adapter and configured worker source bytes."""

        self._check_worker_script()
        adapter = pb._normalize_runtime_source(
            _LOADED_SLURM_ADAPTER_IDENTITY,
            where="loaded PrismaBuild SLURM adapter",
        )
        pb._verify_runtime_source_unchanged(
            adapter,
            where="PrismaBuild SLURM adapter",
            changed_message=(
                "PrismaBuild SLURM adapter changed after module import"
            ),
        )
        worker_launcher = pb._identify_runtime_source(
            self.worker_script, where="PrismaBuild SLURM worker launcher"
        )
        self._check_worker_script()
        body: dict[str, object] = {
            "schema": _SLURM_RUNTIME_SCHEMA_V1,
            "adapter": adapter,
            "worker_launcher": worker_launcher,
        }
        return _validate_slurm_runtime(
            {**body, "runtime_sha256": pb.canonical_sha256(body)}
        )

    def _verify_submission_runtime_unchanged(self, value: object) -> None:
        """Recheck the sealed transport sources immediately before sbatch."""

        runtime = _validate_slurm_runtime(value)
        adapter = runtime["adapter"]
        worker_launcher = runtime["worker_launcher"]
        assert isinstance(adapter, Mapping)
        assert isinstance(worker_launcher, Mapping)
        loaded_adapter = pb._normalize_runtime_source(
            _LOADED_SLURM_ADAPTER_IDENTITY,
            where="loaded PrismaBuild SLURM adapter",
        )
        if adapter != loaded_adapter:
            raise pb.LocalActionError(
                "sealed SLURM adapter differs from the module import identity"
            )
        if worker_launcher["path"] != str(self.worker_script):
            raise pb.LocalActionError(
                "sealed SLURM worker launcher differs from adapter configuration"
            )
        pb._verify_runtime_source_unchanged(
            adapter,
            where="PrismaBuild SLURM adapter",
            changed_message=(
                "PrismaBuild SLURM adapter changed after submission sealing"
            ),
        )
        pb._verify_runtime_source_unchanged(
            worker_launcher,
            where="PrismaBuild SLURM worker launcher",
            changed_message=(
                "PrismaBuild SLURM worker launcher changed after submission sealing"
            ),
        )
        self._check_worker_script()

    @staticmethod
    def _cluster_args(job_id: SlurmJobId) -> list[str]:
        # ``--local`` defeats FederationParameters=fed_display for the only
        # legacy/diagnostic clusterless identity we accept. New bindings are
        # always qualified by the adapter's sealed single cluster.
        return (
            [f"--clusters={job_id.cluster}"]
            if job_id.cluster is not None
            else ["--local"]
        )

    def _cas(self, *, create: bool) -> pb.PrismaBuildCAS:
        if create:
            _ensure_real_directory(self.cas_root, root=self.cas_root, where="CAS root")
        elif not _real_directory_chain_exists(self.cas_root, where="CAS root"):
            raise pb.CASUnavailableError(f"CAS root does not exist: {self.cas_root}")
        return pb.PrismaBuildCAS(self.cas_root)

    def _submission_directory(self, action_key: str) -> Path:
        return (
            self.cas_root
            / "submissions"
            / _SUBMISSION_NAMESPACE
            / action_key[:2]
            / action_key
        )

    def _submission_paths(self, action_key: str) -> tuple[Path, Path]:
        directory = self._submission_directory(action_key)
        return directory / "intent.json", directory / "job.json"

    @staticmethod
    def _read_state_bytes(path: Path, *, where: str) -> bytes | None:
        if not _real_directory_chain_exists(path.parent, where=where):
            return None
        try:
            return pb._read_regular_file_nofollow(
                path,
                where=where,
                require_readonly=True,
                max_bytes=_MAX_STATE_BYTES,
            )
        except FileNotFoundError as exc:
            if _real_directory_chain_exists(path.parent, where=where):
                return None
            raise pb.CASTamperError(
                f"{where} parent disappeared during read: {path.parent}"
            ) from exc

    @staticmethod
    def _read_state_file(path: Path, *, where: str) -> object | None:
        raw = SlurmAdapter._read_state_bytes(path, where=where)
        if raw is None:
            return None
        try:
            return pb._decode_strict_json(raw, where=where)
        except pb.ActionContractError as exc:
            raise pb.CASTamperError(str(exc)) from exc

    def _publish_state_file(self, path: Path, value: Mapping[str, object]) -> bool:
        raw = pb._canonical_file_bytes(value)
        if len(raw) > _MAX_STATE_BYTES:
            raise pb.ActionContractError(
                "SLURM durable state exceeds the 16 MiB byte bound"
            )
        _ensure_real_directory(
            path.parent,
            root=self.cas_root,
            where="SLURM durable state",
        )
        return _atomic_publish_nofollow(
            path,
            raw,
            where="SLURM durable state",
        )

    def _load_submission_intent(
        self, action_key: str
    ) -> dict[str, object] | None:
        intent_path, _ = self._submission_paths(action_key)
        value = self._read_state_file(intent_path, where="SLURM submission intent")
        if value is None:
            return None
        intent = _validate_submission_intent(
            value, expected_action_key=action_key
        )
        raw = self._read_state_bytes(intent_path, where="SLURM submission intent")
        if raw is None:
            raise pb.CASTamperError("SLURM submission intent disappeared during read")
        if raw != pb._canonical_file_bytes(intent):
            raise pb.CASTamperError(
                "SLURM submission intent bytes are not canonical JSON"
            )
        return intent

    def _load_job_binding(
        self, intent: Mapping[str, object]
    ) -> dict[str, object] | None:
        _, binding_path = self._submission_paths(str(intent["action_key"]))
        value = self._read_state_file(binding_path, where="SLURM job binding")
        if value is None:
            return None
        binding = _validate_job_binding(value, intent=intent)
        raw = self._read_state_bytes(binding_path, where="SLURM job binding")
        if raw is None:
            raise pb.CASTamperError("SLURM job binding disappeared during read")
        if raw != pb._canonical_file_bytes(binding):
            raise pb.CASTamperError(
                "SLURM job binding bytes are not canonical JSON"
            )
        return binding

    def _publish_submission_intent(
        self, intent: Mapping[str, object]
    ) -> bool:
        normalized = _validate_submission_intent(
            intent, expected_action_key=str(intent.get("action_key"))
        )
        intent_path, _ = self._submission_paths(str(normalized["action_key"]))
        won = self._publish_state_file(intent_path, normalized)
        observed = self._load_submission_intent(str(normalized["action_key"]))
        if observed != normalized:
            raise SlurmAdoptionError(
                "an existing durable submission intent binds the action to a "
                "different sealed submit specification"
            )
        return won

    def _publish_job_binding(
        self, intent: Mapping[str, object], job_id: SlurmJobId
    ) -> dict[str, object]:
        if job_id.cluster is None:
            job_id = SlurmJobId(number=job_id.number, cluster=self.cluster)
        if job_id.cluster != self.cluster:
            raise SlurmAdoptionError(
                "SLURM returned a job id outside the sealed cluster"
            )
        body: dict[str, object] = {
            "schema": SLURM_JOB_BINDING_SCHEMA_V1,
            "action_key": intent["action_key"],
            "submission_key": intent["submission_key"],
            "job_id": str(job_id),
        }
        binding = {**body, "binding_sha256": pb.canonical_sha256(body)}
        normalized = _validate_job_binding(binding, intent=intent)
        _, binding_path = self._submission_paths(str(intent["action_key"]))
        self._publish_state_file(binding_path, normalized)
        observed = self._load_job_binding(intent)
        if observed != normalized:
            raise SlurmAdoptionError(
                "durable SLURM job binding conflicts with the adopted allocation"
            )
        return normalized

    def _submit_spec(
        self,
        *,
        action_key: str,
        request_path: Path,
        checkout_root: Path,
        resources: SlurmResources,
        placement: SlurmPlacement,
        max_polls: int,
        max_requeues: int,
        poll_interval_seconds: float,
        recompute: bool,
    ) -> dict[str, object]:
        expected_request = (
            self.cas_root
            / "requests"
            / action_key[:2]
            / f"{action_key}.json"
        )
        if request_path != expected_request:
            raise pb.ActionContractError(
                "published action request path differs from its canonical address"
            )
        value: dict[str, object] = {
            "schema": _SLURM_SUBMIT_SPEC_SCHEMA_V2,
            "action_key": action_key,
            "cluster": self.cluster,
            "cas_root": str(self.cas_root),
            "log_root": str(self.log_root),
            "worker_script": str(self.worker_script),
            "worker_argv": self._worker_argv(
                request_path=request_path,
                checkout_root=checkout_root,
                recompute=recompute,
            ),
            "checkout_root": str(checkout_root),
            "commands": {
                "sbatch": str(self.sbatch),
                "squeue": str(self.squeue),
                "sacct": str(self.sacct),
                "scancel": str(self.scancel),
                "scontrol": str(self.scontrol),
            },
            "resources": {
                "cpus": resources.cpus,
                "memory_mib": resources.memory_mib,
                "gpus": resources.gpus,
                "constraint": resources.constraint,
                "partition": resources.partition,
                "account": resources.account,
                "qos": resources.qos,
                "time_limit": resources.time_limit,
            },
            "placement": {
                "platform_key": placement.platform_key,
                "host_class": placement.host_class,
            },
            "retry_policy": {
                "max_polls": max_polls,
                "max_requeues": max_requeues,
                "poll_interval_seconds": poll_interval_seconds,
            },
            "recompute": recompute,
            "submit_environment": self.submit_environment,
            "runtime": self._capture_submission_runtime(),
        }
        return _validate_submit_spec(value)

    @staticmethod
    def _submission_intent(submit_spec: Mapping[str, object]) -> dict[str, object]:
        submission_key = pb.canonical_sha256(submit_spec)
        body: dict[str, object] = {
            "schema": SLURM_SUBMISSION_INTENT_SCHEMA_V2,
            "action_key": submit_spec["action_key"],
            "submission_key": submission_key,
            "job_name": f"pqb-{submission_key}",
            "comment": f"prismabuild:{submission_key}",
            "submit_spec": dict(submit_spec),
        }
        return {**body, "intent_sha256": pb.canonical_sha256(body)}

    def _discover_job_binding(
        self, intent: Mapping[str, object]
    ) -> dict[str, object]:
        """Adopt the sole accounting row carrying the durable submit token.

        The intent is written before ``sbatch``.  Therefore no matching row is
        ambiguous: the caller may have died immediately before submission, or
        the accepted job may not yet be visible in accounting.  That state is
        retryable but never authorizes a second ``sbatch``.
        """

        argv = [
            str(self.sacct),
            "--noheader",
            "--parsable2",
            "--allocations",
            # Requeues, federation, resize, and JobID rollover can all leave
            # duplicate accounting records.  Surface every one so adoption
            # rejects ambiguity instead of accepting sacct's newest-only view.
            "--duplicates",
            f"--clusters={self.cluster}",
            f"--name={intent['job_name']}",
            "--starttime=1970-01-01",
            f"--format={_SACCT_ADOPTION_FORMAT}",
        ]
        completed = self._run(argv)
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not rows:
            raise SlurmAdoptionError(
                "durable submission intent has no visible SLURM allocation; "
                "sbatch acceptance is ambiguous and PrismaBuild will not resubmit"
            )
        if len(rows) != 1:
            raise SlurmAdoptionError(
                "durable submission intent matches multiple SLURM allocations; "
                "duplicate or ambiguous scheduler state must be repaired"
            )
        fields = [field.strip() for field in rows[0].split("|")]
        if len(fields) != 5:
            raise SlurmProtocolError(
                f"sacct returned malformed adoption row: {rows[0]!r}"
            )
        number_raw, cluster_raw, job_name, comment, raw_state = fields
        cluster = cluster_raw or None
        try:
            job = SlurmJobId(number=int(number_raw), cluster=cluster)
        except (TypeError, ValueError, pb.ActionContractError) as exc:
            raise SlurmProtocolError(
                f"sacct returned malformed adoption job identity: {rows[0]!r}"
            ) from exc
        if job_name != intent["job_name"] or comment != intent["comment"]:
            raise SlurmAdoptionError(
                "sacct adoption row does not carry the exact durable name/comment token"
            )
        if job.cluster != self.cluster:
            raise SlurmAdoptionError(
                "sacct adoption row is outside the sealed SLURM cluster"
            )
        _normalize_state(raw_state)
        return self._publish_job_binding(intent, job)

    def _bound_job(
        self, action: Mapping[str, object], job_id: SlurmJobId | str
    ) -> tuple[dict[str, object], SlurmJobId]:
        key = str(action["action_key"])
        intent = self._load_submission_intent(key)
        if intent is None:
            raise SlurmAdoptionError(
                "action has no durable SLURM submission intent; raw job ids are not adoptable"
            )
        binding = self._load_job_binding(intent)
        if binding is None:
            binding = self._discover_job_binding(intent)
        recorded = parse_job_id(str(binding["job_id"]))
        supplied = parse_job_id(job_id) if isinstance(job_id, str) else job_id
        if not isinstance(supplied, SlurmJobId):
            raise pb.ActionContractError("job_id must be a SlurmJobId or parsable string")
        if supplied.cluster is None:
            supplied = SlurmJobId(number=supplied.number, cluster=self.cluster)
        if supplied != recorded:
            raise SlurmAdoptionError(
                "supplied SLURM job id differs from the durable action binding"
            )
        return intent, recorded

    def _transition_directory(self, action_key: str, kind: str) -> Path:
        return self._submission_directory(action_key) / "transitions" / kind

    def _load_retry_transitions(
        self,
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
        kind: Literal["poll", "requeue"],
        max_poll_attempt: int | None = None,
    ) -> list[dict[str, object]]:
        directory = self._transition_directory(str(intent["action_key"]), f"{kind}s")
        if not _real_directory_chain_exists(
            directory, where=f"SLURM {kind} transition directory"
        ):
            return []
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot inspect SLURM {kind} transition directory: {directory}: {exc}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise pb.CASTamperError(
                f"SLURM {kind} transition path is not a real directory: {directory}"
            )
        numbered: list[tuple[int, int, Path]] = []
        retry_policy = intent["submit_spec"]["retry_policy"]  # type: ignore[index]
        assert isinstance(retry_policy, Mapping)
        if kind == "poll":
            if max_poll_attempt is None:
                raise AssertionError("poll transition validation requires max attempt")
            maximum_entries = int(retry_policy["max_polls"]) * (
                max_poll_attempt + 1
            )
        else:
            maximum_entries = int(retry_policy["max_requeues"])
        temporary_entries = 0
        directory_fd = _open_directory_nofollow(
            directory, where=f"SLURM {kind} transition directory"
        )
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    path = directory / entry.name
                    match = (
                        _POLL_FILENAME_RE.fullmatch(entry.name)
                        if kind == "poll"
                        else _COUNTER_FILENAME_RE.fullmatch(entry.name)
                    )
                    if match is None:
                        # A killed first-writer can leave a private temp inode,
                        # but an unbounded pile is hostile state, not recovery.
                        if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                            temporary_entries += 1
                            if temporary_entries > _MAX_STALE_TEMP_ENTRIES:
                                raise pb.CASTamperError(
                                    f"too many stale SLURM {kind} temp entries"
                                )
                            continue
                        raise pb.CASTamperError(
                            f"unexpected SLURM {kind} transition entry: {path}"
                        )
                    ordinal = int(match.group("ordinal"))
                    attempt = (
                        int(match.group("attempt"))
                        if kind == "poll"
                        else ordinal - 1
                    )
                    numbered.append((attempt, ordinal, path))
                    if len(numbered) > maximum_entries:
                        raise pb.CASTamperError(
                            f"SLURM {kind} transition count exceeds the sealed maximum"
                        )
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot list SLURM {kind} transitions: {directory}: {exc}"
            ) from exc
        finally:
            os.close(directory_fd)
        numbered.sort(key=lambda row: (row[0], row[1]))
        if kind == "requeue":
            if [ordinal for _, ordinal, _ in numbered] != list(
                range(1, len(numbered) + 1)
            ):
                raise pb.CASTamperError(
                    "SLURM requeue transition ordinals are not a contiguous prefix"
                )
        else:
            assert max_poll_attempt is not None
            by_attempt: dict[int, list[int]] = {}
            for attempt, ordinal, _ in numbered:
                if attempt > max_poll_attempt:
                    raise pb.CASTamperError(
                        "SLURM poll transition names an impossible retry attempt"
                    )
                by_attempt.setdefault(attempt, []).append(ordinal)
            for attempt, ordinals in by_attempt.items():
                if ordinals != list(range(1, len(ordinals) + 1)):
                    raise pb.CASTamperError(
                        f"SLURM poll transition ordinals for attempt {attempt} "
                        "are not a contiguous prefix"
                    )
        transitions: list[dict[str, object]] = []
        for attempt, ordinal, path in numbered:
            value = self._read_state_file(
                path, where=f"SLURM {kind} transition {ordinal}"
            )
            if value is None:
                raise pb.CASTamperError(
                    f"SLURM {kind} transition {ordinal} disappeared after listing"
                )
            transition = _validate_retry_transition(
                value,
                intent=intent,
                binding=binding,
                expected_kind=kind,
                expected_attempt=attempt,
                expected_ordinal=ordinal,
            )
            raw = self._read_state_bytes(
                path, where=f"SLURM {kind} transition {ordinal}"
            )
            if raw is None:
                raise pb.CASTamperError(
                    f"SLURM {kind} transition {ordinal} disappeared during read"
                )
            if raw != pb._canonical_file_bytes(transition):
                raise pb.CASTamperError(
                    f"SLURM {kind} transition {ordinal} is not canonical JSON"
                )
            transitions.append(transition)
        return transitions

    def _load_requeue_observations(
        self,
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
        requeues: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        directory = self._transition_directory(
            str(intent["action_key"]), "requeue_observed"
        )
        if not _real_directory_chain_exists(
            directory, where="SLURM requeue-observation directory"
        ):
            return []
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot inspect SLURM requeue-observation directory: {exc}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise pb.CASTamperError(
                "SLURM requeue-observation path is not a real directory"
            )
        numbered: list[tuple[int, Path]] = []
        temporary_entries = 0
        directory_fd = _open_directory_nofollow(
            directory, where="SLURM requeue-observation directory"
        )
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    path = directory / entry.name
                    match = _COUNTER_FILENAME_RE.fullmatch(entry.name)
                    if match is None:
                        if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                            temporary_entries += 1
                            if temporary_entries > _MAX_STALE_TEMP_ENTRIES:
                                raise pb.CASTamperError(
                                    "too many stale SLURM requeue-observation temp entries"
                                )
                            continue
                        raise pb.CASTamperError(
                            f"unexpected SLURM requeue-observation entry: {path}"
                        )
                    numbered.append((int(match.group("ordinal")), path))
                    if len(numbered) > len(requeues):
                        raise pb.CASTamperError(
                            "SLURM requeue observations exceed durable claims"
                        )
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot list SLURM requeue observations: {exc}"
            ) from exc
        finally:
            os.close(directory_fd)
        numbered.sort(key=lambda row: row[0])
        ordinals = [ordinal for ordinal, _ in numbered]
        if ordinals != list(range(1, len(numbered) + 1)):
            raise pb.CASTamperError(
                "SLURM requeue-observation ordinals are not a contiguous prefix"
            )
        if len(numbered) > len(requeues) or len(numbered) < max(0, len(requeues) - 1):
            raise pb.CASTamperError(
                "SLURM requeue claims and positive-transition observations disagree"
            )
        observations: list[dict[str, object]] = []
        for ordinal, path in numbered:
            value = self._read_state_file(
                path, where=f"SLURM requeue observation {ordinal}"
            )
            if value is None:
                raise pb.CASTamperError(
                    f"SLURM requeue observation {ordinal} disappeared after listing"
                )
            observation = _validate_requeue_observed(
                value,
                intent=intent,
                binding=binding,
                expected_ordinal=ordinal,
            )
            raw = self._read_state_bytes(
                path, where=f"SLURM requeue observation {ordinal}"
            )
            if raw is None:
                raise pb.CASTamperError(
                    f"SLURM requeue observation {ordinal} disappeared during read"
                )
            if raw != pb._canonical_file_bytes(observation):
                raise pb.CASTamperError(
                    f"SLURM requeue observation {ordinal} is not canonical JSON"
                )
            observations.append(observation)
        return observations

    def _retry_state(
        self, action: Mapping[str, object], job_id: SlurmJobId | str
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        SlurmJobId,
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        intent, job = self._bound_job(action, job_id)
        binding = self._load_job_binding(intent)
        if binding is None:
            raise pb.CASTamperError("SLURM job binding disappeared during retry-state load")
        requeues = self._load_retry_transitions(
            intent=intent, binding=binding, kind="requeue"
        )
        observations = self._load_requeue_observations(
            intent=intent, binding=binding, requeues=requeues
        )
        polls = self._load_retry_transitions(
            intent=intent,
            binding=binding,
            kind="poll",
            max_poll_attempt=len(requeues),
        )
        retry_policy = intent["submit_spec"]["retry_policy"]  # type: ignore[index]
        assert isinstance(retry_policy, Mapping)
        max_requeues = int(retry_policy["max_requeues"])
        max_polls = int(retry_policy["max_polls"])
        if len(requeues) > max_requeues:
            raise pb.CASTamperError(
                "durable SLURM requeue count exceeds the sealed maximum"
            )
        for attempt in range(len(requeues) + 1):
            count = sum(1 for row in polls if row["attempt"] == attempt)
            if count > max_polls:
                raise pb.CASTamperError(
                    "durable SLURM poll count exceeds the sealed maximum"
                )
        return intent, binding, job, polls, requeues, observations

    def retry_progress(
        self, action: object, job_id: SlurmJobId | str
    ) -> SlurmRetryProgress:
        normalized = pb.validate_action(action)
        _, _, _, polls, requeues, observations = self._retry_state(
            normalized, job_id
        )
        current_polls = sum(
            1 for transition in polls if transition["attempt"] == len(requeues)
        )
        return SlurmRetryProgress(
            polls=current_polls,
            requeues=len(requeues),
            latest_requeue_observed=(
                not requeues or len(observations) == len(requeues)
            ),
        )

    @staticmethod
    def _transition_body(
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
        kind: Literal["poll", "requeue"],
        attempt: int,
        ordinal: int,
        claimed_unix_ns: int,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": SLURM_RETRY_TRANSITION_SCHEMA_V1,
            "action_key": intent["action_key"],
            "submission_key": intent["submission_key"],
            "job_id": binding["job_id"],
            "kind": kind,
            "attempt": attempt,
            "ordinal": ordinal,
            "claimed_unix_ns": claimed_unix_ns,
        }
        return {**body, "transition_sha256": pb.canonical_sha256(body)}

    def _claim_retry_transition(
        self,
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
        kind: Literal["poll", "requeue"],
        attempt: int,
        ordinal: int,
        claimed_unix_ns: int,
    ) -> None:
        transition = self._transition_body(
            intent=intent,
            binding=binding,
            kind=kind,
            attempt=attempt,
            ordinal=ordinal,
            claimed_unix_ns=claimed_unix_ns,
        )
        filename = (
            f"{attempt:08d}-{ordinal:08d}.json"
            if kind == "poll"
            else f"{ordinal:08d}.json"
        )
        path = self._transition_directory(
            str(intent["action_key"]), f"{kind}s"
        ) / filename
        won = self._publish_state_file(path, transition)
        value = self._read_state_file(
            path, where=f"SLURM {kind} transition {ordinal}"
        )
        observed = _validate_retry_transition(
            value,
            intent=intent,
            binding=binding,
            expected_kind=kind,
            expected_attempt=attempt,
            expected_ordinal=ordinal,
        )
        if won and observed != transition:
            raise pb.CASTamperError(
                f"SLURM {kind} transition race produced conflicting state"
            )
        if not won:
            raise SlurmAdoptionError(
                f"another orchestrator already claimed SLURM {kind} transition {ordinal}"
            )

    def claim_poll(
        self,
        action: object,
        job_id: SlurmJobId | str,
        *,
        max_polls: int,
    ) -> SlurmRetryProgress | None:
        """Consume one durable poll slot, or return ``None`` at the bound."""

        limit = _positive_integer(max_polls, where="max_polls")
        normalized = pb.validate_action(action)
        intent, binding, _, polls, requeues, observations = self._retry_state(
            normalized, job_id
        )
        retry_policy = intent["submit_spec"]["retry_policy"]  # type: ignore[index]
        assert isinstance(retry_policy, Mapping)
        if limit != retry_policy["max_polls"]:
            raise SlurmAdoptionError(
                "requested poll budget differs from the sealed submission policy"
            )
        attempt = len(requeues)
        current_polls = sum(
            1 for transition in polls if transition["attempt"] == attempt
        )
        if current_polls >= limit:
            return None
        now = time.time_ns()
        if now < 0:
            raise SlurmAdoptionError("wall clock is before the Unix epoch")
        if now > _MAX_UNIX_NS:
            raise SlurmAdoptionError(
                "wall clock exceeds the signed 64-bit Unix-nanosecond range"
            )
        if current_polls:
            previous = max(
                (
                    transition
                    for transition in polls
                    if transition["attempt"] == attempt
                ),
                key=lambda transition: int(transition["ordinal"]),
            )
            previous_ns = int(previous["claimed_unix_ns"])
            if now < previous_ns:
                raise SlurmAdoptionError(
                    "wall clock moved behind the prior durable poll claim"
                )
            interval_ns = math.ceil(float(retry_policy["poll_interval_seconds"]) * 1e9)
            earliest = previous_ns + interval_ns
            while now < earliest:
                time.sleep((earliest - now) / 1e9)
                next_now = time.time_ns()
                if next_now < now:
                    raise SlurmAdoptionError(
                        "wall clock moved backwards while pacing a durable poll"
                    )
                now = next_now
        self._claim_retry_transition(
            intent=intent,
            binding=binding,
            kind="poll",
            attempt=attempt,
            ordinal=current_polls + 1,
            claimed_unix_ns=now,
        )
        return SlurmRetryProgress(
            polls=current_polls + 1,
            requeues=len(requeues),
            latest_requeue_observed=(
                not requeues or len(observations) == len(requeues)
            ),
        )

    def _mark_latest_requeue_observed(
        self, action: Mapping[str, object], job_id: SlurmJobId
    ) -> None:
        intent, binding, _, _, requeues, observations = self._retry_state(
            action, job_id
        )
        if not requeues or len(observations) == len(requeues):
            return
        ordinal = len(requeues)
        body: dict[str, object] = {
            "schema": SLURM_REQUEUE_OBSERVED_SCHEMA_V1,
            "action_key": intent["action_key"],
            "submission_key": intent["submission_key"],
            "job_id": binding["job_id"],
            "ordinal": ordinal,
        }
        observation = {**body, "observed_sha256": pb.canonical_sha256(body)}
        path = self._transition_directory(
            str(intent["action_key"]), "requeue_observed"
        ) / f"{ordinal:08d}.json"
        self._publish_state_file(path, observation)
        value = self._read_state_file(
            path, where=f"SLURM requeue observation {ordinal}"
        )
        observed = _validate_requeue_observed(
            value,
            intent=intent,
            binding=binding,
            expected_ordinal=ordinal,
        )
        if observed != observation:
            raise pb.CASTamperError(
                "SLURM requeue observation race produced conflicting state"
            )

    def _mutation_directory(self, action_key: str) -> Path:
        return self._submission_directory(action_key) / "mutations"

    def _load_mutations(
        self,
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
    ) -> list[dict[str, object]]:
        directory = self._mutation_directory(str(intent["action_key"]))
        if not _real_directory_chain_exists(
            directory, where="SLURM mutation directory"
        ):
            return []
        retry_policy = intent["submit_spec"]["retry_policy"]  # type: ignore[index]
        assert isinstance(retry_policy, Mapping)
        maximum_entries = int(retry_policy["max_requeues"]) + 1
        numbered: list[tuple[int, Path]] = []
        temporary_entries = 0
        directory_fd = _open_directory_nofollow(
            directory, where="SLURM mutation directory"
        )
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    path = directory / entry.name
                    match = _COUNTER_FILENAME_RE.fullmatch(entry.name)
                    if match is None:
                        if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                            temporary_entries += 1
                            if temporary_entries > _MAX_STALE_TEMP_ENTRIES:
                                raise pb.CASTamperError(
                                    "too many stale SLURM mutation temp entries"
                                )
                            continue
                        raise pb.CASTamperError(
                            f"unexpected SLURM mutation entry: {path}"
                        )
                    numbered.append((int(match.group("ordinal")), path))
                    if len(numbered) > maximum_entries:
                        raise pb.CASTamperError(
                            "SLURM mutation count exceeds the sealed maximum"
                        )
        except OSError as exc:
            raise pb.CASUnavailableError(
                f"cannot list SLURM mutations: {directory}: {exc}"
            ) from exc
        finally:
            os.close(directory_fd)
        numbered.sort(key=lambda row: row[0])
        if [ordinal for ordinal, _ in numbered] != list(
            range(1, len(numbered) + 1)
        ):
            raise pb.CASTamperError(
                "SLURM mutation ordinals are not a contiguous prefix"
            )
        mutations: list[dict[str, object]] = []
        for ordinal, path in numbered:
            value = self._read_state_file(
                path, where=f"SLURM mutation {ordinal}"
            )
            if value is None:
                raise pb.CASTamperError(
                    f"SLURM mutation {ordinal} disappeared after listing"
                )
            mutation = _validate_mutation(
                value,
                intent=intent,
                binding=binding,
                expected_ordinal=ordinal,
            )
            raw = self._read_state_bytes(path, where=f"SLURM mutation {ordinal}")
            if raw is None:
                raise pb.CASTamperError(
                    f"SLURM mutation {ordinal} disappeared during read"
                )
            if raw != pb._canonical_file_bytes(mutation):
                raise pb.CASTamperError(
                    f"SLURM mutation {ordinal} is not canonical JSON"
                )
            mutations.append(mutation)
        cancel_positions = [
            index for index, row in enumerate(mutations) if row["kind"] == "cancel"
        ]
        if len(cancel_positions) > 1 or (
            cancel_positions and cancel_positions[0] != len(mutations) - 1
        ):
            raise pb.CASTamperError(
                "SLURM cancel mutation is not the unique final mutation"
            )
        requeues = sum(1 for row in mutations if row["kind"] == "requeue")
        if requeues > int(retry_policy["max_requeues"]):
            raise pb.CASTamperError(
                "SLURM mutation requeue count exceeds the sealed maximum"
            )
        return mutations

    def _claim_mutation(
        self,
        *,
        intent: Mapping[str, object],
        binding: Mapping[str, object],
        kind: Literal["cancel", "requeue"],
    ) -> dict[str, object]:
        mutations = self._load_mutations(intent=intent, binding=binding)
        if any(row["kind"] == "cancel" for row in mutations):
            raise SlurmAdoptionError(
                "a final SLURM cancel claim already exists; scheduler replay is ambiguous"
            )
        retry_policy = intent["submit_spec"]["retry_policy"]  # type: ignore[index]
        assert isinstance(retry_policy, Mapping)
        if kind == "requeue":
            used = sum(1 for row in mutations if row["kind"] == "requeue")
            if used >= int(retry_policy["max_requeues"]):
                raise SlurmProtocolError(
                    "SLURM requeue mutation would exceed the sealed maximum"
                )
        ordinal = len(mutations) + 1
        body: dict[str, object] = {
            "schema": SLURM_MUTATION_SCHEMA_V1,
            "action_key": intent["action_key"],
            "submission_key": intent["submission_key"],
            "job_id": binding["job_id"],
            "kind": kind,
            "ordinal": ordinal,
        }
        mutation = {**body, "mutation_sha256": pb.canonical_sha256(body)}
        path = self._mutation_directory(str(intent["action_key"])) / f"{ordinal:08d}.json"
        won = self._publish_state_file(path, mutation)
        observed_value = self._read_state_file(
            path, where=f"SLURM mutation {ordinal}"
        )
        observed = _validate_mutation(
            observed_value,
            intent=intent,
            binding=binding,
            expected_ordinal=ordinal,
        )
        if won and observed != mutation:
            raise pb.CASTamperError(
                "SLURM mutation race produced conflicting state"
            )
        if not won:
            raise SlurmAdoptionError(
                f"another orchestrator already claimed SLURM mutation {ordinal}"
            )
        return mutation

    def _worker_argv(
        self,
        *,
        request_path: Path,
        checkout_root: Path,
        recompute: bool,
    ) -> list[str]:
        argv = [
            str(self.worker_script),
            "run-local",
            "--require-slurm-initial-start",
            "--action",
            str(request_path),
            "--cas-root",
            str(self.cas_root),
            "--checkout-root",
            str(checkout_root),
        ]
        if recompute:
            raise pb.ActionContractError("SLURM recompute is disabled")
        return argv

    def submit(
        self,
        action: object,
        *,
        checkout_root: str | Path,
        resources: SlurmResources,
        placement: SlurmPlacement,
        max_polls: int = 1,
        max_requeues: int = 0,
        poll_interval_seconds: float = 5.0,
        recompute: bool = False,
    ) -> SlurmSubmission:
        """Submit an action or return its already-verified CAS receipt."""

        if type(recompute) is not bool:
            raise pb.ActionContractError("recompute must be a boolean")
        if recompute:
            raise pb.ActionContractError(
                "SLURM recompute is disabled because a receipt closes its lineage"
            )
        max_polls = _bounded_counter(max_polls, where="max_polls", positive=True)
        max_requeues = _bounded_counter(
            max_requeues, where="max_requeues", positive=False
        )
        if max_requeues != 0:
            raise pb.ActionContractError(
                "SLURM same-job requeue is disabled; max_requeues must be zero"
            )
        poll_interval_seconds = _poll_interval(
            poll_interval_seconds, where="poll_interval_seconds"
        )
        normalized = pb.validate_action(action)
        validate_placement_scope(
            normalized, placement=placement, resources=resources
        )
        cas = self._cas(create=True)
        cached = _validated_receipt(cas, normalized)
        if cached is not None:
            return SlurmSubmission(
                status="cache_hit",
                action_key=str(normalized["action_key"]),
                job_id=None,
                receipt=cached,
                submission_key=None,
            )
        root = _absolute_path(checkout_root, where="checkout root")
        if not _real_directory_chain_exists(root, where="SLURM checkout root"):
            raise pb.ActionContractError(f"checkout root is not a directory: {root}")
        self._check_worker_script()
        request = publish_action_request(normalized, cas_root=self.cas_root)
        key = str(normalized["action_key"])
        log_directory = self.log_root / key[:2]
        _ensure_real_directory(
            log_directory, root=self.log_root, where="SLURM log root"
        )
        submit_spec = self._submit_spec(
            action_key=key,
            request_path=request,
            checkout_root=root,
            resources=resources,
            placement=placement,
            max_polls=max_polls,
            max_requeues=max_requeues,
            poll_interval_seconds=poll_interval_seconds,
            recompute=recompute,
        )
        intent = self._submission_intent(submit_spec)
        intent_won = self._publish_submission_intent(intent)
        if not intent_won:
            binding = self._load_job_binding(intent)
            if binding is None:
                binding = self._discover_job_binding(intent)
            return SlurmSubmission(
                status="adopted",
                action_key=key,
                job_id=parse_job_id(str(binding["job_id"])),
                receipt=None,
                submission_key=str(intent["submission_key"]),
            )
        argv = [
            str(self.sbatch),
            "--parsable",
            f"--clusters={self.cluster}",
            # NONE implicitly invokes Slurm's --get-user-env path. NIL exports
            # only scheduler/SPANK variables and requires the absolute worker
            # path already enforced by this adapter.
            "--export=NIL",
            # Slurm uses one Requeue flag for explicit and automatic/admin
            # restarts. Positive same-job retries are de-menued until those
            # restart causes can be cryptographically/durably distinguished.
            "--no-requeue",
            "--nodes=1",
            "--ntasks=1",
            "--open-mode=append",
            f"--job-name={intent['job_name']}",
            f"--comment={intent['comment']}",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory_mib}M",
        ]
        if resources.gpus:
            argv.append(f"--gpus={resources.gpus}")
        argv.extend([
            f"--constraint={resources.constraint}",
            f"--partition={resources.partition}",
            f"--account={resources.account}",
            f"--qos={resources.qos}",
            f"--time={resources.time_limit}",
            f"--chdir={root}",
            f"--output={log_directory / (key + '-%j.out')}",
            f"--error={log_directory / (key + '-%j.err')}",
        ])
        worker_argv = submit_spec["worker_argv"]
        assert isinstance(worker_argv, list)
        argv.append(f"--wrap=exec {shlex.join(worker_argv)}")
        self._verify_submission_runtime_unchanged(submit_spec["runtime"])
        completed = self._run(argv)
        output = completed.stdout.strip()
        if "\n" in output or "\r" in output:
            raise SlurmProtocolError(
                f"sbatch --parsable returned multiple lines: {completed.stdout!r}"
            )
        job_id = parse_job_id(output)
        binding = self._publish_job_binding(intent, job_id)
        job_id = parse_job_id(str(binding["job_id"]))
        return SlurmSubmission(
            status="submitted",
            action_key=key,
            job_id=job_id,
            receipt=None,
            submission_key=str(intent["submission_key"]),
        )

    def _query_state(
        self, job_id: SlurmJobId, *, intent: Mapping[str, object]
    ) -> tuple[
        str, Literal["pending", "running", "completed", "cancelled", "failed"]
    ]:
        selector = str(job_id.number)
        squeue_argv = [
            str(self.squeue),
            "--noheader",
            f"--jobs={selector}",
            "--format=%i|%T|%j|%k",
            *self._cluster_args(job_id),
        ]
        current = self._run(squeue_argv)
        lines = [line.strip() for line in current.stdout.splitlines() if line.strip()]
        if len(lines) > 1:
            raise SlurmProtocolError(
                f"squeue returned ambiguous rows for job {job_id}: {lines!r}"
            )
        if lines:
            return self._parse_state_line(
                lines[0], job_id=job_id, intent=intent, source="squeue"
            )

        accounting_argv = [
            str(self.sacct),
            "--noheader",
            "--parsable2",
            "--allocations",
            "--duplicates",
            f"--jobs={selector}",
            f"--format={_SACCT_STATE_FORMAT}",
            *self._cluster_args(job_id),
        ]
        accounting = self._run(accounting_argv)
        rows = [line.strip() for line in accounting.stdout.splitlines() if line.strip()]
        if not rows:
            # A newly submitted or requeued allocation can briefly be absent
            # from both controllers while sacct ingests it. This is neither
            # success nor a terminal verdict. The orchestrator's bounded poll
            # budget owns the fail-closed timeout.
            return "NOT_VISIBLE", "pending"
        if len(rows) != 1:
            raise SlurmProtocolError(
                "sacct returned ambiguous allocation rows "
                f"for job {job_id}: {rows!r}"
            )
        return self._parse_state_line(
            rows[0], job_id=job_id, intent=intent, source="sacct"
        )

    @staticmethod
    def _parse_state_line(
        line: str,
        *,
        job_id: SlurmJobId,
        intent: Mapping[str, object],
        source: str,
    ) -> tuple[str, str]:
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4:
            raise SlurmProtocolError(
                f"{source} returned malformed or wrong-job row for {job_id}: {line!r}"
            )
        raw_job, raw_state, job_name, comment = fields
        try:
            observed_number = int(raw_job)
        except ValueError as exc:
            raise SlurmProtocolError(
                f"{source} returned malformed or wrong-job row for {job_id}: {line!r}"
            ) from exc
        if observed_number != job_id.number:
            raise SlurmProtocolError(
                f"{source} returned malformed or wrong-job row for {job_id}: {line!r}"
            )
        if job_name != intent["job_name"] or comment != intent["comment"]:
            raise SlurmAdoptionError(
                f"{source} job {job_id} does not carry the durable submission identity"
            )
        return _normalize_state(raw_state)

    def resolve(self, action: object, job_id: SlurmJobId | str) -> SlurmResolution:
        """Resolve an allocation, treating a valid CAS receipt as sole success."""

        normalized = pb.validate_action(action)
        cas = self._cas(create=False)
        receipt = _validated_receipt(cas, normalized)
        if receipt is not None:
            job = parse_job_id(job_id) if isinstance(job_id, str) else job_id
            if not isinstance(job, SlurmJobId):
                raise pb.ActionContractError(
                    "job_id must be a SlurmJobId or parsable string"
                )
            return SlurmResolution(
                status="succeeded",
                action_key=str(normalized["action_key"]),
                job_id=job,
                slurm_state=None,
                reason="verified CAS receipt exists",
                receipt=receipt,
                payload_path=cas.result_path(receipt, normalized),
            )

        intent, job = self._bound_job(normalized, job_id)
        raw_state, category = self._query_state(job, intent=intent)
        if category == "pending":
            if raw_state != "NOT_VISIBLE":
                self._mark_latest_requeue_observed(normalized, job)
            reason = (
                "SLURM allocation is not yet visible in squeue or sacct"
                if raw_state == "NOT_VISIBLE"
                else "SLURM allocation has not finished"
            )
            return SlurmResolution(
                status="pending",
                action_key=str(normalized["action_key"]),
                job_id=job,
                slurm_state=raw_state,
                reason=reason,
                receipt=None,
                payload_path=None,
            )
        if category == "running":
            self._mark_latest_requeue_observed(normalized, job)
            return SlurmResolution(
                status="running",
                action_key=str(normalized["action_key"]),
                job_id=job,
                slurm_state=raw_state,
                reason="SLURM allocation is active",
                receipt=None,
                payload_path=None,
            )
        if category == "cancelled":
            status: Literal["cancelled", "failed"] = "cancelled"
            reason = "SLURM allocation was cancelled without a CAS receipt"
        elif category == "completed":
            status = "failed"
            reason = "SLURM reported COMPLETED but no valid CAS receipt exists"
        else:
            status = "failed"
            reason = f"SLURM allocation ended in {raw_state} without a CAS receipt"
        return SlurmResolution(
            status=status,
            action_key=str(normalized["action_key"]),
            job_id=job,
            slurm_state=raw_state,
            reason=reason,
            receipt=None,
            payload_path=None,
        )

    def cancel(self, action: object, job_id: SlurmJobId | str) -> bool:
        """Cancel only the allocation durably and visibly bound to ``action``.

        Returns ``False`` when the exact allocation is already terminal or the
        CAS receipt already closed the action.  An accounting visibility gap is
        ambiguous and refuses instead of sending ``scancel`` to a raw id.
        """

        normalized = pb.validate_action(action)
        cas = self._cas(create=False)
        if _validated_receipt(cas, normalized) is not None:
            return False
        intent, binding, job, _, _, _ = self._retry_state(normalized, job_id)
        mutations = self._load_mutations(intent=intent, binding=binding)
        if any(row["kind"] == "cancel" for row in mutations):
            raise SlurmAdoptionError(
                "a durable cancel claim already exists; scancel outcome is ambiguous"
            )
        raw_state, category = self._query_state(job, intent=intent)
        if raw_state == "NOT_VISIBLE":
            raise SlurmAdoptionError(
                "bound SLURM allocation is not visible; cancellation is ambiguous"
            )
        if category not in {"pending", "running"}:
            return False
        self._claim_mutation(intent=intent, binding=binding, kind="cancel")
        if _validated_receipt(cas, normalized) is not None:
            return False
        # The append is the sole mutation serialization point. Re-query after
        # winning it, then issue at most one RPC. A crash/timeout after this
        # point leaves an intentionally ambiguous final claim that is never
        # replayed by a later orchestrator.
        raw_state, category = self._query_state(job, intent=intent)
        if raw_state == "NOT_VISIBLE":
            raise SlurmAdoptionError(
                "bound SLURM allocation vanished after cancel claim"
            )
        if category not in {"pending", "running"}:
            return False
        self._run(
            [
                str(self.scancel),
                *self._cluster_args(job),
                str(job.number),
            ]
        )
        return True

    def requeue(
        self,
        action: object,
        job_id: SlurmJobId | str,
        *,
        max_requeues: int,
    ) -> bool:
        """Refuse same-job retry until Slurm restart lineage can be authorized."""

        normalized = pb.validate_action(action)
        cas = self._cas(create=False)
        if _validated_receipt(cas, normalized) is not None:
            return False
        _nonnegative_integer(max_requeues, where="max_requeues")
        # Validate the durable binding/journal before returning the policy
        # refusal so corrupted state cannot be hidden behind a disabled API.
        intent, binding, _, _, _, _ = self._retry_state(normalized, job_id)
        self._load_mutations(intent=intent, binding=binding)
        raise SlurmProtocolError(
            "same-job requeue is disabled: Slurm cannot safely separate explicit "
            "authorization from automatic or administrator restart"
        )


__all__ = [
    "SLURM_JOB_BINDING_SCHEMA_V1",
    "SLURM_MUTATION_SCHEMA_V1",
    "SLURM_REQUEUE_OBSERVED_SCHEMA_V1",
    "SLURM_RETRY_TRANSITION_SCHEMA_V1",
    "SLURM_SUBMISSION_INTENT_SCHEMA_V1",
    "SLURM_SUBMISSION_INTENT_SCHEMA_V2",
    "SlurmAdapter",
    "SlurmAdoptionError",
    "SlurmAdapterError",
    "SlurmCommandError",
    "SlurmJobId",
    "SlurmPlacement",
    "SlurmProtocolError",
    "SlurmResolution",
    "SlurmResources",
    "SlurmRetryProgress",
    "SlurmSubmission",
    "SlurmUnavailableError",
    "parse_job_id",
    "publish_action_request",
    "validate_placement_scope",
]
