"""Fail-closed SLURM transport for immutable PrismaBuild actions.

SLURM is only the resource layer.  It may report that an allocation completed,
but a PrismaBuild action is successful only after :class:`PrismaBuildCAS`
returns a fully verified receipt for that exact action.  This module therefore
does not maintain a second result database or infer success from scheduler
state.

The adapter intentionally uses argv arrays, ``shell=False``, ``--export=NONE``,
and a small closed submit environment.  The remote worker receives one
content-addressed, immutable action request and invokes the ordinary
``prismabuild run-local`` command; it does not receive task argv through shell
text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Literal

from . import prismabuild as pb


_SLURM_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_SCOPE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/-]{0,255}\Z")
_CLUSTER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TIME_RE = re.compile(r"(?:(?:[0-9]+)-)?(?:[0-9]{1,2}):[0-5][0-9]:[0-5][0-9]\Z")
_JOB_ID_RE = re.compile(
    r"(?P<number>[1-9][0-9]*)(?:;(?P<cluster>[A-Za-z0-9][A-Za-z0-9._-]{0,63}))?\Z"
)
_STATE_LINE_RE = re.compile(r"(?P<job>[1-9][0-9]*)\|(?P<state>[^|\r\n]+)\Z")
_CANCELLED_RE = re.compile(r"CANCELLED(?: by [0-9]+)?\Z")

_PENDING_STATES = frozenset(
    {
        "CONFIGURING",
        "PENDING",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING",
        "RESV_DEL_HOLD",
    }
)
_RUNNING_STATES = frozenset(
    {"COMPLETING", "RUNNING", "SIGNALING", "STAGE_OUT", "SUSPENDED"}
)
_FAILED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "STOPPED",
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


def _absolute_path(value: str | Path, *, where: str, root_ok: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute() or (not root_ok and path == Path("/")):
        raise pb.ActionContractError(f"{where} must be a non-root absolute path")
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
    status: Literal["cache_hit", "submitted"]
    action_key: str
    job_id: SlurmJobId | None
    receipt: Mapping[str, object] | None


@dataclass(frozen=True)
class SlurmResolution:
    status: Literal["succeeded", "pending", "running", "cancelled", "failed"]
    action_key: str
    job_id: SlurmJobId
    slurm_state: str | None
    reason: str
    receipt: Mapping[str, object] | None
    payload_path: Path | None


def parse_job_id(value: str) -> SlurmJobId:
    """Parse the exact output of ``sbatch --parsable``."""

    match = _JOB_ID_RE.fullmatch(value)
    if match is None:
        raise SlurmProtocolError(f"malformed sbatch job id: {value!r}")
    cluster = match.group("cluster")
    return SlurmJobId(number=int(match.group("number")), cluster=cluster)


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
    return pb.PrismaBuildCAS(root).publish_action_request(action)


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


class SlurmAdapter:
    """Submit and resolve PrismaBuild actions through a SLURM installation."""

    def __init__(
        self,
        *,
        cas_root: str | Path,
        log_root: str | Path,
        worker_script: str | Path,
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

    @staticmethod
    def _cluster_args(job_id: SlurmJobId) -> list[str]:
        return [f"--clusters={job_id.cluster}"] if job_id.cluster is not None else []

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
            "--action",
            str(request_path),
            "--cas-root",
            str(self.cas_root),
            "--checkout-root",
            str(checkout_root),
        ]
        if recompute:
            argv.append("--recompute")
        return argv

    def submit(
        self,
        action: object,
        *,
        checkout_root: str | Path,
        resources: SlurmResources,
        placement: SlurmPlacement,
        recompute: bool = False,
    ) -> SlurmSubmission:
        """Submit an action or return its already-verified CAS receipt."""

        normalized = pb.validate_action(action)
        validate_placement_scope(
            normalized, placement=placement, resources=resources
        )
        cas = pb.PrismaBuildCAS(self.cas_root)
        if not recompute:
            cached = _validated_receipt(cas, normalized)
            if cached is not None:
                return SlurmSubmission(
                    status="cache_hit",
                    action_key=str(normalized["action_key"]),
                    job_id=None,
                    receipt=cached,
                )
        root = _absolute_path(checkout_root, where="checkout root")
        if not root.is_dir():
            raise pb.ActionContractError(f"checkout root is not a directory: {root}")
        self._check_worker_script()
        request = publish_action_request(normalized, cas_root=self.cas_root)
        key = str(normalized["action_key"])
        log_directory = self.log_root / key[:2]
        log_directory.mkdir(parents=True, exist_ok=True)
        worker_argv = self._worker_argv(
            request_path=request,
            checkout_root=root,
            recompute=recompute,
        )
        argv = [
            str(self.sbatch),
            "--parsable",
            "--export=NONE",
            "--requeue",
            "--nodes=1",
            "--ntasks=1",
            "--open-mode=append",
            f"--job-name=pq-{key[:24]}",
            f"--comment=prismabuild:{key}",
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
            str(self.worker_script),
            *worker_argv[1:],
        ])
        completed = self._run(argv)
        output = completed.stdout.strip()
        if "\n" in output or "\r" in output:
            raise SlurmProtocolError(
                f"sbatch --parsable returned multiple lines: {completed.stdout!r}"
            )
        job_id = parse_job_id(output)
        return SlurmSubmission(
            status="submitted",
            action_key=key,
            job_id=job_id,
            receipt=None,
        )

    def _query_state(self, job_id: SlurmJobId) -> tuple[str, str]:
        selector = str(job_id.number)
        squeue_argv = [
            str(self.squeue),
            "--noheader",
            f"--jobs={selector}",
            "--format=%i|%T",
            *self._cluster_args(job_id),
        ]
        current = self._run(squeue_argv)
        lines = [line.strip() for line in current.stdout.splitlines() if line.strip()]
        if len(lines) > 1:
            raise SlurmProtocolError(
                f"squeue returned ambiguous rows for job {job_id}: {lines!r}"
            )
        if lines:
            return self._parse_state_line(lines[0], job_id=job_id, source="squeue")

        accounting_argv = [
            str(self.sacct),
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={selector}",
            "--format=JobIDRaw,State",
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
        return self._parse_state_line(rows[0], job_id=job_id, source="sacct")

    @staticmethod
    def _parse_state_line(
        line: str, *, job_id: SlurmJobId, source: str
    ) -> tuple[str, str]:
        match = _STATE_LINE_RE.fullmatch(line)
        if match is None or int(match.group("job")) != job_id.number:
            raise SlurmProtocolError(
                f"{source} returned malformed or wrong-job row for {job_id}: {line!r}"
            )
        return _normalize_state(match.group("state"))

    def resolve(self, action: object, job_id: SlurmJobId | str) -> SlurmResolution:
        """Resolve an allocation, treating a valid CAS receipt as sole success."""

        normalized = pb.validate_action(action)
        job = parse_job_id(job_id) if isinstance(job_id, str) else job_id
        if not isinstance(job, SlurmJobId):
            raise pb.ActionContractError("job_id must be a SlurmJobId or parsable string")
        cas = pb.PrismaBuildCAS(self.cas_root)
        receipt = _validated_receipt(cas, normalized)
        if receipt is not None:
            return SlurmResolution(
                status="succeeded",
                action_key=str(normalized["action_key"]),
                job_id=job,
                slurm_state=None,
                reason="verified CAS receipt exists",
                receipt=receipt,
                payload_path=cas.result_path(receipt, normalized),
            )

        raw_state, category = self._query_state(job)
        if category == "pending":
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

    def cancel(self, job_id: SlurmJobId | str) -> None:
        job = parse_job_id(job_id) if isinstance(job_id, str) else job_id
        if not isinstance(job, SlurmJobId):
            raise pb.ActionContractError("job_id must be a SlurmJobId or parsable string")
        self._run(
            [
                str(self.scancel),
                *self._cluster_args(job),
                str(job.number),
            ]
        )

    def requeue(self, action: object, job_id: SlurmJobId | str) -> bool:
        """Explicitly retry an allocation unless the CAS already closed it.

        Returns ``False`` when a valid receipt makes the retry a no-op.  A
        requeued worker checks the same CAS first, so scheduler retries remain
        idempotent.  Dirty partial outputs still fail closed in the core worker.
        """

        normalized = pb.validate_action(action)
        job = parse_job_id(job_id) if isinstance(job_id, str) else job_id
        if not isinstance(job, SlurmJobId):
            raise pb.ActionContractError("job_id must be a SlurmJobId or parsable string")
        if _validated_receipt(pb.PrismaBuildCAS(self.cas_root), normalized) is not None:
            return False
        self._run(
            [
                str(self.scontrol),
                *self._cluster_args(job),
                "requeue",
                str(job.number),
            ]
        )
        return True


__all__ = [
    "SlurmAdapter",
    "SlurmAdapterError",
    "SlurmCommandError",
    "SlurmJobId",
    "SlurmPlacement",
    "SlurmProtocolError",
    "SlurmResolution",
    "SlurmResources",
    "SlurmSubmission",
    "SlurmUnavailableError",
    "parse_job_id",
    "publish_action_request",
    "validate_placement_scope",
]
