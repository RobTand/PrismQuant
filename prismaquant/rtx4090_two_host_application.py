"""Self-contained live application for the deterministic two-host campaign.

This module composes the pure campaign coordinator with concrete transports,
host admission, GPU-start guards, verified artifact barriers, and lease
release.  Operators invoke one CLI; no agent or manual SSH scheduler is part
of the runtime protocol.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import time
from types import ModuleType
from typing import Any, Callable

from prismaquant.cluster_campaign_contract import (
    LEGACY_CAMPAIGN_MANIFEST_SCHEMA,
    STAGE_DAG,
    canonical_sha256,
    validate_campaign_manifest,
)
from prismaquant.cluster_host_admission import (
    GPU_LEASE_SCHEMA,
    GPU_LEASE_OPERATION_SCHEMA,
    GPU_START_GUARD_RECEIPT_SCHEMA,
    HOST_PRE_ADMISSION_RECEIPT_SCHEMA,
    build_host_action_request,
)
from prismaquant.cluster_live_runtime import (
    ArtifactBarrierReceipt,
    LiveCampaignRuntime,
    _derived_control_paths,
    build_live_campaign_runtime,
)
from prismaquant.cluster_transport import (
    ClusterTransportError,
    JobReceipt,
    RunRequest,
    canonical_json_bytes,
    list_regular_file_children_nofollow,
    read_regular_file_nofollow,
)
from prismaquant.rtx4090_two_host_campaign import (
    CampaignCoordinator,
    RECEIPT_DIR,
)


HOST_ACTION_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_two_host_application.host_action.v1"
)
GUARD_JOURNAL_SCHEMA = (
    "prismaquant.rtx4090_two_host_application.guard_journal.v1"
)
MODEL_ADMISSION_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_two_host_application.model_admission.v1"
)
APPLICATION_RECEIPT_DIR = "application-receipts"
PRE_ADMISSION_DIR = "pre-admission"
BARRIER_DIR = "barriers"
MODEL_ADMISSION_DIR = "model-admission"
GUARD_DIR = "gpu-start-guards"
LEASE_RELEASE_DIR = "lease-release"

_HOST_ACTION_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "host_id",
    "action",
    "operation_token",
    "action_attempt_index",
    "request",
    "request_sha256",
    "job",
    "job_receipt_sha256",
    "result",
    "result_sha256",
    "identity_sha256",
})
_GUARD_JOURNAL_KEYS = frozenset({
    "schema",
    "campaign_identity_sha256",
    "host_id",
    "job_id",
    "request_sha256",
    "checks",
    "identity_sha256",
})
_GPU_LEASE_KEYS = frozenset({
    "schema",
    "campaign_id",
    "campaign_identity_sha256",
    "host_id",
    "host_identity_sha256",
    "gpu_uuid",
    "identity_sha256",
})
_GPU_LEASE_OPERATION_KEYS = frozenset({
    "schema",
    "action",
    "disposition",
    "lease",
    "identity_sha256",
})

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_STAGE_INDEX = {spec.stage: index for index, spec in enumerate(STAGE_DAG)}
_HOST_ACTION_POLL_SECONDS = 0.1
_HOST_ACTION_WAIT_SECONDS = 330.0


class RTX4090TwoHostApplicationError(RuntimeError):
    """The live application could not prove a required transition."""


def _state_path_without_symlink_ancestors(path: Path) -> Path:
    """Validate controller state without following any existing symlink."""

    if (
        not path.is_absolute()
        or path == Path("/")
        or ".." in path.parts
    ):
        raise RTX4090TwoHostApplicationError(
            "application state directory must be a normalized non-root absolute path"
        )
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RTX4090TwoHostApplicationError(
                "application state ancestry is unreadable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RTX4090TwoHostApplicationError(
                "application state ancestry contains a symlink"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise RTX4090TwoHostApplicationError(
                "application state ancestry contains a non-directory"
            )
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise RTX4090TwoHostApplicationError(
            "application state ancestry cannot be resolved"
        ) from exc


def _validate_application_state_path(
    manifest: Mapping[str, object], state_dir: str | Path,
) -> Path:
    """Prove controller state is outside the local container write surface."""

    state = Path(state_dir)
    effective_state = _state_path_without_symlink_ancestors(state)
    local_host = next(
        host for host in manifest["hosts"]  # type: ignore[index]
        if host["transport"]["kind"] == "local"  # type: ignore[index]
    )
    local_roots = local_host["roots"]
    assert isinstance(local_roots, Mapping)
    candidates: list[tuple[str, Path, bool]] = [
        (field, Path(str(local_roots[field])), field in {
            "run_root", "worker_state_root",
        })
        for field in (
            "model_root", "dataset_path", "snapshot_root", "run_root",
            "worker_state_root",
        )
    ]
    candidates.extend(
        (f"derived_control_{index}", Path(path.as_posix()), False)
        for index, path in enumerate(_derived_control_paths(
            local_host, str(manifest["identity_sha256"]),
        ))
    )
    for field, declared_root, container_writable in candidates:
        try:
            effective_root = declared_root.resolve(strict=False)
        except OSError as exc:
            raise RTX4090TwoHostApplicationError(
                f"local declared root {field} cannot be resolved"
            ) from exc
        if (
            state == declared_root
            or state.is_relative_to(declared_root)
            or declared_root.is_relative_to(state)
            or effective_state == effective_root
            or effective_state.is_relative_to(effective_root)
            or effective_root.is_relative_to(effective_state)
        ):
            if container_writable:
                raise RTX4090TwoHostApplicationError(
                    "application state must be outside container-writable "
                    f"mount {field}"
                )
            raise RTX4090TwoHostApplicationError(
                f"application state overlaps local declared root {field}"
            )
    return state


def _verify_controller_snapshot(
    manifest: Mapping[str, object],
    *,
    source_file: str | Path = __file__,
    verifier: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Require the controller itself to execute from the sealed local snapshot."""

    normalized = validate_campaign_manifest(manifest)
    local_host = next(
        host for host in normalized["hosts"]  # type: ignore[index]
        if host["transport"]["kind"] == "local"  # type: ignore[index]
    )
    roots = local_host["roots"]
    producer = normalized["producer"]
    assert isinstance(roots, Mapping) and isinstance(producer, Mapping)
    raw_snapshot = Path(str(roots["snapshot_root"]))
    raw_source = Path(source_file)
    try:
        if (
            not raw_snapshot.is_absolute()
            or not raw_source.is_absolute()
            or raw_snapshot.is_symlink()
            or raw_source.is_symlink()
        ):
            raise RTX4090TwoHostApplicationError(
                "controller source/snapshot must be absolute non-symlink paths"
            )
        snapshot = raw_snapshot.resolve(strict=True)
        source = raw_source.resolve(strict=True)
    except OSError as exc:
        raise RTX4090TwoHostApplicationError(
            "controller source/snapshot is absent or unreadable"
        ) from exc
    source_root = source.parents[1]
    if (
        snapshot != raw_snapshot
        or source != raw_source
        or not snapshot.is_dir()
        or not source.is_file()
        or source_root != snapshot
    ):
        raise RTX4090TwoHostApplicationError(
            "live controller is not executing from the manifest's local "
            "immutable snapshot"
        )

    active_verifier = verifier
    if active_verifier is None:
        expected_tool = snapshot / "tools/prismaquant_runtime_snapshot.py"
        try:
            if expected_tool.is_symlink():
                raise OSError("snapshot verifier is a symlink")
            tool_source = read_regular_file_nofollow(
                expected_tool, where="snapshot verifier",
            )
            snapshot_tool = ModuleType(
                "_prismaquant_campaign_snapshot_verifier"
            )
            snapshot_tool.__file__ = str(expected_tool)
            exec(
                compile(
                    tool_source,
                    str(expected_tool),
                    "exec",
                    dont_inherit=True,
                ),
                snapshot_tool.__dict__,
            )
        except Exception as exc:
            raise RTX4090TwoHostApplicationError(
                "snapshot verifier cannot be imported from the controller snapshot"
            ) from exc
        try:
            observed_tool = Path(snapshot_tool.__file__).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise RTX4090TwoHostApplicationError(
                "snapshot verifier origin is unreadable"
            ) from exc
        if observed_tool != expected_tool:
            raise RTX4090TwoHostApplicationError(
                "snapshot verifier resolved outside the controller snapshot"
            )
        active_verifier = snapshot_tool.verify_snapshot

    try:
        raw_receipt = active_verifier(
            snapshot,
            expected_commit=str(producer["commit"]),
            expected_tree=str(producer["tree"]),
            expected_closure_sha256=str(producer["snapshot_sha256"]),
        )
    except Exception as exc:
        raise RTX4090TwoHostApplicationError(
            f"controller snapshot verification failed: {exc}"
        ) from exc
    receipt = dict(raw_receipt)
    if (
        receipt.get("snapshot") != str(snapshot)
        or receipt.get("commit") != producer["commit"]
        or receipt.get("tree") != producer["tree"]
        or receipt.get("closure_sha256") != producer["snapshot_sha256"]
        or type(receipt.get("entry_count")) is not int
        or int(receipt["entry_count"]) < 1
    ):
        raise RTX4090TwoHostApplicationError(
            "controller snapshot receipt differs from the sealed producer"
        )
    return receipt


def _strict_json_bytes(payload: bytes, *, where: str) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RTX4090TwoHostApplicationError(
                    f"{where} contains duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {value}")
            ),
        )
    except RTX4090TwoHostApplicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RTX4090TwoHostApplicationError(
            f"{where} is not strict JSON"
        ) from exc


def _read_canonical_mapping(path: Path, *, where: str) -> dict[str, object]:
    try:
        payload = read_regular_file_nofollow(path, where=where)
    except ClusterTransportError as exc:
        raise RTX4090TwoHostApplicationError(f"{where} is unreadable") from exc
    value = _strict_json_bytes(payload, where=where)
    if not isinstance(value, Mapping) or canonical_json_bytes(value) + b"\n" != payload:
        raise RTX4090TwoHostApplicationError(
            f"{where} is not a canonical JSON object"
        )
    return dict(value)


def _read_sealed_mapping(path: Path, *, where: str) -> dict[str, object]:
    """Read a digest-sealed receipt whose producer owns its JSON formatting."""
    try:
        payload = read_regular_file_nofollow(path, where=where)
    except ClusterTransportError as exc:
        raise RTX4090TwoHostApplicationError(f"{where} is unreadable") from exc
    value = _strict_json_bytes(payload, where=where)
    return _validate_sealed(value, where=where)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_no_clobber(path: Path, value: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            existing = read_regular_file_nofollow(
                path, where="existing application receipt",
            )
        except ClusterTransportError as exc:
            raise RTX4090TwoHostApplicationError(
                f"existing application receipt is unsafe: {path}"
            ) from exc
        if existing != payload:
            raise RTX4090TwoHostApplicationError(
                f"existing application receipt differs: {path}"
            )
        return
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _replace_canonical(path: Path, value: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError as exc:
        raise RTX4090TwoHostApplicationError(
            f"stale application receipt staging file exists: {temporary}"
        ) from exc
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _seal(body: Mapping[str, object]) -> dict[str, object]:
    return {**body, "identity_sha256": canonical_sha256(body)}


def _expected_gpu_lease(
    manifest: Mapping[str, object], host_id: str,
) -> dict[str, object]:
    host = next(
        (
            candidate for candidate in manifest["hosts"]  # type: ignore[index]
            if candidate["id"] == host_id
        ),
        None,
    )
    if not isinstance(host, Mapping):
        raise RTX4090TwoHostApplicationError(
            f"unknown host for expected GPU lease: {host_id}"
        )
    expected = host.get("expected")
    if not isinstance(expected, Mapping):  # pragma: no cover - manifest gate
        raise RTX4090TwoHostApplicationError(
            f"host expected identity is absent: {host_id}"
        )
    gpu = expected.get("gpu")
    if not isinstance(gpu, Mapping):  # pragma: no cover - manifest gate
        raise RTX4090TwoHostApplicationError(
            f"host GPU identity is absent: {host_id}"
        )
    lease = _seal({
        "schema": GPU_LEASE_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_id": host_id,
        "host_identity_sha256": canonical_sha256(host),
        "gpu_uuid": gpu["uuid"],
    })
    if set(lease) != _GPU_LEASE_KEYS:  # pragma: no cover - construction guard
        raise RTX4090TwoHostApplicationError(
            f"expected GPU lease fields differ for {host_id}"
        )
    return lease


def _validate_sealed(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RTX4090TwoHostApplicationError(f"{where} is not an object")
    result = dict(value)
    digest = result.pop("identity_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(result):
        raise RTX4090TwoHostApplicationError(f"{where} digest differs")
    return {**result, "identity_sha256": digest}


def _normalize_job(
    value: object,
    request: RunRequest,
    *,
    require_succeeded: bool = True,
) -> JobReceipt:
    try:
        receipt = value if isinstance(value, JobReceipt) else JobReceipt.from_payload(value)
    except (TypeError, ValueError) as exc:
        raise RTX4090TwoHostApplicationError(
            "host action returned an invalid job receipt"
        ) from exc
    if (
        receipt.job_id != request.job_id
        or receipt.request_sha256 != request.request_sha256
        or (require_succeeded and not receipt.succeeded)
    ):
        raise RTX4090TwoHostApplicationError(
            f"host action job {request.job_id} did not succeed exactly"
        )
    return receipt


def _host_action_result(receipt: JobReceipt, *, where: str) -> dict[str, object]:
    payload = receipt.stdout
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise RTX4090TwoHostApplicationError(
            f"{where} did not emit exactly one JSON receipt"
        )
    value = _strict_json_bytes(payload[:-1], where=where)
    return _validate_sealed(value, where=where)


@dataclass
class _HostActionExecutor:
    manifest: Mapping[str, object]
    host_id: str
    transport: Any

    def _wait_for_terminal(
        self,
        request: RunRequest,
        initial: object | None,
        *,
        deadline: float,
    ) -> JobReceipt:
        receipt = (
            None
            if initial is None
            else _normalize_job(
                initial, request, require_succeeded=False,
            )
        )
        last_status_error: Exception | None = None
        while receipt is None or receipt.state == "running":
            if time.monotonic() >= deadline:
                detail = (
                    f": {last_status_error}"
                    if last_status_error is not None else ""
                )
                raise RTX4090TwoHostApplicationError(
                    f"host action job {request.job_id} did not reach a durable "
                    f"terminal state{detail}"
                )
            time.sleep(_HOST_ACTION_POLL_SECONDS)
            try:
                receipt = _normalize_job(
                    self.transport.status(request.job_id),
                    request,
                    require_succeeded=False,
                )
                last_status_error = None
            except Exception as exc:
                last_status_error = exc
        return _normalize_job(receipt, request, require_succeeded=False)

    @staticmethod
    def _attempt_token(token: str, attempt_index: int) -> str:
        value = f"{token}-{attempt_index:03d}"
        if len(value) > 32:
            raise RTX4090TwoHostApplicationError(
                "host action logical token cannot carry a retry index"
            )
        return value

    def run(self, action: str, *, token: str) -> dict[str, object]:
        retry = self.manifest["policy"]["retry"]  # type: ignore[index]
        max_new_attempts = int(retry["max_attempts"])
        new_attempts = 0
        last_failure: JobReceipt | None = None
        for attempt_index in range(1000):
            request = build_host_action_request(
                action,  # type: ignore[arg-type]
                self.manifest,
                self.host_id,
                operation_token=self._attempt_token(token, attempt_index),
            )
            deadline = time.monotonic() + _HOST_ACTION_WAIT_SECONDS
            raw_job: object | None
            try:
                raw_job = self.transport.status(request.job_id)
            except Exception:
                if new_attempts >= max_new_attempts:
                    break
                new_attempts += 1
                try:
                    raw_job = self.transport.start(request)
                except Exception:
                    # A start response may be lost after the durable claim.
                    # Poll the exact ID rather than issuing a colliding start.
                    raw_job = None
            job = self._wait_for_terminal(
                request, raw_job, deadline=deadline,
            )
            if not job.succeeded:
                last_failure = job
                continue
            result = _host_action_result(
                job, where=f"host action {action} on {self.host_id}",
            )
            body: dict[str, object] = {
                "schema": HOST_ACTION_RECEIPT_SCHEMA,
                "campaign_identity_sha256": self.manifest["identity_sha256"],
                "host_id": self.host_id,
                "action": action,
                "operation_token": token,
                "action_attempt_index": attempt_index,
                "request": request.to_payload(),
                "request_sha256": request.request_sha256,
                "job": job.to_payload(),
                "job_receipt_sha256": job.receipt_sha256,
                "result": result,
                "result_sha256": str(result["identity_sha256"]),
            }
            return _seal(body)
        detail = (
            "no unused deterministic action token remains"
            if last_failure is None
            else f"last terminal state was {last_failure.state}"
        )
        raise RTX4090TwoHostApplicationError(
            f"host action {action} exhausted {max_new_attempts} new "
            f"attempt(s) on {self.host_id}; {detail}"
        )

    def require_prior_terminal_failure(
        self,
        action: str,
        *,
        token: str,
        before_attempt_index: int,
        verify_only: bool,
    ) -> None:
        """Prove a prior exact action could have crossed its receipt window."""

        if before_attempt_index <= 0:
            raise RTX4090TwoHostApplicationError(
                f"host action {action} has no prior retry attempt on {self.host_id}"
            )
        reader = self.transport.status
        if verify_only:
            reader = getattr(self.transport, "inspect", None)
            if not callable(reader):
                raise RTX4090TwoHostApplicationError(
                    f"host action transport for {self.host_id} has no "
                    "non-mutating durable inspection path"
                )
        for attempt_index in range(before_attempt_index):
            request = build_host_action_request(
                action,  # type: ignore[arg-type]
                self.manifest,
                self.host_id,
                operation_token=self._attempt_token(token, attempt_index),
            )
            try:
                job = _normalize_job(
                    reader(request.job_id),
                    request,
                    require_succeeded=False,
                )
            except Exception:
                continue
            if job.state in {"failed", "timed_out", "transport_error"}:
                return
        raise RTX4090TwoHostApplicationError(
            f"host action {action} has no exact prior terminal failure on "
            f"{self.host_id}"
        )


class _GuardedTransport:
    """Run a fresh host launch check immediately before each container start."""

    def __init__(
        self,
        inner: Any,
        *,
        application: "LiveCampaignApplication",
        host_id: str,
    ) -> None:
        self.inner = inner
        self.application = application
        self.host_id = host_id
        self.cadence_seconds = inner.cadence_seconds

    def start(self, request: RunRequest) -> object:
        if dict(request.env).get("PQ_CAMPAIGN_START_GUARD") == "1":
            self.application.record_gpu_start_guard(self.host_id, request)
        return self.inner.start(request)

    def status(self, job_id: str) -> object:
        return self.inner.status(job_id)

    def sample_telemetry(self) -> object:
        return self.inner.sample_telemetry()

    def validate_existing_guard(self, request: RunRequest) -> None:
        if dict(request.env).get("PQ_CAMPAIGN_START_GUARD") == "1":
            self.application.validate_gpu_start_guard(self.host_id, request)


class LiveCampaignApplication:
    """Application-owned lifecycle around :class:`CampaignCoordinator`."""

    def __init__(
        self,
        manifest: Mapping[str, object],
        state_dir: str | Path,
        *,
        runtime: LiveCampaignRuntime,
    ) -> None:
        self.manifest = validate_campaign_manifest(manifest)
        self.state_dir = _validate_application_state_path(
            self.manifest, state_dir,
        )
        self.runtime = runtime
        self.application_root = self.state_dir / APPLICATION_RECEIPT_DIR
        raw_transports = {
            str(host["id"]): (
                runtime.local_transport
                if host["transport"]["kind"] == "local"  # type: ignore[index]
                else runtime.ssh_transport
            )
            for host in self.manifest["hosts"]  # type: ignore[index]
        }
        self._actions = {
            host_id: _HostActionExecutor(self.manifest, host_id, transport)
            for host_id, transport in raw_transports.items()
        }
        guarded = {
            host_id: _GuardedTransport(
                runtime.transports[host_id], application=self, host_id=host_id,
            )
            for host_id in raw_transports
        }
        self.coordinator = CampaignCoordinator(
            self.manifest,
            self.state_dir,
            transports=guarded,
            artifact_inspectors=runtime.artifact_inspectors,
            stage_preconditioner=self.stage_preconditions,
        )

    def _path(self, category: str, name: str) -> Path:
        if _SAFE_NAME.fullmatch(name) is None:
            raise RTX4090TwoHostApplicationError(
                f"unsafe application receipt name {name!r}"
            )
        return self.application_root / category / f"{name}.json"

    def _load_host_action(
        self,
        path: Path,
        *,
        action: str,
        host_id: str,
        token: str,
    ) -> dict[str, object]:
        value = _validate_sealed(
            _read_canonical_mapping(path, where=f"host action {path.name}"),
            where=f"host action {path.name}",
        )
        attempt_index = value.get("action_attempt_index")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or not 0 <= attempt_index <= 999
        ):
            raise RTX4090TwoHostApplicationError(
                f"stored host action retry index differs: {path}"
            )
        expected_request = build_host_action_request(
            action,  # type: ignore[arg-type]
            self.manifest,
            host_id,
            operation_token=_HostActionExecutor._attempt_token(
                token, attempt_index,
            ),
        )
        if (
            set(value) != _HOST_ACTION_KEYS
            or value.get("schema") != HOST_ACTION_RECEIPT_SCHEMA
            or value.get("campaign_identity_sha256")
            != self.manifest["identity_sha256"]
            or value.get("host_id") != host_id
            or value.get("action") != action
            or value.get("operation_token") != token
            or value.get("action_attempt_index") != attempt_index
            or value.get("request") != expected_request.to_payload()
            or value.get("request_sha256") != expected_request.request_sha256
        ):
            raise RTX4090TwoHostApplicationError(
                f"stored host action binding differs: {path}"
            )
        job = _normalize_job(value.get("job"), expected_request)
        result = _host_action_result(
            job, where=f"stored host action {action} on {host_id}",
        )
        if (
            value.get("job_receipt_sha256") != job.receipt_sha256
            or value.get("result") != result
            or value.get("result_sha256") != result["identity_sha256"]
        ):
            raise RTX4090TwoHostApplicationError(
                f"stored host action payload differs: {path}"
            )
        return value

    def _ensure_host_action(
        self,
        *,
        category: str,
        name: str,
        action: str,
        host_id: str,
        token: str,
        verify_only: bool,
    ) -> dict[str, object]:
        path = self._path(category, name)
        if path.exists():
            return self._load_host_action(
                path, action=action, host_id=host_id, token=token,
            )
        if verify_only:
            raise RTX4090TwoHostApplicationError(
                f"required host action receipt is absent: {path}"
            )
        value = self._actions[host_id].run(action, token=token)
        _write_no_clobber(path, value)
        return self._load_host_action(
            path, action=action, host_id=host_id, token=token,
        )

    def _ensure_barrier(
        self, stage: str, *, verify_only: bool,
    ) -> ArtifactBarrierReceipt | None:
        routes = self.runtime.route_specs_for_stage(stage)
        if not routes:
            return None
        path = self._path(BARRIER_DIR, stage)
        if path.exists():
            raw = _read_canonical_mapping(
                path, where=f"artifact barrier {stage}",
            )
            return self.runtime.validate_barrier_receipt(stage, raw)
        if verify_only:
            raise RTX4090TwoHostApplicationError(
                f"required artifact barrier is absent: {stage}"
            )
        receipt = self.runtime.synchronize_before_stage(stage)
        if receipt is None:
            raise RTX4090TwoHostApplicationError(
                f"runtime omitted required artifact barrier {stage}"
            )
        _write_no_clobber(path, receipt.to_payload())
        return self.runtime.validate_barrier_receipt(
            stage,
            _read_canonical_mapping(path, where=f"artifact barrier {stage}"),
        )

    def _pre_admissions(self, *, verify_only: bool) -> tuple[dict[str, object], ...]:
        values = []
        for host in self.manifest["hosts"]:  # type: ignore[index]
            host_id = str(host["id"])
            value = self._ensure_host_action(
                category=PRE_ADMISSION_DIR,
                name=host_id,
                action="pre-admit",
                host_id=host_id,
                token="initial",
                verify_only=verify_only,
            )
            result = value["result"]
            if (
                not isinstance(result, Mapping)
                or result.get("schema") != HOST_PRE_ADMISSION_RECEIPT_SCHEMA
                or result.get("campaign_identity_sha256")
                != self.manifest["identity_sha256"]
                or result.get("host_id") != host_id
                or not isinstance(result.get("lease"), Mapping)
            ):
                raise RTX4090TwoHostApplicationError(
                    f"pre-admission result differs for {host_id}"
                )
            values.append(value)
        return tuple(values)

    def _source_execution_receipt(self, host_id: str) -> dict[str, object]:
        path = (
            self.state_dir / RECEIPT_DIR /
            f"worker_source_identity:{host_id}.json"
        )
        return _read_sealed_mapping(
            path, where=f"worker source execution receipt for {host_id}",
        )

    def _model_admission_value(self, host_id: str) -> dict[str, object]:
        execution = self._source_execution_receipt(host_id)
        try:
            job = JobReceipt.from_payload(execution["job"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RTX4090TwoHostApplicationError(
                f"worker source job receipt is invalid for {host_id}"
            ) from exc
        lines = [line for line in job.stdout.splitlines() if line.strip()]
        if not lines:
            raise RTX4090TwoHostApplicationError(
                f"worker source job emitted no receipt for {host_id}"
            )
        source = _strict_json_bytes(
            lines[-1], where=f"worker source receipt for {host_id}",
        )
        if not isinstance(source, Mapping):
            raise RTX4090TwoHostApplicationError(
                f"worker source receipt is not an object for {host_id}"
            )
        portable = source.get("portable_identity")
        inputs = self.manifest["inputs"]
        assert isinstance(inputs, Mapping)
        if (
            source.get("schema")
            != "prismaquant.sample_parallel_probe.worker_source_cache_receipt.v1"
            or not isinstance(portable, Mapping)
            or portable.get("schema")
            != "prismaquant.streamed_model.portable_content.v1"
            or portable.get("portable_content_sha256")
            != inputs["model_content_sha256"]
            or type(portable.get("checkpoint_shards")) is not int
            or int(portable["checkpoint_shards"]) < 1
            or type(portable.get("checkpoint_tensors")) is not int
            or int(portable["checkpoint_tensors"]) < 1
        ):
            raise RTX4090TwoHostApplicationError(
                f"worker source content identity differs for {host_id}"
            )
        body: dict[str, object] = {
            "schema": MODEL_ADMISSION_RECEIPT_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "host_id": host_id,
            "source_execution_receipt_sha256": execution["identity_sha256"],
            "job_receipt_sha256": job.receipt_sha256,
            "portable_identity": dict(portable),
        }
        return _seal(body)

    def _model_admissions(self, *, verify_only: bool) -> tuple[dict[str, object], ...]:
        values = []
        for host in self.manifest["hosts"]:  # type: ignore[index]
            host_id = str(host["id"])
            path = self._path(MODEL_ADMISSION_DIR, host_id)
            expected = self._model_admission_value(host_id)
            if path.exists():
                stored = _validate_sealed(
                    _read_canonical_mapping(
                        path, where=f"model admission for {host_id}",
                    ),
                    where=f"model admission for {host_id}",
                )
                if stored != expected:
                    raise RTX4090TwoHostApplicationError(
                        f"model admission receipt differs for {host_id}"
                    )
            elif verify_only:
                raise RTX4090TwoHostApplicationError(
                    f"model admission receipt is absent for {host_id}"
                )
            else:
                _write_no_clobber(path, expected)
                stored = expected
            values.append(stored)
        return tuple(values)

    def stage_preconditions(
        self, stage: str, *, verify_only: bool,
    ) -> Sequence[str]:
        if (
            not verify_only
            and self.manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA
        ):
            raise RTX4090TwoHostApplicationError(
                "legacy campaign manifests are read-only"
            )
        if stage not in _STAGE_INDEX:
            raise RTX4090TwoHostApplicationError(
                f"unknown campaign stage {stage!r}"
            )
        receipts: list[Mapping[str, object]] = list(
            self._pre_admissions(verify_only=True)
        )
        barrier = self._ensure_barrier(stage, verify_only=verify_only)
        if barrier is not None:
            receipts.append(barrier.to_payload())
        if _STAGE_INDEX[stage] >= _STAGE_INDEX["sample_ce"]:
            receipts.extend(self._model_admissions(verify_only=verify_only))
        return tuple(sorted({str(item["identity_sha256"]) for item in receipts}))

    def prepare(self, *, resume: bool) -> None:
        if self.manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
            raise RTX4090TwoHostApplicationError(
                "legacy campaign manifests are read-only"
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # The remote host cannot execute even the stdlib-only admission module
        # until the exact immutable snapshot has been published there.
        self._ensure_barrier("host_preflight", verify_only=False)
        # Existing receipts are always validated exactly; a resume is also
        # allowed to adopt or finish a host action whose peer already holds
        # the campaign lease.
        self._pre_admissions(verify_only=False)

    def _guard_path(self, request: RunRequest) -> Path:
        return self._path(GUARD_DIR, request.job_id)

    def _load_guard_journal(
        self, host_id: str, request: RunRequest,
    ) -> dict[str, object] | None:
        path = self._guard_path(request)
        if not path.exists():
            return None
        value = _validate_sealed(
            _read_canonical_mapping(path, where=f"GPU guard {request.job_id}"),
            where=f"GPU guard {request.job_id}",
        )
        checks = value.get("checks")
        if (
            set(value) != _GUARD_JOURNAL_KEYS
            or value.get("schema") != GUARD_JOURNAL_SCHEMA
            or value.get("campaign_identity_sha256")
            != self.manifest["identity_sha256"]
            or value.get("host_id") != host_id
            or value.get("job_id") != request.job_id
            or value.get("request_sha256") != request.request_sha256
            or not isinstance(checks, list)
            or not checks
        ):
            raise RTX4090TwoHostApplicationError(
                f"GPU guard journal differs for {request.job_id}"
            )
        for index, check in enumerate(checks):
            token = f"{request.request_sha256[:20]}-{index:02d}"
            expected_path = self._path(
                GUARD_DIR, f"{request.job_id}.check-{index:02d}",
            )
            stored = self._load_host_action(
                expected_path,
                action="guard",
                host_id=host_id,
                token=token,
            )
            result = stored.get("result")
            if (
                check != stored
                or not isinstance(result, Mapping)
                or result.get("schema") != GPU_START_GUARD_RECEIPT_SCHEMA
                or result.get("campaign_identity_sha256")
                != self.manifest["identity_sha256"]
                or result.get("host_id") != host_id
            ):
                raise RTX4090TwoHostApplicationError(
                    f"GPU guard check differs for {request.job_id}"
                )
        return value

    def record_gpu_start_guard(self, host_id: str, request: RunRequest) -> None:
        if self.manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
            raise RTX4090TwoHostApplicationError(
                "legacy campaign manifests are read-only"
            )
        current = self._load_guard_journal(host_id, request)
        checks = [] if current is None else list(current["checks"])
        index = len(checks)
        if index > 99:
            raise RTX4090TwoHostApplicationError(
                f"GPU guard journal exhausted for {request.job_id}"
            )
        token = f"{request.request_sha256[:20]}-{index:02d}"
        check_path = self._path(
            GUARD_DIR, f"{request.job_id}.check-{index:02d}",
        )
        check = self._actions[host_id].run("guard", token=token)
        _write_no_clobber(check_path, check)
        check = self._load_host_action(
            check_path, action="guard", host_id=host_id, token=token,
        )
        result = check["result"]
        if (
            not isinstance(result, Mapping)
            or result.get("schema") != GPU_START_GUARD_RECEIPT_SCHEMA
            or result.get("campaign_identity_sha256")
            != self.manifest["identity_sha256"]
            or result.get("host_id") != host_id
        ):
            raise RTX4090TwoHostApplicationError(
                f"GPU guard result differs for {request.job_id}"
            )
        checks.append(check)
        body: dict[str, object] = {
            "schema": GUARD_JOURNAL_SCHEMA,
            "campaign_identity_sha256": self.manifest["identity_sha256"],
            "host_id": host_id,
            "job_id": request.job_id,
            "request_sha256": request.request_sha256,
            "checks": checks,
        }
        value = _seal(body)
        path = self._guard_path(request)
        if current is None:
            _write_no_clobber(path, value)
        else:
            _replace_canonical(path, value)
        self.validate_gpu_start_guard(host_id, request)

    def validate_gpu_start_guard(self, host_id: str, request: RunRequest) -> None:
        if self._load_guard_journal(host_id, request) is None:
            raise RTX4090TwoHostApplicationError(
                f"GPU job has no durable start guard: {request.job_id}"
            )

    def _release_leases(self, *, verify_only: bool) -> tuple[dict[str, object], ...]:
        values = []
        for host in self.manifest["hosts"]:  # type: ignore[index]
            host_id = str(host["id"])
            value = self._ensure_host_action(
                category=LEASE_RELEASE_DIR,
                name=host_id,
                action="release",
                host_id=host_id,
                token="complete",
                verify_only=verify_only,
            )
            result = value["result"]
            expected_lease = _expected_gpu_lease(self.manifest, host_id)
            if (
                not isinstance(result, Mapping)
                or set(result) != _GPU_LEASE_OPERATION_KEYS
                or result.get("schema") != GPU_LEASE_OPERATION_SCHEMA
                or result.get("action") != "release"
                or result.get("lease") != expected_lease
            ):
                raise RTX4090TwoHostApplicationError(
                    f"lease-release result differs for {host_id}"
                )
            disposition = result.get("disposition")
            if disposition == "already_absent":
                self._actions[host_id].require_prior_terminal_failure(
                    "release",
                    token="complete",
                    before_attempt_index=int(value["action_attempt_index"]),
                    verify_only=verify_only,
                )
            elif disposition != "released":
                raise RTX4090TwoHostApplicationError(
                    f"lease-release result differs for {host_id}"
                )
            values.append(value)
        return tuple(values)

    def verify_guard_receipts(self) -> None:
        required: set[str] = set()
        receipts_root = self.state_dir / RECEIPT_DIR
        try:
            receipt_paths = list_regular_file_children_nofollow(
                receipts_root,
                where="execution receipt directory",
                allow_missing=True,
            )
        except ClusterTransportError as exc:
            raise RTX4090TwoHostApplicationError(
                "execution receipt directory is unsafe"
            ) from exc
        for path in receipt_paths:
            receipt = _read_sealed_mapping(
                path, where=f"execution receipt {path.name}",
            )
            command = self.coordinator.commands[str(receipt["work_id"])]
            if command.module is None:
                continue
            job = JobReceipt.from_payload(receipt["job"])
            request = self.coordinator._request(
                command,
                attempt_index=int(receipt["attempt_index"]),
                precondition_receipt_sha256s=receipt[
                    "precondition_receipt_sha256s"
                ],
            )
            if job.request_sha256 != request.request_sha256:
                raise RTX4090TwoHostApplicationError(
                    f"guarded execution request differs for {path.name}"
                )
            self.validate_gpu_start_guard(str(receipt["host_id"]), request)
            required.add(f"{request.job_id}.json")

        allowed: dict[str, tuple[str, RunRequest]] = {}
        preconditions: dict[str, Sequence[str]] = {}
        for command in self.coordinator.commands.values():
            if command.module is None:
                continue
            values = preconditions.setdefault(
                command.stage,
                self.stage_preconditions(command.stage, verify_only=True),
            )
            for attempt_index in range(self.coordinator.max_attempts):
                request = self.coordinator._request(
                    command,
                    attempt_index=attempt_index,
                    precondition_receipt_sha256s=values,
                )
                filename = f"{request.job_id}.json"
                if filename in allowed:
                    raise RTX4090TwoHostApplicationError(
                        "guarded request catalog contains a duplicate job ID"
                    )
                allowed[filename] = (command.host_id, request)

        guard_root = self.application_root / GUARD_DIR
        try:
            guard_paths = list_regular_file_children_nofollow(
                guard_root,
                where="GPU start-guard directory",
                allow_missing=True,
            )
        except ClusterTransportError as exc:
            raise RTX4090TwoHostApplicationError(
                "GPU start-guard directory is unsafe"
            ) from exc
        actual_journals = {
            path.name for path in guard_paths
            if ".check-" not in path.name
        }
        if not required.issubset(actual_journals):
            raise RTX4090TwoHostApplicationError(
                "successful guarded executions are missing start journals"
            )
        unknown = actual_journals.difference(allowed)
        if unknown:
            raise RTX4090TwoHostApplicationError(
                "start-guard journals include a noncanonical campaign attempt"
            )

        expected_checks: set[str] = set()
        for filename in sorted(actual_journals):
            host_id, request = allowed[filename]
            journal = self._load_guard_journal(host_id, request)
            if journal is None:  # pragma: no cover - file was just enumerated
                raise RTX4090TwoHostApplicationError(
                    f"start-guard journal disappeared: {filename}"
                )
            checks = journal["checks"]
            assert isinstance(checks, list)
            expected_checks.update(
                f"{request.job_id}.check-{index:02d}.json"
                for index in range(len(checks))
            )
        actual_checks = {
            path.name for path in guard_paths
            if ".check-" in path.name
        }
        if actual_checks != expected_checks:
            raise RTX4090TwoHostApplicationError(
                "start-guard action receipts differ from their journals"
            )

    def verify_application(self, *, require_release: bool) -> None:
        self._pre_admissions(verify_only=True)
        for stage in self.runtime.barrier_stages:
            self._ensure_barrier(stage, verify_only=True)
        self._model_admissions(verify_only=True)
        self.verify_guard_receipts()
        if require_release:
            self._release_leases(verify_only=True)

    def run_to_completion(self, *, resume: bool) -> dict[str, object]:
        self.prepare(resume=resume)
        result = self.coordinator.run_to_completion(resume=resume)
        self.verify_application(require_release=False)
        self._release_leases(verify_only=False)
        self.verify_application(require_release=True)
        return result

    def verify(self) -> dict[str, object]:
        result = self.coordinator.verify()
        if self.manifest["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
            if result["complete"]:
                self.verify_application(require_release=True)
            else:
                # An incomplete historical generation has no release receipt.
                # Its compatibility route validates only immutable campaign
                # state and the exact guards for completed GPU children.
                self.verify_guard_receipts()
        else:
            self.verify_application(require_release=True)
        return result


def build_live_campaign_application(
    manifest: Mapping[str, object],
    state_dir: str | Path,
    *,
    initialize: bool,
) -> LiveCampaignApplication:
    normalized = validate_campaign_manifest(manifest)
    if initialize and normalized["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
        raise RTX4090TwoHostApplicationError(
            "legacy campaign manifests are read-only"
        )
    _validate_application_state_path(normalized, state_dir)
    historical_helper_source: bytes | None = None
    if normalized["schema"] == LEGACY_CAMPAIGN_MANIFEST_SCHEMA:
        local_host = next(
            host for host in normalized["hosts"]  # type: ignore[index]
            if host["transport"]["kind"] == "local"  # type: ignore[index]
        )
        roots = local_host["roots"]
        assert isinstance(roots, Mapping)
        snapshot = Path(str(roots["snapshot_root"]))
        _verify_controller_snapshot(
            normalized,
            source_file=(
                snapshot / "prismaquant/rtx4090_two_host_application.py"
            ),
        )
        helper_path = snapshot / "prismaquant/cluster_transport.py"
        try:
            helper_stat = helper_path.lstat()
            if not stat.S_ISREG(helper_stat.st_mode) or helper_path.is_symlink():
                raise RTX4090TwoHostApplicationError(
                    "historical transport helper is not a regular file"
                )
            historical_helper_source = read_regular_file_nofollow(
                helper_path, where="historical transport helper",
            )
        except (OSError, ClusterTransportError) as exc:
            raise RTX4090TwoHostApplicationError(
                "historical transport helper is absent or unreadable"
            ) from exc
        if not historical_helper_source:
            raise RTX4090TwoHostApplicationError(
                "historical transport helper is empty"
            )
    else:
        # A current campaign must execute from its manifest-pinned snapshot.
        _verify_controller_snapshot(normalized)
    runtime = build_live_campaign_runtime(
        normalized,
        initialize=initialize,
        ssh_helper_source=historical_helper_source,
    )
    return LiveCampaignApplication(normalized, state_dir, runtime=runtime)


__all__ = [
    "APPLICATION_RECEIPT_DIR",
    "GUARD_JOURNAL_SCHEMA",
    "HOST_ACTION_RECEIPT_SCHEMA",
    "LiveCampaignApplication",
    "MODEL_ADMISSION_RECEIPT_SCHEMA",
    "RTX4090TwoHostApplicationError",
    "build_live_campaign_application",
]
