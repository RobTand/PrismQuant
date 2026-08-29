from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import pytest

import prismaquant.rtx4090_two_host_campaign as campaign_module

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CAMPAIGN_STATE_SCHEMA,
    LEGACY_CAMPAIGN_MANIFEST_SCHEMA,
    STAGE_RECEIPT_SCHEMA,
    ClusterCampaignContractError,
    StageAssignment,
    bind_gridbook_runtime_contract,
    canonical_sha256,
    next_ready_assignments,
    seal_campaign_manifest,
)
from prismaquant.cluster_transport import (
    GpuSample,
    JobConflictError,
    JobNotFoundError,
    JobReceipt,
    ManifestEntry,
    RunRequest,
    TelemetryUnavailableError,
    TelemetrySnapshot,
    TreeManifest,
)
from prismaquant.rtx4090_two_host_campaign import (
    ATTEMPT_TELEMETRY_SCHEMA,
    ATTEMPT_TELEMETRY_RECORD_SCHEMA,
    CampaignCoordinator,
    EXECUTION_RECEIPT_SCHEMA,
    LEGACY_ATTEMPT_TELEMETRY_SCHEMA,
    LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA,
    LEGACY_EXECUTION_RECEIPT_SCHEMA,
    RTX4090TwoHostCampaignError,
    _host_output_path,
    _normalize_attempt_telemetry_evidence,
    _normalize_telemetry,
    _stage_receipt,
    build_command_plan,
    main,
)
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_TARGET_BYTES,
    RTX4090_QWEN38_FORMAT_MENU,
)


def test_sealed_artifact_target_matches_the_numerical_burn_policy() -> None:
    target = _manifest()["artifact_target"]

    assert target["artifact_max_bytes"] == RTX4090_CONTEXT_FIRST_TARGET_BYTES
    assert target["physical_formats"] == list(RTX4090_QWEN38_FORMAT_MENU[:-1])
    assert target["terminal_format"] == RTX4090_QWEN38_FORMAT_MENU[-1]


def _host(
    host_id: str,
    *,
    local: bool,
    gpu_name: str,
    gpu_uuid: str,
    compute_capability: list[int],
) -> dict[str, object]:
    suffix = host_id
    transport: dict[str, object] = (
        {"kind": "local"}
        if local
        else {
            "kind": "ssh",
            "host": "sparklina.example.test",
            "port": 22,
            "user": "campaign_runner",
        }
    )
    return {
        "id": host_id,
        "transport": transport,
        "roots": {
            "model_root": f"/srv/{suffix}/model",
            "dataset_path": f"/srv/{suffix}/data/calibration.jsonl",
            "snapshot_root": f"/srv/{suffix}/snapshot",
            "run_root": f"/srv/{suffix}/campaign-run",
            "worker_state_root": f"/srv/{suffix}/worker-state",
        },
        "expected": {
            "hostname": f"{suffix}-host",
            "gpu": {
                "name": gpu_name,
                "uuid": gpu_uuid,
                "compute_capability": compute_capability,
                "device_count": 1,
            },
            "image_digest": "sha256:" + "4" * 64,
            "producer_commit": "1" * 40,
            "uid": 1000,
            "gid": 1000,
        },
    }


def _manifest() -> dict[str, object]:
    # The logical coordinator is deliberately the SSH GB10 host.  RTX4090 is
    # the burn policy name, not an execution-host admission predicate.
    return seal_campaign_manifest({
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_id": "qwen38-27b-bf16-two-host-burn",
        "coordinator": "zeta",
        "artifact_target": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_capability": [8, 9],
            "artifact_max_bytes": 18_000_000_000,
            "disposition": "validation_only",
            "source_dtype": "bf16",
            "physical_formats": [
                "FP8_CB_K40", "FP8_CB_K44", "FP8_CB_K48", "FP8_E4M3",
            ],
            "terminal_format": "BF16",
            "allocation_objective": "context_first",
        },
        "producer": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "snapshot_sha256": "3" * 64,
            "image_digest": "sha256:" + "4" * 64,
        },
        "inputs": {
            "model_content_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "gridbook_runtime_contract": bind_gridbook_runtime_contract({
                "schema": "gridbook.runtime-contract.v11",
                "test_fixture": True,
            }),
            "sample_parallel": {
                "nsamples": 32,
                "seqlen": 1024,
                "calib_seed": 42,
                "activation_rows_limit": 1024,
            },
        },
        "policy": {
            "retry": {"max_attempts": 2},
            "telemetry": {
                "interval_milliseconds": 1000,
                "maximum_observation_gap_milliseconds": 30_000,
                "minimum_successful_sample_percent": 50,
                "require_positive_gpu_utilization": True,
            },
            "resources": {
                "coordinator_min_free_bytes": 100 * 1024**3,
                "worker_min_free_bytes": 40 * 1024**3,
                "min_mem_available_bytes": 64 * 1024**3,
            },
            "outputs": {
                "owner": "coordinator",
                "transfer_mode": "sha256_no_clobber",
            },
        },
        "hosts": [
            _host(
                "zeta",
                local=False,
                gpu_name="NVIDIA GB10",
                gpu_uuid="GPU-ZETA-0123456789",
                compute_capability=[12, 1],
            ),
            _host(
                "alpha",
                local=True,
                gpu_name="NVIDIA GeForce RTX 4090",
                gpu_uuid="GPU-ALPHA-0123456789",
                compute_capability=[8, 9],
            ),
        ],
    })


def _argument(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


class _FakeTransport:
    def __init__(
        self,
        expected: dict[str, object],
        *,
        fail_once_stage: str | None = None,
        omit_telemetry_stage: str | None = None,
        interrupt_telemetry_once_stage: str | None = None,
    ) -> None:
        self.expected = expected
        self.fail_once_stage = fail_once_stage
        self.omit_telemetry_stage = omit_telemetry_stage
        self.interrupt_telemetry_once_stage = interrupt_telemetry_once_stage
        self.start_calls: list[object] = []
        self.status_calls: list[str] = []
        self.jobs: dict[str, dict[str, object]] = {}
        self._counter = 0
        self._telemetry_counter = 0
        self._interrupted = False
        self._active_job_id: str | None = None
        self._lock = threading.Lock()

    @property
    def calls(self) -> list[object]:
        return self.start_calls

    @staticmethod
    def _stage(job_id: str) -> str:
        for stage in (
            "host_preflight",
            "prepare_calibration",
            "coordinator_source_identity",
            "prepare_run_contract",
            "build_sample_cover",
            "worker_source_identity",
            "sample_ce",
            "merge_importance",
            "sample_fisher",
            "merge_sample_probe",
            "derive_col_weights",
            "attest_execution",
            "prepare_burn",
            "measure_burn",
            "merge_burn",
            "allocate",
            "export_validation_artifact",
            "qualify_validation_artifact",
        ):
            if f"-{stage}-" in job_id:
                return stage
        raise AssertionError(f"unknown fixed-stage job id {job_id}")

    @staticmethod
    def _attempt(job_id: str) -> int:
        match = re.search(r"-attempt-(\d+)$", job_id)
        assert match is not None
        return int(match.group(1))

    @staticmethod
    def _receipt(
        request, *, state: str, ordinal: int,
    ) -> JobReceipt:
        running = state == "running"
        return JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state=state,
            started_ns=ordinal * 10 + 1,
            finished_ns=None if running else ordinal * 10 + 9,
            returncode=(None if running or state == "transport_error"
                        else (0 if state == "succeeded" else 9)),
            pid=1000 + ordinal,
            stdout=b"" if running else b"ok\n",
            stderr=b"" if state in {"running", "succeeded"} else b"failed\n",
            transport="fake",
        )

    def preload(self, request, *, state: str = "succeeded") -> None:
        with self._lock:
            self._counter += 1
            ordinal = self._counter
            self.jobs[request.job_id] = {
                "request": request,
                "ordinal": ordinal,
                "polls": 0,
                "final_state": state,
                "receipt": self._receipt(
                    request, state=state, ordinal=ordinal,
                ),
            }

    def start(self, request):
        with self._lock:
            if request.job_id in self.jobs:
                raise JobConflictError("duplicate fake job id")
            self.start_calls.append(request)
            self._counter += 1
            ordinal = self._counter
            stage = self._stage(request.job_id)
            final_state = (
                "failed"
                if stage == self.fail_once_stage
                and self._attempt(request.job_id) == 0
                else "succeeded"
            )
            running = self._receipt(
                request, state="running", ordinal=ordinal,
            )
            self.jobs[request.job_id] = {
                "request": request,
                "ordinal": ordinal,
                "polls": 0,
                "final_state": final_state,
                "receipt": running,
            }
            self._active_job_id = request.job_id
            return running

    def status(self, job_id: str):
        with self._lock:
            self.status_calls.append(job_id)
            if job_id not in self.jobs:
                raise JobNotFoundError("unknown fake job")
            job = self.jobs[job_id]
            receipt = job["receipt"]
            assert isinstance(receipt, JobReceipt)
            if receipt.state != "running":
                return receipt
            job["polls"] = int(job["polls"]) + 1
            if int(job["polls"]) <= 1:
                self._active_job_id = job_id
                return receipt
            request = job["request"]
            final = self._receipt(
                request,
                state=str(job["final_state"]),
                ordinal=int(job["ordinal"]),
            )
            job["receipt"] = final
            return final

    def sample_telemetry(self):
        with self._lock:
            assert self._active_job_id is not None
            stage = self._stage(self._active_job_id)
            job = self.jobs[self._active_job_id]
            sample_number = int(job.get("sample_number", 0)) + 1
            job["sample_number"] = sample_number
            if (
                stage == self.interrupt_telemetry_once_stage
                and sample_number == 2
                and not self._interrupted
            ):
                self._interrupted = True
                raise TelemetryUnavailableError(
                    "injected telemetry interruption"
                )
            self._telemetry_counter += 1
            captured_ns = (
                int(job["ordinal"]) * 10 + 1 + 2 * sample_number
            )
        gpu = self.expected["gpu"]
        utilization = (
            0.0
            if stage == self.omit_telemetry_stage
            else (75.0 if sample_number % 2 else 0.0)
        )
        return TelemetrySnapshot(
            captured_ns=captured_ns,
            host_mem_available_bytes=24_000_000_000 - captured_ns,
            gpus=(GpuSample(
                timestamp=f"2026-08-24T00:00:{sample_number:02d}Z",
                index=0,
                name=str(gpu["name"]),
                uuid=str(gpu["uuid"]),
                pci_bus_id="00000000:01:00.0",
                gpu_utilization_pct=utilization,
                memory_utilization_pct=25.0,
                memory_used_mib=4096.0,
                memory_total_mib=24576.0,
                temperature_c=55.0,
                power_w=200.0,
            ),),
        )


class _FakeArtifactInspector:
    def __init__(self) -> None:
        self.missing: set[str] = set()
        self.revisions: dict[str, int] = {}

    def inspect_artifact(self, absolute_path: str) -> TreeManifest:
        if absolute_path in self.missing:
            raise FileNotFoundError(absolute_path)
        revision = self.revisions.get(absolute_path, 0)
        digest = hashlib.sha256(
            f"{absolute_path}:{revision}".encode("utf-8")
        ).hexdigest()
        name = Path(absolute_path).name
        if name in {"act", "activation_cache", "validation-artifact"}:
            return TreeManifest(
                root_kind="directory",
                entries=(ManifestEntry("content.bin", "file", 1, digest),),
            )
        return TreeManifest(
            root_kind="file",
            entries=(ManifestEntry("payload", "file", 1, digest),),
        )


class _BlockingTransport(_FakeTransport):
    def __init__(
        self,
        expected: dict[str, object],
        blocked: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(expected)
        self.blocked = blocked
        self.release = release
        self._did_block = False

    def status(self, job_id: str):
        if (
            job_id in self.jobs
            and self._stage(job_id) == "host_preflight"
            and not self._did_block
        ):
            self._did_block = True
            self.blocked.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test did not release blocked transport")
        return super().status(job_id)


class _RestartAfterTelemetryGapTransport(_FakeTransport):
    def __init__(self, expected: dict[str, object], *, stage: str) -> None:
        super().__init__(expected)
        self.stage = stage
        self.gap_injected = False
        self._status_interrupt_pending = False
        self._hold_adoption_status = False

    def sample_telemetry(self):
        assert self._active_job_id is not None
        if (
            self._stage(self._active_job_id) == self.stage
            and not self.gap_injected
        ):
            self.gap_injected = True
            self._status_interrupt_pending = True
            raise TelemetryUnavailableError("injected durable telemetry gap")
        return super().sample_telemetry()

    def status(self, job_id: str):
        if self._status_interrupt_pending:
            self._status_interrupt_pending = False
            self._hold_adoption_status = True
            raise RuntimeError("injected coordinator status interruption")
        if self._hold_adoption_status:
            self._hold_adoption_status = False
            with self._lock:
                self.status_calls.append(job_id)
                receipt = self.jobs[job_id]["receipt"]
                assert isinstance(receipt, JobReceipt)
                return receipt
        return super().status(job_id)


class _ConflictAfterConfirmedAbsenceTransport(_FakeTransport):
    def __init__(self, expected: dict[str, object]) -> None:
        super().__init__(expected)
        self._confirmed_absence = False

    def status(self, job_id: str):
        if not self._confirmed_absence:
            self._confirmed_absence = True
            self.status_calls.append(job_id)
            raise JobNotFoundError("confirmed absent before injected race")
        return super().status(job_id)

    def start(self, request):
        super().start(request)
        raise JobConflictError("injected competing exact start")


def _transports(
    manifest: dict[str, object], **kwargs: object,
) -> dict[str, _FakeTransport]:
    return {
        str(host["id"]): _FakeTransport(host["expected"], **kwargs)
        for host in manifest["hosts"]
    }


def _inspectors(
    manifest: dict[str, object],
) -> dict[str, _FakeArtifactInspector]:
    return {
        str(host["id"]): _FakeArtifactInspector()
        for host in manifest["hosts"]
    }


def _coordinator(
    manifest: dict[str, object],
    state_dir: Path,
    transports: dict[str, _FakeTransport],
    inspectors: dict[str, _FakeArtifactInspector] | None = None,
    stage_preconditioner=None,
) -> CampaignCoordinator:
    return CampaignCoordinator(
        manifest,
        state_dir,
        transports=transports,
        artifact_inspectors=inspectors or _inspectors(manifest),
        stage_preconditioner=stage_preconditioner,
        sleep=lambda _: None,
    )


def _attempt_telemetry_state(
    samples: tuple[TelemetrySnapshot, ...] = (),
    *,
    sampling_failure_count: int = 0,
    sampling_integrity_failure_count: int = 0,
    consecutive_sampling_failure_count: int = 0,
    maximum_consecutive_sampling_failure_count: int | None = None,
    missing_prior_journal: bool = False,
    identity_sha256: str = "a" * 64,
) -> dict[str, object]:
    maximum_consecutive = (
        sampling_failure_count + sampling_integrity_failure_count
        if maximum_consecutive_sampling_failure_count is None
        else maximum_consecutive_sampling_failure_count
    )
    return {
        "schema": ATTEMPT_TELEMETRY_SCHEMA,
        "samples": [sample.to_payload() for sample in samples],
        "sampling_failure_count": sampling_failure_count,
        "sampling_integrity_failure_count": sampling_integrity_failure_count,
        "consecutive_sampling_failure_count": (
            consecutive_sampling_failure_count
        ),
        "maximum_consecutive_sampling_failure_count": maximum_consecutive,
        "missing_prior_journal": missing_prior_journal,
        "identity_sha256": identity_sha256,
    }


def _telemetry_sample(
    manifest: dict[str, object],
    host_id: str,
    captured_ns: int,
    *,
    utilization: float | None = 75.0,
) -> TelemetrySnapshot:
    host = next(host for host in manifest["hosts"] if host["id"] == host_id)
    gpu = host["expected"]["gpu"]
    return TelemetrySnapshot(
        captured_ns=captured_ns,
        host_mem_available_bytes=24_000_000_000,
        gpus=(GpuSample(
            timestamp="2026-08-24T00:00:00Z",
            index=0,
            name=str(gpu["name"]),
            uuid=str(gpu["uuid"]),
            pci_bus_id="00000000:01:00.0",
            gpu_utilization_pct=utilization,
            memory_utilization_pct=25.0,
            memory_used_mib=4096.0,
            memory_total_mib=24576.0,
            temperature_c=55.0,
            power_w=200.0,
        ),),
    )


def _write_legacy_attempt_journal(
    coordinator: CampaignCoordinator,
    command,
    request: RunRequest,
    *,
    attempt_index: int = 0,
    samples: tuple[TelemetrySnapshot, ...] = (),
) -> Path:
    body = {
        "schema": LEGACY_ATTEMPT_TELEMETRY_SCHEMA,
        "campaign_identity_sha256": coordinator.manifest["identity_sha256"],
        "work_id": command.work_id,
        "attempt_index": attempt_index,
        "request_sha256": request.request_sha256,
        "samples": [sample.to_payload() for sample in samples],
    }
    payload = {**body, "identity_sha256": canonical_sha256(body)}
    path = coordinator._attempt_telemetry_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _succeeded_job(
    request: RunRequest,
    *,
    started_ns: int,
    finished_ns: int,
    stdout: bytes = b"ok\n",
) -> JobReceipt:
    return JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="succeeded",
        started_ns=started_ns,
        finished_ns=finished_ns,
        returncode=0,
        pid=123,
        stdout=stdout,
        stderr=b"",
        transport="fake",
    )


def _direct_gpu_stage_receipt(
    manifest: dict[str, object],
    *,
    started_ns: int,
    finished_ns: int,
    samples: tuple[TelemetrySnapshot, ...],
    sampling_failure_count: int = 0,
) -> dict[str, object]:
    coordinator = _coordinator(
        manifest,
        Path("/tmp/prismaquant-direct-stage-receipt-fixture"),
        _transports(manifest),
    )
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    state = _attempt_telemetry_state(
        samples,
        sampling_failure_count=sampling_failure_count,
    )
    return _stage_receipt(
        manifest,
        command,
        request,
        _succeeded_job(
            request,
            started_ns=started_ns,
            finished_ns=finished_ns,
        ),
        state,
        (),
        (),
        attempt_index=0,
        artifact_inspector=_FakeArtifactInspector(),
    )


def _legacy_v2_manifest() -> dict[str, object]:
    body = json.loads(json.dumps(_manifest()))
    body.pop("identity_sha256")
    body["schema"] = LEGACY_CAMPAIGN_MANIFEST_SCHEMA
    body["artifact_target"]["physical_formats"] = [
        "FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48", "FP8_E4M3",
    ]
    telemetry = body["policy"]["telemetry"]
    telemetry.pop("maximum_observation_gap_milliseconds")
    telemetry.pop("minimum_successful_sample_percent")
    return {**body, "identity_sha256": canonical_sha256(body)}


def _replace_campaign_identity(
    value: object,
    *,
    old_identity: str,
    new_identity: str,
) -> object:
    if isinstance(value, str):
        return value.replace(old_identity, new_identity)
    if isinstance(value, list):
        return [
            _replace_campaign_identity(
                item,
                old_identity=old_identity,
                new_identity=new_identity,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_campaign_identity(
                item,
                old_identity=old_identity,
                new_identity=new_identity,
            )
            for key, item in value.items()
        }
    return value


def _legacy_v2_command_plan(
    current_manifest: dict[str, object],
    legacy_manifest: dict[str, object],
) -> dict[str, object]:
    current_plan = build_command_plan(current_manifest)
    old_identity = str(current_manifest["identity_sha256"])
    new_identity = str(legacy_manifest["identity_sha256"])
    rebound = _replace_campaign_identity(
        current_plan,
        old_identity=old_identity,
        new_identity=new_identity,
    )
    assert isinstance(rebound, dict)
    rebound.pop("identity_sha256")
    commands = []
    for raw_command in rebound["commands"]:
        command_body = dict(raw_command)
        command_body.pop("identity_sha256")
        commands.append({
            **command_body,
            "identity_sha256": canonical_sha256(command_body),
        })
    rebound["commands"] = commands
    rebound["campaign_identity_sha256"] = new_identity
    return {**rebound, "identity_sha256": canonical_sha256(rebound)}


def _write_legacy_v2_read_only_fixture(
    tmp_path: Path,
    *,
    precondition_receipt_sha256s: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    current_manifest = _manifest()
    manifest = _legacy_v2_manifest()
    plan = _legacy_v2_command_plan(current_manifest, manifest)
    command = next(
        row for row in plan["commands"]
        if row["work_id"] == "host_preflight:alpha"
    )
    request_env = {name: value for name, value in command["env"]}
    request_env.update({
        "PQ_CAMPAIGN_PRECONDITIONS_SHA256": canonical_sha256(
            list(precondition_receipt_sha256s),
        ),
        "PQ_CAMPAIGN_GPU_BEARING": "0",
        "PQ_CAMPAIGN_START_GUARD": "0",
    })
    request = RunRequest(
        job_id=(
            f"pq-{str(manifest['identity_sha256'])[:16]}-00-"
            "host_preflight-alpha-attempt-00"
        ),
        argv=tuple(command["argv"]),
        cwd="/",
        env=tuple(sorted(request_env.items())),
        timeout_seconds=600.0,
        stdin=b"",
        inherit_env=False,
    )
    host = next(host for host in manifest["hosts"] if host["id"] == "alpha")
    _, telemetry_summary = _normalize_telemetry(
        (),
        required=False,
        expected_gpu=host["expected"]["gpu"],
    )
    job = _succeeded_job(
        request,
        started_ns=1,
        finished_ns=2,
    ).to_payload()
    receipt_body = {
        "schema": LEGACY_EXECUTION_RECEIPT_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_identity_sha256": canonical_sha256(host),
        "producer_identity_sha256": canonical_sha256(manifest["producer"]),
        "work_id": command["work_id"],
        "stage": command["stage"],
        "host_id": command["host_id"],
        "assignment_index": command["assignment_index"],
        "attempt_index": 0,
        "command_identity_sha256": command["identity_sha256"],
        "request_sha256": request.request_sha256,
        "dependency_receipt_sha256s": [],
        "precondition_receipt_sha256s": list(
            precondition_receipt_sha256s,
        ),
        "job": job,
        "gpu_bearing": False,
        "gpu_activity_requirement": "not_applicable",
        "telemetry": [],
        "telemetry_sha256": canonical_sha256([]),
        "telemetry_summary": telemetry_summary,
        "output_artifacts": [],
    }
    receipt = {
        **receipt_body,
        "identity_sha256": canonical_sha256(receipt_body),
    }
    completion = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "work_id": command["work_id"],
        "stage": command["stage"],
        "host_id": command["host_id"],
        "assignment_index": command["assignment_index"],
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_identity_sha256": canonical_sha256(host),
        "producer_identity_sha256": canonical_sha256(manifest["producer"]),
        "receipt_sha256": receipt["identity_sha256"],
    }
    state_body = {
        "schema": CAMPAIGN_STATE_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "completions": [completion],
    }
    state = {**state_body, "identity_sha256": canonical_sha256(state_body)}
    state_dir = tmp_path / "legacy-state"
    receipt_dir = state_dir / "receipts"
    receipt_dir.mkdir(parents=True)

    def write(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest_path = state_dir / "campaign-manifest.json"
    write(manifest_path, manifest)
    write(state_dir / "command-plan.json", plan)
    write(state_dir / "campaign-state.json", state)
    write(receipt_dir / "host_preflight:alpha.json", receipt)
    return manifest_path, state_dir


def _filesystem_snapshot(root: Path) -> dict[Path, bytes | None]:
    return {
        path.relative_to(root): (None if path.is_dir() else path.read_bytes())
        for path in root.rglob("*")
    }


def test_plan_is_closed_and_maps_hosts_to_partitions_and_stripes_0_and_1() -> None:
    manifest = _manifest()
    plan = build_command_plan(manifest)
    commands = plan["commands"]
    assert isinstance(commands, list)
    assert len(commands) == 23

    ce = sorted(
        (row for row in commands if row["stage"] == "sample_ce"),
        key=lambda row: row["host_id"],
    )
    burn = sorted(
        (row for row in commands if row["stage"] == "measure_burn"),
        key=lambda row: row["host_id"],
    )
    assert [row["worker_index"] for row in ce] == [0, 1]
    assert [int(_argument(row["module_argv"], "--sample-partition-index"))
            for row in ce] == [0, 1]
    assert [int(_argument(row["module_argv"], "--stripe"))
            for row in burn] == [0, 1]
    assert all(row["assignment_index"] > 1 for row in ce + burn)

    coordinator_rows = [
        row for row in commands
        if row["stage"] in {
            "prepare_calibration",
            "prepare_burn",
            "allocate",
            "export_validation_artifact",
            "qualify_validation_artifact",
        }
    ]
    assert {row["host_id"] for row in coordinator_rows} == {"zeta"}
    prepare = next(row for row in commands if row["stage"] == "prepare_burn")
    assert "eugr/spark-vllm@sha256:" + "4" * 64 in prepare["argv"]
    assert "type=bind,src=/srv/zeta/model,dst=/model,readonly" in prepare["argv"]
    assert "type=bind,src=/srv/zeta/campaign-run,dst=/run" in prepare["argv"]
    assert "PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1" in prepare["argv"]
    fisher = next(
        row for row in commands
        if row["stage"] == "sample_fisher" and row["host_id"] == "alpha"
    )
    assert set(fisher["inputs"]) >= {
        "/model", "/dataset", "/pq", "/run/calibration.pt",
        "/run/run-contract.json", "/run/cover.json", "/run/global-ce.json",
    }
    assert all("shell" not in row for row in commands)
    export = next(
        row for row in commands
        if row["stage"] == "export_validation_artifact"
    )
    assert export["gpu_bearing"] is True
    assert export["outputs"] == [
        "/run/burn/gridbook-runtime-contract.json",
        "/run/validation-artifact",
    ]
    encoded = _argument(
        export["module_argv"], "--runtime-contract-payload-base64",
    )
    assert json.loads(base64.b64decode(encoded)) == {
        "schema": "gridbook.runtime-contract.v11",
        "test_fixture": True,
    }
    qualify = next(
        row for row in commands
        if row["stage"] == "qualify_validation_artifact"
    )
    assert qualify["gpu_bearing"] is False
    assert qualify["outputs"] == [
        "/run/burn/validation-package-receipt.json",
    ]


def test_only_gpu_stages_receive_devices_and_all_containers_use_launch_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    gpu = coordinator.commands["sample_ce:alpha"]
    cpu = coordinator.commands["allocate:zeta"]
    for command in (gpu, cpu):
        assert command.argv[:4] == ("python3", "-P", "-B", "-s")
        assert "prismaquant.cluster_host_admission" in command.argv
        assert "guarded-launch" in command.argv
        assert f"io.prismaquant.campaign={manifest['identity_sha256']}" in (
            command.argv
        )
        request = coordinator._request(command, attempt_index=0)
        assert json.loads(request.stdin) == manifest
        assert request.timeout_seconds is not None
        timeout_index = command.argv.index("--timeout-seconds")
        inner_timeout = float(command.argv[timeout_index + 1])
        assert request.timeout_seconds - inner_timeout == 600.0
        work_index = command.argv.index("--work-id")
        assert command.argv[work_index + 1] == command.work_id
        assert dict(request.env)["PQ_CAMPAIGN_START_GUARD"] == "1"
    assert "--gpus" in gpu.argv
    assert "--gpus" not in cpu.argv
    assert "NVIDIA_VISIBLE_DEVICES=void" not in gpu.argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in cpu.argv


def test_stage_preconditions_are_bound_and_revalidated_during_verify(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    version = {"sample_ce": "v1"}

    def precondition(stage: str, *, verify_only: bool):
        del verify_only
        marker = version.get(stage, "stable")
        return (hashlib.sha256(f"{stage}:{marker}".encode()).hexdigest(),)

    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        _transports(manifest),
        stage_preconditioner=precondition,
    )
    coordinator.run_to_completion(resume=False)
    receipt = json.loads(
        (tmp_path / "state/receipts/sample_ce:alpha.json").read_text()
    )
    assert receipt["precondition_receipt_sha256s"] == [
        hashlib.sha256(b"sample_ce:v1").hexdigest()
    ]

    version["sample_ce"] = "v2"
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="precondition_receipt_sha256s differs",
    ):
        coordinator.verify()


def test_existing_transport_job_requires_a_valid_durable_launch_guard(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = CampaignCoordinator(manifest, tmp_path / "state")
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    receipt = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="succeeded",
        started_ns=1,
        finished_ns=2,
        returncode=0,
        pid=123,
        stdout=b"ok\n",
        stderr=b"",
        transport="existing",
    )

    class ExistingWithoutGuard:
        def status(self, job_id):
            assert job_id == request.job_id
            return receipt

        def start(self, _request):  # pragma: no cover - adoption only
            raise AssertionError("existing job must not be restarted")

        def validate_existing_guard(self, _request):
            raise RuntimeError("guard receipt absent")

    with pytest.raises(RTX4090TwoHostCampaignError, match="no valid launch guard"):
        coordinator._adopt_or_start(ExistingWithoutGuard(), request)


def test_adopted_gpu_job_without_prior_journal_is_fail_latched(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    transport = transports["alpha"]
    transport.start(request)

    job, telemetry_state = coordinator._monitor_attempt(
        command,
        transport,
        request,
        0,
    )

    assert job["state"] == "succeeded"
    assert telemetry_state is not None
    assert telemetry_state["missing_prior_journal"] is True
    assert telemetry_state["sampling_integrity_failure_count"] == 1
    assert len(transport.start_calls) == 1
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="telemetry integrity evidence invalidates",
    ):
        _stage_receipt(
            manifest,
            command,
            request,
            job,
            telemetry_state,
            (),
            (),
            attempt_index=0,
            artifact_inspector=_FakeArtifactInspector(),
        )


def test_unavailable_initial_status_never_authorizes_start_or_clean_journal(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    transport = transports["alpha"]
    transport.preload(request, state="running")
    original_status = transport.status
    first_status = True

    def unavailable_once(job_id: str):
        nonlocal first_status
        if first_status:
            first_status = False
            raise TelemetryUnavailableError("status backend unavailable")
        return original_status(job_id)

    transport.status = unavailable_once

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="cannot determine whether transport job.*refusing start",
    ):
        coordinator._monitor_attempt(command, transport, request, 0)

    assert transport.start_calls == []
    assert not coordinator._attempt_telemetry_path(request).exists()


@pytest.mark.parametrize(
    "journal_kind",
    [
        "valid-v1",
        "malformed-json",
        "malformed-current",
        "current-with-sample",
        "current-with-failure",
    ],
)
def test_confirmed_absence_with_unusable_prior_journal_hard_blocks_start(
    tmp_path: Path,
    journal_kind: str,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / journal_kind,
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    path = coordinator._attempt_telemetry_path(request)

    if journal_kind == "valid-v1":
        _write_legacy_attempt_journal(coordinator, command, request)
    elif journal_kind == "malformed-json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{malformed journal\n")
    else:
        coordinator._initialize_attempt_telemetry(command, request, 0)
        if journal_kind == "malformed-current":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["samples"] = {"not": "a list"}
            body = dict(payload)
            body.pop("identity_sha256")
            payload["identity_sha256"] = canonical_sha256(body)
            path.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        elif journal_kind == "current-with-sample":
            coordinator._begin_attempt_telemetry_sample(
                command,
                request,
                0,
            )
            coordinator._append_attempt_telemetry(
                command,
                request,
                0,
                _telemetry_sample(manifest, "alpha", 3),
            )
        else:
            assert journal_kind == "current-with-failure"
            coordinator._record_attempt_telemetry_failure(
                command,
                request,
                0,
                integrity_failure=False,
            )
    original = path.read_bytes()
    transport = transports["alpha"]

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match=(
            "unusable prior telemetry but no durable transport state; "
            "refusing any retry"
        ),
    ):
        coordinator._monitor_attempt(command, transport, request, 0)

    assert transport.status_calls == [request.job_id]
    assert transport.start_calls == []
    assert path.read_bytes() == original


def test_confirmed_absence_with_pristine_current_journal_remains_startable(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    transport = transports["alpha"]

    job, telemetry_state = coordinator._monitor_attempt(
        command,
        transport,
        request,
        0,
    )

    assert job["state"] == "succeeded"
    assert len(transport.start_calls) == 1
    assert transport.start_calls[0] == request
    assert transport.status_calls[0] == request.job_id
    assert telemetry_state is not None
    assert telemetry_state["schema"] == ATTEMPT_TELEMETRY_SCHEMA
    assert telemetry_state["missing_prior_journal"] is False
    assert telemetry_state["sampling_integrity_failure_count"] == 0
    assert len(telemetry_state["samples"]) == 2


def test_confirmed_absence_then_start_conflict_is_adopted_with_fail_latch(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    alpha = next(host for host in manifest["hosts"] if host["id"] == "alpha")
    transport = _ConflictAfterConfirmedAbsenceTransport(alpha["expected"])
    transports = _transports(manifest)
    transports["alpha"] = transport
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)

    job, telemetry_state = coordinator._monitor_attempt(
        command,
        transport,
        request,
        0,
    )

    assert job["state"] == "succeeded"
    assert len(transport.start_calls) == 1
    assert telemetry_state is not None
    assert telemetry_state["sampling_integrity_failure_count"] == 1
    assert telemetry_state["missing_prior_journal"] is True
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="telemetry integrity evidence invalidates",
    ):
        _stage_receipt(
            manifest,
            command,
            request,
            job,
            telemetry_state,
            (),
            (),
            attempt_index=0,
            artifact_inspector=_FakeArtifactInspector(),
        )


def test_pristine_journal_then_start_conflict_is_unconditionally_fail_latched(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    alpha = next(host for host in manifest["hosts"] if host["id"] == "alpha")
    transport = _ConflictAfterConfirmedAbsenceTransport(alpha["expected"])
    transports = _transports(manifest)
    transports["alpha"] = transport
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    path = coordinator._attempt_telemetry_path(request)
    pristine = path.read_bytes()

    job, telemetry_state = coordinator._monitor_attempt(
        command,
        transport,
        request,
        0,
    )

    assert job["state"] == "succeeded"
    assert len(transport.start_calls) == 1
    assert len(transport.status_calls) >= 3
    assert telemetry_state is not None
    assert telemetry_state["sampling_integrity_failure_count"] == 1
    assert telemetry_state["missing_prior_journal"] is True
    assert telemetry_state[
        "maximum_consecutive_sampling_failure_count"
    ] == 1
    assert path.read_bytes() != pristine


def test_execute_aborts_all_retries_for_absent_job_with_unusable_journal(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    assignment = StageAssignment(
        work_id=command.work_id,
        stage=command.stage,
        host_id=command.host_id,
        assignment_index=command.assignment_index,
    )
    poisoned_request = coordinator._request(command, attempt_index=0)
    path = _write_legacy_attempt_journal(
        coordinator,
        command,
        poisoned_request,
    )
    original = path.read_bytes()
    next_request = coordinator._request(command, attempt_index=1)
    transport = transports["alpha"]

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match=(
            "unusable prior telemetry but no durable transport state; "
            "refusing any retry"
        ),
    ):
        coordinator._execute(
            assignment,
            dependency_receipt_sha256s=(),
            precondition_receipt_sha256s=(),
        )

    assert transport.status_calls == [poisoned_request.job_id]
    assert transport.start_calls == []
    assert path.read_bytes() == original
    assert not coordinator._attempt_telemetry_path(next_request).exists()


def test_execute_retries_only_after_adopted_malformed_journal_job_is_terminal(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    assignment = StageAssignment(
        work_id=command.work_id,
        stage=command.stage,
        host_id=command.host_id,
        assignment_index=command.assignment_index,
    )
    adopted_request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(
        command,
        adopted_request,
        0,
    )
    path = coordinator._attempt_telemetry_path(adopted_request)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"] = {"not": "a list"}
    body = dict(payload)
    body.pop("identity_sha256")
    payload["identity_sha256"] = canonical_sha256(body)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    original = path.read_bytes()
    transport = transports["alpha"]
    transport.start(adopted_request)

    returned_assignment, receipt = coordinator._execute(
        assignment,
        dependency_receipt_sha256s=(),
        precondition_receipt_sha256s=(),
    )

    assert returned_assignment == assignment
    assert receipt["attempt_index"] == 1
    assert [request.job_id for request in transport.start_calls] == [
        adopted_request.job_id,
        coordinator._request(command, attempt_index=1).job_id,
    ]
    adopted_job = transport.jobs[adopted_request.job_id]["receipt"]
    assert isinstance(adopted_job, JobReceipt)
    assert adopted_job.state == "succeeded"
    assert transport.status_calls[:2] == [
        adopted_request.job_id,
        adopted_request.job_id,
    ]
    assert path.read_bytes() == original


def test_v1_attempt_telemetry_journal_loads_as_a_durable_fail_latch(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    sample = _telemetry_sample(manifest, "alpha", 3).to_payload()
    body = {
        "schema": LEGACY_ATTEMPT_TELEMETRY_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "work_id": command.work_id,
        "attempt_index": 0,
        "request_sha256": request.request_sha256,
        "samples": [sample],
    }
    payload = {**body, "identity_sha256": canonical_sha256(body)}
    path = coordinator._attempt_telemetry_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = coordinator._load_attempt_telemetry(
        command,
        request,
        0,
    )

    assert state == {
        "schema": LEGACY_ATTEMPT_TELEMETRY_SCHEMA,
        "samples": [sample],
        "sampling_failure_count": 0,
        "sampling_integrity_failure_count": 1,
        "consecutive_sampling_failure_count": 1,
        "maximum_consecutive_sampling_failure_count": 1,
        "missing_prior_journal": True,
        "identity_sha256": payload["identity_sha256"],
    }


def test_v2_attempt_telemetry_journal_remains_read_only_compatible(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    sample = _telemetry_sample(manifest, "alpha", 3).to_payload()
    body = {
        "schema": LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA,
        "campaign_identity_sha256": manifest["identity_sha256"],
        "work_id": command.work_id,
        "attempt_index": 0,
        "request_sha256": request.request_sha256,
        "samples": [sample],
        "sampling_failure_count": 1,
        "sampling_integrity_failure_count": 0,
        "consecutive_sampling_failure_count": 1,
        "maximum_consecutive_sampling_failure_count": 1,
        "missing_prior_journal": False,
    }
    payload = {**body, "identity_sha256": canonical_sha256(body)}
    path = coordinator._attempt_telemetry_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert state["schema"] == LEGACY_ATTEMPT_TELEMETRY_V2_SCHEMA
    assert state["samples"] == [sample]
    assert state["sampling_failure_count"] == 1
    original = path.read_bytes()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="legacy attempt telemetry has no mutable head",
    ):
        coordinator._begin_attempt_telemetry_sample(command, request, 0)
    assert path.read_bytes() == original
    samples, summary, identity = _normalize_attempt_telemetry_evidence(
        manifest,
        command,
        _succeeded_job(
            request, started_ns=1, finished_ns=4,
        ).to_payload(),
        state,
        activity_requirement="positive_utilization_required",
        allow_legacy_v2=True,
    )
    assert samples == [sample]
    assert summary["sampling_failure_count"] == 1
    assert identity == payload["identity_sha256"]
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="requires a current attempt telemetry journal",
    ):
        _normalize_attempt_telemetry_evidence(
            manifest,
            command,
            _succeeded_job(
                request, started_ns=1, finished_ns=4,
            ).to_payload(),
            state,
            activity_requirement="positive_utilization_required",
        )


def test_v3_attempt_journal_keeps_bounded_head_and_immutable_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            coordinator,
            "_load_attempt_telemetry",
            lambda *_args, **_kwargs: pytest.fail(
                "hot append reread the complete observation history"
            ),
        )
        for ordinal in range(128):
            coordinator._begin_attempt_telemetry_sample(command, request, 0)
            coordinator._append_attempt_telemetry(
                command,
                request,
                0,
                _telemetry_sample(manifest, "alpha", ordinal + 1),
            )

    head_path = coordinator._attempt_telemetry_path(request)
    head = json.loads(head_path.read_text(encoding="utf-8"))
    records = sorted(
        coordinator._attempt_telemetry_records_dir(request).glob("*.json")
    )
    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert "samples" not in head
    assert head["record_count"] == 128
    assert head["next_ordinal"] == 128
    assert head_path.stat().st_size < 2_000
    assert len(records) == 128
    assert state["samples"][0]["captured_ns"] == 1
    assert state["samples"][-1]["captured_ns"] == 128
    first = json.loads(records[0].read_text(encoding="utf-8"))
    assert first["schema"] == ATTEMPT_TELEMETRY_RECORD_SCHEMA
    assert first["previous_record_sha256"] is None
    assert json.loads(records[1].read_text(encoding="utf-8"))[
        "previous_record_sha256"
    ] == first["identity_sha256"]


def test_pending_sample_is_durably_integrity_latched_on_resume(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)

    coordinator._begin_attempt_telemetry_sample(command, request, 0)
    pending = json.loads(
        coordinator._attempt_telemetry_path(request).read_text(encoding="utf-8")
    )
    assert pending["pending_sample_ordinal"] == 0

    recovered = coordinator._recover_attempt_telemetry(command, request, 0)
    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert recovered["pending_sample_ordinal"] is None
    assert state["sampling_integrity_failure_count"] == 1
    assert state["maximum_consecutive_sampling_failure_count"] == 1
    record = json.loads(
        coordinator._attempt_telemetry_record_path(request, 0).read_text(
            encoding="utf-8"
        )
    )
    assert record["outcome"] == "integrity"


def test_record_published_before_head_is_adopted_without_laundering_sample(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    head = coordinator._begin_attempt_telemetry_sample(command, request, 0)
    sample = _telemetry_sample(manifest, "alpha", 7).to_payload()
    record = coordinator._attempt_telemetry_record_payload(
        command,
        request,
        0,
        ordinal=0,
        previous_record_sha256=None,
        outcome="sample",
        sample=sample,
    )
    record_path = coordinator._attempt_telemetry_record_path(request, 0)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    assert head["pending_sample_ordinal"] == 0

    recovered = coordinator._recover_attempt_telemetry(command, request, 0)
    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert recovered["head_record_sha256"] == record["identity_sha256"]
    assert state["sampling_integrity_failure_count"] == 0
    assert state["samples"] == [sample]


def test_interrupted_start_phase_is_fail_latched_before_adoption(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    coordinator._mark_attempt_start_invoked(command, request, 0)

    recovered = coordinator._recover_attempt_telemetry(command, request, 0)
    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert recovered["start_phase"] == "claimed"
    assert recovered["start_disposition"] == "recovered_ambiguous_start"
    assert state["sampling_integrity_failure_count"] == 1
    assert state["missing_prior_journal"] is True


class _InjectedStartJournalCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "boundary",
    (
        "before_intent",
        "after_intent",
        "after_record_publish",
        "after_head_commit",
        "after_claim",
    ),
)
def test_missing_prefix_start_claim_recovers_every_wal_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / boundary, _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    coordinator._mark_attempt_start_invoked(command, request, 0)
    record_path = coordinator._attempt_telemetry_record_path(request, 0)
    original_replace = coordinator._replace_attempt_telemetry_head
    original_write = campaign_module._write_exact_no_clobber
    original_commit = coordinator._commit_attempt_telemetry_record

    def replace_then_maybe_crash(*args, **changes):
        phase = changes.get("start_phase")
        if boundary == "before_intent" and phase == "claim_pending":
            raise _InjectedStartJournalCrash(boundary)
        result = original_replace(*args, **changes)
        if boundary == "after_intent" and phase == "claim_pending":
            raise _InjectedStartJournalCrash(boundary)
        if boundary == "after_claim" and phase == "claimed":
            raise _InjectedStartJournalCrash(boundary)
        return result

    def write_then_maybe_crash(path, value):
        original_write(path, value)
        if boundary == "after_record_publish" and path == record_path:
            raise _InjectedStartJournalCrash(boundary)

    def commit_then_maybe_crash(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        if boundary == "after_head_commit":
            raise _InjectedStartJournalCrash(boundary)
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(
            coordinator,
            "_replace_attempt_telemetry_head",
            replace_then_maybe_crash,
        )
        scoped.setattr(
            campaign_module,
            "_write_exact_no_clobber",
            write_then_maybe_crash,
        )
        scoped.setattr(
            coordinator,
            "_commit_attempt_telemetry_record",
            commit_then_maybe_crash,
        )
        with pytest.raises(_InjectedStartJournalCrash, match=boundary):
            coordinator._mark_attempt_start_claimed(
                command,
                request,
                0,
                "existing_after_conflict",
            )

    intermediate = json.loads(
        coordinator._attempt_telemetry_path(request).read_text(
            encoding="utf-8",
        )
    )
    if intermediate["start_phase"] == "claimed":
        assert intermediate["sampling_integrity_failure_count"] >= 1
        assert intermediate["missing_prior_journal"] is True
    else:
        assert intermediate["start_phase"] in {
            "start_invoked", "claim_pending",
        }

    recovered = coordinator._recover_attempt_telemetry(
        command, request, 0,
    )
    state = coordinator._load_attempt_telemetry(command, request, 0)

    assert recovered["start_phase"] == "claimed"
    assert recovered["start_disposition"] == (
        "recovered_ambiguous_start"
        if boundary == "before_intent" else "existing_after_conflict"
    )
    assert state["sampling_integrity_failure_count"] == 1
    assert state["missing_prior_journal"] is True
    assert len(list(
        coordinator._attempt_telemetry_records_dir(request).glob("*.json")
    )) == 1


@pytest.mark.parametrize(
    "disposition",
    ("existing", "existing_after_conflict", "recovered_ambiguous_start"),
)
def test_claimed_missing_prefix_dispositions_require_durable_integrity(
    tmp_path: Path,
    disposition: str,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / disposition, _transports(manifest),
    )
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="attempt telemetry head fields are invalid",
    ):
        coordinator._attempt_telemetry_head_payload(
            command,
            request,
            0,
            start_phase="claimed",
            start_disposition=disposition,
            next_ordinal=0,
            record_count=0,
            head_record_sha256=None,
            pending_sample_ordinal=None,
            last_sample_captured_ns=None,
            sampling_failure_count=0,
            sampling_integrity_failure_count=0,
            consecutive_sampling_failure_count=0,
            maximum_consecutive_sampling_failure_count=0,
            missing_prior_journal=False,
        )


def test_attempt_record_hash_chain_refuses_resealed_prefix_tamper(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    for captured_ns in (3, 4):
        coordinator._begin_attempt_telemetry_sample(command, request, 0)
        coordinator._append_attempt_telemetry(
            command,
            request,
            0,
            _telemetry_sample(manifest, "alpha", captured_ns),
        )
    first_path = coordinator._attempt_telemetry_record_path(request, 0)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    first["sample"]["captured_ns"] = 2
    first_body = dict(first)
    first_body.pop("identity_sha256")
    first["identity_sha256"] = canonical_sha256(first_body)
    first_path.write_text(
        json.dumps(first, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="record 1 digest or binding differs",
    ):
        coordinator._load_attempt_telemetry(command, request, 0)


def test_v2_execution_receipt_refuses_a_v1_attempt_journal() -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        Path("/tmp/prismaquant-v1-journal-v2-receipt-fixture"),
        _transports(manifest),
    )
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    legacy_state = _attempt_telemetry_state((
        _telemetry_sample(manifest, "alpha", 5),
    ))
    legacy_state["schema"] = LEGACY_ATTEMPT_TELEMETRY_SCHEMA

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="requires a current attempt telemetry journal",
    ):
        _stage_receipt(
            manifest,
            command,
            request,
            _succeeded_job(request, started_ns=1, finished_ns=10),
            legacy_state,
            (),
            (),
            attempt_index=0,
            artifact_inspector=_FakeArtifactInspector(),
        )


def test_adopted_live_gpu_job_with_v1_journal_is_fail_latched_to_terminal(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
    )
    coordinator.initialize()
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    _write_legacy_attempt_journal(coordinator, command, request)
    transport = transports["alpha"]
    transport.start(request)

    job, telemetry_state = coordinator._monitor_attempt(
        command,
        transport,
        request,
        0,
    )

    assert job["state"] == "succeeded"
    assert len(transport.start_calls) == 1
    assert telemetry_state is not None
    assert telemetry_state["schema"] == LEGACY_ATTEMPT_TELEMETRY_SCHEMA
    assert telemetry_state["sampling_integrity_failure_count"] == 1
    assert telemetry_state["missing_prior_journal"] is True
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="requires a current attempt telemetry journal",
    ):
        _stage_receipt(
            manifest,
            command,
            request,
            job,
            telemetry_state,
            (),
            (),
            attempt_index=0,
            artifact_inspector=_FakeArtifactInspector(),
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("sampling_failure_count", "1"),
        ("sampling_failure_count", True),
        ("sampling_integrity_failure_count", "1"),
        ("sampling_integrity_failure_count", False),
        ("consecutive_sampling_failure_count", "0"),
        ("consecutive_sampling_failure_count", True),
        ("maximum_consecutive_sampling_failure_count", "0"),
        ("maximum_consecutive_sampling_failure_count", False),
    ],
)
def test_v2_journal_loader_rejects_string_and_boolean_counters(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        tmp_path / f"state-{field}-{invalid_value!s}",
        _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    path = coordinator._attempt_telemetry_path(request)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = invalid_value
    body = dict(payload)
    body.pop("identity_sha256")
    payload["identity_sha256"] = canonical_sha256(body)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="telemetry sampling counters are invalid",
    ):
        coordinator._load_attempt_telemetry(command, request, 0)


def test_v2_journal_rejects_failures_with_zero_maximum_consecutive_count(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        _transports(manifest),
    )
    coordinator.initialize()
    command = coordinator.commands["sample_ce:alpha"]
    request = coordinator._request(command, attempt_index=0)
    coordinator._initialize_attempt_telemetry(command, request, 0)
    path = coordinator._attempt_telemetry_path(request)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sampling_failure_count"] = 1
    assert payload["maximum_consecutive_sampling_failure_count"] == 0
    body = dict(payload)
    body.pop("identity_sha256")
    payload["identity_sha256"] = canonical_sha256(body)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="counter relationships are invalid",
    ):
        coordinator._load_attempt_telemetry(command, request, 0)


def test_attempt_job_ids_remain_valid_for_maximum_length_host_ids(
    tmp_path: Path,
) -> None:
    body = json.loads(json.dumps(_manifest()))
    body.pop("identity_sha256")
    replacements = {"alpha": "a" * 128, "zeta": "z" * 128}
    for host in body["hosts"]:
        host["id"] = replacements[host["id"]]
    body["coordinator"] = replacements[body["coordinator"]]
    manifest = seal_campaign_manifest(body)
    coordinator = CampaignCoordinator(manifest, tmp_path / "state")

    requests = [
        coordinator._request(command, attempt_index=attempt)
        for command in coordinator.commands.values()
        for attempt in range(2)
    ]

    assert all(len(request.job_id) <= 128 for request in requests)
    assert all("-host-" in request.job_id for request in requests)
    assert len({request.job_id for request in requests}) == len(requests)


def test_fake_transport_completes_all_barriers_and_records_kpi_receipts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    inspectors = _inspectors(manifest)
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    assert result["completed_assignments"] == 23
    assert sum(len(transport.calls) for transport in transports.values()) == 23
    receipt_path = tmp_path / "state" / "receipts" / "sample_fisher:alpha.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = receipt["telemetry_summary"]
    assert summary["live_cuda_observed"] is True
    assert summary["gpu_utilization"] == {
        "count": 2,
        "mean": 37.5,
        "p50": 37.5,
        "p95": 71.25,
        "active_fraction": 0.5,
        "active_fraction_gt_0": 0.5,
        "active_fraction_gt_50": 0.5,
        "active_fraction_gt_90": 0.0,
    }
    assert len(receipt["dependency_receipt_sha256s"]) == 1
    assert receipt["attempt_index"] == 0
    assert [row["container_path"] for row in receipt["output_artifacts"]] == [
        "/run/worker-0/probe.pkl",
        "/run/worker-0/act",
    ]
    assert receipt["output_artifacts"][1]["expected_kind"] == "directory"
    assert coordinator.verify() == result


def test_failed_attempt_uses_bounded_new_id_and_only_retry_resume_flags(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    transports["alpha"].fail_once_stage = "measure_burn"
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    requests = [
        request for transport in transports.values()
        for request in transport.start_calls
    ]
    measure = [
        request for request in requests if "-measure_burn-alpha-" in request.job_id
    ]
    assert [request.job_id.rsplit("-", 1)[1] for request in measure] == [
        "00", "01",
    ]
    assert "--resume" not in measure[0].argv
    assert "--resume" in measure[1].argv
    allocate = next(
        request for request in requests if "-allocate-" in request.job_id
    )
    assert "--resume" not in allocate.argv
    assert all(request.inherit_env is False for request in requests)
    receipt = json.loads(
        (tmp_path / "state" / "receipts" / "measure_burn:alpha.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["attempt_index"] == 1


def test_validation_export_retry_uses_exact_resume_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    transports["zeta"].fail_once_stage = "export_validation_artifact"
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    exports = [
        request for request in transports["zeta"].start_calls
        if "-export_validation_artifact-" in request.job_id
    ]
    assert len(exports) == 2
    assert "--resume" not in exports[0].argv
    assert "--resume" in exports[1].argv
    assert exports[0].argv == exports[1].argv[:-1]


def test_transient_telemetry_failure_does_not_abandon_running_job(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    transports["zeta"].interrupt_telemetry_once_stage = (
        "coordinator_source_identity"
    )
    inspectors = _inspectors(manifest)
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    matching_starts = [
        request for request in transports["zeta"].start_calls
        if "-coordinator_source_identity-" in request.job_id
    ]
    assert len(matching_starts) == 1
    telemetry_paths = sorted(
        (tmp_path / "state" / "attempt-telemetry").glob(
            "*coordinator_source_identity*.json"
        )
    )
    assert len(telemetry_paths) == 1
    command = coordinator.commands["coordinator_source_identity:zeta"]
    journal = coordinator._load_attempt_telemetry(
        command, matching_starts[0], 0,
    )
    assert len(journal["samples"]) == 1
    assert journal["sampling_failure_count"] == 1
    assert journal["samples"][0]["gpus"][0]["gpu_utilization_pct"] == 75.0

    receipt = json.loads(
        (tmp_path / "state" / "receipts" /
         "coordinator_source_identity:zeta.json").read_text(encoding="utf-8")
    )
    assert len(receipt["telemetry"]) == 1
    assert receipt["telemetry_summary"]["live_cuda_observed"] is True
    assert receipt["telemetry_summary"]["sampling_failure_count"] == 1
    allocate = next(
        request for request in transports["zeta"].start_calls
        if "-allocate-" in request.job_id
    )
    assert allocate.job_id.endswith("-attempt-00")
    assert "--resume" not in allocate.argv


def test_restart_adopts_one_job_and_preserves_transient_failure_count(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    zeta = next(host for host in manifest["hosts"] if host["id"] == "zeta")
    transports["zeta"] = _RestartAfterTelemetryGapTransport(
        zeta["expected"],
        stage="coordinator_source_identity",
    )
    inspectors = _inspectors(manifest)
    first = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
        inspectors,
    )

    with pytest.raises(RTX4090TwoHostCampaignError, match="status failed"):
        first.run_to_completion(resume=False)

    resumed = _coordinator(
        manifest,
        tmp_path / "state",
        transports,
        inspectors,
    )
    result = resumed.run_to_completion(resume=True)

    assert result["complete"] is True
    starts = [
        request for request in transports["zeta"].start_calls
        if "-coordinator_source_identity-" in request.job_id
    ]
    assert len(starts) == 1
    journal = resumed._load_attempt_telemetry(
        resumed.commands["coordinator_source_identity:zeta"], starts[0], 0,
    )
    assert journal["sampling_failure_count"] == 1
    assert len(journal["samples"]) == 2
    receipt = json.loads(
        (tmp_path / "state" / "receipts" /
         "coordinator_source_identity:zeta.json").read_text(encoding="utf-8")
    )
    assert receipt["attempt_index"] == 0
    assert receipt["telemetry_summary"]["sampling_failure_count"] == 1


def test_telemetry_success_fraction_accepts_exactly_fifty_percent() -> None:
    manifest = _manifest()
    receipt = _direct_gpu_stage_receipt(
        manifest,
        started_ns=1,
        finished_ns=20_000_000_001,
        samples=(
            _telemetry_sample(manifest, "alpha", 10_000_000_001),
        ),
        sampling_failure_count=1,
    )

    assert receipt["telemetry_summary"]["sampling_attempt_count"] == 2
    assert receipt["telemetry_summary"]["successful_sample_percent"] == 50.0


def test_telemetry_success_fraction_below_fifty_percent_is_refused() -> None:
    manifest = _manifest()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="successful sample percentage is below policy",
    ):
        _direct_gpu_stage_receipt(
            manifest,
            started_ns=1,
            finished_ns=20_000_000_001,
            samples=(
                _telemetry_sample(manifest, "alpha", 10_000_000_001),
            ),
            sampling_failure_count=2,
        )


def test_telemetry_gap_accepts_exactly_thirty_seconds_at_job_boundaries() -> None:
    manifest = _manifest()
    receipt = _direct_gpu_stage_receipt(
        manifest,
        started_ns=1,
        finished_ns=60_000_000_001,
        samples=(
            _telemetry_sample(manifest, "alpha", 30_000_000_001),
        ),
    )

    assert (
        receipt["telemetry_summary"]["maximum_observation_gap_ns"]
        == 30_000_000_000
    )


def test_telemetry_gap_over_thirty_seconds_at_job_boundary_is_refused() -> None:
    manifest = _manifest()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="maximum observation gap exceeds policy",
    ):
        _direct_gpu_stage_receipt(
            manifest,
            started_ns=1,
            finished_ns=60_000_000_002,
            samples=(
                _telemetry_sample(manifest, "alpha", 30_000_000_001),
            ),
        )


def test_trailing_only_attempt0_source_reuse_is_accepted_but_not_observed() -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        Path("/tmp/prismaquant-trailing-source-reuse-fixture"),
        _transports(manifest),
    )
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    source = {
        "schema": (
            "prismaquant.sample_parallel_probe."
            "worker_source_cache_receipt.v1"
        ),
        "disposition": "validated_reuse",
        "model": "/model",
        "cache": "/worker-state/source-identity-cache.json",
        "cache_sha256": "a" * 64,
        "identity": {"schema": "fixture-identity"},
        "portable_identity": {
            "schema": "prismaquant.streamed_model.portable_content.v1",
            "portable_content_sha256": manifest["inputs"][
                "model_content_sha256"
            ],
        },
    }
    receipt = _stage_receipt(
        manifest,
        command,
        request,
        _succeeded_job(
            request,
            started_ns=1,
            finished_ns=10,
            stdout=json.dumps(source, sort_keys=True).encode() + b"\n",
        ),
        _attempt_telemetry_state((
            _telemetry_sample(manifest, "alpha", 11),
        )),
        (),
        (),
        attempt_index=0,
        artifact_inspector=_FakeArtifactInspector(),
    )

    assert receipt["gpu_activity_requirement"] == (
        "waived_validated_source_cache_reuse"
    )
    assert receipt["telemetry"] == []
    assert receipt["telemetry_summary"][
        "outside_job_lifetime_sample_count"
    ] == 1
    assert receipt["telemetry_summary"]["sampling_attempt_count"] == 1
    assert receipt["telemetry_summary"]["successful_sample_count"] == 0


def test_trailing_only_numerical_telemetry_is_refused_as_unobserved() -> None:
    manifest = _manifest()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="no utilization telemetry",
    ):
        _direct_gpu_stage_receipt(
            manifest,
            started_ns=1,
            finished_ns=10,
            samples=(
                _telemetry_sample(manifest, "alpha", 11),
            ),
        )


def test_trailing_sample_is_excluded_but_remains_in_sampling_denominator() -> None:
    manifest = _manifest()
    receipt = _direct_gpu_stage_receipt(
        manifest,
        started_ns=1,
        finished_ns=10,
        samples=(
            _telemetry_sample(manifest, "alpha", 5),
            _telemetry_sample(manifest, "alpha", 11),
        ),
    )

    assert [sample["captured_ns"] for sample in receipt["telemetry"]] == [5]
    summary = receipt["telemetry_summary"]
    assert summary["outside_job_lifetime_sample_count"] == 1
    assert summary["sampling_attempt_count"] == 2
    assert summary["successful_sample_count"] == 1
    assert summary["successful_sample_percent"] == 50.0


def test_numerical_stage_with_only_na_gpu_utilization_is_refused() -> None:
    manifest = _manifest()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="no positive utilization sample",
    ):
        _direct_gpu_stage_receipt(
            manifest,
            started_ns=1,
            finished_ns=10,
            samples=(
                _telemetry_sample(
                    manifest,
                    "alpha",
                    5,
                    utilization=None,
                ),
            ),
        )


def test_na_utilization_sample_does_not_bridge_observation_gap() -> None:
    manifest = _manifest()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="maximum observation gap exceeds policy",
    ):
        _direct_gpu_stage_receipt(
            manifest,
            started_ns=1,
            finished_ns=60_000_000_001,
            samples=(
                _telemetry_sample(manifest, "alpha", 10_000_000_001),
                _telemetry_sample(
                    manifest,
                    "alpha",
                    30_000_000_001,
                    utilization=None,
                ),
                _telemetry_sample(manifest, "alpha", 50_000_000_001),
            ),
        )


@pytest.mark.parametrize("fault", ["generic_exception", "malformed_sample"])
def test_integrity_sampling_fault_finishes_live_job_then_retries_attempt(
    tmp_path: Path,
    fault: str,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    transport = transports["alpha"]
    original_sample = transport.sample_telemetry
    target_calls = 0

    def sample_with_one_integrity_fault():
        nonlocal target_calls
        job_id = transport._active_job_id
        if job_id is not None and transport._stage(job_id) == "sample_ce":
            target_calls += 1
            if target_calls == 2:
                if fault == "generic_exception":
                    raise RuntimeError("injected telemetry parser failure")
                return {"malformed": True}
        return original_sample()

    transport.sample_telemetry = sample_with_one_integrity_fault
    coordinator = _coordinator(
        manifest,
        tmp_path / fault,
        transports,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    requests = [
        request for request in transport.start_calls
        if "-sample_ce-alpha-" in request.job_id
    ]
    assert [request.job_id.rsplit("-", 1)[1] for request in requests] == [
        "00", "01",
    ]
    first_job = transport.jobs[requests[0].job_id]["receipt"]
    assert isinstance(first_job, JobReceipt)
    assert first_job.state == "succeeded"
    first_journal = json.loads(
        coordinator._attempt_telemetry_path(requests[0]).read_text(
            encoding="utf-8"
        )
    )
    assert first_journal["sampling_integrity_failure_count"] == 1
    receipt = json.loads(
        (tmp_path / fault / "receipts" / "sample_ce:alpha.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["attempt_index"] == 1


def test_wholly_unobserved_gpu_job_still_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    transports = _transports(manifest)

    def unavailable_telemetry():
        raise TelemetryUnavailableError(
            "injected persistent telemetry failure"
        )

    transports["zeta"].sample_telemetry = unavailable_telemetry
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="no utilization telemetry",
    ):
        coordinator.run_to_completion(resume=False)

    matching_starts = [
        request for request in transports["zeta"].start_calls
        if "-coordinator_source_identity-" in request.job_id
    ]
    assert len(matching_starts) == 2
    telemetry_paths = sorted(
        (tmp_path / "state" / "attempt-telemetry").glob(
            "*coordinator_source_identity*.json"
        )
    )
    assert len(telemetry_paths) == 2
    command = coordinator.commands["coordinator_source_identity:zeta"]
    for attempt_index, (request, path) in enumerate(
        zip(matching_starts, telemetry_paths, strict=True)
    ):
        assert path == coordinator._attempt_telemetry_path(request)
        journal = coordinator._load_attempt_telemetry(
            command, request, attempt_index,
        )
        assert journal["samples"] == []
        assert journal["sampling_failure_count"] == 2


def test_existing_job_with_different_request_identity_is_never_adopted(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )
    state = coordinator.initialize()
    command = coordinator.commands["host_preflight:alpha"]
    expected = coordinator._request(command, attempt_index=0)
    foreign = RunRequest(
        job_id=expected.job_id,
        argv=("/bin/false",),
        cwd="/",
        env=expected.env,
        inherit_env=False,
    )
    transports["alpha"].preload(foreign)

    with pytest.raises(RTX4090TwoHostCampaignError, match="failed assignment"):
        coordinator.advance(state, resume=True)

    assert all(
        request.job_id != expected.job_id
        for request in transports["alpha"].start_calls
    )
    persisted = coordinator.load_state()
    assert any(
        row["work_id"] == "host_preflight:zeta"
        for row in persisted["completions"]
    )


def test_missing_output_exhausts_attempts_without_completing_stage(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    inspectors = _inspectors(manifest)
    missing = "/srv/zeta/campaign-run/calibration.pt"
    inspectors["zeta"].missing.add(missing)
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )

    with pytest.raises(RTX4090TwoHostCampaignError, match="absent or unverifiable"):
        coordinator.run_to_completion(resume=False)

    requests = [
        request for request in transports["zeta"].start_calls
        if "-prepare_calibration-" in request.job_id
    ]
    assert len(requests) == 2
    assert len({request.job_id for request in requests}) == 2
    assert not any(
        row["work_id"] == "prepare_calibration:zeta"
        for row in coordinator.load_state()["completions"]
    )


def test_output_ledger_is_revalidated_and_refuses_content_drift(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    inspectors = _inspectors(manifest)
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )
    coordinator.run_to_completion(resume=False)
    path = "/srv/alpha/campaign-run/worker-0/probe.pkl"
    inspectors["alpha"].revisions[path] = 1

    with pytest.raises(RTX4090TwoHostCampaignError, match="content drifted"):
        coordinator.verify()


@pytest.mark.parametrize("unsafe", [
    "/run/../outside",
    "/worker-state/../../outside",
    "/model/not-an-output",
    "run/relative",
])
def test_output_path_resolution_refuses_traversal_and_readonly_roots(
    unsafe: str,
) -> None:
    manifest = _manifest()
    host = next(host for host in manifest["hosts"] if host["id"] == "alpha")
    with pytest.raises(RTX4090TwoHostCampaignError, match="output|escapes"):
        _host_output_path(host, unsafe)


def test_parallel_peer_receipt_and_state_are_persisted_before_barrier_returns(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    blocked = threading.Event()
    release = threading.Event()
    zeta = next(host for host in manifest["hosts"] if host["id"] == "zeta")
    transports["zeta"] = _BlockingTransport(
        zeta["expected"], blocked, release,
    )
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = coordinator.run_to_completion(resume=False)
        except BaseException as exc:  # surfaced in the test thread
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert blocked.wait(timeout=5)
    alpha_receipt = (
        tmp_path / "state" / "receipts" / "host_preflight:alpha.json"
    )
    deadline = time.monotonic() + 5
    persisted = coordinator.load_state()
    while time.monotonic() < deadline:
        alpha_completed = any(
            row["work_id"] == "host_preflight:alpha"
            for row in persisted["completions"]
        )
        if alpha_receipt.exists() and alpha_completed:
            break
        time.sleep(0.01)
        persisted = coordinator.load_state()
    assert alpha_receipt.is_file()
    assert any(
        row["work_id"] == "host_preflight:alpha"
        for row in persisted["completions"]
    )
    assert not any(
        row["work_id"] == "host_preflight:zeta"
        for row in persisted["completions"]
    )

    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["result"]["complete"] is True


def test_state_and_receipt_json_reads_reject_duplicate_members(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.initialize()
    state_path = tmp_path / "state" / "campaign-state.json"
    state_text = state_path.read_text(encoding="utf-8")
    state_path.write_text(
        state_text.replace(
            '  "schema":', '  "schema": "duplicate",\n  "schema":', 1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RTX4090TwoHostCampaignError, match="duplicate JSON"):
        coordinator.load_state()

    second = _coordinator(
        manifest, tmp_path / "second", _transports(manifest),
    )
    second.run_to_completion(resume=False)
    receipt_path = (
        tmp_path / "second" / "receipts" / "host_preflight:alpha.json"
    )
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        receipt_text.replace(
            '  "schema":', '  "schema": "duplicate",\n  "schema":', 1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RTX4090TwoHostCampaignError, match="duplicate JSON"):
        second.verify()


def test_gpu_stage_without_positive_utilization_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    transports = _transports(manifest, omit_telemetry_stage="sample_ce")
    coordinator = _coordinator(
        manifest, tmp_path / "state", transports,
    )

    with pytest.raises(RTX4090TwoHostCampaignError, match="positive utilization"):
        coordinator.run_to_completion(resume=False)


def test_validated_source_cache_reuse_is_the_only_zero_activity_waiver(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    command = coordinator.commands["worker_source_identity:alpha"]
    request = coordinator._request(command, attempt_index=0)
    source = {
        "schema": (
            "prismaquant.sample_parallel_probe."
            "worker_source_cache_receipt.v1"
        ),
        "disposition": "validated_reuse",
        "model": "/model",
        "cache": "/worker-state/source-identity-cache.json",
        "cache_sha256": "a" * 64,
        "identity": {"schema": "fixture-identity"},
        "portable_identity": {
            "schema": "prismaquant.streamed_model.portable_content.v1",
            "portable_content_sha256": manifest["inputs"][
                "model_content_sha256"
            ],
        },
    }

    def job(
        payload: dict[str, object],
        run_request: RunRequest = request,
    ) -> JobReceipt:
        return JobReceipt(
            job_id=run_request.job_id,
            request_sha256=run_request.request_sha256,
            state="succeeded",
            started_ns=1,
            finished_ns=2,
            returncode=0,
            pid=123,
            stdout=json.dumps(payload, sort_keys=True).encode() + b"\n",
            stderr=b"",
            transport="fake",
        )

    receipt = _stage_receipt(
        manifest,
        command,
        request,
        job(source),
        _attempt_telemetry_state(),
        (),
        (),
        attempt_index=0,
        artifact_inspector=_FakeArtifactInspector(),
    )

    assert receipt["gpu_activity_requirement"] == (
        "waived_validated_source_cache_reuse"
    )
    assert receipt["telemetry"] == []
    assert receipt["telemetry_summary"]["live_cuda_observed"] is False

    fresh = {**source, "disposition": "created"}
    with pytest.raises(RTX4090TwoHostCampaignError, match="no utilization"):
        _stage_receipt(
            manifest,
            command,
            request,
            job(fresh),
            _attempt_telemetry_state(),
            (),
            (),
            attempt_index=0,
            artifact_inspector=_FakeArtifactInspector(),
        )

    retry_request = coordinator._request(command, attempt_index=1)
    with pytest.raises(RTX4090TwoHostCampaignError, match="no utilization"):
        _stage_receipt(
            manifest,
            command,
            retry_request,
            job(source, retry_request),
            _attempt_telemetry_state(identity_sha256="b" * 64),
            (),
            (),
            attempt_index=1,
            artifact_inspector=_FakeArtifactInspector(),
        )


def test_verify_refuses_missing_or_tampered_execution_receipt(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest, tmp_path / "state", _transports(manifest),
    )
    coordinator.run_to_completion(resume=False)
    path = tmp_path / "state" / "receipts" / "measure_burn:alpha.json"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(RTX4090TwoHostCampaignError, match="receipt files differ"):
        coordinator.verify()
    path.write_bytes(original)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["telemetry_summary"]["live_cuda_observed"] = False
    path.write_text(
        json.dumps(raw, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RTX4090TwoHostCampaignError, match="digest differs"):
        coordinator.verify()


def _substitute_identical_symlink(path: Path) -> None:
    target = path.with_name(f"{path.name}.symlink-target")
    if path.is_dir():
        path.rename(target)
        path.symlink_to(target.name, target_is_directory=True)
        return
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target.name)


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize(
    "artifact",
    ("manifest", "plan", "state", "receipt", "receipt_directory"),
)
def test_current_and_legacy_verify_reject_stored_symlink_substitution(
    tmp_path: Path,
    legacy: bool,
    artifact: str,
) -> None:
    if legacy:
        manifest_path, state_dir = _write_legacy_v2_read_only_fixture(
            tmp_path,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        coordinator = CampaignCoordinator(manifest, state_dir)
        receipt_path = (
            state_dir / "receipts" / "host_preflight:alpha.json"
        )
    else:
        manifest = _manifest()
        state_dir = tmp_path / "current-state"
        coordinator = _coordinator(
            manifest, state_dir, _transports(manifest),
        )
        coordinator.run_to_completion(resume=False)
        receipt_path = (
            state_dir / "receipts" / "measure_burn:alpha.json"
        )
    paths = {
        "manifest": state_dir / "campaign-manifest.json",
        "plan": state_dir / "command-plan.json",
        "state": state_dir / "campaign-state.json",
        "receipt": receipt_path,
        "receipt_directory": state_dir / "receipts",
    }
    _substitute_identical_symlink(paths[artifact])

    with pytest.raises(RTX4090TwoHostCampaignError):
        coordinator.verify()


def test_verify_cli_rejects_symlink_manifest_input_before_open(
    tmp_path: Path,
) -> None:
    manifest_path, state_dir = _write_legacy_v2_read_only_fixture(tmp_path)
    _substitute_identical_symlink(manifest_path)

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="campaign manifest is invalid",
    ):
        main([
            "verify",
            "--manifest", str(manifest_path),
            "--state-dir", str(state_dir),
        ])


def test_manifest_v2_state_and_v1_receipt_verify_end_to_end_read_only(
    tmp_path: Path,
) -> None:
    manifest_path, state_dir = _write_legacy_v2_read_only_fixture(tmp_path)
    before = _filesystem_snapshot(state_dir)

    assert main(
        [
            "verify",
            "--manifest", str(manifest_path),
            "--state-dir", str(state_dir),
        ],
        transport_factory=lambda _manifest: {},
        artifact_inspector_factory=lambda _manifest: {},
    ) == 0

    assert _filesystem_snapshot(state_dir) == before


def test_manifest_v2_v1_receipt_binds_nonempty_stage_precondition_end_to_end(
    tmp_path: Path,
) -> None:
    precondition_sha256 = hashlib.sha256(
        b"legacy-host-preflight-precondition",
    ).hexdigest()
    manifest_path, state_dir = _write_legacy_v2_read_only_fixture(
        tmp_path,
        precondition_receipt_sha256s=(precondition_sha256,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calls: list[tuple[str, bool]] = []

    def stage_preconditioner(stage: str, *, verify_only: bool):
        calls.append((stage, verify_only))
        return (precondition_sha256,)

    coordinator = CampaignCoordinator(
        manifest,
        state_dir,
        stage_preconditioner=stage_preconditioner,
    )
    before = _filesystem_snapshot(state_dir)

    status = coordinator.verify()

    assert status["completed_assignments"] == 1
    assert status["complete"] is False
    assert calls == [("host_preflight", True)]
    assert _filesystem_snapshot(state_dir) == before


def test_manifest_v2_advance_and_execute_are_read_only_and_start_nothing(
    tmp_path: Path,
) -> None:
    manifest_path, state_dir = _write_legacy_v2_read_only_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transports = _transports(manifest)
    coordinator = _coordinator(
        manifest,
        state_dir,
        transports,
    )
    state = coordinator.load_state()
    ready = next_ready_assignments(manifest, state)
    assert ready
    before = _filesystem_snapshot(state_dir)

    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="legacy campaign manifests are read-only; cannot advance",
    ):
        coordinator.advance(state, resume=True)
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="legacy campaign manifests are read-only; cannot execute",
    ):
        coordinator._execute(
            ready[0],
            dependency_receipt_sha256s=(),
            precondition_receipt_sha256s=(),
        )
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="legacy campaign manifests are read-only; cannot evaluate",
    ):
        coordinator._stage_preconditions("host_preflight", verify_only=False)

    assert sum(
        len(transport.start_calls) for transport in transports.values()
    ) == 0
    assert _filesystem_snapshot(state_dir) == before


def test_manifest_v2_cannot_be_resealed_or_planned_and_writes_nothing(
    tmp_path: Path,
) -> None:
    manifest = _legacy_v2_manifest()
    manifest_body = dict(manifest)
    manifest_body.pop("identity_sha256")
    with pytest.raises(
        ClusterCampaignContractError,
        match="must use the current schema",
    ):
        seal_campaign_manifest(manifest_body)

    manifest_path = tmp_path / "legacy-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"
    before = _filesystem_snapshot(tmp_path)
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="legacy campaign manifests are read-only",
    ):
        main([
            "plan",
            "--manifest", str(manifest_path),
            "--output", str(output),
        ])

    assert not output.exists()
    assert _filesystem_snapshot(tmp_path) == before


def test_v2_execution_receipt_binds_exact_journal_and_refuses_drift_or_absence(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = _coordinator(
        manifest,
        tmp_path / "state",
        _transports(manifest),
    )
    coordinator.run_to_completion(resume=False)
    receipt = json.loads(
        (tmp_path / "state" / "receipts" /
         "measure_burn:alpha.json").read_text(encoding="utf-8")
    )
    assert receipt["schema"] == EXECUTION_RECEIPT_SCHEMA
    command = coordinator.commands["measure_burn:alpha"]
    request = coordinator._request(
        command,
        attempt_index=receipt["attempt_index"],
    )
    journal_path = coordinator._attempt_telemetry_path(request)
    original = journal_path.read_bytes()
    journal = json.loads(original)
    assert (
        receipt["attempt_telemetry_identity_sha256"]
        == journal["identity_sha256"]
    )

    journal["sampling_failure_count"] = 1
    journal["consecutive_sampling_failure_count"] = 1
    journal["maximum_consecutive_sampling_failure_count"] = 1
    body = dict(journal)
    body.pop("identity_sha256")
    journal["identity_sha256"] = canonical_sha256(body)
    journal_path.write_text(
        json.dumps(journal, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="attempt telemetry head/record chain differs",
    ):
        coordinator.verify()

    journal_path.write_bytes(original)
    assert coordinator.verify()["complete"] is True
    journal_path.unlink()
    with pytest.raises(
        RTX4090TwoHostCampaignError,
        match="attempt telemetry is absent",
    ):
        coordinator.verify()


def test_cli_plan_is_deterministic_and_live_run_uses_default_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "plan.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    assert main([
        "plan", "--manifest", str(manifest_path), "--output", str(output),
    ]) == 0
    first = output.read_bytes()
    assert main([
        "plan", "--manifest", str(manifest_path), "--output", str(output),
    ]) == 0
    assert output.read_bytes() == first

    import prismaquant.rtx4090_two_host_application as live_application

    observed: dict[str, object] = {}

    class _Application:
        def run_to_completion(self, *, resume: bool):
            observed["resume"] = resume
            return {"complete": True}

    def build(application_manifest, state_dir, *, initialize):
        observed.update({
            "manifest": application_manifest,
            "state_dir": state_dir,
            "initialize": initialize,
        })
        return _Application()

    monkeypatch.setattr(live_application, "build_live_campaign_application", build)
    assert main([
        "run", "--manifest", str(manifest_path),
        "--state-dir", str(tmp_path / "live-state"),
    ]) == 0
    assert observed == {
        "manifest": manifest,
        "state_dir": str(tmp_path / "live-state"),
        "initialize": True,
        "resume": False,
    }


def test_cli_legacy_verify_routes_through_application_verify_without_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, state_dir = _write_legacy_v2_read_only_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import prismaquant.rtx4090_two_host_application as live_application

    observed: dict[str, object] = {"run_calls": 0, "verify_calls": 0}

    class _Application:
        def run_to_completion(self, *, resume: bool):
            del resume
            observed["run_calls"] = int(observed["run_calls"]) + 1
            pytest.fail("legacy verify must not run the application")

        def verify(self):
            observed["verify_calls"] = int(observed["verify_calls"]) + 1
            return {"complete": False}

    def build(application_manifest, application_state_dir, *, initialize):
        observed.update({
            "manifest": application_manifest,
            "state_dir": application_state_dir,
            "initialize": initialize,
        })
        return _Application()

    monkeypatch.setattr(
        live_application,
        "build_live_campaign_application",
        build,
    )

    assert main([
        "verify", "--manifest", str(manifest_path),
        "--state-dir", str(state_dir),
    ]) == 0
    assert observed == {
        "run_calls": 0,
        "verify_calls": 1,
        "manifest": manifest,
        "state_dir": str(state_dir),
        "initialize": False,
    }
