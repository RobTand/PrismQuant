from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import pytest

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    bind_gridbook_runtime_contract,
    seal_campaign_manifest,
)
from prismaquant.cluster_transport import (
    GpuSample,
    JobReceipt,
    ManifestEntry,
    RunRequest,
    TelemetrySnapshot,
    TreeManifest,
)
from prismaquant.rtx4090_two_host_campaign import (
    CampaignCoordinator,
    RTX4090TwoHostCampaignError,
    _host_output_path,
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
    assert target["physical_formats"] == [
        RTX4090_QWEN38_FORMAT_MENU[0],
        RTX4090_QWEN38_FORMAT_MENU[3],
        RTX4090_QWEN38_FORMAT_MENU[-3],
        RTX4090_QWEN38_FORMAT_MENU[-2],
    ]
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
                "FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48", "FP8_E4M3",
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
                raise RuntimeError("duplicate fake job id")
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
                raise RuntimeError("unknown fake job")
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
                raise RuntimeError("injected telemetry interruption")
            self._telemetry_counter += 1
            captured_ns = self._telemetry_counter * 100
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


def test_restart_adopts_running_job_and_uses_durable_live_telemetry(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    transports["zeta"].interrupt_telemetry_once_stage = (
        "coordinator_source_identity"
    )
    inspectors = _inspectors(manifest)
    first = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )

    with pytest.raises(RTX4090TwoHostCampaignError, match="live telemetry"):
        first.run_to_completion(resume=False)

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
    journal = json.loads(telemetry_paths[0].read_text(encoding="utf-8"))
    assert len(journal["samples"]) == 1
    assert journal["samples"][0]["gpus"][0]["gpu_utilization_pct"] == 75.0

    resumed = _coordinator(
        manifest, tmp_path / "state", transports, inspectors,
    )
    result = resumed.run_to_completion(resume=True)

    assert result["complete"] is True
    assert len([
        request for request in transports["zeta"].start_calls
        if "-coordinator_source_identity-" in request.job_id
    ]) == 1
    receipt = json.loads(
        (tmp_path / "state" / "receipts" /
         "coordinator_source_identity:zeta.json").read_text(encoding="utf-8")
    )
    assert len(receipt["telemetry"]) == 1
    assert receipt["telemetry_summary"]["live_cuda_observed"] is True
    allocate = next(
        request for request in transports["zeta"].start_calls
        if "-allocate-" in request.job_id
    )
    assert allocate.job_id.endswith("-attempt-00")
    assert "--resume" not in allocate.argv


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

    def job(payload: dict[str, object]) -> JobReceipt:
        return JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
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
        (),
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
            (),
            (),
            (),
            attempt_index=0,
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
    path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RTX4090TwoHostCampaignError, match="digest differs"):
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
