from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prismaquant.rtx4090_two_host_application as application_module

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    canonical_sha256,
    seal_campaign_manifest,
)
from prismaquant.cluster_host_admission import (
    GPU_LEASE_SCHEMA,
    GPU_LEASE_OPERATION_SCHEMA,
    GPU_START_GUARD_RECEIPT_SCHEMA,
    HOST_PRE_ADMISSION_RECEIPT_SCHEMA,
)
from prismaquant.cluster_transport import (
    GpuSample,
    JobReceipt,
    TelemetrySnapshot,
    build_tree_manifest,
    canonical_json_bytes,
)
from prismaquant.rtx4090_two_host_application import (
    APPLICATION_RECEIPT_DIR,
    GUARD_DIR,
    LEASE_RELEASE_DIR,
    LiveCampaignApplication,
    RTX4090TwoHostApplicationError,
    _verify_controller_snapshot,
)
from prismaquant.rtx4090_two_host_campaign import (
    _expected_output_kind,
    _host_output_path,
)


def _host(root: Path, host_id: str, *, local: bool) -> dict[str, object]:
    return {
        "id": host_id,
        "transport": (
            {"kind": "local"}
            if local else {
                "kind": "ssh",
                "host": "peer.example.test",
                "port": 22,
                "user": "campaign_runner",
            }
        ),
        "roots": {
            "model_root": str(root / "model"),
            "dataset_path": str(root / "data" / "calibration.jsonl"),
            "snapshot_root": str(root / "snapshot"),
            "run_root": str(root / "run"),
            "worker_state_root": str(root / "worker-state"),
        },
        "expected": {
            "hostname": f"{host_id}-host",
            "gpu": {
                "name": "NVIDIA GB10",
                "uuid": f"GPU-{host_id.upper()}-0123456789",
                "compute_capability": [12, 1],
                "device_count": 1,
            },
            "image_digest": "sha256:" + "4" * 64,
            "producer_commit": "1" * 40,
            "uid": 1000,
            "gid": 1000,
        },
    }


def _manifest(tmp_path: Path) -> dict[str, object]:
    return seal_campaign_manifest({
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_id": "application-e2e-test",
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
            _host(tmp_path / "zeta", "zeta", local=False),
            _host(tmp_path / "alpha", "alpha", local=True),
        ],
    })


def _sealed(body: dict[str, object]) -> dict[str, object]:
    return {**body, "identity_sha256": canonical_sha256(body)}


def _expected_lease(manifest, host_id):
    host = next(host for host in manifest["hosts"] if host["id"] == host_id)
    return _sealed({
        "schema": GPU_LEASE_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "campaign_identity_sha256": manifest["identity_sha256"],
        "host_id": host_id,
        "host_identity_sha256": canonical_sha256(host),
        "gpu_uuid": host["expected"]["gpu"]["uuid"],
    })


class _ActionTransport:
    def __init__(self, manifest, host_id):
        self.manifest = manifest
        self.host_id = host_id
        self.counter = 0
        self.jobs = {}

    def run(self, request):
        self.counter += 1
        action = request.argv[-3]
        if action == "pre-admit":
            lease = _expected_lease(self.manifest, self.host_id)
            result = _sealed({
                "schema": HOST_PRE_ADMISSION_RECEIPT_SCHEMA,
                "campaign_identity_sha256": self.manifest["identity_sha256"],
                "host_id": self.host_id,
                "lease": _sealed({
                    "schema": GPU_LEASE_OPERATION_SCHEMA,
                    "action": "acquire",
                    "disposition": "acquired",
                    "lease": lease,
                }),
            })
        elif action == "guard":
            result = _sealed({
                "schema": GPU_START_GUARD_RECEIPT_SCHEMA,
                "campaign_identity_sha256": self.manifest["identity_sha256"],
                "host_id": self.host_id,
                "lease": {"lease": "held"},
            })
        elif action == "release":
            result = _sealed({
                "schema": GPU_LEASE_OPERATION_SCHEMA,
                "action": "release",
                "disposition": "released",
                "lease": _expected_lease(self.manifest, self.host_id),
            })
        else:  # pragma: no cover - fixed application actions only
            raise AssertionError(action)
        stdout = canonical_json_bytes(result) + b"\n"
        receipt = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state="succeeded",
            started_ns=self.counter * 10 + 1,
            finished_ns=self.counter * 10 + 2,
            returncode=0,
            pid=1000 + self.counter,
            stdout=stdout,
            stderr=b"",
            transport=f"action:{self.host_id}",
        )
        self.jobs[request.job_id] = receipt
        return receipt

    def start(self, request):
        return self.run(request)

    def status(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]


class _LostStartResponseActionTransport(_ActionTransport):
    def __init__(self, manifest, host_id):
        super().__init__(manifest, host_id)
        self.lost = False

    def start(self, request):
        receipt = super().start(request)
        if not self.lost:
            self.lost = True
            raise RuntimeError("injected lost start response")
        return receipt


class _TerminalFailureActionTransport(_ActionTransport):
    def __init__(self, manifest, host_id, *, failures: int):
        super().__init__(manifest, host_id)
        self.failures = failures

    def run(self, request):
        action = request.argv[-3]
        if action == "pre-admit" and self.failures > 0:
            self.failures -= 1
            self.counter += 1
            receipt = JobReceipt(
                job_id=request.job_id,
                request_sha256=request.request_sha256,
                state="failed",
                started_ns=self.counter * 10 + 1,
                finished_ns=self.counter * 10 + 2,
                returncode=9,
                pid=1000 + self.counter,
                stdout=b"",
                stderr=b"injected transient admission failure\n",
                transport=f"action:{self.host_id}",
            )
            self.jobs[request.job_id] = receipt
            return receipt
        return super().run(request)


class _ReleaseCrashWindowActionTransport(_ActionTransport):
    def __init__(self, manifest, host_id):
        super().__init__(manifest, host_id)
        self.release_attempts = 0

    def run(self, request):
        action = request.argv[-3]
        if action != "release":
            return super().run(request)
        self.release_attempts += 1
        self.counter += 1
        if self.release_attempts == 1:
            # Model a worker that crossed the durable host-side unlink but
            # died before it could publish the release result to transport.
            receipt = JobReceipt(
                job_id=request.job_id,
                request_sha256=request.request_sha256,
                state="transport_error",
                started_ns=self.counter * 10 + 1,
                finished_ns=self.counter * 10 + 2,
                returncode=None,
                pid=1000 + self.counter,
                stdout=b"",
                stderr=b"injected post-release receipt crash\n",
                transport=f"action:{self.host_id}",
            )
        else:
            result = _sealed({
                "schema": GPU_LEASE_OPERATION_SCHEMA,
                "action": "release",
                "disposition": "already_absent",
                "lease": _expected_lease(self.manifest, self.host_id),
            })
            receipt = JobReceipt(
                job_id=request.job_id,
                request_sha256=request.request_sha256,
                state="succeeded",
                started_ns=self.counter * 10 + 1,
                finished_ns=self.counter * 10 + 2,
                returncode=0,
                pid=1000 + self.counter,
                stdout=canonical_json_bytes(result) + b"\n",
                stderr=b"",
                transport=f"action:{self.host_id}",
            )
        self.jobs[request.job_id] = receipt
        return receipt


class _FreshAbsentReleaseActionTransport(_ActionTransport):
    def run(self, request):
        if request.argv[-3] != "release":
            return super().run(request)
        self.counter += 1
        result = _sealed({
            "schema": GPU_LEASE_OPERATION_SCHEMA,
            "action": "release",
            "disposition": "already_absent",
            "lease": _expected_lease(self.manifest, self.host_id),
        })
        receipt = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state="succeeded",
            started_ns=self.counter * 10 + 1,
            finished_ns=self.counter * 10 + 2,
            returncode=0,
            pid=1000 + self.counter,
            stdout=canonical_json_bytes(result) + b"\n",
            stderr=b"",
            transport=f"action:{self.host_id}",
        )
        self.jobs[request.job_id] = receipt
        return receipt


class _Inspector:
    def inspect_artifact(self, path):
        return build_tree_manifest(path)


class _ExecutionTransport:
    cadence_seconds = 1.0

    def __init__(self, manifest, host_id, *, fail_once_stage=None):
        self.manifest = manifest
        self.host_id = host_id
        self.application = None
        self.jobs = {}
        self.counter = 0
        self.fail_once_stage = fail_once_stage

    def _command(self, request):
        assert self.application is not None
        return next(
            command for command in self.application.coordinator.commands.values()
            if command.host_id == self.host_id
            and f"-{command.stage}-{self.host_id}-" in request.job_id
        )

    def status(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def start(self, request):
        self.counter += 1
        command = self._command(request)
        failed = (
            command.stage == self.fail_once_stage
            and request.job_id.endswith("-attempt-00")
        )
        host = next(
            host for host in self.manifest["hosts"] if host["id"] == self.host_id
        )
        for container_path in (() if failed else command.outputs):
            host_path, _ = _host_output_path(host, container_path)
            path = Path(host_path)
            if _expected_output_kind(container_path) == "directory":
                path.mkdir(parents=True, exist_ok=True)
                (path / "payload.bin").write_bytes(command.work_id.encode())
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(command.work_id.encode())
        stdout = b"ok\n"
        if command.stage == "worker_source_identity":
            source = {
                "schema": (
                    "prismaquant.sample_parallel_probe."
                    "worker_source_cache_receipt.v1"
                ),
                "portable_identity": {
                    "schema": "prismaquant.streamed_model.portable_content.v1",
                    "portable_content_sha256": self.manifest["inputs"][
                        "model_content_sha256"
                    ],
                    "checkpoint_shards": 18,
                    "checkpoint_tensors": 866,
                },
            }
            stdout = json.dumps(source, sort_keys=True).encode() + b"\n"
        running = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state="running",
            started_ns=self.counter * 100 + 1,
            finished_ns=None,
            returncode=None,
            pid=2000 + self.counter,
            stdout=b"",
            stderr=b"",
            transport=f"execution:{self.host_id}",
        )
        self.jobs[request.job_id] = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state="failed" if failed else "succeeded",
            started_ns=running.started_ns,
            finished_ns=running.started_ns + 5,
            returncode=9 if failed else 0,
            pid=running.pid,
            stdout=b"" if failed else stdout,
            stderr=b"injected stage failure\n" if failed else b"",
            transport=running.transport,
        )
        return running

    def sample_telemetry(self):
        host = next(
            host for host in self.manifest["hosts"] if host["id"] == self.host_id
        )
        gpu = host["expected"]["gpu"]
        return TelemetrySnapshot(
            captured_ns=10_000 + self.counter,
            host_mem_available_bytes=128 * 1024**3,
            gpus=(GpuSample(
                timestamp="2026-08-24T00:00:00Z",
                index=0,
                name=gpu["name"],
                uuid=gpu["uuid"],
                pci_bus_id="00000000:01:00.0",
                gpu_utilization_pct=75.0,
                memory_utilization_pct=25.0,
                memory_used_mib=4096.0,
                memory_total_mib=24576.0,
                temperature_c=55.0,
                power_w=200.0,
            ),),
        )


def test_controller_must_execute_from_and_verify_the_local_sealed_snapshot(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    snapshot = tmp_path / "alpha/snapshot"
    source = snapshot / "prismaquant/rtx4090_two_host_application.py"
    source.parent.mkdir(parents=True)
    source.write_text("# exact controller fixture\n")
    producer = manifest["producer"]
    calls = []

    def verifier(root, **expected):
        calls.append((root, expected))
        return {
            "schema": "prismaquant.runtime_source_snapshot.v1",
            "snapshot": str(snapshot),
            "commit": producer["commit"],
            "tree": producer["tree"],
            "closure_sha256": producer["snapshot_sha256"],
            "entry_count": 123,
        }

    receipt = _verify_controller_snapshot(
        manifest, source_file=source, verifier=verifier,
    )

    assert receipt["entry_count"] == 123
    assert calls == [(
        snapshot,
        {
            "expected_commit": producer["commit"],
            "expected_tree": producer["tree"],
            "expected_closure_sha256": producer["snapshot_sha256"],
        },
    )]
    with pytest.raises(RTX4090TwoHostApplicationError, match="not executing"):
        _verify_controller_snapshot(
            manifest,
            source_file=Path(__file__).resolve(),
            verifier=verifier,
        )


def test_controller_state_cannot_live_in_a_container_writable_mount(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        host["id"]: _ActionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports={
            host_id: _ExecutionTransport(manifest, host_id)
            for host_id in actions
        },
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    local = next(
        host for host in manifest["hosts"]
        if host["transport"]["kind"] == "local"
    )
    unsafe = Path(local["roots"]["run_root"]) / "controller-state"

    with pytest.raises(
        RTX4090TwoHostApplicationError, match="outside container-writable",
    ):
        LiveCampaignApplication(manifest, unsafe, runtime=runtime)

    for field in ("model_root", "snapshot_root"):
        with pytest.raises(
            RTX4090TwoHostApplicationError,
            match=f"overlaps local declared root {field}",
        ):
            LiveCampaignApplication(
                manifest,
                Path(local["roots"][field]) / "controller-state",
                runtime=runtime,
            )

    derived_control = (
        Path(local["roots"]["worker_state_root"]).parent
        / ".prismaquant-cluster-control" / manifest["identity_sha256"]
    )
    with pytest.raises(
        RTX4090TwoHostApplicationError,
        match="overlaps local declared root derived_control_0",
    ):
        LiveCampaignApplication(
            manifest,
            derived_control / "controller-state",
            runtime=runtime,
        )


def test_controller_state_rejects_symlinked_ancestor_before_runtime_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    local = next(
        host for host in manifest["hosts"]
        if host["transport"]["kind"] == "local"
    )
    writable = Path(local["roots"]["run_root"])
    writable.mkdir(parents=True)
    alias = tmp_path / "controller-alias"
    alias.symlink_to(writable, target_is_directory=True)

    monkeypatch.setattr(
        application_module,
        "_verify_controller_snapshot",
        lambda *_args, **_kwargs: pytest.fail("snapshot verification ran"),
    )
    monkeypatch.setattr(
        application_module,
        "build_live_campaign_runtime",
        lambda *_args, **_kwargs: pytest.fail("live runtime mutated endpoints"),
    )

    with pytest.raises(
        RTX4090TwoHostApplicationError, match="ancestry contains a symlink",
    ):
        application_module.build_live_campaign_application(
            manifest,
            alias / "controller-state",
            initialize=True,
        )


def test_application_owns_admission_guards_model_gate_and_lease_release(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        host["id"]: _ActionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    executions = {
        host["id"]: _ExecutionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports=executions,
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )
    for transport in executions.values():
        transport.application = application

    result = application.run_to_completion(resume=False)

    assert result["complete"] is True
    assert application.coordinator.status()["completed_assignments"] == 21
    application.verify()
    root = state_dir / APPLICATION_RECEIPT_DIR
    # Every container launch is guarded; only the two bare-host preflights are
    # exempt from the 21-assignment schedule.
    assert len(list((root / GUARD_DIR).glob("*.check-*.json"))) == 19
    assert len(list((root / LEASE_RELEASE_DIR).glob("*.json"))) == 2
    assert all(
        receipt["precondition_receipt_sha256s"]
        for receipt in (
            json.loads(path.read_text())
            for path in (state_dir / "receipts").glob("*.json")
        )
    )


def test_release_resume_accepts_exact_tombstone_after_prior_failed_attempt(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        "alpha": _ActionTransport(manifest, "alpha"),
        "zeta": _ReleaseCrashWindowActionTransport(manifest, "zeta"),
    }
    executions = {
        host_id: _ExecutionTransport(manifest, host_id)
        for host_id in actions
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports=executions,
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(manifest, state_dir, runtime=runtime)
    for transport in executions.values():
        transport.application = application

    result = application.run_to_completion(resume=False)

    assert result["complete"] is True
    application.verify()
    assert actions["zeta"].release_attempts == 2
    receipt = json.loads((
        state_dir / APPLICATION_RECEIPT_DIR / LEASE_RELEASE_DIR / "zeta.json"
    ).read_text(encoding="utf-8"))
    assert receipt["action_attempt_index"] == 1
    assert receipt["result"]["disposition"] == "already_absent"
    assert receipt["result"]["lease"] == _expected_lease(manifest, "zeta")


def test_fresh_release_refuses_already_absent_without_a_prior_exact_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        "alpha": _ActionTransport(manifest, "alpha"),
        "zeta": _FreshAbsentReleaseActionTransport(manifest, "zeta"),
    }
    executions = {
        host_id: _ExecutionTransport(manifest, host_id)
        for host_id in actions
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports=executions,
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    application = LiveCampaignApplication(
        manifest, tmp_path / "controller-state", runtime=runtime,
    )
    for transport in executions.values():
        transport.application = application

    with pytest.raises(
        RTX4090TwoHostApplicationError, match="has no prior retry attempt",
    ):
        application.run_to_completion(resume=False)


def test_host_action_adopts_exact_job_after_lost_start_response(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        "alpha": _ActionTransport(manifest, "alpha"),
        "zeta": _LostStartResponseActionTransport(manifest, "zeta"),
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports={
            host_id: _ExecutionTransport(manifest, host_id)
            for host_id in actions
        },
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )

    application.prepare(resume=False)

    assert actions["zeta"].lost is True
    assert actions["zeta"].counter == 1
    assert len(list(
        (state_dir / APPLICATION_RECEIPT_DIR / "pre-admission").glob("*.json")
    )) == 2


def test_resume_completes_partial_admission_with_new_bounded_action_tokens(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    ordered_ids = [str(host["id"]) for host in manifest["hosts"]]
    first_id, failing_id = ordered_ids
    actions = {
        first_id: _ActionTransport(manifest, first_id),
        failing_id: _TerminalFailureActionTransport(
            manifest, failing_id, failures=2,
        ),
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports={
            host_id: _ExecutionTransport(manifest, host_id)
            for host_id in actions
        },
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )

    with pytest.raises(
        RTX4090TwoHostApplicationError, match="exhausted 2 new attempt",
    ):
        application.prepare(resume=False)

    pre_root = state_dir / APPLICATION_RECEIPT_DIR / "pre-admission"
    assert (pre_root / f"{first_id}.json").is_file()
    assert not (pre_root / f"{failing_id}.json").exists()

    application.prepare(resume=True)

    recovered = json.loads(
        (pre_root / f"{failing_id}.json").read_text(encoding="utf-8")
    )
    assert recovered["action_attempt_index"] == 2
    assert actions[first_id].counter == 1
    assert actions[failing_id].counter == 3


def test_failed_execution_attempt_keeps_a_valid_extra_start_guard(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        host["id"]: _ActionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    executions = {
        host_id: _ExecutionTransport(
            manifest,
            host_id,
            fail_once_stage="measure_burn" if host_id == "alpha" else None,
        )
        for host_id in actions
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports=executions,
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )
    for transport in executions.values():
        transport.application = application

    result = application.run_to_completion(resume=False)

    assert result["complete"] is True
    application.verify()
    guard_root = state_dir / APPLICATION_RECEIPT_DIR / GUARD_DIR
    journals = [
        path for path in guard_root.glob("*.json")
        if ".check-" not in path.name
    ]
    assert len(journals) == 20
    receipt = json.loads(
        (state_dir / "receipts" / "measure_burn:alpha.json").read_text()
    )
    assert receipt["attempt_index"] == 1


def test_application_retains_leases_when_campaign_does_not_verify(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        host["id"]: _ActionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports={
            host_id: _ExecutionTransport(manifest, host_id)
            for host_id in actions
        },
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )

    def fail_campaign(*, resume: bool):
        del resume
        raise RuntimeError("injected campaign failure")

    application.coordinator.run_to_completion = fail_campaign
    with pytest.raises(RuntimeError, match="injected campaign failure"):
        application.run_to_completion(resume=False)

    assert all(transport.counter == 1 for transport in actions.values())
    assert not (
        state_dir / APPLICATION_RECEIPT_DIR / LEASE_RELEASE_DIR
    ).exists()


def test_application_verifies_its_receipts_before_releasing_leases(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    actions = {
        host["id"]: _ActionTransport(manifest, host["id"])
        for host in manifest["hosts"]
    }
    runtime = SimpleNamespace(
        local_transport=actions["alpha"],
        ssh_transport=actions["zeta"],
        transports={
            host_id: _ExecutionTransport(manifest, host_id)
            for host_id in actions
        },
        artifact_inspectors={host_id: _Inspector() for host_id in actions},
        barrier_stages=(),
        route_specs_for_stage=lambda _stage: (),
    )
    state_dir = tmp_path / "controller-state"
    application = LiveCampaignApplication(
        manifest, state_dir, runtime=runtime,
    )
    application.coordinator.run_to_completion = (
        lambda *, resume: {"complete": not resume}
    )

    def refuse_application_receipts(*, require_release: bool):
        assert require_release is False
        raise RuntimeError("injected application receipt drift")

    application.verify_application = refuse_application_receipts
    with pytest.raises(RuntimeError, match="application receipt drift"):
        application.run_to_completion(resume=False)

    assert all(transport.counter == 1 for transport in actions.values())
    assert not (
        state_dir / APPLICATION_RECEIPT_DIR / LEASE_RELEASE_DIR
    ).exists()
