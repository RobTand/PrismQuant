from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    seal_campaign_manifest,
)
from prismaquant.cluster_transport import GpuSample, JobReceipt, TelemetrySnapshot
from prismaquant.rtx4090_two_host_campaign import (
    CampaignCoordinator,
    RTX4090TwoHostCampaignError,
    build_command_plan,
    main,
)


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
    ) -> None:
        self.expected = expected
        self.fail_once_stage = fail_once_stage
        self.omit_telemetry_stage = omit_telemetry_stage
        self.calls: list[object] = []
        self._counter = 0
        self._failed = False
        self._lock = threading.Lock()

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
        ):
            if f"-{stage}-" in job_id:
                return stage
        raise AssertionError(f"unknown fixed-stage job id {job_id}")

    def run_with_telemetry(self, request):
        stage = self._stage(request.job_id)
        with self._lock:
            self.calls.append(request)
            self._counter += 1
            ordinal = self._counter
            should_fail = (
                stage == self.fail_once_stage and not self._failed
            )
            if should_fail:
                self._failed = True
        if should_fail:
            raise RuntimeError(f"injected {stage} failure")
        receipt = JobReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            state="succeeded",
            started_ns=ordinal * 10 + 1,
            finished_ns=ordinal * 10 + 9,
            returncode=0,
            pid=1000 + ordinal,
            stdout=b"ok\n",
            stderr=b"",
            transport="fake",
        )
        gpu_stages = {
            "coordinator_source_identity",
            "worker_source_identity",
            "sample_ce",
            "sample_fisher",
            "measure_burn",
        }
        if stage not in gpu_stages or stage == self.omit_telemetry_stage:
            return receipt, ()
        gpu = self.expected["gpu"]
        samples = tuple(
            TelemetrySnapshot(
                captured_ns=ordinal * 100 + offset,
                host_mem_available_bytes=24_000_000_000 - offset,
                gpus=(GpuSample(
                    timestamp=f"2026-08-24T00:00:0{offset}Z",
                    index=0,
                    name=str(gpu["name"]),
                    uuid=str(gpu["uuid"]),
                    pci_bus_id="00000000:01:00.0",
                    gpu_utilization_pct=0.0 if offset == 1 else 75.0,
                    memory_utilization_pct=25.0,
                    memory_used_mib=4096.0,
                    memory_total_mib=24576.0,
                    temperature_c=55.0,
                    power_w=200.0,
                ),),
            )
            for offset in (1, 2)
        )
        return receipt, samples


def _transports(
    manifest: dict[str, object], **kwargs: object,
) -> dict[str, _FakeTransport]:
    return {
        str(host["id"]): _FakeTransport(host["expected"], **kwargs)
        for host in manifest["hosts"]
    }


def test_plan_is_closed_and_maps_hosts_to_partitions_and_stripes_0_and_1() -> None:
    manifest = _manifest()
    plan = build_command_plan(manifest)
    commands = plan["commands"]
    assert isinstance(commands, list)
    assert len(commands) == 21

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
        if row["stage"] in {"prepare_calibration", "prepare_burn", "allocate"}
    ]
    assert {row["host_id"] for row in coordinator_rows} == {"zeta"}
    prepare = next(row for row in commands if row["stage"] == "prepare_burn")
    assert "eugr/spark-vllm@sha256:" + "4" * 64 in prepare["argv"]
    assert "type=bind,src=/srv/zeta/model,dst=/model,readonly" in prepare["argv"]
    assert "type=bind,src=/srv/zeta/campaign-run,dst=/run" in prepare["argv"]
    assert "PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1" in prepare["argv"]
    assert all("shell" not in row for row in commands)


def test_fake_transport_completes_all_barriers_and_records_kpi_receipts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    transports = _transports(manifest)
    coordinator = CampaignCoordinator(
        manifest, tmp_path / "state", transports=transports,
    )

    result = coordinator.run_to_completion(resume=False)

    assert result["complete"] is True
    assert result["completed_assignments"] == 21
    assert sum(len(transport.calls) for transport in transports.values()) == 21
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
    assert coordinator.verify() == result


def test_resume_preserves_successful_peer_and_uses_only_fixed_resume_flags(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    first = _transports(manifest)
    first["zeta"].fail_once_stage = "sample_fisher"
    coordinator = CampaignCoordinator(
        manifest, tmp_path / "state", transports=first,
    )
    with pytest.raises(RTX4090TwoHostCampaignError, match="failed assignment"):
        coordinator.run_to_completion(resume=False)

    state = coordinator.load_state()
    completed_fisher = [
        row["host_id"] for row in state["completions"]
        if row["stage"] == "sample_fisher"
    ]
    assert len(completed_fisher) == 1

    second = _transports(manifest)
    resumed = CampaignCoordinator(
        manifest, tmp_path / "state", transports=second,
    )
    result = resumed.run_to_completion(resume=True)
    assert result["complete"] is True
    resumed_job_ids = [
        request.job_id
        for transport in second.values()
        for request in transport.calls
    ]
    assert not any(
        f"-sample_fisher-{completed_fisher[0]}" in job_id
        for job_id in resumed_job_ids
    )
    resumed_requests = [
        request
        for transport in second.values()
        for request in transport.calls
    ]
    measure = [request for request in resumed_requests if "-measure_burn-" in request.job_id]
    allocate = next(
        request for request in resumed_requests if "-allocate-" in request.job_id
    )
    assert len(measure) == 2
    assert all("--resume" in request.argv for request in measure)
    assert "--resume" in allocate.argv
    assert all(request.inherit_env is False for request in resumed_requests)


def test_gpu_stage_without_positive_utilization_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    transports = _transports(manifest, omit_telemetry_stage="sample_ce")
    coordinator = CampaignCoordinator(
        manifest, tmp_path / "state", transports=transports,
    )

    with pytest.raises(RTX4090TwoHostCampaignError, match="utilization telemetry"):
        coordinator.run_to_completion(resume=False)


def test_verify_refuses_missing_or_tampered_execution_receipt(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    coordinator = CampaignCoordinator(
        manifest, tmp_path / "state", transports=_transports(manifest),
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


def test_cli_plan_is_deterministic_and_live_run_is_fail_closed(
    tmp_path: Path,
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

    with pytest.raises(RTX4090TwoHostCampaignError, match="no telemetry-capable"):
        main([
            "run", "--manifest", str(manifest_path),
            "--state-dir", str(tmp_path / "live-state"),
        ])
