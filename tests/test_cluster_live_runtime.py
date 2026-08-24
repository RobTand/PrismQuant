from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import prismaquant.cluster_live_runtime as live_runtime
from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    seal_campaign_manifest,
)
from prismaquant.cluster_live_runtime import (
    ARTIFACT_BARRIER_RECEIPT_SCHEMA,
    ClusterLiveRuntimeError,
    DirectorySkeletonReceipt,
    LiveCampaignRuntime,
    LocalArtifactInspector,
    _ManifestSSHTransport,
    _create_local_directory_skeleton,
    _initialize_remote_directory_skeleton,
    build_live_campaign_runtime,
)
from prismaquant.cluster_live_transport import (
    HelperBootstrapReceipt,
    RsyncTransferReceipt,
)
from prismaquant.cluster_transport import (
    LocalTransport,
    SSHTransport,
    build_tree_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    verify_tree_manifest,
)


def _host(
    root: Path,
    host_id: str,
    *,
    local: bool,
) -> dict[str, object]:
    transport: dict[str, object]
    if local:
        transport = {"kind": "local"}
    else:
        transport = {
            "kind": "ssh",
            "host": "sparklina.example.test",
            "port": 2222,
            "user": "campaign_runner",
        }
    producer_commit = "1" * 40
    image_digest = "sha256:" + "4" * 64
    return {
        "id": host_id,
        "transport": transport,
        "roots": {
            "model_root": str(root / "model"),
            "dataset_path": str(root / "dataset" / "calibration.pt"),
            "snapshot_root": str(root / "snapshot"),
            "run_root": str(root / "run"),
            "worker_state_root": str(root / "worker-state"),
        },
        "expected": {
            "hostname": f"host-{host_id}",
            "gpu": {
                "name": "NVIDIA GB10",
                "uuid": f"GPU-{host_id.upper()}-0123456789",
                "compute_capability": [12, 1],
                "device_count": 1,
            },
            "image_digest": image_digest,
            "producer_commit": producer_commit,
            "uid": 1000,
            "gid": 1000,
        },
    }


def _manifest(tmp_path: Path, *, coordinator: str = "zeta") -> dict[str, object]:
    return seal_campaign_manifest(
        {
            "schema": CAMPAIGN_MANIFEST_SCHEMA,
            "campaign_id": "qwen38-27b-two-host-live-test",
            "coordinator": coordinator,
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
            # Reversed deliberately; the sealed contract canonicalizes worker
            # indices independently of control-process locality.
            "hosts": [
                _host(tmp_path / "remote", "zeta", local=False),
                _host(tmp_path / "local", "alpha", local=True),
            ],
        }
    )


class FakeVerifiedTransfer:
    def __init__(
        self,
        ssh,
        *,
        remote_stage_root,
        local_stage_root,
        **kwargs,
    ):
        self.ssh = ssh
        self.remote_stage_root = str(remote_stage_root)
        self.local_stage_root = Path(local_stage_root)
        self.calls: list[tuple[str, str, str, bool]] = []
        self._completed_ns = 100

    def inspect_artifact(self, absolute_host_path: str):
        return build_tree_manifest(Path(absolute_host_path))

    def _copy(self, direction: str, source: str, destination: str):
        source_path = Path(source)
        destination_path = Path(destination)
        manifest = build_tree_manifest(source_path)
        already_present = False
        if destination_path.exists() or destination_path.is_symlink():
            verify_tree_manifest(destination_path, manifest)
            already_present = True
        elif manifest.root_kind == "directory":
            shutil.copytree(source_path, destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)
        verify_tree_manifest(destination_path, manifest)
        self._completed_ns += 1
        self.calls.append((direction, source, destination, already_present))
        return RsyncTransferReceipt(
            direction=direction,
            source=source,
            destination=destination,
            manifest_sha256=manifest.identity_sha256,
            total_bytes=manifest.total_bytes,
            entry_count=len(manifest.entries),
            content_stage=str(
                self.local_stage_root / manifest.identity_sha256 / "payload"
            ),
            already_present=already_present,
            completed_ns=self._completed_ns,
        )

    def upload(self, source: str, remote_destination: str):
        return self._copy("upload", str(source), remote_destination)

    def download(self, remote_source: str, destination: str):
        return self._copy("download", remote_source, str(destination))


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def initialize_remote(ssh, receipt, *, run_impl):
        return _create_local_directory_skeleton(receipt)

    def bootstrap(ssh, *, run_impl):
        spec = ssh.helper_install_spec()
        return HelperBootstrapReceipt(
            remote_path=spec.remote_path,
            source_sha256=spec.source_sha256,
            size_bytes=spec.size_bytes,
            already_present=False,
        )

    monkeypatch.setattr(
        live_runtime,
        "_initialize_remote_directory_skeleton",
        initialize_remote,
    )
    monkeypatch.setattr(live_runtime, "bootstrap_ssh_helper", bootstrap)
    monkeypatch.setattr(
        live_runtime,
        "VerifiedRsyncSSHTransfer",
        FakeVerifiedTransfer,
    )


def _build_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    coordinator: str = "zeta",
) -> LiveCampaignRuntime:
    _install_fakes(monkeypatch)
    return build_live_campaign_runtime(
        _manifest(tmp_path, coordinator=coordinator),
        local_run_impl=lambda *args, **kwargs: None,
        local_popen_impl=lambda *args, **kwargs: None,
        ssh_run_impl=lambda *args, **kwargs: None,
        rsync_run_impl=lambda *args, **kwargs: None,
    )


def test_factory_builds_one_transport_per_manifest_host_and_fixed_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    remote_initializations = []
    bootstraps = []

    def initialize_remote(ssh, receipt, *, run_impl):
        remote_initializations.append((ssh, receipt, run_impl))
        return receipt

    def bootstrap(ssh, *, run_impl):
        bootstraps.append((ssh, run_impl))
        spec = ssh.helper_install_spec()
        return HelperBootstrapReceipt(
            spec.remote_path,
            spec.source_sha256,
            spec.size_bytes,
            False,
        )

    monkeypatch.setattr(
        live_runtime,
        "_initialize_remote_directory_skeleton",
        initialize_remote,
    )
    monkeypatch.setattr(live_runtime, "bootstrap_ssh_helper", bootstrap)

    fake_run = lambda *args, **kwargs: None
    runtime = build_live_campaign_runtime(
        manifest,
        local_run_impl=fake_run,
        local_popen_impl=fake_run,
        ssh_run_impl=fake_run,
        rsync_run_impl=fake_run,
    )

    assert type(runtime.local_transport) is LocalTransport
    assert isinstance(runtime.ssh_transport, SSHTransport)
    assert list(runtime.transports) == ["alpha", "zeta"]
    assert runtime.transports["alpha"].transport is runtime.local_transport
    assert runtime.transports["zeta"].transport is runtime.ssh_transport
    assert runtime.transports["alpha"].cadence_seconds == 1.0
    assert runtime.transports["zeta"].cadence_seconds == 1.0
    assert runtime.transports["alpha"].require_samples is False
    assert len(remote_initializations) == 1
    assert len(bootstraps) == 1
    assert remote_initializations[0][0] is runtime.ssh_transport
    assert bootstraps[0][0] is runtime.ssh_transport
    assert runtime.ssh_transport.host == "campaign_runner@sparklina.example.test"
    assert runtime.ssh_transport.command_argv[
        runtime.ssh_transport.command_argv.index("--") - 2 :
        runtime.ssh_transport.command_argv.index("--")
    ] == ("-p", "2222")
    assert isinstance(runtime.artifact_inspectors["alpha"], LocalArtifactInspector)
    assert runtime.artifact_inspectors["zeta"] is runtime.transfer
    assert runtime.ssh_transport.remote_helper_path.startswith(
        str(tmp_path / "remote" / "worker-state" / "cluster-runtime")
    )

    local_receipt = runtime.directory_receipts["alpha"]
    for directory in local_receipt.directories:
        assert Path(directory).is_dir()
        assert not Path(directory).is_symlink()
    assert str(tmp_path / "local" / "snapshot") not in local_receipt.directories
    assert str(tmp_path / "local" / "snapshot").rsplit("/", 1)[0] in (
        local_receipt.directories
    )


def test_remote_directory_request_uses_stdin_not_manifest_shell_fragments() -> None:
    ssh = _ManifestSSHTransport(
        "campaign_runner@sparklina.example.test",
        port=2207,
        remote_helper_path="/opt/prismaquant/helper/cluster_transport.py",
    )
    receipt = DirectorySkeletonReceipt(
        campaign_identity_sha256="a" * 64,
        host_id="zeta",
        directories=("/srv/secret-campaign/run", "/srv/secret-campaign/state"),
    )
    observed = {}

    def run(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        request = json.loads(
            base64.b64decode(kwargs["input"].strip(), validate=True)
        )
        response = {
            "schema": live_runtime.REMOTE_DIRECTORY_RESPONSE_SCHEMA,
            "campaign_identity_sha256": request["campaign_identity_sha256"],
            "host_id": request["host_id"],
            "directories": request["directories"],
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=canonical_json_bytes(response),
            stderr=b"",
        )

    assert _initialize_remote_directory_skeleton(ssh, receipt, run_impl=run) == receipt
    command = " ".join(observed["argv"])
    assert "/srv/secret-campaign" not in command
    assert "a" * 64 not in command
    assert observed["shell"] is False
    assert observed["check"] is False
    assert observed["argv"][observed["argv"].index("--") - 2 : observed["argv"].index("--")] == [
        "-p",
        "2207",
    ]


def test_directory_skeleton_refuses_symlink_and_non_directory_components(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    linked_receipt = DirectorySkeletonReceipt(
        campaign_identity_sha256="a" * 64,
        host_id="alpha",
        directories=(str(link / "child"),),
    )
    with pytest.raises(ClusterLiveRuntimeError, match="real directory"):
        _create_local_directory_skeleton(linked_receipt)

    regular = tmp_path / "regular"
    regular.write_text("not a directory")
    file_receipt = DirectorySkeletonReceipt(
        campaign_identity_sha256="a" * 64,
        host_id="alpha",
        directories=(str(regular / "child"),),
    )
    with pytest.raises(ClusterLiveRuntimeError, match="real directory"):
        _create_local_directory_skeleton(file_receipt)


_DIRECTORY_ROUTES = frozenset(
    {"immutable_snapshot", "worker_0_act", "worker_1_act", "merged_bundle", "burn_plan"}
)


def _write_route_source(route) -> None:
    source = Path(route.source_path)
    if source.exists():
        return
    if route.name in _DIRECTORY_ROUTES:
        source.mkdir(parents=True)
        (source / "payload.bin").write_bytes((route.name + "\n").encode())
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((route.name + "\n").encode())


def test_remote_coordinator_stage_barriers_route_all_artifacts_and_reuse_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_fake_runtime(tmp_path, monkeypatch, coordinator="zeta")
    stages = (
        "host_preflight",
        "sample_ce",
        "merge_importance",
        "sample_fisher",
        "merge_sample_probe",
        "measure_burn",
        "merge_burn",
    )
    expected_counts = (1, 4, 1, 1, 2, 4, 1)
    for stage in stages:
        for route in runtime.route_specs_for_stage(stage):
            _write_route_source(route)

    receipts = []
    for stage, count in zip(stages, expected_counts, strict=True):
        receipt = runtime.synchronize_before_stage(stage)
        assert receipt is not None
        assert len(receipt.transfers) == count
        payload = receipt.to_payload()
        assert payload["schema"] == ARTIFACT_BARRIER_RECEIPT_SCHEMA
        body = {key: value for key, value in payload.items() if key != "identity_sha256"}
        assert payload["identity_sha256"] == canonical_json_sha256(body)
        receipts.append(receipt)
        for item in receipt.transfers:
            assert build_tree_manifest(Path(item.route.source_path)) == build_tree_manifest(
                Path(item.route.destination_path)
            )

    assert len(runtime.transfer.calls) == sum(expected_counts)
    assert [item.transfer.direction for item in receipts[0].transfers] == ["upload"]
    assert {item.transfer.direction for item in receipts[1].transfers} == {"download"}
    assert [item.transfer.direction for item in receipts[2].transfers] == ["upload"]
    assert [item.transfer.direction for item in receipts[3].transfers] == ["download"]
    assert {item.transfer.direction for item in receipts[4].transfers} == {"upload"}
    assert {item.transfer.direction for item in receipts[5].transfers} == {"download"}
    assert [item.transfer.direction for item in receipts[6].transfers] == ["upload"]

    reused = runtime.synchronize_before_stage("sample_ce")
    assert reused is not None
    assert all(item.transfer.already_present for item in reused.transfers)
    assert runtime.synchronize_before_stage("prepare_calibration") is None
    with pytest.raises(ClusterLiveRuntimeError, match="unknown campaign stage"):
        runtime.synchronize_before_stage("invented")


def test_route_catalog_reverses_directions_when_coordinator_is_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_fake_runtime(tmp_path, monkeypatch, coordinator="alpha")

    assert runtime.route_specs_for_stage("host_preflight")[0].source_host_id == "alpha"
    assert {
        (route.source_host_id, route.destination_host_id)
        for route in runtime.route_specs_for_stage("sample_ce")
    } == {("alpha", "zeta")}
    for stage in ("merge_importance", "merge_sample_probe", "merge_burn"):
        assert {
            (route.source_host_id, route.destination_host_id)
            for route in runtime.route_specs_for_stage(stage)
        } == {("zeta", "alpha")}
    for stage in ("sample_fisher", "measure_burn"):
        assert {
            (route.source_host_id, route.destination_host_id)
            for route in runtime.route_specs_for_stage(stage)
        } == {("alpha", "zeta")}
    assert runtime.route_specs_for_stage("merge_importance")[0].name == "worker_1_ce"


def test_barrier_fails_closed_on_divergent_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_fake_runtime(tmp_path, monkeypatch, coordinator="zeta")
    route = runtime.route_specs_for_stage("host_preflight")[0]
    _write_route_source(route)
    destination = Path(route.destination_path)
    destination.mkdir(parents=True)
    (destination / "payload.bin").write_bytes(b"different\n")

    with pytest.raises(ClusterLiveRuntimeError, match="artifact transfer"):
        runtime.synchronize_before_stage("host_preflight")
    assert (destination / "payload.bin").read_bytes() == b"different\n"


def test_barrier_rejects_receipt_that_does_not_bind_exact_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadReceiptTransfer(FakeVerifiedTransfer):
        def upload(self, source: str, remote_destination: str):
            receipt = super().upload(source, remote_destination)
            return replace(receipt, destination=remote_destination + ".wrong")

    _install_fakes(monkeypatch)
    monkeypatch.setattr(
        live_runtime,
        "VerifiedRsyncSSHTransfer",
        BadReceiptTransfer,
    )
    runtime = build_live_campaign_runtime(_manifest(tmp_path))
    route = runtime.route_specs_for_stage("host_preflight")[0]
    _write_route_source(route)

    with pytest.raises(ClusterLiveRuntimeError, match="receipt differs"):
        runtime.synchronize_before_stage("host_preflight")

