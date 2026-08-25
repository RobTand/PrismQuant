from __future__ import annotations

import base64
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

import pytest

import prismaquant.cluster_live_runtime as live_runtime
from prismaquant.cluster_campaign_contract import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CANONICAL_CONTAINER_PATHS,
    STAGE_DAG,
    bind_gridbook_runtime_contract,
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
from prismaquant.rtx4090_two_host_campaign import build_command_plan


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
    monkeypatch.setattr(
        live_runtime,
        "_verify_endpoint_identities",
        lambda local, remote, _ssh, **_kwargs: {
            str(local["id"]): {"fixture": "local-endpoint"},
            str(remote["id"]): {"fixture": "remote-endpoint"},
        },
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
    monkeypatch.setattr(
        live_runtime,
        "_verify_endpoint_identities",
        lambda local, remote, _ssh, **_kwargs: {
            str(local["id"]): {"fixture": "local-endpoint"},
            str(remote["id"]): {"fixture": "remote-endpoint"},
        },
    )

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
        str(tmp_path / "remote" / ".prismaquant-cluster-control")
    )
    assert runtime.ssh_transport.remote_state_root == str(
        tmp_path / "remote" / ".prismaquant-cluster-control"
        / manifest["identity_sha256"] / "transport"
    )
    assert runtime.ssh_transport.remote_state_root in runtime.ssh_transport.command_argv[-1]
    assert not Path(runtime.ssh_transport.remote_state_root).is_relative_to(
        tmp_path / "remote" / "worker-state"
    )

    local_receipt = runtime.directory_receipts["alpha"]
    for directory in local_receipt.directories:
        assert Path(directory).is_dir()
        assert not Path(directory).is_symlink()
    assert str(tmp_path / "local" / "snapshot") not in local_receipt.directories
    assert str(tmp_path / "local" / "snapshot").rsplit("/", 1)[0] in (
        local_receipt.directories
    )
    assert all(
        not directory.startswith(str(tmp_path / "local" / "worker-state") + "/")
        or "/cluster-runtime/" not in directory
        for directory in local_receipt.directories
    )


def test_read_only_open_does_not_create_or_bootstrap_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_runtime,
        "_initialize_remote_directory_skeleton",
        lambda *_args, **_kwargs: pytest.fail("read-only open initialized remote"),
    )
    monkeypatch.setattr(
        live_runtime,
        "bootstrap_ssh_helper",
        lambda *_args, **_kwargs: pytest.fail("read-only open bootstrapped helper"),
    )
    runtime = build_live_campaign_runtime(
        _manifest(tmp_path), initialize=False,
    )

    assert runtime.helper_bootstrap_receipt is None
    assert not (tmp_path / "local" / "run").exists()
    assert not (tmp_path / "remote" / "run").exists()


def test_endpoint_identity_is_checked_before_any_runtime_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        live_runtime,
        "_create_local_directory_skeleton",
        lambda *_args, **_kwargs: pytest.fail("local directories mutated"),
    )
    monkeypatch.setattr(
        live_runtime,
        "_initialize_remote_directory_skeleton",
        lambda *_args, **_kwargs: pytest.fail("remote directories mutated"),
    )
    monkeypatch.setattr(
        live_runtime,
        "bootstrap_ssh_helper",
        lambda *_args, **_kwargs: pytest.fail("remote helper mutated"),
    )

    def reject(*_args, **_kwargs):
        raise ClusterLiveRuntimeError("read-only endpoint identity differs")

    with pytest.raises(ClusterLiveRuntimeError, match="identity differs"):
        build_live_campaign_runtime(
            manifest,
            endpoint_verifier=reject,
        )

    assert not (tmp_path / "local" / "run").exists()
    assert not (tmp_path / "remote" / "run").exists()


@pytest.mark.parametrize(
    ("field", "colliding_name", "message"),
    (
        (
            "worker_state_root",
            ".prismaquant-cluster-control",
            "runtime control root overlaps declared worker_state_root",
        ),
        (
            "run_root",
            ".prismaquant-transfer-stage",
            "run transfer root overlaps declared run_root",
        ),
    ),
)
def test_derived_control_roots_cannot_overlap_declared_writable_mounts(
    tmp_path: Path,
    field: str,
    colliding_name: str,
    message: str,
) -> None:
    raw = _manifest(tmp_path)
    raw.pop("identity_sha256")
    local = next(
        host for host in raw["hosts"]
        if host["transport"]["kind"] == "local"
    )
    local["roots"][field] = str(tmp_path / "local" / colliding_name)
    manifest = seal_campaign_manifest(raw)
    endpoint_called = False

    def endpoint(*_args, **_kwargs):
        nonlocal endpoint_called
        endpoint_called = True
        return {}

    with pytest.raises(ClusterLiveRuntimeError, match=message):
        build_live_campaign_runtime(
            manifest,
            endpoint_verifier=endpoint,
        )

    assert endpoint_called is False
    assert not (tmp_path / "local" / colliding_name).exists()


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
        assert runtime.validate_barrier_receipt(
            stage, receipt.to_payload(),
        ) == receipt
        for item in receipt.transfers:
            assert build_tree_manifest(Path(item.route.source_path)) == build_tree_manifest(
                Path(item.route.destination_path)
            )

    assert len(runtime.snapshot_transfer.calls) == 1
    assert len(runtime.transfer.calls) == sum(expected_counts) - 1
    assert runtime.snapshot_transfer.remote_stage_root.startswith(
        str(tmp_path / "remote")
    )
    assert runtime.transfer.remote_stage_root.startswith(
        str(tmp_path / "remote" / ".prismaquant-transfer-stage")
    )
    assert not runtime.transfer.remote_stage_root.startswith(
        str(tmp_path / "remote" / "run") + "/"
    )
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


@pytest.mark.parametrize("coordinator", ["alpha", "zeta"])
def test_every_fixed_command_input_is_local_or_arrives_at_a_prior_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordinator: str,
) -> None:
    runtime = _build_fake_runtime(
        tmp_path, monkeypatch, coordinator=coordinator,
    )
    plan = build_command_plan(runtime.manifest)
    commands_by_stage: dict[str, list[dict[str, object]]] = {}
    for command in plan["commands"]:
        assert isinstance(command, dict)
        commands_by_stage.setdefault(str(command["stage"]), []).append(command)

    hosts = {
        str(host["id"]): host for host in runtime.manifest["hosts"]
    }
    available: dict[str, dict[str, str]] = {}
    for host_id, host in hosts.items():
        kind = str(host["transport"]["kind"])
        available[host_id] = {
            "/model": "directory",
            "/dataset": "file",
        }
        if kind == "local":
            available[host_id]["/pq"] = "directory"

    def container_path(host_id: str, absolute_host_path: str) -> str:
        host = hosts[host_id]
        roots = host["roots"]
        for field, canonical_root in CANONICAL_CONTAINER_PATHS.items():
            root = PurePosixPath(str(roots[field]))
            candidate = PurePosixPath(absolute_host_path)
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if relative == PurePosixPath("."):
                return canonical_root
            return (PurePosixPath(canonical_root) / relative).as_posix()
        raise AssertionError(
            f"route path {absolute_host_path!r} is outside host {host_id!r} roots"
        )

    def is_available(host_id: str, required: str) -> bool:
        required_path = PurePosixPath(required)
        for present, kind in available[host_id].items():
            present_path = PurePosixPath(present)
            if required_path == present_path:
                return True
            if kind == "directory":
                try:
                    required_path.relative_to(present_path)
                except ValueError:
                    continue
                return True
        return False

    assert runtime.barrier_stages == tuple(
        spec.stage for spec in STAGE_DAG
        if runtime.route_specs_for_stage(spec.stage)
    )
    for spec in STAGE_DAG:
        for route in runtime.route_specs_for_stage(spec.stage):
            source = container_path(route.source_host_id, route.source_path)
            assert is_available(route.source_host_id, source) or any(
                PurePosixPath(present).is_relative_to(PurePosixPath(source))
                for present in available[route.source_host_id]
            ), f"route source is not produced before {spec.stage}: {route.name}"
            destination = container_path(
                route.destination_host_id, route.destination_path,
            )
            available[route.destination_host_id][destination] = (
                "directory" if route.name in _DIRECTORY_ROUTES else "file"
            )

        commands = commands_by_stage[spec.stage]
        for command in commands:
            host_id = str(command["host_id"])
            for required in command["inputs"]:
                assert is_available(host_id, str(required)), (
                    f"{command['work_id']} lacks declared input {required}"
                )
        for command in commands:
            host_id = str(command["host_id"])
            for output in command["outputs"]:
                path = str(output)
                available[host_id][path] = (
                    "directory"
                    if PurePosixPath(path).name in {"act", "activation_cache"}
                    else "file"
                )


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
