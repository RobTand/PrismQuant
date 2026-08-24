from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from prismaquant.cluster_live_transport import (
    BOOTSTRAP_RECEIPT_SCHEMA,
    ClusterLiveTransportError,
    REMOTE_OP_RESPONSE_SCHEMA,
    TelemetryJobAdapter,
    VerifiedRsyncSSHTransfer,
    _BOOTSTRAP_PROGRAM,
    _REMOTE_FILE_OP_PROGRAM,
    bootstrap_ssh_helper,
)
from prismaquant.cluster_transport import (
    GpuSample,
    JobReceipt,
    LocalTransport,
    RunRequest,
    SSHTransport,
    TelemetrySnapshot,
    build_tree_manifest,
    canonical_json_bytes,
    verify_tree_manifest,
)


def completed(returncode=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def gpu_sample(utilization: float) -> GpuSample:
    return GpuSample(
        timestamp="2026/08/24 12:00:00.000",
        index=0,
        name="NVIDIA GPU",
        uuid="GPU-1",
        pci_bus_id="0000:01:00.0",
        gpu_utilization_pct=utilization,
        memory_utilization_pct=5.0,
        memory_used_mib=1.0,
        memory_total_mib=2.0,
        temperature_c=50.0,
        power_w=100.0,
    )


def receipt(request: RunRequest, state: str) -> JobReceipt:
    return JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state=state,
        started_ns=10,
        finished_ns=20 if state != "running" else None,
        returncode=0 if state == "succeeded" else None,
        pid=123 if state == "running" else None,
        stdout=b"",
        stderr=b"",
        transport="fake",
    )


class FakeClock:
    def __init__(self):
        self.value = 100.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.value

    def sleep(self, duration: float):
        self.sleeps.append(duration)
        self.value += duration


class FakeMonitoredTransport:
    def __init__(self, request, statuses, samples):
        self.request = request
        self.statuses = list(statuses)
        self.samples = list(samples)
        self.calls: list[str] = []

    def start(self, request):
        self.calls.append("start")
        assert request == self.request
        return receipt(request, "running")

    def status(self, job_id):
        self.calls.append("status")
        assert job_id == self.request.job_id
        return self.statuses.pop(0)

    def sample_telemetry(self):
        self.calls.append("sample")
        value = self.samples.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_telemetry_adapter_samples_immediately_then_at_injected_cadence():
    request = RunRequest("monitored", ("/bin/work",))
    snapshots = [
        TelemetrySnapshot(1000, 100, (gpu_sample(10),)),
        TelemetrySnapshot(2000, 90, (gpu_sample(20),)),
    ]
    backend = FakeMonitoredTransport(
        request,
        [receipt(request, "running"), receipt(request, "succeeded")],
        snapshots,
    )
    clock = FakeClock()
    adapter = TelemetryJobAdapter(
        backend,
        cadence_seconds=2.5,
        clock=clock,
        sleep=clock.sleep,
    )

    final, observed = adapter.run_with_telemetry(request)

    assert final.state == "succeeded"
    assert observed == tuple(snapshots)
    assert backend.calls == ["start", "sample", "status", "sample", "status"]
    assert clock.sleeps == [2.5]


def test_telemetry_adapter_exposes_durable_primitives_for_attempt_adoption():
    request = RunRequest("adopt", ("/bin/work",))
    snapshot = TelemetrySnapshot(1000, 100, (gpu_sample(10),))
    backend = FakeMonitoredTransport(
        request,
        [receipt(request, "succeeded")],
        [snapshot],
    )
    adapter = TelemetryJobAdapter(backend, cadence_seconds=3.0)

    assert adapter.start(request).state == "running"
    assert adapter.status(request.job_id).state == "succeeded"
    assert adapter.sample_telemetry() == snapshot
    assert adapter.cadence_seconds == 3.0


def test_telemetry_adapter_fails_closed_without_a_live_sample():
    request = RunRequest("instant", ("/bin/work",))

    class InstantTransport:
        def start(self, request):
            return receipt(request, "succeeded")

        def status(self, job_id):
            raise AssertionError("status must not be called")

        def sample_telemetry(self):
            raise AssertionError("post-hoc sample must not be used")

    with pytest.raises(ClusterLiveTransportError, match="before any live telemetry"):
        TelemetryJobAdapter(InstantTransport()).run_with_telemetry(request)

    final, samples = TelemetryJobAdapter(
        InstantTransport(), require_samples=False
    ).run_with_telemetry(request)
    assert final.state == "succeeded"
    assert samples == ()


def test_telemetry_adapter_fails_on_sampling_error_and_nonmonotonic_samples():
    request = RunRequest("sample-error", ("/bin/work",))
    failed_backend = FakeMonitoredTransport(
        request,
        [receipt(request, "succeeded")],
        [RuntimeError("nvidia-smi failed")],
    )
    with pytest.raises(ClusterLiveTransportError, match="sampling failed"):
        TelemetryJobAdapter(failed_backend).run_with_telemetry(request)

    snapshots = [
        TelemetrySnapshot(1000, 100, (gpu_sample(1),)),
        TelemetrySnapshot(1000, 90, (gpu_sample(2),)),
    ]
    backend = FakeMonitoredTransport(
        request,
        [receipt(request, "running"), receipt(request, "succeeded")],
        snapshots,
    )
    clock = FakeClock()
    with pytest.raises(ClusterLiveTransportError, match="strictly increasing"):
        TelemetryJobAdapter(
            backend, clock=clock, sleep=clock.sleep
        ).run_with_telemetry(request)


def test_telemetry_adapter_monitor_timeout_is_clock_driven():
    request = RunRequest("timeout", ("/bin/work",))
    snapshots = [
        TelemetrySnapshot(1000, 100, (gpu_sample(1),)),
        TelemetrySnapshot(2000, 100, (gpu_sample(1),)),
    ]
    backend = FakeMonitoredTransport(
        request,
        [receipt(request, "running"), receipt(request, "running")],
        snapshots,
    )
    clock = FakeClock()
    with pytest.raises(ClusterLiveTransportError, match="monitor timeout"):
        TelemetryJobAdapter(
            backend,
            cadence_seconds=2.0,
            monitor_timeout_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
        ).run_with_telemetry(request)


class RecordingBootstrapRunner:
    def __init__(self):
        self.calls = []
        self.request = None

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        raw = base64.b64decode(kwargs["input"].strip(), validate=True)
        self.request = json.loads(raw)
        response = {
            "schema": BOOTSTRAP_RECEIPT_SCHEMA,
            "remote_path": self.request["remote_path"],
            "source_sha256": self.request["source_sha256"],
            "size_bytes": self.request["size_bytes"],
            "already_present": False,
        }
        return completed(stdout=canonical_json_bytes(response))


def test_helper_bootstrap_uses_fixed_python_and_base64_stdin_only():
    runner = RecordingBootstrapRunner()
    ssh = SSHTransport(
        "rtx4090",
        remote_helper_path="/opt/prismaquant/cluster_transport.py",
        run_impl=runner,
    )

    bootstrap = bootstrap_ssh_helper(ssh)

    assert bootstrap.remote_path == "/opt/prismaquant/cluster_transport.py"
    assert runner.request["source_sha256"] == bootstrap.source_sha256
    assert len(base64.b64decode(runner.request["source_b64"])) == bootstrap.size_bytes
    argv, kwargs = runner.calls[0]
    command = " ".join(argv)
    assert kwargs["shell"] is False
    assert kwargs["input"].strip() == base64.b64encode(
        canonical_json_bytes(runner.request)
    )
    assert "/opt/prismaquant/cluster_transport.py" not in command
    assert bootstrap.source_sha256 not in command
    assert argv[-1].startswith("exec /usr/bin/python3 -P -B -s -c ")


def test_fixed_remote_programs_are_syntax_valid_and_bootstrap_fails_closed():
    compile(_BOOTSTRAP_PROGRAM, "<bootstrap>", "exec")
    compile(_REMOTE_FILE_OP_PROGRAM, "<remote-file-op>", "exec")
    ssh = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=lambda *args, **kwargs: completed(
            returncode=2, stderr=b"FileExistsError: different bytes"
        ),
    )
    with pytest.raises(ClusterLiveTransportError, match="different bytes"):
        bootstrap_ssh_helper(ssh)


class FakeRemoteOperations:
    def __init__(self, *, manifest=None):
        self.manifest = manifest
        self.calls = []

    def __call__(self, argv, **kwargs):
        raw = base64.b64decode(kwargs["input"].strip(), validate=True)
        request = json.loads(raw)
        self.calls.append((list(argv), dict(kwargs), request))
        action = request["action"]
        payload = request["payload"]
        if action == "manifest":
            manifest = self.manifest
            result = {
                "manifest": manifest.to_payload(),
                "manifest_sha256": manifest.identity_sha256,
            }
        elif action == "prepare_upload":
            digest = payload["manifest_sha256"]
            result = {
                "stage_path": f"{payload['remote_stage_root']}/{digest}/payload",
                "reused": False,
            }
        elif action == "publish_upload":
            result = {
                "destination": payload["destination"],
                "already_present": False,
                "manifest": payload["manifest"],
                "manifest_sha256": payload["manifest_sha256"],
            }
        else:
            raise AssertionError(action)
        response = {
            "schema": REMOTE_OP_RESPONSE_SCHEMA,
            "kind": action,
            "payload": result,
        }
        return completed(stdout=canonical_json_bytes(response))


class RecordingRsync:
    def __init__(self, callback=None, *, returncode=0):
        self.callback = callback
        self.returncode = returncode
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.callback is not None:
            self.callback(list(argv))
        return completed(
            returncode=self.returncode,
            stderr=b"rsync failed" if self.returncode else b"",
        )


def test_upload_uses_content_stage_rsync_and_remote_destination_manifest(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data").write_bytes(b"payload")
    (source / "empty").mkdir()
    expected = build_tree_manifest(source)
    remote = FakeRemoteOperations()
    rsync = RecordingRsync()
    ssh = SSHTransport(
        "rtx4090",
        remote_helper_path="/opt/pq/cluster_transport.py",
        run_impl=remote,
    )
    transfer = VerifiedRsyncSSHTransfer(
        ssh,
        remote_stage_root="/srv/pq/stage",
        local_stage_root=tmp_path / "local-stage",
        rsync_run_impl=rsync,
    )

    result = transfer.upload(source, "/srv/pq/artifacts/model")

    assert result.direction == "upload"
    assert result.manifest_sha256 == expected.identity_sha256
    assert result.content_stage == (
        f"/srv/pq/stage/{expected.identity_sha256}/payload"
    )
    assert [call[2]["action"] for call in remote.calls] == [
        "prepare_upload",
        "publish_upload",
    ]
    rsync_argv, rsync_kwargs = rsync.calls[0]
    assert rsync_kwargs["shell"] is False
    assert "--protect-args" in rsync_argv
    assert rsync_argv[-2] == str(source) + "/"
    assert rsync_argv[-1] == (
        f"rtx4090:/srv/pq/stage/{expected.identity_sha256}/payload/"
    )
    for argv, kwargs, request in remote.calls:
        command = " ".join(argv)
        assert kwargs["shell"] is False
        assert "/srv/pq/artifacts/model" not in command
        assert request == json.loads(base64.b64decode(kwargs["input"].strip()))


def test_download_atomically_publishes_stage_without_second_full_copy(
    tmp_path, monkeypatch
):
    pretend_remote = tmp_path / "pretend-remote"
    (pretend_remote / "nested").mkdir(parents=True)
    (pretend_remote / "nested/data").write_bytes(b"download")
    (pretend_remote / "empty").mkdir()
    expected = build_tree_manifest(pretend_remote)
    remote = FakeRemoteOperations(manifest=expected)

    def emulate_download(argv):
        destination = Path(argv[-1].rstrip("/"))
        shutil.copytree(pretend_remote, destination, dirs_exist_ok=True)

    rsync = RecordingRsync(emulate_download)
    ssh = SSHTransport(
        "sparky",
        remote_helper_path="/opt/pq/cluster_transport.py",
        run_impl=remote,
    )
    local_stage = tmp_path / "content-stage"
    destination = tmp_path / "published"
    transfer = VerifiedRsyncSSHTransfer(
        ssh,
        remote_stage_root="/srv/pq/stage",
        local_stage_root=local_stage,
        rsync_run_impl=rsync,
    )

    def forbidden_second_copy(*args, **kwargs):
        raise AssertionError("verified download must rename, not duplicate, its stage")

    monkeypatch.setattr(LocalTransport, "copy_verified", forbidden_second_copy)

    result = transfer.download("/srv/pq/source", destination)

    assert result.direction == "download"
    assert result.manifest_sha256 == expected.identity_sha256
    verify_tree_manifest(destination, expected)
    assert not (local_stage / expected.identity_sha256).exists()
    argv, kwargs = rsync.calls[0]
    assert kwargs["shell"] is False
    assert argv[-2] == "sparky:/srv/pq/source/"
    assert expected.identity_sha256 in argv[-1]
    assert transfer.inspect_artifact("/srv/pq/source") == expected
    before = (destination / "nested/data").read_bytes()
    reused = transfer.download("/srv/pq/source", destination)
    assert reused.already_present is True
    assert len(rsync.calls) == 1
    assert (destination / "nested/data").read_bytes() == before
    (destination / "nested/data").write_bytes(b"different")
    with pytest.raises(ClusterLiveTransportError, match="exists but differs"):
        transfer.download("/srv/pq/source", destination)
    assert (destination / "nested/data").read_bytes() == b"different"
    assert len(rsync.calls) == 1


def test_upload_rsync_failure_never_requests_remote_publish(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    remote = FakeRemoteOperations()
    rsync = RecordingRsync(returncode=23)
    ssh = SSHTransport(
        "host",
        remote_helper_path="/opt/pq/cluster_transport.py",
        run_impl=remote,
    )
    transfer = VerifiedRsyncSSHTransfer(
        ssh,
        remote_stage_root="/srv/pq/stage",
        local_stage_root=tmp_path / "stage",
        rsync_run_impl=rsync,
    )

    with pytest.raises(ClusterLiveTransportError, match="rsync exited 23"):
        transfer.upload(source, "/srv/pq/output")
    assert [call[2]["action"] for call in remote.calls] == ["prepare_upload"]
