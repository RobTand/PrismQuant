from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from prismaquant.cluster_transport import (
    ClusterTransportError,
    GpuSample,
    HELPER_RESPONSE_SCHEMA,
    JobConflictError,
    JobReceipt,
    LocalTransport,
    ManifestEntry,
    ManifestError,
    RunRequest,
    SSHTransport,
    TelemetrySnapshot,
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
    parse_mem_available,
    parse_nvidia_smi_csv,
    summarize_utilization,
    verify_tree_manifest,
)


class RecordingRunner:
    def __init__(self, result=None, *, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result(argv, **kwargs)
        return self.result


def completed(returncode=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def helper_job_response(receipt: JobReceipt) -> bytes:
    return canonical_json_bytes(
        {
            "schema": HELPER_RESPONSE_SCHEMA,
            "kind": "job_receipt",
            "payload": receipt.to_payload(),
        }
    )


def test_run_request_identity_is_order_independent_and_round_trips():
    left = RunRequest(
        "probe-1",
        ("/bin/tool", "--literal", "$(not-a-shell)"),
        cwd="/campaign/work",
        env={"ZED": "2", "ALPHA": "1"},
        timeout_seconds=12,
        stdin=b"binary\x00input",
        inherit_env=False,
    )
    right = RunRequest(
        "probe-1",
        left.argv,
        cwd=left.cwd,
        env=(("ALPHA", "1"), ("ZED", "2")),
        timeout_seconds=12.0,
        stdin=left.stdin,
        inherit_env=False,
    )

    assert left == right
    assert left.request_sha256 == right.request_sha256
    assert left.as_dict() == left.to_payload()
    assert RunRequest.from_payload(left.to_payload()) == left
    assert canonical_json_bytes(left.to_payload()) == canonical_json_bytes(
        right.to_payload()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"job_id": "../escape", "argv": ("true",)},
        {"job_id": "ok", "argv": ()},
        {"job_id": "ok", "argv": ("has\x00nul",)},
        {"job_id": "ok", "argv": ("true",), "cwd": "relative"},
        {"job_id": "ok", "argv": ("true",), "env": (("BAD-NAME", "x"),)},
        {"job_id": "ok", "argv": ("true",), "timeout_seconds": 0},
    ],
)
def test_run_request_refuses_noncanonical_or_unsafe_fields(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RunRequest(**kwargs)


def test_job_receipt_binds_output_bytes_and_refuses_tampering():
    request = RunRequest("receipt", ("/bin/true",))
    receipt = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="succeeded",
        started_ns=10,
        finished_ns=20,
        returncode=0,
        pid=None,
        stdout=b"out\x00",
        stderr=b"err",
        transport="local",
    )
    assert receipt.succeeded
    assert len(receipt.receipt_sha256) == 64
    assert receipt.as_dict() == receipt.to_payload()
    assert JobReceipt.from_payload(receipt.to_payload()) == receipt

    tampered = receipt.to_payload()
    tampered["stdout_b64"] = base64.b64encode(b"different").decode()
    with pytest.raises(ValueError, match="digest mismatch"):
        JobReceipt.from_payload(tampered)


def test_local_run_is_shell_free_and_persists_final_receipt(tmp_path):
    runner = RecordingRunner(completed(stdout=b"answer", stderr=b"note"))
    transport = LocalTransport(tmp_path / "state", run_impl=runner)
    request = RunRequest(
        "local-run",
        ("/usr/bin/program", "a b", "'; touch /tmp/never; #"),
        cwd=str(tmp_path),
        env={"ONLY_THIS": "value"},
        stdin=b"request-body",
        inherit_env=False,
    )

    receipt = transport.run(request)

    assert receipt.state == "succeeded"
    assert receipt.returncode == 0
    assert receipt.stdout == b"answer"
    assert transport.status(request.job_id) == receipt
    argv, kwargs = runner.calls[0]
    assert argv == list(request.argv)
    assert kwargs["shell"] is False
    assert kwargs["input"] == b"request-body"
    assert kwargs["env"] == {"ONLY_THIS": "value"}
    stored_request = (tmp_path / "state/jobs/local-run/request.json").read_bytes()
    assert stored_request == canonical_json_bytes(request.to_payload())


def test_local_run_records_failure_timeout_and_conflict(tmp_path):
    failed_transport = LocalTransport(
        tmp_path / "failed", run_impl=RecordingRunner(completed(returncode=7, stderr=b"bad"))
    )
    failed = failed_transport.run(RunRequest("failure", ("/bin/fail",)))
    assert (failed.state, failed.returncode, failed.stderr) == ("failed", 7, b"bad")

    timeout = subprocess.TimeoutExpired(
        cmd=["/bin/slow"], timeout=1, output=b"partial", stderr=b"late"
    )
    timed_transport = LocalTransport(
        tmp_path / "timed", run_impl=RecordingRunner(error=timeout)
    )
    timed = timed_transport.run(RunRequest("timeout", ("/bin/slow",)))
    assert timed.state == "timed_out"
    assert timed.stdout == b"partial"
    assert timed.stderr == b"late"

    with pytest.raises(JobConflictError, match="refusing overwrite"):
        failed_transport.run(RunRequest("failure", ("/bin/other",)))


def test_local_start_uses_detached_fixed_worker_argv_and_shell_false(tmp_path):
    calls = []

    class FakeProcess:
        pid = 1234

    def fake_popen(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return FakeProcess()

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=fake_popen,
        pid_alive=lambda pid: pid == 1234,
    )
    request = RunRequest("detached", ("/bin/echo", "not on worker argv"))

    receipt = transport.start(request)

    assert receipt.state == "running"
    assert transport.status("detached") == receipt
    argv, kwargs = calls[0]
    assert argv[-3:] == [str(tmp_path / "state"), "detached", "local"]
    assert "not on worker argv" not in argv
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True

    with pytest.raises(JobConflictError):
        transport.start(request)


def test_local_status_binds_request_identity_and_fails_closed_on_dead_pid(tmp_path):
    class FakeProcess:
        pid = 987654

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=lambda *args, **kwargs: FakeProcess(),
        pid_alive=lambda pid: False,
    )
    request = RunRequest("dead-worker", ("/bin/work",))
    running = transport.start(request)
    assert running.pid == FakeProcess.pid

    failed = transport.status(request.job_id)

    assert failed.state == "transport_error"
    assert b"not alive" in failed.stderr
    assert transport.status(request.job_id) == failed

    receipt_path = tmp_path / "state/jobs/dead-worker/receipt.json"
    tampered = failed.to_payload()
    tampered["request_sha256"] = "0" * 64
    receipt_path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ClusterTransportError, match="identity mismatch"):
        transport.status(request.job_id)


def test_ssh_request_is_stdin_only_canonical_base64_and_shell_free(tmp_path):
    request = RunRequest(
        "ssh-run",
        ("/bin/echo", "$(touch /tmp/owned)", "'; reboot; #"),
        cwd="/remote/work with spaces",
        env={"TOKEN": "manifest-only-secret"},
    )
    remote_receipt = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="succeeded",
        started_ns=100,
        finished_ns=200,
        returncode=0,
        pid=None,
        stdout=b"ok",
        stderr=b"",
        transport="remote",
    )
    runner = RecordingRunner(completed(stdout=helper_job_response(remote_receipt)))
    transport = SSHTransport(
        "rtx4090",
        remote_helper_path="/opt/prismaquant/cluster_transport.py",
        run_impl=runner,
    )

    receipt = transport.run(request)

    assert receipt.transport == "ssh:rtx4090"
    argv, kwargs = runner.calls[0]
    assert kwargs["shell"] is False
    command = " ".join(argv)
    for private_value in (*request.argv, request.cwd, "manifest-only-secret"):
        assert private_value not in command
    assert argv[-1] == (
        "exec /usr/bin/python3 -P -B -s "
        "/opt/prismaquant/cluster_transport.py --remote-helper"
    )
    wire = kwargs["input"].strip()
    envelope_bytes = base64.b64decode(wire, validate=True)
    assert envelope_bytes == canonical_json_bytes(json.loads(envelope_bytes))
    envelope = json.loads(envelope_bytes)
    assert envelope["action"] == "run"
    assert envelope["payload"] == request.to_payload()


def test_ssh_start_and_status_are_explicit_helper_actions():
    request = RunRequest("remote-job", ("/bin/work",))
    running = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="running",
        started_ns=100,
        finished_ns=None,
        returncode=None,
        pid=None,
        stdout=b"",
        stderr=b"",
        transport="remote",
    )
    replies = [helper_job_response(running), helper_job_response(running)]

    def fake_run(argv, **kwargs):
        return completed(stdout=replies.pop(0))

    runner = RecordingRunner(fake_run)
    transport = SSHTransport(
        "sparky",
        remote_helper_path="/srv/prismaquant/cluster_transport.py",
        run_impl=runner,
    )

    assert transport.start(request).state == "running"
    assert transport.status(request.job_id).state == "running"
    actions = []
    payloads = []
    for _, kwargs in runner.calls:
        envelope = json.loads(base64.b64decode(kwargs["input"].strip()))
        actions.append(envelope["action"])
        payloads.append(envelope["payload"])
    assert actions == ["start", "status"]
    assert payloads[1] == {"job_id": "remote-job"}


@pytest.mark.parametrize(
    "host,helper",
    [
        ("-oProxyCommand=bad", "/safe/helper.py"),
        ("host;bad", "/safe/helper.py"),
        ("host", "relative/helper.py"),
        ("host", "/safe/../bad.py"),
        ("host", "/safe/has space.py"),
    ],
)
def test_ssh_configuration_refuses_option_and_remote_command_injection(host, helper):
    with pytest.raises(ValueError):
        SSHTransport(host, remote_helper_path=helper)


def test_ssh_refuses_noncanonical_or_failed_helper_response():
    noncanonical = b'{"schema": "prismaquant.cluster_transport.helper_response.v1"}'
    bad_json = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(completed(stdout=noncanonical)),
    )
    with pytest.raises(ClusterTransportError, match="invalid canonical JSON"):
        bad_json.status("job")

    failed = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(completed(returncode=255, stderr=b"connection refused")),
    )
    with pytest.raises(ClusterTransportError, match="connection refused"):
        failed.status("job")

    install = failed.helper_install_spec()
    assert install.remote_path == "/safe/helper.py"
    assert install.size_bytes == len(install.source)
    assert install.source_sha256 == hashlib.sha256(install.source).hexdigest()
    assert base64.b64decode(install.to_payload()["source_b64"]) == install.source


def test_tree_manifest_is_canonical_and_verifies_nested_and_empty_dirs(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "a.txt").write_bytes(b"alpha")
    (source / "nested/b.bin").write_bytes(b"\x00\x01")

    manifest = build_tree_manifest(source)

    assert manifest.root_kind == "directory"
    assert [entry.path for entry in manifest.entries] == [
        "a.txt",
        "empty",
        "nested",
        "nested/b.bin",
    ]
    assert manifest.total_bytes == 7
    assert TreeManifest.from_payload(manifest.to_payload()) == manifest
    assert TreeManifest.from_json_bytes(manifest.to_json_bytes()) == manifest
    verify_tree_manifest(source, manifest)


def test_manifest_parser_refuses_traversal_duplicates_unsorted_and_symlinks(tmp_path):
    digest = hashlib.sha256(b"x").hexdigest()
    safe = ManifestEntry("a", "file", 1, digest).to_payload()
    base = {"schema": "prismaquant.cluster_transport.tree_manifest.v1", "root_kind": "directory"}

    with pytest.raises(ManifestError, match="traversal"):
        TreeManifest.from_payload({**base, "entries": [{**safe, "path": "../escape"}]})
    with pytest.raises(ManifestError, match="duplicate"):
        TreeManifest.from_payload({**base, "entries": [safe, safe]})
    with pytest.raises(ManifestError, match="sorted"):
        TreeManifest.from_payload(
            {**base, "entries": [{**safe, "path": "z"}, {**safe, "path": "a"}]}
        )
    duplicate_member_json = (
        b'{"entries":[],"root_kind":"directory","root_kind":"directory",'
        b'"schema":"prismaquant.cluster_transport.tree_manifest.v1"}'
    )
    with pytest.raises(ValueError, match="strict JSON"):
        TreeManifest.from_json_bytes(duplicate_member_json)

    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("data")
    (source / "link").symlink_to("target")
    with pytest.raises(ManifestError, match="symlinks"):
        build_tree_manifest(source)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "file").write_text("x")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ManifestError, match="traverse a symlink"):
        build_tree_manifest(linked_parent / "file")


def test_local_verified_copy_rehashes_and_never_clobbers(tmp_path):
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "sub/data").write_bytes(b"payload")
    manifest = build_tree_manifest(source)
    transport = LocalTransport(tmp_path / "state")
    destination = tmp_path / "published"

    receipt = transport.copy_verified(
        source, destination, expected_manifest=manifest
    )

    assert receipt.manifest_sha256 == manifest.identity_sha256
    assert receipt.total_bytes == len(b"payload")
    verify_tree_manifest(destination, manifest)
    with pytest.raises(FileExistsError):
        transport.copy_verified(source, destination, expected_manifest=manifest)
    assert (destination / "sub/data").read_bytes() == b"payload"


def test_local_verified_file_copy_refuses_expected_manifest_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    expected = build_tree_manifest(source)
    source.write_bytes(b"second")
    destination = tmp_path / "destination.bin"

    with pytest.raises(ManifestError, match="differs from expected"):
        LocalTransport(tmp_path / "state").copy_verified(
            source, destination, expected_manifest=expected
        )
    assert not destination.exists()


def test_local_verified_copy_refuses_symlinked_destination_parent(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ManifestError, match="traverse a symlink"):
        LocalTransport(tmp_path / "state").copy_verified(
            source, linked_parent / "destination"
        )


def test_nvidia_csv_and_memavailable_parsing_are_unit_explicit():
    csv_payload = (
        '2026/08/24 12:00:00.000, 0, "NVIDIA, Test GPU", GPU-1, '
        "00000000:01:00.0, 25, 10, 123.5, 16384, 55, 87.25\n"
        "2026/08/24 12:00:00.000, 1, NVIDIA Other, GPU-2, "
        "00000000:02:00.0, [N/A], 0, 0, 8192, N/A, N/A\n"
    )

    samples = parse_nvidia_smi_csv(csv_payload)

    assert len(samples) == 2
    assert samples[0].name == "NVIDIA, Test GPU"
    assert samples[0].gpu_utilization_pct == 25.0
    assert samples[1].gpu_utilization_pct is None
    assert samples[1].temperature_c is None
    assert parse_mem_available("MemTotal: 2 kB\nMemAvailable: 12345 kB\n") == 12345 * 1024
    with pytest.raises(ValueError, match="exactly one"):
        parse_mem_available("MemTotal: 123 kB\n")


def make_gpu(index: int, uuid: str, gpu: float, memory: float) -> GpuSample:
    return GpuSample(
        timestamp="t",
        index=index,
        name="GPU",
        uuid=uuid,
        pci_bus_id=f"0000:{index:02x}:00.0",
        gpu_utilization_pct=gpu,
        memory_utilization_pct=memory,
        memory_used_mib=1.0,
        memory_total_mib=2.0,
        temperature_c=50.0,
        power_w=100.0,
    )


def test_utilization_summary_has_deterministic_mean_percentiles_activity_and_host_memory():
    snapshots = (
        TelemetrySnapshot(1, 1000, (make_gpu(0, "GPU-0", 0, 0), make_gpu(1, "GPU-1", 10, 20))),
        TelemetrySnapshot(2, 500, (make_gpu(0, "GPU-0", 20, 40), make_gpu(1, "GPU-1", 100, 80))),
    )

    summary = summarize_utilization(snapshots, active_threshold_pct=1)

    assert (summary.snapshot_count, summary.gpu_sample_count, summary.gpu_count) == (2, 4, 2)
    assert summary.gpu_utilization.mean == 32.5
    assert summary.gpu_utilization.p50 == 15.0
    assert summary.gpu_utilization.p95 == pytest.approx(88.0)
    assert summary.gpu_utilization.active_fraction == 0.75
    assert summary.gpu_utilization.active_fraction_gt_0 == 0.75
    assert summary.gpu_utilization.active_fraction_gt_50 == 0.25
    assert summary.gpu_utilization.active_fraction_gt_90 == 0.25
    assert summary.memory_utilization.mean == 35.0
    assert summary.memory_utilization.p50 == 30.0
    assert summary.memory_utilization.p95 == pytest.approx(74.0)
    assert summary.memory_utilization.active_fraction == 0.75
    assert summary.memory_utilization.active_fraction_gt_0 == 0.75
    assert summary.memory_utilization.active_fraction_gt_50 == 0.25
    assert summary.memory_utilization.active_fraction_gt_90 == 0.0
    assert summary.host_mem_available_bytes.mean == 750.0
    assert summary.host_mem_available_bytes.p50 == 750.0
    assert summary.host_mem_available_bytes.p95 == pytest.approx(975.0)
    assert json.loads(canonical_json_bytes(summary.to_payload()))[
        "gpu_utilization"
    ]["active_fraction_gt_90"] == 0.25


def test_local_telemetry_uses_fixed_nvidia_query_and_injected_meminfo(tmp_path):
    payload = (
        "2026/08/24 12:00:00.000, 0, GPU, GPU-0, 0000:01:00.0, "
        "5, 6, 7, 8, 9, 10\n"
    ).encode()
    runner = RecordingRunner(completed(stdout=payload))
    transport = LocalTransport(
        tmp_path / "state",
        run_impl=runner,
        meminfo_reader=lambda: "MemAvailable: 42 kB\n",
    )

    snapshot = transport.sample_telemetry()

    assert snapshot.host_mem_available_bytes == 42 * 1024
    assert snapshot.gpus[0].gpu_utilization_pct == 5.0
    argv, kwargs = runner.calls[0]
    assert argv[0] == "nvidia-smi"
    assert argv[1].startswith("--query-gpu=timestamp,index,name,uuid")
    assert argv[2] == "--format=csv,noheader,nounits"
    assert kwargs["shell"] is False


def test_empty_utilization_summary_is_total_and_deterministic():
    summary = summarize_utilization(())
    assert summary.snapshot_count == 0
    assert summary.gpu_utilization.mean is None
    assert summary.gpu_utilization.active_fraction is None
    assert summary.host_mem_available_bytes.count == 0
