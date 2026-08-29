from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

import prismaquant.cluster_transport as cluster_transport
from prismaquant.cluster_transport import (
    ClusterTransportError,
    GpuSample,
    HELPER_ERROR_SCHEMA,
    HELPER_ENVELOPE_SCHEMA,
    HELPER_RESPONSE_SCHEMA,
    JobConflictError,
    JobNotFoundError,
    JobReceipt,
    LocalTransport,
    ManifestEntry,
    ManifestError,
    RunRequest,
    SSHTransport,
    TelemetryIntegrityError,
    TelemetrySnapshot,
    TelemetryUnavailableError,
    TreeManifest,
    build_tree_manifest,
    canonical_json_bytes,
    parse_mem_available,
    parse_nvidia_smi_csv,
    summarize_utilization,
    verify_tree_manifest,
    write_exact_bytes_no_clobber,
)


_HISTORICAL_HELPER_COMMIT = "6bde48a"
_HISTORICAL_HELPER_SHA256 = (
    "9fc47559fba4e0d1d4276a05375b2813dd91403991f61ad60629a8a914b50195"
)


def _historical_helper_source() -> bytes:
    repo = Path(__file__).parents[1]
    result = subprocess.run(
        [
            "git",
            "show",
            f"{_HISTORICAL_HELPER_COMMIT}:prismaquant/cluster_transport.py",
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert hashlib.sha256(result.stdout).hexdigest() == _HISTORICAL_HELPER_SHA256
    assert b"def inspect(" not in result.stdout
    return result.stdout


def _filesystem_snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    snapshot: dict[str, tuple[int, bytes | None]] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        snapshot[relative] = (mode, path.read_bytes() if path.is_file() else None)
    return snapshot


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


def test_exact_no_clobber_writer_is_idempotent_but_rejects_drift(tmp_path):
    output = tmp_path / "receipt.json"
    write_exact_bytes_no_clobber(output, b"exact\n")
    write_exact_bytes_no_clobber(output, b"exact\n")
    assert output.read_bytes() == b"exact\n"

    with pytest.raises(ClusterTransportError, match="differs"):
        write_exact_bytes_no_clobber(output, b"different\n")


def helper_job_response(receipt: JobReceipt) -> bytes:
    return canonical_json_bytes(
        {
            "schema": HELPER_RESPONSE_SCHEMA,
            "kind": "job_receipt",
            "payload": receipt.to_payload(),
        }
    )


def helper_response(kind: str, payload: object) -> bytes:
    return canonical_json_bytes(
        {
            "schema": HELPER_RESPONSE_SCHEMA,
            "kind": kind,
            "payload": payload,
        }
    )


def helper_error_response(
    error_kind: str,
    exception_type: str,
    message: object,
) -> bytes:
    return helper_response(
        "error",
        {
            "schema": HELPER_ERROR_SCHEMA,
            "error_kind": error_kind,
            "exception_type": exception_type,
            "message": message,
        },
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
        process_identity_reader=lambda pid: {
            "schema": "prismaquant.cluster_transport.process_identity.v1",
            "pid": pid,
            "boot_id": "fixture",
            "start_ticks": 1,
        },
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


def test_local_status_distinguishes_absence_from_a_claimed_job_id(tmp_path):
    transport = LocalTransport(tmp_path / "state")

    with pytest.raises(JobNotFoundError, match="job 'absent' does not exist"):
        transport.status("absent")

    claimed_root = tmp_path / "state" / "jobs" / "claimed"
    claimed_root.mkdir(parents=True)
    with pytest.raises(ClusterTransportError) as invalid_state:
        transport.status("claimed")
    assert type(invalid_state.value) is ClusterTransportError

    with pytest.raises(JobConflictError, match="refusing overwrite"):
        transport.start(RunRequest("claimed", ("/bin/true",)))


def test_real_detached_local_worker_publishes_and_is_adopted(tmp_path) -> None:
    transport = LocalTransport(tmp_path / "state")
    request = RunRequest(
        "real-detached-worker",
        (
            sys.executable,
            "-P",
            "-B",
            "-s",
            "-c",
            "import sys; sys.stdout.write('detached-ok\\n')",
        ),
        env=(("LANG", "C.UTF-8"),),
        timeout_seconds=10.0,
    )

    running = transport.start(request)
    assert running.state == "running"
    deadline = time.monotonic() + 10.0
    while True:
        receipt = transport.status(request.job_id)
        if receipt.state != "running":
            break
        if time.monotonic() >= deadline:
            pytest.fail("real detached worker did not publish a terminal receipt")
        time.sleep(0.02)

    assert receipt.state == "succeeded"
    assert receipt.returncode == 0
    assert receipt.stdout == b"detached-ok\n"
    assert transport.status(request.job_id) == receipt


def test_local_status_binds_request_identity_and_fails_closed_on_dead_pid(tmp_path):
    class FakeProcess:
        pid = 987654

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=lambda *args, **kwargs: FakeProcess(),
        pid_alive=lambda pid: False,
        process_identity_reader=lambda pid: {
            "schema": "prismaquant.cluster_transport.process_identity.v1",
            "pid": pid,
            "boot_id": "fixture",
            "start_ticks": 1,
        },
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


def test_local_inspect_never_reconciles_stale_running_receipt(tmp_path):
    class FakeProcess:
        pid = 987655

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=lambda *args, **kwargs: FakeProcess(),
        pid_alive=lambda _pid: False,
        process_identity_reader=lambda pid: {
            "schema": "prismaquant.cluster_transport.process_identity.v1",
            "pid": pid,
            "boot_id": "fixture",
            "start_ticks": 1,
        },
    )
    request = RunRequest("inspect-stale-worker", ("/bin/work",))
    assert transport.start(request).state == "running"
    job_root = tmp_path / "state/jobs/inspect-stale-worker"
    before = {
        child.name: child.read_bytes()
        for child in job_root.iterdir()
        if child.is_file()
    }

    assert transport.inspect(request.job_id).state == "running"
    after = {
        child.name: child.read_bytes()
        for child in job_root.iterdir()
        if child.is_file()
    }

    assert after == before
    assert transport.status(request.job_id).state == "transport_error"
    assert (job_root / "receipt.json").read_bytes() != before["receipt.json"]


@pytest.mark.parametrize(
    "substitution", ("job_root", "request", "receipt", "child"),
)
def test_local_status_rejects_symlinked_durable_job_state(
    tmp_path,
    substitution,
):
    transport = LocalTransport(
        tmp_path / "state",
        run_impl=RecordingRunner(completed()),
    )
    request = RunRequest("linked-job", ("/bin/true",))
    transport.run(request)
    job_root = tmp_path / "state/jobs/linked-job"
    if substitution == "job_root":
        target = job_root.with_name("linked-job-target")
        job_root.rename(target)
        job_root.symlink_to(target.name, target_is_directory=True)
    elif substitution in {"request", "receipt"}:
        path = job_root / f"{substitution}.json"
        target = job_root / f"{substitution}.target"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target.name)
    else:
        target = job_root / "extra.target"
        target.write_bytes(b"ignored before no-follow enumeration\n")
        (job_root / "extra-child").symlink_to(target.name)

    with pytest.raises(ClusterTransportError):
        transport.status(request.job_id)


def test_local_status_rejects_live_pid_reuse_by_boot_scoped_identity(tmp_path):
    class FakeProcess:
        pid = 4321

    reads = 0

    def identity(pid):
        nonlocal reads
        reads += 1
        return {
            "schema": "prismaquant.cluster_transport.process_identity.v1",
            "pid": pid,
            "boot_id": "fixture",
            "start_ticks": reads,
        }

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=lambda *args, **kwargs: FakeProcess(),
        pid_alive=lambda _pid: True,
        process_identity_reader=identity,
    )
    request = RunRequest("reused-worker-pid", ("/bin/work",))

    assert transport.start(request).state == "running"
    failed = transport.status(request.job_id)

    assert failed.state == "transport_error"
    assert b"identity is not alive" in failed.stderr


@pytest.mark.parametrize("process_state", ["Z", "X", "x"])
def test_linux_process_identity_rejects_zombie_and_dead_states(
    monkeypatch: pytest.MonkeyPatch,
    process_state: str,
) -> None:
    pid = 4321
    boot_id = "01234567-89ab-cdef-0123-456789abcdef"
    # Fields after ``comm`` begin with state (proc field 3); starttime is
    # field 22 and therefore index 19 in this suffix.
    stat_payload = f"{pid} (detached worker) {process_state} " + " ".join(
        ["1"] * 19
    )

    def read_text(path: Path, *, encoding: str) -> str:
        assert encoding == "ascii"
        if path == Path("/proc/sys/kernel/random/boot_id"):
            return boot_id + "\n"
        if path == Path(f"/proc/{pid}/stat"):
            return stat_payload
        raise AssertionError(path)

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(ProcessLookupError):
        cluster_transport._linux_process_identity(pid)


def test_local_status_terminates_a_zombie_identity_even_when_kill_zero_is_live(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        pid = 2468

    reads = 0

    def identity(pid: int) -> dict[str, object]:
        nonlocal reads
        reads += 1
        if reads > 1:
            raise ProcessLookupError(pid)
        return {
            "schema": "prismaquant.cluster_transport.process_identity.v1",
            "pid": pid,
            "boot_id": "fixture",
            "start_ticks": 1,
        }

    transport = LocalTransport(
        tmp_path / "state",
        popen_impl=lambda *args, **kwargs: FakeProcess(),
        pid_alive=lambda _pid: True,
        process_identity_reader=identity,
    )
    request = RunRequest("zombie-worker", ("/bin/work",))

    assert transport.start(request).state == "running"
    failed = transport.status(request.job_id)

    assert failed.state == "transport_error"
    assert b"identity is not alive" in failed.stderr
    assert transport.status(request.job_id) == failed


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
    remote_argv = shlex.split(argv[-1])
    install = transport.helper_install_spec()
    assert remote_argv[:6] == [
        "exec", "/usr/bin/python3", "-P", "-B", "-s", "-c",
    ]
    assert remote_argv[7:9] == [
        "/opt/prismaquant/cluster_transport.py",
        install.source_sha256,
    ]
    assert remote_argv[-1] == "--remote-helper"
    launcher_source = base64.b64decode(remote_argv[9], validate=True)
    assert launcher_source.decode() == (
        cluster_transport._SSH_HELPER_LAUNCHER_PROGRAM
    )
    assert hashlib.sha256(launcher_source).hexdigest() == remote_argv[10]
    launcher = remote_argv[6]
    assert "O_NOFOLLOW" in launcher
    assert "exec(compile(source" in launcher
    assert "/opt/prismaquant/cluster_transport.py" not in launcher
    wire = kwargs["input"].strip()
    envelope_bytes = base64.b64decode(wire, validate=True)
    assert envelope_bytes == canonical_json_bytes(json.loads(envelope_bytes))
    envelope = json.loads(envelope_bytes)
    assert envelope["action"] == "run"
    assert envelope["payload"] == request.to_payload()


def test_ssh_start_status_and_inspect_are_explicit_helper_actions():
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
    replies = [
        helper_job_response(running),
        helper_job_response(running),
        helper_job_response(running),
    ]

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
    assert transport.inspect(request.job_id).state == "running"
    actions = []
    payloads = []
    for _, kwargs in runner.calls:
        envelope = json.loads(base64.b64decode(kwargs["input"].strip()))
        actions.append(envelope["action"])
        payloads.append(envelope["payload"])
    assert actions == ["start", "status", "inspect"]
    assert payloads[1:] == [
        {"job_id": "remote-job"},
        {"job_id": "remote-job"},
    ]


class LocalShellSSHRunner:
    """Execute only the fixed remote-command argument for launcher tests."""

    def __init__(self, *, home: Path | None = None):
        self.calls = []
        self.home = home

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        environment = dict(os.environ)
        if self.home is not None:
            environment["HOME"] = str(self.home)
        return subprocess.run(
            ["/bin/sh", "-c", argv[-1]],
            input=kwargs.get("input"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=kwargs.get("timeout", 10),
            env=environment,
        )


@pytest.mark.parametrize("historical", [False, True])
def test_ssh_launcher_executes_only_planned_held_helper_bytes(
    tmp_path: Path,
    historical: bool,
) -> None:
    source = (
        _historical_helper_source()
        if historical
        else cluster_transport.canonical_helper_source()
    )
    helper = tmp_path / "remote" / "cluster_transport.py"
    helper.parent.mkdir()
    runner = LocalShellSSHRunner(home=tmp_path / "home")
    transport = SSHTransport(
        "fixture",
        remote_helper_path=str(helper),
        helper_source=source,
        run_impl=runner,
    )
    spec = transport.helper_install_spec()
    helper.write_bytes(spec.source)

    if historical:
        receipt = transport.run(RunRequest("held-historical", ("/bin/true",)))
        assert receipt.succeeded
    else:
        with pytest.raises(JobNotFoundError):
            transport.inspect("definitely-missing")

    remote_argv = shlex.split(runner.calls[-1][0][-1])
    assert remote_argv[7:9] == [str(helper), spec.source_sha256]
    assert remote_argv[-1] == "--remote-helper"
    assert remote_argv[6] == cluster_transport._SSH_HELPER_LAUNCHER_PROGRAM
    assert not (helper.parent / "__pycache__").exists()


@pytest.mark.parametrize("historical", [False, True])
def test_ssh_launcher_refuses_post_plan_helper_substitution_without_execution(
    tmp_path: Path,
    historical: bool,
) -> None:
    source = (
        _historical_helper_source()
        if historical
        else cluster_transport.canonical_helper_source()
    )
    helper = tmp_path / "remote" / "cluster_transport.py"
    helper.parent.mkdir()
    marker = helper.parent / "MALICIOUS_EXECUTED"
    runner = LocalShellSSHRunner(home=tmp_path / "home")
    transport = SSHTransport(
        "fixture",
        remote_helper_path=str(helper),
        helper_source=source,
        run_impl=runner,
    )
    planned = transport.helper_install_spec()
    helper.write_bytes(planned.source)
    helper.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )
    before_names = sorted(path.name for path in helper.parent.iterdir())

    with pytest.raises(ClusterTransportError, match="digest differs"):
        transport.inspect("must-not-run")

    assert not marker.exists()
    assert not (helper.parent / "__pycache__").exists()
    assert sorted(path.name for path in helper.parent.iterdir()) == before_names


@pytest.mark.parametrize("historical", [False, True])
def test_ssh_held_launcher_secures_real_detached_worker_entry(
    tmp_path: Path,
    historical: bool,
) -> None:
    source = (
        _historical_helper_source()
        if historical
        else cluster_transport.canonical_helper_source()
    )
    helper = tmp_path / "remote" / "cluster_transport.py"
    helper.parent.mkdir()
    helper.write_bytes(source)
    runner = LocalShellSSHRunner(home=tmp_path / "home")
    transport = SSHTransport(
        "fixture",
        remote_helper_path=str(helper),
        helper_source=source,
        run_impl=runner,
    )
    request = RunRequest(
        "held-detached-worker",
        (
            sys.executable,
            "-P",
            "-B",
            "-s",
            "-c",
            "import sys; sys.stdout.write('held-worker-ok\\n')",
        ),
        timeout_seconds=10.0,
    )

    running = transport.start(request)
    assert running.state == "running"
    deadline = time.monotonic() + 10.0
    while True:
        receipt = transport.status(request.job_id)
        if receipt.state != "running":
            break
        if time.monotonic() >= deadline:
            pytest.fail("held detached worker did not publish a terminal receipt")
        time.sleep(0.02)

    assert receipt.succeeded
    assert receipt.stdout == b"held-worker-ok\n"
    assert not (helper.parent / "__pycache__").exists()


@pytest.mark.parametrize("historical", [False, True])
def test_detached_worker_refuses_helper_substitution_before_state_or_payload(
    tmp_path: Path,
    historical: bool,
) -> None:
    source = (
        _historical_helper_source()
        if historical
        else cluster_transport.canonical_helper_source()
    )
    helper = tmp_path / "remote" / "cluster_transport.py"
    helper.parent.mkdir()
    helper.write_bytes(source)
    state_root = tmp_path / "state"
    job_root = state_root / "jobs" / "worker-substitution"
    job_root.mkdir(parents=True)
    marker = tmp_path / "payload-executed"
    request = RunRequest(
        "worker-substitution",
        (
            sys.executable,
            "-P",
            "-B",
            "-s",
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')",
        ),
    )
    running = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="running",
        started_ns=1,
        finished_ns=None,
        returncode=None,
        pid=None,
        stdout=b"",
        stderr=b"",
        transport="remote",
    )
    (job_root / "request.json").write_bytes(
        canonical_json_bytes(request.to_payload())
    )
    (job_root / "receipt.json").write_bytes(
        canonical_json_bytes(running.to_payload())
    )
    (job_root / "launched").write_bytes(b"ready\n")
    planned_digest = hashlib.sha256(source).hexdigest()
    helper.write_text("raise RuntimeError('substituted helper executed')\n")
    launcher = cluster_transport._SSH_HELPER_LAUNCHER_PROGRAM
    launcher_bytes = launcher.encode("utf-8")
    before = _filesystem_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-B",
            "-s",
            "-c",
            launcher,
            str(helper),
            planned_digest,
            base64.b64encode(launcher_bytes).decode("ascii"),
            hashlib.sha256(launcher_bytes).hexdigest(),
            "--local-worker",
            str(state_root),
            request.job_id,
            "remote",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=10,
    )

    assert completed.returncode == 126
    assert b"digest differs from bootstrap plan" in completed.stderr
    assert not marker.exists()
    assert _filesystem_snapshot(tmp_path) == before


def test_ssh_launcher_refuses_symlinked_helper_ancestry(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    helper = real / "cluster_transport.py"
    helper.write_bytes(cluster_transport.canonical_helper_source())
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    transport = SSHTransport(
        "fixture",
        remote_helper_path=str(linked / helper.name),
        run_impl=LocalShellSSHRunner(),
    )

    with pytest.raises(ClusterTransportError, match="ancestry"):
        transport.inspect("must-not-run")

    assert not (real / "__pycache__").exists()


def test_remote_helper_uses_explicit_campaign_state_root(tmp_path: Path) -> None:
    state_root = tmp_path / "sealed-worker-state" / "transport"
    request = RunRequest("rooted-helper", ("/bin/true",))
    envelope = {
        "schema": HELPER_ENVELOPE_SCHEMA,
        "action": "run",
        "payload": request.to_payload(),
    }
    helper = Path(__file__).parents[1] / "prismaquant/cluster_transport.py"

    completed = subprocess.run(
        [
            sys.executable, "-P", "-B", "-s", str(helper),
            "--remote-helper", "--remote-helper-state-root", str(state_root),
        ],
        input=base64.b64encode(canonical_json_bytes(envelope)) + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    response = json.loads(completed.stdout)
    assert response["kind"] == "job_receipt"
    assert JobReceipt.from_payload(response["payload"]).succeeded
    assert (state_root / "jobs/rooted-helper/request.json").is_file()
    assert (state_root / "jobs/rooted-helper/receipt.json").is_file()


def test_remote_helper_inspect_is_read_only_for_stale_running_job(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "sealed-worker-state" / "transport"
    job_root = state_root / "jobs" / "stale-remote"
    job_root.mkdir(parents=True)
    request = RunRequest("stale-remote", ("/bin/true",))
    running = JobReceipt(
        job_id=request.job_id,
        request_sha256=request.request_sha256,
        state="running",
        started_ns=1,
        finished_ns=None,
        returncode=None,
        pid=999_999_999,
        stdout=b"",
        stderr=b"",
        transport="remote",
    )
    (job_root / "request.json").write_bytes(
        canonical_json_bytes(request.to_payload())
    )
    receipt_path = job_root / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(running.to_payload()))
    before = receipt_path.read_bytes()
    helper = Path(__file__).parents[1] / "prismaquant/cluster_transport.py"

    def invoke(action: str) -> JobReceipt:
        envelope = {
            "schema": HELPER_ENVELOPE_SCHEMA,
            "action": action,
            "payload": {"job_id": request.job_id},
        }
        result = subprocess.run(
            [
                sys.executable,
                "-P",
                "-B",
                "-s",
                str(helper),
                "--remote-helper",
                "--remote-helper-state-root",
                str(state_root),
            ],
            input=base64.b64encode(canonical_json_bytes(envelope)) + b"\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr.decode()
        response = json.loads(result.stdout)
        assert response["kind"] == "job_receipt"
        return JobReceipt.from_payload(response["payload"])

    assert invoke("inspect").state == "running"
    assert receipt_path.read_bytes() == before
    assert invoke("status").state == "transport_error"
    assert receipt_path.read_bytes() != before


def test_remote_helper_emits_structured_telemetry_integrity_error(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "sealed-worker-state" / "transport"
    envelope = {
        "schema": HELPER_ENVELOPE_SCHEMA,
        "action": "telemetry",
        "payload": {"unexpected": True},
    }
    helper = Path(__file__).parents[1] / "prismaquant/cluster_transport.py"

    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-B",
            "-s",
            str(helper),
            "--remote-helper",
            "--remote-helper-state-root",
            str(state_root),
        ],
        input=base64.b64encode(canonical_json_bytes(envelope)) + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()
    response = json.loads(result.stdout)
    assert response == {
        "schema": HELPER_RESPONSE_SCHEMA,
        "kind": "error",
        "payload": {
            "schema": HELPER_ERROR_SCHEMA,
            "error_kind": "telemetry_integrity",
            "exception_type": "ValueError",
            "message": "telemetry payload must be empty",
        },
    }


def test_helper_error_payload_names_unavailability_without_message_prefixes():
    payload = cluster_transport._helper_error_payload(
        TelemetryUnavailableError("driver query failed"),
        action="telemetry",
    )

    assert payload == {
        "schema": HELPER_ERROR_SCHEMA,
        "error_kind": "telemetry_unavailable",
        "exception_type": "TelemetryUnavailableError",
        "message": "driver query failed",
    }


@pytest.mark.parametrize(
    ("error", "action", "error_kind"),
    [
        (JobNotFoundError("missing"), "status", "job_not_found"),
        (JobConflictError("claimed"), "start", "job_conflict"),
    ],
)
def test_helper_error_payload_preserves_typed_job_errors(
    error,
    action,
    error_kind,
):
    payload = cluster_transport._helper_error_payload(error, action=action)

    assert payload == {
        "schema": HELPER_ERROR_SCHEMA,
        "error_kind": error_kind,
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def test_remote_helper_preserves_not_found_and_conflict(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "sealed-worker-state" / "transport"
    helper = Path(__file__).parents[1] / "prismaquant/cluster_transport.py"

    def invoke(envelope: dict[str, object]) -> dict[str, object]:
        result = subprocess.run(
            [
                sys.executable,
                "-P",
                "-B",
                "-s",
                str(helper),
                "--remote-helper",
                "--remote-helper-state-root",
                str(state_root),
            ],
            input=base64.b64encode(canonical_json_bytes(envelope)) + b"\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr.decode()
        return json.loads(result.stdout)

    missing = invoke(
        {
            "schema": HELPER_ENVELOPE_SCHEMA,
            "action": "status",
            "payload": {"job_id": "missing"},
        }
    )
    assert missing == {
        "schema": HELPER_RESPONSE_SCHEMA,
        "kind": "error",
        "payload": {
            "schema": HELPER_ERROR_SCHEMA,
            "error_kind": "job_not_found",
            "exception_type": "JobNotFoundError",
            "message": "job 'missing' does not exist",
        },
    }

    (state_root / "jobs" / "claimed").mkdir(parents=True)
    request = RunRequest("claimed", ("/bin/true",))
    conflict = invoke(
        {
            "schema": HELPER_ENVELOPE_SCHEMA,
            "action": "start",
            "payload": request.to_payload(),
        }
    )
    assert conflict == {
        "schema": HELPER_RESPONSE_SCHEMA,
        "kind": "error",
        "payload": {
            "schema": HELPER_ERROR_SCHEMA,
            "error_kind": "job_conflict",
            "exception_type": "JobConflictError",
            "message": "job 'claimed' already exists; refusing overwrite",
        },
    }


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


@pytest.mark.parametrize(
    ("action", "error_kind", "exception_type", "expected_type"),
    [
        ("status", "job_not_found", "JobNotFoundError", JobNotFoundError),
        ("start", "job_conflict", "JobConflictError", JobConflictError),
    ],
)
def test_ssh_job_actions_restore_structured_remote_errors(
    action,
    error_kind,
    exception_type,
    expected_type,
):
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(
            completed(
                stdout=helper_error_response(
                    error_kind,
                    exception_type,
                    "remote detail",
                )
            )
        ),
    )

    with pytest.raises(expected_type, match=f"{exception_type}: remote detail"):
        if action == "status":
            transport.status("remote-job")
        else:
            transport.start(RunRequest("remote-job", ("/bin/true",)))


def test_ssh_job_operation_error_remains_generic():
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(
            completed(
                stdout=helper_error_response(
                    "operation_error",
                    "OSError",
                    "remote detail",
                )
            )
        ),
    )

    with pytest.raises(ClusterTransportError) as raised:
        transport.status("remote-job")
    assert type(raised.value) is ClusterTransportError


@pytest.mark.parametrize(
    "response",
    [
        helper_error_response("job_not_found", "RuntimeError", "spoofed"),
        helper_error_response("job_conflict", "RuntimeError", "spoofed"),
        helper_error_response(
            "operation_error", "JobNotFoundError", "spoofed",
        ),
        helper_error_response(
            "operation_error", "JobConflictError", "spoofed",
        ),
    ],
)
def test_ssh_job_actions_reject_contradictory_typed_errors(response):
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(completed(stdout=response)),
    )

    with pytest.raises(ClusterTransportError) as raised:
        transport.status("remote-job")
    assert type(raised.value) is ClusterTransportError


@pytest.mark.parametrize(
    ("error_kind", "exception_type", "expected_type"),
    [
        (
            "telemetry_unavailable",
            "TelemetryUnavailableError",
            TelemetryUnavailableError,
        ),
        ("telemetry_integrity", "TelemetryIntegrityError", TelemetryIntegrityError),
        ("telemetry_integrity", "ValueError", TelemetryIntegrityError),
        ("operation_error", "RuntimeError", TelemetryIntegrityError),
    ],
)
def test_ssh_telemetry_preserves_structured_remote_error_classification(
    error_kind,
    exception_type,
    expected_type,
):
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(
            completed(
                stdout=helper_error_response(
                    error_kind,
                    exception_type,
                    "remote detail",
                )
            )
        ),
    )

    with pytest.raises(expected_type, match=f"{exception_type}: remote detail"):
        transport.sample_telemetry()


@pytest.mark.parametrize(
    "runner",
    [
        RecordingRunner(
            error=subprocess.TimeoutExpired(cmd=["ssh"], timeout=30),
        ),
        RecordingRunner(
            completed(returncode=255, stderr=b"connection refused"),
        ),
        RecordingRunner(error=OSError("ssh executable unavailable")),
    ],
)
def test_ssh_telemetry_transport_failures_are_unavailable(runner):
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=runner,
    )

    with pytest.raises(TelemetryUnavailableError, match="SSH telemetry query"):
        transport.sample_telemetry()


@pytest.mark.parametrize(
    "response",
    [
        # A legacy/free-form error string cannot impersonate the typed class.
        helper_response("error", "TelemetryUnavailableError: spoofed"),
        helper_error_response(
            "telemetry_unavailable", "ValueError", "contradictory",
        ),
        helper_error_response(
            "telemetry_integrity", "TelemetryUnavailableError", "contradictory",
        ),
        helper_error_response("unknown", "ValueError", "unknown kind"),
        helper_error_response("telemetry_integrity", "ValueError", 7),
        helper_response("job_receipt", {}),
        helper_response("telemetry", {"captured_ns": 1}),
        b'{"kind":"telemetry", "payload":{}}',
    ],
)
def test_ssh_telemetry_malformed_helper_responses_are_integrity_errors(response):
    transport = SSHTransport(
        "host",
        remote_helper_path="/safe/helper.py",
        run_impl=RecordingRunner(completed(stdout=response)),
    )

    with pytest.raises(TelemetryIntegrityError):
        transport.sample_telemetry()


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


@pytest.mark.parametrize(
    "runner",
    [
        RecordingRunner(
            error=subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=15),
        ),
        RecordingRunner(completed(returncode=1, stderr=b"driver unavailable")),
        RecordingRunner(error=OSError("nvidia-smi unavailable")),
    ],
)
def test_local_telemetry_query_failures_are_unavailable(tmp_path, runner):
    transport = LocalTransport(
        tmp_path / "state",
        run_impl=runner,
        meminfo_reader=lambda: "MemAvailable: 42 kB\n",
    )

    with pytest.raises(TelemetryUnavailableError):
        transport.sample_telemetry()


@pytest.mark.parametrize(
    "payload",
    [
        b"not,enough,fields\n",
        (
            b"time, bad-index, GPU, GPU-0, 0000:01:00.0, "
            b"5, 6, 7, 8, 9, 10\n"
        ),
        (
            b"time, 0, GPU, GPU-0, 0000:01:00.0, "
            b"not-a-number, 6, 7, 8, 9, 10\n"
        ),
        object(),
    ],
)
def test_local_malformed_nvidia_telemetry_is_an_integrity_error(
    tmp_path,
    payload,
):
    transport = LocalTransport(
        tmp_path / "state",
        run_impl=RecordingRunner(completed(stdout=payload)),
        meminfo_reader=lambda: "MemAvailable: 42 kB\n",
    )

    with pytest.raises(TelemetryIntegrityError, match="nvidia-smi"):
        transport.sample_telemetry()


@pytest.mark.parametrize(
    "meminfo",
    [
        "MemTotal: 42 kB\n",
        "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
        "MemAvailable: not-a-number kB\n",
        object(),
    ],
)
def test_local_malformed_meminfo_is_an_integrity_error(tmp_path, meminfo):
    payload = (
        b"time, 0, GPU, GPU-0, 0000:01:00.0, "
        b"5, 6, 7, 8, 9, 10\n"
    )
    transport = LocalTransport(
        tmp_path / "state",
        run_impl=RecordingRunner(completed(stdout=payload)),
        meminfo_reader=lambda: meminfo,
    )

    with pytest.raises(TelemetryIntegrityError, match="host memory telemetry"):
        transport.sample_telemetry()


def test_local_unreadable_meminfo_is_unavailable(tmp_path):
    payload = (
        b"time, 0, GPU, GPU-0, 0000:01:00.0, "
        b"5, 6, 7, 8, 9, 10\n"
    )

    def unavailable_meminfo():
        raise OSError("meminfo unavailable")

    transport = LocalTransport(
        tmp_path / "state",
        run_impl=RecordingRunner(completed(stdout=payload)),
        meminfo_reader=unavailable_meminfo,
    )

    with pytest.raises(TelemetryUnavailableError, match="host memory telemetry"):
        transport.sample_telemetry()


def test_empty_utilization_summary_is_total_and_deterministic():
    summary = summarize_utilization(())
    assert summary.snapshot_count == 0
    assert summary.gpu_utilization.mean is None
    assert summary.gpu_utilization.active_fraction is None
    assert summary.host_mem_available_bytes.count == 0
