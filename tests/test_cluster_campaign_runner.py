from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import prismaquant.cluster_campaign as campaign


_RUNNER_TOOL = Path(campaign.__file__).resolve().parents[1] / "tools" / (
    "run_cluster_campaign.py"
)
_WRITE_BYTES = (
    "from pathlib import Path; import sys; "
    "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
    "Path(sys.argv[1]).write_bytes(sys.argv[2].encode('utf-8'))"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _receipt(path: Path, text: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _digest(text)}


def _local_host(tmp_path: Path) -> dict[str, object]:
    return {
        "id": "sparky",
        "transport": {"kind": "local"},
        "work_root": str((tmp_path / "sparky-work").resolve()),
    }


def _fake_ssh(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "fake-ssh"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "marker = args.index('--')\n"
        "remote = args[marker + 2:]\n"
        "if not remote:\n"
        "    raise SystemExit(91)\n"
        "os.execv(remote[0], remote)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    known_hosts = tmp_path / "known-hosts"
    known_hosts.write_text("sparklina ssh-ed25519 AAAAtest\n", encoding="utf-8")
    return executable.resolve(), known_hosts.resolve()


def _ssh_host(tmp_path: Path) -> dict[str, object]:
    executable, known_hosts = _fake_ssh(tmp_path)
    return {
        "id": "sparklina",
        "transport": {
            "kind": "ssh",
            "host": "sparklina",
            "port": 22,
            "user": "rob",
            "known_hosts": str(known_hosts),
            "ssh_executable": str(executable),
            "remote_helper_argv": [
                sys.executable,
                str(Path(campaign.__file__).resolve()),
                "_exec-request",
            ],
            "connect_timeout_seconds": 5,
        },
        "work_root": str((tmp_path / "sparklina-work").resolve()),
    }


def _stage(
    tmp_path: Path,
    *,
    stage_id: str,
    host_id: str,
    dependencies: list[str],
    argv: list[str],
    receipts: list[dict[str, str]],
    max_attempts: int = 2,
    timeout_seconds: int = 10,
) -> dict[str, object]:
    return {
        "id": stage_id,
        "host_id": host_id,
        "dependencies": dependencies,
        "argv": argv,
        "cwd": str(tmp_path.resolve()),
        # The command executable is absolute, so a completely closed empty env
        # is valid.  The worker adds only its reserved ownership token.
        "env": {},
        "receipts": receipts,
        "max_attempts": max_attempts,
        "timeout_seconds": timeout_seconds,
    }


def _manifest(
    tmp_path: Path,
    stages: list[dict[str, object]],
    *,
    ssh: bool = False,
    max_parallel: int | None = None,
) -> dict[str, object]:
    hosts = [_local_host(tmp_path)]
    if ssh:
        hosts.append(_ssh_host(tmp_path))
    body = {
        "schema": campaign.CAMPAIGN_MANIFEST_SCHEMA_V2,
        "campaign_id": "qwen38-125b-release",
        "coordinator": "sparky",
        "max_parallel": max_parallel or len(hosts),
        "hosts": hosts,
        "stages": stages,
    }
    return campaign.seal_campaign_manifest_v2(body)


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _load_state(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return campaign.validate_campaign_state_v2(raw, manifest)


def test_v2_manifest_is_strict_canonical_and_preserves_v1_contract(tmp_path):
    receipt = _receipt(tmp_path / "done.json", "done")
    stage = _stage(
        tmp_path,
        stage_id="prepare",
        host_id="sparky",
        dependencies=[],
        argv=[sys.executable, "-c", _WRITE_BYTES, receipt["path"], "done"],
        receipts=[receipt],
    )
    manifest = _manifest(tmp_path, [stage])

    assert manifest["schema"] == campaign.CAMPAIGN_MANIFEST_SCHEMA_V2
    assert campaign.validate_campaign_manifest_v2(manifest) == manifest
    from prismaquant.cluster_campaign_contract import CAMPAIGN_MANIFEST_SCHEMA

    assert CAMPAIGN_MANIFEST_SCHEMA == "prismaquant.cluster_campaign.manifest.v1"

    argv_string = copy.deepcopy(manifest)
    argv_string["stages"][0]["argv"] = "python -c bad"
    argv_string["identity_sha256"] = campaign.canonical_sha256(
        {key: argv_string[key] for key in argv_string if key != "identity_sha256"}
    )
    with pytest.raises(campaign.CampaignContractError, match="argv-style"):
        campaign.validate_campaign_manifest_v2(argv_string)

    cyclic_body = {
        key: copy.deepcopy(manifest[key])
        for key in manifest
        if key != "identity_sha256"
    }
    cyclic_body["stages"][0]["dependencies"] = ["prepare"]
    with pytest.raises(campaign.CampaignContractError, match="depends on itself"):
        campaign.seal_campaign_manifest_v2(cyclic_body)

    shell_bootstrap = {
        key: copy.deepcopy(manifest[key])
        for key in manifest
        if key != "identity_sha256"
    }
    shell_bootstrap["hosts"].append(_ssh_host(tmp_path))
    shell_bootstrap["max_parallel"] = 2
    shell_bootstrap["hosts"][1]["transport"]["remote_helper_argv"] = [
        "python3;touch /tmp/owned"
    ]
    with pytest.raises(campaign.CampaignContractError, match="invalid value"):
        campaign.seal_campaign_manifest_v2(shell_bootstrap)


def test_local_and_ssh_stages_form_parallel_barrier_then_collate(tmp_path):
    local_path = tmp_path / "receipts" / "local.json"
    remote_path = tmp_path / "receipts" / "remote.json"
    collate_path = tmp_path / "receipts" / "collate.json"
    local_receipt = _receipt(local_path, "local")
    remote_receipt = _receipt(remote_path, "remote")
    collate_receipt = _receipt(collate_path, "collated")
    stages = [
        _stage(
            tmp_path,
            stage_id="render-sparky",
            host_id="sparky",
            dependencies=[],
            argv=[
                sys.executable,
                "-c",
                _WRITE_BYTES,
                str(local_path),
                "local",
            ],
            receipts=[local_receipt],
        ),
        _stage(
            tmp_path,
            stage_id="render-sparklina",
            host_id="sparklina",
            dependencies=[],
            argv=[
                sys.executable,
                "-c",
                _WRITE_BYTES,
                str(remote_path),
                "remote",
            ],
            receipts=[remote_receipt],
        ),
        # Transfer/collation has no privileged runner path.  It is one explicit
        # argv stage behind the exact two-host receipt barrier.
        _stage(
            tmp_path,
            stage_id="collate",
            host_id="sparky",
            dependencies=["render-sparky", "render-sparklina"],
            argv=[
                sys.executable,
                "-c",
                _WRITE_BYTES,
                str(collate_path),
                "collated",
            ],
            receipts=[collate_receipt],
        ),
    ]
    manifest = _manifest(tmp_path, stages, ssh=True, max_parallel=2)
    state_path = tmp_path / "campaign-state.json"

    state = campaign.run_campaign_v2(
        manifest, state_path, poll_interval=0.01
    )

    assert {row["status"] for row in state["stages"].values()} == {"succeeded"}
    starts = [
        state["stages"][stage_id]["attempts"][0]["started_unix_ns"]
        for stage_id in ("render-sparky", "render-sparklina")
    ]
    collate_start = state["stages"]["collate"]["attempts"][0]["started_unix_ns"]
    render_finishes = [
        state["stages"][stage_id]["attempts"][0]["finished_unix_ns"]
        for stage_id in ("render-sparky", "render-sparklina")
    ]
    assert abs(starts[0] - starts[1]) < 2_000_000_000
    assert collate_start >= max(render_finishes)
    assert collate_path.read_text(encoding="utf-8") == "collated"

    # A complete replay validates state and returns without launching another
    # attempt or rewriting any receipt.
    replay = campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)
    assert replay == state
    assert all(len(row["attempts"]) == 1 for row in replay["stages"].values())

    collate_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(
        campaign.CampaignTerminalFailure,
        match="collate:73",
    ):
        campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)


def test_scheduler_refills_a_freed_host_while_another_host_is_active(tmp_path):
    fast_path = tmp_path / "fast.receipt"
    gate_path = tmp_path / "gate.receipt"
    waiting_path = tmp_path / "waiting.receipt"
    wait_script = (
        "from pathlib import Path; import sys, time; "
        "gate=Path(sys.argv[1]); out=Path(sys.argv[2]); "
        "deadline=time.monotonic()+12; "
        "exec(\"while not gate.exists() and time.monotonic() < deadline:\\n"
        " time.sleep(0.02)\"); "
        "sys.exit(8) if not gate.exists() else out.write_bytes(b'waiting')"
    )
    stages = [
        _stage(
            tmp_path,
            stage_id="fast-local",
            host_id="sparky",
            dependencies=[],
            argv=[sys.executable, "-c", _WRITE_BYTES, str(fast_path), "fast"],
            receipts=[_receipt(fast_path, "fast")],
        ),
        _stage(
            tmp_path,
            stage_id="waiting-remote",
            host_id="sparklina",
            dependencies=[],
            argv=[
                sys.executable,
                "-c",
                wait_script,
                str(gate_path),
                str(waiting_path),
            ],
            receipts=[_receipt(waiting_path, "waiting")],
            max_attempts=1,
            timeout_seconds=20,
        ),
        _stage(
            tmp_path,
            stage_id="refill-local",
            host_id="sparky",
            dependencies=["fast-local"],
            argv=[sys.executable, "-c", _WRITE_BYTES, str(gate_path), "gate"],
            receipts=[_receipt(gate_path, "gate")],
        ),
    ]
    manifest = _manifest(tmp_path, stages, ssh=True, max_parallel=2)

    state = campaign.run_campaign_v2(
        manifest,
        tmp_path / "refill-state.json",
        poll_interval=0.01,
    )

    waiting_attempt = state["stages"]["waiting-remote"]["attempts"]
    assert len(waiting_attempt) == 1
    refill_finished = state["stages"]["refill-local"]["attempts"][0][
        "finished_unix_ns"
    ]
    assert waiting_attempt[0]["finished_unix_ns"] >= refill_finished


def test_bounded_retry_reuses_exact_receipt_gate(tmp_path):
    marker = tmp_path / "attempted-once"
    receipt_path = tmp_path / "retry-receipt.json"
    receipt = _receipt(receipt_path, "accepted")
    script = (
        "from pathlib import Path; import sys; "
        "marker=Path(sys.argv[1]); out=Path(sys.argv[2]); "
        "first=not marker.exists(); marker.write_text('seen'); "
        "sys.exit(9) if first else out.write_bytes(b'accepted')"
    )
    stage = _stage(
        tmp_path,
        stage_id="retryable-render",
        host_id="sparky",
        dependencies=[],
        argv=[sys.executable, "-c", script, str(marker), str(receipt_path)],
        receipts=[receipt],
        max_attempts=2,
    )
    manifest = _manifest(tmp_path, [stage])
    state_path = tmp_path / "retry-state.json"

    state = campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)

    attempts = state["stages"]["retryable-render"]["attempts"]
    assert [item["status"] for item in attempts] == ["failed", "succeeded"]
    assert [item["number"] for item in attempts] == [1, 2]
    assert receipt_path.read_bytes() == b"accepted"


def test_sealed_stage_wraps_dynamic_output_and_withholds_receipt_on_failure(
    tmp_path,
):
    dynamic_output = tmp_path / "dynamic-plan.json"
    receipt_path = tmp_path / "receipts" / "plan.complete.json"
    child_argv = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "Path(sys.argv[1]).write_text(str(time.time_ns()))"
        ),
        str(dynamic_output),
    ]
    token = "qwen38-125b:materialize-plan"
    receipt = {
        "path": str(receipt_path.resolve()),
        "sha256": campaign.sealed_stage_receipt_sha256(token, child_argv),
    }
    wrapped_argv = [
        sys.executable,
        str(_RUNNER_TOOL),
        "sealed-stage",
        "--receipt",
        str(receipt_path),
        "--token",
        token,
        "--",
        *child_argv,
    ]
    manifest = _manifest(
        tmp_path,
        [
            _stage(
                tmp_path,
                stage_id="materialize-plan",
                host_id="sparky",
                dependencies=[],
                argv=wrapped_argv,
                receipts=[receipt],
            )
        ],
    )

    state = campaign.run_campaign_v2(
        manifest,
        tmp_path / "sealed-state.json",
        poll_interval=0.01,
    )

    assert state["stages"]["materialize-plan"]["status"] == "succeeded"
    assert dynamic_output.read_text(encoding="utf-8").isdigit()
    expected = campaign.sealed_stage_receipt_bytes(token, child_argv)
    assert receipt_path.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == receipt["sha256"]

    hash_command = subprocess.run(
        [
            sys.executable,
            str(_RUNNER_TOOL),
            "sealed-stage-receipt-sha256",
            "--token",
            token,
            "--",
            *child_argv,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert hash_command.stdout.strip() == receipt["sha256"]

    failed_receipt = tmp_path / "receipts" / "failed.complete.json"
    returncode = campaign.run_sealed_stage(
        receipt_path=failed_receipt,
        token="qwen38-125b:failing-plan",
        child_argv=[sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert returncode == 7
    assert not failed_receipt.exists()


def test_wrong_existing_receipt_is_terminal_and_command_never_runs(tmp_path):
    marker = tmp_path / "command-ran"
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("wrong", encoding="utf-8")
    receipt = _receipt(receipt_path, "right")
    stage = _stage(
        tmp_path,
        stage_id="fail-closed",
        host_id="sparky",
        dependencies=[],
        argv=[
            sys.executable,
            "-c",
            _WRITE_BYTES,
            str(marker),
            "should-not-run",
        ],
        receipts=[receipt],
        max_attempts=3,
    )
    manifest = _manifest(tmp_path, [stage])
    state_path = tmp_path / "mismatch-state.json"

    with pytest.raises(campaign.CampaignTerminalFailure, match="fail-closed"):
        campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)

    state = _load_state(state_path, manifest)
    row = state["stages"]["fail-closed"]
    assert row["status"] == "terminal_failed"
    assert len(row["attempts"]) == 1
    assert not marker.exists()
    assert receipt_path.read_text(encoding="utf-8") == "wrong"


def test_terminal_failure_stops_other_owned_active_helpers(tmp_path):
    wrong_path = tmp_path / "bad.receipt"
    wrong_path.write_bytes(b"wrong")
    slow_path = tmp_path / "slow.receipt"
    slow_script = (
        "from pathlib import Path; import sys, time; "
        "time.sleep(30); Path(sys.argv[1]).write_bytes(b'slow')"
    )
    stages = [
        _stage(
            tmp_path,
            stage_id="bad-local",
            host_id="sparky",
            dependencies=[],
            argv=[sys.executable, "-c", "raise SystemExit(91)"],
            receipts=[_receipt(wrong_path, "expected")],
            max_attempts=1,
        ),
        _stage(
            tmp_path,
            stage_id="slow-remote",
            host_id="sparklina",
            dependencies=[],
            argv=[sys.executable, "-c", slow_script, str(slow_path)],
            receipts=[_receipt(slow_path, "slow")],
            max_attempts=2,
            timeout_seconds=60,
        ),
    ]
    manifest = _manifest(tmp_path, stages, ssh=True, max_parallel=2)
    state_path = tmp_path / "stop-active-state.json"

    with pytest.raises(campaign.CampaignTerminalFailure, match="bad-local"):
        campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)

    state = _load_state(state_path, manifest)
    assert state["stages"]["bad-local"]["status"] == "terminal_failed"
    assert state["stages"]["slow-remote"]["status"] == "terminal_failed"
    assert "owned helper stopped" in state["stages"]["slow-remote"][
        "attempts"
    ][0]["detail"]
    time.sleep(0.1)
    assert not slow_path.exists()


def test_state_hash_tamper_and_concurrent_coordinator_lock_are_refused(tmp_path):
    receipt_path = tmp_path / "done.json"
    receipt = _receipt(receipt_path, "done")
    stage = _stage(
        tmp_path,
        stage_id="done",
        host_id="sparky",
        dependencies=[],
        argv=[sys.executable, "-c", _WRITE_BYTES, str(receipt_path), "done"],
        receipts=[receipt],
    )
    manifest = _manifest(tmp_path, [stage])
    state_path = tmp_path / "state.json"
    campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)

    with campaign._StateStore(state_path) as first:
        assert first.load(manifest) is not None
        with pytest.raises(campaign.CampaignLockBusy, match="another coordinator"):
            with campaign._StateStore(state_path):
                pass

    state_alias = tmp_path / "state-alias.json"
    state_alias.symlink_to(state_path)
    with pytest.raises(campaign.CampaignContractError, match="symlink"):
        campaign.run_campaign_v2(manifest, state_alias, poll_interval=0.01)

    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["revision"] += 1
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(campaign.CampaignContractError, match="identity_sha256"):
        campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)


def test_resume_after_coordinator_death_recovers_owned_helper_receipt(tmp_path):
    receipt_path = tmp_path / "slow-receipt.json"
    receipt = _receipt(receipt_path, "slow-done")
    script = (
        "from pathlib import Path; import sys, time; "
        "time.sleep(1.0); Path(sys.argv[1]).write_bytes(b'slow-done')"
    )
    stage = _stage(
        tmp_path,
        stage_id="slow-stage",
        host_id="sparky",
        dependencies=[],
        argv=[sys.executable, "-c", script, str(receipt_path)],
        receipts=[receipt],
        max_attempts=2,
        timeout_seconds=10,
    )
    manifest = _manifest(tmp_path, [stage])
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "recovery-state.json"
    _write_manifest(manifest_path, manifest)

    coordinator = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "prismaquant.cluster_campaign",
            "run",
            "--manifest",
            str(manifest_path),
            "--state",
            str(state_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 8.0
    observed_running = False
    while time.monotonic() < deadline:
        if state_path.is_file():
            try:
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                attempt = raw["stages"]["slow-stage"]["attempts"][-1]
                observed_running = (
                    raw["stages"]["slow-stage"]["status"] == "running"
                    and attempt["pid"] is not None
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                observed_running = False
            if observed_running:
                break
        time.sleep(0.02)
    assert observed_running, (
        "coordinator never recorded slow-stage as running with a pid; "
        f"state={state_path.read_text(encoding='utf-8') if state_path.is_file() else '(absent)'}"
    )
    coordinator.terminate()
    coordinator.wait(timeout=5)

    # Wait for the orphaned helper to finish, on its receipt rather than on
    # the clock. What this test asserts is that a resume ADOPTS a completed
    # helper's work instead of redoing it -- so the helper has to have
    # completed. Probing the instant the coordinator dies asserts something
    # else: that the helper wins a race against the resume. It usually did,
    # and lost under load, which is how this arrived in CI as a flake.
    helper_deadline = time.monotonic() + 30.0
    while time.monotonic() < helper_deadline and not receipt_path.is_file():
        time.sleep(0.02)
    assert receipt_path.is_file(), (
        "the orphaned helper never wrote its receipt, so there is nothing "
        "for the resume to adopt -- the coordinator's termination most "
        "likely took the helper with it"
    )

    state = campaign.run_campaign_v2(
        manifest, state_path, poll_interval=0.02
    )

    attempts = state["stages"]["slow-stage"]["attempts"]
    assert len(attempts) == 1, (
        "the resume started a second attempt instead of adopting the "
        f"completed helper's work: {attempts}"
    )
    assert attempts[0]["status"] == "succeeded"
    assert "exact receipts recovered" in attempts[0]["detail"]
    assert receipt_path.read_bytes() == b"slow-done"


def test_abrupt_worker_death_leaves_stage_lock_owned_by_child(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    receipt_path = tmp_path / "orphan.receipt"
    receipt = _receipt(receipt_path, "orphan-complete")
    script = (
        "from pathlib import Path; import os, sys, time; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(2.5); "
        "Path(sys.argv[2]).write_bytes(b'orphan-complete')"
    )
    stage = _stage(
        tmp_path,
        stage_id="lock-survives-worker",
        host_id="sparky",
        dependencies=[],
        argv=[
            sys.executable,
            "-c",
            script,
            str(child_pid_path),
            str(receipt_path),
        ],
        receipts=[receipt],
        timeout_seconds=10,
    )
    manifest = _manifest(tmp_path, [stage])
    owner = campaign._owner_token(
        manifest["identity_sha256"], "lock-survives-worker", 1
    )
    request = campaign._worker_request(
        manifest,
        manifest["hosts"][0],
        manifest["stages"][0],
        owner_token=owner,
        verify_only=False,
    )
    worker = subprocess.Popen(
        [sys.executable, str(_RUNNER_TOOL), "_exec-request"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child_pid = None
    child_ticks = None
    try:
        assert worker.stdin is not None
        worker.stdin.write(json.dumps(request).encode("utf-8"))
        worker.stdin.close()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not child_pid_path.is_file():
            time.sleep(0.02)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        snapshot = campaign._proc_snapshot(child_pid)
        assert snapshot is not None
        child_ticks = snapshot[0]

        worker.kill()
        worker.wait(timeout=5)
        with pytest.raises(TimeoutError, match="worker lock"):
            campaign._worker_lock(
                Path(request["lock_path"]),
                Path(request["work_root"]),
                timeout_seconds=1,
            )

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not receipt_path.is_file():
            time.sleep(0.02)
        assert receipt_path.read_bytes() == b"orphan-complete"
        lock_fd = campaign._worker_lock(
            Path(request["lock_path"]),
            Path(request["work_root"]),
            timeout_seconds=1,
        )
        os.close(lock_fd)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        if child_pid is not None and child_ticks is not None:
            snapshot = campaign._proc_snapshot(child_pid)
            if snapshot is not None and snapshot[0] == child_ticks:
                try:
                    os.killpg(child_pid, 9)
                except ProcessLookupError:
                    pass


def test_ambiguous_recorded_pid_fails_closed_without_killing_it(tmp_path):
    receipt_path = tmp_path / "never.json"
    receipt = _receipt(receipt_path, "never")
    stage = _stage(
        tmp_path,
        stage_id="ambiguous",
        host_id="sparky",
        dependencies=[],
        argv=[sys.executable, "-c", _WRITE_BYTES, str(receipt_path), "never"],
        receipts=[receipt],
        max_attempts=2,
    )
    manifest = _manifest(tmp_path, [stage])
    state = campaign.initial_campaign_state_v2(manifest)
    ticks, proc_state = campaign._proc_snapshot(os.getpid())
    assert proc_state != "Z"
    body = {key: copy.deepcopy(state[key]) for key in campaign._STATE_BODY_KEYS}
    owner = campaign._owner_token(
        manifest["identity_sha256"], "ambiguous", 1
    )
    log_path = (tmp_path / "ambiguous.log").resolve()
    log_path.write_bytes(b"")
    body["revision"] = 1
    body["stages"]["ambiguous"] = {
        "status": "running",
        "attempts": [
            {
                "number": 1,
                "owner_token": owner,
                "status": "running",
                "pid": os.getpid(),
                "pid_start_ticks": ticks,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "log_path": str(log_path),
                "detail": "",
            }
        ],
        "receipts": [],
    }
    running = campaign._seal_state(body)
    campaign.validate_campaign_state_v2(running, manifest)
    state_path = tmp_path / "ambiguous-state.json"
    state_path.write_text(json.dumps(running), encoding="utf-8")

    with pytest.raises(campaign.CampaignTerminalFailure, match="ambiguous"):
        campaign.run_campaign_v2(manifest, state_path, poll_interval=0.01)

    assert os.getpid() > 0
    final = _load_state(state_path, manifest)
    attempt = final["stages"]["ambiguous"]["attempts"][0]
    assert final["stages"]["ambiguous"]["status"] == "terminal_failed"
    assert "ownership is ambiguous" in attempt["detail"]
    assert not receipt_path.exists()
