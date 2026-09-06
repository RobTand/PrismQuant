from __future__ import annotations

from collections.abc import Callable
import contextlib
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


# ---------------------------------------------------------------------------
# Cross-process waits.
#
# Every wait in this file is "a child process will make an observable thing
# appear" -- a state file, a pid file, a receipt.  What that costs is dominated
# by interpreter start plus this package's import graph, not by the work the
# child was asked to do.  Measured on sparklina (2026-09-05, CPU only): a cold
# `python -m prismaquant.cluster_campaign` reached its first state file in
# 2.7s on an idle box and in 10.9 / 17.3 / 19.3s at 1-minute load 41 / 76 / 90.
#
# A wall-clock constant sized against the first number is a coin flip under the
# second, which is what issue #199 measured and what turned three tests in this
# file red under `-n 2` on a loaded box.  So no wait here is sized in seconds.
# Each waits while the producer can still satisfy it and gives up the moment it
# cannot.  A slow box makes this file slower; it cannot make it red.  A
# producer that dies without doing the work, which is the failure these tests
# are actually about, still fails at once and by name.
# ---------------------------------------------------------------------------

_WEDGE_BACKSTOP_SECONDS = 120
"""Last-resort bound on every cross-process wait and stage in this file.

Not a budget for the work.  It answers only "the producer is wedged and nothing
will ever satisfy this", so it sits far above anything a healthy run reaches --
the slowest wait measured under load was 19.3s.  A healthy run never observes
it; a run that does has found a real defect, and reports it as one.
"""


def _wait_until(
    condition: Callable[[], bool],
    *,
    unmet: str,
    alive: Callable[[], bool] | None = None,
    diagnose: Callable[[], str] | None = None,
    poll_interval: float = 0.02,
) -> None:
    """Wait for ``condition`` for as long as its producer can still satisfy it.

    ``alive`` reports whether the process expected to satisfy ``condition`` is
    still running.  When it stops being true the condition gets one last look --
    the producer may have done the work in the window before it exited -- and
    then the wait fails, because nothing can satisfy it any more.  Omit
    ``alive`` only where the test holds no handle on the producer, such as a
    deliberately orphaned helper; the backstop is then the sole bound.

    ``unmet`` is the failure sentence and ``diagnose`` appends state to it.
    """
    backstop = time.monotonic() + _WEDGE_BACKSTOP_SECONDS

    def _fail(why: str) -> None:
        detail = f"; {diagnose()}" if diagnose is not None else ""
        raise AssertionError(f"{unmet} -- {why}{detail}")

    while True:
        if condition():
            return
        if alive is not None and not alive():
            if condition():
                return
            _fail("its producer exited first")
        if time.monotonic() >= backstop:
            _fail(
                f"nothing happened in {_WEDGE_BACKSTOP_SECONDS}s, so the "
                "producer is wedged"
            )
        time.sleep(poll_interval)


def _nonempty(path: Path) -> Callable[[], bool]:
    """Predicate for "this file exists *and* its bytes have landed".

    A file becomes visible in its directory before the writer's bytes reach it,
    so waiting on existence alone hands the next line a half-written file.  That
    is not theoretical: with the waits above fixed, `-n 2` under load got as far
    as reading `child.pid` and raised `ValueError: invalid literal for int() with
    base 10: ''`.  Every producer here writes its file in one `write_text` /
    `write_bytes` call, so non-empty is the same thing as complete.
    """

    def _ready() -> bool:
        try:
            return path.stat().st_size > 0
        except OSError:
            return False

    return _ready


# ---------------------------------------------------------------------------
# Children this file deliberately orphans.
#
# Two tests here kill the process that would otherwise enforce the stage's
# `timeout_seconds` -- the coordinator in one, the worker in the other -- and
# then rely on the stage child polling a release file at 50 Hz.  Once its
# enforcer is dead, writing that file is the ONLY thing left that can ever end
# the child (`cluster_campaign.py:1164` gives it its own session; the worker's
# `TimeoutExpired` -> `_terminate_child_process_group` is what kills it in a
# healthy run, and that worker is gone by construction).
#
# So "the release always happens" is not part of either test's arrangement: it
# is the difference between a test that failed and a `python -c` that spins on
# a release file which, once `tmp_path` is cleaned up, can never appear
# (RobTand/prismaquant#219).  One home for it, below, rather than a copy per
# test -- and the exit is asserted, not merely arranged, because "this test
# leaves nothing running" is a property of the test that a shared box pays for.
# ---------------------------------------------------------------------------


class _OrphanedHelper:
    """The deliberately orphaned stage child, and its release file."""

    def __init__(self, release_path: Path) -> None:
        self.release_path = release_path
        self.pid: int | None = None
        self._ticks: int | None = None

    def adopt(self, pid: int) -> None:
        """Take the handle the child announced, with its start time, so a
        recycled pid can never be mistaken for it."""
        snapshot = campaign._proc_snapshot(pid)
        assert snapshot is not None, (
            f"the orphaned helper announced pid {pid}, which is not running"
        )
        self.pid, self._ticks = pid, snapshot[0]

    def release(self) -> None:
        """Let the child finish.  Idempotent, so a test can release in line
        where the ordering is load-bearing and still be released on the way
        out of any path that never reached that line."""
        self.release_path.write_bytes(b"go")

    def running(self) -> bool:
        if self.pid is None or self._ticks is None:
            return False
        snapshot = campaign._proc_snapshot(self.pid)
        return (snapshot is not None and snapshot[0] == self._ticks
                and snapshot[1] != "Z")

    def assert_exited(self) -> None:
        """The property under test, not a side effect: nothing this test
        started is still running when it ends."""
        _wait_until(
            lambda: not self.running(),
            unmet=(
                f"the deliberately orphaned helper (pid {self.pid}) is still "
                "running after its release file landed, so this test would "
                "have leaked a spinning process onto the box"
            ),
        )


@contextlib.contextmanager
def _orphaned_helper(release_path: Path):
    """Own the release file for a child a test orphans on purpose.

    On EVERY path out -- a passing test, a failing assertion, a wedged wait, a
    `KeyboardInterrupt`, an interpreter dying between two lines -- the child is
    released, and killed outright if it did not take the release.
    """
    helper = _OrphanedHelper(release_path)
    try:
        yield helper
    finally:
        helper.release()
        if helper.running():
            try:
                if os.getpgid(helper.pid) == helper.pid:
                    os.killpg(helper.pid, 9)  # its own session: take the group
                else:
                    os.kill(helper.pid, 9)
            except (ProcessLookupError, PermissionError, OSError):
                pass


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
    # The runner's own bound on the stage argv, and the same rule as the waits
    # above: no test in this file asserts that a stage times out, so this is a
    # wedge detector rather than a budget.  Every stage argv here is a trivial
    # `python -c` or the sealed-stage wrapper, whose cost is interpreter start
    # rather than the work in the argv -- which is why the previous default of
    # 10s terminal-failed materialize-plan under load, where the wrapper pays a
    # cold prismaquant import and then starts a second interpreter.
    timeout_seconds: int = _WEDGE_BACKSTOP_SECONDS,
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
    # No private deadline in the child: the stage's own timeout_seconds is the
    # authoritative bound and the worker enforces it by terminating the child's
    # process group (cluster_campaign.py, `_exec-request`). A second, shorter
    # guess in here only turned a slow gate into `exit 8` -- a terminal stage
    # failure -- on a box where the gate stage was merely still starting up.
    wait_script = (
        "from pathlib import Path; import sys, time; "
        "gate=Path(sys.argv[1]); out=Path(sys.argv[2]); "
        "exec(\"while not gate.exists():\\n time.sleep(0.02)\"); "
        "out.write_bytes(b'waiting')"
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
    started_path = tmp_path / "helper-started"
    release_path = tmp_path / "release-the-helper"
    # The helper announces itself, then waits for this test to release it,
    # instead of sleeping 1.0s.  Two races die with that sleep, and both are
    # the load-sensitive kind:
    #
    #  * The state records `running` with a pid as soon as the coordinator has
    #    *spawned* its worker (cluster_campaign.py `bind_process`), which is
    #    before the worker has been handed its request on stdin.  Terminating
    #    the coordinator in that window closed the pipe, the worker exited at
    #    EOF without ever launching a stage child, and no receipt was ever
    #    written.  Waiting for the helper to announce itself proves the child
    #    exists before the coordinator is allowed to die.
    #  * Once the child did exist, the test had whatever was left of its 1.0s
    #    sleep to kill the coordinator.  Lose that and the coordinator finishes
    #    the campaign normally, so the resume has nothing to adopt and the
    #    "exact receipts recovered" assertion fails instead.
    #
    # It announces its PID rather than a marker word, so this test can hold a
    # handle on the child it is about to orphan and assert it is gone at the
    # end instead of hoping (#219).
    script = (
        "from pathlib import Path; import os, sys, time; "
        "release=Path(sys.argv[2]); "
        "Path(sys.argv[3]).write_text(str(os.getpid())); "
        "exec(\"while not release.exists():\\n time.sleep(0.02)\"); "
        "Path(sys.argv[1]).write_bytes(b'slow-done')"
    )
    stage = _stage(
        tmp_path,
        stage_id="slow-stage",
        host_id="sparky",
        dependencies=[],
        argv=[
            sys.executable,
            "-c",
            script,
            str(receipt_path),
            str(release_path),
            str(started_path),
        ],
        receipts=[receipt],
        max_attempts=2,
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
    def _recorded_running_with_pid() -> bool:
        if not state_path.is_file():
            return False
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            attempt = raw["stages"]["slow-stage"]["attempts"][-1]
        except (json.JSONDecodeError, KeyError, IndexError):
            return False
        return (
            raw["stages"]["slow-stage"]["status"] == "running"
            and attempt["pid"] is not None
        )

    # Everything from here on runs with the helper's release owned by the
    # context manager, because from the moment the coordinator dies nothing
    # else can end that child: an assertion that fails, a wedged wait, or the
    # session being killed between `terminate()` and the release used to leave
    # it polling a file that `tmp_path` cleanup then made unreachable (#219).
    with _orphaned_helper(release_path) as helper:
        # Nearly all of this latency is the coordinator's cold start rather
        # than its campaign work, so the wait is on the coordinator, not on a
        # clock.
        _wait_until(
            _recorded_running_with_pid,
            unmet="coordinator never recorded slow-stage as running with a pid",
            alive=lambda: coordinator.poll() is None,
            diagnose=lambda: "state="
            + (
                state_path.read_text(encoding="utf-8")
                if state_path.is_file()
                else "(absent)"
            ),
        )
        # ...and the stage child must actually exist before the coordinator is
        # allowed to die, or the worker exits at EOF with no work to do and
        # there is never a receipt to adopt.  The coordinator has to stay alive
        # to get the worker that far, so this wait is bounded by it too.
        _wait_until(
            _nonempty(started_path),
            unmet="the coordinator never got a stage child running",
            alive=lambda: coordinator.poll() is None,
        )
        helper.adopt(int(started_path.read_text(encoding="utf-8").strip()))

        try:
            coordinator.terminate()
            _wait_until(
                lambda: coordinator.poll() is not None,
                unmet="coordinator did not exit after terminate()",
            )

            # Now let the orphaned helper finish, and wait on its receipt
            # rather than on the clock. What this test asserts is that a resume
            # ADOPTS a completed helper's work instead of redoing it -- so the
            # helper has to have completed. Probing the instant the coordinator
            # dies asserts something else: that the helper wins a race against
            # the resume. It usually did, and lost under load, which is how
            # this arrived in CI as a flake.  The release is written here, and
            # again by `_orphaned_helper` on the way out, because the ordering
            # matters on the passing path and the guarantee matters on all of
            # them.
            helper.release()
            _wait_until(
                _nonempty(receipt_path),
                unmet=(
                    "the orphaned helper never wrote its receipt, so there is "
                    "nothing for the resume to adopt -- the coordinator's "
                    "termination most likely took the helper with it"
                ),
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
            # The child this test orphaned on purpose has to be gone before
            # the test is allowed to pass.
            helper.assert_exited()
        finally:
            if coordinator.poll() is None:
                coordinator.kill()
                _wait_until(
                    lambda: coordinator.poll() is not None,
                    unmet="coordinator did not exit after kill() during cleanup",
                )


def test_abrupt_worker_death_leaves_stage_lock_owned_by_child(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    receipt_path = tmp_path / "orphan.receipt"
    release_path = tmp_path / "release-the-child"
    receipt = _receipt(receipt_path, "orphan-complete")
    # The child holds the stage lock until this test releases it, rather than
    # for a fixed 2.5s.  The window this test needs -- kill the worker, prove
    # the lock is still held -- was previously whatever was left of that sleep
    # after a cold interpreter start, which is the same load-sensitive budget
    # as the deadlines this file just lost.
    script = (
        "from pathlib import Path; import os, sys, time; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "release=Path(sys.argv[3]); "
        "exec(\"while not release.exists():\\n time.sleep(0.02)\"); "
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
            str(release_path),
        ],
        receipts=[receipt],
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
    # `_orphaned_helper` owns "the child is unblocked, and gone, on every path
    # out"; this test owns only its own worker (#219).
    with _orphaned_helper(release_path) as helper:
        try:
            assert worker.stdin is not None
            worker.stdin.write(json.dumps(request).encode("utf-8"))
            worker.stdin.close()
            _wait_until(
                _nonempty(child_pid_path),
                unmet="the worker never launched the stage child",
                alive=lambda: worker.poll() is None,
            )
            child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
            helper.adopt(child_pid)

            worker.kill()
            _wait_until(
                lambda: worker.poll() is not None,
                unmet="worker did not exit after kill()",
            )
            # The child is still holding the lock by construction -- it is
            # blocked on the release file -- so a short timeout here is the
            # assertion, not a budget: the lock must refuse to open promptly.
            with pytest.raises(TimeoutError, match="worker lock"):
                campaign._worker_lock(
                    Path(request["lock_path"]),
                    Path(request["work_root"]),
                    timeout_seconds=1,
                )

            helper.release()
            # The stage child outlived its worker in its own process group, so
            # its liveness is a pid probe rather than a Popen handle.  Waiting
            # on the receipt to exist also keeps a slow box from turning this
            # into a FileNotFoundError instead of a named failure.
            _wait_until(
                _nonempty(receipt_path),
                unmet="the orphaned stage child never wrote its receipt",
                alive=lambda: campaign._proc_snapshot(child_pid) is not None,
            )
            assert receipt_path.read_bytes() == b"orphan-complete"
            # The child releases the lock by exiting, which it has not
            # necessarily done at the instant the receipt lands.  Bound that on
            # the backstop rather than on a guess at how long an interpreter
            # takes to shut down.
            lock_fd = campaign._worker_lock(
                Path(request["lock_path"]),
                Path(request["work_root"]),
                timeout_seconds=_WEDGE_BACKSTOP_SECONDS,
            )
            os.close(lock_fd)
            helper.assert_exited()
        finally:
            if worker.poll() is None:
                worker.kill()
                _wait_until(
                    lambda: worker.poll() is not None,
                    unmet="worker did not exit after kill() during cleanup",
                )


@pytest.mark.parametrize(
    "second_stat, expected",
    [
        ("42 Z", "gone"),
        (FileNotFoundError(), "gone"),
        (ProcessLookupError(), "gone"),
        ("42 S", "mismatch"),
        ("43 S", "mismatch"),
        ("43 Z", "mismatch"),
        (PermissionError(), "mismatch"),
        (OSError("stat I/O failed"), "mismatch"),
        ("malformed", "mismatch"),
        ("invalid S", "mismatch"),
    ],
    ids=[
        "zombie", "missing", "esrch", "live-owner-mismatch", "pid-reused",
        "pid-reused-zombie", "permission-denied", "io-error", "malformed",
        "invalid-ticks",
    ],
)
@pytest.mark.parametrize("owner_read_error", [False, True], ids=["empty-env", "env-error"])
def test_owned_process_state_rechecks_exit_after_failed_owner_read(
    monkeypatch, second_stat, expected, owner_read_error
):
    # Force the exact exit window: stat reports our live helper, then environ
    # loses its owner as the helper exits. No scheduling or sleep is involved.
    snapshots = iter(["42 S", second_stat])

    def read_stat(path, **kwargs):
        assert str(path) == "/proc/424242/stat"
        value = next(snapshots)
        if isinstance(value, OSError):
            raise value
        if value == "malformed":
            return value
        ticks, state = value.split()
        return "424242 (helper) " + " ".join([state] + ["0"] * 18 + [ticks])

    def read_environ(path):
        assert str(path) == "/proc/424242/environ"
        if owner_read_error:
            raise ProcessLookupError()
        return b""

    monkeypatch.setattr(Path, "read_text", read_stat)
    monkeypatch.setattr(Path, "read_bytes", read_environ)
    assert campaign._owned_process_state(424242, 42, "owner") == expected


@pytest.mark.parametrize("owner_matches, expected", [(True, "owned"), (False, "mismatch")])
def test_owned_process_state_live_identity(monkeypatch, owner_matches, expected):
    monkeypatch.setattr(campaign, "_proc_snapshot", lambda *a, **kw: (42, "S"))
    monkeypatch.setattr(campaign, "_proc_has_owner", lambda *a: owner_matches)
    assert campaign._owned_process_state(424242, 42, "owner") == expected


@pytest.mark.parametrize("state", ["S", "Z"])
def test_recorded_process_pid_reuse_is_never_owned_or_killed(monkeypatch, state):
    monkeypatch.setattr(campaign, "_proc_snapshot", lambda *a, **kw: (43, state))
    monkeypatch.setattr(
        campaign, "_proc_has_owner",
        lambda *a: pytest.fail("a reused PID must not be checked for ownership"),
    )
    monkeypatch.setattr(
        os, "killpg", lambda *a: pytest.fail("a reused PID must not be killed")
    )
    assert campaign._owned_process_state(424242, 42, "owner") == "mismatch"
    assert not campaign._terminate_recorded_owned_process(424242, 42, "owner")


@pytest.mark.parametrize("error", [PermissionError(), OSError("stat I/O failed")])
def test_recorded_process_unreadable_stat_is_never_killed(monkeypatch, error):
    def read_stat(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", read_stat)
    monkeypatch.setattr(
        os, "killpg", lambda *a: pytest.fail("unreadable identity must not be killed")
    )
    assert campaign._owned_process_state(424242, 42, "owner") == "mismatch"
    assert not campaign._terminate_recorded_owned_process(424242, 42, "owner")


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
