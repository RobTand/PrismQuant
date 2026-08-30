"""Pull-queue invariants in tools/pqwork.py.

The tool exists because agent-dispatched work stopped when the agent did.
Every property pinned here is one the recovery story rests on: a claim that
two boxes can both win, a lease that never expires, or a "done" that trusts
an exit code all put the queue back to losing work silently.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest

# These tests exercise repo-root `tools/` scripts, which are not part of the
# installed package, so they can only run from a checkout. The release
# pipeline runs the suite against a non-editable install from outside the
# checkout (docs/RELEASING.md); skip there instead of failing collection.
if not (Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "pqwork.py"
_DEPLOY_TOOL_PATH = (Path(__file__).resolve().parents[1]
                     / "tools" / "deploy_pqwork.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("pqwork", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pqwork = _load_tool()


def _load_deploy_tool():
    spec = importlib.util.spec_from_file_location("deploy_pqwork",
                                                  _DEPLOY_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pqdeploy = _load_deploy_tool()


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    """Redirect the queue and $HOME into the sandbox.

    `QUEUE_ROOT` is read into a module constant at import time, so patching
    the env var alone would still let a test scribble on the real
    /mnt/shared/pq-queue. $HOME is patched too because `Job` creates
    the child's TMPDIR under it, and because `bash -lc` sources login
    profiles that could otherwise reach into the real dotfiles.
    """
    root = tmp_path / "pq-queue"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PQ_QUEUE_ROOT", str(root))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(pqwork, "QUEUE_ROOT", root)
    pqwork.ensure_layout()
    return root


def enqueue(item_id, cmd="true", *, cwd=None, **kw):
    """Write a ready item with the same shape `cmd_enqueue` produces."""
    item = {
        "id": item_id,
        "desc": "",
        "cmd": cmd,
        "cwd": str(cwd) if cwd else str(Path.home()),
        "hosts": None,
        "requires": [],
        "needs_gpu": False,
        "receipt": None,
        "after": [],
        "priority": 50,
        "timeout_s": None,
        "max_attempts": 3,
        "env": {},
        "enqueued_at": time.time(),
        "attempts": 0,
    }
    item.update(kw)
    pqwork.write_json_atomic(pqwork.item_path(pqwork.READY, item_id), item)
    return item


def state_of(item_id):
    for state in pqwork.STATE_DIRS:
        if pqwork.item_path(state, item_id).exists():
            return state
    return None


def read_state(state, item_id):
    return json.loads(pqwork.item_path(state, item_id).read_text())


def backdate_lease(item_id, seconds):
    lease = json.loads(pqwork.lease_path(item_id).read_text())
    lease["beat"] = time.time() - seconds
    pqwork.write_json_atomic(pqwork.lease_path(item_id), lease)


def host_resource(host, *, gpu=6, mem=115.0,
                  gpu_capacity=6, mem_capacity=121.6):
    return {
        "host": host,
        "gpu_slots": gpu,
        "mem_gb": mem,
        "gpu_capacity": gpu_capacity,
        "mem_capacity_gb": mem_capacity,
    }


# --------------------------------------------------------------------------
# 1. receipt gating beats exit code


def _run_sync(item, host="mine"):
    """Execute one claimed item to completion and return its terminal state.

    The worker runs jobs on threads so it can police memory while they run,
    so there is no synchronous entry point in the tool itself. Driving the
    real ``Job`` and joining keeps these tests on the production path rather
    than on a reimplementation of it.
    """
    job = pqwork.Job(item, host)
    job.start()
    job.thread.join()
    return job.state


def test_exit_zero_without_receipt_is_not_done(queue, tmp_path):
    # A systemd unit once reported success for work that never ran; the
    # receipt is the only evidence the artifact actually landed.
    receipt = tmp_path / "never-written"
    enqueue("noreceipt", cmd="true", receipt=str(receipt), max_attempts=1,
            cwd=tmp_path)
    item = pqwork.try_claim("noreceipt", "mine")
    assert _run_sync(item) == "failed"
    assert state_of("noreceipt") == pqwork.FAILED
    assert read_state(pqwork.FAILED, "noreceipt")["exit_code"] == 0
    assert read_state(pqwork.FAILED, "noreceipt")["outcome"].startswith("no_receipt")


def test_exit_zero_without_receipt_retries_before_failing(queue, tmp_path):
    # With attempts left the same evidence gap requeues rather than failing,
    # so the "not done" property is independent of the retry budget.
    enqueue("retryme", cmd="true", receipt=str(tmp_path / "nope"),
            max_attempts=3, cwd=tmp_path)
    item = pqwork.try_claim("retryme", "mine")
    assert _run_sync(item) == "requeued"
    assert state_of("retryme") == pqwork.READY


def test_nonzero_exit_with_receipt_is_done(queue, tmp_path):
    # The converse: the exit code is not the authority in either direction.
    receipt = tmp_path / "landed"
    enqueue("badrc", cmd=f"echo landed > {receipt}; exit 3", receipt=str(receipt),
            cwd=tmp_path)
    item = pqwork.try_claim("badrc", "mine")
    assert _run_sync(item) == "done"
    assert state_of("badrc") == pqwork.DONE
    assert read_state(pqwork.DONE, "badrc")["exit_code"] == 3


# --------------------------------------------------------------------------
# 2. skip-if-done


def test_existing_receipt_skips_execution(queue, tmp_path):
    # This is what makes a stale-lease requeue safe: re-running finished
    # work must be a no-op, not a second execution.
    receipt = tmp_path / "already"
    receipt.write_text("done earlier")
    sentinel = tmp_path / "ran"
    enqueue("skipme", cmd=f"touch {sentinel}", receipt=str(receipt), cwd=tmp_path)
    item = pqwork.try_claim("skipme", "mine")
    assert _run_sync(item) == "already_complete"
    assert state_of("skipme") == pqwork.DONE
    assert read_state(pqwork.DONE, "skipme")["outcome"] == "already_complete"
    assert not sentinel.exists()


# --------------------------------------------------------------------------
# 3. atomic claim


def test_double_claim_yields_exactly_one_winner(queue):
    enqueue("contested")
    first = pqwork.try_claim("contested", "boxa")
    second = pqwork.try_claim("contested", "boxb")
    assert (first is None) != (second is None)
    assert first is not None and second is None
    # Count .json only: claimed/ also holds the sidecar .lease file.
    assert len(list(pqwork.qdir(pqwork.CLAIMED).glob("*.json"))) == 1
    assert not pqwork.item_path(pqwork.READY, "contested").exists()


def test_gated_actor_holds_nothing_and_ready_job_is_admitted(
        queue, tmp_path, monkeypatch):
    """Waiting for a launch gate is not itself a resource reservation."""
    owner = tmp_path / "gated-owner.json"
    # The external actor has not passed its gate, so it has made no acquire
    # call and owns no partial resource on either target.
    assert pqwork.read_reservation_ledger()["reservations"] == {}
    for host in ("sparky", "gx10-6b77"):
        assert pqwork.reserved_gpu_slots(host)[0] == 0
        assert pqwork.reserved_memory_gb(host)[0] == 0
    assert not owner.exists()

    receipt = tmp_path / "backfill.receipt"
    enqueue("backfill-while-gated",
            cmd=f"echo admitted > {receipt}", receipt=str(receipt),
            cwd=tmp_path, hosts=["sparky"], mem_gb=8.0)
    monkeypatch.setattr(pqwork, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(pqwork, "capping_supported", lambda: False)
    assert pqwork.worker_loop("sparky", set(), 6, 2, True,
                              8.0, 20.0, 0.0) == 0
    assert state_of("backfill-while-gated") == pqwork.DONE
    assert receipt.read_text().strip() == "admitted"
    heartbeat = pqwork.read_item(pqwork.worker_heartbeat_path("sparky"))
    assert heartbeat["pid"] == os.getpid()
    assert heartbeat["source_sha256"] == pqwork.loaded_source_sha256()


def test_multi_host_reservation_is_atomic_or_absent(queue, tmp_path):
    """A conflict on box B never leaves a hold behind on box A."""
    owner = tmp_path / "reservation-owner.json"
    claims = [host_resource("sparky"), host_resource("gx10-6b77")]
    (pqwork.qdir(pqwork.RESERVED) / "gx10-6b77.mem").write_text(
        "10\nconflicting work\n")

    with pytest.raises(pqwork.ReservationBusy):
        pqwork.acquire_reservation_set(
            "arm-a", claims, owner, "two-box serve",
            require_no_claimed=True)
    assert pqwork.read_reservation_ledger()["reservations"] == {}
    assert pqwork.reserved_memory_gb("sparky")[0] == 0
    assert pqwork.reserved_gpu_slots("sparky")[0] == 0
    assert not owner.exists()

    (pqwork.qdir(pqwork.RESERVED) / "gx10-6b77.mem").unlink()
    record = pqwork.acquire_reservation_set(
        "arm-a", claims, owner, "two-box serve", require_no_claimed=True)
    assert {claim["host"] for claim in record["claims"]} == {
        "sparky", "gx10-6b77"}
    for host in ("sparky", "gx10-6b77"):
        assert pqwork.reserved_gpu_slots(host)[0] == 6
        assert pqwork.reserved_memory_gb(host)[0] == pytest.approx(115.0)

    pqwork.release_reservation_set("arm-a", owner)
    for host in ("sparky", "gx10-6b77"):
        assert pqwork.reserved_gpu_slots(host)[0] == 0
        assert pqwork.reserved_memory_gb(host)[0] == 0


@pytest.mark.parametrize("_race", range(20))
def test_reservation_and_worker_admission_race_has_one_winner(
        queue, tmp_path, _race):
    """A full two-box hold and a target-host claim can never coexist."""
    item = enqueue("racing-job", hosts=["sparky"], needs_gpu=True,
                   mem_gb=1.0, cwd=tmp_path)
    owner = tmp_path / "race-owner.json"
    claims = [host_resource("sparky"), host_resource("gx10-6b77")]
    governor = _governor(total_gb=128.0)
    governor.samples.append((time.time(), 100.0))
    barrier = threading.Barrier(2)
    outcomes = {}

    def reserve():
        barrier.wait()
        try:
            pqwork.acquire_reservation_set(
                "race-set", claims, owner, "race",
                require_no_claimed=True)
            outcomes["reservation"] = True
        except pqwork.ReservationBusy:
            outcomes["reservation"] = False

    def admit():
        barrier.wait()
        outcomes["claim"] = pqwork.try_admit_item(
            item, "sparky", set(), True, False, 6, 2,
            governor, []) is not None

    threads = [threading.Thread(target=reserve),
               threading.Thread(target=admit)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes["reservation"] != outcomes["claim"]
    ledger_has_set = "race-set" in (
        pqwork.read_reservation_ledger()["reservations"])
    assert ledger_has_set == outcomes["reservation"]
    assert (state_of("racing-job") == pqwork.CLAIMED) == outcomes["claim"]


def test_wrong_owner_cannot_verify_or_release_reservation(queue, tmp_path):
    owner = tmp_path / "owner.json"
    impostor = tmp_path / "impostor.json"
    pqwork.acquire_reservation_set(
        "owned-set", [host_resource("sparky")], owner, "owned")
    impostor.write_text(json.dumps({
        "reservation_id": "owned-set", "token": "wrong-token"}))

    with pytest.raises(pqwork.ReservationOwnershipError):
        pqwork.verify_reservation_set("owned-set", impostor)
    with pytest.raises(pqwork.ReservationOwnershipError):
        pqwork.release_reservation_set("owned-set", impostor)
    assert "owned-set" in pqwork.read_reservation_ledger()["reservations"]


def test_low_level_claim_cannot_bypass_reservation(queue, tmp_path):
    owner = tmp_path / "owner.json"
    pqwork.acquire_reservation_set(
        "held-set", [host_resource("sparky")], owner, "held")
    enqueue("must-wait", hosts=["sparky"], needs_gpu=True, mem_gb=1.0)

    assert pqwork.try_claim("must-wait", "sparky") is None
    assert state_of("must-wait") == pqwork.READY
    pqwork.release_reservation_set("held-set", owner)
    assert pqwork.try_claim("must-wait", "sparky") is not None


def test_malformed_reservation_ledger_fails_closed(queue, tmp_path):
    pqwork.reservation_ledger_path().write_text("not-json")
    assert pqwork.reserved_gpu_slots("sparky")[0] > 1_000_000
    assert pqwork.reserved_memory_gb("sparky")[0] == float("inf")
    with pytest.raises(RuntimeError, match="malformed reservation ledger"):
        pqwork.acquire_reservation_set(
            "blocked", [host_resource("sparky")], tmp_path / "owner.json",
            "must not overwrite evidence")


def test_null_host_in_reservation_ledger_fails_closed(queue):
    pqwork.write_json_atomic(pqwork.reservation_ledger_path(), {
        "version": 2,
        "updated_at": time.time(),
        "reservations": {
            "broken": {
                "id": "broken",
                "owner_sha256": "0" * 64,
                "claims": [{
                    "host": None, "gpu_slots": 1, "mem_gb": 1.0,
                    "gpu_capacity": 6, "mem_capacity_gb": 121.6,
                }],
            },
        },
    })
    assert pqwork.reserved_memory_gb("sparky")[0] == float("inf")


def test_stale_admission_guard_is_fatal_not_retryable_busy(queue, tmp_path):
    lock = pqwork.admission_lock_path()
    lock.mkdir()
    pqwork.write_json_atomic(lock / "owner.json", {
        "token": "possibly-live-owner",
        "acquired_at": time.time() - pqwork.ADMISSION_LOCK_STALE_S - 1,
    })
    args = Namespace(
        id="blocked", owner_file=str(tmp_path / "owner.json"),
        host_resources=["sparky:6:115:6:121.6"],
        require_no_claimed=True, why="must block")

    assert pqwork.cmd_reserve_set_acquire(args) == 70
    assert not (tmp_path / "owner.json").exists()
    assert not pqwork.reservation_ledger_path().exists()


@pytest.mark.parametrize("corrupt_payload", ["not-json", "[1]"])
def test_unreadable_claim_blocks_reservation_acquisition(
        queue, tmp_path, corrupt_payload):
    pqwork.item_path(pqwork.CLAIMED, "corrupt").write_text(corrupt_payload)
    with pytest.raises(pqwork.ReservationBusy, match="corrupt"):
        pqwork.acquire_reservation_set(
            "blocked-by-claim", [host_resource("sparky")],
            tmp_path / "owner.json", "must fail closed")
    assert pqwork.read_reservation_ledger()["reservations"] == {}


def test_host_capacity_registry_survives_release_and_rejects_inflation(
        queue, tmp_path):
    owner = tmp_path / "owner.json"
    pqwork.acquire_reservation_set(
        "first", [host_resource("sparky")], owner, "establish capacity")
    assert (owner.stat().st_mode & 0o777) == 0o600
    pqwork.release_reservation_set("first", owner)

    with pytest.raises(ValueError, match="capacity conflicts"):
        pqwork.acquire_reservation_set(
            "inflated",
            [host_resource("sparky", mem_capacity=999.0)],
            tmp_path / "inflated-owner.json", "must not inflate")


def test_legacy_nan_memory_hold_fails_closed(queue, tmp_path):
    (pqwork.qdir(pqwork.RESERVED) / "sparky.mem").write_text(
        "nan\ncorrupt legacy hold\n")
    assert pqwork.reserved_memory_gb("sparky")[0] == float("inf")
    with pytest.raises(pqwork.ReservationBusy):
        pqwork.acquire_reservation_set(
            "blocked-by-nan", [host_resource("sparky")],
            tmp_path / "owner.json", "must fail closed")


def test_release_cannot_unlink_reacquired_owner_capability(
        queue, tmp_path, monkeypatch):
    owner = tmp_path / "owner.json"
    claims = [host_resource("sparky")]
    pqwork.acquire_reservation_set("same-id", claims, owner, "first")
    ledger_released = threading.Event()
    allow_release = threading.Event()
    original_write = pqwork.write_json_atomic

    def pause_after_release(path, payload):
        original_write(path, payload)
        if (path == pqwork.reservation_ledger_path()
                and "same-id" not in payload.get("reservations", {})):
            ledger_released.set()
            assert allow_release.wait(5)

    monkeypatch.setattr(pqwork, "write_json_atomic", pause_after_release)
    outcomes = {}

    def release():
        pqwork.release_reservation_set("same-id", owner)
        outcomes["released"] = True

    def reacquire():
        ledger_released.wait(5)
        pqwork.acquire_reservation_set("same-id", claims, owner, "second")
        outcomes["reacquired"] = True

    release_thread = threading.Thread(target=release)
    acquire_thread = threading.Thread(target=reacquire)
    release_thread.start()
    acquire_thread.start()
    assert ledger_released.wait(5)
    allow_release.set()
    release_thread.join()
    acquire_thread.join()

    assert outcomes == {"released": True, "reacquired": True}
    assert owner.exists()
    assert pqwork.verify_reservation_set("same-id", owner)["why"] == "second"


def test_reserve_set_cli_exit_contract_is_0_75_77_2(
        queue, tmp_path, capsys):
    owner = tmp_path / "owner.json"
    common = [
        "--owner-file", str(owner),
        "--host-resource", "sparky:6:115:6:121.6",
    ]
    assert pqwork.main([
        "reserve-set", "acquire", "--id", "cli-owner", *common,
    ]) == 0

    assert pqwork.main([
        "reserve-set", "acquire", "--id", "cli-contender",
        "--owner-file", str(tmp_path / "contender.json"),
        "--host-resource", "sparky:6:115:6:121.6",
    ]) == 75

    impostor = tmp_path / "impostor.json"
    impostor.write_text(json.dumps({
        "reservation_id": "cli-owner", "token": "wrong"}))
    assert pqwork.main([
        "reserve-set", "verify", "--id", "cli-owner",
        "--owner-file", str(impostor),
    ]) == 77

    assert pqwork.main([
        "reserve-set", "acquire", "--id", "invalid-resource",
        "--owner-file", str(tmp_path / "invalid.json"),
        "--host-resource", "sparky:not-a-number:1:6:121.6",
    ]) == 2
    assert pqwork.main([
        "reserve-set", "release", "--id", "cli-owner",
        "--owner-file", str(owner),
    ]) == 0
    capsys.readouterr()


# --------------------------------------------------------------------------
# 4. stale-lease reap


def test_stale_lease_requeues(queue, tmp_path):
    enqueue("dead", cwd=tmp_path)
    pqwork.try_claim("dead", "gone")
    backdate_lease("dead", pqwork.LEASE_TIMEOUT_S + 1)
    assert pqwork.reap() == 1
    assert state_of("dead") == pqwork.READY
    # The attempt the dead box consumed is retained, so a box that crashes
    # on every claim cannot loop forever.
    assert read_state(pqwork.READY, "dead")["attempts"] == 1


def test_stale_lease_at_max_attempts_fails(queue, tmp_path):
    enqueue("exhausted", max_attempts=1, cwd=tmp_path)
    pqwork.try_claim("exhausted", "gone")
    backdate_lease("exhausted", pqwork.LEASE_TIMEOUT_S + 1)
    assert pqwork.reap() == 1
    assert state_of("exhausted") == pqwork.FAILED
    assert read_state(pqwork.FAILED, "exhausted")["outcome"] == "lease_lost_max_attempts"


def test_stale_lease_with_receipt_is_done(queue, tmp_path):
    # The work finished and only the bookkeeping was lost; requeueing it
    # would redo a completed job for no reason.
    receipt = tmp_path / "receipt"
    enqueue("finished", receipt=str(receipt), cwd=tmp_path)
    pqwork.try_claim("finished", "gone")
    receipt.write_text("x")
    backdate_lease("finished", pqwork.LEASE_TIMEOUT_S + 1)
    assert pqwork.reap() == 1
    assert state_of("finished") == pqwork.DONE
    assert read_state(pqwork.DONE, "finished")["outcome"] == "receipt_found_after_lease_loss"


def test_fresh_lease_is_not_reaped(queue, tmp_path):
    enqueue("alive", cwd=tmp_path)
    pqwork.try_claim("alive", "mine")
    assert pqwork.reap() == 0
    assert state_of("alive") == pqwork.CLAIMED


def test_terminal_job_stale_orphan_lease_is_reaped_automatically(
        queue, tmp_path):
    """A terminal record cannot leave a stale claimed-sidecar forever."""
    enqueue("terminal-orphan", cwd=tmp_path)
    pqwork.try_claim("terminal-orphan", "gone")
    backdate_lease("terminal-orphan", pqwork.LEASE_TIMEOUT_S + 1)

    claimed = pqwork.item_path(pqwork.CLAIMED, "terminal-orphan")
    terminal = pqwork.item_path(pqwork.FAILED, "terminal-orphan")
    os.rename(claimed, terminal)

    assert pqwork.reap() == 1
    assert not pqwork.lease_path("terminal-orphan").exists()
    assert (pqwork.qdir(pqwork.REAPED) / "terminal-orphan.lease").exists()
    assert state_of("terminal-orphan") == pqwork.FAILED


def test_terminal_job_fresh_orphan_lease_is_reaped_automatically(
        queue, tmp_path):
    """Terminal state, independently of TTL, makes its lease obsolete."""
    enqueue("fresh-terminal-orphan", cwd=tmp_path)
    pqwork.try_claim("fresh-terminal-orphan", "gone")
    claimed = pqwork.item_path(pqwork.CLAIMED, "fresh-terminal-orphan")
    terminal = pqwork.item_path(pqwork.DONE, "fresh-terminal-orphan")
    os.rename(claimed, terminal)

    assert pqwork.reap() == 1
    assert not pqwork.lease_path("fresh-terminal-orphan").exists()
    assert (pqwork.qdir(pqwork.REAPED)
            / "fresh-terminal-orphan.lease").exists()
    assert state_of("fresh-terminal-orphan") == pqwork.DONE


def test_terminal_job_malformed_orphan_lease_is_reaped_automatically(
        queue, tmp_path):
    """Terminal truth wins even when an abandoned heartbeat is corrupt."""
    enqueue("terminal-malformed", cwd=tmp_path)
    pqwork.try_claim("terminal-malformed", "gone")
    claimed = pqwork.item_path(pqwork.CLAIMED, "terminal-malformed")
    terminal = pqwork.item_path(pqwork.FAILED, "terminal-malformed")
    os.rename(claimed, terminal)
    pqwork.lease_path("terminal-malformed").write_text(
        '{"host": "dead-box", "beat": "not-a-number"}')

    assert pqwork.reap() == 1
    assert not pqwork.lease_path("terminal-malformed").exists()
    assert (pqwork.qdir(pqwork.REAPED)
            / "terminal-malformed.lease").exists()


def test_malformed_claim_heartbeat_fails_closed_into_recovery(
        queue, tmp_path):
    enqueue("malformed-claim", cwd=tmp_path)
    assert pqwork.try_claim("malformed-claim", "dead-box") is not None
    pqwork.lease_path("malformed-claim").write_text(
        '{"host": "dead-box", "beat": "not-a-number"}')

    assert pqwork.reap() == 1
    assert state_of("malformed-claim") == pqwork.READY


def test_stale_orphan_lease_without_item_record_is_reaped(queue):
    pqwork.write_lease("lost-record", "gone", 123)
    backdate_lease("lost-record", pqwork.LEASE_TIMEOUT_S + 1)

    assert pqwork.reap() == 1
    assert not pqwork.lease_path("lost-record").exists()
    assert (pqwork.qdir(pqwork.REAPED) / "lost-record.lease").exists()


# --------------------------------------------------------------------------
# 5. dependency gating


def test_after_gates_on_receipt_existence(queue, tmp_path):
    dep = tmp_path / "upstream.json"
    item = enqueue("downstream", after=[str(dep)], cwd=tmp_path)
    assert not pqwork.is_runnable(item, "mine", set(), True)
    dep.write_text("{}")
    assert pqwork.is_runnable(item, "mine", set(), True)


# --------------------------------------------------------------------------
# 6. host pinning and tags


def test_host_pin_excludes_other_boxes(queue):
    # /home/rob is box-local, so an item whose inputs live there is not
    # portable; the pin is the only thing preventing a doomed claim.
    item = enqueue("pinned", hosts=["other"])
    assert not pqwork.is_runnable(item, "mine", set(), True)
    assert pqwork.is_runnable(item, "other", set(), True)


def test_required_tag_excludes_worker_without_it(queue):
    item = enqueue("needs-rocm", requires=["rocm"])
    assert not pqwork.is_runnable(item, "mine", {"cuda"}, True)
    assert pqwork.is_runnable(item, "mine", {"cuda", "rocm"}, True)


def test_gpu_item_not_runnable_on_cpu_only_worker(queue):
    item = enqueue("gpujob", needs_gpu=True)
    assert not pqwork.is_runnable(item, "mine", set(), False)
    assert pqwork.is_runnable(item, "mine", set(), True)


# --------------------------------------------------------------------------
# 7. GPU slot capacity


def test_gpu_slot_blocks_second_gpu_item_but_not_cpu_work(queue, tmp_path):
    # Unified memory means two big GPU jobs contend for one pool, so the
    # slot cap has to hold even while cheap CPU work keeps flowing.
    enqueue("gpu-held", needs_gpu=True, cwd=tmp_path)
    assert pqwork.try_claim("gpu-held", "mine") is not None

    # The blocked GPU item sorts first, so picking the CPU item can only
    # happen by way of the slot check.
    enqueue("gpu-next", needs_gpu=True, priority=99, cwd=tmp_path)
    cpu_receipt = tmp_path / "cpu.done"
    enqueue("cpu-job", cmd=f"touch {cpu_receipt}", priority=10, cwd=tmp_path)

    pqwork.worker_loop("mine", set(), gpu_slots=1, cpu_slots=2, once=True, reserve_gb=0.0, horizon_s=20.0,
                       settle_s=0.0, admission="optimistic")

    assert state_of("gpu-next") == pqwork.READY
    assert state_of("cpu-job") == pqwork.DONE
    assert cpu_receipt.exists()


# --------------------------------------------------------------------------
# 8. priority and FIFO ordering


def test_priority_then_enqueue_order(queue, tmp_path):
    order = tmp_path / "order.txt"
    # Fixed timestamps: successive time.time() calls are too coarse to
    # guarantee the tie-break is actually being exercised.
    enqueue("low", cmd=f"echo low >> {order}", priority=10,
            enqueued_at=100.0, cwd=tmp_path)
    enqueue("high-late", cmd=f"echo high-late >> {order}", priority=90,
            enqueued_at=300.0, cwd=tmp_path)
    enqueue("high-early", cmd=f"echo high-early >> {order}", priority=90,
            enqueued_at=200.0, cwd=tmp_path)

    for _ in range(3):
        pqwork.worker_loop("mine", set(), gpu_slots=0, cpu_slots=2, once=True, reserve_gb=0.0, horizon_s=20.0,
                       settle_s=0.0, admission="optimistic")

    assert order.read_text().split() == ["high-early", "high-late", "low"]


# --------------------------------------------------------------------------
# 9. TMPDIR is never /tmp for a child


def test_child_tmpdir_is_box_local_not_slash_tmp(queue, tmp_path, monkeypatch):
    # /tmp was cleared by an OOM once and took the MiniMax artifacts with
    # it; every child gets a TMPDIR under $HOME instead.
    captured = tmp_path / "tmpdir.txt"
    # An inherited TMPDIR must be overridden, not passed through: the parent
    # here is deliberately pointed at /tmp.
    monkeypatch.setenv("TMPDIR", "/tmp")
    enqueue("tmpdir", cmd=f'echo "$TMPDIR" > {captured}', cwd=tmp_path)
    item = pqwork.try_claim("tmpdir", "mine")
    assert _run_sync(item) == "done"

    value = captured.read_text().strip()
    assert value
    assert value != "/tmp"
    assert Path(value).is_relative_to(Path.home())
    assert Path(value).is_dir()


def test_empty_receipt_is_not_a_receipt(queue):
    """A zero-byte file must not count as completion.

    A shell redirect creates its target the instant it opens it, so
    `cmd > out` produces a "receipt" before the work has done anything. The
    queue gated on existence while jobs gated on `test -s`; the two disagreed
    and an item reached done/ with no real receipt.
    """
    receipt = queue / "empty.receipt"
    item = _enqueue(queue, "hollow", cmd=f"touch {receipt}", receipt=str(receipt),
                    max_attempts=1)
    pqwork.try_claim("hollow", "mine")
    assert _run_sync(item) == "failed"
    assert receipt.exists(), "the command did create the file"
    assert (pqwork.qdir(pqwork.FAILED) / "hollow.json").exists()
    assert not (pqwork.qdir(pqwork.DONE) / "hollow.json").exists()


def test_empty_receipt_does_not_satisfy_a_dependency(queue):
    """`--after` uses the same predicate, or a dependent starts on nothing."""
    dep = queue / "dep.receipt"
    dep.touch()
    item = {"id": "waiter", "after": [str(dep)], "cmd": "true"}
    assert not pqwork.is_runnable(item, "mine", set(), True)
    dep.write_text("1")
    assert pqwork.is_runnable(item, "mine", set(), True)


def test_empty_receipt_is_not_a_receipt(queue, tmp_path):
    """A zero-byte file must not count as completion.

    A shell redirect creates its target the instant it opens it, so
    ``cmd > out`` produces a "receipt" before the work has done anything. The
    queue once gated on existence while jobs gated on ``test -s``; the two
    disagreed and an item reached done/ carrying no real receipt.
    """
    receipt = tmp_path / "empty.receipt"
    enqueue("hollow", cmd=f"touch {receipt}", receipt=str(receipt),
            max_attempts=1, cwd=tmp_path)
    item = pqwork.try_claim("hollow", "mine")
    assert _run_sync(item) == "failed"
    assert receipt.exists(), "the command really did create the file"
    assert state_of("hollow") == pqwork.FAILED


def test_empty_receipt_does_not_satisfy_a_dependency(queue, tmp_path):
    """``--after`` shares the predicate, or a dependent starts on nothing."""
    dep = tmp_path / "dep.receipt"
    dep.touch()
    item = {"id": "waiter", "after": [str(dep)], "cmd": "true"}
    assert not pqwork.is_runnable(item, "mine", set(), True)
    dep.write_text("1")
    assert pqwork.is_runnable(item, "mine", set(), True)


def _systemd_user_available() -> bool:
    """Whether transient user units can actually be started here.

    The cap depends on cgroup delegation to the user manager, which is a
    property of the box, not of the code. Skipping where it is absent keeps
    the check honest: a pass has to mean the cap fired, never that the
    machinery was unavailable and the test quietly agreed.
    """
    try:
        out = subprocess.run(["systemd-run", "--user", "--quiet", "--wait",
                              "--", "/bin/true"], capture_output=True,
                             timeout=30)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def test_unit_name_is_a_legal_unit_name():
    assert pqwork.unit_name("probe-27b") == "pqjob-probe-27b"
    # Item ids are free-form; a unit name is not.
    assert pqwork.unit_name("a/b c#d") == "pqjob-a-b-c-d"


def test_a_job_without_a_declared_budget_stays_a_direct_child(queue):
    """No declaration, no cap -- and no unit.

    The cap is what a declared budget buys. Leaving undeclared jobs on the
    plain child path keeps them evictable by the governor, which is the only
    lever that reaches a job whose appetite was never stated.
    """
    item = {"id": "plain", "cmd": "true", "attempts": 1, "max_attempts": 1,
            "enqueued_at": time.time()}
    (pqwork.qdir(pqwork.CLAIMED) / "plain.json").write_text(json.dumps(item))
    job = pqwork.Job(item, "testhost")
    job.start()
    for _ in range(100):
        if not job.alive():
            break
        time.sleep(0.1)
    assert job.unit is None


@pytest.mark.skipif(not _systemd_user_available(),
                    reason="no delegated user cgroup on this box")
def test_a_job_that_exceeds_its_budget_is_killed_and_not_retried(queue):
    """The offender dies, and it dies once.

    Retrying is what a transient failure earns. A job that asked for 1 GB and
    wanted 3 is not transient -- the declaration is wrong, and two more
    attempts only reproduce it. The recorded outcome has to name the fix.
    """
    hog = Path(queue) / "hog.py"
    hog.write_text(
        "buf = []\n"
        "for _ in range(60):\n"
        "    buf.append(bytearray(64 * 1024 * 1024))\n")
    receipt = str(Path(queue) / "hog.receipt")
    item = {"id": "hog", "cmd": f"python3 {hog}", "mem_gb": 1.0,
            "receipt": receipt, "attempts": 1, "max_attempts": 3,
            "enqueued_at": time.time()}
    (pqwork.qdir(pqwork.CLAIMED) / "hog.json").write_text(json.dumps(item))
    job = pqwork.Job(item, "testhost")
    job.start()
    for _ in range(600):
        if not job.alive():
            break
        time.sleep(0.1)
    assert not job.alive(), "capped job never terminated"
    assert job.oomed, "the cgroup did not kill it -- the cap is not enforcing"
    assert not Path(receipt).exists()
    assert state_of("hog") == pqwork.FAILED, "a wrong budget must not be retried"
    rec = json.loads((pqwork.qdir(pqwork.FAILED) / "hog.json").read_text())
    assert "budget_exceeded" in rec["outcome"]
    assert "--mem-gb" in rec["outcome"], "the outcome must name the fix"


def test_only_declared_refuses_work_that_did_not_ask_for_this_box(queue):
    """Adding an unlike box must not break work that predates it.

    An item naming no host and no tags is eligible everywhere, which is right
    for interchangeable boxes and wrong the moment one differs. A worker in
    only-declared mode takes it only if it was asked for.
    """
    plain = {"id": "plain", "cmd": "true"}
    by_host = {"id": "byhost", "cmd": "true", "hosts": ["wslgpu"]}
    by_tag = {"id": "bytag", "cmd": "true", "requires": ["x86"]}
    tags = {"x86"}

    # Default mode: the undeclared item is fair game.
    assert pqwork.is_runnable(plain, "wslgpu", tags, False)
    # Only-declared: it is not, but both declared forms still are.
    assert not pqwork.is_runnable(plain, "wslgpu", tags, False,
                                  only_declared=True)
    assert pqwork.is_runnable(by_host, "wslgpu", tags, False,
                              only_declared=True)
    assert pqwork.is_runnable(by_tag, "wslgpu", tags, False,
                              only_declared=True)
    # A tag this worker does not carry is still refused in either mode.
    assert not pqwork.is_runnable({"id": "gb10only", "cmd": "true",
                                   "requires": ["gb10"]},
                                  "wslgpu", tags, False)


# --------------------------------------------------------------------------
# 10. memory-governor trends and eviction attribution


def _governor(*, total_gb=128.0):
    return pqwork.MemoryGovernor(reserve_gb=24.0, horizon_s=60.0,
                                 settle_s=0.0, total_gb=total_gb)


def _drive_sample(monkeypatch, governor, sampled_at, available_gb):
    monkeypatch.setattr(pqwork.time, "time", lambda: sampled_at)
    monkeypatch.setattr(pqwork, "mem_available_gb", lambda: available_gb)
    return governor.sample()


def _drive_pressure(monkeypatch, *, start=0.0):
    governor = _governor()
    monkeypatch.setattr(pqwork, "swap_free_gb", lambda: 16.0)
    for offset, available in [(0, 90.0), (2, 87.5),
                              (4, 85.0), (6, 82.5)]:
        _drive_sample(monkeypatch, governor, start + offset, available)
    return governor


def test_currency_log_trace_is_flat_and_does_not_evict(monkeypatch):
    """The sparse application log proves a plateau, not a falling box."""
    monkeypatch.setattr(pqwork, "swap_free_gb", lambda: 16.0)
    # Seconds and MemAvailable GB transcribed from pq-currency-4b-v2's fifth
    # run, 01:40:45--02:27:45.  The job moved only 1 GB in 47 minutes and
    # remained 56.9 GB above the worker's 24 GB reserve.
    governor = _governor()
    for sample in [
        (0, 81.9), (234, 81.4), (468, 81.4), (703, 81.3),
        (937, 81.3), (1172, 81.3), (1406, 81.3), (1642, 81.3),
        (1877, 81.3), (2115, 81.3), (2351, 81.3), (2585, 81.3),
        (2820, 80.9),
    ]:
        _drive_sample(monkeypatch, governor, *sample)
        assert not governor.must_evict([object()])


def test_one_endpoint_step_cannot_override_a_flat_trace(monkeypatch):
    """One alarming pair cannot become a 60-second trajectory."""
    monkeypatch.setattr(pqwork, "swap_free_gb", lambda: 16.0)
    governor = _governor()
    # lina's swap leaves the configured horizon at 60 seconds.  The old
    # endpoint estimator turns this isolated 1 GB/s step into a projected
    # 60 GB fall and fires immediately; subsequent samples show a plateau.
    for sample in [(0, 82.9), (2, 80.9), (4, 80.9),
                   (6, 80.9), (8, 80.9)]:
        _drive_sample(monkeypatch, governor, *sample)
        assert not governor.must_evict([object()])


def test_sustained_fast_drop_still_evicts(monkeypatch):
    governor = _drive_pressure(monkeypatch)
    assert governor.must_evict([object()])


def test_single_memory_blip_then_recovery_does_not_evict(monkeypatch):
    monkeypatch.setattr(pqwork, "swap_free_gb", lambda: 16.0)
    governor = _governor()
    for sample in [(0, 81.3), (2, 81.3), (4, 30.0),
                   (6, 81.3), (8, 81.3)]:
        _drive_sample(monkeypatch, governor, *sample)
    assert not governor.must_evict([object()])


def _claimed_job(item_id, **fields):
    enqueue(item_id, attempts=0, max_attempts=3, **fields)
    item = pqwork.try_claim(item_id, "mine")
    assert item is not None
    return pqwork.Job(item, "mine")


def test_declared_job_samples_its_cgroup_memory_current(queue, tmp_path):
    job = _claimed_job("measured", mem_gb=55.0)
    job.unit = pqwork.unit_name(job.id)
    memory_current = tmp_path / "memory.current"
    job._memory_current_path = memory_current

    for sampled_at, footprint_gb in [
        (0, 20.0), (2, 21.0), (4, 22.0), (6, 23.0),
    ]:
        memory_current.write_text(str(int(footprint_gb * 1024 ** 3)))
        assert job.sample_footprint(sampled_at) == footprint_gb

    assert list(job.footprint_samples) == [
        (0, 20.0), (2, 21.0), (4, 22.0), (6, 23.0),
    ]
    assert job.footprint_source == f"cgroup:{memory_current}"


def test_collateral_eviction_records_policy_age_without_fit_blame(queue):
    """Victim policy is not evidence that the selected job caused pressure."""
    collateral = _claimed_job("collateral", evictions=1)
    collateral.footprint_samples.extend([
        (0, 20.0), (2, 20.0), (4, 20.0), (6, 20.0),
    ])
    attribution = collateral.classify_eviction(box_drop_rate_gb_s=1.0)
    assert attribution["classification"] == "collateral"
    collateral.evicted = True
    collateral.eviction_attribution = attribution
    assert collateral._finish_eviction()

    record = read_state(pqwork.READY, "collateral")
    assert record["outcome"] == "evicted_as_collateral"
    assert record["evictions"] == 2
    assert record.get("attributed_evictions", 0) == 0
    assert record["collateral_evictions"] == 1
    assert record["protection_earned"] is True
    assert (record["not_before"] - record["last_evicted_at"]
            == pytest.approx(pqwork.EVICT_BACKOFF_S))
    assert record["eviction_attribution"]["victim_growth_gb_s"] == 0.0
    assert state_of("collateral") != pqwork.FAILED

    growing = _claimed_job("growing", evictions=0)
    growing.footprint_samples.extend([
        (0, 20.0), (2, 22.0), (4, 24.0), (6, 26.0),
    ])
    attribution = growing.classify_eviction(box_drop_rate_gb_s=1.0)
    assert attribution["classification"] == "victim_growth"
    growing.evicted = True
    growing.eviction_attribution = attribution
    assert growing._finish_eviction()

    measured = read_state(pqwork.READY, "growing")
    assert measured["outcome"] == "evicted_for_own_memory_growth"
    assert measured["attributed_evictions"] == 1
    assert measured["evictions"] == 1
    assert state_of("growing") != pqwork.FAILED


def _live_job(item_id, *, priority, evictions, started,
              evicted_runtime_s=0.0, mem_gb=55.0):
    item = {
        "id": item_id,
        "cmd": "true",
        "priority": priority,
        "evictions": evictions,
        "evicted_runtime_s": evicted_runtime_s,
        "evictable": True,
        "mem_gb": mem_gb,
    }
    job = pqwork.Job(item, "mine")
    job.started = started
    job.alive = lambda: True
    return job


def test_restart_does_not_make_the_evicted_item_the_next_victim(monkeypatch):
    governor = _drive_pressure(monkeypatch, start=1000.0)
    incumbent = _live_job("incumbent", priority=50, evictions=0,
                          started=900.0)
    restarted = _live_job("restarted", priority=50, evictions=1,
                          evicted_runtime_s=200.0, started=1005.0)

    jobs = [incumbent, restarted]
    assert governor.must_evict(jobs)
    assert pqwork.choose_victim(jobs) is incumbent


def test_fitting_item_ages_into_protection_and_completes(
        queue, tmp_path, monkeypatch):
    """A low-priority fit survives an unbounded stream within two losses."""
    receipt = tmp_path / "target.receipt"
    enqueue("target", cmd=f"echo complete > {receipt}", cwd=tmp_path,
            receipt=str(receipt), mem_gb=55.0, priority=10)
    target_item = read_state(pqwork.READY, "target")
    protection_after = getattr(pqwork, "EVICTIONS_BEFORE_PROTECTION", 2)
    survived = False

    for attempt in range(protection_after + 1):
        now = 2000.0 + attempt * 20.0
        governor = _drive_pressure(monkeypatch, start=now - 6.0)
        item_evictions = int(target_item.get("evictions", 0))
        target = _live_job(
            "target", priority=10,
            evictions=item_evictions,
            evicted_runtime_s=float(target_item.get("evicted_runtime_s", 0.0)),
            started=now - 1.0)
        arrival = _live_job(f"arrival-{attempt}", priority=50, evictions=0,
                            started=now - 10.0)
        assert governor.must_evict([target, arrival])
        victim = pqwork.choose_victim([target, arrival])
        if victim is arrival:
            survived = True
            break
        assert victim is target
        target_item["evictions"] = item_evictions + 1
        target_item["evicted_runtime_s"] = (
            float(target_item.get("evicted_runtime_s", 0.0)) + 1.0)

    assert survived, "the fitting target kept losing to newer arrivals"
    assert target_item["evictions"] == protection_after

    # Aging also reserves admission: no newcomer can consume the target's
    # 55 GB while it waits, and its running attempt is non-evictable.
    newcomer = {"id": "newcomer", "priority": 99, "evictions": 0,
                "enqueued_at": 0.0, "mem_gb": 1.0}
    target_item["enqueued_at"] = 1.0
    assert governor.admission_candidates([newcomer, target_item]) == [target_item]
    assert not target.evictable
    _drive_sample(monkeypatch, governor, now + 2.0, 50.0)
    assert not governor.may_admit(target_item, [arrival])
    _drive_sample(monkeypatch, governor, now + 4.0, 100.0)
    assert governor.may_admit(target_item, [])

    pqwork.write_json_atomic(pqwork.item_path(pqwork.READY, "target"),
                             target_item)
    monkeypatch.setattr(pqwork, "capping_supported", lambda: False)
    claimed = pqwork.try_claim("target", "mine")
    assert claimed is not None
    assert _run_sync(claimed) == "done"
    assert receipt.read_text().strip() == "complete"


def test_eviction_count_pins_cooldown_reservation_until_eventual_admission(
        queue, monkeypatch):
    """The protection threshold is an admission guarantee, not a counter."""
    now = time.time()
    enqueue("aged", priority=1, mem_gb=55.0,
            evictions=pqwork.EVICTIONS_BEFORE_PROTECTION,
            not_before=now + 60.0, enqueued_at=now - 100.0)
    enqueue("newcomer", priority=99, mem_gb=1.0, enqueued_at=now)

    visible = pqwork.ready_candidates("mine", set(), True)
    assert [item["id"] for item in visible] == ["newcomer", "aged"]
    governor = _governor(total_gb=100.0)
    assert [item["id"] for item in
            governor.admission_candidates(visible)] == ["aged"]
    assert not pqwork.is_runnable(visible[1], "mine", set(), True)

    # The worker's final runnable check leaves both items ready during the
    # protected item's cooldown instead of backfilling with the newcomer.
    monkeypatch.setattr(pqwork, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(pqwork, "capping_supported", lambda: False)
    assert pqwork.worker_loop("mine", set(), 1, 1, True,
                              8.0, 20.0, 0.0) == 0
    assert state_of("aged") == pqwork.READY
    assert state_of("newcomer") == pqwork.READY

    aged = read_state(pqwork.READY, "aged")
    aged["not_before"] = 0
    pqwork.write_json_atomic(pqwork.item_path(pqwork.READY, "aged"), aged)
    assert pqwork.worker_loop("mine", set(), 1, 1, True,
                              8.0, 20.0, 0.0) == 0
    assert state_of("aged") == pqwork.DONE
    assert state_of("newcomer") == pqwork.READY


def test_non_evictable_job_waits_unclaimed_until_declaration_fits(
        queue, tmp_path, monkeypatch):
    """Irreversible work cannot turn a positive margin negative on start."""
    receipt = tmp_path / "pinned.receipt"
    enqueue(
        "pinned", cmd=f"echo complete > {receipt}", cwd=tmp_path,
        receipt=str(receipt), mem_gb=55.0, evictable=False,
        priority=95, enqueued_at=1.0)
    enqueue(
        "backfill", cwd=tmp_path, mem_gb=1.0,
        priority=10, enqueued_at=2.0)

    # This reconstructs the production admission: 72.8 GB available minus
    # the worker's 24 GB reserve leaves +48.8 GB before the claim, but the
    # pinned job declares 55 GB.  The old optimistic branch admitted it and
    # only the post-claim log exposed the resulting -6.2 GB commitment.
    available = [72.8]
    monkeypatch.setattr(pqwork, "mem_available_gb", lambda: available[0])
    monkeypatch.setattr(pqwork, "capping_supported", lambda: False)
    assert pqwork.worker_loop(
        "mine", set(), 0, 2, True, 24.0, 20.0, 60.0) == 0

    # Waiting is neither a claim nor an external resource hold.  It does pin
    # admission order so repeated backfill cannot starve irreversible work.
    assert state_of("pinned") == pqwork.READY
    assert state_of("backfill") == pqwork.READY
    assert list(pqwork.qdir(pqwork.CLAIMED).glob("*.json")) == []
    assert pqwork.read_reservation_ledger()["reservations"] == {}
    assert not receipt.exists()

    available[0] = 80.0  # 56 GB headroom: the complete declaration now fits.
    assert pqwork.worker_loop(
        "mine", set(), 0, 2, True, 24.0, 20.0, 60.0) == 0
    assert state_of("pinned") == pqwork.DONE
    assert state_of("backfill") == pqwork.READY
    assert receipt.read_text().strip() == "complete"


def test_evictable_job_keeps_optimistic_positive_headroom_probe(monkeypatch):
    """The safety boundary does not silently turn all admission strict."""
    governor = _governor()
    _drive_sample(monkeypatch, governor, time.time(), 72.8)
    item = {"id": "probe", "mem_gb": 55.0, "evictable": True,
            "evictions": 0}

    assert governor.headroom([]) == pytest.approx(48.8)
    assert governor.may_admit(item, [])


def test_protected_dependency_gate_allows_backfill(queue):
    """Protection cannot reserve a box before external dependencies exist."""
    enqueue("gated-aged", priority=99,
            evictions=pqwork.EVICTIONS_BEFORE_PROTECTION,
            after=[str(queue / "missing.receipt")])
    enqueue("backfill", priority=1)

    assert [item["id"] for item in
            pqwork.ready_candidates("mine", set(), True)] == ["backfill"]


def test_only_measured_absolute_impossibility_is_terminal(queue):
    enqueue("oversized", mem_gb=80.0)
    item = read_state(pqwork.READY, "oversized")
    governor = _governor(total_gb=100.0)

    assert pqwork.fail_measured_impossibility(item, governor)
    terminal = read_state(pqwork.FAILED, "oversized")
    assert terminal["outcome"] == "measured_footprint_exceeds_box_capacity"
    measurement = terminal["fit_measurement"]
    assert measurement["source"] == "declared"
    assert measurement["footprint_gb"] == pytest.approx(80.0)
    assert measurement["box_total_gb"] == pytest.approx(100.0)
    assert measurement["reserve_gb"] == pytest.approx(24.0)
    assert measurement["capacity_gb"] == pytest.approx(76.0)
    assert measurement["measured_at"] > 0


def test_deployment_propagates_identical_copies_only_between_claims(
        queue, tmp_path, monkeypatch):
    source = tmp_path / "repo-pqwork.py"
    source.write_text("#!/usr/bin/env python3\nprint('new')\n")
    shared = queue / "bin" / "pqwork.py"
    shared.parent.mkdir()
    shared.write_text("old shared copy\n")
    local = tmp_path / "local" / "pqwork.py"
    monkeypatch.setattr(pqdeploy, "LOCAL_TARGET", local)

    digest = pqdeploy.deploy(source, queue, ["localhost"], restart=False)
    assert pqdeploy.sha256(shared) == digest
    assert pqdeploy.sha256(local) == digest
    assert not (queue / pqdeploy.DEPLOY_HOLD).exists()

    pqwork.write_json_atomic(queue / pqwork.CLAIMED / "busy.json",
                             {"id": "busy"})
    source.write_text("#!/usr/bin/env python3\nprint('must not land')\n")
    before = shared.read_bytes()
    with pytest.raises(RuntimeError, match="safe only between claims"):
        pqdeploy.deploy(source, queue, ["localhost"], restart=False)
    assert shared.read_bytes() == before
    assert local.read_bytes() == before
    assert not (queue / pqdeploy.DEPLOY_HOLD).exists()


def test_deployment_hold_blocks_new_claim_candidates(queue):
    enqueue("waiting")
    assert [item["id"] for item in
            pqwork.ready_candidates("mine", set(), True)] == ["waiting"]
    pqwork.deployment_hold_path().write_text("deployment in progress\n")
    assert pqwork.ready_candidates("mine", set(), True) == []


@pytest.mark.parametrize("_race", range(10))
def test_deployment_hold_and_worker_claim_have_one_linearized_winner(
        queue, tmp_path, _race):
    item = enqueue("deploy-race", hosts=["sparky"], needs_gpu=True,
                   mem_gb=1.0, cwd=tmp_path)
    governor = _governor(total_gb=128.0)
    governor.samples.append((time.time(), 100.0))
    barrier = threading.Barrier(2)
    outcomes = {}

    def hold():
        barrier.wait()
        try:
            pqdeploy.acquire_hold(queue)
            outcomes["hold"] = True
        except RuntimeError:
            outcomes["hold"] = False

    def claim():
        barrier.wait()
        outcomes["claim"] = pqwork.try_admit_item(
            item, "sparky", set(), True, False, 6, 2,
            governor, []) is not None

    threads = [threading.Thread(target=hold),
               threading.Thread(target=claim)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes["hold"] != outcomes["claim"]
    assert (queue / pqdeploy.DEPLOY_HOLD).exists() == outcomes["hold"]
    assert (state_of("deploy-race") == pqwork.CLAIMED) == outcomes["claim"]
