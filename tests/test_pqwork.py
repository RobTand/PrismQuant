"""Pull-queue invariants in tools/pqwork.py.

The tool exists because agent-dispatched work stopped when the agent did.
Every property pinned here is one the recovery story rests on: a claim that
two boxes can both win, a lease that never expires, or a "done" that trusts
an exit code all put the queue back to losing work silently.
"""
from __future__ import annotations

import importlib.util
import subprocess
import json
import time
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


def _load_tool():
    spec = importlib.util.spec_from_file_location("pqwork", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pqwork = _load_tool()


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
