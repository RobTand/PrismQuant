#!/usr/bin/env python3
"""Shared pull-queue for quantization work across the box fleet.

Why this exists
---------------
Until now every computational job was dispatched by hand from an agent
session: ssh to a box, ``systemd-run`` a unit, watch it.  That makes the
agent the scheduler, and it fails in the three ways observed on
2026-08-29 -- sparklina idle at 4.5 W for ~20 minutes with work waiting,
wsl-gpu at load 0.00 while the board said "queued", and three jobs dying
with nothing rescheduled when a session hit its limit at 20:40 ET.

The fix is to invert the direction.  Agents *enqueue* work items to a
directory on shared storage; each box runs a worker loop that *pulls* the
next item it is eligible for.  The agent is a producer, never the
dispatcher, so an agent going away stops new work from being *created*
but never stops queued work from *running*.

Design notes that are load-bearing
----------------------------------
* **Claiming is ``rename()``, never ``flock``.**  ``rename`` is atomic on
  NFSv4; advisory locking over NFS is version-dependent and has burned
  people.  The loser of a race gets ``FileNotFoundError`` and moves on.
* **A claim is a lease, not a grant.**  The claiming worker rewrites the
  lease file's timestamp every ``HEARTBEAT_S``.  Any worker may requeue a
  claim whose lease has gone stale, which is what makes a dead box
  recoverable rather than a permanent hole in the queue.
* **Completion is gated on the receipt, not on process state.**  A unit
  can report success while its work did not happen (see
  ``systemd_collect_reports_success_for_failed_units``), and an exit code
  says nothing about whether the artifact landed.  An item that declares
  a ``receipt`` path is complete when that path exists and only then.
* **Receipt-gating is also what makes requeue safe.**  Re-running an item
  whose receipt already exists is a no-op skip, so a stale-lease requeue
  cannot double-execute finished work.
* **Nothing is written to ``/tmp``.**  ``TMPDIR`` is forced to a
  box-local queue directory under ``$HOME`` for every child.

Deliberately not built: cross-box migration of work whose inputs live on
box-local storage.  ``/home/rob`` is *local per box*, not shared, so an
item is portable only if its inputs are on ``/mnt/shared``.  Items pin
themselves with ``hosts`` when they are not portable; the initial
utilization win is backfill-on-idle per box.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

QUEUE_ROOT = Path(os.environ.get("PQ_QUEUE_ROOT", "/mnt/shared/pq-queue"))

READY = "ready"
CLAIMED = "claimed"
DONE = "done"
FAILED = "failed"
LOGS = "logs"
RESERVED = "reserved"
STATE_DIRS = (READY, CLAIMED, DONE, FAILED, LOGS, RESERVED)

# A lease is refreshed this often and considered abandoned after this long.
# The timeout dwarfs any plausible NTP skew between boxes, which matters
# because the lease carries an absolute epoch written by the *claiming*
# host and compared against the *reaping* host's clock.
HEARTBEAT_S = 30
LEASE_TIMEOUT_S = 300

POLL_S = 10


# --------------------------------------------------------------------------
# layout helpers


def qdir(name: str) -> Path:
    return QUEUE_ROOT / name


def ensure_layout() -> None:
    for d in STATE_DIRS:
        qdir(d).mkdir(parents=True, exist_ok=True)


def item_path(state: str, item_id: str) -> Path:
    return qdir(state) / f"{item_id}.json"


def lease_path(item_id: str) -> Path:
    return qdir(CLAIMED) / f"{item_id}.lease"


def log_path(item_id: str) -> Path:
    return qdir(LOGS) / f"{item_id}.log"


def read_item(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file in the same directory, then rename.

    Same directory matters: rename is only atomic within a filesystem, and
    a reader must never observe a half-written item.
    """
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.rename(tmp, path)


def reserved_gpu_slots(host: str) -> tuple[int, str]:
    """GPU slots on this box held by something outside the queue.

    Work launched before the queue existed -- a campaign started by hand, a
    long interactive job -- is invisible to the scheduler, and stacking a
    queued GPU item on top of it would over-subscribe a single unified
    memory pool.  The alternative, inferring "the GPU is busy" from
    telemetry, is not available here: on GB10 ``gpu_utilization`` reads 96%
    for a stalled kernel and a saturated one alike, so any such check would
    be a guess wearing a measurement's clothes.

    So the holder declares itself.  ``reserved/<host>.gpu`` contains the
    number of slots held and, on the following lines, who holds them and
    why.  Removing the file returns the slots.  An absent file means
    nothing is held, which is the common case.
    """
    f = qdir(RESERVED) / f"{host}.gpu"
    try:
        text = f.read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return 0, ""
    if not text:
        return 0, ""
    head, _, rest = text.partition("\n")
    try:
        n = int(head.strip())
    except ValueError:
        # A malformed reservation is treated as one held slot rather than
        # as zero: failing toward "do not stack" is the safe direction.
        return 1, text
    return max(0, n), rest.strip()


# --------------------------------------------------------------------------
# eligibility


def receipts_present(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def is_runnable(item: dict, host: str, tags: set[str], has_gpu: bool) -> bool:
    """Can *this* worker run *this* item right now?

    Slot availability is checked separately -- this answers eligibility
    only, so that ``status`` can report "runnable but no free slot" as
    distinct from "not eligible here".
    """
    hosts = item.get("hosts")
    if hosts and host not in hosts:
        return False
    required = set(item.get("requires") or ())
    if not required.issubset(tags):
        return False
    if item.get("needs_gpu") and not has_gpu:
        return False
    return receipts_present(item.get("after") or [])


def sort_key(item: dict) -> tuple:
    # Higher priority first, then oldest first, so a burst of equal-priority
    # items drains in enqueue order instead of by filename.
    return (-int(item.get("priority", 50)), float(item.get("enqueued_at", 0)))


# --------------------------------------------------------------------------
# lease bookkeeping


def write_lease(item_id: str, host: str, pid: int) -> None:
    write_json_atomic(
        lease_path(item_id),
        {"host": host, "pid": pid, "beat": time.time()},
    )


def lease_is_stale(item_id: str) -> bool:
    lease = read_item(lease_path(item_id))
    if lease is None:
        # A claimed item with no readable lease is orphaned by definition:
        # the claimer died between the rename and the first lease write.
        return True
    return (time.time() - float(lease.get("beat", 0))) > LEASE_TIMEOUT_S


def claimed_on_host(host: str) -> list[dict]:
    out = []
    for p in sorted(qdir(CLAIMED).glob("*.json")):
        lease = read_item(lease_path(p.stem))
        if lease and lease.get("host") == host:
            item = read_item(p)
            if item:
                out.append(item)
    return out


# --------------------------------------------------------------------------
# state transitions


def move_item(item_id: str, src: str, dst: str, patch: dict | None = None) -> bool:
    src_path = item_path(src, item_id)
    item = read_item(src_path)
    if item is None:
        return False
    if patch:
        item.update(patch)
    dst_path = item_path(dst, item_id)
    try:
        write_json_atomic(dst_path, item)
        os.unlink(src_path)
    except FileNotFoundError:
        return False
    lease_path(item_id).unlink(missing_ok=True)
    return True


def reap(verbose: bool = False) -> int:
    """Requeue claims whose lease has gone stale.

    This is the recovery path for a box that died mid-job.  Without it the
    queue develops permanent holes and we are back to "nothing
    rescheduled", one level down from where we started.
    """
    n = 0
    for p in sorted(qdir(CLAIMED).glob("*.json")):
        item_id = p.stem
        if not lease_is_stale(item_id):
            continue
        item = read_item(p)
        if item is None:
            continue
        receipt = item.get("receipt")
        if receipt and Path(receipt).exists():
            # It finished; only the bookkeeping was lost.
            if move_item(item_id, CLAIMED, DONE, {"finished_at": time.time(),
                                                  "outcome": "receipt_found_after_lease_loss"}):
                n += 1
                if verbose:
                    print(f"[reap] {item_id}: receipt present -> done")
            continue
        attempts = int(item.get("attempts", 0))
        if attempts >= int(item.get("max_attempts", 3)):
            if move_item(item_id, CLAIMED, FAILED,
                         {"outcome": "lease_lost_max_attempts", "finished_at": time.time()}):
                n += 1
                if verbose:
                    print(f"[reap] {item_id}: lease lost, attempts exhausted -> failed")
            continue
        if move_item(item_id, CLAIMED, READY, {"outcome": None}):
            n += 1
            if verbose:
                print(f"[reap] {item_id}: stale lease -> requeued (attempt {attempts})")
    return n


def try_claim(item_id: str, host: str) -> dict | None:
    """Atomically take ownership of a ready item.

    ``rename`` is the whole concurrency story.  Exactly one worker's
    rename succeeds; every other worker sees ``FileNotFoundError`` and
    moves to the next candidate.
    """
    src = item_path(READY, item_id)
    dst = item_path(CLAIMED, item_id)
    try:
        os.rename(src, dst)
    except (FileNotFoundError, OSError):
        return None
    write_lease(item_id, host, os.getpid())
    item = read_item(dst)
    if item is not None:
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["claimed_by"] = host
        item["claimed_at"] = time.time()
        write_json_atomic(dst, item)
    return item


# --------------------------------------------------------------------------
# execution


def run_item(item: dict, host: str) -> str:
    """Execute one claimed item and return its terminal state name.

    The job runs as a direct child of the worker rather than as its own
    transient unit.  That is deliberate: if the worker dies the child dies
    with it, the lease goes stale, and the reaper requeues -- one recovery
    path instead of an orphaned unit nothing is watching.
    """
    item_id = item["id"]
    receipt = item.get("receipt")

    if receipt and Path(receipt).exists():
        # Idempotency gate.  This is what makes requeue-on-stale-lease safe
        # and what lets a campaign be re-enqueued wholesale after a partial
        # run without redoing the finished arms.
        move_item(item_id, CLAIMED, DONE,
                  {"outcome": "already_complete", "finished_at": time.time()})
        return "already_complete"

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (item.get("env") or {}).items()})
    # Never /tmp: it was cleared by an OOM once and took artifacts with it.
    local_tmp = Path.home() / ".pq-queue-tmp"
    local_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(local_tmp)
    env["PQ_ITEM_ID"] = item_id
    env["PQ_WORKER_HOST"] = host

    cwd = item.get("cwd") or str(Path.home())
    timeout = item.get("timeout_s")
    log = log_path(item_id)

    beat_deadline = time.time() + HEARTBEAT_S
    started = time.time()
    with open(log, "a", buffering=1) as fh:
        fh.write(f"\n==== pqwork {item_id} on {host} at {time.strftime('%F %T')} ====\n")
        fh.write(f"cwd={cwd}\ncmd={item['cmd']}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            ["bash", "-lc", item["cmd"]],
            cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            try:
                rc = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                now = time.time()
                if now >= beat_deadline:
                    write_lease(item_id, host, proc.pid)
                    beat_deadline = now + HEARTBEAT_S
                if timeout and (now - started) > float(timeout):
                    proc.kill()
                    rc = -9
                    fh.write(f"\n[pqwork] killed: exceeded timeout_s={timeout}\n")
                    break

    elapsed = time.time() - started
    # Gate on the receipt when one is declared; an exit code only says the
    # process ended, not that the artifact landed.
    if receipt:
        ok = Path(receipt).exists()
        outcome = "receipt_present" if ok else f"no_receipt (rc={rc})"
    else:
        ok = rc == 0
        outcome = f"rc={rc}"

    patch = {"outcome": outcome, "exit_code": rc,
             "elapsed_s": round(elapsed, 1), "finished_at": time.time()}
    if ok:
        move_item(item_id, CLAIMED, DONE, patch)
        return "done"

    attempts = int(item.get("attempts", 0))
    if attempts < int(item.get("max_attempts", 3)):
        move_item(item_id, CLAIMED, READY, patch)
        return "requeued"
    move_item(item_id, CLAIMED, FAILED, patch)
    return "failed"


# --------------------------------------------------------------------------
# the worker loop


def worker_loop(host: str, tags: set[str], gpu_slots: int, cpu_slots: int,
                once: bool) -> int:
    ensure_layout()
    has_gpu = gpu_slots > 0
    print(f"[pqwork] worker up on {host} tags={sorted(tags) or '-'} "
          f"gpu_slots={gpu_slots} cpu_slots={cpu_slots} queue={QUEUE_ROOT}", flush=True)
    last_held = None
    while True:
        reap(verbose=True)
        mine = claimed_on_host(host)
        gpu_busy = sum(1 for i in mine if i.get("needs_gpu"))
        cpu_busy = len(mine) - gpu_busy
        held, held_why = reserved_gpu_slots(host)
        gpu_capacity = max(0, gpu_slots - held)
        if held and held != last_held:
            print(f"[pqwork] {held} GPU slot(s) reserved outside the queue"
                  f"{': ' + held_why.splitlines()[0] if held_why else ''}"
                  f" -> gpu capacity {gpu_capacity}", flush=True)
        last_held = held

        candidates = []
        for p in qdir(READY).glob("*.json"):
            item = read_item(p)
            if item and is_runnable(item, host, tags, has_gpu):
                candidates.append(item)
        candidates.sort(key=sort_key)

        picked = None
        for item in candidates:
            if item.get("needs_gpu"):
                if gpu_busy >= gpu_capacity:
                    continue
            elif cpu_busy >= cpu_slots:
                continue
            picked = try_claim(item["id"], host)
            if picked is not None:
                break

        if picked is None:
            if once:
                print("[pqwork] nothing runnable here", flush=True)
                return 0
            time.sleep(POLL_S)
            continue

        print(f"[pqwork] running {picked['id']} "
              f"(gpu={bool(picked.get('needs_gpu'))}) -> {log_path(picked['id'])}", flush=True)
        state = run_item(picked, host)
        print(f"[pqwork] {picked['id']}: {state}", flush=True)
        if once:
            return 0


# --------------------------------------------------------------------------
# CLI


def cmd_enqueue(a: argparse.Namespace) -> int:
    ensure_layout()
    for state in (CLAIMED, DONE, FAILED, READY):
        if item_path(state, a.id).exists() and not a.force:
            print(f"error: item '{a.id}' already exists in {state}/ "
                  f"(use --force to replace)", file=sys.stderr)
            return 1
    if a.force:
        for state in (READY, DONE, FAILED):
            item_path(state, a.id).unlink(missing_ok=True)
    item = {
        "id": a.id,
        "desc": a.desc or "",
        "cmd": a.cmd,
        "cwd": a.cwd,
        "hosts": a.host or None,
        "requires": a.require or [],
        "needs_gpu": bool(a.gpu),
        "receipt": a.receipt,
        "after": a.after or [],
        "priority": a.priority,
        "timeout_s": a.timeout_s,
        "max_attempts": a.max_attempts,
        "env": dict(kv.split("=", 1) for kv in (a.env or [])),
        "enqueued_at": time.time(),
        "attempts": 0,
    }
    write_json_atomic(item_path(READY, a.id), item)
    print(f"queued {a.id} -> {item_path(READY, a.id)}")
    return 0


def _fmt_age(ts: float | None) -> str:
    if not ts:
        return "-"
    d = time.time() - float(ts)
    if d < 90:
        return f"{int(d)}s"
    if d < 5400:
        return f"{int(d/60)}m"
    return f"{d/3600:.1f}h"


def cmd_status(a: argparse.Namespace) -> int:
    ensure_layout()
    for f in sorted(qdir(RESERVED).glob("*.gpu")):
        host = f.stem
        n, why = reserved_gpu_slots(host)
        print(f"RESERVED  {host}: {n} GPU slot(s) -- {why.splitlines()[0] if why else '?'}")
    for state in (CLAIMED, READY, FAILED, DONE):
        items = [read_item(p) for p in sorted(qdir(state).glob("*.json"))]
        items = [i for i in items if i]
        if state == DONE:
            items.sort(key=lambda i: -float(i.get("finished_at", 0)))
            items = items[:a.done_limit]
        else:
            items.sort(key=sort_key)
        if not items:
            continue
        print(f"\n{state.upper()} ({len(list(qdir(state).glob('*.json')))})")
        for i in items:
            bits = []
            if i.get("needs_gpu"):
                bits.append("gpu")
            if i.get("hosts"):
                bits.append("@" + ",".join(i["hosts"]))
            if state == CLAIMED:
                lease = read_item(lease_path(i["id"]))
                where = lease.get("host") if lease else "?"
                stale = " STALE" if lease_is_stale(i["id"]) else ""
                bits.append(f"on {where} {_fmt_age(i.get('claimed_at'))}{stale}")
            elif state == READY:
                pending = [p for p in (i.get("after") or []) if not Path(p).exists()]
                if pending:
                    bits.append(f"blocked on {len(pending)} receipt(s)")
                bits.append(f"waiting {_fmt_age(i.get('enqueued_at'))}")
            elif state in (DONE, FAILED):
                bits.append(str(i.get("outcome")))
                if i.get("elapsed_s"):
                    bits.append(f"{i['elapsed_s']}s")
            print(f"  p{i.get('priority', 50):<3} {i['id']:<38} {' | '.join(bits)}")
            if a.verbose and i.get("desc"):
                print(f"        {i['desc']}")
    return 0


def cmd_reap(a: argparse.Namespace) -> int:
    ensure_layout()
    n = reap(verbose=True)
    print(f"reaped {n}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    host = a.host or socket.gethostname().split(".")[0]
    return worker_loop(host, set(a.tag or []), a.gpu_slots, a.cpu_slots, a.once)


def cmd_cancel(a: argparse.Namespace) -> int:
    for state in (READY, CLAIMED):
        if item_path(state, a.id).exists():
            move_item(a.id, state, FAILED,
                      {"outcome": "cancelled", "finished_at": time.time()})
            print(f"cancelled {a.id} (was {state})")
            return 0
    print(f"error: no ready/claimed item '{a.id}'", file=sys.stderr)
    return 1


def cmd_requeue(a: argparse.Namespace) -> int:
    for state in (FAILED, DONE):
        if item_path(state, a.id).exists():
            move_item(a.id, state, READY, {"outcome": None, "attempts": 0})
            print(f"requeued {a.id} (was {state})")
            return 0
    print(f"error: no failed/done item '{a.id}'", file=sys.stderr)
    return 1


def cmd_reserve(a: argparse.Namespace) -> int:
    ensure_layout()
    f = qdir(RESERVED) / f"{a.host}.gpu"
    if a.release:
        existed = f.exists()
        f.unlink(missing_ok=True)
        print(f"released GPU reservation on {a.host}" if existed
              else f"no GPU reservation held on {a.host}")
        return 0
    body = f"{a.slots}\n{a.why or 'unattributed'}\n"
    f.write_text(body)
    print(f"reserved {a.slots} GPU slot(s) on {a.host}: {a.why or 'unattributed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pqwork", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("enqueue", help="add a work item to the shared queue")
    e.add_argument("--id", required=True, help="unique item id (also the log filename)")
    e.add_argument("--cmd", required=True, help="shell command, run under bash -lc")
    e.add_argument("--desc", default="", help="one-line human description")
    e.add_argument("--cwd", default=str(Path.home()))
    e.add_argument("--host", action="append",
                   help="pin to a host; repeatable. Required when the item's "
                        "inputs live on box-local storage such as /home/rob.")
    e.add_argument("--require", action="append",
                   help="worker tag this item needs; repeatable")
    e.add_argument("--gpu", action="store_true", help="consumes a GPU slot")
    e.add_argument("--receipt", help="path that must exist for the item to count "
                                     "as complete; also the skip-if-done gate")
    e.add_argument("--after", action="append",
                   help="receipt path that must exist before this item is runnable")
    e.add_argument("--priority", type=int, default=50, help="higher runs first")
    e.add_argument("--timeout-s", type=float, default=None, dest="timeout_s")
    e.add_argument("--max-attempts", type=int, default=3, dest="max_attempts")
    e.add_argument("--env", action="append", metavar="K=V")
    e.add_argument("--force", action="store_true", help="replace an existing item")
    e.set_defaults(func=cmd_enqueue)

    r = sub.add_parser("run", help="run the pull loop for this box")
    r.add_argument("--host", default=None)
    r.add_argument("--tag", action="append", help="capability tag; repeatable")
    r.add_argument("--gpu-slots", type=int, default=1, dest="gpu_slots",
                   help="concurrent GPU items. Default 1: unified memory means "
                        "two big jobs contend for one pool.")
    r.add_argument("--cpu-slots", type=int, default=2, dest="cpu_slots")
    r.add_argument("--once", action="store_true", help="claim at most one item, then exit")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="show the queue")
    s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--done-limit", type=int, default=10, dest="done_limit")
    s.set_defaults(func=cmd_status)

    p = sub.add_parser("reap", help="requeue items whose lease went stale")
    p.set_defaults(func=cmd_reap)

    c = sub.add_parser("cancel", help="move a ready/claimed item to failed")
    c.add_argument("id")
    c.set_defaults(func=cmd_cancel)

    q = sub.add_parser("requeue", help="move a failed/done item back to ready")
    q.add_argument("id")
    q.set_defaults(func=cmd_requeue)

    v = sub.add_parser("reserve", help="hold GPU slots for work outside the queue")
    v.add_argument("--host", default=None)
    v.add_argument("--slots", type=int, default=1)
    v.add_argument("--why", help="who holds it and why; shown by status")
    v.add_argument("--release", action="store_true", help="give the slots back")
    v.set_defaults(func=cmd_reserve)

    a = ap.parse_args(argv)
    if getattr(a, "host", None) is None and a.command == "reserve":
        a.host = socket.gethostname().split(".")[0]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
