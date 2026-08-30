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
import collections
import json
import os
import signal
import socket
import subprocess
import sys
import threading
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

# Memory admission and eviction.  On GB10 the GPU and the host share one
# physical pool, so /proc/meminfo MemAvailable *is* the GPU memory ceiling --
# every dedicated GPU-memory source on this box reads null or a fake zero.
# There is also no swap (SwapTotal: 0), so MemAvailable reaching the wall
# means the kernel OOM killer picks a victim immediately, with no cushion and
# by its own heuristic.  It has picked wrong here before.  Evicting early is
# how we choose the victim ourselves.
# An evicted item must not be re-admitted immediately.  Aggressive
# admission and zero backoff together are a livelock: the box evicts, sees
# headroom, re-admits the same job, and evicts it again, forever, making no
# progress while looking busy.  Backoff is the necessary complement to
# "start it and find out" -- observed doing exactly this on 2026-08-29
# before the cooldown existed.
EVICT_BACKOFF_BASE_S = 60.0
EVICT_BACKOFF_CAP_S = 1800.0
MAX_EVICTIONS = 5

MEM_SAMPLE_S = 2.0
MEM_RATE_WINDOW_S = 30.0
EVICT_GRACE_S = 10.0


# --------------------------------------------------------------------------
# layout helpers


def qdir(name: str) -> Path:
    return QUEUE_ROOT / name


def queue_reachable() -> bool:
    """Is the shared queue root actually present?

    Kept separate from ``ensure_layout`` so a worker can tell "the share is
    away" apart from "the share is here but empty". Creating the layout on a
    *missing* autofs mountpoint would silently build a local shadow queue
    that no other box can see, which is worse than doing nothing.
    """
    try:
        return QUEUE_ROOT.is_dir()
    except OSError:
        return False


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


def reserved_memory_gb(host: str) -> tuple[float, str]:
    """Memory on this box held by something outside the queue, in GB.

    The governor can evict its own items, but it cannot evict a job it did
    not start -- and on 2026-08-30 a hand-launched job took sparklina from
    68 GB to 4 GB in 100 seconds, filled swap, and left the box unreachable
    over SSH for five minutes. No admission policy over MemAvailable alone
    can prevent that, because MemAvailable does not say how much more the
    neighbour intends to take.

    So the neighbour declares it, the same way it declares a GPU slot. The
    declared amount is held out of headroom for as long as the file exists,
    which makes the queue conservative exactly where it cannot police the
    outcome.
    """
    f = qdir(RESERVED) / f"{host}.mem"
    try:
        text = f.read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return 0.0, ""
    if not text:
        return 0.0, ""
    head, _, rest = text.partition("\n")
    try:
        return max(0.0, float(head.strip())), rest.strip()
    except ValueError:
        return 0.0, text


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


def receipt_satisfied(path: str) -> bool:
    """Is this receipt genuinely present and non-empty?

    Opened and read rather than stat'd, for two reasons. An empty file is not
    a receipt -- a shell redirect creates one the instant it opens the target,
    before the work has produced anything, and a job that writes its receipt
    as its final act can be interrupted mid-write. And on NFS a stat can be
    served from the attribute cache, while an open forces a fresh lookup
    under close-to-open consistency.

    This must agree exactly with whatever a job's own command tests. When the
    queue gated on ``exists()`` while jobs gated on ``test -s``, the two
    disagreed and an item reached ``done/`` carrying no receipt at all.
    """
    try:
        with open(path, "rb") as fh:
            return bool(fh.read(1))
    except OSError:
        return False


def receipts_present(paths: list[str]) -> bool:
    return all(receipt_satisfied(p) for p in paths)


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
    if time.time() < float(item.get("not_before") or 0):
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
        if receipt and receipt_satisfied(receipt):
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


def swap_free_gb() -> float:
    """Swap still available, in GB; 0.0 when swap is off.

    Swap is not extra capacity here -- paging a multi-gigabyte tensor
    workload is ruinous for throughput -- but it is *time*.  It converts a
    hard OOM kill chosen by the kernel into a slow slide we can detect and
    act on, which is the difference between choosing the victim and being
    handed one. Note sparky ran with swap disabled from 2026-07-25 as an
    OOM-livelock mitigation, so this legitimately reads 0 there.
    """
    try:
        with open("/proc/meminfo") as fh:
            vals = {}
            for line in fh:
                k, _, rest = line.partition(":")
                if k in ("SwapFree", "SwapTotal"):
                    vals[k] = int(rest.split()[0]) / (1024.0 * 1024.0)
            return vals.get("SwapFree", 0.0)
    except OSError:
        return 0.0


def mem_available_gb() -> float:
    """Free memory this box can still hand out, in GB.

    MemAvailable rather than MemFree: it already accounts for reclaimable
    page cache, which is the number that actually predicts whether the next
    allocation succeeds.  On GB10 it is also the GPU ceiling, because the GPU
    and the host share one physical pool.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")


class Job:
    """One claimed item executing as a child process group."""

    def __init__(self, item: dict, host: str):
        self.item = item
        self.id = item["id"]
        self.host = host
        self.proc: subprocess.Popen | None = None
        self.started = time.time()
        self.avail_at_start = mem_available_gb()
        self.evicted = False
        self.state: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def declared_gb(self) -> float:
        try:
            return float(self.item.get("mem_gb") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def evictable(self) -> bool:
        return bool(self.item.get("evictable", True))

    def _run(self) -> None:
        item, host = self.item, self.host
        receipt = item.get("receipt")
        if receipt and receipt_satisfied(receipt):
            # Idempotency gate.  This is what makes requeue-on-eviction and
            # requeue-on-stale-lease safe, and what lets a partially finished
            # campaign be re-enqueued wholesale without redoing its finished
            # arms.
            move_item(self.id, CLAIMED, DONE,
                      {"outcome": "already_complete", "finished_at": time.time()})
            self.state = "already_complete"
            return

        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (item.get("env") or {}).items()})
        # Never /tmp: it was cleared by an OOM once and took artifacts with it.
        local_tmp = Path.home() / ".pq-queue-tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(local_tmp)
        env["PQ_ITEM_ID"] = self.id
        env["PQ_WORKER_HOST"] = host

        log = log_path(self.id)
        with open(log, "a", buffering=1) as fh:
            fh.write(f"\n==== pqwork {self.id} on {host} at "
                     f"{time.strftime('%F %T')} ====\n")
            fh.write(f"cwd={item.get('cwd')}\ncmd={item['cmd']}\n"
                     f"mem_available_at_start={self.avail_at_start:.1f}GB\n\n")
            # start_new_session so the whole job -- and anything it spawns --
            # is one process group we can signal as a unit. Killing only the
            # shell would leave the actual worker holding the memory.
            self.proc = subprocess.Popen(
                ["bash", "-lc", item["cmd"]],
                cwd=item.get("cwd") or str(Path.home()), env=env,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            )
            rc = self.proc.wait()
            elapsed = time.time() - self.started
            if self.evicted:
                fh.write(f"\n[pqwork] evicted to avoid OOM after {elapsed:.0f}s\n")

        if self.evicted:
            # An eviction is not a failure of the work, so it does not spend
            # an attempt -- a box under memory pressure would otherwise burn
            # through max_attempts on items that never got to run. It does
            # spend an *eviction*, which backs the item off and eventually
            # concludes it does not fit here.
            evictions = int(item.get("evictions", 0)) + 1
            if evictions >= MAX_EVICTIONS:
                move_item(self.id, CLAIMED, FAILED,
                          {"outcome": f"evicted_{evictions}x_does_not_fit",
                           "evictions": evictions, "finished_at": time.time()})
                self.state = "does_not_fit"
                return
            backoff = min(EVICT_BACKOFF_BASE_S * (2 ** (evictions - 1)),
                          EVICT_BACKOFF_CAP_S)
            move_item(self.id, CLAIMED, READY,
                      {"outcome": "evicted_for_memory",
                       "evictions": evictions,
                       "not_before": time.time() + backoff,
                       "attempts": max(0, int(item.get("attempts", 1)) - 1)})
            self.state = f"evicted (backoff {backoff:.0f}s)"
            return

        receipt = item.get("receipt")
        if receipt:
            ok = receipt_satisfied(receipt)
            outcome = "receipt_present" if ok else f"no_receipt (rc={rc})"
        else:
            ok = rc == 0
            outcome = f"rc={rc}"
        patch = {"outcome": outcome, "exit_code": rc,
                 "elapsed_s": round(elapsed, 1), "finished_at": time.time()}
        if ok:
            move_item(self.id, CLAIMED, DONE, patch)
            self.state = "done"
        elif int(item.get("attempts", 0)) < int(item.get("max_attempts", 3)):
            move_item(self.id, CLAIMED, READY, patch)
            self.state = "requeued"
        else:
            move_item(self.id, CLAIMED, FAILED, patch)
            self.state = "failed"

    def start(self) -> None:
        self.thread.start()

    def alive(self) -> bool:
        return self.thread.is_alive()

    def evict(self) -> None:
        """Stop this job so its memory comes back, and requeue it.

        SIGTERM to the process group first so a job with a cleanup handler
        can flush a partial receipt, then SIGKILL. Signalling the group
        rather than the shell matters: the shell is not the process holding
        the tens of gigabytes.
        """
        self.evicted = True
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + EVICT_GRACE_S
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.5)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class MemoryGovernor:
    """Decides what may start and what must stop, from MemAvailable.

    The trigger is a *trajectory*, not a level.  A level alone is either too
    conservative (refusing work on a box that is merely warm) or too late (by
    the time MemAvailable is low, a job allocating fast is already past the
    point where killing it can help).  Projecting the current drop rate
    forward answers the question that actually matters -- will we hit the
    wall before an eviction can free anything -- and that horizon is roughly
    the time it takes to signal a process group and have the kernel reclaim
    its pages.
    """

    def __init__(self, reserve_gb: float, horizon_s: float, settle_s: float,
                 admission: str = "optimistic"):
        self.reserve_gb = reserve_gb
        self.horizon_s = horizon_s
        self.settle_s = settle_s
        self.admission = admission
        self.external_hold_gb = 0.0
        self.samples: collections.deque = collections.deque()

    def sample(self) -> float:
        now, avail = time.time(), mem_available_gb()
        self.samples.append((now, avail))
        while self.samples and now - self.samples[0][0] > MEM_RATE_WINDOW_S:
            self.samples.popleft()
        return avail

    def drop_rate_gb_s(self) -> float:
        """GB/s at which MemAvailable is falling; negative means recovering."""
        if len(self.samples) < 2:
            return 0.0
        (t0, a0), (t1, a1) = self.samples[0], self.samples[-1]
        dt = t1 - t0
        return (a0 - a1) / dt if dt > 0 else 0.0

    def set_external_hold(self, gb: float) -> None:
        self.external_hold_gb = gb

    def headroom(self, jobs: list) -> float:
        """Memory we may still commit, after unrealized declarations.

        A job that started seconds ago may not have allocated yet, so its
        declared footprint is not visible in MemAvailable.  Counting it until
        it has had time to settle prevents admitting three large jobs in the
        same instant on the strength of memory the first one is about to take.
        """
        now = time.time()
        unrealized = sum(
            j.declared_gb for j in jobs
            if j.declared_gb and (now - j.started) < self.settle_s
        )
        if not self.samples:
            return 0.0
        return (self.samples[-1][1] - self.reserve_gb
                - self.external_hold_gb - unrealized)

    def may_admit(self, item: dict, jobs: list) -> bool:
        """Optimistic by default: start it and find out.

        Rob, 2026-08-29: *"I'd rather be overly aggressive enqueuing items
        and then killing them than just having them wait in line."*  So a
        declared ``mem_gb`` larger than current headroom is not a refusal --
        estimates are guesses, and a job idling in ``ready/`` while the box
        has memory free is a certain loss, where an eviction is only a
        possible one.  The declaration still does real work: it is held
        against headroom during the settle window so several large jobs
        cannot all start in the same instant on the strength of memory the
        first one has not taken yet.

        ``strict`` restores refusal-on-declaration for a box where restarting
        work is genuinely expensive.
        """
        headroom = self.headroom(jobs)
        if self.admission == "strict":
            try:
                want = float(item.get("mem_gb") or 0.0)
            except (TypeError, ValueError):
                want = 0.0
            return headroom >= want
        return headroom > 0

    def must_evict(self, jobs: list) -> bool:
        """Are we projected to hit the wall before an eviction could help?

        Swap extends the horizon rather than the capacity: with swap on,
        crossing the reserve starts a slow slide instead of an immediate
        kernel kill, so there is more time to act and the trigger can wait
        for a real trend. With swap off -- sparky's state since 2026-07-25 --
        there is no cushion at all and the projection has to fire earlier.
        """
        if not self.samples or not jobs:
            return False
        avail = self.samples[-1][1]
        if avail <= self.reserve_gb:
            return True
        horizon = self.horizon_s
        if swap_free_gb() <= 0.5:
            horizon *= 2.0
        projected = avail - self.drop_rate_gb_s() * horizon
        return projected < self.reserve_gb


def choose_victim(jobs: list):
    """The newest, lowest-priority evictable job.

    Newest because it is the one that pushed the box over and has the least
    sunk compute to lose; lowest priority first so gold-path work outlives
    speculative work. A job marked non-evictable is never chosen -- which
    means a box can still OOM if everything running is pinned, and that is
    the operator's declared choice rather than a surprise.
    """
    victims = [j for j in jobs if j.evictable and j.alive()]
    if not victims:
        return None
    return sorted(victims, key=lambda j: (int(j.item.get("priority", 50)), -j.started))[0]


def wait_for_queue(once: bool) -> None:
    """Block until the shared queue root appears, backing off as we go.

    A worker whose queue has gone away must wait, never exit. On 2026-08-30
    the opposite arrangement took a box out of the pool for an hour: the
    unit ran its own binary from the NFS path, the mount dropped, systemd
    restarted it every 15 s, each restart re-triggered the autofs mount, and
    the automount unit hit ``mount-start-limit-hit`` and failed *permanently*.
    A transient mount failure became a persistent one because the retry was
    too eager and lived in the wrong place.

    Hence both halves of the fix: the binary is installed box-locally so the
    worker can run at all without the share, and the wait backs off so
    polling can never re-trip systemd's start limit.
    """
    delay = 15.0
    announced = False
    while not queue_reachable():
        if once:
            return
        if not announced:
            print(f"[pqwork] queue root {QUEUE_ROOT} is not reachable; "
                  f"waiting (backing off, not restarting)", flush=True)
            announced = True
        time.sleep(delay)
        delay = min(delay * 2, 300.0)
    if announced:
        print(f"[pqwork] queue root {QUEUE_ROOT} is back", flush=True)


def worker_loop(host: str, tags: set, gpu_slots: int, cpu_slots: int,
                once: bool, reserve_gb: float, horizon_s: float,
                settle_s: float, admission: str = "optimistic") -> int:
    wait_for_queue(once)
    ensure_layout()
    has_gpu = gpu_slots > 0
    gov = MemoryGovernor(reserve_gb, horizon_s, settle_s, admission)
    jobs: list = []
    print(f"[pqwork] worker up on {host} tags={sorted(tags) or '-'} "
          f"gpu_slots={gpu_slots} cpu_slots={cpu_slots} "
          f"mem_reserve={reserve_gb:.0f}GB horizon={horizon_s:.0f}s "
          f"admission={admission} swap_free={swap_free_gb():.0f}GB "
          f"queue={QUEUE_ROOT}", flush=True)
    last_held = None
    last_mem_hold = 0.0
    last_beat = 0.0
    while True:
        if not queue_reachable():
            # Losing the share mid-run is not fatal: running jobs keep their
            # own local state, and bookkeeping resumes when it returns.
            wait_for_queue(once)
            ensure_layout()
        avail = gov.sample()
        jobs = [j for j in jobs if j.alive()]

        # Memory policing comes before admission: give the box back its
        # headroom before considering whether to take on more.
        if gov.must_evict(jobs):
            victim = choose_victim(jobs)
            if victim is not None:
                rate = gov.drop_rate_gb_s()
                print(f"[pqwork] EVICT {victim.id}: MemAvailable {avail:.1f}GB "
                      f"falling {rate:.2f}GB/s -> projected "
                      f"{avail - rate * horizon_s:.1f}GB below the "
                      f"{reserve_gb:.0f}GB reserve", flush=True)
                victim.evict()

        now = time.time()
        if now - last_beat >= HEARTBEAT_S:
            for j in jobs:
                write_lease(j.id, host, j.proc.pid if j.proc else os.getpid())
            last_beat = now

        reap(verbose=True)
        mine = claimed_on_host(host)
        gpu_busy = sum(1 for i in mine if i.get("needs_gpu"))
        cpu_busy = len(mine) - gpu_busy
        held_mem, held_mem_why = reserved_memory_gb(host)
        gov.set_external_hold(held_mem)
        if held_mem != last_mem_hold:
            if held_mem:
                print(f"[pqwork] {held_mem:.0f}GB held outside the queue"
                      f"{': ' + held_mem_why.splitlines()[0] if held_mem_why else ''}",
                      flush=True)
            last_mem_hold = held_mem
        held, held_why = reserved_gpu_slots(host)
        gpu_capacity = max(0, gpu_slots - held)
        if held != last_held:
            if held:
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

        started_any = False
        for item in candidates:
            if item.get("needs_gpu"):
                if gpu_busy >= gpu_capacity:
                    continue
            elif cpu_busy >= cpu_slots:
                continue
            if not gov.may_admit(item, jobs):
                continue
            claimed = try_claim(item["id"], host)
            if claimed is None:
                continue
            job = Job(claimed, host)
            jobs.append(job)
            job.start()
            started_any = True
            print(f"[pqwork] running {job.id} (gpu={bool(claimed.get('needs_gpu'))}, "
                  f"declared={job.declared_gb or '-'}GB, "
                  f"headroom={gov.headroom(jobs):.1f}GB) -> {log_path(job.id)}",
                  flush=True)
            if claimed.get("needs_gpu"):
                gpu_busy += 1
            else:
                cpu_busy += 1
            if once:
                # --once means exactly one item, as its help says. The
                # threaded rewrite briefly let it claim a whole slot's worth,
                # which made the order several jobs recorded their own start
                # in a race -- it passed on an idle box and failed at load 28.
                break

        if once:
            if not started_any and not jobs:
                print("[pqwork] nothing runnable here", flush=True)
                return 0
            for j in jobs:
                j.thread.join()
                print(f"[pqwork] {j.id}: {j.state}", flush=True)
            return 0

        for j in list(jobs):
            if not j.alive() and j.state:
                print(f"[pqwork] {j.id}: {j.state}", flush=True)
        time.sleep(MEM_SAMPLE_S if jobs else POLL_S)


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
        "mem_gb": a.mem_gb,
        "evictable": not a.no_evict,
        "receipt": a.receipt,
        "after": a.after or [],
        "priority": a.priority,
        "timeout_s": a.timeout_s,
        "max_attempts": a.max_attempts,
        "env": dict(kv.split("=", 1) for kv in (a.env or [])),
        "enqueued_at": time.time(),
        "attempts": 0,
        "evictions": 0,
        "not_before": 0,
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
        mem, _ = reserved_memory_gb(host)
        held = f"{n} GPU slot(s)" + (f" + {mem:.0f}GB" if mem else "")
        print(f"RESERVED  {host}: {held} -- {why.splitlines()[0] if why else '?'}")
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
                pending = [p for p in (i.get("after") or []) if not receipt_satisfied(p)]
                if pending:
                    bits.append(f"blocked on {len(pending)} receipt(s)")
                cool = float(i.get("not_before") or 0) - time.time()
                if cool > 0:
                    bits.append(f"cooling {int(cool)}s after "
                                f"{i.get('evictions')} eviction(s)")
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
    return worker_loop(host, set(a.tag or []), a.gpu_slots, a.cpu_slots,
                       a.once, a.mem_reserve_gb, a.evict_horizon_s,
                       a.mem_settle_s, a.admission)


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
            move_item(a.id, state, READY, {"outcome": None, "attempts": 0,
                                           "evictions": 0, "not_before": 0})
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
        (qdir(RESERVED) / f"{a.host}.mem").unlink(missing_ok=True)
        print(f"released GPU reservation on {a.host}" if existed
              else f"no GPU reservation held on {a.host}")
        return 0
    f.write_text(f"{a.slots}\n{a.why or 'unattributed'}\n")
    msg = f"reserved {a.slots} GPU slot(s) on {a.host}"
    if a.mem_gb:
        (qdir(RESERVED) / f"{a.host}.mem").write_text(
            f"{a.mem_gb}\n{a.why or 'unattributed'}\n")
        msg += f" and {a.mem_gb:.0f}GB"
    print(f"{msg}: {a.why or 'unattributed'}")
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
    e.add_argument("--mem-gb", type=float, default=None, dest="mem_gb",
                   help="expected peak memory. Optional: an item that "
                        "declares nothing is admitted optimistically and "
                        "policed by eviction, which is the intended way to "
                        "find out what something costs.")
    e.add_argument("--no-evict", action="store_true", dest="no_evict",
                   help="never evict this item under memory pressure. Use for "
                        "gold-path work whose restart is expensive; note that "
                        "a box where everything is pinned can still OOM.")
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
    r.add_argument("--gpu-slots", type=int, default=6, dest="gpu_slots",
                   help="concurrency cap on GPU items. Memory, not this "
                        "number, is the real gate -- the cap only bounds how "
                        "many can be in flight at once.")
    r.add_argument("--cpu-slots", type=int, default=2, dest="cpu_slots")
    r.add_argument("--once", action="store_true", help="claim at most one item, then exit")
    r.add_argument("--mem-reserve-gb", type=float, default=8.0,
                   dest="mem_reserve_gb",
                   help="MemAvailable floor. This is a declared operating "
                        "reserve, not a derived constant: it has to cover "
                        "whatever the kernel and everything off-queue need, "
                        "which no measurement on this box reports.")
    r.add_argument("--evict-horizon-s", type=float, default=20.0,
                   dest="evict_horizon_s",
                   help="evict when the current drop rate projects through "
                        "the reserve within this many seconds")
    r.add_argument("--admission", choices=("optimistic", "strict"),
                   default="optimistic",
                   help="optimistic (default) starts work whenever any "
                        "headroom exists and relies on eviction; strict "
                        "refuses an item whose declared mem_gb exceeds "
                        "headroom.")
    r.add_argument("--mem-settle-s", type=float, default=120.0,
                   dest="mem_settle_s",
                   help="how long a just-started job's declared mem_gb is "
                        "held against headroom before MemAvailable is "
                        "trusted to reflect it")
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
    v.add_argument("--mem-gb", type=float, default=None, dest="mem_gb",
                   help="memory this off-queue job will take. Held out of "
                        "headroom so the queue does not admit work onto "
                        "memory a neighbour has not allocated yet.")
    v.add_argument("--why", help="who holds it and why; shown by status")
    v.add_argument("--release", action="store_true", help="give the slots back")
    v.set_defaults(func=cmd_reserve)

    a = ap.parse_args(argv)
    if getattr(a, "host", None) is None and a.command == "reserve":
        a.host = socket.gethostname().split(".")[0]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
