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
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import statistics
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
REAPED = "reaped"
WORKERS = "workers"
DEPLOY_HOLD = ".deploying"
STATE_DIRS = (READY, CLAIMED, DONE, FAILED, LOGS, RESERVED, REAPED, WORKERS)

# A multi-host external launch is represented by one record in one ledger.
# Publishing the ledger with rename makes a reservation set visible on every
# host all at once.  The admission guard serializes that publication with the
# worker's final ready->claimed transition; without both pieces, an atomic
# manifest could still race a claim made from a stale capacity sample.
RESERVATION_LEDGER = "reservations.v2.json"
ADMISSION_LOCK = ".admission-lock"
ADMISSION_LOCK_STALE_S = 120.0
ADMISSION_LOCK_WAIT_S = 10.0

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
# An evicted item must not be re-admitted immediately.  Aggressive admission
# and zero cooldown together are a livelock, but exponential backoff merely
# makes the same losing race take longer.  Every policy eviction therefore
# earns scheduler age; after two, the next attempt reserves its footprint and
# is non-evictable.  The fixed cooldown damps immediate churn without turning
# lost races into an ever-longer bench.
EVICT_BACKOFF_S = 60.0
EVICTIONS_BEFORE_PROTECTION = 2

MEM_SAMPLE_S = 2.0
MEM_RATE_WINDOW_S = 30.0
MIN_TREND_INTERVALS = 3
ATTRIBUTION_MIN_FRACTION = 0.5
CGROUP_LOOKUP_RETRY_S = 5.0
EVICT_GRACE_S = 10.0


# --------------------------------------------------------------------------
# layout helpers


def qdir(name: str) -> Path:
    return QUEUE_ROOT / name


def deployment_hold_path() -> Path:
    return QUEUE_ROOT / DEPLOY_HOLD


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


def reservation_ledger_path() -> Path:
    return qdir(RESERVED) / RESERVATION_LEDGER


def admission_lock_path() -> Path:
    return qdir(RESERVED) / ADMISSION_LOCK


def item_path(state: str, item_id: str) -> Path:
    return qdir(state) / f"{item_id}.json"


def lease_path(item_id: str) -> Path:
    return qdir(CLAIMED) / f"{item_id}.lease"


def log_path(item_id: str) -> Path:
    return qdir(LOGS) / f"{item_id}.log"


def worker_heartbeat_path(host: str) -> Path:
    return qdir(WORKERS) / f"{host}.json"


def read_item(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file in the same directory, then rename.

    Same directory matters: rename is only atomic within a filesystem, and
    a reader must never observe a half-written item.
    """
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.rename(tmp, path)


def loaded_source_sha256() -> str:
    """Hash the exact script bytes this interpreter loaded."""
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_worker_heartbeat(host: str, source_sha256: str,
                           started_at: float) -> None:
    write_json_atomic(worker_heartbeat_path(host), {
        "host": host,
        "pid": os.getpid(),
        "source_sha256": source_sha256,
        "started_at": started_at,
        "beat": time.time(),
    })


class AdmissionLockTimeout(RuntimeError):
    """The short queue-admission critical section could not be entered."""


class AdmissionLockStale(AdmissionLockTimeout):
    """A possibly-live old holder prevents safe timestamp-based recovery."""


def _admission_lock_age(lock: Path) -> float | None:
    """Return lock age for diagnostics; never break a lock by timestamp."""
    owner = read_item(lock / "owner.json")
    try:
        acquired_at = float((owner or {}).get("acquired_at", lock.stat().st_mtime))
    except (OSError, TypeError, ValueError):
        return None
    return max(0.0, time.time() - acquired_at)


@contextlib.contextmanager
def admission_guard(timeout_s: float = ADMISSION_LOCK_WAIT_S):
    """Serialize reservation changes with final worker admission on NFS.

    ``flock`` is deliberately not used: the queue lives on NFSv4 and its
    advisory-lock semantics depend on mount configuration.  Directory
    creation is one atomic namespace operation.  Critical sections contain
    only small JSON reads, an atomic rename, and (for workers) one claim.
    """
    ensure_layout()
    lock = admission_lock_path()
    token = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            os.mkdir(lock)
        except FileExistsError:
            age = _admission_lock_age(lock)
            if age is not None and age > ADMISSION_LOCK_STALE_S:
                # Timestamp-based lock stealing is not fencing. A paused old
                # holder could resume and publish after its directory was
                # moved, violating the exact atomicity this guard exists to
                # provide. Surface an explicit blocker for operator-verified
                # recovery instead.
                raise AdmissionLockStale(
                    f"stale queue admission guard at {lock} "
                    f"(age {age:.1f}s); refusing unsafe lock stealing")
            if time.monotonic() >= deadline:
                raise AdmissionLockTimeout(
                    f"timed out waiting for queue admission guard {lock}")
            time.sleep(0.05)
            continue
        try:
            write_json_atomic(
                lock / "owner.json",
                {"token": token, "pid": os.getpid(),
                 "host": socket.gethostname().split(".")[0],
                 "acquired_at": time.time()},
            )
        except Exception:
            try:
                (lock / "owner.json").unlink(missing_ok=True)
                (lock / f"owner.json.tmp.{os.getpid()}").unlink(
                    missing_ok=True)
                lock.rmdir()
            except OSError:
                pass
            raise
        break
    try:
        yield
    finally:
        owner = read_item(lock / "owner.json")
        if owner and owner.get("token") == token:
            try:
                (lock / "owner.json").unlink(missing_ok=True)
                lock.rmdir()
            except OSError:
                # An operator can move a diagnosed stale lock after proving
                # its owner dead. Token checking prevents this holder from
                # removing a successor's lock if it later resumes.
                pass


def _empty_reservation_ledger() -> dict:
    return {"version": 2, "updated_at": 0.0,
            "capacities": {}, "reservations": {}}


def read_reservation_ledger(*, strict: bool = False) -> dict:
    """Read and validate the v2 reservation ledger.

    Capacity readers fail closed on corruption.  Mutators use ``strict`` and
    refuse to replace evidence they cannot understand.
    """
    path = reservation_ledger_path()
    try:
        payload = json.loads(path.read_text())
        if (not isinstance(payload, dict)
                or payload.get("version") != 2
                or not isinstance(payload.get("capacities"), dict)
                or not isinstance(payload.get("reservations"), dict)):
            raise ValueError("invalid reservation ledger schema")
        for host, capacity in payload["capacities"].items():
            if (not isinstance(host, str) or not isinstance(capacity, dict)):
                raise ValueError("invalid host capacity record")
            parse_host_resource(
                f"{host}:0:0:{capacity.get('gpu_capacity')}:"
                f"{capacity.get('mem_capacity_gb')}")
        for reservation_id, record in payload["reservations"].items():
            if (not isinstance(reservation_id, str)
                    or not isinstance(record, dict)
                    or record.get("id") != reservation_id
                    or not isinstance(record.get("owner_sha256"), str)
                    or not isinstance(record.get("claims"), list)):
                raise ValueError("invalid reservation record")
            for claim in record["claims"]:
                if (not isinstance(claim, dict)
                        or not isinstance(claim.get("host"), str)):
                    raise ValueError("invalid reservation claim")
                normalized_claim = parse_host_resource(
                    f"{claim.get('host')}:{claim.get('gpu_slots')}:"
                    f"{claim.get('mem_gb')}:{claim.get('gpu_capacity')}:"
                    f"{claim.get('mem_capacity_gb')}")
                expected_capacity = {
                    "gpu_capacity": normalized_claim["gpu_capacity"],
                    "mem_capacity_gb": normalized_claim["mem_capacity_gb"],
                }
                if payload["capacities"].get(
                        normalized_claim["host"]) != expected_capacity:
                    raise ValueError(
                        "reservation claim disagrees with host capacity")
        return payload
    except FileNotFoundError:
        return _empty_reservation_ledger()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if strict:
            raise RuntimeError(f"malformed reservation ledger {path}: {exc}")
        return {"version": 2, "updated_at": 0.0,
                "capacities": {}, "reservations": {},
                "malformed": str(exc)}


def _ledger_holds(host: str) -> tuple[int, float, list[str], bool]:
    ledger = read_reservation_ledger()
    if ledger.get("malformed"):
        # There is no honest way to infer free capacity from unreadable
        # claims.  A huge hold stops admission and makes the fault visible.
        return (2 ** 31 - 1, float("inf"),
                [f"MALFORMED {reservation_ledger_path()}: "
                 f"{ledger['malformed']}"], True)
    gpu_slots = 0
    mem_gb = 0.0
    reasons = []
    for reservation_id, record in ledger["reservations"].items():
        for claim in record["claims"]:
            if claim.get("host") != host:
                continue
            try:
                gpu_slots += max(0, int(claim.get("gpu_slots", 0)))
                mem_gb += max(0.0, float(claim.get("mem_gb", 0.0)))
            except (TypeError, ValueError):
                return (2 ** 31 - 1, float("inf"),
                        [f"MALFORMED reservation {reservation_id}"], True)
            reasons.append(
                f"set={reservation_id}: {record.get('why') or 'unattributed'}")
    return gpu_slots, mem_gb, reasons, False


def _legacy_reserved_memory_gb(host: str) -> tuple[float, str]:
    f = qdir(RESERVED) / f"{host}.mem"
    try:
        text = f.read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return 0.0, ""
    if not text:
        return 0.0, ""
    head, _, rest = text.partition("\n")
    try:
        value = float(head.strip())
    except ValueError:
        # An unreadable external hold cannot establish free capacity.
        return float("inf"), f"MALFORMED {f}: {text}"
    if not math.isfinite(value):
        return float("inf"), f"MALFORMED {f}: {text}"
    return max(0.0, value), rest.strip()


def _legacy_reserved_gpu_slots(host: str) -> tuple[int, str]:
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
        return 2 ** 31 - 1, f"MALFORMED {f}: {text}"
    return max(0, n), rest.strip()


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
    legacy_gb, legacy_why = _legacy_reserved_memory_gb(host)
    _, set_gb, set_reasons, _ = _ledger_holds(host)
    reasons = ([legacy_why] if legacy_why else []) + set_reasons
    return legacy_gb + set_gb, "\n".join(reasons)


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
    legacy_slots, legacy_why = _legacy_reserved_gpu_slots(host)
    set_slots, _, set_reasons, _ = _ledger_holds(host)
    reasons = ([legacy_why] if legacy_why else []) + set_reasons
    return legacy_slots + set_slots, "\n".join(reasons)


class ReservationBusy(RuntimeError):
    """A reservation set cannot be admitted without exceeding capacity."""


class ReservationOwnershipError(RuntimeError):
    """The supplied owner capability does not control this reservation."""


_RESERVATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def parse_host_resource(spec: str) -> dict:
    """Parse HOST:GPU:MEM_GB:GPU_CAPACITY:MEM_CAPACITY_GB."""
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(
            "host resource must be "
            "HOST:GPU_SLOTS:MEM_GB:GPU_CAPACITY:MEM_CAPACITY_GB")
    host, gpu, mem, gpu_capacity, mem_capacity = parts
    if not host or not _RESERVATION_ID_RE.fullmatch(host):
        raise ValueError(f"invalid host in resource: {host!r}")
    try:
        claim = {
            "host": host,
            "gpu_slots": int(gpu),
            "mem_gb": float(mem),
            "gpu_capacity": int(gpu_capacity),
            "mem_capacity_gb": float(mem_capacity),
        }
    except ValueError as exc:
        raise ValueError(f"invalid numeric resource in {spec!r}") from exc
    if (claim["gpu_slots"] < 0 or claim["mem_gb"] < 0
            or not math.isfinite(claim["mem_gb"])
            or claim["gpu_capacity"] <= 0
            or claim["mem_capacity_gb"] <= 0
            or not math.isfinite(claim["mem_capacity_gb"])
            or claim["gpu_slots"] > claim["gpu_capacity"]
            or claim["mem_gb"] > claim["mem_capacity_gb"]):
        raise ValueError(f"resource outside declared capacity: {spec!r}")
    return claim


def _validate_reservation_claims(claims: list[dict]) -> list[dict]:
    if not claims:
        raise ValueError("at least one --host-resource is required")
    normalized = []
    seen = set()
    for raw in claims:
        if isinstance(raw, str):
            claim = parse_host_resource(raw)
        elif isinstance(raw, dict):
            claim = dict(raw)
            if not isinstance(claim.get("host"), str):
                raise ValueError("reservation host must be a string")
        else:
            raise ValueError("reservation claim must be a string or object")
        host = claim.get("host")
        if host in seen:
            raise ValueError(f"duplicate host in reservation set: {host}")
        seen.add(host)
        # Round-trip through the parser so direct API callers get the same
        # validation and JSON-normalized types as CLI callers.
        normalized.append(parse_host_resource(
            f"{host}:{claim.get('gpu_slots')}:{claim.get('mem_gb')}:"
            f"{claim.get('gpu_capacity')}:{claim.get('mem_capacity_gb')}"))
    return sorted(normalized, key=lambda claim: claim["host"])


def _owner_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read_owner_capability(owner_file: Path, reservation_id: str) -> str | None:
    owner = read_item(owner_file)
    if owner is None:
        return None
    token = owner.get("token")
    if owner.get("reservation_id") != reservation_id or not isinstance(token, str):
        raise ReservationOwnershipError(
            f"owner file {owner_file} does not name reservation {reservation_id}")
    return token


def _write_owner_capability(owner_file: Path, reservation_id: str,
                            token: str) -> None:
    owner_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reservation_id": reservation_id,
        "token": token,
        "created_at": time.time(),
    }
    staged = owner_file.with_name(
        f".{owner_file.name}.owner-{os.getpid()}-{time.time_ns()}")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(staged, owner_file)
    finally:
        staged.unlink(missing_ok=True)


def _claimed_by_host() -> tuple[dict[str, list[dict]], list[dict]]:
    by_host: dict[str, list[dict]] = collections.defaultdict(list)
    unknown = []
    for path in sorted(qdir(CLAIMED).glob("*.json")):
        item = read_item(path)
        if item is None:
            unknown.append({"id": path.stem, "unreadable": True})
            continue
        lease = read_item(lease_path(path.stem)) or {}
        host = lease.get("host") or item.get("claimed_by")
        if isinstance(host, str) and host:
            by_host[host].append(item)
        else:
            unknown.append(item)
    return dict(by_host), unknown


def _ledger_totals(ledger: dict, host: str) -> tuple[int, float]:
    gpu_slots = 0
    mem_gb = 0.0
    for record in ledger["reservations"].values():
        for claim in record["claims"]:
            if claim.get("host") != host:
                continue
            try:
                gpu_slots += max(0, int(claim.get("gpu_slots", 0)))
                mem_gb += max(0.0, float(claim.get("mem_gb", 0.0)))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"malformed reservation claim for host {host}") from exc
    return gpu_slots, mem_gb


def acquire_reservation_set(reservation_id: str, claims: list[dict],
                            owner_file: Path, why: str,
                            require_no_claimed: bool = False) -> dict:
    """Atomically acquire every host claim, or publish none of them."""
    ensure_layout()
    if not _RESERVATION_ID_RE.fullmatch(reservation_id):
        raise ValueError(f"invalid reservation id: {reservation_id!r}")
    normalized = _validate_reservation_claims(claims)
    owner_file = Path(owner_file)
    with admission_guard():
        ledger = read_reservation_ledger(strict=True)
        existing = ledger["reservations"].get(reservation_id)
        existing_token = _read_owner_capability(owner_file, reservation_id)
        if existing is not None:
            if (existing_token is None
                    or not secrets.compare_digest(
                        existing.get("owner_sha256", ""),
                        _owner_digest(existing_token))):
                raise ReservationOwnershipError(
                    f"reservation {reservation_id} is owned by another capability")
            if existing.get("claims") != normalized:
                raise ReservationOwnershipError(
                    f"reservation {reservation_id} already has different resources")
            return existing

        by_host, unknown_claims = _claimed_by_host()
        if unknown_claims:
            raise ReservationBusy(
                "claimed item with unreadable/unknown host prevents safe acquisition: "
                + ", ".join(str(item.get("id")) for item in unknown_claims))
        for claim in normalized:
            host = claim["host"]
            queue_items = by_host.get(host, [])
            if require_no_claimed and queue_items:
                ids = ", ".join(str(item.get("id")) for item in queue_items)
                raise ReservationBusy(f"{host} has claimed queue work: {ids}")

            legacy_gpu, _ = _legacy_reserved_gpu_slots(host)
            legacy_mem, _ = _legacy_reserved_memory_gb(host)
            ledger_gpu, ledger_mem = _ledger_totals(ledger, host)
            capacity = {
                "gpu_capacity": claim["gpu_capacity"],
                "mem_capacity_gb": claim["mem_capacity_gb"],
            }
            configured_capacity = ledger["capacities"].get(host)
            if (configured_capacity is not None
                    and configured_capacity != capacity):
                raise ValueError(
                    f"{host} capacity conflicts with ledger: requested "
                    f"{capacity}, configured {configured_capacity}")
            claimed_gpu = sum(1 for item in queue_items
                              if item.get("needs_gpu"))
            claimed_mem = sum(
                _positive_item_gb(item, "mem_gb")
                or _positive_item_gb(item, "observed_peak_gb") or 0.0
                for item in queue_items)
            total_gpu = (legacy_gpu + ledger_gpu + claimed_gpu
                         + claim["gpu_slots"])
            total_mem = (legacy_mem + ledger_mem + claimed_mem
                         + claim["mem_gb"])
            if total_gpu > claim["gpu_capacity"]:
                raise ReservationBusy(
                    f"{host} GPU capacity: need {total_gpu}, "
                    f"capacity {claim['gpu_capacity']}")
            if total_mem > claim["mem_capacity_gb"]:
                raise ReservationBusy(
                    f"{host} memory capacity: need {total_mem:.3f}GB, "
                    f"capacity {claim['mem_capacity_gb']:.3f}GB")
            ledger["capacities"].setdefault(host, capacity)

        token = existing_token or secrets.token_urlsafe(32)
        if existing_token is None:
            _write_owner_capability(owner_file, reservation_id, token)
        record = {
            "id": reservation_id,
            "claims": normalized,
            "why": why or "unattributed",
            "owner_sha256": _owner_digest(token),
            "acquired_at": time.time(),
        }
        ledger["reservations"][reservation_id] = record
        ledger["updated_at"] = time.time()
        try:
            write_json_atomic(reservation_ledger_path(), ledger)
        except Exception:
            if existing_token is None:
                owner_file.unlink(missing_ok=True)
            raise
        return record


def verify_reservation_set(reservation_id: str, owner_file: Path) -> dict:
    """Return a token-authenticated reservation record without its token."""
    with admission_guard():
        ledger = read_reservation_ledger(strict=True)
        record = ledger["reservations"].get(reservation_id)
        token = _read_owner_capability(Path(owner_file), reservation_id)
        if (record is None or token is None
                or not secrets.compare_digest(
                    record.get("owner_sha256", ""), _owner_digest(token))):
            raise ReservationOwnershipError(
                f"reservation {reservation_id} is absent or owned elsewhere")
        return record


def release_reservation_set(reservation_id: str, owner_file: Path) -> dict:
    """Atomically release a complete set, only with its owner capability."""
    owner_file = Path(owner_file)
    with admission_guard():
        ledger = read_reservation_ledger(strict=True)
        record = ledger["reservations"].get(reservation_id)
        token = _read_owner_capability(owner_file, reservation_id)
        if (record is None or token is None
                or not secrets.compare_digest(
                    record.get("owner_sha256", ""), _owner_digest(token))):
            raise ReservationOwnershipError(
                f"reservation {reservation_id} is absent or owned elsewhere")
        del ledger["reservations"][reservation_id]
        ledger["updated_at"] = time.time()
        write_json_atomic(reservation_ledger_path(), ledger)
        owner_file.unlink(missing_ok=True)
    return record


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


def placement_matches(item: dict, host: str, tags: set[str], has_gpu: bool,
                      only_declared: bool = False) -> bool:
    """Can this worker ever run this item, independent of time and gates?"""
    hosts = item.get("hosts")
    required = set(item.get("requires") or ())
    if only_declared and not hosts and not required:
        return False
    if hosts and host not in hosts:
        return False
    if not required.issubset(tags):
        return False
    if item.get("needs_gpu") and not has_gpu:
        return False
    return True


def is_runnable(item: dict, host: str, tags: set[str], has_gpu: bool,
                only_declared: bool = False) -> bool:
    """Can *this* worker run *this* item right now?

    Slot availability is checked separately -- this answers eligibility
    only, so that ``status`` can report "runnable but no free slot" as
    distinct from "not eligible here".

    ``only_declared`` inverts the default for a worker that is not
    interchangeable with the others. Normally an item that names no host and
    no tags is eligible everywhere, which is right for a fleet of identical
    boxes and wrong the moment one differs: a job written for the Sparks --
    their venv, their repo checkout, their GPU -- would be claimed by an
    unlike box and fail there, and the failure would look like the job's
    fault rather than the placement's. Under ``only_declared`` such a worker
    takes only items that ask for it by name or by tag, so adding a box
    cannot break work that predates it.
    """
    if not placement_matches(item, host, tags, has_gpu, only_declared):
        return False
    if time.time() < float(item.get("not_before") or 0):
        return False
    return receipts_present(item.get("after") or [])


def can_reserve_admission(item: dict, host: str, tags: set[str],
                          has_gpu: bool,
                          only_declared: bool = False) -> bool:
    """May an aged item pin admission while its eviction cooldown expires?

    Receipt dependencies remain real gates: an item waiting on unrelated
    work must not freeze a box.  The fixed post-eviction cooldown is
    different.  It is scheduler-imposed, and hiding a protected item during
    that cooldown lets a stream of newcomers consume its next attempt before
    the scheduler ever sees it.  Once protection is earned, keep the item in
    admission selection while ignoring only ``not_before``; the worker still
    refuses to claim it until the timestamp passes.
    """
    return (item_is_protected(item)
            and placement_matches(item, host, tags, has_gpu, only_declared)
            and receipts_present(item.get("after") or []))


def sort_key(item: dict) -> tuple:
    # Higher priority first, then oldest first, so a burst of equal-priority
    # items drains in enqueue order instead of by filename.
    return (-int(item.get("priority", 50)), float(item.get("enqueued_at", 0)))


def ready_candidates(host: str, tags: set, has_gpu: bool,
                     only_declared: bool = False) -> list[dict]:
    """Runnable items plus aged cooldown waiters that reserve admission."""
    if deployment_hold_path().exists():
        return []
    candidates = []
    for path in qdir(READY).glob("*.json"):
        item = read_item(path)
        if item and (is_runnable(item, host, tags, has_gpu, only_declared)
                     or can_reserve_admission(
                         item, host, tags, has_gpu, only_declared)):
            candidates.append(item)
    candidates.sort(key=sort_key)
    return candidates


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
    try:
        beat = float(lease.get("beat", 0))
    except (TypeError, ValueError):
        # A malformed heartbeat cannot establish a live lease. Fail toward
        # recovery instead of crashing every worker's reap loop forever.
        return True
    return (time.time() - beat) > LEASE_TIMEOUT_S


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


def _archive_orphan_lease(item_id: str) -> Path | None:
    """Atomically remove an orphan lease from ``claimed/`` for forensics.

    The first archive keeps the same basename operators already use when
    manually reaping a lease. A repeated incarnation of the same item gets a
    unique suffix rather than overwriting the earlier evidence.
    """
    source = lease_path(item_id)
    destination = qdir(REAPED) / f"{item_id}.lease"
    if destination.exists():
        destination = qdir(REAPED) / (
            f"{item_id}.{time.time_ns()}.{os.getpid()}.lease")
    try:
        os.rename(source, destination)
    except FileNotFoundError:
        return None
    return destination


def _reap_orphan_leases(verbose: bool = False) -> int:
    """Reap lease sidecars whose claim is gone and cannot be live.

    A terminal item makes the lease obsolete immediately. With no item in
    ``claimed/`` and no terminal record, retain a fresh lease for one TTL in
    case an in-flight NFS transition is still settling; once its heartbeat is
    stale it is orphaned by definition and is archived as well.
    """
    n = 0
    for lease in sorted(qdir(CLAIMED).glob("*.lease")):
        item_id = lease.stem
        if item_path(CLAIMED, item_id).exists():
            continue
        terminal = next(
            (state for state in (DONE, FAILED)
             if item_path(state, item_id).exists()),
            None,
        )
        if terminal is None and not lease_is_stale(item_id):
            continue
        archived = _archive_orphan_lease(item_id)
        if archived is None:
            continue
        n += 1
        if verbose:
            reason = f"terminal in {terminal}/" if terminal else "heartbeat stale"
            print(f"[reap] {item_id}: orphan lease ({reason}) -> {archived}")
    return n


def reap(verbose: bool = False) -> int:
    """Requeue stale claims and remove leases that no longer own a claim.

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
    return n + _reap_orphan_leases(verbose=verbose)


def _try_claim_unlocked(item_id: str, host: str) -> dict | None:
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


def try_claim(item_id: str, host: str) -> dict | None:
    """Conservatively take a low-level claim at the linearization point.

    The worker uses :func:`try_admit_item`, which knows exact slot and memory
    capacities.  This compatibility helper has no capacity arguments, so it
    must fail closed when any relevant external hold exists rather than let a
    caller bypass reservation admission.
    """
    try:
        with admission_guard():
            if deployment_hold_path().exists():
                return None
            item = read_item(item_path(READY, item_id))
            if item is None:
                return None
            held_mem, _ = reserved_memory_gb(host)
            held_gpu, _ = reserved_gpu_slots(host)
            if held_mem > 0 or (item.get("needs_gpu") and held_gpu > 0):
                return None
            return _try_claim_unlocked(item_id, host)
    except AdmissionLockTimeout:
        return None


def try_admit_item(item: dict, host: str, tags: set[str], has_gpu: bool,
                   only_declared: bool, gpu_slots: int, cpu_slots: int,
                   governor, jobs: list) -> dict | None:
    """Recheck capacity and claim at the reservation linearization point."""
    with admission_guard(timeout_s=1.0):
        if (deployment_hold_path().exists()
                or not is_runnable(
                    item, host, tags, has_gpu, only_declared)):
            return None
        mine = claimed_on_host(host)
        gpu_busy = sum(1 for claimed_item in mine
                       if claimed_item.get("needs_gpu"))
        cpu_busy = len(mine) - gpu_busy
        held_mem, _ = reserved_memory_gb(host)
        held_gpu, _ = reserved_gpu_slots(host)
        gpu_capacity = max(0, gpu_slots - held_gpu)
        governor.set_external_hold(held_mem)
        if item.get("needs_gpu"):
            if gpu_busy >= gpu_capacity:
                return None
        elif cpu_busy >= cpu_slots:
            return None
        if not governor.may_admit(item, jobs):
            return None
        return _try_claim_unlocked(item["id"], host)


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


def mem_total_gb() -> float:
    """Physical memory governed by this worker, in GB."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")


def _positive_item_gb(item: dict, key: str) -> float | None:
    try:
        value = float(item.get(key) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def item_eviction_count(item: dict) -> int:
    """Policy evictions earned by an item; never a fit verdict."""
    try:
        return max(0, int(item.get("evictions", 0)))
    except (TypeError, ValueError):
        return 0


def item_is_protected(item: dict) -> bool:
    """Whether scheduler aging guarantees this item's next attempt."""
    return item_eviction_count(item) >= EVICTIONS_BEFORE_PROTECTION


def _median_consecutive_slope(samples) -> float | None:
    """Return the median value-growth slope, or ``None`` without a trend.

    Consecutive-pair slopes are used instead of an endpoint difference so
    every sample in the window participates.  Their median was chosen over a
    least-squares fit because a single interior outlier contributes one steep
    rise and one steep fall, while an endpoint outlier contributes only one
    bad interval; neither can dominate the median.  Requiring at least three
    valid intervals also means one sample pair can never constitute a trend.
    """
    ordered = list(samples)
    slopes = []
    for (t0, value0), (t1, value1) in zip(ordered, ordered[1:]):
        dt = t1 - t0
        if dt > 0:
            slopes.append((value1 - value0) / dt)
    if len(slopes) < MIN_TREND_INTERVALS:
        return None
    return float(statistics.median(slopes))


def _process_tree_rss_gb(root_pid: int) -> float | None:
    """Resident bytes for a direct child and its descendants.

    An uncapped job is launched as a process group, but Linux exposes RSS per
    process rather than per group.  Walking ``children`` captures the shell
    and the workload it launched without charging unrelated processes to the
    victim.  Races with process exit are expected and simply omit that sample.
    """
    pending = [root_pid]
    seen = set()
    resident_pages = 0
    observed = False
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        proc_root = Path("/proc") / str(pid)
        try:
            children = (proc_root / "task" / str(pid) / "children").read_text()
            pending.extend(int(child) for child in children.split())
        except (OSError, ValueError):
            pass
        try:
            fields = (proc_root / "statm").read_text().split()
            resident_pages += int(fields[1])
            observed = True
        except (IndexError, OSError, ValueError):
            pass
    if not observed:
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 ** 3)


def unit_name(item_id: str) -> str:
    """systemd unit name for a job. Only [A-Za-z0-9:_.\\-] survive."""
    safe = "".join(c if (c.isalnum() or c in ":_.-") else "-" for c in item_id)
    return f"pqjob-{safe}"


class Job:
    """One claimed item, run either as a child process group or, when it
    declares ``mem_gb``, inside a transient systemd scope with a hard cgroup
    memory cap.

    The cap is the important half. A userspace monitor that watches memory and
    kills something is the wrong shape -- it is reactive, it can only evict
    what it started, and on unified memory it is the recorded Ray failure mode
    (a monitor killing healthy ranks). ``MemoryMax`` inverts that: the job
    that exceeds its own declared budget is killed by its own cgroup, and
    nothing else on the box is a candidate. Measured here: a 300 M cap against
    a runaway allocator gave ``memory.events oom-kill 2`` and exit 9, with no
    other process disturbed.

    ``--scope`` was tried first, because it keeps the job as a direct child
    and so keeps ``killpg`` working; it hung under the cap rather than
    killing. The transient service form enforces correctly, so eviction of a
    capped job goes through ``systemctl --user stop`` instead.
    """

    def __init__(self, item: dict, host: str):
        self.item = item
        self.id = item["id"]
        self.host = host
        self.proc: subprocess.Popen | None = None
        self.unit: str | None = None
        self.oomed = False
        self.started = time.time()
        self.avail_at_start = mem_available_gb()
        self.evicted = False
        self.eviction_attribution: dict | None = None
        self.footprint_samples: collections.deque = collections.deque()
        self.footprint_source: str | None = None
        self._memory_current_path: Path | None = None
        self._next_cgroup_lookup_at = 0.0
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
        return (bool(self.item.get("evictable", True))
                and not item_is_protected(self.item))

    def sunk_runtime_s(self, now: float | None = None) -> float:
        """Measured runtime discarded across this and earlier attempts."""
        try:
            earlier = max(0.0, float(self.item.get("evicted_runtime_s") or 0.0))
        except (TypeError, ValueError):
            earlier = 0.0
        current = max(0.0, (time.time() if now is None else now) - self.started)
        return earlier + current

    def _cgroup_footprint_gb(self) -> float | None:
        """Read this transient service's ``memory.current`` when available."""
        path = self._memory_current_path
        if path is None:
            now = time.monotonic()
            if now < self._next_cgroup_lookup_at:
                return None
            self._next_cgroup_lookup_at = now + CGROUP_LOOKUP_RETRY_S
            try:
                out = subprocess.run(
                    ["systemctl", "--user", "show", self.unit,
                     "-p", "ControlGroup", "--value"],
                    capture_output=True, text=True, timeout=5)
                control_group = (out.stdout or "").strip()
                relative = Path(control_group.lstrip("/"))
                if (out.returncode == 0 and control_group not in ("", "/")
                        and control_group.startswith("/")
                        and ".." not in relative.parts):
                    path = Path("/sys/fs/cgroup") / relative / "memory.current"
                    self._memory_current_path = path
            except (OSError, subprocess.SubprocessError):
                return None
        if path is None:
            return None
        try:
            return int(path.read_text().strip()) / (1024.0 ** 3)
        except (OSError, ValueError):
            # A unit may disappear between ``systemctl show`` and the read.
            self._memory_current_path = None
            return None

    def _read_footprint_gb(self) -> tuple[float, str] | None:
        if self.unit is not None:
            value = self._cgroup_footprint_gb()
            if value is None:
                return None
            return value, f"cgroup:{self._memory_current_path}"
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return None
        value = _process_tree_rss_gb(proc.pid)
        if value is None:
            return None
        return value, f"rss_tree:{proc.pid}"

    def sample_footprint(self, now: float | None = None) -> float | None:
        """Sample this job's own resident footprint for later attribution."""
        reading = self._read_footprint_gb()
        if reading is None:
            return None
        value, source = reading
        if self.footprint_source is not None and source != self.footprint_source:
            # A source change is a different accounting boundary; joining the
            # values would manufacture a slope at the seam.
            self.footprint_samples.clear()
        self.footprint_source = source
        sampled_at = time.time() if now is None else now
        self.footprint_samples.append((sampled_at, value))
        while (self.footprint_samples
               and sampled_at - self.footprint_samples[0][0]
               > MEM_RATE_WINDOW_S):
            self.footprint_samples.popleft()
        return value

    def classify_eviction(self, box_drop_rate_gb_s: float) -> dict:
        """Attribute pressure only when this victim measurably drove it.

        A growing victim is not automatically the cause: unrelated work can
        consume most of the box while the selected job grows slowly.  A
        ``victim_growth`` classification therefore requires a sustained own-
        footprint slope that explains at least half of the sustained box-wide
        MemAvailable fall.  Flat, recovering, or minority growth is recorded
        as collateral.  Missing samples are recorded as unknown, never turned
        into a factual fit verdict.
        """
        growth = _median_consecutive_slope(self.footprint_samples)
        sample_count = len(self.footprint_samples)
        window_s = (self.footprint_samples[-1][0]
                    - self.footprint_samples[0][0]) if sample_count >= 2 else 0.0
        record = {
            "classification": "unknown",
            "box_drop_gb_s": round(float(box_drop_rate_gb_s), 6),
            "victim_growth_gb_s": (None if growth is None
                                     else round(float(growth), 6)),
            "victim_growth_share": None,
            "victim_footprint_start_gb": (
                round(float(self.footprint_samples[0][1]), 3)
                if sample_count else None),
            "victim_footprint_gb": (
                round(float(self.footprint_samples[-1][1]), 3)
                if sample_count else None),
            "sample_count": sample_count,
            "window_s": round(window_s, 3),
            "source": self.footprint_source,
            "measured_at": time.time(),
        }
        if growth is None:
            record["reason"] = "insufficient_footprint_samples"
            return record
        if box_drop_rate_gb_s <= 0:
            record.update(classification="collateral",
                          reason="box_not_falling")
            return record
        share = max(0.0, growth) / box_drop_rate_gb_s
        record["victim_growth_share"] = round(share, 6)
        if growth > 0 and share >= ATTRIBUTION_MIN_FRACTION:
            record.update(classification="victim_growth",
                          reason="victim_growth_explains_box_drop")
        else:
            record.update(classification="collateral",
                          reason="victim_not_growing_enough_to_explain_drop")
        return record

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
            cwd = item.get("cwd") or str(Path.home())
            if self.declared_gb > 0 and capping_supported():
                # A declared budget buys real enforcement: exceed it and your
                # own cgroup kills you, rather than the kernel picking a
                # victim from the whole box.
                self.unit = unit_name(self.id)
                subprocess.run(["systemctl", "--user", "reset-failed",
                                self.unit], capture_output=True)
                cmd = [
                    "systemd-run", "--user", "--quiet", "--wait",
                    f"--unit={self.unit}",
                    "-p", f"MemoryMax={self.declared_gb:.0f}G",
                    "-p", "MemorySwapMax=0",
                    "-p", "MemoryAccounting=yes",
                    "-p", f"WorkingDirectory={cwd}",
                    "-p", f"StandardOutput=append:{log}",
                    "-p", f"StandardError=append:{log}",
                ]
                for k, v in env.items():
                    if k in ("TMPDIR", "PQ_ITEM_ID", "PQ_WORKER_HOST", "PATH",
                             "HOME", "PYTHONPATH"):
                        cmd += ["--setenv", f"{k}={v}"]
                cmd += ["--", "/bin/bash", "-lc", item["cmd"]]
                fh.write(f"[pqwork] capped at {self.declared_gb:.0f}G "
                         f"in unit {self.unit}\n")
                fh.flush()
                self.proc = subprocess.Popen(cmd, env=env,
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.STDOUT,
                                             start_new_session=True)
                self.proc.wait()
                rc, self.oomed = unit_result(self.unit)
                if self.oomed:
                    fh.write(f"[pqwork] killed by its own cgroup: exceeded the "
                             f"declared {self.declared_gb:.0f}G budget\n")
                    fh.flush()
                subprocess.run(["systemctl", "--user", "reset-failed",
                                self.unit], capture_output=True)
            else:
                # start_new_session so the whole job -- and anything it spawns
                # -- is one process group we can signal as a unit. Killing only
                # the shell would leave the actual worker holding the memory.
                if self.declared_gb > 0:
                    fh.write("[pqwork] UNCAPPED: this box cannot start a "
                             "transient user unit, so the declared "
                             f"{self.declared_gb:.0f}G budget is a hint only\n")
                    fh.flush()
                self.proc = subprocess.Popen(
                    ["bash", "-lc", item["cmd"]],
                    cwd=cwd, env=env,
                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                )
                rc = self.proc.wait()
            elapsed = time.time() - self.started
            if self.evicted:
                classification = (self.eviction_attribution or {}).get(
                    "classification", "unknown")
                fh.write(f"\n[pqwork] evicted to avoid OOM after {elapsed:.0f}s "
                         f"(attribution={classification})\n")

        if self.evicted:
            self._finish_eviction()
            return

        if self.oomed and not receipt_satisfied(item.get("receipt") or ""):
            # Retrying is pointless: the same command under the same budget
            # dies the same way. Say what the fix is instead of burning two
            # more attempts on it.
            move_item(self.id, CLAIMED, FAILED, {
                "outcome": f"budget_exceeded (declared {self.declared_gb:.0f}GB"
                           f"; re-enqueue with a larger --mem-gb)",
                "exit_code": rc, "elapsed_s": round(elapsed, 1),
                "finished_at": time.time()})
            self.state = "budget exceeded"
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

    def _finish_eviction(self) -> bool:
        """Requeue with more scheduler protection, never a fit verdict."""
        if not self.evicted:
            return False
        item = self.item
        attribution = self.eviction_attribution or {
            "classification": "unknown",
            "reason": "eviction_without_attribution_measurement",
            "box_drop_gb_s": None,
            "victim_growth_gb_s": None,
            "victim_growth_share": None,
            "victim_footprint_start_gb": None,
            "victim_footprint_gb": None,
            "sample_count": len(self.footprint_samples),
            "window_s": 0.0,
            "source": self.footprint_source,
            "measured_at": time.time(),
        }
        now = time.time()
        evictions = item_eviction_count(item) + 1
        observed = [value for _, value in self.footprint_samples]
        prior_peak = _positive_item_gb(item, "observed_peak_gb")
        if prior_peak is not None:
            observed.append(prior_peak)
        common = {
            "eviction_attribution": attribution,
            "last_evicted_at": now,
            "attempts": max(0, int(item.get("attempts", 1)) - 1),
            "evictions": evictions,
            "evicted_runtime_s": round(self.sunk_runtime_s(now), 3),
            "protection_earned": (
                evictions >= EVICTIONS_BEFORE_PROTECTION),
        }
        if observed:
            common["observed_peak_gb"] = round(max(observed), 3)

        classification = attribution.get("classification")
        if classification == "victim_growth":
            field = "attributed_evictions"
            outcome = "evicted_for_own_memory_growth"
            state = "evicted for own growth"
        elif classification == "collateral":
            field = "collateral_evictions"
            outcome = "evicted_as_collateral"
            state = "evicted as collateral"
        else:
            field = "unattributed_evictions"
            outcome = "evicted_without_attribution"
            state = "evicted without attribution"
        common[field] = int(item.get(field, 0)) + 1
        moved = move_item(
            self.id, CLAIMED, READY,
            {**common, "outcome": outcome,
             "not_before": now + EVICT_BACKOFF_S})
        if common["protection_earned"]:
            state += "; next attempt protected"
        self.state = f"{state} (cooldown {EVICT_BACKOFF_S:.0f}s)"
        return moved

    def start(self) -> None:
        self.thread.start()

    def alive(self) -> bool:
        return self.thread.is_alive()

    def evict(self, attribution: dict | None = None) -> None:
        """Stop this job so its memory comes back, and requeue it.

        SIGTERM to the process group first so a job with a cleanup handler
        can flush a partial receipt, then SIGKILL. Signalling the group
        rather than the shell matters: the shell is not the process holding
        the tens of gigabytes.
        """
        self.eviction_attribution = attribution
        self.evicted = True
        if self.unit:
            # A capped job lives in its own unit, not in our process group.
            subprocess.run(["systemctl", "--user", "stop", self.unit],
                           capture_output=True)
            return
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


_CAP_SUPPORTED: bool | None = None


def capping_supported() -> bool:
    """Whether this box can actually start a capped transient user unit.

    Probed once, at worker startup, rather than assumed. Capping needs cgroup
    delegation to the user manager, which is a property of the box. Without
    the probe a box that lacks it would fail every capped job the instant it
    started, and receipt-gating would faithfully requeue each one forever --
    a capability gap wearing the costume of a flaky job.
    """
    global _CAP_SUPPORTED
    if _CAP_SUPPORTED is None:
        try:
            out = subprocess.run(
                ["systemd-run", "--user", "--quiet", "--wait",
                 "-p", "MemoryMax=64M", "-p", "MemoryAccounting=yes",
                 "--", "/bin/true"],
                capture_output=True, timeout=60)
            _CAP_SUPPORTED = out.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _CAP_SUPPORTED = False
    return _CAP_SUPPORTED


def unit_result(unit: str) -> tuple[int, bool]:
    """``(exit_code, was_oom_killed)`` for a finished transient unit.

    ``systemd-run --wait`` returns its own success, not the service's, so the
    status has to be read back. Neither value decides whether the work
    succeeded -- that is the receipt's job, and a unit's reported Result is
    not evidence of anything but how the process ended. The OOM flag is used
    for one narrow purpose: to say why, and to stop retrying a job whose
    declared budget is simply too small.
    """
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ExecMainStatus",
             "-p", "Result", "--value"],
            capture_output=True, text=True, timeout=15)
        lines = [ln.strip() for ln in (out.stdout or "").splitlines()]
        rc = next((int(ln) for ln in lines if ln.lstrip("-").isdigit()), -1)
        return rc, any(ln == "oom-kill" for ln in lines)
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1, False


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
                 admission: str = "optimistic", total_gb: float | None = None):
        self.reserve_gb = reserve_gb
        self.horizon_s = horizon_s
        self.settle_s = settle_s
        self.admission = admission
        self.total_gb = mem_total_gb() if total_gb is None else total_gb
        self.external_hold_gb = 0.0
        self.samples: collections.deque = collections.deque()

    @property
    def capacity_gb(self) -> float:
        """Absolute job capacity after the worker's operating reserve."""
        return max(0.0, self.total_gb - self.reserve_gb)

    def sample(self) -> float:
        now, avail = time.time(), mem_available_gb()
        self.samples.append((now, avail))
        while self.samples and now - self.samples[0][0] > MEM_RATE_WINDOW_S:
            self.samples.popleft()
        return avail

    def drop_rate_gb_s(self) -> float:
        """Sustained GB/s fall; negative means sustained recovery.

        This is the negative median consecutive-pair slope.  See
        :func:`_median_consecutive_slope` for the outlier and minimum-interval
        contract.
        """
        slope = _median_consecutive_slope(self.samples)
        return -slope if slope is not None else 0.0

    def projection_horizon_s(self) -> float:
        horizon = self.horizon_s
        if swap_free_gb() <= 0.5:
            horizon *= 2.0
        return horizon

    def projected_available_gb(self) -> float:
        if not self.samples:
            return float("inf")
        return (self.samples[-1][1]
                - self.drop_rate_gb_s() * self.projection_horizon_s())

    def set_external_hold(self, gb: float) -> None:
        self.external_hold_gb = gb

    def reservation_gb(self, item: dict) -> float | None:
        """Measured/declarative reservation, or ``None`` for exclusive run."""
        return (_positive_item_gb(item, "mem_gb")
                or _positive_item_gb(item, "observed_peak_gb"))

    def measured_impossibility(self, item: dict) -> dict | None:
        """Evidence that this item exceeds absolute capacity on this box."""
        if item.get("receipt") and receipt_satisfied(item["receipt"]):
            return None
        for source, key in (("declared", "mem_gb"),
                            ("observed", "observed_peak_gb")):
            footprint = _positive_item_gb(item, key)
            if footprint is not None and footprint > self.capacity_gb:
                return {
                    "source": source,
                    "footprint_gb": round(footprint, 3),
                    "box_total_gb": round(self.total_gb, 3),
                    "reserve_gb": round(self.reserve_gb, 3),
                    "capacity_gb": round(self.capacity_gb, 3),
                    "measured_at": time.time(),
                }
        return None

    def admission_candidates(self, candidates: list[dict]) -> list[dict]:
        """Let the most-aged protected item hold admission until it can start.

        Once an item has earned protection, admitting another arrival ahead of
        it would recreate starvation one layer earlier.  Returning only the
        most-aged reservation drains existing work without letting a stream of
        newcomers consume the memory or slot it is waiting for.
        """
        protected = [item for item in candidates if item_is_protected(item)]
        if not protected:
            return candidates
        reserved = min(
            protected,
            key=lambda item: (-item_eviction_count(item),
                              -int(item.get("priority", 50)),
                              float(item.get("enqueued_at", 0))))
        return [reserved]

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
        if item_is_protected(item):
            reservation = self.reservation_gb(item)
            if reservation is None:
                # No declared or observed size: drain the box and give the
                # protected item an exclusive attempt rather than inventing a
                # number that MemAvailable cannot support.
                return not jobs and headroom > 0
            return headroom >= reservation
        if self.admission == "strict":
            want = _positive_item_gb(item, "mem_gb") or 0.0
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
        return self.projected_available_gb() < self.reserve_gb


def fail_measured_impossibility(item: dict, governor: MemoryGovernor) -> bool:
    """Fail a ready item only from an explicit absolute-capacity measurement."""
    measurement = governor.measured_impossibility(item)
    if measurement is None:
        return False
    return move_item(
        item["id"], READY, FAILED,
        {"outcome": "measured_footprint_exceeds_box_capacity",
         "fit_measurement": measurement,
         "finished_at": time.time()})


def choose_victim(jobs: list):
    """The lowest-priority job with the least measured sunk runtime.

    ``started`` is not a sunk-work measurement: a restarted victim is newest
    by construction, so newest-first selects the output of its own eviction
    again.  ``sunk_runtime_s`` carries discarded runtime across attempts and
    makes restart history protective.  After the bounded aging threshold the
    job is non-evictable altogether. A user-pinned item remains non-evictable
    independently of scheduler protection.
    """
    victims = [j for j in jobs if j.evictable and j.alive()]
    if not victims:
        return None
    now = time.time()
    return min(victims,
               key=lambda job: (int(job.item.get("priority", 50)),
                                job.sunk_runtime_s(now), job.id))


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
                settle_s: float, admission: str = "optimistic",
                only_declared: bool = False) -> int:
    wait_for_queue(once)
    ensure_layout()
    worker_started_at = time.time()
    worker_source_sha256 = loaded_source_sha256()
    write_worker_heartbeat(host, worker_source_sha256, worker_started_at)
    has_gpu = gpu_slots > 0
    gov = MemoryGovernor(reserve_gb, horizon_s, settle_s, admission)
    jobs: list = []
    print(f"[pqwork] worker up on {host} tags={sorted(tags) or '-'} "
          f"gpu_slots={gpu_slots} cpu_slots={cpu_slots} "
          f"mem_reserve={reserve_gb:.0f}GB horizon={horizon_s:.0f}s "
          f"admission={admission} swap_free={swap_free_gb():.0f}GB "
          f"caps={'on' if capping_supported() else 'UNAVAILABLE'} "
          f"{'only-declared ' if only_declared else ''}"
          f"source_sha256={worker_source_sha256} queue={QUEUE_ROOT}",
          flush=True)
    last_held = None
    last_mem_hold = 0.0
    last_reservation = None
    last_deploy_hold = False
    last_beat = 0.0
    while True:
        if not queue_reachable():
            # Losing the share mid-run is not fatal: running jobs keep their
            # own local state, and bookkeeping resumes when it returns.
            wait_for_queue(once)
            ensure_layout()
        avail = gov.sample()
        jobs = [j for j in jobs if j.alive()]
        sampled_at = gov.samples[-1][0]
        for job in jobs:
            job.sample_footprint(sampled_at)

        # Memory policing comes before admission: give the box back its
        # headroom before considering whether to take on more.
        if gov.must_evict(jobs):
            victim = choose_victim(jobs)
            if victim is not None:
                rate = gov.drop_rate_gb_s()
                attribution = victim.classify_eviction(rate)
                growth = attribution["victim_growth_gb_s"]
                growth_text = ("unmeasured" if growth is None
                               else f"{growth:.2f}GB/s")
                print(f"[pqwork] EVICT {victim.id}: MemAvailable {avail:.1f}GB "
                      f"falling {rate:.2f}GB/s -> projected "
                      f"{gov.projected_available_gb():.1f}GB below the "
                      f"{reserve_gb:.0f}GB reserve; victim growth "
                      f"{growth_text} ({attribution['classification']})",
                      flush=True)
                victim.evict(attribution)

        now = time.time()
        if now - last_beat >= HEARTBEAT_S:
            for j in jobs:
                write_lease(j.id, host, j.proc.pid if j.proc else os.getpid())
            write_worker_heartbeat(
                host, worker_source_sha256, worker_started_at)
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

        deploy_held = deployment_hold_path().exists()
        if deploy_held != last_deploy_hold:
            if deploy_held:
                print("[pqwork] deployment hold active; admitting no new claims",
                      flush=True)
            else:
                print("[pqwork] deployment hold released", flush=True)
            last_deploy_hold = deploy_held

        candidates = ready_candidates(host, tags, has_gpu, only_declared)

        possible = []
        for item in candidates:
            impossibility = gov.measured_impossibility(item)
            if impossibility is None:
                possible.append(item)
                continue
            if fail_measured_impossibility(item, gov):
                print(f"[pqwork] FAIL {item['id']}: "
                      f"{impossibility['source']} footprint "
                      f"{impossibility['footprint_gb']:.1f}GB exceeds "
                      f"box capacity {impossibility['capacity_gb']:.1f}GB",
                      flush=True)
        candidates = gov.admission_candidates(possible)
        reservation = (candidates[0] if len(candidates) == 1
                       and item_is_protected(candidates[0]) else None)
        reservation_id = reservation["id"] if reservation else None
        if reservation_id != last_reservation:
            if reservation is not None:
                want = gov.reservation_gb(reservation)
                requirement = (f"{want:.1f}GB" if want is not None
                               else "an exclusive slot")
                print(f"[pqwork] protecting {reservation_id} after "
                      f"{item_eviction_count(reservation)} policy evictions; "
                      f"reserving {requirement}", flush=True)
            last_reservation = reservation_id

        started_any = False
        for item in candidates:
            # ``ready_candidates`` deliberately includes a protected item
            # during its scheduler-imposed cooldown so it can reserve the
            # next admission.  Reservation is not execution: recheck every
            # real gate immediately before claiming it.
            if not is_runnable(item, host, tags, has_gpu, only_declared):
                continue
            try:
                # Reservation acquisition uses the same guard inside this
                # helper. Exactly one side of the race can publish.
                claimed = try_admit_item(
                    item, host, tags, has_gpu, only_declared,
                    gpu_slots, cpu_slots, gov, jobs)
            except AdmissionLockTimeout as exc:
                print(f"[pqwork] admission guard unavailable: {exc}",
                      flush=True)
                continue
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
        "attributed_evictions": 0,
        "collateral_evictions": 0,
        "unattributed_evictions": 0,
        "evicted_runtime_s": 0.0,
        "protection_earned": False,
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
                if item_is_protected(i):
                    bits.append("protected")
            elif state == READY:
                pending = [p for p in (i.get("after") or []) if not receipt_satisfied(p)]
                if pending:
                    bits.append(f"blocked on {len(pending)} receipt(s)")
                if item_is_protected(i):
                    bits.append("protected reservation")
                cool = float(i.get("not_before") or 0) - time.time()
                if cool > 0:
                    cause = str(i.get("outcome") or "eviction").replace("_", " ")
                    bits.append(f"cooling {int(cool)}s after {cause}")
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
                       a.mem_settle_s, a.admission, a.only_declared)


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
                                           "evictions": 0,
                                           "attributed_evictions": 0,
                                           "collateral_evictions": 0,
                                           "unattributed_evictions": 0,
                                           "evicted_runtime_s": 0.0,
                                           "protection_earned": False,
                                           "not_before": 0})
            print(f"requeued {a.id} (was {state})")
            return 0
    print(f"error: no failed/done item '{a.id}'", file=sys.stderr)
    return 1


def cmd_reserve(a: argparse.Namespace) -> int:
    ensure_layout()
    f = qdir(RESERVED) / f"{a.host}.gpu"
    try:
        with admission_guard():
            if a.release:
                existed = f.exists()
                f.unlink(missing_ok=True)
                (qdir(RESERVED) / f"{a.host}.mem").unlink(missing_ok=True)
            else:
                f.write_text(f"{a.slots}\n{a.why or 'unattributed'}\n")
                if a.mem_gb:
                    (qdir(RESERVED) / f"{a.host}.mem").write_text(
                        f"{a.mem_gb}\n{a.why or 'unattributed'}\n")
    except AdmissionLockTimeout as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 70 if isinstance(exc, AdmissionLockStale) else 75
    if a.release:
        print(f"released GPU reservation on {a.host}" if existed
              else f"no GPU reservation held on {a.host}")
        return 0
    msg = f"reserved {a.slots} GPU slot(s) on {a.host}"
    if a.mem_gb:
        msg += f" and {a.mem_gb:.0f}GB"
    print(f"{msg}: {a.why or 'unattributed'}")
    return 0


def _reservation_error(exc: Exception) -> int:
    if isinstance(exc, AdmissionLockStale):
        print(f"fatal admission-lock error: {exc}", file=sys.stderr)
        return 70
    if isinstance(exc, (ReservationBusy, AdmissionLockTimeout)):
        print(f"busy: {exc}", file=sys.stderr)
        return 75
    if isinstance(exc, ReservationOwnershipError):
        print(f"ownership error: {exc}", file=sys.stderr)
        return 77
    print(f"error: {exc}", file=sys.stderr)
    return 2


def cmd_reserve_set_acquire(a: argparse.Namespace) -> int:
    try:
        record = acquire_reservation_set(
            a.id, a.host_resources, Path(a.owner_file), a.why,
            require_no_claimed=a.require_no_claimed)
    except (ValueError, RuntimeError) as exc:
        return _reservation_error(exc)
    print(json.dumps({
        "id": record["id"],
        "claims": record["claims"],
        "why": record["why"],
        "acquired_at": record["acquired_at"],
    }, sort_keys=True))
    return 0


def cmd_reserve_set_verify(a: argparse.Namespace) -> int:
    try:
        record = verify_reservation_set(a.id, Path(a.owner_file))
    except RuntimeError as exc:
        return _reservation_error(exc)
    print(json.dumps({
        "id": record["id"],
        "claims": record["claims"],
        "why": record["why"],
        "acquired_at": record["acquired_at"],
    }, sort_keys=True))
    return 0


def cmd_reserve_set_release(a: argparse.Namespace) -> int:
    try:
        record = release_reservation_set(a.id, Path(a.owner_file))
    except RuntimeError as exc:
        return _reservation_error(exc)
    print(json.dumps({"released": record["id"],
                      "claims": record["claims"]}, sort_keys=True))
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
    r.add_argument("--only-declared", action="store_true",
                   help="claim only items that name this host (--host) or a "
                        "tag this worker carries (--require). Use on any box "
                        "that is not interchangeable with the others, so "
                        "work written for the Sparks is never claimed by an "
                        "unlike machine.")
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

    rs = sub.add_parser(
        "reserve-set",
        help="atomically acquire/verify/release a multi-host reservation")
    rs_sub = rs.add_subparsers(dest="reserve_set_command", required=True)

    rsa = rs_sub.add_parser("acquire", help="acquire every host or none")
    rsa.add_argument("--id", required=True)
    rsa.add_argument("--owner-file", required=True,
                     help="0600 capability file; never copy it into receipts")
    rsa.add_argument("--host-resource", "--claim", action="append",
                     required=True, dest="host_resources",
                     metavar="HOST:GPU:MEM:GPU_CAP:MEM_CAP",
                     help="repeat once per host")
    rsa.add_argument("--require-no-claimed", action="store_true",
                     help="return 75 if any target host has a claimed item")
    rsa.add_argument("--why", default="")
    rsa.set_defaults(func=cmd_reserve_set_acquire)

    rsv = rs_sub.add_parser("verify", help="verify ownership and show claims")
    rsv.add_argument("--id", required=True)
    rsv.add_argument("--owner-file", required=True)
    rsv.set_defaults(func=cmd_reserve_set_verify)

    rsr = rs_sub.add_parser("release", help="release every host atomically")
    rsr.add_argument("--id", required=True)
    rsr.add_argument("--owner-file", required=True)
    rsr.set_defaults(func=cmd_reserve_set_release)

    a = ap.parse_args(argv)
    if getattr(a, "host", None) is None and a.command == "reserve":
        a.host = socket.gethostname().split(".")[0]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
