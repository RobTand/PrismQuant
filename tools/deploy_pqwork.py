#!/usr/bin/env python3
"""Atomically deploy pqwork from this checkout to the two live workers.

Run only between claims.  The command holds new admissions, refuses to replace
any copy while ``claimed/*.json`` is non-empty, verifies identical hashes, and
restarts the user services before releasing the hold.  The first deployment of
a worker that predates the hold mechanism still relies on the operator choosing
an actually idle window; later deployments close the admission race themselves.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_QUEUE_ROOT = Path("/mnt/shared/pq-queue")
DEFAULT_HOSTS = ("sparky", "sparklina")
LOCAL_TARGET = Path("/home/rob/.local/bin/pqwork.py")
DEPLOY_HOLD = ".deploying"
ADMISSION_LOCK = ".admission-lock"
ADMISSION_LOCK_WAIT_S = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_install(source: Path, target: Path) -> None:
    """Replace ``target`` atomically without using ``/tmp``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.parent / f".{target.name}.deploy-{os.getpid()}"
    try:
        shutil.copyfile(source, staged)
        staged.chmod(0o755)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def claimed_items(queue_root: Path) -> list[str]:
    return sorted(path.stem for path in (queue_root / "claimed").glob("*.json"))


@contextlib.contextmanager
def admission_guard(queue_root: Path):
    """Use the same NFS namespace mutex as worker final admission."""
    lock = queue_root / "reserved" / ADMISSION_LOCK
    token = f"deploy:{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
    deadline = time.monotonic() + ADMISSION_LOCK_WAIT_S
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for queue admission guard {lock}; "
                    "refusing unsafe lock stealing")
            time.sleep(0.05)
    owner = lock / "owner.json"
    try:
        try:
            owner.write_text(json.dumps({
                "token": token,
                "host": socket.gethostname().split(".")[0],
                "pid": os.getpid(),
                "acquired_at": time.time(),
                "purpose": "pqwork deployment hold acquisition",
            }))
        except Exception:
            owner.unlink(missing_ok=True)
            lock.rmdir()
            raise
        yield
    finally:
        try:
            current = json.loads(owner.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        if current.get("token") == token:
            owner.unlink(missing_ok=True)
            try:
                lock.rmdir()
            except OSError:
                pass


def acquire_hold(queue_root: Path) -> Path:
    """Publish the deployment hold and check claims at one linearization point."""
    hold = queue_root / DEPLOY_HOLD
    with admission_guard(queue_root):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(hold, flags, 0o644)
        with os.fdopen(fd, "w") as fh:
            fh.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
        running = claimed_items(queue_root)
        if running:
            hold.unlink(missing_ok=True)
            raise RuntimeError(
                "deployment is safe only between claims; currently claimed: "
                + ", ".join(running))
    return hold


def is_local_host(host: str) -> bool:
    local = socket.gethostname().split(".")[0]
    return host in (local, "localhost")


def validate_host(host: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
        raise ValueError(f"invalid host name: {host!r}")


def run_remote(host: str, command: str) -> str:
    result = subprocess.run(["ssh", host, command], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def remote_install(host: str, shared_source: Path) -> None:
    source = shlex.quote(str(shared_source))
    target = shlex.quote(str(LOCAL_TARGET))
    # $$ must expand remotely, so quote the fixed directory and construct the
    # basename in the remote shell rather than interpolating user input.
    command = (
        "set -eu; "
        f"src={source}; dst={target}; "
        "stage=/home/rob/.local/bin/.pqwork.py.deploy-$$; "
        "trap 'rm -f \"$stage\"' EXIT; "
        "install -m 0755 \"$src\" \"$stage\"; "
        "mv -f \"$stage\" \"$dst\"; trap - EXIT"
    )
    run_remote(host, command)


def restart_worker(host: str) -> None:
    command = ["systemctl", "--user", "restart", "pqwork"]
    if is_local_host(host):
        subprocess.run(command, check=True)
    else:
        run_remote(host, "systemctl --user restart pqwork")


def installed_hash(host: str) -> str:
    if is_local_host(host):
        return sha256(LOCAL_TARGET)
    output = run_remote(host, f"sha256sum {shlex.quote(str(LOCAL_TARGET))}")
    return output.split()[0] if output else ""


def deploy(source: Path, queue_root: Path, hosts: list[str],
           *, restart: bool = True) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"repo source does not exist: {source}")
    if not (queue_root / "claimed").is_dir():
        raise FileNotFoundError(
            f"queue root has no claimed/ directory: {queue_root}")
    for host in hosts:
        validate_host(host)
        if not is_local_host(host):
            run_remote(host, "true")

    source_digest = sha256(source)
    hold = acquire_hold(queue_root)
    replacement_started = False
    completed = False
    try:
        running = claimed_items(queue_root)
        if running:
            raise RuntimeError(
                "deployment is safe only between claims; currently claimed: "
                + ", ".join(running))

        shared_target = queue_root / "bin" / "pqwork.py"
        replacement_started = True
        atomic_install(source, shared_target)
        expected = sha256(shared_target)
        if expected != source_digest or sha256(source) != source_digest:
            raise RuntimeError(
                "source changed during deployment or shared copy differs: "
                f"source_before={source_digest} shared={expected} "
                f"source_after={sha256(source)}")

        for host in hosts:
            if is_local_host(host):
                atomic_install(shared_target, LOCAL_TARGET)
            else:
                remote_install(host, shared_target)

        actual = {host: installed_hash(host) for host in hosts}
        wrong = {host: digest for host, digest in actual.items()
                 if digest != expected}
        if wrong:
            raise RuntimeError(f"installed hashes differ from {expected}: {wrong}")

        if restart:
            for host in hosts:
                restart_worker(host)
        completed = True
        return expected
    finally:
        # Once any deployed byte changed, a failure can leave versions split.
        # Retain the hold so no new work enters a partially deployed fleet.
        if completed or not replacement_started:
            hold.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path,
        default=Path(__file__).resolve().with_name("pqwork.py"),
        help="repo pqwork.py to deploy (default: sibling of this script)")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--host", action="append", dest="hosts",
                        help="target worker; repeatable (default: both GB10s)")
    parser.add_argument(
        "--no-restart", action="store_true",
        help="install and verify copies but leave services stopped; intended "
             "for the first rollout from hold-unaware workers")
    args = parser.parse_args(argv)
    hosts = args.hosts or list(DEFAULT_HOSTS)
    try:
        digest = deploy(args.source, args.queue_root, hosts,
                        restart=not args.no_restart)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        held = args.queue_root / DEPLOY_HOLD
        suffix = (f"; admission hold retained at {held}"
                  if held.exists() else "")
        print(f"pqwork deployment refused: {exc}{suffix}", file=sys.stderr)
        return 1
    action = "workers not restarted" if args.no_restart else "workers restarted"
    print(f"deployed pqwork sha256={digest} via shared storage to "
          f"{', '.join(hosts)}; {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
