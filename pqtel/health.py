#!/usr/bin/python3
"""pqtel health -- per-box standing-defect report.

The rule this module enforces: a box it could not reach gets no verdict.
Every check carries a status of OK, DEFECT, INFO or UNVERIFIED, and
UNVERIFIED is never silently upgraded. In particular, lina answers on its
Netdata HTTP port, which makes "Netdata reachable" a verified fact about
lina -- and makes nothing else about lina verified, because linger, the guard
flag, unit states and CSV freshness are local state that this session has no
path to. Reachable-for-metrics is not reachable-for-state.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqtel import CSV_DIR  # noqa: E402

OK = "OK"
DEFECT = "DEFECT"
INFO = "INFO"
UNVERIFIED = "UNVERIFIED"

GUARD_FLAG = "/etc/gpu-guardian.disable"
GUARD_UNIT = "gpu-guardian"
OLD_VITALS_DIR = "/home/rob/dq-runs/telemetry"
NETDATA_LOCAL = "http://127.0.0.1:19999"

# The peer is "the other box", resolved from this box's hostname. Hardcoding
# one address made `pqtel health` on lina point its remote section at lina --
# it reported 8 UNVERIFIED checks against itself and labelled the result
# "lina (remote)" while standing on lina. A two-box cluster still has to be
# told which end it is standing on.
#
# `lina` is the operator's name for the host whose `hostname` is gx10-6b77;
# both spellings map to the same peer entry so either box resolves the other.
PEERS = {
    "sparky": ("lina / 192.168.1.110", "http://192.168.1.110:19999",
               "sparklina"),
    "gx10-6b77": ("sparky / 192.168.1.180", "http://192.168.1.180:19999",
                  "sparky"),
    "lina": ("sparky / 192.168.1.180", "http://192.168.1.180:19999",
             "sparky"),
}


def peer_for(host):
    """Return (label, netdata_url) for the other box, or None if unknown.

    An unrecognized hostname yields no peer rather than a guessed one: a
    remote section aimed at the wrong box is worse than no remote section,
    because every row in it reads as a fact about a machine it never touched.
    """
    return PEERS.get(host)


def ssh_probe(alias, timeout=8):
    """Is there a working shell path to `alias`?  MEASURE it, never assert it.

    The previous form hardcoded "no ssh key; hostname does not resolve" and
    emitted it as a fact for eight checks per run.  Both boxes have carried a
    working NVIDIA-Sync alias over the 10.100.96.x link the whole time
    (sparky -> `sparklina`, lina -> `sparky`), so every one of those rows was
    a false statement about the environment.  A claim about another machine is
    attested or refused; it is not assumed.
    """
    if not alias:
        return False, "no ssh alias configured for this peer in PEERS"
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
             "-o", "StrictHostKeyChecking=accept-new", alias, "true"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, "ssh %s failed to launch: %s" % (alias, exc)
    if proc.returncode == 0:
        return True, alias
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, "ssh %s exited %d: %s" % (
        alias, proc.returncode, err[-1] if err else "no output")


def remote_health(alias, timeout=90):
    """Run the peer's OWN local checks over ssh and return its boxes, or None.

    `--local-only` on the far side is what stops this recursing: the peer
    reports on itself and does not turn around and probe back.
    """
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", alias,
             "cd ~/prismaquant && python3 -m pqtel health --json --local-only"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "ssh %s: %s" % (alias, exc)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return None, "remote pqtel health exited %d: %s" % (
            proc.returncode, err[-1] if err else "no output")
    try:
        return json.loads(proc.stdout), None
    except ValueError as exc:
        return None, "remote pqtel health returned non-JSON: %s" % exc


class Report:
    def __init__(self):
        self.boxes = []

    def box(self, name):
        entry = {"box": name, "checks": []}
        self.boxes.append(entry)
        return entry

    @staticmethod
    def add(entry, name, status, detail):
        entry["checks"].append({"check": name, "status": status,
                                "detail": detail})

    def render(self):
        lines = []
        defects = 0
        unverified = 0
        for entry in self.boxes:
            lines.append("=== %s ===" % entry["box"])
            width = max((len(c["check"]) for c in entry["checks"]), default=0)
            for check in entry["checks"]:
                if check["status"] == DEFECT:
                    defects += 1
                elif check["status"] == UNVERIFIED:
                    unverified += 1
                lines.append("  %-10s %-*s  %s" % (
                    check["status"], width, check["check"], check["detail"]))
            lines.append("")
        lines.append("%d DEFECT, %d UNVERIFIED" % (defects, unverified))
        return "\n".join(lines), defects, unverified


def _systemctl(args, timeout=5):
    try:
        proc = subprocess.run(["systemctl"] + args, capture_output=True,
                              text=True, timeout=timeout)
        return proc.stdout.strip(), proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        return "error: %s" % exc, 127


def _http_ok(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return None, str(exc)


def _last_csv_row_epoch(path):
    """Epoch seconds of the last data row, parsed from the row itself.

    File mtime would be a proxy; a recorder that is alive but wedged mid-write
    still touches mtime. The row's own timestamp is the fact.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 8192))
            tail = fh.read().splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        first = line.split(b",")[0].strip()
        if not first or not first.replace(b".", b"", 1).isdigit():
            continue
        value = float(first)
        # pqteld writes epoch_ms; the old recorder wrote epoch seconds.
        return value / 1000.0 if value > 1e11 else value
    return None


def _age_text(seconds):
    if seconds is None:
        return "unknown"
    if seconds < 120:
        return "%.0f s" % seconds
    if seconds < 7200:
        return "%.1f min" % (seconds / 60.0)
    return "%.1f h" % (seconds / 3600.0)


def _stamp(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


# ---------------------------------------------------------------------------
# Individual checks (local box).
# ---------------------------------------------------------------------------

def check_pqteld(report, entry, csv_dir):
    state, _ = _systemctl(["--user", "is-active", "pqteld"])
    files = []
    try:
        files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    except OSError:
        pass
    if not files:
        report.add(entry, "pqteld recorder", DEFECT,
                   "unit %s; no CSV under %s" % (state, csv_dir))
        return
    newest = os.path.join(csv_dir, files[-1])
    epoch = _last_csv_row_epoch(newest)
    age = None if epoch is None else time.time() - epoch
    detail = "unit %s; last row %s (%s ago); %s" % (
        state, _stamp(epoch) if epoch else "unparsable", _age_text(age), newest)
    if state == "active" and age is not None and age < 30:
        report.add(entry, "pqteld recorder", OK, detail)
    else:
        report.add(entry, "pqteld recorder", DEFECT, detail)


def check_old_vitals(report, entry, host):
    """The predecessor CSV recorder -- graded against whether it still matters.

    The old form graded this ABSOLUTELY: any stale file was a DEFECT, on the
    stated premise that "the ad-hoc recorders under /home/rob/dq-runs/telemetry/
    are what the pipeline scripts still read." That premise was checked on
    2026-08-28 and is false -- the only files in the tree that touch
    /home/rob/dq-runs/telemetry are that recorder's own vitals_record.sh and
    vitals_query.sh, and the Claude vitals hook reads nvidia-smi directly, not
    the CSV. A deliberately retired predecessor is not a defect; that is the
    same mistake the two-state guardian check made.

    So the verdict is conditional on the SUCCESSOR: if pqteld is carrying
    telemetry, a dead predecessor is retirement (INFO). If pqteld is ALSO
    dead, the box has no recorder at all and the staleness is a real DEFECT --
    which is the only situation in which this file was ever the live source.
    """
    pqteld_ok = any(c.get("check") == "pqteld recorder"
                    and c.get("status") == OK
                    for c in entry.get("checks", ()))
    try:
        files = sorted(f for f in os.listdir(OLD_VITALS_DIR)
                       if f.startswith("vitals-") and f.endswith(".csv"))
    except OSError as exc:
        report.add(entry, "old vitals recorder", UNVERIFIED,
                   "%s unreadable: %s" % (OLD_VITALS_DIR, exc))
        return
    if not files:
        report.add(entry, "old vitals recorder", INFO,
                   "no vitals-*.csv under %s" % OLD_VITALS_DIR)
        return
    newest = os.path.join(OLD_VITALS_DIR, files[-1])
    epoch = _last_csv_row_epoch(newest)
    age = None if epoch is None else time.time() - epoch
    detail = "last row %s (%s ago); %s" % (
        _stamp(epoch) if epoch else "unparsable", _age_text(age), newest)
    if age is not None and age < 300:
        report.add(entry, "old vitals recorder", OK, detail)
    elif pqteld_ok:
        report.add(entry, "old vitals recorder", INFO,
                   detail + " -- RETIRED: superseded by pqteld (live), and no "
                            "consumer outside its own vitals_{record,query}.sh "
                            "remains in the tree")
    else:
        report.add(entry, "old vitals recorder", DEFECT,
                   detail + " -- dead, AND pqteld is not OK: this box has no "
                            "live telemetry recorder at all")


def check_gpu_guardian(report, entry):
    """The disarm flag versus the unit state -- as three states, not two.

    The old form read flag-absence as "armed" regardless of whether the unit
    was enabled at all, so a deliberately retired guardian printed
    "unit inactive; ... guard armed" -- a contradiction in one row. Retirement
    by the operator, disarming by flag, and a unit that should be running but
    is not are three different facts and get three different verdicts.

    This check REPORTS only. Nothing here re-arms the guard, deletes the flag,
    or restarts the unit.
    """
    unit_state, _ = _systemctl(["is-active", GUARD_UNIT])
    enabled_state, _ = _systemctl(["is-enabled", GUARD_UNIT])
    retired = enabled_state in ("disabled", "masked", "not-found")

    flag = None
    try:
        flag = os.stat(GUARD_FLAG)
    except FileNotFoundError:
        pass
    except OSError as exc:
        report.add(entry, "gpu guardian", UNVERIFIED,
                   "cannot stat %s: %s" % (GUARD_FLAG, exc))
        return

    if retired:
        # Operator retired the guardian. A leftover flag is inert residue,
        # not a disarm -- there is nothing left for it to disarm.
        detail = "RETIRED: %s is %s (is-active '%s')" % (
            GUARD_UNIT, enabled_state, unit_state)
        if flag is not None:
            detail += ("; %s is leftover residue and now means nothing -- "
                       "remove it so it cannot be misread later" % GUARD_FLAG)
        report.add(entry, "gpu guardian", OK, detail)
        return

    if flag is None:
        if unit_state == "active":
            report.add(entry, "gpu guardian", OK,
                       "unit active (%s); no %s -- guard armed"
                       % (enabled_state, GUARD_FLAG))
        else:
            report.add(entry, "gpu guardian", DEFECT,
                       "unit is %s but is-active says '%s' and no %s exists "
                       "-- the guard is neither retired nor running"
                       % (enabled_state, unit_state, GUARD_FLAG))
        return

    try:
        import pwd
        owner = pwd.getpwuid(flag.st_uid).pw_name
    except (ImportError, KeyError):
        owner = "uid=%d" % flag.st_uid
    detail = ("DISARMED: %s exists (owner %s, %d bytes, mtime %s, %s ago) "
              "while %s is %s / is-active '%s'. Report only -- "
              "nothing here re-arms it." % (
                  GUARD_FLAG, owner, flag.st_size, _stamp(flag.st_mtime),
                  _age_text(time.time() - flag.st_mtime), GUARD_UNIT,
                  enabled_state, unit_state))
    report.add(entry, "gpu guardian", DEFECT, detail)


def check_netdata_unit(report, entry):
    """Defect (c): Netdata's own unit is not hardened against OOM.

    Restart=on-failure does not restart after a SIGKILL from the OOM killer
    (systemd sees a signal, not a failure exit, unless the unit says
    otherwise), and OOMScoreAdjust=0 leaves the collector as eligible for the
    kill as the workload it is collecting on. A collector that dies inside the
    collapse it is recording is the exact failure that made this project
    necessary.
    """
    out, rc = _systemctl(["show", "netdata.service",
                          "-p", "Restart", "-p", "OOMScoreAdjust",
                          "-p", "OOMPolicy"])
    if rc != 0:
        report.add(entry, "netdata unit", UNVERIFIED,
                   "systemctl show failed: %s" % out)
        return
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    restart = props.get("Restart", "?")
    oom = props.get("OOMScoreAdjust", "?")
    detail = "Restart=%s, OOMScoreAdjust=%s, OOMPolicy=%s" % (
        restart, oom, props.get("OOMPolicy", "?"))
    if restart == "always" and oom.lstrip("-").isdigit() and int(oom) < 0:
        report.add(entry, "netdata unit", OK, detail + " -- hardened")
    else:
        report.add(entry, "netdata unit", DEFECT,
                   detail + " -- unhardened; drop-in owed (needs sudo)")


def check_netdata_reachable(report, entry, url, label):
    status, body = _http_ok(url + "/api/v1/info")
    if status != 200:
        report.add(entry, label, DEFECT if url == NETDATA_LOCAL else UNVERIFIED,
                   "%s unreachable: %s" % (url, body))
        return None
    try:
        info = json.loads(body)
        version = info.get("version", "?")
        hostname = (info.get("mirrored_hosts") or ["?"])[0]
    except (ValueError, TypeError, IndexError):
        version, hostname = "?", "?"
    report.add(entry, label, OK,
               "%s -> HTTP 200, netdata %s, host %s" % (url, version, hostname))
    return info


def check_linger(report, entry, user="rob"):
    try:
        proc = subprocess.run(["loginctl", "show-user", user],
                              capture_output=True, text=True, timeout=5)
        for line in proc.stdout.splitlines():
            if line.startswith("Linger="):
                value = line.split("=", 1)[1]
                status = OK if value == "yes" else DEFECT
                detail = "Linger=%s for %s" % (value, user)
                if value != "yes":
                    detail += (" -- systemd --user cannot start at boot; "
                               "run: sudo loginctl enable-linger %s" % user)
                report.add(entry, "linger", status, detail)
                return
    except (OSError, subprocess.SubprocessError) as exc:
        report.add(entry, "linger", UNVERIFIED, "loginctl failed: %s" % exc)
        return
    report.add(entry, "linger", UNVERIFIED, "Linger= not reported by loginctl")


def check_profiling_gates(report, entry):
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as fh:
            paranoid = fh.read().strip()
    except OSError as exc:
        report.add(entry, "perf_event_paranoid", UNVERIFIED, str(exc))
    else:
        # <=2 permits unprivileged CPU sampling and --cpuctxsw.
        status = OK if paranoid.lstrip("-").isdigit() and int(paranoid) <= 2 \
            else DEFECT
        detail = "kernel.perf_event_paranoid=%s" % paranoid
        if status == DEFECT:
            detail += (" -- CPU sampling disabled; run: echo "
                       "'kernel.perf_event_paranoid=2' | sudo tee "
                       "/etc/sysctl.d/99-perf.conf && sudo sysctl --system")
        report.add(entry, "perf_event_paranoid", status, detail)

    value = None
    try:
        with open("/proc/driver/nvidia/params") as fh:
            for line in fh:
                if line.startswith("RmProfilingAdminOnly:"):
                    value = line.split(":", 1)[1].strip()
    except OSError as exc:
        report.add(entry, "RmProfilingAdminOnly", UNVERIFIED, str(exc))
        return
    if value is None:
        report.add(entry, "RmProfilingAdminOnly", UNVERIFIED,
                   "key absent from /proc/driver/nvidia/params")
        return
    status = OK if value == "0" else DEFECT
    detail = "RmProfilingAdminOnly: %s" % value
    if status == DEFECT:
        detail += (" -- GPU counters/ncu blocked; run: echo 'options nvidia "
                   "NVreg_RestrictProfilingToAdminUsers=0' | sudo tee "
                   "/etc/modprobe.d/nvidia-profiling.conf && sudo "
                   "update-initramfs -u && sudo reboot")
    report.add(entry, "RmProfilingAdminOnly", status, detail)


def check_disk(report, entry, path="/home/rob"):
    try:
        # f_bavail, not f_bfree: this filesystem reserves 24.4 M blocks for
        # root, so f_bfree reads 402 GB where f_bavail reads 302 GB. Reported
        # in GiB to match `df -h` and the vitals hook.
        st = os.statvfs(path)
    except OSError as exc:
        report.add(entry, "free disk", UNVERIFIED, str(exc))
        return
    free_b = st.f_bavail * st.f_frsize
    total_b = st.f_blocks * st.f_frsize
    pct = 100.0 * free_b / total_b
    detail = "%.1f GiB free of %.1f GiB (%.1f%%) on %s" % (
        free_b / 2 ** 30, total_b / 2 ** 30, pct, path)
    # Rob's floor is 5% free; 10% is the pqtel gc refusal threshold.
    report.add(entry, "free disk", OK if pct >= 10 else DEFECT, detail)


def check_gpu_memory_source(report, entry):
    """Standing hardware fact, recorded so nobody re-litigates it.

    Every GPU-memory telemetry source on GB10 returns [N/A] or None. The
    uvm_residual column derived from /proc/meminfo is the only per-box GPU
    memory number that exists here.
    """
    report.add(entry, "gpu mem source", INFO,
               "GB10 unified memory: nvidia-smi memory.* = [N/A], netdata "
               "frame_buffer dims = None. /proc/meminfo is the only source; "
               "pqteld records uvm_residual_kb.")


# ---------------------------------------------------------------------------

def run(csv_dir=CSV_DIR, local_only=False):
    report = Report()
    host = os.uname().nodename

    local = report.box("%s (local)" % host)
    check_pqteld(report, local, csv_dir)
    check_old_vitals(report, local, host)
    check_gpu_guardian(report, local)
    check_netdata_unit(report, local)
    check_netdata_reachable(report, local, NETDATA_LOCAL, "netdata reachable")
    check_linger(report, local)
    check_profiling_gates(report, local)
    check_disk(report, local)
    check_gpu_memory_source(report, local)

    if local_only:
        # Invoked over ssh BY the other box: report on ourselves only, so the
        # peer section cannot recurse back across the link.
        return report

    peer = peer_for(host)
    if peer is None:
        remote = report.box("peer (remote)")
        report.add(report.boxes[-1], "peer identity", UNVERIFIED,
                   "hostname %r is in no known peer pair; add it to PEERS "
                   "rather than letting this box guess which machine is the "
                   "other end" % host)
    else:
        label, url, alias = peer
        remote = report.box("%s (remote)" % label)
        check_netdata_reachable(report, remote, url, "netdata reachable")
        ok, detail = ssh_probe(alias)
        if not ok:
            report.add(remote, "shell path", UNVERIFIED,
                       "%s -- only Netdata HTTP at %s" % (detail, url))
            for name in ("pqteld recorder", "old vitals recorder",
                         "gpu guardian", "netdata unit hardening", "linger",
                         "perf_event_paranoid", "RmProfilingAdminOnly",
                         "free disk"):
                report.add(remote, name, UNVERIFIED,
                           "no shell path (see 'shell path' above)")
        else:
            boxes, err = remote_health(alias)
            if boxes is None:
                report.add(remote, "shell path", UNVERIFIED,
                           "ssh %s works but the remote check failed: %s"
                           % (alias, err))
            else:
                report.add(remote, "shell path", OK,
                           "ssh %s -- rows below are the peer's OWN local "
                           "checks, run over that shell" % alias)
                for far in boxes:
                    if not far.get("box", "").endswith("(local)"):
                        continue
                    for check in far.get("checks", []):
                        report.add(remote, check.get("check", "?"),
                                   check.get("status", UNVERIFIED),
                                   check.get("detail", ""))

    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Run the checks ONCE. Rendering and serializing both read the same
    # Report; calling run() per output format would double every check,
    # including the HTTP round trips to the peer.
    report = run(local_only="--local-only" in argv)
    if "--json" in argv:
        print(json.dumps(report.boxes, indent=2))
    else:
        print(report.render()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
