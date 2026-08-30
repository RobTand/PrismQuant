#!/usr/bin/python3
"""pqtel -- the telemetry CLI. Step 1 verbs only: now, health, gc.

capture, window, incident, top, preflight, selftest, memdump, offset and the
MCP layer are later steps of the observability plan and are deliberately not
implemented here. An unimplemented verb refuses by name rather than being
absent, so a caller learns which step it is waiting on.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqtel import CSV_DIR, __version__  # noqa: E402

LINA_NETDATA = "http://192.168.1.110:19999"

# Verbs the plan defines but step 1 does not build. Named so a caller gets a
# refusal that says which step owns it, not a bare "unknown verb".
NOT_YET = {
    "window": "step 2", "incident": "step 2", "top": "step 2",
    "selftest": "step 3", "preflight": "step 3",
    "capture": "step 4", "runs": "step 4", "serve-wrap": "step 4",
    "pyspy": "step 4", "memdump": "step 5", "offset": "step 5",
    "setup-check": "step 5", "install": "step 5",
}

USAGE = """pqtel %s -- box telemetry (step 1: recorder + health)

  pqtel now                  One line per box, the hook's format.
  pqtel health [--json]      Per-box standing-defect report. Refuses a
                             verdict for any box it could not reach.
  pqtel gc [--days N] [--dry-run]
                             Delete CSVs older than N days (default 30).
                             Refuses when free disk is under 10%%.
  pqtel record [args]        Run the recorder in the foreground (systemd
                             runs this; you normally do not).

Data:  %s
Code:  %s
""" % (__version__, CSV_DIR, os.path.dirname(os.path.abspath(__file__)))

GC_MIN_FREE_PCT = 10.0


def disk_free(path):
    """Free and total bytes, using f_bavail -- what a non-root writer gets.

    f_bavail, not f_bfree: this filesystem reserves 24.4 M blocks for root, so
    f_bfree reads 402 GB where f_bavail reads 302 GB. A 100 GB overstatement
    is enough to walk a 10% floor past a box that is actually tighter.
    (shutil.disk_usage().free already uses f_bavail; statvfs is used here to
    make the choice explicit rather than inherited.)

    Sizes are reported in GiB, matching `df -h` and the vitals hook. The
    decimal-GB spelling reads 302 against df's 282 for the same bytes, and
    that gap looks like a bug every time someone compares the two.
    """
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize


def _meminfo():
    out = {}
    with open("/proc/meminfo", "rb") as fh:
        for line in fh:
            key, _, rest = line.decode().partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0])
    return out


def _nvidia_once():
    import subprocess
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,temperature.gpu,"
             "clocks_event_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    return [p.strip() for p in line[0].split(",")]


def _lina_line():
    def query(context):
        url = ("%s/api/v2/data?contexts=%s&after=-1&points=1&format=json2"
               "&group_by=context" % (LINA_NETDATA, context))
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                doc = json.loads(resp.read())
        except (urllib.error.URLError, socket.timeout, OSError, ValueError):
            return None
        try:
            value = doc["result"]["data"][0][1]
            return value[0] if isinstance(value, list) else value
        except (KeyError, IndexError, TypeError):
            return None
    mem = query("mem.available")
    load = query("system.load")
    gpu = query("nvidia_smi.gpu_utilization")
    if mem is None:
        # Unreachable is stated, never guessed at from a cached value.
        return "lina: UNREACHABLE (netdata %s)" % LINA_NETDATA
    parts = ["mem %.0fG avail" % (mem / 1024.0)]
    if load is not None:
        parts.append("load %.2f" % load)
    if gpu is not None:
        parts.append("gpu %.0f%%" % gpu)
    return "lina: " + ", ".join(parts)


def verb_now(argv):
    mem = _meminfo()
    total_g = mem["MemTotal"] / 1048576.0
    avail_g = mem["MemAvailable"] / 1048576.0
    anon_g = mem.get("AnonPages", 0) / 1048576.0
    residual_g = (mem["MemTotal"] - sum(
        mem.get(k, 0) for k in ("MemFree", "Buffers", "Cached", "Slab",
                                "AnonPages", "PageTables", "KernelStack")
    )) / 1048576.0
    gpu = _nvidia_once()
    if gpu:
        gpu_s = "gpu %s%% %sW %sC" % (gpu[0], gpu[1], gpu[2])
        if gpu[3] not in ("0x0000000000000000", "0x0000000000000001"):
            gpu_s += " thr:%s" % gpu[3]
    else:
        gpu_s = "gpu n/a"
    with open("/proc/loadavg") as fh:
        load1 = fh.read().split()[0]
    free_b, _ = disk_free("/home/rob")
    print("[%s] mem %.1fG/%.1fG avail (anon %.1fG, uvm-residual %.1fG) | %s "
          "| load %s | disk %.0fG free | %s" % (
              os.uname().nodename, avail_g, total_g, anon_g, residual_g,
              gpu_s, load1, free_b / 2 ** 30, _lina_line()))
    return 0


def verb_health(argv):
    from pqtel import health
    return health.main(argv)


def verb_gc(argv):
    """Retention. Refuses under 10% free disk, deliberately.

    Low disk is exactly when an incident is likely in progress and the CSV is
    the flight recorder, so gc declines to delete evidence mid-crisis. Free
    space then has to come from somewhere a human chose.
    """
    days = 30
    dry = "--dry-run" in argv
    for i, arg in enumerate(argv):
        if arg == "--days" and i + 1 < len(argv):
            days = int(argv[i + 1])
    free_b, total_b = disk_free(
        CSV_DIR if os.path.isdir(CSV_DIR) else "/home/rob")
    pct = 100.0 * free_b / total_b
    if pct < GC_MIN_FREE_PCT:
        print("REFUSED: free disk %.1f%% is under the %.0f%% floor. gc will "
              "not delete telemetry while the box is short on space -- that is "
              "when the recording matters most. Free space by hand, then "
              "re-run." % (pct, GC_MIN_FREE_PCT))
        return 2
    cutoff = time.time() - days * 86400
    removed, kept, freed = [], 0, 0
    try:
        names = sorted(os.listdir(CSV_DIR))
    except OSError as exc:
        print("REFUSED: cannot list %s: %s" % (CSV_DIR, exc))
        return 2
    for name in names:
        if not name.endswith(".csv"):
            continue
        path = os.path.join(CSV_DIR, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if st.st_mtime < cutoff:
            freed += st.st_size
            removed.append(name)
            if not dry:
                try:
                    os.unlink(path)
                except OSError as exc:
                    print("  could not remove %s: %s" % (name, exc))
        else:
            kept += 1
    print("gc: retention %d days, free disk %.1f%%%s" % (
        days, pct, " (DRY RUN)" if dry else ""))
    for name in removed:
        print("  %s %s" % ("would remove" if dry else "removed", name))
    print("  %d removed, %d kept, %.1f MB freed" % (
        len(removed), kept, freed / 1e6))
    return 0


def verb_record(argv):
    from pqtel import recorder
    return recorder.main(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    verb, rest = argv[0], argv[1:]
    if verb in ("-V", "--version", "version"):
        print("pqtel %s" % __version__)
        return 0
    handlers = {"now": verb_now, "health": verb_health, "gc": verb_gc,
                "record": verb_record}
    if verb in handlers:
        return handlers[verb](rest)
    if verb in NOT_YET:
        print("pqtel %s is not built yet: it belongs to %s of the "
              "observability plan. Step 1 shipped `now`, `health` and `gc`."
              % (verb, NOT_YET[verb]), file=sys.stderr)
        return 3
    print("unknown verb %r\n\n%s" % (verb, USAGE), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
