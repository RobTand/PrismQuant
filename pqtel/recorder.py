#!/usr/bin/python3
"""pqteld -- the crash-proof telemetry recorder for GB10 / DGX Spark.

Why this exists rather than another Netdata module: Netdata's memory model
assumes a discrete GPU framebuffer. On GB10 every `nvidia-smi memory.*` field
returns `[N/A]` and Netdata's `nvidia_smi.*_frame_buffer_memory_usage`
dimensions read `None`, so no generic collector can report GPU memory here.
`/proc/meminfo` is the only honest source, and the `uvm_residual` column below
is the only per-box GPU-memory number that exists on this hardware.

Durability contract
-------------------
Rows are appended with a single unbuffered `os.write` to a file opened
`O_APPEND`. The bytes are in the page cache before the call returns, so a row
survives `SIGKILL`, the OOM killer, and a `systemctl restart` -- everything
except a power cut, which would need `fsync` per row and is not worth the
write amplification. There is no Python-level buffering anywhere on the write
path.

Missing-data contract
---------------------
An empty CSV cell means "not measured". `-1` sentinels are forbidden: the
predecessor recorder wrote `gpu_mem_mb=-1` for every row of its life, which is
a lie that survives into any plot or allocator that reads it. Fields sampled
at a slower cadence than the row rate are empty on the rows between their
ticks, which is the same convention.

Cadences
--------
  2   Hz  /proc/meminfo, /proc/pressure/{memory,io}, /proc/vmstat + derived
  1   Hz  NVML, via ONE long-lived `nvidia-smi ... -l 1` (never a spawn per
          sample), carried forward into 2 Hz rows with a staleness cutoff
  0.2 Hz  guard-flag state, cgroup slice memory, top-8 process table
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqtel import CSV_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# nvidia-smi field allowlist / denylist.
#
# HARDWARE: NVIDIA GB10 (DGX Spark, "sparky"/"lina"), Blackwell sm_121,
# 128 GB UNIFIED memory -- CPU and GPU share one physical pool. There is no
# discrete framebuffer, so the framebuffer-shaped NVML fields do not merely
# read zero, they read the literal string `[N/A]`. nvidia-smi does NOT error on
# them; it prints `[N/A]` and exits 0. A recorder that queries one gets a
# column that looks healthy from every angle except a stimulus test.
#
# Verified on sparky 2026-08-28 with
#   nvidia-smi --query-gpu=<field> --format=csv,noheader,nounits
# ---------------------------------------------------------------------------

# Fields that return a real value on GB10.
NVML_ALLOW = [
    "utilization.gpu",
    "clocks.sm",
    "clocks.gr",
    "power.draw",
    "temperature.gpu",
    "pstate",
    "clocks_event_reasons.active",  # kept as the hex string, never parsed to int
]

# Fields that return `[N/A]` on GB10. Do not re-add one of these in six months
# because it works on a discrete card; it will silently produce a dead column.
# The value is the measured 2026-08-28 reading.
NVML_DENY = {
    # utilization.memory is the dangerous one, and it is why this recorder
    # ships with a stimulus test. It does NOT return `[N/A]`; it returns a
    # hard `0`, which looks like a measurement. Measured 2026-08-28 on sparky:
    # 71 consecutive samples over 14 s of a saturated GPU (utilization.gpu 96%,
    # power 88 W, clocks.sm 2.3 GHz, including a 2 GiB device-to-device copy
    # loop) reported max=0%, min=0%, avg=0% -- `nvidia-smi -q -d UTILIZATION`
    # agrees. Same root cause as the framebuffer fields: no discrete memory
    # controller to sample. The observability plan's section 3 listed this
    # field as returning a real value on GB10; the stimulus test falsified it.
    "utilization.memory": "always 0 -- no discrete memory controller",
    "memory.used": "[N/A] -- unified memory, no discrete framebuffer",
    "memory.total": "[N/A] -- unified memory; use /proc/meminfo MemTotal",
    "memory.free": "[N/A] -- unified memory; use /proc/meminfo MemAvailable",
    "memory.reserved": "[N/A] -- unified memory",
    "clocks.mem": "[N/A] -- no separate memory clock domain",
    "power.limit": "[N/A]",
    "power.default_limit": "[N/A]",
    "power.min_limit": "[N/A]",
    "power.max_limit": "[N/A]",
    "fan.speed": "[N/A] -- passively cooled",
    "temperature.memory": "N/A -- no discrete memory die sensor",
    "ecc.mode.current": "[N/A]",
    "ecc.errors.corrected.volatile.total": "[N/A]",
    "mig.mode.current": "[N/A] -- MIG unsupported on GB10",
    "accounting.mode": "Disabled -- per-process GPU accounting unavailable",
}

# Per-process GPU memory is deliberately absent. `nvidia-smi pmon`,
# `pmon -s um` and `--query-compute-apps` all return empty on GB10, and the
# `dmem` cgroup controller is unpopulated. The finest attribution that exists
# here is per-cgroup-slice memory.current, which is what CG_SLICES records.
CG_SLICES = [
    ("cg_system_slice_kb", "/sys/fs/cgroup/system.slice/memory.current"),
    ("cg_user_slice_kb", "/sys/fs/cgroup/user.slice/memory.current"),
]

MEMINFO_KEYS = [
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable",
    "Slab", "AnonPages", "Shmem", "PageTables", "KernelStack", "Dirty",
    "SwapTotal", "SwapFree",
]

VMSTAT_KEYS = [
    "oom_kill", "allocstall_normal", "allocstall_movable", "pgsteal_direct",
    "pgmajfault",
]

# uvm_residual = MemTotal - (everything the kernel can name).
# What is left over is memory the kernel has handed out but does not account
# in any of these buckets -- on GB10 that is dominated by the NVIDIA driver's
# UVM allocations, i.e. "GPU memory". It is a residual, not a driver-reported
# figure, and it is the only such number available on this hardware.
UVM_SUBTRAHENDS = [
    "MemFree", "Buffers", "Cached", "Slab", "AnonPages", "PageTables",
    "KernelStack",
]

# mem_slope_gbs is the least-squares slope of MemAvailable over the trailing
# 60 s, in GB/s, SIGNED: negative means memory is being consumed. It turns
# "memory is low" into "memory is falling at 4.1 GB/s, 12 s to floor".
SLOPE_WINDOW_S = 60.0

# Row schema. Order is the CSV column order. A change here must change
# SCHEMA_VERSION, which forces a new file rather than misaligned columns.
SCHEMA_VERSION = 2   # v2 dropped gpu_mem_util: proven dead, see NVML_DENY
COLUMNS = (
    ["epoch_ms", "dt_ms", "boot_id"]
    + MEMINFO_KEYS
    + ["psi_mem_full_avg10", "psi_mem_full_total", "psi_io_full_avg10"]
    + VMSTAT_KEYS
    + ["uvm_residual_kb", "mem_slope_gbs"]
    + ["gpu_util", "clocks_sm", "clocks_gr", "power_draw_w",
       "temp_gpu_c", "pstate", "clocks_event_reasons_active"]
    + ["guard_flag_present", "guard_flag_age_s", "guard_unit_active"]
    + [name for name, _ in CG_SLICES]
    + ["top_rss", "top_read_bytes", "top_rchar"]
)

GUARD_FLAG = "/etc/gpu-guardian.disable"
GUARD_UNIT = "gpu-guardian"

_SANITIZE = re.compile(r"[,|:\r\n]")


def _sanitize(text):
    """Make a string safe for a CSV cell and for the `|`/`:` packed lists.

    Process comms are attacker-shaped by accident: `ray::RayWorkerP` carries
    colons, and a comma in a comm would corrupt every column after it.
    """
    return _SANITIZE.sub("_", text)


# ---------------------------------------------------------------------------
# /proc readers. Each returns a dict of column -> string; a key that is absent
# from the dict becomes an empty cell, never a sentinel.
# ---------------------------------------------------------------------------

def read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo", "rb") as fh:
            for line in fh:
                key, _, rest = line.decode().partition(":")
                if key in MEMINFO_KEYS:
                    out[key] = rest.split()[0]
    except OSError:
        return {}
    return out


def read_pressure():
    out = {}
    for path, prefix in (("/proc/pressure/memory", "psi_mem"),
                         ("/proc/pressure/io", "psi_io")):
        try:
            with open(path, "rb") as fh:
                for line in fh:
                    parts = line.decode().split()
                    if not parts or parts[0] != "full":
                        continue
                    for field in parts[1:]:
                        name, _, value = field.partition("=")
                        if name == "avg10":
                            out[prefix + "_full_avg10"] = value
                        elif name == "total" and prefix == "psi_mem":
                            out["psi_mem_full_total"] = value
        except OSError:
            pass
    return out


def read_vmstat():
    out = {}
    try:
        with open("/proc/vmstat", "rb") as fh:
            for line in fh:
                key, _, value = line.decode().partition(" ")
                if key in VMSTAT_KEYS:
                    out[key] = value.strip()
    except OSError:
        pass
    return out


def read_boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def read_cgroup_slices():
    out = {}
    for name, path in CG_SLICES:
        try:
            with open(path) as fh:
                out[name] = str(int(fh.read().strip()) // 1024)
        except (OSError, ValueError):
            pass
    return out


def read_guard_state():
    """Guard-flag state versus unit state.

    A root-owned zero-byte /etc/gpu-guardian.disable sitting next to a
    `systemctl is-active` of `active` is invisible to every generic collector,
    and is exactly the disarm that a build walks into. The recorder never
    touches the flag; it only records it.
    """
    out = {}
    try:
        st = os.stat(GUARD_FLAG)
        out["guard_flag_present"] = "1"
        out["guard_flag_age_s"] = "%d" % max(0, int(time.time() - st.st_mtime))
    except FileNotFoundError:
        out["guard_flag_present"] = "0"
    except OSError:
        pass
    try:
        proc = subprocess.run(["systemctl", "is-active", GUARD_UNIT],
                              capture_output=True, text=True, timeout=5)
        out["guard_unit_active"] = _sanitize(proc.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def read_top_processes(n=8):
    """Top n processes by RSS, with read_bytes and rchar as separate columns.

    read_bytes counts block-device traffic and rchar counts bytes read through
    the syscall; their divergence is how you tell a page-cache hit from an
    NVMe read. Both are empty for a process whose /proc/PID/io we may not read.
    """
    rows = []
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return {}
    for pid in pids:
        try:
            with open("/proc/%s/statm" % pid, "rb") as fh:
                rss_pages = int(fh.read().split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if rss_pages == 0:
            continue
        try:
            with open("/proc/%s/comm" % pid, "rb") as fh:
                comm = _sanitize(fh.read().decode("utf-8", "replace").strip())
        except OSError:
            continue
        rows.append((rss_pages, pid, comm))
    if not rows:
        return {}
    rows.sort(reverse=True)
    rows = rows[:n]
    page_mb = os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)

    rss_parts, rb_parts, rc_parts = [], [], []
    for rss_pages, pid, comm in rows:
        tag = "%s:%s" % (pid, comm)
        rss_parts.append("%s:%d" % (tag, int(rss_pages * page_mb)))
        read_bytes = rchar = None
        try:
            with open("/proc/%s/io" % pid, "rb") as fh:
                for line in fh:
                    key, _, value = line.decode().partition(":")
                    if key == "read_bytes":
                        read_bytes = int(value)
                    elif key == "rchar":
                        rchar = int(value)
        except (OSError, ValueError):
            pass
        # An unreadable /proc/PID/io yields no entry at all, not a zero:
        # "we may not look" and "it read nothing" are different facts.
        if read_bytes is not None:
            rb_parts.append("%s:%d" % (tag, read_bytes // (1024 * 1024)))
        if rchar is not None:
            rc_parts.append("%s:%d" % (tag, rchar // (1024 * 1024)))
    return {
        "top_rss": "|".join(rss_parts),
        "top_read_bytes": "|".join(rb_parts),
        "top_rchar": "|".join(rc_parts),
    }


# ---------------------------------------------------------------------------
# NVML via one long-lived nvidia-smi.
# ---------------------------------------------------------------------------

class NvmlStream:
    """One `nvidia-smi --query-gpu=... -l 1` process, read by a thread.

    nvidia-smi line-buffers when its stdout is a pipe (verified on sparky),
    so a plain readline loop delivers a sample per second without a pty.

    Staleness is the point of the timestamp: a reader that wedges would
    otherwise freeze its last value into every subsequent row forever, which
    is the silently-dead-column failure this recorder exists to catch. Past
    STALE_S the cells go empty and the child is respawned with backoff.
    """

    STALE_S = 3.0
    FIELDS = ["gpu_util", "clocks_sm", "clocks_gr", "power_draw_w",
              "temp_gpu_c", "pstate", "clocks_event_reasons_active"]

    def __init__(self):
        self._lock = threading.Lock()
        self._values = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cmd = ["nvidia-smi",
                   "--query-gpu=" + ",".join(NVML_ALLOW),
                   "--format=csv,noheader,nounits", "-l", "1"]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1)
            except OSError:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            backoff = 1.0
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(NVML_ALLOW):
                    continue
                # `[N/A]` must never reach a cell as a value. It cannot happen
                # for the allowlist above, but if a future GB10 driver drops a
                # field the cell goes empty rather than carrying the string.
                parts = ["" if p in ("[N/A]", "N/A", "") else _sanitize(p)
                         for p in parts]
                with self._lock:
                    self._values = parts
                    self._stamp = time.monotonic()
            try:
                self._proc.kill()
            except OSError:
                pass
            if not self._stop.is_set():
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)

    def sample(self):
        with self._lock:
            values, stamp = self._values, self._stamp
        if values is None or (time.monotonic() - stamp) > self.STALE_S:
            return {}
        return dict(zip(self.FIELDS, values))


# ---------------------------------------------------------------------------
# The append-only writer.
# ---------------------------------------------------------------------------

class BlackBox:
    """Append-only CSV, one unbuffered os.write per row, daily rotation.

    On appending to an existing file the header is compared to the current
    schema. A mismatch starts a suffixed sibling file rather than writing rows
    whose columns no longer mean what the header says.
    """

    def __init__(self, csv_dir, host):
        self.csv_dir = csv_dir
        self.host = host
        self.fd = None
        self.day = None
        self.path = None
        self.header = ",".join(COLUMNS) + "\n"

    def _open_for(self, day):
        os.makedirs(self.csv_dir, exist_ok=True)
        # The schema version is in the FILENAME, so a schema change lands in a
        # visibly different file instead of a `.v1` suffix whose meaning a
        # reader has to guess. The header comparison below stays as a
        # belt-and-braces guard for an edit that forgets to bump the version.
        base = "pqteld-%s-%s.s%d" % (self.host, day, SCHEMA_VERSION)
        for attempt in range(100):
            suffix = "" if attempt == 0 else ".v%d" % attempt
            path = os.path.join(self.csv_dir, base + suffix + ".csv")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    with open(path, "r") as fh:
                        existing = fh.readline()
                except OSError:
                    continue
                if existing != self.header:
                    continue  # schema drift: try the next suffix
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                return fd, path
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            if os.lseek(fd, 0, os.SEEK_END) == 0:
                os.write(fd, self.header.encode())
            return fd, path
        raise RuntimeError("could not open a CSV for %s" % base)

    def write(self, row):
        day = time.strftime("%Y%m%d", time.localtime())
        if day != self.day:
            if self.fd is not None:
                os.close(self.fd)
            self.fd, self.path = self._open_for(day)
            self.day = day
        line = ",".join(row.get(col, "") for col in COLUMNS) + "\n"
        # One write() per row. O_APPEND makes the offset update atomic, and
        # the bytes are page-cache resident on return, so the row survives
        # SIGKILL and the OOM killer.
        os.write(self.fd, line.encode())

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# ---------------------------------------------------------------------------
# Derived columns.
# ---------------------------------------------------------------------------

def least_squares_slope_gbs(history):
    """Signed d(MemAvailable)/dt in GB/s over the trailing window.

    Negative means memory is being consumed. Returns None when fewer than
    three samples span the window -- an under-determined slope is not measured,
    so its cell stays empty.
    """
    if len(history) < 3:
        return None
    t0 = history[-1][0]
    pts = [(t - t0, kb) for t, kb in history if t0 - t <= SLOPE_WINDOW_S]
    if len(pts) < 3:
        return None
    n = float(len(pts))
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return None
    slope_kb_per_s = (n * sxy - sx * sy) / denom
    return slope_kb_per_s / (1000.0 * 1000.0)  # kB/s -> GB/s


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    interval = 0.5          # 2 Hz
    slow_every = 10         # 0.2 Hz -> every 10 ticks
    csv_dir = CSV_DIR
    host = os.uname().nodename
    for i, arg in enumerate(argv):
        if arg == "--csv-dir" and i + 1 < len(argv):
            csv_dir = argv[i + 1]
        elif arg == "--interval" and i + 1 < len(argv):
            interval = float(argv[i + 1])
        elif arg == "--once":
            slow_every = 1

    once = "--once" in argv
    box = BlackBox(csv_dir, host)
    nvml = NvmlStream()
    nvml.start()
    # Wait briefly for the first NVML line so row 1 is not empty for no
    # reason. Bounded: if nvidia-smi never answers, the columns stay empty and
    # the recorder still runs -- /proc telemetry does not depend on the GPU.
    warm_deadline = time.monotonic() + 2.0
    while not nvml.sample() and time.monotonic() < warm_deadline:
        time.sleep(0.05)
    boot_id = read_boot_id()

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    history = []          # (monotonic_s, MemAvailable_kB) for the slope
    slow_cache = {}
    tick = 0
    last_mono = None
    next_at = time.monotonic()
    try:
        while not stop.is_set():
            now_mono = time.monotonic()
            row = {"epoch_ms": "%d" % (time.time() * 1000), "boot_id": boot_id}
            if last_mono is not None:
                row["dt_ms"] = "%.1f" % ((now_mono - last_mono) * 1000.0)
            last_mono = now_mono

            mem = read_meminfo()
            row.update(mem)
            row.update(read_pressure())
            row.update(read_vmstat())

            if all(k in mem for k in ["MemTotal"] + UVM_SUBTRAHENDS):
                residual = int(mem["MemTotal"]) - sum(
                    int(mem[k]) for k in UVM_SUBTRAHENDS)
                row["uvm_residual_kb"] = str(residual)
            if "MemAvailable" in mem:
                history.append((now_mono, int(mem["MemAvailable"])))
                cutoff = now_mono - SLOPE_WINDOW_S
                while history and history[0][0] < cutoff:
                    history.pop(0)
                slope = least_squares_slope_gbs(history)
                if slope is not None:
                    row["mem_slope_gbs"] = "%.4f" % slope

            row.update(nvml.sample())

            if tick % slow_every == 0:
                slow_cache = {}
                slow_cache.update(read_guard_state())
                slow_cache.update(read_cgroup_slices())
                slow_cache.update(read_top_processes())
                row.update(slow_cache)
            # Rows between slow ticks leave those cells EMPTY rather than
            # repeating a stale value: empty means "not measured on this row".

            box.write(row)
            tick += 1
            if once:
                break
            next_at += interval
            delay = next_at - time.monotonic()
            if delay < 0:
                next_at = time.monotonic()   # we fell behind; do not burn down
                delay = 0
            stop.wait(delay)
    finally:
        nvml.stop()
        box.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
