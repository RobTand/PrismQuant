#!/usr/bin/python3
"""Stimulus test for pqteld: prove no column is silently dead.

This hardware specializes in the silent dead column. `nvidia-smi` returns the
string `[N/A]` and exits 0 rather than erroring, so a recorder that queries a
non-existent field writes a column that looks healthy from every angle except
a stimulus test. The same is true of a wedged NVML reader freezing its last
value, and of a /proc key that a kernel upgrade renamed.

The method: apply named stimuli, then check that each column MOVED across the
run rather than merely being non-empty. A column that is constant across a
stimulus that should move it is a defect.

Five stimuli, because the columns fall into groups that no single stimulus
covers:

  anon   ~10 GB of touched anonymous memory. Moves MemFree, MemAvailable,
         AnonPages, mem_slope_gbs and the process table. Does NOT move
         uvm_residual_kb -- anon is one of the subtrahends, so it cancels out
         by construction. That is the correct behaviour, not a dead column.
  cuda   a ~3 GB CUDA allocation. This is what moves uvm_residual_kb, and it
         is the only thing that does: on GB10 there is no framebuffer field,
         so the residual is the GPU-memory signal.
  gpu    a short matmul loop (<10 s). Moves gpu_util, clocks, power, pstate
         and temperature.
  io     an O_DIRECT read of a multi-GB file, bypassing the page cache, so
         read_bytes moves and not only rchar.
  psi    a cgroup scope with MemoryHigh=64M touching a 64 MiB buffer. This
         is the only stimulus that moves psi_mem_full_*, because memory PSI
         counts reclaim STALL and a 10 GiB allocation into 116 GiB of free
         memory stalls on nothing. MemoryHigh throttles and forces reclaim;
         MemoryMax would OOM-kill instead, which would move oom_kill by
         manufacturing the incident the recorder exists to record. Do not.

This driver is a TEST HARNESS and may use the cu130 venv's torch for the CUDA
stimulus. The recorder itself remains stdlib-only; nothing here is imported by
it. Run:

    /usr/bin/python3 pqtel/stimulus_check.py --phase anon|io|psi
    <venv>/bin/python  pqtel/stimulus_check.py --phase cuda
    /usr/bin/python3 pqtel/stimulus_check.py --phase report --since-ms <T0>

where T0 is `date +%s%3N` captured before the first phase.
"""

import os
import subprocess
import sys
import time

CSV_DIR = "/home/rob/pqtel/csv"
BIG_FILE = "/home/rob/models/Qwen3-4B/model-00001-of-00003.safetensors"

# Columns that no safe stimulus can move, with the reason. These are checked
# for a parsable value, not for movement. Manufacturing an OOM to move
# oom_kill is not a test, it is an outage.
CONSTANT_BY_DESIGN = {
    "boot_id": "constant until reboot",
    "MemTotal": "constant for the life of the box",
    "SwapTotal": "0 -- no swap configured on this box",
    "SwapFree": "0 -- no swap configured on this box",
    "oom_kill": "incident-only counter; not provoked deliberately",
    # These three stayed 0 through the psi phase, which is consistent:
    # MemoryHigh drives cgroup/kswapd-style reclaim, not DIRECT reclaim,
    # and nothing here allocates into a genuinely exhausted zone. So they
    # are not "exercised and confirmed live" -- they did not move under
    # the bounded stimulus, and their parse is verified by inspection
    # against /proc/vmstat only. A real incident is what moves them.
    "allocstall_normal": "did not move under bounded stimulus; parse verified by inspection",
    "allocstall_movable": "did not move under bounded stimulus; parse verified by inspection",
    "pgsteal_direct": "did not move under bounded stimulus; parse verified by inspection",
    "guard_flag_present": "1 while the root-owned disarm flag exists",
    "guard_unit_active": "'active'; the unit is not touched by this test",
}


def phase_anon(seconds=14, gib=10):
    print("[anon] allocating %d GiB and touching every page" % gib)
    chunks = []
    for _ in range(gib):
        buf = bytearray(1 << 30)
        for off in range(0, len(buf), 4096):
            buf[off] = 1
        chunks.append(buf)
    print("[anon] holding for %ds" % seconds)
    time.sleep(seconds)
    del chunks
    print("[anon] released")


def phase_io(seconds=10):
    print("[io] O_DIRECT read of %s" % BIG_FILE)
    end = time.time() + seconds
    total = 0
    while time.time() < end:
        proc = subprocess.run(
            ["dd", "if=" + BIG_FILE, "of=/dev/null", "bs=8M", "count=200",
             "iflag=direct"], capture_output=True, text=True)
        if proc.returncode != 0:
            print("[io] dd failed: %s" % proc.stderr.strip()[:200])
            break
        total += 1
    print("[io] %d passes" % total)


def phase_psi(seconds=14):
    """Bounded memory-pressure stimulus, contained in a cgroup.

    MemoryHigh throttles the scope into reclaim; it does NOT OOM-kill. The
    system-wide /proc/pressure/memory counters rise because reclaim stall is
    aggregated across all tasks, so this exercises the recorder's parse of
    both psi_mem_full_avg10 and psi_mem_full_total without touching the box's
    real headroom.
    """
    print("[psi] cgroup scope, MemoryHigh=64M, thrashing for %ds" % seconds)
    before = open("/proc/pressure/memory").read().strip().replace("\n", " | ")
    print("[psi] before: %s" % before)
    # The deadline is checked INSIDE the touch loop, every 64 KiB, not only
    # between allocations. Under a hard MemoryHigh throttle each faulted page
    # waits on reclaim, so a coarser check does not bound the run at all:
    # checking only between 1 GiB buffers left a 14 s budget still running at
    # 104 s, and checking every 16 MiB left it running at 164 s.
    script = (
        "import time\n"
        "end = time.time() + %d\n"
        "buf = bytearray(64 << 20)\n"
        "while time.time() < end:\n"
        "    for off in range(0, len(buf), 4096):\n"
        "        buf[off] = (buf[off] + 1) & 0xFF\n"
        "        if (off & 0xFFFF) == 0 and time.time() > end:\n"
        "            break\n" % seconds)
    unit = "pqtel-psi-stimulus"
    try:
        proc = subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "--collect",
             "--unit", unit, "-p", "MemoryHigh=64M", "-p", "MemoryMax=2G",
             "/usr/bin/python3", "-c", script],
            capture_output=True, text=True, timeout=seconds + 60)
        if proc.returncode != 0:
            print("[psi] scope exited %d: %s"
                  % (proc.returncode, proc.stderr.strip()[:200]))
    except subprocess.TimeoutExpired:
        # Killing systemd-run does not kill the scope's child, and an orphan
        # that outlives its cgroup runs with no memory limit at all. Stop the
        # scope explicitly.
        print("[psi] timed out; stopping the scope")
        subprocess.run(["systemctl", "--user", "stop", unit + ".scope"],
                       capture_output=True)
    after = open("/proc/pressure/memory").read().strip().replace("\n", " | ")
    print("[psi] after:  %s" % after)
    print("[psi] %s" % next(
        l.strip() for l in open("/proc/vmstat") if l.startswith("oom_kill")))


def phase_cuda(seconds=9, gib=3):
    import torch
    print("[cuda] torch %s, device %s" % (torch.__version__,
                                          torch.cuda.get_device_name(0)))
    hold = torch.empty(int(gib * (1 << 30) // 2), dtype=torch.float16,
                       device="cuda")
    hold.fill_(1.0)
    torch.cuda.synchronize()
    print("[cuda] %d GiB resident; spinning matmuls for %ds" % (gib, seconds))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        for _ in range(50):
            a = a @ b
            a = a / a.abs().amax()
        torch.cuda.synchronize()
        n += 50
    print("[cuda] %d matmuls" % n)
    del hold, a, b
    torch.cuda.empty_cache()
    print("[cuda] released")


def _rows(path, since_ms):
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        out = []
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) != len(header) or not parts[0].isdigit():
                continue
            if int(parts[0]) >= since_ms:
                out.append(parts)
    return header, out


def report(since_ms):
    files = sorted(f for f in os.listdir(CSV_DIR) if f.endswith(".csv"))
    path = os.path.join(CSV_DIR, files[-1])
    header, rows = _rows(path, since_ms)
    print("stimulus window: %d rows from %s\n" % (len(rows), path))
    if not rows:
        print("NO ROWS -- recorder was not running")
        return 1
    width = max(len(h) for h in header)
    verdict_bad = []
    for i, col in enumerate(header):
        values = [r[i] for r in rows]
        present = [v for v in values if v != ""]
        distinct = sorted(set(present))
        if not present:
            print("  %-*s  DEAD        no value in any row" % (width, col))
            verdict_bad.append(col)
            continue
        moved = len(distinct) > 1
        # A slow-cadence column is empty on the rows between its ticks; that
        # is the "not measured" convention, not a gap in coverage.
        cadence = "" if len(present) == len(values) else \
            "  [%d/%d rows -- 0.2 Hz field]" % (len(present), len(values))
        if col in CONSTANT_BY_DESIGN and not moved:
            print("  %-*s  BY-DESIGN   %-22s  %s%s" % (
                width, col, distinct[0][:22], CONSTANT_BY_DESIGN[col], cadence))
        elif moved:
            numeric = all(_is_num(v) for v in present)
            if numeric:
                lo = min(float(v) for v in present)
                hi = max(float(v) for v in present)
                print("  %-*s  MOVED       %d distinct, %g .. %g%s" % (
                    width, col, len(distinct), lo, hi, cadence))
            else:
                print("  %-*s  MOVED       %d distinct: %s%s" % (
                    width, col, len(distinct),
                    ", ".join(d[:28] for d in distinct[:3]), cadence))
        else:
            print("  %-*s  CONSTANT    %-22s  <-- investigate%s" % (
                width, col, distinct[0][:22], cadence))
            verdict_bad.append(col)
    print()
    if verdict_bad:
        print("SUSPECT COLUMNS: %s" % ", ".join(verdict_bad))
        return 1
    print("every column either moved or is constant by design")
    return 0


def _is_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def main():
    argv = sys.argv[1:]
    phase = argv[argv.index("--phase") + 1] if "--phase" in argv else "report"
    if phase == "anon":
        phase_anon()
    elif phase == "io":
        phase_io()
    elif phase == "psi":
        phase_psi()
    elif phase == "cuda":
        phase_cuda()
    elif phase == "report":
        since = int(argv[argv.index("--since-ms") + 1])
        return report(since)
    else:
        print("unknown phase %r" % phase, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
