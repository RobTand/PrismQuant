#!/usr/bin/env python3
"""TP=2 decode feasibility harness for the 10 GbE two-DGX-Spark cluster.

The open question this tool settles: is tensor-parallel (TP=2) decode across
two DGX Spark boxes joined by plain 10 GbE TCP (no RDMA, no ConnectX) faster
or slower than single-box decode?

Decode is memory-bandwidth bound, so TP=2 roughly halves per-GPU weight reads.
The cost is 2*L all-reduces per token of hidden_size bf16 elements each.  Wire
time at those sizes is negligible on 10 GbE; LATENCY dominates because every
one of the 128 sync points (Qwen3.8-27B) stalls the token.  Crossover rule:

    TP=2 wins iff   net_overhead_per_token < TPOT_single / 2

The decisive unknown is per-all-reduce latency AT OUR MESSAGE SIZES ON THIS
LINK.  Three modes turn that into a one-command measurement:

  loopback   Two ranks on localhost.  Measures the SOFTWARE FLOOR (gloo +
             TCP stack + sync overhead) with zero wire time.  Every number it
             produces is a LOWER BOUND on the real two-box number, never an
             estimate of it.  Runs today on one box.
  netbench   Two ranks over the real link (run the same command on both boxes;
             see the banner it prints).  Small-message all-reduce latency at
             the exact sizes our models need, plus an iperf3 bandwidth leg
             when iperf3 exists on both boxes.
  decide     Apply a measured (or assumed) per-collective latency to the model
             specs below and print the WIN/LOSE/UNKNOWN verdict table.

Honesty contract (standing rule: never sell a screen as a result):
  * Output distinguishes MEASURED from ASSUMED everywhere latency enters.
  * Loopback numbers are labelled LOWER BOUND on every line they influence.
  * The caveats that can invalidate the arithmetic are printed by this tool,
    not left in a side document.

CPU/TCP only: gloo needs no GPU, so this harness is safe to run while another
job holds /home/rob/dq-runs/gpu.lock.  Scratch stays under /home/rob/dq-runs/;
paths under host /tmp are refused.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set

BF16_BYTES = 2
LINK_BPS_DEFAULT = 10e9  # 10 GbE TCP, no RDMA (verified: enP7s7 at 10000 Mb/s)

# Decode all-reduce payload sizes: bytes = hidden_size x bf16 for the models
# in MODEL_SPECS.  Swept exactly; do not pad or round these.
DEFAULT_SIZES_BYTES: tuple[int, ...] = (8192, 10240, 12288, 16384)

WARMUP_MIN = 200      # spec: >= 200 warmup iterations
ITERS_MIN = 2000      # spec: >= 2000 measured iterations
SYNC_EVERY_DEFAULT = 200   # re-barrier this often so skew cannot accumulate
PORT_DEFAULT = 29517       # rendezvous port; iperf3 leg uses port+1

SCHEMA = "prismaquant.tp_feasibility_result.v1"
SCRATCH_ROOT = Path("/home/rob/dq-runs")
DEFAULT_OUT_DIR = SCRATCH_ROOT / "ox-wave3-2026-08-23" / "tpbench"

# Verdict labels.  A verdict is NEVER printed when TPOT is unknown.
VERDICT_WIN = "WIN"
VERDICT_LOSE = "LOSE"
VERDICT_UNKNOWN = "UNKNOWN-NEEDS-TPOT"

# Provenance labels.  MEASURED vs ASSUMED is the load-bearing distinction.
LABEL_NETBENCH = "MEASURED / REAL-LINK (gloo over TCP)"
LABEL_LOOPBACK = "MEASURED / LOOPBACK LOWER BOUND (software floor only)"
LABEL_ASSUMED = "ASSUMED / user-supplied number"

# ---------------------------------------------------------------------------
# Model specs (the fixtures for the decide-mode arithmetic).
#
# hidden_size / layers are architectural constants.  tpot_ms_single is the
# MEASURED single-box TPOT of the shipped artifact on THIS box; None means no
# trusted measurement exists yet, and any model with None gets
# UNKNOWN-NEEDS-TPOT, never a WIN/LOSE.
#
# moe=True models route tokens through extra collectives beyond the 2*L
# all-reduces counted here (see CAVEATS).
MODEL_SPECS: dict[str, dict[str, Any]] = {
    # measured 2026-08 CB artifact, single DGX Spark (see tpbench report)
    "Qwen3.8-27B-CB":         {"hidden_size": 5120, "layers": 64, "tpot_ms_single": 64.4, "moe": False},
    "Qwen3.5-122B-A10B":      {"hidden_size": 6144, "layers": 80, "tpot_ms_single": None, "moe": True},
    "Mistral-Medium-3.5-128B": {"hidden_size": 6144, "layers": 88, "tpot_ms_single": None, "moe": False},
    "DSv4-Flash":             {"hidden_size": 4096, "layers": 43, "tpot_ms_single": None, "moe": False},
}

CAVEATS = [
    "(a) CB GEMV kernel time may NOT halve cleanly under TP=2: the codebook "
    "GEMV kernels are not purely bandwidth-bound, so per-GPU decode time can "
    "fall by less than 2x.",
    "(b) MoE routing adds collectives beyond the 2*L all-reduces counted "
    "here; the true sync count for MoE models is higher than tabulated.",
    "(c) Latency was measured with the gloo backend over TCP.  gloo is "
    "slower than NCCL, so a gloo number is PESSIMISTIC for a future "
    "NCCL-over-TCP deployment.",
    "(d) The crossover rule TPOT/2 assumes the memory-bound part of decode "
    "scales ideally across boxes; wire-time assumes exclusive use of the "
    "10 GbE link.",
]


# ---------------------------------------------------------------------------
# Pure arithmetic (unit-tested; no torch.distributed needed).


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' semantics) of q in [0,100]."""
    if not values:
        raise ValueError("percentile() of an empty sample")
    vs = sorted(float(v) for v in values)
    if len(vs) == 1:
        return vs[0]
    pos = (len(vs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)


def collective_math(
    hidden_size: int,
    layers: int,
    latency_us: float,
    link_bps: float = LINK_BPS_DEFAULT,
) -> Dict[str, Optional[float]]:
    """Per-token TP=2 collectives arithmetic for one model.

    Returns all-reduce count, byte counts, wire time at link_bps, and the
    added per-token latency implied by latency_us per collective.
    """
    all_reduces_per_token = 2 * layers
    bytes_per_allreduce = hidden_size * BF16_BYTES
    bytes_per_token = all_reduces_per_token * bytes_per_allreduce
    wire_ms = bytes_per_token * 8.0 / link_bps * 1000.0
    added_ms = all_reduces_per_token * latency_us / 1000.0
    return {
        "all_reduces_per_token": all_reduces_per_token,
        "bytes_per_allreduce": bytes_per_allreduce,
        "bytes_per_token": bytes_per_token,
        "wire_ms_at_link": wire_ms,
        "added_ms": added_ms,
        "latency_us": latency_us,
    }


def verdict_for(tpot_ms_single: Optional[float], added_ms: float) -> str:
    """WIN iff added per-token network latency beats the TPOT/2 budget.

    Strict inequality: equal-to-budget counts as LOSE (no free lunch).  A
    model without a measured single-box TPOT gets VERDICT_UNKNOWN, never a
    WIN/LOSE, regardless of the numbers.
    """
    if tpot_ms_single is None:
        return VERDICT_UNKNOWN
    return VERDICT_WIN if added_ms < tpot_ms_single / 2.0 else VERDICT_LOSE


def build_decision_rows(
    latency_us: Optional[float] = None,
    link_bps: float = LINK_BPS_DEFAULT,
) -> List[Dict[str, Any]]:
    """Decision rows for every model in MODEL_SPECS, stable order.

    When latency_us is given it prices every row identically (flat assumed
    number); when None the 'verdict' key is left None for the caller to fill
    from size-matched measured percentiles.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        m = collective_math(spec["hidden_size"], spec["layers"],
                            latency_us if latency_us is not None else 0.0, link_bps)
        tpot = spec["tpot_ms_single"]
        budget_ms = None if tpot is None else tpot / 2.0
        verdict = None
        if latency_us is not None:
            verdict = verdict_for(tpot, float(m["added_ms"]))
        rows.append({
            "model": name,
            "hidden_size": spec["hidden_size"],
            "layers": spec["layers"],
            "moe": spec["moe"],
            "tpot_ms_single": tpot,
            "budget_ms": budget_ms,
            "budget_us_per_allreduce": None if budget_ms is None
            else budget_ms * 1000.0 / m["all_reduces_per_token"],
            "verdict": verdict,
            **m,
        })
    return rows


def check_scratch_path(path: Path) -> Path:
    """Refuse host /tmp; everything scratch lives under /home/rob/dq-runs."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(Path("/tmp")):
        raise ValueError(f"refusing /tmp path ({resolved}); scratch belongs under {SCRATCH_ROOT}")
    return resolved


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _primary_ip(peer: Optional[str]) -> str:
    """Best-effort IPv4 this box would use to reach peer (UDP connect trick)."""
    target = peer or (socket.gethostbyname(socket.gethostname()))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 9))  # no packet is sent for UDP
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def _parse_cpu_pin(spec: Optional[str]) -> Optional[List[Set[int]]]:
    """'16-17,18-19' -> [{16,17}, {18,19}] (one core-set per rank).

    Ranges matter: each rank runs a main thread plus gloo device thread(s);
    forcing both onto one core serialises them against each other.
    """
    if not spec:
        return None
    ranks: List[Set[int]] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cores = set(range(int(lo), int(hi) + 1))
        else:
            cores = {int(part)}
        if not cores:
            raise SystemExit(f"--cpu-pin: empty core range '{part}'")
        ranks.append(cores)
    flat = [c for cs in ranks for c in cs]
    if len(flat) != len(set(flat)):
        raise SystemExit("--cpu-pin: a core appears in two ranks' sets")
    return ranks


def _load_snapshot() -> Dict[str, float]:
    """1/5/15-minute load averages, recorded so contention is provable later."""
    try:
        l1, l5, l15 = os.getloadavg()
        ncpu = os.cpu_count() or 1
        return {"load1": round(l1, 2), "load5": round(l5, 2),
                "load15": round(l15, 2), "cpus": ncpu}
    except OSError:  # pragma: no cover - getloadavg is universal on linux
        return {}


# ---------------------------------------------------------------------------
# Benchmark core.


def _init_pg(rank: int, world_size: int, init_method: str):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU/TCP only, ever
    import torch
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    return torch, dist


def _rank_bench(
    rank: int,
    world_size: int,
    init_method: str,
    sizes_bytes: Sequence[int],
    warmup: int,
    iters: int,
    sync_every: int,
    result_q=None,
    cpu_pin: Optional[Sequence[Set[int]]] = None,
) -> None:
    """One rank of the latency sweep.  Rank 0 pushes per-size stats to result_q."""
    if cpu_pin is not None:
        os.sched_setaffinity(0, set(cpu_pin[rank]))
    torch, dist = _init_pg(rank, world_size, init_method)
    try:
        for nbytes in sizes_bytes:
            numel = nbytes // BF16_BYTES
            t = torch.empty(numel, dtype=torch.bfloat16)
            t.normal_()
            for _ in range(warmup):
                dist.all_reduce(t)
            dist.barrier()
            lat_us: List[float] = []
            for i in range(iters):
                if sync_every > 0 and i % sync_every == 0:
                    dist.barrier()
                t0 = time.perf_counter_ns()
                dist.all_reduce(t)
                lat_us.append((time.perf_counter_ns() - t0) / 1000.0)
            dist.barrier()
            if rank == 0 and result_q is not None:
                result_q.put({
                    "bytes": nbytes,
                    "hidden_size": numel,
                    "p50_us": percentile(lat_us, 50),
                    "p90_us": percentile(lat_us, 90),
                    "p99_us": percentile(lat_us, 99),
                    "mean_us": sum(lat_us) / len(lat_us),
                    "min_us": min(lat_us),
                    "max_us": max(lat_us),
                    "n": len(lat_us),
                })
    finally:
        dist.destroy_process_group()


def _spawn_loopback(args: argparse.Namespace, sizes_bytes: Sequence[int],
                    warmup: int, iters: int, sync_every: int) -> List[Dict[str, Any]]:
    """Two ranks on localhost; returns rank-0's per-size stat dicts."""
    import multiprocessing as mp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    cpu_pin = _parse_cpu_pin(getattr(args, "cpu_pin", None))
    if cpu_pin is not None and len(cpu_pin) != 2:
        raise SystemExit("--cpu-pin takes one core or range per rank, e.g. '16-17,18-19'")
    procs = [
        ctx.Process(
            target=_rank_bench,
            args=(rank, 2, f"tcp://127.0.0.1:{port}",
                  sizes_bytes, warmup, iters, sync_every, result_q, cpu_pin),
        )
        for rank in (0, 1)
    ]
    for p in procs:
        p.start()
    results = [result_q.get(timeout=max(60, (warmup + iters) // 50)) for _ in sizes_bytes]
    for p in procs:
        p.join(timeout=120)
        if p.exitcode not in (0, None):
            raise RuntimeError(f"loopback rank exited with {p.exitcode}")
    return sorted(results, key=lambda r: r["bytes"])


def _iperf3_leg(rank: int, dist: Any, peer: Optional[str], port: int, seconds: int) -> Optional[Dict[str, Any]]:
    """Cross-box iperf3 bandwidth leg, orchestrated over the live process group.

    Rank 0 serves, rank 1 clients.  Skips (returns None) when either box lacks
    the binary or anything goes wrong; never raises into the latency results.
    """
    import torch

    def _which() -> bool:
        return subprocess.run(["which", "iperf3"], capture_output=True).returncode == 0

    have = torch.tensor([1.0 if _which() else 0.0])
    dist.all_reduce(have, op=dist.ReduceOp.MIN)
    if int(have.item()) != 1:
        if rank == 0:
            print("iperf3 bandwidth leg: SKIPPED (iperf3 not present on both boxes)")
        return None
    iport = port + 1
    server = None
    try:
        if rank == 0:
            server = subprocess.Popen(
                ["iperf3", "-s", "-1", "-p", str(iport)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.7)
        go = torch.zeros(1, dtype=torch.int64)
        dist.broadcast(go, src=0)  # go signal after server is listening
        if rank == 1:
            assert peer, "--peer (the rank-0 box) is required for the iperf3 leg"
            proc = subprocess.run(
                ["iperf3", "-c", peer, "-p", str(iport), "-t", str(seconds), "-J"],
                capture_output=True, text=True, timeout=seconds + 60)
            report = json.loads(proc.stdout)
            gbps = report["end"]["sum_received"]["bits_per_second"] / 1e9
            dist.broadcast(torch.tensor([gbps]), src=1)
        else:
            gbps = float(dist.broadcast(torch.zeros(1), src=1).item())
        if rank == 0:
            print(f"iperf3 bandwidth leg: MEASURED {gbps:.2f} Gbit/s "
                  f"(tcp, {seconds}s, rank1->rank0 receive)")
        return {"ran": True, "gbits_s": round(gbps, 3), "seconds": seconds}
    except Exception as exc:  # noqa: BLE001 - this leg is strictly optional
        if rank == 0:
            print(f"iperf3 bandwidth leg: SKIPPED ({exc})")
        return None
    finally:
        if server is not None and server.poll() is None:
            server.terminate()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"wrote {path}")


def _bench_provenance(mode: str, label: str, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "label": label,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "peer": getattr(args, "peer", None),
        "git_sha": _git_sha(),
        "argv": sys.argv,
        "backend": "gloo",
        "transport": "TCP",
        "load": _load_snapshot(),
        "cpu_pin": getattr(args, "cpu_pin", None),
        "note": ("loopback numbers are a LOWER BOUND on the real two-box link, "
                 "not an estimate of it") if mode == "loopback" else None,
    }


def _print_latency_table(results: List[Dict[str, Any]], label: str, lower_bound: bool) -> None:
    tag = " [LOOPBACK LOWER BOUND]" if lower_bound else ""
    print(f"\nper-collective latency ({label}){tag}:")
    print(f"{'bytes':>7} {'H':>6} {'p50 us':>10} {'p90 us':>10} {'p99 us':>10} "
          f"{'mean us':>10} {'min us':>10} {'n':>6}")
    for r in results:
        print(f"{r['bytes']:7d} {r['hidden_size']:6d} {r['p50_us']:10.1f} "
              f"{r['p90_us']:10.1f} {r['p99_us']:10.1f} {r['mean_us']:10.1f} "
              f"{r['min_us']:10.1f} {r['n']:6d}")
    if lower_bound:
        print("NOTE: every number above is a software-floor LOWER BOUND with zero "
              "wire time; the real 10 GbE cross-box number will be higher.")


# ---------------------------------------------------------------------------
# Modes.


def mode_loopback(args: argparse.Namespace) -> int:
    sizes = _resolve_sizes(args)
    warmup, iters, sync_every = _resolve_iters(args)
    print("=" * 78)
    print("MODE: loopback -- two ranks on LOCALHOST (127.0.0.1)")
    print("This measures the SOFTWARE FLOOR: gloo + TCP loopback + Python sync")
    print("overhead, with ZERO wire time.  It is a LOWER BOUND on the real")
    print("two-box number over 10 GbE -- NOT an estimate of that number.")
    print("=" * 78)
    results = _spawn_loopback(args, sizes, warmup, iters, sync_every)
    _print_latency_table(results, LABEL_LOOPBACK, lower_bound=True)
    load = _load_snapshot()
    if load:
        print(f"system context at measurement time: load {load['load1']}/{load['load5']} "
              f"on {load['cpus']} cpus (contention inflates these numbers)")
        print(f"cpu pin: {args.cpu_pin or 'none'}")
    print("iperf3 bandwidth leg: N/A in loopback mode (there is no wire)")
    payload = _bench_provenance("loopback", LABEL_LOOPBACK, args)
    payload.update({
        "config": {"world_size": 2, "warmup": warmup, "iters": iters,
                   "sync_every": sync_every, "sizes_bytes": list(sizes)},
        "results": results,
        "iperf3": None,
    })
    _write_json(check_scratch_path(args.json_out), payload)
    _print_caveats()
    return 0


def mode_netbench(args: argparse.Namespace) -> int:
    if not args.peer:
        print("error: --mode netbench requires --peer HOST (the other box)\n",
              file=sys.stderr)
        return 2
    sizes = _resolve_sizes(args)
    warmup, iters, sync_every = _resolve_iters(args)
    print("=" * 78)
    print("MODE: netbench -- REAL LINK, one process per box (gloo over TCP)")
    print("Protocol: run this EXACT command on BOTH boxes, differing only in")
    print(f"  --rank : this invocation is rank {args.rank}.")
    print(f"  Rendezvous: tcp://{args.peer}:{args.port} "
          f"(iperf3 leg, if present, uses port {args.port + 1}).")
    print("=" * 78)
    if args.rank == 0:
        store_host = _primary_ip(args.peer)
        init_method = f"tcp://{store_host}:{args.port}"
        print(f"rank 0 hosting rendezvous on {store_host}:{args.port}, waiting for rank 1 ...")
    else:
        init_method = f"tcp://{args.peer}:{args.port}"

    cpu_pin = _parse_cpu_pin(args.cpu_pin)
    if cpu_pin is not None:
        if len(cpu_pin) != 2:
            raise SystemExit("--cpu-pin takes one core or range per rank, e.g. '16-17,18-19'")
        os.sched_setaffinity(0, cpu_pin[args.rank])
    _, dist = _init_pg(args.rank, 2, init_method)
    results: List[Dict[str, Any]] = []
    try:
        for nbytes in sizes:
            numel = nbytes // BF16_BYTES
            t = torch.empty(numel, dtype=torch.bfloat16)
            t.normal_()
            for _ in range(warmup):
                dist.all_reduce(t)
            dist.barrier()
            lat_us: List[float] = []
            for i in range(iters):
                if sync_every > 0 and i % sync_every == 0:
                    dist.barrier()
                t0 = time.perf_counter_ns()
                dist.all_reduce(t)
                lat_us.append((time.perf_counter_ns() - t0) / 1000.0)
            dist.barrier()
            if args.rank == 0:
                results.append({
                    "bytes": nbytes,
                    "hidden_size": numel,
                    "p50_us": percentile(lat_us, 50),
                    "p90_us": percentile(lat_us, 90),
                    "p99_us": percentile(lat_us, 99),
                    "mean_us": sum(lat_us) / len(lat_us),
                    "min_us": min(lat_us),
                    "max_us": max(lat_us),
                    "n": len(lat_us),
                })
        iperf = None
        if args.rank == 0:
            print("\nlatency sweep done; running optional bandwidth leg ...")
        iperf = _iperf3_leg(args.rank, dist, args.peer, args.port, args.iperf3_time)
    finally:
        dist.destroy_process_group()

    if args.rank == 0:
        _print_latency_table(results, LABEL_NETBENCH, lower_bound=False)
        payload = _bench_provenance("netbench", LABEL_NETBENCH, args)
        payload.update({
            "config": {"world_size": 2, "warmup": warmup, "iters": iters,
                       "sync_every": sync_every, "sizes_bytes": list(sizes)},
            "results": results,
            "iperf3": iperf,
        })
        _write_json(check_scratch_path(args.json_out), payload)
        _print_caveats()
    return 0


def _print_caveats() -> None:
    print("\nCAVEATS that can invalidate the WIN/LOSE arithmetic:")
    for c in CAVEATS:
        print(f"  {c}")


def mode_decide(args: argparse.Namespace) -> int:
    per_size, flat, label = _resolve_decide_latency(args)
    lower_bound = label is LABEL_LOOPBACK
    print("=" * 78)
    print("DECIDE: TP=2-vs-single-box decode crossover (rule: TP=2 wins iff")
    print("        net overhead per token < TPOT_single / 2)")
    print(f"latency source: {label}")
    if lower_bound:
        print("WARNING: loopback latency is a LOWER BOUND.  Verdicts below can only")
        print("be upgraded to final by --mode netbench across the two real boxes.")
    print("=" * 78)

    rows = build_decision_rows(link_bps=args.link_gbps * 1e9)
    hdr = (f"{'model':<26} {'AR/tok':>7} {'KB/tok':>9} {'wire ms':>8} "
           f"{'add p50 ms':>11} {'add p99 ms':>11} {'TPOT ms':>9} "
           f"{'budget ms':>10} {'us/AR':>8} {'verdict':>18}")
    print(hdr)
    print("-" * len(hdr))
    unknown_settle: List[str] = []
    for r in rows:
        p50_us, p99_us, note = _latency_for_row(r, per_size, flat)
        ar = int(r["all_reduces_per_token"])
        add_p50_ms = ar * p50_us / 1000.0
        add_p99_ms = ar * p99_us / 1000.0
        verdict = verdict_for(r["tpot_ms_single"], add_p50_ms)
        tpot = "unknown" if r["tpot_ms_single"] is None else f"{r['tpot_ms_single']:.1f}"
        budget = "-" if r["budget_ms"] is None else f"{r['budget_ms']:.1f}"
        us_ar = "-" if r["budget_us_per_allreduce"] is None \
            else f"<{r['budget_us_per_allreduce']:.0f}"
        flag = ""
        if verdict == VERDICT_WIN and r["budget_ms"] is not None \
                and add_p99_ms >= r["budget_ms"]:
            flag = " (marginal at p99)"
        print(f"{r['model']:<26} {ar:>7d} {r['bytes_per_token'] / 1024.0:>9.1f} "
              f"{r['wire_ms_at_link']:>8.3f} "
              f"{add_p50_ms:>11.2f} {add_p99_ms:>11.2f} "
              f"{tpot:>9} {budget:>10} {us_ar:>8} {verdict:>18}{flag}")
        print(f"{'':26} latency {p50_us:.1f}/{p99_us:.1f} us p50/p99 -- {note}")
        if r["moe"]:
            print(f"{'':26} note: MoE -- routing adds collectives beyond "
                  f"{ar}/token (caveat b)")
        if verdict == VERDICT_UNKNOWN:
            unknown_settle.append(r["model"])
        elif lower_bound:
            print(f"{'':26} NOTE: this verdict rides on a LOOPBACK LOWER BOUND")

    if unknown_settle:
        print(f"\nUNKNOWN-NEEDS-TPOT ({len(unknown_settle)}): {', '.join(unknown_settle)}")
        print("What settles them: a measured single-box TPOT (ms/token) for each model")
        print("on THIS box, on the same calibration contract as the Qwen3.8-27B-CB")
        print("measurement; budget is then TPOT/2 spread over the tabulated AR count.")
    _print_caveats()
    if lower_bound:
        print("\nReminder: loopback = software floor, a LOWER BOUND only.  The cluster")
        print("decision is settled solely by netbench p50/p99 across the two boxes.")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing.


def _resolve_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    sizes = tuple(sorted(set(int(b) for b in args.sizes_bytes.split(","))))
    bad = [b for b in sizes if b % BF16_BYTES]
    if bad:
        raise SystemExit(f"--sizes-bytes entries must be bf16-even multiples: {bad}")
    return sizes


def _resolve_iters(args: argparse.Namespace) -> tuple[int, int, int]:
    warmup = max(WARMUP_MIN, args.warmup)
    iters = max(ITERS_MIN, args.iters)
    if (warmup, iters) != (args.warmup, args.iters):
        print(f"note: raised to spec minimums (warmup>={WARMUP_MIN}, iters>={ITERS_MIN})")
    return warmup, iters, max(0, args.sync_every)


def _resolve_decide_latency(
    args: argparse.Namespace,
) -> tuple[Optional[Dict[int, Dict[str, float]]], Optional[float], str]:
    """Resolve decide-mode latency inputs.

    Returns (per_size, flat, label):
      per_size  maps all-reduce payload bytes -> {"p50_us", "p99_us"} taken
                from a bench-mode JSON (MEASURED), or None.
      flat      a single user-supplied microseconds figure used for both the
                p50 and p99 columns (ASSUMED), or None.
    """
    if args.latency_json:
        payload = json.loads(Path(args.latency_json).read_text(encoding="ascii"))
        per_size = {
            int(r["bytes"]): {"p50_us": float(r["p50_us"]), "p99_us": float(r["p99_us"])}
            for r in payload["results"]
        }
        raw = str(payload.get("label", ""))
        label = LABEL_LOOPBACK if "LOOPBACK" in raw else (
            LABEL_NETBENCH if raw.startswith("MEASURED") else LABEL_ASSUMED)
        return per_size, None, label
    if args.latency_us is not None:
        return None, float(args.latency_us), LABEL_ASSUMED
    raise SystemExit("decide needs --latency-us N (assumed) or --latency-json FILE (measured)")


def _latency_for_row(
    row: Dict[str, Any],
    per_size: Optional[Dict[int, Dict[str, float]]],
    flat: Optional[float],
) -> tuple[float, float, str]:
    """(p50_us, p99_us, provenance-note) for one decision row.

    MEASURED beats ASSUMED: when a bench JSON is present, each model is priced
    at its OWN all-reduce size (bytes = hidden_size x bf16); models whose exact
    size was not swept fall back to the largest swept size, flagged as such.
    """
    want = row["hidden_size"] * BF16_BYTES
    if per_size:
        if want in per_size:
            s = per_size[want]
            return s["p50_us"], s["p99_us"], "matched size"
        biggest = max(per_size)
        s = per_size[biggest]
        return s["p50_us"], s["p99_us"], f"fallback: largest swept size ({biggest}B)"
    assert flat is not None
    return flat, flat, "single assumed number (used for both p50 and p99)"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tp_decode_feasibility.py",
        description="TP=2 decode feasibility across a 10 GbE two-DGX-Spark cluster "
                    "(gloo/CPU/TCP only -- safe while gpu.lock is held).")
    ap.add_argument("--mode", required=True, choices=["netbench", "loopback", "decide"])
    ap.add_argument("--peer", help="other box's address (netbench: rank 1 connects here; "
                                   "used to pick rank 0's interface)")
    ap.add_argument("--rank", type=int, default=0, choices=[0, 1],
                    help="netbench only: this process's rank (default 0)")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT,
                    help=f"rendezvous port (default {PORT_DEFAULT}; iperf3 leg uses port+1)")
    ap.add_argument("--warmup", type=int, default=WARMUP_MIN,
                    help=f"warmup iterations per size (spec floor {WARMUP_MIN})")
    ap.add_argument("--iters", type=int, default=ITERS_MIN,
                    help=f"measured iterations per size (spec floor {ITERS_MIN})")
    ap.add_argument("--sync-every", type=int, default=SYNC_EVERY_DEFAULT,
                    help="re-barrier this often during measurement (0 disables; "
                         f"default {SYNC_EVERY_DEFAULT})")
    ap.add_argument("--sizes-bytes", default=",".join(str(b) for b in DEFAULT_SIZES_BYTES),
                    help="comma-separated all-reduce payload sizes in bf16 bytes "
                         f"(default: {','.join(map(str, DEFAULT_SIZES_BYTES))} "
                         "= H x 2 for H in 4096/5120/6144/8192)")
    ap.add_argument("--cpu-pin", default=None,
                    help="pin ranks to cores to cut scheduler noise, one core or "
                         "range per rank, e.g. '16-17,18-19' (loopback: rank0 cores, "
                         "rank1 cores; netbench: this rank's set)")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_OUT_DIR / "last_bench.json",
                    help=f"where bench modes write their result JSON (default under {DEFAULT_OUT_DIR})")
    # decide-mode inputs
    ap.add_argument("--latency-us", type=float, default=None,
                    help="ASSUMED per-all-reduce latency in microseconds")
    ap.add_argument("--latency-json", default=None,
                    help="MEASURED latency: JSON written by a bench mode")
    ap.add_argument("--latency-stat", choices=["p50_us", "p90_us", "p99_us"], default="p50_us",
                    help="which percentile to take from --latency-json (default p50_us)")
    ap.add_argument("--link-gbps", type=float, default=10.0,
                    help="link rate for wire-time accounting (default 10)")
    ap.add_argument("--iperf3-time", type=int, default=5,
                    help="seconds for the optional iperf3 leg (netbench only)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "loopback":
        return mode_loopback(args)
    if args.mode == "netbench":
        return mode_netbench(args)
    return mode_decide(args)


if __name__ == "__main__":
    raise SystemExit(main())
