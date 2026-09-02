#!/usr/bin/env python3
"""Collect, rather than hand-author, one DSpark matched-performance arm.

The release workflow has three explicit steps:

``declare-policy``
    Freeze the strict non-regression/headroom policy before either serve.

``start-sampler``
    Run inside the future serving container *before* vLLM starts.  A detached
    stdlib-only child samples ``/proc/meminfo:MemAvailable`` every second and
    records the container cgroup's OOM counters.

``collect-arm``
    After READY, adopt that sampler, create the pre/post live manifests, run
    one warm and one measured fixed 8x128 workload, reconcile Prometheus,
    snapshot and parse the serve log (graph/KV/routes), and publish one
    self-digesting report only after the validator replays it.

This module never launches vLLM and never accepts response, counter, memory,
KV, graph, route, or manifest JSON from an operator.  Those facts are observed
from the live server and kernel log by this process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .dspark_matched_performance import (
    ARMS,
    EXPECTED_OUTPUT_TOKENS,
    KV_CACHE_MEMORY_BYTES,
    KV_CAPACITY_SCHEMA,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MEMORY_SAMPLE_INTERVAL_MS,
    MEMORY_SCHEMA,
    MODEL_LEN,
    MTP_ARM,
    MTP_BINDING_SCHEMA,
    NO_MTP_ARM,
    REPORT_SCHEMA,
    START_MEM_AVAILABLE_FLOOR_BYTES,
    WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES,
    WALL_CHRONOLOGY_ABS_TOLERANCE_SECONDS,
    WORKLOAD_SCHEMA,
    canonical_sha256,
    collector_tool_identity,
    file_sha256,
    release_policy,
    validate_arm_report,
)
from .dspark_serving_profile import (
    DSPARK_IMAGE,
    DSPARK_SERVER_ENV_ALLOWLIST,
    collect_route_census,
    load_runtime_evidence,
    validate_route_census,
)
from .validate_dspark_target_draft import (
    DSPARK_ACCEPTANCE_SCHEMA,
    DSPARK_EXPECTED_GENERATION_TOKENS,
    DSPARK_LAUNCH_SWITCHES,
    DSPARK_MAX_TOKENS,
    DSPARK_NUM_SPECULATIVE_TOKENS,
    DSPARK_POSITION_ZERO_MINIMUM,
    DSPARK_PROMPTS,
    DSPARK_PROMPT_CORPUS_SHA256,
    DSPARK_PROMPT_SHA256,
    DSPARK_SPECULATIVE_CONFIG,
    DSPARK_WORKLOAD_SCHEMA,
    _SNAPSHOT_KEYS,
    _canonical_base_url,
    _expected_launch_options,
    _expected_no_mtp_launch_options,
    _http_json,
    _http_text,
    _metric_snapshot,
    _served_model,
    validate_acceptance_suite,
    validate_dspark_graph_capture_log,
    validate_no_mtp_graph_capture_log,
)


POLICY_DECLARATION_SCHEMA = "prismaquant.dspark_performance_policy_declaration.v1"
SAMPLER_STATE_SCHEMA = "prismaquant.dspark_memory_sampler_state.v1"
SOURCE_SNAPSHOT_SCHEMA = "prismaquant.runtime_source_snapshot.v1"

_PHASES = ("startup", "ready", "warmup", "measured", "post")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PROM_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
_LABEL = re.compile(r'(?:^|,)([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"\\])*")')
_KV_CAPACITY = re.compile(
    r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for "
    r"([0-9,]+) tokens per request: ([0-9]+(?:\.[0-9]+)?)x"
)
_KV_RESERVATION = re.compile(
    r"reserved 1\.6 GiB memory for KV Cache as specified by "
    r"kv_cache_memory_bytes config"
)
_OOM_LOG_PATTERNS = (
    "torch.OutOfMemoryError",
    "CUDA out of memory.",
    "Memory cgroup out of memory",
    "oom-kill:",
)
_REQUEST_COUNTER_KEYS = {
    "completed_requests",
    "generation_tokens",
    "failed_requests",
    "timed_out_requests",
}


class DSparkCollectorError(RuntimeError):
    """The collector could not produce release-authoritative evidence."""


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime | None = None) -> str:
    return (value or _utc_now_dt()).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, where: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DSparkCollectorError(f"{where} is not canonical UTC time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DSparkCollectorError(f"{where} is not ISO-8601") from exc
    return parsed.astimezone(timezone.utc)


def _strict_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DSparkCollectorError(
            f"{source}: expected one real regular JSON file"
        )

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DSparkCollectorError(f"{source}: duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DSparkCollectorError(f"{source}: non-finite number {value}")
            ),
        )
    except DSparkCollectorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DSparkCollectorError(f"{source}: unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DSparkCollectorError(f"{source}: expected one JSON object")
    return payload


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_no_clobber(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    if not destination.is_absolute():
        raise DSparkCollectorError(f"output path must be absolute: {destination}")
    if destination.is_symlink():
        raise DSparkCollectorError(f"output path is a symlink: {destination}")
    if destination.resolve(strict=False) != destination:
        raise DSparkCollectorError(
            f"output path has a symlinked/non-canonical component: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise DSparkCollectorError(
            f"output parent is a symlink: {destination.parent}"
        )
    parent = destination.parent.resolve(strict=True)
    temporary = parent / f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise DSparkCollectorError(
                f"refusing to replace existing evidence: {destination}"
            ) from exc
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json_no_clobber(path: str | Path, payload: object) -> None:
    _atomic_no_clobber(path, _json_bytes(payload))


def _release_source_identity() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify this module is executing from one complete archived Git tree."""
    root = Path(__file__).resolve(strict=True).parents[1]
    manifest_path = root / ".prismaquant-runtime-snapshot.json"
    manifest = _strict_json(manifest_path)
    if (
        manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or _COMMIT.fullmatch(str(manifest.get("commit", ""))) is None
        or _COMMIT.fullmatch(str(manifest.get("tree", ""))) is None
        or _SHA256.fullmatch(str(manifest.get("closure_sha256", ""))) is None
    ):
        raise DSparkCollectorError(
            "collector must run from a valid content-addressed runtime snapshot"
        )
    from tools.prismaquant_runtime_snapshot import SnapshotError, verify_snapshot

    try:
        verified = verify_snapshot(
            root,
            expected_commit=str(manifest["commit"]),
            expected_tree=str(manifest["tree"]),
            expected_closure_sha256=str(manifest["closure_sha256"]),
        )
    except (OSError, SnapshotError) as exc:
        raise DSparkCollectorError(
            f"runtime source snapshot verification failed: {exc}"
        ) from exc
    compact = {
        "schema": verified["schema"],
        "commit": verified["commit"],
        "tree": verified["tree"],
        "closure_sha256": verified["closure_sha256"],
        "entry_count": verified["entry_count"],
    }
    tool = collector_tool_identity(source_root=root, git_commit=compact["commit"])
    return root, compact, tool


def _load_policy_declaration(
    path: str | Path,
    *,
    source_snapshot: Mapping[str, Any],
    tool: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    declaration = _strict_json(path)
    recorded = declaration.get("declaration_sha256")
    unstamped = dict(declaration)
    unstamped.pop("declaration_sha256", None)
    if (
        set(declaration)
        != {
            "schema",
            "declared_at",
            "policy",
            "source_snapshot",
            "tool",
            "declaration_sha256",
        }
        or declaration.get("schema") != POLICY_DECLARATION_SCHEMA
        or declaration.get("source_snapshot") != dict(source_snapshot)
        or declaration.get("tool") != dict(tool)
        or declaration.get("policy")
        != release_policy(predeclared_at=str(declaration.get("declared_at", "")))
        or recorded != canonical_sha256(unstamped)
    ):
        raise DSparkCollectorError("performance policy declaration is stale or foreign")
    return declaration, file_sha256(path)


def declare_policy(output: str | Path) -> dict[str, Any]:
    _root, source, tool = _release_source_identity()
    declared_at = _utc()
    payload: dict[str, Any] = {
        "schema": POLICY_DECLARATION_SCHEMA,
        "declared_at": declared_at,
        "policy": release_policy(predeclared_at=declared_at),
        "source_snapshot": source,
        "tool": tool,
    }
    payload["declaration_sha256"] = canonical_sha256(payload)
    _write_json_no_clobber(output, payload)
    return payload


def _read_mem_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    value = int(fields[1]) * 1024
                    if value > 0:
                        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise DSparkCollectorError("cannot read /proc/meminfo MemAvailable") from exc
    raise DSparkCollectorError("/proc/meminfo has no valid MemAvailable")


def _cgroup_memory_events() -> tuple[Path, dict[str, int]]:
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text(
            encoding="ascii"
        ).splitlines()
        matches = [line.split("::", 1)[1] for line in cgroup_lines if "::" in line]
        if len(matches) != 1:
            raise DSparkCollectorError("cannot resolve one cgroup-v2 membership")
        relative = matches[0].lstrip("/")
        path = (Path("/sys/fs/cgroup") / relative / "memory.events").resolve(
            strict=True
        )
        values: dict[str, int] = {}
        for line in path.read_text(encoding="ascii").splitlines():
            name, raw = line.split()
            values[name] = int(raw)
    except DSparkCollectorError:
        raise
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise DSparkCollectorError("cannot read cgroup-v2 memory.events") from exc
    if any(name not in values or values[name] < 0 for name in ("oom", "oom_kill")):
        raise DSparkCollectorError("cgroup memory.events lacks OOM counters")
    return path, {"oom": values["oom"], "oom_kill": values["oom_kill"]}


def _proc_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 2 :].split()
        value = int(tail[19])
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise DSparkCollectorError(f"cannot identify sampler pid {pid}") from exc
    if value <= 0:
        raise DSparkCollectorError("sampler process start time is invalid")
    return value


def _marker(state_dir: Path, name: str) -> Path:
    return state_dir / name


def _read_phase(state_dir: Path) -> str:
    path = _marker(state_dir, "phase")
    if path.is_symlink() or not path.is_file():
        raise DSparkCollectorError("memory sampler phase file is unsafe")
    try:
        phase = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise DSparkCollectorError("memory sampler phase is unreadable") from exc
    if phase not in _PHASES:
        raise DSparkCollectorError("memory sampler phase is invalid")
    return phase


def _wait_for(path: Path, *, timeout: float, detail: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise DSparkCollectorError(detail)


def _assert_server_not_started() -> None:
    from tools.serve_fingerprint import find_server_pids

    pids = find_server_pids()
    if pids:
        raise DSparkCollectorError(
            "memory sampler must start before vLLM; live server processes "
            f"already exist: {pids}"
        )


def start_sampler(
    *, state_dir: str | Path, policy_declaration_path: str | Path
) -> dict[str, Any]:
    root, source, tool = _release_source_identity()
    declaration, declaration_file_sha = _load_policy_declaration(
        policy_declaration_path, source_snapshot=source, tool=tool
    )
    _assert_server_not_started()
    initial_memory = _read_mem_available()
    if initial_memory < START_MEM_AVAILABLE_FLOOR_BYTES:
        raise DSparkCollectorError(
            "MemAvailable is below the 110-GiB pre-start floor; the sampler "
            "must start before vLLM"
        )
    events_path, events = _cgroup_memory_events()
    directory = Path(state_dir)
    if (
        not directory.is_absolute()
        or directory.is_symlink()
        or directory.resolve(strict=False) != directory
    ):
        raise DSparkCollectorError("sampler state path must be absolute/non-symlink")
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise DSparkCollectorError(
            f"refusing to reuse sampler state directory: {directory}"
        ) from exc
    resolved_directory = directory.resolve(strict=True)
    if resolved_directory != directory:
        raise DSparkCollectorError(
            "sampler state path has a symlinked/non-canonical component"
        )
    _atomic_no_clobber(_marker(directory, "phase"), b"startup\n")
    _atomic_no_clobber(_marker(directory, "samples.jsonl"), b"")
    ledger_stat = _marker(directory, "samples.jsonl").stat(follow_symlinks=False)
    policy_path = Path(policy_declaration_path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise DSparkCollectorError(
            "policy declaration path must be one real regular file"
        )
    policy_path = policy_path.resolve(strict=True)
    read_fd, write_fd = os.pipe()
    sampler_path = root / "tools" / "dspark_memavailable_sampler.py"
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(sampler_path),
                "--state-dir",
                str(directory),
                "--start-fd",
                str(read_fd),
                "--interval-ms",
                str(MEMORY_SAMPLE_INTERVAL_MS),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(read_fd,),
            start_new_session=True,
        )
    except OSError as exc:
        os.close(read_fd)
        os.close(write_fd)
        raise DSparkCollectorError(
            "cannot start the stdlib-only memory sampler"
        ) from exc
    os.close(read_fd)
    pid = process.pid
    try:
        state: dict[str, Any] = {
            "schema": SAMPLER_STATE_SCHEMA,
            "created_at": _utc(),
            "source_snapshot": source,
            "tool": tool,
            "policy_declaration": declaration,
            "policy_declaration_path": str(policy_path),
            "policy_declaration_file_sha256": declaration_file_sha,
            "sampler_pid": pid,
            "sampler_proc_start_ticks": _proc_start_ticks(pid),
            "sample_interval_ms": MEMORY_SAMPLE_INTERVAL_MS,
            "sample_ledger_device": ledger_stat.st_dev,
            "sample_ledger_inode": ledger_stat.st_ino,
            "cgroup_memory_events_path": str(events_path),
            "cgroup_memory_events_before": events,
            "initial_mem_available_bytes": initial_memory,
            "source_root_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
        }
        _write_json_no_clobber(_marker(directory, "state.json"), state)
        os.write(write_fd, b"1")
    finally:
        os.close(write_fd)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _marker(directory, "error.json").exists():
            break
        if _marker(directory, "ready").exists():
            break
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if _marker(directory, "error.json").exists():
        raise DSparkCollectorError(
            f"memory sampler failed: {_strict_json(_marker(directory, 'error.json'))}"
        )
    if not _marker(directory, "ready").exists():
        raise DSparkCollectorError(
            "memory sampler did not write its initial observation"
        )
    return state


def _load_sampler_state(state_dir: Path) -> dict[str, Any]:
    state = _strict_json(_marker(state_dir, "state.json"))
    expected = {
        "schema",
        "created_at",
        "source_snapshot",
        "tool",
        "policy_declaration",
        "policy_declaration_path",
        "policy_declaration_file_sha256",
        "sampler_pid",
        "sampler_proc_start_ticks",
        "sample_interval_ms",
        "sample_ledger_device",
        "sample_ledger_inode",
        "cgroup_memory_events_path",
        "cgroup_memory_events_before",
        "initial_mem_available_bytes",
        "source_root_sha256",
    }
    if set(state) != expected or state.get("schema") != SAMPLER_STATE_SCHEMA:
        raise DSparkCollectorError("memory sampler state schema/keys differ")
    _parse_utc(state.get("created_at"), where="memory sampler creation")
    initial_memory = state.get("initial_mem_available_bytes")
    before_events = state.get("cgroup_memory_events_before")
    if (
        isinstance(initial_memory, bool)
        or not isinstance(initial_memory, int)
        or initial_memory < START_MEM_AVAILABLE_FLOOR_BYTES
        or not isinstance(before_events, Mapping)
        or set(before_events) != {"oom", "oom_kill"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in before_events.values()
        )
        or not isinstance(state.get("cgroup_memory_events_path"), str)
        or not Path(str(state["cgroup_memory_events_path"])).is_absolute()
        or _SHA256.fullmatch(str(state.get("source_root_sha256", ""))) is None
        or not _SHA256.fullmatch(
            str(state.get("policy_declaration_file_sha256", ""))
        )
    ):
        raise DSparkCollectorError("memory sampler state values differ")
    pid = state.get("sampler_pid")
    ticks = state.get("sampler_proc_start_ticks")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or isinstance(ticks, bool)
        or not isinstance(ticks, int)
        or ticks <= 0
        or _proc_start_ticks(pid) != ticks
    ):
        raise DSparkCollectorError("memory sampler process identity changed")
    if state.get("sample_interval_ms") != MEMORY_SAMPLE_INTERVAL_MS:
        raise DSparkCollectorError("memory sampler interval differs")
    for name in ("sample_ledger_device", "sample_ledger_inode"):
        value = state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DSparkCollectorError("memory sample ledger identity is invalid")
    ledger = _marker(state_dir, "samples.jsonl")
    if ledger.is_symlink() or not ledger.is_file():
        raise DSparkCollectorError("memory sample ledger is absent or unsafe")
    ledger_stat = ledger.stat(follow_symlinks=False)
    if (
        state.get("sample_ledger_device") != ledger_stat.st_dev
        or state.get("sample_ledger_inode") != ledger_stat.st_ino
    ):
        raise DSparkCollectorError("memory sample ledger identity changed")
    return state


def _read_samples(state_dir: Path) -> list[dict[str, Any]]:
    path = _marker(state_dir, "samples.jsonl")
    if path.is_symlink() or not path.is_file():
        raise DSparkCollectorError("memory sample ledger is absent or unsafe")
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="ascii") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                lines = handle.read().splitlines()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeError) as exc:
        raise DSparkCollectorError("memory sample ledger is unreadable") from exc
    for index, raw in enumerate(lines):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DSparkCollectorError(
                f"memory sample {index} is malformed"
            ) from exc
        if (
            not isinstance(row, dict)
            or set(row)
            != {"sequence", "observed_at", "phase", "mem_available_bytes"}
            or row.get("sequence") != index
            or row.get("phase") not in _PHASES
            or isinstance(row.get("mem_available_bytes"), bool)
            or not isinstance(row.get("mem_available_bytes"), int)
            or int(row.get("mem_available_bytes")) <= 0
        ):
            raise DSparkCollectorError(f"memory sample {index} differs")
        _parse_utc(row["observed_at"], where=f"memory sample {index}")
        result.append(row)
    if not result:
        raise DSparkCollectorError("memory sample ledger is empty")
    return result


def _set_phase(state_dir: Path, phase: str) -> int:
    if phase not in _PHASES:
        raise DSparkCollectorError(f"unknown sampler phase {phase}")
    before = len(_read_samples(state_dir))
    phase_path = _marker(state_dir, "phase")
    if phase_path.is_symlink() or not phase_path.is_file():
        raise DSparkCollectorError("memory sampler phase file is unsafe")
    temporary = _marker(state_dir, f".phase-{os.getpid()}-{time.time_ns()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.write(descriptor, (phase + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, phase_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return before


def _wait_phase_sample(
    state_dir: Path,
    phase: str,
    *,
    after_sequence: int,
    not_before: datetime | None = None,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _marker(state_dir, "error.json").exists():
            raise DSparkCollectorError(
                f"memory sampler failed: {_strict_json(_marker(state_dir, 'error.json'))}"
            )
        rows = _read_samples(state_dir)
        for row in rows[after_sequence:]:
            observed = _parse_utc(row["observed_at"], where="memory phase sample")
            if row["phase"] == phase and (
                not_before is None or observed >= not_before
            ):
                return rows
        time.sleep(0.05)
    raise DSparkCollectorError(f"memory sampler did not observe phase {phase}")


def _stop_sampler(state_dir: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not _marker(state_dir, "stop").exists():
        _atomic_no_clobber(_marker(state_dir, "stop"), b"stop\n")
    _wait_for(
        _marker(state_dir, "stopped"),
        timeout=5.0,
        detail="memory sampler did not stop cleanly",
    )
    if _marker(state_dir, "error.json").exists():
        raise DSparkCollectorError(
            f"memory sampler failed: {_strict_json(_marker(state_dir, 'error.json'))}"
        )
    return _read_samples(state_dir)


def _prometheus_rows(text: str, metric: str) -> list[tuple[dict[str, str], float]]:
    rows: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        match = _PROM_LINE.fullmatch(line)
        if match is None or match.group("name") != metric:
            continue
        labels_raw = match.group("labels") or ""
        labels: dict[str, str] = {}
        position = 0
        for label in _LABEL.finditer(labels_raw):
            if label.start() != position or label.group(1) in labels:
                raise DSparkCollectorError(f"Prometheus {metric} labels are malformed")
            position = label.end()
            labels[label.group(1)] = json.loads(label.group(2))
        if position != len(labels_raw):
            raise DSparkCollectorError(f"Prometheus {metric} labels are malformed")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise DSparkCollectorError(f"Prometheus {metric} is not finite")
        rows.append((labels, value))
    if not rows:
        raise DSparkCollectorError(f"Prometheus metric {metric} is absent")
    return rows


def _integer_metric(value: float, *, where: str) -> int:
    if not value.is_integer() or value < 0:
        raise DSparkCollectorError(f"{where} is not a non-negative counter")
    return int(value)


def _request_counter_snapshot(base_url: str, served_model: str) -> dict[str, int]:
    text = _http_text(base_url, "/metrics")
    generation_rows = _prometheus_rows(text, "vllm:generation_tokens_total")
    generation = [
        value
        for labels, value in generation_rows
        if set(labels) == {"model_name", "engine"}
        and labels.get("model_name") == served_model
        and labels.get("engine") == "0"
    ]
    if len(generation) != 1 or len(generation_rows) != 1:
        raise DSparkCollectorError(
            "generation counter does not name exactly the measured model/engine"
        )
    request_rows = _prometheus_rows(text, "vllm:request_success_total")
    selected: dict[str, int] = {}
    for labels, value in request_rows:
        if (
            set(labels) != {"model_name", "engine", "finished_reason"}
            or labels.get("model_name") != served_model
            or labels.get("engine") != "0"
        ):
            raise DSparkCollectorError(
                "request counters include another model or engine"
            )
        reason = labels.get("finished_reason")
        if reason not in {"stop", "length", "abort", "error", "repetition"}:
            raise DSparkCollectorError(
                f"request counters include unknown finish reason {reason!r}"
            )
        if reason in selected:
            raise DSparkCollectorError("request finish-reason counter is duplicated")
        selected[str(reason)] = _integer_metric(value, where=f"request {reason}")
    for reason in ("stop", "length", "abort", "error", "repetition"):
        selected.setdefault(reason, 0)
    return {
        "completed_requests": sum(selected.values()),
        "generation_tokens": _integer_metric(generation[0], where="generation tokens"),
        "failed_requests": selected["error"],
        "timed_out_requests": selected["abort"],
    }


def _counter_ledger(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, Any]:
    if set(before) != _REQUEST_COUNTER_KEYS or set(after) != _REQUEST_COUNTER_KEYS:
        raise DSparkCollectorError("Prometheus counter ledger keys differ")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for snapshot in (before, after)
        for value in snapshot.values()
    ):
        raise DSparkCollectorError("Prometheus counter ledger is not integral")
    delta = {key: int(after[key]) - int(before[key]) for key in before}
    if any(value < 0 for value in delta.values()):
        raise DSparkCollectorError("Prometheus counters regressed")
    return {"before": dict(before), "after": dict(after), "delta": delta}


def _run_fixed_workload(base_url: str, served_model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(DSPARK_PROMPTS):
        response, elapsed = _http_json(
            base_url,
            "/v1/chat/completions",
            payload={
                "model": served_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": DSPARK_MAX_TOKENS,
                "ignore_eos": True,
                "stream": False,
            },
            timeout=600,
        )
        choices = response.get("choices")
        usage = response.get("usage")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], Mapping)
            or not isinstance(choices[0].get("message"), Mapping)
            or not isinstance(choices[0]["message"].get("content"), str)
            or not isinstance(usage, Mapping)
        ):
            raise DSparkCollectorError(f"response {index} is not one chat completion")
        content = str(choices[0]["message"]["content"])
        finish_reason = choices[0].get("finish_reason")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            finish_reason != "length"
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or completion_tokens != DSPARK_MAX_TOKENS
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed <= 0
        ):
            raise DSparkCollectorError(
                f"response {index} differs from the fixed 128-token contract"
            )
        rows.append(
            {
                "index": index,
                "prompt_sha256": DSPARK_PROMPT_SHA256[index],
                "content_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "request_seconds": elapsed,
            }
        )
    return rows


def _matched_workload() -> dict[str, Any]:
    return {
        "schema": WORKLOAD_SCHEMA,
        "prompt_count": len(DSPARK_PROMPTS),
        "prompt_sha256": list(DSPARK_PROMPT_SHA256),
        "prompt_corpus_sha256": DSPARK_PROMPT_CORPUS_SHA256,
        "max_tokens_per_prompt": DSPARK_MAX_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "warmup_repetitions": 1,
        "measured_repetitions": 1,
        "max_concurrency": 1,
    }


def _acceptance_from_observations(
    *,
    base_url: str,
    served_model: str,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    responses: Sequence[Mapping[str, Any]],
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, Any]:
    delta = {key: float(after[key]) - float(before[key]) for key in _SNAPSHOT_KEYS}
    drafts = delta["drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    positions = [
        delta[f"accepted_position_{index}"]
        for index in range(DSPARK_NUM_SPECULATIVE_TOKENS)
    ]
    result = {
        "schema": DSPARK_ACCEPTANCE_SCHEMA,
        "started_at": started_at,
        "finished_at": finished_at,
        "base_url": base_url,
        "served_model": served_model,
        "workload": {
            "schema": DSPARK_WORKLOAD_SCHEMA,
            "prompt_count": len(DSPARK_PROMPTS),
            "prompt_sha256": list(DSPARK_PROMPT_SHA256),
            "prompt_corpus_sha256": DSPARK_PROMPT_CORPUS_SHA256,
            "max_tokens_per_prompt": DSPARK_MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        },
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
        "wall_seconds": wall_seconds,
        "served_output_tokens_per_second": (
            DSPARK_EXPECTED_GENERATION_TOKENS / wall_seconds
        ),
        "responses": [dict(row) for row in responses],
        "acceptance": {
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "accepted_tokens_per_position": positions,
            "aggregate_rate": accepted / draft_tokens,
            "accepted_tokens_per_cycle": accepted / drafts,
            "mean_acceptance_length": 1.0 + accepted / drafts,
            "per_position_rates": [value / drafts for value in positions],
            "position_zero_gate_minimum": DSPARK_POSITION_ZERO_MINIMUM,
            "position_zero_gate_passed": (
                positions[0] / drafts >= DSPARK_POSITION_ZERO_MINIMUM
            ),
        },
    }
    return validate_acceptance_suite(result, expected_served_model=served_model)


def _collect_manifest(
    *,
    arm: str,
    base_url: str,
    artifact_dir: str | Path,
    draft_artifact_dir: str | Path | None,
    runtime_evidence_path: str | Path,
    phase: str,
) -> dict[str, Any]:
    from tools.serve_fingerprint import collect_manifest

    runtime = load_runtime_evidence(runtime_evidence_path, verify_files=True)
    extra = {
        "dspark_serving_profile": runtime["profile_receipt"],
        "dspark_runtime_evidence": runtime,
    }
    manifest = collect_manifest(
        pids=None,
        image=DSPARK_IMAGE,
        artifact_dir=artifact_dir,
        draft_artifact_dir=(draft_artifact_dir if arm == MTP_ARM else None),
        base_url=base_url,
        attestation_phase=phase,
        server_environment_names=DSPARK_SERVER_ENV_ALLOWLIST,
        extra=extra,
    )
    if not isinstance(manifest, dict):
        raise DSparkCollectorError("serve fingerprint did not return an object")
    return manifest


def _snapshot_log(source: str | Path, destination: str | Path) -> Path:
    path = Path(source)
    if path.is_symlink():
        raise DSparkCollectorError("serve log must be one real regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DSparkCollectorError("serve log is unreadable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise DSparkCollectorError(
                "serve log must be one non-empty regular file"
            )

        def read_prefix(size: int) -> bytes:
            chunks: list[bytes] = []
            offset = 0
            while offset < size:
                chunk = os.pread(descriptor, min(1 << 20, size - offset), offset)
                if not chunk:
                    raise DSparkCollectorError(
                        "serve log shrank while it was being snapshotted"
                    )
                chunks.append(chunk)
                offset += len(chunk)
            return b"".join(chunks)

        data = read_prefix(info.st_size)
        if read_prefix(info.st_size) != data:
            raise DSparkCollectorError(
                "serve log prefix changed while it was being snapshotted"
            )
        final_info = os.fstat(descriptor)
        if (
            final_info.st_dev != info.st_dev
            or final_info.st_ino != info.st_ino
            or final_info.st_size < info.st_size
        ):
            raise DSparkCollectorError(
                "serve log identity changed while it was being snapshotted"
            )
    finally:
        os.close(descriptor)
    if not data:
        raise DSparkCollectorError("serve log is empty")
    _atomic_no_clobber(destination, data)
    return Path(destination)


def _kv_capacity(log_path: Path, *, graph: Mapping[str, Any]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    reservations = _KV_RESERVATION.findall(text)
    capacities = _KV_CAPACITY.findall(text)
    if len(reservations) != 1 or len(capacities) != 1:
        raise DSparkCollectorError(
            "serve log does not contain one exact 1.6-GiB KV reservation/capacity"
        )
    capacity_raw, model_len_raw, concurrency_raw = capacities[0]
    capacity = int(capacity_raw.replace(",", ""))
    model_len = int(model_len_raw.replace(",", ""))
    concurrency = float(concurrency_raw)
    expected_concurrency = capacity / MODEL_LEN
    if (
        model_len != MODEL_LEN
        or capacity < MODEL_LEN
        or concurrency < 1.0
        or abs(concurrency - expected_concurrency) > 0.011
    ):
        raise DSparkCollectorError("serve log does not admit one full 256K request")
    return {
        "schema": KV_CAPACITY_SCHEMA,
        "dtype": "fp8",
        "requested_bytes": KV_CACHE_MEMORY_BYTES,
        "allocated_bytes": KV_CACHE_MEMORY_BYTES,
        "capacity_tokens": capacity,
        "max_model_len": MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "concurrency_at_max_model_len": concurrency,
        "profile_log_sha256": graph["serve_log_sha256"],
        "capacity_verified": True,
    }


def _memory_evidence(
    *,
    samples: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    graph: Mapping[str, Any],
    log_path: Path,
    server_alive_after: bool,
) -> dict[str, Any]:
    by_phase = {
        phase: [row for row in samples if row.get("phase") == phase]
        for phase in _PHASES
    }
    if any(not rows for rows in by_phase.values()):
        raise DSparkCollectorError("continuous memory ledger misses a required phase")
    values = [int(row["mem_available_bytes"]) for row in samples]
    start = values[0]
    ready = int(by_phase["ready"][0]["mem_available_bytes"])
    startup_min = min(
        int(row["mem_available_bytes"])
        for phase in ("startup", "ready")
        for row in by_phase[phase]
    )
    measured_min = min(
        int(row["mem_available_bytes"]) for row in by_phase["measured"]
    )
    post = int(by_phase["post"][0]["mem_available_bytes"])
    events_path, events_after = _cgroup_memory_events()
    if str(events_path) != state.get("cgroup_memory_events_path"):
        raise DSparkCollectorError("collector cgroup changed during the serve")
    before = state.get("cgroup_memory_events_before")
    if not isinstance(before, Mapping):
        raise DSparkCollectorError("sampler lacks initial cgroup OOM counters")
    oom_events = int(events_after["oom"]) - int(before.get("oom", -1))
    oom_kills = int(events_after["oom_kill"]) - int(before.get("oom_kill", -1))
    if oom_events < 0 or oom_kills < 0:
        raise DSparkCollectorError("cgroup OOM counters regressed")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_oom = any(pattern in log_text for pattern in _OOM_LOG_PATTERNS)
    minimum = min(values)
    return {
        "schema": MEMORY_SCHEMA,
        "memory_kind": "nvidia_gb10_unified",
        "sampler": "/proc/meminfo:MemAvailable",
        "sample_interval_ms": MEMORY_SAMPLE_INTERVAL_MS,
        "samples": [dict(row) for row in samples],
        "start_mem_available_bytes": start,
        "ready_mem_available_bytes": ready,
        "startup_min_mem_available_bytes": startup_min,
        "measured_min_mem_available_bytes": measured_min,
        "post_mem_available_bytes": post,
        "minimum_mem_available_bytes": minimum,
        "model_residency_bytes": max(0, start - ready),
        "startup_transient_bytes": max(0, ready - startup_min),
        "measured_transient_bytes": max(0, ready - measured_min),
        "watchdog_floor_bytes": WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES,
        "watchdog_tripped": minimum < WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES,
        "oom_events": oom_events,
        "oom_kill_detected": bool(oom_kills or log_oom),
        "server_alive_after": server_alive_after,
        "kv_cache": _kv_capacity(log_path, graph=graph),
    }


def _validate_live_manifest(
    manifest: Mapping[str, Any], *, arm: str, served_model: str
) -> None:
    from .dspark_serving_profile import validate_dspark_serve_manifest

    options = (
        _expected_launch_options(served_model)
        if arm == MTP_ARM
        else _expected_no_mtp_launch_options(served_model)
    )
    expected_speculative = DSPARK_SPECULATIVE_CONFIG if arm == MTP_ARM else None
    validate_dspark_serve_manifest(
        manifest,
        arm="graph",
        expected_served_model=served_model,
        requires_moe_marlin=True,
        expected_model_sha=str(manifest.get("artifact_binding", {}).get("model_sha")),
        expected_speculative_config=expected_speculative,
        expected_launch_options=options,
        expected_launch_switches=DSPARK_LAUNCH_SWITCHES,
    )


def collect_arm(
    *,
    arm: str,
    state_dir: str | Path,
    base_url: str,
    artifact_dir: str | Path,
    draft_artifact_dir: str | Path | None,
    runtime_evidence_path: str | Path,
    serve_log_path: str | Path,
    serve_log_snapshot_out: str | Path,
    pre_manifest_out: str | Path,
    post_manifest_out: str | Path,
    output_json: str | Path,
    acceptance_output_json: str | Path | None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise DSparkCollectorError(f"unknown performance arm {arm!r}")
    if (arm == MTP_ARM) != (acceptance_output_json is not None):
        raise DSparkCollectorError(
            "only the MTP arm requires one acceptance-output JSON"
        )
    if arm == MTP_ARM and draft_artifact_dir is None:
        raise DSparkCollectorError("MTP collection requires the draft artifact")
    if arm == NO_MTP_ARM and draft_artifact_dir is not None:
        raise DSparkCollectorError("target-only collection cannot bind a draft")
    if Path(artifact_dir) != Path("/model") or (
        arm == MTP_ARM and Path(str(draft_artifact_dir)) != Path("/draft")
    ):
        raise DSparkCollectorError(
            "matched collection requires exact /model and (for MTP) /draft mounts"
        )
    outputs = [
        Path(serve_log_snapshot_out),
        Path(pre_manifest_out),
        Path(post_manifest_out),
        Path(output_json),
    ] + ([Path(acceptance_output_json)] if acceptance_output_json else [])
    if any(not path.is_absolute() or ".." in path.parts for path in outputs):
        raise DSparkCollectorError(
            "collector output paths must be absolute and normalized"
        )
    canonical_outputs = [path.resolve(strict=False) for path in outputs]
    if any(
        path != canonical
        for path, canonical in zip(outputs, canonical_outputs)
    ):
        raise DSparkCollectorError(
            "collector output paths cannot contain symlinked components"
        )
    if len(set(canonical_outputs)) != len(outputs):
        raise DSparkCollectorError("collector output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise DSparkCollectorError("collector refuses an existing output path")

    root, source, tool = _release_source_identity()
    raw_directory = Path(state_dir)
    if not raw_directory.is_absolute() or raw_directory.is_symlink():
        raise DSparkCollectorError("sampler state directory is unsafe")
    directory = raw_directory.resolve(strict=True)
    if directory != raw_directory or not directory.is_dir():
        raise DSparkCollectorError("sampler state directory is unsafe")
    protected_roots = [root]
    for label, raw_path in (
        ("target artifact", artifact_dir),
        ("draft artifact", draft_artifact_dir),
    ):
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.is_symlink() or not path.is_dir():
            raise DSparkCollectorError(f"{label} must be one real directory")
        protected_roots.append(path.resolve(strict=True))
    for candidate in (directory, *outputs):
        resolved = candidate.resolve(strict=False)
        if any(
            resolved == protected or protected in resolved.parents
            for protected in protected_roots
        ):
            raise DSparkCollectorError(
                "sampler/evidence outputs cannot mutate source or model artifacts"
            )
    state = _load_sampler_state(directory)
    if state.get("source_snapshot") != source or state.get("tool") != tool:
        raise DSparkCollectorError("sampler and collector source identities differ")
    if state.get("source_root_sha256") != hashlib.sha256(str(root).encode()).hexdigest():
        raise DSparkCollectorError("sampler and collector snapshot roots differ")
    events_path, events_now = _cgroup_memory_events()
    events_before = state.get("cgroup_memory_events_before")
    if (
        str(events_path) != state.get("cgroup_memory_events_path")
        or not isinstance(events_before, Mapping)
        or events_now
        != {name: int(events_before[name]) for name in ("oom", "oom_kill")}
    ):
        raise DSparkCollectorError(
            "collector cgroup identity changed or an OOM preceded READY"
        )
    if _read_phase(directory) != "startup" or any(
        _marker(directory, name).exists()
        for name in ("claimed", "stop", "stopped")
    ):
        raise DSparkCollectorError("memory sampler is not one unused startup observer")
    declaration = state.get("policy_declaration")
    if not isinstance(declaration, Mapping):
        raise DSparkCollectorError("sampler policy declaration is missing")
    replayed_declaration, declaration_file_sha = _load_policy_declaration(
        str(state.get("policy_declaration_path", "")),
        source_snapshot=source,
        tool=tool,
    )
    if (
        replayed_declaration != dict(declaration)
        or declaration_file_sha
        != state.get("policy_declaration_file_sha256")
    ):
        raise DSparkCollectorError(
            "sampler policy declaration file changed after startup"
        )
    if _parse_utc(declaration.get("declared_at"), where="policy declaration") > _parse_utc(
        state.get("created_at"), where="sampler creation"
    ):
        raise DSparkCollectorError("performance policy was declared after sampling began")
    _atomic_no_clobber(_marker(directory, "claimed"), (arm + "\n").encode())

    base_url = _canonical_base_url(base_url)
    stopped = False
    try:
        served_model = _served_model(base_url)
        ready_from = _set_phase(directory, "ready")
        _wait_phase_sample(directory, "ready", after_sequence=ready_from)
        pre = _collect_manifest(
            arm=arm,
            base_url=base_url,
            artifact_dir=artifact_dir,
            draft_artifact_dir=draft_artifact_dir,
            runtime_evidence_path=runtime_evidence_path,
            phase="pre",
        )
        _validate_live_manifest(pre, arm=arm, served_model=served_model)
        _write_json_no_clobber(pre_manifest_out, pre)

        warm_from = _set_phase(directory, "warmup")
        _wait_phase_sample(directory, "warmup", after_sequence=warm_from)
        warmup = _run_fixed_workload(base_url, served_model)
        warm_finished = _utc_now_dt()
        warm_count = len(_read_samples(directory))
        _wait_phase_sample(
            directory,
            "warmup",
            after_sequence=warm_count,
            not_before=warm_finished,
        )

        measured_from = _set_phase(directory, "measured")
        _wait_phase_sample(directory, "measured", after_sequence=measured_from)
        before = _request_counter_snapshot(base_url, served_model)
        spec_before = _metric_snapshot(base_url) if arm == MTP_ARM else None
        started_dt = _utc_now_dt()
        started_mono = time.monotonic()
        responses = _run_fixed_workload(base_url, served_model)
        wall_seconds = time.monotonic() - started_mono
        finished_dt = _utc_now_dt()
        chronology = (finished_dt - started_dt).total_seconds()
        if (
            abs(chronology - wall_seconds)
            > WALL_CHRONOLOGY_ABS_TOLERANCE_SECONDS
        ):
            raise DSparkCollectorError(
                "measured monotonic and UTC wall intervals differ"
            )
        finished_at = _utc(finished_dt)
        started_at = _utc(started_dt)
        after = _request_counter_snapshot(base_url, served_model)
        spec_after = _metric_snapshot(base_url) if arm == MTP_ARM else None
        measured_count = len(_read_samples(directory))
        _wait_phase_sample(
            directory,
            "measured",
            after_sequence=measured_count,
            not_before=finished_dt,
        )

        post_from = _set_phase(directory, "post")
        _wait_phase_sample(directory, "post", after_sequence=post_from)
        if _served_model(base_url) != served_model:
            raise DSparkCollectorError("served model changed during collection")
        post = _collect_manifest(
            arm=arm,
            base_url=base_url,
            artifact_dir=artifact_dir,
            draft_artifact_dir=draft_artifact_dir,
            runtime_evidence_path=runtime_evidence_path,
            phase="post",
        )
        _validate_live_manifest(post, arm=arm, served_model=served_model)
        _write_json_no_clobber(post_manifest_out, post)
        log_snapshot = _snapshot_log(serve_log_path, serve_log_snapshot_out)
        graph = (
            validate_dspark_graph_capture_log(log_snapshot)
            if arm == MTP_ARM
            else validate_no_mtp_graph_capture_log(log_snapshot)
        )
        server_alive = _served_model(base_url) == served_model
        samples = _stop_sampler(directory, state)
        stopped = True
        counters = _counter_ledger(before, after)
        memory = _memory_evidence(
            samples=samples,
            state=state,
            graph=graph,
            log_path=log_snapshot,
            server_alive_after=server_alive,
        )
        target_binding = pre.get("artifact_binding")
        if not isinstance(target_binding, Mapping):
            raise DSparkCollectorError("pre manifest lacks target artifact binding")
        target_sha = str(target_binding.get("model_sha", ""))
        if _SHA256.fullmatch(target_sha) is None:
            raise DSparkCollectorError("pre manifest target model SHA is invalid")

        acceptance: dict[str, Any] = {}
        routes: dict[str, Any] = {}
        mtp: dict[str, Any] | None = None
        if arm == MTP_ARM:
            if not isinstance(spec_before, Mapping) or not isinstance(spec_after, Mapping):
                raise DSparkCollectorError("MTP speculative counters are missing")
            acceptance = _acceptance_from_observations(
                base_url=base_url,
                served_model=served_model,
                started_at=started_at,
                finished_at=finished_at,
                wall_seconds=wall_seconds,
                responses=responses,
                before=spec_before,
                after=spec_after,
            )
            routes = collect_route_census(log_snapshot)
            validate_route_census(routes)
            draft_binding = pre.get("draft_artifact_binding")
            if not isinstance(draft_binding, Mapping):
                raise DSparkCollectorError("MTP pre manifest lacks draft binding")
            mtp = {
                "schema": MTP_BINDING_SCHEMA,
                "draft_model_sha": draft_binding.get("model_sha"),
                "draft_artifact_binding": dict(draft_binding),
                "draft_format": "NVFP4_CB_K12",
                "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
                "acceptance": acceptance,
                "routes": routes,
                "routes_sha256": canonical_sha256(routes),
            }
            _write_json_no_clobber(acceptance_output_json, acceptance)

        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "served_model": served_model,
            "started_at": started_at,
            "finished_at": finished_at,
            "tool": tool,
            "policy": dict(declaration["policy"]),
            "target_model_sha": target_sha,
            "pre_manifest": pre,
            "post_manifest": post,
            "graph_capture": graph,
            "workload": _matched_workload(),
            "warmup_responses": warmup,
            "responses": responses,
            "counters": counters,
            "warm_decode": {
                "output_tokens": EXPECTED_OUTPUT_TOKENS,
                "wall_seconds": wall_seconds,
                "output_tokens_per_second": EXPECTED_OUTPUT_TOKENS / wall_seconds,
            },
            "memory": memory,
            "mtp": mtp,
        }
        report["report_sha256"] = canonical_sha256(report)
        output = Path(output_json)
        staging = output.parent / f".{output.name}.validate-{os.getpid()}-{time.time_ns()}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        _atomic_no_clobber(staging.resolve(), _json_bytes(report))
        try:
            validate_arm_report(
                report,
                report_path=staging,
                arm=arm,
                expected_target_sha=target_sha,
                expected_draft_sha=(
                    str(pre.get("draft_artifact_binding", {}).get("model_sha", ""))
                    if arm == MTP_ARM
                    else "0" * 64
                ),
                expected_workload=_matched_workload(),
                expected_acceptance=acceptance,
                expected_routes=routes,
                expected_speculative_config=(
                    DSPARK_SPECULATIVE_CONFIG if arm == MTP_ARM else None
                ),
                expected_launch_options=(
                    _expected_launch_options(served_model)
                    if arm == MTP_ARM
                    else _expected_no_mtp_launch_options(served_model)
                ),
                expected_launch_switches=DSPARK_LAUNCH_SWITCHES,
                requires_moe_marlin=True,
                expected_no_mtp_graph_capture=(
                    graph if arm == NO_MTP_ARM else None
                ),
                expected_mtp_pre_manifest=(pre if arm == MTP_ARM else None),
                expected_mtp_post_manifest=(post if arm == MTP_ARM else None),
                expected_mtp_graph_capture=(graph if arm == MTP_ARM else None),
            )
            _atomic_no_clobber(output, staging.read_bytes())
        finally:
            try:
                staging.unlink()
            except FileNotFoundError:
                pass
        return report
    finally:
        if not stopped:
            try:
                _stop_sampler(directory, state)
            except BaseException:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    declare = commands.add_parser("declare-policy")
    declare.add_argument("--output", required=True)

    sampler = commands.add_parser("start-sampler")
    sampler.add_argument("--state-dir", required=True)
    sampler.add_argument("--policy-declaration", required=True)

    collect = commands.add_parser("collect-arm")
    collect.add_argument("--arm", required=True, choices=ARMS)
    collect.add_argument("--state-dir", required=True)
    collect.add_argument("--base-url", required=True)
    collect.add_argument("--artifact-dir", default="/model")
    collect.add_argument("--draft-artifact-dir")
    collect.add_argument("--runtime-evidence", required=True)
    collect.add_argument("--serve-log", required=True)
    collect.add_argument("--serve-log-snapshot-out", required=True)
    collect.add_argument("--pre-manifest-out", required=True)
    collect.add_argument("--post-manifest-out", required=True)
    collect.add_argument("--output-json", required=True)
    collect.add_argument("--acceptance-output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "declare-policy":
            payload = declare_policy(args.output)
            print(
                f"[dspark-collector] policy={payload['declaration_sha256']} "
                f"out={args.output}"
            )
            return 0
        if args.command == "start-sampler":
            state = start_sampler(
                state_dir=args.state_dir,
                policy_declaration_path=args.policy_declaration,
            )
            print(
                f"[dspark-collector] sampler_pid={state['sampler_pid']} "
                f"state={args.state_dir}"
            )
            return 0
        report = collect_arm(
            arm=args.arm,
            state_dir=args.state_dir,
            base_url=args.base_url,
            artifact_dir=args.artifact_dir,
            draft_artifact_dir=args.draft_artifact_dir,
            runtime_evidence_path=args.runtime_evidence,
            serve_log_path=args.serve_log,
            serve_log_snapshot_out=args.serve_log_snapshot_out,
            pre_manifest_out=args.pre_manifest_out,
            post_manifest_out=args.post_manifest_out,
            output_json=args.output_json,
            acceptance_output_json=args.acceptance_output_json,
        )
        print(
            f"[dspark-collector] arm={report['arm']} "
            f"tps={report['warm_decode']['output_tokens_per_second']:.6f} "
            f"report={report['report_sha256']} out={args.output_json}"
        )
        return 0
    except Exception as exc:
        print(f"dspark-collector: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
