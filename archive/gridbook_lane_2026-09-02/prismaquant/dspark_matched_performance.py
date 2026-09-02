"""Matched target-only versus K12-DSpark release performance contract.

This module owns evidence validation only.  It deliberately does not start a
server or invent measurements.  The two production serves are measured by
:mod:`prismaquant.dspark_matched_performance_collector`, which writes one
self-digesting report apiece; :mod:`prismaquant.validate_dspark_target_draft`
validates those reports against the already-bound artifacts and process
manifests, then persists the compact result returned here in both reciprocal
``mtp.dspark`` shipcard records.

The comparison is intentionally narrow: the exact 256K, 1.6-GiB FP8-KV,
single-sequence workload on one compiled Gridbook stack, with the
speculative configuration and its necessary graph capture shapes as the only
arm differences.  The MTP arm must be at least as fast as the target-only arm.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .dspark_serving_profile import (
    DSPARK_IMAGE as DSV4_SPARK_VLLM_IMAGE,
    DSPARK_VLLM_VERSION as DSV4_SPARK_VLLM_VERSION,
    DSparkServingProfileError,
    validate_dspark_serve_manifest,
)


REPORT_SCHEMA = "prismaquant.dspark_matched_performance_arm.v1"
ARM_EVIDENCE_SCHEMA = "prismaquant.dspark_matched_performance_arm_evidence.v1"
RESULT_SCHEMA = "prismaquant.dspark_matched_performance.v1"
POLICY_SCHEMA = "prismaquant.dspark_matched_performance_policy.v1"
WORKLOAD_SCHEMA = "prismaquant.dspark_matched_decode_workload.v1"
MEMORY_SCHEMA = "prismaquant.dspark_unified_memory_evidence.v1"
MEMORY_RECEIPT_SCHEMA = "prismaquant.dspark_unified_memory_receipt.v1"
KV_CAPACITY_SCHEMA = "prismaquant.dspark_kv_capacity.v1"
TOOL_SCHEMA = "prismaquant.dspark_matched_performance_tool.v2"
MTP_BINDING_SCHEMA = "prismaquant.dspark_matched_performance_mtp.v1"
MTP_RECEIPT_SCHEMA = "prismaquant.dspark_matched_performance_mtp_receipt.v1"

NO_MTP_ARM = "no_mtp"
MTP_ARM = "mtp_k12"
ARMS = (NO_MTP_ARM, MTP_ARM)

GIB = 1024 ** 3
MODEL_LEN = 262_144
KV_CACHE_MEMORY_BYTES = 1_717_986_918
MAX_NUM_SEQS = 1
MAX_NUM_BATCHED_TOKENS = 512
START_MEM_AVAILABLE_FLOOR_BYTES = 110 * GIB
READY_MEM_AVAILABLE_FLOOR_BYTES = 8 * GIB
WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES = 4 * GIB
MIN_MTP_TO_NO_MTP_THROUGHPUT_RATIO = 1.0
MEMORY_SAMPLE_INTERVAL_MS = 1_000
MEMORY_MAX_SAMPLE_GAP_MS = 2_500
WALL_CHRONOLOGY_ABS_TOLERANCE_SECONDS = 0.001

EXPECTED_PROMPT_COUNT = 8
EXPECTED_OUTPUT_TOKENS_PER_PROMPT = 128
EXPECTED_OUTPUT_TOKENS = (
    EXPECTED_PROMPT_COUNT * EXPECTED_OUTPUT_TOKENS_PER_PROMPT
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_GRAPH_CAPTURE_RE = re.compile(
    r"Graph capturing finished in [0-9]+ secs, took -?[0-9.]+ GiB"
)
_COUNTER_KEYS = (
    "completed_requests",
    "generation_tokens",
    "failed_requests",
    "timed_out_requests",
)
_MEMORY_PHASES = ("startup", "ready", "warmup", "measured", "post")

COLLECTOR_TOOL_NAME = "dspark_matched_performance_collector.py"
COLLECTOR_SOURCE_PATHS = (
    "prismaquant/dspark_matched_performance_collector.py",
    "prismaquant/dspark_matched_performance.py",
    "prismaquant/dspark_serving_profile.py",
    "prismaquant/validate_dspark_target_draft.py",
    "tools/dspark_memavailable_sampler.py",
    "tools/prismaquant_runtime_snapshot.py",
    "tools/prismaquant_source_bootstrap.py",
    "tools/serve_fingerprint.py",
)


class DSparkMatchedPerformanceError(RuntimeError):
    """The paired performance/headroom evidence is incomplete or failed."""


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise DSparkMatchedPerformanceError(
            f"{where} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_sha(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DSparkMatchedPerformanceError(f"{where} is not lowercase SHA-256")
    return value


def _require_positive_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DSparkMatchedPerformanceError(f"{where} is not a positive integer")
    return value


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DSparkMatchedPerformanceError(
            f"{where} is not a non-negative integer"
        )
    return value


def _finite(value: object, *, where: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DSparkMatchedPerformanceError(f"{where} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise DSparkMatchedPerformanceError(f"{where} is not finite and valid")
    return result


def _parse_time(value: object, *, where: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DSparkMatchedPerformanceError(f"{where} is not canonical UTC time")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DSparkMatchedPerformanceError(f"{where} is not ISO-8601") from exc
    if result.tzinfo is None:
        raise DSparkMatchedPerformanceError(f"{where} has no timezone")
    return result.astimezone(timezone.utc)


def release_policy(*, predeclared_at: str) -> dict[str, Any]:
    """Return the one currently authorized DSpark comparison policy."""
    _parse_time(predeclared_at, where="performance policy predeclared_at")
    return {
        "schema": POLICY_SCHEMA,
        "predeclared_at": predeclared_at,
        "minimum_mtp_to_no_mtp_throughput_ratio": (
            MIN_MTP_TO_NO_MTP_THROUGHPUT_RATIO
        ),
        "minimum_start_mem_available_bytes": (
            START_MEM_AVAILABLE_FLOOR_BYTES
        ),
        "minimum_ready_mem_available_bytes": (
            READY_MEM_AVAILABLE_FLOOR_BYTES
        ),
        "minimum_watchdog_mem_available_bytes": (
            WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES
        ),
        "required_max_model_len": MODEL_LEN,
        "required_kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
        "required_kv_capacity_tokens": MODEL_LEN,
    }


def _validate_policy(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("performance policy is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "predeclared_at",
            "minimum_mtp_to_no_mtp_throughput_ratio",
            "minimum_start_mem_available_bytes",
            "minimum_ready_mem_available_bytes",
            "minimum_watchdog_mem_available_bytes",
            "required_max_model_len",
            "required_kv_cache_memory_bytes",
            "required_kv_capacity_tokens",
        },
        where="performance policy",
    )
    _finite(
        payload.get("minimum_mtp_to_no_mtp_throughput_ratio"),
        where="minimum MTP/no-MTP throughput ratio",
        positive=True,
    )
    for name in (
        "minimum_start_mem_available_bytes",
        "minimum_ready_mem_available_bytes",
        "minimum_watchdog_mem_available_bytes",
        "required_max_model_len",
        "required_kv_cache_memory_bytes",
        "required_kv_capacity_tokens",
    ):
        _require_positive_int(payload.get(name), where=f"performance policy {name}")
    expected = release_policy(predeclared_at=str(payload.get("predeclared_at", "")))
    if dict(payload) != expected:
        raise DSparkMatchedPerformanceError(
            "performance policy differs from the strict DSpark release policy"
        )
    return dict(payload)


def collector_tool_identity(
    *, source_root: str | Path, git_commit: str
) -> dict[str, Any]:
    """Build the exact source closure stamped by the arm collector.

    The clean commit is checked by the collector's release-source boundary.
    This helper owns the byte closure so report creation and validation cannot
    silently disagree about which files constitute the measurement tool.
    """
    if _COMMIT_RE.fullmatch(git_commit) is None:
        raise DSparkMatchedPerformanceError(
            "collector Git identity is not one full lowercase commit"
        )
    root = Path(source_root)
    if not root.is_absolute() or root.is_symlink():
        raise DSparkMatchedPerformanceError(
            "collector source root must be one absolute non-symlink directory"
        )
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise DSparkMatchedPerformanceError(
            "collector source root is unreadable"
        ) from exc
    source_files: dict[str, str] = {}
    for relative in COLLECTOR_SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise DSparkMatchedPerformanceError(
                f"collector source closure member is absent or unsafe: {relative}"
            )
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise DSparkMatchedPerformanceError(
                f"collector source closure member escapes its root: {relative}"
            ) from exc
        source_files[relative] = file_sha256(resolved)
    source_files = dict(sorted(source_files.items()))
    return {
        "schema": TOOL_SCHEMA,
        "name": COLLECTOR_TOOL_NAME,
        "git_commit": git_commit,
        "collector_source_sha256": source_files[
            "prismaquant/dspark_matched_performance_collector.py"
        ],
        "source_files": source_files,
        "source_files_sha256": canonical_sha256(source_files),
    }


def _validate_tool(
    payload: object, *, verify_local_source: bool = False
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("performance report tool is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "name",
            "git_commit",
            "collector_source_sha256",
            "source_files",
            "source_files_sha256",
        },
        where="performance report tool",
    )
    source_files = payload.get("source_files")
    if (
        payload.get("schema") != TOOL_SCHEMA
        or payload.get("name") != COLLECTOR_TOOL_NAME
        or not isinstance(payload.get("git_commit"), str)
        or _COMMIT_RE.fullmatch(str(payload.get("git_commit"))) is None
        or not isinstance(source_files, Mapping)
        or set(source_files) != set(COLLECTOR_SOURCE_PATHS)
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for path, digest in source_files.items()
        )
        or payload.get("collector_source_sha256")
        != source_files.get(
            "prismaquant/dspark_matched_performance_collector.py"
        )
        or payload.get("source_files_sha256")
        != canonical_sha256(dict(sorted(source_files.items())))
    ):
        raise DSparkMatchedPerformanceError(
            "performance reports require the exact source-closed PrismaQuant collector"
        )
    if verify_local_source:
        expected = collector_tool_identity(
            source_root=Path(__file__).resolve(strict=True).parents[1],
            git_commit=str(payload["git_commit"]),
        )
        if dict(payload) != expected:
            raise DSparkMatchedPerformanceError(
                "performance report collector bytes differ from the attester source"
            )
    return dict(payload)


def _validate_response_ledger(
    payload: object,
    *,
    workload: Mapping[str, Any],
    where: str,
) -> list[dict[str, Any]]:
    rows = payload
    prompt_sha = workload.get("prompt_sha256")
    if (
        not isinstance(rows, list)
        or not isinstance(prompt_sha, list)
        or len(rows) != EXPECTED_PROMPT_COUNT
        or len(prompt_sha) != EXPECTED_PROMPT_COUNT
    ):
        raise DSparkMatchedPerformanceError(f"{where} is incomplete")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DSparkMatchedPerformanceError(f"{where}[{index}] is malformed")
        _require_exact_keys(
            row,
            {
                "index",
                "prompt_sha256",
                "content_sha256",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
                "request_seconds",
            },
            where=f"{where}[{index}]",
        )
        if (
            row.get("index") != index
            or row.get("prompt_sha256") != prompt_sha[index]
            or _SHA256_RE.fullmatch(str(row.get("content_sha256", ""))) is None
            or row.get("finish_reason") != "length"
            or _require_positive_int(
                row.get("prompt_tokens"), where=f"{where}[{index}].prompt_tokens"
            ) <= 0
            or row.get("completion_tokens") != EXPECTED_OUTPUT_TOKENS_PER_PROMPT
        ):
            raise DSparkMatchedPerformanceError(
                f"{where}[{index}] differs from the fixed-token contract"
            )
        _finite(
            row.get("request_seconds"),
            where=f"{where}[{index}].request_seconds",
            positive=True,
        )
        result.append(dict(row))
    return result


def _validate_counters(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("performance counters are missing")
    _require_exact_keys(
        payload, {"before", "after", "delta"}, where="performance counters"
    )
    snapshots: dict[str, dict[str, int]] = {}
    for label in ("before", "after", "delta"):
        raw = payload.get(label)
        if not isinstance(raw, Mapping) or set(raw) != set(_COUNTER_KEYS):
            raise DSparkMatchedPerformanceError(
                f"performance counters.{label} keys differ"
            )
        snapshots[label] = {
            key: _require_nonnegative_int(
                raw[key], where=f"performance counters.{label}.{key}"
            )
            for key in _COUNTER_KEYS
        }
    for key in _COUNTER_KEYS:
        if (
            snapshots["after"][key] - snapshots["before"][key]
            != snapshots["delta"][key]
        ):
            raise DSparkMatchedPerformanceError(
                f"performance counter arithmetic differs for {key}"
            )
    expected_delta = {
        "completed_requests": EXPECTED_PROMPT_COUNT,
        "generation_tokens": EXPECTED_OUTPUT_TOKENS,
        "failed_requests": 0,
        "timed_out_requests": 0,
    }
    if snapshots["delta"] != expected_delta:
        raise DSparkMatchedPerformanceError(
            "performance counter interval is contaminated or incomplete"
        )
    return {key: snapshots[key] for key in ("before", "after", "delta")}


def _validate_graph_capture(
    payload: object, *, arm: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("graph-capture evidence is missing")
    _require_exact_keys(
        payload,
        {"serve_log_sha256", "capture_marker", "capture_sizes"},
        where="graph-capture evidence",
    )
    _require_sha(payload.get("serve_log_sha256"), where="serve log digest")
    if _GRAPH_CAPTURE_RE.fullmatch(str(payload.get("capture_marker", ""))) is None:
        raise DSparkMatchedPerformanceError("graph capture marker is malformed")
    expected_sizes = [1] if arm == NO_MTP_ARM else [5, 6]
    capture_sizes = payload.get("capture_sizes")
    if (
        not isinstance(capture_sizes, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in capture_sizes
        )
        or capture_sizes != expected_sizes
    ):
        raise DSparkMatchedPerformanceError(
            f"{arm} graph capture sizes differ from {expected_sizes}"
        )
    return dict(payload)


def _validate_kv_capacity(
    payload: object, *, graph_capture: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("KV capacity evidence is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "dtype",
            "requested_bytes",
            "allocated_bytes",
            "capacity_tokens",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "concurrency_at_max_model_len",
            "profile_log_sha256",
            "capacity_verified",
        },
        where="KV capacity evidence",
    )
    concurrency = _finite(
        payload.get("concurrency_at_max_model_len"),
        where="KV concurrency at 256K",
        positive=True,
    )
    for name in (
        "requested_bytes",
        "allocated_bytes",
        "capacity_tokens",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        _require_positive_int(payload.get(name), where=f"KV capacity {name}")
    expected_concurrency = float(payload.get("capacity_tokens")) / MODEL_LEN
    if (
        payload.get("schema") != KV_CAPACITY_SCHEMA
        or payload.get("dtype") != "fp8"
        or payload.get("requested_bytes") != KV_CACHE_MEMORY_BYTES
        or payload.get("allocated_bytes") != KV_CACHE_MEMORY_BYTES
        or int(payload.get("capacity_tokens")) < MODEL_LEN
        or payload.get("max_model_len") != MODEL_LEN
        or payload.get("max_num_seqs") != MAX_NUM_SEQS
        or payload.get("max_num_batched_tokens") != MAX_NUM_BATCHED_TOKENS
        or concurrency < 1.0
        # vLLM prints this value to two decimal places.  Bind it back to the
        # exact token capacity while allowing only that display rounding.
        or abs(concurrency - expected_concurrency) > 0.011
        or payload.get("profile_log_sha256")
        != graph_capture.get("serve_log_sha256")
        or payload.get("capacity_verified") is not True
    ):
        raise DSparkMatchedPerformanceError(
            "KV evidence does not prove the exact 256K/1.6-GiB FP8 capacity"
        )
    return dict(payload)


def _validate_memory(
    payload: object,
    *,
    report_started: datetime,
    report_finished: datetime,
    pre_created: datetime,
    post_created: datetime,
    graph_capture: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("memory evidence is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "memory_kind",
            "sampler",
            "sample_interval_ms",
            "samples",
            "start_mem_available_bytes",
            "ready_mem_available_bytes",
            "startup_min_mem_available_bytes",
            "measured_min_mem_available_bytes",
            "post_mem_available_bytes",
            "minimum_mem_available_bytes",
            "model_residency_bytes",
            "startup_transient_bytes",
            "measured_transient_bytes",
            "watchdog_floor_bytes",
            "watchdog_tripped",
            "oom_events",
            "oom_kill_detected",
            "server_alive_after",
            "kv_cache",
        },
        where="memory evidence",
    )
    if (
        payload.get("schema") != MEMORY_SCHEMA
        or payload.get("memory_kind") != "nvidia_gb10_unified"
        or payload.get("sampler") != "/proc/meminfo:MemAvailable"
        or payload.get("sample_interval_ms") != MEMORY_SAMPLE_INTERVAL_MS
    ):
        raise DSparkMatchedPerformanceError(
            "memory evidence is not the production GB10 MemAvailable sampler"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < len(_MEMORY_PHASES):
        raise DSparkMatchedPerformanceError("memory sample ledger is incomplete")
    phase_rank = {phase: index for index, phase in enumerate(_MEMORY_PHASES)}
    validated_samples: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    previous_rank = -1
    sample_gaps_ms: list[float] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise DSparkMatchedPerformanceError(
                f"memory samples[{index}] is malformed"
            )
        _require_exact_keys(
            row,
            {"sequence", "observed_at", "phase", "mem_available_bytes"},
            where=f"memory samples[{index}]",
        )
        phase = row.get("phase")
        if row.get("sequence") != index or phase not in phase_rank:
            raise DSparkMatchedPerformanceError(
                f"memory samples[{index}] has invalid sequence/phase"
            )
        rank = phase_rank[str(phase)]
        if rank < previous_rank:
            raise DSparkMatchedPerformanceError("memory sample phases regress")
        observed = _parse_time(
            row.get("observed_at"), where=f"memory samples[{index}].observed_at"
        )
        if previous_time is not None:
            gap_ms = (observed - previous_time).total_seconds() * 1000.0
            if gap_ms <= 0 or gap_ms > MEMORY_MAX_SAMPLE_GAP_MS:
                raise DSparkMatchedPerformanceError(
                    "memory sampler has a non-positive or excessive observation gap"
                )
            sample_gaps_ms.append(gap_ms)
        _require_positive_int(
            row.get("mem_available_bytes"),
            where=f"memory samples[{index}].mem_available_bytes",
        )
        validated_samples.append(dict(row))
        previous_time = observed
        previous_rank = rank
    phases = [str(row["phase"]) for row in validated_samples]
    if (
        phases[0] != "startup"
        or phases[-1] != "post"
        or any(phase not in phases for phase in _MEMORY_PHASES)
    ):
        raise DSparkMatchedPerformanceError(
            "memory sample ledger does not cover startup/ready/warmup/measured/post"
        )
    by_phase: dict[str, list[dict[str, Any]]] = {
        phase: [row for row in validated_samples if row["phase"] == phase]
        for phase in _MEMORY_PHASES
    }
    ready_time = _parse_time(
        by_phase["ready"][0]["observed_at"], where="ready memory sample"
    )
    first_warmup = _parse_time(
        by_phase["warmup"][0]["observed_at"], where="first warmup sample"
    )
    last_warmup = _parse_time(
        by_phase["warmup"][-1]["observed_at"], where="last warmup sample"
    )
    first_measured = _parse_time(
        by_phase["measured"][0]["observed_at"], where="first measured sample"
    )
    last_measured = _parse_time(
        by_phase["measured"][-1]["observed_at"], where="last measured sample"
    )
    post_time = _parse_time(
        by_phase["post"][0]["observed_at"], where="post memory sample"
    )
    if not (
        ready_time <= pre_created <= first_warmup <= last_warmup
        <= report_started
        and first_measured <= report_started < report_finished <= last_measured
        and report_finished <= post_time <= post_created
    ):
        raise DSparkMatchedPerformanceError(
            "memory samples do not bracket warmup and measured workload chronology"
        )

    values = [int(row["mem_available_bytes"]) for row in validated_samples]
    start_value = values[0]
    ready_value = int(by_phase["ready"][0]["mem_available_bytes"])
    startup_min = min(
        int(row["mem_available_bytes"])
        for phase in ("startup", "ready")
        for row in by_phase[phase]
    )
    measured_min = min(
        int(row["mem_available_bytes"]) for row in by_phase["measured"]
    )
    post_value = int(by_phase["post"][0]["mem_available_bytes"])
    minimum = min(values)
    model_residency = max(0, start_value - ready_value)
    startup_transient = max(0, ready_value - startup_min)
    measured_transient = max(0, ready_value - measured_min)
    derived = {
        "start_mem_available_bytes": start_value,
        "ready_mem_available_bytes": ready_value,
        "startup_min_mem_available_bytes": startup_min,
        "measured_min_mem_available_bytes": measured_min,
        "post_mem_available_bytes": post_value,
        "minimum_mem_available_bytes": minimum,
        "model_residency_bytes": model_residency,
        "startup_transient_bytes": startup_transient,
        "measured_transient_bytes": measured_transient,
    }
    for key, expected in derived.items():
        if payload.get(key) != expected:
            raise DSparkMatchedPerformanceError(
                f"memory evidence {key} does not replay from its sample ledger"
            )
    if model_residency <= 0:
        raise DSparkMatchedPerformanceError(
            "memory evidence does not record positive model residency"
        )
    if (
        start_value < START_MEM_AVAILABLE_FLOOR_BYTES
        or ready_value < READY_MEM_AVAILABLE_FLOOR_BYTES
        or post_value < READY_MEM_AVAILABLE_FLOOR_BYTES
        or minimum < WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES
        or payload.get("watchdog_floor_bytes")
        != WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES
        or payload.get("watchdog_tripped") is not False
        or _require_nonnegative_int(
            payload.get("oom_events"), where="memory OOM events"
        ) != 0
        or payload.get("oom_kill_detected") is not False
        or payload.get("server_alive_after") is not True
    ):
        raise DSparkMatchedPerformanceError(
            "memory evidence fails the production 110/8/4-GiB headroom, "
            "watchdog, or OOM policy"
        )
    kv_cache = _validate_kv_capacity(
        payload.get("kv_cache"), graph_capture=graph_capture
    )
    phase_summary: dict[str, dict[str, Any]] = {}
    for phase in _MEMORY_PHASES:
        rows = by_phase[phase]
        phase_values = [int(row["mem_available_bytes"]) for row in rows]
        phase_summary[phase] = {
            "count": len(rows),
            "first_observed_at": rows[0]["observed_at"],
            "last_observed_at": rows[-1]["observed_at"],
            "first_mem_available_bytes": phase_values[0],
            "last_mem_available_bytes": phase_values[-1],
            "minimum_mem_available_bytes": min(phase_values),
            "maximum_mem_available_bytes": max(phase_values),
        }
    # The full one-second ledger can contain thousands of rows during model
    # startup.  Shipcards retain a replayable phase summary plus digests of
    # both that ledger and the complete source report instead of duplicating
    # the raw samples into two reciprocal cards.
    receipt: dict[str, Any] = {
        "schema": MEMORY_RECEIPT_SCHEMA,
        "source_schema": MEMORY_SCHEMA,
        "memory_kind": payload["memory_kind"],
        "sampler": payload["sampler"],
        "sample_interval_ms": payload["sample_interval_ms"],
        "sample_count": len(validated_samples),
        "sample_ledger_sha256": canonical_sha256(validated_samples),
        "maximum_sample_gap_ms": max(sample_gaps_ms),
        "phase_summary": phase_summary,
        **derived,
        "watchdog_floor_bytes": payload["watchdog_floor_bytes"],
        "watchdog_tripped": payload["watchdog_tripped"],
        "oom_events": payload["oom_events"],
        "oom_kill_detected": payload["oom_kill_detected"],
        "server_alive_after": payload["server_alive_after"],
        "kv_cache": kv_cache,
    }
    return _validate_memory_receipt(
        receipt,
        report_started=report_started,
        report_finished=report_finished,
        pre_created=pre_created,
        post_created=post_created,
        graph_capture=graph_capture,
    )


def _validate_memory_receipt(
    payload: object,
    *,
    report_started: datetime,
    report_finished: datetime,
    pre_created: datetime,
    post_created: datetime,
    graph_capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the compact projection of a validated continuous sample ledger."""
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("compact memory receipt is missing")
    derived_keys = {
        "start_mem_available_bytes",
        "ready_mem_available_bytes",
        "startup_min_mem_available_bytes",
        "measured_min_mem_available_bytes",
        "post_mem_available_bytes",
        "minimum_mem_available_bytes",
        "model_residency_bytes",
        "startup_transient_bytes",
        "measured_transient_bytes",
    }
    _require_exact_keys(
        payload,
        {
            "schema",
            "source_schema",
            "memory_kind",
            "sampler",
            "sample_interval_ms",
            "sample_count",
            "sample_ledger_sha256",
            "maximum_sample_gap_ms",
            "phase_summary",
            *derived_keys,
            "watchdog_floor_bytes",
            "watchdog_tripped",
            "oom_events",
            "oom_kill_detected",
            "server_alive_after",
            "kv_cache",
        },
        where="compact memory receipt",
    )
    sample_count = _require_positive_int(
        payload.get("sample_count"), where="compact memory sample count"
    )
    _require_sha(
        payload.get("sample_ledger_sha256"),
        where="compact memory sample-ledger digest",
    )
    maximum_gap = _finite(
        payload.get("maximum_sample_gap_ms"),
        where="compact memory maximum sample gap",
        positive=True,
    )
    if (
        payload.get("schema") != MEMORY_RECEIPT_SCHEMA
        or payload.get("source_schema") != MEMORY_SCHEMA
        or payload.get("memory_kind") != "nvidia_gb10_unified"
        or payload.get("sampler") != "/proc/meminfo:MemAvailable"
        or payload.get("sample_interval_ms") != MEMORY_SAMPLE_INTERVAL_MS
        or sample_count < len(_MEMORY_PHASES)
        or maximum_gap > MEMORY_MAX_SAMPLE_GAP_MS
    ):
        raise DSparkMatchedPerformanceError(
            "compact memory receipt is not the production continuous GB10 sampler"
        )
    raw_summary = payload.get("phase_summary")
    if not isinstance(raw_summary, Mapping) or set(raw_summary) != set(
        _MEMORY_PHASES
    ):
        raise DSparkMatchedPerformanceError(
            "compact memory receipt phase summary differs"
        )
    phase_keys = {
        "count",
        "first_observed_at",
        "last_observed_at",
        "first_mem_available_bytes",
        "last_mem_available_bytes",
        "minimum_mem_available_bytes",
        "maximum_mem_available_bytes",
    }
    phase_summary: dict[str, dict[str, Any]] = {}
    previous_last: datetime | None = None
    for phase in _MEMORY_PHASES:
        raw = raw_summary.get(phase)
        if not isinstance(raw, Mapping):
            raise DSparkMatchedPerformanceError(
                f"compact memory {phase} phase is missing"
            )
        _require_exact_keys(raw, phase_keys, where=f"compact memory {phase} phase")
        count = _require_positive_int(
            raw.get("count"), where=f"compact memory {phase} count"
        )
        first = _parse_time(
            raw.get("first_observed_at"), where=f"compact memory {phase} first"
        )
        last = _parse_time(
            raw.get("last_observed_at"), where=f"compact memory {phase} last"
        )
        values = {
            key: _require_positive_int(
                raw.get(key), where=f"compact memory {phase} {key}"
            )
            for key in (
                "first_mem_available_bytes",
                "last_mem_available_bytes",
                "minimum_mem_available_bytes",
                "maximum_mem_available_bytes",
            )
        }
        if (
            last < first
            or (count == 1 and last != first)
            or (count > 1 and last <= first)
            or (
                count > 1
                and (last - first).total_seconds() * 1000.0
                > (count - 1) * maximum_gap
            )
            or values["minimum_mem_available_bytes"]
            > values["maximum_mem_available_bytes"]
            or not (
                values["minimum_mem_available_bytes"]
                <= values["first_mem_available_bytes"]
                <= values["maximum_mem_available_bytes"]
            )
            or not (
                values["minimum_mem_available_bytes"]
                <= values["last_mem_available_bytes"]
                <= values["maximum_mem_available_bytes"]
            )
            or (
                previous_last is not None
                and (
                    first <= previous_last
                    or (first - previous_last).total_seconds() * 1000.0
                    > maximum_gap
                )
            )
        ):
            raise DSparkMatchedPerformanceError(
                f"compact memory {phase} phase chronology/range differs"
            )
        phase_summary[phase] = dict(raw)
        previous_last = last
    if sum(int(row["count"]) for row in phase_summary.values()) != sample_count:
        raise DSparkMatchedPerformanceError(
            "compact memory phase counts do not replay"
        )
    ready_time = _parse_time(
        phase_summary["ready"]["first_observed_at"],
        where="compact ready memory sample",
    )
    first_warmup = _parse_time(
        phase_summary["warmup"]["first_observed_at"],
        where="compact first warmup memory sample",
    )
    last_warmup = _parse_time(
        phase_summary["warmup"]["last_observed_at"],
        where="compact last warmup memory sample",
    )
    first_measured = _parse_time(
        phase_summary["measured"]["first_observed_at"],
        where="compact first measured memory sample",
    )
    last_measured = _parse_time(
        phase_summary["measured"]["last_observed_at"],
        where="compact last measured memory sample",
    )
    post_time = _parse_time(
        phase_summary["post"]["first_observed_at"],
        where="compact post memory sample",
    )
    if not (
        ready_time <= pre_created <= first_warmup <= last_warmup
        <= report_started
        and first_measured <= report_started < report_finished <= last_measured
        and report_finished <= post_time <= post_created
    ):
        raise DSparkMatchedPerformanceError(
            "compact memory receipt does not bracket warmup/measurement"
        )
    start_value = int(
        phase_summary["startup"]["first_mem_available_bytes"]
    )
    ready_value = int(phase_summary["ready"]["first_mem_available_bytes"])
    startup_min = min(
        int(phase_summary[phase]["minimum_mem_available_bytes"])
        for phase in ("startup", "ready")
    )
    measured_min = int(
        phase_summary["measured"]["minimum_mem_available_bytes"]
    )
    post_value = int(phase_summary["post"]["first_mem_available_bytes"])
    minimum = min(
        int(row["minimum_mem_available_bytes"])
        for row in phase_summary.values()
    )
    expected_derived = {
        "start_mem_available_bytes": start_value,
        "ready_mem_available_bytes": ready_value,
        "startup_min_mem_available_bytes": startup_min,
        "measured_min_mem_available_bytes": measured_min,
        "post_mem_available_bytes": post_value,
        "minimum_mem_available_bytes": minimum,
        "model_residency_bytes": max(0, start_value - ready_value),
        "startup_transient_bytes": max(0, ready_value - startup_min),
        "measured_transient_bytes": max(0, ready_value - measured_min),
    }
    if any(payload.get(key) != value for key, value in expected_derived.items()):
        raise DSparkMatchedPerformanceError(
            "compact memory summary does not replay from its phase receipt"
        )
    if expected_derived["model_residency_bytes"] <= 0:
        raise DSparkMatchedPerformanceError(
            "compact memory receipt has no positive model residency"
        )
    if (
        start_value < START_MEM_AVAILABLE_FLOOR_BYTES
        or ready_value < READY_MEM_AVAILABLE_FLOOR_BYTES
        or post_value < READY_MEM_AVAILABLE_FLOOR_BYTES
        or minimum < WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES
        or payload.get("watchdog_floor_bytes")
        != WATCHDOG_MEM_AVAILABLE_FLOOR_BYTES
        or payload.get("watchdog_tripped") is not False
        or _require_nonnegative_int(
            payload.get("oom_events"), where="compact memory OOM events"
        ) != 0
        or payload.get("oom_kill_detected") is not False
        or payload.get("server_alive_after") is not True
    ):
        raise DSparkMatchedPerformanceError(
            "compact memory receipt fails the production 110/8/4-GiB "
            "headroom, watchdog, or OOM policy"
        )
    _validate_kv_capacity(payload.get("kv_cache"), graph_capture=graph_capture)
    return dict(payload)


def _validate_workload(
    payload: object, *, expected_workload: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or dict(payload) != dict(expected_workload):
        raise DSparkMatchedPerformanceError(
            "matched performance workload differs from the fixed release workload"
        )
    if (
        payload.get("schema") != WORKLOAD_SCHEMA
        or payload.get("prompt_count") != EXPECTED_PROMPT_COUNT
        or payload.get("max_tokens_per_prompt")
        != EXPECTED_OUTPUT_TOKENS_PER_PROMPT
        or payload.get("warmup_repetitions") != 1
        or payload.get("measured_repetitions") != 1
        or payload.get("max_concurrency") != 1
        or payload.get("temperature") != 0
        or payload.get("ignore_eos") is not True
        or payload.get("stream") is not False
    ):
        raise DSparkMatchedPerformanceError(
            "matched performance workload is not one warm then one measured 8x128 decode"
        )
    return dict(payload)


def _mtp_receipt(
    *,
    expected_draft_sha: str,
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    draft_binding: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": MTP_RECEIPT_SCHEMA,
        "draft_model_sha": expected_draft_sha,
        "draft_artifact_binding": dict(draft_binding),
        "draft_format": "NVFP4_CB_K12",
        "num_speculative_tokens": 5,
        "acceptance_sha256": canonical_sha256(expected_acceptance),
        "acceptance_interval": {
            "started_at": expected_acceptance.get("started_at"),
            "finished_at": expected_acceptance.get("finished_at"),
            "served_model": expected_acceptance.get("served_model"),
        },
        "acceptance_summary": expected_acceptance.get("acceptance"),
        "acceptance_counters_sha256": canonical_sha256({
            key: expected_acceptance.get(key)
            for key in ("before", "after", "delta")
        }),
        "routes_receipt_sha256": expected_routes.get("receipt_sha256"),
        "routes_sha256": expected_routes.get("routes_sha256"),
        "routes_counts": expected_routes.get("counts"),
        "route_serve_log_sha256": expected_routes.get("serve_log_sha256"),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_mtp_report_binding(
    payload: object,
    *,
    arm: str,
    expected_draft_sha: str,
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    draft_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if arm == NO_MTP_ARM:
        if payload is not None or draft_binding is not None:
            raise DSparkMatchedPerformanceError(
                "target-only performance arm unexpectedly binds an MTP draft"
            )
        return None
    if not isinstance(payload, Mapping) or not isinstance(draft_binding, Mapping):
        raise DSparkMatchedPerformanceError("MTP performance binding is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "draft_model_sha",
            "draft_artifact_binding",
            "draft_format",
            "num_speculative_tokens",
            "acceptance",
            "routes",
            "routes_sha256",
        },
        where="MTP performance binding",
    )
    routes = payload.get("routes")
    if not isinstance(routes, Mapping):
        raise DSparkMatchedPerformanceError("MTP route evidence is missing")
    if (
        payload.get("schema") != MTP_BINDING_SCHEMA
        or payload.get("draft_model_sha") != expected_draft_sha
        or payload.get("draft_artifact_binding") != dict(draft_binding)
        or payload.get("draft_format") != "NVFP4_CB_K12"
        or payload.get("num_speculative_tokens") != 5
        or payload.get("acceptance") != dict(expected_acceptance)
        or dict(routes) != dict(expected_routes)
        or payload.get("routes_sha256") != canonical_sha256(routes)
    ):
        raise DSparkMatchedPerformanceError(
            "MTP performance report differs from the exact draft, acceptance, or route evidence"
        )
    return _mtp_receipt(
        expected_draft_sha=expected_draft_sha,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        draft_binding=draft_binding,
    )


def _validate_mtp_receipt(
    payload: object,
    *,
    arm: str,
    expected_draft_sha: str,
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    draft_binding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if arm == NO_MTP_ARM:
        if payload is not None or draft_binding is not None:
            raise DSparkMatchedPerformanceError(
                "target-only compact evidence unexpectedly binds an MTP draft"
            )
        return None
    if not isinstance(payload, Mapping) or not isinstance(draft_binding, Mapping):
        raise DSparkMatchedPerformanceError("compact MTP binding is missing")
    expected = _mtp_receipt(
        expected_draft_sha=expected_draft_sha,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        draft_binding=draft_binding,
    )
    if dict(payload) != expected:
        raise DSparkMatchedPerformanceError(
            "compact MTP draft/acceptance/route receipt differs"
        )
    return dict(payload)


def _manifest_compact_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    host = manifest.get("host_identity")
    environment = manifest.get("server_process_environment")
    profile = manifest.get("dspark_serving_profile")
    runtime = manifest.get("dspark_runtime_evidence")
    return {
        "image": manifest.get("image"),
        "gpu_name": manifest.get("gpu_name"),
        "gpu_uuid": manifest.get("gpu_uuid"),
        "gpu_count": manifest.get("gpu_count"),
        "driver_version": manifest.get("driver_version"),
        "host_boot_id": host.get("boot_id") if isinstance(host, Mapping) else None,
        "host_machine_id_sha256": (
            host.get("machine_id_sha256") if isinstance(host, Mapping) else None
        ),
        "package_versions": manifest.get("package_versions"),
        "gridbook_runtime_pin": manifest.get("gridbook_runtime_pin"),
        "dspark_serving_profile_sha256": (
            profile.get("receipt_sha256")
            if isinstance(profile, Mapping) else None
        ),
        "dspark_runtime_evidence_sha256": (
            runtime.get("evidence_sha256")
            if isinstance(runtime, Mapping) else None
        ),
        "gridbook_distribution": manifest.get("gridbook_distribution"),
        "resident_extensions": manifest.get("resident_extensions"),
        "residency_readable": manifest.get("residency_readable"),
        "server_environment": (
            environment.get("values") if isinstance(environment, Mapping) else None
        ),
        "target_artifact_binding": manifest.get("artifact_binding"),
    }


def _launch_contract(
    *,
    expected_speculative_config: Mapping[str, Any] | None,
    expected_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
) -> dict[str, Any]:
    return {
        "options": dict(expected_launch_options),
        "switches": sorted(expected_launch_switches),
        "speculative_config": (
            dict(expected_speculative_config)
            if expected_speculative_config is not None
            else None
        ),
    }


def _validate_launch_contract(
    payload: object,
    *,
    expected_speculative_config: Mapping[str, Any] | None,
    expected_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
) -> dict[str, Any]:
    expected = _launch_contract(
        expected_speculative_config=expected_speculative_config,
        expected_launch_options=expected_launch_options,
        expected_launch_switches=expected_launch_switches,
    )
    if not isinstance(payload, Mapping) or dict(payload) != expected:
        raise DSparkMatchedPerformanceError(
            "compact arm launch contract differs from the validated manifest"
        )
    return dict(payload)


def _validate_manifest_pair(
    report: Mapping[str, Any],
    *,
    arm: str,
    expected_target_sha: str,
    expected_draft_sha: str,
    expected_speculative_config: Mapping[str, Any] | None,
    expected_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
    requires_moe_marlin: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    pre = report.get("pre_manifest")
    post = report.get("post_manifest")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        raise DSparkMatchedPerformanceError(
            f"{arm} performance report lacks pre/post serve manifests"
        )
    served_model = str(report.get("served_model", ""))
    fingerprints: list[str] = []
    for phase, manifest in (("pre", pre), ("post", post)):
        if manifest.get("attestation_phase") != phase:
            raise DSparkMatchedPerformanceError(
                f"{arm} {phase} manifest has wrong chronology role"
            )
        try:
            fingerprint = validate_dspark_serve_manifest(
                manifest,
                arm="graph",
                expected_served_model=served_model,
                requires_moe_marlin=requires_moe_marlin,
                expected_model_sha=expected_target_sha,
                expected_speculative_config=expected_speculative_config,
                expected_launch_options=expected_launch_options,
                expected_launch_switches=expected_launch_switches,
            )
        except DSparkServingProfileError as exc:
            raise DSparkMatchedPerformanceError(
                f"{arm} {phase} serve manifest differs: {exc}"
            ) from exc
        fingerprints.append(fingerprint)
        draft_binding = manifest.get("draft_artifact_binding")
        if arm == NO_MTP_ARM:
            if draft_binding is not None:
                raise DSparkMatchedPerformanceError(
                    "target-only serve manifest unexpectedly binds a draft"
                )
        elif (
            not isinstance(draft_binding, Mapping)
            or draft_binding.get("model_sha") != expected_draft_sha
            or draft_binding.get("launch_model") != "/draft"
        ):
            raise DSparkMatchedPerformanceError(
                "MTP serve manifest is not bound to the exact draft artifact"
            )
    if fingerprints[0] != fingerprints[1]:
        raise DSparkMatchedPerformanceError(
            f"{arm} pre/post stable serve fingerprints differ"
        )
    for key in (
        "serve_session_id",
        "processes",
        "artifact_binding",
        "draft_artifact_binding",
        "gridbook_runtime_pin",
    ):
        if pre.get(key) != post.get(key):
            raise DSparkMatchedPerformanceError(
                f"{arm} pre/post manifests do not preserve {key}"
            )
    return dict(pre), dict(post), fingerprints[0]


def _report_unstamped(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("report_sha256", None)
    return result


def _arm_evidence_unstamped(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("evidence_sha256", None)
    return result


def validate_arm_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
    arm: str,
    expected_target_sha: str,
    expected_draft_sha: str,
    expected_workload: Mapping[str, Any],
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    expected_speculative_config: Mapping[str, Any] | None,
    expected_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
    requires_moe_marlin: bool,
    expected_no_mtp_graph_capture: Mapping[str, Any] | None = None,
    expected_mtp_pre_manifest: Mapping[str, Any] | None = None,
    expected_mtp_post_manifest: Mapping[str, Any] | None = None,
    expected_mtp_graph_capture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one full report and return its shipcard-sized raw projection."""
    if arm not in ARMS:
        raise DSparkMatchedPerformanceError(f"unknown performance arm {arm!r}")
    _require_exact_keys(
        report,
        {
            "schema",
            "arm",
            "served_model",
            "started_at",
            "finished_at",
            "tool",
            "policy",
            "target_model_sha",
            "pre_manifest",
            "post_manifest",
            "graph_capture",
            "workload",
            "warmup_responses",
            "responses",
            "counters",
            "warm_decode",
            "memory",
            "mtp",
            "report_sha256",
        },
        where=f"{arm} performance report",
    )
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("arm") != arm
        or report.get("target_model_sha") != expected_target_sha
    ):
        raise DSparkMatchedPerformanceError(
            f"{arm} report schema/role/target identity differs"
        )
    if load_report(report_path) != dict(report):
        raise DSparkMatchedPerformanceError(
            f"{arm} report payload differs from its JSON file"
        )
    recorded_report_sha = _require_sha(
        report.get("report_sha256"), where=f"{arm} report digest"
    )
    if recorded_report_sha != canonical_sha256(_report_unstamped(report)):
        raise DSparkMatchedPerformanceError(f"{arm} report digest is stale")
    file_digest = file_sha256(report_path)
    started = _parse_time(report.get("started_at"), where=f"{arm} started_at")
    finished = _parse_time(report.get("finished_at"), where=f"{arm} finished_at")
    if finished <= started:
        raise DSparkMatchedPerformanceError(f"{arm} report chronology is invalid")
    tool = _validate_tool(report.get("tool"), verify_local_source=True)
    policy = _validate_policy(report.get("policy"))
    if _parse_time(policy["predeclared_at"], where="policy predeclared_at") > started:
        raise DSparkMatchedPerformanceError(
            "throughput threshold was not declared before measurement"
        )
    pre, post, stable_fingerprint = _validate_manifest_pair(
        report,
        arm=arm,
        expected_target_sha=expected_target_sha,
        expected_draft_sha=expected_draft_sha,
        expected_speculative_config=expected_speculative_config,
        expected_launch_options=expected_launch_options,
        expected_launch_switches=expected_launch_switches,
        requires_moe_marlin=requires_moe_marlin,
    )
    pre_created = _parse_time(pre.get("created"), where=f"{arm} pre created")
    post_created = _parse_time(post.get("created"), where=f"{arm} post created")
    if not (pre_created <= started < finished <= post_created):
        raise DSparkMatchedPerformanceError(
            f"{arm} serve snapshots do not bracket the measured workload"
        )
    if _parse_time(policy["predeclared_at"], where="policy predeclared_at") > pre_created:
        raise DSparkMatchedPerformanceError(
            "throughput threshold was not declared before the serve snapshot"
        )
    if arm == MTP_ARM:
        if (
            expected_mtp_pre_manifest is None
            or expected_mtp_post_manifest is None
            or dict(pre) != dict(expected_mtp_pre_manifest)
            or dict(post) != dict(expected_mtp_post_manifest)
        ):
            raise DSparkMatchedPerformanceError(
                "MTP throughput report is not the acceptance-gated serve session"
            )
    graph = _validate_graph_capture(report.get("graph_capture"), arm=arm)
    if arm == NO_MTP_ARM and (
        expected_no_mtp_graph_capture is None
        or graph != dict(expected_no_mtp_graph_capture)
    ):
        raise DSparkMatchedPerformanceError(
            "no-MTP throughput report graph evidence differs from its serve log"
        )
    if arm == MTP_ARM and (
        expected_mtp_graph_capture is None
        or graph != dict(expected_mtp_graph_capture)
    ):
        raise DSparkMatchedPerformanceError(
            "MTP throughput report graph evidence differs from pair attestation"
        )
    workload = _validate_workload(
        report.get("workload"), expected_workload=expected_workload
    )
    warmup = _validate_response_ledger(
        report.get("warmup_responses"), workload=workload, where="warmup responses"
    )
    responses = _validate_response_ledger(
        report.get("responses"), workload=workload, where="measured responses"
    )
    if [row["content_sha256"] for row in warmup] != [
        row["content_sha256"] for row in responses
    ]:
        raise DSparkMatchedPerformanceError(
            f"{arm} warmup and measured deterministic outputs differ"
        )
    counters = _validate_counters(report.get("counters"))
    throughput = report.get("warm_decode")
    if not isinstance(throughput, Mapping):
        raise DSparkMatchedPerformanceError("warm decode summary is missing")
    _require_exact_keys(
        throughput,
        {"output_tokens", "wall_seconds", "output_tokens_per_second"},
        where="warm decode summary",
    )
    wall = _finite(
        throughput.get("wall_seconds"), where="warm decode wall", positive=True
    )
    observed_tps = _finite(
        throughput.get("output_tokens_per_second"),
        where="warm decode throughput",
        positive=True,
    )
    chronology = (finished - started).total_seconds()
    if (
        throughput.get("output_tokens") != EXPECTED_OUTPUT_TOKENS
        or abs(wall - chronology) > WALL_CHRONOLOGY_ABS_TOLERANCE_SECONDS
        or not math.isclose(
            observed_tps,
            EXPECTED_OUTPUT_TOKENS / wall,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise DSparkMatchedPerformanceError(
            "warm decode throughput does not replay from exact output tokens and wall time"
        )
    memory = _validate_memory(
        report.get("memory"),
        report_started=started,
        report_finished=finished,
        pre_created=pre_created,
        post_created=post_created,
        graph_capture=graph,
    )
    draft_binding = pre.get("draft_artifact_binding")
    mtp = _validate_mtp_report_binding(
        report.get("mtp"),
        arm=arm,
        expected_draft_sha=expected_draft_sha,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        draft_binding=(dict(draft_binding) if isinstance(draft_binding, Mapping) else None),
    )
    if arm == MTP_ARM and (
        report.get("started_at") != expected_acceptance.get("started_at")
        or report.get("finished_at") != expected_acceptance.get("finished_at")
        or report.get("served_model") != expected_acceptance.get("served_model")
        or responses != expected_acceptance.get("responses")
    ):
        raise DSparkMatchedPerformanceError(
            "MTP throughput responses/interval are not the exact acceptance suite"
        )

    evidence: dict[str, Any] = {
        "schema": ARM_EVIDENCE_SCHEMA,
        "arm": arm,
        "served_model": report["served_model"],
        "started_at": report["started_at"],
        "finished_at": report["finished_at"],
        "tool": tool,
        "policy": policy,
        "target_model_sha": expected_target_sha,
        "target_artifact_binding": dict(pre["artifact_binding"]),
        "draft_artifact_binding": (
            dict(draft_binding) if isinstance(draft_binding, Mapping) else None
        ),
        "serve_session_id": pre["serve_session_id"],
        "stable_serve_fingerprint": stable_fingerprint,
        "performance_stack_fingerprint": pre.get(
            "performance_stack_fingerprint"
        ),
        "pre_created": pre["created"],
        "post_created": post["created"],
        "pre_manifest_sha256": canonical_sha256(pre),
        "post_manifest_sha256": canonical_sha256(post),
        "manifest_identity": _manifest_compact_identity(pre),
        "launch_contract": _launch_contract(
            expected_speculative_config=expected_speculative_config,
            expected_launch_options=expected_launch_options,
            expected_launch_switches=expected_launch_switches,
        ),
        "graph_capture": graph,
        "workload": workload,
        "warmup_responses": warmup,
        "responses": responses,
        "counters": counters,
        "warm_decode": dict(throughput),
        "memory": memory,
        "mtp": mtp,
        "report_payload_sha256": recorded_report_sha,
        "report_file_sha256": file_digest,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _validate_arm_evidence(
    payload: object,
    *,
    arm: str,
    expected_target_sha: str,
    expected_draft_sha: str,
    expected_workload: Mapping[str, Any],
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    expected_speculative_config: Mapping[str, Any] | None,
    expected_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError(f"{arm} compact evidence is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "arm",
            "served_model",
            "started_at",
            "finished_at",
            "tool",
            "policy",
            "target_model_sha",
            "target_artifact_binding",
            "draft_artifact_binding",
            "serve_session_id",
            "stable_serve_fingerprint",
            "performance_stack_fingerprint",
            "pre_created",
            "post_created",
            "pre_manifest_sha256",
            "post_manifest_sha256",
            "manifest_identity",
            "launch_contract",
            "graph_capture",
            "workload",
            "warmup_responses",
            "responses",
            "counters",
            "warm_decode",
            "memory",
            "mtp",
            "report_payload_sha256",
            "report_file_sha256",
            "evidence_sha256",
        },
        where=f"{arm} compact evidence",
    )
    recorded = _require_sha(
        payload.get("evidence_sha256"), where=f"{arm} compact evidence digest"
    )
    if recorded != canonical_sha256(_arm_evidence_unstamped(payload)):
        raise DSparkMatchedPerformanceError(f"{arm} compact evidence digest is stale")
    if (
        payload.get("schema") != ARM_EVIDENCE_SCHEMA
        or payload.get("arm") != arm
        or payload.get("target_model_sha") != expected_target_sha
    ):
        raise DSparkMatchedPerformanceError(f"{arm} compact role/target differs")
    target_binding = payload.get("target_artifact_binding")
    draft_binding = payload.get("draft_artifact_binding")
    if (
        not isinstance(target_binding, Mapping)
        or target_binding.get("model_sha") != expected_target_sha
        or target_binding.get("launch_model") != "/model"
    ):
        raise DSparkMatchedPerformanceError(
            f"{arm} compact target artifact binding differs"
        )
    if arm == MTP_ARM and (
        not isinstance(draft_binding, Mapping)
        or draft_binding.get("model_sha") != expected_draft_sha
        or draft_binding.get("launch_model") != "/draft"
    ):
        raise DSparkMatchedPerformanceError(
            "compact MTP draft artifact binding differs"
        )
    for key in (
        "serve_session_id",
        "stable_serve_fingerprint",
        "performance_stack_fingerprint",
        "pre_manifest_sha256",
        "post_manifest_sha256",
        "report_payload_sha256",
        "report_file_sha256",
    ):
        _require_sha(payload.get(key), where=f"{arm} {key}")
    started = _parse_time(payload.get("started_at"), where=f"{arm} started_at")
    finished = _parse_time(payload.get("finished_at"), where=f"{arm} finished_at")
    pre_created = _parse_time(payload.get("pre_created"), where=f"{arm} pre_created")
    post_created = _parse_time(payload.get("post_created"), where=f"{arm} post_created")
    if not (pre_created <= started < finished <= post_created):
        raise DSparkMatchedPerformanceError(f"{arm} compact chronology differs")
    _validate_tool(payload.get("tool"))
    policy = _validate_policy(payload.get("policy"))
    if _parse_time(policy["predeclared_at"], where="policy predeclared_at") > pre_created:
        raise DSparkMatchedPerformanceError("performance threshold was declared too late")
    workload = _validate_workload(
        payload.get("workload"), expected_workload=expected_workload
    )
    _validate_launch_contract(
        payload.get("launch_contract"),
        expected_speculative_config=expected_speculative_config,
        expected_launch_options=expected_launch_options,
        expected_launch_switches=expected_launch_switches,
    )
    warmup = _validate_response_ledger(
        payload.get("warmup_responses"), workload=workload, where="warmup responses"
    )
    responses = _validate_response_ledger(
        payload.get("responses"), workload=workload, where="measured responses"
    )
    if [row["content_sha256"] for row in warmup] != [
        row["content_sha256"] for row in responses
    ]:
        raise DSparkMatchedPerformanceError(
            f"{arm} compact deterministic outputs differ"
        )
    _validate_counters(payload.get("counters"))
    graph = _validate_graph_capture(payload.get("graph_capture"), arm=arm)
    throughput = payload.get("warm_decode")
    if not isinstance(throughput, Mapping):
        raise DSparkMatchedPerformanceError("compact throughput is missing")
    _require_exact_keys(
        throughput,
        {"output_tokens", "wall_seconds", "output_tokens_per_second"},
        where="compact warm decode",
    )
    wall = _finite(throughput.get("wall_seconds"), where="compact wall", positive=True)
    tps = _finite(
        throughput.get("output_tokens_per_second"),
        where="compact throughput",
        positive=True,
    )
    if (
        throughput.get("output_tokens") != EXPECTED_OUTPUT_TOKENS
        or abs(wall - (finished - started).total_seconds())
        > WALL_CHRONOLOGY_ABS_TOLERANCE_SECONDS
        or not math.isclose(
            tps, EXPECTED_OUTPUT_TOKENS / wall, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise DSparkMatchedPerformanceError("compact throughput arithmetic differs")
    _validate_memory_receipt(
        payload.get("memory"),
        report_started=started,
        report_finished=finished,
        pre_created=pre_created,
        post_created=post_created,
        graph_capture=graph,
    )
    _validate_mtp_receipt(
        payload.get("mtp"),
        arm=arm,
        expected_draft_sha=expected_draft_sha,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        draft_binding=(dict(draft_binding) if isinstance(draft_binding, Mapping) else None),
    )
    return dict(payload)


def _common_manifest_identity(
    no_mtp: Mapping[str, Any], mtp: Mapping[str, Any]
) -> dict[str, Any]:
    left = no_mtp.get("manifest_identity")
    right = mtp.get("manifest_identity")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise DSparkMatchedPerformanceError("arm manifest identity is missing")
    if dict(left) != dict(right):
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP image, runtime, target, environment, residency, host, "
            f"or GPU differs: {differing}"
        )
    if (
        left.get("target_artifact_binding")
        != no_mtp.get("target_artifact_binding")
        or right.get("target_artifact_binding")
        != mtp.get("target_artifact_binding")
    ):
        raise DSparkMatchedPerformanceError(
            "compact arm manifest identity differs from its target artifact binding"
        )
    _require_sha(
        left.get("dspark_serving_profile_sha256"),
        where="common DSpark serving-profile digest",
    )
    _require_sha(
        left.get("dspark_runtime_evidence_sha256"),
        where="common DSpark runtime-evidence digest",
    )
    return dict(left)


def build_matched_result(
    *,
    no_mtp_evidence: Mapping[str, Any],
    mtp_evidence: Mapping[str, Any],
    target_model_sha: str,
    draft_model_sha: str,
    runtime_pin: Mapping[str, Any],
    expected_workload: Mapping[str, Any],
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    expected_no_mtp_launch_options: Mapping[str, str],
    expected_mtp_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
    expected_mtp_speculative_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two validated arms and build the reciprocal receipt block."""
    no_mtp = _validate_arm_evidence(
        no_mtp_evidence,
        arm=NO_MTP_ARM,
        expected_target_sha=target_model_sha,
        expected_draft_sha=draft_model_sha,
        expected_workload=expected_workload,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        expected_speculative_config=None,
        expected_launch_options=expected_no_mtp_launch_options,
        expected_launch_switches=expected_launch_switches,
    )
    mtp = _validate_arm_evidence(
        mtp_evidence,
        arm=MTP_ARM,
        expected_target_sha=target_model_sha,
        expected_draft_sha=draft_model_sha,
        expected_workload=expected_workload,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        expected_speculative_config=expected_mtp_speculative_config,
        expected_launch_options=expected_mtp_launch_options,
        expected_launch_switches=expected_launch_switches,
    )
    if no_mtp["tool"] != mtp["tool"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP reports were not generated by the same exact tool commit"
        )
    if no_mtp["policy"] != mtp["policy"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP reports do not share one predeclared release policy"
        )
    if no_mtp["workload"] != mtp["workload"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP reports do not share one exact workload"
        )
    if no_mtp["served_model"] != mtp["served_model"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP reports do not share one served-model configuration"
        )
    no_options = dict(no_mtp["launch_contract"]["options"])
    mtp_options = dict(mtp["launch_contract"]["options"])
    no_compilation = no_options.pop("--compilation-config", None)
    mtp_compilation = mtp_options.pop("--compilation-config", None)
    no_speculative = no_options.pop("--speculative-config", None)
    mtp_speculative = mtp_options.pop("--speculative-config", None)
    if (
        no_options != mtp_options
        or no_mtp["launch_contract"]["switches"]
        != mtp["launch_contract"]["switches"]
        or no_speculative is not None
        or mtp_speculative is None
        or no_compilation is None
        or mtp_compilation is None
    ):
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP launch contracts differ outside speculative graphing"
        )
    if no_mtp["target_artifact_binding"] != mtp["target_artifact_binding"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP reports do not serve the same exact target artifact"
        )
    common_identity = _common_manifest_identity(no_mtp, mtp)
    if common_identity.get("gridbook_runtime_pin") != dict(runtime_pin):
        raise DSparkMatchedPerformanceError(
            "matched arm manifests name a different Gridbook runtime pin"
        )
    if no_mtp["serve_session_id"] == mtp["serve_session_id"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP and MTP arms must be distinct live server sessions"
        )
    if no_mtp["report_file_sha256"] == mtp["report_file_sha256"]:
        raise DSparkMatchedPerformanceError(
            "no-MTP and MTP arms cannot reuse one report file"
        )
    no_rows = no_mtp["responses"]
    mtp_rows = mtp["responses"]
    if [row["prompt_tokens"] for row in no_rows] != [
        row["prompt_tokens"] for row in mtp_rows
    ] or [row["content_sha256"] for row in no_rows] != [
        row["content_sha256"] for row in mtp_rows
    ]:
        raise DSparkMatchedPerformanceError(
            "no-MTP/MTP measured tokenization or deterministic outputs differ"
        )
    no_tps = float(no_mtp["warm_decode"]["output_tokens_per_second"])
    mtp_tps = float(mtp["warm_decode"]["output_tokens_per_second"])
    ratio = mtp_tps / no_tps
    absolute_delta = mtp_tps - no_tps
    threshold = float(
        no_mtp["policy"]["minimum_mtp_to_no_mtp_throughput_ratio"]
    )
    passed = ratio >= threshold
    comparison = {
        "no_mtp_warm_output_tokens_per_second": no_tps,
        "mtp_warm_output_tokens_per_second": mtp_tps,
        "mtp_to_no_mtp_ratio": ratio,
        "mtp_minus_no_mtp_tokens_per_second": absolute_delta,
        "minimum_ratio": threshold,
        "passed": passed,
    }
    if not passed:
        raise DSparkMatchedPerformanceError(
            "K12 MTP warm decode throughput regresses the matched no-MTP arm: "
            f"ratio={ratio:.6f} < {threshold:.6f}"
        )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "target_model_sha": target_model_sha,
        "draft_model_sha": draft_model_sha,
        "runtime_pin": dict(runtime_pin),
        "tool": dict(no_mtp["tool"]),
        "policy": dict(no_mtp["policy"]),
        "workload": dict(no_mtp["workload"]),
        "common_manifest_identity": common_identity,
        "common_manifest_identity_sha256": canonical_sha256(common_identity),
        "no_mtp": no_mtp,
        "mtp": mtp,
        "comparison": comparison,
    }
    result["evidence_sha256"] = canonical_sha256(result)
    return result


def validate_matched_result(
    payload: object,
    *,
    expected_target_sha: str,
    expected_draft_sha: str,
    expected_runtime_pin: Mapping[str, Any],
    expected_workload: Mapping[str, Any],
    expected_acceptance: Mapping[str, Any],
    expected_routes: Mapping[str, Any],
    expected_no_mtp_launch_options: Mapping[str, str],
    expected_mtp_launch_options: Mapping[str, str],
    expected_launch_switches: set[str] | frozenset[str],
    expected_mtp_speculative_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all paired arithmetic and raw compact evidence from a shipcard."""
    if not isinstance(payload, Mapping):
        raise DSparkMatchedPerformanceError("matched performance evidence is missing")
    _require_exact_keys(
        payload,
        {
            "schema",
            "target_model_sha",
            "draft_model_sha",
            "runtime_pin",
            "tool",
            "policy",
            "workload",
            "common_manifest_identity",
            "common_manifest_identity_sha256",
            "no_mtp",
            "mtp",
            "comparison",
            "evidence_sha256",
        },
        where="matched performance evidence",
    )
    recorded = _require_sha(
        payload.get("evidence_sha256"), where="matched performance digest"
    )
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256")
    if recorded != canonical_sha256(unstamped):
        raise DSparkMatchedPerformanceError("matched performance digest is stale")
    if (
        payload.get("schema") != RESULT_SCHEMA
        or payload.get("target_model_sha") != expected_target_sha
        or payload.get("draft_model_sha") != expected_draft_sha
        or payload.get("runtime_pin") != dict(expected_runtime_pin)
    ):
        raise DSparkMatchedPerformanceError(
            "matched performance artifact/runtime identity differs"
        )
    rebuilt = build_matched_result(
        no_mtp_evidence=payload["no_mtp"],
        mtp_evidence=payload["mtp"],
        target_model_sha=expected_target_sha,
        draft_model_sha=expected_draft_sha,
        runtime_pin=expected_runtime_pin,
        expected_workload=expected_workload,
        expected_acceptance=expected_acceptance,
        expected_routes=expected_routes,
        expected_no_mtp_launch_options=expected_no_mtp_launch_options,
        expected_mtp_launch_options=expected_mtp_launch_options,
        expected_launch_switches=expected_launch_switches,
        expected_mtp_speculative_config=expected_mtp_speculative_config,
    )
    if rebuilt != dict(payload):
        raise DSparkMatchedPerformanceError(
            "matched performance summaries do not replay from arm evidence"
        )
    return rebuilt


def load_report(path: str | Path) -> dict[str, Any]:
    """Load strict JSON without silently accepting duplicate keys/NaNs."""
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DSparkMatchedPerformanceError(
                    f"{path}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DSparkMatchedPerformanceError(
                    f"{path}: non-finite JSON number {value}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DSparkMatchedPerformanceError(
            f"{path}: performance report is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DSparkMatchedPerformanceError(f"{path}: expected JSON object")
    return payload
