#!/usr/bin/env python3
"""Production DSpark target+draft endpoint and paired-shipcard gate.

The native Gridbook gates prove the target checkpoint in isolation.  This gate
proves the separate claim that one exact target and one exact, production-
attested K12 DSpark sidecar were loaded together, speculative decoding really
executed, the first draft position retained useful acceptance on an
uncontaminated fixed-token workload, and the exact MTP stack did not regress a
matched compiled target-only serve or violate the production 256K/headroom
policy.

The source-closed arm collector owns process snapshots and the release
workload; ``run-suite`` remains a diagnostic entry point and cannot author a
release performance report::

  python -m prismaquant.dspark_matched_performance_collector declare-policy ...
  python -m prismaquant.dspark_matched_performance_collector start-sampler ...
  # Start the separately specified vLLM serve only after the sampler is ready.
  python -m prismaquant.dspark_matched_performance_collector collect-arm ...
  python -m prismaquant.validate_dspark_target_draft attest \
      --no-mtp-performance-report ... --mtp-performance-report ... \
      --baseline-comparison ...

``attest`` fills the optional ``mtp.dspark`` slot in *both* shipcards.  Default
verification replays every non-null optional claim, and draft-sidecar
provenance requires the slot even if the mutable card value is absent/null.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
import urllib.parse
import urllib.request

from .dspark_matched_performance import (
    DSparkMatchedPerformanceError,
    MTP_ARM,
    NO_MTP_ARM,
    WORKLOAD_SCHEMA as DSPARK_MATCHED_WORKLOAD_SCHEMA,
    build_matched_result,
    load_report as load_performance_report,
    validate_arm_report,
    validate_matched_result,
)
from .dspark_serving_profile import (
    DSPARK_IMAGE as DSV4_SPARK_VLLM_IMAGE,
    DSPARK_VLLM_VERSION as DSV4_SPARK_VLLM_VERSION,
    DSparkServingProfileError,
    collect_route_census,
    validate_baseline_comparison_evidence,
    validate_dspark_serve_manifest,
    validate_route_census,
    validate_runtime_evidence,
    validate_serving_profile_receipt,
)

from .shipcard import (
    assert_weight_stat_attestation,
    compute_model_sha,
    ensure_optional_slot,
    fill_slot,
    git_provenance,
    load_shipcard,
    make_record,
)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _endpoint_api():
    # Acceptance collection is intentionally stdlib/lightweight.  Artifact
    # decoding pulls in the quantization stack and is needed only by attest or
    # local shipcard replay, so do not retain it in the live measurement client.
    from . import validate_cb_endpoint

    return validate_cb_endpoint


def _gridbook_runtime_pin():
    return _endpoint_api()._gridbook_runtime_pin()


def validate_cb_artifact(*args, **kwargs):
    return _endpoint_api().validate_cb_artifact(*args, **kwargs)


def validate_cb_artifact_decode_contract(*args, **kwargs):
    return _endpoint_api().validate_cb_artifact_decode_contract(*args, **kwargs)


def validate_dspark_production_render_attestation(*args, **kwargs):
    return _endpoint_api().validate_dspark_production_render_attestation(
        *args, **kwargs
    )


DSPARK_ACCEPTANCE_SCHEMA = "prismaquant.dspark_acceptance_suite.v1"
DSPARK_WORKLOAD_SCHEMA = "prismaquant.dspark_acceptance_workload.v1"
DSPARK_PAIR_RESULT_SCHEMA = "prismaquant.dspark_target_draft_validation.v1"
DSPARK_SHIPCARD_TOOL = "validate_dspark_target_draft.py"
DSPARK_SHIPCARD_SLOT = "mtp.dspark"
DSPARK_DRAFT_FORMAT = "NVFP4_CB_K12"
DSPARK_NUM_SPECULATIVE_TOKENS = 5
DSPARK_POSITION_ZERO_MINIMUM = 0.60
DSPARK_MAX_TOKENS = 128
DSPARK_EXPECTED_GENERATION_TOKENS = 1024
DSPARK_MODEL_LEN = 262144
DSPARK_KV_CACHE_BYTES = 1717986918
DSPARK_GRAPH_COMPILATION_CONFIG = (
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
    '"cudagraph_capture_sizes":[5,6],"max_cudagraph_capture_size":6}'
)
DSPARK_NO_MTP_GRAPH_COMPILATION_CONFIG = (
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
    '"cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}'
)
DSPARK_SPECULATIVE_CONFIG: Mapping[str, Any] = {
    "method": "dspark",
    "model": "/draft",
    "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
    "draft_sample_method": "greedy",
    "quantization": "gridbook",
    "moe_backend": "marlin",
    "kv_cache_dtype": "fp8",
    "max_model_len": DSPARK_MODEL_LEN,
}
DSPARK_PROMPTS = (
    "Explain how virtual memory works, using technically precise prose and concrete examples.",
    "Describe the major stages of photosynthesis and explain where each stage occurs in a plant cell.",
    "Compare optimistic and pessimistic concurrency control in database systems, including practical tradeoffs.",
    "Explain why the sky appears blue during the day and often red near sunset.",
    "Describe how a compiler transforms source code into an executable program, from parsing through linking.",
    "Explain the difference between symmetric and asymmetric cryptography and how hybrid protocols use both.",
    "Give a rigorous but accessible explanation of gradient descent and the role of the learning rate.",
    "Explain how TCP establishes a connection and provides reliable ordered delivery over an unreliable network.",
)
DSPARK_PROMPT_SHA256 = tuple(
    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    for prompt in DSPARK_PROMPTS
)
DSPARK_PROMPT_CORPUS_SHA256 = _canonical_json_sha256(
    list(DSPARK_PROMPT_SHA256)
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GRAPH_CAPTURE_RE = re.compile(
    r"Graph capturing finished in [0-9]+ secs, took -?[0-9.]+ GiB"
)
DSPARK_LAUNCH_SWITCHES = frozenset({
    "--trust-remote-code",
    "--no-enable-prefix-caching",
    "--enable-chunked-prefill",
    "--enable-auto-tool-choice",
})
_SNAPSHOT_KEYS = (
    "generation_tokens",
    "drafts",
    "draft_tokens",
    "accepted_tokens",
    "accepted_position_0",
    "accepted_position_1",
    "accepted_position_2",
    "accepted_position_3",
    "accepted_position_4",
)


class DSparkValidationError(RuntimeError):
    """The target+draft pair did not satisfy the DSpark release gate."""


def _canonical_sha256(payload: object) -> str:
    return _canonical_json_sha256(payload)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    if set(payload) != expected:
        raise DSparkValidationError(
            f"{where} keys differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _finite(value: object, *, where: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DSparkValidationError(f"{where} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise DSparkValidationError(f"{where} is not finite and valid")
    return result


def _parse_time(value: object, *, where: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DSparkValidationError(f"{where} is not canonical UTC time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DSparkValidationError(f"{where} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise DSparkValidationError(f"{where} has no timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.port is None
    ):
        raise DSparkValidationError(
            "DSpark release acceptance must target an explicit loopback HTTP origin"
        )
    host = f"[{parsed.hostname}]" if ":" in str(parsed.hostname) else parsed.hostname
    return f"http://{host}:{parsed.port}"


def _http_json(
    base_url: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        base_url + path,
        data=(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if payload is not None else None
        ),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        elapsed = time.monotonic() - started
        if response.status != 200:
            raise DSparkValidationError(
                f"{request.method} {path} returned HTTP {response.status}"
            )
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise DSparkValidationError(f"{path} did not return a JSON object")
    return decoded, elapsed


def _http_text(base_url: str, path: str) -> str:
    request = urllib.request.Request(base_url + path, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        if response.status != 200:
            raise DSparkValidationError(
                f"GET {path} returned HTTP {response.status}"
            )
    return body.decode("utf-8")


def _metric_value(
    text: str, metric: str, *, position: int | None = None
) -> float:
    prefix = f"vllm:{metric}{{"
    matches: list[float] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        if position is not None and f'position="{position}"' not in line:
            continue
        if position is None and "position=" in line:
            continue
        try:
            value = float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError) as exc:
            raise DSparkValidationError(
                f"Prometheus metric {metric} is malformed"
            ) from exc
        matches.append(value)
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise DSparkValidationError(
            f"expected exactly one finite vllm:{metric} sample, got {len(matches)}"
        )
    return matches[0]


def _metric_snapshot(base_url: str) -> dict[str, float]:
    text = _http_text(base_url, "/metrics")
    result = {
        "generation_tokens": _metric_value(text, "generation_tokens_total"),
        "drafts": _metric_value(text, "spec_decode_num_drafts_total"),
        "draft_tokens": _metric_value(
            text, "spec_decode_num_draft_tokens_total"
        ),
        "accepted_tokens": _metric_value(
            text, "spec_decode_num_accepted_tokens_total"
        ),
    }
    for position in range(DSPARK_NUM_SPECULATIVE_TOKENS):
        result[f"accepted_position_{position}"] = _metric_value(
            text,
            "spec_decode_num_accepted_tokens_per_pos_total",
            position=position,
        )
    return result


def _served_model(base_url: str) -> str:
    payload, _ = _http_json(base_url, "/v1/models")
    rows = payload.get("data")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or not isinstance(rows[0].get("id"), str)
        or not rows[0].get("id")
    ):
        raise DSparkValidationError("/v1/models must expose exactly one model")
    return str(rows[0]["id"])


def run_acceptance_suite(base_url: str) -> dict[str, Any]:
    """Run the fixed 8x128 workload and return its digestable raw evidence."""
    base_url = _canonical_base_url(base_url)
    served_model = _served_model(base_url)
    before = _metric_snapshot(base_url)
    started_at = _utc_now()
    wall_started = time.monotonic()
    responses: list[dict[str, Any]] = []
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
            or not isinstance(usage, Mapping)
            or not isinstance(choices[0].get("message"), Mapping)
            or not isinstance(choices[0]["message"].get("content"), str)
        ):
            raise DSparkValidationError(
                f"acceptance response {index} is not one chat completion"
            )
        content = str(choices[0]["message"]["content"])
        responses.append({
            "index": index,
            "prompt_sha256": DSPARK_PROMPT_SHA256[index],
            "content_sha256": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
            "finish_reason": choices[0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "request_seconds": elapsed,
        })
    wall_seconds = time.monotonic() - wall_started
    finished_at = _utc_now()
    after = _metric_snapshot(base_url)
    model_after = _served_model(base_url)
    if model_after != served_model:
        raise DSparkValidationError("served model changed during acceptance suite")
    delta = {key: after[key] - before[key] for key in _SNAPSHOT_KEYS}
    drafts = delta["drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    accepted_per_position = [
        delta[f"accepted_position_{position}"]
        for position in range(DSPARK_NUM_SPECULATIVE_TOKENS)
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
        "before": before,
        "after": after,
        "delta": delta,
        "wall_seconds": wall_seconds,
        "served_output_tokens_per_second": (
            delta["generation_tokens"] / wall_seconds
        ),
        "responses": responses,
        "acceptance": {
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "accepted_tokens_per_position": accepted_per_position,
            "aggregate_rate": accepted / draft_tokens,
            "accepted_tokens_per_cycle": accepted / drafts,
            "mean_acceptance_length": 1.0 + accepted / drafts,
            "per_position_rates": [
                value / drafts for value in accepted_per_position
            ],
            "position_zero_gate_minimum": DSPARK_POSITION_ZERO_MINIMUM,
            "position_zero_gate_passed": (
                accepted_per_position[0] / drafts
                >= DSPARK_POSITION_ZERO_MINIMUM
            ),
        },
    }
    validate_acceptance_suite(result, expected_served_model=served_model)
    return result


def validate_acceptance_suite(
    payload: Mapping[str, Any], *, expected_served_model: str
) -> dict[str, Any]:
    """Replay workload closure and every speculative-metric calculation."""
    expected_keys = {
        "schema", "started_at", "finished_at", "base_url", "served_model",
        "workload", "before", "after", "delta", "wall_seconds",
        "served_output_tokens_per_second", "responses", "acceptance",
    }
    _require_exact_keys(payload, expected_keys, where="DSpark acceptance suite")
    if payload.get("schema") != DSPARK_ACCEPTANCE_SCHEMA:
        raise DSparkValidationError("unsupported DSpark acceptance schema")
    if payload.get("served_model") != expected_served_model:
        raise DSparkValidationError("acceptance suite names a different model")
    _canonical_base_url(str(payload.get("base_url", "")))
    started = _parse_time(payload.get("started_at"), where="suite started_at")
    finished = _parse_time(payload.get("finished_at"), where="suite finished_at")
    if finished <= started:
        raise DSparkValidationError("acceptance suite chronology is invalid")

    expected_workload = {
        "schema": DSPARK_WORKLOAD_SCHEMA,
        "prompt_count": len(DSPARK_PROMPTS),
        "prompt_sha256": list(DSPARK_PROMPT_SHA256),
        "prompt_corpus_sha256": DSPARK_PROMPT_CORPUS_SHA256,
        "max_tokens_per_prompt": DSPARK_MAX_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    if payload.get("workload") != expected_workload:
        raise DSparkValidationError("DSpark acceptance workload differs")

    snapshots: dict[str, dict[str, float]] = {}
    for label in ("before", "after", "delta"):
        raw = payload.get(label)
        if not isinstance(raw, Mapping) or set(raw) != set(_SNAPSHOT_KEYS):
            raise DSparkValidationError(f"acceptance {label} metric set differs")
        snapshots[label] = {
            key: _finite(raw[key], where=f"{label}.{key}")
            for key in _SNAPSHOT_KEYS
        }
    for key in _SNAPSHOT_KEYS:
        expected_delta = snapshots["after"][key] - snapshots["before"][key]
        observed_delta = snapshots["delta"][key]
        if not math.isclose(
            observed_delta, expected_delta, rel_tol=0.0, abs_tol=1e-9
        ) or not observed_delta.is_integer():
            raise DSparkValidationError(
                f"acceptance delta arithmetic for {key} differs"
            )
    delta = snapshots["delta"]
    if delta["generation_tokens"] != DSPARK_EXPECTED_GENERATION_TOKENS:
        raise DSparkValidationError(
            "acceptance metric interval was contaminated or incomplete"
        )
    drafts = delta["drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]
    positions = [
        delta[f"accepted_position_{position}"]
        for position in range(DSPARK_NUM_SPECULATIVE_TOKENS)
    ]
    if (
        drafts <= 0
        or draft_tokens != drafts * DSPARK_NUM_SPECULATIVE_TOKENS
        or accepted != math.fsum(positions)
        or accepted > draft_tokens
        or positions != sorted(positions, reverse=True)
        or any(value > drafts for value in positions)
    ):
        raise DSparkValidationError(
            "DSpark speculative draft/acceptance counters do not reconcile"
        )

    responses = payload.get("responses")
    if not isinstance(responses, list) or len(responses) != len(DSPARK_PROMPTS):
        raise DSparkValidationError("acceptance response ledger is incomplete")
    for index, row in enumerate(responses):
        if not isinstance(row, Mapping):
            raise DSparkValidationError(f"acceptance response {index} is malformed")
        _require_exact_keys(
            row,
            {
                "index", "prompt_sha256", "content_sha256", "finish_reason",
                "prompt_tokens", "completion_tokens", "request_seconds",
            },
            where=f"acceptance response {index}",
        )
        if (
            row.get("index") != index
            or row.get("prompt_sha256") != DSPARK_PROMPT_SHA256[index]
            or _SHA256_RE.fullmatch(str(row.get("content_sha256", ""))) is None
            or row.get("finish_reason") != "length"
            or isinstance(row.get("prompt_tokens"), bool)
            or not isinstance(row.get("prompt_tokens"), int)
            or int(row.get("prompt_tokens", 0)) <= 0
            or row.get("completion_tokens") != DSPARK_MAX_TOKENS
        ):
            raise DSparkValidationError(
                f"acceptance response {index} differs from fixed-token contract"
            )
        _finite(
            row.get("request_seconds"),
            where=f"response {index} request_seconds",
            positive=True,
        )

    wall = _finite(payload.get("wall_seconds"), where="suite wall", positive=True)
    chronology = (finished - started).total_seconds()
    if abs(wall - chronology) > 2.0:
        raise DSparkValidationError(
            "acceptance monotonic and UTC wall intervals differ excessively"
        )
    throughput = _finite(
        payload.get("served_output_tokens_per_second"),
        where="suite throughput",
        positive=True,
    )
    if not math.isclose(
        throughput,
        DSPARK_EXPECTED_GENERATION_TOKENS / wall,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise DSparkValidationError("acceptance throughput arithmetic differs")

    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise DSparkValidationError("acceptance summary is missing")
    _require_exact_keys(
        acceptance,
        {
            "drafts", "draft_tokens", "accepted_tokens",
            "accepted_tokens_per_position", "aggregate_rate",
            "accepted_tokens_per_cycle", "mean_acceptance_length",
            "per_position_rates", "position_zero_gate_minimum",
            "position_zero_gate_passed",
        },
        where="acceptance summary",
    )
    rates = [value / drafts for value in positions]
    expected_values = {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "accepted_tokens_per_position": positions,
        "aggregate_rate": accepted / draft_tokens,
        "accepted_tokens_per_cycle": accepted / drafts,
        "mean_acceptance_length": 1.0 + accepted / drafts,
        "per_position_rates": rates,
        "position_zero_gate_minimum": DSPARK_POSITION_ZERO_MINIMUM,
        "position_zero_gate_passed": rates[0] >= DSPARK_POSITION_ZERO_MINIMUM,
    }
    if dict(acceptance) != expected_values or acceptance.get(
        "position_zero_gate_passed"
    ) is not True:
        raise DSparkValidationError(
            "DSpark acceptance summary/gate does not replay exactly"
        )
    return dict(payload)


def validate_dspark_graph_capture_log(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", "replace")
    forbidden = (
        "Skipping CUDA graph capture",
        "Overriding cudagraph_mode to PIECEWISE",
        "Overriding cudagraph_mode from FULL_DECODE_ONLY to PIECEWISE",
    )
    marker = _GRAPH_CAPTURE_RE.search(text)
    if (
        marker is None
        or any(value in text for value in forbidden)
        or "'cudagraph_capture_sizes': [5, 6]" not in text
        or "'max_cudagraph_capture_size': 6" not in text
        or "SpeculativeConfig(method='dspark', model='/draft', num_spec_tokens=5)"
        not in text
    ):
        raise DSparkValidationError(
            "serve log does not prove the exact DSpark [5,6] graph capture"
        )
    return {
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_marker": marker.group(0),
        "capture_sizes": [5, 6],
    }


def validate_no_mtp_graph_capture_log(path: str | Path) -> dict[str, Any]:
    """Prove the matched target-only arm compiled its one-token decode graph."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", "replace")
    forbidden = (
        "Skipping CUDA graph capture",
        "Overriding cudagraph_mode to PIECEWISE",
        "Overriding cudagraph_mode from FULL_DECODE_ONLY to PIECEWISE",
        "SpeculativeConfig(",
        "method='dspark'",
    )
    marker = _GRAPH_CAPTURE_RE.search(text)
    if (
        marker is None
        or any(value in text for value in forbidden)
        or text.count(marker.group(0)) != 1
        or "'cudagraph_capture_sizes': [1]" not in text
        or "'max_cudagraph_capture_size': 1" not in text
        or "'cudagraph_capture_sizes': [5, 6]" in text
    ):
        raise DSparkValidationError(
            "serve log does not prove the exact non-speculative [1] graph capture"
        )
    return {
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_marker": marker.group(0),
        "capture_sizes": [1],
    }


def _expected_launch_options(served_model: str) -> dict[str, str]:
    return {
        "--served-model-name": served_model,
        "--host": "0.0.0.0",
        "--port": "8000",
        "--tokenizer-mode": "deepseek_v4",
        "--generation-config": "vllm",
        "--quantization": "gridbook",
        "--tensor-parallel-size": "1",
        "--kv-cache-dtype": "fp8",
        "--kv-cache-memory-bytes": str(DSPARK_KV_CACHE_BYTES),
        "--max-model-len": str(DSPARK_MODEL_LEN),
        "--max-num-seqs": "1",
        "--max-num-batched-tokens": "512",
        "--gpu-memory-utilization": "0.90",
        "--moe-backend": "marlin",
        "--tool-call-parser": "deepseek_v4",
        "--speculative-config": json.dumps(
            DSPARK_SPECULATIVE_CONFIG, separators=(",", ":")
        ),
        "--default-chat-template-kwargs": '{"enable_thinking":false}',
        "--compilation-config": DSPARK_GRAPH_COMPILATION_CONFIG,
    }


def _expected_no_mtp_launch_options(served_model: str) -> dict[str, str]:
    options = _expected_launch_options(served_model)
    options.pop("--speculative-config")
    options["--compilation-config"] = DSPARK_NO_MTP_GRAPH_COMPILATION_CONFIG
    return options


def _matched_performance_workload() -> dict[str, Any]:
    return {
        "schema": DSPARK_MATCHED_WORKLOAD_SCHEMA,
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


def _validate_artifact_binding(
    binding: object,
    *,
    expected_model_sha: str,
    expected_launch_model: str,
    where: str,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise DSparkValidationError(f"{where} is missing")
    _require_exact_keys(
        binding,
        {
            "schema", "resolved_path", "launch_model", "model_sha",
            "artifact_inventory_sha256", "artifact_bytes",
        },
        where=where,
    )
    if (
        binding.get("schema") != "prismaquant.served_artifact_binding/1"
        or binding.get("launch_model") != expected_launch_model
        or binding.get("model_sha") != expected_model_sha
        or _SHA256_RE.fullmatch(
            str(binding.get("artifact_inventory_sha256", ""))
        ) is None
        or isinstance(binding.get("artifact_bytes"), bool)
        or not isinstance(binding.get("artifact_bytes"), int)
        or int(binding.get("artifact_bytes", 0)) <= 0
        or not isinstance(binding.get("resolved_path"), str)
        or not Path(str(binding.get("resolved_path"))).is_absolute()
    ):
        raise DSparkValidationError(f"{where} differs from exact artifact")
    return dict(binding)


def _validate_card_identity(
    shipcard_path: str | Path, model_dir: str | Path, *, expected_sha: str
) -> dict[str, Any]:
    card = load_shipcard(shipcard_path)
    if card.get("model_sha") != expected_sha:
        raise DSparkValidationError(
            f"{shipcard_path}: shipcard names a different artifact"
        )
    assert_weight_stat_attestation(card, model_dir)
    return card


def build_pair_result(
    *,
    target_model_dir: str | Path,
    draft_model_dir: str | Path,
    target_shipcard: str | Path,
    draft_shipcard: str | Path,
    pre_manifest: Mapping[str, Any],
    post_manifest: Mapping[str, Any],
    pre_manifest_path: str | Path,
    post_manifest_path: str | Path,
    acceptance: Mapping[str, Any],
    acceptance_path: str | Path,
    serve_log_path: str | Path,
    no_mtp_serve_log_path: str | Path,
    no_mtp_performance_report: Mapping[str, Any],
    no_mtp_performance_report_path: str | Path,
    mtp_performance_report: Mapping[str, Any],
    mtp_performance_report_path: str | Path,
    baseline_comparison: Mapping[str, Any],
    baseline_comparison_path: str | Path,
) -> dict[str, Any]:
    """Validate every pair input and build one self-contained receipt."""
    for label, path, expected in (
        ("pre manifest", pre_manifest_path, pre_manifest),
        ("post manifest", post_manifest_path, post_manifest),
        ("acceptance", acceptance_path, acceptance),
        (
            "no-MTP performance report",
            no_mtp_performance_report_path,
            no_mtp_performance_report,
        ),
        (
            "MTP performance report",
            mtp_performance_report_path,
            mtp_performance_report,
        ),
        (
            "baseline comparison",
            baseline_comparison_path,
            baseline_comparison,
        ),
    ):
        try:
            observed = load_performance_report(path)
        except DSparkMatchedPerformanceError as exc:
            raise DSparkValidationError(f"{label} is unreadable: {exc}") from exc
        if observed != dict(expected):
            raise DSparkValidationError(
                f"{label} payload differs from its digest-bound JSON path"
            )
    runtime_pin = _gridbook_runtime_pin()  # fail closed while 0.8.6 is pending
    target_quant = validate_cb_artifact(target_model_dir)
    draft_quant = validate_cb_artifact(draft_model_dir)
    target_decode = validate_cb_artifact_decode_contract(
        target_model_dir, target_quant, runtime_pin=runtime_pin
    )
    draft_decode = validate_cb_artifact_decode_contract(
        draft_model_dir, draft_quant, runtime_pin=runtime_pin
    )
    if target_decode.get("schema") != "prismaquant.cb_artifact_decode_contract.v1":
        raise DSparkValidationError(
            "DSpark target must be the complete body/source-overlay artifact"
        )
    if (
        draft_decode.get("schema")
        != "prismaquant.cb_artifact_decode_contract.v2"
        or draft_decode.get("mode") != "quantized_dspark_cb_sidecar"
    ):
        raise DSparkValidationError(
            "DSpark draft is not the complete quantized sidecar"
        )
    draft_render = validate_dspark_production_render_attestation(draft_quant)
    draft_formats = {
        group.get("format")
        for group in (draft_quant.get("config_groups") or {}).values()
        if isinstance(group, Mapping) and group.get("format") != "source-passthrough"
    }
    if draft_formats != {DSPARK_DRAFT_FORMAT}:
        raise DSparkValidationError(
            f"DSpark draft must be uniform {DSPARK_DRAFT_FORMAT}, got {draft_formats}"
        )

    target_sha = compute_model_sha(target_model_dir)
    draft_sha = compute_model_sha(draft_model_dir)
    _validate_card_identity(target_shipcard, target_model_dir, expected_sha=target_sha)
    _validate_card_identity(draft_shipcard, draft_model_dir, expected_sha=draft_sha)
    target_source = (target_quant.get("provenance") or {}).get(
        "source_model_identity"
    )
    draft_source = draft_render.get("source_model_identity")
    if target_source != draft_source:
        raise DSparkValidationError(
            "target and DSpark draft derive from different source checkpoints"
        )

    served_model = str(acceptance.get("served_model", ""))
    expected_options = _expected_launch_options(served_model)
    expected_switches = DSPARK_LAUNCH_SWITCHES
    manifest_fingerprints: list[str] = []
    for phase, manifest in (("pre", pre_manifest), ("post", post_manifest)):
        if manifest.get("attestation_phase") != phase:
            raise DSparkValidationError(
                f"DSpark {phase} manifest has wrong chronology role"
            )
        try:
            fingerprint = validate_dspark_serve_manifest(
                manifest,
                arm="graph",
                expected_served_model=served_model,
                requires_moe_marlin=True,
                expected_model_sha=target_sha,
                expected_speculative_config=DSPARK_SPECULATIVE_CONFIG,
                expected_launch_options=expected_options,
                expected_launch_switches=expected_switches,
            )
        except DSparkServingProfileError as exc:
            raise DSparkValidationError(
                f"DSpark {phase} serve manifest differs: {exc}"
            ) from exc
        manifest_fingerprints.append(fingerprint)
        _validate_artifact_binding(
            manifest.get("artifact_binding"),
            expected_model_sha=target_sha,
            expected_launch_model="/model",
            where=f"{phase} target artifact binding",
        )
        _validate_artifact_binding(
            manifest.get("draft_artifact_binding"),
            expected_model_sha=draft_sha,
            expected_launch_model="/draft",
            where=f"{phase} draft artifact binding",
        )

    if manifest_fingerprints[0] != manifest_fingerprints[1]:
        raise DSparkValidationError(
            "DSpark pre/post stable serve fingerprints differ; resident "
            "extensions, packages, capabilities, or another serving-stack "
            "field changed during acceptance"
        )
    if (
        pre_manifest.get("serve_session_id")
        != post_manifest.get("serve_session_id")
        or pre_manifest.get("processes") != post_manifest.get("processes")
        or pre_manifest.get("artifact_binding")
        != post_manifest.get("artifact_binding")
        or pre_manifest.get("draft_artifact_binding")
        != post_manifest.get("draft_artifact_binding")
        or pre_manifest.get("gridbook_runtime_pin")
        != post_manifest.get("gridbook_runtime_pin")
    ):
        raise DSparkValidationError(
            "DSpark pre/post manifests do not bracket one unchanged server session"
        )
    validated_acceptance = validate_acceptance_suite(
        acceptance, expected_served_model=served_model
    )
    manifest_pre_time = _parse_time(
        pre_manifest.get("created"), where="pre manifest created"
    )
    manifest_post_time = _parse_time(
        post_manifest.get("created"), where="post manifest created"
    )
    suite_start = _parse_time(
        validated_acceptance.get("started_at"), where="suite start"
    )
    suite_finish = _parse_time(
        validated_acceptance.get("finished_at"), where="suite finish"
    )
    if not (manifest_pre_time <= suite_start < suite_finish <= manifest_post_time):
        raise DSparkValidationError(
            "DSpark serve snapshots do not chronologically bracket acceptance"
        )
    graph_capture = validate_dspark_graph_capture_log(serve_log_path)
    no_mtp_graph_capture = validate_no_mtp_graph_capture_log(
        no_mtp_serve_log_path
    )

    try:
        profile_receipt = validate_serving_profile_receipt(
            pre_manifest.get("dspark_serving_profile", {}),
            expected_runtime_pin=runtime_pin,
        )
        runtime_evidence = validate_runtime_evidence(
            pre_manifest.get("dspark_runtime_evidence", {}),
            expected_runtime_pin=runtime_pin,
            verify_files=True,
        )
        post_profile = validate_serving_profile_receipt(
            post_manifest.get("dspark_serving_profile", {}),
            expected_runtime_pin=runtime_pin,
        )
        post_runtime = validate_runtime_evidence(
            post_manifest.get("dspark_runtime_evidence", {}),
            expected_runtime_pin=runtime_pin,
        )
        route_census = collect_route_census(serve_log_path)
        validate_route_census(route_census)
        validated_baseline = validate_baseline_comparison_evidence(
            baseline_comparison,
            verify_files=True,
            require_complete=False,
        )
    except DSparkServingProfileError as exc:
        raise DSparkValidationError(
            f"paired DSpark serving/profile evidence differs: {exc}"
        ) from exc
    if profile_receipt != post_profile or runtime_evidence != post_runtime:
        raise DSparkValidationError(
            "DSpark pre/post profile or runtime evidence changed"
        )
    if runtime_evidence.get("profile_receipt") != profile_receipt:
        raise DSparkValidationError(
            "DSpark profile and runtime evidence disagree"
        )
    if route_census.get("serve_log_sha256") != graph_capture.get(
        "serve_log_sha256"
    ):
        raise DSparkValidationError(
            "DSpark graph and route census do not derive from one serve log"
        )
    baseline_model = validated_baseline.get("local_model")
    if (
        not isinstance(baseline_model, Mapping)
        or baseline_model.get("target_model_sha256") != target_sha
        or baseline_model.get("draft_model_sha256") != draft_sha
    ):
        raise DSparkValidationError(
            "Entrpi-comparison rows do not name the exact target/draft pair"
        )

    matched_workload = _matched_performance_workload()
    no_mtp_options = _expected_no_mtp_launch_options(served_model)
    try:
        no_mtp_evidence = validate_arm_report(
            no_mtp_performance_report,
            report_path=no_mtp_performance_report_path,
            arm=NO_MTP_ARM,
            expected_target_sha=target_sha,
            expected_draft_sha=draft_sha,
            expected_workload=matched_workload,
            expected_acceptance=validated_acceptance,
            expected_routes=route_census,
            expected_speculative_config=None,
            expected_launch_options=no_mtp_options,
            expected_launch_switches=expected_switches,
            requires_moe_marlin=True,
            expected_no_mtp_graph_capture=no_mtp_graph_capture,
        )
        mtp_evidence = validate_arm_report(
            mtp_performance_report,
            report_path=mtp_performance_report_path,
            arm=MTP_ARM,
            expected_target_sha=target_sha,
            expected_draft_sha=draft_sha,
            expected_workload=matched_workload,
            expected_acceptance=validated_acceptance,
            expected_routes=route_census,
            expected_speculative_config=DSPARK_SPECULATIVE_CONFIG,
            expected_launch_options=expected_options,
            expected_launch_switches=expected_switches,
            requires_moe_marlin=True,
            expected_mtp_pre_manifest=pre_manifest,
            expected_mtp_post_manifest=post_manifest,
            expected_mtp_graph_capture=graph_capture,
        )
        matched_performance = build_matched_result(
            no_mtp_evidence=no_mtp_evidence,
            mtp_evidence=mtp_evidence,
            target_model_sha=target_sha,
            draft_model_sha=draft_sha,
            runtime_pin=runtime_pin,
            expected_workload=matched_workload,
            expected_acceptance=validated_acceptance,
            expected_routes=route_census,
            expected_no_mtp_launch_options=no_mtp_options,
            expected_mtp_launch_options=expected_options,
            expected_launch_switches=expected_switches,
            expected_mtp_speculative_config=DSPARK_SPECULATIVE_CONFIG,
        )
    except DSparkMatchedPerformanceError as exc:
        raise DSparkValidationError(
            f"matched no-MTP/MTP performance gate failed: {exc}"
        ) from exc

    target_binding = dict(pre_manifest["artifact_binding"])
    draft_binding = dict(pre_manifest["draft_artifact_binding"])
    result: dict[str, Any] = {
        "schema": DSPARK_PAIR_RESULT_SCHEMA,
        "target_model_sha": target_sha,
        "draft_model_sha": draft_sha,
        "source_model_identity": dict(target_source),
        "target_decode_contract_sha256": _canonical_sha256(target_decode),
        "draft_decode_contract_sha256": _canonical_sha256(draft_decode),
        "draft_render_attestation": draft_render,
        "target_artifact_binding": target_binding,
        "draft_artifact_binding": draft_binding,
        "runtime_pin": runtime_pin,
        "dspark_serving_profile": profile_receipt,
        "dspark_runtime_evidence": runtime_evidence,
        "serving_stack": {
            "image": DSV4_SPARK_VLLM_IMAGE,
            "vllm_version": DSV4_SPARK_VLLM_VERSION,
            "speculative_config": dict(DSPARK_SPECULATIVE_CONFIG),
            "launch_options": expected_options,
            "launch_switches": sorted(expected_switches),
        },
        "serve_session_id": str(pre_manifest["serve_session_id"]),
        "pre_serve_fingerprint": manifest_fingerprints[0],
        "post_serve_fingerprint": manifest_fingerprints[1],
        "pre_manifest_sha256": _file_sha256(pre_manifest_path),
        "post_manifest_sha256": _file_sha256(post_manifest_path),
        "graph_capture": graph_capture,
        "route_census": route_census,
        "acceptance_suite": validated_acceptance,
        "acceptance_suite_sha256": _file_sha256(acceptance_path),
        "baseline_comparison": validated_baseline,
        "baseline_comparison_file_sha256": _file_sha256(
            baseline_comparison_path
        ),
        "matched_performance": matched_performance,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    validate_pair_result(result)
    return result


def validate_pair_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the compact pair receipt persisted in each shipcard."""
    expected_keys = {
        "schema", "target_model_sha", "draft_model_sha",
        "source_model_identity", "target_decode_contract_sha256",
        "draft_decode_contract_sha256", "draft_render_attestation",
        "target_artifact_binding", "draft_artifact_binding", "runtime_pin",
        "dspark_serving_profile", "dspark_runtime_evidence",
        "serving_stack", "serve_session_id", "pre_serve_fingerprint",
        "post_serve_fingerprint", "pre_manifest_sha256",
        "post_manifest_sha256", "graph_capture", "route_census",
        "acceptance_suite", "acceptance_suite_sha256",
        "baseline_comparison", "baseline_comparison_file_sha256",
        "matched_performance", "receipt_sha256",
    }
    _require_exact_keys(payload, expected_keys, where="DSpark pair receipt")
    if payload.get("schema") != DSPARK_PAIR_RESULT_SCHEMA:
        raise DSparkValidationError("unsupported DSpark pair receipt schema")
    for name in (
        "target_model_sha", "draft_model_sha", "target_decode_contract_sha256",
        "draft_decode_contract_sha256", "serve_session_id",
        "pre_serve_fingerprint", "post_serve_fingerprint",
        "pre_manifest_sha256", "post_manifest_sha256",
        "acceptance_suite_sha256", "baseline_comparison_file_sha256",
        "receipt_sha256",
    ):
        if _SHA256_RE.fullmatch(str(payload.get(name, ""))) is None:
            raise DSparkValidationError(f"DSpark pair {name} is not SHA-256")
    if payload.get("pre_serve_fingerprint") != payload.get(
        "post_serve_fingerprint"
    ):
        raise DSparkValidationError(
            "DSpark pair pre/post stable serve fingerprints differ"
        )
    unstamped = dict(payload)
    recorded_receipt = str(unstamped.pop("receipt_sha256"))
    if recorded_receipt != _canonical_sha256(unstamped):
        raise DSparkValidationError("DSpark pair receipt digest is stale")
    target_sha = str(payload["target_model_sha"])
    draft_sha = str(payload["draft_model_sha"])
    _validate_artifact_binding(
        payload.get("target_artifact_binding"),
        expected_model_sha=target_sha,
        expected_launch_model="/model",
        where="pair target artifact binding",
    )
    _validate_artifact_binding(
        payload.get("draft_artifact_binding"),
        expected_model_sha=draft_sha,
        expected_launch_model="/draft",
        where="pair draft artifact binding",
    )
    stack = payload.get("serving_stack")
    if not isinstance(stack, Mapping) or stack != {
        "image": DSV4_SPARK_VLLM_IMAGE,
        "vllm_version": DSV4_SPARK_VLLM_VERSION,
        "speculative_config": dict(DSPARK_SPECULATIVE_CONFIG),
        "launch_options": _expected_launch_options(
            str((payload.get("acceptance_suite") or {}).get("served_model", ""))
        ),
        "launch_switches": sorted(DSPARK_LAUNCH_SWITCHES),
    }:
        raise DSparkValidationError("DSpark pair serving stack differs")
    try:
        runtime_pin = _gridbook_runtime_pin()
    except Exception as exc:
        raise DSparkValidationError(f"current Gridbook pin unavailable: {exc}") from exc
    if payload.get("runtime_pin") != runtime_pin:
        raise DSparkValidationError("DSpark pair runtime pin differs")
    try:
        profile_receipt = validate_serving_profile_receipt(
            payload.get("dspark_serving_profile", {}),
            expected_runtime_pin=runtime_pin,
        )
        runtime_evidence = validate_runtime_evidence(
            payload.get("dspark_runtime_evidence", {}),
            expected_runtime_pin=runtime_pin,
        )
    except DSparkServingProfileError as exc:
        raise DSparkValidationError(
            f"DSpark pair profile/runtime evidence differs: {exc}"
        ) from exc
    if runtime_evidence.get("profile_receipt") != profile_receipt:
        raise DSparkValidationError(
            "DSpark pair profile and runtime evidence disagree"
        )
    source = payload.get("source_model_identity")
    render = payload.get("draft_render_attestation")
    if (
        not isinstance(source, Mapping)
        or not isinstance(render, Mapping)
        or render.get("schema") != "prismaquant.dspark_cb_render_recipe.v1"
        or render.get("source_model_identity") != source
        or _SHA256_RE.fullmatch(str(render.get("recipe_sha256", ""))) is None
        or _SHA256_RE.fullmatch(
            str(render.get("source_weights_sha256", ""))
        ) is None
        or isinstance(render.get("source_weights_entries"), bool)
        or not isinstance(render.get("source_weights_entries"), int)
        or int(render.get("source_weights_entries", 0)) <= 0
        or _SHA256_RE.fullmatch(
            str(render.get("attestation_sha256", ""))
        ) is None
    ):
        raise DSparkValidationError(
            "DSpark pair has no source-complete draft render attestation"
        )
    graph = payload.get("graph_capture")
    if (
        not isinstance(graph, Mapping)
        or set(graph) != {
            "serve_log_sha256", "capture_marker", "capture_sizes"
        }
        or _SHA256_RE.fullmatch(
            str(graph.get("serve_log_sha256", ""))
        ) is None
        or _GRAPH_CAPTURE_RE.fullmatch(
            str(graph.get("capture_marker", ""))
        ) is None
        or graph.get("capture_sizes") != [5, 6]
    ):
        raise DSparkValidationError("DSpark graph-capture receipt is malformed")
    acceptance = payload.get("acceptance_suite")
    if not isinstance(acceptance, Mapping):
        raise DSparkValidationError("DSpark pair has no acceptance suite")
    validated_acceptance = validate_acceptance_suite(
        acceptance, expected_served_model=str(acceptance.get("served_model", ""))
    )
    try:
        route_census = validate_route_census(payload.get("route_census", {}))
        baseline = validate_baseline_comparison_evidence(
            payload.get("baseline_comparison", {}),
            require_complete=False,
        )
    except DSparkServingProfileError as exc:
        raise DSparkValidationError(
            f"DSpark route/baseline evidence differs: {exc}"
        ) from exc
    if route_census.get("serve_log_sha256") != graph.get("serve_log_sha256"):
        raise DSparkValidationError(
            "DSpark route census and graph evidence name different serve logs"
        )
    baseline_model = baseline.get("local_model")
    if (
        not isinstance(baseline_model, Mapping)
        or baseline_model.get("target_model_sha256") != target_sha
        or baseline_model.get("draft_model_sha256") != draft_sha
    ):
        raise DSparkValidationError(
            "DSpark baseline comparison names a different target/draft pair"
        )
    expected_options = _expected_launch_options(
        str(validated_acceptance.get("served_model", ""))
    )
    try:
        matched = validate_matched_result(
            payload.get("matched_performance"),
            expected_target_sha=target_sha,
            expected_draft_sha=draft_sha,
            expected_runtime_pin=runtime_pin,
            expected_workload=_matched_performance_workload(),
            expected_acceptance=validated_acceptance,
            expected_routes=route_census,
            expected_no_mtp_launch_options=_expected_no_mtp_launch_options(
                str(validated_acceptance.get("served_model", ""))
            ),
            expected_mtp_launch_options=expected_options,
            expected_launch_switches=DSPARK_LAUNCH_SWITCHES,
            expected_mtp_speculative_config=DSPARK_SPECULATIVE_CONFIG,
        )
    except DSparkMatchedPerformanceError as exc:
        raise DSparkValidationError(
            f"DSpark matched-performance evidence differs: {exc}"
        ) from exc
    if (
        matched["mtp"]["stable_serve_fingerprint"]
        != payload.get("post_serve_fingerprint")
        or matched["mtp"]["target_artifact_binding"]
        != payload.get("target_artifact_binding")
        or matched["mtp"]["draft_artifact_binding"]
        != payload.get("draft_artifact_binding")
        or matched["common_manifest_identity"].get(
            "dspark_serving_profile_sha256"
        ) != profile_receipt.get("receipt_sha256")
        or matched["common_manifest_identity"].get(
            "dspark_runtime_evidence_sha256"
        ) != runtime_evidence.get("evidence_sha256")
    ):
        raise DSparkValidationError(
            "DSpark matched-performance arm differs from the paired serve/runtime"
        )
    return dict(payload)


def validate_dspark_shipcard_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | Path | None = None,
) -> list[str]:
    """Shipcard replay hook for either role of one paired DSpark claim."""
    problems: list[str] = []
    prefix = f"{slot}: "
    try:
        if slot != DSPARK_SHIPCARD_SLOT:
            raise DSparkValidationError("wrong DSpark shipcard slot")
        expected_record_keys = {
            "slot", "tool", "filled_at", "passed", "model_sha",
            "spec_decode_detected", "serve_fingerprint", "git_commit",
            "detail", "metrics", "artifact_role", "peer_model_sha",
        }
        _require_exact_keys(record, expected_record_keys, where="DSpark shipcard record")
        role = record.get("artifact_role")
        metrics = record.get("metrics")
        if role not in {"target", "draft"} or not isinstance(metrics, Mapping):
            raise DSparkValidationError("DSpark shipcard role/metrics are malformed")
        validated = validate_pair_result(metrics)
        local_key = f"{role}_model_sha"
        peer_key = "draft_model_sha" if role == "target" else "target_model_sha"
        if (
            record.get("slot") != slot
            or record.get("tool") != DSPARK_SHIPCARD_TOOL
            or record.get("passed") is not True
            or record.get("spec_decode_detected") is not True
            or record.get("model_sha") != validated.get(local_key)
            or record.get("peer_model_sha") != validated.get(peer_key)
            or record.get("serve_fingerprint")
            != validated.get("post_serve_fingerprint")
            or record.get("git_commit")
            != (validated.get("matched_performance") or {}).get("tool", {}).get(
                "git_commit"
            )
            or (
                _SHA256_RE.fullmatch(
                    str(record.get("git_commit", ""))
                ) is None
                and re.fullmatch(
                    r"[0-9a-f]{40}", str(record.get("git_commit", ""))
                ) is None
            )
        ):
            raise DSparkValidationError("DSpark shipcard record fields differ")
        _parse_time(record.get("filled_at"), where="DSpark filled_at")
        if model_dir is not None:
            observed = compute_model_sha(model_dir)
            if observed != record.get("model_sha"):
                raise DSparkValidationError("local DSpark artifact SHA differs")
            if role == "draft":
                quant = validate_cb_artifact(model_dir)
                local_attestation = validate_dspark_production_render_attestation(
                    quant
                )
                if local_attestation != validated.get("draft_render_attestation"):
                    raise DSparkValidationError(
                        "local draft render attestation differs from pair receipt"
                    )
    except Exception as exc:
        problems.append(prefix + str(exc))
    return problems


def attach_pair_records(
    result: Mapping[str, Any],
    *,
    target_shipcard: str | Path,
    draft_shipcard: str | Path,
) -> None:
    validated = validate_pair_result(result)
    git = git_provenance()
    commit = git.get("commit")
    if git.get("dirty") is not False or not isinstance(commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", commit
    ) is None:
        raise DSparkValidationError(
            "DSpark shipcard attestation requires one clean exact PrismaQuant commit"
        )
    report_commit = (validated.get("matched_performance") or {}).get(
        "tool", {}
    ).get("git_commit")
    if report_commit != commit:
        raise DSparkValidationError(
            "DSpark performance reports were not collected by the clean "
            "PrismaQuant commit filling the reciprocal shipcards"
        )
    cards = {
        "target": (Path(target_shipcard), str(validated["target_model_sha"])),
        "draft": (Path(draft_shipcard), str(validated["draft_model_sha"])),
    }
    records: dict[str, dict[str, Any]] = {}
    for role, (_path, model_sha) in cards.items():
        peer_role = "draft" if role == "target" else "target"
        records[role] = make_record(
            slot=DSPARK_SHIPCARD_SLOT,
            tool=DSPARK_SHIPCARD_TOOL,
            passed=True,
            model_sha=model_sha,
            metrics=validated,
            detail=(
                "exact target+production-K12-draft DSpark pair passed the "
                "bracketed 8x128 acceptance, [5,6] graph, 256K/headroom, "
                "and matched no-MTP throughput non-regression gates"
            ),
            spec_decode_detected=True,
            serve_fingerprint=str(validated["post_serve_fingerprint"]),
            git_commit=commit,
            extra={
                "artifact_role": role,
                "peer_model_sha": str(validated[f"{peer_role}_model_sha"]),
            },
        )
        problems = validate_dspark_shipcard_record(
            DSPARK_SHIPCARD_SLOT, records[role]
        )
        if problems:
            raise DSparkValidationError(problems[0])
    # Both records have been fully validated before either card is mutated.
    # The slot is optional until a release explicitly claims MTP.
    for role in ("draft", "target"):
        path, _model_sha = cards[role]
        ensure_optional_slot(path, DSPARK_SHIPCARD_SLOT)
        fill_slot(path, DSPARK_SHIPCARD_SLOT, records[role])


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DSparkValidationError(f"{path}: expected JSON object")
    return payload


def _cmd_run_suite(args: argparse.Namespace) -> int:
    result = run_acceptance_suite(args.base_url)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[dspark] acceptance passed position0="
        f"{result['acceptance']['per_position_rates'][0]:.4f} "
        f"output_tps={result['served_output_tokens_per_second']:.3f} out={out}"
    )
    return 0


def _cmd_attest(args: argparse.Namespace) -> int:
    pre = _load_json(args.pre_manifest)
    post = _load_json(args.post_manifest)
    acceptance = _load_json(args.acceptance_json)
    no_mtp_performance = load_performance_report(args.no_mtp_performance_report)
    mtp_performance = load_performance_report(args.mtp_performance_report)
    baseline_comparison = _load_json(args.baseline_comparison)
    result = build_pair_result(
        target_model_dir=args.target_model_dir,
        draft_model_dir=args.draft_model_dir,
        target_shipcard=args.target_shipcard,
        draft_shipcard=args.draft_shipcard,
        pre_manifest=pre,
        post_manifest=post,
        pre_manifest_path=args.pre_manifest,
        post_manifest_path=args.post_manifest,
        acceptance=acceptance,
        acceptance_path=args.acceptance_json,
        serve_log_path=args.serve_log,
        no_mtp_serve_log_path=args.no_mtp_serve_log,
        no_mtp_performance_report=no_mtp_performance,
        no_mtp_performance_report_path=args.no_mtp_performance_report,
        mtp_performance_report=mtp_performance,
        mtp_performance_report_path=args.mtp_performance_report,
        baseline_comparison=baseline_comparison,
        baseline_comparison_path=args.baseline_comparison,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_fill_shipcards:
        attach_pair_records(
            result,
            target_shipcard=args.target_shipcard,
            draft_shipcard=args.draft_shipcard,
        )
    print(
        f"[dspark] target+draft production claim passed receipt="
        f"{result['receipt_sha256']} out={out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-suite", help="run fixed DSpark acceptance suite")
    run.add_argument("--base-url", required=True)
    run.add_argument("--output-json", required=True)
    run.set_defaults(func=_cmd_run_suite)

    attest = sub.add_parser(
        "attest", help="validate the bracketed pair and fill both shipcards"
    )
    attest.add_argument("--target-model-dir", required=True)
    attest.add_argument("--draft-model-dir", required=True)
    attest.add_argument("--target-shipcard", required=True)
    attest.add_argument("--draft-shipcard", required=True)
    attest.add_argument("--pre-manifest", required=True)
    attest.add_argument("--post-manifest", required=True)
    attest.add_argument("--acceptance-json", required=True)
    attest.add_argument("--serve-log", required=True)
    attest.add_argument("--no-mtp-serve-log", required=True)
    attest.add_argument("--no-mtp-performance-report", required=True)
    attest.add_argument("--mtp-performance-report", required=True)
    attest.add_argument("--baseline-comparison", required=True)
    attest.add_argument("--output-json", required=True)
    attest.add_argument(
        "--no-fill-shipcards",
        action="store_true",
        help="validate/write the receipt without mutating either shipcard",
    )
    attest.set_defaults(func=_cmd_attest)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
