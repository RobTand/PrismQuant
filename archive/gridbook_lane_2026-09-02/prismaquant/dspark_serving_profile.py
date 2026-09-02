"""Closed Gridbook-0.8.7 serving profile for the paired DSpark release gate.

This is deliberately separate from :mod:`prismaquant.gridbook_environment`.
That module is the producer/gold authority and tracks the producer pin (0.8.5
when this profile was written, 0.8.11 since 2026-08-21); changing it to
describe a later serving experiment would retroactively change the meaning of
historical evidence.  The profile here is a consumer-side contract
used only when the released target and the separately rendered DSpark draft
are served together.

The module is torch-free at import time.  ``collect_runtime_evidence`` imports
the already-pinned runtime only when an operator explicitly asks for a live
preflight receipt.  All replay and source/cache inspection helpers are CPU
safe.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES,
    GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA,
    GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA,
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
    GRIDBOOK_SERVING_RUNTIME_REPOSITORY,
    GridbookServingRuntimePin,
    load_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)


DSPARK_SERVING_PROFILE_SCHEMA = "prismaquant.dspark_serving_profile.v1"
DSPARK_SERVING_PROFILE_RECEIPT_SCHEMA = (
    "prismaquant.dspark_serving_profile_receipt.v1"
)
DSPARK_RUNTIME_EVIDENCE_SCHEMA = (
    "prismaquant.dspark_serving_runtime_evidence.v1"
)
DSPARK_ROUTE_CENSUS_SCHEMA = "prismaquant.dspark_route_census.v1"
DSPARK_CACHE_LOG_EVIDENCE_SCHEMA = (
    "prismaquant.dspark_flashinfer_cache_log.v1"
)
DSPARK_BASELINE_COMPARISON_SCHEMA = (
    "prismaquant.dspark_baseline_comparison.v1"
)
DSPARK_BASELINE_LOCAL_MODEL_SCHEMA = (
    "prismaquant.dspark_baseline_local_model.v1"
)
DSPARK_BASELINE_PREFILL_ROW_SCHEMA = (
    "prismaquant.dspark_baseline_prefill_row.v1"
)
DSPARK_BASELINE_DECODE_ROW_SCHEMA = (
    "prismaquant.dspark_baseline_decode_row.v1"
)
DSPARK_PROFILE_ID = "dsv4-dspark-gridbook-0.8.7-v2-k5-256k"

DSPARK_IMAGE = (
    "eugr/spark-vllm@sha256:"
    "58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
DSPARK_VLLM_VERSION = "0.26.1rc1.dev693+g7f7a32cfe.d20260812"
DSPARK_VLLM_COMMIT = "7f7a32cfec0f1bc5b73c37200b86631523a1ea8f"
DSPARK_TORCH_VERSION = "2.13.0+cu130"
DSPARK_FLASHINFER_DISTRIBUTION = "flashinfer-python"
DSPARK_FLASHINFER_VERSION = "0.6.18"
DSPARK_FLASHINFER_COMMIT = "9ffd99510d92b883f154fc9f2e3d5aac93e231ca"
DSPARK_FLASHINFER_DSV4_DISPATCH = (64, 256)

DSPARK_NUM_SPECULATIVE_TOKENS = 5
DSPARK_MODEL_LEN = 262144
DSPARK_KV_CACHE_DTYPE = "fp8"
DSPARK_KV_CACHE_BYTES = 1717986918
DSPARK_MAX_NUM_SEQS = 1
DSPARK_MAX_NUM_BATCHED_TOKENS = 512
DSPARK_GRAPH_CAPTURE_SIZES = (5, 6)
DSPARK_MAX_GRAPH_CAPTURE_SIZE = 6
DSPARK_MOE_BACKEND = "marlin"
DSPARK_TOOL_CALL_PARSER = "deepseek_v4"
DSPARK_DECODE_CONTRACT = "v1"

DSPARK_FLASHINFER_CUSTOM_OP = "sparse_mla_sm120_decode_dsv4"
DSPARK_FLASHINFER_RUNNER = "SparseMlaDecodeV3Runner"
DSPARK_FLASHINFER_TOKEN_BUCKETS = (1, 4, 8, 16, 32, 64)
DSPARK_FLASHINFER_NUM_HEADS = 64
DSPARK_FLASHINFER_HEAD_DIM = 512
DSPARK_FLASHINFER_MAIN_TOPK = 128
DSPARK_FLASHINFER_EXTRA_TOPK = 2048
DSPARK_FLASHINFER_SPLIT_COUNT = 34

DSPARK_TARGET_LAYER_COUNT = 43
DSPARK_DRAFT_LAYER_COUNT = 3
DSPARK_TARGET_FP4_V2_ROUTES = 70
DSPARK_DRAFT_FP4_V2_ROUTES = 6
DSPARK_FP8_INHERITED_ROUTES = 16
DSPARK_TOTAL_ROUTES = 92

ENTRPI_DSV4_REFERENCE = MappingProxyType({
    "schema": "prismaquant.external_baseline_reference.v1",
    "reference_id": "entrpi-ds4-on-spark-v0.5.0-gb10-dsv4-0731",
    "reference_only": True,
    "derives_release_pass": False,
    "benchmark_stamp": "2026-08-01",
    "wrapper_source": {
        "repository": "https://github.com/Entrpi/ds4-on-spark.git",
        "revision_kind": "main_snapshot",
        "commit": "78a9b2862695775e130b6e9d51ebde37e81a7f26",
        "files": {
            "README.md": (
                "bfdcdf7dc91bf8f4e5df962c1a85bc4d8ace1ac6a216b2d55fb5aecde5963d3a"
            ),
            "docs/v050_upstream_overlay.svg": (
                "95dd82bc7c70cea0c0268be21de531da22c94a6bded01c428bf6963642f8b242"
            ),
            "docs/v050_conc_throughput.svg": (
                "293668b692ae8392a97406e1d3c1e3e904adf5bd2b08c35a64bf79efacdb662f"
            ),
        },
    },
    "engine_source": {
        "repository": "https://github.com/Entrpi/ds4.git",
        "tag": "v0.5.0",
        "annotated_tag_object": "0c0febf18ba6ddc86843fa9b47acce01fd74a780",
        "commit": "d9c8587f3e080ce40fd961a9dd09c66c294a6b10",
        "changelog_sha256": (
            "df381f46542b29bcdb6c538dc898faa0155c1935d618f12fa5d3ea9525306c45"
        ),
    },
    "hardware": {
        "gpu": "NVIDIA GB10",
        "compute_capability": "sm_121",
        "gpu_count": 1,
        "unified_memory_gib": 128,
        "usable_memory_gib_approx": 119,
    },
    "model": {
        "identity": "DeepSeek-V4-Flash-0731",
        "container": "GGUF",
        "model_gib_approx": 81,
        "routed_gate_up": "IQ2_XXS",
        "routed_down": "Q2_K",
        "dense": "Q8_0",
        "lora_and_compressor": "F16",
        "norms": "F32",
        "draft_gib_approx": 7,
    },
    "claims": {
        "prefill": [
            {
                "context_tokens": 2048,
                "tokens_per_second": 960.0,
                "precision": "approximate",
            },
            {
                "context_tokens": 12288,
                "tokens_per_second": 1010.0,
                "precision": "approximate",
            },
            {
                "context_tokens": 65536,
                "tokens_per_second": 933.0,
                "precision": "approximate",
            },
            {
                "context_tokens": 517963,
                "wall_seconds": 667.1,
                "tokens_per_second": 776.4,
                "precision": "reported_exact",
            },
        ],
        "decode": [
            {
                "context_tokens": 12288,
                "concurrency": 1,
                "tokens_per_second": 28.0,
                "precision": "external_summary_approximate",
            },
            {
                "context_tokens": 240000,
                "concurrency": 1,
                "milliseconds_per_token": 45.7,
                "tokens_per_verify_step": 2.9,
                "precision": "reported_engine_side",
            },
            {
                "context_tokens": 12288,
                "concurrency": 12,
                "aggregate_tokens_per_second": 59.0,
                "output_tokens_per_request": 192,
                "precision": "reported_served_wall",
            },
        ],
    },
})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SOURCE_SUFFIXES = frozenset({".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"})
_PREFIXED_IDENTIFIER_RE = re.compile(
    r"(?<![A-Z0-9_])((?:PRISMAQUANT|GRIDBOOK|VLLM)_[A-Z0-9_]+)"
    r"(?![A-Z0-9_])"
)
_DIRECT_ENVIRONMENT_RES = (
    re.compile(r"os[.]environ[.]get[(]\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    re.compile(r"os[.]environ\[\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    re.compile(r"(?:std::)?getenv[(]\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
)


class DSparkServingProfileError(ValueError):
    """The current paired DSpark serving contract is not exact."""


# A full literal registry is intentional.  Importing the producer/gold
# registry here as authority would make that producer contract control a later
# consumer lane and would prevent either side from detecting namespace drift.
# (Concretely: that registry has since advanced with its pin to 0.8.11, while
# this literal set stays the 0.8.7 audit it was written from.)
DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST = tuple(sorted((
    "CUDACXX",
    "CXX",
    "GRIDBOOK_MXFP8_DENSE",
    "PRISMAQUANT_CB_BF16_SM120",
    "PRISMAQUANT_CB_DECODE",
    "PRISMAQUANT_CB_DECODE_CONTRACT",
    "PRISMAQUANT_CB_EXPAND",
    "PRISMAQUANT_CB_EXT_DIR",
    "PRISMAQUANT_CB_FP4V2_SCHED",
    "PRISMAQUANT_CB_FP4_FUSED_MIDM",
    "PRISMAQUANT_CB_FP8_GEMV_V2",
    "PRISMAQUANT_CB_FP8_SCHED",
    "PRISMAQUANT_CB_FUSED_FP4",
    "PRISMAQUANT_CB_FUSED_FP4_MOE",
    "PRISMAQUANT_CB_FUSED_MIDM",
    "PRISMAQUANT_CB_GEMV",
    "PRISMAQUANT_CB_GROUPED_TRIM",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R",
    "PRISMAQUANT_CB_PREFILL",
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES",
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
    "PRISMAQUANT_CB_W2_ROWS",
    "PRISMAQUANT_CB_W2_SCHED",
    "PRISMAQUANT_CB_W2_WARPS",
    "PRISMAQUANT_CUTLASS_INCLUDE",
    "PRISMAQUANT_DEBUG_PREFIXES",
    "PRISMAQUANT_PRELOAD_FUSED",
    "PRISMAQUANT_SKIP_CB_CAST_CHECK",
    "VLLM_USE_DEEP_GEMM",
)))

_profile_environment = {
    "CUDACXX": None,
    "CXX": None,
    "GRIDBOOK_MXFP8_DENSE": None,
    "PRISMAQUANT_CB_BF16_SM120": "0",
    "PRISMAQUANT_CB_DECODE": None,
    "PRISMAQUANT_CB_DECODE_CONTRACT": DSPARK_DECODE_CONTRACT,
    "PRISMAQUANT_CB_EXPAND": None,
    "PRISMAQUANT_CB_EXT_DIR": "/opt/gridbook/ext-cache",
    "PRISMAQUANT_CB_FP4V2_SCHED": None,
    "PRISMAQUANT_CB_FP4_FUSED_MIDM": "0",
    # v0.8.7 lanes, pinned OFF: they did not exist in the runtime this
    # profile's accepted prefill/decode numbers were taken on, so
    # enabling either is a measurement decision, not bookkeeping.
    "PRISMAQUANT_CB_FP8_GEMV_V2": "0",
    "PRISMAQUANT_CB_FP8_SCHED": None,
    "PRISMAQUANT_CB_FUSED_FP4": None,
    "PRISMAQUANT_CB_FUSED_FP4_MOE": None,
    "PRISMAQUANT_CB_FUSED_MIDM": "1",
    "PRISMAQUANT_CB_GEMV": "v2",
    "PRISMAQUANT_CB_GROUPED_TRIM": "1",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B": "0",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG": "0",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R": "0",
    "PRISMAQUANT_CB_PREFILL": None,
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES": "1073741824",
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK": None,
    "PRISMAQUANT_CB_W2_ROWS": None,
    "PRISMAQUANT_CB_W2_SCHED": None,
    "PRISMAQUANT_CB_W2_WARPS": None,
    "PRISMAQUANT_CUTLASS_INCLUDE": None,
    "PRISMAQUANT_DEBUG_PREFIXES": None,
    "PRISMAQUANT_PRELOAD_FUSED": "1",
    "PRISMAQUANT_SKIP_CB_CAST_CHECK": "0",
    "VLLM_USE_DEEP_GEMM": "0",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}
if set(_profile_environment) != set(DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST) | {
    "PYTORCH_ALLOC_CONF"
}:  # pragma: no cover - import-time authoring invariant
    raise RuntimeError("DSpark serving environment registry is incomplete")
DSPARK_PROFILE_ENVIRONMENT = MappingProxyType(_profile_environment)
DSPARK_PROFILE_SET_ENVIRONMENT = MappingProxyType({
    name: value for name, value in _profile_environment.items()
    if value is not None
})
DSPARK_PROFILE_CLEARED_ENVIRONMENT = tuple(sorted(
    name for name, value in _profile_environment.items() if value is None
))

# The process snapshot gets its own allowlist rather than widening the old
# endpoint/gold allowlist.  Thus old manifests and fingerprints retain their
# exact original meaning while paired DSpark manifests also bind allocator
# policy.
DSPARK_SERVER_ENV_ALLOWLIST = tuple(sorted({
    "PQ_GRIDBOOK_RUNTIME_COMMIT",
    "PQ_GRIDBOOK_RUNTIME_VERSION",
    "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    *DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST,
    "PYTORCH_ALLOC_CONF",
}))


@dataclass(frozen=True)
class DSparkServingProfile:
    """The static, pin-independent portion of the release recipe."""

    profile_id: str = DSPARK_PROFILE_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DSPARK_SERVING_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "gridbook_version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
            "environment": dict(DSPARK_PROFILE_ENVIRONMENT),
            "runtime": {
                "image": DSPARK_IMAGE,
                "vllm": {
                    "version": DSPARK_VLLM_VERSION,
                    "commit": DSPARK_VLLM_COMMIT,
                },
                "torch_version": DSPARK_TORCH_VERSION,
                "flashinfer": {
                    "distribution": DSPARK_FLASHINFER_DISTRIBUTION,
                    "version": DSPARK_FLASHINFER_VERSION,
                    "commit": DSPARK_FLASHINFER_COMMIT,
                    "native_dsv4_dispatch": list(
                        DSPARK_FLASHINFER_DSV4_DISPATCH
                    ),
                },
                "vllm_moe_skip_padding": True,
                "max_context_tuned_cache": {
                    "custom_op": DSPARK_FLASHINFER_CUSTOM_OP,
                    "runner": DSPARK_FLASHINFER_RUNNER,
                    "token_buckets": list(DSPARK_FLASHINFER_TOKEN_BUCKETS),
                    "num_heads": DSPARK_FLASHINFER_NUM_HEADS,
                    "head_dim": DSPARK_FLASHINFER_HEAD_DIM,
                    "main_topk": DSPARK_FLASHINFER_MAIN_TOPK,
                    "extra_topk": DSPARK_FLASHINFER_EXTRA_TOPK,
                    "split_count": DSPARK_FLASHINFER_SPLIT_COUNT,
                    "target_entry_count": len(
                        DSPARK_FLASHINFER_TOKEN_BUCKETS
                    ),
                },
            },
            "launch": {
                "num_speculative_tokens": DSPARK_NUM_SPECULATIVE_TOKENS,
                "max_model_len": DSPARK_MODEL_LEN,
                "kv_cache_dtype": DSPARK_KV_CACHE_DTYPE,
                "kv_cache_memory_bytes": DSPARK_KV_CACHE_BYTES,
                "max_num_seqs": DSPARK_MAX_NUM_SEQS,
                "max_num_batched_tokens": DSPARK_MAX_NUM_BATCHED_TOKENS,
                "graph_capture_sizes": list(DSPARK_GRAPH_CAPTURE_SIZES),
                "max_graph_capture_size": DSPARK_MAX_GRAPH_CAPTURE_SIZE,
                "moe_backend": DSPARK_MOE_BACKEND,
                "tool_call_parser": DSPARK_TOOL_CALL_PARSER,
            },
            "routes": {
                "target_layers": DSPARK_TARGET_LAYER_COUNT,
                "draft_layers": DSPARK_DRAFT_LAYER_COUNT,
                "target_fp4_v2": DSPARK_TARGET_FP4_V2_ROUTES,
                "draft_fp4_v2": DSPARK_DRAFT_FP4_V2_ROUTES,
                "fp8_inherited": DSPARK_FP8_INHERITED_ROUTES,
                "fallback": 0,
                "unscoped": 0,
                "total": DSPARK_TOTAL_ROUTES,
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def _canonical_sha256(payload: object) -> str:
    try:
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DSparkServingProfileError(
            "DSpark evidence is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


DSPARK_SERVING_PROFILE = DSparkServingProfile()
DSPARK_SERVING_PROFILE_SHA256 = DSPARK_SERVING_PROFILE.sha256
ENTRPI_DSV4_REFERENCE_SHA256 = _canonical_sha256(dict(ENTRPI_DSV4_REFERENCE))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    if set(payload) != expected:
        raise DSparkServingProfileError(
            f"{where} keys differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _pin_payload(pin: GridbookServingRuntimePin | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(pin, GridbookServingRuntimePin):
        payload = {
            "schema": pin.schema,
            "repository": pin.repository,
            "commit": pin.commit,
            "version": pin.version,
            "version_is_release": pin.version_is_release,
            "wheel_sha256": pin.wheel_sha256,
            "runtime_contract_schema": pin.runtime_contract_schema,
            "required_abi_features": dict(pin.required_abi_features),
        }
    elif isinstance(pin, Mapping):
        payload = dict(pin)
    else:
        raise DSparkServingProfileError("Gridbook serving pin is not a mapping")
    _require_exact_keys(
        payload,
        {
            "schema", "repository", "commit", "version",
            "version_is_release", "wheel_sha256", "runtime_contract_schema",
            "required_abi_features",
        },
        where="Gridbook serving pin",
    )
    if (
        payload["schema"] != GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA
        or payload["repository"] != GRIDBOOK_SERVING_RUNTIME_REPOSITORY
        or payload["version"] != GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION
        or payload["version_is_release"] is not True
        or _COMMIT_RE.fullmatch(str(payload["commit"])) is None
        or _SHA256_RE.fullmatch(str(payload["wheel_sha256"])) is None
        or payload["runtime_contract_schema"]
        != GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA
        or payload["required_abi_features"]
        != GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES
    ):
        raise DSparkServingProfileError(
            "Gridbook serving pin is not one exact released "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION}/"
            f"{GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA} pin"
        )
    return payload


def current_gridbook_serving_pin_payload() -> dict[str, Any]:
    pin = load_gridbook_serving_runtime_pin()
    try:
        require_exact_gridbook_serving_runtime_release(pin)
    except Exception as exc:
        raise DSparkServingProfileError(
            f"current Gridbook serving pin is unavailable: {exc}"
        ) from exc
    return _pin_payload(pin)


def serving_profile_receipt(
    runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pin = (
        current_gridbook_serving_pin_payload()
        if runtime_pin is None else _pin_payload(runtime_pin)
    )
    receipt: dict[str, Any] = {
        "schema": DSPARK_SERVING_PROFILE_RECEIPT_SCHEMA,
        "profile": DSPARK_SERVING_PROFILE.as_dict(),
        "profile_sha256": DSPARK_SERVING_PROFILE_SHA256,
        "runtime_pin": pin,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_serving_profile_receipt(
    payload: Mapping[str, Any],
    *,
    expected_runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema", "profile", "profile_sha256", "runtime_pin",
            "receipt_sha256",
        },
        where="DSpark serving profile receipt",
    )
    expected_pin = (
        current_gridbook_serving_pin_payload()
        if expected_runtime_pin is None else _pin_payload(expected_runtime_pin)
    )
    expected = serving_profile_receipt(expected_pin)
    if dict(payload) != expected:
        raise DSparkServingProfileError(
            "DSpark serving profile receipt differs from the selected release"
        )
    return dict(payload)


def snapshot_profile_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    source = os.environ if environ is None else environ
    return {
        name: source.get(name)
        for name in sorted(DSPARK_PROFILE_ENVIRONMENT)
    }


def apply_profile_environment(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str | None]:
    target = os.environ if environ is None else environ
    for name in DSPARK_PROFILE_ENVIRONMENT:
        target.pop(name, None)
    target.update(DSPARK_PROFILE_SET_ENVIRONMENT)
    return snapshot_profile_environment(target)


def attest_profile_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    observed = snapshot_profile_environment(environ)
    expected = dict(DSPARK_PROFILE_ENVIRONMENT)
    if observed != expected:
        differences = [
            name for name in sorted(expected)
            if observed.get(name) != expected[name]
        ]
        raise DSparkServingProfileError(
            "DSpark serving environment differs for " + ", ".join(differences)
        )
    return {
        "schema": "prismaquant.dspark_serving_environment.v1",
        "profile_sha256": DSPARK_SERVING_PROFILE_SHA256,
        "environment": observed,
    }


def expected_server_environment(
    runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, str]:
    pin = (
        current_gridbook_serving_pin_payload()
        if runtime_pin is None else _pin_payload(runtime_pin)
    )
    return {
        **dict(DSPARK_PROFILE_SET_ENVIRONMENT),
        "PQ_GRIDBOOK_RUNTIME_COMMIT": str(pin["commit"]),
        "PQ_GRIDBOOK_RUNTIME_VERSION": str(pin["version"]),
        "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256": str(pin["wheel_sha256"]),
        "PYTHONSAFEPATH": "1",
    }


# These are the exact 0.8.7 source identifiers audited for this serving
# profile.  ``VLLM_MOE_SKIP_PADDING`` is a vLLM capability mentioned by
# Gridbook but resolved and attested through ``vllm.envs``; the W2 wildcard is
# prose explaining the three individually registered W2 overrides.
GRIDBOOK_087_SOURCE_NON_ENVIRONMENT_IDENTIFIERS = MappingProxyType({
    "PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA": "artifact schema constant",
    "PRISMAQUANT_CB_W2_": "documentation wildcard for registered W2 knobs",
    "PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE": "retired no-op in a comment",
    "VLLM_COMPILE": "external vLLM setting mentioned in documentation",
    "VLLM_CUTLASS": "vLLM backend enum member",
    "VLLM_MOE_SKIP_PADDING": "resolved vLLM capability, not a Gridbook env read",
    "VLLM_TEST_FORCE_FP8_MARLIN": "external vLLM test flag in prose",
})
GRIDBOOK_087_VISIBLE_REGISTERED_ENVIRONMENT = frozenset(
    set(DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST)
    - {"PRISMAQUANT_CB_DECODE", "PRISMAQUANT_CB_EXPAND"}
)
GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS = frozenset({
    *GRIDBOOK_087_VISIBLE_REGISTERED_ENVIRONMENT,
    *GRIDBOOK_087_SOURCE_NON_ENVIRONMENT_IDENTIFIERS,
})


@dataclass(frozen=True)
class Gridbook087SourceEnvironmentScan:
    source_root: str
    identifiers: tuple[str, ...]
    registered_environment: tuple[str, ...]
    classified_non_environment: tuple[str, ...]
    unknown_identifiers: tuple[str, ...]
    missing_expected_identifiers: tuple[str, ...]
    locations: tuple[tuple[str, tuple[str, ...]], ...]

    def receipt(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "prismaquant.gridbook_0_8_6_source_environment.v1",
            "identifiers": list(self.identifiers),
            "registered_environment": list(self.registered_environment),
            "classified_non_environment": list(
                self.classified_non_environment
            ),
            "unknown_identifiers": list(self.unknown_identifiers),
            "missing_expected_identifiers": list(
                self.missing_expected_identifiers
            ),
            "locations": [
                [name, list(paths)] for name, paths in self.locations
            ],
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


def _gridbook_package_root(
    source_root: str | os.PathLike[str],
) -> tuple[Path, Path]:
    root = Path(source_root).resolve()
    package = root / "gridbook"
    if package.is_dir():
        return root, package
    if root.name == "gridbook" and (root / "lane_select.py").is_file():
        return root.parent, root
    raise DSparkServingProfileError(
        f"{root}: expected a Gridbook root containing gridbook/"
    )


def scan_gridbook_087_source_environment(
    source_root: str | os.PathLike[str],
) -> Gridbook087SourceEnvironmentScan:
    repo_root, package_root = _gridbook_package_root(source_root)
    locations: dict[str, set[str]] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DSparkServingProfileError(
                f"cannot scan Gridbook source {path}: {exc}"
            ) from exc
        names = set(_PREFIXED_IDENTIFIER_RE.findall(text))
        for pattern in _DIRECT_ENVIRONMENT_RES:
            names.update(pattern.findall(text))
        relative = str(path.relative_to(repo_root))
        for name in names:
            locations.setdefault(name, set()).add(relative)
    identifiers = tuple(sorted(locations))
    registered = tuple(
        name for name in identifiers
        if name in DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST
    )
    classified = tuple(
        name for name in identifiers
        if name in GRIDBOOK_087_SOURCE_NON_ENVIRONMENT_IDENTIFIERS
    )
    known = set(registered) | set(classified)
    return Gridbook087SourceEnvironmentScan(
        source_root=str(repo_root),
        identifiers=identifiers,
        registered_environment=registered,
        classified_non_environment=classified,
        unknown_identifiers=tuple(
            name for name in identifiers if name not in known
        ),
        missing_expected_identifiers=tuple(sorted(
            GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS - set(identifiers)
        )),
        locations=tuple(
            (name, tuple(sorted(paths)))
            for name, paths in sorted(locations.items())
        ),
    )


def require_gridbook_087_source_compatible(
    source_root: str | os.PathLike[str],
) -> Gridbook087SourceEnvironmentScan:
    report = scan_gridbook_087_source_environment(source_root)
    if report.unknown_identifiers or report.missing_expected_identifiers:
        raise DSparkServingProfileError(
            "Gridbook 0.8.7 source environment drift: unknown="
            f"{list(report.unknown_identifiers)}, missing="
            f"{list(report.missing_expected_identifiers)}"
        )
    return report


def _validate_source_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema", "identifiers", "registered_environment",
            "classified_non_environment", "unknown_identifiers",
            "missing_expected_identifiers", "locations", "receipt_sha256",
        },
        where="Gridbook 0.8.7 source receipt",
    )
    unstamped = dict(payload)
    recorded = unstamped.pop("receipt_sha256")
    if (
        payload.get("schema")
        != "prismaquant.gridbook_0_8_6_source_environment.v1"
        or payload.get("identifiers")
        != sorted(GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS)
        or payload.get("registered_environment")
        != sorted(GRIDBOOK_087_VISIBLE_REGISTERED_ENVIRONMENT)
        or payload.get("classified_non_environment")
        != sorted(GRIDBOOK_087_SOURCE_NON_ENVIRONMENT_IDENTIFIERS)
        or payload.get("unknown_identifiers") != []
        or payload.get("missing_expected_identifiers") != []
        or not isinstance(payload.get("locations"), list)
        or len(payload["locations"]) != len(
            GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS
        )
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or row[0] not in GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS
            or not isinstance(row[1], list)
            or not row[1]
            or row[1] != sorted(set(row[1]))
            or any(not isinstance(path, str) or not path for path in row[1])
            for row in payload["locations"]
        )
        or [row[0] for row in payload["locations"]]
        != sorted(GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS)
        or recorded != _canonical_sha256(unstamped)
    ):
        raise DSparkServingProfileError(
            "Gridbook 0.8.7 source namespace receipt is not closed"
        )
    return dict(payload)


def _parsed_cache_key(key: str) -> tuple[Any, ...] | None:
    if key == "_metadata":
        return None
    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, tuple) else None


def _target_cache_bucket(key: str) -> int | None:
    parsed = _parsed_cache_key(key)
    if parsed is None or len(parsed) != 4:
        return None
    custom_op, runner, shapes, extras = parsed
    if (
        custom_op != DSPARK_FLASHINFER_CUSTOM_OP
        or runner != DSPARK_FLASHINFER_RUNNER
        or not isinstance(shapes, tuple)
        or len(shapes) != 10
    ):
        return None
    query_shape = shapes[0]
    if not isinstance(query_shape, tuple) or len(query_shape) != 3:
        return None
    bucket = query_shape[0]
    if bucket not in DSPARK_FLASHINFER_TOKEN_BUCKETS:
        return None
    expected_shapes = (
        (bucket, DSPARK_FLASHINFER_NUM_HEADS, DSPARK_FLASHINFER_HEAD_DIM),
        (bucket, DSPARK_FLASHINFER_MAIN_TOPK),
        (-1, DSPARK_FLASHINFER_NUM_HEADS, DSPARK_FLASHINFER_SPLIT_COUNT,
         DSPARK_FLASHINFER_HEAD_DIM),
        (-1, DSPARK_FLASHINFER_NUM_HEADS, DSPARK_FLASHINFER_SPLIT_COUNT),
        (-1, DSPARK_FLASHINFER_NUM_HEADS, DSPARK_FLASHINFER_HEAD_DIM),
        (-1, DSPARK_FLASHINFER_NUM_HEADS),
        (bucket,),
        (DSPARK_FLASHINFER_NUM_HEADS,),
        (bucket, DSPARK_FLASHINFER_EXTRA_TOPK),
        (bucket,),
    )
    if shapes != expected_shapes or extras != (
        True, True, DSPARK_FLASHINFER_EXTRA_TOPK, True
    ):
        return None
    return int(bucket)


def inspect_flashinfer_tuned_cache(path: str | os.PathLike[str]) -> dict[str, Any]:
    cache_path = Path(path).resolve()
    try:
        raw = cache_path.read_bytes()
        cache = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DSparkServingProfileError(
            f"FlashInfer tuned cache is unreadable: {exc}"
        ) from exc
    if not isinstance(cache, dict) or not isinstance(cache.get("_metadata"), dict):
        raise DSparkServingProfileError(
            "FlashInfer tuned cache must be an object with _metadata"
        )
    if cache["_metadata"].get("flashinfer_version") != DSPARK_FLASHINFER_VERSION:
        raise DSparkServingProfileError(
            "FlashInfer tuned cache version differs"
        )
    targets = {
        key: value for key, value in cache.items()
        if _target_cache_bucket(key) is not None
    }
    buckets = sorted(_target_cache_bucket(key) for key in targets)
    if buckets != list(DSPARK_FLASHINFER_TOKEN_BUCKETS):
        raise DSparkServingProfileError(
            "FlashInfer tuned cache lacks the exact extra_topk=2048 buckets"
        )
    tactics: dict[str, int] = {}
    for key, value in targets.items():
        bucket = _target_cache_bucket(key)
        if (
            bucket is None
            or not isinstance(value, list)
            or len(value) != 2
            or value[0] != DSPARK_FLASHINFER_RUNNER
            or isinstance(value[1], bool)
            or not isinstance(value[1], int)
            or value[1] <= 0
        ):
            raise DSparkServingProfileError(
                "FlashInfer tuned cache contains an untuned/fallback tactic"
            )
        tactics[str(bucket)] = value[1]
    return {
        "path": str(cache_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entry_count": len(cache) - 1,
        "target_entry_count": len(targets),
        "target_buckets": list(DSPARK_FLASHINFER_TOKEN_BUCKETS),
        "target_tactics": dict(sorted(tactics.items(), key=lambda row: int(row[0]))),
        "shape_contract": {
            "custom_op": DSPARK_FLASHINFER_CUSTOM_OP,
            "runner": DSPARK_FLASHINFER_RUNNER,
            "num_heads": DSPARK_FLASHINFER_NUM_HEADS,
            "head_dim": DSPARK_FLASHINFER_HEAD_DIM,
            "main_topk": DSPARK_FLASHINFER_MAIN_TOPK,
            "extra_topk": DSPARK_FLASHINFER_EXTRA_TOPK,
            "split_count": DSPARK_FLASHINFER_SPLIT_COUNT,
        },
    }


def _validate_cache_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "path", "sha256", "entry_count", "target_entry_count",
            "target_buckets", "target_tactics", "shape_contract",
        },
        where="FlashInfer tuned-cache receipt",
    )
    expected_shape = DSPARK_SERVING_PROFILE.as_dict()["runtime"][
        "max_context_tuned_cache"
    ].copy()
    expected_shape.pop("token_buckets")
    expected_shape.pop("target_entry_count")
    tactics = payload.get("target_tactics")
    if (
        not isinstance(payload.get("path"), str)
        or not Path(str(payload["path"])).is_absolute()
        or _SHA256_RE.fullmatch(str(payload.get("sha256", ""))) is None
        or isinstance(payload.get("entry_count"), bool)
        or not isinstance(payload.get("entry_count"), int)
        or payload["entry_count"] < len(DSPARK_FLASHINFER_TOKEN_BUCKETS)
        or payload.get("target_entry_count")
        != len(DSPARK_FLASHINFER_TOKEN_BUCKETS)
        or payload.get("target_buckets")
        != list(DSPARK_FLASHINFER_TOKEN_BUCKETS)
        or payload.get("shape_contract") != expected_shape
        or not isinstance(tactics, Mapping)
        or set(tactics) != {str(value) for value in DSPARK_FLASHINFER_TOKEN_BUCKETS}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in tactics.values()
        )
    ):
        raise DSparkServingProfileError(
            "FlashInfer tuned-cache receipt is not the exact max-context capability"
        )
    return dict(payload)


def _build_metadata_receipt(path: str | os.PathLike[str]) -> dict[str, Any]:
    metadata_path = Path(path).resolve()
    try:
        raw = metadata_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise DSparkServingProfileError(
            f"image build metadata is unreadable: {exc}"
        ) from exc
    commits: dict[str, str] = {}
    for name in ("vllm_commit", "flashinfer_commit"):
        matches = re.findall(
            rf"(?m)^{re.escape(name)}:\s*([0-9a-f]{{40}})\s*$", text
        )
        if len(matches) != 1:
            raise DSparkServingProfileError(
                f"image build metadata must declare exactly one {name}"
            )
        commits[name] = matches[0]
    if commits != {
        "vllm_commit": DSPARK_VLLM_COMMIT,
        "flashinfer_commit": DSPARK_FLASHINFER_COMMIT,
    }:
        raise DSparkServingProfileError(
            "image build metadata runtime commits differ"
        )
    return {
        "path": str(metadata_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **commits,
    }


def _installed_gridbook_source_root() -> Path:
    try:
        distribution = importlib_metadata.distribution("gridbook")
    except Exception as exc:
        raise DSparkServingProfileError(
            "installed Gridbook distribution is unavailable"
        ) from exc
    package = Path(distribution.locate_file("gridbook")).resolve()
    if not package.is_dir():
        raise DSparkServingProfileError(
            "installed Gridbook package source is unavailable"
        )
    return package.parent


def collect_runtime_evidence(
    *,
    build_metadata_path: str | os.PathLike[str],
    flashinfer_cache_path: str | os.PathLike[str],
    gridbook_source_root: str | os.PathLike[str] | None = None,
    runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect one live 0.8.7/vLLM/FlashInfer preflight receipt."""

    pin = (
        current_gridbook_serving_pin_payload()
        if runtime_pin is None else _pin_payload(runtime_pin)
    )
    attest_profile_environment()
    source = require_gridbook_087_source_compatible(
        gridbook_source_root or _installed_gridbook_source_root()
    ).receipt()
    cache = inspect_flashinfer_tuned_cache(flashinfer_cache_path)
    metadata = _build_metadata_receipt(build_metadata_path)
    packages = {
        "gridbook": importlib_metadata.version("gridbook"),
        "vllm": importlib_metadata.version("vllm"),
        "torch": importlib_metadata.version("torch"),
        DSPARK_FLASHINFER_DISTRIBUTION: importlib_metadata.version(
            DSPARK_FLASHINFER_DISTRIBUTION
        ),
    }
    try:
        import flashinfer
        import torch
        import vllm
        from flashinfer.mla import _sparse_mla_sm120
        from vllm import envs as vllm_envs

        # Runtime code is resolved only in this explicit live-preflight path.
        # Keep the producer module free of a static Gridbook dependency so its
        # ordinary import/export paths retain the external-runtime boundary.
        gridbook = importlib.import_module("gridbook")
    except Exception as exc:
        raise DSparkServingProfileError(
            f"DSpark runtime capability import failed: {type(exc).__name__}: {exc}"
        ) from exc
    module_versions = {
        "gridbook": getattr(gridbook, "__version__", None),
        "vllm": getattr(vllm, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "flashinfer": getattr(flashinfer, "__version__", None),
    }
    dispatch = getattr(_sparse_mla_sm120, "_DECODE_DSV4_DISPATCH", ())
    capabilities = {
        "flashinfer_native_dsv4_dispatch": {
            "head_dim": DSPARK_FLASHINFER_DSV4_DISPATCH[0],
            "topk": DSPARK_FLASHINFER_DSV4_DISPATCH[1],
            "present": DSPARK_FLASHINFER_DSV4_DISPATCH in dispatch,
        },
        "vllm_moe_skip_padding": getattr(
            vllm_envs, "VLLM_MOE_SKIP_PADDING", None
        ),
    }
    evidence: dict[str, Any] = {
        "schema": DSPARK_RUNTIME_EVIDENCE_SCHEMA,
        "profile_receipt": serving_profile_receipt(pin),
        "packages": packages,
        "module_versions": module_versions,
        "build_metadata": metadata,
        "capabilities": capabilities,
        "max_context_tuned_cache": cache,
        "gridbook_source_environment": source,
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return validate_runtime_evidence(evidence, expected_runtime_pin=pin)


def validate_runtime_evidence(
    payload: Mapping[str, Any],
    *,
    expected_runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema", "profile_receipt", "packages", "module_versions",
            "build_metadata", "capabilities", "max_context_tuned_cache",
            "gridbook_source_environment", "evidence_sha256",
        },
        where="DSpark runtime evidence",
    )
    pin = (
        current_gridbook_serving_pin_payload()
        if expected_runtime_pin is None else _pin_payload(expected_runtime_pin)
    )
    validate_serving_profile_receipt(
        payload.get("profile_receipt", {}), expected_runtime_pin=pin
    )
    expected_packages = {
        "gridbook": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
        "vllm": DSPARK_VLLM_VERSION,
        "torch": DSPARK_TORCH_VERSION,
        DSPARK_FLASHINFER_DISTRIBUTION: DSPARK_FLASHINFER_VERSION,
    }
    expected_modules = {
        "gridbook": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
        "vllm": DSPARK_VLLM_VERSION,
        "torch": DSPARK_TORCH_VERSION,
        "flashinfer": DSPARK_FLASHINFER_VERSION,
    }
    metadata = payload.get("build_metadata")
    capabilities = payload.get("capabilities")
    if (
        payload.get("schema") != DSPARK_RUNTIME_EVIDENCE_SCHEMA
        or payload.get("packages") != expected_packages
        or payload.get("module_versions") != expected_modules
        or not isinstance(metadata, Mapping)
        or set(metadata) != {
            "path", "sha256", "vllm_commit", "flashinfer_commit"
        }
        or not isinstance(metadata.get("path"), str)
        or not Path(str(metadata.get("path"))).is_absolute()
        or _SHA256_RE.fullmatch(str(metadata.get("sha256", ""))) is None
        or metadata.get("vllm_commit") != DSPARK_VLLM_COMMIT
        or metadata.get("flashinfer_commit") != DSPARK_FLASHINFER_COMMIT
        or capabilities != {
            "flashinfer_native_dsv4_dispatch": {
                "head_dim": DSPARK_FLASHINFER_DSV4_DISPATCH[0],
                "topk": DSPARK_FLASHINFER_DSV4_DISPATCH[1],
                "present": True,
            },
            "vllm_moe_skip_padding": True,
        }
    ):
        raise DSparkServingProfileError(
            "DSpark runtime package/commit/capability closure differs"
        )
    cache = _validate_cache_receipt(
        payload.get("max_context_tuned_cache", {})
    )
    _validate_source_receipt(payload.get("gridbook_source_environment", {}))
    unstamped = dict(payload)
    recorded = unstamped.pop("evidence_sha256")
    if recorded != _canonical_sha256(unstamped):
        raise DSparkServingProfileError(
            "DSpark runtime evidence digest is stale"
        )
    if verify_files:
        if _file_sha256(Path(str(metadata["path"]))) != metadata["sha256"]:
            raise DSparkServingProfileError("image build metadata bytes changed")
        observed_cache = inspect_flashinfer_tuned_cache(cache["path"])
        if observed_cache != cache:
            raise DSparkServingProfileError(
                "FlashInfer tuned-cache bytes/capability changed"
            )
        observed_source = require_gridbook_087_source_compatible(
            _installed_gridbook_source_root()
        ).receipt()
        if observed_source != payload["gridbook_source_environment"]:
            raise DSparkServingProfileError(
                "installed Gridbook source namespace changed"
            )
    return dict(payload)


_ROUTE_RE = re.compile(
    r"\[prismaquant-cb\] cb_gemv_kernel "
    r"model[.]layers[.]([0-9]+)[.]ffn[.]experts[.](w13|w2) "
    r"k=([0-9]+) n_sub=([0-9]+) type_size=([0-9]+) K=([0-9]+) "
    r"-> (v2|inherited) \(([^\n]*)\)"
)


def collect_route_census(path: str | os.PathLike[str]) -> dict[str, Any]:
    log_path = Path(path)
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", "replace")
    marker_count = text.count("[prismaquant-cb] cb_gemv_kernel ")
    matches = list(_ROUTE_RE.finditer(text))
    if marker_count != len(matches):
        raise DSparkServingProfileError(
            "Gridbook route log contains malformed/unscoped cb_gemv rows"
        )
    if text.count("[prismaquant-cb] cb_gemv=v2") != 1:
        raise DSparkServingProfileError(
            "Gridbook route log does not resolve cb_gemv=v2 exactly once"
        )
    fallback_markers = (
        "WARNING: CB-GEMV-v2 unavailable",
        "every CB layer on that device decodes on the INHERITED kernel",
        "persistent-B decode-in-mainloop",
    )
    if any(marker in text for marker in fallback_markers):
        raise DSparkServingProfileError(
            "Gridbook route log selected fallback or persistent-B execution"
        )
    routes: list[dict[str, Any]] = []
    route_ids: set[str] = set()
    fallback_count = 0
    for match in matches:
        layer, stack, k_bits, n_sub, type_size, width, kernel, reason = (
            match.groups()
        )
        layer_id = int(layer)
        k_value = int(k_bits)
        n_sub_value = int(n_sub)
        type_size_value = int(type_size)
        width_value = int(width)
        route_id = f"model.layers.{layer_id}.ffn.experts.{stack}"
        if route_id in route_ids:
            raise DSparkServingProfileError(
                f"Gridbook route census duplicates {route_id}"
            )
        route_ids.add(route_id)
        if layer_id < DSPARK_TARGET_LAYER_COUNT:
            role = "target"
        elif layer_id < DSPARK_TARGET_LAYER_COUNT + DSPARK_DRAFT_LAYER_COUNT:
            role = "draft"
        else:
            raise DSparkServingProfileError(
                f"Gridbook route census has unscoped layer {layer_id}"
            )
        if width_value != (4096 if stack == "w13" else 2048):
            raise DSparkServingProfileError(
                f"Gridbook route {route_id} has the wrong K width"
            )
        if n_sub_value == 2 and type_size_value == 4 * k_value + 9:
            family = "fp4_v2"
            if kernel != "v2" or reason != "mode=v2":
                fallback_count += 1
        elif (
            role == "target"
            and n_sub_value == 4
            and type_size_value == 4 * k_value
        ):
            family = "fp8"
            if kernel != "inherited" or reason != "not fp4-CB two-tier v2":
                fallback_count += 1
        else:
            raise DSparkServingProfileError(
                f"Gridbook route {route_id} has an unscoped format"
            )
        if role == "draft" and (family != "fp4_v2" or k_value != 12):
            raise DSparkServingProfileError(
                f"Gridbook draft route {route_id} is not uniform K12 FP4-v2"
            )
        routes.append({
            "route_id": route_id,
            "artifact_role": role,
            "layer": layer_id,
            "stack": stack,
            "format_family": family,
            "k_bits": k_value,
            "n_sub": n_sub_value,
            "type_size": type_size_value,
            "in_features": width_value,
            "kernel": kernel,
            "reason": reason,
        })
    routes.sort(key=lambda row: (row["layer"], 0 if row["stack"] == "w13" else 1))
    counts = {
        "target_fp4_v2": sum(
            row["artifact_role"] == "target"
            and row["format_family"] == "fp4_v2"
            and row["kernel"] == "v2"
            for row in routes
        ),
        "draft_fp4_v2": sum(
            row["artifact_role"] == "draft"
            and row["format_family"] == "fp4_v2"
            and row["kernel"] == "v2"
            for row in routes
        ),
        "fp8_inherited": sum(
            row["format_family"] == "fp8"
            and row["kernel"] == "inherited"
            for row in routes
        ),
        "fallback": fallback_count,
        "unscoped": marker_count - len(matches),
        "total": len(routes),
    }
    expected_counts = {
        "target_fp4_v2": DSPARK_TARGET_FP4_V2_ROUTES,
        "draft_fp4_v2": DSPARK_DRAFT_FP4_V2_ROUTES,
        "fp8_inherited": DSPARK_FP8_INHERITED_ROUTES,
        "fallback": 0,
        "unscoped": 0,
        "total": DSPARK_TOTAL_ROUTES,
    }
    expected_ids = {
        f"model.layers.{layer}.ffn.experts.{stack}"
        for layer in range(DSPARK_TARGET_LAYER_COUNT + DSPARK_DRAFT_LAYER_COUNT)
        for stack in ("w13", "w2")
    }
    if counts != expected_counts or route_ids != expected_ids:
        raise DSparkServingProfileError(
            f"Gridbook route census differs: {counts!r}"
        )
    receipt: dict[str, Any] = {
        "schema": DSPARK_ROUTE_CENSUS_SCHEMA,
        "profile_sha256": DSPARK_SERVING_PROFILE_SHA256,
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
        "gemv_mode": "v2",
        "counts": counts,
        "routes": routes,
        "routes_sha256": _canonical_sha256(routes),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_route_census(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema", "profile_sha256", "serve_log_sha256", "gemv_mode",
            "counts", "routes", "routes_sha256", "receipt_sha256",
        },
        where="DSpark route census",
    )
    routes = payload.get("routes")
    if not isinstance(routes, list):
        raise DSparkServingProfileError("DSpark route ledger is missing")
    route_keys = {
        "route_id", "artifact_role", "layer", "stack", "format_family",
        "k_bits", "n_sub", "type_size", "in_features", "kernel", "reason",
    }
    if any(
        not isinstance(row, Mapping) or set(row) != route_keys
        for row in routes
    ):
        raise DSparkServingProfileError("DSpark route ledger row keys differ")
    expected_counts = {
        "target_fp4_v2": DSPARK_TARGET_FP4_V2_ROUTES,
        "draft_fp4_v2": DSPARK_DRAFT_FP4_V2_ROUTES,
        "fp8_inherited": DSPARK_FP8_INHERITED_ROUTES,
        "fallback": 0,
        "unscoped": 0,
        "total": DSPARK_TOTAL_ROUTES,
    }
    ids: set[str] = set()
    replay = {name: 0 for name in expected_counts}
    replay["total"] = len(routes)
    for index, row in enumerate(routes):
        layer = row.get("layer")
        stack = row.get("stack")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or layer >= DSPARK_TARGET_LAYER_COUNT + DSPARK_DRAFT_LAYER_COUNT
            or stack not in {"w13", "w2"}
        ):
            raise DSparkServingProfileError(
                f"DSpark route ledger row {index} is unscoped"
            )
        route_id = f"model.layers.{layer}.ffn.experts.{stack}"
        role = "target" if layer < DSPARK_TARGET_LAYER_COUNT else "draft"
        if row.get("route_id") != route_id or row.get("artifact_role") != role:
            raise DSparkServingProfileError(
                f"DSpark route ledger row {index} identity differs"
            )
        if route_id in ids:
            raise DSparkServingProfileError("DSpark route ledger duplicates a route")
        ids.add(route_id)
        width = 4096 if stack == "w13" else 2048
        if row.get("in_features") != width:
            raise DSparkServingProfileError("DSpark route width differs")
        family = row.get("format_family")
        k_bits = row.get("k_bits")
        if (
            isinstance(k_bits, bool)
            or not isinstance(k_bits, int)
            or isinstance(row.get("n_sub"), bool)
            or not isinstance(row.get("n_sub"), int)
            or isinstance(row.get("type_size"), bool)
            or not isinstance(row.get("type_size"), int)
        ):
            raise DSparkServingProfileError("DSpark route numeric fields differ")
        if family == "fp4_v2":
            if (
                row.get("n_sub") != 2
                or row.get("type_size") != 4 * k_bits + 9
                or row.get("kernel") != "v2"
                or row.get("reason") != "mode=v2"
                or (role == "draft" and k_bits != 12)
            ):
                raise DSparkServingProfileError("DSpark FP4-v2 route differs")
            replay[f"{role}_fp4_v2"] += 1
        elif family == "fp8" and role == "target":
            if (
                row.get("n_sub") != 4
                or row.get("type_size") != 4 * k_bits
                or row.get("kernel") != "inherited"
                or row.get("reason") != "not fp4-CB two-tier v2"
            ):
                raise DSparkServingProfileError("DSpark FP8 route differs")
            replay["fp8_inherited"] += 1
        else:
            raise DSparkServingProfileError("DSpark route format/role differs")
    replay["fallback"] = 0
    replay["unscoped"] = 0
    expected_ids = {
        f"model.layers.{layer}.ffn.experts.{stack}"
        for layer in range(DSPARK_TARGET_LAYER_COUNT + DSPARK_DRAFT_LAYER_COUNT)
        for stack in ("w13", "w2")
    }
    unstamped = dict(payload)
    recorded = unstamped.pop("receipt_sha256")
    if (
        payload.get("schema") != DSPARK_ROUTE_CENSUS_SCHEMA
        or payload.get("profile_sha256") != DSPARK_SERVING_PROFILE_SHA256
        or _SHA256_RE.fullmatch(str(payload.get("serve_log_sha256", ""))) is None
        or payload.get("gemv_mode") != "v2"
        or payload.get("counts") != expected_counts
        or replay != expected_counts
        or ids != expected_ids
        or payload.get("routes_sha256") != _canonical_sha256(routes)
        or recorded != _canonical_sha256(unstamped)
    ):
        raise DSparkServingProfileError(
            "DSpark route census does not replay to 76 FP4-v2/16 FP8-inherited"
        )
    return dict(payload)


_CACHE_PATH_RE = re.compile(
    r"Autotuning FlashInfer SM120 sparse MLA DSv4 decode with cache: ([^\n]+)"
)


def collect_cache_log_evidence(
    path: str | os.PathLike[str],
    runtime_evidence: Mapping[str, Any],
    *,
    expected_runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated_runtime = validate_runtime_evidence(
        runtime_evidence, expected_runtime_pin=expected_runtime_pin
    )
    cache = validated_runtime["max_context_tuned_cache"]
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", "replace")
    paths = _CACHE_PATH_RE.findall(text)
    expected_path = str(cache["path"])
    hit_marker = (
        "Config cache hit for sparse_mla_sm120_decode_dsv4 "
        "(runner=SparseMlaDecodeV3Runner, source=config file)"
    )
    using_marker = f"Using FlashInfer autotune cache file: {expected_path}"
    loaded_marker = (
        "FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 0 "
        f"from {expected_path}."
    )
    if (
        paths != [expected_path]
        or text.count(hit_marker) != 1
        or text.count(using_marker) != 1
        or text.count(loaded_marker) != 1
        or "tactic=-1" in text
    ):
        raise DSparkServingProfileError(
            "serve log does not prove the exact tuned extra_topk=2048 cache hit"
        )
    receipt: dict[str, Any] = {
        "schema": DSPARK_CACHE_LOG_EVIDENCE_SCHEMA,
        "profile_sha256": DSPARK_SERVING_PROFILE_SHA256,
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
        "cache_path": expected_path,
        "cache_sha256": cache["sha256"],
        "custom_op": DSPARK_FLASHINFER_CUSTOM_OP,
        "runner": DSPARK_FLASHINFER_RUNNER,
        "extra_topk": DSPARK_FLASHINFER_EXTRA_TOPK,
        "cache_hit": True,
        "fallback_tactic_count": 0,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def validate_cache_log_evidence(
    payload: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    *,
    expected_runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = validate_runtime_evidence(
        runtime_evidence, expected_runtime_pin=expected_runtime_pin
    )
    _require_exact_keys(
        payload,
        {
            "schema", "profile_sha256", "serve_log_sha256", "cache_path",
            "cache_sha256", "custom_op", "runner", "extra_topk",
            "cache_hit", "fallback_tactic_count", "receipt_sha256",
        },
        where="DSpark FlashInfer cache-log evidence",
    )
    unstamped = dict(payload)
    recorded = unstamped.pop("receipt_sha256")
    cache = runtime["max_context_tuned_cache"]
    if (
        payload.get("schema") != DSPARK_CACHE_LOG_EVIDENCE_SCHEMA
        or payload.get("profile_sha256") != DSPARK_SERVING_PROFILE_SHA256
        or _SHA256_RE.fullmatch(str(payload.get("serve_log_sha256", ""))) is None
        or payload.get("cache_path") != cache["path"]
        or payload.get("cache_sha256") != cache["sha256"]
        or payload.get("custom_op") != DSPARK_FLASHINFER_CUSTOM_OP
        or payload.get("runner") != DSPARK_FLASHINFER_RUNNER
        or payload.get("extra_topk") != DSPARK_FLASHINFER_EXTRA_TOPK
        or payload.get("cache_hit") is not True
        or payload.get("fallback_tactic_count") != 0
        or recorded != _canonical_sha256(unstamped)
    ):
        raise DSparkServingProfileError(
            "DSpark FlashInfer cache-log evidence differs"
        )
    return dict(payload)


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DSparkServingProfileError(
                f"duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def load_runtime_evidence(
    path: str | os.PathLike[str], *, verify_files: bool = False
) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DSparkServingProfileError(
            f"cannot read DSpark runtime evidence: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DSparkServingProfileError("DSpark runtime evidence is not an object")
    return validate_runtime_evidence(payload, verify_files=verify_files)


def validate_dspark_serve_manifest(
    payload: Mapping[str, Any],
    *,
    arm: str,
    expected_served_model: str,
    requires_moe_marlin: bool,
    expected_model_sha: str | None = None,
    expected_speculative_config: Mapping[str, Any] | None = None,
    expected_launch_options: Mapping[str, str] | None = None,
    expected_launch_switches: set[str] | frozenset[str] | None = None,
    expected_runtime_pin: GridbookServingRuntimePin | Mapping[str, Any] | None = None,
) -> str:
    """Replay a generic server proof under the separate paired-DSpark profile.

    Artifact binding, listener/session identity, residency, launch argv, and
    imported Gridbook distribution remain owned by the shared endpoint
    validator.  Only its historical environment default is replaced; the
    embedded profile/runtime evidence is then replayed independently here.
    """

    pin = (
        current_gridbook_serving_pin_payload()
        if expected_runtime_pin is None else _pin_payload(expected_runtime_pin)
    )
    profile_receipt = payload.get("dspark_serving_profile")
    runtime_evidence = payload.get("dspark_runtime_evidence")
    if not isinstance(profile_receipt, Mapping) or not isinstance(
        runtime_evidence, Mapping
    ):
        raise DSparkServingProfileError(
            "serve manifest has no paired-DSpark profile/runtime evidence"
        )
    validate_serving_profile_receipt(
        profile_receipt, expected_runtime_pin=pin
    )
    validated_runtime = validate_runtime_evidence(
        runtime_evidence, expected_runtime_pin=pin
    )
    if validated_runtime.get("profile_receipt") != profile_receipt:
        raise DSparkServingProfileError(
            "serve manifest profile and runtime evidence disagree"
        )
    from .validate_cb_endpoint import (
        CBEndpointValidationError,
        validate_serve_manifest,
    )

    try:
        return validate_serve_manifest(
            payload,
            arm=arm,
            expected_image=DSPARK_IMAGE,
            expected_vllm_version=DSPARK_VLLM_VERSION,
            expected_served_model=expected_served_model,
            requires_moe_marlin=requires_moe_marlin,
            expected_model_sha=expected_model_sha,
            expected_speculative_config=expected_speculative_config,
            expected_launch_options=expected_launch_options,
            expected_launch_switches=expected_launch_switches,
            expected_server_environment=expected_server_environment(pin),
            expected_server_environment_allowlist=DSPARK_SERVER_ENV_ALLOWLIST,
        )
    except CBEndpointValidationError as exc:
        raise DSparkServingProfileError(
            f"paired-DSpark serve manifest differs: {exc}"
        ) from exc


_BASELINE_MODEL_KEYS = {
    "schema", "evidence_path", "evidence_sha256", "target_model_sha256",
    "draft_model_sha256", "model_loaded_bytes", "quant_profile",
    "identity_sha256",
}
_BASELINE_QUANT_KEYS = {
    "target_container", "target_quant_method", "target_format",
    "target_profile_id", "target_quantizable_bpp", "target_loaded_bytes",
    "draft_container", "draft_quant_method", "draft_format",
    "draft_quantizable_bpp", "draft_loaded_bytes", "total_loaded_bytes",
    "kv_cache_dtype",
}
_BASELINE_PREFILL_KEYS = {
    "schema", "name", "status", "evidence_path", "evidence_sha256",
    "prompt_sha256", "tokenizer_sha256", "requested_uncached_prompt_tokens",
    "observed_uncached_prompt_tokens", "context_start_tokens",
    "context_end_tokens", "max_model_len", "max_num_batched_tokens",
    "prefill_chunk_tokens", "prefix_cache_enabled",
    "prefix_cache_hits_before", "prefix_cache_hits_after",
    "prefix_cache_hit_delta", "concurrency", "batch_size", "wall_seconds",
    "internal_prefill_seconds", "internal_prefill_tokens",
    "internal_prefill_tokens_per_second", "endpoint_prompt_tokens_before",
    "endpoint_prompt_tokens_after", "endpoint_prompt_token_delta",
    "unavailable_reason", "row_sha256",
}
_BASELINE_DECODE_KEYS = {
    "schema", "name", "evidence_path", "evidence_sha256",
    "prompt_corpus_sha256", "tokenizer_sha256", "context_tokens",
    "max_model_len", "max_num_batched_tokens", "prefill_chunk_tokens",
    "prefix_cache_enabled", "prefix_cache_hits_before",
    "prefix_cache_hits_after", "prefix_cache_hit_delta", "concurrency",
    "batch_size", "request_count", "output_tokens_per_request",
    "observed_output_tokens_total", "prefill_included", "wall_seconds",
    "aggregate_output_tokens_per_second", "internal_decode_seconds",
    "internal_output_tokens", "internal_output_tokens_per_second",
    "endpoint_generation_tokens_before", "endpoint_generation_tokens_after",
    "endpoint_generation_token_delta", "row_sha256",
}
_BASELINE_PREFILL_DEPTHS = (2048, 65536, 517963)
_BASELINE_DECODE_SHAPES = ((12288, 1), (240000, 1), (12288, 12))
_BASELINE_REQUIRED_GAPS = frozenset({
    "engine_runtime_and_container_differ",
    "model_weight_quantization_and_loaded_bytes_differ",
    "reference_prompt_and_tokenizer_inputs_are_not_published_exactly",
    "reference_prefill_claims_are_engine_side_while_local_rows_bind_wall_and_internal_metrics",
})


def _positive_number(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DSparkServingProfileError(f"{where} is not numeric")
    result = float(value)
    if not (result > 0.0 and result < float("inf")):
        raise DSparkServingProfileError(f"{where} is not positive and finite")
    return result


def _nonnegative_integer(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DSparkServingProfileError(
            f"{where} is not a non-negative integer"
        )
    return value


def _validate_evidence_path(
    path: object,
    sha256: object,
    *,
    where: str,
    verify_files: bool,
) -> None:
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or _SHA256_RE.fullmatch(str(sha256 or "")) is None
    ):
        raise DSparkServingProfileError(
            f"{where} path/SHA-256 identity is malformed"
        )
    if verify_files:
        try:
            observed = _file_sha256(Path(path))
        except OSError as exc:
            raise DSparkServingProfileError(
                f"{where} evidence path is unreadable: {exc}"
            ) from exc
        if observed != sha256:
            raise DSparkServingProfileError(
                f"{where} evidence bytes changed"
            )


def _validate_local_model_identity(
    payload: Mapping[str, Any], *, verify_files: bool
) -> dict[str, Any]:
    _require_exact_keys(
        payload, _BASELINE_MODEL_KEYS, where="baseline local model identity"
    )
    _validate_evidence_path(
        payload.get("evidence_path"), payload.get("evidence_sha256"),
        where="baseline local model", verify_files=verify_files,
    )
    quant = payload.get("quant_profile")
    if not isinstance(quant, Mapping):
        raise DSparkServingProfileError("baseline local quant profile is missing")
    _require_exact_keys(
        quant, _BASELINE_QUANT_KEYS, where="baseline local quant profile"
    )
    for name in (
        "target_model_sha256", "draft_model_sha256", "identity_sha256"
    ):
        if _SHA256_RE.fullmatch(str(payload.get(name, ""))) is None:
            raise DSparkServingProfileError(
                f"baseline local model {name} is not SHA-256"
            )
    for name in (
        "target_loaded_bytes", "draft_loaded_bytes", "total_loaded_bytes"
    ):
        if _nonnegative_integer(quant.get(name), where=f"quant_profile.{name}") <= 0:
            raise DSparkServingProfileError(
                f"quant_profile.{name} must be positive"
            )
    if (
        quant["target_container"] != "gridbook"
        or quant["target_quant_method"] != "gridbook"
        or quant["target_format"] != "nvfp4_cb"
        or not isinstance(quant["target_profile_id"], str)
        or not quant["target_profile_id"]
        or quant["draft_container"] != "gridbook"
        or quant["draft_quant_method"] != "gridbook"
        or quant["draft_format"] != "NVFP4_CB_K12"
        or quant["kv_cache_dtype"] != DSPARK_KV_CACHE_DTYPE
        or quant["total_loaded_bytes"]
        != quant["target_loaded_bytes"] + quant["draft_loaded_bytes"]
        or payload.get("model_loaded_bytes") != quant["total_loaded_bytes"]
    ):
        raise DSparkServingProfileError(
            "baseline local model/quant profile differs from the paired CB lane"
        )
    _positive_number(
        quant["target_quantizable_bpp"],
        where="quant_profile.target_quantizable_bpp",
    )
    _positive_number(
        quant["draft_quantizable_bpp"],
        where="quant_profile.draft_quantizable_bpp",
    )
    unstamped = dict(payload)
    recorded = unstamped.pop("identity_sha256")
    if recorded != _canonical_sha256(unstamped):
        raise DSparkServingProfileError(
            "baseline local model identity digest is stale"
        )
    return dict(payload)


def _validate_prefill_row(
    payload: Mapping[str, Any],
    *,
    expected_tokens: int,
    verify_files: bool,
) -> dict[str, Any]:
    _require_exact_keys(
        payload, _BASELINE_PREFILL_KEYS,
        where=f"baseline prefill row {expected_tokens}",
    )
    _validate_evidence_path(
        payload.get("evidence_path"), payload.get("evidence_sha256"),
        where=f"baseline prefill row {expected_tokens}",
        verify_files=verify_files,
    )
    if (
        payload.get("schema") != DSPARK_BASELINE_PREFILL_ROW_SCHEMA
        or payload.get("name") != f"uncached-context-{expected_tokens}"
        or payload.get("requested_uncached_prompt_tokens") != expected_tokens
        or _SHA256_RE.fullmatch(str(payload.get("prompt_sha256", ""))) is None
        or _SHA256_RE.fullmatch(str(payload.get("tokenizer_sha256", ""))) is None
        or payload.get("max_num_batched_tokens")
        != DSPARK_MAX_NUM_BATCHED_TOKENS
        or payload.get("prefill_chunk_tokens")
        != DSPARK_MAX_NUM_BATCHED_TOKENS
        or payload.get("prefix_cache_enabled") is not False
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
    ):
        raise DSparkServingProfileError(
            f"baseline prefill row {expected_tokens} static contract differs"
        )
    max_model_len = _nonnegative_integer(
        payload.get("max_model_len"), where="prefill.max_model_len"
    )
    status = payload.get("status")
    if status == "collected":
        observed = payload.get("observed_uncached_prompt_tokens")
        start = payload.get("context_start_tokens")
        end = payload.get("context_end_tokens")
        if (
            observed != expected_tokens
            or start != 0
            or end != expected_tokens
            or max_model_len < expected_tokens
            or payload.get("prefix_cache_hit_delta") != 0
            or payload.get("prefix_cache_hits_after")
            != payload.get("prefix_cache_hits_before")
            or payload.get("unavailable_reason") is not None
        ):
            raise DSparkServingProfileError(
                f"baseline prefill row {expected_tokens} is not uncached/complete"
            )
        for name in (
            "prefix_cache_hits_before", "prefix_cache_hits_after",
            "endpoint_prompt_tokens_before", "endpoint_prompt_tokens_after",
            "endpoint_prompt_token_delta", "internal_prefill_tokens",
        ):
            _nonnegative_integer(payload.get(name), where=f"prefill.{name}")
        if (
            payload["prefix_cache_hits_before"] != 0
            or payload["endpoint_prompt_token_delta"] != expected_tokens
            or payload["endpoint_prompt_tokens_after"]
            - payload["endpoint_prompt_tokens_before"] != expected_tokens
            or payload["internal_prefill_tokens"] != expected_tokens
        ):
            raise DSparkServingProfileError(
                f"baseline prefill row {expected_tokens} counters differ"
            )
        wall = _positive_number(
            payload.get("wall_seconds"), where="prefill.wall_seconds"
        )
        internal = _positive_number(
            payload.get("internal_prefill_seconds"),
            where="prefill.internal_prefill_seconds",
        )
        internal_tps = _positive_number(
            payload.get("internal_prefill_tokens_per_second"),
            where="prefill.internal_prefill_tokens_per_second",
        )
        if (
            not math.isclose(
                internal_tps, expected_tokens / internal,
                rel_tol=1e-12, abs_tol=1e-12,
            )
            or internal > wall
        ):
            raise DSparkServingProfileError(
                f"baseline prefill row {expected_tokens} timing arithmetic differs"
            )
    elif status == "unavailable":
        nullable = (
            "observed_uncached_prompt_tokens", "context_end_tokens",
            "prefix_cache_hits_before", "prefix_cache_hits_after",
            "prefix_cache_hit_delta", "wall_seconds", "internal_prefill_seconds",
            "internal_prefill_tokens", "internal_prefill_tokens_per_second",
            "endpoint_prompt_tokens_before", "endpoint_prompt_tokens_after",
            "endpoint_prompt_token_delta",
        )
        if (
            expected_tokens != 517963
            or max_model_len >= expected_tokens
            or payload.get("context_start_tokens") != 0
            or any(payload.get(name) is not None for name in nullable)
            or payload.get("unavailable_reason")
            != "requested_context_exceeds_selected_limit"
        ):
            raise DSparkServingProfileError(
                "only the 517963 frontier may be unavailable for context limit"
            )
    else:
        raise DSparkServingProfileError(
            f"baseline prefill row {expected_tokens} status differs"
        )
    unstamped = dict(payload)
    recorded = unstamped.pop("row_sha256")
    if recorded != _canonical_sha256(unstamped):
        raise DSparkServingProfileError(
            f"baseline prefill row {expected_tokens} digest is stale"
        )
    return dict(payload)


def _validate_decode_row(
    payload: Mapping[str, Any],
    *,
    context_tokens: int,
    concurrency: int,
    verify_files: bool,
) -> dict[str, Any]:
    _require_exact_keys(
        payload, _BASELINE_DECODE_KEYS,
        where=f"baseline decode row {context_tokens}/c{concurrency}",
    )
    _validate_evidence_path(
        payload.get("evidence_path"), payload.get("evidence_sha256"),
        where=f"baseline decode row {context_tokens}/c{concurrency}",
        verify_files=verify_files,
    )
    if (
        payload.get("schema") != DSPARK_BASELINE_DECODE_ROW_SCHEMA
        or payload.get("name") != f"context-{context_tokens}-concurrency-{concurrency}"
        or payload.get("context_tokens") != context_tokens
        or payload.get("concurrency") != concurrency
        or payload.get("batch_size") != concurrency
        or payload.get("request_count") != concurrency
        or _SHA256_RE.fullmatch(
            str(payload.get("prompt_corpus_sha256", ""))
        ) is None
        or _SHA256_RE.fullmatch(str(payload.get("tokenizer_sha256", ""))) is None
        or payload.get("max_num_batched_tokens")
        != DSPARK_MAX_NUM_BATCHED_TOKENS
        or payload.get("prefill_chunk_tokens")
        != DSPARK_MAX_NUM_BATCHED_TOKENS
        or payload.get("prefix_cache_enabled") is not False
        or payload.get("prefix_cache_hit_delta") != 0
        or payload.get("prefix_cache_hits_before") != 0
        or payload.get("prefix_cache_hits_after") != 0
        or payload.get("prefill_included") is not False
    ):
        raise DSparkServingProfileError(
            f"baseline decode row {context_tokens}/c{concurrency} contract differs"
        )
    if _nonnegative_integer(
        payload.get("max_model_len"), where="decode.max_model_len"
    ) < context_tokens:
        raise DSparkServingProfileError("baseline decode context exceeds run limit")
    output_per_request = _nonnegative_integer(
        payload.get("output_tokens_per_request"),
        where="decode.output_tokens_per_request",
    )
    if output_per_request <= 0:
        raise DSparkServingProfileError(
            "baseline decode output_tokens_per_request must be positive"
        )
    total = output_per_request * concurrency
    for name in (
        "observed_output_tokens_total", "internal_output_tokens",
        "endpoint_generation_tokens_before", "endpoint_generation_tokens_after",
        "endpoint_generation_token_delta",
    ):
        _nonnegative_integer(payload.get(name), where=f"decode.{name}")
    if (
        payload["observed_output_tokens_total"] != total
        or payload["internal_output_tokens"] != total
        or payload["endpoint_generation_token_delta"] != total
        or payload["endpoint_generation_tokens_after"]
        - payload["endpoint_generation_tokens_before"] != total
    ):
        raise DSparkServingProfileError(
            f"baseline decode row {context_tokens}/c{concurrency} counters differ"
        )
    wall = _positive_number(
        payload.get("wall_seconds"), where="decode.wall_seconds"
    )
    aggregate = _positive_number(
        payload.get("aggregate_output_tokens_per_second"),
        where="decode.aggregate_output_tokens_per_second",
    )
    internal_seconds = _positive_number(
        payload.get("internal_decode_seconds"),
        where="decode.internal_decode_seconds",
    )
    internal_tps = _positive_number(
        payload.get("internal_output_tokens_per_second"),
        where="decode.internal_output_tokens_per_second",
    )
    if (
        not math.isclose(aggregate, total / wall, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(
            internal_tps, total / internal_seconds,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    ):
        raise DSparkServingProfileError(
            f"baseline decode row {context_tokens}/c{concurrency} arithmetic differs"
        )
    unstamped = dict(payload)
    recorded = unstamped.pop("row_sha256")
    if recorded != _canonical_sha256(unstamped):
        raise DSparkServingProfileError(
            f"baseline decode row {context_tokens}/c{concurrency} digest is stale"
        )
    return dict(payload)


def stamp_baseline_unit(payload: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    """Return a canonical self-digest-bound local model or measurement row."""

    if digest_field not in {"identity_sha256", "row_sha256"}:
        raise DSparkServingProfileError("unsupported baseline unit digest field")
    result = dict(payload)
    result.pop(digest_field, None)
    result[digest_field] = _canonical_sha256(result)
    return result


def build_baseline_comparison_evidence(
    *,
    local_model: Mapping[str, Any],
    prefill_frontier: Sequence[Mapping[str, Any]],
    decode_concurrency: Sequence[Mapping[str, Any]],
    comparability_gaps: Sequence[str],
) -> dict[str, Any]:
    """Build and replay a reference-only Entrpi comparison receipt."""

    prefill = [dict(row) for row in prefill_frontier]
    decode = [dict(row) for row in decode_concurrency]
    missing = [
        row.get("name") for row in prefill
        if row.get("status") != "collected"
    ]
    gaps = sorted(set(comparability_gaps))
    payload: dict[str, Any] = {
        "schema": DSPARK_BASELINE_COMPARISON_SCHEMA,
        "profile_sha256": DSPARK_SERVING_PROFILE_SHA256,
        "reference": json.loads(json.dumps(dict(ENTRPI_DSV4_REFERENCE))),
        "reference_sha256": ENTRPI_DSV4_REFERENCE_SHA256,
        "local_model": dict(local_model),
        "prefill_frontier": prefill,
        "decode_concurrency": decode,
        "comparability": {
            "reference_only": True,
            "derives_release_pass": False,
            "comparison_complete": not missing,
            "thresholds": [],
            "missing_measurements": missing,
            "gaps": gaps,
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return validate_baseline_comparison_evidence(payload)


def validate_baseline_comparison_evidence(
    payload: Mapping[str, Any],
    *,
    verify_files: bool = False,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Replay exact local metrics without treating Entrpi claims as a gate."""

    _require_exact_keys(
        payload,
        {
            "schema", "profile_sha256", "reference", "reference_sha256",
            "local_model", "prefill_frontier", "decode_concurrency",
            "comparability", "evidence_sha256",
        },
        where="DSpark baseline comparison",
    )
    expected_reference = json.loads(json.dumps(dict(ENTRPI_DSV4_REFERENCE)))
    if (
        payload.get("schema") != DSPARK_BASELINE_COMPARISON_SCHEMA
        or payload.get("profile_sha256") != DSPARK_SERVING_PROFILE_SHA256
        or payload.get("reference") != expected_reference
        or payload.get("reference_sha256") != ENTRPI_DSV4_REFERENCE_SHA256
    ):
        raise DSparkServingProfileError(
            "DSpark baseline reference/profile identity differs"
        )
    local_model = payload.get("local_model")
    if not isinstance(local_model, Mapping):
        raise DSparkServingProfileError("DSpark baseline local model is missing")
    _validate_local_model_identity(local_model, verify_files=verify_files)
    prefill = payload.get("prefill_frontier")
    if not isinstance(prefill, list) or len(prefill) != len(
        _BASELINE_PREFILL_DEPTHS
    ):
        raise DSparkServingProfileError(
            "DSpark baseline prefill frontier is incomplete"
        )
    for row, depth in zip(prefill, _BASELINE_PREFILL_DEPTHS, strict=True):
        if not isinstance(row, Mapping):
            raise DSparkServingProfileError("DSpark baseline prefill row is malformed")
        _validate_prefill_row(
            row, expected_tokens=depth, verify_files=verify_files
        )
    decode = payload.get("decode_concurrency")
    if not isinstance(decode, list) or len(decode) != len(
        _BASELINE_DECODE_SHAPES
    ):
        raise DSparkServingProfileError(
            "DSpark baseline decode/concurrency frontier is incomplete"
        )
    for row, (context, concurrency) in zip(
        decode, _BASELINE_DECODE_SHAPES, strict=True
    ):
        if not isinstance(row, Mapping):
            raise DSparkServingProfileError("DSpark baseline decode row is malformed")
        _validate_decode_row(
            row,
            context_tokens=context,
            concurrency=concurrency,
            verify_files=verify_files,
        )
    missing = [
        row["name"] for row in prefill if row.get("status") != "collected"
    ]
    comparability = payload.get("comparability")
    if not isinstance(comparability, Mapping):
        raise DSparkServingProfileError(
            "DSpark baseline comparability closure is missing"
        )
    _require_exact_keys(
        comparability,
        {
            "reference_only", "derives_release_pass", "comparison_complete",
            "thresholds", "missing_measurements", "gaps",
        },
        where="DSpark baseline comparability",
    )
    gaps = comparability.get("gaps")
    if (
        not isinstance(gaps, list)
        or any(not isinstance(gap, str) or not gap for gap in gaps)
    ):
        raise DSparkServingProfileError(
            "DSpark baseline comparability gaps are malformed"
        )
    required_gaps = set(_BASELINE_REQUIRED_GAPS)
    if missing:
        required_gaps.add(
            "selected_release_context_limit_below_entrpi_517963_frontier"
        )
    if (
        comparability.get("reference_only") is not True
        or comparability.get("derives_release_pass") is not False
        or comparability.get("comparison_complete") is not (not missing)
        or comparability.get("thresholds") != []
        or comparability.get("missing_measurements") != missing
        or gaps != sorted(set(gaps))
        or not required_gaps.issubset(gaps)
    ):
        raise DSparkServingProfileError(
            "DSpark baseline comparability gaps/completeness differ"
        )
    if require_complete and missing:
        raise DSparkServingProfileError(
            "DSpark baseline comparison is incomplete: " + ", ".join(missing)
        )
    unstamped = dict(payload)
    recorded = unstamped.pop("evidence_sha256")
    if recorded != _canonical_sha256(unstamped):
        raise DSparkServingProfileError(
            "DSpark baseline comparison digest is stale"
        )
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--build-metadata", default="/workspace/build-metadata.yaml"
    )
    parser.add_argument("--flashinfer-cache", required=True)
    parser.add_argument("--gridbook-source-root")
    args = parser.parse_args(argv)
    evidence = collect_runtime_evidence(
        build_metadata_path=args.build_metadata,
        flashinfer_cache_path=args.flashinfer_cache,
        gridbook_source_root=args.gridbook_source_root,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


__all__ = [
    "DSPARK_BASELINE_COMPARISON_SCHEMA",
    "DSPARK_BASELINE_DECODE_ROW_SCHEMA",
    "DSPARK_BASELINE_LOCAL_MODEL_SCHEMA",
    "DSPARK_BASELINE_PREFILL_ROW_SCHEMA",
    "DSPARK_CACHE_LOG_EVIDENCE_SCHEMA",
    "DSPARK_FLASHINFER_COMMIT",
    "DSPARK_FLASHINFER_DISTRIBUTION",
    "DSPARK_FLASHINFER_DSV4_DISPATCH",
    "DSPARK_FLASHINFER_EXTRA_TOPK",
    "DSPARK_FLASHINFER_VERSION",
    "DSPARK_GRIDBOOK_ENVIRONMENT_ALLOWLIST",
    "DSPARK_IMAGE",
    "DSPARK_PROFILE_ENVIRONMENT",
    "DSPARK_PROFILE_SET_ENVIRONMENT",
    "DSPARK_ROUTE_CENSUS_SCHEMA",
    "DSPARK_RUNTIME_EVIDENCE_SCHEMA",
    "DSPARK_SERVER_ENV_ALLOWLIST",
    "DSPARK_SERVING_PROFILE",
    "DSPARK_SERVING_PROFILE_RECEIPT_SCHEMA",
    "DSPARK_SERVING_PROFILE_SCHEMA",
    "DSPARK_SERVING_PROFILE_SHA256",
    "DSPARK_TORCH_VERSION",
    "DSPARK_VLLM_COMMIT",
    "DSPARK_VLLM_VERSION",
    "ENTRPI_DSV4_REFERENCE",
    "ENTRPI_DSV4_REFERENCE_SHA256",
    "DSparkServingProfile",
    "DSparkServingProfileError",
    "Gridbook087SourceEnvironmentScan",
    "apply_profile_environment",
    "attest_profile_environment",
    "build_baseline_comparison_evidence",
    "collect_cache_log_evidence",
    "collect_route_census",
    "collect_runtime_evidence",
    "current_gridbook_serving_pin_payload",
    "expected_server_environment",
    "inspect_flashinfer_tuned_cache",
    "load_runtime_evidence",
    "require_gridbook_087_source_compatible",
    "scan_gridbook_087_source_environment",
    "serving_profile_receipt",
    "snapshot_profile_environment",
    "stamp_baseline_unit",
    "validate_baseline_comparison_evidence",
    "validate_cache_log_evidence",
    "validate_dspark_serve_manifest",
    "validate_route_census",
    "validate_runtime_evidence",
    "validate_serving_profile_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
