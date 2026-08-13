#!/usr/bin/env python3
"""Fail-closed Gridbook CB load/generation gate over an OpenAI endpoint.

Unlike :mod:`prismaquant.validate_native_export`, this module does not create a
vLLM engine.  The Gridbook runtime lives in a separately pinned serving
container, so this validator talks to that already-running server and binds its
verdict to a server-side ``serve_manifest.json``.  It proves only three facts:

* the requested CB artifact is the artifact named by the shipcard;
* the exact pinned Gridbook/eugr stack served in the requested eager/graph arm;
* two identical greedy requests returned the same non-empty completion.

It deliberately makes no quality or throughput claim.  Those remain the
``ship_gate`` and gold-lane slots.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .shipcard import (
    _manifest_gridbook_runtime_pin,
    _verify_gridbook_distribution_identity,
    compute_model_sha,
    fill_slot,
    git_provenance,
    load_shipcard,
    make_record,
)
from .gridbook_assignment import artifact_requires_moe_backend_marlin
from .gridbook_environment import CANONICAL_GOLD_ENVIRONMENT
from .gridbook_runtime_pin import (
    GridbookRuntimePinError,
    load_gridbook_runtime_pin,
    require_exact_gridbook_runtime_release,
)
from .allocator_candidates import ROUTE_PENDING_PASSTHROUGH_FORMATS


DSV4_SPARK_VLLM_IMAGE = (
    "eugr/spark-vllm@sha256:"
    "7bf752a9fa225b528b27c6a1118cb1727cddd7c383096d83281010c4f8b407bc"
)
DSV4_SPARK_VLLM_VERSION = "0.26.1rc1.dev515+g653ebb52d.d20260808"
DSV4_SPARK_VLLM_COMMIT = "653ebb52d"
DSV4_SPARK_GPU_NAME = "NVIDIA GB10"
DSV4_MXFP4_WIRE_ID = "mxfp4_e2m1_ue8m0_g32"
DSV4_GRAPH_COMPILATION_CONFIG = (
    '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
    '"cudagraph_capture_sizes":[1]}'
)
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 8
DSV4_SERVED_MODEL_PREFIX = "dsv4-flash-gridbook-"
ENDPOINT_CONTRACT_SCHEMA = "prismaquant.cb_endpoint_contract.v1"
ARTIFACT_DECODE_CONTRACT_SCHEMA = (
    "prismaquant.cb_artifact_decode_contract.v1"
)
ENDPOINT_RESULT_SCHEMA = "prismaquant.cb_endpoint_validation.v1"
ENDPOINT_SESSION_SCHEMA = "prismaquant.cb_endpoint_session.v1"
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_GRAPH_CAPTURE_RE = re.compile(
    r"Graph capturing finished in [0-9]+ secs, took -?[0-9.]+ GiB"
)
_SESSION_MODEL_RE = re.compile(
    re.escape(DSV4_SERVED_MODEL_PREFIX) + r"[0-9a-f]{32}"
)
_FULL_DECODE_CONFIG_RE = re.compile(
    r"Initializing a V1 LLM engine .*?compilation_config=\{.*?"
    r"'cudagraph_mode': <CUDAGraphMode\.FULL_DECODE_ONLY:.*?"
    r"'cudagraph_capture_sizes': \[1\]",
)
_GRIDBOOK_NATIVE_EXTENSION_RE = re.compile(
    r"^(?:prismaquant_cb(?:_v2)?_ext|pq_cb_|pq_mxfp8_dense_|"
    r"pq_fp8_source_w8a16_)"
)
_ENDPOINT_SERVE_ENVIRONMENT = {
    **dict(CANONICAL_GOLD_ENVIRONMENT),
    # Native eager/graph gates intentionally preload all extension families so
    # both arms start from matching residency.  This is the sole override of
    # the canonical Gridbook gold profile.
    "PRISMAQUANT_PRELOAD_FUSED": "1",
    # The endpoint runner mounts one persistent, non-/tmp build cache.  This
    # residency/build input is path identity rather than a numerical lever,
    # but it is still attested exactly because Gridbook reads it at import.
    "PRISMAQUANT_CB_EXT_DIR": "/opt/gridbook/ext-cache",
}
_REQUIRED_SERVE_ENV = {
    name: value
    for name, value in _ENDPOINT_SERVE_ENVIRONMENT.items()
    if value is not None
}
_SERVE_VALUE_FLAGS = frozenset({
    "--served-model-name",
    "--host",
    "--port",
    "--tokenizer-mode",
    "--generation-config",
    "--quantization",
    "--tensor-parallel-size",
    "--kv-cache-dtype",
    "--kv-cache-memory-bytes",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--gpu-memory-utilization",
    "--compilation-config",
    "--moe-backend",
})
_SERVE_SWITCH_FLAGS = frozenset({
    "--trust-remote-code",
    "--no-enable-prefix-caching",
    "--enforce-eager",
})


class CBEndpointValidationError(RuntimeError):
    """The CB endpoint or its serving identity did not satisfy the gate."""


def _gridbook_runtime_pin() -> dict[str, Any]:
    try:
        pin = load_gridbook_runtime_pin()
        require_exact_gridbook_runtime_release(pin)
    except (GridbookRuntimePinError, OSError, UnicodeError) as exc:
        raise CBEndpointValidationError(
            f"Gridbook runtime pin is not one strict released commit: {exc}"
        ) from exc
    return {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "required_abi_features": dict(pin.required_abi_features),
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_session_model_name(value: object) -> str:
    if not isinstance(value, str) or _SESSION_MODEL_RE.fullmatch(value) is None:
        raise CBEndpointValidationError(
            "served model name must contain the fixed DSv4 prefix plus a fresh "
            "128-bit lowercase hexadecimal session nonce"
        )
    return value


def _expected_serve_environment(runtime_pin: Mapping[str, Any]) -> dict[str, str]:
    """Exact set-valued process environment for the endpoint/perf profile."""
    commit = runtime_pin.get("commit")
    version = runtime_pin.get("version")
    if not isinstance(commit, str) or not isinstance(version, str):
        raise CBEndpointValidationError("Gridbook runtime pin is incomplete")
    return {
        **_REQUIRED_SERVE_ENV,
        "PQ_GRIDBOOK_RUNTIME_COMMIT": commit,
        "PQ_GRIDBOOK_RUNTIME_VERSION": version,
        "PYTHONSAFEPATH": "1",
    }


def _validate_closed_server_environment(
    payload: Mapping[str, Any],
    *,
    runtime_pin: Mapping[str, Any],
) -> dict[str, str]:
    """Replay the complete allowlist against every inspected server process."""
    try:
        from tools.serve_fingerprint import SERVER_ENV_ALLOWLIST
    except Exception as exc:  # pragma: no cover - packaging misuse
        raise CBEndpointValidationError(
            "tools.serve_fingerprint is unavailable; run from the PrismaQuant tree"
        ) from exc

    expected = _expected_serve_environment(runtime_pin)
    process_environment = payload.get("server_process_environment")
    pq_env = payload.get("pq_env")
    rows = process_environment.get("processes") if isinstance(
        process_environment, Mapping
    ) else None
    readable_pids = process_environment.get("readable_pids") if isinstance(
        process_environment, Mapping
    ) else None
    processes = payload.get("processes")
    process_pids = [
        row.get("pid") for row in processes
        if isinstance(row, Mapping)
    ] if isinstance(processes, list) else []
    if (
        not isinstance(process_environment, Mapping)
        or set(process_environment) != {
            "schema", "allowlist", "readable_pids", "unreadable_pids",
            "consistent", "values", "processes",
        }
        or process_environment.get("schema")
        != "prismaquant.server_process_environment/1"
        or process_environment.get("allowlist") != sorted(SERVER_ENV_ALLOWLIST)
        or process_environment.get("unreadable_pids") != []
        or process_environment.get("consistent") is not True
        or process_environment.get("values") != expected
        or pq_env != expected
        or not isinstance(rows, list)
        or not rows
        or not isinstance(readable_pids, list)
        or sorted(readable_pids) != sorted(process_pids)
        or len(readable_pids) != len(set(readable_pids))
        or len(rows) != len(process_pids)
    ):
        raise CBEndpointValidationError(
            "serve manifest environment is not the exact live-process Gridbook "
            f"contract; expected exact values for {sorted(expected)}"
        )
    row_pids: list[int] = []
    expected_sha = _canonical_json_sha256(expected)
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"pid", "values", "sha256"}
            or not isinstance(row.get("pid"), int)
            or isinstance(row.get("pid"), bool)
            or row.get("values") != expected
            or row.get("sha256") != expected_sha
        ):
            raise CBEndpointValidationError(
                f"serve manifest process environment row {index} differs from contract"
            )
        row_pids.append(int(row["pid"]))
    if sorted(row_pids) != sorted(process_pids) or len(row_pids) != len(
        set(row_pids)
    ):
        raise CBEndpointValidationError(
            "serve manifest process environment rows do not cover its process census"
        )
    return expected


def _validate_live_server_session(
    payload: Mapping[str, Any],
    *,
    expected_served_model: str,
) -> dict[str, Any]:
    """Replay process/session/listener ownership and the live models response."""
    try:
        from tools.serve_fingerprint import (
            models_endpoint_binding_identity,
            process_identity_sha256,
            serve_session_fingerprint,
        )
    except Exception as exc:  # pragma: no cover - packaging misuse
        raise CBEndpointValidationError(
            "tools.serve_fingerprint is unavailable; run from the PrismaQuant tree"
        ) from exc

    host = payload.get("host_identity")
    machine_sha = host.get("machine_id_sha256") if isinstance(
        host, Mapping
    ) else None
    if (
        not isinstance(host, Mapping)
        or set(host) != {"hostname", "boot_id", "machine_id_sha256", "pid_namespace"}
        or host.get("hostname") != payload.get("hostname")
        or not isinstance(host.get("hostname"), str)
        or not host.get("hostname")
        or not isinstance(host.get("boot_id"), str)
        or not host.get("boot_id")
        or (
            machine_sha is not None
            and (
                not isinstance(machine_sha, str)
                or _FINGERPRINT_RE.fullmatch(machine_sha) is None
            )
        )
        or not isinstance(host.get("pid_namespace"), str)
        or not host.get("pid_namespace")
    ):
        raise CBEndpointValidationError(
            "serve manifest has no complete stable host identity"
        )

    launch_argv = payload.get("launch_argv")
    processes = payload.get("processes")
    if not isinstance(launch_argv, list) or not isinstance(processes, list) or not processes:
        raise CBEndpointValidationError(
            "serve manifest has no concrete server process census"
        )
    process_ids: list[int] = []
    process_hashes: list[str] = []
    launch_pids: list[int] = []
    compact_processes: list[dict[str, Any]] = []
    for index, process in enumerate(processes):
        pid = process.get("pid") if isinstance(process, Mapping) else None
        argv = process.get("argv") if isinstance(process, Mapping) else None
        identity_sha = process.get("identity_sha256") if isinstance(
            process, Mapping
        ) else None
        if (
            not isinstance(process, Mapping)
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(token, str) or not token for token in argv)
            or process.get("cmdline") != " ".join(argv)
            or isinstance(process.get("start_time_ticks"), bool)
            or not isinstance(process.get("start_time_ticks"), int)
            or process.get("start_time_ticks", -1) < 0
            or not isinstance(process.get("pid_namespace"), str)
            or not process.get("pid_namespace")
            or not isinstance(process.get("executable"), str)
            or not process.get("executable")
            or not isinstance(identity_sha, str)
            or _FINGERPRINT_RE.fullmatch(identity_sha) is None
            or identity_sha != process_identity_sha256(
                process, boot_id=str(host["boot_id"])
            )
        ):
            raise CBEndpointValidationError(
                f"serve manifest process {index} identity is malformed or stale"
            )
        process_ids.append(pid)
        process_hashes.append(identity_sha)
        compact_processes.append({"pid": pid, "identity_sha256": identity_sha})
        if argv == launch_argv:
            launch_pids.append(pid)
    if (
        len(process_ids) != len(set(process_ids))
        or len(process_hashes) != len(set(process_hashes))
        or len(launch_pids) != 1
    ):
        raise CBEndpointValidationError(
            "serve manifest does not identify one unique launch process set"
        )

    census = payload.get("listener_census")
    census_rows = census.get("listeners") if isinstance(census, Mapping) else None
    listener = payload.get("listener_binding")
    listener_rows = listener.get("listeners") if isinstance(listener, Mapping) else None
    valid_rows = True
    inodes: list[str] = []
    if isinstance(census_rows, list):
        for row in census_rows:
            owners = row.get("pids") if isinstance(row, Mapping) else None
            inode = row.get("socket_inode") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or set(row) != {"family", "address", "port", "socket_inode", "pids"}
                or row.get("family") not in {"ipv4", "ipv6"}
                or not isinstance(row.get("address"), str)
                or not row.get("address")
                or isinstance(row.get("port"), bool)
                or not isinstance(row.get("port"), int)
                or not 0 < row.get("port", 0) <= 65535
                or not isinstance(inode, str)
                or not inode.isdigit()
                or not isinstance(owners, list)
                or not owners
                or len(owners) != len(set(owners))
                or not set(owners).issubset(process_ids)
            ):
                valid_rows = False
                break
            inodes.append(inode)
    if (
        not isinstance(census, Mapping)
        or census.get("schema") != "prismaquant.server_tcp_listeners/1"
        or census.get("tables_readable") is not True
        or census.get("unreadable_pids") != []
        or not isinstance(census_rows, list)
        or not valid_rows
        or len(inodes) != len(set(inodes))
        or not isinstance(listener, Mapping)
        or set(listener) != {
            "schema", "base_url", "launch_host", "launch_port", "listeners"
        }
        or listener.get("schema") != "prismaquant.server_listener_binding/1"
        or listener.get("base_url") != "http://127.0.0.1:8000"
        or listener.get("launch_host") != "0.0.0.0"
        or listener.get("launch_port") != 8000
        or not isinstance(listener_rows, list)
        or not listener_rows
        or listener_rows
        != [
            row for row in census_rows
            if isinstance(row, Mapping) and row.get("port") == 8000
        ]
        or launch_pids[0]
        not in {
            pid
            for row in listener_rows
            if isinstance(row, Mapping)
            for pid in row.get("pids", [])
        }
    ):
        raise CBEndpointValidationError(
            "serve manifest listener is not owned by its exact launch process"
        )

    session_id = payload.get("serve_session_id")
    if (
        not isinstance(session_id, str)
        or _FINGERPRINT_RE.fullmatch(session_id) is None
        or session_id != serve_session_fingerprint(payload)
    ):
        raise CBEndpointValidationError(
            "serve manifest session identity is missing or stale"
        )
    try:
        endpoint_identity = models_endpoint_binding_identity(
            payload.get("models_endpoint_binding")
        )
    except Exception as exc:
        raise CBEndpointValidationError(
            f"serve manifest models endpoint binding is invalid: {exc}"
        ) from exc
    model_identity = endpoint_identity.get("model")
    gpu_uuid = payload.get("gpu_uuid")
    if (
        not isinstance(model_identity, Mapping)
        or model_identity.get("id") != expected_served_model
        or model_identity.get("root") != "/model"
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or len(gpu_uuid) <= len("GPU-")
    ):
        raise CBEndpointValidationError(
            "serve manifest models endpoint/GPU does not identify the launched /model session"
        )
    return {
        "schema": ENDPOINT_SESSION_SCHEMA,
        "serve_session_id": session_id,
        "host_identity": dict(host),
        "gpu_uuid": gpu_uuid,
        "processes": sorted(compact_processes, key=lambda row: row["pid"]),
        "launch_pid": launch_pids[0],
        "listener_binding": dict(listener),
        "models_endpoint_binding": endpoint_identity,
    }


def _canonical_launch_contract(
    argv: Sequence[object],
    *,
    arm: str,
    expected_served_model: str,
    requires_moe_marlin: bool,
) -> dict[str, Any]:
    """Parse the complete vLLM serve CLI and reject every undeclared flag."""
    tokens = [str(value) for value in argv]
    serve_positions = [
        index for index, token in enumerate(tokens) if token == "serve"
    ]
    if len(serve_positions) != 1:
        raise CBEndpointValidationError(
            "serve argv must contain exactly one vLLM 'serve' command"
        )
    serve_index = serve_positions[0]
    if serve_index < 1 or Path(tokens[serve_index - 1]).name != "vllm":
        raise CBEndpointValidationError(
            "serve argv must invoke the pinned vllm launcher"
        )
    tail = tokens[serve_index + 1:]
    if not tail or tail[0] != "/model":
        raise CBEndpointValidationError(
            "serve argv must use /model as its sole positional model"
        )

    options: dict[str, str] = {}
    switches: set[str] = set()
    index = 1
    while index < len(tail):
        token = tail[index]
        inline_value: str | None = None
        flag = token
        if token.startswith("--") and "=" in token:
            flag, inline_value = token.split("=", 1)
        if flag in _SERVE_SWITCH_FLAGS:
            if inline_value is not None or flag in switches:
                raise CBEndpointValidationError(
                    f"serve argv duplicates or assigns switch {flag}"
                )
            switches.add(flag)
            index += 1
            continue
        if flag not in _SERVE_VALUE_FLAGS:
            raise CBEndpointValidationError(
                f"serve argv contains an undeclared option or positional token {token!r}"
            )
        if flag in options:
            raise CBEndpointValidationError(
                f"serve argv declares {flag} more than once"
            )
        if inline_value is None:
            if index + 1 >= len(tail) or tail[index + 1].startswith("--"):
                raise CBEndpointValidationError(
                    f"serve argv has no value for {flag}"
                )
            inline_value = tail[index + 1]
            index += 2
        else:
            index += 1
        if inline_value == "":
            raise CBEndpointValidationError(
                f"serve argv has an empty value for {flag}"
            )
        options[flag] = inline_value

    expected_options = {
        "--served-model-name": expected_served_model,
        "--host": "0.0.0.0",
        "--port": "8000",
        "--tokenizer-mode": "deepseek_v4",
        "--generation-config": "vllm",
        "--quantization": "gridbook",
        "--tensor-parallel-size": "1",
        "--kv-cache-dtype": "fp8",
        "--kv-cache-memory-bytes": "1073741824",
        "--max-model-len": "8192",
        "--max-num-seqs": "1",
        "--max-num-batched-tokens": "512",
        "--gpu-memory-utilization": "0.90",
    }
    if arm == "graph":
        expected_options["--compilation-config"] = (
            DSV4_GRAPH_COMPILATION_CONFIG
        )
    if requires_moe_marlin:
        expected_options["--moe-backend"] = "marlin"
    if options != expected_options:
        raise CBEndpointValidationError(
            "serve argv value-option contract differs: "
            f"got {options!r}, expected {expected_options!r}"
        )

    expected_switches = {
        "--trust-remote-code",
        "--no-enable-prefix-caching",
    }
    if arm == "eager":
        expected_switches.add("--enforce-eager")
    if switches != expected_switches:
        raise CBEndpointValidationError(
            "serve argv switch contract differs: "
            f"got {sorted(switches)!r}, expected {sorted(expected_switches)!r}"
        )
    return {
        "model": "/model",
        "served_model_name": expected_served_model,
        "options": dict(sorted(options.items())),
        "switches": sorted(switches),
        "requires_moe_backend_marlin": bool(requires_moe_marlin),
    }


def validate_serve_manifest(
    payload: Mapping[str, Any],
    *,
    arm: str,
    expected_image: str = DSV4_SPARK_VLLM_IMAGE,
    expected_vllm_version: str = DSV4_SPARK_VLLM_VERSION,
    expected_served_model: str,
    requires_moe_marlin: bool,
    expected_model_sha: str | None = None,
) -> str:
    """Validate server-side identity and return its exact fingerprint."""
    if arm not in {"eager", "graph"}:
        raise CBEndpointValidationError(f"unknown validation arm {arm!r}")
    _require_session_model_name(expected_served_model)

    # Import the existing canonicalization only here so ordinary module import
    # remains stdlib + shipcard.  This validator is run from a repository mount
    # by the DSv4 driver, beside the tool that wrote the manifest.
    try:
        from tools.serve_fingerprint import (
            MANIFEST_SCHEMA,
            elide_argv_paths,
            fingerprint,
        )
    except Exception as exc:  # pragma: no cover - packaging misuse
        raise CBEndpointValidationError(
            "tools.serve_fingerprint is unavailable; run from the PrismaQuant tree"
        ) from exc

    if payload.get("schema") != MANIFEST_SCHEMA:
        raise CBEndpointValidationError(
            f"unsupported serve manifest schema {payload.get('schema')!r}"
        )
    recorded_fingerprint = payload.get("serve_fingerprint")
    if (
        not isinstance(recorded_fingerprint, str)
        or _FINGERPRINT_RE.fullmatch(recorded_fingerprint) is None
        or recorded_fingerprint != fingerprint(payload)
    ):
        raise CBEndpointValidationError("serve manifest fingerprint is missing or stale")
    if payload.get("source") != "server":
        raise CBEndpointValidationError(
            f"serve manifest must be server-side, got source={payload.get('source')!r}"
        )
    if payload.get("residency_readable") is not True:
        raise CBEndpointValidationError(
            "serve manifest could not read every server process address space"
        )
    if payload.get("image") != expected_image:
        raise CBEndpointValidationError(
            f"serve image {payload.get('image')!r} != exact DSv4 image {expected_image!r}"
        )
    if payload.get("model") != "/model":
        raise CBEndpointValidationError(
            f"serve manifest model must be the exact /model mount, got {payload.get('model')!r}"
        )
    binding = payload.get("artifact_binding")
    if expected_model_sha is not None and (
        not isinstance(binding, Mapping)
        or binding.get("schema") != "prismaquant.served_artifact_binding/1"
        or binding.get("model_sha") != expected_model_sha
        or not isinstance(binding.get("artifact_bytes"), int)
        or isinstance(binding.get("artifact_bytes"), bool)
        or binding.get("artifact_bytes", 0) <= 0
        or not isinstance(binding.get("artifact_inventory_sha256"), str)
        or _FINGERPRINT_RE.fullmatch(
            str(binding.get("artifact_inventory_sha256"))
        ) is None
    ):
        raise CBEndpointValidationError(
            "serve manifest is not bound to the exact mounted artifact"
        )
    if not expected_served_model or payload.get(
        "served_model_name"
    ) != expected_served_model:
        raise CBEndpointValidationError(
            "serve manifest served-model-name does not match the requested endpoint id"
        )
    if payload.get("gpu_name") != DSV4_SPARK_GPU_NAME or payload.get(
        "gpu_count"
    ) != 1:
        raise CBEndpointValidationError(
            f"serve must run on one {DSV4_SPARK_GPU_NAME}; got "
            f"gpu_name={payload.get('gpu_name')!r}, gpu_count={payload.get('gpu_count')!r}"
        )

    expected_eager = arm == "eager"
    if payload.get("enforce_eager") is not expected_eager:
        raise CBEndpointValidationError(
            f"{arm} arm has enforce_eager={payload.get('enforce_eager')!r}"
        )
    if payload.get("quantization") != "gridbook":
        raise CBEndpointValidationError("serve did not use --quantization gridbook")
    if payload.get("kv_cache_dtype") != "fp8":
        raise CBEndpointValidationError("serve did not use --kv-cache-dtype fp8")
    speculative_config = payload.get("speculative_config")
    if speculative_config is not None:
        raise CBEndpointValidationError(
            "speculative decoding is configured; CB eager/graph gates require it off"
        )

    argv = payload.get("launch_argv")
    if not isinstance(argv, list) or not argv:
        raise CBEndpointValidationError("serve manifest has no launch argv")
    _canonical_launch_contract(
        argv,
        arm=arm,
        expected_served_model=expected_served_model,
        requires_moe_marlin=requires_moe_marlin,
    )
    if payload.get("launch_flags") != elide_argv_paths(
        [str(value) for value in argv]
    ):
        raise CBEndpointValidationError(
            "serve manifest launch_flags are not the canonicalized launch argv"
        )

    extensions = payload.get("resident_extensions")
    if (
        not isinstance(extensions, list)
        or any(not isinstance(name, str) for name in extensions)
        or extensions != sorted(set(extensions))
        or not any(
        isinstance(name, str)
        and _GRIDBOOK_NATIVE_EXTENSION_RE.match(name) is not None
        and ".so" in name
        for name in extensions
        )
    ):
        raise CBEndpointValidationError(
            "serve manifest has no resident reviewed Gridbook-native CUDA extension"
        )
    runtime_pin = _gridbook_runtime_pin()
    manifest_pin = _manifest_gridbook_runtime_pin(payload, runtime_pin)
    if manifest_pin is None:
        raise CBEndpointValidationError(
            "served Gridbook runtime pin differs from the tracked release pin"
        )
    distribution_problems = _verify_gridbook_distribution_identity(
        "native_export",
        payload,
        runtime_pin,
        canonical_sha=_canonical_json_sha256,
    )
    if distribution_problems:
        raise CBEndpointValidationError(
            "serve manifest does not attest the exact imported Gridbook "
            f"distribution: {distribution_problems[0]}"
        )
    _validate_closed_server_environment(payload, runtime_pin=runtime_pin)
    _validate_live_server_session(
        payload,
        expected_served_model=expected_served_model,
    )
    packages = payload.get("package_versions")
    if not isinstance(packages, Mapping):
        raise CBEndpointValidationError("serve manifest has no package versions")
    if packages.get("gridbook") != runtime_pin["version"]:
        raise CBEndpointValidationError(
            f"installed Gridbook {packages.get('gridbook')!r} != {runtime_pin['version']!r}"
        )
    if packages.get("vllm") != expected_vllm_version:
        raise CBEndpointValidationError(
            f"installed vLLM {packages.get('vllm')!r} != {expected_vllm_version!r}"
        )
    return recorded_fingerprint


def validate_cb_artifact(model_dir: str | Path) -> dict[str, Any]:
    """Fail unless ``model_dir`` is a structurally complete Gridbook CB export."""
    root = Path(model_dir)
    config_path = root / "config.json"
    quant_path = root / "quant_config.json"
    if not config_path.is_file() or not quant_path.is_file():
        raise CBEndpointValidationError(
            "CB artifact must contain config.json and quant_config.json"
        )
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CBEndpointValidationError(f"CB artifact JSON is unreadable: {exc}") from exc
    if not isinstance(model_config, dict) or model_config.get(
        "model_type"
    ) != "deepseek_v4":
        raise CBEndpointValidationError(
            f"artifact is not DSv4: model_type={getattr(model_config, 'get', lambda *_: None)('model_type')!r}"
        )
    if not isinstance(quant_config, dict) or quant_config.get(
        "quant_method"
    ) != "gridbook" or quant_config.get("format") != "nvfp4_cb":
        raise CBEndpointValidationError(
            "artifact is not an nvfp4_cb Gridbook export"
        )
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "weight_content_manifest"
    ) is None:
        raise CBEndpointValidationError(
            "CB artifact has no exact weight_content_manifest"
        )
    acknowledgements = provenance.get(
        "route_pending_passthrough_acknowledged", []
    )
    if not isinstance(acknowledgements, list) or acknowledgements != sorted(
        set(str(value) for value in acknowledgements)
    ):
        raise CBEndpointValidationError(
            "DSv4 artifact route-pending acknowledgements are not canonical"
        )
    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise CBEndpointValidationError("CB artifact has no config_groups")
    if not any(
        isinstance(group, Mapping)
        and group.get("format") == "source-passthrough"
        and group.get("source_format") == "FP8_BLOCK_UE8M0_SOURCE"
        for group in groups.values()
    ):
        raise CBEndpointValidationError(
            "DSv4 artifact has no declared FP8_BLOCK_UE8M0_SOURCE overlay"
        )
    source_formats = {
        str(group.get("source_format"))
        for group in groups.values()
        if isinstance(group, Mapping)
        and group.get("format") == "source-passthrough"
        and isinstance(group.get("source_format"), str)
    }
    required_acknowledgements = sorted(
        source_formats & ROUTE_PENDING_PASSTHROUGH_FORMATS
    )
    missing_acknowledgements = sorted(
        set(required_acknowledgements) - set(acknowledgements)
    )
    if missing_acknowledgements:
        raise CBEndpointValidationError(
            "DSv4 artifact did not record required route-pending "
            f"acknowledgements: {missing_acknowledgements}"
        )
    codebook_file = quant_config.get("codebook_file")
    if not isinstance(codebook_file, str) or not codebook_file.endswith(".pqcb"):
        raise CBEndpointValidationError("CB artifact has no declared .pqcb sidecar")
    sidecar = root / codebook_file
    if sidecar.parent != root or not sidecar.is_file():
        raise CBEndpointValidationError(
            f"declared CB sidecar is missing or escapes the artifact: {codebook_file!r}"
        )
    return quant_config


def validate_cb_artifact_decode_contract(
    model_dir: str | Path,
    quant_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove header completeness and the released three-stage DSpark overlay."""
    root = Path(model_dir)
    if quant_config is None:
        quant_config = validate_cb_artifact(root)
    try:
        from .artifact_completeness import assert_artifact_complete
        from .dspark_source_metadata import (
            discover_dspark_source_overlay_from_artifact,
        )

        completeness = assert_artifact_complete(root)
        overlay = discover_dspark_source_overlay_from_artifact(root)
    except Exception as exc:
        raise CBEndpointValidationError(
            f"artifact decode coverage is incomplete: {type(exc).__name__}: {exc}"
        ) from exc
    if overlay is None:
        raise CBEndpointValidationError(
            "artifact is not the released DSv4Flash three-stage DSpark topology"
        )

    overlay_provenance = overlay.provenance()
    recorded_overlay = (
        (quant_config.get("provenance") or {}).get("dspark_source_overlay")
        if isinstance(quant_config.get("provenance"), Mapping)
        else None
    )
    if recorded_overlay != overlay_provenance:
        raise CBEndpointValidationError(
            "artifact DSpark overlay provenance does not match its tensor headers"
        )

    completeness_payload = asdict(completeness)
    classified = (
        len(completeness.cb_units)
        + len(completeness.passthrough_units)
        + len(completeness.verbatim_namespace_units)
    )
    overlay_payload = {
        "provenance": overlay_provenance,
        "physical_targets": dict(overlay.physical_targets),
        "construction_units": dict(overlay.construction_units),
        "physical_to_construction_unit": dict(
            overlay.physical_to_construction_unit
        ),
    }
    evidence: dict[str, Any] = {
        "schema": ARTIFACT_DECODE_CONTRACT_SCHEMA,
        "complete": completeness.ok,
        "completeness_sha256": _canonical_json_sha256(completeness_payload),
        "declared_unit_count": len(completeness.declared_units),
        "cb_unit_count": len(completeness.cb_units),
        "passthrough_unit_count": len(completeness.passthrough_units),
        "verbatim_namespace_unit_count": len(
            completeness.verbatim_namespace_units
        ),
        "classified_unit_count": classified,
        "route_pending_acknowledged": sorted(
            str(value)
            for value in completeness.route_pending_acknowledged
        ),
        "excluded_namespaces": sorted(
            str(value) for value in completeness.excluded_namespaces
        ),
        "dspark_overlay": overlay_provenance,
        "dspark_overlay_sha256": _canonical_json_sha256(overlay_payload),
        "requires_moe_backend_marlin": artifact_requires_moe_backend_marlin(
            quant_config
        ),
    }
    evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
    _validate_artifact_decode_record(evidence)
    return evidence


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise CBEndpointValidationError(
            f"{where} keys differ: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
        raise CBEndpointValidationError(f"{where} is not a lowercase SHA-256")
    return value


def _validate_artifact_decode_record(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "complete",
        "completeness_sha256",
        "declared_unit_count",
        "cb_unit_count",
        "passthrough_unit_count",
        "verbatim_namespace_unit_count",
        "classified_unit_count",
        "route_pending_acknowledged",
        "excluded_namespaces",
        "dspark_overlay",
        "dspark_overlay_sha256",
        "requires_moe_backend_marlin",
        "evidence_sha256",
    }
    _require_exact_keys(payload, expected_keys, where="artifact decode contract")
    if payload.get("schema") != ARTIFACT_DECODE_CONTRACT_SCHEMA or payload.get(
        "complete"
    ) is not True:
        raise CBEndpointValidationError(
            "artifact decode contract does not attest complete coverage"
        )
    for key in (
        "declared_unit_count",
        "cb_unit_count",
        "passthrough_unit_count",
        "verbatim_namespace_unit_count",
        "classified_unit_count",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CBEndpointValidationError(
                f"artifact decode contract {key} is not a non-negative integer"
            )
    if payload.get("declared_unit_count", 0) <= 0 or payload.get(
        "classified_unit_count", 0
    ) <= 0:
        raise CBEndpointValidationError(
            "artifact decode contract classified no declared source weights"
        )
    if payload.get("classified_unit_count") != sum(
        int(payload[key]) for key in (
            "cb_unit_count",
            "passthrough_unit_count",
            "verbatim_namespace_unit_count",
        )
    ):
        raise CBEndpointValidationError(
            "artifact decode contract classified-unit accounting differs"
        )
    acknowledged = payload.get("route_pending_acknowledged")
    if not isinstance(acknowledged, list) or acknowledged != sorted(
        set(str(value) for value in acknowledged)
    ):
        raise CBEndpointValidationError(
            "artifact decode contract route-pending acknowledgements are not "
            "canonical"
        )
    excluded = payload.get("excluded_namespaces")
    if not isinstance(excluded, list) or excluded != sorted(
        set(str(value) for value in excluded)
    ):
        raise CBEndpointValidationError(
            "artifact decode contract excluded namespaces are not canonical"
        )
    if not isinstance(payload.get("requires_moe_backend_marlin"), bool):
        raise CBEndpointValidationError(
            "artifact decode contract has no boolean marlin-route decision"
        )
    _require_sha256(
        payload.get("completeness_sha256"), where="completeness digest"
    )
    _require_sha256(
        payload.get("dspark_overlay_sha256"), where="DSpark overlay digest"
    )

    overlay = payload.get("dspark_overlay")
    if not isinstance(overlay, Mapping):
        raise CBEndpointValidationError(
            "artifact decode contract has no DSpark overlay provenance"
        )
    from .dspark_source_metadata import (
        DSPARK_OVERLAY_SCHEMA,
        DSPARK_STAGE_COUNT,
    )

    overlay_keys = {
        "schema",
        "physical_namespace",
        "construction_namespace",
        "num_hidden_layers",
        "n_mtp_layers",
        "physical_stage_ids",
        "construction_layer_ids",
        "physical_target_counts",
        "construction_unit_count",
        "tensor_bytes_rewritten",
    }
    _require_exact_keys(overlay, overlay_keys, where="DSpark overlay provenance")
    num_layers = overlay.get("num_hidden_layers")
    if (
        overlay.get("schema") != DSPARK_OVERLAY_SCHEMA
        or overlay.get("physical_namespace") != "mtp.{stage}"
        or overlay.get("construction_namespace")
        != "model.layers.{num_hidden_layers+stage}"
        or not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or num_layers <= 0
        or overlay.get("n_mtp_layers") != DSPARK_STAGE_COUNT
        or overlay.get("physical_stage_ids") != list(range(DSPARK_STAGE_COUNT))
        or overlay.get("construction_layer_ids")
        != [num_layers + stage for stage in range(DSPARK_STAGE_COUNT)]
        or overlay.get("construction_unit_count")
        != DSPARK_STAGE_COUNT * 7 + 1
        or overlay.get("tensor_bytes_rewritten") != 0
    ):
        raise CBEndpointValidationError(
            "artifact decode contract carries an invalid DSpark topology"
        )
    target_counts = overlay.get("physical_target_counts")
    if not isinstance(target_counts, Mapping) or not target_counts or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in target_counts.values()
    ):
        raise CBEndpointValidationError(
            "artifact decode contract has invalid DSpark physical target counts"
        )

    recorded_evidence_sha = _require_sha256(
        payload.get("evidence_sha256"), where="artifact evidence digest"
    )
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256")
    if recorded_evidence_sha != _canonical_json_sha256(unstamped):
        raise CBEndpointValidationError(
            "artifact decode contract evidence digest is stale"
        )


def validate_graph_capture_log(path: str | Path) -> dict[str, Any]:
    """Bind the graph arm to positive vLLM capture evidence after generation."""
    log_path = Path(path)
    try:
        raw = log_path.read_bytes()
    except OSError as exc:
        raise CBEndpointValidationError(
            f"graph serve log cannot be read: {log_path}"
        ) from exc
    text = raw.decode("utf-8", "replace")
    if "Skipping CUDA graph capture" in text:
        raise CBEndpointValidationError("vLLM explicitly skipped CUDA graph capture")
    if _FULL_DECODE_CONFIG_RE.search(text) is None:
        raise CBEndpointValidationError(
            "serve log does not attest resolved FULL_DECODE_ONLY capture size [1]"
        )
    downgraded = (
        "Overriding cudagraph_mode to PIECEWISE",
        "Overriding cudagraph_mode from FULL_DECODE_ONLY to PIECEWISE",
        "setting cudagraph_mode=PIECEWISE",
        "setting cudagraph_mode=NONE",
    )
    if any(marker in text for marker in downgraded):
        raise CBEndpointValidationError(
            "vLLM log reports a CUDA-graph mode downgrade or disable"
        )
    match = _GRAPH_CAPTURE_RE.search(text)
    if match is None:
        raise CBEndpointValidationError(
            "serve log has no positive 'Graph capturing finished' evidence"
        )
    return {
        "capture_marker": match.group(0),
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
        "serve_log": str(log_path.resolve()),
    }


def _endpoint(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CBEndpointValidationError(f"invalid OpenAI base URL {base_url!r}")
    if parsed.path.rstrip("/")[-3:] != "/v1":
        raise CBEndpointValidationError(
            f"OpenAI base URL must end in /v1, got {base_url!r}"
        )
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _request_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise CBEndpointValidationError(
                    f"{method} {url} returned HTTP {response.status}"
                )
            decoded = json.loads(response.read().decode("utf-8"))
    except CBEndpointValidationError:
        raise
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CBEndpointValidationError(
            f"{method} {url} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise CBEndpointValidationError(f"{method} {url} did not return a JSON object")
    return decoded


Requester = Callable[[str, str, Mapping[str, Any] | None, float], dict[str, Any]]


def _resolve_served_model(
    base_url: str,
    requested: str | None,
    *,
    timeout: float,
    requester: Requester,
) -> str:
    response = requester("GET", _endpoint(base_url, "models"), None, timeout)
    rows = response.get("data")
    if not isinstance(rows, list):
        raise CBEndpointValidationError("/v1/models response has no data list")
    model_ids = [
        row.get("id") for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]
    if requested is not None:
        if requested not in model_ids:
            raise CBEndpointValidationError(
                f"requested model {requested!r} is not served; endpoint has {model_ids!r}"
            )
        return requested
    if len(model_ids) != 1:
        raise CBEndpointValidationError(
            f"endpoint must expose exactly one model when --model-name is omitted; got {model_ids!r}"
        )
    return model_ids[0]


def _observe_models_endpoint_binding(
    base_url: str,
    requested: str,
    *,
    timeout: float,
    requester: Requester,
) -> dict[str, Any]:
    """Obtain the stable `/models` projection through the smoke requester."""
    response = requester("GET", _endpoint(base_url, "models"), None, timeout)
    try:
        from tools.serve_fingerprint import models_endpoint_binding_from_bytes

        raw = json.dumps(
            response,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        binding = models_endpoint_binding_from_bytes(
            raw,
            request_url=_endpoint(base_url, "models"),
            expected_served_model=requested,
        )
        from tools.serve_fingerprint import models_endpoint_binding_identity

        return models_endpoint_binding_identity(binding)
    except Exception as exc:
        raise CBEndpointValidationError(
            f"live models endpoint identity is invalid: {exc}"
        ) from exc


def _completion_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CBEndpointValidationError("completion response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("text"), str):
        raise CBEndpointValidationError("completion choice has no text")
    text = str(choice["text"])
    if not text.strip():
        raise CBEndpointValidationError("completion was empty or whitespace-only")
    return text


def run_endpoint_smoke(
    *,
    base_url: str,
    model_name: str | None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 8,
    timeout: float = 300.0,
    requester: Requester = _request_json,
    expected_models_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run two identical greedy requests and return non-quality metrics."""
    if max_tokens <= 0:
        raise CBEndpointValidationError("max_tokens must be positive")
    if model_name is None:
        served_model = _resolve_served_model(
            base_url, model_name, timeout=timeout, requester=requester
        )
        live_models_identity = None
    else:
        served_model = model_name
        live_models_identity = _observe_models_endpoint_binding(
            base_url,
            served_model,
            timeout=timeout,
            requester=requester,
        )
        if (
            expected_models_identity is not None
            and dict(live_models_identity) != dict(expected_models_identity)
        ):
            raise CBEndpointValidationError(
                "validation endpoint is not the attested live server session"
            )
    request_payload = {
        "model": served_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
    }
    texts = []
    for _ in range(2):
        response = requester(
            "POST",
            _endpoint(base_url, "completions"),
            request_payload,
            timeout,
        )
        texts.append(_completion_text(response))
    if texts[0] != texts[1]:
        raise CBEndpointValidationError(
            "two identical greedy completion requests returned different text"
        )
    encoded = texts[0].encode("utf-8")
    return {
        "served_model": served_model,
        "generated_chars": len(texts[0]),
        "generated_utf8_bytes": len(encoded),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "deterministic_repeats": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
        "max_tokens": max_tokens,
        "models_endpoint_identity": live_models_identity,
    }


def build_endpoint_contract(
    *,
    arm: str,
    model_sha: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    artifact_decode: Mapping[str, Any],
    endpoint_smoke: Mapping[str, Any],
    cuda_graph: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the path-independent receipt the shipcard verifier can replay."""
    _validate_artifact_decode_record(artifact_decode)
    served_model = _require_session_model_name(manifest.get("served_model_name"))
    launch = _canonical_launch_contract(
        manifest.get("launch_argv") or (),
        arm=arm,
        expected_served_model=served_model,
        requires_moe_marlin=bool(
            artifact_decode.get("requires_moe_backend_marlin")
        ),
    )
    pin = _gridbook_runtime_pin()
    environment = manifest.get("pq_env")
    extensions = manifest.get("resident_extensions")
    if (
        not isinstance(environment, Mapping)
        or dict(environment) != _expected_serve_environment(pin)
        or not isinstance(extensions, list)
    ):
        raise CBEndpointValidationError(
            "serve manifest lacks contract environment or extension residency"
        )
    session = _validate_live_server_session(
        manifest,
        expected_served_model=served_model,
    )
    graph_record = None
    if cuda_graph is not None:
        graph_record = {
            "capture_marker": cuda_graph.get("capture_marker"),
            "serve_log_sha256": cuda_graph.get("serve_log_sha256"),
        }
    contract: dict[str, Any] = {
        "schema": ENDPOINT_CONTRACT_SCHEMA,
        "arm": arm,
        "model_sha": model_sha,
        "artifact_decode": dict(artifact_decode),
        "serve_manifest": {
            "sha256": manifest_sha256,
            "serve_fingerprint": manifest.get("serve_fingerprint"),
            "artifact_binding": manifest.get("artifact_binding"),
        },
        "stack": {
            "image": DSV4_SPARK_VLLM_IMAGE,
            "gpu_name": DSV4_SPARK_GPU_NAME,
            "gpu_count": 1,
            "vllm_version": DSV4_SPARK_VLLM_VERSION,
            "vllm_commit": DSV4_SPARK_VLLM_COMMIT,
            "gridbook_runtime_version": pin["version"],
            "gridbook_runtime_commit": pin["commit"],
        },
        "launch": launch,
        "environment": {
            str(key): str(value)
            for key, value in sorted(environment.items())
        },
        "resident_extensions": sorted(str(value) for value in extensions),
        "endpoint_session": session,
        "endpoint_smoke": dict(endpoint_smoke),
        "cuda_graph": graph_record,
    }
    contract["contract_sha256"] = _canonical_json_sha256(contract)
    validate_endpoint_contract_record(
        contract,
        arm=arm,
        model_sha=model_sha,
        serve_fingerprint=str(manifest.get("serve_fingerprint")),
    )
    return contract


def validate_endpoint_contract_record(
    payload: Mapping[str, Any],
    *,
    arm: str,
    model_sha: str | None = None,
    serve_fingerprint: str | None = None,
) -> None:
    """Replay every structural assertion encoded in a native CB receipt."""
    if arm not in {"eager", "graph"}:
        raise CBEndpointValidationError(f"invalid endpoint-contract arm {arm!r}")
    expected_keys = {
        "schema",
        "arm",
        "model_sha",
        "artifact_decode",
        "serve_manifest",
        "stack",
        "launch",
        "environment",
        "resident_extensions",
        "endpoint_session",
        "endpoint_smoke",
        "cuda_graph",
        "contract_sha256",
    }
    _require_exact_keys(payload, expected_keys, where="endpoint contract")
    if payload.get("schema") != ENDPOINT_CONTRACT_SCHEMA or payload.get(
        "arm"
    ) != arm:
        raise CBEndpointValidationError(
            "endpoint contract schema or arm differs from its native slot"
        )
    contract_model_sha = _require_sha256(
        payload.get("model_sha"), where="endpoint contract model_sha"
    )
    if model_sha is not None and contract_model_sha != model_sha:
        raise CBEndpointValidationError(
            "endpoint contract model_sha differs from its native record"
        )

    artifact_decode = payload.get("artifact_decode")
    if not isinstance(artifact_decode, Mapping):
        raise CBEndpointValidationError(
            "endpoint contract has no artifact decode proof"
        )
    _validate_artifact_decode_record(artifact_decode)

    serve_manifest = payload.get("serve_manifest")
    if not isinstance(serve_manifest, Mapping):
        raise CBEndpointValidationError(
            "endpoint contract has no serve-manifest binding"
        )
    _require_exact_keys(
        serve_manifest,
        {"sha256", "serve_fingerprint", "artifact_binding"},
        where="serve-manifest binding",
    )
    _require_sha256(
        serve_manifest.get("sha256"), where="serve-manifest content digest"
    )
    manifest_fingerprint = _require_sha256(
        serve_manifest.get("serve_fingerprint"),
        where="serve-manifest fingerprint",
    )
    if serve_fingerprint is not None and manifest_fingerprint != serve_fingerprint:
        raise CBEndpointValidationError(
            "endpoint contract fingerprint differs from its native record"
        )
    artifact_binding = serve_manifest.get("artifact_binding")
    if (
        not isinstance(artifact_binding, Mapping)
        or artifact_binding.get("schema")
        != "prismaquant.served_artifact_binding/1"
        or artifact_binding.get("model_sha") != contract_model_sha
        or not isinstance(artifact_binding.get("artifact_bytes"), int)
        or isinstance(artifact_binding.get("artifact_bytes"), bool)
        or artifact_binding.get("artifact_bytes", 0) <= 0
        or _FINGERPRINT_RE.fullmatch(
            str(artifact_binding.get("artifact_inventory_sha256", ""))
        ) is None
    ):
        raise CBEndpointValidationError(
            "endpoint contract has no exact served-artifact binding"
        )

    stack = payload.get("stack")
    if not isinstance(stack, Mapping):
        raise CBEndpointValidationError("endpoint contract has no serving stack")
    pin = _gridbook_runtime_pin()
    expected_stack = {
        "image": DSV4_SPARK_VLLM_IMAGE,
        "gpu_name": DSV4_SPARK_GPU_NAME,
        "gpu_count": 1,
        "vllm_version": DSV4_SPARK_VLLM_VERSION,
        "vllm_commit": DSV4_SPARK_VLLM_COMMIT,
        "gridbook_runtime_version": pin["version"],
        "gridbook_runtime_commit": pin["commit"],
    }
    if dict(stack) != expected_stack:
        raise CBEndpointValidationError(
            f"endpoint contract serving stack differs: {dict(stack)!r}"
        )

    launch = payload.get("launch")
    if not isinstance(launch, Mapping):
        raise CBEndpointValidationError("endpoint contract has no launch contract")
    _require_exact_keys(
        launch,
        {
            "model",
            "served_model_name",
            "options",
            "switches",
            "requires_moe_backend_marlin",
        },
        where="launch contract",
    )
    served_model = launch.get("served_model_name")
    options = launch.get("options")
    switches = launch.get("switches")
    requires_marlin = artifact_decode.get("requires_moe_backend_marlin")
    if (
        not isinstance(served_model, str)
        or _SESSION_MODEL_RE.fullmatch(served_model) is None
        or launch.get("model") != "/model"
        or not isinstance(options, Mapping)
        or not isinstance(switches, list)
        or launch.get("requires_moe_backend_marlin") is not requires_marlin
    ):
        raise CBEndpointValidationError("endpoint launch contract is malformed")
    replay_argv: list[str] = ["/usr/local/bin/vllm", "serve", "/model"]
    for flag, value in options.items():
        replay_argv.extend((str(flag), str(value)))
    replay_argv.extend(str(value) for value in switches)
    replayed = _canonical_launch_contract(
        replay_argv,
        arm=arm,
        expected_served_model=served_model,
        requires_moe_marlin=bool(requires_marlin),
    )
    if dict(launch) != replayed:
        raise CBEndpointValidationError(
            "endpoint launch contract is not in canonical form"
        )

    environment = payload.get("environment")
    expected_environment = _expected_serve_environment(pin)
    if not isinstance(environment, Mapping) or dict(
        environment
    ) != expected_environment:
        observed = dict(environment) if isinstance(environment, Mapping) else {}
        differing = sorted(
            name for name in set(observed) | set(expected_environment)
            if observed.get(name) != expected_environment.get(name)
            or (name in observed) != (name in expected_environment)
        )
        raise CBEndpointValidationError(
            "endpoint contract environment differs from the complete "
            f"Gridbook endpoint profile: {differing}"
        )

    extensions = payload.get("resident_extensions")
    if (
        not isinstance(extensions, list)
        or any(not isinstance(value, str) for value in extensions)
        or extensions != sorted(set(extensions))
    ):
        raise CBEndpointValidationError(
            "endpoint contract resident extensions are not canonical"
        )
    if not any(
        isinstance(name, str)
        and _GRIDBOOK_NATIVE_EXTENSION_RE.match(name) is not None
        and ".so" in name
        for name in extensions
    ):
        raise CBEndpointValidationError(
            "endpoint contract has no resident Gridbook-native CUDA extension"
        )

    endpoint_session = payload.get("endpoint_session")
    session_keys = {
        "schema", "serve_session_id", "host_identity", "gpu_uuid",
        "processes", "launch_pid", "listener_binding",
        "models_endpoint_binding",
    }
    if (
        not isinstance(endpoint_session, Mapping)
        or set(endpoint_session) != session_keys
        or endpoint_session.get("schema") != ENDPOINT_SESSION_SCHEMA
        or _FINGERPRINT_RE.fullmatch(
            str(endpoint_session.get("serve_session_id", ""))
        ) is None
        or not isinstance(endpoint_session.get("processes"), list)
        or not endpoint_session.get("processes")
        or endpoint_session.get("launch_pid") not in {
            row.get("pid")
            for row in endpoint_session.get("processes", [])
            if isinstance(row, Mapping)
        }
    ):
        raise CBEndpointValidationError(
            "endpoint contract has no closed live-server session binding"
        )
    try:
        from tools.serve_fingerprint import (
            models_endpoint_binding_identity,
            serve_session_fingerprint,
        )

        models_identity = models_endpoint_binding_identity(
            endpoint_session["models_endpoint_binding"]
        )
    except Exception as exc:
        raise CBEndpointValidationError(
            f"endpoint contract models identity is invalid: {exc}"
        ) from exc
    model_identity = models_identity.get("model")
    if not isinstance(model_identity, Mapping) or (
        model_identity.get("id") != served_model
        or model_identity.get("root") != "/model"
    ):
        raise CBEndpointValidationError(
            "endpoint contract session does not bind the launched served model"
        )
    session_processes = endpoint_session.get("processes")
    host_identity = endpoint_session.get("host_identity")
    machine_sha = host_identity.get("machine_id_sha256") if isinstance(
        host_identity, Mapping
    ) else None
    gpu_uuid = endpoint_session.get("gpu_uuid")
    if (
        not isinstance(host_identity, Mapping)
        or set(host_identity)
        != {"hostname", "boot_id", "machine_id_sha256", "pid_namespace"}
        or not isinstance(host_identity.get("hostname"), str)
        or not host_identity.get("hostname")
        or not isinstance(host_identity.get("boot_id"), str)
        or not host_identity.get("boot_id")
        or (
            machine_sha is not None
            and _FINGERPRINT_RE.fullmatch(str(machine_sha)) is None
        )
        or not isinstance(host_identity.get("pid_namespace"), str)
        or not host_identity.get("pid_namespace")
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or len(gpu_uuid) <= len("GPU-")
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"pid", "identity_sha256"}
            or isinstance(row.get("pid"), bool)
            or not isinstance(row.get("pid"), int)
            or row.get("pid", 0) <= 0
            or _FINGERPRINT_RE.fullmatch(
                str(row.get("identity_sha256", ""))
            ) is None
            for row in session_processes
        )
        or session_processes
        != sorted(session_processes, key=lambda row: row["pid"])
        or len({row["pid"] for row in session_processes}) != len(session_processes)
        or len({row["identity_sha256"] for row in session_processes})
        != len(session_processes)
    ):
        raise CBEndpointValidationError(
            "endpoint contract session process census is malformed"
        )
    listener = endpoint_session.get("listener_binding")
    listener_rows = listener.get("listeners") if isinstance(
        listener, Mapping
    ) else None
    process_pids = {row["pid"] for row in session_processes}
    valid_listener_rows = isinstance(listener_rows, list) and bool(listener_rows)
    listener_inodes: list[str] = []
    if valid_listener_rows:
        for row in listener_rows:
            owners = row.get("pids") if isinstance(row, Mapping) else None
            inode = row.get("socket_inode") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"family", "address", "port", "socket_inode", "pids"}
                or row.get("family") not in {"ipv4", "ipv6"}
                or not isinstance(row.get("address"), str)
                or not row.get("address")
                or row.get("port") != 8000
                or not isinstance(inode, str)
                or not inode.isdigit()
                or not isinstance(owners, list)
                or not owners
                or len(owners) != len(set(owners))
                or not set(owners).issubset(process_pids)
            ):
                valid_listener_rows = False
                break
            listener_inodes.append(inode)
    if (
        not isinstance(listener, Mapping)
        or set(listener)
        != {"schema", "base_url", "launch_host", "launch_port", "listeners"}
        or listener.get("schema") != "prismaquant.server_listener_binding/1"
        or listener.get("base_url") != "http://127.0.0.1:8000"
        or listener.get("launch_host") != "0.0.0.0"
        or listener.get("launch_port") != 8000
        or not valid_listener_rows
        or len(listener_inodes) != len(set(listener_inodes))
        or endpoint_session.get("launch_pid")
        not in {
            pid
            for row in listener_rows or []
            if isinstance(row, Mapping)
            for pid in row.get("pids", [])
        }
        or endpoint_session.get("serve_session_id")
        != serve_session_fingerprint(endpoint_session)
        or endpoint_session.get("models_endpoint_binding") != models_identity
    ):
        raise CBEndpointValidationError(
            "endpoint contract session listener is malformed or unowned"
        )

    smoke = payload.get("endpoint_smoke")
    if not isinstance(smoke, Mapping):
        raise CBEndpointValidationError("endpoint contract has no smoke proof")
    smoke_keys = {
        "served_model",
        "generated_chars",
        "generated_utf8_bytes",
        "output_sha256",
        "prompt_sha256",
        "deterministic_repeats",
        "temperature",
        "top_p",
        "seed",
        "n",
        "stream",
        "max_tokens",
        "models_endpoint_identity",
    }
    _require_exact_keys(smoke, smoke_keys, where="endpoint smoke proof")
    if (
        smoke.get("served_model") != served_model
        or not isinstance(smoke.get("generated_chars"), int)
        or isinstance(smoke.get("generated_chars"), bool)
        or smoke.get("generated_chars", 0) <= 0
        or not isinstance(smoke.get("generated_utf8_bytes"), int)
        or isinstance(smoke.get("generated_utf8_bytes"), bool)
        or smoke.get("generated_utf8_bytes", 0) <= 0
        or smoke.get("deterministic_repeats") != 2
        or smoke.get("temperature") != 0.0
        or smoke.get("top_p") != 1.0
        or smoke.get("seed") != 0
        or smoke.get("n") != 1
        or smoke.get("stream") is not False
        or smoke.get("max_tokens") != DEFAULT_MAX_TOKENS
        or smoke.get("prompt_sha256")
        != hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest()
        or smoke.get("models_endpoint_identity")
        != endpoint_session.get("models_endpoint_binding")
    ):
        raise CBEndpointValidationError(
            "endpoint contract smoke request/result differs from the exact gate"
        )
    _require_sha256(smoke.get("output_sha256"), where="smoke output digest")

    graph = payload.get("cuda_graph")
    if arm == "graph":
        if not isinstance(graph, Mapping):
            raise CBEndpointValidationError(
                "graph endpoint contract has no capture evidence"
            )
        _require_exact_keys(
            graph,
            {"capture_marker", "serve_log_sha256"},
            where="CUDA-graph evidence",
        )
        if not isinstance(graph.get("capture_marker"), str) or not str(
            graph.get("capture_marker")
        ).startswith("Graph capturing finished"):
            raise CBEndpointValidationError(
                "graph endpoint contract has no positive capture marker"
            )
        _require_sha256(
            graph.get("serve_log_sha256"), where="graph serve-log digest"
        )
    elif graph is not None:
        raise CBEndpointValidationError(
            "eager endpoint contract unexpectedly carries graph evidence"
        )

    recorded_contract_sha = _require_sha256(
        payload.get("contract_sha256"), where="endpoint contract digest"
    )
    unstamped = dict(payload)
    unstamped.pop("contract_sha256")
    if recorded_contract_sha != _canonical_json_sha256(unstamped):
        raise CBEndpointValidationError("endpoint contract digest is stale")


def _write_result(path: str | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def commit_deferred_result(
    result_path: str | Path,
    shipcard_path: str | Path,
    model_dir: str | Path,
) -> dict[str, Any]:
    """Close one native slot only after the shell's final safety checks pass."""
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get(
        "schema"
    ) != ENDPOINT_RESULT_SCHEMA or payload.get("passed") is not True:
        raise CBEndpointValidationError("deferred endpoint result is not a PASS")
    slot = payload.get("slot")
    if slot not in {"native_export.eager", "native_export.graph"}:
        raise CBEndpointValidationError(f"invalid deferred endpoint slot {slot!r}")
    record = payload.get("record")
    if not isinstance(record, dict) or record.get("slot") != slot or record.get(
        "tool"
    ) != "validate_cb_endpoint.py" or record.get("passed") is not True:
        raise CBEndpointValidationError("deferred endpoint record is malformed")
    if (
        payload.get("metrics") != record.get("metrics")
        or payload.get("serve_fingerprint") != record.get("serve_fingerprint")
        or record.get("spec_decode_detected") is not False
    ):
        raise CBEndpointValidationError(
            "deferred endpoint result and native record differ"
        )
    arm = str(slot).rsplit(".", 1)[-1]
    metrics = record.get("metrics")
    contract = (
        metrics.get("endpoint_contract")
        if isinstance(metrics, Mapping)
        else None
    )
    if not isinstance(contract, Mapping):
        raise CBEndpointValidationError(
            "deferred endpoint record has no canonical endpoint contract"
        )
    validate_endpoint_contract_record(
        contract,
        arm=arm,
        model_sha=record.get("model_sha"),
        serve_fingerprint=record.get("serve_fingerprint"),
    )

    manifest_path = record.get("serve_manifest")
    if not isinstance(manifest_path, str):
        raise CBEndpointValidationError(
            "deferred endpoint record has no serve-manifest evidence path"
        )
    try:
        manifest_digest = hashlib.sha256(
            Path(manifest_path).read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise CBEndpointValidationError(
            f"deferred serve manifest is unreadable: {exc}"
        ) from exc
    if manifest_digest != contract["serve_manifest"]["sha256"]:
        raise CBEndpointValidationError(
            "deferred serve-manifest content differs from its receipt digest"
        )
    if arm == "graph":
        graph = metrics.get("cuda_graph")
        graph_log = graph.get("serve_log") if isinstance(graph, Mapping) else None
        if not isinstance(graph_log, str):
            raise CBEndpointValidationError(
                "deferred graph record has no serve-log evidence path"
            )
        try:
            graph_digest = hashlib.sha256(
                Path(graph_log).read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise CBEndpointValidationError(
                f"deferred graph serve log is unreadable: {exc}"
            ) from exc
        if graph_digest != contract["cuda_graph"]["serve_log_sha256"]:
            raise CBEndpointValidationError(
                "deferred graph serve log differs from its receipt digest"
            )
    model_sha = compute_model_sha(model_dir)
    card = load_shipcard(shipcard_path)
    if payload.get("model_sha") != model_sha or record.get(
        "model_sha"
    ) != model_sha or card.get("model_sha") != model_sha:
        raise CBEndpointValidationError(
            "deferred endpoint result, record, shipcard, and artifact identity differ"
        )
    return fill_slot(shipcard_path, str(slot), record)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("eager", "graph"), required=True)
    parser.add_argument("--base-url", required=True,
                        help="Already-running OpenAI base URL ending in /v1.")
    parser.add_argument("--model-dir", required=True,
                        help="Host path of the CB artifact being served.")
    parser.add_argument("--model-name", required=True,
                        help="Exact expected /v1/models id.")
    parser.add_argument("--serve-manifest", required=True,
                        help="Server-side manifest written after READY.")
    parser.add_argument("--serve-log", default=None,
                        help="Graph arm: vLLM log containing positive capture evidence.")
    parser.add_argument("--shipcard", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--defer-shipcard-fill", action="store_true",
                        help="Write a commit-ready result; do not mutate the shipcard yet.")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_sha: str | None = None
    serve_fingerprint: str | None = None
    manifest_verified = False
    metrics: dict[str, Any] = {"arm": args.arm}
    passed = False
    try:
        if args.prompt != DEFAULT_PROMPT or args.max_tokens != DEFAULT_MAX_TOKENS:
            raise CBEndpointValidationError(
                "exact endpoint gate requires its canonical prompt and 8-token smoke"
            )
        quant_config = validate_cb_artifact(model_dir)
        artifact_decode = validate_cb_artifact_decode_contract(
            model_dir, quant_config
        )
        model_sha = compute_model_sha(model_dir)
        card = load_shipcard(args.shipcard)
        if card.get("model_sha") != model_sha:
            raise CBEndpointValidationError(
                "shipcard model_sha does not match the artifact being served"
            )
        if card.get("weight_stat_attestation") is None:
            raise CBEndpointValidationError(
                "CB shipcard has no post-export weight stat attestation"
            )
        from .shipcard import assert_weight_stat_attestation

        assert_weight_stat_attestation(card, model_dir)
        if (card.get("build") or {}).get("quant_method") != "gridbook":
            raise CBEndpointValidationError(
                "shipcard was not opened by a Gridbook CB exporter"
            )
        manifest_raw = Path(args.serve_manifest).read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        manifest = json.loads(manifest_raw.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise CBEndpointValidationError("serve manifest is not a JSON object")
        serve_fingerprint = validate_serve_manifest(
            manifest,
            arm=args.arm,
            expected_served_model=args.model_name,
            requires_moe_marlin=bool(
                artifact_decode["requires_moe_backend_marlin"]
            ),
            expected_model_sha=model_sha,
        )
        manifest_verified = True
        endpoint_session = _validate_live_server_session(
            manifest,
            expected_served_model=args.model_name,
        )
        endpoint_smoke = run_endpoint_smoke(
            base_url=args.base_url,
            model_name=args.model_name,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            expected_models_identity=endpoint_session[
                "models_endpoint_binding"
            ],
        )
        metrics.update(endpoint_smoke)
        metrics.update({
            "enforce_eager": args.arm == "eager",
            "quantization": "gridbook",
            "kv_cache_dtype": "fp8",
            "tensor_parallel_size": 1,
            "gridbook_runtime_commit": _gridbook_runtime_pin()["commit"],
            "gridbook_runtime_version": _gridbook_runtime_pin()["version"],
            "vllm_version": DSV4_SPARK_VLLM_VERSION,
            "vllm_commit": DSV4_SPARK_VLLM_COMMIT,
            "serve_manifest_sha256": manifest_sha256,
        })
        graph_evidence: dict[str, Any] | None = None
        if args.arm == "graph":
            if not args.serve_log:
                raise CBEndpointValidationError(
                    "graph arm requires --serve-log positive capture evidence"
                )
            graph_evidence = validate_graph_capture_log(args.serve_log)
            metrics["cuda_graph"] = graph_evidence
        elif args.serve_log:
            raise CBEndpointValidationError(
                "eager arm must not claim graph-capture log evidence"
            )
        metrics["endpoint_contract"] = build_endpoint_contract(
            arm=args.arm,
            model_sha=model_sha,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            artifact_decode=artifact_decode,
            endpoint_smoke=endpoint_smoke,
            cuda_graph=graph_evidence,
        )
        detail = (
            f"{args.arm}: exact-pinned CB endpoint returned identical non-empty "
            "greedy completions"
        )
        passed = True
    except Exception as exc:
        detail = f"{args.arm}: {type(exc).__name__}: {exc}"

    slot = f"native_export.{args.arm}"
    record = make_record(
        slot=slot,
        tool="validate_cb_endpoint.py",
        passed=passed,
        model_sha=model_sha,
        metrics=metrics,
        detail=detail,
        spec_decode_detected=False if manifest_verified else None,
        serve_fingerprint=serve_fingerprint,
        git_commit=git_provenance().get("commit"),
        extra={"serve_manifest": str(Path(args.serve_manifest).resolve())},
    )
    result = {
        "schema": ENDPOINT_RESULT_SCHEMA,
        "passed": passed,
        "slot": slot,
        "model_sha": model_sha,
        "serve_fingerprint": serve_fingerprint,
        "detail": detail,
        "metrics": metrics,
        "record": record,
    }
    _write_result(args.output_json, result)
    # The exact shell driver defers this write until its final process/memory
    # checks pass.  Direct callers retain the fail-closed one-shot behavior.
    if not args.defer_shipcard_fill:
        fill_slot(args.shipcard, slot, record)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
