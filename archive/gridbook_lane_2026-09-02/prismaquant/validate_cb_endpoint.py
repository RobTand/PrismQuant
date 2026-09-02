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
    CB_LANE_DSV4_FLASH,
    CB_LANE_GRIDBOOK_GENERIC,
    _manifest_gridbook_runtime_pin,
    _verify_gridbook_distribution_identity,
    cb_serving_lane,
    compute_model_sha,
    fill_slot,
    git_provenance,
    load_shipcard,
    make_record,
)
from .gridbook_assignment import artifact_requires_moe_backend_marlin
from .gridbook_environment import CANONICAL_GOLD_ENVIRONMENT
from .gridbook_serving_runtime_pin import (
    GridbookServingRuntimePinError,
    load_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)
from .allocator_candidates import ROUTE_PENDING_PASSTHROUGH_FORMATS


DSV4_SPARK_VLLM_IMAGE = (
    "eugr/spark-vllm@sha256:"
    "58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869"
)
DSV4_SPARK_VLLM_VERSION = "0.26.1rc1.dev693+g7f7a32cfe.d20260812"
DSV4_SPARK_VLLM_COMMIT = "7f7a32cfe"
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
ARTIFACT_DECODE_CONTRACT_SCHEMA_V2 = (
    "prismaquant.cb_artifact_decode_contract.v2"
)
#: The third decode topology: a CB artifact that is neither of the two DSpark
#: shapes -- no MTP construction namespace to bridge and no FP8 source overlay
#: to reconstruct, just config groups over the checkpoint's own tensors. It is
#: a separate MODE rather than a version bump of either DSpark schema, because
#: the evidence it can offer is different, not newer.
ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN = (
    "prismaquant.cb_artifact_decode_contract.plain.v1"
)
CB_PLAIN_MODE = "cb_plain"
CB_ROUTED_MOE_RUNTIME_FEATURE = "routed_moe_per_role_codebook_lut"
CB_ROUTED_MOE_RUNTIME_FEATURE_VERSION = 1
DSPARK_CB_SIDECAR_SCHEMA = "prismaquant.dspark_cb_sidecar.v1"
DSPARK_CB_SIDECAR_MODE = "quantized_dspark_cb_sidecar"
DSPARK_CB_RUNTIME_FEATURE = "dspark_construction_physical_bridge"
DSPARK_CB_RUNTIME_FEATURE_VERSION = 1
DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE = "source_fp8_block128_w8a16"
DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION = 1
DSPARK_CB_RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v12"
#: The source namespace holding DeepSeek-V4's MTP layers. A DSv4 release either
#: constructs its DSpark stages FROM this namespace (the in-band overlay), IS
#: this namespace (the draft sidecar), or deliberately OMITS it because the
#: draft ships as a second artifact. Spelled here rather than shared because
#: every other site that needs it is a `startswith("mtp.")` literal already;
#: unifying them is a refactor, not part of this contract.
DSPARK_MTP_SOURCE_PREFIX = "mtp."
DSPARK_RENDER_RECIPE_SCHEMA = "prismaquant.dspark_cb_render_recipe.v1"
DSPARK_RENDER_SOURCE_BINDING = "streamed_decoded_cb_source.v1"
DSPARK_CB_RETAINED_GLUE_TENSOR_COUNT = 47
DSPARK_CB_HYBRID_WEIGHT_ONLY_TENSOR_COUNT = 82
ENDPOINT_RESULT_SCHEMA = "prismaquant.cb_endpoint_validation.v1"
ENDPOINT_SESSION_SCHEMA = "prismaquant.cb_endpoint_session.v1"
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_GRAPH_CAPTURE_RE = re.compile(
    r"Graph capturing finished in [0-9]+ secs, took -?[0-9.]+ GiB"
)
GRIDBOOK_SERVED_MODEL_PREFIX = "pq-cb-gridbook-"
#: The Gridbook 0.8.11 serving image, by content digest.  It is built FROM the
#: eugr Spark base above and reports the identical vLLM version/commit, so the
#: two lanes differ only in which container carries the Gridbook wheel and
#: which tokenizer the architecture needs.
GRIDBOOK_SPARK_VLLM_IMAGE = (
    "gridbook@sha256:"
    "9fb09dfb5eb24c0d06701ea040c371202d4d96c17041c35c78e6a7b776c6170d"
)

#: Per-lane serving constants.  Everything OUTSIDE this table -- vLLM
#: version/commit, GPU, the graph compilation config, the whole gate parameter
#: set (8192 len / 1 seq / 1 GiB KV / 0.90 util), the environment profile --
#: was verified lane-invariant and stays a single shared pin.
#:
#: Only three things actually vary: the tokenizer the architecture requires,
#: the container that carries the Gridbook wheel, and the served-model brand.
#: The 32-hex nonce is NOT a lane property -- it is the per-run freshness proof
#: that a receipt cannot be replayed from an earlier serve, and every lane
#: keeps it.
CB_SERVING_LANE_SPECS: Mapping[str, Mapping[str, str]] = {
    CB_LANE_DSV4_FLASH: {
        "served_model_prefix": DSV4_SERVED_MODEL_PREFIX,
        "tokenizer_mode": "deepseek_v4",
        "image": DSV4_SPARK_VLLM_IMAGE,
    },
    CB_LANE_GRIDBOOK_GENERIC: {
        "served_model_prefix": GRIDBOOK_SERVED_MODEL_PREFIX,
        "tokenizer_mode": "auto",
        "image": GRIDBOOK_SPARK_VLLM_IMAGE,
    },
}


def cb_lane_spec(lane: str) -> Mapping[str, str]:
    """Return one lane's serving constants, refusing an unknown lane."""
    spec = CB_SERVING_LANE_SPECS.get(lane)
    if spec is None:
        raise CBEndpointValidationError(f"unknown CB serving lane {lane!r}")
    return spec


_SESSION_MODEL_RES = {
    lane: re.compile(re.escape(spec["served_model_prefix"]) + r"[0-9a-f]{32}")
    for lane, spec in CB_SERVING_LANE_SPECS.items()
}
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
    "--tool-call-parser",
    "--speculative-config",
    "--default-chat-template-kwargs",
})
_SERVE_SWITCH_FLAGS = frozenset({
    "--trust-remote-code",
    "--no-enable-prefix-caching",
    "--enforce-eager",
    "--enable-chunked-prefill",
    "--enable-auto-tool-choice",
})


class CBEndpointValidationError(RuntimeError):
    """The CB endpoint or its serving identity did not satisfy the gate."""


def _gridbook_runtime_pin() -> dict[str, Any]:
    try:
        pin = load_gridbook_serving_runtime_pin()
        require_exact_gridbook_serving_runtime_release(pin)
    except (GridbookServingRuntimePinError, OSError, UnicodeError) as exc:
        raise CBEndpointValidationError(
            f"Gridbook runtime pin is not one strict released commit: {exc}"
        ) from exc
    return {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "wheel_sha256": pin.wheel_sha256,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "required_abi_features": dict(pin.required_abi_features),
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dspark_production_render_attestation(
    quant_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a replayable, source-complete DSpark render attestation.

    Structural sidecar validation remains useful for research artifacts.  A
    release claim is stricter: it must prove the exact source tensors observed
    by the one-pass renderer and persist the complete recipe whose digest was
    checked against source config/header, assignment, and imatrix inputs before
    export.  In particular, this rejects the historical K12 draft carrying
    ``render_identity_verified=false``.
    """
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CBEndpointValidationError(
            "DSpark production artifact has no provenance object"
        )
    if provenance.get("render_identity_verified") is not True:
        raise CBEndpointValidationError(
            "DSpark production artifact does not attest a verified render; "
            "re-export without --allow-unstamped-research"
        )
    tensor_formats = provenance.get("tensor_formats")
    try:
        from . import format_registry as fr
        from .nvfp4_cb_footprint import is_cb_format

        if not isinstance(tensor_formats, Mapping):
            raise TypeError("tensor_formats is not a mapping")
        expected_cb_scope: dict[str, list[str]] = {}
        for raw_qname, raw_format in tensor_formats.items():
            if not isinstance(raw_qname, str) or not raw_qname:
                raise TypeError("tensor_formats contains an invalid qname")
            if not isinstance(raw_format, str) or not raw_format.strip():
                raise TypeError(
                    f"tensor_formats[{raw_qname!r}] is not a format string"
                )
            canonical = fr.canonical_format_name(raw_format.strip().upper())
            if is_cb_format(canonical):
                expected_cb_scope[raw_qname] = [canonical]
        expected_cb_scope = dict(sorted(expected_cb_scope.items()))
        if not expected_cb_scope:
            raise ValueError("finalized assignment contains no CB members")
    except Exception as exc:
        raise CBEndpointValidationError(
            "DSpark artifact has no canonical finalized CB render scope: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    render_identity = provenance.get("cb_render_identity")
    render_scope = render_identity.get("cb_formats_by_qname") if isinstance(
        render_identity, Mapping
    ) else None
    source_shapes = render_identity.get("source_weights_shapes") if isinstance(
        render_identity, Mapping
    ) else None
    source_content = render_identity.get(
        "source_weights_content_sha256"
    ) if isinstance(render_identity, Mapping) else None
    expected_qnames = set(expected_cb_scope)
    if (
        render_scope != expected_cb_scope
        or not isinstance(source_shapes, Mapping)
        or set(source_shapes) != expected_qnames
        or not isinstance(source_content, Mapping)
        or set(source_content) != expected_qnames
    ):
        raise CBEndpointValidationError(
            "DSpark finalized tensor_formats CB scope differs from the exact "
            "render/source-complete scope"
        )
    try:
        from .production_weight_cache import (
            validate_cb_render_identity_metadata,
        )

        validate_cb_render_identity_metadata(
            render_identity,
            expected_formats_by_qname=expected_cb_scope,
            require_source_complete=True,
            where="DSpark production artifact",
        )
    except Exception as exc:
        raise CBEndpointValidationError(
            "DSpark production render identity is not source-complete: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(render_identity, Mapping):  # narrowed by validator
        raise AssertionError("validated render identity is not a mapping")

    attestation = provenance.get("dspark_render_attestation")
    attestation_keys = {
        "schema",
        "source_binding",
        "recipe",
        "recipe_sha256",
        "source_weights_sha256",
        "source_weights_entries",
    }
    if not isinstance(attestation, Mapping):
        raise CBEndpointValidationError(
            "DSpark production artifact has no render attestation"
        )
    _require_exact_keys(
        attestation,
        attestation_keys,
        where="DSpark production render attestation",
    )
    recipe = attestation.get("recipe")
    recipe_keys = {
        "schema",
        "source_binding",
        "source_model_identity",
        "source_config_sha256",
        "mtp_header_identity_sha256",
        "assignment_sha256",
        "col_weights_sha256",
        "render_identity_seed_sha256",
    }
    if not isinstance(recipe, Mapping):
        raise CBEndpointValidationError(
            "DSpark production render attestation has no replayable recipe"
        )
    _require_exact_keys(
        recipe, recipe_keys, where="DSpark production render recipe"
    )
    if (
        attestation.get("schema") != DSPARK_RENDER_RECIPE_SCHEMA
        or attestation.get("source_binding") != DSPARK_RENDER_SOURCE_BINDING
        or recipe.get("schema") != DSPARK_RENDER_RECIPE_SCHEMA
        or recipe.get("source_binding") != DSPARK_RENDER_SOURCE_BINDING
    ):
        raise CBEndpointValidationError(
            "DSpark production render recipe schema/source binding differs"
        )
    for name in (
        "source_config_sha256",
        "mtp_header_identity_sha256",
        "assignment_sha256",
        "col_weights_sha256",
        "render_identity_seed_sha256",
    ):
        _require_sha256(recipe.get(name), where=f"DSpark render recipe {name}")
    if attestation.get("recipe_sha256") != _canonical_json_sha256(recipe):
        raise CBEndpointValidationError(
            "DSpark production render recipe digest is stale"
        )

    source_identity = provenance.get("source_model_identity")
    source_identity_keys = {
        "schema",
        "content_sha256",
        "resolved_commit",
        "checkpoint_shards",
        "checkpoint_tensors",
    }
    if (
        not isinstance(source_identity, Mapping)
        or set(source_identity) != source_identity_keys
        or source_identity.get("schema")
        != "prismaquant.streamed_model.identity.v1"
        or _FINGERPRINT_RE.fullmatch(
            str(source_identity.get("content_sha256", ""))
        ) is None
        or any(
            isinstance(source_identity.get(name), bool)
            or not isinstance(source_identity.get(name), int)
            or int(source_identity.get(name, 0)) <= 0
            for name in ("checkpoint_shards", "checkpoint_tensors")
        )
        or (
            source_identity.get("resolved_commit") is not None
            and (
                not isinstance(source_identity.get("resolved_commit"), str)
                or not source_identity.get("resolved_commit")
            )
        )
        or recipe.get("source_model_identity") != source_identity
    ):
        raise CBEndpointValidationError(
            "DSpark render recipe is not bound to the compact complete-source "
            "checkpoint identity"
        )

    try:
        from .nvfp4_cb_footprint import assignment_serialization_sha256

        if not isinstance(tensor_formats, Mapping):
            raise TypeError("tensor_formats is not a mapping")
        recipe_assignment = dict(tensor_formats)
        # ``mtp.0.main_proj`` is immutable DSpark glue rather than an
        # allocator-owned Linear.  The exporter adds its mandatory W8A16 route
        # after validating the input recipe; remove only that one producer-
        # injected route when replaying the original assignment digest.
        if recipe_assignment.pop("mtp.0.main_proj", None) is None:
            raise ValueError("mandatory mtp.0.main_proj route is absent")
        assignment_sha = assignment_serialization_sha256(recipe_assignment)
    except Exception as exc:
        raise CBEndpointValidationError(
            "DSpark artifact has no canonical finalized assignment: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if recipe.get("assignment_sha256") != assignment_sha:
        raise CBEndpointValidationError(
            "DSpark render recipe assignment differs from artifact provenance"
        )
    if recipe.get("col_weights_sha256") != render_identity.get(
        "col_weights_sha256"
    ):
        raise CBEndpointValidationError(
            "DSpark render recipe imatrix digest differs from render identity"
        )

    # The streaming collector changes only these source-value fields after it
    # has observed every decoded CB input.  Reconstruct the pristine seed and
    # replay the recipe's seed digest exactly.
    render_seed = json.loads(json.dumps(render_identity))
    render_seed["source_weights_complete"] = False
    render_seed["source_weights_shapes"] = {}
    render_seed["source_weights_content_sha256"] = {}
    render_seed["source_weights_sha256"] = None
    if recipe.get("render_identity_seed_sha256") != _canonical_json_sha256(
        render_seed
    ):
        raise CBEndpointValidationError(
            "DSpark render identity does not derive from the attested recipe seed"
        )
    source_content = render_identity.get("source_weights_content_sha256")
    if (
        attestation.get("source_weights_sha256")
        != render_identity.get("source_weights_sha256")
        or not isinstance(source_content, Mapping)
        or attestation.get("source_weights_entries") != len(source_content)
    ):
        raise CBEndpointValidationError(
            "DSpark render attestation source-weight closure differs"
        )
    return {
        "schema": DSPARK_RENDER_RECIPE_SCHEMA,
        "source_model_identity": dict(source_identity),
        "recipe_sha256": str(attestation["recipe_sha256"]),
        "source_weights_sha256": str(attestation["source_weights_sha256"]),
        "source_weights_entries": int(attestation["source_weights_entries"]),
        "attestation_sha256": _canonical_json_sha256(attestation),
    }


def _require_session_model_name(value: object, *, lane: str) -> str:
    """The served-model name must be this lane's brand plus a fresh nonce.

    Fail-closed in BOTH directions: a DSv4 artifact served under the generic
    prefix is refused exactly as a generic artifact served under the DSv4 one
    is, so the brand cannot be borrowed to satisfy a lane's contract.
    """
    pattern = _SESSION_MODEL_RES.get(lane)
    if pattern is None:
        raise CBEndpointValidationError(f"unknown CB serving lane {lane!r}")
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CBEndpointValidationError(
            f"served model name must be the fixed {lane} prefix "
            f"{cb_lane_spec(lane)['served_model_prefix']!r} plus a fresh "
            "128-bit lowercase hexadecimal session nonce; "
            f"got {value!r}"
        )
    return value


def _expected_serve_environment(runtime_pin: Mapping[str, Any]) -> dict[str, str]:
    """Exact set-valued process environment for the endpoint/perf profile."""
    commit = runtime_pin.get("commit")
    version = runtime_pin.get("version")
    wheel_sha256 = runtime_pin.get("wheel_sha256")
    if (
        not isinstance(commit, str)
        or not isinstance(version, str)
        or re.fullmatch(r"[0-9a-f]{64}", str(wheel_sha256 or "")) is None
    ):
        raise CBEndpointValidationError("Gridbook runtime pin is incomplete")
    return {
        **_REQUIRED_SERVE_ENV,
        "PQ_GRIDBOOK_RUNTIME_COMMIT": commit,
        "PQ_GRIDBOOK_RUNTIME_VERSION": version,
        "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256": str(wheel_sha256),
        "PYTHONSAFEPATH": "1",
    }


def _validate_closed_server_environment(
    payload: Mapping[str, Any],
    *,
    runtime_pin: Mapping[str, Any],
    expected_environment: Mapping[str, str] | None = None,
    expected_allowlist: Sequence[str] | None = None,
) -> dict[str, str]:
    """Replay the complete allowlist against every inspected server process."""
    try:
        from tools.serve_fingerprint import SERVER_ENV_ALLOWLIST
    except Exception as exc:  # pragma: no cover - packaging misuse
        raise CBEndpointValidationError(
            "tools.serve_fingerprint is unavailable; run from the PrismaQuant tree"
        ) from exc

    expected = (
        _expected_serve_environment(runtime_pin)
        if expected_environment is None
        else dict(expected_environment)
    )
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        for name, value in expected.items()
    ):
        raise CBEndpointValidationError(
            "serve environment override is not one exact string mapping"
        )
    allowlist = (
        tuple(SERVER_ENV_ALLOWLIST)
        if expected_allowlist is None
        else tuple(expected_allowlist)
    )
    if (
        not allowlist
        or any(not isinstance(name, str) or not name for name in allowlist)
        or len(allowlist) != len(set(allowlist))
        or not set(expected).issubset(allowlist)
    ):
        raise CBEndpointValidationError(
            "serve environment allowlist override is incomplete"
        )
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
        or process_environment.get("allowlist") != sorted(allowlist)
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
    lane: str = CB_LANE_DSV4_FLASH,
    expected_served_model: str,
    requires_moe_marlin: bool,
    expected_options_override: Mapping[str, str] | None = None,
    expected_switches_override: set[str] | frozenset[str] | None = None,
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
        # Lane-derived: `deepseek_v4` names a tokenizer only the DSv4 vendored
        # code registers.  Every lane still passes the flag EXPLICITLY -- an
        # exact-match contract must never have an absent-flag case, or an
        # omitted tokenizer silently becomes whatever vLLM defaults to.
        "--tokenizer-mode": cb_lane_spec(lane)["tokenizer_mode"],
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
    if expected_options_override is not None:
        expected_options = dict(expected_options_override)
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
    if expected_switches_override is not None:
        expected_switches = set(expected_switches_override)
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
    lane: str = CB_LANE_DSV4_FLASH,
    expected_image: str | None = None,
    expected_vllm_version: str = DSV4_SPARK_VLLM_VERSION,
    expected_served_model: str,
    requires_moe_marlin: bool,
    expected_model_sha: str | None = None,
    expected_speculative_config: Mapping[str, Any] | None = None,
    expected_launch_options: Mapping[str, str] | None = None,
    expected_launch_switches: set[str] | frozenset[str] | None = None,
    expected_server_environment: Mapping[str, str] | None = None,
    expected_server_environment_allowlist: Sequence[str] | None = None,
) -> str:
    """Validate server-side identity and return its exact fingerprint."""
    if arm not in {"eager", "graph"}:
        raise CBEndpointValidationError(f"unknown validation arm {arm!r}")
    lane_spec = cb_lane_spec(lane)
    if expected_image is None:
        expected_image = lane_spec["image"]
    _require_session_model_name(expected_served_model, lane=lane)

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
            f"serve image {payload.get('image')!r} != exact {lane} image "
            f"{expected_image!r}"
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
    if expected_speculative_config is None and speculative_config is not None:
        raise CBEndpointValidationError(
            "speculative decoding is configured; CB eager/graph gates require it off"
        )
    if expected_speculative_config is not None:
        try:
            observed_speculative_config = json.loads(
                str(speculative_config)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CBEndpointValidationError(
                "serve manifest speculative config is not strict JSON"
            ) from exc
        if observed_speculative_config != dict(expected_speculative_config):
            raise CBEndpointValidationError(
                "serve manifest speculative config differs from the DSpark recipe"
            )

    argv = payload.get("launch_argv")
    if not isinstance(argv, list) or not argv:
        raise CBEndpointValidationError("serve manifest has no launch argv")
    _canonical_launch_contract(
        argv,
        arm=arm,
        lane=lane,
        expected_served_model=expected_served_model,
        requires_moe_marlin=requires_moe_marlin,
        expected_options_override=expected_launch_options,
        expected_switches_override=expected_launch_switches,
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
    _validate_closed_server_environment(
        payload,
        runtime_pin=runtime_pin,
        expected_environment=expected_server_environment,
        expected_allowlist=expected_server_environment_allowlist,
    )
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
    # The architecture is NOT pinned here.  What this gate proves is that the
    # bytes on disk are the shipcard's Gridbook nvfp4_cb export and that the
    # pinned stack served them; which architecture they encode is the lane's
    # business (`cb_serving_lane`), and it selects the tokenizer and container
    # rather than gating admission.  Pinning `deepseek_v4` here refused every
    # correct non-DSv4 CB artifact -- the same defect as the DSv4 gold contract.
    if not isinstance(model_config, dict) or not model_config.get("model_type"):
        raise CBEndpointValidationError(
            "CB artifact config.json declares no model_type"
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
            "CB artifact route-pending acknowledgements are not canonical"
        )
    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise CBEndpointValidationError("CB artifact has no config_groups")
    # An FP8 block-scaled source overlay is a property of the SOURCE, not of
    # CB: DSv4-Flash ships native FP8 weights that must be declared as
    # passthrough rather than silently requantized, and demanding the overlay
    # is how that is enforced.  A BF16-source model has nothing to declare, so
    # requiring it refused every such artifact.  The real invariant -- that
    # whatever passthrough formats ARE declared carry their route-pending
    # acknowledgements -- is checked below and stays on every lane.
    if cb_serving_lane(root) == CB_LANE_DSV4_FLASH and not any(
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
            "CB artifact did not record required route-pending "
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


def _expected_dspark_cb_physical_targets(*, n_mtp_layers: int) -> list[str]:
    tails = (
        "attn.wkv",
        "attn.wo_b",
        "attn.wq_a",
        "attn.wq_b",
        "ffn.experts.down_proj",
        "ffn.experts.gate_up_proj",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w2",
        "ffn.shared_experts.w3",
    )
    return sorted(
        f"mtp.{stage}.{tail}"
        for stage in range(n_mtp_layers)
        for tail in tails
    )


def _expected_dspark_source_passthrough_mapping(
    *, num_hidden_layers: int, n_mtp_layers: int
) -> dict[str, str]:
    return dict(sorted({
        "mtp.0.main_proj": "model.main_proj",
        **{
            f"mtp.{stage}.attn.wo_a": (
                f"model.layers.{num_hidden_layers + stage}.attn.wo_a"
            )
            for stage in range(n_mtp_layers)
        },
    }.items()))


def _canonical_string_list(value: object, *, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise CBEndpointValidationError(
            f"{where} must be a sorted duplicate-free list of nonempty strings"
        )
    return list(value)


def _validate_dspark_cb_sidecar_artifact(
    root: Path,
    *,
    model_config: Mapping[str, Any],
    quant_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the physical/config namespace split of a quantized draft.

    This is intentionally a header-only proof.  A sidecar's CB groups are
    named where Gridbook constructs the decoder, while the serialized tensors
    retain the checkpoint's ``mtp.*`` names.  Both sets, every CB companion,
    and the four immutable W8A16 source projections (``main_proj`` plus one
    grouped-BMM ``wo_a`` per stage) are closed here before an endpoint can
    claim it decoded the artifact.
    """
    from .artifact_completeness import (
        _read_safetensors_header,
        read_artifact_header,
    )
    from .cb_export_config import (
        _two_tier_scale_coding,
        parse_source_passthrough_declaration,
        source_passthrough_config_group,
        source_passthrough_wire_id,
    )
    from .cb_layout import (
        FP4_GROUP,
        SCALE_CODING_TWO_TIER,
        SCALE_CODING_V1,
        SUPERBLOCK,
        VEC_DIM,
        codebook_subtable_shapes,
        parse_format_name,
        type_size,
    )
    from .dspark_source_metadata import (
        DSPARK_STAGE_COUNT,
        _released_dspark_tensor_layout,
        dspark_cb_construction_target_for_physical_output,
        validate_dspark_target_bridge,
    )
    from .export_native_compressed import _explicit_regex
    from .nvfp4_activation_contract import (
        FP4_GROUP_SIZE,
        NVFP4_ACTIVATION_CONTRACT_KEY,
        NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        NVFP4_ACTIVATION_EXECUTION,
        NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
    )

    provenance = quant_config.get("provenance")
    sidecar = (
        provenance.get("dspark_cb_sidecar")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(sidecar, Mapping):
        raise CBEndpointValidationError(
            "quantized DSpark artifact has no sidecar provenance"
        )
    sidecar_keys = {
        "schema",
        "num_hidden_layers",
        "n_mtp_layers",
        "physical_namespace",
        "construction_namespace",
        "physical_cb_targets",
        "construction_cb_targets",
        "source_passthrough_targets",
        "source_passthrough_physical_to_construction",
        "activation_bridge_present",
    }
    _require_exact_keys(
        sidecar, sidecar_keys, where="DSpark CB sidecar provenance"
    )
    num_hidden_layers = model_config.get("num_hidden_layers")
    n_mtp_layers = model_config.get("n_mtp_layers")
    if (
        not isinstance(num_hidden_layers, int)
        or isinstance(num_hidden_layers, bool)
        or num_hidden_layers <= 0
        or n_mtp_layers != DSPARK_STAGE_COUNT
        or sidecar.get("schema") != DSPARK_CB_SIDECAR_SCHEMA
        or sidecar.get("num_hidden_layers") != num_hidden_layers
        or sidecar.get("n_mtp_layers") != DSPARK_STAGE_COUNT
        or sidecar.get("physical_namespace") != "mtp.{stage}"
        or sidecar.get("construction_namespace")
        != "model.layers.{num_hidden_layers+stage}"
        or not isinstance(sidecar.get("activation_bridge_present"), bool)
    ):
        raise CBEndpointValidationError(
            "DSpark CB sidecar provenance disagrees with model topology"
        )
    target_layer_ids = model_config.get("dspark_target_layer_ids")
    if (
        not isinstance(target_layer_ids, list)
        or len(target_layer_ids) != DSPARK_STAGE_COUNT
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= num_hidden_layers
            for value in target_layer_ids
        )
        or len(set(target_layer_ids)) != DSPARK_STAGE_COUNT
        or model_config.get("num_nextn_predict_layers") != 1
        or model_config.get("n_shared_experts") != 1
        or str(model_config.get("expert_dtype", "")).lower()
        not in {"fp4", "mxfp4", "mx_fp4"}
        or not isinstance(model_config.get("dspark_block_size"), int)
        or isinstance(model_config.get("dspark_block_size"), bool)
        or int(model_config["dspark_block_size"]) <= 0
        or not isinstance(model_config.get("dspark_markov_rank"), int)
        or isinstance(model_config.get("dspark_markov_rank"), bool)
        or int(model_config["dspark_markov_rank"]) <= 0
    ):
        raise CBEndpointValidationError(
            "DSpark sidecar model config is not the released three-stage "
            "construction contract"
        )

    expected_physical = _expected_dspark_cb_physical_targets(
        n_mtp_layers=DSPARK_STAGE_COUNT
    )
    physical = _canonical_string_list(
        sidecar.get("physical_cb_targets"),
        where="DSpark physical CB targets",
    )
    if physical != expected_physical:
        raise CBEndpointValidationError(
            "DSpark sidecar does not contain the exact three-stage decoder "
            "target set"
        )
    try:
        physical_to_construction = {
            target: dspark_cb_construction_target_for_physical_output(
                target, model_config
            )
            for target in physical
        }
    except Exception as exc:
        raise CBEndpointValidationError(
            f"DSpark sidecar namespace mapping is invalid: {exc}"
        ) from exc
    construction = _canonical_string_list(
        sidecar.get("construction_cb_targets"),
        where="DSpark construction CB targets",
    )
    if construction != sorted(physical_to_construction.values()):
        raise CBEndpointValidationError(
            "DSpark physical and construction CB targets do not map exactly"
        )
    expected_source_mapping = _expected_dspark_source_passthrough_mapping(
        num_hidden_layers=num_hidden_layers,
        n_mtp_layers=DSPARK_STAGE_COUNT,
    )
    if (
        sidecar.get("source_passthrough_targets")
        != sorted(expected_source_mapping)
        or sidecar.get("source_passthrough_physical_to_construction")
        != expected_source_mapping
    ):
        raise CBEndpointValidationError(
            "DSpark sidecar source routes must be exactly main_proj plus one "
            "grouped-BMM wo_a per stage, with physical/construction mapping"
        )

    groups = quant_config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise CBEndpointValidationError(
            "DSpark CB sidecar has no config groups"
        )
    cb_schemes: dict[str, Mapping[str, Any]] = {}
    cb_formats: dict[str, str] = {}
    cb_layouts: dict[str, tuple[object, int]] = {}
    source_groups: list[Mapping[str, Any]] = []
    for raw_name, raw_group in groups.items():
        if not isinstance(raw_group, Mapping):
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} is not an object"
            )
        if raw_group.get("format") == "source-passthrough":
            source_groups.append(raw_group)
            continue
        scheme = raw_group.get("scheme")
        targets = raw_group.get("targets")
        raw_format = raw_group.get("format")
        parsed_format = (
            parse_format_name(raw_format)
            if isinstance(raw_format, str)
            else None
        )
        if (
            not isinstance(scheme, Mapping)
            or parsed_format is None
            or not isinstance(targets, list)
            or any(not isinstance(target, str) or not target for target in targets)
            or targets != sorted(set(targets))
        ):
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} is not a canonical CB group"
            )
        family, format_k = parsed_format
        canonical_format = family.name(format_k)
        if raw_format != canonical_format:
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} format label is not "
                f"canonical: {raw_format!r} != {canonical_format!r}"
            )
        scale_coding = (
            SCALE_CODING_TWO_TIER
            if family.grid == "fp4"
            else SCALE_CODING_V1
        )
        expected_layout = {
            "grid": family.grid,
            "mode": family.mode,
            "k": format_k,
            "superblock": SUPERBLOCK,
            "group_size": FP4_GROUP if family.grid == "fp4" else 0,
            "vec_dim": VEC_DIM,
            "n_sub": family.n_sub,
            "type_size": type_size(format_k, family.grid, scale_coding),
            "act_bits": 4 if family.grid == "fp4" else 8,
        }
        mismatched_layout = sorted(
            key for key, expected in expected_layout.items()
            if scheme.get(key) != expected
            or (
                isinstance(expected, int)
                and isinstance(scheme.get(key), bool)
            )
        )
        expected_scheme_keys = {
            *expected_layout,
            "codebook_source",
            "codebook_ref",
            "codebook_group",
        }
        if family.grid == "fp4":
            expected_scheme_keys.add("scale_coding")
            if scheme.get("scale_coding") != _two_tier_scale_coding():
                mismatched_layout.append("scale_coding")
        if scheme.get("activation_contract") is not None:
            expected_scheme_keys.add("activation_contract")
        if set(scheme) != expected_scheme_keys:
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} scheme keys differ from "
                f"canonical {canonical_format}: missing="
                f"{sorted(expected_scheme_keys - set(scheme))}, extra="
                f"{sorted(set(scheme) - expected_scheme_keys)}"
            )
        if mismatched_layout:
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} format/scheme layout "
                f"disagree for {canonical_format}: "
                f"{sorted(set(mismatched_layout))}"
            )
        codebook_source = scheme.get("codebook_source")
        codebook_group = scheme.get("codebook_group")
        if (
            codebook_source not in {"lattice", "learned"}
            or (codebook_source == "lattice" and codebook_group is not None)
            or (
                codebook_source == "learned"
                and (not isinstance(codebook_group, str) or not codebook_group)
            )
        ):
            raise CBEndpointValidationError(
                f"DSpark config group {raw_name!r} has non-canonical "
                "codebook source/group identity"
            )
        for target in targets:
            if target in cb_schemes:
                raise CBEndpointValidationError(
                    f"DSpark construction target {target!r} is declared twice"
                )
            cb_schemes[target] = scheme
            cb_formats[target] = canonical_format
            cb_layouts[target] = (family, format_k)
    if sorted(cb_schemes) != construction:
        raise CBEndpointValidationError(
            "DSpark config groups do not exactly cover construction targets"
        )

    tensor_formats = provenance.get("tensor_formats")
    if not isinstance(tensor_formats, Mapping):
        raise CBEndpointValidationError(
            "DSpark sidecar provenance has no finalized tensor_formats assignment"
        )
    expected_cb_assignment: dict[str, str] = {}
    raw_num_experts = model_config.get("n_routed_experts")
    if (
        isinstance(raw_num_experts, bool)
        or not isinstance(raw_num_experts, int)
        or raw_num_experts <= 0
    ):
        raise CBEndpointValidationError(
            "DSpark sidecar model config has no positive routed-expert count"
        )
    num_experts = raw_num_experts
    for physical_target, construction_target in physical_to_construction.items():
        format_name = cb_formats[construction_target]
        if physical_target.endswith(".ffn.experts.gate_up_proj"):
            prefix = physical_target.removesuffix("gate_up_proj")
            members = (
                f"{prefix}{expert}.{projection}"
                for expert in range(num_experts)
                for projection in ("gate_proj", "up_proj")
            )
        elif physical_target.endswith(".ffn.experts.down_proj"):
            prefix = physical_target.removesuffix("down_proj")
            members = (
                f"{prefix}{expert}.down_proj"
                for expert in range(num_experts)
            )
        else:
            members = (physical_target,)
        expected_cb_assignment.update(
            (member, format_name) for member in members
        )
    observed_cb_assignment: dict[str, str] = {}
    for raw_qname, raw_format in tensor_formats.items():
        if (
            not isinstance(raw_qname, str)
            or not raw_qname
            or not isinstance(raw_format, str)
            or not raw_format
        ):
            raise CBEndpointValidationError(
                "DSpark finalized tensor_formats assignment is malformed"
            )
        parsed = parse_format_name(raw_format)
        if parsed is None:
            if "_CB_" in raw_format.upper():
                raise CBEndpointValidationError(
                    f"DSpark tensor_formats contains unknown CB label "
                    f"{raw_format!r} at {raw_qname!r}"
                )
            continue
        member_family, member_k = parsed
        canonical = member_family.name(member_k)
        if raw_format != canonical:
            raise CBEndpointValidationError(
                f"DSpark tensor_formats label at {raw_qname!r} is not "
                f"canonical: {raw_format!r} != {canonical!r}"
            )
        observed_cb_assignment[raw_qname] = canonical
    if observed_cb_assignment != dict(sorted(expected_cb_assignment.items())):
        missing = sorted(
            set(expected_cb_assignment) - set(observed_cb_assignment)
        )
        extra = sorted(
            set(observed_cb_assignment) - set(expected_cb_assignment)
        )
        mismatched = sorted(
            qname
            for qname in set(expected_cb_assignment) & set(observed_cb_assignment)
            if expected_cb_assignment[qname] != observed_cb_assignment[qname]
        )
        raise CBEndpointValidationError(
            "DSpark config-group formats differ from finalized physical "
            "tensor_formats assignment: missing="
            f"{missing[:8]}, extra={extra[:8]}, mismatched={mismatched[:8]}"
        )

    codebook_file = quant_config.get("codebook_file")
    codebook_path = root / str(codebook_file)
    try:
        codebook_header = _read_safetensors_header(codebook_path)
    except Exception as exc:
        raise CBEndpointValidationError(
            f"DSpark codebook sidecar header is unreadable: {exc}"
        ) from exc
    referenced_codebooks: set[str] = set()
    expected_codebook_contract: dict[str, tuple[str, list[int]]] = {}
    for construction_target, scheme in cb_schemes.items():
        refs = scheme.get("codebook_ref")
        if (
            not isinstance(refs, list)
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or refs != list(dict.fromkeys(refs))
        ):
            raise CBEndpointValidationError(
                f"{construction_target}: CB scheme has no canonical codebook refs"
            )
        family, k = cb_layouts[construction_target]
        expected_shapes = codebook_subtable_shapes(
            k, family.mode, family.n_sub
        )
        if len(refs) != len(expected_shapes):
            raise CBEndpointValidationError(
                f"{construction_target}: CB scheme refs disagree with the "
                f"canonical {cb_formats[construction_target]} subtable layout"
            )
        expected_ref_suffixes = (
            [
                f".{cb_formats[construction_target]}.sub{index}"
                for index in range(len(expected_shapes))
            ]
            if len(expected_shapes) > 1
            else [f".{cb_formats[construction_target]}"]
        )
        for ref, raw_shape, suffix in zip(
            refs, expected_shapes, expected_ref_suffixes, strict=True
        ):
            if not ref.startswith("cb_codebook.") or not ref.endswith(suffix):
                raise CBEndpointValidationError(
                    f"{construction_target}: codebook ref {ref!r} is not "
                    f"canonical for {cb_formats[construction_target]}"
                )
            shape = list(raw_shape)
            prior = expected_codebook_contract.setdefault(ref, ("F16", shape))
            if prior != ("F16", shape):
                raise CBEndpointValidationError(
                    f"{ref}: CB schemes disagree on the codebook tensor shape"
                )
            referenced_codebooks.add(ref)
    if set(codebook_header) != referenced_codebooks:
        raise CBEndpointValidationError(
            "DSpark .pqcb tensors do not exactly equal config-group refs: missing="
            f"{sorted(referenced_codebooks - set(codebook_header))[:8]}, extra="
            f"{sorted(set(codebook_header) - referenced_codebooks)[:8]}"
        )
    for name, (dtype, shape) in expected_codebook_contract.items():
        meta = codebook_header[name]
        if meta.get("dtype") != dtype or meta.get("shape") != shape:
            raise CBEndpointValidationError(
                f"{name}: codebook tensor must be {dtype} {shape}, got "
                f"{meta.get('dtype')} {meta.get('shape')}"
            )
    codebook_payload_bytes = sum(
        int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
        for meta in codebook_header.values()
    )
    serialized_payload = provenance.get("serialized_payload")
    codebook_digests = provenance.get("codebook_sha256")
    if (
        not isinstance(serialized_payload, Mapping)
        or serialized_payload.get("n_tensors") != len(physical)
        or serialized_payload.get("codebook_sidecar_bytes")
        != codebook_payload_bytes
        or not isinstance(serialized_payload.get("sidecars"), list)
        or not isinstance(codebook_digests, Mapping)
        or set(codebook_digests) != referenced_codebooks
        or any(
            _FINGERPRINT_RE.fullmatch(str(value)) is None
            for value in codebook_digests.values()
        )
    ):
        raise CBEndpointValidationError(
            "DSpark CB serialized-payload/codebook provenance is incomplete"
        )
    sidecar_refs = {
        str(ref)
        for record in serialized_payload["sidecars"]
        if isinstance(record, Mapping)
        for ref in (record.get("codebook_ref") or ())
    }
    if sidecar_refs != referenced_codebooks:
        raise CBEndpointValidationError(
            "DSpark serialized-payload refs differ from config-group refs"
        )
    try:
        import torch
        from safetensors.torch import load_file as load_safetensors

        codebook_tensors = load_safetensors(
            str(codebook_path), device="cpu"
        )
        observed_codebook_digests = {
            name: hashlib.sha256(
                tensor.to(torch.float16).cpu().numpy().tobytes()
            ).hexdigest()
            for name, tensor in codebook_tensors.items()
        }
    except Exception as exc:
        raise CBEndpointValidationError(
            "DSpark .pqcb tensor payloads are unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if observed_codebook_digests != dict(sorted(codebook_digests.items())):
        mismatched = sorted(
            name for name in referenced_codebooks
            if observed_codebook_digests.get(name) != codebook_digests.get(name)
        )
        raise CBEndpointValidationError(
            "DSpark .pqcb logical FP16 tensor payload SHA-256 differs from "
            f"provenance.codebook_sha256; mismatched={mismatched[:8]}"
        )

    expected_source_group = source_passthrough_config_group(
        "FP8_BLOCK_UE8M0_SOURCE"
    )
    if len(source_groups) != 1:
        raise CBEndpointValidationError(
            "DSpark sidecar must have exactly one source-passthrough group"
        )
    observed_source_group = dict(source_groups[0])
    source_group_targets = observed_source_group.pop("targets", None)
    expected_source_group_targets = sorted(
        _explicit_regex(target) for target in expected_source_mapping
    )
    if (
        observed_source_group != expected_source_group
        or source_group_targets != expected_source_group_targets
    ):
        raise CBEndpointValidationError(
            "DSpark source bases are not the exact block-128 source-FP8 "
            "W8A16 group"
        )
    try:
        declared_source = parse_source_passthrough_declaration(quant_config)
    except Exception as exc:
        raise CBEndpointValidationError(
            f"DSpark source-passthrough declaration is invalid: {exc}"
        ) from exc
    source_wire_id = source_passthrough_wire_id(
        "FP8_BLOCK_UE8M0_SOURCE"
    )
    expected_source_declaration = {
        construction: source_wire_id
        for construction in expected_source_mapping.values()
    }
    if declared_source != expected_source_declaration:
        raise CBEndpointValidationError(
            "DSpark delegated-native declaration must contain exactly the "
            "four W8A16 main_proj/wo_a construction routes"
        )

    execution_contracts = quant_config.get("execution_contracts")
    execution: Mapping[str, Any] | None = None
    if execution_contracts is not None:
        if (
            not isinstance(execution_contracts, Mapping)
            or set(execution_contracts) != {NVFP4_ACTIVATION_CONTRACT_KEY}
            or not isinstance(
                execution_contracts.get(NVFP4_ACTIVATION_CONTRACT_KEY), Mapping
            )
        ):
            raise CBEndpointValidationError(
                "DSpark sidecar execution_contracts is not the one NVFP4 "
                "activation contract"
            )
        execution = execution_contracts[NVFP4_ACTIVATION_CONTRACT_KEY]
    bridge_present = "dspark_target_bridge" in quant_config
    bridge = quant_config.get("dspark_target_bridge")
    activation_present = execution is not None
    if (
        sidecar.get("activation_bridge_present") is not activation_present
        or bridge_present is not activation_present
        or (activation_present and not isinstance(bridge, Mapping))
    ):
        raise CBEndpointValidationError(
            "DSpark activation execution contract and target bridge must be "
            "wholly present or wholly absent"
        )

    activation_targets: set[str] = set()
    if execution is not None:
        target_names = _canonical_string_list(
            execution.get("target_names"),
            where="DSpark activation execution targets",
        )
        if (
            execution.get("schema") != NVFP4_ACTIVATION_CONTRACT_SCHEMA
            or execution.get("contract") != NVFP4_ACTIVATION_EXECUTION
            or execution.get("group_size") != FP4_GROUP_SIZE
            or execution.get("tensor_suffix")
            != NVFP4_INPUT_GLOBAL_SCALE_SUFFIX
            or execution.get("value_dtype") != "float32"
            or execution.get("target_count") != len(target_names)
            or _FINGERPRINT_RE.fullmatch(
                str(execution.get("target_values_sha256", ""))
            ) is None
            or not set(target_names).issubset(physical)
        ):
            raise CBEndpointValidationError(
                "DSpark activation execution contract is malformed or names "
                "a non-CB physical target"
            )
        activation_targets = set(target_names)
        contracted_construction = sorted(
            physical_to_construction[target] for target in target_names
        )
        try:
            validate_dspark_target_bridge(
                bridge,
                model_config,
                contracted_cb_construction_targets=contracted_construction,
                activation_execution_contract=execution,
            )
        except Exception as exc:
            raise CBEndpointValidationError(
                f"DSpark activation target bridge is invalid: {exc}"
            ) from exc

    for physical_target, construction_target in physical_to_construction.items():
        scheme = cb_schemes[construction_target]
        activation_ref = scheme.get("activation_contract")
        if physical_target in activation_targets:
            if (
                scheme.get("grid") != "fp4"
                or activation_ref != NVFP4_ACTIVATION_CONTRACT_KEY
            ):
                raise CBEndpointValidationError(
                    f"{construction_target}: activation bridge target has no "
                    "matching FP4 scheme contract"
                )
        elif "activation_contract" in scheme:
            raise CBEndpointValidationError(
                f"{construction_target}: scheme claims an activation contract "
                "without a bridged physical target"
            )

    try:
        header = read_artifact_header(root)
    except Exception as exc:
        raise CBEndpointValidationError(
            f"DSpark sidecar safetensors headers are unreadable: {exc}"
        ) from exc
    try:
        expected_source_layout, source_targets = _released_dspark_tensor_layout(
            hidden_size=int(model_config["hidden_size"]),
            num_heads=int(model_config["num_attention_heads"]),
            head_dim=int(model_config["head_dim"]),
            q_lora_rank=int(model_config["q_lora_rank"]),
            o_groups=int(model_config["o_groups"]),
            o_lora_rank=int(model_config["o_lora_rank"]),
            moe_intermediate_size=int(model_config["moe_intermediate_size"]),
            num_experts=int(model_config["n_routed_experts"]),
            vocab_size=int(model_config["vocab_size"]),
            markov_rank=int(model_config["dspark_markov_rank"]),
            target_layer_count=len(model_config["dspark_target_layer_ids"]),
            hc_mult=int(model_config.get("hc_mult", 4)),
            fp8_block=(128, 128),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CBEndpointValidationError(
            f"DSpark sidecar config cannot derive the released tensor layout: {exc}"
        ) from exc
    expected_source_bases = set(expected_source_mapping)
    source_decoder_bases = set(source_targets) - {"mtp.0.main_proj"}
    source_passthrough_decoder_bases = (
        expected_source_bases - {"mtp.0.main_proj"}
    )
    cb_source_decoder_bases = (
        source_decoder_bases - source_passthrough_decoder_bases
    )
    source_decoder_planes = {
        base + suffix
        for base in source_decoder_bases
        for suffix in (".weight", ".scale")
    }
    expected_glue = {
        name: contract
        for name, contract in expected_source_layout.items()
        if name not in source_decoder_planes
        and not name.startswith("mtp.0.main_proj.")
    }
    if len(expected_glue) != DSPARK_CB_RETAINED_GLUE_TENSOR_COUNT:
        raise CBEndpointValidationError(
            "internal DSpark released-layout contract no longer has exactly "
            f"{DSPARK_CB_RETAINED_GLUE_TENSOR_COUNT} retained glue tensors; "
            f"got {len(expected_glue)}"
        )

    source_decoder_suffixes = (
        ".weight",
        ".scale",
        ".weight_scale",
        ".weight_scale_inv",
    )
    duplicate_source_planes = sorted(
        name
        for name in header
        for suffix in source_decoder_suffixes
        if name.endswith(suffix)
        and name[: -len(suffix)] in cb_source_decoder_bases
    )
    if duplicate_source_planes:
        raise CBEndpointValidationError(
            "DSpark CB sidecar also carries source planes for CB decoder "
            "targets: "
            f"{duplicate_source_planes[:8]}"
        )

    expected_qweights = {target + ".cb_qweight" for target in physical}
    expected_companions = {
        target + ".weight_scale"
        for target, construction_target in physical_to_construction.items()
        if cb_schemes[construction_target].get("grid") == "fp8"
    } | {
        target + ".input_global_scale" for target in activation_targets
    }
    expected_source_names = {
        name
        for base in expected_source_bases
        for name in (base + ".weight", base + ".scale")
    }
    missing_source_contracts = sorted(
        expected_source_names - set(expected_source_layout)
    )
    if missing_source_contracts:
        raise CBEndpointValidationError(
            "internal released DSpark layout lacks hybrid W8A16 source "
            f"planes: {missing_source_contracts}"
        )
    expected_mtp_names = (
        set(expected_glue)
        | expected_qweights
        | expected_companions
        | expected_source_names
    )
    base_weight_only_names = (
        set(expected_glue) | expected_qweights | expected_source_names
    )
    if len(base_weight_only_names) != DSPARK_CB_HYBRID_WEIGHT_ONLY_TENSOR_COUNT:
        raise CBEndpointValidationError(
            "internal DSpark hybrid contract must contain exactly "
            f"{DSPARK_CB_HYBRID_WEIGHT_ONLY_TENSOR_COUNT} weight-only tensors "
            "(27 CB qweights + 47 glue + 8 W8A16 source planes), got "
            f"{len(base_weight_only_names)}"
        )

    observed_qweights = {
        name for name in header if name.endswith(".cb_qweight")
    }
    if observed_qweights != expected_qweights:
        raise CBEndpointValidationError(
            "DSpark physical qweight headers do not exactly match the "
            "construction config: missing="
            f"{sorted(expected_qweights - observed_qweights)[:8]}, extra="
            f"{sorted(observed_qweights - expected_qweights)[:8]}"
        )

    companion_suffixes = (".weight_scale", ".input_global_scale")
    observed_companions = {
        name
        for name in header
        if name.endswith(companion_suffixes)
        and any(name.startswith(target + ".") for target in physical)
    }
    if observed_companions != expected_companions:
        raise CBEndpointValidationError(
            "DSpark CB companion planes do not match their scheme/activation "
            "contracts: missing="
            f"{sorted(expected_companions - observed_companions)[:8]}, extra="
            f"{sorted(observed_companions - expected_companions)[:8]}"
        )

    if set(header) != expected_mtp_names:
        raise CBEndpointValidationError(
            "DSpark sidecar tensor namespace is not the closed decoder/glue "
            "layout: missing="
            f"{sorted(expected_mtp_names - set(header))[:8]}, extra="
            f"{sorted(set(header) - expected_mtp_names)[:8]}"
        )
    for name, contract in expected_glue.items():
        meta = header[name]
        if (
            meta.get("dtype") not in contract.dtypes
            or meta.get("shape") != list(contract.shape)
        ):
            raise CBEndpointValidationError(
                f"{name}: DSpark glue tensor differs from the released "
                f"dtype/shape contract {sorted(contract.dtypes)} {contract.shape}"
            )

    for physical_target, construction_target in physical_to_construction.items():
        qweight_name = physical_target + ".cb_qweight"
        qweight_meta = header[qweight_name]
        qweight_shape = qweight_meta.get("shape")
        scheme = cb_schemes[construction_target]
        source_contract = expected_source_layout.get(
            physical_target + ".weight"
        )
        if source_contract is None:
            if physical_target.endswith(".gate_up_proj"):
                expert_source = physical_target[: -len("gate_up_proj")] + "0.w1.weight"
                member = expected_source_layout.get(expert_source)
                logical_shape = (
                    int(model_config["n_routed_experts"]),
                    2 * member.shape[0],
                    member.shape[1] * 2,
                ) if member is not None else None
            elif physical_target.endswith(".down_proj"):
                expert_source = physical_target[: -len("down_proj")] + "0.w2.weight"
                member = expected_source_layout.get(expert_source)
                logical_shape = (
                    int(model_config["n_routed_experts"]),
                    member.shape[0],
                    member.shape[1] * 2,
                ) if member is not None else None
            else:
                logical_shape = None
        else:
            logical_shape = source_contract.shape
        type_size = scheme.get("type_size")
        superblock = scheme.get("superblock")
        if (
            logical_shape is None
            or not isinstance(type_size, int)
            or isinstance(type_size, bool)
            or type_size <= 0
            or not isinstance(superblock, int)
            or isinstance(superblock, bool)
            or superblock <= 0
            or logical_shape[-1] % superblock
        ):
            raise CBEndpointValidationError(
                f"{construction_target}: scheme cannot derive an exact qweight shape"
            )
        expected_qweight_shape = [
            *logical_shape[:-1],
            logical_shape[-1] // superblock * type_size,
        ]
        if (
            qweight_meta.get("dtype") != "U8"
            or qweight_shape != expected_qweight_shape
        ):
            raise CBEndpointValidationError(
                f"{qweight_name}: qweight header must be U8 "
                f"{expected_qweight_shape}, got {qweight_meta.get('dtype')} "
                f"{qweight_shape}"
            )
    for name in expected_companions:
        meta = header[name]
        target = name.rsplit(".", 1)[0]
        construction_target = physical_to_construction[target]
        qweight_shape = header[target + ".cb_qweight"]["shape"]
        expected_shape = (
            qweight_shape[:-1]
            if name.endswith(".weight_scale")
            else [1]
        )
        if meta.get("dtype") != "F32" or (
            meta.get("shape") != expected_shape
        ):
            raise CBEndpointValidationError(
                f"{name}: CB companion must be F32 {expected_shape}, got "
                f"{meta.get('dtype')} {meta.get('shape')} for "
                f"{construction_target}"
            )

    for name in sorted(expected_source_names):
        contract = expected_source_layout[name]
        meta = header.get(name)
        if (
            not isinstance(meta, Mapping)
            or meta.get("dtype") not in contract.dtypes
            or meta.get("shape") != list(contract.shape)
        ):
            raise CBEndpointValidationError(
                f"{name}: hybrid W8A16 source plane differs from the released "
                f"dtype/shape contract {sorted(contract.dtypes)} "
                f"{contract.shape}"
            )

    structural_names = sorted(
        set(expected_glue)
        | expected_qweights
        | expected_companions
        | expected_source_names
    )
    header_contract = {
        name: {
            "dtype": header[name].get("dtype"),
            "shape": header[name].get("shape"),
        }
        for name in structural_names
    }
    return {
        "provenance": dict(sidecar),
        "physical_to_construction": dict(sorted(physical_to_construction.items())),
        "header_contract": header_contract,
        "source_passthrough": dict(expected_source_declaration),
        "activation_execution_contract": (
            dict(execution) if execution is not None else None
        ),
        "dspark_target_bridge": dict(bridge) if isinstance(bridge, Mapping) else None,
    }


def _unit_claim_names(unit: str, profile, is_routed_expert_unit) -> tuple[str, ...]:
    """The names a config group has to claim to fully cover *unit*.

    Almost always that is the unit itself, and the caller's cover is then a
    plain per-tensor cover exactly as before.

    A PACKED ROUTED EXPERT STACK is the exception: one physical tensor
    carrying two logical roles, where the CB ABI binds a codebook *per role*.
    A per-role learned layer therefore ships two config groups naming
    ``…experts.gate_proj`` and ``…experts.up_proj``, and neither of them names
    the ``…experts.gate_up_proj`` that is actually on disk.  Read per tensor,
    both groups look empty and the tensor looks unclaimed -- a correct
    artifact reported as broken, the same blind spot the completeness gate had
    (11 DSv4-Flash stacks, 2026-08-16).  Read per role, each group covers the
    role it names and the stack is covered when both roles are.

    This is the pinned runtime's own resolution rule rather than a widening.
    The pinned Gridbook's ``_resolve_moe_codebook_roles`` (read at 0.8.5,
    unchanged in the pinned 0.8.11) gathers ``codebook_ref`` per
    role across the targets matching a stack, refuses two targets claiming one
    role with different books, and refuses a stack naming no book for some
    role.  Checker and consumer thus still refuse together.  The decomposition
    is the profile's declarative ``packed_experts.projection_splits`` -- the
    table the exporter used to emit the halves, and the one Gridbook keeps as
    ``_FUSED_FALLBACK`` because DeepseekV4 has no ``packed_modules_mapping``.

    Bounded like the completeness fix: routed-expert parent on dotted
    boundaries, and a leaf that decomposes only to itself is not multi-role.
    """

    if "." not in unit:
        return (unit,)
    parent, leaf = unit.rsplit(".", 1)
    if not is_routed_expert_unit(parent):
        return (unit,)
    try:
        members = tuple(profile.packed_expert_projection_names(leaf))
    except Exception:                          # pragma: no cover - defensive
        members = ()
    if len(members) < 2:
        return (unit,)
    return tuple(f"{parent}.{member}" for member in members)


def _validate_plain_cb_artifact(
    root: Path, quant_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove a plain CB artifact's config groups cover what it actually ships.

    The DSpark modes get their substance from a topology: a construction
    namespace to bridge, or an FP8 source to reconstruct.  A plain CB artifact
    has neither, so its evidence has to be the cover itself -- every quantized
    plane on disk claimed by exactly one declared mechanism, every config-group
    target resolved to a tensor that exists, every group's units carrying the
    SAME planes, and every codebook a group references present in the ``.pqcb``
    with the payload digest its provenance claims.

    All four are computed from the artifact's own bytes.  The provenance
    inventories (``weight_content_manifest``, ``tensor_formats``) are claims,
    not evidence, so they are deliberately not restated here.
    """
    from .artifact_completeness import (
        _checkpoint_spellings,
        _claimed_by_self_or_ancestor,
        _detect_profile_quietly,
        _group_claimed_units,
        _is_routed_expert_unit,
        _QUANTIZED_WEIGHT_PLANE_SUFFIXES,
        _SCALE_BEARING_DTYPES,
        _read_safetensors_header,
        _unit_variants,
        read_artifact_header,
    )
    from .cb_layout import codebook_subtable_shapes, parse_format_name

    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CBEndpointValidationError("CB artifact has no provenance object")
    groups = quant_config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise CBEndpointValidationError("CB artifact declares no config groups")

    header = read_artifact_header(root)
    profile = _detect_profile_quietly(root)

    #: unit -> every plane it ships, so a group whose members disagree on their
    #: planes (one lost its scale) cannot pass as uniform.
    planes_by_unit: dict[str, set[str]] = {}
    quantized_units: set[str] = set()
    for name, meta in header.items():
        if "." not in name:
            continue
        unit, leaf = name.rsplit(".", 1)
        planes_by_unit.setdefault(unit, set()).add(leaf)
        if name.endswith(".weight"):
            if meta.get("dtype") in _SCALE_BEARING_DTYPES:
                quantized_units.add(unit)
            continue
        if any(
            name.endswith(suffix) for suffix in _QUANTIZED_WEIGHT_PLANE_SUFFIXES
        ):
            quantized_units.add(unit)
    if not quantized_units:
        raise CBEndpointValidationError(
            "CB artifact ships no quantized weight plane at all"
        )

    embedding_claimed = {
        spelling
        for entry in (
            (quant_config.get("quantized_embedding") or {}).get("units") or {}
        )
        for spelling in _checkpoint_spellings(str(entry), profile)
    }
    ignored = {
        spelling
        for entry in (quant_config.get("ignore") or ())
        for spelling in _checkpoint_spellings(str(entry), profile)
    }
    # THE THIRD DECLARED MECHANISM. An FP8-source model can ship a Linear as a
    # verbatim copy of its source bytes -- quantized, scale-bearing, on disk,
    # and deliberately in no config group. `artifact_completeness` has always
    # recognized it (`source_passthrough.units`, 336 of them on the DSv4-Flash
    # body); this cover was written against artifacts that carried none, so it
    # read every passthrough Linear as claimed by nothing. Same key path and
    # same spelling normalization as the completeness gate, so the two agree on
    # which units are declared. Matched by EXACT variant rather than by
    # ancestry, also to match that gate: a passthrough declaration names one
    # tensor's own bytes, and letting `layers.0.attn` cover everything under it
    # would silently exempt siblings that were never declared.
    passthrough_declared = {
        spelling
        for entry in (
            (quant_config.get("source_passthrough") or {}).get("units") or {}
        )
        for spelling in _checkpoint_spellings(str(entry), profile)
    }

    def _is_passthrough(unit: str) -> bool:
        return bool(_unit_variants(unit, profile) & passthrough_declared)

    #: unit -> the claim names that together cover it. One entry per unit
    #: everywhere except a packed routed expert stack, which needs one per
    #: role; see :func:`_unit_claim_names`.
    claim_names_by_unit = {
        unit: _unit_claim_names(unit, profile, _is_routed_expert_unit)
        for unit in quantized_units
    }
    unit_by_claim_name = {
        claim: unit
        for unit, claims in claim_names_by_unit.items()
        for claim in claims
    }

    group_cover: list[dict[str, Any]] = []
    #: claim name -> the groups claiming it. Keyed by claim rather than by unit
    #: so the two roles of one packed stack are two distinct claims: gate and
    #: up carrying different books is the ABI, not an overlap.
    claimed_by: dict[str, list[str]] = {}
    referenced_codebooks: set[str] = set()
    #: Every distinct unit some config group covers. Kept as a SET because two
    #: per-role groups legitimately cover one packed stack, so summing the
    #: per-group counts would count that stack twice.
    group_units: set[str] = set()
    expected_codebook_contract: dict[str, tuple[str, list[int]]] = {}
    requires_routed_moe_lut = False
    for group_name in sorted(groups):
        group = groups[group_name]
        if not isinstance(group, Mapping):
            raise CBEndpointValidationError(
                f"{group_name}: config group is not an object"
            )
        targets = group.get("targets")
        if not isinstance(targets, list) or not targets:
            raise CBEndpointValidationError(
                f"{group_name}: config group claims no target"
            )
        spellings = {
            spelling
            for entry in _group_claimed_units({"config_groups": {group_name: group}})
            for spelling in _checkpoint_spellings(str(entry), profile)
        }
        claims: list[str] = []
        for unit, unit_claims in claim_names_by_unit.items():
            if _claimed_by_self_or_ancestor(unit, spellings, profile):
                # A target naming the packed tensor -- or any ancestor of it,
                # e.g. the whole `…ffn.experts` stack -- speaks for every role
                # it carries with the one book it declares. That is the lattice
                # spelling, and it stays legal.
                claims.extend(unit_claims)
                continue
            if len(unit_claims) == 1:
                continue
            claims.extend(
                claim
                for claim in unit_claims
                if _claimed_by_self_or_ancestor(claim, spellings, profile)
            )
        claims = sorted(claims)
        if not claims:
            raise CBEndpointValidationError(
                f"{group_name}: no tensor on disk answers to any of its "
                f"{len(targets)} targets"
            )
        members = sorted({unit_by_claim_name[claim] for claim in claims})
        for claim in claims:
            claimed_by.setdefault(claim, []).append(group_name)
        signatures = {
            tuple(sorted(planes_by_unit.get(unit, ()))) for unit in members
        }
        if len(signatures) != 1:
            raise CBEndpointValidationError(
                f"{group_name}: its units do not all ship the same planes, so "
                f"the group is half-exported: {sorted(signatures)[:3]}"
            )
        (signature,) = signatures
        format_name = group.get("format")
        parsed = parse_format_name(str(format_name)) if format_name else None
        # A CB group carries its layout under `scheme`; a delegated stock group
        # (the FP8 head) carries compressed-tensors' own `weights` block and no
        # scheme at all.
        scheme = group.get("scheme")
        if parsed is not None and not isinstance(scheme, Mapping):
            raise CBEndpointValidationError(
                f"{group_name}: CB group {format_name!r} declares no scheme"
            )
        refs = scheme.get("codebook_ref") if isinstance(scheme, Mapping) else None
        if parsed is None:
            if refs:
                raise CBEndpointValidationError(
                    f"{group_name}: a non-CB group cannot reference a codebook"
                )
        else:
            family, format_k = parsed
            if str(format_name) != family.name(format_k):
                raise CBEndpointValidationError(
                    f"{group_name}: format label {format_name!r} is not canonical"
                )
            expected_shapes = codebook_subtable_shapes(
                format_k, family.mode, family.n_sub
            )
            expected_suffixes = (
                [f".{format_name}.sub{index}" for index in range(len(expected_shapes))]
                if len(expected_shapes) > 1
                else [f".{format_name}"]
            )
            if (
                not isinstance(refs, list)
                or len(refs) != len(expected_shapes)
                or refs != list(dict.fromkeys(refs))
            ):
                raise CBEndpointValidationError(
                    f"{group_name}: CB group refs disagree with the canonical "
                    f"{format_name} subtable layout"
                )
            for ref, raw_shape, suffix in zip(
                refs, expected_shapes, expected_suffixes, strict=True
            ):
                if (
                    not isinstance(ref, str)
                    or not ref.startswith("cb_codebook.")
                    or not ref.endswith(suffix)
                ):
                    raise CBEndpointValidationError(
                        f"{group_name}: codebook ref {ref!r} is not canonical "
                        f"for {format_name}"
                    )
                shape = list(raw_shape)
                prior = expected_codebook_contract.setdefault(ref, ("F16", shape))
                if prior != ("F16", shape):
                    raise CBEndpointValidationError(
                        f"{ref}: CB groups disagree on the codebook tensor shape"
                    )
                referenced_codebooks.add(ref)
        if any(".experts" in str(target) for target in targets):
            requires_routed_moe_lut = True
        group_units.update(members)
        group_cover.append(
            {
                "group": group_name,
                "format": str(format_name),
                "target_count": len(targets),
                "unit_count": len(members),
                "planes": list(signature),
                "codebook_refs": sorted(str(ref) for ref in (refs or ())),
            }
        )

    overclaimed = sorted(
        claim for claim, names in claimed_by.items() if len(names) > 1
    )
    if overclaimed:
        raise CBEndpointValidationError(
            "config groups overlap -- these units are claimed more than once, "
            f"so their format is ambiguous: {overclaimed[:8]}"
        )
    # Refused even when the duplicate claims agree on their book, which
    # Gridbook's `setdefault` would tolerate. Stricter than the consumer is the
    # fail-closed direction, and a shipped artifact declaring one role twice is
    # an exporter bug worth surfacing rather than absorbing.
    unclaimed = sorted(
        unit
        for unit in quantized_units
        if not any(claim in claimed_by for claim in claim_names_by_unit[unit])
        and not _claimed_by_self_or_ancestor(unit, embedding_claimed, profile)
        and not _is_passthrough(unit)
    )
    if unclaimed:
        raise CBEndpointValidationError(
            "quantized tensors claimed by no config group and no embedding "
            f"declaration: {unclaimed[:8]}"
        )
    # NO contradiction check for passthrough-and-grouped. The two records are
    # complementary by design, not alternatives: `source_passthrough_config_group`
    # emits a scheme-less group stating LAYOUT while the declaration states
    # ROUTING ("routing is stated once, in the `source_passthrough`
    # declaration", cb_export_config.py). On the DSv4-Flash body 272 of 336
    # passthrough units carry both, and the served 112.69 artifact has the same
    # population -- refusing the pair would refuse every FP8-source release.
    half_claimed = sorted(
        f"{unit} (missing "
        f"{[claim for claim in claim_names_by_unit[unit] if claim not in claimed_by]})"
        for unit in quantized_units
        if any(claim in claimed_by for claim in claim_names_by_unit[unit])
        and not all(claim in claimed_by for claim in claim_names_by_unit[unit])
    )
    if half_claimed:
        # EVERY role, never any -- the mirror of Gridbook's own refusal to
        # decode a stack that names no book for some role. A half-declared
        # packed stack would otherwise pass here and then decode the
        # undeclared rows with the other role's codebook.
        raise CBEndpointValidationError(
            "packed routed stacks whose roles are only partly claimed, so the "
            f"unclaimed rows have no codebook: {half_claimed[:8]}"
        )
    contradicted = sorted(
        unit
        for unit in quantized_units
        if _claimed_by_self_or_ancestor(unit, ignored, profile)
    )
    if contradicted:
        raise CBEndpointValidationError(
            "quantized tensors are also listed in `ignore`, which claims they "
            f"are plain floats: {contradicted[:8]}"
        )

    codebook_file = quant_config.get("codebook_file")
    codebook_digests = provenance.get("codebook_sha256")
    if referenced_codebooks:
        if not isinstance(codebook_file, str) or not codebook_file.endswith(".pqcb"):
            raise CBEndpointValidationError(
                "CB artifact references codebooks but declares no .pqcb sidecar"
            )
        try:
            codebook_header = _read_safetensors_header(root / codebook_file)
        except Exception as exc:
            raise CBEndpointValidationError(
                f"CB codebook sidecar header is unreadable: {exc}"
            ) from exc
        if set(codebook_header) != referenced_codebooks:
            raise CBEndpointValidationError(
                ".pqcb tensors do not exactly equal config-group refs: missing="
                f"{sorted(referenced_codebooks - set(codebook_header))[:8]}, "
                f"extra={sorted(set(codebook_header) - referenced_codebooks)[:8]}"
            )
        for name, (dtype, shape) in expected_codebook_contract.items():
            meta = codebook_header[name]
            if meta.get("dtype") != dtype or meta.get("shape") != shape:
                raise CBEndpointValidationError(
                    f"{name}: codebook tensor must be {dtype} {shape}, got "
                    f"{meta.get('dtype')} {meta.get('shape')}"
                )
        if (
            not isinstance(codebook_digests, Mapping)
            or set(codebook_digests) != referenced_codebooks
            or any(
                _FINGERPRINT_RE.fullmatch(str(value)) is None
                for value in codebook_digests.values()
            )
        ):
            raise CBEndpointValidationError(
                "provenance.codebook_sha256 does not digest exactly the "
                "codebooks the config groups reference"
            )
        try:
            import torch
            from safetensors.torch import load_file as load_safetensors

            observed = {
                name: hashlib.sha256(
                    tensor.to(torch.float16).cpu().numpy().tobytes()
                ).hexdigest()
                for name, tensor in load_safetensors(
                    str(root / codebook_file), device="cpu"
                ).items()
            }
        except Exception as exc:
            raise CBEndpointValidationError(
                f".pqcb tensor payloads are unreadable: {type(exc).__name__}: {exc}"
            ) from exc
        mismatched = sorted(
            name
            for name in referenced_codebooks
            if observed.get(name) != codebook_digests.get(name)
        )
        if mismatched:
            raise CBEndpointValidationError(
                ".pqcb payload SHA-256 differs from provenance.codebook_sha256; "
                f"mismatched={mismatched[:8]}"
            )

    renderer_abi = provenance.get("renderer_abi")
    if not isinstance(renderer_abi, str) or not renderer_abi:
        raise CBEndpointValidationError(
            "CB artifact declares no renderer ABI"
        )

    # Derived, never asserted: a feature is required because the artifact
    # SHIPS the thing that needs it.  A pin missing a feature the artifact
    # never uses must not refuse it, and a pin missing one it does use must.
    required_features: dict[str, int] = {}
    if requires_routed_moe_lut:
        required_features[CB_ROUTED_MOE_RUNTIME_FEATURE] = (
            CB_ROUTED_MOE_RUNTIME_FEATURE_VERSION
        )
    if any(
        isinstance(group, Mapping)
        and group.get("source_format") == "FP8_BLOCK_UE8M0_SOURCE"
        for group in groups.values()
    ):
        required_features[DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE] = (
            DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION
        )

    return {
        "renderer_abi": renderer_abi,
        "codebook_file": str(codebook_file) if codebook_file else None,
        "codebook_ref_count": len(referenced_codebooks),
        # Internal: cross-checked against the completeness classification by
        # the caller, deliberately NOT emitted into the receipt. A receipt
        # carries per-group counts and no unit names, so it cannot dedupe a
        # stack two per-role groups share; the exact check belongs where the
        # sets exist.
        "group_units": sorted(group_units),
        "quantized_unit_count": len(quantized_units),
        "tensor_count": len(header),
        "group_cover": group_cover,
        "required_runtime_features": required_features,
        "cover_sha256": _canonical_json_sha256(
            {
                "claimed_by": {
                    claim: names[0] for claim, names in sorted(claimed_by.items())
                },
                # Units no group claims. Passthrough is excluded so the label
                # stays true; it is deliberately NOT added as a new key, which
                # would move this digest for every artifact ever validated,
                # and it cannot hide a real difference -- a unit that is
                # passthrough here and CB elsewhere shows up in `claimed_by`
                # there, and one that is passthrough here and nothing there is
                # refused there.
                "embedding_units": sorted(
                    unit
                    for unit in quantized_units
                    if not any(
                        claim in claimed_by for claim in claim_names_by_unit[unit]
                    )
                    and not _is_passthrough(unit)
                ),
                "ignore": sorted(ignored),
                "planes_by_unit": {
                    unit: sorted(planes_by_unit.get(unit, ()))
                    for unit in sorted(quantized_units)
                },
            }
        ),
    }


def _recorded_excluded_namespaces(
    provenance: Mapping[str, Any],
) -> frozenset[str]:
    """Namespaces the producer recorded as deliberately omitted.

    Absent or malformed reads as "nothing was omitted", which is the
    fail-closed direction: the caller uses this to grant a permission, so an
    unreadable declaration must not grant it.
    """

    recorded = provenance.get("excluded_namespaces")
    if not isinstance(recorded, (list, tuple)):
        return frozenset()
    return frozenset(str(value) for value in recorded)


def validate_cb_artifact_decode_contract(
    model_dir: str | Path,
    quant_config: Mapping[str, Any] | None = None,
    *,
    runtime_pin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove complete source-overlay or quantized-sidecar decode coverage.

    ``runtime_pin`` is optional so a producer can validate artifact structure
    before a Gridbook release is cut.  Endpoint/release callers pass the live
    pin and therefore cannot serve a sidecar until the namespace-bridge ABI is
    present in that exact release.
    """
    root = Path(model_dir)
    if quant_config is None:
        quant_config = validate_cb_artifact(root)
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CBEndpointValidationError(
            "CB artifact has no provenance object"
        )
    recorded_overlay = provenance.get("dspark_source_overlay")
    recorded_sidecar = provenance.get("dspark_cb_sidecar")
    lane = cb_serving_lane(root)
    if recorded_overlay is not None and recorded_sidecar is not None:
        raise CBEndpointValidationError(
            "artifact provenance declares both dspark_source_overlay and "
            "dspark_cb_sidecar, which are alternative topologies"
        )
    if lane == CB_LANE_DSV4_FLASH:
        # The DSv4 release IS a DSpark topology; a DSv4 artifact presenting
        # itself as plain would shed the bridge contract its serving stack
        # depends on. There are THREE topologies, not two:
        #
        #   in-band  -- one artifact carrying the MTP construction stages,
        #               declared as `dspark_source_overlay`;
        #   draft    -- the `mtp.`-subset draft alone, declared as
        #               `dspark_cb_sidecar` (what `--dspark-cb-sidecar` builds,
        #               and it requires exactly `--subset-prefix mtp.`);
        #   split    -- the BODY half of a two-artifact release, whose `mtp.`
        #               namespace was deliberately excluded because the draft
        #               above ships beside it.
        #
        # The split body declares neither topology object, because neither
        # describes it: it constructs no MTP stages and it is not the draft.
        # What it does declare is the omission itself. Requiring that
        # declaration is what keeps this from becoming a hole -- an artifact
        # that merely LOST its bridge records no exclusion and still refuses,
        # so the gate still fails closed on the case it was written for.
        # (`excluded_namespaces` is producer-written: the exporter records the
        # `--exclude-namespace` set it was given.)
        if recorded_overlay is None and recorded_sidecar is None:
            if DSPARK_MTP_SOURCE_PREFIX not in _recorded_excluded_namespaces(
                provenance
            ):
                raise CBEndpointValidationError(
                    "DSv4 artifact provenance must declare one of "
                    "dspark_source_overlay or dspark_cb_sidecar, or record "
                    f"{DSPARK_MTP_SOURCE_PREFIX!r} in excluded_namespaces to "
                    "declare itself the body half of a split release"
                )
        elif recorded_overlay is not None and (
            DSPARK_MTP_SOURCE_PREFIX in _recorded_excluded_namespaces(
                provenance
            )
        ):
            # The overlay is BUILT from the source MTP layers. Claiming both
            # would promise construction stages out of bytes the artifact
            # states it did not ship.
            raise CBEndpointValidationError(
                "DSv4 artifact declares dspark_source_overlay while recording "
                f"{DSPARK_MTP_SOURCE_PREFIX!r} as an excluded namespace; the "
                "overlay is constructed from exactly those source layers"
            )
    elif recorded_overlay is not None or recorded_sidecar is not None:
        raise CBEndpointValidationError(
            f"artifact on lane {lane!r} declares a DSpark topology, which only "
            f"the {CB_LANE_DSV4_FLASH!r} lane serves"
        )

    plain_mode = recorded_overlay is None and recorded_sidecar is None
    sidecar_mode = recorded_sidecar is not None
    try:
        from .artifact_completeness import assert_artifact_complete

        completeness = assert_artifact_complete(
            root,
            **(
                {"verbatim_prefixes": ()}
                if sidecar_mode or plain_mode
                else {}
            ),
        )
    except Exception as exc:
        raise CBEndpointValidationError(
            f"artifact decode coverage is incomplete: {type(exc).__name__}: {exc}"
        ) from exc

    completeness_payload = asdict(completeness)
    classified = (
        len(completeness.cb_units)
        + len(completeness.passthrough_units)
        + len(completeness.verbatim_namespace_units)
    )
    common_evidence: dict[str, Any] = {
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
        "requires_moe_backend_marlin": artifact_requires_moe_backend_marlin(
            quant_config
        ),
    }

    if plain_mode:
        cover = _validate_plain_cb_artifact(root, quant_config)
        embedding_unit_count = len(getattr(completeness, "embedding_units", ()))
        if cover["quantized_unit_count"] != classified + embedding_unit_count:
            raise CBEndpointValidationError(
                "plain CB decode accounting differs from the completeness "
                f"classification: {cover['quantized_unit_count']} quantized "
                f"unit(s) on disk vs {classified + embedding_unit_count} "
                "classified"
            )
        # The exact form of the receipt's cover/CB cross-check, done here
        # because this is the only place holding both the cover's DISTINCT unit
        # set and the completeness classification; the replay keeps a weaker
        # bound for the same reason it cannot dedupe. Passthrough is subtracted
        # rather than filtered out of the cover: the classifier gives a
        # passthrough declaration precedence over a group claim, so a unit with
        # both is counted there as passthrough and must be counted the same way
        # here.
        covered_cb = set(cover["group_units"]) - set(completeness.passthrough_units)
        if covered_cb != set(completeness.cb_units):
            raise CBEndpointValidationError(
                "plain CB decode group cover does not account for exactly the "
                "classified CB units: covered-not-classified="
                f"{sorted(covered_cb - set(completeness.cb_units))[:6]}, "
                "classified-not-covered="
                f"{sorted(set(completeness.cb_units) - covered_cb)[:6]}"
            )
        evidence = {
            "schema": ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN,
            "mode": CB_PLAIN_MODE,
            **common_evidence,
            "embedding_unit_count": embedding_unit_count,
            "renderer_abi": cover["renderer_abi"],
            "codebook_file": cover["codebook_file"],
            "codebook_ref_count": cover["codebook_ref_count"],
            "quantized_unit_count": cover["quantized_unit_count"],
            "tensor_count": cover["tensor_count"],
            "group_cover": cover["group_cover"],
            "cover_sha256": cover["cover_sha256"],
            "required_runtime_features": cover["required_runtime_features"],
            "required_runtime_contract_schema": (
                DSPARK_CB_RUNTIME_CONTRACT_SCHEMA
            ),
        }
        evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
        _validate_artifact_decode_record(evidence)
        if runtime_pin is not None:
            _require_artifact_decode_runtime_features(evidence, runtime_pin)
        return evidence

    if sidecar_mode:
        try:
            model_config = json.loads(
                (root / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CBEndpointValidationError(
                f"DSpark sidecar model config is unreadable: {exc}"
            ) from exc
        if not isinstance(model_config, Mapping):
            raise CBEndpointValidationError(
                "DSpark sidecar model config is not an object"
            )
        sidecar_payload = _validate_dspark_cb_sidecar_artifact(
            root,
            model_config=model_config,
            quant_config=quant_config,
        )
        evidence = {
            "schema": ARTIFACT_DECODE_CONTRACT_SCHEMA_V2,
            "mode": DSPARK_CB_SIDECAR_MODE,
            **common_evidence,
            "dspark_cb_sidecar": dict(recorded_sidecar),
            "dspark_cb_sidecar_sha256": _canonical_json_sha256(
                sidecar_payload
            ),
            "activation_bridge_present": bool(
                recorded_sidecar.get("activation_bridge_present")
            ),
            "required_runtime_features": {
                DSPARK_CB_RUNTIME_FEATURE: DSPARK_CB_RUNTIME_FEATURE_VERSION,
                DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
                    DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
            },
            "required_runtime_contract_schema": (
                DSPARK_CB_RUNTIME_CONTRACT_SCHEMA
            ),
        }
        evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
        _validate_artifact_decode_record(evidence)
        if runtime_pin is not None:
            _require_artifact_decode_runtime_features(evidence, runtime_pin)
        return evidence

    try:
        from .dspark_source_metadata import (
            discover_dspark_source_overlay_from_artifact,
        )

        overlay = discover_dspark_source_overlay_from_artifact(root)
    except Exception as exc:
        raise CBEndpointValidationError(
            "artifact DSpark overlay headers are invalid: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if overlay is None:
        raise CBEndpointValidationError(
            "artifact is not the released DSv4Flash three-stage DSpark topology"
        )
    overlay_provenance = overlay.provenance()
    if recorded_overlay != overlay_provenance:
        raise CBEndpointValidationError(
            "artifact DSpark overlay provenance does not match its tensor headers"
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
        **common_evidence,
        "dspark_overlay": overlay_provenance,
        "dspark_overlay_sha256": _canonical_json_sha256(overlay_payload),
    }
    evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
    _validate_artifact_decode_record(evidence)
    if runtime_pin is not None:
        _require_artifact_decode_runtime_features(evidence, runtime_pin)
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


def _validate_dspark_cb_sidecar_decode_record(
    payload: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "mode",
        "complete",
        "completeness_sha256",
        "declared_unit_count",
        "cb_unit_count",
        "passthrough_unit_count",
        "verbatim_namespace_unit_count",
        "classified_unit_count",
        "route_pending_acknowledged",
        "excluded_namespaces",
        "dspark_cb_sidecar",
        "dspark_cb_sidecar_sha256",
        "activation_bridge_present",
        "required_runtime_features",
        "required_runtime_contract_schema",
        "requires_moe_backend_marlin",
        "evidence_sha256",
    }
    _require_exact_keys(
        payload, expected_keys, where="DSpark CB sidecar decode contract"
    )
    if (
        payload.get("schema") != ARTIFACT_DECODE_CONTRACT_SCHEMA_V2
        or payload.get("mode") != DSPARK_CB_SIDECAR_MODE
        or payload.get("complete") is not True
    ):
        raise CBEndpointValidationError(
            "DSpark CB sidecar decode contract does not attest complete coverage"
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
                f"DSpark CB sidecar decode contract {key} is not a "
                "non-negative integer"
            )
    if (
        payload.get("declared_unit_count", 0) <= 0
        or payload.get("classified_unit_count", 0) <= 0
        or payload.get("verbatim_namespace_unit_count") != 0
        or payload.get("classified_unit_count")
        != sum(
            int(payload[key])
            for key in (
                "cb_unit_count",
                "passthrough_unit_count",
                "verbatim_namespace_unit_count",
            )
        )
    ):
        raise CBEndpointValidationError(
            "DSpark CB sidecar decode accounting is incomplete or admits a "
            "verbatim draft namespace"
        )
    for key, where in (
        ("route_pending_acknowledged", "route-pending acknowledgements"),
        ("excluded_namespaces", "excluded namespaces"),
    ):
        value = payload.get(key)
        if (
            not isinstance(value, list)
            or value != sorted(set(str(item) for item in value))
        ):
            raise CBEndpointValidationError(
                f"DSpark CB sidecar {where} are not canonical"
            )
    if payload.get("excluded_namespaces") != []:
        raise CBEndpointValidationError(
            "DSpark CB sidecar cannot exclude a namespace from decode coverage"
        )
    if not isinstance(payload.get("requires_moe_backend_marlin"), bool):
        raise CBEndpointValidationError(
            "DSpark CB sidecar has no boolean marlin-route decision"
        )
    _require_sha256(
        payload.get("completeness_sha256"), where="completeness digest"
    )
    _require_sha256(
        payload.get("dspark_cb_sidecar_sha256"),
        where="DSpark CB sidecar digest",
    )

    sidecar = payload.get("dspark_cb_sidecar")
    if not isinstance(sidecar, Mapping):
        raise CBEndpointValidationError(
            "DSpark CB sidecar decode contract has no provenance"
        )
    sidecar_keys = {
        "schema",
        "num_hidden_layers",
        "n_mtp_layers",
        "physical_namespace",
        "construction_namespace",
        "physical_cb_targets",
        "construction_cb_targets",
        "source_passthrough_targets",
        "source_passthrough_physical_to_construction",
        "activation_bridge_present",
    }
    _require_exact_keys(
        sidecar, sidecar_keys, where="DSpark CB sidecar provenance"
    )
    num_hidden_layers = sidecar.get("num_hidden_layers")
    n_mtp_layers = sidecar.get("n_mtp_layers")
    expected_source_mapping = (
        _expected_dspark_source_passthrough_mapping(
            num_hidden_layers=num_hidden_layers,
            n_mtp_layers=3,
        )
        if isinstance(num_hidden_layers, int)
        and not isinstance(num_hidden_layers, bool)
        and num_hidden_layers > 0
        else None
    )
    if (
        sidecar.get("schema") != DSPARK_CB_SIDECAR_SCHEMA
        or not isinstance(num_hidden_layers, int)
        or isinstance(num_hidden_layers, bool)
        or num_hidden_layers <= 0
        or n_mtp_layers != 3
        or sidecar.get("physical_namespace") != "mtp.{stage}"
        or sidecar.get("construction_namespace")
        != "model.layers.{num_hidden_layers+stage}"
        or sidecar.get("source_passthrough_targets")
        != sorted(expected_source_mapping or {})
        or sidecar.get("source_passthrough_physical_to_construction")
        != expected_source_mapping
        or not isinstance(sidecar.get("activation_bridge_present"), bool)
        or sidecar.get("activation_bridge_present")
        is not payload.get("activation_bridge_present")
    ):
        raise CBEndpointValidationError(
            "DSpark CB sidecar receipt carries invalid topology or bridge state"
        )
    physical = _canonical_string_list(
        sidecar.get("physical_cb_targets"),
        where="DSpark receipt physical CB targets",
    )
    expected_physical = _expected_dspark_cb_physical_targets(
        n_mtp_layers=3
    )
    construction = _canonical_string_list(
        sidecar.get("construction_cb_targets"),
        where="DSpark receipt construction CB targets",
    )
    expected_construction = sorted(
        "model.layers."
        f"{num_hidden_layers + int(target.split('.')[1])}."
        + ".".join(target.split(".")[2:])
        for target in expected_physical
    )
    if physical != expected_physical or construction != expected_construction:
        raise CBEndpointValidationError(
            "DSpark CB sidecar receipt target sets are not the exact "
            "three-stage mapping"
        )
    if payload.get("required_runtime_features") != {
        DSPARK_CB_RUNTIME_FEATURE: DSPARK_CB_RUNTIME_FEATURE_VERSION,
        DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE:
            DSPARK_CB_SOURCE_FP8_RUNTIME_FEATURE_VERSION,
    }:
        raise CBEndpointValidationError(
            "DSpark CB sidecar receipt does not pin its Gridbook namespace "
            "and source-FP8 W8A16 ABIs"
        )
    if (
        payload.get("required_runtime_contract_schema")
        != DSPARK_CB_RUNTIME_CONTRACT_SCHEMA
    ):
        raise CBEndpointValidationError(
            "DSpark CB sidecar receipt does not require the Gridbook "
            "runtime-contract v4 schema"
        )

    recorded_evidence_sha = _require_sha256(
        payload.get("evidence_sha256"), where="artifact evidence digest"
    )
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256")
    if recorded_evidence_sha != _canonical_json_sha256(unstamped):
        raise CBEndpointValidationError(
            "DSpark CB sidecar decode evidence digest is stale"
        )


def _require_artifact_decode_runtime_features(
    artifact_decode: Mapping[str, Any],
    runtime_pin: Mapping[str, Any],
) -> None:
    """Require every decode-receipt ABI from the exact serving release."""
    required = artifact_decode.get("required_runtime_features")
    if required is None:
        return
    if not isinstance(required, Mapping):
        raise CBEndpointValidationError(
            "artifact decode runtime feature requirements are malformed"
        )
    required_schema = artifact_decode.get(
        "required_runtime_contract_schema"
    )
    if (
        required_schema != DSPARK_CB_RUNTIME_CONTRACT_SCHEMA
        or runtime_pin.get("runtime_contract_schema") != required_schema
    ):
        raise CBEndpointValidationError(
            "Gridbook runtime pin does not implement the DSpark sidecar "
            "runtime-contract schema: required="
            f"{required_schema!r}, observed="
            f"{runtime_pin.get('runtime_contract_schema')!r}"
        )
    observed = runtime_pin.get("required_abi_features")
    if not isinstance(observed, Mapping):
        raise CBEndpointValidationError(
            "Gridbook runtime pin has no ABI feature map"
        )
    mismatches = {
        str(name): {"required": version, "observed": observed.get(name)}
        for name, version in required.items()
        if observed.get(name) != version
    }
    if mismatches:
        raise CBEndpointValidationError(
            "Gridbook runtime pin does not implement the DSpark sidecar "
            f"decode ABI: {mismatches}"
        )


def _validate_cb_plain_decode_record(payload: Mapping[str, Any]) -> None:
    """Replay a plain CB decode receipt with no disk and no server.

    Everything here is a number that could have come out differently: drop a
    target and a group's unit count stops matching, lose a scale plane and its
    group's plane signature stops being uniform, rebuild the books and the
    codebook ref count moves.  A receipt whose numbers are structurally fixed
    proves nothing, which is why the unit counts are required to be positive
    and to add up rather than merely to be present.
    """
    expected_keys = {
        "schema",
        "mode",
        "complete",
        "completeness_sha256",
        "declared_unit_count",
        "cb_unit_count",
        "passthrough_unit_count",
        "verbatim_namespace_unit_count",
        "classified_unit_count",
        "embedding_unit_count",
        "route_pending_acknowledged",
        "excluded_namespaces",
        "renderer_abi",
        "codebook_file",
        "codebook_ref_count",
        "quantized_unit_count",
        "tensor_count",
        "group_cover",
        "cover_sha256",
        "required_runtime_features",
        "required_runtime_contract_schema",
        "requires_moe_backend_marlin",
        "evidence_sha256",
    }
    _require_exact_keys(payload, expected_keys, where="plain CB decode contract")
    if (
        payload.get("schema") != ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN
        or payload.get("mode") != CB_PLAIN_MODE
        or payload.get("complete") is not True
    ):
        raise CBEndpointValidationError(
            "plain CB decode contract does not attest complete coverage"
        )
    for key in (
        "declared_unit_count",
        "cb_unit_count",
        "passthrough_unit_count",
        "verbatim_namespace_unit_count",
        "classified_unit_count",
        "embedding_unit_count",
        "codebook_ref_count",
        "quantized_unit_count",
        "tensor_count",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CBEndpointValidationError(
                f"plain CB decode contract {key} is not a non-negative integer"
            )
    if payload.get("verbatim_namespace_unit_count") != 0:
        raise CBEndpointValidationError(
            "a plain CB artifact has no namespace that ships undeclared on "
            "purpose, so it cannot claim verbatim units"
        )
    if payload.get("classified_unit_count") != sum(
        int(payload[key])
        for key in (
            "cb_unit_count",
            "passthrough_unit_count",
            "verbatim_namespace_unit_count",
        )
    ):
        raise CBEndpointValidationError(
            "plain CB decode classified-unit accounting differs"
        )
    if payload.get("quantized_unit_count") != int(
        payload["classified_unit_count"]
    ) + int(payload["embedding_unit_count"]):
        raise CBEndpointValidationError(
            "plain CB decode leaves quantized tensors unclassified"
        )
    if payload.get("cb_unit_count", 0) <= 0:
        raise CBEndpointValidationError(
            "plain CB decode contract classified no CB unit at all"
        )
    if int(payload["tensor_count"]) < int(payload["quantized_unit_count"]):
        raise CBEndpointValidationError(
            "plain CB decode contract counts more quantized units than tensors"
        )
    for key, where in (
        ("route_pending_acknowledged", "route-pending acknowledgements"),
        ("excluded_namespaces", "excluded namespaces"),
    ):
        value = payload.get(key)
        if not isinstance(value, list) or value != sorted(
            set(str(item) for item in value)
        ):
            raise CBEndpointValidationError(
                f"plain CB decode {where} are not canonical"
            )
    if payload.get("excluded_namespaces") not in ([], [DSPARK_MTP_SOURCE_PREFIX]):
        # A plain artifact covers everything it ships, with exactly one named
        # exception: the body half of a split DSv4 release omits `mtp.` because
        # the draft is a second artifact with its own receipt. Spelled as a
        # closed set rather than "any exclusion is fine" -- this replay has no
        # disk and no lane, so the only thing keeping it honest is that the one
        # permitted omission is named here.
        raise CBEndpointValidationError(
            "a plain CB artifact cannot exclude a namespace from decode "
            f"coverage other than {DSPARK_MTP_SOURCE_PREFIX!r}: "
            f"{payload.get('excluded_namespaces')!r}"
        )
    if not isinstance(payload.get("requires_moe_backend_marlin"), bool):
        raise CBEndpointValidationError(
            "plain CB decode contract has no boolean marlin-route decision"
        )
    renderer_abi = payload.get("renderer_abi")
    if not isinstance(renderer_abi, str) or not renderer_abi:
        raise CBEndpointValidationError(
            "plain CB decode contract names no renderer ABI"
        )
    codebook_file = payload.get("codebook_file")
    if int(payload["codebook_ref_count"]) > 0:
        if not isinstance(codebook_file, str) or not codebook_file.endswith(
            ".pqcb"
        ):
            raise CBEndpointValidationError(
                "plain CB decode contract references codebooks with no .pqcb"
            )
    elif codebook_file is not None and not isinstance(codebook_file, str):
        raise CBEndpointValidationError(
            "plain CB decode contract codebook file is malformed"
        )
    cover = payload.get("group_cover")
    if not isinstance(cover, list) or not cover:
        raise CBEndpointValidationError(
            "plain CB decode contract covers no config group"
        )
    covered_units = 0
    seen_groups: set[str] = set()
    for entry in cover:
        if not isinstance(entry, Mapping):
            raise CBEndpointValidationError(
                "plain CB decode group cover entry is not an object"
            )
        _require_exact_keys(
            entry,
            {
                "group",
                "format",
                "target_count",
                "unit_count",
                "planes",
                "codebook_refs",
            },
            where="plain CB decode group cover",
        )
        name = entry.get("group")
        if not isinstance(name, str) or not name or name in seen_groups:
            raise CBEndpointValidationError(
                "plain CB decode group cover repeats or omits a group name"
            )
        seen_groups.add(name)
        for key in ("target_count", "unit_count"):
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CBEndpointValidationError(
                    f"{name}: plain CB decode {key} is not a positive integer"
                )
        planes = entry.get("planes")
        if (
            not isinstance(planes, list)
            or not planes
            or planes != sorted(set(str(plane) for plane in planes))
        ):
            raise CBEndpointValidationError(
                f"{name}: plain CB decode plane signature is empty or "
                "not canonical"
            )
        refs = entry.get("codebook_refs")
        if not isinstance(refs, list) or refs != sorted(
            set(str(ref) for ref in refs)
        ):
            raise CBEndpointValidationError(
                f"{name}: plain CB decode codebook refs are not canonical"
            )
        covered_units += int(entry["unit_count"])
    if covered_units < int(payload["cb_unit_count"]):
        # A BOUND, not the equality it used to be. Two per-role groups may
        # cover one packed stack, so this sum is an upper-bound view of a set
        # the replay cannot dedupe: it holds no unit names. What it can still
        # prove is that the cover does not fall SHORT of the classification,
        # i.e. no classified CB unit went uncovered. Exact set equality is
        # asserted where the sets exist, at the assembly site.
        raise CBEndpointValidationError(
            "plain CB decode group cover falls short of the classified CB "
            f"units: {covered_units} vs {payload['cb_unit_count']}"
        )
    features = payload.get("required_runtime_features")
    if not isinstance(features, Mapping) or any(
        not isinstance(name, str)
        or not isinstance(version, int)
        or isinstance(version, bool)
        for name, version in features.items()
    ):
        raise CBEndpointValidationError(
            "plain CB decode runtime feature requirements are malformed"
        )
    if payload.get("required_runtime_contract_schema") != (
        DSPARK_CB_RUNTIME_CONTRACT_SCHEMA
    ):
        raise CBEndpointValidationError(
            "plain CB decode contract does not name the Gridbook runtime "
            "contract schema"
        )
    _require_sha256(
        payload.get("completeness_sha256"), where="completeness digest"
    )
    _require_sha256(payload.get("cover_sha256"), where="plain CB cover digest")
    recorded_evidence_sha = _require_sha256(
        payload.get("evidence_sha256"), where="artifact evidence digest"
    )
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256")
    if recorded_evidence_sha != _canonical_json_sha256(unstamped):
        raise CBEndpointValidationError(
            "plain CB decode evidence digest is stale"
        )


def _validate_artifact_decode_record(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") == ARTIFACT_DECODE_CONTRACT_SCHEMA_PLAIN:
        _validate_cb_plain_decode_record(payload)
        return
    if payload.get("schema") == ARTIFACT_DECODE_CONTRACT_SCHEMA_V2:
        _validate_dspark_cb_sidecar_decode_record(payload)
        return
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
    lane: str = CB_LANE_DSV4_FLASH,
    model_sha: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    artifact_decode: Mapping[str, Any],
    endpoint_smoke: Mapping[str, Any],
    cuda_graph: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the path-independent receipt the shipcard verifier can replay."""
    _validate_artifact_decode_record(artifact_decode)
    pin = _gridbook_runtime_pin()
    _require_artifact_decode_runtime_features(artifact_decode, pin)
    lane_spec = cb_lane_spec(lane)
    served_model = _require_session_model_name(
        manifest.get("served_model_name"), lane=lane
    )
    launch = _canonical_launch_contract(
        manifest.get("launch_argv") or (),
        arm=arm,
        lane=lane,
        expected_served_model=served_model,
        requires_moe_marlin=bool(
            artifact_decode.get("requires_moe_backend_marlin")
        ),
    )
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
        "lane": lane,
        "model_sha": model_sha,
        "artifact_decode": dict(artifact_decode),
        "serve_manifest": {
            "sha256": manifest_sha256,
            "serve_fingerprint": manifest.get("serve_fingerprint"),
            "artifact_binding": manifest.get("artifact_binding"),
        },
        "stack": {
            "image": lane_spec["image"],
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
        lane=lane,
        model_sha=model_sha,
        serve_fingerprint=str(manifest.get("serve_fingerprint")),
    )
    return contract


def validate_endpoint_contract_record(
    payload: Mapping[str, Any],
    *,
    arm: str,
    lane: str | None = None,
    model_sha: str | None = None,
    serve_fingerprint: str | None = None,
) -> None:
    """Replay every structural assertion encoded in a native CB receipt.

    `lane` is the caller's INDEPENDENT determination -- publication re-derives
    it from the artifact's own `config.json` (`cb_serving_lane`).  The receipt
    also names its lane, and the two must agree: a receipt may declare which
    contract it was built against, but it may not choose which contract it is
    judged by.  `None` means "no independent opinion" and checks only that the
    declared lane is one this validator knows.
    """
    if arm not in {"eager", "graph"}:
        raise CBEndpointValidationError(f"invalid endpoint-contract arm {arm!r}")
    declared_lane = payload.get("lane")
    if not isinstance(declared_lane, str):
        raise CBEndpointValidationError("endpoint contract declares no serving lane")
    lane_spec = cb_lane_spec(declared_lane)
    if lane is not None and declared_lane != lane:
        raise CBEndpointValidationError(
            f"endpoint contract declares lane {declared_lane!r}, but the "
            f"artifact on disk is on lane {lane!r}"
        )
    expected_keys = {
        "schema",
        "arm",
        "lane",
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
    _require_artifact_decode_runtime_features(artifact_decode, pin)
    expected_stack = {
        "image": lane_spec["image"],
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
        or _SESSION_MODEL_RES[declared_lane].fullmatch(served_model) is None
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
        lane=declared_lane,
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
        lane=cb_serving_lane(model_dir),
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
        lane = cb_serving_lane(model_dir)
        metrics["serving_lane"] = lane
        quant_config = validate_cb_artifact(model_dir)
        artifact_decode = validate_cb_artifact_decode_contract(
            model_dir, quant_config, runtime_pin=_gridbook_runtime_pin()
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
            lane=lane,
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
            lane=lane,
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
