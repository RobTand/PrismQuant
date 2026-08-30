#!/usr/bin/env python3
"""Physical RTX 4090 release gate for the strict Qwen3.8 FP8-CB lane.

This validator is intentionally separate from the historical DSpark endpoint
gate.  It consumes a live server-side fingerprint, an explicitly selected
Gridbook release pin/contract, a fresh vLLM compile log, and two deterministic
endpoint responses.  It never imports Gridbook or vLLM.  The resulting compact
contract is replayable from ``shipcard.json`` and remains blocking until the
same Gridbook release is the repository's immutable tracked pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .cb_layout import FP8_PRODUCT_RUNGS
from .gridbook_environment import (
    CANONICAL_GOLD_ENVIRONMENT,
)
from .rtx4090_graph_contract import (
    RTX4090_COMPILATION_BACKEND,
    RTX4090_COMPILATION_MODE,
    RTX4090_CUDAGRAPH_CAPTURE_SIZES,
    RTX4090_CUDAGRAPH_MODE,
    RTX4090_MAX_MODEL_LEN,
    RTX4090_MAX_NUM_SEQS,
    compilation_config_json,
    validate_compile_cache_preflight,
    validate_rtx4090_graph_log,
)
from .rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES,
    RTX4090_QWEN38_POLICY_ID,
    RTX4090_QWEN38_POLICY_SCHEMA,
    is_rtx4090_validation_only_policy,
    validate_qwen38_dense_config,
    validate_rtx4090_quant_config_manifest,
)
from .shipcard import (
    _manifest_gridbook_runtime_pin,
    _released_gridbook_runtime_pin,
    _verify_gridbook_distribution_identity,
    compute_model_sha,
    fill_slot,
    git_provenance,
    load_shipcard,
    make_record,
)
from .validate_cb_endpoint import (
    CBEndpointValidationError,
    ENDPOINT_SESSION_SCHEMA,
    _canonical_launch_contract,
    _validate_closed_server_environment,
    _validate_live_server_session,
    run_endpoint_smoke,
)


RTX4090_FP8_CB_CONTRACT_SCHEMA = "prismaquant.rtx4090_fp8_cb_endpoint.v1"
RTX4090_FP8_CB_SLOT = "rtx4090.fp8_cb"
RTX4090_FP8_CB_TOOL = "validate_rtx4090_fp8_cb.py"
RTX4090_GPU_NAME = "NVIDIA GeForce RTX 4090"
RTX4090_COMPUTE_CAPABILITY = (8, 9)
RTX4090_KV_CACHE_DTYPE = "fp8"
RTX4090_KV_CACHE_BYTES = 4 * 1024**3
RTX4090_MAX_NUM_BATCHED_TOKENS = 32768
RTX4090_TENSOR_PARALLEL_SIZE = 1
RTX4090_GPU_MEMORY_UTILIZATION = "0.95"
RTX4090_GRIDBOOK_REPOSITORY = "https://github.com/RobTand/gridbook.git"
RTX4090_SERVING_PIN_SCHEMA = "prismaquant.gridbook_serving_runtime_pin.v1"
RTX4090_VLLM_REPOSITORY = "https://github.com/vllm-project/vllm.git"
RTX4090_VLLM_RUNTIME_PIN_SCHEMA = "prismaquant.vllm_runtime_pin.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RELEASE_VERSION_RE = re.compile(r"[0-9]+[.][0-9]+[.][0-9]+")
_IMAGE_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_MAIN_EXTENSION_RE = re.compile(
    r"^prismaquant_cb(?:_v2)?_ext(?:[.][^/]*)?[.]so$"
)
_SERVED_MODEL_RE = re.compile(
    r"qwen38-rtx4090-(?P<model>[0-9a-f]{32})-(?P<nonce>[0-9a-f]{32})"
)


class RTX4090FP8CBValidationError(RuntimeError):
    """The physical Ada serving receipt is absent, stale, or contradictory."""


def _validate_served_model_name(name: str, *, model_sha: str) -> str:
    match = _SERVED_MODEL_RE.fullmatch(str(name))
    if (
        match is None
        or _SHA256_RE.fullmatch(str(model_sha)) is None
        or match.group("model") != str(model_sha)[:32]
    ):
        raise RTX4090FP8CBValidationError(
            "served model name must bind the artifact digest and one fresh "
            "128-bit session nonce"
        )
    return match.group("nonce")


def _canonical_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RTX4090FP8CBValidationError(
            "receipt contains a non-canonical JSON value"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise RTX4090FP8CBValidationError(
            f"required evidence file cannot be read: {source}"
        ) from exc
    return {"bytes": size, "sha256": digest.hexdigest()}


def _load_json(path: str | Path, *, where: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    try:
        raw = source.read_bytes()

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key!r}")
                result[key] = value
            return result

        payload = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RTX4090FP8CBValidationError(
            f"{where} is not strict UTF-8 JSON: {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RTX4090FP8CBValidationError(f"{where} must be a JSON object")
    return payload, raw


def rtx4090_artifact_content_expectations(
    model_dir: str | Path,
) -> dict[str, Any]:
    """Load the strict content ledgers without opening a weight container."""

    root = Path(model_dir)
    quant, _raw = _load_json(
        root / "quant_config.json", where="RTX4090 quantization manifest"
    )
    provenance = quant.get("provenance")
    weight_manifest = provenance.get("weight_content_manifest") if isinstance(
        provenance, Mapping
    ) else None
    tensor_identity = provenance.get("tensor_payload_identity") if isinstance(
        provenance, Mapping
    ) else None
    tensor_ledger = tensor_identity.get("tensor_sha256") if isinstance(
        tensor_identity, Mapping
    ) else None
    files = weight_manifest.get("files") if isinstance(
        weight_manifest, Mapping
    ) else None
    if (
        not isinstance(weight_manifest, Mapping)
        or set(weight_manifest) != {"schema", "algorithm", "files"}
        or weight_manifest.get("schema")
        != "prismaquant.weight_content_manifest/1"
        or weight_manifest.get("algorithm") != "sha256"
        or not isinstance(files, Mapping)
        or not files
        or not isinstance(tensor_ledger, Mapping)
        or not tensor_ledger
    ):
        raise RTX4090FP8CBValidationError(
            "strict artifact has no closed weight/tensor content ledgers"
        )
    normalized_files: dict[str, dict[str, Any]] = {}
    for name, row in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or row.get("bytes", -1) < 0
            or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
        ):
            raise RTX4090FP8CBValidationError(
                f"invalid strict weight content row {name!r}"
            )
        normalized_files[name] = {
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
    normalized_tensors: dict[str, str] = {}
    for name, digest in tensor_ledger.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise RTX4090FP8CBValidationError(
                f"invalid strict tensor content row {name!r}"
            )
        normalized_tensors[name] = digest

    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.exists():
        if single_path.exists():
            raise RTX4090FP8CBValidationError(
                "strict artifact cannot carry both indexed and single-file weights"
            )
        index, _index_raw = _load_json(
            index_path, where="RTX4090 safetensors index"
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RTX4090FP8CBValidationError(
                "strict safetensors index has no weight_map"
            )
        tensor_to_file = {
            str(name): str(filename) for name, filename in weight_map.items()
        }
    else:
        if set(normalized_files) != {single_path.name}:
            raise RTX4090FP8CBValidationError(
                "strict artifact has no exact single-file or indexed layout"
            )
        tensor_to_file = {
            name: single_path.name for name in normalized_tensors
        }
    if (
        set(tensor_to_file) != set(normalized_tensors)
        or set(tensor_to_file.values()) != set(normalized_files)
        or any(
            Path(filename).name != filename
            or not filename.endswith(".safetensors")
            for filename in tensor_to_file.values()
        )
    ):
        raise RTX4090FP8CBValidationError(
            "strict safetensors index differs from the content ledgers"
        )
    return {
        "weight_manifest": {
            "schema": "prismaquant.weight_content_manifest/1",
            "algorithm": "sha256",
            "files": dict(sorted(normalized_files.items())),
        },
        "tensor_sha256": dict(sorted(normalized_tensors.items())),
        "tensor_to_file": dict(sorted(tensor_to_file.items())),
    }


def _validate_rtx4090_content_receipt_shape(
    receipt: object,
    *,
    expectations: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Replay one-pass accounting and optional portable content ledgers."""

    required = {
        "schema", "source", "content_read_passes", "content_bytes_read",
        "read_calls", "root", "files",
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != required
        or receipt.get("schema")
        != "prismaquant.safetensors_content_receipt/1"
        or receipt.get("source") != "verified_read"
        or receipt.get("content_read_passes") != 1
        or type(receipt.get("content_bytes_read")) is not int
        or receipt.get("content_bytes_read", 0) <= 0
        or type(receipt.get("read_calls")) is not int
        or receipt.get("read_calls", 0) <= 0
    ):
        raise RTX4090FP8CBValidationError(
            "strict artifact content receipt is missing or not one verified pass"
        )
    root_record = receipt.get("root")
    rows = receipt.get("files")
    if (
        not isinstance(root_record, Mapping)
        or set(root_record) != {"device", "inode"}
        or any(type(root_record.get(key)) is not int for key in root_record)
        or root_record.get("device", -1) < 0
        or root_record.get("inode", 0) <= 0
        or not isinstance(rows, Mapping)
        or not rows
    ):
        raise RTX4090FP8CBValidationError(
            "strict artifact content receipt has invalid filesystem scope"
        )
    expected_manifest = expectations.get("weight_manifest") if isinstance(
        expectations, Mapping
    ) else None
    expected_files = expected_manifest.get("files") if isinstance(
        expected_manifest, Mapping
    ) else None
    expected_tensors = expectations.get("tensor_sha256") if isinstance(
        expectations, Mapping
    ) else None
    expected_map = expectations.get("tensor_to_file") if isinstance(
        expectations, Mapping
    ) else None
    if expected_files is not None and set(rows) != set(expected_files):
        raise RTX4090FP8CBValidationError(
            "strict artifact content receipt file set differs from its manifest"
        )
    observed_tensors: dict[str, str] = {}
    observed_map: dict[str, str] = {}
    observed_bytes = 0
    for name, row in rows.items():
        row_stat = row.get("stat") if isinstance(row, Mapping) else None
        row_tensors = row.get("tensor_sha256") if isinstance(
            row, Mapping
        ) else None
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or not isinstance(row, Mapping)
            or set(row) != {"stat", "sha256", "tensor_sha256"}
            or not isinstance(row_stat, Mapping)
            or set(row_stat) != {
                "device", "inode", "bytes", "mtime_ns", "ctime_ns"
            }
            or any(type(row_stat.get(key)) is not int for key in row_stat)
            or row_stat.get("device", -1) < 0
            or row_stat.get("inode", 0) <= 0
            or row_stat.get("bytes", -1) < 0
            or row_stat.get("mtime_ns", -1) < 0
            or row_stat.get("ctime_ns", -1) < 0
            or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
            or not isinstance(row_tensors, Mapping)
            or not row_tensors
        ):
            raise RTX4090FP8CBValidationError(
                f"strict artifact content receipt row is invalid: {name!r}"
            )
        if expected_files is not None and (
            row.get("sha256") != expected_files[name].get("sha256")
            or row_stat.get("bytes") != expected_files[name].get("bytes")
        ):
            raise RTX4090FP8CBValidationError(
                f"strict artifact content receipt differs for {name!r}"
            )
        observed_bytes += int(row_stat["bytes"])
        for tensor_name, digest in row_tensors.items():
            if (
                not isinstance(tensor_name, str)
                or not tensor_name
                or tensor_name in observed_tensors
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
            ):
                raise RTX4090FP8CBValidationError(
                    "strict artifact content receipt tensor ledger is invalid"
                )
            observed_tensors[tensor_name] = digest
            observed_map[tensor_name] = name
    if receipt.get("content_bytes_read") != observed_bytes:
        raise RTX4090FP8CBValidationError(
            "strict artifact content receipt did not read every container byte once"
        )
    if expected_tensors is not None and (
        dict(sorted(observed_tensors.items())) != dict(expected_tensors)
        or dict(sorted(observed_map.items())) != dict(expected_map)
    ):
        raise RTX4090FP8CBValidationError(
            "strict artifact content receipt tensor ledger/map differs"
        )
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def create_rtx4090_artifact_content_receipt(
    model_dir: str | Path,
) -> dict[str, Any]:
    """Read every finalized safetensors byte exactly once and bind its stat."""

    expectations = rtx4090_artifact_content_expectations(model_dir)
    from .shipcard import verify_safetensors_content_once

    try:
        receipt = verify_safetensors_content_once(
            model_dir,
            expected_weight_manifest=expectations["weight_manifest"],
            expected_tensor_sha256=expectations["tensor_sha256"],
            expected_tensor_to_file=expectations["tensor_to_file"],
        )
    except (OSError, ValueError) as exc:
        raise RTX4090FP8CBValidationError(str(exc)) from exc
    return _validate_rtx4090_content_receipt_shape(
        receipt, expectations=expectations
    )


def validate_rtx4090_artifact_content_receipt_portable(
    model_dir: str | Path,
    receipt: object,
) -> dict[str, Any]:
    """Validate content identities without reopening any weight container."""

    return _validate_rtx4090_content_receipt_shape(
        receipt,
        expectations=rtx4090_artifact_content_expectations(model_dir),
    )


def validate_rtx4090_artifact_content_receipt(
    model_dir: str | Path,
    receipt: object,
) -> dict[str, Any]:
    """Recheck the portable receipt and live stats, never weight content."""

    normalized = validate_rtx4090_artifact_content_receipt_portable(
        model_dir, receipt
    )
    expectations = rtx4090_artifact_content_expectations(model_dir)
    from .shipcard import validate_safetensors_content_receipt

    try:
        validate_safetensors_content_receipt(
            model_dir,
            normalized,
            expected_weight_manifest=expectations["weight_manifest"],
            expected_tensor_sha256=expectations["tensor_sha256"],
            expected_tensor_to_file=expectations["tensor_to_file"],
        )
    except (OSError, ValueError) as exc:
        raise RTX4090FP8CBValidationError(str(exc)) from exc
    return normalized


def validate_candidate_runtime_pin(
    payload: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a not-yet-tracked release pin without blessing a pending pin."""

    required = {
        "schema",
        "repository",
        "commit",
        "version",
        "version_is_release",
        "wheel_sha256",
        "runtime_contract_schema",
        "required_abi_features",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RTX4090FP8CBValidationError(
            "candidate Gridbook serving pin is not a closed v1 pin"
        )
    if (
        payload.get("schema") != RTX4090_SERVING_PIN_SCHEMA
        or payload.get("repository") != RTX4090_GRIDBOOK_REPOSITORY
        or not isinstance(payload.get("commit"), str)
        or _COMMIT_RE.fullmatch(str(payload.get("commit"))) is None
        or not isinstance(payload.get("version"), str)
        or _RELEASE_VERSION_RE.fullmatch(str(payload.get("version"))) is None
        or payload.get("version_is_release") is not True
        or not isinstance(payload.get("wheel_sha256"), str)
        or _SHA256_RE.fullmatch(str(payload.get("wheel_sha256"))) is None
    ):
        raise RTX4090FP8CBValidationError(
            "candidate Gridbook pin must name one immutable semantic release"
        )
    schema = runtime_contract.get("schema")
    abi = runtime_contract.get("abi_features")
    if (
        schema != "gridbook.runtime-contract.v11"
        or payload.get("runtime_contract_schema") != schema
        or not isinstance(abi, Mapping)
        or not abi
        or any(
            not isinstance(name, str)
            or not name
            or type(value) is not int
            or value <= 0
            for name, value in abi.items()
        )
        or payload.get("required_abi_features") != abi
    ):
        raise RTX4090FP8CBValidationError(
            "candidate pin does not exactly bind the v11 runtime ABI"
        )
    return {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in payload.items()
    }


def validate_candidate_vllm_runtime_pin(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one exact official vLLM VCS install plus RECORD identity."""

    required = {
        "schema",
        "repository",
        "commit",
        "version",
        "record_sha256",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != RTX4090_VLLM_RUNTIME_PIN_SCHEMA
        or payload.get("repository") != RTX4090_VLLM_REPOSITORY
        or _COMMIT_RE.fullmatch(str(payload.get("commit", ""))) is None
        or not isinstance(payload.get("version"), str)
        or not payload.get("version")
        or _SHA256_RE.fullmatch(str(payload.get("record_sha256", ""))) is None
    ):
        raise RTX4090FP8CBValidationError(
            "candidate vLLM pin must be the closed official VCS/RECORD identity"
        )
    return {key: payload[key] for key in sorted(required)}


def rtx4090_serve_environment(runtime_pin: Mapping[str, Any]) -> dict[str, str]:
    """Exact SM89-safe Gridbook environment captured from every server PID."""

    expected = {
        name: str(value)
        for name, value in CANONICAL_GOLD_ENVIRONMENT.items()
        if value is not None
    }
    # The historical GB10 gold profile enables the SM12x fused mid-M module
    # and preloads every fused family.  Both are invalid on Ada.  The strict
    # lane deliberately selects Gridbook's native GEMV / expand+CUTLASS routes.
    expected.update({
        "PRISMAQUANT_CB_FUSED_MIDM": "0",
        "PRISMAQUANT_PRELOAD_FUSED": "0",
        "PRISMAQUANT_CB_EXT_DIR": "/opt/gridbook/ext-cache",
        "PQ_GRIDBOOK_RUNTIME_COMMIT": str(runtime_pin["commit"]),
        "PQ_GRIDBOOK_RUNTIME_VERSION": str(runtime_pin["version"]),
        "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256": str(runtime_pin["wheel_sha256"]),
        "PYTHONSAFEPATH": "1",
    })
    return dict(sorted(expected.items()))


def rtx4090_serve_environment_allowlist() -> tuple[str, ...]:
    # Keep the validator and the stdlib-only in-container collector on one
    # exact projection.  In particular this includes v11's two additional
    # source-level selectors, both of which must be absent on this FP8 lane.
    from tools.serve_fingerprint import RTX4090_SERVER_ENV_ALLOWLIST

    return tuple(RTX4090_SERVER_ENV_ALLOWLIST)


def rtx4090_launch_options(
    *,
    arm: str,
    served_model: str,
    compile_cache: str | None = None,
) -> dict[str, str]:
    if arm not in {"eager", "graph"}:
        raise RTX4090FP8CBValidationError(f"unknown serving arm {arm!r}")
    options = {
        "--served-model-name": served_model,
        "--host": "0.0.0.0",
        "--port": "8000",
        "--tokenizer-mode": "auto",
        "--generation-config": "vllm",
        "--quantization": "gridbook",
        "--tensor-parallel-size": str(RTX4090_TENSOR_PARALLEL_SIZE),
        "--kv-cache-dtype": RTX4090_KV_CACHE_DTYPE,
        "--kv-cache-memory-bytes": str(RTX4090_KV_CACHE_BYTES),
        "--max-model-len": str(RTX4090_MAX_MODEL_LEN),
        "--max-num-seqs": str(RTX4090_MAX_NUM_SEQS),
        "--max-num-batched-tokens": str(RTX4090_MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization": RTX4090_GPU_MEMORY_UTILIZATION,
    }
    if arm == "graph":
        if compile_cache is None:
            raise RTX4090FP8CBValidationError(
                "graph arm requires one fresh absolute compile-cache path"
            )
        options["--compilation-config"] = compilation_config_json(
            cache_dir=compile_cache
        )
    elif compile_cache is not None:
        raise RTX4090FP8CBValidationError(
            "eager arm must not carry a graph compile-cache path"
        )
    return options


def rtx4090_launch_switches(*, arm: str) -> frozenset[str]:
    switches = {
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
    }
    if arm == "eager":
        switches.add("--enforce-eager")
    return frozenset(switches)


def _validate_vllm_compilation_provenance(
    payload: object,
    *,
    expected_version: str,
    expected_runtime_pin: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "name",
        "version",
        "distribution_package_root",
        "module_origin",
        "wrapper_path",
        "wrapper_identity",
        "compile_contract",
        "runtime_pin",
        "direct_url",
        "direct_url_path",
        "direct_url_identity",
        "record_path",
        "record_identity",
        "identity_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RTX4090FP8CBValidationError(
            "live vLLM fullgraph provenance is missing or not closed"
        )
    runtime_pin = validate_candidate_vllm_runtime_pin(expected_runtime_pin)
    identity = payload.get("wrapper_identity")
    contract = payload.get("compile_contract")
    direct_identity = payload.get("direct_url_identity")
    record_identity = payload.get("record_identity")
    if (
        payload.get("schema")
        != "prismaquant.vllm_compilation_provenance/1"
        or payload.get("name") != "vllm"
        or payload.get("version") != expected_version
        or expected_version != runtime_pin["version"]
        or payload.get("runtime_pin") != runtime_pin
    ):
        raise RTX4090FP8CBValidationError(
            "installed vLLM identity differs from the exact runtime pin"
        )
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"bytes", "sha256"}
        or type(identity.get("bytes")) is not int
        or identity.get("bytes", 0) <= 0
        or _SHA256_RE.fullmatch(str(identity.get("sha256", ""))) is None
        or contract
        != {
            "direct_torch_compile_calls": 1,
            "fullgraph": True,
            "dynamic": False,
            "backend_explicit": True,
        }
    ):
        raise RTX4090FP8CBValidationError(
            "installed vLLM wrapper does not prove fullgraph=True, "
            "dynamic=False with an explicit backend"
        )
    try:
        from tools.serve_fingerprint import validate_vllm_pep610_direct_url

        validate_vllm_pep610_direct_url(payload.get("direct_url"), runtime_pin)
    except Exception as exc:
        raise RTX4090FP8CBValidationError(str(exc)) from exc
    if (
        not isinstance(direct_identity, Mapping)
        or set(direct_identity) != {"bytes", "sha256"}
        or type(direct_identity.get("bytes")) is not int
        or direct_identity.get("bytes", 0) <= 0
        or _SHA256_RE.fullmatch(
            str(direct_identity.get("sha256", ""))
        ) is None
        or not isinstance(record_identity, Mapping)
        or set(record_identity) != {"bytes", "sha256"}
        or type(record_identity.get("bytes")) is not int
        or record_identity.get("bytes", 0) <= 0
        or record_identity.get("sha256") != runtime_pin["record_sha256"]
    ):
        raise RTX4090FP8CBValidationError(
            "installed vLLM direct_url/RECORD identities are incomplete"
        )
    root = PurePosixPath(str(payload.get("distribution_package_root", "")))
    module = PurePosixPath(str(payload.get("module_origin", "")))
    wrapper = PurePosixPath(str(payload.get("wrapper_path", "")))
    direct = PurePosixPath(str(payload.get("direct_url_path", "")))
    record = PurePosixPath(str(payload.get("record_path", "")))
    if (
        not root.is_absolute()
        or str(root) == "/"
        or ".." in root.parts
        or not module.is_absolute()
        or not wrapper.is_absolute()
        or not direct.is_absolute()
        or not record.is_absolute()
        or module != root / "__init__.py"
        or wrapper != root / "compilation" / "wrapper.py"
        or direct.parent != record.parent
        or direct.parent.parent != root.parent
        or direct.name != "direct_url.json"
        or record.name != "RECORD"
    ):
        raise RTX4090FP8CBValidationError(
            "vLLM compilation source paths escape the installed distribution"
        )
    body = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if payload.get("identity_sha256") != _canonical_sha256(body):
        raise RTX4090FP8CBValidationError(
            "vLLM compilation provenance identity is stale"
        )
    return {key: value for key, value in payload.items()}


def _validate_runtime_attestation(
    payload: object,
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay the complete device-qualified dense SM89 route projection."""

    required = {
        "runtime_contract_schema",
        "runtime_contract_sha256",
        "lane_eligibility_schema",
        "platform",
        "device_capability",
        "family",
        "structure",
        "rungs",
        "regime_routes",
        "requires_serve_flags",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RTX4090FP8CBValidationError(
            "runtime attestation is not the closed v11/lane-v2 projection"
        )
    contract_sha = str(payload.get("runtime_contract_sha256", ""))
    rungs = payload.get("rungs")
    routes = payload.get("regime_routes")
    if (
        payload.get("runtime_contract_schema")
        != "gridbook.runtime-contract.v11"
        or payload.get("lane_eligibility_schema")
        != "gridbook.lane-eligibility.v2"
        or payload.get("platform") != "sm_89"
        or payload.get("device_capability")
        != list(RTX4090_COMPUTE_CAPABILITY)
        or payload.get("family") != "FP8_CB_K"
        or payload.get("structure") != "dense"
        or _SHA256_RE.fullmatch(contract_sha) is None
        or (
            expected_contract_sha256 is not None
            and contract_sha != expected_contract_sha256
        )
        or rungs != list(FP8_PRODUCT_RUNGS)
        or payload.get("requires_serve_flags") != []
        or not isinstance(routes, list)
        or len(routes) != 2 * len(FP8_PRODUCT_RUNGS)
    ):
        raise RTX4090FP8CBValidationError(
            "runtime attestation is not the exact full K4..K48 step-4, "
            "flag-free, device-qualified SM89 dense FP8-CB route projection"
        )
    observed_pairs: set[tuple[int, str]] = set()
    for route in routes:
        if (
            not isinstance(route, Mapping)
            or set(route)
            != {
                "rung",
                "regime",
                "cell_id",
                "route_status",
                "qualification",
                "requires_serve_flags",
            }
            or route.get("rung") not in rungs
            or route.get("regime") not in {"decode", "batch"}
            or not isinstance(route.get("cell_id"), str)
            or not route.get("cell_id")
            or route.get("route_status") not in {
                "backed",
                "backed_with_serve_flag",
            }
            or route.get("qualification") != "device_qualified"
            or route.get("requires_serve_flags") != []
        ):
            raise RTX4090FP8CBValidationError(
                "runtime route row is not backed, flag-free, and device-qualified"
            )
        pair = (int(route["rung"]), str(route["regime"]))
        if pair in observed_pairs:
            raise RTX4090FP8CBValidationError(
                "runtime attestation repeats a rung/regime route"
            )
        observed_pairs.add(pair)
    expected_pairs = {
        (int(rung), regime)
        for rung in FP8_PRODUCT_RUNGS
        for regime in ("decode", "batch")
    }
    if observed_pairs != expected_pairs:
        raise RTX4090FP8CBValidationError(
            "runtime attestation does not cover every rung in both regimes"
        )
    return {key: value for key, value in payload.items()}


def artifact_runtime_attestation(model_dir: str | Path) -> dict[str, Any]:
    try:
        quant = json.loads(
            (Path(model_dir) / "quant_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090FP8CBValidationError(
            f"artifact producer policy cannot be read: {exc}"
        ) from exc
    provenance = quant.get("provenance") if isinstance(quant, Mapping) else None
    policy = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    attestation = policy.get("runtime_attestation") if isinstance(
        policy, Mapping
    ) else None
    return _validate_runtime_attestation(attestation)


def _validate_rtx4090_artifact(
    model_dir: str | Path,
    *,
    runtime_contract: Mapping[str, Any],
    expected_model_sha: str | None = None,
    finalized_census: bool,
    artifact_content_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(model_dir)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        quant_config = json.loads(
            (root / "quant_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090FP8CBValidationError(
            f"strict RTX4090 artifact JSON cannot be read: {exc}"
        ) from exc
    try:
        validate_qwen38_dense_config(config)
        provenance = quant_config.get("provenance") if isinstance(
            quant_config, Mapping
        ) else None
        policy_stamp = provenance.get("producer_policy") if isinstance(
            provenance, Mapping
        ) else None
        if is_rtx4090_validation_only_policy(policy_stamp):
            raise RTX4090FP8CBValidationError(
                "UNRELEASABLE_VALIDATION_ONLY artifact is categorically "
                "ineligible for the physical RTX4090 strict validator"
            )
        policy = validate_rtx4090_quant_config_manifest(
            quant_config,
            runtime_contract=runtime_contract,
            artifact_dir=(root if finalized_census else None),
            artifact_content_receipt=artifact_content_receipt,
        )
        from tools.serve_fingerprint import artifact_binding

        binding = artifact_binding(root, launch_model="/model")
    except Exception as exc:
        raise RTX4090FP8CBValidationError(
            f"strict RTX4090 artifact validation failed: {exc}"
        ) from exc
    if expected_model_sha is not None and binding.get(
        "model_sha"
    ) != expected_model_sha:
        raise RTX4090FP8CBValidationError(
            "artifact model_sha differs from the opened shipcard"
        )
    artifact_bytes = binding.get("artifact_bytes")
    if (
        type(artifact_bytes) is not int
        or artifact_bytes <= 0
        or artifact_bytes > RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES
        or artifact_bytes != policy.get("artifact_bytes")
    ):
        raise RTX4090FP8CBValidationError(
            "artifact inventory is not one positive whole-directory byte count "
            "at or below 18,000,000,000"
        )
    return dict(binding)


def validate_rtx4090_artifact_metadata(
    model_dir: str | Path,
    *,
    runtime_contract: Mapping[str, Any],
    expected_model_sha: str | None = None,
) -> dict[str, Any]:
    """Validate strict metadata/inventory without reading weight content."""

    return _validate_rtx4090_artifact(
        model_dir,
        runtime_contract=runtime_contract,
        expected_model_sha=expected_model_sha,
        finalized_census=False,
    )


def validate_rtx4090_artifact(
    model_dir: str | Path,
    *,
    runtime_contract: Mapping[str, Any],
    expected_model_sha: str | None = None,
    artifact_content_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the finalized census, reusing a stat-bound content receipt."""

    return _validate_rtx4090_artifact(
        model_dir,
        runtime_contract=runtime_contract,
        expected_model_sha=expected_model_sha,
        finalized_census=True,
        artifact_content_receipt=artifact_content_receipt,
    )


def validate_rtx4090_serve_manifest(
    payload: Mapping[str, Any],
    *,
    arm: str,
    expected_image: str,
    expected_served_model: str,
    expected_model_sha: str,
    expected_artifact_binding: Mapping[str, Any],
    expected_artifact_content_receipt: Mapping[str, Any],
    runtime_pin: Mapping[str, Any],
    vllm_runtime_pin: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    expected_runtime_attestation: Mapping[str, Any],
    runtime_contract_file_identity: Mapping[str, Any],
    compile_cache: str | None = None,
) -> dict[str, Any]:
    """Validate one live physical-4090 server and return receipt ingredients."""

    if _IMAGE_RE.fullmatch(expected_image) is None:
        raise RTX4090FP8CBValidationError(
            "expected serving image must be immutable name@sha256:<digest>"
        )
    _validate_served_model_name(
        expected_served_model, model_sha=expected_model_sha
    )
    if expected_artifact_binding.get("model_sha") != expected_model_sha:
        raise RTX4090FP8CBValidationError(
            "validated artifact binding differs from the opened shipcard"
        )
    if payload.get("artifact_content_receipt") != dict(
        expected_artifact_content_receipt
    ):
        raise RTX4090FP8CBValidationError(
            "serve manifest differs from the one-pass artifact content receipt"
        )
    try:
        from tools.serve_fingerprint import (
            MANIFEST_SCHEMA,
            elide_argv_paths,
            fingerprint,
        )

        if payload.get("schema") != MANIFEST_SCHEMA:
            raise RTX4090FP8CBValidationError(
                f"unsupported serve-manifest schema {payload.get('schema')!r}"
            )
        recorded_fingerprint = payload.get("serve_fingerprint")
        if (
            not isinstance(recorded_fingerprint, str)
            or _SHA256_RE.fullmatch(recorded_fingerprint) is None
            or recorded_fingerprint != fingerprint(payload)
        ):
            raise RTX4090FP8CBValidationError(
                "serve manifest fingerprint is missing or stale"
            )
        if payload.get("source") != "server" or payload.get(
            "attestation_phase"
        ) != "post":
            raise RTX4090FP8CBValidationError(
                "receipt requires a post-run server-side manifest"
            )
        if payload.get("residency_readable") is not True:
            raise RTX4090FP8CBValidationError(
                "serve manifest could not inspect every server process"
            )
        if payload.get("image") != expected_image:
            raise RTX4090FP8CBValidationError(
                "serve manifest image differs from the immutable candidate"
            )
        if payload.get("model") != "/model" or payload.get(
            "served_model_name"
        ) != expected_served_model:
            raise RTX4090FP8CBValidationError(
                "serve manifest does not identify the requested /model session"
            )
        if (
            payload.get("gpu_name") != RTX4090_GPU_NAME
            or payload.get("gpu_count") != 1
            or payload.get("compute_capability")
            != list(RTX4090_COMPUTE_CAPABILITY)
            or payload.get("gpu_compute_capabilities")
            != [list(RTX4090_COMPUTE_CAPABILITY)]
        ):
            raise RTX4090FP8CBValidationError(
                "serve did not run on exactly one physical RTX 4090 (SM89)"
            )
        observed_binding = payload.get("artifact_binding")
        binding_keys = {
            "schema",
            "launch_model",
            "model_sha",
            "artifact_inventory_sha256",
            "artifact_bytes",
        }
        expected_binding = {
            key: expected_artifact_binding.get(key) for key in binding_keys
        }
        observed_binding_projection = {
            key: observed_binding.get(key) for key in binding_keys
        } if isinstance(observed_binding, Mapping) else None
        binding_differs = not isinstance(observed_binding, Mapping)
        if isinstance(observed_binding, Mapping):
            binding_differs = (
                observed_binding_projection != expected_binding
                or observed_binding.get("resolved_path") != "/model"
            )
        if binding_differs:
            raise RTX4090FP8CBValidationError(
                "serve manifest artifact binding differs from the validated artifact"
            )
        expected_eager = arm == "eager"
        if (
            payload.get("enforce_eager") is not expected_eager
            or payload.get("quantization") != "gridbook"
            or payload.get("kv_cache_dtype") != RTX4090_KV_CACHE_DTYPE
            or payload.get("speculative_config") is not None
        ):
            raise RTX4090FP8CBValidationError(
                "serve manifest eager/quantization/KV/no-spec posture differs"
            )
        launch = _canonical_launch_contract(
            payload.get("launch_argv") or (),
            arm=arm,
            lane="gridbook_generic",
            expected_served_model=expected_served_model,
            requires_moe_marlin=False,
            expected_options_override=rtx4090_launch_options(
                arm=arm,
                served_model=expected_served_model,
                compile_cache=compile_cache,
            ),
            expected_switches_override=rtx4090_launch_switches(arm=arm),
        )
        argv = [str(value) for value in payload.get("launch_argv") or ()]
        if payload.get("launch_flags") != elide_argv_paths(argv):
            raise RTX4090FP8CBValidationError(
                "serve manifest launch_flags are not canonical"
            )
        extensions = payload.get("resident_extensions")
        if (
            not isinstance(extensions, list)
            or extensions != sorted(set(extensions))
            or not any(
                isinstance(name, str)
                and _MAIN_EXTENSION_RE.fullmatch(name) is not None
                for name in extensions
            )
        ):
            raise RTX4090FP8CBValidationError(
                "serve has no resident native Gridbook CB extension"
            )
        observed_pin = _manifest_gridbook_runtime_pin(payload, runtime_pin)
        if observed_pin is None:
            raise RTX4090FP8CBValidationError(
                "live Gridbook runtime differs from the candidate release pin"
            )
        distribution_problems = _verify_gridbook_distribution_identity(
            RTX4090_FP8_CB_SLOT,
            payload,
            runtime_pin,
            canonical_sha=_canonical_sha256,
        )
        if distribution_problems:
            raise RTX4090FP8CBValidationError(distribution_problems[0])
        distribution = payload.get("gridbook_distribution")
        source_files = distribution.get("source_files") if isinstance(
            distribution, Mapping
        ) else None
        contract_source = source_files.get(
            "gridbook/runtime_contract.json"
        ) if isinstance(source_files, Mapping) else None
        if contract_source != dict(runtime_contract_file_identity):
            raise RTX4090FP8CBValidationError(
                "installed Gridbook runtime_contract.json differs from candidate bytes"
            )
        environment = _validate_closed_server_environment(
            payload,
            runtime_pin=runtime_pin,
            expected_environment=rtx4090_serve_environment(runtime_pin),
            expected_allowlist=rtx4090_serve_environment_allowlist(),
        )
        session = _validate_live_server_session(
            payload, expected_served_model=expected_served_model
        )
        model_identity = session.get("models_endpoint_binding", {}).get(
            "model"
        )
        if (
            not isinstance(model_identity, Mapping)
            or model_identity.get("max_model_len") != RTX4090_MAX_MODEL_LEN
        ):
            raise RTX4090FP8CBValidationError(
                "live /v1/models identity does not resolve max_model_len=32768"
            )
        packages = payload.get("package_versions")
        if (
            not isinstance(packages, Mapping)
            or packages.get("gridbook") != runtime_pin.get("version")
            or not isinstance(packages.get("vllm"), str)
            or not packages.get("vllm")
        ):
            raise RTX4090FP8CBValidationError(
                "installed Gridbook/vLLM package versions are incomplete"
            )
        vllm = _validate_vllm_compilation_provenance(
            payload.get("vllm_compilation_provenance"),
            expected_version=str(packages["vllm"]),
            expected_runtime_pin=vllm_runtime_pin,
        )
    except RTX4090FP8CBValidationError:
        raise
    except (CBEndpointValidationError, KeyError, TypeError, ValueError) as exc:
        raise RTX4090FP8CBValidationError(str(exc)) from exc

    contract_sha = _canonical_sha256(runtime_contract)
    policy_stamp = _validate_runtime_attestation(
        expected_runtime_attestation,
        expected_contract_sha256=contract_sha,
    )
    return {
        "serve_fingerprint": recorded_fingerprint,
        "gpu": {
            "name": RTX4090_GPU_NAME,
            "uuid": payload.get("gpu_uuid"),
            "count": 1,
            "compute_capability": list(RTX4090_COMPUTE_CAPABILITY),
            "driver_version": payload.get("driver_version"),
        },
        "launch": launch,
        "environment": environment,
        "session": session,
        "runtime_pin": dict(runtime_pin),
        "artifact_content_receipt": dict(
            expected_artifact_content_receipt
        ),
        "vllm_runtime_pin": validate_candidate_vllm_runtime_pin(
            vllm_runtime_pin
        ),
        "runtime_attestation": policy_stamp,
        "gridbook_distribution": dict(distribution),
        "resident_extensions": list(extensions),
        "packages": {
            "gridbook": packages["gridbook"],
            "vllm": packages["vllm"],
        },
        "vllm_compilation_provenance": vllm,
    }


def build_rtx4090_endpoint_contract(
    *,
    arm: str,
    artifact_binding: Mapping[str, Any],
    expected_image: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    validated_manifest: Mapping[str, Any],
    runtime_contract_file_identity: Mapping[str, Any],
    endpoint_smoke: Mapping[str, Any],
    graph_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if arm == "graph" and not isinstance(graph_receipt, Mapping):
        raise RTX4090FP8CBValidationError(
            "graph endpoint contract requires positive compile/capture evidence"
        )
    if arm == "eager" and graph_receipt is not None:
        raise RTX4090FP8CBValidationError(
            "eager endpoint contract must not carry graph evidence"
        )
    smoke = dict(endpoint_smoke)
    if (
        smoke.get("deterministic_repeats") != 2
        or smoke.get("served_model")
        != manifest.get("served_model_name")
        or smoke.get("models_endpoint_identity")
        != validated_manifest.get("session", {}).get(
            "models_endpoint_binding"
        )
    ):
        raise RTX4090FP8CBValidationError(
            "endpoint smoke is not bound to the attested live session"
        )
    graph = None
    if graph_receipt is not None:
        graph = {
            key: value
            for key, value in graph_receipt.items()
            if key != "serve_log"
        }
    contract = {
        "schema": RTX4090_FP8_CB_CONTRACT_SCHEMA,
        "arm": arm,
        "policy": {
            "schema": RTX4090_QWEN38_POLICY_SCHEMA,
            "id": RTX4090_QWEN38_POLICY_ID,
        },
        "artifact": {
            key: value
            for key, value in artifact_binding.items()
            if key != "resolved_path"
        },
        "artifact_content_receipt": dict(
            validated_manifest["artifact_content_receipt"]
        ),
        "image": expected_image,
        "serve_manifest": {
            "sha256": manifest_sha256,
            "serve_fingerprint": validated_manifest["serve_fingerprint"],
        },
        "gpu": dict(validated_manifest["gpu"]),
        "launch": dict(validated_manifest["launch"]),
        "environment": dict(validated_manifest["environment"]),
        "session": dict(validated_manifest["session"]),
        "runtime_pin": dict(validated_manifest["runtime_pin"]),
        "vllm_runtime_pin": dict(validated_manifest["vllm_runtime_pin"]),
        "runtime_attestation": dict(
            validated_manifest["runtime_attestation"]
        ),
        "runtime_contract_file_identity": dict(
            runtime_contract_file_identity
        ),
        "gridbook_distribution": dict(
            validated_manifest["gridbook_distribution"]
        ),
        "resident_extensions": list(
            validated_manifest["resident_extensions"]
        ),
        "packages": dict(validated_manifest["packages"]),
        "vllm_compilation_provenance": dict(
            validated_manifest["vllm_compilation_provenance"]
        ),
        "endpoint_smoke": smoke,
        "graph": graph,
    }
    contract["identity_sha256"] = _canonical_sha256(contract)
    return contract


def _tracked_pin_dict() -> dict[str, Any]:
    pin = _released_gridbook_runtime_pin()
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


def verify_rtx4090_shipcard_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | Path | None,
) -> list[str]:
    """Replay a compact strict-Ada receipt without trusting ``passed=true``."""

    problems: list[str] = []

    def problem(detail: str) -> None:
        problems.append(f"{slot}: {detail}")

    expected_arm = "eager" if slot == "native_export.eager" else "graph"
    if slot not in {
        "native_export.eager",
        "native_export.graph",
        RTX4090_FP8_CB_SLOT,
    }:
        return [f"{slot}: unsupported RTX4090 receipt slot"]
    if record.get("tool") != RTX4090_FP8_CB_TOOL:
        problem(f"not filled by {RTX4090_FP8_CB_TOOL}")
    metrics = record.get("metrics")
    contract = metrics.get("rtx4090_contract") if isinstance(
        metrics, Mapping
    ) else None
    if not isinstance(contract, Mapping):
        return problems + [f"{slot}: missing structured RTX4090 contract"]
    identity = contract.get("identity_sha256")
    body = {key: value for key, value in contract.items() if key != "identity_sha256"}
    if (
        contract.get("schema") != RTX4090_FP8_CB_CONTRACT_SCHEMA
        or contract.get("arm") != expected_arm
        or not isinstance(identity, str)
        or _SHA256_RE.fullmatch(identity) is None
        or identity != _canonical_sha256(body)
    ):
        problem("endpoint contract schema/arm/self-identity is stale")
    if contract.get("policy") != {
        "schema": RTX4090_QWEN38_POLICY_SCHEMA,
        "id": RTX4090_QWEN38_POLICY_ID,
    }:
        problem("policy identity differs from the strict FP8-only campaign")
    if (
        not isinstance(contract.get("image"), str)
        or _IMAGE_RE.fullmatch(str(contract.get("image"))) is None
    ):
        problem("serving image is not immutable")
    gpu = contract.get("gpu")
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("name") != RTX4090_GPU_NAME
        or gpu.get("count") != 1
        or gpu.get("compute_capability")
        != list(RTX4090_COMPUTE_CAPABILITY)
        or not isinstance(gpu.get("uuid"), str)
        or not str(gpu.get("uuid")).startswith("GPU-")
    ):
        problem("receipt does not identify exactly one physical RTX4090/SM89")
    manifest = contract.get("serve_manifest")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"sha256", "serve_fingerprint"}
        or _SHA256_RE.fullmatch(str(manifest.get("sha256", ""))) is None
        or _SHA256_RE.fullmatch(
            str(manifest.get("serve_fingerprint", ""))
        )
        is None
        or record.get("serve_fingerprint")
        != manifest.get("serve_fingerprint")
    ):
        problem("serve-manifest/fingerprint binding is malformed")
    artifact = contract.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != "prismaquant.served_artifact_binding/1"
        or artifact.get("model_sha") != record.get("model_sha")
        or type(artifact.get("artifact_bytes")) is not int
        or not 0 < artifact.get("artifact_bytes", 0)
        <= RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES
        or _SHA256_RE.fullmatch(
            str(artifact.get("artifact_inventory_sha256", ""))
        )
        is None
    ):
        problem("artifact identity/18GB ceiling binding is malformed")
    content_receipt = contract.get("artifact_content_receipt")
    try:
        if model_dir is None:
            _validate_rtx4090_content_receipt_shape(
                content_receipt, expectations=None
            )
        else:
            validate_rtx4090_artifact_content_receipt_portable(
                model_dir, content_receipt
            )
    except Exception as exc:
        problem(f"artifact one-pass content receipt is invalid — {exc}")
    if model_dir is not None:
        try:
            if compute_model_sha(model_dir) != artifact.get("model_sha"):
                problem("on-disk artifact differs from the physical receipt")
            config = json.loads(
                (Path(model_dir) / "config.json").read_text(encoding="utf-8")
            )
            quant = json.loads(
                (Path(model_dir) / "quant_config.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_qwen38_dense_config(config)
            validate_rtx4090_quant_config_manifest(
                quant, require_policy_stamp=False
            )
            stamp = quant.get("provenance", {}).get("producer_policy")
            attestation = stamp.get("runtime_attestation") if isinstance(
                stamp, Mapping
            ) else None
            if (
                not isinstance(stamp, Mapping)
                or stamp.get("schema") != RTX4090_QWEN38_POLICY_SCHEMA
                or stamp.get("id") != RTX4090_QWEN38_POLICY_ID
                or attestation != contract.get("runtime_attestation")
            ):
                problem("artifact producer stamp differs from served runtime contract")
        except Exception as exc:
            problem(f"on-disk strict artifact replay failed — {exc}")
    runtime_pin = contract.get("runtime_pin")
    try:
        tracked_pin = _tracked_pin_dict()
    except Exception as exc:
        problem(f"tracked Gridbook release pin unavailable — {exc}")
        tracked_pin = None
    if tracked_pin is not None and runtime_pin != tracked_pin:
        problem("served Gridbook release is not the current immutable tracked pin")
    runtime_attestation = contract.get("runtime_attestation")
    runtime_file = contract.get("runtime_contract_file_identity")
    distribution = contract.get("gridbook_distribution")
    source_files = distribution.get("source_files") if isinstance(
        distribution, Mapping
    ) else None
    runtime_source_differs = not isinstance(source_files, Mapping)
    if isinstance(source_files, Mapping):
        runtime_source_differs = (
            source_files.get("gridbook/runtime_contract.json") != runtime_file
        )
    try:
        _validate_runtime_attestation(runtime_attestation)
    except Exception as exc:
        problem(str(exc))
    if not isinstance(runtime_file, Mapping) or runtime_source_differs:
        problem("v11 runtime-contract source/canonical identity is incomplete")
    if isinstance(runtime_pin, Mapping) and isinstance(distribution, Mapping):
        distribution_problems = _verify_gridbook_distribution_identity(
            slot,
            {
                "gridbook_runtime_pin": runtime_pin,
                "gridbook_distribution": distribution,
            },
            runtime_pin,
            canonical_sha=_canonical_sha256,
        )
        problems.extend(distribution_problems)
    packages = contract.get("packages")
    vllm_runtime_pin = contract.get("vllm_runtime_pin")
    if (
        not isinstance(packages, Mapping)
        or not isinstance(runtime_pin, Mapping)
        or packages.get("gridbook") != runtime_pin.get("version")
        or not isinstance(packages.get("vllm"), str)
        or not packages.get("vllm")
        or not isinstance(vllm_runtime_pin, Mapping)
        or packages.get("vllm") != vllm_runtime_pin.get("version")
    ):
        problem("served package versions differ from the release pin")
    try:
        _validate_vllm_compilation_provenance(
            contract.get("vllm_compilation_provenance"),
            expected_version=str(packages.get("vllm", ""))
            if isinstance(packages, Mapping)
            else "",
            expected_runtime_pin=(
                vllm_runtime_pin
                if isinstance(vllm_runtime_pin, Mapping)
                else {}
            ),
        )
    except Exception as exc:
        problem(str(exc))
    launch = contract.get("launch")
    graph = contract.get("graph")
    smoke_for_launch = contract.get("endpoint_smoke")
    served_model = (
        str(smoke_for_launch.get("served_model", ""))
        if isinstance(smoke_for_launch, Mapping)
        else ""
    )
    try:
        session_nonce = _validate_served_model_name(
            served_model,
            model_sha=str(artifact.get("model_sha", ""))
            if isinstance(artifact, Mapping)
            else "",
        )
    except Exception as exc:
        problem(str(exc))
        session_nonce = ""
    session = contract.get("session")
    endpoint_identity: Mapping[str, Any] | None = None
    try:
        from tools.serve_fingerprint import models_endpoint_binding_identity

        if (
            not isinstance(session, Mapping)
            or session.get("schema") != ENDPOINT_SESSION_SCHEMA
        ):
            raise RTX4090FP8CBValidationError(
                "endpoint session receipt is missing or malformed"
            )
        endpoint_identity = models_endpoint_binding_identity(
            session.get("models_endpoint_binding")
        )
        endpoint_model = endpoint_identity.get("model")
        if (
            not isinstance(endpoint_model, Mapping)
            or endpoint_model.get("id") != served_model
            or endpoint_model.get("root") != "/model"
            or endpoint_model.get("max_model_len") != RTX4090_MAX_MODEL_LEN
        ):
            raise RTX4090FP8CBValidationError(
                "endpoint session does not bind the nonce-bearing served model"
            )
    except Exception as exc:
        problem(str(exc))
    configured_compile_cache = (
        graph.get("configured_compile_cache_root")
        if expected_arm == "graph" and isinstance(graph, Mapping)
        else None
    )
    try:
        expected_options = rtx4090_launch_options(
            arm=expected_arm,
            served_model=served_model,
            compile_cache=configured_compile_cache,
        )
    except Exception as exc:
        problem(f"graph launch cache binding is malformed — {exc}")
        expected_options = {}
    expected_launch = {
        "model": "/model",
        "served_model_name": served_model,
        "options": dict(sorted(expected_options.items())),
        "switches": sorted(rtx4090_launch_switches(arm=expected_arm)),
        "requires_moe_backend_marlin": False,
    }
    if launch != expected_launch:
        problem("launch contract differs from exact 32K/4GiB/TP1 profile")
    if expected_arm == "graph":
        configured_root = (
            PurePosixPath(str(
                graph.get("configured_compile_cache_root", "")
            ))
            if isinstance(graph, Mapping)
            else PurePosixPath(".")
        )
        observed_cache = (
            PurePosixPath(str(graph.get("compile_cache", "")))
            if isinstance(graph, Mapping)
            else PurePosixPath(".")
        )
        freshness = graph.get("compile_cache_freshness") if isinstance(
            graph, Mapping
        ) else None
        freshness_keys = {
            "schema",
            "session_nonce",
            "configured_container_root",
            "preflight_sha256",
            "directory_device",
            "directory_inode",
            "post_file_count",
            "post_total_bytes",
            "post_tree_sha256",
        }
        freshness_valid = (
            isinstance(freshness, Mapping)
            and set(freshness) == freshness_keys
            and freshness.get("schema")
            == "prismaquant.rtx4090_compile_cache_preflight.v1"
            and freshness.get("session_nonce") == session_nonce
            and freshness.get("configured_container_root")
            == str(configured_root)
            and _SHA256_RE.fullmatch(
                str(freshness.get("preflight_sha256", ""))
            ) is not None
            and type(freshness.get("directory_device")) is int
            and freshness.get("directory_device", -1) >= 0
            and type(freshness.get("directory_inode")) is int
            and freshness.get("directory_inode", 0) > 0
            and type(freshness.get("post_file_count")) is int
            and freshness.get("post_file_count", 0) > 0
            and type(freshness.get("post_total_bytes")) is int
            and freshness.get("post_total_bytes", 0) > 0
            and _SHA256_RE.fullmatch(
                str(freshness.get("post_tree_sha256", ""))
            ) is not None
        )
        if (
            not isinstance(graph, Mapping)
            or graph.get("schema")
            != "prismaquant.rtx4090_graph_contract.v1"
            or graph.get("compilation_mode") != RTX4090_COMPILATION_MODE
            or graph.get("compilation_backend")
            != RTX4090_COMPILATION_BACKEND
            or graph.get("cudagraph_mode") != RTX4090_CUDAGRAPH_MODE
            or graph.get("capture_sizes")
            != list(RTX4090_CUDAGRAPH_CAPTURE_SIZES)
            or graph.get("max_model_len") != RTX4090_MAX_MODEL_LEN
            or not configured_root.is_absolute()
            or ".." in configured_root.parts
            or str(configured_root) in {"/", "."}
            or not observed_cache.is_absolute()
            or ".." in observed_cache.parts
            or (
                observed_cache != configured_root
                and configured_root not in observed_cache.parents
            )
            or graph.get("piecewise_capture_count")
            != len(RTX4090_CUDAGRAPH_CAPTURE_SIZES)
            or graph.get("full_capture_count")
            != len(RTX4090_CUDAGRAPH_CAPTURE_SIZES)
            or _SHA256_RE.fullmatch(
                str(graph.get("serve_log_sha256", ""))
            )
            is None
            or not freshness_valid
        ):
            problem("fullgraph mode-3 compile/CUDA-graph receipt is incomplete")
    elif graph is not None:
        problem("eager receipt unexpectedly carries graph evidence")
    smoke = contract.get("endpoint_smoke")
    if (
        not isinstance(smoke, Mapping)
        or smoke.get("deterministic_repeats") != 2
        or smoke.get("temperature") != 0.0
        or smoke.get("top_p") != 1.0
        or smoke.get("seed") != 0
        or smoke.get("n") != 1
        or smoke.get("stream") is not False
        or type(smoke.get("generated_utf8_bytes")) is not int
        or smoke.get("generated_utf8_bytes", 0) <= 0
        or _SHA256_RE.fullmatch(str(smoke.get("output_sha256", ""))) is None
        or smoke.get("models_endpoint_identity") != endpoint_identity
    ):
        problem("endpoint deterministic generation proof is incomplete")
    return problems


def _record_for_contract(
    *,
    slot: str,
    contract: Mapping[str, Any],
    model_sha: str,
    git_commit: str,
) -> dict[str, Any]:
    return make_record(
        slot=slot,
        tool=RTX4090_FP8_CB_TOOL,
        passed=True,
        model_sha=model_sha,
        metrics={"rtx4090_contract": dict(contract)},
        detail="physical RTX4090 FP8-CB endpoint contract passed",
        spec_decode_detected=False,
        serve_fingerprint=str(
            contract.get("serve_manifest", {}).get("serve_fingerprint")
        ),
        git_commit=git_commit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--shipcard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-pin", type=Path, required=True)
    parser.add_argument("--vllm-runtime-pin", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--arm", choices=("eager", "graph"), required=True)
    parser.add_argument("--serve-log", type=Path)
    parser.add_argument("--compile-cache")
    parser.add_argument("--compile-cache-host-root", type=Path)
    parser.add_argument("--compile-cache-preflight", type=Path)
    parser.add_argument("--session-nonce")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        runtime_contract, _contract_raw = _load_json(
            args.runtime_contract, where="Gridbook runtime contract"
        )
        runtime_pin_payload, _pin_raw = _load_json(
            args.runtime_pin, where="candidate Gridbook serving pin"
        )
        runtime_pin = validate_candidate_runtime_pin(
            runtime_pin_payload, runtime_contract=runtime_contract
        )
        vllm_pin_payload, _vllm_pin_raw = _load_json(
            args.vllm_runtime_pin, where="candidate vLLM runtime pin"
        )
        vllm_runtime_pin = validate_candidate_vllm_runtime_pin(
            vllm_pin_payload
        )
        contract_file_identity = _file_identity(args.runtime_contract)
        card = load_shipcard(args.shipcard)
        manifest, manifest_raw = _load_json(
            args.manifest, where="serve manifest"
        )
        artifact_content_receipt = (
            validate_rtx4090_artifact_content_receipt(
                args.model_dir,
                manifest.get("artifact_content_receipt"),
            )
        )
        artifact = validate_rtx4090_artifact_metadata(
            args.model_dir,
            runtime_contract=runtime_contract,
            expected_model_sha=str(card.get("model_sha", "")),
        )
        runtime_attestation = artifact_runtime_attestation(args.model_dir)
        if args.arm == "graph":
            if (
                args.serve_log is None
                or args.compile_cache is None
                or args.compile_cache_host_root is None
                or args.compile_cache_preflight is None
                or args.session_nonce is None
            ):
                raise RTX4090FP8CBValidationError(
                    "graph arm requires serve-log, container/host compile-cache, "
                    "preflight receipt, and session nonce"
                )
            expected_nonce = _validate_served_model_name(
                args.served_model, model_sha=str(card.get("model_sha", ""))
            )
            if args.session_nonce != expected_nonce:
                raise RTX4090FP8CBValidationError(
                    "graph session nonce differs from the served-model identity"
                )
            graph_receipt = validate_rtx4090_graph_log(
                args.serve_log,
                expected_compile_cache_root=args.compile_cache,
            )
            graph_receipt["compile_cache_freshness"] = (
                validate_compile_cache_preflight(
                    args.compile_cache_host_root,
                    args.compile_cache_preflight,
                    configured_container_root=args.compile_cache,
                    session_nonce=args.session_nonce,
                )
            )
        else:
            if any(value is not None for value in (
                args.serve_log,
                args.compile_cache,
                args.compile_cache_host_root,
                args.compile_cache_preflight,
                args.session_nonce,
            )):
                raise RTX4090FP8CBValidationError(
                    "eager arm rejects graph-only evidence options"
                )
            graph_receipt = None
        validated = validate_rtx4090_serve_manifest(
            manifest,
            arm=args.arm,
            expected_image=args.expected_image,
            expected_served_model=args.served_model,
            expected_model_sha=str(card.get("model_sha", "")),
            expected_artifact_binding=artifact,
            expected_artifact_content_receipt=artifact_content_receipt,
            runtime_pin=runtime_pin,
            vllm_runtime_pin=vllm_runtime_pin,
            runtime_contract=runtime_contract,
            expected_runtime_attestation=runtime_attestation,
            runtime_contract_file_identity=contract_file_identity,
            compile_cache=args.compile_cache,
        )
        models_identity = validated["session"]["models_endpoint_binding"]
        smoke = run_endpoint_smoke(
            base_url=args.base_url,
            model_name=args.served_model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            expected_models_identity=models_identity,
        )
        endpoint_contract = build_rtx4090_endpoint_contract(
            arm=args.arm,
            artifact_binding=artifact,
            expected_image=args.expected_image,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            validated_manifest=validated,
            runtime_contract_file_identity=contract_file_identity,
            endpoint_smoke=smoke,
            graph_receipt=graph_receipt,
        )
        provenance = git_provenance()
        commit = provenance.get("commit")
        if (
            not isinstance(commit, str)
            or _COMMIT_RE.fullmatch(commit) is None
            or provenance.get("dirty") is not False
        ):
            raise RTX4090FP8CBValidationError(
                "release receipt requires one clean full PrismaQuant commit"
            )
        native_slot = f"native_export.{args.arm}"
        fill_slot(
            args.shipcard,
            native_slot,
            _record_for_contract(
                slot=native_slot,
                contract=endpoint_contract,
                model_sha=str(card["model_sha"]),
                git_commit=commit,
            ),
        )
        if args.arm == "graph":
            fill_slot(
                args.shipcard,
                RTX4090_FP8_CB_SLOT,
                _record_for_contract(
                    slot=RTX4090_FP8_CB_SLOT,
                    contract=endpoint_contract,
                    model_sha=str(card["model_sha"]),
                    git_commit=commit,
                ),
            )
        print(json.dumps(endpoint_contract, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"[rtx4090-fp8-cb] REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
