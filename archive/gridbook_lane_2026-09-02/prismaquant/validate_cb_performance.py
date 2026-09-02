#!/usr/bin/env python3
"""Paired, matched-budget performance gate for Gridbook CB artifacts.

Gridbook's ``gridbook-bench-serve`` intentionally emits one-arm measurements
and never decides parity.  This module consumes a predeclared comparison
manifest plus paired ``gridbook.vllm-bench-serve.v2`` reports, binds them to the
candidate shipcard and exact artifact inventory, and makes the paired decision.

The implementation is stdlib-only.  It does not run a server or reinterpret
vLLM timings; it pairs the block values already validated by Gridbook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gridbook_environment import (
    CANONICAL_GOLD_ENVIRONMENT,
    GRIDBOOK_ENVIRONMENT_SCHEMA,
)
from .gridbook_serving_runtime_pin import (
    GridbookServingRuntimePin,
    GridbookServingRuntimePinError,
    load_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)
from .native_baseline_feasibility import (
    SCHEMA as NATIVE_BASELINE_FEASIBILITY_SCHEMA,
    certificate_sha256 as native_feasibility_certificate_sha256,
    validate_native_baseline_certificate,
)
from .shipcard import (
    _manifest_gridbook_runtime_pin,
    _verify_gridbook_distribution_identity,
    assert_weight_stat_attestation,
    compute_model_sha,
    fill_slot,
    git_provenance,
    load_shipcard,
    verify as verify_shipcard,
)
from .validate_cb_endpoint import (
    DSV4_SPARK_GPU_NAME,
    DSV4_SPARK_VLLM_COMMIT,
    DSV4_SPARK_VLLM_IMAGE,
    DSV4_SPARK_VLLM_VERSION,
)


MANIFEST_SCHEMA = "prismaquant.cb_performance_manifest/1"
RESULT_SCHEMA = "prismaquant.cb_performance_parity/1"
EVIDENCE_SCHEMA = "prismaquant.cb_performance_evidence/1"
REPORT_SCHEMA = "gridbook.vllm-bench-serve.v2"
TELEMETRY_SCHEMA = "prismaquant.cb_performance_telemetry/1"
DISPLACED_CONTAINER_SCHEMA = "prismaquant.displaced_container_eligibility/1"
SLOT = "perf.matched_budget_parity"
TOOL = "validate_cb_performance.py"
DSV4_NUM_HIDDEN_LAYERS = 43
DSV4_NUM_ROUTED_EXPERTS = 256
MAX_PARITY_TOLERANCE = 0.05

_SHA256 = frozenset("0123456789abcdef")
_GRIDBOOK_NATIVE_EXTENSION_RE = re.compile(
    r"^(?:prismaquant_cb(?:_v2)?_ext|pq_cb_|pq_mxfp8_dense_|"
    r"pq_fp8_source_w8a16_)"
)
_CB_FORMAT_RE = re.compile(r"^(?:NVFP4_CB|FP8_CB)_K[0-9]+$")
_GRIDBOOK_BACKEND_RE = re.compile(r"^gridbook-[a-z0-9][a-z0-9._-]*$")
_DELEGATED_WIRE_TO_FORMAT = {
    "mxfp4_e2m1_ue8m0_g32": "MXFP4_SOURCE",
    "fp8_e4m3_ue8m0_block128": "FP8_BLOCK_UE8M0_SOURCE",
    "mxfp8_e4m3_e8m0_g32": "MXFP8_UE8M0_G32",
}
_PHASES = ("prefill", "decode", "mixed")
_REQUIRED_TELEMETRY = frozenset(
    {
        "routing_per_layer_per_step",
        "expert_occupancy",
        "active_experts",
        "grouped_moe_whole_operator",
    }
)
_GROUPED_OPERATOR_STAGES = frozenset(
    {"routing", "packing", "launches", "kernel", "combine"}
)
_ALLOWED_ARM_DIFFERENCE_PATHS = frozenset(
    {
        "artifacts.model_id",
        "artifacts.whole_served_artifact_bytes",
        "artifacts.budget_headroom_bytes",
        "artifacts.payload",
        "artifacts.inventory",
        "execution_identity.format_rung",
        "execution_identity.serialization",
        "execution_identity.quant_contract",
        "execution_identity.kernel_backend",
        "execution_identity.fallback_state",
        "execution_identity.manifest",
        "server.base_url",
        "server.served_model_name",
    }
)
_THROUGHPUT_METRICS = frozenset(
    {"request_throughput", "output_throughput", "total_token_throughput"}
)
_PHASE_METRICS = {
    "prefill": ("p95_ttft_ms",),
    "decode": ("p95_tpot_ms", "p95_itl_ms", "output_throughput"),
    "mixed": (
        "p95_ttft_ms",
        "p95_tpot_ms",
        "p95_itl_ms",
        "p95_e2el_ms",
        "request_throughput",
        "output_throughput",
    ),
}
# Performance intentionally preloads every Gridbook extension family so the
# candidate and displaced-container arms have identical residency.  Every
# other Gridbook input is the canonical gold value. The full map keeps
# the absent variables reviewable; the live process/report representation
# contains only the set-valued projection, with absence proven by the complete
# serve-fingerprint allowlist.
_PERFORMANCE_SERVER_ENVIRONMENT = {
    **dict(CANONICAL_GOLD_ENVIRONMENT),
    "PRISMAQUANT_PRELOAD_FUSED": "1",
}
_REQUIRED_SERVER_ENVIRONMENT = {
    name: value
    for name, value in _PERFORMANCE_SERVER_ENVIRONMENT.items()
    if value is not None
}
_REQUIRED_SERVER_ENVIRONMENT["PYTHONSAFEPATH"] = "1"


class CBPerformanceValidationError(RuntimeError):
    """The paired performance evidence is incomplete or failed parity."""


def _exact_gridbook_runtime_pin() -> GridbookServingRuntimePin:
    """Load the one reviewed Gridbook release accepted by this validator."""

    pin = load_gridbook_serving_runtime_pin()
    try:
        require_exact_gridbook_serving_runtime_release(pin)
    except GridbookServingRuntimePinError as exc:
        raise CBPerformanceValidationError(
            f"performance gate requires the exact Gridbook release: {exc}"
        ) from exc
    return pin


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CBPerformanceValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CBPerformanceValidationError(
                    f"{label} contains non-finite JSON number {value}"
                )
            ),
        )
    except CBPerformanceValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CBPerformanceValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CBPerformanceValidationError(f"{label} must contain one JSON object")
    return payload, raw


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _digest(encoded)


def _require_sha(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise CBPerformanceValidationError(f"{where} must be a lowercase SHA-256")
    return value


def _reference(base: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CBPerformanceValidationError(f"{where} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _read_bound_json(
    base: Path, reference: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], str, Path]:
    path = _reference(base, reference.get("path"), f"{label}.path")
    expected = _require_sha(reference.get("sha256"), f"{label}.sha256")
    payload, raw = _load_json(path, label=label)
    actual = _digest(raw)
    if actual != expected:
        raise CBPerformanceValidationError(
            f"{label} SHA-256 mismatch: declared={expected}, actual={actual}"
        )
    return payload, actual, path


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CBPerformanceValidationError(f"{where} must be a positive integer")
    return value


def _finite_positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CBPerformanceValidationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CBPerformanceValidationError(f"{where} must be finite and positive")
    return result


def _iso_datetime(value: object, where: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CBPerformanceValidationError(f"{where} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CBPerformanceValidationError(
            f"{where} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise CBPerformanceValidationError(f"{where} must include a timezone")
    return parsed


def _artifact_inventory(root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config, _ = _load_json(root / "config.json", label="candidate config")
    expected_config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": DSV4_NUM_HIDDEN_LAYERS,
        "n_routed_experts": DSV4_NUM_ROUTED_EXPERTS,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise CBPerformanceValidationError(
                f"candidate config {key}={config.get(key)!r}, expected {expected!r}"
            )
    quant_path = root / "quant_config.json"
    quant_config, _ = _load_json(quant_path, label="candidate quant_config")
    if quant_config.get("quant_method") != "gridbook" or quant_config.get(
        "format"
    ) != "nvfp4_cb":
        raise CBPerformanceValidationError(
            f"{root} is not a Gridbook nvfp4_cb artifact"
        )
    provenance = quant_config.get("provenance")
    inventory = provenance.get("artifact_inventory") if isinstance(
        provenance, Mapping
    ) else None
    if not isinstance(inventory, Mapping) or inventory.get(
        "schema"
    ) != "prismaquant.cb_export_artifact_inventory.v1":
        raise CBPerformanceValidationError(
            "candidate quant_config has no finalized CB artifact inventory"
        )
    if inventory.get("scope") != "all_regular_files_recursive":
        raise CBPerformanceValidationError("candidate inventory scope is not recursive")
    file_bytes = inventory.get("file_bytes")
    if not isinstance(file_bytes, Mapping) or not file_bytes:
        raise CBPerformanceValidationError("candidate inventory file_bytes is empty")
    observed: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CBPerformanceValidationError(
                f"candidate artifact contains symlink {path.relative_to(root)}"
            )
        if path.is_file():
            observed[path.relative_to(root).as_posix()] = int(path.stat().st_size)
    declared: dict[str, int] = {}
    for name, size in file_bytes.items():
        if not isinstance(name, str) or not name or name.startswith("/") or ".." in Path(name).parts:
            raise CBPerformanceValidationError(f"invalid inventory path {name!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CBPerformanceValidationError(f"invalid inventory byte count for {name!r}")
        declared[name] = size
    if observed != declared:
        raise CBPerformanceValidationError(
            "candidate recursive file sizes differ from quant_config inventory"
        )
    total = sum(declared.values())
    if inventory.get("export_directory_bytes") != total:
        raise CBPerformanceValidationError(
            "candidate inventory total differs from its exact file-byte sum"
        )
    inventory_digest = _digest(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return dict(inventory), inventory_digest, quant_config


def _cb_scheme_by_format(
    quant_config: Mapping[str, Any], *, label: str
) -> dict[str, Mapping[str, Any]]:
    groups = quant_config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise CBPerformanceValidationError(
            f"{label} quant_config has no concrete config groups"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for group_name, raw_group in groups.items():
        if not isinstance(group_name, str) or not isinstance(raw_group, Mapping):
            raise CBPerformanceValidationError(
                f"{label} quant_config has a malformed config group"
            )
        fmt = raw_group.get("format")
        scheme = raw_group.get("scheme")
        if not isinstance(fmt, str) or _CB_FORMAT_RE.fullmatch(fmt) is None:
            continue
        if not isinstance(scheme, Mapping):
            raise CBPerformanceValidationError(
                f"{label} CB config group {group_name!r} has no scheme"
            )
        prior = result.setdefault(fmt, scheme)
        scale = scheme.get("scale_coding")
        prior_scale = prior.get("scale_coding")
        if scale != prior_scale or scheme.get("grid") != prior.get("grid"):
            raise CBPerformanceValidationError(
                f"{label} CB format {fmt!r} has inconsistent route contracts"
            )
    return result


def _route_contract(
    fmt: str,
    *,
    cb_schemes: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, str]:
    """Canonical execution identity implied by one finalized artifact route."""
    if _CB_FORMAT_RE.fullmatch(fmt):
        scheme = cb_schemes.get(fmt)
        if not isinstance(scheme, Mapping):
            raise CBPerformanceValidationError(
                f"{label} assigns {fmt!r} without a matching CB config group"
            )
        grid = scheme.get("grid")
        if (fmt.startswith("NVFP4_CB_") and grid != "fp4") or (
            fmt.startswith("FP8_CB_") and grid != "fp8"
        ):
            raise CBPerformanceValidationError(
                f"{label} format {fmt!r} disagrees with its serialized CB grid"
            )
        raw_scale = scheme.get("scale_coding")
        if raw_scale is None:
            scale_coding = "v1"
        elif isinstance(raw_scale, Mapping) and raw_scale.get("kind") == "two_tier":
            scale_coding = "two_tier"
        else:
            raise CBPerformanceValidationError(
                f"{label} format {fmt!r} has an unknown CB scale coding"
            )
        return {
            "format_rung": fmt,
            "serialized_layout": "product-codebook-indices-v1",
            "scale_coding": scale_coding,
            "quant_contract": "W4A4" if grid == "fp4" else "W8A8",
            "backend_policy": "gridbook",
        }
    native = {
        "MXFP4_SOURCE": {
            "serialized_layout": "source-passthrough",
            "scale_coding": "ue8m0-g32",
            "quant_contract": "W4A16",
            "backend_policy": "vllm-marlin",
        },
        "FP8_BLOCK_UE8M0_SOURCE": {
            "serialized_layout": "source-passthrough",
            "scale_coding": "ue8m0-block128",
            "quant_contract": "W8A16",
            "backend_policy": "gridbook",
        },
        "MXFP8_UE8M0_G32": {
            "serialized_layout": "gridbook-native",
            "scale_coding": "e8m0-g32",
            "quant_contract": "W8A8",
            "backend_policy": "gridbook",
        },
        "NVFP4": {
            "serialized_layout": "compressed-tensors",
            "scale_coding": "ue4m3",
            "quant_contract": "W4A4",
            "backend_policy": "vllm-cutlass",
        },
        "FP8_DYNAMIC": {
            "serialized_layout": "compressed-tensors",
            "scale_coding": "fp32-block-scale",
            "quant_contract": "W8A8",
            "backend_policy": "vllm-cutlass",
        },
        "FP8_SOURCE": {
            "serialized_layout": "compressed-tensors",
            "scale_coding": "fp32-block-scale",
            "quant_contract": "W8A8",
            "backend_policy": "vllm-cutlass",
        },
        "BF16": {
            "serialized_layout": "plain",
            "scale_coding": "none",
            "quant_contract": "W16A16",
            "backend_policy": "vllm-native",
        },
    }.get(fmt)
    if native is None:
        raise CBPerformanceValidationError(
            f"{label} assigns unsupported serving format {fmt!r}"
        )
    return {"format_rung": fmt, **native}


def _packed_layer_and_member(member: str) -> tuple[str, str, int] | None:
    match = re.fullmatch(
        r"model[.]layers[.]([0-9]+)[.]mlp[.]experts[.]([0-9]+)[.]"
        r"(gate_proj|up_proj|down_proj)",
        member,
    )
    if match is None:
        return None
    family = "w13" if match.group(3) in {"gate_proj", "up_proj"} else "w2"
    return match.group(1), family, int(match.group(2))


def _derive_expected_execution_assignments(
    quant_config: Mapping[str, Any],
    certified_units: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, str]]:
    """Derive benchmark routes from inventory-bound artifact declarations.

    Report rows are observations.  ``tensor_formats``, the delegated-native
    declaration, and the per-expert partition are the independently bound
    producer records that decide what every certified serving unit must run.
    """
    provenance = quant_config.get("provenance")
    tensor_formats = provenance.get("tensor_formats") if isinstance(
        provenance, Mapping
    ) else None
    if not isinstance(tensor_formats, Mapping) or not tensor_formats:
        raise CBPerformanceValidationError(
            f"{label} quant_config lacks the finalized tensor_formats route map; "
            "re-export with the current producer"
        )
    formats: dict[str, str] = {}
    for unit, fmt in tensor_formats.items():
        if (
            not isinstance(unit, str)
            or not unit
            or not isinstance(fmt, str)
            or not fmt
        ):
            raise CBPerformanceValidationError(
                f"{label} tensor_formats route map is malformed"
            )
        formats[unit] = fmt

    delegated_record = quant_config.get("source_passthrough")
    delegated: dict[str, str] = {}
    if delegated_record is not None:
        units = delegated_record.get("units") if isinstance(
            delegated_record, Mapping
        ) else None
        if (
            not isinstance(delegated_record, Mapping)
            or set(delegated_record) != {"version", "units"}
            or delegated_record.get("version") != 1
            or not isinstance(units, Mapping)
            or not units
        ):
            raise CBPerformanceValidationError(
                f"{label} has a malformed delegated-native route declaration"
            )
        for unit, wire in units.items():
            if (
                not isinstance(unit, str)
                or not unit
                or not isinstance(wire, str)
                or wire not in _DELEGATED_WIRE_TO_FORMAT
            ):
                raise CBPerformanceValidationError(
                    f"{label} has an unknown delegated-native route"
                )
            delegated[unit] = _DELEGATED_WIRE_TO_FORMAT[wire]

    cb_schemes = _cb_scheme_by_format(quant_config, label=label)
    per_expert = quant_config.get("per_expert_format_groups")
    per_expert_layers = per_expert.get("layers") if isinstance(
        per_expert, Mapping
    ) else None
    if per_expert is not None and (
        set(per_expert) != {"version", "layers"}
        or per_expert.get("version") != 1
        or not isinstance(per_expert_layers, Mapping)
        or not per_expert_layers
    ):
        raise CBPerformanceValidationError(
            f"{label} has a malformed per-expert route declaration"
        )

    expected: list[dict[str, str]] = []
    covered_members: set[str] = set()
    covered_delegated: set[str] = set()
    for index, raw_unit in enumerate(certified_units):
        if not isinstance(raw_unit, Mapping):
            raise CBPerformanceValidationError(
                f"certified serving unit {index} is malformed"
            )
        kind = raw_unit.get("kind")
        unit_name = raw_unit.get("name")
        if not isinstance(unit_name, str) or not unit_name:
            raise CBPerformanceValidationError(
                f"certified serving unit {index} has no name"
            )
        if kind == "construction":
            fmt = delegated.get(unit_name)
            if fmt is None:
                raise CBPerformanceValidationError(
                    f"{label} construction unit {unit_name!r} has no bound route"
                )
            covered_delegated.add(unit_name)
            expected.append({
                "unit": unit_name,
                **_route_contract(fmt, cb_schemes=cb_schemes, label=label),
            })
            continue
        members = raw_unit.get("members")
        if kind != "serving" or not isinstance(members, list) or not members:
            raise CBPerformanceValidationError(
                f"certified serving unit {unit_name!r} is incomplete"
            )
        if any(not isinstance(member, str) or member not in formats for member in members):
            missing = [member for member in members if member not in formats]
            raise CBPerformanceValidationError(
                f"{label} route map does not cover certified members {missing[:8]}"
            )
        covered_members.update(members)
        packed = [_packed_layer_and_member(member) for member in members]
        layers = {entry[0] for entry in packed if entry is not None}
        if all(entry is not None for entry in packed) and len(layers) == 1 and (
            isinstance(per_expert_layers, Mapping)
            and next(iter(layers)) in per_expert_layers
        ):
            layer = next(iter(layers))
            layer_record = per_expert_layers[layer]
            if not isinstance(layer_record, Mapping) or set(layer_record) != {"w13", "w2"}:
                raise CBPerformanceValidationError(
                    f"{label} per-expert layer {layer} is malformed"
                )
            member_index: dict[tuple[str, int], list[str]] = {}
            for member, member_route in zip(members, packed, strict=True):
                assert member_route is not None
                member_index.setdefault(
                    (member_route[1], member_route[2]), []
                ).append(member)
            if any(
                len(route_members) != (2 if family == "w13" else 1)
                for (family, _expert), route_members in member_index.items()
            ):
                raise CBPerformanceValidationError(
                    f"{label} certified per-expert unit has an incomplete "
                    "gate/up/down member partition"
                )
            seen_member_keys: set[tuple[str, int]] = set()
            for family in ("w13", "w2"):
                entries = layer_record.get(family)
                if not isinstance(entries, list) or not entries:
                    raise CBPerformanceValidationError(
                        f"{label} per-expert layer {layer}/{family} is empty"
                    )
                for entry in entries:
                    if not isinstance(entry, Mapping) or set(entry) != {
                        "format_wire_id", "expert_ids", "tensor_prefix"
                    }:
                        raise CBPerformanceValidationError(
                            f"{label} per-expert layer {layer}/{family} is malformed"
                        )
                    wire = entry.get("format_wire_id")
                    fmt = _DELEGATED_WIRE_TO_FORMAT.get(str(wire), str(wire))
                    expert_ids = entry.get("expert_ids")
                    tensor_prefix = entry.get("tensor_prefix")
                    if (
                        not isinstance(wire, str)
                        or not isinstance(expert_ids, list)
                        or not expert_ids
                        or expert_ids != sorted(set(expert_ids))
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in expert_ids)
                        or not isinstance(tensor_prefix, str)
                        or not tensor_prefix
                    ):
                        raise CBPerformanceValidationError(
                            f"{label} per-expert layer {layer}/{family} values are malformed"
                        )
                    for expert in expert_ids:
                        key = (family, expert)
                        route_members = member_index.get(key)
                        if (
                            route_members is None
                            or key in seen_member_keys
                            or {formats[member] for member in route_members} != {fmt}
                        ):
                            raise CBPerformanceValidationError(
                                f"{label} per-expert declaration disagrees with tensor_formats"
                            )
                        seen_member_keys.add(key)
                    expected.append({
                        # ``MXFP4_SOURCE`` deliberately reuses the physical
                        # expert tensor prefix across w13 and w2.  The runtime
                        # dispatches those as independent family/format lanes
                        # from the declaration, so a bare tensor prefix both
                        # collapses two real routes and violates the execution-
                        # manifest uniqueness contract.  Encode the consumer
                        # declaration's complete (tensor-prefix, family,
                        # wire-id) dispatch identity.
                        "unit": f"{tensor_prefix}/{family}/{wire}",
                        **_route_contract(fmt, cb_schemes=cb_schemes, label=label),
                    })
            if seen_member_keys != set(member_index):
                raise CBPerformanceValidationError(
                    f"{label} per-expert declaration does not cover its certified unit"
                )
            continue

        unit_formats = {formats[member] for member in members}
        if len(unit_formats) != 1:
            raise CBPerformanceValidationError(
                f"{label} certified atomic unit {unit_name!r} mixes formats "
                "without a per-expert route declaration"
            )
        fmt = next(iter(unit_formats))
        declaration_id = unit_name.removeprefix("group:")
        if ".mlp.experts." in declaration_id:
            declaration_id = declaration_id.split(".mlp.experts.", 1)[0] + ".mlp.experts"
        declared_fmt = delegated.get(declaration_id)
        if fmt in _DELEGATED_WIRE_TO_FORMAT.values():
            if declared_fmt != fmt:
                raise CBPerformanceValidationError(
                    f"{label} delegated route {declaration_id!r} disagrees with tensor_formats"
                )
            covered_delegated.add(declaration_id)
        elif declared_fmt is not None:
            raise CBPerformanceValidationError(
                f"{label} route {declaration_id!r} is both native and {fmt!r}"
            )
        expected.append({
            "unit": unit_name,
            **_route_contract(fmt, cb_schemes=cb_schemes, label=label),
        })

    extra_members = sorted(set(formats) - covered_members)
    if extra_members:
        raise CBPerformanceValidationError(
            f"{label} tensor_formats includes uncertified serving members "
            f"{extra_members[:8]}"
        )
    # Source-only/floor units (for example DSv4's unallocated block-FP8
    # attention leaves) are deliberately outside the quantizable probe census
    # but remain live serving routes.  The finalized delegated declaration is
    # their authority; include every such unit instead of silently omitting it
    # from the performance manifest.
    for unit in sorted(set(delegated) - covered_delegated):
        expected.append({
            "unit": unit,
            **_route_contract(
                delegated[unit], cb_schemes=cb_schemes, label=label
            ),
        })
    units = [row["unit"] for row in expected]
    if len(units) != len(set(units)):
        raise CBPerformanceValidationError(
            f"{label} artifact-derived execution route ids are not unique"
        )
    return sorted(expected, key=lambda row: row["unit"])


def _validate_execution_routes(
    report: Mapping[str, Any],
    expected: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> None:
    execution = (report.get("metadata") or {}).get("execution_identity")
    manifest = execution.get("manifest") if isinstance(execution, Mapping) else None
    assignments = manifest.get("assignments") if isinstance(manifest, Mapping) else None
    if not isinstance(assignments, list):
        raise CBPerformanceValidationError(f"{label} has no execution assignments")
    if [row.get("unit") for row in assignments if isinstance(row, Mapping)] != sorted(
        row["unit"] for row in expected
    ):
        raise CBPerformanceValidationError(
            f"{label} execution manifest does not equal the artifact-derived route census"
        )
    expected_by_unit = {row["unit"]: row for row in expected}
    concrete_fields = (
        "format_rung", "serialized_layout", "scale_coding", "quant_contract"
    )
    for assignment in assignments:
        assert isinstance(assignment, Mapping)
        expected_row = expected_by_unit[str(assignment["unit"])]
        if any(assignment.get(field) != expected_row[field] for field in concrete_fields):
            raise CBPerformanceValidationError(
                f"{label} execution assignment {assignment['unit']!r} disagrees "
                "with finalized quant_config"
            )
        backend = assignment.get("kernel_backend")
        policy = expected_row["backend_policy"]
        if (
            not isinstance(backend, str)
            or (
                _GRIDBOOK_BACKEND_RE.fullmatch(backend) is None
                if policy == "gridbook"
                else backend != policy
            )
        ):
            mismatch = (
                f"non-Gridbook backend {backend!r}"
                if policy == "gridbook"
                else f"backend {backend!r}, expected route {policy!r}"
            )
            raise CBPerformanceValidationError(
                f"{label} execution assignment {assignment['unit']!r} uses "
                f"{mismatch}"
            )
        if assignment.get("fallback_state") not in {"none", "none-observed"}:
            raise CBPerformanceValidationError(
                f"{label} execution assignment {assignment['unit']!r} used a fallback"
            )

    assert isinstance(execution, Mapping)
    summaries = {
        "format_rung": execution.get("format_rung"),
        "serialized_layout": (execution.get("serialization") or {}).get("layout"),
        "scale_coding": (execution.get("serialization") or {}).get("scale_coding"),
        "quant_contract": execution.get("quant_contract"),
        "kernel_backend": execution.get("kernel_backend"),
        "fallback_state": execution.get("fallback_state"),
    }
    for field, observed_summary in summaries.items():
        values = {str(row[field]) for row in assignments}
        expected_summary = next(iter(values)) if len(values) == 1 else "mixed"
        if observed_summary != expected_summary:
            raise CBPerformanceValidationError(
                f"{label} execution summary {field}={observed_summary!r}, "
                f"expected {expected_summary!r} from concrete assignments"
            )


def _validate_report(
    report: Mapping[str, Any], *, label: str, pin_commit: str,
    pin_version: str, pin_wheel_sha256: str
) -> None:
    expected = {
        "schema": REPORT_SCHEMA,
        "status": "success",
        "evidence_scope": "single-arm-serving-measurement",
        "measurement_valid": True,
        "release_eligible": True,
        "parity_acceptance": False,
        "release_acceptance": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise CBPerformanceValidationError(
                f"{label}.{key}={report.get(key)!r}, expected {value!r}"
            )
    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CBPerformanceValidationError(f"{label}.metadata is missing")
    measurement_provenance = metadata.get("measurement_provenance")
    required_provenance = {
        "digest_bound_inputs_verified_before_requests": True,
        "digest_bound_inputs_verified_after_requests": True,
        "git_state_verified_after_requests": True,
        "client_runtime_verified_after_requests": True,
    }
    if not isinstance(measurement_provenance, Mapping) or any(
        measurement_provenance.get(key) is not expected
        for key, expected in required_provenance.items()
    ):
        raise CBPerformanceValidationError(
            f"{label} lacks complete pre/post measurement provenance checks"
        )
    git = metadata.get("git")
    software = metadata.get("software")
    if not isinstance(git, Mapping) or git.get("commit") != pin_commit:
        raise CBPerformanceValidationError(f"{label} was not run at the tracked Gridbook commit")
    if git.get("dirty") is not False or git.get("release_eligible") is not True:
        raise CBPerformanceValidationError(f"{label} Gridbook checkout is not release-eligible")
    if not isinstance(software, Mapping) or software.get("gridbook_version") != pin_version:
        raise CBPerformanceValidationError(f"{label} Gridbook package version is not pinned")
    artifacts = metadata.get("artifacts")
    execution = metadata.get("execution_identity")
    server = metadata.get("server")
    dispatch = metadata.get("dispatch")
    if not isinstance(artifacts, Mapping) or artifacts.get(
        "image_id"
    ) != DSV4_SPARK_VLLM_IMAGE:
        raise CBPerformanceValidationError(
            f"{label} did not use the exact DSv4 Spark serving image"
        )
    if not isinstance(execution, Mapping):
        raise CBPerformanceValidationError(f"{label} has no execution identity")
    if execution.get("tensor_parallel_size") != 1:
        raise CBPerformanceValidationError(f"{label} must execute at TP=1")
    runtime = execution.get("server_runtime_id")
    if runtime != DSV4_SPARK_VLLM_VERSION or DSV4_SPARK_VLLM_COMMIT not in runtime:
        raise CBPerformanceValidationError(
            f"{label} did not use exact vLLM {DSV4_SPARK_VLLM_VERSION}"
        )
    hardware = execution.get("hardware")
    if not isinstance(hardware, Mapping) or hardware.get("gpu_id") != DSV4_SPARK_GPU_NAME:
        raise CBPerformanceValidationError(
            f"{label} did not execute on one {DSV4_SPARK_GPU_NAME}"
        )
    execution_manifest = execution.get("manifest")
    inventory = artifacts.get("inventory")
    assignments = execution_manifest.get("assignments") if isinstance(
        execution_manifest, Mapping
    ) else None
    if (
        not isinstance(execution_manifest, Mapping)
        or execution_manifest.get("schema") != "gridbook.execution-manifest.v1"
        or execution_manifest.get("coverage") != "all_serving_units"
        or execution_manifest.get("tensor_parallel_size") != 1
        or not isinstance(execution_manifest.get("assignment_count"), int)
        or execution_manifest.get("assignment_count", 0) <= 0
        or not isinstance(assignments, list)
        or len(assignments) != execution_manifest.get("assignment_count")
        or not isinstance(inventory, Mapping)
        or execution_manifest.get("artifact_inventory_sha256")
        != inventory.get("sha256")
    ):
        raise CBPerformanceValidationError(
            f"{label} lacks a complete inventory-bound TP=1 execution manifest"
        )
    assignment_fields = (
        "unit",
        "format_rung",
        "serialized_layout",
        "scale_coding",
        "quant_contract",
        "kernel_backend",
        "fallback_state",
    )
    units: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping) or any(
            not isinstance(assignment.get(field), str)
            or not assignment.get(field, "").strip()
            or assignment.get(field, "").strip().lower() == "mixed"
            for field in assignment_fields
        ):
            raise CBPerformanceValidationError(
                f"{label} execution assignment {index} is not concrete"
            )
        unit = str(assignment["unit"])
        if unit in units:
            raise CBPerformanceValidationError(
                f"{label} execution manifest repeats serving unit {unit!r}"
            )
        units.add(unit)
    if not isinstance(server, Mapping) or server.get("prefix_caching") != "off":
        raise CBPerformanceValidationError(f"{label} did not disable prefix caching")
    evidence = server.get("evidence")
    attachments = evidence.get("attachments") if isinstance(evidence, Mapping) else None
    if not isinstance(attachments, list) or not attachments:
        raise CBPerformanceValidationError(f"{label} has no bound startup/dispatch evidence")
    for index, attachment in enumerate(attachments):
        if (
            not isinstance(attachment, Mapping)
            or not isinstance(attachment.get("reference"), str)
            or not attachment.get("reference")
            or not isinstance(attachment.get("bytes"), int)
            or attachment.get("bytes", 0) <= 0
        ):
            raise CBPerformanceValidationError(
                f"{label} server evidence attachment {index} is malformed"
            )
        _require_sha(
            attachment.get("sha256"),
            f"{label} server evidence attachment {index}.sha256",
        )
    server_environment = dispatch.get("server_environment") if isinstance(
        dispatch, Mapping
    ) else None
    environment_values = server_environment.get("values") if isinstance(
        server_environment, Mapping
    ) else None
    required_report_environment = {
        **_REQUIRED_SERVER_ENVIRONMENT,
        "PQ_GRIDBOOK_RUNTIME_COMMIT": pin_commit,
        "PQ_GRIDBOOK_RUNTIME_VERSION": pin_version,
        "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256": pin_wheel_sha256,
    }
    if (
        not isinstance(environment_values, Mapping)
        or dict(environment_values) != required_report_environment
    ):
        raise CBPerformanceValidationError(
            f"{label} lacks the exact DSv4 Gridbook dispatch environment"
        )
    summary = report.get("summary")
    blocks = report.get("blocks")
    if not isinstance(summary, Mapping) or not isinstance(blocks, list) or len(blocks) < 3:
        raise CBPerformanceValidationError(f"{label} lacks three validated blocks")
    if summary.get("completed_blocks") != len(blocks):
        raise CBPerformanceValidationError(f"{label} block count is inconsistent")
    for index, block in enumerate(blocks):
        raw = block.get("raw_result") if isinstance(block, Mapping) else None
        if (
            not isinstance(block, Mapping)
            or block.get("index") != index + 1
            or block.get("status") != "success"
            or block.get("returncode") != 0
            or block.get("validation_error") is not None
            or not isinstance(raw, Mapping)
        ):
            raise CBPerformanceValidationError(
                f"{label}.blocks[{index}] is not a complete successful Gridbook block"
            )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CBPerformanceValidationError(f"{label} summary metrics are missing")
    for metric, row in metrics.items():
        if not isinstance(metric, str) or not isinstance(row, Mapping):
            raise CBPerformanceValidationError(f"{label} summary metric is malformed")
        values = row.get("values")
        observed = [block["raw_result"].get(metric) for block in blocks]
        if values != observed:
            raise CBPerformanceValidationError(
                f"{label} summary metric {metric!r} differs from raw validated blocks"
            )


def _server_tokens(report: Mapping[str, Any], label: str) -> list[str]:
    metadata = report.get("metadata")
    server = metadata.get("server") if isinstance(metadata, Mapping) else None
    recorded = server.get("recorded_args") if isinstance(server, Mapping) else None
    if not isinstance(recorded, list) or not recorded or not all(
        isinstance(value, str) and value for value in recorded
    ):
        raise CBPerformanceValidationError(f"{label} has no exact recorded server argv")
    tokens: list[str] = []
    try:
        for value in recorded:
            tokens.extend(shlex.split(value))
    except ValueError as exc:
        raise CBPerformanceValidationError(f"{label} server argv is not parseable") from exc
    return tokens


def _flag_values(tokens: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == flag:
            values.append(tokens[index + 1] if index + 1 < len(tokens) else "")
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    return values


def _normalized_server_tokens(report: Mapping[str, Any], label: str) -> list[str]:
    tokens = _server_tokens(report, label)
    if tokens.count("--no-enable-prefix-caching") != 1 or "--enable-prefix-caching" in tokens:
        raise CBPerformanceValidationError(
            f"{label} must record exactly --no-enable-prefix-caching"
        )
    if _flag_values(tokens, "--tensor-parallel-size") != ["1"]:
        raise CBPerformanceValidationError(
            f"{label} must record exactly --tensor-parallel-size 1"
        )
    if _flag_values(tokens, "--quantization") != ["gridbook"]:
        raise CBPerformanceValidationError(
            f"{label} must record exactly --quantization gridbook"
        )
    if _flag_values(tokens, "--kv-cache-dtype") != ["fp8"]:
        raise CBPerformanceValidationError(
            f"{label} must record exactly --kv-cache-dtype fp8"
        )
    normalized: list[str] = []
    masked_flags = {"--model", "--served-model-name"}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        matched = next(
            (flag for flag in masked_flags if token == flag or token.startswith(flag + "=")),
            None,
        )
        if matched is None:
            normalized.append(token)
            index += 1
            continue
        if token == matched:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise CBPerformanceValidationError(f"{label} {matched} has no value")
            normalized.extend([matched, "<ARM_MODEL>"])
            index += 2
        else:
            normalized.append(matched + "=<ARM_MODEL>")
            index += 1
    return normalized


def _remove_path(payload: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _matched_metadata(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    allowed_paths: Sequence[str],
    *,
    cell_id: str,
) -> None:
    if len(allowed_paths) != len(set(allowed_paths)) or any(
        path not in _ALLOWED_ARM_DIFFERENCE_PATHS for path in allowed_paths
    ):
        raise CBPerformanceValidationError(
            f"cell {cell_id} declares an unknown or duplicate arm difference path"
        )
    candidate_metadata = json.loads(json.dumps(candidate.get("metadata")))
    baseline_metadata = json.loads(json.dumps(baseline.get("metadata")))
    candidate_server = candidate_metadata.get("server")
    baseline_server = baseline_metadata.get("server")
    if _normalized_server_tokens(candidate, f"cell {cell_id} candidate") != (
        _normalized_server_tokens(baseline, f"cell {cell_id} baseline")
    ):
        raise CBPerformanceValidationError(
            f"cell {cell_id} server argv differs outside model/served-name values"
        )
    candidate_evidence = candidate_server.get("evidence") if isinstance(
        candidate_server, Mapping
    ) else None
    baseline_evidence = baseline_server.get("evidence") if isinstance(
        baseline_server, Mapping
    ) else None
    candidate_attachments = candidate_evidence.get("attachments") if isinstance(
        candidate_evidence, Mapping
    ) else None
    baseline_attachments = baseline_evidence.get("attachments") if isinstance(
        baseline_evidence, Mapping
    ) else None
    if not isinstance(candidate_attachments, list) or not isinstance(
        baseline_attachments, list
    ) or len(candidate_attachments) != len(baseline_attachments):
        raise CBPerformanceValidationError(
            f"cell {cell_id} server evidence attachment counts differ"
        )
    for path in ("server.recorded_args", "server.evidence"):
        _remove_path(candidate_metadata, path)
        _remove_path(baseline_metadata, path)
    for path in allowed_paths:
        _remove_path(candidate_metadata, path)
        _remove_path(baseline_metadata, path)
    if candidate_metadata != baseline_metadata:
        raise CBPerformanceValidationError(
            f"cell {cell_id} candidate/baseline stack, hardware, settings, or workload differ outside the declared arm fields"
        )


def _report_artifact(
    report: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    metadata = report.get("metadata")
    artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
    if not isinstance(artifacts, Mapping):
        raise CBPerformanceValidationError(f"{label} has no artifact metadata")
    return artifacts


def _validate_report_artifact(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    byte_budget: int,
    label: str,
) -> None:
    artifacts = _report_artifact(report, label=label)
    expected_bytes = _positive_int(expected.get("artifact_bytes"), f"{label} expected bytes")
    expected_inventory = _require_sha(
        expected.get("artifact_inventory_sha256"), f"{label} expected inventory"
    )
    if artifacts.get("model_id") != expected.get("model_id"):
        raise CBPerformanceValidationError(f"{label} model_id differs from the manifest")
    if artifacts.get("whole_served_artifact_bytes") != expected_bytes:
        raise CBPerformanceValidationError(f"{label} artifact bytes differ from the manifest")
    if artifacts.get("byte_budget_bytes") != byte_budget or artifacts.get(
        "within_byte_budget"
    ) is not True:
        raise CBPerformanceValidationError(f"{label} does not attest the matched byte budget")
    if expected_bytes > byte_budget:
        raise CBPerformanceValidationError(f"{label} exceeds the matched byte budget")
    inventory = artifacts.get("inventory")
    if not isinstance(inventory, Mapping) or inventory.get("sha256") != expected_inventory:
        raise CBPerformanceValidationError(f"{label} inventory digest differs from the manifest")
    if inventory.get("computed_total_bytes") != expected_bytes:
        raise CBPerformanceValidationError(f"{label} inventory total is inconsistent")


def _validate_benchmark_commands(
    report: Mapping[str, Any], *, expected_model_id: str, label: str
) -> None:
    """Bind each measured block to the report's declared endpoint identity."""
    metadata = report.get("metadata")
    server = metadata.get("server") if isinstance(metadata, Mapping) else None
    artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
    if not isinstance(server, Mapping) or not isinstance(artifacts, Mapping):
        raise CBPerformanceValidationError(f"{label} lacks server/artifact metadata")
    workload = metadata.get("workload")
    if not isinstance(workload, Mapping):
        raise CBPerformanceValidationError(f"{label} has no workload metadata")
    expected_flags = {
        "--base-url": server.get("base_url"),
        "--endpoint": server.get("endpoint"),
        "--served-model-name": expected_model_id,
        "--model": artifacts.get("benchmark_model"),
        "--tokenizer": artifacts.get("tokenizer"),
        "--dataset-name": "random",
        "--random-input-len": str(workload.get("requested_random_input_len")),
        "--random-output-len": str(workload.get("output_len")),
        "--random-range-ratio": json.dumps(
            {"input": workload.get("input_range_ratio"), "output": 0.0},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--num-prompts": str(workload.get("num_prompts_per_block")),
        "--num-warmups": str(workload.get("warmups_per_block")),
        "--max-concurrency": str(workload.get("max_concurrency")),
        "--request-rate": str(workload.get("request_rate")),
        "--burstiness": str(workload.get("request_burstiness")),
        "--temperature": "0",
    }
    if any(not isinstance(value, str) or not value for value in expected_flags.values()):
        raise CBPerformanceValidationError(
            f"{label} has an incomplete benchmark endpoint identity"
        )
    blocks = report.get("blocks")
    assert isinstance(blocks, list)  # validated by _validate_report
    seeds = workload.get("dataset_block_seeds")
    percentiles = workload.get("percentiles")
    metrics = workload.get("metrics")
    if (
        workload.get("dataset") != "random"
        or workload.get("sampling")
        != {"strategy": "greedy", "temperature": 0.0, "sampling_seed": None}
        or workload.get("ignore_eos") is not True
        or workload.get("streaming") is not True
        or not isinstance(seeds, list)
        or len(seeds) != len(blocks)
        or not isinstance(percentiles, list)
        or not percentiles
        or not isinstance(metrics, list)
        or not metrics
    ):
        raise CBPerformanceValidationError(
            f"{label} workload metadata is not the exact release benchmark contract"
        )
    for index, block in enumerate(blocks):
        command = block.get("command") if isinstance(block, Mapping) else None
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            raise CBPerformanceValidationError(
                f"{label}.blocks[{index}] has no exact benchmark command"
            )
        for flag, expected in expected_flags.items():
            if _flag_values(command, flag) != [expected]:
                raise CBPerformanceValidationError(
                    f"{label}.blocks[{index}] does not target the declared "
                    f"server {flag}={expected!r}"
                )
        expected_block_flags = {
            "--seed": str(seeds[index]),
            "--percentile-metrics": ",".join(str(value) for value in metrics),
            "--metric-percentiles": ",".join(
                str(int(value)) if float(value).is_integer() else str(value)
                for value in percentiles
            ),
        }
        for flag, expected in expected_block_flags.items():
            if _flag_values(command, flag) != [expected]:
                raise CBPerformanceValidationError(
                    f"{label}.blocks[{index}] workload {flag} differs from metadata"
                )
        for switch in (
            "--ignore-eos",
            "--disable-shuffle",
            "--save-result",
            "--save-detailed",
        ):
            if command.count(switch) != 1:
                raise CBPerformanceValidationError(
                    f"{label}.blocks[{index}] must record exactly one {switch}"
                )


def _load_performance_serve_manifests(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    evidence_base: Path,
    attestation_references: Mapping[str, Any],
    pin_repository: str,
    pin_commit: str,
    pin_version: str,
    pin_wheel_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Replay the digest-bound live-server identity attached to one report.

    Gridbook deliberately treats its server evidence as opaque bytes.  The
    paired PrismaQuant gate is where a release report upgrades one attachment
    into a semantically checked ``serve_manifest.json`` and binds the measured
    endpoint, process, runtime, argv, environment, and mounted artifact.
    """
    try:
        from tools.serve_fingerprint import (
            MANIFEST_SCHEMA as SERVE_MANIFEST_SCHEMA,
            SERVER_ENV_ALLOWLIST,
            elide_argv_paths,
            fingerprint,
            normalize_performance_argv,
            performance_stack_fingerprint,
            process_identity_sha256,
            serve_session_fingerprint,
        )
    except Exception as exc:  # pragma: no cover - checkout/package misuse
        raise CBPerformanceValidationError(
            "tools.serve_fingerprint is unavailable; validate from the PrismaQuant tree"
        ) from exc

    metadata = report.get("metadata")
    server = metadata.get("server") if isinstance(metadata, Mapping) else None
    dispatch = metadata.get("dispatch") if isinstance(metadata, Mapping) else None
    evidence = server.get("evidence") if isinstance(server, Mapping) else None
    attachments = evidence.get("attachments") if isinstance(evidence, Mapping) else None
    if not isinstance(attachments, list) or not attachments:
        raise CBPerformanceValidationError(f"{label} has no server evidence")

    report_attachments: dict[str, tuple[str, int]] = {}
    for index, attachment in enumerate(attachments):
        assert isinstance(attachment, Mapping)  # validated by _validate_report
        path = _reference(
            evidence_base,
            attachment.get("reference"),
            f"{label} server evidence attachment {index}.reference",
        )
        if path.is_symlink() or not path.is_file():
            raise CBPerformanceValidationError(
                f"{label} server evidence attachment {index} is not a regular non-symlink file"
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CBPerformanceValidationError(
                f"cannot read {label} server evidence attachment {index}: {exc}"
            ) from exc
        actual_digest = _digest(raw)
        expected_digest = _require_sha(
            attachment.get("sha256"),
            f"{label} server evidence attachment {index}.sha256",
        )
        if actual_digest != expected_digest or len(raw) != attachment.get("bytes"):
            raise CBPerformanceValidationError(
                f"{label} server evidence attachment {index} changed after measurement"
            )
        normalized_path = str(path.resolve())
        if normalized_path in report_attachments:
            raise CBPerformanceValidationError(
                f"{label} repeats one server evidence attachment"
            )
        report_attachments[normalized_path] = (actual_digest, len(raw))

    if not isinstance(attestation_references, Mapping) or set(
        attestation_references
    ) != {"pre", "post"}:
        raise CBPerformanceValidationError(
            f"{label} comparison cell must bind exact pre/post serve attestations"
        )
    by_phase: dict[str, tuple[Mapping[str, Any], str, Path]] = {}
    for phase in ("pre", "post"):
        reference = attestation_references.get(phase)
        if not isinstance(reference, Mapping):
            raise CBPerformanceValidationError(
                f"{label} {phase} serve attestation reference is malformed"
            )
        manifest, digest, path = _read_bound_json(
            evidence_base,
            reference,
            label=f"{label} {phase} serve attestation",
        )
        if (
            manifest.get("schema") != SERVE_MANIFEST_SCHEMA
            or manifest.get("attestation_phase") != phase
        ):
            raise CBPerformanceValidationError(
                f"{label} {phase} reference is not a {phase} {SERVE_MANIFEST_SCHEMA}"
            )
        by_phase[phase] = (manifest, digest, path)
    pre, pre_digest, pre_path = by_phase["pre"]
    post, post_digest, _ = by_phase["post"]
    if report_attachments.get(str(pre_path.resolve()), (None,))[0] != pre_digest:
        raise CBPerformanceValidationError(
            f"{label} report does not digest-bind the exact pre-server attestation"
        )
    if any(digest == post_digest for digest, _ in report_attachments.values()):
        raise CBPerformanceValidationError(
            f"{label} post-server attestation incorrectly predates the benchmark report"
        )

    started = _iso_datetime(report.get("started_at"), f"{label}.started_at")
    finished = _iso_datetime(report.get("finished_at"), f"{label}.finished_at")
    pre_created = _iso_datetime(pre.get("created"), f"{label} pre serve manifest.created")
    post_created = _iso_datetime(post.get("created"), f"{label} post serve manifest.created")
    if not pre_created <= started < finished <= post_created:
        raise CBPerformanceValidationError(
            f"{label} serve attestation chronology does not bracket the benchmark"
        )
    # The two snapshots describe one immutable live serve.  Only their wall
    # clock and chronology role may differ; comparing a hand-picked subset
    # would let a newly added observed field silently drift during a run.
    chronology_fields = {"created", "attestation_phase", "serve_fingerprint"}
    immutable_keys = (set(pre) | set(post)) - chronology_fields
    changed = sorted(key for key in immutable_keys if pre.get(key) != post.get(key))
    if changed:
        raise CBPerformanceValidationError(
            f"{label} live server identity changed during measurement: {changed}"
        )
    manifest = pre

    recorded_fingerprint = manifest.get("serve_fingerprint")
    if (
        not isinstance(recorded_fingerprint, str)
        or recorded_fingerprint != fingerprint(manifest)
        or not isinstance(post.get("serve_fingerprint"), str)
        or post.get("serve_fingerprint") != fingerprint(post)
    ):
        raise CBPerformanceValidationError(f"{label} serve manifest fingerprint is stale")
    expected_binding = {
        "schema": "prismaquant.served_artifact_binding/1",
        "model_sha": expected.get("model_sha"),
        "artifact_inventory_sha256": expected.get("artifact_inventory_sha256"),
        "artifact_bytes": expected.get("artifact_bytes"),
    }
    binding = manifest.get("artifact_binding")
    if not isinstance(binding, Mapping) or any(
        binding.get(key) != value for key, value in expected_binding.items()
    ):
        raise CBPerformanceValidationError(
            f"{label} live server is not bound to the exact expected artifact"
        )
    if binding.get("resolved_path") != "/model" or binding.get("launch_model") != "/model":
        raise CBPerformanceValidationError(
            f"{label} live artifact path is not the exact /model launch mount"
        )
    expected_model_id = expected.get("model_id")
    if not isinstance(expected_model_id, str) or not expected_model_id:
        raise CBPerformanceValidationError(f"{label} expected model_id is missing")
    _validate_benchmark_commands(
        report, expected_model_id=expected_model_id, label=label
    )

    launch_argv = manifest.get("launch_argv")
    recorded_args = server.get("recorded_args") if isinstance(server, Mapping) else None
    if (
        not isinstance(launch_argv, list)
        or not launch_argv
        or not all(isinstance(token, str) and token for token in launch_argv)
        or recorded_args != launch_argv
    ):
        raise CBPerformanceValidationError(
            f"{label} report argv is not the exact live-server launch argv"
        )
    if manifest.get("launch_flags") != elide_argv_paths(launch_argv):
        raise CBPerformanceValidationError(f"{label} serve launch_flags are stale")
    if manifest.get("normalized_performance_argv") != normalize_performance_argv(
        launch_argv
    ):
        raise CBPerformanceValidationError(
            f"{label} normalized server argv is missing or stale"
        )
    if _flag_values(launch_argv, "--served-model-name") != [expected_model_id]:
        raise CBPerformanceValidationError(
            f"{label} live server argv does not bind the expected served model name"
        )
    if (
        manifest.get("source") != "server"
        or manifest.get("model") != "/model"
        or manifest.get("served_model_name") != expected_model_id
        or manifest.get("image") != DSV4_SPARK_VLLM_IMAGE
        or manifest.get("residency_readable") is not True
        or manifest.get("quantization") != "gridbook"
        or manifest.get("kv_cache_dtype") != "fp8"
        or manifest.get("gpu_name") != DSV4_SPARK_GPU_NAME
        or not isinstance(manifest.get("gpu_uuid"), str)
        or not str(manifest.get("gpu_uuid")).startswith("GPU-")
        or manifest.get("gpu_count") != 1
        or not isinstance(manifest.get("serve_session_id"), str)
        or len(str(manifest.get("serve_session_id"))) != 64
    ):
        raise CBPerformanceValidationError(
            f"{label} serve manifest has the wrong source/model/image/Gridbook/GPU contract"
        )
    if _manifest_gridbook_runtime_pin(manifest, {
        "commit": pin_commit,
        "version": pin_version,
        "wheel_sha256": pin_wheel_sha256,
    }) is None:
        raise CBPerformanceValidationError(f"{label} live server uses the wrong Gridbook pin")
    distribution_problems = _verify_gridbook_distribution_identity(
        "perf.matched_budget_parity",
        manifest,
        {
            "repository": pin_repository,
            "commit": pin_commit,
            "version": pin_version,
            "wheel_sha256": pin_wheel_sha256,
        },
        canonical_sha=_canonical_json_sha256,
    )
    if distribution_problems:
        raise CBPerformanceValidationError(
            f"{label} live server does not attest the exact imported Gridbook "
            f"distribution: {distribution_problems[0]}"
        )
    packages = manifest.get("package_versions")
    if (
        not isinstance(packages, Mapping)
        or packages.get("gridbook") != pin_version
        or packages.get("vllm") != DSV4_SPARK_VLLM_VERSION
    ):
        raise CBPerformanceValidationError(
            f"{label} live server package versions are not the release pins"
        )
    extensions = manifest.get("resident_extensions")
    if (
        not isinstance(extensions, list)
        or extensions != sorted(set(extensions))
        or not any(
            isinstance(name, str)
            and ".so" in name
            and _GRIDBOOK_NATIVE_EXTENSION_RE.match(name) is not None
            for name in extensions
        )
    ):
        raise CBPerformanceValidationError(
            f"{label} live server has no resident reviewed Gridbook CUDA extension"
        )
    server_environment = dispatch.get("server_environment") if isinstance(
        dispatch, Mapping
    ) else None
    declared_environment = server_environment.get("values") if isinstance(
        server_environment, Mapping
    ) else None
    process_environment = manifest.get("server_process_environment")
    observed_environment = process_environment.get("values") if isinstance(
        process_environment, Mapping
    ) else None
    required_process_environment = {
        **_REQUIRED_SERVER_ENVIRONMENT,
        "PQ_GRIDBOOK_RUNTIME_COMMIT": pin_commit,
        "PQ_GRIDBOOK_RUNTIME_VERSION": pin_version,
        "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256": pin_wheel_sha256,
    }
    if (
        not isinstance(declared_environment, Mapping)
        or not isinstance(process_environment, Mapping)
        or process_environment.get("schema")
        != "prismaquant.server_process_environment/1"
        or process_environment.get("allowlist") != sorted(SERVER_ENV_ALLOWLIST)
        or process_environment.get("consistent") is not True
        or process_environment.get("unreadable_pids") != []
        or observed_environment != declared_environment
        or manifest.get("pq_env") != declared_environment
        or dict(declared_environment) != required_process_environment
    ):
        raise CBPerformanceValidationError(
            f"{label} report environment is not the exact live-server environment"
        )
    listener = manifest.get("listener_binding")
    try:
        parsed_url = urllib.parse.urlsplit(str(server.get("base_url")))
    except ValueError as exc:
        raise CBPerformanceValidationError(f"{label} base URL is invalid") from exc
    if (
        not isinstance(listener, Mapping)
        or listener.get("schema") != "prismaquant.server_listener_binding/1"
        or listener.get("base_url") != server.get("base_url")
        or listener.get("launch_port") != parsed_url.port
        or parsed_url.scheme != "http"
        or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not isinstance(listener.get("listeners"), list)
        or not listener.get("listeners")
    ):
        raise CBPerformanceValidationError(
            f"{label} report base URL is not bound to the actual local listener"
        )
    if manifest.get("performance_stack_fingerprint") != performance_stack_fingerprint(
        manifest
    ):
        raise CBPerformanceValidationError(
            f"{label} performance stack fingerprint is stale"
        )
    host = manifest.get("host_identity")
    if (
        not isinstance(host, Mapping)
        or host.get("hostname") != manifest.get("hostname")
        or not isinstance(host.get("boot_id"), str)
        or not host.get("boot_id")
        or (
            host.get("machine_id_sha256") is not None
            and (
                not isinstance(host.get("machine_id_sha256"), str)
                or len(str(host.get("machine_id_sha256"))) != 64
            )
        )
        or not isinstance(host.get("pid_namespace"), str)
    ):
        raise CBPerformanceValidationError(f"{label} stable host identity is incomplete")
    processes = manifest.get("processes")
    if not isinstance(processes, list) or not processes:
        raise CBPerformanceValidationError(f"{label} serve manifest has no server processes")
    process_ids: list[int] = []
    process_hashes: list[str] = []
    api_server_pids: list[int] = []
    for index, process in enumerate(processes):
        pid = process.get("pid") if isinstance(process, Mapping) else None
        cmdline = process.get("cmdline") if isinstance(process, Mapping) else None
        argv = process.get("argv") if isinstance(process, Mapping) else None
        identity_sha = process.get("identity_sha256") if isinstance(
            process, Mapping
        ) else None
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(cmdline, str)
            or not cmdline
            or not isinstance(argv, list)
            or not argv
            or cmdline != " ".join(str(value) for value in argv)
            or not isinstance(process.get("start_time_ticks"), int)
            or process.get("start_time_ticks", -1) < 0
            or not isinstance(process.get("pid_namespace"), str)
            or not isinstance(process.get("executable"), str)
            or not isinstance(identity_sha, str)
            or len(identity_sha) != 64
        ):
            raise CBPerformanceValidationError(
                f"{label} serve manifest process {index} is malformed"
            )
        process_ids.append(pid)
        if identity_sha != process_identity_sha256(
            process, boot_id=str(host.get("boot_id"))
        ):
            raise CBPerformanceValidationError(
                f"{label} serve process {index} identity digest is stale"
            )
        process_hashes.append(identity_sha)
        if argv == launch_argv:
            api_server_pids.append(pid)
    if (
        len(process_ids) != len(set(process_ids))
        or len(process_hashes) != len(set(process_hashes))
        or len(api_server_pids) != 1
    ):
        raise CBPerformanceValidationError(
            f"{label} serve manifest does not identify one concrete vLLM process set"
        )
    if manifest.get("serve_session_id") != serve_session_fingerprint(manifest):
        raise CBPerformanceValidationError(f"{label} serve session identity is stale")
    environment_rows = process_environment.get("processes")
    readable_pids = process_environment.get("readable_pids")
    if (
        not isinstance(environment_rows, list)
        or not isinstance(readable_pids, list)
        or readable_pids != sorted(process_ids)
        or len(environment_rows) != len(process_ids)
    ):
        raise CBPerformanceValidationError(
            f"{label} server-process environment PID coverage is incomplete"
        )
    environment_pids: list[int] = []
    for row_index, row in enumerate(environment_rows):
        if not isinstance(row, Mapping) or set(row) != {"pid", "values", "sha256"}:
            raise CBPerformanceValidationError(
                f"{label} server-process environment row {row_index} is malformed"
            )
        row_pid = row.get("pid")
        values = row.get("values")
        if (
            isinstance(row_pid, bool)
            or not isinstance(row_pid, int)
            or not isinstance(values, Mapping)
            or dict(values) != required_process_environment
        ):
            raise CBPerformanceValidationError(
                f"{label} server-process environment row {row_index} differs from contract"
            )
        canonical = json.dumps(
            values, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if row.get("sha256") != _digest(canonical):
            raise CBPerformanceValidationError(
                f"{label} server-process environment row {row_index} digest is stale"
            )
        environment_pids.append(row_pid)
    if sorted(environment_pids) != sorted(process_ids) or len(
        environment_pids
    ) != len(set(environment_pids)):
        raise CBPerformanceValidationError(
            f"{label} server-process environment rows do not equal the process census"
        )
    census = manifest.get("listener_census")
    census_rows = census.get("listeners") if isinstance(census, Mapping) else None
    valid_listener_rows = True
    listener_inodes: list[str] = []
    if isinstance(census_rows, list):
        for row in census_rows:
            owners = row.get("pids") if isinstance(row, Mapping) else None
            inode = row.get("socket_inode") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"family", "address", "port", "socket_inode", "pids"}
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
                valid_listener_rows = False
                break
            listener_inodes.append(inode)
    if (
        not isinstance(census, Mapping)
        or census.get("schema") != "prismaquant.server_tcp_listeners/1"
        or census.get("tables_readable") is not True
        or census.get("unreadable_pids") != []
        or not isinstance(census_rows, list)
        or not valid_listener_rows
        or len(listener_inodes) != len(set(listener_inodes))
        or listener.get("listeners")
        != [
            row
            for row in census_rows
            if isinstance(row, Mapping)
            and row.get("port") == listener.get("launch_port")
        ]
        or api_server_pids[0]
        not in {
            pid
            for row in listener.get("listeners", [])
            if isinstance(row, Mapping)
            for pid in row.get("pids", [])
        }
    ):
        raise CBPerformanceValidationError(
            f"{label} listener binding differs from its exact process socket census"
        )
    return {
        "pre_sha256": pre_digest,
        "post_sha256": post_digest,
        "pre_serve_fingerprint": recorded_fingerprint,
        "post_serve_fingerprint": post["serve_fingerprint"],
        "performance_stack_fingerprint": manifest["performance_stack_fingerprint"],
        "serve_session_id": manifest["serve_session_id"],
        "host_boot_id": host["boot_id"],
        "host_machine_id_sha256": host["machine_id_sha256"],
        "gpu_uuid": manifest["gpu_uuid"],
        "process_identity": (
            manifest["serve_session_id"],
            tuple(sorted(process_hashes)),
        ),
        "artifact_identity": (
            expected_binding["model_sha"],
            expected_binding["artifact_inventory_sha256"],
            expected_binding["artifact_bytes"],
        ),
    }


def _metric_values(report: Mapping[str, Any], metric: str, label: str) -> list[float]:
    summary = report.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, Mapping) else None
    row = metrics.get(metric) if isinstance(metrics, Mapping) else None
    values = row.get("values") if isinstance(row, Mapping) else None
    if not isinstance(values, list) or len(values) < 3:
        raise CBPerformanceValidationError(f"{label} lacks block values for {metric}")
    return [
        _finite_positive(value, f"{label}.{metric}.values[{index}]")
        for index, value in enumerate(values)
    ]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cell_ratios(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], *, phase: str, cell_id: str
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    for metric in _PHASE_METRICS[phase]:
        candidate_values = _metric_values(candidate, metric, f"{cell_id} candidate")
        baseline_values = _metric_values(baseline, metric, f"{cell_id} baseline")
        if len(candidate_values) != len(baseline_values):
            raise CBPerformanceValidationError(f"cell {cell_id} has unpaired blocks for {metric}")
        if metric in _THROUGHPUT_METRICS:
            ratios = [left / right for left, right in zip(candidate_values, baseline_values)]
            direction = "candidate/baseline"
        else:
            ratios = [right / left for left, right in zip(candidate_values, baseline_values)]
            direction = "baseline/candidate"
        conservative = _percentile(ratios, 5.0)
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "paired_values": [
                    [left, right]
                    for left, right in zip(candidate_values, baseline_values)
                ],
                "paired_ratios": ratios,
                "median_ratio": statistics.median(ratios),
                "conservative_p05_ratio": conservative,
            }
        )
    return rows, min(row["conservative_p05_ratio"] for row in rows)


def _server_chunked_state(report: Mapping[str, Any], label: str) -> bool:
    metadata = report.get("metadata")
    server = metadata.get("server") if isinstance(metadata, Mapping) else None
    args = server.get("recorded_args") if isinstance(server, Mapping) else None
    if not isinstance(args, list):
        raise CBPerformanceValidationError(f"{label} has no recorded server arguments")
    enabled = "--enable-chunked-prefill" in args
    disabled = "--no-enable-chunked-prefill" in args
    if enabled == disabled:
        raise CBPerformanceValidationError(
            f"{label} must record exactly one chunked-prefill state"
        )
    return enabled


def _validate_coverage(
    manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], reports: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CBPerformanceValidationError("manifest coverage is missing")
    cell_ids = [str(cell.get("id")) for cell in cells]
    if coverage.get("required_cell_ids") != cell_ids:
        raise CBPerformanceValidationError(
            "coverage.required_cell_ids must exactly equal the ordered cell list"
        )
    shipped_max = _positive_int(
        manifest.get("shipped_max_concurrency"), "shipped_max_concurrency"
    )
    if shipped_max < 8:
        raise CBPerformanceValidationError("shipped maximum concurrency cannot be below 8")
    required_concurrency = sorted({1, 2, 4, 8, shipped_max})
    if coverage.get("phases") != list(_PHASES):
        raise CBPerformanceValidationError("coverage phases must be prefill/decode/mixed")
    if coverage.get("concurrencies") != required_concurrency:
        raise CBPerformanceValidationError("coverage concurrency ladder is incomplete")
    if coverage.get("chunked_prefill") != [False, True]:
        raise CBPerformanceValidationError("coverage must include chunked prefill off and on")
    if coverage.get("decode_modes") != ["plain", "shipped"]:
        raise CBPerformanceValidationError("coverage must include plain and shipped decode")
    if coverage.get("nonzero_input_distribution") is not True:
        raise CBPerformanceValidationError("coverage must require a nonzero input distribution")

    observed_phases: set[str] = set()
    observed_concurrency: set[int] = set()
    observed_chunked: set[bool] = set()
    observed_decode_modes: set[str] = set()
    observed_tuples: set[tuple[str, int, bool, str | None]] = set()
    has_distribution = False
    for cell in cells:
        cell_id = str(cell["id"])
        phase = cell.get("phase")
        if phase not in _PHASES:
            raise CBPerformanceValidationError(f"cell {cell_id} has invalid phase")
        candidate, baseline = reports[cell_id]
        workload = candidate["metadata"].get("workload")
        if not isinstance(workload, Mapping):
            raise CBPerformanceValidationError(f"cell {cell_id} has no workload")
        concurrency = _positive_int(
            workload.get("max_concurrency"), f"cell {cell_id} concurrency"
        )
        if cell.get("concurrency") != concurrency:
            raise CBPerformanceValidationError(
                f"cell {cell_id} declared concurrency differs from its report"
            )
        chunked = cell.get("chunked_prefill")
        if not isinstance(chunked, bool):
            raise CBPerformanceValidationError(f"cell {cell_id} chunked_prefill must be boolean")
        if _server_chunked_state(candidate, f"cell {cell_id} candidate") != chunked or _server_chunked_state(
            baseline, f"cell {cell_id} baseline"
        ) != chunked:
            raise CBPerformanceValidationError(f"cell {cell_id} chunked-prefill declaration is false")
        decode_mode = cell.get("decode_mode")
        if phase == "decode":
            if decode_mode not in {"plain", "shipped"}:
                raise CBPerformanceValidationError(f"cell {cell_id} decode mode is invalid")
            speculation = workload.get("speculative_decoding")
            speculative_mode = speculation.get("mode") if isinstance(speculation, Mapping) else None
            if decode_mode == "plain" and (speculative_mode != "off" or concurrency != 1):
                raise CBPerformanceValidationError(
                    f"cell {cell_id} plain decode must be non-speculative batch-1"
                )
            if decode_mode == "shipped" and speculative_mode != "on" and concurrency <= 1:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} shipped decode must exercise speculation or batching"
                )
            observed_decode_modes.add(str(decode_mode))
        else:
            if decode_mode is not None:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} sets decode_mode outside the decode phase"
                )
        output_len = _positive_int(workload.get("output_len"), f"cell {cell_id} output_len")
        if phase == "prefill" and output_len != 1:
            raise CBPerformanceValidationError(f"cell {cell_id} prefill must request one token")
        if phase != "prefill" and output_len <= 1:
            raise CBPerformanceValidationError(f"cell {cell_id} must exercise multi-token decode")
        ratio = workload.get("input_range_ratio")
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and float(ratio) > 0:
            has_distribution = True
        observed_phases.add(str(phase))
        observed_concurrency.add(concurrency)
        observed_chunked.add(chunked)
        cell_tuple = (str(phase), concurrency, chunked, decode_mode)
        if cell_tuple in observed_tuples:
            raise CBPerformanceValidationError(
                f"matrix repeats configuration tuple {cell_tuple}"
            )
        observed_tuples.add(cell_tuple)
    if observed_phases != set(_PHASES):
        raise CBPerformanceValidationError("matrix does not execute every phase")
    if not set(required_concurrency).issubset(observed_concurrency):
        raise CBPerformanceValidationError("matrix does not execute the full concurrency ladder")
    if observed_chunked != {False, True}:
        raise CBPerformanceValidationError("matrix does not execute both chunked-prefill states")
    if observed_decode_modes != {"plain", "shipped"}:
        raise CBPerformanceValidationError("matrix does not execute both decode modes")
    if not has_distribution:
        raise CBPerformanceValidationError("matrix has no nonzero input-length distribution")
    required_tuples: set[tuple[str, int, bool, str | None]] = set()
    for concurrency in required_concurrency:
        for chunked in (False, True):
            required_tuples.add(("prefill", concurrency, chunked, None))
            required_tuples.add(("mixed", concurrency, chunked, None))
            required_tuples.add(("decode", concurrency, chunked, "shipped"))
    for chunked in (False, True):
        required_tuples.add(("decode", 1, chunked, "plain"))
    if observed_tuples != required_tuples:
        missing = sorted(required_tuples - observed_tuples)
        extra = sorted(observed_tuples - required_tuples)
        raise CBPerformanceValidationError(
            "matrix does not equal the predeclared release Cartesian product: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return {
        "phases": list(_PHASES),
        "concurrencies": required_concurrency,
        "chunked_prefill": [False, True],
        "decode_modes": ["plain", "shipped"],
        "nonzero_input_distribution": True,
        "shipped_max_concurrency": shipped_max,
        "configuration_tuple_count": len(required_tuples),
    }


def _validate_native_feasibility(
    manifest: Mapping[str, Any],
    base: Path,
    *,
    byte_budget: int,
    pin_commit: str,
    pin_version: str,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    reference = manifest.get("native_baseline_feasibility")
    if not isinstance(reference, Mapping):
        raise CBPerformanceValidationError(
            "native_baseline_feasibility reference is missing"
        )
    certificate, digest, _ = _read_bound_json(
        base, reference, label="native baseline feasibility certificate"
    )
    try:
        validate_native_baseline_certificate(certificate)
    except (TypeError, ValueError) as exc:
        raise CBPerformanceValidationError(
            f"native baseline feasibility certificate is invalid: {exc}"
        ) from exc
    if certificate.get("schema") != NATIVE_BASELINE_FEASIBILITY_SCHEMA or certificate.get(
        "status"
    ) != "infeasible":
        raise CBPerformanceValidationError(
            "native baseline certificate must prove exact no-CB infeasibility"
        )
    if certificate.get("certificate_sha256") != native_feasibility_certificate_sha256(
        certificate
    ):
        raise CBPerformanceValidationError(
            "native baseline certificate canonical identity is stale"
        )
    if (certificate.get("byte_budget") or {}).get("budget_bytes") != byte_budget:
        raise CBPerformanceValidationError(
            "native baseline infeasibility was not proven at the comparison budget"
        )
    if certificate.get("source_model") != manifest.get("source_model_binding"):
        raise CBPerformanceValidationError(
            "native baseline certificate source model differs from the comparison"
        )
    contract = certificate.get("contract")
    if not isinstance(contract, Mapping):
        raise CBPerformanceValidationError("native baseline certificate contract is missing")
    model = contract.get("model_profile")
    serving = contract.get("serving_profile")
    lane = contract.get("lane")
    runtime = contract.get("gridbook_runtime_pin")
    if (
        not isinstance(model, Mapping)
        or model.get("id") != "deepseek_v4"
        or not isinstance(serving, Mapping)
        or serving.get("id") != "nvfp4_cb"
        or not isinstance(lane, Mapping)
        or lane.get("id") != "nvfp4_cb"
        or lane.get("export_container") != "nvfp4_cb"
        or not isinstance(runtime, Mapping)
        or runtime.get("commit") != pin_commit
        or runtime.get("version") != pin_version
        or runtime.get("version_is_release") is not True
    ):
        raise CBPerformanceValidationError(
            "native baseline feasibility certificate binds the wrong DSv4/Gridbook contract"
        )
    coverage = certificate.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CBPerformanceValidationError(
            "native feasibility certificate has no exact serving-unit census"
        )
    body_count = coverage.get("body_serving_unit_count")
    construction_count = coverage.get("construction_unit_count")
    if (
        isinstance(body_count, bool)
        or not isinstance(body_count, int)
        or body_count <= 0
        or isinstance(construction_count, bool)
        or not isinstance(construction_count, int)
        or construction_count < 0
    ):
        raise CBPerformanceValidationError(
            "native feasibility certificate has no exact serving-unit census"
        )
    units = certificate.get("units")
    if not isinstance(units, list):
        raise CBPerformanceValidationError(
            "native feasibility certificate has no serving-unit ledger"
        )
    names = frozenset(
        str(row.get("name"))
        for row in units
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    )
    if len(names) != body_count + construction_count:
        raise CBPerformanceValidationError(
            "native feasibility certificate serving-unit ledger is incomplete"
        )
    return digest, tuple(dict(row) for row in units if isinstance(row, Mapping))


def _validate_displaced_container(
    manifest: Mapping[str, Any],
    base: Path,
    *,
    byte_budget: int,
    candidate_source_identity: Mapping[str, Any],
    baselines: Mapping[str, Any],
    pin_commit: str,
    pin_version: str,
    candidate_root: Path,
    candidate_model_sha: str,
    candidate_inventory_sha256: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    declaration = manifest.get("displaced_container")
    if not isinstance(declaration, Mapping):
        raise CBPerformanceValidationError("displaced_container declaration is missing")
    reason = declaration.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CBPerformanceValidationError("displaced_container reason must be explicit")
    mechanism = declaration.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise CBPerformanceValidationError(
            "displaced_container mechanism must be explicit"
        )
    artifact_root = _reference(
        base, declaration.get("artifact_dir"), "displaced_container.artifact_dir"
    ).resolve()
    if artifact_root == candidate_root.resolve():
        raise CBPerformanceValidationError(
            "displaced container aliases the candidate artifact"
        )
    if not artifact_root.is_dir():
        raise CBPerformanceValidationError(
            f"displaced container artifact directory does not exist: {artifact_root}"
        )
    inventory, inventory_digest, quant_config = _artifact_inventory(artifact_root)
    artifact_bytes = _positive_int(
        inventory.get("export_directory_bytes"), "displaced container artifact bytes"
    )
    if artifact_bytes > byte_budget or inventory.get(
        "whole_artifact_budget_bytes"
    ) != byte_budget:
        raise CBPerformanceValidationError(
            "displaced container is not bound to and within the comparison budget"
        )
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CBPerformanceValidationError("displaced container provenance is missing")
    source_identity = provenance.get("source_model_identity")
    if source_identity != candidate_source_identity:
        raise CBPerformanceValidationError(
            "candidate and displaced container do not bind the same source model"
        )
    assignment_sha = _require_sha(
        provenance.get("assignment_sha256"),
        "displaced container provenance.assignment_sha256",
    )
    weight_manifest = provenance.get("weight_content_manifest")
    if not isinstance(weight_manifest, Mapping):
        raise CBPerformanceValidationError(
            "displaced container lacks an immutable weight content manifest; "
            "re-export it with the current exporter"
        )
    weight_manifest_digest = _digest(
        json.dumps(
            weight_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    shipcard_path = artifact_root / "shipcard.json"
    if "shipcard.json" not in inventory["file_bytes"] or not shipcard_path.is_file():
        raise CBPerformanceValidationError(
            "displaced container has no current shipcard; re-export before using it"
        )
    baseline_card = load_shipcard(shipcard_path)
    baseline_model_sha = compute_model_sha(artifact_root)
    if baseline_model_sha == candidate_model_sha:
        raise CBPerformanceValidationError(
            "displaced container model identity is identical to the candidate"
        )
    if baseline_card.get("model_sha") != baseline_model_sha:
        raise CBPerformanceValidationError(
            "displaced container shipcard model_sha differs from the artifact"
        )
    baseline_build = baseline_card.get("build")
    if (
        not isinstance(baseline_build, Mapping)
        or baseline_build.get("quant_method") != "gridbook"
    ):
        raise CBPerformanceValidationError(
            "displaced container shipcard is not a Gridbook build receipt"
        )
    layer_config_sha = _require_sha(
        baseline_build.get("layer_config_sha"),
        "displaced container shipcard build.layer_config_sha",
    )
    if not isinstance(baseline_card.get("weight_stat_attestation"), Mapping):
        raise CBPerformanceValidationError(
            "displaced container lacks a current weight-stat attestation; "
            "re-export or re-attest it"
        )
    try:
        assert_weight_stat_attestation(baseline_card, artifact_root)
    except ValueError as exc:
        raise CBPerformanceValidationError(
            f"displaced container weight attestation is stale: {exc}"
        ) from exc
    endpoint_problems = verify_shipcard(
        baseline_card,
        model_dir=artifact_root,
        required=("native_export.eager", "native_export.graph"),
    )
    if endpoint_problems:
        raise CBPerformanceValidationError(
            "displaced container is not endpoint-eligible: "
            + "; ".join(endpoint_problems)
        )
    assignment_reference = declaration.get("assignment_receipt")
    if not isinstance(assignment_reference, Mapping):
        raise CBPerformanceValidationError(
            "displaced container has no digest-bound assignment receipt"
        )
    assignment_receipt, assignment_receipt_digest, _ = _read_bound_json(
        base, assignment_reference, label="displaced assignment receipt"
    )
    cost_currency = assignment_receipt.get("cost_currency")
    model_id = declaration.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise CBPerformanceValidationError("displaced_container model_id is missing")
    if (
        assignment_receipt.get("schema")
        != "prismaquant.displaced_assignment/1"
        or assignment_receipt.get("status") != "eligible"
        or assignment_receipt.get("mechanism") != mechanism
        or assignment_receipt.get("reason") != reason
        or assignment_receipt.get("model_id") != model_id
        or assignment_receipt.get("model_sha") != baseline_model_sha
        or assignment_receipt.get("assignment_sha256") != assignment_sha
        or assignment_receipt.get("artifact_inventory_sha256") != inventory_digest
        or assignment_receipt.get("artifact_bytes") != artifact_bytes
        or assignment_receipt.get("byte_budget") != byte_budget
        or assignment_receipt.get("source_model_identity")
        != candidate_source_identity
        or assignment_receipt.get("layer_config_sha256") != layer_config_sha
        or not isinstance(cost_currency, str)
        or not cost_currency.strip()
    ):
        raise CBPerformanceValidationError(
            "displaced assignment receipt does not bind the exact "
            "artifact/source/budget/mechanism"
        )
    identity = {
        "model_id": model_id,
        "model_sha": baseline_model_sha,
        "artifact_inventory_sha256": inventory_digest,
        "artifact_bytes": artifact_bytes,
    }
    for key, expected in {
        **identity,
        "assignment_sha256": assignment_sha,
    }.items():
        if declaration.get(key) != expected:
            raise CBPerformanceValidationError(
                f"displaced_container {key} differs from the artifact"
            )
    for phase in _PHASES:
        if baselines.get(phase) != identity:
            raise CBPerformanceValidationError(
                f"{phase} baseline is not the exact displaced container identity"
            )
    endpoint_records = {
        slot: hashlib.sha256(
            json.dumps(
                baseline_card["slots"][slot],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for slot in ("native_export.eager", "native_export.graph")
    }
    eligibility = {
        "schema": DISPLACED_CONTAINER_SCHEMA,
        "status": "eligible",
        "reason": reason,
        "mechanism": mechanism,
        **identity,
        "assignment_sha256": assignment_sha,
        "layer_config_sha256": layer_config_sha,
        "cost_currency": cost_currency,
        "byte_budget": byte_budget,
        "source_model_identity": dict(candidate_source_identity),
        "weight_content_manifest_sha256": weight_manifest_digest,
        "shipcard_sha256": _digest(shipcard_path.read_bytes()),
        "endpoint_record_sha256": endpoint_records,
        "assignment_receipt_sha256": assignment_receipt_digest,
        "gridbook_runtime": {"commit": pin_commit, "version": pin_version},
    }
    eligibility_digest = _digest(
        json.dumps(
            eligibility,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return eligibility_digest, eligibility, quant_config


def _validate_telemetry(
    manifest: Mapping[str, Any],
    base: Path,
    *,
    cell_ids: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    baselines: Mapping[str, Any],
    pin_commit: str,
    pin_version: str,
) -> list[dict[str, Any]]:
    contract = manifest.get("telemetry_contract")
    if not isinstance(contract, Mapping):
        raise CBPerformanceValidationError("telemetry contract is missing")
    if contract.get("routing_per_layer_per_step") is not True or contract.get(
        "expert_occupancy"
    ) is not True or contract.get("active_experts") is not True:
        raise CBPerformanceValidationError("routing/occupancy/active-expert telemetry is incomplete")
    stages = contract.get("grouped_operator_includes")
    if not isinstance(stages, list) or set(stages) != _GROUPED_OPERATOR_STAGES:
        raise CBPerformanceValidationError("grouped-MoE telemetry does not cover the whole operator")
    attachments = manifest.get("telemetry")
    if not isinstance(attachments, list):
        raise CBPerformanceValidationError("telemetry attachments are missing")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    parsed: dict[
        str, dict[str, dict[str, dict[int, dict[str, Mapping[str, Any]]]]]
    ] = {}
    phase_by_cell = {str(cell["id"]): str(cell["phase"]) for cell in cells}
    for index, attachment in enumerate(attachments):
        if not isinstance(attachment, Mapping):
            raise CBPerformanceValidationError(f"telemetry[{index}] is not an object")
        kind = attachment.get("kind")
        if kind in seen or kind not in _REQUIRED_TELEMETRY:
            raise CBPerformanceValidationError(f"telemetry[{index}] has invalid kind {kind!r}")
        seen.add(str(kind))
        covered = attachment.get("cell_ids")
        if covered != list(cell_ids):
            raise CBPerformanceValidationError(
                f"telemetry {kind} must cover the entire ordered comparison matrix"
            )
        path = _reference(base, attachment.get("path"), f"telemetry[{index}].path")
        expected = _require_sha(attachment.get("sha256"), f"telemetry[{index}].sha256")
        try:
            payload, raw = _load_json(path, label=f"telemetry {kind}")
        except CBPerformanceValidationError:
            raise
        actual = _digest(raw)
        if actual != expected:
            raise CBPerformanceValidationError(f"telemetry {kind} SHA-256 mismatch")
        expected_header = {
            "schema": TELEMETRY_SCHEMA,
            "kind": kind,
            "gridbook_runtime": {"commit": pin_commit, "version": pin_version},
            "cell_ids": list(cell_ids),
        }
        for key, value in expected_header.items():
            if payload.get(key) != value:
                raise CBPerformanceValidationError(
                    f"telemetry {kind} {key} differs from the comparison contract"
                )
        arms = payload.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {"candidate", "baseline"}:
            raise CBPerformanceValidationError(
                f"telemetry {kind} must contain candidate and baseline arms"
            )
        kind_rows: dict[str, dict[str, dict[int, dict[str, Mapping[str, Any]]]]] = {}
        for arm in ("candidate", "baseline"):
            arm_payload = arms.get(arm)
            arm_cells = arm_payload.get("cells") if isinstance(
                arm_payload, Mapping
            ) else None
            if not isinstance(arm_cells, list) or [
                row.get("cell_id") if isinstance(row, Mapping) else None
                for row in arm_cells
            ] != list(cell_ids):
                raise CBPerformanceValidationError(
                    f"telemetry {kind} {arm} cells do not exactly cover the matrix"
                )
            kind_rows[arm] = {}
            for cell_row in arm_cells:
                assert isinstance(cell_row, Mapping)
                cell_id = str(cell_row["cell_id"])
                expected_identity = candidate if arm == "candidate" else baselines[
                    phase_by_cell[cell_id]
                ]
                identity_fields = (
                    "model_id",
                    "model_sha",
                    "artifact_inventory_sha256",
                    "artifact_bytes",
                )
                if any(
                    cell_row.get(field) != expected_identity.get(field)
                    for field in identity_fields
                ):
                    raise CBPerformanceValidationError(
                        f"telemetry {kind} {arm} {cell_id} model identity differs"
                    )
                layers = cell_row.get("layers")
                if not isinstance(layers, list) or [
                    layer.get("layer_id") if isinstance(layer, Mapping) else None
                    for layer in layers
                ] != list(range(DSV4_NUM_HIDDEN_LAYERS)):
                    raise CBPerformanceValidationError(
                        f"telemetry {kind} {arm} {cell_id} does not cover all DSv4 layers"
                    )
                kind_rows[arm][cell_id] = {}
                for layer in layers:
                    assert isinstance(layer, Mapping)
                    layer_id = int(layer["layer_id"])
                    steps = layer.get("steps")
                    if not isinstance(steps, list) or not steps:
                        raise CBPerformanceValidationError(
                            f"telemetry {kind} {arm} {cell_id} layer {layer_id} has no steps"
                        )
                    step_rows: dict[str, Mapping[str, Any]] = {}
                    for step in steps:
                        if not isinstance(step, Mapping):
                            raise CBPerformanceValidationError(
                                f"telemetry {kind} {arm} {cell_id} layer {layer_id} has malformed step"
                            )
                        step_id = step.get("step_id")
                        if not isinstance(step_id, str) or not step_id or step_id in step_rows:
                            raise CBPerformanceValidationError(
                                f"telemetry {kind} {arm} {cell_id} layer {layer_id} has invalid step_id"
                            )
                        if not isinstance(step.get("run_label"), str) or not step.get("run_label"):
                            raise CBPerformanceValidationError(
                                f"telemetry {kind} {arm} {cell_id} layer {layer_id} lacks run label"
                            )
                        if isinstance(step.get("block_seed"), bool) or not isinstance(
                            step.get("block_seed"), int
                        ):
                            raise CBPerformanceValidationError(
                                f"telemetry {kind} {arm} {cell_id} layer {layer_id} lacks block seed"
                            )
                        _iso_datetime(
                            step.get("timestamp"),
                            f"telemetry {kind} {arm} {cell_id} layer {layer_id} timestamp",
                        )
                        if kind == "routing_per_layer_per_step":
                            histogram = step.get("expert_histogram")
                            if not isinstance(histogram, Mapping) or not histogram:
                                raise CBPerformanceValidationError(
                                    f"routing telemetry {arm} {cell_id} layer {layer_id} is empty"
                                )
                            for expert, count in histogram.items():
                                if (
                                    not isinstance(expert, str)
                                    or not expert.isdigit()
                                    or not 0 <= int(expert) < DSV4_NUM_ROUTED_EXPERTS
                                    or isinstance(count, bool)
                                    or not isinstance(count, int)
                                    or count <= 0
                                ):
                                    raise CBPerformanceValidationError(
                                        f"routing telemetry {arm} {cell_id} layer {layer_id} has invalid expert count"
                                    )
                        elif kind == "expert_occupancy":
                            occupancy = step.get("expert_occupancy")
                            if not isinstance(occupancy, Mapping) or not occupancy:
                                raise CBPerformanceValidationError(
                                    f"occupancy telemetry {arm} {cell_id} layer {layer_id} is empty"
                                )
                            for expert, value in occupancy.items():
                                if (
                                    not isinstance(expert, str)
                                    or not expert.isdigit()
                                    or not 0 <= int(expert) < DSV4_NUM_ROUTED_EXPERTS
                                    or isinstance(value, bool)
                                    or not isinstance(value, (int, float))
                                    or not math.isfinite(float(value))
                                    or not 0 < float(value) <= 1
                                ):
                                    raise CBPerformanceValidationError(
                                        f"occupancy telemetry {arm} {cell_id} layer {layer_id} is invalid"
                                    )
                        elif kind == "active_experts":
                            active = step.get("active_experts")
                            if (
                                not isinstance(active, list)
                                or not active
                                or len(active) != len(set(active))
                                or any(
                                    isinstance(expert, bool)
                                    or not isinstance(expert, int)
                                    or not 0 <= expert < DSV4_NUM_ROUTED_EXPERTS
                                    for expert in active
                                )
                            ):
                                raise CBPerformanceValidationError(
                                    f"active-expert telemetry {arm} {cell_id} layer {layer_id} is invalid"
                                )
                        else:
                            stages_ms = step.get("stages_ms")
                            if (
                                step.get("fallback_state") != "none"
                                or not isinstance(stages_ms, Mapping)
                                or set(stages_ms) != _GROUPED_OPERATOR_STAGES
                                or any(
                                    isinstance(value, bool)
                                    or not isinstance(value, (int, float))
                                    or not math.isfinite(float(value))
                                    or float(value) <= 0
                                    for value in stages_ms.values()
                                )
                                or isinstance(step.get("routed_tokens"), bool)
                                or not isinstance(step.get("routed_tokens"), int)
                                or step.get("routed_tokens", 0) <= 0
                            ):
                                raise CBPerformanceValidationError(
                                    f"grouped-MoE whole-operator telemetry {arm} {cell_id} layer {layer_id} is invalid"
                                )
                        step_rows[step_id] = step
                    kind_rows[arm][cell_id][layer_id] = step_rows
        parsed[str(kind)] = kind_rows
        result.append({"kind": kind, "sha256": actual, "cell_ids": list(cell_ids)})
    if seen != _REQUIRED_TELEMETRY:
        raise CBPerformanceValidationError("telemetry attachments do not cover all required classes")
    for arm in ("candidate", "baseline"):
        for cell_id in cell_ids:
            for layer_id in range(DSV4_NUM_HIDDEN_LAYERS):
                kind_steps = {
                    kind: parsed[kind][arm][cell_id][layer_id]
                    for kind in _REQUIRED_TELEMETRY
                }
                step_ids = {
                    kind: set(steps) for kind, steps in kind_steps.items()
                }
                if len({frozenset(value) for value in step_ids.values()}) != 1:
                    raise CBPerformanceValidationError(
                        f"telemetry step coverage differs across kinds for {arm} {cell_id} layer {layer_id}"
                    )
                for step_id in next(iter(step_ids.values())):
                    routing = kind_steps["routing_per_layer_per_step"][step_id]
                    occupancy = kind_steps["expert_occupancy"][step_id]
                    active = kind_steps["active_experts"][step_id]
                    grouped = kind_steps["grouped_moe_whole_operator"][step_id]
                    for field in ("run_label", "block_seed", "timestamp"):
                        if len(
                            {
                                step.get(field)
                                for step in (routing, occupancy, active, grouped)
                            }
                        ) != 1:
                            raise CBPerformanceValidationError(
                                f"telemetry {field} differs across kinds for "
                                f"{arm} {cell_id} layer {layer_id} step {step_id}"
                            )
                    histogram = routing["expert_histogram"]
                    occupied = occupancy["expert_occupancy"]
                    active_ids = {str(expert) for expert in active["active_experts"]}
                    if set(histogram) != set(occupied) or set(histogram) != active_ids:
                        raise CBPerformanceValidationError(
                            f"routing/occupancy/active-expert sets disagree for {arm} {cell_id} layer {layer_id} step {step_id}"
                        )
                    if sum(histogram.values()) != grouped["routed_tokens"]:
                        raise CBPerformanceValidationError(
                            f"grouped routed-token count disagrees for {arm} {cell_id} layer {layer_id} step {step_id}"
                        )
                    routed_tokens = sum(histogram.values())
                    if any(
                        not math.isclose(
                            float(occupied[expert]),
                            float(count) / routed_tokens,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        for expert, count in histogram.items()
                    ) or not math.isclose(
                        sum(float(value) for value in occupied.values()),
                        1.0,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    ):
                        raise CBPerformanceValidationError(
                            f"expert occupancy does not reconcile with routing counts for "
                            f"{arm} {cell_id} layer {layer_id} step {step_id}"
                        )
    return sorted(result, key=lambda row: str(row["kind"]))


def validate_cb_performance(
    shipcard_path: str | os.PathLike,
    manifest_path: str | os.PathLike,
    *,
    fill: bool = True,
) -> dict[str, Any]:
    """Validate a paired matrix and return its shipcard-ready passing record."""
    card_path = Path(shipcard_path)
    root = card_path.resolve().parent
    card = load_shipcard(card_path)
    if (card.get("build") or {}).get("quant_method") != "gridbook":
        raise CBPerformanceValidationError("performance parity gate requires a Gridbook CB shipcard")
    model_sha = compute_model_sha(root)
    if card.get("model_sha") != model_sha:
        raise CBPerformanceValidationError("candidate shipcard model_sha differs from the artifact")
    inventory, inventory_digest, candidate_quant_config = _artifact_inventory(root)

    manifest_file = Path(manifest_path)
    manifest, manifest_raw = _load_json(manifest_file, label="comparison manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CBPerformanceValidationError(f"unsupported comparison manifest schema {manifest.get('schema')!r}")
    base = manifest_file.resolve().parent
    pin = _exact_gridbook_runtime_pin()
    producer_git = git_provenance()
    manifest_runtime = manifest.get("prismaquant_runtime")
    if (
        not isinstance(manifest_runtime, Mapping)
        or manifest_runtime.get("validator") != TOOL
        or manifest_runtime.get("commit") != producer_git.get("commit")
        or producer_git.get("dirty") is not False
    ):
        raise CBPerformanceValidationError(
            "comparison does not bind this exact clean PrismaQuant validator commit"
        )
    manifest_pin = manifest.get("gridbook_runtime")
    if manifest_pin != {"commit": pin.commit, "version": pin.version} or not pin.version_is_release:
        raise CBPerformanceValidationError("comparison manifest does not bind the released Gridbook pin")
    candidate_identity = manifest.get("candidate")
    if not isinstance(candidate_identity, Mapping):
        raise CBPerformanceValidationError("comparison manifest candidate identity is missing")
    byte_budget = _positive_int(manifest.get("byte_budget"), "byte_budget")
    candidate_bytes = _positive_int(
        inventory.get("export_directory_bytes"), "candidate inventory bytes"
    )
    expected_candidate = {
        "model_sha": model_sha,
        "artifact_inventory_sha256": inventory_digest,
        "artifact_bytes": candidate_bytes,
    }
    for key, expected in expected_candidate.items():
        if candidate_identity.get(key) != expected:
            raise CBPerformanceValidationError(f"candidate {key} differs from the artifact")
    if candidate_bytes > byte_budget:
        raise CBPerformanceValidationError("candidate artifact exceeds the matched byte budget")
    if inventory.get("whole_artifact_budget_bytes") != byte_budget:
        raise CBPerformanceValidationError(
            "candidate export is not bound to the exact comparison budget"
        )

    candidate_provenance = candidate_quant_config.get("provenance")
    candidate_source_identity = candidate_provenance.get(
        "source_model_identity"
    ) if isinstance(candidate_provenance, Mapping) else None
    if (
        not isinstance(candidate_source_identity, Mapping)
        or candidate_source_identity.get("schema")
        != "prismaquant.streamed_model.identity.v1"
        or (
            candidate_source_identity.get("resolved_commit") is not None
            and (
                not isinstance(candidate_source_identity.get("resolved_commit"), str)
                or not candidate_source_identity.get("resolved_commit")
            )
        )
        or isinstance(candidate_source_identity.get("checkpoint_shards"), bool)
        or not isinstance(candidate_source_identity.get("checkpoint_shards"), int)
        or candidate_source_identity.get("checkpoint_shards", 0) <= 0
        or isinstance(candidate_source_identity.get("checkpoint_tensors"), bool)
        or not isinstance(candidate_source_identity.get("checkpoint_tensors"), int)
        or candidate_source_identity.get("checkpoint_tensors", 0) <= 0
    ):
        raise CBPerformanceValidationError(
            "candidate lacks a complete production source-model identity"
        )
    _require_sha(
        candidate_source_identity.get("content_sha256"),
        "candidate source_model_identity.content_sha256",
    )

    parity_floor = _finite_positive(manifest.get("parity_floor"), "parity_floor")
    if parity_floor > 1.0:
        raise CBPerformanceValidationError("parity_floor cannot exceed strict parity 1.0")
    tolerance = manifest.get("predeclared_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0 <= float(tolerance) <= MAX_PARITY_TOLERANCE
    ):
        raise CBPerformanceValidationError("predeclared_tolerance must be finite and nonnegative")
    if not math.isclose(parity_floor, 1.0 - float(tolerance), abs_tol=1e-12, rel_tol=0):
        raise CBPerformanceValidationError("parity_floor must equal 1 - predeclared_tolerance")
    if float(tolerance) > 0 and (
        not isinstance(manifest.get("tolerance_rationale"), str)
        or not manifest.get("tolerance_rationale", "").strip()
    ):
        raise CBPerformanceValidationError("a nonzero tolerance requires a predeclared rationale")
    predeclared_at = _iso_datetime(manifest.get("predeclared_at"), "predeclared_at")

    baselines = manifest.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(_PHASES):
        raise CBPerformanceValidationError("manifest must declare a phase-specific baseline")
    native_feasibility_digest, certified_serving_units = _validate_native_feasibility(
        manifest,
        base,
        byte_budget=byte_budget,
        pin_commit=pin.commit,
        pin_version=pin.version,
    )
    native_source_binding = manifest.get("source_model_binding")
    if (
        not isinstance(native_source_binding, Mapping)
        or native_source_binding.get("content_sha256")
        != candidate_source_identity.get("content_sha256")
    ):
        raise CBPerformanceValidationError(
            "native infeasibility and candidate identities do not bind the same source content"
        )
    (
        displaced_digest,
        displaced_eligibility,
        displaced_quant_config,
    ) = _validate_displaced_container(
        manifest,
        base,
        byte_budget=byte_budget,
        candidate_source_identity=candidate_source_identity,
        baselines=baselines,
        pin_commit=pin.commit,
        pin_version=pin.version,
        candidate_root=root,
        candidate_model_sha=model_sha,
        candidate_inventory_sha256=inventory_digest,
    )
    expected_candidate_routes = _derive_expected_execution_assignments(
        candidate_quant_config,
        certified_serving_units,
        label="candidate artifact",
    )
    expected_baseline_routes = _derive_expected_execution_assignments(
        displaced_quant_config,
        certified_serving_units,
        label="displaced baseline artifact",
    )

    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise CBPerformanceValidationError("comparison matrix is empty")
    cells: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    report_pairs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    report_evidence: list[dict[str, str]] = []
    seen_report_digests: set[str] = set()
    process_artifacts: dict[tuple[str, tuple[int, ...]], tuple[object, ...]] = {}
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            raise CBPerformanceValidationError(f"cells[{index}] is not an object")
        cell_id = raw_cell.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen_ids:
            raise CBPerformanceValidationError(f"cells[{index}] has invalid or duplicate id")
        seen_ids.add(cell_id)
        phase = raw_cell.get("phase")
        if phase not in _PHASES:
            raise CBPerformanceValidationError(f"cell {cell_id} has invalid phase")
        candidate_ref = raw_cell.get("candidate_report")
        baseline_ref = raw_cell.get("baseline_report")
        if not isinstance(candidate_ref, Mapping) or not isinstance(baseline_ref, Mapping):
            raise CBPerformanceValidationError(f"cell {cell_id} lacks paired report references")
        candidate_report, candidate_digest, _ = _read_bound_json(
            base, candidate_ref, label=f"cell {cell_id} candidate report"
        )
        baseline_report, baseline_digest, _ = _read_bound_json(
            base, baseline_ref, label=f"cell {cell_id} baseline report"
        )
        for report_digest in (candidate_digest, baseline_digest):
            if report_digest in seen_report_digests:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} reuses a benchmark report from another arm/cell"
                )
            seen_report_digests.add(report_digest)
        _validate_report(
            candidate_report,
            label=f"cell {cell_id} candidate report",
            pin_commit=pin.commit,
            pin_version=pin.version,
            pin_wheel_sha256=pin.wheel_sha256,
        )
        _validate_report(
            baseline_report,
            label=f"cell {cell_id} baseline report",
            pin_commit=pin.commit,
            pin_version=pin.version,
            pin_wheel_sha256=pin.wheel_sha256,
        )
        _validate_execution_routes(
            candidate_report,
            expected_candidate_routes,
            label=f"cell {cell_id} candidate",
        )
        _validate_execution_routes(
            baseline_report,
            expected_baseline_routes,
            label=f"cell {cell_id} baseline",
        )
        for report_label, report in (("candidate", candidate_report), ("baseline", baseline_report)):
            started_at = _iso_datetime(report.get("started_at"), f"cell {cell_id} {report_label}.started_at")
            if started_at <= predeclared_at:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} {report_label} predates the comparison declaration"
                )
        _matched_metadata(
            candidate_report,
            baseline_report,
            raw_cell.get("allowed_arm_difference_paths", []),
            cell_id=cell_id,
        )
        _validate_report_artifact(
            candidate_report,
            candidate_identity,
            byte_budget=byte_budget,
            label=f"cell {cell_id} candidate",
        )
        candidate_server_binding = _load_performance_serve_manifests(
            candidate_report,
            candidate_identity,
            evidence_base=base,
            attestation_references=raw_cell.get(
                "candidate_serve_attestations", {}
            ),
            pin_repository=pin.repository,
            pin_commit=pin.commit,
            pin_version=pin.version,
            pin_wheel_sha256=pin.wheel_sha256,
            label=f"cell {cell_id} candidate",
        )
        baseline_identity = baselines[str(phase)]
        if not isinstance(baseline_identity, Mapping):
            raise CBPerformanceValidationError(f"{phase} baseline identity is malformed")
        _validate_report_artifact(
            baseline_report,
            baseline_identity,
            byte_budget=byte_budget,
            label=f"cell {cell_id} baseline",
        )
        baseline_server_binding = _load_performance_serve_manifests(
            baseline_report,
            baseline_identity,
            evidence_base=base,
            attestation_references=raw_cell.get(
                "baseline_serve_attestations", {}
            ),
            pin_repository=pin.repository,
            pin_commit=pin.commit,
            pin_version=pin.version,
            pin_wheel_sha256=pin.wheel_sha256,
            label=f"cell {cell_id} baseline",
        )
        if {
            candidate_server_binding["pre_sha256"],
            candidate_server_binding["post_sha256"],
        } & {
            baseline_server_binding["pre_sha256"],
            baseline_server_binding["post_sha256"],
        }:
            raise CBPerformanceValidationError(
                f"cell {cell_id} candidate and baseline reuse one serve manifest"
            )
        if (
            candidate_server_binding["performance_stack_fingerprint"]
            != baseline_server_binding["performance_stack_fingerprint"]
        ):
            raise CBPerformanceValidationError(
                f"cell {cell_id} candidate and baseline serving stacks differ"
            )
        for identity_key in (
            "host_boot_id",
            "gpu_uuid",
        ):
            if candidate_server_binding[identity_key] != baseline_server_binding[
                identity_key
            ]:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} candidate and baseline do not run in the "
                    f"same Spark host session ({identity_key})"
                )
        for arm, binding in (
            ("candidate", candidate_server_binding),
            ("baseline", baseline_server_binding),
        ):
            process_identity = binding["process_identity"]
            artifact_identity = binding["artifact_identity"]
            assert isinstance(process_identity, tuple)
            assert isinstance(artifact_identity, tuple)
            previous = process_artifacts.setdefault(process_identity, artifact_identity)
            if previous != artifact_identity:
                raise CBPerformanceValidationError(
                    f"cell {cell_id} {arm} reuses one live server process identity "
                    "for different candidate/baseline artifacts"
                )
        report_pairs[cell_id] = (candidate_report, baseline_report)
        report_evidence.append(
            {
                "cell_id": cell_id,
                "candidate_sha256": candidate_digest,
                "baseline_sha256": baseline_digest,
                "candidate_pre_serve_manifest_sha256": candidate_server_binding[
                    "pre_sha256"
                ],
                "candidate_post_serve_manifest_sha256": candidate_server_binding[
                    "post_sha256"
                ],
                "baseline_pre_serve_manifest_sha256": baseline_server_binding[
                    "pre_sha256"
                ],
                "baseline_post_serve_manifest_sha256": baseline_server_binding[
                    "post_sha256"
                ],
                "candidate_serve_session_id": candidate_server_binding[
                    "serve_session_id"
                ],
                "baseline_serve_session_id": baseline_server_binding[
                    "serve_session_id"
                ],
                "performance_stack_fingerprint": candidate_server_binding[
                    "performance_stack_fingerprint"
                ],
            }
        )
        cells.append(raw_cell)

    coverage = _validate_coverage(manifest, cells, report_pairs)
    telemetry = _validate_telemetry(
        manifest,
        base,
        cell_ids=[str(cell["id"]) for cell in cells],
        cells=cells,
        candidate=candidate_identity,
        baselines=baselines,
        pin_commit=pin.commit,
        pin_version=pin.version,
    )
    verdicts: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["id"])
        ratios, conservative = _cell_ratios(
            *report_pairs[cell_id], phase=str(cell["phase"]), cell_id=cell_id
        )
        verdicts.append(
            {
                "id": cell_id,
                "phase": cell["phase"],
                "concurrency": cell["concurrency"],
                "chunked_prefill": cell["chunked_prefill"],
                "decode_mode": cell.get("decode_mode"),
                "metrics": ratios,
                "min_conservative_ratio": conservative,
                "passed": conservative >= parity_floor,
            }
        )
    minimum = min(float(cell["min_conservative_ratio"]) for cell in verdicts)
    if minimum < parity_floor:
        failures = [cell["id"] for cell in verdicts if not cell["passed"]]
        raise CBPerformanceValidationError(
            f"paired performance parity failed in cells {failures}: min={minimum:.6f}, floor={parity_floor:.6f}"
        )
    canonical_matrix = json.dumps(
        verdicts, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    matrix_digest = _digest(canonical_matrix)
    manifest_digest = _digest(manifest_raw)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "comparison_manifest_sha256": manifest_digest,
        "paired_reports": report_evidence,
        "telemetry_sha256": telemetry,
        "displaced_container": displaced_eligibility,
        "displaced_container_eligibility_sha256": displaced_digest,
        "native_baseline_feasibility_sha256": native_feasibility_digest,
    }
    metrics = {
        "schema": RESULT_SCHEMA,
        "gridbook_runtime_commit": pin.commit,
        "gridbook_runtime_version": pin.version,
        "prismaquant_runtime_commit": producer_git["commit"],
        "byte_budget": byte_budget,
        "candidate_artifact_bytes": candidate_bytes,
        "candidate_inventory_sha256": inventory_digest,
        "matrix_digest": matrix_digest,
        "cell_verdicts": verdicts,
        "cell_count": len(verdicts),
        "coverage": coverage,
        "min_conservative_ratio": minimum,
        "parity_floor": parity_floor,
        "predeclared_tolerance": float(tolerance),
        "tolerance_rationale": manifest.get("tolerance_rationale"),
        "predeclared_at": manifest.get("predeclared_at"),
        "displaced_container_eligibility_sha256": displaced_digest,
        "displaced_container_model_sha": displaced_eligibility["model_sha"],
        "displaced_container_inventory_sha256": displaced_eligibility[
            "artifact_inventory_sha256"
        ],
        "displaced_container_artifact_bytes": displaced_eligibility[
            "artifact_bytes"
        ],
        "displaced_container_assignment_sha256": displaced_eligibility[
            "assignment_sha256"
        ],
        "displaced_container_reason": displaced_eligibility["reason"],
        "native_baseline_feasibility_sha256": native_feasibility_digest,
        "server_environment_contract": {
            "schema": GRIDBOOK_ENVIRONMENT_SCHEMA,
            "profile": "matched_budget_performance",
            "base_profile": "canonical_gold",
            "overrides": {"PRISMAQUANT_PRELOAD_FUSED": "1"},
            "environment": dict(_PERFORMANCE_SERVER_ENVIRONMENT),
        },
    }
    record: dict[str, Any] = {
        "slot": SLOT,
        "tool": TOOL,
        "filled_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": True,
        "model_sha": model_sha,
        "spec_decode_detected": None,
        "serve_fingerprint": None,
        "git_commit": producer_git.get("commit"),
        "detail": f"{len(verdicts)} matched-budget cells passed conservative parity",
        "metrics": metrics,
        "evidence": evidence,
    }
    if fill:
        slots = card.get("slots")
        if isinstance(slots, Mapping) and SLOT in slots:
            fill_slot(card_path, SLOT, record)
        else:
            print(
                f"[cb-performance] validated, but {card_path} has no {SLOT!r} slot; record not filled",
                file=sys.stderr,
            )
    return {"schema": RESULT_SCHEMA, "status": "success", "record": record, "cells": verdicts}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shipcard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-fill", action="store_true", help="validate and write a record without mutating the shipcard"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_cb_performance(
            args.shipcard, args.manifest, fill=not args.no_fill
        )
        _atomic_write(args.output, result)
    except CBPerformanceValidationError as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 2
    print(f"saved passing matched-budget parity record to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
