"""Exact, categorical policy for Gridbook validation-only producers.

Validation-only is an artifact disposition, never a weak form of release
qualification.  This module owns the generic refusal vocabulary and the one
currently packaged exact policy: dense Qwen3.8 on Gridbook's untagged 0.9.1
SM120 candidate.  Export requires an explicit runtime contract whose canonical
JSON identity matches the packaged candidate pin.  Shipcard and publication
then re-derive the permanent refusal from ``quant_config.json``.

The separately implemented RTX4090 validation-only policy remains intact.  It
is recognized here only so release tooling has one categorical detector rather
than an architecture-specific exception for every validation producer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .cb_layout import FP8_PRODUCT_RUNGS, NVFP4_PRODUCT_RUNGS
from .gridbook_execution_contract import (
    GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
    GridbookExecutionContractError,
    parse_gridbook_execution_contract,
)
from .gridbook_format_contract import (
    GridbookFormatContractError,
    gridbook_format_rungs,
)
from .layer_config import canonicalize_assignment, read_layer_config_metadata


GRIDBOOK_VALIDATION_CANDIDATE_PIN_SCHEMA = (
    "prismaquant.gridbook_validation_candidate_pin.v1"
)
GRIDBOOK_VALIDATION_ONLY_POLICY_SCHEMA = (
    "prismaquant.gridbook_validation_only_policy.v1"
)
GRIDBOOK_VALIDATION_ONLY_ROUTE_STATUS_SCHEMA = (
    "prismaquant.gridbook_validation_only_route_status.v1"
)
VALIDATION_ONLY_DISPOSITION = "UNRELEASABLE_VALIDATION_ONLY"
VALIDATION_ONLY_QUALIFICATION_CEILING = "compile_only"

SM120_VALIDATION_POLICY_ID = "qwen38_sm120_cb_validation_only"
SM120_VALIDATION_SERVING_PROFILE = "qwen38_sm120_cb_validation_only"
SM120_VALIDATION_TARGET_PLATFORM = "sm_120"
SM120_VALIDATION_DEVICE_CAPABILITY = (12, 0)

SM120_CANDIDATE_PIN_ID = "gridbook_0_9_1_sm120_validation_candidate"
SM120_CANDIDATE_GRIDBOOK_VERSION = "0.9.1"
SM120_CANDIDATE_GRIDBOOK_COMMIT = (
    "d7827c507c2869184803f53314e589fc2dacbdb6"
)
SM120_CANDIDATE_GRIDBOOK_TREE = (
    "7371475656a11b57408b47508a4dd22937fe5c3f"
)
SM120_CANDIDATE_RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v11"
SM120_CANDIDATE_RUNTIME_CONTRACT_VERSION = 11
SM120_CANDIDATE_RUNTIME_CONTRACT_CANONICAL_SHA256 = (
    "15a1e3aedc5ed3da2bf04b3c28546bf11528f6976723f971321dfebda223098c"
)
SM120_CANDIDATE_RUNTIME_CONTRACT_FILE_SHA256 = (
    "585d5563cc69913937ab4c4a3b0cc6428a5c072d6a49b57ed03f8c28b8699b0d"
)

_ASSET_DIR = Path(__file__).resolve().parent / "gridbook_runtime"
SM120_VALIDATION_CANDIDATE_PIN_PATH = (
    _ASSET_DIR / "gridbook_sm120_validation_candidate_pin.json"
)
SM120_VALIDATION_CANDIDATE_CONTRACT_PATH = (
    _ASSET_DIR / "gridbook_runtime_contract.0.9.1-candidate.json"
)

_PIN_KEYS = {
    "schema",
    "id",
    "target_platform",
    "artifact_disposition",
    "runtime_qualification_ceiling",
    "gridbook",
    "runtime_contract",
}
_PIN_GRIDBOOK_KEYS = {
    "version", "commit", "tree", "version_is_release", "release_tag",
}
_PIN_CONTRACT_KEYS = {
    "schema",
    "contract_version",
    "lane_eligibility_schema",
    "canonical_json_sha256",
    "file_sha256",
}


class GridbookValidationOnlyPolicyError(ValueError):
    """A validation-only producer or immutable candidate binding differs."""


@dataclass(frozen=True)
class ValidationOnlyInspection:
    """Release-tool verdict derived from quant-config authority."""

    required: bool
    valid: bool
    serving_profile: str | None = None
    detail: str = ""


def _clone(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(
    source: Mapping[str, Any] | str | Path,
    *,
    where: str,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return _clone(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: {path} must contain one JSON object"
        )
    return dict(payload)


def _require_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    if set(payload) != expected:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: fields differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _expected_candidate_pin() -> dict[str, Any]:
    return {
        "schema": GRIDBOOK_VALIDATION_CANDIDATE_PIN_SCHEMA,
        "id": SM120_CANDIDATE_PIN_ID,
        "target_platform": SM120_VALIDATION_TARGET_PLATFORM,
        "artifact_disposition": VALIDATION_ONLY_DISPOSITION,
        "runtime_qualification_ceiling": (
            VALIDATION_ONLY_QUALIFICATION_CEILING
        ),
        "gridbook": {
            "version": SM120_CANDIDATE_GRIDBOOK_VERSION,
            "commit": SM120_CANDIDATE_GRIDBOOK_COMMIT,
            "tree": SM120_CANDIDATE_GRIDBOOK_TREE,
            "version_is_release": False,
            "release_tag": None,
        },
        "runtime_contract": {
            "schema": SM120_CANDIDATE_RUNTIME_CONTRACT_SCHEMA,
            "contract_version": SM120_CANDIDATE_RUNTIME_CONTRACT_VERSION,
            "lane_eligibility_schema": GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
            "canonical_json_sha256": (
                SM120_CANDIDATE_RUNTIME_CONTRACT_CANONICAL_SHA256
            ),
            "file_sha256": SM120_CANDIDATE_RUNTIME_CONTRACT_FILE_SHA256,
        },
    }


def load_sm120_validation_candidate_pin(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and exactly validate the packaged untagged candidate identity."""

    source = SM120_VALIDATION_CANDIDATE_PIN_PATH if path is None else Path(path)
    payload = _load_json_object(source, where="SM120 validation candidate pin")
    _require_keys(payload, _PIN_KEYS, where="SM120 validation candidate pin")
    gridbook = payload.get("gridbook")
    contract = payload.get("runtime_contract")
    if not isinstance(gridbook, Mapping) or not isinstance(contract, Mapping):
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate pin: gridbook and runtime_contract "
            "must be objects"
        )
    _require_keys(
        gridbook,
        _PIN_GRIDBOOK_KEYS,
        where="SM120 validation candidate pin.gridbook",
    )
    _require_keys(
        contract,
        _PIN_CONTRACT_KEYS,
        where="SM120 validation candidate pin.runtime_contract",
    )
    expected = _expected_candidate_pin()
    if payload != expected:
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate pin differs from the reviewed exact "
            "Gridbook 0.9.1 untagged candidate identity"
        )
    return _clone(payload)


def _runtime_attestation(
    contract: Mapping[str, Any],
    *,
    pin: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        parsed = parse_gridbook_execution_contract(
            contract, where="SM120 validation candidate runtime contract"
        )
        nvfp4 = gridbook_format_rungs(contract, "NVFP4_CB_K")
        fp8 = gridbook_format_rungs(contract, "FP8_CB_K")
    except (GridbookExecutionContractError, GridbookFormatContractError) as exc:
        raise GridbookValidationOnlyPolicyError(str(exc)) from exc
    platform = [
        item for item in parsed.platforms
        if item.id == SM120_VALIDATION_TARGET_PLATFORM
        and item.device_capability == SM120_VALIDATION_DEVICE_CAPABILITY
    ]
    if len(platform) != 1:
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate contract lacks the exact sm_120/[12,0] "
            "platform identity"
        )
    if nvfp4.producer_rungs != NVFP4_PRODUCT_RUNGS:
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate contract NVFP4 producer rungs differ "
            f"from {list(NVFP4_PRODUCT_RUNGS)}"
        )
    if fp8.producer_rungs != FP8_PRODUCT_RUNGS:
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate contract FP8 producer rungs differ "
            f"from {list(FP8_PRODUCT_RUNGS)}"
        )
    cells = tuple(
        item for item in parsed.cells
        if item.platform == SM120_VALIDATION_TARGET_PLATFORM
    )
    if not cells or any(
        item.qualification != VALIDATION_ONLY_QUALIFICATION_CEILING
        for item in cells
    ):
        raise GridbookValidationOnlyPolicyError(
            "SM120 validation candidate cells must exist and be compile_only"
        )
    return {
        "runtime_contract_schema": str(contract.get("schema")),
        "runtime_contract_version": contract.get("contract_version"),
        "lane_eligibility_schema": str(
            contract.get("lane_eligibility", {}).get("schema")
        ),
        "runtime_contract_canonical_json_sha256": canonical_json_sha256(
            contract
        ),
        "runtime_contract_file_sha256": pin["runtime_contract"][
            "file_sha256"
        ],
        "target_platform": SM120_VALIDATION_TARGET_PLATFORM,
        "device_capability": list(SM120_VALIDATION_DEVICE_CAPABILITY),
        "qualification_ceiling": VALIDATION_ONLY_QUALIFICATION_CEILING,
        "route_statuses": sorted({item.route_status for item in cells}),
        "cell_ids": sorted(item.id for item in cells),
        "producer_rungs": {
            "NVFP4_CB_K": list(NVFP4_PRODUCT_RUNGS),
            "FP8_CB_K": list(FP8_PRODUCT_RUNGS),
        },
    }


def require_sm120_validation_runtime_contract(
    source: Mapping[str, Any] | str | Path,
    *,
    where: str = "SM120 validation Gridbook runtime contract",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the explicit contract to equal the packaged candidate pin."""

    contract = _load_json_object(source, where=where)
    pin = load_sm120_validation_candidate_pin()
    expected_contract = pin["runtime_contract"]
    digest = canonical_json_sha256(contract)
    if (
        contract.get("schema") != expected_contract["schema"]
        or contract.get("contract_version")
        != expected_contract["contract_version"]
        or not isinstance(contract.get("lane_eligibility"), Mapping)
        or contract["lane_eligibility"].get("schema")
        != expected_contract["lane_eligibility_schema"]
        or digest != expected_contract["canonical_json_sha256"]
    ):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: contract differs from exact candidate "
            f"{SM120_CANDIDATE_GRIDBOOK_COMMIT}; observed schema="
            f"{contract.get('schema')!r}, version="
            f"{contract.get('contract_version')!r}, lane_schema="
            f"{getattr(contract.get('lane_eligibility'), 'get', lambda *_: None)('schema')!r}, "
            f"canonical_sha256={digest}"
        )
    return contract, _runtime_attestation(contract, pin=pin)


def sm120_validation_only_policy_stamp(
    runtime_contract: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build the exact immutable artifact policy from an explicit contract."""

    _contract, attestation = require_sm120_validation_runtime_contract(
        runtime_contract
    )
    pin = load_sm120_validation_candidate_pin()
    return {
        "schema": GRIDBOOK_VALIDATION_ONLY_POLICY_SCHEMA,
        "id": SM120_VALIDATION_POLICY_ID,
        "serving_profile": SM120_VALIDATION_SERVING_PROFILE,
        "target_platform": SM120_VALIDATION_TARGET_PLATFORM,
        "artifact_disposition": VALIDATION_ONLY_DISPOSITION,
        "runtime_qualification_ceiling": (
            VALIDATION_ONLY_QUALIFICATION_CEILING
        ),
        "candidate_runtime": {
            "pin_schema": pin["schema"],
            "pin_id": pin["id"],
            "pin_canonical_json_sha256": canonical_json_sha256(pin),
            "gridbook": _clone(pin["gridbook"]),
        },
        "runtime_attestation": attestation,
    }


def validate_sm120_validation_only_policy_stamp(
    payload: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any] | str | Path | None = None,
    where: str = "SM120 validation-only producer policy",
) -> dict[str, Any]:
    """Reject every missing, extra, tampered, or cross-policy field."""

    if not isinstance(payload, Mapping):
        raise GridbookValidationOnlyPolicyError(f"{where}: stamp is required")
    source = (
        SM120_VALIDATION_CANDIDATE_CONTRACT_PATH
        if runtime_contract is None
        else runtime_contract
    )
    expected = sm120_validation_only_policy_stamp(source)
    if dict(payload) != expected:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: stamp differs from exact policy/profile "
            f"{SM120_VALIDATION_POLICY_ID!r} and candidate runtime identity"
        )
    return expected


def prepare_gridbook_validation_only_export_policy(
    *,
    layer_config_path: str | Path,
    producer_policy: str | None,
    runtime_contract: Mapping[str, Any] | str | Path | None,
    where: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Require exact policy inputs when layer metadata selects SM120.

    The layer-config stamp is authoritative for whether the obligation exists.
    An environment override may further narrow ordinary format checks, but it
    cannot erase a validation-only producer identity already selected by the
    allocator.
    """

    metadata = read_layer_config_metadata(layer_config_path)
    profile = str(metadata.get("target_profile") or "").strip()
    if profile != SM120_VALIDATION_SERVING_PROFILE:
        if producer_policy == SM120_VALIDATION_POLICY_ID:
            raise GridbookValidationOnlyPolicyError(
                f"{where}: producer policy {producer_policy!r} cannot be "
                f"replayed onto layer metadata target_profile={profile!r}"
            )
        return None
    if producer_policy != SM120_VALIDATION_POLICY_ID:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: layer metadata target_profile={profile!r} requires "
            f"producer_policy={SM120_VALIDATION_POLICY_ID!r}; observed "
            f"{producer_policy!r}"
        )
    if runtime_contract is None:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: {producer_policy!r} requires an explicit exact "
            "Gridbook candidate runtime contract"
        )
    contract, _attestation = require_sm120_validation_runtime_contract(
        runtime_contract, where=f"{where} explicit Gridbook contract"
    )
    return contract, sm120_validation_only_policy_stamp(contract)


def sm120_validation_only_route_status_stamp(
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe selected bytes without converting compile-only into support."""

    policy = validate_sm120_validation_only_policy_stamp(producer_policy)
    try:
        canonical = canonicalize_assignment(assignment)
    except (TypeError, ValueError) as exc:
        raise GridbookValidationOnlyPolicyError(
            f"SM120 validation-only assignment is invalid: {exc}"
        ) from exc
    from .serving_profiles import check_serving_format

    refused = [
        (qname, fmt)
        for qname, fmt in canonical.items()
        if not check_serving_format(
            SM120_VALIDATION_SERVING_PROFILE, qname, fmt
        ).legal
    ]
    if refused:
        raise GridbookValidationOnlyPolicyError(
            f"SM120 validation-only assignment contains refused formats: "
            f"{refused[:8]}"
        )
    by_family: dict[str, dict[str, Any]] = {}
    for prefix, family in (
        ("NVFP4_CB_K", "NVFP4_CB_K"),
        ("FP8_CB_K", "FP8_CB_K"),
    ):
        selected = [fmt for fmt in canonical.values() if fmt.startswith(prefix)]
        by_family[family] = {
            "selected_units": len(selected),
            "selected_rungs": sorted({
                int(fmt.rsplit("K", 1)[1]) for fmt in selected
            }),
        }
    return {
        "schema": GRIDBOOK_VALIDATION_ONLY_ROUTE_STATUS_SCHEMA,
        "authority": "producer_policy.runtime_attestation",
        "serving_profile": SM120_VALIDATION_SERVING_PROFILE,
        "target_platform": SM120_VALIDATION_TARGET_PLATFORM,
        "artifact_disposition": VALIDATION_ONLY_DISPOSITION,
        "qualification_ceiling": VALIDATION_ONLY_QUALIFICATION_CEILING,
        "release_eligible": False,
        "selected_units": len(canonical),
        "selected_cb": by_family,
        "delegated_terminals": {
            name: sum(fmt == name for fmt in canonical.values())
            for name in ("NVFP4", "FP8_E4M3", "BF16")
        },
        "runtime_attestation": _clone(policy["runtime_attestation"]),
    }


def validate_sm120_validation_only_route_status(
    payload: Mapping[str, Any],
    *,
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    expected = sm120_validation_only_route_status_stamp(
        producer_policy, assignment
    )
    if not isinstance(payload, Mapping) or dict(payload) != expected:
        raise GridbookValidationOnlyPolicyError(
            f"{where}: route status differs from exact validation-only "
            "policy, assignment, or candidate runtime"
        )
    return expected


def sm120_validation_only_route_status_summary(
    payload: Mapping[str, Any],
    *,
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    validated = validate_sm120_validation_only_route_status(
        payload,
        producer_policy=producer_policy,
        assignment=assignment,
        where=where,
    )
    runtime = validated["runtime_attestation"]
    return {
        "schema": GRIDBOOK_VALIDATION_ONLY_ROUTE_STATUS_SCHEMA,
        "authority": validated["authority"],
        "serving_profile": validated["serving_profile"],
        "target_platform": validated["target_platform"],
        "artifact_disposition": VALIDATION_ONLY_DISPOSITION,
        "qualification_ceiling": VALIDATION_ONLY_QUALIFICATION_CEILING,
        "release_eligible": False,
        "selected_units": validated["selected_units"],
        "selected_cb": _clone(validated["selected_cb"]),
        "delegated_terminals": _clone(validated["delegated_terminals"]),
        "candidate_route_statuses": list(runtime["route_statuses"]),
        "runtime_contract_sha256": runtime[
            "runtime_contract_canonical_json_sha256"
        ],
    }


def validate_sm120_validation_only_quant_config(
    quant_config: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any] | str | Path | None = None,
    where: str = "SM120 validation-only quant_config",
) -> dict[str, Any]:
    """Replay the exact stamp and route binding from finalized provenance."""

    if not isinstance(quant_config, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: quant_config must be an object"
        )
    if quant_config.get("quant_method") != "gridbook":
        raise GridbookValidationOnlyPolicyError(
            f"{where}: quant_method must be 'gridbook'"
        )
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: provenance object is required"
        )
    policy = provenance.get("producer_policy")
    if not isinstance(policy, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: provenance.producer_policy is required"
        )
    validate_sm120_validation_only_policy_stamp(
        policy, runtime_contract=runtime_contract, where=f"{where} policy"
    )
    assignment = provenance.get("tensor_formats")
    if not isinstance(assignment, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: provenance.tensor_formats is required"
        )
    route = provenance.get("cb_route_status")
    if not isinstance(route, Mapping):
        raise GridbookValidationOnlyPolicyError(
            f"{where}: provenance.cb_route_status is required"
        )
    validate_sm120_validation_only_route_status(
        route,
        producer_policy=policy,
        assignment=assignment,
        where=f"{where} cb_route_status",
    )
    return _clone(quant_config)


def _hint_values(quant_config: Mapping[str, Any]) -> set[str]:
    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        return set()
    hints: set[str] = set()
    policy = provenance.get("producer_policy")
    if isinstance(policy, Mapping):
        hints.update(str(policy.get(key) or "") for key in (
            "id", "serving_profile", "artifact_disposition",
        ))
    elif isinstance(policy, str):
        hints.add(policy)
    route = provenance.get("cb_route_status")
    if isinstance(route, Mapping):
        hints.update(str(route.get(key) or "") for key in (
            "serving_profile", "target_profile", "artifact_disposition",
        ))
    return {item for item in hints if item}


def inspect_validation_only_quant_config(
    quant_config: Mapping[str, Any],
) -> ValidationOnlyInspection:
    """Identify valid and malformed validation-only artifacts fail-closed."""

    if not isinstance(quant_config, Mapping):
        return ValidationOnlyInspection(False, False)
    hints = _hint_values(quant_config)
    sm120_required = bool(hints & {
        SM120_VALIDATION_POLICY_ID,
        SM120_VALIDATION_SERVING_PROFILE,
    })
    if sm120_required:
        try:
            validate_sm120_validation_only_quant_config(quant_config)
        except GridbookValidationOnlyPolicyError as exc:
            return ValidationOnlyInspection(
                True,
                False,
                SM120_VALIDATION_SERVING_PROFILE,
                f"required validation-only stamp is missing or malformed: {exc}",
            )
        return ValidationOnlyInspection(
            True,
            True,
            SM120_VALIDATION_SERVING_PROFILE,
            "exact SM120 validation-only policy stamp",
        )

    # Preserve the existing RTX4090 producer while giving release tools one
    # architecture-neutral decision point.
    from .rtx4090_qwen38_policy import (
        RTX4090_VALIDATION_ONLY_POLICY_ID,
        RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
        is_rtx4090_validation_only_policy,
    )

    rtx_required = bool(hints & {
        RTX4090_VALIDATION_ONLY_POLICY_ID,
        RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
    })
    provenance = quant_config.get("provenance")
    policy = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    if rtx_required or is_rtx4090_validation_only_policy(policy):
        valid = is_rtx4090_validation_only_policy(policy)
        return ValidationOnlyInspection(
            True,
            valid,
            RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
            (
                "RTX4090 validation-only policy stamp"
                if valid
                else "required RTX4090 validation-only stamp is malformed"
            ),
        )
    if VALIDATION_ONLY_DISPOSITION in hints:
        return ValidationOnlyInspection(
            True,
            False,
            None,
            "unrecognized policy carries the categorical validation-only "
            "disposition",
        )
    return ValidationOnlyInspection(False, False)


def validation_only_policy_build_fields(
    policy: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Return exact card echoes, rejecting a malformed known policy."""

    if not isinstance(policy, Mapping):
        return None
    hints = {
        str(policy.get(key) or "")
        for key in ("id", "serving_profile", "artifact_disposition")
    }
    if hints & {SM120_VALIDATION_POLICY_ID, SM120_VALIDATION_SERVING_PROFILE}:
        exact = validate_sm120_validation_only_policy_stamp(policy)
        return {
            "producer_policy": str(exact["id"]),
            "serving_profile": str(exact["serving_profile"]),
            "artifact_disposition": str(exact["artifact_disposition"]),
        }
    from .rtx4090_qwen38_policy import is_rtx4090_validation_only_policy

    if is_rtx4090_validation_only_policy(policy):
        return {
            "producer_policy": str(policy["id"]),
            "serving_profile": str(policy.get("serving_profile") or ""),
            "artifact_disposition": str(policy["artifact_disposition"]),
        }
    if VALIDATION_ONLY_DISPOSITION in hints:
        raise GridbookValidationOnlyPolicyError(
            "validation-only producer policy is malformed or unrecognized"
        )
    return None


def validation_only_build_hint(build: Mapping[str, Any] | None) -> bool:
    """Recognize a card echo that requires fail-closed quant-config replay."""

    if not isinstance(build, Mapping):
        return False
    if build.get("artifact_disposition") == VALIDATION_ONLY_DISPOSITION:
        return True
    from .rtx4090_qwen38_policy import (
        RTX4090_VALIDATION_ONLY_POLICY_ID,
        RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
    )

    return bool(
        build.get("serving_profile") in {
            SM120_VALIDATION_SERVING_PROFILE,
            RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
        }
        or build.get("producer_policy") in {
            SM120_VALIDATION_POLICY_ID,
            RTX4090_VALIDATION_ONLY_POLICY_ID,
        }
    )


__all__ = [
    "GRIDBOOK_VALIDATION_CANDIDATE_PIN_SCHEMA",
    "GRIDBOOK_VALIDATION_ONLY_POLICY_SCHEMA",
    "GRIDBOOK_VALIDATION_ONLY_ROUTE_STATUS_SCHEMA",
    "GridbookValidationOnlyPolicyError",
    "SM120_CANDIDATE_GRIDBOOK_COMMIT",
    "SM120_CANDIDATE_GRIDBOOK_TREE",
    "SM120_CANDIDATE_GRIDBOOK_VERSION",
    "SM120_CANDIDATE_RUNTIME_CONTRACT_CANONICAL_SHA256",
    "SM120_CANDIDATE_RUNTIME_CONTRACT_FILE_SHA256",
    "SM120_VALIDATION_CANDIDATE_CONTRACT_PATH",
    "SM120_VALIDATION_CANDIDATE_PIN_PATH",
    "SM120_VALIDATION_POLICY_ID",
    "SM120_VALIDATION_SERVING_PROFILE",
    "SM120_VALIDATION_TARGET_PLATFORM",
    "VALIDATION_ONLY_DISPOSITION",
    "VALIDATION_ONLY_QUALIFICATION_CEILING",
    "ValidationOnlyInspection",
    "canonical_json_sha256",
    "inspect_validation_only_quant_config",
    "load_sm120_validation_candidate_pin",
    "prepare_gridbook_validation_only_export_policy",
    "require_sm120_validation_runtime_contract",
    "sm120_validation_only_policy_stamp",
    "sm120_validation_only_route_status_stamp",
    "sm120_validation_only_route_status_summary",
    "validate_sm120_validation_only_policy_stamp",
    "validate_sm120_validation_only_quant_config",
    "validate_sm120_validation_only_route_status",
    "validation_only_build_hint",
    "validation_only_policy_build_fields",
]
