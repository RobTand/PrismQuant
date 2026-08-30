"""Immutable contracts for probe-once, export-many artifact collections.

This module is an offline control plane.  It does not add a format, alter the
allocator, or participate in a serving hot path.  Its job is to give future
formats and device targets stable names before those implementations exist.

Every authoritative object is a strict JSON envelope.  ``payload_sha256`` is
the semantic object ID; optional locators live outside the hashed payload so a
mount point, Hugging Face URL, or local staging path cannot change identity.
References bind both the semantic identity and the exact portable content.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from prismaquant.cost_stage_checkpoint import canonical_json, canonical_json_sha256
from prismaquant.schemas import SchemaValidationError


CANDIDATE_SCHEMA = "prismaquant.artifact_collection.candidate.v1"
CATALOG_SCHEMA = "prismaquant.artifact_collection.candidate_catalog.v1"
TARGET_SCHEMA = "prismaquant.artifact_collection.target_profile.v1"
CONTRACT_SCHEMA = "prismaquant.artifact_collection.contract.v1"
RECEIPT_SCHEMA = "prismaquant.artifact_collection.stage_receipt.v1"
MANIFEST_SCHEMA = "prismaquant.artifact_collection.manifest.v1"
REFERENCE_SCHEMA = "prismaquant.artifact_collection.reference.v1"
BYTE_BREAKDOWN_SCHEMA = "prismaquant.artifact_collection.byte_breakdown.v1"
LEGACY_AUDIT_SCHEMA = "prismaquant.artifact_collection.legacy_export_audit.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STAGES = frozenset({
    "measure",
    "solve",
    "validate",
    "export",
    "qualify",
    "publish",
})


class ArtifactCollectionError(SchemaValidationError):
    """Raised when an artifact-collection object is not exact and self-bound."""


def _fail(where: str, message: str) -> None:
    raise ArtifactCollectionError(f"{where}: {message}")


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(where, "expected a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    where: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        _fail(where, f"field set differs (missing={missing}, extra={extra})")


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "expected a nonempty string")
    return value


def _sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(where, "expected a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, *, where: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        _fail(where, f"expected a {qualifier} integer")
    return value


def _canonical_object(value: object, *, where: str, nonempty: bool = True) -> dict[str, object]:
    value = _mapping(value, where=where)
    try:
        canonical = canonical_json(value, where=where)
    except ValueError as exc:
        raise ArtifactCollectionError(f"{where}: not finite canonical JSON data") from exc
    if not isinstance(canonical, dict):  # Mapping canonicalizes to a dict.
        _fail(where, "expected an object after canonicalization")
    if nonempty and not canonical:
        _fail(where, "object must not be empty")
    return canonical


def _sorted_unique_texts(value: object, *, where: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, "expected an array of strings")
    rows = [_text(item, where=f"{where}[{index}]") for index, item in enumerate(value)]
    if rows != sorted(set(rows)):
        _fail(where, "must be sorted and contain no duplicates")
    return rows


def _canonical_texts(value: object, *, where: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, "expected an array of strings")
    rows = [_text(item, where=f"{where}[{index}]") for index, item in enumerate(value)]
    if len(rows) != len(set(rows)):
        _fail(where, "must contain no duplicates")
    return sorted(rows)


def _canonical_bytes(value: object, *, where: str) -> bytes:
    try:
        canonical = canonical_json(value, where=where)
    except ValueError as exc:
        raise ArtifactCollectionError(f"{where}: not finite canonical JSON data") from exc
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_locators(value: object, *, where: str) -> dict[str, list[str]]:
    mapping = _mapping(value, where=where)
    result: dict[str, list[str]] = {}
    for subject_id, raw_locations in mapping.items():
        _sha256(subject_id, where=f"{where} key")
        if isinstance(raw_locations, (str, bytes)) or not isinstance(raw_locations, Sequence):
            _fail(f"{where}.{subject_id}", "expected an array of locator strings")
        locations = [
            _text(item, where=f"{where}.{subject_id}[{index}]")
            for index, item in enumerate(raw_locations)
        ]
        if locations != sorted(set(locations)):
            _fail(f"{where}.{subject_id}", "must be sorted and contain no duplicates")
        result[subject_id] = locations
    return dict(sorted(result.items()))


def seal_record(
    schema: str,
    payload: Mapping[str, object],
    *,
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Return a canonical semantic envelope.

    Locators are deliberately advisory: they are validated and serialized but
    excluded from ``payload_sha256``.  The referenced bytes remain protected by
    each reference's content digest.
    """
    schema = _text(schema, where="schema")
    canonical_payload = _canonical_object(payload, where=f"{schema}.payload")
    record: dict[str, object] = {
        "schema": schema,
        "payload": canonical_payload,
        "payload_sha256": canonical_json_sha256(
            canonical_payload, where=f"{schema}.payload"
        ),
    }
    if locators is not None:
        record["locators"] = _validate_locators(locators, where=f"{schema}.locators")
    verify_record(record)
    return record


def _reject_reference_equivocation(
    value: object,
    *,
    where: str,
    subjects: dict[tuple[str, str], tuple[str, int]] | None = None,
) -> None:
    """Reject one semantic subject resolving to multiple portable contents."""
    if subjects is None:
        subjects = {}
    if isinstance(value, Mapping):
        if value.get("schema") == REFERENCE_SCHEMA:
            reference = _validate_reference(value, where=where)
            subject = (
                str(reference["subject_schema"]),
                str(reference["subject_id"]),
            )
            content = _mapping(reference["content"], where=f"{where}.content")
            physical = (str(content["sha256"]), int(content["size_bytes"]))
            prior = subjects.get(subject)
            if prior is not None and prior != physical:
                _fail(where, f"semantic reference equivocation for {subject}")
            subjects[subject] = physical
            return
        for key, child in value.items():
            _reject_reference_equivocation(
                child, where=f"{where}.{key}", subjects=subjects
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_reference_equivocation(
                child, where=f"{where}[{index}]", subjects=subjects
            )


def verify_record(record: Mapping[str, object]) -> dict[str, object]:
    """Verify the strict envelope and any recognized payload schema."""
    value = _mapping(record, where="record")
    allowed = {"schema", "payload", "payload_sha256"}
    if "locators" in value:
        allowed.add("locators")
    _exact_keys(value, allowed, where="record")
    schema = _text(value["schema"], where="record.schema")
    payload = _canonical_object(value["payload"], where="record.payload")
    observed = _sha256(value["payload_sha256"], where="record.payload_sha256")
    expected = canonical_json_sha256(payload, where="record.payload")
    if observed != expected:
        _fail("record.payload_sha256", f"differs (stored={observed}, computed={expected})")
    if "locators" in value:
        _validate_locators(value["locators"], where="record.locators")
    _reject_reference_equivocation(payload, where="record.payload")
    validator = _PAYLOAD_VALIDATORS.get(schema)
    if validator is None and schema == LEGACY_AUDIT_SCHEMA:
        # Lazy import avoids making the generic control plane depend on a
        # legacy adapter at module-import time while keeping load_record strict.
        from prismaquant.artifact_collection_legacy import (
            validate_legacy_audit_payload,
        )

        validator = validate_legacy_audit_payload
    if validator is not None:
        validator(payload)
    canonical_record: dict[str, object] = {
        "schema": schema,
        "payload": payload,
        "payload_sha256": observed,
    }
    if "locators" in value:
        canonical_record["locators"] = _validate_locators(
            value["locators"], where="record.locators"
        )
    return canonical_record


def semantic_record_bytes(record: Mapping[str, object]) -> bytes:
    """Canonical portable bytes, excluding advisory locator annotations."""
    verified = verify_record(record)
    portable = {key: verified[key] for key in ("schema", "payload", "payload_sha256")}
    return _canonical_bytes(portable, where="semantic record") + b"\n"


def write_record(path: str | os.PathLike[str], record: Mapping[str, object]) -> None:
    """Publish portable semantic bytes exactly once.

    Advisory locators are an in-memory resolution overlay and are deliberately
    omitted.  Consequently ``reference_for_record(record)`` always describes
    the exact bytes written here.
    """
    destination = Path(path)
    verified = verify_record(record)
    encoded = semantic_record_bytes(verified)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _fail(str(destination), "already exists and will not be replaced")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            _fail(str(destination), "appeared concurrently and will not be replaced")
            raise AssertionError("unreachable") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reject_duplicate_members(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate member {key!r}")
        result[key] = value
    return result


def load_record(path: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ArtifactCollectionError(f"JSON: non-finite value {item}")
            ),
        )
    except ArtifactCollectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCollectionError(f"unreadable artifact collection record: {source}") from exc
    return verify_record(_mapping(value, where=str(source)))


def make_reference(
    *,
    subject_schema: str,
    subject_id: str,
    content_sha256: str,
    size_bytes: int,
) -> dict[str, object]:
    reference: dict[str, object] = {
        "schema": REFERENCE_SCHEMA,
        "subject_schema": _text(subject_schema, where="reference.subject_schema"),
        "subject_id": _sha256(subject_id, where="reference.subject_id"),
        "content": {
            "sha256": _sha256(content_sha256, where="reference.content.sha256"),
            "size_bytes": _nonnegative_int(
                size_bytes, where="reference.content.size_bytes"
            ),
        },
    }
    _validate_reference(reference, where="reference")
    return reference


def reference_for_record(record: Mapping[str, object]) -> dict[str, object]:
    verified = verify_record(record)
    encoded = semantic_record_bytes(verified)
    return make_reference(
        subject_schema=str(verified["schema"]),
        subject_id=str(verified["payload_sha256"]),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _validate_reference(value: object, *, where: str) -> dict[str, object]:
    reference = _mapping(value, where=where)
    _exact_keys(
        reference,
        {"schema", "subject_schema", "subject_id", "content"},
        where=where,
    )
    if reference["schema"] != REFERENCE_SCHEMA:
        _fail(f"{where}.schema", f"expected {REFERENCE_SCHEMA!r}")
    _text(reference["subject_schema"], where=f"{where}.subject_schema")
    _sha256(reference["subject_id"], where=f"{where}.subject_id")
    content = _mapping(reference["content"], where=f"{where}.content")
    _exact_keys(content, {"sha256", "size_bytes"}, where=f"{where}.content")
    _sha256(content["sha256"], where=f"{where}.content.sha256")
    _nonnegative_int(content["size_bytes"], where=f"{where}.content.size_bytes")
    canonical = canonical_json(reference, where=where)
    assert isinstance(canonical, dict)
    return canonical


def _reference_sort_key(reference: Mapping[str, object]) -> tuple[str, str, str]:
    content = _mapping(reference["content"], where="reference.content")
    return (
        str(reference["subject_schema"]),
        str(reference["subject_id"]),
        str(content["sha256"]),
    )


def _sorted_unique_references(value: object, *, where: str) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, "expected an array of references")
    references = [
        _validate_reference(item, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    ]
    keys = [_reference_sort_key(item) for item in references]
    if keys != sorted(keys):
        _fail(where, "must be sorted")
    subjects: dict[tuple[str, str], tuple[str, int]] = {}
    for index, reference in enumerate(references):
        subject = (str(reference["subject_schema"]), str(reference["subject_id"]))
        content = _mapping(reference["content"], where=f"{where}[{index}].content")
        physical = (str(content["sha256"]), int(content["size_bytes"]))
        prior = subjects.get(subject)
        if prior is not None:
            qualifier = " with conflicting content" if prior != physical else ""
            _fail(where, f"duplicate semantic reference{qualifier}: {subject}")
        subjects[subject] = physical
    return references


def _behavior_identity(payload: Mapping[str, object]) -> str:
    behavior = {
        key: payload[key]
        for key in (
            "format_semantics",
            "basis",
            "render_contract",
            "scale_contract",
            "activation_contract",
        )
    }
    return canonical_json_sha256(behavior, where="candidate behavior identity")


def make_candidate(
    *,
    format_semantics: Mapping[str, object],
    basis_kind: str,
    basis_scope: str,
    basis_assets: Sequence[Mapping[str, object]],
    render_contract: Mapping[str, object],
    scale_contract: Mapping[str, object],
    activation_contract: Mapping[str, object] | None,
    serialization_contract: Mapping[str, object],
    runtime_contract: Mapping[str, object],
    required_probe_features: Sequence[str],
    required_runtime_features: Sequence[str],
    applicability: Mapping[str, object],
    shared_resources: Sequence[Mapping[str, object]],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Create a candidate whose identity is richer than a format/rung label."""
    payload: dict[str, object] = {
        "format_semantics": _canonical_object(
            format_semantics, where="candidate.format_semantics"
        ),
        "basis": {
            "kind": _text(basis_kind, where="candidate.basis.kind"),
            "scope": _text(basis_scope, where="candidate.basis.scope"),
            "assets": sorted(
                [_validate_reference(item, where="candidate.basis.assets") for item in basis_assets],
                key=_reference_sort_key,
            ),
        },
        "render_contract": _validate_reference(
            render_contract, where="candidate.render_contract"
        ),
        "scale_contract": _validate_reference(
            scale_contract, where="candidate.scale_contract"
        ),
        "activation_contract": (
            _validate_reference(activation_contract, where="candidate.activation_contract")
            if activation_contract is not None
            else None
        ),
        "serialization_contract": _validate_reference(
            serialization_contract, where="candidate.serialization_contract"
        ),
        "runtime_contract": _validate_reference(
            runtime_contract, where="candidate.runtime_contract"
        ),
        "required_probe_features": _canonical_texts(
            required_probe_features, where="candidate.required_probe_features"
        ),
        "required_runtime_features": _canonical_texts(
            required_runtime_features, where="candidate.required_runtime_features"
        ),
        "applicability": _canonical_object(applicability, where="candidate.applicability"),
        "shared_resources": sorted(
            [_validate_reference(item, where="candidate.shared_resources") for item in shared_resources],
            key=_reference_sort_key,
        ),
    }
    payload["behavior_id"] = _behavior_identity(payload)
    record = seal_record(CANDIDATE_SCHEMA, payload, locators=locators)
    return verify_record(record)


def _validate_candidate(payload: Mapping[str, object]) -> None:
    expected = {
        "format_semantics",
        "basis",
        "render_contract",
        "scale_contract",
        "activation_contract",
        "serialization_contract",
        "runtime_contract",
        "required_probe_features",
        "required_runtime_features",
        "applicability",
        "shared_resources",
        "behavior_id",
    }
    _exact_keys(payload, expected, where="candidate.payload")
    _canonical_object(payload["format_semantics"], where="candidate.format_semantics")
    basis = _mapping(payload["basis"], where="candidate.basis")
    _exact_keys(basis, {"kind", "scope", "assets"}, where="candidate.basis")
    _text(basis["kind"], where="candidate.basis.kind")
    _text(basis["scope"], where="candidate.basis.scope")
    _sorted_unique_references(basis["assets"], where="candidate.basis.assets")
    for field in (
        "render_contract",
        "scale_contract",
        "serialization_contract",
        "runtime_contract",
    ):
        _validate_reference(payload[field], where=f"candidate.{field}")
    if payload["activation_contract"] is not None:
        _validate_reference(payload["activation_contract"], where="candidate.activation_contract")
    _sorted_unique_texts(payload["required_probe_features"], where="candidate.required_probe_features")
    _sorted_unique_texts(payload["required_runtime_features"], where="candidate.required_runtime_features")
    _canonical_object(payload["applicability"], where="candidate.applicability")
    _sorted_unique_references(payload["shared_resources"], where="candidate.shared_resources")
    observed = _sha256(payload["behavior_id"], where="candidate.behavior_id")
    expected_behavior = _behavior_identity(payload)
    if observed != expected_behavior:
        _fail("candidate.behavior_id", "does not bind numerical behavior fields")


def make_candidate_catalog(
    candidates: Sequence[Mapping[str, object]],
    *,
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Create an explicit candidate set; holes and cross-family overlap are valid."""
    references = sorted(
        [reference_for_record(candidate) for candidate in candidates],
        key=_reference_sort_key,
    )
    record = seal_record(CATALOG_SCHEMA, {"candidates": references}, locators=locators)
    return verify_record(record)


def _validate_catalog(payload: Mapping[str, object]) -> None:
    _exact_keys(payload, {"candidates"}, where="candidate_catalog.payload")
    candidates = _sorted_unique_references(payload["candidates"], where="candidate_catalog.candidates")
    for index, reference in enumerate(candidates):
        if reference["subject_schema"] != CANDIDATE_SCHEMA:
            _fail(f"candidate_catalog.candidates[{index}]", "does not reference a candidate")


def make_target_profile(
    *,
    artifact_byte_ceiling: int,
    usable_vram_bytes: int,
    accounting_rule: Mapping[str, object],
    device_profile: Mapping[str, object],
    workload: Mapping[str, object],
    placement_constraints: Mapping[str, object],
    exclusions: Sequence[str],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    payload = {
        "artifact_byte_ceiling": _nonnegative_int(
            artifact_byte_ceiling, where="target.artifact_byte_ceiling", positive=True
        ),
        "usable_vram_bytes": _nonnegative_int(
            usable_vram_bytes, where="target.usable_vram_bytes", positive=True
        ),
        "accounting_rule": _validate_reference(accounting_rule, where="target.accounting_rule"),
        "device_profile": _validate_reference(device_profile, where="target.device_profile"),
        "workload": _validate_reference(workload, where="target.workload"),
        "placement_constraints": _canonical_object(
            placement_constraints, where="target.placement_constraints"
        ),
        "exclusions": _canonical_texts(exclusions, where="target.exclusions"),
    }
    record = seal_record(TARGET_SCHEMA, payload, locators=locators)
    return verify_record(record)


def _validate_target(payload: Mapping[str, object]) -> None:
    expected = {
        "artifact_byte_ceiling",
        "usable_vram_bytes",
        "accounting_rule",
        "device_profile",
        "workload",
        "placement_constraints",
        "exclusions",
    }
    _exact_keys(payload, expected, where="target.payload")
    _nonnegative_int(payload["artifact_byte_ceiling"], where="target.artifact_byte_ceiling", positive=True)
    _nonnegative_int(payload["usable_vram_bytes"], where="target.usable_vram_bytes", positive=True)
    for field in ("accounting_rule", "device_profile", "workload"):
        _validate_reference(payload[field], where=f"target.{field}")
    _canonical_object(payload["placement_constraints"], where="target.placement_constraints")
    _sorted_unique_texts(payload["exclusions"], where="target.exclusions")


def make_collection_contract(
    *,
    model_snapshot: Mapping[str, object],
    probe_campaign: Mapping[str, object],
    candidate_catalog: Mapping[str, object],
    cost_snapshot: Mapping[str, object],
    accounting_rule: Mapping[str, object],
    variants: Mapping[str, Mapping[str, Mapping[str, object]]],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    variant_rows: list[dict[str, object]] = []
    for key, variant in sorted(variants.items()):
        row = _mapping(variant, where=f"collection.variants.{key}")
        _exact_keys(row, {"target_profile", "export_contract"}, where=f"collection.variants.{key}")
        target_record = verify_record(
            _mapping(row["target_profile"], where=f"collection.variants.{key}.target_profile")
        )
        if target_record["schema"] != TARGET_SCHEMA:
            _fail(
                f"collection.variants.{key}.target_profile",
                "expected a target-profile record",
            )
        target_payload = _mapping(
            target_record["payload"],
            where=f"collection.variants.{key}.target_profile.payload",
        )
        target_accounting = _validate_reference(
            target_payload["accounting_rule"],
            where=f"collection.variants.{key}.target_profile.accounting_rule",
        )
        common_accounting = _validate_reference(
            accounting_rule, where="collection.accounting_rule"
        )
        if target_accounting != common_accounting:
            _fail(
                f"collection.variants.{key}.target_profile.accounting_rule",
                "differs from the collection accounting rule",
            )
        variant_rows.append({
            "key": _text(key, where="collection.variant.key"),
            "target_profile": reference_for_record(target_record),
            "accounting_rule": common_accounting,
            "export_contract": _validate_reference(
                row["export_contract"], where=f"collection.variants.{key}.export_contract"
            ),
        })
    payload = {
        "model_snapshot": _validate_reference(model_snapshot, where="collection.model_snapshot"),
        "probe_campaign": _validate_reference(probe_campaign, where="collection.probe_campaign"),
        "candidate_catalog": _validate_reference(candidate_catalog, where="collection.candidate_catalog"),
        "cost_snapshot": _validate_reference(cost_snapshot, where="collection.cost_snapshot"),
        "accounting_rule": _validate_reference(accounting_rule, where="collection.accounting_rule"),
        "variants": variant_rows,
    }
    record = seal_record(CONTRACT_SCHEMA, payload, locators=locators)
    return verify_record(record)


def _validate_contract(payload: Mapping[str, object]) -> None:
    expected = {
        "model_snapshot",
        "probe_campaign",
        "candidate_catalog",
        "cost_snapshot",
        "accounting_rule",
        "variants",
    }
    _exact_keys(payload, expected, where="collection.payload")
    for field in expected - {"variants"}:
        _validate_reference(payload[field], where=f"collection.{field}")
    variants = payload["variants"]
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence) or not variants:
        _fail("collection.variants", "expected a nonempty array")
    keys: list[str] = []
    for index, raw in enumerate(variants):
        row = _mapping(raw, where=f"collection.variants[{index}]")
        _exact_keys(
            row,
            {"key", "target_profile", "accounting_rule", "export_contract"},
            where=f"collection.variants[{index}]",
        )
        keys.append(_text(row["key"], where=f"collection.variants[{index}].key"))
        target = _validate_reference(row["target_profile"], where=f"collection.variants[{index}].target_profile")
        if target["subject_schema"] != TARGET_SCHEMA:
            _fail(f"collection.variants[{index}].target_profile", "does not reference a target profile")
        variant_accounting = _validate_reference(
            row["accounting_rule"],
            where=f"collection.variants[{index}].accounting_rule",
        )
        if variant_accounting != payload["accounting_rule"]:
            _fail(
                f"collection.variants[{index}].accounting_rule",
                "differs from the collection accounting rule",
            )
        _validate_reference(row["export_contract"], where=f"collection.variants[{index}].export_contract")
    if keys != sorted(set(keys)):
        _fail("collection.variants", "keys must be sorted and unique")


def make_stage_receipt(
    *,
    collection_contract: Mapping[str, object],
    variant_key: str | None,
    stage: str,
    outcome: str,
    inputs: Sequence[Mapping[str, object]],
    outputs: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    payload = {
        "collection_contract": _validate_reference(
            collection_contract, where="receipt.collection_contract"
        ),
        "variant_key": (
            _text(variant_key, where="receipt.variant_key")
            if variant_key is not None
            else None
        ),
        "stage": _text(stage, where="receipt.stage"),
        "outcome": _text(outcome, where="receipt.outcome"),
        "inputs": sorted(
            [_validate_reference(item, where="receipt.inputs") for item in inputs],
            key=_reference_sort_key,
        ),
        "outputs": sorted(
            [_validate_reference(item, where="receipt.outputs") for item in outputs],
            key=_reference_sort_key,
        ),
        "evidence": sorted(
            [_validate_reference(item, where="receipt.evidence") for item in evidence],
            key=_reference_sort_key,
        ),
        "producer": _validate_reference(producer, where="receipt.producer"),
    }
    record = seal_record(RECEIPT_SCHEMA, payload, locators=locators)
    return verify_record(record)


def _validate_receipt(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "collection_contract",
            "variant_key",
            "stage",
            "outcome",
            "inputs",
            "outputs",
            "evidence",
            "producer",
        },
        where="receipt.payload",
    )
    contract = _validate_reference(
        payload["collection_contract"], where="receipt.collection_contract"
    )
    if contract["subject_schema"] != CONTRACT_SCHEMA:
        _fail("receipt.collection_contract", "does not reference a collection contract")
    stage = _text(payload["stage"], where="receipt.stage")
    if stage not in _STAGES:
        _fail("receipt.stage", f"unknown stage {stage!r}")
    outcome = _text(payload["outcome"], where="receipt.outcome")
    if outcome not in {"accepted", "rejected"}:
        _fail("receipt.outcome", "expected 'accepted' or 'rejected'")
    variant_key = payload["variant_key"]
    if stage == "measure":
        if variant_key is not None:
            _fail("receipt.variant_key", "measure receipts must bind the common campaign")
    elif not isinstance(variant_key, str) or not variant_key:
        _fail("receipt.variant_key", f"{stage} receipts must bind a variant key")
    inputs = _sorted_unique_references(payload["inputs"], where="receipt.inputs")
    outputs = _sorted_unique_references(payload["outputs"], where="receipt.outputs")
    evidence = _sorted_unique_references(payload["evidence"], where="receipt.evidence")
    if not inputs:
        _fail("receipt.inputs", "must not be empty")
    if not evidence:
        _fail("receipt.evidence", "must not be empty")
    if outcome == "accepted" and not outputs:
        _fail("receipt.outputs", "accepted receipts must name an output")
    _validate_reference(payload["producer"], where="receipt.producer")


def make_collection_manifest(
    *,
    collection_contract: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    contract_record = verify_record(collection_contract)
    if contract_record["schema"] != CONTRACT_SCHEMA:
        _fail("manifest.collection_contract", "expected a collection-contract record")
    contract_reference = reference_for_record(contract_record)
    contract_payload = _mapping(
        contract_record["payload"], where="manifest.collection_contract.payload"
    )
    variant_keys = {
        str(_mapping(row, where="manifest.contract.variant")["key"])
        for row in contract_payload["variants"]  # type: ignore[union-attr]
    }
    receipt_references: list[dict[str, object]] = []
    for index, receipt in enumerate(receipts):
        receipt_record = verify_record(receipt)
        if receipt_record["schema"] != RECEIPT_SCHEMA:
            _fail(f"manifest.receipts[{index}]", "expected a stage-receipt record")
        receipt_payload = _mapping(
            receipt_record["payload"], where=f"manifest.receipts[{index}].payload"
        )
        if receipt_payload["collection_contract"] != contract_reference:
            _fail(
                f"manifest.receipts[{index}].collection_contract",
                "does not bind this collection contract",
            )
        variant_key = receipt_payload["variant_key"]
        if variant_key is not None and variant_key not in variant_keys:
            _fail(
                f"manifest.receipts[{index}].variant_key",
                f"unknown collection variant {variant_key!r}",
            )
        receipt_references.append(reference_for_record(receipt_record))
    payload = {
        "collection_contract": contract_reference,
        "receipts": sorted(receipt_references, key=_reference_sort_key),
    }
    record = seal_record(MANIFEST_SCHEMA, payload, locators=locators)
    return verify_record(record)


def _validate_manifest(payload: Mapping[str, object]) -> None:
    _exact_keys(payload, {"collection_contract", "receipts"}, where="manifest.payload")
    contract = _validate_reference(payload["collection_contract"], where="manifest.collection_contract")
    if contract["subject_schema"] != CONTRACT_SCHEMA:
        _fail("manifest.collection_contract", "does not reference a collection contract")
    receipts = _sorted_unique_references(payload["receipts"], where="manifest.receipts")
    for index, receipt in enumerate(receipts):
        if receipt["subject_schema"] != RECEIPT_SCHEMA:
            _fail(f"manifest.receipts[{index}]", "does not reference a stage receipt")


def assignment_byte_breakdown(
    assignments: Sequence[Mapping[str, object]],
    *,
    fixed_resources: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Price local bytes plus each content-identical shared resource once."""
    local_bytes = 0
    unit_ids: set[str] = set()
    resources: dict[str, dict[str, object]] = {}
    subjects: dict[tuple[str, str], tuple[str, int]] = {}

    def claim(raw: object, *, where: str) -> None:
        reference = _validate_reference(raw, where=where)
        content = _mapping(reference["content"], where=f"{where}.content")
        digest = str(content["sha256"])
        subject = (str(reference["subject_schema"]), str(reference["subject_id"]))
        physical = (digest, int(content["size_bytes"]))
        prior_physical = subjects.get(subject)
        if prior_physical is not None and prior_physical != physical:
            _fail(where, f"semantic reference equivocation for {subject}")
        subjects[subject] = physical
        prior = resources.get(digest)
        if prior is not None and prior["content"]["size_bytes"] != content["size_bytes"]:  # type: ignore[index]
            _fail(where, "same content digest is claimed with inconsistent sizes")
        if prior is None or _reference_sort_key(reference) < _reference_sort_key(prior):
            resources[digest] = reference

    for index, raw in enumerate(assignments):
        row = _mapping(raw, where=f"assignments[{index}]")
        _exact_keys(row, {"unit_id", "candidate_id", "local_bytes", "shared_resources"}, where=f"assignments[{index}]")
        unit_id = _text(row["unit_id"], where=f"assignments[{index}].unit_id")
        if unit_id in unit_ids:
            _fail(f"assignments[{index}].unit_id", f"duplicate unit {unit_id!r}")
        unit_ids.add(unit_id)
        _sha256(row["candidate_id"], where=f"assignments[{index}].candidate_id")
        local_bytes += _nonnegative_int(row["local_bytes"], where=f"assignments[{index}].local_bytes")
        shared = row["shared_resources"]
        if isinstance(shared, (str, bytes)) or not isinstance(shared, Sequence):
            _fail(f"assignments[{index}].shared_resources", "expected an array")
        for resource_index, resource in enumerate(shared):
            claim(resource, where=f"assignments[{index}].shared_resources[{resource_index}]")
    for index, resource in enumerate(fixed_resources):
        claim(resource, where=f"fixed_resources[{index}]")
    ordered_resources = sorted(resources.values(), key=_reference_sort_key)
    shared_bytes = sum(
        int(_mapping(item["content"], where="resource.content")["size_bytes"])
        for item in ordered_resources
    )
    return {
        "schema": BYTE_BREAKDOWN_SCHEMA,
        "unit_count": len(unit_ids),
        "local_bytes": local_bytes,
        "shared_bytes": shared_bytes,
        "total_bytes": local_bytes + shared_bytes,
        "unique_shared_resources": ordered_resources,
    }


_PAYLOAD_VALIDATORS = {
    CANDIDATE_SCHEMA: _validate_candidate,
    CATALOG_SCHEMA: _validate_catalog,
    TARGET_SCHEMA: _validate_target,
    CONTRACT_SCHEMA: _validate_contract,
    RECEIPT_SCHEMA: _validate_receipt,
    MANIFEST_SCHEMA: _validate_manifest,
}
