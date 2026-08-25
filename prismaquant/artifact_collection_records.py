"""Typed records for the artifact-collection control plane.

The foundation in :mod:`prismaquant.artifact_collection` owns envelopes,
references, candidates, targets, and immutable stage receipts.  This module
owns the value-bearing records that make probe-once/export-many operationally
auditable.  It deliberately contains no quantizer or scheduler logic.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import reduce
from operator import mul
from pathlib import PurePosixPath
from typing import Callable

from prismaquant.artifact_collection import (
    AUXILIARY_SCHEMA,
    ArtifactCollectionError,
    CANDIDATE_SCHEMA,
    CATALOG_SCHEMA,
    CONTRACT_SCHEMA,
    COST_SNAPSHOT_SCHEMA,
    DEVICE_QUALIFICATION_SCHEMA,
    EXPORT_SCHEMA,
    LEGACY_AUDIT_SCHEMA,
    MANIFEST_SCHEMA,
    MARKET_SNAPSHOT_SCHEMA,
    MODEL_SNAPSHOT_SCHEMA,
    PROBE_CAMPAIGN_SCHEMA,
    QUALIFICATION_EVIDENCE_SCHEMA,
    RECEIPT_SCHEMA,
    REFERENCE_SCHEMA,
    RELEASE_DECISION_SCHEMA,
    SOLVE_SCHEMA,
    TARGET_SCHEMA,
    UNIT_LEDGER_SCHEMA,
    _canonical_object,
    _canonical_texts,
    _exact_keys,
    _fail,
    _mapping,
    _nonnegative_int,
    _reference_sort_key,
    _sha256,
    _sorted_unique_references,
    _sorted_unique_texts,
    _text,
    _validate_reference,
    assignment_byte_breakdown,
    reference_for_record,
    seal_record,
    verify_record,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256


_DISPOSITIONS = frozenset({"assign", "exclude"})
_COST_STATES = frozenset({"measured", "derived", "unavailable"})
_CHECK_OUTCOMES = frozenset({"passed", "failed"})


def _reference(
    value: object,
    *,
    schema: str | None,
    where: str,
) -> dict[str, object]:
    result = _validate_reference(value, where=where)
    if schema is not None and result["subject_schema"] != schema:
        _fail(where, f"expected a reference to {schema!r}")
    return result


def _record_reference(
    value: Mapping[str, object],
    *,
    schema: str,
    where: str,
) -> tuple[dict[str, object], dict[str, object]]:
    record = verify_record(value)
    if record["schema"] != schema:
        _fail(where, f"expected a {schema!r} record")
    payload = _mapping(record["payload"], where=f"{where}.payload")
    return reference_for_record(record), dict(payload)


def _positive_shape(value: object, *, where: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, "expected a nonempty integer shape")
    result = [
        _nonnegative_int(item, where=f"{where}[{index}]", positive=True)
        for index, item in enumerate(value)
    ]
    if not result:
        _fail(where, "expected a nonempty integer shape")
    return result


def _finite_metrics(value: object, *, where: str) -> dict[str, object]:
    # canonical_json rejects NaN/Inf and non-JSON scalar types.  Empty metrics
    # are meaningful for unavailable cells, so permit them here.
    return _canonical_object(value, where=where, nonempty=False)


def _utc_timestamp(value: object, *, where: str) -> str:
    text = _text(value, where=where)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ArtifactCollectionError(
            f"{where}: expected canonical UTC second timestamp"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        _fail(where, "expected canonical UTC second timestamp")
    return text


def _canonical_references(value: object, *, where: str) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(where, "expected an array of references")
    references = sorted(
        [
            _reference(item, schema=None, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ],
        key=_reference_sort_key,
    )
    _sorted_unique_references(references, where=where)
    return references


def _portable_relative_path(value: object, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or text in {".", ".."}
        or any(part in {".", ".."} for part in path.parts)
        or str(path) != text
    ):
        _fail(where, "expected a canonical portable relative path")
    return text


def make_auxiliary(
    kind: str,
    data: Mapping[str, object],
    *,
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Create a typed envelope for a small external contract or fixture."""

    return seal_record(
        AUXILIARY_SCHEMA,
        {
            "kind": _text(kind, where="auxiliary.kind"),
            "data": _canonical_object(
                data, where="auxiliary.data", nonempty=False
            ),
        },
        locators=locators,
    )


def _validate_auxiliary(payload: Mapping[str, object]) -> None:
    _exact_keys(payload, {"kind", "data"}, where="auxiliary.payload")
    _text(payload["kind"], where="auxiliary.kind")
    _canonical_object(payload["data"], where="auxiliary.data", nonempty=False)


def make_unit_ledger(
    *,
    model_content_sha256: str,
    units: Sequence[Mapping[str, object]],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Seal an exhaustive assignment/exclusion census of source units."""

    rows: list[dict[str, object]] = []
    for index, raw in enumerate(units):
        row = _mapping(raw, where=f"unit_ledger.units[{index}]")
        _exact_keys(
            row,
            {"unit_id", "qname", "shape", "parameter_count", "disposition"},
            where=f"unit_ledger.units[{index}]",
        )
        shape = _positive_shape(
            row["shape"], where=f"unit_ledger.units[{index}].shape"
        )
        parameter_count = _nonnegative_int(
            row["parameter_count"],
            where=f"unit_ledger.units[{index}].parameter_count",
            positive=True,
        )
        if reduce(mul, shape, 1) != parameter_count:
            _fail(
                f"unit_ledger.units[{index}].parameter_count",
                "does not equal the shape product",
            )
        disposition = _text(
            row["disposition"],
            where=f"unit_ledger.units[{index}].disposition",
        )
        if disposition not in _DISPOSITIONS:
            _fail(
                f"unit_ledger.units[{index}].disposition",
                f"expected one of {sorted(_DISPOSITIONS)}",
            )
        rows.append(
            {
                "unit_id": _text(
                    row["unit_id"], where=f"unit_ledger.units[{index}].unit_id"
                ),
                "qname": _text(
                    row["qname"], where=f"unit_ledger.units[{index}].qname"
                ),
                "shape": shape,
                "parameter_count": parameter_count,
                "disposition": disposition,
            }
        )
    if not rows:
        _fail("unit_ledger.units", "must not be empty")
    rows.sort(key=lambda item: str(item["unit_id"]))
    payload = {
        "model_content_sha256": _sha256(
            model_content_sha256, where="unit_ledger.model_content_sha256"
        ),
        "units": rows,
        "unit_count": len(rows),
        "parameter_count": sum(int(row["parameter_count"]) for row in rows),
        "unit_set_sha256": canonical_json_sha256(
            rows, where="unit_ledger.units"
        ),
    }
    return seal_record(UNIT_LEDGER_SCHEMA, payload, locators=locators)


def _validate_unit_ledger(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "model_content_sha256",
            "units",
            "unit_count",
            "parameter_count",
            "unit_set_sha256",
        },
        where="unit_ledger.payload",
    )
    _sha256(payload["model_content_sha256"], where="unit_ledger.model_content_sha256")
    raw_units = payload["units"]
    if isinstance(raw_units, (str, bytes)) or not isinstance(raw_units, Sequence):
        _fail("unit_ledger.units", "expected an array")
    normalized = make_unit_ledger(
        model_content_sha256=str(payload["model_content_sha256"]),
        units=[_mapping(row, where="unit_ledger.units") for row in raw_units],
    )["payload"]
    if dict(payload) != normalized:
        _fail("unit_ledger.payload", "is not the canonical exhaustive ledger")
    unit_ids = [str(row["unit_id"]) for row in raw_units]  # type: ignore[index]
    qnames = [str(row["qname"]) for row in raw_units]  # type: ignore[index]
    if unit_ids != sorted(set(unit_ids)):
        _fail("unit_ledger.units", "unit IDs must be sorted and unique")
    if len(qnames) != len(set(qnames)):
        _fail("unit_ledger.units", "qnames must be unique")


def make_model_snapshot(
    *,
    source_content: Mapping[str, object],
    model_profile: Mapping[str, object],
    unit_ledger: Mapping[str, object],
    source_parameter_count: int,
    source_tree_sha256: str,
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    ledger_ref, ledger = _record_reference(
        unit_ledger,
        schema=UNIT_LEDGER_SCHEMA,
        where="model_snapshot.unit_ledger",
    )
    units = ledger["units"]
    assert isinstance(units, list)
    required = sorted(
        str(row["unit_id"])
        for row in units
        if isinstance(row, Mapping) and row["disposition"] == "assign"
    )
    excluded = sorted(
        str(row["unit_id"])
        for row in units
        if isinstance(row, Mapping) and row["disposition"] == "exclude"
    )
    parameter_count = _nonnegative_int(
        source_parameter_count,
        where="model_snapshot.source_parameter_count",
        positive=True,
    )
    if parameter_count != ledger["parameter_count"]:
        _fail(
            "model_snapshot.source_parameter_count",
            "differs from the exhaustive unit ledger",
        )
    source_reference = _reference(
        source_content, schema=None, where="model_snapshot.source_content"
    )
    source_physical = _mapping(
        source_reference["content"], where="model_snapshot.source_content.content"
    )
    if source_physical["sha256"] != ledger["model_content_sha256"]:
        _fail(
            "model_snapshot.source_content",
            "content digest differs from the unit ledger model digest",
        )
    payload = {
        "source_content": source_reference,
        "model_profile": _reference(
            model_profile, schema=None, where="model_snapshot.model_profile"
        ),
        "unit_ledger": ledger_ref,
        "model_content_sha256": str(ledger["model_content_sha256"]),
        "source_parameter_count": parameter_count,
        "source_tree_sha256": _sha256(
            source_tree_sha256, where="model_snapshot.source_tree_sha256"
        ),
        "assignment_required_unit_ids": required,
        "excluded_unit_ids": excluded,
        "assignment_required_sha256": canonical_json_sha256(
            required, where="model_snapshot.assignment_required_unit_ids"
        ),
        "excluded_sha256": canonical_json_sha256(
            excluded, where="model_snapshot.excluded_unit_ids"
        ),
        "producer": _reference(
            producer, schema=None, where="model_snapshot.producer"
        ),
    }
    return seal_record(MODEL_SNAPSHOT_SCHEMA, payload, locators=locators)


def _validate_model_snapshot(payload: Mapping[str, object]) -> None:
    expected = {
        "source_content",
        "model_profile",
        "unit_ledger",
        "model_content_sha256",
        "source_parameter_count",
        "source_tree_sha256",
        "assignment_required_unit_ids",
        "excluded_unit_ids",
        "assignment_required_sha256",
        "excluded_sha256",
        "producer",
    }
    _exact_keys(payload, expected, where="model_snapshot.payload")
    _reference(payload["source_content"], schema=None, where="model_snapshot.source_content")
    _reference(payload["model_profile"], schema=None, where="model_snapshot.model_profile")
    _reference(payload["unit_ledger"], schema=UNIT_LEDGER_SCHEMA, where="model_snapshot.unit_ledger")
    _reference(payload["producer"], schema=None, where="model_snapshot.producer")
    _sha256(payload["model_content_sha256"], where="model_snapshot.model_content_sha256")
    source_reference = _reference(
        payload["source_content"],
        schema=None,
        where="model_snapshot.source_content",
    )
    source_physical = _mapping(
        source_reference["content"], where="model_snapshot.source_content.content"
    )
    if source_physical["sha256"] != payload["model_content_sha256"]:
        _fail(
            "model_snapshot.source_content",
            "content digest differs from model_content_sha256",
        )
    _nonnegative_int(
        payload["source_parameter_count"],
        where="model_snapshot.source_parameter_count",
        positive=True,
    )
    _sha256(payload["source_tree_sha256"], where="model_snapshot.source_tree_sha256")
    required = _sorted_unique_texts(
        payload["assignment_required_unit_ids"],
        where="model_snapshot.assignment_required_unit_ids",
    )
    excluded = _sorted_unique_texts(
        payload["excluded_unit_ids"], where="model_snapshot.excluded_unit_ids"
    )
    if set(required) & set(excluded):
        _fail("model_snapshot", "required and excluded unit sets overlap")
    if payload["assignment_required_sha256"] != canonical_json_sha256(
        required, where="model_snapshot.assignment_required_unit_ids"
    ):
        _fail("model_snapshot.assignment_required_sha256", "differs")
    if payload["excluded_sha256"] != canonical_json_sha256(
        excluded, where="model_snapshot.excluded_unit_ids"
    ):
        _fail("model_snapshot.excluded_sha256", "differs")


def make_probe_campaign(
    *,
    model_snapshot: Mapping[str, object],
    probe_blob: Mapping[str, object],
    calibration: Mapping[str, object],
    token_content: Mapping[str, object],
    measured_features: Sequence[str],
    covered_unit_ids: Sequence[str],
    missing_unit_ids: Sequence[str],
    merge_receipt: Mapping[str, object],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    covered = _canonical_texts(covered_unit_ids, where="probe.covered_unit_ids")
    missing = _canonical_texts(missing_unit_ids, where="probe.missing_unit_ids")
    if set(covered) & set(missing):
        _fail("probe", "covered and missing unit sets overlap")
    payload = {
        "model_snapshot": _reference(
            model_snapshot, schema=MODEL_SNAPSHOT_SCHEMA, where="probe.model_snapshot"
        ),
        "probe_blob": _reference(probe_blob, schema=None, where="probe.probe_blob"),
        "calibration": _reference(calibration, schema=None, where="probe.calibration"),
        "token_content": _reference(token_content, schema=None, where="probe.token_content"),
        "measured_features": _canonical_texts(
            measured_features, where="probe.measured_features"
        ),
        "covered_unit_ids": covered,
        "missing_unit_ids": missing,
        "merge_receipt": _reference(
            merge_receipt, schema=None, where="probe.merge_receipt"
        ),
        "producer": _reference(producer, schema=None, where="probe.producer"),
    }
    return seal_record(PROBE_CAMPAIGN_SCHEMA, payload, locators=locators)


def _validate_probe_campaign(payload: Mapping[str, object]) -> None:
    expected = {
        "model_snapshot", "probe_blob", "calibration", "token_content",
        "measured_features", "covered_unit_ids", "missing_unit_ids",
        "merge_receipt", "producer",
    }
    _exact_keys(payload, expected, where="probe.payload")
    _reference(payload["model_snapshot"], schema=MODEL_SNAPSHOT_SCHEMA, where="probe.model_snapshot")
    for field in ("probe_blob", "calibration", "token_content", "merge_receipt", "producer"):
        _reference(payload[field], schema=None, where=f"probe.{field}")
    _sorted_unique_texts(payload["measured_features"], where="probe.measured_features")
    covered = _sorted_unique_texts(payload["covered_unit_ids"], where="probe.covered_unit_ids")
    missing = _sorted_unique_texts(payload["missing_unit_ids"], where="probe.missing_unit_ids")
    if set(covered) & set(missing):
        _fail("probe", "covered and missing unit sets overlap")


def make_cost_snapshot(
    *,
    model_snapshot: Mapping[str, object],
    probe_campaign: Mapping[str, object],
    candidate_catalog: Mapping[str, object],
    observations: Mapping[str, object],
    metric_contracts: Mapping[str, object],
    accounting_rule: Mapping[str, object],
    cells: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(cells):
        row = _mapping(raw, where=f"cost.cells[{index}]")
        _exact_keys(
            row,
            {"unit_id", "candidate_id", "status", "metrics", "anchors"},
            where=f"cost.cells[{index}]",
        )
        status = _text(row["status"], where=f"cost.cells[{index}].status")
        if status not in _COST_STATES:
            _fail(f"cost.cells[{index}].status", "unknown cost provenance")
        metrics = _finite_metrics(row["metrics"], where=f"cost.cells[{index}].metrics")
        anchors = _canonical_references(
            row["anchors"], where=f"cost.cells[{index}].anchors"
        )
        if status == "measured" and not metrics:
            _fail(f"cost.cells[{index}].metrics", "measured cell has no metrics")
        if status == "derived" and (not metrics or not anchors):
            _fail(f"cost.cells[{index}]", "derived cell requires metrics and anchors")
        if status == "unavailable" and (metrics or anchors):
            _fail(f"cost.cells[{index}]", "unavailable cell must carry no values")
        if status != "unavailable":
            _nonnegative_int(
                metrics.get("local_bytes"),
                where=f"cost.cells[{index}].metrics.local_bytes",
            )
        normalized.append({
            "unit_id": _text(row["unit_id"], where=f"cost.cells[{index}].unit_id"),
            "candidate_id": _sha256(row["candidate_id"], where=f"cost.cells[{index}].candidate_id"),
            "status": status,
            "metrics": metrics,
            "anchors": anchors,
        })
    normalized.sort(key=lambda row: (str(row["unit_id"]), str(row["candidate_id"])))
    counts = {state: sum(row["status"] == state for row in normalized) for state in sorted(_COST_STATES)}
    payload = {
        "model_snapshot": _reference(model_snapshot, schema=MODEL_SNAPSHOT_SCHEMA, where="cost.model_snapshot"),
        "probe_campaign": _reference(probe_campaign, schema=PROBE_CAMPAIGN_SCHEMA, where="cost.probe_campaign"),
        "candidate_catalog": _reference(candidate_catalog, schema=CATALOG_SCHEMA, where="cost.candidate_catalog"),
        "observations": _reference(observations, schema=None, where="cost.observations"),
        "metric_contracts": _canonical_object(metric_contracts, where="cost.metric_contracts"),
        "accounting_rule": _reference(accounting_rule, schema=None, where="cost.accounting_rule"),
        "cells": normalized,
        "coverage": {"total": len(normalized), **counts},
        "producer": _reference(producer, schema=None, where="cost.producer"),
    }
    return seal_record(COST_SNAPSHOT_SCHEMA, payload, locators=locators)


def _validate_cost_snapshot(payload: Mapping[str, object]) -> None:
    expected = {"model_snapshot", "probe_campaign", "candidate_catalog", "observations", "metric_contracts", "accounting_rule", "cells", "coverage", "producer"}
    _exact_keys(payload, expected, where="cost.payload")
    _reference(payload["model_snapshot"], schema=MODEL_SNAPSHOT_SCHEMA, where="cost.model_snapshot")
    _reference(payload["probe_campaign"], schema=PROBE_CAMPAIGN_SCHEMA, where="cost.probe_campaign")
    _reference(payload["candidate_catalog"], schema=CATALOG_SCHEMA, where="cost.candidate_catalog")
    for field in ("observations", "accounting_rule", "producer"):
        _reference(payload[field], schema=None, where=f"cost.{field}")
    _canonical_object(payload["metric_contracts"], where="cost.metric_contracts")
    rebuilt = make_cost_snapshot(
        model_snapshot=payload["model_snapshot"],
        probe_campaign=payload["probe_campaign"],
        candidate_catalog=payload["candidate_catalog"],
        observations=payload["observations"],
        metric_contracts=payload["metric_contracts"],
        accounting_rule=payload["accounting_rule"],
        cells=payload["cells"],  # type: ignore[arg-type]
        producer=payload["producer"],
    )["payload"]
    if dict(payload) != rebuilt:
        _fail("cost.payload", "is not canonical or its coverage differs")


def make_solve(
    *,
    model_snapshot: Mapping[str, object],
    probe_campaign: Mapping[str, object],
    candidate_catalog: Mapping[str, object],
    cost_snapshot: Mapping[str, object],
    target_profile: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
    fixed_resources: Sequence[Mapping[str, object]],
    solver: Mapping[str, object],
    predicted_metrics: Mapping[str, object],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(assignments):
        row = _mapping(raw, where=f"solve.assignments[{index}]")
        _exact_keys(row, {"unit_id", "candidate_id", "partition", "local_bytes", "shared_resources"}, where=f"solve.assignments[{index}]")
        partition = _text(row["partition"], where=f"solve.assignments[{index}].partition")
        if partition not in {"optimized", "fixed"}:
            _fail(f"solve.assignments[{index}].partition", "expected optimized or fixed")
        shared = _canonical_references(
            row["shared_resources"],
            where=f"solve.assignments[{index}].shared_resources",
        )
        rows.append({
            "unit_id": _text(row["unit_id"], where=f"solve.assignments[{index}].unit_id"),
            "candidate_id": _sha256(row["candidate_id"], where=f"solve.assignments[{index}].candidate_id"),
            "partition": partition,
            "local_bytes": _nonnegative_int(row["local_bytes"], where=f"solve.assignments[{index}].local_bytes"),
            "shared_resources": shared,
        })
    rows.sort(key=lambda row: str(row["unit_id"]))
    fixed = _canonical_references(
        fixed_resources, where="solve.fixed_resources"
    )
    breakdown = assignment_byte_breakdown(
        [
            {
                key: row[key]
                for key in (
                    "unit_id",
                    "candidate_id",
                    "local_bytes",
                    "shared_resources",
                )
            }
            for row in rows
        ],
        fixed_resources=fixed,
    )
    payload = {
        "model_snapshot": _reference(model_snapshot, schema=MODEL_SNAPSHOT_SCHEMA, where="solve.model_snapshot"),
        "probe_campaign": _reference(probe_campaign, schema=PROBE_CAMPAIGN_SCHEMA, where="solve.probe_campaign"),
        "candidate_catalog": _reference(candidate_catalog, schema=CATALOG_SCHEMA, where="solve.candidate_catalog"),
        "cost_snapshot": _reference(cost_snapshot, schema=COST_SNAPSHOT_SCHEMA, where="solve.cost_snapshot"),
        "target_profile": _reference(target_profile, schema=TARGET_SCHEMA, where="solve.target_profile"),
        "assignments": rows,
        "fixed_resources": fixed,
        "assignment_sha256": canonical_json_sha256(rows, where="solve.assignments"),
        "byte_breakdown": breakdown,
        "solver": _canonical_object(solver, where="solve.solver"),
        "predicted_metrics": _finite_metrics(predicted_metrics, where="solve.predicted_metrics"),
        "producer": _reference(producer, schema=None, where="solve.producer"),
    }
    return seal_record(SOLVE_SCHEMA, payload, locators=locators)


def _validate_solve(payload: Mapping[str, object]) -> None:
    expected = {"model_snapshot", "probe_campaign", "candidate_catalog", "cost_snapshot", "target_profile", "assignments", "fixed_resources", "assignment_sha256", "byte_breakdown", "solver", "predicted_metrics", "producer"}
    _exact_keys(payload, expected, where="solve.payload")
    schemas = {"model_snapshot": MODEL_SNAPSHOT_SCHEMA, "probe_campaign": PROBE_CAMPAIGN_SCHEMA, "candidate_catalog": CATALOG_SCHEMA, "cost_snapshot": COST_SNAPSHOT_SCHEMA, "target_profile": TARGET_SCHEMA}
    for field, schema in schemas.items():
        _reference(payload[field], schema=schema, where=f"solve.{field}")
    rebuilt = make_solve(
        model_snapshot=payload["model_snapshot"], probe_campaign=payload["probe_campaign"],
        candidate_catalog=payload["candidate_catalog"], cost_snapshot=payload["cost_snapshot"],
        target_profile=payload["target_profile"], assignments=payload["assignments"],  # type: ignore[arg-type]
        fixed_resources=payload["fixed_resources"], solver=payload["solver"],  # type: ignore[arg-type]
        predicted_metrics=payload["predicted_metrics"], producer=payload["producer"],  # type: ignore[arg-type]
    )["payload"]
    if dict(payload) != rebuilt:
        _fail("solve.payload", "is not canonical or its accounting differs")


def _inventory_rows(value: object, *, kind: str) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"export.{kind}", "expected an array")
    rows: list[dict[str, object]] = []
    expected = (
        {"path", "size_bytes", "sha256"}
        if kind == "files"
        else {"unit_id", "name", "file", "dtype", "shape", "sha256"}
    )
    for index, raw in enumerate(value):
        row = _mapping(raw, where=f"export.{kind}[{index}]")
        _exact_keys(row, expected, where=f"export.{kind}[{index}]")
        if kind == "files":
            rows.append({"path": _portable_relative_path(row["path"], where=f"export.files[{index}].path"), "size_bytes": _nonnegative_int(row["size_bytes"], where=f"export.files[{index}].size_bytes"), "sha256": _sha256(row["sha256"], where=f"export.files[{index}].sha256")})
        else:
            rows.append({"unit_id": _text(row["unit_id"], where=f"export.tensors[{index}].unit_id"), "name": _text(row["name"], where=f"export.tensors[{index}].name"), "file": _portable_relative_path(row["file"], where=f"export.tensors[{index}].file"), "dtype": _text(row["dtype"], where=f"export.tensors[{index}].dtype"), "shape": _positive_shape(row["shape"], where=f"export.tensors[{index}].shape"), "sha256": _sha256(row["sha256"], where=f"export.tensors[{index}].sha256")})
    key = "path" if kind == "files" else "name"
    rows.sort(key=lambda row: str(row[key]))
    if [row[key] for row in rows] != sorted({row[key] for row in rows}):
        _fail(f"export.{kind}", f"{key}s must be unique")
    return rows


def make_export_record(
    *,
    solve: Mapping[str, object], target_profile: Mapping[str, object],
    artifact: Mapping[str, object], files: Sequence[Mapping[str, object]],
    tensors: Sequence[Mapping[str, object]], codebooks: Sequence[Mapping[str, object]],
    runtime_artifact: Mapping[str, object], byte_scope: str,
    measured_bytes: int, producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    solve_ref, solve_payload = _record_reference(
        solve, schema=SOLVE_SCHEMA, where="export.solve"
    )
    file_rows = _inventory_rows(files, kind="files")
    tensor_rows = _inventory_rows(tensors, kind="tensors")
    measured = _nonnegative_int(measured_bytes, where="export.measured_bytes", positive=True)
    if sum(int(row["size_bytes"]) for row in file_rows) != measured:
        _fail("export.measured_bytes", "does not equal the complete file inventory")
    file_names = {str(row["path"]) for row in file_rows}
    if any(str(row["file"]) not in file_names for row in tensor_rows):
        _fail("export.tensors", "a tensor names a file outside the inventory")
    payload = {
        "solve": solve_ref,
        "target_profile": _reference(target_profile, schema=TARGET_SCHEMA, where="export.target_profile"),
        "artifact": _reference(artifact, schema=None, where="export.artifact"),
        "assignment_sha256": _sha256(
            solve_payload["assignment_sha256"],
            where="export.assignment_sha256",
        ),
        "files": file_rows,
        "tensors": tensor_rows,
        "codebooks": _canonical_references(codebooks, where="export.codebooks"),
        "runtime_artifact": _reference(runtime_artifact, schema=None, where="export.runtime_artifact"),
        "byte_measurement": {"scope": _text(byte_scope, where="export.byte_scope"), "bytes": measured},
        "inventory_sha256": canonical_json_sha256({"files": file_rows, "tensors": tensor_rows}, where="export.inventory"),
        "producer": _reference(producer, schema=None, where="export.producer"),
    }
    return seal_record(EXPORT_SCHEMA, payload, locators=locators)


def _validate_export(payload: Mapping[str, object]) -> None:
    expected = {"solve", "target_profile", "artifact", "assignment_sha256", "files", "tensors", "codebooks", "runtime_artifact", "byte_measurement", "inventory_sha256", "producer"}
    _exact_keys(payload, expected, where="export.payload")
    _reference(payload["solve"], schema=SOLVE_SCHEMA, where="export.solve")
    _reference(payload["target_profile"], schema=TARGET_SCHEMA, where="export.target_profile")
    for field in ("artifact", "runtime_artifact", "producer"):
        _reference(payload[field], schema=None, where=f"export.{field}")
    _sha256(payload["assignment_sha256"], where="export.assignment_sha256")
    files = _inventory_rows(payload["files"], kind="files")
    tensors = _inventory_rows(payload["tensors"], kind="tensors")
    _sorted_unique_references(payload["codebooks"], where="export.codebooks")
    measurement = _mapping(payload["byte_measurement"], where="export.byte_measurement")
    _exact_keys(measurement, {"scope", "bytes"}, where="export.byte_measurement")
    _text(measurement["scope"], where="export.byte_measurement.scope")
    measured = _nonnegative_int(measurement["bytes"], where="export.byte_measurement.bytes", positive=True)
    if measured != sum(int(row["size_bytes"]) for row in files):
        _fail("export.byte_measurement.bytes", "does not equal file inventory")
    file_names = {str(row["path"]) for row in files}
    if any(str(row["file"]) not in file_names for row in tensors):
        _fail("export.tensors", "a tensor names a file outside the inventory")
    if payload["inventory_sha256"] != canonical_json_sha256({"files": files, "tensors": tensors}, where="export.inventory"):
        _fail("export.inventory_sha256", "differs")


def make_qualification_evidence(
    *,
    export: Mapping[str, object],
    runtime_contract: Mapping[str, object],
    device_profile: Mapping[str, object],
    workload: Mapping[str, object],
    placement: Mapping[str, object],
    check_id: str,
    outcome: str,
    measurement: Mapping[str, object],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    checked_outcome = _text(outcome, where="qualification_evidence.outcome")
    if checked_outcome not in _CHECK_OUTCOMES:
        _fail("qualification_evidence.outcome", "expected passed or failed")
    payload = {
        "export": _reference(
            export, schema=EXPORT_SCHEMA, where="qualification_evidence.export"
        ),
        "runtime_contract": _reference(
            runtime_contract,
            schema=None,
            where="qualification_evidence.runtime_contract",
        ),
        "device_profile": _reference(
            device_profile,
            schema=None,
            where="qualification_evidence.device_profile",
        ),
        "workload": _reference(
            workload, schema=None, where="qualification_evidence.workload"
        ),
        "placement": _canonical_object(
            placement, where="qualification_evidence.placement"
        ),
        "check_id": _text(check_id, where="qualification_evidence.check_id"),
        "outcome": checked_outcome,
        "measurement": _canonical_object(
            measurement, where="qualification_evidence.measurement"
        ),
        "producer": _reference(
            producer, schema=None, where="qualification_evidence.producer"
        ),
    }
    return seal_record(QUALIFICATION_EVIDENCE_SCHEMA, payload, locators=locators)


def _validate_qualification_evidence(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "export",
            "runtime_contract",
            "device_profile",
            "workload",
            "placement",
            "check_id",
            "outcome",
            "measurement",
            "producer",
        },
        where="qualification_evidence.payload",
    )
    _reference(
        payload["export"],
        schema=EXPORT_SCHEMA,
        where="qualification_evidence.export",
    )
    for field in ("runtime_contract", "device_profile", "workload", "producer"):
        _reference(
            payload[field], schema=None, where=f"qualification_evidence.{field}"
        )
    _canonical_object(
        payload["placement"], where="qualification_evidence.placement"
    )
    _text(payload["check_id"], where="qualification_evidence.check_id")
    outcome = _text(payload["outcome"], where="qualification_evidence.outcome")
    if outcome not in _CHECK_OUTCOMES:
        _fail("qualification_evidence.outcome", "expected passed or failed")
    _canonical_object(
        payload["measurement"], where="qualification_evidence.measurement"
    )


def make_device_qualification(
    *,
    export: Mapping[str, object], target_profile: Mapping[str, object],
    runtime_contract: Mapping[str, object], device_profile: Mapping[str, object],
    workload: Mapping[str, object], placement: Mapping[str, object],
    required_checks: Sequence[str], checks: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    required = _canonical_texts(required_checks, where="qualification.required_checks")
    if not required:
        _fail("qualification.required_checks", "must not be empty")
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(checks):
        row = _mapping(raw, where=f"qualification.checks[{index}]")
        _exact_keys(row, {"id", "outcome", "evidence"}, where=f"qualification.checks[{index}]")
        outcome = _text(row["outcome"], where=f"qualification.checks[{index}].outcome")
        if outcome not in _CHECK_OUTCOMES:
            _fail(f"qualification.checks[{index}].outcome", "expected passed or failed")
        evidence = _canonical_references(
            row["evidence"], where=f"qualification.checks[{index}].evidence"
        )
        if not evidence:
            _fail(f"qualification.checks[{index}].evidence", "must not be empty")
        for evidence_index, reference in enumerate(evidence):
            if reference["subject_schema"] != QUALIFICATION_EVIDENCE_SCHEMA:
                _fail(
                    f"qualification.checks[{index}].evidence[{evidence_index}]",
                    "does not reference qualification evidence",
                )
        normalized.append({"id": _text(row["id"], where=f"qualification.checks[{index}].id"), "outcome": outcome, "evidence": evidence})
    normalized.sort(key=lambda row: str(row["id"]))
    ids = [str(row["id"]) for row in normalized]
    if ids != required:
        _fail("qualification.checks", "check IDs differ from required_checks")
    verdict = "accepted" if all(row["outcome"] == "passed" for row in normalized) else "rejected"
    payload = {
        "export": _reference(export, schema=EXPORT_SCHEMA, where="qualification.export"),
        "target_profile": _reference(target_profile, schema=TARGET_SCHEMA, where="qualification.target_profile"),
        "runtime_contract": _reference(runtime_contract, schema=None, where="qualification.runtime_contract"),
        "device_profile": _reference(device_profile, schema=None, where="qualification.device_profile"),
        "workload": _reference(workload, schema=None, where="qualification.workload"),
        "placement": _canonical_object(placement, where="qualification.placement"),
        "required_checks": required,
        "checks": normalized,
        "verdict": verdict,
        "producer": _reference(producer, schema=None, where="qualification.producer"),
    }
    return seal_record(DEVICE_QUALIFICATION_SCHEMA, payload, locators=locators)


def _validate_device_qualification(payload: Mapping[str, object]) -> None:
    expected = {"export", "target_profile", "runtime_contract", "device_profile", "workload", "placement", "required_checks", "checks", "verdict", "producer"}
    _exact_keys(payload, expected, where="qualification.payload")
    _reference(payload["export"], schema=EXPORT_SCHEMA, where="qualification.export")
    _reference(payload["target_profile"], schema=TARGET_SCHEMA, where="qualification.target_profile")
    for field in ("runtime_contract", "device_profile", "workload", "producer"):
        _reference(payload[field], schema=None, where=f"qualification.{field}")
    _canonical_object(payload["placement"], where="qualification.placement")
    required = _sorted_unique_texts(payload["required_checks"], where="qualification.required_checks")
    if not required:
        _fail("qualification.required_checks", "must not be empty")
    checks = payload["checks"]
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
        _fail("qualification.checks", "expected an array")
    ids: list[str] = []
    outcomes: list[str] = []
    for index, raw in enumerate(checks):
        row = _mapping(raw, where=f"qualification.checks[{index}]")
        _exact_keys(row, {"id", "outcome", "evidence"}, where=f"qualification.checks[{index}]")
        ids.append(_text(row["id"], where=f"qualification.checks[{index}].id"))
        outcome = _text(row["outcome"], where=f"qualification.checks[{index}].outcome")
        if outcome not in _CHECK_OUTCOMES:
            _fail(f"qualification.checks[{index}].outcome", "expected passed or failed")
        outcomes.append(outcome)
        evidence = _sorted_unique_references(row["evidence"], where=f"qualification.checks[{index}].evidence")
        if not evidence:
            _fail(f"qualification.checks[{index}].evidence", "must not be empty")
        for evidence_index, reference in enumerate(evidence):
            if reference["subject_schema"] != QUALIFICATION_EVIDENCE_SCHEMA:
                _fail(
                    f"qualification.checks[{index}].evidence[{evidence_index}]",
                    "does not reference qualification evidence",
                )
    if ids != required:
        _fail("qualification.checks", "check IDs differ from required_checks")
    expected_verdict = "accepted" if all(item == "passed" for item in outcomes) else "rejected"
    if payload["verdict"] != expected_verdict:
        _fail("qualification.verdict", "differs from check outcomes")


def make_market_snapshot(
    *,
    observed_at: str,
    sources: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(sources):
        row = _mapping(raw, where=f"market.sources[{index}]")
        _exact_keys(row, {"source_id", "scope", "collection_method", "raw_receipt", "observations"}, where=f"market.sources[{index}]")
        observations = _canonical_object(row["observations"], where=f"market.sources[{index}].observations")
        rows.append({"source_id": _text(row["source_id"], where=f"market.sources[{index}].source_id"), "scope": _text(row["scope"], where=f"market.sources[{index}].scope"), "collection_method": _text(row["collection_method"], where=f"market.sources[{index}].collection_method"), "raw_receipt": _reference(row["raw_receipt"], schema=None, where=f"market.sources[{index}].raw_receipt"), "observations": observations})
    rows.sort(key=lambda row: str(row["source_id"]))
    if [row["source_id"] for row in rows] != sorted({row["source_id"] for row in rows}):
        _fail("market.sources", "source IDs must be unique")
    payload = {"observed_at": _utc_timestamp(observed_at, where="market.observed_at"), "sources": rows, "producer": _reference(producer, schema=None, where="market.producer")}
    return seal_record(MARKET_SNAPSHOT_SCHEMA, payload, locators=locators)


def _validate_market_snapshot(payload: Mapping[str, object]) -> None:
    _exact_keys(payload, {"observed_at", "sources", "producer"}, where="market.payload")
    _utc_timestamp(payload["observed_at"], where="market.observed_at")
    _reference(payload["producer"], schema=None, where="market.producer")
    rebuilt = make_market_snapshot(observed_at=str(payload["observed_at"]), sources=payload["sources"], producer=payload["producer"])["payload"]  # type: ignore[arg-type]
    if dict(payload) != rebuilt:
        _fail("market.payload", "is not canonical")
    for index, raw in enumerate(payload["sources"]):  # type: ignore[union-attr]
        row = _mapping(raw, where=f"market.sources[{index}]")
        source_id = "".join(
            character
            for character in str(row["source_id"]).lower()
            if character.isalnum()
        )
        scope = "_".join(
            str(row["scope"]).lower().replace("-", " ").split()
        )
        if ("huggingface" in source_id or source_id == "hf") and (
            "installed_base" in scope or "unique_machine" in scope
        ):
            _fail(
                f"market.sources[{index}].scope",
                "Hugging Face incidences are not an installed-base or "
                "unique-machine measurement",
            )


def make_release_decision(
    *,
    collection_contract: Mapping[str, object],
    market_snapshot: Mapping[str, object], policy: Mapping[str, object],
    included: Sequence[Mapping[str, object]], rejected: Sequence[Mapping[str, object]],
    producer: Mapping[str, object],
    locators: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    accepted_rows: list[dict[str, object]] = []
    for index, raw in enumerate(included):
        row = _mapping(raw, where=f"release.included[{index}]")
        _exact_keys(row, {"variant_key", "alias", "target_profile", "export", "qualification"}, where=f"release.included[{index}]")
        accepted_rows.append({"variant_key": _text(row["variant_key"], where=f"release.included[{index}].variant_key"), "alias": _text(row["alias"], where=f"release.included[{index}].alias"), "target_profile": _reference(row["target_profile"], schema=TARGET_SCHEMA, where=f"release.included[{index}].target_profile"), "export": _reference(row["export"], schema=EXPORT_SCHEMA, where=f"release.included[{index}].export"), "qualification": _reference(row["qualification"], schema=DEVICE_QUALIFICATION_SCHEMA, where=f"release.included[{index}].qualification")})
    accepted_rows.sort(key=lambda row: str(row["variant_key"]))
    rejected_rows: list[dict[str, object]] = []
    for index, raw in enumerate(rejected):
        row = _mapping(raw, where=f"release.rejected[{index}]")
        _exact_keys(row, {"variant_key", "target_profile", "reason"}, where=f"release.rejected[{index}]")
        rejected_rows.append({"variant_key": _text(row["variant_key"], where=f"release.rejected[{index}].variant_key"), "target_profile": _reference(row["target_profile"], schema=TARGET_SCHEMA, where=f"release.rejected[{index}].target_profile"), "reason": _text(row["reason"], where=f"release.rejected[{index}].reason")})
    rejected_rows.sort(key=lambda row: str(row["variant_key"]))
    keys = [str(row["variant_key"]) for row in (*accepted_rows, *rejected_rows)]
    if len(keys) != len(set(keys)):
        _fail("release", "a variant is both included/rejected or duplicated")
    if not keys:
        _fail("release", "decision universe must not be empty")
    payload = {"collection_contract": _reference(collection_contract, schema=CONTRACT_SCHEMA, where="release.collection_contract"), "market_snapshot": _reference(market_snapshot, schema=MARKET_SNAPSHOT_SCHEMA, where="release.market_snapshot"), "policy": _reference(policy, schema=None, where="release.policy"), "included": accepted_rows, "rejected": rejected_rows, "producer": _reference(producer, schema=None, where="release.producer")}
    return seal_record(RELEASE_DECISION_SCHEMA, payload, locators=locators)


def _validate_release_decision(payload: Mapping[str, object]) -> None:
    _exact_keys(payload, {"collection_contract", "market_snapshot", "policy", "included", "rejected", "producer"}, where="release.payload")
    _reference(payload["collection_contract"], schema=CONTRACT_SCHEMA, where="release.collection_contract")
    _reference(payload["market_snapshot"], schema=MARKET_SNAPSHOT_SCHEMA, where="release.market_snapshot")
    _reference(payload["policy"], schema=None, where="release.policy")
    _reference(payload["producer"], schema=None, where="release.producer")
    rebuilt = make_release_decision(collection_contract=payload["collection_contract"], market_snapshot=payload["market_snapshot"], policy=payload["policy"], included=payload["included"], rejected=payload["rejected"], producer=payload["producer"])["payload"]  # type: ignore[arg-type]
    if dict(payload) != rebuilt:
        _fail("release.payload", "is not canonical")


def verify_collection_graph(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Resolve and reconcile one closed set of collection records.

    This is intentionally a pure verifier.  It never follows locators or
    mutates a status file; callers must provide every referenced semantic
    record whose payload participates in a cross-record invariant.
    """

    indexed: dict[tuple[str, str], dict[str, object]] = {}
    references: dict[tuple[str, str], dict[str, object]] = {}
    for index, raw in enumerate(records):
        record = verify_record(raw)
        key = (str(record["schema"]), str(record["payload_sha256"]))
        reference = reference_for_record(record)
        prior = references.get(key)
        if prior is not None:
            qualifier = (
                "equivocates on physical content"
                if prior != reference
                else "is supplied more than once"
            )
            _fail(f"graph.records[{index}]", f"semantic record {key} {qualifier}")
        indexed[key] = record
        references[key] = reference

    def resolve(
        raw: object,
        *,
        schema: str | None,
        where: str,
    ) -> dict[str, object]:
        reference = _reference(raw, schema=schema, where=where)
        key = (
            str(reference["subject_schema"]),
            str(reference["subject_id"]),
        )
        record = indexed.get(key)
        if record is None:
            _fail(where, f"referenced semantic record {key} is absent")
        if reference_for_record(record) != reference:
            _fail(where, "reference content differs from the supplied record")
        return record

    def payload(record: Mapping[str, object], *, where: str) -> Mapping[str, object]:
        return _mapping(record["payload"], where=where)

    # This verifier consumes a closed semantic set.  External byte contracts
    # may remain references, but every reference to a record type owned by
    # this control plane must resolve to the exact supplied portable bytes.
    owned_schemas = set(PAYLOAD_VALIDATORS) | {
        CANDIDATE_SCHEMA,
        CATALOG_SCHEMA,
        TARGET_SCHEMA,
        CONTRACT_SCHEMA,
        RECEIPT_SCHEMA,
        MANIFEST_SCHEMA,
        LEGACY_AUDIT_SCHEMA,
    }
    graph_subjects: dict[tuple[str, str], tuple[str, int]] = {}
    graph_content_sizes: dict[str, int] = {}

    def require_closed(value: object, *, where: str) -> None:
        if isinstance(value, Mapping):
            if value.get("schema") == REFERENCE_SCHEMA:
                reference = _reference(value, schema=None, where=where)
                subject = (
                    str(reference["subject_schema"]),
                    str(reference["subject_id"]),
                )
                content = _mapping(reference["content"], where=f"{where}.content")
                physical = (str(content["sha256"]), int(content["size_bytes"]))
                prior_physical = graph_subjects.get(subject)
                if prior_physical is not None and prior_physical != physical:
                    _fail(where, f"semantic reference equivocation for {subject}")
                graph_subjects[subject] = physical
                prior_size = graph_content_sizes.get(physical[0])
                if prior_size is not None and prior_size != physical[1]:
                    _fail(
                        where,
                        "one physical content digest has inconsistent sizes",
                    )
                graph_content_sizes[physical[0]] = physical[1]
                if str(reference["subject_schema"]) in owned_schemas:
                    resolve(reference, schema=None, where=where)
                return
            for field, child in value.items():
                require_closed(child, where=f"{where}.{field}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                require_closed(child, where=f"{where}[{index}]")

    for index, record in enumerate(indexed.values()):
        require_closed(record["payload"], where=f"graph.records[{index}].payload")

    # Collection contracts are joins, not bags of independently valid refs.
    for key, record in indexed.items():
        if key[0] != CONTRACT_SCHEMA:
            continue
        contract = payload(record, where="graph.contract")
        model_record = resolve(
            contract["model_snapshot"],
            schema=MODEL_SNAPSHOT_SCHEMA,
            where="graph.contract.model_snapshot",
        )
        probe_record = resolve(
            contract["probe_campaign"],
            schema=PROBE_CAMPAIGN_SCHEMA,
            where="graph.contract.probe_campaign",
        )
        resolve(
            contract["candidate_catalog"],
            schema=CATALOG_SCHEMA,
            where="graph.contract.candidate_catalog",
        )
        cost_record = resolve(
            contract["cost_snapshot"],
            schema=COST_SNAPSHOT_SCHEMA,
            where="graph.contract.cost_snapshot",
        )
        probe = payload(probe_record, where="graph.contract.probe")
        cost = payload(cost_record, where="graph.contract.cost")
        if probe["model_snapshot"] != reference_for_record(model_record):
            _fail("graph.contract", "probe binds a different model snapshot")
        for field in (
            "model_snapshot",
            "probe_campaign",
            "candidate_catalog",
        ):
            if cost[field] != contract[field]:
                _fail("graph.contract", f"cost binds a different {field}")
        if cost["accounting_rule"] != contract["accounting_rule"]:
            _fail("graph.contract", "cost accounting rule differs")
        for index, raw_variant in enumerate(contract["variants"]):  # type: ignore[union-attr]
            variant = _mapping(
                raw_variant, where=f"graph.contract.variants[{index}]"
            )
            target_record = resolve(
                variant["target_profile"],
                schema=TARGET_SCHEMA,
                where=f"graph.contract.variants[{index}].target_profile",
            )
            target = payload(target_record, where="graph.contract.target")
            if (
                variant["accounting_rule"] != contract["accounting_rule"]
                or target["accounting_rule"] != contract["accounting_rule"]
            ):
                _fail(
                    f"graph.contract.variants[{index}]",
                    "target accounting rule differs",
                )

    # Receipts and manifests are immutable evidence indexes over one contract.
    for key, record in indexed.items():
        if key[0] != RECEIPT_SCHEMA:
            continue
        receipt = payload(record, where="graph.receipt")
        contract_record = resolve(
            receipt["collection_contract"],
            schema=CONTRACT_SCHEMA,
            where="graph.receipt.collection_contract",
        )
        contract = payload(contract_record, where="graph.receipt.contract")
        variants = {
            str(_mapping(row, where="graph.receipt.variant")["key"])
            for row in contract["variants"]  # type: ignore[union-attr]
        }
        variant_key = receipt["variant_key"]
        if variant_key is not None and variant_key not in variants:
            _fail(
                "graph.receipt.variant_key",
                f"unknown collection variant {variant_key!r}",
            )
        expected_output_schema = {
            "measure": PROBE_CAMPAIGN_SCHEMA,
            "solve": SOLVE_SCHEMA,
            "export": EXPORT_SCHEMA,
            "qualify": DEVICE_QUALIFICATION_SCHEMA,
            "publish": RELEASE_DECISION_SCHEMA,
        }.get(str(receipt["stage"]))
        if receipt["outcome"] == "accepted" and expected_output_schema is not None:
            output_schemas = {
                str(_mapping(ref, where="graph.receipt.output")["subject_schema"])
                for ref in receipt["outputs"]  # type: ignore[union-attr]
            }
            if expected_output_schema not in output_schemas:
                _fail(
                    "graph.receipt.outputs",
                    f"accepted {receipt['stage']} receipt lacks "
                    f"{expected_output_schema!r}",
                )

    for key, record in indexed.items():
        if key[0] != MANIFEST_SCHEMA:
            continue
        manifest = payload(record, where="graph.manifest")
        contract_record = resolve(
            manifest["collection_contract"],
            schema=CONTRACT_SCHEMA,
            where="graph.manifest.collection_contract",
        )
        contract_reference = reference_for_record(contract_record)
        for index, raw_reference in enumerate(manifest["receipts"]):  # type: ignore[union-attr]
            receipt_record = resolve(
                raw_reference,
                schema=RECEIPT_SCHEMA,
                where=f"graph.manifest.receipts[{index}]",
            )
            receipt = payload(receipt_record, where="graph.manifest.receipt")
            if receipt["collection_contract"] != contract_reference:
                _fail(
                    f"graph.manifest.receipts[{index}]",
                    "receipt binds a different collection contract",
                )

    # Model snapshots must reproduce the exhaustive ledger projection.
    for key, record in indexed.items():
        if key[0] != MODEL_SNAPSHOT_SCHEMA:
            continue
        model = payload(record, where="graph.model_snapshot")
        ledger_record = resolve(
            model["unit_ledger"],
            schema=UNIT_LEDGER_SCHEMA,
            where="graph.model_snapshot.unit_ledger",
        )
        ledger = payload(ledger_record, where="graph.unit_ledger")
        if model["model_content_sha256"] != ledger["model_content_sha256"]:
            _fail("graph.model_snapshot", "model content differs from unit ledger")
        if model["source_parameter_count"] != ledger["parameter_count"]:
            _fail(
                "graph.model_snapshot",
                "source parameter count differs from unit ledger",
            )
        units = ledger["units"]
        assert isinstance(units, list)
        required = sorted(
            str(row["unit_id"])
            for row in units
            if isinstance(row, Mapping) and row["disposition"] == "assign"
        )
        excluded = sorted(
            str(row["unit_id"])
            for row in units
            if isinstance(row, Mapping) and row["disposition"] == "exclude"
        )
        if model["assignment_required_unit_ids"] != required or model[
            "excluded_unit_ids"
        ] != excluded:
            _fail("graph.model_snapshot", "unit-ledger projection differs")

    # Probe coverage is exhaustive over assignment-required units.
    for key, record in indexed.items():
        if key[0] != PROBE_CAMPAIGN_SCHEMA:
            continue
        probe = payload(record, where="graph.probe")
        model_record = resolve(
            probe["model_snapshot"],
            schema=MODEL_SNAPSHOT_SCHEMA,
            where="graph.probe.model_snapshot",
        )
        model = payload(model_record, where="graph.probe.model")
        observed = set(probe["covered_unit_ids"]) | set(probe["missing_unit_ids"])
        expected = set(model["assignment_required_unit_ids"])
        if observed != expected:
            _fail("graph.probe", "covered+missing units do not equal model requirements")

    # Costs may name only model units and catalog candidates, once per cell.
    for key, record in indexed.items():
        if key[0] != COST_SNAPSHOT_SCHEMA:
            continue
        cost = payload(record, where="graph.cost")
        model_record = resolve(cost["model_snapshot"], schema=MODEL_SNAPSHOT_SCHEMA, where="graph.cost.model_snapshot")
        probe_record = resolve(cost["probe_campaign"], schema=PROBE_CAMPAIGN_SCHEMA, where="graph.cost.probe_campaign")
        catalog_record = resolve(cost["candidate_catalog"], schema=CATALOG_SCHEMA, where="graph.cost.candidate_catalog")
        probe = payload(probe_record, where="graph.cost.probe")
        if probe["model_snapshot"] != cost["model_snapshot"]:
            _fail("graph.cost", "probe and cost bind different model snapshots")
        model = payload(model_record, where="graph.cost.model")
        unit_ids = set(model["assignment_required_unit_ids"])
        catalog = payload(catalog_record, where="graph.cost.catalog")
        candidates: dict[str, Mapping[str, object]] = {}
        for raw_reference in catalog["candidates"]:  # type: ignore[union-attr]
            candidate_record = resolve(
                raw_reference,
                schema=CANDIDATE_SCHEMA,
                where="graph.cost.candidate",
            )
            candidates[str(candidate_record["payload_sha256"])] = payload(
                candidate_record, where="graph.cost.candidate.payload"
            )
        candidate_ids = set(candidates)
        seen: set[tuple[str, str]] = set()
        measured_features = set(probe["measured_features"])
        missing_units = set(probe["missing_unit_ids"])
        for index, raw_cell in enumerate(cost["cells"]):  # type: ignore[union-attr]
            cell = _mapping(raw_cell, where=f"graph.cost.cells[{index}]")
            pair = (str(cell["unit_id"]), str(cell["candidate_id"]))
            if pair in seen:
                _fail("graph.cost.cells", f"duplicate cell {pair}")
            seen.add(pair)
            if pair[0] not in unit_ids or pair[1] not in candidate_ids:
                _fail("graph.cost.cells", f"cell {pair} is outside model/catalog")
            if cell["status"] != "unavailable":
                candidate = candidates[pair[1]]
                if (
                    not set(candidate["required_probe_features"])
                    <= measured_features
                    or pair[0] in missing_units
                ):
                    _fail(
                        "graph.cost.cells",
                        f"probe is insufficient for usable cost cell {pair}",
                    )
                metrics = _mapping(
                    cell["metrics"], where=f"graph.cost.cells[{index}].metrics"
                )
                _nonnegative_int(
                    metrics.get("local_bytes"),
                    where=f"graph.cost.cells[{index}].metrics.local_bytes",
                )
        expected_pairs = {
            (unit_id, candidate_id)
            for unit_id in unit_ids
            for candidate_id in candidate_ids
        }
        if seen != expected_pairs:
            missing = sorted(expected_pairs - seen)
            extra = sorted(seen - expected_pairs)
            _fail(
                "graph.cost.cells",
                f"cost matrix is not exhaustive (missing={missing}, extra={extra})",
            )

    # A solve is exhaustive, cost-backed, feature-sufficient, and under its
    # target's exact byte ceiling.
    for key, record in indexed.items():
        if key[0] != SOLVE_SCHEMA:
            continue
        solve = payload(record, where="graph.solve")
        model_record = resolve(solve["model_snapshot"], schema=MODEL_SNAPSHOT_SCHEMA, where="graph.solve.model_snapshot")
        probe_record = resolve(solve["probe_campaign"], schema=PROBE_CAMPAIGN_SCHEMA, where="graph.solve.probe_campaign")
        catalog_record = resolve(solve["candidate_catalog"], schema=CATALOG_SCHEMA, where="graph.solve.candidate_catalog")
        cost_record = resolve(solve["cost_snapshot"], schema=COST_SNAPSHOT_SCHEMA, where="graph.solve.cost_snapshot")
        target_record = resolve(solve["target_profile"], schema=TARGET_SCHEMA, where="graph.solve.target_profile")
        model = payload(model_record, where="graph.solve.model")
        probe = payload(probe_record, where="graph.solve.probe")
        catalog = payload(catalog_record, where="graph.solve.catalog")
        cost = payload(cost_record, where="graph.solve.cost")
        target = payload(target_record, where="graph.solve.target")
        for field in ("model_snapshot", "probe_campaign", "candidate_catalog"):
            if cost[field] != solve[field]:
                _fail("graph.solve", f"cost and solve bind different {field}")
        expected_units = set(model["assignment_required_unit_ids"])
        rows = solve["assignments"]
        assert isinstance(rows, list)
        assigned_units = [str(row["unit_id"]) for row in rows if isinstance(row, Mapping)]
        if set(assigned_units) != expected_units or len(assigned_units) != len(expected_units):
            _fail("graph.solve.assignments", "must assign every required unit exactly once")
        candidates: dict[str, Mapping[str, object]] = {}
        for raw_ref in catalog["candidates"]:  # type: ignore[union-attr]
            candidate_record = resolve(raw_ref, schema=CANDIDATE_SCHEMA, where="graph.solve.candidate")
            candidates[str(candidate_record["payload_sha256"])] = payload(candidate_record, where="graph.solve.candidate.payload")
        available_cells = {
            (str(cell["unit_id"]), str(cell["candidate_id"])): cell
            for cell in cost["cells"]  # type: ignore[union-attr]
            if isinstance(cell, Mapping) and cell["status"] != "unavailable"
        }
        measured_features = set(probe["measured_features"])
        missing_units = set(probe["missing_unit_ids"])
        for row in rows:
            assert isinstance(row, Mapping)
            pair = (str(row["unit_id"]), str(row["candidate_id"]))
            if pair not in available_cells:
                _fail("graph.solve.assignments", f"choice {pair} has no usable cost cell")
            candidate = candidates.get(pair[1])
            if candidate is None:
                _fail("graph.solve.assignments", f"unknown candidate {pair[1]}")
            required_features = set(candidate["required_probe_features"])
            if not required_features <= measured_features or pair[0] in missing_units:
                _fail("graph.solve.assignments", f"probe is insufficient for choice {pair}")
            metrics = _mapping(
                available_cells[pair]["metrics"],
                where="graph.solve.cost_cell.metrics",
            )
            if row["local_bytes"] != metrics["local_bytes"]:
                _fail(
                    "graph.solve.assignments",
                    f"choice {pair} local bytes differ from its cost cell",
                )
            if row["shared_resources"] != candidate["shared_resources"]:
                _fail(
                    "graph.solve.assignments",
                    f"choice {pair} shared resources differ from its candidate",
                )
        if cost["accounting_rule"] != target["accounting_rule"]:
            _fail("graph.solve", "cost and target accounting rules differ")
        if solve["fixed_resources"] != target["fixed_resources"]:
            _fail("graph.solve", "fixed resources differ from target authority")
        breakdown = _mapping(solve["byte_breakdown"], where="graph.solve.byte_breakdown")
        if int(breakdown["total_bytes"]) > int(target["artifact_byte_ceiling"]):
            _fail("graph.solve.byte_breakdown", "exceeds target artifact ceiling")

    # Exports and qualifications cannot be retargeted after the solve.
    for key, record in indexed.items():
        if key[0] != EXPORT_SCHEMA:
            continue
        export = payload(record, where="graph.export")
        solve_record = resolve(export["solve"], schema=SOLVE_SCHEMA, where="graph.export.solve")
        target_record = resolve(export["target_profile"], schema=TARGET_SCHEMA, where="graph.export.target_profile")
        solve = payload(solve_record, where="graph.export.solve.payload")
        target = payload(target_record, where="graph.export.target.payload")
        if export["target_profile"] != solve["target_profile"] or export["assignment_sha256"] != solve["assignment_sha256"]:
            _fail("graph.export", "target or assignment differs from solve")
        measured = _mapping(export["byte_measurement"], where="graph.export.byte_measurement")
        if measured["scope"] != target["artifact_byte_scope"]:
            _fail("graph.export", "byte measurement scope differs from target")
        if int(measured["bytes"]) > int(target["artifact_byte_ceiling"]):
            _fail("graph.export", "whole export exceeds target artifact ceiling")
        model_record = resolve(
            solve["model_snapshot"],
            schema=MODEL_SNAPSHOT_SCHEMA,
            where="graph.export.model_snapshot",
        )
        model = payload(model_record, where="graph.export.model")
        ledger_record = resolve(
            model["unit_ledger"],
            schema=UNIT_LEDGER_SCHEMA,
            where="graph.export.unit_ledger",
        )
        ledger = payload(ledger_record, where="graph.export.ledger")
        unit_qnames = {
            str(row["unit_id"]): str(row["qname"])
            for row in ledger["units"]  # type: ignore[union-attr]
            if isinstance(row, Mapping)
        }
        tensors = export["tensors"]
        assert isinstance(tensors, list)
        tensor_units = {
            str(row["unit_id"])
            for row in tensors
            if isinstance(row, Mapping)
        }
        if tensor_units != set(unit_qnames):
            _fail(
                "graph.export.tensors",
                "tensor unit coverage differs from the exhaustive ledger",
            )
        for unit_id, qname in unit_qnames.items():
            if not any(
                isinstance(row, Mapping)
                and row["unit_id"] == unit_id
                and row["name"] == qname
                for row in tensors
            ):
                _fail(
                    "graph.export.tensors",
                    f"unit {unit_id!r} has no primary tensor named {qname!r}",
                )
        catalog_record = resolve(
            solve["candidate_catalog"],
            schema=CATALOG_SCHEMA,
            where="graph.export.candidate_catalog",
        )
        catalog = payload(catalog_record, where="graph.export.catalog")
        candidates: dict[str, Mapping[str, object]] = {}
        for raw_reference in catalog["candidates"]:  # type: ignore[union-attr]
            candidate_record = resolve(
                raw_reference,
                schema=CANDIDATE_SCHEMA,
                where="graph.export.candidate",
            )
            candidates[str(candidate_record["payload_sha256"])] = payload(
                candidate_record, where="graph.export.candidate.payload"
            )
        expected_codebooks: dict[tuple[str, str, str], dict[str, object]] = {}
        for raw_assignment in solve["assignments"]:  # type: ignore[union-attr]
            assignment = _mapping(
                raw_assignment, where="graph.export.assignment"
            )
            candidate = candidates[str(assignment["candidate_id"])]
            if candidate["runtime_contract"] != export["runtime_artifact"]:
                _fail(
                    "graph.export.runtime_artifact",
                    "differs from a selected candidate runtime contract",
                )
            basis = _mapping(candidate["basis"], where="graph.export.candidate.basis")
            for raw_reference in basis["assets"]:  # type: ignore[union-attr]
                reference = _reference(
                    raw_reference, schema=None, where="graph.export.codebook"
                )
                expected_codebooks[_reference_sort_key(reference)] = reference
        if export["codebooks"] != [
            expected_codebooks[key] for key in sorted(expected_codebooks)
        ]:
            _fail(
                "graph.export.codebooks",
                "inventory differs from selected candidate basis assets",
            )

    for key, record in indexed.items():
        if key[0] != DEVICE_QUALIFICATION_SCHEMA:
            continue
        qualification = payload(record, where="graph.qualification")
        export_record = resolve(qualification["export"], schema=EXPORT_SCHEMA, where="graph.qualification.export")
        target_record = resolve(qualification["target_profile"], schema=TARGET_SCHEMA, where="graph.qualification.target_profile")
        export = payload(export_record, where="graph.qualification.export.payload")
        target = payload(target_record, where="graph.qualification.target.payload")
        if export["target_profile"] != qualification["target_profile"]:
            _fail("graph.qualification", "export and qualification targets differ")
        if qualification["device_profile"] != target["device_profile"] or qualification["workload"] != target["workload"]:
            _fail("graph.qualification", "device/workload differs from target")
        if qualification["placement"] != target["placement_constraints"]:
            _fail("graph.qualification", "placement differs from target")
        if qualification["runtime_contract"] != export["runtime_artifact"]:
            _fail("graph.qualification", "runtime contract differs from export")
        if qualification["required_checks"] != target["required_qualification_checks"]:
            _fail("graph.qualification", "required checks differ from target policy")
        for check_index, raw_check in enumerate(qualification["checks"]):  # type: ignore[union-attr]
            check = _mapping(
                raw_check, where=f"graph.qualification.checks[{check_index}]"
            )
            for evidence_index, raw_reference in enumerate(check["evidence"]):  # type: ignore[union-attr]
                evidence_record = resolve(
                    raw_reference,
                    schema=QUALIFICATION_EVIDENCE_SCHEMA,
                    where=(
                        f"graph.qualification.checks[{check_index}]."
                        f"evidence[{evidence_index}]"
                    ),
                )
                evidence = payload(
                    evidence_record, where="graph.qualification.evidence"
                )
                expected_bindings = {
                    "export": qualification["export"],
                    "runtime_contract": qualification["runtime_contract"],
                    "device_profile": qualification["device_profile"],
                    "workload": qualification["workload"],
                    "placement": qualification["placement"],
                    "check_id": check["id"],
                    "outcome": check["outcome"],
                }
                if any(
                    evidence[field] != expected
                    for field, expected in expected_bindings.items()
                ):
                    _fail(
                        "graph.qualification.evidence",
                        "evidence bindings differ from qualification",
                    )

    for key, record in indexed.items():
        if key[0] != RELEASE_DECISION_SCHEMA:
            continue
        release = payload(record, where="graph.release")
        contract_record = resolve(
            release["collection_contract"],
            schema=CONTRACT_SCHEMA,
            where="graph.release.collection_contract",
        )
        contract = payload(contract_record, where="graph.release.contract")
        resolve(release["market_snapshot"], schema=MARKET_SNAPSHOT_SCHEMA, where="graph.release.market_snapshot")
        variant_targets = {
            str(row["key"]): row["target_profile"]
            for row in contract["variants"]  # type: ignore[union-attr]
            if isinstance(row, Mapping)
        }
        decided_variants: set[str] = set()
        aliases: set[str] = set()
        for index, raw_row in enumerate(release["included"]):  # type: ignore[union-attr]
            row = _mapping(raw_row, where=f"graph.release.included[{index}]")
            export_record = resolve(row["export"], schema=EXPORT_SCHEMA, where=f"graph.release.included[{index}].export")
            qualification_record = resolve(row["qualification"], schema=DEVICE_QUALIFICATION_SCHEMA, where=f"graph.release.included[{index}].qualification")
            target_record = resolve(row["target_profile"], schema=TARGET_SCHEMA, where=f"graph.release.included[{index}].target")
            export = payload(export_record, where="graph.release.export")
            qualification = payload(qualification_record, where="graph.release.qualification")
            variant_key = str(row["variant_key"])
            if variant_targets.get(variant_key) != row["target_profile"]:
                _fail(
                    "graph.release",
                    f"included variant {variant_key!r} differs from contract",
                )
            if qualification["verdict"] != "accepted":
                _fail("graph.release", "included artifact is not qualified")
            if qualification["export"] != row["export"] or qualification["target_profile"] != row["target_profile"] or export["target_profile"] != row["target_profile"]:
                _fail("graph.release", "included export/qualification/target differ")
            solve_record = resolve(
                export["solve"],
                schema=SOLVE_SCHEMA,
                where=f"graph.release.included[{index}].solve",
            )
            solve = payload(solve_record, where="graph.release.solve")
            for field in (
                "model_snapshot",
                "probe_campaign",
                "candidate_catalog",
                "cost_snapshot",
            ):
                if solve[field] != contract[field]:
                    _fail(
                        "graph.release",
                        f"included solve differs from contract {field}",
                    )
            alias = str(row["alias"])
            if alias in aliases:
                _fail("graph.release", f"duplicate artifact alias {alias!r}")
            aliases.add(alias)
            decided_variants.add(variant_key)
            del target_record  # resolved above for exact content binding
        for index, raw_row in enumerate(release["rejected"]):  # type: ignore[union-attr]
            row = _mapping(raw_row, where=f"graph.release.rejected[{index}]")
            target_record = resolve(
                row["target_profile"],
                schema=TARGET_SCHEMA,
                where=f"graph.release.rejected[{index}].target",
            )
            variant_key = str(row["variant_key"])
            if variant_targets.get(variant_key) != row["target_profile"]:
                _fail(
                    "graph.release",
                    f"rejected variant {variant_key!r} differs from contract",
                )
            decided_variants.add(variant_key)
            del target_record
        if decided_variants != set(variant_targets):
            _fail(
                "graph.release",
                "included and rejected variants do not exhaust the contract",
            )

    return {
        "schema": "prismaquant.artifact_collection.graph_verification.v1",
        "record_count": len(indexed),
        "schemas": {
            schema: sum(key[0] == schema for key in indexed)
            for schema in sorted({key[0] for key in indexed})
        },
        "identity_sha256": canonical_json_sha256(
            sorted([list(key) for key in indexed]), where="collection graph"
        ),
    }


PAYLOAD_VALIDATORS: dict[str, Callable[[Mapping[str, object]], None]] = {
    AUXILIARY_SCHEMA: _validate_auxiliary,
    UNIT_LEDGER_SCHEMA: _validate_unit_ledger,
    MODEL_SNAPSHOT_SCHEMA: _validate_model_snapshot,
    PROBE_CAMPAIGN_SCHEMA: _validate_probe_campaign,
    COST_SNAPSHOT_SCHEMA: _validate_cost_snapshot,
    SOLVE_SCHEMA: _validate_solve,
    EXPORT_SCHEMA: _validate_export,
    QUALIFICATION_EVIDENCE_SCHEMA: _validate_qualification_evidence,
    DEVICE_QUALIFICATION_SCHEMA: _validate_device_qualification,
    MARKET_SNAPSHOT_SCHEMA: _validate_market_snapshot,
    RELEASE_DECISION_SCHEMA: _validate_release_decision,
}


__all__ = [
    "make_auxiliary",
    "make_unit_ledger",
    "make_model_snapshot",
    "make_probe_campaign",
    "make_cost_snapshot",
    "make_solve",
    "make_export_record",
    "make_qualification_evidence",
    "make_device_qualification",
    "make_market_snapshot",
    "make_release_decision",
    "verify_collection_graph",
]
