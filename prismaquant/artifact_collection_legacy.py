"""Read-only importer for pre-collection PrismaQuant export metadata.

Legacy exports conflate several byte and unit scopes.  This adapter preserves
the recorded producer inventory, independently observes the current package,
and names each census explicitly.  It never repairs or rewrites an artifact.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from prismaquant.artifact_collection import (
    ArtifactCollectionError,
    LEGACY_AUDIT_SCHEMA,
    make_reference,
    seal_record,
    verify_record,
)


_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}


def _fail(where: str, message: str) -> None:
    raise ArtifactCollectionError(f"{where}: {message}")


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(where, "expected a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, where: str) -> None:
    if set(value) != expected:
        _fail(
            where,
            "field set differs "
            f"(missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)})",
        )


def _integer(value: object, *, where: str, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected an integer")
    if nonnegative and value < 0:
        _fail(where, "expected a non-negative integer")
    return value


def _reject_duplicate_members(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate member {key!r}")
        result[key] = value
    return result


def _load_json_source(
    path: Path, *, logical_schema: str
) -> tuple[dict[str, object], dict[str, object], str]:
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ArtifactCollectionError(f"JSON: non-finite value {item}")
            ),
        )
    except ArtifactCollectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCollectionError(f"unreadable JSON file: {path}") from exc
    if not isinstance(value, dict):
        _fail(str(path), "expected a top-level object")
    hexdigest = hashlib.sha256(encoded).hexdigest()
    return (
        value,
        make_reference(
            subject_schema=logical_schema,
            subject_id=hexdigest,
            content_sha256=hexdigest,
            size_bytes=len(encoded),
        ),
        hexdigest,
    )


def _tensor_bytes(record: object, *, where: str) -> tuple[int, str, int]:
    row = _mapping(record, where=where)
    if set(row) != {"shape", "dtype"}:
        _fail(where, "expected exactly shape and dtype")
    dtype = row["dtype"]
    if not isinstance(dtype, str) or dtype.upper() not in _DTYPE_BITS:
        _fail(f"{where}.dtype", f"unsupported dtype {dtype!r}")
    shape = row["shape"]
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        _fail(f"{where}.shape", "expected an array")
    elements = 1
    for index, dimension in enumerate(shape):
        dimension = _integer(dimension, where=f"{where}.shape[{index}]")
        elements *= dimension
    bits = _DTYPE_BITS[dtype.upper()]
    total_bits = elements * bits
    if total_bits % 8:
        _fail(where, "tensor bit count is not byte aligned")
    return total_bits // 8, dtype.upper(), len(shape)


def _safe_recorded_files(value: object) -> dict[str, int]:
    mapping = _mapping(value, where="artifact_inventory.file_bytes")
    result: dict[str, int] = {}
    for raw_name, raw_size in mapping.items():
        if not isinstance(raw_name, str) or not raw_name:
            _fail("artifact_inventory.file_bytes", "file name must be nonempty")
        path = Path(raw_name)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            _fail("artifact_inventory.file_bytes", f"unsafe relative path {raw_name!r}")
        normalized = path.as_posix()
        if normalized in result:
            _fail(
                "artifact_inventory.file_bytes",
                f"multiple names normalize to {normalized!r}",
            )
        result[normalized] = _integer(
            raw_size, where=f"artifact_inventory.file_bytes.{raw_name}"
        )
    return dict(sorted(result.items()))


def _observe_regular_files(root: Path) -> dict[str, int]:
    if root.is_symlink() or not root.is_dir():
        _fail("artifact_root", "must be a real directory, not a symlink")
    result: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail("artifact_root", f"contains symlink {path.relative_to(root)}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = int(path.stat().st_size)
    return result


def audit_legacy_export(
    *,
    quant_config_path: str | Path,
    shapes_path: str | Path,
    artifact_root: str | Path | None = None,
    namespace_prefix: str = "mtp.",
) -> dict[str, object]:
    """Build a content-addressed census without hashing the model weights.

    ``shapes_path`` is a trusted safetensors-header projection with one
    ``{shape, dtype}`` object per physical tensor.  The optional artifact-root
    observation is stat-only and is labelled as such; it is not an artifact
    content identity.
    """
    quant_path = Path(quant_config_path)
    shape_path = Path(shapes_path)
    if not isinstance(namespace_prefix, str) or not namespace_prefix:
        _fail("namespace_prefix", "must be a nonempty string")
    quant, quant_ref, quant_sha = _load_json_source(
        quant_path, logical_schema="prismaquant.legacy.quant_config.json"
    )
    shapes, shapes_ref, shapes_sha = _load_json_source(
        shape_path, logical_schema="prismaquant.legacy.safetensors_shapes.json"
    )

    provenance = _mapping(quant.get("provenance"), where="quant_config.provenance")
    formats = _mapping(
        provenance.get("tensor_formats"), where="quant_config.provenance.tensor_formats"
    )
    if not formats:
        _fail("quant_config.provenance.tensor_formats", "must not be empty")
    format_histogram: Counter[str] = Counter()
    for qname, format_name in formats.items():
        if not isinstance(qname, str) or not qname:
            _fail("tensor_formats", "unit names must be nonempty strings")
        if not isinstance(format_name, str) or not format_name:
            _fail(f"tensor_formats.{qname}", "format must be a nonempty string")
        format_histogram[format_name] += 1

    # Historical exporters recorded this as a CB-only scalar count, not a
    # qname collection.  The complete assignment authority is tensor_formats.
    cb_target_count = _integer(
        provenance.get("cb_targets"), where="quant_config.provenance.cb_targets"
    )
    if cb_target_count > len(formats):
        _fail("cb_targets", "CB-only count exceeds the complete format assignment")

    namespace_tensor_bytes = 0
    namespace_matrix_bytes = 0
    namespace_tensor_count = 0
    namespace_matrix_modules: set[str] = set()
    namespace_dtype_histogram: Counter[str] = Counter()
    for tensor_name, tensor_record in shapes.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            _fail("shapes", "tensor names must be nonempty strings")
        size_bytes, dtype, rank = _tensor_bytes(tensor_record, where=f"shapes.{tensor_name}")
        if tensor_name.startswith(namespace_prefix):
            namespace_tensor_count += 1
            namespace_tensor_bytes += size_bytes
            namespace_dtype_histogram[dtype] += 1
            if tensor_name.endswith(".weight") and rank == 2:
                namespace_matrix_modules.add(tensor_name[: -len(".weight")])
                namespace_matrix_bytes += size_bytes
    namespace_support_count = namespace_tensor_count - len(namespace_matrix_modules)
    namespace_support_bytes = namespace_tensor_bytes - namespace_matrix_bytes
    complete_module_names = set(formats) | namespace_matrix_modules

    inventory = _mapping(
        provenance.get("artifact_inventory"),
        where="quant_config.provenance.artifact_inventory",
    )
    serialized = _mapping(
        provenance.get("serialized_payload"),
        where="quant_config.provenance.serialized_payload",
    )
    recorded_files = _safe_recorded_files(inventory.get("file_bytes"))
    recorded_total = _integer(
        inventory.get("export_directory_bytes"),
        where="artifact_inventory.export_directory_bytes",
    )
    if sum(recorded_files.values()) != recorded_total:
        _fail("artifact_inventory", "file_bytes do not sum to export_directory_bytes")

    body_tensor_bytes = _integer(
        inventory.get("cb_tensor_payload_bytes"),
        where="artifact_inventory.cb_tensor_payload_bytes",
    )
    body_sidecar_bytes = _integer(
        inventory.get("cb_codebook_sidecar_bytes"),
        where="artifact_inventory.cb_codebook_sidecar_bytes",
    )
    body_serialized_bytes = _integer(
        inventory.get("cb_serialized_payload_bytes"),
        where="artifact_inventory.cb_serialized_payload_bytes",
    )
    if body_tensor_bytes + body_sidecar_bytes != body_serialized_bytes:
        _fail("artifact_inventory", "CB tensor and sidecar bytes do not sum to serialized bytes")

    serialized_tensor_fields = (
        "index_bytes",
        "fp4_scale_bytes",
        "fp8_row_scale_bytes",
        "global_scale_bytes",
        "input_global_scale_bytes",
    )
    serialized_tensor_bytes = sum(
        _integer(serialized.get(field), where=f"serialized_payload.{field}")
        for field in serialized_tensor_fields
    )
    if serialized_tensor_bytes != body_tensor_bytes:
        _fail("serialized_payload", "component tensor bytes differ from artifact inventory")
    if _integer(
        serialized.get("codebook_sidecar_bytes"),
        where="serialized_payload.codebook_sidecar_bytes",
    ) != body_sidecar_bytes:
        _fail("serialized_payload", "sidecar bytes differ from artifact inventory")
    if _integer(serialized.get("n_tensors"), where="serialized_payload.n_tensors") != cb_target_count:
        _fail("serialized_payload.n_tensors", "does not equal the CB-only target count")

    fixed_residual_recorded = recorded_total - body_serialized_bytes - namespace_tensor_bytes
    if fixed_residual_recorded < 0:
        _fail("byte_ledger", "body plus namespace bytes exceed recorded artifact bytes")

    observed: dict[str, object] | None = None
    drift_bytes: int | None = None
    fixed_residual_observed: int | None = None
    if artifact_root is not None:
        root = Path(artifact_root)
        observed_files = _observe_regular_files(root)
        observed_total = sum(observed_files.values())
        drift_bytes = observed_total - recorded_total
        fixed_residual_observed = observed_total - body_serialized_bytes - namespace_tensor_bytes
        missing = sorted(set(recorded_files) - set(observed_files))
        extra = sorted(set(observed_files) - set(recorded_files))
        mismatched = {
            name: {
                "recorded_bytes": recorded_files[name],
                "observed_bytes": observed_files[name],
            }
            for name in sorted(set(recorded_files) & set(observed_files))
            if recorded_files[name] != observed_files[name]
        }
        observed = {
            "scope": "all_regular_files_recursive",
            "observation_kind": "stat_size_only_unsealed",
            "total_bytes": observed_total,
            "file_count": len(observed_files),
            "file_bytes": observed_files,
            "missing_recorded_files": missing,
            "extra_files": extra,
            "size_mismatches": mismatched,
        }

    payload: dict[str, object] = {
        "sources": {
            "quant_config": quant_ref,
            "safetensors_shapes": shapes_ref,
        },
        "assignment_census": {
            "format_assignment_unit_count": len(formats),
            "cb_assignment_unit_count": cb_target_count,
            "auxiliary_assignment_unit_count": len(formats) - cb_target_count,
            "format_histogram": dict(sorted(format_histogram.items())),
            "namespace_matrix_module_count": len(namespace_matrix_modules),
            "format_or_namespace_matrix_module_count": len(complete_module_names),
        },
        "physical_tensor_census": {
            "physical_model_tensor_count": len(shapes),
            "namespace_prefix": namespace_prefix,
            "namespace_tensor_count": namespace_tensor_count,
            "namespace_tensor_bytes": namespace_tensor_bytes,
            "namespace_matrix_weight_count": len(namespace_matrix_modules),
            "namespace_matrix_weight_bytes": namespace_matrix_bytes,
            "namespace_support_tensor_count": namespace_support_count,
            "namespace_support_tensor_bytes": namespace_support_bytes,
            "namespace_dtype_histogram": dict(sorted(namespace_dtype_histogram.items())),
        },
        "byte_ledger": {
            "body_tensor_bytes": body_tensor_bytes,
            "body_shared_sidecar_bytes": body_sidecar_bytes,
            "body_serialized_bytes": body_serialized_bytes,
            "namespace_tensor_bytes": namespace_tensor_bytes,
            "fixed_residual_recorded_bytes": fixed_residual_recorded,
            "recorded": {
                "scope": inventory.get("scope"),
                "total_bytes": recorded_total,
                "file_count": len(recorded_files),
                "file_bytes": recorded_files,
            },
            "observed": observed,
            "recursive_drift_bytes": drift_bytes,
            "fixed_residual_observed_bytes": fixed_residual_observed,
        },
    }
    locators: dict[str, list[str]] = {}
    for digest, path in (
        (quant_sha, quant_path),
        (shapes_sha, shape_path),
    ):
        locators.setdefault(digest, []).append(str(path.resolve()))
    locators = {key: sorted(set(values)) for key, values in locators.items()}
    return verify_legacy_audit(
        seal_record(LEGACY_AUDIT_SCHEMA, payload, locators=locators)
    )


def validate_legacy_audit_payload(payload: Mapping[str, object]) -> None:
    """Strict payload validator used by the generic record loader."""
    payload = _mapping(payload, where="legacy_audit.payload")
    expected = {"sources", "assignment_census", "physical_tensor_census", "byte_ledger"}
    _exact_keys(payload, expected, where="legacy_audit.payload")

    sources = _mapping(payload["sources"], where="legacy_audit.sources")
    _exact_keys(
        sources,
        {"quant_config", "safetensors_shapes"},
        where="legacy_audit.sources",
    )
    for name, raw_reference in sources.items():
        reference = _mapping(raw_reference, where=f"legacy_audit.sources.{name}")
        _exact_keys(
            reference,
            {"schema", "subject_schema", "subject_id", "content"},
            where=f"legacy_audit.sources.{name}",
        )
        content = _mapping(
            reference["content"], where=f"legacy_audit.sources.{name}.content"
        )
        _exact_keys(
            content,
            {"sha256", "size_bytes"},
            where=f"legacy_audit.sources.{name}.content",
        )
        if reference["subject_id"] != content["sha256"]:
            _fail(
                f"legacy_audit.sources.{name}",
                "raw legacy source semantic and content digests must agree",
            )
        size_bytes = _integer(
            content["size_bytes"],
            where=f"legacy_audit.sources.{name}.content.size_bytes",
        )
        rebuilt = make_reference(
            subject_schema=reference["subject_schema"],
            subject_id=reference["subject_id"],
            content_sha256=content["sha256"],
            size_bytes=size_bytes,
        )
        if reference != rebuilt:
            _fail(f"legacy_audit.sources.{name}", "reference is not canonical")

    assignment = _mapping(
        payload["assignment_census"], where="legacy_audit.assignment_census"
    )
    _exact_keys(
        assignment,
        {
            "format_assignment_unit_count",
            "cb_assignment_unit_count",
            "auxiliary_assignment_unit_count",
            "format_histogram",
            "namespace_matrix_module_count",
            "format_or_namespace_matrix_module_count",
        },
        where="legacy_audit.assignment_census",
    )
    format_count = _integer(
        assignment["format_assignment_unit_count"],
        where="assignment_census.format_assignment_unit_count",
    )
    cb_count = _integer(
        assignment["cb_assignment_unit_count"],
        where="assignment_census.cb_assignment_unit_count",
    )
    auxiliary_count = _integer(
        assignment["auxiliary_assignment_unit_count"],
        where="assignment_census.auxiliary_assignment_unit_count",
    )
    if cb_count + auxiliary_count != format_count:
        _fail("assignment_census", "CB and auxiliary counts do not cover format assignments")
    histogram = _mapping(
        assignment["format_histogram"], where="assignment_census.format_histogram"
    )
    if sum(
        _integer(count, where=f"assignment_census.format_histogram.{name}")
        for name, count in histogram.items()
    ) != format_count:
        _fail("assignment_census.format_histogram", "does not sum to format assignments")
    namespace_matrix_count = _integer(
        assignment["namespace_matrix_module_count"],
        where="assignment_census.namespace_matrix_module_count",
    )
    complete_count = _integer(
        assignment["format_or_namespace_matrix_module_count"],
        where="assignment_census.format_or_namespace_matrix_module_count",
    )
    if complete_count < max(format_count, namespace_matrix_count):
        _fail("assignment_census", "union module count is smaller than an input set")

    physical = _mapping(
        payload["physical_tensor_census"],
        where="legacy_audit.physical_tensor_census",
    )
    _exact_keys(
        physical,
        {
            "physical_model_tensor_count",
            "namespace_prefix",
            "namespace_tensor_count",
            "namespace_tensor_bytes",
            "namespace_matrix_weight_count",
            "namespace_matrix_weight_bytes",
            "namespace_support_tensor_count",
            "namespace_support_tensor_bytes",
            "namespace_dtype_histogram",
        },
        where="legacy_audit.physical_tensor_census",
    )
    if not isinstance(physical["namespace_prefix"], str) or not physical["namespace_prefix"]:
        _fail("physical_tensor_census.namespace_prefix", "expected a nonempty string")
    physical_total_count = _integer(
        physical["physical_model_tensor_count"],
        where="physical_tensor_census.physical_model_tensor_count",
    )
    namespace_count = _integer(
        physical["namespace_tensor_count"],
        where="physical_tensor_census.namespace_tensor_count",
    )
    matrix_count = _integer(
        physical["namespace_matrix_weight_count"],
        where="physical_tensor_census.namespace_matrix_weight_count",
    )
    support_count = _integer(
        physical["namespace_support_tensor_count"],
        where="physical_tensor_census.namespace_support_tensor_count",
    )
    if matrix_count + support_count != namespace_count or namespace_count > physical_total_count:
        _fail("physical_tensor_census", "tensor counts do not reconcile")
    if namespace_matrix_count != matrix_count:
        _fail(
            "assignment_census.namespace_matrix_module_count",
            "differs from the physical namespace matrix census",
        )
    namespace_bytes = _integer(
        physical["namespace_tensor_bytes"],
        where="physical_tensor_census.namespace_tensor_bytes",
    )
    matrix_bytes = _integer(
        physical["namespace_matrix_weight_bytes"],
        where="physical_tensor_census.namespace_matrix_weight_bytes",
    )
    support_bytes = _integer(
        physical["namespace_support_tensor_bytes"],
        where="physical_tensor_census.namespace_support_tensor_bytes",
    )
    if matrix_bytes + support_bytes != namespace_bytes:
        _fail("physical_tensor_census", "namespace bytes do not reconcile")
    dtype_histogram = _mapping(
        physical["namespace_dtype_histogram"],
        where="physical_tensor_census.namespace_dtype_histogram",
    )
    if sum(
        _integer(count, where=f"physical_tensor_census.namespace_dtype_histogram.{name}")
        for name, count in dtype_histogram.items()
    ) != namespace_count:
        _fail("physical_tensor_census.namespace_dtype_histogram", "does not sum to namespace count")

    ledger = _mapping(payload["byte_ledger"], where="legacy_audit.byte_ledger")
    _exact_keys(
        ledger,
        {
            "body_tensor_bytes",
            "body_shared_sidecar_bytes",
            "body_serialized_bytes",
            "namespace_tensor_bytes",
            "fixed_residual_recorded_bytes",
            "recorded",
            "observed",
            "recursive_drift_bytes",
            "fixed_residual_observed_bytes",
        },
        where="legacy_audit.byte_ledger",
    )
    body_tensor = _integer(
        ledger["body_tensor_bytes"], where="byte_ledger.body_tensor_bytes"
    )
    body_sidecar = _integer(
        ledger["body_shared_sidecar_bytes"],
        where="byte_ledger.body_shared_sidecar_bytes",
    )
    body = _integer(ledger.get("body_serialized_bytes"), where="byte_ledger.body_serialized_bytes")
    if body_tensor + body_sidecar != body:
        _fail("byte_ledger", "body tensor and sidecar bytes do not sum to serialized body")
    namespace = _integer(ledger.get("namespace_tensor_bytes"), where="byte_ledger.namespace_tensor_bytes")
    if namespace != namespace_bytes:
        _fail("byte_ledger.namespace_tensor_bytes", "differs from physical tensor census")
    residual = _integer(
        ledger.get("fixed_residual_recorded_bytes"),
        where="byte_ledger.fixed_residual_recorded_bytes",
    )
    recorded = _mapping(ledger.get("recorded"), where="byte_ledger.recorded")
    _exact_keys(
        recorded,
        {"scope", "total_bytes", "file_count", "file_bytes"},
        where="byte_ledger.recorded",
    )
    if recorded["scope"] != "all_regular_files_recursive":
        _fail("byte_ledger.recorded.scope", "unsupported recorded byte scope")
    recorded_files = _safe_recorded_files(recorded["file_bytes"])
    if _integer(recorded["file_count"], where="byte_ledger.recorded.file_count") != len(recorded_files):
        _fail("byte_ledger.recorded.file_count", "does not equal file ledger length")
    total = _integer(recorded.get("total_bytes"), where="byte_ledger.recorded.total_bytes")
    if sum(recorded_files.values()) != total:
        _fail("byte_ledger.recorded", "file ledger does not sum to total bytes")
    if body + namespace + residual != total:
        _fail("byte_ledger", "recorded body, namespace, and residual do not balance")

    observed = ledger["observed"]
    if observed is None:
        if ledger["recursive_drift_bytes"] is not None or ledger["fixed_residual_observed_bytes"] is not None:
            _fail("byte_ledger", "absent observation must not synthesize drift or residual")
    else:
        observed = _mapping(observed, where="byte_ledger.observed")
        _exact_keys(
            observed,
            {
                "scope",
                "observation_kind",
                "total_bytes",
                "file_count",
                "file_bytes",
                "missing_recorded_files",
                "extra_files",
                "size_mismatches",
            },
            where="byte_ledger.observed",
        )
        if observed["scope"] != "all_regular_files_recursive":
            _fail("byte_ledger.observed.scope", "unsupported observed byte scope")
        if observed["observation_kind"] != "stat_size_only_unsealed":
            _fail("byte_ledger.observed.observation_kind", "unsupported observation kind")
        observed_files = _safe_recorded_files(observed["file_bytes"])
        observed_total = _integer(
            observed["total_bytes"], where="byte_ledger.observed.total_bytes"
        )
        if sum(observed_files.values()) != observed_total:
            _fail("byte_ledger.observed", "file ledger does not sum to total bytes")
        if _integer(observed["file_count"], where="byte_ledger.observed.file_count") != len(observed_files):
            _fail("byte_ledger.observed.file_count", "does not equal file ledger length")
        expected_missing = sorted(set(recorded_files) - set(observed_files))
        expected_extra = sorted(set(observed_files) - set(recorded_files))
        if observed["missing_recorded_files"] != expected_missing:
            _fail("byte_ledger.observed.missing_recorded_files", "does not match file ledgers")
        if observed["extra_files"] != expected_extra:
            _fail("byte_ledger.observed.extra_files", "does not match file ledgers")
        expected_mismatches = {
            name: {
                "recorded_bytes": recorded_files[name],
                "observed_bytes": observed_files[name],
            }
            for name in sorted(set(recorded_files) & set(observed_files))
            if recorded_files[name] != observed_files[name]
        }
        if observed["size_mismatches"] != expected_mismatches:
            _fail("byte_ledger.observed.size_mismatches", "does not match file ledgers")
        drift = _integer(
            ledger["recursive_drift_bytes"],
            where="byte_ledger.recursive_drift_bytes",
            nonnegative=False,
        )
        if drift != observed_total - total:
            _fail("byte_ledger.recursive_drift_bytes", "does not match observed minus recorded")
        observed_residual = _integer(
            ledger["fixed_residual_observed_bytes"],
            where="byte_ledger.fixed_residual_observed_bytes",
        )
        if observed_residual != observed_total - body - namespace:
            _fail("byte_ledger.fixed_residual_observed_bytes", "does not balance observed bytes")
    return None


def verify_legacy_audit(record: Mapping[str, object]) -> dict[str, object]:
    verified = verify_record(record)
    if verified["schema"] != LEGACY_AUDIT_SCHEMA:
        _fail("record.schema", f"expected {LEGACY_AUDIT_SCHEMA!r}")
    return verified
