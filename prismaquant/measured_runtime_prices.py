"""Strict, opt-in measured operator prices for discrete allocation proposals.

Extends the dispatch-table input boundary, not its served-SLO evidence. Prices
are medians of repeated GPU operator timings for a whole serving unit. They
are never inferred from activation width, encoder time, or family speed hints.
The sequential operator-sum model can propose assignments under a declared
budget; it cannot certify end-to-end p95 TTFT/ITL. Producer admission and final
served validation remain independent gates. No measured table ships here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .lane_eligibility import ServingContext
from .serve_dispatch_table import DispatchTableError

SCHEMA = "prismaquant.measured_runtime_prices.v1"
CONTEXT_SCHEMA = "prismaquant.measured_runtime_context.v1"
PROVENANCE_CONTEXT_SCHEMA = "prismaquant.measured_runtime_context.v2"
PROVENANCE_TABLE_SCHEMA = "prismaquant.measured_runtime_prices.v2"
PROVENANCE_IDENTITY_KIND = "prismaquant.runtime_provenance_relation.v1"
RESOURCE_FIELDS = ("prefill_ms", "decode_ms", "serialized_bytes", "resident_bytes",
                   "peak_scratch_bytes", "activation_bytes", "kv_bytes")


class RuntimePriceError(DispatchTableError):
    """Missing, malformed, stale, or mismatched measured proposal evidence."""


def _object(value: Any, fields: tuple[str, ...], where: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise RuntimePriceError(f"{where}: expected exactly fields {sorted(fields)}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RuntimePriceError(f"{where}: expected a nonempty trimmed string")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimePriceError(f"{where}: expected lowercase SHA-256")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimePriceError(f"{where}: expected integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise RuntimePriceError(f"{where}: expected finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _timestamp(value: Any, where: str) -> datetime:
    try:
        result = datetime.fromisoformat(_string(value, where).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimePriceError(f"{where}: invalid ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise RuntimePriceError(f"{where}: timestamp must explicitly use UTC")
    return result


def _json(path: str | Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimePriceError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, ValueError) as exc:
        raise RuntimePriceError(f"cannot load {path}: {exc}") from exc


def identity_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RuntimeContext:
    """Workload/source identity extending the existing lane ServingContext.

    ``runtime_sha256`` is the digest of the full measured runtime manifest:
    packages/plugins/kernel builds, launch arguments, scheduler/chunking,
    graph capture settings, and parallel topology. An image digest alone is
    insufficient because these can change inside one image. ``gpu_identity``
    identifies the actual device/configuration used by that manifest. The
    explicit workload fields are additional checked coordinates, not a licence
    to omit unlisted runtime settings from the manifest identity. Version 2
    instead names a provenance-relation identity that retains every original
    runtime manifest and explicitly verifies their common coordinates.
    """

    serving_context: ServingContext
    gpu_identity: str
    runtime_sha256: str
    source_sha256: str
    calibration_sha256: str
    prompt_tokens: int
    batch_size: int
    tensor_parallel: int
    graph_mode: str
    operator_routes: Mapping[str, Mapping[str, str]]
    runtime_identity_kind: str | None = None

    def __post_init__(self):
        if not isinstance(self.serving_context, ServingContext):
            raise RuntimePriceError("serving_context must be a ServingContext")
        _string(self.gpu_identity, "gpu_identity")
        if self.runtime_identity_kind not in (None, PROVENANCE_IDENTITY_KIND):
            raise RuntimePriceError("unknown measured runtime identity kind")
        for name in ("runtime_sha256", "source_sha256", "calibration_sha256"):
            _sha(getattr(self, name), name)
        for name in ("prompt_tokens", "batch_size", "tensor_parallel"):
            _integer(getattr(self, name), name, 1)
        _string(self.graph_mode, "graph_mode")
        if not isinstance(self.operator_routes, Mapping) or not self.operator_routes:
            raise RuntimePriceError("operator_routes must be a nonempty unit/format/route mapping")
        routes = {}
        for unit, formats in sorted(self.operator_routes.items()):
            _string(unit, "operator_routes unit")
            if not isinstance(formats, Mapping) or not formats:
                raise RuntimePriceError("operator_routes unit must have format routes")
            routes[unit] = MappingProxyType({_string(fmt, "route format"): _string(route, "operator route")
                                            for fmt, route in sorted(formats.items())})
        object.__setattr__(self, "operator_routes", MappingProxyType(routes))

    def operator_route(self, unit: str, fmt: str) -> str:
        try:
            return self.operator_routes[unit][fmt]
        except KeyError as exc:
            raise RuntimePriceError(f"missing expected operator route for {(unit, fmt)}") from exc

    def as_dict(self) -> dict:
        return {"schema": PROVENANCE_CONTEXT_SCHEMA if self.runtime_identity_kind else CONTEXT_SCHEMA,
                **({"runtime_identity_kind": self.runtime_identity_kind} if self.runtime_identity_kind else {}),
                "serving_context": self.serving_context.as_dict(),
                **{name: getattr(self, name) for name in (
                    "gpu_identity", "runtime_sha256", "source_sha256", "calibration_sha256",
                    "prompt_tokens", "batch_size", "tensor_parallel", "graph_mode")},
                "operator_routes": {unit: dict(formats) for unit, formats in self.operator_routes.items()}}


def parse_runtime_context(payload: Mapping) -> RuntimeContext:
    fields = ("schema", "serving_context", "gpu_identity", "runtime_sha256", "source_sha256",
              "calibration_sha256", "prompt_tokens", "batch_size", "tensor_parallel", "graph_mode", "operator_routes")
    if not isinstance(payload, Mapping):
        raise RuntimePriceError("runtime context: expected an object")
    if payload.get("schema") == PROVENANCE_CONTEXT_SCHEMA:
        fields += ("runtime_identity_kind",)
        if payload.get("runtime_identity_kind") != PROVENANCE_IDENTITY_KIND:
            raise RuntimePriceError("v2 runtime context requires an explicit provenance relation identity")
    _object(payload, fields, "runtime context")
    if payload["schema"] not in (CONTEXT_SCHEMA, PROVENANCE_CONTEXT_SCHEMA):
        raise RuntimePriceError(f"runtime context schema must be {CONTEXT_SCHEMA}")
    serving = _object(payload["serving_context"], ("platform", "structure", "residency", "runtime_image", "execution_mode"), "serving_context")
    try:
        return RuntimeContext(serving_context=ServingContext(**serving),
                              **{field: payload[field] for field in fields if field not in ("schema", "serving_context")})
    except (ValueError, TypeError) as exc:
        raise RuntimePriceError(f"runtime context: {exc}") from exc


def load_runtime_context(path: str | Path) -> RuntimeContext:
    return parse_runtime_context(_json(path))


@dataclass(frozen=True)
class RuntimeResources:
    """Separate wire bytes, resident terminal weights, and transient allocations.

    Timings/permanently resident weights add across sequential serving units. Scratch and
    activation use independent maxima; global KV and fixed overhead are added
    once. This conservative composition assumes no overlapping unit execution.
    """

    prefill_ms: float
    decode_ms: float | None
    serialized_bytes: int
    resident_bytes: int
    peak_scratch_bytes: int
    activation_bytes: int
    kv_bytes: int = 0

    def __post_init__(self):
        _number(self.prefill_ms, "prefill_ms")
        if self.decode_ms is not None:
            _number(self.decode_ms, "decode_ms")
        for name in RESOURCE_FIELDS[2:]:
            _integer(getattr(self, name), name)

    def as_dict(self) -> dict:
        return {field: getattr(self, field) for field in RESOURCE_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping) -> RuntimeResources:
        return cls(**_object(payload, RESOURCE_FIELDS, "resources"))


@dataclass(frozen=True)
class RuntimeBinding:
    """Bind whole-unit timings to exact candidate members and joint operators.

    Joint operator identity covers source/rendered tensor content, full
    activation/scales contract, arithmetic, and aligned calibration probes.
    A fused group's measured row must identify every member; leaf medians
    cannot be silently summed into a fused operator measurement.
    """

    member_formats: Mapping[str, str]
    member_operator_identity_sha256: Mapping[str, str]
    member_shapes: Mapping[str, tuple[int, ...]]
    operator_route: str

    def __post_init__(self):
        _string(self.operator_route, "operator_route")
        for name in ("member_formats", "member_operator_identity_sha256", "member_shapes"):
            if not isinstance(getattr(self, name), Mapping) or not getattr(self, name):
                raise RuntimePriceError(f"{name} must be a nonempty mapping")
        if not set(self.member_formats) == set(self.member_operator_identity_sha256) == set(self.member_shapes):
            raise RuntimePriceError("binding member maps must have identical keys")
        formats, digests, shapes = {}, {}, {}
        for unit in sorted(self.member_formats):
            _string(unit, "member unit")
            formats[unit] = _string(self.member_formats[unit], "member format")
            digests[unit] = _sha(self.member_operator_identity_sha256[unit], "joint operator identity")
            shape = self.member_shapes[unit]
            if not isinstance(shape, (tuple, list)) or not shape:
                raise RuntimePriceError("member shape must be a nonempty dimension sequence")
            shapes[unit] = tuple(_integer(dim, "shape dimension", 1) for dim in shape)
        object.__setattr__(self, "member_formats", MappingProxyType(formats))
        object.__setattr__(self, "member_operator_identity_sha256", MappingProxyType(digests))
        object.__setattr__(self, "member_shapes", MappingProxyType(shapes))

    def as_dict(self) -> dict:
        return {"member_formats": dict(self.member_formats),
                "member_operator_identity_sha256": dict(self.member_operator_identity_sha256),
                "member_shapes": {unit: list(shape) for unit, shape in self.member_shapes.items()},
                "operator_route": self.operator_route}

    @classmethod
    def from_dict(cls, payload: Mapping) -> RuntimeBinding:
        return cls(**_object(payload, ("member_formats", "member_operator_identity_sha256", "member_shapes", "operator_route"), "binding"))


@dataclass(frozen=True)
class OperatorMeasurement:
    """Raw repeated per-invocation GPU durations with attributable evidence."""

    method: str
    samples_ms: tuple[float, ...]
    warmup_iterations: int
    receipt_path: str
    receipt_sha256: str

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @classmethod
    def from_dict(cls, payload: Mapping) -> OperatorMeasurement:
        _object(payload, ("method", "samples_ms", "warmup_iterations", "receipt_path", "receipt_sha256"), "operator measurement")
        if payload["method"] not in ("cuda_events", "gpu_profiler", "synchronized_gpu_wall_clock"):
            raise RuntimePriceError("measurement method must time actual GPU operator execution")
        samples = payload["samples_ms"]
        if not isinstance(samples, (tuple, list)) or len(samples) < 3:
            raise RuntimePriceError("operator measurement requires at least three repeated samples")
        return cls(payload["method"], tuple(_number(v, "samples_ms", positive=True) for v in samples),
                   _integer(payload["warmup_iterations"], "warmup_iterations", 1),
                   _string(payload["receipt_path"], "receipt_path"), _sha(payload["receipt_sha256"], "receipt_sha256"))

    def as_dict(self) -> dict:
        return {"method": self.method, "samples_ms": list(self.samples_ms),
                "warmup_iterations": self.warmup_iterations, "receipt_path": self.receipt_path,
                "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class MeasuredRuntimeRow:
    unit: str
    fmt: str
    binding: RuntimeBinding
    resources: RuntimeResources
    prefill: OperatorMeasurement
    decode: OperatorMeasurement | None

    @property
    def key(self) -> tuple[str, str]:
        return self.unit, self.fmt

    def as_dict(self) -> dict:
        return {"unit": self.unit, "format": self.fmt, "binding": self.binding.as_dict(),
                "resources": self.resources.as_dict(), "prefill": self.prefill.as_dict(),
                "decode": self.decode.as_dict() if self.decode else None}


@dataclass(frozen=True)
class MeasuredRuntimeTable:
    table_id: str
    context: RuntimeContext
    cost_sha256: str
    measured_at: str
    valid_until: str
    fixed_assignment: Mapping[str, str]
    fixed_resources: RuntimeResources
    fixed_resources_receipt_path: str
    fixed_resources_receipt_sha256: str
    rows: tuple[MeasuredRuntimeRow, ...]
    source_path: str = ""
    runtime_provenance: Mapping | None = None
    native_receipt_bindings: tuple[Mapping, ...] = ()
    producer_admitted: bool = False

    def as_dict(self) -> dict:
        return {"schema": PROVENANCE_TABLE_SCHEMA if self.runtime_provenance is not None else SCHEMA,
                **({"runtime_provenance": dict(self.runtime_provenance),
                    "native_receipt_bindings": [{key: dict(value) if isinstance(value, Mapping) else value
                                                  for key, value in binding.items()}
                                                 for binding in self.native_receipt_bindings]}
                   if self.runtime_provenance is not None else {}),
                "table_id": self.table_id, "status": "proposal_data",
                "composition": "sequential_operator_sum", "context": self.context.as_dict(),
                "cost_sha256": self.cost_sha256, "measured_at": self.measured_at,
                "valid_until": self.valid_until, "fixed_assignment": dict(self.fixed_assignment),
                "fixed_resources": self.fixed_resources.as_dict(),
                "fixed_resources_receipt_path": self.fixed_resources_receipt_path,
                "fixed_resources_receipt_sha256": self.fixed_resources_receipt_sha256,
                "rows": [row.as_dict() for row in self.rows]}

    def identity(self) -> dict:
        return {"schema": self.as_dict()["schema"], "table_id": self.table_id, "sha256": identity_sha256(self.as_dict()),
                "cost_sha256": self.cost_sha256, "context": self.context.as_dict(),
                "source_path": self.source_path, "status": "proposal_data", "slo_eligible": False,
                "composition": "sequential_operator_sum", "measured_at": self.measured_at,
                "valid_until": self.valid_until, "n_rows": len(self.rows)}


def parse_measured_runtime_table(payload: Mapping, *, expected_context: RuntimeContext,
                                 expected_cost_sha256: str, now: datetime | None = None,
                                 source_path: str = "") -> MeasuredRuntimeTable:
    """Validate an explicit table; caller supplies independent workload/cost identity.

    Parsing validates the evidence declarations. Loading additionally verifies
    local raw receipt content hashes. Version 2 additionally requires producer
    admission; parsing alone cannot supply allocation resources.
    """
    fields = ("schema", "table_id", "status", "composition", "context", "cost_sha256",
                      "measured_at", "valid_until", "fixed_assignment", "fixed_resources",
                      "fixed_resources_receipt_path", "fixed_resources_receipt_sha256", "rows")
    if not isinstance(payload, Mapping):
        raise RuntimePriceError("runtime table: expected an object")
    is_provenance = payload.get("schema") == PROVENANCE_TABLE_SCHEMA
    if is_provenance:
        fields += ("runtime_provenance", "native_receipt_bindings")
    _object(payload, fields, "runtime table")
    if payload["schema"] not in (SCHEMA, PROVENANCE_TABLE_SCHEMA) or payload["status"] != "proposal_data":
        raise RuntimePriceError("runtime table requires current schema and proposal_data status")
    if payload["composition"] != "sequential_operator_sum":
        raise RuntimePriceError("only explicit sequential_operator_sum composition is supported")
    context = parse_runtime_context(payload["context"])
    if is_provenance != (context.runtime_identity_kind is not None):
        raise RuntimePriceError("runtime table/context provenance version mismatch")
    if context != expected_context:
        raise RuntimePriceError("runtime context mismatch against independently supplied expected context")
    cost_sha256 = _sha(payload["cost_sha256"], "cost_sha256")
    if cost_sha256 != _sha(expected_cost_sha256, "expected_cost_sha256"):
        raise RuntimePriceError("stale runtime table: cost payload SHA-256 mismatch")
    measured = _timestamp(payload["measured_at"], "measured_at")
    expires = _timestamp(payload["valid_until"], "valid_until")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise RuntimePriceError("now must have a timezone")
    if not measured <= current < expires:
        raise RuntimePriceError("stale or future runtime table measurement window")
    fixed = payload["fixed_assignment"]
    if not isinstance(fixed, Mapping):
        raise RuntimePriceError("fixed_assignment must be an explicit mapping")
    fixed = MappingProxyType({_string(k, "fixed unit"): _string(v, "fixed format") for k, v in sorted(fixed.items())})
    if not isinstance(payload["rows"], list) or not payload["rows"]:
        raise RuntimePriceError("rows must be a nonempty list")
    rows = []
    seen = set()
    for raw in payload["rows"]:
        _object(raw, ("unit", "format", "binding", "resources", "prefill", "decode"), "runtime row")
        unit, fmt = _string(raw["unit"], "unit"), _string(raw["format"], "format")
        key = unit, fmt
        if key in seen:
            raise RuntimePriceError(f"duplicate runtime row {key}")
        seen.add(key)
        binding = RuntimeBinding.from_dict(raw["binding"])
        if binding.operator_route != context.operator_route(unit, fmt):
            raise RuntimePriceError(f"operator route mismatch for {key}")
        resources = RuntimeResources.from_dict(raw["resources"])
        prefill = OperatorMeasurement.from_dict(raw["prefill"])
        decode = OperatorMeasurement.from_dict(raw["decode"]) if raw["decode"] is not None else None
        if resources.prefill_ms != prefill.median_ms or resources.decode_ms != (decode.median_ms if decode else None):
            raise RuntimePriceError(f"{key}: resource times must equal medians of measured operator samples")
        if resources.kv_bytes:
            raise RuntimePriceError("KV belongs to fixed_resources, not per-unit rows")
        rows.append(MeasuredRuntimeRow(unit, fmt, binding, resources, prefill, decode))
    provenance, receipt_bindings = None, ()
    if is_provenance:
        reference = _object(payload["runtime_provenance"], ("path", "sha256"), "runtime provenance artifact")
        provenance = MappingProxyType({"path": _string(reference["path"], "runtime provenance path"),
                                       "sha256": _sha(reference["sha256"], "runtime provenance digest")})
        if not isinstance(payload["native_receipt_bindings"], list):
            raise RuntimePriceError("native receipt bindings must be a list")
        frozen = []
        for item in payload["native_receipt_bindings"]:
            _object(item, ("unit", "format", "run_id", "panel", "receipt", "memory_trace"), "native receipt binding")
            binding = {key: _string(item[key], "native receipt " + key) for key in ("unit", "format", "run_id")}
            for key in ("panel", "receipt", "memory_trace"):
                ref = _object(item[key], ("path", "sha256"), "native " + key + " artifact")
                binding[key] = MappingProxyType({"path": _string(ref["path"], key + " path"),
                                                 "sha256": _sha(ref["sha256"], key + " digest")})
            frozen.append(MappingProxyType(binding))
        receipt_bindings = tuple(frozen)
    return MeasuredRuntimeTable(_string(payload["table_id"], "table_id"), context, cost_sha256,
                                payload["measured_at"], payload["valid_until"], fixed,
                                RuntimeResources.from_dict(payload["fixed_resources"]),
                                _string(payload["fixed_resources_receipt_path"], "fixed_resources_receipt_path"),
                                _sha(payload["fixed_resources_receipt_sha256"], "fixed_resources_receipt_sha256"),
                                tuple(sorted(rows, key=lambda row: row.key)), source_path,
                                provenance, receipt_bindings)


def load_measured_runtime_table(path: str | Path, *, expected_context: RuntimeContext,
                                expected_cost_sha256: str, now: datetime | None = None) -> MeasuredRuntimeTable:
    table = parse_measured_runtime_table(_json(path), expected_context=expected_context,
                                        expected_cost_sha256=expected_cost_sha256, now=now, source_path=str(path))
    receipts = {(table.fixed_resources_receipt_path, table.fixed_resources_receipt_sha256)}
    for row in table.rows:
        for measurement in (row.prefill, row.decode):
            if measurement:
                receipts.add((measurement.receipt_path, measurement.receipt_sha256))
    for receipt, expected in sorted(receipts):
        receipt_path = Path(receipt)
        if not receipt_path.is_absolute():
            receipt_path = Path(path).parent / receipt_path
        try:
            with receipt_path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise RuntimePriceError(f"cannot read measurement receipt {receipt_path}: {exc}") from exc
        if actual != expected:
            raise RuntimePriceError(f"measurement receipt SHA-256 mismatch: {receipt_path}")
    if table.runtime_provenance is not None:
        from .runtime_provenance import admit_runtime_provenance
        admit_runtime_provenance(table)
        table = replace(table, producer_admitted=True)
    return table


def build_runtime_resources(table: MeasuredRuntimeTable, candidates: Mapping[str, list], *,
                            expected_bindings: Mapping[tuple[str, str], RuntimeBinding]) -> dict[tuple[str, str], RuntimeResources]:
    """Price every candidate exactly; no family fallback or unmeasured group sums."""
    if table.runtime_provenance is not None and not table.producer_admitted:
        raise RuntimePriceError("v2 runtime prices require producer admission through the loader")
    rows = {row.key: row for row in table.rows}
    result = {}
    for unit, options in sorted(candidates.items()):
        for candidate in sorted(options, key=lambda c: c.fmt):
            key = unit, candidate.fmt
            if key in result:
                raise RuntimePriceError(f"duplicate candidate {key}")
            if key not in rows or key not in expected_bindings:
                raise RuntimePriceError(f"missing measured runtime row or independent binding for {key}")
            row = rows[key]
            binding = expected_bindings[key]
            if not isinstance(binding, RuntimeBinding) or binding != row.binding:
                raise RuntimePriceError(f"runtime operator binding mismatch for {key}")
            members = candidate.member_formats if getattr(candidate, "member_formats", None) is not None else {unit: candidate.fmt}
            if dict(binding.member_formats) != members:
                raise RuntimePriceError(f"whole serving-unit member formats mismatch for {key}")
            if type(candidate.memory_bytes) is not int or row.resources.serialized_bytes != candidate.memory_bytes:
                raise RuntimePriceError(f"serialized byte mismatch for {key}")
            result[key] = row.resources
    return result
