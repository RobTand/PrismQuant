"""Platform-agnostic anchored cost measurement and extrapolation.

The core knows no quantization format names.  A platform mapping plugin owns
the candidate ladder, the shape-transfer equivalence partition, the exact
production renderer hook, the anchor-rung policy, and extra provenance.  The
core enforces the resulting partition mechanically: neither fitting nor
application can cross a plugin-declared equivalence boundary.

AURA ``predicted_dloss`` is the sole allocation currency.  It already carries
input activation weighting, downstream sensitivity, and the KL Fisher.  Shape
extrapolation therefore multiplies a measured production-anchor level by one
within-segment ratio and never applies another sensitivity or sibling cost.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
from typing import Literal, Protocol, runtime_checkable

from prismaquant.allocator_candidates import (
    ANCHORED_AURA_COST_CURRENCY,
    ANCHORED_AURA_COST_SOURCE,
    SOURCE_PASSTHROUGH_COST_SOURCE,
)
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
    prepare_journal,
    write_unit,
)


# Re-exported, not redefined: the allocator's admission branch keys off these
# exact strings, and two spellings of one contract is how a near-miss silently
# routes an anchored row down the generic weight-only path.
AURA_CURRENCY = ANCHORED_AURA_COST_CURRENCY
PRODUCTION_RENDER_SOURCE = ANCHORED_AURA_COST_SOURCE
ANCHOR_SEGMENT_FIELDS = ("family", "role", "equivalence_class")
_FORBIDDEN_COST_FIELDS = frozenset({
    "h_trace",
    "cw_m2",
    "imatrix_dispersion",
    "activation_mse",
    "output_mse",
    "weight_mse",
})
_ALLOCATOR_IDENTITY_SCHEMA = (
    "prismaquant.anchored_allocator_invocation_identity.v1"
)
_ALLOCATOR_RECEIPT_SCHEMA = "prismaquant.anchored_allocator_invocation.v2"
_ALLOCATOR_IDENTITY_FILE = "anchored_allocator_identity.json"
_ALLOCATOR_RECEIPT_FILE = "anchored_allocator_invocation.json"
_ALLOCATOR_OUTPUT_FILES = ("layer_config.json", "selection.json")


class AnchoredCostError(RuntimeError):
    """A fail-closed plugin, fit, render, or pricing contract violation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _allocator_invocation_identity(
    *,
    command: Sequence[str],
    environment_updates: Mapping[str, str] | None,
    invocation_provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    return dict(canonical_json({
        "schema": _ALLOCATOR_IDENTITY_SCHEMA,
        "command": [str(value) for value in command],
        "environment_updates": {
            str(name): str(value)
            for name, value in (environment_updates or {}).items()
        },
        "invocation_provenance": dict(invocation_provenance or {}),
    }, where="anchored allocator invocation identity"))


def _write_allocator_identity(
    output: Path,
    identity: Mapping[str, object],
    identity_sha256: str,
) -> None:
    marker = {
        "schema": _ALLOCATOR_IDENTITY_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": dict(identity),
    }
    atomic_write_bytes(
        output / _ALLOCATOR_IDENTITY_FILE,
        json.dumps(
            marker, indent=2, sort_keys=True, allow_nan=False,
        ).encode(),
    )


def _load_allocator_identity(
    output: Path,
    *,
    expected_identity: Mapping[str, object],
    expected_sha256: str,
) -> None:
    marker_path = output / _ALLOCATOR_IDENTITY_FILE
    try:
        marker = json.loads(marker_path.read_text())
    except Exception as exc:
        raise AnchoredCostError(
            f"allocator resume identity at {marker_path} is unreadable; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(marker, Mapping):
        raise AnchoredCostError(
            "allocator resume identity is not an object; refusing reuse or "
            "recompute"
        )
    stored_identity = marker.get("identity")
    try:
        stored_sha256 = canonical_json_sha256(
            stored_identity, where="stored allocator invocation identity",
        )
    except (TypeError, ValueError) as exc:
        raise AnchoredCostError(
            "allocator resume identity is corrupt; refusing reuse or "
            "recompute"
        ) from exc
    if (
        marker.get("schema") != _ALLOCATOR_IDENTITY_SCHEMA
        or marker.get("identity_sha256") != stored_sha256
    ):
        raise AnchoredCostError(
            "allocator resume identity checksum differs; refusing reuse or "
            "recompute"
        )
    if stored_sha256 != expected_sha256 or stored_identity != expected_identity:
        raise AnchoredCostError(
            "allocator invocation identity mismatch: "
            f"stored={stored_sha256} current={expected_sha256}; refusing "
            "reuse or recompute"
        )


def _allocator_output_descriptors(output: Path) -> dict[str, object]:
    descriptors: dict[str, object] = {}
    for name in _ALLOCATOR_OUTPUT_FILES:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise AnchoredCostError(
                f"allocator returned success without {path}"
            )
        descriptors[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return descriptors


def _validate_completed_allocator_output(
    output: Path, *, identity_sha256: str,
) -> bool:
    receipt_path = output / _ALLOCATOR_RECEIPT_FILE
    if not receipt_path.exists():
        return False
    if not receipt_path.is_file():
        raise AnchoredCostError(
            f"allocator completion receipt is not a file: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except Exception as exc:
        raise AnchoredCostError(
            "allocator completion receipt is unreadable; refusing reuse or "
            "recompute"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise AnchoredCostError(
            "allocator completion receipt is not an object; refusing reuse "
            "or recompute"
        )
    stored_checksum = receipt.get("receipt_sha256")
    checksum_body = dict(receipt)
    checksum_body.pop("receipt_sha256", None)
    expected_checksum = canonical_json_sha256(
        checksum_body, where="allocator completion receipt",
    )
    if stored_checksum != expected_checksum:
        raise AnchoredCostError(
            "allocator completion receipt checksum differs; refusing reuse "
            "or recompute"
        )
    if (
        receipt.get("schema") != _ALLOCATOR_RECEIPT_SCHEMA
        or receipt.get("identity_sha256") != identity_sha256
    ):
        raise AnchoredCostError(
            "allocator completion receipt identity differs; refusing reuse "
            "or recompute"
        )
    try:
        receipt_identity_sha256 = canonical_json_sha256(
            receipt.get("identity"),
            where="allocator receipt invocation identity",
        )
    except (TypeError, ValueError) as exc:
        raise AnchoredCostError(
            "allocator completion receipt identity is corrupt; refusing "
            "reuse or recompute"
        ) from exc
    if receipt_identity_sha256 != identity_sha256:
        raise AnchoredCostError(
            "allocator completion receipt invocation identity differs; "
            "refusing reuse or recompute"
        )
    stored_outputs = receipt.get("outputs")
    if not isinstance(stored_outputs, Mapping):
        raise AnchoredCostError(
            "allocator completion receipt lacks output checksums; refusing "
            "reuse or recompute"
        )
    actual_outputs = _allocator_output_descriptors(output)
    if stored_outputs != actual_outputs:
        raise AnchoredCostError(
            "allocator completed output checksum differs; refusing reuse or "
            "recompute"
        )
    return True


def _relocate_incomplete_directory(
    path: Path, *, identity_sha256: str | None,
) -> Path:
    """Preserve an incomplete external-stage directory before retrying."""
    label = identity_sha256[:12] if identity_sha256 else "unbound"
    base = path.with_name(f"{path.name}.incomplete-{label}")
    relocated = base
    suffix = 0
    while relocated.exists() or relocated.is_symlink():
        suffix += 1
        relocated = base.with_name(f"{base.name}-{suffix}")
    path.rename(relocated)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return relocated


@dataclass(frozen=True, order=True)
class SegmentKey:
    """One plugin-declared shape-transfer equivalence segment."""

    family: str
    role: str
    equivalence_class: str

    def __post_init__(self) -> None:
        for name in ANCHOR_SEGMENT_FIELDS:
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"segment {name} must be nonempty")
            object.__setattr__(self, name, value)

    @property
    def stamp(self) -> str:
        return "|".join(getattr(self, name) for name in ANCHOR_SEGMENT_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ANCHOR_SEGMENT_FIELDS}


@dataclass(frozen=True)
class CandidateSpec:
    """One already source-gated platform candidate.

    ``bits`` is the plugin's ladder rate declaration. ``payload_bytes`` is
    the exact allocator-owned serialized rate for this concrete unit; the core
    never reimplements source legality or byte accounting. ``shape_features``
    is the plugin-declared design basis used to fit log shape.  A plugin may
    declare multiple numeric features without teaching the core their
    platform-specific meaning.
    """

    format_name: str
    bits: float
    payload_bytes: int
    family: str
    equivalence_class: str
    shape_features: tuple[float, ...]
    coordinate: float
    terminal: bool = False
    allocator_selectable: bool = True

    def __post_init__(self) -> None:
        for name in ("format_name", "family", "equivalence_class"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"candidate {name} must be nonempty")
            object.__setattr__(self, name, value)
        bits = float(self.bits)
        coordinate = float(self.coordinate)
        if not math.isfinite(bits) or bits <= 0.0:
            raise ValueError("candidate bits must be finite and positive")
        if not math.isfinite(coordinate):
            raise ValueError("candidate coordinate must be finite")
        payload_bytes = int(self.payload_bytes)
        if payload_bytes <= 0:
            raise ValueError("candidate payload_bytes must be positive")
        features = tuple(float(value) for value in self.shape_features)
        if not self.terminal and not features:
            raise ValueError("nonterminal candidate needs shape features")
        if any(not math.isfinite(value) for value in features):
            raise ValueError("candidate shape features must be finite")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(self, "shape_features", features)
        if not isinstance(self.allocator_selectable, bool):
            raise ValueError("candidate allocator_selectable must be boolean")


@dataclass(frozen=True)
class UnitSpec:
    qname: str
    role: str
    unit_class: str
    candidates: tuple[CandidateSpec, ...]
    n_params: int
    serving_group: str | None = None

    def __post_init__(self) -> None:
        for name in ("qname", "role", "unit_class"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"unit {name} must be nonempty")
            object.__setattr__(self, name, value)
        candidates = tuple(self.candidates)
        names = [candidate.format_name for candidate in candidates]
        if not candidates or len(names) != len(set(names)):
            raise ValueError("unit candidates are empty or duplicated")
        if sum(candidate.terminal for candidate in candidates) != 1:
            raise ValueError("unit must declare exactly one passthrough terminal")
        if not any(not candidate.terminal for candidate in candidates):
            raise ValueError("unit has no renderable candidate")
        if not any(candidate.allocator_selectable for candidate in candidates):
            raise ValueError("unit has no allocator-selectable candidate")
        if int(self.n_params) <= 0:
            raise ValueError("unit n_params must be positive")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "n_params", int(self.n_params))

    def segment_for(self, candidate: CandidateSpec) -> SegmentKey:
        if candidate.terminal:
            raise ValueError("passthrough terminal has no transfer segment")
        return SegmentKey(
            candidate.family, self.role, candidate.equivalence_class,
        )


@dataclass(frozen=True)
class PluginDeclaration:
    plugin_id: str
    plugin_version: str
    equivalence_contract: str

    def __post_init__(self) -> None:
        for name in ("plugin_id", "plugin_version", "equivalence_contract"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"plugin declaration {name} is empty")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, str]:
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "equivalence_contract": self.equivalence_contract,
        }


@runtime_checkable
class AnchoredFormatPlugin(Protocol):
    """Five declarations supplied by one platform mapping plugin."""

    def plugin_identity(self) -> PluginDeclaration | Mapping[str, object]: ...

    def describe_candidate(
        self, unit: UnitSpec, format_name: str,
    ) -> CandidateSpec: ...

    def select_anchor(
        self,
        unit: UnitSpec,
        segment: SegmentKey,
        candidates: Sequence[CandidateSpec],
    ) -> str: ...

    def render(
        self, request: "RenderRequest",
    ) -> "ScalarRenderResult | Mapping[str, object]": ...

    def provenance_identity_fields(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class RenderRequest:
    qname: str
    segment: SegmentKey
    format_name: str
    purpose: Literal["anchor", "panel", "validation"] = "anchor"

    @property
    def request_id(self) -> str:
        body = self.to_dict(include_request_id=False)
        digest = canonical_json_sha256(body, where="anchored render request")
        return f"{self.qname}::{self.purpose}::{digest}"

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "qname": self.qname,
            "segment": self.segment.to_dict(),
            "format": self.format_name,
            "purpose": self.purpose,
        }
        if include_request_id:
            body["request_id"] = self.request_id
        return body


@dataclass(frozen=True)
class ScalarRenderResult:
    predicted_dloss: float
    weight_mse_diagnostic: float | None = None

    def __post_init__(self) -> None:
        value = float(self.predicted_dloss)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("rendered AURA must be finite and nonnegative")
        object.__setattr__(self, "predicted_dloss", value)
        if self.weight_mse_diagnostic is not None:
            diagnostic = float(self.weight_mse_diagnostic)
            if not math.isfinite(diagnostic) or diagnostic < 0.0:
                raise ValueError("weight-MSE diagnostic must be finite >= 0")
            object.__setattr__(self, "weight_mse_diagnostic", diagnostic)


@dataclass(frozen=True)
class ProductionRenderReceipt:
    """Identity-bound proof that one scalar came from the production arm."""

    request: RenderRequest
    scalar: ScalarRenderResult
    arm_identity_sha256: str
    payload_identity_sha256: str
    receipt_sha256: str
    render_source: str
    cost_currency: str
    fisher_application_count: int
    rendered_weight_persisted: bool

    def __post_init__(self) -> None:
        for name in (
            "arm_identity_sha256", "payload_identity_sha256", "receipt_sha256",
        ):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"production receipt {name} is not SHA-256")
            object.__setattr__(self, name, value)
        if self.render_source != PRODUCTION_RENDER_SOURCE:
            raise ValueError("receipt is not a production-arm render")
        if self.cost_currency != AURA_CURRENCY:
            raise ValueError("receipt does not carry the AURA currency")
        if int(self.fisher_application_count) != 1:
            raise ValueError("receipt applies the Fisher more than once")
        if self.rendered_weight_persisted is not False:
            raise ValueError("production receipt persisted a rendered weight")
        expected = canonical_json_sha256(
            self.to_dict(include_receipt_sha256=False),
            where="production render receipt",
        )
        if self.receipt_sha256 != expected:
            raise ValueError("production render receipt checksum differs")

    def to_dict(
        self, *, include_receipt_sha256: bool = True,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "request": self.request.to_dict(),
            "predicted_dloss": self.scalar.predicted_dloss,
            "weight_mse_diagnostic": self.scalar.weight_mse_diagnostic,
            "arm_identity_sha256": self.arm_identity_sha256,
            "payload_identity_sha256": self.payload_identity_sha256,
            "render_source": self.render_source,
            "cost_currency": self.cost_currency,
            "fisher_application_count": self.fisher_application_count,
            "rendered_weight_persisted": self.rendered_weight_persisted,
        }
        if include_receipt_sha256:
            body["receipt_sha256"] = self.receipt_sha256
        return body


def make_production_render_receipt(
    request: RenderRequest,
    scalar: ScalarRenderResult,
    *,
    arm_identity: object,
    payload_identity: object,
) -> ProductionRenderReceipt:
    """Create the one generic receipt shape used by every mapping plugin."""
    arm_sha = canonical_json_sha256(
        arm_identity, where="production render arm identity",
    )
    payload_sha = canonical_json_sha256(
        payload_identity, where="production render payload identity",
    )
    return make_production_render_receipt_from_hashes(
        request,
        scalar,
        arm_identity_sha256=arm_sha,
        payload_identity_sha256=payload_sha,
    )


def make_production_render_receipt_from_hashes(
    request: RenderRequest,
    scalar: ScalarRenderResult,
    *,
    arm_identity_sha256: str,
    payload_identity_sha256: str,
) -> ProductionRenderReceipt:
    """Create a receipt from once-validated global identity digests.

    Streamed mapping adapters may have a multi-megabyte global identity.  They
    hash that object once, then use this constructor for each scalar.  The
    per-cell receipt checksum and ``ProductionRenderReceipt`` validation stay
    unchanged; only redundant serialization of the global identity is
    removed.
    """
    arm_sha = str(arm_identity_sha256).lower()
    payload_sha = str(payload_identity_sha256).lower()
    body = {
        "request": request.to_dict(),
        "predicted_dloss": scalar.predicted_dloss,
        "weight_mse_diagnostic": scalar.weight_mse_diagnostic,
        "arm_identity_sha256": arm_sha,
        "payload_identity_sha256": payload_sha,
        "render_source": PRODUCTION_RENDER_SOURCE,
        "cost_currency": AURA_CURRENCY,
        "fisher_application_count": 1,
        "rendered_weight_persisted": False,
    }
    return ProductionRenderReceipt(
        request=request,
        scalar=scalar,
        arm_identity_sha256=arm_sha,
        payload_identity_sha256=payload_sha,
        receipt_sha256=canonical_json_sha256(
            body, where="production render receipt",
        ),
        render_source=PRODUCTION_RENDER_SOURCE,
        cost_currency=AURA_CURRENCY,
        fisher_application_count=1,
        rendered_weight_persisted=False,
    )


@dataclass(frozen=True)
class AnchorScalar:
    qname: str
    segment: SegmentKey
    format_name: str
    predicted_dloss: float
    receipt: ProductionRenderReceipt

    def __post_init__(self) -> None:
        value = float(self.predicted_dloss)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("anchor AURA must be finite and nonnegative")
        expected = RenderRequest(
            self.qname, self.segment, self.format_name, "anchor",
        )
        if self.receipt.request != expected:
            raise ValueError("anchor receipt request differs")
        if self.receipt.scalar.predicted_dloss != value:
            raise ValueError("anchor level differs from its render receipt")
        object.__setattr__(self, "predicted_dloss", value)

    @property
    def render_source(self) -> str:
        return self.receipt.render_source


@dataclass(frozen=True)
class ShapeObservation:
    qname: str
    segment: SegmentKey
    format_name: str
    predicted_dloss: float
    weight_mse_diagnostic: float | None = None
    receipt: ProductionRenderReceipt | None = None

    def __post_init__(self) -> None:
        if self.receipt is None:
            raise ValueError("shape observation has no production receipt")
        request = self.receipt.request
        if (
            request.qname != self.qname
            or request.segment != self.segment
            or request.format_name != self.format_name
            or request.purpose not in {"panel", "validation"}
        ):
            raise ValueError("shape observation receipt request differs")
        if self.receipt.scalar.predicted_dloss != float(self.predicted_dloss):
            raise ValueError("shape observation level differs from receipt")
        if self.receipt.scalar.weight_mse_diagnostic != (
            None if self.weight_mse_diagnostic is None
            else float(self.weight_mse_diagnostic)
        ):
            raise ValueError("shape diagnostic differs from receipt")


@dataclass(frozen=True)
class ShapeFit:
    segment: SegmentKey
    g_by_format: Mapping[str, float]
    reference_format: str
    coefficients: tuple[float, ...]
    design_rank: int
    design_rank_required: int
    n_units: int
    n_observations: int
    arm_identity_sha256: str
    payload_identity_sha256: str
    panel_receipts_sha256: str
    shape_fit_currency: str = AURA_CURRENCY
    aura_vs_weight_diagnostic: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.shape_fit_currency != AURA_CURRENCY:
            raise ValueError("shape fit is not in the AURA currency")
        for name in (
            "arm_identity_sha256",
            "payload_identity_sha256",
            "panel_receipts_sha256",
        ):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"shape fit {name} is invalid")
            object.__setattr__(self, name, digest)

    def ratio(self, target: str, anchor: str) -> float:
        try:
            numerator = float(self.g_by_format[target])
            denominator = float(self.g_by_format[anchor])
        except KeyError as exc:
            raise AnchoredCostError(
                f"fit {self.segment.stamp} lacks {exc.args[0]!r}"
            ) from exc
        if numerator <= 0.0 or denominator <= 0.0:
            raise AnchoredCostError("shape ratio is not strictly positive")
        return numerator / denominator


@dataclass(frozen=True)
class PricedCell:
    qname: str
    candidate: CandidateSpec
    predicted_dloss: float
    segment: SegmentKey | None
    cost_source: str
    anchor_format: str | None = None
    anchor_predicted_dloss: float | None = None
    shape_ratio: float | None = None
    anchor_receipt_sha256: str | None = None
    arm_identity_sha256: str | None = None

    def allocation_entry(self) -> dict[str, object]:
        out: dict[str, object] = {
            "predicted_dloss": float(self.predicted_dloss),
            "memory_bytes": int(self.candidate.payload_bytes),
            "cost_currency": AURA_CURRENCY,
            "cost_source": self.cost_source,
        }
        if self.segment is not None:
            out.update({
                "anchor_segment": self.segment.to_dict(),
                "anchor_format": self.anchor_format,
                "anchor_predicted_dloss": self.anchor_predicted_dloss,
                "shape_ratio": self.shape_ratio,
                "anchor_receipt_sha256": self.anchor_receipt_sha256,
                "arm_identity_sha256": self.arm_identity_sha256,
                "fisher_application_count": 1,
            })
        return out


@dataclass(frozen=True)
class HullResult:
    vertices: tuple[str, ...]
    interior: tuple[str, ...]


def _plugin_identity(plugin: AnchoredFormatPlugin) -> dict[str, object]:
    raw = plugin.plugin_identity()
    if isinstance(raw, PluginDeclaration):
        body: Mapping[str, object] = raw.to_dict()
    elif isinstance(raw, Mapping):
        body = raw
    else:
        raise AnchoredCostError("plugin_identity must be a declaration/object")
    canonical = dict(canonical_json(body, where="anchored plugin identity"))
    if not canonical:
        raise AnchoredCostError("plugin identity is empty")
    return canonical


def _plugin_provenance(plugin: AnchoredFormatPlugin) -> dict[str, object]:
    raw = plugin.provenance_identity_fields()
    if not isinstance(raw, Mapping):
        raise AnchoredCostError("plugin provenance must be an object")
    canonical = dict(canonical_json(raw, where="anchored plugin provenance"))
    arm = canonical.get("arm_identity")
    if not isinstance(arm, (str, Mapping)) or not arm:
        raise AnchoredCostError("plugin provenance has no arm_identity")
    return canonical


def candidates_by_segment(
    unit: UnitSpec,
    plugin: AnchoredFormatPlugin,
) -> dict[SegmentKey, tuple[CandidateSpec, ...]]:
    grouped: dict[SegmentKey, list[CandidateSpec]] = defaultdict(list)
    for declared in unit.candidates:
        resolved = plugin.describe_candidate(unit, declared.format_name)
        if resolved != declared:
            raise AnchoredCostError(
                f"{unit.qname}/{declared.format_name}: plugin description "
                "differs from the identity-bound unit ladder"
            )
        if resolved.terminal:
            continue
        grouped[unit.segment_for(resolved)].append(resolved)
    return {
        segment: tuple(sorted(values, key=lambda item: (
            item.bits, item.coordinate, item.format_name,
        )))
        for segment, values in grouped.items()
    }


def plan_anchor_requests(
    units: Sequence[UnitSpec],
    plugin: AnchoredFormatPlugin,
) -> tuple[RenderRequest, ...]:
    """Plan exactly one production render per legal unit/segment."""
    _plugin_identity(plugin)
    _plugin_provenance(plugin)
    requests: list[RenderRequest] = []
    seen: set[str] = set()
    for unit in sorted(units, key=lambda item: item.qname):
        if unit.qname in seen:
            raise AnchoredCostError(f"duplicate unit {unit.qname!r}")
        seen.add(unit.qname)
        segments = candidates_by_segment(unit, plugin)
        if not segments:
            raise AnchoredCostError(f"{unit.qname}: no renderable segment")
        for segment, candidates in sorted(segments.items()):
            anchor = str(plugin.select_anchor(unit, segment, candidates))
            legal = {candidate.format_name for candidate in candidates}
            if anchor not in legal:
                raise AnchoredCostError(
                    f"{unit.qname}/{segment.stamp}: plugin anchor {anchor!r} "
                    f"is outside its segment {sorted(legal)}"
                )
            requests.append(RenderRequest(
                unit.qname, segment, anchor, "anchor",
            ))
    return tuple(requests)


def _coerce_scalar(
    raw: ScalarRenderResult | Mapping[str, object],
) -> ScalarRenderResult:
    if isinstance(raw, ScalarRenderResult):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("renderer must return scalar data")
    forbidden = sorted(_FORBIDDEN_COST_FIELDS & set(raw))
    if forbidden:
        raise AnchoredCostError(
            f"renderer returned forbidden parallel/sensitivity cost inputs "
            f"{forbidden}"
        )
    allowed = {"predicted_dloss", "weight_mse_diagnostic"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise AnchoredCostError(
            f"renderer returned non-scalar/unknown fields {unknown}"
        )
    return ScalarRenderResult(
        predicted_dloss=float(raw["predicted_dloss"]),
        weight_mse_diagnostic=(
            float(raw["weight_mse_diagnostic"])
            if raw.get("weight_mse_diagnostic") is not None else None
        ),
    )


def run_scalar_render_campaign(
    requests: Iterable[RenderRequest],
    plugin: AnchoredFormatPlugin,
    *,
    checkpoint_dir: str | Path,
    identity: Mapping[str, object],
    resume: bool,
    stage: str = "anchored-production-render",
) -> dict[str, ProductionRenderReceipt]:
    """Render/checkpoint scalar cells by semantic identity, never position."""
    planned = tuple(sorted(requests, key=lambda item: item.request_id))
    request_ids = [request.request_id for request in planned]
    if len(request_ids) != len(set(request_ids)):
        raise AnchoredCostError("render plan contains duplicate requests")
    bound_identity = dict(canonical_json(identity, where="campaign identity"))
    required_identity = {
        "model_identity", "menu_identity", "calibration_identity",
    }
    missing_identity = sorted(
        name for name in required_identity
        if name not in bound_identity or bound_identity[name] in (None, "", {})
    )
    if missing_identity:
        raise AnchoredCostError(
            "campaign identity lacks required fields "
            f"{missing_identity}"
        )
    plugin_provenance = _plugin_provenance(plugin)
    bound_identity.update({
        "plugin_identity": _plugin_identity(plugin),
        "plugin_provenance": plugin_provenance,
        "cost_currency": AURA_CURRENCY,
        "fisher_application_count": 1,
        "requests_sha256": canonical_json_sha256(
            [request.to_dict() for request in planned],
            where="anchored render plan",
        ),
    })
    root, identity_sha, completed = prepare_journal(
        checkpoint_dir,
        stage=stage,
        resume=resume,
        identity=bound_identity,
        qnames=request_ids,
    )
    receipt_payload_identity = {
        name: value for name, value in bound_identity.items()
        if name != "requests_sha256"
    }
    results: dict[str, ProductionRenderReceipt] = {}
    for request in planned:
        state = completed.get(request.request_id)
        scalar: ScalarRenderResult
        if state is not None:
            raw_receipt = state.get("production_render_receipt")
            if not isinstance(raw_receipt, Mapping):
                raise AnchoredCostError(
                    "checkpoint has no production render receipt"
                )
            scalar = ScalarRenderResult(
                float(raw_receipt["predicted_dloss"]),
                (
                    float(raw_receipt["weight_mse_diagnostic"])
                    if raw_receipt.get("weight_mse_diagnostic") is not None
                    else None
                ),
            )
            receipt = make_production_render_receipt(
                request,
                scalar,
                arm_identity=plugin_provenance["arm_identity"],
                payload_identity=receipt_payload_identity,
            )
            if dict(raw_receipt) != receipt.to_dict():
                raise AnchoredCostError(
                    "checkpoint production receipt semantics differ"
                )
        else:
            scalar = _coerce_scalar(plugin.render(request))
            receipt = make_production_render_receipt(
                request,
                scalar,
                arm_identity=plugin_provenance["arm_identity"],
                payload_identity=receipt_payload_identity,
            )
            write_unit(
                root,
                stage=stage,
                qname=request.request_id,
                identity_sha256=identity_sha,
                state={
                    "production_render_receipt": receipt.to_dict(),
                },
            )
        results[request.request_id] = receipt
    return results


def anchors_from_results(
    requests: Sequence[RenderRequest],
    results: Mapping[str, ProductionRenderReceipt],
) -> dict[tuple[str, SegmentKey], AnchorScalar]:
    anchors: dict[tuple[str, SegmentKey], AnchorScalar] = {}
    for request in requests:
        if request.purpose != "anchor":
            continue
        key = (request.qname, request.segment)
        if key in anchors:
            raise AnchoredCostError(f"duplicate anchor {key}")
        receipt = results[request.request_id]
        if not isinstance(receipt, ProductionRenderReceipt):
            raise AnchoredCostError(
                "bare scalar cannot become a production anchor"
            )
        if receipt.request != request:
            raise AnchoredCostError("anchor result request identity differs")
        anchors[key] = AnchorScalar(
            request.qname,
            request.segment,
            request.format_name,
            receipt.scalar.predicted_dloss,
            receipt,
        )
    return anchors


def _rank_and_solve(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
) -> tuple[int, tuple[float, ...]]:
    """Solve centered least squares through normal equations with pivoting."""
    if not x_rows:
        raise AnchoredCostError("shape design is empty")
    width = len(x_rows[0])
    if width < 1 or any(len(row) != width for row in x_rows):
        raise AnchoredCostError("shape feature width is invalid")
    # Plugin features have no prescribed unit.  Normalize every column before
    # rank detection/solving so a plugin expressing the same coordinate in
    # 1e-7 rather than 1 cannot turn an identifiable design into rank zero.
    column_scales = [
        math.sqrt(math.fsum(row[index] * row[index] for row in x_rows))
        for index in range(width)
    ]
    active = [scale > 0.0 and math.isfinite(scale) for scale in column_scales]
    if not all(active):
        return sum(active), tuple()
    normalized = [
        [row[index] / column_scales[index] for index in range(width)]
        for row in x_rows
    ]
    gram = [
        [math.fsum(row[i] * row[j] for row in normalized) for j in range(width)]
        for i in range(width)
    ]
    rhs = [
        math.fsum(row[i] * value for row, value in zip(normalized, y_rows))
        for i in range(width)
    ]
    scale = max((abs(value) for row in gram for value in row), default=1.0)
    tolerance = scale * 1e-12
    augmented = [gram[index] + [rhs[index]] for index in range(width)]
    rank = 0
    for column in range(width):
        pivot = max(
            range(rank, width), key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= tolerance:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        divisor = augmented[rank][column]
        augmented[rank] = [value / divisor for value in augmented[rank]]
        for row in range(width):
            if row == rank:
                continue
            factor = augmented[row][column]
            if abs(factor) <= tolerance:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[rank],
                )
            ]
        rank += 1
    if rank != width:
        return rank, tuple()
    solution = [0.0] * width
    for row in range(width):
        pivot_columns = [
            column for column in range(width)
            if abs(augmented[row][column] - 1.0) <= 1e-9
            and all(
                abs(augmented[other][column]) <= 1e-9
                for other in range(width) if other != row
            )
        ]
        if len(pivot_columns) != 1:
            raise AnchoredCostError("shape solve pivot reconstruction failed")
        solution[pivot_columns[0]] = augmented[row][-1]
    return rank, tuple(
        solution[index] / column_scales[index] for index in range(width)
    )


def _fit_currency(
    observations: Sequence[ShapeObservation],
    candidates: Sequence[CandidateSpec],
    *,
    value_getter: Callable[[ShapeObservation], float | None],
    currency_name: str,
) -> tuple[dict[str, float], tuple[float, ...], int]:
    by_format = {candidate.format_name: candidate for candidate in candidates}
    widths = {len(candidate.shape_features) for candidate in candidates}
    if len(widths) != 1:
        raise AnchoredCostError("segment candidates use mixed shape bases")
    width = next(iter(widths))
    by_unit: dict[str, list[tuple[tuple[float, ...], float]]] = defaultdict(list)
    for observation in observations:
        if observation.format_name not in by_format:
            raise AnchoredCostError(
                f"{currency_name} observation lies outside target segment"
            )
        raw = value_getter(observation)
        if raw is None:
            raise AnchoredCostError(f"{currency_name} diagnostic is incomplete")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise AnchoredCostError(
                f"{currency_name} fit requires positive finite cells"
            )
        by_unit[observation.qname].append((
            by_format[observation.format_name].shape_features,
            math.log10(value),
        ))
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[float] = []
    for qname, rows in sorted(by_unit.items()):
        if len(rows) < 2:
            raise AnchoredCostError(
                f"panel unit {qname!r} has fewer than two rungs"
            )
        feature_means = tuple(
            math.fsum(features[index] for features, _value in rows) / len(rows)
            for index in range(width)
        )
        value_mean = math.fsum(value for _features, value in rows) / len(rows)
        for features, value in rows:
            x_rows.append(tuple(
                features[index] - feature_means[index]
                for index in range(width)
            ))
            y_rows.append(value - value_mean)
    rank, coefficients = _rank_and_solve(x_rows, y_rows)
    if rank != width:
        raise AnchoredCostError(
            f"{currency_name} panel design rank is {rank} of {width}"
        )
    reference = min(candidates, key=lambda item: (
        item.bits, item.coordinate, item.format_name,
    ))
    g = {
        candidate.format_name: 10.0 ** math.fsum(
            coefficient * (feature - reference.shape_features[index])
            for index, (coefficient, feature) in enumerate(zip(
                coefficients, candidate.shape_features,
            ))
        )
        for candidate in candidates
    }
    return g, coefficients, rank


def fit_segment_shape(
    observations: Sequence[ShapeObservation],
    *,
    segment: SegmentKey,
    candidates: Sequence[CandidateSpec],
) -> ShapeFit:
    """Fit one plugin feature basis inside exactly one declared segment."""
    if not observations or not candidates:
        raise AnchoredCostError("cannot fit an empty segment")
    if {observation.segment for observation in observations} != {segment}:
        raise AnchoredCostError(
            "one shape fit may not span family, role, or equivalence class"
        )
    receipts = [observation.receipt for observation in observations]
    if any(receipt is None for receipt in receipts):
        raise AnchoredCostError("shape fit contains an unreceipted cell")
    typed_receipts = [
        receipt for receipt in receipts
        if isinstance(receipt, ProductionRenderReceipt)
    ]
    if len(typed_receipts) != len(receipts):
        raise AnchoredCostError("shape fit receipt type is invalid")
    if len({receipt.arm_identity_sha256 for receipt in typed_receipts}) != 1:
        raise AnchoredCostError("shape fit spans production arms")
    if len({receipt.payload_identity_sha256 for receipt in typed_receipts}) != 1:
        raise AnchoredCostError("shape fit spans render payload identities")
    for candidate in candidates:
        candidate_segment = SegmentKey(
            candidate.family, segment.role, candidate.equivalence_class,
        )
        if candidate.terminal or candidate_segment != segment:
            raise AnchoredCostError(
                "shape target ladder crosses a declared equivalence boundary"
            )
    aura_g, coefficients, rank = _fit_currency(
        observations,
        candidates,
        value_getter=lambda item: item.predicted_dloss,
        currency_name="AURA",
    )
    diagnostic = None
    weight_values = [item.weight_mse_diagnostic for item in observations]
    if any(value is not None for value in weight_values):
        if not all(value is not None for value in weight_values):
            raise AnchoredCostError("weight-MSE diagnostic coverage is partial")
        weight_g, weight_coefficients, weight_rank = _fit_currency(
            observations,
            candidates,
            value_getter=lambda item: item.weight_mse_diagnostic,
            currency_name="weight-MSE diagnostic",
        )
        differences = {
            name: math.log10(aura_g[name] / weight_g[name]) for name in aura_g
        }
        diagnostic = {
            "currency_invariance_test_only": True,
            "weight_coefficients": list(weight_coefficients),
            "weight_design_rank": weight_rank,
            "dex_difference_by_format": differences,
            "rms_dex": math.sqrt(
                math.fsum(value * value for value in differences.values())
                / len(differences)
            ),
            "max_abs_dex": max(abs(value) for value in differences.values()),
        }
    reference = min(candidates, key=lambda item: (
        item.bits, item.coordinate, item.format_name,
    ))
    return ShapeFit(
        segment=segment,
        g_by_format=aura_g,
        reference_format=reference.format_name,
        coefficients=coefficients,
        design_rank=rank,
        design_rank_required=len(reference.shape_features),
        n_units=len({item.qname for item in observations}),
        n_observations=len(observations),
        arm_identity_sha256=typed_receipts[0].arm_identity_sha256,
        payload_identity_sha256=typed_receipts[0].payload_identity_sha256,
        panel_receipts_sha256=canonical_json_sha256(
            sorted(receipt.receipt_sha256 for receipt in typed_receipts),
            where="shape fitting panel receipts",
        ),
        aura_vs_weight_diagnostic=diagnostic,
    )


def lower_convex_hull(
    points: Mapping[str, tuple[float, float]],
) -> HullResult:
    if not points:
        raise AnchoredCostError("cannot compute an empty hull")
    ordered = sorted(
        (float(bits), float(cost), str(name))
        for name, (bits, cost) in points.items()
    )
    if any(
        not math.isfinite(bits) or not math.isfinite(cost) or cost <= 0.0
        for bits, cost, _name in ordered
    ):
        raise AnchoredCostError(
            "hull proof requires finite, strictly positive shape costs"
        )
    deduplicated: list[tuple[float, float, str]] = []
    for point in ordered:
        if deduplicated and point[0] == deduplicated[-1][0]:
            if (point[1], point[2]) < (
                deduplicated[-1][1], deduplicated[-1][2],
            ):
                deduplicated[-1] = point
            continue
        deduplicated.append(point)
    # A higher-rate point whose cost is no lower is Pareto dominated before
    # convexity is considered.  Keeping such a right-hand endpoint would call
    # it a hull vertex even though no positive byte price could select it.
    frontier: list[tuple[float, float, str]] = []
    best_cost = math.inf
    for point in deduplicated:
        if point[1] < best_cost:
            frontier.append(point)
            best_cost = point[1]
    hull: list[tuple[float, float, str]] = []
    for point in frontier:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            cross = (
                (middle[0] - left[0]) * (point[1] - middle[1])
                - (middle[1] - left[1]) * (point[0] - middle[0])
            )
            if cross > 1e-15:
                break
            hull.pop()
        hull.append(point)
    vertices = tuple(point[2] for point in hull)
    return HullResult(
        vertices=vertices,
        interior=tuple(sorted(
            set(points) - set(vertices), key=lambda name: (points[name][0], name),
        )),
    )


def price_anchored_candidates(
    units: Sequence[UnitSpec],
    plugin: AnchoredFormatPlugin,
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
    fits: Mapping[SegmentKey, ShapeFit],
) -> dict[str, tuple[PricedCell, ...]]:
    """Price every legal candidate from a real same-segment anchor."""
    rows: dict[str, tuple[PricedCell, ...]] = {}
    plugin_arm_sha = canonical_json_sha256(
        _plugin_provenance(plugin)["arm_identity"],
        where="anchored pricing production arm",
    )
    for unit in sorted(units, key=lambda item: item.qname):
        priced: list[PricedCell] = []
        for segment, candidates in sorted(candidates_by_segment(unit, plugin).items()):
            anchor = anchors.get((unit.qname, segment))
            if anchor is None or anchor.segment != segment:
                raise AnchoredCostError(
                    f"{unit.qname}/{segment.stamp}: no same-segment real anchor"
                )
            expected_request = RenderRequest(
                unit.qname, segment, anchor.format_name, "anchor",
            )
            if (
                anchor.qname != unit.qname
                or anchor.receipt.request != expected_request
                or anchor.receipt.arm_identity_sha256 != plugin_arm_sha
            ):
                raise AnchoredCostError(
                    "anchor receipt qname/request/production arm differs"
                )
            legal = {candidate.format_name for candidate in candidates}
            if anchor.format_name not in legal:
                raise AnchoredCostError("anchor lies across equivalence boundary")
            fit = fits.get(segment)
            if fit is None or fit.segment != segment:
                raise AnchoredCostError("fit lies across equivalence boundary")
            if (
                fit.shape_fit_currency != AURA_CURRENCY
                or fit.arm_identity_sha256 != plugin_arm_sha
                or fit.payload_identity_sha256
                != anchor.receipt.payload_identity_sha256
            ):
                raise AnchoredCostError(
                    "fit and anchor do not share one AURA render identity"
                )
            for candidate in candidates:
                ratio = fit.ratio(candidate.format_name, anchor.format_name)
                value = anchor.predicted_dloss * ratio
                if not math.isfinite(value) or value < 0.0:
                    raise AnchoredCostError("extrapolated AURA is invalid")
                priced.append(PricedCell(
                    unit.qname,
                    candidate,
                    value,
                    segment,
                    "anchored_aura_extrapolation",
                    anchor.format_name,
                    anchor.predicted_dloss,
                    ratio,
                    anchor.receipt.receipt_sha256,
                    plugin_arm_sha,
                ))
        terminal = next(
            candidate for candidate in unit.candidates if candidate.terminal
        )
        if terminal.allocator_selectable:
            priced.append(PricedCell(
                unit.qname,
                terminal,
                0.0,
                None,
                # The allocator's byte-verbatim contract, not a label of our
                # own: cost_entry_is_source_passthrough() refuses anything
                # else, and a near-miss string silently misclassifies the
                # activation branch.
                SOURCE_PASSTHROUGH_COST_SOURCE,
            ))
        rows[unit.qname] = tuple(sorted(
            priced, key=lambda item: (
                item.candidate.payload_bytes, item.candidate.format_name,
            ),
        ))
    assert_aura_only_cost_table(rows)
    return rows


def assert_aura_only_cost_table(
    rows: Mapping[str, Sequence[PricedCell]],
) -> None:
    for qname, cells in rows.items():
        anchored = [cell for cell in cells if cell.segment is not None]
        if not anchored:
            raise AnchoredCostError(f"{qname}: no anchored candidate")
        for cell in anchored:
            entry = cell.allocation_entry()
            forbidden = sorted(_FORBIDDEN_COST_FIELDS & set(entry))
            if forbidden:
                raise AnchoredCostError(
                    f"{qname}/{cell.candidate.format_name}: forbidden cost "
                    f"inputs {forbidden}"
                )
            if entry.get("fisher_application_count") != 1:
                raise AnchoredCostError("AURA Fisher application count differs")
            if not entry.get("anchor_receipt_sha256") or not entry.get(
                "arm_identity_sha256"
            ):
                raise AnchoredCostError("anchored cell has no render receipt")
            if cell.cost_source != "anchored_aura_extrapolation":
                raise AnchoredCostError("candidate is not anchor-priced")


def extrapolation_distance_report(
    units: Sequence[UnitSpec],
    plugin: AnchoredFormatPlugin,
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
    assignment: Mapping[str, str],
) -> dict[str, object]:
    """Report selected-to-anchor distance in plugin rung coordinates."""
    distribution: Counter[float] = Counter()
    rows: list[dict[str, object]] = []
    by_name = {unit.qname: unit for unit in units}
    if len(by_name) != len(units):
        raise AnchoredCostError("exposure input contains duplicate units")
    missing = sorted(set(by_name) - set(assignment))
    extra = sorted(set(assignment) - set(by_name))
    if missing or extra:
        raise AnchoredCostError(
            "exposure assignment differs from the complete unit set: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    plugin_arm_sha = canonical_json_sha256(
        _plugin_provenance(plugin)["arm_identity"],
        where="exposure production arm",
    )
    terminal_count = 0
    extrapolated_count = 0
    for qname, selected_name in sorted(assignment.items()):
        unit = by_name[qname]
        # Re-resolve the complete ladder against the plugin declaration before
        # interpreting a selected label or its transfer coordinate.
        candidates_by_segment(unit, plugin)
        candidate = next((
            item for item in unit.candidates
            if item.format_name == selected_name
        ), None)
        if candidate is None:
            raise AnchoredCostError(
                f"{qname}: exposure selection {selected_name!r} is illegal"
            )
        if not candidate.allocator_selectable:
            raise AnchoredCostError(
                f"{qname}: exposure selection {selected_name!r} was retained "
                "only as source/render identity and is not allocator-selectable"
            )
        if candidate.terminal:
            terminal_count += 1
            rows.append({
                "qname": qname,
                "terminal": True,
                "selected_format": selected_name,
                "coordinate_distance": None,
            })
            continue
        segment = unit.segment_for(candidate)
        anchor = anchors.get((qname, segment))
        if anchor is None:
            raise AnchoredCostError(
                f"{qname}/{segment.stamp}: exposure anchor is absent"
            )
        expected_request = RenderRequest(
            qname, segment, anchor.format_name, "anchor",
        )
        anchor_candidate = next((
            item for item in unit.candidates
            if item.format_name == anchor.format_name
        ), None)
        if (
            anchor.qname != qname
            or anchor.receipt.request != expected_request
            or anchor.receipt.arm_identity_sha256 != plugin_arm_sha
            or anchor_candidate is None
            or anchor_candidate.terminal
            or unit.segment_for(anchor_candidate) != segment
        ):
            raise AnchoredCostError(
                f"{qname}/{segment.stamp}: exposure anchor receipt is invalid"
            )
        distance = abs(candidate.coordinate - anchor_candidate.coordinate)
        distribution[distance] += 1
        extrapolated_count += distance > 0.0
        rows.append({
            "qname": qname,
            "terminal": False,
            "segment": segment.to_dict(),
            "anchor_format": anchor.format_name,
            "selected_format": selected_name,
            "coordinate_distance": distance,
        })
    return {
        "unit": "plugin_rung_coordinate",
        "count": len(rows) - terminal_count,
        "total_unit_count": len(rows),
        "anchored_selection_count": len(rows) - terminal_count,
        "terminal_selection_count": terminal_count,
        "extrapolated_selection_count": extrapolated_count,
        "extrapolated_fraction_of_all_units": (
            extrapolated_count / len(rows) if rows else 0.0
        ),
        "extrapolated_fraction_of_anchored_units": (
            extrapolated_count / (len(rows) - terminal_count)
            if len(rows) > terminal_count else 0.0
        ),
        "distribution": [
            {"distance": distance, "unit_count": count}
            for distance, count in sorted(distribution.items())
        ],
        "rows": rows,
    }


def run_allocator_once(
    *,
    command: Sequence[str],
    output_dir: str | Path,
    environment_updates: Mapping[str, str] | None = None,
    invocation_provenance: Mapping[str, object] | None = None,
    pass_fds: Sequence[int] = (),
    resume: bool = False,
) -> Path:
    """Invoke one allocator or reuse its exact identity-bound completion.

    A failed/interrupted invocation retains both its output and immutable
    identity marker.  On an exact ``resume``, that incomplete directory is
    moved to a sibling recovery path before a clean retry; it is never
    deleted.  A completed invocation is reused only after both output files
    and its receipt checksum validate.
    """
    output = Path(output_dir)
    identity = _allocator_invocation_identity(
        command=command,
        environment_updates=environment_updates,
        invocation_provenance=invocation_provenance,
    )
    identity_sha256 = canonical_json_sha256(
        identity, where="anchored allocator invocation identity",
    )
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            raise AnchoredCostError(
                f"allocator output is not a real directory: {output}"
            )
        if not resume:
            raise AnchoredCostError(
                f"allocator output exists: {output}; refusing overwrite"
            )
        identity_path = output / _ALLOCATOR_IDENTITY_FILE
        if identity_path.is_file():
            _load_allocator_identity(
                output,
                expected_identity=identity,
                expected_sha256=identity_sha256,
            )
            if _validate_completed_allocator_output(
                output, identity_sha256=identity_sha256,
            ):
                return output
            _relocate_incomplete_directory(
                output, identity_sha256=identity_sha256,
            )
        else:
            # A crash can occur after mkdir and before the first atomic marker
            # publish. Preserve that unbound state and retry in a clean path.
            _relocate_incomplete_directory(output, identity_sha256=None)
    output.mkdir(parents=True, exist_ok=False)
    _write_allocator_identity(output, identity, identity_sha256)
    environment = dict(os.environ)
    environment.update({
        str(name): str(value)
        for name, value in (environment_updates or {}).items()
    })
    inherited_fds = tuple(int(value) for value in pass_fds)
    if (
        any(value < 0 for value in inherited_fds)
        or len(set(inherited_fds)) != len(inherited_fds)
    ):
        raise AnchoredCostError("allocator pass_fds are invalid")
    completed = subprocess.run(
        [str(value) for value in command], check=False, env=environment,
        pass_fds=inherited_fds,
    )
    if completed.returncode != 0:
        raise AnchoredCostError(
            f"allocator failed with exit code {completed.returncode}"
        )
    outputs = _allocator_output_descriptors(output)
    canonical_provenance = dict(identity["invocation_provenance"])
    reserved_receipt_fields = {
        "schema", "identity_sha256", "identity", "command",
        "environment_updates", "invocation_provenance", "outputs",
        "receipt_sha256",
    }
    top_level_provenance = {
        name: value for name, value in canonical_provenance.items()
        if name not in reserved_receipt_fields
    }
    receipt: dict[str, object] = {
        "schema": _ALLOCATOR_RECEIPT_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": identity,
        "command": list(identity["command"]),
        "environment_updates": dict(identity["environment_updates"]),
        "invocation_provenance": canonical_provenance,
        "outputs": outputs,
        # Preserve the v1 receipt's convenient top-level provenance fields
        # while binding their authoritative copy inside the identity.
        **top_level_provenance,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(
        receipt, where="allocator completion receipt",
    )
    atomic_write_bytes(
        output / _ALLOCATOR_RECEIPT_FILE,
        json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ).encode(),
    )
    return output


__all__ = [
    "ANCHOR_SEGMENT_FIELDS",
    "AURA_CURRENCY",
    "AnchorScalar",
    "AnchoredCostError",
    "AnchoredFormatPlugin",
    "CandidateSpec",
    "HullResult",
    "PluginDeclaration",
    "PricedCell",
    "PRODUCTION_RENDER_SOURCE",
    "ProductionRenderReceipt",
    "RenderRequest",
    "ScalarRenderResult",
    "SegmentKey",
    "ShapeFit",
    "ShapeObservation",
    "UnitSpec",
    "anchors_from_results",
    "assert_aura_only_cost_table",
    "candidates_by_segment",
    "extrapolation_distance_report",
    "fit_segment_shape",
    "lower_convex_hull",
    "make_production_render_receipt",
    "make_production_render_receipt_from_hashes",
    "plan_anchor_requests",
    "price_anchored_candidates",
    "run_allocator_once",
    "run_scalar_render_campaign",
]
