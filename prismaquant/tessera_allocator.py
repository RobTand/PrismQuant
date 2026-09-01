"""Research-only bridge from parameterized Trellis wires to allocator points.

The production format registry, producer menu, exporters, and pipeline do not
import this module.  A Trellis rate is addressed by ``(family, q256, layout,
pre-render recipe identity)`` and is priced with the exact serialized tensor
payload byte count.  The address is not a rendered wire-content digest.
The resulting ephemeral keys may be mixed with any other
``allocator_solver.Candidate`` objects in an offline DP/Lagrangian experiment;
they grant no render, export, or serving authority.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
import math
import re
from types import MappingProxyType

from .allocator_solver import Candidate
from .serving_profiles import (
    ResolvedServingLane,
    ServingFormatDecision,
    check_serving_format,
    check_serving_shape,
    load_serving_profile,
    serving_lane_route,
)
from .tessera_footprint import (
    tessera_tensor_payload_breakdown,
    validate_tessera_tensor_payload_breakdown,
)
from .tessera_formats import (
    RATE_SURFACE_ADAPTIVE,
    TesseraFamily,
    TesseraFormatError,
    TesseraRateSurface,
    get_tessera_family,
)


TESSERA_ALLOCATOR_CANDIDATE_SCHEMA = (
    "prismaquant.tessera_allocator_candidate.v1"
)
TESSERA_PARETO_FRONTIER_SCHEMA = "prismaquant.tessera_pareto_frontier.v1"
TESSERA_ADAPTIVE_RATE_PROPOSAL_SCHEMA = (
    "prismaquant.trellis_adaptive_rate_proposal.v1"
)

_VARIANT_LABEL = re.compile(r"[A-Za-z0-9_.-]+")
_SM_PLATFORM = re.compile(r"sm[_-]?([0-9]+)")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _deep_freeze(value: object) -> object:
    """Copy JSON-shaped metadata into recursively immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _deep_freeze(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    """Return a fresh JSON-shaped copy of recursively frozen metadata."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_deep_thaw(item) for item in value)
    return value


def _finite_float(
    value: object,
    *,
    field: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TesseraFormatError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TesseraFormatError(f"{field} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " nonnegative" if nonnegative else ""
        raise TesseraFormatError(f"{field} must be a finite{suffix} number")
    return result


@dataclass(frozen=True, slots=True)
class _ExactNonnegativeRational:
    """Internal exact ordering key that also admits zero distance."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or self.numerator < 0:
            raise TesseraFormatError(
                "exact distance numerator must be a nonnegative integer"
            )
        if type(self.denominator) is not int or self.denominator <= 0:
            raise TesseraFormatError(
                "exact distance denominator must be a positive integer"
            )
        divisor = math.gcd(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", self.numerator // divisor)
        object.__setattr__(self, "denominator", self.denominator // divisor)
        try:
            rounded = self.numerator / self.denominator
        except OverflowError as exc:
            raise TesseraFormatError(
                "exact distance exceeds the finite binary64 diagnostic domain"
            ) from exc
        if not math.isfinite(rounded):
            raise TesseraFormatError(
                "exact distance exceeds the finite binary64 diagnostic domain"
            )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ExactNonnegativeRational):
            return NotImplemented
        return (
            self.numerator * other.denominator
            < other.numerator * self.denominator
        )

    @property
    def rounded_binary64_diagnostic(self) -> float:
        return self.numerator / self.denominator

    def as_dict(self) -> dict[str, object]:
        return {
            "numerator_decimal": str(self.numerator),
            "denominator_decimal": str(self.denominator),
            "rounded_binary64_diagnostic": (
                self.rounded_binary64_diagnostic
            ),
        }


@dataclass(frozen=True, slots=True)
class TesseraExactMarginal:
    """One canonical positive rational loss-per-byte threshold.

    Measured losses are finite binary64 values, hence exact dyadic rationals.
    Dividing their exact difference by an integer byte delta need not remain
    dyadic, so both integers are retained.  Ordering uses integer
    cross-products; the rounded float is report-only.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or self.numerator <= 0:
            raise TesseraFormatError(
                "exact marginal numerator must be a positive integer"
            )
        if type(self.denominator) is not int or self.denominator <= 0:
            raise TesseraFormatError(
                "exact marginal denominator must be a positive integer"
            )
        divisor = math.gcd(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", self.numerator // divisor)
        object.__setattr__(self, "denominator", self.denominator // divisor)
        try:
            rounded = self.numerator / self.denominator
        except OverflowError as exc:
            raise TesseraFormatError(
                "exact marginal exceeds the finite binary64 diagnostic domain"
            ) from exc
        if not math.isfinite(rounded):
            raise TesseraFormatError(
                "exact marginal exceeds the finite binary64 diagnostic domain"
            )

    @property
    def rounded_binary64_diagnostic(self) -> float:
        """Nearest binary64 display value; never an allocator threshold."""

        return self.numerator / self.denominator

    def compare(self, other: "TesseraExactMarginal") -> int:
        """Compare two positive rationals using exact integer products."""

        if not isinstance(other, TesseraExactMarginal):
            raise TypeError("exact marginal comparison requires like types")
        left = self.numerator * other.denominator
        right = other.numerator * self.denominator
        return (left > right) - (left < right)

    def greater_than_binary64_ratio(
        self,
        numerator: int,
        denominator: int,
    ) -> bool:
        """Return ``self > numerator/denominator`` without float rounding."""

        return self.numerator * denominator > numerator * self.denominator

    def distance_from_binary64_ratio(
        self,
        numerator: int,
        denominator: int,
    ) -> _ExactNonnegativeRational:
        return _ExactNonnegativeRational(
            abs(self.numerator * denominator - numerator * self.denominator),
            self.denominator * denominator,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            # Decimal strings avoid silently narrowing arbitrary-size exact
            # integers in JSON consumers whose number type is binary64.
            "numerator_decimal": str(self.numerator),
            "denominator_decimal": str(self.denominator),
            "rounded_binary64_diagnostic": (
                self.rounded_binary64_diagnostic
            ),
        }


def _exact_positive_float_difference(
    higher: float,
    lower: float,
    *,
    field: str,
) -> tuple[int, int]:
    """Return the reduced exact rational ``higher - lower``."""

    if not math.isfinite(higher) or not math.isfinite(lower) or higher <= lower:
        raise TesseraFormatError(
            f"{field} requires a strictly positive finite loss reduction"
        )
    high_numerator, high_denominator = higher.as_integer_ratio()
    low_numerator, low_denominator = lower.as_integer_ratio()
    numerator = (
        high_numerator * low_denominator
        - low_numerator * high_denominator
    )
    denominator = high_denominator * low_denominator
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _exact_marginal_loss_per_byte(
    higher_loss: float,
    lower_loss: float,
    delta_bytes: int,
    *,
    field: str,
) -> TesseraExactMarginal:
    if type(delta_bytes) is not int or delta_bytes <= 0:
        raise TesseraFormatError(f"{field} requires a positive byte increase")
    numerator, denominator = _exact_positive_float_difference(
        higher_loss,
        lower_loss,
        field=field,
    )
    return TesseraExactMarginal(numerator, denominator * delta_bytes)


def _uncertainty_bounds_loss_per_byte(
    mean_reduction: float,
    cheaper_stderr: float,
    dearer_stderr: float,
    delta_bytes: int,
    uncertainty_z: float,
) -> tuple[float, float]:
    """Compute finite reported CI bounds, rejecting every derived overflow."""

    stderr_sum = cheaper_stderr + dearer_stderr
    if not math.isfinite(stderr_sum):
        raise TesseraFormatError(
            "derived marginal uncertainty stderr sum is not finite"
        )
    uncertainty_radius = uncertainty_z * stderr_sum
    if not math.isfinite(uncertainty_radius):
        raise TesseraFormatError(
            "derived marginal uncertainty radius is not finite"
        )
    low_numerator = mean_reduction - uncertainty_radius
    high_numerator = mean_reduction + uncertainty_radius
    if not math.isfinite(low_numerator) or not math.isfinite(high_numerator):
        raise TesseraFormatError(
            "derived marginal uncertainty interval is not finite"
        )
    low = low_numerator / delta_bytes
    high = high_numerator / delta_bytes
    if not math.isfinite(low) or not math.isfinite(high):
        raise TesseraFormatError(
            "derived marginal uncertainty loss-per-byte bounds are not finite"
        )
    return low, high


def _decision_payload(decision: ServingFormatDecision) -> dict[str, object]:
    return {
        "legal": bool(decision.legal),
        "reason": decision.reason,
        "detail": decision.detail,
        "rule": decision.rule,
    }


def _canonical_servability_gate(
    gate: "TesseraServabilityGate",
) -> "TesseraServabilityGate":
    """Copy every sequence nested in a resolved lane into immutable tuples."""

    if not isinstance(gate, TesseraServabilityGate):
        raise TesseraFormatError(
            "candidate servability must be TesseraServabilityGate"
        )
    lane = gate.serving_lane
    if lane is None:
        return gate
    try:
        fused_rungs = tuple(lane.fused_mid_m_rungs)
        fused_range = (
            None
            if lane.fused_mid_m_range is None
            else tuple(lane.fused_mid_m_range)
        )
        serve_flags = tuple(lane.requires_serve_flags)
    except TypeError as exc:
        raise TesseraFormatError(
            "resolved serving-lane sequences must be iterable"
        ) from exc
    if any(type(value) is not int for value in fused_rungs):
        raise TesseraFormatError(
            "resolved serving-lane fused rungs must be JSON integers"
        )
    if fused_range is not None and (
        len(fused_range) != 2
        or any(type(value) is not int for value in fused_range)
    ):
        raise TesseraFormatError(
            "resolved serving-lane fused range must be two JSON integers"
        )
    if any(not isinstance(value, str) for value in serve_flags):
        raise TesseraFormatError(
            "resolved serving-lane required flags must be strings"
        )
    copied_lane = replace(
        lane,
        fused_mid_m_rungs=fused_rungs,
        fused_mid_m_range=fused_range,
        requires_serve_flags=serve_flags,
    )
    return replace(gate, serving_lane=copied_lane)


@dataclass(frozen=True, slots=True)
class TesseraServabilityGate:
    """Existing target-profile decisions plus the Trellis capability floor."""

    target_profile: str
    profile_emulation_only: bool
    target_platform: str | None
    format_decision: ServingFormatDecision
    shape_decision: ServingFormatDecision
    capability_legal: bool
    capability_detail: str
    serving_lane: ResolvedServingLane | None

    @property
    def legal(self) -> bool:
        return (
            self.format_decision.legal
            and self.shape_decision.legal
            and self.capability_legal
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_profile": self.target_profile,
            "profile_emulation_only": self.profile_emulation_only,
            "target_platform": self.target_platform,
            "legal_for_research_allocator": self.legal,
            "format": _decision_payload(self.format_decision),
            "shape": _decision_payload(self.shape_decision),
            "capability": {
                "legal": self.capability_legal,
                "detail": self.capability_detail,
            },
            "serving_lane": (
                self.serving_lane.as_dict()
                if self.serving_lane is not None
                else None
            ),
            # A legal emulation/profile decision is not producer promotion.
            "producer_eligible": False,
        }


@dataclass(frozen=True, slots=True)
class TesseraAllocatorCandidate:
    """One measured, exactly priced pre-render recipe for one allocator unit."""

    unit_name: str
    family: str
    body_rate_q256: int
    layout: str
    variant_label: str | None
    footprint: Mapping[str, object]
    predicted_dloss_mean: float
    predicted_dloss_stderr: float
    servability: TesseraServabilityGate

    def __post_init__(self) -> None:
        if not isinstance(self.footprint, Mapping):
            raise TesseraFormatError("candidate footprint must be a mapping")
        validated_footprint = validate_tessera_tensor_payload_breakdown(
            self.footprint
        )
        frozen = _deep_freeze(validated_footprint)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "footprint", frozen)
        object.__setattr__(
            self,
            "servability",
            _canonical_servability_gate(self.servability),
        )
        if str(self.footprint.get("family")) != self.family:
            raise TesseraFormatError("candidate family differs from its footprint")
        if int(self.footprint.get("body_rate_q256", -1)) != self.body_rate_q256:
            raise TesseraFormatError("candidate q256 rate differs from its footprint")
        if str(self.footprint.get("layout")) != self.layout:
            raise TesseraFormatError("candidate layout differs from its footprint")
        recipe_identity = str(
            self.footprint.get("pre_render_recipe_identity_sha256", "")
        )
        if _SHA256.fullmatch(recipe_identity) is None:
            raise TesseraFormatError(
                "candidate pre-render recipe identity must be lowercase SHA-256"
            )
        object.__setattr__(
            self,
            "predicted_dloss_mean",
            _finite_float(
                self.predicted_dloss_mean,
                field="predicted_dloss_mean",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "predicted_dloss_stderr",
            _finite_float(
                self.predicted_dloss_stderr,
                field="predicted_dloss_stderr",
                nonnegative=True,
            ),
        )

    @property
    def predicted_dloss_objective(self) -> float:
        """The measured/shrunk point estimate, with no internal UCB penalty."""

        return self.predicted_dloss_mean

    @property
    def allocator_key(self) -> str:
        return (
            f"__TESSERA_RESEARCH__:{self.family}:"
            f"q256={self.body_rate_q256}:layout={self.layout}:"
            f"recipe={self.pre_render_recipe_identity_sha256}"
        )

    @property
    def memory_bytes(self) -> int:
        return int(self.footprint["total_bytes"])

    @property
    def bits_per_param(self) -> float:
        return float(self.footprint["exact_bpw"])

    @property
    def pre_render_recipe_identity_sha256(self) -> str:
        return str(self.footprint["pre_render_recipe_identity_sha256"])

    @property
    def shape(self) -> tuple[int, int]:
        raw = self.footprint["shape"]
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise TesseraFormatError("candidate footprint has an invalid shape")
        return int(raw[0]), int(raw[1])

    @property
    def n_params(self) -> int:
        rows, columns = self.shape
        return rows * columns

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": TESSERA_ALLOCATOR_CANDIDATE_SCHEMA,
            "unit_name": self.unit_name,
            "allocator_key": self.allocator_key,
            "family": self.family,
            "body_rate_q256": self.body_rate_q256,
            "layout": self.layout,
            "variant_label": self.variant_label,
            "footprint": _deep_thaw(self.footprint),
            "memory_bytes": self.memory_bytes,
            "bits_per_param": self.bits_per_param,
            "pre_render_recipe_identity_sha256": (
                self.pre_render_recipe_identity_sha256
            ),
            "rendered_wire_identity_sha256": None,
            "predicted_dloss_mean": self.predicted_dloss_mean,
            "predicted_dloss_stderr": self.predicted_dloss_stderr,
            "predicted_dloss_objective": self.predicted_dloss_objective,
            "objective_source": (
                "measured_or_shrunk_point_estimate_no_internal_ucb_penalty"
            ),
            "servability": self.servability.as_dict(),
            "research_only": True,
            "producer_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])

    def to_solver_candidate(self) -> Candidate:
        """Convert to the unchanged allocator candidate contract.

        The target-profile gate is fail-closed here.  Callers may retain the
        full record to report a denied experiment point, but cannot feed it to
        the solver accidentally. Exact candidate bytes do not make the default
        solver's quantized budget state an exact-byte certificate.
        """

        if not self.servability.legal:
            raise TesseraFormatError(
                f"{self.unit_name}/{self.allocator_key} is denied by target "
                f"profile {self.servability.target_profile!r}"
            )
        return Candidate(
            fmt=self.allocator_key,
            bits_per_param=self.bits_per_param,
            memory_bytes=self.memory_bytes,
            predicted_dloss=self.predicted_dloss_objective,
            serialized_identity=self.pre_render_recipe_identity_sha256,
            # Trellis schedule/scale/alphabet bytes are tensor-local wire
            # planes, not a separately deduplicated physical codebook.
            serialized_sidecar_identity=None,
            serving_lane=self.servability.serving_lane,
        )


def _capability_gate(
    spec: TesseraFamily,
    target_platform: str | None,
) -> tuple[bool, str]:
    if target_platform is None:
        return True, (
            "profile declares no exact hardware platform; capability remains "
            "an experiment admission responsibility"
        )
    match = _SM_PLATFORM.fullmatch(target_platform)
    if match is None:
        return False, (
            f"cannot resolve exact SM capability from target platform "
            f"{target_platform!r}"
        )
    observed = int(match.group(1))
    legal = observed >= spec.minimum_capability_sm
    return legal, (
        f"target {target_platform} resolves SM{observed}; {spec.family} "
        f"requires SM{spec.minimum_capability_sm}+"
    )


def build_tessera_allocator_candidate(
    unit_name: str,
    shape: Sequence[int],
    *,
    family: str | TesseraFamily,
    body_rate_q256: int,
    layout: str,
    schedule: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    predicted_dloss: float,
    predicted_dloss_stderr: float = 0.0,
    target_profile: str | None = "research",
    qname: str | None = None,
    packed_expert: bool | None = None,
    sidecar_header_bytes: int = 0,
    variant_label: str | None = None,
) -> TesseraAllocatorCandidate:
    """Validate and price one pre-render recipe, reporting uncertainty separately."""

    if not isinstance(unit_name, str) or not unit_name:
        raise TesseraFormatError("unit_name must be a nonempty string")
    if variant_label is not None and (
        not isinstance(variant_label, str)
        or _VARIANT_LABEL.fullmatch(variant_label) is None
    ):
        raise TesseraFormatError(
            "variant_label must match [A-Za-z0-9_.-]+"
        )
    spec = get_tessera_family(family)
    footprint = tessera_tensor_payload_breakdown(
        shape,
        family=spec,
        body_rate_q256=body_rate_q256,
        layout=layout,
        schedule=schedule,
        alphabets=alphabets,
        sidecar_header_bytes=sidecar_header_bytes,
    )
    mean = _finite_float(
        predicted_dloss,
        field="predicted_dloss",
        nonnegative=True,
    )
    stderr = _finite_float(
        predicted_dloss_stderr,
        field="predicted_dloss_stderr",
        nonnegative=True,
    )
    profile_id = str(target_profile or "research")
    wire_format = str(footprint["format"])
    format_decision = check_serving_format(
        profile_id,
        qname,
        wire_format,
        packed_expert=packed_expert,
    )
    dims = tuple(shape)
    shape_decision = check_serving_shape(
        profile_id,
        wire_format,
        qname=qname,
        in_features=int(dims[1]),
        out_features=int(dims[0]),
    )
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        profile_emulation_only = False
        target_platform = None
    else:
        profile_emulation_only = profile.emulation_only
        target_platform = profile.target_platform
    capability_legal, capability_detail = _capability_gate(
        spec, target_platform,
    )
    gate = TesseraServabilityGate(
        target_profile=profile_id,
        profile_emulation_only=profile_emulation_only,
        target_platform=target_platform,
        format_decision=format_decision,
        shape_decision=shape_decision,
        capability_legal=capability_legal,
        capability_detail=capability_detail,
        serving_lane=serving_lane_route(profile_id, wire_format),
    )
    return TesseraAllocatorCandidate(
        unit_name=unit_name,
        family=spec.family,
        body_rate_q256=int(footprint["body_rate_q256"]),
        layout=layout,
        variant_label=variant_label,
        footprint=footprint,
        predicted_dloss_mean=mean,
        predicted_dloss_stderr=stderr,
        servability=gate,
    )


def _candidate_order_key(
    record: TesseraAllocatorCandidate,
) -> tuple[object, ...]:
    """Total deterministic ordering, including the complete record identity."""

    return (
        record.memory_bytes,
        record.predicted_dloss_objective,
        record.predicted_dloss_mean,
        record.allocator_key,
        record.pre_render_recipe_identity_sha256,
        record.identity_sha256,
    )


def _validate_unit_measurements(
    records: Sequence[TesseraAllocatorCandidate],
) -> tuple[TesseraAllocatorCandidate, ...]:
    """Validate unit-local domains and unique pre-render recipe addresses."""

    materialized = tuple(records)
    grouped: dict[str, list[TesseraAllocatorCandidate]] = {}
    for record in materialized:
        if not isinstance(record, TesseraAllocatorCandidate):
            raise TypeError("records must contain TesseraAllocatorCandidate")
        grouped.setdefault(record.unit_name, []).append(record)
    for unit_name, unit_records in grouped.items():
        shapes = {record.shape for record in unit_records}
        n_params = {record.n_params for record in unit_records}
        if len(shapes) != 1 or len(n_params) != 1:
            raise TesseraFormatError(
                f"unit {unit_name!r} mixes tensor shapes/n_params; exact "
                "byte candidates must describe one physical tensor"
            )
        profiles = {
            record.servability.target_profile for record in unit_records
        }
        if len(profiles) != 1:
            raise TesseraFormatError(
                f"unit {unit_name!r} mixes target profiles; servability "
                "decisions are not one comparable allocator domain"
            )
        seen_addresses: dict[tuple[object, ...], str] = {}
        for record in unit_records:
            address = (
                record.family,
                record.body_rate_q256,
                record.layout,
                record.pre_render_recipe_identity_sha256,
            )
            previous = seen_addresses.get(address)
            if previous is not None:
                raise TesseraFormatError(
                    f"unit {unit_name!r} has duplicate Trellis pre-render "
                    f"recipe/address {record.pre_render_recipe_identity_sha256}; "
                    "aggregate repeated measurements upstream"
                )
            seen_addresses[address] = record.identity_sha256
    return materialized


def tessera_solver_candidate_menu(
    records: Sequence[TesseraAllocatorCandidate],
    *,
    fail_on_denied: bool = False,
) -> dict[str, list[Candidate]]:
    """Return deterministic menus for the unchanged global allocation DP.

    Denied records are normally retained only in the caller's audit payload and
    omitted here, matching ordinary candidate construction. ``fail_on_denied``
    is useful when an experiment declares that every requested point must be
    servable under its target profile.
    """

    materialized = _validate_unit_measurements(records)
    grouped: dict[str, list[TesseraAllocatorCandidate]] = {}
    for record in materialized:
        if not record.servability.legal:
            if fail_on_denied:
                record.to_solver_candidate()
            continue
        grouped.setdefault(record.unit_name, []).append(record)

    result: dict[str, list[Candidate]] = {}
    for unit_name in sorted(grouped):
        ordered = sorted(
            grouped[unit_name],
            key=_candidate_order_key,
        )
        keys = [record.allocator_key for record in ordered]
        if len(keys) != len(set(keys)):
            raise TesseraFormatError(
                f"unit {unit_name!r} has duplicate ephemeral allocator keys; "
                "pre-render recipe identities must be collision-free"
            )
        result[unit_name] = [record.to_solver_candidate() for record in ordered]
    return result


@dataclass(frozen=True, slots=True)
class TesseraMarginalSegment:
    """One exact-byte segment with exact rational marginal thresholds."""

    cheaper_identity_sha256: str
    dearer_identity_sha256: str
    cheaper_q256: int
    dearer_q256: int
    delta_bytes: int
    objective_loss_reduction: float
    mean_loss_reduction: float
    objective_marginal_loss_per_byte_rounded_binary64_diagnostic: float
    mean_marginal_loss_per_byte_rounded_binary64_diagnostic: float
    objective_marginal_loss_per_byte_exact: TesseraExactMarginal
    mean_marginal_loss_per_byte_exact: TesseraExactMarginal
    uncertainty_low_loss_per_byte: float
    uncertainty_high_loss_per_byte: float

    def __post_init__(self) -> None:
        if type(self.delta_bytes) is not int or self.delta_bytes <= 0:
            raise TesseraFormatError(
                "marginal segment delta_bytes must be positive"
            )
        for name in (
            "objective_loss_reduction",
            "mean_loss_reduction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise TesseraFormatError(
                    f"marginal segment {name} must be a positive finite float"
                )
        for name in (
            "objective_marginal_loss_per_byte_rounded_binary64_diagnostic",
            "mean_marginal_loss_per_byte_rounded_binary64_diagnostic",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TesseraFormatError(
                    f"marginal segment {name} diagnostic must be a finite "
                    "nonnegative float"
                )
        for exact_name, rounded_name in (
            (
                "objective_marginal_loss_per_byte_exact",
                "objective_marginal_loss_per_byte_rounded_binary64_diagnostic",
            ),
            (
                "mean_marginal_loss_per_byte_exact",
                "mean_marginal_loss_per_byte_rounded_binary64_diagnostic",
            ),
        ):
            exact = getattr(self, exact_name)
            if not isinstance(exact, TesseraExactMarginal):
                raise TesseraFormatError(
                    f"marginal segment {exact_name} must be exact rational data"
                )
            if exact.rounded_binary64_diagnostic != getattr(self, rounded_name):
                raise TesseraFormatError(
                    f"marginal segment {rounded_name} diagnostic drifted from "
                    "its exact rational value"
                )
        low = float(self.uncertainty_low_loss_per_byte)
        high = float(self.uncertainty_high_loss_per_byte)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise TesseraFormatError(
                "marginal segment uncertainty bounds must be finite and ordered"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "cheaper_identity_sha256": self.cheaper_identity_sha256,
            "dearer_identity_sha256": self.dearer_identity_sha256,
            "cheaper_q256": self.cheaper_q256,
            "dearer_q256": self.dearer_q256,
            "delta_bytes": self.delta_bytes,
            "objective_loss_reduction": self.objective_loss_reduction,
            "mean_loss_reduction": self.mean_loss_reduction,
            "objective_marginal_loss_per_byte_exact": (
                self.objective_marginal_loss_per_byte_exact.as_dict()
            ),
            "mean_marginal_loss_per_byte_exact": (
                self.mean_marginal_loss_per_byte_exact.as_dict()
            ),
            "objective_marginal_loss_per_byte_rounded_binary64_diagnostic": (
                self.objective_marginal_loss_per_byte_rounded_binary64_diagnostic
            ),
            "mean_marginal_loss_per_byte_rounded_binary64_diagnostic": (
                self.mean_marginal_loss_per_byte_rounded_binary64_diagnostic
            ),
            "uncertainty_low_loss_per_byte": (
                self.uncertainty_low_loss_per_byte
            ),
            "uncertainty_high_loss_per_byte": (
                self.uncertainty_high_loss_per_byte
            ),
        }


@dataclass(frozen=True, slots=True)
class TesseraParetoFrontier:
    unit_name: str
    candidates: tuple[TesseraAllocatorCandidate, ...]
    segments: tuple[TesseraMarginalSegment, ...]
    uncertainty_z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "uncertainty_z", _finite_float(
            self.uncertainty_z,
            field="uncertainty_z",
            nonnegative=True,
        ))

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": TESSERA_PARETO_FRONTIER_SCHEMA,
            "unit_name": self.unit_name,
            "uncertainty_z": self.uncertainty_z,
            "objective": (
                "measured_or_shrunk_predicted_dloss_no_internal_ucb_penalty"
            ),
            "byte_basis": "exact_side_information_inclusive_tensor_payload",
            "candidate_identity_sha256": [
                candidate.identity_sha256 for candidate in self.candidates
            ],
            "segments": [segment.as_dict() for segment in self.segments],
            "research_only": True,
            "producer_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


def tessera_pareto_frontier(
    records: Sequence[TesseraAllocatorCandidate],
    *,
    uncertainty_z: float = 1.96,
) -> TesseraParetoFrontier:
    """Return the exact-byte objective frontier and local marginal slopes."""

    z_value = _finite_float(
        uncertainty_z,
        field="uncertainty_z",
        nonnegative=True,
    )
    materialized = _validate_unit_measurements(records)
    units = {record.unit_name for record in materialized}
    if len(units) != 1:
        raise TesseraFormatError(
            "Pareto frontier records must describe exactly one allocator unit"
        )
    eligible = [
        record for record in materialized if record.servability.legal
    ]
    if not eligible:
        raise TesseraFormatError("Pareto frontier has no profile-legal records")
    ordered = sorted(eligible, key=_candidate_order_key)
    frontier: list[TesseraAllocatorCandidate] = []
    best_objective = math.inf
    for record in ordered:
        if record.predicted_dloss_objective < best_objective:
            frontier.append(record)
            best_objective = record.predicted_dloss_objective

    segments = [
        _marginal_segment(
            cheaper,
            dearer,
            uncertainty_z=z_value,
        )
        for cheaper, dearer in zip(frontier, frontier[1:])
    ]
    return TesseraParetoFrontier(
        unit_name=next(iter(units)),
        candidates=tuple(frontier),
        segments=tuple(segments),
        uncertainty_z=z_value,
    )


@dataclass(frozen=True, slots=True)
class TesseraLambdaChoice:
    """One O(log H) lookup result on a tensor's compressed RD hull."""

    candidate: TesseraAllocatorCandidate
    hull_index: int
    lambda_loss_per_byte: float
    comparisons: int
    cheaper_breakpoint_loss_per_byte_exact: TesseraExactMarginal | None
    dearer_breakpoint_loss_per_byte_exact: TesseraExactMarginal | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TesseraAllocatorCandidate):
            raise TesseraFormatError(
                "lambda choice candidate must be TesseraAllocatorCandidate"
            )
        if type(self.hull_index) is not int or self.hull_index < 0:
            raise TesseraFormatError(
                "lambda choice hull_index must be a nonnegative integer"
            )
        if type(self.comparisons) is not int or self.comparisons < 0:
            raise TesseraFormatError(
                "lambda choice comparisons must be a nonnegative integer"
            )
        object.__setattr__(
            self,
            "lambda_loss_per_byte",
            _finite_float(
                self.lambda_loss_per_byte,
                field="lambda_loss_per_byte",
                nonnegative=True,
            ),
        )
        for field_name in (
            "cheaper_breakpoint_loss_per_byte_exact",
            "dearer_breakpoint_loss_per_byte_exact",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, TesseraExactMarginal):
                raise TesseraFormatError(
                    f"lambda choice {field_name} must be exact rational data"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_identity_sha256": self.candidate.identity_sha256,
            "allocator_key": self.candidate.allocator_key,
            "hull_index": self.hull_index,
            "lambda_loss_per_byte": self.lambda_loss_per_byte,
            "lambda_binary64_exact_ratio": {
                "numerator_decimal": str(
                    self.lambda_loss_per_byte.as_integer_ratio()[0]
                ),
                "denominator_decimal": str(
                    self.lambda_loss_per_byte.as_integer_ratio()[1]
                ),
            },
            "comparisons": self.comparisons,
            "cheaper_breakpoint_loss_per_byte_exact": (
                self.cheaper_breakpoint_loss_per_byte_exact.as_dict()
                if self.cheaper_breakpoint_loss_per_byte_exact is not None
                else None
            ),
            "dearer_breakpoint_loss_per_byte_exact": (
                self.dearer_breakpoint_loss_per_byte_exact.as_dict()
                if self.dearer_breakpoint_loss_per_byte_exact is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TesseraRateDistortionHull:
    """Per-tensor lower convex RD hull for marginal-price selection."""

    unit_name: str
    candidates: tuple[TesseraAllocatorCandidate, ...]
    segments: tuple[TesseraMarginalSegment, ...]
    uncertainty_z: float
    pareto_frontier_identity_sha256: str
    exact_breakpoints_loss_per_byte: tuple[TesseraExactMarginal, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    breakpoints_loss_per_byte_rounded_binary64_diagnostic: tuple[
        float, ...
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "uncertainty_z", _finite_float(
            self.uncertainty_z,
            field="uncertainty_z",
            nonnegative=True,
        ))
        if not self.candidates:
            raise TesseraFormatError("RD hull must contain a candidate")
        if len(self.segments) != len(self.candidates) - 1:
            raise TesseraFormatError(
                "RD hull segment count must be one less than candidate count"
            )
        benefits = tuple(
            _exact_marginal_loss_per_byte(
                cheaper.predicted_dloss_objective,
                dearer.predicted_dloss_objective,
                dearer.memory_bytes - cheaper.memory_bytes,
                field="RD hull exact marginal loss per byte",
            )
            for cheaper, dearer in zip(
                self.candidates,
                self.candidates[1:],
            )
        )
        if any(
            left.compare(right) <= 0
            for left, right in zip(benefits, benefits[1:])
        ):
            raise TesseraFormatError(
                "RD hull exact marginal benefits must be strictly decreasing"
            )
        segment_benefits = tuple(
            segment.objective_marginal_loss_per_byte_exact
            for segment in self.segments
        )
        if segment_benefits != benefits:
            raise TesseraFormatError(
                "RD hull segment exact marginals differ from its candidates"
            )
        object.__setattr__(self, "exact_breakpoints_loss_per_byte", benefits)
        object.__setattr__(
            self,
            "breakpoints_loss_per_byte_rounded_binary64_diagnostic",
            tuple(value.rounded_binary64_diagnostic for value in benefits),
        )

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": "prismaquant.trellis_rd_hull.v1",
            "unit_name": self.unit_name,
            "candidate_identity_sha256": [
                candidate.identity_sha256 for candidate in self.candidates
            ],
            "segments": [segment.as_dict() for segment in self.segments],
            "uncertainty_z": self.uncertainty_z,
            "pareto_frontier_identity_sha256": (
                self.pareto_frontier_identity_sha256
            ),
            "cheapest_legal_floor": {
                "candidate_identity_sha256": self.candidates[0].identity_sha256,
                "memory_bytes": self.candidates[0].memory_bytes,
                "bits_per_param": self.candidates[0].bits_per_param,
            },
            "lookup_complexity": "O(log_hull_points)",
            "exact_breakpoints_precomputed": True,
            "exact_breakpoints_loss_per_byte": [
                value.as_dict()
                for value in self.exact_breakpoints_loss_per_byte
            ],
            "rounded_breakpoint_diagnostics_only": list(
                self.breakpoints_loss_per_byte_rounded_binary64_diagnostic
            ),
            "maximum_lambda_lookup_comparisons": (
                len(self.candidates) - 1
            ).bit_length(),
            "research_only": True,
            "producer_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])

    @property
    def cheapest_legal_floor(self) -> TesseraAllocatorCandidate:
        """Cheapest exact-byte candidate admitted by the measured profile."""

        return self.candidates[0]

    def choice_at_lambda(self, lambda_loss_per_byte: float) -> TesseraLambdaChoice:
        """Choose ``argmin(loss + lambda*bytes)`` by breakpoint bisection.

        Equality at a breakpoint chooses the cheaper candidate, making repeated
        calls and global byte monotonicity deterministic.
        """

        alpha = _finite_float(
            lambda_loss_per_byte,
            field="lambda_loss_per_byte",
            nonnegative=True,
        )
        benefits = self.exact_breakpoints_loss_per_byte
        alpha_numerator, alpha_denominator = alpha.as_integer_ratio()
        lo = 0
        hi = len(benefits)
        comparisons = 0
        # Benefits are strictly decreasing. Find the count of leading
        # breakpoints strictly greater than lambda; that count is the chosen
        # candidate index. Strictness gives the cheaper tie policy.
        while lo < hi:
            middle = (lo + hi) // 2
            comparisons += 1
            if benefits[middle].greater_than_binary64_ratio(
                alpha_numerator,
                alpha_denominator,
            ):
                lo = middle + 1
            else:
                hi = middle
        index = lo
        return TesseraLambdaChoice(
            candidate=self.candidates[index],
            hull_index=index,
            lambda_loss_per_byte=alpha,
            comparisons=comparisons,
            cheaper_breakpoint_loss_per_byte_exact=(
                benefits[index - 1] if index > 0 else None
            ),
            dearer_breakpoint_loss_per_byte_exact=(
                benefits[index] if index < len(benefits) else None
            ),
        )


def _marginal_segment(
    cheaper: TesseraAllocatorCandidate,
    dearer: TesseraAllocatorCandidate,
    *,
    uncertainty_z: float,
) -> TesseraMarginalSegment:
    delta_bytes = dearer.memory_bytes - cheaper.memory_bytes
    if delta_bytes <= 0:
        raise TesseraFormatError("RD segment bytes must increase strictly")
    objective_reduction = (
        cheaper.predicted_dloss_objective
        - dearer.predicted_dloss_objective
    )
    mean_reduction = (
        cheaper.predicted_dloss_mean - dearer.predicted_dloss_mean
    )
    objective_marginal_exact = _exact_marginal_loss_per_byte(
        cheaper.predicted_dloss_objective,
        dearer.predicted_dloss_objective,
        delta_bytes,
        field="objective marginal loss per byte",
    )
    mean_marginal_exact = _exact_marginal_loss_per_byte(
        cheaper.predicted_dloss_mean,
        dearer.predicted_dloss_mean,
        delta_bytes,
        field="mean marginal loss per byte",
    )
    uncertainty_low, uncertainty_high = _uncertainty_bounds_loss_per_byte(
        mean_reduction,
        cheaper.predicted_dloss_stderr,
        dearer.predicted_dloss_stderr,
        delta_bytes,
        uncertainty_z,
    )
    return TesseraMarginalSegment(
        cheaper_identity_sha256=cheaper.identity_sha256,
        dearer_identity_sha256=dearer.identity_sha256,
        cheaper_q256=cheaper.body_rate_q256,
        dearer_q256=dearer.body_rate_q256,
        delta_bytes=delta_bytes,
        objective_loss_reduction=objective_reduction,
        mean_loss_reduction=mean_reduction,
        objective_marginal_loss_per_byte_rounded_binary64_diagnostic=(
            objective_marginal_exact.rounded_binary64_diagnostic
        ),
        mean_marginal_loss_per_byte_rounded_binary64_diagnostic=(
            mean_marginal_exact.rounded_binary64_diagnostic
        ),
        objective_marginal_loss_per_byte_exact=objective_marginal_exact,
        mean_marginal_loss_per_byte_exact=mean_marginal_exact,
        uncertainty_low_loss_per_byte=uncertainty_low,
        uncertainty_high_loss_per_byte=uncertainty_high,
    )


def tessera_rate_distortion_hull(
    records: Sequence[TesseraAllocatorCandidate],
    *,
    uncertainty_z: float = 1.96,
) -> TesseraRateDistortionHull:
    """Compress one tensor's measured menu to its Lagrangian-supported hull."""

    frontier = tessera_pareto_frontier(
        records,
        uncertainty_z=uncertainty_z,
    )
    hull: list[TesseraAllocatorCandidate] = []
    for candidate in frontier.candidates:
        while len(hull) >= 2:
            earlier, previous = hull[-2], hull[-1]
            previous_benefit = _exact_marginal_loss_per_byte(
                earlier.predicted_dloss_objective,
                previous.predicted_dloss_objective,
                previous.memory_bytes - earlier.memory_bytes,
                field="RD hull previous marginal loss per byte",
            )
            candidate_benefit = _exact_marginal_loss_per_byte(
                previous.predicted_dloss_objective,
                candidate.predicted_dloss_objective,
                candidate.memory_bytes - previous.memory_bytes,
                field="RD hull candidate marginal loss per byte",
            )
            # Exact convex-hull orientation.  Both binary64 losses are first
            # converted to their exact dyadic ratios; slope order is decided
            # only by integer cross-products, never rounded division.
            if previous_benefit.compare(candidate_benefit) > 0:
                break
            hull.pop()
        hull.append(candidate)
    segments = tuple(
        _marginal_segment(
            cheaper,
            dearer,
            uncertainty_z=frontier.uncertainty_z,
        )
        for cheaper, dearer in zip(hull, hull[1:])
    )
    return TesseraRateDistortionHull(
        unit_name=frontier.unit_name,
        candidates=tuple(hull),
        segments=segments,
        uncertainty_z=frontier.uncertainty_z,
        pareto_frontier_identity_sha256=frontier.identity_sha256,
    )


def trellis_lambda_choices(
    hulls: Mapping[str, TesseraRateDistortionHull],
    lambda_loss_per_byte: float,
) -> dict[str, TesseraLambdaChoice]:
    """Select all tensors for one price in O(U log U + sum(log H_u)).

    The per-hull breakpoints are precomputed; sorting the ``U`` mapping keys is
    the separate cost required for deterministic multi-unit output ordering.
    """

    result: dict[str, TesseraLambdaChoice] = {}
    for unit_name in sorted(hulls):
        hull = hulls[unit_name]
        if hull.unit_name != unit_name:
            raise TesseraFormatError(
                f"RD hull key {unit_name!r} differs from hull unit "
                f"{hull.unit_name!r}"
            )
        result[unit_name] = hull.choice_at_lambda(lambda_loss_per_byte)
    return result


def trellis_local_repair_solver_menu(
    frontier: TesseraParetoFrontier,
    choice: TesseraLambdaChoice,
    *,
    neighbor_count: int = 1,
    complete_pareto: bool = False,
    include_ci_overlap: bool = True,
    ci_uncertainty_z: float | None = None,
) -> dict[str, list[Candidate]]:
    """Build an uncertified candidate menu around one global-lambda choice.

    The frontier retains Pareto points that no scalar lambda supports, which
    later integer-budget repair may still need. ``complete_pareto=True``
    returns all measured Pareto points. Otherwise the
    selected point plus ``neighbor_count`` points on each exact-byte side form
    an explicit local window. By default, every point whose loss-plus-lambda-
    bytes confidence interval overlaps the selected point is unioned into that
    window.

    This function only constructs a menu. ``allocator_solver`` may quantize
    its internal budget state and does not provide an exact-byte feasibility or
    global-optimality certificate. Any downstream assignment must therefore be
    filtered against the exact integer ``memory_bytes`` sum and accompanied by
    a method-appropriate feasibility/optimality certificate before it is
    treated as a solved budget. This research bridge does not implement that
    later solver or certificate.
    """

    if type(neighbor_count) is not int or neighbor_count < 0:
        raise TesseraFormatError("neighbor_count must be a nonnegative integer")
    if choice.candidate.unit_name != frontier.unit_name:
        raise TesseraFormatError("lambda choice and Pareto frontier units differ")
    identities = [
        candidate.identity_sha256 for candidate in frontier.candidates
    ]
    try:
        selected = identities.index(choice.candidate.identity_sha256)
    except ValueError as exc:
        raise TesseraFormatError(
            "lambda choice is absent from the supplied Pareto frontier"
        ) from exc
    if complete_pareto:
        records = frontier.candidates
    else:
        start = max(0, selected - neighbor_count)
        stop = min(len(frontier.candidates), selected + neighbor_count + 1)
        selected_records = list(frontier.candidates[start:stop])
        if include_ci_overlap:
            z_value = (
                frontier.uncertainty_z
                if ci_uncertainty_z is None
                else _finite_float(
                    ci_uncertainty_z,
                    field="ci_uncertainty_z",
                    nonnegative=True,
                )
            )
            chosen = choice.candidate
            chosen_radius = z_value * chosen.predicted_dloss_stderr
            if not math.isfinite(chosen_radius):
                raise TesseraFormatError(
                    "derived CI overlap z * chosen stderr is not finite"
                )
            for candidate in frontier.candidates:
                candidate_radius = (
                    z_value * candidate.predicted_dloss_stderr
                )
                if not math.isfinite(candidate_radius):
                    raise TesseraFormatError(
                        "derived CI overlap z * candidate stderr is not finite"
                    )
                radius_sum = chosen_radius + candidate_radius
                if not math.isfinite(radius_sum):
                    raise TesseraFormatError(
                        "derived CI overlap uncertainty-radius sum is not finite"
                    )
                loss_delta = (
                    candidate.predicted_dloss_mean
                    - chosen.predicted_dloss_mean
                )
                if not math.isfinite(loss_delta):
                    raise TesseraFormatError(
                        "derived CI overlap centered loss delta is not finite"
                    )
                byte_delta = candidate.memory_bytes - chosen.memory_bytes
                try:
                    priced_byte_delta = (
                        choice.lambda_loss_per_byte * byte_delta
                    )
                except OverflowError as exc:
                    raise TesseraFormatError(
                        "derived CI overlap lambda * delta_bytes is not finite"
                    ) from exc
                if not math.isfinite(priced_byte_delta):
                    raise TesseraFormatError(
                        "derived CI overlap lambda * delta_bytes is not finite"
                    )
                centered_score_delta = loss_delta + priced_byte_delta
                if not math.isfinite(centered_score_delta):
                    raise TesseraFormatError(
                        "derived CI overlap centered score sum is not finite"
                    )
                if abs(centered_score_delta) <= radius_sum:
                    selected_records.append(candidate)
        by_identity = {
            record.identity_sha256: record for record in selected_records
        }
        records = tuple(
            sorted(
                by_identity.values(),
                key=_candidate_order_key,
            )
        )
    return tessera_solver_candidate_menu(records, fail_on_denied=True)


@dataclass(frozen=True, slots=True)
class TrellisAdaptiveRateProposal:
    surface: TesseraRateSurface
    alpha_loss_per_byte: float | None
    ranked_brackets: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.surface, TesseraRateSurface):
            raise TesseraFormatError(
                "adaptive proposal surface must be TesseraRateSurface"
            )
        if self.alpha_loss_per_byte is not None:
            object.__setattr__(self, "alpha_loss_per_byte", _finite_float(
                self.alpha_loss_per_byte,
                field="alpha_loss_per_byte",
                nonnegative=True,
            ))
        frozen_rows: list[Mapping[str, object]] = []
        for row in self.ranked_brackets:
            if not isinstance(row, Mapping):
                raise TesseraFormatError(
                    "adaptive ranked_brackets entries must be mappings"
                )
            frozen = _deep_freeze(row)
            assert isinstance(frozen, Mapping)
            frozen_rows.append(frozen)
        object.__setattr__(self, "ranked_brackets", tuple(frozen_rows))

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": TESSERA_ADAPTIVE_RATE_PROPOSAL_SCHEMA,
            "surface": self.surface.as_dict(),
            "alpha_loss_per_byte": self.alpha_loss_per_byte,
            "ranked_brackets": [
                _deep_thaw(row) for row in self.ranked_brackets
            ],
            "research_only": True,
            "producer_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


def adaptive_trellis_rate_surface(
    family: str | TesseraFamily,
    records: Sequence[TesseraAllocatorCandidate],
    *,
    alpha_loss_per_byte: float | None = None,
    max_new_points: int = 1,
    minimum_q256_gap: int = 2,
    uncertainty_z: float = 1.96,
) -> TrellisAdaptiveRateProposal:
    """Bisect deterministic Pareto brackets, prioritizing a marginal alpha.

    With ``alpha_loss_per_byte`` set, a segment whose uncertainty interval
    contains alpha ranks first, followed by interval distance, exact rational
    point-estimate distance, and the wider q256 gap. Without alpha, wider
    unresolved q256 intervals rank first, then wider slope uncertainty.
    Observed interior points subdivide their hull segment before bisection, so
    an off-hull measured midpoint cannot stall progress. Existing and proposed rates are
    returned together as a content-addressed adaptive surface.
    """

    spec = get_tessera_family(family)
    if type(max_new_points) is not int or max_new_points < 0:
        raise TesseraFormatError("max_new_points must be a nonnegative integer")
    if type(minimum_q256_gap) is not int or minimum_q256_gap < 2:
        raise TesseraFormatError("minimum_q256_gap must be an integer >= 2")
    alpha = (
        None
        if alpha_loss_per_byte is None
        else _finite_float(
            alpha_loss_per_byte,
            field="alpha_loss_per_byte",
            nonnegative=True,
        )
    )
    materialized = _validate_unit_measurements(records)
    if any(record.family != spec.family for record in materialized):
        raise TesseraFormatError(
            "adaptive surface records must all belong to the requested family"
        )
    layouts = {record.layout for record in materialized}
    if len(layouts) != 1:
        raise TesseraFormatError(
            "adaptive surface records must use one Trellis layout; q256 "
            "brackets cannot mix fixed-quota and tight-offset curves"
        )
    eligible = [
        record for record in materialized if record.servability.legal
    ]
    hull = tessera_rate_distortion_hull(
        eligible,
        uncertainty_z=uncertainty_z,
    )
    by_identity = {
        record.identity_sha256: record for record in hull.candidates
    }
    observed_rates = {record.body_rate_q256 for record in eligible}
    observed_rates_sorted = tuple(sorted(observed_rates))
    hull_rates = tuple(
        record.body_rate_q256 for record in hull.candidates
    )
    if any(left >= right for left, right in zip(hull_rates, hull_rates[1:])):
        raise TesseraFormatError(
            "adaptive surface exact-byte hull must increase strictly in q256; "
            "use a homogeneous rate curve or refine recipes separately"
        )
    pending: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for segment in hull.segments:
        cheaper = by_identity[segment.cheaper_identity_sha256]
        dearer = by_identity[segment.dearer_identity_sha256]
        q_lo = min(cheaper.body_rate_q256, dearer.body_rate_q256)
        q_hi = max(cheaper.body_rate_q256, dearer.body_rate_q256)
        low = segment.uncertainty_low_loss_per_byte
        high = segment.uncertainty_high_loss_per_byte
        width = high - low
        if not math.isfinite(width):
            raise TesseraFormatError(
                "derived adaptive marginal uncertainty width is not finite"
            )
        if alpha is None:
            bracketed = None
            interval_distance = None
            point_distance_exact = None
        else:
            bracketed = low <= alpha <= high
            interval_distance = (
                0.0
                if bracketed
                else (low - alpha if alpha < low else alpha - high)
            )
            alpha_numerator, alpha_denominator = alpha.as_integer_ratio()
            point_distance_exact = (
                segment.objective_marginal_loss_per_byte_exact
                .distance_from_binary64_ratio(
                    alpha_numerator,
                    alpha_denominator,
                )
            )
            if (
                not math.isfinite(interval_distance)
                or not math.isfinite(
                    point_distance_exact.rounded_binary64_diagnostic
                )
            ):
                raise TesseraFormatError(
                    "derived adaptive alpha distance is not finite"
                )
        q_lo_index = bisect_left(observed_rates_sorted, q_lo)
        q_hi_index = bisect_left(observed_rates_sorted, q_hi)
        interior = observed_rates_sorted[
            bisect_right(observed_rates_sorted, q_lo):q_hi_index
        ]
        common: dict[str, object] = {
            **segment.as_dict(),
            "parent_hull_q256_interval": [q_lo, q_hi],
            # Inclusive indices into surface.anchor_q256 bind the complete
            # observed parent interval once, without copying its O(P) interior
            # rate list into every one of O(P) unresolved subinterval rows.
            "parent_anchor_index_interval_inclusive": [
                q_lo_index,
                q_hi_index,
            ],
            "alpha_bracketed_by_uncertainty": bracketed,
            "alpha_interval_distance": interval_distance,
            "alpha_point_distance_exact": (
                point_distance_exact.as_dict()
                if point_distance_exact is not None
                else None
            ),
        }
        boundaries = (
            q_lo,
            *interior,
            q_hi,
        )
        for sub_lo, sub_hi in zip(boundaries, boundaries[1:]):
            q_gap = sub_hi - sub_lo
            if q_gap < minimum_q256_gap:
                continue
            midpoint = (sub_lo + sub_hi) // 2
            if alpha is None:
                priority: tuple[object, ...] = (
                    -q_gap,
                    -width,
                    q_lo,
                    q_hi,
                    sub_lo,
                    sub_hi,
                    segment.cheaper_identity_sha256,
                    segment.dearer_identity_sha256,
                )
            else:
                priority = (
                    not bracketed,
                    interval_distance,
                    point_distance_exact,
                    -q_gap,
                    q_lo,
                    q_hi,
                    sub_lo,
                    sub_hi,
                    segment.cheaper_identity_sha256,
                    segment.dearer_identity_sha256,
                )
            row = {
                **common,
                "q256_interval": [sub_lo, sub_hi],
                "q256_gap": q_gap,
                "proposed_q256": midpoint,
                "subdivision_depth": 0,
                "bracket_endpoints_observed": [True, True],
            }
            pending.append((priority, row))

    heapq.heapify(pending)
    proposed: list[int] = []
    used = set(observed_rates)
    ranked_rows: list[dict[str, object]] = []
    while pending and len(proposed) < max_new_points:
        priority, row = heapq.heappop(pending)
        midpoint = int(row["proposed_q256"])
        if midpoint in used:
            raise AssertionError("adaptive subdivision proposed a used q256")
        ranked_rows.append({**row, "selected_for_refinement": True})
        proposed.append(midpoint)
        used.add(midpoint)

        # When the explicit cap permits another point, keep bisecting this
        # unresolved gap.  This makes one call productive beyond the initial
        # number of hull segments while retaining O(measured + cap) state.
        if len(proposed) < max_new_points:
            sub_lo, sub_hi = (
                int(value) for value in row["q256_interval"]
            )
            depth = int(row["subdivision_depth"]) + 1
            for child_lo, child_hi in (
                (sub_lo, midpoint),
                (midpoint, sub_hi),
            ):
                child_gap = child_hi - child_lo
                if child_gap < minimum_q256_gap:
                    continue
                child_midpoint = (child_lo + child_hi) // 2
                child_priority = (
                    *priority[:3],
                    -child_gap,
                    priority[4],
                    priority[5],
                    child_lo,
                    child_hi,
                    *priority[-2:],
                ) if alpha is not None else (
                    -child_gap,
                    *priority[1:4],
                    child_lo,
                    child_hi,
                    *priority[-2:],
                )
                child = {
                    **row,
                    "q256_interval": [child_lo, child_hi],
                    "q256_gap": child_gap,
                    "proposed_q256": child_midpoint,
                    "subdivision_depth": depth,
                    "bracket_endpoints_observed": [
                        child_lo in observed_rates,
                        child_hi in observed_rates,
                    ],
                }
                heapq.heappush(pending, (child_priority, child))
    ranked_rows.extend(
        {**row, "selected_for_refinement": False}
        for _priority, row in sorted(pending, key=lambda item: item[0])
    )
    proposed_tuple = tuple(sorted(proposed))
    surface = TesseraRateSurface(
        family=spec.family,
        mode=RATE_SURFACE_ADAPTIVE,
        bounds_q256=spec.mathematical_q256_bounds,
        anchor_q256=tuple(sorted(observed_rates)),
        proposed_q256=proposed_tuple,
        source_identity_sha256=hull.identity_sha256,
    )
    return TrellisAdaptiveRateProposal(
        surface=surface,
        alpha_loss_per_byte=alpha,
        ranked_brackets=tuple(ranked_rows),
    )


__all__ = [
    "TESSERA_ADAPTIVE_RATE_PROPOSAL_SCHEMA",
    "TESSERA_ALLOCATOR_CANDIDATE_SCHEMA",
    "TESSERA_PARETO_FRONTIER_SCHEMA",
    "TrellisAdaptiveRateProposal",
    "TesseraAllocatorCandidate",
    "TesseraExactMarginal",
    "TesseraLambdaChoice",
    "TesseraMarginalSegment",
    "TesseraParetoFrontier",
    "TesseraRateDistortionHull",
    "TesseraServabilityGate",
    "adaptive_trellis_rate_surface",
    "build_tessera_allocator_candidate",
    "tessera_pareto_frontier",
    "trellis_lambda_choices",
    "trellis_local_repair_solver_menu",
    "tessera_rate_distortion_hull",
    "tessera_solver_candidate_menu",
]
