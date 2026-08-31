"""Continuous trellis rate surface: dense rungs from sparse measured anchors.

RESEARCH ONLY.  Nothing here is imported by ``run-pipeline.sh``, the default
format menu, or any export path.

The problem this closes
-----------------------
``trellis_formats`` addresses 761 integer E2M1 rungs and 1785 E4M3 rungs, and
``trellis_allocator`` can price and rank any of them exactly.  What was
missing is the step between: a rung has no cost until something *encodes* it,
and encoding every rung on every Linear is not affordable.  So the addressable
surface was continuous while the *measured* surface was five points.

``adaptive_trellis_rate_surface`` is the complement of this module, not a
substitute: it proposes **which rung to measure next** by bisecting hull
brackets around a target marginal price.  It never predicts an unmeasured
rung.  Campaign planning here predicts a dense surface so the next real
measurement can be proposed.  The stricter allocation-regret path is only a
retrospective validation: it requires measured truth at every candidate rung
and therefore saves no allocator-use encodes.

What it does NOT assume
-----------------------
No parametric rate law.  A global log-linear fit is wrong on this family and
the ladder shows exactly where: E4M3 wSNR runs 11.04/16.79/22.12/27.35/29.57
dB at rates 2-6, i.e. ~5.4 dB per bit until it saturates against the scalar
E4M3 ceiling, where the last step delivers 2.22.  A straight line through
those anchors overpredicts the top of the range badly.  The allocator-capable
fit is therefore piecewise-linear gain in dB over a matched measured scalar
subgrid.  Raw ``log2(dloss)`` interpolation survives only as an explicit
campaign-planning fallback.  Both operate between bracketing anchors only;
extrapolation and nonmonotone measurements are refused.  A prediction is
sealed before an excluded holdout is measured, and :func:`allocation_regret`
validates only the exact densified menu it graded with its internal greedy
marginal allocator.  That result does not transfer to another allocator.

Gate on the decision, not the residual
--------------------------------------
The one-anchor campaign established this the expensive way: per-superblock
D(R2)->D(R3) transfer has a 0.32 log2 residual (~25% relative D) and yet
costs 0.04-0.23% allocation regret, because noise only misranks blocks that
were near-indifferent anyway.  A residual-style bar would have refused a
near-omniscient decision.  :func:`leave_one_anchor_out` is therefore a
diagnostic and the immutable, menu-bound result of :func:`allocation_regret`
is a retrospective validation record, not a population or consumer-allocator
license.

Currency
--------
``predicted_dloss`` means whatever objective measured the anchors, and a DP
that mixes objectives is meaningless.  A surface therefore carries a declared
``currency`` string, and :func:`rate_surface_solver_menu` refuses to build a
menu from surfaces that disagree about it.  Note for anyone wiring the
trellis ladder in directly: the ladder's weighted SSE uses a per-input-channel
activation second moment, which is an output-MSE proxy, **not** the AURA
KL-adjoint objective the production DP prices in.  Those are different
currencies and must not be mixed.

Provenance
----------
Every emitted candidate is stamped ``interpolated`` unless its rung exactly
matches a measured anchor, in which case it is stamped ``measured`` and
carries the anchor's own value.  An interpolated rung is a *proposal*: the
render encodes it for real and the held-out KL gate judges it, which is the
house rule (surrogates generate, real KL selects).  Nothing here may be
reported as a measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
import re
from typing import Callable, Iterator, Mapping, Sequence

from .trellis_allocator import (
    TrellisAllocatorCandidate,
    build_trellis_allocator_candidate,
    trellis_solver_candidate_menu,
)
from .trellis_formats import (
    LAYOUT_TIGHT_OFFSETS,
    MIN_TRELLIS_STEPS,
    SUPERBLOCK_WEIGHTS,
    TrellisFamily,
    TrellisFormatError,
    get_trellis_family,
    validate_body_rate_q256,
)

__all__ = [
    "FIT_MATCHED_SCALAR_GAIN_DB",
    "FIT_RAW_DLOSS_LOG2",
    "PROVENANCE_INTERPOLATED",
    "PROVENANCE_MEASURED",
    "SURFACE_USE_ALLOCATOR_COST",
    "SURFACE_USE_CAMPAIGN_PLANNING",
    "TrellisAllocationRegretGate",
    "TrellisCurveIdentity",
    "TrellisDensifiedSurface",
    "TrellisHoldoutGrade",
    "TrellisHoldoutSeal",
    "TrellisMeasuredRateAnchor",
    "TrellisRateSurface",
    "TrellisScalarBackbonePoint",
    "allocation_regret",
    "densify_rate_surface",
    "fit_rate_surface",
    "grade_rate_surface_holdout",
    "leave_one_anchor_out",
    "rate_surface_solver_menu",
    "seal_rate_surface_holdout",
    "uniform_column_schedule",
]


PROVENANCE_MEASURED = "measured"
PROVENANCE_INTERPOLATED = "interpolated"
SURFACE_USE_ALLOCATOR_COST = "allocator_cost"
SURFACE_USE_CAMPAIGN_PLANNING = "campaign_planning"
FIT_MATCHED_SCALAR_GAIN_DB = "matched_scalar_gain_db"
FIT_RAW_DLOSS_LOG2 = "raw_dloss_log2"

_INTENDED_USES = frozenset(
    {SURFACE_USE_ALLOCATOR_COST, SURFACE_USE_CAMPAIGN_PLANNING}
)
_RENDER_PATHS = frozenset({"rtn", "gptq", "ldlq", "rotated"})
_POPULATIONS = frozenset({"dense", "routed"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_SCHEMA = "prismaquant.trellis_curve_identity.v1"
_SURFACE_SCHEMA = "prismaquant.interpolated_trellis_rate_surface.v2"
_BACKBONE_SCHEMA = "prismaquant.trellis_scalar_backbone_point.v1"
_ANCHOR_SCHEMA = "prismaquant.trellis_measured_rate_anchor.v1"
_HOLDOUT_SEAL_SCHEMA = "prismaquant.trellis_rate_holdout_seal.v1"
_HOLDOUT_GRADE_SCHEMA = "prismaquant.trellis_rate_holdout_grade.v1"
_REGRET_GATE_SCHEMA = "prismaquant.trellis_allocation_regret_gate.v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TrellisFormatError(f"{field} must be a nonempty NUL-free string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrellisFormatError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TrellisFormatError(f"{field} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrellisFormatError(f"{field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TrellisFormatError(f"{field} must be positive and finite")
    return result


def _nonnegative_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TrellisFormatError(f"{field} must be nonnegative and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrellisFormatError(f"{field} must be nonnegative and finite") from exc
    if not math.isfinite(result) or result < 0.0:
        raise TrellisFormatError(f"{field} must be nonnegative and finite")
    return result


@dataclass(frozen=True, slots=True)
class TrellisCurveIdentity:
    """The complete frozen domain in which interpolation is meaningful."""

    wire_family: str
    selector: str
    alphabet_policy: str
    alphabet_fitting_scope: str
    scale_plane: str
    scale_coding: str
    encode_tier: str
    schedule_policy: str
    render_path: str
    corpus_id: str
    importance_id: str
    population: str
    render_recipe_identity_sha256: str
    codec_closure_sha256: str

    def __post_init__(self) -> None:
        spec = get_trellis_family(self.wire_family)
        if self.wire_family != spec.family:
            raise TrellisFormatError("curve wire_family must be canonical")
        for field in (
            "selector",
            "alphabet_policy",
            "alphabet_fitting_scope",
            "scale_plane",
            "scale_coding",
            "encode_tier",
            "schedule_policy",
            "render_path",
            "corpus_id",
            "importance_id",
            "population",
        ):
            object.__setattr__(
                self,
                field,
                _nonempty_text(getattr(self, field), field=f"curve_identity.{field}"),
            )
        if self.scale_plane != spec.scale_contract:
            raise TrellisFormatError(
                "curve scale_plane differs from the wire family's scale contract"
            )
        if self.render_path not in _RENDER_PATHS:
            raise TrellisFormatError(
                f"curve render_path must be one of {sorted(_RENDER_PATHS)}"
            )
        if self.population not in _POPULATIONS:
            raise TrellisFormatError(
                f"curve population must be one of {sorted(_POPULATIONS)}"
            )
        object.__setattr__(
            self,
            "render_recipe_identity_sha256",
            _sha256(
                self.render_recipe_identity_sha256,
                field="curve_identity.render_recipe_identity_sha256",
            ),
        )
        object.__setattr__(
            self,
            "codec_closure_sha256",
            _sha256(
                self.codec_closure_sha256,
                field="curve_identity.codec_closure_sha256",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": _IDENTITY_SCHEMA,
            "wire_family": self.wire_family,
            "selector": self.selector,
            "alphabet_policy": self.alphabet_policy,
            "alphabet_fitting_scope": self.alphabet_fitting_scope,
            "scale_plane": self.scale_plane,
            "scale_coding": self.scale_coding,
            "encode_tier": self.encode_tier,
            "schedule_policy": self.schedule_policy,
            "render_path": self.render_path,
            "corpus_id": self.corpus_id,
            "importance_id": self.importance_id,
            "population": self.population,
            "render_recipe_identity_sha256": (
                self.render_recipe_identity_sha256
            ),
            "codec_closure_sha256": self.codec_closure_sha256,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


@dataclass(frozen=True, slots=True)
class TrellisScalarBackbonePoint:
    """One measured scalar control under the exact curve context."""

    curve_identity: TrellisCurveIdentity
    body_rate_q256: int
    dloss: float
    currency: str
    backbone_closure_sha256: str
    context_parity_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.curve_identity, TrellisCurveIdentity):
            raise TrellisFormatError("scalar backbone needs a curve identity")
        validate_body_rate_q256(
            get_trellis_family(self.curve_identity.wire_family),
            self.body_rate_q256,
        )
        object.__setattr__(
            self, "dloss", _positive_finite(self.dloss, field="backbone.dloss")
        )
        object.__setattr__(
            self,
            "currency",
            _nonempty_text(self.currency, field="backbone.currency"),
        )
        object.__setattr__(
            self,
            "backbone_closure_sha256",
            _sha256(
                self.backbone_closure_sha256,
                field="backbone.backbone_closure_sha256",
            ),
        )
        if type(self.context_parity_verified) is not bool:
            raise TrellisFormatError(
                "backbone.context_parity_verified must be a boolean"
            )

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": _BACKBONE_SCHEMA,
            "curve_identity_sha256": self.curve_identity.identity_sha256,
            "body_rate_q256": self.body_rate_q256,
            "dloss": self.dloss,
            "currency": self.currency,
            "backbone_closure_sha256": self.backbone_closure_sha256,
            "context_parity_verified": self.context_parity_verified,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


@dataclass(frozen=True, slots=True)
class TrellisMeasuredRateAnchor:
    """A measured candidate sealed to its curve and optional scalar control."""

    candidate: TrellisAllocatorCandidate
    curve_identity: TrellisCurveIdentity
    currency: str
    scalar_backbone: TrellisScalarBackbonePoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TrellisAllocatorCandidate):
            raise TrellisFormatError("measured anchor needs an allocator candidate")
        if not isinstance(self.curve_identity, TrellisCurveIdentity):
            raise TrellisFormatError("measured anchor needs a curve identity")
        object.__setattr__(
            self,
            "currency",
            _nonempty_text(self.currency, field="measured_anchor.currency"),
        )
        if self.candidate.family != self.curve_identity.wire_family:
            raise TrellisFormatError(
                "measured anchor wire family differs from its curve identity"
            )
        if self.candidate.variant_label != PROVENANCE_MEASURED:
            raise TrellisFormatError(
                "a measured anchor candidate must be explicitly labelled measured"
            )
        if self.candidate.footprint.get("scale_contract") != self.curve_identity.scale_plane:
            raise TrellisFormatError(
                "measured anchor scale plane differs from its curve identity"
            )
        if self.scalar_backbone is not None:
            if not isinstance(self.scalar_backbone, TrellisScalarBackbonePoint):
                raise TrellisFormatError("anchor scalar_backbone has invalid type")
            if self.scalar_backbone.curve_identity != self.curve_identity:
                raise TrellisFormatError(
                    "anchor and scalar backbone curve identities differ"
                )
            if self.scalar_backbone.currency != self.currency:
                raise TrellisFormatError(
                    "anchor and scalar backbone currencies differ"
                )
            if self.scalar_backbone.body_rate_q256 != self.candidate.body_rate_q256:
                raise TrellisFormatError(
                    "anchor and scalar backbone q256 rates differ"
                )

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": _ANCHOR_SCHEMA,
            "candidate_identity_sha256": self.candidate.identity_sha256,
            "curve_identity_sha256": self.curve_identity.identity_sha256,
            "currency": self.currency,
            "body_rate_q256": self.candidate.body_rate_q256,
            "measured_dloss": self.candidate.predicted_dloss_mean,
            "measured_stderr": self.candidate.predicted_dloss_stderr,
            "scalar_backbone": (
                self.scalar_backbone.as_dict()
                if self.scalar_backbone is not None
                else None
            ),
        }
        return {**body, "anchor_sha256": _canonical_sha256(body)}

    @property
    def anchor_sha256(self) -> str:
        return str(self.as_dict()["anchor_sha256"])


def uniform_column_schedule(
    columns: int,
    body_rate_q256: int,
    *,
    family: str | TrellisFamily,
) -> tuple[int, ...]:
    """Return the flattest legal per-input-column schedule hitting a rate.

    ``gridbook.trellis.wire.v1`` carries one rate code per input column,
    shared across every output row (``schedule_scope ==
    "tensor_input_column_shared_across_rows"``), and the ``tight_offsets``
    layout explicitly permits variable block totals so long as the tensor-wide
    total lands within one physical body bit of the declared q256.  That is
    what makes a continuous rate surface expressible at all: the achievable
    totals are the integers, so the rate resolution is ``256/columns`` q256
    per step -- 0.25 q256 on a 1024-column Linear.

    Flattest-legal is the neutral default: rates differ by at most one across
    columns, ordered deterministically.  A caller holding per-column
    importance should shape the schedule itself; this function deliberately
    does not invent a ranking it cannot justify.
    """

    spec = get_trellis_family(family)
    rate = validate_body_rate_q256(spec, body_rate_q256)
    if type(columns) is not int or columns <= 0:
        raise TrellisFormatError("columns must be a positive integer")
    if columns % SUPERBLOCK_WEIGHTS:
        raise TrellisFormatError(
            f"columns must be a multiple of {SUPERBLOCK_WEIGHTS}; a short "
            f"final block is legal on the wire but its rate accounting is "
            f"the caller's to declare, not this helper's to guess"
        )

    total_bits = round(rate * columns / SUPERBLOCK_WEIGHTS)
    base, remainder = divmod(total_bits, columns)
    if base < 1 or (base + (1 if remainder else 0)) > spec.bypass_rate:
        raise TrellisFormatError(
            f"body rate {rate} q256 needs per-column rates outside "
            f"[1, {spec.bypass_rate}] for {columns} columns"
        )
    # SPREAD the remainder across columns rather than packing it at the
    # front.  Contiguous placement concentrates the dearer columns into whole
    # superblocks, and on E2M1 (bypass_rate 4) a block made entirely of
    # bypass columns has ZERO coded steps and the wire refuses it -- at rates
    # the research range genuinely uses, not in some unused corner.  A
    # Bresenham distribution gives every block its proportional share and is
    # deterministic.
    schedule = [
        base + 1
        if (column * remainder) // columns
        != ((column + 1) * remainder) // columns
        else base
        for column in range(columns)
    ]

    # Every block needs MIN_TRELLIS_STEPS genuinely coded (non-bypass)
    # positions.  Flattest-legal only risks this at the very top of the
    # range, above the research ceiling, but the check is cheap and a silent
    # violation would surface as an unexplained wire refusal much later.
    for start in range(0, columns, SUPERBLOCK_WEIGHTS):
        block = schedule[start:start + SUPERBLOCK_WEIGHTS]
        coded = sum(value < spec.bypass_rate for value in block)
        if coded < MIN_TRELLIS_STEPS:
            raise TrellisFormatError(
                f"body rate {rate} q256 leaves {coded} coded steps in block "
                f"at column {start}; {MIN_TRELLIS_STEPS} are required"
            )
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class TrellisRateSurface:
    """One identity-bound curve fitted only between measured anchors."""

    unit_name: str
    family: str
    layout: str
    shape: tuple[int, int]
    target_profile: str
    currency: str
    curve_identity: TrellisCurveIdentity
    fit_response: str
    anchor_q256: tuple[int, ...]
    anchor_dloss: tuple[float, ...]
    anchor_stderr: tuple[float, ...]
    anchor_sha256: tuple[str, ...]
    scalar_backbone_closure_sha256: str | None = None
    anchor_backbone_dloss: tuple[float, ...] = ()
    anchor_backbone_point_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "anchor_q256",
            "anchor_dloss",
            "anchor_stderr",
            "anchor_sha256",
            "anchor_backbone_dloss",
            "anchor_backbone_point_sha256",
        ):
            raw = getattr(self, field)
            if isinstance(raw, (str, bytes, bytearray, Mapping)):
                raise TrellisFormatError(f"surface {field} must be a sequence")
            try:
                object.__setattr__(self, field, tuple(raw))
            except TypeError as exc:
                raise TrellisFormatError(
                    f"surface {field} must be a sequence"
                ) from exc
        if not isinstance(self.curve_identity, TrellisCurveIdentity):
            raise TrellisFormatError("surface needs a complete curve identity")
        if self.family != self.curve_identity.wire_family:
            raise TrellisFormatError(
                "surface family differs from its frozen curve identity"
            )
        _nonempty_text(self.unit_name, field="surface.unit_name")
        _nonempty_text(self.layout, field="surface.layout")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise TrellisFormatError("surface shape must be a sequence") from exc
        if (
            len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise TrellisFormatError("surface shape must be two positive integers")
        object.__setattr__(self, "shape", shape)
        _nonempty_text(self.target_profile, field="surface.target_profile")
        _nonempty_text(self.currency, field="surface.currency")
        if self.fit_response not in {
            FIT_MATCHED_SCALAR_GAIN_DB,
            FIT_RAW_DLOSS_LOG2,
        }:
            raise TrellisFormatError("surface fit_response is unsupported")
        if len(self.anchor_q256) < 2:
            raise TrellisFormatError(
                "a rate surface needs at least two measured anchors; one "
                "anchor is a point, not a surface"
            )
        if not (
            len(self.anchor_q256)
            == len(self.anchor_dloss)
            == len(self.anchor_stderr)
            == len(self.anchor_sha256)
        ):
            raise TrellisFormatError("anchor arrays must agree in length")
        spec = get_trellis_family(self.family)
        for rate in self.anchor_q256:
            validate_body_rate_q256(spec, rate)
        for left, right in zip(self.anchor_q256, self.anchor_q256[1:]):
            if left >= right:
                raise TrellisFormatError(
                    "anchor rates must be strictly increasing"
                )
        object.__setattr__(
            self,
            "anchor_dloss",
            tuple(
                _positive_finite(value, field="anchor dloss")
                for value in self.anchor_dloss
            ),
        )
        object.__setattr__(
            self,
            "anchor_stderr",
            tuple(
                _nonnegative_finite(value, field="anchor stderr")
                for value in self.anchor_stderr
            ),
        )
        object.__setattr__(
            self,
            "anchor_sha256",
            tuple(
                _sha256(value, field="anchor_sha256")
                for value in self.anchor_sha256
            ),
        )
        if len(set(self.anchor_sha256)) != len(self.anchor_sha256):
            raise TrellisFormatError("surface repeats a measured anchor identity")
        for left, right in zip(self.anchor_dloss, self.anchor_dloss[1:]):
            if right >= left:
                raise TrellisFormatError(
                    "anchor dloss must strictly decrease as rate rises; a "
                    "non-monotone anchor set is a measurement problem, and "
                    "interpolating through it would launder it into a cost"
                )
        if self.fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
            if self.scalar_backbone_closure_sha256 is None:
                raise TrellisFormatError(
                    "gain-fit surfaces need a scalar backbone closure digest"
                )
            object.__setattr__(
                self,
                "scalar_backbone_closure_sha256",
                _sha256(
                    self.scalar_backbone_closure_sha256,
                    field="surface.scalar_backbone_closure_sha256",
                ),
            )
            if not (
                len(self.anchor_backbone_dloss)
                == len(self.anchor_backbone_point_sha256)
                == len(self.anchor_q256)
            ):
                raise TrellisFormatError(
                    "gain-fit surfaces need one matched scalar backbone per anchor"
                )
            object.__setattr__(
                self,
                "anchor_backbone_dloss",
                tuple(
                    _positive_finite(value, field="anchor backbone dloss")
                    for value in self.anchor_backbone_dloss
                ),
            )
            object.__setattr__(
                self,
                "anchor_backbone_point_sha256",
                tuple(
                    _sha256(value, field="anchor backbone point identity")
                    for value in self.anchor_backbone_point_sha256
                ),
            )
            for left, right in zip(
                self.anchor_backbone_dloss,
                self.anchor_backbone_dloss[1:],
            ):
                if right >= left:
                    raise TrellisFormatError(
                        "matched scalar-backbone anchor loss must strictly "
                        "decrease as q256 rises"
                    )
        elif (
            self.scalar_backbone_closure_sha256 is not None
            or self.anchor_backbone_dloss
            or self.anchor_backbone_point_sha256
        ):
            raise TrellisFormatError(
                "raw-dloss surfaces must not claim a matched scalar backbone"
            )

    @property
    def q256_range(self) -> tuple[int, int]:
        return (self.anchor_q256[0], self.anchor_q256[-1])

    @property
    def licensed_uses(self) -> tuple[str, ...]:
        """Closed compatibility vocabulary, not a statistical use license."""

        if self.fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
            return (
                SURFACE_USE_ALLOCATOR_COST,
                SURFACE_USE_CAMPAIGN_PLANNING,
            )
        return (SURFACE_USE_CAMPAIGN_PLANNING,)

    def authorize_use(self, intended_use: str) -> None:
        if intended_use not in _INTENDED_USES:
            raise TrellisFormatError(
                "interpolated rate surfaces are available only for allocator-cost "
                "and campaign-planning use; menu verdict, family verdict, and "
                "publication use are refused"
            )
        if intended_use not in self.licensed_uses:
            raise TrellisFormatError(
                "raw-dloss interpolation is planning-only; allocator-cost use "
                "requires a matched scalar backbone with verified context parity"
            )

    def _bracket(self, body_rate_q256: int) -> tuple[int, int, float]:
        if type(body_rate_q256) is not int:
            raise TrellisFormatError("body_rate_q256 must be an integer")
        validate_body_rate_q256(get_trellis_family(self.family), body_rate_q256)
        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TrellisFormatError(
                f"rung {body_rate_q256} is outside the measured envelope "
                f"[{low}, {high}]; extrapolating a trellis rate surface is "
                f"refused -- measure another anchor instead"
            )
        upper = next(
            index
            for index, rate in enumerate(self.anchor_q256)
            if rate >= body_rate_q256
        )
        if self.anchor_q256[upper] == body_rate_q256:
            return upper, upper, 0.0
        lower = upper - 1
        span = self.anchor_q256[upper] - self.anchor_q256[lower]
        weight = (body_rate_q256 - self.anchor_q256[lower]) / span
        return lower, upper, weight

    def _predict(
        self,
        body_rate_q256: int,
        *,
        scalar_backbone: TrellisScalarBackbonePoint | None,
    ) -> float:
        lower, upper, weight = self._bracket(body_rate_q256)
        if self.fit_response == FIT_RAW_DLOSS_LOG2:
            if scalar_backbone is not None:
                raise TrellisFormatError(
                    "raw-dloss prediction must not claim an unused scalar backbone"
                )
            if lower == upper:
                return self.anchor_dloss[lower]
            log_low = math.log2(self.anchor_dloss[lower])
            log_high = math.log2(self.anchor_dloss[upper])
            predicted = float(2.0 ** (log_low + weight * (log_high - log_low)))
        else:
            if scalar_backbone is None:
                raise TrellisFormatError(
                    "gain interpolation requires a matched scalar backbone point"
                )
            if not isinstance(scalar_backbone, TrellisScalarBackbonePoint):
                raise TrellisFormatError("scalar_backbone has invalid type")
            if scalar_backbone.curve_identity != self.curve_identity:
                raise TrellisFormatError(
                    "scalar backbone curve identity differs from the surface"
                )
            if scalar_backbone.currency != self.currency:
                raise TrellisFormatError(
                    "scalar backbone currency differs from the fitted surface"
                )
            if scalar_backbone.body_rate_q256 != body_rate_q256:
                raise TrellisFormatError(
                    "scalar backbone q256 differs from the predicted rung"
                )
            if not scalar_backbone.context_parity_verified:
                raise TrellisFormatError(
                    "scalar backbone context parity is not verified"
                )
            if (
                scalar_backbone.backbone_closure_sha256
                != self.scalar_backbone_closure_sha256
            ):
                raise TrellisFormatError(
                    "scalar backbone closure differs from the fitted subgrid"
                )
            if lower == upper:
                if (
                    scalar_backbone.identity_sha256
                    != self.anchor_backbone_point_sha256[lower]
                ):
                    raise TrellisFormatError(
                        "anchor prediction used a different measured scalar point"
                    )
                return self.anchor_dloss[lower]
            predicted = self._predict_from_backbone_dloss(
                lower,
                upper,
                weight,
                scalar_backbone.dloss,
            )
        if not math.isfinite(predicted) or predicted <= 0.0:
            raise TrellisFormatError("interpolation produced invalid dloss")
        if not self.anchor_dloss[upper] < predicted < self.anchor_dloss[lower]:
            raise TrellisFormatError(
                "interpolated dloss is non-monotone inside its measured bracket; "
                "measure the rung instead"
            )
        return float(predicted)

    def _predict_from_backbone_dloss(
        self,
        lower: int,
        upper: int,
        weight: float,
        backbone_dloss: float,
    ) -> float:
        """Internal gain interpolation after provenance/context validation."""

        gain_low = 10.0 * math.log10(
            self.anchor_backbone_dloss[lower] / self.anchor_dloss[lower]
        )
        gain_high = 10.0 * math.log10(
            self.anchor_backbone_dloss[upper] / self.anchor_dloss[upper]
        )
        gain = gain_low + weight * (gain_high - gain_low)
        return float(backbone_dloss / (10.0 ** (gain / 10.0)))

    def predict(
        self,
        body_rate_q256: int,
        *,
        intended_use: str,
        scalar_backbone: TrellisScalarBackbonePoint | None = None,
    ) -> float:
        """Predict one rung under an explicit closed-vocabulary intended use."""

        self.authorize_use(intended_use)
        return self._predict(body_rate_q256, scalar_backbone=scalar_backbone)

    def predict_stderr(self, body_rate_q256: int, *, intended_use: str) -> float:
        """Anchor stderr at an anchor; the wider bracketing one between them.

        Deliberately conservative: an interpolated rung inherits the larger
        of the two anchors it sits between, so uncertainty never shrinks by
        the act of interpolating.
        """

        self.authorize_use(intended_use)
        lower, upper, _ = self._bracket(body_rate_q256)
        if lower == upper:
            return self.anchor_stderr[lower]
        return max(self.anchor_stderr[upper - 1], self.anchor_stderr[upper])

    def provenance(self, body_rate_q256: int) -> str:
        self._bracket(body_rate_q256)
        return (
            PROVENANCE_MEASURED
            if body_rate_q256 in self.anchor_q256
            else PROVENANCE_INTERPOLATED
        )

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": _SURFACE_SCHEMA,
            "unit_name": self.unit_name,
            "family": self.family,
            "layout": self.layout,
            "shape": list(self.shape),
            "target_profile": self.target_profile,
            "currency": self.currency,
            "curve_identity": self.curve_identity.as_dict(),
            "fit_response": self.fit_response,
            "anchor_q256": list(self.anchor_q256),
            "anchor_dloss": list(self.anchor_dloss),
            "anchor_stderr": list(self.anchor_stderr),
            "anchor_sha256": list(self.anchor_sha256),
            "scalar_backbone_closure_sha256": (
                self.scalar_backbone_closure_sha256
            ),
            "anchor_backbone_dloss": list(self.anchor_backbone_dloss),
            "anchor_backbone_point_sha256": list(
                self.anchor_backbone_point_sha256
            ),
            "licensed_uses": list(self.licensed_uses),
            "fit_axis": "body_rate_q256",
            "research_only": True,
            "producer_eligible": False,
            "menu_verdict_eligible": False,
            "family_verdict_eligible": False,
            "publication_eligible": False,
        }
        return {**body, "surface_sha256": _canonical_sha256(body)}

    @property
    def surface_sha256(self) -> str:
        return str(self.as_dict()["surface_sha256"])


def fit_rate_surface(
    records: Sequence[TrellisMeasuredRateAnchor],
    *,
    currency: str,
    curve_identity: TrellisCurveIdentity,
    fit_response: str,
) -> TrellisRateSurface:
    """Build an identity-bound surface from measured anchors only.

    The scalar-backbone gain response is the allocator-capable path.  Raw
    dloss remains explicit and planning-only.  Even that fallback requires
    identity-bound measured records: accepting a bare allocator candidate and
    attaching a caller-supplied identity here would invent provenance.  QTIP,
    LDLQ, and rotated identities therefore cannot borrow an unbound or
    zero-anchor curve either.
    """

    if not isinstance(currency, str) or not currency:
        raise TrellisFormatError(
            "currency must be a nonempty string naming the objective the "
            "anchors were measured in"
        )
    if not isinstance(curve_identity, TrellisCurveIdentity):
        raise TrellisFormatError("fit requires a complete curve_identity")
    if fit_response not in {FIT_MATCHED_SCALAR_GAIN_DB, FIT_RAW_DLOSS_LOG2}:
        raise TrellisFormatError("fit_response is unsupported")
    materialized = list(records)
    if not materialized:
        raise TrellisFormatError("a rate surface needs anchor records")
    normalized: list[
        tuple[
            TrellisAllocatorCandidate,
            str,
            TrellisScalarBackbonePoint | None,
        ]
    ] = []
    for raw in materialized:
        if isinstance(raw, TrellisMeasuredRateAnchor):
            if raw.curve_identity != curve_identity:
                raise TrellisFormatError(
                    "measured anchor curve identity drifted from the fit"
                )
            if raw.currency != currency:
                raise TrellisFormatError(
                    "measured anchor currency drifted from the fit"
                )
            normalized.append(
                (raw.candidate, raw.anchor_sha256, raw.scalar_backbone)
            )
        else:
            raise TrellisFormatError(
                "fit records must be identity-bound measured anchors; bare "
                "allocator candidates cannot substantiate a curve identity"
            )
    if fit_response == FIT_RAW_DLOSS_LOG2 and any(
        backbone is not None for _, _, backbone in normalized
    ):
        raise TrellisFormatError(
            "matched scalar backbones were supplied; fit their gain instead of "
            "discarding the context-normalizing control"
        )
    units = {record.unit_name for record, _, _ in normalized}
    families = {record.family for record, _, _ in normalized}
    layouts = {record.layout for record, _, _ in normalized}
    shapes = {
        tuple(record.footprint.get("shape", ())) for record, _, _ in normalized
    }
    target_profiles = {
        record.servability.target_profile for record, _, _ in normalized
    }
    if len(units) != 1:
        raise TrellisFormatError("anchors must describe exactly one unit")
    if len(families) != 1:
        raise TrellisFormatError("anchors must share one trellis family")
    if families != {curve_identity.wire_family}:
        raise TrellisFormatError(
            "anchor wire family differs from the frozen curve identity"
        )
    if len(layouts) != 1:
        raise TrellisFormatError(
            "anchors must share one layout; fixed-quota and tight-offset "
            "curves are different rate axes and cannot be interpolated across"
        )
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise TrellisFormatError(
            "anchors must share one exact tensor shape; shape drift changes "
            "the schedule and exact-byte curve"
        )
    if len(target_profiles) != 1:
        raise TrellisFormatError(
            "anchors must share one exact target profile"
        )
    for record, _, _ in normalized:
        if record.footprint.get("scale_contract") != curve_identity.scale_plane:
            raise TrellisFormatError(
                "anchor scale plane differs from the frozen curve identity"
            )
    ordered = sorted(normalized, key=lambda row: row[0].body_rate_q256)
    rates = tuple(record.body_rate_q256 for record, _, _ in ordered)
    if len(set(rates)) != len(rates):
        raise TrellisFormatError("anchors must not repeat a rate")
    backbone_dloss: tuple[float, ...] = ()
    backbone_sha256: tuple[str, ...] = ()
    if fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
        controls: list[TrellisScalarBackbonePoint] = []
        for record, _, backbone in ordered:
            if backbone is None:
                raise TrellisFormatError(
                    "gain fitting refuses a missing scalar backbone"
                )
            if not backbone.context_parity_verified:
                raise TrellisFormatError(
                    "gain fitting requires verified scalar-backbone context parity"
                )
            if backbone.curve_identity != curve_identity:
                raise TrellisFormatError(
                    "scalar backbone curve identity drifted from the fit"
                )
            if backbone.body_rate_q256 != record.body_rate_q256:
                raise TrellisFormatError(
                    "scalar backbone and anchor rates differ"
                )
            controls.append(backbone)
        if len({control.backbone_closure_sha256 for control in controls}) != 1:
            raise TrellisFormatError(
                "gain fitting scalar anchors use different backbone closures"
            )
        backbone_dloss = tuple(control.dloss for control in controls)
        backbone_sha256 = tuple(control.identity_sha256 for control in controls)
    return TrellisRateSurface(
        unit_name=next(iter(units)),
        family=next(iter(families)),
        layout=next(iter(layouts)),
        shape=next(iter(shapes)),
        target_profile=next(iter(target_profiles)),
        currency=currency,
        curve_identity=curve_identity,
        fit_response=fit_response,
        anchor_q256=rates,
        anchor_dloss=tuple(
            float(record.predicted_dloss_mean) for record, _, _ in ordered
        ),
        anchor_stderr=tuple(
            float(record.predicted_dloss_stderr) for record, _, _ in ordered
        ),
        anchor_sha256=tuple(anchor_sha for _, anchor_sha, _ in ordered),
        scalar_backbone_closure_sha256=(
            controls[0].backbone_closure_sha256
            if fit_response == FIT_MATCHED_SCALAR_GAIN_DB
            else None
        ),
        anchor_backbone_dloss=backbone_dloss,
        anchor_backbone_point_sha256=backbone_sha256,
    )


@dataclass(frozen=True, slots=True)
class TrellisDensifiedSurface(Sequence[TrellisAllocatorCandidate]):
    """Exactly priced candidates bound back to the surface that predicted them."""

    surface: TrellisRateSurface
    intended_use: str
    candidates: tuple[TrellisAllocatorCandidate, ...]
    scalar_backbone_point_sha256: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.surface, TrellisRateSurface):
            raise TrellisFormatError("densified surface needs its source surface")
        self.surface.authorize_use(self.intended_use)
        if isinstance(self.candidates, (str, bytes, bytearray, Mapping)):
            raise TrellisFormatError("densified candidates must be a sequence")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates:
            raise TrellisFormatError(
                "a densified surface must contain at least one requested rung"
            )
        try:
            backbone_bindings = tuple(sorted(
                (rate, digest)
                for rate, digest in self.scalar_backbone_point_sha256
            ))
        except (TypeError, ValueError) as exc:
            raise TrellisFormatError(
                "densified scalar-backbone bindings must be q256/digest pairs"
            ) from exc
        object.__setattr__(
            self, "scalar_backbone_point_sha256", backbone_bindings
        )
        for candidate in self.candidates:
            if not isinstance(candidate, TrellisAllocatorCandidate):
                raise TrellisFormatError(
                    "densified surface contains a non-Trellis candidate"
                )
            if (
                candidate.unit_name != self.surface.unit_name
                or candidate.family != self.surface.family
                or candidate.layout != self.surface.layout
                or candidate.servability.target_profile
                != self.surface.target_profile
            ):
                raise TrellisFormatError(
                    "densified candidate drifted from its source surface"
                )
            if tuple(candidate.footprint.get("shape", ())) != self.surface.shape:
                raise TrellisFormatError(
                    "densified candidate shape drifted from its source surface"
                )
            if (
                candidate.footprint.get("scale_contract")
                != self.surface.curve_identity.scale_plane
            ):
                raise TrellisFormatError(
                    "densified candidate scale plane drifted from its curve"
                )
            self.surface._bracket(candidate.body_rate_q256)
            if candidate.variant_label != self.surface.provenance(
                candidate.body_rate_q256
            ):
                raise TrellisFormatError(
                    "densified candidate provenance disagrees with the surface"
                )
        rates = [candidate.body_rate_q256 for candidate in self.candidates]
        if len(set(rates)) != len(rates):
            raise TrellisFormatError("densified surface repeats a q256 rung")
        ordered_candidates = sorted(
            self.candidates, key=lambda candidate: candidate.body_rate_q256
        )
        for left, right in zip(ordered_candidates, ordered_candidates[1:]):
            if right.predicted_dloss_mean >= left.predicted_dloss_mean:
                raise TrellisFormatError(
                    "densified predicted loss is non-monotone; measure and "
                    "re-anchor instead of allocating the interpolant"
                )
        binding_rates = []
        for rate, digest in self.scalar_backbone_point_sha256:
            if type(rate) is not int:
                raise TrellisFormatError(
                    "densified scalar-backbone q256 must be an integer"
                )
            _sha256(digest, field="densified scalar-backbone identity")
            binding_rates.append(rate)
        if len(set(binding_rates)) != len(binding_rates):
            raise TrellisFormatError(
                "densified surface repeats a scalar-backbone binding"
            )
        if self.surface.fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
            if set(binding_rates) != set(rates):
                raise TrellisFormatError(
                    "gain densification must bind one scalar point per rung"
                )
        elif binding_rates:
            raise TrellisFormatError(
                "raw-dloss densification must not bind scalar-backbone points"
            )

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[TrellisAllocatorCandidate]:
        return iter(self.candidates)

    def __getitem__(
        self, index: int | slice
    ) -> TrellisAllocatorCandidate | tuple[TrellisAllocatorCandidate, ...]:
        return self.candidates[index]

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": "prismaquant.densified_trellis_rate_surface.v1",
            "surface_sha256": self.surface.surface_sha256,
            "curve_identity_sha256": self.surface.curve_identity.identity_sha256,
            "intended_use": self.intended_use,
            "candidate_identity_sha256": [
                candidate.identity_sha256 for candidate in self.candidates
            ],
            "scalar_backbone_point_sha256_by_q256": {
                str(rate): digest
                for rate, digest in self.scalar_backbone_point_sha256
            },
            "pricing_axis": "body_rate_q256",
            "byte_pricing": "exact_serialized_tensor_payload_bytes",
            "research_only": True,
            "producer_eligible": False,
            "menu_verdict_eligible": False,
            "family_verdict_eligible": False,
            "publication_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


def densify_rate_surface(
    surface: TrellisRateSurface,
    shape: Sequence[int],
    *,
    q256_values: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    intended_use: str,
    schedule_policy: str,
    scalar_backbone: Mapping[int, TrellisScalarBackbonePoint] | None = None,
    schedule_for: Callable[[int, int], Sequence[int]] | None = None,
    target_profile: str | None = "research",
    qname: str | None = None,
    packed_expert: bool | None = None,
    sidecar_header_bytes: int = 0,
) -> TrellisDensifiedSurface:
    """Price a dense set of rungs from one interpolated surface.

    Each rung gets a real schedule and therefore an exact byte footprint --
    the rate is interpolated, the *bytes* never are.  ``schedule_for`` may
    override the flattest-legal default for callers holding per-column
    importance; it receives ``(columns, body_rate_q256)`` and must return a
    schedule the wire accepts.
    """

    if not isinstance(surface, TrellisRateSurface):
        raise TrellisFormatError("surface must be a TrellisRateSurface")
    surface.authorize_use(intended_use)
    if schedule_policy != surface.curve_identity.schedule_policy:
        raise TrellisFormatError(
            "densification schedule policy differs from the frozen curve identity"
        )
    if schedule_for is None and schedule_policy != "uniform_bresenham":
        raise TrellisFormatError(
            "a non-uniform schedule policy requires its explicit schedule builder"
        )
    if schedule_for is not None and schedule_policy == "uniform_bresenham":
        raise TrellisFormatError(
            "uniform_bresenham must use the canonical built-in schedule builder"
        )
    dims = tuple(shape)
    if len(dims) != 2:
        raise TrellisFormatError("shape must be two dimensions")
    if dims != surface.shape:
        raise TrellisFormatError(
            "densification shape differs from the measured surface shape"
        )
    if target_profile != surface.target_profile:
        raise TrellisFormatError(
            "densification target profile differs from the measured surface"
        )
    columns = dims[1]
    spec = get_trellis_family(surface.family)
    raw_rates = tuple(q256_values)
    if any(type(value) is not int for value in raw_rates):
        raise TrellisFormatError("q256_values must contain JSON integers")
    if not raw_rates:
        raise TrellisFormatError("q256_values must contain at least one rung")
    if len(set(raw_rates)) != len(raw_rates):
        raise TrellisFormatError(
            "q256_values must not repeat a requested rung"
        )
    requested_rates = set(raw_rates)
    if surface.fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
        if scalar_backbone is None or set(scalar_backbone) != requested_rates:
            raise TrellisFormatError(
                "gain densification requires exactly one scalar point per rung"
            )
    elif scalar_backbone is not None:
        raise TrellisFormatError(
            "raw-dloss densification must not receive a scalar backbone"
        )
    built: list[TrellisAllocatorCandidate] = []
    for rate in sorted(set(raw_rates)):
        schedule = (
            schedule_for(columns, rate)
            if schedule_for is not None
            else uniform_column_schedule(
                columns, rate, family=surface.family,
            )
        )
        # Each rung uses only the rates its own schedule contains, and
        # `validate_alphabets` requires the mapping to match EXACTLY -- a
        # superset is refused.  So the caller supplies every alphabet it
        # holds and the rung selects the ones it actually spends.
        used = {rate for rate in schedule if rate < spec.bypass_rate}
        missing = used - set(alphabets)
        if missing:
            raise TrellisFormatError(
                f"rung {rate} needs alphabets for rates {sorted(missing)}; "
                f"got {sorted(alphabets)}"
            )
        built.append(
            build_trellis_allocator_candidate(
                surface.unit_name,
                dims,
                family=surface.family,
                body_rate_q256=rate,
                layout=surface.layout,
                schedule=schedule,
                alphabets={key: alphabets[key] for key in sorted(used)},
                predicted_dloss=surface.predict(
                    rate,
                    intended_use=intended_use,
                    scalar_backbone=(
                        scalar_backbone.get(rate)
                        if scalar_backbone is not None
                        else None
                    ),
                ),
                predicted_dloss_stderr=surface.predict_stderr(
                    rate, intended_use=intended_use
                ),
                target_profile=target_profile,
                qname=qname,
                packed_expert=packed_expert,
                sidecar_header_bytes=sidecar_header_bytes,
                variant_label=surface.provenance(rate),
            )
        )
    return TrellisDensifiedSurface(
        surface=surface,
        intended_use=intended_use,
        candidates=tuple(built),
        scalar_backbone_point_sha256=(
            tuple(
                (rate, scalar_backbone[rate].identity_sha256)
                for rate in sorted(requested_rates)
            )
            if scalar_backbone is not None
            else ()
        ),
    )


@dataclass(frozen=True, slots=True)
class TrellisHoldoutSeal:
    """A prediction committed before its excluded measured rung is supplied."""

    surface_sha256: str
    curve_identity_sha256: str
    body_rate_q256: int
    pre_render_recipe_identity_sha256: str
    predicted_dloss: float
    scalar_backbone_point_sha256: str | None
    seal_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.surface_sha256, field="holdout.surface_sha256")
        _sha256(
            self.curve_identity_sha256,
            field="holdout.curve_identity_sha256",
        )
        if type(self.body_rate_q256) is not int:
            raise TrellisFormatError("holdout body_rate_q256 must be an integer")
        _sha256(
            self.pre_render_recipe_identity_sha256,
            field="holdout.pre_render_recipe_identity_sha256",
        )
        object.__setattr__(
            self,
            "predicted_dloss",
            _positive_finite(
                self.predicted_dloss, field="holdout.predicted_dloss"
            ),
        )
        if self.scalar_backbone_point_sha256 is not None:
            _sha256(
                self.scalar_backbone_point_sha256,
                field="holdout.scalar_backbone_point_sha256",
            )
        expected = _canonical_sha256(self._body())
        if self.seal_sha256 != expected:
            raise TrellisFormatError("holdout seal digest does not match its body")

    def _body(self) -> dict[str, object]:
        return {
            "schema": _HOLDOUT_SEAL_SCHEMA,
            "surface_sha256": self.surface_sha256,
            "curve_identity_sha256": self.curve_identity_sha256,
            "body_rate_q256": self.body_rate_q256,
            "pre_render_recipe_identity_sha256": (
                self.pre_render_recipe_identity_sha256
            ),
            "predicted_dloss": self.predicted_dloss,
            "scalar_backbone_point_sha256": (
                self.scalar_backbone_point_sha256
            ),
            "prediction_precedes_measurement": True,
            "research_only": True,
            "publication_eligible": False,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._body(), "seal_sha256": self.seal_sha256}


def seal_rate_surface_holdout(
    surface: TrellisRateSurface,
    body_rate_q256: int,
    *,
    pre_render_recipe_identity_sha256: str,
    scalar_backbone: TrellisScalarBackbonePoint | None = None,
) -> TrellisHoldoutSeal:
    """Seal one excluded interior prediction before its measurement is known."""

    if not isinstance(surface, TrellisRateSurface):
        raise TrellisFormatError("holdout sealing needs a rate surface")
    low, high = surface.q256_range
    if not low < body_rate_q256 < high:
        raise TrellisFormatError(
            "a holdout must be strictly interior to the measured envelope"
        )
    if body_rate_q256 in surface.anchor_q256:
        raise TrellisFormatError(
            "a fitted anchor cannot also be the excluded measured holdout"
        )
    recipe_sha256 = _sha256(
        pre_render_recipe_identity_sha256,
        field="holdout.pre_render_recipe_identity_sha256",
    )
    if (
        surface.fit_response == FIT_RAW_DLOSS_LOG2
        and scalar_backbone is not None
    ):
        raise TrellisFormatError(
            "raw-dloss holdouts must not claim an unused scalar backbone"
        )
    predicted = surface.predict(
        body_rate_q256,
        intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
        scalar_backbone=scalar_backbone,
    )
    body = {
        "schema": _HOLDOUT_SEAL_SCHEMA,
        "surface_sha256": surface.surface_sha256,
        "curve_identity_sha256": surface.curve_identity.identity_sha256,
        "body_rate_q256": body_rate_q256,
        "pre_render_recipe_identity_sha256": recipe_sha256,
        "predicted_dloss": predicted,
        "scalar_backbone_point_sha256": (
            scalar_backbone.identity_sha256
            if scalar_backbone is not None
            else None
        ),
        "prediction_precedes_measurement": True,
        "research_only": True,
        "publication_eligible": False,
    }
    return TrellisHoldoutSeal(
        surface_sha256=surface.surface_sha256,
        curve_identity_sha256=surface.curve_identity.identity_sha256,
        body_rate_q256=body_rate_q256,
        pre_render_recipe_identity_sha256=recipe_sha256,
        predicted_dloss=predicted,
        scalar_backbone_point_sha256=(
            scalar_backbone.identity_sha256
            if scalar_backbone is not None
            else None
        ),
        seal_sha256=_canonical_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class TrellisHoldoutGrade:
    """Measured grading of a pre-existing holdout prediction seal."""

    surface_sha256: str
    curve_identity_sha256: str
    body_rate_q256: int
    seal_sha256: str
    measured_anchor_sha256: str
    pre_render_recipe_identity_sha256: str
    scalar_backbone_point_sha256: str | None
    predicted_dloss: float
    measured_dloss: float
    log2_error: float
    relative_error_pct: float
    grade_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "surface_sha256",
            "curve_identity_sha256",
            "seal_sha256",
            "measured_anchor_sha256",
            "pre_render_recipe_identity_sha256",
        ):
            _sha256(getattr(self, field), field=f"holdout_grade.{field}")
        if self.scalar_backbone_point_sha256 is not None:
            _sha256(
                self.scalar_backbone_point_sha256,
                field="holdout_grade.scalar_backbone_point_sha256",
            )
        if type(self.body_rate_q256) is not int:
            raise TrellisFormatError("holdout grade q256 must be an integer")
        predicted = _positive_finite(
            self.predicted_dloss, field="holdout predicted dloss"
        )
        measured = _positive_finite(
            self.measured_dloss, field="holdout measured dloss"
        )
        object.__setattr__(self, "predicted_dloss", predicted)
        object.__setattr__(self, "measured_dloss", measured)
        if isinstance(self.log2_error, bool) or isinstance(
            self.relative_error_pct, bool
        ):
            raise TrellisFormatError("holdout residuals must be finite")
        try:
            log2_error = float(self.log2_error)
            relative_error_pct = float(self.relative_error_pct)
        except (TypeError, ValueError) as exc:
            raise TrellisFormatError("holdout residuals must be finite") from exc
        if not math.isfinite(log2_error) or not math.isfinite(
            relative_error_pct
        ):
            raise TrellisFormatError("holdout residuals must be finite")
        object.__setattr__(self, "log2_error", log2_error)
        object.__setattr__(self, "relative_error_pct", relative_error_pct)
        expected_log2_error = math.log2(predicted / measured)
        expected_relative_error_pct = (predicted / measured - 1.0) * 100.0
        if (
            log2_error != expected_log2_error
            or relative_error_pct != expected_relative_error_pct
        ):
            raise TrellisFormatError(
                "holdout residuals disagree with predicted/measured loss"
            )
        expected_seal_body = {
            "schema": _HOLDOUT_SEAL_SCHEMA,
            "surface_sha256": self.surface_sha256,
            "curve_identity_sha256": self.curve_identity_sha256,
            "body_rate_q256": self.body_rate_q256,
            "pre_render_recipe_identity_sha256": (
                self.pre_render_recipe_identity_sha256
            ),
            "predicted_dloss": predicted,
            "scalar_backbone_point_sha256": (
                self.scalar_backbone_point_sha256
            ),
            "prediction_precedes_measurement": True,
            "research_only": True,
            "publication_eligible": False,
        }
        if self.seal_sha256 != _canonical_sha256(expected_seal_body):
            raise TrellisFormatError(
                "holdout grade seal digest does not match its embedded seal body"
            )
        if self.grade_sha256 != _canonical_sha256(self._body()):
            raise TrellisFormatError("holdout grade digest does not match its body")

    def _body(self) -> dict[str, object]:
        return {
            "schema": _HOLDOUT_GRADE_SCHEMA,
            "surface_sha256": self.surface_sha256,
            "curve_identity_sha256": self.curve_identity_sha256,
            "body_rate_q256": self.body_rate_q256,
            "seal_sha256": self.seal_sha256,
            "measured_anchor_sha256": self.measured_anchor_sha256,
            "pre_render_recipe_identity_sha256": (
                self.pre_render_recipe_identity_sha256
            ),
            "scalar_backbone_point_sha256": (
                self.scalar_backbone_point_sha256
            ),
            "predicted_dloss": self.predicted_dloss,
            "measured_dloss": self.measured_dloss,
            "log2_error": self.log2_error,
            "relative_error_pct": self.relative_error_pct,
            "research_only": True,
            "publication_eligible": False,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._body(), "grade_sha256": self.grade_sha256}


def grade_rate_surface_holdout(
    surface: TrellisRateSurface,
    seal: TrellisHoldoutSeal,
    measured_anchor: TrellisMeasuredRateAnchor,
) -> TrellisHoldoutGrade:
    """Bind a measured, excluded anchor to its earlier prediction seal."""

    if not isinstance(surface, TrellisRateSurface):
        raise TrellisFormatError("holdout grading needs a rate surface")
    if not isinstance(seal, TrellisHoldoutSeal):
        raise TrellisFormatError("holdout grading needs a valid seal")
    if not isinstance(measured_anchor, TrellisMeasuredRateAnchor):
        raise TrellisFormatError("holdout grading needs a measured anchor")
    if seal.surface_sha256 != surface.surface_sha256:
        raise TrellisFormatError("holdout seal names a different fitted surface")
    if seal.curve_identity_sha256 != surface.curve_identity.identity_sha256:
        raise TrellisFormatError("holdout seal curve identity drifted")
    if measured_anchor.curve_identity != surface.curve_identity:
        raise TrellisFormatError("holdout measurement curve identity drifted")
    if measured_anchor.currency != surface.currency:
        raise TrellisFormatError("holdout measurement currency drifted")
    if (
        measured_anchor.candidate.unit_name != surface.unit_name
        or tuple(measured_anchor.candidate.footprint.get("shape", ()))
        != surface.shape
        or measured_anchor.candidate.family != surface.family
        or measured_anchor.candidate.layout != surface.layout
        or measured_anchor.candidate.servability.target_profile
        != surface.target_profile
    ):
        raise TrellisFormatError(
            "holdout measurement unit/shape/wire context drifted"
        )
    if measured_anchor.candidate.body_rate_q256 != seal.body_rate_q256:
        raise TrellisFormatError("holdout measurement rate differs from its seal")
    if (
        measured_anchor.candidate.footprint.get(
            "pre_render_recipe_identity_sha256"
        )
        != seal.pre_render_recipe_identity_sha256
    ):
        raise TrellisFormatError(
            "holdout measurement used a different precommitted wire recipe"
        )
    if seal.body_rate_q256 in surface.anchor_q256:
        raise TrellisFormatError("holdout leaked into the fitted anchor set")
    prediction_backbone: TrellisScalarBackbonePoint | None = None
    if surface.fit_response == FIT_MATCHED_SCALAR_GAIN_DB:
        measured_backbone = measured_anchor.scalar_backbone
        if measured_backbone is None:
            raise TrellisFormatError(
                "gain holdout measurement lacks its matched scalar backbone"
            )
        if not measured_backbone.context_parity_verified:
            raise TrellisFormatError(
                "gain holdout scalar-backbone context parity is not verified"
            )
        if (
            seal.scalar_backbone_point_sha256 is None
            or measured_backbone.identity_sha256
            != seal.scalar_backbone_point_sha256
        ):
            raise TrellisFormatError(
                "holdout measurement used a different scalar backbone point "
                "than the sealed prediction"
            )
        prediction_backbone = measured_backbone
    elif (
        seal.scalar_backbone_point_sha256 is not None
        or measured_anchor.scalar_backbone is not None
    ):
        raise TrellisFormatError(
            "raw-dloss holdout must not import scalar-backbone provenance"
        )
    expected_prediction = surface.predict(
        seal.body_rate_q256,
        intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
        scalar_backbone=prediction_backbone,
    )
    if seal.predicted_dloss != expected_prediction:
        raise TrellisFormatError(
            "holdout seal prediction differs from the bound fitted surface"
        )
    measured = measured_anchor.candidate.predicted_dloss_mean
    log_error = math.log2(seal.predicted_dloss / measured)
    relative = (seal.predicted_dloss / measured - 1.0) * 100.0
    body = {
        "schema": _HOLDOUT_GRADE_SCHEMA,
        "surface_sha256": surface.surface_sha256,
        "curve_identity_sha256": surface.curve_identity.identity_sha256,
        "body_rate_q256": seal.body_rate_q256,
        "seal_sha256": seal.seal_sha256,
        "measured_anchor_sha256": measured_anchor.anchor_sha256,
        "pre_render_recipe_identity_sha256": (
            seal.pre_render_recipe_identity_sha256
        ),
        "scalar_backbone_point_sha256": (
            seal.scalar_backbone_point_sha256
        ),
        "predicted_dloss": seal.predicted_dloss,
        "measured_dloss": measured,
        "log2_error": log_error,
        "relative_error_pct": relative,
        "research_only": True,
        "publication_eligible": False,
    }
    return TrellisHoldoutGrade(
        surface_sha256=surface.surface_sha256,
        curve_identity_sha256=surface.curve_identity.identity_sha256,
        body_rate_q256=seal.body_rate_q256,
        seal_sha256=seal.seal_sha256,
        measured_anchor_sha256=measured_anchor.anchor_sha256,
        pre_render_recipe_identity_sha256=(
            seal.pre_render_recipe_identity_sha256
        ),
        scalar_backbone_point_sha256=seal.scalar_backbone_point_sha256,
        predicted_dloss=seal.predicted_dloss,
        measured_dloss=measured,
        log2_error=log_error,
        relative_error_pct=relative,
        grade_sha256=_canonical_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class TrellisAllocationRegretGate:
    """Immutable retrospective result for the internal greedy allocator."""

    byte_budget: int
    max_regret_pct: float
    regret_pct: float
    assignment_agreement: float
    interpolated_assignment_fraction: float
    true_loss_deciding_on_interpolated: float
    true_loss_deciding_on_truth: float
    assignment_interpolated: tuple[tuple[str, int], ...]
    assignment_truth: tuple[tuple[str, int], ...]
    surface_bindings: tuple[tuple[str, str], ...]
    densified_bindings: tuple[tuple[str, str], ...]
    truth_bindings: tuple[tuple[str, str], ...]
    holdout_grade_sha256: tuple[tuple[str, tuple[str, ...]], ...]
    bracket_agreement: bool
    passed: bool
    gate_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "assignment_interpolated",
            "assignment_truth",
            "surface_bindings",
            "densified_bindings",
            "truth_bindings",
        ):
            try:
                frozen = tuple((unit, value) for unit, value in getattr(self, field))
            except (TypeError, ValueError) as exc:
                raise TrellisFormatError(
                    f"regret gate {field} must be key/value pairs"
                ) from exc
            object.__setattr__(self, field, frozen)
        try:
            frozen_grades = tuple(
                (unit, tuple(digests))
                for unit, digests in self.holdout_grade_sha256
            )
        except (TypeError, ValueError) as exc:
            raise TrellisFormatError(
                "regret gate holdout grades must be unit/digest pairs"
            ) from exc
        object.__setattr__(self, "holdout_grade_sha256", frozen_grades)
        if type(self.byte_budget) is not int or self.byte_budget <= 0:
            raise TrellisFormatError("regret gate byte_budget must be positive")
        for field in (
            "max_regret_pct",
            "assignment_agreement",
            "interpolated_assignment_fraction",
            "true_loss_deciding_on_interpolated",
            "true_loss_deciding_on_truth",
        ):
            raw_value = getattr(self, field)
            if isinstance(raw_value, bool):
                raise TrellisFormatError(f"regret gate {field} is invalid")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TrellisFormatError(
                    f"regret gate {field} is invalid"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise TrellisFormatError(f"regret gate {field} is invalid")
            object.__setattr__(self, field, value)
        if isinstance(self.regret_pct, bool):
            raise TrellisFormatError(
                "regret gate regret_pct must be nonnegative and finite"
            )
        try:
            object.__setattr__(self, "regret_pct", float(self.regret_pct))
        except (TypeError, ValueError) as exc:
            raise TrellisFormatError(
                "regret gate regret_pct must be nonnegative and finite"
            ) from exc
        if not math.isfinite(self.regret_pct) or self.regret_pct < 0.0:
            raise TrellisFormatError(
                "regret gate regret_pct must be nonnegative and finite"
            )
        if not 0.0 <= self.assignment_agreement <= 1.0:
            raise TrellisFormatError("assignment_agreement must be in [0, 1]")
        if not 0.0 <= self.interpolated_assignment_fraction <= 1.0:
            raise TrellisFormatError(
                "interpolated_assignment_fraction must be in [0, 1]"
            )
        if (
            self.true_loss_deciding_on_interpolated <= 0.0
            or self.true_loss_deciding_on_truth <= 0.0
        ):
            raise TrellisFormatError(
                "regret gate true losses must be positive and finite"
            )
        if type(self.bracket_agreement) is not bool or not self.bracket_agreement:
            raise TrellisFormatError("allocation gate requires bracket agreement")
        if type(self.passed) is not bool or self.passed != (
            self.regret_pct <= self.max_regret_pct
        ):
            raise TrellisFormatError("regret gate pass bit disagrees with its bound")
        for unit_name, digest in self.surface_bindings:
            _nonempty_text(unit_name, field="regret gate unit")
            _sha256(digest, field="regret gate surface binding")
        for unit_name, digest in self.densified_bindings:
            _nonempty_text(unit_name, field="regret gate unit")
            _sha256(digest, field="regret gate densified binding")
        for unit_name, digest in self.truth_bindings:
            _nonempty_text(unit_name, field="regret gate unit")
            _sha256(digest, field="regret gate truth binding")
        for field in ("assignment_interpolated", "assignment_truth"):
            for unit_name, rate in getattr(self, field):
                _nonempty_text(unit_name, field="regret gate assignment unit")
                if type(rate) is not int or rate <= 0:
                    raise TrellisFormatError(
                        f"regret gate {field} rates must be positive q256 integers"
                    )
        for unit_name, digests in self.holdout_grade_sha256:
            _nonempty_text(unit_name, field="regret gate holdout unit")
            if not digests:
                raise TrellisFormatError(
                    "every allocation surface needs a measured holdout grade"
                )
            if len(set(digests)) != len(digests):
                raise TrellisFormatError(
                    "regret gate repeats a holdout-grade identity"
                )
            for digest in digests:
                _sha256(digest, field="regret gate holdout grade")
        all_grade_digests = [
            digest
            for _unit_name, digests in self.holdout_grade_sha256
            for digest in digests
        ]
        if len(set(all_grade_digests)) != len(all_grade_digests):
            raise TrellisFormatError(
                "regret gate reuses a holdout-grade identity across units"
            )
        unit_sets = [
            {unit for unit, _ in pairs}
            for pairs in (
                self.assignment_interpolated,
                self.assignment_truth,
                self.surface_bindings,
                self.densified_bindings,
                self.truth_bindings,
                self.holdout_grade_sha256,
            )
        ]
        if any(len(units) != len(pairs) for units, pairs in zip(
            unit_sets,
            (
                self.assignment_interpolated,
                self.assignment_truth,
                self.surface_bindings,
                self.densified_bindings,
                self.truth_bindings,
                self.holdout_grade_sha256,
            ),
        )) or any(units != unit_sets[0] for units in unit_sets[1:]):
            raise TrellisFormatError(
                "regret gate unit bindings must be unique and agree exactly"
            )
        if not unit_sets[0]:
            raise TrellisFormatError(
                "regret gate must bind at least one allocator unit"
            )
        interpolated_assignment = dict(self.assignment_interpolated)
        truth_assignment = dict(self.assignment_truth)
        expected_agreement = sum(
            interpolated_assignment[unit_name] == truth_assignment[unit_name]
            for unit_name in truth_assignment
        ) / len(truth_assignment)
        if self.assignment_agreement != expected_agreement:
            raise TrellisFormatError(
                "regret gate assignment agreement disagrees with its assignments"
            )
        raw_regret = (
            self.true_loss_deciding_on_interpolated
            / self.true_loss_deciding_on_truth
            - 1.0
        ) * 100.0
        if raw_regret < -1e-12:
            raise TrellisFormatError(
                "regret gate claims an interpolated decision that beats its "
                "truth-deciding reference"
            )
        expected_regret = max(0.0, raw_regret)
        if self.regret_pct != expected_regret:
            raise TrellisFormatError(
                "regret gate percentage disagrees with its bound true losses"
            )
        if self.gate_sha256 != _canonical_sha256(self._body()):
            raise TrellisFormatError("regret gate digest does not match its body")

    def _body(self) -> dict[str, object]:
        return {
            "schema": _REGRET_GATE_SCHEMA,
            "licensed_use": SURFACE_USE_ALLOCATOR_COST,
            "byte_budget": self.byte_budget,
            "max_regret_pct": self.max_regret_pct,
            "regret_pct": self.regret_pct,
            "assignment_agreement": self.assignment_agreement,
            "interpolated_assignment_fraction": (
                self.interpolated_assignment_fraction
            ),
            "true_loss_deciding_on_interpolated": (
                self.true_loss_deciding_on_interpolated
            ),
            "true_loss_deciding_on_truth": self.true_loss_deciding_on_truth,
            "assignment_interpolated": dict(self.assignment_interpolated),
            "assignment_truth": dict(self.assignment_truth),
            "surface_bindings": dict(self.surface_bindings),
            "densified_bindings": dict(self.densified_bindings),
            "truth_bindings": dict(self.truth_bindings),
            "holdout_grade_sha256": {
                unit: list(digests)
                for unit, digests in self.holdout_grade_sha256
            },
            "bracket_agreement": self.bracket_agreement,
            "passed": self.passed,
            "research_only": True,
            "menu_verdict_eligible": False,
            "family_verdict_eligible": False,
            "publication_eligible": False,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._body(), "gate_sha256": self.gate_sha256}

    def __getitem__(self, key: str) -> object:
        """Small compatibility facade for the former report dictionary."""

        return self.as_dict()[key]


def rate_surface_solver_menu(
    densified: Mapping[str, TrellisDensifiedSurface],
    *,
    currencies: Mapping[str, str],
    intended_use: str,
    regret_gate: TrellisAllocationRegretGate,
    byte_budget: int,
    fail_on_denied: bool = True,
) -> dict[str, list]:
    """Materialize the exact menu covered by a retrospective research gate.

    This is an ephemeral solver menu, not a format-menu verdict.  Campaign
    planning can inspect :class:`TrellisDensifiedSurface` directly; it cannot
    call this allocator bridge.  Publication/family/menu-verdict spellings are
    rejected by the closed use vocabulary before any candidate is returned.
    The gate evaluates only :func:`allocation_regret`'s internal greedy
    marginal rule.  Returning the menu does not transfer its regret result to
    ``allocator_solver`` or any other consuming allocator.
    """

    if intended_use != SURFACE_USE_ALLOCATOR_COST:
        raise TrellisFormatError(
            "the solver bridge is available only for allocator-cost research; "
            "campaign planning does not create an allocation menu"
        )
    if fail_on_denied is not True:
        raise TrellisFormatError(
            "a regret-bound exact menu cannot silently filter denied candidates"
        )
    if not densified:
        raise TrellisFormatError(
            "the allocator bridge needs at least one densified surface"
        )
    if not isinstance(regret_gate, TrellisAllocationRegretGate):
        raise TrellisFormatError(
            "allocator-cost interpolation requires a sealed allocation-regret gate"
        )
    if not regret_gate.passed:
        raise TrellisFormatError(
            f"allocation regret {regret_gate.regret_pct:.9g}% exceeds the "
            f"sealed {regret_gate.max_regret_pct:.9g}% bound"
        )
    if type(byte_budget) is not int or byte_budget != regret_gate.byte_budget:
        raise TrellisFormatError(
            "solver byte budget differs from the budget graded by allocation regret"
        )
    if set(currencies) != set(densified):
        raise TrellisFormatError(
            "densified units and currency declarations must match exactly"
        )
    declared = set(currencies.values())
    if len(declared) != 1:
        raise TrellisFormatError(
            f"rate surface menu mixes objectives {sorted(declared)}; one DP "
            f"prices in one currency"
        )
    gate_bindings = dict(regret_gate.surface_bindings)
    gate_dense_bindings = dict(regret_gate.densified_bindings)
    gate_interpolated_assignment = dict(regret_gate.assignment_interpolated)
    gate_truth_assignment = dict(regret_gate.assignment_truth)
    if set(gate_bindings) != set(densified):
        raise TrellisFormatError(
            "allocation-regret gate covers a different unit set"
        )
    flat: list[TrellisAllocatorCandidate] = []
    spent_interpolated = 0
    spent_truth = 0
    interpolated_count = 0
    shared_curve_identity: TrellisCurveIdentity | None = None
    for unit_name in sorted(densified):
        built = densified[unit_name]
        if not isinstance(built, TrellisDensifiedSurface):
            raise TrellisFormatError(
                "allocator bridge requires identity-bound densified surfaces"
            )
        if shared_curve_identity is None:
            shared_curve_identity = built.surface.curve_identity
        elif built.surface.curve_identity != shared_curve_identity:
            raise TrellisFormatError(
                "allocator bridge units must share one curve identity"
            )
        if built.intended_use != SURFACE_USE_ALLOCATOR_COST:
            raise TrellisFormatError(
                "planning-only densification cannot enter the allocator"
            )
        if built.surface.currency != currencies[unit_name]:
            raise TrellisFormatError(
                f"unit {unit_name!r} currency differs from its fitted surface"
            )
        if gate_bindings[unit_name] != built.surface.surface_sha256:
            raise TrellisFormatError(
                "densified surface identity differs from the regret gate"
            )
        if gate_dense_bindings.get(unit_name) != built.identity_sha256:
            raise TrellisFormatError(
                "exact candidate menu differs from the validated regret record"
            )
        candidates_by_rate = {
            candidate.body_rate_q256: candidate for candidate in built
        }
        if (
            gate_interpolated_assignment.get(unit_name)
            not in candidates_by_rate
            or gate_truth_assignment.get(unit_name)
            not in candidates_by_rate
        ):
            raise TrellisFormatError(
                "regret-gate assignment names a rung outside the exact menu"
            )
        interpolated_rate = gate_interpolated_assignment[unit_name]
        truth_rate = gate_truth_assignment[unit_name]
        spent_interpolated += int(
            candidates_by_rate[interpolated_rate].footprint["total_bytes"]
        )
        spent_truth += int(
            candidates_by_rate[truth_rate].footprint["total_bytes"]
        )
        interpolated_count += interpolated_rate not in built.surface.anchor_q256
        for candidate in built:
            if not candidate.servability.legal:
                raise TrellisFormatError(
                    "a regret-bound exact menu contains a target-profile-denied "
                    "candidate and cannot be filtered after grading"
                )
        flat.extend(built)
    if spent_interpolated > byte_budget or spent_truth > byte_budget:
        raise TrellisFormatError(
            "regret-gate assignment exceeds its exact byte budget"
        )
    expected_mixing_fraction = interpolated_count / len(densified)
    if regret_gate.interpolated_assignment_fraction != expected_mixing_fraction:
        raise TrellisFormatError(
            "regret-gate interpolation fraction disagrees with its assignment"
        )
    return trellis_solver_candidate_menu(flat, fail_on_denied=True)


def leave_one_anchor_out(
    surface: TrellisRateSurface,
) -> dict[str, object]:
    """Diagnostic: how well do the anchors predict each other?

    Drops one interior anchor at a time, rebuilds the surface from the rest,
    and reports the log2 error at the dropped rung.  Endpoints cannot be
    dropped -- removing one would require extrapolation, which is refused.

    This is a DIAGNOSTIC, not a gate.  Use :func:`allocation_regret`.
    """

    if len(surface.anchor_q256) < 3:
        return {
            "interior_anchors": 0,
            "note": "fewer than three anchors: nothing droppable",
        }
    errors: list[dict[str, float]] = []
    for index in range(1, len(surface.anchor_q256) - 1):
        kept = TrellisRateSurface(
            unit_name=surface.unit_name,
            family=surface.family,
            layout=surface.layout,
            shape=surface.shape,
            target_profile=surface.target_profile,
            currency=surface.currency,
            curve_identity=surface.curve_identity,
            fit_response=surface.fit_response,
            anchor_q256=(
                surface.anchor_q256[:index] + surface.anchor_q256[index + 1:]
            ),
            anchor_dloss=(
                surface.anchor_dloss[:index] + surface.anchor_dloss[index + 1:]
            ),
            anchor_stderr=(
                surface.anchor_stderr[:index]
                + surface.anchor_stderr[index + 1:]
            ),
            anchor_sha256=(
                surface.anchor_sha256[:index]
                + surface.anchor_sha256[index + 1:]
            ),
            scalar_backbone_closure_sha256=(
                surface.scalar_backbone_closure_sha256
            ),
            anchor_backbone_dloss=(
                surface.anchor_backbone_dloss[:index]
                + surface.anchor_backbone_dloss[index + 1:]
                if surface.fit_response == FIT_MATCHED_SCALAR_GAIN_DB
                else ()
            ),
            anchor_backbone_point_sha256=(
                surface.anchor_backbone_point_sha256[:index]
                + surface.anchor_backbone_point_sha256[index + 1:]
                if surface.fit_response == FIT_MATCHED_SCALAR_GAIN_DB
                else ()
            ),
        )
        rate = surface.anchor_q256[index]
        truth = surface.anchor_dloss[index]
        if kept.fit_response == FIT_RAW_DLOSS_LOG2:
            predicted = kept.predict(
                rate, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
            )
        else:
            lower, upper, weight = kept._bracket(rate)
            predicted = kept._predict_from_backbone_dloss(
                lower,
                upper,
                weight,
                surface.anchor_backbone_dloss[index],
            )
            if not kept.anchor_dloss[upper] < predicted < kept.anchor_dloss[lower]:
                raise TrellisFormatError(
                    "leave-one-anchor-out gain prediction became non-monotone"
                )
        errors.append(
            {
                "q256": float(rate),
                "true_dloss": truth,
                "predicted_dloss": predicted,
                "log2_error": math.log2(predicted / truth),
                "rel_error_pct": (predicted / truth - 1.0) * 100.0,
            }
        )
    magnitudes = [abs(entry["log2_error"]) for entry in errors]
    return {
        "interior_anchors": len(errors),
        "max_abs_log2_error": max(magnitudes),
        "median_abs_log2_error": sorted(magnitudes)[len(magnitudes) // 2],
        "per_anchor": errors,
    }


def allocation_regret(
    surfaces: Mapping[str, TrellisRateSurface],
    truth: Mapping[str, Mapping[int, TrellisMeasuredRateAnchor]],
    *,
    densified: Mapping[str, TrellisDensifiedSurface],
    scalar_backbones: Mapping[
        str, Mapping[int, TrellisScalarBackbonePoint]
    ],
    holdout_grades: Mapping[str, Sequence[TrellisHoldoutGrade]],
    byte_budget: int,
    max_regret_pct: float,
    bracket_agreement: bool,
) -> TrellisAllocationRegretGate:
    """Retrospectively measure what interpolation cost the greedy decision.

    Allocates a byte budget across units twice -- once on the interpolated
    surface, once on measured truth -- and reports the true objective of each.
    Both allocations are scored with TRUTH, so the number is the real cost of
    deciding on interpolated values.

    This uses greedy marginal allocation. It is exact only for separable
    CONVEX per-unit curves, and real trellis
    interiors are not reliably convex -- 142 of 196 tensors have native
    fractional rates sagging above their own chord.  The regret number stays
    valid because BOTH arms run the identical allocator, so the comparison
    isolates the interpolation. It is neither a claim of optimal allocation
    nor evidence about a different consuming allocator.
    """

    if not surfaces:
        raise TrellisFormatError("allocation regret needs at least one surface")
    unit_names = set(surfaces)
    if not (
        set(truth)
        == set(densified)
        == set(scalar_backbones)
        == set(holdout_grades)
        == unit_names
    ):
        raise TrellisFormatError(
            "surfaces, truth, exact densified menus, scalar backbones, and holdouts must "
            "cover exactly the same units"
        )
    if type(bracket_agreement) is not bool or not bracket_agreement:
        raise TrellisFormatError(
            "scale-price bracket disagreement fails closed; interpolation "
            "cannot narrow a bracket"
        )
    bound = _nonnegative_finite(max_regret_pct, field="max_regret_pct")
    if type(byte_budget) is not int or byte_budget <= 0:
        raise TrellisFormatError("byte_budget must be a positive integer")
    if any(
        not isinstance(surface, TrellisRateSurface)
        for surface in surfaces.values()
    ):
        raise TrellisFormatError("allocation regret received an invalid surface")
    if len({surface.currency for surface in surfaces.values()}) != 1:
        raise TrellisFormatError("allocation regret mixes objective currencies")
    shared_curve_identity = next(iter(surfaces.values())).curve_identity
    if any(
        surface.curve_identity != shared_curve_identity
        for surface in surfaces.values()
    ):
        raise TrellisFormatError(
            "allocation regret units must share one curve identity"
        )

    normalized_truth: dict[str, dict[int, float]] = {}
    normalized_truth_anchors: dict[
        str, dict[int, TrellisMeasuredRateAnchor]
    ] = {}
    normalized_bytes: dict[str, dict[int, int]] = {}
    normalized_backbones: dict[
        str, dict[int, TrellisScalarBackbonePoint]
    ] = {}
    normalized_grades: dict[str, tuple[TrellisHoldoutGrade, ...]] = {}
    for unit_name in sorted(surfaces):
        surface = surfaces[unit_name]
        if surface.unit_name != unit_name:
            raise TrellisFormatError(
                "allocation mapping key differs from the surface unit name"
            )
        surface.authorize_use(SURFACE_USE_ALLOCATOR_COST)
        built = densified[unit_name]
        if not isinstance(built, TrellisDensifiedSurface):
            raise TrellisFormatError(
                "allocation regret requires identity-bound densified surfaces"
            )
        if built.surface != surface or built.intended_use != SURFACE_USE_ALLOCATOR_COST:
            raise TrellisFormatError(
                "allocation densification differs from its allocator-cost surface"
            )
        candidates_by_rate = {
            candidate.body_rate_q256: candidate for candidate in built
        }
        densified_backbone_bindings = dict(
            built.scalar_backbone_point_sha256
        )
        if len(candidates_by_rate) != len(built):
            raise TrellisFormatError("allocation densification repeats a q256 rung")
        rates = set(truth[unit_name])
        if (
            not rates
            or rates != set(candidates_by_rate)
            or rates != set(scalar_backbones[unit_name])
        ):
            raise TrellisFormatError(
                f"unit {unit_name!r} truth/backbone/densified rate sets differ"
            )
        if any(type(rate) is not int for rate in rates):
            raise TrellisFormatError("allocation truth rates must be q256 integers")
        ordered_rates = sorted(rates)
        truth_anchors: dict[int, TrellisMeasuredRateAnchor] = {}
        values: dict[int, float] = {}
        for rate in ordered_rates:
            truth_anchor = truth[unit_name][rate]
            if not isinstance(truth_anchor, TrellisMeasuredRateAnchor):
                raise TrellisFormatError(
                    "allocation truth must use identity-bound measured anchors"
                )
            if (
                truth_anchor.curve_identity != surface.curve_identity
                or truth_anchor.currency != surface.currency
                or truth_anchor.candidate.unit_name != unit_name
                or truth_anchor.candidate.body_rate_q256 != rate
            ):
                raise TrellisFormatError(
                    "allocation truth anchor identity/unit/rate drifted"
                )
            measured_recipe = truth_anchor.candidate.footprint.get(
                "pre_render_recipe_identity_sha256"
            )
            predicted_recipe = candidates_by_rate[rate].footprint.get(
                "pre_render_recipe_identity_sha256"
            )
            if measured_recipe != predicted_recipe:
                raise TrellisFormatError(
                    "allocation truth measured a different exact wire recipe"
                )
            if (
                truth_anchor.candidate.servability
                != candidates_by_rate[rate].servability
            ):
                raise TrellisFormatError(
                    "allocation truth and densified target-profile context differ"
                )
            truth_anchors[rate] = truth_anchor
            values[rate] = _positive_finite(
                truth_anchor.candidate.predicted_dloss_mean,
                field=f"truth[{unit_name}][{rate}]",
            )
        for left, right in zip(ordered_rates, ordered_rates[1:]):
            if values[right] >= values[left]:
                raise TrellisFormatError(
                    f"unit {unit_name!r} measured truth is non-monotone; "
                    "re-anchor instead of grading an interpolant through it"
                )
        bytes_for_unit: dict[int, int] = {}
        backbones_for_unit: dict[int, TrellisScalarBackbonePoint] = {}
        previous_bytes = -1
        for rate in ordered_rates:
            surface._bracket(rate)
            candidate = candidates_by_rate[rate]
            if not candidate.servability.legal:
                raise TrellisFormatError(
                    "allocation regret cannot grade a menu containing a "
                    "target-profile-denied candidate"
                )
            byte_count = candidate.footprint.get("total_bytes")
            if type(byte_count) is not int or byte_count <= 0:
                raise TrellisFormatError("unit byte prices must be positive integers")
            if byte_count < previous_bytes:
                raise TrellisFormatError(
                    "exact serialized bytes must not decrease as q256 rises"
                )
            previous_bytes = byte_count
            bytes_for_unit[rate] = byte_count
            backbone = scalar_backbones[unit_name][rate]
            if not isinstance(backbone, TrellisScalarBackbonePoint):
                raise TrellisFormatError("allocation backbone point has invalid type")
            if (
                backbone.curve_identity != surface.curve_identity
                or backbone.body_rate_q256 != rate
            ):
                raise TrellisFormatError(
                    "allocation scalar backbone identity/rate drifted"
                )
            if not backbone.context_parity_verified:
                raise TrellisFormatError(
                    "allocation scalar backbone lacks verified context parity"
                )
            if densified_backbone_bindings.get(rate) != backbone.identity_sha256:
                raise TrellisFormatError(
                    "allocation scalar backbone differs from the densified binding"
                )
            measured_backbone = truth_anchors[rate].scalar_backbone
            if (
                measured_backbone is None
                or measured_backbone.identity_sha256 != backbone.identity_sha256
            ):
                raise TrellisFormatError(
                    "allocation truth and scalar backbone measurement disagree"
                )
            backbones_for_unit[rate] = backbone
            expected = surface.predict(
                rate,
                intended_use=SURFACE_USE_ALLOCATOR_COST,
                scalar_backbone=backbone,
            )
            if not math.isclose(
                candidate.predicted_dloss_mean,
                expected,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise TrellisFormatError(
                    "densified candidate loss differs from the bound surface/backbone"
                )
            expected_stderr = surface.predict_stderr(
                rate, intended_use=SURFACE_USE_ALLOCATOR_COST
            )
            if candidate.predicted_dloss_stderr != expected_stderr:
                raise TrellisFormatError(
                    "densified candidate stderr differs from the bound surface"
                )
        grades = tuple(holdout_grades[unit_name])
        for left, right in zip(ordered_rates, ordered_rates[1:]):
            if backbones_for_unit[right].dloss >= backbones_for_unit[left].dloss:
                raise TrellisFormatError(
                    f"unit {unit_name!r} scalar-backbone truth is non-monotone"
                )
        if not grades:
            raise TrellisFormatError(
                f"unit {unit_name!r} has no measured interior holdout"
            )
        if any(not isinstance(grade, TrellisHoldoutGrade) for grade in grades):
            raise TrellisFormatError("holdout grade has invalid type")
        if (
            len({grade.grade_sha256 for grade in grades}) != len(grades)
            or len({grade.body_rate_q256 for grade in grades}) != len(grades)
        ):
            raise TrellisFormatError(
                "allocation holdout grades must be unique measured rungs"
            )
        for grade in grades:
            if (
                grade.surface_sha256 != surface.surface_sha256
                or grade.curve_identity_sha256
                != surface.curve_identity.identity_sha256
            ):
                raise TrellisFormatError("holdout grade identity drifted")
            if grade.body_rate_q256 not in rates:
                raise TrellisFormatError(
                    "holdout rate is absent from the measured truth menu"
                )
            if grade.body_rate_q256 in surface.anchor_q256:
                raise TrellisFormatError("holdout leaked into the fit")
            if values[grade.body_rate_q256] != grade.measured_dloss:
                raise TrellisFormatError(
                    "holdout grade and allocation truth disagree"
                )
            if (
                grade.measured_anchor_sha256
                != truth_anchors[grade.body_rate_q256].anchor_sha256
            ):
                raise TrellisFormatError(
                    "holdout grade names a different measured truth anchor"
                )
            if (
                grade.pre_render_recipe_identity_sha256
                != candidates_by_rate[
                    grade.body_rate_q256
                ].footprint.get("pre_render_recipe_identity_sha256")
            ):
                raise TrellisFormatError(
                    "holdout grade names a different densified wire recipe"
                )
            if (
                grade.scalar_backbone_point_sha256
                != backbones_for_unit[grade.body_rate_q256].identity_sha256
            ):
                raise TrellisFormatError(
                    "holdout grade and allocation scalar backbone disagree"
                )
            expected_prediction = surface.predict(
                grade.body_rate_q256,
                intended_use=SURFACE_USE_ALLOCATOR_COST,
                scalar_backbone=backbones_for_unit[grade.body_rate_q256],
            )
            if grade.predicted_dloss != expected_prediction:
                raise TrellisFormatError(
                    "holdout grade prediction differs from the bound surface"
                )
        normalized_truth[unit_name] = values
        normalized_truth_anchors[unit_name] = truth_anchors
        normalized_bytes[unit_name] = bytes_for_unit
        normalized_backbones[unit_name] = backbones_for_unit
        normalized_grades[unit_name] = grades

    def allocate(score: Mapping[str, Mapping[int, float]]) -> dict[str, int]:
        # Alignment makes bytes(q256) non-injective. Collapse each equal-byte
        # plateau to its lowest-loss rung before marginal allocation; silently
        # skipping zero-byte improvements would violate exact pricing.
        options: dict[str, tuple[int, ...]] = {}
        for unit_name in sorted(score):
            best_by_bytes: dict[int, int] = {}
            for rate in sorted(score[unit_name]):
                byte_count = normalized_bytes[unit_name][rate]
                incumbent = best_by_bytes.get(byte_count)
                if incumbent is None or (
                    score[unit_name][rate], rate
                ) < (
                    score[unit_name][incumbent], incumbent
                ):
                    best_by_bytes[byte_count] = rate
            options[unit_name] = tuple(
                best_by_bytes[byte_count]
                for byte_count in sorted(best_by_bytes)
            )
        chosen: dict[str, int] = {}
        for unit_name in sorted(score):
            chosen[unit_name] = options[unit_name][0]
        spent = sum(
            normalized_bytes[unit_name][rate]
            for unit_name, rate in chosen.items()
        )
        if spent > byte_budget:
            raise TrellisFormatError(
                f"the cheapest assignment already costs {spent} bytes, over "
                f"the {byte_budget}-byte budget"
            )
        while True:
            best_unit = None
            best_rate = None
            best_gain = 0.0
            for unit_name, rate in chosen.items():
                rates = options[unit_name]
                position = rates.index(rate)
                for candidate in rates[position + 1:]:
                    delta_bytes = (
                        normalized_bytes[unit_name][candidate]
                        - normalized_bytes[unit_name][rate]
                    )
                    if delta_bytes <= 0 or spent + delta_bytes > byte_budget:
                        continue
                    delta_loss = (
                        score[unit_name][rate] - score[unit_name][candidate]
                    )
                    gain = delta_loss / delta_bytes
                    if gain > best_gain:
                        best_gain = gain
                        best_unit = unit_name
                        best_rate = candidate
            if best_unit is None:
                return chosen
            spent += (
                normalized_bytes[best_unit][best_rate]
                - normalized_bytes[best_unit][chosen[best_unit]]
            )
            assert best_rate is not None
            chosen[best_unit] = best_rate

    predicted_score = {
        unit_name: {
            rate: surfaces[unit_name].predict(
                rate,
                intended_use=SURFACE_USE_ALLOCATOR_COST,
                scalar_backbone=normalized_backbones[unit_name][rate],
            )
            for rate in normalized_truth[unit_name]
        }
        for unit_name in surfaces
    }
    on_interpolated = allocate(predicted_score)
    on_truth = allocate(normalized_truth)
    loss_interpolated = sum(
        normalized_truth[unit_name][rate]
        for unit_name, rate in on_interpolated.items()
    )
    loss_truth = sum(
        normalized_truth[unit_name][rate]
        for unit_name, rate in on_truth.items()
    )
    raw_regret = (loss_interpolated / loss_truth - 1.0) * 100.0
    if raw_regret < -1e-12:
        raise TrellisFormatError(
            "the interpolated decision beat the measured-truth allocator on "
            "measured truth; the grading allocator is not an oracle for this "
            "menu, so allocation regret cannot validate it"
        )
    regret = max(0.0, raw_regret)
    agreement = sum(
        on_interpolated[unit] == on_truth[unit] for unit in on_truth
    ) / len(on_truth)
    mixing_fraction = sum(
        rate not in surfaces[unit_name].anchor_q256
        for unit_name, rate in on_interpolated.items()
    ) / len(on_interpolated)
    assignments_interpolated = tuple(sorted(on_interpolated.items()))
    assignments_truth = tuple(sorted(on_truth.items()))
    surface_bindings = tuple(
        (unit_name, surfaces[unit_name].surface_sha256)
        for unit_name in sorted(surfaces)
    )
    densified_bindings = tuple(
        (unit_name, densified[unit_name].identity_sha256)
        for unit_name in sorted(densified)
    )
    truth_bindings = tuple(
        (
            unit_name,
            _canonical_sha256(
                {
                    "schema": "prismaquant.trellis_allocation_truth.v1",
                    "unit_name": unit_name,
                    "measured_anchor_sha256_by_q256": {
                        str(rate): anchor.anchor_sha256
                        for rate, anchor in sorted(
                            normalized_truth_anchors[unit_name].items()
                        )
                    },
                }
            ),
        )
        for unit_name in sorted(normalized_truth_anchors)
    )
    grade_bindings = tuple(
        (
            unit_name,
            tuple(
                sorted(grade.grade_sha256 for grade in normalized_grades[unit_name])
            ),
        )
        for unit_name in sorted(normalized_grades)
    )
    body = {
        "schema": _REGRET_GATE_SCHEMA,
        "licensed_use": SURFACE_USE_ALLOCATOR_COST,
        "byte_budget": byte_budget,
        "max_regret_pct": bound,
        "true_loss_deciding_on_interpolated": loss_interpolated,
        "true_loss_deciding_on_truth": loss_truth,
        "regret_pct": regret,
        "assignment_agreement": agreement,
        "interpolated_assignment_fraction": mixing_fraction,
        "assignment_interpolated": dict(assignments_interpolated),
        "assignment_truth": dict(assignments_truth),
        "surface_bindings": dict(surface_bindings),
        "densified_bindings": dict(densified_bindings),
        "truth_bindings": dict(truth_bindings),
        "holdout_grade_sha256": {
            unit_name: list(digests) for unit_name, digests in grade_bindings
        },
        "bracket_agreement": True,
        "passed": regret <= bound,
        "research_only": True,
        "menu_verdict_eligible": False,
        "family_verdict_eligible": False,
        "publication_eligible": False,
    }
    return TrellisAllocationRegretGate(
        byte_budget=byte_budget,
        max_regret_pct=bound,
        regret_pct=regret,
        assignment_agreement=agreement,
        interpolated_assignment_fraction=mixing_fraction,
        true_loss_deciding_on_interpolated=loss_interpolated,
        true_loss_deciding_on_truth=loss_truth,
        assignment_interpolated=assignments_interpolated,
        assignment_truth=assignments_truth,
        surface_bindings=surface_bindings,
        densified_bindings=densified_bindings,
        truth_bindings=truth_bindings,
        holdout_grade_sha256=grade_bindings,
        bracket_agreement=True,
        passed=regret <= bound,
        gate_sha256=_canonical_sha256(body),
    )
