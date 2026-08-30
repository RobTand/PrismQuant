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
rung.  This module predicts, so that the allocator sees a dense menu; the two
compose as measure -> interpolate -> allocate -> propose the next anchor where
lambda* actually landed.

What it does NOT assume
-----------------------
No parametric rate law.  A global log-linear fit is wrong on this family and
the ladder shows exactly where: E4M3 wSNR runs 11.04/16.79/22.12/27.35/29.57
dB at rates 2-6, i.e. ~5.4 dB per bit until it saturates against the scalar
E4M3 ceiling, where the last step delivers 2.22.  A straight line through
those anchors overpredicts the top of the range badly.  So interpolation is
**monotone piecewise-linear in (q256, log2 dloss) between bracketing anchors
only**, and extrapolation beyond the measured envelope is refused rather than
guessed.  Interpolation error is then a measurable quantity, not a modelling
assumption: :func:`leave_one_anchor_out` reports it, and
:func:`allocation_regret` reports the only thing that actually matters.

Gate on the decision, not the residual
--------------------------------------
The one-anchor campaign established this the expensive way: per-superblock
D(R2)->D(R3) transfer has a 0.32 log2 residual (~25% relative D) and yet
costs 0.04-0.23% allocation regret, because noise only misranks blocks that
were near-indifferent anyway.  A residual-style bar would have refused a
near-omniscient decision.  :func:`leave_one_anchor_out` is therefore a
diagnostic and :func:`allocation_regret` is the gate.

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

There is one legitimate bridge: a **measured** per-unit currency conversion.
The trellis-currency campaign (``dq-runs/trellis-currency-20260829`` on
sparklina, schema ``prismaquant.trellis.currency_gap.analysis.*``) measured
``kappa(L, rate) = D_aura / D_sse`` per Linear per rung on real weights and
established two facts this module encodes:

* a **per-Linear** scalar kappa (the geometric mean of the per-rate ratios)
  allocates within the probe-noise floor -- converting a unit's SSE surface
  by its own measured kappa is licensed;
* a **global** scalar kappa is refused by the same measurement -- its
  allocation regret is an order of magnitude worse than per-Linear.  There
  is therefore no broadcast path here: :func:`convert_surface_currency`
  takes one unit's own rate-resolved evidence and binds it to that unit by
  name.

The conversion does not weaken the menu refusal.  A converted surface passes
:func:`rate_surface_solver_menu` only because its ``currency`` string now
genuinely equals the DP's, and it carries a structured
:class:`CurrencyConversionStamp` recording the source objective, the target
objective, the kappa used, its per-rate spread, and the measurement identity
-- a gate reads the stamp, never prose.

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

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

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
    "CurrencyConversionStamp",
    "TrellisKappaEvidence",
    "TrellisRateSurface",
    "allocation_regret",
    "convert_surface_currency",
    "densify_rate_surface",
    "fit_rate_surface",
    "leave_one_anchor_out",
    "rate_surface_solver_menu",
    "uniform_column_schedule",
]


PROVENANCE_MEASURED = "measured"
PROVENANCE_INTERPOLATED = "interpolated"

# Suffix appended to a candidate's variant_label when its surface was
# currency-converted, so no candidate leaves a converted surface unmarked.
# The full structured record is the surface's `conversion` stamp.
CONVERTED_LABEL_SUFFIX = ".converted"


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
class TrellisKappaEvidence:
    """One unit's OWN measured currency ratios, rate-resolved.

    ``kappa[j]`` is ``D_target / D_source`` measured at rung
    ``rate_q256[j]`` for this unit, exactly as the currency-gap campaign
    defines it (``D_aura / D_sse`` per Linear per rung).  The conversion
    collapses the vector to the campaign's per-Linear scalar -- the
    geometric mean over rates -- itself, so a pre-collapsed scalar cannot
    be passed: rate-resolved evidence is required, both to compute the
    scalar the measured way and because the per-rate spread is the carrier
    of the conversion's own uncertainty.

    ``measurement_source`` and ``measurement_schema`` identify the analysis
    artifact the kappas came from (path plus schema string), so the stamp on
    a converted surface traces back to the measurement.
    """

    unit_name: str
    source_currency: str
    target_currency: str
    rate_q256: tuple[int, ...]
    kappa: tuple[float, ...]
    measurement_source: str
    measurement_schema: str

    def __post_init__(self) -> None:
        for label in ("unit_name", "source_currency", "target_currency",
                      "measurement_source", "measurement_schema"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise TrellisFormatError(
                    f"kappa evidence {label} must be a nonempty string"
                )
        if self.source_currency == self.target_currency:
            raise TrellisFormatError(
                "kappa evidence converts between two DIFFERENT objectives; "
                "source and target currency are the same string"
            )
        if len(self.rate_q256) < 2:
            raise TrellisFormatError(
                "kappa evidence needs ratios at two or more rates; a single "
                "number is a scalar, not rate-resolved evidence, and its "
                "per-rate spread -- the carrier of the conversion's own "
                "uncertainty -- would be unmeasurable"
            )
        if len(self.rate_q256) != len(self.kappa):
            raise TrellisFormatError(
                "kappa evidence rates and ratios must agree in length"
            )
        for left, right in zip(self.rate_q256, self.rate_q256[1:]):
            if type(left) is not int or type(right) is not int:
                raise TrellisFormatError(
                    "kappa evidence rates must be integers on the q256 axis"
                )
            if left >= right:
                raise TrellisFormatError(
                    "kappa evidence rates must be strictly increasing"
                )
        for value in self.kappa:
            if not math.isfinite(value) or value <= 0.0:
                raise TrellisFormatError(
                    "kappa evidence ratios must be positive and finite"
                )

    @property
    def kappa_geomean(self) -> float:
        """The campaign's per-Linear scalar: geometric mean over rates."""

        return float(
            math.exp(math.fsum(math.log(k) for k in self.kappa)
                     / len(self.kappa))
        )

    @property
    def kappa_log_sd(self) -> float:
        """Sample standard deviation (ddof=1) of ``log kappa`` over rates."""

        logs = [math.log(k) for k in self.kappa]
        mean = math.fsum(logs) / len(logs)
        return math.sqrt(
            math.fsum((v - mean) ** 2 for v in logs) / (len(logs) - 1)
        )

    @property
    def kappa_predictive_log_sd(self) -> float:
        """Predictive log-sd of kappa at one rung: ``s * sqrt(1 + 1/m)``.

        The measured near-zero log-log slope licenses treating the per-rate
        kappas as exchangeable draws around the unit's level; the error of
        pricing one rung with the estimated level is then the draw's own
        variance plus the level estimate's, ``s^2 * (1 + 1/m)``.  Both
        terms come from that model and the data -- no tuned constant.
        """

        return self.kappa_log_sd * math.sqrt(1.0 + 1.0 / len(self.kappa))

    @property
    def kappa_spread_maxmin(self) -> float:
        return max(self.kappa) / min(self.kappa)


@dataclass(frozen=True, slots=True)
class CurrencyConversionStamp:
    """Structured record of one surface's currency conversion.

    This is what a gate reads (never prose): which objective the anchors
    were measured in, which objective they are now denominated in, the
    per-unit kappa actually applied, its per-rate spread, and the identity
    of the measurement that produced it.
    """

    unit_name: str
    source_currency: str
    target_currency: str
    kappa: float
    kappa_log_sd: float
    kappa_predictive_log_sd: float
    kappa_spread_maxmin: float
    evidence_rate_q256: tuple[int, ...]
    evidence_kappa: tuple[float, ...]
    measurement_source: str
    measurement_schema: str

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_name": self.unit_name,
            "source_currency": self.source_currency,
            "target_currency": self.target_currency,
            "kappa": self.kappa,
            "kappa_log_sd": self.kappa_log_sd,
            "kappa_predictive_log_sd": self.kappa_predictive_log_sd,
            "kappa_spread_maxmin": self.kappa_spread_maxmin,
            "evidence_rate_q256": list(self.evidence_rate_q256),
            "evidence_kappa": list(self.evidence_kappa),
            "measurement_source": self.measurement_source,
            "measurement_schema": self.measurement_schema,
        }


@dataclass(frozen=True, slots=True)
class TrellisRateSurface:
    """One unit's monotone rate surface, interpolated between measured anchors."""

    unit_name: str
    family: str
    layout: str
    currency: str
    anchor_q256: tuple[int, ...]
    anchor_dloss: tuple[float, ...]
    anchor_stderr: tuple[float, ...]
    conversion: CurrencyConversionStamp | None = None

    def __post_init__(self) -> None:
        if len(self.anchor_q256) < 2:
            raise TrellisFormatError(
                "a rate surface needs at least two measured anchors; one "
                "anchor is a point, not a surface"
            )
        if not (
            len(self.anchor_q256)
            == len(self.anchor_dloss)
            == len(self.anchor_stderr)
        ):
            raise TrellisFormatError("anchor arrays must agree in length")
        for left, right in zip(self.anchor_q256, self.anchor_q256[1:]):
            if left >= right:
                raise TrellisFormatError(
                    "anchor rates must be strictly increasing"
                )
        for value in self.anchor_dloss:
            if not math.isfinite(value) or value <= 0.0:
                raise TrellisFormatError(
                    "anchor dloss must be positive and finite; a zero or "
                    "negative loss cannot be interpolated in log space"
                )
        for left, right in zip(self.anchor_dloss, self.anchor_dloss[1:]):
            if right >= left:
                raise TrellisFormatError(
                    "anchor dloss must strictly decrease as rate rises; a "
                    "non-monotone anchor set is a measurement problem, and "
                    "interpolating through it would launder it into a cost"
                )
        if self.conversion is not None and not isinstance(
            self.conversion, CurrencyConversionStamp
        ):
            raise TrellisFormatError(
                "conversion must be a CurrencyConversionStamp or None"
            )

    @property
    def q256_range(self) -> tuple[int, int]:
        return (self.anchor_q256[0], self.anchor_q256[-1])

    def predict(self, body_rate_q256: int) -> float:
        """Interpolated dloss at one rung.  Refuses to extrapolate."""

        if type(body_rate_q256) is not int:
            raise TrellisFormatError("body_rate_q256 must be an integer")
        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TrellisFormatError(
                f"rung {body_rate_q256} is outside the measured envelope "
                f"[{low}, {high}]; extrapolating a trellis rate surface is "
                f"refused -- measure another anchor instead"
            )
        for index, rate in enumerate(self.anchor_q256):
            if rate == body_rate_q256:
                return self.anchor_dloss[index]
        upper = next(
            index
            for index, rate in enumerate(self.anchor_q256)
            if rate > body_rate_q256
        )
        lower = upper - 1
        span = self.anchor_q256[upper] - self.anchor_q256[lower]
        weight = (body_rate_q256 - self.anchor_q256[lower]) / span
        log_low = math.log2(self.anchor_dloss[lower])
        log_high = math.log2(self.anchor_dloss[upper])
        return float(2.0 ** (log_low + weight * (log_high - log_low)))

    def predict_stderr(self, body_rate_q256: int) -> float:
        """Anchor stderr at an anchor; the wider bracketing one between them.

        Deliberately conservative: an interpolated rung inherits the larger
        of the two anchors it sits between, so uncertainty never shrinks by
        the act of interpolating.
        """

        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TrellisFormatError(
                f"rung {body_rate_q256} is outside [{low}, {high}]"
            )
        for index, rate in enumerate(self.anchor_q256):
            if rate == body_rate_q256:
                return self.anchor_stderr[index]
        upper = next(
            index
            for index, rate in enumerate(self.anchor_q256)
            if rate > body_rate_q256
        )
        return max(self.anchor_stderr[upper - 1], self.anchor_stderr[upper])

    def provenance(self, body_rate_q256: int) -> str:
        return (
            PROVENANCE_MEASURED
            if body_rate_q256 in self.anchor_q256
            else PROVENANCE_INTERPOLATED
        )


def fit_rate_surface(
    records: Sequence[TrellisAllocatorCandidate],
    *,
    currency: str,
) -> TrellisRateSurface:
    """Build one unit's surface from its measured anchor candidates."""

    if not isinstance(currency, str) or not currency:
        raise TrellisFormatError(
            "currency must be a nonempty string naming the objective the "
            "anchors were measured in"
        )
    materialized = list(records)
    if not materialized:
        raise TrellisFormatError("a rate surface needs anchor records")
    units = {record.unit_name for record in materialized}
    families = {record.family for record in materialized}
    layouts = {record.layout for record in materialized}
    if len(units) != 1:
        raise TrellisFormatError("anchors must describe exactly one unit")
    if len(families) != 1:
        raise TrellisFormatError("anchors must share one trellis family")
    if len(layouts) != 1:
        raise TrellisFormatError(
            "anchors must share one layout; fixed-quota and tight-offset "
            "curves are different rate axes and cannot be interpolated across"
        )
    ordered = sorted(materialized, key=lambda record: record.body_rate_q256)
    rates = tuple(record.body_rate_q256 for record in ordered)
    if len(set(rates)) != len(rates):
        raise TrellisFormatError("anchors must not repeat a rate")
    return TrellisRateSurface(
        unit_name=next(iter(units)),
        family=next(iter(families)),
        layout=next(iter(layouts)),
        currency=currency,
        anchor_q256=rates,
        anchor_dloss=tuple(
            float(record.predicted_dloss_mean) for record in ordered
        ),
        anchor_stderr=tuple(
            float(record.predicted_dloss_stderr) for record in ordered
        ),
    )


def convert_surface_currency(
    surface: TrellisRateSurface,
    evidence: TrellisKappaEvidence,
) -> TrellisRateSurface:
    """Re-denominate one surface into another objective, per measured kappa.

    The conversion applies the currency-gap campaign's per-Linear scalar:
    the geometric mean over rates of this unit's OWN measured
    ``D_target / D_source`` ratios.  Anchor values scale by that kappa;
    anchor stderr propagates BOTH the scaled measurement error and the
    conversion's own uncertainty:

        stderr' = kappa * dloss * sqrt((stderr / dloss)^2 + sigma^2)

    where ``sigma = s * sqrt(1 + 1/m)`` is the predictive log-sd of kappa
    at one rung (``s`` the ddof-1 sample sd of ``log kappa`` over the
    ``m`` measured rates).  Derivation: the target-currency cost at rung r
    is ``kappa_r * D_source(r)``; the measured near-zero log-log slope of
    kappa licenses modelling the per-rate kappas as exchangeable draws
    around the unit's level, so pricing rung r with the estimated level
    errs in log space with variance ``s^2`` (the draw) plus ``s^2 / m``
    (the level estimate).  First-order propagation combines that log-sd in
    quadrature with the source measurement's relative error; it understates
    the width for extreme spreads (delta method), but is monotone in the
    spread and assumes no distribution.  The same kappa and sigma apply at
    every anchor, including rates where kappa was directly observed --
    per-rate conversion was not the measured arm; the per-Linear scalar
    was.

    Refusals, all structural (scope rules, no tuned constant):

    * evidence for a DIFFERENT unit -- kappa is per-unit by measurement;
      broadcasting one unit's scalar is the global-scalar arm the campaign
      refused outright (an order of magnitude worse allocation regret than
      per-Linear; see the ``measurement_source`` analysis artifact);
    * evidence whose source currency is not the surface's -- the ratios
      measure a different bridge;
    * a surface that is ALREADY converted -- chaining conversions would
      multiply scalars whose joint error nothing measured; convert from
      the surface denominated in the objective the evidence measured;
    * anchors outside the evidence's rate envelope -- kappa's
      near-rate-independence is a measured fact about the rates it was
      measured at, and applying it beyond them is the same extrapolation
      :meth:`TrellisRateSurface.predict` refuses (a claim inherits the
      scope of the artifact it was measured on).
    """

    if not isinstance(surface, TrellisRateSurface):
        raise TrellisFormatError("surface must be a TrellisRateSurface")
    if not isinstance(evidence, TrellisKappaEvidence):
        raise TrellisFormatError(
            "evidence must be a TrellisKappaEvidence carrying this unit's "
            "own rate-resolved kappa measurements"
        )
    if evidence.unit_name != surface.unit_name:
        raise TrellisFormatError(
            f"kappa evidence for {evidence.unit_name!r} cannot convert "
            f"{surface.unit_name!r}: currency conversion is per-unit only. "
            f"A shared (global) kappa scalar is refused by measurement -- "
            f"the currency-gap campaign found its allocation regret an "
            f"order of magnitude worse than per-Linear kappa (see "
            f"{evidence.measurement_source}, schema "
            f"{evidence.measurement_schema}). Measure this unit's own "
            f"kappa instead"
        )
    if surface.conversion is not None:
        raise TrellisFormatError(
            "surface is already currency-converted; chaining conversions "
            "would multiply scalars whose joint error nothing measured. "
            "Convert from the surface denominated in the objective the "
            "evidence actually measured"
        )
    if evidence.source_currency != surface.currency:
        raise TrellisFormatError(
            f"kappa evidence measures {evidence.source_currency!r} -> "
            f"{evidence.target_currency!r} but the surface is denominated "
            f"in {surface.currency!r}; the ratios do not price this bridge"
        )
    low, high = surface.q256_range
    if low < evidence.rate_q256[0] or high > evidence.rate_q256[-1]:
        raise TrellisFormatError(
            f"surface anchors span [{low}, {high}] q256 but kappa was "
            f"measured on [{evidence.rate_q256[0]}, "
            f"{evidence.rate_q256[-1]}]; applying a measured ratio beyond "
            f"its measured rate envelope is extrapolation and is refused "
            f"-- measure kappa at the missing rates instead"
        )
    kappa = evidence.kappa_geomean
    sigma = evidence.kappa_predictive_log_sd
    converted_dloss = tuple(
        kappa * value for value in surface.anchor_dloss
    )
    converted_stderr = tuple(
        kappa
        * dloss
        * math.sqrt((stderr / dloss) ** 2 + sigma**2)
        for dloss, stderr in zip(surface.anchor_dloss, surface.anchor_stderr)
    )
    stamp = CurrencyConversionStamp(
        unit_name=surface.unit_name,
        source_currency=surface.currency,
        target_currency=evidence.target_currency,
        kappa=kappa,
        kappa_log_sd=evidence.kappa_log_sd,
        kappa_predictive_log_sd=sigma,
        kappa_spread_maxmin=evidence.kappa_spread_maxmin,
        evidence_rate_q256=tuple(evidence.rate_q256),
        evidence_kappa=tuple(evidence.kappa),
        measurement_source=evidence.measurement_source,
        measurement_schema=evidence.measurement_schema,
    )
    return TrellisRateSurface(
        unit_name=surface.unit_name,
        family=surface.family,
        layout=surface.layout,
        currency=evidence.target_currency,
        anchor_q256=surface.anchor_q256,
        anchor_dloss=converted_dloss,
        anchor_stderr=converted_stderr,
        conversion=stamp,
    )


def densify_rate_surface(
    surface: TrellisRateSurface,
    shape: Sequence[int],
    *,
    q256_values: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    schedule_for: "callable | None" = None,
    target_profile: str | None = "research",
    qname: str | None = None,
    packed_expert: bool | None = None,
    sidecar_header_bytes: int = 0,
) -> tuple[TrellisAllocatorCandidate, ...]:
    """Price a dense set of rungs from one interpolated surface.

    Each rung gets a real schedule and therefore an exact byte footprint --
    the rate is interpolated, the *bytes* never are.  ``schedule_for`` may
    override the flattest-legal default for callers holding per-column
    importance; it receives ``(columns, body_rate_q256)`` and must return a
    schedule the wire accepts.
    """

    dims = tuple(shape)
    if len(dims) != 2:
        raise TrellisFormatError("shape must be two dimensions")
    columns = dims[1]
    spec = get_trellis_family(surface.family)
    # A candidate from a converted surface must say so: the interpolation
    # provenance keeps its measured/interpolated stamp and the label gains
    # a suffix, so no candidate leaves a conversion unmarked.  The full
    # structured record is the surface's `conversion` stamp.
    label_suffix = (
        CONVERTED_LABEL_SUFFIX if surface.conversion is not None else ""
    )
    built: list[TrellisAllocatorCandidate] = []
    for rate in sorted(set(int(value) for value in q256_values)):
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
                predicted_dloss=surface.predict(rate),
                predicted_dloss_stderr=surface.predict_stderr(rate),
                target_profile=target_profile,
                qname=qname,
                packed_expert=packed_expert,
                sidecar_header_bytes=sidecar_header_bytes,
                variant_label=surface.provenance(rate) + label_suffix,
            )
        )
    return tuple(built)


def rate_surface_solver_menu(
    densified: Mapping[str, Sequence[TrellisAllocatorCandidate]],
    *,
    currencies: Mapping[str, str],
    fail_on_denied: bool = False,
) -> dict[str, list]:
    """Hand a dense multi-unit surface to the unchanged global allocation DP.

    Refuses a menu whose units disagree about currency: a DP that ranks an
    AURA-priced unit against an output-MSE-priced one is not solving any
    stated objective.
    """

    declared = set(currencies.values())
    if len(declared) != 1:
        raise TrellisFormatError(
            f"rate surface menu mixes objectives {sorted(declared)}; one DP "
            f"prices in one currency"
        )
    missing = set(densified) - set(currencies)
    if missing:
        raise TrellisFormatError(
            f"units {sorted(missing)} have no declared currency"
        )
    flat: list[TrellisAllocatorCandidate] = []
    for unit_name in sorted(densified):
        flat.extend(densified[unit_name])
    return trellis_solver_candidate_menu(flat, fail_on_denied=fail_on_denied)


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
            currency=surface.currency,
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
        )
        rate = surface.anchor_q256[index]
        truth = surface.anchor_dloss[index]
        predicted = kept.predict(rate)
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
    truth: Mapping[str, Mapping[int, float]],
    *,
    unit_bytes: Mapping[str, Mapping[int, int]],
    byte_budget: int,
) -> dict[str, object]:
    """The gate: what does interpolating COST the decision, not the estimate?

    Allocates a byte budget across units twice -- once on the interpolated
    surface, once on measured truth -- and reports the true objective of each.
    Both allocations are scored with TRUTH, so the number is the real cost of
    deciding on interpolated values.

    Greedy marginal allocation, which is the same rule the lambda path uses.
    It is exact only for separable CONVEX per-unit curves, and real trellis
    interiors are not reliably convex -- 142 of 196 tensors have native
    fractional rates sagging above their own chord.  The regret number stays
    valid because BOTH arms run the identical allocator, so the comparison
    isolates the interpolation; it is not a claim of optimal allocation.
    """

    if type(byte_budget) is not int or byte_budget <= 0:
        raise TrellisFormatError("byte_budget must be a positive integer")

    def allocate(score: Mapping[str, Mapping[int, float]]) -> dict[str, int]:
        chosen: dict[str, int] = {}
        for unit_name in sorted(score):
            rates = sorted(score[unit_name])
            chosen[unit_name] = rates[0]
        spent = sum(
            unit_bytes[unit_name][rate] for unit_name, rate in chosen.items()
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
                rates = sorted(score[unit_name])
                position = rates.index(rate)
                for candidate in rates[position + 1:]:
                    delta_bytes = (
                        unit_bytes[unit_name][candidate]
                        - unit_bytes[unit_name][rate]
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
                unit_bytes[best_unit][best_rate]
                - unit_bytes[best_unit][chosen[best_unit]]
            )
            chosen[best_unit] = best_rate

    predicted_score = {
        unit_name: {
            rate: surfaces[unit_name].predict(rate)
            for rate in truth[unit_name]
        }
        for unit_name in surfaces
    }
    on_interpolated = allocate(predicted_score)
    on_truth = allocate(truth)
    loss_interpolated = sum(
        truth[unit_name][rate] for unit_name, rate in on_interpolated.items()
    )
    loss_truth = sum(
        truth[unit_name][rate] for unit_name, rate in on_truth.items()
    )
    return {
        "byte_budget": byte_budget,
        "true_loss_deciding_on_interpolated": loss_interpolated,
        "true_loss_deciding_on_truth": loss_truth,
        "regret_pct": (loss_interpolated / loss_truth - 1.0) * 100.0,
        "assignment_agreement": sum(
            on_interpolated[unit] == on_truth[unit] for unit in on_truth
        )
        / len(on_truth),
        "assignment_interpolated": on_interpolated,
        "assignment_truth": on_truth,
    }
