"""Continuous trellis rate surface: dense rungs from sparse measured anchors.

RESEARCH ONLY.  Nothing here is imported by ``run-pipeline.sh``, the default
format menu, or any export path.

The problem this closes
-----------------------
``tessera_formats`` addresses 761 integer E2M1 rungs and 1785 E4M3 rungs, and
``tessera_allocator`` can price and rank any of them exactly.  What was
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

from .tessera_allocator import (
    TesseraAllocatorCandidate,
    build_tessera_allocator_candidate,
    tessera_solver_candidate_menu,
)
from .tessera_formats import (
    MIN_TRELLIS_STEPS,
    SUPERBLOCK_WEIGHTS,
    TesseraFamily,
    TesseraFormatError,
    get_tessera_family,
    validate_body_rate_q256,
)

__all__ = [
    "TesseraRateSurface",
    "allocation_regret",
    "densify_rate_surface",
    "fit_rate_surface",
    "leave_one_anchor_out",
    "rate_surface_solver_menu",
    "uniform_column_schedule",
]


PROVENANCE_MEASURED = "measured"
PROVENANCE_INTERPOLATED = "interpolated"


def uniform_column_schedule(
    columns: int,
    body_rate_q256: int,
    *,
    family: str | TesseraFamily,
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

    spec = get_tessera_family(family)
    rate = validate_body_rate_q256(spec, body_rate_q256)
    if type(columns) is not int or columns <= 0:
        raise TesseraFormatError("columns must be a positive integer")
    if columns % SUPERBLOCK_WEIGHTS:
        raise TesseraFormatError(
            f"columns must be a multiple of {SUPERBLOCK_WEIGHTS}; a short "
            f"final block is legal on the wire but its rate accounting is "
            f"the caller's to declare, not this helper's to guess"
        )

    total_bits = round(rate * columns / SUPERBLOCK_WEIGHTS)
    base, remainder = divmod(total_bits, columns)
    if base < 1 or (base + (1 if remainder else 0)) > spec.bypass_rate:
        raise TesseraFormatError(
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
            raise TesseraFormatError(
                f"body rate {rate} q256 leaves {coded} coded steps in block "
                f"at column {start}; {MIN_TRELLIS_STEPS} are required"
            )
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class TesseraRateSurface:
    """One unit's monotone rate surface, interpolated between measured anchors."""

    unit_name: str
    family: str
    layout: str
    currency: str
    anchor_q256: tuple[int, ...]
    anchor_dloss: tuple[float, ...]
    anchor_stderr: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.anchor_q256) < 2:
            raise TesseraFormatError(
                "a rate surface needs at least two measured anchors; one "
                "anchor is a point, not a surface"
            )
        if not (
            len(self.anchor_q256)
            == len(self.anchor_dloss)
            == len(self.anchor_stderr)
        ):
            raise TesseraFormatError("anchor arrays must agree in length")
        for left, right in zip(self.anchor_q256, self.anchor_q256[1:]):
            if left >= right:
                raise TesseraFormatError(
                    "anchor rates must be strictly increasing"
                )
        for value in self.anchor_dloss:
            if not math.isfinite(value) or value <= 0.0:
                raise TesseraFormatError(
                    "anchor dloss must be positive and finite; a zero or "
                    "negative loss cannot be interpolated in log space"
                )
        for left, right in zip(self.anchor_dloss, self.anchor_dloss[1:]):
            if right >= left:
                raise TesseraFormatError(
                    "anchor dloss must strictly decrease as rate rises; a "
                    "non-monotone anchor set is a measurement problem, and "
                    "interpolating through it would launder it into a cost"
                )

    @property
    def q256_range(self) -> tuple[int, int]:
        return (self.anchor_q256[0], self.anchor_q256[-1])

    def predict(self, body_rate_q256: int) -> float:
        """Interpolated dloss at one rung.  Refuses to extrapolate."""

        if type(body_rate_q256) is not int:
            raise TesseraFormatError("body_rate_q256 must be an integer")
        low, high = self.q256_range
        if not low <= body_rate_q256 <= high:
            raise TesseraFormatError(
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
            raise TesseraFormatError(
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
    records: Sequence[TesseraAllocatorCandidate],
    *,
    currency: str,
) -> TesseraRateSurface:
    """Build one unit's surface from its measured anchor candidates."""

    if not isinstance(currency, str) or not currency:
        raise TesseraFormatError(
            "currency must be a nonempty string naming the objective the "
            "anchors were measured in"
        )
    materialized = list(records)
    if not materialized:
        raise TesseraFormatError("a rate surface needs anchor records")
    units = {record.unit_name for record in materialized}
    families = {record.family for record in materialized}
    layouts = {record.layout for record in materialized}
    if len(units) != 1:
        raise TesseraFormatError("anchors must describe exactly one unit")
    if len(families) != 1:
        raise TesseraFormatError("anchors must share one trellis family")
    if len(layouts) != 1:
        raise TesseraFormatError(
            "anchors must share one layout; fixed-quota and tight-offset "
            "curves are different rate axes and cannot be interpolated across"
        )
    ordered = sorted(materialized, key=lambda record: record.body_rate_q256)
    rates = tuple(record.body_rate_q256 for record in ordered)
    if len(set(rates)) != len(rates):
        raise TesseraFormatError("anchors must not repeat a rate")
    return TesseraRateSurface(
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


def densify_rate_surface(
    surface: TesseraRateSurface,
    shape: Sequence[int],
    *,
    q256_values: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
    schedule_for: "callable | None" = None,
    target_profile: str | None = "research",
    qname: str | None = None,
    packed_expert: bool | None = None,
    sidecar_header_bytes: int = 0,
) -> tuple[TesseraAllocatorCandidate, ...]:
    """Price a dense set of rungs from one interpolated surface.

    Each rung gets a real schedule and therefore an exact byte footprint --
    the rate is interpolated, the *bytes* never are.  ``schedule_for`` may
    override the flattest-legal default for callers holding per-column
    importance; it receives ``(columns, body_rate_q256)`` and must return a
    schedule the wire accepts.
    """

    dims = tuple(shape)
    if len(dims) != 2:
        raise TesseraFormatError("shape must be two dimensions")
    columns = dims[1]
    spec = get_tessera_family(surface.family)
    built: list[TesseraAllocatorCandidate] = []
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
            raise TesseraFormatError(
                f"rung {rate} needs alphabets for rates {sorted(missing)}; "
                f"got {sorted(alphabets)}"
            )
        built.append(
            build_tessera_allocator_candidate(
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
                variant_label=surface.provenance(rate),
            )
        )
    return tuple(built)


def rate_surface_solver_menu(
    densified: Mapping[str, Sequence[TesseraAllocatorCandidate]],
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
        raise TesseraFormatError(
            f"rate surface menu mixes objectives {sorted(declared)}; one DP "
            f"prices in one currency"
        )
    missing = set(densified) - set(currencies)
    if missing:
        raise TesseraFormatError(
            f"units {sorted(missing)} have no declared currency"
        )
    flat: list[TesseraAllocatorCandidate] = []
    for unit_name in sorted(densified):
        flat.extend(densified[unit_name])
    return tessera_solver_candidate_menu(flat, fail_on_denied=fail_on_denied)


def leave_one_anchor_out(
    surface: TesseraRateSurface,
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
        kept = TesseraRateSurface(
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
    surfaces: Mapping[str, TesseraRateSurface],
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
        raise TesseraFormatError("byte_budget must be a positive integer")

    def allocate(score: Mapping[str, Mapping[int, float]]) -> dict[str, int]:
        chosen: dict[str, int] = {}
        for unit_name in sorted(score):
            rates = sorted(score[unit_name])
            chosen[unit_name] = rates[0]
        spent = sum(
            unit_bytes[unit_name][rate] for unit_name, rate in chosen.items()
        )
        if spent > byte_budget:
            raise TesseraFormatError(
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
