"""The CB export route-status gate (campaign rule R3).

Principle 9 judges serving eligibility *per artifact, at export*: the
serving-route provenance is a **gate input**, not a log. This module is that
gate for the codebook lane. It walks every selected unit, resolves the route the
pinned Gridbook serving release attests for that unit's structural facts, and:

* **refuses** the export when a unit has no backed route for its declared
  target -- unless the artifact declares a non-native target platform or
  carries an explicit per-run override, either of which is stamped;
* **records** -- per unit, and summarized on the shipcard -- a unit whose
  decode route is backed but whose above-threshold route takes an announced
  fallback. That is the measured DSv4 state and it is a recorded fact, not a
  refusal;
* **reports unattested** -- loudly, and without emitting a single
  backed/fallback counter -- when the pinned release publishes no eligibility
  table. The counters do not exist in that payload, so the
  ``units_on_fallback_route = 0`` defect (absence of evidence rendered as
  evidence of absence) is unrepresentable rather than merely discouraged.

The gate never touches the allocator's menu. An allocator that wants an
unbacked route is *reporting a serving gap*, and that signal is the point
(principle 1); what the gate refuses is shipping the bytes.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .gridbook_execution_contract import (
    QUALIFICATION_COMPILE_ONLY,
    QUALIFICATION_DEVICE_QUALIFIED,
)
from .gridbook_lane_eligibility import (
    ROUTE_ATTESTATION_SCHEMA,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_FALLBACK,
    ROUTE_STATUS_UNATTESTED,
    ROUTE_STATUS_UNBACKED,
    EligibilityTable,
    UnitRoute,
    UnitStructuralFacts,
    load_eligibility_table,
    resolve_unit_route,
)


#: Per-run escape hatch. Principle 9 permits an export to proceed over an
#: unbacked route only on an EXPLICIT override, and requires it stamped. The
#: value must be the reason, not a bare ``1``: a stamped override whose
#: rationale is "1" documents nothing, and this record is what a reviewer reads
#: when the artifact turns out to serve badly.
#: ``PQ_``-prefixed to match the adjacent route gate on this lane
#: (``PQ_ALLOW_ROUTE_PENDING``, the source-passthrough one). Deliberately a
#: DIFFERENT gate: that one governs passthrough rungs, this one lane routes.
ROUTE_OVERRIDE_ENV = "PQ_CB_ROUTE_STATUS_OVERRIDE"

#: The artifact declares it does not target a native route on this platform.
#: Distinct from the override: this is a property of the artifact's intent
#: ("built for a platform whose native lane does not exist"), not a decision to
#: ship past a gate.
NON_NATIVE_TARGET_ENV = "PQ_CB_NON_NATIVE_TARGET"


class CBRouteStatusRefusal(RuntimeError):
    """Export refused: a selected unit has no backed serving route."""


@dataclass(frozen=True)
class RouteGateVerdict:
    """The gate's result: a provenance payload plus what it decided."""

    provenance: dict[str, Any]
    refused: bool
    refusal_reason: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def attested(self) -> bool:
        return bool(self.provenance.get("attestation", {}).get(
            "status") == "present")


def evaluate_cb_route_status(
    units: Iterable[UnitStructuralFacts],
    *,
    table: EligibilityTable | None = None,
    target_profile: str = "nvfp4_cb",
    override_reason: str | None = None,
    non_native_target: str | None = None,
) -> RouteGateVerdict:
    """Resolve every unit's route and decide whether this export may proceed.

    ``override_reason`` and ``non_native_target`` default to the environment
    (:data:`ROUTE_OVERRIDE_ENV`, :data:`NON_NATIVE_TARGET_ENV`) so a driver can
    declare either without a signature change, and BOTH are stamped into the
    returned provenance whether or not they end up being used -- an override
    that silently did nothing is still a thing a reviewer must be able to see.
    """
    if table is None:
        table = load_eligibility_table()
    if override_reason is None:
        override_reason = os.environ.get(ROUTE_OVERRIDE_ENV) or None
    if non_native_target is None:
        non_native_target = os.environ.get(NON_NATIVE_TARGET_ENV) or None

    target_platform: str | None = None
    if table.execution_contract is not None:
        from .serving_profiles import load_serving_profile

        # Platform identity comes from the named, versioned serving profile.
        # It is never inferred from the host GPU and never treated as a
        # minimum capability. Unknown profiles remain a loud configuration
        # error through load_serving_profile's existing fail-closed behavior.
        target_platform = load_serving_profile(target_profile).target_platform

    routes = [
        resolve_unit_route(
            facts,
            table,
            target_platform=target_platform,
        )
        for facts in units
    ]
    attestation = table.provenance()

    base: dict[str, Any] = {
        "schema": ROUTE_ATTESTATION_SCHEMA,
        "target_profile": target_profile,
        "attestation": attestation,
        "units_total": len(routes),
        "declared_non_native_target": non_native_target,
        "override": _override_record(override_reason),
    }
    if table.execution_contract is not None:
        base.update({
            "target_platform": target_platform,
            "required_qualification": QUALIFICATION_DEVICE_QUALIFIED,
        })

    if not table.present:
        # DELIBERATE SHAPE: no backed/fallback/unbacked counters exist here.
        # The whole defect this rule closes is a zero that read as a verdict,
        # so the absent payload cannot be misread as "0 units on a fallback
        # route" -- there is no such key to misread.
        base.update({
            "route_attestation": ROUTE_STATUS_UNATTESTED,
            "units_unattested": len(routes),
            "by_unit": [route.as_dict() for route in routes],
            "unattested_detail": attestation.get("reason", ""),
        })
        warning = (
            f"CB route status is UNATTESTED for {len(routes)} unit(s): "
            f"{attestation.get('reason', '')} No route claim is made for this "
            "artifact. Report it as 'route status not attested for this lane', "
            "never as zero units on a fallback route (principle 12)."
        )
        return RouteGateVerdict(
            provenance=base, refused=False, warnings=(warning,))

    unbacked = [r for r in routes if r.route_status == ROUTE_STATUS_UNBACKED]
    compile_only = [
        route for route in routes
        if any(
            regime.qualification == QUALIFICATION_COMPILE_ONLY
            for regime in route.regimes
        )
    ]
    fallback = [r for r in routes if r.fallback_regimes]
    flagged = [
        r for r in routes
        if r.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    ]
    status_counts: Counter[str] = Counter(r.route_status for r in routes)
    regime_counts: dict[str, Counter[str]] = {}
    for route in routes:
        for regime in route.regimes:
            regime_counts.setdefault(regime.regime, Counter())[
                regime.route_status] += 1

    base.update({
        "route_attestation": "attested",
        "units_by_route_status": dict(sorted(status_counts.items())),
        "units_backed": status_counts[ROUTE_STATUS_BACKED],
        "units_backed_with_serve_flag": status_counts[
            ROUTE_STATUS_BACKED_WITH_SERVE_FLAG],
        "units_unbacked": len(unbacked),
        "units_with_announced_fallback": len(fallback),
        "requires_serve_flags": sorted({
            flag for r in flagged for flag in r.requires_serve_flags
        }),
        "by_regime": {
            regime: dict(sorted(counts.items()))
            for regime, counts in sorted(regime_counts.items())
        },
        "announced_fallback_units": sorted(
            r.facts.qname for r in fallback),
        "unbacked_units": sorted(r.facts.qname for r in unbacked),
        "by_unit": [route.as_dict() for route in routes],
    })
    if table.execution_contract is not None:
        base.update({
            "units_compile_only": len(compile_only),
            "compile_only_units": sorted(
                route.facts.qname for route in compile_only
            ),
        })

    warnings: list[str] = []
    if fallback:
        warnings.append(
            f"{len(fallback)} unit(s) ride an ANNOUNCED FALLBACK route above "
            "their regime threshold while their decode route is backed. This "
            "is a recorded state, not a refusal, but principle 12 requires it "
            "next to any bpp or KL claim for this artifact: "
            f"{sorted(r.facts.qname for r in fallback)[:6]}"
            f"{' ...' if len(fallback) > 6 else ''}"
        )
    if flagged:
        warnings.append(
            "serving these bytes on their backed route requires operator "
            f"flag(s) {sorted({f for r in flagged for f in r.requires_serve_flags})}"
            "; the flags are part of the serving contract and travel with the "
            "artifact, not a tuning hint."
        )

    if compile_only:
        # Compile-only is structural cross-compile evidence, not an unbacked
        # production route that an experiment may elect to waive. The strict
        # validation-only producer has its own immutable unreleasable policy;
        # this generic producer/release gate must never turn that evidence into
        # a releasable claim through either legacy escape hatch.
        base["unbacked_disposition"] = "compile_only_refused"
        reason = (
            f"CB export refused: {len(compile_only)} of {len(routes)} selected "
            "unit(s) resolve through compile_only Gridbook v2 cells. "
            "Producer/release routing requires exact device_qualified cells "
            "for every regime. This qualification refusal cannot be waived "
            f"by {ROUTE_OVERRIDE_ENV} or {NON_NATIVE_TARGET_ENV}.\n"
            f"{_unbacked_detail(compile_only)}"
        )
        return RouteGateVerdict(
            provenance=base,
            refused=True,
            refusal_reason=reason,
            warnings=tuple(warnings),
        )

    if not unbacked:
        return RouteGateVerdict(
            provenance=base, refused=False, warnings=tuple(warnings))

    detail = _unbacked_detail(unbacked)
    if non_native_target:
        base["unbacked_disposition"] = "declared_non_native_target"
        warnings.append(
            f"{len(unbacked)} unit(s) have NO backed route; the artifact "
            f"declares non-native target {non_native_target!r}, which is "
            "stamped. A win on a non-native kernel is not a win on the named "
            "hardware (principle 12).")
        return RouteGateVerdict(
            provenance=base, refused=False, warnings=tuple(warnings))
    if override_reason:
        base["unbacked_disposition"] = "explicit_override"
        warnings.append(
            f"{len(unbacked)} unit(s) have NO backed route; an explicit "
            f"per-run override is stamped: {override_reason!r}")
        return RouteGateVerdict(
            provenance=base, refused=False, warnings=tuple(warnings))

    base["unbacked_disposition"] = "refused"
    reason = (
        f"CB export refused: {len(unbacked)} of {len(routes)} selected unit(s) "
        f"have NO backed serving route under the pinned Gridbook release "
        f"{table.runtime_version} ({table.runtime_commit[:12]}).\n{detail}\n"
        "Principle 9 judges eligibility per artifact at export, so this fails "
        "closed. The fixes, in order of preference: give these units a rung "
        "whose route the pinned runtime attests; advance the serving pin to a "
        "release that backs them; declare a non-native target platform via "
        f"{NON_NATIVE_TARGET_ENV}; or, last, set {ROUTE_OVERRIDE_ENV} to the "
        "REASON this artifact ships anyway -- it is stamped on the shipcard "
        "and read by whoever inherits the serving problem."
    )
    return RouteGateVerdict(
        provenance=base,
        refused=True,
        refusal_reason=reason,
        warnings=tuple(warnings),
    )


def require_cb_route_status(
    units: Iterable[UnitStructuralFacts],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the gate and raise :class:`CBRouteStatusRefusal` on a refusal.

    Returns the provenance payload for stamping. The caller is expected to
    surface :attr:`RouteGateVerdict.warnings`; they are the recorded half of
    principle 9 and are not optional colour.
    """
    verdict = evaluate_cb_route_status(units, **kwargs)
    if verdict.refused:
        raise CBRouteStatusRefusal(verdict.refusal_reason)
    return verdict.provenance


def gate_cb_export_units(
    *,
    assignment: Mapping[str, str],
    quantized_targets: Iterable[str],
    routed_units: Iterable[str],
    role_split_units: Iterable[str],
    shape_of,
    allow_unbacked_route: str | None = None,
    non_native_target: str | None = None,
    target_profile: str = "nvfp4_cb",
    exporter: str = "export_nvfp4_cb",
) -> dict[str, Any]:
    """Run the route-status gate over one export's selected units.

    Both CB exporters call this with their own spellings of the same facts, so
    the verdict cannot drift between the streaming and in-memory paths.

    ``shape_of(qname)`` returns the unit's logical shape: ``(out, in)`` for a
    dense Linear, ``(n_experts, summed_rows, in)`` for a packed expert stack.
    ``role_split_units`` is the set of stacks whose projections bind more than
    one codebook -- the fact that decides persistent-B eligibility.

    Raises :class:`CBRouteStatusRefusal` when a unit has no backed route and
    neither an explicit override nor a declared non-native target is present.
    Prints the recorded half (announced fallbacks, required serve flags) to
    stderr; those are principle 9's record, not optional colour.
    """
    import sys

    from .gridbook_lane_eligibility import (
        load_published_formats,
        unit_structural_facts,
    )

    published = load_published_formats()
    routed = set(routed_units)
    split = set(role_split_units)

    facts = []
    for qname in sorted(quantized_targets):
        shape = tuple(int(v) for v in shape_of(qname))
        if len(shape) == 3:
            in_features, out_features = shape[2], shape[1]
        else:
            in_features, out_features = shape[-1], shape[0]
        facts.append(unit_structural_facts(
            qname,
            str(assignment[qname]),
            is_routed_moe=qname in routed or len(shape) == 3,
            role_split=qname in split,
            in_features=in_features,
            out_features=out_features,
            published_formats=published,
        ))

    verdict = evaluate_cb_route_status(
        facts,
        target_profile=target_profile,
        override_reason=allow_unbacked_route,
        non_native_target=non_native_target,
    )
    for warning in verdict.warnings:
        print(f"{exporter}: route-status: {warning}", file=sys.stderr)
    if verdict.refused:
        raise CBRouteStatusRefusal(verdict.refusal_reason)
    return verdict.provenance


def shipcard_route_summary(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The compact per-route census principle 12 puts next to the bpp claim.

    Mirrors the provenance's own shape rule: when the route is unattested the
    summary carries counts of *unattested* units and no route counters at all.
    """
    attestation = dict(provenance.get("attestation") or {})
    summary: dict[str, Any] = {
        "schema": ROUTE_ATTESTATION_SCHEMA,
        "route_attestation": provenance.get("route_attestation"),
        "gridbook_serving_version": attestation.get("gridbook_serving_version"),
        "gridbook_serving_commit": attestation.get("gridbook_serving_commit"),
        "contract_sha256": attestation.get("contract_sha256"),
        "units_total": provenance.get("units_total"),
    }
    if provenance.get("route_attestation") == ROUTE_STATUS_UNATTESTED:
        summary["units_unattested"] = provenance.get("units_unattested")
        summary["detail"] = provenance.get("unattested_detail", "")
        return summary
    for key in (
        "target_platform",
        "required_qualification",
        "units_by_route_status",
        "units_backed",
        "units_backed_with_serve_flag",
        "units_unbacked",
        "units_with_announced_fallback",
        "requires_serve_flags",
        "by_regime",
        "unbacked_disposition",
        "units_compile_only",
        "compile_only_units",
    ):
        if key in provenance:
            summary[key] = provenance[key]
    summary["declared_non_native_target"] = provenance.get(
        "declared_non_native_target")
    summary["override"] = provenance.get("override")
    return summary


def _override_record(reason: str | None) -> dict[str, Any] | None:
    if not reason:
        return None
    return {"env": ROUTE_OVERRIDE_ENV, "reason": reason}


def _unbacked_detail(unbacked: Sequence[UnitRoute]) -> str:
    lines = []
    for route in unbacked[:8]:
        dead = ", ".join(
            f"{r.regime}={r.route_status}" for r in route.regimes) or "none"
        lines.append(
            f"  {route.facts.qname}  [{route.facts.format_name} "
            f"n_sub={route.facts.n_sub} role_split={route.facts.role_split} "
            f"out={route.facts.out_features}]  regimes: {dead}")
    if len(unbacked) > 8:
        lines.append(f"  ... and {len(unbacked) - 8} more")
    return "\n".join(lines)


__all__ = [
    "ROUTE_OVERRIDE_ENV",
    "NON_NATIVE_TARGET_ENV",
    "CBRouteStatusRefusal",
    "RouteGateVerdict",
    "evaluate_cb_route_status",
    "require_cb_route_status",
    "shipcard_route_summary",
]
