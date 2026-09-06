"""Hard serving constraints as a second selection axis — ultraplan P5c.

``docs/lanes/nvfp4-cb/format-speed-policy.md`` §1 specifies the production
selection problem and, until this module, deferred it:

    minimize    predicted_quality_loss(a)
    subject to  exact_whole_artifact_bytes(a) <= B
                p95_TTFT(a, workload)          <= SLO_prefill
                p95_ITL(a, workload)           <= SLO_decode_itl
                p05_TPS(a, workload)           >= SLO_decode_tps
                resident + KV + peak_scratch   <= device_budget

This module evaluates the four constraints below the byte line. What it does
NOT do is as important as what it does:

* **No lambda.** Latency never enters the objective. The objective is, and
  stays, minimum predicted Delta-loss. An assignment that misses an SLO is
  INFEASIBLE — it is removed from the candidate set, never "scored worse".
  Policy §1 is explicit ("Latency is not blended into the objective. There is
  no ``lambda``, no single phase-weighted ``serve_ms``"), and NATIVE-PARITY
  forbids the blend.
* **Prefill and decode are separate constraints**, never averaged, "because a
  format can move them in opposite directions" — which is precisely the
  FP8-CB case the audit measured: 1.44x on dense prefill and parity on
  batch-1 decode.
* **No default workload mix.** Policy §1: "no default workload mix hidden in
  the allocator". The mix is an operator input; with none supplied the
  constraint axis is inert and every code path is byte-identical to the
  pre-P5c allocator.

The aggregation model
---------------------

An **additive layer-time model** over the exact expanded assignment. For one
phase and one M-regime arena:

    predicted_phase_ms = reference_ms(arena)
                         * SUM_u  share_u * relative_unit_cost(u, arena)

and across the workload's regimes, ``SUM_regime mix_weight * predicted_ms``.
``share_u`` is unit ``u``'s share of allocated parameters, and
``relative_unit_cost`` is the dispatch-table row for
``(dispatch family of u's format, phase, regime, u's resolved serving lane)``.

Its assumptions are named, not buried. Every one of them is a reason this
output is proposal data:

* **A1 (additivity).** Whole-phase time is the sum of per-unit times. No
  overlap, no scheduler batching, no CUDA-graph effect, no attention/KV/router
  time, no non-Linear work.
* **A2 (parameter-share weighting).** A unit's share of the reference phase
  time equals its share of allocated parameters. Sound only while the phase is
  traffic-bound and units have comparable arithmetic intensity per byte; it is
  wrong wherever a unit's cost is set by something other than its weight
  bytes.
* **A3 (route locality).** A unit's relative cost depends only on
  ``(family, phase, regime, lane)`` — not on its neighbours, its position in
  the stack, or what the rest of the assignment chose.
* **A4 (regime uniformity).** One workload mix applies to every unit. Real
  serving gives different layers different M at the same instant.
* **A5 (baseline transfer).** The arena reference was measured on some model,
  shape, sequence length, and runtime; applying it to the model being
  allocated is a transfer, not a measurement of that model.
* **A6 (resident bytes).** Resident weight bytes are the exact serialized
  tensor payload the byte accounting already computes. Runtime dequantization
  buffers, allocator fragmentation and graph pools are the operator's
  ``peak_scratch`` input, not modelled here.
* **A7 (single-stream throughput identity).** ``p05_TPS = 1000 / p95_ITL_ms``.
  Exact for the batch-1 single-stream measurements the published record holds;
  it is NOT served throughput under concurrency.
* **A8 (statistic transfer).** Most published references are single-seed point
  measurements, not the streaming percentiles policy §3 requires. Where the
  SLO is a percentile and the reference is not, the mismatch is recorded as a
  caveat on the check rather than silently accepted.

Because of A1-A8 this module **cannot** and does not claim to compute served
p95s. It computes a *predicted, table-driven proposal* for them, exactly as
policy §1 frames it: "Per-layer or per-operator timing tables may generate
candidate assignments. They are not final evidence." The stamps this module
emits say so on every path, mirroring the existing
``additive_candidate_proposal_then_exact_assignment_filter`` honesty stamp.

Where it is enforced
--------------------

At **assignment level**, in the allocator's exact-payload / byte-budget
ratchet — the same place the exact byte filter already sits — and not inside
``solve_allocation``'s bits-DP. Three reasons, in order:

1. The DP's semantics for the unconstrained case must not change (they are
   pinned by test). A second DP dimension would change them for every run.
2. The aggregation is a parameter-share-weighted sum over the EXPANDED
   assignment, including super-item expansion and serving-unit promotion. The
   DP does not see that object; the exact filter does. Enforcing on the
   pre-promotion proposal would certify an assignment nobody ships.
3. Claiming honesty. The byte axis already documents itself as
   "additive candidate proposal, then exact assignment filter" with
   ``global_optimality_claimed: false``. A latency axis inside the DP would
   invite exactly the global-optimality claim the outer exact loop cannot
   support. Filtering at assignment level makes the weaker, true claim: every
   assignment the ratchet ACCEPTS is feasible on both axes; the ratchet
   searches a bounded set of probes and does not enumerate the feasible set.

Candidate-level pruning is deliberately NOT done. A per-unit bound on a
parameter-share-weighted SUM is only provable when a single unit alone
exceeds the budget, which cannot happen for any unit whose share is small —
i.e. never on a real model. A prune that fires only on degenerate inputs is
cost without benefit, and it would need the DP to carry a second price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .serve_dispatch_table import (
    ServeDispatchTable,
    dispatch_family_for_format,
)

SCHEMA = "prismaquant.serve_constraints.v1"

#: The aggregation model's identity, stamped into every artifact that carries
#: a verdict. It names the model AND its proposal status in one string, so a
#: consumer reading only the JSON cannot mistake it for served evidence.
AGGREGATION_MODEL = (
    "additive_layer_time__param_share_weighted__table_driven_proposal"
)

#: Mirrors ``allocator``'s existing solver-contract stamp. The constrained
#: solver filters at assignment level over a bounded probe set; it does not
#: enumerate the feasible set, so it claims no global optimality.
SOLVER_CONTRACT = (
    "min_predicted_dloss__subject_to_hard_byte_and_serving_constraints__"
    "assignment_level_feasibility_filter_over_probed_assignments"
)

#: Policy §1's reference rule, carried with any relative-tax number so the
#: denominator cannot drift into "sum of each unit's independently fastest
#: format" (which need not fit the byte budget at all).
RELATIVE_TAX_REFERENCE_RULE = (
    "A relative speed tax is quoted against the FASTEST GLOBALLY FEASIBLE "
    "ASSIGNMENT under the same whole-artifact byte budget, memory limits, "
    "legality rules and serving-unit coupling, defined INDEPENDENTLY for "
    "prefill and decode and never blended. Summing each unit's independently "
    "fastest format is invalid: that combination can exceed the byte budget, "
    "especially below 4.5 bpp. Any denominator narrower than that (for "
    "example the fastest assignment among the probes a search happened to "
    "evaluate) must say so in the same breath."
)

_ASSUMPTIONS = (
    "A1 additivity: whole-phase time is the sum of per-unit times; no "
    "overlap, scheduler batching, CUDA-graph effect, attention/KV/router or "
    "other non-Linear work is modelled.",
    "A2 parameter-share weighting: a unit's share of the reference phase time "
    "equals its share of allocated parameters.",
    "A3 route locality: a unit's relative cost depends only on (family, "
    "phase, M-regime, resolved serving lane).",
    "A4 regime uniformity: one workload mix applies to every unit.",
    "A5 baseline transfer: the arena reference was measured on another model "
    "/ shape / runtime; applying it here is a transfer, not a measurement.",
    "A6 resident bytes: resident weight bytes are the exact serialized tensor "
    "payload; runtime scratch is the operator's peak_scratch input.",
    "A7 single-stream identity: p05_TPS = 1000 / p95_ITL_ms; not served "
    "throughput under concurrency.",
    "A8 statistic transfer: where the SLO is a percentile and the arena "
    "reference is a point measurement, the mismatch is recorded, not erased.",
)

#: Deterministic evaluation order. Also the tie-break order for naming a
#: binding constraint, so two runs on the same inputs name the same one.
CONSTRAINT_ORDER = (
    "p95_ttft_ms",
    "p95_itl_ms",
    "p05_tps",
    "device_memory_bytes",
)


class ServeConstraintError(ValueError):
    """An operator-supplied constraint input is unusable."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkloadMix:
    """Per-phase M-regime weights. There is no default; see policy §1."""

    by_phase: Mapping[str, tuple[tuple[str, float], ...]]
    spec: str = ""

    @classmethod
    def parse(cls, spec: str | None) -> "WorkloadMix | None":
        """Parse ``phase:regime=weight`` entries separated by commas.

        Example::

            prefill:dense_prefill_1400=1.0,decode:decode_batch1=1.0

        Per-phase weights must sum to 1.0 (within 1e-6). A mix that does not
        is rejected rather than renormalized: silently rescaling an operator's
        stated workload changes the constraint they thought they set.
        """
        if spec is None or not str(spec).strip():
            return None
        raw = str(spec)
        acc: dict[str, list[tuple[str, float]]] = {}
        for chunk in raw.split(","):
            piece = chunk.strip()
            if not piece:
                continue
            if ":" not in piece or "=" not in piece:
                raise ServeConstraintError(
                    f"workload mix entry {piece!r} is not "
                    "'phase:m_regime=weight'"
                )
            phase, rest = piece.split(":", 1)
            regime, weight_s = rest.rsplit("=", 1)
            phase = phase.strip()
            regime = regime.strip()
            try:
                weight = float(weight_s)
            except ValueError:
                raise ServeConstraintError(
                    f"workload mix entry {piece!r}: weight {weight_s!r} is "
                    "not a number"
                ) from None
            if not math.isfinite(weight) or weight <= 0.0:
                raise ServeConstraintError(
                    f"workload mix entry {piece!r}: weight must be finite "
                    "and > 0"
                )
            if not phase or not regime:
                raise ServeConstraintError(
                    f"workload mix entry {piece!r}: phase and m_regime must "
                    "be non-empty"
                )
            acc.setdefault(phase, []).append((regime, weight))
        by_phase: dict[str, tuple[tuple[str, float], ...]] = {}
        for phase, entries in sorted(acc.items()):
            total = math.fsum(w for _r, w in entries)
            if abs(total - 1.0) > 1e-6:
                raise ServeConstraintError(
                    f"workload mix for phase {phase!r} sums to {total!r}, not "
                    "1.0. State the mix you mean; the allocator will not "
                    "renormalize it for you."
                )
            names = [r for r, _w in entries]
            if len(set(names)) != len(names):
                raise ServeConstraintError(
                    f"workload mix for phase {phase!r} repeats an m_regime"
                )
            by_phase[phase] = tuple(sorted(entries))
        return cls(by_phase=by_phase, spec=raw)

    def regimes(self, phase: str) -> tuple[tuple[str, float], ...]:
        return tuple(self.by_phase.get(phase, ()))

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "by_phase": {
                phase: [{"m_regime": r, "weight": w} for r, w in entries]
                for phase, entries in sorted(self.by_phase.items())
            },
        }


@dataclass(frozen=True)
class ServeSLOs:
    """The operator's hard deployment limits. All fields optional."""

    p95_ttft_ms: float | None = None
    p95_itl_ms: float | None = None
    p05_tps: float | None = None
    device_budget_bytes: int | None = None
    kv_bytes: int = 0
    peak_scratch_bytes: int = 0

    @property
    def any_latency(self) -> bool:
        return any(
            v is not None
            for v in (self.p95_ttft_ms, self.p95_itl_ms, self.p05_tps)
        )

    @property
    def any_set(self) -> bool:
        return self.any_latency or self.device_budget_bytes is not None

    def phases_constrained(self) -> tuple[str, ...]:
        out = []
        if self.p95_ttft_ms is not None:
            out.append("prefill")
        if self.p95_itl_ms is not None or self.p05_tps is not None:
            out.append("decode")
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "p95_ttft_ms": self.p95_ttft_ms,
            "p95_itl_ms": self.p95_itl_ms,
            "p05_tps": self.p05_tps,
            "device_budget_bytes": self.device_budget_bytes,
            "kv_bytes": int(self.kv_bytes),
            "peak_scratch_bytes": int(self.peak_scratch_bytes),
        }


@dataclass(frozen=True)
class ServeConstraintContext:
    """Everything the evaluator needs, resolved once per allocator run.

    ``active`` is false whenever the operator supplied no table or no SLOs.
    An inactive context must leave every code path byte-identical to the
    pre-P5c allocator, so the only thing it produces is a stamp saying the
    constraints were absent.
    """

    table: ServeDispatchTable | None = None
    mix: WorkloadMix | None = None
    slos: ServeSLOs = field(default_factory=ServeSLOs)
    inactive_reason: str = "no_dispatch_table_and_no_slos_supplied"

    @property
    def active(self) -> bool:
        return (
            self.table is not None
            and self.mix is not None
            and self.slos.any_set
        )

    def validate(self) -> None:
        """Refuse an incoherent activation up front, by name.

        Fail-closed on the combinations that would otherwise silently evaluate
        nothing: an SLO with no table or mix to price it, or a constrained
        phase the mix never weights.
        """
        if not self.slos.any_set:
            return
        if self.slos.any_latency:
            if self.table is None:
                raise ServeConstraintError(
                    "a latency SLO was supplied but no serve dispatch table "
                    "(--serve-dispatch-table). Latency constraints are priced "
                    "from measured rows; there is no built-in cost model to "
                    "fall back on, and inventing one is what P5c exists to "
                    "prevent."
                )
            if self.mix is None:
                raise ServeConstraintError(
                    "a latency SLO was supplied but no workload mix "
                    "(--serve-workload-mix). Policy §1 forbids a default "
                    "workload mix hidden in the allocator: 'Operators choose "
                    "explicit SLOs'."
                )
            for phase in self.slos.phases_constrained():
                if not self.mix.regimes(phase):
                    raise ServeConstraintError(
                        f"an SLO constrains the {phase!r} phase but the "
                        f"workload mix weights no {phase!r} M-regime"
                    )
                for regime, _w in self.mix.regimes(phase):
                    arena = self.table.arena(phase, regime)
                    if arena is None:
                        raise ServeConstraintError(
                            f"workload mix names {phase}:{regime}, which the "
                            f"dispatch table {self.table.table_id!r} does not "
                            "declare as an arena"
                        )

    def stamp_inactive(self) -> dict:
        return {
            "schema": SCHEMA,
            "active": False,
            "reason": (
                self.inactive_reason if not self.slos.any_set
                else "constraints_supplied_but_context_incomplete"
            ),
            "aggregation_model": None,
            "solver_contract": (
                "additive_candidate_proposal_then_exact_assignment_filter"
            ),
            "note": (
                "No serving constraint was evaluated. Selection is the "
                "pre-P5c objective — minimum predicted Delta-loss subject to "
                "the whole-artifact byte budget alone — and no latency or "
                "device-memory claim is implied by this artifact."
            ),
            "dispatch_table": (
                self.table.identity() if self.table is not None else None
            ),
            "workload_mix": (
                self.mix.as_dict() if self.mix is not None else None
            ),
            "slos": self.slos.as_dict(),
        }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstraintCheck:
    """One hard constraint's verdict. ``satisfied`` is the only gate."""

    name: str
    predicted: float | None
    limit: float
    units: str
    direction: str          # "<=" or ">="
    satisfied: bool
    slack: float | None     # limit - predicted (">=": predicted - limit)
    unpriced_reason: str | None = None
    caveats: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "constraint": self.name,
            "predicted": self.predicted,
            "limit": self.limit,
            "units": self.units,
            "direction": self.direction,
            "satisfied": bool(self.satisfied),
            "slack": self.slack,
            "unpriced_reason": self.unpriced_reason,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class ServeFeasibility:
    """The verdict for one concrete assignment."""

    active: bool
    feasible: bool
    checks: tuple[ConstraintCheck, ...]
    binding_constraint: str | None
    predicted: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def violations(self) -> tuple[ConstraintCheck, ...]:
        return tuple(c for c in self.checks if not c.satisfied)

    def violation_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.violations)

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "active": bool(self.active),
            "feasible": bool(self.feasible),
            "binding_constraint": self.binding_constraint,
            "checks": [c.as_dict() for c in self.checks],
            "violations": [c.as_dict() for c in self.violations],
            "predicted": dict(self.predicted),
            "coverage": dict(self.coverage),
            **dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Lane resolution
# --------------------------------------------------------------------------- #
def lane_key_for(lane: Any, arena_m: int | None) -> str:
    """Which dispatch-table ``lane`` column a unit's route reads.

    ``lane`` is a ``serving_profiles.ResolvedServingLane`` or ``None``.

    This is the audit's "the allocator should see that trade" point made
    mechanical, and it is why P5b had to land first. A rung whose fused lane
    the pinned Gridbook version does NOT instantiate is priced with its
    FALLBACK route's row — never the fused lane's — even though both rungs
    carry the same format-family name. Gridbook 0.7.0 backs FP8-CB fused
    mid-M for K in {28,32,36,40,44,48} while production permits every
    K28..K48, so on the published 27B K36..K47 ladder this distinction is
    load-bearing for five of eight rungs — permanently, since gridbook K1.2
    resolved to a ``k % 4 == 0`` format+TMA law rather than to five missing
    kernel instantiations.

    The M range matters too: the fused lane is declared for 9 <= M <= 128, so
    an arena measured at M = 1400 takes the fallback row even for a backed
    rung. An arena that does not say what M it was measured at cannot claim
    the fused lane at all.
    """
    if lane is None:
        return "native"
    if not getattr(lane, "fused_mid_m_backed", False):
        return "fallback"
    m_range = getattr(lane, "fused_mid_m_range", None)
    if arena_m is None or m_range is None:
        return "fallback"
    lo, hi = int(m_range[0]), int(m_range[1])
    return "fused_mid_m" if lo <= int(arena_m) <= hi else "fallback"


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #
def _phase_prediction(
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    phase: str,
    table: ServeDispatchTable,
    mix: WorkloadMix,
    lane_for: Callable[[str, str], Any],
) -> tuple[float | None, str | None, dict]:
    """Predicted phase milliseconds, or ``(None, reason, detail)``.

    Fail-closed everywhere: a unit with no dispatch row, an arena with no
    absolute reference, and an arena that is an isolated-operator
    microbenchmark all make the phase UNPRICED. An unpriced phase can never
    be certified feasible — "we could not price it" is not "it passed".
    """
    names = sorted(assignment)
    shares: dict[str, float] = {}
    total_params = 0
    for name in names:
        entry = stats.get(name)
        n_params = int((entry or {}).get("n_params", 0) or 0)
        if n_params > 0:
            shares[name] = float(n_params)
            total_params += n_params
    detail: dict = {
        "units_priced": 0,
        "units_without_params": len(names) - len(shares),
        "unpriced_units": [],
        "unpriced_families": [],
        "regimes": [],
        "lane_counts": {},
    }
    if total_params <= 0:
        return None, "assignment_has_no_parameterized_units", detail
    for name in shares:
        shares[name] /= float(total_params)

    phase_ms = 0.0
    missing_families: set[str] = set()
    unpriced: list[str] = []
    lane_counts: dict[str, int] = {}
    for regime, weight in mix.regimes(phase):
        arena = table.arena(phase, regime)
        if arena is None:
            return None, f"arena_not_declared:{phase}:{regime}", detail
        if arena.metric == "operator_ms":
            return None, (
                f"arena_is_isolated_operator_microbenchmark:{phase}:{regime}"
            ), detail
        reference_ms = arena.reference_ms()
        if reference_ms is None:
            return None, (
                f"arena_has_no_absolute_reference:{phase}:{regime}"
            ), detail
        weighted_rel = 0.0
        for name in sorted(shares):
            fmt = str(assignment[name])
            family = dispatch_family_for_format(fmt)
            lane_name = lane_key_for(lane_for(name, fmt), arena.m)
            lane_counts[lane_name] = lane_counts.get(lane_name, 0) + 1
            row = table.row(family, phase, regime, lane_name)
            if row is None:
                missing_families.add(f"{family}/{lane_name}")
                if len(unpriced) < 12:
                    unpriced.append(f"{name}:{fmt}->{family}/{lane_name}")
                continue
            weighted_rel += shares[name] * row.relative_unit_cost
        if missing_families:
            detail["unpriced_units"] = unpriced
            detail["unpriced_families"] = sorted(missing_families)
            detail["lane_counts"] = dict(sorted(lane_counts.items()))
            return None, (
                "no_dispatch_row_for_"
                + ",".join(sorted(missing_families))
                + f"_in_{phase}:{regime}"
            ), detail
        detail["regimes"].append({
            "m_regime": regime,
            "weight": weight,
            "m": arena.m,
            "reference_route": arena.reference_route,
            "reference_ms": reference_ms,
            "reference_metric": arena.metric,
            "reference_statistic": arena.statistic,
            "weighted_relative_unit_cost": weighted_rel,
            "predicted_ms": reference_ms * weighted_rel,
        })
        phase_ms += weight * reference_ms * weighted_rel

    detail["units_priced"] = len(shares)
    detail["lane_counts"] = dict(sorted(lane_counts.items()))
    return phase_ms, None, detail


def _statistic_caveats(detail: Mapping[str, Any], slo_statistic: str
                       ) -> tuple[str, ...]:
    """A8: name any percentile/point-measurement mismatch."""
    out: list[str] = []
    for regime in detail.get("regimes", ()):
        stat = str(regime.get("reference_statistic", ""))
        if stat != slo_statistic:
            out.append(
                f"arena {regime.get('m_regime')!r} reference statistic is "
                f"{stat!r}, but the SLO is a {slo_statistic!r} limit; the "
                "predicted value is not a percentile (assumption A8)"
            )
    return tuple(out)


def evaluate_assignment(
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    context: ServeConstraintContext,
    *,
    lane_for: Callable[[str, str], Any] | None = None,
    resident_bytes: int | None = None,
) -> ServeFeasibility:
    """Evaluate policy §1's hard serving constraints for one assignment.

    ``lane_for(name, fmt)`` returns the unit's resolved serving lane (P5b) or
    ``None``; omit it and every unit is priced on its format's ``native``
    lane, which is correct only for a menu with no CB rungs.

    ``resident_bytes`` is the exact serialized tensor payload of this
    assignment (assumption A6); required whenever a device budget is set.
    """
    if not context.active:
        return ServeFeasibility(
            active=False,
            feasible=True,
            checks=(),
            binding_constraint=None,
            predicted={},
            coverage={},
            provenance=context.stamp_inactive(),
        )

    table = context.table
    mix = context.mix
    slos = context.slos
    assert table is not None and mix is not None  # context.active
    resolve_lane = lane_for or (lambda _name, _fmt: None)

    predicted: dict[str, float | None] = {}
    coverage: dict[str, Any] = {}
    checks: list[ConstraintCheck] = []

    phase_ms: dict[str, float | None] = {}
    phase_reason: dict[str, str | None] = {}
    for phase in ("prefill", "decode"):
        if phase not in slos.phases_constrained():
            continue
        ms, reason, detail = _phase_prediction(
            assignment, stats, phase=phase, table=table, mix=mix,
            lane_for=resolve_lane,
        )
        phase_ms[phase] = ms
        phase_reason[phase] = reason
        coverage[phase] = detail

    def _latency_check(name: str, phase: str, limit: float, direction: str,
                       units: str, slo_statistic: str) -> ConstraintCheck:
        ms = phase_ms.get(phase)
        reason = phase_reason.get(phase)
        if ms is None:
            return ConstraintCheck(
                name=name, predicted=None, limit=float(limit), units=units,
                direction=direction, satisfied=False, slack=None,
                unpriced_reason=reason or "phase_not_priced",
            )
        detail = coverage.get(phase, {})
        caveats = _statistic_caveats(detail, slo_statistic)
        if direction == "<=":
            value = ms
            satisfied = value <= float(limit)
            slack = float(limit) - value
        else:
            # p05 TPS from p95 ITL, assumption A7.
            value = 1000.0 / ms if ms > 0 else float("inf")
            satisfied = value >= float(limit)
            slack = value - float(limit)
            caveats = caveats + (
                "predicted p05_TPS is the single-stream identity "
                "1000 / predicted p95_ITL_ms (assumption A7), not a measured "
                "served throughput under concurrency",
            )
        return ConstraintCheck(
            name=name, predicted=value, limit=float(limit), units=units,
            direction=direction, satisfied=satisfied, slack=slack,
            caveats=caveats,
        )

    if slos.p95_ttft_ms is not None:
        checks.append(_latency_check(
            "p95_ttft_ms", "prefill", slos.p95_ttft_ms, "<=", "ms", "p95"))
        predicted["p95_ttft_ms"] = phase_ms.get("prefill")
    if slos.p95_itl_ms is not None:
        checks.append(_latency_check(
            "p95_itl_ms", "decode", slos.p95_itl_ms, "<=", "ms", "p95"))
        predicted["p95_itl_ms"] = phase_ms.get("decode")
    if slos.p05_tps is not None:
        checks.append(_latency_check(
            "p05_tps", "decode", slos.p05_tps, ">=", "tok/s", "p05"))
        decode_ms = phase_ms.get("decode")
        predicted["p05_tps"] = (
            1000.0 / decode_ms if decode_ms else None)

    if slos.device_budget_bytes is not None:
        if resident_bytes is None:
            checks.append(ConstraintCheck(
                name="device_memory_bytes", predicted=None,
                limit=float(slos.device_budget_bytes), units="bytes",
                direction="<=", satisfied=False, slack=None,
                unpriced_reason="resident_bytes_not_supplied",
            ))
            predicted["device_memory_bytes"] = None
        else:
            total = (
                float(resident_bytes)
                + float(slos.kv_bytes)
                + float(slos.peak_scratch_bytes)
            )
            checks.append(ConstraintCheck(
                name="device_memory_bytes", predicted=total,
                limit=float(slos.device_budget_bytes), units="bytes",
                direction="<=",
                satisfied=total <= float(slos.device_budget_bytes),
                slack=float(slos.device_budget_bytes) - total,
                caveats=(
                    "resident weight bytes are the exact serialized tensor "
                    "payload; KV and peak scratch are operator inputs "
                    "(assumption A6)",
                ),
            ))
            predicted["device_memory_bytes"] = total
            coverage["memory"] = {
                "resident_bytes": int(resident_bytes),
                "kv_bytes": int(slos.kv_bytes),
                "peak_scratch_bytes": int(slos.peak_scratch_bytes),
            }

    order = {name: i for i, name in enumerate(CONSTRAINT_ORDER)}
    checks.sort(key=lambda c: (order.get(c.name, len(order)), c.name))
    feasible = all(c.satisfied for c in checks)

    # The binding constraint. Infeasible: the FIRST violated in the canonical
    # order, so the message is stable across runs. Feasible: the satisfied
    # check with the least RELATIVE slack — the one that would bind first if
    # the assignment moved — with the canonical order breaking ties.
    binding: str | None = None
    if not feasible:
        binding = next(c.name for c in checks if not c.satisfied)
    elif checks:
        def _relative_slack(c: ConstraintCheck) -> float:
            if c.slack is None or c.limit == 0:
                return float("inf")
            return abs(c.slack) / abs(float(c.limit))
        binding = min(
            checks,
            key=lambda c: (_relative_slack(c), order.get(c.name, len(order))),
        ).name

    provenance = {
        "aggregation_model": AGGREGATION_MODEL,
        "aggregation_assumptions": list(_ASSUMPTIONS),
        "solver_contract": SOLVER_CONTRACT,
        "global_optimality_claimed": False,
        "objective": "min_predicted_dloss__latency_enters_only_as_feasibility",
        "lambda_blended_objective": False,
        "evidence_status": (
            "proposal_data__table_driven__served_NATIVE_PARITY_protocol_is_"
            "the_release_gate"
        ),
        "relative_tax_reference_rule": RELATIVE_TAX_REFERENCE_RULE,
        "dispatch_table": table.identity(),
        "workload_mix": mix.as_dict(),
        "slos": slos.as_dict(),
        "constraint_order": list(CONSTRAINT_ORDER),
    }
    return ServeFeasibility(
        active=True,
        feasible=feasible,
        checks=tuple(checks),
        binding_constraint=binding,
        predicted=predicted,
        coverage=coverage,
        provenance=provenance,
    )


def evaluate_measured_assignment(
    assignment: Mapping[str, str],
    *,
    option_assignments: Mapping[tuple[str, str], Mapping[str, str]],
    resources: Mapping[tuple[str, str], Any],
    fixed_assignment: Mapping[str, str],
    fixed_resources: Any,
    slos: ServeSLOs,
    table_identity: Mapping[str, Any],
) -> ServeFeasibility:
    """Reprice the full expanded assignment using exact measured group rows.

    A whole fused/packed operator is charged once. Its measured row must
    describe precisely the selected member formats; leaf measurements never
    stand in for a fused execution. Fixed resources cover the declared fixed
    auxiliary assignment and immutable runtime work. Candidate scratch and
    activation buffers are sequential peaks, above those fixed allocations.
    This is an operator-sum proposal, never a p95 or end-to-end certificate.
    """
    if any(assignment.get(name) != fmt for name, fmt in fixed_assignment.items()):
        raise ServeConstraintError("measured runtime fixed auxiliary assignment changed")
    remaining = {name: fmt for name, fmt in assignment.items()
                 if name not in fixed_assignment}
    by_unit: dict[str, list[tuple[tuple[str, str], Mapping[str, str]]]] = {}
    for key, members in option_assignments.items():
        by_unit.setdefault(key[0], []).append((key, members))
    selected = []
    covered: set[str] = set()
    for unit, options in sorted(by_unit.items()):
        matches = [(key, members) for key, members in options
                   if members and all(remaining.get(name) == fmt
                                      for name, fmt in members.items())]
        if len(matches) != 1:
            raise ServeConstraintError(
                f"measured runtime unit {unit!r} requires exactly one matching "
                "whole-operator row after expansion/promotion")
        key, members = matches[0]
        if covered.intersection(members):
            raise ServeConstraintError("measured runtime operator rows overlap")
        covered.update(members)
        if key not in resources:
            raise ServeConstraintError(f"measured runtime row is missing: {key!r}")
        selected.append(resources[key])
    if covered != set(remaining):
        raise ServeConstraintError(
            "measured runtime assignment coverage mismatch: "
            f"unpriced={sorted(set(remaining) - covered)[:12]}")

    prefill = math.fsum([fixed_resources.prefill_ms]
                        + [row.prefill_ms for row in selected])
    decode_rows = [fixed_resources.decode_ms] + [row.decode_ms for row in selected]
    decode = (math.fsum(decode_rows)
              if all(value is not None for value in decode_rows) else None)
    resident = fixed_resources.resident_bytes + sum(row.resident_bytes for row in selected)
    activation = fixed_resources.activation_bytes + max(
        (row.activation_bytes for row in selected), default=0)
    scratch = fixed_resources.peak_scratch_bytes + max(
        (row.peak_scratch_bytes for row in selected), default=0)
    kv = fixed_resources.kv_bytes + slos.kv_bytes
    device = resident + activation + scratch + kv + slos.peak_scratch_bytes
    caveats = (
        "Sum of measured operator medians under the declared runtime/workload; "
        "this prediction cannot certify p95 TTFT, p95 ITL or an end-to-end SLO.",
        "Research proposal: validate against a fixed teacher on shared held-out "
        "samples, then run end-to-end serving and existing promotion gates.",
    )
    checks: list[ConstraintCheck] = []
    for name, value, limit, units in (
        ("operator_sum_prefill_ms", prefill, slos.p95_ttft_ms, "ms"),
        ("operator_sum_decode_ms", decode, slos.p95_itl_ms, "ms"),
        ("device_memory_bytes", device, slos.device_budget_bytes, "bytes"),
    ):
        if limit is not None:
            checks.append(ConstraintCheck(
                name=name, predicted=value, limit=limit, units=units,
                direction="<=", satisfied=value is not None and value <= limit,
                slack=limit - value if value is not None else None,
                unpriced_reason="decode_not_measured" if value is None else None,
                caveats=caveats if units == "ms" else (),
            ))
    if slos.p05_tps is not None:
        raise ServeConstraintError(
            "measured operator sums cannot certify p05 throughput; use a decode latency budget")
    feasible = all(check.satisfied for check in checks)
    binding = (next((check.name for check in checks if not check.satisfied), None)
               if not feasible else min(checks, key=lambda check:
                   (check.slack / check.limit if check.limit else float("inf"))).name
               if checks else None)
    return ServeFeasibility(
        active=True, feasible=feasible, checks=tuple(checks),
        binding_constraint=binding,
        predicted={"operator_sum_prefill_ms": prefill,
                   "operator_sum_decode_ms": decode,
                   "device_memory_bytes": device},
        coverage={"units_priced": len(selected), "members_priced": len(covered),
                  "fixed_auxiliary_units": len(fixed_assignment),
                  "memory": {"resident_bytes": resident, "activation_bytes": activation,
                             "peak_scratch_bytes": scratch, "kv_bytes": kv,
                             "operator_scratch_reserve_bytes": slos.peak_scratch_bytes,
                             "serialized_bytes": fixed_resources.serialized_bytes
                             + sum(row.serialized_bytes for row in selected)}},
        provenance={"aggregation_model": "measured_whole_operator_sum",
                    "solver_contract": "exact_discrete_runtime_frontier_then_expanded_assignment_check",
                    "global_optimality_claimed": False,
                    "objective": "min_predicted_dloss_subject_to_separate_resource_limits",
                    "lambda_blended_objective": False,
                    "evidence_status": "research_operator_sum_proposal",
                    "certifies_end_to_end_slo": False,
                    "certifies_p95": False,
                    "measured_runtime_table": dict(table_identity),
                    "slos": slos.as_dict(), "caveats": list(caveats)},
    )


def rejection_record(
    feasibility: ServeFeasibility,
    *,
    stage: str,
    target_bits: float,
    achieved_bits: float | None = None,
    dloss: float | None = None,
) -> dict:
    """One "this assignment was rejected for THIS SLO" row.

    Policy §1 requires the verdict to be recoverable: an operator must be able
    to read from the artifact which assignments the constraint axis removed
    and which limit removed them, rather than infer it from a selection that
    silently moved.
    """
    return {
        "stage": stage,
        "target_bits": float(target_bits),
        "achieved_bits": (
            float(achieved_bits) if achieved_bits is not None else None),
        "dloss": float(dloss) if dloss is not None else None,
        "binding_constraint": feasibility.binding_constraint,
        "violated_constraints": list(feasibility.violation_names()),
        "violations": [c.as_dict() for c in feasibility.violations],
        "predicted": dict(feasibility.predicted),
    }


def fastest_feasible_summary(
    probes: Sequence[Mapping[str, Any]],
    *,
    scope_note: str,
    prediction_metrics: Sequence[str] = ("p95_ttft_ms", "p95_itl_ms"),
) -> dict:
    """Per-phase fastest FEASIBLE probe, with policy §1's reference rule.

    ``probes`` are rows carrying ``label`` and ``predicted`` maps. The phases
    are summarized INDEPENDENTLY and never blended, per policy §1 ("Define the
    fastest feasible reference independently for prefill and decode").

    ``scope_note`` must state how narrow the candidate set is. The rule's
    denominator is the fastest *globally feasible* assignment; a search that
    probed twenty targets has not enumerated that set, and saying so is the
    difference between a bounded claim and a false one.
    """
    out: dict = {
        "reference_rule": RELATIVE_TAX_REFERENCE_RULE,
        "scope": scope_note,
        "per_phase": {},
    }
    for metric in prediction_metrics:
        rows = [
            p for p in probes
            if p.get("feasible")
            and isinstance(p.get("predicted", {}).get(metric), (int, float))
        ]
        if not rows:
            out["per_phase"][metric] = None
            continue
        best = min(
            rows,
            key=lambda p: (float(p["predicted"][metric]), str(p.get("label"))),
        )
        out["per_phase"][metric] = {
            "label": best.get("label"),
            "predicted": float(best["predicted"][metric]),
            "n_feasible_probes_considered": len(rows),
        }
    return out


__all__ = [
    "AGGREGATION_MODEL",
    "CONSTRAINT_ORDER",
    "ConstraintCheck",
    "RELATIVE_TAX_REFERENCE_RULE",
    "SCHEMA",
    "SOLVER_CONTRACT",
    "ServeConstraintContext",
    "ServeConstraintError",
    "ServeFeasibility",
    "ServeSLOs",
    "WorkloadMix",
    "evaluate_assignment",
    "evaluate_measured_assignment",
    "fastest_feasible_summary",
    "lane_key_for",
    "rejection_record",
]
