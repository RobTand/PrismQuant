"""Serving-lane eligibility, ATTESTED from Gridbook's packaged contract (R3).

Principle 14: a claim about another runtime is *derived from a machine-readable
table the pinned runtime publishes, or refused*. This module is the consumption
half of that rule for the CB lane. It never encodes what Gridbook does; it reads
what Gridbook *says* it does, from the packaged ``runtime_contract.json`` copy
bound to the serving pin, and reports ``UNATTESTED`` when the pinned release
publishes no such table.

Why this exists (the measured defect). The shipped DSv4 87 GB artifact carries
11 routed FP8-CB layers whose ``gate_proj``/``up_proj`` bind distinct learned
codebooks. Gridbook's persistent-B prefill lane refuses per-role split books, so
those layers take the announced expand+grouped-bridge route above the token
threshold. Nothing in the producer knew: no serving-profile lane declared a
structured ``route_status``, so persistent-B eligibility was not a gate input
and a user discovered it at serve time. Its twin on the vanilla-vLLM lane is
``units_on_fallback_route=0`` -- vacuous, because no spec declares route status
at all, so the counter is reachable only by never having looked.

The shape of the fix is therefore as important as the values:

* Verdicts are NEVER literals in this repository. A serving-profile lane
  declares which eligibility key it consults; the verdict is resolved here from
  the pinned contract.
* Absence is LOUD and typed. When the pinned contract publishes no eligibility
  table every unit resolves to :data:`ROUTE_STATUS_UNATTESTED`, and the
  provenance payload omits the backed/fallback counters entirely rather than
  reporting them as zero. A vacuous zero must be unrepresentable.
* Route status alone never removes an honestly priced rung from the allocator's
  menu (principle 1). It gates EXPORT, per artifact, per principle 9.

Vocabulary note. Principle 9's lane enum is
``backed | backed_with_serve_flag | unbacked``; this module uses it verbatim,
plus ``unattested`` for the absent-table state and, at *regime* granularity
only, ``fallback`` for a route that serves by an announced non-native path.
``allocator_candidates.ROUTE_STATUS_*`` is a DIFFERENT and older tri-state
(``backed | pending | blocked``) describing source-passthrough contracts; the
two are deliberately not unified here -- one describes a passthrough rung's
audit state, the other a lane's executed route under the pinned release.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gridbook_execution_contract import (
    GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
    QUALIFICATION_DEVICE_QUALIFIED,
    GridbookExecutionContract,
    GridbookExecutionContractError,
    parse_gridbook_execution_contract,
)


#: Schema of the eligibility table PrismaQuant consumes. Gridbook publishes it
#: inside its packaged ``runtime_contract.json`` under the ``lane_eligibility``
#: key; ``docs/design/gridbook_lane_eligibility_contract.md`` is the normative
#: specification handed to that repository.
LANE_ELIGIBILITY_V1_SCHEMA = "gridbook.lane-eligibility.v1"
# Historical public name: keep it pinned to v1 so released-pin fixtures and
# downstream consumers do not silently change schema when v2 is added.
LANE_ELIGIBILITY_SCHEMA = LANE_ELIGIBILITY_V1_SCHEMA

#: Schema of the provenance payload this module produces.
ROUTE_ATTESTATION_SCHEMA = "prismaquant.cb_route_attestation.v1"

CONTRACT_INDEX_SCHEMA = "prismaquant.gridbook_runtime_contract_index.v1"

# --- Principle 9's lane vocabulary, verbatim. -------------------------------
ROUTE_STATUS_BACKED = "backed"
ROUTE_STATUS_BACKED_WITH_SERVE_FLAG = "backed_with_serve_flag"
ROUTE_STATUS_UNBACKED = "unbacked"
#: Not one of principle 9's three: the honest state when the pinned runtime
#: publishes no eligibility table. It is a REFUSAL TO CLAIM, not a verdict.
ROUTE_STATUS_UNATTESTED = "unattested"
#: Regime granularity only. The route serves, by an announced non-native path.
#: Rolls up into a unit-level ``backed`` plus a recorded fallback regime.
ROUTE_STATUS_FALLBACK = "fallback"

LANE_ROUTE_STATUSES = frozenset({
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_UNBACKED,
})
REGIME_ROUTE_STATUSES = LANE_ROUTE_STATUSES | {ROUTE_STATUS_FALLBACK}

#: Structural classes a unit can belong to. The two take different Gridbook
#: dispatch paths and therefore different eligibility rules.
STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = frozenset({STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE})

_ASSET_DIR = Path(__file__).resolve().parent / "gridbook_runtime"
_INDEX_PATH = _ASSET_DIR / "gridbook_runtime_contract_index.json"


class GridbookLaneEligibilityError(ValueError):
    """The materialized contract or its eligibility table is malformed."""


# ---------------------------------------------------------------------------
# Structural facts of one selected unit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitStructuralFacts:
    """What the EXPORT knows about a unit, in the runtime's own vocabulary.

    Every field is a structural fact of the bytes the exporter is about to
    write, not a producer opinion: the payload family and rung name the codec,
    ``n_sub`` the sub-table split, ``role_split`` whether an expert stack binds
    more than one codebook across its projections, and the two shape fields the
    load gates. An eligibility rule may predicate on any of them.

    ``role_split`` is the fact the DSv4 defect turned on and the one no
    producer-side structure carried: it is knowable ONLY at export, after the
    per-``(qname, format)`` codebook cells resolve, which is why the gate lives
    at export rather than at allocation.
    """

    qname: str
    format_name: str
    payload_family: str
    k: int | None
    n_sub: int | None
    structure: str
    role_split: bool
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise GridbookLaneEligibilityError(
                f"{self.qname}: structure must be one of "
                f"{sorted(STRUCTURES)}, got {self.structure!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "qname": self.qname,
            "format": self.format_name,
            "payload_family": self.payload_family,
            "k": self.k,
            "n_sub": self.n_sub,
            "structure": self.structure,
            "role_split": self.role_split,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    def fact(self, name: str) -> Any:
        if name not in _PREDICABLE_FACTS:
            raise GridbookLaneEligibilityError(
                f"eligibility rule predicates on unknown fact {name!r}; "
                f"the attestable facts are {sorted(_PREDICABLE_FACTS)}")
        return getattr(self, _PREDICABLE_FACTS[name])


#: The closed set of facts a packaged eligibility rule may predicate on. A rule
#: naming anything else is a malformed contract, not a silently ignored rule --
#: an unknown predicate that no-ops would let a newer runtime's narrower rule
#: read as unconditionally eligible.
_PREDICABLE_FACTS: dict[str, str] = {
    "payload_family": "payload_family",
    "k": "k",
    "n_sub": "n_sub",
    "role_split": "role_split",
    "in_features": "in_features",
    "out_features": "out_features",
}


# ---------------------------------------------------------------------------
# The packaged table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityRule:
    """One packaged lane rule: which units, in which regime, on what route."""

    id: str
    regime: str
    structure: str
    route_status: str
    requires_serve_flags: tuple[str, ...] = ()
    predicates: tuple[tuple[str, str, Any], ...] = ()
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "EligibilityRule":
        _require_keys(
            payload, where,
            required={"id", "regime", "structure", "route_status"},
            optional={"requires_serve_flags", "predicates", "detail"},
        )
        status = str(payload["route_status"])
        if status not in REGIME_ROUTE_STATUSES:
            raise GridbookLaneEligibilityError(
                f"{where}.route_status must be one of "
                f"{sorted(REGIME_ROUTE_STATUSES)}, got {status!r}")
        structure = str(payload["structure"])
        if structure not in STRUCTURES:
            raise GridbookLaneEligibilityError(
                f"{where}.structure must be one of {sorted(STRUCTURES)}, "
                f"got {structure!r}")
        flags = tuple(str(v) for v in payload.get("requires_serve_flags", ()))
        if flags and status != ROUTE_STATUS_BACKED_WITH_SERVE_FLAG:
            raise GridbookLaneEligibilityError(
                f"{where}: requires_serve_flags is non-empty but route_status "
                f"is {status!r}; a flag-gated route is "
                f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r} by definition")
        if status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG and not flags:
            raise GridbookLaneEligibilityError(
                f"{where}: route_status is "
                f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r} but no serve flag is "
                "named; an operator cannot reach an unnamed flag")
        return cls(
            id=str(payload["id"]),
            regime=str(payload["regime"]),
            structure=structure,
            route_status=status,
            requires_serve_flags=flags,
            predicates=_parse_predicates(payload.get("predicates", ()), where),
            detail=str(payload.get("detail", "")),
        )

    def matches(self, facts: UnitStructuralFacts) -> bool:
        return all(
            _predicate_holds(facts.fact(name), op, value)
            for name, op, value in self.predicates
        )


@dataclass(frozen=True)
class EligibilityTable:
    """The packaged ``lane_eligibility`` block, or the ABSENT sentinel.

    ``present is False`` is not an error and not a zero. It is the state in
    which this repository declines to make a route claim at all.
    """

    present: bool
    runtime_version: str
    runtime_commit: str
    contract_sha256: str
    schema: str = ""
    regimes: tuple[str, ...] = ()
    rules: tuple[EligibilityRule, ...] = ()
    absent_reason: str = ""
    execution_contract: GridbookExecutionContract | None = None

    def regimes_for(self, structure: str) -> tuple[str, ...]:
        return tuple(
            regime for regime in self.regimes
            if any(rule.regime == regime and rule.structure == structure
                   for rule in self.rules)
        ) or self.regimes

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "present" if self.present else "absent",
            "gridbook_serving_version": self.runtime_version,
            "gridbook_serving_commit": self.runtime_commit,
            "contract_sha256": self.contract_sha256,
            "lane_eligibility_schema": self.schema or None,
        }
        if not self.present:
            payload["reason"] = self.absent_reason
        elif self.execution_contract is not None:
            payload["regimes"] = list(self.regimes)
            payload["platforms"] = [
                platform.id for platform in self.execution_contract.platforms
            ]
            payload["cell_ids"] = [
                cell.id for cell in self.execution_contract.cells
            ]
            payload["required_producer_qualification"] = (
                QUALIFICATION_DEVICE_QUALIFIED
            )
        else:
            payload["regimes"] = list(self.regimes)
            payload["rule_ids"] = [rule.id for rule in self.rules]
        return payload


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeRoute:
    regime: str
    route_status: str
    rule_id: str | None
    requires_serve_flags: tuple[str, ...] = ()
    detail: str = ""
    platform: str | None = None
    qualification: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "regime": self.regime,
            "route_status": self.route_status,
            "rule_id": self.rule_id,
            "requires_serve_flags": list(self.requires_serve_flags),
            "detail": self.detail,
        }
        # Preserve the released v1 payload byte-for-shape: these fields exist
        # only when a v2 execution cell supplied them.
        if self.platform is not None:
            payload["platform"] = self.platform
        if self.qualification is not None:
            payload["qualification"] = self.qualification
        return payload


@dataclass(frozen=True)
class UnitRoute:
    """One unit's resolved route status across every declared regime."""

    facts: UnitStructuralFacts
    route_status: str
    regimes: tuple[RegimeRoute, ...] = ()
    requires_serve_flags: tuple[str, ...] = ()

    @property
    def attested(self) -> bool:
        return self.route_status != ROUTE_STATUS_UNATTESTED

    @property
    def fallback_regimes(self) -> tuple[str, ...]:
        return tuple(
            r.regime for r in self.regimes
            if r.route_status == ROUTE_STATUS_FALLBACK
        )

    @property
    def dead_regimes(self) -> tuple[str, ...]:
        return tuple(
            r.regime for r in self.regimes
            if r.route_status == ROUTE_STATUS_UNBACKED
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **self.facts.as_dict(),
            "route_status": self.route_status,
            "requires_serve_flags": list(self.requires_serve_flags),
            "regime_routes": [r.as_dict() for r in self.regimes],
        }
        if self.attested:
            payload["announced_fallback_regimes"] = list(self.fallback_regimes)
            payload["no_route_regimes"] = list(self.dead_regimes)
        return payload


def resolve_unit_route(
    facts: UnitStructuralFacts,
    table: EligibilityTable,
    *,
    target_platform: str | None = None,
) -> UnitRoute:
    """Resolve one unit's route status against the pinned eligibility table.

    Absent table -> ``unattested`` with no regime detail. Never a zero, never a
    guess, and never principle 9's ``backed`` by default.
    """
    if not table.present:
        return UnitRoute(facts=facts, route_status=ROUTE_STATUS_UNATTESTED)
    if table.execution_contract is not None:
        return _resolve_v2_unit_route(
            facts,
            table,
            target_platform=target_platform,
        )

    regimes: list[RegimeRoute] = []
    for regime in table.regimes_for(facts.structure):
        best: EligibilityRule | None = None
        for rule in table.rules:
            if rule.regime != regime or rule.structure != facts.structure:
                continue
            if not rule.matches(facts):
                continue
            if best is None or _REGIME_RANK[rule.route_status] > _REGIME_RANK[
                    best.route_status]:
                best = rule
        if best is None:
            # No packaged rule claims this unit in this regime. That is the
            # fail-closed direction: an undeclared route backs nothing.
            regimes.append(RegimeRoute(
                regime=regime,
                route_status=ROUTE_STATUS_UNBACKED,
                rule_id=None,
                detail=(
                    "no packaged lane rule matches this unit's structural "
                    "facts in this regime; an undeclared route backs nothing"
                ),
            ))
            continue
        regimes.append(RegimeRoute(
            regime=regime,
            route_status=best.route_status,
            rule_id=best.id,
            requires_serve_flags=best.requires_serve_flags,
            detail=best.detail,
        ))

    backed = [
        r for r in regimes
        if r.route_status in (ROUTE_STATUS_BACKED,
                              ROUTE_STATUS_BACKED_WITH_SERVE_FLAG)
    ]
    dead = [r for r in regimes if r.route_status == ROUTE_STATUS_UNBACKED]
    flags = tuple(sorted({
        flag for r in backed for flag in r.requires_serve_flags
    }))

    if not backed or dead:
        # Principle 9: a unit with no backed route for its declared target --
        # or one the runtime refuses outright in some regime -- is UNBACKED.
        # An announced fallback is NOT this case; it serves.
        status = ROUTE_STATUS_UNBACKED
    elif flags:
        status = ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    else:
        status = ROUTE_STATUS_BACKED

    return UnitRoute(
        facts=facts,
        route_status=status,
        regimes=tuple(regimes),
        requires_serve_flags=flags,
    )


def _resolve_v2_unit_route(
    facts: UnitStructuralFacts,
    table: EligibilityTable,
    *,
    target_platform: str | None,
) -> UnitRoute:
    """Resolve v2 cells for one exact declared producer platform.

    The table is closed-world.  There is no capability inheritance, family or
    structure coercion, nearest rung, regime default, or JSON-order tie break.
    Producer/release routing requires ``device_qualified``; a compile-only
    winner is returned as an explicitly diagnosed unbacked regime so the gate
    can make that refusal categorical and non-forceable.
    """

    execution = table.execution_contract
    if execution is None:  # pragma: no cover - guarded by caller
        raise AssertionError("v2 resolver requires an execution contract")
    declared_platforms = {platform.id for platform in execution.platforms}
    if not target_platform:
        return _v2_unbacked_unit(
            facts,
            table,
            detail=(
                "the target serving profile declares no exact Gridbook "
                "platform id; v2 routes cannot be inferred from GPU names or "
                "minimum capabilities"
            ),
        )
    if target_platform not in declared_platforms:
        return _v2_unbacked_unit(
            facts,
            table,
            platform=target_platform,
            detail=(
                f"exact platform {target_platform!r} is not declared by the "
                "packaged v2 table; newer platforms never inherit older cells"
            ),
        )

    regimes: list[RegimeRoute] = []
    cell_facts = {
        "role_split": facts.role_split,
        "in_features": facts.in_features,
        "out_features": facts.out_features,
    }
    for regime in table.regimes:
        matches = [
            cell for cell in execution.cells
            if cell.platform == target_platform
            and cell.family == facts.payload_family
            and cell.structure == facts.structure
            and cell.regime == regime
            and facts.k is not None
            and facts.k in cell.rungs
            and cell.matches(cell_facts)
        ]
        if not matches:
            regimes.append(RegimeRoute(
                regime=regime,
                route_status=ROUTE_STATUS_UNBACKED,
                rule_id=None,
                platform=target_platform,
                detail=(
                    "closed-world v2 table has no exact platform/family/"
                    "structure/regime/rung/predicate cell for this unit"
                ),
            ))
            continue
        winning_rank = max(
            _REGIME_RANK[cell.route_status] for cell in matches
        )
        strongest = [
            cell for cell in matches
            if _REGIME_RANK[cell.route_status] == winning_rank
        ]
        if len(strongest) != 1:
            raise GridbookLaneEligibilityError(
                "ambiguous strongest v2 route cells "
                f"{[cell.id for cell in strongest]} for "
                f"{target_platform}/{facts.structure}/{regime}/"
                f"{facts.payload_family} K{facts.k}; equal-ranked overlaps "
                "are forbidden rather than resolved by JSON order"
            )
        winner = strongest[0]
        if winner.qualification != QUALIFICATION_DEVICE_QUALIFIED:
            regimes.append(RegimeRoute(
                regime=regime,
                route_status=ROUTE_STATUS_UNBACKED,
                rule_id=winner.id,
                platform=target_platform,
                qualification=winner.qualification,
                requires_serve_flags=winner.requires_serve_flags,
                detail=(
                    f"v2 winner {winner.id!r} is qualification="
                    f"{winner.qualification!r}; producer/release routing "
                    f"requires {QUALIFICATION_DEVICE_QUALIFIED!r}"
                ),
            ))
            continue
        regimes.append(RegimeRoute(
            regime=regime,
            route_status=winner.route_status,
            rule_id=winner.id,
            platform=target_platform,
            qualification=winner.qualification,
            requires_serve_flags=winner.requires_serve_flags,
        ))

    return _roll_up_unit_route(facts, regimes)


def _v2_unbacked_unit(
    facts: UnitStructuralFacts,
    table: EligibilityTable,
    *,
    detail: str,
    platform: str | None = None,
) -> UnitRoute:
    return _roll_up_unit_route(facts, [
        RegimeRoute(
            regime=regime,
            route_status=ROUTE_STATUS_UNBACKED,
            rule_id=None,
            platform=platform,
            detail=detail,
        )
        for regime in table.regimes
    ])


def _roll_up_unit_route(
    facts: UnitStructuralFacts,
    regimes: Sequence[RegimeRoute],
) -> UnitRoute:
    backed = [
        route for route in regimes
        if route.route_status in {
            ROUTE_STATUS_BACKED,
            ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
        }
    ]
    dead = [
        route for route in regimes
        if route.route_status == ROUTE_STATUS_UNBACKED
    ]
    flags = tuple(sorted({
        flag for route in backed for flag in route.requires_serve_flags
    }))
    if not backed or dead:
        status = ROUTE_STATUS_UNBACKED
    elif flags:
        status = ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    else:
        status = ROUTE_STATUS_BACKED
    return UnitRoute(
        facts=facts,
        route_status=status,
        regimes=tuple(regimes),
        requires_serve_flags=flags,
    )


_REGIME_RANK = {
    ROUTE_STATUS_UNBACKED: 0,
    ROUTE_STATUS_FALLBACK: 1,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG: 2,
    ROUTE_STATUS_BACKED: 3,
}


# ---------------------------------------------------------------------------
# Loading the pinned contract
# ---------------------------------------------------------------------------
def load_contract_index(path: Path | None = None) -> dict[str, Any]:
    index_path = Path(path) if path is not None else _INDEX_PATH
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridbookLaneEligibilityError(
            f"cannot read {index_path}: {exc}") from exc
    if payload.get("schema") != CONTRACT_INDEX_SCHEMA:
        raise GridbookLaneEligibilityError(
            f"{index_path}: schema must be {CONTRACT_INDEX_SCHEMA!r}")
    return payload


def materialized_contract_path(version: str, *, index: Mapping[str, Any] | None = None
                               ) -> Path | None:
    """The local byte-verbatim copy of the packaged contract for *version*."""
    payload = dict(index or load_contract_index())
    for entry in payload.get("contracts", ()):
        if str(entry.get("version")) == str(version):
            return _ASSET_DIR / str(entry["path"])
    return None


def load_eligibility_table(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> EligibilityTable:
    """Load the eligibility table attested by the pinned SERVING release.

    The SERVING pin resolves this, deliberately and explicitly. PrismaQuant
    carries two Gridbook pins -- a producer pin and a serving pin, both naming
    0.8.11 since 2026-08-21 -- and
    ``serving_profiles.gridbook_runtime_version()`` reads the PRODUCER one.
    Route status is a statement about what the *serve* executes, so resolving
    it through the producer pin would attest the wrong release whenever the
    two diverge again; the lockstep test makes divergence loud, not
    impossible, and this module stays bound to the serving pin regardless.
    That exact confusion already produced one defect (the feasibility
    certifier stamped the producer pin into a serving-scoped claim).
    """
    if contract_path is not None:
        version = str(version or "")
        commit = ""
        path: Path | None = Path(contract_path)
        entry: Mapping[str, Any] = {}
    else:
        from .gridbook_serving_runtime_pin import (
            load_gridbook_serving_runtime_pin,
        )

        pin = load_gridbook_serving_runtime_pin()
        version = str(version or pin.version)
        commit = pin.commit
        index = load_contract_index()
        entry = next(
            (e for e in index.get("contracts", ())
             if str(e.get("version")) == version),
            {},
        )
        path = materialized_contract_path(version, index=index)
        if entry and str(entry.get("commit")) != commit:
            raise GridbookLaneEligibilityError(
                f"materialized contract for Gridbook {version} names commit "
                f"{entry.get('commit')!r} but the serving pin names "
                f"{commit!r}; re-materialize the packaged contract from the "
                "pinned commit rather than editing either record")

    if path is None or not path.exists():
        return EligibilityTable(
            present=False,
            runtime_version=version or "",
            runtime_commit=commit,
            contract_sha256="",
            absent_reason=(
                f"no materialized Gridbook runtime contract for pinned "
                f"serving release {version!r}. A pinned release with no "
                "contract attests nothing; add its byte-verbatim packaged "
                "runtime_contract.json to prismaquant/gridbook_runtime/ and "
                "index it. Route status stays UNATTESTED until then."
            ),
        )

    sha = _sha256(path)
    if entry and str(entry.get("sha256")) not in ("", sha):
        raise GridbookLaneEligibilityError(
            f"{path}: sha256 {sha} does not match the indexed "
            f"{entry.get('sha256')}; the materialized contract has drifted "
            "from the release it claims to be")

    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GridbookLaneEligibilityError(
            f"cannot read {path}: {exc}") from exc

    block = contract.get("lane_eligibility")
    if block is None:
        return EligibilityTable(
            present=False,
            runtime_version=version or "",
            runtime_commit=commit,
            contract_sha256=sha,
            absent_reason=(
                f"Gridbook {version} packages no 'lane_eligibility' table in "
                "runtime_contract.json, so no serving-lane route can be "
                "attested for this pin. This is a REFUSAL TO CLAIM, not a "
                "clean bill: the runtime's lane predicates (persistent-B "
                "role-split refusal, fused mid-M rung law, token-count regime "
                "thresholds, operator serve flags) exist but are not "
                "published. See docs/design/gridbook_lane_eligibility_"
                "contract.md for the table Gridbook must add."
            ),
        )

    return _parse_table(block, contract, version or "", commit, sha)


def load_published_formats(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """The pinned release's PUBLISHED format table, keyed by family prefix.

    Unlike the eligibility table, this one Gridbook 0.8.11 really does package:
    ``formats[]`` carries ``family``, ``name_pattern``, ``grid``, ``mode``,
    ``n_sub``, ``rungs``, ``layout_versions`` and ``moe_layout_versions``. So a
    unit's payload family, sub-table split and rung legality are genuinely
    DERIVED here rather than read out of ``cb_layout``'s local table -- which is
    the point of principle 14 even when the lane question stays unanswered.
    """
    path = Path(contract_path) if contract_path is not None else None
    if path is None:
        from .gridbook_serving_runtime_pin import (
            load_gridbook_serving_runtime_pin,
        )

        version = str(version or load_gridbook_serving_runtime_pin().version)
        path = materialized_contract_path(version)
    if path is None or not path.exists():
        return {}
    contract = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["family"]): dict(entry)
        for entry in contract.get("formats", ())
        if isinstance(entry, Mapping) and entry.get("family")
    }


def unit_structural_facts(
    qname: str,
    format_name: str,
    *,
    is_routed_moe: bool,
    role_split: bool,
    in_features: int,
    out_features: int,
    published_formats: Mapping[str, Mapping[str, Any]] | None = None,
) -> UnitStructuralFacts:
    """Build one unit's facts, with family/n_sub/k DERIVED from the contract.

    ``role_split`` is the fact the DSv4 defect turned on. It is True when the
    unit's expert stack binds more than one codebook across its projections --
    knowable only after the per-``(qname, format)`` codebook cells resolve, and
    therefore only at export.
    """
    if published_formats is None:
        published_formats = load_published_formats()

    family = ""
    k: int | None = None
    n_sub: int | None = None
    upper = str(format_name).upper()
    for prefix, entry in sorted(
            published_formats.items(), key=lambda kv: -len(kv[0])):
        if not upper.startswith(prefix):
            continue
        suffix = upper[len(prefix):]
        if not suffix.isdigit():
            continue
        rung = int(suffix)
        if rung not in {int(r) for r in entry.get("rungs", ())}:
            # The pinned release does not instantiate this rung. Leaving k None
            # makes every k-predicate fail closed rather than pass silently.
            family = prefix
            break
        family, k, n_sub = prefix, rung, int(entry["n_sub"])
        break
    else:
        # Not a CB payload (BF16, a SOURCE passthrough, a stock CT rung). It has
        # a lane, so it still gets facts; it just has no family/rung/n_sub.
        family = upper

    return UnitStructuralFacts(
        qname=str(qname),
        format_name=str(format_name),
        payload_family=family,
        k=k,
        n_sub=n_sub,
        structure=STRUCTURE_ROUTED_MOE if is_routed_moe else STRUCTURE_DENSE,
        role_split=bool(role_split),
        in_features=int(in_features),
        out_features=int(out_features),
    )


def _parse_table(block: Any, contract: Mapping[str, Any], version: str,
                 commit: str, sha: str
                 ) -> EligibilityTable:
    where = "runtime_contract.lane_eligibility"
    if not isinstance(block, Mapping):
        raise GridbookLaneEligibilityError(f"{where} must be a JSON object")
    schema = block.get("schema")
    if schema == GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA:
        try:
            execution = parse_gridbook_execution_contract(
                contract,
                where="runtime_contract",
            )
        except GridbookExecutionContractError as exc:
            raise GridbookLaneEligibilityError(str(exc)) from exc
        return EligibilityTable(
            present=True,
            runtime_version=version,
            runtime_commit=commit,
            contract_sha256=sha,
            schema=str(schema),
            regimes=tuple(str(regime) for regime in block["regimes"]),
            execution_contract=execution,
        )
    if schema != LANE_ELIGIBILITY_V1_SCHEMA:
        raise GridbookLaneEligibilityError(
            f"{where}.schema must be one of "
            f"{[LANE_ELIGIBILITY_V1_SCHEMA, GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA]}, "
            f"got {schema!r}"
        )
    _require_keys(
        block, where,
        required={"schema", "regimes", "lanes"},
        optional={"detail"},
    )
    regimes = tuple(str(r) for r in block["regimes"])
    if not regimes or len(set(regimes)) != len(regimes):
        raise GridbookLaneEligibilityError(
            f"{where}.regimes must be a non-empty list of unique ids")
    rules = tuple(
        EligibilityRule.from_dict(rule, f"{where}.lanes[{i}]")
        for i, rule in enumerate(block["lanes"])
    )
    for rule in rules:
        if rule.regime not in regimes:
            raise GridbookLaneEligibilityError(
                f"{where}.lanes[{rule.id!r}].regime {rule.regime!r} is not a "
                f"declared regime {list(regimes)}")
    ids = [rule.id for rule in rules]
    if len(set(ids)) != len(ids):
        raise GridbookLaneEligibilityError(f"{where}.lanes ids must be unique")
    return EligibilityTable(
        present=True,
        runtime_version=version,
        runtime_commit=commit,
        contract_sha256=sha,
        schema=str(block["schema"]),
        regimes=regimes,
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
_PREDICATE_OPS = frozenset({
    "equals", "in", "multiple_of", "at_least", "at_most",
})


def _parse_predicates(payload: Any, where: str
                      ) -> tuple[tuple[str, str, Any], ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise GridbookLaneEligibilityError(
            f"{where}.predicates must be a JSON array")
    out: list[tuple[str, str, Any]] = []
    for i, item in enumerate(payload):
        spot = f"{where}.predicates[{i}]"
        if not isinstance(item, Mapping):
            raise GridbookLaneEligibilityError(f"{spot} must be a JSON object")
        _require_keys(item, spot, required={"fact", "op", "value"}, optional=set())
        fact = str(item["fact"])
        if fact not in _PREDICABLE_FACTS:
            raise GridbookLaneEligibilityError(
                f"{spot}.fact {fact!r} is not an attestable structural fact; "
                f"the closed set is {sorted(_PREDICABLE_FACTS)}. An unknown "
                "predicate is a malformed contract, never a no-op rule.")
        op = str(item["op"])
        if op not in _PREDICATE_OPS:
            raise GridbookLaneEligibilityError(
                f"{spot}.op {op!r} is not one of {sorted(_PREDICATE_OPS)}")
        out.append((fact, op, item["value"]))
    return tuple(out)


def _predicate_holds(actual: Any, op: str, value: Any) -> bool:
    if actual is None:
        # A rule that predicates on a fact the unit does not have cannot claim
        # it. Fail-closed, not "unconstrained".
        return False
    if op == "equals":
        return actual == value
    if op == "in":
        return actual in list(value)
    if op == "multiple_of":
        return int(value) != 0 and int(actual) % int(value) == 0
    if op == "at_least":
        return int(actual) >= int(value)
    if op == "at_most":
        return int(actual) <= int(value)
    raise GridbookLaneEligibilityError(f"unknown predicate op {op!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_keys(payload: Mapping[str, Any], where: str, *,
                  required: set[str], optional: set[str]) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing:
        raise GridbookLaneEligibilityError(f"{where}: missing field(s) {missing}")
    if extra:
        raise GridbookLaneEligibilityError(f"{where}: unknown field(s) {extra}")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "LANE_ELIGIBILITY_V1_SCHEMA",
    "LANE_ELIGIBILITY_SCHEMA",
    "GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA",
    "ROUTE_ATTESTATION_SCHEMA",
    "CONTRACT_INDEX_SCHEMA",
    "ROUTE_STATUS_BACKED",
    "ROUTE_STATUS_BACKED_WITH_SERVE_FLAG",
    "ROUTE_STATUS_UNBACKED",
    "ROUTE_STATUS_UNATTESTED",
    "ROUTE_STATUS_FALLBACK",
    "LANE_ROUTE_STATUSES",
    "REGIME_ROUTE_STATUSES",
    "STRUCTURE_DENSE",
    "STRUCTURE_ROUTED_MOE",
    "GridbookLaneEligibilityError",
    "UnitStructuralFacts",
    "EligibilityRule",
    "EligibilityTable",
    "RegimeRoute",
    "UnitRoute",
    "resolve_unit_route",
    "load_contract_index",
    "load_eligibility_table",
    "load_published_formats",
    "materialized_contract_path",
    "unit_structural_facts",
]
