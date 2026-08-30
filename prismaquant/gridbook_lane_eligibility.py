"""Serving-lane eligibility, ATTESTED from Gridbook's packaged contract (R3).

Principle 14: a claim about another runtime is *derived from a machine-readable
table the pinned runtime publishes, or refused*. This module is the consumption
half of that rule. It never encodes what Gridbook does; it reads what Gridbook
*says* it does, from the packaged ``runtime_contract.json`` copy bound to the
serving pin, and reports ``UNATTESTED`` when the pinned release publishes no
claim covering a unit.

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

Schema v3, and why absence carries the whole weight
---------------------------------------------------
``gridbook.lane-eligibility.v3`` is a **closed-world cell table**. It declares
``platforms``, ``regimes``, ``structures`` and a list of ``cells``, each cell
naming exactly one ``(platform, family, structure, regime)`` and the rung set it
covers -- ``rungs`` (CB codebook K) for a ``cb_product`` family, ``rungs_q256``
(body bits per 256 weights) for a ``tcq_trellis`` family. Which discriminator a
cell uses is NOT a key on the cell: it is decided by whether the cell's family
appears in ``formats[]`` with ``kind == "tcq_trellis"``, exactly as Gridbook's
own validator decides it.

The publisher's cell status vocabulary is ``backed | backed_with_serve_flag |
fallback``. There is deliberately **no ``unbacked`` cell**: Gridbook does not
enumerate what it cannot serve, so the ONLY negative signal a v3 table carries
is *absence* -- no cell names this platform, this family, this rung. This
module therefore resolves an uncovered unit to :data:`ROUTE_STATUS_UNATTESTED`
rather than inventing ``unbacked`` from silence, and the export gate fails
closed on an unattested unit whose family the contract governs. A parser that
silently admitted an unlisted rate would turn the one negative signal the table
has into no signal at all.

Scope, derived rather than typed. A unit is *in scope* when its payload family
appears in the pinned contract's ``formats[]`` table -- that is the runtime
saying "I decode these bytes", so its eligibility table is the authority for
them. BF16, a SOURCE passthrough and a stock compressed-tensors rung derive no
family, land out of scope, and are counted and reported rather than refused.
The scope test comes from the published table, never from a list typed here.

Vocabulary note. Principle 9's lane enum is
``backed | backed_with_serve_flag | unbacked``; this module uses it verbatim,
plus ``unattested`` for the no-claim state and, at *regime* granularity only,
``fallback`` for a route that serves by an announced non-native path.
``allocator_candidates.ROUTE_STATUS_*`` is a DIFFERENT and older tri-state
(``backed | pending | blocked``) describing source-passthrough contracts; the
two are deliberately not unified here -- one describes a passthrough rung's
audit state, the other a lane's executed route under the pinned release.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


#: Schema of the eligibility table PrismaQuant consumes. Gridbook publishes it
#: inside its packaged ``runtime_contract.json`` under the ``lane_eligibility``
#: key; ``docs/design/gridbook_lane_eligibility_contract.md`` is the normative
#: specification handed to that repository.
LANE_ELIGIBILITY_SCHEMA = "gridbook.lane-eligibility.v3"

#: Schema of the provenance payload this module produces. Bumped with the table
#: schema: the payload gained the scope census and the qualification /
#: activation-contract histograms, so a v1 reader would silently under-report.
ROUTE_ATTESTATION_SCHEMA = "prismaquant.cb_route_attestation.v2"

CONTRACT_INDEX_SCHEMA = "prismaquant.gridbook_runtime_contract_index.v1"

# --- Principle 9's lane vocabulary, verbatim. -------------------------------
ROUTE_STATUS_BACKED = "backed"
ROUTE_STATUS_BACKED_WITH_SERVE_FLAG = "backed_with_serve_flag"
ROUTE_STATUS_UNBACKED = "unbacked"
#: Not one of principle 9's three: the honest state when the pinned runtime
#: publishes no claim covering this unit -- because it packages no eligibility
#: table at all, or because no cell in the table it does package names this
#: platform/family/rung. It is a REFUSAL TO CLAIM, not a verdict.
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

#: The CLOSED set a packaged v3 cell may declare, mirroring Gridbook's own
#: ``_LANE_ROUTE_STATUSES`` exactly. ``unbacked`` is absent on purpose: the
#: runtime never enumerates what it refuses, so a cell claiming ``unbacked`` is
#: a table this repository must not have been handed. Accepting one would make
#: this parser laxer than the publisher's own validator.
CELL_ROUTE_STATUSES = frozenset({
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_FALLBACK,
})

#: How far a cell's claim was taken. ``compile_only`` means the kernels
#: cross-compile for that compute capability and nothing more; only
#: ``device_qualified`` means a real serve on that device loaded, dispatched
#: and generated. Both are recorded; neither is silently upgraded.
QUALIFICATION_COMPILE_ONLY = "compile_only"
QUALIFICATION_DEVICE_QUALIFIED = "device_qualified"
CELL_QUALIFICATIONS = frozenset({
    QUALIFICATION_COMPILE_ONLY,
    QUALIFICATION_DEVICE_QUALIFIED,
})

#: Structural classes a unit can belong to. The two take different Gridbook
#: dispatch paths and therefore different eligibility cells.
STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = frozenset({STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE})

#: The ``formats[].kind`` discriminator. It lives on the FORMAT row, never on a
#: lane cell -- a cell's rung vocabulary follows from its family's kind.
FORMAT_KIND_CB_PRODUCT = "cb_product"
FORMAT_KIND_TCQ_TRELLIS = "tcq_trellis"
FORMAT_KINDS = frozenset({FORMAT_KIND_CB_PRODUCT, FORMAT_KIND_TCQ_TRELLIS})

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
    ``n_sub`` the sub-table split, ``rate_q256`` a trellis unit's body bits per
    256 weights, ``role_split`` whether an expert stack binds more than one
    codebook across its projections, and the two shape fields the load gates.
    An eligibility cell may predicate on any of them.

    ``k`` and ``rate_q256`` are the two rung vocabularies and they are mutually
    exclusive by construction: a ``cb_product`` family carries a codebook ``k``,
    a ``tcq_trellis`` family carries a rate. Neither is ever a rounded bpw.
    Both stay ``None`` when the pinned release publishes no such rung, so every
    rung predicate and every cell match fails closed rather than passing on a
    rate the runtime never listed.

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
    #: Trellis body bits per 256 weights. ``None`` for every CB / passthrough /
    #: stock unit, and ``None`` for a trellis name whose rate falls outside the
    #: family's published ``reader_rate_range_q256``.
    rate_q256: int | None = None

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise GridbookLaneEligibilityError(
                f"{self.qname}: structure must be one of "
                f"{sorted(STRUCTURES)}, got {self.structure!r}")
        if self.k is not None and self.rate_q256 is not None:
            raise GridbookLaneEligibilityError(
                f"{self.qname}: a unit carries a codebook rung OR a trellis "
                f"rate, never both; got k={self.k} rate_q256={self.rate_q256}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "qname": self.qname,
            "format": self.format_name,
            "payload_family": self.payload_family,
            "k": self.k,
            "n_sub": self.n_sub,
            "rate_q256": self.rate_q256,
            "structure": self.structure,
            "role_split": self.role_split,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    def fact(self, name: str) -> Any:
        if name not in _PREDICABLE_FACTS:
            raise GridbookLaneEligibilityError(
                f"eligibility cell predicates on unknown fact {name!r}; "
                f"the attestable facts are {sorted(_PREDICABLE_FACTS)}")
        return getattr(self, _PREDICABLE_FACTS[name])


#: The closed set of facts a packaged eligibility cell may predicate on. A cell
#: naming anything else is a malformed contract, not a silently ignored cell --
#: an unknown predicate that no-ops would let a newer runtime's narrower cell
#: read as unconditionally eligible.
_PREDICABLE_FACTS: dict[str, str] = {
    "payload_family": "payload_family",
    "k": "k",
    "n_sub": "n_sub",
    "rate_q256": "rate_q256",
    "role_split": "role_split",
    "in_features": "in_features",
    "out_features": "out_features",
}


# ---------------------------------------------------------------------------
# The packaged table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityCell:
    """One packaged v3 cell: which bytes, where, in which regime, on what route.

    A cell is scoped to exactly one ``(platform, family, structure, regime)``
    and covers an explicit, non-empty rung list. It carries no prose: Gridbook's
    v3 validator refuses ``detail``/``rationale`` keys on a cell, because a gate
    cannot read prose (principle 14).
    """

    id: str
    platform: str
    family: str
    structure: str
    regime: str
    route_status: str
    qualification: str
    #: CB codebook rungs. Empty for a trellis cell.
    rungs: tuple[int, ...] = ()
    #: Trellis body bits per 256 weights. Empty for a CB cell.
    rungs_q256: tuple[int, ...] = ()
    #: The activation contract this route executes. Trellis cells only; a CB
    #: cell publishes none and this stays "".
    activation_contract: str = ""
    requires_serve_flags: tuple[str, ...] = ()
    predicates: tuple[tuple[str, str, Any], ...] = ()
    is_trellis: bool = False

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        where: str,
        *,
        trellis_families: frozenset[str],
    ) -> "EligibilityCell":
        if not isinstance(payload, Mapping):
            raise GridbookLaneEligibilityError(f"{where} must be a JSON object")
        family = str(payload.get("family", ""))
        if not family:
            raise GridbookLaneEligibilityError(
                f"{where}: cell must name a payload family")
        # The rung vocabulary follows the FAMILY's kind, exactly as Gridbook's
        # own validator dispatches it. A cell carries no ``kind`` key.
        is_trellis = family in trellis_families
        rung_key = "rungs_q256" if is_trellis else "rungs"
        required = {
            "id", "platform", "family", "structure", "regime", rung_key,
            "route_status", "qualification", "requires_serve_flags",
            "predicates",
        }
        if is_trellis:
            required.add("activation_contract")
        _require_keys(payload, where, required=required, optional=set())

        status = str(payload["route_status"])
        if status not in CELL_ROUTE_STATUSES:
            raise GridbookLaneEligibilityError(
                f"{where}.route_status must be one of "
                f"{sorted(CELL_ROUTE_STATUSES)}, got {status!r}. The runtime "
                "does not enumerate what it refuses; absence, not an "
                f"{ROUTE_STATUS_UNBACKED!r} cell, is how a v3 table says no.")
        qualification = str(payload["qualification"])
        if qualification not in CELL_QUALIFICATIONS:
            raise GridbookLaneEligibilityError(
                f"{where}.qualification must be one of "
                f"{sorted(CELL_QUALIFICATIONS)}, got {qualification!r}")
        structure = str(payload["structure"])
        if structure not in STRUCTURES:
            raise GridbookLaneEligibilityError(
                f"{where}.structure must be one of {sorted(STRUCTURES)}, "
                f"got {structure!r}")

        rungs = _parse_rungs(payload[rung_key], f"{where}.{rung_key}")

        activation_contract = ""
        if is_trellis:
            activation_contract = str(payload["activation_contract"])
            if not activation_contract:
                raise GridbookLaneEligibilityError(
                    f"{where}.activation_contract must name the contract this "
                    "route executes; an empty one attests nothing")

        flags = tuple(str(v) for v in payload["requires_serve_flags"])
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
            platform=str(payload["platform"]),
            family=family,
            structure=structure,
            regime=str(payload["regime"]),
            route_status=status,
            qualification=qualification,
            rungs=() if is_trellis else rungs,
            rungs_q256=rungs if is_trellis else (),
            activation_contract=activation_contract,
            requires_serve_flags=flags,
            predicates=_parse_predicates(payload["predicates"], where),
            is_trellis=is_trellis,
        )

    def covers_rung(self, facts: UnitStructuralFacts) -> bool:
        """Whether this cell's published rung list names the unit's rung.

        A unit whose rung is ``None`` -- an unpublished CB K, a trellis rate
        outside the family's reader range -- is covered by nothing. That is the
        point of the list: absence means unattested.
        """
        if self.is_trellis:
            return (facts.rate_q256 is not None
                    and facts.rate_q256 in self.rungs_q256)
        return facts.k is not None and facts.k in self.rungs

    def matches(self, facts: UnitStructuralFacts) -> bool:
        return all(
            _predicate_holds(facts.fact(name), op, value)
            for name, op, value in self.predicates
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "platform": self.platform,
            "family": self.family,
            "structure": self.structure,
            "regime": self.regime,
            "route_status": self.route_status,
            "qualification": self.qualification,
        }
        if self.is_trellis:
            payload["rungs_q256"] = list(self.rungs_q256)
            payload["activation_contract"] = self.activation_contract
        else:
            payload["rungs"] = list(self.rungs)
        payload["requires_serve_flags"] = list(self.requires_serve_flags)
        return payload


@dataclass(frozen=True)
class EligibilityTable:
    """The packaged ``lane_eligibility`` block, or the ABSENT sentinel.

    ``present is False`` is not an error and not a zero. It is the state in
    which this repository declines to make a route claim at all.

    ``families`` is the set of payload families the pinned contract's
    ``formats[]`` table publishes. It is the SCOPE of this table's authority:
    inside it, silence is a refusal; outside it, the runtime has said nothing
    about these bytes one way or the other and the gate reports rather than
    refuses.
    """

    present: bool
    runtime_version: str
    runtime_commit: str
    contract_sha256: str
    schema: str = ""
    platforms: tuple[str, ...] = ()
    regimes: tuple[str, ...] = ()
    structures: tuple[str, ...] = ()
    cells: tuple[EligibilityCell, ...] = ()
    families: frozenset[str] = frozenset()
    trellis_families: frozenset[str] = frozenset()
    absent_reason: str = ""

    def governs(self, family: str) -> bool:
        """Whether the pinned contract publishes a codec for this family."""
        return family in self.families

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
        else:
            payload["platforms"] = list(self.platforms)
            payload["regimes"] = list(self.regimes)
            payload["structures"] = list(self.structures)
            payload["published_families"] = sorted(self.families)
            payload["trellis_families"] = sorted(self.trellis_families)
            payload["cell_ids"] = [cell.id for cell in self.cells]
        return payload


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeRoute:
    regime: str
    route_status: str
    cell_id: str | None
    requires_serve_flags: tuple[str, ...] = ()
    qualification: str = ""
    activation_contract: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "route_status": self.route_status,
            "cell_id": self.cell_id,
            "requires_serve_flags": list(self.requires_serve_flags),
            "qualification": self.qualification or None,
            "activation_contract": self.activation_contract or None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class UnitRoute:
    """One unit's resolved route status across every declared regime."""

    facts: UnitStructuralFacts
    route_status: str
    regimes: tuple[RegimeRoute, ...] = ()
    requires_serve_flags: tuple[str, ...] = ()
    #: True when the pinned contract publishes this unit's payload family, i.e.
    #: when the eligibility table is the authority for these bytes.
    in_scope: bool = True
    #: Why no claim was made. Empty unless the status is ``unattested``.
    unattested_reason: str = ""

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
    def unattested_regimes(self) -> tuple[str, ...]:
        return tuple(
            r.regime for r in self.regimes
            if r.route_status == ROUTE_STATUS_UNATTESTED
        )

    @property
    def qualifications(self) -> tuple[str, ...]:
        return tuple(sorted({
            r.qualification for r in self.regimes if r.qualification
        }))

    @property
    def activation_contracts(self) -> tuple[str, ...]:
        return tuple(sorted({
            r.activation_contract for r in self.regimes
            if r.activation_contract
        }))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **self.facts.as_dict(),
            "route_status": self.route_status,
            "in_scope": self.in_scope,
            "requires_serve_flags": list(self.requires_serve_flags),
            "regime_routes": [r.as_dict() for r in self.regimes],
        }
        if self.attested:
            payload["announced_fallback_regimes"] = list(self.fallback_regimes)
            payload["qualifications"] = list(self.qualifications)
            payload["activation_contracts"] = list(self.activation_contracts)
        else:
            payload["unattested_reason"] = self.unattested_reason
            payload["unattested_regimes"] = list(self.unattested_regimes)
        return payload


def resolve_unit_route(
    facts: UnitStructuralFacts,
    table: EligibilityTable,
    *,
    platform: str | None = None,
) -> UnitRoute:
    """Resolve one unit's route status against the pinned eligibility table.

    Absent table -> ``unattested`` with no regime detail. Never a zero, never a
    guess, and never principle 9's ``backed`` by default.

    ``platform`` is the exact Gridbook platform id the artifact targets (the
    serving profile's ``target_platform``, e.g. ``sm_121``). v3 cells are
    platform-scoped, so resolving without one cannot name a route: a missing or
    unpublished platform yields ``unattested``, never a match-any.
    """
    if not table.present:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            unattested_reason=table.absent_reason,
        )

    if not table.governs(facts.payload_family):
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=False,
            unattested_reason=(
                f"payload family {facts.payload_family!r} is not published in "
                f"the pinned release's formats table "
                f"({sorted(table.families)}); the lane eligibility table is "
                "not the authority for these bytes and makes no claim about "
                "them either way"
            ),
        )

    if not platform:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            unattested_reason=(
                "no declared target platform; v3 cells are platform-scoped, so "
                "no route can be named without one. Declare "
                "'target_platform' on the serving profile this artifact "
                f"targets; the pinned release publishes {list(table.platforms)}"
            ),
        )
    if platform not in table.platforms:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            unattested_reason=(
                f"the pinned release publishes no lane cells for platform "
                f"{platform!r} (it publishes {list(table.platforms)}); an "
                "unpublished platform attests nothing"
            ),
        )

    candidates = [
        cell for cell in table.cells
        if cell.platform == platform
        and cell.family == facts.payload_family
        and cell.structure == facts.structure
        and cell.covers_rung(facts)
        and cell.matches(facts)
    ]

    regimes: list[RegimeRoute] = []
    for regime in table.regimes:
        best: EligibilityCell | None = None
        for cell in candidates:
            if cell.regime != regime:
                continue
            if best is None or _CELL_RANK[cell.route_status] > _CELL_RANK[
                    best.route_status]:
                best = cell
        if best is None:
            # No packaged cell names this unit in this regime. Under a
            # closed-world v3 table that is the ONLY negative signal there is,
            # and it must not be laundered into a verdict: the honest state is
            # "the runtime made no claim", and the export gate fails closed on
            # it for any family the contract governs.
            regimes.append(RegimeRoute(
                regime=regime,
                route_status=ROUTE_STATUS_UNATTESTED,
                cell_id=None,
                detail=(
                    "no packaged lane cell names this platform, family, "
                    "structure and rung in this regime; a rung the table does "
                    "not list is unattested, never admitted"
                ),
            ))
            continue
        regimes.append(RegimeRoute(
            regime=regime,
            route_status=best.route_status,
            cell_id=best.id,
            requires_serve_flags=best.requires_serve_flags,
            qualification=best.qualification,
            activation_contract=best.activation_contract,
        ))

    unclaimed = [
        r for r in regimes if r.route_status == ROUTE_STATUS_UNATTESTED
    ]
    backed = [
        r for r in regimes
        if r.route_status in (ROUTE_STATUS_BACKED,
                              ROUTE_STATUS_BACKED_WITH_SERVE_FLAG)
    ]
    flags = tuple(sorted({
        flag for r in backed for flag in r.requires_serve_flags
    }))

    if unclaimed:
        # Partial coverage is not coverage. One unclaimed regime means the
        # runtime has not said this unit serves everywhere it will be asked to.
        status = ROUTE_STATUS_UNATTESTED
        reason = (
            f"no lane cell covers regime(s) "
            f"{[r.regime for r in unclaimed]} for {facts.payload_family} "
            f"rung {facts.k if facts.k is not None else facts.rate_q256!r} on "
            f"{platform}"
        )
        return UnitRoute(
            facts=facts,
            route_status=status,
            regimes=tuple(regimes),
            unattested_reason=reason,
        )

    if not backed:
        # Every regime serves, but none natively. Principle 9: a unit with no
        # backed route for its declared target is UNBACKED. This IS attested --
        # the runtime published a fallback for each regime and nothing better.
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


#: Ranking among the cells that DO match, so the best published route wins a
#: regime. ``unattested`` is deliberately absent: it is produced by absence, is
#: never a cell status, and therefore never competes here.
_CELL_RANK = {
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

    return _parse_table(
        block, contract.get("formats", ()), version or "", commit, sha)


def load_published_formats(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """The pinned release's PUBLISHED format table, keyed by family.

    ``formats[]`` carries ``family``, ``name_pattern`` and, since contract v12,
    a ``kind`` discriminator: a ``cb_product`` row carries ``grid``/``mode``/
    ``n_sub``/``rungs``, a ``tcq_trellis`` row carries
    ``candidate_rungs_q256``/``reader_rate_range_q256``/
    ``native_terminal_q256``. A unit's payload family, sub-table split, rung
    legality and trellis rate are therefore genuinely DERIVED here rather than
    read out of a local table -- which is the point of principle 14.
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


def _name_prefix(entry: Mapping[str, Any]) -> str:
    """The literal head of a format's ``name_pattern``, e.g. ``TCQ_E2M1_R``.

    Keying on the pattern rather than on the family is what lets one resolver
    serve both kinds: a CB family IS its name prefix (``FP8_CB_K``), a trellis
    family is not (``TCQ_E2M1_R256`` names rates around a 256-weight block, and
    is never a rung of itself).
    """
    pattern = str(entry.get("name_pattern", ""))
    head, sep, _ = pattern.partition("{k}")
    if not sep:
        return ""
    return head.upper()


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
    """Build one unit's facts, with family/n_sub/k/rate DERIVED from the contract.

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
    rate_q256: int | None = None
    upper = str(format_name).upper()

    candidates = sorted(
        ((_name_prefix(entry), str(fam), entry)
         for fam, entry in published_formats.items()),
        key=lambda item: -len(item[0]),
    )
    for prefix, fam, entry in candidates:
        if not prefix or not upper.startswith(prefix):
            continue
        suffix = upper[len(prefix):]
        if not suffix.isdigit():
            continue
        value = int(suffix)
        family = fam
        if str(entry.get("kind", FORMAT_KIND_CB_PRODUCT)) == (
                FORMAT_KIND_TCQ_TRELLIS):
            lo, hi = (int(v) for v in entry["reader_rate_range_q256"])
            if lo <= value <= hi:
                rate_q256 = value
            # Outside the published reader range the rate stays None, so every
            # cell's rung list fails to cover it and the unit is unattested.
            break
        if value in {int(r) for r in entry.get("rungs", ())}:
            k, n_sub = value, int(entry["n_sub"])
        # The pinned release does not instantiate this rung: leaving k None
        # makes every k-predicate and every cell match fail closed rather than
        # pass silently.
        break
    else:
        # Not a payload the pinned contract publishes (BF16, a SOURCE
        # passthrough, a stock CT rung). It has a lane, so it still gets facts;
        # it just has no published family, rung or n_sub, and resolves outside
        # the eligibility table's scope.
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
        rate_q256=rate_q256,
    )


def _published_families(formats: Any) -> tuple[frozenset[str], frozenset[str]]:
    """``(all families, trellis families)`` from the contract's formats table."""
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise GridbookLaneEligibilityError(
            "runtime_contract.formats must be a JSON array; the lane table's "
            "rung vocabulary is decided by each family's published kind")
    families: set[str] = set()
    trellis: set[str] = set()
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise GridbookLaneEligibilityError(
                f"runtime_contract.formats[{i}] must be a JSON object")
        family = str(entry.get("family", ""))
        if not family:
            raise GridbookLaneEligibilityError(
                f"runtime_contract.formats[{i}] publishes no family")
        kind = str(entry.get("kind", FORMAT_KIND_CB_PRODUCT))
        if kind not in FORMAT_KINDS:
            raise GridbookLaneEligibilityError(
                f"runtime_contract.formats[{i}].kind {kind!r} is not one of "
                f"{sorted(FORMAT_KINDS)}")
        families.add(family)
        if kind == FORMAT_KIND_TCQ_TRELLIS:
            trellis.add(family)
    return frozenset(families), frozenset(trellis)


def _parse_table(block: Any, formats: Any, version: str, commit: str, sha: str
                 ) -> EligibilityTable:
    where = "runtime_contract.lane_eligibility"
    if not isinstance(block, Mapping):
        raise GridbookLaneEligibilityError(f"{where} must be a JSON object")
    # The schema string is checked BEFORE the key set, deliberately. An older
    # table fails both, and "missing field(s) ['cells', 'platforms']" would
    # send its reader off to add keys to a v2 block rather than to
    # re-materialize the contract from a release that publishes v3.
    if block.get("schema") != LANE_ELIGIBILITY_SCHEMA:
        raise GridbookLaneEligibilityError(
            f"{where}.schema must be {LANE_ELIGIBILITY_SCHEMA!r}, got "
            f"{block['schema']!r}. An older lane table is not a subset of this "
            "one -- v1/v2 cells are not platform-scoped and carry no rung "
            "list, so reading one here would admit every rung on every "
            "platform. Re-materialize the contract from a release that "
            "publishes the current schema.")
    _require_keys(
        block, where,
        required={"schema", "platforms", "regimes", "structures", "cells"},
        optional=set(),
    )

    platforms_block = block["platforms"]
    if not isinstance(platforms_block, Mapping) or not platforms_block:
        raise GridbookLaneEligibilityError(
            f"{where}.platforms must be a non-empty JSON object keyed by "
            "platform id")
    platforms = tuple(str(p) for p in platforms_block)

    regimes = tuple(str(r) for r in block["regimes"])
    if not regimes or len(set(regimes)) != len(regimes):
        raise GridbookLaneEligibilityError(
            f"{where}.regimes must be a non-empty list of unique ids")

    structures = tuple(str(s) for s in block["structures"])
    if not structures or len(set(structures)) != len(structures):
        raise GridbookLaneEligibilityError(
            f"{where}.structures must be a non-empty list of unique ids")
    unknown = sorted(set(structures) - STRUCTURES)
    if unknown:
        raise GridbookLaneEligibilityError(
            f"{where}.structures names {unknown}, which this repository has no "
            f"dispatch path for; the known set is {sorted(STRUCTURES)}")

    families, trellis_families = _published_families(formats)

    cells_block = block["cells"]
    if not isinstance(cells_block, Sequence) or isinstance(
            cells_block, (str, bytes)):
        raise GridbookLaneEligibilityError(f"{where}.cells must be a JSON array")
    cells = tuple(
        EligibilityCell.from_dict(
            cell, f"{where}.cells[{i}]", trellis_families=trellis_families)
        for i, cell in enumerate(cells_block)
    )

    for cell in cells:
        if cell.regime not in regimes:
            raise GridbookLaneEligibilityError(
                f"{where}.cells[{cell.id!r}].regime {cell.regime!r} is not a "
                f"declared regime {list(regimes)}")
        if cell.platform not in platforms:
            raise GridbookLaneEligibilityError(
                f"{where}.cells[{cell.id!r}].platform {cell.platform!r} is not "
                f"a declared platform {list(platforms)}")
        if cell.structure not in structures:
            raise GridbookLaneEligibilityError(
                f"{where}.cells[{cell.id!r}].structure {cell.structure!r} is "
                f"not a declared structure {list(structures)}")
        if cell.family not in families:
            raise GridbookLaneEligibilityError(
                f"{where}.cells[{cell.id!r}].family {cell.family!r} is not "
                f"published in runtime_contract.formats "
                f"({sorted(families)}); a lane cell for a codec the runtime "
                "does not publish attests a route to nothing")
    ids = [cell.id for cell in cells]
    if len(set(ids)) != len(ids):
        raise GridbookLaneEligibilityError(f"{where}.cells ids must be unique")

    return EligibilityTable(
        present=True,
        runtime_version=version,
        runtime_commit=commit,
        contract_sha256=sha,
        schema=str(block["schema"]),
        platforms=platforms,
        regimes=regimes,
        structures=structures,
        cells=cells,
        families=families,
        trellis_families=trellis_families,
    )


def _parse_rungs(payload: Any, where: str) -> tuple[int, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise GridbookLaneEligibilityError(f"{where} must be a JSON array")
    if not payload:
        raise GridbookLaneEligibilityError(
            f"{where} must name at least one rung; an empty rung list covers "
            "nothing and would silently make its cell unreachable")
    out: list[int] = []
    for i, item in enumerate(payload):
        if isinstance(item, bool) or not isinstance(item, int):
            raise GridbookLaneEligibilityError(
                f"{where}[{i}] must be an integer rung, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise GridbookLaneEligibilityError(f"{where} must not repeat a rung")
    return tuple(out)


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
        # A cell that predicates on a fact the unit does not have cannot claim
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
    "LANE_ELIGIBILITY_SCHEMA",
    "ROUTE_ATTESTATION_SCHEMA",
    "CONTRACT_INDEX_SCHEMA",
    "ROUTE_STATUS_BACKED",
    "ROUTE_STATUS_BACKED_WITH_SERVE_FLAG",
    "ROUTE_STATUS_UNBACKED",
    "ROUTE_STATUS_UNATTESTED",
    "ROUTE_STATUS_FALLBACK",
    "LANE_ROUTE_STATUSES",
    "REGIME_ROUTE_STATUSES",
    "CELL_ROUTE_STATUSES",
    "CELL_QUALIFICATIONS",
    "QUALIFICATION_COMPILE_ONLY",
    "QUALIFICATION_DEVICE_QUALIFIED",
    "FORMAT_KIND_CB_PRODUCT",
    "FORMAT_KIND_TCQ_TRELLIS",
    "FORMAT_KINDS",
    "STRUCTURE_DENSE",
    "STRUCTURE_ROUTED_MOE",
    "GridbookLaneEligibilityError",
    "UnitStructuralFacts",
    "EligibilityCell",
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
