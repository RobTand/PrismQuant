"""Torch-free consumer for Gridbook v11 device-route qualification.

``formats[].producer_rungs`` attests wire-format production.  The independent
``lane_eligibility`` v2 table attests routes on one exact compute capability.
Gridbook deliberately does not attest PrismaQuant's torch.compile/CUDA-graph
policy; the strict RTX4090 lane validates the real serve log separately.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .gridbook_format_contract import (
    GridbookFormatContractError,
    gridbook_format_rungs,
    validate_gridbook_cb_rung_contract,
)


GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA = "gridbook.lane-eligibility.v2"
GRIDBOOK_LANE_REGIMES = ("decode", "batch")
GRIDBOOK_LANE_STRUCTURES = ("dense", "routed_moe")

ROUTE_STATUS_BACKED = "backed"
ROUTE_STATUS_BACKED_WITH_SERVE_FLAG = "backed_with_serve_flag"
ROUTE_STATUS_FALLBACK = "fallback"
ROUTE_STATUSES = frozenset({
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_FALLBACK,
})

QUALIFICATION_COMPILE_ONLY = "compile_only"
QUALIFICATION_DEVICE_QUALIFIED = "device_qualified"
QUALIFICATIONS = frozenset({
    QUALIFICATION_COMPILE_ONLY,
    QUALIFICATION_DEVICE_QUALIFIED,
})

_LANE_KEYS = {"schema", "platforms", "regimes", "structures", "cells"}
_PLATFORM_KEYS = {"compute_capability"}
_CELL_KEYS = {
    "id",
    "platform",
    "family",
    "structure",
    "regime",
    "rungs",
    "route_status",
    "qualification",
    "requires_serve_flags",
    "predicates",
}
_PREDICATE_KEYS = {"fact", "op", "value"}
_PREDICABLE_FACTS = frozenset({
    "role_split", "in_features", "out_features",
})
_PREDICATE_OPS = frozenset({
    "equals", "in", "multiple_of", "at_least", "at_most",
})
_ROUTE_RANK = {
    ROUTE_STATUS_FALLBACK: 0,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG: 1,
    ROUTE_STATUS_BACKED: 2,
}


class GridbookExecutionContractError(ValueError):
    """The capability-scoped Gridbook route contract is absent or invalid."""


@dataclass(frozen=True)
class GridbookPlatform:
    id: str
    device_capability: tuple[int, int]

    @property
    def capability_sm(self) -> int:
        return 10 * self.device_capability[0] + self.device_capability[1]


@dataclass(frozen=True)
class GridbookPredicate:
    fact: str
    op: str
    value: Any

    def matches(self, facts: Mapping[str, Any]) -> bool:
        actual = facts.get(self.fact)
        if actual is None:
            return False
        if self.op == "equals":
            return actual == self.value
        if self.op == "in":
            return actual in self.value
        if self.op == "multiple_of":
            return int(actual) % int(self.value) == 0
        if self.op == "at_least":
            return int(actual) >= int(self.value)
        if self.op == "at_most":
            return int(actual) <= int(self.value)
        raise AssertionError(f"unvalidated predicate operator {self.op!r}")


@dataclass(frozen=True)
class GridbookExecutionCell:
    id: str
    platform: str
    family: str
    structure: str
    regime: str
    rungs: tuple[int, ...]
    route_status: str
    qualification: str
    requires_serve_flags: tuple[str, ...]
    predicates: tuple[GridbookPredicate, ...]

    @property
    def producer_legal(self) -> bool:
        return (
            self.qualification == QUALIFICATION_DEVICE_QUALIFIED
            and self.route_status in {
                ROUTE_STATUS_BACKED,
                ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
            }
        )

    def matches(self, facts: Mapping[str, Any]) -> bool:
        return all(predicate.matches(facts) for predicate in self.predicates)


@dataclass(frozen=True)
class GridbookRegimeResolution:
    rung: int
    regime: str
    cell_id: str
    route_status: str
    qualification: str
    requires_serve_flags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "regime": self.regime,
            "cell_id": self.cell_id,
            "route_status": self.route_status,
            "qualification": self.qualification,
            "requires_serve_flags": list(self.requires_serve_flags),
        }


@dataclass(frozen=True)
class GridbookDeviceRouteAttestation:
    platform: GridbookPlatform
    family: str
    structure: str
    rungs: tuple[int, ...]
    resolutions: tuple[GridbookRegimeResolution, ...]
    requires_serve_flags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_eligibility_schema": GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
            "platform": self.platform.id,
            "device_capability": list(self.platform.device_capability),
            "family": self.family,
            "structure": self.structure,
            "rungs": list(self.rungs),
            "regime_routes": [item.as_dict() for item in self.resolutions],
            "requires_serve_flags": list(self.requires_serve_flags),
        }


@dataclass(frozen=True)
class GridbookExecutionContract:
    platforms: tuple[GridbookPlatform, ...]
    cells: tuple[GridbookExecutionCell, ...]


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise GridbookExecutionContractError(
            f"{where}: keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string_array(
    value: Any, *, where: str, nonempty: bool = False
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise GridbookExecutionContractError(f"{where} must be a JSON array")
    result = tuple(value)
    if (
        (nonempty and not result)
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise GridbookExecutionContractError(
            f"{where} must contain unique nonempty strings"
        )
    return tuple(str(item) for item in result)


def _strict_rungs(value: Any, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise GridbookExecutionContractError(f"{where} must be a JSON array")
    result = tuple(value)
    if (
        not result
        or any(type(item) is not int or item <= 0 for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise GridbookExecutionContractError(
            f"{where} must be nonempty sorted unique positive integers"
        )
    return tuple(int(item) for item in result)


def _parse_predicate(payload: Any, *, where: str) -> GridbookPredicate:
    if not isinstance(payload, Mapping):
        raise GridbookExecutionContractError(f"{where} must be an object")
    _require_exact_keys(payload, _PREDICATE_KEYS, where=where)
    fact = payload["fact"]
    op = payload["op"]
    value = payload["value"]
    if fact not in _PREDICABLE_FACTS:
        raise GridbookExecutionContractError(
            f"{where}.fact must be one of {sorted(_PREDICABLE_FACTS)}"
        )
    if op not in _PREDICATE_OPS:
        raise GridbookExecutionContractError(
            f"{where}.op must be one of {sorted(_PREDICATE_OPS)}"
        )
    if op == "in":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or not value
        ):
            raise GridbookExecutionContractError(
                f"{where}.value must be a nonempty JSON array for op='in'"
            )
        value = tuple(value)
    elif op in {"multiple_of", "at_least", "at_most"}:
        if type(value) is not int or (op == "multiple_of" and value <= 0):
            raise GridbookExecutionContractError(
                f"{where}.value must be an integer"
                + (" > 0" if op == "multiple_of" else "")
            )
    return GridbookPredicate(fact=str(fact), op=str(op), value=value)


def _parse_cell(
    payload: Any,
    *,
    contract: Mapping[str, Any],
    where: str,
    platform_ids: set[str],
) -> GridbookExecutionCell:
    if not isinstance(payload, Mapping):
        raise GridbookExecutionContractError(f"{where} must be an object")
    _require_exact_keys(payload, _CELL_KEYS, where=where)
    cell_id = payload["id"]
    platform = payload["platform"]
    family = payload["family"]
    structure = payload["structure"]
    regime = payload["regime"]
    route_status = payload["route_status"]
    qualification = payload["qualification"]
    if not isinstance(cell_id, str) or not cell_id:
        raise GridbookExecutionContractError(
            f"{where}.id must be a nonempty string"
        )
    if platform not in platform_ids:
        raise GridbookExecutionContractError(
            f"{where}.platform must name a declared platform"
        )
    if not isinstance(family, str) or not family:
        raise GridbookExecutionContractError(
            f"{where}.family must be a nonempty string"
        )
    if structure not in GRIDBOOK_LANE_STRUCTURES:
        raise GridbookExecutionContractError(
            f"{where}.structure must be one of {list(GRIDBOOK_LANE_STRUCTURES)}"
        )
    if regime not in GRIDBOOK_LANE_REGIMES:
        raise GridbookExecutionContractError(
            f"{where}.regime must be one of {list(GRIDBOOK_LANE_REGIMES)}"
        )
    if route_status not in ROUTE_STATUSES:
        raise GridbookExecutionContractError(
            f"{where}.route_status must be one of {sorted(ROUTE_STATUSES)}"
        )
    if qualification not in QUALIFICATIONS:
        raise GridbookExecutionContractError(
            f"{where}.qualification must be one of {sorted(QUALIFICATIONS)}"
        )
    rungs = _strict_rungs(payload["rungs"], where=f"{where}.rungs")
    try:
        declared = gridbook_format_rungs(contract, family)
    except GridbookFormatContractError as exc:
        raise GridbookExecutionContractError(str(exc)) from exc
    if declared.producer_rungs is None:
        raise GridbookExecutionContractError(
            f"{where}: family {family!r} has no producer-rung attestation"
        )
    outside = sorted(set(rungs) - set(declared.producer_rungs))
    if outside:
        raise GridbookExecutionContractError(
            f"{where}.rungs are outside formats[{family!r}].producer_rungs: "
            f"{outside}"
        )
    flags = _string_array(
        payload["requires_serve_flags"],
        where=f"{where}.requires_serve_flags",
    )
    if flags and route_status != ROUTE_STATUS_BACKED_WITH_SERVE_FLAG:
        raise GridbookExecutionContractError(
            f"{where}: serve flags require route_status="
            f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r}"
        )
    if route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG and not flags:
        raise GridbookExecutionContractError(
            f"{where}: a flag-backed route must name its serve flags"
        )
    raw_predicates = payload["predicates"]
    if (
        not isinstance(raw_predicates, Sequence)
        or isinstance(raw_predicates, (str, bytes, bytearray))
    ):
        raise GridbookExecutionContractError(
            f"{where}.predicates must be a JSON array"
        )
    predicates = tuple(
        _parse_predicate(item, where=f"{where}.predicates[{index}]")
        for index, item in enumerate(raw_predicates)
    )
    return GridbookExecutionCell(
        id=cell_id,
        platform=str(platform),
        family=family,
        structure=str(structure),
        regime=str(regime),
        rungs=rungs,
        route_status=str(route_status),
        qualification=str(qualification),
        requires_serve_flags=flags,
        predicates=predicates,
    )


def parse_gridbook_execution_contract(
    contract: Mapping[str, Any],
    *,
    where: str = "Gridbook runtime contract",
) -> GridbookExecutionContract:
    """Parse the closed-world v2 table after validating v11 producer rungs."""

    try:
        validate_gridbook_cb_rung_contract(
            contract, require_producer_attestation=True, where=where
        )
    except GridbookFormatContractError as exc:
        raise GridbookExecutionContractError(str(exc)) from exc
    block = contract.get("lane_eligibility")
    if not isinstance(block, Mapping):
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility v2 is required"
        )
    _require_exact_keys(block, _LANE_KEYS, where=f"{where}.lane_eligibility")
    if block["schema"] != GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA:
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility.schema must be "
            f"{GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA!r}"
        )
    regimes = _string_array(
        block["regimes"],
        where=f"{where}.lane_eligibility.regimes",
        nonempty=True,
    )
    if regimes != GRIDBOOK_LANE_REGIMES:
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility.regimes must exactly equal "
            f"{list(GRIDBOOK_LANE_REGIMES)}"
        )
    structures = _string_array(
        block["structures"],
        where=f"{where}.lane_eligibility.structures",
        nonempty=True,
    )
    if structures != GRIDBOOK_LANE_STRUCTURES:
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility.structures must exactly equal "
            f"{list(GRIDBOOK_LANE_STRUCTURES)}"
        )
    raw_platforms = block["platforms"]
    if not isinstance(raw_platforms, Mapping) or not raw_platforms:
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility.platforms must be a nonempty object"
        )
    platforms: list[GridbookPlatform] = []
    for platform_id, raw in raw_platforms.items():
        spot = f"{where}.lane_eligibility.platforms[{platform_id!r}]"
        if not isinstance(platform_id, str) or not platform_id.startswith("sm_"):
            raise GridbookExecutionContractError(
                f"{spot}: platform ids must use the sm_<capability> spelling"
            )
        if not isinstance(raw, Mapping):
            raise GridbookExecutionContractError(f"{spot} must be an object")
        _require_exact_keys(raw, _PLATFORM_KEYS, where=spot)
        capability = raw["compute_capability"]
        if (
            not isinstance(capability, Sequence)
            or isinstance(capability, (str, bytes, bytearray))
            or len(capability) != 2
            or any(type(part) is not int or part < 0 for part in capability)
        ):
            raise GridbookExecutionContractError(
                f"{spot}.compute_capability must be exactly two nonnegative ints"
            )
        pair = (int(capability[0]), int(capability[1]))
        if platform_id != f"sm_{10 * pair[0] + pair[1]}":
            raise GridbookExecutionContractError(
                f"{spot}: id disagrees with compute_capability {list(pair)}"
            )
        platforms.append(GridbookPlatform(platform_id, pair))
    capabilities = [item.device_capability for item in platforms]
    if len(capabilities) != len(set(capabilities)):
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility platforms repeat a compute capability"
        )
    raw_cells = block["cells"]
    if (
        not isinstance(raw_cells, Sequence)
        or isinstance(raw_cells, (str, bytes, bytearray))
    ):
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility.cells must be a JSON array"
        )
    platform_ids = {item.id for item in platforms}
    cells = tuple(
        _parse_cell(
            item,
            contract=contract,
            where=f"{where}.lane_eligibility.cells[{index}]",
            platform_ids=platform_ids,
        )
        for index, item in enumerate(raw_cells)
    )
    ids = [item.id for item in cells]
    if len(ids) != len(set(ids)):
        raise GridbookExecutionContractError(
            f"{where}.lane_eligibility cell ids must be unique"
        )
    return GridbookExecutionContract(tuple(platforms), cells)


def _require_gridbook_routes(
    contract: Mapping[str, Any],
    *,
    family: str,
    device_capability: tuple[int, int],
    structure: str,
    rungs: Sequence[int],
    facts_by_rung: Mapping[int, Mapping[str, Any]] | None = None,
    required_qualification: str,
    where: str = "Gridbook runtime contract",
) -> GridbookDeviceRouteAttestation:
    """Require backed winners at one exact qualification in every cell.

    Capability matching is exact: an sm90 cell cannot qualify sm89 and a
    numerically newer capability never inherits an older cell. Missing cells
    are the closed-world spelling of an unbacked route.
    """

    parsed = parse_gridbook_execution_contract(contract, where=where)
    requested = _strict_rungs(rungs, where=f"{where} requested rungs")
    try:
        declared = gridbook_format_rungs(contract, family, where=where)
    except GridbookFormatContractError as exc:
        raise GridbookExecutionContractError(str(exc)) from exc
    assert declared.producer_rungs is not None
    outside = sorted(set(requested) - set(declared.producer_rungs))
    if outside:
        raise GridbookExecutionContractError(
            f"{where}: requested rungs are outside formats[].producer_rungs: "
            f"{outside}"
        )
    matching_platforms = [
        item for item in parsed.platforms
        if item.device_capability == tuple(device_capability)
    ]
    if len(matching_platforms) != 1:
        raise GridbookExecutionContractError(
            f"{where}: exact capability {tuple(device_capability)} maps to "
            f"{[item.id for item in matching_platforms]} platforms"
        )
    platform = matching_platforms[0]
    if structure not in GRIDBOOK_LANE_STRUCTURES:
        raise GridbookExecutionContractError(
            f"{where}: structure must be one of {list(GRIDBOOK_LANE_STRUCTURES)}"
        )
    overrides = facts_by_rung or {}
    unknown_rungs = sorted(set(overrides) - set(requested))
    if unknown_rungs:
        raise GridbookExecutionContractError(
            f"{where}: facts_by_rung contains unrequested rungs {unknown_rungs}"
        )
    resolutions: list[GridbookRegimeResolution] = []
    all_flags: set[str] = set()
    for rung in requested:
        facts: dict[str, Any] = {
            "role_split": False if structure == "dense" else None,
            "in_features": None,
            "out_features": None,
        }
        extra = overrides.get(rung, {})
        if not isinstance(extra, Mapping):
            raise GridbookExecutionContractError(
                f"{where}: facts_by_rung[{rung}] must be an object"
            )
        unknown_facts = sorted(set(extra) - _PREDICABLE_FACTS)
        if unknown_facts:
            raise GridbookExecutionContractError(
                f"{where}: facts_by_rung[{rung}] has unknown facts "
                f"{unknown_facts}"
            )
        facts.update(extra)
        for regime in GRIDBOOK_LANE_REGIMES:
            matches = [
                cell for cell in parsed.cells
                if cell.platform == platform.id
                and cell.family == family
                and cell.structure == structure
                and cell.regime == regime
                and rung in cell.rungs
                and cell.matches(facts)
            ]
            if not matches:
                raise GridbookExecutionContractError(
                    f"{where}: closed-world table has no {platform.id}/"
                    f"{structure}/{regime} route for {family} K{rung}"
                )
            winning_rank = max(_ROUTE_RANK[cell.route_status] for cell in matches)
            strongest = [
                cell for cell in matches
                if _ROUTE_RANK[cell.route_status] == winning_rank
            ]
            if len(strongest) != 1:
                raise GridbookExecutionContractError(
                    f"{where}: ambiguous strongest route cells "
                    f"{[cell.id for cell in strongest]} for {platform.id}/"
                    f"{structure}/{regime}/{family} K{rung}; equal-ranked "
                    "overlaps are forbidden rather than resolved by JSON order"
                )
            winner = strongest[0]
            if (
                winner.route_status not in {
                    ROUTE_STATUS_BACKED,
                    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
                }
                or winner.qualification != required_qualification
            ):
                raise GridbookExecutionContractError(
                    f"{where}: route winner {winner.id!r} for {platform.id}/"
                    f"{structure}/{regime}/{family} K{rung} is "
                    f"route_status={winner.route_status!r}, "
                    f"qualification={winner.qualification!r}; this consumer "
                    f"requires backed* plus {required_qualification}"
                )
            all_flags.update(winner.requires_serve_flags)
            resolutions.append(GridbookRegimeResolution(
                rung=rung,
                regime=regime,
                cell_id=winner.id,
                route_status=winner.route_status,
                qualification=winner.qualification,
                requires_serve_flags=winner.requires_serve_flags,
            ))
    return GridbookDeviceRouteAttestation(
        platform=platform,
        family=str(family),
        structure=str(structure),
        rungs=requested,
        resolutions=tuple(resolutions),
        requires_serve_flags=tuple(sorted(all_flags)),
    )


def require_device_qualified_gridbook_routes(
    contract: Mapping[str, Any],
    *,
    family: str,
    device_capability: tuple[int, int],
    structure: str,
    rungs: Sequence[int],
    facts_by_rung: Mapping[int, Mapping[str, Any]] | None = None,
    where: str = "Gridbook runtime contract",
) -> GridbookDeviceRouteAttestation:
    """Require backed, device-qualified winners in every regime and rung.

    This remains the only production-qualification entry point.  Structural
    cross-compile evidence must use the separately named compile-only helper
    and therefore cannot accidentally weaken a production caller.
    """

    return _require_gridbook_routes(
        contract,
        family=family,
        device_capability=device_capability,
        structure=structure,
        rungs=rungs,
        facts_by_rung=facts_by_rung,
        required_qualification=QUALIFICATION_DEVICE_QUALIFIED,
        where=where,
    )


def require_compile_only_gridbook_routes(
    contract: Mapping[str, Any],
    *,
    family: str,
    device_capability: tuple[int, int],
    structure: str,
    rungs: Sequence[int],
    facts_by_rung: Mapping[int, Mapping[str, Any]] | None = None,
    where: str = "Gridbook runtime contract",
) -> GridbookDeviceRouteAttestation:
    """Require exact backed ``compile_only`` structural cells.

    The returned attestation is deliberately not producer-legal.  This helper
    exists solely for artifacts whose immutable policy stamp says they are
    validation-only and unreleasable.
    """

    return _require_gridbook_routes(
        contract,
        family=family,
        device_capability=device_capability,
        structure=structure,
        rungs=rungs,
        facts_by_rung=facts_by_rung,
        required_qualification=QUALIFICATION_COMPILE_ONLY,
        where=where,
    )


def attested_min_capability_sm(
    contract: Mapping[str, Any],
    *,
    family: str,
    structure: str,
    rungs: Sequence[int],
) -> int:
    """Summarize exact qualified cells; never interpolate capability support."""

    parsed = parse_gridbook_execution_contract(contract)
    qualified: list[int] = []
    for platform in parsed.platforms:
        try:
            require_device_qualified_gridbook_routes(
                contract,
                family=family,
                device_capability=platform.device_capability,
                structure=structure,
                rungs=rungs,
            )
        except GridbookExecutionContractError:
            continue
        qualified.append(platform.capability_sm)
    if not qualified:
        raise GridbookExecutionContractError(
            f"no exact device-qualified {family}/{structure} execution cell "
            f"covers rungs {list(rungs)}"
        )
    return min(qualified)


__all__ = [
    "GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA",
    "GRIDBOOK_LANE_REGIMES",
    "GRIDBOOK_LANE_STRUCTURES",
    "GridbookDeviceRouteAttestation",
    "GridbookExecutionCell",
    "GridbookExecutionContract",
    "GridbookExecutionContractError",
    "GridbookPlatform",
    "GridbookPredicate",
    "GridbookRegimeResolution",
    "QUALIFICATION_COMPILE_ONLY",
    "QUALIFICATION_DEVICE_QUALIFIED",
    "ROUTE_STATUS_BACKED",
    "ROUTE_STATUS_BACKED_WITH_SERVE_FLAG",
    "ROUTE_STATUS_FALLBACK",
    "attested_min_capability_sm",
    "parse_gridbook_execution_contract",
    "require_compile_only_gridbook_routes",
    "require_device_qualified_gridbook_routes",
]
