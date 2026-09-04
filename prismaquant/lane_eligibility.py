"""Serving-lane eligibility, ATTESTED from the pinned runtime's own contract.

Principle 14: a claim about another runtime is *derived from a machine-readable
table the pinned runtime publishes, or refused*. This module is the consumption
half of that rule. It never encodes what a serving runtime does; it reads what
that runtime *says* it does, from the packaged ``runtime_contract.json`` its
installed distribution carries, and reports ``UNATTESTED`` when the pinned
release publishes no claim covering a unit.

The live publisher is Tessera's own vLLM plugin (``tessera.serving``). Until
2026-09-02 it was the Gridbook codebook plugin, and this module was named
``gridbook_lane_eligibility``; the Gridbook lane was retired that day (Rob:
"put Tessera in PrismaQuant and remove Gridbook") and the module was renamed
to the neutral name it should always have had. See
``archive/gridbook_lane_2026-09-02/README.md``.

Why this exists (the measured defect, on the retired lane). The shipped DSv4
87 GB codebook artifact carried 11 routed FP8-CB layers whose
``gate_proj``/``up_proj`` bound distinct learned codebooks. That runtime's
persistent-B prefill lane refused per-role split books, so those layers took
the announced expand+grouped-bridge route above the token threshold. Nothing in
the producer knew: no serving-profile lane declared a structured
``route_status``, so eligibility was not a gate input and a user discovered it
at serve time. Its twin on the vanilla-vLLM lane is
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

Schemas v3/v4/v5, and why absence carries the whole weight
---------------------------------------------------
``tessera.lane-eligibility.v3`` is a **closed-world cell table**. It declares
``platforms``, ``regimes``, ``structures`` and a list of ``cells``, each cell
naming exactly one ``(platform, family, structure, regime)`` and the rung set it
covers -- ``rungs`` (codebook K) for a ``cb_product`` family, ``rungs_q256``
(body bits per 256 weights) for a RATE-addressed family. Which discriminator a
cell uses is NOT a key on the cell: it is decided by whether the cell's family
appears in ``formats[]`` with a rate-addressed ``kind``
(:data:`RATE_ADDRESSED_FORMAT_KINDS`), exactly as the publisher's own
validator decides it.

``tessera.lane-eligibility.v4`` adds a required non-empty ``executes`` set of
``{symbol, decoder}`` launches and makes residency an explicit resolution
axis. A caller must name a residency; two cells in the same scope may never
claim the same residency. The published serve flag selects that axis, and the
family's ``residency_modes`` bounds it. Legacy v3 tables retain their original
resolution semantics and never acquire fabricated launch claims.

``tessera.lane-eligibility.v5`` additionally requires every cell to name its
exact image digest and execution-mode scope. Missing target context is
unattested, never a request to use the global dense image or another cell.

One parser, and why the vocabulary is wider than one publisher
---------------------------------------------------------------
``gridbook.lane-eligibility.v3`` was the same wire format from the retired
lane, and this parser served both. What remains of that is vocabulary, not a
second authority: the ``cb_product`` kind, its ``rungs`` discriminator and the
``tcq_trellis`` rate-addressed kind are still parsed, because they are the
closed-world grammar a v3 table is written in, and a parser that silently
dropped a kind would mis-read a table rather than refuse it. Only
:data:`LANE_ELIGIBILITY_SCHEMAS` decides whose tables are accepted, and since
2026-09-02 that set names Tessera alone.

The publisher's cell status vocabulary is ``backed | backed_with_serve_flag |
fallback``. There is deliberately **no ``unbacked`` cell**: a runtime does not
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
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


#: Schema of the eligibility table PrismaQuant consumes, published by Tessera's
#: own vLLM plugin
#: (``tessera.serving``, entry point ``tessera``, ``quant_method: "tessera"``).
#: v4 adds launches/residency; v5 adds exact runtime image/execution scope.
#: The parser owns these grammars; plugin requirements remain optional only
#: for explicitly identified legacy v3 tables.
#:
#: Until 2026-09-02 this set also carried ``gridbook.lane-eligibility.v3``, the
#: same wire format published by the retired Gridbook codebook lane. That lane
#: was removed with Rob's decision to put Tessera in PrismaQuant and remove
#: Gridbook; see ``archive/gridbook_lane_2026-09-02/README.md``.
LANE_ELIGIBILITY_SCHEMA_TESSERA = "tessera.lane-eligibility.v5"
LANE_ELIGIBILITY_SCHEMA_TESSERA_V4 = "tessera.lane-eligibility.v4"
LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3 = "tessera.lane-eligibility.v3"

#: Every eligibility-table schema this parser accepts. The check is a set
#: membership, never a prefix match: an unrecognised vendor is a table this
#: repository was not handed, and an unlisted version is not treated as a
#: subset of either supported grammar (see ``_parse_table``).
LANE_ELIGIBILITY_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3,
})

#: The residency vocabulary in Tessera's v4 flag grammar. Each format row
#: publishes the subset its route supports; no cell can widen that subset.
TESSERA_RESIDENCY_MODES = frozenset({"resident", "streamed"})
TESSERA_EXECUTION_MODES = frozenset({"eager", "compiled"})
_LAUNCH_SCHEMAS = frozenset({
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4, LANE_ELIGIBILITY_SCHEMA_TESSERA,
})
_DIGEST_IMAGE = re.compile(
    r"[a-z0-9][a-z0-9._/-]*[a-z0-9]@sha256:[0-9a-f]{64}")

#: Schema of the provenance payload this module produces. It was
#: ``prismaquant.cb_route_attestation.v2`` until 2026-09-02, when the Gridbook
#: codebook lane was retired: the name and its ``gridbook_serving_*`` fields
#: both named a runtime that no longer has a lane here, and the only reader of
#: those fields (``cb_route_status_gate``) went into the archive with it, so
#: the rename costs no shipped artifact a reader.
ROUTE_ATTESTATION_SCHEMA = "prismaquant.lane_route_attestation.v3"

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

#: The CLOSED set a packaged cell may declare, mirroring the publisher's
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

#: Structural classes a unit can belong to. The two take different runtime
#: dispatch paths and therefore different eligibility cells.
STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = frozenset({STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE})

#: The ``formats[].kind`` discriminator. It lives on the FORMAT row, never on a
#: lane cell -- a cell's rung vocabulary follows from its family's kind.
FORMAT_KIND_CB_PRODUCT = "cb_product"
FORMAT_KIND_TCQ_TRELLIS = "tcq_trellis"
#: Tessera's discriminator for the same idea: a family addressed by a RATE
#: (body bits per 256 weights), not by a codebook size. ``tcq_trellis`` is
#: the retired lane's spelling of it and ``tessera_wire`` is Tessera's; both resolve to
#: the ``rungs_q256`` rung vocabulary and to ``EligibilityCell.is_trellis``.
FORMAT_KIND_TESSERA_WIRE = "tessera_wire"
FORMAT_KINDS = frozenset({
    FORMAT_KIND_CB_PRODUCT,
    FORMAT_KIND_TCQ_TRELLIS,
    FORMAT_KIND_TESSERA_WIRE,
})

#: The kinds whose rung axis is a RATE. ``EligibilityCell.is_trellis`` means
#: exactly "rate-addressed" -- the name is historical, from the era when
#: ``tcq_trellis`` was the only such kind -- and every dispatch on the rung
#: vocabulary tests membership here, never one kind constant. There are two
#: such dispatch sites (``_published_families`` and ``resolve_payload_rung``)
#: and they must agree, or a name resolves to a family with no rate and every
#: downstream cell match fails closed for the wrong reason.
RATE_ADDRESSED_FORMAT_KINDS = frozenset({
    FORMAT_KIND_TCQ_TRELLIS,
    FORMAT_KIND_TESSERA_WIRE,
})

class LaneEligibilityError(ValueError):
    """The materialized contract or its eligibility table is malformed."""


@dataclass(frozen=True)
class ServingContext:
    """The explicit target of a cell lookup; no field has a runtime default."""

    platform: str
    structure: str
    residency: str
    runtime_image: str
    execution_mode: str

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, str) or not value.strip():
                raise LaneEligibilityError(f"serving_context.{name} must be a non-empty string")
        for name, allowed in (("structure", STRUCTURES),
                              ("residency", TESSERA_RESIDENCY_MODES),
                              ("execution_mode", TESSERA_EXECUTION_MODES)):
            if getattr(self, name) not in allowed:
                raise LaneEligibilityError(
                    f"serving_context.{name} must be one of {sorted(allowed)}")
        if not _DIGEST_IMAGE.fullmatch(self.runtime_image):
            raise LaneEligibilityError(
                "serving_context.runtime_image must be an exact repository@sha256:<64 lowercase hex> reference")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def key(self) -> tuple[str, ...]:
        return tuple(self.as_dict().values())


def cell_matches_serving_context(cell: Any, context: ServingContext) -> bool:
    """Match a parsed v5 cell's whole scope, shared by every admission path."""
    return (
        cell.platform == context.platform
        and cell.structure == context.structure
        and context.residency in cell.residency_modes
        and cell.runtime_image == context.runtime_image
        and context.execution_mode in cell.execution_modes
    )


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
    a RATE-addressed family (``tcq_trellis`` or ``tessera_wire``) carries a
    rate. Neither is ever a rounded bpw.
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
            raise LaneEligibilityError(
                f"{self.qname}: structure must be one of "
                f"{sorted(STRUCTURES)}, got {self.structure!r}")
        if self.k is not None and self.rate_q256 is not None:
            raise LaneEligibilityError(
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
            raise LaneEligibilityError(
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
    """One packaged cell: bytes, platform, regime, residency and launch.

    A cell is scoped to exactly one ``(platform, family, structure, regime)``
    and covers an explicit, non-empty rung list. It carries no prose: a
    validator refuses ``detail``/``rationale`` keys on a cell, because a gate
    cannot read prose (principle 14). Legacy v3 alone permits an absent plugin
    key. v4 requires Tessera, launch declarations and residency flags.
    """

    id: str
    platform: str
    family: str
    structure: str
    regime: str
    route_status: str
    qualification: str
    #: CB codebook rungs. Empty for a rate-addressed cell.
    rungs: tuple[int, ...] = ()
    #: Body bits per 256 weights. Empty for a CB cell.
    rungs_q256: tuple[int, ...] = ()
    #: The activation contract this route executes. Rate-addressed cells only;
    #: a CB cell publishes none and this stays "".
    activation_contract: str = ""
    requires_serve_flags: tuple[str, ...] = ()
    #: The vLLM plugin whose installation this route requires, or "" when the
    #: route is reachable in the pinned runtime as shipped. It is a
    #: machine-readable CELL field rather than prose because an export gate
    #: has to be able to refuse an artifact whose serve command would not
    #: install the plugin -- stock vLLM has no reader for Tessera bytes, so
    #: those routes are plugin-gated, not merely flag-gated. Retired-lane cells
    #: publish none and this stays "".
    requires_plugin: str = ""
    predicates: tuple[tuple[str, str, Any], ...] = ()
    #: "This cell's family is addressed by a RATE, not by a codebook size."
    #: The name is historical -- ``tcq_trellis`` was the only such kind when it
    #: was chosen -- and ``tessera_wire`` families set it too.
    is_trellis: bool = False
    #: v4's published launches, retained as pairs rather than inferred from IDs.
    executes: tuple[tuple[str, str], ...] = ()
    #: Parsed from the v4 cell's explicit TESSERA_SERVE_MODE flag.
    residency_modes: tuple[str, ...] = ()
    runtime_image: str = ""
    execution_modes: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        where: str,
        *,
        trellis_families: frozenset[str],
        schema: str = LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3,
        residency_modes: Sequence[str] = (),
    ) -> "EligibilityCell":
        if not isinstance(payload, Mapping):
            raise LaneEligibilityError(f"{where} must be a JSON object")
        if schema not in LANE_ELIGIBILITY_SCHEMAS:
            raise LaneEligibilityError(f"{where}: unsupported lane schema {schema!r}")
        family = str(payload.get("family", ""))
        if not family:
            raise LaneEligibilityError(
                f"{where}: cell must name a payload family")
        # The rung vocabulary follows the FAMILY's kind, exactly as the publisher's
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
        is_v4 = schema in _LAUNCH_SCHEMAS
        is_v5 = schema == LANE_ELIGIBILITY_SCHEMA_TESSERA
        if is_v4:
            required.update({"requires_plugin", "executes"})
        if is_v5:
            required.add("runtime")
        _require_keys(payload, where, required=required,
                      optional=set() if is_v4 else {"requires_plugin"})

        status = str(payload["route_status"])
        if status not in CELL_ROUTE_STATUSES:
            raise LaneEligibilityError(
                f"{where}.route_status must be one of "
                f"{sorted(CELL_ROUTE_STATUSES)}, got {status!r}. The runtime "
                "does not enumerate what it refuses; absence, not an "
                f"{ROUTE_STATUS_UNBACKED!r} cell, is how a lane table says no.")
        qualification = str(payload["qualification"])
        if qualification not in CELL_QUALIFICATIONS:
            raise LaneEligibilityError(
                f"{where}.qualification must be one of "
                f"{sorted(CELL_QUALIFICATIONS)}, got {qualification!r}")
        structure = str(payload["structure"])
        if structure not in STRUCTURES:
            raise LaneEligibilityError(
                f"{where}.structure must be one of {sorted(STRUCTURES)}, "
                f"got {structure!r}")

        rungs = _parse_rungs(payload[rung_key], f"{where}.{rung_key}")

        activation_contract = ""
        if is_trellis:
            activation_contract = str(payload["activation_contract"])
            if not activation_contract:
                raise LaneEligibilityError(
                    f"{where}.activation_contract must name the contract this "
                    "route executes; an empty one attests nothing")

        requires_plugin = str(payload.get("requires_plugin", ""))
        if is_v4 and payload["requires_plugin"] != "tessera":
            raise LaneEligibilityError(
                f"{where}.requires_plugin must be 'tessera'; stock vLLM "
                "has no reader for these bytes")
        if requires_plugin and status not in LANE_ROUTE_STATUSES:
            # Mirrors the ``requires_serve_flags`` rule below. A plugin
            # requirement is an instruction for reaching a route that EXISTS;
            # naming one on a cell whose route is an announced fallback says
            # nothing an operator can act on, and would let a reader believe a
            # plugin install turns a fallback into a native route.
            raise LaneEligibilityError(
                f"{where}: requires_plugin is {requires_plugin!r} but "
                f"route_status is {status!r}; a plugin requirement is only "
                f"meaningful on a cell whose route is one of "
                f"{sorted(LANE_ROUTE_STATUSES)}")
        executes: tuple[tuple[str, str], ...] = ()
        cell_modes: tuple[str, ...] = ()
        if is_v4:
            executes, cell_modes = parse_v4_cell_contract(
                payload, where, residency_modes=residency_modes)
        runtime_image = ""
        execution_modes: tuple[str, ...] = ()
        if is_v5:
            runtime_image, execution_modes = parse_v5_runtime(payload["runtime"], where + ".runtime")
        flags = tuple(str(v) for v in payload["requires_serve_flags"])
        if flags and status != ROUTE_STATUS_BACKED_WITH_SERVE_FLAG:
            raise LaneEligibilityError(
                f"{where}: requires_serve_flags is non-empty but route_status "
                f"is {status!r}; a flag-gated route is "
                f"{ROUTE_STATUS_BACKED_WITH_SERVE_FLAG!r} by definition")
        if status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG and not flags:
            raise LaneEligibilityError(
                f"{where}.requires_serve_flags: route_status is "
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
            requires_plugin=requires_plugin,
            predicates=_parse_predicates(payload["predicates"], where),
            is_trellis=is_trellis,
            executes=executes,
            residency_modes=cell_modes,
            runtime_image=runtime_image,
            execution_modes=execution_modes,
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
        if self.requires_plugin:
            # Emitted only when non-empty, so a keyless cell's serialization
            # is byte-identical to what it was before this key existed.
            payload["requires_plugin"] = self.requires_plugin
        if self.executes:
            payload["executes"] = [
                {"symbol": symbol, "decoder": decoder}
                for symbol, decoder in self.executes
            ]
        if self.runtime_image:
            payload["runtime"] = {
                "image": self.runtime_image, "execution_modes": list(self.execution_modes),
            }
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
            "serving_runtime_version": self.runtime_version,
            "serving_runtime_commit": self.runtime_commit,
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
            required_plugins = sorted({
                cell.requires_plugin for cell in self.cells
                if cell.requires_plugin
            })
            if required_plugins:
                # Only when non-empty: a table without the key keeps a payload
                # is unchanged by this widening.
                payload["required_plugins"] = required_plugins
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
    #: The vLLM plugins the matched cell requires, aggregated exactly as
    #: ``requires_serve_flags`` is. A tuple rather than a scalar because it
    #: rolls up the same way at unit granularity, and one shape at both levels
    #: is what stops a consumer having to special-case the regime view.
    requires_plugins: tuple[str, ...] = ()
    qualification: str = ""
    activation_contract: str = ""
    detail: str = ""
    executes: tuple[tuple[str, str], ...] = ()
    residency: str = ""
    runtime_image: str = ""
    execution_mode: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "regime": self.regime,
            "route_status": self.route_status,
            "cell_id": self.cell_id,
            "requires_serve_flags": list(self.requires_serve_flags),
            "qualification": self.qualification or None,
            "activation_contract": self.activation_contract or None,
            "detail": self.detail,
        }
        if self.requires_plugins:
            payload["requires_plugins"] = list(self.requires_plugins)
        if self.executes:
            payload["executes"] = [
                {"symbol": symbol, "decoder": decoder}
                for symbol, decoder in self.executes
            ]
        if self.residency:
            payload["residency"] = self.residency
        if self.runtime_image:
            payload["runtime_image"] = self.runtime_image
            payload["execution_mode"] = self.execution_mode
        return payload


@dataclass(frozen=True)
class UnitRoute:
    """One unit's resolved route status across every declared regime."""

    facts: UnitStructuralFacts
    route_status: str
    regimes: tuple[RegimeRoute, ...] = ()
    requires_serve_flags: tuple[str, ...] = ()
    #: The vLLM plugins every backed regime of this unit requires, aggregated
    #: over the same regimes ``requires_serve_flags`` is aggregated over. An
    #: artifact whose units carry a non-empty set is servable ONLY where those
    #: plugins are installed, which is a fact its serve command and its
    #: shipcard have to carry.
    requires_plugins: tuple[str, ...] = ()
    #: True when the pinned contract publishes this unit's payload family, i.e.
    #: when the eligibility table is the authority for these bytes; False when
    #: it does not. ``None`` means the question was never asked, which is the
    #: only honest value when no table was consulted at all (an absent index,
    #: an unreadable contract). It is NOT a synonym for True: a default that
    #: rides into provenance unevaluated is the same defect class as a zero
    #: that reads as a verdict, and ``as_dict`` omits the key rather than
    #: publish one.
    in_scope: bool | None = None
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
            "requires_serve_flags": list(self.requires_serve_flags),
            "regime_routes": [r.as_dict() for r in self.regimes],
        }
        if self.requires_plugins:
            payload["requires_plugins"] = list(self.requires_plugins)
        if self.in_scope is not None:
            payload["in_scope"] = self.in_scope
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
    residency: str | None = None,
    runtime_image: str | None = None,
    execution_mode: str | None = None,
) -> UnitRoute:
    """Resolve one unit's route status against the pinned eligibility table.

    Absent table -> ``unattested`` with no regime detail. Never a zero, never a
    guess, and never principle 9's ``backed`` by default.

    ``platform`` is the exact runtime platform id the artifact targets (the
    serving profile's ``target_platform``, e.g. ``sm_121``). Lane cells are
    platform-scoped, so resolving without one cannot name a route: a missing or
    unpublished platform yields ``unattested``, never a match-any.

    v4 and v5 additionally require ``residency``, the explicit serve mode for the
    artifact. It filters cells before route selection; omitting it cannot
    choose whichever same-scope cell happened to be listed first. v3 keeps
    its original behavior and makes no claim about executed launches.
    V5 also requires ``runtime_image`` and ``execution_mode``. Every regime
    must resolve on that same complete target; cells from different runtime
    scopes cannot jointly attest one artifact.
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
            in_scope=True,
            unattested_reason=(
                "no declared target platform; lane cells are platform-scoped, so "
                "no route can be named without one. Declare "
                "'target_platform' on the serving profile this artifact "
                f"targets; the pinned release publishes {list(table.platforms)}"
            ),
        )
    if platform not in table.platforms:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=True,
            unattested_reason=(
                f"the pinned release publishes no lane cells for platform "
                f"{platform!r} (it publishes {list(table.platforms)}); an "
                "unpublished platform attests nothing"
            ),
        )

    is_v4 = table.schema in _LAUNCH_SCHEMAS
    is_v5 = table.schema == LANE_ELIGIBILITY_SCHEMA_TESSERA
    if is_v4 and residency not in TESSERA_RESIDENCY_MODES:
        return UnitRoute(
            facts=facts,
            route_status=ROUTE_STATUS_UNATTESTED,
            in_scope=True,
            unattested_reason=(
                f"no declared supported residency (got {residency!r}); v4 "
                "cells require an explicit residency to identify their "
                f"launches: {sorted(TESSERA_RESIDENCY_MODES)}"
            ),
        )

    serving_context = None
    if is_v5:
        try:
            serving_context = ServingContext(
                platform=platform, structure=facts.structure, residency=residency,
                runtime_image=runtime_image, execution_mode=execution_mode)
        except LaneEligibilityError as exc:
            return UnitRoute(facts=facts, route_status=ROUTE_STATUS_UNATTESTED,
                             in_scope=True, unattested_reason=str(exc))

    candidates = [
        cell for cell in table.cells
        if cell.platform == platform
        and cell.family == facts.payload_family
        and cell.structure == facts.structure
        and (not is_v4 or residency in cell.residency_modes)
        and (not is_v5 or cell_matches_serving_context(cell, serving_context))
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
            # closed-world table that is the ONLY negative signal there is,
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
            requires_plugins=(
                (best.requires_plugin,) if best.requires_plugin else ()),
            qualification=best.qualification,
            activation_contract=best.activation_contract,
            executes=best.executes,
            residency=str(residency) if is_v4 else "",
            runtime_image=str(runtime_image) if is_v5 else "",
            execution_mode=str(execution_mode) if is_v5 else "",
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
    plugins = tuple(sorted({
        plugin for r in backed for plugin in r.requires_plugins
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
            + (f" at residency {residency!r}" if is_v4 else "")
            + (f", runtime_image {runtime_image!r}, execution_mode {execution_mode!r}"
               if is_v5 else "")
        )
        return UnitRoute(
            facts=facts,
            route_status=status,
            regimes=tuple(regimes),
            in_scope=True,
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
        in_scope=True,
        requires_serve_flags=flags,
        requires_plugins=plugins,
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
def load_eligibility_table(
    version: str | None = None,
    *,
    contract_path: Path | None = None,
) -> EligibilityTable:
    """Load the eligibility table the pinned SERVING runtime packages.

    ``contract_path`` names the packaged ``runtime_contract.json`` of the
    runtime whose routes are being attested; the caller resolves it from that
    runtime's own installed package, never from a copy in this repository
    (``tessera_render.tessera_serving_contract_path`` is the one live caller).

    Until 2026-09-02 this function had a second mode: with no
    ``contract_path`` it read Gridbook's SERVING pin and the byte-verbatim
    contract copy indexed under ``prismaquant/gridbook_runtime/``. The
    Gridbook lane is retired (``archive/gridbook_lane_2026-09-02/``), the
    materialized copies went with it, and there is no default table any more.
    Calling with neither argument is therefore not an error but an honest
    ABSENCE: it returns a table with ``present=False``, so every unit resolves
    to ``UNATTESTED`` and the export gate fails closed, which is exactly what
    "no pinned runtime claims this route" should mean.
    """
    version = str(version or "")
    commit = ""
    path: Path | None = Path(contract_path) if contract_path is not None else None

    if path is None or not path.exists():
        return EligibilityTable(
            present=False,
            runtime_version=version,
            runtime_commit=commit,
            contract_sha256="",
            absent_reason=(
                "no packaged runtime contract was supplied, so no serving "
                "lane can be attested. Pass the pinned runtime's own "
                "runtime_contract.json (contract_path=). Route status stays "
                "UNATTESTED until then."
            ),
        )

    sha = _sha256(path)

    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaneEligibilityError(
            f"cannot read {path}: {exc}") from exc

    block = contract.get("lane_eligibility")
    if block is None:
        return EligibilityTable(
            present=False,
            runtime_version=version or "",
            runtime_commit=commit,
            contract_sha256=sha,
            absent_reason=(
                f"{path} packages no 'lane_eligibility' table, so no "
                "serving-lane route can be attested for this pin. This is a "
                "REFUSAL TO CLAIM, not a clean bill: the runtime's lane "
                "predicates exist but are not published."
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
    ``n_sub``/``rungs``, while a RATE-addressed row (``tcq_trellis`` in a
    retired Gridbook contract, ``tessera_wire`` in Tessera's) carries
    ``attested_rungs_q256`` (named ``candidate_rungs_q256`` before contract v2,
    which Tessera keeps as a deprecated alias and says it drops at schema v2)
    /``reader_rate_range_q256``/``native_terminal_q256``. A unit's payload family, sub-table split, rung
    legality and body rate are therefore genuinely DERIVED here rather than
    read out of a local table -- which is the point of principle 14.
    """
    path = Path(contract_path) if contract_path is not None else None
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
    serve every kind: a CB family IS its name prefix (``FP8_CB_K``), a
    rate-addressed family is not (``TCQ_E2M1_R256`` and ``TESSERA_E2M1_K2_R896``
    name rates around a 256-weight block, and are never a rung of themselves).
    """
    pattern = str(entry.get("name_pattern", ""))
    head, sep, _ = pattern.partition("{k}")
    if not sep:
        return ""
    return head.upper()


def resolve_payload_rung(
    format_name: str,
    published_formats: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, int | None, int | None]:
    """``(payload_family, k, rate_q256)`` for a format name, DERIVED.

    The one place a format name is turned into the runtime's own vocabulary,
    so the export gate and the serving-lane resolver cannot disagree about
    what ``FP8_CB_K44`` or ``TCQ_E2M1_R512`` is. Both rung fields are ``None``
    when the pinned release publishes no such rung, which is what makes every
    downstream match fail closed instead of admitting an unlisted rate.

    Returns the raw upper-cased name as the family when the contract publishes
    no codec for it (BF16, a SOURCE passthrough, a stock CT rung), which is
    also the signal that the lane table is not the authority for those bytes.
    """
    if published_formats is None:
        published_formats = load_published_formats()

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
        if str(entry.get("kind", FORMAT_KIND_CB_PRODUCT)) in (
                RATE_ADDRESSED_FORMAT_KINDS):
            lo, hi = (int(v) for v in entry["reader_rate_range_q256"])
            # Outside the published reader range the rate stays None, so every
            # cell's rung list fails to cover it and the unit is unattested.
            return fam, None, (value if lo <= value <= hi else None)
        if value in {int(r) for r in entry.get("rungs", ())}:
            return fam, value, None
        # The pinned release does not instantiate this rung: leaving k None
        # makes every k-predicate and every cell match fail closed rather than
        # pass silently.
        return fam, None, None
    return upper, None, None


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

    family, k, rate_q256 = resolve_payload_rung(format_name, published_formats)
    n_sub: int | None = None
    if k is not None:
        entry = published_formats.get(family, {})
        n_sub = int(entry["n_sub"])

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
    """``(all families, rate-addressed families)`` from the formats table.

    The second set is what ``EligibilityCell.is_trellis`` is built from, and
    it holds every family whose ``kind`` is in
    :data:`RATE_ADDRESSED_FORMAT_KINDS` -- the retired lane's ``tcq_trellis`` and
    Tessera's ``tessera_wire`` alike. Such a family's cells carry
    ``rungs_q256``; a ``cb_product`` family's carry ``rungs``.
    """
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise LaneEligibilityError(
            "runtime_contract.formats must be a JSON array; the lane table's "
            "rung vocabulary is decided by each family's published kind")
    families: set[str] = set()
    trellis: set[str] = set()
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}] must be a JSON object")
        family = str(entry.get("family", ""))
        if not family:
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}] publishes no family")
        kind = str(entry.get("kind", FORMAT_KIND_CB_PRODUCT))
        if kind not in FORMAT_KINDS:
            raise LaneEligibilityError(
                f"runtime_contract.formats[{i}].kind {kind!r} is not one of "
                f"{sorted(FORMAT_KINDS)}")
        families.add(family)
        if kind in RATE_ADDRESSED_FORMAT_KINDS:
            trellis.add(family)
    return frozenset(families), frozenset(trellis)


def _parse_table(block: Any, formats: Any, version: str, commit: str, sha: str
                 ) -> EligibilityTable:
    where = "runtime_contract.lane_eligibility"
    if not isinstance(block, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    # The schema string is checked BEFORE the key set, deliberately. An older
    # table fails both, and "missing field(s) ['cells', 'platforms']" would
    # send its reader off to add keys to a v2 block rather than to
    # re-materialize the contract from a release with a supported schema.
    if block.get("schema") not in LANE_ELIGIBILITY_SCHEMAS:
        raise LaneEligibilityError(
            f"{where}.schema must be one of "
            f"{sorted(LANE_ELIGIBILITY_SCHEMAS)}, got {block.get('schema')!r}. "
            "An older lane table is not a subset of these -- v1/v2 cells are "
            "not platform-scoped and carry no rung list, so reading one here "
            "would admit every rung on every platform. An unrecognised VENDOR "
            "prefix is a table this repository was not handed at all. Either "
            "way, re-materialize the contract from a release that publishes a "
            "schema named above rather than editing the table.")
    _require_keys(
        block, where,
        required={"schema", "platforms", "regimes", "structures", "cells"},
        optional=set(),
    )

    platforms_block = block["platforms"]
    if not isinstance(platforms_block, Mapping) or not platforms_block:
        raise LaneEligibilityError(
            f"{where}.platforms must be a non-empty JSON object keyed by "
            "platform id")
    platforms = tuple(str(p) for p in platforms_block)

    regimes = tuple(str(r) for r in block["regimes"])
    if not regimes or len(set(regimes)) != len(regimes):
        raise LaneEligibilityError(
            f"{where}.regimes must be a non-empty list of unique ids")

    structures = tuple(str(s) for s in block["structures"])
    if not structures or len(set(structures)) != len(structures):
        raise LaneEligibilityError(
            f"{where}.structures must be a non-empty list of unique ids")
    unknown = sorted(set(structures) - STRUCTURES)
    if unknown:
        raise LaneEligibilityError(
            f"{where}.structures names {unknown}, which this repository has no "
            f"dispatch path for; the known set is {sorted(STRUCTURES)}")

    families, trellis_families = _published_families(formats)
    schema = str(block["schema"])
    is_v4 = schema in _LAUNCH_SCHEMAS
    is_v5 = schema == LANE_ELIGIBILITY_SCHEMA_TESSERA
    family_modes: dict[str, tuple[str, ...]] = {}
    if is_v4:
        for i, entry in enumerate(formats):
            family = str(entry["family"])
            modes = entry.get("residency_modes")
            if (not isinstance(modes, list) or not modes
                    or any(not isinstance(mode, str)
                           or mode not in TESSERA_RESIDENCY_MODES for mode in modes)
                    or len(set(modes)) != len(modes)):
                raise LaneEligibilityError(
                    f"runtime_contract.formats[{i}].residency_modes must "
                    "publish a non-empty list of distinct supported "
                    f"residencies {sorted(TESSERA_RESIDENCY_MODES)}")
            family_modes[family] = tuple(modes)

    cells_block = block["cells"]
    if not isinstance(cells_block, Sequence) or isinstance(
            cells_block, (str, bytes)):
        raise LaneEligibilityError(f"{where}.cells must be a JSON array")
    cells = tuple(
        EligibilityCell.from_dict(
            cell, f"{where}.cells[{i}]", trellis_families=trellis_families,
            schema=schema,
            residency_modes=(family_modes.get(str(cell.get("family", "")), ())
                             if isinstance(cell, Mapping) else ()))
        for i, cell in enumerate(cells_block)
    )

    for cell in cells:
        if cell.regime not in regimes:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].regime {cell.regime!r} is not a "
                f"declared regime {list(regimes)}")
        if cell.platform not in platforms:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].platform {cell.platform!r} is not "
                f"a declared platform {list(platforms)}")
        if cell.structure not in structures:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].structure {cell.structure!r} is "
                f"not a declared structure {list(structures)}")
        if cell.family not in families:
            raise LaneEligibilityError(
                f"{where}.cells[{cell.id!r}].family {cell.family!r} is not "
                f"published in runtime_contract.formats "
                f"({sorted(families)}); a lane cell for a codec the runtime "
                "does not publish attests a route to nothing")
    ids = [cell.id for cell in cells]
    if len(set(ids)) != len(ids):
        raise LaneEligibilityError(f"{where}.cells ids must be unique")
    if is_v4:
        scopes: dict[tuple[str, ...], str] = {}
        for cell in cells:
            for mode in cell.residency_modes:
                for execution in cell.execution_modes if is_v5 else ("",):
                    scope = (cell.platform, cell.family, cell.structure, cell.regime, mode)
                    if is_v5:
                        scope += (cell.runtime_image, execution)
                    previous = scopes.get(scope)
                    if previous is not None:
                        raise LaneEligibilityError(
                            f"{where}.cells {previous!r} and {cell.id!r} both cover "
                            f"{scope}; overlapping serving scopes make route "
                            "resolution depend on cell order")
                    scopes[scope] = cell.id

    return EligibilityTable(
        present=True,
        runtime_version=version,
        runtime_commit=commit,
        contract_sha256=sha,
        schema=schema,
        platforms=platforms,
        regimes=regimes,
        structures=structures,
        cells=cells,
        families=families,
        trellis_families=trellis_families,
    )


def parse_v5_runtime(payload: Any, where: str) -> tuple[str, tuple[str, ...]]:
    """The v5 runtime grammar, shared by both contract readers."""
    if not isinstance(payload, Mapping):
        raise LaneEligibilityError(f"{where} must be a JSON object")
    _require_keys(payload, where, required={"image", "execution_modes"}, optional=set())
    image = payload["image"]
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise LaneEligibilityError(
            f"{where}.image must be an exact repository@sha256:<64 lowercase hex> reference")
    modes = payload["execution_modes"]
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) or mode not in TESSERA_EXECUTION_MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise LaneEligibilityError(
            f"{where}.execution_modes must be a non-empty list of distinct values from "
            f"{sorted(TESSERA_EXECUTION_MODES)}")
    return image, tuple(modes)


def parse_v4_cell_contract(
    payload: Mapping[str, Any],
    where: str,
    *,
    residency_modes: Sequence[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Parse the v4 launch set and residency selector, without runtime imports.

    The runtime owns whether a launch is correct for its route. This consumer
    verifies the published grammar and preserves that claim; it never derives
    a launch from a family name or a cell ID.
    """
    raw_executes = payload.get("executes")
    if not isinstance(raw_executes, list) or not raw_executes:
        raise LaneEligibilityError(
            f"{where}.executes must be a non-empty JSON array of "
            "{symbol, decoder} objects")
    launches: list[tuple[str, str]] = []
    for i, launch in enumerate(raw_executes):
        spot = f"{where}.executes[{i}]"
        if not isinstance(launch, Mapping):
            raise LaneEligibilityError(f"{spot} must be a JSON object")
        _require_keys(launch, spot, required={"symbol", "decoder"}, optional=set())
        if any(not isinstance(launch[key], str) or not launch[key].strip()
               for key in ("symbol", "decoder")):
            raise LaneEligibilityError(
                f"{spot}.symbol and decoder must be non-empty strings")
        launches.append((launch["symbol"], launch["decoder"]))
    if len(set(launches)) != len(launches):
        raise LaneEligibilityError(
            f"{where}.executes must not repeat a (symbol, decoder) pair")

    head = "TESSERA_SERVE_MODE="
    flags = payload.get("requires_serve_flags")
    if (not isinstance(flags, list)
            or any(not isinstance(flag, str) or not flag for flag in flags)):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags must be a JSON array of non-empty strings")
    named = [flag for flag in flags if flag.startswith(head)]
    if len(named) != 1:
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags must name exactly one "
            "TESSERA_SERVE_MODE residency flag")
    modes = tuple(named[0][len(head):].split("|"))
    if (len(set(modes)) != len(modes)
            or any(mode not in TESSERA_RESIDENCY_MODES for mode in modes)):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags names invalid or repeated "
            f"residency values {list(modes)}")
    if not set(modes).issubset(residency_modes):
        raise LaneEligibilityError(
            f"{where}.requires_serve_flags residency {list(modes)} exceeds "
            f"the family's published residency_modes {list(residency_modes)}")
    return tuple(launches), modes


def _parse_rungs(payload: Any, where: str) -> tuple[int, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise LaneEligibilityError(f"{where} must be a JSON array")
    if not payload:
        raise LaneEligibilityError(
            f"{where} must name at least one rung; an empty rung list covers "
            "nothing and would silently make its cell unreachable")
    out: list[int] = []
    for i, item in enumerate(payload):
        if isinstance(item, bool) or not isinstance(item, int):
            raise LaneEligibilityError(
                f"{where}[{i}] must be an integer rung, got {item!r}")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise LaneEligibilityError(f"{where} must not repeat a rung")
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
        raise LaneEligibilityError(
            f"{where}.predicates must be a JSON array")
    out: list[tuple[str, str, Any]] = []
    for i, item in enumerate(payload):
        spot = f"{where}.predicates[{i}]"
        if not isinstance(item, Mapping):
            raise LaneEligibilityError(f"{spot} must be a JSON object")
        _require_keys(item, spot, required={"fact", "op", "value"}, optional=set())
        fact = str(item["fact"])
        if fact not in _PREDICABLE_FACTS:
            raise LaneEligibilityError(
                f"{spot}.fact {fact!r} is not an attestable structural fact; "
                f"the closed set is {sorted(_PREDICABLE_FACTS)}. An unknown "
                "predicate is a malformed contract, never a no-op rule.")
        op = str(item["op"])
        if op not in _PREDICATE_OPS:
            raise LaneEligibilityError(
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
    raise LaneEligibilityError(f"unknown predicate op {op!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_keys(payload: Mapping[str, Any], where: str, *,
                  required: set[str], optional: set[str]) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing:
        raise LaneEligibilityError(f"{where}: missing field(s) {missing}")
    if extra:
        raise LaneEligibilityError(f"{where}: unknown field(s) {extra}")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "LANE_ELIGIBILITY_SCHEMA_TESSERA",
    "LANE_ELIGIBILITY_SCHEMA_TESSERA_LEGACY_V3",
    "LANE_ELIGIBILITY_SCHEMAS",
    "parse_v4_cell_contract",
    "ROUTE_ATTESTATION_SCHEMA",
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
    "FORMAT_KIND_TESSERA_WIRE",
    "FORMAT_KINDS",
    "RATE_ADDRESSED_FORMAT_KINDS",
    "STRUCTURE_DENSE",
    "STRUCTURE_ROUTED_MOE",
    "LaneEligibilityError",
    "UnitStructuralFacts",
    "EligibilityCell",
    "EligibilityTable",
    "RegimeRoute",
    "UnitRoute",
    "resolve_unit_route",
    "load_eligibility_table",
    "load_published_formats",
    "resolve_payload_rung",
    "unit_structural_facts",
]
