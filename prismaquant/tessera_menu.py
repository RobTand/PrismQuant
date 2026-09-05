"""The production menu over Tessera's continuous rate axis.

``tessera_formats`` says which rungs *exist* (thousands of them across four
families, at a 1/256-bpp quantum) and ``tessera_render`` says how to price one.
This module is the third thing the production allocator needs: **which of them
may enter a menu for THIS unit, on THIS target**, and it is the only place that
answers it.

Three gates, and they are deliberately separate
-----------------------------------------------
1. **The wire can carry it.**  ``tessera_rung_is_serialisable`` -- the grid's
   digest is a permanent commitment in Tessera's ``SERIALISABLE_GRIDS``.  A rung
   that fails this dies in ``alphabet_plane()`` at export, after the allocation
   and the whole production cache have been built, so the menu asks up front.
2. **The shape can carry it.**  Tessera's own raise sites, asked by
   construction rather than by catching: arity divides the rows, the
   super-symbol span divides the trellis positions, the rung's Bresenham root
   is realisable over the column count, and the block scale plane's period
   divides the tensor.  Plus the **serving TP degree**, which shards the shape
   before the runtime ever sees it (:func:`tessera_shard_granularity`).
3. **A pinned runtime executes it.**  :func:`route_admission` -- ONE lookup,
   against the pinned serving release's own published contract (principle 14).

Gate 3 reads the current pinned runtime and serving scope
--------------------------------------------------------
The table that answers is **Tessera's own** packaged
``tessera/serving/runtime_contract.json``. Its native cells declare the
families, rates and serving contexts that are qualified. Since 2026-09-04,
``tessera_serving_runtime_pin`` names an exact Tessera commit and the SHA-256
of the contract it packages. Matching cells can be admitted when the installed
contract matches that digest and the requested scope; an absent cell or a
different contract remains unattested. Reader support for a wider rate range
does not qualify every rate for serving. Moving the pin edits the JSON and
module constants together, including when a new producer API requires a newer
commit with unchanged contract bytes.

Which is why the ``detail`` on an unattested rung names the conjunct that
actually refused (``tessera_render.tessera_lane_admission`` returns it beside
the verdict).  "Unattested" alone is four different facts, and reporting the
wrong one sends a reader to re-pin a contract that already has the row.

So the menu carries a **mode**, and the default is the attested one::

    MENU_ATTESTED  (default)   only rungs a pinned runtime attests
    MENU_RESEARCH  (opt-in)    every writable, shape-legal rung, each stamped
                               ROUTE_STATUS_UNATTESTED

``MENU_RESEARCH`` is not an override of the gate; it is the gate reporting
itself.  Principle 1 is explicit that route status "never removes an honestly
priced rung from the menu -- an allocator that wants an unservable rung is
reporting a serving gap"; principle 9 puts the refusal at **export**, per
artifact, on ``route_status``.  So a research-mode allocation is a measurement
that names its own unservability on every unit and in its provenance, and the
export gate refuses it for the same reason it would refuse any unattested unit.
Nothing here can make an unattested rung exportable.

The rate axis is priced by campaign, not by enumeration
-------------------------------------------------------
Four families address thousands of rungs at every shape -- 6764 on a
2048x1024 unit, 3709 of them the 16-bit family's.  Rendering all of them is
not a cost model, it is a week.  :mod:`prismaquant.tessera_campaign` renders a small
measured **anchor** set per (unit, family) and interpolates the rest through
``tessera_rate_surface``, with leave-one-anchor-out and allocation-regret gates
deciding how many anchors that is.  This module supplies the *legal* set the
campaign samples from and the dense menu it fills in.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .lane_eligibility import ServingContext

from .lane_eligibility import (
    SCOPED_LANE_SCHEMAS,
    legacy_runtime_scope_refusal,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_UNATTESTED,
)
from .tessera_formats import (
    SUPERBLOCK_WEIGHTS,
    TesseraFamily,
    TesseraFormatError,
    family_q256_bounds,
    get_tessera_family,
    lazily_sized_cache,
    parse_tessera_format_name,
    scale_plane_name,
    tessera_family,
    tessera_serving_route,
    tessera_wire_recipe,
    Q256_UNIT,
)

__all__ = [
    "MENU_ATTESTED",
    "MENU_MODES",
    "MENU_MODE_ENV",
    "MENU_RESEARCH",
    "MENU_TOKEN",
    "PARALLEL_COLUMN",
    "PARALLEL_NONE",
    "PARALLEL_ROW",
    "MenuRung",
    "RouteAdmission",
    "TesseraMenuError",
    "check_tessera_activation_agreement",
    "collapse_to_dp_bins",
    "expand_tessera_menu",
    "MENU_CACHE_SHAPES_ENV",
    "DEFAULT_MENU_CACHE_SHAPES",
    "menu_rungs_per_shape",
    "menu_cache_shapes",
    "menu_cache_bound",
    "menu_scaled_cache",
    "GEOMETRY_CACHE_SHAPES",
    "menu_families",
    "menu_mode",
    "prune_dominated",
    "fused_module_licence",
    "route_admission",
    "shard_shape",
    "tessera_shape_legal",
    "tessera_shard_granularity",
    "tessera_tp_legal",
]


class TesseraMenuError(ValueError):
    """A Tessera menu request is malformed."""


#: The token a FORMATS menu carries.  It is NOT a format name: a format name
#: addresses one rung, and this addresses a continuous family set whose members
#: differ per unit (E4M3's per-unit planes make its rate a function of the
#: shape, so even the byte cost is not knowable until a Linear is named).
MENU_TOKEN = "TESSERA"

#: Only rungs the pinned serving release attests.  The production default.
MENU_ATTESTED = "attested"
#: Every writable, shape-legal rung, each stamped ``unattested``.  Opt-in, and
#: refused by the export gate exactly as any other unattested unit is.
MENU_RESEARCH = "research"
MENU_MODES = frozenset({MENU_ATTESTED, MENU_RESEARCH})

#: The lever that selects the mode.  ONE name, read in ONE place
#: (:func:`menu_mode`), stamped into every allocation's provenance.  Unset is
#: the production default, so a run that never mentions it cannot silently
#: allocate onto rungs no runtime attests.
MENU_MODE_ENV = "PRISMAQUANT_TESSERA_MENU"


def menu_mode(value: "str | None" = None) -> str:
    """The menu mode in force.  ``value`` overrides the environment.

    Explicit and refusing: an unrecognised spelling raises rather than falling
    back to either mode, because both failure directions are bad -- silently
    attested drops the whole menu, silently research ships an unattested
    allocation.
    """
    import os

    raw = value if value is not None else os.environ.get(MENU_MODE_ENV, "")
    text = str(raw).strip().lower() or MENU_ATTESTED
    if text not in MENU_MODES:
        raise TesseraMenuError(
            f"{MENU_MODE_ENV}={raw!r} is not a Tessera menu mode; expected one "
            f"of {sorted(MENU_MODES)}"
        )
    return text

#: Route statuses under which a pinned runtime executes a rung natively.
_NATIVE_ROUTE_STATUSES = frozenset(
    {ROUTE_STATUS_BACKED, ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}
)

#: How tensor parallelism cuts a Linear.  Column-parallel shards the OUTPUT
#: features (q/k/v/gate/up); row-parallel shards the INPUT features
#: (o_proj/down_proj); ``none`` is a unit no TP rank splits (a packed expert
#: under expert parallelism, or a replicated head).
PARALLEL_COLUMN = "column"
PARALLEL_ROW = "row"
PARALLEL_NONE = "none"
_PARALLEL_KINDS = frozenset({PARALLEL_COLUMN, PARALLEL_ROW, PARALLEL_NONE})

#: Nothing here enumerates grids or arities. The base set is
#: ``tessera_formats._HARDWARE_BASES`` -- the grids that materialise into a
#: stock format at load, which is the same table
#: ``TesseraFamily.terminal_format`` and ``tessera_serving_route`` read for the
#: activation contract, so a family cannot appear on the menu without a route
#: or carry a route the menu does not know about. A new hardware grid (the
#: BF16/W16A16 family under construction on the Tessera side) joins the menu by
#: landing in that table and nowhere else.
#:
#: Arity is bounded by the encoder's own anchor budget rather than by a tuple:
#: a family costs ``2**(arity*log2(base_size))`` anchors scored per trellis
#: step, ``tessera_family`` refuses at or above ``ANCHOR_BUDGET_BITS``, and
#: that refusal is the filter (see :func:`menu_families`). The ceiling below is
#: only a loop bound; the refusal is what decides.
_MAX_ARITY_SEARCH = 8


# ---------------------------------------------------------------------------
# Gate 3: the one attestation read
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RouteAdmission:
    """What a pinned runtime says about one Tessera rung.  DERIVED.

    Every field is read from a table the runtime published, or refused.  The
    ``detail`` string explains; nothing reads it as a value (principle 14).
    """

    format_name: str
    #: ``TESSERA_E2M1_K2`` -- the payload family the contract would publish.
    payload_family: str
    #: The activation contract the decoded tile executes under, from
    #: ``tessera_serving_route``.  A *layout* fact, which is all a producer may
    #: assert on its own; it is not the attestation.  What makes it safe to
    #: price is :func:`check_tessera_activation_agreement`, which
    #: :func:`route_admission` runs whenever cells attest the rung: the priced
    #: triple must project to the cells' executed ``activation_contract``.
    activation_contract: str
    terminal_format: "str | None"
    act_bits: "int | None"
    act_group_size: "int | None"
    min_capability_sm: int
    #: ``backed`` / ``backed_with_serve_flag`` / ``unattested``.  Read off the
    #: attesting cell, never typed here: a cell that says
    #: ``backed_with_serve_flag`` is not a ``backed`` one, and collapsing the
    #: two loses the flags the serve needs.
    route_status: str
    #: Can Tessera's wire carry these bytes at all?  Independent of the runtime.
    serialisable: bool
    #: Which table answered.  ``tessera_packaged_contract:...`` in production
    #: -- the ``runtime_contract.json`` Tessera's own vLLM plugin packages --
    #: and ``tessera_dev_pin:...`` under the development override.  Both carry
    #: the release they were read from, so two runs against different Tessera
    #: builds cannot both claim "the Tessera contract".
    source: str
    detail: str = ""
    #: The serve flags the attesting cell requires, verbatim.  Empty when
    #: nothing attests -- absence, not "no flags needed".
    requires_serve_flags: tuple[str, ...] = ()
    #: The largest tensor-parallel world size the contract attests for this
    #: family, or ``None`` when no contract governs it.  ``closed_world``
    #: semantics: absence is "not attested at any degree", not "any degree".
    max_world_size: "int | None" = None
    #: The caller's explicit target, never inferred from a matching cell.
    serving_context: "ServingContext | None" = None
    #: Whether this contract requires the complete serving scope for admission.
    requires_serving_context: bool = False

    @property
    def attested(self) -> bool:
        """Does a pinned runtime execute this rung natively?"""
        return self.route_status in _NATIVE_ROUTE_STATUSES

    def admits(self, mode: str = MENU_ATTESTED) -> bool:
        """Does this rung enter a menu built in ``mode``?"""
        if not self.serialisable:
            return False
        if mode == MENU_RESEARCH:
            return True
        if mode == MENU_ATTESTED:
            return self.attested
        raise TesseraMenuError(
            f"unknown menu mode {mode!r}; expected one of {sorted(MENU_MODES)}"
        )


#: Where the attestation is read from when no development contract is pinned:
#: the ``runtime_contract.json`` packaged inside Tessera's own vLLM plugin,
#: parsed by the shared ``lane_eligibility`` reader.  It travels into every
#: unit's provenance so an artifact records which table admitted it -- and,
#: when nothing does, which table declined to.
#:
#: It named ``gridbook_serving_runtime_pin:lane_eligibility`` until 2026-09-02.
#: That pin is archived (``archive/gridbook_lane_2026-09-02/``) and stopped
#: governing Tessera admission when Gridbook withdrew its Tessera lane, so the
#: string had become a claim about a runtime that no longer answers -- exactly
#: the unattested assertion principle 14 refuses, in the field a gate reads.
_PACKAGED_ATTESTATION_SOURCE = "tessera_packaged_contract:lane_eligibility"
#: Where it is read from under the development override.  The commit is
#: appended by :func:`_tessera_attestation_source`, so two runs against
#: different Tessera builds cannot both claim "the Tessera contract".
_TESSERA_ATTESTATION_SOURCE = "tessera_dev_pin:runtime_contract"


def _tessera_attestation_source(contract) -> str:
    return f"{_TESSERA_ATTESTATION_SOURCE}:{contract.commit[:12]}"


def _packaged_attestation_source(table) -> str:
    """The packaged contract's provenance string, stamped with its release."""
    version = getattr(table, "runtime_version", "") or "unknown"
    return f"{_PACKAGED_ATTESTATION_SOURCE}:{version}"


def tessera_runtime_contract():
    """The pinned Tessera contract, or ``None``.  **The one read.**

    Every Tessera attestation in this module goes through here, so "which
    table answered" is a single fact per run rather than a per-call race.
    ``None`` is production (no RELEASE tag exists, so nothing is attested); a
    mismatched or malformed pin raises rather than degrading to ``None``,
    because a stale pin that silently empties the menu is the exact failure
    the pin exists to prevent.
    """
    from .tessera_runtime_contract import load_tessera_contract

    return load_tessera_contract()


def fused_module_licence():
    """What the pinned runtime says a fused module's roles may disagree about.

    The :class:`~prismaquant.tessera_runtime_contract.FusedModuleLicence` from
    the contract's ``fused_module`` block, or ``None`` when no contract is
    pinned.  Routed through :func:`tessera_runtime_contract` -- this module's
    declared one read -- so the licence and the route admission cannot come
    from two different Tessera builds inside one run, and so substituting that
    one function substitutes both.

    ``None`` is the **absence** of a licence, not a permissive default.  The
    group knapsack's per-member rungs are a claim about what the runtime's
    loader accepts (RobTand/tessera#37), so with no table to derive it from
    the fold declines and the group keeps the per-NAME options, which assert
    nothing.
    """
    contract = tessera_runtime_contract()
    return None if contract is None else contract.fused_module


def tessera_tp_world_attested(
    family: "str | TesseraFamily", tp_degree: int,
) -> tuple[bool, str]:
    """Does the pinned contract attest this family at this world size?

    The contract's ``tensor_parallel`` block is ``closed_world``: a family it
    does not list is attested at no degree at all, and a listed one is
    attested up to its ``max_world_size``. This is the *attestation* leg of TP
    legality and it is independent of the *geometry* leg
    (:func:`tessera_shard_granularity`) -- a shape that shards cleanly at TP=8
    is still not servable at TP=8 if the runtime says it serves one rank. Both
    must pass, and the failing one is named in the reason so a receipt can say
    which.
    """
    spec = get_tessera_family(family)
    tp = int(tp_degree)
    if tp <= 1:
        # There is no tensor-parallel claim to attest on a whole unit. This
        # leg answers about the DEGREE only; whether the rung is served at all
        # is ``route_admission``'s question, and answering it twice here would
        # make the TP gate a second route gate that refuses Tessera outright
        # whenever no contract is pinned -- including on the research menu,
        # whose entire purpose is to price rungs nothing attests.
        return True, ""
    contract = tessera_runtime_contract()
    if contract is None:
        return False, (
            "tp_unattested: no Tessera runtime contract is pinned, so no world "
            "size is attested for any family"
        )
    declared = contract.max_world_size.get(spec.name)
    if declared is None:
        return False, (
            f"tp_unattested: the pinned contract's closed-world tensor_parallel "
            f"block does not list {spec.name}"
        )
    if tp > int(declared):
        return False, (
            f"tp{tp}_unattested: the pinned contract attests {spec.name} up to "
            f"world size {int(declared)}"
        )
    return True, ""


def check_tessera_activation_agreement(name, route, cell_contracts) -> None:
    """Refuse a priced A side the attesting cells do not execute.

    The allocator prices ``route`` -- ``tessera_serving_route``'s layout fact,
    carried on every :class:`RouteAdmission`, byte breakdown and campaign row
    -- while the serve executes the attesting cells' ``activation_contract``,
    and the two vocabularies do not match character for character
    (``w4a4-nvfp4-e2m1-group16-ue4m3`` vs ``e2m1_group16_ue4m3_static``).  So
    the comparison is on the projection both sides price in --
    ``(act_bits, act_group_size)`` -- via
    :func:`prismaquant.tessera_runtime_contract.cell_activation_projection`.
    Cells that disagree with each other, a cell vocabulary the projection
    does not transcribe, or a priced triple the cells do not execute all
    raise: pricing one A side while the serve executes another is the
    currency error that misallocated 87 GB on NVFP4_CB (2026-08-17), and an
    admission that papered over it would be the unread *value* on the active
    pricing path (#165).  No attesting cells is agreement by absence -- the
    rung is unattested for its own reason, and there is nothing to disagree
    with.
    """
    from .tessera_runtime_contract import (
        TesseraContractError, cell_activation_projection,
    )

    distinct = tuple(dict.fromkeys(str(value) for value in cell_contracts))
    if len(distinct) > 1:
        raise TesseraMenuError(
            f"{name}: the attesting cells disagree about the executed A side "
            f"({sorted(distinct)}); one rung cannot execute two activation "
            "contracts and this reader will not pick one"
        )
    if not distinct:
        return
    try:
        bits, group = cell_activation_projection(distinct[0])
    except TesseraContractError as exc:
        raise TesseraMenuError(f"{name}: {exc}") from exc
    if (route.act_bits, route.act_group_size) != (bits, group):
        raise TesseraMenuError(
            f"{name}: priced activation {route.contract} "
            f"(act_bits={route.act_bits}, "
            f"act_group_size={route.act_group_size}) is not what the "
            f"attesting cells execute ({distinct[0]}: act_bits={bits}, "
            f"act_group_size={group}); pricing one A side while the serve "
            "executes another is a currency error"
        )


def _attested_cell_conditions(name: str, cells) -> tuple[str, tuple[str, ...]]:
    """Keep the same cell-derived serving conditions on both pin paths."""
    statuses = {cell.route_status for cell in cells}
    if len(statuses) != 1:
        named = ", ".join(
            f"{getattr(cell, 'cell_id', getattr(cell, 'id', 'unknown'))}={cell.route_status}"
            for cell in cells)
        raise TesseraMenuError(
            f"{name}: the pinned contract's native cells disagree about "
            f"route status ({sorted(statuses)}): {named}; "
            "one rung cannot be two routes and this reader will not pick one")
    return next(iter(statuses)), tuple(dict.fromkeys(
        flag for cell in cells for flag in cell.requires_serve_flags))


def route_admission(
    name: str, *, serving_context: "ServingContext | None" = None,
) -> RouteAdmission:
    """The pinned runtime's verdict on one Tessera rung.  **The one seam.**

    This is the single function in the production path that reads a serving
    contract on Tessera's behalf.  Everything else -- the menu, the campaign,
    the candidate gate, the provenance -- consumes :class:`RouteAdmission`, so
    when the Tessera serving lane publishes its own ``runtime_contract.json``
    the swap is this function's body and nothing else.

    **Deliberately not memoised**, and the reason is worth keeping. A cache
    over a runtime-contract read outlives the contract, and here that is not
    hypothetical: an ``lru_cache`` keyed on the rung name silently inverted
    ``test_no_tessera_rung_is_producer_eligible_on_the_pinned_release``, which
    runs after a test that patches the attestation to True and duly read back
    "eligible" for a rung nothing serves. Putting the resolved callables in the
    key fixes that case and not the next one -- a test that patches
    ``_pinned_serving_table`` instead changes the contract without changing any
    function object, and the stale verdict comes straight back (it did). The
    key this cache would need is the identity of the contract itself, which it
    cannot state, so it does not get a cache. It does not need one: the whole
    lookup is ~0.1 ms (parse 0.002, attest 0.016, serialisable 0.072, recipe +
    route 0.006 ms) *because the serialisable leg asks the family's grid rather
    than the rung* -- ask it per rung and one 16-bit menu spends 34 ms a rung
    re-hashing 65536 values -- the expensive part of a menu is the per-rung
    Bresenham
    realisability check.  That check *is* memoised per ``(family, rung, shape)``
    -- ``_shard_geometry`` and ``tessera_footprint._exact_bits_for_shape``,
    both sized by :func:`menu_cache_bound` -- but ``expand_tessera_menu``
    itself is not, and never was: it returns a fresh list its callers are free
    to keep.

    Today it delegates to ``tessera_render.tessera_lane_admission``, which
    resolves the rung against **Tessera's own** packaged
    ``runtime_contract.json`` through the shared ``lane_eligibility`` parser.
    That table publishes the attested Tessera families and their receipted
    rungs, so what refuses is the third conjunct whenever the installed
    Tessera is not the pinned commit/digest, and the fourth whenever the
    cell's own packaged evidence records a degenerate smoke.  The answer is
    then :data:`ROUTE_STATUS_UNATTESTED`.  The ``detail`` says which conjunct, verbatim from the
    admission, because a rung refused by the pin and a rung refused by an
    absent cell need different fixes and must not read the same.

    Before any of that, it runs :func:`check_tessera_activation_agreement`
    whenever cells attest the rung: the priced ``route`` and the executed
    cell ``activation_contract`` must project to the same ``(act_bits,
    act_group_size)``, on both the dev-pin and the packaged-contract paths.

    A v5 development contract requires the caller's complete serving context.
    Its own scoped lookup checks every required regime under that one target;
    an absent context cannot borrow the runtime or structure of another cell.
    A legacy contract retains its context-free admission but cannot attest a
    newly supplied runtime context it has no per-cell evidence to answer.
    """
    from .tessera_render import (
        _pinned_serving_table, tessera_attesting_cells, tessera_lane_admission,
        tessera_lane_attested, tessera_rung_is_serialisable,
    )

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        raise TesseraMenuError(f"{name!r} is not a Tessera format name")
    family, rung = parsed
    recipe = tessera_wire_recipe(family, rung)
    route = tessera_serving_route(family, recipe, rung)
    serialisable = tessera_rung_is_serialisable(name)

    contract = tessera_runtime_contract()
    flags: tuple[str, ...] = ()
    world: "int | None" = None
    requires_context = False
    if contract is not None:
        source = _tessera_attestation_source(contract)
        world = contract.max_world_size.get(family.name)
        requires_context = bool(getattr(contract, "requires_serving_context", False))
        legacy_scope_reason = (
            legacy_runtime_scope_refusal(getattr(contract, "lane_schema", "legacy"))
            if serving_context is not None and not requires_context else ""
        )
        cells = (
            (contract.native_cells(family.name, rung, serving_context=serving_context)
             if serving_context is not None else contract.native_cells(family.name, rung))
            if not legacy_scope_reason and contract.governs(family.name) else ()
        )
        if cells:
            status, flags = _attested_cell_conditions(name, cells)
            detail = (
                f"the pinned Tessera contract attests {len(cells)} native "
                f"cell(s) naming this rate: "
                f"{', '.join(cell.cell_id for cell in cells)}"
            )
            check_tessera_activation_agreement(
                name, route,
                [cell.activation_contract for cell in cells],
            )
        else:
            status = ROUTE_STATUS_UNATTESTED
            published = sorted(contract.attested_rungs.get(family.name, ()))
            detail = legacy_scope_reason or (
                f"the pinned Tessera contract publishes {family.name} at "
                f"rungs {published} and no native cell covers R{rung}"
                if contract.governs(family.name) else
                f"the pinned Tessera contract does not publish {family.name}"
            )
            if requires_context:
                detail += (
                    "; no explicit serving context was supplied"
                    if serving_context is None else
                    f" under serving context {serving_context.as_dict()}"
                )
    else:
        table, formats = _pinned_serving_table()
        source = _packaged_attestation_source(table)
        requires_context = table.schema in SCOPED_LANE_SCHEMAS
        scope = ({"serving_context": serving_context}
                 if serving_context is not None else {})
        # The VERDICT stays ``tessera_lane_attested``: it is the seam every
        # gate consults and the one the tests substitute.  The REASON is read
        # separately and only when the verdict is False, so patching the
        # verdict cannot invent a rationale the contract never gave.
        attesting = tessera_attesting_cells(name, table=table, formats=formats, **scope)
        if bool(tessera_lane_attested(name, table=table, formats=formats, **scope)):
            if not attesting:
                raise TesseraMenuError(f"{name}: admission has no attesting native cells")
            status, flags = _attested_cell_conditions(name, attesting)
            from .tessera_runtime_contract import published_tensor_parallel_limits
            from .tessera_render import tessera_serving_contract_path
            from importlib.resources import as_file
            with as_file(tessera_serving_contract_path()) as path:
                world = published_tensor_parallel_limits(
                    str(path), table.contract_sha256).get(family.name)
            detail = (
                "the packaged Tessera contract attests a device-qualified "
                "native cell naming this rate, under a reviewed release pin"
            )
        else:
            status = ROUTE_STATUS_UNATTESTED
            detail = tessera_lane_admission(name, table=table, formats=formats, **scope)[1]
        if attesting:
            check_tessera_activation_agreement(
                name, route,
                [cell.activation_contract for cell in attesting],
            )
    return RouteAdmission(
        format_name=name,
        payload_family=family.name,
        activation_contract=route.contract,
        terminal_format=route.terminal_format,
        act_bits=route.act_bits,
        act_group_size=route.act_group_size,
        min_capability_sm=route.min_capability_sm,
        route_status=status,
        serialisable=serialisable,
        source=source,
        detail=detail,
        requires_serve_flags=flags,
        max_world_size=world,
        serving_context=serving_context,
        requires_serving_context=requires_context,
    )


#: Lane id stamped on a Tessera candidate's resolved route. Deliberately not
#: one of the profile-declared lane ids: no serving-profile spec declares a
#: Tessera lane, and inventing a spec entry that says one exists would be the
#: hand-typed verdict principle 14 refuses. This id says only "this route was
#: resolved by the Tessera admission seam", and ``route_status`` carries the
#: verdict.
TESSERA_LANE_ID = "tessera_admission"


def tessera_resolved_serving_lane(
    name: str, *, runtime_version: str = "",
    serving_context: "ServingContext | None" = None,
):
    """The resolved serving route of one Tessera rung, or None.

    ``serving_profiles.serving_lane_route`` covers formats by NAME out of a
    profile's declared lanes, and no profile declares ~3000 Tessera rungs, so
    it returns ``None`` for every one of them. ``None`` is not a neutral
    answer downstream: ``selection_serving_lane_provenance`` reads it as
    ``no_declared_lane``, which reports a MISSING declaration when what is
    actually true is that a declaration exists and says "not attested". Those
    are different facts and principle 9 wants the second one on the card.

    So a Tessera candidate's route is resolved here instead, from
    :func:`route_admission` -- the same one seam, so the lane a unit is
    stamped with and the gate that admitted it cannot disagree. Every field is
    derived: ``route_status`` and its source come from the admission,
    ``activation_contract`` from ``tessera_serving_route``. Nothing is
    asserted -- ``route_status`` and ``requires_serve_flags`` are the attesting
    cell's own words -- and ``fused_mid_m_backed`` is False because no pinned
    release publishes a fused mid-M table for these bytes: absence, priced as
    absence.
    """
    from .serving_profiles import ResolvedServingLane, serving_runtime_version

    if parse_tessera_format_name(name) is None:
        return None
    admission = (route_admission(name, serving_context=serving_context)
                 if serving_context is not None else route_admission(name))
    return ResolvedServingLane(
        lane_id=TESSERA_LANE_ID,
        format=name,
        activation_contract=admission.activation_contract,
        fallback_route="expand_and_gemm",
        fused_mid_m_backed=False,
        fused_mid_m_rungs=(),
        fused_mid_m_range=None,
        runtime_version=(runtime_version or serving_runtime_version()),
        rungs_source=admission.source,
        rung=None,
        detail=admission.detail,
        route_status=admission.route_status,
        requires_serve_flags=admission.requires_serve_flags,
        route_status_source=admission.source,
        serving_context=getattr(admission, "serving_context", serving_context),
    )


# ---------------------------------------------------------------------------
# The menu cache bound
# ---------------------------------------------------------------------------
#
# Two memos below price one rung on one shape: ``_shard_geometry`` here and
# ``tessera_footprint._exact_bits_for_shape``.  Both are keyed by
# ``(family, rung, rows, cols)``, so their size is a product of two factors,
# and both factors are named rather than picked -- see
# :func:`menu_rungs_per_shape` and :func:`menu_cache_shapes`.  The bound they
# multiply out to used to be the literal 4096, which was comfortably more than
# a menu at three families and *less than one shape's menu* at four: a single
# pass over one shape then evicted its own entries and the next pass recomputed
# every one of them (tessera#46).

MENU_CACHE_SHAPES_ENV = "PRISMAQUANT_TESSERA_MENU_CACHE_SHAPES"

#: Distinct 2-D Linear shapes in GLM-5.3-Flash, the largest model this menu is
#: asked about -- **counted, not assumed**.  Reading the safetensors headers of
#: ``/mnt/shared/models/GLM-5.3-Flash-BF16`` (120 shards) finds 37,861 two-
#: dimensional ``*.weight`` tensors outside the embedding and exactly 25
#: distinct ``(rows, cols)`` among them, the widest being ``(154880, 4096)``
#: and the most common ``(2048, 4096)`` at 24,854 tensors.  That ratio is the
#: whole reason these memos exist: a pass walks *units*, and units repeat
#: shapes roughly 1500:1, so a cache that holds every distinct shape of a model
#: of this class prices each one once.  Recount it on another model with::
#:
#:     for each shard: read the safetensors header, count distinct
#:     tuple(v["shape"]) over 2-D "*.weight" entries
DEFAULT_MENU_CACHE_SHAPES = 25


def menu_rungs_per_shape() -> int:
    """The widest menu one shape can produce.  Computed, never guessed.

    :func:`expand_tessera_menu` walks every ``q256`` in each family's
    ``mathematical_q256_bounds`` at ``step_q256=1`` and prices each one at the
    shape it was handed, so this sum *is* the number of entries one pass over
    one shape fills -- 6916 today, against 3055 before the 16-bit family was
    admitted.  It is the floor under :func:`menu_cache_bound`: a memo smaller
    than this evicts a shape's own entries while that shape is still being
    priced, which is exactly the failure tessera#46 reports.

    Asked of :func:`menu_families` rather than listed, so admitting a family
    moves this number and the bound with it.  Deliberately **not** memoised on
    its own: it is a sum over four specs, and a second memo over a memo that a
    test can clear (``menu_families.cache_clear``) is a staleness class bought
    for nothing.
    """
    total = 0
    for spec in menu_families():
        lo, hi = spec.mathematical_q256_bounds
        total += int(hi) - int(lo) + 1
    return total


def menu_cache_shapes() -> int:
    """How many distinct Linear shapes one pass keeps live.  Configured.

    This is the memory decision, and it is the only one here: every other term
    in the bound is computed.  The default is
    :data:`DEFAULT_MENU_CACHE_SHAPES` -- a count taken off a real 122B MoE, not
    a round number -- and ``PRISMAQUANT_TESSERA_MENU_CACHE_SHAPES`` moves it for
    a model with a wider shape roster or a box with less memory to spend.  What
    it may **not** do is drop below one shape.  One shape is not a taste, it is
    the requirement -- a memo that cannot hold the menu it is being asked to
    build evicts its own entries, which is tessera#46 -- so a setting under it
    is refused rather than clamped, and the refusal names the reason.
    """
    import os

    raw = os.environ.get(MENU_CACHE_SHAPES_ENV, "")
    if not raw.strip():
        return DEFAULT_MENU_CACHE_SHAPES
    try:
        shapes = int(raw)
    except ValueError:
        raise TesseraMenuError(
            f"{MENU_CACHE_SHAPES_ENV}={raw!r} is not an integer number of shapes"
        ) from None
    if shapes < 1:
        raise TesseraMenuError(
            f"{MENU_CACHE_SHAPES_ENV}={raw!r}: a menu memo holds whole shapes, "
            f"so it must retain at least one"
        )
    return shapes


#: Shapes the **geometry** memo retains -- one, and the reason is its entry,
#: not its taste.  ``_ShardGeometry.rates`` is Tessera's whole Bresenham column
#: schedule, one integer per column, so an entry costs O(cols) rather than the
#: flat ~350 B a byte total costs: ``tracemalloc`` over a saturating pass
#: measures **8,735 B an entry at 1024 columns and 131,615 B at 16,384**, i.e.
#: 58 MiB and 868 MiB for a single shape's 6916 rungs.  Retaining 25 shapes
#: there would commit 1.4 GiB on GLM's narrowest expert and 21 GiB on its
#: widest -- a memory decision nobody would make deliberately, so it is
#: declined deliberately instead.  One shape is the whole requirement anyway:
#: it is exactly what makes a shape unable to evict itself, and this memo fills
#: on the TP>1 path alone (at tp=1 ``tessera_shape_legal`` answers without it,
#: measured: 0 fills in a 6764-rung research pass).
GEOMETRY_CACHE_SHAPES = 1


def menu_cache_bound(shapes: "int | None" = None) -> int:
    """Entries a per-(rung, shape) menu memo may hold.

    ``menu_rungs_per_shape() * shapes`` -- the widest menu one shape can
    produce, times the number of shapes that memo keeps live.  The first factor
    is computed and the second is stated; neither is a round number chosen to
    be safe, and the product is 172,900 today against the 4096 that could not
    hold one shape.

    ``shapes`` defaults to :func:`menu_cache_shapes`.  It is a parameter
    because the two memos this sizes do **not** cost the same per entry, and
    averaging them would hide a factor of 25 to 375 depending on the column
    count: see :data:`GEOMETRY_CACHE_SHAPES`.

    The memory the default commits is measured, not estimated.  ``tracemalloc``
    around a saturating pass, differenced against the same pass with the memo
    cleared, puts ``_exact_bits_for_shape`` at **351 B an entry** -- and at 351
    B at every shape measured, ``(256,256)``, ``(2048,1024)`` and
    ``(4096,16384)``, because the entry is a ``Fraction`` and a tuple key and
    neither grows with the tensor.  So one retained shape is **2.32 MiB** and
    the default 25 shapes is **57.9 MiB**, for the whole distinct-shape roster
    of a 122B MoE.  ``_shard_geometry`` is the other case and is sized apart.
    """
    per_shape = menu_cache_shapes() if shapes is None else int(shapes)
    return menu_rungs_per_shape() * per_shape


def menu_scaled_cache(fn=None, *, shapes: "int | None" = None):
    """Memoise a per-(rung, shape) function at :func:`menu_cache_bound`.

    ``lru_cache`` fixes its size when the decorator runs, and this bound cannot
    be known then: it asks :func:`menu_families`, which builds every family and
    asks Tessera which grids serialise.  So the memo is built on the first call
    and the wrapper forwards ``cache_info`` / ``cache_clear`` -- which is what
    lets a test read the bound off the live memo instead of restating it.

    ``shapes`` overrides the shape retention for one memo whose entries are not
    the flat ones the default was measured on.

    The mechanism itself lives in ``tessera_formats.lazily_sized_cache``, which
    is the leaf module and where the *other* two lazily sized memos are
    (prismaquant#134): this function is the *bound*, which is the part only the
    menu knows.  Two spellings of one wrapper is how two memos drift apart.
    """
    def decorate(target):
        return lazily_sized_cache(lambda: menu_cache_bound(shapes))(target)

    return decorate if fn is None else decorate(fn)


# ---------------------------------------------------------------------------
# Gate 2: shape, and the TP shard of it
# ---------------------------------------------------------------------------

class _BodyShape(NamedTuple):
    """``unit.body_bits`` seen through the one attribute a granularity reads."""

    shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _ShardGeometry:
    """The fields ``tessera.layout.shard_granularity`` reads off an encoded unit.

    A granularity is a property of the *encoded* unit -- Tessera derives it
    from the very checks ``slice_unit`` applies, so a granularity it reports is
    one that slices -- and principle 14 says PrismaQuant asks for that number
    rather than restating the derivation.  What a menu cannot afford is the
    encode: the answer is wanted for ~3000 rungs on every unit, before anything
    is rendered.

    So this object carries the geometry an encode *would* produce and nothing
    else, and Tessera's own function reads it.  Every field is set explicitly.
    Three of them (``span``, ``body``, ``scale_plane``) are read there through
    ``getattr(unit, name, default)``, so a stub that merely omitted one would
    silently answer for a different wire; ``test_shard_granularity_matches_a_
    real_encoded_unit`` pins the whole object by encoding real units under all
    three scale planes and comparing the two answers.
    """

    body_bits: _BodyShape
    rates: tuple[int, ...]
    span: int
    body: object
    scale_plane: object
    group: int
    half: int
    released_positions: int


@menu_scaled_cache(shapes=GEOMETRY_CACHE_SHAPES)
def _shard_geometry(
    family_name: str, body_rate_q256: int, rows: int, cols: int,
) -> _ShardGeometry:
    """The unit-shaped geometry of one rung at one shape.  Raises if unrealisable.

    Keyed by the family's **name**, and it refuses anything else: a
    ``TesseraFamily`` and its ``.name`` resolve to the same answer through
    ``get_tessera_family`` but are two different memo keys, which would double
    the key space against :func:`menu_cache_bound` and halve the hit rate
    without changing a single number the function returns.  Callers normalise.
    """
    from tessera.manifest import BodyKind, ScalePlaneKind

    from .tessera_render import TESSERA_GROUP, TESSERA_HALF

    if not isinstance(family_name, str):
        raise TesseraMenuError(
            f"_shard_geometry is keyed by family name; got {type(family_name).__name__}"
        )
    spec = get_tessera_family(family_name)
    recipe = tessera_wire_recipe(spec, body_rate_q256)
    body = BodyKind(recipe.body)
    if rows % spec.arity:
        raise TesseraMenuError(
            f"{rows} rows is not a whole number of arity-{spec.arity} tuples"
        )
    try:
        rates = spec.column_schedule(body_rate_q256, cols, recipe=recipe)
    except Exception as exc:  # tessera.grammar's GrammarError, by any name
        raise TesseraMenuError(f"schedule: {exc}") from exc
    return _ShardGeometry(
        body_bits=_BodyShape((rows // spec.arity, int(cols))),
        rates=tuple(int(r) for r in rates),
        # A window body has no super-symbols; ``encode_linear`` resolves the
        # span to 1 there and so does ``render_tessera_weight``.
        span=1 if body is BodyKind.WINDOW else int(recipe.span),
        body=body,
        scale_plane=ScalePlaneKind(recipe.scale_plane),
        group=int(TESSERA_GROUP),
        half=int(TESSERA_HALF),
        # ``render_tessera_weight`` and ``export_checkpoint`` both encode at
        # ``completion=0``, which releases no position. A rung that released
        # any would raise the column granularity to the superblock, which is
        # why this is a stated zero rather than an omitted field.
        released_positions=0,
    )


def tessera_shard_granularity(
    family: "str | TesseraFamily",
    body_rate_q256: int,
    shape: Sequence[int],
) -> tuple[int, int]:
    """``(row_granularity, col_granularity)`` for one rung.  **The one seam.**

    A tensor-parallel rank is handed a *slice* of the Linear, and a Tessera unit
    is not slice-agnostic: the trellis runs down a column across ``arity``-row
    tuples grouped into ``span``-wide super-symbols, the block scale plane tiles
    the input axis in fixed periods, and a mixed Bresenham schedule only closes
    its quota on a whole superblock.  A slice that cuts any of those periods
    produces bytes no reader can decode alone.

    **The number is Tessera's, not ours.**  ``tessera.layout.shard_granularity``
    derives both periods from the checks ``slice_unit`` itself applies; this
    function builds the unit-shaped geometry that rung implies at that shape
    (:class:`_ShardGeometry`) and asks.  Until 2026-09-02 Tessera published no
    such function and this module derived the periods itself, which was a
    restatement waiting to drift -- and it did: the fallback answered
    ``col_granularity == 1`` for every CHANNEL-plane rung, while Tessera's
    answer is the 256-column superblock for any rung whose Bresenham schedule
    is *mixed*, i.e. for all but the handful of integer-rate rungs. That is the
    single most consequential TP fact on this menu, and no amount of care in a
    local derivation would have produced it.

    ``shape`` is the **whole** unit's ``[rows, cols]``: a granularity is a
    property of the unit being cut, and the rate schedule that decides
    ``mixed`` is a function of its column count. The sharded extents are then
    checked against it in :func:`tessera_tp_legal`.
    """
    spec = get_tessera_family(family)
    geometry = _shard_geometry(
        spec.name, int(body_rate_q256), int(shape[-2]), int(shape[-1]),
    )
    from tessera.layout import shard_granularity as _tessera_granularity

    rows, cols = _tessera_granularity(
        geometry, SUPERBLOCK_WEIGHTS, int(spec.arity),
    )
    return int(rows), int(cols)


def shard_shape(
    shape: Sequence[int], tp_degree: int, parallel_kind: str,
) -> tuple[int, ...]:
    """The per-rank shape a TP degree hands the runtime.

    ``parallel_kind`` follows vLLM's own split: column-parallel Linears shard
    the OUTPUT features, row-parallel shard the INPUT features, and a unit no
    rank splits keeps its whole shape.  A packed ``(experts, out, in)`` stack
    is sharded by expert (expert parallelism), which leaves each expert's 2-D
    unit whole -- so it is ``PARALLEL_NONE`` here, and its expert count is not
    this function's business.
    """
    dims = [int(d) for d in shape]
    tp = int(tp_degree)
    if tp < 1:
        raise TesseraMenuError(f"tp_degree must be >= 1, got {tp_degree}")
    if parallel_kind not in _PARALLEL_KINDS:
        raise TesseraMenuError(
            f"unknown parallel kind {parallel_kind!r}; expected one of "
            f"{sorted(_PARALLEL_KINDS)}"
        )
    if tp == 1 or parallel_kind == PARALLEL_NONE:
        return tuple(dims)
    axis = -2 if parallel_kind == PARALLEL_COLUMN else -1
    if dims[axis] % tp:
        # Not a Tessera fact: this shape cannot be served at this TP at all.
        raise TesseraMenuError(
            f"shape {tuple(dims)} is not divisible by tp_degree {tp} on the "
            f"{parallel_kind}-parallel axis"
        )
    dims[axis] //= tp
    return tuple(dims)


def tessera_tp_legal(
    family: "str | TesseraFamily",
    body_rate_q256: int,
    shape: Sequence[int],
    *,
    tp_degree: int = 1,
    parallel_kind: str = PARALLEL_NONE,
    require_attested_world: bool = False,
) -> tuple[bool, str]:
    """Is this rung legal on every rank at ``tp_degree``?  ``(legal, reason)``.

    Three questions when the menu is the attested one, two when it is not.
    ``require_attested_world`` adds the first: does the pinned runtime contract
    say it serves this family at this world size at all
    (:func:`tessera_tp_world_attested`)?  That is a *different* question from
    the geometry below -- a unit can shard perfectly and still have no runtime
    that serves the shards -- and the two legs are kept separate so a refusal
    names which one answered.  The research menu passes ``False``: it prices
    unattested rungs deliberately and stamps every one of them, so a second
    attestation refusal there would only hide the pricing.

    Then two questions, both asked of Tessera:

    * **Can the whole unit be cut this way at all?**
      ``tessera.layout.can_shard(unit, tp, axis)`` -- and the axis mapping is
      Tessera's, not a local convention: a *column-parallel* Linear (q/k/v,
      gate/up) splits vLLM's output features, which are this unit's **rows**,
      so ``PARALLEL_COLUMN`` asks ``axis="row"``.  Inverting that pair would
      gate exactly the wrong half of a model and answer plausibly on both.
    * **Is the rung legal on the shard a rank actually holds?**
      ``tessera_shape_legal`` on the sharded shape: at TP=8 a ``[2048, 1024]``
      column-parallel Linear is eight ``[256, 1024]`` units, each encoded and
      decoded on its own rank, and a rung legal on the whole is not
      automatically legal on an eighth of it (its Bresenham root has to close
      its quota over the shard's own column count).

    ``parallel_kind`` is the unit's, from the caller's serving profile;
    ``PARALLEL_NONE`` (a packed expert under expert parallelism, a replicated
    head) shards nothing and answers the shape question alone.
    """
    spec = get_tessera_family(family)
    tp = int(tp_degree)
    if tp < 1:
        raise TesseraMenuError(f"tp_degree must be >= 1, got {tp_degree}")
    if require_attested_world:
        attested, why = tessera_tp_world_attested(spec, tp)
        if not attested:
            return False, why
    try:
        sharded = shard_shape(shape, tp, parallel_kind)
    except TesseraMenuError as exc:
        return False, f"tp_shape:{exc}"
    if tp > 1 and parallel_kind != PARALLEL_NONE:
        from tessera.layout import can_shard as _tessera_can_shard

        axis = "row" if parallel_kind == PARALLEL_COLUMN else "column"
        try:
            geometry = _shard_geometry(
                spec.name, int(body_rate_q256), int(shape[-2]), int(shape[-1]),
            )
        except TesseraMenuError as exc:
            return False, f"tp{tp}_geometry:{exc}"
        if not _tessera_can_shard(
            geometry, tp, axis, SUPERBLOCK_WEIGHTS, int(spec.arity)
        ):
            row_gran, col_gran = tessera_shard_granularity(
                spec, body_rate_q256, shape,
            )
            gran = row_gran if axis == "row" else col_gran
            extent = int(shape[-2]) if axis == "row" else int(shape[-1])
            return False, (
                f"tp{tp}_{axis}_granularity: {extent} {axis}s cut {tp} ways is "
                f"not a multiple of {gran} (tessera.layout.shard_granularity)"
            )
    return tessera_shape_legal(spec, body_rate_q256, sharded)


def tessera_shape_legal(
    family: "str | TesseraFamily",
    body_rate_q256: int,
    shape: Sequence[int],
) -> tuple[bool, str]:
    """Can Tessera encode this rung on this shape?  ``(legal, reason)``.

    Asked by construction against Tessera's own predicates, never by catching a
    render: an encode is seconds, and a menu that discovers illegality by
    tracing has already paid for it.  The rules, each with the raise site it
    mirrors:

    * ``rows % arity`` -- ``tessera.export`` / ``tessera.layout._counts_for``
    * ``steps % span`` -- ``tessera.encode`` / ``tessera.layout._counts_for``
    * the Bresenham root realisable over the column count -- asked by *calling*
      ``family.column_schedule``, which is ``tessera.grammar``'s own scheduler,
      so this cannot drift from what the encoder will do
    * the block scale plane's period divides ``rows * columns`` -- the plane is
      built by ``flat.reshape(-1, half)``, whose failure is a bare torch
      ``RuntimeError`` rather than a ``GrammarError``, i.e. exactly the one a
      ``except GrammarError`` guard would miss
    """
    spec = get_tessera_family(family)
    dims = tuple(int(d) for d in shape)
    if len(dims) not in (2, 3):
        return False, f"rank{len(dims)}: Tessera takes a 2-D Linear or a 3-D stack"
    rows, cols = dims[-2], dims[-1]
    if rows <= 0 or cols <= 0:
        return False, f"degenerate shape {dims}"
    recipe = tessera_wire_recipe(spec, body_rate_q256)

    from tessera.manifest import BodyKind

    if rows % spec.arity:
        return False, (
            f"arity: {rows} rows is not a whole number of arity-{spec.arity} "
            "tuples"
        )
    body = BodyKind(recipe.body)
    span = 1 if body is BodyKind.WINDOW else int(recipe.span)
    steps = rows // spec.arity
    if steps % span:
        return False, (
            f"span: {steps} trellis positions is not a whole number of span-"
            f"{span} super-symbols"
        )
    try:
        spec.column_schedule(body_rate_q256, cols, recipe=recipe)
    except Exception as exc:  # tessera.grammar's GrammarError, by any name
        return False, f"schedule: {exc}"

    from .tessera_render import TESSERA_GROUP, TESSERA_HALF

    plane = scale_plane_name(recipe.scale_plane)
    positions = rows * cols
    if plane == "s6b" and positions % TESSERA_GROUP:
        return False, (
            f"scale_plane: {positions} positions is not a whole number of "
            f"{TESSERA_GROUP}-weight E8M0 groups"
        )
    if plane in ("s6b", "lut16") and positions % TESSERA_HALF:
        return False, (
            f"scale_plane: {positions} positions is not a whole number of "
            f"{TESSERA_HALF}-weight halves"
        )
    return True, ""


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MenuRung:
    """One priceable rung of one family on one unit, with its route."""

    format_name: str
    family: str
    base: str
    arity: int
    body_rate_q256: int
    #: Exact serialized bits over this unit's parameters -- planes included,
    #: from Tessera's own byte accountant.  A ``Fraction``: the rate is
    #: fractional by construction and a float here would round the byte gate.
    bits_per_param: Fraction
    memory_bytes: int
    admission: RouteAdmission
    #: The TP degree this rung was gated for, stamped so an artifact records
    #: which shard shape its legality was decided on.
    tp_degree: int
    parallel_kind: str

    @property
    def bpp(self) -> float:
        return float(self.bits_per_param)

    @property
    def route_status(self) -> str:
        return self.admission.route_status

    def as_dict(self) -> dict:
        payload = {
            "format": self.format_name,
            "family": self.family,
            "body_rate_q256": self.body_rate_q256,
            "bits_per_param": float(self.bits_per_param),
            "memory_bytes": int(self.memory_bytes),
            "activation_contract": self.admission.activation_contract,
            "route_status": self.admission.route_status,
            "route_status_source": self.admission.source,
            "terminal_format": self.admission.terminal_format,
            "act_bits": self.admission.act_bits,
            "min_capability_sm": self.admission.min_capability_sm,
            "tp_degree": int(self.tp_degree),
            "tp_parallel_kind": self.parallel_kind,
        }
        if self.admission.serving_context is not None:
            payload["serving_context"] = self.admission.serving_context.as_dict()
        return payload


@lru_cache(maxsize=1)
def menu_families() -> tuple[TesseraFamily, ...]:
    """The families a production menu enumerates, asked rather than listed.

    ``tessera_family`` refuses a family whose **body** the encoder cannot
    afford, so the refusal is the filter.  The wall is the TCQ body's, not the
    grid's: a coset step scores ``2^payload_bits`` anchors, so E4M3 at arity 2
    is refused at 65536 of them -- while BF16, whose grid is twice as wide
    again, is admitted, because every rung of it is the WINDOW body, which
    scores ``2^window_bits`` states and has no forest at all.  A family whose
    grid is not in Tessera's ``SERIALISABLE_GRIDS`` is dropped for the same
    reason ``tessera_rung_is_serialisable`` exists: it renders and cannot be
    written.
    """
    from .tessera_formats import _HARDWARE_BASES
    from .tessera_render import family_grid_is_serialisable

    out: list[TesseraFamily] = []
    for base in sorted(_HARDWARE_BASES):
        for arity in range(1, _MAX_ARITY_SEARCH + 1):
            try:
                spec = tessera_family(base, arity)
            except TesseraFormatError:
                continue  # the encoder refuses this family's cost
            try:
                writable = family_grid_is_serialisable(spec)
            except Exception:
                continue
            if writable:
                out.append(spec)
    out.sort(key=lambda f: (f.base, f.arity))
    return tuple(out)


def expand_tessera_menu(
    shape: Sequence[int],
    *,
    mode: str = MENU_ATTESTED,
    families: "Sequence[TesseraFamily] | None" = None,
    step_q256: int = 1,
    tp_degree: int = 1,
    parallel_kind: str = PARALLEL_NONE,
    max_act_bits: "int | None" = None,
    serving_context: "ServingContext | None" = None,
) -> list[MenuRung]:
    """Every Tessera rung legal for one unit, cheapest first.

    The three gates in order, each cheap before the next: family (is it
    buildable and writable), rung (does the shape carry it at this TP), route
    (does a pinned runtime execute it).  The byte price is Tessera's own exact
    accountant at this unit's shape -- not a rate, because a CHANNEL row field
    and a WINDOW table are charged per unit, so the same rung costs a different
    bpp on a ``[3072, 1024]`` Linear than on a ``[1024, 1024]`` one.

    ``max_act_bits`` is the **W(n<=A)** rule: a weight rate above the route's
    activation width buys nothing a wider route would not buy more cheaply, so
    the menu never offers it.  It defaults to the route's own ``act_bits``,
    which is the construction the brief asks for; passing a number narrows it
    further (a target that will only ever serve A4, say).
    """
    if mode not in MENU_MODES:
        raise TesseraMenuError(
            f"unknown menu mode {mode!r}; expected one of {sorted(MENU_MODES)}"
        )
    from .tessera_footprint import tessera_exact_bits_for_shape

    dims = tuple(int(d) for d in shape)
    specs = tuple(families) if families is not None else menu_families()
    n_params = 1
    for d in dims:
        n_params *= d
    out: list[MenuRung] = []
    for spec in specs:
        lo, hi = spec.mathematical_q256_bounds
        for rung in range(lo, hi + 1, max(1, int(step_q256))):
            name = spec.format_name(rung)
            admission = (route_admission(name, serving_context=serving_context)
                         if serving_context is not None else route_admission(name))
            if not admission.admits(mode):
                continue
            legal, _reason = tessera_tp_legal(
                spec, rung, dims,
                tp_degree=tp_degree, parallel_kind=parallel_kind,
                require_attested_world=(mode == MENU_ATTESTED),
            )
            if not legal:
                continue
            recipe = tessera_wire_recipe(spec, rung)
            bits = tessera_exact_bits_for_shape(spec, rung, dims, recipe=recipe)
            bpp = bits / n_params
            # W(n<=A): never offer a weight rate wider than the route serves.
            # The weight rate is the body's coding rate, ``rung / 256`` bits
            # per weight -- not ``bpp``, which also carries the scale plane,
            # a window table or a forest.  Those bytes buy the route nothing
            # either way and are not what "a 4-bit weight" means; comparing
            # ``bpp`` here dropped E2M1x2's attested cap rung (R896, 3.5 b/wt)
            # from the A4 menu the day the forest was charged (#126), and
            # would have dropped every family's cap rung from its own route.
            ceiling = admission.act_bits if max_act_bits is None else max_act_bits
            if ceiling is not None and Fraction(int(rung), Q256_UNIT) > int(ceiling):
                continue
            out.append(MenuRung(
                format_name=name,
                family=spec.name,
                base=spec.base,
                arity=spec.arity,
                body_rate_q256=rung,
                bits_per_param=bpp,
                memory_bytes=int(bits) // 8,
                admission=admission,
                tp_degree=int(tp_degree),
                parallel_kind=parallel_kind,
            ))
    out.sort(key=lambda r: (r.bits_per_param, r.family, r.body_rate_q256))
    return out


# ---------------------------------------------------------------------------
# Making a 3000-rung menu tractable, exactly
# ---------------------------------------------------------------------------

def prune_dominated(
    rows: Sequence[tuple[int, float, object]],
) -> list[tuple[int, float, object]]:
    """Drop only rows another row beats on BOTH axes.  Exact, not a hull.

    ``rows`` is ``(memory_bytes, cost, payload)``.  A row is dominated when
    another row is no larger in bytes and no larger in cost, and strictly
    better on at least one -- which is the only reduction a multi-choice
    knapsack admits without changing its answer, because the DP's budget is
    discrete and a point strictly inside the convex hull can still be the
    optimum for one particular remaining capacity.  Hull pruning would drop
    exactly those points, so it is refused here rather than offered behind a
    flag.

    Ties on both axes keep the first row in the sorted order, so the reduction
    is deterministic.
    """
    ordered = sorted(rows, key=lambda r: (int(r[0]), float(r[1])))
    kept: list[tuple[int, float, object]] = []
    best = float("inf")
    for row in ordered:
        cost = float(row[1])
        if cost < best:
            kept.append(row)
            best = cost
    return kept


def collapse_to_dp_bins(
    rows: Sequence[tuple[int, float, object]],
    *,
    baseline_bits_per_param: float,
    n_params: int,
    total_params: int,
    bit_precision: float,
) -> list[tuple[int, float, object]]:
    """Keep one row per distinct DP bin.  Exact **for this DP**, and says so.

    ``allocator_solver.solve_allocation`` charges a candidate
    ``round(((bpp - baseline_bpp) * n_params/total_params) / bit_precision)``
    bins and can express nothing finer.  Two rungs that land in the same bin are
    indistinguishable *to the solver*, so keeping the lower-cost one changes no
    answer it could have given -- which is a different and weaker statement than
    :func:`prune_dominated`'s, and is why the two are separate functions with
    separately reported counts.

    The distinction matters for reading a receipt.  If an allocation's selected
    rates look coarse, this tells you whether the campaign priced few rungs
    (a campaign result) or the DP's bin width swallowed them (a
    ``--bit-precision`` result).  On a 0.6B model a 3M-parameter Linear holds a
    fraction near 0.007 of the body, so Tessera's 1/256-bpp step is ~2.7e-5
    average bits and the default ``bit_precision=1e-4`` resolves roughly one
    rung in four.  Neither number is a constant here: both are reported.
    """
    from .allocator_solver import _charged_bins

    if total_params <= 0 or n_params <= 0:
        return list(rows)
    fraction = float(n_params) / float(total_params)
    best: dict[int, tuple[int, float, object]] = {}
    for row in sorted(rows, key=lambda r: (int(r[0]), float(r[1]))):
        bits_per_param = float(row[0]) * 8.0 / float(n_params)
        d_avg = (bits_per_param - float(baseline_bits_per_param)) * fraction
        dbins = _charged_bins(d_avg, float(bit_precision))
        prior = best.get(dbins)
        if prior is None or float(row[1]) < float(prior[1]):
            best[dbins] = row
    return [best[k] for k in sorted(best)]


def expand_menu_tokens(names, priced_formats=()) -> list[str]:
    """Replace the ``TESSERA`` menu token with the rungs this run priced.

    ``FORMATS=NVFP4,FP8_DYNAMIC,TESSERA`` is how a launcher asks for "the
    stock two, plus Tessera". The token cannot expand to a static list: a
    family's realisable rungs depend on the unit's column count, and a single
    0.6B Linear carries thousands of them across the four families, so the menu
    is a per-unit object and not a comma-separated string. What it expands to
    instead is the set of Tessera columns the run's own cost table holds --
    every rung an anchor campaign priced, and nothing else.

    That is the honest expansion in both directions. Wider is impossible: a
    rung with no cost row is dropped by ``build_candidates`` regardless.
    Narrower would be a heuristic choosing the DP's candidate set, which
    principle 1 vetoes.

    One filter survives that veto, and it is principle 9's, not an exception:
    **the token expands only to rungs the pinned runtime attests**. A cost
    table is priced under whatever menu mode the campaign ran (``research``
    prices the whole realisable axis on purpose), and the attested set is a
    property of the *runtime*, not of the campaign. So a research-priced table
    read back on the default path holds thousands of columns the pinned
    contract does not publish. Expanding to them would only have
    ``require_producer_formats`` refuse the entire run -- the whole menu, not
    the unbacked part of it -- and the DP would never see the rungs that ARE
    backed. The filter is the same predicate that guard refuses on, so the
    token cannot expand to something the guard then rejects: one rule, two
    uses, no second copy of the legality decision. An explicitly named
    reader-only rung still refuses, exactly as before; only the token narrows,
    and it reports what it narrowed so the run says out loud which axis it is
    allocating over.

    Order is preserved and duplicates removed, so a menu with no token is
    returned unchanged (modulo de-duplication) and no caller needs to know
    whether Tessera is in play.
    """
    return expand_menu_tokens_report(names, priced_formats)[0]


def expand_menu_tokens_report(names, priced_formats=(), *, context_by_unit=None) -> tuple[list[str], list[str]]:
    """``(menu, dropped)`` -- the expansion, and the priced-but-unattested rungs.

    Split out so the caller can *print* the narrowing rather than discover it
    as a smaller menu. ``dropped`` is empty whenever the token is absent.
    With explicit contexts the shared menu is their admitted union; unit-level
    candidate admission, not this union, decides which unit may use each rung.
    """
    from . import format_registry as fr

    if context_by_unit is not None:
        context_by_unit = {context.key(): context for context in context_by_unit.values()
                           if context is not None}
    priced_all = [
        str(name) for name in priced_formats
        if isinstance(name, str) and name.startswith("TESSERA_")
    ]
    dropped: list[str] = []
    priced: list[str] = []
    for name in priced_all:
        eligible = fr.format_is_producer_eligible(name, **(
            {"context_by_unit": context_by_unit} if context_by_unit is not None else {}))
        (priced if eligible else dropped).append(name)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        expansion = priced if str(name) == MENU_TOKEN else [str(name)]
        for item in expansion:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out, ([] if not any(str(n) == MENU_TOKEN for n in names) else dropped)


#: How a family's rungs are priced. Today every family answers the same way,
#: and the value of naming it per family is that the answer is a property OF
#: the family rather than a global assumption -- Tessera's embedded axis is a
#: decode-time completion axis, so it produces weights and not wires, and a
#: WINDOW body has no completion axis at all.
ANCHOR_FRESH_ENCODE = "fresh_encode_per_anchor"


def family_anchor_rule(family: "str | TesseraFamily") -> str:
    """How this family's anchors must be produced.

    ``fresh_encode_per_anchor`` for every family that exists today, and the
    reason is measured rather than assumed: ``encode_linear`` writes
    ``completion=0``, ``build_unit_artifact`` writes exactly one terminal, and
    the WINDOW body has no completion axis at all, so **there is no truncated
    wire** -- the "embedded ladder" is a decode-time completion axis producing
    weights, not a cheaper way to serialise a lower rate. A family that one day
    does write a nested wire answers differently here and the campaign reads
    this, rather than the campaign carrying a global assumption that a new
    family would silently inherit.
    """
    from .tessera_formats import get_tessera_family

    get_tessera_family(family)          # refuses an unknown family
    return ANCHOR_FRESH_ENCODE


def assert_uniform_hessian_identity(costs: "dict") -> dict:
    """Every Tessera row in one cost table must share one Hessian identity.

    The encoder's shipping default consumes a per-unit ``XᵀX``; a row priced
    without one describes different bytes than a row priced with one, at the
    same format name and the same bpp. A table holding both is not a menu with
    a caveat, it is two menus wearing one label, and the DP will happily trade
    a row of the first against a row of the second.

    This is the merge guard the encoder-drift incident argued for, applied to
    the cost table instead of the encoder: two halves built by different code
    were invisible precisely because nothing compared them. Returns the single
    identity (as a dict) for stamping, or raises.

    Rows written before this field existed carry no ``hessian_identity``; they
    are counted and reported as ``unstamped`` rather than assumed to match,
    because "no claim" and "a matching claim" are different facts (P14).

    The identity includes ``text_sha`` even when ``supplied`` is false, so two
    weights-only campaigns on different calibration draws are refused as mixed
    although their *bytes* are identical. That is deliberate and not a bug: the
    bytes are draw-independent but the **scores** attached to them are not --
    ``output_mse`` is measured against that draw's activations -- and it is the
    scores the DP trades.

    The per-row ``applied`` flag is deliberately **not** part of the key. At the
    pinned Tessera an activation-aware encode needs a CHANNEL scale plane, so an
    E4M3 row is H-aware and an E2M1 row cannot be, in the same table, from one
    campaign, on one draw. Those rows *are* comparable -- each is the price of
    its own rung's shipping bytes -- and refusing the table would refuse the
    mixed-family menu this whole stage exists to build. What must match is the
    DRAW and the run's intent, which is what the key holds.

    **The key is the whole identity, the required triple first.**  Tessera's
    own required roster -- ``text_sha256`` / ``fit_tokens`` / ``fit_ids_sha256``,
    read from ``tessera_hessian.HESSIAN_IDENTITY_FIELDS`` rather than retyped
    -- is what names *which* Hessian shaped the bytes, and until 2026-09-05
    this guard compared only the legacy ``(supplied, text_sha, token_count,
    kwarg)`` projection, so any number of distinct modern identities collapsed
    to one key and a table merged from two draws allocated as one
    (RobTand/prismaquant#195).  Three row classes now exist and are refused
    from merging with each other:

    * **modern** -- carries the full required triple; keyed on the triple AND
      the legacy aliases, so aliases that agree cannot launder a triple that
      does not.
    * **legacy** -- carries none of the triple (written before it existed);
      keyed on the legacy fields alone, and never merged with a modern row:
      "no claim about the triple" and "a matching triple" are different facts
      (P14).
    * **partial** -- carries some of the triple; that is a malformed identity
      and is refused by name rather than downgraded to the legacy comparison.

    The returned dict carries the canonical triple beside the legacy
    projection, so the allocator's ``tessera_hessian`` metadata block -- which
    stamps this return verbatim into ``layer_config.json`` -- gives export a
    Hessian identity it can bind a capture against.

    **The capture's content digest is part of the key too**
    (RobTand/prismaquant#204).  The triple names the token draw and not the
    Hessian: two captures of one draw with different ``H`` -- a different
    sequence layout, a re-capture, a rewritten payload -- carry one triple
    and encode different bytes at the same format name.  The campaign stamps
    every row with ``capture_sha256``, the digest of exactly the payload it
    wrote for the export leg, and rows carrying two digests under one triple
    are two captures, refused from merging.  ``None`` (a weights-only
    campaign, or a table written before the digest existed) is its own
    value, never wildcarded against a digest: the returned ``capture_sha256``
    is what the export gate binds the payload to, and a table without one
    yields an allocation the gate refuses as unbound.
    """
    from .tessera_hessian import HESSIAN_IDENTITY_FIELDS

    required = tuple(HESSIAN_IDENTITY_FIELDS)
    seen: dict[tuple, int] = {}
    unstamped = 0
    legacy_rows = 0
    for _qname, rows in (costs or {}).items():
        if not isinstance(rows, dict):
            continue
        for fmt, row in rows.items():
            if not isinstance(row, dict) or parse_tessera_format_name(fmt) is None:
                continue
            ident = row.get("hessian_identity")
            if not isinstance(ident, dict):
                unstamped += 1
                continue
            modern = tuple(ident.get(field) for field in required)
            if any(value is None for value in modern):
                if any(value is not None for value in modern):
                    raise ValueError(
                        f"{_qname}[{fmt}]: hessian_identity carries a partial "
                        f"required triple "
                        f"{ {f: ident.get(f) for f in required} }; "
                        f"tessera.export.HESSIAN_IDENTITY requires all of "
                        f"{list(required)}. A partial identity cannot name a "
                        "draw and is refused rather than collapsed to the "
                        "legacy fields."
                    )
                schema, modern = "legacy", None
                legacy_rows += 1
            else:
                schema = "modern"
            key = (schema, modern, bool(ident.get("supplied")),
                   ident.get("text_sha"), ident.get("token_count"),
                   ident.get("kwarg"), ident.get("capture_sha256"))
            seen[key] = seen.get(key, 0) + 1
    if len(seen) > 1:
        raise ValueError(
            "Tessera cost table mixes Hessian identities: "
            + "; ".join(
                f"{k[0]} triple={None if k[1] is None else tuple(str(v)[:12] for v in k[1])} "
                f"supplied={k[2]} text_sha={str(k[3])[:12]} "
                f"tokens={k[4]} kwarg={k[5]} "
                f"capture_sha256={None if k[6] is None else str(k[6])[:12]} "
                f"({n} rows)"
                for k, n in sorted(seen.items(), key=lambda kv: -kv[1]))
            + ". Rows priced with and without a Hessian, on different "
              "calibration draws, or against different captures of one draw, "
              "are not comparable prices of the same bytes. Rebuild the cost "
              "table with one campaign, or allocate them separately."
        )
    (key,) = tuple(seen) if seen else (None,)
    triple = dict(zip(required, key[1])) if key is not None and key[1] is not None \
        else {field: None for field in required}
    return {
        "supplied": None if key is None else key[2],
        "text_sha": None if key is None else key[3],
        "token_count": None if key is None else key[4],
        "kwarg": None if key is None else key[5],
        **triple,
        "capture_sha256": None if key is None else key[6],
        "identity_schema": None if key is None else key[0],
        "stamped_rows": sum(seen.values()),
        "legacy_rows": int(legacy_rows),
        "unstamped_rows": int(unstamped),
    }


def priced_static_scales(assignment: "Mapping[str, str]",
                         costs: "Mapping") -> dict:
    """The static A-side scale VALUE each selected Tessera unit was priced under.

    ``{"schema": PRICED_STATIC_SCALES_SCHEMA, "units": {unit: scale}}`` for
    every unit the assignment gives a Tessera format whose cost row carries
    ``input_global_scale`` (the campaign stamps it on every row whose route
    executes the static NVFP4 contract, measured and interpolated alike).
    Empty ``units`` when no selected row carries one -- still a claim, and a
    different one from the block being absent.

    The allocator stamps this beside ``tessera_hessian`` in the layer_config
    metadata; the export gate (``tessera_export_lane.
    require_priced_export_inputs``) reads the file the exporter is handed
    and refuses a W4A4 unit whose served scalar is not the value here
    (RobTand/prismaquant#204).  Until then the gate checked only that the
    unit's key existed in the file, so a file carrying 1.0 and one carrying
    10000.0 at the same key were both "the priced scales".

    A selected W4A4 unit whose row carries no scale is deliberately left
    out rather than given a default: the gate refuses it by name as unbound,
    which is the honest answer for a row that never said what priced it.
    """
    from .tessera_export_lane import PRICED_STATIC_SCALES_SCHEMA

    units: dict[str, float] = {}
    for name, fmt in sorted((assignment or {}).items()):
        fmt = str(fmt)
        if parse_tessera_format_name(fmt) is None:
            continue
        rows = (costs or {}).get(name)
        row = rows.get(fmt) if isinstance(rows, Mapping) else None
        scale = row.get("input_global_scale") if isinstance(row, Mapping) \
            else None
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            continue
        units[name] = float(scale)
    return {"schema": PRICED_STATIC_SCALES_SCHEMA, "units": units}


__all__ = list(__all__) + ["assert_uniform_hessian_identity",
                           "priced_static_scales"]


# ---------------------------------------------------------------------------
# The surrogate does not rank Tessera rungs.  Measured 2026-09-02, served.
# ---------------------------------------------------------------------------
#
# ``docs/measurements/tessera-allocated-served-2026-09-02.md`` took this menu's
# own allocations through export and serve on Qwen3-0.6B and compared them with
# a **byte-matched uniform** arm.  The allocation lost at every budget: served
# KL-vs-BF16 0.3485 against 0.1746 at 4.0 bpp, 2.33x at 3.0, 2.88x at 5.0, with
# bytes exact to the bit and 112/112 modules on the declared route.
#
# The loss is in the cost model, on the units the cost model priced: a separator
# pair serving only the seven measured Linears reads 0.02517 allocated against
# 0.01306 uniform (1.93x), which is 95% of the whole-body gap in log terms, and
# the same seven units score 1.13x *better* in the allocator's own currency.
#
# It is a **slope** error, not an ordering error, and that is why nothing in
# this module caught it.  ``TesseraRateSurface`` refuses a non-monotone anchor
# set rather than laundering it into a cost, so the ORDINAL assumption is
# guarded -- but the measured failure never violates it: the surrogate got the
# sign of the expensive move right (it charged ``down_proj`` at R749 3.69x, its
# largest penalty) and mispriced the *gain* from bits above R1006, scoring the
# other six moves as a 1.30x net win where serving says 1.19x loss.  Every
# construction downstream of the cost -- surface interpolation, dominance
# pruning, bin collapse, the group Minkowski fold, the DP itself -- is exact
# GIVEN the cost and adds no independent check on it.
#
# So this is recorded, not repaired (the repair is not this seam's), and it is
# recorded where a machine can read it: printed by the allocator and stamped
# into every allocation's provenance, because a terminal warning is not a
# property of the artifact.
TESSERA_SURROGATE_RANK_MEASUREMENT = (
    "docs/measurements/tessera-allocated-served-2026-09-02.md"
)


def surrogate_selection_caveat() -> dict:
    """The measured status of surrogate ranking on Tessera rungs.

    Returned as data so it lands in ``layer_config.json`` provenance and a
    shipcard reader can refuse on it, rather than living only in a log line.
    """
    return {
        "surrogate_ranks_tessera_rungs": "measured_false",
        "measurement": TESSERA_SURROGATE_RANK_MEASUREMENT,
        "model": "Qwen3-0.6B (dense), served through the Tessera plugin",
        "served_kl_allocated_over_byte_matched_uniform": {
            "3.0": 2.33, "4.0": 2.00, "5.0": 2.88,
        },
        "priced_units_only": {
            "allocated": 0.02517, "uniform": 0.01306, "ratio": 1.93,
        },
        "failure_mode": (
            "rate-distortion slope: the sign of a deep cut is priced right and "
            "the gain from bits above the uniform rung is overstated, so the "
            "allocator finances cuts with gains that do not exist"
        ),
        "guarded_by_this_module": (
            "ordinal only -- TesseraRateSurface refuses a non-monotone anchor "
            "set; a slope error inside a monotone surface is invisible to it"
        ),
        "requires_before_promotion": [
            "SELECTION_MODE=validated-surrogate (real held-out KL selects)",
            "a byte-matched uniform arm served beside the candidate",
        ],
        "status": "this assignment is a CANDIDATE, not a selection",
    }
