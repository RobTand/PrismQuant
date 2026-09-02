"""The production menu over Tessera's continuous rate axis.

``tessera_formats`` says which rungs *exist* (~3000 of them across three
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

Gate 3 is closed for every Tessera rung today
---------------------------------------------
The pinned serving release is ``gridbook 0.9.1``, whose contract publishes the
families ``NVFP4_CB_K``, ``FP8_CB_K``, ``TCQ_E2M1_R256`` and ``TCQ_E4M3_R256``.
``gridbook_lane_eligibility.resolve_payload_rung`` matches a format name against
those families' ``name_pattern`` heads, and no Tessera name
(``TESSERA_E2M1_K2_R896``) starts with one, so every Tessera rung resolves to a
family the table does not govern and :data:`ROUTE_STATUS_UNATTESTED` is the
honest answer.  Re-pinning to a release that publishes Tessera's own rows is
what flips it -- an edit here cannot, and that is the point.

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
Three families address 3075 rungs at every shape.  Rendering all of them is not
a cost model, it is a week.  :mod:`prismaquant.tessera_campaign` renders a small
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

from .gridbook_lane_eligibility import (
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
    parse_tessera_format_name,
    scale_plane_name,
    tessera_family,
    tessera_serving_route,
    tessera_wire_recipe,
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
    "collapse_to_dp_bins",
    "expand_tessera_menu",
    "menu_families",
    "menu_mode",
    "prune_dominated",
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
    #: assert on its own; it is not the attestation.
    activation_contract: str
    terminal_format: "str | None"
    act_bits: "int | None"
    act_group_size: "int | None"
    min_capability_sm: int
    #: ``backed`` / ``backed_with_serve_flag`` / ``unattested``.
    route_status: str
    #: Can Tessera's wire carry these bytes at all?  Independent of the runtime.
    serialisable: bool
    #: Which table answered.  Swaps when the Tessera contract lands.
    source: str
    detail: str = ""

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


#: Where the attestation is read from today.  When Tessera publishes its own
#: ``runtime_contract.json`` this string changes with the lookup below, and it
#: travels into every unit's provenance so an artifact records which table
#: admitted it.
_ATTESTATION_SOURCE = "gridbook_serving_runtime_pin:lane_eligibility"


def route_admission(name: str) -> RouteAdmission:
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
    route 0.006 ms), the expensive part of a menu is the per-rung Bresenham
    realisability check, and ``expand_tessera_menu`` is memoised per
    ``(shape, mode, tp)`` above it.

    Today it delegates to ``tessera_render.tessera_lane_attested``, which
    resolves the rung against the pinned Gridbook release's ``lane_eligibility``
    cells.  That table publishes no Tessera family, so the honest answer for
    every rung is :data:`ROUTE_STATUS_UNATTESTED` -- absence, not ``unbacked``,
    because the contract is a closed world about the families it *does* publish
    and says nothing at all about these bytes.
    """
    from .tessera_render import (
        tessera_lane_attested, tessera_rung_is_serialisable,
    )

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        raise TesseraMenuError(f"{name!r} is not a Tessera format name")
    family, rung = parsed
    recipe = tessera_wire_recipe(family, rung)
    route = tessera_serving_route(family, recipe, rung)
    serialisable = tessera_rung_is_serialisable(name)

    attested = bool(tessera_lane_attested(name))
    if attested:
        status = ROUTE_STATUS_BACKED
        detail = "the pinned serving release publishes a cell naming this rate"
    else:
        status = ROUTE_STATUS_UNATTESTED
        detail = (
            "the pinned serving release publishes no cell covering this "
            "family and rate"
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
        source=_ATTESTATION_SOURCE,
        detail=detail,
    )


#: Lane id stamped on a Tessera candidate's resolved route. Deliberately not
#: one of the profile-declared lane ids: no serving-profile spec declares a
#: Tessera lane, and inventing a spec entry that says one exists would be the
#: hand-typed verdict principle 14 refuses. This id says only "this route was
#: resolved by the Tessera admission seam", and ``route_status`` carries the
#: verdict.
TESSERA_LANE_ID = "tessera_admission"


def tessera_resolved_serving_lane(name: str, *, runtime_version: str = ""):
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
    asserted, and ``fused_mid_m_backed`` is False because no pinned release
    publishes a fused mid-M table for these bytes -- absence, priced as
    absence.
    """
    from .serving_profiles import ResolvedServingLane, gridbook_runtime_version

    if parse_tessera_format_name(name) is None:
        return None
    admission = route_admission(name)
    return ResolvedServingLane(
        lane_id=TESSERA_LANE_ID,
        format=name,
        activation_contract=admission.activation_contract,
        fallback_route="expand_and_gemm",
        fused_mid_m_backed=False,
        fused_mid_m_rungs=(),
        fused_mid_m_range=None,
        runtime_version=(runtime_version or gridbook_runtime_version()),
        rungs_source=admission.source,
        rung=None,
        detail=admission.detail,
        route_status=admission.route_status,
        requires_serve_flags=(),
        route_status_source=admission.source,
    )


# ---------------------------------------------------------------------------
# Gate 2: shape, and the TP shard of it
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def tessera_shard_granularity(
    family: "str | TesseraFamily", body_rate_q256: int,
) -> tuple[int, int]:
    """``(row_granularity, col_granularity)`` for one rung.  **The one seam.**

    A tensor-parallel rank is handed a *slice* of the Linear, and a Tessera unit
    is not slice-agnostic: the trellis runs down a column across ``arity``-row
    tuples grouped into ``span``-wide super-symbols, and the block scale plane
    tiles the input axis in fixed periods.  A slice that cuts either period
    produces bytes no reader can decode alone.

    **Where the numbers come from.**  Tessera will publish
    ``tessera.layout.shard_granularity(unit)``; until it does, both are derived
    from the wire's own plane geometry rather than typed:

    * **rows** -- ``arity * span``.  ``layout._counts_for`` refuses a geometry
      whose ``rows % arity`` is non-zero and whose ``steps % span`` is non-zero
      (``GrammarError``), and ``encode_unit`` raises the same two.  A WINDOW
      body is ``span == 1``, so its row period is the arity alone -- 1 for
      E4M3, 2 for E2M1x2 -- which is the "1 for WINDOW bodies" statement made
      exact for a tuple grid.
    * **columns** -- the scale plane's period along the input axis.
      ``_counts_for`` sizes ``SCALE_REFINE`` as ``positions // half_weights``
      (16) and ``SCALE_BASE`` as ``positions // group_weights`` (32), and the
      halves run row-major along the input axis, so an S6b plane cuts on 32, a
      LUT plane on 16, and a CHANNEL plane -- one fp16 per output row, no block
      plane at all -- on 1.

    The *rate schedule* is a second, rung-dependent column constraint that is
    not a granularity: a Bresenham root is realisable over ``n`` columns only
    when the fractional part times ``n`` is an integer.  That is checked
    directly, against Tessera's own scheduler, in :func:`tessera_shape_legal`.
    """
    spec = get_tessera_family(family)
    recipe = tessera_wire_recipe(spec, body_rate_q256)
    try:  # Tessera's own answer, the moment it exists.
        from tessera.layout import shard_granularity as _tessera_granularity
    except ImportError:
        _tessera_granularity = None
    if _tessera_granularity is not None:  # pragma: no cover - lands with tessera
        rows, cols = _tessera_granularity(spec.payload_grid(), recipe)
        return int(rows), int(cols)

    from tessera.manifest import BodyKind

    body = BodyKind(recipe.body)
    span = 1 if body is BodyKind.WINDOW else int(recipe.span)
    row_granularity = int(spec.arity) * span

    from .tessera_render import TESSERA_GROUP, TESSERA_HALF

    plane = scale_plane_name(recipe.scale_plane)
    if plane == "s6b":
        col_granularity = int(TESSERA_GROUP)
    elif plane == "lut16":
        col_granularity = int(TESSERA_HALF)
    else:  # "channel": one fp16 per output row, no block plane on the columns
        col_granularity = 1
    return row_granularity, col_granularity


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
) -> tuple[bool, str]:
    """Is this rung legal on every rank at ``tp_degree``?  ``(legal, reason)``.

    The gate is the shard's shape, not the whole tensor's: at TP=8 a
    ``[2048, 1024]`` column-parallel Linear is eight ``[256, 1024]`` units, each
    encoded and decoded on its own rank, and a rung that is legal on the whole
    is not automatically legal on an eighth of it.  Both periods bind --
    ``row_granularity`` on the sharded output axis, ``col_granularity`` and the
    Bresenham root on the sharded input axis.
    """
    spec = get_tessera_family(family)
    tp = int(tp_degree)
    if tp < 1:
        raise TesseraMenuError(f"tp_degree must be >= 1, got {tp_degree}")
    try:
        sharded = shard_shape(shape, tp, parallel_kind)
    except TesseraMenuError as exc:
        return False, f"tp_shape:{exc}"
    row_gran, col_gran = tessera_shard_granularity(spec, body_rate_q256)
    rows, cols = int(sharded[-2]), int(sharded[-1])
    if rows % row_gran:
        return False, (
            f"tp{tp}_row_granularity: {rows} sharded rows is not a multiple of "
            f"{row_gran} (arity x span)"
        )
    if cols % col_gran:
        return False, (
            f"tp{tp}_col_granularity: {cols} sharded columns is not a multiple "
            f"of {col_gran} (scale-plane period)"
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
        return {
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


@lru_cache(maxsize=1)
def menu_families() -> tuple[TesseraFamily, ...]:
    """The families a production menu enumerates, asked rather than listed.

    ``tessera_family`` refuses a family the encoder cannot afford (E4M3 at
    arity 2 needs 65536 anchors scored per trellis step, above the encoder's own
    wall), so the refusal is the filter.  A family whose grid is not in
    Tessera's ``SERIALISABLE_GRIDS`` is dropped for the same reason
    ``tessera_rung_is_serialisable`` exists: it renders and cannot be written.
    """
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest

    from .tessera_formats import _HARDWARE_BASES

    out: list[TesseraFamily] = []
    for base in sorted(_HARDWARE_BASES):
        for arity in range(1, _MAX_ARITY_SEARCH + 1):
            try:
                spec = tessera_family(base, arity)
            except TesseraFormatError:
                continue  # the encoder refuses this family's cost
            try:
                digest = grid_digest(spec.payload_grid())
            except Exception:
                continue
            if digest in SERIALISABLE_GRIDS:
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
            admission = route_admission(name)
            if not admission.admits(mode):
                continue
            legal, _reason = tessera_tp_legal(
                spec, rung, dims,
                tp_degree=tp_degree, parallel_kind=parallel_kind,
            )
            if not legal:
                continue
            recipe = tessera_wire_recipe(spec, rung)
            bits = tessera_exact_bits_for_shape(spec, rung, dims, recipe=recipe)
            bpp = bits / n_params
            # W(n<=A): never offer a weight rate wider than the route serves.
            ceiling = admission.act_bits if max_act_bits is None else max_act_bits
            if ceiling is not None and bpp > Fraction(int(ceiling)):
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
    0.6B Linear carries ~3000 of them across the three families, so the menu
    is a per-unit object and not a comma-separated string. What it expands to
    instead is the set of Tessera columns the run's own cost table holds --
    every rung an anchor campaign priced, and nothing else.

    That is the honest expansion in both directions. Wider is impossible: a
    rung with no cost row is dropped by ``build_candidates`` regardless, and
    naming it in the menu only makes ``require_producer_formats`` refuse the
    whole run. Narrower would be a heuristic choosing the DP's candidate set,
    which principle 1 vetoes.

    Order is preserved and duplicates removed, so a menu with no token is
    returned unchanged (modulo de-duplication) and no caller needs to know
    whether Tessera is in play.
    """
    priced = [
        str(name) for name in priced_formats
        if isinstance(name, str) and name.startswith("TESSERA_")
    ]
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        expansion = priced if str(name) == MENU_TOKEN else [str(name)]
        for item in expansion:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


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
