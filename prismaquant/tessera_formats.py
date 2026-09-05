"""Tessera's grid-space, in the shape PrismaQuant's rate-distortion allocator wants.

This module is the seam that replaces ``trellis_formats`` (archived
2026-09-02 under ``archive/trellis_wire_2026-09-02/``, #118).  The story is
worth stating because the diff looks bigger than the change is.

``trellis_allocator`` (1768 lines) and ``trellis_rate_surface`` (876) were
written for the Gridbook rate-256 tail-biting trellis, but neither file mentions
Gridbook or TCQ even once: they are exact-marginal pricing, Pareto frontiers,
lambda choice, RD hulls, leave-one-anchor-out and allocation regret -- pricing
machinery that happens to have been pointed at a trellis.  Between them they
consume exactly **five** and **seven** names from ``trellis_formats``, and the
load-bearing one is a frozen dataclass describing a *family*.

So retiring the Gridbook attempt does not mean deleting the pricing.  It means
pointing that seam at Tessera.

**This is not building Tessera out of Gridbook's internals.**  The direction is
the opposite one: the pricing machinery is PrismaQuant's, the format authority
is Tessera's, and the Gridbook-specific vocabulary is what gets walled off.
Nothing here reimplements Tessera -- every number is read from the ``tessera``
package, and a second copy of a rate constant is a drift bug waiting for a rate
to change.

The space, and why it is continuous
-----------------------------------
A family is a **(base grid, arity)** pair.  A base grid of ``G`` codes at arity
``k`` gives a code space of ``G**k``, so ``payload_bits = k*log2(G)``, and the
rate cap is a property of the *body* the family's wire recipe names: the TCQ
trellis spends one payload bit on its convolutional code, so its cap is
``payload_bits - 1`` (``|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R)`` has to close at
``2^payload_bits`` exactly), while the WINDOW body shapes with history rather
than a code bit and caps at ``payload_bits``.  Ask :func:`family_rate_cap`,
never the subtraction.  A code covers ``k`` positions, so the
per-position body rate is ``rate/k`` -- which is how ``k=2`` fills the rungs
between the ``k=1`` ones, and how 4.0 bpp becomes addressable at all.

Within a family the rate axis is **continuous at a 1/256-bpp quantum**, not a
handful of rungs.  A root rate is realised as a mixed per-column Bresenham
schedule, and a root is realisable over ``n`` columns exactly when
``(root - floor(root)) * n`` is an integer.  Rates are quoted in q256 -- 256ths
of a bit -- and the superblock is 256 columns, so that product is
``root_q256 mod 256``, an integer by construction.  **The q256 grid is exactly
the realisable set at superblock scale.**  Verified exhaustively: ~9500 rungs
across ten families spanning 1.00 to 8.00 bpp, zero unrealisable
(``tessera/tests/test_grid_space_continuity.py``).

Realisability is not quality.  This module says which rungs can be *encoded*;
which are worth encoding is a measurement, and only four of them have one.
"""
from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, wraps
import json
import re
import threading

__all__ = [
    "ANCHOR_BUDGET_BITS",
    "Q256_UNIT",
    "RATE_SURFACE_ADAPTIVE",
    "RATE_SURFACE_ALL_LEGAL",
    "RATE_SURFACE_DENSE",
    "RATE_SURFACE_MODES",
    "FUSED_MODULE_RATE_FIELD",
    "FUSED_MODULE_RUNG_FIELDS",
    "FUSED_MODULE_SHAPE_FIELDS",
    "SCALE_PLANE_BITS_Q256",
    "SCALE_LUT_BITS_Q256",
    "SCALE_PLANE_NAMES",
    "TesseraServingRoute",
    "family_q256_bounds",
    "fused_shared_signature",
    "family_rate_cap",
    "materialised_terminal_format",
    "tessera_serving_route",
    "recipe_from_wire_names",
    "scale_plane_name",
    "tessera_wire_defaults",
    "clear_recipe_cache",
    "tessera_wire_recipe",
    "wire_overhead_q256",
    "SUPERBLOCK_WEIGHTS",
    "TesseraFamily",
    "TesseraFormatError",
    "TesseraRateSurface",
    "artifact_bpp",
    "enumerate_grid_space",
    "family_cache_bound",
    "get_tessera_family",
    "grid_space_rung_keys",
    "lazily_sized_cache",
    "recipe_cache_bound",
    "parse_tessera_format_name",
    "realisable_rungs",
    "tessera_family",
    "validate_body_rate_q256",
]


class TesseraFormatError(ValueError):
    """A Tessera family, rung, or rate schedule is invalid."""


try:  # pragma: no cover - exercised by the import-failure path
    from tessera.alphabet import (
        BF16_GRID, E2M1_GRID, E4M3_GRID, lloyd_max_grid, tuple_grid,
    )
    from tessera.errors import GrammarError
    from tessera import export as _tessera_export
    from tessera.export import WireRecipe, wire_recipe
    from tessera.grammar import (
        Q256_UNIT,
        bresenham_rate_schedule,
        forest_plane_bytes,
        root_from_q256,
        superblock_quota_ok,
    )
    from tessera.manifest import BodyKind, ScalePlaneKind
except ImportError as exc:  # pragma: no cover
    raise TesseraFormatError(
        "prismaquant.tessera_formats requires the `tessera` package, which is "
        "the authority for Tessera's grammar.  Install it editable from the "
        "Tessera checkout (`pip install -e /path/to/tessera --no-deps`).  This "
        "module deliberately carries no copy of the constants."
    ) from exc


#: Tessera's superblock, in positions.  The scale plane is quoted per
#: superblock and the rate quota is kept per superblock, so this is the unit
#: the allocator already accounts in.
SUPERBLOCK_WEIGHTS = 256

# ``MIN_TRELLIS_STEPS`` used to live here (value 8): a per-256-column floor
# on genuinely coded (non-bypass) positions, inherited verbatim from the
# retired Gridbook wire (``archive/trellis_wire_2026-09-02/``,
# ``MIN_TRELLIS_STEPS = STATE_MEMORY_BITS``).  It is deleted, not lowered,
# because the authority it deferred to -- ``tessera.grammar`` /
# ``tessera.manifest`` -- has no such floor: a schedule is legal iff every
# rate is in ``1..cap`` and the quota closes (``validate_rate_schedule`` +
# ``superblock_quota_ok``), and "bypass" is the old wire's word --
# ``TesseraFamily`` has no ``bypass_rate``, every scheduled rate is a coded
# trellis rate.  Verified against the installed ``tessera`` package 2026-09-03
# (``grammar`` exposes no minimum-step constant; ``manifest`` validates only
# length, quota and per-superblock quota; ``trellis.ConvCode`` refuses only
# ``memory < 1``, a property of the code, not of a schedule).  Do not re-add
# a floor here: ask ``TesseraFamily.column_schedule``, which is
# ``bresenham_rate_schedule`` and refuses what the wire refuses.

#: The S6b scale plane: an E8M0 base byte per group of 32 plus a 4-bit
#: refinement per half of 16.  A flat 0.5 bits per position on top of the body.
#: Priced here so the allocator never has to know the layout.
SCALE_PLANE_BITS_Q256 = Q256_UNIT // 2
#: The LUT scale plane (tessera schema minor 1, 2026-09-01): the same 4-bit
#: word per half of 16, now an index into a per-unit sixteen-entry E4M3 table
#: carried in the manifest, and no base plane.  A flat 0.25 bits per position.
SCALE_LUT_BITS_Q256 = Q256_UNIT // 4


#: The wire's spelling of a scale plane, in the vocabulary this module and
#: ``tessera_footprint`` have always quoted planes in.  ``channel`` is schema
#: minor 3: one fp16 per output row on DIAG_SV, no block plane at all.
SCALE_PLANE_NAMES: Mapping[int, str] = {
    ScalePlaneKind.S6B: "s6b",
    ScalePlaneKind.LUT: "lut16",
    ScalePlaneKind.CHANNEL: "channel",
}
_PLANE_KINDS: Mapping[str, "ScalePlaneKind"] = {
    name: kind for kind, name in SCALE_PLANE_NAMES.items()
}


def scale_plane_name(plane: "ScalePlaneKind | str") -> str:
    """``ScalePlaneKind`` -> the string this module prices, or pass a string through."""
    if isinstance(plane, str):
        if plane not in _PLANE_KINDS:
            legal = ", ".join(sorted(_PLANE_KINDS))
            raise TesseraFormatError(f"unknown scale plane {plane!r}; {legal}")
        return plane
    try:
        return SCALE_PLANE_NAMES[ScalePlaneKind(plane)]
    except (KeyError, ValueError) as exc:
        raise TesseraFormatError(f"unknown scale plane {plane!r}") from exc


# ---------------------------------------------------------------------------
# Sizing a memo over the space it is keyed on
# ---------------------------------------------------------------------------
#
# Every memo on this path is keyed on a *rung*, and there are thousands of
# them: one pass over the space asks each key once, so a memo smaller than the
# space evicts its own entries mid-pass and answers nothing on the next pass.
# That is RobTand/tessera#46, and a bigger literal is the same defect with a
# bigger number in it -- ``enumerate_grid_space`` went from eleven families to
# twelve on 2026-09-02 with nobody touching a cache line.  So the bound is
# computed from the space, and the mechanism below exists because the space
# cannot be counted at import time (counting it builds every grid).


def lazily_sized_cache(bound):
    """``lru_cache`` whose ``maxsize`` is computed on the first call.

    ``lru_cache`` fixes its size when the decorator runs, which for a bound
    derived from :func:`enumerate_grid_space` is too early: counting the space
    builds every grid in it, and a module import may not do that.  So the memo
    is built on the first call and the wrapper forwards ``cache_info`` and
    ``cache_clear`` -- which is also what lets a test read the bound off the
    live memo instead of restating it, and what makes a cleared memo pick up a
    bound that has since moved.

    A bound that asks the memo it is sizing is refused rather than quietly
    answered uncached: that is a design error in the bound (the sizing would
    recurse), and hiding it behind a fallback would hide the one thing this
    wrapper cannot fix.  :func:`recipe_cache_bound` is deliberately stated
    against ``payload_bits`` for that reason -- see its docstring.

    The refusal is *reentrancy*, which is a property of one thread, so it is
    keyed on the thread that is sizing and the build itself is behind a lock.
    A second thread arriving mid-build waits and then finds the memo, rather
    than reading a "reentrant" refusal it did not cause -- these functions are
    called from the render path, which encodes on worker threads.
    """

    def decorate(target):
        memo: "list[object]" = []
        sizing: "list[int]" = []
        lock = threading.Lock()

        def _memo():
            if memo:
                return memo[0]
            if sizing and sizing[0] == threading.get_ident():
                raise TesseraFormatError(
                    f"sizing the {target.__name__} memo asked {target.__name__} "
                    "for an answer: a bound that reenters the memo it sizes "
                    "cannot be computed. State the bound against a property "
                    "that does not go through this function."
                )
            with lock:
                if memo:
                    return memo[0]
                sizing.append(threading.get_ident())
                try:
                    size = int(bound())
                    if size < 1:
                        raise TesseraFormatError(
                            f"the bound for the {target.__name__} memo counted "
                            f"{size} keys; a memo sized zero is not a memo, and "
                            "an empty key space means the space, not the bound, "
                            "is wrong"
                        )
                    memo.append(lru_cache(maxsize=size)(target))
                finally:
                    sizing.clear()
                return memo[0]

        @wraps(target)
        def call(*args):
            return _memo()(*args)

        call.cache_info = lambda: _memo().cache_info()
        call.cache_clear = lambda: memo.clear()
        return call

    return decorate


def grid_space_rung_keys() -> int:
    """Distinct rung keys the grid space can address.  Computed, never listed.

    Every memo keyed on a rung -- a full format name, or a
    ``(family, rung)`` tuple -- ranges over exactly this set, so this is the
    floor under its bound: a memo smaller than this evicts entries a single
    pass over the space is still filling (tessera#46).  **It is not
    ``tessera_menu.menu_rungs_per_shape()``**, which counts the four families
    :func:`~prismaquant.tessera_menu.menu_families` admits; these memos are
    keyed on a name, and a name is admitted by
    :func:`parse_tessera_format_name`, which takes any family
    :func:`enumerate_grid_space` can build -- twelve today, the eight
    ``TESSERA_LM*`` families included.  Sizing them off the menu would leave
    them 2x undersized the day something prices an ``LM*`` family, which is the
    same drift one level up.

    The interval is counted at the **grammar's** cap, ``payload_bits`` per
    code, rather than at :func:`family_rate_cap`'s recipe-dependent one, for
    two reasons and neither is slack:

    * ``family_rate_cap`` reads the family's recipe, i.e. it calls
      ``tessera_wire_recipe`` -> ``_recipe_for``, so a bound stated against it
      would reenter the memo it is sizing;
    * ``tessera_wire_recipe(family, rung)`` never validates ``rung``, so every
      rung up to the grammar's cap is a reachable key there whatever the
      recipe currently says.  A recipe flip that widens a cap (E4M3's
      scheduled TCQ -> WINDOW move is one) moves the reachable set and does not
      move this number.

    It over-counts the *name* space by the rungs above each recipe's cap --
    14,988 against 13,068 today -- and that costs nothing: ``lru_cache`` does
    not preallocate, so an entry exists only once a key is asked.

    A family whose arity does not divide the q256 grid has no rung interval at
    all (:func:`family_q256_bounds` refuses it) and contributes no keys, which
    is why it is skipped rather than rounded.
    """
    total = 0
    for spec in enumerate_grid_space():
        lo, rem = divmod(Q256_UNIT, spec.arity)
        if rem:
            continue        # no integer rung interval; not a key space at all
        hi = spec.payload_bits * Q256_UNIT // spec.arity
        total += hi - lo + 1
    return total


def recipe_cache_bound() -> int:
    """Entries the wire-recipe memo may hold: every rung, plus every family.

    :func:`_recipe_for` is keyed ``(base, base_size, arity, rung)``.  The
    ``(base, base_size, arity)`` triple is one family -- ``base_size`` is
    redundant with ``base``, which is why the count is per family and not per
    triple -- so the key space is :func:`grid_space_rung_keys` plus the one
    ``rung=None`` entry per family that ``tessera_wire_recipe`` asks for when a
    caller wants the family's rung-independent wire (``TesseraFamily.recipe``,
    ``family_rate_cap``, every bounds question).  15,000 today, against the
    ``maxsize=512`` that held it until 2026-09-03: 512 was under the space by
    25x, and a full menu pass measured **34,580 hits and 6,920 misses on its
    second pass** -- every hit intra-rung, five lookups of one key while that
    rung is priced, and not one hit across rungs.
    """
    return grid_space_rung_keys() + sum(1 for _ in enumerate_grid_space())


def family_cache_bound() -> int:
    """Entries a family-keyed memo may hold: every (base, arity) asked.

    ``enumerate_grid_space`` asks its bases at ``_GRID_SPACE_ARITIES``;
    ``menu_families`` asks every hardware base at arities 1 through the menu's
    own search ceiling.  The union is the key space of ``_build_grid``,
    ``_tcq_body_is_reachable``, ``tessera_family``, ``_grid_for`` and
    ``family_grid_is_serialisable`` -- 32 pairs today, of which a dozen build.
    A refused family raises inside the memo and stores nothing, so the bound
    only has to hold what *can* be stored; over-counting costs nothing because
    ``lru_cache`` does not preallocate.

    The menu's ceiling is read off ``tessera_menu`` at sizing time rather than
    restated, and the bases off ``_HARDWARE_BASES``: admitting a base or
    widening the search moves this number, and the memo rebuilt after a clear
    picks it up.  The import is deferred because ``tessera_menu`` reads this
    module at its own import; the bound runs on first memo use, never then.
    """
    from .tessera_menu import _MAX_ARITY_SEARCH

    menu_pairs = {
        (base, arity)
        for base in _HARDWARE_BASES
        for arity in range(1, _MAX_ARITY_SEARCH + 1)
    }
    grid_pairs = {
        (base, arity)
        for base in (*_HARDWARE_BASES, *_MEASURED_FREE_BASES)
        for arity in _GRID_SPACE_ARITIES
    }
    return len(menu_pairs | grid_pairs)


@lazily_sized_cache(recipe_cache_bound)
def _recipe_for(base: str, base_size: int, arity: int, rung: "int | None") -> "WireRecipe":
    # Resolved off the module, not off the name bound at import: the recipe a
    # grid gets is Tessera's decision and it moves (E4M3 is scheduled to flip
    # from TCQ to WINDOW over CHANNEL), so a test -- or a future in-process
    # override -- must be able to substitute ``tessera.export.wire_recipe``
    # and be seen here.  Pair any such substitution with
    # :func:`clear_recipe_cache`.
    return _tessera_export.wire_recipe(_build_grid(base, base_size, arity), rung)


def clear_recipe_cache() -> None:
    """Forget every memoised wire recipe.

    The seam caches ``wire_recipe`` per (grid, rung) because the allocator asks
    for it once per candidate.  A caller that changes what Tessera answers --
    only tests do this today -- must clear the cache, or the seam keeps pricing
    and rendering the world it saw first.

    ``_tcq_body_is_reachable`` is cleared with it: it memoises a decision
    derived from ``recipe_table`` over the same grids, so a substituted wire
    that leaves it standing would move the prices and not the anchor wall.
    """

    _recipe_for.cache_clear()
    _tcq_body_is_reachable.cache_clear()


def tessera_wire_recipe(
    family: "str | TesseraFamily", rung: "int | None" = None
) -> "WireRecipe":
    """The wire ``tessera``'s exporter writes for ``family`` at ``rung``.

    One lookup, and it is the package's own: ``tessera.export.wire_recipe`` is
    the single statement of which body, span, scale plane and window the
    exporter puts on the wire for a grid at a rung, and every consumer here --
    the render leg, both accountants, the synthesized ``FormatSpec`` -- reads
    it rather than carrying a copy.  A PrismaQuant-side default would be the
    second spelling of one decision, and the 2026-09-01 span/plane flip is
    exactly the change a stale copy prices wrongly.

    ``rung=None`` asks for the family's *rung-independent* recipe, which is
    what the family's bounds are stated against.  A recipe may legally vary
    with the rung: the exporter records it per rung range (``wire.recipes`` in
    the config, contiguous q256 intervals, with the flat
    ``body``/``scale.plane``/``trellis.span`` keys reading "per-rung" when they
    differ across the table), and that table is built from this same
    per-(grid, rung) lookup, so the seam and the config cannot describe
    different wires.  What that costs here is only the family-level
    *interval*: every function that prices, renders or validates a rung takes
    the rung and gets the rung's own recipe, while a bounds question asked
    without one gets the family's rung-independent answer.
    """
    spec = get_tessera_family(family)
    return _recipe_for(spec.base, spec.base_size, spec.arity, rung)


# ``recipe_is_shape_free(recipe) -> bool`` used to live here, and it answered
# "does this recipe's overhead have a per-position rate at all?".  It is
# deleted rather than repaired, because since issue #126 the answer is **no for
# every recipe Tessera writes** and a predicate that is constantly False is an
# invitation to re-add the True branch:
#
# * a WINDOW body charges its ``2^L``-byte table per unit;
# * a CHANNEL scale plane charges 16 bits per output *row*;
# * a TCQ body charges its **forest** -- the ALPHABET and DESCENDANT planes,
#   ``2^(R+1)`` and ``2^(cap+1)`` bytes per distinct rate in the schedule
#   (``tessera.grammar.forest_plane_bytes``).  That last one was the hole: the
#   accountant priced the position planes only, so every E2M1x2 unit at the
#   coset cap came out 512 B light and every arity-1 E2M1 unit 20-44 B light,
#   which is 0.13 bpp on a 96x320 expert and invisible on a 1024x3072 dense
#   Linear.  It is the third term of its class and it is the one that leaves
#   nothing behind.
#
# So the statement lives in exactly one place now -- the raise in
# ``wire_overhead_q256`` -- and every Tessera ``FormatSpec`` carries a
# ``bits_for_shape_fn`` and no ``exact_bits_per_param``.  ``FormatSpec``
# already holds exactly one of the two, so no consumer can pick the cheaper.


def recipe_from_wire_names(
    span: int, scale_plane: str, body: "str | BodyKind" = "tcq", window_bits: int = 0,
) -> "WireRecipe":
    """Build a ``WireRecipe`` from the string vocabulary this module quotes.

    The accountants have always taken ``span`` and a plane *name*; this is the
    adapter that keeps those call sites working now that the recipe is the
    unit of account, and the one place the two vocabularies meet.
    """
    kind = _PLANE_KINDS[scale_plane_name(scale_plane)]
    if isinstance(body, str):
        try:
            body = {"tcq": BodyKind.TCQ, "window": BodyKind.WINDOW}[body.lower()]
        except KeyError as exc:
            raise TesseraFormatError(f"unknown body {body!r}; tcq or window") from exc
    return WireRecipe(
        body=BodyKind(body), span=int(span), scale_plane=kind,
        window_bits=int(window_bits),
    )


def tessera_wire_defaults() -> "tuple[int, str]":
    """``(span, scale plane)`` the tessera exporter writes today.

    The **rung-independent projection** of :func:`tessera_wire_recipe`, kept
    for the callers that predate the recipe and want the two scalars.  It
    drops the body, the window and the recipe's per-grid variation, so it is
    wrong the day the recipe stops being one wire for every family -- which is
    why it refuses rather than guesses: if the hardware families resolve to
    different ``(span, plane)`` pairs, or to a body that is not TCQ, it raises
    and names :func:`tessera_wire_recipe`.  New code should ask for the recipe
    of the family and rung it is pricing.
    """
    recipes = {
        (r.span, scale_plane_name(r.scale_plane), BodyKind(r.body))
        for r in (tessera_wire_recipe(f) for f in _hardware_families())
    }
    if len(recipes) != 1:
        raise TesseraFormatError(
            "the exporter no longer writes one wire for every family "
            f"({sorted((s, p, b.name) for s, p, b in recipes)}); ask "
            "tessera_wire_recipe(family, rung) for the wire being priced"
        )
    span, plane, body = next(iter(recipes))
    if body is not BodyKind.TCQ:
        raise TesseraFormatError(
            f"the default wire's body is {body.name}, which this two-scalar "
            "projection cannot state; ask tessera_wire_recipe(family, rung)"
        )
    return int(span), plane


def _hardware_families() -> "tuple[TesseraFamily, ...]":
    """The families whose recipe a global default has to hold for.

    Free (Lloyd-Max) grids are excluded because they are unexportable by
    construction -- no identifier reproduces their values -- so the exporter
    never writes a wire for one.
    """
    return tuple(
        tessera_family(base, arity)
        for base in sorted(_HARDWARE_BASES)
        for arity in (1, 2)
        if _family_fits_the_anchor_budget(base, arity)
    )


def _family_fits_the_anchor_budget(base: str, arity: int) -> bool:
    try:
        tessera_family(base, arity)
    except TesseraFormatError:
        return False
    return True


def _resolved_recipe(
    spec: "TesseraFamily",
    recipe: "WireRecipe | None",
    span: "int | None",
    scale_plane: "str | ScalePlaneKind | None",
    rung: "int | None" = None,
) -> "WireRecipe":
    """One recipe from the three ways a caller may name one.

    Precedence: an explicit ``recipe`` wins; otherwise the family's recipe at
    ``rung``.

    Naming ``span``/``scale_plane`` asks for the **coset-trellis** wire with
    that span and that plane, not for the family's current wire with two
    fields swapped.  That spelling predates the recipe and only ever described
    a TCQ body -- a span and a block plane are the two knobs that body has --
    so it is how a caller says "price this the way the artifact I am comparing
    against was priced".  Reading it as a mutation of the family's recipe would
    have quietly answered a WINDOW question with a span-1 s6b label on it the
    day E4M3 flipped, which is a wire that has never existed.  Ask for the
    window wire by passing ``recipe=``.
    """
    if recipe is not None:
        if span is not None or scale_plane is not None:
            raise TesseraFormatError(
                "name a recipe or the span/scale_plane scalars, not both"
            )
        return recipe
    if span is None and scale_plane is None:
        return tessera_wire_recipe(spec, rung)
    base = tessera_wire_recipe(spec, rung)
    kind = (
        ScalePlaneKind(base.scale_plane) if scale_plane is None
        else _PLANE_KINDS[scale_plane_name(scale_plane)]
    )
    if kind is ScalePlaneKind.CHANNEL:
        raise TesseraFormatError(
            "the CHANNEL plane is not one of the coset trellis's planes; "
            "name it with recipe=WireRecipe(...) so the body is stated too"
        )
    return WireRecipe(
        body=BodyKind.TCQ,
        span=(base.span if base.body is BodyKind.TCQ else 1)
        if span is None else int(span),
        scale_plane=kind,
        window_bits=0,
        window_seed=0,
        window_sigma=None,
        channel_sigma=None,
    )


def wire_overhead_q256(
    spec: "TesseraFamily",
    span: "int | None" = None,
    scale_plane: "str | None" = None,
    *,
    recipe: "WireRecipe | None" = None,
    shape: "Sequence[int] | None" = None,
    rung: "int | None" = None,
) -> Fraction:
    """Per-position q256 the wire adds on top of the body rate.

    Five terms, one per thing the wire charges that is not a body bit:

    * **span labels.**  A span-L trellis stores one select bit per L *codes*
      instead of one per code and ``L - 1`` two-bit labels: ``(L - 1) / L``
      extra bits per code, i.e. per ``arity`` positions.  A WINDOW body has no
      super-symbols and pays none of it.
    * **a block scale plane**: a flat ``SCALE_PLANE_BITS_Q256`` (S6b) or
      ``SCALE_LUT_BITS_Q256`` (LUT).
    * **a CHANNEL scale plane**: 16 bits per output *row* on DIAG_SV, which is
      ``16 / columns`` per position -- a per-unit cost, so it needs ``shape``.
    * **a WINDOW body's table**: ``code_bytes * 2^window_bits`` bytes inline
      on the ALPHABET plane, ``8 * code_bytes * 2^L / (rows * columns)`` per
      position -- likewise per-unit, likewise needs ``shape``.  The width is
      the *grid's* (``PayloadGrid.code_bytes``), not a constant: a code is as
      wide as the code space, and BF16's code IS a bf16 word, so its table is
      two bytes an entry.  ``tessera.calculator.terminal_rate`` takes the same
      figure from the same place, which is what keeps this accountant and the
      wire agreeing byte for byte on the 16-bit route.
    * **a TCQ body's forest**: the ALPHABET plane holds ``2^(R+1)`` anchor
      codes and the DESCENDANT plane ``2^(cap+1)`` bytes, once per *distinct*
      rate the schedule uses, both written inline in the unit
      (``tessera.unit_artifact._forest_planes``).  Sized by
      :func:`tessera.grammar.forest_plane_bytes`, **called and not restated**:
      a second implementation of one accountant is the defect this term is
      here to repair, and Tessera's is the one the exporter writes with.  It
      needs the ``rung`` as well as the ``shape``, because the forest is a sum
      over the schedule's distinct rates and an arity-1 schedule that spans
      two of them carries two forests -- which is why R511 outweighs the
      uniform R512 above it.

    ``shape`` is ``(rows, columns)`` in **weight** space, and a recipe with a
    per-unit term and no shape **raises** rather than dropping the term.  That
    is now *every* recipe: a WINDOW body has its table, a CHANNEL plane its
    row field, and a TCQ body its forest, so no Tessera rung has a
    bits-per-parameter rate and every synthesized ``FormatSpec`` prices
    through ``bits_for_shape_fn``.  A floor is not an acceptable substitute:
    the only consumer of a shape-free value was
    ``FormatSpec.exact_bits_per_param``, which ``memory_bytes_for_shape``
    multiplies by the parameter count as an exact rate, and a ``Fraction``
    cannot carry "this is a floor".  Returning the floor there is the drop,
    not the guard against it.

    Until 2026-09-03 this had four terms and no forest, and the docstring said
    "nothing raises today -- every family's recipe is TCQ over a block plane".
    That sentence was the bug: TCQ over a block plane is exactly the wire that
    was being priced 512 bytes light (RobTand/prismaquant#126), and the
    accountant's own shape-free branch was what hid it.

    At ``span=1, s6b`` the block-plane term is the half-bit every pre-minor-1
    figure carried.
    """
    wire = _resolved_recipe(spec, recipe, span, scale_plane, rung)
    if wire.span < 1:
        raise TesseraFormatError(f"span must be positive, got {wire.span}")
    plane = scale_plane_name(wire.scale_plane)
    body = BodyKind(wire.body)

    total = Fraction(0)
    if body is not BodyKind.WINDOW:
        total += Fraction((wire.span - 1) * Q256_UNIT, wire.span * spec.arity)
    if plane == "s6b":
        total += SCALE_PLANE_BITS_Q256
    elif plane == "lut16":
        total += SCALE_LUT_BITS_Q256

    if shape is None:
        raise TesseraFormatError(
            f"{spec.name}: this wire charges per unit, not per position "
            f"(body={body.name}, plane={plane}), so its rate is only defined "
            "at a shape; pass shape=(rows, columns)."
        )
    dims = tuple(int(v) for v in shape)
    if len(dims) != 2 or any(v <= 0 for v in dims):
        raise TesseraFormatError(
            f"shape must be two positive integers, got {tuple(shape)}"
        )
    rows, columns = dims
    if plane == "channel":
        # DIAG_SV: one fp16 per output row (``layout._counts_for``), over the
        # unit's positions.  Weight rows, not code rows -- the scale is per
        # output channel and a tuple code covers ``arity`` of them.
        total += Fraction(16 * Q256_UNIT, columns)
    if body is BodyKind.WINDOW:
        # Existence first: a WINDOW rung is realisable over these columns or
        # it is not a rung here, and Tessera's accountant asks the same
        # question before it prices (``terminal_rate`` -> ``bresenham_rate_
        # schedule``).  The rates themselves price nothing on this body.
        if rung is not None:
            _schedule_rates(spec, int(rung), columns, wire)
        # The ALPHABET plane *is* the table: ``2^L`` grid codes of
        # ``code_bytes`` each, inline in the unit
        # (``tessera.calculator.terminal_rate``, ``code_bytes=``).
        total += Fraction(
            spec.code_bytes * (1 << wire.window_bits) * 8 * Q256_UNIT,
            rows * columns,
        )
    else:
        # The forest, sized by tessera's own ``forest_plane_bytes``.  A byte
        # per anchor whatever the grid's ``code_bytes`` is -- the plane holds
        # anchor *codes*, indices into the grid, not the values
        # (``tessera.alphabet.AnchorForest.alphabet_plane``), which is why
        # ``code_bytes`` scales the window table above and not this.
        if rung is None:
            raise TesseraFormatError(
                f"{spec.name}: a TCQ body's forest is sized by the schedule's "
                "distinct rates, so its rate is only defined at a rung; pass "
                "rung=<body_rate_q256>."
            )
        total += Fraction(
            _forest_bytes(spec, int(rung), columns, wire) * 8 * Q256_UNIT,
            rows * columns,
        )
    return total


@lru_cache(maxsize=8192)
def _schedule_rates(
    spec: "TesseraFamily", rung: int, columns: int, wire,
) -> tuple[int, ...]:
    """The per-column rates realising ``rung`` over ``columns`` -- or a refusal.

    One walk, for every body.  ``bresenham_rate_schedule`` decides which
    ``(rung, columns)`` pairs EXIST -- 257/256 over 320 columns needs 5/4
    columns at rate 2 and is nobody's rung -- and ``tessera.calculator.
    terminal_rate`` makes the same call before it prices anything, WINDOW or
    TCQ.  Until 2026-09-03 only the TCQ branch walked it (to size the forest),
    so a WINDOW rung the encoder refuses at this width was priced, offered to
    the DP, and could be selected: the allocator building what it cannot
    build.  The refusal arrives as ``tessera.errors.GrammarError``, which is
    not a ``ValueError``; it is re-raised in this module's own type exactly as
    ``TesseraFamily.__post_init__`` does with ``tuple_grid``'s refusal, so
    every caller that guards with ``except TesseraFormatError`` --
    ``menu_families``, ``enumerate_grid_space``, ``expand_tessera_menu`` --
    sees one refusal type.
    """
    try:
        return tuple(spec.column_schedule(rung, columns, recipe=wire))
    except GrammarError as exc:
        raise TesseraFormatError(
            f"{spec.name}: rung {rung} is not realisable over {columns} "
            f"columns -- {exc}"
        ) from exc


@lru_cache(maxsize=8192)
def _forest_bytes(spec: "TesseraFamily", rung: int, columns: int, wire) -> int:
    """ALPHABET + DESCENDANT bytes one TCQ unit's forest weighs on the wire.

    Memoised because the allocator prices thousands of rungs and the schedule
    is a ``columns``-long Bresenham walk; the answer depends only on the four
    arguments, all of them hashable.  ``column_schedule`` refuses a rung the
    quota cannot close over these columns, which is the same refusal
    ``tessera.calculator.terminal_rate`` makes, so the two accountants agree
    about which rungs exist as well as about what they cost.

    That refusal arrives as ``tessera.errors.GrammarError``, which descends
    from ``TesseraError`` and **not** from ``ValueError``, so it is re-raised
    in this module's own type exactly as ``TesseraFamily.__post_init__`` does
    with ``tuple_grid``'s refusal.  Charging the forest is what first put a
    schedule walk underneath ``artifact_bpp``: before it, an unrealisable
    ``(rung, columns)`` pair returned a shape-free number, and every caller
    that guards with ``except TesseraFormatError`` -- ``menu_families``,
    ``enumerate_grid_space``, ``expand_tessera_menu`` -- would have seen the
    raw grammar error propagate out of the byte gate instead.
    """
    rates = _schedule_rates(spec, rung, columns, wire)
    return sum(forest_plane_bytes(rates, family_rate_cap(spec, wire)))


#: A **TCQ** step scores ``2**payload_bits`` anchors, so on that body the code
#: space is a *cost* and not just an addressing choice.  65 536 anchors per
#: step is the level the encoder already refuses -- it is why k=4 over E2M1 is
#: not offered -- so a family whose wire reaches it on a TCQ body is refused
#: here too, rather than being handed to a DP that would cheerfully select
#: something nothing can encode.  E4M3 at arity 2 lands exactly on this wall
#: and is excluded by it, which is the intended reading and not an off-by-one.
#:
#: **It is a property of the body, not of the grid**, and reading it as the
#: latter cost the whole 16-bit half of the rate axis.  A WINDOW body has no
#: forest at all -- ``tessera.export._plan_for`` returns the grid in the
#: forests' place -- and scores ``2^window_bits`` states per step, 16 384 at
#: the default L=14, independent of how wide the grid is; the 65 536-code
#: BF16 grid is touched once per unit, to snap that table.  Tessera says so
#: itself, in ``alphabet.SERIALISABLE_GRIDS``: "BF16 reaches the same code
#: count and is admitted because the window body never scores the grid -- it
#: scores ``2^window_bits`` states -- so the two are not the same question."
#: Nothing here caps the window's own width: ``export._window_bits_for``
#: widens L to the rate rather than refusing it, and inventing a wall Tessera
#: does not have would be this module deciding the DP's candidate set.
ANCHOR_BUDGET_BITS = 16

RATE_SURFACE_ALL_LEGAL = "all_legal"
RATE_SURFACE_DENSE = "dense"
RATE_SURFACE_ADAPTIVE = "adaptive"
RATE_SURFACE_MODES = frozenset({
    RATE_SURFACE_ALL_LEGAL,
    RATE_SURFACE_DENSE,
    RATE_SURFACE_ADAPTIVE,
})

#: Hardware grids materialise into a stock format at load, so an artifact over
#: them serves on a runtime that has never heard of Tessera.  Free (Lloyd-Max)
#: grids do not: `lloyd_max_grid` sets ``native=None`` precisely to say so.
_HARDWARE_BASES: Mapping[str, tuple[int, str]] = {
    # base -> (size, terminal format it materialises into)
    "E2M1": (16, "NVFP4"),
    "E4M3": (256, "FP8_E4M3"),
    # 65536 codes, and it belongs here for the same reason the other two do:
    # its values come from the bit pattern, so a reader rebuilds them from the
    # name, and its decoded tile is a plain BF16 tensor -- "a plain BF16
    # tensor (W16A16)", in ``tessera.export.wire_recipe``'s own words.  The
    # wall above admits it because its wire is the WINDOW body at every rung.
    "BF16": (65536, "BF16"),
}
_FREE_BASE = re.compile(r"^LM(\d+)$")
_FORMAT_NAME = re.compile(r"^TESSERA_([A-Z0-9]+)_K(\d+)_R(\d+)$")

#: Free bases ``enumerate_grid_space`` measures.  Named rather than listed at
#: the call site so :func:`family_cache_bound` counts the same roster the
#: enumerator builds -- a hand-listed default is how the 16-bit family stayed
#: invisible once already.
_MEASURED_FREE_BASES = ("LM8", "LM16", "LM32", "LM64")

#: Arities ``enumerate_grid_space`` builds per base by default.  Named for the
#: same reason: the signature default below and the bound both read it.
_GRID_SPACE_ARITIES = (1, 2)

#: A whole fused/packed serving GROUP at one family, holding a different rung
#: per member.  It is not a rung and deliberately does not parse as one --
#: ``parse_tessera_format_name`` returns ``None`` and ``get_format`` raises --
#: because there is no single rate it could stand for: the group's members have
#: different shapes and different sensitivities, which is the entire reason the
#: option exists.  Its bytes and its cost come from the ``Candidate`` the
#: aggregation built, and its per-member rungs from
#: ``Candidate.member_formats``; anything that tries to resolve it to a spec is
#: asking the wrong question and should fail loudly rather than be handed a
#: fabricated rate.
_GROUP_NAME = re.compile(r"^TESSERA_([A-Z0-9]+)_K(\d+)_G(\d+)$")

LANE_STOCK = "stock"
LANE_KERNEL = "kernel"


@dataclass(frozen=True, slots=True)
class TesseraFamily:
    """One (base grid, arity) pair, and the rungs it can address."""

    base: str
    base_size: int
    arity: int

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise TesseraFormatError(f"arity must be >= 1, got {self.arity}")
        if self.base_size < 2 or self.base_size & (self.base_size - 1):
            raise TesseraFormatError(
                f"base grid size must be a power of two >= 2, got {self.base_size}"
            )
        if self.payload_bits < ANCHOR_BUDGET_BITS:
            return          # nothing this narrow can reach the wall
        try:
            tcq = _tcq_body_is_reachable(self.base, self.base_size, self.arity)
        except GrammarError as exc:
            # Tessera declines to build the grid at all (``tuple_grid`` refuses
            # above 2^16 codes).  Its refusal, re-raised in this module's own
            # error type so every caller's ``except TesseraFormatError`` --
            # ``menu_families``, ``enumerate_grid_space`` -- keeps working.
            raise TesseraFormatError(
                f"{self.name}: tessera will not build this grid -- {exc}"
            ) from exc
        if tcq:
            raise TesseraFormatError(
                f"{self.name} needs {1 << self.payload_bits} anchors scored per "
                f"trellis step, at or above the {1 << ANCHOR_BUDGET_BITS} wall "
                f"the encoder already refuses, and its wire reaches the TCQ "
                f"body at some rung. "
                "This is a cost refusal, not a grammar one: the rungs are "
                "legal, nothing can afford to encode them.  A grid this wide "
                "whose every rung is the WINDOW body is NOT refused -- the "
                "window scores 2^L states, never the grid."
            )

    @property
    def name(self) -> str:
        return f"TESSERA_{self.base}_K{self.arity}"

    @property
    def family(self) -> str:
        """Alias for :attr:`name`.

        The RD allocator identifies a family by ``spec.family``; keeping the
        alias means retargeting it is an import swap rather than a rewrite,
        which is the whole point of the seam being three fields wide.
        """
        return self.name

    @property
    def payload_bits(self) -> int:
        """Width of the code space: ``arity * log2(base_size)``."""
        return (self.base_size.bit_length() - 1) * self.arity

    @property
    def code_bytes(self) -> int:
        """Bytes one code occupies on a code plane -- **Tessera's answer**.

        ``PayloadGrid.code_bytes``, read rather than derived: the ALPHABET and
        DESCENDANT planes store codes, a code is as wide as the grid, and
        BF16's code *is* a bf16 word.  The writer, the reader and this
        accountant must not disagree about it, so there is one authority and
        it is the grid's.
        """
        return int(self.payload_grid().code_bytes)

    @property
    def rate_cap(self) -> int:
        """Largest legal rate per code, **under this family's recipe**.

        The TCQ trellis spends one bit of the payload on its convolutional
        code, so ``|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R)`` closes at ``2^P``
        with ``cap = payload_bits - 1``.  The WINDOW body's shaping is the
        ``L - R`` bits of history a position shares with its predecessors, not
        a code bit, so a position there may spend the grid's whole width and
        the cap is ``payload_bits`` (``tessera.export.plan_for``,
        ``unit_artifact``).  Ask :func:`family_rate_cap` for the cap under a
        recipe this family does not carry today.
        """
        return family_rate_cap(self)

    @property
    def recipe(self) -> "WireRecipe":
        """The wire the exporter writes for this family (rung-independent)."""
        return tessera_wire_recipe(self)

    @property
    def lane(self) -> str:
        """``stock`` if it materialises into a hardware format, else ``kernel``.

        Materialisability is a property of the *base* grid, not the tuple: a
        k=2 code over E2M1 decodes to two E2M1 nibbles, so it materialises into
        an ordinary NVFP4 tensor exactly as the scalar one does.  (``tessera``'s
        ``tuple_grid`` drops ``native``, which understates this; the base is the
        honest place to ask.)
        """
        return LANE_STOCK if self.base in _HARDWARE_BASES else LANE_KERNEL

    @property
    def terminal_format(self) -> "str | None":
        """What this *base grid*'s values can be written as, plane aside.

        Whether a unit actually materialises into it takes the scale plane too
        -- an E4M3 tile over a per-16 block plane is FP8 bytes no stock kernel
        reads.  :func:`materialised_terminal_format` is the recipe-aware
        question, and it is the one an accountant or a gate should ask.
        """
        spec = _HARDWARE_BASES.get(self.base)
        return None if spec is None else spec[1]

    @property
    def minimum_capability_sm(self) -> "int | None":
        """The minimum SM that executes this base's stock route.  One read.

        The base names its terminal format in ``_HARDWARE_BASES``; the
        terminal names the registry row whose contract the rung executes;
        the row owns the floor.  Read through :func:`_hardware_min_sm`, the
        same function the admitted ``tessera_serving_route`` branches read,
        so the allocator's capability gate and the menu's route cannot drift
        (issue #168: E2M1 said 120 here and 100 there).  ``None`` for a free
        grid, which materialises into no hardware format.
        """
        if self.base not in _HARDWARE_BASES:
            return None
        return _hardware_min_sm(self.base)

    @property
    def mathematical_q256_bounds(self) -> tuple[int, int]:
        """Inclusive per-position q256 bounds on the *body*, before the scales.

        A rate is legal from 1 bit per code up to the cap, and a code covers
        ``arity`` positions.  Both ends are exact because ``Q256_UNIT`` is a
        power of two; a non-dividing arity is refused rather than rounded.

        The cap is the *recipe's* (:attr:`rate_cap`), so a family whose wire is
        the WINDOW body advertises the wider interval -- up to ``payload_bits``
        per code rather than ``payload_bits - 1``.  Under the TCQ recipe every
        family ships today this is unchanged.
        """
        return family_q256_bounds(self)

    # ``artifact_q256_bounds`` used to live here: the body interval shifted by
    # a flat ``wire_overhead_q256``.  It is deleted, not repaired, because both
    # halves of its premise are gone.  The overhead is no longer flat in the
    # *shape* -- a window table, a CHANNEL row field and a TCQ forest are all
    # charged per unit -- and since issue #126 it is not flat in the *rung*
    # either: the forest is a sum over the schedule's distinct rates, so R511
    # carries two forests and R512 one, and "the interval, shifted" cannot
    # state that.  A family advertises a body interval
    # (``mathematical_q256_bounds``, which is what the allocator and the menu
    # read) and an accountant that takes a rung and a shape; there is no third
    # thing in between.  Nothing in ``prismaquant/`` called it.

    def format_name(self, body_rate_q256: int, *,
                    recipe: "WireRecipe | None" = None) -> str:
        validate_body_rate_q256(self, body_rate_q256, recipe=recipe)
        return f"{self.name}_R{body_rate_q256}"

    def root_rate(self, body_rate_q256: int, *,
                  recipe: "WireRecipe | None" = None) -> Fraction:
        """Per-*code* root rate for a rung, which is what a schedule quotes."""
        validate_body_rate_q256(self, body_rate_q256, recipe=recipe)
        return root_from_q256(body_rate_q256 * self.arity)

    def column_schedule(
        self, body_rate_q256: int, n_columns: int = SUPERBLOCK_WEIGHTS, *,
        recipe: "WireRecipe | None" = None,
    ) -> tuple[int, ...]:
        """The canonical per-column rates realising a rung.  Raises if it cannot.

        This is the function that makes "continuous" checkable rather than
        asserted: an allocator that selects a rung is selecting something the
        encoder can be handed directly.  The cap is the recipe's, which is what
        ``tessera.export.plan_for`` hands the encoder.
        """
        root = self.root_rate(body_rate_q256, recipe=recipe)
        return bresenham_rate_schedule(
            root, n_columns, family_rate_cap(self, recipe)
        )

    def payload_grid(self):
        """The live Tessera grid object -- built by tessera, never by us."""
        return _build_grid(self.base, self.base_size, self.arity)


@lazily_sized_cache(family_cache_bound)
def _build_grid(base: str, base_size: int, arity: int):
    """One family's grid, built by tessera and never here.

    Sized by :func:`family_cache_bound`: ``base_size`` is redundant with
    ``base``, so the key space is the (base, arity) pairs the enumerators ask.
    """
    if base in _HARDWARE_BASES:
        scalar = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID, "BF16": BF16_GRID}[base]
    else:
        scalar = lloyd_max_grid(base_size)
    return scalar if arity == 1 else tuple_grid(scalar, arity)


@lazily_sized_cache(family_cache_bound)
def _tcq_body_is_reachable(base: str, base_size: int, arity: int) -> bool:
    """Does this grid's wire use the TCQ body at ANY rung?

    Asked of ``tessera.export.recipe_table``, which resolves the recipe at
    every rung of a grid and returns contiguous ranges, so this needs no
    assumption that a family's body is rung-independent -- E2M1x2's is not
    (WINDOW below the coset cap, TCQ at it), and a family that one day varies
    across the anchor wall answers correctly here without anyone noticing they
    had to think about it.  0.6-4.6 ms per grid, memoised, and only the
    over-budget families ever ask.

    Sized by :func:`family_cache_bound`: keyed per family, asked per family.
    """
    table = _tessera_export.recipe_table(_build_grid(base, base_size, arity))
    return any(BodyKind(entry.recipe.body) is BodyKind.TCQ for entry in table)


@lazily_sized_cache(family_cache_bound)
def tessera_family(base: str, arity: int = 1) -> TesseraFamily:
    """Build a family from a base-grid name and an arity.

    ``base`` is ``E2M1``, ``E4M3``, or ``LM<n>`` for a free Lloyd-Max grid of
    ``n`` levels.  Free grids are kernel-lane only, which is a fact about the
    grid and not a policy choice: they materialise into no hardware format.

    Sized by :func:`family_cache_bound`: keyed per family, asked per family.
    """
    hardware = _HARDWARE_BASES.get(base)
    if hardware is not None:
        size = hardware[0]
    else:
        match = _FREE_BASE.match(base or "")
        if match is None:
            legal = ", ".join(sorted(_HARDWARE_BASES)) + ", or LM<n>"
            raise TesseraFormatError(
                f"unknown Tessera base grid {base!r}; legal bases are {legal}"
            )
        size = int(match.group(1))
    return TesseraFamily(base=base, base_size=size, arity=arity)


def get_tessera_family(family: "str | TesseraFamily") -> TesseraFamily:
    """Accept a family, a family name, or a full format name."""
    if isinstance(family, TesseraFamily):
        return family
    if not isinstance(family, str):
        raise TesseraFormatError(f"not a Tessera family: {family!r}")
    parsed = parse_tessera_format_name(family)
    if parsed is not None:
        return parsed[0]
    head = family.removeprefix("TESSERA_")
    base, sep, arity = head.rpartition("_K")
    if not sep or not arity.isdigit():
        raise TesseraFormatError(
            f"unknown Tessera family {family!r}; expected TESSERA_<base>_K<arity>"
        )
    return tessera_family(base, int(arity))


def family_rate_cap(
    spec: "str | TesseraFamily", recipe: "WireRecipe | None" = None
) -> int:
    """Largest legal rate per code under ``recipe`` (default: the family's).

    ``payload_bits - 1`` for the TCQ trellis, ``payload_bits`` for the WINDOW
    body -- the same dispatch ``tessera.export.plan_for`` makes when it builds
    the schedule the encoder runs, so a rung this module offers is a rung the
    encoder accepts.
    """
    fam = get_tessera_family(spec)
    wire = tessera_wire_recipe(fam) if recipe is None else recipe
    if BodyKind(wire.body) is BodyKind.WINDOW:
        return fam.payload_bits
    return fam.payload_bits - 1


def family_q256_bounds(
    spec: "str | TesseraFamily", recipe: "WireRecipe | None" = None
) -> tuple[int, int]:
    """Inclusive per-position q256 bounds on the BODY under ``recipe``."""
    fam = get_tessera_family(spec)
    lo = Fraction(Q256_UNIT, fam.arity)
    hi = Fraction(family_rate_cap(fam, recipe) * Q256_UNIT, fam.arity)
    if lo.denominator != 1 or hi.denominator != 1:
        raise TesseraFormatError(
            f"{fam.name}: arity {fam.arity} does not divide the q256 grid "
            f"exactly; bounds {lo}..{hi} are not integers"
        )
    return (int(lo), int(hi))


def validate_body_rate_q256(
    family: "str | TesseraFamily", body_rate_q256: int, *,
    recipe: "WireRecipe | None" = None,
) -> int:
    spec = get_tessera_family(family)
    if type(body_rate_q256) is not int:
        raise TesseraFormatError("body_rate_q256 must be a JSON integer")
    lower, upper = family_q256_bounds(spec, recipe)
    if not lower <= body_rate_q256 <= upper:
        raise TesseraFormatError(
            f"{spec.name} body_rate_q256 must be in [{lower}, {upper}], "
            f"got {body_rate_q256}"
        )
    return body_rate_q256


def realisable_rungs(
    family: "str | TesseraFamily", step_q256: int = 1, *,
    recipe: "WireRecipe | None" = None,
) -> range:
    """Every addressable rung of a family, as per-position q256.

    ``step_q256=1`` is the true resolution -- 1/256 of a bit per position --
    and every value in it is realisable at superblock scale.  A coarser step is
    a *budget* decision (how many rungs to measure), never a correctness one.
    """
    spec = get_tessera_family(family)
    if step_q256 < 1:
        raise TesseraFormatError(f"step_q256 must be >= 1, got {step_q256}")
    lo, hi = family_q256_bounds(spec, recipe)
    return range(lo, hi + 1, step_q256)


def artifact_bpp(
    family: "str | TesseraFamily",
    body_rate_q256: int,
    completion: "int | None" = 0,
    span: "int | None" = None,
    scale_plane: "str | None" = None,
    *,
    recipe: "WireRecipe | None" = None,
    shape: "Sequence[int] | None" = None,
) -> Fraction:
    """Bits per position the artifact weighs: body, completion, wire overhead.

    ``recipe`` -- or the ``span``/``scale_plane`` scalars, which override the
    fields they name -- defaults to what the tessera exporter writes for this
    family at this rung (``tessera_wire_recipe``): since 2026-09-01 a span-2
    trellis over a LUT plane, which at ``E2M1_K2_R896`` is 3.75 body + 0.25
    plane -- the same position-domain size as the span-1 S6b wire it replaced,
    at 1.125x lower output-space error on the GLM experts (tessera
    ``experiments/tessera_wire_default_check.py``).  On an arity-1 family the
    stored labels cost 0.25 more per position than the LUT plane saves, so
    those rungs weigh 0.25 bpp more than they did.

    **4.0 is not the artifact's rate, and quoting it as one was #126.**  On
    top of the position planes a TCQ body writes its forest -- 512 bytes at
    the E2M1x2 coset cap -- per *unit*, so R896 is 4.0556 bpp on a 96x768
    expert, 4.1333 on a 96x320 one and 4.0013 on a 1024x3072 dense Linear.
    That is why ``shape`` is not optional.

    The rate is **two-dimensional**.  A column at body rate ``R`` writes ``R``
    body bits and may spend up to ``cap - R`` further bits selecting among the
    descendants its trellis subset reaches; the encoder spends
    ``min(completion, cap - R)`` of them.  ``completion=0`` is the exporter's
    default and the measured optimum -- at every matched artifact size, body
    rate buys more accuracy than completion depth does (up to 1.4x on
    ``E2M1_K1``) -- and ``None`` means full depth, where body + completion sum
    to ``cap`` and every rung of a family weighs the same.

    ``shape`` is ``(rows, columns)``, and it is required on **every** recipe,
    because every recipe charges something per unit rather than per position:
    a CHANNEL scale plane's row field, a WINDOW body's table, or a TCQ body's
    forest.  Without it they raise; see :func:`wire_overhead_q256` for why a
    documented floor would be the silent drop rather than the guard against
    it.  Under a WINDOW body ``completion`` must be 0: the window table is
    flat, not a forest.

    That last sentence was, from 2026-09-01 until later the same day, this
    function's entire contract: it returned the family's cap regardless of the
    rung, because the serialiser wrote the COMPLETION plane at full width
    whatever depth the encoder used, so the ladder really was flat on disk.
    That was a bug in three places (tessera `a96064b`, `eec18ba`) and not a
    property of the format.  Both errors are worth remembering, because they
    point opposite ways: quoting ``(q256+128)/256`` against a full-width
    completion plane underpriced R256 by 133%, and quoting the cap against an
    honest one overprices it by the same amount.  The size is what the
    accountant writes, and the accountant now follows the spend.
    """
    spec = get_tessera_family(family)
    wire = _resolved_recipe(spec, recipe, span, scale_plane, body_rate_q256)
    validate_body_rate_q256(spec, body_rate_q256, recipe=wire)
    cap_q256 = Fraction(family_rate_cap(spec, wire) * Q256_UNIT, spec.arity)
    body = Fraction(body_rate_q256)
    room = cap_q256 - body
    if completion is None:
        spent = room
    else:
        if completion < 0:
            raise TesseraFormatError(
                f"completion depth {completion} is negative")
        # ``completion`` counts bits per CODE and a code covers ``arity``
        # positions, so it enters the per-position rate divided by the arity --
        # the same conversion ``bits_per_position`` makes for the body.  It is
        # also capped per column at ``cap - R``: a family cannot spend
        # completion it has no room for, and charging for it would reintroduce
        # the overcharge above from the other side.
        spent = min(room, Fraction(completion * Q256_UNIT, spec.arity))
    if BodyKind(wire.body) is BodyKind.WINDOW and spent:
        raise TesseraFormatError(
            f"{spec.name}: a WINDOW body has no completion axis "
            "(tessera.encode.encode_unit); price it at completion=0"
        )
    overhead = wire_overhead_q256(
        spec, recipe=wire, shape=shape, rung=body_rate_q256,
    )
    return (body + spent + overhead) / Q256_UNIT


@dataclass(frozen=True, slots=True)
class TesseraServingRoute:
    """What a decoded Tessera tile executes as, on which hardware.

    The fifth axis of the grammar (grid x body x rate x scale plane x
    **route**), and it is a joint property of the *base grid* and the *scale
    plane*, not of either alone:

    * **E2M1 over a per-16 block plane** (S6b or LUT) decodes to an ordinary
      NVFP4 tensor -- E2M1 nibbles plus one E4M3 per 16 -- so it executes W4A4
      on the NVFP4 MMA (``tessera.decode.materialize_nvfp4``).
    * **E4M3 over a CHANNEL plane**, at arity 1, decodes to the stock
      ``compressed-tensors`` ``strategy: channel`` FP8 pair and executes W8A8
      on the FP8 MMA (``tessera.decode.materialize_fp8``, which refuses
      anything else).
    * **Everything else** -- E4M3 over a block plane, E2M1 over a CHANNEL
      plane, any free grid -- has no stock MMA layout.  The body is decoded
      inside the GEMV and the matmul runs at the activation dtype: weight-only,
      kernel lane, and no runtime serves it today.

    This is a statement about *layouts*, which is all a producer may assert on
    its own.  It is **not** an attestation that a pinned runtime routes these
    bytes (principle 14): that is ``tessera_render.tessera_lane_attested``,
    a lookup against the contract Tessera's OWN vLLM plugin packages
    (``tessera.serving``, ``quant_method: "tessera"``), ANDed with the reviewed
    release pin in ``prismaquant/tessera_runtime/``; a rung is
    ``producer_eligible`` only when a published cell names it and that pin
    names a release.  The route is what the artifact *would* execute as
    once a lane is attested, and pricing it is how the allocator stops
    comparing a W4A4 rung against a W8A8 one as if the A side were free.
    """

    contract: str
    terminal_format: "str | None"
    act_bits: "int | None"
    act_dtype_name: "str | None"
    act_group_size: "int | None"
    min_capability_sm: int
    #: The registry format whose activation RTN models this route's A side,
    #: or None for a weight-only route.
    activation_source_format: "str | None" = None

    @property
    def materialises(self) -> bool:
        """Does the decoded tile land in a stock hardware format?"""
        return self.terminal_format is not None


#: The kernel lane's floor.  Not a stock MMA claim: the Triton decode kernel
#: is a plain CUDA kernel, so sm80 is the compile floor the rest of the
#: registry uses for a weight-only format, not a routed-tensor-core capability.
_KERNEL_LANE_SM = 80

_KERNEL_ROUTE = TesseraServingRoute(
    contract="w?a16-tessera-kernel-decode",
    terminal_format=None,
    act_bits=None,
    act_dtype_name=None,
    act_group_size=None,
    min_capability_sm=_KERNEL_LANE_SM,
)


def _hardware_min_sm(base: str) -> int:
    """The minimum SM that executes ``base``'s stock route.  The one read.

    The base names its terminal format in ``_HARDWARE_BASES``; the terminal
    names the registry row whose contract the rung executes (NVFP4's A-side
    for E2M1, the FP8 pair for E4M3, the plain tensor for BF16); the row
    owns the floor, exactly as the activation quantiser is taken by
    reference from the same row.  Both live gates -- the admitted
    ``tessera_serving_route`` branches and
    ``TesseraFamily.minimum_capability_sm`` (hence the allocator's
    ``_capability_gate``) -- read through here, so the two can never
    disagree about the same contract.  Fail-closed: a base with no entry,
    or a terminal naming no registry row, raises rather than admitting
    hardware on a guessed floor.
    """
    try:
        terminal = _HARDWARE_BASES[base][1]
    except KeyError as exc:
        raise TesseraFormatError(
            f"no hardware base {base!r}; legal bases are "
            f"{', '.join(sorted(_HARDWARE_BASES))}"
        ) from exc
    from . import format_registry as fr

    try:
        return int(fr.get_format(terminal).min_capability_sm)
    except KeyError as exc:
        raise TesseraFormatError(
            f"{base}: terminal format {terminal!r} names no registry row; "
            "capability is unknown and the gate refuses"
        ) from exc


def tessera_serving_route(
    family: "str | TesseraFamily",
    recipe: "WireRecipe | None" = None,
    rung: "int | None" = None,
) -> TesseraServingRoute:
    """The route a unit of ``family`` under ``recipe`` executes on.

    ``recipe`` defaults to the wire the exporter writes for the family.
    """
    spec = get_tessera_family(family)
    wire = tessera_wire_recipe(spec, rung) if recipe is None else recipe
    plane = scale_plane_name(wire.scale_plane)
    hardware = _HARDWARE_BASES.get(spec.base)
    if hardware is None:
        return _KERNEL_ROUTE
    _size, terminal = hardware
    if spec.base == "E2M1" and plane in ("s6b", "lut16"):
        return TesseraServingRoute(
            contract="w4a4-nvfp4-e2m1-group16-ue4m3",
            terminal_format=terminal,
            act_bits=4,
            act_dtype_name="fp4_e2m1",
            act_group_size=16,
            min_capability_sm=_hardware_min_sm(spec.base),
            activation_source_format="NVFP4",
        )
    if spec.base == "BF16" and plane == "channel" and spec.arity == 1:
        # A statement about the LAYOUT the decode lands in, which is all a
        # producer may assert on its own (see this function's docstring):
        # Tessera's ``wire_recipe`` says the BF16 grid's "decoded tile is a
        # plain BF16 tensor (W16A16)", so the A side is unquantised and there
        # is no registry row whose activation RTN models it.  Whether any
        # runtime ROUTES these bytes is ``route_admission``'s question, and
        # today the answer is no -- Tessera issue #9.
        return TesseraServingRoute(
            contract="w16a16-bf16-channel",
            terminal_format=terminal,
            act_bits=16,
            act_dtype_name="bfloat16",
            act_group_size=0,
            min_capability_sm=_hardware_min_sm(spec.base),
            activation_source_format=None,
        )
    if spec.base == "E4M3" and plane == "channel" and spec.arity == 1:
        # ``materialize_fp8`` refuses a tuple grid: the FP8 MMA takes one
        # scale per output channel over scalar E4M3 bytes.
        return TesseraServingRoute(
            contract="w8a8-dynamic-e4m3-channel",
            terminal_format=terminal,
            act_bits=8,
            act_dtype_name="fp8_e4m3",
            act_group_size=0,
            min_capability_sm=_hardware_min_sm(spec.base),
            activation_source_format="FP8_E4M3",
        )
    return _KERNEL_ROUTE


def route_static_activation_contract(
    route: TesseraServingRoute,
) -> "StaticActivationContract | None":
    """The A-side served STATIC-scale contract of ``route``, or None.

    "Does this route execute a static per-unit activation scale" has exactly
    one derivation, and it is this one: the route names the registry row whose
    contract it executes (``activation_source_format``), and **the row owns the
    answer** (``FormatSpec.static_activation_contract``).  Never a compare of
    that name against ``"NVFP4"`` -- which is the same rule
    ``StaticActivationContract``'s own docstring states for specs, one level
    down (#205), and the same shape as :func:`_hardware_min_sm` above, which
    takes the capability floor by reference from the terminal row rather than
    restating it.

    A name compare answers the same question today only because ``NVFP4`` is
    the single row carrying a contract; the field is typed as a spec-level
    property, not an NVFP4 flag, so the day a second row gets one the compare
    silently splits the pipeline in two (#221).

    Returned as the CONTRACT, not a bool, because the consumers need what is
    in it: ``tessera_render.synthesize_tessera_spec`` re-stamps it
    ``measured_as_served=True`` onto the rung's spec, and the campaign prices
    and refuses through ``execution`` and ``quantize_dequantize``.  A bool
    would have sent each of them back to the row for the rest.
    """
    source = route.activation_source_format
    if source is None:
        # Weight-only (or unquantised-A) route: no A-side row, so no contract.
        return None
    from . import format_registry as fr

    return fr.get_format(source).static_activation_contract


def materialised_terminal_format(
    family: "str | TesseraFamily",
    recipe: "WireRecipe | None" = None,
    rung: "int | None" = None,
) -> "str | None":
    """The stock format this ``(family, recipe)`` decodes into, or None.

    Narrower than :attr:`TesseraFamily.terminal_format`, which names what the
    *base grid*'s values can be written as and is blind to the plane: an E4M3
    tile over a per-16 block plane is still E4M3 bytes, but no stock kernel
    reads FP8 weights with a per-16 E4M3 block scale, so it does not
    materialise.  The plane is half the answer.
    """
    return tessera_serving_route(family, recipe, rung).terminal_format


def enumerate_grid_space(
    bases: "Sequence[str] | None" = None,
    arities: Sequence[int] = _GRID_SPACE_ARITIES,
) -> Iterator[TesseraFamily]:
    """Every family the cost budget admits, cheapest code space first.

    Defaults to **every** hardware base plus the free grids that have been
    measured.  Both halves are read off ``_HARDWARE_BASES`` and
    ``_MEASURED_FREE_BASES`` rather than listed, because a hand-listed default
    is how the 16-bit family stayed invisible after the wall stopped refusing
    it.  Families over the anchor budget are skipped rather than raising,
    because enumerating a space is asking what is *available*.
    """
    if bases is None:
        bases = (*sorted(_HARDWARE_BASES), *_MEASURED_FREE_BASES)
    seen = []
    for base in bases:
        for arity in arities:
            try:
                seen.append(tessera_family(base, arity))
            except TesseraFormatError:
                continue
    seen.sort(key=lambda f: (f.payload_bits, f.base, f.arity))
    yield from seen



def is_tessera_group_option(fmt: object) -> bool:
    """True for a whole-group option name (one family, a rung per member)."""
    return isinstance(fmt, str) and _GROUP_NAME.match(fmt) is not None


def tessera_group_option_name(family: str, index: int) -> str:
    """The name for the ``index``-th option of ``family`` on one group."""
    return f"{family}_G{int(index)}"


def parse_tessera_format_name(name: object) -> "tuple[TesseraFamily, int] | None":
    """Split ``TESSERA_E2M1_K2_R896`` into family and rung, or return None.

    Returns None rather than raising for anything that is not Tessera-shaped,
    because every caller is asking "is this one of mine?" about a menu that
    also holds NVFP4, FP8 and BF16.  A name that *is* Tessera-shaped but names
    an illegal rung raises, because that is a real error and silence there
    would put an unpriced format in front of the DP.
    """
    if not isinstance(name, str):
        return None
    match = _FORMAT_NAME.match(name)
    if match is None:
        return None
    base, arity, rung = match.group(1), int(match.group(2)), int(match.group(3))
    spec = tessera_family(base, arity)
    return (spec, validate_body_rate_q256(spec, rung))


@dataclass(frozen=True, slots=True)
class TesseraRateSurface:
    """A deterministic, allocator-addressable rate surface over one family.

    Same contract the trellis surface had, and for the same reason: a rung is a
    parameter of one family, not a registry entry and not a promise that some
    separately compiled kernel exists for it.  ``adaptive`` surfaces carry the
    identity of the measured RD hull that proposed their rates, so a surface
    can always be traced back to the measurements that justified it.
    """

    family: str
    mode: str
    bounds_q256: tuple[int, ...]
    step_q256: "int | None" = None
    anchor_q256: tuple[int, ...] = ()
    proposed_q256: tuple[int, ...] = ()
    source_identity_sha256: "str | None" = None

    def __post_init__(self) -> None:
        spec = get_tessera_family(self.family)
        if self.mode not in RATE_SURFACE_MODES:
            legal = ", ".join(sorted(RATE_SURFACE_MODES))
            raise TesseraFormatError(
                f"rate-surface mode {self.mode!r} must be one of {legal}"
            )
        for field in ("bounds_q256", "anchor_q256", "proposed_q256"):
            raw = getattr(self, field)
            if isinstance(raw, (str, bytes, bytearray, Mapping)) or not isinstance(
                raw, Sequence
            ):
                raise TesseraFormatError(
                    f"rate-surface {field} must be an integer sequence"
                )
            object.__setattr__(self, field, tuple(raw))
        if len(self.bounds_q256) != 2:
            raise TesseraFormatError("rate-surface bounds_q256 must be a (lo, hi) pair")
        lo, hi = self.bounds_q256
        for rung in (lo, hi, *self.anchor_q256, *self.proposed_q256):
            validate_body_rate_q256(spec, rung)
        if lo > hi:
            raise TesseraFormatError(
                f"rate-surface bounds_q256 must be ordered, got ({lo}, {hi})"
            )
        if self.mode == RATE_SURFACE_ADAPTIVE and not self.source_identity_sha256:
            raise TesseraFormatError(
                "an adaptive rate surface must name the measured RD hull that "
                "proposed its rates; an unattributed proposal is a guess"
            )

    def rungs(self) -> range:
        lo, hi = self.bounds_q256
        return range(lo, hi + 1, self.step_q256 or 1)

    def identity(self) -> str:
        """Canonical JSON of the surface, for provenance stamping."""
        return json.dumps(
            {
                "family": self.family,
                "mode": self.mode,
                "bounds_q256": list(self.bounds_q256),
                "step_q256": self.step_q256,
                "anchor_q256": list(self.anchor_q256),
                "proposed_q256": list(self.proposed_q256),
                "source_identity_sha256": self.source_identity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


#: The ``fused_module.fields`` names a Tessera **rung** decides, and that this
#: module can therefore evaluate from a format name alone.
#:
#: The Tessera runtime contract publishes, per field, whether one vLLM-fused
#: module's roles must agree on it (``shared``) or may differ (``per_member``).
#: These four are the ones a group knapsack can violate by handing two members
#: different rungs:
#:
#: * ``family`` and ``grid`` -- a Tessera family IS a (base grid, arity) pair
#:   here, so the family component pins the grid too and both read off the
#:   name;
#: * ``body`` and ``plane`` -- read from ``tessera.export.wire_recipe`` AT THAT
#:   RUNG, because the recipe is a function of ``(grid, q256)`` and not of the
#:   grid alone.  On ``E2M1x2`` it flips at the TCQ cap: rungs below 896 write
#:   a WINDOW body and 896 writes TCQ.  Two members of one family at 850 and
#:   896 therefore put two bodies in one module, which the contract marks
#:   ``shared``.
FUSED_MODULE_RUNG_FIELDS = ("family", "grid", "body", "plane")

#: The ``fused_module.fields`` names **no rung can move**, listed to be
#: RECOGNISED rather than evaluated.
#:
#: The rule, not the roster: a field belongs here when the value is fixed
#: before the allocator chooses anything, so no choice it makes can put two
#: values of it inside one module.  ``rows`` and ``columns`` come from the
#: tensor and ``structure`` from the model graph.  They are named because a
#: ``shared`` field this module can neither evaluate nor recognise is refused
#: (``allocator_candidates._fused_group_licence``) -- silently folding as
#: though it were absent is the assertion reading the contract exists to
#: prevent -- and because *recognised* is not *unchecked*: whether the members
#: of one fused group really do read the same ``columns`` is decided by the
#: profile that grouped them, and ``aggregate_fused_siblings`` checks it there.
#: ``structure`` and ``rows`` get no such check on this side: ``rows`` the
#: contract marks ``per_member`` anyway, and one ``structure`` per model is the
#: only thing a per-model graph can produce.
FUSED_MODULE_SHAPE_FIELDS = ("structure", "columns", "rows")

#: The rate's field name in that block -- the one field the group knapsack
#: varies, and therefore the one whose licence decides whether the fold may
#: run at all.
FUSED_MODULE_RATE_FIELD = "q256"


def fused_shared_signature(
    fmt: object, shared_fields: "Collection[str]"
) -> "tuple[tuple[str, str], ...] | None":
    """What ``fmt`` commits one fused module to, over the contract's shared set.

    Two rungs may sit in the same module exactly when this returns the same
    value for both.  ``None`` means ``fmt`` is not a Tessera rung name, i.e.
    the question does not apply.

    ``shared_fields`` comes from the pinned runtime contract
    (``TesseraContract.fused_module.shared_fields()``) and is never a literal
    here: which fields a fused module's roles must agree on is a fact about
    the serving runtime, so it is read from the table that runtime publishes
    or it is refused (principle 14).  Passing a narrower set widens what a
    group may hold and a wider one tightens it -- both directions follow the
    contract, which is the point.

    Deliberately lenient about the rung's legality: this answers "can these
    two live in one module", not "is this rung realisable", and the second
    question has its own gate (:func:`validate_body_rate_q256`, the menu's
    realisability check).  Raising here would make a coherence question fail
    on an unrelated one.
    """
    if not isinstance(fmt, str):
        return None
    match = _FORMAT_NAME.match(fmt)
    if match is None:
        return None
    base, arity, rung = match.group(1), int(match.group(2)), int(match.group(3))
    wanted_set = set(shared_fields)
    wanted = [
        field for field in (*FUSED_MODULE_RUNG_FIELDS, FUSED_MODULE_RATE_FIELD)
        if field in wanted_set
    ]
    if not wanted:
        return ()
    spec = tessera_family(base, arity)
    out: list[tuple[str, str]] = []
    recipe = None
    for field in wanted:
        if field == "family":
            out.append((field, spec.name))
        elif field == "grid":
            # The grid's own spelling, from the authority that builds it, so
            # this cannot drift from Tessera's name for the same object.
            out.append((field, str(
                _build_grid(spec.base, spec.base_size, spec.arity).name)))
        elif field == FUSED_MODULE_RATE_FIELD:
            out.append((field, str(rung)))
        else:
            if recipe is None:
                recipe = tessera_wire_recipe(spec, rung)
            if field == "body":
                out.append((field, BodyKind(recipe.body).name))
            else:                              # "plane"
                out.append((field, scale_plane_name(recipe.scale_plane)))
    return tuple(out)


def format_promotion_class(fmt: object) -> str:
    """The identity every member of one serving unit must share.

    Serving-unit promotion (``allocator_solver._promote_group_components``)
    exists because fused siblings and packed experts are ONE tensor to the
    runtime, so their members cannot disagree about the thing the runtime
    dispatches on.  For every stock format that thing is the format itself,
    and this function returns the name unchanged -- so a menu with no Tessera
    rung in it promotes exactly as it did before this function existed.

    A Tessera rung is two facts glued into one name: the **family**
    (``TESSERA_E2M1_K2``), which is the grid and arity the decoder is compiled
    for, and the **rung** (``_R896``), which is a point on that family's
    continuous rate axis.  Only the first is a dispatch property.  Returning
    the family lets promotion require the shared decoder while leaving the
    rate free per member -- which is the whole reason a continuous axis is
    worth having, since q_proj (2048x1024) and k_proj (1024x1024) are
    different tensors with different sensitivities fused into one qkv_proj.

    Pure string work on the same anchored grammar
    (:data:`_FORMAT_NAME`) the parser uses, deliberately: promotion runs
    inside the numpy DP and must not pay for a grid build, and a second copy
    of the name grammar would be a drift bug waiting for a family to be added.

    .. warning::

       This function is a *name* projection and nothing more.  Whether a
       runtime serves a fused module whose members hold different rungs of one
       family is a fact about that runtime, and it is **not** decided here: it
       is published, per field, in the Tessera contract's ``fused_module``
       block (``q256: per_member`` since contract v6, RobTand/tessera#37) and
       read through ``tessera_menu.fused_module_licence`` -- that module's declared
       one read.  The group
       knapsack folds over that block and holds every ``shared`` field fixed
       -- including ``body`` and ``plane``, which the family alone does NOT
       fix, because ``wire_recipe`` is a function of ``(grid, q256)``; see
       :func:`fused_shared_signature`.  Returning the family here is what lets
       ``allocator_solver`` promote a unit onto one decoder; it is not a claim
       that the rate is free, and callers must not read it as one.

       Principle 9's export gate still decides whether such an assignment
       ships, and ``mixed_rung_receipt`` is ``false``: no vLLM serve has
       covered a mixed-rung module, only a decode identity.
    """
    if not isinstance(fmt, str):
        return str(fmt)
    match = _FORMAT_NAME.match(fmt) or _GROUP_NAME.match(fmt)
    if match is None:
        return fmt
    return f"TESSERA_{match.group(1)}_K{int(match.group(2))}"
