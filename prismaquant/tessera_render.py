"""Render a Linear through Tessera, in the shape the production cache wants.

The render contract PrismaQuant already has is exactly the one Tessera needs:
``render_production_weight(weight, fmt, qname=...)`` returns a **dequantized
weight of the same shape and dtype**, and every stage downstream -- the AURA
cost, the allocator's per-(Linear, format) price, the real-KL validation, the
exported bytes -- is defined against that tensor.  So a Tessera rung becomes a
first-class allocator candidate the moment encode->decode is reachable from a
format name.  No export path and no serving backend are required to price it,
which is the point: the measurement that decides whether Tessera is worth a
vLLM backend can be taken before the backend exists.

**Rungs are family parameters, not registry entries.**  ``tessera_formats``
states this deliberately -- a family addresses ~9500 rungs at 1/256-bpp
resolution, and materialising those as static ``REGISTRY`` rows would turn a
continuous rate axis into a menu someone has to maintain.  So the specs are
*synthesized on demand* from the name, and ``format_registry.get_format`` falls
back here for anything Tessera-shaped.  Every existing consumer that resolves a
format by name keeps working, unchanged.

**Nothing here reimplements Tessera.**  The grid, the rate schedule, the
forests, the Viterbi and the reconstruction all come from the ``tessera``
package; this module is the adapter and holds no numeric constant of its own.
A second copy of a rate constant is a drift bug waiting for a rate to change.
"""
from __future__ import annotations

import functools

from functools import lru_cache

import torch
from tessera import export as _tessera_export

from .tessera_formats import (
    TesseraFamily,
    grid_space_cache,
    parse_tessera_format_name,
    scale_plane_name,
    tessera_wire_recipe,
)
from .tessera_serving_runtime_pin import TESSERA_SERVING_PLUGIN_NAME

__all__ = [
    "TESSERA_CONV_MEMORY",
    "TESSERA_GROUP",
    "TESSERA_HALF",
    "is_tessera_format",
    "render_tessera_weight",
    "tessera_lane_attested",
    "tessera_serving_contract_path",
    "synthesize_tessera_spec",
    "tessera_quantize_dequantize",
]

#: The convolutional code the encoder profile commits to.  Memory 6 is the
#: order every measured Tessera figure was produced at; it is not a free
#: parameter here, because changing it changes ``encoder_profile_id`` and so
#: changes which artifacts a reader will accept.  Read from
#: ``tessera.export.DEFAULT_CODE`` rather than restated: a second spelling of
#: the exporter's code is a rendering confound waiting for the code to change.
TESSERA_CONV_MEMORY = _tessera_export.DEFAULT_CODE.memory

#: Segment-2b scale geometry.  These are the values ``artifact_bpp`` already
#: prices (``SCALE_PLANE_BITS_Q256``); rendering on different ones would make
#: the surrogate and the accountant disagree about the same artifact.  Also
#: the exporter's (``tessera.export.DEFAULT_GROUP``/``DEFAULT_HALF``).
TESSERA_GROUP = _tessera_export.DEFAULT_GROUP
TESSERA_HALF = _tessera_export.DEFAULT_HALF

#: The shape at which a per-unit recipe's ``weight_bits`` label is quoted.
#: A label, not a price: the exact size of such a rung is
#: ``FormatSpec.bits_for_shape(shape)``, and this constant exists only so the
#: registry's integer field has a stated meaning instead of an invented one.
#: Legal for every family (columns a multiple of the 256-column superblock,
#: rows a multiple of arity x span) and representative of a production Linear.
_LABEL_SHAPE = (2048, 4096)


#: Route statuses under which a cell says a native route EXECUTES.  A
#: ``fallback`` cell attests a serve, not a native one, and admits nothing.
_NATIVE_ROUTE_STATUSES = frozenset({"backed", "backed_with_serve_flag"})

#: The vLLM plugin every Tessera route requires.  Aliased from the pin module
#: rather than retyped: one spelling of the runtime's identity, so a rename
#: cannot leave the gate checking a name nothing publishes.
TESSERA_SERVING_PLUGIN = TESSERA_SERVING_PLUGIN_NAME


def tessera_serving_contract_path():
    """The contract Tessera's own serving plugin PACKAGES, as a real path.

    ``importlib.resources`` rather than repo-root arithmetic, so a wheel
    install, an editable install and an in-repo checkout all resolve
    identically -- and so this reads the table of the ``tessera.serving``
    package that is actually importable, never a copy.  The JSON is read
    directly rather than through ``tessera.serving.contract`` because that
    module's validator imports the plugin's dispatch tables, which is a
    serving-side import a producer must not need on a machine with no GPU.

    Importing ``tessera`` here is allowed and importing ``gridbook`` is not:
    Tessera is already a producer-side dependency of this repository (the
    render adapter above encodes and decodes through ``tessera.export``),
    whereas the CB serving runtime is installed only in a serving container.
    """
    from importlib import resources

    return resources.files("tessera.serving").joinpath("runtime_contract.json")


@functools.lru_cache(maxsize=1)
def _pinned_serving_table():
    """Tessera's packaged eligibility table and format rows, loaded once.

    Until 2026-09-02 this read GRIDBOOK's materialized contract, because
    Gridbook was the only runtime that could serve Tessera bytes.  Tessera now
    ships its own vLLM plugin and its own ``runtime_contract.json``, and
    Gridbook's Tessera lane is withdrawn (its contract v14, which carried the
    two Tessera rows, was never released), so the authority for what executes
    Tessera bytes is Tessera's table.
    """
    from importlib.resources import as_file
    import json

    from .lane_eligibility import (
        load_eligibility_table, load_published_formats,
    )
    with as_file(tessera_serving_contract_path()) as path:
        # The runtime's own version string, so the table's provenance names a
        # release rather than an empty string.  It is a LABEL here; what
        # admits a rung is the pin, below.
        version = str(
            json.loads(path.read_text(encoding="utf-8"))
            .get("versions", {}).get("tessera", ""))
        return (load_eligibility_table(version, contract_path=path),
                load_published_formats(version, contract_path=path))


def _release_pin_satisfied() -> bool:
    """Is the tracked Tessera serving pin an exact reviewed release?

    False today, and by the pin: there is no Tessera release tag, so the
    tracked pin carries PENDING sentinels and
    ``require_exact_tessera_runtime_release`` refuses them.  The module
    attribute is looked up at call time so a test can substitute a
    released-pin fixture without reaching inside this function.
    """
    from . import tessera_serving_runtime_pin as pin_module

    try:
        pin_module.require_exact_tessera_runtime_release(
            pin_module.load_tessera_serving_runtime_pin())
    except pin_module.TesseraServingRuntimePinError:
        return False
    return True


def tessera_lane_attested(name: str, *, table=None, formats=None) -> bool:
    """Does a pinned runtime execute this Tessera rung natively?  DERIVED.

    Principle 9 makes this a *measured platform fact*, and principle 14 says
    the fact is read from the runtime's own table, never asserted here.  The
    runtime is Tessera's own vLLM plugin (``tessera.serving``, entry point
    ``tessera``, selected by a checkpoint's ``quant_method: "tessera"``, one
    operator knob ``TESSERA_SERVE_MODE``), and the table is the
    ``runtime_contract.json`` that plugin packages.  Three conjuncts, each of
    which fails closed on its own:

    1. **The table admits the rung.**  The contract publishes the name's
       payload family (``TESSERA_E2M1_K2`` for ``TESSERA_E2M1_K2_R896``) and
       carries a ``device_qualified`` cell whose route executes natively and
       whose ``rungs_q256`` names this rate, on any platform it names.  No
       table, an unpublished family, a rate no cell names, a ``compile_only``
       cell or a ``fallback`` route all answer False.
    2. **Every matching cell states its plugin requirement.**  Stock vLLM has
       no reader for these bytes, so a Tessera route is plugin-gated rather
       than merely flag-gated, and the cell says so in a field a gate can read
       (``requires_plugin``).  A cell that claims a route while naming no
       plugin is a CONTRACT DEFECT, and this function RAISES rather than
       admitting it: silently admitting would produce an artifact whose serve
       command need not install the runtime that reads it.  The requirement is
       carried into provenance by
       ``lane_eligibility.resolve_unit_route``, which aggregates it
       onto every ``UnitRoute`` / ``RegimeRoute`` (``requires_plugins``) and
       into ``EligibilityTable.provenance()`` (``required_plugins``) -- the
       payload the export gate stamps.
    3. **The pinned runtime is an exact reviewed release.**  Without this
       conjunct the answer would flip to True the moment the ``tessera``
       package became importable -- which it already is, as a producer-side
       render dependency -- and a producer-side import is not a serving
       release.  There is no Tessera release tag today, so
       ``require_exact_tessera_runtime_release`` refuses the tracked pin's
       PENDING sentinels and **every Tessera rung is producer-ineligible by
       the pin, not by an edit here**.

    The per-artifact question -- THIS platform, THIS unit's regimes -- stays
    with ``lane_eligibility.resolve_unit_route`` at export; this is
    the menu-level admission, one lookup.

    History: until 2026-09-02 this was a module constant (``False``), then a
    lookup against GRIDBOOK's serving pin, whose unreleased contract v13/v14
    carried the Tessera rows.  Gridbook's Tessera lane is withdrawn and the
    Gridbook pin no longer governs Tessera admission.
    """
    from .lane_eligibility import (
        LaneEligibilityError, resolve_payload_rung,
    )
    if table is None or formats is None:
        pinned_table, pinned_formats = _pinned_serving_table()
        table = pinned_table if table is None else table
        formats = pinned_formats if formats is None else formats
    if not table.present:
        return False
    family, _k, rate = resolve_payload_rung(name, published_formats=formats)
    if rate is None or not table.governs(family):
        return False
    matched = [
        cell for cell in table.cells
        if cell.is_trellis
        and cell.family == family
        and rate in cell.rungs_q256
        and cell.qualification == "device_qualified"
        and cell.route_status in _NATIVE_ROUTE_STATUSES
    ]
    if not matched:
        return False
    # Conjunct 2 is checked BEFORE the pin, deliberately.  A defective packaged
    # contract must be loud the day it lands, not on the day Rob cuts a tag.
    unstated = [cell.id for cell in matched
                if cell.requires_plugin != TESSERA_SERVING_PLUGIN]
    if unstated:
        raise LaneEligibilityError(
            f"{name}: cell(s) {unstated} claim a native Tessera route without "
            f"declaring requires_plugin={TESSERA_SERVING_PLUGIN!r}. Stock vLLM "
            "has no reader for Tessera bytes, so every such route is "
            "plugin-gated; a cell that omits the requirement would let an "
            "artifact be admitted whose serve command need not install the "
            "runtime that reads it. This is a contract defect -- fix the "
            "packaged table, never this gate.")
    return _release_pin_satisfied()


def is_tessera_format(name: object) -> bool:
    """True for a Tessera-shaped format name, without raising on others."""
    try:
        return parse_tessera_format_name(name) is not None
    except Exception:
        # A Tessera-shaped name naming an illegal rung raises inside the parser.
        # That is a real error for a caller that means to *use* the format, but
        # this predicate is only ever asking "is this one of mine?", and the
        # answer there is yes -- let the render path raise with the detail.
        return isinstance(name, str) and name.startswith("TESSERA_")


@lru_cache(maxsize=64)
def _grid_for(family: TesseraFamily):
    """The grid this family renders through -- **the family's own**.

    Deliberately not a second base->grid map.  This function held one until
    2026-09-02, listing ``E2M1`` and ``E4M3``, while ``tessera_formats`` held
    the map the pricing, the footprint and the menu read.  When ``BF16`` was
    admitted to that map the family became namable, priceable and enumerable
    and *silently unallocatable*: the copy here still called it a free base,
    so ``tessera_rung_is_serialisable`` answered False, so
    ``_producer_eligible`` answered False, so the menu dropped every rung of
    it in **both** modes.  Nothing raised, and the family looked finished
    from every side an accountant can see.  One map, read here.

    The refusal that remains is a fact about the grid rather than a fact
    about this module's knowledge: a free/Lloyd-Max base is measurable but
    not serialisable -- its values are fitted to the tensor and no identifier
    reproduces them, so it needs a VALUES plane first.  ``_HARDWARE_BASES``
    is the single statement of which bases are not that.
    """
    from .tessera_formats import _HARDWARE_BASES

    if family.base not in _HARDWARE_BASES:
        raise NotImplementedError(
            f"{family.name}: only hardware base grids render today "
            f"({sorted(_HARDWARE_BASES)}). A free/Lloyd-Max base is measurable "
            "but not serialisable -- its values are fitted to the tensor and "
            "no identifier reproduces them, so it needs a VALUES plane first."
        )
    return family.payload_grid()


@lru_cache(maxsize=64)
def family_grid_is_serialisable(family: TesseraFamily) -> bool:
    """Is this family's grid a permanent wire commitment?

    A property of the **grid**, so it is asked once per family and never once
    per rate.  It was asked per rung until 2026-09-02, behind an ``lru_cache``
    sized 64 against menus of thousands of names -- so the cache thrashed and
    every rung re-hashed the whole grid.  That cost nothing while the widest
    grid held 256 values and became the dominant cost of a menu the moment the
    16-bit family was admitted at 65536: one 2048x1024 unit's *attested* menu
    went from 0.19 s to 52 s, and a profile put 234 of 234.5 s in
    ``grid_digest`` -- computing 6916 times, at 34 ms each, an answer that
    cannot vary with the rate.  ``route_admission`` budgets 0.072 ms for this
    leg; asking the grid rather than the rung is what keeps that true.
    """
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest

    try:
        grid = _grid_for(family)
    except NotImplementedError:
        return False           # a free base: no identifier reproduces its values
    return grid_digest(grid) in SERIALISABLE_GRIDS


def clear_serialisable_cache() -> None:
    """Forget which grids are wire commitments.

    ``SERIALISABLE_GRIDS`` is a permanent registry, so memoising against it is
    safe in production and only a test ever moves it.  When one does, it has to
    clear **both** levels: the answer is cached per family as well as per rung
    since 2026-09-02, and a test that clears only the rung cache reads the
    family's stale verdict and passes for the wrong reason -- which is what
    ``test_a_rung_that_renders_can_still_be_unwritable`` did the moment the
    per-family memo landed under it.  One function, so that a third level, if
    it ever appears, is cleared by everyone who already calls this.
    """

    tessera_rung_is_serialisable.cache_clear()
    family_grid_is_serialisable.cache_clear()


@grid_space_cache
def tessera_rung_is_serialisable(name: str) -> bool:
    """Can the *wire* carry this rung's bytes at all?

    Distinct from "does a runtime serve it".  ``_grid_for`` admits any
    *hardware* base, while the reader resolves a grid by digest against
    ``SERIALISABLE_GRIDS``, which is a permanent wire commitment.  A rung that
    renders and is not committed dies in ``alphabet_plane()`` at export time --
    after the allocation and the whole production cache have been built -- so
    the menu must be able to ask up front rather than discover it in a
    traceback.

    **Sized by the space it is keyed over, which is the sentence above.**  The
    key is a full format name, so one entry per rung of every family the grid
    space admits -- ``tessera_formats.grid_space_cache_bound()``, 15,000 today
    -- and emphatically *not* ``menu_rungs_per_shape()``: the menu is a
    four-family subset of the twelve ``_grid_for`` will render, and a bound
    taken from it would be a factor of two under on the day anything prices an
    ``LM*`` family.  The literal it replaces was ``maxsize=4096`` against 6,916
    menu rungs, and it measured **0 hits / 13,832 misses** over two passes of
    one shape's menu: every lookup missed, twice over, because a pass evicted
    its own entries before it reached the end (prismaquant#134, the sibling of
    tessera#46).

    ``E4M3`` used to be that gap and no longer is: it was writable all along
    and merely missing from the registry, which tessera `a4de134` fixed after
    checking that its values are reader-reconstructible from the byte pattern
    exactly as E2M1's are.  The two sets coincide today, and this predicate
    still has to exist: it is what keeps them from silently diverging when a
    fourth grid renders.
    """
    parsed = parse_tessera_format_name(name)
    if parsed is None:
        return False
    return family_grid_is_serialisable(parsed[0])


def _producer_eligible(name: str) -> bool:
    """Producer-eligibility for one rung: the AND of two independent gates.

    (a) **The wire can carry it** -- the grid's digest is a permanent
    commitment in ``SERIALISABLE_GRIDS``.  Unconditional: a rung that reaches
    the DP and cannot be written dies at export with the whole production cache
    already built.

    (b) **A pinned runtime executes it** -- read from the pinned serving
    release's own contract through ``tessera_menu.route_admission``, the single
    seam.  Under ``PRISMAQUANT_TESSERA_MENU=research`` this half is answered
    ``unattested`` rather than refused, and the rung enters the menu carrying
    that status for the export gate to fail closed on (principles 1 and 9).

    Conflating the two is how a rung reaches the DP that cannot be written, so
    they stay separate here even though both currently answer the same way.
    """
    from .tessera_menu import menu_mode, route_admission

    if not tessera_rung_is_serialisable(name):
        return False
    return route_admission(name).admits(menu_mode())


def _plan(family: TesseraFamily, body_rate_q256: int, n_columns: int, recipe):
    """Rate schedule and forests for one (family, rung, width, recipe).

    This does not build a plan; it asks the exporter for **its** plan.
    ``tessera.export.plan_for`` is the function ``encode_linear`` calls, it
    is ``lru_cache``d there (the forests are an exhaustive per-rate
    optimisation, identical for every Linear of the same width at the same
    rung -- on a 288-expert MoE layer, 864 units sharing one plan), and it
    already dispatches the two things a second implementation here would get
    wrong: the body-dependent rate cap, and the Gaussian a TCQ forest is
    optimised against under a CHANNEL plane.  Under a WINDOW body it returns
    the grid itself in the forests' place, which is what ``encode_unit`` and
    ``reconstruct_unit`` both accept there.
    """
    from tessera.manifest import BodyKind, ScalePlaneKind
    from tessera.scale_channel import default_channel_sigma

    grid = _grid_for(family)
    body = BodyKind(recipe.body)
    channel_sigma = recipe.channel_sigma
    if ScalePlaneKind(recipe.scale_plane) is ScalePlaneKind.CHANNEL:
        if channel_sigma is None:
            channel_sigma = default_channel_sigma(grid)
        source_sigma = channel_sigma
    else:
        source_sigma = None
    rates, forests = _tessera_export.plan_for(
        grid, body_rate_q256, n_columns, body, source_sigma
    )
    return grid, rates, forests, channel_sigma


def render_tessera_weight(
    weight: torch.Tensor,
    name: str,
    *,
    col_weights: "torch.Tensor | None" = None,
    recipe=None,
) -> torch.Tensor:
    """Encode ``weight`` at the rung ``name`` names and return the reconstruction.

    The returned tensor is what a Tessera artifact at this rung decodes to, so
    it is simultaneously the surrogate's error source, the KL validation's
    weight, and the bytes' meaning -- the rendering identity principle 8
    requires, established by construction rather than by keeping three code
    paths in step.

    The wire it renders is the **recipe**: ``tessera.export.wire_recipe`` is the
    single statement of which body, span, scale plane and window the exporter
    writes for this grid at this rung, and every knob below is resolved from it
    exactly as ``encode_linear`` resolves its own ``None`` wire kwargs.  Passing
    ``recipe`` explicitly renders a wire the exporter does not write by default
    -- which is how a candidate recipe gets priced before it is the default,
    and how the test that pins this function against ``encode_linear`` reaches
    the WINDOW and CHANNEL wires.

    This deliberately stops short of the byte path.  ``encode_linear`` verifies
    that ``read_unit_artifact`` equals ``reconstruct_unit``, so rendering the
    encoder's reconstruction *is* rendering the bytes -- while a grid that
    renders but is not a wire commitment (``tessera_rung_is_serialisable``)
    still renders here, which is the distinction the menu has to be able to
    draw before it allocates.

    ``weight`` is ``[out_features, in_features]``.  The trellis runs down
    columns and a k-tuple covers ``arity`` consecutive **rows**, so tuples are
    consecutive output channels at one input position, and the scale plane's
    halves run along the input axis in row-major order -- the same axis
    NVFP4's group-16 scales run along.
    """
    from tessera.decode import reconstruct_unit
    from tessera.encode import encode_unit
    from tessera.manifest import BodyKind, RotationState

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a Tessera format name")
    family, rung = parsed

    # Refuse rather than ignore.  The render contract passes ``col_weights``
    # for the imatrix-weighted families; Tessera's encoder does not consume it
    # yet, and silently dropping it would price a lever that was never applied
    # -- the same failure shape as an activation dict keyed by the wrong name.
    if col_weights is not None:
        raise NotImplementedError(
            "Tessera render does not consume col_weights yet; it must not be "
            "silently ignored. Add imatrix weighting to encode_unit first."
        )

    if weight.ndim != 2:
        raise ValueError(
            f"{name}: Tessera renders a 2-D Linear, got shape {tuple(weight.shape)}. "
            "A packed 3-D expert stack is rendered per expert, which is how "
            "every other format already keys MoE units."
        )
    rows, cols = weight.shape
    wire = tessera_wire_recipe(family, rung) if recipe is None else recipe
    body = BodyKind(wire.body)
    # A window body has no super-symbols; ``encode_linear`` resolves the span
    # to 1 there rather than making every window caller escape a default that
    # does not apply, and so does this.
    span = 1 if body is BodyKind.WINDOW else int(wire.span)
    grid, rates, forests, channel_sigma = _plan(family, rung, cols, wire)
    if rows % grid.arity:
        raise ValueError(
            f"{name}: {rows} output features is not a whole number of arity-"
            f"{grid.arity} tuples. The rung is legal; this shape cannot carry it."
        )

    unit = encode_unit(
        weight,
        forests,
        rates,
        _tessera_export.DEFAULT_CODE,
        rotation=RotationState.NONE,
        with_diagonals=False,
        completion=0,
        group=TESSERA_GROUP,
        half=TESSERA_HALF,
        # Every encode setting below is the exporter's, read from the
        # exporter: the surrogate prices the bytes that ship (principle 8).
        # ``encode_unit``'s own defaults are the pre-minor-1 wire, kept so old
        # artifacts stay reproducible, so leaving any of them out renders a
        # different tensor than ``encode_linear`` writes.
        scale_refit=_tessera_export.DEFAULT_SCALE_REFIT,
        span=span,
        scale_plane=wire.scale_plane,
        trellis_weighting=_tessera_export.DEFAULT_TRELLIS_WEIGHTING,
        body=body,
        window_bits=wire.window_bits,
        window_seed=wire.window_seed,
        window_sigma=wire.window_sigma,
        channel_sigma=channel_sigma,
    )
    out = reconstruct_unit(unit, forests, _tessera_export.DEFAULT_CODE)
    return out.to(dtype=weight.dtype, device=weight.device)


def tessera_quantize_dequantize(name: str, recipe=None):
    """A one-argument RTN-shaped callable, for ``FormatSpec.quantize_dequantize``.

    Tessera *is* the quantizer -- there is no post-hoc error compensation to
    layer on top, which is precisely the "format-first over GPTQ compensation"
    preference: choosing the right ``(format, transform)`` beats correcting a
    wrong one afterwards.  So the registry callable is the whole render.

    ``recipe`` is closed over so a spec synthesized for a wire the exporter
    does not write by default still *renders* that wire.  A spec priced at one
    recipe whose callable rendered another would be principle 8's drift with
    both halves inside one ``FormatSpec``.
    """

    def _qdq(w: torch.Tensor) -> torch.Tensor:
        return render_tessera_weight(w, name, recipe=recipe)

    return _qdq


def _identity_activation(x: torch.Tensor) -> torch.Tensor:
    """A weight-only format's activation path: the kernel reads bf16."""
    return x


def synthesize_tessera_spec(name: str, *, recipe=None, shape=None):
    """Build a ``FormatSpec`` for a Tessera rung on demand, or return None.

    Returning None rather than raising for a non-Tessera name is what lets
    ``get_format`` use this as a fallback without reordering its own error
    handling: an unknown name must still produce ``get_format``'s KeyError,
    naming the registry, not a Tessera parse failure.

    ``recipe`` is the wire being described and defaults to the one the
    exporter writes for this family at this rung.

    A recipe that charges something **per unit** -- a CHANNEL row field, a
    WINDOW table -- has no bits-per-parameter rate at all: the same rung costs
    a different rate on a 2048x4096 tensor than on a 96x768 one.  Such a spec
    is synthesized with no ``exact_bits_per_param`` and with a
    ``bits_for_shape_fn`` instead (:class:`TesseraShapeRate`), so every byte
    accountant in the tree prices it at the shape it actually has and the
    shape-free number is refused rather than floored.  ``shape`` is accepted
    for the shape-free case, where it is redundant, and does not change what a
    per-unit recipe returns.
    """
    from . import format_registry as fr
    from .tessera_footprint import TesseraShapeRate
    from .tessera_formats import (
        artifact_bpp, recipe_is_shape_free, scale_plane_name,
        tessera_serving_route, tessera_wire_recipe,
    )

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        return None
    family, rung = parsed
    wire = tessera_wire_recipe(family, rung) if recipe is None else recipe
    plane = scale_plane_name(wire.scale_plane)

    shape_free = recipe_is_shape_free(wire)
    if shape_free:
        bpp = artifact_bpp(family, rung, recipe=wire, shape=shape)
        exact_rate = bpp
        shape_rate = None
    else:
        # The price is a function, not a rate.  ``bpp`` below survives only as
        # the integer ``weight_bits`` label: it is the rate at a documented
        # reference shape, NOT a bound over shapes (a window table costs a
        # whole extra bit per weight on a 64x512 unit and 0.004 on a
        # 2048x4096 one), and nothing prices bytes with it.
        bpp = artifact_bpp(family, rung, recipe=wire, shape=_LABEL_SHAPE)
        exact_rate = None
        shape_rate = TesseraShapeRate(family.name, rung, wire)
    # Descriptive fields for the scale plane the wire carries.  ``s6b`` is an
    # E8M0 byte per 32 plus a nibble per 16; ``lut16`` is a nibble per 16
    # indexing a per-unit E4M3 table; ``channel`` is one fp16 per output row
    # over an fp32 global and no block plane at all (schema minor 3).  The
    # first two materialise to one E4M3 per 16 at load; the exact rate travels
    # in ``exact_bits_per_param`` in every case.
    if plane == "s6b":
        scale_fields = dict(
            group_size=TESSERA_GROUP, scale_bits=8, scale_dtype_name="uint8_e8m0")
    elif plane == "lut16":
        scale_fields = dict(
            group_size=TESSERA_HALF, scale_bits=4,
            scale_dtype_name="uint4_lut16_e4m3")
    else:
        # ``group_size=0`` is the registry's spelling for per-output-channel
        # (``FormatSpec.scale_count_for_shape``), the same one ``FP8_E4M3``
        # uses.
        scale_fields = dict(
            group_size=0, scale_bits=16, scale_dtype_name="fp16_row_x_fp32_global")

    # --- the fifth axis: what the decoded tile executes as -----------------
    # Carried in the fields the registry already uses for exactly this --
    # ``act_bits``/``act_dtype_name``/``act_group_size``/``min_capability_sm``,
    # read through the single predicate ``FormatSpec.act_quant_changes_input``
    # (``format_registry.py:101``) -- rather than in a new field, so the
    # allocator's bit-exact cost short-circuit, the KL validator's activation
    # assignment and the caches all see a Tessera rung's A side the way they
    # see NVFP4's (``format_registry.py:738``) and FP8's (``:878``).
    #
    # The activation RTN is the *serving* format's own callable, taken by
    # reference: an A-side priced with a second implementation of NVFP4's
    # dynamic per-group quantiser would be a rendering confound on the axis
    # that measurement showed dominates at W4A4.
    route = tessera_serving_route(family, wire, rung)
    if route.activation_source_format is None:
        # Weight-only: the identity, spelled the way every A16 row in the
        # registry spells it (``NVFP4A16``, ``INT8_W8A16``).
        activation_qdq = _identity_activation
    else:
        activation_qdq = fr.get_format(
            route.activation_source_format).activation_quantize_dequantize

    # The layer_config entry, which is how an allocation survives the trip to
    # disk and back.  ``schemas.validate_layer_config_payload`` requires
    # ``data_type``, and ``layer_config.canonicalize_format`` has to be able to
    # recover THIS rung -- not the family, the rung -- so the entry carries the
    # name itself rather than fields a torch-free parser would have to
    # recompose out of a second copy of Tessera's grammar.  ``bits`` is the
    # same integer ceiling ``weight_bits`` is, and for the same reason: it is
    # the field an old reader looks at, and a floor there would under-count.
    # The activation fields are the route's, so a reader that never heard of
    # Tessera still sees the right A side.
    def _tessera_autoround_config() -> dict:
        cfg = {
            "data_type": "tessera",
            "bits": -(-bpp.numerator // bpp.denominator),
            "tessera_format": name,
            "tessera_family": family.name,
            "tessera_body_rate_q256": int(rung),
            "tessera_body": str(getattr(wire.body, "name", wire.body)),
            "tessera_scale_plane": plane,
            "sym": True,
            "group_size": int(scale_fields["group_size"]),
        }
        if fr.act_bits_quantize_input(route.act_bits):
            cfg.update(
                act_bits=int(route.act_bits),
                act_data_type=str(route.act_dtype_name),
                act_group_size=int(route.act_group_size or 0),
                act_dynamic=True,
                act_sym=True,
            )
        return cfg

    return fr.FormatSpec(
        name=name,
        autoround_config=_tessera_autoround_config,
        # ``weight_bits`` is the integer field the accountant reads; Tessera's
        # rate is fractional by construction, so the exact value travels in
        # ``exact_bits_per_param`` and this is the ceiling for anything that
        # wants one number. Reporting a floor here would under-count every
        # artifact.  Under a per-unit recipe it is the label described above:
        # exact at the reference shape, coarse everywhere else, and never the
        # thing a byte accountant reads.
        weight_bits=-(-bpp.numerator // bpp.denominator),
        **scale_fields,
        weight_element_dtype=f"tessera_{family.base.lower()}_k{family.arity}",
        act_bits=route.act_bits,
        act_dtype_name=route.act_dtype_name,
        act_group_size=route.act_group_size,
        family=family.name,
        min_capability_sm=route.min_capability_sm,
        quantize_dequantize=tessera_quantize_dequantize(name, wire),
        activation_quantize_dequantize=activation_qdq,
        # Producer-eligibility is the AND of two independent gates, and
        # conflating them is how a rung reaches the DP that cannot be written:
        #   (a) the wire can carry it -- the grid's digest is a permanent
        #       commitment in SERIALISABLE_GRIDS;
        #   (b) a pinned runtime executes it -- read from the pinned serving
        #       release's own contract (``tessera_lane_attested``), so a rung
        #       is admitted by a cell the runtime published, never by an edit
        #       here (principles 9 and 14).
        # (a) is per-rung and settled here; (b) is one lookup so that a route
        # attestation CANNOT silently admit an unwritable rung.  ``route``
        # above is a layout fact, NOT an attestation: a rung whose tile would
        # materialise into NVFP4 is still not producer-eligible until the
        # pinned release's table carries a cell that names it.
        # Asked through ``tessera_menu.route_admission``, the single seam that
        # reads a serving contract on Tessera's behalf, so this spec and the
        # allocator's menu cannot disagree about the same rung -- and so the
        # research mode (``PRISMAQUANT_TESSERA_MENU``) reaches the one gate
        # ``run-pipeline.sh`` and ``require_producer_formats`` consult, rather
        # than being a second admission the eligibility check never sees. The
        # wire half (a) is unconditional in both modes; only the attestation
        # half (b) relaxes, and it relaxes into ``route_status: unattested``
        # stamped on the candidate, which is what export fails closed on.
        producer_eligible=_producer_eligible(name),
        # The shape-aware price, for a wire whose per-unit planes make the
        # rate a function of the tensor.  Exactly one of this and
        # ``exact_bits_per_param`` is set: a rung either has a rate or it has
        # an accountant, never both, so no consumer can pick the cheaper one.
        bits_for_shape_fn=shape_rate,
        # The whole rate, body and scale planes together -- which is what
        # ``artifact_bpp`` computes.  Without this the generic accountant
        # charges ``ceil(bpp)`` plus a *second* group-scale term on top of a
        # number that already includes the scales: R896 priced at 4.25 bpp
        # against an artifact the exporter's byte-exact accountant measures at
        # 4.002.  The allocator would have been ranking Tessera against NVFP4
        # on a 6% overcharge it invented.
        exact_bits_per_param=exact_rate,
    )


# ---------------------------------------------------------------------------
# The encoder seam: one function owns the call into Tessera's byte path.
# ---------------------------------------------------------------------------

class HessianContractError(RuntimeError):
    """The Hessian contract was not met, and the encode must not proceed.

    Its own class because the campaign's anchor loop wraps ``_measure_anchor``
    in ``except Exception: continue`` -- an anchor that fails for its own
    reasons should not abort a multi-hour run -- and a contract refusal is
    exactly the thing that must *not* be absorbed by that. A refusal quietly
    turned into a skipped anchor is a cost table that is weights-only in the
    rows that were hard and H-aware in the rows that were easy, with nothing
    on it saying so.
    """


#: The encoder keywords ``ActivationSource.for_unit`` produces.  Read from
#: Tessera's own object rather than typed, and probed against the pinned
#: ``encode_linear_planes`` signature, so "this build can consume an H" is a
#: derived fact and not a constant somebody kept in step.
#:
#: The encoder's shipping default is *activation-aware*: given ``H = XᵀX`` for
#: a unit it applies LDLQ (sigma 1.0, block 32) plus an exact full-Hessian
#: row-scale refit, and weights-only encodes stay byte-identical.  So a rung
#: priced without an H is **not** the rung that ships, and the difference is
#: not a rounding one -- served KL on Qwen3-0.6B went 0.1512 -> 0.1046 at
#: byte-identical wire bpp.
#:
#: The probe is deliberately *not* "try the call and catch ``TypeError``":
#: that swallows every unrelated argument error the encoder raises and would
#: silently downgrade a shipping price to a weights-only one.


#: The keywords ``for_unit`` emits that carry the Hessian itself.  A plane
#: whose emitted set contains neither has no activation-aware encode: the
#: encode would be byte-identical to the weights-only one, and forwarding the
#: rest would be an H-free encode wearing an H-aware stamp.
_H_BEARING_KWARGS = frozenset({"ldl", "refit_metric"})


@lru_cache(maxsize=1)
def _encoder_accepts_hessian() -> "tuple[bool, tuple[str, ...], tuple[str, ...]]":
    """``(accepted, required kwargs, encoder parameters)`` for the pinned build.

    ``required`` is the **union over every scale plane**, because the emitted
    set is plane-dependent -- a plane whose refit objective is ``"plain"``
    emits no ``refit_metric`` -- and this tuple is what the seam's forwarded-key
    whitelist is checked against.  Probing one plane would narrow the whitelist
    to that plane's keys and make the seam refuse another plane's legitimate
    keyword as "unknown".  Ask :func:`_encoder_kwargs_for_plane` for one plane's
    own set.

    Cached on nothing but the process: the pinned Tessera is a property of the
    interpreter's import, not of an argument.
    """
    import inspect

    params = tuple(
        inspect.signature(_tessera_export.encode_linear_planes).parameters)
    required = tuple(sorted(
        set().union(*(_encoder_kwargs_for_plane(plane)
                      for plane in _tessera_export._PLANE_NAMES))))
    return (all(k in params for k in required), required, params)


@lru_cache(maxsize=8)
def _encoder_kwargs_for_plane(scale_plane) -> "frozenset[str]":
    """The keywords ``ActivationSource.for_unit`` emits on ``scale_plane``.

    Asked of Tessera's own object on a throwaway unit rather than listed here.
    This is the derivation :func:`rung_accepts_hessian` reads, and it is why
    that predicate is no longer a restatement: what a plane admits is whatever
    the pinned source emits for it, so a Tessera release that adds, removes or
    re-plumbs a keyword moves the predicate with it (principle 14).
    """
    import torch

    from .tessera_hessian import HESSIAN_IDENTITY_FIELDS, activation_source

    identity = {f: ("" if f.endswith("sha256") else 0)
                for f in HESSIAN_IDENTITY_FIELDS}
    # An identity matrix, at the LDL block width, so the probe exercises the
    # real ``block_ldl`` path rather than a shape it refuses.
    width = int(activation_source({}, identity).ldlq_block)
    probe = activation_source({"probe": torch.eye(width)}, identity)
    return frozenset(probe.for_unit("probe.weight", width,
                                    scale_plane=scale_plane))


def tessera_encoder_hessian_status() -> dict:
    """What this build can do with a Hessian -- for provenance and refusals."""
    accepted, required, params = _encoder_accepts_hessian()
    from .tessera_hessian import encoder_recipe

    return {
        "kwargs": list(required),
        "accepted": bool(accepted),
        "recipe": encoder_recipe(),
        "reason": (
            "supported"
            if accepted
            else "pinned tessera.export.encode_linear_planes accepts "
                 f"{params}, which does not cover {required}"
        ),
    }


def rung_accepts_hessian(format_name: str, recipe=None) -> bool:
    """Does this rung's WIRE admit an activation-aware encode?

    **Derived, not restated.**  The answer is whether the pinned
    ``ActivationSource`` emits an H-bearing keyword -- ``ldl`` or
    ``refit_metric`` -- for the plane this rung's recipe resolves to.  If it
    emits neither, the H-aware encode and the weights-only encode are the same
    bytes and pricing it weights-only is pricing what ships.  If it emits
    either, they are not, and a weights-only price is a price of bytes the
    exporter will not write.

    This used to restate Tessera's condition -- "the CHANNEL scale plane, and
    only it, at ``f3e7d0a``" -- quoting two refusals verbatim:

    * ``ldl`` -- "LDLQ is implemented for the CHANNEL scale plane; a block
      plane's per-column-span scales would have to be scheduled with it"
    * ``refit_metric`` -- "read only by the CHANNEL plane's refit"

    **Tessera deleted both guards on 2026-09-02**
    (``docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md``: "That
    reason does not survive reading the loop it guards"), and the restatement
    went stale where nothing could see it.  ``refit_metric`` is now refused
    only on S6b, ``ldl`` on no plane at all, and the same release keyed the
    refit objective **by** plane -- which is the ``scale_plane`` argument
    ``for_unit`` now demands.  One relaxation, two symptoms.

    What the staleness cost: on the LUT plane the H-aware wire is measured at
    served KL 0.5310 against the weights-only wire's 0.6404 on Qwen3-0.6B at
    the E2M1x2 q896 cap, at **identical bytes** (both 220,301,312 wire bytes,
    4.0018 bpp).  PrismaQuant priced and rendered every E2M1 rung weights-only
    while its own export lane hands the encode to Tessera's exporter
    (``experiments/export_tessera_serving.py:633``), which applies
    ``for_unit(..., scale_plane=recipe.scale_plane)`` on every plane whenever a
    Hessian is supplied.  The surrogate was pricing bytes the export does not
    write -- principle 8, across the repository boundary.

    ``recipe`` is the resolved wire the encode itself uses, threaded rather
    than looked up again: ``_recipe_for`` is memoised and a
    ``clear_recipe_cache`` between two lookups (tests do this) is exactly the
    split that would let the price and the encode disagree.

    One residual plane guard is Tessera's alone and is deliberately not
    mirrored here: ``refit_reach_floor`` is CHANNEL-only, and the source
    default is ``False``, so it reaches a block plane only when a caller turns
    it on -- at which point Tessera raises by name, which is the runtime
    speaking for itself rather than this module speaking for it.

    ``test_the_hessian_applies_exactly_where_tessera_says_it_does`` pins the
    derivation against real encodes on every family, each built with its own
    resolved plane: the predicate and the raise site agree, or the test fails.
    """
    from .tessera_formats import parse_tessera_format_name

    parsed = parse_tessera_format_name(format_name)
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a Tessera format name")
    family, rung = parsed
    wire = tessera_wire_recipe(family, rung) if recipe is None else recipe
    emitted = _encoder_kwargs_for_plane(wire.scale_plane)
    return bool(emitted & _H_BEARING_KWARGS)


def encode_tessera_unit(
    weight,
    format_name: str,
    *,
    activation_kwargs: "dict | None" = None,
    hessian_required: bool = True,
    verify: bool = False,
    recipe=None,
):
    """``(render, blob)`` for one rung -- **the** call into Tessera's byte path.

    Every PrismaQuant price of a Tessera rung goes through here, so that what
    the surrogate scores, what the KL validation would run, and what an export
    would ship are one encode with one set of inputs (principle 8).  The render
    returned is ``read_unit_artifact(blob)`` -- the bytes, decoded -- not a
    second reconstruction that happens to agree.

    ``activation_kwargs`` is what ``tessera_hessian.encoder_kwargs`` returned
    for THIS unit **on this rung's scale plane**: the block-LDL of its
    regularised ``XᵀX`` and the plane's refit metric.  They carry no rate, so
    they arrive pre-computed rather than as a Hessian this function would
    re-factorise once per rate; they do carry a plane, so the caller memoises
    them per ``(unit, plane)`` and not per unit.

    ``recipe`` is the resolved wire, threaded from the caller so that the
    plane the kwargs were built on, the plane the predicate reads, and the
    plane the encode writes are one object rather than three lookups.  Omit it
    and this resolves its own; the two agree today because ``_recipe_for`` is
    memoised, and disagree the moment anything clears that cache between two
    lookups -- which is exactly what ``clear_recipe_cache`` exists for and
    what the tests do.

    **The invariant that makes "the same recipe" true across the boundary:**
    the only keywords forwarded to ``encode_linear`` are ``grid``/``q256``/
    ``name``/``verify`` plus the set ``ActivationSource.for_unit`` emits, and
    anything else is refused below by name.  No recipe axis -- ``span``,
    ``scale_plane``, ``body``, ``window_bits`` -- can reach the encoder from
    here, so Tessera resolves the recipe internally with no override and its
    answer equals ``tessera_wire_recipe(family, rung)`` by value.  That
    equality is what the refusal in ``for_unit`` is warning about, and it
    holds because the whitelist is structural rather than because a caller
    remembered.

    ``hessian_required=True`` is the default for the same reason
    ``render_production_weight`` defaults ``ldlq_missing_activation_ok=False``:
    a render that quietly drops its activation input prices a different tensor
    than the one that ships, and does so silently.  Pass ``False`` to price
    weights-only *deliberately*; the caller must then stamp that on every row.
    """
    from tessera.unit_artifact import read_unit_artifact

    from .tessera_formats import parse_tessera_format_name

    parsed = parse_tessera_format_name(format_name)
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a Tessera format name")
    family, rung = parsed
    wire = tessera_wire_recipe(family, rung) if recipe is None else recipe

    accepted, required, _params = _encoder_accepts_hessian()
    if not rung_accepts_hessian(format_name, wire):
        # The pinned source emits no H-bearing keyword for this plane, so the
        # H-aware encode and the weights-only one are the same bytes: requiring
        # a Hessian would refuse the only bytes this rung has. Kwargs built
        # anyway are refused rather than dropped -- forwarding them would be an
        # H-free encode wearing an H-aware stamp.
        #
        # Dormant at the pin, not dead: every plane PrismaQuant resolves emits
        # both ``ldl`` and ``refit_metric`` since Tessera's 2026-09-02 release
        # (see ``rung_accepts_hessian``). It stays because the branch is a
        # consequence of what the source emits, and the source is Tessera's to
        # change.
        if activation_kwargs:
            raise HessianContractError(
                f"{format_name}: the pinned ActivationSource emits no "
                f"H-bearing keyword on the "
                f"{_tessera_export._PLANE_NAMES[wire.scale_plane]!r} scale "
                "plane, so this encode is weights-only whatever is passed. Ask "
                "rung_accepts_hessian() before building them."
            )
        hessian_required = False
    if hessian_required and not accepted:
        status = tessera_encoder_hessian_status()
        raise HessianContractError(
            "Tessera rungs cannot be priced with a Hessian on this build: "
            f"{status['reason']}. The H-aware encoder is the shipping default, "
            "so pricing without one prices bytes that are not the bytes that "
            "ship. Re-pin Tessera to a build whose encoder takes an "
            "ActivationSource, or price weights-only deliberately with "
            "hessian_required=False (campaign: --hessian off), which stamps "
            "hessian.supplied=false on every row it writes."
        )
    if hessian_required and not activation_kwargs:
        raise HessianContractError(
            f"{format_name} on a unit with no Hessian: hessian_required=True "
            "but no activation kwargs were passed. A missing H is a missing "
            "input, not a default."
        )
    if activation_kwargs and not hessian_required:
        # Both directions are bad and both are silent, so neither is allowed:
        # encoding H-aware bytes under a ``supplied=false`` stamp is the same
        # class of error as dropping an H that was supplied.
        raise HessianContractError(
            f"{format_name}: activation kwargs were passed with "
            "hessian_required=False. Weights-only means no Hessian is applied, "
            "not one that is applied and unrecorded."
        )

    kwargs = dict(grid=_grid_for(family), q256=int(rung), name=format_name,
                  verify=bool(verify))
    if activation_kwargs:
        unknown = sorted(set(activation_kwargs) - set(required))
        if unknown:
            raise HessianContractError(
                f"{format_name}: {unknown} are not keywords "
                "ActivationSource.for_unit produces; this seam forwards that "
                "object's output and nothing else."
            )
        kwargs.update(activation_kwargs)
    unit = _tessera_export.encode_linear(weight, **kwargs)
    render = read_unit_artifact(unit.blob, device=str(weight.device))
    return render.to(dtype=weight.dtype, device=weight.device), unit.blob


__all__ += [
    "HessianContractError",
    "encode_tessera_unit",
    "rung_accepts_hessian",
    "tessera_encoder_hessian_status",
]


def render_tessera_production(
    weight,
    fmt: str,
    *,
    qname: str,
    activations,
    levers,
):
    """The production render for a Tessera rung: **decoded wire bytes**.

    ``render_production_weight`` reaches this before its format cascade, so a
    ``TESSERA_*`` unit in a ``layer_config`` renders through Tessera's encoder
    rather than falling to the registry's weights-only ``quantize_dequantize``
    reconstruction.  Two things were wrong with that fallback and both are
    silent: it is a reconstruction rather than the bytes, and it is
    weights-only, while the encoder's shipping default is H-aware.  Principle 8
    wants the surrogate, the KL validation and the shipped bytes to be *one*
    render; this is the function that makes that true for Tessera.

    The Hessian is ``XᵀX`` over ``activations[qname]`` -- PrismaQuant's own
    calibration rows, the same cache the AURA probe reads, never the held-out
    split the KL selection uses -- formed by ``tessera_hessian.
    hessian_from_rows``, the same function the anchor campaign calls, so the
    two paths cannot form it two ways.  A **missing** ``qname`` is a hard
    failure, not a fallback: the recurring landmine in this codebase is a
    render whose activation lookup misses by a key and silently prices RTN.

    The draw's identity rides in ``levers['tessera_hessian_identity']`` and is
    **required**: an ``ActivationSource`` refuses a provenance missing
    ``text_sha256`` / ``fit_tokens`` / ``fit_ids_sha256``, and that refusal is
    what makes "the campaign priced this rung and the cache rendered it" a
    checkable claim rather than an assumption. Whoever fills the activation
    cache stamps the triple; there is no default, because an identity of
    ``None`` compares equal to another ``None``.

    ``levers['tessera_weights_only']`` is the deliberate opt-out and must be
    stamped by whoever sets it.
    """
    from .tessera_formats import parse_tessera_format_name
    from .tessera_hessian import activation_source, encoder_kwargs, hessian_from_rows

    # ONE resolve for this render, threaded to the predicate, to the kwargs and
    # to the encode. Three independent lookups would agree today only because
    # ``_recipe_for`` memoises them; the plane priced and the plane encoded have
    # to be the same object, not the same value by luck.
    parsed = parse_tessera_format_name(fmt)
    if parsed is None:
        raise ValueError(f"{fmt!r} is not a Tessera format name")
    wire = tessera_wire_recipe(*parsed)

    weights_only = bool(levers.get("tessera_weights_only", False)) if levers \
        else False
    acts = None
    # When the lever says weights-only, no Hessian is FORMED -- not merely not
    # required. Forming one and letting ``encode_tessera_unit`` drop it is the
    # same class of error in the other direction: a lever named
    # ``tessera_weights_only`` shipping H-aware bytes under a
    # ``supplied=false`` stamp. The campaign's ``--hessian off`` collects no H
    # at all, and this must mean the same thing.
    if not rung_accepts_hessian(fmt, wire):
        # The pinned source emits no H-bearing keyword on this plane, so the
        # weights-only bytes ARE this rung's shipping bytes. Priced as such,
        # without a Hessian and without a refusal.
        render, _blob = encode_tessera_unit(
            weight, fmt, hessian_required=False, recipe=wire)
        return render
    if activations is not None and not weights_only:
        try:
            acts = activations[qname]
        except KeyError:
            acts = None
        except TypeError:
            acts = None
    if acts is None:
        if not weights_only:
            raise HessianContractError(
                f"{qname}={fmt}: no calibration activations for this qname, so "
                "no Hessian can be formed. The Tessera encoder's shipping "
                "default is H-aware; rendering without one would price bytes "
                "that are not the bytes that ship. Fix the activation key (the "
                "render cache is keyed by qname) or set the "
                "'tessera_weights_only' lever deliberately."
            )
        render, _blob = encode_tessera_unit(
            weight, fmt, hessian_required=False, recipe=wire)
        return render
    rows = acts.detach().to(device=weight.device, dtype=torch.float32)
    if int(rows.shape[-1]) != int(weight.shape[1]):
        raise HessianContractError(
            f"{qname}={fmt}: activation rows have {int(rows.shape[-1])} "
            f"columns but the weight has {int(weight.shape[1])} inputs; "
            "this is a wrong-key or wrong-shard activation, not a Hessian."
        )
    identity = dict((levers or {}).get("tessera_hessian_identity") or {})
    if not identity:
        raise HessianContractError(
            f"{qname}={fmt}: the activation cache supplied rows but no "
            "'tessera_hessian_identity' lever. Bytes shaped by a Hessian are "
            "not reproducible from the weights, so the capture that shaped "
            "them has to be named (text_sha256 / fit_tokens / fit_ids_sha256) "
            "or nothing downstream can tell this render from one built on a "
            "different draw."
        )
    source = activation_source({qname: hessian_from_rows(rows)}, identity)
    kwargs = encoder_kwargs(source, qname, int(weight.shape[1]), weight.device,
                            scale_plane=wire.scale_plane)
    render, _blob = encode_tessera_unit(
        weight, fmt, activation_kwargs=kwargs, hessian_required=True,
        recipe=wire)
    return render


__all__ += ["render_tessera_production"]
