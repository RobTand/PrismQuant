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
    parse_tessera_format_name,
    scale_plane_name,
    tessera_wire_recipe,
)

__all__ = [
    "TESSERA_CONV_MEMORY",
    "TESSERA_GROUP",
    "TESSERA_HALF",
    "is_tessera_format",
    "render_tessera_weight",
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


@functools.lru_cache(maxsize=1)
def _pinned_serving_table():
    """The eligibility table of the pinned SERVING release, loaded once."""
    from .gridbook_lane_eligibility import (
        load_eligibility_table, load_published_formats,
    )
    return load_eligibility_table(), load_published_formats()


def tessera_lane_attested(name: str, *, table=None, formats=None) -> bool:
    """Does a pinned runtime execute this Tessera rung natively?  DERIVED.

    Principle 9 makes this a *measured platform fact*, and principle 14 says
    the fact is read from the runtime's own table, never asserted here: the
    rung is attested when the pinned serving release's contract publishes its
    payload family (``TESSERA_E2M1_K2`` for ``TESSERA_E2M1_K2_R896``) and
    carries a ``device_qualified`` cell whose route executes natively and
    whose ``rungs_q256`` names this rate, on any platform the contract names.
    The per-artifact question -- THIS platform, THIS unit's regimes -- stays
    with ``gridbook_lane_eligibility.resolve_unit_route`` at export; this is
    the menu-level admission, one lookup, and it fails closed: no table, a
    family the contract does not publish, a rate no cell names, a
    ``compile_only`` cell or a ``fallback`` route all answer False.

    Until 2026-09-02 this was a module constant (``False``); Gridbook contract
    v13 was the first to carry a Tessera row (``TESSERA_E2M1_K2``) and v14
    carries one per family (``TESSERA_E4M3_K1`` for the FP8 route as well),
    and a serving pin on a release that packages them is what flips the
    answer, not an edit here.
    """
    from .gridbook_lane_eligibility import resolve_payload_rung
    if table is None or formats is None:
        pinned_table, pinned_formats = _pinned_serving_table()
        table = pinned_table if table is None else table
        formats = pinned_formats if formats is None else formats
    if not table.present:
        return False
    family, _k, rate = resolve_payload_rung(name, published_formats=formats)
    if rate is None or not table.governs(family):
        return False
    return any(
        cell.is_trellis
        and cell.family == family
        and rate in cell.rungs_q256
        and cell.qualification == "device_qualified"
        and cell.route_status in _NATIVE_ROUTE_STATUSES
        for cell in table.cells
    )


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
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid

    bases = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID}
    if family.base not in bases:
        raise NotImplementedError(
            f"{family.name}: only hardware base grids render today "
            f"({sorted(bases)}). A free/Lloyd-Max base is measurable but not "
            "serialisable -- its values are fitted to the tensor and no "
            "identifier reproduces them, so it needs a VALUES plane first."
        )
    return tuple_grid(bases[family.base], family.arity)


@lru_cache(maxsize=64)
def tessera_rung_is_serialisable(name: str) -> bool:
    """Can the *wire* carry this rung's bytes at all?

    Distinct from "does a runtime serve it".  ``_grid_for`` admits any
    *hardware* base, while the reader resolves a grid by digest against
    ``SERIALISABLE_GRIDS``, which is a permanent wire commitment.  A rung that
    renders and is not committed dies in ``alphabet_plane()`` at export time --
    after the allocation and the whole production cache have been built -- so
    the menu must be able to ask up front rather than discover it in a
    traceback.

    ``E4M3`` used to be that gap and no longer is: it was writable all along
    and merely missing from the registry, which tessera `a4de134` fixed after
    checking that its values are reader-reconstructible from the byte pattern
    exactly as E2M1's are.  The two sets coincide today, and this predicate
    still has to exist: it is what keeps them from silently diverging when a
    fourth grid renders.
    """
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        return False
    try:
        grid = _grid_for(parsed[0])
    except NotImplementedError:
        return False           # a free base: no identifier reproduces its values
    return grid_digest(grid) in SERIALISABLE_GRIDS


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
        if route.act_bits is not None:
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
