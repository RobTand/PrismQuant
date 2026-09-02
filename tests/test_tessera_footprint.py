"""Bytes for a Tessera Linear, and the allocator pricing that rests on them.

The property under test is that PrismaQuant's byte budget, Tessera's exact-byte
accountant and the artifact an exporter would write are **one number**.  So the
tests here check the pricing against the published ladder rather than against
themselves, and check that the whole grid-space prices at the bounds the family
descriptor advertises -- a family that claims [1.00, 4.00] bpp and prices at
[1.50, 7.50] is not a family, it is two disagreeing statements.

Every "the rung is not a rate" assertion in this file was written on
2026-09-01 against a serialiser that wrote the COMPLETION plane at full width
whatever depth the encoder spent, and every one of them is inverted below.
The rung *is* a rate; the flat ladder was a bug in three places (tessera
`a96064b`, `eec18ba`).
"""
from fractions import Fraction

import pytest

from prismaquant.tessera_allocator import build_tessera_allocator_candidate
from prismaquant.tessera_footprint import (
    TESSERA_TENSOR_PAYLOAD_SCHEMA,
    tessera_tensor_payload_breakdown,
    validate_tessera_tensor_payload_breakdown,
)
from prismaquant.tessera_formats import (
    TesseraFormatError,
    artifact_bpp,
    enumerate_grid_space,
    get_tessera_family,
    realisable_rungs,
    recipe_is_shape_free,
    scale_plane_name,
    tessera_wire_recipe,
)
from tessera.manifest import BodyKind

SHAPE = (4096, 4096)

# (family, body q256, published artifact bpp) -- published on the span-1 S6b
# wire (tessera schema minor 0), which is therefore named explicitly below: a
# published figure is a property of the wire it was measured on.
MEASURED = [
    ("TESSERA_E2M1_K1", 768, 3.5),
    ("TESSERA_E2M1_K2", 896, 4.0),
    ("TESSERA_LM16_K2", 896, 4.0),
    ("TESSERA_LM16_K1", 768, 3.5),
    # Sub-cap rungs, published as 4.5 and 5.5 and worth exactly that.  They
    # were briefly "corrected" to 7.5 when the serialiser was found writing
    # COMPLETION at full width whatever depth the encoder spent; that was a
    # bug's arithmetic, fixed in tessera `a96064b`.
    ("TESSERA_E4M3_K1", 1024, 4.5),
    ("TESSERA_E4M3_K1", 1280, 5.5),
]

# The same rungs under the exporter's default wire, which is a function of the
# grid AND the rung since 2026-09-02.  On the span-2 coset trellis over LUT16
# the stored labels cost exactly what the plane saves at arity 2, so K2 R896
# stays 4.0, while at arity 1 they cost a quarter-bit more and K1 rungs weigh
# +0.25.  E4M3 left that wire entirely: the window body over the CHANNEL plane
# pays no labels and no block plane, and instead pays two *per-unit* terms --
# one fp16 per output row and a 2^14-byte table -- which at this 4096x4096
# shape come to 16/4096 + 2^14*8/4096^2 = 0.01171875 bpp.
DEFAULT_WIRE = [
    ("TESSERA_E2M1_K1", 768, 3.75),
    ("TESSERA_E2M1_K2", 896, 4.0),
    ("TESSERA_LM16_K2", 896, 4.0),
    ("TESSERA_LM16_K1", 768, 3.75),
    ("TESSERA_E4M3_K1", 1024, 4.01171875),
    ("TESSERA_E4M3_K1", 1280, 5.01171875),
]


@pytest.mark.parametrize("family,q256,bpp", MEASURED)
def test_the_measured_ladder_prices_at_its_published_bpp(family, q256, bpp):
    spec = get_tessera_family(family)
    out = tessera_tensor_payload_breakdown(
        SHAPE, family=spec, body_rate_q256=q256, span=1, scale_plane="s6b",
    )
    assert out["exact_bpw"] == pytest.approx(bpp, abs=1e-9)
    assert out["schema"] == TESSERA_TENSOR_PAYLOAD_SCHEMA
    assert out["wire_schema"] == "prismaquant.tessera.v1"
    assert out["trellis_span"] == 1 and out["scale_contract"] == "s6b"


@pytest.mark.parametrize("family,q256,bpp", DEFAULT_WIRE)
def test_the_default_wire_prices_the_bytes_the_exporter_writes(family, q256, bpp):
    """No wire named: the price is the exporter's wire, read from tessera."""
    spec = get_tessera_family(family)
    out = tessera_tensor_payload_breakdown(SHAPE, family=spec, body_rate_q256=q256)
    assert out["exact_bpw"] == pytest.approx(bpp, abs=1e-9)
    # The wire is read off the recipe, not asserted as a constant: which body
    # and plane this rung gets is tessera's decision per (grid, rung).
    wire = tessera_wire_recipe(spec, q256)
    assert out["trellis_span"] == wire.span
    assert out["scale_contract"] == scale_plane_name(wire.scale_plane)
    assert out["body_kind"] == wire.body.name.lower()
    # and the revalidation re-derives the same wire from the record itself
    assert validate_tessera_tensor_payload_breakdown(out)["exact_bpw"] == out["exact_bpw"]


def test_every_family_prices_at_the_bounds_it_advertises():
    """Catches the arity bug: a k=2 code spans k rows, so the code grid is
    `rows // k`.  Priced without that, k=2 families came out at exactly `k`
    times their real cost -- LM64 k=2 at 11.5 bpp against a 6.00 ceiling."""
    for spec in enumerate_grid_space():
        rungs = realisable_rungs(spec)
        # The interval a family advertises is stated on its *family-level*
        # recipe (the rung-independent one), so the ends are priced on that
        # same wire -- otherwise this compares a window rung against a coset
        # interval and calls the difference a bug.
        wire = spec.recipe
        per_unit = not recipe_is_shape_free(wire)
        bounds = None if per_unit else spec.artifact_q256_bounds
        for end, rung in enumerate((rungs[0], rungs[-1])):
            out = tessera_tensor_payload_breakdown(
                SHAPE, family=spec, body_rate_q256=rung, recipe=wire,
                alphabet_bytes=0 if per_unit else None,
            )
            priced = Fraction(*out["exact_bpw_rational"])
            want = (
                artifact_bpp(spec, rung, recipe=wire, shape=SHAPE) if per_unit
                else Fraction(bounds[end], 256)
            )
            assert priced == want, (spec.name, rung)


def test_bytes_are_monotone_in_the_rung():
    """A rate axis the allocator can search has to be ordered, and it is.

    This assertion was inverted for part of 2026-09-01 -- "every rung of a
    family serialises to the identical byte count" -- which was true of the
    serialiser and false of the format.  Bytes rise with the rung again, and
    strictly: two rungs pricing the same would mean the COMPLETION plane had
    gone back to absorbing what BODY gives up."""
    spec = get_tessera_family("TESSERA_E4M3_K1")
    sizes = [
        tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=q
        )["total_bytes"]
        for q in realisable_rungs(spec, step_q256=64)
    ]
    assert sizes == sorted(sizes), sizes
    assert len(set(sizes)) == len(sizes), sizes


def test_a_schedule_that_does_not_realise_its_rung_is_refused():
    """Otherwise a candidate is priced for a unit the artifact does not hold."""
    spec = get_tessera_family("TESSERA_E4M3_K1")
    honest = spec.column_schedule(1024, 4096)
    tampered = (honest[0] + 1,) + honest[1:]
    with pytest.raises(TesseraFormatError, match="not rung"):
        tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=1024, schedule=tampered
        )


def test_an_edited_footprint_fails_its_own_content_address():
    spec = get_tessera_family("TESSERA_E2M1_K1")
    out = tessera_tensor_payload_breakdown(SHAPE, family=spec, body_rate_q256=768)
    assert validate_tessera_tensor_payload_breakdown(out) == out
    edited = dict(out)
    edited["total_bytes"] = out["total_bytes"] - 1024
    with pytest.raises(TesseraFormatError, match="does not address its own"):
        validate_tessera_tensor_payload_breakdown(edited)


def test_rows_must_divide_by_the_arity():
    spec = get_tessera_family("TESSERA_E2M1_K2")
    with pytest.raises(TesseraFormatError, match="not a multiple of arity"):
        tessera_tensor_payload_breakdown(
            (4095, 4096), family=spec, body_rate_q256=896
        )


def test_columns_must_be_whole_superblocks():
    spec = get_tessera_family("TESSERA_E2M1_K1")
    with pytest.raises(TesseraFormatError, match="superblock"):
        tessera_tensor_payload_breakdown(
            (4096, 4000), family=spec, body_rate_q256=768
        )


def test_the_lane_travels_with_the_price():
    """A byte win on a lane with no runtime is not a byte win."""
    stock = tessera_tensor_payload_breakdown(
        SHAPE, family=get_tessera_family("TESSERA_E2M1_K2"), body_rate_q256=896
    )
    kernel = tessera_tensor_payload_breakdown(
        SHAPE, family=get_tessera_family("TESSERA_LM16_K2"), body_rate_q256=896
    )
    assert stock["lane"] == "stock" and stock["materialises"] is True
    assert stock["terminal_format"] == "NVFP4"
    assert kernel["lane"] == "kernel" and kernel["materialises"] is False
    assert kernel["terminal_format"] is None
    # Same rung, same bytes: the lane is about serving, not size.
    assert stock["total_bytes"] == kernel["total_bytes"]


# --- the allocator, end to end ---------------------------------------------

def _price(spec, q256, dloss=1.0):
    schedule = spec.column_schedule(q256, SHAPE[1])
    # The coset trellis ships an anchor table per distinct rate; the window
    # body's ALPHABET plane *is* its 2^L table and is already priced, so
    # supplying anchors alongside it is refused rather than double-counted.
    window = tessera_wire_recipe(spec, q256).body is BodyKind.WINDOW
    alphabets = (
        {} if window
        else {r: tuple(range(1 << (r + 1))) for r in set(schedule)}
    )
    return build_tessera_allocator_candidate(
        "layers.0.mlp.gate_proj", SHAPE, family=spec, body_rate_q256=q256,
        layout="tight", schedule=schedule, alphabets=alphabets,
        predicted_dloss=dloss, predicted_dloss_stderr=0.0,
        target_profile="research",
    )


@pytest.mark.parametrize("family,q256,bpp", DEFAULT_WIRE)
def test_the_allocator_prices_a_tessera_rung(family, q256, bpp):
    """The allocator prices what the exporter writes: the default wire."""
    candidate = _price(get_tessera_family(family), q256)
    assert candidate.family == family
    assert candidate.body_rate_q256 == q256
    assert candidate.n_params == SHAPE[0] * SHAPE[1]
    assert candidate.memory_bytes * 8 == pytest.approx(
        candidate.bits_per_param * candidate.n_params, rel=1e-9
    )
    # On the coset trellis, strictly *above* the ladder figure, because
    # `_price` supplies the anchor tables and they are real bytes.  The
    # published ladder quotes body+scale; a priced candidate also carries its
    # alphabets, and pretending otherwise would understate every artifact by
    # the side information it must ship.  The window body has no separate
    # anchor table -- its ALPHABET plane is the 2^L window and the ladder
    # figure already includes it -- so there the two are equal.
    window = tessera_wire_recipe(family, q256).body is BodyKind.WINDOW
    if window:
        assert candidate.bits_per_param == pytest.approx(bpp, abs=1e-9)
    else:
        assert candidate.bits_per_param > bpp
        assert candidate.bits_per_param - bpp < 1e-3


def test_the_rung_axis_is_a_rate_axis_the_allocator_can_search():
    """Adjacent rungs differ, and differ in order.

    This is the load-bearing claim behind treating Tessera as a continuously
    rateable format, and it briefly read the other way -- "the serialised
    payload is one size per family, so the only non-dominated rung is the
    family's top one".  A Tessera menu is not a menu of families each
    contributing one size; it is a continuum at a 1/256-bpp quantum, which is
    the whole reason the DP wants this format.

    One q256 step is 1/256 of a bit per weight, so adjacent rungs differ by a
    few parts in 10^3 -- small, but ordered and real, unlike the anchor-table
    side information that was mistaken for it.
    """
    spec = get_tessera_family("TESSERA_E4M3_K1")

    rungs = (1020, 1021, 1022, 1023, 1024)
    bodies = [
        tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=q
        )["exact_bpw"]
        for q in rungs
    ]
    assert bodies == sorted(bodies) and len(set(bodies)) == len(rungs)
    for lower, upper in zip(bodies, bodies[1:]):
        assert upper - lower == Fraction(1, 256)

    bpps = [_price(spec, q).bits_per_param for q in rungs]
    assert bpps == sorted(bpps), bpps


# --------------------------------------------- the two accountants are one


@pytest.mark.parametrize("q256", [128, 384, 640, 768, 896])
def test_the_registry_and_the_footprint_price_the_same_bytes(q256):
    """``FormatSpec`` and ``tessera_footprint`` must not be two opinions.

    They are read by different consumers: the DP's per-format bit cost goes
    through ``FormatSpec.effective_bits_for_shape``
    (``allocator_solver.py:748``) and the byte-budget gate goes through
    ``FormatSpec.memory_bytes_for_shape`` (``footprint.py``), while the
    Tessera candidate builder prices with ``tessera_tensor_payload_breakdown``.
    They disagreed: the registry charged ``ceil(artifact_bpp)`` *plus* a group
    scale term on top of a rate that already included the scale planes, so
    R896 priced at 4.25 bpp against an artifact that is 4.00.  The allocator
    was ranking Tessera against NVFP4 on a 6.25% overcharge it invented, which
    is principle 8's drift with the two halves inside one repository.
    """
    from prismaquant import format_registry as fr

    shape = (4096, 1536)
    spec = fr.get_format(f"TESSERA_E2M1_K2_R{q256}")
    breakdown = dict(
        tessera_tensor_payload_breakdown(
            shape, family="TESSERA_E2M1_K2", body_rate_q256=q256
        )
    )
    assert spec.memory_bytes_for_shape(shape) == breakdown["total_bytes"]


@pytest.mark.parametrize("q256", [128, 384, 640, 768, 896])
def test_the_rung_name_is_the_rate(q256):
    """The R-number is the body rate, and the registry reads it as a size.

    What sits on top of the body is the *rung's own wire*.  At the coset
    trellis's cap (R896) that is the span-2 LUT16 pair, a quarter-bit of plane
    plus a quarter-bit of labels; below the cap E2M1x2 is the window body over
    the same plane since 2026-09-02, which pays no labels and instead pays its
    2^12-byte table once per unit.  For part of 2026-09-01 this asserted the
    family cap at every rung instead, because the serialiser was writing the
    COMPLETION plane at its full ``cap - R`` width whatever depth the encoder
    spent.  Both readings have been wrong in opposite directions by the same
    amount, which is why this pins the formula rather than a single number.
    """
    from prismaquant import format_registry as fr

    shape = (4096, 1536)
    spec = fr.get_format(f"TESSERA_E2M1_K2_R{q256}")
    wire = tessera_wire_recipe("TESSERA_E2M1_K2", q256)
    extra = Fraction(64, 256)                       # LUT16: a nibble per 16
    if wire.body is BodyKind.WINDOW:
        extra += Fraction((1 << wire.window_bits) * 8, shape[0] * shape[1])
    else:
        extra += Fraction((wire.span - 1) * 256, wire.span * 2) / 256
    want = Fraction(q256, 256) + extra
    assert spec.bits_for_shape(shape) == want * shape[0] * shape[1]
    assert spec.effective_bits_for_shape(shape) == float(want)


def test_an_exact_rate_is_the_whole_rate_not_a_body_rate():
    """Declaring ``exact_bits_per_param`` must *replace* the group-scale term.

    If it were additive, every Tessera rung would be overcharged by
    ``scale_bits/group_size`` -- silently, since both numbers are plausible.
    """
    from fractions import Fraction

    from prismaquant import format_registry as fr

    spec = fr.FormatSpec(
        name="EXACTLY_FOUR", weight_bits=4, group_size=32, scale_bits=8,
        scale_dtype_name="fp8_e4m3", weight_element_dtype="fp4_e2m1",
        exact_bits_per_param=Fraction(4),
    )
    assert spec.effective_bits_for_shape((256, 256)) == 4.0


# ------------------------------------ the accountants price the whole grammar

RECIPE_SHAPES = ((2048, 4096), (96, 768))
RECIPE_FAMILIES = ("TESSERA_E2M1_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1")


def _wire_recipes():
    from prismaquant.tessera_formats import recipe_from_wire_names, tessera_wire_recipe

    return [
        ("the exporter's default", tessera_wire_recipe("TESSERA_E2M1_K2")),
        ("span 1 / s6b (minor 0)", recipe_from_wire_names(1, "s6b")),
        ("tcq / channel (minor 3)", recipe_from_wire_names(2, "channel")),
        ("window L=8 / lut16 (minor 2)", recipe_from_wire_names(1, "lut16", "window", 8)),
        ("window L=8 / channel", recipe_from_wire_names(1, "channel", "window", 8)),
    ]


@pytest.mark.parametrize("shape", RECIPE_SHAPES)
def test_the_footprint_prices_the_recipe_the_calculator_prices(shape):
    """Three accountants, one number, across the whole grammar.

    ``tessera.calculator.terminal_rate`` is the authority; PrismaQuant's
    closed form (``artifact_bpp``) is what the DP reads and its byte path
    (``tessera_tensor_payload_breakdown``) is what the budget gate reads.  They
    are pinned together here for every body, span and scale plane the wire has,
    not just for the one the exporter writes today, because the default is
    going to move and a test of the default would go quiet exactly then.
    """
    from tessera.calculator import terminal_rate

    from prismaquant.tessera_formats import (
        artifact_bpp, family_q256_bounds, family_rate_cap, scale_plane_name,
    )

    rows, columns = shape
    checked = 0
    for label, wire in _wire_recipes():
        plane = scale_plane_name(wire.scale_plane)
        for name in RECIPE_FAMILIES:
            spec = get_tessera_family(name)
            lo, hi = family_q256_bounds(spec, wire)
            for q in (lo, (lo + hi) // 2, hi):
                breakdown = tessera_tensor_payload_breakdown(
                    shape, family=spec, body_rate_q256=q, recipe=wire)
                priced = Fraction(*breakdown["exact_bpw_rational"])
                closed = artifact_bpp(spec, q, 0, recipe=wire, shape=shape)
                exact = terminal_rate(
                    q * spec.arity, rows, columns,
                    with_scale_base=plane == "s6b",
                    with_scale_refine=plane in ("s6b", "lut16"),
                    with_row_scale=plane == "channel",
                    span=wire.span, cap=family_rate_cap(spec, wire),
                    arity=spec.arity, completion=0, window_bits=wire.window_bits,
                )
                assert priced == exact == closed, (label, name, q, shape)
                checked += 1
    assert checked == len(_wire_recipes()) * len(RECIPE_FAMILIES) * 3


def test_the_channel_row_field_is_per_output_channel_not_per_code_row():
    """The bug the geometry convention was migrated to prevent.

    The footprint used to declare the layout in *code* space -- ``rows/arity``
    rows with the group geometry divided by the same factor -- which cancels
    exactly for the per-code and per-block planes, so every TCQ artifact priced
    identically either way.  It does not cancel for a per-*row* plane: DIAG_SV
    holds one fp16 per **output channel** (``tessera.layout._counts_for``), and
    a tuple code covers ``arity`` of them, so code-space rows under-declared a
    CHANNEL scale plane by exactly the arity.  The convention is now the wire's
    -- weight-space ``Geometry`` plus an explicit ``arity``, what
    ``tessera.unit_artifact`` and ``tessera.calculator`` both use.
    """
    from prismaquant.tessera_formats import recipe_from_wire_names

    rows, columns = 2048, 4096
    # Span 1, so the body plane is exactly the rung's bits per position and
    # whatever is left over is the scale plane.
    channel = recipe_from_wire_names(1, "channel")
    lut = recipe_from_wire_names(1, "lut16")

    k1 = tessera_tensor_payload_breakdown(
        (rows, columns), family="TESSERA_E2M1_K1", body_rate_q256=768,
        recipe=channel)
    k2 = tessera_tensor_payload_breakdown(
        (rows, columns), family="TESSERA_E2M1_K2", body_rate_q256=896,
        recipe=channel)
    # 2048 output rows x fp16 = 4096 bytes of row field, at BOTH arities.  A
    # code-space geometry would have charged k=2 half of that.
    for breakdown, body_bits_per_position in ((k1, Fraction(3)), (k2, Fraction(7, 2))):
        body_bytes = int(body_bits_per_position * rows * columns) // 8
        assert breakdown["total_bytes"] - body_bytes == rows * 2

    # And the row field really is the whole plane: dropping to a block plane
    # swaps 16 bits per row for 4 bits per 16 weights.
    k2_lut = tessera_tensor_payload_breakdown(
        (rows, columns), family="TESSERA_E2M1_K2", body_rate_q256=896,
        recipe=lut)
    assert (k2_lut["total_bytes"] - k2["total_bytes"]
            == rows * columns // 16 // 2 - rows * 2)


def test_a_window_or_channel_footprint_revalidates_as_itself():
    """A recorded footprint carries the recipe it priced, not the default.

    ``validate_tessera_tensor_payload_breakdown`` re-derives the arithmetic
    rather than trusting it, so it has to rebuild the same wire -- body, window
    width, span and plane -- from the report.  Without the body and window
    fields a WINDOW footprint would be re-priced as a TCQ one and fail its own
    revalidation, which is the shape of the failure the recipe fields exist to
    prevent: a report that is refused for naming the truth.
    """
    from prismaquant.tessera_formats import recipe_from_wire_names

    for _label, wire in _wire_recipes():
        breakdown = tessera_tensor_payload_breakdown(
            (2048, 4096), family="TESSERA_E4M3_K1", body_rate_q256=1024,
            recipe=wire)
        assert validate_tessera_tensor_payload_breakdown(breakdown) == breakdown

    window = recipe_from_wire_names(1, "channel", "window", 8)
    breakdown = dict(tessera_tensor_payload_breakdown(
        (2048, 4096), family="TESSERA_E4M3_K1", body_rate_q256=1024,
        recipe=window))
    assert breakdown["body_kind"] == "window"
    assert breakdown["window_bits"] == 8
    assert breakdown["scale_contract"] == "channel"
    assert breakdown["rate_cap"] == 8              # payload_bits, not minus one
    # The window table is charged: 2^8 one-byte grid codes, inline in the unit.
    assert breakdown["alphabet_bytes"] == 256

    # A report that lies about its body no longer addresses its own contents.
    breakdown["body_kind"] = "tcq"
    breakdown["window_bits"] = 0
    with pytest.raises(TesseraFormatError):
        validate_tessera_tensor_payload_breakdown(breakdown)


def test_the_route_travels_with_the_price_and_it_needs_the_plane():
    """A byte win on a route no kernel takes is not a byte win (principle 12).

    ``lane`` is the *grid's* question -- can these values be a hardware format
    at all.  Whether they actually materialise takes the scale plane too: an
    E4M3 tile over a per-16 block plane is FP8 bytes that no stock kernel
    reads, and an E2M1 tile over a per-channel plane is not the NVFP4 layout.
    """
    from prismaquant.tessera_formats import recipe_from_wire_names

    block = recipe_from_wire_names(2, "lut16")
    channel = recipe_from_wire_names(2, "channel")
    shape = (2048, 4096)

    nvfp4 = tessera_tensor_payload_breakdown(
        shape, family="TESSERA_E2M1_K2", body_rate_q256=896, recipe=block)
    assert nvfp4["lane"] == "stock" and nvfp4["materialises"] is True
    assert nvfp4["terminal_format"] == "NVFP4"
    assert nvfp4["activation_contract"] == "w4a4-nvfp4-e2m1-group16-ue4m3"
    from prismaquant import format_registry as fr
    assert nvfp4["min_capability_sm"] == fr.get_format("NVFP4").min_capability_sm

    fp8 = tessera_tensor_payload_breakdown(
        shape, family="TESSERA_E4M3_K1", body_rate_q256=1024, recipe=channel)
    assert fp8["lane"] == "stock" and fp8["materialises"] is True
    assert fp8["terminal_format"] == "FP8_E4M3"
    assert fp8["activation_contract"] == "w8a8-dynamic-e4m3-channel"
    assert fp8["min_capability_sm"] == fr.get_format("FP8_E4M3").min_capability_sm

    for family, rung, wire in (("TESSERA_E4M3_K1", 1024, block),
                               ("TESSERA_E2M1_K2", 896, channel)):
        kernel = tessera_tensor_payload_breakdown(
            shape, family=family, body_rate_q256=rung, recipe=wire)
        # Stock-lane grid, kernel-lane route: the plane is the other half.
        assert kernel["lane"] == "stock"
        assert kernel["materialises"] is False
        assert kernel["terminal_format"] is None
        assert kernel["activation_contract"] == "w?a16-tessera-kernel-decode"
