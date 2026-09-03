"""The Tessera grid-space seam, pinned against the rungs that were measured.

The point of this file is not coverage.  ``tessera_formats`` exists to let
PrismaQuant's allocator price Tessera continuously, and two things have to hold
for that to be honest rather than decorative:

1. The rungs with published weight-space numbers must be namable, and must
   price at the bpp those numbers were published at.  An allocator addressing
   rungs nobody measured is addressing fiction.
2. Every rung the module offers must hand the encoder a schedule it will
   accept.  "Continuous" is a claim about realisability, so it is tested by
   realising, not by asserting a range.
"""
from fractions import Fraction

import torch
import pytest

from prismaquant.tessera_formats import (
    ANCHOR_BUDGET_BITS,
    LANE_KERNEL,
    LANE_STOCK,
    SCALE_LUT_BITS_Q256,
    SCALE_PLANE_BITS_Q256,
    tessera_wire_defaults,
    wire_overhead_q256,
    SUPERBLOCK_WEIGHTS,
    TesseraFormatError,
    TesseraRateSurface,
    artifact_bpp,
    enumerate_grid_space,
    family_q256_bounds,
    family_rate_cap,
    get_tessera_family,
    parse_tessera_format_name,
    realisable_rungs,
    recipe_from_wire_names,
    scale_plane_name,
    tessera_family,
    tessera_wire_recipe,
    validate_body_rate_q256,
)
from tessera.calculator import terminal_rate
from tessera.export import TCQ_RECIPE, recipe_table
from tessera.grammar import forest_plane_bytes
from tessera.manifest import BodyKind

#: Big enough that the per-unit terms are small, and a multiple of the
#: 256-column superblock so every rung is realisable over it.
REFERENCE_SHAPE = (4096, 4096)


def _forest_bpp(spec, rung, shape, wire=None):
    """What a TCQ unit's forest costs per position at ``shape``.

    A TCQ body writes an ALPHABET and a DESCENDANT plane per unit -- one
    anchor table and one descendant table per *distinct* rate in the schedule
    -- so a rung's artifact rate is its position-domain rate plus this
    (RobTand/prismaquant#126).  The published ladders below are position-domain
    figures, so they are compared against ``artifact_bpp`` minus this term
    rather than restated at the artifact's rate; the size comes from Tessera's
    own ``forest_plane_bytes``, never from a formula written here.
    """
    wire = tessera_wire_recipe(spec, rung) if wire is None else wire
    if BodyKind(wire.body) is not BodyKind.TCQ:
        return Fraction(0)
    rows, columns = shape
    rates = spec.column_schedule(rung, columns, recipe=wire)
    return Fraction(
        sum(forest_plane_bytes(rates, family_rate_cap(spec, wire))) * 8,
        rows * columns,
    )


def _coset_rungs(spec):
    """The rungs of ``spec`` the exporter writes on the coset trellis.

    Since 2026-09-02 a family's wire is a function of the rung -- E2M1x2 is the
    window body below the trellis's cap and the trellis at it -- so a test
    about the trellis has to ask which rungs those are rather than assume the
    family.  Returns them in rung order, possibly empty (E4M3 is the window
    body everywhere).
    """
    return [
        q for q in realisable_rungs(spec, recipe=TCQ_RECIPE)
        if tessera_wire_recipe(spec, q).body is BodyKind.TCQ
    ]

# (label, family name, body q256, artifact bpp as measured on built bytes)
#
# Every entry is ``q256/256 + 0.5`` -- the body the rung names plus the S6b
# scale plane -- at the exporter's ``completion=0``.  The two E4M3 rows are
# sub-cap and are the ones that moved twice on 2026-09-01: published at 4.5
# and 5.5, "corrected" to 7.5 when the serialiser was found writing the
# COMPLETION plane at full width, and back to 4.5 and 5.5 once that was fixed
# (tessera `a96064b`).  4.5 and 5.5 were right the whole time; the middle
# value was a bug's arithmetic.
MEASURED_LADDER = [
    ("E2M1 scalar R=3", "TESSERA_E2M1_K1", 768, 3.5),
    ("E2M1 k=2 R=7", "TESSERA_E2M1_K2", 896, 4.0),
    ("free-16 k=2 R=7", "TESSERA_LM16_K2", 896, 4.0),
    ("free-16 scalar R=3", "TESSERA_LM16_K1", 768, 3.5),
    ("E4M3 R=4 (sub-cap)", "TESSERA_E4M3_K1", 1024, 4.5),
    ("E4M3 R=5 (sub-cap)", "TESSERA_E4M3_K1", 1280, 5.5),
]


@pytest.mark.parametrize("label,family,q256,bpp", MEASURED_LADDER)
def test_the_measured_rungs_price_at_their_published_bpp(label, family, q256, bpp):
    """Published on the span-1 S6b wire (tessera schema minor 0), so priced
    on it: a published figure belongs to the wire it was measured on.

    And to the *domain* it was measured in.  These are **position-domain**
    rates -- body plus block plane, the quantity the ladder was derived as --
    while ``artifact_bpp`` states what a unit weighs on the wire, which since
    #126 includes the forest.  The two differ by a per-unit term, so the
    published figure is recovered by subtracting it rather than by asking the
    accountant for a number it no longer means.  Tessera made the same split
    on its side: ``terminal_rate``'s ``with_forest`` defaults to False so its
    published position-domain figures still mean what they were derived as.
    """
    spec = get_tessera_family(family)
    assert validate_body_rate_q256(spec, q256) == q256
    wire = recipe_from_wire_names(1, "s6b")
    priced = artifact_bpp(
        spec, q256, span=1, scale_plane="s6b", shape=REFERENCE_SHAPE)
    position_domain = priced - _forest_bpp(spec, q256, REFERENCE_SHAPE, wire)
    assert position_domain == Fraction(int(bpp * 256), 256), label
    # ...and the artifact really does weigh more than the published rate.
    assert priced > position_domain, label


def test_the_default_wire_keeps_k2_at_four_and_lifts_arity_one_by_a_quarter():
    """The coset-trellis wire (span 2 over a LUT plane): the stored labels cost
    (L-1)/L bits per CODE, the LUT plane saves a quarter-bit per position, so at
    arity 2 the two cancel and at arity 1 they do not.

    E2M1x2 keeps that wire **at** the trellis's cap, which is the rung 4.0 bpp
    is addressed at, so the position-domain headline is unmoved by the
    2026-09-02 flip.  "4.0 bpp" is that headline and not the artifact's rate:
    the unit also carries a 512-byte forest, which is 0.0013 bpp here and
    0.1333 on a 96x320 expert (#126).
    """
    k2 = get_tessera_family("TESSERA_E2M1_K2")
    k1 = get_tessera_family("TESSERA_E2M1_K1")
    priced_k2 = artifact_bpp(k2, 896, shape=REFERENCE_SHAPE)
    assert priced_k2 - _forest_bpp(k2, 896, REFERENCE_SHAPE) == Fraction(4)
    assert (artifact_bpp(k1, 768, shape=REFERENCE_SHAPE)
            - _forest_bpp(k1, 768, REFERENCE_SHAPE)) == Fraction(15, 4)
    # The span-2 LUT wire and the span-1 S6b wire weigh the same, forest and
    # all: both are TCQ over the same schedule, so the forest cancels.
    assert priced_k2 == artifact_bpp(
        k2, 896, span=1, scale_plane="s6b", shape=REFERENCE_SHAPE)


def test_the_e4m3_rung_has_a_size_but_no_rate():
    """E4M3's wire is the window body over the CHANNEL plane at every rung.

    Both of its planes are charged per *unit* -- one fp16 per output row, one
    ``2^14``-byte table -- so the rung has no bits-per-parameter rate to quote
    and the shape-free accountant refuses instead of quoting a floor.  Named a
    shape, it is exact, and the two entry points that answer it (the closed
    form here, the ``FormatSpec`` the registry synthesizes) must agree.
    """
    from prismaquant import format_registry as fr

    assert BodyKind(tessera_wire_recipe("TESSERA_E4M3_K1", 1024).body) is (
        BodyKind.WINDOW)
    with pytest.raises(TesseraFormatError, match="per unit"):
        artifact_bpp("TESSERA_E4M3_K1", 1024)

    shape = (2048, 4096)
    # 4.0 body + 16/4096 row field + 2^14 * 8 / (2048*4096) table.
    priced = artifact_bpp("TESSERA_E4M3_K1", 1024, shape=shape)
    assert priced == Fraction(4) + Fraction(16, 4096) + Fraction(
        (1 << 14) * 8, 2048 * 4096)
    assert priced == Fraction(1029, 256)

    spec = fr.get_format("TESSERA_E4M3_K1_R1024")
    assert spec.bits_for_shape(shape) == priced * shape[0] * shape[1]
    # A narrower unit pays more for the same two per-unit planes.
    assert artifact_bpp("TESSERA_E4M3_K1", 1024, shape=(96, 768)) > priced


@pytest.mark.parametrize("label,family,q256,bpp", MEASURED_LADDER)
def test_every_measured_rung_realises_a_schedule(label, family, q256, bpp):
    spec = get_tessera_family(family)
    schedule = spec.column_schedule(q256)
    assert len(schedule) == SUPERBLOCK_WEIGHTS
    assert min(schedule) >= 1 and max(schedule) <= spec.rate_cap
    assert sum(schedule) == q256 * spec.arity * SUPERBLOCK_WEIGHTS // 256


def test_continuity_every_offered_rung_is_encodable():
    """The load-bearing claim: the module offers nothing it cannot realise."""
    for spec in enumerate_grid_space():
        rungs = realisable_rungs(spec)
        assert len(rungs) > 1
        # Endpoints plus an interior sample; the exhaustive sweep over all
        # ~9000 rungs lives in tessera's own test_grid_space_continuity.
        for q in (rungs[0], rungs[len(rungs) // 3], rungs[-1]):
            schedule = spec.column_schedule(q)
            assert sum(schedule) * 256 == q * spec.arity * SUPERBLOCK_WEIGHTS


def test_arity_is_what_fills_the_rungs_between():
    """k=2 addresses 4.0 bpp, which the scalar family provably cannot."""
    scalar = tessera_family("E2M1", 1)
    tup = tessera_family("E2M1", 2)
    with pytest.raises(TesseraFormatError):
        validate_body_rate_q256(scalar, 896)
    assert validate_body_rate_q256(tup, 896) == 896
    assert tup.root_rate(896) == Fraction(7)          # 7 bits per k=2 code
    # 3.5 body + 0.5 scale in the position domain, plus the unit's forest.
    assert (artifact_bpp(tup, 896, shape=REFERENCE_SHAPE)
            - _forest_bpp(tup, 896, REFERENCE_SHAPE)) == Fraction(4)


def test_rate_cap_closes_the_code_space():
    """|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R) must equal 2^payload_bits.

    That closure is the *coset trellis's*, and so is the cap it fixes.  The
    window body shapes with the ``L - R`` bits of history a position shares
    with its predecessors rather than with a code bit, so it caps a whole bit
    higher and closes nothing -- which is why the cap is asked of the recipe
    (:func:`family_rate_cap`) and never re-derived as a subtraction.
    """
    from tessera.export import TCQ_RECIPE

    for spec in enumerate_grid_space():
        coset_cap = family_rate_cap(spec, TCQ_RECIPE)
        assert coset_cap == spec.payload_bits - 1
        for rate in range(1, coset_cap + 1):
            assert (1 << (rate + 1)) * (1 << (coset_cap - rate)) == (
                1 << spec.payload_bits
            )
        # And the family advertises its own wire's cap, not the coset one.
        window = spec.recipe.body is not BodyKind.TCQ
        assert spec.rate_cap == spec.payload_bits - (0 if window else 1)


def test_the_lane_follows_the_base_grid_not_the_arity():
    """A k=2 code over E2M1 still decodes to E2M1 nibbles, so it is stock."""
    assert tessera_family("E2M1", 1).lane == LANE_STOCK
    assert tessera_family("E2M1", 2).lane == LANE_STOCK
    assert tessera_family("E4M3", 1).lane == LANE_STOCK
    assert tessera_family("LM16", 1).lane == LANE_KERNEL
    assert tessera_family("LM16", 2).lane == LANE_KERNEL
    assert tessera_family("LM16", 1).terminal_format is None
    assert tessera_family("E2M1", 2).terminal_format == "NVFP4"


def test_the_family_reads_its_grid_from_tessera_not_from_here():
    for spec in enumerate_grid_space():
        grid = spec.payload_grid()
        assert grid.payload_bits == spec.payload_bits
        assert grid.arity == spec.arity
        # ``PayloadGrid.rate_cap`` is the coset trellis's ceiling -- it is what
        # ``tessera.export.tcq_cap_q256`` reads to decide where E2M1x2 leaves
        # the window body -- so it is that cap this must match, not whatever
        # the family's current recipe advertises.
        assert grid.rate_cap == family_rate_cap(spec, TCQ_RECIPE)


def test_a_code_space_at_the_encoder_wall_is_refused():
    """E4M3 k=2 is 65536 anchors per step -- what got k=4 refused."""
    with pytest.raises(TesseraFormatError, match="wall the encoder already refuses"):
        tessera_family("E4M3", 2)
    with pytest.raises(TesseraFormatError):
        tessera_family("E2M1", 4)
    # The budget is the TCQ body's, not the grid's, so this is the invariant
    # that holds over the whole space -- and the flat ``payload_bits <
    # ANCHOR_BUDGET_BITS`` it replaces was the wrong one: it is what made the
    # 16-bit family unnamable and took the top half of the rate axis with it.
    for spec in enumerate_grid_space():
        if spec.payload_bits < ANCHOR_BUDGET_BITS:
            continue
        bodies = {
            BodyKind(entry.recipe.body)
            for entry in recipe_table(spec.payload_grid())
        }
        assert BodyKind.TCQ not in bodies, (
            f"{spec.name} is offered at {spec.payload_bits} payload bits and "
            "its wire reaches the TCQ body: that is 2^payload_bits anchors "
            "scored per step, which the encoder refuses"
        )


def test_the_window_body_admits_a_grid_the_trellis_could_not_afford():
    """The 16-bit family exists, and it exists for the reason Tessera gives.

    ``TesseraFamily.__post_init__`` used to refuse ``payload_bits >= 16`` flat,
    which reads the anchor budget as a property of the GRID.  It is a property
    of the BODY: a WINDOW step scores ``2^window_bits`` states (16384 at the
    default L=14) and has no forest at all, so the 65536-code BF16 grid is
    touched once per unit to snap its table.  Tessera says exactly this in
    ``alphabet.SERIALISABLE_GRIDS``, and the consequence of not hearing it was
    that PrismaQuant could not name the family at all.
    """
    fam = tessera_family("BF16")
    assert fam.payload_bits == ANCHOR_BUDGET_BITS
    assert fam.name == "TESSERA_BF16_K1"
    # every rung of this grid is the window body; nothing here is a TCQ rung
    assert {BodyKind(e.recipe.body) for e in recipe_table(fam.payload_grid())} == {
        BodyKind.WINDOW}
    # ...and the family is now enumerable, which is what "allocatable" needs
    assert "TESSERA_BF16_K1" in {f.name for f in enumerate_grid_space()}
    # the cap is the window's -- the whole width of the grid, not width - 1
    assert family_rate_cap(fam) == 16
    assert fam.mathematical_q256_bounds == (256, 4096)
    # BF16 at k=2 stays refused, and by Tessera's own raise rather than ours
    with pytest.raises(TesseraFormatError, match="tessera will not build this grid"):
        tessera_family("BF16", 2)


def test_the_bf16_family_is_allocatable_and_not_merely_namable(monkeypatch):
    """Namable, priceable and enumerable is not the same as allocatable.

    The family cleared every gate an accountant can see -- ``tessera_family``
    built it, ``enumerate_grid_space`` listed it, both footprints priced it --
    and the allocator still could not pick a single rung of it, because
    ``tessera_render._grid_for`` held a SECOND base->grid map that had never
    heard of BF16.  It answered ``NotImplementedError`` ("a free base"), so
    ``tessera_rung_is_serialisable`` answered False, so ``_producer_eligible``
    answered False, so the menu dropped every BF16 rung in *both* modes and
    ``require_producer_formats`` -- the allocator's own gate -- refused the
    name.  Nothing raised anywhere; the family simply never appeared.

    So this test asks the question the issue asked, in the allocator's own
    vocabulary rather than the family's: does the registry synthesize a spec,
    does the guard accept the name, does the menu carry rungs, and does each
    of those rungs resolve back through ``get_format``.  A test that only
    asks ``tessera_formats`` cannot see a second map living somewhere else.
    """
    from prismaquant import format_registry as fr
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_footprint import tessera_exact_bits_for_shape
    from prismaquant.tessera_render import tessera_rung_is_serialisable

    name = "TESSERA_BF16_K1_R2048"
    fam = tessera_family("BF16")
    shape = (2048, 1024)

    # (a) the wire gate: this grid's digest is one of Tessera's commitments
    assert tessera_rung_is_serialisable(name) is True

    monkeypatch.setenv(tm.MENU_MODE_ENV, tm.MENU_RESEARCH)

    # (b) the registry synthesizes a spec, and its A side is the route's
    spec = fr.get_format(name)
    assert (spec.act_bits, spec.act_dtype_name, spec.act_group_size) == (
        16, "bfloat16", 0)
    assert spec.bits_for_shape_fn is not None
    assert Fraction(spec.bits_for_shape_fn(shape)) == tessera_exact_bits_for_shape(
        fam, 2048, shape)

    # (c) the allocator's own gate accepts the name
    assert fr.format_is_producer_eligible(name) is True
    fr.require_producer_formats([name], where="bf16 allocatability")

    # (d) the menu carries rungs of it, and every one resolves back
    rungs = tm.expand_tessera_menu(
        shape, mode=tm.MENU_RESEARCH, families=(fam,), step_q256=256)
    assert rungs, "the research menu holds no BF16 rung"
    for rung in rungs:
        assert fr.get_format(rung.format_name).act_bits == 16
        assert rung.admission.terminal_format == "BF16"
        assert rung.admission.route_status == tm.ROUTE_STATUS_UNATTESTED

    # ...and attestation is untouched: the pinned contract publishes no BF16
    # cell, so the attested menu still refuses the family outright.
    monkeypatch.setenv(tm.MENU_MODE_ENV, tm.MENU_ATTESTED)
    assert fr.format_is_producer_eligible(name) is False
    with pytest.raises(ValueError, match="producer-eligible"):
        fr.require_producer_formats([name], where="bf16 allocatability")


def _scalar_grid_for_test(base):
    from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID

    return {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID, "BF16": BF16_GRID}[base]


def test_the_wire_commitment_is_asked_of_the_grid_not_of_every_rate(monkeypatch):
    """A menu digests each grid once, however many rungs it enumerates.

    Whether a grid is a permanent wire commitment cannot vary with the rate
    the body runs at, so asking it per rung is pure recomputation -- and the
    recomputation is a SHA over every value in the grid.  It was free while
    the widest grid held 256 values and stopped being free the moment the
    16-bit family was admitted at 65536: one 2048x1024 unit's **attested**
    menu went from 0.19 s to 52 s, with 234 of 234.5 profiled seconds inside
    ``grid_digest``, because the per-rung ``lru_cache`` was sized 64 against a
    menu of 6916 names and thrashed on every one.

    So the guard counts calls rather than seconds: a clock measures the box,
    and the defect here is a quantity of work.  It is stated as "at most once
    per family" because that is the property that makes the cost independent
    of how many rates a family offers.
    """
    import tessera.alphabet as alphabet

    from prismaquant import tessera_menu as tm
    from prismaquant import tessera_render as tr

    monkeypatch.delenv(tm.MENU_MODE_ENV, raising=False)

    calls = []
    real_digest = alphabet.grid_digest

    def counting_digest(grid):
        calls.append(len(grid.values))
        return real_digest(grid)

    monkeypatch.setattr(alphabet, "grid_digest", counting_digest)

    # Cleared by name so the guard measures the *work*, not the presence of
    # whichever helper happens to hold the memo today.
    for cached in ("family_grid_is_serialisable", "tessera_rung_is_serialisable",
                   "_grid_for"):
        fn = getattr(tr, cached, None)
        if fn is not None:
            fn.cache_clear()
    tm.menu_families.cache_clear()

    families = tm.menu_families()
    baseline = len(calls)
    # Once per *candidate* family -- the enumeration digests the arities it
    # then rejects too, and that bound is the search space, not the menu.
    from prismaquant.tessera_formats import _HARDWARE_BASES

    ceiling = len(_HARDWARE_BASES) * tm._MAX_ARITY_SEARCH
    assert len(families) <= baseline <= ceiling, (
        f"{baseline} digests to enumerate {len(families)} families "
        f"from {ceiling} candidates"
    )

    rungs = tm.expand_tessera_menu((2048, 1024), mode=tm.MENU_ATTESTED, step_q256=16)
    assert len(rungs) >= 0  # the attested set may be empty; the cost may not vary
    after_menu = len(calls)

    # The whole point: enumerating hundreds of rungs adds no digests at all.
    assert after_menu == baseline, (
        f"{after_menu - baseline} extra grid digests while expanding a menu; "
        "the wire commitment is a property of the grid, not of the rate"
    )

    # And the research menu, which keeps every priced rung, is no different.
    research = tm.expand_tessera_menu(
        (2048, 1024), mode=tm.MENU_RESEARCH, step_q256=16
    )
    assert research, "the research menu should carry rungs"
    assert len(calls) == baseline, (
        f"{len(calls) - baseline} extra grid digests over {len(research)} "
        "research rungs"
    )

    # Guard the guard: the counter is wired to something that really is called.
    assert baseline >= 1 and max(calls) >= 256


def test_the_render_leg_and_the_accountants_read_one_grid():
    """Regression guard (passes on both sides): one grid per family, everywhere.

    ``tessera_render._grid_for`` used to build its own
    ``tuple_grid(SCALAR, arity)`` while the pricing built ``_build_grid``,
    which returns the scalar itself at arity 1.  The two agreed by luck --
    ``tuple_grid(g, 1) is g`` -- so the duplication cost nothing until a base
    was added to one map and not the other, and then it cost the whole family.
    This pins the agreement as a property rather than a coincidence: the
    render leg's grid IS the family's, for every family the menu can build,
    so a grid added anywhere is added everywhere.  It also states, for the
    record, that fixing the map moved no bytes -- the grid the renderer gets
    for the pre-existing families is the object it always got.
    """
    from tessera.alphabet import tuple_grid
    from prismaquant.tessera_render import _grid_for

    checked = 0
    for fam in enumerate_grid_space():
        try:
            grid = _grid_for(fam)
        except NotImplementedError:
            assert fam.lane == LANE_KERNEL, fam.name   # free grids only
            continue
        assert grid is fam.payload_grid(), fam.name
        # ...and identical to the spelling the render leg used before
        assert grid == tuple_grid(_scalar_grid_for_test(fam.base), fam.arity)
        checked += 1
    assert checked >= 4, checked


@pytest.mark.parametrize("base", ["BF16", "E4M3"])
@pytest.mark.parametrize("shape", [(2048, 1024), (512, 256), (4096, 4096)])
def test_a_window_table_is_charged_at_the_grids_own_code_width(base, shape):
    """The ALPHABET plane holds ``code_bytes * 2^L`` bytes, not ``2^L``.

    A code is as wide as the code space and BF16's code IS a bf16 word, so its
    window table is two bytes an entry.  Charging it at one byte under-prices
    the 16-bit route by half its table -- 0.0625 bpp on a 2048x1024 unit at
    L=14 -- and the byte budget is spent in this currency, so the accountant
    and the wire have to agree exactly.  Pinned against
    ``tessera.calculator.terminal_rate``, which takes ``code_bytes`` from the
    same grid, as exact Fractions.
    """
    fam = tessera_family(base)
    lo, hi = fam.mathematical_q256_bounds
    for rung in (lo, 1024, 2048, hi):
        if not lo <= rung <= hi:
            continue
        wire = tessera_wire_recipe(fam, rung)
        assert BodyKind(wire.body) is BodyKind.WINDOW
        assert artifact_bpp(fam, rung, shape=shape) == terminal_rate(
            rung, shape[0], shape[1],
            with_scale_base=False, with_scale_refine=False, with_row_scale=True,
            cap=family_rate_cap(fam), arity=fam.arity, span=1,
            window_bits=wire.window_bits, code_bytes=fam.code_bytes,
        ), (base, rung, shape)
    assert fam.code_bytes == (2 if base == "BF16" else 1)


def test_format_names_round_trip_and_foreign_names_are_not_claimed():
    for spec in enumerate_grid_space():
        lo, hi = spec.mathematical_q256_bounds
        for q in (lo, hi):
            assert parse_tessera_format_name(spec.format_name(q)) == (spec, q)
    for foreign in ("NVFP4", "FP8_DYNAMIC", "BF16", "TCQ_E2M1_R256", None, 17):
        assert parse_tessera_format_name(foreign) is None


def test_a_tessera_shaped_name_with_an_illegal_rung_raises():
    """Silence would put an unpriced format in front of the DP."""
    with pytest.raises(TesseraFormatError):
        parse_tessera_format_name("TESSERA_E2M1_K1_R9999")
    with pytest.raises(TesseraFormatError):
        parse_tessera_format_name("TESSERA_E4M3_K2_R1024")   # refused family


def test_the_scale_plane_is_half_a_bit_everywhere():
    """A flat half-bit on top of the body, at BOTH ends of the family.

    "Flat" is now a statement about the *scale plane* alone, not about the
    wire's whole overhead.  The overhead also carries the per-unit forest, so
    it varies with both the shape and the rung -- a two-rate schedule carries
    two forests -- and the family-level "interval shifted by a constant"
    (``artifact_q256_bounds``) is gone with it (#126).  What survives, and is
    what this test was ever about, is that the block plane costs the same at
    both ends of every family: measure it as the overhead at a fixed shape
    minus the forest at that shape.
    """
    assert SCALE_PLANE_BITS_Q256 == 128
    rows, columns = REFERENCE_SHAPE
    checked = 0
    for spec in enumerate_grid_space():
        for span, plane, flat in (
            (1, "s6b", SCALE_PLANE_BITS_Q256),
            (2, "lut16", SCALE_LUT_BITS_Q256 + Fraction(128, spec.arity)),
        ):
            wire = recipe_from_wire_names(span, plane)
            # The bounds are the *recipe's*: a TCQ body caps a bit lower than
            # the window body a family's own wire may name.
            body_lo, body_hi = family_q256_bounds(spec, wire)
            for rung in (body_lo, body_hi):
                extra = wire_overhead_q256(
                    spec, span, plane, shape=REFERENCE_SHAPE, rung=rung)
                forest = _forest_bpp(spec, rung, REFERENCE_SHAPE, wire) * 256
                assert extra - forest == flat, (spec.name, rung, plane)
                checked += 1
    assert checked >= 16, checked
    # And the per-unit half is real: without a shape there is no rate at all.
    for spec in enumerate_grid_space():
        with pytest.raises(TesseraFormatError, match="per unit"):
            wire_overhead_q256(spec, 1, "s6b")


def test_an_adaptive_surface_must_name_its_evidence():
    with pytest.raises(TesseraFormatError, match="unattributed"):
        TesseraRateSurface(
            family="TESSERA_E2M1_K1", mode="adaptive", bounds_q256=(256, 768)
        )
    ok = TesseraRateSurface(
        family="TESSERA_E2M1_K1", mode="adaptive", bounds_q256=(256, 768),
        anchor_q256=(512, 768), source_identity_sha256="0" * 64,
    )
    assert "0" * 64 in ok.identity()
    assert len(ok.rungs()) == 513


def test_a_surface_refuses_rungs_its_family_cannot_address():
    with pytest.raises(TesseraFormatError):
        TesseraRateSurface(
            family="TESSERA_E2M1_K1", mode="dense", bounds_q256=(256, 896)
        )
    with pytest.raises(TesseraFormatError):
        TesseraRateSurface(
            family="TESSERA_E2M1_K1", mode="dense", bounds_q256=(768, 256)
        )


def test_unknown_bases_name_the_legal_set():
    with pytest.raises(TesseraFormatError, match="LM"):
        tessera_family("TCQ_E2M1", 1)
    with pytest.raises(TesseraFormatError):
        get_tessera_family("TCQ_E2M1_R256")


def test_the_bpp_formula_agrees_with_tesseras_exact_byte_accountant():
    """The guard against the two repos drifting apart on what a rung weighs.

    `artifact_bpp` is a closed form because the allocator prices thousands of
    rungs per Linear and cannot afford to build a plane layout for each.  That
    makes it a *second* statement of something tessera already computes
    exactly, and a second statement is a drift bug waiting to happen.

    ``completion`` is the load-bearing argument, and it is checked at BOTH
    ends.  This test once passed ``completion=cap`` because ``unit_artifact``
    wrote the COMPLETION plane at full width whatever depth the encoder used --
    so the flat ladder that produced looked like the format.  It was a bug in
    three places (tessera `a96064b`, `eec18ba`); the plane now follows the
    spend.  Pinning both ends is what stops either error from coming back: at
    ``completion=0`` the size must track the rung, and at ``completion=cap`` it
    must be the family's ceiling at every rung.

    The ``arity != 1`` skip was the other half of the blind spot: it hid the
    one family where the old formula and the accountant disagreed outright
    (K2 R896 -- 4.0 against 2.25).  Both arities are checked now.
    """
    from tessera.calculator import terminal_rate

    checked = 0
    for spec in enumerate_grid_space():
        rungs = _coset_rungs(spec)
        if not rungs:
            continue
        cap = family_rate_cap(spec, TCQ_RECIPE)
        for q in (rungs[0], rungs[len(rungs) // 2], rungs[-1]):
            wire = tessera_wire_recipe(spec, q)
            plane = scale_plane_name(wire.scale_plane)
            # ``terminal_rate`` takes q256 per *code* -- it calls
            # ``root_from_q256`` directly -- while a rung is quoted per
            # position, so the arity factor has to be applied by the caller.
            for depth in (0, cap):
                exact = terminal_rate(
                    q * spec.arity, *REFERENCE_SHAPE, with_scale_refine=True,
                    with_scale_base=plane == "s6b", span=wire.span,
                    cap=cap, arity=spec.arity, completion=depth,
                    # The wire's own figure, forest and all -- ``with_forest``
                    # is off by default so the calculator's published
                    # position-domain figures keep meaning what they meant,
                    # and a caller pricing a whole unit passes True (#126).
                    with_forest=True,
                )
                assert artifact_bpp(
                    spec, q, depth, shape=REFERENCE_SHAPE) == exact, (
                        spec.name, q, depth)
                checked += 1
            assert artifact_bpp(spec, q, None, shape=REFERENCE_SHAPE) == (
                artifact_bpp(spec, q, cap, shape=REFERENCE_SHAPE))
    assert checked >= 12


def test_the_rung_sets_the_size_and_the_completion_depth_is_the_other_axis():
    """Stated on its own so neither of the two errors can come back.

    A column at body rate ``R`` writes ``R`` bits and may spend up to
    ``cap - R`` completion bits.  At ``completion=0`` -- the exporter's default
    and the measured optimum -- the artifact tracks the rung one for one, and
    the ladder is a continuous, strictly increasing size axis.  At full depth
    body and completion sum to ``cap`` and every rung of a family really does
    weigh the same; that is a corner of the rate surface, not the surface.

    For a stretch on 2026-09-01 this file asserted the corner as the whole
    thing, because the serialiser wrote the plane at full width regardless.
    """
    checked = 0
    for spec in enumerate_grid_space():
        # The completion plane is the coset trellis's second axis; the window
        # body has none (``artifact_bpp`` refuses a depth under it), so this is
        # a statement about the rungs whose wire is that trellis.
        rungs = _coset_rungs(spec)
        if not rungs:
            continue
        cap = family_rate_cap(spec, TCQ_RECIPE)
        wire = tessera_wire_recipe(spec, rungs[0])
        # The overhead is per unit now, so it is measured at a shape -- and
        # the forest inside it depends on the rung, so the flat part is what
        # is left after subtracting it.  ``_forest_bpp`` is already per
        # position; the rest is quoted in q256, hence the /256.
        def _flat(rung):
            return (wire_overhead_q256(
                spec, recipe=wire, shape=REFERENCE_SHAPE, rung=rung) / 256
                - _forest_bpp(spec, rung, REFERENCE_SHAPE, wire))
        overhead = _flat(rungs[0])
        assert all(_flat(q) == overhead for q in rungs), spec.name
        ceiling = Fraction(cap * 256, spec.arity) / 256 + overhead

        priced = [
            artifact_bpp(spec, q, recipe=wire, shape=REFERENCE_SHAPE)
            - _forest_bpp(spec, q, REFERENCE_SHAPE, wire)
            for q in rungs
        ]
        assert priced == sorted(priced), spec.name
        assert len(set(priced)) == len(rungs), (spec.name, "flat ladder")
        for rung, bpp in zip(rungs, priced):
            assert bpp == Fraction(rung, 256) + overhead
        if rungs[-1] * spec.arity == cap * 256:
            assert priced[-1] == ceiling

        at_full = {
            artifact_bpp(spec, q, None, recipe=wire, shape=REFERENCE_SHAPE)
            - _forest_bpp(spec, q, REFERENCE_SHAPE, wire)
            for q in rungs
        }
        assert at_full == {ceiling}, (spec.name, sorted(map(float, at_full)))
        checked += 1
    assert checked, "no family left with a completion axis to check"


# --------------------------------------------------- the wire gate vs the lane


def test_a_rung_that_renders_can_still_be_unwritable():
    """``_grid_for`` admits any grid the renderer can build, but only grids
    whose digest is a wire commitment can be written -- and the exporter's
    refusal lands after allocation and after the production cache is built,
    which on a GLM-scale run is hours of work discarded at the last step.  The
    predicate has to be answerable up front.

    The example used to be E4M3, which was writable all along and merely
    missing from the registry (tessera `a4de134` admitted it).  With it in,
    the renderable set and the writable set **coincide today**: the renderer
    takes hardware base grids only, and all four hardware-derived grids
    Tessera commits to -- E2M1, E2M1x2, E4M3 and the two-byte-coded BF16 --
    are registered.  A free Lloyd-Max grid is unwritable for a reason no
    registry line can fix -- its values are fitted to the tensor, so no reader
    rebuilds them from a name -- but it does not render either, so it cannot
    demonstrate the gap.

    The gap is therefore asserted through the mechanism rather than through an
    example: remove a grid's digest from the wire commitment and the predicate
    must go False *while the render still succeeds*.  That is the case the
    predicate exists for, and it will recur the moment a fourth grid renders.
    """
    from unittest import mock

    import tessera.alphabet as alphabet
    from prismaquant.tessera_render import (
        clear_serialisable_cache,
        render_tessera_weight,
        tessera_rung_is_serialisable,
    )

    torch.manual_seed(0)
    weight = torch.randn(64, 512) * 0.02
    assert render_tessera_weight(weight, "TESSERA_E4M3_K1_R1024").shape == (64, 512)
    assert tessera_rung_is_serialisable("TESSERA_E4M3_K1_R1024") is True
    assert tessera_rung_is_serialisable("TESSERA_E2M1_K2_R896") is True

    without = {digest: grid for digest, grid in alphabet.SERIALISABLE_GRIDS.items()
               if grid.name != "E4M3"}
    assert len(without) == len(alphabet.SERIALISABLE_GRIDS) - 1
    with mock.patch.object(alphabet, "SERIALISABLE_GRIDS", without):
        # The predicate is memoised at two levels -- per rung, and per family
        # since the digest became the dominant cost of a menu -- so a test that
        # patches the registry underneath it must clear both or it reads its
        # own earlier answer.  The seam's own function does that, so a third
        # level would be cleared here without this test being edited again.
        clear_serialisable_cache()
        # Still renders -- if it stopped, this would pass for the wrong reason.
        assert render_tessera_weight(
            weight, "TESSERA_E4M3_K1_R1024").shape == (64, 512)
        assert tessera_rung_is_serialisable("TESSERA_E4M3_K1_R1024") is False
    # Both levels again on the way out: the verdict cached under the patched
    # registry is the one the next test in this file would otherwise read.
    clear_serialisable_cache()


def test_an_attested_lane_cannot_admit_an_unwritable_rung():
    """``producer_eligible`` is the AND of "the wire can carry it" and "a
    runtime serves it".  The second is one lookup against the pinned serving
    contract, so the whole point of the AND is that a route attestation --
    the deliberate act that follows a serve receipt -- must not quietly put a
    rung on the menu that the exporter will refuse."""
    from unittest import mock

    from prismaquant import tessera_render as tr

    with mock.patch.object(tr, "tessera_lane_attested", lambda name, **kw: True):
        assert tr.synthesize_tessera_spec("TESSERA_E2M1_K2_R896").producer_eligible
        assert tr.synthesize_tessera_spec(
            "TESSERA_E4M3_K1_R1024"
        ).producer_eligible
        assert not tr.synthesize_tessera_spec(
            "TESSERA_LM16_K1_R768"
        ).producer_eligible


def test_no_tessera_rung_is_producer_eligible_on_the_pinned_release():
    """Principle 9, and the whole point of the pin.

    Tessera's OWN packaged contract publishes both families with
    ``device_qualified`` native cells, and the ``tessera`` package is
    importable here (it is a producer-side render dependency).  Producer
    eligibility is still False, and the *reason* is the tracked serving pin:
    there is no Tessera release tag, so it carries PENDING sentinels.  Both
    halves are asserted, because "False" without "because the pin" is a
    correlation, and the day someone deletes the pin conjunct this test has to
    fail rather than keep passing on an unrelated absence.

    The full-blooded admission logic lives in
    ``tests/test_tessera_lane_admission.py``.
    """
    from prismaquant import tessera_render as tr
    from prismaquant.tessera_render import synthesize_tessera_spec
    from prismaquant.tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        load_tessera_serving_runtime_pin,
        require_exact_tessera_runtime_release,
    )

    # The table is PRESENT and DOES govern both families -- so the refusal
    # below cannot be coming from an absent or silent contract.
    table, _formats = tr._pinned_serving_table()
    assert table.present
    assert table.governs("TESSERA_E2M1_K2") and table.governs("TESSERA_E4M3_K1")

    with pytest.raises(TesseraServingRuntimePinError, match="PENDING"):
        require_exact_tessera_runtime_release(load_tessera_serving_runtime_pin())
    assert tr._release_pin_satisfied() is False

    for name in ("TESSERA_E2M1_K2_R896", "TESSERA_E4M3_K1_R1024",
                 "TESSERA_E2M1_K1_R640"):
        assert tr.tessera_lane_attested(name) is False
        assert synthesize_tessera_spec(name).producer_eligible is False


# ------------------------------------------------- the recipe is the unit of account
#
# ``tessera.export.wire_recipe`` is the single statement of which body, span,
# scale plane and window the exporter writes for a grid at a rung.  Everything
# below is written against *explicitly constructed* recipes rather than the
# current default, because the default is going to move -- the window body over
# a CHANNEL plane is measured better on E4M3 and is held behind a kernel gate,
# not a doubt -- and a test that only exercises today's wire would go quiet on
# the day the flip happens, which is the day it is needed.

TCQ_FAMILIES = ("TESSERA_E2M1_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1")

#: Two shapes, one square-ish and one narrow and not a power of two, so a term
#: that is right per position and wrong per unit cannot hide in a coincidence.
CROSS_CHECK_SHAPES = ((2048, 4096), (96, 768))


def _recipes_under_test():
    from prismaquant.tessera_formats import recipe_from_wire_names, tessera_wire_recipe

    return [
        ("the exporter's default", tessera_wire_recipe("TESSERA_E2M1_K2")),
        ("span 1 / s6b (minor 0)", recipe_from_wire_names(1, "s6b")),
        ("span 2 / lut16 (minor 1)", recipe_from_wire_names(2, "lut16")),
        ("tcq / channel (minor 3)", recipe_from_wire_names(2, "channel")),
        ("window L=8 / lut16", recipe_from_wire_names(1, "lut16", "window", 8)),
        ("window L=8 / channel", recipe_from_wire_names(1, "channel", "window", 8)),
    ]


@pytest.mark.parametrize("shape", CROSS_CHECK_SHAPES)
def test_the_closed_form_prices_every_recipe_the_calculator_prices(shape):
    """``artifact_bpp`` is a second statement of what tessera computes exactly.

    The allocator prices thousands of rungs per Linear and cannot afford to
    build a plane layout for each, so the closed form has to exist -- and a
    second statement of one number is a drift bug waiting for the wire to
    change.  ``tessera.calculator.terminal_rate`` is the authority; this pins
    the two together across the whole grammar rather than across the one
    recipe that happens to be the default.

    Five terms have to line up, and each of them has been wrong somewhere at
    some point: the body's rate at the *recipe's* cap (``payload_bits`` under a
    window body, ``payload_bits - 1`` under the TCQ trellis), the span labels,
    the scale plane, and the three **per-unit** costs -- a CHANNEL plane's
    fp16 per output row, a window body's ``2^L``-byte table, and a TCQ body's
    forest -- none of which has a per-position rate until a shape is named.
    The forest is the one that was missing until 2026-09-03 (#126), and it is
    also why ``with_forest=True`` is passed here: the calculator's default is
    the position domain its published figures were derived in.
    """
    from fractions import Fraction

    from tessera.calculator import terminal_rate

    from prismaquant.tessera_formats import (
        family_q256_bounds, family_rate_cap, scale_plane_name,
    )

    rows, columns = shape
    checked = 0
    for label, wire in _recipes_under_test():
        plane = scale_plane_name(wire.scale_plane)
        for name in TCQ_FAMILIES:
            spec = get_tessera_family(name)
            lo, hi = family_q256_bounds(spec, wire)
            for q in (lo, (lo + hi) // 2, hi):
                mine = artifact_bpp(spec, q, 0, recipe=wire, shape=shape)
                exact = terminal_rate(
                    q * spec.arity, rows, columns,
                    with_scale_base=plane == "s6b",
                    with_scale_refine=plane in ("s6b", "lut16"),
                    with_row_scale=plane == "channel",
                    span=wire.span,
                    cap=family_rate_cap(spec, wire),
                    arity=spec.arity,
                    completion=0,
                    window_bits=wire.window_bits,
                    with_forest=BodyKind(wire.body) is BodyKind.TCQ,
                )
                assert mine == exact, (label, name, q, shape, mine, exact)
                assert isinstance(mine, Fraction)
                checked += 1
    assert checked == len(_recipes_under_test()) * len(TCQ_FAMILIES) * 3


def test_a_per_unit_wire_term_has_no_price_without_a_shape():
    """Every Tessera wire charges something per *unit*, so none has a rate.

    Three terms, and none of them is bits per position until a shape is named:
    a CHANNEL plane's ``16/columns`` row field, a window body's
    ``8 * 2^L / (rows*columns)`` table, and a TCQ body's forest, which is the
    third and the one that closed the last shape-free branch (#126).  The
    accountant refuses them rather than dropping them, because the only
    consumer of a shape-free value is ``FormatSpec.exact_bits_per_param``,
    which ``memory_bytes_for_shape`` multiplies by the parameter count as an
    *exact* rate, and a ``Fraction`` cannot carry "this is a floor".

    The previous version of this test split the wires into a shape-free half
    and a per-unit half and asserted both were non-empty.  The shape-free half
    is now empty by construction, which is the fix rather than a loss of
    coverage: it was populated entirely by TCQ-over-a-block-plane rungs, i.e.
    by exactly the rungs that were priced 512 bytes light.
    """
    from prismaquant.tessera_formats import (
        recipe_from_wire_names, tessera_wire_recipe, wire_overhead_q256,
    )

    per_unit = 0
    for name in TCQ_FAMILIES:
        spec = get_tessera_family(name)
        for q in (spec.mathematical_q256_bounds[0], 1024):
            try:
                validate_body_rate_q256(spec, q)
            except TesseraFormatError:
                continue
            with pytest.raises(TesseraFormatError, match="per unit"):
                artifact_bpp(spec, q)
            assert artifact_bpp(spec, q, shape=(2048, 4096)) > 0
            per_unit += 1
    assert per_unit, per_unit

    spec = get_tessera_family("TESSERA_E4M3_K1")
    for wire in (
        recipe_from_wire_names(2, "channel"),
        recipe_from_wire_names(1, "lut16", "window", 8),
        recipe_from_wire_names(2, "lut16"),          # TCQ over a block plane
    ):
        with pytest.raises(TesseraFormatError, match="per unit"):
            wire_overhead_q256(spec, recipe=wire)
        with pytest.raises(TesseraFormatError, match="per unit"):
            artifact_bpp(spec, 1024, 0, recipe=wire)
        priced = artifact_bpp(spec, 1024, 0, recipe=wire, shape=(2048, 4096))
        assert priced > Fraction(4)

    # A TCQ rung priced with a shape but no rung has no forest to size, and
    # says so rather than dropping it.
    with pytest.raises(TesseraFormatError, match="only defined at a rung"):
        wire_overhead_q256(
            spec, recipe=recipe_from_wire_names(2, "lut16"),
            shape=(2048, 4096))

    # The per-unit terms are shape-dependent in the direction they must be:
    # a narrower tensor pays more per weight for the same row field.
    chan = recipe_from_wire_names(2, "channel")
    assert (artifact_bpp(spec, 1024, 0, recipe=chan, shape=(2048, 512))
            > artifact_bpp(spec, 1024, 0, recipe=chan, shape=(2048, 4096)))


def test_a_window_recipe_widens_the_family_bounds():
    """The rate ceiling is the *body's*, and the two bodies differ by a bit.

    The TCQ trellis spends one bit of the payload on its convolutional code, so
    ``|A_R| * |D(a)|`` closes at ``2^P`` with ``cap = payload_bits - 1``.  The
    window body's shaping is the ``L - R`` bits of history a position shares
    with its predecessors, not a code bit, so a position may spend the grid's
    whole width -- ``R = payload_bits`` is an ordinary rung there, which is
    E2M1x2 at 4.0 body bits per weight (``tessera.export._plan_for``).

    The TCQ numbers are asserted as literals: this is the interval the DP
    searches, and it must not move as a side effect of the bounds learning
    about a body they do not use.
    """
    from prismaquant.tessera_formats import (
        family_q256_bounds, family_rate_cap, recipe_from_wire_names,
    )

    tcq = recipe_from_wire_names(2, "lut16")
    window = recipe_from_wire_names(1, "lut16", "window", 8)

    literal = {
        "TESSERA_E2M1_K1": ((256, 768), (256, 1024), 3, 4),
        "TESSERA_E2M1_K2": ((128, 896), (128, 1024), 7, 8),
        "TESSERA_E4M3_K1": ((256, 1792), (256, 2048), 7, 8),
    }
    for name, (tcq_bounds, win_bounds, tcq_cap, win_cap) in literal.items():
        spec = get_tessera_family(name)
        # The family advertises its own wire's interval, and the two literals
        # below say which that is for each grid: E4M3 is the window body
        # everywhere since 2026-09-02, the two E2M1 families are the trellis at
        # the rung their bounds are stated at.
        expected = win_bounds if name == "TESSERA_E4M3_K1" else tcq_bounds
        assert spec.mathematical_q256_bounds == expected, name
        assert spec.rate_cap == (win_cap if name == "TESSERA_E4M3_K1"
                                 else tcq_cap), name
        assert family_q256_bounds(spec, tcq) == tcq_bounds, name
        assert family_rate_cap(spec, tcq) == tcq_cap, name
        # Wider under a window recipe, at the top end only.
        assert family_q256_bounds(spec, window) == win_bounds, name
        assert family_rate_cap(spec, window) == win_cap, name

        # The three functions that answer "which rungs exist" must agree with
        # the bounds, or the menu offers something the encoder refuses.
        lo, hi = win_bounds
        assert realisable_rungs(spec, recipe=window)[0] == lo
        assert realisable_rungs(spec, recipe=window)[-1] == hi
        assert validate_body_rate_q256(spec, hi, recipe=window) == hi
        with pytest.raises(TesseraFormatError):
            # Above the coset trellis's cap, named explicitly: E4M3's default
            # recipe is the window body now, and under it `hi` is legal.
            validate_body_rate_q256(spec, hi, recipe=tcq)
        with pytest.raises(TesseraFormatError):
            validate_body_rate_q256(spec, hi + 1, recipe=window)
        # And the schedule really is buildable at the widened ceiling.
        schedule = spec.column_schedule(hi, SUPERBLOCK_WEIGHTS, recipe=window)
        assert max(schedule) == win_cap and len(schedule) == SUPERBLOCK_WEIGHTS


def test_the_two_scalar_wire_projection_refuses_what_it_cannot_state():
    """``tessera_wire_defaults`` is a projection, and it says so by failing.

    It drops the body, the window and any per-family variation, so it is wrong
    the day the exporter stops writing one TCQ wire for every grid.  Rather
    than return two plausible scalars for a wire that is no longer described by
    two scalars, it raises and names the recipe function.
    """
    from unittest import mock

    from prismaquant import tessera_formats as tf

    # Today it raises, and that is the correct answer rather than a
    # regression: since 2026-09-02 E4M3 is the window body over CHANNEL while
    # E2M1 is the span-2 coset trellis over LUT16, so there is no pair of
    # scalars that describes the exporter's wire.  The message names both the
    # divergence it saw and the function that does answer the question.
    with pytest.raises(TesseraFormatError, match="tessera_wire_recipe"):
        tessera_wire_defaults()

    # It still answers in a world where one wire really does cover every
    # family -- which is what it meant before the flip, and all it ever meant.
    uniform = tf.recipe_from_wire_names(2, "lut16")
    with mock.patch.object(tf, "tessera_wire_recipe", lambda *a, **k: uniform):
        assert tessera_wire_defaults() == (2, "lut16")

    window = tf.recipe_from_wire_names(1, "channel", "window", 14)
    with mock.patch.object(tf, "tessera_wire_recipe", lambda *a, **k: window):
        with pytest.raises(TesseraFormatError, match="tessera_wire_recipe"):
            tessera_wire_defaults()


# ------------------------------------------------------ the fifth axis: route


def test_the_route_is_a_property_of_the_grid_and_the_plane_together():
    """What a decoded tile executes as, carried in the fields the registry
    already uses for it.

    The activation contract is not a property of the grid alone: an E4M3 tile
    over a per-16 block plane is still E4M3 bytes, and no stock kernel reads
    FP8 weights at that scale granularity.  It takes both.

    The representation is deliberately not a new field.  ``act_bits`` /
    ``act_dtype_name`` / ``act_group_size`` / ``min_capability_sm`` are how
    ``NVFP4`` (``format_registry.py:738``) and ``FP8_E4M3`` (``:878``) declare
    the same thing, and ``FormatSpec.act_quant_changes_input`` (``:101``) is
    the single predicate every consumer -- the bit-exact cost short-circuit,
    the KL validator's activation assignment, the caches -- reads it through.
    """
    from prismaquant.tessera_formats import recipe_from_wire_names
    from prismaquant.tessera_render import synthesize_tessera_spec

    # E2M1 over a per-16 block plane -> NVFP4 -> W4A4 on Blackwell.
    spec = synthesize_tessera_spec("TESSERA_E2M1_K2_R896")
    assert (spec.act_bits, spec.act_dtype_name, spec.act_group_size) == (
        4, "fp4_e2m1", 16)
    # By reference from the registry row whose A-side the rung executes.
    from prismaquant import format_registry as fr
    assert spec.min_capability_sm == fr.get_format("NVFP4").min_capability_sm
    assert spec.act_quant_changes_input is True

    # E4M3 over a CHANNEL plane -> the stock per-channel FP8 pair -> W8A8.
    channel = recipe_from_wire_names(1, "channel", "window", 8)
    spec = synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024", recipe=channel, shape=(2048, 4096))
    assert (spec.act_bits, spec.act_dtype_name, spec.act_group_size) == (
        8, "fp8_e4m3", 0)
    assert spec.min_capability_sm == fr.get_format("FP8_E4M3").min_capability_sm
    assert spec.act_quant_changes_input is True
    assert spec.group_size == 0        # per-output-channel, as FP8_E4M3 spells it

    # E4M3's *default* wire is that pair since 2026-09-02, so the rung the
    # registry synthesizes by name executes W8A8 with no recipe named at all.
    by_name = synthesize_tessera_spec("TESSERA_E4M3_K1_R1024", shape=(2048, 4096))
    assert (by_name.act_bits, by_name.act_dtype_name) == (8, "fp8_e4m3")

    # The two combinations with no stock MMA layout: kernel lane, weight-only.
    kernel_only = [
        # E4M3 over a per-16 block plane: still E4M3 bytes, but no kernel reads
        # FP8 weights at that granularity.
        ("TESSERA_E4M3_K1_R1024", recipe_from_wire_names(2, "lut16")),
        ("TESSERA_E2M1_K2_R896", recipe_from_wire_names(2, "channel")),
        ("TESSERA_LM16_K2_R896", None),                       # a free grid
    ]
    for name, wire in kernel_only:
        spec = synthesize_tessera_spec(
            name, recipe=wire, shape=None if wire is None else (2048, 4096))
        assert spec.act_bits is None, name
        assert spec.act_dtype_name is None, name
        assert spec.act_quant_changes_input is False, name
        assert spec.min_capability_sm == 80, name

    # And none of it is a serving claim (principle 9 / 14): a layout that
    # *would* materialise is still not a route a pinned runtime was measured
    # taking, so nothing is producer-eligible.
    for name, wire in [("TESSERA_E2M1_K2_R896", None),
                       ("TESSERA_E4M3_K1_R1024", channel)]:
        spec = synthesize_tessera_spec(
            name, recipe=wire, shape=None if wire is None else (2048, 4096))
        assert spec.producer_eligible is False, name


def test_the_activation_side_is_the_serving_formats_own_quantiser():
    """A W4A4 rung's A side must be NVFP4's, not a second implementation.

    Measurement said the activation leg is what dominates at W4A4 (the EXL3
    gap moves 1.72x -> 1.20x when the A side is priced as it serves), so an
    A-side priced by a private copy of the group-16 dynamic RTN is a rendering
    confound on the axis that decides the comparison.  Taken by reference from
    the registry, it cannot drift.
    """
    import torch

    from prismaquant import format_registry as fr
    from prismaquant.tessera_formats import recipe_from_wire_names
    from prismaquant.tessera_render import synthesize_tessera_spec

    torch.manual_seed(0)
    x = torch.randn(8, 256)

    w4 = synthesize_tessera_spec("TESSERA_E2M1_K2_R896")
    assert torch.equal(
        w4.activation_quantize_dequantize(x),
        fr.get_format("NVFP4").activation_quantize_dequantize(x),
    )

    w8 = synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024",
        recipe=recipe_from_wire_names(1, "channel", "window", 8),
        shape=(2048, 4096),
    )
    assert torch.equal(
        w8.activation_quantize_dequantize(x),
        fr.get_format("FP8_E4M3").activation_quantize_dequantize(x),
    )

    # Weight-only stays the identity -- E4M3 over a block plane, which is the
    # wire E4M3 left on 2026-09-02 and still a legal thing to price.
    kernel = synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024", recipe=recipe_from_wire_names(2, "lut16"))
    assert torch.equal(kernel.activation_quantize_dequantize(x), x)

    # And the W side: a spec synthesized for a recipe must *render* that
    # recipe, or the price and the callable inside one FormatSpec describe two
    # different artifacts.
    weight = torch.randn(64, 512) * 0.02
    from prismaquant.tessera_render import render_tessera_weight

    assert torch.equal(
        w8.quantize_dequantize(weight),
        render_tessera_weight(
            weight, "TESSERA_E4M3_K1_R1024",
            recipe=recipe_from_wire_names(1, "channel", "window", 8)),
    )
    assert not torch.equal(
        w8.quantize_dequantize(weight),
        render_tessera_weight(
            weight, "TESSERA_E4M3_K1_R1024",
            recipe=recipe_from_wire_names(2, "lut16")),
    )


# ---------------------------------------------- one rendering, two code paths


@pytest.mark.parametrize("label,name,recipe_args", [
    ("the exporter's default wire", "TESSERA_E2M1_K2_R896", None),
    ("the exporter's default wire", "TESSERA_E4M3_K1_R1024", None),
    ("the exporter's default wire, below the E2M1x2 cap", "TESSERA_E2M1_K2_R768", None),
    ("span 1 / s6b (minor 0)", "TESSERA_E2M1_K1_R768", (1, "s6b", "tcq", 0)),
    ("window L=8 / channel (minor 3)", "TESSERA_E4M3_K1_R1024",
     (1, "channel", "window", 8)),
    ("window L=8 / lut16 (minor 2)", "TESSERA_E2M1_K2_R896",
     (1, "lut16", "window", 8)),
])
def test_the_render_leg_and_the_exporter_are_one_rendering(label, name, recipe_args):
    """Principle 8, established by construction rather than by agreement.

    ``render_tessera_weight`` is what the AURA cost, the allocator's price and
    the real-KL validation all measure; ``encode_linear`` writes what ships.
    They are two code paths, so the identity has to be asserted: every encode
    setting the exporter resolves -- the code, the group/half geometry, the
    scale refit count, the trellis weighting, and now the whole recipe (body,
    span, plane, window bits and seed, source sigmas) -- has to be resolved the
    same way on both sides.  ``encode_linear(verify=True)`` already pins its
    own bytes to its reconstruction, so equality here means the surrogate is
    measuring the artifact.

    Leaving *any* of those out is silent: it renders a legal tensor at the
    wrong wire.  ``encode_unit``'s own defaults are the pre-minor-1 wire, kept
    so old artifacts stay reproducible, which is exactly what makes the
    omission plausible.
    """
    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact

    from prismaquant.tessera_formats import recipe_from_wire_names
    from prismaquant.tessera_render import _grid_for, render_tessera_weight

    family, rung = parse_tessera_format_name(name)
    recipe = None if recipe_args is None else recipe_from_wire_names(*recipe_args)

    torch.manual_seed(0)
    weight = torch.randn(64, 512, dtype=torch.float32) * 0.02

    rendered = render_tessera_weight(weight, name, recipe=recipe)

    kwargs = {} if recipe is None else dict(
        span=recipe.span, scale_plane=recipe.scale_plane, body=recipe.body,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
    )
    unit = encode_linear(
        weight, grid=_grid_for(family), q256=rung, name=name, **kwargs)
    off_the_wire = read_unit_artifact(unit.blob, device=weight.device)

    assert torch.equal(rendered, off_the_wire.to(weight.dtype)), label


# ---------------------------------------------------------------------------
# The rung-keyed memo bound (prismaquant#134, and tessera#46 one level up)
#
# Two memos on this path are keyed on a RUNG -- ``tessera_rung_is_serialisable``
# on the full format name, ``_recipe_for`` on ``(base, base_size, arity, rung)``
# -- and both were sized by a round number, 4096 and 512.  These tests pin the
# RULE, "a memo survives one pass over the space its key ranges over", and
# never a count.  A count written here would be the same defect one level up:
# 6916 is the right answer to a different question (the four families
# ``menu_families`` admits) and is short of the twelve ``enumerate_grid_space``
# builds, so a memo sized off it goes undersized the day something prices an
# ``LM*`` family.
# ---------------------------------------------------------------------------

def _rung_keyed_memos():
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_render as tr

    return {
        "tessera_render.tessera_rung_is_serialisable":
            tr.tessera_rung_is_serialisable,
        "tessera_formats._recipe_for": tfm._recipe_for,
    }


def _grid_space_rung_names():
    """Every rung name the grid space addresses, which is the memos' key set."""
    return [
        spec.format_name(rung)
        for spec in enumerate_grid_space()
        for rung in realisable_rungs(spec)
    ]


def test_the_rung_key_space_is_the_grid_space_and_not_the_menu():
    """Which space, and why it is not the one tessera#46 counted.

    Both memos are keyed on a rung of *any* family a name can name, and
    ``parse_tessera_format_name`` admits every family
    ``enumerate_grid_space`` can build -- the eight ``TESSERA_LM*`` ones
    included -- while ``menu_rungs_per_shape`` counts the four the menu
    admits.  The bound has to cover the wider set, so this pins the
    containment rather than either number.
    """
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_formats import (
        grid_space_rung_keys, recipe_cache_bound,
    )

    families = list(enumerate_grid_space())
    reachable = len(_grid_space_rung_names())

    # The bound covers every reachable rung name, and every rung above a
    # recipe's cap that ``tessera_wire_recipe`` will still accept as a key.
    assert grid_space_rung_keys() >= reachable
    assert recipe_cache_bound() >= reachable + len(families)

    # And the menu's count does not cover it: strictly fewer families, so
    # strictly fewer rungs, so sizing either memo off the menu is undersized
    # by construction -- today by about 2x.
    menu = {spec.name for spec in tm.menu_families()}
    grid = {spec.name for spec in families}
    assert menu < grid, "the menu is meant to be a subset of the grid space"
    assert grid_space_rung_keys() > tm.menu_rungs_per_shape()


def test_every_rung_keyed_memo_survives_one_pass_over_its_key_space():
    """The rule.  Not ``maxsize == <number>`` -- ``maxsize >= the key space``.

    A memo smaller than its key space evicts its own entries while a pass is
    still filling them, and then answers nothing on the next pass: measured at
    0 hits against 6916 misses *per pass* on the serialisable memo at
    ``maxsize=4096``, and 0 cross-rung hits on ``_recipe_for`` at 512.
    """
    from prismaquant.tessera_formats import enumerate_grid_space

    families = list(enumerate_grid_space())
    rungs = len(_grid_space_rung_names())
    floors = {
        "tessera_render.tessera_rung_is_serialisable": rungs,
        # plus the one rung-independent (``rung=None``) entry per family.
        "tessera_formats._recipe_for": rungs + len(families),
    }
    for name, memo in _rung_keyed_memos().items():
        held = memo.cache_info().maxsize
        assert held is None or held >= floors[name], (
            f"{name} holds {held} entries against a {floors[name]}-key space: "
            "one pass evicts itself")


def test_a_second_pass_over_the_whole_key_space_recomputes_nothing():
    """The behaviour the bound buys, measured on the memo rather than a clock.

    A wall-clock assertion would be a coin flip on a loaded box; hits and
    misses say the same thing and say it exactly.  The second half of the test
    is its own sensitivity check: the same sweep against the literal that
    shipped until 2026-09-03 must reproduce the reported failure, or this test
    would pass over the bug it is about.
    """
    from functools import lru_cache

    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_render as tr

    names = _grid_space_rung_names()

    tr.clear_serialisable_cache()
    tfm.clear_recipe_cache()
    for name in names:
        tr.tessera_rung_is_serialisable(name)
    cold = tr.tessera_rung_is_serialisable.cache_info()
    assert cold.hits == 0 and cold.misses == len(names)
    for name in names:
        tr.tessera_rung_is_serialisable(name)
    warm = tr.tessera_rung_is_serialisable.cache_info()
    assert warm.misses == cold.misses, (
        "a repeat pass re-answered rungs the memo had already answered")
    assert warm.hits == len(names)

    tfm.clear_recipe_cache()
    for spec in enumerate_grid_space():
        tessera_wire_recipe(spec)                   # the rung=None entry
        for rung in realisable_rungs(spec):
            tessera_wire_recipe(spec, rung)
    cold = tfm._recipe_for.cache_info()
    for spec in enumerate_grid_space():
        tessera_wire_recipe(spec)
        for rung in realisable_rungs(spec):
            tessera_wire_recipe(spec, rung)
    warm = tfm._recipe_for.cache_info()
    assert warm.misses == cold.misses
    assert warm.currsize == cold.currsize
    # ``>=`` and not ``==``: ``realisable_rungs`` asks the family's own
    # rung-independent recipe once per family to find the interval, so a sweep
    # makes twelve more lookups than it has keys.  Those are hits too.
    assert warm.hits - cold.hits >= cold.misses

    # Sensitivity: the pre-fix literal, on the same sweep.  13,068 distinct
    # keys cycled through 4096 slots is the worst case for an LRU -- every
    # access evicts the entry the next access wants -- which is exactly the
    # "0 hits / 13 832 misses" the issue reported from a two-pass menu.
    undersized = lru_cache(maxsize=4096)(
        tr.tessera_rung_is_serialisable.__wrapped__)
    for _ in range(2):
        for name in names:
            undersized(name)
    starved = undersized.cache_info()
    assert starved.hits == 0
    assert starved.misses == 2 * len(names)
    assert starved.currsize == 4096


def test_the_bound_moves_when_the_family_roster_moves(monkeypatch):
    """Derived, not restated: shrink the roster and the live memos shrink.

    This is the test a literal cannot pass.  ``enumerate_grid_space`` went from
    eleven families to twelve on 2026-09-02 with nobody touching a cache line,
    so the property worth pinning is that the bound is a function of the
    roster -- checked by moving the roster now, rather than by waiting for the
    next family and hoping someone notices.  BF16 is the one removed because
    admitting it is the growth that broke the old numbers.
    """
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_render as tr

    def _rebuild():
        tr.clear_serialisable_cache()
        tfm.clear_recipe_cache()

    wide_keys = tfm.grid_space_rung_keys()
    wide_bound = tfm.recipe_cache_bound()
    assert tr.tessera_rung_is_serialisable.cache_info().maxsize == wide_keys
    assert tfm._recipe_for.cache_info().maxsize == wide_bound

    narrower = {base: spec for base, spec in tfm._HARDWARE_BASES.items()
                if base != "BF16"}
    assert len(narrower) == len(tfm._HARDWARE_BASES) - 1
    try:
        monkeypatch.setattr(tfm, "_HARDWARE_BASES", narrower)
        _rebuild()
        narrow_keys = tfm.grid_space_rung_keys()
        assert narrow_keys < wide_keys
        assert tr.tessera_rung_is_serialisable.cache_info().maxsize == narrow_keys
        assert tfm._recipe_for.cache_info().maxsize == tfm.recipe_cache_bound()
        assert tfm.recipe_cache_bound() < wide_bound
    finally:
        # The memos were BUILT while the roster was short.  Restoring the
        # roster does not resize them, so the clear has to come after the undo
        # or every later test in the session runs undersized.
        monkeypatch.undo()
        _rebuild()

    assert tfm.grid_space_rung_keys() == wide_keys
    assert tr.tessera_rung_is_serialisable.cache_info().maxsize == wide_keys
    assert tfm._recipe_for.cache_info().maxsize == wide_bound


def test_sizing_the_recipe_memo_does_not_ask_the_recipe_memo():
    """Why the interval is counted at ``payload_bits`` and not at the cap.

    ``family_rate_cap`` reads the family's recipe, so a bound stated against it
    would call ``_recipe_for`` while sizing ``_recipe_for``.  The guard in
    ``lazily_sized_cache`` refuses that rather than answering it uncached, and
    this pins that the live bound does not need the guard: forcing the lazy
    build must leave the memo empty.
    """
    from prismaquant import tessera_formats as tfm

    tfm.clear_recipe_cache()
    info = tfm._recipe_for.cache_info()          # forces the sizing
    assert info.maxsize == tfm.recipe_cache_bound()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0), (
        "counting the key space asked the memo it was sizing")


def test_a_bound_that_reenters_the_memo_it_sizes_is_refused():
    """The guard itself, on a toy memo: refused loudly, not answered quietly.

    A fallback that silently answered uncached would hide the one failure this
    wrapper cannot repair -- a bound that cannot be computed at all.
    """
    from prismaquant.tessera_formats import lazily_sized_cache

    @lazily_sized_cache(lambda: doubled(1))
    def doubled(n):
        return 2 * n

    with pytest.raises(TesseraFormatError, match="reenters the memo it sizes"):
        doubled(2)


def test_a_second_unit_asks_neither_rung_keyed_memo_again():
    """The reuse these memos exist for is per UNIT, not per pass.

    Both answers are shape-independent, and ``tessera_campaign`` expands a menu
    once per Linear (``for name in targets``), so on a model whose 37,861 units
    share 25 distinct shapes the same 6916 names are asked ~37,861 times.  At
    ``maxsize=4096`` none of that was cashed: a *different* shape's pass paid
    6916 misses on each memo, every time.  So the check is a second pass at a
    second shape, where a shape-keyed memo legitimately misses and these two
    must not.
    """
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_menu as tm
    from prismaquant import tessera_render as tr

    tr.clear_serialisable_cache()
    tfm.clear_recipe_cache()
    tm.expand_tessera_menu((256, 256), mode=tm.MENU_RESEARCH, step_q256=1)
    cold_names = tr.tessera_rung_is_serialisable.cache_info()
    cold_recipes = tfm._recipe_for.cache_info()
    assert cold_names.misses > 0 and cold_recipes.misses > 0

    tm.expand_tessera_menu((512, 256), mode=tm.MENU_RESEARCH, step_q256=1)
    warm_names = tr.tessera_rung_is_serialisable.cache_info()
    warm_recipes = tfm._recipe_for.cache_info()

    assert warm_names.misses == cold_names.misses, (
        "a second unit re-answered rungs the first unit had already answered")
    assert warm_names.hits - cold_names.hits == cold_names.misses
    assert warm_recipes.misses == cold_recipes.misses
    assert warm_recipes.hits > cold_recipes.hits


# ---------------------------------------------------------------------------
# The family-keyed memo bound (prismaquant#148)
#
# Five more memos are keyed on a FAMILY -- ``_build_grid``,
# ``_tcq_body_is_reachable`` and ``tessera_family`` here, ``_grid_for`` and
# ``family_grid_is_serialisable`` in tessera_render -- and all five were sized
# by round numbers (64/64/256/64/64) against a roster that moved twice in a
# week.  These tests pin the RULE, "a memo holds every (base, arity) the
# enumerators ask", and never a count.  A count written here would be the same
# defect one level up: the union of the two enumerators' loops, not the dozen
# families they build today.
# ---------------------------------------------------------------------------

def _family_keyed_memos():
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_render as tr

    return {
        "tessera_formats._build_grid": tfm._build_grid,
        "tessera_formats._tcq_body_is_reachable": tfm._tcq_body_is_reachable,
        "tessera_formats.tessera_family": tfm.tessera_family,
        "tessera_render._grid_for": tr._grid_for,
        "tessera_render.family_grid_is_serialisable":
            tr.family_grid_is_serialisable,
    }


def _asked_family_pairs():
    """Every (base, arity) the two enumerators ask, from their loop inputs.

    Not from their outputs: both enumerators skip the families the encoder
    refuses, while the bound has to be stated against what is ASKED -- the
    grid space's bases at its arities, and the menu's hardware bases at its
    searched arities.  Read off the constants that own those loops, never
    restated here.
    """
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_menu as tm

    grid = {(base, arity)
            for base in (*tfm._HARDWARE_BASES, *tfm._MEASURED_FREE_BASES)
            for arity in tfm._GRID_SPACE_ARITIES}
    menu = {(base, arity)
            for base in tfm._HARDWARE_BASES
            for arity in range(1, tm._MAX_ARITY_SEARCH + 1)}
    return grid | menu


def _buildable_families():
    """Every family either enumerator yields, deduplicated by (base, arity)."""
    from prismaquant import tessera_menu as tm

    seen = {}
    for spec in list(enumerate_grid_space()) + list(tm.menu_families()):
        seen.setdefault((spec.base, spec.arity), spec)
    assert seen, "an empty family space would let every bound below pass"
    return list(seen.values())


def test_every_family_keyed_memo_is_sized_off_the_family_space():
    """The rule for #148.  Not ``maxsize == <number>`` -- ``== the bound``.

    The pre-fix literals (64/64/256/64/64) all fail the equality: 64 and 256
    are round numbers against a 32-pair space, correct today only because no
    eviction has been measured yet.
    """
    from prismaquant import tessera_formats as tfm

    bound = tfm.family_cache_bound()
    assert bound >= len(_asked_family_pairs()), (
        f"the bound holds {bound} entries against "
        f"{len(_asked_family_pairs())} asked (base, arity) pairs")
    for name, memo in _family_keyed_memos().items():
        assert memo.cache_info().maxsize == bound, (
            f"{name} holds {memo.cache_info().maxsize} entries against a "
            f"{bound}-family space: a round number, not the roster")


def test_a_second_pass_over_the_family_space_recomputes_nothing():
    """The behaviour the bound buys, measured on the memo rather than a clock.

    ``_grid_for`` and ``_tcq_body_is_reachable`` are covered by the sizing
    test above, not here: the first refuses the free bases (a miss on every
    pass by construction) and the second is only ever asked of the
    over-budget families, so neither stores the whole space.
    """
    from prismaquant import tessera_formats as tfm
    from prismaquant import tessera_render as tr

    families = _buildable_families()

    tfm._build_grid.cache_clear()
    tfm.tessera_family.cache_clear()
    tr.family_grid_is_serialisable.cache_clear()
    for spec in families:
        tfm.tessera_family(spec.base, spec.arity)
        tfm._build_grid(spec.base, spec.base_size, spec.arity)
        tr.family_grid_is_serialisable(spec)
    cold = {
        "tessera_family": tfm.tessera_family.cache_info(),
        "_build_grid": tfm._build_grid.cache_info(),
        "family_grid_is_serialisable":
            tr.family_grid_is_serialisable.cache_info(),
    }
    assert all(info.misses == len(families) for info in cold.values())
    for spec in families:
        tfm.tessera_family(spec.base, spec.arity)
        tfm._build_grid(spec.base, spec.base_size, spec.arity)
        tr.family_grid_is_serialisable(spec)
    warm = {
        "tessera_family": tfm.tessera_family.cache_info(),
        "_build_grid": tfm._build_grid.cache_info(),
        "family_grid_is_serialisable":
            tr.family_grid_is_serialisable.cache_info(),
    }
    for name in cold:
        assert warm[name].misses == cold[name].misses, (
            f"{name}: a repeat pass re-answered families it had answered")
        assert warm[name].hits - cold[name].hits == len(families)


def test_the_family_bound_moves_when_the_roster_moves(monkeypatch):
    """Derived, not restated: shrink the roster and the live memos shrink.

    The companion to the rung-space version above.  A literal cannot pass
    this: the bound is read off ``_HARDWARE_BASES`` at sizing time, so moving
    the roster moves every family-keyed memo with it.
    """
    from prismaquant import tessera_formats as tfm

    def _rebuild():
        for memo in _family_keyed_memos().values():
            memo.cache_clear()

    wide = tfm.family_cache_bound()
    for name, memo in _family_keyed_memos().items():
        assert memo.cache_info().maxsize == wide, name

    narrower = {base: spec for base, spec in tfm._HARDWARE_BASES.items()
                if base != "BF16"}
    assert len(narrower) == len(tfm._HARDWARE_BASES) - 1
    try:
        monkeypatch.setattr(tfm, "_HARDWARE_BASES", narrower)
        _rebuild()
        narrow = tfm.family_cache_bound()
        assert narrow < wide
        for name, memo in _family_keyed_memos().items():
            assert memo.cache_info().maxsize == narrow, name
        assert narrow >= len(_asked_family_pairs())
    finally:
        # The memos were BUILT while the roster was short.  Restoring the
        # roster does not resize them, so the clear has to come after the undo
        # or every later test in the session runs undersized.
        monkeypatch.undo()
        _rebuild()

    assert tfm.family_cache_bound() == wide
    for name, memo in _family_keyed_memos().items():
        assert memo.cache_info().maxsize == wide, name
