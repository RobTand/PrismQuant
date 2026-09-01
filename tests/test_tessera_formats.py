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
    get_tessera_family,
    parse_tessera_format_name,
    realisable_rungs,
    tessera_family,
    validate_body_rate_q256,
)

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
    on it: a published figure belongs to the wire it was measured on."""
    spec = get_tessera_family(family)
    assert validate_body_rate_q256(spec, q256) == q256
    assert artifact_bpp(spec, q256, span=1, scale_plane="s6b") == Fraction(int(bpp * 256), 256), label


def test_the_default_wire_keeps_k2_at_four_and_lifts_arity_one_by_a_quarter():
    """The exporter's wire since 2026-09-01 (span 2 over a LUT plane): the
    stored labels cost (L-1)/L bits per CODE, the LUT plane saves a quarter-bit
    per position, so at arity 2 the two cancel and at arity 1 they do not."""
    assert artifact_bpp("TESSERA_E2M1_K2", 896) == Fraction(4)
    assert artifact_bpp("TESSERA_E2M1_K1", 768) == Fraction(15, 4)
    assert artifact_bpp("TESSERA_E4M3_K1", 1024) == Fraction(19, 4)
    assert artifact_bpp("TESSERA_E2M1_K2", 896) == artifact_bpp(
        "TESSERA_E2M1_K2", 896, span=1, scale_plane="s6b")


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
    assert artifact_bpp(tup, 896) == Fraction(4)      # 3.5 body + 0.5 scale


def test_rate_cap_closes_the_code_space():
    """|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R) must equal 2^payload_bits."""
    for spec in enumerate_grid_space():
        assert spec.rate_cap == spec.payload_bits - 1
        for rate in range(1, spec.rate_cap + 1):
            assert (1 << (rate + 1)) * (1 << (spec.rate_cap - rate)) == (
                1 << spec.payload_bits
            )


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
        assert grid.rate_cap == spec.rate_cap


def test_a_code_space_at_the_encoder_wall_is_refused():
    """E4M3 k=2 is 65536 anchors per step -- what got k=4 refused."""
    with pytest.raises(TesseraFormatError, match="wall the encoder already refuses"):
        tessera_family("E4M3", 2)
    with pytest.raises(TesseraFormatError):
        tessera_family("E2M1", 4)
    assert all(f.payload_bits < ANCHOR_BUDGET_BITS for f in enumerate_grid_space())


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

    This checked only the top for a stretch, because the artifact bounds had
    been collapsed to a point at the cap on the reading that COMPLETION
    backfills whatever BODY does not spend.  Checking both ends is what makes
    it a statement about the scale plane rather than about the rate span: if
    the two ends ever needed different offsets, one of them would not be the
    scale plane.
    """
    assert SCALE_PLANE_BITS_Q256 == 128
    for spec in enumerate_grid_space():
        body_lo, body_hi = spec.mathematical_q256_bounds
        art_lo, art_hi = spec.artifact_q256_bounds
        extra = wire_overhead_q256(spec)
        assert art_lo - body_lo == extra, spec.name
        assert art_hi - body_hi == extra, spec.name
        assert art_hi > art_lo, (spec.name, "a family advertises an interval")
    # The overhead is the exporter's wire and nothing else: span-1 S6b is the
    # historical half-bit; the shipping span-2 LUT wire is a quarter-bit of
    # plane plus half a bit per CODE of stored labels.
    for spec in enumerate_grid_space():
        assert wire_overhead_q256(spec, 1, "s6b") == SCALE_PLANE_BITS_Q256
        assert wire_overhead_q256(spec, 2, "lut16") == (
            SCALE_LUT_BITS_Q256 + Fraction(128, spec.arity)
        )


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
        rungs = realisable_rungs(spec)
        for q in (rungs[0], rungs[len(rungs) // 2], rungs[-1]):
            # ``terminal_rate`` takes q256 per *code* -- it calls
            # ``root_from_q256`` directly -- while a rung is quoted per
            # position, so the arity factor has to be applied by the caller.
            for depth in (0, spec.rate_cap):
                span, plane = tessera_wire_defaults()
                exact = terminal_rate(
                    q * spec.arity, 4096, 4096, with_scale_refine=True,
                    with_scale_base=plane == "s6b", span=span,
                    cap=spec.rate_cap, arity=spec.arity, completion=depth,
                )
                assert artifact_bpp(spec, q, depth) == exact, (spec.name, q, depth)
                checked += 1
            assert artifact_bpp(spec, q, None) == artifact_bpp(
                spec, q, spec.rate_cap)
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
    for spec in enumerate_grid_space():
        rungs = realisable_rungs(spec)
        overhead = wire_overhead_q256(spec) / 256
        ceiling = Fraction(spec.rate_cap * 256, spec.arity) / 256 + overhead

        priced = [artifact_bpp(spec, q) for q in rungs]
        assert priced == sorted(priced), spec.name
        assert len(set(priced)) == len(rungs), (spec.name, "flat ladder")
        assert priced[-1] == ceiling
        for rung, bpp in zip(rungs, priced):
            assert bpp == Fraction(rung, 256) + overhead

        at_full = {artifact_bpp(spec, q, None) for q in rungs}
        assert at_full == {ceiling}, (spec.name, sorted(map(float, at_full)))


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
    takes hardware base grids only, and all three hardware-derived grids that
    fit in a one-byte plane entry are now registered.  A free Lloyd-Max grid is
    unwritable for a reason no registry line can fix -- its values are fitted
    to the tensor, so no reader rebuilds them from a name -- but it does not
    render either, so it cannot demonstrate the gap.

    The gap is therefore asserted through the mechanism rather than through an
    example: remove a grid's digest from the wire commitment and the predicate
    must go False *while the render still succeeds*.  That is the case the
    predicate exists for, and it will recur the moment a fourth grid renders.
    """
    from unittest import mock

    import tessera.alphabet as alphabet
    from prismaquant.tessera_render import (
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
        # The predicate is lru_cached -- it is asked once per rung per run on a
        # menu of thousands -- so a test that patches the registry underneath
        # it must clear the cache or it reads its own earlier answer.
        tessera_rung_is_serialisable.cache_clear()
        # Still renders -- if it stopped, this would pass for the wrong reason.
        assert render_tessera_weight(
            weight, "TESSERA_E4M3_K1_R1024").shape == (64, 512)
        assert tessera_rung_is_serialisable("TESSERA_E4M3_K1_R1024") is False
    tessera_rung_is_serialisable.cache_clear()


def test_flipping_the_serving_lane_cannot_admit_an_unwritable_rung():
    """``producer_eligible`` is the AND of "the wire can carry it" and "a
    runtime serves it".  The second is one module-level flag, so the whole
    point of the AND is that flipping it -- the deliberate act that follows an
    attested route -- must not quietly put a rung on the menu that the exporter
    will refuse."""
    from unittest import mock

    from prismaquant import tessera_render as tr

    with mock.patch.object(tr, "_TESSERA_SERVING_LANE_EXISTS", True):
        assert tr.synthesize_tessera_spec("TESSERA_E2M1_K2_R896").producer_eligible
        assert tr.synthesize_tessera_spec(
            "TESSERA_E4M3_K1_R1024"
        ).producer_eligible
        assert not tr.synthesize_tessera_spec(
            "TESSERA_LM16_K1_R768"
        ).producer_eligible


def test_no_tessera_rung_is_producer_eligible_today():
    """Principle 9: nothing serves these bytes yet, so registry membership must
    not be mistaken for a serving route."""
    from prismaquant.tessera_render import synthesize_tessera_spec

    for name in ("TESSERA_E2M1_K2_R896", "TESSERA_E2M1_K1_R640"):
        assert synthesize_tessera_spec(name).producer_eligible is False
