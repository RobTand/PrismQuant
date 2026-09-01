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

import pytest

from prismaquant.tessera_formats import (
    ANCHOR_BUDGET_BITS,
    LANE_KERNEL,
    LANE_STOCK,
    SCALE_PLANE_BITS_Q256,
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

# (label, family name, body q256, artifact bpp as published)
MEASURED_LADDER = [
    ("E2M1 scalar R=3", "TESSERA_E2M1_K1", 768, 3.5),
    ("E2M1 k=2 R=7", "TESSERA_E2M1_K2", 896, 4.0),
    ("free-16 k=2 R=7", "TESSERA_LM16_K2", 896, 4.0),
    ("free-16 scalar R=3", "TESSERA_LM16_K1", 768, 3.5),
    ("E4M3 R=4", "TESSERA_E4M3_K1", 1024, 4.5),
    ("E4M3 R=5", "TESSERA_E4M3_K1", 1280, 5.5),
]


@pytest.mark.parametrize("label,family,q256,bpp", MEASURED_LADDER)
def test_the_measured_rungs_price_at_their_published_bpp(label, family, q256, bpp):
    spec = get_tessera_family(family)
    assert validate_body_rate_q256(spec, q256) == q256
    assert artifact_bpp(spec, q256) == Fraction(int(bpp * 256), 256), label


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
    assert SCALE_PLANE_BITS_Q256 == 128
    for spec in enumerate_grid_space():
        body_lo, _ = spec.mathematical_q256_bounds
        art_lo, _ = spec.artifact_q256_bounds
        assert art_lo - body_lo == SCALE_PLANE_BITS_Q256


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

    `artifact_bpp` is a closed form -- body plus a flat half-bit -- because the
    allocator prices thousands of rungs per Linear and cannot afford to build a
    plane layout for each.  That makes it a *second* statement of something
    tessera already computes exactly, and a second statement is a drift bug
    waiting to happen.  So it is pinned here against the authority.
    """
    from tessera.calculator import terminal_rate

    checked = 0
    for spec in enumerate_grid_space():
        if spec.arity != 1:
            continue  # the accountant's schedule is quoted per code, k=1 only
        rungs = realisable_rungs(spec)
        for q in (rungs[0], rungs[len(rungs) // 2], rungs[-1]):
            exact = terminal_rate(
                q, 4096, 4096, with_scale_refine=True, cap=spec.rate_cap
            )
            assert artifact_bpp(spec, q) == exact, (spec.name, q)
            checked += 1
    assert checked >= 12
