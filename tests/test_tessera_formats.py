"""The Tessera family seam, pinned against the rungs that were actually measured.

The point of this file is not coverage.  It is that ``tessera_formats`` exists
to let PrismaQuant's allocator price Tessera, and an allocator that addresses
rungs nobody measured is addressing fiction.  So the load-bearing test is the
ladder test: the three arms with published weight-space numbers must be
namable, and their artifact bpp must come out at the figure those measurements
were reported at.
"""
from fractions import Fraction

import pytest

from prismaquant.tessera_formats import (
    SCALE_PLANE_BITS_Q256,
    TESSERA4,
    TESSERA4_K2,
    TESSERA8,
    TESSERA_FAMILIES,
    TesseraFormatError,
    TesseraRateSurface,
    get_tessera_family,
    parse_tessera_format_name,
    validate_body_rate_q256,
)


# (label, family, body q256, artifact bpp as published)
MEASURED_LADDER = [
    ("E2M1 scalar R=3", "TESSERA4", 768, 3.5),
    ("E2M1 k=2 R=7", "TESSERA4K2", 896, 4.0),
    ("E4M3 R=4", "TESSERA8", 1024, 4.5),
    ("E4M3 R=5", "TESSERA8", 1280, 5.5),
]


@pytest.mark.parametrize("label,family,q256,bpp", MEASURED_LADDER)
def test_the_measured_rungs_are_addressable(label, family, q256, bpp):
    spec = get_tessera_family(family)
    assert validate_body_rate_q256(spec, q256) == q256
    artifact_q256 = q256 + SCALE_PLANE_BITS_Q256
    assert artifact_q256 / 256 == pytest.approx(bpp), label


def test_arity_is_what_fills_the_rungs_between():
    """k=2 addresses 4.0 bpp, which the scalar family provably cannot."""
    scalar = get_tessera_family("TESSERA4")
    tup = get_tessera_family("TESSERA4K2")
    with pytest.raises(TesseraFormatError):
        validate_body_rate_q256(scalar, 896)      # above the scalar cap
    assert validate_body_rate_q256(tup, 896) == 896
    assert tup.root_rate(896) == Fraction(7, 2)   # 3.5 body bits per position


def test_rate_cap_closes_the_code_space():
    """|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R) must equal 2^payload_bits."""
    for spec in TESSERA_FAMILIES.values():
        assert spec.rate_cap == spec.payload_bits - 1
        for rate in range(1, spec.rate_cap + 1):
            anchors = 1 << (rate + 1)
            descendants = 1 << (spec.rate_cap - rate)
            assert anchors * descendants == 1 << spec.payload_bits


def test_the_family_reads_its_grid_from_tessera_not_from_here():
    """The grid object is built by tessera; a second copy would drift."""
    for spec in TESSERA_FAMILIES.values():
        grid = spec.payload_grid()
        assert grid.payload_bits == spec.payload_bits
        assert grid.arity == spec.arity
        assert grid.rate_cap == spec.rate_cap


def test_format_names_round_trip_and_foreign_names_are_not_claimed():
    for spec in TESSERA_FAMILIES.values():
        lo, hi = spec.mathematical_q256_bounds
        for q in (lo, hi):
            assert parse_tessera_format_name(spec.format_name(q)) == (spec, q)
    for foreign in ("NVFP4", "FP8_DYNAMIC", "BF16", "TCQ_E2M1_R768", None, 17):
        assert parse_tessera_format_name(foreign) is None


def test_a_tessera_shaped_name_with_an_illegal_rung_raises():
    """Silence here would let a bad rung reach the DP as an unpriced format."""
    with pytest.raises(TesseraFormatError):
        parse_tessera_format_name("TESSERA4_R9999")
    with pytest.raises(TesseraFormatError):
        parse_tessera_format_name("TESSERA4_Rabc")


def test_bounds_are_exact_integers_in_q256():
    for spec in TESSERA_FAMILIES.values():
        lo, hi = spec.mathematical_q256_bounds
        assert isinstance(lo, int) and isinstance(hi, int)
        assert lo * spec.arity == 256
        assert hi * spec.arity == spec.rate_cap * 256


def test_an_adaptive_surface_must_name_its_evidence():
    with pytest.raises(TesseraFormatError, match="unattributed"):
        TesseraRateSurface(family="TESSERA4", mode="adaptive", bounds_q256=(256, 768))
    ok = TesseraRateSurface(
        family="TESSERA4", mode="adaptive", bounds_q256=(256, 768),
        anchor_q256=(512, 768), source_identity_sha256="0" * 64,
    )
    assert "0" * 64 in ok.identity()


def test_a_surface_refuses_rungs_its_family_cannot_address():
    with pytest.raises(TesseraFormatError):
        TesseraRateSurface(family="TESSERA4", mode="dense", bounds_q256=(256, 896))
    with pytest.raises(TesseraFormatError):
        TesseraRateSurface(family="TESSERA4", mode="dense", bounds_q256=(768, 256))


def test_unknown_family_names_the_legal_set():
    with pytest.raises(TesseraFormatError, match="TESSERA4"):
        get_tessera_family("TCQ_E2M1_R256")
