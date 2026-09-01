"""Bytes for a Tessera Linear, and the allocator pricing that rests on them.

The property under test is that PrismaQuant's byte budget, Tessera's exact-byte
accountant and the artifact an exporter would write are **one number**.  So the
tests here check the pricing against the published ladder rather than against
themselves, and check that the whole grid-space prices at the bounds the family
descriptor advertises -- a family that claims [1.00, 4.00] bpp and prices at
[1.50, 7.50] is not a family, it is two disagreeing statements.
"""
import pytest

from prismaquant.tessera_allocator import build_tessera_allocator_candidate
from prismaquant.tessera_footprint import (
    TESSERA_TENSOR_PAYLOAD_SCHEMA,
    tessera_tensor_payload_breakdown,
    validate_tessera_tensor_payload_breakdown,
)
from prismaquant.tessera_formats import (
    TesseraFormatError,
    enumerate_grid_space,
    get_tessera_family,
    realisable_rungs,
)

SHAPE = (4096, 4096)

# (family, body q256, published artifact bpp)
MEASURED = [
    ("TESSERA_E2M1_K1", 768, 3.5),
    ("TESSERA_E2M1_K2", 896, 4.0),
    ("TESSERA_LM16_K2", 896, 4.0),
    ("TESSERA_LM16_K1", 768, 3.5),
    # Sub-cap rungs.  Published as 4.5 and 5.5; the bytes the encoder writes
    # for either are 7.5 bpp, because BODY + COMPLETION is the cap at every
    # rung.  See test_the_rung_does_not_set_the_size.
    ("TESSERA_E4M3_K1", 1024, 7.5),
    ("TESSERA_E4M3_K1", 1280, 7.5),
]


@pytest.mark.parametrize("family,q256,bpp", MEASURED)
def test_the_measured_ladder_prices_at_its_published_bpp(family, q256, bpp):
    spec = get_tessera_family(family)
    out = tessera_tensor_payload_breakdown(SHAPE, family=spec, body_rate_q256=q256)
    assert out["exact_bpw"] == pytest.approx(bpp, abs=1e-9)
    assert out["schema"] == TESSERA_TENSOR_PAYLOAD_SCHEMA
    assert out["wire_schema"] == "prismaquant.tessera.v1"


def test_every_family_prices_at_the_bounds_it_advertises():
    """Catches the arity bug: a k=2 code spans k rows, so the code grid is
    `rows // k`.  Priced without that, k=2 families came out at exactly `k`
    times their real cost -- LM64 k=2 at 11.5 bpp against a 6.00 ceiling."""
    for spec in enumerate_grid_space():
        rungs = realisable_rungs(spec)
        lo_q256, hi_q256 = spec.artifact_q256_bounds
        lo = tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=rungs[0]
        )
        hi = tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=rungs[-1]
        )
        assert lo["exact_bpw"] == pytest.approx(lo_q256 / 256, abs=1e-9), spec.name
        assert hi["exact_bpw"] == pytest.approx(hi_q256 / 256, abs=1e-9), spec.name


def test_bytes_are_invariant_in_the_rung():
    """This asserted bytes were *monotone* in the rung until 2026-09-01, on the
    reasoning that "a rate axis the allocator can search has to be ordered".

    The axis is not a rate axis.  BODY spends ``R`` bits per code and
    COMPLETION spends ``cap - R``, so every rung of a family serialises to the
    identical byte count and the rung buys quality, not size.  The premise was
    wrong rather than the implementation, so the assertion is inverted rather
    than relaxed: a single differing byte here means one of the two planes
    stopped tracking the other."""
    spec = get_tessera_family("TESSERA_E4M3_K1")
    sizes = {
        tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=q
        )["total_bytes"]
        for q in realisable_rungs(spec, step_q256=64)
    }
    assert len(sizes) == 1, sorted(sizes)


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
    alphabets = {r: tuple(range(1 << (r + 1))) for r in set(schedule)}
    return build_tessera_allocator_candidate(
        "layers.0.mlp.gate_proj", SHAPE, family=spec, body_rate_q256=q256,
        layout="tight", schedule=schedule, alphabets=alphabets,
        predicted_dloss=dloss, predicted_dloss_stderr=0.0,
        target_profile="research",
    )


@pytest.mark.parametrize("family,q256,bpp", MEASURED)
def test_the_allocator_prices_a_tessera_rung(family, q256, bpp):
    candidate = _price(get_tessera_family(family), q256)
    assert candidate.family == family
    assert candidate.body_rate_q256 == q256
    assert candidate.n_params == SHAPE[0] * SHAPE[1]
    assert candidate.memory_bytes * 8 == pytest.approx(
        candidate.bits_per_param * candidate.n_params, rel=1e-9
    )
    # Strictly *above* the ladder figure, because `_price` supplies the anchor
    # tables and they are real bytes.  The published ladder quotes body+scale;
    # a priced candidate also carries its alphabets, and pretending otherwise
    # would understate every artifact by the side information it must ship.
    assert candidate.bits_per_param > bpp
    assert candidate.bits_per_param - bpp < 1e-3


def test_the_rung_axis_is_not_a_rate_axis_the_allocator_can_search():
    """This asserted the opposite -- "adjacent rungs differ, and differ in
    order" -- and it was the load-bearing claim behind treating Tessera as a
    continuously-rateable format.  It is false.

    BODY and COMPLETION trade off exactly, so the serialised payload is one
    size per family.  The only thing that still moves between adjacent rungs is
    **anchor-table side information**: a mixed-rate rung ships two alphabet
    tables where a rung sitting exactly on an integer rate ships one.  That
    varies by a few parts in 10^5, is not monotone, and is emphatically not a
    rate the DP can trade against NVFP4.

    Consequence, and the reason this is worth a test rather than a comment: a
    Tessera menu is a menu of *families*, each contributing exactly one size.
    Sweeping q256 inside a family gives the allocator nothing to choose
    between on bytes, so the only non-dominated rung is the family's top one.
    """
    spec = get_tessera_family("TESSERA_E4M3_K1")

    bodies = {
        tessera_tensor_payload_breakdown(
            SHAPE, family=spec, body_rate_q256=q
        )["exact_bpw"]
        for q in (1020, 1021, 1022, 1023, 1024)
    }
    assert len(bodies) == 1, sorted(map(float, bodies))

    # The priced candidate carries its alphabets too, so it is not *exactly*
    # constant -- but the spread is side information, far below a rung's worth
    # of anything, and unordered.
    bpps = [_price(spec, q).bits_per_param for q in (1020, 1021, 1022, 1023, 1024)]
    assert max(bpps) - min(bpps) < 1e-3
    assert bpps != sorted(bpps), "if this ever sorts, check it is not a real rate"


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
def test_the_rung_name_is_not_a_rate(q256):
    """The R-number is a *quality* setting, and the registry must not read it
    as a size.

    This test asserted the opposite until 2026-09-01 -- that the price is
    ``(q256 + 128)/256`` -- and it passed, because every artifact that had ever
    been built sat at a family's top rung, which is the single rung where that
    formula is right.  Off the top rung it underprices: R128 quoted 1.0 bpp
    against 4.0 bpp of bytes.

    What the wire actually does is spend ``cap`` bits per code at every rung,
    split between BODY (``R``, chosen by the trellis's joint search) and
    COMPLETION (``cap - R``, chosen greedily per position).  Raising R moves
    bits from the greedy plane to the searched one, which is why error falls
    with R while size does not move at all.
    """
    from prismaquant import format_registry as fr

    spec = fr.get_format(f"TESSERA_E2M1_K2_R{q256}")
    assert spec.effective_bits_for_shape((4096, 1536)) == 4.0


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
