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
    ("TESSERA_E4M3_K1", 1024, 4.5),
    ("TESSERA_E4M3_K1", 1280, 5.5),
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


def test_bytes_are_monotone_in_the_rung():
    """A rate axis the allocator can search has to be ordered."""
    spec = get_tessera_family("TESSERA_E4M3_K1")
    previous = None
    for q in realisable_rungs(spec, step_q256=64):
        out = tessera_tensor_payload_breakdown(SHAPE, family=spec, body_rate_q256=q)
        if previous is not None:
            assert out["total_bytes"] > previous
        previous = out["total_bytes"]


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


def test_the_allocator_sees_a_continuous_axis_not_a_few_rungs():
    """The whole point: adjacent rungs differ, and differ in order."""
    spec = get_tessera_family("TESSERA_E4M3_K1")
    priced = [_price(spec, q) for q in (1020, 1021, 1022, 1023, 1024)]
    bpps = [c.bits_per_param for c in priced]
    assert bpps == sorted(bpps)
    assert len(set(bpps)) == len(bpps), "adjacent q256 rungs must be distinguishable"
    # One q256 apart is 1/256 of a bit of *body*; the total moves by slightly
    # less across this window because 1024 needs one anchor table where the
    # mixed-rate rungs below it need two.  The body step is exact:
    bodies = [
        tessera_tensor_payload_breakdown(SHAPE, family=spec, body_rate_q256=q)
        for q in (1020, 1024)
    ]
    assert bodies[1]["exact_bpw"] - bodies[0]["exact_bpw"] == pytest.approx(
        4 / 256, abs=1e-9
    )


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
def test_the_registry_prices_the_rate_the_rung_name_states(q256):
    """``artifact_bpp`` is ``(q256 + scale_plane)/256``; the price must be it.

    A rung's R-number is its per-position body rate, and the scale planes add a
    fixed 128/256.  If the registry's number is not exactly that, then the name
    of the format and the cost of the format are different facts.
    """
    from prismaquant import format_registry as fr

    spec = fr.get_format(f"TESSERA_E2M1_K2_R{q256}")
    assert spec.effective_bits_for_shape((4096, 1536)) == (q256 + 128) / 256


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
