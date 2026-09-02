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
    family_rate_cap,
    get_tessera_family,
    parse_tessera_format_name,
    realisable_rungs,
    recipe_is_shape_free,
    scale_plane_name,
    tessera_family,
    tessera_wire_recipe,
    validate_body_rate_q256,
)
from tessera.export import TCQ_RECIPE
from tessera.manifest import BodyKind


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
    on it: a published figure belongs to the wire it was measured on."""
    spec = get_tessera_family(family)
    assert validate_body_rate_q256(spec, q256) == q256
    assert artifact_bpp(spec, q256, span=1, scale_plane="s6b") == Fraction(int(bpp * 256), 256), label


def test_the_default_wire_keeps_k2_at_four_and_lifts_arity_one_by_a_quarter():
    """The coset-trellis wire (span 2 over a LUT plane): the stored labels cost
    (L-1)/L bits per CODE, the LUT plane saves a quarter-bit per position, so at
    arity 2 the two cancel and at arity 1 they do not.

    E2M1x2 keeps that wire **at** the trellis's cap, which is the rung 4.0 bpp
    is addressed at, so the headline number is unmoved by the 2026-09-02 flip.
    """
    assert artifact_bpp("TESSERA_E2M1_K2", 896) == Fraction(4)
    assert artifact_bpp("TESSERA_E2M1_K1", 768) == Fraction(15, 4)
    assert artifact_bpp("TESSERA_E2M1_K2", 896) == artifact_bpp(
        "TESSERA_E2M1_K2", 896, span=1, scale_plane="s6b")


def test_the_e4m3_rung_has_a_size_but_no_rate():
    """E4M3's wire is the window body over the CHANNEL plane at every rung.

    Both of its planes are charged per *unit* -- one fp16 per output row, one
    ``2^14``-byte table -- so the rung has no bits-per-parameter rate to quote
    and the shape-free accountant refuses instead of quoting a floor.  Named a
    shape, it is exact, and the two entry points that answer it (the closed
    form here, the ``FormatSpec`` the registry synthesizes) must agree.
    """
    from prismaquant import format_registry as fr

    assert not recipe_is_shape_free(tessera_wire_recipe("TESSERA_E4M3_K1", 1024))
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
    assert artifact_bpp(tup, 896) == Fraction(4)      # 3.5 body + 0.5 scale


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
    checked = 0
    for spec in enumerate_grid_space():
        if not recipe_is_shape_free(spec.recipe):
            # A per-unit wire has no per-position overhead to compare against;
            # its offset is a function of the shape and is checked there.
            continue
        body_lo, body_hi = spec.mathematical_q256_bounds
        art_lo, art_hi = spec.artifact_q256_bounds
        extra = wire_overhead_q256(spec)
        assert art_lo - body_lo == extra, spec.name
        assert art_hi - body_hi == extra, spec.name
        assert art_hi > art_lo, (spec.name, "a family advertises an interval")
        checked += 1
    assert checked, "no family left with a per-position wire to check"
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
                    q * spec.arity, 4096, 4096, with_scale_refine=True,
                    with_scale_base=plane == "s6b", span=wire.span,
                    cap=cap, arity=spec.arity, completion=depth,
                )
                assert artifact_bpp(spec, q, depth) == exact, (spec.name, q, depth)
                checked += 1
            assert artifact_bpp(spec, q, None) == artifact_bpp(spec, q, cap)
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
        overhead = wire_overhead_q256(spec, recipe=wire) / 256
        ceiling = Fraction(cap * 256, spec.arity) / 256 + overhead

        priced = [artifact_bpp(spec, q, recipe=wire) for q in rungs]
        assert priced == sorted(priced), spec.name
        assert len(set(priced)) == len(rungs), (spec.name, "flat ladder")
        for rung, bpp in zip(rungs, priced):
            assert bpp == Fraction(rung, 256) + overhead
        if rungs[-1] * spec.arity == cap * 256:
            assert priced[-1] == ceiling

        at_full = {artifact_bpp(spec, q, None, recipe=wire) for q in rungs}
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
    """Principle 9: the pinned serving release (Gridbook 0.9.1, contract v12)
    publishes no Tessera row, so registry membership must not be mistaken for
    a serving route.  The day a pin lands on a release whose contract carries
    the ``TESSERA_E2M1_K2`` row (v13 is the first), this test's premise ends
    and the eligible set is whatever that contract's cells name."""
    from prismaquant import tessera_render as tr
    from prismaquant.tessera_render import synthesize_tessera_spec

    table, _formats = tr._pinned_serving_table()
    assert not table.governs("TESSERA_E2M1_K2")
    for name in ("TESSERA_E2M1_K2_R896", "TESSERA_E2M1_K1_R640"):
        assert tr.tessera_lane_attested(name) is False
        assert synthesize_tessera_spec(name).producer_eligible is False


def _v13_shaped_contract(tmp_path, *, qualification="device_qualified",
                         route_status="backed_with_serve_flag", rungs=(896,)):
    """The v12 fixture plus the Tessera row and cells Gridbook contract v13
    publishes -- a FIXTURE for the lookup's logic, never an attestation: the
    real cells reach this repository only through a re-pin."""
    import json

    from test_cb_route_status_gate import V12_FIXTURE, _cell

    contract = json.loads(V12_FIXTURE.read_text())
    contract["formats"].append({
        "kind": "tcq_trellis", "family": "TESSERA_E2M1_K2",
        "name_pattern": "TESSERA_E2M1_K2_R{k}",
        "candidate_rungs_q256": [896], "reader_rate_range_q256": [896, 896],
        "native_terminal_q256": 1024, "residency_modes": ["resident", "streamed"],
    })
    flags = ["GRIDBOOK_TESSERA_NVFP4=1", "GRIDBOOK_TESSERA_NVFP4_MODE=resident|streamed"]
    for regime in ("decode", "batch"):
        contract["lane_eligibility"]["cells"].append(_cell(
            f"tessera_e2m1_k2_dense_sm121_{regime}", platform="sm_121",
            family="TESSERA_E2M1_K2", structure="dense", regime=regime,
            route_status=route_status, qualification=qualification,
            rungs_q256=list(rungs), activation_contract="e2m1_group16_ue4m3_static",
            flags=flags if route_status == "backed_with_serve_flag" else ()))
    path = tmp_path / "v13.json"
    path.write_text(json.dumps(contract))
    return path


def _v14_shaped_contract(tmp_path):
    """The v13 fixture plus what Gridbook contract v14 adds: the
    ``TESSERA_E4M3_K1`` row (the FP8 route of the same lane), its two cells,
    and the lane's unified flag pair on every Tessera cell.  A FIXTURE for
    the lookup's logic, never an attestation."""
    import json

    from test_cb_route_status_gate import _cell

    path = _v13_shaped_contract(tmp_path)
    contract = json.loads(path.read_text())
    flags = ["GRIDBOOK_TESSERA=1", "GRIDBOOK_TESSERA_MODE=resident|streamed"]
    for cell in contract["lane_eligibility"]["cells"]:
        if str(cell.get("family", "")).startswith("TESSERA_"):
            cell["requires_serve_flags"] = list(flags)
    contract["formats"].append({
        "kind": "tcq_trellis", "family": "TESSERA_E4M3_K1",
        "name_pattern": "TESSERA_E4M3_K1_R{k}",
        "candidate_rungs_q256": [1024], "reader_rate_range_q256": [1024, 1024],
        "native_terminal_q256": 2048, "residency_modes": ["resident", "streamed"],
    })
    for regime in ("decode", "batch"):
        contract["lane_eligibility"]["cells"].append(_cell(
            f"tessera_e4m3_k1_dense_sm121_{regime}", platform="sm_121",
            family="TESSERA_E4M3_K1", structure="dense", regime=regime,
            route_status="backed_with_serve_flag", qualification="device_qualified",
            rungs_q256=[1024], activation_contract="fp8_per_token_dynamic", flags=flags))
    out = tmp_path / "v14.json"
    out.write_text(json.dumps(contract))
    return out


def test_the_fp8_family_is_admitted_by_a_v14_shaped_contract_and_no_earlier(tmp_path):
    """Contract v14 is the first to carry ``TESSERA_E4M3_K1`` (the FP8 W8A8
    route of the same lane, receipted at q1024 only).  The lookup admits
    exactly that rung from such a table, still refuses it from a v13-shaped
    one, and the pinned release (0.9.1 / v12) admits neither family."""
    from prismaquant import tessera_render as tr
    from prismaquant.gridbook_lane_eligibility import (
        load_eligibility_table, load_published_formats,
    )

    def _load(path):
        return (load_eligibility_table("0.9.1-test", contract_path=path),
                load_published_formats(contract_path=path))

    (tmp_path / "v13").mkdir()
    t13, f13 = _load(_v13_shaped_contract(tmp_path / "v13"))
    assert not tr.tessera_lane_attested("TESSERA_E4M3_K1_R1024", table=t13, formats=f13)

    table, formats = _load(_v14_shaped_contract(tmp_path))
    assert table.governs("TESSERA_E4M3_K1") and table.governs("TESSERA_E2M1_K2")
    assert tr.tessera_lane_attested("TESSERA_E4M3_K1_R1024", table=table, formats=formats)
    assert tr.tessera_lane_attested("TESSERA_E2M1_K2_R896", table=table, formats=formats)
    # rates the FP8 receipt did not cover, and the grid's cap itself
    for name in ("TESSERA_E4M3_K1_R896", "TESSERA_E4M3_K1_R2048"):
        assert not tr.tessera_lane_attested(name, table=table, formats=formats)
    assert tr.tessera_lane_attested("TESSERA_E4M3_K1_R1024") is False   # the pinned release


def test_tessera_lane_attested_is_read_from_the_contracts_own_cells(tmp_path):
    """One lookup, failing closed on every axis: the family must be published,
    a cell must name the rate, the cell must be device-qualified and its
    route native.  Nothing in this repository can widen it."""
    from unittest import mock

    from prismaquant import tessera_render as tr
    from prismaquant.gridbook_lane_eligibility import (
        load_eligibility_table, load_published_formats,
    )

    def _load(path):
        return (load_eligibility_table("0.9.1-test", contract_path=path),
                load_published_formats(contract_path=path))

    table, formats = _load(_v13_shaped_contract(tmp_path))
    assert table.governs("TESSERA_E2M1_K2")
    assert tr.tessera_lane_attested("TESSERA_E2M1_K2_R896", table=table, formats=formats)
    # a serialisable rate no cell names
    assert not tr.tessera_lane_attested("TESSERA_E2M1_K2_R640", table=table, formats=formats)
    # a family the contract does not publish
    assert not tr.tessera_lane_attested("TESSERA_E2M1_K1_R640", table=table, formats=formats)
    assert not tr.tessera_lane_attested("TESSERA_E4M3_K1_R1024", table=table, formats=formats)
    # a cell that only cross-compiled, and one whose route is a fallback
    (tmp_path / "c").mkdir()
    t2, f2 = _load(_v13_shaped_contract(tmp_path / "c", qualification="compile_only"))
    assert not tr.tessera_lane_attested("TESSERA_E2M1_K2_R896", table=t2, formats=f2)
    (tmp_path / "f").mkdir()
    t3, f3 = _load(_v13_shaped_contract(tmp_path / "f", route_status="fallback"))
    assert not tr.tessera_lane_attested("TESSERA_E2M1_K2_R896", table=t3, formats=f3)
    # and the spec reads the same lookup: the menu admits the attested rung only
    with mock.patch.object(tr, "_pinned_serving_table", lambda: (table, formats)):
        assert tr.synthesize_tessera_spec("TESSERA_E2M1_K2_R896").producer_eligible
        assert not tr.synthesize_tessera_spec("TESSERA_E2M1_K2_R640").producer_eligible
        assert not tr.synthesize_tessera_spec("TESSERA_E4M3_K1_R1024").producer_eligible


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

    Four terms have to line up, and each of them has been wrong somewhere at
    some point: the body's rate at the *recipe's* cap (``payload_bits`` under a
    window body, ``payload_bits - 1`` under the TCQ trellis), the span labels,
    the scale plane, and the two **per-unit** costs -- a CHANNEL plane's fp16
    per output row, and a window body's ``2^L``-byte table -- which have no
    per-position rate at all until a shape is named.
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
                )
                assert mine == exact, (label, name, q, shape, mine, exact)
                assert isinstance(mine, Fraction)
                checked += 1
    assert checked == len(_recipes_under_test()) * len(TCQ_FAMILIES) * 3


def test_a_per_unit_wire_term_has_no_price_without_a_shape():
    """The CHANNEL row field and the window table are charged per *unit*.

    They are ``16/columns`` and ``8 * 2^L / (rows*columns)`` bits per position,
    which is to say they are not bits per position at all until a shape is
    named.  The shape-free accountant refuses them rather than dropping them:
    the only consumer of the shape-free value is
    ``FormatSpec.exact_bits_per_param``, which ``memory_bytes_for_shape``
    multiplies by the parameter count as an *exact* rate, and a ``Fraction``
    cannot carry "this is a floor".  Returning the floor there would be the
    silent drop, not the guard against it.

    Which of today's wires this bites is a per-(grid, rung) fact, so the test
    asks rather than assumes: a rung on the coset trellis over a block plane
    still prices with no shape at all, and one on the window body or the
    CHANNEL plane refuses.  Both halves are checked on the live recipes.
    """
    from prismaquant.tessera_formats import (
        recipe_from_wire_names, recipe_is_shape_free, tessera_wire_recipe,
        wire_overhead_q256,
    )

    free = per_unit = 0
    for name in TCQ_FAMILIES:
        spec = get_tessera_family(name)
        for q in (spec.mathematical_q256_bounds[0], 1024):
            try:
                validate_body_rate_q256(spec, q)
            except TesseraFormatError:
                continue
            wire = tessera_wire_recipe(spec, q)
            if recipe_is_shape_free(wire):
                assert artifact_bpp(spec, q) > 0
                free += 1
            else:
                with pytest.raises(TesseraFormatError, match="per unit"):
                    artifact_bpp(spec, q)
                assert artifact_bpp(spec, q, shape=(2048, 4096)) > 0
                per_unit += 1
    assert free and per_unit, (free, per_unit)

    spec = get_tessera_family("TESSERA_E4M3_K1")
    for wire in (
        recipe_from_wire_names(2, "channel"),
        recipe_from_wire_names(1, "lut16", "window", 8),
    ):
        assert not recipe_is_shape_free(wire)
        with pytest.raises(TesseraFormatError, match="per unit"):
            wire_overhead_q256(spec, recipe=wire)
        with pytest.raises(TesseraFormatError, match="per unit"):
            artifact_bpp(spec, 1024, 0, recipe=wire)
        priced = artifact_bpp(spec, 1024, 0, recipe=wire, shape=(2048, 4096))
        assert priced > Fraction(4)

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
