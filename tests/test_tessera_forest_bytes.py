"""A TCQ body's forest is on the wire, so it is in the price (issue #126).

``artifact_bpp`` priced the E2M1x2 rung at the coset cap as exactly 4.0 bpp on
every shape, and the exporter wrote 512 bytes more than that on every shape.
The missing term is the TCQ body's **forest**: the ALPHABET plane (the anchor
codes, ``2^(R+1)`` per distinct rate in the schedule) and the DESCENDANT plane
(``2^(cap+1)`` per distinct rate).  Both are written per *unit*, so the error
is a fixed 512 B at the E2M1x2 cap -- 0.1333 bpp on a 96x320 expert unit and
0.0013 bpp on a 1024x3072 dense one -- and it is largest exactly where an MoE
plan spends most of its units.  On arity-1 E2M1 the same hole is 20-44 B, and
it has a second mechanism the issue does not report: a schedule that spans two
distinct rates carries **two** forests, so R511 outweighs the uniform R512
above it.

Three numbers have to be one number, and this is where that is asserted:

* ``prismaquant.tessera_formats.artifact_bpp(...) * n_params / 8`` -- what the
  DP spends its byte budget in;
* ``tessera.control.unit_wire_bits(...) / 8`` -- Tessera's own exact per-unit
  figure, which since tessera ``889fc3a`` charges the forest through
  ``tessera.grammar.forest_plane_bytes``;
* ``tessera.export.encode_linear(...).exact_bytes`` -- the bytes that ship.

The first two are arithmetic and are checked at every shape in the issue's
table.  The third is an encode, so it is checked at the four shapes whose
encodes are seconds; the two large ones are :func:`test_the_exporter_agrees_at_
the_large_shapes_too`, behind ``PRISMAQUANT_TESSERA_SLOW_ENCODE`` because a
3072-column encode is minutes on CPU and the identity it adds is the same one.
Measured green on all six on 2026-09-03; the run is in the issue thread.

CPU only, and the seed is irrelevant: ``exact_bytes`` is a property of the
layout, not of the weights.
"""

import os
from fractions import Fraction

import pytest
from tessera.errors import GrammarError
import torch

from prismaquant.tessera_footprint import tessera_tensor_payload_breakdown
from prismaquant.tessera_formats import (
    TesseraFormatError,
    artifact_bpp,
    family_rate_cap,
    get_tessera_family,
    tessera_wire_recipe,
)

from tessera.control import grid_for_name, unit_wire_bits
from tessera.export import encode_linear
from tessera.grammar import forest_plane_bytes
from tessera.manifest import BodyKind

#: The six shapes the issue tabulates: four MoE-expert-sized units where the
#: fixed 512 B is 0.05-0.13 bpp, and two dense ones where it rounds away.
ISSUE_SHAPES = ((96, 768), (64, 512), (96, 320), (64, 640), (512, 2048), (1024, 3072))
SMALL_SHAPES = ISSUE_SHAPES[:4]
LARGE_SHAPES = ISSUE_SHAPES[4:]

#: ``(family, tessera grid name, rungs)``.  R895/R896 straddle the E2M1x2
#: recipe change -- a window body with a 4096-byte table, then the coset
#: trellis with a 512-byte forest -- and R511/R512 are the arity-1 pair where
#: the two-forest mechanism lives.
FAMILIES = (
    ("TESSERA_E2M1_K2", "E2M1x2", (895, 896)),
    ("TESSERA_E2M1_K1", "E2M1", (511, 512)),
)

_SLOW = os.environ.get("PRISMAQUANT_TESSERA_SLOW_ENCODE") == "1"


def _cases(shapes):
    """Every (family, grid, rung, shape) the *grammar* admits at that shape.

    A rung is not universally legal: the Bresenham quota has to close over the
    unit's columns, so R895 is unrealisable over 320 and R511 over 320 or 640.
    Filtering here rather than hardcoding a list is what keeps this table a
    statement about bytes instead of a statement about which rungs exist.
    """
    out = []
    for family, grid, rungs in FAMILIES:
        spec = get_tessera_family(family)
        for shape in shapes:
            for rung in rungs:
                wire = tessera_wire_recipe(family, rung)
                try:
                    spec.column_schedule(rung, shape[1], recipe=wire)
                except Exception:
                    continue
                out.append(pytest.param(family, grid, rung, shape,
                                        id=f"{grid}-R{rung}-{shape[0]}x{shape[1]}"))
    return out


def _priced_bytes(family, rung, shape):
    rows, columns = shape
    bpp = artifact_bpp(family, rung, shape=shape)
    priced = bpp * rows * columns
    assert priced.denominator == 1, (family, rung, shape, bpp)
    assert priced % 8 == 0, "a plane region is a whole number of bytes"
    return priced // 8


@pytest.mark.parametrize("family,grid,rung,shape", _cases(ISSUE_SHAPES))
def test_the_accountant_and_the_wire_are_one_number(family, grid, rung, shape):
    """``artifact_bpp`` must equal ``unit_wire_bits`` exactly, at every shape.

    No tolerance: both are exact ``Fraction`` arithmetic over integer plane
    extents, and the byte budget is spent in this currency.  This is the leg
    that covers all six shapes, because it costs nothing to evaluate.
    """
    rows, columns = shape
    assert _priced_bytes(family, rung, shape) == unit_wire_bits(
        grid, rung, rows, columns
    ) / 8


@pytest.mark.parametrize("family,grid,rung,shape", _cases(SMALL_SHAPES))
def test_the_accountant_prices_what_the_exporter_writes(family, grid, rung, shape):
    """...and both must equal the bytes ``encode_linear`` serializes.

    ``tessera_tensor_payload_breakdown`` is checked on the same line, because
    it is the *other* PrismaQuant accountant -- the one the allocator's byte
    path reads -- and it was handing ``build_planes`` an empty descendant blob,
    so it charged zero descendant bytes on every rung.  It refuses a column
    count that is not a whole number of 256-column superblocks, which two of
    the issue's shapes are not, so it is asserted where it is defined.
    """
    rows, columns = shape
    priced = _priced_bytes(family, rung, shape)
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(rows, columns), grid=grid_for_name(grid), q256=rung
    ).exact_bytes
    assert priced == exported, (family, rung, shape)
    if columns % 256 == 0:
        breakdown = tessera_tensor_payload_breakdown(
            shape, family=family, body_rate_q256=rung
        )
        assert breakdown["total_bytes"] == exported, (family, rung, shape)


@pytest.mark.slow
@pytest.mark.skipif(not _SLOW, reason="PRISMAQUANT_TESSERA_SLOW_ENCODE!=1")
@pytest.mark.parametrize("family,grid,rung,shape", _cases(LARGE_SHAPES))
def test_the_exporter_agrees_at_the_large_shapes_too(family, grid, rung, shape):
    """The same identity where the 512 B rounds to 0.001 bpp.

    Worth stating once even though it cannot fail if the small shapes pass:
    the term is per unit, so a large shape is the case where an accountant
    that dropped it would look right.
    """
    rows, columns = shape
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(rows, columns), grid=grid_for_name(grid), q256=rung
    ).exact_bytes
    assert _priced_bytes(family, rung, shape) == exported


def test_the_hole_the_issue_measured_is_the_forest_and_nothing_else():
    """512 B at the E2M1x2 cap, 24 B at R512, 44 B at R511 -- from the grammar.

    The under-report the issue tabulated is exactly
    ``forest_plane_bytes(schedule, cap)`` summed, so this pins the *size* of
    the correction against Tessera's own statement of it rather than against a
    number copied out of the issue.  R511's schedule spans rates 1 and 2 and
    therefore carries two forests, which is why it is the larger of the two
    arity-1 corrections and why it outweighs the uniform rung above it.
    """
    spec = get_tessera_family("TESSERA_E2M1_K2")
    wire = tessera_wire_recipe("TESSERA_E2M1_K2", 896)
    assert BodyKind(wire.body) is BodyKind.TCQ
    # A k=2 code has 8 payload bits, so its TCQ cap is 7 and the coset rung
    # schedules every column at 7: one 256-code alphabet, one 256-byte
    # descendant table.  The cap is the *recipe's*, never a subtraction.
    cap = family_rate_cap(spec, wire)
    rates = spec.column_schedule(896, 768, recipe=wire)
    assert (cap, set(rates)) == (7, {7})
    assert sum(forest_plane_bytes(rates, cap)) == 512

    k1 = get_tessera_family("TESSERA_E2M1_K1")
    assert family_rate_cap(k1) == 3
    schedule_511 = k1.column_schedule(511, 512, recipe=tessera_wire_recipe(
        "TESSERA_E2M1_K1", 511))
    schedule_512 = k1.column_schedule(512, 512, recipe=tessera_wire_recipe(
        "TESSERA_E2M1_K1", 512))
    assert set(schedule_511) == {1, 2} and set(schedule_512) == {2}
    assert sum(forest_plane_bytes(schedule_511, 3)) == 44
    assert sum(forest_plane_bytes(schedule_512, 3)) == 24
    # ...and the two-forest rung really does outweigh the uniform one above it.
    assert _priced_bytes("TESSERA_E2M1_K1", 511, (64, 512)) > _priced_bytes(
        "TESSERA_E2M1_K1", 512, (64, 512))


def test_no_tessera_rung_has_a_rate_without_a_shape():
    """The design consequence, stated so it cannot be quietly undone.

    Every wire Tessera writes now charges something per *unit*: a TCQ body its
    forest, a WINDOW body its table, a CHANNEL plane its row field.  So there
    is no shape-free bits-per-parameter for any Tessera rung, and asking for
    one raises instead of returning a floor -- a ``Fraction`` cannot carry
    "this is a lower bound", and the only consumer of a shape-free rate,
    ``FormatSpec.exact_bits_per_param``, is multiplied by a parameter count as
    an exact rate.
    """
    from prismaquant.tessera_formats import enumerate_grid_space, wire_overhead_q256

    checked = 0
    for spec in enumerate_grid_space():
        lo, hi = spec.mathematical_q256_bounds
        for rung in (lo, hi):
            with pytest.raises(TesseraFormatError, match="per unit"):
                wire_overhead_q256(spec, recipe=tessera_wire_recipe(spec, rung),
                                   rung=rung)
            with pytest.raises(TesseraFormatError, match="per unit"):
                artifact_bpp(spec, rung)
            checked += 1
    assert checked >= 8, checked


def test_the_synthesized_format_spec_prices_by_shape_and_never_by_a_scalar():
    """``exact_bits_per_param`` is None on every Tessera rung; the fn is set.

    ``FormatSpec`` holds exactly one of the two, so no consumer can pick the
    cheaper one -- and with the forest charged, the scalar is the one that
    cannot exist.  ``memory_bytes_for_shape`` is what the byte gate reads, so
    it is asserted here against the same three numbers above.
    """
    from prismaquant.tessera_render import synthesize_tessera_spec

    # ``bits_for_shape_fn`` routes through ``tessera_tensor_payload_breakdown``,
    # which refuses a column count that is not a whole number of 256-column
    # superblocks, so every shape here is one.
    for name, shape in (("TESSERA_E2M1_K2_R896", (64, 512)),
                        ("TESSERA_E2M1_K1_R512", (96, 768)),
                        ("TESSERA_E4M3_K1_R1024", (96, 512))):
        spec = synthesize_tessera_spec(name)
        assert spec.exact_bits_per_param is None, name
        assert spec.bits_for_shape_fn is not None, name
        family, rung = name.rsplit("_R", 1)
        assert spec.memory_bytes_for_shape(shape) == Fraction(
            _priced_bytes(family, int(rung), shape)
        ), (name, shape)


# ---------------------------------------------------------------------------
# The guard that does not need the next plane to be named
# ---------------------------------------------------------------------------

#: Three shapes for the exhaustive sweep: an MoE expert column count where a
#: per-unit term is 0.1 bpp, a wider expert, and a dense Linear where the same
#: term rounds to 0.002 bpp and an accountant that dropped it would look right.
SWEEP_SHAPES = ((96, 320), (96, 768), (2048, 1024))


def _grid_name(spec) -> str:
    """Tessera's own name for a family's payload grid."""
    return spec.base if spec.arity == 1 else f"{spec.base}x{spec.arity}"


def test_every_realisable_rung_of_every_family_prices_what_tessera_prices():
    """The property, swept: no rung of no family, at no shape, disagrees.

    This is the test that would catch a *thirteenth* plane without anyone
    editing it.  Both sides are exact ``Fraction`` arithmetic over integer
    plane extents, so the sweep is seconds rather than encodes, and neither
    side is a table restated beside the other: the left is
    ``artifact_bpp``, which derives its per-unit terms from
    ``tessera.grammar.forest_plane_bytes`` and the recipe
    ``tessera.export.wire_recipe`` resolves, and the right is
    ``tessera.control.unit_wire_bits``, which is
    ``tessera.calculator.terminal_rate`` -- the accountant the serializer and
    the parser both defer to.  Add a plane to the wire and Tessera's
    accountant moves; PrismaQuant's does not unless the term it derives from
    covers it, and this goes red.

    **What it catches:** every plane Tessera charges and PrismaQuant does not,
    at every realisable rung of every family in the menu, at three shapes --
    which is the failure #126 was.  Before the fix it was red on 514 of the
    2436 (family, rung) pairs at each shape: all 513 rungs of the arity-1 TCQ
    family at 20-56 B, and the one E2M1x2 coset-cap rung at 512 B.

    **What it does not catch:** a plane *both* accountants forget.  That leg is
    ``encode_linear``, and it is asserted above (E2M1, both arities, both sides
    of the cap) and in
    :func:`test_the_window_families_agree_with_the_exporter_too` (E4M3, BF16),
    at the shapes whose encodes are seconds.  It also says nothing about
    header and manifest side bytes, which both figures exclude by the same
    definition, nor about a **shard**: ``artifact_bpp`` prices a whole unit,
    and a tensor-parallel shard writes an INITIAL_STATE plane that neither
    side of this identity carries (RobTand/prismaquant#129).
    """
    from prismaquant.tessera_menu import menu_families

    checked = 0
    refused = 0
    total_rungs = 0
    families = 0
    for spec in menu_families():
        families += 1
        grid = _grid_name(spec)
        lo, hi = spec.mathematical_q256_bounds
        total_rungs += hi - lo + 1
        for rows, columns in SWEEP_SHAPES:
            for rung in range(lo, hi + 1):
                try:
                    priced = _priced_bytes(spec.name, rung, (rows, columns))
                except TesseraFormatError:
                    # The quota does not close over these columns.  Then
                    # Tessera must refuse the same pair: the two accountants
                    # agree about which rungs EXIST, not only what they cost.
                    with pytest.raises(GrammarError):
                        unit_wire_bits(grid, rung, rows, columns)
                    refused += 1
                    continue
                assert priced == unit_wire_bits(grid, rung, rows, columns) / 8, (
                    spec.name, rung, (rows, columns)
                )
                checked += 1
    # Not vacuous, and the bound is derived rather than typed: 256 divides
    # 768 and 1024, so every rung of every family is realisable at those two
    # widths and the sweep must have priced all of them there; 320 is not a
    # multiple of 256, so some rungs are refused there and the refusal path
    # above must have been exercised.
    assert families >= 4, families
    assert checked >= 2 * total_rungs, (checked, total_rungs)
    assert refused > 0, refused


@pytest.mark.parametrize("family,grid,rung", (
    ("TESSERA_E4M3_K1", "E4M3", 256),
    ("TESSERA_BF16_K1", "BF16", 256),
))
def test_the_window_families_agree_with_the_exporter_too(family, grid, rung):
    """The exporter leg on the two WINDOW/CHANNEL families, not just E2M1.

    The forest is a TCQ term, so these two families were never wrong -- which
    is exactly why they belong here.  The identity above is between two
    accountants; if a plane were added that *both* of them missed, only an
    encode would say so, and an encode leg that covers one body kind is not a
    statement about the wire.  One rung each, at the family's floor, on a
    64x512 unit: about nine seconds apiece on CPU, against minutes at L=14 on
    a wide one.
    """
    torch.manual_seed(11)
    exported = encode_linear(
        torch.randn(64, 512), grid=grid_for_name(grid), q256=rung
    ).exact_bytes
    assert _priced_bytes(family, rung, (64, 512)) == exported


def test_an_unrealisable_rung_refuses_in_this_modules_own_error_type():
    """A grammar refusal must not cross the seam as a ``tessera`` exception.

    Charging the forest is what first put a Bresenham walk underneath
    ``artifact_bpp``: before it, a TCQ rung over a block plane had a shape-free
    price and never asked whether the quota closed.  ``bresenham_rate_schedule``
    raises ``tessera.errors.GrammarError``, which descends from
    ``TesseraError`` and **not** from ``ValueError``, so every caller in this
    tree that guards a price with ``except TesseraFormatError`` --
    ``menu_families``, ``enumerate_grid_space``, ``expand_tessera_menu`` --
    would have seen it propagate straight out of the byte gate.  Re-raised in
    this module's type, exactly as ``TesseraFamily.__post_init__`` does with
    ``tuple_grid``'s refusal.
    """
    from tessera.errors import GrammarError

    assert not issubclass(GrammarError, ValueError)
    # R257 needs 5/4 columns at rate 2 over 320 columns: the quota cannot close.
    with pytest.raises(TesseraFormatError, match="not realisable over 320"):
        artifact_bpp("TESSERA_E2M1_K1", 257, shape=(96, 320))
    # ...and it is still a rung the grammar admits at a column count that works.
    assert _priced_bytes("TESSERA_E2M1_K1", 257, (96, 768)) > 0
