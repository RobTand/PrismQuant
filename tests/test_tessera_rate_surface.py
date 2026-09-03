"""The schedule helper asks Tessera's grammar; it carries no trellis minimum.

Issue #161: ``uniform_column_schedule`` refused any 256-column block with
fewer than ``MIN_TRELLIS_STEPS = 8`` coded (non-bypass) columns, while the
wire the schedules are checked against -- ``tessera.grammar`` /
``tessera.manifest`` -- has no per-block coded minimum at all: a schedule is
legal iff every rate is in ``1..cap`` and the quota closes
(``validate_rate_schedule`` + ``superblock_quota_ok``).  The "bypass" word
itself is the retired Gridbook wire's: ``TesseraFamily`` has no
``bypass_rate``, so every call raised ``AttributeError`` before reaching any
guard.  These tests pin the grammar as the authority rather than any number.
"""

import pytest

from prismaquant.tessera_formats import (
    TesseraFormatError,
    family_rate_cap,
    get_tessera_family,
)
from prismaquant.tessera_rate_surface import (
    TesseraRateSurface,
    densify_rate_surface,
    uniform_column_schedule,
)


def _surface(family="TESSERA_E2M1_K1", lo=256, mid=512, hi=768):
    return TesseraRateSurface(
        unit_name="u",
        family=family,
        layout="tight",
        currency="kl",
        anchor_q256=(lo, mid, hi),
        anchor_dloss=(4.0, 2.0, 1.0),
        anchor_stderr=(0.1, 0.1, 0.1),
    )


def test_top_rung_schedule_is_accepted():
    """The rung the old guard refused: zero sub-cap columns in the block."""
    spec = get_tessera_family("TESSERA_E2M1_K1")
    cap = family_rate_cap(spec)
    top = cap * 256 // spec.arity
    schedule = uniform_column_schedule(256, top, family="TESSERA_E2M1_K1")
    assert len(schedule) == 256
    assert set(schedule) == {cap}
    assert sum(schedule) * 256 == top * spec.arity * 256


def test_schedule_is_the_grammar_schedule():
    """The helper returns what ``TesseraFamily.column_schedule`` returns."""
    for family, rung, columns in [
        ("TESSERA_E2M1_K1", 256, 256),
        ("TESSERA_E2M1_K1", 512, 512),
        ("TESSERA_E2M1_K1", 768, 1024),
        ("TESSERA_E2M1_K1", 600, 256),
        ("TESSERA_E2M1_K2", 896, 256),
        ("TESSERA_E4M3_K1", 256, 256),
    ]:
        spec = get_tessera_family(family)
        assert uniform_column_schedule(
            columns, rung, family=family,
        ) == spec.column_schedule(rung, columns)


def test_out_of_bounds_rung_is_refused():
    spec = get_tessera_family("TESSERA_E2M1_K1")
    cap = family_rate_cap(spec)
    with pytest.raises(TesseraFormatError):
        uniform_column_schedule(
            256, cap * 256 // spec.arity + 1, family="TESSERA_E2M1_K1",
        )


def test_short_block_width_is_refused():
    with pytest.raises(TesseraFormatError):
        uniform_column_schedule(320, 512, family="TESSERA_E2M1_K1")


def test_no_trellis_minimum_exported():
    """The grammar owns the minimum, so the seam must not name one."""
    import prismaquant.tessera_formats as formats

    assert not hasattr(formats, "MIN_TRELLIS_STEPS")
    assert "MIN_TRELLIS_STEPS" not in formats.__all__


def test_densify_prices_the_top_rung():
    """End to end: the allocator's most-wanted rung densifies, with bytes."""
    alphabets = {1: [0] * 4, 2: [0] * 8, 3: [0] * 16}
    built = densify_rate_surface(
        _surface(), (96, 256), q256_values=[768], alphabets=alphabets,
    )
    assert len(built) == 1
    assert built[0].body_rate_q256 == 768
