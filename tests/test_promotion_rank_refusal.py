"""Promotion refuses a format it cannot rank, and says what is missing.

``format_rank`` is a dense ordinal over *this run's* menu, and promotion's only
question is which member of a serving unit carries the most expensive format.
Handed a format outside that menu, the ``max(..., key=format_rank[fmt])`` in
``_promote_group_components`` raised a bare ``KeyError: '<fmt>'`` -- which reads
as a corrupt DP assignment when what it means is that the rank table does not
cover the assignment (#129).

It is a latent gap rather than a live crash: on the production path every
assignable format is drawn from the same menu that built the rank, so the
question always has an answer today. The reason to fix it anyway is that the
diagnostic is what a future candidate source -- one pricing a format from its
candidate bytes rather than from a ``FormatSpec`` -- would land on, and a bare
KeyError sends the reader to the wrong system.
"""
from __future__ import annotations

import pytest

from prismaquant.allocator_solver import (
    FormatRankUnknown,
    promote_serving_units,
    promote_fused,
)
from prismaquant.model_profiles import DefaultProfile

RANK = {"NVFP4": 0, "FP8_DYNAMIC": 1, "BF16": 2}
PROFILE = DefaultProfile()

#: q/k/v are one fused module under the default profile, so these two names are
#: one serving unit and promotion must choose between their formats.
FUSED = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj")


def test_an_unranked_format_refuses_and_names_what_is_missing():
    assignment = {FUSED[0]: "NVFP4", FUSED[1]: "FUTURE_FMT_X"}

    with pytest.raises(FormatRankUnknown) as exc:
        promote_serving_units(assignment, RANK, profile=PROFILE)

    msg = str(exc.value)
    # The four things a reader needs to act.
    assert "'FUTURE_FMT_X'" in msg, msg                       # which format
    assert "self_attn.q_proj" in msg, msg                     # which unit
    assert "2 members" in msg, msg
    assert "_promote_group_components max-choice" in msg, msg  # where it fired
    assert "['NVFP4', 'FP8_DYNAMIC', 'BF16']" in msg, msg      # the menu, in order
    assert "exact serialized rate" in msg, msg                 # what would fix it
    # And it must not offer to guess.
    assert "will not invent" in msg, msg


def test_the_refusal_is_still_a_keyerror_for_existing_handlers():
    """Subclassing matters: callers already write ``except KeyError``."""
    assignment = {FUSED[0]: "NVFP4", FUSED[1]: "FUTURE_FMT_X"}

    with pytest.raises(KeyError):
        promote_serving_units(assignment, RANK, profile=PROFILE)


def test_the_message_survives_keyerror_str():
    """``KeyError.__str__`` reprs its argument, so a multi-line diagnostic
    would otherwise arrive as one backslash-escaped blob."""
    assignment = {FUSED[0]: "NVFP4", FUSED[1]: "FUTURE_FMT_X"}

    with pytest.raises(FormatRankUnknown) as exc:
        promote_serving_units(assignment, RANK, profile=PROFILE)

    msg = str(exc.value)
    assert "\n" in msg, "the diagnostic must survive as multiple lines"
    assert not msg.startswith("'"), msg
    assert "\\n" not in msg, msg


def test_the_legacy_repass_refuses_too_when_reached_directly():
    """``promote_fused``'s own rank list was a bare lookup.

    It is shadowed on the normal path -- ``promote_serving_units`` raises at
    the max-choice first -- so this pins the site rather than a live bug. The
    point is that no bare ``format_rank[...]`` is left on a name that came out
    of an assignment, because the next caller to arrive by another path is
    exactly how the original defect would come back.
    """
    import prismaquant.allocator_solver as solver

    assignment = {FUSED[0]: "NVFP4", FUSED[1]: "FUTURE_FMT_X"}
    # Neutralise the component pass so the legacy repass is the first to look.
    original = solver._promote_group_components
    solver._promote_group_components = lambda a, *args, **kw: dict(a)
    try:
        with pytest.raises(FormatRankUnknown) as exc:
            promote_fused(assignment, RANK, profile=PROFILE)
    finally:
        solver._promote_group_components = original
    assert "promote_fused legacy repass" in str(exc.value)


def test_a_single_member_unit_with_an_unranked_format_still_passes_through():
    """Not an over-correction: a component of one asks no ranking question.

    ``_promote_group_components`` skips components below two members, so a lone
    Linear carrying an unranked format has nothing to be promoted against. That
    was true before and stays true -- the fix names a missing answer, it does
    not invent a requirement.
    """
    assignment = {"model.layers.0.mlp.down_proj": "FUTURE_FMT_X"}

    out = promote_serving_units(assignment, RANK, profile=PROFILE)

    assert out == assignment


def test_a_fully_ranked_assignment_promotes_exactly_as_before():
    """The regression guard: the refusal must not change any real decision."""
    assignment = {FUSED[0]: "NVFP4", FUSED[1]: "BF16"}

    out = promote_serving_units(assignment, RANK, profile=PROFILE)

    assert out == {FUSED[0]: "BF16", FUSED[1]: "BF16"}, out
