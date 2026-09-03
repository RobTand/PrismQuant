"""The solver's per-member rung relaxation is licensed, and fused-only.

Issue #140. ``allocator_solver._resolve_family_group`` moved a serving unit
onto one Tessera family while leaving each member's rate free -- on the
strength of its own docstring, which asserted the shared-decoder constraint
instead of reading it, and for EVERY serving-unit kind: ``promote_serving_units``
flattens fused groups and packed-expert groups into one list, and the family
branch fired for any component whose max-rank format had a coarser promotion
class. Tessera's contract scopes the relaxation to one vLLM-fused module
(``fused_module``) and publishes nothing about per-expert rungs
(``expert_parallel`` is a closed world with no unit in it).

So, with the pinned contract licensing ``q256: per_member``:

* a fused group of Tessera rungs keeps its per-member rates;
* a packed-expert group of the SAME rungs lands on one rung;

and marking ``q256`` ``shared`` -- or licensing nothing at all -- collapses
both to one rung. A component that unions a fused group with a packed one is
ambiguous (neither licence scope covers it) and refuses rather than picking a
side.

The licences below are constructed, not read off the installed contract: what
is pinned is the rule (the ``q256`` word decides), not today's eight-field
roster. The installed table's own word is pinned where it is read --
``tests/test_tessera_menu.py`` -- not here.
"""
from __future__ import annotations

import pytest

from prismaquant.allocator_solver import promote_serving_units
from prismaquant.tessera_formats import format_promotion_class

_LO, _HI = "TESSERA_E2M1_K2_R512", "TESSERA_E2M1_K2_R896"


def _rank():
    return {_LO: 0, _HI: 1}


def _licence(q256):
    """The pinned contract's ``fused_module`` block, as the solver reads it.

    Function-level import: the constructor lives in
    ``tessera_runtime_contract`` beside the parser the pin reviews, and this
    file pins what the solver DOES with the word, not where the word comes
    from.
    """
    from prismaquant.tessera_runtime_contract import (
        FUSED_MODULE_SCHEMA,
        FusedModuleLicence,
    )

    return FusedModuleLicence(
        schema=FUSED_MODULE_SCHEMA,
        fields={"q256": q256},
        sidecar_q256="int_or_per_role_list",
        mixed_rung_receipt=False,
    )


class _FusedProfile:
    """q/k/v form one fused serving unit; nothing is a packed expert."""

    def fused_sibling_group(self, name: str) -> str | None:
        if name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj"):
            return "blk.qkv"
        return None

    def packed_expert_format_group(self, name: str) -> str | None:
        return None


class _PackedProfile:
    """Three expert projections form one packed serving unit; nothing fuses."""

    def fused_sibling_group(self, name: str) -> str | None:
        return None

    def packed_expert_format_group(self, name: str) -> str | None:
        if name.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj"):
            return "blk.mlp.experts"
        return None


class _OverlappingProfile:
    """``o.mid`` sits in a fused group AND a packed one: the ambiguous case."""

    def fused_sibling_group(self, name: str) -> str | None:
        return "overlap.fused" if name in {"o.left", "o.mid"} else None

    def packed_expert_format_group(self, name: str) -> str | None:
        return "overlap.moe" if name in {"o.mid", "o.right"} else None


_FUSED_MEMBERS = ("blk.q_proj", "blk.k_proj", "blk.v_proj")
_PACKED_MEMBERS = (
    "blk.mlp.experts.gate_proj",
    "blk.mlp.experts.up_proj",
    "blk.mlp.experts.down_proj",
)


def _mixed_assignment(members):
    return {members[0]: _LO, members[1]: _HI, members[2]: _LO}


def test_fused_group_keeps_per_member_rates_under_a_per_member_licence():
    """The contrast arm: the licensed path preserves what the fix keeps."""
    assignment = _mixed_assignment(_FUSED_MEMBERS)
    legal = {m: set(_rank()) for m in _FUSED_MEMBERS}
    out = promote_serving_units(
        dict(assignment), _rank(), profile=_FusedProfile(),
        legal_formats=legal, fused_licence=_licence("per_member"))
    assert out == assignment
    assert {format_promotion_class(f) for f in out.values()} == {
        "TESSERA_E2M1_K2"}
    assert len(set(out.values())) == 2, out


def test_packed_expert_group_lands_on_one_rung():
    """The sharper defect: per-expert rungs is a claim nothing attests.

    Same rungs, same ranks, same legality as the fused arm -- the only
    difference is the KIND of serving unit, which the contract does not cover.
    """
    assignment = _mixed_assignment(_PACKED_MEMBERS)
    legal = {m: set(_rank()) for m in _PACKED_MEMBERS}
    out = promote_serving_units(
        dict(assignment), _rank(), profile=_PackedProfile(),
        legal_formats=legal, fused_licence=_licence("per_member"))
    assert set(out.values()) == {_HI}, out


def test_a_shared_q256_licence_collapses_both_kinds():
    """A re-tightened contract stops the relaxation; it does not narrow it."""
    for members, profile in ((_FUSED_MEMBERS, _FusedProfile()),
                             (_PACKED_MEMBERS, _PackedProfile())):
        assignment = _mixed_assignment(members)
        legal = {m: set(_rank()) for m in members}
        out = promote_serving_units(
            dict(assignment), _rank(), profile=profile,
            legal_formats=legal, fused_licence=_licence("shared"))
        assert set(out.values()) == {_HI}, (members, out)


def test_no_licence_collapses_both_kinds():
    """Absence is not permission: with no table to derive it from, one rung."""
    for members, profile in ((_FUSED_MEMBERS, _FusedProfile()),
                             (_PACKED_MEMBERS, _PackedProfile())):
        assignment = _mixed_assignment(members)
        legal = {m: set(_rank()) for m in members}
        out = promote_serving_units(
            dict(assignment), _rank(), profile=profile,
            legal_formats=legal, fused_licence=None)
        assert set(out.values()) == {_HI}, (members, out)


def test_the_unset_licence_reads_no_pin_and_collapses(monkeypatch):
    """The default is the fail-closed read, not a second permissive default.

    With no ``fused_licence`` passed, promotion reads the pinned contract
    through ``tessera_menu.fused_module_licence``; with no pin in the
    environment that read is ``None`` -- the absence of a licence -- so even
    a fused group lands on one rung. The pin is deleted rather than assumed
    absent, so a leaked ``dev_pin`` fixture fails this loudly instead of
    passing it for the wrong reason.
    """
    from prismaquant import tessera_runtime_contract as trc

    monkeypatch.delenv(trc.TESSERA_DEV_PIN_ENV, raising=False)
    assignment = _mixed_assignment(_FUSED_MEMBERS)
    legal = {m: set(_rank()) for m in _FUSED_MEMBERS}
    out = promote_serving_units(
        dict(assignment), _rank(), profile=_FusedProfile(),
        legal_formats=legal)
    assert set(out.values()) == {_HI}, out


def test_a_component_unioning_fused_with_packed_refuses():
    """Neither licence scope covers a mixed component, so it picks no side.

    Fires only where the family branch would: a stock-format overlap keeps
    the uniform path it always had (pinned by
    ``test_serving_unit_promotion_handles_overlapping_groups_order_independently``).
    """
    members = ("o.left", "o.mid", "o.right")
    assignment = {m: _HI for m in members}
    assignment["o.left"] = _LO
    legal = {m: set(_rank()) for m in members}
    with pytest.raises(AssertionError, match="fused.*packed|packed.*fused"):
        promote_serving_units(
            dict(assignment), _rank(), profile=_OverlappingProfile(),
            legal_formats=legal, fused_licence=_licence("per_member"))
