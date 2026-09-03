"""``promote_fused`` threads the rung licence instead of collapsing it.

Issue #169. ``allocator_solver.promote_fused`` forwarded to
``promote_serving_units`` without ``fused_licence``, and its legacy per-group
repass then asserted it "cannot write" -- reasoning that was true before #140
(a connected component is a superset of each group it contains, so every
group is already uniform) and is false after it (a licensed fused group is
legitimately NOT uniform). On a licensed assignment the repass either
collapsed every member to ``max(ranks)`` -- silently discarding exactly the
per-member rungs #140 exists to preserve -- or tripped the post-check with
"siblings have mixed formats after promotion", blaming a LEGAL licensed
assignment for being unservable.

So, with the pinned contract licensing ``q256: per_member``, a fused group
of Tessera rungs driven through ``promote_fused`` keeps its per-member rates;
with ``q256: shared`` -- or no licence at all -- it lands on one rung. The
licences below are constructed, not read off the installed contract: what is
pinned is the rule (the ``q256`` word decides), not today's field roster.
"""
from __future__ import annotations

from prismaquant.allocator_solver import promote_fused
from prismaquant.tessera_formats import format_promotion_class

_LO, _HI = "TESSERA_E2M1_K2_R512", "TESSERA_E2M1_K2_R896"


def _rank():
    return {_LO: 0, _HI: 1}


def _licence(q256):
    """The pinned contract's ``fused_module`` block, as the solver reads it."""
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


_FUSED_MEMBERS = ("blk.q_proj", "blk.k_proj", "blk.v_proj")


def _mixed_assignment():
    return {_FUSED_MEMBERS[0]: _LO, _FUSED_MEMBERS[1]: _HI,
            _FUSED_MEMBERS[2]: _LO}


def test_promote_fused_keeps_per_member_rates_under_a_per_member_licence():
    """The #169 arm: the licensed path preserves what the repass collapsed."""
    assignment = _mixed_assignment()
    legal = {m: set(_rank()) for m in _FUSED_MEMBERS}
    out = promote_fused(
        dict(assignment), _rank(), profile=_FusedProfile(),
        legal_formats=legal, fused_licence=_licence("per_member"))
    assert out == assignment
    assert {format_promotion_class(f) for f in out.values()} == {
        "TESSERA_E2M1_K2"}
    assert len(set(out.values())) == 2, out


def test_promote_fused_collapses_without_a_per_member_licence():
    """A re-tightened contract -- or none at all -- is one rung per group."""
    for licence in (_licence("shared"), None):
        assignment = _mixed_assignment()
        legal = {m: set(_rank()) for m in _FUSED_MEMBERS}
        out = promote_fused(
            dict(assignment), _rank(), profile=_FusedProfile(),
            legal_formats=legal, fused_licence=licence)
        assert set(out.values()) == {_HI}, (licence, out)
