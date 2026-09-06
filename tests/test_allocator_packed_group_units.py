"""Packed-MoE serving groups as first-class DP decision units.

A packed expert block (DeepSeek-V4: 768 per-expert Linears, ~6.44B params
per layer) is
atomic at serve time — vLLM's FusedMoE loads the whole group under one
scheme. The DP used to price upgrades PER ROW (a tiny row's upgrade is
charged one bin, ~0.0001 avg bits) while ``promote_serving_units`` charged
the WHOLE group (~0.1 avg bits): a ~1000x price mismatch that put mispriced
expert rows at the top of the per-bin ranking, over-tightened the
feasibility bisection, and starved attention/shared rows at the format
floor while headroom sat unused.

``aggregate_packed_serving_groups`` collapses each packed group into ONE
multi-choice DP item — per-format cost = exact sum of member
predicted_dloss, per-format bytes = exact sum of member bytes — so the DP
and the serving constraint price identical moves and MoE promotion becomes
a validated no-op.

These tests pin:
  - aggregation groups packed-expert rows and passes everything else through
  - the unit's per-format predicted_dloss / memory_bytes are exact member sums
  - expand_packed_group_assignment broadcasts the group decision to members
  - a group with no common legal format falls back to individual rows
  - END-TO-END: the per-row mispricing starved the dense row (old path);
    with groups as DP units the dense row wins the headroom (new path)
  - the whole group still flips as one unit when the budget affords it
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    _PACKED_GROUP_MARKER,
    PackedExpertRoleUnknown,
    aggregate_fused_siblings,
    aggregate_packed_serving_groups,
    expand_packed_group_assignment,
)
from prismaquant.allocator_solver import (
    Candidate,
    compute_achieved,
    promote_serving_units,
    solve_with_promotion,
)
from prismaquant.model_profiles import DefaultProfile


class _PackedProfile:
    """Groups ``...mlp.experts.<n>.<proj>`` rows per layer (DSv4-style)."""

    def packed_expert_format_group(self, name: str) -> str | None:
        parts = name.split(".")
        if "experts" not in parts:
            return None
        idx = parts.index("experts")
        tail = parts[idx + 1:]
        if len(tail) == 2 and tail[0].isdigit():
            parent = ".".join(parts[: idx + 1])
            return f"{parent}::__packed_format__:gate_proj,up_proj,down_proj"
        return None

    def packed_expert_role_group(self, name: str) -> str | None:
        # Expert leaf naming is PROFILE knowledge (the allocator does not parse
        # model names), so delegate to the real base-profile derivation rather
        # than restating a projection table in the test fixture.
        return DefaultProfile().packed_expert_role_group(name)

    def fused_sibling_group(self, name: str) -> str | None:
        return None


_LOW, _HIGH = "IQ2_XXS", "Q6_K"


def _specs():
    return [fr.REGISTRY[_LOW], fr.REGISTRY[_HIGH]]


def _mk_candidates(n_params: int, dloss_low: float, dloss_high: float):
    """Two-rung candidate list with exact byte math (bpp from the registry)."""
    cands = []
    for fmt, dloss in ((_LOW, dloss_low), (_HIGH, dloss_high)):
        bpp = fr.REGISTRY[fmt].effective_bits
        cands.append(Candidate(
            fmt=fmt,
            bits_per_param=bpp,
            memory_bytes=int(round(bpp * n_params / 8.0)),
            predicted_dloss=dloss,
        ))
    return cands


def _mk_moe_fixture(n_experts: int = 8, expert_params: int = 4096,
                    dense_params: int = 1 << 20):
    """One packed expert block + one dense attention row + one shared row."""
    stats: dict = {}
    candidates: dict = {}
    expert_names = []
    for e in range(n_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.mlp.experts.{e}.{proj}"
            expert_names.append(name)
            stats[name] = {
                "h_trace": 0.001,
                "n_params": expert_params,
                "in_features": 256,
                "out_features": expert_params // 256,
                "n_tokens_seen": 7,
            }
            # Tiny per-row gain: experts are cheap to leave at the floor.
            candidates[name] = _mk_candidates(expert_params, 0.05, 0.04)
    dense = "model.layers.0.self_attn.o_proj"
    stats[dense] = {
        "h_trace": 5.0, "n_params": dense_params,
        "in_features": 1024, "out_features": dense_params // 1024,
        "n_tokens_seen": 32768,
    }
    # Huge gain: the dense row deserves the headroom.
    candidates[dense] = _mk_candidates(dense_params, 100.0, 0.0)
    return stats, candidates, expert_names, dense


# ---------------------------------------------------------------------------
# Aggregation mechanics
# ---------------------------------------------------------------------------

def test_aggregation_groups_expert_rows_and_passes_dense_through():
    stats, cands, expert_names, dense = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    stats_ext, costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())

    supers = [n for n in cands_ext if _PACKED_GROUP_MARKER in n]
    assert len(supers) == 1, f"expected 1 packed group, got {supers}"
    assert dense in cands_ext
    for m in expert_names:
        assert m not in cands_ext, f"member row {m} leaked into DP items"

    entry = stats_ext[supers[0]]
    assert sorted(entry["_packed_group_members"]) == sorted(expert_names)
    assert entry["n_params"] == sum(
        stats[m]["n_params"] for m in expert_names)


def test_group_unit_prices_are_exact_member_sums():
    stats, cands, expert_names, _ = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)

    for cand in cands_ext[super_name]:
        expected_dloss = sum(
            next(c for c in cands[m] if c.fmt == cand.fmt).predicted_dloss
            for m in expert_names
        )
        expected_bytes = sum(
            next(c for c in cands[m] if c.fmt == cand.fmt).memory_bytes
            for m in expert_names
        )
        assert abs(cand.predicted_dloss - expected_dloss) < 1e-9
        assert cand.memory_bytes == expected_bytes
        n_params = stats_ext[super_name]["n_params"]
        assert abs(cand.bits_per_param - 8.0 * expected_bytes / n_params) < 1e-9
    # compute_achieved must resolve the unit's exact bytes via its candidate.
    mem_map = stats_ext[super_name]["_memory_bytes_by_format"]
    assert set(mem_map) == {c.fmt for c in cands_ext[super_name]}


def test_expand_broadcasts_group_format_to_all_members():
    stats, cands, expert_names, dense = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)

    assignment = {super_name: _HIGH, dense: _LOW}
    expanded = expand_packed_group_assignment(assignment, stats_ext)
    assert not any(_PACKED_GROUP_MARKER in n for n in expanded)
    for m in expert_names:
        assert expanded[m] == _HIGH
    assert expanded[dense] == _LOW
    # Coherent by construction: serving promotion is a no-op.
    rank = {_LOW: 0, _HIGH: 1}
    assert promote_serving_units(
        expanded, rank, profile=_PackedProfile()) == expanded


def test_singleton_group_passes_through():
    stats, cands, expert_names, dense = _mk_moe_fixture(n_experts=1)
    solo = expert_names[0]
    keep = {solo: cands[solo], dense: cands[dense]}
    stats_keep = {solo: stats[solo], dense: stats[dense]}

    class _SoloProfile(_PackedProfile):
        def packed_expert_format_group(self, name):
            return "solo.group" if name == solo else None

    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats_keep, {}, _specs(), keep, _SoloProfile())
    assert not any(_PACKED_GROUP_MARKER in n for n in cands_ext)
    assert solo in cands_ext and dense in cands_ext


def test_no_common_format_falls_back_to_individual_rows():
    stats, cands, expert_names, dense = _mk_moe_fixture(n_experts=2)
    # Cripple one member: only the LOW rung is legal for it.
    crippled = expert_names[0]
    cands[crippled] = [c for c in cands[crippled] if c.fmt == _LOW]
    # And another member only carries the HIGH rung: intersection is empty.
    cands[expert_names[1]] = [
        c for c in cands[expert_names[1]] if c.fmt == _HIGH]

    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    assert not any(_PACKED_GROUP_MARKER in n for n in cands_ext), (
        "empty-intersection group must not become a DP unit")
    for m in expert_names:
        assert m in cands_ext, f"fallback must keep {m} as an individual row"


def test_fused_aggregation_skips_packed_units():
    stats, cands, _, _ = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    stats_ext, costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    stats2, costs2, cands2 = aggregate_fused_siblings(
        stats_ext, costs_ext, _specs(), cands_ext, _PackedProfile())
    assert set(cands2) == set(cands_ext)


# ---------------------------------------------------------------------------
# End-to-end: DP and serving constraint price identical moves
# ---------------------------------------------------------------------------

def _solver_env():
    specs = {s.name: s for s in _specs()}
    rank = {_LOW: 0, _HIGH: 1}
    return specs, rank


def test_per_row_pricing_disagrees_with_serving_charge_old_path():
    """Documents the audit's price mismatch on the UN-aggREGATED path: the
    raw DP upgrades a strict subset of a packed group (charging only the
    subset's bins), then whole-group serving promotion charges the rest —
    the promoted footprint exceeds what the DP believed it paid. That gap
    is what over-tightened the feasibility loop and starved dense rows at
    scale. With group units (next test) the gap is identically zero."""
    from prismaquant.allocator_solver import solve_allocation

    stats, cands, expert_names, dense = _mk_moe_fixture(
        n_experts=8, expert_params=4096, dense_params=1 << 20)
    specs, rank = _solver_env()
    low, high = fr.REGISTRY[_LOW].effective_bits, fr.REGISTRY[_HIGH].effective_bits
    total = sum(stats[n]["n_params"] for n in stats)
    dense_cost_bits = (high - low) * stats[dense]["n_params"] / total
    # Affords the dense upgrade plus a strict subset of expert rows.
    target = low + dense_cost_bits + 0.05

    result = solve_allocation(stats, cands, target, 0.001)
    assert result is not None
    assign, _chosen = result
    upgraded = [m for m in expert_names if assign[m] == _HIGH]
    assert upgraded and len(upgraded) < len(expert_names), (
        "fixture must make the DP upgrade a strict subset of the group — "
        "if the solver changed, re-derive this fixture")

    ach_dp, _ = compute_achieved(stats, assign, specs, candidates=cands)
    promoted = promote_serving_units(assign, rank, profile=_PackedProfile())
    ach_serving, _ = compute_achieved(stats, promoted, specs, candidates=cands)
    assert ach_serving > ach_dp + 1e-9, (
        "serving charge must exceed the DP's per-row price (the audit's "
        "mispricing)")


def test_group_units_fund_the_dense_row_and_use_headroom():
    """The fix: with the packed group as ONE DP unit, the same budget goes
    to the dominant dloss/GB row (dense), experts stay at the floor, and
    the emitted per-tensor assignment is serving-coherent without any
    promotion adjustment."""
    stats, cands, expert_names, dense = _mk_moe_fixture(
        n_experts=8, expert_params=4096, dense_params=1 << 20)
    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    specs, rank = _solver_env()
    low, high = fr.REGISTRY[_LOW].effective_bits, fr.REGISTRY[_HIGH].effective_bits
    total = sum(stats[n]["n_params"] for n in stats)
    dense_cost_bits = (high - low) * stats[dense]["n_params"] / total
    target = low + dense_cost_bits + 0.05

    assign, achieved = solve_with_promotion(
        stats_ext, cands_ext, target, specs, rank,
        bit_precision=0.001, profile=_PackedProfile())
    assert assign is not None
    assert achieved <= target + 0.01
    assert assign[dense] == _HIGH, "dense row must win the headroom now"
    super_name = next(n for n in assign if _PACKED_GROUP_MARKER in n)
    assert assign[super_name] == _LOW

    expanded = expand_packed_group_assignment(assign, stats_ext)
    promoted = promote_serving_units(expanded, rank, profile=_PackedProfile())
    assert promoted == expanded, "MoE promotion must be a no-op on group units"


def test_role_group_comes_from_the_profile_not_the_allocator():
    """The role bucket is the packed 3D PARENT the projection belongs to, and
    the profile derives it — the allocator no longer carries its own projection
    table (``base.py``: "the solver asks the profile for groups; it does not
    parse model names itself"). Behaviour for today's leaf names is unchanged:
    gate/up (``w1``/``w3``) on one unit, down (``w2``) on the other."""
    prof = DefaultProfile()
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.7.gate_proj") == "gate_up_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.7.up_proj") == "gate_up_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.7.down_proj") == "down_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.gate_up_proj") == "gate_up_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.0.w1") == "gate_up_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.0.w3") == "gate_up_proj"
    assert prof.packed_expert_role_group(
        "model.layers.3.mlp.experts.0.w2") == "down_proj"
    # Not an expert projection at all.
    assert prof.packed_expert_role_group(
        "model.layers.3.self_attn.o_proj") is None
    # No projection table left in the allocator.
    import prismaquant.allocator_candidates as ac
    assert not hasattr(ac, "packed_projection_role_group")
    assert not hasattr(ac, "_PACKED_ROLE_GROUPS")


def test_packed_expert_role_group_covers_every_spec_leaf_name():
    """Every expert leaf name any shipped structure spec declares must get a
    role from its own profile, and the gate_up/down partition must match the
    convention the allocator used to hardcode. Proven over
    ``model_profiles/specs/*.json`` so a spec cannot introduce a leaf the role
    split silently cannot key."""
    from prismaquant.model_profiles import registry as prof_registry

    profiles = {p().name: p() for p in prof_registry._REGISTERED}
    profiles["default"] = DefaultProfile()
    expected = {
        "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
        "gate_up_proj": "gate_up_proj", "w1": "gate_up_proj",
        "w3": "gate_up_proj",
        "down_proj": "down_proj", "w2": "down_proj",
    }

    spec_dir = Path(prof_registry.__file__).resolve().parent / "specs"
    spec_files = sorted(spec_dir.glob("*.json"))
    assert spec_files, f"no structure specs found under {spec_dir}"
    checked = 0
    for path in spec_files:
        payload = json.loads(path.read_text())
        packed = payload.get("packed_experts") or {}
        leaves = set(packed.get("param_names") or ())
        splits = packed.get("projection_splits") or {}
        for parent, projections in splits.items():
            leaves.add(parent)
            leaves.update(projections)
        for group in packed.get("format_groups") or ():
            leaves.update(group)
        if not leaves:
            continue
        profile = profiles.get(path.stem)
        assert profile is not None, (
            f"spec {path.name} has no registered profile named {path.stem!r}")
        for leaf in sorted(leaves):
            assert leaf in expected, (
                f"{path.name} declares expert leaf {leaf!r} with no expected "
                "role in this test — extend the profile's projection_splits "
                "and this table together")
            for qname in (
                f"model.layers.0.mlp.experts.{leaf}",
                f"model.layers.0.mlp.experts.3.{leaf}",
            ):
                got = profile.packed_expert_role_group(qname)
                assert got == expected[leaf], (
                    f"{path.stem}: {qname} -> {got!r}, want "
                    f"{expected[leaf]!r}")
                checked += 1
    assert checked >= 2 * len(expected), checked


def test_role_split_hard_errors_on_a_leaf_the_profile_cannot_name():
    """A new architecture whose expert leaves the profile does not declare must
    FAIL, not silently fall back to one layer-uniform unit: the split is opt-in
    and hard-gated, so the operator asked for role units and would otherwise
    ship a different allocation with no signal. The error names the qname and
    points at the declarative fix."""
    from prismaquant.allocator_candidates import packed_role_split_profile

    class _NewArch:
        """Groups an unfamiliar expert leaf, and cannot name its role."""

        def packed_expert_format_group(self, name: str) -> str | None:
            return "layer0::experts" if ".experts." in name else None

        def packed_expert_role_group(self, name: str) -> str | None:
            return DefaultProfile().packed_expert_role_group(name)

    prof = packed_role_split_profile(_NewArch())
    with pytest.raises(PackedExpertRoleUnknown) as exc:
        prof.packed_expert_format_group("model.layers.0.mlp.experts.0.wi_0")
    msg = str(exc.value)
    assert "wi_0" in msg and "projection_splits" in msg
    # A profile with no role accessor at all is the same explicit failure.
    class _NoRoles:
        def packed_expert_format_group(self, name: str) -> str | None:
            return "layer0::experts" if ".experts." in name else None

    with pytest.raises(PackedExpertRoleUnknown):
        packed_role_split_profile(_NoRoles()).packed_expert_format_group(
            "model.layers.0.mlp.experts.0.gate_proj")
    # Non-expert names still pass through untouched (no group, no role).
    assert prof.packed_expert_format_group(
        "model.layers.0.self_attn.o_proj") is None


def test_role_split_error_is_not_swallowed_into_ungrouped():
    """The aggregation loop guards ``group_fn`` with a broad except (profiles
    without the method), which would turn the explicit "cannot name this role"
    verdict into "this row has no group" — the silent degradation the hard
    error exists to prevent. It must escape."""
    from prismaquant.allocator_candidates import packed_role_split_profile

    class _NewArch:
        def packed_expert_format_group(self, name: str) -> str | None:
            return "layer0::experts" if ".experts." in name else None

        def packed_expert_role_group(self, name: str) -> str | None:
            return None

        def fused_sibling_group(self, name: str) -> str | None:
            return None

    name = "model.layers.0.mlp.experts.0.wi_0"
    stats = {name: {"h_trace": 1.0, "n_params": 4096,
                    "in_features": 64, "out_features": 64}}
    cands = {name: _mk_candidates(4096, 0.05, 0.04)}
    prof = packed_role_split_profile(_NewArch())
    with pytest.raises(PackedExpertRoleUnknown):
        aggregate_packed_serving_groups(stats, {name: {}}, _specs(),
                                        cands, prof)


def test_role_split_profile_yields_gate_up_and_down_units():
    """--packed-role-split granularity: per layer, gate+up projections form
    one DP/serving unit and down projections another; promotion through the
    SAME wrapped profile treats role-mixed layer formats as coherent."""
    from prismaquant.allocator_candidates import packed_role_split_profile

    stats, cands, expert_names, dense = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    prof = packed_role_split_profile(_PackedProfile())
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, prof)

    supers = [n for n in cands_ext if _PACKED_GROUP_MARKER in n]
    assert len(supers) == 2, f"expected gate_up + down units, got {supers}"
    gate_up = frozenset(
        m for m in expert_names if m.endswith((".gate_proj", ".up_proj")))
    down = frozenset(m for m in expert_names if m.endswith(".down_proj"))
    got = {frozenset(stats_ext[s]["_packed_group_members"]) for s in supers}
    assert got == {gate_up, down}

    # A role-mixed layer (gate/up LOW, down HIGH) must be serving-coherent
    # under the same wrapped profile: promotion is a no-op.
    assignment = {dense: _LOW}
    for s in supers:
        role_is_down = stats_ext[s]["_packed_group_key"].endswith(
            "::role:down_proj")
        assignment[s] = _HIGH if role_is_down else _LOW
    expanded = expand_packed_group_assignment(assignment, stats_ext)
    rank = {_LOW: 0, _HIGH: 1}
    assert promote_serving_units(expanded, rank, profile=prof) == expanded
    for m in expert_names:
        assert expanded[m] == (_HIGH if m in down else _LOW)


def test_role_split_default_off_keeps_layer_uniform_units():
    from prismaquant.allocator_candidates import packed_role_split_profile

    stats, cands, expert_names, _ = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    _stats2, _costs2, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    assert sum(1 for n in cands_ext if _PACKED_GROUP_MARKER in n) == 1

    class _NoPacked:  # pass-through for profiles without packed groups
        pass

    p = _NoPacked()
    assert packed_role_split_profile(p) is p


def test_whole_group_still_promotes_when_budget_affords_it():
    stats, cands, expert_names, dense = _mk_moe_fixture(
        n_experts=8, expert_params=4096, dense_params=1 << 20)
    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    specs, rank = _solver_env()
    high = fr.REGISTRY[_HIGH].effective_bits

    assign, achieved = solve_with_promotion(
        stats_ext, cands_ext, high, specs, rank,
        bit_precision=0.001, profile=_PackedProfile())
    assert assign is not None
    super_name = next(n for n in assign if _PACKED_GROUP_MARKER in n)
    assert assign[super_name] == _HIGH, "group must flip as one unit"
    expanded = expand_packed_group_assignment(assign, stats_ext)
    for m in expert_names:
        assert expanded[m] == _HIGH

    ach, _dloss = compute_achieved(stats_ext, assign, specs,
                                   candidates=cands_ext)
    assert abs(ach - achieved) < 1e-9


def test_packed_super_item_name_keeps_expert_prefix_for_name_scoped_rules():
    """Packed super-item names keep the member prefix (....mlp.experts....),
    so a name-conditioned serving-profile rule (when.regex) binds identically
    at the per-member build_candidates gate and the post-aggregation profile
    filter — a rule written against member names cannot silently stop
    matching once the group becomes one DP unit."""
    from prismaquant.serving_profiles import ServingProfile

    stats, cands, expert_names, dense = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    _stats2, _costs2, cands2 = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    super_names = [n for n in cands2 if _PACKED_GROUP_MARKER in n]
    assert super_names

    profile = ServingProfile.from_dict({
        "schema": "prismaquant.serving_profile.v1",
        "id": "unit_expert_regex",
        "format_rules": [{
            "id": "expert_menu",
            "when": {"regex": r"\.mlp\.experts\."},
            "allow_formats": [_LOW],
            "reason": "profile_mismatch",
        }],
    })
    for name in expert_names + super_names:
        assert ".mlp.experts." in name
        assert profile.check_format(name, _LOW).legal
        assert not profile.check_format(name, _HIGH).legal, name
    # Non-expert rows are untouched by the scoped rule.
    assert profile.check_format(dense, _HIGH).legal


# ---------------------------------------------------------------------------
# Super-item stat self-consistency + UCB stderr aggregation
# ---------------------------------------------------------------------------

def test_super_item_stats_do_not_impersonate_one_member_shape():
    """The super item's n_params is the group SUM, so carrying members[0]'s
    (in_features, out_features) would make ``_shape_from_stats`` hand every
    byte fallback ONE member's shape for the whole group. The entry must not
    lie about its shape: without per-member features it falls back to the
    rank-1 (n_params,) view, and exact bytes stay available per format via
    ``_memory_bytes_by_format`` / the candidate."""
    from prismaquant.allocator_solver import _shape_from_stats

    stats, cands, expert_names, _ = _mk_moe_fixture()
    costs = {n: {} for n in stats}
    stats_ext, _, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)
    entry = stats_ext[super_name]
    n_params = sum(stats[m]["n_params"] for m in expert_names)
    assert entry["n_params"] == n_params
    assert _shape_from_stats(entry) == (n_params,), (
        "super-item shape must be total-parameter-consistent, not one "
        "member's (out, in)")
    for cand in cands_ext[super_name]:
        assert entry["_memory_bytes_by_format"][cand.fmt] == cand.memory_bytes


def test_group_ucb_stderr_preserves_conservative_bound(monkeypatch):
    """Packed members may share AURA probes. Without verified alignment,
    prices and grouped cost rows retain the conservative sum of stderrs."""
    from prismaquant.allocator_candidates import build_candidates

    z = 2.0
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", str(z))
    base = {_LOW: 0.05, _HIGH: 0.01}
    stderr = {_LOW: 0.004, _HIGH: 0.002}
    n_members = 4
    stats, costs = {}, {}
    for e in range(n_members):
        name = f"model.layers.0.mlp.experts.{e}.gate_proj"
        stats[name] = {"h_trace": 0.5, "n_params": 65536,
                       "in_features": 256, "out_features": 256,
                       "n_tokens_seen": 7}
        costs[name] = {
            f: {"predicted_dloss": base[f],
                "predicted_dloss_stderr": stderr[f]}
            for f in base
        }
    cands = build_candidates(stats, costs, _specs())
    _stats2, costs2, cands2 = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    super_name = next(n for n in cands2 if _PACKED_GROUP_MARKER in n)
    for fmt in base:
        agg = n_members * stderr[fmt]
        cand = next(c for c in cands2[super_name] if c.fmt == fmt)
        assert abs(cand.predicted_dloss
                   - (n_members * base[fmt] + z * agg)) < 1e-12
        entry = costs2[super_name][fmt]
        assert abs(entry["predicted_dloss"] - n_members * base[fmt]) < 1e-12
        assert abs(entry["predicted_dloss_stderr"] - agg) < 1e-12

    # z == 0 (default): bit-identical to the exact member-candidate sum.
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    cands0 = build_candidates(stats, costs, _specs())
    _stats0, costs0, cands0x = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands0, _PackedProfile())
    super0 = next(n for n in cands0x if _PACKED_GROUP_MARKER in n)
    for fmt in base:
        cand = next(c for c in cands0x[super0] if c.fmt == fmt)
        exact_sum = sum(next(c.predicted_dloss for c in cands0[name]
                             if c.fmt == fmt) for name in sorted(stats))
        assert cand.predicted_dloss.hex() == exact_sum.hex()
        agg = n_members * stderr[fmt]
        assert abs(costs0[super0][fmt]["predicted_dloss_stderr"] - agg) < 1e-12


def test_role_split_requires_profile_capability():
    """--packed-role-split is only legal when the resolved serving profile
    DECLARES per-role expert schemes: a role-split checkpoint carries
    different expert formats for gate_up vs down projections of one MoE
    layer, which vLLM's compressed-tensors packed-MoE path cannot load
    (one scheme per FusedMoE layer) but the GGUF lane can (expert tensors
    stack per projection)."""
    import pytest

    from prismaquant.serving_profiles import (
        load_serving_profile,
        require_per_role_expert_scheme_support,
    )

    # Capability declarations: only the GGUF lane opts in.
    assert load_serving_profile("gguf").supports_per_role_expert_schemes
    for profile_id in ("research", "vllm_packed_moe",
                       "vllm_qwen3_5_packed_moe"):
        assert not load_serving_profile(
            profile_id).supports_per_role_expert_schemes, profile_id

    # Supporting profile: accepted, resolved profile returned.
    prof = require_per_role_expert_scheme_support(
        "gguf", flag="--packed-role-split")
    assert prof.id == "gguf"

    # Non-supporting profile: hard error naming the flag, the profile,
    # and why (one scheme per FusedMoE layer).
    with pytest.raises(SystemExit) as exc:
        require_per_role_expert_scheme_support(
            "vllm_packed_moe", flag="--packed-role-split")
    msg = str(exc.value)
    assert "--packed-role-split" in msg
    assert "vllm_packed_moe" in msg
    assert "FusedMoE" in msg

    # Default (research) and unknown profiles: hard error, never a
    # silent pass.
    with pytest.raises(SystemExit):
        require_per_role_expert_scheme_support(None)
    with pytest.raises(SystemExit):
        require_per_role_expert_scheme_support("no_such_profile")


# ---------------------------------------------------------------------------
# A promoted-onto-illegal-format row must not be priced at zero Δloss
# ---------------------------------------------------------------------------

def test_compute_achieved_refuses_to_price_a_promoted_illegal_format():
    """A PRICED row whose assigned format has no candidate is a hard error.

    Serving-unit promotion can move a Linear onto a format its own candidate
    set never offered (only possible when a group's member menus are
    disjoint). Scoring that row at 0 Δloss made the unpriced state look
    CHEAPEST — precisely what ``solve_with_promotion``'s min-Δloss ratchet
    now selects on — and the error only surfaced later, in
    ``compute_assignment_predicted_dloss``. Fail at the miscosting instead.
    """
    import pytest

    stats, cands, expert_names, _dense = _mk_moe_fixture(n_experts=2)
    specs, _rank = _solver_env()
    crippled = expert_names[0]
    cands[crippled] = [c for c in cands[crippled] if c.fmt == _LOW]

    legal = {n: _LOW for n in cands}
    ach, dloss = compute_achieved(stats, legal, specs, candidates=cands)
    assert ach > 0.0 and dloss > 0.0

    promoted = dict(legal)
    promoted[crippled] = _HIGH  # what whole-group promotion would do
    with pytest.raises(AssertionError, match="no candidate exists to price"):
        compute_achieved(stats, promoted, specs, candidates=cands)

    # Names that are NOT DP rows keep the unpriced byte-only path (aux /
    # fixed-format rows, and the legacy candidates=None call form).
    ach_nc, dloss_nc = compute_achieved(stats, legal, specs)
    assert ach_nc > 0.0 and dloss_nc == 0.0


def test_empty_intersection_fallback_is_not_allocatable():
    """The empty-intersection fallback keeps the group visible in the DP, but
    it is NOT a repair: members draw from disjoint candidate lists, so
    whole-group promotion always lands on a format illegal for someone, and
    the solve refuses to price it at any target. Pins the corrected docstring
    (the old one claimed promotion would repair coherence).

    The verdict now comes from PROMOTION rather than from pricing: promotion is
    handed the per-row legal-format sets (issue #28), so it reports the empty
    intersection with the members and their sets instead of writing an illegal
    format and letting compute_achieved complain that it cannot be priced."""
    import pytest

    stats, cands, expert_names, _dense = _mk_moe_fixture(n_experts=2)
    cands[expert_names[0]] = [
        c for c in cands[expert_names[0]] if c.fmt == _LOW]
    cands[expert_names[1]] = [
        c for c in cands[expert_names[1]] if c.fmt == _HIGH]

    costs = {n: {} for n in stats}
    stats_ext, _costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(), cands, _PackedProfile())
    assert not any(_PACKED_GROUP_MARKER in n for n in cands_ext), (
        "empty-intersection group must not become a DP unit")

    specs, rank = _solver_env()
    low = fr.REGISTRY[_LOW].effective_bits
    high = fr.REGISTRY[_HIGH].effective_bits
    for target in (low + 0.05, high):
        with pytest.raises(AssertionError,
                           match="shares no format that is legal for every "
                                 "member") as exc:
            solve_with_promotion(
                stats_ext, cands_ext, target, specs, rank,
                bit_precision=0.001, profile=_PackedProfile())
        msg = str(exc.value)
        assert expert_names[0] in msg and expert_names[1] in msg
        assert "common legal formats: []" in msg
