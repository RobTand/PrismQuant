"""Tests for the allocator's fused-sibling pre-aggregation path and the
convergence-based iteration stop in `solve_with_promotion`.

Fused-sibling coupling (q/k/v must share one format, gate/up must share
one format) used to be enforced as a post-pass (`promote_fused`) that
inflated the achieved bpp above the target whenever the DP picked
different formats for siblings. The new `aggregate_fused_siblings`
pre-pass collapses each sibling group into a single DP candidate, so
the knapsack can't pick mixed-sibling solutions and the overshoot
vanishes.

These tests pin:

  - aggregate_fused_siblings respects the profile's fused_sibling_group
    and groups 2+-member groups while passing singletons through
  - The super-Linear's per-format predicted_dloss equals the exact sum
    of member predicted_dlosses (sibling aggregation already guarantees
    this mathematically; we replicate it here for siblings)
  - expand_fused_sibling_assignment broadcasts the super-Linear's
    chosen format back to every member
  - Pre-aggregation + DP achieves the target bit budget exactly, vs
    the post-promote pipeline which overshoots
  - solve_with_promotion stops early when consecutive iterations stall
"""
from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator_solver import promote_serving_units
from prismaquant.allocator import (
    Candidate,
    _FUSED_SIBLING_MARKER,
    _validate_assignment_candidate_membership,
    aggregate_fused_siblings,
    build_candidates,
    compute_achieved,
    expand_fused_sibling_assignment,
    promote_moe_pair,
    solve_with_promotion,
    validate_default_profile_format_menu,
    validate_final_serving_promotion_noop,
)
from prismaquant.model_profiles import DefaultProfile
from prismaquant.tessera_formats import is_tessera_group_option


# ---------------------------------------------------------------------------
# Minimal test fixture: a fake profile that knows about qkv_proj siblings
# ---------------------------------------------------------------------------
class _FakeProfile:
    """Profile stub: q/k/v at prefix P form one sibling group keyed by P.
    o_proj and standalone Linears get no group (passes through)."""

    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith(".q_proj") or name.endswith(".k_proj") or name.endswith(".v_proj"):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith(".gate_proj") or name.endswith(".up_proj"):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


class _FakePackedProfile:
    """Profile stub for a non-Qwen expert naming scheme.

    The old allocator regex only knew ``.experts`` paths. This profile proves
    serving-format coupling can now come from the model/profile layer.
    """

    def packed_expert_format_group(self, name: str) -> str | None:
        if name.startswith("blocks.0.router_bank.") and name.endswith(
            (".left", ".right")
        ):
            return "blocks.0.router_bank.lr"
        return None


class _OverlappingProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        return "overlap.fused" if name in {"overlap.left", "overlap.mid"} else None

    def packed_expert_format_group(self, name: str) -> str | None:
        return "overlap.moe" if name in {"overlap.mid", "overlap.right"} else None


def _mk_stats_and_costs():
    """Build a tiny 1-layer model: 3 qkv Linears + 1 o_proj + 2 gate/up + 1 down.
    Sizes are deliberately asymmetric so we can detect wrong aggregation."""
    layer = "model.layers.0"
    names = [
        f"{layer}.self_attn.q_proj",
        f"{layer}.self_attn.k_proj",
        f"{layer}.self_attn.v_proj",
        f"{layer}.self_attn.o_proj",
        f"{layer}.mlp.gate_proj",
        f"{layer}.mlp.up_proj",
        f"{layer}.mlp.down_proj",
    ]
    stats = {}
    costs = {}
    # Per-Linear: different h_trace values so predicted_dloss sums are
    # distinguishable, different shapes so params differ.
    shapes = {
        "q_proj": (4096, 4096),
        "k_proj": (1024, 4096),
        "v_proj": (1024, 4096),
        "o_proj": (4096, 4096),
        "gate_proj": (11008, 4096),
        "up_proj": (11008, 4096),
        "down_proj": (4096, 11008),
    }
    h_traces = {
        "q_proj": 0.5, "k_proj": 0.3, "v_proj": 0.7,
        "o_proj": 0.4, "gate_proj": 0.8, "up_proj": 0.6, "down_proj": 0.9,
    }
    for name in names:
        leaf = name.rsplit(".", 1)[1]
        d_out, d_in = shapes[leaf]
        stats[name] = {
            "h_trace": h_traces[leaf],
            "n_params": d_out * d_in,
            "in_features": d_in,
            "out_features": d_out,
        }
        # Mock per-format costs: NVFP4 is cheap but high Δloss,
        # BF16 is expensive but zero Δloss.
        costs[name] = {
            "NVFP4": {"weight_mse": 0.02, "predicted_dloss": 0.5 * h_traces[leaf] * 0.02},
            "BF16":  {"weight_mse": 0.0,  "predicted_dloss": 0.0},
        }
    return names, stats, costs


def _format_specs():
    return [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]


# ---------------------------------------------------------------------------
# aggregate_fused_siblings
# ---------------------------------------------------------------------------

def test_aggregation_groups_qkv_and_gate_up_only():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    # qkv → one super-Linear. gate+up → one super-Linear. o_proj + down_proj
    # stay on their own. 7 entries collapse to 4.
    supers = [n for n in cands_ext if _FUSED_SIBLING_MARKER in n]
    assert len(supers) == 2, f"expected 2 super-Linears (qkv, gate_up), got {supers}"
    assert len(cands_ext) == 4, (
        f"expected 4 total entries (2 super + o_proj + down_proj); got "
        f"{sorted(cands_ext)}"
    )

    # Per-leaf names that got aggregated should NOT appear in cands_ext.
    for leaf in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        aggregated_away = [n for n in cands_ext if n.endswith("." + leaf)]
        assert not aggregated_away, f"{leaf} leaked: {aggregated_away}"

    # o_proj and down_proj pass through.
    assert any(n.endswith(".o_proj") for n in cands_ext)
    assert any(n.endswith(".down_proj") for n in cands_ext)


def test_super_linear_predicted_dloss_is_sum_of_members():
    """Super-Linear's Δloss for format f must equal Σ per-sibling
    predicted_dloss. This matches the Fisher-diagonal cost model:
    Δloss decomposes additively over weights, therefore over Linears,
    therefore over siblings.

    (Earlier versions of this test pinned max×n aggregation — that was
    motivated by a contaminated perplexity measurement that has since
    been shown to be a validator bug, not a real breakage. Sum is
    the aggregation that matches the cost-model math. See the
    aggregate_fused_siblings docstring for the full rationale.)
    """
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    qkv_super = next(n for n in cands_ext if "qkv_proj" in n)
    qkv_members = [n for n in names if n.endswith((".q_proj", ".k_proj", ".v_proj"))]

    for c in cands_ext[qkv_super]:
        expected = sum(costs[m][c.fmt]["predicted_dloss"] for m in qkv_members)
        assert abs(c.predicted_dloss - expected) < 1e-9, (
            f"format {c.fmt}: super Δloss={c.predicted_dloss} "
            f"vs expected sum={expected}"
        )


def test_super_linear_uses_format_alias_cost_entries_and_gains():
    names, stats, costs = _mk_stats_and_costs()
    qkv_members = [n for n in names if n.endswith((".q_proj", ".k_proj", ".v_proj"))]
    for name in qkv_members:
        h = stats[name]["h_trace"]
        costs[name] = {
            "FP8_DYNAMIC": {
                "weight_mse": 0.01,
                "predicted_dloss": 0.5 * h * 0.01,
            },
            "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
        }
    specs = [fr.get_format("FP8_E4M3"), fr.REGISTRY["BF16"]]
    cands = build_candidates(
        stats,
        costs,
        specs,
        calibrated_gains={"FP8_DYNAMIC": 2.0},
    )

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats,
        costs,
        specs,
        cands,
        _FakeProfile(),
        calibrated_gains={"FP8_DYNAMIC": 2.0},
    )

    qkv_super = next(n for n in cands_ext if "qkv_proj" in n)
    assert "error" not in costs_ext[qkv_super]["FP8_E4M3"]
    cand = next(c for c in cands_ext[qkv_super] if c.fmt == "FP8_E4M3")
    expected = sum(
        costs[name]["FP8_DYNAMIC"]["predicted_dloss"]
        for name in qkv_members
    ) * 2.0
    assert abs(cand.predicted_dloss - expected) < 1e-12


def test_asymmetric_sensitivity_sums_contributions():
    """Concrete values for the asymmetric-sensitivity case: q and k
    insensitive (Δloss=1 each), v sensitive (Δloss=1000). The
    super-Linear at NVFP4 must see sum = 1002 (not max*3 = 3000).

    Note: this test intentionally does NOT assert "the DP picks BF16
    for this group." Whether the DP picks BF16 depends on the ratio
    of 1002 to the BF16 budget cost, which is a property of the whole
    knapsack — not of this one sibling group. The correct aggregation
    is sum; whether the DP then promotes to BF16 is a separate DP
    question driven by the total bit-budget. Safety against individual
    sensitive Linears being dropped to unsafe formats is a
    format-constraint concern, orthogonal to this aggregation."""
    import prismaquant.format_registry as fr
    from prismaquant.allocator import aggregate_fused_siblings, build_candidates
    layer = "model.layers.0"
    names_ = [f"{layer}.self_attn.q_proj",
              f"{layer}.self_attn.k_proj",
              f"{layer}.self_attn.v_proj"]
    stats = {}
    costs = {}
    dlosses = {"q_proj": 1.0, "k_proj": 1.0, "v_proj": 1000.0}
    for n in names_:
        leaf = n.rsplit(".", 1)[1]
        stats[n] = {"h_trace": 1.0, "n_params": 1024 * 1024,
                    "in_features": 1024, "out_features": 1024}
        costs[n] = {
            "NVFP4": {"weight_mse": dlosses[leaf] / 0.5,
                      "predicted_dloss": dlosses[leaf]},
            "BF16":  {"weight_mse": 0.0, "predicted_dloss": 0.0},
        }
    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]
    cands = build_candidates(stats, costs, specs)
    _, _, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _FakeProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)
    nvfp4_cand = next(c for c in cands_ext[super_name] if c.fmt == "NVFP4")
    assert nvfp4_cand.predicted_dloss == 1002.0, (
        f"sum-aggregation should give 1 + 1 + 1000 = 1002; "
        f"got {nvfp4_cand.predicted_dloss}"
    )


def test_expand_broadcasts_super_format_to_members():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    # Fake a DP assignment: pick BF16 for qkv_super, NVFP4 for gate_up super,
    # and NVFP4 for the singletons.
    assignment = {}
    for n in cands_ext:
        if "qkv_proj" in n:
            assignment[n] = "BF16"
        else:
            assignment[n] = "NVFP4"

    expanded = expand_fused_sibling_assignment(assignment, stats_ext)

    # Every q/k/v should get BF16; gate/up should get NVFP4.
    for m in names:
        if m.endswith((".q_proj", ".k_proj", ".v_proj")):
            assert expanded[m] == "BF16", f"{m} should be BF16, got {expanded[m]}"
        elif m.endswith((".gate_proj", ".up_proj")):
            assert expanded[m] == "NVFP4", f"{m} should be NVFP4, got {expanded[m]}"
        else:
            assert expanded[m] == "NVFP4"
    # The super-Linear markers should NOT appear in the expanded output.
    assert not any(_FUSED_SIBLING_MARKER in n for n in expanded)


def test_singleton_groups_pass_through_unchanged():
    """A profile reporting a group key for only ONE member should NOT
    aggregate — there's no benefit, and aggregation would change the
    entry name."""
    class _SingletonProfile:
        def fused_sibling_group(self, name):
            # Only q_proj gets a group key; k_proj and v_proj get None.
            # Even though the key is non-None for q_proj, it's a singleton
            # because no other member shares it.
            return name.rsplit(".", 1)[0] + ".solo" if name.endswith(".q_proj") else None

    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _SingletonProfile())
    # No `__siblings__` markers should appear — the one q_proj group had
    # only one member.
    assert not any(_FUSED_SIBLING_MARKER in n for n in cands_ext)
    assert "model.layers.0.self_attn.q_proj" in cands_ext


def test_fused_group_with_disjoint_member_menus_hard_errors():
    """A fused group whose members share NO legal format must raise, not
    vanish. Pre-fix, ``cands`` came out empty, ``stats_ext``/``costs_ext``
    still got the super item and ``candidates_ext`` did not — so the whole
    q/k/v group silently disappeared from the DP and was never assigned a
    format.

    Fused siblings must load under ONE format, so this state is not
    allocatable at all (any promoted format is illegal for some member, and
    ``compute_achieved`` refuses to price that) — the fallback-to-individual-
    rows treatment ``aggregate_packed_serving_groups`` gives packed groups
    would only defer the same failure past the solve. The disjointness here is
    built through the REAL legality gate, from the two upstream causes the
    message names: a passthrough-source mismatch on one sibling and an
    applicability mask on the other."""
    import pytest

    layer = "model.layers.0.self_attn"
    q, k = f"{layer}.q_proj", f"{layer}.k_proj"
    stats = {
        # q_proj: FP16 source -> BF16 passthrough illegal (never synthesize),
        # while NVFP4 remains below the source rate.
        q: {"h_trace": 0.5, "n_params": 4096 * 4096,
            "in_features": 4096, "out_features": 4096},
        # k_proj: in_features not divisible by the NVFP4 group -> masked.
        k: {"h_trace": 0.3, "n_params": 1024 * 4098,
            "in_features": 4098, "out_features": 1024},
    }
    costs = {
        n: {"NVFP4": {"weight_mse": 0.02, "predicted_dloss": 0.01},
            "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0}}
        for n in stats
    }
    specs = _format_specs()
    cands = build_candidates(
        stats, costs, specs,
        source_manifest={q: "f16", k: "bf16"},
    )
    assert [c.fmt for c in cands[q]] == ["NVFP4"]
    assert [c.fmt for c in cands[k]] == ["BF16"]

    with pytest.raises(AssertionError) as exc:
        aggregate_fused_siblings(stats, costs, specs, cands, _FakeProfile())
    msg = str(exc.value)
    assert "qkv_proj" in msg                      # the group
    assert q in msg and k in msg                  # its members
    assert "['NVFP4']" in msg and "['BF16']" in msg   # each member's menu
    assert "common formats: []" in msg
    for cause in ("cost row", "applicability mask", "passthrough-source"):
        assert cause in msg, cause


def test_aggregation_is_no_op_without_profile():
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile=None)
    assert cands_ext is cands or cands_ext == cands


def test_promote_moe_pair_uses_profile_format_groups():
    assignment = {
        "blocks.0.router_bank.left": "NVFP4",
        "blocks.0.router_bank.right": "BF16",
        "blocks.0.router_bank.other": "NVFP4",
    }
    promoted = promote_moe_pair(
        assignment,
        {"NVFP4": 0, "BF16": 1},
        profile=_FakePackedProfile(),
    )
    assert promoted["blocks.0.router_bank.left"] == "BF16"
    assert promoted["blocks.0.router_bank.right"] == "BF16"
    assert promoted["blocks.0.router_bank.other"] == "NVFP4"


def test_promote_moe_pair_uses_default_profile_common_packed_groups():
    assignment = {
        "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
        "model.layers.0.mlp.experts.down_proj": "BF16",
        "model.layers.1.mlp.experts.0.gate_proj": "NVFP4",
        "model.layers.1.mlp.experts.0.up_proj": "BF16",
        "model.layers.1.mlp.experts.0.down_proj": "NVFP4",
        "model.layers.2.mlp.experts.0.w1": "NVFP4",
        "model.layers.2.mlp.experts.0.w2": "BF16",
        "model.layers.2.mlp.experts.0.w3": "NVFP4",
    }

    promoted = promote_moe_pair(assignment, {"NVFP4": 0, "BF16": 1})

    assert promoted["model.layers.0.mlp.experts.gate_up_proj"] == "BF16"
    assert promoted["model.layers.0.mlp.experts.down_proj"] == "BF16"
    assert promoted["model.layers.1.mlp.experts.0.gate_proj"] == "BF16"
    assert promoted["model.layers.1.mlp.experts.0.up_proj"] == "BF16"
    assert promoted["model.layers.1.mlp.experts.0.down_proj"] == "BF16"
    assert promoted["model.layers.2.mlp.experts.0.w1"] == "BF16"
    assert promoted["model.layers.2.mlp.experts.0.w2"] == "BF16"
    assert promoted["model.layers.2.mlp.experts.0.w3"] == "BF16"


def test_serving_unit_promotion_handles_overlapping_groups_order_independently():
    promoted = promote_serving_units(
        {
            "overlap.left": "NVFP4",
            "overlap.mid": "MXFP8_E4M3",
            "overlap.right": "BF16",
        },
        {"NVFP4": 0, "MXFP8_E4M3": 1, "BF16": 2},
        profile=_OverlappingProfile(),
    )

    assert promoted == {
        "overlap.left": "BF16",
        "overlap.mid": "BF16",
        "overlap.right": "BF16",
    }


def test_assignment_candidate_membership_rejects_unavailable_promoted_format():
    candidates = {
        "expert.gate": [Candidate("NVFP4", 4.5, 9, 0.1)],
        "expert.down": [Candidate("BF16", 16.0, 32, 0.0)],
    }
    assignment = {
        "expert.gate": "BF16",
        "expert.down": "BF16",
        "mtp.proj": "NVFP4",
    }
    fixed = {"mtp.proj": Candidate("NVFP4", 4.5, 9, 0.1)}

    try:
        _validate_assignment_candidate_membership(
            assignment,
            candidates,
            fixed_chosen_candidates=fixed,
        )
    except SystemExit as exc:
        assert "expert.gate" in str(exc)
        assert "mtp.proj" not in str(exc)
    else:
        raise AssertionError("expected promoted unavailable format to fail")


def test_default_profile_rejects_multi_format_menu_without_escape():
    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]]

    try:
        validate_default_profile_format_menu(DefaultProfile(), specs)
    except SystemExit as exc:
        assert "DefaultProfile only enforces its fallback fused groups" in str(exc)
    else:
        raise AssertionError("multi-format DefaultProfile menu should fail")


def test_default_profile_allows_single_format_or_explicit_escape():
    validate_default_profile_format_menu(
        DefaultProfile(),
        [fr.REGISTRY["NVFP4"]],
    )
    validate_default_profile_format_menu(
        DefaultProfile(),
        [fr.REGISTRY["NVFP4"], fr.REGISTRY["BF16"]],
        allow_default_profile=True,
    )


def test_final_serving_promotion_must_be_noop_for_accounting():
    validate_final_serving_promotion_noop({"a": "NVFP4"}, {"a": "NVFP4"})

    try:
        validate_final_serving_promotion_noop(
            {"a": "NVFP4"},
            {"a": "BF16"},
        )
    except SystemExit as exc:
        assert "achieved_bits/Delta-loss" in str(exc)
        assert "a" in str(exc)
    else:
        raise AssertionError("changed final promotion should fail")


# ---------------------------------------------------------------------------
# End-to-end: pre-aggregation hits target without overshoot
# ---------------------------------------------------------------------------

def test_pre_aggregation_respects_budget_without_overshoot():
    """With siblings pre-aggregated, the DP's solution is already
    sibling-consistent. The promote_fused post-pass becomes a no-op,
    and solve_with_promotion returns on iteration 1 with achieved ≤ target."""
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    # Aggregate first.
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile)

    format_specs = {s.name: s for s in specs}
    format_rank = {"NVFP4": 0, "BF16": 1}

    target = 8.0  # something mid-range between NVFP4 (~4.5) and BF16 (16)
    assignment, achieved = solve_with_promotion(
        stats_ext, cands_ext, target,
        format_specs, format_rank,
        bit_precision=0.001,
        profile=profile,
    )
    assert assignment is not None
    assert achieved <= target + 0.01, (
        f"aggregated path should hit budget: target={target}, achieved={achieved}"
    )


# ---------------------------------------------------------------------------
# Convergence-based stopping
# ---------------------------------------------------------------------------

def test_solve_with_promotion_returns_feasible_iterate_on_plateau():
    """Construct a tiny problem where promotion overshoots and tightening
    plateaus (serving-group atomicity pins the promoted outcome).
    Termination contract: the loop returns the best FEASIBLE iterate
    (achieved <= target + tolerance) or (None, nan) when nothing ever fit —
    it must never fabricate an over-target 'feasible' result."""
    # Use the un-aggregated path so promote_fused has siblings to coerce.
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    profile = _FakeProfile()

    format_specs = {s.name: s for s in specs}
    format_rank = {"NVFP4": 0, "BF16": 1}

    # Target just above the NVFP4 floor. promote_fused may bump q/k/v to
    # BF16 if the DP picks mixed formats, overshooting; tightening can
    # only push until we hit the floor. The stall guard caps iteration.
    target = 4.6
    assignment, achieved = solve_with_promotion(
        stats, cands, target, format_specs, format_rank,
        bit_precision=0.001,
        stall_threshold=1e-3,
        stall_grace=2,
        max_iters=40,
        profile=profile,
    )
    # The all-NVFP4 floor (~4.5) fits under target=4.6, so a feasible
    # iterate exists and must be returned — and it must actually be
    # feasible, not an over-target fabrication.
    assert assignment is not None
    assert isinstance(achieved, float)
    assert achieved <= target + 0.01, (
        f"stall exit returned an over-target iterate: target={target}, "
        f"achieved={achieved}")


# ---------------------------------------------------------------------------
# Ratchet objective: min predicted Δloss among feasible iterates
# ---------------------------------------------------------------------------

def _mk_single_tensor_ratchet_fixture(dloss_by_fmt: dict[str, float]):
    """One 1000-param tensor with candidates at 4/5/6/7 bpp.

    predicted_dloss per format comes from ``dloss_by_fmt`` so tests can
    make Δloss(bits) deliberately non-monotone (more bits is NOT always
    better — the CLAUDE.md §5 5.5-vs-6.0 bpp lesson).
    """
    stats = {"w": {"n_params": 1000}}
    bpp = {"F4": 4.0, "F5": 5.0, "F6": 6.0, "F7": 7.0}
    cands = {
        "w": [
            Candidate(fmt=f, bits_per_param=b,
                      memory_bytes=int(b * 1000 / 8),
                      predicted_dloss=dloss_by_fmt[f])
            for f, b in bpp.items()
        ]
    }
    return stats, cands


def _scripted_solve_allocation(fmt_for_t):
    """Stand-in for solve_allocation: pick a format purely from the
    tightened target, emulating promotion-driven non-monotonicity
    without needing a real coupled model."""
    def fake(stats, candidates, t, bit_precision):
        return {"w": fmt_for_t(t)}, None
    return fake


def test_ratchet_keeps_min_dloss_feasible_iterate_over_denser(monkeypatch):
    """A feasible lower-Δloss iterate must beat a denser higher-Δloss one.

    Script: t=6.0 -> F7 (7.0 bits, overshoots); tightened t=5.5 -> F5
    (5.0 bits, Δloss 1.0, feasible); bisection mid t=5.75 -> F6
    (6.0 bits, Δloss 3.0, feasible and denser but WORSE). The old
    max-achieved ratchet shipped F6; the min-Δloss ratchet must ship F5.
    """
    from prismaquant import allocator_solver as als

    stats, cands = _mk_single_tensor_ratchet_fixture(
        {"F4": 5.0, "F5": 1.0, "F6": 3.0, "F7": 0.5})

    def fmt_for_t(t):
        if t >= 5.9:
            return "F7"
        if t >= 5.6:
            return "F6"
        return "F5"

    monkeypatch.setattr(als, "solve_allocation",
                        _scripted_solve_allocation(fmt_for_t))

    assignment, achieved = solve_with_promotion(
        stats, cands, 6.0, {}, {},
        bit_precision=0.001,
        profile=None,
    )
    assert assignment == {"w": "F5"}, (
        f"ratchet must keep the min-Δloss feasible iterate (F5, Δloss 1.0) "
        f"over the denser F6 (Δloss 3.0); got {assignment}")
    assert abs(achieved - 5.0) < 1e-9
    # Feasibility contract is unchanged.
    assert achieved <= 6.0 + 0.01


def test_ratchet_tie_breaks_toward_denser_iterate(monkeypatch):
    """Equal Δloss: keep the denser (higher achieved bits) iterate."""
    from prismaquant import allocator_solver as als

    stats, cands = _mk_single_tensor_ratchet_fixture(
        {"F4": 5.0, "F5": 1.0, "F6": 1.0, "F7": 0.5})

    def fmt_for_t(t):
        if t >= 5.9:
            return "F7"
        if t >= 5.6:
            return "F6"
        return "F5"

    monkeypatch.setattr(als, "solve_allocation",
                        _scripted_solve_allocation(fmt_for_t))

    assignment, achieved = solve_with_promotion(
        stats, cands, 6.0, {}, {},
        bit_precision=0.001,
        profile=None,
    )
    assert assignment == {"w": "F6"}, (
        f"on a Δloss tie the denser feasible iterate wins; got {assignment}")
    assert abs(achieved - 6.0) < 1e-9
    assert achieved <= 6.0 + 0.01


# ---------------------------------------------------------------------------
# Termination contract: FEASIBLE iterate or INFEASIBLE — never a silent
# over-target return
# ---------------------------------------------------------------------------

def test_infeasible_rung_returns_none_and_nan(monkeypatch):
    """A rung where NO iterate ever fits must report INFEASIBLE.

    Regression guard for the three silent fallbacks: returning the previous
    over-target iterate, accepting an arbitrarily deep undershoot, and the
    stall exit handing back an iterate still far above target. Callers
    (Pareto curve, byte-budget bisection) key off `assignment is None`, so a
    restored silent over-target return must fail here.
    """
    import math

    from prismaquant import allocator_solver as als

    stats, cands = _mk_single_tensor_ratchet_fixture(
        {"F4": 5.0, "F5": 1.0, "F6": 3.0, "F7": 0.5})
    # Every tightened target promotes to the 7.0-bit format: nothing fits
    # under target=6.0 + 0.01, at any tightening, all the way to the floor.
    monkeypatch.setattr(als, "solve_allocation",
                        _scripted_solve_allocation(lambda t: "F7"))

    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, cands, 6.0, {}, {},
        bit_precision=0.001,
        profile=None,
        diagnostics=diag,
    )
    assert assignment is None, (
        f"infeasible rung must not fabricate an assignment; got {assignment}")
    assert math.isnan(achieved), (
        f"infeasible rung must report nan achieved bits; got {achieved}")
    # The diagnostics the caller needs to write an actionable message.
    assert diag["feasible"] is False
    assert diag["achieved_bits"] is None
    assert abs(diag["min_bits"] - 4.0) < 1e-9
    assert abs(diag["closest_achieved_bits"] - 7.0) < 1e-9
    assert abs(diag["floor_achieved_bits"] - 7.0) < 1e-9
    assert diag["evals"] >= 1


def test_below_format_floor_target_is_infeasible():
    """A target under the cheapest legal assignment is INFEASIBLE, and the
    floor is reported so the caller can say what target WOULD work."""
    import math

    stats, cands = _mk_single_tensor_ratchet_fixture(
        {"F4": 5.0, "F5": 1.0, "F6": 3.0, "F7": 0.5})
    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, cands, 3.0, {}, {},
        bit_precision=0.001,
        profile=None,
        diagnostics=diag,
    )
    assert assignment is None
    assert math.isnan(achieved)
    assert abs(diag["min_bits"] - 4.0) < 1e-9
    assert diag["feasible"] is False


def test_feasible_rung_satisfies_the_tolerance_contract(monkeypatch):
    """The other half of the contract: whenever an assignment IS returned it
    is feasible, and the diagnostics describe that returned iterate."""
    from prismaquant import allocator_solver as als

    stats, cands = _mk_single_tensor_ratchet_fixture(
        {"F4": 5.0, "F5": 1.0, "F6": 3.0, "F7": 0.5})

    def fmt_for_t(t):
        return "F7" if t >= 5.9 else "F5"

    monkeypatch.setattr(als, "solve_allocation",
                        _scripted_solve_allocation(fmt_for_t))

    target, tol = 6.0, 0.01
    diag = {}
    assignment, achieved = solve_with_promotion(
        stats, cands, target, {}, {},
        bit_precision=0.001,
        overshoot_tolerance=tol,
        profile=None,
        diagnostics=diag,
    )
    assert assignment is not None
    assert achieved <= target + tol
    assert diag["feasible"] is True
    assert abs(diag["achieved_bits"] - achieved) < 1e-12
    assert abs(diag["predicted_dloss"] - 1.0) < 1e-12  # F5
    # The over-target iterate that was seen and rejected is still reported.
    assert abs(diag["closest_achieved_bits"] - 7.0) < 1e-9


# ---------------------------------------------------------------------------
# UCB hedge on fused super-items: independence aggregate, not linear sum
# ---------------------------------------------------------------------------

def _mk_qkv_ucb_fixture(base: dict, stderr: dict):
    layer = "model.layers.0.self_attn"
    members = sorted(f"{layer}.{p}" for p in ("q_proj", "k_proj", "v_proj"))
    stats, costs = {}, {}
    for name in members:
        stats[name] = {"h_trace": 0.5, "n_params": 65536,
                       "in_features": 256, "out_features": 256,
                       "n_tokens_seen": 7}
        costs[name] = {
            f: {"predicted_dloss": base[f], "predicted_dloss_stderr": stderr[f]}
            for f in base
        }
    return members, stats, costs


def test_fused_group_ucb_stderr_aggregates_in_quadrature(monkeypatch):
    """A qkv triple must hedge at √3-independence, not 3x linear.

    Members' dloss estimates are independent measurements, so the stderr of
    the group SUM is sqrt(Σ stderr²). Summing per-member UCB'd terms charged
    the LINEAR z·Σ(stderr) — a √N over-hedge — and the super cost entry
    dropped predicted_dloss_stderr entirely, zeroing the hedge for consumers
    of the aggregated table. Same contract the packed path already pins.
    """
    z = 2.0
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", str(z))
    base = {"NVFP4": 0.05, "BF16": 0.01}
    stderr = {"NVFP4": 0.004, "BF16": 0.002}
    members, stats, costs = _mk_qkv_ucb_fixture(base, stderr)
    specs = _format_specs()

    cands = build_candidates(stats, costs, specs)
    _s, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _FakeProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)
    n_members = len(members)
    sum_h = sum(stats[m]["h_trace"] for m in members)

    checked = 0
    for cand in cands_ext[super_name]:
        fmt = cand.fmt
        assert fmt in base, f"unexpected format {fmt} in fixture"
        agg = (n_members * stderr[fmt] ** 2) ** 0.5
        exact_sum = 0.0
        for _ in members:
            exact_sum += base[fmt]
        assert abs(cand.predicted_dloss - (exact_sum + z * agg)) < 1e-12
        entry = costs_ext[super_name][fmt]
        assert abs(entry["predicted_dloss"] - exact_sum) < 1e-12
        assert abs(entry["predicted_dloss_stderr"] - agg) < 1e-12
        # weight_mse is derived from the UN-hedged sum (no z contamination).
        assert abs(entry["weight_mse"] - exact_sum / (0.5 * sum_h)) < 1e-12
        # Strictly cheaper than the linear hedge it replaces.
        assert z * agg < z * n_members * stderr[fmt] - 1e-12
        checked += 1
    assert checked >= 1


def test_fused_group_hedge_is_identity_at_z_zero(monkeypatch):
    """At the default z == 0 the fused super-item price stays bit-for-bit the
    exact accumulated member sum — the pre-fix formula."""
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "0")
    base = {"NVFP4": 0.05, "BF16": 0.01}
    stderr = {"NVFP4": 0.004, "BF16": 0.002}
    members, stats, costs = _mk_qkv_ucb_fixture(base, stderr)
    specs = _format_specs()

    cands = build_candidates(stats, costs, specs)
    _s, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _FakeProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)

    for cand in cands_ext[super_name]:
        exact_sum = 0.0
        for _ in members:
            exact_sum += base[cand.fmt]
        assert cand.predicted_dloss == exact_sum, (
            "z == 0 must be a bit-for-bit identity on the member sum")
        assert costs_ext[super_name][cand.fmt]["predicted_dloss"] == exact_sum


# ---------------------------------------------------------------------------
# One family per group, a rate per member: the group's exact knapsack
# ---------------------------------------------------------------------------

@pytest.fixture
def tessera_dev_pin(monkeypatch):
    """Turn the Tessera development pin on for one test.

    Never autouse: without it ``fused_module_licence()`` is ``None`` and the
    group knapsack declines to enumerate a rate per member, which is
    production and is what
    ``test_the_fold_declines_when_no_runtime_contract_is_pinned`` asserts. A
    leaked pin would silently invert that.
    """
    from prismaquant import tessera_runtime_contract as trc

    monkeypatch.setenv(trc.TESSERA_DEV_PIN_ENV, trc.TESSERA_DEV_PIN_COMMIT)
    return trc.TESSERA_DEV_PIN_COMMIT


def _installed_fused_licence():
    """The fused-module licence the INSTALLED Tessera actually publishes.

    Parsed from the packaged ``runtime_contract.json``'s own bytes, through
    the same reader ``load_tessera_contract`` calls, so these tests fold under
    the table on this box rather than under a licence a test author typed. A
    hand-written licence would let the fold's tests pass while the contract
    said something else, which is the failure this whole change is about.
    """
    import hashlib

    from prismaquant import tessera_runtime_contract as trc

    path = trc.contract_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return trc._load_at(str(path), sha, "installed").fused_module


def _licence_with(**overrides):
    """The installed licence with some ``fields`` licences overridden.

    ``None`` as a value DELETES the field, which models a contract that goes
    silent about it -- a different thing from one that marks it ``shared``,
    and both have to refuse.
    """
    from dataclasses import replace

    base = _installed_fused_licence()
    fields = dict(base.fields)
    for name, licence in overrides.items():
        if licence is None:
            fields.pop(name, None)
        else:
            fields[name] = licence
    return replace(base, fields=fields)


def _tessera_member_candidates(rows, family="TESSERA_E4M3_K1"):
    """``[(bytes, cost), ...]`` -> candidates on one family."""
    return [
        Candidate(fmt=f"{family}_R{100 + i}", bits_per_param=0.0,
                  memory_bytes=int(b), predicted_dloss=float(c))
        for i, (b, c) in enumerate(rows)
    ]


def test_group_knapsack_equals_brute_force_including_a_nonconvex_pocket():
    """The fold must be the group's EXACT multi-choice knapsack.

    Dominance pruning, not a convex hull: the budget is discrete, so an option
    strictly inside the hull can still be the unique optimum at one particular
    remaining capacity, and hull pruning drops exactly those. The two menus
    below are built with such a pocket, and the fold is checked against brute
    force over every one of the 36 member combinations.
    """
    from prismaquant.allocator_candidates import tessera_group_composites

    # Member A carries a pocket: (30, 5.0) is inside the hull of (20, 9.0)
    # and (40, 2.0) but is not dominated by either.
    a = [(10, 20.0), (20, 9.0), (30, 5.0), (40, 2.0), (50, 1.5), (60, 1.0)]
    b = [(10, 18.0), (20, 8.0), (30, 4.5), (40, 2.2), (50, 1.4), (60, 0.9)]
    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        members[0]: _tessera_member_candidates(a),
        members[1]: _tessera_member_candidates(b),
    }
    options = tessera_group_composites(
        members, candidates, n_params=1000,
        licence=_installed_fused_licence())

    brute: dict[int, float] = {}
    for ab, ac in a:
        for bb, bc in b:
            total_b, total_c = ab + bb, ac + bc
            if total_b not in brute or total_c < brute[total_b]:
                brute[total_b] = total_c
    # The Pareto set of the brute-force sum, by the same dominance rule.
    want = []
    best = None
    for total_b in sorted(brute):
        if best is None or brute[total_b] < best:
            want.append((total_b, brute[total_b]))
            best = brute[total_b]

    got = sorted((int(o.memory_bytes), round(float(o.predicted_dloss), 12))
                 for o in options)
    assert got == [(b, round(c, 12)) for b, c in want]
    # Every option carries the rung each member holds, and one family.
    for option in options:
        assert set(option.member_formats) == set(members)
        assert {f.rsplit("_R", 1)[0] for f in option.member_formats.values()} \
            == {"TESSERA_E4M3_K1"}


def test_the_group_knapsack_beats_one_shared_rung_where_it_should():
    """The whole point: members with different sensitivities want different
    rates. With A steep and B flat, the best equal-byte split is asymmetric,
    and the one-rung menu cannot express it."""
    from prismaquant.allocator_candidates import tessera_group_composites

    a = [(10, 100.0), (20, 40.0), (30, 10.0)]     # steep: bytes buy a lot
    b = [(10, 6.0), (20, 5.0), (30, 4.0)]         # flat: bytes buy little
    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        members[0]: _tessera_member_candidates(a),
        members[1]: _tessera_member_candidates(b),
    }
    options = {int(o.memory_bytes): o
               for o in tessera_group_composites(
                   members, candidates, n_params=1000,
                   licence=_installed_fused_licence())}
    # At 40 bytes the uniform split (20, 20) costs 45.0; the group knapsack
    # spends them where they buy the most: (30, 10) costs 16.0.
    assert options[40].predicted_dloss == 16.0
    assert options[40].member_formats["m.q_proj"].endswith("_R102")
    assert options[40].member_formats["m.k_proj"].endswith("_R100")


def test_group_options_refuse_a_ucb_hedge_rather_than_price_it_wrong():
    """``z*sqrt(sum stderr^2)`` is not additive, so the fold cannot carry it
    on two coordinates. Refuse, do not approximate."""
    import pytest

    from prismaquant.allocator_candidates import tessera_group_composites

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: _tessera_member_candidates([(10, 2.0), (20, 1.0)]) for m in members
    }
    with pytest.raises(NotImplementedError, match="not additive"):
        tessera_group_composites(
            members, candidates, n_params=1000,
            licence=_installed_fused_licence(), ucb_z=1.0)


def test_expansion_gives_each_member_its_own_rung(tessera_dev_pin):
    """A whole-group option is not a format name and must never be broadcast.
    Expansion reads the per-member map the aggregation wrote beside it."""
    stats = {
        "model.layers.0.self_attn.q_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 4,
            "out_features": 4},
        "model.layers.0.self_attn.k_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 4,
            "out_features": 4},
    }
    members = sorted(stats)
    costs = {n: {} for n in stats}
    candidates = {
        members[0]: _tessera_member_candidates([(10, 4.0), (20, 1.0)]),
        members[1]: _tessera_member_candidates([(10, 3.0), (20, 2.0)]),
    }
    stats_ext, _costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, [], candidates, _FakeProfile())
    super_name = next(n for n in cands_ext if _FUSED_SIBLING_MARKER in n)
    option = next(c for c in cands_ext[super_name]
                  if int(c.memory_bytes) == 30)
    expanded = expand_fused_sibling_assignment(
        {super_name: option.fmt}, stats_ext)
    assert expanded == option.member_formats
    assert len(set(expanded.values())) == 2      # a rate per member


def test_a_group_option_without_a_member_map_is_refused():
    import pytest

    stats_ext = {
        f"m{_FUSED_SIBLING_MARKER}g": {"_fused_siblings": ["m.q", "m.k"]},
    }
    with pytest.raises(AssertionError, match="per-member rung map"):
        expand_fused_sibling_assignment(
            {f"m{_FUSED_SIBLING_MARKER}g": "TESSERA_E4M3_K1_G3"}, stats_ext)


def test_a_stock_menu_gains_no_group_options_at_all():
    """A run with no Tessera rung on the menu must be byte-identical to one
    built before the group knapsack existed."""
    names, stats, costs = _mk_stats_and_costs()
    specs = _format_specs()
    cands = build_candidates(stats, costs, specs)
    stats_ext, _costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, _FakeProfile())
    for name, per_name in cands_ext.items():
        for candidate in per_name:
            assert candidate.member_formats is None, (name, candidate.fmt)
        if _FUSED_SIBLING_MARKER in name:
            assert "_fused_member_formats" not in stats_ext[name]
            # And no licence receipt either: a group whose members carry only
            # stock formats never asks what the runtime frees, so it must not
            # carry an answer -- least of all on an unpinned run, where the
            # answer would be "no contract is pinned" written onto every fused
            # group of every production run (#132).
            assert "_tessera_group_menu" not in stats_ext[name], (
                name, stats_ext[name].get("_tessera_group_menu"))


def test_promotion_leaves_a_mixed_rung_group_alone():
    """The serving constraint is the shared decoder, not the shared rate: an
    expanded mixed-rung group is already coherent and promotion must not
    'repair' it into one rung."""
    assignment = {
        "model.layers.0.self_attn.q_proj": "TESSERA_E4M3_K1_R694",
        "model.layers.0.self_attn.k_proj": "TESSERA_E4M3_K1_R920",
        "model.layers.0.self_attn.v_proj": "TESSERA_E4M3_K1_R1372",
    }
    format_rank = {
        "TESSERA_E4M3_K1_R694": 0,
        "TESSERA_E4M3_K1_R920": 1,
        "TESSERA_E4M3_K1_R1372": 2,
    }
    legal = {name: set(format_rank) for name in assignment}
    out = promote_serving_units(
        dict(assignment), format_rank, profile=_FakeProfile(),
        legal_formats=legal)
    assert out == assignment


# ---------------------------------------------------------------------------
# The per-member rung licence is the CONTRACT's, not a docstring (pq #132)
#
# ``tessera_group_composites`` gives each member of a fused group its own rung.
# That is a claim about what the serving runtime's loader accepts, and it used
# to be carried in prose. Tessera's ``runtime_contract.json`` publishes it
# machine-readably (``fused_module.fields``, checked on the Tessera side
# against ``scheme.FUSED_MODULE_FIELDS``, the dict the loader itself gates on,
# RobTand/tessera#37 at contract v6). These tests are what fails if the
# allocator keeps enumerating per-member composites after the contract stops
# licensing them -- in either direction, because the same read narrows the fold
# as well as opening it.
# ---------------------------------------------------------------------------

def test_the_fold_declines_when_no_runtime_contract_is_pinned():
    """No table, no licence -- and the absence of a statement is not a yes.

    Production is exactly this state: no Tessera RELEASE tag exists, so
    ``tessera_menu.fused_module_licence()`` is ``None``. The fold must then
    offer nothing and say why, rather than fall back on the reading its own
    docstring used to assert (RobTand/prismaquant#132).
    """
    from prismaquant.allocator_candidates import (
        FUSED_LICENCE_UNPINNED, tessera_group_composites,
    )

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: _tessera_member_candidates([(10, 2.0), (20, 1.0)]) for m in members
    }
    report: dict = {}
    options = tessera_group_composites(
        members, candidates, n_params=1000, licence=None, report=report)
    assert options == []
    stamp = report["__licence__"]
    assert stamp["folded"] is False
    assert stamp["q256"] == FUSED_LICENCE_UNPINNED
    assert stamp["contract_shared_fields"] == []
    assert stamp["partitioned_on"] == []
    assert "no Tessera runtime contract is pinned" in stamp["note"]


def test_withdrawing_the_per_member_rate_licence_stops_the_fold():
    """**THE regression test this issue asked for.**

    The licence for a rate per member is
    ``fused_module.fields.q256 == "per_member"``, published by the runtime
    that loads the module (Tessera contract v6, RobTand/tessera#37). If a
    later contract re-tightens it to ``shared``, an allocator that kept
    enumerating per-member composites would allocate rungs the exporter
    refuses, and nothing would raise until export. So: same members, same
    menus, one field moved in the licence, and the mixed-rung options must be
    gone.
    """
    from prismaquant.allocator_candidates import tessera_group_composites

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        members[0]: _tessera_member_candidates([(10, 9.0), (20, 1.0)]),
        members[1]: _tessera_member_candidates([(10, 5.0), (20, 4.5)]),
    }
    licensed = tessera_group_composites(
        members, candidates, n_params=1000,
        licence=_installed_fused_licence())
    assert [o for o in licensed if len(set(o.member_formats.values())) > 1], (
        "the control arm must actually contain a mixed-rung option, or the "
        "withdrawal below proves nothing")

    report: dict = {}
    withdrawn = tessera_group_composites(
        members, candidates, n_params=1000,
        licence=_licence_with(q256="shared"), report=report)
    assert withdrawn == [], (
        "the contract withdrew the per-member rate licence and the fold kept "
        "enumerating per-member composites")
    stamp = report["__licence__"]
    assert stamp["q256"] == "shared"
    assert stamp["folded"] is False
    assert "q256" in stamp["contract_shared_fields"]
    assert "q256" in stamp["partitioned_on"]


def test_one_family_is_not_one_decoder_so_two_bodies_never_share_a_module():
    """``body`` is ``shared`` and the family does not fix it.

    ``tessera.export.wire_recipe`` is a function of ``(grid, q256)``:
    ``E2M1x2`` writes a WINDOW body at every rung below the coset cap and TCQ
    at 896. So "one family per group" -- the constraint the fold's docstring
    used to assert -- is strictly weaker than the contract's ``shared`` set,
    and a family-only fold offers the DP a module with two bodies in it. The
    numbers below are chosen so that the illegal pair sits ON the Pareto
    frontier if the fold partitions by family alone; before this change it did,
    at 30 bytes.
    """
    from prismaquant.allocator_candidates import tessera_group_composites
    from prismaquant.tessera_formats import tessera_wire_recipe
    from tessera.manifest import BodyKind

    fam = "TESSERA_E2M1_K2"
    assert BodyKind(tessera_wire_recipe(fam, 800).body) is BodyKind.WINDOW
    assert BodyKind(tessera_wire_recipe(fam, 896).body) is BodyKind.TCQ

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        members[0]: [
            Candidate(fmt=f"{fam}_R800", bits_per_param=0.0,
                      memory_bytes=10, predicted_dloss=9.0),
            Candidate(fmt=f"{fam}_R896", bits_per_param=0.0,
                      memory_bytes=20, predicted_dloss=1.0),
        ],
        members[1]: [
            Candidate(fmt=f"{fam}_R850", bits_per_param=0.0,
                      memory_bytes=10, predicted_dloss=8.0),
            Candidate(fmt=f"{fam}_R896", bits_per_param=0.0,
                      memory_bytes=25, predicted_dloss=7.0),
        ],
    }
    report: dict = {}
    options = tessera_group_composites(
        members, candidates, n_params=1000,
        licence=_installed_fused_licence(), report=report)

    def bodies(option):
        return {
            BodyKind(tessera_wire_recipe(
                fmt.rsplit("_R", 1)[0], int(fmt.rsplit("_R", 1)[1])).body).name
            for fmt in option.member_formats.values()
        }

    assert options, "the two legal cells must each yield an option"
    for option in options:
        assert len(bodies(option)) == 1, (
            f"{option.fmt} puts {sorted(bodies(option))} in one fused "
            "module, and the contract marks fused_module.body shared")
    # The specific pair a family-only fold offered: k at the TCQ cap (25
    # bytes) is dominated, but q at the cap (20 bytes) with k on the WINDOW
    # body (10) totals 30 at cost 9.0 and dominates nothing legal, so it
    # survived pruning and reached the DP.
    assert 30 not in {int(o.memory_bytes) for o in options}
    # Two cells, reported separately, so provenance shows the partition.
    cells = [k for k in report if not k.startswith("__")]
    assert len(cells) == 2, cells
    assert all("body=" in k for k in cells), cells


def test_a_contract_that_freed_the_body_would_widen_the_same_fold():
    """The read runs both ways, which is what makes it a read.

    Same members, same rungs; only the contract moves. With ``body``, ``plane``
    and ``grid`` marked ``per_member`` the two coherence classes merge and the
    mixed pairing appears. Nothing in the allocator changed -- so the
    constraint really is coming from the table and not from a local rule with a
    contract-shaped comment on it.
    """
    from prismaquant.allocator_candidates import tessera_group_composites

    fam = "TESSERA_E2M1_K2"
    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: [
            Candidate(fmt=f"{fam}_R850", bits_per_param=0.0,
                      memory_bytes=10, predicted_dloss=8.0),
            Candidate(fmt=f"{fam}_R896", bits_per_param=0.0,
                      memory_bytes=20, predicted_dloss=1.0),
        ]
        for m in members
    }
    freed = _licence_with(body="per_member", plane="per_member",
                          grid="per_member")
    options = tessera_group_composites(
        members, candidates, n_params=1000, licence=freed)
    mixed = [o for o in options
             if len({f.rsplit("_R", 1)[1]
                     for f in o.member_formats.values()}) > 1]
    assert mixed, "freeing the body must let the two rungs share a module"


def test_an_unevaluable_shared_field_refuses_rather_than_ignoring_it():
    """A ``shared`` field this allocator cannot compute is a refusal.

    The fold holds every shared field fixed across a group. If a later
    contract marks something shared that nothing here can evaluate, folding as
    though the field were absent is precisely the silent assertion reading the
    table exists to prevent -- and it is the reason the two field tuples in
    ``tessera_formats`` are a closed vocabulary rather than a filter.
    """
    from prismaquant.allocator_candidates import tessera_group_composites

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: _tessera_member_candidates([(10, 2.0), (20, 1.0)]) for m in members
    }
    with pytest.raises(NotImplementedError, match="cannot evaluate"):
        tessera_group_composites(
            members, candidates, n_params=1000,
            licence=_licence_with(interleave="shared"))


def test_a_contract_silent_about_a_field_is_not_a_contract_permitting_it():
    """Drop ``body`` from the licence and the fold refuses, not relaxes."""
    from prismaquant.allocator_candidates import tessera_group_composites
    from prismaquant.tessera_runtime_contract import TesseraContractError

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: _tessera_member_candidates([(10, 2.0), (20, 1.0)]) for m in members
    }
    with pytest.raises(TesseraContractError, match="publishes no licence"):
        tessera_group_composites(
            members, candidates, n_params=1000,
            licence=_licence_with(body=None))


def test_a_group_of_unequal_input_widths_declines_the_fold_with_a_reason():
    """``columns`` is ``shared`` and it is the one shared field a rung cannot
    move, so it is checked where it CAN differ: on the group a profile built.

    A refusal of the fold, not of the run. The shape of a profile's fused
    groups is not this function's to veto, and the per-NAME path that owns such
    a group asserts nothing per member -- so the group keeps its uniform
    options and the receipt carries the named reason.
    """
    stats = {
        "model.layers.0.self_attn.q_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 4,
            "out_features": 4},
        "model.layers.0.self_attn.k_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 8,
            "out_features": 4},
    }
    members = sorted(stats)
    costs = {n: {"BF16": {"weight_mse": 0.0, "predicted_dloss": 0.5}}
             for n in stats}
    specs = [fr.REGISTRY["BF16"]]
    candidates = {
        m: _tessera_member_candidates([(10, 4.0), (20, 1.0)]) + [
            Candidate(fmt="BF16", bits_per_param=16.0, memory_bytes=32,
                      predicted_dloss=0.5)]
        for m in members
    }
    stats_ext, _c, cands_ext = _aggregate_under_installed_contract(
        stats, costs, specs, candidates)
    super_name = next(n for n in stats_ext if _FUSED_SIBLING_MARKER in n)
    stamp = stats_ext[super_name]["_tessera_group_menu"]["__licence__"]
    assert stamp["folded"] is False
    assert "4, 8" in stamp["declined"] or "[4, 8]" in stamp["declined"], stamp
    assert not any(is_tessera_group_option(c.fmt)
                   for c in cands_ext[super_name])
    assert "_fused_member_formats" not in stats_ext[super_name]


def _aggregate_under_installed_contract(stats, costs, specs, candidates,
                                        licence=None):
    """``aggregate_fused_siblings`` with the module's ONE read substituted.

    Substituting ``tessera_menu.tessera_runtime_contract`` -- the one read the
    licence and the route admission both go through -- is the same
    construction ``test_tessera_menu_real_table`` uses, and it is what makes
    "one read per run" testable: one patch moves both.
    """
    import hashlib
    from dataclasses import replace
    from unittest import mock

    from prismaquant import tessera_menu as tm
    from prismaquant import tessera_runtime_contract as trc

    path = trc.contract_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = trc._load_at(str(path), sha, "installed")
    assert contract.fused_module.fields["q256"] == "per_member", (
        "the installed contract must license the fold, or these tests prove "
        "nothing")
    if licence is not None:
        contract = replace(contract, fused_module=licence)
    with mock.patch.object(tm, "tessera_runtime_contract",
                           lambda: contract):
        return aggregate_fused_siblings(
            stats, costs, specs, candidates, _FakeProfile())


def test_aggregation_stops_at_the_contract_end_to_end():
    """The same withdrawal, through ``aggregate_fused_siblings``.

    The unit tests above drive the fold directly. This one drives the path the
    allocator actually runs, so a future refactor that stops threading the
    licence through the caller is caught too: no ``_fused_member_formats``
    entry may survive once the contract says ``shared``.
    """
    stats = {
        "model.layers.0.self_attn.q_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 4,
            "out_features": 4},
        "model.layers.0.self_attn.k_proj": {
            "h_trace": 1.0, "n_params": 16, "in_features": 4,
            "out_features": 4},
    }
    members = sorted(stats)
    # A stock rung on the menu beside the Tessera ones, so the per-NAME path
    # always has an option to build: the point of this test is what the group
    # knapsack adds, not the empty-menu refusal one member down.
    costs = {n: {"BF16": {"weight_mse": 0.0, "predicted_dloss": 0.5}}
             for n in stats}
    specs = [fr.REGISTRY["BF16"]]
    candidates = {
        members[0]: _tessera_member_candidates([(10, 4.0), (20, 1.0)]) + [
            Candidate(fmt="BF16", bits_per_param=16.0, memory_bytes=32,
                      predicted_dloss=0.5)],
        members[1]: _tessera_member_candidates([(10, 3.0), (20, 2.0)]) + [
            Candidate(fmt="BF16", bits_per_param=16.0, memory_bytes=32,
                      predicted_dloss=0.5)],
    }

    # With the installed contract the group carries a mixed-rung option.
    stats_ext, _c, _cx = _aggregate_under_installed_contract(
        stats, costs, specs, candidates)
    super_name = next(n for n in stats_ext if _FUSED_SIBLING_MARKER in n)
    by_option = stats_ext[super_name]["_fused_member_formats"]
    assert any(len(set(m.values())) > 1 for m in by_option.values())

    # Withdraw the licence in the parsed contract and nothing else.
    stats_ext2, _c2, cands_ext2 = _aggregate_under_installed_contract(
        stats, costs, specs, candidates, licence=_licence_with(q256="shared"))
    super_name2 = next(n for n in stats_ext2 if _FUSED_SIBLING_MARKER in n)
    assert "_fused_member_formats" not in stats_ext2[super_name2], (
        "the contract withdrew the per-member licence; the aggregation must "
        "carry no per-member rung map at all")
    assert not any(is_tessera_group_option(c.fmt)
                   for c in cands_ext2[super_name2]), (
        "no whole-group option may survive the withdrawal")
    stamp = stats_ext2[super_name2]["_tessera_group_menu"]["__licence__"]
    assert stamp["q256"] == "shared"
    assert stamp["folded"] is False


def test_every_parsed_licence_value_reaches_the_folds_receipt():
    """The rule behind decision 1: a value a gate reads is a value it records.

    ``contract_answer`` carries the ``fused_module`` keys the reader parses
    (``tests/test_tessera_menu`` pins that half). This is the other half:
    EVERY field the reader parses -- the schema id included, since it is the
    reader's own refusal gate and the receipt has to say which schema was
    refused against -- is named in the ``__licence__`` stamp the fold writes,
    so "the answer's keys are the keys some gate reads" is checked rather than
    asserted, and a field parsed for nobody shows up here as a missing stamp
    key.
    """
    import dataclasses

    from prismaquant.allocator_candidates import tessera_group_composites
    from prismaquant.tessera_runtime_contract import FusedModuleLicence

    members = ["m.q_proj", "m.k_proj"]
    candidates = {
        m: _tessera_member_candidates([(10, 2.0), (20, 1.0)]) for m in members
    }
    report: dict = {}
    assert tessera_group_composites(
        members, candidates, n_params=1000,
        licence=_installed_fused_licence(), report=report)
    parsed = {f.name for f in dataclasses.fields(FusedModuleLicence)}
    stamped = set(report["__licence__"])
    assert parsed <= stamped, sorted(parsed - stamped)
