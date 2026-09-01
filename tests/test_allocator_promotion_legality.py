"""Serving-unit promotion must land on a format EVERY member can run.

Issue #28. Packed-MoE experts and fused siblings (q/k/v, gate/up) are atomic
at serve time — vLLM selects one scheme per FusedMoE layer and one scheme per
fused module — so union-find promotion collapses each connected component onto
a single format. It used to pick the highest-RANK format assigned to any member
with no check that the format is legal for the rest of them, and members of one
unit do NOT share a shape:

  - gate/up vs down differ on the reduce dim, so an odd ``moe_intermediate_size``
    can leave a projection indivisible by one format's group size while the
    other projection is fine (1712 = 107x16, but 1712 % 32 == 16);
  - fused siblings share the reduce dim but not ``out_features``, and the
    out_features-keyed kernel rules are real (the MXFP8 N >= 128 rule that
    masks DeltaNet's ``in_proj_a`` on the shipped Qwen3.6-27B).

Export then coerced the ONE offending member to BF16, producing a quantized +
BF16 mix inside a single serving unit that only the fused-coherence gate at the
very END of export caught. The fix hands promotion the per-row legal-format
sets — which ``build_candidates`` already computed through
``check_stats_format_applicability`` — so the shared format is legal for every
member by construction.

Every legality fixture here is built through the REAL gate (``build_candidates``
+ the real registry specs + the real ``DefaultProfile`` grouping), not from
hand-written sets, so a change in the applicability rules shows up here.
"""
from __future__ import annotations

import itertools
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator import validate_final_serving_promotion_noop
from prismaquant.allocator_candidates import (
    _PACKED_GROUP_MARKER,
    aggregate_packed_serving_groups,
    build_candidates,
    expand_packed_group_assignment,
)
from prismaquant.allocator_solver import (
    compute_achieved,
    legal_formats_from_candidates,
    promote_serving_units,
    solve_with_promotion,
)
from prismaquant.model_profiles import DefaultProfile


# ---------------------------------------------------------------------------
# Fixtures: real shapes -> real candidate sets
# ---------------------------------------------------------------------------

def _specs(names):
    return [fr.REGISTRY[n] for n in names]


def _rank(names):
    ordered = sorted(_specs(names), key=lambda s: s.effective_bits)
    return {s.name: i for i, s in enumerate(ordered)}


def _stats_and_costs(shapes: dict[str, tuple[int, int]], fmt_names):
    """Probe/cost tables for ``{qname: (out_features, in_features)}``."""
    stats, costs = {}, {}
    for name, (out_f, in_f) in shapes.items():
        stats[name] = {
            "h_trace": 0.7,
            "n_params": out_f * in_f,
            "in_features": in_f,
            "out_features": out_f,
            "n_tokens_seen": 4096,
        }
        # Cheaper formats cost more predicted loss, as measured cost tables do.
        costs[name] = {
            f: {"weight_mse": 0.5 / max(fr.REGISTRY[f].effective_bits, 1e-9)}
            for f in fmt_names
        }
    return stats, costs


def _legality(shapes, fmt_names, *, source_manifest=None):
    """Per-row legal-format sets, through ``build_candidates``."""
    stats, costs = _stats_and_costs(shapes, fmt_names)
    cands = build_candidates(
        stats, costs, _specs(fmt_names), source_manifest=source_manifest)
    return stats, costs, cands, legal_formats_from_candidates(cands)


# An MoE layer whose moe_intermediate_size (1712) is divisible by 16 but not by
# 32: gate/up keep MXFP8, down does not, and every member keeps NVFP4/BF16.
_HIDDEN, _MOE_INTERMEDIATE = 2048, 1712
_EXPERT = "model.layers.0.mlp.experts.0"
_MOE_SHAPES = {
    f"{_EXPERT}.gate_proj": (_MOE_INTERMEDIATE, _HIDDEN),
    f"{_EXPERT}.up_proj": (_MOE_INTERMEDIATE, _HIDDEN),
    f"{_EXPERT}.down_proj": (_HIDDEN, _MOE_INTERMEDIATE),
}
_MOE_MEMBERS = tuple(_MOE_SHAPES)


# A second MoE layer whose moe_intermediate_size (1704) is divisible by 8 but
# not by 16, so EVERY grouped format is masked on down_proj and only the
# per-tensor rungs (FP8_E4M3, BF16) are legal for the whole unit. This is the
# shape that gives promotion two legal-for-all formats above the assignment,
# which is what makes "cheapest at or above" distinguishable from "highest".
_ODD_MOE_INTERMEDIATE = 1704
_ODD_MOE_SHAPES = {
    f"{_EXPERT}.gate_proj": (_ODD_MOE_INTERMEDIATE, _HIDDEN),
    f"{_EXPERT}.up_proj": (_ODD_MOE_INTERMEDIATE, _HIDDEN),
    f"{_EXPERT}.down_proj": (_HIDDEN, _ODD_MOE_INTERMEDIATE),
}
_ODD_MOE_MEMBERS = tuple(_ODD_MOE_SHAPES)


# ---------------------------------------------------------------------------
# The defect and the fix
# ---------------------------------------------------------------------------

def test_packed_group_promotion_picks_a_format_legal_for_every_member():
    """The primary #28 case, end to end through the real legality gate.

    The DP gives gate/up MXFP8 and down NVFP4 (down is the cheap row). MXFP8 is
    the highest-rank format assigned, and it is ILLEGAL for down. Unconstrained
    promotion writes it anyway; with legality the whole unit moves to BF16 —
    the cheapest format at or above the DP's choice that all three can run.
    """
    fmts = ["NVFP4", "MXFP8_E4M3", "BF16"]
    _stats, _costs, cands, legal = _legality(_MOE_SHAPES, fmts)
    gate, up, down = _MOE_MEMBERS

    # Pinned through the real gate, not asserted by fiat.
    assert legal[gate] == {"NVFP4", "MXFP8_E4M3", "BF16"}
    assert legal[up] == {"NVFP4", "MXFP8_E4M3", "BF16"}
    assert legal[down] == {"NVFP4", "BF16"}, (
        "fixture must make MXFP8 illegal for down_proj only")

    rank = _rank(fmts)
    prof = DefaultProfile()
    assignment = {gate: "MXFP8_E4M3", up: "MXFP8_E4M3", down: "NVFP4"}

    # The defect: max-rank promotion writes MXFP8 onto a row whose own
    # candidate set never offered it.
    legacy = promote_serving_units(assignment, rank, profile=prof)
    assert legacy[down] == "MXFP8_E4M3"
    with pytest.raises(AssertionError, match="no candidate exists to price"):
        compute_achieved(_stats, legacy, {s.name: s for s in _specs(fmts)},
                         candidates=cands)

    # The fix: one format, legal for all three, and priceable.
    promoted = promote_serving_units(
        assignment, rank, profile=prof, legal_formats=legal)
    assert set(promoted.values()) == {"BF16"}, promoted
    compute_achieved(_stats, promoted, {s.name: s for s in _specs(fmts)},
                     candidates=cands)


def test_promotion_takes_the_cheapest_legal_format_at_or_above_the_dp_choice():
    """Two legal-for-all formats sit above the DP's choice; promotion takes the
    CHEAPER one, not BF16.

    Promotion has always been non-degrading (no member ends below the format
    the DP picked for it) and ``solve_with_promotion``'s tightening loop is
    built to absorb the bits promotion charges — so the rule is "cheapest legal
    format at or above the max-rank assignment", which here is FP8 (8.02 bpp),
    not the highest legal-for-all format (BF16, 16 bpp).
    """
    fmts = ["NVFP4", "MXFP4", "FP8_E4M3", "BF16"]
    _stats, _costs, _cands, legal = _legality(_ODD_MOE_SHAPES, fmts)
    gate, up, down = _ODD_MOE_MEMBERS
    assert legal[down] == {"FP8_E4M3", "BF16"}, (
        "fixture must mask every grouped format on down_proj, leaving exactly "
        "the two per-tensor formats legal for the whole unit")

    promoted = promote_serving_units(
        {gate: "NVFP4", up: "NVFP4", down: "FP8_E4M3"},
        _rank(fmts), profile=DefaultProfile(), legal_formats=legal)
    assert set(promoted.values()) == {"FP8_E4M3"}, promoted


def test_promotion_downgrades_when_nothing_legal_ranks_above():
    """When no legal-for-all format is at or above the max-rank assignment, the
    unit takes the highest-rank format it CAN run — legitimately below what the
    DP gave some members, because the unit must be uniform and a member cannot
    keep a format the unit cannot serve. Two rungs only: this isolates the
    branch (never a shippable menu).
    """
    fmts = ["NVFP4", "MXFP8_E4M3"]
    _stats, _costs, _cands, legal = _legality(_MOE_SHAPES, fmts)
    gate, up, down = _MOE_MEMBERS
    assert legal[down] == {"NVFP4"}

    promoted = promote_serving_units(
        {gate: "MXFP8_E4M3", up: "MXFP8_E4M3", down: "NVFP4"},
        _rank(fmts), profile=DefaultProfile(), legal_formats=legal)
    assert set(promoted.values()) == {"NVFP4"}, promoted


def test_fused_sibling_promotion_respects_out_features_masking():
    """Fused siblings share the reduce dim but not ``out_features``, so an
    out_features-keyed kernel rule (MXFP8 needs N >= 128) can make the
    highest-rank format illegal for k/v while q keeps it.
    """
    fmts = ["NVFP4", "MXFP8_E4M3", "BF16"]
    attn = "model.layers.0.self_attn"
    shapes = {
        f"{attn}.q_proj": (4096, _HIDDEN),
        f"{attn}.k_proj": (64, _HIDDEN),
        f"{attn}.v_proj": (64, _HIDDEN),
    }
    _stats, _costs, _cands, legal = _legality(shapes, fmts)
    assert legal[f"{attn}.q_proj"] == {"NVFP4", "MXFP8_E4M3", "BF16"}
    assert legal[f"{attn}.k_proj"] == {"NVFP4", "BF16"}, (
        "fixture must mask MXFP8 on the small-N sibling")

    assignment = {
        f"{attn}.q_proj": "MXFP8_E4M3",
        f"{attn}.k_proj": "NVFP4",
        f"{attn}.v_proj": "NVFP4",
    }
    rank = _rank(fmts)
    prof = DefaultProfile()
    assert promote_serving_units(assignment, rank, profile=prof)[
        f"{attn}.k_proj"] == "MXFP8_E4M3", "documents the pre-fix behaviour"
    promoted = promote_serving_units(
        assignment, rank, profile=prof, legal_formats=legal)
    assert set(promoted.values()) == {"BF16"}, promoted


# ---------------------------------------------------------------------------
# No common legal format: report, never allocate around it
# ---------------------------------------------------------------------------

def test_no_common_legal_format_raises_naming_members_and_sets():
    """Same verdict as ``aggregate_packed_serving_groups``' corrected docstring:
    an empty intersection is an upstream cost/legality bug, not a state to
    promote around. Built through the real gate from the documented third cause
    — a unit whose members have different SOURCE dtypes loses the passthrough
    formats from opposite ends.
    """
    fmts = ["NVFP4", "FP8_SOURCE", "BF16"]
    shapes = {
        # in_features 1000: indivisible by 16 (NVFP4) and by 128 (FP8_SOURCE).
        f"{_EXPERT}.gate_proj": (512, 1000),
        f"{_EXPERT}.down_proj": (1024, 2048),
    }
    manifest = {f"{_EXPERT}.gate_proj": "bf16", f"{_EXPERT}.down_proj": "fp8"}
    _stats, _costs, _cands, legal = _legality(
        shapes, fmts, source_manifest=manifest)
    gate, down = tuple(shapes)
    assert legal[gate] == {"BF16"}
    assert legal[down] == {"NVFP4", "FP8_SOURCE"}

    with pytest.raises(AssertionError) as exc:
        promote_serving_units(
            {gate: "BF16", down: "NVFP4"},
            _rank(fmts), profile=DefaultProfile(), legal_formats=legal)
    msg = str(exc.value)
    assert "shares no format that is legal for every member" in msg
    assert "common legal formats: []" in msg
    for name in (gate, down):
        assert name in msg
    assert "['BF16']" in msg and "['FP8_SOURCE', 'NVFP4']" in msg
    for cause in ("cost row", "applicability mask", "passthrough-source"):
        assert cause in msg, cause


# ---------------------------------------------------------------------------
# Non-regression: no legality info, and clean dims
# ---------------------------------------------------------------------------

def _max_rank_expectation(assignment, rank, members):
    """The pre-fix rule: every member of the unit takes the highest-rank
    format assigned to any of them."""
    best = max((assignment[m] for m in members), key=lambda f: rank[f])
    out = dict(assignment)
    for m in members:
        out[m] = best
    return out


def test_no_legality_info_reproduces_the_max_rank_behaviour():
    """Callers that cannot supply legality must behave exactly as before —
    including on the illegal-for-a-member fixture, where the old rule writes a
    format the row cannot run. ``None`` and an empty map are both "unknown".
    """
    fmts = ["NVFP4", "MXFP8_E4M3", "BF16"]
    rank = _rank(fmts)
    prof = DefaultProfile()
    for combo in itertools.product(fmts, repeat=len(_MOE_MEMBERS)):
        assignment = dict(zip(_MOE_MEMBERS, combo))
        expected = _max_rank_expectation(assignment, rank, _MOE_MEMBERS)
        assert promote_serving_units(
            assignment, rank, profile=prof) == expected
        assert promote_serving_units(
            assignment, rank, profile=prof, legal_formats=None) == expected
        assert promote_serving_units(
            assignment, rank, profile=prof, legal_formats={}) == expected
        # A row absent from the map is UNCONSTRAINED, not illegal.
        partial = {_MOE_MEMBERS[0]: set(fmts)}
        assert promote_serving_units(
            assignment, rank, profile=prof, legal_formats=partial) == expected


def test_clean_dividing_shapes_are_untouched_by_legality():
    """The shipped-artifact non-regression bar: on models whose dims divide
    cleanly for every format on the menu (Qwen3.6-27B, Qwen3.5-35B-A3B),
    every member's legal set is the full menu, so the legality-aware pass is
    identical to the legacy one for EVERY assignment of the unit.
    """
    fmts = ["NVFP4", "MXFP8_E4M3", "BF16"]
    clean = {
        f"{_EXPERT}.gate_proj": (17408, 5120),
        f"{_EXPERT}.up_proj": (17408, 5120),
        f"{_EXPERT}.down_proj": (5120, 17408),
    }
    _stats, _costs, _cands, legal = _legality(clean, fmts)
    members = tuple(clean)
    assert all(legal[m] == set(fmts) for m in members)

    rank = _rank(fmts)
    prof = DefaultProfile()
    for combo in itertools.product(fmts, repeat=len(members)):
        assignment = dict(zip(members, combo))
        assert (promote_serving_units(assignment, rank, profile=prof,
                                     legal_formats=legal)
                == promote_serving_units(assignment, rank, profile=prof))


# ---------------------------------------------------------------------------
# The aggregated (shipping) path stays a validated no-op
# ---------------------------------------------------------------------------

def test_aggregated_path_stays_a_validated_no_op():
    """``aggregate_packed_serving_groups`` already offers a unit only the
    formats legal for EVERY member (it intersects the member candidate sets,
    which encode shape applicability), so the shipping path never needed
    promotion to repair anything — and still does not once promotion is
    legality-aware. Pinned on the awkward-``moe_intermediate_size`` fixture
    that breaks the un-aggregated path.
    """
    fmts = ["NVFP4", "MXFP8_E4M3", "BF16"]
    shapes = dict(_MOE_SHAPES)
    dense = "model.layers.0.self_attn.o_proj"
    shapes[dense] = (_HIDDEN, _HIDDEN)
    stats, costs, cands, legal = _legality(shapes, fmts)
    prof = DefaultProfile()

    stats_ext, _costs_ext, cands_ext = aggregate_packed_serving_groups(
        stats, costs, _specs(fmts), cands, prof)
    super_name = next(n for n in cands_ext if _PACKED_GROUP_MARKER in n)
    assert {c.fmt for c in cands_ext[super_name]} == {"NVFP4", "BF16"}, (
        "the unit's menu must already exclude the format down_proj cannot run")

    specs = {s.name: s for s in _specs(fmts)}
    rank = _rank(fmts)
    floor = sum(
        min(c.bits_per_param for c in cs) * stats_ext[n]["n_params"]
        for n, cs in cands_ext.items()
    ) / sum(stats_ext[n]["n_params"] for n in cands_ext)
    for target in (floor, floor + 1.0, floor + 6.0):
        assign, _achieved = solve_with_promotion(
            stats_ext, cands_ext, target, specs, rank,
            bit_precision=0.001, profile=prof)
        assert assign is not None, target
        expanded = expand_packed_group_assignment(assign, stats_ext)
        promoted = promote_serving_units(
            expanded, rank, profile=prof, legal_formats=legal)
        assert promoted == expanded, (target, expanded, promoted)
        validate_final_serving_promotion_noop(expanded, promoted)
        for name, fmt in expanded.items():
            assert fmt in legal[name], (name, fmt)


# ---------------------------------------------------------------------------
# End to end: the un-aggregated path, where promotion is the ONLY mechanism
# ---------------------------------------------------------------------------

def test_cli_no_packed_aggregation_emits_a_coherent_legal_moe_unit(tmp_path):
    """``--no-packed-aggregation`` prices expert rows individually, so whole-unit
    promotion is the only thing keeping a FusedMoE layer coherent — and the
    format it lands on has to be one every projection can run.

    The DP here wants MXFP8 for gate/up (expensive at NVFP4) and NVFP4 for down
    (cheap), and ``moe_intermediate_size=1712`` makes MXFP8 illegal for down.
    Pre-fix, promotion wrote MXFP8 onto down and ``compute_achieved`` aborted
    the whole run ("no candidate exists to price its predicted loss"). The run
    must now finish and emit ONE format for the unit, legal for all of it.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
    }))

    shapes = {
        "model.layers.0.self_attn.o_proj": (_HIDDEN, _HIDDEN),
    }
    experts = []
    for e in range(2):
        for proj, shape in (
            ("gate_proj", (_MOE_INTERMEDIATE, _HIDDEN)),
            ("up_proj", (_MOE_INTERMEDIATE, _HIDDEN)),
            ("down_proj", (_HIDDEN, _MOE_INTERMEDIATE)),
        ):
            name = f"model.layers.0.mlp.experts.{e}.{proj}"
            experts.append(name)
            shapes[name] = shape

    stats, costs = {}, {}
    for name, (out_f, in_f) in shapes.items():
        stats[name] = {
            "h_trace": 1.0,
            "n_params": out_f * in_f,
            "in_features": in_f,
            "out_features": out_f,
        }
        # gate/up are expensive at 4 bits, down is nearly free: the DP splits
        # the unit and promotion has to reconcile it.
        cheap_at_nvfp4 = name.endswith("down_proj")
        costs[name] = {
            "NVFP4": {"predicted_dloss": 0.001 if cheap_at_nvfp4 else 20.0},
            "MXFP8_E4M3": {"predicted_dloss": 0.0005},
            "BF16": {"predicted_dloss": 0.0},
        }

    probe_path = tmp_path / "probe.pkl"
    cost_path = tmp_path / "cost.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats, "meta": {"model": str(model_dir)}}, f)
    with open(cost_path, "wb") as f:
        pickle.dump(
            {"costs": costs, "formats": ["NVFP4", "MXFP8_E4M3", "BF16"]}, f)

    layer_config = tmp_path / "layer_config.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "prismaquant.allocator",
            "--probe", str(probe_path),
            "--costs", str(cost_path),
            "--model-override", str(model_dir),
            "--formats", "NVFP4,MXFP8_E4M3,BF16",
            "--no-packed-aggregation",
            "--target-bits", "8.4",
            "--pareto-targets", "4.6,8.4",
            "--bit-precision", "0.05",
            "--layer-config", str(layer_config),
            "--pareto-csv", str(tmp_path / "pareto.csv"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # The canonical recipe parser, so this pins the EMITTED artifact.
    from prismaquant.layer_config import load_assignment

    assignment = load_assignment(layer_config)
    expert_formats = {assignment[n] for n in experts}
    assert len(expert_formats) == 1, (
        f"FusedMoE layer must load under ONE format, got {expert_formats}")
    assert expert_formats != {"MXFP8_E4M3"}, (
        "MXFP8 is group-32 and moe_intermediate_size=1712 is not: the unit "
        "must not be promoted onto a format down_proj cannot run")
