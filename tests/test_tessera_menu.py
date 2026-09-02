"""The three gates a Tessera rung passes before the DP can see it.

Tessera's rate axis is continuous at a 1/256-bpp quantum, so "the menu" is not
a list somebody typed -- it is whatever three independent gates agree on:

* the **wire** can serialise these bytes (Tessera's own grammar),
* the **shape** tiles, including the shard a declared tensor-parallel degree
  hands the runtime,
* the pinned **runtime** attests the route (or the run explicitly opts into a
  research menu that says it does not).

These tests pin each gate separately, pin that the reductions applied
afterwards are the exact ones (dominance, never a hull), and pin that a menu
with no Tessera rung on it behaves exactly as it did before any of this
existed.
"""
from fractions import Fraction

import pytest

from prismaquant import tessera_menu as tm
from prismaquant.tessera_formats import (
    TesseraFormatError, format_promotion_class, parse_tessera_format_name,
)

SHAPE = (2048, 1024)


# ---------------------------------------------------------------------------
# Gate 1: the families, and that they are Tessera's answer and not ours
# ---------------------------------------------------------------------------

def test_families_are_the_serialisable_set_tessera_declares():
    names = sorted(f.name for f in tm.menu_families())
    assert names == [
        "TESSERA_E2M1_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1",
    ], names


def test_e4m3_arity_2_is_refused_by_tessera_not_by_us():
    """The brief listed E4M3 K1/K2; the encoder refuses K2 and the code wins.

    Pinned as a test rather than a comment because the refusal is a COST
    refusal (2**16 anchors scored per trellis step), so it would come back the
    day the anchor budget moves -- and the menu would silently widen.
    """
    from prismaquant.tessera_formats import tessera_family

    with pytest.raises(TesseraFormatError, match="anchors scored"):
        tessera_family("E4M3", 2)


# ---------------------------------------------------------------------------
# Gate 2: shape, and the tensor-parallel shard of it
# ---------------------------------------------------------------------------

def test_shard_granularity_is_read_from_one_function():
    """One seam, and it derives from plane geometry until Tessera exports one.

    ``tessera.layout.shard_granularity`` is the function this will call once
    that branch merges; the derivation below is the stand-in and must agree
    with the plane the wire actually carries.
    """
    for name in ("TESSERA_E2M1_K1_R512", "TESSERA_E2M1_K2_R896",
                 "TESSERA_E4M3_K1_R896"):
        family, rung = parse_tessera_format_name(name)
        rows, cols = tm.tessera_shard_granularity(family, rung)
        assert rows >= 1 and cols >= 1
        assert cols in (1, 16, 32), (name, cols)


def test_tp_degree_is_a_per_unit_legality_input_and_knows_the_direction():
    """TP legality is not one number: it depends on WHICH axis is sharded.

    A Tessera rate is realised as a per-column Bresenham schedule over the
    reduce dimension, so what a rank can encode is a function of ITS column
    count. Column-parallel Linears (q/k/v/gate/up) shard ``out_features`` and
    keep every column, so their realisable set is unchanged at any TP degree;
    row-parallel ones (o_proj/down_proj) shard ``in_features``, and at TP=8
    a 1024-column unit becomes a 128-column one, which is a strictly smaller
    realisable set. A gate that took only a degree and not a direction would
    get one of these two wrong.
    """
    mode = tm.MENU_RESEARCH
    col_1 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=1, parallel_kind=tm.PARALLEL_COLUMN)
    col_8 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=8, parallel_kind=tm.PARALLEL_COLUMN)
    assert len(col_8) == len(col_1), (len(col_8), len(col_1))

    row_1 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=1, parallel_kind=tm.PARALLEL_ROW)
    row_8 = tm.expand_tessera_menu(
        SHAPE, mode=mode, tp_degree=8, parallel_kind=tm.PARALLEL_ROW)
    assert len(row_8) < len(row_1), (len(row_8), len(row_1))
    assert row_8, "TP=8 must not empty the menu on a shape that shards"


def test_tp_gate_refuses_a_shard_that_does_not_divide():
    """An indivisible axis is a refusal with a named reason, not a silent pass."""
    family, rung = parse_tessera_format_name("TESSERA_E2M1_K2_R896")
    ok, reason = tm.tessera_tp_legal(
        family, rung, (2048, 1000), tp_degree=8,
        parallel_kind=tm.PARALLEL_ROW)
    assert not ok
    assert reason


def test_tp_gate_reason_and_provenance_reach_the_candidate_gate():
    """The allocator's own applicability gate consumes the TP verdict."""
    from prismaquant import allocator_candidates as ac
    from prismaquant import format_registry as fr

    verdict = ac._tensor_parallel_applicability(
        "TESSERA_E2M1_K2_R896", qname="model.layers.0.self_attn.q_proj",
        target_profile="tessera_research_sm121",
        in_features=1024, out_features=2048, packed_expert=False,
    )
    assert verdict.provenance is not None
    assert "tp_degree" in verdict.provenance
    assert "tp_parallel_kind" in verdict.provenance


def test_tp_gate_is_inert_for_stock_formats():
    from prismaquant import allocator_candidates as ac
    from prismaquant import format_registry as fr

    verdict = ac._tensor_parallel_applicability(
        "NVFP4", qname="model.layers.0.mlp.down_proj",
        target_profile="tessera_research_sm121",
        in_features=3072, out_features=1024, packed_expert=False,
    )
    assert verdict.legal and verdict.provenance is None


# ---------------------------------------------------------------------------
# Gate 3: the route, attested or explicitly not
# ---------------------------------------------------------------------------

def test_route_admission_never_asserts():
    """Every admission names the table that answered (principle 14)."""
    adm = tm.route_admission("TESSERA_E2M1_K2_R896")
    assert adm.source
    assert adm.route_status in {
        tm.ROUTE_STATUS_BACKED, tm.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
        tm.ROUTE_STATUS_UNATTESTED,
    }


def test_attested_menu_is_closed_under_the_current_pin():
    """A measured fact about the pinned release, recorded as a test.

    The pinned Gridbook contract publishes no Tessera cell, so the DEFAULT
    menu holds no Tessera rung at all. This is the honest state of the gate,
    not a bug: principle 9 says a format is production-eligible only when a
    pinned runtime attests it, and nothing does yet. When the Tessera lane
    publishes its contract this test flips, and that is the signal that the
    default menu just changed.
    """
    rungs = tm.expand_tessera_menu(SHAPE, mode=tm.MENU_ATTESTED)
    assert rungs == []


def test_research_menu_is_dense_and_stamps_its_status():
    rungs = tm.expand_tessera_menu(SHAPE, mode=tm.MENU_RESEARCH)
    assert len(rungs) > 500, len(rungs)
    rates = {r.bits_per_param for r in rungs}
    assert len(rates) > 100, len(rates)
    assert all(r.admission.route_status == tm.ROUTE_STATUS_UNATTESTED
               for r in rungs)


def test_menu_mode_refuses_an_unknown_spelling():
    with pytest.raises(tm.TesseraMenuError):
        tm.menu_mode("sortof")


def test_bytes_are_exact_and_monotone_in_rate():
    from prismaquant import format_registry as fr

    rungs = sorted(
        tm.expand_tessera_menu(SHAPE, mode=tm.MENU_RESEARCH),
        key=lambda r: r.body_rate_q256,
    )
    by_family: dict[str, list] = {}
    for rung in rungs:
        by_family.setdefault(rung.family, []).append(rung)
    for family, group in by_family.items():
        prev = -1
        for rung in group:
            spec = fr.get_format(rung.format_name)
            assert spec.memory_bytes_for_shape(SHAPE) == rung.memory_bytes
            assert rung.memory_bytes >= prev, (family, rung.format_name)
            prev = rung.memory_bytes


# ---------------------------------------------------------------------------
# The reductions: what may be dropped, and what may never be
# ---------------------------------------------------------------------------

def test_prune_dominated_keeps_a_point_inside_the_hull():
    """The reduction is exact dominance. A convex hull is NOT admissible.

    (100, 10) and (300, 1) are the hull; (200, 6) sits strictly above the
    segment between them and a hull prune would drop it. The knapsack's budget
    is discrete, so at a remaining capacity that fits 200 bytes and not 300 it
    is the optimum -- dropping it would change the DP's answer.
    """
    rows = [(100, 10.0, "a"), (200, 6.0, "b"), (300, 1.0, "c")]
    kept = [r[2] for r in tm.prune_dominated(rows)]
    assert kept == ["a", "b", "c"]


def test_prune_dominated_drops_only_both_axes_worse():
    rows = [(100, 10.0, "a"), (150, 12.0, "worse-both"), (300, 1.0, "c")]
    kept = [r[2] for r in tm.prune_dominated(rows)]
    assert kept == ["a", "c"]


def test_collapse_to_dp_bins_agrees_with_the_solver():
    from prismaquant.allocator_solver import _charged_bins

    n_params, total = 2_048 * 1_024, 500_000_000
    rows = [
        (int(bpp * n_params / 8), 1.0 / bpp, f"r{i}")
        for i, bpp in enumerate([3.0, 3.001, 3.5, 4.0])
    ]
    kept = tm.collapse_to_dp_bins(
        rows, baseline_bits_per_param=3.0, n_params=n_params,
        total_params=total, bit_precision=1e-4)
    bins = {
        _charged_bins(
            (row[0] * 8.0 / n_params - 3.0) * (n_params / total), 1e-4)
        for row in kept
    }
    assert len(bins) == len(kept)


def test_reduce_continuous_menu_is_a_noop_without_tessera():
    from prismaquant.allocator_candidates import Candidate, reduce_continuous_menu

    cands = {
        "x": [
            Candidate(fmt="NVFP4", bits_per_param=4.5, memory_bytes=100,
                      predicted_dloss=1.0),
            Candidate(fmt="FP8_E4M3", bits_per_param=8.0, memory_bytes=200,
                      predicted_dloss=0.5),
        ]
    }
    report: dict = {}
    out = reduce_continuous_menu(
        cands, {"x": {"n_params": 1000}}, bit_precision=1e-4, report=report)
    assert out == cands
    assert report == {}


# ---------------------------------------------------------------------------
# Promotion: the unit shares a family, not a rate
# ---------------------------------------------------------------------------

def test_promotion_class_is_the_family_for_tessera_and_the_name_otherwise():
    assert format_promotion_class("TESSERA_E2M1_K2_R896") == "TESSERA_E2M1_K2"
    assert format_promotion_class("NVFP4") == "NVFP4"
    assert format_promotion_class("FP8_E4M3") == "FP8_E4M3"


def test_fused_siblings_share_a_family_with_rates_free():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {
        "TESSERA_E2M1_K2_R512": 0,
        "TESSERA_E2M1_K2_R896": 1,
        "TESSERA_E4M3_K1_R896": 2,
    }
    legal = {name: set(rank) for name in ("q", "k", "v")}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R512",
         "k": "TESSERA_E2M1_K2_R896",
         "v": "TESSERA_E2M1_K2_R512"},
        rank, [["q", "k", "v"]], legal,
    )
    assert {format_promotion_class(f) for f in out.values()} == {
        "TESSERA_E2M1_K2"}
    # ...and the rates did NOT collapse to one.
    assert len(set(out.values())) == 2, out


def test_promotion_is_non_degrading_within_the_family():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"TESSERA_E2M1_K2_R512": 0, "TESSERA_E2M1_K2_R896": 1}
    legal = {"q": set(rank), "k": {"TESSERA_E2M1_K2_R896"}}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R512", "k": "TESSERA_E2M1_K2_R896"},
        rank, [["q", "k"]], legal)
    assert out["k"] == "TESSERA_E2M1_K2_R896"
    assert out["q"] == "TESSERA_E2M1_K2_R512"


def test_promotion_unchanged_for_a_stock_menu():
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"NVFP4": 0, "FP8_E4M3": 1, "BF16": 2}
    legal = {n: set(rank) for n in ("q", "k", "v")}
    out = _promote_group_components(
        {"q": "NVFP4", "k": "FP8_E4M3", "v": "NVFP4"},
        rank, [["q", "k", "v"]], legal)
    assert out == {"q": "FP8_E4M3", "k": "FP8_E4M3", "v": "FP8_E4M3"}


def test_promotion_falls_back_to_uniform_when_the_family_is_illegal():
    """The relaxation may only widen what promotion accepts, never fail."""
    from prismaquant.allocator_solver import _promote_group_components

    rank = {"TESSERA_E2M1_K2_R896": 0, "BF16": 1}
    legal = {"q": {"TESSERA_E2M1_K2_R896", "BF16"}, "k": {"BF16"}}
    out = _promote_group_components(
        {"q": "TESSERA_E2M1_K2_R896", "k": "BF16"},
        rank, [["q", "k"]], legal)
    assert out == {"q": "BF16", "k": "BF16"}


# ---------------------------------------------------------------------------
# The route travels with the choice
# ---------------------------------------------------------------------------

def test_serving_lane_route_resolves_a_tessera_rung():
    """A Tessera rung reports 'unattested', never 'no_declared_lane'.

    Absence of a declaration and a declaration of absence are different facts,
    and principle 9 wants the second one countable.
    """
    from prismaquant.serving_profiles import serving_lane_route

    lane = serving_lane_route(
        "tessera_research_sm121", "TESSERA_E2M1_K2_R896")
    assert lane is not None
    assert lane.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert lane.route_status_source
    assert lane.activation_contract


def test_selection_provenance_counts_tessera_units_by_route_status():
    from prismaquant.allocator_candidates import selection_serving_lane_provenance

    prov = selection_serving_lane_provenance(
        {"a": "TESSERA_E2M1_K2_R896", "b": "TESSERA_E4M3_K1_R896"},
        None, "tessera_research_sm121")
    assert prov["units_without_declared_lane"] == 0
    assert prov["route_status_counts"].get(tm.ROUTE_STATUS_UNATTESTED) == 2


def test_layer_config_roundtrip_recovers_the_rung():
    from prismaquant.layer_config import canonicalize_format
    from prismaquant import format_registry as fr

    name = "TESSERA_E2M1_K2_R896"
    entry = fr.get_format(name).autoround_config()
    assert entry["data_type"] == "tessera"
    assert canonicalize_format(entry) == name
