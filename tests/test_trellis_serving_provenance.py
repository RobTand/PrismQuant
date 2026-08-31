"""WO-D D5: Trellis serving-lane provenance, shipcard and served-route reconciliation.

Tests per WO-D D5:
1. TCQ_E2M1_R256@512 dense on sm_121 resolves to exact cell id, backed_with_serve_flag, contract e2m1_group16_ue4m3_static.
2. @640 (producer candidate with no cell) resolves unbacked and export/shipcard refuses.
3. Routed-MoE trellis unit resolves unbacked naming missing cell.
4. backed_with_serve_flag artifact with no recorded serve flags fails shipcard.
5. validate_native_export refuses when emitted route histogram disagrees with priced one.
6. Fixture contract whose cells differ from pinned one changes every one of the above.

Principle: Never retype contract values as literals in source under test; derive
them. Fixture test proves nothing is hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant.allocator_candidates import selection_serving_lane_provenance
from prismaquant.gridbook_lane_eligibility import (
    load_eligibility_table,
    load_published_formats,
    unit_structural_facts,
    resolve_unit_route,
)
from prismaquant.shipcard import verify
from prismaquant.validate_native_export import verify_trellis_priced_vs_served

REPO = Path(__file__).resolve().parents[1]
FIXTURE_V12 = REPO / "prismaquant" / "gridbook_runtime" / "gridbook_runtime_contract.0.9.1.json"


def _published():
    return load_published_formats()


def _table():
    return load_eligibility_table()


# ---------------------------------------------------------------------------
# 1. @512 dense resolves to exact cell, backed_with_serve_flag
# ---------------------------------------------------------------------------
def test_e2m1_512_dense_resolves_backed_with_serve_flag():
    # Derive expected cell ids from the pinned contract instead of retyping?
    # We assert the known values but also derive that they exist in the table.
    table = _table()
    published = _published()
    facts = unit_structural_facts(
        "model.layers.0.mlp.down_proj", "TCQ_E2M1_R512",
        is_routed_moe=False, role_split=False,
        in_features=4096, out_features=4096,
        published_formats=published,
    )
    route = resolve_unit_route(facts, table, platform="sm_121")
    # Must be backed_with_serve_flag, derived not assumed
    assert route.route_status == "backed_with_serve_flag"
    assert route.requires_serve_flags == ("GRIDBOOK_TRELLIS_E2M1=1", "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed")
    assert route.activation_contracts == ("e2m1_group16_ue4m3_static",)
    assert route.qualifications == ("device_qualified",)
    # Cell ids derived from table
    decode_cells = [c for c in table.cells if c.id == "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4"]
    batch_cells = [c for c in table.cells if c.id == "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4"]
    assert decode_cells and batch_cells, "contract must contain both trellis E2M1 dense cells"
    # Route must name those cell ids per regime
    regime_map = {r.regime: r.cell_id for r in route.regimes}
    assert regime_map["decode"] == "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4"
    assert regime_map["batch"] == "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4"
    # Also via selection_serving_lane_provenance
    prov = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov["trellis_units"][0]["route_status"] == "backed_with_serve_flag"
    assert prov["trellis_units"][0]["activation_contract"] == "e2m1_group16_ue4m3_static"
    assert prov["trellis_units"][0]["qualification"] == "device_qualified"
    assert prov["trellis_units"][0]["cell_ids"] == [
        "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4",
        "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4",
    ]
    assert prov["trellis_route_histogram"] == {
        "TCQ_E2M1_R256:e2m1_group16_ue4m3_static:backed_with_serve_flag": 1
    }


# ---------------------------------------------------------------------------
# 2. @640 (candidate with no cell) resolves unbacked and refuses
# ---------------------------------------------------------------------------
def test_e2m1_640_resolves_unbacked_and_shipcard_refuses():
    published = _published()
    table = _table()
    facts = unit_structural_facts(
        "model.layers.0.mlp.down_proj", "TCQ_E2M1_R640",
        is_routed_moe=False, role_split=False,
        in_features=4096, out_features=4096,
        published_formats=published,
    )
    route = resolve_unit_route(facts, table, platform="sm_121")
    # Candidate rung 640 is in formats[].candidate_rungs_q256 but no cell lists it
    # so it must be unbacked (mapped from unattested for in_scope)
    assert route.route_status == "unattested"  # raw
    # Via provenance, exposed as unbacked
    prov = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R640"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov["trellis_units"][0]["route_status"] == "unbacked"
    assert "640" in prov["trellis_units"][0]["unattested_reason"]
    assert prov["trellis_route_histogram"] == {"TCQ_E2M1_R256::unbacked": 1}
    # Shipcard must refuse
    card = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov["trellis_units"],
            "trellis_route_histogram": prov["trellis_route_histogram"],
            "trellis_requires_serve_flags": prov["trellis_requires_serve_flags"],
        },
        "selection_serving_lane_provenance": prov,
    }
    problems = verify(card, required=[])
    assert any("unbacked" in p for p in problems), f"expected unbacked refusal, got {problems}"
    # With explicit non-native target, it should pass that gate (but may still fail histogram? histogram present so ok)
    card_override = dict(card)
    card_override["declared_non_native_target"] = "sm_90"
    card_override["trellis_route_status"] = dict(card["trellis_route_status"])
    card_override["trellis_route_status"]["declared_non_native_target"] = "sm_90"
    problems2 = verify(card_override, required=[])
    assert not any("unbacked" in p and "non-native" not in p for p in problems2), f"override should stamp, got {problems2}"


# ---------------------------------------------------------------------------
# 3. Routed-MoE trellis unit resolves unbacked naming missing cell
# ---------------------------------------------------------------------------
def test_routed_moe_trellis_resolves_unbacked_naming_missing_cell():
    published = _published()
    table = _table()
    facts = unit_structural_facts(
        "model.layers.0.mlp.experts.gate_up_proj", "TCQ_E2M1_R512",
        is_routed_moe=True, role_split=False,
        in_features=4096, out_features=4096,
        published_formats=published,
    )
    route = resolve_unit_route(facts, table, platform="sm_121")
    assert route.route_status == "unattested"
    assert route.in_scope is True
    # The reason should mention that no cell covers this structure/route
    assert "routed_moe" in route.unattested_reason or "structure" in route.unattested_reason or "covers" in route.unattested_reason
    prov = selection_serving_lane_provenance(
        {"model.layers.0.mlp.experts.gate_up_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov["trellis_units"][0]["structure"] == "routed_moe"
    assert prov["trellis_units"][0]["route_status"] == "unbacked"
    # The detail should name the missing routed_moe cell gap
    assert "routed_moe" in prov["trellis_units"][0]["unattested_reason"].lower() or "structure" in prov["trellis_units"][0]["unattested_reason"].lower() or "no lane cell" in prov["trellis_units"][0]["unattested_reason"]


# ---------------------------------------------------------------------------
# 4. backed_with_serve_flag without recorded flags fails shipcard
# ---------------------------------------------------------------------------
def test_backed_with_serve_flag_without_flags_fails_shipcard():
    prov = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    # Artifact claims backed_with_serve_flag but records no flags
    card = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov["trellis_units"],
            "trellis_route_histogram": prov["trellis_route_histogram"],
            "trellis_requires_serve_flags": [],  # missing
        },
        "selection_serving_lane_provenance": prov,
    }
    problems = verify(card, required=[])
    assert any("backed_with_serve_flag" in p and "requires_serve_flags" in p for p in problems)
    # With correct flags, it should not fail that gate
    card_ok = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov["trellis_units"],
            "trellis_route_histogram": prov["trellis_route_histogram"],
            "trellis_requires_serve_flags": prov["trellis_requires_serve_flags"],
        },
        "selection_serving_lane_provenance": prov,
    }
    problems_ok = verify(card_ok, required=[])
    assert not any("backed_with_serve_flag" in p and "requires_serve_flags" in p for p in problems_ok)


# ---------------------------------------------------------------------------
# 5. validate_native_export refuses on histogram disagreement
# ---------------------------------------------------------------------------
def test_validate_native_export_refuses_on_histogram_disagreement():
    prov = selection_serving_lane_provenance(
        {"a": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    priced = {
        "activation_contracts": prov["activation_contracts"],
        "trellis_route_histogram": prov["trellis_route_histogram"],
        "trellis_units": prov["trellis_units"],
    }
    # Served histogram that disagrees (e.g., claims fp8 instead of e2m1)
    served_disagree = {"fp8_per_token_dynamic": 1}
    problems = verify_trellis_priced_vs_served(priced, served_disagree)
    assert any("disagree" in p for p in problems), f"expected disagreement refusal, got {problems}"
    # Served that matches should pass
    served_match = {"e2m1_group16_ue4m3_static": 1}
    problems_match = verify_trellis_priced_vs_served(priced, served_match)
    assert problems_match == [], f"expected no refusal on match, got {problems_match}"
    # No telemetry at all when priced has trellis should also refuse (lack of evidence)
    problems_none = verify_trellis_priced_vs_served(priced, None)
    assert any("telemetry absent" in p or "lack of evidence" in p for p in problems_none)


# ---------------------------------------------------------------------------
# 6. Fixture contract whose cells differ changes every one of the above
# ---------------------------------------------------------------------------
def test_fixture_contract_changes_all_above(tmp_path, monkeypatch):
    # Build a fixture contract that swaps rungs and contracts
    orig_contract = json.loads(FIXTURE_V12.read_text())
    # Change trellis cells: make 640 backed, 512 unbacked, different contract/flags, and add routed_moe cell
    fixture_cells = [
        {
            "id": "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4",
            "platform": "sm_121",
            "family": "TCQ_E2M1_R256",
            "structure": "dense",
            "regime": "decode",
            "rungs_q256": [640],  # swapped
            "activation_contract": "fixture_contract_e2m1",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["FIXTURE_FLAG=1"],
            "predicates": [],
        },
        {
            "id": "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4",
            "platform": "sm_121",
            "family": "TCQ_E2M1_R256",
            "structure": "dense",
            "regime": "batch",
            "rungs_q256": [640],
            "activation_contract": "fixture_contract_e2m1",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["FIXTURE_FLAG=1"],
            "predicates": [],
        },
        {
            "id": "trellis_e2m1_routed_sm121_decode_scaled_mm_w4a4",
            "platform": "sm_121",
            "family": "TCQ_E2M1_R256",
            "structure": "routed_moe",
            "regime": "decode",
            "rungs_q256": [512],
            "activation_contract": "fixture_contract_e2m1",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["FIXTURE_FLAG=1"],
            "predicates": [],
        },
        {
            "id": "trellis_e2m1_routed_sm121_batch_scaled_mm_w4a4",
            "platform": "sm_121",
            "family": "TCQ_E2M1_R256",
            "structure": "routed_moe",
            "regime": "batch",
            "rungs_q256": [512],
            "activation_contract": "fixture_contract_e2m1",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["FIXTURE_FLAG=1"],
            "predicates": [],
        },
        # Keep E4M3 same for simplicity
        {
            "id": "trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8",
            "platform": "sm_121",
            "family": "TCQ_E4M3_R256",
            "structure": "dense",
            "regime": "decode",
            "rungs_q256": [1152],
            "activation_contract": "fp8_per_token_dynamic",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["GRIDBOOK_TRELLIS_E4M3=1", "GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed"],
            "predicates": [],
        },
        {
            "id": "trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8",
            "platform": "sm_121",
            "family": "TCQ_E4M3_R256",
            "structure": "dense",
            "regime": "batch",
            "rungs_q256": [1152],
            "activation_contract": "fp8_per_token_dynamic",
            "route_status": "backed_with_serve_flag",
            "qualification": "device_qualified",
            "requires_serve_flags": ["GRIDBOOK_TRELLIS_E4M3=1", "GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed"],
            "predicates": [],
        },
    ]
    fixture_payload = dict(orig_contract)
    fixture_payload["lane_eligibility"] = {
        "schema": "gridbook.lane-eligibility.v3",
        "platforms": {"sm_121": {"compute_capability": [12, 1]}},
        "regimes": ["decode", "batch"],
        "structures": ["dense", "routed_moe"],
        "cells": fixture_cells,
    }
    fixture_path = tmp_path / "fixture_contract.json"
    fixture_path.write_text(json.dumps(fixture_payload))

    # Patch loader to use fixture
    from prismaquant import gridbook_lane_eligibility as gle
    from prismaquant import serving_profiles as sp

    orig_load_table = gle.load_eligibility_table
    orig_load_formats = gle.load_published_formats

    def fixture_table(*a, **kw):
        return orig_load_table(contract_path=fixture_path)

    # Need published formats that match fixture (same as orig)
    orig_published = gle.load_published_formats()

    monkeypatch.setattr(gle, "load_eligibility_table", fixture_table)
    # Also ensure serving_profiles cache reset so platform lookup still works
    sp._reset_eligibility_table_cache()
    # published formats unchanged
    monkeypatch.setattr(gle, "load_published_formats", lambda *a, **kw: orig_published)
    # Force allocator_candidates to re-resolve platform (it caches per call, not globally)
    # Now test that all prior expectations flip

    # 1. @512 dense now should be UNBACKED (since fixture has 640)
    prov_512 = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov_512["trellis_units"][0]["route_status"] == "unbacked", f"fixture 512 should be unbacked, got {prov_512['trellis_units'][0]}"
    assert prov_512["trellis_units"][0]["activation_contract"] == "", "unbacked has no contract"

    # 2. @640 dense now BACKED with fixture contract/flags
    prov_640 = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R640"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov_640["trellis_units"][0]["route_status"] == "backed_with_serve_flag"
    assert prov_640["trellis_units"][0]["activation_contract"] == "fixture_contract_e2m1"
    assert prov_640["trellis_units"][0]["requires_serve_flags"] == ["FIXTURE_FLAG=1"]
    assert prov_640["trellis_units"][0]["cell_ids"] == [
        "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4",
        "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4",
    ]

    # 3. Routed MoE now BACKED (fixture has routed cell)
    prov_routed = selection_serving_lane_provenance(
        {"model.layers.0.mlp.experts.gate_up_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov_routed["trellis_units"][0]["route_status"] == "backed_with_serve_flag"
    assert prov_routed["trellis_units"][0]["structure"] == "routed_moe"

    # 4. Flag check now expects FIXTURE_FLAG
    card = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov_640["trellis_units"],
            "trellis_route_histogram": prov_640["trellis_route_histogram"],
            "trellis_requires_serve_flags": [],  # missing fixture flag
        },
        "selection_serving_lane_provenance": prov_640,
    }
    problems = verify(card, required=[])
    assert any("FIXTURE_FLAG" in p or "requires_serve_flags" in p for p in problems)

    # 5. Served vs priced histogram now should match fixture contract, not old
    priced = {
        "activation_contracts": prov_640["activation_contracts"],
        "trellis_route_histogram": prov_640["trellis_route_histogram"],
        "trellis_units": prov_640["trellis_units"],
    }
    # Old served (e2m1_group16_ue4m3_static) should now disagree
    served_old = {"e2m1_group16_ue4m3_static": 1}
    problems_old = verify_trellis_priced_vs_served(priced, served_old)
    assert any("disagree" in p for p in problems_old)
    served_new = {"fixture_contract_e2m1": 1}
    assert verify_trellis_priced_vs_served(priced, served_new) == []

    # Clean up cache
    sp._reset_eligibility_table_cache()


def test_trellis_tp1_required_and_fused_unit_eligible():
    """WO-C correction: trellis serving is TP=1 only, fused units are one wire."""
    # Check that provenance records TP=1 for any trellis artifact
    prov = selection_serving_lane_provenance(
        {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov["trellis_tensor_parallel_world_size"] == 1
    assert prov["trellis_requires_tp1"] is True
    # Fused unit: one wire per merged module (qkv_proj)
    prov_fused = selection_serving_lane_provenance(
        {"model.layers.0.self_attn.qkv_proj": "TCQ_E2M1_R512"},
        None, target_profile="trellis_research_sm121",
    )
    assert prov_fused["trellis_units"][0]["qname"] == "model.layers.0.self_attn.qkv_proj"
    assert prov_fused["trellis_units"][0]["route_status"] == "backed_with_serve_flag"
    # Shipcard must refuse TP>1
    card_tp2 = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov["trellis_units"],
            "trellis_route_histogram": prov["trellis_route_histogram"],
            "trellis_requires_serve_flags": prov["trellis_requires_serve_flags"],
            "trellis_tensor_parallel_world_size": 2,
        },
        "selection_serving_lane_provenance": prov,
        "tensor_parallel_world_size": 2,
    }
    problems = verify(card_tp2, required=[])
    assert any("TP=1" in p for p in problems), f"expected TP refusal, got {problems}"
    # TP=1 should pass (no TP error)
    card_tp1 = {
        "model_sha": "abc",
        "slots": {},
        "build": {"achieved_bpp": {"value": 3.0}},
        "trellis_route_status": {
            "trellis_units": prov["trellis_units"],
            "trellis_route_histogram": prov["trellis_route_histogram"],
            "trellis_requires_serve_flags": prov["trellis_requires_serve_flags"],
            "trellis_tensor_parallel_world_size": 1,
        },
        "selection_serving_lane_provenance": prov,
        "tensor_parallel_world_size": 1,
    }
    assert not any("TP=1" in p for p in verify(card_tp1, required=[]))
