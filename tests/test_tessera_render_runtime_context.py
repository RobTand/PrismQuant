"""Released-pin rendering admission cannot borrow another v5 runtime's cells."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render


NAME = "TESSERA_E4M3_K1_R1024"
FAMILY = "TESSERA_E4M3_K1"
IMAGE = "example/dense@sha256:" + "a" * 64
MOE_IMAGE = "example/moe@sha256:" + "b" * 64


def _context(**changed):
    values = dict(platform="sm_121", structure="dense", residency="resident",
                  runtime_image=IMAGE, execution_mode="eager")
    values.update(changed)
    return lane.ServingContext(**values)


@pytest.fixture
def released_table(monkeypatch):
    rows = []
    for structure, image, modes in (("dense", IMAGE, ["eager", "compiled"]),
                                    ("routed_moe", MOE_IMAGE, ["eager"])):
        for regime in ("decode", "batch"):
            rows.append({
                "id": f"{structure}_{regime}", "platform": "sm_121",
                "family": FAMILY, "structure": structure, "regime": regime,
                "rungs_q256": [1024], "activation_contract": "fp8_per_token_dynamic",
                "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
                "requires_plugin": "tessera", "predicates": [],
                "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
                "executes": [{"symbol": "torch.bmm", "decoder": "torch_window"}],
                "runtime": {"image": image, "execution_modes": modes},
            })
    formats = {FAMILY: {"family": FAMILY, "kind": "tessera_wire",
                       "candidate_rungs_q256": [1024],
                       "residency_modes": ["resident", "streamed"]}}
    table = lane._parse_table({
        "schema": "tessera.lane-eligibility.v5", "platforms": {"sm_121": {}},
        "regimes": ["decode", "batch"], "structures": ["dense", "routed_moe"],
        "cells": rows,
    }, list(formats.values()), "fixture", "fixture", "fixture")
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    return table, formats


def test_unbound_v5_released_pin_does_not_admit_any_runtime(released_table):
    assert render.tessera_attesting_cells(NAME) == ()
    attested, reason = render.tessera_lane_admission(NAME)
    assert not attested
    assert "serving context" in reason
    assert not render.tessera_lane_attested(NAME)


@pytest.mark.parametrize("context", [
    _context(), _context(execution_mode="compiled"),
    _context(structure="routed_moe", runtime_image=MOE_IMAGE),
])
def test_explicit_v5_context_selects_its_complete_regime_population(released_table, context):
    table, _ = released_table
    cells = render.tessera_attesting_cells(NAME, serving_context=context)
    assert {cell.regime for cell in cells} == set(table.regimes)
    assert all(lane.cell_matches_serving_context(cell, context) for cell in cells)
    assert render.tessera_lane_admission(NAME, serving_context=context) == (True, "")
    assert render.tessera_lane_attested(NAME, serving_context=context)


@pytest.mark.parametrize("context", [
    _context(platform="sm_120"), _context(residency="streamed"),
    _context(runtime_image=MOE_IMAGE), _context(structure="routed_moe"),
    _context(structure="routed_moe", runtime_image=MOE_IMAGE, execution_mode="compiled"),
])
def test_v5_released_pin_cannot_borrow_any_mismatched_scope(released_table, context):
    assert render.tessera_attesting_cells(NAME, serving_context=context) == ()
    assert not render.tessera_lane_attested(NAME, serving_context=context)


def test_v5_cannot_cover_regimes_by_combining_different_images(released_table):
    table, formats = released_table
    cells = tuple(replace(cell, runtime_image=MOE_IMAGE)
                  if cell.structure == "dense" and cell.regime == "batch" else cell
                  for cell in table.cells)
    table = replace(table, cells=cells)
    assert render.tessera_attesting_cells(
        NAME, table=table, formats=formats, serving_context=_context()) == ()
    assert not render.tessera_lane_attested(
        NAME, table=table, formats=formats, serving_context=_context())


def test_spec_synthesis_threads_explicit_context_through_the_menu(monkeypatch):
    from prismaquant import tessera_menu as menu
    expected = _context()
    seen = []
    def admission(name, *, serving_context=None):
        assert name == NAME
        seen.append(serving_context)
        return SimpleNamespace(admits=lambda _mode: serving_context == expected)
    monkeypatch.setattr(menu, "route_admission", admission)
    monkeypatch.setattr(render, "tessera_rung_is_serialisable", lambda _name: True)
    assert render._producer_eligible(NAME, serving_context=expected)
    spec = render.synthesize_tessera_spec(NAME, serving_context=expected)
    assert spec.producer_eligible
    assert seen == [expected, expected]
    assert not render._producer_eligible(NAME)


def test_legacy_v4_renderer_keeps_its_prior_context_free_lookup(released_table):
    table, formats = released_table
    legacy = replace(table, schema=lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V4)
    assert render.tessera_lane_attested(NAME, table=legacy, formats=formats)
