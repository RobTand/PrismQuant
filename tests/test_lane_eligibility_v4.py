"""Tessera v4 resolves launches at an explicit residency, without a serve."""
from __future__ import annotations

from copy import deepcopy

import pytest

from prismaquant.lane_eligibility import (
    LaneEligibilityError,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_UNATTESTED,
    UnitStructuralFacts,
    _parse_table,
    resolve_unit_route,
)


#: The lane the streamed cells launch through, declared so the reader can
#: bind the launch to its extension.  No ``requires``: this file is about the
#: v4 launch grammar, not the v20 predicate (``test_tessera_lane_requires``).
_NATIVE_EXTENSIONS = [{"module_name_prefix": "tessera_window_gemv",
                       "lane": {"decoder": "window_gemv"}}]


def _contract():
    family = "TESSERA_E4M3_K1"
    cells = []
    for regime in ("decode", "batch"):
        for residency in ("resident", "streamed"):
            executes = [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}]
            if residency == "streamed":
                executes = [{"symbol": "tessera_window_gemv::gemv", "decoder": "window_gemv"}]
                if regime == "batch":
                    executes.append({"symbol": "torch._scaled_mm", "decoder": "window_gemv"})
            cells.append({
                "id": f"tessera_e4m3_k1_dense_sm121_{regime}_{residency}",
                "platform": "sm_121", "family": family, "structure": "dense",
                "regime": regime, "rungs_q256": [1024],
                "activation_contract": "fp8_per_token_dynamic", "executes": executes,
                "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
                "requires_plugin": "tessera",
                "requires_serve_flags": [f"TESSERA_SERVE_MODE={residency}"], "predicates": [],
            })
    return {
        "schema": "tessera.lane-eligibility.v4",
        "platforms": {"sm_121": {}}, "regimes": ["decode", "batch"],
        "structures": ["dense"], "cells": cells,
    }, [{"family": family, "kind": "tessera_wire", "residency_modes": ["resident", "streamed"]}]


def _parse(block=None, formats=None):
    if block is None:
        block, formats = _contract()
    return _parse_table(block, formats, "0.1.0", "test-commit", "test-sha",
                        native_extensions=_NATIVE_EXTENSIONS)


def _facts(structure="dense"):
    return UnitStructuralFacts(
        qname="model.layers.0.self_attn.q_proj", format_name="TESSERA_E4M3_K1_R1024",
        payload_family="TESSERA_E4M3_K1", k=None, n_sub=None, rate_q256=1024,
        structure=structure, role_split=False, in_features=1024, out_features=1024,
    )


@pytest.mark.parametrize("residency", ["resident", "streamed"])
def test_v4_resolves_residency_and_preserves_launches_independent_of_cell_order(residency):
    block, formats = _contract()
    expected = {
        cell["regime"]: cell["executes"] for cell in block["cells"]
        if cell["requires_serve_flags"] == [f"TESSERA_SERVE_MODE={residency}"]
    }
    table = _parse(block, formats)
    result = resolve_unit_route(_facts(), table, platform="sm_121", residency=residency)
    assert result.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    assert {route.regime: route.as_dict()["executes"] for route in result.regimes} == expected
    assert {cell.id: cell.as_dict()["executes"] for cell in table.cells} == {
        cell["id"]: cell["executes"] for cell in block["cells"]
    }
    block["cells"].reverse()
    reversed_result = resolve_unit_route(_facts(), _parse(block, formats), platform="sm_121", residency=residency)
    assert reversed_result.as_dict() == result.as_dict()


@pytest.mark.parametrize("residency", [None, "unknown"])
def test_v4_without_a_declared_residency_is_unattested(residency):
    result = resolve_unit_route(_facts(), _parse(), platform="sm_121", residency=residency)
    assert result.route_status == ROUTE_STATUS_UNATTESTED
    assert "residency" in result.unattested_reason
    assert result.in_scope is True


def test_v4_cannot_invent_moe_attestation_from_dense_cells():
    result = resolve_unit_route(_facts("routed_moe"), _parse(), platform="sm_121", residency="streamed")
    assert result.route_status == ROUTE_STATUS_UNATTESTED
    assert all(route.cell_id is None for route in result.regimes)


@pytest.mark.parametrize("launches", [
    [],
    [{"symbol": "torch.mm"}],
    [{"symbol": "torch.mm", "decoder": "torch_window", "rationale": "native"}],
    [{"symbol": "", "decoder": "torch_window"}],
    [{"symbol": "torch.mm", "decoder": None}],
    [{"symbol": "torch.mm", "decoder": "torch_window"}] * 2,
])
def test_v4_rejects_malformed_or_repeated_launches(launches):
    block, formats = _contract()
    block["cells"][0]["executes"] = launches
    with pytest.raises(LaneEligibilityError, match="executes"):
        _parse(block, formats)


@pytest.mark.parametrize("plugin", [None, "gridbook", ""])
def test_v4_requires_tessera_plugin(plugin):
    block, formats = _contract()
    if plugin is None:
        del block["cells"][0]["requires_plugin"]
    else:
        block["cells"][0]["requires_plugin"] = plugin
    with pytest.raises(LaneEligibilityError, match="requires_plugin"):
        _parse(block, formats)


@pytest.mark.parametrize("flags", [
    [], ["TESSERA_SERVE_MODE=resident|resident"], ["TESSERA_SERVE_MODE=unknown"],
    ["TESSERA_SERVE_MODE=resident", "TESSERA_SERVE_MODE=streamed"],
    "TESSERA_SERVE_MODE=resident",
])
def test_v4_requires_one_well_formed_residency_flag(flags):
    block, formats = _contract()
    block["cells"][0]["requires_serve_flags"] = flags
    with pytest.raises(LaneEligibilityError, match="requires_serve_flags"):
        _parse(block, formats)


def test_v4_rejects_overlapping_residencies_even_when_rungs_differ():
    block, formats = _contract()
    overlapping = deepcopy(block["cells"][0])
    overlapping["id"] = "another_cell"
    overlapping["rungs_q256"] = [1280]
    block["cells"].append(overlapping)
    with pytest.raises(LaneEligibilityError, match="overlap|both cover"):
        _parse(block, formats)


def test_v4_residency_must_be_published_by_its_family():
    block, formats = _contract()
    formats[0]["residency_modes"] = ["resident"]
    with pytest.raises(LaneEligibilityError, match="residency"):
        _parse(block, formats)


def test_v3_compatibility_keeps_legacy_resolution_without_fabricated_launches():
    block, formats = _contract()
    block["schema"] = "tessera.lane-eligibility.v3"
    block["cells"] = [cell for cell in block["cells"] if cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"]]
    for cell in block["cells"]:
        del cell["executes"]
        del cell["requires_plugin"]
    table = _parse(block, formats)
    result = resolve_unit_route(_facts(), table, platform="sm_121")
    assert result.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    assert all("executes" not in route.as_dict() for route in result.regimes)


def test_gridbook_schema_is_still_retired():
    block, formats = _contract()
    block["schema"] = "gridbook.lane-eligibility.v4"
    with pytest.raises(LaneEligibilityError, match="schema"):
        _parse(block, formats)
