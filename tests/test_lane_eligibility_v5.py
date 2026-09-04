"""Synthetic v5 claims attest only their exact runtime and execution scope."""
from copy import deepcopy

import pytest

from prismaquant import lane_eligibility as lane


DENSE_IMAGE = "example/dense@sha256:" + "a" * 64
MOE_IMAGE = "example/moe@sha256:" + "b" * 64


def _contract():
    family = "TESSERA_E4M3_K1"
    cells = []
    for structure, image, execution_modes in (
        ("dense", DENSE_IMAGE, ["eager", "compiled"]),
        ("routed_moe", MOE_IMAGE, ["eager"]),
    ):
        for regime in ("decode", "batch"):
            cells.append({
                "id": f"{structure}_{regime}", "platform": "sm_121",
                "family": family, "structure": structure, "regime": regime,
                "rungs_q256": [1024], "activation_contract": "fp8_per_token_dynamic",
                "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
                "requires_plugin": "tessera", "predicates": [],
                "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
                "executes": [{"symbol": "torch.bmm" if structure == "routed_moe" else "torch._scaled_mm",
                              "decoder": "torch_window"}],
                "runtime": {"image": image, "execution_modes": execution_modes},
            })
    return {
        "schema": "tessera.lane-eligibility.v5", "platforms": {"sm_121": {}},
        "regimes": ["decode", "batch"], "structures": ["dense", "routed_moe"],
        "cells": cells,
    }, [{"family": family, "kind": "tessera_wire", "residency_modes": ["resident", "streamed"]}]


def _parse(block=None, formats=None):
    if block is None:
        block, formats = _contract()
    return lane._parse_table(block, formats, "fixture", "fixture", "fixture")


def _facts(structure="dense"):
    return lane.UnitStructuralFacts(
        qname="synthetic.weight", format_name="TESSERA_E4M3_K1_R1024",
        payload_family="TESSERA_E4M3_K1", k=None, n_sub=None, rate_q256=1024,
        structure=structure, role_split=False, in_features=1024, out_features=1024,
    )


@pytest.mark.parametrize("structure,image,mode", [
    ("dense", DENSE_IMAGE, "eager"), ("dense", DENSE_IMAGE, "compiled"),
    ("routed_moe", MOE_IMAGE, "eager"),
])
def test_v5_exact_scope_resolves_and_retains_observed_context(structure, image, mode):
    block, formats = _contract()
    table = _parse(block, formats)
    result = lane.resolve_unit_route(_facts(structure), table, platform="sm_121",
                                    residency="resident", runtime_image=image, execution_mode=mode)
    assert result.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    assert all(route.as_dict()["runtime_image"] == image for route in result.regimes)
    assert all(route.as_dict()["execution_mode"] == mode for route in result.regimes)
    assert {cell.id: cell.as_dict()["runtime"] for cell in table.cells} == {
        cell["id"]: cell["runtime"] for cell in block["cells"]
    }
    block["cells"].reverse()
    reverse = lane.resolve_unit_route(_facts(structure), _parse(block, formats),
                                     platform="sm_121", residency="resident",
                                     runtime_image=image, execution_mode=mode)
    assert reverse.as_dict() == result.as_dict()


@pytest.mark.parametrize("structure,image,mode", [
    ("routed_moe", MOE_IMAGE, "compiled"), ("routed_moe", DENSE_IMAGE, "eager"),
    ("dense", MOE_IMAGE, "eager"), ("dense", None, "eager"),
    ("dense", DENSE_IMAGE, None), ("dense", DENSE_IMAGE, "automatic"),
])
def test_v5_missing_or_wrong_runtime_scope_cannot_borrow_a_cell(structure, image, mode):
    result = lane.resolve_unit_route(_facts(structure), _parse(), platform="sm_121",
                                    residency="resident", runtime_image=image, execution_mode=mode)
    assert result.route_status == lane.ROUTE_STATUS_UNATTESTED
    assert result.in_scope is True


@pytest.mark.parametrize("runtime", [
    None, {}, {"image": DENSE_IMAGE}, {"execution_modes": ["eager"]},
    {"image": "example/dense:latest", "execution_modes": ["eager"]},
    {"image": DENSE_IMAGE, "execution_modes": []},
    {"image": DENSE_IMAGE, "execution_modes": ["automatic"]},
    {"image": DENSE_IMAGE, "execution_modes": ["eager", "eager"]},
    {"image": DENSE_IMAGE, "execution_modes": "eager"},
    {"image": DENSE_IMAGE, "execution_modes": ["eager"], "default": True},
])
def test_v5_runtime_grammar_refuses_missing_unknown_or_mutable_scope(runtime):
    block, formats = _contract()
    if runtime is None:
        del block["cells"][0]["runtime"]
    else:
        block["cells"][0]["runtime"] = runtime
    with pytest.raises(lane.LaneEligibilityError, match="runtime"):
        _parse(block, formats)


def test_v5_overlap_is_checked_per_runtime_and_execution_mode():
    block, formats = _contract()
    cell = deepcopy(block["cells"][0])
    cell["id"] = "duplicate"
    cell["runtime"]["execution_modes"] = ["compiled"]
    block["cells"].append(cell)
    with pytest.raises(lane.LaneEligibilityError, match="overlap|both cover"):
        _parse(block, formats)
    cell["runtime"]["image"] = MOE_IMAGE
    assert len(_parse(block, formats).cells) == len(block["cells"])


def test_v5_regimes_cannot_combine_across_runtime_images():
    block, formats = _contract()
    next(cell for cell in block["cells"] if cell["id"] == "dense_batch")["runtime"]["image"] = MOE_IMAGE
    result = lane.resolve_unit_route(_facts(), _parse(block, formats), platform="sm_121",
                                    residency="resident", runtime_image=DENSE_IMAGE, execution_mode="eager")
    assert result.route_status == lane.ROUTE_STATUS_UNATTESTED
    assert any(route.cell_id is None for route in result.regimes)
