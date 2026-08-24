"""Capability-scoped v11 Gridbook execution-cell consumption."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from prismaquant.cb_layout import (
    FP8_ACCEPTED_RUNGS,
    FP8_PRODUCT_RUNGS,
    NVFP4_ACCEPTED_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)
from prismaquant.gridbook_execution_contract import (
    GridbookExecutionContractError,
    attested_min_capability_sm,
    parse_gridbook_execution_contract,
    require_compile_only_gridbook_routes,
    require_device_qualified_gridbook_routes,
)
from prismaquant.gridbook_format_contract import (
    GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA,
)


REPO = Path(__file__).resolve().parents[1]
CURRENT = (
    REPO
    / "prismaquant"
    / "gridbook_runtime"
    / "gridbook_runtime_contract.0.8.11.json"
)


def _v11_contract() -> dict:
    contract = json.loads(CURRENT.read_text(encoding="utf-8"))
    contract["schema"] = GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA
    contract["contract_version"] = 11
    for entry in contract["formats"]:
        if entry["family"] == "FP8_CB_K":
            entry["rungs"] = list(FP8_ACCEPTED_RUNGS)
            entry["producer_rungs"] = list(FP8_PRODUCT_RUNGS)
        elif entry["family"] == "NVFP4_CB_K":
            entry["rungs"] = list(NVFP4_ACCEPTED_RUNGS)
            entry["producer_rungs"] = list(NVFP4_PRODUCT_RUNGS)
        else:
            entry["producer_rungs"] = list(entry["rungs"])
    contract["lane_eligibility"] = {
        "schema": "gridbook.lane-eligibility.v2",
        "platforms": {"sm_89": {"compute_capability": [8, 9]}},
        "regimes": ["decode", "batch"],
        "structures": ["dense", "routed_moe"],
        "cells": [
            {
                "id": "fp8_cb_dense_sm89_decode",
                "platform": "sm_89",
                "family": "FP8_CB_K",
                "structure": "dense",
                "regime": "decode",
                "rungs": list(FP8_PRODUCT_RUNGS),
                "route_status": "backed",
                "qualification": "compile_only",
                "requires_serve_flags": [],
                "predicates": [],
            },
            {
                "id": "fp8_cb_dense_sm89_batch",
                "platform": "sm_89",
                "family": "FP8_CB_K",
                "structure": "dense",
                "regime": "batch",
                "rungs": list(FP8_PRODUCT_RUNGS),
                "route_status": "backed",
                "qualification": "compile_only",
                "requires_serve_flags": [],
                "predicates": [],
            },
        ],
    }
    return contract


def _cells(contract: dict) -> list[dict]:
    return contract["lane_eligibility"]["cells"]


def _qualify(contract: dict) -> dict:
    for cell in _cells(contract):
        cell["qualification"] = "device_qualified"
    return contract


def test_initial_sm89_cells_are_structural_but_compile_only():
    contract = _v11_contract()
    parsed = parse_gridbook_execution_contract(contract)
    assert len(parsed.cells) == 2
    assert parsed.platforms[0].capability_sm == 89
    assert all(not cell.producer_legal for cell in parsed.cells)
    with pytest.raises(GridbookExecutionContractError, match="compile_only"):
        require_device_qualified_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=(4, 20, 48),
        )
    structural = require_compile_only_gridbook_routes(
        contract,
        family="FP8_CB_K",
        device_capability=(8, 9),
        structure="dense",
        rungs=(4, 20, 48),
    )
    assert {row.qualification for row in structural.resolutions} == {
        "compile_only"
    }
    with pytest.raises(GridbookExecutionContractError, match="no exact"):
        attested_min_capability_sm(
            contract,
            family="FP8_CB_K",
            structure="dense",
            rungs=(20,),
        )


def test_only_device_qualified_routes_cover_exact_sm89_and_every_regime():
    contract = _qualify(_v11_contract())
    attestation = require_device_qualified_gridbook_routes(
        contract,
        family="FP8_CB_K",
        device_capability=(8, 9),
        structure="dense",
        rungs=(4, 20, 48),
    )
    assert attestation.platform.id == "sm_89"
    assert {item.regime for item in attestation.resolutions} == {
        "decode", "batch"
    }
    assert len(attestation.resolutions) == 6
    assert attested_min_capability_sm(
        contract,
        family="FP8_CB_K",
        structure="dense",
        rungs=(4, 20, 48),
    ) == 89

    # Exact platform mapping, never a numerical minimum-capability inference.
    with pytest.raises(GridbookExecutionContractError, match=r"maps to \[\]"):
        require_device_qualified_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(9, 0),
            structure="dense",
            rungs=(20,),
        )


def test_nvfp4_k1_k25_need_exact_device_qualified_routes_in_both_regimes():
    contract = _v11_contract()
    contract["lane_eligibility"]["platforms"]["sm_121"] = {
        "compute_capability": [12, 1]
    }
    for regime in ("decode", "batch"):
        _cells(contract).append({
            "id": f"nvfp4_cb_dense_sm121_{regime}",
            "platform": "sm_121",
            "family": "NVFP4_CB_K",
            "structure": "dense",
            "regime": regime,
            "rungs": list(NVFP4_PRODUCT_RUNGS),
            "route_status": "backed",
            "qualification": "device_qualified",
            "requires_serve_flags": [],
            "predicates": [],
        })
    attestation = require_device_qualified_gridbook_routes(
        contract,
        family="NVFP4_CB_K",
        device_capability=(12, 1),
        structure="dense",
        rungs=(1, 25),
    )
    assert attestation.rungs == (1, 25)
    assert len(attestation.resolutions) == 4

    _cells(contract)[-1]["rungs"].remove(25)
    with pytest.raises(GridbookExecutionContractError, match="no sm_121"):
        require_device_qualified_gridbook_routes(
            contract,
            family="NVFP4_CB_K",
            device_capability=(12, 1),
            structure="dense",
            rungs=(1, 25),
        )


def test_unsupported_nvfp4_rung_cannot_enter_an_execution_cell():
    contract = _v11_contract()
    _cells(contract)[0]["family"] = "NVFP4_CB_K"
    _cells(contract)[0]["rungs"] = [25, 26]
    with pytest.raises(
        GridbookExecutionContractError,
        match=r"outside formats\['NVFP4_CB_K'\]\.producer_rungs",
    ):
        parse_gridbook_execution_contract(contract)


def test_sm120_compile_only_nvfp4_scaffold_preserves_fallback_boundaries():
    contract = _v11_contract()
    contract["lane_eligibility"]["platforms"]["sm_120"] = {
        "compute_capability": [12, 0]
    }
    rungs = list(NVFP4_PRODUCT_RUNGS)
    _cells(contract).extend([
        {
            "id": "nvfp4_cb_dense_sm120_decode_cuda_gemv",
            "platform": "sm_120",
            "family": "NVFP4_CB_K",
            "structure": "dense",
            "regime": "decode",
            "rungs": rungs,
            "route_status": "backed",
            "qualification": "compile_only",
            "requires_serve_flags": [],
            "predicates": [],
        },
        {
            "id": "nvfp4_cb_dense_sm120_batch_expand_bf16",
            "platform": "sm_120",
            "family": "NVFP4_CB_K",
            "structure": "dense",
            "regime": "batch",
            "rungs": rungs,
            "route_status": "fallback",
            "qualification": "compile_only",
            "requires_serve_flags": [],
            "predicates": [],
        },
        {
            "id": "nvfp4_cb_routed_sm120_decode_cuda_gemv",
            "platform": "sm_120",
            "family": "NVFP4_CB_K",
            "structure": "routed_moe",
            "regime": "decode",
            "rungs": rungs,
            "route_status": "backed",
            "qualification": "compile_only",
            "requires_serve_flags": [],
            "predicates": [],
        },
        {
            "id": "nvfp4_cb_routed_sm120_batch_persistent_b",
            "platform": "sm_120",
            "family": "NVFP4_CB_K",
            "structure": "routed_moe",
            "regime": "batch",
            "rungs": rungs,
            "route_status": "backed",
            "qualification": "compile_only",
            "requires_serve_flags": [],
            "predicates": [
                {"fact": "role_split", "op": "equals", "value": False}
            ],
        },
        {
            "id": "nvfp4_cb_routed_sm120_batch_expand_bf16",
            "platform": "sm_120",
            "family": "NVFP4_CB_K",
            "structure": "routed_moe",
            "regime": "batch",
            "rungs": rungs,
            "route_status": "fallback",
            "qualification": "compile_only",
            "requires_serve_flags": [],
            "predicates": [],
        },
    ])

    parsed = parse_gridbook_execution_contract(contract)
    sm120 = [cell for cell in parsed.cells if cell.platform == "sm_120"]
    assert len(sm120) == 5
    assert not any(cell.producer_legal for cell in sm120)

    facts = {rung: {"role_split": False} for rung in rungs}
    structural = require_compile_only_gridbook_routes(
        contract,
        family="NVFP4_CB_K",
        device_capability=(12, 0),
        structure="routed_moe",
        rungs=rungs,
        facts_by_rung=facts,
    )
    assert len(structural.resolutions) == 2 * len(rungs)
    assert {row.route_status for row in structural.resolutions} == {"backed"}

    with pytest.raises(GridbookExecutionContractError, match="fallback"):
        require_compile_only_gridbook_routes(
            contract,
            family="NVFP4_CB_K",
            device_capability=(12, 0),
            structure="dense",
            rungs=(1, 25),
        )
    with pytest.raises(GridbookExecutionContractError, match="fallback"):
        require_compile_only_gridbook_routes(
            contract,
            family="NVFP4_CB_K",
            device_capability=(12, 0),
            structure="routed_moe",
            rungs=(1, 25),
            facts_by_rung={
                1: {"role_split": True},
                25: {"role_split": True},
            },
        )
    with pytest.raises(GridbookExecutionContractError, match="compile_only"):
        require_device_qualified_gridbook_routes(
            contract,
            family="NVFP4_CB_K",
            device_capability=(12, 0),
            structure="routed_moe",
            rungs=(1, 25),
            facts_by_rung={
                1: {"role_split": False},
                25: {"role_split": False},
            },
        )


def test_absent_or_fallback_regime_is_not_producer_legal():
    absent = _qualify(_v11_contract())
    _cells(absent).pop()
    with pytest.raises(GridbookExecutionContractError, match="no sm_89"):
        require_device_qualified_gridbook_routes(
            absent,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=(20,),
        )

    fallback = _qualify(_v11_contract())
    _cells(fallback)[1]["route_status"] = "fallback"
    with pytest.raises(GridbookExecutionContractError, match="fallback"):
        require_device_qualified_gridbook_routes(
            fallback,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=(20,),
        )


def test_predicates_are_closed_and_missing_shape_facts_fail_closed():
    contract = _qualify(_v11_contract())
    _cells(contract)[1]["predicates"] = [
        {"fact": "in_features", "op": "multiple_of", "value": 256}
    ]
    with pytest.raises(GridbookExecutionContractError, match="no sm_89"):
        require_device_qualified_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=(20,),
        )
    assert require_device_qualified_gridbook_routes(
        contract,
        family="FP8_CB_K",
        device_capability=(8, 9),
        structure="dense",
        rungs=(20,),
        facts_by_rung={20: {"in_features": 5120}},
    ).rungs == (20,)


@pytest.mark.parametrize("reverse", (False, True))
def test_equal_rank_overlaps_fail_closed_independent_of_cell_order(reverse):
    contract = _qualify(_v11_contract())
    duplicate = deepcopy(_cells(contract)[0])
    duplicate["id"] = "fp8_cb_dense_sm89_decode_shadow"
    duplicate["qualification"] = "compile_only"
    if reverse:
        _cells(contract).insert(0, duplicate)
    else:
        _cells(contract).append(duplicate)

    with pytest.raises(GridbookExecutionContractError, match="ambiguous strongest"):
        require_device_qualified_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=(20,),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda c: c["lane_eligibility"].__setitem__("semantics", "open"),
            "keys differ",
        ),
        (
            lambda c: c["lane_eligibility"]["platforms"]["sm_89"].__setitem__(
                "compute_capability", [9, 0]
            ),
            "disagrees",
        ),
        (
            lambda c: _cells(c)[0].__setitem__("structure", "moe"),
            "structure",
        ),
        (
            lambda c: _cells(c)[0].__setitem__("route_status", "unbacked"),
            "route_status",
        ),
        (
            lambda c: _cells(c)[0].__setitem__("qualification", "qualified"),
            "qualification",
        ),
        (
            lambda c: _cells(c)[0]["rungs"].append(29),
            "sorted unique|outside formats",
        ),
        (
            lambda c: _cells(c)[0].__setitem__(
                "predicates",
                [{"fact": "payload_family", "op": "equals", "value": "FP8_CB_K"}],
            ),
            "fact must be one of",
        ),
    ),
)
def test_v2_is_closed_world_and_rejects_unknown_fields_enums_and_rungs(
    mutation, message
):
    contract = _v11_contract()
    mutation(contract)
    with pytest.raises(GridbookExecutionContractError, match=message):
        parse_gridbook_execution_contract(contract)


def test_contract_version_must_move_with_v11_schema():
    contract = _v11_contract()
    contract["contract_version"] = 4
    with pytest.raises(GridbookExecutionContractError, match="move together"):
        parse_gridbook_execution_contract(contract)
