"""Producer/reader rung separation at the immutable Gridbook boundary."""
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
from prismaquant.gridbook_format_contract import (
    GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA,
    GridbookFormatContractError,
    gridbook_contract_attests_producer_format,
    gridbook_format_rungs,
    validate_gridbook_cb_rung_contract,
)


REPO = Path(__file__).resolve().parents[1]
CURRENT = (
    REPO
    / "prismaquant"
    / "gridbook_runtime"
    / "gridbook_runtime_contract.0.8.11.json"
)


def _legacy_contract() -> dict:
    return json.loads(CURRENT.read_text(encoding="utf-8"))


def _synthetic_v11() -> dict:
    """Future schema fixture only; it is deliberately not a release pin."""

    contract = deepcopy(_legacy_contract())
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
    return contract


def _fp8_entry(contract: dict) -> dict:
    return next(
        entry for entry in contract["formats"]
        if entry["family"] == "FP8_CB_K"
    )


def test_current_v4_is_reader_compatible_but_not_producer_attestation():
    contract = _legacy_contract()
    declared = gridbook_format_rungs(contract, "FP8_CB_K")
    assert declared.accepted_rungs == tuple(range(28, 49))
    assert declared.producer_rungs is None
    assert not declared.producer_rungs_attested
    validate_gridbook_cb_rung_contract(contract)
    with pytest.raises(
        GridbookFormatContractError,
        match="producer-rung attestation requires",
    ):
        validate_gridbook_cb_rung_contract(
            contract, require_producer_attestation=True
        )


def test_v11_explicitly_attests_exact_reader_and_producer_domains():
    contract = _synthetic_v11()
    declarations = validate_gridbook_cb_rung_contract(
        contract, require_producer_attestation=True
    )
    fp8 = declarations["FP8_CB_K"]
    assert fp8.accepted_rungs == FP8_ACCEPTED_RUNGS
    assert fp8.producer_rungs == FP8_PRODUCT_RUNGS
    assert fp8.producer_rungs_attested
    assert gridbook_contract_attests_producer_format(contract, "FP8_CB_K40")
    assert gridbook_contract_attests_producer_format(contract, "fp8_cb_k48")
    assert not gridbook_contract_attests_producer_format(contract, "FP8_CB_K28")
    assert not gridbook_contract_attests_producer_format(
        contract, "FP8_CB_K29"
    )
    assert not gridbook_contract_attests_producer_format(contract, "FP8_CB_K25")
    nvfp4 = declarations["NVFP4_CB_K"]
    assert nvfp4.accepted_rungs == NVFP4_ACCEPTED_RUNGS
    assert nvfp4.producer_rungs == NVFP4_PRODUCT_RUNGS
    assert gridbook_contract_attests_producer_format(contract, "NVFP4_CB_K12")
    assert gridbook_contract_attests_producer_format(contract, "NVFP4_CB_K24")
    assert not gridbook_contract_attests_producer_format(contract, "NVFP4_CB_K1")
    assert not gridbook_contract_attests_producer_format(contract, "NVFP4_CB_K25")
    assert not gridbook_contract_attests_producer_format(
        contract, "NVFP4_CB_K26"
    )
    assert not gridbook_contract_attests_producer_format(
        contract, "NVFP4_CB_K32"
    )
    assert not gridbook_contract_attests_producer_format(
        contract, "NVFP4_CB_K33"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda e: e.pop("producer_rungs"), "producer_rungs is required"),
        (
            lambda e: e.__setitem__("producer_rungs", [40, 44, 44, 48]),
            "sorted and unique",
        ),
        (
            lambda e: e.__setitem__("producer_rungs", [40, 44, 52]),
            "outside its reader rungs",
        ),
        (
            lambda e: e.__setitem__("producer_rungs", [40, 44, "48"]),
            "positive JSON integers",
        ),
    ],
)
def test_v11_rejects_malformed_or_unversioned_producer_rungs(
    mutation, message
):
    contract = _synthetic_v11()
    mutation(_fp8_entry(contract))
    with pytest.raises(GridbookFormatContractError, match=message):
        gridbook_format_rungs(contract, "FP8_CB_K")

    if "producer_rungs" in _fp8_entry(contract):
        contract["schema"] = "gridbook.runtime-contract.v9"
        contract["contract_version"] = 9
        with pytest.raises(GridbookFormatContractError, match="exactly v4.*v11"):
            gridbook_format_rungs(contract, "FP8_CB_K")


def test_schema_and_contract_version_must_move_together():
    contract = _synthetic_v11()
    contract["contract_version"] = 4
    with pytest.raises(GridbookFormatContractError, match="move together"):
        gridbook_format_rungs(contract, "FP8_CB_K")


def test_v11_must_match_prismaquant_product_ladder_exactly():
    contract = _synthetic_v11()
    _fp8_entry(contract)["producer_rungs"] = [40, 44]
    with pytest.raises(GridbookFormatContractError, match="producer rungs differ"):
        validate_gridbook_cb_rung_contract(contract)
