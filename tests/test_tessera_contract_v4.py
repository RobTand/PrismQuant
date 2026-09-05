"""A v4 development pin reads the same cell grammar as export admission.

The v4 table is a FIXTURE, down-converted from the installed contract by
``conftest.legacy_v4_contract``. Until 2026-09-04 this file read the installed
contract and asserted it WAS v4, which stopped being true when Tessera shipped
``tessera.lane-eligibility.v6``; the reviewed-answer test below still reads the
real installed contract, because the dev pin is an assertion about that file and
about nothing this test made up.
"""
from __future__ import annotations

import copy
import json

import pytest

from prismaquant import tessera_runtime_contract as contract
from prismaquant import tessera_serving_runtime_pin as release


def _packaged():
    """The REAL installed contract -- the dev pin's own subject."""
    return json.loads(contract.contract_path().read_text(encoding="utf-8"))


def _parse(payload):
    return contract._parse(payload, commit="fixture", sha="fixture", path="fixture")


def test_current_v4_contract_keeps_every_launch_and_residency(legacy_v4_contract):
    payload = legacy_v4_contract
    assert payload["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v4"
    parsed = _parse(payload)
    expected = {cell["id"]: cell for cell in payload["lane_eligibility"]["cells"]}
    assert {cell.cell_id for cell in parsed.cells} == set(expected)
    for cell in parsed.cells:
        assert set(cell.executes) == {
            (launch["symbol"], launch["decoder"])
            for launch in expected[cell.cell_id]["executes"]
        }
        flag = next(flag for flag in expected[cell.cell_id]["requires_serve_flags"]
                    if flag.startswith("TESSERA_SERVE_MODE="))
        assert set(cell.residency_modes) == set(flag.split("=", 1)[1].split("|"))


def test_the_reviewed_development_answer_is_the_installed_contracts(monkeypatch):
    """The dev pin reads the REAL contract, and names the same Tessera the
    serving pin does.

    Until 2026-09-04 this test ended by asserting that the development pin did
    not open the RELEASE gate, because that gate refused a PENDING pin. Rob
    retired the tag, so the serving pin is resolved and the two pins now have
    to agree instead -- letting them drift is how two of this repository's own
    spec files came to disagree about one runtime.
    """
    monkeypatch.setenv(contract.TESSERA_DEV_PIN_ENV, "answer-regression")
    parsed = contract.load_tessera_contract()
    assert contract.contract_answer(parsed) == contract.TESSERA_DEV_PIN_ANSWER
    assert parsed.identity()["bytes_are_the_reviewed_bytes"] is True
    assert {cell.structure for cell in parsed.cells} == {
        cell["structure"] for cell in _packaged()["lane_eligibility"]["cells"]
    }
    pin = release.load_tessera_serving_runtime_pin()
    assert contract.TESSERA_DEV_PIN_COMMIT == pin.commit
    assert contract.TESSERA_DEV_PIN_CONTRACT_SHA256 == pin.contract_sha256


def test_launch_change_requires_development_pin_review():
    payload = _packaged()
    original = contract.contract_answer(_parse(payload))
    changed = copy.deepcopy(payload)
    changed["lane_eligibility"]["cells"][0]["executes"][0]["decoder"] = "unreviewed_decoder"
    moved = contract.contract_answer(_parse(changed))
    drift = contract._answer_drift(original, moved)
    assert drift, "a changed served decoder cannot retain the reviewed answer"
    assert "cells[" in "\n".join(drift)


@pytest.mark.parametrize("bad_launches", [
    [],
    [{"symbol": "torch.mm"}],
    [{"symbol": "torch.mm", "decoder": "torch_window", "unknown": True}],
    [{"symbol": "", "decoder": "torch_window"}],
    [{"symbol": "torch.mm", "decoder": "torch_window"}] * 2,
])
def test_development_reader_uses_shared_v4_launch_refusals(bad_launches):
    payload = _packaged()
    payload["lane_eligibility"]["cells"][0]["executes"] = bad_launches
    with pytest.raises(contract.TesseraContractError, match="executes"):
        _parse(payload)


def test_development_reader_refuses_overlapping_residency_cells():
    payload = _packaged()
    duplicate = copy.deepcopy(payload["lane_eligibility"]["cells"][0])
    duplicate["id"] += "_duplicate"
    payload["lane_eligibility"]["cells"].append(duplicate)
    with pytest.raises(contract.TesseraContractError, match="residenc|overlap"):
        _parse(payload)
