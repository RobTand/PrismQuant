"""Shape-free development admission cannot attest a predicated route."""
from __future__ import annotations

import json

import pytest

from prismaquant import tessera_runtime_contract as contract


def test_development_reader_refuses_a_cell_it_cannot_evaluate():
    payload = json.loads(contract.contract_path().read_text(encoding="utf-8"))
    payload["lane_eligibility"]["cells"][0]["predicates"] = [
        {"fact": "in_features", "op": "at_least", "value": 2048}
    ]
    with pytest.raises(contract.TesseraContractError, match="predicates"):
        contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
