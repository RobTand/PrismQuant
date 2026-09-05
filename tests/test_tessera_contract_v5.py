"""V5 development admission retains scope and cannot invent a target."""
import copy
import json
import os
from pathlib import Path

import pytest
from conftest import down_convert_lane_table

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_runtime_contract as contract


DENSE_IMAGE = "example/dense@sha256:" + "a" * 64
MOE_IMAGE = "example/moe@sha256:" + "b" * 64


def _payload():
    """A v5 lane table this file OWNS rather than borrows.

    This file is about the v5 grammar. Relabelling whatever Tessera happens to
    publish made it assert, silently, that the installed contract still IS v5 --
    and since Tessera's PR #176 it is v6, whose cells carry an ``evidence``
    block v5 has no field for. Down-converting keeps every part the grammar did
    not change (families, rungs, route statuses) honest and real, and drops
    exactly the newer fields, so the test exercises v5 and nothing else.
    """
    payload = down_convert_lane_table(
        json.loads(contract.contract_path().read_text(encoding="utf-8")),
        "tessera.lane-eligibility.v5")
    # Dense-only, decided here rather than inherited: v6 publishes routed_moe
    # cells too, and this file's scope tests ask what a routed_moe CONTEXT gets
    # from a table that covers only dense.
    block = payload["lane_eligibility"]
    block["cells"] = [c for c in block["cells"] if c["structure"] == "dense"]
    block["structures"] = [s for s in block["structures"] if s == "dense"]
    for cell in payload["lane_eligibility"]["cells"]:
        cell["runtime"] = {"image": DENSE_IMAGE, "execution_modes": ["eager"]}
    return payload


def _parse(payload):
    return contract._parse(payload, commit="fixture", sha="fixture", path="fixture")


def _context(**changed):
    fields = dict(platform="sm_121", structure="dense", residency="resident",
                  runtime_image=DENSE_IMAGE, execution_mode="eager")
    fields.update(changed)
    return lane.ServingContext(**fields)


def test_v5_development_requires_context_and_retains_exact_cell_scope():
    parsed = _parse(_payload())
    assert parsed.lane_schema == "tessera.lane-eligibility.v5"
    assert parsed.requires_serving_context is True
    assert parsed.native_cells("TESSERA_E4M3_K1", 1024) == ()
    selected = parsed.native_cells("TESSERA_E4M3_K1", 1024, serving_context=_context())
    assert {cell.regime for cell in selected} == set(parsed.regimes)
    assert all(cell.runtime_image == DENSE_IMAGE for cell in selected)
    assert all(cell.execution_modes == ("eager",) for cell in selected)


@pytest.mark.parametrize("changed", [
    {"runtime_image": MOE_IMAGE}, {"execution_mode": "compiled"},
    {"structure": "routed_moe"}, {"platform": "sm_120"},
])
def test_v5_development_cannot_borrow_cells_from_another_scope(changed):
    parsed = _parse(_payload())
    assert parsed.native_cells("TESSERA_E4M3_K1", 1024, serving_context=_context(**changed)) == ()


def test_v5_development_requires_all_regimes_on_one_runtime():
    payload = _payload()
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["regime"] == "batch":
            cell["runtime"]["image"] = MOE_IMAGE
    parsed = _parse(payload)
    assert parsed.native_cells("TESSERA_E4M3_K1", 1024, serving_context=_context()) == ()


@pytest.mark.parametrize("field,value", [
    ("image", MOE_IMAGE), ("execution_modes", ["eager", "compiled"]),
])
def test_v5_runtime_scope_changes_require_answer_review(field, value):
    payload = _payload()
    before = contract.contract_answer(_parse(payload))
    changed = copy.deepcopy(payload)
    changed["lane_eligibility"]["cells"][0]["runtime"][field] = value
    after = contract.contract_answer(_parse(changed))
    assert before["lane_schema"] == "tessera.lane-eligibility.v5"
    assert contract._answer_drift(before, after)


def test_v5_development_uses_shared_runtime_grammar():
    payload = _payload()
    payload["lane_eligibility"]["cells"][0]["runtime"]["execution_modes"] = ["automatic"]
    with pytest.raises(contract.TesseraContractError, match="runtime.*execution_modes"):
        _parse(payload)


def test_v5_required_regimes_are_part_of_the_reviewed_answer():
    payload = _payload()
    before = contract.contract_answer(_parse(payload))
    payload["lane_eligibility"]["regimes"].append("additional_required_regime")
    after = contract.contract_answer(_parse(payload))
    assert contract._answer_drift(before, after)


def test_explicit_publisher_v5_contract_admits_only_complete_matching_contexts():
    """Interop uses explicit immutable publisher bytes, never a vendored copy."""
    path = os.environ.get("PRISMAQUANT_TEST_TESSERA_V5_CONTRACT")
    if not path:
        pytest.skip("explicit immutable v5 publisher contract was not supplied")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v5"
    parsed = _parse(payload)
    # Required scope is not inferred from whichever native cell appears first.
    for cell in parsed.cells:
        for rate in cell.rungs_q256:
            assert parsed.native_cells(cell.family, rate) == ()
    assert parsed.requires_serving_context
    positive_contexts = set()
    for cell in parsed.cells:
        if not cell.native:
            continue
        for rate in cell.rungs_q256:
            for residency in cell.residency_modes:
                for execution in cell.execution_modes:
                    context = lane.ServingContext(
                        platform=cell.platform, structure=cell.structure,
                        residency=residency, runtime_image=cell.runtime_image,
                        execution_mode=execution)
                    expected = tuple(other for other in parsed.cells
                                     if other.native and other.family == cell.family
                                     and rate in other.rungs_q256
                                     and lane.cell_matches_serving_context(other, context))
                    complete = {other.regime for other in expected} == set(parsed.regimes)
                    observed = parsed.native_cells(cell.family, rate, serving_context=context)
                    assert observed == (expected if complete else ())
                    if complete:
                        positive_contexts.add(context.key())
    assert positive_contexts, "the supplied publisher contract has no complete positive scope"
