"""Shared pytest fixtures.

Deliberately minimal: this repo had no conftest before 2026-08-30. It carried
one NON-autouse fixture, ``synthetic_cb_target``, which let a codebook export
test declare that its bodies were CPU fixtures rather than served artifacts,
through the route-status gate's own ``PQ_CB_NON_NATIVE_TARGET`` declaration.
That gate, that declaration and the export tests that used it all went into
``archive/gridbook_lane_2026-09-02/`` when the Gridbook codebook serving lane
was retired on 2026-09-02, so the fixture has no subject left. Nothing here
was ever autouse, so collection semantics are unchanged either way.
"""
from __future__ import annotations

import copy
import json

import pytest


def _installed_contract() -> dict:
    """The Tessera contract the environment installs, parsed."""
    from importlib.resources import as_file

    from prismaquant import tessera_runtime_contract as contract

    with as_file(contract.contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def down_convert_lane_table(payload: dict, schema: str) -> dict:
    """The installed contract, expressed in an OLDER lane grammar.

    Legacy-grammar tests need a legacy table. They used to get one by reading
    the installed contract, which was v4 at the time -- so on the day Tessera
    shipped ``tessera.lane-eligibility.v6`` (its PR #176) every such test began
    asserting that the installed contract was a version it no longer is. That
    is a fixture problem, not a reader problem: a test about a legacy grammar
    must OWN its legacy fixture, exactly as ``test_tessera_contract_v5`` already
    did by rewriting ``runtime`` onto every cell.

    Down-converting rather than hand-writing keeps the fixture honest about
    everything the grammar did not change -- families, rungs, route statuses,
    launches -- so the test still exercises real cells. It is a FIXTURE and
    never an attestation: nothing derived from it is recorded anywhere.

    ``schema`` is ``tessera.lane-eligibility.v5`` (drop the v6 ``evidence``
    block and the ``runtime`` version fields) or ``...v4`` (drop the whole
    per-cell ``runtime`` scope as well).
    """
    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    lane["schema"] = schema
    for cell in lane["cells"]:
        cell.pop("evidence", None)
        if schema.endswith(".v4"):
            cell.pop("runtime", None)
        else:
            runtime = cell.get("runtime", {})
            cell["runtime"] = {"image": runtime["image"],
                               "execution_modes": runtime["execution_modes"]}
    return payload


@pytest.fixture
def legacy_v4_contract() -> dict:
    """The installed contract expressed as a v4 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v4")


@pytest.fixture
def legacy_v5_contract() -> dict:
    """The installed contract expressed as a v5 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v5")
