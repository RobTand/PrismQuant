"""Lane-eligibility schema v6: per-cell runtime versions, and EVIDENCE.

Tessera's PR #176 moved the lane table to ``tessera.lane-eligibility.v6``.
Three things changed and each one is read here against the INSTALLED contract,
never against a fixture this repository wrote for itself:

* ``versions.attested_on`` -- one global claim about the runtime every cell was
  measured on -- is gone, replaced by ``versions.default_serve_image`` plus a
  per-cell ``runtime {image, execution_modes, vllm, torch}``. The global field
  was false the moment a single cell was measured on a different build, which
  is exactly what happened: the routed-MoE cells were measured on a dev wheel
  inside ``eugr/spark-vllm``, not on the pinned ``vllm/vllm-openai``.
* Every cell carries a required ``evidence {grade, kl, smoke}`` block -- the
  first field in the table that says something about QUALITY rather than
  dispatch.
* The grade is DERIVED from the KL entries' kinds. This reader re-derives it
  rather than reading it as written, because "the grade is read off the
  entries, never asserted beside them" is the publisher's own rule and a
  consumer that trusted the written value would be trusting an assertion where
  a derivation exists (principle 14).

The evidence gate itself is the subject of the second half. What it refuses is
a MEASURED serving defect -- a greedy smoke the runtime recorded as degenerate
-- and not a structure. Nothing in this repository names ``routed_moe`` as
ineligible; the two cells refused today are refused for what was measured on
them.
"""
import copy
import hashlib
import json

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render
from prismaquant import tessera_runtime_contract as contract
from conftest import down_convert_lane_table


FAMILY = "TESSERA_E4M3_K1"
NAME = "TESSERA_E4M3_K1_R1024"
RATE = 1024
MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"
MOE_BATCH = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"


def _raw() -> tuple[dict, str]:
    with as_file(contract.contract_path()) as path:
        raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


@pytest.fixture
def payload():
    return _raw()[0]


@pytest.fixture
def table(payload):
    return lane._parse_table(payload["lane_eligibility"], payload["formats"],
                             "", "", _raw()[1],
                             native_extensions=payload["native_extensions"])


def _parse(payload):
    return contract._parse(payload, commit="fixture", sha="fixture", path="fixture")


def _cell(payload, cell_id):
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise AssertionError(f"the installed contract publishes no cell {cell_id!r}")


def _moe_context(payload):
    return lane.ServingContext(
        platform="sm_121", structure="routed_moe", residency="resident",
        runtime_image=_cell(payload, MOE_DECODE)["runtime"]["image"],
        execution_mode="eager")


def _facts(structure):
    return lane.UnitStructuralFacts(
        qname="fixture.weight", format_name=NAME, payload_family=FAMILY,
        k=None, n_sub=None, rate_q256=RATE, structure=structure,
        role_split=False, in_features=1024, out_features=1024)


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
def test_the_installed_contract_is_read_at_its_own_schema(table, payload):
    """The installed table is SCOPED and reads at the schema it publishes.

    Until 2026-09-05 this asserted the installed schema was v6. Tessera moved
    to v7 and v8 (``test_tessera_lane_v8.py``); a test about v6 owns a v6
    fixture now, exactly as the v4/v5 tests do.
    """
    assert table.schema in lane.SCOPED_LANE_SCHEMAS
    assert table.schema == payload["lane_eligibility"]["schema"]
    assert lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6 in lane.SCOPED_LANE_SCHEMAS
    assert lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V5 in lane.SCOPED_LANE_SCHEMAS, (
        "v5 must stay SCOPED when v6 becomes current; a version bump that "
        "demotes the previous grammar to 'legacy unscoped' silently widens "
        "what a legacy table is allowed to attest")
    assert _parse(payload).lane_schema == table.schema
    v6 = down_convert_lane_table(payload, lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6)
    assert lane._parse_table(v6["lane_eligibility"], v6["formats"], "", "", "x",
                             native_extensions=v6["native_extensions"]
                             ).schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6


def test_every_cell_names_the_build_it_was_measured_under(table):
    for cell in table.cells:
        assert cell.runtime_image and cell.runtime_vllm and cell.runtime_torch, (
            f"{cell.id} publishes no complete v6 runtime scope; the image alone "
            "stopped identifying the build the day a cell was measured on a dev "
            "wheel inside a pinned image")


def test_the_global_attested_on_claim_is_gone_and_the_default_is_read(payload):
    assert "attested_on" not in payload["versions"]
    parsed = _parse(payload)
    assert parsed.default_serve_image == payload["versions"]["default_serve_image"]
    assert parsed.identity()["default_serve_image"] == parsed.default_serve_image


def test_the_cells_do_not_all_share_the_default_serve_image(payload):
    """The reason ``attested_on`` had to go, read off the installed table."""
    images = {c["runtime"]["image"] for c in payload["lane_eligibility"]["cells"]}
    assert len(images) > 1
    assert payload["versions"]["default_serve_image"] in images


def test_a_runtime_block_missing_the_versions_is_refused(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_DECODE)["runtime"].pop("vllm")
    with pytest.raises(lane.LaneEligibilityError, match="missing field"):
        lane._parse_table(broken["lane_eligibility"], broken["formats"], "", "", "x",
                          native_extensions=broken["native_extensions"])


def test_a_cell_with_no_evidence_block_is_refused(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_DECODE).pop("evidence")
    with pytest.raises(lane.LaneEligibilityError, match="missing field"):
        lane._parse_table(broken["lane_eligibility"], broken["formats"], "", "", "x",
                          native_extensions=broken["native_extensions"])


def test_the_grade_is_derived_from_the_entries_never_read_as_written(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_BATCH)["evidence"]["grade"] = "kl_full_vocab"
    with pytest.raises(lane.LaneEligibilityError,
                       match="the grade is read off the entries"):
        lane._parse_table(broken["lane_eligibility"], broken["formats"], "", "", "x",
                          native_extensions=broken["native_extensions"])


def test_a_bound_scored_in_another_regime_is_another_cells_evidence(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_BATCH)["evidence"]["kl"][0]["regime"] = "decode"
    with pytest.raises(lane.LaneEligibilityError,
                       match="is not the cell's regime"):
        lane._parse_table(broken["lane_eligibility"], broken["formats"], "", "", "x",
                          native_extensions=broken["native_extensions"])


def test_an_unknown_smoke_status_is_refused_never_read_as_passing(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_DECODE)["evidence"]["smoke"]["status"] = "looked_fine"
    with pytest.raises(lane.LaneEligibilityError, match="smoke.status must be one of"):
        lane._parse_table(broken["lane_eligibility"], broken["formats"], "", "", "x",
                          native_extensions=broken["native_extensions"])


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_the_routed_moe_cells_are_refused_on_their_own_recorded_smoke(table):
    refused = {c.id: lane.cell_evidence_admits(c) for c in table.cells
               if not lane.cell_evidence_admits(c)[0]}
    assert set(refused) == {MOE_DECODE, MOE_BATCH}
    for cell_id, (_, why) in refused.items():
        assert "repetitive" in why and cell_id in why
        assert "generate correctly" in why


def test_no_dense_cell_is_refused_so_the_shipping_menu_does_not_move(table):
    for cell in table.cells:
        if cell.structure == lane.STRUCTURE_DENSE:
            assert lane.cell_evidence_admits(cell)[0], cell.id


def test_the_gate_is_evidence_not_structure(table, payload):
    """Flip the smoke, and the same routed-MoE cells admit.

    This is what makes the refusal a MEASURED fact rather than a ban: nothing
    in the predicate mentions ``routed_moe``, so the day Tessera records a
    clean smoke the cell stops being refused HERE -- and is stopped instead by
    the pin, which is a review event with a human in it.
    """
    healed = copy.deepcopy(payload)
    for cell_id in (MOE_DECODE, MOE_BATCH):
        _cell(healed, cell_id)["evidence"]["smoke"]["status"] = "recorded"
    parsed = lane._parse_table(healed["lane_eligibility"], healed["formats"],
                               "", "", "x",
                               native_extensions=healed["native_extensions"])
    assert all(lane.cell_evidence_admits(c)[0] for c in parsed.cells)


def test_the_export_gate_refuses_the_routed_moe_unit_and_names_the_cell(table, payload):
    route = lane.resolve_unit_route(
        _facts("routed_moe"), table, platform="sm_121", residency="resident",
        runtime_image=_moe_context(payload).runtime_image, execution_mode="eager")
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
    named = {r.cell_id for r in route.regimes}
    assert named == {MOE_DECODE, MOE_BATCH}, (
        "a cell that exists and was refused on measured evidence must not read "
        "as a cell that does not exist")
    for regime in route.regimes:
        assert "repetitive" in regime.detail


def test_the_dense_unit_still_resolves_backed_with_its_grade_recorded(table, payload):
    image = _cell(payload, "tessera_e4m3_k1_dense_sm121_decode_resident")["runtime"]["image"]
    route = lane.resolve_unit_route(
        _facts("dense"), table, platform="sm_121", residency="resident",
        runtime_image=image, execution_mode="eager")
    assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    grades = {r.evidence_grade for r in route.regimes}
    assert grades and "" not in grades, (
        "principle 12: a shipcard has to say which grade attested each unit")
    assert all(r.as_dict()["evidence_grade"] for r in route.regimes)


def test_the_development_menu_reads_the_same_predicate(payload):
    parsed = _parse(payload)
    assert parsed.native_cells(FAMILY, RATE,
                               serving_context=_moe_context(payload)) == ()


def test_the_renderer_names_the_evidence_when_it_refuses(payload, table, monkeypatch):
    formats = {row["family"]: row for row in payload["formats"]}
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    admitted, reason = render.tessera_lane_admission(
        NAME, serving_context=_moe_context(payload))
    assert not admitted
    assert "evidence refuses this route" in reason
    assert "repetitive" in reason


def test_the_evidence_block_is_part_of_the_reviewed_answer(payload):
    """A flipped smoke must re-stale the pin, not silently promote a route."""
    before = contract.contract_answer(_parse(payload))
    moved = copy.deepcopy(payload)
    _cell(moved, MOE_DECODE)["evidence"]["smoke"]["status"] = "recorded"
    after = contract.contract_answer(_parse(moved))
    assert before != after
    assert contract._answer_drift(before, after)
