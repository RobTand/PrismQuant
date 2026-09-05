"""Lane-eligibility schema v9: the smoke's RECORD, and who derives from it.

Tessera's contract v22 (its #327) moves the lane table to
``tessera.lane-eligibility.v9``. The change is one nullable key inside
``smoke``::

    "record": null
    "record": {"instrument": ..., "rule": ..., "reference": ...,
               "rows": [{"prompt", "form", "interface", "status",
                         "reference_status"}, ...]}

and it exists because of what #327 found in v21: both ``routed_moe`` cells
published ``smoke.status: "recorded"`` on an aggregation rule that lived only
in a dated measurements file, was derived and checked by nothing, and was
satisfiable by an EMPTY completion. v9 puts the rule and the rows it was
applied to inside the contract, where a consumer can re-derive the status
instead of trusting it.

**What this reader must therefore not do is re-implement that rule.** The
grade is re-derived here (from the KL kinds) and the v7 attribution is
re-derived here (from the control) because both are one-line projections this
repository can state exactly; a repetition rule over completion text is not,
and a second implementation of it is precisely the two-homes defect #327 is
about. So on a v9 table PrismaQuant calls Tessera's own
``derive_smoke_status`` / ``derive_smoke_attribution`` and refuses when the
published value disagrees with what they return, and reads
``EVIDENCE_SMOKE_INTERFACES`` / ``EVIDENCE_SMOKE_FORMS`` from Tessera rather
than typing them beside it.

These tests therefore pin the DELEGATION, not the rule: the fixtures make
Tessera's derivation disagree with the published value on purpose and require
the reader to refuse. That is checkable today, against the installed contract
up-converted to v9, and stays true whatever rule Tessera settles on.
"""
from __future__ import annotations

import copy
import json

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_runtime_contract as contract


MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"
MOE_BATCH = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"
RECORDED_RECEIPT = "docs/measurements/moe-smoke-recorded-2026-09-05.md"

#: A record shaped exactly as Tessera #327 describes it. The prompt text is a
#: label, not a completion: the contract records WHICH prompt was run and what
#: the two arms did, never the generated text.
RECORD = {
    "instrument": "tools/tessera_moe_smoke.py",
    "rule": "repetitive iff the completion ends in a cycle of >= 2 full periods",
    "reference": "bf16_source",
    "rows": [
        {"prompt": "capital-of-france", "form": "campaign",
         "interface": "chat_template", "status": "recorded",
         "reference_status": "recorded"},
        {"prompt": "capital-of-france", "form": "pure_greedy",
         "interface": "chat_template", "status": "recorded",
         "reference_status": "recorded"},
        {"prompt": "capital-of-france", "form": "campaign",
         "interface": "raw_completion", "status": "repetitive",
         "reference_status": "repetitive"},
    ],
}


def _installed() -> dict:
    with as_file(contract.contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _cell(payload: dict, cell_id: str) -> dict:
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise AssertionError(f"the installed contract publishes no cell {cell_id!r}")


def _settle_attribution(payload: dict) -> dict:
    """Set every cell's attribution to what Tessera derives for it.

    The fixture must be SELF-CONSISTENT under whichever Tessera is installed:
    the v21 runtime derives from the control alone, the v22 runtime derives
    from the record, and a fixture that typed either answer would start
    failing on the re-pin for a reason that has nothing to do with the reader.
    So the fixture asks the same function the reader asks -- and the tests that
    are ABOUT the derivation make it disagree afterwards, on purpose.
    """
    from tessera.serving import contract as ts

    for cell in payload["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        smoke["attribution"] = ts.derive_smoke_attribution(dict(smoke))
    return payload


def _up_convert(payload: dict, *, record_cells=(MOE_DECODE, MOE_BATCH)) -> dict:
    """The installed table, re-labelled v9 with ``smoke.record`` added.

    Every cell gains the key, because v9 requires it; only the routed-MoE
    cells carry a non-null record, which is the shape Tessera #327 describes
    (a cell nobody re-ran records ``null``).  Written here rather than taken
    from a fixture file so it tracks whatever the pin installs.
    """
    moved = copy.deepcopy(payload)
    moved["lane_eligibility"]["schema"] = lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V9
    for cell in moved["lane_eligibility"]["cells"]:
        cell["evidence"]["smoke"]["record"] = (
            copy.deepcopy(RECORD) if cell["id"] in record_cells else None)
    return _settle_attribution(moved)


def _table(payload: dict):
    block = payload["lane_eligibility"]
    return lane._parse_table(block, payload["formats"], "", "", "x",
                             native_extensions=payload["native_extensions"])


def _parsed_cell(table, cell_id):
    for cell in table.cells:
        if cell.id == cell_id:
            return cell
    raise AssertionError(cell_id)


@pytest.fixture
def tessera_v9(monkeypatch):
    """Tessera's v9 surface, real when the pin has it and stubbed when not.

    The pin moves to the v22 contract in one reviewed commit; until it does,
    the installed ``tessera.serving.contract`` has no v9 helpers, and a test
    that skipped would leave this reader's v9 path unexercised on the branch
    that introduces it.  The stub is deliberately NOT a second implementation
    of the repetition rule: every test below drives it to a chosen answer and
    asserts what this reader does with that answer.
    """
    from tessera.serving import contract as ts

    if not hasattr(ts, "EVIDENCE_SMOKE_INTERFACES"):
        monkeypatch.setattr(
            ts, "EVIDENCE_SMOKE_INTERFACES", ("raw_completion", "chat_template"),
            raising=False)
    if not hasattr(ts, "EVIDENCE_SMOKE_FORMS"):
        monkeypatch.setattr(
            ts, "EVIDENCE_SMOKE_FORMS", ("campaign", "pure_greedy"), raising=False)
    if not hasattr(ts, "derive_smoke_status"):
        def derive_smoke_status(smoke):
            record = smoke.get("record")
            if not record:
                return smoke.get("status")
            statuses = {row["status"] for row in record["rows"]
                        if row["interface"] == "chat_template"}
            return (lane.EVIDENCE_SMOKE_REPETITIVE
                    if lane.EVIDENCE_SMOKE_REPETITIVE in statuses
                    else lane.EVIDENCE_SMOKE_RECORDED)
        monkeypatch.setattr(ts, "derive_smoke_status", derive_smoke_status,
                            raising=False)
    return ts


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
def test_a_v9_table_parses_and_a_null_record_is_not_a_crash(tessera_v9):
    """Eight of the ten cells publish ``record: null``; none of them may throw."""
    table = _table(_up_convert(_installed()))
    assert table.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V9
    null_records = [cell for cell in table.cells if cell.evidence.smoke_record is None]
    assert len(null_records) == len(table.cells) - 2, [c.id for c in table.cells]
    for cell in null_records:
        assert cell.evidence.smoke_status in lane.EVIDENCE_SMOKE_STATUSES


def test_the_record_is_parsed_closed_and_reaches_provenance(tessera_v9):
    table = _table(_up_convert(_installed()))
    record = _parsed_cell(table, MOE_DECODE).evidence.smoke_record
    assert record is not None
    assert record.instrument == RECORD["instrument"]
    assert record.rule == RECORD["rule"]
    assert record.reference == RECORD["reference"]
    assert len(record.rows) == len(RECORD["rows"])
    assert record.as_dict() == RECORD
    # and it is part of the reviewed answer, so a record that moves re-stales
    # the pin rather than sliding in under an unchanged status.
    assert record.answer() in _parsed_cell(table, MOE_DECODE).evidence.answer()


def test_v9_is_scoped_and_carries_every_earlier_evidence_field(tessera_v9):
    for group in (lane.SCOPED_LANE_SCHEMAS, lane.EVIDENCE_LANE_SCHEMAS,
                  lane.ATTRIBUTED_SMOKE_LANE_SCHEMAS,
                  lane.ENCODER_SCOPED_LANE_SCHEMAS,
                  lane.LANE_ELIGIBILITY_SCHEMAS):
        assert lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V9 in group
    for older in (lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V5,
                  lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6,
                  lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V7,
                  lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V8):
        assert older in lane.SCOPED_LANE_SCHEMAS, (
            f"{older} must stay SCOPED when v9 becomes current; a version bump "
            "that demotes a previous grammar to 'legacy unscoped' silently "
            "widens what a legacy table is allowed to attest")


def test_a_record_on_a_v8_table_is_refused_as_an_unknown_field():
    """v9's key is not readable by the v8 grammar, and is not read as one."""
    payload = _installed()
    _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"] = copy.deepcopy(RECORD)
    with pytest.raises(lane.LaneEligibilityError, match="unknown field"):
        _table(payload)


def test_a_v9_table_missing_the_record_key_is_refused(tessera_v9):
    payload = _up_convert(_installed())
    del _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"]
    with pytest.raises(lane.LaneEligibilityError, match="record"):
        _table(payload)


@pytest.mark.parametrize("mutation,match", [
    ({"instrument": ""}, "instrument"),
    ({"rule": ""}, "rule"),
    ({"reference": ""}, "reference"),
    ({"rows": []}, "non-empty"),
])
def test_the_record_head_is_refused_when_it_says_nothing(tessera_v9, mutation, match):
    """A record with no rule, no instrument or no rows attests nothing.

    The empty ``rows`` case is #327's own hole in contract shape: a status of
    ``recorded`` over zero observations is the empty completion again.
    """
    payload = _up_convert(_installed())
    _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"].update(mutation)
    with pytest.raises(lane.LaneEligibilityError, match=match):
        _table(payload)


@pytest.mark.parametrize("field,value", [
    ("form", "whatever_form"),
    ("interface", "whatever_interface"),
    ("status", "fine"),
    ("reference_status", "fine"),
])
def test_a_row_outside_the_published_vocabulary_is_refused(tessera_v9, field, value):
    payload = _up_convert(_installed())
    _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"]["rows"][0][field] = value
    with pytest.raises(lane.LaneEligibilityError, match=field):
        _table(payload)


def test_an_extra_key_on_a_row_is_refused(tessera_v9):
    payload = _up_convert(_installed())
    _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"]["rows"][0]["notes"] = "x"
    with pytest.raises(lane.LaneEligibilityError, match="unknown field"):
        _table(payload)


def test_the_vocabularies_come_from_tessera_and_are_not_typed_here(tessera_v9,
                                                                   monkeypatch):
    """One rule, one home: widen Tessera's vocabulary and this reader widens.

    If the interface list were transcribed into this module, an interface
    Tessera adds would be refused here until somebody re-typed it -- which is
    how the two halves of one contract drift apart.
    """
    payload = _up_convert(_installed())
    _cell(payload, MOE_DECODE)["evidence"]["smoke"]["record"]["rows"][0][
        "interface"] = "harness_replay"
    with pytest.raises(lane.LaneEligibilityError, match="interface"):
        _table(payload)
    monkeypatch.setattr(
        tessera_v9, "EVIDENCE_SMOKE_INTERFACES",
        tuple(tessera_v9.EVIDENCE_SMOKE_INTERFACES) + ("harness_replay",))
    assert _table(payload).schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V9


# ---------------------------------------------------------------------------
# The delegation: who derives the status
# ---------------------------------------------------------------------------
def test_the_status_is_re_derived_through_tesseras_own_function(tessera_v9,
                                                                monkeypatch):
    """The published status must equal what Tessera's rule derives.

    This is the consumer half of #327: v21 published a status no reader could
    check.  Driven by making the derivation disagree, so the test pins the
    delegation and not the rule -- exactly what this repository is entitled to
    assert about somebody else's measurement.
    """
    payload = _up_convert(_installed())
    monkeypatch.setattr(
        tessera_v9, "derive_smoke_status",
        lambda smoke: (lane.EVIDENCE_SMOKE_REPETITIVE
                       if smoke.get("record") else smoke.get("status")))
    with pytest.raises(lane.LaneEligibilityError) as excinfo:
        _table(payload)
    message = str(excinfo.value)
    assert "smoke.status" in message
    assert lane.EVIDENCE_SMOKE_REPETITIVE in message
    assert "Tessera" in message


def test_a_record_free_cell_keeps_the_status_it_publishes(tessera_v9):
    """``record: null`` derives nothing, so the published status stands.

    The eight dense cells are in that state on the up-converted table and are
    parsed unchanged; a reader that treated "no record" as "no evidence" would
    refuse the whole dense lane on a schema bump that said nothing about it.
    """
    table = _table(_up_convert(_installed()))
    installed = _table(_installed())
    for cell in table.cells:
        if cell.evidence.smoke_record is None:
            assert (cell.evidence.smoke_status
                    == _parsed_cell(installed, cell.id).evidence.smoke_status)


def test_the_attribution_is_re_derived_through_tesseras_own_function(tessera_v9,
                                                                     monkeypatch):
    """v9 derives the attribution from the RECORD, so the reader asks Tessera.

    On v7/v8 the attribution is a one-line projection of the control and this
    module mirrors it; on v9 it is a projection of the rows, which is Tessera's
    to state.  Driven the same way: make the derivation disagree, require a
    refusal that names both values.
    """
    payload = _up_convert(_installed())
    monkeypatch.setattr(
        tessera_v9, "derive_smoke_attribution",
        lambda smoke: (lane.EVIDENCE_ATTRIBUTION_NOT_SHARED
                       if smoke.get("record") else lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED),
        raising=False)
    with pytest.raises(lane.LaneEligibilityError) as excinfo:
        _table(payload)
    message = str(excinfo.value)
    assert "attribution" in message
    assert lane.EVIDENCE_ATTRIBUTION_NOT_SHARED in message


def test_the_v7_control_derivation_still_agrees_with_tesseras(tessera_v9):
    """The mirrored v7 rule is checked against its one home rather than assumed.

    ``derive_smoke_attribution`` is transcribed in this module for v7/v8
    tables.  A transcription that silently diverges is the same defect #327
    reports one level down, so it is compared to Tessera's own function here.
    """
    from tessera.serving import contract as ts

    for control, expected in (
            (None, lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED),
            (lane.SmokeControl(reference=lane.EVIDENCE_CONTROL_BF16_SOURCE,
                               outcome=lane.EVIDENCE_OUTCOME_IDENTICAL,
                               receipt="docs/measurements/x.md"),
             lane.EVIDENCE_ATTRIBUTION_SHARED),
            (lane.SmokeControl(reference=lane.EVIDENCE_CONTROL_BF16_SOURCE,
                               outcome=lane.EVIDENCE_OUTCOME_DIFFERENT,
                               receipt="docs/measurements/x.md"),
             lane.EVIDENCE_ATTRIBUTION_NOT_SHARED)):
        assert lane.derive_smoke_attribution(control) == expected
        assert ts.derive_smoke_attribution(
            {"control": control.as_dict() if control else None}) == expected


# ---------------------------------------------------------------------------
# The gate, on a v9 table
# ---------------------------------------------------------------------------
def test_the_admission_predicate_is_unchanged_and_reads_the_v9_status(tessera_v9):
    """v9 changes the grammar, not the rule this repository applies.

    ``cell_evidence_admits`` is still the status-only predicate and
    ``EVIDENCE_SMOKE_REFUSALS`` is still ``{repetitive}``; both routed-MoE
    cells publish ``recorded`` under v9 and are admitted by it, with the same
    predicate that refused them at v17-v20.
    """
    assert lane.EVIDENCE_SMOKE_REFUSALS == frozenset({lane.EVIDENCE_SMOKE_REPETITIVE})
    table = _table(_up_convert(_installed()))
    for cell_id in (MOE_DECODE, MOE_BATCH):
        cell = _parsed_cell(table, cell_id)
        assert cell.evidence.smoke_status == lane.EVIDENCE_SMOKE_RECORDED
        assert lane.cell_evidence_admits(cell) == (True, "")
    for cell in table.cells:
        admitted, why = lane.cell_evidence_admits(cell)
        assert admitted is (
            cell.evidence.smoke_status not in lane.EVIDENCE_SMOKE_REFUSALS), (
                cell.id, why)


def test_a_repetitive_row_set_refuses_the_cell_through_the_same_predicate(
        tessera_v9, monkeypatch):
    """And the other way, so the admission above is a reading and not a constant."""
    payload = _up_convert(_installed())
    for cell_id in (MOE_DECODE, MOE_BATCH):
        smoke = _cell(payload, cell_id)["evidence"]["smoke"]
        smoke["status"] = lane.EVIDENCE_SMOKE_REPETITIVE
        for row in smoke["record"]["rows"]:
            row["status"] = lane.EVIDENCE_SMOKE_REPETITIVE
    table = _table(_settle_attribution(payload))
    for cell_id in (MOE_DECODE, MOE_BATCH):
        admitted, why = lane.cell_evidence_admits(_parsed_cell(table, cell_id))
        assert not admitted
        assert lane.EVIDENCE_SMOKE_REPETITIVE in why and cell_id in why
