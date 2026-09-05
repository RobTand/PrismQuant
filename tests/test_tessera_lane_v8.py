"""Lane-eligibility schemas v7 and v8: a smoke's CONTROL, and the ENCODER SCOPE.

Tessera's contract v18 (its #195) moved the lane table to
``tessera.lane-eligibility.v7`` and v19 (its #198) to ``v8``. Both are read
here against the INSTALLED contract, never against a fixture this repository
wrote for itself, exactly as ``test_tessera_lane_v6.py`` reads v6.

* v7: ``smoke`` gains ``control`` -- the reference a greedy smoke was compared
  against, ``null`` when nobody ran one -- and ``attribution``, DERIVED from
  the control and checked, exactly as the grade is derived from the KL
  entries. ``shared_with_reference`` says the unquantised source returned the
  same completion; it does not say the route generated correctly.
* v8: ``evidence`` gains ``artifact`` -- ``null`` when no encoder-reproduction
  comparison was recorded, otherwise the historical artifact the cell's KL was
  measured on, the encoder commit that wrote it, and a single-unit re-encode
  screen at a named later commit. It never changes the grade.

What this reader does with the new fields is the subject of the second half:
it parses them closed, refuses by name what it does not understand, carries
them into the refusal text and into route provenance, and does NOT adopt the
consumer rule Tessera's v18 changelog states ("refuses on status 'repetitive'
AND attribution other than 'shared_with_reference'"). On the v20 table that
rule would have ADMITTED the two routed-MoE cells on the strength of a control
alone, which is a promotion decision and not a reader's; prismaquant #198 put
it to Rob and Tessera answered with a measurement: contract v21 (its #313)
re-ran the smoke through the checkpoint's own chat template, both cells read
``status: recorded`` with the control retired (``control: null``,
``attribution: unattributed`` -- the shape the dense cells already used), and
the unchanged status-only rule stops refusing them.

**What these tests therefore do NOT assert is that routed MoE is admitted.**
That verdict is not settled: RobTand/tessera#327 (P1) reports v21's
``recorded`` rests on a repetition rule that lives only in a dated
measurements file, is derived and checked by nothing, and is satisfiable by an
empty completion, so the published status may move again; prismaquant #198
stays open for the promotion either way. The gate tests below assert the
MECHANISM instead -- that the predicate's answer is a function of the status
the pinned table publishes, and that the export gate reads that same answer
rather than forming its own -- which holds under either verdict and fails if
this reader ever starts deciding routed MoE for itself.

Since v21 no installed cell carries a control, so the control GRAMMAR is
exercised here on the installed routed-MoE cells with the v20 control
transplanted back (``_with_v20_control``: the exact block v20 published), and
the refusal text on that same shape. Everything about the installed table is
still read off the installed table.
"""
import copy
import hashlib
import json

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_runtime_contract as contract
from conftest import down_convert_lane_table


FAMILY = "TESSERA_E4M3_K1"
NAME = "TESSERA_E4M3_K1_R1024"
RATE = 1024
MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"
MOE_BATCH = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"
DENSE_DECODE = "tessera_e4m3_k1_dense_sm121_decode_resident"
E2M1_DECODE = "tessera_e2m1_k2_dense_sm121_decode"
CONTROL_RECEIPT = "docs/measurements/moe-evidence-debt-2026-09-04.md"
RECORDED_RECEIPT = "docs/measurements/moe-smoke-recorded-2026-09-05.md"
REPETITIVE_RECEIPT = "docs/measurements/tessera-lfm-campaign-2026-09-04.md"

#: The smoke block contract v20 published on both routed-MoE cells (its #195
#: control over the v17 degenerate smoke), verbatim. v21 retired it.
V20_SMOKE = {
    "status": "repetitive",
    "receipt": REPETITIVE_RECEIPT,
    "attribution": "shared_with_reference",
    "control": {"reference": "bf16_source", "outcome": "identical_completion",
                "receipt": CONTROL_RECEIPT},
}


def _raw() -> tuple[dict, str]:
    with as_file(contract.contract_path()) as path:
        raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


@pytest.fixture
def payload():
    return _raw()[0]


def _table(payload):
    return lane._parse_table(payload["lane_eligibility"], payload["formats"],
                             "", "", "x", native_extensions=payload["native_extensions"])


@pytest.fixture
def table(payload):
    return _table(payload)


def _parse(payload):
    return contract._parse(payload, commit="fixture", sha="fixture", path="fixture")


def _cell(payload, cell_id):
    for cell in payload["lane_eligibility"]["cells"]:
        if cell["id"] == cell_id:
            return cell
    raise AssertionError(f"the installed contract publishes no cell {cell_id!r}")


def _parsed_cell(table, cell_id):
    for cell in table.cells:
        if cell.id == cell_id:
            return cell
    raise AssertionError(cell_id)


def _with_v20_control(payload, *cell_ids):
    """The installed table with v20's smoke transplanted onto routed-MoE cells."""
    moved = copy.deepcopy(payload)
    for cell_id in cell_ids or (MOE_DECODE, MOE_BATCH):
        _cell(moved, cell_id)["evidence"]["smoke"] = copy.deepcopy(V20_SMOKE)
    return moved


def _facts(structure):
    return lane.UnitStructuralFacts(
        qname="fixture.weight", format_name=NAME, payload_family=FAMILY,
        k=None, n_sub=None, rate_q256=RATE, structure=structure,
        role_split=False, in_features=1024, out_features=1024)


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
def test_the_installed_contract_is_read_at_its_own_schema(table, payload):
    assert table.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V8
    assert lane.LANE_ELIGIBILITY_SCHEMA_TESSERA == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V8
    for older in (lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V5,
                  lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6,
                  lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V7):
        assert older in lane.SCOPED_LANE_SCHEMAS, (
            f"{older} must stay SCOPED when v8 becomes current; a version bump "
            "that demotes a previous grammar to 'legacy unscoped' silently "
            "widens what a legacy table is allowed to attest")
    assert _parse(payload).lane_schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V8


def test_every_cell_publishes_a_derived_attribution(table):
    for cell in table.cells:
        assert cell.evidence is not None
        assert cell.evidence.smoke_attribution in lane.EVIDENCE_SMOKE_ATTRIBUTIONS
        assert cell.evidence.smoke_attribution == lane.derive_smoke_attribution(
            cell.evidence.smoke_control)


def test_the_routed_moe_cells_read_recorded_with_the_v20_control_retired(table):
    """v21 (Tessera #313): the smoke was re-measured, not re-attributed.

    This records what the PINNED bytes publish -- which is what a pin is for,
    and it moves only in a reviewed re-pin -- not what those bytes are worth.
    RobTand/tessera#327 questions the second thing; see the module docstring.
    """
    for cell_id in (MOE_DECODE, MOE_BATCH):
        evidence = _parsed_cell(table, cell_id).evidence
        assert evidence.smoke_status == lane.EVIDENCE_SMOKE_RECORDED
        assert evidence.smoke_receipt == RECORDED_RECEIPT
        assert evidence.smoke_control is None
        assert evidence.smoke_attribution == lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED


def test_the_v20_control_reads_as_v20_published_it(payload):
    table = _table(_with_v20_control(payload))
    for cell_id in (MOE_DECODE, MOE_BATCH):
        evidence = _parsed_cell(table, cell_id).evidence
        assert evidence.smoke_status == lane.EVIDENCE_SMOKE_REPETITIVE
        assert evidence.smoke_attribution == lane.EVIDENCE_ATTRIBUTION_SHARED
        assert evidence.smoke_control is not None
        assert evidence.smoke_control.reference == lane.EVIDENCE_CONTROL_BF16_SOURCE
        assert evidence.smoke_control.outcome == lane.EVIDENCE_OUTCOME_IDENTICAL
        assert evidence.smoke_control.receipt == CONTROL_RECEIPT


def test_every_installed_cell_carries_no_control_and_reads_unattributed(table):
    for cell in table.cells:
        assert cell.evidence.smoke_control is None, cell.id
        assert cell.evidence.smoke_attribution == lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED


def test_an_attribution_its_control_does_not_derive_is_refused(payload):
    broken = _with_v20_control(payload)
    _cell(broken, MOE_DECODE)["evidence"]["smoke"]["attribution"] = "unattributed"
    with pytest.raises(lane.LaneEligibilityError,
                       match="the attribution is read off the control"):
        _table(broken)


def test_an_unknown_attribution_is_refused_by_name(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, MOE_DECODE)["evidence"]["smoke"]["attribution"] = "looked_like_the_model"
    with pytest.raises(lane.LaneEligibilityError,
                       match="smoke.attribution must be one of"):
        _table(broken)


def test_a_not_recorded_smoke_that_names_a_control_is_refused(payload):
    broken = copy.deepcopy(payload)
    smoke = _cell(broken, E2M1_DECODE)["evidence"]["smoke"]
    assert smoke["status"] == "not_recorded"
    smoke["control"] = dict(V20_SMOKE["control"])
    smoke["attribution"] = "shared_with_reference"
    with pytest.raises(lane.LaneEligibilityError,
                       match="nothing for a reference to have matched"):
        _table(broken)


@pytest.mark.parametrize("member,value,expect", [
    ("reference", "fp16_source", "control.reference must be one of"),
    ("outcome", "similar_completion", "control.outcome must be one of"),
    ("receipt", "notes/moe.md", "control.receipt must be a repository path"),
])
def test_a_control_this_reader_cannot_name_is_refused(payload, member, value, expect):
    broken = _with_v20_control(payload)
    _cell(broken, MOE_DECODE)["evidence"]["smoke"]["control"][member] = value
    with pytest.raises(lane.LaneEligibilityError, match=expect):
        _table(broken)


def test_a_control_with_a_field_this_reader_does_not_know_is_refused(payload):
    broken = _with_v20_control(payload)
    _cell(broken, MOE_DECODE)["evidence"]["smoke"]["control"]["scope"] = "same prompt"
    with pytest.raises(lane.LaneEligibilityError, match="unknown field"):
        _table(broken)


def test_a_smoke_block_without_the_v7_fields_is_refused_at_v8(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, DENSE_DECODE)["evidence"]["smoke"].pop("attribution")
    with pytest.raises(lane.LaneEligibilityError, match="missing field"):
        _table(broken)


def test_a_cell_without_the_v8_artifact_field_is_refused(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, DENSE_DECODE)["evidence"].pop("artifact")
    with pytest.raises(lane.LaneEligibilityError, match="missing field"):
        _table(broken)


def test_the_e4m3_dense_cells_scope_their_kl_to_the_encoder_that_wrote_it(table):
    """v19's correction, read off the installed table.

    The four dense E4M3 cells name the checkpoint their KL was measured on and
    the encoder commit that wrote it; a same-source re-encode at a later commit
    produced a DIFFERENT payload. That is a fact a shipcard has to carry: the
    KL number attests bytes the current encoder does not reproduce.
    """
    scoped = {c.id: c.evidence.artifact for c in table.cells
              if c.evidence.artifact is not None}
    assert scoped, "no cell scopes its evidence to an encoder"
    for cell_id, artifact in scoped.items():
        assert cell_id.startswith("tessera_e4m3_k1_dense_"), cell_id
        assert len(artifact.encoder_commit) == 40
        assert len(artifact.reencode_encoder_commit) == 40
        assert artifact.reencode_payload in lane.EVIDENCE_PAYLOAD_RELATIONS
        assert artifact.reencode_weight_error in lane.EVIDENCE_WEIGHT_ERROR_RELATIONS
        assert artifact.reencode_receipt.startswith(lane.EVIDENCE_RECEIPT_ROOT)
        if artifact.reencode_payload == lane.EVIDENCE_PAYLOAD_IDENTICAL:
            assert artifact.reencode_weight_error == lane.EVIDENCE_WEIGHT_ERROR_EQUAL


def test_an_identical_payload_with_a_moved_weight_error_is_refused(payload):
    broken = copy.deepcopy(payload)
    reencode = _cell(broken, DENSE_DECODE)["evidence"]["artifact"]["reencode"]
    reencode["payload"] = "identical"
    reencode["weight_error"] = "lower"
    with pytest.raises(lane.LaneEligibilityError,
                       match="weight_error must be 'equal' for an identical payload"):
        _table(broken)


@pytest.mark.parametrize("path,value,expect", [
    (("encoder_commit",), "8070ec6", "encoder_commit must be a full lowercase"),
    (("reencode", "encoder_commit"), "3317036", "encoder_commit must be a full lowercase"),
    (("id",), "../checkpoints/x", "portable relative artifact identifier"),
    (("reencode", "unit"), "  ", "unit must name the single unit compared"),
    (("reencode", "metric"), "served_kl", "metric must be 'weight_sse'"),
    (("reencode", "payload"), "similar", "payload must be one of"),
    (("reencode", "weight_error"), "much_lower", "weight_error must be one of"),
    (("reencode", "receipt"), "notes/x.md", "receipt must be a repository path"),
])
def test_a_malformed_artifact_is_refused_by_name(payload, path, value, expect):
    broken = copy.deepcopy(payload)
    node = _cell(broken, DENSE_DECODE)["evidence"]["artifact"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(lane.LaneEligibilityError, match=expect):
        _table(broken)


def test_an_artifact_with_a_field_this_reader_does_not_know_is_refused(payload):
    broken = copy.deepcopy(payload)
    _cell(broken, DENSE_DECODE)["evidence"]["artifact"]["reencode"]["kl"] = 0.01
    with pytest.raises(lane.LaneEligibilityError, match="unknown field"):
        _table(broken)


# ---------------------------------------------------------------------------
# Older grammars keep their own fixtures
# ---------------------------------------------------------------------------
def test_a_v7_table_reads_with_no_artifact_and_a_v6_table_with_no_control(payload):
    payload = _with_v20_control(payload)      # a v7 table with a control to lose
    v7 = down_convert_lane_table(payload, lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V7)
    parsed = _table(v7)
    assert parsed.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V7
    moe = _parsed_cell(parsed, MOE_DECODE).evidence
    assert moe.artifact is None
    assert moe.smoke_attribution == lane.EVIDENCE_ATTRIBUTION_SHARED

    v6 = down_convert_lane_table(payload, lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6)
    parsed = _table(v6)
    assert parsed.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6
    moe = _parsed_cell(parsed, MOE_DECODE).evidence
    assert moe.artifact is None
    assert moe.smoke_control is None
    assert moe.smoke_attribution == "", (
        "a v6 table published no attribution; reading one as 'unattributed' "
        "would put a v7 word into the mouth of a grammar that never spoke it")


def test_a_v7_field_on_a_v6_table_is_refused(payload):
    v6 = down_convert_lane_table(payload, lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V6)
    _cell(v6, MOE_DECODE)["evidence"]["smoke"]["attribution"] = "unattributed"
    with pytest.raises(lane.LaneEligibilityError, match="unknown field"):
        _table(v6)


# ---------------------------------------------------------------------------
# The gate: what the reader does with what it now reads
# ---------------------------------------------------------------------------
def test_the_routed_moe_cells_are_decided_by_the_status_the_table_publishes(table):
    """The predicate is a function of the PUBLISHED status, and of nothing else.

    Deliberately NOT "routed MoE is admitted". What the pinned table says about
    those two cells is Tessera's to move -- v17 through v20 published
    ``repetitive`` and this predicate refused them, v21 publishes ``recorded``
    and it does not -- and what v21's ``recorded`` is worth is itself open
    (RobTand/tessera#327: the repetition rule behind it lives only in a dated
    measurements file, is derived and checked by nothing, and is satisfiable by
    an empty completion; prismaquant #198 holds the promotion). Asserting the
    outcome here would pin a verdict this repository does not own and would
    have to be rewritten whichever way #327 lands.

    What this repository DOES owe is that the answer track the published status
    and nothing else, so that is what is asserted: read the status off the
    installed cell, and require the predicate to agree with it. It holds under
    either verdict, and it fails the moment the reader starts deciding routed
    MoE on the structure, the grade, the attribution or an edit.
    """
    for cell_id in (MOE_DECODE, MOE_BATCH):
        cell = _parsed_cell(table, cell_id)
        published = cell.evidence.smoke_status
        admitted, why = lane.cell_evidence_admits(cell)
        assert admitted is (published not in lane.EVIDENCE_SMOKE_REFUSALS), (
            cell_id, published, why)
        # and the answer is spelled the same way a dense cell's is: no
        # structure-specific text on either leg.
        assert (why == "") is admitted
        if not admitted:
            assert published in why and cell_id in why
            assert "routed_moe" not in why


def test_the_routed_moe_refusal_names_the_control_and_the_rule_it_did_not_apply(payload):
    """The v20 shape, read by the same rule: refused, and the refusal says
    it read the control and chose not to decide on it."""
    table = _table(_with_v20_control(payload))
    for cell_id in (MOE_DECODE, MOE_BATCH):
        admitted, why = lane.cell_evidence_admits(_parsed_cell(table, cell_id))
        assert not admitted
        assert "repetitive" in why and cell_id in why
        assert "shared_with_reference" in why
        assert "bf16_source" in why and "identical_completion" in why
        assert CONTROL_RECEIPT in why
        assert "#198" in why, (
            "the refusal must say that PrismaQuant read the attribution and "
            "chose not to decide on it, and where that decision is held")


def test_no_dense_cell_is_refused(table):
    for cell in table.cells:
        if cell.structure == lane.STRUCTURE_DENSE:
            assert lane.cell_evidence_admits(cell)[0], cell.id


def test_the_status_alone_decides_whatever_the_attribution_says(payload):
    """The rule this repository applies, stated as a test so a change is a review.

    Tessera's v18 changelog expected a consumer to admit ``repetitive`` when
    the attribution is ``shared_with_reference``. This reader does not: that
    would admit a structure this producer has never shipped on the strength of
    a smoke whose only recorded outcome degenerated (prismaquant #198). The
    status alone decides -- ``repetitive`` refuses under EITHER attribution.

    The positive leg is asserted the same way, on a status transplanted onto
    the same cell rather than on whatever the pin happens to publish, so this
    test states the rule and not a verdict about routed MoE (RobTand/tessera#327
    may move the published status either way).
    """
    shared = _with_v20_control(payload)
    not_shared = _with_v20_control(payload)
    smoke = _cell(not_shared, MOE_DECODE)["evidence"]["smoke"]
    smoke["control"]["outcome"] = "different_completion"
    smoke["attribution"] = "not_shared_with_reference"
    for fixture in (shared, not_shared):
        cell = _parsed_cell(_table(fixture), MOE_DECODE)
        assert not lane.cell_evidence_admits(cell)[0]
    # ...and the same cell with a non-refused status admits, whatever the
    # attribution says, because the status is the whole rule.
    recorded = _with_v20_control(payload)
    _cell(recorded, MOE_DECODE)["evidence"]["smoke"] = {
        "status": lane.EVIDENCE_SMOKE_RECORDED, "receipt": RECORDED_RECEIPT,
        "attribution": lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED, "control": None}
    assert lane.cell_evidence_admits(_parsed_cell(_table(recorded), MOE_DECODE))[0]


def test_the_export_gate_records_the_attribution_beside_the_refusal(payload):
    image = _cell(payload, MOE_DECODE)["runtime"]["image"]
    route = lane.resolve_unit_route(
        _facts("routed_moe"), _table(_with_v20_control(payload)),
        platform="sm_121", residency="resident",
        runtime_image=image, execution_mode="eager")
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
    for regime in route.regimes:
        assert regime.cell_id in (MOE_DECODE, MOE_BATCH)
        assert "shared_with_reference" in regime.detail


def test_the_export_gate_answers_the_routed_moe_unit_from_the_same_predicate(
        table, payload):
    """The export gate reads ``cell_evidence_admits``, it does not re-decide.

    Principle 8: a rung the menu offers and the export refuses (or the reverse)
    is the split brain the one-predicate design exists to stop. So the expected
    route status is COMPUTED from the predicate's answer on the two cells the
    scope selects, never typed -- which keeps this test honest whichever way
    the pinned table's routed-MoE smoke status lands (RobTand/tessera#327,
    prismaquant #198), and still fails if the gate ever grows a second opinion.
    Provenance is asserted either way, because a shipcard has to record what
    attested (or refused) a unit whatever the verdict was.
    """
    image = _cell(payload, MOE_DECODE)["runtime"]["image"]
    admits = {cell_id: lane.cell_evidence_admits(_parsed_cell(table, cell_id))[0]
              for cell_id in (MOE_DECODE, MOE_BATCH)}
    route = lane.resolve_unit_route(
        _facts("routed_moe"), table, platform="sm_121", residency="resident",
        runtime_image=image, execution_mode="eager")
    expected = (lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG if all(admits.values())
                else lane.ROUTE_STATUS_UNATTESTED)
    assert route.route_status == expected, (admits, route.route_status)
    assert {r.cell_id for r in route.regimes} == {MOE_DECODE, MOE_BATCH}
    for regime in route.regimes:
        recorded = regime.as_dict()
        assert recorded["evidence_attribution"] == (
            _parsed_cell(table, regime.cell_id).evidence.smoke_attribution)
        assert recorded["evidence_artifact"] is None


def test_the_dense_route_carries_attribution_and_encoder_scope_into_provenance(table, payload):
    image = _cell(payload, DENSE_DECODE)["runtime"]["image"]
    route = lane.resolve_unit_route(
        _facts("dense"), table, platform="sm_121", residency="resident",
        runtime_image=image, execution_mode="eager")
    assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    for regime in route.regimes:
        recorded = regime.as_dict()
        assert recorded["evidence_attribution"] == lane.EVIDENCE_ATTRIBUTION_UNATTRIBUTED
        artifact = recorded["evidence_artifact"]
        assert artifact is not None, (
            "principle 12: the shipcard has to say which encoder wrote the "
            "bytes this unit's KL was measured on")
        assert set(artifact) == {"id", "encoder_commit", "reencode"}
        assert set(artifact["reencode"]) == {
            "encoder_commit", "unit", "payload", "metric", "weight_error", "receipt"}


def test_the_attribution_and_artifact_are_part_of_the_reviewed_answer(payload):
    """A moved control or a moved encoder scope must re-stale the pin."""
    before = contract.contract_answer(_parse(payload))

    # v20's control back on the cell: the pin the v21 re-pin replaced.
    moved = _with_v20_control(payload, MOE_DECODE)
    after = contract.contract_answer(_parse(moved))
    assert before != after and contract._answer_drift(before, after)

    # and a moved control outcome/attribution on that shape moves it again.
    moved_again = _with_v20_control(payload, MOE_DECODE)
    smoke = _cell(moved_again, MOE_DECODE)["evidence"]["smoke"]
    smoke["control"]["outcome"] = "different_completion"
    smoke["attribution"] = "not_shared_with_reference"
    later = contract.contract_answer(_parse(moved_again))
    assert after != later and contract._answer_drift(after, later)

    moved = copy.deepcopy(payload)
    _cell(moved, DENSE_DECODE)["evidence"]["artifact"] = None
    after = contract.contract_answer(_parse(moved))
    assert before != after and contract._answer_drift(before, after)
