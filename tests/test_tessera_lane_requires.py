"""The lane PREDICATE: ``native_extensions[].lane.requires``, consumed.

Tessera's contract v20 (its #264) publishes, on every ``native_extensions[]``
row, a ``lane`` block: the ``decoder`` the extension serves and -- for the
window-GEMV kernel -- ``requires``, the predicate a unit's WIRE must satisfy
for that kernel to read it: column rates, window bits, body, plane, no release
overrides, no diagonals, no rotation, no start state, a scalar grid.  The
loader refuses a unit that fails it.  A producer that SELECTED such a unit
would ship an artifact whose serve substitutes another decoder or refuses,
and it would find out at serve time -- the defect ``lane_eligibility`` exists
to stop, one field over.

One rule, one home.  The DECISION is Tessera's
(``tessera.serving.scheme.decide_lane_requirements``) and is called, never
restated: nothing in this repository spells "rates must be in [1, 2, 4]".
PrismaQuant's half is the FACTS -- ``tessera_render.planned_wire_facts``,
the wire this producer's encoder is going to write for a family at a rung,
read off the same recipe and the same decoration constants
``render_tessera_weight`` encodes with -- and the READER, which parses the
predicate closed at Tessera's own vocabulary and refuses by name whatever it
does not understand.  ``lane_eligibility.cell_lane_admits`` is the gate, and
every admission leg reads it: the menu (``tessera_attesting_cells``), the
development contract (``TesseraContract.native_cells``) and the per-unit
export gate (``resolve_unit_route``).

Consequence on the pinned table, stated because it is what a reviewer needs:
the cells that execute the ``window_gemv`` lane are the two E4M3 STREAMED
cells, this producer plans rate (4,), window 14, WINDOW/CHANNEL, undecorated,
arity 1 for them, and the predicate passes.  Nothing is refused today.  The
BF16 rung the lane's ``routes`` also names would NOT pass -- rung 1792 plans
rate 7 -- and no BF16 cell claims that launch; the test below pins that a
cell which did would be refused by name at all three legs.
"""
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render
from prismaquant import tessera_runtime_contract as contract


WINDOW_LANE = "tessera_window_gemv"
NVFP4_LANE = "tessera_nvfp4_"
E4M3 = "TESSERA_E4M3_K1"
E4M3_NAME = "TESSERA_E4M3_K1_R1024"
E4M3_RATE = 1024
BF16 = "TESSERA_BF16_K1"
BF16_NAME = "TESSERA_BF16_K1_R1792"
BF16_RATE = 1792
BF16_DECODE = "tessera_bf16_k1_dense_sm121_decode"
STREAMED = (
    "tessera_e4m3_k1_dense_sm121_decode_streamed",
    "tessera_e4m3_k1_dense_sm121_batch_streamed",
)
WINDOW_LAUNCH = {"symbol": "tessera_window_gemv::gemv", "decoder": "window_gemv"}


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


def _formats(payload):
    return {row["family"]: row for row in payload["formats"]}


def _row(payload, prefix):
    for row in payload["native_extensions"]:
        if row["module_name_prefix"] == prefix:
            return row
    raise AssertionError(f"the installed contract publishes no extension {prefix!r}")


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


def _context(payload, cell_id, residency):
    cell = _cell(payload, cell_id)
    return lane.ServingContext(
        platform=cell["platform"], structure=cell["structure"], residency=residency,
        runtime_image=cell["runtime"]["image"],
        execution_mode=cell["runtime"]["execution_modes"][0])


def _canonical(requires):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in requires.items()}


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
def test_the_installed_predicate_is_read_closed_at_tesseras_vocabulary(table, payload):
    claims = {claim.extension: claim for claim in table.lanes}
    assert set(claims) == {WINDOW_LANE, NVFP4_LANE}
    window = claims[WINDOW_LANE]
    assert window.decoder == _row(payload, WINDOW_LANE)["lane"]["decoder"]
    assert window.requires == _canonical(_row(payload, WINDOW_LANE)["lane"]["requires"])
    assert claims[NVFP4_LANE].requires is None, (
        "a lane that publishes no predicate is the route's own eligibility, "
        "not an empty predicate")
    # The vocabulary this reader closes is exactly what the pinned lane
    # publishes today: a requirement the lane grows is refused by name below,
    # never skipped.
    assert set(lane.LANE_REQUIREMENT_FIELDS) == set(window.requires)
    assert table.provenance()["lanes"] == [
        claim.answer() | {"extension": claim.extension} for claim in table.lanes]


def test_a_table_is_not_readable_without_the_extension_table(payload):
    block, formats = payload["lane_eligibility"], payload["formats"]
    with pytest.raises(TypeError):
        lane._parse_table(block, formats, "", "", "x")
    with pytest.raises(lane.LaneEligibilityError, match="native_extensions"):
        lane._parse_table(block, formats, "", "", "x", native_extensions=None)


def _mutate(payload, **changes):
    moved = copy.deepcopy(payload)
    block = _row(moved, WINDOW_LANE)["lane"]
    for key, value in changes.items():
        if key == "lane":
            _row(moved, WINDOW_LANE)["lane"] = value
        elif key.startswith("requires."):
            block["requires"][key.split(".", 1)[1]] = value
        elif key == "requires":
            block["requires"] = value
        else:
            block[key] = value
    return moved


@pytest.mark.parametrize("changes, names", [
    ({"requires.span": [2]}, ("span", "cannot decide")),
    ({"requires": {}}, ("requires", "empty")),
    ({"requires": [1, 2, 4]}, ("requires", "object")),
    ({"requires.column_rates": [4, 2, 1]}, ("column_rates", "ascending")),
    ({"requires.column_rates": [1, 1, 2]}, ("column_rates", "ascending")),
    ({"requires.window_bits": [0]}, ("window_bits", "positive")),
    ({"requires.grid_arities": []}, ("grid_arities", "non-empty")),
    ({"requires.release_overrides": "false"}, ("release_overrides", "JSON boolean")),
    ({"requires.rotation": ["two_sided"]}, ("rotation", "two_sided")),
    ({"requires.rotation": []}, ("rotation", "non-empty")),
    ({"requires.body": "trellis"}, ("body", "trellis")),
    ({"requires.plane": "lut"}, ("plane", "lut")),
    ({"kernel": "gemv"}, ("unknown field", "kernel")),
    ({"lane": {"requires": {"column_rates": [4]}}}, ("missing field", "decoder")),
    ({"lane": "window_gemv"}, ("lane", "object")),
])
def test_a_predicate_this_reader_does_not_understand_is_refused_by_name(
        payload, changes, names):
    moved = _mutate(payload, **changes)
    with pytest.raises(lane.LaneEligibilityError) as refused:
        _table(moved)
    for name in names:
        assert name in str(refused.value), (name, str(refused.value))
    assert WINDOW_LANE in str(refused.value)


def test_the_development_contract_reads_the_same_grammar(payload):
    moved = _mutate(payload, **{"requires.span": [2]})
    with pytest.raises(contract.TesseraContractError, match="span"):
        contract._parse(moved, commit="fixture", sha="fixture", path="fixture")


# ---------------------------------------------------------------------------
# The facts: what this producer plans is what its encoder writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, family, rate", [
    (E4M3_NAME, E4M3, E4M3_RATE),
    (BF16_NAME, BF16, BF16_RATE),
])
def test_planned_wire_facts_are_the_facts_the_encoded_bytes_carry(name, family, rate):
    """Principle 8: the gate decides on the plan, the export writes the bytes,
    and the two must be one object.  Read the facts off a unit encoded by the
    render path itself, in Tessera's OWN byte-side vocabulary."""
    from tessera.decode import reconstruct_unit
    from tessera.serving.scheme import wire_facts_of_parsed

    torch.manual_seed(0)
    weight = torch.randn(4, 512, dtype=torch.bfloat16)
    unit, forests = render._encode_planned_unit(weight, name)
    spec, _rung = render.parse_tessera_format_name(name)
    on_bytes = wire_facts_of_parsed(SimpleNamespace(unit=unit, grid=render._grid_for(spec)))
    planned = render.planned_wire_facts(family, rate)
    assert set(planned) == set(on_bytes)
    assert set(on_bytes["rates"]) == set(planned["rates"])
    for key in set(planned) - {"rates"}:
        assert planned[key] == on_bytes[key], key
    assert None not in planned.values()
    # And the render IS that unit's reconstruction, not a second encode.
    rendered = render.render_tessera_weight(weight, name)
    assert torch.equal(
        rendered, reconstruct_unit(unit, forests, render._tessera_export.DEFAULT_CODE)
        .to(dtype=weight.dtype))


def test_the_planned_facts_speak_the_decision_cores_whole_vocabulary(payload):
    """Every published requirement is decided on a fact this producer
    supplies: an 'was not read' refusal would mean the plan is silent on a
    condition the loader enforces."""
    from tessera.serving.scheme import decide_lane_requirements

    requires = _row(payload, WINDOW_LANE)["lane"]["requires"]
    assert decide_lane_requirements(
        WINDOW_LANE, requires, render.planned_wire_facts(E4M3, E4M3_RATE)) == []
    refusals = decide_lane_requirements(
        WINDOW_LANE, requires, render.planned_wire_facts(BF16, BF16_RATE))
    assert refusals and all("was not read" not in line for line in refusals)
    assert all(line.startswith("column_rates") for line in refusals), refusals


def test_the_planned_decoration_is_the_render_decoration():
    """The constants the facts are read from are the constants the encoder
    is called with -- one home, by name."""
    planned = render.planned_wire_facts(E4M3, E4M3_RATE)
    assert planned["rotation"] == render.TESSERA_PLANNED_ROTATION.name
    assert planned["diagonals"] is render.TESSERA_PLANNED_DIAGONALS
    assert planned["release_overrides"] == render.TESSERA_PLANNED_RELEASE_OVERRIDES
    assert planned["start_state"] is False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_every_lane_gated_cell_on_the_pinned_table_admits_this_producers_plan(table):
    gated = {cell.id for cell in table.cells
             if lane.lane_claim_for_cell(cell, table.lanes) is not None}
    assert gated == set(STREAMED), (
        "the cells subject to the window-GEMV predicate are exactly the ones "
        "that execute it; a decoder no lane names is the route's own path")
    for cell in table.cells:
        for rung in cell.rungs_q256:
            admits, why = lane.cell_lane_admits(cell, rung, table.lanes)
            assert admits, (cell.id, why)
            assert why == ""


def _claiming_bf16(payload):
    moved = copy.deepcopy(payload)
    _cell(moved, BF16_DECODE)["executes"] = [WINDOW_LAUNCH]
    return moved


def test_a_cell_claiming_the_lane_for_a_rung_it_refuses_is_refused_by_name(payload):
    moved = _claiming_bf16(payload)
    table = _table(moved)
    cell = _parsed_cell(table, BF16_DECODE)
    admits, why = lane.cell_lane_admits(cell, BF16_RATE, table.lanes)
    assert not admits
    for name in (BF16_DECODE, WINDOW_LANE, "window_gemv", "column_rates", "[7]"):
        assert name in why, (name, why)
    # Sibling cell on the same family is untouched: refusal is per launch.
    batch = _parsed_cell(table, "tessera_bf16_k1_dense_sm121_batch")
    assert lane.cell_lane_admits(batch, BF16_RATE, table.lanes) == (True, "")


def test_all_three_admission_legs_read_the_one_gate(payload, monkeypatch):
    moved = _claiming_bf16(payload)
    table, formats = _table(moved), _formats(moved)
    context = _context(moved, BF16_DECODE, "resident")

    # 1. The menu.
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    assert render.tessera_attesting_cells(BF16_NAME, serving_context=context) == ()
    admitted, reason = render.tessera_lane_admission(BF16_NAME, serving_context=context)
    assert not admitted
    assert "column_rates" in reason and BF16_DECODE in reason and WINDOW_LANE in reason
    assert not render.tessera_lane_attested(BF16_NAME, serving_context=context)

    # 2. The per-unit export gate names the cell AND the reason.
    facts = lane.UnitStructuralFacts(
        qname="fixture.weight", format_name=BF16_NAME, payload_family=BF16,
        k=None, n_sub=None, rate_q256=BF16_RATE, structure="dense",
        role_split=False, in_features=1024, out_features=1024)
    route = lane.resolve_unit_route(
        facts, table, platform=context.platform, residency="resident",
        runtime_image=context.runtime_image, execution_mode=context.execution_mode)
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED
    decode = {r.regime: r for r in route.regimes}["decode"]
    assert decode.cell_id == BF16_DECODE
    assert "column_rates" in decode.detail

    # 3. The development contract.
    parsed = contract._parse(moved, commit="fixture", sha="fixture", path="fixture")
    assert parsed.native_cells(BF16, BF16_RATE, serving_context=context) == ()


def test_the_untouched_table_admits_the_same_bf16_rung_at_every_leg(payload, monkeypatch):
    """The control for the test above: the refusal is the launch claim, not
    the family or the rung."""
    table, formats = _table(payload), _formats(payload)
    context = _context(payload, BF16_DECODE, "resident")
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    assert render.tessera_lane_admission(BF16_NAME, serving_context=context) == (True, "")
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    assert parsed.native_cells(BF16, BF16_RATE, serving_context=context)


def test_a_decorated_plan_is_refused_at_every_requirement_it_breaks(table, monkeypatch):
    """The predicate is the whole predicate: the loader reads all nine, so
    the gate cannot decide fewer."""
    decorated = dict(render.planned_wire_facts(E4M3, E4M3_RATE))
    decorated.update(rates=(5,), window_bits=12, release_overrides=3, diagonals=True,
                     rotation="R_IN_ONLY", start_state=True, grid_arity=2)
    monkeypatch.setattr(render, "planned_wire_facts", lambda family, rung: decorated)
    cell = _parsed_cell(table, STREAMED[0])
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, table.lanes)
    assert not admits
    for name in ("column_rates", "window_bits", "release_overrides", "diagonals",
                 "rotation", "start_state", "grid_arities"):
        assert name in why, (name, why)


def test_a_requirement_the_decision_core_cannot_decide_refuses_rather_than_skips(table):
    claim = dataclasses.replace(
        [c for c in table.lanes if c.extension == WINDOW_LANE][0],
        requires={"span": (2,)})
    cell = _parsed_cell(table, STREAMED[0])
    with pytest.raises(lane.LaneEligibilityError, match="span"):
        lane.cell_lane_admits(cell, E4M3_RATE, (claim,))


def test_a_family_this_producer_cannot_plan_is_refused_not_passed(table):
    cell = dataclasses.replace(_parsed_cell(table, STREAMED[0]), family="TESSERA_E5M2_K1")
    admits, why = lane.cell_lane_admits(cell, E4M3_RATE, table.lanes)
    assert not admits
    assert "TESSERA_E5M2_K1" in why and "plan" in why


def test_a_lane_gated_cell_without_a_rung_is_refused(table):
    cell = _parsed_cell(table, STREAMED[0])
    admits, why = lane.cell_lane_admits(cell, None, table.lanes)
    assert not admits and "rung" in why


# ---------------------------------------------------------------------------
# The reviewed answer
# ---------------------------------------------------------------------------
def test_the_lane_predicate_is_part_of_the_reviewed_answer(payload):
    """A widened predicate must re-stale the pin: it changes what this
    producer may select."""
    before = contract.contract_answer(
        contract._parse(payload, commit="fixture", sha="fixture", path="fixture"))
    rows = {row["module_name_prefix"]: row for row in before["native_extensions"]}
    assert rows[WINDOW_LANE]["lane"] == {
        "decoder": "window_gemv",
        "requires": _row(payload, WINDOW_LANE)["lane"]["requires"]}
    assert rows[NVFP4_LANE]["lane"] == {"decoder": "native_span2", "requires": None}
    moved = _mutate(payload, **{"requires.column_rates": [1, 2, 4, 8]})
    after = contract.contract_answer(
        contract._parse(moved, commit="fixture", sha="fixture", path="fixture"))
    assert before != after
    drift = contract._answer_drift(before, after)
    assert any(f"native_extensions[{WINDOW_LANE}].lane" in line for line in drift), drift
