"""A declared lane gate and an enforceable lane gate are the same object.

RobTand/prismaquant#119: *"the declared gates are named but never run"*.  The
narrow reading is that nothing spawns a container and runs them, which is
R16's open half and stays open.  The reading these tests pin is the structural
one, and it is principle 9: **a gate that names a check it does not perform is
a confession log, not a gate.**

Three links were missing on 2026-09-02, and each is a different failure:

1. ``lane_specs/tessera.json``'s ``route.census`` -- principle 12's second
   leg, the only place this lane compares the route it PRICED against the
   route it SERVED -- carried ``shipcard_slot: null``.  A gate with no slot is
   recorded nowhere, so nothing can refuse on it, and nothing distinguished it
   from a gate whose slot someone had forgotten to fill in.
2. The build lane never opened a ship record on this lane at all.
   ``export_native_compressed.py`` is the only thing in the tree that called
   ``build_shipcard``, and the Tessera arm calls Tessera's exporter and then
   ``exit 0``s ~130 lines above the driver's shipcard block.  So every gate
   the lane declares was enforced by nothing;
   ``tools/publish_artifact.py`` refused such an artifact only for ABSENCE of
   a card, which an operator dissolves by writing a base card by hand -- and a
   hand-written base card never carries the lane's own gates.
3. The two Tessera-repository scripts the arm shells out to were a hardcoded
   ``for`` loop in ``run-pipeline.sh`` and a sentence in the spec's ``notes``.

Fail-before on ``17b27bd``: ``LaneGate.from_dict`` accepted a null slot with no
reason; ``lane_gate_slots``, ``lane_shipcard`` and ``LaneProducerTool`` did not
exist; ``build_shipcard`` took no ``lane``; ``run-pipeline.sh``'s tessera arm
matched ``experiments/plan_from_layer_config.py``.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from prismaquant.lane_spec import (
    LaneGate,
    LaneProducerTool,
    LaneSpec,
    all_lane_specs,
    lane_gate_report,
    lane_spec_for_container,
    load_lane_spec,
)
from prismaquant.lane_shipcard import (
    LaneShipcardError,
    open_lane_shipcard,
)
from prismaquant.model_profiles.structure import EXPORT_LANES
from prismaquant.shipcard import (
    REQUIRED_SLOTS,
    lane_gate_slots,
    load_shipcard,
    required_slots,
    verify,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "prismaquant" / "run-pipeline.sh").read_text(encoding="utf-8")
LANE_IDS = list(EXPORT_LANES)


def _artifact(tmp_path, name="exported"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return model_dir


# ---------------------------------------------------------------------------
# 1 -- a gate with no slot must SAY it is advisory by construction
# ---------------------------------------------------------------------------
def test_a_gate_that_records_nothing_must_declare_why():
    with pytest.raises(ValueError, match="unrecorded_reason"):
        LaneGate.from_dict({
            "id": "route.census",
            "runner": "python tools/tessera_route_census.py",
            "shipcard_slot": None,
        })


def test_a_gate_cannot_be_both_recorded_and_unrecorded():
    with pytest.raises(ValueError, match="not unrecorded"):
        LaneGate.from_dict({
            "id": "gold.kl",
            "runner": "tools/measure_vllm_full_kl.py",
            "shipcard_slot": "gold.kl",
            "unrecorded_reason": "no it isn't",
        })


def test_an_unrecorded_gate_is_reported_as_unrecorded_not_as_unfilled():
    """The report must not let 'nobody ran it' read as 'nobody can record it'."""
    gate = LaneGate.from_dict({
        "id": "diagnostic",
        "runner": "echo hello",
        "shipcard_slot": None,
        "unrecorded_reason": "operator diagnostic; nothing to record",
    })
    spec = LaneSpec.from_dict({
        "schema": "prismaquant.lane_spec.v1",
        "id": "x", "export_container": "x", "runtime": "x",
        "wired_architectures": ["*"],
        "endpoint": {"kind": "none"},
        "kl_evaluator": {"kind": "validate_assignments_kl", "entrypoint": "x:y"},
        "gates": [json.loads(json.dumps({
            "id": gate.id, "runner": gate.runner, "shipcard_slot": None,
            "unrecorded_reason": gate.unrecorded_reason}))],
    })
    row = lane_gate_report(spec)[0]
    assert row["recorded"] is False
    assert row["filled"] is False
    assert row["unrecorded_reason"] == "operator diagnostic; nothing to record"


def test_route_census_is_now_a_recorded_gate_on_the_tessera_lane():
    """The specific gate #119 is about, and principle 12's second leg."""
    spec = load_lane_spec("tessera")
    census = spec.gate("route.census")
    assert census is not None
    assert census.recorded, (
        "route.census closes no shipcard slot, so the priced-vs-served route "
        "comparison is refused on by nothing")
    assert census.shipcard_slot == "route.census"


@pytest.mark.parametrize("lane", LANE_IDS)
def test_no_live_lane_declares_an_unexplained_gate(lane):
    for gate in lane_spec_for_container(lane).gates:
        assert gate.recorded or gate.unrecorded_reason


# ---------------------------------------------------------------------------
# 2 -- the roster fields the specs now own
# ---------------------------------------------------------------------------
def test_a_lane_must_declare_its_architecture_roster():
    payload = {
        "schema": "prismaquant.lane_spec.v1",
        "id": "x", "export_container": "x", "runtime": "x",
        "endpoint": {"kind": "none"},
        "kl_evaluator": {"kind": "validate_assignments_kl", "entrypoint": "x:y"},
    }
    with pytest.raises(ValueError, match="wired_architectures"):
        LaneSpec.from_dict(payload)
    with pytest.raises(ValueError, match="is empty"):
        LaneSpec.from_dict({**payload, "wired_architectures": []})


@pytest.mark.parametrize("lane", LANE_IDS)
def test_every_lane_spec_carries_a_non_empty_roster(lane):
    assert lane_spec_for_container(lane).wired_architectures


def test_a_producer_tool_without_a_stability_promise_must_name_an_issue():
    with pytest.raises(ValueError, match="tracking_issue"):
        LaneProducerTool.from_dict({
            "repo_env": "TESSERA_REPO",
            "path": "experiments/whatever.py",
            "stability": "unsupported_experiments",
        })


def test_a_misspelt_stability_does_not_read_as_supported():
    with pytest.raises(ValueError, match="stability"):
        LaneProducerTool.from_dict({
            "repo_env": "TESSERA_REPO",
            "path": "experiments/whatever.py",
            "stability": "supproted",
        })


def test_the_tessera_arms_experiments_scripts_are_declared_not_hardcoded():
    """#119 part 2. The dependency is recorded where a reader sees it."""
    spec = load_lane_spec("tessera")
    declared = {t.path for t in spec.producer_tools}
    assert declared == {
        "experiments/plan_from_layer_config.py",
        "experiments/export_tessera_serving.py",
        # The producer's expert projection (PrismaQuant #183): declared here
        # for the same reason as the other two.
        "experiments/tessera_producer_plan.py",
    }
    for tool in spec.producer_tools:
        assert tool.repo_env == "TESSERA_REPO"
        assert tool.stability == "unsupported_experiments"
        assert "119" in tool.tracking_issue or "183" in tool.tracking_issue
    # and the driver no longer carries its own copy of the roster
    assert "experiments/plan_from_layer_config.py" in DRIVER, (
        "the arm still CALLS the tool")
    assert "_tessera_tool" not in DRIVER, (
        "the existence check moved into the preflight, which reads the "
        "lane spec's producer_tools")


def test_the_preflight_refuses_a_declared_tool_that_is_not_there(tmp_path):
    from prismaquant.tessera_export_lane import (
        TesseraExportLaneError,
        require_producer_tools,
    )

    with pytest.raises(TesseraExportLaneError, match="TESSERA_REPO is unset"):
        require_producer_tools(env={})
    with pytest.raises(TesseraExportLaneError, match="does not exist"):
        require_producer_tools(env={"TESSERA_REPO": str(tmp_path)})
    for tool in load_lane_spec("tessera").producer_tools:
        path = tmp_path / tool.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    assert len(require_producer_tools(env={"TESSERA_REPO": str(tmp_path)})) == 3


# ---------------------------------------------------------------------------
# 3 -- the slot set a card opens is the lane's own gate set
# ---------------------------------------------------------------------------
def test_lane_gate_slots_reads_the_lane_declaration():
    assert "route.census" in lane_gate_slots("tessera")
    assert "route.census" not in lane_gate_slots("gguf")
    # A card with no lane keeps exactly its historical requirement.
    assert lane_gate_slots(None) == ()
    assert lane_gate_slots("no_such_lane") == ()


def test_a_lane_declaring_a_slot_the_shipcard_has_no_name_for_is_REFUSED(
    monkeypatch,
):
    """Not filtered away. #162 closed the roster but kept the refusal.

    `lane_gate_slots` admits a declared slot that is in the base set or names
    a verifier in `shipcard.LANE_SLOT_VERIFIERS`. Dropping any other slot
    would mean the lane declared a gate, the card never opened it, and
    `verify` passed an artifact that closed nothing -- the same silence one
    link earlier that #119 reported. So it raises, and the refusal names the
    lane and the slot.
    """
    import dataclasses

    from prismaquant import lane_spec as lane_spec_mod

    real = lane_spec_for_container("tessera")
    bogus = dataclasses.replace(
        real,
        gates=real.gates + (
            LaneGate(id="route.entropy", runner="true",
                     shipcard_slot="route.entropy"),
        ),
    )
    monkeypatch.setattr(
        lane_spec_mod, "lane_spec_for_container", lambda _lane: bogus)
    with pytest.raises(KeyError, match="route.entropy"):
        lane_gate_slots("tessera")


def test_a_lane_can_add_a_requirement_and_can_never_subtract_one():
    """UNION, not replacement.

    The GGUF lane declares no `native_export.graph` gate. If lane-derivation
    REPLACED the base set, declaring fewer gates would quietly lower the bar --
    the exact move principle 1 forbids. Pinned so a future refactor cannot
    turn a declaration into an exemption.
    """
    gguf_card = {"lane": "gguf", "slots": {}}
    assert set(REQUIRED_SLOTS).issubset(set(required_slots(gguf_card)))
    assert "native_export.graph" in required_slots(gguf_card)
    assert "native_export.graph" not in lane_gate_slots("gguf")

    tessera_card = {"lane": "tessera", "slots": {}}
    assert set(required_slots(tessera_card)) == set(REQUIRED_SLOTS) | {
        "route.census"}

    legacy_card = {"slots": {}}
    assert required_slots(legacy_card) == tuple(REQUIRED_SLOTS)


# ---------------------------------------------------------------------------
# 4 -- the build lane opens the record, and the record refuses
# ---------------------------------------------------------------------------
def test_opening_a_tessera_record_opens_the_gates_the_lane_declares(tmp_path):
    model_dir = _artifact(tmp_path)
    path = open_lane_shipcard(model_dir, "tessera")
    card = load_shipcard(path)
    assert card["lane"] == "tessera"
    assert set(card["slots"]) == set(REQUIRED_SLOTS) | {
        "route.census", "uniform_control"}
    assert all(v is None for v in card["slots"].values())


def test_a_tessera_card_owes_its_lanes_gates_AND_the_rate_axis_control(
    tmp_path,
):
    """Two mechanisms, one card, and neither knows about the other.

    `route.census` is opened because the LANE declares the gate
    (`lane_gate_slots`); `uniform_control` is opened because the ARTIFACT has
    a rate axis (`_is_rate_axis_artifact`, #121). They compose by union in
    `required_slots`, which is the property worth pinning: a card that owed
    only one of them would be a card one of the two closures does not cover,
    and nothing else in the tree exercises both on one card.
    """
    model_dir = _artifact(tmp_path)
    card = load_shipcard(open_lane_shipcard(model_dir, "tessera"))
    required = set(required_slots(card, model_dir=model_dir))
    assert "route.census" in required
    assert "uniform_control" in required
    assert set(REQUIRED_SLOTS).issubset(required)


def test_the_rate_axis_obligation_survives_a_missing_artifact_config(tmp_path):
    """The card must carry it too, not only the checkpoint.

    `_is_rate_axis_artifact` ORs the card's build block with the artifact's
    own `config.json` because an obligation a single erasure removes is not
    an obligation. `open_lane_shipcard` therefore stamps `export_container`
    into the build block: without it a Tessera card would owe the control on
    the strength of a file outside the card.
    """
    model_dir = _artifact(tmp_path)
    card = load_shipcard(open_lane_shipcard(model_dir, "tessera"))
    assert card["build"]["export_container"] == "tessera"
    # No model_dir at all -- the CLI's `verify` defaults to none.
    assert "uniform_control" in required_slots(card)


def test_an_unclosed_tessera_record_refuses_on_the_route_census(tmp_path):
    """The whole chain, end to end: declared -> opened -> refused on."""
    model_dir = _artifact(tmp_path)
    card = load_shipcard(open_lane_shipcard(model_dir, "tessera"))
    problems = verify(card, model_dir=model_dir)
    assert "route.census: UNFILLED" in problems
    # ... and it is refused for a REASON the lane declares, not because the
    # card is missing: publish_artifact's absence refusal is dissolved by
    # writing any card, and this one is not.
    assert len(problems) >= len(REQUIRED_SLOTS) + 1


def test_reopening_a_record_does_not_silently_discard_filled_slots(tmp_path):
    model_dir = _artifact(tmp_path)
    open_lane_shipcard(model_dir, "tessera")
    with pytest.raises(LaneShipcardError, match="already exists"):
        open_lane_shipcard(model_dir, "tessera")
    open_lane_shipcard(model_dir, "tessera", overwrite=True)


def test_a_record_cannot_be_opened_for_a_lane_outside_the_vocabulary(tmp_path):
    model_dir = _artifact(tmp_path)
    with pytest.raises(ValueError, match="unknown export lane"):
        open_lane_shipcard(model_dir, "nvfp4_cb")


@pytest.mark.parametrize("lane", LANE_IDS)
def test_every_lane_can_open_a_record(tmp_path, lane):
    """A lane in the vocabulary whose card cannot be opened cannot ship."""
    model_dir = _artifact(tmp_path, name=f"exported-{lane}")
    card = load_shipcard(open_lane_shipcard(model_dir, lane))
    assert card["lane"] == lane
    assert set(REQUIRED_SLOTS).issubset(set(card["slots"]))


def test_the_tessera_arm_opens_the_record_it_says_it_opens():
    """The driver's own banner claimed R13 held on this lane; it did not."""
    assert "prismaquant.lane_shipcard open" in DRIVER
    arm = DRIVER.split('if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then')[-1]
    arm = arm.split("\nfi\n")[0]
    assert "--lane tessera" in arm
    assert "--artifact" in arm


def test_the_arm_keeps_an_existing_record_instead_of_dying_on_the_refusal():
    """Skip-if-exists, like every other stage in this driver.

    `open_lane_shipcard` refuses an existing card without `--overwrite`,
    because re-opening one discards every slot the serve lane has filled. The
    driver's stages are skip-if-exists, so a re-run over a completed build
    reaches this step with a card already there: the refusal is correct and
    dying on it is not -- and the first spelling's message ("unpublishable
    until one exists") was false in exactly that case, one existed.
    """
    arm = DRIVER.split('if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then')[-1]
    arm = arm.split("\nfi\n")[0]
    assert '[[ -f "${WORK_DIR}/exported/shipcard.json" ]]' in arm
    # Every other refusal in this arm is `exit 2`; this one was `exit 1`.
    open_step = arm.split("prismaquant.lane_shipcard open")[-1]
    assert "exit 2" in open_step.split("\n  echo")[0]


def test_the_gate_set_of_every_lane_is_reachable_from_its_spec_alone():
    """No branch-per-lane anywhere: a fourth lane declares and is covered."""
    for spec in all_lane_specs():
        declared = set(spec.shipcard_slots())
        assert declared == set(lane_gate_slots(spec.export_container)), (
            f"{spec.id}: declared slots {sorted(declared)} are not all known "
            "to the shipcard")


def test_a_card_names_the_export_container_not_the_spec_filename():
    """One spelling on the card, and it is the one the operator sets.

    `lane_specs/compressed_tensors.json` is named with an underscore and its
    `export_container` is hyphenated; a card that stamped the FILE name would
    make `EXPORT_CONTAINER=compressed-tensors` and a card's `lane` two
    different strings for one lane. Found by the per-lane property in this
    file, which is the point of quantifying over the vocabulary rather than
    asserting the one lane under test.
    """
    for spec in all_lane_specs():
        assert lane_gate_slots(spec.export_container) == lane_gate_slots(
            spec.id), f"{spec.id}: the two spellings resolve differently"


@pytest.mark.parametrize("lane", LANE_IDS)
def test_the_driver_exports_the_env_var_each_declaration_names(lane):
    """A gate that reads an env var needs the driver to EXPORT it.

    `TESSERA_REPO` was set with `: "${TESSERA_REPO:=...}"` and never exported,
    which was invisible while the existence check was a bash loop in the same
    shell and would have made the preflight refuse every run the moment it
    moved into Python. The property is per declaration, so a fourth lane
    naming a different repo env var is covered.
    """
    for tool in lane_spec_for_container(lane).producer_tools:
        assert f"export {tool.repo_env}" in DRIVER, (
            f"{lane}: {tool.repo_env} is read by the preflight but not "
            "exported by the driver")


# ---------------------------------------------------------------------------
# 5 -- #162: the slot vocabulary is derived, and every derived slot replays
# ---------------------------------------------------------------------------
def test_every_derived_lane_slot_names_a_verifier_verify_replays():
    """A slot admitted by derivation alone gets the generic checks and no replay.

    That is the "written into the receipt and replayed by nothing" shape
    (#156 / #157), reached by widening the vocabulary rather than by a
    half-written verifier. So the vocabulary derivation and the verifier
    registry are one invariant: every lane-declared slot outside the base set
    must name the verifier `verify` replays for it. Quantified over
    `all_lane_specs()`, so a fourth lane's novel slot is covered without a
    test edit.
    """
    import prismaquant.shipcard as shipcard_mod

    base = set(REQUIRED_SLOTS) | set(shipcard_mod.OPTIONAL_SLOTS)
    derived = sorted({
        slot
        for spec in all_lane_specs()
        for slot in spec.shipcard_slots()
        if slot not in base
    })
    assert derived, (
        "no lane declares a slot beyond the base set; the property is vacuous")
    verifiers = getattr(shipcard_mod, "LANE_SLOT_VERIFIERS", {})
    missing = [slot for slot in derived if slot not in verifiers]
    assert not missing, (
        f"lane-declared slot(s) {missing} name no verifier: a record with "
        "passed=True would close them on the generic checks alone. Register "
        "the verifier beside the slot (shipcard.LANE_SLOT_VERIFIERS)")
    for slot in derived:
        assert callable(verifiers[slot]), f"{slot}: verifier is not callable"


def test_verify_replays_route_census_evidence(tmp_path, monkeypatch):
    """`route.census` refuses silence, and refuses a wrong census.

    The replay is the #136 priced-vs-served comparison, dispatched through
    `shipcard.LANE_SLOT_VERIFIERS`: a bare `passed=True` record with no
    carried census is refused, a receipt whose records replay to agreement
    closes the slot, and a `passed=true` over substitute-decoder records is
    refused at publication.

    Flat rows are the historical receipt, attestable only where no current
    scoped table exists to refuse them -- the installed runtime is stood
    down for this test (`tests/test_tessera_route_receipt.py` covers the
    refusal the pinned table gives the same rows).
    """
    import prismaquant.tessera_route_receipt as receipt
    from prismaquant.shipcard import make_record, make_route_census_record

    def _absent():
        raise ModuleNotFoundError("No module named 'tessera'", name="tessera")

    monkeypatch.setattr(receipt, "_current_scoped_contract", _absent)
    model_dir = _artifact(tmp_path)
    card = load_shipcard(open_lane_shipcard(model_dir, "tessera"))
    sha = card["model_sha"]
    priced = ["TESSERA_NVFP4"]
    substitutes = ["torch_materialize_stock"]

    bare = make_record(
        slot="route.census", tool="tessera_route_census",
        passed=True, model_sha=sha, detail="routes match")
    card["slots"]["route.census"] = bare
    problems = verify(card, model_dir=model_dir, required=["route.census"])
    assert any("route.census" in problem for problem in problems), (
        "verify passed a route.census record that carries no census "
        f"evidence: problems={problems!r}")

    good = make_route_census_record(
        tool="tessera_route_census", model_sha=sha,
        priced_routes=priced,
        route_records=[{
            "route": "TESSERA_NVFP4", "decoder": "tessera_nvfp4", "count": 12,
        }],
        substitute_decoders=substitutes)
    assert good["passed"] is True
    card["slots"]["route.census"] = good
    problems = verify(card, model_dir=model_dir, required=["route.census"])
    assert not [p for p in problems if "route.census" in p], problems

    lying = make_route_census_record(
        tool="tessera_route_census", model_sha=sha,
        priced_routes=priced,
        route_records=[{
            "route": "TESSERA_NVFP4", "decoder": "torch_materialize_stock",
            "count": 12,
        }],
        substitute_decoders=substitutes)
    assert lying["passed"] is False
    lying = dict(lying)
    lying["passed"] = True
    card["slots"]["route.census"] = lying
    problems = verify(card, model_dir=model_dir, required=["route.census"])
    assert any("route.census" in problem for problem in problems), (
        "verify passed a route.census record whose verdict disagrees with "
        f"its own records: problems={problems!r}")
    census_problems = [p for p in problems if "route.census" in p]
    assert len(census_problems) == len(set(census_problems)), (
        "route.census replayed twice -- the registry dispatch overlaps the "
        f"explicit branch: problems={problems!r}")


def test_a_fourth_lane_slot_with_a_verifier_needs_no_roster_edit(monkeypatch):
    """Adding a lane with a novel gate is one spec plus one verifier.

    A fourth lane declares `route.entropy` closing a slot no live lane names.
    With a verifier registered beside it, the lane's gates open, `make_record`
    accepts the slot, and `verify` replays the evidence -- with no edit to any
    roster in `shipcard.py`. Without the verifier the declaration is still
    refused (the existing `..._is_REFUSED` test pins that half).
    """
    import dataclasses

    import prismaquant.shipcard as shipcard_mod
    from prismaquant import lane_spec as lane_spec_mod

    verifiers = getattr(shipcard_mod, "LANE_SLOT_VERIFIERS", None)
    assert verifiers is not None, (
        "shipcard has no verifier registry: a fourth lane's novel slot still "
        "needs a roster edit (RobTand/prismaquant#162)")

    def _verify_entropy(slot, record, *, card=None, model_dir=None):
        block = record.get("route_entropy")
        if not isinstance(block, Mapping):
            return [f"{slot}: carries no route_entropy block, "
                    "so there is nothing to replay"]
        if block.get("match") is not True:
            return [f"{slot}: route_entropy match is "
                    f"{block.get('match')!r}, not True"]
        return []

    monkeypatch.setitem(verifiers, "route.entropy", _verify_entropy)
    real = lane_spec_for_container("tessera")
    bogus = dataclasses.replace(
        real,
        gates=real.gates + (
            LaneGate(id="route.entropy", runner="true",
                     shipcard_slot="route.entropy"),
        ),
    )
    monkeypatch.setattr(
        lane_spec_mod, "lane_spec_for_container", lambda _lane: bogus)

    assert "route.entropy" in lane_gate_slots("tessera")
    assert "route.entropy" in shipcard_mod.required_slots(
        {"lane": "tessera", "slots": {}})

    sha = "abc"
    passing = shipcard_mod.make_record(
        slot="route.entropy", tool="true", passed=True, model_sha=sha,
        extra={"route_entropy": {"match": True}})
    problems = shipcard_mod.verify(
        {"model_sha": sha, "slots": {"route.entropy": passing}},
        required=["route.entropy"])
    assert not [p for p in problems if "route.entropy" in p], problems

    failing = dict(passing)
    failing["route_entropy"] = {"match": False}
    problems = shipcard_mod.verify(
        {"model_sha": sha, "slots": {"route.entropy": failing}},
        required=["route.entropy"])
    assert any("route.entropy" in p for p in problems), (
        "verify passed a route.entropy record whose evidence refuses: "
        f"problems={problems!r}")
