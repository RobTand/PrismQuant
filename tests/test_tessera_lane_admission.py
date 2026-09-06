"""Tessera's producer-side admission, against Tessera's OWN packaged contract.

Since 2026-09-02 Tessera serves itself: package ``tessera.serving``, entry
point ``tessera = "tessera.serving:register"`` under
``vllm.general_plugins``, ``quant_method: "tessera"``, one operator knob
``TESSERA_SERVE_MODE``.  Gridbook's Tessera lane is withdrawn -- its contract
v14, which carried the two Tessera rows, was never released, so nothing that
shipped is broken by the move.

What this module pins is the *shape of the refusal*, because the values will
move and the shape must not:

* the packaged Tessera table parses through the SHARED eligibility parser
  (`lane_eligibility`), whose three widenings are additive;
* every Tessera cell carries ``requires_plugin: "tessera"``, and a cell that
  claims a route without one is refused rather than admitted;
* the answer is False today BY THE PIN, and True under a released-pin fixture
  on the REAL packaged contract -- which is what makes "False" a fact about the
  release boundary rather than about an absent table;
* the lane spec's ``served_activation_quantization.executes`` is DERIVED from
  that table, not typed, so principle 14 holds in the field that asserts what
  the runtime executes.
"""
from __future__ import annotations

import json

import pytest

from prismaquant import tessera_render as tr
from prismaquant.lane_eligibility import (
    EVIDENCE_SMOKE_REFUSALS,
    FORMAT_KIND_TESSERA_WIRE,
    LANE_ELIGIBILITY_SCHEMA_TESSERA,
    LaneEligibilityError,
    cell_evidence_admits,
    load_eligibility_table,
    load_published_formats,
)
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_PLUGIN_NAME,
    TESSERA_SERVING_RESIDENCY_ENV,
    TESSERA_SERVING_RUNTIME_COMMIT_PENDING,
    TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
    TESSERA_SERVING_RUNTIME_REPOSITORY,
    TESSERA_SERVING_RUNTIME_VERSION_PENDING,
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    parse_tessera_serving_runtime_pin,
    require_exact_tessera_runtime_pin,
    require_pinned_tessera_runtime,
    tessera_serving_runtime_pin_path,
)

#: A resolved-looking commit that is NOT any real Tessera commit. It exists to
#: prove the machinery, and the constants are monkeypatched to match it, so no
#: fixture here can ever be mistaken for an attestation.
FIXTURE_COMMIT = "0" * 39 + "1"
FIXTURE_VERSION = "0.1.0"


def _default_serve_image() -> str:
    """The serve image the packaged contract publishes as its default."""
    return _packaged_contract()["versions"]["default_serve_image"]


def _dense_context():
    """The scope the eight dense cells of the pinned (v8) table publish.

    The table is scoped: a cell attests a rung only at an exact platform,
    image, execution mode and residency, so an admission is asked at one --
    the operator's ``--tessera-*`` flags in production.  A context-free ask
    is refused by SCOPE, asserted apart from the pin's refusal below.
    """
    from prismaquant.lane_eligibility import ServingContext
    return ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=_default_serve_image(), execution_mode="eager")


def _packaged_contract() -> dict:
    from importlib.resources import as_file

    with as_file(tr.tessera_serving_contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _load(path):
    """The table and format rows a contract file publishes."""
    return (load_eligibility_table("fixture", contract_path=path),
            load_published_formats("fixture", contract_path=path))


def _write(tmp_path, contract, name="contract.json"):
    path = tmp_path / name
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


@pytest.fixture()
def released_pin(tmp_path, monkeypatch):
    """A RESOLVED Tessera pin, built through the real reader and the real gate.

    The fixture writes a pin file with a resolved commit/version/digest, points
    the three pinned constants at it, and substitutes the loader on the pin
    MODULE -- which is where ``tessera_render._release_pin_satisfied`` looks it
    up, at call time.  Nothing here weakens the parser or the gate: the digest
    is the INSTALLED contract's real one, so the enforced conjunct runs for
    real and only the two recorded-identity values are synthetic.
    """
    payload = {
        "schema": TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
        "repository": TESSERA_SERVING_RUNTIME_REPOSITORY,
        "commit": FIXTURE_COMMIT,
        "version": FIXTURE_VERSION,
        "version_is_release": True,
        "contract_sha256": pin_module.installed_tessera_contract_sha256(),
        "runtime_contract_schema": "tessera.runtime-contract.v1",
        "plugin_entry_point": "tessera = tessera.serving:register",
        "serving_residency_env": TESSERA_SERVING_RESIDENCY_ENV,
        "serving_native_extensions": [
            {"module_name_prefix": "tessera_nvfp4_",
             "filename_glob": "tessera_nvfp4_*.so",
             "match": "basename_fnmatch",
             # A synthetic released pin, not a transcription of the runtime:
             # the release gate under test reads commit/version, never this
             # block.  It carries the contracted shape so the fixture parses.
             "when_unavailable": {
                 "resident": {"status": "substituted",
                              "decoder": "torch_materialize_stock"},
                 "streamed": {"status": "refused", "decoder": None}}},
        ],
    }
    path = tmp_path / "released_pin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_COMMIT", FIXTURE_COMMIT)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_VERSION", FIXTURE_VERSION)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256",
        payload["contract_sha256"])
    pin = pin_module.load_tessera_serving_runtime_pin(path)
    monkeypatch.setattr(
        pin_module, "load_tessera_serving_runtime_pin", lambda *a, **k: pin)
    require_pinned_tessera_runtime(pin)   # the real gate, satisfied
    return pin


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------
def test_the_tracked_pin_is_an_exact_commit_and_the_installed_contracts_digest():
    """Admission answers TODAY BY THE PIN, not by an edit.

    Until 2026-09-04 this asserted the opposite -- the pin carried PENDING
    sentinels and refused everything, because there was no Tessera release
    tag. Rob retired the tag requirement, so what the pin now carries is an
    exact commit plus the SHA-256 of the contract that commit packages, and
    the gate compares the digest against the INSTALLED bytes. Same property,
    a fact this process can check instead of a tag nobody wants to cut.
    """
    pin = load_tessera_serving_runtime_pin()
    assert pin.commit_is_resolved and pin.version_is_resolved
    assert pin.contract_sha256_is_resolved
    assert pin.plugin_entry_point == "tessera = tessera.serving:register"

    require_pinned_tessera_runtime(pin)
    assert tr._release_pin_satisfied() is True

    with pytest.raises(TesseraServingRuntimePinError,
                       match="the installed Tessera is not the pinned Tessera"):
        require_exact_tessera_runtime_pin(
            pin, installed_contract_sha256="e" * 64)


def test_the_pin_file_on_disk_is_the_one_the_reader_parses():
    """No second spelling of the pin: the tracked JSON round-trips."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    assert parse_tessera_serving_runtime_pin(payload) == (
        load_tessera_serving_runtime_pin())


def test_a_pending_pin_cannot_be_marked_released():
    """The one structural rule that stops a half-edit admitting anything.

    ``version_is_release`` is advisory since pin schema v2 -- no gate reads it
    -- but the rule that keeps it TRUE when it is read is still enforced.
    """
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["commit"] = TESSERA_SERVING_RUNTIME_COMMIT_PENDING
    payload["version_is_release"] = True
    with pytest.raises(TesseraServingRuntimePinError, match="cannot be marked"):
        parse_tessera_serving_runtime_pin(payload)


def test_a_pin_that_is_not_the_reviewed_one_is_refused():
    """Editing the JSON alone admits nothing: the constants must move too,
    in the same reviewed commit."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload.update(commit="a" * 40, version="9.9.9", version_is_release=True)
    pin = parse_tessera_serving_runtime_pin(payload)
    with pytest.raises(TesseraServingRuntimePinError, match="ONE reviewed change"):
        require_exact_tessera_runtime_pin(
            pin, installed_contract_sha256=payload["contract_sha256"])


# ---------------------------------------------------------------------------
# The packaged table, through the shared parser
# ---------------------------------------------------------------------------
def test_the_packaged_tessera_contract_parses_and_every_cell_names_the_plugin():
    """The shared parser reads Tessera's table, and the plugin requirement is
    a parsed FIELD on every cell -- never prose a gate cannot read."""
    from importlib.resources import as_file

    with as_file(tr.tessera_serving_contract_path()) as path:
        table, formats = _load(path)
        published_schema = json.loads(path.read_text(encoding="utf-8"))[
            "lane_eligibility"]["schema"]

    assert table.present
    # Parser capability can advance before the reviewed development pin does.
    # This assertion belongs to the actual installed publisher, not the newest
    # schema the consumer knows how to read.
    assert table.schema == published_schema
    assert table.cells
    for cell in table.cells:
        assert cell.requires_plugin == TESSERA_SERVING_PLUGIN_NAME
        # rate-addressed, so the rung vocabulary is rungs_q256
        assert cell.is_trellis and cell.rungs_q256 and not cell.rungs
        assert cell.as_dict()["requires_plugin"] == TESSERA_SERVING_PLUGIN_NAME
    assert table.provenance()["required_plugins"] == [
        TESSERA_SERVING_PLUGIN_NAME]
    assert {e["kind"] for e in formats.values()} == {FORMAT_KIND_TESSERA_WIRE}
    assert table.trellis_families == frozenset(formats)
    # Since contract v17 the table DECLARES routed_moe. Declaring a structure
    # is not admitting it: v17 through v20 published, on those cells, a greedy
    # smoke that degenerated and the evidence gate refused them; v21 (Tessera
    # #313) re-measured it through the checkpoint's own chat template and
    # records "recorded", so the SAME gate stops refusing them. The honest
    # assertion is neither verdict -- what v21's "recorded" is worth is open
    # (RobTand/tessera#327, prismaquant #198) -- but that admission FOLLOWS the
    # cells' own published evidence, read from the table and from nothing else.
    assert table.structures == ("dense", "routed_moe")
    routed = [cell for cell in table.cells if cell.structure == "routed_moe"]
    assert routed
    for cell in routed:
        admitted, why = cell_evidence_admits(cell)
        assert admitted is (
            cell.evidence.smoke_status not in EVIDENCE_SMOKE_REFUSALS), (
                cell.id, cell.evidence.smoke_status, why)


def test_a_cell_claiming_a_route_with_no_plugin_requirement_is_refused(tmp_path):
    """A contract defect must be LOUD, not silently admitted.

    A Tessera cell that claims a native route while naming no plugin would let
    an artifact be admitted whose serve command need not install the runtime
    that reads its bytes. ``tessera_lane_attested`` raises on it rather than
    answering either True or False -- and it raises BEFORE the pin conjunct, so
    the defect surfaces today rather than on the day a tag is cut.
    """
    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["requires_plugin"] = ""
    with pytest.raises(LaneEligibilityError, match="requires_plugin"):
        table, formats = _load(_write(tmp_path, contract, "no_plugin.json"))
        tr.tessera_lane_attested(
            "TESSERA_E2M1_K2_R896", table=table, formats=formats)


def test_a_plugin_requirement_on_a_fallback_cell_is_refused(tmp_path):
    """The validity rule mirrors ``requires_serve_flags``: a plugin
    requirement is an instruction for reaching a route that EXISTS. Naming one
    on an announced fallback would read as "install this and it goes native"."""
    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell["route_status"] = "fallback"
        cell["requires_serve_flags"] = []
    with pytest.raises(LaneEligibilityError, match="requires_plugin"):
        _load(_write(tmp_path, contract, "fallback_plugin.json"))


# ---------------------------------------------------------------------------
# The lookup
# ---------------------------------------------------------------------------
def test_admission_is_true_under_a_released_pin_on_the_real_packaged_contract(
        released_pin):
    """The other half of "False by the pin".

    With ONLY the release boundary satisfied -- the real packaged contract,
    the real parser, the real cell logic -- both receipted rungs are admitted
    and a rate the contract does not publish is not. That is what proves the
    refusal above is the pin's and not an artefact of an unreadable table.
    """
    assert tr._release_pin_satisfied() is True
    context = _dense_context()
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R896", serving_context=context) is True
    assert tr.tessera_lane_attested(
        "TESSERA_E4M3_K1_R1024", serving_context=context) is True
    # a serialisable rate no cell names, on a published family
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R512", serving_context=context) is False
    # the family's own terminal rate, which the reader range excludes
    assert tr.tessera_lane_attested(
        "TESSERA_E4M3_K1_R2048", serving_context=context) is False
    # a family the contract does not publish at all
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K1_R640", serving_context=context) is False
    # and the same admitted rung, asked with no scope at all: refused by the
    # SCOPE of the v8 table, not by the pin the fixture satisfied.
    assert tr.tessera_lane_attested("TESSERA_E2M1_K2_R896") is False


def test_the_synthesized_spec_reads_the_same_lookup(released_pin):
    """``producer_eligible`` is the AND of "the wire can carry it" and "a
    runtime serves it". The second conjunct is this lookup, so the menu admits
    exactly the attested rungs and nothing else -- at the scope the lookup is
    asked; a context-free spec on the scoped table is not eligible."""
    context = _dense_context()
    assert tr.synthesize_tessera_spec(
        "TESSERA_E2M1_K2_R896", serving_context=context).producer_eligible is True
    assert tr.synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024", serving_context=context).producer_eligible is True
    assert tr.synthesize_tessera_spec(
        "TESSERA_E2M1_K2_R512", serving_context=context).producer_eligible is False
    assert tr.synthesize_tessera_spec(
        "TESSERA_E2M1_K2_R896").producer_eligible is False


@pytest.mark.parametrize("mutation,name", [
    ({"qualification": "compile_only"}, "compile_only"),
    ({"route_status": "fallback", "requires_serve_flags": [],
      "requires_plugin": ""}, "fallback"),
])
def test_the_lookup_fails_closed_on_every_cell_axis(
        tmp_path, released_pin, mutation, name):
    """Cross-compilation is not a serve, and an announced fallback is not a
    native route. Neither admits, even with the release boundary satisfied."""
    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        cell.update(mutation)
    if name == "fallback":
        # Removing the required v4 plugin/residency declarations is malformed
        # before it can become a fallback candidate for menu admission.
        with pytest.raises(LaneEligibilityError, match="requires_plugin"):
            _load(_write(tmp_path, contract, f"{name}.json"))
        return
    table, formats = _load(_write(tmp_path, contract, f"{name}.json"))
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R896", table=table, formats=formats) is False


def test_the_rungs_a_cell_names_are_the_whole_admitted_set(tmp_path,
                                                           released_pin):
    """Absence is the only negative signal a closed-world v3 table carries.
    Dropping a rung from the cells removes it from the menu; nothing in this
    repository can widen it back."""
    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["family"] == "TESSERA_E2M1_K2":
            cell["rungs_q256"] = [896]
        else:
            cell["rungs_q256"] = [1024]
    for row in contract["formats"]:
        if row["family"] == "TESSERA_E2M1_K2":
            row["attested_rungs_q256"] = [896]
            row["reader_rate_range_q256"] = [768, 896]
    table, formats = _load(_write(tmp_path, contract, "narrow.json"))
    context = _dense_context()
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R896", table=table, formats=formats,
        serving_context=context) is True
    # inside the reader range, so the rate RESOLVES -- and is still refused,
    # because no cell names it.
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R768", table=table, formats=formats,
        serving_context=context) is False


def test_an_absent_table_admits_nothing(tmp_path, released_pin):
    """A released pin over a contract with no lane table is UNATTESTED, not a
    clean bill."""
    contract = _packaged_contract()
    contract.pop("lane_eligibility")
    table, formats = _load(_write(tmp_path, contract, "no_table.json"))
    assert table.present is False
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R896", table=table, formats=formats) is False


# ---------------------------------------------------------------------------
# The lane spec's principle-14 field
# ---------------------------------------------------------------------------
def test_lane_spec_executes_is_derived_from_the_packaged_contract():
    """The preflight this lane spec's ``if_this_changes`` note promises.

    ``served_activation_quantization.executes`` states what the serving runtime
    EXECUTES, so it is derived from the table the runtime publishes or it is a
    principle-14 violation written into a principle-14 field. Each glob is a
    ``formats[]`` row's ``name_pattern`` with ``{k}`` replaced by ``*``.
    """
    from prismaquant.lane_spec import load_lane_spec

    contract = _packaged_contract()
    derived = {
        str(row["name_pattern"]).replace("{k}", "*")
        for row in contract["formats"]
    }
    spec = load_lane_spec("tessera")
    assert spec.served_activation_quantization is not None
    assert set(spec.served_activation_quantization.executes) == derived
    # Derived by a second implementation rather than typed: this literal
    # re-staled within a day of being written, when the runtime published
    # TESSERA_BF16_K1.  What is pinned is the rule, not the roster.
    rows = _packaged_contract()["formats"]
    assert derived == {r["name_pattern"].replace("{k}", "*") for r in rows}
    assert len(derived) == len(rows) and derived

    # and every published rung of every family matches its own glob
    for row in contract["formats"]:
        # the current key, with the alias Tessera deprecates at schema v2 as a
        # fallback -- the same precedence tessera_runtime_contract.py parses by.
        for rung in row.get("attested_rungs_q256", row.get("candidate_rungs_q256")):
            name = str(row["name_pattern"]).replace("{k}", str(rung))
            assert spec.served_activation_quantization.matches(name)


def test_lane_spec_activation_contracts_match_the_cells():
    """The priced A side is what the attesting cells execute -- checked in code.

    Issue #165: the allocator prices ``tessera_serving_route``'s layout fact
    while the serve executes the cells' ``activation_contract``, and the two
    vocabularies do not match character for character
    (``w4a4-nvfp4-e2m1-group16-ue4m3`` vs ``e2m1_group16_ue4m3_static``).  The
    roster used to be a literal pair here, and it went stale the day
    Tessera's contract reached v5 and the 16-bit family gained its two cells
    -- a third family the test had no room for.  So the roster is read from
    the contract, and what is asserted is the property that has to hold at
    any number of families: a cell's family agrees with itself about its A
    side, and the value it publishes is the value the admission prices.

    The second leg is deliberately NOT substring containment in the lane
    spec's rationale prose: a test gate that reads prose is fixed by editing
    prose, which is the exact edit the field's own ``if_this_changes`` note
    forbids.  It is the runtime comparison
    (``tessera_menu.check_tessera_activation_agreement``), exercised here on
    the real packaged table through the real seam.
    """
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_formats import (
        tessera_serving_route, tessera_wire_recipe,
    )
    from prismaquant.tessera_runtime_contract import cell_activation_projection

    contract = _packaged_contract()
    by_family: dict[str, set[str]] = {}
    for cell in contract["lane_eligibility"]["cells"]:
        by_family.setdefault(cell["family"], set()).add(
            cell["activation_contract"])
    assert by_family, "the packaged contract publishes no cells at all"
    for family, contracts in by_family.items():
        assert len(contracts) == 1, (
            f"{family} publishes {len(contracts)} activation contracts across "
            f"its cells: {sorted(contracts)}")
    # and the formats[] row agrees with the cells that name it
    rows = {r["name_pattern"].split("_R{")[0]: r for r in contract["formats"]}
    for family, contracts in by_family.items():
        assert rows[family]["activation_contract"] == next(iter(contracts))

    # and the producer prices what the cells execute, at every rung the
    # cells attest -- derived from both owners, never a literal triple.
    for family, contracts in sorted(by_family.items()):
        bits, group = cell_activation_projection(next(iter(contracts)))
        rungs = sorted({
            rung
            for cell in contract["lane_eligibility"]["cells"]
            if cell["family"] == family
            for rung in cell["rungs_q256"]
        })
        assert rungs, f"{family} attests no rungs at all"
        for rung in rungs:
            wire = tessera_wire_recipe(family, rung)
            route = tessera_serving_route(family, wire, rung)
            assert (route.act_bits, route.act_group_size) == (bits, group), (
                f"{family} R{rung}: priced {(route.act_bits, route.act_group_size)} "
                f"but the cells execute {next(iter(contracts))} -> {(bits, group)}")
            # through the real seam, on the real packaged table: agreement
            # holds, so admission reports the priced value without raising.
            admission = tm.route_admission(f"{family}_R{rung}")
            assert admission.activation_contract == route.contract


def test_a_priced_a_side_the_cells_do_not_execute_is_refused():
    """Disagreement between the priced route and the executed contract raises."""
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_formats import (
        tessera_serving_route, tessera_wire_recipe,
    )

    wire = tessera_wire_recipe("TESSERA_E2M1_K2", 896)
    route = tessera_serving_route("TESSERA_E2M1_K2", wire, 896)
    assert (route.act_bits, route.act_group_size) == (4, 16)
    with pytest.raises(tm.TesseraMenuError, match="not what the attesting cells execute"):
        tm.check_tessera_activation_agreement(
            "TESSERA_E2M1_K2_R896", route, ["fp8_per_token_dynamic"])


def test_cells_disagreeing_about_the_a_side_are_refused():
    """One rung cannot execute two activation contracts."""
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_formats import (
        tessera_serving_route, tessera_wire_recipe,
    )

    wire = tessera_wire_recipe("TESSERA_E2M1_K2", 896)
    route = tessera_serving_route("TESSERA_E2M1_K2", wire, 896)
    with pytest.raises(tm.TesseraMenuError, match="disagree about the executed A side"):
        tm.check_tessera_activation_agreement(
            "TESSERA_E2M1_K2_R896", route,
            ["e2m1_group16_ue4m3_static", "fp8_per_token_dynamic"])


def test_an_unknown_cell_vocabulary_is_refused_not_guessed():
    """A runtime vocabulary the projection does not transcribe raises."""
    from prismaquant import tessera_menu as tm
    from prismaquant.tessera_formats import (
        tessera_serving_route, tessera_wire_recipe,
    )

    wire = tessera_wire_recipe("TESSERA_E2M1_K2", 896)
    route = tessera_serving_route("TESSERA_E2M1_K2", wire, 896)
    with pytest.raises(tm.TesseraMenuError, match="not a published vocabulary"):
        tm.check_tessera_activation_agreement(
            "TESSERA_E2M1_K2_R896", route, ["w99a99-future-static"])


def test_route_admission_refuses_a_drifted_a_side(tmp_path, monkeypatch):
    """The seam, not just the helper: a packaged table whose cells execute a
    different A side than the producer prices makes ``route_admission`` raise.
    """
    from prismaquant import tessera_menu as tm

    contract = _packaged_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["family"] == "TESSERA_E2M1_K2":
            cell["activation_contract"] = "fp8_per_token_dynamic"
    path = _write(tmp_path, contract, "a_side_drift.json")
    table, formats = _load(path)
    monkeypatch.setattr(tr, "_pinned_serving_table", lambda: (table, formats))
    # All metadata must come from this fixture's bytes, including the TP
    # ceiling. Otherwise digest consistency refuses before the A-side check.
    monkeypatch.setattr(tr, "tessera_serving_contract_path", lambda: path)
    with pytest.raises(tm.TesseraMenuError, match="not what the attesting cells execute"):
        tm.route_admission("TESSERA_E2M1_K2_R896", serving_context=_dense_context())


def test_the_tessera_lane_is_declared_and_advisory():
    from prismaquant.lane_spec import load_lane_spec

    spec = load_lane_spec("tessera")
    assert spec.runtime == "vllm+tessera_plugin"
    assert spec.advisory_gates is True
    assert spec.endpoint.kind == "openai"
    # No enable flag: the checkpoint selects the plugin. The one operator knob
    # is the residency, and the serve command carries it.
    assert "${TESSERA_SERVE_MODE}" in spec.serve_command
    assert any("tessera_plugin_served.sh" in t for t in spec.serve_command)
    # Fail-closed admission is stated where an operator reads it.
    assert any("release tag" in note.lower() for note in spec.notes)
    assert any("routed_moe" in note or "dense-only" in note.lower()
               for note in spec.notes)


# ---------------------------------------------------------------------------
# The provenance the answer travels under
# ---------------------------------------------------------------------------
def test_the_stamped_attestation_source_names_the_table_that_answered():
    """``RouteAdmission.source`` is provenance, and provenance must not lie.

    Until this test, every Tessera unit built without a development override
    was stamped ``gridbook_serving_runtime_pin:lane_eligibility`` -- a module
    archived on 2026-09-02 (``archive/gridbook_lane_2026-09-02/``) whose pin
    no longer governs anything.  The table that actually answers is Tessera's
    OWN packaged ``runtime_contract.json``, read through
    ``tessera_render._pinned_serving_table``.  Principle 14 is about the
    *value* a gate reads, not only about where the verdict comes from: a
    correct verdict carrying the name of a retired runtime is still an
    unattested claim about a runtime.
    """
    from prismaquant import tessera_menu as tm

    table, _formats = tr._pinned_serving_table()
    admission = tm.route_admission("TESSERA_E2M1_K2_R896")

    assert "gridbook" not in admission.source
    assert admission.source.startswith("tessera_packaged_contract:")
    # and it names the release the table came from, so two runs against
    # different packaged contracts cannot both claim "the Tessera contract".
    assert table.runtime_version
    assert admission.source.endswith(table.runtime_version)


def test_the_unattested_detail_names_the_conjunct_that_actually_refused():
    """The context-free refusal today is the SCOPE's, and the detail says so.

    The packaged contract publishes ``TESSERA_E2M1_K2`` and carries a
    device-qualified native cell naming R896 -- the released-pin test above
    proves it by admitting the same rung at the dense scope.  So a detail
    reading "the pinned serving release publishes no cell covering this
    family and rate" is false about the contract on disk, and it points a
    reader at re-pinning a table that already carries the row.  Before the
    pin resolved, the conjunct that refused here was the PIN (and the detail
    said "pin"); with the pin resolved and the table scoped (v8), the
    conjunct is the missing serving context, and the detail names that.
    """
    from prismaquant import tessera_menu as tm

    admission = tm.route_admission("TESSERA_E2M1_K2_R896")
    assert admission.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert "no cell covering" not in admission.detail
    assert admission.requires_serving_context is True
    assert "explicit serving context" in admission.detail
    assert "pin" not in admission.detail
    # the same rung, at the scope the cell publishes: admitted, by the pin.
    scoped = tm.route_admission("TESSERA_E2M1_K2_R896", serving_context=_dense_context())
    assert scoped.route_status in {tm.ROUTE_STATUS_BACKED, tm.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}

    # and a family the packaged contract never publishes keeps its OWN reason,
    # so the two refusals are not spelled the same: this one is fixed by a
    # contract that names the family, not by cutting a release tag.
    absent = tm.route_admission("TESSERA_E2M1_K1_R640")
    assert absent.route_status == tm.ROUTE_STATUS_UNATTESTED
    assert "pin" not in absent.detail
    assert "formats table publishes" in absent.detail
