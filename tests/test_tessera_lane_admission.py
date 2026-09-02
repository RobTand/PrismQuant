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
    FORMAT_KIND_TESSERA_WIRE,
    LANE_ELIGIBILITY_SCHEMA_TESSERA,
    LaneEligibilityError,
    load_eligibility_table,
    load_published_formats,
)
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_PLUGIN_NAME,
    TESSERA_SERVING_RUNTIME_COMMIT_PENDING,
    TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
    TESSERA_SERVING_RUNTIME_REPOSITORY,
    TESSERA_SERVING_RUNTIME_VERSION_PENDING,
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    parse_tessera_serving_runtime_pin,
    require_exact_tessera_runtime_release,
    tessera_serving_runtime_pin_path,
)

#: A resolved-looking commit that is NOT any real Tessera commit. It exists to
#: prove the machinery, and the constants are monkeypatched to match it, so no
#: fixture here can ever be mistaken for an attestation.
FIXTURE_COMMIT = "0" * 39 + "1"
FIXTURE_VERSION = "0.1.0"


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
    """A RELEASED Tessera pin, built through the real reader.

    The fixture writes a pin file with a resolved commit/version, points the
    two reviewed-release constants at it, and substitutes the loader on the pin
    MODULE -- which is where ``tessera_render._release_pin_satisfied`` looks it
    up, at call time.  Nothing here weakens the parser or the release check:
    both run exactly as they will on the day Rob cuts a tag.
    """
    payload = {
        "schema": TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
        "repository": TESSERA_SERVING_RUNTIME_REPOSITORY,
        "commit": FIXTURE_COMMIT,
        "version": FIXTURE_VERSION,
        "version_is_release": True,
        "runtime_contract_schema": "tessera.runtime-contract.v1",
        "plugin_entry_point": "tessera = tessera.serving:register",
    }
    path = tmp_path / "released_pin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_RELEASE_COMMIT", FIXTURE_COMMIT)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_RELEASE_VERSION", FIXTURE_VERSION)
    pin = pin_module.load_tessera_serving_runtime_pin(path)
    monkeypatch.setattr(
        pin_module, "load_tessera_serving_runtime_pin", lambda *a, **k: pin)
    require_exact_tessera_runtime_release(pin)   # the real gate, satisfied
    return pin


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------
def test_the_tracked_pin_is_pending_and_the_release_gate_refuses_it():
    """Admission answers False TODAY BY THE PIN, not by an edit.

    The tracked pin is structurally valid -- so it can be reviewed -- and is
    refused by the only function any gate consults. Both sentinels are pinned
    verbatim: they are the conspicuous marks that say "no release tag exists",
    and a silent drift to a resolved-looking value would be an admission.
    """
    pin = load_tessera_serving_runtime_pin()
    assert pin.commit == TESSERA_SERVING_RUNTIME_COMMIT_PENDING
    assert pin.version == TESSERA_SERVING_RUNTIME_VERSION_PENDING
    assert pin.version_is_release is False
    assert pin.commit_is_resolved is False
    assert pin.version_is_resolved is False
    assert pin.plugin_entry_point == "tessera = tessera.serving:register"

    with pytest.raises(TesseraServingRuntimePinError, match="PENDING"):
        require_exact_tessera_runtime_release(pin)
    assert tr._release_pin_satisfied() is False


def test_the_pin_file_on_disk_is_the_one_the_reader_parses():
    """No second spelling of the pin: the tracked JSON round-trips."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    assert parse_tessera_serving_runtime_pin(payload) == (
        load_tessera_serving_runtime_pin())


def test_a_pending_pin_cannot_be_marked_released():
    """The one structural rule that stops a half-edit admitting anything."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["version_is_release"] = True
    with pytest.raises(TesseraServingRuntimePinError, match="cannot be marked"):
        parse_tessera_serving_runtime_pin(payload)


def test_a_resolved_pin_that_is_not_the_reviewed_release_is_refused():
    """Resolving the JSON alone admits nothing: the constants must move too,
    in the same reviewed commit."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload.update(commit="a" * 40, version="9.9.9", version_is_release=True)
    pin = parse_tessera_serving_runtime_pin(payload)
    with pytest.raises(TesseraServingRuntimePinError, match="reviewed release"):
        require_exact_tessera_runtime_release(pin)


# ---------------------------------------------------------------------------
# The packaged table, through the shared parser
# ---------------------------------------------------------------------------
def test_the_packaged_tessera_contract_parses_and_every_cell_names_the_plugin():
    """The shared parser reads Tessera's table, and the plugin requirement is
    a parsed FIELD on every cell -- never prose a gate cannot read."""
    from importlib.resources import as_file

    with as_file(tr.tessera_serving_contract_path()) as path:
        table, formats = _load(path)

    assert table.present
    assert table.schema == LANE_ELIGIBILITY_SCHEMA_TESSERA
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
    # dense-only, and that is the honest state: no served measurement covers
    # routed experts, so the contract carries no routed_moe cell.
    assert table.structures == ("dense",)


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
    table, formats = _load(_write(tmp_path, contract, "no_plugin.json"))
    with pytest.raises(LaneEligibilityError, match="requires_plugin"):
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
    assert tr.tessera_lane_attested("TESSERA_E2M1_K2_R896") is True
    assert tr.tessera_lane_attested("TESSERA_E4M3_K1_R1024") is True
    # a serialisable rate no cell names, on a published family
    assert tr.tessera_lane_attested("TESSERA_E2M1_K2_R512") is False
    # the family's own terminal rate, which the reader range excludes
    assert tr.tessera_lane_attested("TESSERA_E4M3_K1_R2048") is False
    # a family the contract does not publish at all
    assert tr.tessera_lane_attested("TESSERA_E2M1_K1_R640") is False


def test_the_synthesized_spec_reads_the_same_lookup(released_pin):
    """``producer_eligible`` is the AND of "the wire can carry it" and "a
    runtime serves it". The second conjunct is this lookup, so the menu admits
    exactly the attested rungs and nothing else."""
    assert tr.synthesize_tessera_spec(
        "TESSERA_E2M1_K2_R896").producer_eligible is True
    assert tr.synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024").producer_eligible is True
    assert tr.synthesize_tessera_spec(
        "TESSERA_E2M1_K2_R512").producer_eligible is False


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
            row["candidate_rungs_q256"] = [896]
            row["reader_rate_range_q256"] = [768, 896]
    table, formats = _load(_write(tmp_path, contract, "narrow.json"))
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R896", table=table, formats=formats) is True
    # inside the reader range, so the rate RESOLVES -- and is still refused,
    # because no cell names it.
    assert tr.tessera_lane_attested(
        "TESSERA_E2M1_K2_R768", table=table, formats=formats) is False


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
    assert derived == {"TESSERA_E2M1_K2_R*", "TESSERA_E4M3_K1_R*"}

    # and every published rung of every family matches its own glob
    for row in contract["formats"]:
        for rung in row["candidate_rungs_q256"]:
            name = str(row["name_pattern"]).replace("{k}", str(rung))
            assert spec.served_activation_quantization.matches(name)


def test_lane_spec_activation_contracts_match_the_cells():
    """The rationale's two contracts are the cells' own values, per family."""
    contract = _packaged_contract()
    by_family: dict[str, set[str]] = {}
    for cell in contract["lane_eligibility"]["cells"]:
        by_family.setdefault(cell["family"], set()).add(
            cell["activation_contract"])
    assert by_family == {
        "TESSERA_E2M1_K2": {"e2m1_group16_ue4m3_static"},
        "TESSERA_E4M3_K1": {"fp8_per_token_dynamic"},
    }

    from prismaquant.lane_spec import load_lane_spec

    rationale = load_lane_spec(
        "tessera").served_activation_quantization.rationale
    for contracts in by_family.values():
        for value in contracts:
            assert value in rationale


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
