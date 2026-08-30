"""Route status is ATTESTED or refused — never asserted, never a vacuous zero.

THE DEFECT THIS PINS, measured twice.

1. The shipped DSv4 87 GB body carries 11 routed FP8-CB layers whose
   ``gate_proj``/``up_proj`` bind distinct learned codebooks. Gridbook's
   persistent-B prefill lane refuses per-role split books, so above the token
   threshold those layers take the announced expand+grouped-bridge route. No CB
   serving lane declared a structured ``route_status``, so persistent-B
   eligibility was never a gate input and a user discovered it at serve time.

2. Its twin on the vanilla-vLLM lane: ``units_on_fallback_route = 0`` in a
   shipped ``selection.json``, reachable only by never having looked. No spec
   declared route status at all, so every unit ``continue``d before it could be
   counted. Absence of evidence was rendering as evidence of absence.

The property under test is therefore about SHAPE as much as values: an absent
attestation must be *unrepresentable* as a clean bill. The synthetic tables here
exercise the gate's four dispositions; the ABSENT test runs against the REAL
materialized Gridbook 0.8.11 contract, because that is the state the repository
actually ships in and the one a vacuous zero would hide.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant.cb_route_status_gate import (
    NON_NATIVE_TARGET_ENV,
    ROUTE_OVERRIDE_ENV,
    CBRouteStatusRefusal,
    evaluate_cb_route_status,
    gate_cb_export_units,
    require_cb_route_status,
    shipcard_route_summary,
)
from prismaquant.gridbook_lane_eligibility import (
    CELL_ROUTE_STATUSES,
    LANE_ELIGIBILITY_SCHEMA,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    ROUTE_STATUS_FALLBACK,
    ROUTE_STATUS_UNATTESTED,
    ROUTE_STATUS_UNBACKED,
    GridbookLaneEligibilityError,
    UnitStructuralFacts,
    load_contract_index,
    load_eligibility_table,
    load_published_formats,
    materialized_contract_path,
    resolve_unit_route,
    unit_structural_facts,
)


REPO = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO / "prismaquant" / "gridbook_runtime"

#: A byte-verbatim copy of the runtime contract Gridbook publishes at commit
#: 30287aa (contract v12, lane_eligibility v3). It is a TEST FIXTURE and is
#: deliberately NOT in prismaquant/gridbook_runtime/: materializing it there
#: would be the pin bump, which is release-keyed and Rob's call. Its only job
#: is to prove this parser reads the shape the publisher really emits, without
#: importing gridbook (AGENTS.md:38).
V12_FIXTURE = (Path(__file__).resolve().parent / "fixtures"
               / "gridbook_runtime_contract.v12.30287aa.json")
V12_FIXTURE_SHA256 = (
    "836b7831aa8bbad30170bcae56a1b01e08031ac3159914973f0a1bd15edc4f24")

#: The platform the synthetic CB cells below are scoped to. v3 cells are
#: platform-scoped, so every resolution in this file names one explicitly.
TEST_PLATFORM = "sm_120"
TRELLIS_PLATFORM = "sm_121"

_FP8_RUNGS = [28, 40, 44, 48]
_NVFP4_RUNGS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_MIDM_FLAG = "PRISMAQUANT_CB_FP4_FUSED_MIDM=1"
_E2M1_FLAGS = ["GRIDBOOK_TRELLIS_E2M1=1",
               "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed"]


def _cell(cid, *, platform, family, structure, regime, route_status,
          qualification="compile_only", rungs=None, rungs_q256=None,
          activation_contract=None, flags=(), predicates=()):
    """One v3 cell, with the publisher's exact key set for its family kind."""
    payload = {
        "id": cid,
        "platform": platform,
        "family": family,
        "structure": structure,
        "regime": regime,
        "route_status": route_status,
        "qualification": qualification,
        "requires_serve_flags": list(flags),
        "predicates": [dict(p) for p in predicates],
    }
    if rungs_q256 is not None:
        payload["rungs_q256"] = list(rungs_q256)
        payload["activation_contract"] = activation_contract
    else:
        payload["rungs"] = list(rungs)
    return payload


# ---------------------------------------------------------------------------
# A synthetic v3 table modelling the MEASURED Gridbook behaviour.
#
# It is a FIXTURE, never a shipped attestation: writing these values into
# prismaquant/ would be exactly the transcription principle 14 forbids. It
# exists so the gate's logic can be tested over cases the published table does
# not currently contain (a fallback-only unit, a mid-M opt-in flag), and its
# shape is the publisher's own -- platform-scoped cells with an explicit rung
# list, dispatched on each family's ``formats[].kind``.
#
# The base contract is the byte-verbatim v12 fixture, so the CB/trellis kind
# discriminator under test is the real one and not a test invention.
# ---------------------------------------------------------------------------
def _table_payload() -> dict:
    return {
        "schema": LANE_ELIGIBILITY_SCHEMA,
        "platforms": {
            TEST_PLATFORM: {"compute_capability": [12, 0]},
            TRELLIS_PLATFORM: {"compute_capability": [12, 1]},
        },
        "regimes": ["decode", "batch"],
        "structures": ["dense", "routed_moe"],
        "cells": [
            # 0: per-role GEMVs below the token threshold.
            _cell("cb_moe_gemv_decode", platform=TEST_PLATFORM,
                  family="FP8_CB_K", structure="routed_moe", regime="decode",
                  route_status="backed", rungs=_FP8_RUNGS,
                  predicates=[{"fact": "out_features", "op": "multiple_of",
                               "value": 16}]),
            # 1: decode-in-mainloop prefill; canonical books only.
            _cell("cb_moe_persistent_b", platform=TEST_PLATFORM,
                  family="FP8_CB_K", structure="routed_moe", regime="batch",
                  route_status="backed", rungs=_FP8_RUNGS,
                  predicates=[{"fact": "role_split", "op": "equals",
                               "value": False}]),
            # 2: announced expand + grouped bridge; role-split books.
            _cell("cb_moe_expand_bridge", platform=TEST_PLATFORM,
                  family="FP8_CB_K", structure="routed_moe", regime="batch",
                  route_status="fallback", rungs=_FP8_RUNGS),
            # 3/4: dense NVFP4, decode backed and batch behind an opt-in flag.
            _cell("cb_dense_decode", platform=TEST_PLATFORM,
                  family="NVFP4_CB_K", structure="dense", regime="decode",
                  route_status="backed", rungs=_NVFP4_RUNGS,
                  predicates=[{"fact": "in_features", "op": "multiple_of",
                               "value": 256}]),
            _cell("cb_dense_midm_optin", platform=TEST_PLATFORM,
                  family="NVFP4_CB_K", structure="dense", regime="batch",
                  route_status="backed_with_serve_flag", rungs=_NVFP4_RUNGS,
                  flags=[_MIDM_FLAG],
                  predicates=[{"fact": "in_features", "op": "multiple_of",
                               "value": 256}]),
            # 5/6: routed NVFP4 serves in BOTH regimes, natively in NEITHER.
            # This is the only way a v3 table can say "unbacked": by publishing
            # a fallback everywhere and nothing better.
            _cell("cb_moe_nvfp4_expand_decode", platform=TEST_PLATFORM,
                  family="NVFP4_CB_K", structure="routed_moe", regime="decode",
                  route_status="fallback", rungs=_NVFP4_RUNGS),
            _cell("cb_moe_nvfp4_expand_batch", platform=TEST_PLATFORM,
                  family="NVFP4_CB_K", structure="routed_moe", regime="batch",
                  route_status="fallback", rungs=_NVFP4_RUNGS),
            # 7/8: the trellis lane, device-qualified behind operator flags.
            _cell("trellis_e2m1_dense_decode", platform=TRELLIS_PLATFORM,
                  family="TCQ_E2M1_R256", structure="dense", regime="decode",
                  route_status="backed_with_serve_flag",
                  qualification="device_qualified", rungs_q256=[512],
                  activation_contract="e2m1_group16_ue4m3_static",
                  flags=_E2M1_FLAGS),
            _cell("trellis_e2m1_dense_batch", platform=TRELLIS_PLATFORM,
                  family="TCQ_E2M1_R256", structure="dense", regime="batch",
                  route_status="backed_with_serve_flag",
                  qualification="device_qualified", rungs_q256=[512],
                  activation_contract="e2m1_group16_ue4m3_static",
                  flags=_E2M1_FLAGS),
        ],
    }


#: Index of the flag-gated CB cell inside ``_table_payload()["cells"]``.
_MIDM_CELL = 4
#: Index of a trellis cell.
_TRELLIS_CELL = 7


def _contract_with(block, tmp_path: Path, name: str = "c.json") -> Path:
    contract = json.loads(V12_FIXTURE.read_text())
    contract["lane_eligibility"] = block
    path = tmp_path / name
    path.write_text(json.dumps(contract))
    return path


@pytest.fixture()
def attested_table(tmp_path: Path):
    return load_eligibility_table(
        "0.9.1-test",
        contract_path=_contract_with(_table_payload(), tmp_path))


@pytest.fixture(scope="module")
def v12_formats():
    return {
        str(entry["family"]): dict(entry)
        for entry in json.loads(V12_FIXTURE.read_text())["formats"]
    }


def _facts(qname, fmt, *, routed=True, role_split=False,
           in_features=2048, out_features=1408, formats=None):
    return unit_structural_facts(
        qname, fmt,
        is_routed_moe=routed,
        role_split=role_split,
        in_features=in_features,
        out_features=out_features,
        published_formats=(
            formats if formats is not None else load_published_formats()),
    )


def _v12_facts(qname, fmt, **kw):
    """Facts derived from the v12 formats table (the one with trellis rows)."""
    formats = {
        str(entry["family"]): dict(entry)
        for entry in json.loads(V12_FIXTURE.read_text())["formats"]
    }
    return _facts(qname, fmt, formats=formats, **kw)


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------
def test_the_shipped_contract_index_matches_the_serving_pin():
    """The materialized contract is bound to the release it claims to be."""
    index = load_contract_index()
    pin = json.loads(
        (ASSET_DIR / "gridbook_serving_runtime_pin.json").read_text())
    entry = next(e for e in index["contracts"]
                 if e["version"] == pin["version"])
    assert entry["commit"] == pin["commit"]
    assert entry["runtime_contract_schema"] == pin["runtime_contract_schema"]
    path = materialized_contract_path(pin["version"])
    assert path is not None and path.exists()
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_an_unknown_predicate_fact_is_a_malformed_contract_not_a_no_op(tmp_path):
    """An ignored predicate would make a NARROWER rule read as unconditional."""
    block = _table_payload()
    block["cells"][0]["predicates"].append(
        {"fact": "sm_capability", "op": "equals", "value": 121})
    with pytest.raises(GridbookLaneEligibilityError, match="sm_capability"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_flag_gated_route_must_name_its_flag(tmp_path):
    """BITE: an operator cannot reach an unnamed flag."""
    block = _table_payload()
    block["cells"][_MIDM_CELL]["requires_serve_flags"] = []
    with pytest.raises(GridbookLaneEligibilityError, match="unnamed flag"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_trellis_cell_with_an_empty_flag_list_is_refused(tmp_path):
    """BITE, on the shape that matters: the trellis lane IS flag-gated.

    Every published trellis cell is ``backed_with_serve_flag``. Dropping its
    flags would make a route an operator cannot actually reach read as one
    they can, on the exact lane this parser was written to admit.
    """
    block = _table_payload()
    block["cells"][_TRELLIS_CELL]["requires_serve_flags"] = []
    with pytest.raises(GridbookLaneEligibilityError, match="unnamed flag"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_flags_without_the_flag_gated_status_are_refused(tmp_path):
    """The converse: naming flags on a plain ``backed`` cell is incoherent."""
    block = _table_payload()
    block["cells"][0]["requires_serve_flags"] = ["GRIDBOOK_SOMETHING=1"]
    with pytest.raises(GridbookLaneEligibilityError, match="by definition"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_wrong_schema_is_refused(tmp_path):
    block = _table_payload()
    block["schema"] = "gridbook.lane-eligibility.v99"
    with pytest.raises(GridbookLaneEligibilityError, match="schema"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_v2_lane_table_is_refused(tmp_path):
    """BITE: v2 is not a subset of v3 and must not be read as one.

    A v2 table's rules carry no platform and no rung list. Parsing one with a
    v3 reader would make every rule match every platform and every rung -- the
    precise overclaim this rewrite exists to prevent -- so the schema string is
    an equality test, never a floor.
    """
    v2 = {
        "schema": "gridbook.lane-eligibility.v2",
        "regimes": ["decode", "batch"],
        "lanes": [{
            "id": "cb_moe_gemv_decode",
            "regime": "decode",
            "structure": "routed_moe",
            "route_status": "backed",
            "predicates": [],
        }],
    }
    with pytest.raises(GridbookLaneEligibilityError) as exc:
        load_eligibility_table("x", contract_path=_contract_with(v2, tmp_path))
    message = str(exc.value)
    assert "gridbook.lane-eligibility.v3" in message
    assert "v2" in message


def test_family_facts_are_derived_from_the_published_format_table():
    """n_sub and rung legality come from the CONTRACT, not from cb_layout."""
    published = load_published_formats()
    assert {"NVFP4_CB_K", "FP8_CB_K"} <= set(published)
    assert _facts("e", "FP8_CB_K28").n_sub == 4
    assert _facts("e", "NVFP4_CB_K16").n_sub == 2
    # A rung the pinned release does not instantiate leaves k None, so every
    # k-predicate fails closed rather than passing on an invented rung.
    off_law = _facts("e", "NVFP4_CB_K99")
    assert off_law.k is None and off_law.n_sub is None


# ---------------------------------------------------------------------------
# 2. Gate behaviour on a synthetic selection
# ---------------------------------------------------------------------------
def test_backed_unit_passes_with_no_fallback_recorded(attested_table):
    verdict = evaluate_cb_route_status(
        [_facts("l0.experts", "FP8_CB_K28")], table=attested_table,
        target_platform=TEST_PLATFORM)
    assert not verdict.refused
    assert verdict.provenance["units_backed"] == 1
    assert verdict.provenance["units_with_announced_fallback"] == 0
    assert verdict.provenance["units_unbacked"] == 0


def test_the_dsv4_per_role_case_is_recorded_not_refused(attested_table):
    """decode backed + batch announced fallback = the MEASURED DSv4 state."""
    unit = _facts("l22.experts", "FP8_CB_K28", role_split=True)
    route = resolve_unit_route(unit, attested_table, platform=TEST_PLATFORM)

    assert route.route_status == ROUTE_STATUS_BACKED
    assert route.fallback_regimes == ("batch",)
    by_regime = {r.regime: r for r in route.regimes}
    assert by_regime["decode"].cell_id == "cb_moe_gemv_decode"
    assert by_regime["batch"].cell_id == "cb_moe_expand_bridge"

    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert not verdict.refused, "an announced fallback SERVES; refusing is wrong"
    assert verdict.provenance["units_with_announced_fallback"] == 1
    assert verdict.provenance["announced_fallback_units"] == ["l22.experts"]
    assert verdict.provenance["by_regime"]["batch"]["fallback"] == 1
    assert any("ANNOUNCED FALLBACK" in w for w in verdict.warnings)


def test_a_unit_no_cell_covers_fails_export_closed_as_unattested(
        attested_table):
    """out_features % 16 != 0 has no decode cell, so nothing claims it.

    Under a v3 table this is UNATTESTED, not ``unbacked``: the runtime never
    enumerates what it refuses, so silence is the only negative signal it has.
    The gate must still fail closed on it -- a gate that admitted an uncovered
    unit would turn that one signal into no signal at all.
    """
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    route = resolve_unit_route(unit, attested_table, platform=TEST_PLATFORM)
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    assert route.in_scope is True
    assert route.unattested_regimes == ("decode",)

    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert verdict.refused
    assert verdict.provenance["units_unattested_in_scope"] == 1
    assert verdict.provenance["units_unbacked"] == 0
    assert verdict.provenance["unbacked_disposition"] == "refused"
    assert "NO backed serving route" in verdict.refusal_reason
    with pytest.raises(CBRouteStatusRefusal):
        require_cb_route_status([unit], table=attested_table,
                                target_platform=TEST_PLATFORM)


def test_a_fallback_only_unit_is_attested_unbacked_and_refused(attested_table):
    """Every regime serves, none natively -- principle 9's ``unbacked``.

    Distinct from the case above: here the runtime DID publish a route for
    every regime, and every one of them is an announced fallback. The status is
    attested and negative, and the gate refuses it just the same.
    """
    unit = _facts("l5.experts", "NVFP4_CB_K16")
    route = resolve_unit_route(unit, attested_table, platform=TEST_PLATFORM)
    assert route.route_status == ROUTE_STATUS_UNBACKED
    assert route.attested is True
    assert route.fallback_regimes == ("decode", "batch")

    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert verdict.refused
    assert verdict.provenance["units_unbacked"] == 1
    assert verdict.provenance["units_unattested_in_scope"] == 0


def test_an_explicit_override_ships_and_is_stamped(attested_table):
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    provenance = require_cb_route_status(
        [unit], table=attested_table, target_platform=TEST_PLATFORM,
        override_reason="research arm; serving gap tracked in gridbook#48")
    assert provenance["unbacked_disposition"] == "explicit_override"
    assert provenance["override"]["reason"].startswith("research arm")
    assert provenance["override"]["env"] == ROUTE_OVERRIDE_ENV
    # The stamp survives into the shipcard summary; an override nothing can
    # read is the confession-log failure mode all over again.
    assert shipcard_route_summary(provenance)["override"]["reason"]


def test_a_declared_non_native_target_ships_and_is_stamped(attested_table):
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    provenance = require_cb_route_status(
        [unit], table=attested_table, target_platform=TEST_PLATFORM,
        non_native_target="sm90-a100-fallback")
    assert provenance["unbacked_disposition"] == "declared_non_native_target"
    assert provenance["declared_non_native_target"] == "sm90-a100-fallback"
    assert shipcard_route_summary(provenance)[
        "declared_non_native_target"] == "sm90-a100-fallback"


def test_a_serve_flag_route_is_backed_with_serve_flag_and_names_the_flag(
        attested_table):
    unit = _facts("attn.o_proj", "NVFP4_CB_K16", routed=False,
                  in_features=2048, out_features=2048)
    route = resolve_unit_route(unit, attested_table, platform=TEST_PLATFORM)
    assert route.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert verdict.provenance["requires_serve_flags"] == [_MIDM_FLAG]
    assert not verdict.refused

    # BITE: backed_with_serve_flag is NOT backed, end to end. Flattening the
    # two is precisely the overclaim Gridbook refused to make.
    p = verdict.provenance
    assert p["units_backed"] == 0
    assert p["units_backed_with_serve_flag"] == 1
    assert p["units_by_route_status"] == {ROUTE_STATUS_BACKED_WITH_SERVE_FLAG: 1}
    summary = shipcard_route_summary(p)
    assert summary["units_backed"] == 0
    assert summary["units_backed_with_serve_flag"] == 1
    assert summary["requires_serve_flags"] == [_MIDM_FLAG]


def test_env_supplies_the_override_when_the_caller_does_not(
        attested_table, monkeypatch):
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    monkeypatch.setenv(ROUTE_OVERRIDE_ENV, "one-off bringup")
    provenance = require_cb_route_status([unit], table=attested_table,
                                         target_platform=TEST_PLATFORM)
    assert provenance["override"]["reason"] == "one-off bringup"
    monkeypatch.delenv(ROUTE_OVERRIDE_ENV)
    monkeypatch.setenv(NON_NATIVE_TARGET_ENV, "gb10-eager")
    provenance = require_cb_route_status([unit], table=attested_table,
                                         target_platform=TEST_PLATFORM)
    assert provenance["declared_non_native_target"] == "gb10-eager"


def test_a_mixed_selection_counts_every_disposition(attested_table):
    units = [
        _facts("l0.experts", "FP8_CB_K28"),
        _facts("l22.experts", "FP8_CB_K28", role_split=True),
        _facts("l23.experts", "FP8_CB_K28", role_split=True),
        _facts("attn.o_proj", "NVFP4_CB_K16", routed=False,
               in_features=2048, out_features=2048),
    ]
    verdict = evaluate_cb_route_status(units, table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert not verdict.refused
    assert verdict.provenance["units_total"] == 4
    assert verdict.provenance["units_with_announced_fallback"] == 2
    assert verdict.provenance["units_backed"] == 3
    assert verdict.provenance["units_backed_with_serve_flag"] == 1


# ---------------------------------------------------------------------------
# 3. The ABSENT path — the exact defect of units_on_fallback_route = 0
# ---------------------------------------------------------------------------
def test_the_real_pinned_release_publishes_no_eligibility_table():
    """Measured, not assumed: Gridbook 0.8.11 packages no lane_eligibility.

    (0.8.11's packaged contract is byte-identical to 0.8.10's; the serving
    pin, not this file, says which release the claim is made of.)
    """
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.11.json").read_text())
    assert "lane_eligibility" not in contract
    # What it DOES publish, so a future reader can see the gap precisely.
    assert set(contract) == {
        "schema", "contract_version", "abi_features", "quant_method",
        "packing", "layout", "formats", "producer_profiles",
    }
    table = load_eligibility_table()
    assert table.present is False
    assert table.runtime_version == "0.8.11"


def test_absent_attestation_reports_unattested_and_never_a_zero():
    """A vacuous zero must be UNREPRESENTABLE, not merely discouraged."""
    units = [_facts("l0.experts", "FP8_CB_K28"),
             _facts("l22.experts", "FP8_CB_K28", role_split=True)]
    verdict = evaluate_cb_route_status(units)  # real pinned table => ABSENT

    assert not verdict.refused, "absence is not evidence of an unbacked route"
    assert not verdict.attested
    p = verdict.provenance
    assert p["route_attestation"] == ROUTE_STATUS_UNATTESTED
    assert p["units_unattested"] == 2
    assert p["attestation"]["status"] == "absent"
    assert p["attestation"]["gridbook_serving_commit"].startswith("187c721")

    # THE POINT: none of the counters that could be misread as a clean bill
    # exist in this payload at all.
    for forbidden in ("units_backed", "units_unbacked",
                      "units_with_announced_fallback", "units_by_route_status",
                      "units_on_fallback_route", "by_regime"):
        assert forbidden not in p, (
            f"{forbidden!r} must not exist under an absent attestation; that "
            "is the units_on_fallback_route=0 defect")
    assert any("UNATTESTED" in w for w in verdict.warnings)

    summary = shipcard_route_summary(p)
    assert summary["units_unattested"] == 2
    assert "units_backed" not in summary and "units_unbacked" not in summary


def _cb_artifact(tmp_path: Path, provenance: dict) -> Path:
    """The minimum an artifact needs for `build_shipcard` to open a card."""
    root = tmp_path / "exported"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "quant_config.json").write_text(
        json.dumps({"quant_method": "gridbook", "provenance": provenance}))
    return root


def test_the_route_census_lands_on_the_written_shipcard(tmp_path,
                                                        attested_table):
    """Principle 9: the disposition is stamped ON THE CARD, not only in
    `quant_config`.

    Both CB exporters stamp the gate verdict into quant-config provenance
    before opening the card, so the card is where a publisher, a reviewer, or
    `publish_artifact` can see it without parsing the export config. A census
    that stops at `quant_config` is the confession-log failure mode with an
    extra step.
    """
    from prismaquant.shipcard import build_shipcard, load_shipcard, \
        write_shipcard

    provenance = require_cb_route_status(
        [_facts("l1.experts", "FP8_CB_K28", out_features=1410)],
        table=attested_table, target_platform=TEST_PLATFORM,
        non_native_target="sm90-a100-fallback")
    root = _cb_artifact(tmp_path, {"cb_route_status": provenance})

    card = build_shipcard(root, build={"quant_method": "gridbook"})
    path = write_shipcard(root / "shipcard.json", card)
    on_disk = load_shipcard(path)["cb_route_status"]

    assert on_disk["unbacked_disposition"] == "declared_non_native_target"
    assert on_disk["declared_non_native_target"] == "sm90-a100-fallback"
    assert on_disk["units_unattested_in_scope"] == 1
    # The card names the release the claim came from, AND the platform the
    # claim is scoped to; a route status with no runtime identity or no
    # hardware scope attests nothing (principle 14 and its corollary).
    assert on_disk["gridbook_serving_version"] == "0.9.1-test"
    assert on_disk["target_platform"] == TEST_PLATFORM


def test_an_unattested_card_publishes_no_route_counters(tmp_path):
    """The card inherits the gate's shape rule, so it cannot read as clean."""
    from prismaquant.shipcard import build_shipcard

    verdict = evaluate_cb_route_status(
        [_facts("l0.experts", "FP8_CB_K28")])  # real pin => ABSENT
    root = _cb_artifact(tmp_path, {"cb_route_status": verdict.provenance})

    summary = build_shipcard(root, build={"quant_method": "gridbook"})[
        "cb_route_status"]
    assert summary["route_attestation"] == "unattested"
    assert summary["units_unattested"] == 1
    for forbidden in ("units_backed", "units_unbacked",
                      "units_on_fallback_route", "units_by_route_status"):
        assert forbidden not in summary


def test_a_card_without_a_route_stamp_omits_the_field(tmp_path):
    """Negative control: the field must not materialize out of nothing.

    Non-CB artifacts never run this gate, and an empty census on one of those
    would be the vacuous zero wearing a different name.
    """
    from prismaquant.shipcard import build_shipcard

    root = _cb_artifact(tmp_path, {})
    assert "cb_route_status" not in build_shipcard(root, build={})


def test_every_unit_resolves_unattested_under_the_real_pin():
    table = load_eligibility_table()
    for fmt in ("FP8_CB_K28", "NVFP4_CB_K16", "BF16"):
        route = resolve_unit_route(_facts("u", fmt), table,
                                   platform=TEST_PLATFORM)
        assert route.route_status == ROUTE_STATUS_UNATTESTED
        assert route.attested is False
        assert route.regimes == ()


def test_a_pin_with_no_materialized_contract_attests_nothing(tmp_path):
    """Fail-closed: an unindexed release backs nothing, it does not pass."""
    table = load_eligibility_table(
        "9.9.9", contract_path=tmp_path / "missing.json")
    assert table.present is False
    assert "no materialized Gridbook runtime contract" in table.absent_reason


# ---------------------------------------------------------------------------
# 4. The serving-profile lane carries the structured field
# ---------------------------------------------------------------------------
def test_every_cb_serving_lane_declares_a_route_status_source():
    spec = json.loads(
        (REPO / "prismaquant" / "serving_profile_specs" / "nvfp4_cb.json"
         ).read_text())
    lanes = spec["serving_lanes"]
    assert lanes, "the CB profile must declare serving lanes"
    for lane in lanes:
        source = lane.get("route_status_source")
        assert source, f"{lane['id']}: no route_status_source (R3)"
        assert source["attestation"] == (
            "gridbook_runtime_contract.lane_eligibility")
        assert source["structures"], f"{lane['id']}: names no structural class"
        # A verdict may NEVER be a literal in this file.
        assert "route_status" not in source, (
            f"{lane['id']}: route_status_source carries a VERDICT. Principle "
            "14 takes a hand-typed claim about another runtime as a refusal; "
            "the spec declares the key, the contract supplies the value.")


def test_the_resolved_lane_exposes_structured_route_status():
    from prismaquant.serving_profiles import serving_lane_route

    lane = serving_lane_route("nvfp4_cb", "FP8_CB_K28")
    assert lane is not None
    payload = lane.as_dict()
    # Principle 9's field, present and structured, on the real pin.
    assert payload["route_status"] == ROUTE_STATUS_UNATTESTED
    assert payload["requires_serve_flags"] == []
    assert payload["route_status_source"].endswith(":absent")
    assert "0.8.11" in payload["route_status_source"]


@pytest.fixture()
def present_pin(monkeypatch):
    """Resolve serving lanes against the PUBLISHED v12 table, not the pin.

    Nothing exercised ``ServingLaneSpec.route_status_for`` on a present table
    before this: the real pin's table is absent, so the function returned at
    its first branch and every line past it was unreached. That is how it kept
    reading ``table.rules`` -- a name lane-eligibility v3 does not have -- and
    would have raised AttributeError the moment the pin advanced. Unused
    because unmeasured is a gap, not evidence.
    """
    from prismaquant import gridbook_lane_eligibility as le
    from prismaquant import serving_profiles as sp

    formats = {
        str(entry["family"]): dict(entry)
        for entry in json.loads(V12_FIXTURE.read_text())["formats"]
    }
    monkeypatch.setattr(
        le, "load_eligibility_table",
        lambda *a, **kw: load_eligibility_table(
            "0.9.1-fixture", contract_path=V12_FIXTURE))
    monkeypatch.setattr(le, "load_published_formats",
                        lambda *a, **kw: formats)
    sp._reset_eligibility_table_cache()
    yield
    sp._reset_eligibility_table_cache()


def _cb_lane(fmt):
    from prismaquant.serving_profiles import load_serving_profile

    profile = load_serving_profile("nvfp4_cb")
    chosen = None
    for lane in profile.serving_lanes:
        if lane.covers(fmt):
            chosen = lane
    assert chosen is not None, f"no CB lane covers {fmt}"
    return chosen


def test_a_present_table_resolves_a_lane_without_touching_table_rules(
        present_pin):
    """The v3 rename is followed through into the serving-lane resolver."""
    status, flags, source = _cb_lane("FP8_CB_K44").route_status_for(
        "FP8_CB_K44", platform="sm_120")
    # sm_120/FP8_CB_K/K44 is covered in both regimes, and the routed_moe batch
    # cell predicates on role_split -- a fact only the export gate holds. The
    # lane says so rather than guessing a lane-wide verdict.
    assert status == "unit_dependent"
    assert flags == ()
    assert source == ("gridbook_runtime_contract:0.9.1-fixture"
                      ":unit_dependent(cb_route_status_gate)")


def test_a_lane_resolved_without_a_platform_stays_unattested(present_pin):
    """BITE: v3 cells are platform-scoped even at lane granularity."""
    status, flags, source = _cb_lane("FP8_CB_K44").route_status_for(
        "FP8_CB_K44", platform=None)
    assert status == ROUTE_STATUS_UNATTESTED
    assert flags == ()
    assert source.endswith(":no_target_platform")


def test_a_lane_whose_rung_no_cell_lists_stays_unattested(present_pin):
    """K32 is a published codec rung; no lane cell covers it."""
    lane = _cb_lane("FP8_CB_K32")
    assert lane.route_status_for("FP8_CB_K32", platform="sm_120")[2].endswith(
        ":rung_not_listed")
    # ...and a rung that IS listed does not take that branch.
    assert not lane.route_status_for(
        "FP8_CB_K44", platform="sm_120")[2].endswith(":rung_not_listed")


def test_a_lane_on_an_unpublished_platform_stays_unattested(present_pin):
    status, _, source = _cb_lane("FP8_CB_K44").route_status_for(
        "FP8_CB_K44", platform="sm_121")
    assert status == ROUTE_STATUS_UNATTESTED
    assert source.endswith(":no_cell"), (
        "the published table names no CB cell on sm_121; saying so is the "
        "table reporting a serving gap, which is the signal it carries")


def test_selection_provenance_distinguishes_unattested_from_zero():
    """The vllm-lane twin: no spec declares route status, so say so."""
    from prismaquant.allocator_candidates import (
        selection_serving_lane_provenance,
    )

    cb = selection_serving_lane_provenance(
        {"a": "FP8_CB_K28", "b": "NVFP4_CB_K16"}, None, "nvfp4_cb")
    assert cb["route_status_counts"] == {"unattested": 2}
    assert cb["route_status_attested"] is False

    vllm = selection_serving_lane_provenance(
        {"a": "NVFP4", "b": "BF16"}, None, "vllm_packed_moe")
    # units_on_fallback_route stays 0 here and is STILL vacuous -- which is
    # exactly why the census below has to be read instead.
    assert vllm["units_on_fallback_route"] == 0
    assert vllm["route_status_counts"] == {"no_declared_lane": 2}
    assert vllm["route_status_attested"] is False


# ---------------------------------------------------------------------------
# 5. The export-side adapter both exporters share
# ---------------------------------------------------------------------------
def test_gate_cb_export_units_maps_stack_shapes_and_role_split(attested_table,
                                                               monkeypatch):
    """A 3-D stack shape is (E, summed_rows, in); a dense one is (out, in)."""
    import prismaquant.cb_route_status_gate as gate

    monkeypatch.setattr(
        gate, "evaluate_cb_route_status",
        lambda facts, **kw: gate.RouteGateVerdict(
            provenance={"seen": [f.as_dict() for f in facts]}, refused=False))
    shapes = {
        "l22.experts": (256, 1408, 2048),
        "attn.o_proj": (2048, 2048),
    }
    provenance = gate.gate_cb_export_units(
        assignment={"l22.experts": "FP8_CB_K28", "attn.o_proj": "NVFP4_CB_K16"},
        quantized_targets=shapes,
        routed_units={"l22.experts"},
        role_split_units={"l22.experts"},
        shape_of=shapes.__getitem__,
    )
    seen = {row["qname"]: row for row in provenance["seen"]}
    assert seen["l22.experts"]["structure"] == "routed_moe"
    assert seen["l22.experts"]["role_split"] is True
    assert seen["l22.experts"]["in_features"] == 2048
    assert seen["l22.experts"]["out_features"] == 1408
    assert seen["attn.o_proj"]["structure"] == "dense"
    assert seen["attn.o_proj"]["out_features"] == 2048


def test_both_cb_exporters_run_the_gate_before_writing_bytes():
    """A gate that runs after the bytes exist is a log, not a gate."""
    import inspect

    from prismaquant import export_nvfp4_cb, export_nvfp4_cb_streaming

    # Scoped to the EXPORTER FUNCTION's body, not the module: a helper's `def`
    # can sit anywhere in the file, so a module-level index compares the wrong
    # two things.
    for fn, writer in (
        (export_nvfp4_cb.export_nvfp4_cb, "_write_cb_containers("),
        (export_nvfp4_cb_streaming.export_nvfp4_cb_streaming, "_StreamWriter("),
    ):
        src = inspect.getsource(fn)
        gate_at = src.index("gate_cb_export_units(")
        write_at = src.index(writer)
        assert gate_at < write_at, (
            f"{fn.__qualname__}: the route gate must precede {writer}")
        # And both must accept the two stamped dispositions.
        assert "allow_unbacked_route" in src
        assert "non_native_target" in src
        # The census must reach the artifact, not just stderr.
        assert "cb_route_status" in src


# ---------------------------------------------------------------------------
# 6. Lane-eligibility v3: rungs_q256, platforms, scope, and what is REFUSED
#
# Every test in this section is a mutation of a table that otherwise parses.
# A bad input proves the check runs; the point is that each mutation BITES.
# ---------------------------------------------------------------------------
def test_the_published_v12_table_parses_and_keeps_its_distinctions(tmp_path):
    """The fixture is Gridbook's own table, byte-verbatim, read end to end."""
    import hashlib

    assert hashlib.sha256(
        V12_FIXTURE.read_bytes()).hexdigest() == V12_FIXTURE_SHA256

    table = load_eligibility_table("0.9.1-fixture", contract_path=V12_FIXTURE)
    assert table.present
    assert table.schema == LANE_ELIGIBILITY_SCHEMA
    assert table.platforms == ("sm_89", "sm_120", "sm_121")
    assert table.regimes == ("decode", "batch")
    assert table.trellis_families == {"TCQ_E2M1_R256", "TCQ_E4M3_R256"}
    assert len(table.cells) == 16

    trellis = [c for c in table.cells if c.is_trellis]
    assert len(trellis) == 4
    for cell in trellis:
        # backed_with_serve_flag, never flattened to backed, and every one
        # names the flags an operator needs.
        assert cell.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
        assert cell.requires_serve_flags
        assert cell.qualification == "device_qualified"
        assert cell.activation_contract
        assert cell.rungs_q256 and not cell.rungs
    for cell in table.cells:
        if not cell.is_trellis:
            assert cell.rungs and not cell.rungs_q256
            assert cell.activation_contract == ""
        # The publisher has no way to spell an outright refusal.
        assert cell.route_status in CELL_ROUTE_STATUSES
        assert cell.route_status != ROUTE_STATUS_UNBACKED


def test_the_published_table_backs_the_trellis_lane_on_sm121():
    """The whole point of the rewrite: trellis becomes servable in our eyes."""
    table = load_eligibility_table("0.9.1-fixture", contract_path=V12_FIXTURE)
    unit = _v12_facts("model.layers.0.mlp.down_proj", "TCQ_E2M1_R512",
                      routed=False)
    assert unit.payload_family == "TCQ_E2M1_R256"
    assert unit.rate_q256 == 512
    assert unit.k is None

    route = resolve_unit_route(unit, table, platform="sm_121")
    assert route.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    assert route.requires_serve_flags == tuple(sorted(_E2M1_FLAGS))
    assert route.activation_contracts == ("e2m1_group16_ue4m3_static",)
    assert route.qualifications == ("device_qualified",)

    verdict = evaluate_cb_route_status([unit], table=table,
                                       target_platform="sm_121")
    assert not verdict.refused
    assert verdict.provenance["units_backed"] == 0
    assert verdict.provenance["units_backed_with_serve_flag"] == 1
    assert verdict.provenance["activation_contracts"] == {
        "e2m1_group16_ue4m3_static": 2}
    assert verdict.provenance["qualifications"] == {"device_qualified": 2}
    assert any("flag" in w for w in verdict.warnings)


def test_a_trellis_rate_no_cell_lists_resolves_unattested():
    """BITE, and the one the whole table exists for.

    640 q256 is a rate the E2M1 family PUBLISHES as a candidate rung and the
    reader accepts, so the unit's facts carry it. No lane cell lists it. A
    parser that admitted it would be strictly worse than one that could not
    parse the table at all, because it would report a route the runtime never
    claimed.
    """
    table = load_eligibility_table("0.9.1-fixture", contract_path=V12_FIXTURE)
    listed = _v12_facts("u", "TCQ_E2M1_R512", routed=False)
    unlisted = _v12_facts("u", "TCQ_E2M1_R640", routed=False)

    # Same family, same platform, same structure -- ONLY the rate differs.
    assert unlisted.payload_family == listed.payload_family
    assert unlisted.rate_q256 == 640

    assert resolve_unit_route(
        listed, table, platform="sm_121"
    ).route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    route = resolve_unit_route(unlisted, table, platform="sm_121")
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    assert route.in_scope is True
    assert route.unattested_regimes == ("decode", "batch")

    verdict = evaluate_cb_route_status([unlisted], table=table,
                                       target_platform="sm_121")
    assert verdict.refused
    assert verdict.provenance["units_unattested_in_scope"] == 1
    assert "rate_q256=640" in verdict.refusal_reason


def test_a_rate_outside_the_published_reader_range_carries_no_rate():
    """Fail closed one step earlier: an out-of-range rate is not even a fact."""
    over = _v12_facts("u", "TCQ_E2M1_R2000", routed=False)
    assert over.payload_family == "TCQ_E2M1_R256"
    assert over.rate_q256 is None, (
        "2000 q256 is above the E2M1 reader range [256, 1016]; carrying it as "
        "a fact would let a predicate or a rung list match on it")
    table = load_eligibility_table("0.9.1-fixture", contract_path=V12_FIXTURE)
    assert resolve_unit_route(
        over, table, platform="sm_121"
    ).route_status == ROUTE_STATUS_UNATTESTED


def test_a_cb_rung_no_cell_lists_resolves_unattested(attested_table):
    """The CB twin: K32 is a legal codec rung that no lane cell covers."""
    unit = _v12_facts("l0.experts", "FP8_CB_K32")
    assert unit.k == 32, "K32 IS published in formats[].rungs"
    assert 32 not in _FP8_RUNGS, "but no lane cell lists it"

    route = resolve_unit_route(unit, attested_table, platform=TEST_PLATFORM)
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert verdict.refused


def test_a_trellis_cell_must_use_rungs_q256_not_rungs(tmp_path):
    """BITE: the rung vocabulary follows the family's kind, not the cell.

    Body bits per 256 weights and a codebook K are different quantities on the
    same axis. Letting a trellis cell spell its rate as ``rungs`` would make
    ``512`` collide with a codebook size the moment either list grew.
    """
    block = _table_payload()
    cell = block["cells"][_TRELLIS_CELL]
    cell["rungs"] = cell.pop("rungs_q256")
    with pytest.raises(GridbookLaneEligibilityError) as exc:
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))
    assert "rungs_q256" in str(exc.value)


def test_a_cb_cell_may_not_carry_a_trellis_rate(tmp_path):
    block = _table_payload()
    block["cells"][0]["rungs_q256"] = [512]
    with pytest.raises(GridbookLaneEligibilityError, match="unknown field"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_trellis_cell_must_name_its_activation_contract(tmp_path):
    """A route with no executed contract attests nothing (principle 14)."""
    block = _table_payload()
    block["cells"][_TRELLIS_CELL]["activation_contract"] = ""
    with pytest.raises(GridbookLaneEligibilityError,
                       match="activation_contract"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))

    block = _table_payload()
    del block["cells"][_TRELLIS_CELL]["activation_contract"]
    with pytest.raises(GridbookLaneEligibilityError, match="missing field"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_an_empty_rung_list_is_refused(tmp_path):
    """An empty list covers nothing and would silently disable its cell."""
    block = _table_payload()
    block["cells"][0]["rungs"] = []
    with pytest.raises(GridbookLaneEligibilityError, match="at least one rung"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_cell_claiming_unbacked_is_refused(tmp_path):
    """The publisher has no such status; accepting one would out-lax it.

    Gridbook's own validator admits ``backed | backed_with_serve_flag |
    fallback``. A table handed to us with an ``unbacked`` cell is not a table
    that runtime produced.
    """
    block = _table_payload()
    block["cells"][0]["route_status"] = ROUTE_STATUS_UNBACKED
    with pytest.raises(GridbookLaneEligibilityError) as exc:
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))
    assert "absence" in str(exc.value)


def test_an_unknown_qualification_is_refused(tmp_path):
    block = _table_payload()
    block["cells"][0]["qualification"] = "probably_fine"
    with pytest.raises(GridbookLaneEligibilityError, match="qualification"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_compile_only_is_recorded_and_never_upgraded(attested_table):
    """``compile_only`` means the kernels build. It is not a serve."""
    verdict = evaluate_cb_route_status(
        [_facts("l0.experts", "FP8_CB_K28")], table=attested_table,
        target_platform=TEST_PLATFORM)
    assert verdict.provenance["qualifications"] == {"compile_only": 2}
    assert any("COMPILE_ONLY" in w for w in verdict.warnings)


def test_a_cell_on_an_undeclared_platform_is_refused(tmp_path):
    block = _table_payload()
    block["cells"][0]["platform"] = "sm_75"
    with pytest.raises(GridbookLaneEligibilityError, match="not a declared"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_a_cell_for_an_unpublished_family_is_refused(tmp_path):
    """A lane cell for a codec the runtime does not publish routes nothing."""
    block = _table_payload()
    block["cells"][0]["family"] = "INT3_CB_K"
    with pytest.raises(GridbookLaneEligibilityError, match="formats"):
        load_eligibility_table(
            "x", contract_path=_contract_with(block, tmp_path))


def test_no_declared_platform_attests_nothing(attested_table):
    """BITE: v3 cells are platform-scoped, so no platform is no claim.

    Resolving without one must NOT fall through to a match-any. This is the
    landmine for whoever bumps the pin: a CB serving profile that declares no
    ``target_platform`` will refuse, and the reason says so in as many words.
    """
    unit = _facts("l0.experts", "FP8_CB_K28")
    route = resolve_unit_route(unit, attested_table, platform=None)
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    assert "no declared target platform" in route.unattested_reason
    assert "platform-scoped" in route.unattested_reason

    verdict = evaluate_cb_route_status([unit], table=attested_table,
                                       target_platform=None,
                                       target_profile="__no_such_profile__")
    assert verdict.refused
    assert verdict.provenance["target_platform"] is None
    assert "UNDECLARED" in verdict.refusal_reason


def test_an_unpublished_platform_attests_nothing(attested_table):
    """sm_89 is not in this table; a unit targeted at it gets no claim."""
    unit = _facts("l0.experts", "FP8_CB_K28")
    route = resolve_unit_route(unit, attested_table, platform="sm_89")
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    assert "sm_89" in route.unattested_reason


def test_the_published_table_makes_no_cb_claim_on_sm121():
    """A measured fact about the published table, not an opinion about it.

    Gridbook's v12 table publishes CB cells for sm_89 and sm_120 only. On
    sm_121 -- the GB10 the flagships are built for -- a CB unit resolves
    UNATTESTED and the gate refuses. That is the table reporting a serving
    gap, which is exactly the signal it exists to carry (principle 1).
    """
    table = load_eligibility_table("0.9.1-fixture", contract_path=V12_FIXTURE)
    assert not any(c.platform == "sm_121" and not c.is_trellis
                   for c in table.cells)
    unit = _v12_facts("l0.experts", "FP8_CB_K48")
    assert resolve_unit_route(
        unit, table, platform="sm_121").route_status == ROUTE_STATUS_UNATTESTED
    assert resolve_unit_route(
        unit, table, platform="sm_120").route_status == ROUTE_STATUS_BACKED


def test_partial_regime_coverage_is_unattested_not_backed(tmp_path):
    """One covered regime is not coverage; it must not read as backed."""
    block = _table_payload()
    block["cells"] = [c for c in block["cells"]
                      if c["id"] != "cb_moe_expand_bridge"
                      and c["id"] != "cb_moe_persistent_b"]
    table = load_eligibility_table(
        "x", contract_path=_contract_with(block, tmp_path))
    route = resolve_unit_route(
        _facts("l0.experts", "FP8_CB_K28"), table, platform=TEST_PLATFORM)
    assert route.route_status == ROUTE_STATUS_UNATTESTED
    assert route.unattested_regimes == ("batch",)
    assert [r.route_status for r in route.regimes] == [
        ROUTE_STATUS_BACKED, ROUTE_STATUS_UNATTESTED]


def test_units_outside_the_published_families_are_reported_not_refused(
        attested_table):
    """The scope test is DERIVED from formats[], never a list typed here.

    BF16, a SOURCE passthrough and a stock CT rung all reach this gate from
    ``export_nvfp4_cb`` (it passes ``(*cb_targets, *stock_targets)``). The lane
    table publishes no codec for them, so it is not the authority for those
    bytes: they are counted and named, and they never read as backed.
    """
    units = [
        _facts("l0.experts", "FP8_CB_K28"),
        _facts("attn.q_proj", "BF16", routed=False, in_features=2048,
               out_features=2048),
        _facts("attn.k_proj", "FP8_SOURCE", routed=False, in_features=2048,
               out_features=512),
    ]
    for unit in units[1:]:
        route = resolve_unit_route(unit, attested_table,
                                   platform=TEST_PLATFORM)
        assert route.route_status == ROUTE_STATUS_UNATTESTED
        assert route.in_scope is False
        assert "not published in" in route.unattested_reason

    verdict = evaluate_cb_route_status(units, table=attested_table,
                                       target_platform=TEST_PLATFORM)
    assert not verdict.refused, (
        "the lane table makes no claim about non-CB bytes; refusing on them "
        "would refuse every real CB export")
    p = verdict.provenance
    assert p["units_total"] == 3
    assert p["units_in_attested_families"] == 1
    assert p["units_outside_attested_families"] == 2
    assert p["outside_attested_families_units"] == ["attn.k_proj",
                                                    "attn.q_proj"]
    assert p["outside_attested_families_formats"] == ["BF16", "FP8_SOURCE"]
    # And they are NOT counted as backed anywhere.
    assert p["units_backed"] == 1
    assert sum(p["units_by_route_status"].values()) == 1
    assert any("out of the attestation's scope" in w for w in verdict.warnings)
    assert shipcard_route_summary(p)["units_outside_attested_families"] == 2


def test_facts_cannot_carry_both_a_codebook_rung_and_a_trellis_rate():
    """The two rung vocabularies are exclusive by construction."""
    with pytest.raises(GridbookLaneEligibilityError, match="never both"):
        UnitStructuralFacts(
            qname="x", format_name="FP8_CB_K28", payload_family="FP8_CB_K",
            k=28, n_sub=4, structure="dense", role_split=False,
            in_features=256, out_features=16, rate_q256=512)


def test_unit_facts_reject_an_unknown_structure():
    with pytest.raises(Exception):
        UnitStructuralFacts(
            qname="x", format_name="FP8_CB_K28", payload_family="FP8_CB_K",
            k=28, n_sub=4, structure="sparse_moe", role_split=False,
            in_features=256, out_features=16)
