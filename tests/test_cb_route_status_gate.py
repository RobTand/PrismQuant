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
from prismaquant.cb_layout import (
    FP8_ACCEPTED_RUNGS,
    FP8_PRODUCT_RUNGS,
    NVFP4_ACCEPTED_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)
from prismaquant.gridbook_format_contract import (
    GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA,
)
from prismaquant.gridbook_lane_eligibility import (
    GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
    LANE_ELIGIBILITY_SCHEMA,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
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


# ---------------------------------------------------------------------------
# A synthetic table modelling the MEASURED Gridbook 0.8.10 behaviour.
#
# It is a FIXTURE, never a shipped attestation: writing these values into
# prismaquant/ would be exactly the transcription principle 14 forbids. It
# exists so the gate's logic can be tested before Gridbook packages the real
# thing, and the values mirror what was measured in gridbook 0.8.10 source
# (moe.py token-count predicate; persistent-B role-split refusal).
# ---------------------------------------------------------------------------
def _table_payload() -> dict:
    return {
        "schema": LANE_ELIGIBILITY_SCHEMA,
        "regimes": ["decode", "batch"],
        "lanes": [
            {
                "id": "cb_moe_gemv_decode",
                "regime": "decode",
                "structure": "routed_moe",
                "route_status": "backed",
                "detail": "per-role GEMVs below the token threshold",
                "predicates": [
                    {"fact": "payload_family", "op": "in",
                     "value": ["FP8_CB_K", "NVFP4_CB_K"]},
                    {"fact": "out_features", "op": "multiple_of", "value": 16},
                ],
            },
            {
                "id": "cb_moe_persistent_b",
                "regime": "batch",
                "structure": "routed_moe",
                "route_status": "backed",
                "detail": "decode-in-mainloop prefill; canonical books only",
                "predicates": [
                    {"fact": "payload_family", "op": "in",
                     "value": ["FP8_CB_K", "NVFP4_CB_K"]},
                    {"fact": "role_split", "op": "equals", "value": False},
                    {"fact": "k", "op": "multiple_of", "value": 4},
                ],
            },
            {
                "id": "cb_moe_expand_bridge",
                "regime": "batch",
                "structure": "routed_moe",
                "route_status": "fallback",
                "detail": "announced expand + grouped bridge; role-split books",
                "predicates": [
                    {"fact": "payload_family", "op": "in",
                     "value": ["FP8_CB_K", "NVFP4_CB_K"]},
                ],
            },
            {
                "id": "cb_dense_decode",
                "regime": "decode",
                "structure": "dense",
                "route_status": "backed",
                "predicates": [
                    {"fact": "in_features", "op": "multiple_of", "value": 256},
                ],
            },
            {
                "id": "cb_dense_midm_optin",
                "regime": "batch",
                "structure": "dense",
                "route_status": "backed_with_serve_flag",
                "requires_serve_flags": ["PRISMAQUANT_CB_FP4_FUSED_MIDM=1"],
                "predicates": [
                    {"fact": "in_features", "op": "multiple_of", "value": 256},
                ],
            },
        ],
    }


@pytest.fixture()
def attested_table(tmp_path: Path):
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.10.json").read_text())
    contract["lane_eligibility"] = _table_payload()
    path = tmp_path / "runtime_contract.json"
    path.write_text(json.dumps(contract))
    return load_eligibility_table("0.8.10-test", contract_path=path)


def _facts(qname, fmt, *, routed=True, role_split=False,
           in_features=2048, out_features=1408):
    return unit_structural_facts(
        qname, fmt,
        is_routed_moe=routed,
        role_split=role_split,
        in_features=in_features,
        out_features=out_features,
        published_formats=load_published_formats(),
    )


def _v2_contract(*, qualification="compile_only") -> dict:
    """Minimal self-contained v11/v2 contract for the export-gate tests."""
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.11.json").read_text())
    contract["schema"] = GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA
    contract["contract_version"] = 11
    for entry in contract["formats"]:
        if entry["family"] == "FP8_CB_K":
            entry["rungs"] = list(FP8_ACCEPTED_RUNGS)
            entry["producer_rungs"] = list(FP8_PRODUCT_RUNGS)
        elif entry["family"] == "NVFP4_CB_K":
            entry["rungs"] = list(NVFP4_ACCEPTED_RUNGS)
            entry["producer_rungs"] = list(NVFP4_PRODUCT_RUNGS)
        else:
            entry["producer_rungs"] = list(entry["rungs"])
    contract["lane_eligibility"] = {
        "schema": GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA,
        "platforms": {"sm_120": {"compute_capability": [12, 0]}},
        "regimes": ["decode", "batch"],
        "structures": ["dense", "routed_moe"],
        "cells": [
            {
                "id": f"fp8_dense_sm120_{regime}",
                "platform": "sm_120",
                "family": "FP8_CB_K",
                "structure": "dense",
                "regime": regime,
                "rungs": [40],
                "route_status": "backed",
                "qualification": qualification,
                "requires_serve_flags": [],
                "predicates": [],
            }
            for regime in ("decode", "batch")
        ],
    }
    return contract


def _write_v2_table(tmp_path: Path, *, qualification="compile_only",
                    mutate=None):
    contract = _v2_contract(qualification=qualification)
    if mutate is not None:
        mutate(contract)
    path = tmp_path / "runtime_contract.v11.json"
    path.write_text(json.dumps(contract))
    return load_eligibility_table("v11-test", contract_path=path)


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
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.10.json").read_text())
    block = _table_payload()
    block["lanes"][0]["predicates"].append(
        {"fact": "sm_capability", "op": "equals", "value": 121})
    contract["lane_eligibility"] = block
    path = tmp_path / "c.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(GridbookLaneEligibilityError, match="sm_capability"):
        load_eligibility_table("x", contract_path=path)


def test_a_flag_gated_route_must_name_its_flag(tmp_path):
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.10.json").read_text())
    block = _table_payload()
    block["lanes"][4]["requires_serve_flags"] = []
    contract["lane_eligibility"] = block
    path = tmp_path / "c.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(GridbookLaneEligibilityError, match="unnamed flag"):
        load_eligibility_table("x", contract_path=path)


def test_a_wrong_schema_is_refused(tmp_path):
    contract = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract.0.8.10.json").read_text())
    block = _table_payload()
    block["schema"] = "gridbook.lane-eligibility.v99"
    contract["lane_eligibility"] = block
    path = tmp_path / "c.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(GridbookLaneEligibilityError, match="schema"):
        load_eligibility_table("x", contract_path=path)


def test_v2_execution_cells_are_loaded_instead_of_misparsed_as_v1(tmp_path):
    table = _write_v2_table(tmp_path)
    assert table.present
    assert table.schema == GRIDBOOK_LANE_ELIGIBILITY_V2_SCHEMA
    assert table.rules == ()
    assert table.execution_contract is not None
    assert table.provenance()["platforms"] == ["sm_120"]
    assert table.provenance()["required_producer_qualification"] == (
        "device_qualified"
    )


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
        [_facts("l0.experts", "FP8_CB_K28")], table=attested_table)
    assert not verdict.refused
    assert verdict.provenance["units_backed"] == 1
    assert verdict.provenance["units_with_announced_fallback"] == 0
    assert verdict.provenance["units_unbacked"] == 0


def test_the_dsv4_per_role_case_is_recorded_not_refused(attested_table):
    """decode backed + batch announced fallback = the MEASURED DSv4 state."""
    unit = _facts("l22.experts", "FP8_CB_K28", role_split=True)
    route = resolve_unit_route(unit, attested_table)

    assert route.route_status == ROUTE_STATUS_BACKED
    assert route.fallback_regimes == ("batch",)
    by_regime = {r.regime: r for r in route.regimes}
    assert by_regime["decode"].rule_id == "cb_moe_gemv_decode"
    assert by_regime["batch"].rule_id == "cb_moe_expand_bridge"

    verdict = evaluate_cb_route_status([unit], table=attested_table)
    assert not verdict.refused, "an announced fallback SERVES; refusing is wrong"
    assert verdict.provenance["units_with_announced_fallback"] == 1
    assert verdict.provenance["announced_fallback_units"] == ["l22.experts"]
    assert verdict.provenance["by_regime"]["batch"]["fallback"] == 1
    assert any("ANNOUNCED FALLBACK" in w for w in verdict.warnings)


def test_a_unit_with_no_backed_route_fails_export_closed(attested_table):
    """out_features % 16 != 0 has no decode rule, so no regime backs it."""
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    verdict = evaluate_cb_route_status([unit], table=attested_table)
    assert verdict.refused
    assert verdict.provenance["units_unbacked"] == 1
    assert verdict.provenance["unbacked_disposition"] == "refused"
    assert "NO backed serving route" in verdict.refusal_reason
    with pytest.raises(CBRouteStatusRefusal):
        require_cb_route_status([unit], table=attested_table)


def test_an_explicit_override_ships_and_is_stamped(attested_table):
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    provenance = require_cb_route_status(
        [unit], table=attested_table,
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
        [unit], table=attested_table, non_native_target="sm90-a100-fallback")
    assert provenance["unbacked_disposition"] == "declared_non_native_target"
    assert provenance["declared_non_native_target"] == "sm90-a100-fallback"
    assert shipcard_route_summary(provenance)[
        "declared_non_native_target"] == "sm90-a100-fallback"


def test_a_serve_flag_route_is_backed_with_serve_flag_and_names_the_flag(
        attested_table):
    unit = _facts("attn.o_proj", "NVFP4_CB_K16", routed=False,
                  in_features=2048, out_features=2048)
    route = resolve_unit_route(unit, attested_table)
    assert route.route_status == ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
    verdict = evaluate_cb_route_status([unit], table=attested_table)
    assert verdict.provenance["requires_serve_flags"] == [
        "PRISMAQUANT_CB_FP4_FUSED_MIDM=1"]
    assert not verdict.refused


def test_env_supplies_the_override_when_the_caller_does_not(
        attested_table, monkeypatch):
    unit = _facts("l1.experts", "FP8_CB_K28", out_features=1410)
    monkeypatch.setenv(ROUTE_OVERRIDE_ENV, "one-off bringup")
    provenance = require_cb_route_status([unit], table=attested_table)
    assert provenance["override"]["reason"] == "one-off bringup"
    monkeypatch.delenv(ROUTE_OVERRIDE_ENV)
    monkeypatch.setenv(NON_NATIVE_TARGET_ENV, "gb10-eager")
    provenance = require_cb_route_status([unit], table=attested_table)
    assert provenance["declared_non_native_target"] == "gb10-eager"


def test_a_mixed_selection_counts_every_disposition(attested_table):
    units = [
        _facts("l0.experts", "FP8_CB_K28"),
        _facts("l22.experts", "FP8_CB_K28", role_split=True),
        _facts("l23.experts", "FP8_CB_K28", role_split=True),
        _facts("attn.o_proj", "NVFP4_CB_K16", routed=False,
               in_features=2048, out_features=2048),
    ]
    verdict = evaluate_cb_route_status(units, table=attested_table)
    assert not verdict.refused
    assert verdict.provenance["units_total"] == 4
    assert verdict.provenance["units_with_announced_fallback"] == 2
    assert verdict.provenance["units_backed"] == 3
    assert verdict.provenance["units_backed_with_serve_flag"] == 1


def test_v1_route_payload_shape_is_unchanged(attested_table):
    route = resolve_unit_route(
        _facts("l0.experts", "FP8_CB_K28"), attested_table
    )
    assert route.regimes
    for regime in route.as_dict()["regime_routes"]:
        assert "platform" not in regime
        assert "qualification" not in regime
    attestation = attested_table.provenance()
    assert "platforms" not in attestation
    assert "cell_ids" not in attestation
    assert "required_producer_qualification" not in attestation


def test_v2_compile_only_is_a_non_forceable_release_refusal(tmp_path):
    table = _write_v2_table(tmp_path, qualification="compile_only")
    verdict = evaluate_cb_route_status(
        [_facts("attn.o_proj", "FP8_CB_K40", routed=False)],
        table=table,
        target_profile="qwen38_sm120_cb_validation_only",
        override_reason="do not let this waive qualification",
        non_native_target="also-not-a-qualification-waiver",
    )
    assert verdict.refused
    assert verdict.provenance["target_platform"] == "sm_120"
    assert verdict.provenance["required_qualification"] == "device_qualified"
    assert verdict.provenance["units_compile_only"] == 1
    assert verdict.provenance["unbacked_disposition"] == (
        "compile_only_refused"
    )
    assert "cannot be waived" in verdict.refusal_reason
    for regime in verdict.provenance["by_unit"][0]["regime_routes"]:
        assert regime["platform"] == "sm_120"
        assert regime["qualification"] == "compile_only"


def test_v2_device_qualified_exact_profile_structure_regimes_and_rung_pass(
        tmp_path):
    table = _write_v2_table(tmp_path, qualification="device_qualified")
    verdict = evaluate_cb_route_status(
        [_facts("attn.o_proj", "FP8_CB_K40", routed=False)],
        table=table,
        target_profile="qwen38_sm120_cb_validation_only",
    )
    assert not verdict.refused
    assert verdict.provenance["units_backed"] == 1
    assert verdict.provenance["units_compile_only"] == 0
    assert verdict.provenance["by_regime"] == {
        "batch": {"backed": 1},
        "decode": {"backed": 1},
    }


@pytest.mark.parametrize(
    ("target_profile", "facts"),
    [
        (
            "qwen38_rtx4090_fp8_cb_validation_only",
            lambda: _facts("attn.o_proj", "FP8_CB_K40", routed=False),
        ),
        (
            "qwen38_sm120_cb_validation_only",
            lambda: _facts("l0.experts", "FP8_CB_K40", routed=True),
        ),
        (
            "qwen38_sm120_cb_validation_only",
            lambda: _facts("attn.o_proj", "FP8_CB_K44", routed=False),
        ),
        (
            "qwen38_sm120_cb_validation_only",
            lambda: _facts("attn.o_proj", "NVFP4_CB_K16", routed=False),
        ),
        (
            "nvfp4_cb",
            lambda: _facts("attn.o_proj", "FP8_CB_K40", routed=False),
        ),
    ],
    ids=[
        "exact-platform",
        "exact-structure",
        "exact-rung",
        "exact-family",
        "no-inference",
    ],
)
def test_v2_missing_exact_route_dimension_fails_closed(
        tmp_path, target_profile, facts):
    table = _write_v2_table(tmp_path, qualification="device_qualified")
    verdict = evaluate_cb_route_status(
        [facts()], table=table, target_profile=target_profile
    )
    assert verdict.refused
    assert verdict.provenance["units_unbacked"] == 1


def test_v2_requires_every_declared_regime(tmp_path):
    def remove_batch(contract):
        contract["lane_eligibility"]["cells"] = [
            cell for cell in contract["lane_eligibility"]["cells"]
            if cell["regime"] != "batch"
        ]

    table = _write_v2_table(
        tmp_path,
        qualification="device_qualified",
        mutate=remove_batch,
    )
    verdict = evaluate_cb_route_status(
        [_facts("attn.o_proj", "FP8_CB_K40", routed=False)],
        table=table,
        target_profile="qwen38_sm120_cb_validation_only",
    )
    assert verdict.refused
    assert verdict.provenance["by_regime"]["decode"] == {"backed": 1}
    assert verdict.provenance["by_regime"]["batch"] == {"unbacked": 1}


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
        table=attested_table,
        non_native_target="sm90-a100-fallback")
    root = _cb_artifact(tmp_path, {"cb_route_status": provenance})

    card = build_shipcard(root, build={"quant_method": "gridbook"})
    path = write_shipcard(root / "shipcard.json", card)
    on_disk = load_shipcard(path)["cb_route_status"]

    assert on_disk["unbacked_disposition"] == "declared_non_native_target"
    assert on_disk["declared_non_native_target"] == "sm90-a100-fallback"
    assert on_disk["units_unbacked"] == 1
    # The card names the release the claim came from; a route status with no
    # runtime identity attests nothing (principle 14).
    assert on_disk["gridbook_serving_version"] == "0.8.10-test"


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
        route = resolve_unit_route(_facts("u", fmt), table)
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
            provenance={
                "seen": [f.as_dict() for f in facts],
                "target_profile": kw["target_profile"],
            },
            refused=False,
        ))
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
        target_profile="qwen38_sm120_cb_validation_only",
    )
    seen = {row["qname"]: row for row in provenance["seen"]}
    assert seen["l22.experts"]["structure"] == "routed_moe"
    assert seen["l22.experts"]["role_split"] is True
    assert seen["l22.experts"]["in_features"] == 2048
    assert seen["l22.experts"]["out_features"] == 1408
    assert seen["attn.o_proj"]["structure"] == "dense"
    assert seen["attn.o_proj"]["out_features"] == 2048
    assert provenance["target_profile"] == "qwen38_sm120_cb_validation_only"


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
        assert "target_profile=route_target_profile" in src
        # The census must reach the artifact, not just stderr.
        assert "cb_route_status" in src


def test_unit_facts_reject_an_unknown_structure():
    with pytest.raises(Exception):
        UnitStructuralFacts(
            qname="x", format_name="FP8_CB_K28", payload_family="FP8_CB_K",
            k=28, n_sub=4, structure="sparse_moe", role_split=False,
            in_features=256, out_features=16)
