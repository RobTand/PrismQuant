"""A legacy family claim cannot attest a newly supplied runtime context.

The v4 table here is a FIXTURE, down-converted from the installed contract by
``conftest.legacy_v4_contract``. Until 2026-09-04 this file read the installed
contract directly and asserted it was v4 -- true when it was written, false the
day Tessera shipped ``tessera.lane-eligibility.v6``. A test about a legacy
grammar has to own its legacy table; reading whichever grammar happens to be
installed is how a fixture becomes a version assertion nobody meant to make.
"""
import json

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as render
from prismaquant import tessera_runtime_contract as contract


NAME = "TESSERA_E4M3_K1_R1024"
FAMILY = "TESSERA_E4M3_K1"


@pytest.fixture
def legacy(monkeypatch, legacy_v4_contract):
    payload = legacy_v4_contract
    assert payload["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v4"
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    table = lane._parse_table(payload["lane_eligibility"], payload["formats"],
                              "fixture", "fixture", "fixture",
                              native_extensions=payload["native_extensions"])
    formats = {row["family"]: row for row in payload["formats"]}
    monkeypatch.setattr(render, "_pinned_serving_table", lambda: (table, formats))
    monkeypatch.setattr(render, "_release_pin_satisfied", lambda: True)
    context = lane.ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=payload["versions"]["default_serve_image"],
        execution_mode="eager")
    return parsed, table, context


def _facts():
    return lane.UnitStructuralFacts(
        qname="fixture.weight", format_name=NAME, payload_family=FAMILY,
        k=None, n_sub=None, rate_q256=1024, structure="dense", role_split=False,
        in_features=1024, out_features=1024)


def test_legacy_development_lookup_keeps_its_context_free_behavior(legacy):
    parsed, _, _ = legacy
    assert parsed.native_cells(FAMILY, 1024)


def test_legacy_development_cannot_attest_even_a_global_image_context(legacy):
    parsed, _, context = legacy
    assert parsed.native_cells(FAMILY, 1024, serving_context=context) == ()


def test_legacy_released_lookup_keeps_its_context_free_behavior(legacy):
    assert render.tessera_lane_attested(NAME)


def test_legacy_renderer_cannot_attach_unreviewed_scope_to_backed_route(legacy):
    _, _, context = legacy
    assert render.tessera_attesting_cells(NAME, serving_context=context) == ()
    admitted, reason = render.tessera_lane_admission(NAME, serving_context=context)
    assert not admitted
    assert "runtime scope" in reason
    assert not render.tessera_lane_attested(NAME, serving_context=context)


@pytest.mark.parametrize("axes", ["image", "execution", "both"])
def test_legacy_generic_resolver_refuses_new_runtime_axes(legacy, axes):
    _, table, context = legacy
    extra = {}
    if axes in {"image", "both"}:
        extra["runtime_image"] = context.runtime_image
    if axes in {"execution", "both"}:
        extra["execution_mode"] = context.execution_mode
    result = lane.resolve_unit_route(_facts(), table, platform=context.platform,
                                    residency=context.residency, **extra)
    assert result.route_status == lane.ROUTE_STATUS_UNATTESTED
    assert "runtime scope" in result.unattested_reason


def test_legacy_generic_residency_only_lookup_keeps_v4_behavior(legacy):
    _, table, context = legacy
    result = lane.resolve_unit_route(_facts(), table, platform=context.platform,
                                    residency=context.residency)
    assert result.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG
