"""Explicit serving scope survives menu lookup, provenance and campaign reuse."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from prismaquant import serving_profiles as profiles
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_menu as menu


@dataclass(frozen=True)
class _Context:
    platform: str = "sm_121"
    structure: str = "dense"
    residency: str = "resident"
    runtime_image: str = "fixture/runtime@sha256:" + "1" * 64
    execution_mode: str = "eager"

    def key(self):
        return tuple(asdict(self).values())

    def as_dict(self):
        return asdict(self)


def _contract(monkeypatch, expected):
    observed = []
    cell = SimpleNamespace(
        cell_id="fixture_cell", route_status="backed_with_serve_flag",
        activation_contract="fp8_per_token_dynamic",
        requires_serve_flags=("TESSERA_SERVE_MODE=resident",),
    )

    def native_cells(family, rate, *, serving_context=None):
        observed.append(serving_context)
        return (cell,) if serving_context == expected else ()

    contract = SimpleNamespace(
        commit="fixture", requires_serving_context=True,
        max_world_size={"TESSERA_E4M3_K1": 1},
        attested_rungs={"TESSERA_E4M3_K1": (1024,)},
        governs=lambda family: family == "TESSERA_E4M3_K1",
        native_cells=native_cells,
    )
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: contract)
    return observed


def test_menu_admission_passes_context_to_the_contract_and_records_it(monkeypatch):
    context = _Context()
    observed = _contract(monkeypatch, context)
    admitted = menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert admitted.attested
    assert observed == [context]
    assert admitted.serving_context == context


@pytest.mark.parametrize("context", [None, _Context(structure="routed_moe")])
def test_v5_menu_admission_cannot_borrow_another_context(monkeypatch, context):
    _contract(monkeypatch, _Context())
    admitted = menu.route_admission("TESSERA_E4M3_K1_R1024", serving_context=context)
    assert not admitted.attested
    assert admitted.requires_serving_context
    assert "context" in admitted.detail


def test_menu_expansion_forwards_explicit_scope(monkeypatch):
    from prismaquant import tessera_footprint

    context = _Context()
    calls = []
    admission = SimpleNamespace(admits=lambda mode: True, act_bits=8)
    family = SimpleNamespace(
        mathematical_q256_bounds=(1024, 1024), name="TESSERA_E4M3_K1",
        base="E4M3", arity=1, format_name=lambda rate: f"TESSERA_E4M3_K1_R{rate}",
    )

    def admit(name, *, serving_context=None):
        calls.append(serving_context)
        return admission

    monkeypatch.setattr(menu, "route_admission", admit)
    monkeypatch.setattr(menu, "tessera_tp_legal", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(menu, "tessera_wire_recipe", lambda *args: None)
    monkeypatch.setattr(tessera_footprint, "tessera_exact_bits_for_shape",
                        lambda *args, **kwargs: Fraction(128))
    rows = menu.expand_tessera_menu((4, 4), families=(family,), serving_context=context)
    assert len(rows) == 1 and rows[0].admission is admission
    assert calls == [context]


def test_resolved_lane_identity_and_serialization_retain_scope(monkeypatch):
    calls = []

    def admit(name, *, serving_context=None):
        calls.append(serving_context)
        return SimpleNamespace(
            activation_contract="fp8_per_token_dynamic", source="fixture",
            detail="fixture", route_status="backed_with_serve_flag",
            requires_serve_flags=("TESSERA_SERVE_MODE=resident",),
            serving_context=serving_context,
        )

    monkeypatch.setattr(menu, "route_admission", admit)
    first = _Context()
    second = replace(first, execution_mode="compiled")
    a = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=first)
    b = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=second)
    assert calls == [first, second]
    assert a.as_dict()["serving_context"] == first.as_dict()
    assert b.as_dict()["serving_context"] == second.as_dict()
    assert a.route_key() != b.route_key()


def test_profile_tessera_fallback_forwards_explicit_scope(monkeypatch):
    context = _Context()
    calls = []
    sentinel = object()
    monkeypatch.setattr(profiles, "load_serving_profile", lambda profile: SimpleNamespace(
        serving_lane_for=lambda *args, **kwargs: None))

    def resolve(name, *, runtime_version="", serving_context=None):
        calls.append(serving_context)
        return sentinel

    monkeypatch.setattr(menu, "tessera_resolved_serving_lane", resolve)
    assert profiles.serving_lane_route(
        "fixture", "TESSERA_E4M3_K1_R1024", serving_context=context) is sentinel
    assert calls == [context]


@pytest.mark.parametrize("different", [
    {"structure": "routed_moe"}, {"residency": "streamed"},
    {"runtime_image": "fixture/runtime@sha256:" + "2" * 64},
    {"execution_mode": "compiled"}, {"platform": "sm_90"},
])
def test_campaign_menu_cache_separates_scope_and_reuses_equal_scope(monkeypatch, different):
    first = _Context()
    second = replace(first, **different)
    calls = []

    def expand(shape, **kwargs):
        context = kwargs.get("serving_context")
        calls.append(context)
        return [context]

    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    weights = {name: SimpleNamespace(shape=(32, 32)) for name in ("a", "b", "c")}
    result = campaign.expand_menus_for_targets(
        weights, list(weights), mode=menu.MENU_ATTESTED, tp_degree=1,
        parallel_kind=menu.PARALLEL_NONE,
        context_by_unit={"a": first, "b": second, "c": replace(first)})
    assert calls == [first, second]
    assert result["a"] is result["c"]
    assert result["a"] is not result["b"]


def test_campaign_partial_context_map_keeps_missing_scope_unbound(monkeypatch):
    first = _Context()
    calls = []

    def expand(shape, **kwargs):
        context = kwargs.get("serving_context")
        calls.append(context)
        return [context]

    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    weights = {name: SimpleNamespace(shape=(32, 32)) for name in ("a", "b")}
    result = campaign.expand_menus_for_targets(
        weights, list(weights), mode=menu.MENU_ATTESTED, tp_degree=1,
        parallel_kind=menu.PARALLEL_NONE, context_by_unit={"a": first})
    assert calls == [first, None]
    assert result["b"] == [None]


def _legacy_contract(monkeypatch):
    # A legacy supplier can answer only family/rate, even when its Python
    # method accepts the new keyword. It supplies no runtime-scope evidence.
    cell = SimpleNamespace(
        cell_id="legacy_cell", route_status="backed_with_serve_flag",
        activation_contract="fp8_per_token_dynamic",
        requires_serve_flags=("TESSERA_SERVE_MODE=resident",),
    )
    contract = SimpleNamespace(
        commit="legacy-fixture", lane_schema="tessera.lane-eligibility.v4",
        requires_serving_context=False,
        max_world_size={"TESSERA_E4M3_K1": 1},
        attested_rungs={"TESSERA_E4M3_K1": (1024,)},
        governs=lambda family: family == "TESSERA_E4M3_K1",
        native_cells=lambda *args, **kwargs: (cell,),
    )
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: contract)


@pytest.mark.parametrize("context", [
    None, _Context(), _Context(execution_mode="compiled"),
    _Context(structure="routed_moe"),
])
def test_legacy_menu_cannot_attest_a_new_runtime_context(monkeypatch, context):
    _legacy_contract(monkeypatch)

    admission = menu.route_admission(
        "TESSERA_E4M3_K1_R1024", serving_context=context)

    assert admission.attested is (context is None)
    assert admission.admits(menu.MENU_ATTESTED) is (context is None)
    assert admission.admits(menu.MENU_RESEARCH)
    assert admission.serving_context == context
    assert not admission.requires_serving_context
    if context is not None:
        assert "context" in admission.detail
        assert "v4" in admission.detail
        assert admission.requires_serve_flags == ()


def test_legacy_context_refusal_survives_resolved_route_provenance(monkeypatch):
    _legacy_contract(monkeypatch)
    context = _Context(structure="routed_moe", execution_mode="compiled")

    resolved = menu.tessera_resolved_serving_lane(
        "TESSERA_E4M3_K1_R1024", serving_context=context)

    assert resolved.route_status == "unattested"
    assert resolved.as_dict()["route_status"] == "unattested"
    assert resolved.as_dict()["serving_context"] == context.as_dict()
    assert resolved.requires_serve_flags == ()
