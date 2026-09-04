"""A format name does not identify the serving route of every model unit."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

import prismaquant.allocator_candidates as ac
from prismaquant import format_registry as fr
from prismaquant import tessera_menu as tm
from prismaquant.allocator_solver import Candidate
from prismaquant.serving_profiles import ResolvedServingLane


_FORMAT = "TESSERA_E2M1_K2_R896"
_PROFILE = "tessera_research_sm121"


@dataclass(frozen=True)
class _Context:
    """The public context protocol, without requiring the new class to import."""

    platform: str = "sm_121"
    structure: str = "dense"
    residency: str = "resident"
    runtime_image: str = "example/vllm@sha256:" + "a" * 64
    execution_mode: str = "eager"

    def key(self) -> tuple[str, ...]:
        return (
            self.platform,
            self.structure,
            self.residency,
            self.runtime_image,
            self.execution_mode,
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _contexts():
    # Equal contexts are separate objects; caching by object identity would
    # make a full model repeat its contract lookup for every Linear.
    return {
        "unit.a": _Context(),
        "unit.b": _Context(),
        "unit.c": _Context(structure="routed_moe"),
    }


def _lane(structure: str) -> ResolvedServingLane:
    return ResolvedServingLane(
        lane_id=f"tessera-{structure}",
        format=_FORMAT,
        activation_contract="W4A4",
        fallback_route=f"native-{structure}",
        fused_mid_m_backed=False,
        fused_mid_m_rungs=(),
        fused_mid_m_range=None,
        runtime_version="test-runtime",
        rungs_source="test-contract",
        route_status="backed",
        route_status_source=f"test-contract:{structure}",
    )


def _record_resolver(monkeypatch):
    calls = []
    lanes = {structure: _lane(structure) for structure in ("dense", "routed_moe")}

    def resolve(profile, fmt, *, serving_context=None, **kwargs):
        assert profile == _PROFILE
        assert fmt == _FORMAT
        assert serving_context is not None, "unit serving context was discarded"
        calls.append(serving_context.key())
        return lanes[serving_context.structure]

    monkeypatch.setattr(ac, "serving_lane_route", resolve)
    return calls, lanes


def _tables(contexts):
    # The same rank-two shape belongs to both dense and expert units. Shape
    # alone cannot supply the structure axis of the attestation query.
    stats = {
        name: {
            "h_trace": 1.0,
            "n_params": 256 * 256,
            "in_features": 256,
            "out_features": 256,
        }
        for name in contexts
    }
    costs = {
        name: {
            _FORMAT: {
                "weight_mse": 1e-4,
                "output_mse": 4e-4,
                "output_mse_measured": True,
            }
        }
        for name in contexts
    }
    return stats, costs


def _fixture_specs(monkeypatch, *names):
    # Build the real specs before admission is mocked. Cost validation asks
    # the registry for them again; re-synthesizing a Tessera spec would make
    # the cache spy count factory admission calls rather than candidate calls.
    original_get_format = fr.get_format
    specs = [original_get_format(name) for name in names]
    by_name = {spec.name: spec for spec in specs}

    def get_format(name):
        canonical = fr.canonical_format_name(name)
        if canonical in by_name:
            return by_name[canonical]
        return original_get_format(name)

    monkeypatch.setattr(fr, "get_format", get_format)
    return specs


def test_candidates_resolve_same_format_separately_for_dense_and_routed_units(monkeypatch):
    spec, = _fixture_specs(monkeypatch, _FORMAT)
    contexts = _contexts()
    stats, costs = _tables(contexts)
    calls, lanes = _record_resolver(monkeypatch)
    monkeypatch.setattr(
        tm, "route_admission",
        lambda *args, **kwargs: SimpleNamespace(requires_serving_context=False),
    )
    # This test isolates route threading; shape legality and wire accounting
    # have their own proof harnesses and are not facts inferred by this mock.
    monkeypatch.setattr(
        ac, "check_stats_format_applicability",
        lambda *args, **kwargs: ac.FormatApplicability(True),
    )
    monkeypatch.setattr(
        ac, "serialized_candidate_payload",
        lambda *args, **kwargs: (32768, None, None),
    )
    monkeypatch.setattr(
        ac, "reduce_continuous_menu",
        lambda candidates, *args, **kwargs: candidates,
    )

    candidates = ac.build_candidates(
        stats, costs, [spec],
        target_profile=_PROFILE,
        context_by_unit=contexts,
    )

    assert set(candidates) == set(contexts)
    for name, context in contexts.items():
        assert len(candidates[name]) == 1
        assert candidates[name][0].serving_lane == lanes[context.structure]
        assert stats[name]["_serving_lane_by_format"][_FORMAT] == lanes[context.structure]
    assert Counter(calls) == Counter({context.key(): 1 for context in contexts.values()})


def test_expanded_selection_resolves_and_caches_routes_by_explicit_unit_context(monkeypatch):
    contexts = _contexts()
    calls, lanes = _record_resolver(monkeypatch)
    assignment = dict.fromkeys(contexts, _FORMAT)

    report = ac.selection_serving_lane_provenance(
        assignment, candidates=None, target_profile=_PROFILE,
        context_by_unit=contexts,
    )

    assert set(report["by_unit"]) == set(assignment)
    for name, context in contexts.items():
        row = report["by_unit"][name]
        assert row["format"] == _FORMAT
        assert row["route"] == lanes[context.structure].as_dict()
    assert Counter(calls) == Counter({context.key(): 1 for context in contexts.values()})
    assert report["by_format"][_FORMAT]["units"] == len(assignment)
    assert report["by_format"][_FORMAT]["route"] is None
    assert report["units_without_declared_lane"] == 0
    assert report["route_status_counts"] == {"backed": len(assignment)}


def test_selected_candidate_routes_survive_conflicting_same_format_provenance(monkeypatch):
    contexts = _contexts()
    assignment = dict.fromkeys(contexts, _FORMAT)
    lanes = {name: _lane(context.structure) for name, context in contexts.items()}
    candidates = {
        name: [Candidate(
            fmt=_FORMAT,
            bits_per_param=4.0,
            memory_bytes=32768,
            predicted_dloss=2e-4,
            activation_pricing="measured_output_mse",
            serving_lane=lanes[name],
        )]
        for name in assignment
    }

    def reject_reresolution(*args, **kwargs):
        raise AssertionError("the selected candidate already carries its priced route")

    monkeypatch.setattr(ac, "serving_lane_route", reject_reresolution)
    report = ac.selection_serving_lane_provenance(
        assignment, candidates=candidates, target_profile=_PROFILE,
        context_by_unit=contexts,
    )

    assert set(report["by_unit"]) == set(assignment)
    for name in assignment:
        assert report["by_unit"][name]["format"] == _FORMAT
        assert report["by_unit"][name]["route"] == lanes[name].as_dict()
    # A single summary route would falsely attest every selected unit against
    # whichever context sorted first. The individual routes remain available.
    assert report["by_format"][_FORMAT]["route"] is None
    assert report["by_format"][_FORMAT]["units"] == len(assignment)
    assert report["activation_pricing_branches"] == {"measured_output_mse": len(assignment)}


@pytest.mark.parametrize("mode", [tm.MENU_ATTESTED, tm.MENU_RESEARCH])
@pytest.mark.parametrize("context", [None, _Context(structure="routed_moe")])
def test_v5_unmatched_context_masks_only_the_attested_candidate(monkeypatch, mode, context):
    specs = _fixture_specs(monkeypatch, _FORMAT, "BF16")
    contexts = {} if context is None else {"unit.a": context}
    stats, costs = _tables({"unit.a": context})
    costs["unit.a"]["BF16"] = {"weight_mse": 0.0, "predicted_dloss": 0.0}
    observed = []

    def admit(name, *, serving_context=None):
        assert name == _FORMAT
        observed.append(serving_context)
        return SimpleNamespace(
            requires_serving_context=True,
            attested=False,
            detail="context does not match any native cell",
            admits=lambda selected_mode: selected_mode == tm.MENU_RESEARCH,
        )

    monkeypatch.setattr(tm, "menu_mode", lambda: mode)
    monkeypatch.setattr(tm, "route_admission", admit)
    monkeypatch.setattr(
        ac, "serving_lane_route",
        lambda profile, fmt, **kwargs: None if fmt == "BF16" else _lane("unattested"),
    )
    monkeypatch.setattr(
        ac, "check_stats_format_applicability",
        lambda *args, **kwargs: ac.FormatApplicability(True),
    )
    monkeypatch.setattr(
        ac, "serialized_candidate_payload",
        lambda *args, **kwargs: (32768, None, None),
    )
    monkeypatch.setattr(
        ac, "reduce_continuous_menu",
        lambda candidates, *args, **kwargs: candidates,
    )
    masks = []

    candidates = ac.build_candidates(
        stats, costs, specs, target_profile=_PROFILE,
        context_by_unit=contexts, mask_records=masks,
    )

    assert observed == [context]
    expected_formats = {"BF16", _FORMAT} if mode == tm.MENU_RESEARCH else {"BF16"}
    assert {candidate.fmt for candidate in candidates["unit.a"]} == expected_formats
    if mode == tm.MENU_ATTESTED:
        assert len(masks) == 1
        assert masks[0]["qname"] == "unit.a"
        assert masks[0]["format"] == _FORMAT
        assert "context" in masks[0]["detail"]
    else:
        assert masks == []


def test_v5_scope_refusal_cannot_silently_remove_a_whole_unit(monkeypatch):
    specs = _fixture_specs(monkeypatch, _FORMAT)
    stats, costs = _tables({"unit.a": None})
    monkeypatch.setattr(tm, "menu_mode", lambda: tm.MENU_ATTESTED)
    monkeypatch.setattr(tm, "route_admission", lambda *args, **kwargs: SimpleNamespace(
        requires_serving_context=True,
        detail="no explicit serving context was supplied",
        admits=lambda mode: False,
    ))
    monkeypatch.setattr(
        ac, "check_stats_format_applicability",
        lambda *args, **kwargs: ac.FormatApplicability(True),
    )
    with pytest.raises(ValueError, match="unit.a"):
        ac.build_candidates(
            stats, costs, specs, target_profile=_PROFILE, context_by_unit={},
        )
