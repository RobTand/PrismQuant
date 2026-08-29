from __future__ import annotations

import json
import math

import pytest

from prismaquant.cb_layout import FP8_PRODUCT_RUNGS
from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.serving_profiles import gridbook_runtime_version
from prismaquant.source_class_format_plan import (
    NONEXPERT_MENU,
    _plan_digest,
    build_source_class_format_plan,
    load_format_plan,
    write_format_plan,
)


# The current producer family is K40/K44/K48. The planner schema still
# requires a strict lower-rate menu declaration, but no maintained source
# class maps to it: compatible FP8 sources take the complete nonexpert menu,
# while the old mxfp4/K32-ceiling class must fail closed.
NONEXPERT_FORMATS = tuple(f"FP8_CB_K{k}" for k in FP8_PRODUCT_RUNGS)
EXPERT_FORMATS = NONEXPERT_FORMATS[:-1]
ON_LAW_PROFILE = "nvfp4_cb"
ON_LAW_EXPERT_FORMATS = EXPERT_FORMATS
ON_LAW_NONEXPERT_FORMATS = NONEXPERT_FORMATS
CONTEXT = CBSerializationContext.production(codebook_source="lattice")


def _stats(shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "h_trace": 1.0,
        "n_params": math.prod(shape),
        "out_features": shape[-2],
        "in_features": shape[-1],
        **({"num_experts": shape[0]} if len(shape) == 3 else {}),
    }


class _Profile:
    def __init__(self, *, fused: bool = False, packed: bool = False):
        self.fused = fused
        self.packed = packed

    def fused_sibling_group(self, qname: str):
        return "layer.fused" if self.fused else None

    def packed_expert_format_group(self, qname: str):
        return "layer.experts" if self.packed else None


def test_source_derived_split_uses_only_current_producer_rungs():
    assert FP8_PRODUCT_RUNGS == (40, 44, 48)
    nonexpert = "model.layers.0.self_attn.o_proj"
    plan = build_source_class_format_plan(
        {nonexpert: _stats((8192, 4096))},
        {nonexpert: "fp8_ue8m0"},
        _Profile(),
        expert_formats=EXPERT_FORMATS,
        nonexpert_formats=NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
    )

    assert plan.menu_id_for(nonexpert) == NONEXPERT_MENU
    assert plan.formats_for(nonexpert) == NONEXPERT_FORMATS

    scheduled = {
        (qname, fmt)
        for qname, formats in plan.formats_by_qname().items()
        for fmt in formats
    }
    assert len(scheduled) == 3
    assert (nonexpert, "FP8_CB_K48") in scheduled


def test_mxfp4_expert_with_k32_ceiling_has_no_current_producer_rung():
    qname = "model.layers.0.mlp.experts.7.gate_proj"
    with pytest.raises(ValueError, match="matches neither declared menu") as caught:
        build_source_class_format_plan(
            {qname: _stats((2048, 4096))},
            {qname: "mxfp4"},
            _Profile(),
            expert_formats=EXPERT_FORMATS,
            nonexpert_formats=NONEXPERT_FORMATS,
            cb_serialization_context=CONTEXT,
        )
    assert "derives legal family []" in str(caught.value)


@pytest.mark.parametrize("group_kind", ["fused", "packed"])
def test_compatible_current_source_group_stays_one_menu(group_kind: str):
    left = "model.layers.0.role_a"
    right = "model.layers.0.role_b"
    profile = _Profile(
        fused=group_kind == "fused",
        packed=group_kind == "packed",
    )

    plan = build_source_class_format_plan(
        {
            left: _stats((4096, 2048)),
            right: _stats((4096, 2048)),
        },
        {left: "fp8", right: "fp8"},
        profile,
        expert_formats=EXPERT_FORMATS,
        nonexpert_formats=NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
    )

    assert plan.formats_for(left) == NONEXPERT_FORMATS
    assert plan.formats_for(right) == NONEXPERT_FORMATS
    assert (left, right) in plan.serving_groups


def test_nonexpert_menu_cannot_be_demand_or_disk_truncated():
    qname = "model.layers.0.self_attn.q_proj"
    with pytest.raises(ValueError, match="complete registered family"):
        build_source_class_format_plan(
            {qname: _stats((4096, 4096))},
            {qname: "fp8"},
            _Profile(),
            expert_formats=EXPERT_FORMATS,
            nonexpert_formats=NONEXPERT_FORMATS[:-1],
            cb_serialization_context=CONTEXT,
        )


def test_plan_round_trip_is_identity_bound(tmp_path):
    qname = "model.layers.0.self_attn.q_proj"
    plan = build_source_class_format_plan(
        {qname: _stats((4096, 4096))},
        {qname: "fp8"},
        _Profile(),
        expert_formats=EXPERT_FORMATS,
        nonexpert_formats=NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
    )
    path = tmp_path / "format_plan.json"
    write_format_plan(plan, path)
    loaded = load_format_plan(path)
    assert loaded.identity_sha256 == plan.identity_sha256
    assert loaded.formats_for(qname) == NONEXPERT_FORMATS

    payload = json.loads(path.read_text())
    payload["units"][qname]["source_kind"] = "mxfp4"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_format_plan(path)


# --- serving-backed restriction --------------------------------------------
#
# Reader-only FP8-CB rungs remain registered for historical reads, but they
# are not legal producer inputs. The serving restriction operates only on the
# current producer family; it must never reintroduce a reader rung.


def _on_law_plan(profile=None):
    nonexpert = "model.layers.0.self_attn.o_proj"
    plan = build_source_class_format_plan(
        {nonexpert: _stats((8192, 4096))},
        {nonexpert: "fp8_ue8m0"},
        profile or _Profile(),
        expert_formats=ON_LAW_EXPERT_FORMATS,
        nonexpert_formats=ON_LAW_NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
        serving_backed_profile=ON_LAW_PROFILE,
    )
    return nonexpert, plan


def test_serving_backed_restriction_admits_exactly_the_on_law_menu():
    nonexpert, plan = _on_law_plan()

    assert plan.menu_id_for(nonexpert) == NONEXPERT_MENU
    assert plan.formats_for(nonexpert) == ON_LAW_NONEXPERT_FORMATS

    scheduled = {
        fmt
        for formats in plan.formats_by_qname().values()
        for fmt in formats
    }
    assert all(int(fmt.rsplit("K", 1)[1]) % 4 == 0 for fmt in scheduled)

    restriction = plan.serving_backed_restriction
    assert restriction is not None
    assert restriction["profile_id"] == ON_LAW_PROFILE
    assert restriction["family"] == "fp8_cb"
    assert restriction["runtime_version"] == gridbook_runtime_version()
    assert restriction["fused_mid_m_rungs"] == [40, 44, 48]
    assert restriction["restricted_out"] == []


def test_current_menu_without_restriction_remains_complete():
    # The current producer family already equals the backed subset. Declaring
    # the restriction still changes plan identity, but omitting it does not
    # make the exact producer menu incomplete.
    qname = "model.layers.0.self_attn.q_proj"
    plan = build_source_class_format_plan(
        {qname: _stats((4096, 4096))},
        {qname: "fp8"},
        _Profile(),
        expert_formats=ON_LAW_EXPERT_FORMATS,
        nonexpert_formats=ON_LAW_NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
    )
    assert plan.formats_for(qname) == ON_LAW_NONEXPERT_FORMATS
    assert plan.serving_backed_restriction is None


def test_serving_backed_restriction_still_refuses_truncation():
    qname = "model.layers.0.self_attn.q_proj"
    with pytest.raises(ValueError, match="backed by the pinned serving") as caught:
        build_source_class_format_plan(
            {qname: _stats((4096, 4096))},
            {qname: "fp8"},
            _Profile(),
            expert_formats=ON_LAW_EXPERT_FORMATS,
            nonexpert_formats=ON_LAW_NONEXPERT_FORMATS[:-1],
            cb_serialization_context=CONTEXT,
            serving_backed_profile=ON_LAW_PROFILE,
        )
    assert "FP8_CB_K48" in str(caught.value)


def test_serving_backed_restriction_refuses_an_unbacked_family():
    # nvfp4_cb's NVFP4-CB lane declares no fused mid-M rung at any runtime, so
    # restricting that family empties it. An empty menu is an error, never a
    # silent truncation.
    qname = "model.layers.0.self_attn.q_proj"
    from prismaquant.cb_layout import NVFP4_PRODUCT_RUNGS

    nvfp4_cb_formats = tuple(
        f"NVFP4_CB_K{k}" for k in NVFP4_PRODUCT_RUNGS
    )
    with pytest.raises(ValueError, match="backs no fused mid-M rung"):
        build_source_class_format_plan(
            {qname: _stats((4096, 4096))},
            {qname: "fp8"},
            _Profile(),
            expert_formats=nvfp4_cb_formats[:2],
            nonexpert_formats=nvfp4_cb_formats,
            cb_serialization_context=CONTEXT,
            serving_backed_profile=ON_LAW_PROFILE,
        )


def test_serving_backed_plan_round_trip_rejects_pin_drift(tmp_path):
    nonexpert, plan = _on_law_plan()
    path = tmp_path / "format_plan.json"
    write_format_plan(plan, path)

    loaded = load_format_plan(path)
    assert loaded.identity_sha256 == plan.identity_sha256
    assert loaded.serving_backed_restriction == plan.serving_backed_restriction
    assert loaded.formats_for(nonexpert) == ON_LAW_NONEXPERT_FORMATS

    # A plan written under one backed set must not be reused under another.
    # Re-stamp the digest so the drift check, not the identity check, is what
    # refuses.
    payload = json.loads(path.read_text())
    payload["serving_backed_restriction"]["runtime_version"] = "0.0.0-other"
    payload.pop("identity_sha256")
    payload["identity_sha256"] = _plan_digest(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different serving-backed restriction"):
        load_format_plan(path)

    historical = load_format_plan(
        path, verify_current_serving_restriction=False
    )
    assert historical.identity_sha256 == payload["identity_sha256"]
    assert historical.serving_backed_restriction == (
        payload["serving_backed_restriction"]
    )
    assert historical.formats_for(nonexpert) == ON_LAW_NONEXPERT_FORMATS
