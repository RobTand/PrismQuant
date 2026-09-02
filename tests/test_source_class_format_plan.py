from __future__ import annotations

import json
import math

import pytest

from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.serving_profiles import serving_runtime_version
from prismaquant.source_class_format_plan import (
    EXPERT_MENU,
    NONEXPERT_MENU,
    _plan_digest,
    build_source_class_format_plan,
    load_format_plan,
    write_format_plan,
)


EXPERT_FORMATS = tuple(f"FP8_CB_K{k}" for k in range(4, 33, 4))
NONEXPERT_FORMATS = tuple(f"FP8_CB_K{k}" for k in range(4, 49, 4))
# The DSv4 on-law menus: the fused mid-M rungs nvfp4_cb backs at the pinned
# runtime, intersected with the byte-exact source-payload ceiling (K33 for a
# routed expert, K48 for a dense row).
ON_LAW_PROFILE = "nvfp4_cb"
ON_LAW_EXPERT_FORMATS = ("FP8_CB_K28", "FP8_CB_K32")
ON_LAW_NONEXPERT_FORMATS = tuple(
    f"FP8_CB_K{k}" for k in (28, 32, 36, 40, 44, 48)
)
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


def test_source_derived_split_prices_no_illegal_expert_cells_and_keeps_k48():
    # Both units are ordinary rank-2 rows. The lower-rate unit is also marked
    # as routed by the fake profile, but the planner never consults that fact:
    # exact source payload alone selects its menu.
    expert = "model.layers.0.mlp.experts.7.gate_proj"
    nonexpert = "model.layers.0.self_attn.o_proj"
    plan = build_source_class_format_plan(
        {
            expert: _stats((2048, 4096)),
            nonexpert: _stats((8192, 4096)),
        },
        {
            expert: "mxfp4",
            nonexpert: "fp8_ue8m0",
        },
        _Profile(),
        expert_formats=EXPERT_FORMATS,
        nonexpert_formats=NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
    )

    assert plan.menu_id_for(expert) == EXPERT_MENU
    assert plan.formats_for(expert) == EXPERT_FORMATS
    assert plan.menu_id_for(nonexpert) == NONEXPERT_MENU
    assert plan.formats_for(nonexpert) == NONEXPERT_FORMATS

    scheduled = {
        (qname, fmt)
        for qname, formats in plan.formats_by_qname().items()
        for fmt in formats
    }
    assert len(scheduled) == 8 + 12
    assert not any(
        qname == expert and int(fmt.rsplit("K", 1)[1]) > 32
        for qname, fmt in scheduled
    )
    assert (nonexpert, "FP8_CB_K48") in scheduled


@pytest.mark.parametrize("group_kind", ["fused", "packed"])
def test_serving_group_cannot_straddle_source_class_menus(group_kind: str):
    low = "model.layers.0.role_a"
    high = "model.layers.0.role_b"
    profile = _Profile(
        fused=group_kind == "fused",
        packed=group_kind == "packed",
    )

    with pytest.raises(ValueError, match="split one fused/packed") as caught:
        build_source_class_format_plan(
            {
                low: _stats((2048, 4096)),
                high: _stats((4096, 2048)),
            },
            {low: "mxfp4", high: "fp8"},
            profile,
            expert_formats=EXPERT_FORMATS,
            nonexpert_formats=NONEXPERT_FORMATS,
            cb_serialization_context=CONTEXT,
        )

    message = str(caught.value)
    assert low in message
    assert high in message
    assert EXPERT_MENU in message
    assert NONEXPERT_MENU in message


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


# --- serving-backed restriction (the DSv4 on-law menu) ---------------------
#
# Off-law FP8-CB rungs remain registered for historical reads, but they are no
# longer legal producer inputs.  This further restriction intersects the full
# K%4 producer family with the pinned runtime's backed set; it is a serving
# legality bound, not a demand- or disk-driven truncation.


def _on_law_plan(profile=None):
    expert = "model.layers.0.mlp.experts.7.gate_proj"
    nonexpert = "model.layers.0.self_attn.o_proj"
    plan = build_source_class_format_plan(
        {expert: _stats((2048, 4096)), nonexpert: _stats((8192, 4096))},
        {expert: "mxfp4", nonexpert: "fp8_ue8m0"},
        profile or _Profile(),
        expert_formats=ON_LAW_EXPERT_FORMATS,
        nonexpert_formats=ON_LAW_NONEXPERT_FORMATS,
        cb_serialization_context=CONTEXT,
        serving_backed_profile=ON_LAW_PROFILE,
    )
    return expert, nonexpert, plan


def test_on_law_menu_without_the_restriction_is_still_refused():
    # The anti-truncation guard is untouched on every other axis: the SAME
    # menus that are legal under a declared serving restriction are refused
    # when the caller declares none.
    qname = "model.layers.0.self_attn.q_proj"
    with pytest.raises(ValueError, match="complete registered family") as caught:
        build_source_class_format_plan(
            {qname: _stats((4096, 4096))},
            {qname: "fp8"},
            _Profile(),
            expert_formats=ON_LAW_EXPERT_FORMATS,
            nonexpert_formats=ON_LAW_NONEXPERT_FORMATS,
            cb_serialization_context=CONTEXT,
        )
    assert "FP8_CB_K4" in str(caught.value)


