from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from prismaquant import format_registry as fr
from prismaquant.cb_layout import (
    FP8_PRODUCT_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)
from prismaquant.dense_anchored_cb import (
    ANCHOR_FORMATS,
    DEFAULT_TARGET_PROFILE,
    FP8_CB_LADDER,
    NVFP4_CB_LADDER,
    PANEL_RUNGS,
    VALIDATION_RUNGS,
    _allocator_command,
    _build_parser,
    _derive_dense_plan,
)
from prismaquant.lane_spec import load_lane_spec
from prismaquant.serving_profiles import (
    check_serving_format,
    load_serving_profile,
    require_profile_export_lane,
    serving_profile_names,
)


SM120_PROFILE = "qwen38_sm120_cb_validation_only"
SM89_PROFILE = "qwen38_rtx4090_fp8_cb"

NVFP4_LADDER = tuple(f"NVFP4_CB_K{k}" for k in NVFP4_PRODUCT_RUNGS)
FP8_LADDER = tuple(f"FP8_CB_K{k}" for k in FP8_PRODUCT_RUNGS)
NATIVE_TERMINALS = frozenset({"NVFP4", "FP8_E4M3", "BF16"})


def _legal_registered(profile_id: str) -> set[str]:
    return {
        name for name in fr.REGISTRY
        if check_serving_format(profile_id, None, name).legal
    }


class _DenseProfileStub:
    def fused_sibling_group(self, _qname: str):
        return None

    def packed_expert_format_group(self, _qname: str):
        return None


def test_sm89_is_closed_fp8_only_and_sm120_registers_both_public_ladders():
    sm89 = load_serving_profile(SM89_PROFILE)
    sm120 = load_serving_profile(SM120_PROFILE)

    assert sm89.target_platform == "sm_89"
    assert sm120.target_platform == "sm_120"
    assert SM120_PROFILE in serving_profile_names()
    assert sm120.producer_policy is None
    assert sm120.export_lane is not None
    assert require_profile_export_lane(SM120_PROFILE, "nvfp4_cb") == "nvfp4_cb"

    expected_sm120 = set(NVFP4_LADDER) | set(FP8_LADDER) | NATIVE_TERMINALS
    assert _legal_registered(SM120_PROFILE) == expected_sm120
    assert check_serving_format(SM120_PROFILE, None, "FP8_DYNAMIC").legal

    assert all(
        not check_serving_format(SM89_PROFILE, None, name).legal
        for name in (*NVFP4_LADDER, "NVFP4")
    )
    assert all(
        check_serving_format(SM89_PROFILE, None, name).legal
        for name in (*FP8_LADDER, "FP8_E4M3", "BF16")
    )


def test_sm120_refuses_reader_only_off_law_and_direct_research_widths():
    for name in (
        "FP8_CB_K29",
        "FP8_CB_K47",
        "NVFP4_CB_K26",
        "NVFP4_CB_K32",
        "FP8_SOURCE",
        "FP8_BLOCK_UE8M0_SOURCE",
        "MXFP4_SOURCE",
        "MXFP8_UE8M0_G32",
    ):
        assert not check_serving_format(SM120_PROFILE, None, name).legal, name


def test_dense_plan_binds_selected_profile_and_complete_public_ladders():
    plan = _derive_dense_plan(
        _DenseProfileStub(),
        ("model.layers.0.mlp.down_proj",),
        target_profile=SM120_PROFILE,
    )

    assert plan.target_profile == SM120_PROFILE
    assert plan.target_platform == "sm_120"
    assert plan.ladders == {
        "nvfp4_cb": NVFP4_LADDER,
        "fp8_cb": FP8_LADDER,
    }
    assert plan.serving_groups == ()
    assert plan.to_dict()["schema"] == (
        "prismaquant.dense_anchored_cb.format_plan.v2"
    )
    assert plan.to_dict()["target_profile"] == SM120_PROFILE
    assert all(
        row["fused_mid_m_is_performance_metadata_not_format_admission"]
        for row in plan.serving_restrictions.values()
    )


def test_dense_allocator_uses_plan_profile_and_both_full_ladders(tmp_path):
    plan = _derive_dense_plan(
        _DenseProfileStub(), ("a",), target_profile=SM120_PROFILE,
    )
    prepared = SimpleNamespace(
        format_plan=plan,
        args=SimpleNamespace(
            probe=tmp_path / "probe.pkl",
            model=tmp_path / "model",
            target_disk_gb=10.0,
            artifact_overhead_reserve_bytes=40_000_000,
            col_weights=tmp_path / "imatrix.pkl",
            threads=4,
        ),
    )
    command = _allocator_command(
        prepared,
        cost_path=tmp_path / "cost.pkl",
        output_dir=tmp_path / "allocation",
    )

    profile_index = command.index("--target-profile")
    formats_index = command.index("--formats")
    assert command[profile_index + 1] == SM120_PROFILE
    assert tuple(command[formats_index + 1].split(",")) == (
        *NVFP4_LADDER, *FP8_LADDER, "BF16",
    )
    assert _build_parser().get_default("target_profile") == SM120_PROFILE
    assert DEFAULT_TARGET_PROFILE == SM120_PROFILE


def test_dense_measurements_span_full_fp8_ladder_without_vacuous_cells():
    assert NVFP4_CB_LADDER == NVFP4_LADDER
    assert FP8_CB_LADDER == FP8_LADDER
    assert ANCHOR_FORMATS[("fp8_cb", "lattice")] == "FP8_CB_K24"
    assert PANEL_RUNGS["fp8_cb"] == (
        "FP8_CB_K4", "FP8_CB_K28", "FP8_CB_K48",
    )
    assert VALIDATION_RUNGS["fp8_cb"] == (
        "FP8_CB_K8", "FP8_CB_K20", "FP8_CB_K36", "FP8_CB_K44",
    )
    assert set(VALIDATION_RUNGS["fp8_cb"]).isdisjoint(
        PANEL_RUNGS["fp8_cb"]
    )
    assert ANCHOR_FORMATS[("fp8_cb", "lattice")] not in (
        *PANEL_RUNGS["fp8_cb"], *VALIDATION_RUNGS["fp8_cb"],
    )


def test_aqua_activation_identity_covers_both_families_without_profile_bias():
    lane = load_lane_spec("nvfp4_cb")
    assert SM120_PROFILE in lane.serving_profiles
    contract = lane.served_activation_quantization
    assert contract is not None
    for name in (
        "NVFP4_CB_K1", "NVFP4_CB_K25", "FP8_CB_K4", "FP8_CB_K48",
    ):
        assert contract.matches(name)

    profile_path = (
        Path(__file__).parents[1]
        / "prismaquant"
        / "serving_profile_specs"
        / f"{SM120_PROFILE}.json"
    )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert "manual family preference" in payload["format_rules"][0]["detail"]
    assert "producer_policy" not in payload
