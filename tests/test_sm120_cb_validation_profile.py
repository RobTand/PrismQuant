from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import format_registry as fr
from prismaquant.cb_layout import (
    FP8_PRODUCT_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)
from prismaquant.dense_anchored_cb import (
    ANCHOR_FORMATS,
    DEFAULT_TARGET_PROFILE,
    DenseCampaignError,
    FP8_CB_LADDER,
    NVFP4_CB_LADDER,
    PANEL_RUNGS,
    VALIDATION_RUNGS,
    _allocator_command,
    _build_parser,
    _derive_dense_plan,
    _require_bf16_body_source_manifest,
)
from prismaquant.lane_spec import load_lane_spec
from prismaquant.gridbook_validation_only_policy import (
    GRIDBOOK_VALIDATION_ONLY_POLICY_SCHEMA,
    SM120_CANDIDATE_GRIDBOOK_COMMIT,
    SM120_CANDIDATE_GRIDBOOK_TREE,
    SM120_CANDIDATE_GRIDBOOK_VERSION,
    SM120_CANDIDATE_RUNTIME_CONTRACT_CANONICAL_SHA256,
    SM120_CANDIDATE_RUNTIME_CONTRACT_FILE_SHA256,
    SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
    SM120_VALIDATION_POLICY_ID,
    VALIDATION_ONLY_DISPOSITION,
    GridbookValidationOnlyPolicyError,
    canonical_json_sha256,
    inspect_validation_only_quant_config,
    load_sm120_validation_candidate_pin,
    prepare_gridbook_validation_only_export_policy,
    require_sm120_validation_runtime_contract,
    sm120_validation_only_policy_stamp,
    sm120_validation_only_route_status_stamp,
    validate_sm120_validation_only_quant_config,
)
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
    assert sm120.producer_policy == SM120_VALIDATION_POLICY_ID
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


def test_sm120_explicitly_denies_w8a16_while_compatibility_readers_remain():
    assert fr.W8A16_COMPAT_FORMAT_NAMES == frozenset({
        "MXFP8A16",
        "INT8_W8A16",
        "FP8_SOURCE",
        "FP8_BLOCK_UE8M0_SOURCE",
    })
    assert fr.W8A16_COMPAT_FORMAT_NAMES == frozenset(
        spec.name for spec in fr.list_formats()
        if spec.weight_bits == 8 and not spec.act_quant_changes_input
    )
    profile = load_serving_profile(SM120_PROFILE)
    rule = next(
        rule for rule in profile.format_rules
        if rule.id == "qwen38_sm120_validation_product_menu"
    )
    assert fr.W8A16_COMPAT_FORMAT_NAMES <= set(rule.deny_formats)
    for name in fr.W8A16_COMPAT_FORMAT_NAMES:
        assert fr.get_format(name).name == name
        assert not check_serving_format(SM120_PROFILE, None, name).legal

    # The generic mixed-container identity stays broad enough to read and
    # reproduce already-published source-FP8 artifacts.  Target maintenance
    # eligibility comes from the exact SM120 profile, not registry deletion.
    for name in ("FP8_SOURCE", "FP8_BLOCK_UE8M0_SOURCE"):
        assert fr.format_is_producer_eligible(name)
        assert check_serving_format("nvfp4_cb", None, name).legal


def test_sm120_rejects_aliases_and_other_a16_or_mx_research_formats():
    for name in (
        "NVFP4A16",
        "MXFP8A16",
        "INT8_W8A16",
        "INT4_W4A16_g128",
        "MXFP8",
        "MXFP8_E4M3",
        "MXFP8_E5M2",
        "FP8_E5M2",
    ):
        assert not check_serving_format(SM120_PROFILE, None, name).legal, name


def test_dense_sm120_campaign_requires_a_complete_bf16_source_body():
    body = ("model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj")
    assert _require_bf16_body_source_manifest(
        body,
        {name: "bf16" for name in body},
    ) == {name: "bf16" for name in body}

    with pytest.raises(DenseCampaignError, match="source census misses"):
        _require_bf16_body_source_manifest(body, {body[0]: "bf16"})
    with pytest.raises(DenseCampaignError, match="one bf16 source class"):
        _require_bf16_body_source_manifest(
            body,
            {body[0]: "bf16", body[1]: "fp8_ue8m0"},
        )


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
    assert ANCHOR_FORMATS[("fp8_cb", "lattice")] == "FP8_CB_K44"
    assert PANEL_RUNGS["fp8_cb"] == ("FP8_CB_K40", "FP8_CB_K48")
    # With only three evidence-backed producer rungs, validation repeats the
    # two non-anchor widths on held-out units. It must not invent a reader-only
    # rung merely to preserve rung-disjoint validation.
    assert VALIDATION_RUNGS["fp8_cb"] == PANEL_RUNGS["fp8_cb"]
    assert ANCHOR_FORMATS[("fp8_cb", "lattice")] not in (
        *PANEL_RUNGS["fp8_cb"], *VALIDATION_RUNGS["fp8_cb"],
    )


def test_aqua_activation_identity_covers_both_families_without_profile_bias():
    lane = load_lane_spec("nvfp4_cb")
    assert SM120_PROFILE in lane.serving_profiles
    contract = lane.served_activation_quantization
    assert contract is not None
    for name in (
        "NVFP4_CB_K12", "NVFP4_CB_K24", "FP8_CB_K28", "FP8_CB_K48",
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
    assert payload["producer_policy"] == SM120_VALIDATION_POLICY_ID


def _sm120_quant_config() -> dict:
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K12",
        "model.layers.0.self_attn.k_proj": "FP8_CB_K48",
        "model.norm": "BF16",
    }
    policy = sm120_validation_only_policy_stamp(
        SM120_VALIDATION_CANDIDATE_CONTRACT_PATH
    )
    route = sm120_validation_only_route_status_stamp(policy, assignment)
    return {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "producer_policy": policy,
            "tensor_formats": assignment,
            "cb_route_status": route,
        },
    }


def test_sm120_candidate_pin_and_contract_have_one_exact_untagged_identity(
    tmp_path,
):
    pin = load_sm120_validation_candidate_pin()
    raw = SM120_VALIDATION_CANDIDATE_CONTRACT_PATH.read_bytes()
    contract = json.loads(raw)

    assert pin["gridbook"] == {
        "version": SM120_CANDIDATE_GRIDBOOK_VERSION,
        "commit": SM120_CANDIDATE_GRIDBOOK_COMMIT,
        "tree": SM120_CANDIDATE_GRIDBOOK_TREE,
        "version_is_release": False,
        "release_tag": None,
    }
    assert contract["schema"] == "gridbook.runtime-contract.v11"
    assert contract["contract_version"] == 11
    assert contract["lane_eligibility"]["schema"] == (
        "gridbook.lane-eligibility.v2"
    )
    assert canonical_json_sha256(contract) == (
        SM120_CANDIDATE_RUNTIME_CONTRACT_CANONICAL_SHA256
    )
    assert hashlib.sha256(raw).hexdigest() == (
        SM120_CANDIDATE_RUNTIME_CONTRACT_FILE_SHA256
    )
    # The immutable candidate is intentionally not rewritten in this
    # contraction: its broader NVFP4 K1..K25 declaration is now an explicit
    # pin blocker until Gridbook supplies a separately reviewed identity.
    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match=r"NVFP4_CB_K reader rungs are unknown.*\[1, 2, 3",
    ):
        require_sm120_validation_runtime_contract(
            SM120_VALIDATION_CANDIDATE_CONTRACT_PATH
        )

    tampered_pin = copy.deepcopy(pin)
    tampered_pin["gridbook"]["version_is_release"] = True
    path = tmp_path / "tampered-candidate-pin.json"
    path.write_text(json.dumps(tampered_pin))
    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="differs from the reviewed exact",
    ):
        load_sm120_validation_candidate_pin(path)


def test_sm120_policy_requires_profile_policy_and_explicit_exact_contract(
    tmp_path,
):
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K16",
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": SM120_PROFILE,
        },
    }))

    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="requires producer_policy",
    ):
        prepare_gridbook_validation_only_export_policy(
            layer_config_path=recipe,
            producer_policy=None,
            runtime_contract=None,
            where="test",
        )
    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="explicit exact",
    ):
        prepare_gridbook_validation_only_export_policy(
            layer_config_path=recipe,
            producer_policy=SM120_VALIDATION_POLICY_ID,
            runtime_contract=None,
            where="test",
        )

    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="NVFP4_CB_K reader rungs are unknown",
    ):
        prepare_gridbook_validation_only_export_policy(
            layer_config_path=recipe,
            producer_policy=SM120_VALIDATION_POLICY_ID,
            runtime_contract=SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
            where="test",
        )


def test_sm120_policy_cannot_be_replayed_on_another_profile(tmp_path):
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K16",
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "target_profile": "nvfp4_cb",
        },
    }))
    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="cannot be replayed",
    ):
        prepare_gridbook_validation_only_export_policy(
            layer_config_path=recipe,
            producer_policy=SM120_VALIDATION_POLICY_ID,
            runtime_contract=SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
            where="test",
        )


def test_sm120_quant_config_cannot_be_minted_from_the_incompatible_pin():
    with pytest.raises(
        GridbookValidationOnlyPolicyError,
        match="NVFP4_CB_K reader rungs are unknown",
    ):
        _sm120_quant_config()
