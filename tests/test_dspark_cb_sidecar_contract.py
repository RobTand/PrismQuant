"""Pure DSpark CB sidecar namespace and activation-bridge contract."""
from __future__ import annotations

from copy import deepcopy

import pytest

from prismaquant.dspark_source_metadata import (
    DSPARK_TARGET_BRIDGE_SCHEMA,
    build_dspark_target_bridge,
    dspark_cb_construction_target_for_physical_output,
    dspark_cb_expected_physical_targets,
    dspark_cb_physical_output_for_construction_target,
    dspark_cb_physical_output_for_recipe_target,
    dspark_cb_physical_source_for_recipe_target,
    dspark_cb_source_passthrough_mapping,
    validate_dspark_target_bridge,
)


def _config() -> dict:
    return {
        "num_hidden_layers": 43,
        "n_mtp_layers": 3,
        "dspark_target_layer_ids": [40, 41, 42],
    }


def _bridge_inputs() -> tuple[list[str], dict]:
    construction = [
        f"model.layers.{43 + stage}.{tail}"
        for stage in range(3)
        for tail in (
            "attn.wq_a",
            "ffn.shared_experts.w2",
            "ffn.experts.gate_up_proj",
            "ffn.experts.down_proj",
        )
    ]
    physical = [
        f"mtp.{stage}.{tail}"
        for stage in range(3)
        for tail in (
            "attn.wq_a",
            "ffn.shared_experts.w2",
            "ffn.experts.gate_up_proj",
            "ffn.experts.down_proj",
        )
    ]
    return construction, {
        "schema": "prismaquant.nvfp4_w4a4_activation.v2",
        "target_count": len(physical),
        "target_names": physical,
    }


@pytest.mark.parametrize(
    ("physical", "construction"),
    (
        ("mtp.0.attn.wq_a", "model.layers.43.attn.wq_a"),
        ("mtp.1.attn.wo_b", "model.layers.44.attn.wo_b"),
        (
            "mtp.2.ffn.shared_experts.w3",
            "model.layers.45.ffn.shared_experts.w3",
        ),
        (
            "mtp.1.ffn.experts.gate_up_proj",
            "model.layers.44.ffn.experts.gate_up_proj",
        ),
        (
            "mtp.2.ffn.experts.down_proj",
            "model.layers.45.ffn.experts.down_proj",
        ),
    ),
)
def test_physical_outputs_and_construction_targets_are_same_tail_inverses(
    physical,
    construction,
):
    assert dspark_cb_construction_target_for_physical_output(
        physical, _config()
    ) == construction
    assert dspark_cb_physical_output_for_construction_target(
        construction, _config()
    ) == physical
    assert dspark_cb_physical_output_for_recipe_target(
        physical, _config()
    ) == physical


@pytest.mark.parametrize(
    ("projection", "source_leaf", "packed_leaf"),
    (
        ("gate_proj", "w1", "gate_up_proj"),
        ("up_proj", "w3", "gate_up_proj"),
        ("down_proj", "w2", "down_proj"),
    ),
)
def test_routed_expert_recipe_members_resolve_source_and_packed_output(
    projection,
    source_leaf,
    packed_leaf,
):
    recipe = f"mtp.1.ffn.experts.217.{projection}"
    assert dspark_cb_physical_source_for_recipe_target(
        recipe, _config()
    ) == f"mtp.1.ffn.experts.217.{source_leaf}"
    output = dspark_cb_physical_output_for_recipe_target(recipe, _config())
    assert output == f"mtp.1.ffn.experts.{packed_leaf}"
    assert dspark_cb_construction_target_for_physical_output(
        output, _config()
    ) == f"model.layers.44.ffn.experts.{packed_leaf}"


def test_dense_and_shared_recipe_sources_remain_physical():
    for target in (
        "mtp.0.attn.wkv",
        "mtp.1.attn.wq_b",
        "mtp.2.ffn.shared_experts.w1",
    ):
        assert dspark_cb_physical_source_for_recipe_target(
            target, _config()
        ) == target
        assert dspark_cb_physical_output_for_recipe_target(
            target, _config()
        ) == target


def test_hybrid_contract_excludes_grouped_bmm_wo_a_from_cb():
    physical = dspark_cb_expected_physical_targets(_config())
    assert len(physical) == 27
    assert all(not target.endswith("attn.wo_a") for target in physical)
    assert dspark_cb_source_passthrough_mapping(_config()) == {
        "mtp.0.attn.wo_a": "model.layers.43.attn.wo_a",
        "mtp.0.main_proj": "model.main_proj",
        "mtp.1.attn.wo_a": "model.layers.44.attn.wo_a",
        "mtp.2.attn.wo_a": "model.layers.45.attn.wo_a",
    }


@pytest.mark.parametrize(
    "target",
    (
        "mtp.0.main_proj",
        "mtp.0.ffn.gate",
        "mtp.3.attn.wq_a",
        "mtp.0.attn.wq_a.weight",
        "model.layers.42.attn.wq_a",
        "model.layers.46.attn.wq_a",
        "model.layers.43.main_proj",
    ),
)
def test_namespace_mapping_rejects_glue_leaves_stages_and_tensor_suffixes(
    target,
):
    if target.startswith("mtp."):
        with pytest.raises(ValueError):
            dspark_cb_construction_target_for_physical_output(
                target, _config()
            )
    else:
        with pytest.raises(ValueError):
            dspark_cb_physical_output_for_construction_target(
                target, _config()
            )


@pytest.mark.parametrize(
    "mutation",
    (
        {"n_mtp_layers": 2},
        {"n_mtp_layers": 4},
        {"dspark_target_layer_ids": [40, 41]},
        {"dspark_target_layer_ids": [40, 40, 42]},
    ),
)
def test_namespace_mapping_rejects_partial_or_invalid_topology(mutation):
    config = _config()
    config.update(mutation)
    with pytest.raises(ValueError):
        dspark_cb_physical_output_for_recipe_target(
            "mtp.0.attn.wq_a", config
        )


def test_source_config_without_emitted_stage_count_uses_released_stage_ids():
    config = _config()
    config.pop("n_mtp_layers")
    assert dspark_cb_construction_target_for_physical_output(
        "mtp.2.attn.wo_a", config
    ) == "model.layers.45.attn.wo_a"


def test_bridge_builds_exact_gridbook_record_and_round_trips_validator():
    construction, execution = _bridge_inputs()
    bridge = build_dspark_target_bridge(
        _config(),
        contracted_cb_construction_targets=construction,
        activation_execution_contract=execution,
    )
    assert bridge == {
        "schema": DSPARK_TARGET_BRIDGE_SCHEMA,
        "num_hidden_layers": 43,
        "n_mtp_layers": 3,
        "construction_to_physical": {
            target: dspark_cb_physical_output_for_construction_target(
                target, _config()
            )
            for target in sorted(construction)
        },
    }
    assert set(bridge["construction_to_physical"]) == set(construction)
    assert set(bridge["construction_to_physical"].values()) == set(
        execution["target_names"]
    )
    assert validate_dspark_target_bridge(
        bridge,
        _config(),
        contracted_cb_construction_targets=construction,
        activation_execution_contract=execution,
    ) == bridge


def test_bridge_is_optional_only_when_the_activation_contract_is_absent():
    assert build_dspark_target_bridge(_config()) is None
    assert validate_dspark_target_bridge(None, _config()) is None
    with pytest.raises(ValueError, match="must be absent"):
        validate_dspark_target_bridge(
            {"schema": DSPARK_TARGET_BRIDGE_SCHEMA}, _config()
        )
    with pytest.raises(ValueError, match="require an activation"):
        build_dspark_target_bridge(
            _config(),
            contracted_cb_construction_targets=[
                "model.layers.43.attn.wq_a"
            ],
        )


def test_bridge_refuses_partial_stage_set():
    construction, execution = _bridge_inputs()
    construction = [
        target for target in construction if not target.startswith("model.layers.45.")
    ]
    execution["target_names"] = [
        target for target in execution["target_names"]
        if not target.startswith("mtp.2.")
    ]
    execution["target_count"] = len(execution["target_names"])
    with pytest.raises(ValueError, match="complete three-stage set"):
        build_dspark_target_bridge(
            _config(),
            contracted_cb_construction_targets=construction,
            activation_execution_contract=execution,
        )


def test_bridge_refuses_physical_contract_mismatch_duplicate_and_main_proj():
    construction, execution = _bridge_inputs()
    mismatched = deepcopy(execution)
    mismatched["target_names"][0] = "mtp.0.attn.wkv"
    with pytest.raises(ValueError, match="must exactly equal"):
        build_dspark_target_bridge(
            _config(),
            contracted_cb_construction_targets=construction,
            activation_execution_contract=mismatched,
        )

    duplicated = deepcopy(execution)
    duplicated["target_names"][1] = duplicated["target_names"][0]
    with pytest.raises(ValueError, match="duplicate"):
        build_dspark_target_bridge(
            _config(),
            contracted_cb_construction_targets=construction,
            activation_execution_contract=duplicated,
        )

    with pytest.raises(ValueError, match="main_proj"):
        build_dspark_target_bridge(
            _config(),
            contracted_cb_construction_targets=[
                "model.layers.43.main_proj",
                "model.layers.44.attn.wq_a",
                "model.layers.45.attn.wq_a",
            ],
            activation_execution_contract={
                "target_count": 3,
                "target_names": [
                    "mtp.0.main_proj",
                    "mtp.1.attn.wq_a",
                    "mtp.2.attn.wq_a",
                ],
            },
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record.update({"schema": "future.schema"}),
        lambda record: record.update({"unexpected": True}),
        lambda record: record["construction_to_physical"].update(
            {"model.layers.43.attn.wq_a": "mtp.0.attn.wkv"}
        ),
        lambda record: record.update({"n_mtp_layers": 2}),
    ),
)
def test_bridge_validator_rejects_schema_shape_or_mapping_drift(mutation):
    construction, execution = _bridge_inputs()
    bridge = build_dspark_target_bridge(
        _config(),
        contracted_cb_construction_targets=construction,
        activation_execution_contract=execution,
    )
    mutation(bridge)
    with pytest.raises(ValueError):
        validate_dspark_target_bridge(
            bridge,
            _config(),
            contracted_cb_construction_targets=construction,
            activation_execution_contract=execution,
        )
