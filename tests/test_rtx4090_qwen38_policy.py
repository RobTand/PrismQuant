"""Strict format/model/artifact gates for the RTX 4090 Qwen3.8 lane."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.validate_rtx4090_fp8_cb as rtx_validator
from prismaquant.cb_layout import (
    FP8_ACCEPTED_RUNGS,
    FP8_PRODUCT_RUNGS,
    NVFP4_ACCEPTED_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)
from prismaquant.format_registry import get_format
from prismaquant.export_native_compressed import FP8_E4M3_SCHEME
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES,
    RTX4090_QWEN38_FORMAT_MENU,
    RTX4090_QWEN38_LAYER_TYPES,
    RTX4090_QWEN38_POLICY_ID,
    RTX4090_QWEN38_SERVING_PROFILE,
    RTX4090_ROUTE_STATUS_SCHEMA,
    RTX4090_VALIDATION_ONLY_DISPOSITION,
    RTX4090_VALIDATION_ONLY_POLICY_ID,
    RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
    RTX4090Qwen38PolicyError,
    prepare_rtx4090_export_policy,
    producer_policy_stamp,
    rtx4090_route_status_stamp,
    rtx4090_route_status_summary,
    require_rtx4090_runtime_contract,
    require_rtx4090_compile_only_runtime_contract,
    rtx4090_graph_requirement,
    validate_qwen38_dense_config,
    validate_rtx4090_assignment,
    validate_rtx4090_format_menu,
    validate_rtx4090_quant_config_manifest,
    validate_rtx4090_route_status,
    validation_only_producer_policy_stamp,
)
from prismaquant.serving_profiles import check_serving_format
from prismaquant.shipcard import build_shipcard


REPO = Path(__file__).resolve().parents[1]
CURRENT = (
    REPO / "prismaquant" / "gridbook_runtime"
    / "gridbook_runtime_contract.0.8.11.json"
)


def _runtime_contract(*, qualified: bool = True) -> dict:
    contract = json.loads(CURRENT.read_text(encoding="utf-8"))
    contract["schema"] = "gridbook.runtime-contract.v11"
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
    qualification = "device_qualified" if qualified else "compile_only"
    contract["lane_eligibility"] = {
        "schema": "gridbook.lane-eligibility.v2",
        "platforms": {"sm_89": {"compute_capability": [8, 9]}},
        "regimes": ["decode", "batch"],
        "structures": ["dense", "routed_moe"],
        "cells": [
            {
                "id": f"fp8_cb_dense_sm89_{regime}",
                "platform": "sm_89",
                "family": "FP8_CB_K",
                "structure": "dense",
                "regime": regime,
                "rungs": list(FP8_PRODUCT_RUNGS),
                "route_status": "backed",
                "qualification": qualification,
                "requires_serve_flags": [],
                "predicates": [],
            }
            for regime in ("decode", "batch")
        ],
    }
    return contract


def _model_config() -> dict:
    return {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "head_dim": 256,
        "num_key_value_heads": 4,
        "num_attention_heads": 24,
        "max_position_embeddings": 32768,
        "layer_types": list(RTX4090_QWEN38_LAYER_TYPES),
        "tie_word_embeddings": False,
    }


def _official_wrapper_config() -> dict:
    text = _model_config()
    text.pop("architectures")
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": text,
        "vision_config": {"model_type": "qwen3_5_vision"},
    }


def _quant_config(contract: dict) -> dict:
    formats = ("FP8_CB_K44", "FP8_E4M3", "BF16")
    return {
        "quant_method": "gridbook",
        "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "format": "FP8_CB_K44",
                "scheme": {
                    "grid": "fp8",
                    "mode": "product",
                    "k": 44,
                    "superblock": 256,
                    "group_size": 0,
                    "vec_dim": 8,
                    "n_sub": 4,
                    "type_size": 176,
                    "act_bits": 8,
                    "codebook_source": "lattice",
                    "codebook_ref": [
                        "lattice__FP8_CB_K44__sub0",
                        "lattice__FP8_CB_K44__sub1",
                        "lattice__FP8_CB_K44__sub2",
                        "lattice__FP8_CB_K44__sub3",
                    ],
                    "codebook_group": None,
                },
                "targets": ["re:^model.layers.0.mlp.down_proj$"],
            },
            "group_1": {
                **deepcopy(FP8_E4M3_SCHEME),
                "targets": ["re:^model.layers.0.self_attn.o_proj$"],
            },
        },
        "provenance": {
            "tensor_formats": {
                "model.layers.0.mlp.down_proj": "FP8_CB_K44",
                "model.layers.0.self_attn.o_proj": "FP8_E4M3",
                "lm_head": "BF16",
            },
            "producer_policy": producer_policy_stamp(contract, formats),
            "serialized_payload": {
                "schema": "prismaquant.cb_serialized_payload.v2",
                "context": {
                    "scale_coding": "v1",
                    "layout_version": 1,
                    "codebook_source": "lattice",
                    "scale_sweep": True,
                    "scale_sweep_scope": "fp8",
                    "ldlq": False,
                    "encode_tier": "balanced",
                    "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
                },
                "index_bytes": 80,
                "fp4_scale_bytes": 0,
                "fp8_row_scale_bytes": 4,
                "input_global_scale_bytes": 0,
                "global_scale_bytes": 0,
                "tensor_payload_bytes": 84,
                "codebook_sidecar_bytes": 4096,
                "total_bytes": 4180,
                "n_tensors": 1,
                "sidecars": [],
            },
            "artifact_inventory": {
                "export_directory_bytes": 18_000_000_000,
            },
        },
    }


def test_profile_and_policy_menu_are_exactly_fp8_cb_product_plus_terminals():
    expected = tuple(
        [f"FP8_CB_K{k}" for k in FP8_PRODUCT_RUNGS]
        + ["FP8_E4M3", "BF16"]
    )
    assert RTX4090_QWEN38_FORMAT_MENU == expected
    assert validate_rtx4090_format_menu(expected) == expected
    assert all(
        get_format(f"FP8_CB_K{k}").min_capability_sm == 89
        for k in FP8_PRODUCT_RUNGS
    )
    for name in expected:
        assert check_serving_format(
            RTX4090_QWEN38_SERVING_PROFILE, None, name
        ).legal
        assert check_serving_format(
            RTX4090_VALIDATION_ONLY_SERVING_PROFILE, None, name
        ).legal


@pytest.mark.parametrize(
    "name",
    (
        "NVFP4",
        "NVFP4_CB_K16",
        "FP8_CB_K29",
        "FP8_CB_K47",
        "MXFP8_E4M3",
        "FP8_SOURCE",
    ),
)
def test_menu_and_assignment_hard_refuse_every_out_of_policy_format(name):
    with pytest.raises(RTX4090Qwen38PolicyError, match="forbidden"):
        validate_rtx4090_format_menu([name])
    with pytest.raises(RTX4090Qwen38PolicyError, match="forbidden"):
        validate_rtx4090_assignment({"model.layers.0.mlp.down_proj": name})
    assert not check_serving_format(
        RTX4090_QWEN38_SERVING_PROFILE, None, name
    ).legal
    assert not check_serving_format(
        RTX4090_VALIDATION_ONLY_SERVING_PROFILE, None, name
    ).legal


def test_assignment_accepts_dict_spelling_and_fp8_alias():
    assignment = validate_rtx4090_assignment({
        "a.weight": {"data_type": "fp8_cb", "cb_k": 40},
        "b": "FP8_DYNAMIC",
        "c": "BF16",
    })
    assert assignment == {
        "a": "FP8_CB_K40",
        "b": "FP8_E4M3",
        "c": "BF16",
    }


def test_assignment_and_manifest_keep_lm_head_immutable_bf16():
    with pytest.raises(RTX4090Qwen38PolicyError, match="immutable BF16"):
        validate_rtx4090_assignment({"lm_head.weight": "FP8_E4M3"})

    contract = _runtime_contract()
    manifest = _quant_config(contract)
    manifest["provenance"]["tensor_formats"]["lm_head"] = "FP8_E4M3"
    with pytest.raises(RTX4090Qwen38PolicyError, match="immutable BF16"):
        validate_rtx4090_quant_config_manifest(
            manifest, runtime_contract=contract
        )


def test_dense_qwen38_identity_pins_the_32k_kv_memory_contract():
    identity = validate_qwen38_dense_config(_model_config())
    assert identity["hidden_size"] == 5120
    assert identity["num_hidden_layers"] == 64
    assert identity["head_dim"] == 256
    assert identity["num_key_value_heads"] == 4
    assert identity["layer_types"].count("full_attention") == 16
    assert identity["source_layout"] == "flattened_text"

    wrapped = validate_qwen38_dense_config(_official_wrapper_config())
    assert wrapped["source_layout"] == "official_wrapper"
    assert wrapped["outer_architecture"] == "Qwen3_5ForConditionalGeneration"
    assert wrapped["architecture"] == "Qwen3_5ForCausalLM"

    for key, bad in (
        ("hidden_size", 4096),
        ("head_dim", 128),
        ("num_key_value_heads", 8),
        ("num_attention_heads", 32),
        ("max_position_embeddings", 16384),
    ):
        wrong = _model_config()
        wrong[key] = bad
        with pytest.raises(RTX4090Qwen38PolicyError, match=key):
            validate_qwen38_dense_config(wrong)

    wrong_schedule = _model_config()
    wrong_schedule["layer_types"][0] = "full_attention"
    with pytest.raises(RTX4090Qwen38PolicyError, match="layer_types"):
        validate_qwen38_dense_config(wrong_schedule)

    moe = _model_config()
    moe["model_type"] = "qwen3_8_moe"
    moe["architectures"] = ["Qwen3_8MoeForCausalLM"]
    moe["num_experts"] = 128
    with pytest.raises(RTX4090Qwen38PolicyError, match="dense Qwen3.8"):
        validate_qwen38_dense_config(moe)


def test_gridbook_device_attestation_and_graph_policy_are_separate():
    compile_only = _runtime_contract(qualified=False)
    with pytest.raises(RTX4090Qwen38PolicyError, match="compile_only"):
        require_rtx4090_runtime_contract(compile_only, ("FP8_CB_K44",))
    structural = require_rtx4090_compile_only_runtime_contract(
        compile_only, ("FP8_CB_K44",)
    )
    assert structural["rungs"] == list(FP8_PRODUCT_RUNGS)
    assert {
        row["qualification"] for row in structural["regime_routes"]
    } == {"compile_only"}
    with pytest.raises(RTX4090Qwen38PolicyError, match="device_qualified"):
        require_rtx4090_compile_only_runtime_contract(
            _runtime_contract(), ("FP8_CB_K44",)
        )

    attested = require_rtx4090_runtime_contract(
        _runtime_contract(), ("FP8_CB_K44",)
    )
    assert attested["platform"] == "sm_89"
    assert attested["rungs"] == list(FP8_PRODUCT_RUNGS)
    assert len(attested["regime_routes"]) == 2 * len(FP8_PRODUCT_RUNGS)
    assert "graph" not in attested
    assert rtx4090_graph_requirement() == {
        "torch_compile_backend": "inductor",
        "torch_compile_fullgraph": True,
        "vllm_compilation_mode": 3,
        "vllm_cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
        "scheduler_max_num_seqs": 64,
        "receipt_schema": "prismaquant.rtx4090_graph_contract.v1",
    }


def test_artifact_subset_still_qualifies_the_complete_producer_ladder():
    attested = require_rtx4090_runtime_contract(
        _runtime_contract(), ("FP8_CB_K44", "FP8_E4M3", "BF16")
    )

    assert attested["rungs"] == list(FP8_PRODUCT_RUNGS)
    assert {
        (row["rung"], row["regime"])
        for row in attested["regime_routes"]
    } == {
        (rung, regime)
        for rung in FP8_PRODUCT_RUNGS
        for regime in ("decode", "batch")
    }

    # Resolver inputs may be a per-Linear assignment population, unlike the
    # public menu validator, whose duplicate refusal remains intentional.
    repeated = require_rtx4090_compile_only_runtime_contract(
        _runtime_contract(qualified=False),
        ("FP8_CB_K44", "FP8_CB_K44", "BF16", "BF16"),
    )
    assert repeated["rungs"] == list(FP8_PRODUCT_RUNGS)


def test_strict_route_status_is_exact_v11_evidence_not_generic_override_state(
    tmp_path,
):
    assignment = {
        "model.layers.0.mlp.down_proj": "FP8_CB_K44",
        "model.layers.0.self_attn.o_proj": "FP8_E4M3",
        "lm_head": "BF16",
    }
    policy = producer_policy_stamp(
        _runtime_contract(), tuple(assignment.values())
    )
    route = rtx4090_route_status_stamp(policy, assignment)
    assert route["schema"] == RTX4090_ROUTE_STATUS_SCHEMA
    assert route["authority"] == "producer_policy.runtime_attestation"
    assert route["selected_fp8_cb_units"] == 1
    assert route["selected_fp8_cb_rungs"] == [44]
    assert "override" not in route
    assert "declared_non_native_target" not in route
    assert validate_rtx4090_route_status(
        route,
        producer_policy=policy,
        assignment=assignment,
        where="test strict route",
    ) == route
    expected_summary = rtx4090_route_status_summary(
        route,
        producer_policy=policy,
        assignment=assignment,
        where="test strict summary",
    )
    assert expected_summary["route_status"] == "backed"
    assert expected_summary["qualification"] == "device_qualified"
    assert expected_summary["regimes"] == ["decode", "batch"]
    assert None not in expected_summary.values()

    (tmp_path / "quant_config.json").write_text(json.dumps({
        "format": "fp8_cb",
        "provenance": {
            "cb_route_status": route,
            "producer_policy": policy,
            "tensor_formats": assignment,
        },
    }))
    card = build_shipcard(tmp_path, build={})
    assert card["cb_route_status"] == expected_summary

    open_schema = deepcopy(route)
    open_schema["override"] = {"reason": "ship anyway"}
    with pytest.raises(RTX4090Qwen38PolicyError, match="differs from the exact"):
        validate_rtx4090_route_status(
            open_schema,
            producer_policy=policy,
            assignment=assignment,
            where="test route override",
        )

    fallback_policy = deepcopy(policy)
    fallback_policy["runtime_attestation"]["regime_routes"][0][
        "route_status"
    ] = "fallback"
    fallback = rtx4090_route_status_stamp(fallback_policy, assignment)
    with pytest.raises(RTX4090Qwen38PolicyError, match="clean, native"):
        validate_rtx4090_route_status(
            fallback,
            producer_policy=fallback_policy,
            assignment=assignment,
            where="test route fallback",
        )


@pytest.mark.parametrize(
    ("missing_rung", "missing_regime"),
    ((40, "decode"), (44, "batch"), (48, "decode")),
)
def test_runtime_contract_rejects_missing_first_middle_or_last_route(
    missing_rung, missing_regime
):
    contract = _runtime_contract()
    cell = next(
        item for item in contract["lane_eligibility"]["cells"]
        if item["regime"] == missing_regime
    )
    cell["rungs"].remove(missing_rung)

    with pytest.raises(
        RTX4090Qwen38PolicyError,
        match=rf"no sm_89/dense/{missing_regime} route.*K{missing_rung}",
    ):
        require_rtx4090_runtime_contract(contract, ("FP8_CB_K44",))


def _full_runtime_attestation() -> dict:
    return producer_policy_stamp(
        _runtime_contract(), ("FP8_CB_K44", "BF16")
    )["runtime_attestation"]


@pytest.mark.parametrize("bad_rung", (29, 52))
def test_runtime_attestation_replay_rejects_off_law_or_extra_rung(bad_rung):
    attestation = deepcopy(_full_runtime_attestation())
    attestation["rungs"] = sorted([*attestation["rungs"], bad_rung])
    for regime in ("decode", "batch"):
        attestation["regime_routes"].append({
            "rung": bad_rung,
            "regime": regime,
            "cell_id": f"unexpected_k{bad_rung}_{regime}",
            "route_status": "backed",
            "qualification": "device_qualified",
            "requires_serve_flags": [],
        })

    with pytest.raises(
        rtx_validator.RTX4090FP8CBValidationError,
        match="exact K40/K44/K48 producer",
    ):
        rtx_validator._validate_runtime_attestation(attestation)


@pytest.mark.parametrize("missing_rung", (40, 44, 48))
def test_runtime_attestation_replay_rejects_missing_ladder_rung(missing_rung):
    attestation = deepcopy(_full_runtime_attestation())
    attestation["rungs"].remove(missing_rung)
    attestation["regime_routes"] = [
        row for row in attestation["regime_routes"]
        if row["rung"] != missing_rung
    ]

    with pytest.raises(
        rtx_validator.RTX4090FP8CBValidationError,
        match="exact K40/K44/K48 producer",
    ):
        rtx_validator._validate_runtime_attestation(attestation)


def test_export_preflight_requires_model_assignment_and_qualified_v11(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(json.dumps(_model_config()))
    contract = _runtime_contract()
    # Exact source-layout/identity behavior has its own census tests.  This
    # policy test isolates the runtime-contract and format preflight ordering.
    monkeypatch.setattr(
        "prismaquant.rtx4090_artifact_census.preflight_rtx4090_source_census",
        lambda **_kwargs: {
            "schema": "prismaquant.rtx4090_qwen38_source_census.v1",
            "source_model_identity": {"content_sha256": "0" * 64},
        },
    )
    resolved, stamp = prepare_rtx4090_export_policy(
        model_dir=tmp_path,
        assignment={"model.layers.0.mlp.down_proj": "FP8_CB_K44"},
        producer_policy=RTX4090_QWEN38_POLICY_ID,
        runtime_contract=contract,
        where="test exporter",
    )
    assert resolved == contract
    assert stamp["target_platform"] == "sm_89"
    assert stamp["runtime_attestation"]["platform"] == "sm_89"

    with pytest.raises(RTX4090Qwen38PolicyError, match="compile_only"):
        prepare_rtx4090_export_policy(
            model_dir=tmp_path,
            assignment={"model.layers.0.mlp.down_proj": "FP8_CB_K44"},
            producer_policy=RTX4090_QWEN38_POLICY_ID,
            runtime_contract=_runtime_contract(qualified=False),
            where="test exporter",
        )
    _, validation_stamp = prepare_rtx4090_export_policy(
        model_dir=tmp_path,
        assignment={"model.layers.0.mlp.down_proj": "FP8_CB_K44"},
        producer_policy=RTX4090_VALIDATION_ONLY_POLICY_ID,
        runtime_contract=_runtime_contract(qualified=False),
        where="test validation-only exporter",
    )
    assert validation_stamp["artifact_disposition"] == (
        RTX4090_VALIDATION_ONLY_DISPOSITION
    )
    assert validation_stamp["runtime_qualification_ceiling"] == "compile_only"
    with pytest.raises(RTX4090Qwen38PolicyError, match="without an explicit"):
        prepare_rtx4090_export_policy(
            model_dir=tmp_path,
            assignment={"model.layers.0.mlp.down_proj": "FP8_CB_K44"},
            producer_policy=None,
            runtime_contract=contract,
            where="test exporter",
        )


def test_runtime_attestation_sha_binds_every_contract_field():
    contract = _runtime_contract()
    changed = deepcopy(contract)
    # This field is outside the lane resolver's predicate inputs, so both
    # contracts remain route-valid.  The immutable handoff still has to bind
    # it: a consumer must never mistake a nearby contract for the qualified
    # release payload.
    changed["abi_features"]["dspark_construction_physical_bridge"] = 2

    original = require_rtx4090_runtime_contract(
        contract, ("FP8_CB_K44",)
    )
    mutated = require_rtx4090_runtime_contract(
        changed, ("FP8_CB_K44",)
    )
    assert original["runtime_contract_sha256"] != (
        mutated["runtime_contract_sha256"]
    )


def test_emitted_quant_config_manifest_is_revalidated():
    contract = _runtime_contract()
    result = validate_rtx4090_quant_config_manifest(
        _quant_config(contract), runtime_contract=contract
    )
    assert result["policy_id"] == RTX4090_QWEN38_POLICY_ID
    assert result["artifact_bytes"] == 18_000_000_000


def test_validation_only_manifest_is_structural_but_never_strict_release():
    contract = _runtime_contract(qualified=False)
    payload = _quant_config(_runtime_contract())
    payload["provenance"]["producer_policy"] = (
        validation_only_producer_policy_stamp(
            contract, ("FP8_CB_K44", "FP8_E4M3", "BF16")
        )
    )
    route = rtx4090_route_status_stamp(
        payload["provenance"]["producer_policy"],
        payload["provenance"]["tensor_formats"],
    )
    payload["provenance"]["cb_route_status"] = route
    with pytest.raises(
        RTX4090Qwen38PolicyError, match="UNRELEASABLE_VALIDATION_ONLY"
    ):
        validate_rtx4090_quant_config_manifest(
            payload, runtime_contract=contract
        )
    result = validate_rtx4090_quant_config_manifest(
        payload,
        runtime_contract=contract,
        allow_unreleasable_validation_only=True,
    )
    assert result["policy_id"] == RTX4090_VALIDATION_ONLY_POLICY_ID
    summary = rtx4090_route_status_summary(
        route,
        producer_policy=payload["provenance"]["producer_policy"],
        assignment=payload["provenance"]["tensor_formats"],
        where="validation-only route",
    )
    assert summary["qualification"] == "compile_only"


def test_physical_rtx4090_validator_categorically_refuses_validation_stamp(
    tmp_path,
):
    (tmp_path / "config.json").write_text(json.dumps(_model_config()))
    (tmp_path / "quant_config.json").write_text(json.dumps({
        "provenance": {
            "producer_policy": validation_only_producer_policy_stamp(
                _runtime_contract(qualified=False), ("FP8_CB_K44",)
            )
        }
    }))

    with pytest.raises(
        rtx_validator.RTX4090FP8CBValidationError,
        match="categorically ineligible",
    ):
        rtx_validator.validate_rtx4090_artifact_metadata(
            tmp_path, runtime_contract=_runtime_contract(qualified=False)
        )


def test_manifest_closes_top_level_provenance_and_source_census_fields():
    contract = _runtime_contract()

    unknown_top = _quant_config(contract)
    unknown_top["future_weight_dispatch"] = {"format": "nvfp4"}
    with pytest.raises(RTX4090Qwen38PolicyError, match="top-level keys"):
        validate_rtx4090_quant_config_manifest(
            unknown_top, runtime_contract=contract
        )

    unknown_provenance = _quant_config(contract)
    unknown_provenance["provenance"]["alternate_source_identity"] = {}
    with pytest.raises(RTX4090Qwen38PolicyError, match="unknown provenance"):
        validate_rtx4090_quant_config_manifest(
            unknown_provenance, runtime_contract=contract
        )

    with_census = _quant_config(contract)
    assignment = with_census["provenance"]["tensor_formats"]
    assignment_sha = hashlib.sha256(json.dumps(
        dict(sorted(assignment.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    with_census["provenance"]["producer_policy"]["source_census"] = {
        "schema": "prismaquant.rtx4090_qwen38_source_census.v1",
        "source_layout": "official_wrapper",
        "source_config_sha256": "1" * 64,
        "aura_staged_config_sha256": "6" * 64,
        "aura_execution_config_sha256": "2" * 64,
        "source_tensor_manifest_sha256": "3" * 64,
        "source_tensor_count": 1199,
        "source_linear_count": 615,
        "assignment_sha256": assignment_sha,
        "source_model_identity": {
            "schema": "prismaquant.streamed_model.identity.v1",
            "content_sha256": "4" * 64,
            "resolved_commit": None,
            "checkpoint_shards": 18,
            "checkpoint_tensors": 1199,
        },
    }
    validate_rtx4090_quant_config_manifest(
        with_census, runtime_contract=contract
    )
    with_census["provenance"]["producer_policy"]["source_census"][
        "alternate_config_digest"
    ] = "5" * 64
    with pytest.raises(RTX4090Qwen38PolicyError, match="source_census fields"):
        validate_rtx4090_quant_config_manifest(
            with_census, runtime_contract=contract
        )


def test_manifest_requires_exact_fp8_wire_schemes():
    contract = _runtime_contract()

    int8 = deepcopy(_quant_config(contract))
    int8["config_groups"]["group_1"]["weights"]["type"] = "int"
    with pytest.raises(RTX4090Qwen38PolicyError, match="exact dynamic E4M3"):
        validate_rtx4090_quant_config_manifest(
            int8, runtime_contract=contract
        )

    wrong_activation = deepcopy(_quant_config(contract))
    wrong_activation["config_groups"]["group_1"]["input_activations"][
        "strategy"
    ] = "tensor"
    with pytest.raises(RTX4090Qwen38PolicyError, match="exact dynamic E4M3"):
        validate_rtx4090_quant_config_manifest(
            wrong_activation, runtime_contract=contract
        )

    fp4_field = deepcopy(_quant_config(contract))
    fp4_field["config_groups"]["group_0"]["scheme"][
        "activation_contract"
    ] = "nvfp4"
    with pytest.raises(RTX4090Qwen38PolicyError, match="closed FP8 wire"):
        validate_rtx4090_quant_config_manifest(
            fp4_field, runtime_contract=contract
        )

    wrong_layout = deepcopy(_quant_config(contract))
    wrong_layout["config_groups"]["group_0"]["scheme"]["type_size"] = 96
    with pytest.raises(RTX4090Qwen38PolicyError, match="numeric layout"):
        validate_rtx4090_quant_config_manifest(
            wrong_layout, runtime_contract=contract
        )


def test_strict_build_launcher_uses_runnable_measured_frontier_and_full_imatrix():
    launcher = (
        REPO / "scripts" / "run_qwen38_rtx4090_fp8_cb_18gb.sh"
    ).read_text(encoding="utf-8")
    pipeline = (REPO / "prismaquant" / "run-pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert "export SELECTION_MODE=validated-surrogate" in launcher
    assert "export PRODUCTION_CACHE=1" in launcher
    assert "export PRODUCTION_RECACHE=0" in launcher
    assert "export CB_IMATRIX_SOURCE=probe" in launcher
    assert "export CB_ACTIVATION_SCOPE=none" in launcher
    assert "export AURA_COST_STREAMING=1" in launcher
    assert "export AURA_COST_DTYPE=bfloat16" in launcher
    assert "export PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE=" in launcher
    assert "export LM_HEAD_FORMAT=BF16" in launcher
    assert "validate_rtx4090_artifact_metadata" in launcher
    assert "validate_rtx4090_artifact(" not in launcher
    assert (
        'permits PRODUCTION_CACHE=1 only for '
        'SELECTION_MODE=validated-surrogate' in pipeline
    )
    assert 'requires PRODUCTION_RECACHE=0' in pipeline
    assert 'learned-v2 requires CB_IMATRIX_SOURCE=probe' in pipeline
    assert '--trainer-version "$trainer_version"' in pipeline
    assert '--promotion-receipt "$CB_LEARNED_PROMOTION_RECEIPT"' in pipeline
    assert '--imatrix-probe "$PROBE_PATH"' in pipeline
    assert '--checkpoint-dir "$AURA_COST_CHECKPOINT_DIR"' in pipeline


def test_validation_only_launcher_is_gb10_bound_and_delegates_exact_pipeline():
    wrapper = (
        REPO
        / "scripts"
        / "run_qwen38_rtx4090_fp8_cb_validation_only_gb10.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO / "scripts" / "run_qwen38_rtx4090_fp8_cb_18gb.sh"
    ).read_text(encoding="utf-8")

    assert "torch.cuda.device_count() != 1" in wrapper
    assert "capability != (12, 1)" in wrapper
    assert '"GB10" not in name.upper()' in wrapper
    assert "RTX4090_BUILD_DISPOSITION=validation_only" in wrapper
    assert "exec \"$PQ_REPO_ROOT/scripts/run_qwen38_rtx4090_fp8_cb_18gb.sh\"" in wrapper
    assert "qwen38_rtx4090_fp8_cb_validation_only" in launcher
    assert "require_rtx4090_compile_only_runtime_contract" in launcher
    assert "validate_rtx4090_validation_only_artifact" in launcher


@pytest.mark.parametrize(
    "mutation",
    (
        lambda q: q["provenance"]["serialized_payload"].__setitem__(
            "schema", "prismaquant.cb_serialized_payload.v3"
        ),
        lambda q: q["provenance"]["serialized_payload"]["context"].__setitem__(
            "activation_contract", "prismaquant.nvfp4_w4a4_activation.v1"
        ),
        lambda q: q["provenance"]["serialized_payload"]["context"].update(
            scale_coding="two_tier", layout_version=2
        ),
        lambda q: q["provenance"]["serialized_payload"]["context"].__setitem__(
            "scale_sweep", False
        ),
        lambda q: q["provenance"]["serialized_payload"]["context"].__setitem__(
            "renderer_abi", "unreviewed"
        ),
        lambda q: q["provenance"]["serialized_payload"]["context"].__setitem__(
            "codebook_source_scope", "fp8"
        ),
        lambda q: q["provenance"]["serialized_payload"].__setitem__(
            "fp4_scale_bytes", 1
        ),
        lambda q: q["provenance"]["serialized_payload"].__setitem__(
            "input_global_scale_bytes", 4
        ),
        lambda q: q.__setitem__("execution_contracts", {}),
    ),
)
def test_manifest_rejects_nvfp4_activation_provenance(mutation):
    contract = _runtime_contract()
    manifest = deepcopy(_quant_config(contract))
    mutation(manifest)
    with pytest.raises(RTX4090Qwen38PolicyError):
        validate_rtx4090_quant_config_manifest(
            manifest, runtime_contract=contract
        )


def test_strict_serve_launcher_selects_the_v11_environment_projection():
    launcher = (
        REPO / "scripts" / "serve_qwen38_rtx4090_fp8_cb.sh"
    ).read_text(encoding="utf-8")
    assert "--server-environment-profile rtx4090_fp8_cb" in launcher
    assert ': "${VLLM_RUNTIME_PIN:?' in launcher
    assert "--vllm-runtime-pin /vllm-runtime-pin.json" in launcher
    assert '--vllm-runtime-pin "$VLLM_RUNTIME_PIN"' in launcher
    assert "validate_rtx4090_artifact_metadata" in launcher
    assert 'PQ_WEIGHT_MOUNTS+=( -v "$MODEL_DIR/$name:/model/$name:ro" )' in launcher
    assert "rtx4090-artifact-preflight" in launcher
    assert (
        launcher.index("rtx4090-artifact-preflight")
        < launcher.index("exec /usr/local/bin/vllm serve")
    )
    assert (
        "--artifact-content-receipt "
        '"$PQ_ARTIFACT_CONTENT_RECEIPT_CONTAINER"' in launcher
    )
    assert "--rtx4090-runtime-contract /gridbook-runtime-contract.json" in launcher
    assert "--max-num-seqs 64" in launcher
    assert rtx_validator.RTX4090_MAX_NUM_SEQS == max(
        rtx_validator.RTX4090_CUDAGRAPH_CAPTURE_SIZES
    )
    assert rtx_validator.rtx4090_launch_options(
        arm="eager", served_model="test"
    )["--max-num-seqs"] == "64"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda q: q["provenance"]["tensor_formats"].__setitem__(
            "model.layers.0.mlp.down_proj", "FP8_CB_K29"
        ),
        lambda q: q["provenance"]["tensor_formats"].__setitem__(
            "model.layers.0.mlp.down_proj", "NVFP4_CB_K16"
        ),
        lambda q: q["config_groups"]["group_1"]["weights"].__setitem__(
            "num_bits", 4
        ),
        lambda q: q.__setitem__("format", "nvfp4_cb"),
        lambda q: q.__setitem__(
            "quantized_embedding", {"formats": {"model.embed_tokens": "nvfp4"}}
        ),
        lambda q: q["provenance"]["producer_policy"][
            "graph_requirement"
        ].__setitem__("vllm_compilation_mode", 0),
        lambda q: q["provenance"]["artifact_inventory"].__setitem__(
            "export_directory_bytes",
            RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES + 1,
        ),
    ),
)
def test_manifest_rejects_forbidden_formats_routes_graph_and_size(mutation):
    contract = _runtime_contract()
    manifest = deepcopy(_quant_config(contract))
    mutation(manifest)
    with pytest.raises(RTX4090Qwen38PolicyError):
        validate_rtx4090_quant_config_manifest(
            manifest, runtime_contract=contract
        )
