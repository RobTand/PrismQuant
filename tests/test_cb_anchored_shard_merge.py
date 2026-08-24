import copy
import hashlib
import pickle

import pytest
import torch

from prismaquant.anchored_cost import RenderRequest, SegmentKey
from prismaquant.cb_anchored_cost import (
    AnchoredCostError,
    anchors_from_streamed_payload,
    merge_streamed_cb_anchor_aura_shards,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.nvfp4_cb_footprint import CBSerializationContext
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
)


def _identity(scope, col_weights, context, *, formats):
    value = build_production_cache_cb_render_identity(
        {name: list(formats) for name in scope},
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    return bind_cb_render_identity_source_weights(
        value,
        {
            name: torch.arange(512, dtype=torch.float32).reshape(2, 256)
            + index
            for index, name in enumerate(scope)
        },
    )


def _payload(name, col_weights, context):
    sparse = _identity(
        [name], col_weights, context, formats=["FP8_CB_K28"]
    )
    expanded = _identity(
        [name], col_weights, context,
        formats=["FP8_CB_K28", "FP8_CB_K32"],
    )
    source_records = {
        name: {
            "shape": sparse["source_weights_shapes"][name],
            "sha256": sparse["source_weights_content_sha256"][name],
        }
    }
    renderer = {
        "schema": "prismaquant.production_anchor_renderer_identity.v1",
        "formats_by_qname": {name: ["FP8_CB_K28"]},
        "requested_entries": 1,
        "calibration_hash": "calibration",
        "max_act_rows": 512,
        "arm_identity": {"arm": "production"},
        "source_model": {"content_sha256": "model"},
        "source_weight_binding": "complete_streamed_model_content_identity",
        "cold_expert_provenance": {},
        "cb_render_identity": sparse,
        "producer_git_commit": "a" * 40,
        "producer_source_sha256": "b" * 64,
        "retention": "one_render_or_explicit_layer_mapping",
        "transient_consumer_identity": {"consumer": "aura"},
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": source_records,
            "identity_sha256": canonical_json_sha256(
                source_records, where="test source records"
            ),
        },
    }
    return {
        "schema": "prismaquant.aura_cost.v1",
        "n_probes": 4,
        "formats": ["FP8_CB_K28", "BF16"],
        "token_scope": "all",
        "stats": {name: {"n_params": 512}},
        "costs": {
            name: {
                "FP8_CB_K28": {
                    "predicted_dloss": 0.25,
                    "dw_source": "production_render",
                    "production_anchor_measured": True,
                }
            }
        },
        "provenance": {
            "seed_base": 7000,
            "temperature": 1.0,
            "dw_dtype": "bfloat16",
            "measurement_dtype": "torch.bfloat16",
            "include_lm_head": False,
            "n_linear_chunks": 1,
            "calib_shape": [1, 8],
            "calib_sha256": hashlib.sha256(b"calib").hexdigest(),
            "calib_hash": "calibration",
            "calib_hashes": ["calibration"],
            "omitted_packed_experts": [],
            "dw_rendered_rows": 0,
            "dw_production_anchor_rows": 1,
            "dw_rtn_fallback_rows": 0,
            "git_commit": "a" * 40,
            "streaming": True,
            "streamed_gradient_harvest": "post_accumulate_per_parameter",
            "streamed_cotangent_rollover": "in_place_per_probe",
            "streamed_boundary_release": "progressive_reverse",
            "cb_cost_provenance_schema": "test",
            "production_anchor_renderer": renderer,
            "production_anchor_render_purposes": {
                name: {"FP8_CB_K28": ["anchor"]}
            },
            "production_anchor_unmeasured_formats_by_qname": {
                name: ["BF16"]
            },
            "production_anchor_purpose_counts": {
                "anchor": 1, "panel": 0, "validation": 0,
            },
            "production_anchor_union_render_count": 1,
            "production_anchor_expected_renders": 1,
            "production_anchor_rendered_this_invocation": 1,
            "production_anchor_restored_renders": 0,
            "production_anchor_max_live_rendered": 1,
            "production_anchor_sparse_render_identity": sparse,
            "production_anchor_sparse_serialized_payload": sparse[
                "cb_serialized_payload"
            ],
            "cb_render_identity": expanded,
            "cb_serialized_payload": expanded["cb_serialized_payload"],
            "cb_anchored_plugin": {"aura_only_cost_currency": True},
            "full_menu_materialized": False,
        },
    }


def _sparse_fp8_ladder_payload(name, col_weights, context):
    measured_formats = ("FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48")
    legal_formats = tuple(f"FP8_CB_K{k}" for k in range(4, 49, 4))
    payload = _payload(name, col_weights, context)
    sparse = _identity(
        [name], col_weights, context, formats=measured_formats
    )
    expanded = _identity(
        [name], col_weights, context, formats=legal_formats
    )
    source_records = {
        name: {
            "shape": sparse["source_weights_shapes"][name],
            "sha256": sparse["source_weights_content_sha256"][name],
        }
    }
    renderer = payload["provenance"]["production_anchor_renderer"]
    renderer.update({
        "formats_by_qname": {name: list(measured_formats)},
        "requested_entries": len(measured_formats),
        "cb_render_identity": sparse,
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": source_records,
            "identity_sha256": canonical_json_sha256(
                source_records, where="test sparse ladder source records"
            ),
        },
    })
    payload.update({
        "formats": [*measured_formats, "FP8_E4M3"],
        "costs": {
            name: {
                format_name: {
                    "predicted_dloss": float(index + 1) / 10.0,
                    "dw_source": "production_render",
                    "production_anchor_measured": True,
                }
                for index, format_name in enumerate(measured_formats)
            }
        },
    })
    payload["provenance"].update({
        "production_anchor_render_purposes": {
            name: {
                "FP8_CB_K4": ["anchor"],
                "FP8_CB_K16": ["panel"],
                "FP8_CB_K48": ["panel"],
            }
        },
        "production_anchor_unmeasured_formats_by_qname": {
            name: ["FP8_E4M3"]
        },
        "production_anchor_purpose_counts": {
            "anchor": 1,
            "panel": 2,
            "validation": 0,
        },
        "production_anchor_union_render_count": len(measured_formats),
        "production_anchor_expected_renders": len(measured_formats),
        "production_anchor_rendered_this_invocation": len(measured_formats),
        "production_anchor_sparse_render_identity": sparse,
        "production_anchor_sparse_serialized_payload": sparse[
            "cb_serialized_payload"
        ],
        "cb_render_identity": expanded,
        "cb_serialized_payload": expanded["cb_serialized_payload"],
    })
    return payload


def test_streamed_anchor_merge_reconstructs_global_receipt_identity():
    context = CBSerializationContext.production(
        codebook_source="lattice", codebook_source_scope="none"
    )
    col_weights = {
        "layer.0.proj": torch.arange(256, dtype=torch.float32),
        "layer.1.proj": torch.arange(256, dtype=torch.float32) + 1,
    }
    shards = [
        _payload("layer.0.proj", col_weights, context),
        _payload("layer.1.proj", col_weights, context),
    ]
    merged = merge_streamed_cb_anchor_aura_shards(
        shards,
        col_weights=col_weights,
        expected_qnames=tuple(col_weights),
        expected_formats_by_qname={
            name: ("FP8_CB_K28",) for name in col_weights
        },
        expected_purposes_by_qname={
            name: {"FP8_CB_K28": ("anchor",)} for name in col_weights
        },
        expected_unmeasured_formats_by_qname={
            name: ("BF16",) for name in col_weights
        },
        expected_legal_cb_formats_by_qname={
            name: ("FP8_CB_K28", "FP8_CB_K32") for name in col_weights
        },
    )
    reversed_merge = merge_streamed_cb_anchor_aura_shards(
        list(reversed(shards)),
        col_weights=col_weights,
        expected_qnames=tuple(col_weights),
    )

    provenance = merged["provenance"]
    assert set(merged["costs"]) == set(col_weights)
    assert set(
        provenance["production_anchor_renderer"]["formats_by_qname"]
    ) == set(col_weights)
    assert set(
        provenance["production_anchor_sparse_render_identity"]
        ["cb_formats_by_qname"]
    ) == set(col_weights)
    assert provenance["streamed_shard_merge"]["exact_disjoint_cover"] is True
    assert pickle.dumps(merged) == pickle.dumps(reversed_merge)

    segment = SegmentKey("fp8_cb", "proj", "lattice")
    requests = tuple(
        RenderRequest(name, segment, "FP8_CB_K28", "anchor")
        for name in sorted(col_weights)
    )
    anchors = anchors_from_streamed_payload(requests, merged)
    receipt_hashes = {
        value.receipt.payload_identity_sha256 for value in anchors.values()
    }
    assert len(receipt_hashes) == 1


def test_streamed_anchor_merge_accepts_numeric_full_fp8_ladder_plan():
    context = CBSerializationContext.production(
        codebook_source="lattice", codebook_source_scope="none"
    )
    col_weights = {
        "layer.0.proj": torch.arange(256, dtype=torch.float32),
        "layer.1.proj": torch.arange(256, dtype=torch.float32) + 1,
    }
    measured_formats = ("FP8_CB_K4", "FP8_CB_K16", "FP8_CB_K48")
    legal_formats = tuple(f"FP8_CB_K{k}" for k in range(4, 49, 4))
    merged = merge_streamed_cb_anchor_aura_shards(
        [
            _sparse_fp8_ladder_payload(name, col_weights, context)
            for name in col_weights
        ],
        col_weights=col_weights,
        expected_qnames=tuple(col_weights),
        expected_formats_by_qname={
            name: measured_formats for name in col_weights
        },
        expected_purposes_by_qname={
            name: {
                "FP8_CB_K4": ("anchor",),
                "FP8_CB_K16": ("panel",),
                "FP8_CB_K48": ("panel",),
            }
            for name in col_weights
        },
        expected_unmeasured_formats_by_qname={
            name: ("FP8_E4M3",) for name in col_weights
        },
        # Producer order is numeric. The persisted CB identity canonicalizes
        # this unordered legal domain lexicographically.
        expected_legal_cb_formats_by_qname={
            name: legal_formats for name in col_weights
        },
    )

    provenance = merged["provenance"]
    assert {
        format_name
        for formats in provenance["production_anchor_renderer"]
        ["formats_by_qname"].values()
        for format_name in formats
    } == set(measured_formats)
    assert {
        format_name
        for formats in provenance["cb_render_identity"]
        ["cb_formats_by_qname"].values()
        for format_name in formats
    } == set(legal_formats)


def test_streamed_anchor_merge_refuses_overlap_and_identity_drift():
    context = CBSerializationContext.production(
        codebook_source="lattice", codebook_source_scope="none"
    )
    col_weights = {
        "layer.0.proj": torch.arange(256, dtype=torch.float32),
        "layer.1.proj": torch.arange(256, dtype=torch.float32) + 1,
    }
    first = _payload("layer.0.proj", col_weights, context)
    with pytest.raises(AnchoredCostError, match="overlap"):
        merge_streamed_cb_anchor_aura_shards(
            [first, copy.deepcopy(first)], col_weights=col_weights
        )

    second = _payload("layer.1.proj", col_weights, context)
    second["provenance"]["calib_hash"] = "different"
    with pytest.raises(AnchoredCostError, match="calib_hash"):
        merge_streamed_cb_anchor_aura_shards(
            [first, second], col_weights=col_weights
        )

    second = _payload("layer.1.proj", col_weights, context)
    second["provenance"]["production_anchor_renderer"]["source_weights"][
        "identity_sha256"
    ] = "0" * 64
    with pytest.raises(AnchoredCostError, match="source-weight identity"):
        merge_streamed_cb_anchor_aura_shards(
            [first, second], col_weights=col_weights
        )

    second = _payload("layer.1.proj", col_weights, context)
    with pytest.raises(AnchoredCostError, match="rendered format plan"):
        merge_streamed_cb_anchor_aura_shards(
            [first, second],
            col_weights=col_weights,
            expected_formats_by_qname={
                name: ("FP8_CB_K28", "FP8_E4M3")
                for name in col_weights
            },
        )
