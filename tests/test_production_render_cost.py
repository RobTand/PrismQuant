from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import build_candidates
from prismaquant.production_render_cost import (
    synthesize_production_render_cost_payload,
)
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
)


def _cache_with_scores() -> ProductionWeightCache:
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True, "joint_scale_opt": True},
        metadata={
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "records": {
                    "layers.0.q_proj|NVFP4": {
                        "qname": "layers.0.q_proj",
                        "format": "NVFP4",
                        "metric": "output_mse",
                        "score": 0.5,
                        "score_sum": 12.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                    "layers.0.q_proj|MXFP8_E4M3": {
                        "qname": "layers.0.q_proj",
                        "format": "MXFP8_E4M3",
                        "metric": "output_mse",
                        "score": 0.25,
                        "score_sum": 6.0,
                        "normalizer": 24.0,
                        "activation_rows": 3,
                    },
                },
            },
        },
    )


def test_production_render_cost_uses_render_score_directly():
    baseline = {
        "formats": ["NVFP4", "MXFP8_E4M3", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "MXFP8_E4M3": {"output_mse": 88.0},
                "BF16": {"predicted_dloss": 0.0},
            },
            "layers.0.o_proj": {
                "NVFP4": {"predicted_dloss": 4.0},
                "MXFP8_E4M3": {"predicted_dloss": 2.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )

    q = cost["costs"]["layers.0.q_proj"]
    assert q["NVFP4"]["predicted_dloss"] == 12.0
    assert q["NVFP4"]["output_mse_measured"] is False
    assert q["NVFP4"]["cost_source"] == "production_render_score"
    assert "activation_output_mse" not in q["NVFP4"]
    assert "activation_output_mse_by_codebook_source" not in q["NVFP4"]
    assert q["MXFP8_E4M3"]["predicted_dloss"] == 6.0
    assert q["BF16"]["predicted_dloss"] == 0.0

    o = cost["costs"]["layers.0.o_proj"]
    assert o["NVFP4"]["predicted_dloss"] == 4.0
    assert o["NVFP4"]["cost_source"] == "fallback_baseline"
    assert cost["meta"]["render_score_entries"] == 2
    assert cost["meta"]["fallback_entries"] == 2


def test_production_render_cost_preserves_optional_cell_scores_without_repricing():
    cache = _cache_with_scores()
    record = cache.metadata["render_scores"]["records"][
        "layers.0.q_proj|NVFP4"
    ]
    record.update({
        "activation_output_mse": 0.5,
        "activation_output_mse_by_codebook_source": {
            "lattice": 0.5,
            "learned": 0.375,
        },
    })
    baseline = {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }

    payload = synthesize_production_render_cost_payload(
        cache,
        baseline,
        score_field="output_mse",
    )
    entry = payload["costs"]["layers.0.q_proj"]["NVFP4"]
    assert entry["activation_output_mse"] == 0.5
    assert entry["activation_output_mse_by_codebook_source"] == {
        "lattice": 0.5,
        "learned": 0.375,
    }

    stats = {
        "layers.0.q_proj": {
            "h_trace": 2.0,
            "out_features": 32,
            "in_features": 32,
            "n_params": 1024,
        },
    }
    enriched = build_candidates(
        stats,
        payload["costs"],
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
    )
    legacy_costs = {
        name: {
            fmt: {
                key: value
                for key, value in row.items()
                if key not in {
                    "activation_output_mse",
                    "activation_output_mse_by_codebook_source",
                }
            }
            for fmt, row in formats.items()
        }
        for name, formats in payload["costs"].items()
    }
    legacy = build_candidates(
        stats,
        legacy_costs,
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
    )
    assert enriched == legacy


def test_production_render_cost_bypasses_h_trace_proxy_in_allocator():
    baseline = {
        "formats": ["NVFP4", "BF16"],
        "costs": {
            "layers.0.q_proj": {
                "NVFP4": {"output_mse": 99.0},
                "BF16": {"predicted_dloss": 0.0},
            },
        },
    }
    cost = synthesize_production_render_cost_payload(
        _cache_with_scores(),
        baseline,
    )
    stats = {
        "layers.0.q_proj": {
            "h_trace": 1000.0,
            "out_features": 32,
            "in_features": 32,
            "n_params": 1024,
        },
    }

    candidates = build_candidates(
        stats,
        cost["costs"],
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
    )
    by_fmt = {cand.fmt: cand for cand in candidates["layers.0.q_proj"]}

    assert by_fmt["NVFP4"].predicted_dloss == 12.0
    assert by_fmt["BF16"].predicted_dloss == 0.0


def test_production_render_cost_can_reject_weight_mse_fallbacks():
    cache = _cache_with_scores()
    cache.metadata["render_scores"]["records"]["layers.0.q_proj|NVFP4"][
        "metric"
    ] = "weight_mse"
    baseline = {
        "formats": ["NVFP4"],
        "costs": {"layers.0.q_proj": {"NVFP4": {"predicted_dloss": 4.0}}},
    }

    with pytest.raises(ValueError, match="non-output metrics"):
        synthesize_production_render_cost_payload(
            cache,
            baseline,
            require_output_metric=True,
        )


# --------------------------------------------------------------------------
# R14 — calibration identity propagation
# --------------------------------------------------------------------------

def test_render_cost_inherits_calib_hash_from_the_cache():
    cache = _cache_with_scores()
    cache.metadata["calib_hash"] = "cachehash"
    payload = synthesize_production_render_cost_payload(
        cache, {"costs": {}, "formats": ["NVFP4"]})
    assert payload["meta"]["calib_hashes"] == ["cachehash"]
    assert payload["meta"]["calib_hash"] == "cachehash"


def test_render_cost_unions_cache_and_baseline_hashes():
    cache = _cache_with_scores()
    cache.metadata["calib_hash"] = "cachehash"
    payload = synthesize_production_render_cost_payload(
        cache,
        {"costs": {}, "formats": ["NVFP4"],
         "meta": {"calib_hashes": ["baselinehash"]}},
    )
    assert payload["meta"]["calib_hashes"] == ["baselinehash", "cachehash"]
    # Ambiguous single-draw identity -> None, so a downstream reader cannot
    # mistake a two-draw cost table for one draw.
    assert payload["meta"]["calib_hash"] is None


def test_render_cost_stays_inert_on_pre_r14_artifacts():
    payload = synthesize_production_render_cost_payload(
        _cache_with_scores(), {"costs": {}, "formats": ["NVFP4"]})
    assert payload["meta"]["calib_hashes"] == []
    assert payload["meta"]["calib_hash"] is None


def test_render_cost_carries_stored_cb_context_not_current_env(monkeypatch):
    from prismaquant.nvfp4_cb_footprint import (
        CBSerializationContext,
        cb_serialization_context_stamp,
    )

    qname = "layers.0.q_proj"
    fmt = "NVFP4_CB_K16"
    context = CBSerializationContext.legacy_v1()
    col_weights = {qname: torch.linspace(0.1, 1.0, 256)}
    identity = build_production_cache_cb_render_identity(
        {qname: (fmt,)},
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(
        identity,
        {qname: torch.zeros(2, 256)},
    )
    cache = ProductionWeightCache(
        weights={(qname, fmt): torch.ones(2, 256)},
        levers={"weighted_vq": True},
        metadata={
            "cb_render_identity": identity,
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "records": {
                    f"{qname}|{fmt}": {
                        "qname": qname,
                        "format": fmt,
                        "metric": "output_mse",
                        "score": 0.25,
                        "score_sum": 2.0,
                        "normalizer": 8.0,
                        "activation_rows": 2,
                    },
                },
            },
        },
    )
    baseline = {
        "formats": [fmt],
        "costs": {qname: {fmt: {"predicted_dloss": 9.0}}},
        "provenance": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                context,
                formats=[fmt],
            ),
        },
    }
    # A consumer launched under today's v2 defaults must not relabel a v1
    # render cache as v2.
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")

    payload = synthesize_production_render_cost_payload(cache, baseline)

    assert payload["provenance"]["cb_serialized_payload"] == (
        identity["cb_serialized_payload"]
    )
    assert payload["provenance"]["cb_render_identity"] == identity


def test_render_cost_rejects_legacy_cb_cache_without_identity():
    fmt = "NVFP4_CB_K16"
    qname = "layers.0.q_proj"
    cache = ProductionWeightCache(
        weights={(qname, fmt): torch.ones(2, 256)},
        levers={},
        metadata={"render_scores": {"records": {}}},
    )
    baseline = {
        "formats": [fmt],
        "costs": {qname: {fmt: {"predicted_dloss": 9.0}}},
    }

    with pytest.raises(ValueError, match="legacy or partially resumed"):
        synthesize_production_render_cost_payload(cache, baseline)


def test_render_cost_does_not_relabel_stale_cb_score_without_cache_tensor():
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    qname = "layers.0.q_proj"
    fmt = "NVFP4_CB_K16"
    source = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    identity = build_production_cache_cb_render_identity(
        {qname: [fmt]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights={qname: torch.ones(256)},
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(
        identity,
        {qname: source},
    )
    cache = ProductionWeightCache(
        # The score exists, but the corresponding fresh rendered tensor does
        # not. It must not be admitted under the otherwise-valid identity.
        weights={},
        levers={"weighted_vq": True},
        metadata={
            "cb_render_identity": identity,
            "render_scores": {
                "records": {
                    f"{qname}|{fmt}": {
                        "qname": qname,
                        "format": fmt,
                        "metric": "output_mse",
                        "score": 0.25,
                        "score_sum": 2.0,
                        "normalizer": 8.0,
                    },
                },
            },
        },
    )
    baseline = {
        "formats": [fmt],
        "costs": {qname: {fmt: {"predicted_dloss": 9.0}}},
        "provenance": {
            "cb_serialized_payload": identity["cb_serialized_payload"],
            "cb_render_identity": identity,
        },
    }

    payload = synthesize_production_render_cost_payload(cache, baseline)

    entry = payload["costs"][qname][fmt]
    assert entry["predicted_dloss"] == 9.0
    assert entry["cost_source"] == "fallback_baseline"
    assert payload["meta"]["render_score_entries"] == 0


def test_render_cost_rejects_cb_fallback_from_different_imatrix():
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    qname = "layers.0.q_proj"
    fmt = "NVFP4_CB_K16"
    context = CBSerializationContext.production()
    source = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    col_a = {qname: torch.ones(256)}
    col_b = {qname: torch.ones(256) * 2}

    def identity(col_weights):
        value = build_production_cache_cb_render_identity(
            {qname: [fmt]},
            cb_serialization_context=context,
            col_weights=col_weights,
            render_levers={"weighted_vq": True},
            render_mechanism_plan=[],
        )
        return bind_cb_render_identity_source_weights(
            value,
            {qname: source},
        )

    cache_identity = identity(col_b)
    baseline_identity = identity(col_a)
    cache = ProductionWeightCache(
        weights={(qname, fmt): source.clone()},
        levers={"weighted_vq": True},
        metadata={
            "cb_render_identity": cache_identity,
            "render_scores": {"records": {}},
        },
    )
    baseline = {
        "formats": [fmt],
        "costs": {qname: {fmt: {"predicted_dloss": 1.0}}},
        "provenance": {
            "cb_serialized_payload": baseline_identity[
                "cb_serialized_payload"
            ],
            "cb_render_identity": baseline_identity,
        },
    }

    with pytest.raises(ValueError, match="imatrix values differ"):
        synthesize_production_render_cost_payload(cache, baseline)
