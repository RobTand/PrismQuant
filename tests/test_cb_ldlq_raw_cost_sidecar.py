"""Raw (no-LDLQ) cost sidecar: capture, metrics, no-op guarantee, extractor.

One LDLQ-gated cost run must yield BOTH tables: the primary gated-LDLQ
metrics and a ``*_raw_render`` sidecar measured on the pre-gate raw
assignment (the identical-env no-LDLQ render), so a no-LDLQ allocation can be
derived without a second multi-hour burn.  When LDLQ is off, nothing changes
— cost pickles stay byte-identical (the no-op guarantee).
"""

from __future__ import annotations

import pickle

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.measure_quant_cost import (
    LDLQ_RAW_SIDECAR_COST_SOURCE,
    _accumulate_result,
    _cb_raw_sidecar_metrics,
    _extrapolate_expert_costs,
    _finalize_results,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
)
from prismaquant import format_registry as fr
from tools.extract_raw_cost_table import (
    RawSidecarExtractionError,
    extract_raw_cost_payload,
    main as extract_main,
)


# ---------------------------------------------------------------------------
# Capture: cb_fields_for_context raw_fields_out
# ---------------------------------------------------------------------------

def test_raw_fields_capture_is_the_identical_no_ldlq_render():
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    col_weights = torch.rand(256, generator=generator) + 0.05
    activation_rows = torch.randn(64, 256, generator=generator)
    spec = fr.get_format("NVFP4_CB_K12")

    ldlq_context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=True)
    raw_out: dict = {}
    gated_fields = cb_fields_for_context(
        spec, weight, context=ldlq_context, col_weights=col_weights,
        activation_rows=activation_rows, raw_fields_out=raw_out,
    )
    assert raw_out.get("ldlq_applied") is True
    assert raw_out["grid"] == "fp4" and raw_out["mode"] == "product"
    assert raw_out["k"] == 12

    raw_context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=False)
    reference = cb_fields_for_context(
        spec, weight, context=raw_context, col_weights=col_weights,
    )
    assert torch.equal(raw_out["fields"]["indices"], reference["indices"])
    assert torch.equal(raw_out["fields"]["scales"], reference["scales"])
    # And the gated result is a valid same-shape assignment of the same bytes.
    assert gated_fields["indices"].shape == reference["indices"].shape


def test_raw_fields_out_untouched_when_ldlq_off():
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(4, 256, generator=generator)
    col_weights = torch.rand(256, generator=generator) + 0.05
    spec = fr.get_format("NVFP4_CB_K12")
    context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=False)
    raw_out: dict = {}
    cb_fields_for_context(
        spec, weight, context=context, col_weights=col_weights,
        raw_fields_out=raw_out,
    )
    assert raw_out == {}


def test_raw_fields_capture_respects_ldlq_scope():
    """Under scope=nvfp4 the fp8 family stays raw and gets NO sidecar."""
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(4, 256, generator=generator)
    col_weights = torch.rand(256, generator=generator) + 0.05
    spec = fr.get_format("FP8_CB_K28")
    context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=True, ldlq_scope="nvfp4")
    raw_out: dict = {}
    cb_fields_for_context(
        spec, weight, context=context, col_weights=col_weights,
        raw_fields_out=raw_out,
    )
    assert raw_out == {}


# ---------------------------------------------------------------------------
# Sidecar metrics: chunked per-expert == monolithic reconstruction
# ---------------------------------------------------------------------------

def _packed_case(seed: int = 6, experts: int = 16, rows: int = 4,
                 columns: int = 256):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(experts, rows, columns, generator=generator) * 0.25
    col_weights = torch.rand(experts, 1, columns, generator=generator) + 0.05
    fields = cb.nvfp4_cb_fields(
        weight, 12, grid="fp4", mode="product", col_weights=col_weights)
    raw_info = {
        "ldlq_applied": True, "fields": fields,
        "grid": "fp4", "mode": "product", "k": 12,
    }
    return weight, raw_info


def test_packed_sidecar_metrics_match_monolithic_reconstruction():
    weight, raw_info = _packed_case()
    h_em = torch.rand(16, 4) + 0.1
    metrics = _cb_raw_sidecar_metrics(
        weight, raw_info, h_em=h_em, want_per_expert=True)
    assert metrics is not None

    full = cb.nvfp4_cb_reconstruct(
        raw_info["fields"], 12, grid="fp4", mode="product").to(weight.dtype)
    err2 = (weight - full).float().pow(2)
    ref_per_expert = err2.mean(dim=(1, 2))
    got_per_expert = metrics["weight_mse_per_expert_raw_render"]
    # Per-expert values are identical elementwise ops on identical operands.
    assert got_per_expert == [float(v) for v in ref_per_expert]
    assert metrics["weight_mse_raw_render"] == pytest.approx(
        float(err2.mean().item()), rel=1e-6)
    ref_dloss = float(0.5 * (h_em * err2.mean(dim=-1)).sum().item())
    assert metrics["predicted_dloss_raw_render"] == pytest.approx(
        ref_dloss, rel=1e-6)


def test_packed_sidecar_sampled_scaling_matches_primary_rule():
    weight, raw_info = _packed_case()
    h_em = torch.rand(16, 4) + 0.1
    base = _cb_raw_sidecar_metrics(
        weight, raw_info, h_em=h_em, want_per_expert=False)
    scaled = _cb_raw_sidecar_metrics(
        weight, raw_info, h_em=h_em, full_expert_count=64,
        want_per_expert=False)
    assert "weight_mse_per_expert_raw_render" not in base
    assert scaled["predicted_dloss_raw_render"] == pytest.approx(
        base["predicted_dloss_raw_render"] * 4.0, rel=1e-9)


def test_dense_sidecar_metrics_match_monolithic_reconstruction():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 256, generator=generator) * 0.25
    col_weights = torch.rand(256, generator=generator) + 0.05
    fields = cb.nvfp4_cb_fields(
        weight, 12, grid="fp4", mode="product", col_weights=col_weights)
    raw_info = {
        "ldlq_applied": True, "fields": fields,
        "grid": "fp4", "mode": "product", "k": 12,
    }
    h_full = torch.rand(8, 256) + 0.1
    metrics = _cb_raw_sidecar_metrics(weight, raw_info, h_full=h_full)
    recon = cb.nvfp4_cb_reconstruct(
        fields, 12, grid="fp4", mode="product").to(weight.dtype)
    err2 = (weight - recon).float().pow(2)
    assert metrics["weight_mse_raw_render"] == float(err2.mean().item())
    assert metrics["predicted_dloss_raw_render"] == float(
        0.5 * (h_full * err2).sum().item())


def test_sidecar_metrics_none_when_ldlq_absent():
    weight, _ = _packed_case()
    assert _cb_raw_sidecar_metrics(weight, None) is None
    assert _cb_raw_sidecar_metrics(weight, {}) is None


# ---------------------------------------------------------------------------
# Accumulator / finalize: no-op guarantee and sidecar emission
# ---------------------------------------------------------------------------

def test_finalize_without_raw_render_keeps_legacy_row_schema():
    bucket: dict = {}
    explicit_none_bucket: dict = {}
    args = ("lin", "NVFP4_CB_K12", 0.5, 0.25, 0.1)
    _accumulate_result(bucket, *args, predicted_dloss=0.01)
    _accumulate_result(
        explicit_none_bucket, *args, predicted_dloss=0.01,
        score_carrier=None,
    )
    results = _finalize_results(bucket)
    explicit_none_results = _finalize_results(explicit_none_bucket)
    expected = {
        "lin": {
            "NVFP4_CB_K12": {
                "weight_mse": 0.5,
                "output_mse": 0.25,
                "rel_output_mse": 0.1,
                "predicted_dloss": 0.01,
            },
        },
    }
    assert results == expected
    assert pickle.dumps(results) == pickle.dumps(expected)
    assert pickle.dumps(results) == pickle.dumps(explicit_none_results)


def test_finalize_carries_optional_activation_scores_additively():
    score_carrier = {
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "learned": 0.125,
            "lattice": 0.25,
        },
    }
    bucket: dict = {}
    _accumulate_result(
        bucket, "lin", "NVFP4_CB_K12", 0.5, 0.25, 0.1,
        predicted_dloss=0.01,
        score_carrier=score_carrier,
    )
    # The accumulator owns a normalized copy, not the caller's nested map.
    score_carrier["activation_output_mse_by_codebook_source"]["lattice"] = 9.0
    entry = _finalize_results(bucket)["lin"]["NVFP4_CB_K12"]
    assert entry == {
        "weight_mse": 0.5,
        "output_mse": 0.25,
        "rel_output_mse": 0.1,
        "predicted_dloss": 0.01,
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "lattice": 0.25,
            "learned": 0.125,
        },
    }


def test_finalize_emits_raw_render_sidecar_fields():
    bucket: dict = {}
    _accumulate_result(
        bucket, "lin", "NVFP4_CB_K12", 0.5, 0.25, 0.1,
        predicted_dloss=0.01,
        weight_mse_per_expert=[0.5, 0.7],
        raw_render={
            "weight_mse_raw_render": 0.6,
            "predicted_dloss_raw_render": 0.02,
            "weight_mse_per_expert_raw_render": [0.6, 0.8],
        },
    )
    entry = _finalize_results(bucket)["lin"]["NVFP4_CB_K12"]
    assert entry["weight_mse"] == 0.5
    assert entry["weight_mse_raw_render"] == 0.6
    assert entry["predicted_dloss_raw_render"] == 0.02
    assert entry["weight_mse_per_expert_raw_render"] == [0.6, 0.8]


def test_extrapolated_expert_rows_carry_raw_scalars():
    results = {
        "experts.0.gate": {
            "NVFP4_CB_K12": {
                "weight_mse": 0.5, "output_mse": 0.0, "rel_output_mse": 0.0,
                "predicted_dloss": 0.01,
                "weight_mse_raw_render": 0.6,
                "predicted_dloss_raw_render": 0.02,
            },
        },
    }
    _extrapolate_expert_costs(
        results, {"experts.1.gate": ["experts.0.gate"]})
    filled = results["experts.1.gate"]["NVFP4_CB_K12"]
    assert filled["expert_cost_extrapolated"] is True
    assert filled["weight_mse_raw_render"] == 0.6
    assert filled["predicted_dloss_raw_render"] == 0.02


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _gated_payload():
    context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=True, ldlq_scope="nvfp4")
    formats = ["NVFP4_CB_K12", "FP8_CB_K28", "NVFP4"]
    stamp = cb_serialization_context_stamp(context, formats=formats)
    costs = {
        "model.layers.0.mlp.experts.gate_up_proj": {
            "NVFP4_CB_K12": {
                "weight_mse": 0.5, "output_mse": 0.3, "rel_output_mse": 0.1,
                "predicted_dloss": 0.01,
                "weight_mse_per_expert": [0.4, 0.6],
                "weight_mse_raw_render": 0.55,
                "predicted_dloss_raw_render": 0.012,
                "weight_mse_per_expert_raw_render": [0.45, 0.65],
            },
            # fp8 family is NOT LDLQ under scope=nvfp4: already raw, no
            # sidecar, must copy unchanged.
            "FP8_CB_K28": {
                "weight_mse": 0.05, "output_mse": 0.02,
                "rel_output_mse": 0.01, "predicted_dloss": 0.001,
            },
            # non-CB row: untouched.
            "NVFP4": {
                "weight_mse": 0.7, "output_mse": 0.4, "rel_output_mse": 0.2,
            },
        },
    }
    return {
        "costs": costs,
        "formats": formats,
        "provenance": {"cb_serialized_payload": stamp},
        "meta": {"model": "synthetic"},
    }


def test_extractor_swaps_ldlq_rows_and_restamps_identity():
    payload = _gated_payload()
    extracted = extract_raw_cost_payload(payload)
    row = extracted["costs"]["model.layers.0.mlp.experts.gate_up_proj"]
    swapped = row["NVFP4_CB_K12"]
    assert swapped["weight_mse"] == 0.55
    assert swapped["predicted_dloss"] == 0.012
    assert swapped["weight_mse_per_expert"] == [0.45, 0.65]
    assert swapped["output_mse"] == 0.0
    assert swapped["output_mse_measured"] is False
    assert swapped["cost_source"] == LDLQ_RAW_SIDECAR_COST_SOURCE
    assert "weight_mse_raw_render" not in swapped
    # Non-LDLQ rows copied verbatim.
    original = _gated_payload()["costs"][
        "model.layers.0.mlp.experts.gate_up_proj"]
    assert row["FP8_CB_K28"] == original["FP8_CB_K28"]
    assert row["NVFP4"] == original["NVFP4"]
    # Identity re-stamped as no-LDLQ; strict rehydration accepts it.
    stamp = extracted["provenance"]["cb_serialized_payload"]
    assert stamp["ldlq"] is False
    assert stamp["ldlq_scope"] == "none"
    assert "ldlq_packed_kernel" not in stamp
    rehydrated = cb_serialization_context_from_stamp(
        stamp, where="test-extract")
    assert rehydrated.ldlq is False and rehydrated.ldlq_scope == "none"
    assert extracted["provenance"]["derived_from_ldlq_gated_cost"][
        "source_cb_serialized_payload"]["ldlq"] is True
    # Source payload not mutated.
    assert payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"][
        "NVFP4_CB_K12"]["weight_mse"] == 0.5


def test_extractor_refuses_missing_sidecar_row():
    payload = _gated_payload()
    del payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"][
        "NVFP4_CB_K12"]["weight_mse_raw_render"]
    with pytest.raises(RawSidecarExtractionError, match="lacks the raw-render sidecar"):
        extract_raw_cost_payload(payload)


def test_extractor_refuses_error_rows_and_partial_sidecars():
    payload = _gated_payload()
    payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"][
        "NVFP4_CB_K12"] = {"error": "boom"}
    with pytest.raises(RawSidecarExtractionError, match="error row"):
        extract_raw_cost_payload(payload)
    payload = _gated_payload()
    del payload["costs"]["model.layers.0.mlp.experts.gate_up_proj"][
        "NVFP4_CB_K12"]["predicted_dloss_raw_render"]
    with pytest.raises(RawSidecarExtractionError, match="predicted_dloss_raw_render"):
        extract_raw_cost_payload(payload)


def test_extractor_refuses_already_raw_payload():
    context = CBSerializationContext.production(
        encode_tier="balanced", ldlq=False)
    payload = _gated_payload()
    payload["provenance"]["cb_serialized_payload"] = (
        cb_serialization_context_stamp(
            context, formats=payload["formats"])
    )
    with pytest.raises(RawSidecarExtractionError, match="already IS a raw"):
        extract_raw_cost_payload(payload)


def test_extractor_cli_roundtrip(tmp_path):
    src = tmp_path / "gated_cost.pkl"
    dst = tmp_path / "raw_cost.pkl"
    with open(src, "wb") as fh:
        pickle.dump(_gated_payload(), fh)
    assert extract_main([str(src), str(dst)]) == 0
    with open(dst, "rb") as fh:
        extracted = pickle.load(fh)
    assert extracted["provenance"]["cb_serialized_payload"]["ldlq"] is False
    assert extracted["meta"]["ldlq_raw_render_extraction"][
        "rows_swapped_to_raw_sidecar"] == 1
    # Refusal path exits 1 without writing.
    bad = _gated_payload()
    del bad["costs"]["model.layers.0.mlp.experts.gate_up_proj"][
        "NVFP4_CB_K12"]["weight_mse_raw_render"]
    src2 = tmp_path / "bad_cost.pkl"
    with open(src2, "wb") as fh:
        pickle.dump(bad, fh)
    dst2 = tmp_path / "bad_out.pkl"
    assert extract_main([str(src2), str(dst2)]) == 1
    assert not dst2.exists()


# ---------------------------------------------------------------------------
# End-to-end through the cost render wrapper (packed, CPU)
# ---------------------------------------------------------------------------

def test_cost_render_wrapper_captures_raw_render(monkeypatch):
    from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize

    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_SCOPE", "nvfp4")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "holdout")
    monkeypatch.delenv("PRISMAQUANT_CB_WARM_STATE_DIR", raising=False)
    monkeypatch.delenv("CB_CODEBOOK_DIGESTS", raising=False)
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(16, 4, 256, generator=generator) * 0.25
    col_weights = torch.rand(16, 1, 256, generator=generator) + 0.05
    activations = tuple(
        torch.randn(32, 256, generator=generator) for _ in range(16)
    )
    spec = fr.get_format("NVFP4_CB_K12")
    raw_out: dict = {}
    w_hat = _cb_cost_quantize_dequantize(
        spec, weight, col_weights=col_weights, qname="synthetic",
        activation_rows=activations, raw_render_out=raw_out,
    )
    assert w_hat.shape == weight.shape
    assert raw_out.get("ldlq_applied") is True
    metrics = _cb_raw_sidecar_metrics(weight, raw_out, want_per_expert=True)
    assert metrics is not None
    assert len(metrics["weight_mse_per_expert_raw_render"]) == 16
    assert metrics["weight_mse_raw_render"] > 0.0
