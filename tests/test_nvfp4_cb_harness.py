"""Phase-0 NVFP4-CB measurement-harness tests.

Covers the three harness modules authored for Phase 0:
  * nvfp4_cb_footprint  — sidecar-aware byte accountant (§1.2 table)
  * index_entropy       — index-stream entropy / redundancy
  * emu_forward_kl      — whole-model emulated forward KL-vs-BF16 (GPU)

The payload tests cross-check the accounting formulas against the codebook
tensors the real exporter materializes, including every registered product
rung.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from prismaquant.cb_layout import (
    FP8_ACCEPTED_RUNGS,
    FP8_PRODUCT_RUNGS,
    NVFP4_PRODUCT_RUNGS,
)

from prismaquant.index_entropy import index_entropy
from prismaquant.nvfp4_cb_footprint import (
    CB_SERIALIZED_PAYLOAD_SCHEMA,
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_footprint,
    cb_serialization_context_stamp,
    cb_tensor_payload_breakdown,
    codebook_sidecar_payload_bytes,
    codebook_subtable_shapes,
    validate_cb_sidecar_tensors,
    validate_cb_serialization_context_stamp,
)

# ---------------------------------------------------------------------------
# Footprint — exact producer payload (v2 default; explicit v1 compatibility)
# ---------------------------------------------------------------------------


def _learned_context(fmt: str, role: str = "w") -> CBSerializationContext:
    count = len(codebook_subtable_shapes(fmt))
    base = f"cb_codebook.{role}.{fmt}"
    refs = [base] if count == 1 else [
        f"{base}.sub{index}" for index in range(count)
    ]
    return CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={
            ref: f"{index + 1:064x}" for index, ref in enumerate(refs)
        },
    )

@pytest.mark.parametrize("k", NVFP4_PRODUCT_RUNGS)
def test_footprint_production_fp4_v2_is_4k_plus_9(k):
    shape = (128, 256)
    n = shape[0] * shape[1]
    fmt = f"NVFP4_CB_K{k}"
    fp = cb_footprint({"w": fmt}, {"w": shape})
    expected_type_size = 4 * k + 9
    expected_body_bytes = (n // 256) * expected_type_size
    assert fp["body_bytes"] == expected_body_bytes
    assert fp["body_bpw"] == pytest.approx(
        expected_type_size * 8 / 256, abs=1e-9
    )
    assert fp["per_tensor"]["w"]["k"] == k
    assert fp["sidecar_bytes"] == codebook_sidecar_payload_bytes(fmt)
    assert fp["global_scale_bytes"] == 0
    assert fp["schema"] == CB_SERIALIZED_PAYLOAD_SCHEMA
    assert fp["serialization_context"]["layout_version"] == 2


def test_legacy_fp4_v1_read_accounting_is_explicit_4k_plus_16():
    shape = (128, 256)
    fmt = "NVFP4_CB_K16"
    fp = cb_footprint(
        {"w": fmt},
        {"w": shape},
        context=CBSerializationContext.legacy_v1(),
    )
    assert fp["body_bytes"] == shape[0] * (4 * 16 + 16)
    assert fp["serialization_context"] == {
        "scale_coding": "v1",
        "layout_version": 1,
        "codebook_source": "lattice",
        "scale_sweep": True,
        "ldlq": False,
        "encode_tier": "balanced",
        "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
    }
    assert fp["global_scale_bytes"] == 0


def test_lattice_and_learned_product_codebooks_have_real_fp16_sidecars():
    k = 16
    shape = (256, 256)
    fmt = f"NVFP4_CB_K{k}"
    base = cb_footprint({"w": fmt}, {"w": shape})
    learned = cb_footprint(
        {"w": fmt}, {"w": shape}, context=_learned_context(fmt))
    # k16 product codebook = two (2^8, 4) FP16 subtables = 4096 B.
    expected_sidecar = 2 * (1 << 8) * 4 * 2
    assert codebook_subtable_shapes(fmt) == ((256, 4), (256, 4))
    assert base["sidecar_bytes"] == expected_sidecar
    assert learned["sidecar_bytes"] == expected_sidecar
    assert learned["total_bytes"] == base["total_bytes"]
    # Physical byte size is the same; sharing identity is not.
    assert base["sidecars"][0]["codebook_source"] == "lattice"
    assert learned["sidecars"][0]["codebook_source"] == "learned"
    assert learned["body_bpw"] == pytest.approx(base["body_bpw"], abs=1e-9)
    assert learned["total_bpw"] > learned["body_bpw"]


def test_footprint_shared_codebook_charged_once():
    k = 12
    shape = (256, 256)
    fmt = f"NVFP4_CB_K{k}"
    count = len(codebook_subtable_shapes(fmt))
    refs = [f"cb_codebook.role0.{fmt}.sub{index}" for index in range(count)]
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_refs={name: refs for name in ("a", "b")},
        codebook_content_digests={
            ref: f"{index + 1:064x}" for index, ref in enumerate(refs)
        },
    )
    fp = cb_footprint(
        {"a": fmt, "b": fmt}, {"a": shape, "b": shape}, context=context)
    # One shared codebook for both tensors → charged once.
    assert fp["sidecar_bytes"] == codebook_sidecar_payload_bytes(fmt) == 1024


@pytest.mark.parametrize("k", FP8_PRODUCT_RUNGS)
def test_footprint_fp8_cb_bpw_exact(k):
    out_f, in_f = 128, 256
    n = out_f * in_f
    fmt = f"FP8_CB_K{k}"
    fp = cb_footprint({"w": fmt}, {"w": (out_f, in_f)})
    # Shipped bytes = k/8 bpw index stream + per-output-channel fp32 scales,
    # counted exactly once whether the registered spec folds the plane into
    # its body (group_size=0/scale_bits=32) or the fallback charges it under
    # channel_scale_bytes.
    tensor_payload = n * k // 64 + 4 * out_f
    sidecar = codebook_sidecar_payload_bytes(fmt)
    assert fp["body_bytes"] + fp["channel_scale_bytes"] == tensor_payload
    assert fp["total_bytes"] == tensor_payload + sidecar
    assert fp["global_scale_bytes"] == 0
    assert fp["sidecar_bytes"] == sidecar
    # Total bpw also includes the once-per-artifact FP16 product subtables.
    assert fp["total_bpw"] == pytest.approx(
        8.0 * (tensor_payload + sidecar) / n, abs=1e-12)
    assert fp["per_tensor"]["w"]["cb_family"] == "fp8"
    assert fp["per_tensor"]["w"]["k"] == k


def test_fp8_cb_codebook_is_four_fp16_subtables_for_both_sources():
    k = 36
    out_f, in_f = 64, 256
    fmt = f"FP8_CB_K{k}"
    base = cb_footprint({"w": fmt}, {"w": (out_f, in_f)})
    learned = cb_footprint(
        {"w": fmt}, {"w": (out_f, in_f)}, context=_learned_context(fmt))
    expected_sidecar = 4 * (1 << 9) * 2 * 2
    assert codebook_subtable_shapes(fmt) == ((512, 2),) * 4
    assert base["sidecar_bytes"] == expected_sidecar
    assert learned["sidecar_bytes"] == expected_sidecar
    assert learned["total_bytes"] == base["total_bytes"]


def test_odd_product_k_splits_larger_subtables_first():
    assert codebook_subtable_shapes("NVFP4_CB_K13") == (
        (128, 4),
        (64, 4),
    )


_REGISTERED_CB_FORMATS = (
    [f"NVFP4_CB_K{k}" for k in NVFP4_PRODUCT_RUNGS]
    + [f"FP8_CB_K{k}" for k in FP8_ACCEPTED_RUNGS]
)


@pytest.mark.parametrize(
    ("fmt", "shapes", "sidecar_bytes"),
    [
        ("NVFP4_CB_K12", ((64, 4), (64, 4)), 1_024),
        ("NVFP4_CB_K24", ((4096, 4), (4096, 4)), 65_536),
    ],
)
def test_nvfp4_endpoint_sidecar_geometry(fmt, shapes, sidecar_bytes):
    assert codebook_subtable_shapes(fmt) == shapes
    assert codebook_sidecar_payload_bytes(fmt) == sidecar_bytes


@pytest.mark.parametrize("fmt", ["NVFP4_CB_K1", "NVFP4_CB_K25"])
def test_direct_nvfp4_research_width_has_no_public_footprint_identity(fmt):
    with pytest.raises(ValueError, match="is not a CB format"):
        codebook_subtable_shapes(fmt)


@pytest.mark.parametrize("k", [29, 47])
def test_legacy_fp8_cb_footprint_remains_exact_but_not_producible(k):
    from prismaquant.format_registry import format_is_producer_eligible

    out_f, in_f = 32, 512
    fmt = f"FP8_CB_K{k}"
    payload = cb_tensor_payload_breakdown(
        fmt,
        (out_f, in_f),
        qname="legacy.weight",
        context=CBSerializationContext.production(),
    )
    expected_index = out_f * (in_f // 256) * 4 * k
    assert payload["index_bytes"] == expected_index
    assert payload["fp8_row_scale_bytes"] == 4 * out_f
    assert payload["tensor_payload_bytes"] == expected_index + 4 * out_f
    assert payload["sidecar_identity"]["payload_bytes"] == (
        codebook_sidecar_payload_bytes(fmt)
    )
    assert not format_is_producer_eligible(fmt)


@pytest.mark.parametrize("fmt", _REGISTERED_CB_FORMATS)
def test_sidecar_formula_matches_exporter_tensor_shapes_and_bytes(fmt):
    from prismaquant import nvfp4_cb_formats as cb
    from prismaquant.export_nvfp4_cb import _codebook_tensors, _parse_cb_format

    grid, mode, k = _parse_cb_format(fmt)
    codebook = cb._resolve_codebook(
        k, grid, mode, None, torch.device("cpu")
    )
    blobs = _codebook_tensors("lattice", fmt, codebook)
    assert tuple(tuple(blob.shape) for blob in blobs.values()) == (
        codebook_subtable_shapes(fmt)
    )
    assert all(blob.dtype == torch.float16 for blob in blobs.values())
    actual_bytes = sum(
        blob.numel() * blob.element_size() for blob in blobs.values()
    )
    assert actual_bytes == codebook_sidecar_payload_bytes(fmt)
    payload = cb_assignment_payload_breakdown(
        {"layer.w": fmt},
        {"layer.w": (64, 256)},
        context=CBSerializationContext.production(),
    )
    assert validate_cb_sidecar_tensors(
        payload, blobs, where="unit"
    ) == actual_bytes
    assert codebook_subtable_shapes("FP8_CB_K37") == (
        (1024, 2),
        (512, 2),
        (512, 2),
        (512, 2),
    )


def test_tensor_breakdown_requires_exact_context_and_shape():
    with pytest.raises(ValueError, match="requires layout_version=2"):
        CBSerializationContext(
            scale_coding="two_tier",
            layout_version=1,
            codebook_source="lattice",
        )
    with pytest.raises(ValueError, match="CBSerializationContext"):
        cb_tensor_payload_breakdown(
            "NVFP4_CB_K16", (64, 256), qname="w", context=None
        )
    ctx = CBSerializationContext.production()
    with pytest.raises(ValueError, match="divisible"):
        cb_tensor_payload_breakdown(
            "NVFP4_CB_K16", (64, 255), qname="w", context=ctx
        )


def test_recipe_context_stamp_rejects_export_layout_drift():
    digests = _learned_context("NVFP4_CB_K16").codebook_content_digests
    production = CBSerializationContext.production(
        codebook_source="learned", codebook_content_digests=digests
    )
    stamp = cb_serialization_context_stamp(production)
    validate_cb_serialization_context_stamp(
        stamp, production, where="unit"
    )
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        validate_cb_serialization_context_stamp(
            stamp,
            CBSerializationContext.legacy_v1(
                codebook_source="learned", codebook_content_digests=digests
            ),
            where="unit",
        )


def test_footprint_mixed_registry_format():
    # A stock (non-CB) format still accounts via the registry; no CB sidecar
    # or global scale for it.
    fp = cb_footprint({"w": "NVFP4"}, {"w": (256, 256)})
    assert fp["sidecar_bytes"] == 0
    assert fp["global_scale_bytes"] == 0
    assert fp["body_bpw"] == pytest.approx(4.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Index entropy
# ---------------------------------------------------------------------------

def test_entropy_uniform_indices_approaches_k():
    k = 10
    torch.manual_seed(0)
    # Large uniform sample over all 2^k symbols → H ≈ k.
    idx = torch.randint(0, 1 << k, (400_000,))
    r = index_entropy(idx, k)
    assert r["H"] == pytest.approx(k, abs=0.05)
    assert r["redundancy"] == pytest.approx(0.0, abs=0.05)
    assert r["redundancy"] == pytest.approx(k - r["H"], abs=1e-9)


def test_entropy_constant_indices_zero():
    idx = torch.full((10_000,), 7, dtype=torch.long)
    r = index_entropy(idx, 12)
    assert r["H"] == pytest.approx(0.0, abs=1e-9)
    assert r["redundancy"] == pytest.approx(12.0, abs=1e-9)
    assert r["H_conditional"] == pytest.approx(0.0, abs=1e-9)


def test_entropy_two_symbol_exact():
    # Balanced two-symbol stream → H = 1 bit exactly.
    idx = torch.tensor([0, 1] * 5000, dtype=torch.long)
    r = index_entropy(idx, 4)
    assert r["H"] == pytest.approx(1.0, abs=1e-6)
    assert r["redundancy"] == pytest.approx(3.0, abs=1e-6)
    # Perfectly predictable from the previous symbol → conditional H ≈ 0.
    assert r["H_conditional"] == pytest.approx(0.0, abs=1e-6)
    assert r["conditional_gain"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Emulated forward KL (GPU)
# ---------------------------------------------------------------------------

_MODEL = "/home/rob/models/Qwen3-0.6B"


def _tiny_dataset(tmp_path: Path) -> str:
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Quantization allocates bits per linear layer to minimize divergence. "
        "Vector codebooks decode to the native floating-point grid.\n\n"
    ) * 6
    p = tmp_path / "held_out.txt"
    p.write_text(text)
    return str(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_identity_is_zero(tmp_path):
    from prismaquant.emu_forward_kl import measure_emulated_kl
    from prismaquant.measure_quant_cost import canonical_linear_name
    from transformers import AutoModelForCausalLM
    import torch.nn as nn

    ds = _tiny_dataset(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    fmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fmap[canonical_linear_name(name)] = {"format": "BF16",
                                                 "col_weights": None}
    del model

    res = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    # BF16 passthrough is bit-identical → KL is exactly zero.
    assert res["kl_all"] == 0.0
    assert res["kl_confident"] == 0.0
    assert res["top1_agreement"] == 1.0
    assert res["n_positions"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_q4k_positive_and_deterministic(tmp_path):
    from prismaquant.emu_forward_kl import measure_emulated_kl
    from prismaquant.measure_quant_cost import canonical_linear_name
    from transformers import AutoModelForCausalLM
    import torch.nn as nn

    ds = _tiny_dataset(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    fmap = {}
    for name, mod in model.named_modules():
        # Quantize only the MLP/attention projections (in_features % 256 == 0
        # is not required for GGUF emulation, but keep it representative).
        if isinstance(mod, nn.Linear):
            fmap[canonical_linear_name(name)] = {"format": "Q4_K",
                                                 "col_weights": None}
    del model

    a = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    b = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=128, max_tokens=256)
    assert a["kl_all"] > 0.0
    assert math.isfinite(a["kl_all"])
    assert math.isfinite(a["kl_confident"])
    # Deterministic across runs (greedy forward, fixed seed).
    assert a["kl_all"] == pytest.approx(b["kl_all"], rel=0, abs=0.0)
    assert a["provenance"]["assignment_sha256"] == b["provenance"]["assignment_sha256"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_missing_target_gate(tmp_path):
    from prismaquant.emu_forward_kl import measure_emulated_kl

    ds = _tiny_dataset(tmp_path)
    fmap = {"not.a.layer.q_proj": {"format": "Q4_K", "col_weights": None}}
    with pytest.raises(ValueError, match="matched no"):
        measure_emulated_kl(
            _MODEL, fmap, ds, device="cuda", seqlen=64, max_tokens=64)
    res = measure_emulated_kl(
        _MODEL, fmap, ds, device="cuda", seqlen=64, max_tokens=64,
        allow_missing_targets=True)
    assert res["n_targets_missing"] == 1
    assert res["missing_targets"] == ["not.a.layer.q_proj"]
    # Nothing swapped → identity forward → KL exactly zero.
    assert res["kl_all"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not Path(_MODEL).exists(), reason="Qwen3-0.6B not present")
def test_emu_kl_act_fallback_gate(tmp_path):
    from prismaquant import format_registry as fr
    from prismaquant.emu_forward_kl import measure_emulated_kl

    def _boom(x):
        raise ValueError("activation emulation unavailable for this shape")

    name = "_TEST_ACT_FAIL"
    fr.register_format(fr.FormatSpec(
        name=name,
        weight_bits=16, group_size=0, scale_bits=0,
        scale_dtype_name="none", weight_element_dtype="bf16",
        act_bits=4, act_dtype_name="fp4_e2m1", act_group_size=16,
        family="nv",
        quantize_dequantize=lambda w: w,
        activation_quantize_dequantize=_boom,
    ))
    try:
        ds = _tiny_dataset(tmp_path)
        fmap = {"model.layers.0.self_attn.q_proj": {"format": name,
                                                    "col_weights": None}}
        with pytest.raises(RuntimeError, match="activation emulation failed"):
            measure_emulated_kl(
                _MODEL, fmap, ds, device="cuda", seqlen=64, max_tokens=64)
        res = measure_emulated_kl(
            _MODEL, fmap, ds, device="cuda", seqlen=64, max_tokens=64,
            allow_act_fallback=True)
        counts = res["act_fallback_counts"]
        assert len(counts) == 1
        (key, n), = counts.items()
        assert "model.layers.0.self_attn.q_proj" in key
        assert n > 0
    finally:
        fr.REGISTRY.pop(name, None)
