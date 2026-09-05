from __future__ import annotations

import torch

from prismaquant import format_registry as fr
from prismaquant.export_native_compressed import (
    _mxfp8_dequantize_2d,
    _rtn_dequant_nvfp4,
    quantize_dequantize_fp8_dynamic,
    quantize_dequantize_mxfp8,
)


def test_plain_fp8_rtn_uses_eager_path(monkeypatch):
    compile_calls = []

    def fake_compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def compiled(*_args, **_kwargs):
            raise AssertionError("plain FP8 RTN should not use torch.compile")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    quantize = fr._make_rtn("fp8_e4m3", 0)
    x = torch.linspace(-3.1, 3.1, steps=64, dtype=torch.float32).reshape(2, 32)
    y = quantize(x)

    assert compile_calls == []
    assert y.shape == x.shape
    assert not torch.equal(y, x)


def test_e5m2_codebook_excludes_special_exp31_values():
    cb = fr._CODEBOOKS["fp8_e5m2"]

    assert float(cb.abs().max()) == 57344.0
    assert not torch.any(cb.abs() > 57344.0)
    assert torch.any(cb == torch.tensor(57344.0))


def test_fp6_codebooks_include_ocp_subnormals():
    e3m2 = fr._CODEBOOKS["fp6_e3m2"]
    e2m3 = fr._CODEBOOKS["fp6_e2m3"]

    for value in (0.0625, 0.125, 0.1875):
        assert torch.any(e3m2 == torch.tensor(value))
    assert not torch.any(e3m2 == torch.tensor(0.15625))

    for i in range(1, 8):
        assert torch.any(e2m3 == torch.tensor(i / 8.0))
    assert not torch.any(e2m3 == torch.tensor(0.0625))


def test_plain_fp8_weight_matches_compressed_tensors_fp8_dynamic():
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
    from compressed_tensors.quantization.utils.helpers import calculate_qparams

    vals = torch.tensor(
        [
            0.0,
            1e-12,
            -1e-12,
            1e-8,
            -1e-8,
            1e-6,
            -1e-6,
            1.0 / 1024.0,
            -1.0 / 1024.0,
            1.0 / 512.0,
            -1.0 / 512.0,
            240.0,
            -240.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
    )
    w = vals.repeat(4, 1)
    w[1] *= 1e-6
    w[2] = 0.0

    registry = fr.get_format("FP8_E4M3").quantize_dequantize(w)

    args = FP8_DYNAMIC["weights"]
    scale, zero_point = calculate_qparams(w.amin(dim=1), w.amax(dim=1), args)
    compressed_tensors = fake_quantize(
        w,
        scale.reshape(-1, 1),
        zero_point.reshape(-1, 1),
        args,
    )

    export_q, export_scale = quantize_dequantize_fp8_dynamic(w)
    export = export_q.float() * export_scale

    assert torch.allclose(registry, compressed_tensors, atol=0.0, rtol=0.0)
    assert torch.allclose(registry, export, atol=0.0, rtol=0.0)


def test_plain_fp8_rank3_activation_matches_compressed_tensors_fp8_dynamic():
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC
    from compressed_tensors.quantization.utils.helpers import (
        compute_dynamic_scales_and_zp,
    )

    torch.manual_seed(19)
    x = torch.randn(2, 5, 32, dtype=torch.float32) * 3.0
    x[0, 0, :8] = torch.tensor(
        [
            0.0,
            1.0 / 1024.0,
            -1.0 / 1024.0,
            1.0 / 512.0,
            -1.0 / 512.0,
            240.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
    )

    registry = fr.get_format("FP8_E4M3").activation_quantize_dequantize(x)

    args = FP8_DYNAMIC["input_activations"]
    dummy = torch.nn.Linear(x.shape[-1], 1, bias=False)
    scale, zero_point = compute_dynamic_scales_and_zp(x, args, dummy)
    compressed_tensors = fake_quantize(x, scale, zero_point, args)

    assert torch.allclose(registry, compressed_tensors, atol=0.0, rtol=0.0)


def test_plain_fp8_rank2_activation_matches_vllm_dynamic_token_reference():
    x = torch.tensor(
        [
            [
                0.0,
                1e-12,
                -1e-12,
                1e-8,
                -1e-8,
                1e-6,
                -1e-6,
                1.0 / 1024.0,
                -1.0 / 1024.0,
                1.0 / 512.0,
                -1.0 / 512.0,
                1.0,
                -1.0,
                448.0,
                -448.0,
                0.25,
            ],
            torch.linspace(-3.0, 3.0, steps=16, dtype=torch.float32),
        ],
        dtype=torch.float32,
    )
    registry = fr.get_format("FP8_E4M3").activation_quantize_dequantize(x)

    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    min_scale = 1.0 / (fp8_max * 512.0)
    scale = (x.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp_min(
        min_scale,
    )
    quant = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    reference = quant.float() * scale

    assert torch.allclose(registry, reference, atol=0.0, rtol=0.0)


def test_mx_e8m0_rtn_matches_export_scale_rounding():
    cases = [torch.linspace(-3.7, 3.7, steps=64, dtype=torch.float32).reshape(2, 32)]
    for seed in range(5):
        torch.manual_seed(seed)
        cases.append(torch.randn(16, 64, dtype=torch.float32) * 10 ** (seed - 2))

    for w in cases:
        registry = fr.get_format("MXFP8_E4M3").quantize_dequantize(w)
        export_q, export_scales = quantize_dequantize_mxfp8(w)
        export = _mxfp8_dequantize_2d(export_q, export_scales)

        assert torch.allclose(registry, export, atol=0.0, rtol=0.0)


def test_nvfp4_registry_rtn_matches_export_scale_convention(monkeypatch):
    import prismaquant.export_native_compressed as enc

    previous = enc._NVFP4_SCALE_RULE
    monkeypatch.setattr(enc, "_NVFP4_SCALE_RULE", enc.NVFP4_SCALE_RULE_STATIC_6)
    try:
        cases = [
            torch.linspace(-3.7, 3.7, steps=64, dtype=torch.float32).reshape(4, 16)
        ]
        for seed in range(5):
            torch.manual_seed(seed)
            cases.append(torch.randn(16, 64, dtype=torch.float32) * 10 ** (seed - 2))

        for w in cases:
            registry = fr.get_format("NVFP4").quantize_dequantize(w)
            export = _rtn_dequant_nvfp4(w, group_size=16)

            assert torch.allclose(registry, export, atol=0.0, rtol=0.0)
    finally:
        monkeypatch.setattr(enc, "_NVFP4_SCALE_RULE", previous)


def test_nvfp4_rank3_weight_matches_export_scale_convention(monkeypatch):
    # WEIGHTS follow the export codec (one rendering everywhere).
    import prismaquant.export_native_compressed as enc

    previous = enc._NVFP4_SCALE_RULE
    monkeypatch.setattr(enc, "_NVFP4_SCALE_RULE", enc.NVFP4_SCALE_RULE_STATIC_6)
    try:
        torch.manual_seed(23)
        x = torch.randn(2, 5, 32, dtype=torch.float32) * 2.0
        registry = fr.get_format("NVFP4").quantize_dequantize(x)
        export = _rtn_dequant_nvfp4(
            x.reshape(-1, x.shape[-1]),
            group_size=16,
        ).reshape_as(x)

        assert torch.allclose(registry, export, atol=0.0, rtol=0.0)
    finally:
        monkeypatch.setattr(enc, "_NVFP4_SCALE_RULE", previous)


def test_nvfp4_activation_emulation_is_batch_independent():
    # ACTIVATIONS deliberately do NOT use the export codec: its per-tensor
    # global scale would make the emulation depend on what else is in the
    # batch, while serve-time activation quant uses a STATIC calibration
    # global. Per-group dynamic RTN keeps each token's quantization a
    # function of that token alone.
    torch.manual_seed(23)
    a = torch.randn(4, 32, dtype=torch.float32)
    fmt = fr.get_format("NVFP4")
    alone = fmt.activation_quantize_dequantize(a)
    with_outlier = fmt.activation_quantize_dequantize(
        torch.cat([a, 1000.0 * torch.ones(1, 32)], dim=0))[:4]
    assert torch.allclose(alone, with_outlier, atol=0.0, rtol=0.0)


def test_nvfp4_weight_emulation_pads_narrow_tensors():
    # cols % 16 != 0 must not crash (zero-pad is exact under max-abs
    # group scaling); regression for the ada08a8 narrow-tensor break.
    x = torch.randn(8, 4, dtype=torch.float32)
    out = fr.get_format("NVFP4").quantize_dequantize(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mxfp8_exported_scales_match_compressed_tensors():
    from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

    torch.manual_seed(11)
    w = torch.randn(16, 96, dtype=torch.float32) * 7.0

    _, export_scales = quantize_dequantize_mxfp8(w)
    grouped = w.reshape(16, 3, 32)
    expected_scales = generate_mx_scales(
        grouped.abs().amax(dim=-1),
        num_bits=8,
    ).to(torch.uint8)

    assert torch.equal(export_scales, expected_scales)


def test_mxfp8_activation_quantizer_matches_vllm_runtime_reference():
    x = torch.randn(9, 64, dtype=torch.float32) * 17.0
    registry = fr.get_format("MXFP8_E4M3").activation_quantize_dequantize(x)

    blocked = x.reshape(9, 2, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    max_pos = float(torch.finfo(torch.float8_e4m3fn).max)
    scale_unbiased = torch.ceil(torch.log2(amax / max_pos)).clamp(-127, 127)
    descale = torch.exp2(scale_unbiased)
    quant = (
        blocked / descale.unsqueeze(-1)
    ).clamp(-max_pos, max_pos).reshape_as(x).to(torch.float8_e4m3fn)
    reference = (
        quant.float().reshape_as(blocked) * descale.unsqueeze(-1)
    ).reshape_as(x)

    assert torch.allclose(registry, reference, atol=0.0, rtol=0.0)


def test_mxfp8_activation_quantizer_uses_e4m3_range():
    block = torch.cat([
        torch.tensor([14.0], dtype=torch.float32),
        torch.linspace(-0.02, 0.02, steps=31, dtype=torch.float32),
    ])
    x = block.repeat(2).reshape(1, 64)

    corrected = fr.get_format("MXFP8_E4M3").activation_quantize_dequantize(x)

    blocked = x.reshape(1, 2, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    raw_amax_scale = torch.exp2(torch.floor(torch.log2(amax)).clamp(-127, 127))
    raw_amax_quant = (
        blocked / raw_amax_scale.unsqueeze(-1)
    ).reshape_as(x).to(torch.float8_e4m3fn)
    raw_amax_reference = (
        raw_amax_quant.float().reshape_as(blocked) * raw_amax_scale.unsqueeze(-1)
    ).reshape_as(x)

    corrected_mse = torch.mean((corrected.float() - x) ** 2)
    raw_amax_mse = torch.mean((raw_amax_reference.float() - x) ** 2)

    assert corrected_mse < raw_amax_mse * 0.01


def test_every_registered_name_resolves_through_an_upper_cased_spelling():
    """One rule, one home: whether this registry answers to a name's CASE is
    the registry's question, and it is settled in ``canonical_format_name``.

    ``canonical_format_name`` already tried the raw spelling and then the
    upper-cased one, which covers "the caller typed lower case, the row is
    upper case".  It did not cover the mirror image -- "the caller (or an
    upstream normalizer) upper-cased, the row is mixed case" -- and exactly
    one registered row is mixed case, ``INT4_W4A16_g128``.  Every caller that
    normalizes a requested format name by upper-casing it therefore produced
    a name no resolver owned, and ``get_format`` raised ``KeyError`` (#218).
    """
    import pytest

    # The precondition that makes case-insensitive resolution unambiguous.
    names = [*fr.REGISTRY, *fr.FORMAT_ALIASES]
    folded = [name.casefold() for name in names]
    assert len(folded) == len(set(folded)), "two format names differ only by case"

    assert "INT4_W4A16_g128" in fr.REGISTRY
    assert fr.canonical_format_name("INT4_W4A16_G128") == "INT4_W4A16_g128"
    assert fr.get_format("INT4_W4A16_G128").name == "INT4_W4A16_g128"
    assert fr.get_format("int4_w4a16_g128").name == "INT4_W4A16_g128"

    # The whole registry, enumerated: no registered name or alias is lost by
    # an upper-case round trip, and none of them resolves to a DIFFERENT spec.
    for name in names:
        assert fr.get_format(name.upper()).name == fr.get_format(name).name
        assert fr.get_format(name.lower()).name == fr.get_format(name).name

    # A name the registry does not own is still unknown, and still raises
    # naming the registry rather than resolving to something arbitrary.
    with pytest.raises(KeyError):
        fr.get_format("NOT_A_REGISTERED_FORMAT_pq218")
