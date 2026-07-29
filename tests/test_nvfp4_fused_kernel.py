import pytest
import torch

# triton is NOT a declared dependency -- it ships with the CUDA torch wheel and
# is absent from a CPU-only install, where this module's kernel import fails at
# collection time and takes the whole suite down with it. Skip the module
# instead: these are GPU kernel tests and have nothing to assert without it.
pytest.importorskip("triton")

from prismaquant import format_registry as fr
from prismaquant.kernels.nvfp4_fused import (
    nvfp4_dequantize_weight,
    nvfp4_fused_aw_matmul,
    nvfp4_pack_weight,
)


def _export_convention_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 activation RTN with the export codec's tie convention.

    The fused kernel rounds exact codebook midpoints half-toward-zero for
    both signs (matching `_round_to_codebook`, §3.15b fix). The registry's
    activation path resolves exact bf16 midpoint ties sign-asymmetrically
    (audit §3.8), so it cannot serve as the oracle at tie positions —
    ~0.036% of bf16 elements land on exact midpoints at large shapes.
    """
    from prismaquant.export_native_compressed import (
        FLOAT_TO_E2M1,
        _round_to_codebook,
    )

    xf = x.float()
    M, K = xf.shape
    g = xf.reshape(M, K // 16, 16)
    scale = (g.abs().amax(dim=-1, keepdim=True) / 6.0).clamp_min(1e-8 / 6.0)
    idx = _round_to_codebook(g / scale)
    cb = torch.tensor(FLOAT_TO_E2M1, device=x.device, dtype=torch.float32)
    q = torch.where((idx & 0x8) != 0, -1.0, 1.0) * cb[idx & 0x7] * scale
    return q.reshape(M, K).to(x.dtype)


def _nvfp4_quant_then_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    nvfp4 = fr.get_format("NVFP4")
    qx = _export_convention_act_qdq(x)
    qw = nvfp4.quantize_dequantize(weight)
    return qx @ qw.t()


def test_nvfp4_pack_weight_matches_format_registry_reference():
    torch.manual_seed(0)
    weight = (torch.randn(17, 64) * 0.2).to(torch.bfloat16)

    w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)
    dequant = nvfp4_dequantize_weight(
        w_packed,
        w_scales,
        w_global_scale,
        dtype=weight.dtype,
    )
    reference = fr.get_format("NVFP4").quantize_dequantize(weight)

    assert w_packed.dtype == torch.uint8
    assert w_packed.shape == (17, 32)
    assert w_scales.shape == (17, 4)
    torch.testing.assert_close(dequant, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (4, 32, 64),
        (17, 96, 128),
        (16, 2560, 2560),
    ],
)
def test_nvfp4_fused_matches_unfused_path(M, N, K):
    torch.manual_seed(1234 + M + N + K)
    device = torch.device("cuda")
    x = (torch.randn(M, K, device=device) * 0.05).to(torch.bfloat16)
    weight = (torch.randn(N, K, device=device) * 0.05).to(torch.bfloat16)

    w_packed, w_scales, w_global_scale = nvfp4_pack_weight(weight)
    out_fused = nvfp4_fused_aw_matmul(x, w_packed, w_scales, w_global_scale)
    out_reference = _nvfp4_quant_then_matmul(x, weight)
    max_abs = (out_fused.float() - out_reference.float()).abs().max().item()

    assert torch.allclose(out_fused, out_reference, atol=6e-3, rtol=2e-2), (
        f"max_abs_diff={max_abs:.6g}"
    )


def test_indices_from_signed_e2m1_values_nearest_with_epsilon():
    """§3.15a (2026-07-02 audit): `_indices_from_signed_e2m1_values` must be
    nearest-neighbor (midpoint boundaries, ties toward zero — the export
    codec's `_round_to_codebook` convention). The old bucketize-on-codes
    mapped a value ε ABOVE a code (a bf16 round-trip artifact) to the NEXT
    code — a full-step error."""
    from prismaquant.kernels.nvfp4_fused import (
        _FP4_E2M1_POS,
        _indices_from_signed_e2m1_values,
    )

    codes = torch.tensor(_FP4_E2M1_POS, dtype=torch.float32)
    eps = 1e-4
    for sign in (1.0, -1.0):
        for i, c in enumerate(_FP4_E2M1_POS):
            for v in (c, c + eps, max(c - eps, 0.0)):
                got = _indices_from_signed_e2m1_values(
                    torch.tensor([sign * v], dtype=torch.float32))
                abs_idx = int(got.item()) & 0x7
                assert abs_idx == i, (
                    f"value {sign * v} -> code index {abs_idx}, want {i}")
                if sign < 0:
                    assert int(got.item()) & 0x8

    # Exact midpoints round toward zero (matches _round_to_codebook).
    midpoints = (codes[1:] + codes[:-1]) / 2.0
    got = _indices_from_signed_e2m1_values(midpoints)
    assert got.tolist() == list(range(len(_FP4_E2M1_POS) - 1))

    # Off-grid values still map to the nearest code.
    got = _indices_from_signed_e2m1_values(
        torch.tensor([0.3, 1.4, 2.6, 5.9], dtype=torch.float32))
    assert (got & 0x7).tolist() == [1, 3, 5, 7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
def test_fused_activation_quant_ties_round_half_toward_zero():
    """§3.15b: the Triton activation quant's tie rounding must be
    round-half-toward-zero for BOTH signs (the old code used >= on the
    negative branch — negative half-ties rounded away from zero)."""
    from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul

    device = torch.device("cuda")
    K = 16
    # One group of 16 with max 6.0 -> activation group scale is exactly
    # 1.0, so the quantized values are the E2M1 rounding of x itself.
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    x_row = torch.zeros(K, dtype=torch.float32)
    x_row[0] = 6.0
    for j, t in enumerate(ties):
        x_row[1 + j] = t
    # Weight: row n reads out slot n via a single 6.0 (exactly packable:
    # group scale 1.0 -> fp8 scale 448 -> effective scale exactly 1.0).
    W = torch.zeros(K, K, dtype=torch.float32)
    for n in range(K):
        W[n, n] = 6.0

    from prismaquant.kernels.nvfp4_fused import nvfp4_pack_weight
    w_packed, w_scales, w_gs = nvfp4_pack_weight(
        W.to(device=device, dtype=torch.bfloat16))

    expected_codes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # half toward zero
    for sign in (1.0, -1.0):
        x = (sign * x_row).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        out = nvfp4_fused_aw_matmul(x, w_packed, w_scales, w_gs)
        got = (out.float()[0, 1:1 + len(ties)] / 6.0).cpu()
        expected = sign * torch.tensor(expected_codes)
        torch.testing.assert_close(got, expected, atol=1e-3, rtol=0.0)
