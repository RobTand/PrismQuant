"""Native vLLM 1970f3ed4 observations, independent of the emulation formula."""
import pytest
import torch

from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_fp8_midpoint_scales_codes_and_signed_zero(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA arithmetic parity requires a GPU worker")
    # Actual canonical LFM prefill row 53 (cols 232,354,496), with its maximum;
    # native diagnostic artifact 8452ec28... includes the independent observations.
    # Negative zero is independently observed in row 40, column 1819.
    values = torch.tensor([[.96875, -.0908203125, .022705078125, .36328125, -0.0]],
                          dtype=torch.bfloat16, device=device)
    actual = fp8_dynamic_activation_qdq_vllm(values)
    expected_scale = torch.tensor([[.0021623882930725813]], dtype=torch.float32, device=device)
    expected_codes = torch.tensor([[448., -44., 11., 176., -0.0]], dtype=torch.float8_e4m3fn, device=device)
    assert torch.equal(actual.scale, expected_scale)
    assert torch.equal(actual.quant.view(torch.uint8), expected_codes.view(torch.uint8))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_tokens_keep_native_scale_floor_and_signed_zero(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA arithmetic parity requires a GPU worker")
    values = torch.tensor([[0., -0.]], dtype=torch.bfloat16, device=device)
    actual = fp8_dynamic_activation_qdq_vllm(values)
    assert torch.equal(actual.scale, torch.full_like(actual.scale, 1 / (448 * 512)))
    assert actual.quant.view(torch.uint8).tolist() == [[0, 128]]
