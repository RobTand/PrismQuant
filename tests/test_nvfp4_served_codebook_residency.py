"""Static-G QDQ must reuse its device constants without changing any output bit."""
import pytest
import torch

from prismaquant import nvfp4_activation_contract as owner


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA numerical/device-residency qualification")
    return torch.device("cuda", torch.cuda.current_device()) if request.param == "cuda" else torch.device("cpu")


def test_warm_qdq_does_not_rebuild_host_codepoints(monkeypatch, device):
    x = torch.ones((32, 64), device=device, dtype=torch.bfloat16)
    expected = owner.nvfp4_activation_qdq_served(x, 1.0)
    original = torch.tensor
    rebuilt = []

    def observe(data, *args, **kwargs):
        if data is owner._E2M1_POSITIVE:
            rebuilt.append(kwargs.get("device"))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", observe)
    for _ in range(3):
        actual = owner.nvfp4_activation_qdq_served(x, 1.0)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert not rebuilt, "warm QDQ rebuilt the invariant host codepoints on its device"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_midpoint_goldens_and_signed_zero(device, dtype):
    # Every block contains +/-6, so G=1 gives a stored/used scale of exactly 1.
    # E2M1 midpoint ties choose the even encoded positive index.
    midpoints = (0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    rounded = (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0)
    x = torch.tensor([[v, -v, 6.0, -6.0] + [0.0] * 12 for v in midpoints],
                     device=device, dtype=dtype)
    expected = torch.tensor([[v, -v, 6.0, -6.0] + [0.0] * 12 for v in rounded],
                            device=device, dtype=dtype)
    actual = owner.nvfp4_activation_qdq_served(x, 1.0)
    assert actual.device == device and actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_scale_underflow_keeps_positive_zero(device):
    # Exactly halfway to the first E4M3 subnormal rounds the stored scale to 0.
    value = 6.0 * 2.0 ** -10
    x = torch.tensor([[value, -value] * 8], device=device)
    actual = owner.nvfp4_activation_qdq_served(x, 1.0)
    expected = torch.zeros_like(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_shared_grid_is_exact_and_device_specific(device):
    from prismaquant import format_registry as fr

    grid = fr._codebook_on_device(fr._CODEBOOKS["fp4_e2m1"], device=device,
                                  dtype=torch.float32)[-len(owner._E2M1_POSITIVE):]
    expected = torch.tensor(owner._E2M1_POSITIVE, device=device, dtype=torch.float32)
    assert grid.device == device and grid.dtype == torch.float32
    assert torch.equal(grid.view(torch.uint8), expected.view(torch.uint8))
