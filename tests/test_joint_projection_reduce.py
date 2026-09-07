"""The experimental fused reduction must preserve the native FP32 sum tree."""
import pytest
import torch

from prismaquant.kernels import joint_projection_reduce as kernel


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA reduction-tree qualification')
    return torch.device('cuda', torch.cuda.current_device())


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.device == expected.device
    assert actual.shape == expected.shape == torch.Size([])
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (actual.item(), expected.item())


@pytest.mark.parametrize('size', [1, 3, 15, 16, 17, 31, 32, 33, 511, 512, 513, 1023, 4096, 3670016])
def test_finite_fp32_products_keep_native_reduction_bits(cuda, size):
    generator = torch.Generator(device=cuda).manual_seed(935 + size)
    left = torch.randn(size, generator=generator, device=cuda)
    right = torch.randn(size, generator=generator, device=cuda)
    assert kernel.fast_path_eligible(left, right)
    exact(kernel.fused_product_sum(left, right), (left * right).sum())


def test_cancellation_subnormals_and_signed_zero(cuda):
    values = [0.0, -0.0, 2.0**-126, 2.0**-140, 2.0**-149,
              2.0**24, 1.0, -(2.0**24), -1.0, 2.0**-24]
    left = torch.tensor(values * 4096, device=cuda)
    right = torch.tensor([1.0, -1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0] * 4096, device=cuda)
    exact(kernel.fused_product_sum(left, right), (left * right).sum())
    for negative in (False, True):
        left = torch.full((4096,), -0.0 if negative else 0.0, device=cuda)
        exact(kernel.fused_product_sum(left, torch.ones_like(left)), (left * 1.0).sum())


def test_strides_alignment_and_empty_keep_reference_semantics(cuda, monkeypatch):
    base = torch.arange(33 * 64, device=cuda, dtype=torch.float32).reshape(33, 64)
    values = [base.T, base[:, ::2], base.flatten()[1:1025], base[:0]]

    def refuse():
        raise AssertionError('ineligible layout attempted the fused kernel')

    monkeypatch.setattr(kernel, 'load_backend', refuse)
    for left in values:
        right = left.clone()
        assert not kernel.fast_path_eligible(left, right)
        exact(kernel.fused_product_sum(left, right), (left * right).sum())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
def test_other_dtypes_keep_reference(cuda, dtype, monkeypatch):
    left = torch.arange(128, device=cuda, dtype=dtype)
    right = torch.ones_like(left)
    monkeypatch.setattr(kernel, 'load_backend', lambda: pytest.fail('dtype entered FP32 kernel'))
    actual = kernel.fused_product_sum(left, right)
    expected = (left * right).sum()
    assert actual.dtype == dtype and actual.device == cuda
    assert torch.equal(actual.reshape(1).view(torch.uint8), expected.reshape(1).view(torch.uint8))


def test_cpu_reference_requires_no_compiler(monkeypatch):
    left = torch.tensor([1.0, -2.0, 3.0])
    monkeypatch.setattr(kernel, 'load_backend', lambda: pytest.fail('CPU entered CUDA compiler'))
    exact(kernel.fused_product_sum(left, left), (left * left).sum())


def test_nondefault_stream_keeps_device_and_dependency_order(cuda):
    kernel.load_backend()
    stream = torch.cuda.Stream(device=cuda)
    with torch.cuda.stream(stream):
        left = torch.arange(1024 * 1024, device=cuda, dtype=torch.float32)
        right = torch.empty_like(left).fill_(0.25)
        actual = kernel.fused_product_sum(left, right)
        expected = (left * right).sum()
    stream.synchronize()
    exact(actual, expected)


def test_autograd_is_explicitly_refused(cuda):
    left = torch.ones(32, device=cuda, requires_grad=True)
    with pytest.raises(RuntimeError, match='no autograd registration'):
        kernel.fused_product_sum(left, left)
    with torch.no_grad():
        exact(kernel.fused_product_sum(left, left), (left * left).sum())
