"""CPU regression for cooperative guard enforcement during output materialization."""
import weakref

import pytest
import torch

from experiments.glm_layer_workspace import failed_collector_row_counts
from prismaquant.tessera_campaign import _collect_activations


def model_and_tokens():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False),
                                torch.nn.Linear(4, 4, bias=False)).eval()
    with torch.no_grad():
        for module in model:
            module.weight.copy_(torch.eye(4))
    return model, [torch.arange(8, dtype=torch.float32).reshape(2, 4)]


@pytest.mark.parametrize('stop_at,expected_transfers', [
    ('before_output_materialization', 0),
    ('after_rows_concat:0', 0),
    ('before_hessian_cpu_transfer:0', 0),
    ('after_hessian_cpu_transfer:0', 1),
])
def test_guard_refuses_before_next_hessian_transfer_and_keeps_observed_rows(monkeypatch, stop_at, expected_transfers):
    model, tokens = model_and_tokens()
    transferred, hessians = [], []
    original_to = torch.Tensor.to
    original_matmul = torch.Tensor.__matmul__

    def matmul(left, right):
        value = original_matmul(left, right)
        if tuple(left.shape) == (4, 2) and tuple(right.shape) == (2, 4):
            hessians.append(weakref.ref(value))
        return value

    def to(tensor, *args, **kwargs):
        if kwargs.get('device') == 'cpu' and tuple(tensor.shape) == (4, 4):
            transferred.append(weakref.ref(tensor))
        return original_to(tensor, *args, **kwargs)

    def check(label):
        if label == stop_at:
            raise RuntimeError('memory guard latched')

    monkeypatch.setattr(torch.Tensor, 'to', to)
    monkeypatch.setattr(torch.Tensor, '__matmul__', matmul)
    with pytest.raises(RuntimeError, match='memory guard latched') as caught:
        _collect_activations(model, ['0', '1'], tokens, 1, 'cpu',
                             want_hessian=True, resource_check=check)
    assert len(transferred) == expected_transfers
    assert all(ref() is None for ref in transferred)
    assert len(hessians) == 2 and all(ref() is None for ref in hessians)
    observed = failed_collector_row_counts(caught.value, _collect_activations, ['0', '1'])
    assert observed['rows'] == {'0': 2, '1': 2}
    assert all(not module._forward_pre_hooks for module in model)


def test_successful_resource_checks_preserve_rows_hessians_counts_and_maxima():
    model, tokens = model_and_tokens()
    baseline = _collect_activations(model, ['0', '1'], tokens, 1, 'cpu', want_hessian=True)
    labels = []
    checked = _collect_activations(model, ['0', '1'], tokens, 1, 'cpu',
                                   want_hessian=True, resource_check=labels.append)
    for position in (0, 1):
        for name in ('0', '1'):
            torch.testing.assert_close(checked[position][name], baseline[position][name], rtol=0, atol=0)
    assert checked[2:] == baseline[2:]
    assert labels[0] == 'before_output_materialization'
    assert labels[-1] == 'after_output_materialization'
    assert [label for label in labels if 'hessian' in label] == [
        'before_hessian_cpu_transfer:0', 'after_hessian_cpu_transfer:0',
        'before_hessian_cpu_transfer:1', 'after_hessian_cpu_transfer:1',
    ]


def test_cuda_guard_failure_returns_cache_after_output_owners_clear(monkeypatch):
    model, tokens = model_and_tokens()
    original_to = torch.Tensor.to
    released = []

    def fake_cuda_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(tensor, *args, **kwargs)

    def check(label):
        if label == 'before_hessian_cpu_transfer:0':
            raise RuntimeError('guard')

    monkeypatch.setattr(torch.Tensor, 'to', fake_cuda_to)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: released.append('cache released'))
    with pytest.raises(RuntimeError, match='guard'):
        _collect_activations(model, ['0', '1'], tokens, 1, 'cuda',
                             want_hessian=True, resource_check=check)
    assert released == ['cache released']
