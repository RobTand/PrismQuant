"""Exact CPU contracts for the opt-in packed-input collector."""
import gc
import io
import weakref

import pytest
import torch

from prismaquant.tessera_campaign import _collect_activations
from test_tessera_campaign_packed import _RoutedModel, _routed_tokens, EXPERT_PREFIX


def packed_targets():
    return [f'{EXPERT_PREFIX}.{expert}.{projection}'
            for expert in range(2) for projection in ('w1', 'w3', 'w2')]


def test_legacy_duplicates_gate_up_gram_work(monkeypatch):
    """Before-change call-count evidence on actual routed tensors, no timing claim."""
    original = torch.Tensor.__matmul__
    calls = []

    def matmul(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original(left, right)

    monkeypatch.setattr(torch.Tensor, '__matmul__', matmul)
    _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(), 3,
                         'cpu', want_hessian=True)
    assert len(calls) == 12  # Six qnames each batch; four unique routed inputs.


def assert_capture_exact(left, right):
    for position in (0, 1):
        assert left[position].keys() == right[position].keys()
        for name, expected in left[position].items():
            actual = right[position][name]
            if expected is None:
                assert actual is None
            else:
                assert torch.equal(actual.contiguous().view(torch.uint8),
                                   expected.contiguous().view(torch.uint8))
    assert left[2:] == right[2:]
    # Compare real serialized per-qname tensor payloads, including storage
    # offsets/sizes; an overallocated prefix must not leak unused rows.
    for name in left[0]:
        payloads = []
        for capture in (left, right):
            stream = io.BytesIO()
            torch.save({'X': capture[0][name], 'H': capture[1].get(name),
                        'count': capture[2][name], 'amax': capture[3][name]}, stream)
            payloads.append(stream.getvalue())
        assert payloads[0] == payloads[1]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('max_rows', [0, 1, 3, 99])
@pytest.mark.parametrize('want_hessian', [False, True])
def test_shared_exact_full_rows_cap_order_and_dense(dtype, max_rows, want_hessian):
    targets = ['model.layers.0.attention', *packed_targets()]
    tokens = [x.to(dtype) for x in _routed_tokens()]
    model = _RoutedModel().to(dtype)
    before = _collect_activations(model, targets, tokens, max_rows, 'cpu',
                                  want_hessian=want_hessian)
    after = _collect_activations(model, targets, tokens, max_rows, 'cpu',
                                 want_hessian=want_hessian, shared_packed_inputs=True)
    assert_capture_exact(before, after)
    for row_dict in after[:2]:
        values = [v for v in row_dict.values() if v is not None]
        assert len({v.untyped_storage().data_ptr() for v in values}) == len(values)


@pytest.mark.parametrize('projections', [('w1',), ('w3',), ('w2',), ('w1', 'w3')])
def test_partial_siblings_share_only_selected_inputs(projections):
    targets = [f'{EXPERT_PREFIX}.1.{p}' for p in projections]
    model, tokens = _RoutedModel(), _routed_tokens()
    before = _collect_activations(model, targets, tokens, 3, 'cpu', want_hessian=True)
    after = _collect_activations(model, targets, tokens, 3, 'cpu', want_hessian=True,
                                 shared_packed_inputs=True)
    assert_capture_exact(before, after)


def test_shared_computes_one_gram_per_actual_input(monkeypatch):
    original = torch.Tensor.__matmul__
    calls = []

    def matmul(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original(left, right)

    monkeypatch.setattr(torch.Tensor, '__matmul__', matmul)
    _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(), 3,
                         'cpu', want_hessian=True, shared_packed_inputs=True)
    assert len(calls) == 8


def test_returned_sibling_mutation_cannot_change_other_outputs():
    result = _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(),
                                  3, 'cpu', want_hessian=True, shared_packed_inputs=True)
    left, right = f'{EXPERT_PREFIX}.0.w1', f'{EXPERT_PREFIX}.0.w3'
    for output in result[:2]:
        expected = {name: value.clone() for name, value in output.items() if name != left}
        output[left].fill_(-777)
        assert all(torch.equal(output[name], value) for name, value in expected.items())
        assert torch.equal(output[right], expected[right])


def test_zero_row_batches_do_not_cap_later_observations():
    tokens = [torch.ones(1, 2, 4), -torch.ones(1, 2, 4)]
    model = _RoutedModel()
    before = _collect_activations(model, packed_targets(), tokens, 1, 'cpu', want_hessian=True)
    after = _collect_activations(model, packed_targets(), tokens, 1, 'cpu', want_hessian=True,
                                 shared_packed_inputs=True)
    assert_capture_exact(before, after)
    assert all(count == 2 for count in after[2].values())


@pytest.mark.parametrize('want_hessian', [False, True])
def test_unobserved_group_still_refuses_and_removes_hooks(want_hessian):
    model = _RoutedModel()
    with pytest.raises(RuntimeError, match='no routed calibration rows'):
        _collect_activations(model, packed_targets(), [torch.ones(1, 2, 4)], 1,
                             'cpu', want_hessian=want_hessian, shared_packed_inputs=True)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks


def test_nan_maximum_preserves_legacy_fmax_policy():
    tokens = _routed_tokens()
    tokens[1][0, :, 2] = float('nan')
    model = _RoutedModel()
    before = _collect_activations(model, packed_targets(), tokens, 3, 'cpu', want_hessian=True)
    after = _collect_activations(model, packed_targets(), tokens, 3, 'cpu', want_hessian=True,
                                 shared_packed_inputs=True)
    assert_capture_exact(before, after)


def test_private_scoring_prefix_survives_reused_mutable_source_buffer():
    model = torch.nn.Sequential(torch.nn.Identity())
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    expected = source.clone()

    def forward(batch):
        model(batch)
        batch.fill_(-999)

    result = _collect_activations(model, ['0'], [source], 3, 'cpu', forward_batch=forward,
                                  want_hessian=True, shared_packed_inputs=True)
    assert torch.equal(result[0]['0'], expected)
    assert torch.equal(result[1]['0'], expected.T @ expected)


@pytest.mark.parametrize('stop_at', [
    'before_output_materialization',
    f'after_rows_concat:{EXPERT_PREFIX}.0.w1',
    f'after_hessian_cpu_transfer:{EXPERT_PREFIX}.0.w1',
    f'before_output_clone:{EXPERT_PREFIX}.0.w3',
    f'after_output_clone:{EXPERT_PREFIX}.0.w3',
    'after_output_materialization',
])
def test_shared_guard_failure_releases_owned_tensors_despite_traceback(monkeypatch, stop_at):
    from experiments.glm_layer_workspace import failed_collector_row_counts

    refs = []
    original_mm, original_empty = torch.Tensor.__matmul__, torch.empty
    original_clone = torch.Tensor.clone

    def mm(left, right):
        value = original_mm(left, right)
        refs.append(weakref.ref(value))
        return value

    def empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if tuple(value.shape) == (3, 4):
            refs.append(weakref.ref(value))
        return value

    def clone(value, *args, **kwargs):
        result = original_clone(value, *args, **kwargs)
        if tuple(result.shape) in ((3, 4), (4, 4)):
            refs.append(weakref.ref(result))
        return result

    def check(label):
        if label == stop_at:
            raise RuntimeError('guard latched')

    monkeypatch.setattr(torch.Tensor, '__matmul__', mm)
    monkeypatch.setattr(torch, 'empty', empty)
    monkeypatch.setattr(torch.Tensor, 'clone', clone)
    model = _RoutedModel()
    with pytest.raises(RuntimeError, match='guard latched') as error:
        _collect_activations(model, packed_targets(), _routed_tokens(), 3, 'cpu',
                             want_hessian=True, shared_packed_inputs=True, resource_check=check)
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks
    observed = failed_collector_row_counts(error.value, _collect_activations, packed_targets())
    assert observed['rows'] == dict.fromkeys(packed_targets(), 4)


def test_shared_group_drains_before_sibling_clone_and_next_transfer():
    labels = []
    _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(), 3, 'cpu',
                         want_hessian=True, shared_packed_inputs=True, resource_check=labels.append)
    owner, sibling, next_owner = (f'{EXPERT_PREFIX}.0.{p}' for p in ('w1', 'w3', 'w2'))
    assert labels.index(f'after_hessian_cpu_transfer:{owner}') < labels.index(f'before_output_clone:{sibling}')
    assert labels.index(f'after_output_clone:{sibling}') < labels.index(f'before_hessian_cpu_transfer:{next_owner}')
    assert f'before_hessian_cpu_transfer:{sibling}' not in labels


def test_shared_forward_failure_drops_accumulator_owners(monkeypatch):
    refs = []
    original = torch.Tensor.__matmul__

    def mm(left, right):
        result = original(left, right)
        refs.append(weakref.ref(result))
        return result

    model = _RoutedModel()

    def forward(batch):
        model(batch)
        raise RuntimeError('source forward failed')

    monkeypatch.setattr(torch.Tensor, '__matmul__', mm)
    with pytest.raises(RuntimeError, match='source forward failed'):
        _collect_activations(model, packed_targets(), _routed_tokens(), 3, 'cpu',
                             want_hessian=True, shared_packed_inputs=True, forward_batch=forward)
    assert refs and all(ref() is None for ref in refs)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks


def test_signed_zero_rows_match_by_bytes_and_serialized_storage():
    model = torch.nn.Sequential(torch.nn.Identity())
    tokens = [torch.tensor([[-0.0, 0.0, -0.0, 1.0]])]
    before = _collect_activations(model, ['0'], tokens, 5, 'cpu', want_hessian=True)
    after = _collect_activations(model, ['0'], tokens, 5, 'cpu', want_hessian=True,
                                 shared_packed_inputs=True)
    assert_capture_exact(before, after)
    assert after[0]['0'].untyped_storage().nbytes() == 16
