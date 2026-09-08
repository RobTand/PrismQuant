"""Actual source-storage lifetime at the original forward/CPU-return boundary."""
import json
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from prismaquant.tessera_campaign import _collect_activations
from test_glm_campaign_streaming import glm_checkpoint


def test_current_packed_storage_dies_before_cpu_return_and_next_stays_resident(
        glm_checkpoint, tmp_path, monkeypatch):
    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    targets = [m.qname for m in profile_declared_packed_expert_projections(reference, profile)]
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1) for _ in range(2)]
    expected = _collect_activations(reference, targets, tokens, 7, 'cpu',
        want_hessian=True, profile=profile, shared_packed_inputs=True)
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.float32, offload_folder=str(tmp_path/'release-offload'), profile=profile,
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    captured = [{}, {}, {}, {}]
    callbacks, forwards = [], []
    original_call = runner._call

    def call(layer, *args, **kwargs):
        forwards.append(layer)
        return original_call(layer, *args, **kwargs)

    monkeypatch.setattr(runner, '_call', call)

    def visit(layer, forward):
        names = [n for n in targets if runner.layer_index_for_qname(n) == layer]
        current = [StorageWeakRef(p.untyped_storage()) for p in runner.layers[layer].parameters()]
        next_storage = []
        next_future = None
        if layer + 1 < runner.num_layers:
            next_future = runner.context._inflight[layer + 1]
            resident = next_future.result()
            next_storage = [StorageWeakRef(t.untyped_storage()) for t in resident.values()]
            del resident

        def completed():
            assert forwards.count(layer) == len(tokens)
            assert all(not module._forward_pre_hooks for module in runner.model.modules())
            before = runner.context.source_residency_snapshot(range(layer, runner.num_layers))
            json.dumps(before)
            runner.context.release_completed_layer(layer)
            after = runner.context.source_residency_snapshot(range(layer, runner.num_layers))
            json.dumps(after)
            assert all(ref.expired() for ref in current), 'old packed views still own current source'
            assert all(not ref.expired() for ref in next_storage)
            assert after['unique_storage_bytes'] < before['unique_storage_bytes']
            if next_future is not None:
                assert runner.context._inflight[layer + 1] is next_future
                assert [x for x in before['owners'] if x['layer'] == layer + 1] == [
                    x for x in after['owners'] if x['layer'] == layer + 1]
            assert layer not in runner.context.layer_cache._cache
            assert all(p.is_meta for p in runner.layers[layer].parameters())
            callbacks.append(layer)

        values = _collect_activations(runner.model, names, tokens, 7, 'cpu',
            want_hessian=True, profile=profile, shared_packed_inputs=True,
            forward_batch=forward, on_forwards_complete=completed)
        for all_values, local in zip(captured, values):
            all_values.update(local)
        if layer + 1 < runner.num_layers:
            import prismaquant.streaming_model as streaming
            monkeypatch.setattr(streaming, '_read_layer_to_device',
                lambda *_args, **_kwargs: pytest.fail('next layer required a cold source read'))

    try:
        runner.visit_layer_batches(tokens, visit)
    finally:
        runner.shutdown()
    assert callbacks == [0, 1]
    assert captured[2:] == list(expected[2:])
    for actual, baseline in zip(captured[:2], expected[:2]):
        assert actual.keys() == baseline.keys()
        for name in actual:
            assert torch.equal(actual[name].view(torch.uint8), baseline[name].view(torch.uint8))


@pytest.mark.parametrize('shared', [False, True])
def test_callback_failure_drains_capture_owners_and_keeps_integer_rows(monkeypatch, shared):
    from experiments.glm_layer_workspace import failed_collector_row_counts
    from test_collector_resource_check import model_and_tokens
    model, tokens = model_and_tokens()
    refs, calls = [], []
    matmul = torch.Tensor.__matmul__

    def track(left, right):
        value = matmul(left, right)
        refs.append(weakref.ref(value))
        return value

    def completed():
        calls.append('completed')
        assert all(not module._forward_pre_hooks for module in model)
        raise RuntimeError('source-release callback failed')

    monkeypatch.setattr(torch.Tensor, '__matmul__', track)
    with pytest.raises(RuntimeError, match='source-release callback failed') as caught:
        _collect_activations(model, ['0', '1'], tokens, 1, 'cpu', want_hessian=True,
            shared_packed_inputs=shared, on_forwards_complete=completed)
    assert calls == ['completed']
    assert len(refs) == 2 and all(ref() is None for ref in refs)
    assert failed_collector_row_counts(caught.value, _collect_activations, ['0', '1'])['rows'] == {
        '0': 2, '1': 2}


@pytest.mark.parametrize('shared', [False, True])
def test_forward_failure_never_calls_successful_source_boundary(monkeypatch, shared):
    from test_collector_resource_check import model_and_tokens
    model, tokens = model_and_tokens()
    called = []
    refs = []
    original = torch.Tensor.__matmul__

    def matmul(left, right):
        value = original(left, right)
        refs.append(weakref.ref(value))
        return value

    def forward(batch):
        model(batch)
        raise RuntimeError('source forward failed')

    monkeypatch.setattr(torch.Tensor, '__matmul__', matmul)
    with pytest.raises(RuntimeError, match='source forward failed') as caught:
        _collect_activations(model, ['0', '1'], tokens, 1, 'cpu', want_hessian=True,
            shared_packed_inputs=shared, forward_batch=forward,
            on_forwards_complete=lambda: called.append(True))
    assert called == []
    assert refs and all(ref() is None for ref in refs)


def test_callback_failure_releases_cuda_cache_without_resource_callback(monkeypatch):
    from test_collector_resource_check import model_and_tokens
    model, tokens = model_and_tokens()
    original_to = torch.Tensor.to
    released = []

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(tensor, *args, **kwargs)

    def completed():
        raise RuntimeError('source callback failed')

    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: released.append(True))
    with pytest.raises(RuntimeError, match='source callback failed'):
        _collect_activations(model, ['0', '1'], tokens, 1, 'cuda', want_hessian=True,
            on_forwards_complete=completed)
    assert released == [True]


@pytest.mark.parametrize('stop_at,groups_completed,qnames_completed', [
    ('after_hessian_cpu_transfer:model.layers.2.feed_forward.experts.0.w1', 0, 0),
    ('after_output_clone:model.layers.2.feed_forward.experts.0.w3', 0, 2),
    ('after_output_group:model.layers.2.feed_forward.experts.0.w1', 1, 2),
])
def test_exact_group_progress_survives_guard_refusal(stop_at, groups_completed, qnames_completed):
    from experiments.glm_layer_workspace import record_materialization_checkpoint
    from test_collector_shared_inputs import packed_targets
    from test_tessera_campaign_packed import _RoutedModel, _routed_tokens, EXPERT_PREFIX
    result = {}

    def check(label):
        record_materialization_checkpoint(result, label, {'time': 1.0, 'cuda_reserved_bytes': 123})
        if label == stop_at:
            raise RuntimeError('continuous guard already latched')

    with pytest.raises(RuntimeError, match='continuous guard already latched'):
        _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(), 3, 'cpu',
            want_hessian=True, shared_packed_inputs=True, resource_check=check)
    progress = result['output_materialization_progress']
    assert progress['groups_started'] == 1
    assert progress['groups_completed'] == groups_completed
    assert progress['qnames_completed'] == qnames_completed
    assert progress['current_group'] == (None if groups_completed else f'{EXPERT_PREFIX}.0.w1')
    assert progress['last_checkpoint'] == stop_at
    assert len(progress['completed_groups']) == groups_completed
    assert len(progress['group_boundaries']) == 1 + groups_completed
    json.dumps(result)


def test_complete_group_counts_describe_independent_outputs():
    from experiments.glm_layer_workspace import record_materialization_checkpoint
    from test_collector_shared_inputs import packed_targets
    from test_tessera_campaign_packed import _RoutedModel, _routed_tokens
    result = {}
    values = _collect_activations(_RoutedModel(), packed_targets(), _routed_tokens(), 3, 'cpu',
        want_hessian=True, shared_packed_inputs=True,
        resource_check=lambda label: record_materialization_checkpoint(result, label, {'time': 1.0}))
    progress = result['output_materialization_progress']
    assert progress['groups_started'] == progress['groups_completed'] == 4
    assert progress['qnames_completed'] == len(values[0]) == 6
    assert progress['current_group'] is None
    assert len(progress['group_boundaries']) == 8
    json.dumps(result)
