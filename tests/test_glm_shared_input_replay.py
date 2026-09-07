"""CPU qualification of the real-input fixture and isolated collector replay."""
import pytest
import torch

from experiments.glm_shared_input_replay import (
    RoutedInputFixture, audit_cpu_storage, compare_captures,
)
from prismaquant import measure_quant_cost, production_weight_cache
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from prismaquant.tessera_campaign import _collect_activations
from test_tessera_campaign_packed import _RoutedModel, _routed_tokens


def capture_fixture(*, max_bytes=1024*1024):
    model = _RoutedModel()
    members = list(profile_declared_packed_expert_projections(model))
    targets = [member.qname for member in members]
    tokens = _routed_tokens() * 4
    fixture = RoutedInputFixture(members, max_batches=len(tokens), max_bytes=max_bytes,
                                 resource_check=lambda _: None)
    with fixture.capture():
        observed = _collect_activations(model, targets, tokens, 0, 'cpu', want_hessian=False)
    return model, targets, tokens, fixture, observed


def test_real_routed_fixture_preserves_source_rows_and_both_arms():
    model, targets, tokens, fixture, observed = capture_fixture()
    source_derive = measure_quant_cost.derive_per_expert_activations
    source_delivery = production_weight_cache._PackedExpertActivationCollector
    manifest = fixture.manifest(3)
    assert manifest['batches'] == 8
    assert manifest['row_counts'] == {0: 16, 1: 16}
    assert manifest['warm_start'] == 2
    visited = []
    before = fixture.replay(model, targets, tokens, 3, 'cpu', shared=False,
                            profile=None, after_batch=visited.append, resource_check=lambda _: None)
    after = fixture.replay(model, targets, tokens, 3, 'cpu', shared=True,
                           profile=None, after_batch=visited.append, resource_check=lambda _: None)
    compare_captures(before, after)
    assert before[2] == after[2] == observed[2]
    assert visited == list(range(1, 9))*2
    assert audit_cpu_storage(before) == audit_cpu_storage(after)
    assert measure_quant_cost.derive_per_expert_activations is source_derive
    assert production_weight_cache._PackedExpertActivationCollector is source_delivery
    fixture.clear()
    assert not fixture.records and fixture.bytes == 0


def test_fixture_cannot_silently_exceed_byte_budget():
    source_derive = measure_quant_cost.derive_per_expert_activations
    with pytest.raises(RuntimeError, match='byte bound'):
        capture_fixture(max_bytes=1)
    assert measure_quant_cost.derive_per_expert_activations is source_derive


def test_missing_row_filled_window_cannot_claim_a_profile():
    _model, _targets, _tokens, fixture, _observed = capture_fixture()
    with pytest.raises(RuntimeError, match='row-filled profile window'):
        fixture.manifest(1000)
    fixture.clear()


def test_replay_failure_restores_original_delivery_and_derivation():
    model, targets, tokens, fixture, _observed = capture_fixture()
    original_derive = measure_quant_cost.derive_per_expert_activations
    original_delivery = production_weight_cache._PackedExpertActivationCollector

    def fail(_label):
        raise RuntimeError('guard refused replay')

    with pytest.raises(RuntimeError, match='guard refused replay'):
        fixture.replay(model, targets, tokens, 3, 'cpu', shared=True, profile=None,
                       after_batch=lambda _: None, resource_check=fail)
    assert measure_quant_cost.derive_per_expert_activations is original_derive
    assert production_weight_cache._PackedExpertActivationCollector is original_delivery
    fixture.clear()


def test_storage_audit_refuses_equal_valued_aliased_siblings():
    value = torch.ones(2, 2)
    capture = ({'gate': value, 'up': value}, {}, {}, {})
    with pytest.raises(RuntimeError, match='aliases'):
        audit_cpu_storage(capture)


def test_compare_refuses_signed_zero_difference():
    def capture(sign):
        return ({'q': torch.tensor([[sign]])}, {'q': torch.ones(1, 1)}, {'q': 1}, {'q': 0.0})
    with pytest.raises(RuntimeError, match='tensor bytes differ'):
        compare_captures(capture(0.0), capture(-0.0))


def test_partial_fixture_guard_failure_releases_clones_with_retained_traceback(monkeypatch):
    import weakref

    model = _RoutedModel()
    members = list(profile_declared_packed_expert_projections(model))
    targets = [member.qname for member in members]
    refs = []
    original_clone = torch.Tensor.clone

    def clone(value, *args, **kwargs):
        result = original_clone(value, *args, **kwargs)
        refs.append(weakref.ref(result))
        return result

    def guard(label):
        if label.startswith('after_routed_fixture_copy:'):
            raise RuntimeError('partial fixture guard')

    monkeypatch.setattr(torch.Tensor, 'clone', clone)
    fixture = RoutedInputFixture(members, max_batches=2, max_bytes=1024*1024,
                                 resource_check=guard)
    with pytest.raises(RuntimeError, match='partial fixture guard') as error:
        with fixture.capture():
            _collect_activations(model, targets, _routed_tokens(), 0, 'cpu', want_hessian=False)
    fixture.clear()
    assert error.value.__traceback__ is not None
    assert refs and all(ref() is None for ref in refs)


def test_qualification_profiles_three_real_cpu_windows_and_seals_payloads(monkeypatch, tmp_path):
    from experiments.glm_shared_input_replay import qualify_collector

    for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    real_profile = torch.profiler.profile

    def cpu_profile(**kwargs):
        kwargs['activities'] = [torch.profiler.ProfilerActivity.CPU]
        return real_profile(**kwargs)

    monkeypatch.setattr(torch.profiler, 'profile', cpu_profile)
    model = _RoutedModel()
    members = list(profile_declared_packed_expert_projections(model))
    report = qualify_collector(model, members, _routed_tokens()*4, 3, 'cpu',
        profile=None, actual_forward=model, out=tmp_path, check_guard=lambda _: None,
        mark=lambda *_args, **_kwargs: None, max_fixture_bytes=1024*1024)
    assert report['status'] == 'exact_for_selected_real_inputs'
    assert len(report['arms']) == 4
    assert len(list(tmp_path.glob('*.trace.json'))) == 12
    assert report['arms'][0]['reference_cpu_bytes_retained_at_begin'] == 0
    assert all(arm['reference_cpu_bytes_retained_at_begin'] > 0 for arm in report['arms'][1:])
    assert all(arm['serialized_unit_sha256'] == report['arms'][0]['serialized_unit_sha256']
               for arm in report['arms'])


def test_energy_uses_identified_device_power_and_refuses_short_windows():
    from experiments.glm_shared_input_replay import summarize_device_energy

    def record(host, uuid, timestamp, power):
        return {'host': host, 'metrics': {f'nvidia_smi.gpu_gpu-{uuid}_power_draw': {
            'last_updated': timestamp, 'dimensions': {'power_draw': {'value': power}}}}}

    records = [record('worker', 'abc', t, p) for t, p in ((0, 10), (10, 20), (20, 20), (30, 5))]
    records += [record('other', 'xyz', t, 140) for t in (0, 10, 20, 30)]
    arms = [{'arm': 'legacy', 'begin': 5, 'end': 25, 'completed_batches': 512},
            {'arm': 'shared', 'begin': 21, 'end': 23, 'completed_batches': 512}]
    result = summarize_device_energy(records, arms, 'GPU-abc')
    assert result['host'] == 'worker'
    assert result['arms'][0]['gross_device_joules'] == 350
    assert result['arms'][0]['completed_collector_batches_per_joule'] == 512/350
    assert result['arms'][1]['gross_device_joules'] is None
