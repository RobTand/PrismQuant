"""Fail-closed projection selection, prewarm and serialized arithmetic policy."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import joint_projection_backend as backend
from prismaquant import format_registry as fr
from prismaquant.joint_aura import SignedJointProjectionLease, arithmetic_identity, identity_sha256


def selector(path='missing.so'):
    qualification, _ = backend._qualification()
    return {'name': backend.FUSED_NAME,
            'binary': {'path': str(path), 'sha256': qualification['build']['binary_sha256']}}


def test_reference_remains_available_without_cuda_compiler_or_artifact(monkeypatch):
    monkeypatch.setattr(backend, '_qualification', lambda: pytest.fail('reference read fused qualification'))
    monkeypatch.setattr(backend.kernel, 'load_backend', lambda: pytest.fail('reference compiled GPU code'))
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: pytest.fail('reference inspected GPU runtime'))
    prepared = backend.prewarm_projection_backend(None, device='cpu')
    layer = torch.nn.Linear(8, 4, bias=False)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8).requires_grad_()
    delta = torch.ones_like(layer.weight)
    with SignedJointProjectionLease({'unit': layer}, {'unit': {'BF16': fr.get_format('BF16')}},
                                    {('unit', 'BF16'): delta}, projection_backend=prepared) as lease:
        lease.begin_probe()
        layer(x).sum().backward()
        values = lease.finish_probe()['unit', 'BF16']
    assert values['weight'] == float((torch.ones((2, 4)).T @ x.detach() * delta).sum())
    assert values['activation'] == values['mixed'] == 0.0
    assert prepared.identity == arithmetic_identity(torch.float32)['projection_backend'] == backend.REFERENCE_IDENTITY


@pytest.mark.parametrize('config', ['fused_fp32_v1', {}, {'name': 'surprise'}, {'name': 'torch', 'enable': True},
    {'name': 'fused_fp32_v1'}, {'name': 'fused_fp32_v1', 'binary': {'path': '/tmp/x', 'sha256': 'a' * 64}},
    {'name': 'fused_fp32_v1', 'binary': {'path': '/tmp/x', 'sha256': None}}])
def test_selector_refuses_unknown_or_unqualified_inputs(config):
    with pytest.raises(ValueError):
        backend.normalize_projection_backend(config)


def test_fused_lease_refuses_missing_explicit_prewarm():
    layer = torch.nn.Linear(8, 4, bias=False)
    with pytest.raises(RuntimeError, match='explicitly prewarmed before the lease'):
        SignedJointProjectionLease({'unit': layer}, {'unit': {'BF16': fr.get_format('BF16')}},
            {('unit', 'BF16'): torch.zeros_like(layer.weight)}, projection_backend=selector())
    assert not layer._forward_hooks


@pytest.mark.parametrize('field', ['torch', 'torch_git', 'cuda', 'machine', 'device', 'headers', 'compiler'])
def test_prewarm_refuses_each_unqualified_runtime_dimension(monkeypatch, field):
    qualification, _ = backend._qualification()
    actual = deepcopy(qualification['runtime'])
    actual[field] = 'different-runtime'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: actual)
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: pytest.fail('unqualified runtime loaded code'))
    with pytest.raises(RuntimeError, match='unqualified runtime identity: ' + field):
        backend.prewarm_projection_backend(selector(), device='cuda:0')


def qualify_mock_runtime(monkeypatch):
    qualification, _ = backend._qualification()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: deepcopy(qualification['runtime']))
    monkeypatch.setattr(backend.kernel, '_source_digest', lambda: qualification['build']['source_sha256'])
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: pytest.fail('invalid input loaded code'))
    return qualification


@pytest.mark.parametrize('field', ['CUDA_FLAGS', 'CPP_FLAGS', 'source'])
def test_prewarm_refuses_changed_source_or_build_flags(monkeypatch, field):
    qualify_mock_runtime(monkeypatch)
    if field == 'source':
        monkeypatch.setattr(backend.kernel, '_source_digest', lambda: '0' * 64)
    else:
        monkeypatch.setattr(backend.kernel, field, getattr(backend.kernel, field) + ['unqualified'])
    with pytest.raises(RuntimeError, match='source/compiler flags differ'):
        backend.prewarm_projection_backend(selector(), device='cuda:0')


def test_binary_bytes_are_checked_before_loading(tmp_path, monkeypatch):
    qualify_mock_runtime(monkeypatch)
    path = tmp_path / 'foreign.so'
    path.write_bytes(b'not the hash-bound qualified binary')
    with pytest.raises(RuntimeError, match='binary bytes differ'):
        backend.prewarm_projection_backend(selector(path), device='cuda:0')


def test_missing_artifact_cannot_trigger_implicit_build(tmp_path, monkeypatch):
    qualify_mock_runtime(monkeypatch)
    monkeypatch.setattr(backend.kernel, 'load_backend', lambda: pytest.fail('missing artifact triggered JIT'))
    with pytest.raises(FileNotFoundError):
        backend.prewarm_projection_backend(selector(tmp_path / 'missing.so'), device='cuda:0')


def test_serialized_backend_identity_is_complete_and_changes_probe_contract():
    qualification, digest = backend._qualification()
    identity = {'schema': backend.SCHEMA, 'name': backend.FUSED_NAME,
                'qualification_sha256': digest, 'build': qualification['build'],
                'runtime': qualification['runtime'], 'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    backend.validate_projection_backend_identity(identity)
    fused = arithmetic_identity(torch.float32, SimpleNamespace(identity=identity))
    assert identity_sha256(fused) != identity_sha256(arithmetic_identity(torch.float32))
    for field in identity:
        invalid = deepcopy(identity)
        invalid.pop(field)
        with pytest.raises(ValueError, match='unqualified backend identity'):
            backend.validate_projection_backend_identity(invalid)


def test_legacy_v1_signed_rows_require_fresh_recomputation():
    from test_joint_aura_assignment_diagnostics import _row
    from prismaquant.joint_aura import validate_joint_aura_entry
    # This test uses the same valid row fixture as assignment diagnostics.
    row = _row("model.layers.0.mlp.down_proj", [1.0, 2.0])
    row['joint_operator_identity']['schema'] = 'prismaquant.joint_aura.operator.v1'
    row['joint_operator_identity_sha256'] = identity_sha256(row['joint_operator_identity'])
    with pytest.raises(ValueError, match='legacy artifacts require fresh prepare and recompute'):
        validate_joint_aura_entry(row)


def test_loader_cannot_substitute_another_binary(tmp_path, monkeypatch):
    qualification = qualify_mock_runtime(monkeypatch)
    intended = tmp_path / 'qualified.so'
    intended.write_bytes(b'fixture, hash mocked below')
    foreign = tmp_path / 'substituted.so'
    foreign.write_bytes(b'foreign fixture')
    monkeypatch.setattr(backend, '_sha', lambda path: qualification['build']['binary_sha256']
                        if Path(path) == intended else '0' * 64)
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: SimpleNamespace(__file__=str(foreign)))
    monkeypatch.setattr(backend.importlib.machinery.ExtensionFileLoader, 'exec_module', lambda *_: None)
    with pytest.raises(RuntimeError, match='actually loaded a different binary'):
        backend.prewarm_projection_backend(selector(intended), device='cuda:0')


def test_prewarmer_device_scope_cannot_be_reused_for_another_device():
    fused = backend._FusedProjection(None, torch.device('cuda:0'), {'qualified_shapes': [[2, 2]]},
                                    seal=backend._PREWARM_SEAL)
    with pytest.raises(RuntimeError, match='not prewarmed for this device'):
        backend.require_prewarmed_projection(fused, device='cuda:1')


def _plan(tmp_path):
    from prismaquant.tessera_joint_aura import SCHEMA
    return {'schema': SCHEMA, 'model': 'fixture', 'inputs': {}, 'output_root': str(tmp_path),
        'calibration_input': {'path': 'fixture', 'sha256': 'a' * 64},
        'execution': {'n_calib_samples': 512, 'calib_seqlen': 512, 'probe_microbatch': 1,
                      'n_probes': 4, 'seed_base': 7000, 'token_scope': 'all', 'temperature': 1.0,
                      'production_act_scales': '0'}, 'profile_tool': 'cprofile',
        'max_render_bytes': 1024, 'max_gpu_bytes': 2048, 'min_free_gib': 0,
        'source_prefetch': {'max_cache_slots': 24, 'prefetch_workers': 4, 'prefetch_lookahead': 4,
                           'cache_headroom_gb': 4.0, 'prefetch_min_available_gb': 2.0,
                           'require_prefetched_residency': True}}


def test_plan_admits_reference_default_but_refuses_unknown_backend(tmp_path):
    from prismaquant.tessera_joint_aura import _load_plan, _sha
    config = _plan(tmp_path)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(config))
    assert _load_plan(path, _sha(path)) == config
    config['execution']['projection_backend'] = {'name': 'unqualified'}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unsupported joint projection backend'):
        _load_plan(path, _sha(path))


@pytest.mark.parametrize('legacy_schema', [True, False])
def test_cost_refuses_legacy_or_backend_changed_preparation_before_cache_adoption(tmp_path, monkeypatch, legacy_schema):
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    from prismaquant import model_profiles, aura_cost, joint_aura
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_: None)
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: object())
    draw = dict(fit_ids_sha256='a' * 64, text_sha256='b' * 64, nsamples=512, seqlen=512, seed=0)
    calibration = {'provenance': draw}
    source = {'fixture': 'source-model-identity'}
    execution = {'fixture': 'source-execution'}
    data = SimpleNamespace(census={'model': 'fixture', 'attention_implementation': 'eager'},
        payload={'provenance': {'hessian': {'calibration_identity': draw}}},
        layer_render_bytes=lambda _: {0: 64}, formats_by_qname={'unit': ['BF16']}, cells={('unit', 'BF16'): {}})
    monkeypatch.setattr(bridge, 'load_measured_anchor_input', lambda *_args, **_kwargs: data)
    monkeypatch.setattr(calibration_data, 'load_calibration_input', lambda *_args, **_kwargs:
        (torch.zeros((512, 512), dtype=torch.int64), calibration))
    runner = SimpleNamespace(model=object(), layer_index_for_qname=lambda _: 0, shutdown=lambda: None)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(cost_streaming, 'build_streamed_model_identity', lambda *_args, **_kwargs: source)
    monkeypatch.setattr(joint_aura, 'source_execution_identity', lambda _: execution)
    monkeypatch.setattr(aura_cost, '_aura_source_sha256', lambda: 'c' * 64)
    monkeypatch.setattr(aura_cost, 'compute_aura_cost_streamed', lambda *_args, **_kwargs:
        pytest.fail('foreign preparation reached the adjoint'))
    monkeypatch.setattr(bridge.pickle, 'loads', lambda _: pytest.fail('foreign preparation adopted PWC'))
    record = {'schema': 'prismaquant.tessera_joint_aura.prepared.v1' if legacy_schema else bridge.PREPARED_SCHEMA,
        'status': 'complete', 'plan_sha256': 'd' * 64, 'implementation_sha256': 'c' * 64,
        'source_model_identity': source, 'source_execution': execution, 'calibration_input': calibration,
        'measured_cells': 1, 'reader_identity': None, 'projection_backend': {'name': 'foreign'}}
    path = tmp_path / 'prepared.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='fresh prepare and recompute' if legacy_schema else 'prepared projection_backend'):
        bridge.execute('run', _plan(tmp_path), plan_sha256='d' * 64,
                       prepared={'path': str(path), 'sha256': bridge._sha(path)})
    assert json.loads((tmp_path / 'run/results.json').read_text())['passed'] is False


def test_repeated_row_admission_uses_prewarmed_metadata_without_file_io(monkeypatch):
    backend._qualification.cache_clear()
    qualification, digest = backend._qualification()
    identity = {'schema': backend.SCHEMA, 'name': backend.FUSED_NAME,
                'qualification_sha256': digest, 'build': qualification['build'],
                'runtime': qualification['runtime'], 'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    monkeypatch.setattr(Path, 'read_bytes', lambda _: pytest.fail('row admission reopened package metadata'))
    backend.validate_projection_backend_identity(identity)
    backend.validate_projection_backend_identity(deepcopy(identity))
