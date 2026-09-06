"""Static activation identity survives sparse, transient production renders."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import format_registry as fr
from prismaquant.joint_aura import activation_identity
from prismaquant.tessera_hessian import calibration_identity
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.streaming_production_cache import StreamedProductionAnchorRenderer
from test_streamed_cost_checkpoints import _model_identity


@pytest.mark.parametrize('one_at_a_time', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='real Tessera anchor render needs CUDA'))])
def test_tessera_only_transient_anchor_keeps_fused_static_identity(
        monkeypatch, one_at_a_time, device):
    import prismaquant.streaming_production_cache as streaming
    import prismaquant.export_native_compressed as export

    fmt = 'TESSERA_E2M1_K1_R768'
    spec = fr.get_format(fmt)
    assert spec.static_activation_contract.measured_as_served
    model = nn.Module()
    model.model = nn.Module()
    model.model.config = SimpleNamespace(layer_types=())
    layer = nn.Module()
    layer.self_attn = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    modules = {}
    maxima = {}
    for leaf, maximum in [('q_proj', 2.0), ('k_proj', 8.0), ('v_proj', 4.0)]:
        module = nn.Linear(256, 16, bias=False, device=device)
        setattr(layer.self_attn, leaf, module)
        name = 'model.layers.0.self_attn.' + leaf
        modules[name] = module
        maxima[name] = maximum
    profile = DefaultProfile()
    assert len({profile.fused_sibling_group(name) for name in modules}) == 1

    class Activations:
        def __contains__(self, name):
            return name in maxima

        def load_with_row_indices(self, name):
            return torch.full((4, 256), maxima[name]), None

    # The CPU arm isolates activation plumbing; the GPU arm uses the actual
    # Tessera weight renderer and the same static activation scoring oracle.
    if device == 'cpu':
        monkeypatch.setattr(streaming, 'render_production_weight',
                            lambda weight, *_args, **_kwargs: weight.detach().clone())
    def refuse_native_weight_globals(*_args, **_kwargs):
        raise AssertionError('Tessera-only plan requested native NVFP4 weight globals')
    monkeypatch.setattr(export, '_compute_nvfp4_joint_global', refuse_native_weight_globals)
    consumer = {'schema': 'test.static-anchor-consumer.v1'}
    plan = {name: (fmt,) for name in modules}
    renderer = StreamedProductionAnchorRenderer(
        model, act_index=Activations(), formats_by_qname=plan,
        levers={
            'gptq': False, 'joint_scale_opt': False,
            'tessera_hessian_identity': calibration_identity(
                'synthetic four-row fused-static-scale fixture',
                [torch.arange(4)], fit_tokens=4),
        }, profile=profile,
        device=device, col_weights={}, cb_serialization_context=None,
        calibration_hash='c' * 64, arm_identity={'arm': 'static-anchor-regression'},
        model_identity=_model_identity('static-anchor-source'), max_act_rows=4,
        transient_consumer_identity=consumer,
    )
    expected_max = max(maxima.values())
    expected_scale = spec.static_activation_contract.require_input_global_scale(
        expected_max, qname=next(iter(modules)), consumer='test oracle')
    identities = {}
    def consume(**kwargs):
        name = kwargs['qname']
        identity = activation_identity(spec, renderer.cache.activation_max_abs, name)
        assert identity['activation_max_abs'] == expected_max
        assert identity['input_global_scale'] == expected_scale
        assert kwargs['render_score']['activation_max_abs'] == expected_max
        assert kwargs['render_score']['input_global_scale'] == expected_scale
        identities[name] = identity
        return {'consumed': True}
    requests = [{name: plan[name]} for name in modules] if one_at_a_time else [plan]
    for requested in requests:
        renderer.render_layer_transient(
            layer=0, modules=modules, formats_by_qname=requested,
            consume_render=consume, consumer_identity=consumer)
        assert renderer.cache.weights == {}
    assert set(identities) == set(modules)
