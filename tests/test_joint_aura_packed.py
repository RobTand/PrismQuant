"""Packed source views produce the same per-Linear joint residual as an oracle."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import prismaquant.aura_cost as aura
import prismaquant.format_registry as fr
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.joint_aura import validate_joint_aura_entry
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
from prismaquant.perturbed_x_cache import _activation_qdq
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from test_streamed_cost_checkpoints import _FakeStreamingContext, _model_identity


class _Experts(nn.Module):
    def __init__(self, width=16, experts=3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(experts, width * 2, width) / 8)
        self.down_proj = nn.Parameter(torch.randn(experts, width, width) / 8)
        self.captures = None

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        y = torch.zeros_like(x)
        # Every token has two explicit routes; expert 2 is never selected.
        for expert, route_weight in ((0, 0.7), (1, 0.3)):
            gate_up = F.linear(x, self.gate_up_proj[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            down = F.linear(hidden, self.down_proj[expert])
            if self.captures is not None:
                gate_up.retain_grad()
                down.retain_grad()
                self.captures.append((expert, x.detach(), gate_up, hidden.detach(), down))
            y = y + route_weight * down
        return y.reshape(shape)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Module()
        self.feed_forward.experts = _Experts()

    def forward(self, hidden_states, **kwargs):
        experts = self.feed_forward.experts
        # Repeated calls must be added while signed within each logical Linear.
        return torch.tanh(0.2 * hidden_states + experts(hidden_states) + 0.25 * experts(-hidden_states))


class _Model(nn.Module):
    def __init__(self, state=None):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(layer_types=())
        self.model.embed_tokens = nn.Embedding(23, 16)
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(16, 23, bias=False)
        if state is not None:
            self.load_state_dict(state)

    def forward(self, ids):
        hidden = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _fixture(state=None):
    torch.manual_seed(715)
    model = _Model(state).eval()
    profile = Lfm2MoeProfile()
    context = _FakeStreamingContext(model)
    runner = StreamedCausalLM(context, profile)
    views = profile_declared_packed_expert_projections(model, profile)
    weights = {(member.qname, 'FP8_E4M3'): member.weight.detach().clone() + 0.03125
               for member in views}
    cache = ProductionWeightCache(weights=weights, levers={},
        activation_max_abs={member.qname: 1.0 for member in views})
    return model, context, runner, profile, cache, views


def _run(runner, profile, cache, **kwargs):
    return aura.compute_aura_cost_streamed(runner, torch.tensor([[1, 2, 3, 4]]),
        ['FP8_DYNAMIC', 'BF16'], n_probes=3, min_free_gib=0,
        production_cache=cache, joint_activation=True, include_routed_experts=True,
        profile=profile, model_identity=_model_identity('packed-joint-source'), **kwargs)


def test_packed_joint_rows_match_full_model_residual_oracle():
    model, context, runner, profile, cache, views = _fixture()
    state = copy.deepcopy(model.state_dict())
    payload = _run(runner, profile, cache)
    assert set(payload['costs']) == {member.qname for member in views}
    assert context.active == set()
    oracle = _Model(state).eval()
    by_name = {member.qname: member for member in profile_declared_packed_expert_projections(oracle, profile)}
    observed_nonzero_cross_term = False
    for probe_index in range(3):
        for layer in oracle.model.layers:
            layer.feed_forward.experts.captures = []
        oracle.zero_grad(set_to_none=True)
        logits = oracle(torch.tensor([[1, 2, 3, 4]])).logits
        fisher_probe_scalar(logits, seed=7000 + probe_index, token_scope='all',
            temperature=1.0, distribution='rademacher').backward()
        expected = {name: 0.0 for name in by_name}
        for layer_index, layer in enumerate(oracle.model.layers):
            for expert, x, gu, h, down in layer.feed_forward.experts.captures:
                for role, inputs, gradient in (
                    ('w1', x, gu.grad[:, :16]),
                    ('w3', x, gu.grad[:, 16:]),
                    ('w2', h, down.grad),
                ):
                    name = f'model.layers.{layer_index}.feed_forward.experts.{expert}.{role}'
                    weight = by_name[name].weight.detach().double()
                    rendered = cache.get(name, 'FP8_E4M3').double()
                    qx = _activation_qdq(inputs, fr.get_format('FP8_E4M3'), cache.activation_max_abs, name)
                    residual = qx.double() @ rendered.T - inputs.double() @ weight.T
                    expected[name] += float((gradient.double() * residual).sum())
        for name, rows in payload['costs'].items():
            row = rows['FP8_E4M3']
            assert validate_joint_aura_entry(row)
            assert row['joint_operator_identity']['qname'] == name
            assert row['probe_ids'] == [7000, 7001, 7002]
            assert row['signed_per_probe'][probe_index] == pytest.approx(expected[name], rel=3e-5, abs=3e-8)
            assert rows['BF16']['signed_per_probe'] == [0.0, 0.0, 0.0]
            if '.experts.2.' in name:
                assert row['signed_per_probe'] == [0.0, 0.0, 0.0]
            observed_nonzero_cross_term |= any(abs(term['mixed']) > 0 for term in row['signed_components_per_probe'])
    assert observed_nonzero_cross_term


def test_packed_joint_checkpoint_resume_keeps_aligned_unit_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1' * 40)
    _, _, runner, profile, cache, _ = _fixture()
    first = _run(runner, profile, cache, checkpoint_dir=tmp_path)
    _, context, runner, profile, cache, _ = _fixture()
    second = _run(runner, profile, cache, checkpoint_dir=tmp_path, resume=True)
    assert first['costs'] == second['costs']
    assert context.install_calls == 0


def test_packed_views_refresh_after_install_and_release_after_unload():
    import weakref
    model, context, runner, profile, cache, _ = _fixture()
    sources = [{name: value.detach().clone() for name, value in layer.named_parameters()}
               for layer in model.model.layers]
    released = []
    original_install, original_unload = context.install, context.unload
    def install(index, **kwargs):
        assert all(reference() is None for reference in released), \
            'logical projection retained a previous streamed source leaf'
        layer = model.model.layers[index]
        for name, value in sources[index].items():
            owner, _, attr = name.rpartition('.')
            layer.get_submodule(owner)._parameters[attr] = nn.Parameter(value.clone(), requires_grad=False)
        return original_install(index, **kwargs)
    def unload(index):
        layer = model.model.layers[index]
        released.extend(weakref.ref(parameter) for parameter in layer.parameters())
        original_unload(index)
        layer.to_empty(device='meta')
    context.install, context.unload = install, unload
    payload = _run(runner, profile, cache)
    assert len(payload['costs']) == 18
    assert all(reference() is None for reference in released)
    assert all(parameter.is_meta for layer in model.model.layers for parameter in layer.parameters())


def test_packed_observer_restores_forward_and_linear_after_backward_failure(monkeypatch):
    import prismaquant.joint_aura as joint
    model, context, runner, profile, cache, _ = _fixture()
    original_linear = F.linear
    original_forwards = [layer.feed_forward.experts.forward for layer in model.model.layers]
    def fail(*args, **kwargs):
        raise RuntimeError('QDQ failure sentinel')
    monkeypatch.setattr(joint, '_activation_qdq', fail)
    with pytest.raises(RuntimeError, match='QDQ failure sentinel'):
        _run(runner, profile, cache)
    assert F.linear is original_linear
    assert context.active == set()
    assert [layer.feed_forward.experts.forward for layer in model.model.layers] == original_forwards
    assert all(parameter.grad is None for parameter in model.parameters())


def test_packed_observer_refuses_a_source_that_escapes_linear_boundaries(monkeypatch):
    def matmul_forward(self, x):
        gate, up = (x @ self.gate_up_proj[0].T).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down_proj[0].T
    monkeypatch.setattr(_Experts, 'forward', matmul_forward)
    model, context, runner, profile, cache, _ = _fixture()
    original_linear = F.linear
    with pytest.raises(RuntimeError, match='did not execute declared F.linear boundaries'):
        _run(runner, profile, cache)
    assert F.linear is original_linear
    assert context.active == set()
    assert all(layer.feed_forward.experts.forward.__func__ is matmul_forward
               for layer in model.model.layers)


def test_packed_joint_refuses_dense_only_renderer_before_source_forward():
    _, context, runner, profile, cache, _ = _fixture()
    with pytest.raises(RuntimeError, match='requires decoded ProductionWeightCache'):
        _run(runner, profile, cache, anchor_renderer=object())
    assert context.install_calls == 0
