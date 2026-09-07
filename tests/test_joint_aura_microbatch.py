from __future__ import annotations

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant.kl_fisher import fisher_probe_scalar
from test_joint_aura_streamed import _fixture
from test_streamed_cost_checkpoints import _model_identity


@pytest.mark.parametrize('scope', ['all', 'last', 'causal'])
def test_row_indexed_fisher_partition_invariance(scope):
    torch.manual_seed(80)
    logits = torch.randn(5, 4, 13, requires_grad=True)
    tokens = 5 * {'all': 4, 'last': 1, 'causal': 3}[scope]
    kwargs = dict(seed=7000, token_scope=scope, distribution='rademacher',
                  token_count_override=tokens)
    full = fisher_probe_scalar(logits, global_row_offset=0, **kwargs)
    expected, = torch.autograd.grad(full, logits)
    for size in (1, 2, 3):
        actual = torch.zeros_like(logits)
        for start in range(0, 5, size):
            part = logits[start:start + size].detach().requires_grad_(True)
            scalar = fisher_probe_scalar(part, global_row_offset=start, **kwargs)
            actual[start:start + size], = torch.autograd.grad(scalar, part)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _run(size, *, checkpoint_dir=None, resume=False, capture_sizes=None):
    _, context, runner, cache = _fixture()
    if capture_sizes is not None:
        capture = runner.capture_boundaries
        def observed(ids):
            capture_sizes.append(len(ids))
            return capture(ids)
        runner.capture_boundaries = observed
    return aura.compute_aura_cost_streamed(
        runner, torch.tensor([[1, 2, 3, 4]] * 5),
        ['FP8_DYNAMIC', 'NVFP4A16', 'BF16'], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity('joint-source'), probe_microbatch=size,
        checkpoint_dir=checkpoint_dir, resume=resume, collect_col_energy=True,
    ), context


def test_joint_microbatch_signed_sum_and_uneven_partition_invariance():
    full, _ = _run(5)
    for size in (1, 2, 3):
        captures = []
        actual, context = _run(size, capture_sizes=captures)
        assert max(captures) <= size
        assert sum(captures) == 5
        assert context.active == set()
        actual_identity = actual['provenance']['probe_identity']
        full_identity = full['provenance']['probe_identity']
        assert actual_identity['noise_layout'] == full_identity['noise_layout']
        assert actual_identity['arithmetic']['execution_partition']['effective_rows'] == size
        assert actual_identity != full_identity
        for name, stats in full['stats'].items():
            assert actual['stats'][name]['h_trace'] == pytest.approx(stats['h_trace'], rel=2e-5)
            torch.testing.assert_close(actual['stats'][name]['fisher_col'], stats['fisher_col'], rtol=2e-5, atol=1e-10)
        for name, rows in full['costs'].items():
            for fmt, expected in rows.items():
                row = actual['costs'][name][fmt]
                assert row['signed_per_probe'] == pytest.approx(expected['signed_per_probe'], rel=2e-5, abs=2e-9)
                assert row['x2_per_probe'] == pytest.approx(expected['x2_per_probe'], rel=4e-5, abs=1e-12)


def test_microbatch_checkpoint_refuses_changed_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1' * 40)
    first, _ = _run(2, checkpoint_dir=tmp_path)
    resumed, context = _run(2, checkpoint_dir=tmp_path, resume=True)
    assert first['costs'] == resumed['costs']
    assert context.install_calls == 0
    with pytest.raises(RuntimeError, match='identity mismatch'):
        _run(3, checkpoint_dir=tmp_path, resume=True)


def test_packed_joint_microbatch_matches_independent_row_oracle():
    from test_joint_aura_packed import _fixture as packed_fixture, _Model
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.perturbed_x_cache import _activation_qdq
    from prismaquant import format_registry as fr
    import copy

    model, _, runner, profile, cache, _ = packed_fixture()
    oracle = _Model(copy.deepcopy(model.state_dict())).eval()
    ids = torch.tensor([[1, 2, 3, 4]] * 3 + [[4, 3, 2, 1]] * 2)
    payload = aura.compute_aura_cost_streamed(
        runner, ids, ['FP8_DYNAMIC', 'BF16'], n_probes=3, min_free_gib=0,
        production_cache=cache, joint_activation=True, include_routed_experts=True,
        profile=profile, model_identity=_model_identity('packed-joint-source'),
        probe_microbatch=2, collect_col_energy=True)
    views = {p.qname: p for p in profile_declared_packed_expert_projections(oracle, profile)}
    nonzero_cross_terms = False
    for probe_index in range(3):
        row_terms = {name: [] for name in views}
        oracle.zero_grad(set_to_none=True)
        for row_index in range(len(ids)):
            for layer in oracle.model.layers:
                layer.feed_forward.experts.captures = []
            logits = oracle(ids[row_index:row_index + 1]).logits
            fisher_probe_scalar(logits, seed=7000 + probe_index, token_scope='all',
                                distribution='rademacher', global_row_offset=row_index,
                                token_count_override=ids.numel()).backward()
            terms = {name: 0.0 for name in views}
            for depth, layer in enumerate(oracle.model.layers):
                for expert, x, gu, h, down in layer.feed_forward.experts.captures:
                    for role, inputs, gradient in (('w1', x, gu.grad[:, :16]),
                                                   ('w3', x, gu.grad[:, 16:]),
                                                   ('w2', h, down.grad)):
                        name = f'model.layers.{depth}.feed_forward.experts.{expert}.{role}'
                        weight = views[name].weight.detach().double()
                        rendered = cache.get(name, 'FP8_E4M3').double()
                        qx = _activation_qdq(inputs, fr.get_format('FP8_E4M3'), cache.activation_max_abs, name)
                        residual = qx.double() @ rendered.T - inputs.double() @ weight.T
                        terms[name] += float((gradient.double() * residual).sum())
            for name in views:
                row_terms[name].append(terms[name])
        for name, samples in row_terms.items():
            row = payload['costs'][name]['FP8_E4M3']
            expected = sum(samples)
            assert row['signed_per_probe'][probe_index] == pytest.approx(expected, rel=4e-5, abs=4e-9)
            assert row['x2_per_probe'][probe_index] == pytest.approx(expected ** 2, rel=8e-5, abs=1e-11)
            nonzero_cross_terms |= abs(expected ** 2 - sum(x*x for x in samples)) > 1e-7
    assert nonzero_cross_terms, 'fixture must distinguish signed-sum-square from sum-of-squares'


@pytest.mark.parametrize('consumer', ['candidate', 'assignment', 'mixed_assignment'])
def test_joint_consumers_refuse_different_execution_partition(consumer):
    from prismaquant.joint_aura import (
        paired_candidate_difference, paired_assignment_difference,
        assignment_probe_summary,
    )
    full, _ = _run(5)
    partitioned, _ = _run(2)
    a = {name: rows['FP8_E4M3'] for name, rows in full['costs'].items()}
    b = {name: rows['FP8_E4M3'] for name, rows in partitioned['costs'].items()}
    names = list(a)
    with pytest.raises(ValueError, match='probe alignment mismatch'):
        if consumer == 'candidate':
            paired_candidate_difference(a[names[0]], b[names[0]])
        elif consumer == 'assignment':
            paired_assignment_difference(a, b)
        else:
            assignment_probe_summary({names[0]: a[names[0]], names[1]: b[names[1]]})
