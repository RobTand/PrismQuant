"""Packed capture must follow the real parent-owned LFM router bias."""
import pytest
import torch
from torch import nn

from prismaquant.measure_quant_cost import derive_per_expert_activations, _packed_router_topk


def _real_block(use_bias=True):
    from transformers import Lfm2MoeConfig
    from transformers.models.lfm2_moe import modeling_lfm2_moe as module
    if not hasattr(module, 'Lfm2MoeTopKRouter'):
        pytest.skip('requires Transformers parent-owned LFM router API (5.15.1 qualified)')
    config = Lfm2MoeConfig(hidden_size=8, moe_intermediate_size=4, num_experts=2,
                          num_experts_per_tok=1, use_expert_bias=use_bias)
    block = module.Lfm2MoeSparseMoeBlock(config).eval()
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[0].fill_(1)
        if use_bias:
            block.expert_bias.copy_(torch.tensor([-2., 2.]))
        block.experts.gate_up_proj.fill_(0.1)
        block.experts.down_proj.fill_(0.2)
    return block


def test_real_lfm_parent_bias_capture_matches_forward():
    block = _real_block()
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    _, weights, indices = block.gate(x, block.expert_bias)
    _, _, no_bias_indices = block.gate(x, torch.zeros_like(block.expert_bias))
    assert torch.equal(indices, torch.ones(3, 1, dtype=torch.long))
    assert torch.equal(no_bias_indices, torch.zeros(3, 1, dtype=torch.long))
    seen = []
    handle = block.experts.register_forward_pre_hook(lambda _module, args: seen.append(args))
    try:
        block(x.unsqueeze(0))
    finally:
        handle.remove()
    assert torch.equal(seen[0][1], indices)
    assert torch.equal(seen[0][2], weights)
    got = derive_per_expert_activations(block.experts, x, block)
    assert got['row_counts'] == [0, 3]
    assert torch.equal(got['gate_up'][1], x)
    assert torch.equal(got['gate_weights'][1], weights[:, 0])
    gate, up = torch.nn.functional.linear(x, block.experts.gate_up_proj[1]).chunk(2, -1)
    torch.testing.assert_close(got['down'][1], block.experts.act_fn(gate) * up)
    from prismaquant.expert_empirical_cost import _replay_down_proj_col_weights
    col_weights = _replay_down_proj_col_weights(block.experts, block, block.gate, x)
    torch.testing.assert_close(col_weights[1, 0], got['down'][1].square().mean(0))


def test_real_lfm_disabled_parent_bias_still_captures():
    block = _real_block(False)
    x = torch.ones(3, 8)
    got = derive_per_expert_activations(block.experts, x, block)
    assert got['row_counts'] == [3, 0]


def test_external_bias_router_without_parent_bias_fails_clearly():
    class ExternalBiasRouter(nn.Module):
        use_expert_bias = True
        def forward(self, hidden_states, expert_bias=None):
            raise AssertionError('must reject missing parent bias before router execution')
    with pytest.raises(ValueError, match='expert_bias.*parent'):
        _packed_router_topk(ExternalBiasRouter(), torch.ones(3, 8))


def test_existing_correction_bias_router_remains_supported():
    class CorrectionRouter(nn.Module):
        def forward(self, hidden_states, e_score_correction_bias):
            logits = hidden_states[:, :2] + e_score_correction_bias
            weights, indices = logits.softmax(-1).topk(1, -1)
            return logits, weights, indices
    x = torch.tensor([[2., 0.], [3., 1.]])
    bias = torch.tensor([-10., 10.])
    indices, weights = _packed_router_topk(CorrectionRouter(), x, bias)
    assert torch.equal(indices, torch.ones(2, 1, dtype=torch.long))
    torch.testing.assert_close(weights, (x + bias).softmax(-1)[:, 1:])
