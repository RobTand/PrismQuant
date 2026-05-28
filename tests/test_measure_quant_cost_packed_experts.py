from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    _batched_quantize,
    _finalize_results,
    _measure_packed_experts,
    _packed_experts_forward_with_weights,
    _packed_router_topk,
)


class _StubHDetail:
    """Minimal HDetailIndex stand-in returning fixed per-channel Fisher."""

    def __init__(self, mapping: dict[str, torch.Tensor]):
        self._m = mapping

    def __contains__(self, name: str) -> bool:
        return name in self._m

    def load(self, name: str) -> torch.Tensor:
        return self._m[name]
from prismaquant.allocator_candidates import cost_entry_uses_measured_output_mse


class TinyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.top_k = 1
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        scores, indices = torch.topk(torch.softmax(logits.float(), dim=-1), 1, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return logits, scores.to(hidden_states.dtype), indices


class TinyPackedExperts(nn.Module):
    def __init__(self, hidden_size: int = 16, intermediate_size: int = 16, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(
                current_state,
                self.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states,
                self.down_proj[expert_idx],
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter(hidden_size=16, num_experts=2)
        self.experts = TinyPackedExperts()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMlp()


def _write_activation_cache(cache_dir, name: str, inputs: torch.Tensor) -> None:
    fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    torch.save({"inputs": inputs, "name": name}, cache_dir / fname)


def test_packed_expert_dloss_is_mean_field_not_product_of_sums():
    """Guard the packed-expert Δloss scale fix.

    h_em[e,m] = Σ_n grad² is the per-row SUM over in-features (channel
    accumulator). The correct on-scale Δloss is the mean-field estimate
    0.5·Σ h_em·mean_n(err²), matching the dense path's 0.5·Σ g²·err². The
    previous code multiplied by N (=in-features), turning it into a
    product-of-sums (Σg²)(Σerr²) that over-counts ~N× and over-promotes
    experts in the allocator. This test pins the mean-field value and
    rejects the ×N regression.
    """
    torch.manual_seed(7)
    model = TinyModel().eval()
    target_names = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}

    # Known per-channel Fisher [E, M] matching each packed weight's (E, M).
    h_map: dict[str, torch.Tensor] = {}
    for name in target_names:
        w = dict(model.named_parameters())[name]
        h_map[name] = torch.rand(w.size(0), w.size(1), dtype=torch.float32) + 0.1
    h_detail = _StubHDetail(h_map)

    spec = fr.get_format("NVFP4")
    accum: dict = {}
    _measure_packed_experts(
        model, target_names, [spec], "cpu", torch.float32, accum,
        act_cache=None, h_detail=h_detail,
    )
    results = _finalize_results(accum)

    for name in target_names:
        w = dict(model.named_parameters())[name].detach().float()
        n_in = w.size(-1)
        err = (w - _batched_quantize(spec, w)).float()
        h_em = h_map[name]
        expected_mean_field = float(
            0.5 * (h_em * err.pow(2).mean(dim=-1)).sum().item()
        )
        old_product_of_sums = expected_mean_field * n_in  # the ×N regression
        got = float(results[name]["NVFP4"]["predicted_dloss"])

        # Matches the mean-field (no ×N) value...
        assert abs(got - expected_mean_field) <= 1e-4 * max(expected_mean_field, 1e-12), (
            f"{name}: dloss {got} != mean-field {expected_mean_field}"
        )
        # ...and is NOT the ~N× product-of-sums (n_in=16 here, so a clear gap).
        assert got < 0.5 * old_product_of_sums, (
            f"{name}: dloss {got} looks like the ×N product-of-sums "
            f"{old_product_of_sums} (regression)"
        )


def test_packed_experts_measure_output_mse_from_expert_activation_cache(tmp_path):
    torch.manual_seed(1234)
    model = TinyModel().eval()
    experts_qname = "mlp.experts"
    target_names = {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    _write_activation_cache(tmp_path, experts_qname, torch.randn(8, 16))
    act_cache = ActivationIndex(
        tmp_path,
        {
            name: {"_packed_experts_module": experts_qname}
            for name in target_names
        },
    )

    accum: dict = {}
    _measure_packed_experts(
        model,
        target_names,
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
        "cpu",
        torch.float32,
        accum,
        act_cache=act_cache,
    )
    results = _finalize_results(accum)

    assert set(results) == target_names
    for name in target_names:
        assert results[name]["BF16"].get("output_mse_measured", True) is True
        assert results[name]["NVFP4"].get("output_mse_measured", True) is True
        assert results[name]["BF16"]["output_mse"] == 0.0
        assert results[name]["NVFP4"]["output_mse"] > 0.0
        assert cost_entry_uses_measured_output_mse(
            {"_packed_experts_module": experts_qname},
            results[name]["NVFP4"],
        )


def test_packed_experts_replay_matches_module_forward():
    torch.manual_seed(5678)
    model = TinyModel().eval()
    X = torch.randn(11, 16)

    top_k_index, top_k_weights = _packed_router_topk(model.mlp.gate, X)
    y_module = model.mlp.experts(X, top_k_index, top_k_weights)
    y_replay = _packed_experts_forward_with_weights(
        model.mlp.experts,
        X,
        top_k_index,
        top_k_weights,
        model.mlp.experts.gate_up_proj,
        model.mlp.experts.down_proj,
    )

    assert torch.allclose(y_replay, y_module)


def test_packed_experts_replay_honors_apply_gate_clamp():
    torch.manual_seed(9012)

    class ClampExperts(TinyPackedExperts):
        def __init__(self):
            super().__init__()
            self.limit = 1.0
            with torch.no_grad():
                self.gate_up_proj.mul_(4.0)

        def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            return self.act_fn(gate) * up

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                gate_up = F.linear(
                    hidden_states[token_idx],
                    self.gate_up_proj[expert_idx],
                )
                current_hidden_states = self._apply_gate(gate_up)
                current_hidden_states = F.linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                )
                current_hidden_states = (
                    current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states

    experts = ClampExperts().eval()
    X = torch.randn(13, 16)
    top_k_index = torch.randint(0, experts.num_experts, (13, 1))
    top_k_weights = torch.ones(13, 1)

    y_module = experts(X, top_k_index, top_k_weights)
    y_replay = _packed_experts_forward_with_weights(
        experts,
        X,
        top_k_index,
        top_k_weights,
        experts.gate_up_proj,
        experts.down_proj,
    )

    assert torch.allclose(y_replay, y_module, atol=1e-5)


def test_packed_router_topk_accepts_indices_weights_tuple_order():
    class SwappedRouter(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            indices = torch.zeros(hidden_states.size(0), 1, dtype=torch.long)
            weights = torch.ones(hidden_states.size(0), 1)
            return torch.empty(hidden_states.size(0), 1), indices, weights

    indices, weights = _packed_router_topk(SwappedRouter(), torch.randn(3, 16))

    assert indices.dtype == torch.long
    assert torch.equal(indices, torch.zeros(3, 1, dtype=torch.long))
    assert torch.equal(weights, torch.ones(3, 1))
