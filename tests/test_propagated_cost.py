import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant import propagated_cost as pc
from prismaquant.allocator_solver import Candidate
from prismaquant.build_rtn_cache import cache_reference_log_probs
from prismaquant.layer_state_cache import LayerHiddenStateCache
from prismaquant.perturbed_x_cache import PerturbedActivationCache
from prismaquant.propagated_cost import (
    FrozenBudgetError,
    GPUMemoryBudgetExceeded,
    L3UnsupportedTargetError,
    L3NeighborhoodEntry,
    build_l3_candidates,
    measure_override_paired_kl_deltas,
    measure_propagated_costs,
    select_formats_for_l3,
    select_l3_neighborhood,
    solve_frozen_l3_neighborhood,
    tail_forward_from_layer,
)


def _specs():
    return [fr.get_format(n) for n in ("NVFP4", "MXFP8", "BF16")]


def test_quant_weight_cache_can_resolve_bf16_from_source_weight():
    model = nn.Module()
    model.proj = nn.Linear(4, 3, bias=False)
    model.proj.weight.data.zero_()
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    cache = pc.build_quant_weight_cache(
        model,
        [
            L3NeighborhoodEntry(
                name="proj",
                current_format="NVFP4",
                formats=("BF16",),
                margin=0.0,
                l2_current_cost=0.0,
            )
        ],
        [fr.get_format("BF16")],
        skip_bf16=False,
        source_weight_resolver=lambda name, fmt: (
            source if (name, fmt) == ("proj", "BF16") else None
        ),
    )

    torch.testing.assert_close(cache.get("proj", "BF16"), source)


class _ResolverBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _ResolverLM(nn.Module):
    def __init__(self, *, vocab: int = 4, hidden: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_ResolverBlock(hidden)])
        self.model.norm = nn.Identity()

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=hidden)


def test_override_kl_uses_source_resolver_when_live_weight_is_floor(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(123)
    model = _ResolverLM().eval()
    calib_ids = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    source = torch.eye(4, dtype=torch.float32)
    target_name = "model.layers.0.proj"
    model.model.layers[0].proj.weight.data.copy_(source)
    with torch.no_grad():
        teacher_logits = model(calib_ids).logits[:, -1:, :]
        ref_log_probs = [
            torch.nn.functional.log_softmax(
                teacher_logits[i:i + 1].float(),
                dim=-1,
            )
            for i in range(calib_ids.size(0))
        ]
    model.model.layers[0].proj.weight.data.zero_()
    monkeypatch.setenv("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
    monkeypatch.setenv("PRISMAQUANT_L3_PREQUANT_CACHE", "0")

    measured = pc.measure_override_set_kl(
        model,
        {target_name: "NVFP4"},
        [{target_name: "BF16"}],
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=1,
        kl_scope="last_token",
        include_activation_quant=False,
        source_weight_resolver=lambda name, fmt: (
            source if (name, fmt) == (target_name, "BF16") else None
        ),
    )

    assert measured == pytest.approx([0.0], abs=1e-7)


def test_cuda_graph_auto_mode_requires_enough_repeated_calls(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_L3_CUDA_GRAPHS", raising=False)
    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_L3_CUDA_GRAPHS",
        call_count=4,
        min_calls=8,
    ) is False
    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_L3_CUDA_GRAPHS",
        call_count=8,
        min_calls=8,
    ) is True

    monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "1")
    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_L3_CUDA_GRAPHS",
        call_count=1,
        min_calls=8,
    ) is True

    monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "0")
    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_L3_CUDA_GRAPHS",
        call_count=64,
        min_calls=8,
    ) is False


def test_cuda_graph_auto_mode_honors_min_calls_override(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_COORD_LANE_CUDA_GRAPHS", raising=False)
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_CUDA_GRAPHS_MIN_CALLS", "3")

    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
        call_count=2,
        min_calls=16,
    ) is False
    assert pc._env_cuda_graphs_enabled_for_call_count(
        "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
        call_count=3,
        min_calls=16,
    ) is True


def test_l3_host_memory_floor_raises_before_oom(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_L3_MIN_HOST_MEM_GB", "24")
    monkeypatch.setattr(pc, "_host_available_memory_gb", lambda: 3.0)

    with pytest.raises(GPUMemoryBudgetExceeded, match="host MemAvailable"):
        pc._enforce_l3_host_memory_floor(
            phase="paired_override_kl",
            chunk_index=3,
        )


def test_l3_host_memory_floor_reduces_pair_lanes(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_L3_MIN_HOST_MEM_GB", "24")
    monkeypatch.setattr(pc, "_host_available_memory_gb", lambda: 46.0)

    assert pc._adjust_l3_max_lanes_for_host_floor(
        8,
        phase="paired_override_kl",
        chunk_index=3,
    ) == 4

    monkeypatch.setattr(pc, "_host_available_memory_gb", lambda: 68.0)
    assert pc._adjust_l3_max_lanes_for_host_floor(
        8,
        phase="paired_override_kl",
        chunk_index=4,
    ) == 6


def _stat(n_params=128 * 128):
    return {
        "n_params": n_params,
        "in_features": 128,
        "out_features": 128,
        "h_trace": 1.0,
        "_memory_bytes_by_format": {
            "NVFP4": int(n_params * 4 / 8),
            "MXFP8": int(n_params * 8 / 8),
            "BF16": int(n_params * 16 / 8),
        },
    }


def _cost_table(current=1.0, cheaper=1.04):
    return {
        "NVFP4": {"predicted_dloss": cheaper},
        "MXFP8": {"predicted_dloss": current},
        "BF16": {"predicted_dloss": 0.0},
    }


class _AmplifyingToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2, bias=False)
        self.l2 = nn.Linear(2, 2, bias=False)
        self.l3 = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.l1.weight.copy_(torch.eye(2))
            self.l2.weight.copy_(10.0 * torch.eye(2))
            self.l3.weight.copy_(torch.eye(2))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return SimpleNamespace(logits=x)


class _ReplayPastCache:
    def __init__(self):
        self.keys = {}

    def update(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        key_states = hidden_states.transpose(1, 2).contiguous()
        previous = self.keys.get(layer_idx)
        if previous is not None:
            key_states = torch.cat([previous, key_states], dim=2)
        self.keys[layer_idx] = key_states


class _CacheReplayLayer(nn.Module):
    def __init__(self, width: int, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width) * (1.0 + 0.05 * layer_idx))

    def forward(
        self,
        hidden_states,
        *,
        past_key_value=None,
        use_cache: bool = True,
    ):
        if use_cache and past_key_value is not None:
            past_key_value.update(self.layer_idx, hidden_states)
        return self.proj(torch.tanh(hidden_states))


class _CacheReplayDecoder(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CacheReplayLayer(width, idx) for idx in range(layers)]
        )
        self.norm = nn.Identity()


class _CacheReplayCausalLM(nn.Module):
    def __init__(self, *, width: int = 8, layers: int = 4):
        super().__init__()
        self.model = _CacheReplayDecoder(width, layers)
        self.lm_head = nn.Identity()

    def forward(self, input_ids, *, use_cache: bool = True):
        hidden = input_ids.float()
        past_key_value = _ReplayPastCache() if use_cache else None
        for layer in self.model.layers:
            hidden = layer(
                hidden,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class _TailLayer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(scale * torch.eye(2))

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + self.proj(hidden_states)


class _TailToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TailLayer(float(i + 1)) for i in range(4)])
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 1.5]]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))


class _TwoProjTailLayer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(scale * torch.tensor([[1.0, 0.25], [-0.5, 1.0]]))
            self.b.weight.copy_(scale * torch.tensor([[0.5, -0.75], [1.25, 0.5]]))

    def forward(self, hidden_states, **_kwargs):
        return self.b(torch.tanh(self.a(hidden_states)))


class _TwoProjTailToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.layers = nn.ModuleList([
            _TwoProjTailLayer(1.0),
            _TwoProjTailLayer(0.5),
        ])
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 1.5]]))

    def forward(self, x):
        self.forward_calls += 1
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))


def _zero_spec():
    return fr.FormatSpec(
        name="ZERO4",
        weight_bits=4,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_zero",
        quantize_dequantize=lambda w: torch.zeros_like(w),
    )


def _identity8_spec():
    return fr.FormatSpec(
        name="IDENT8",
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_identity",
        quantize_dequantize=lambda w: w.clone(),
    )


def _act_zero_identity_weight_spec():
    return fr.FormatSpec(
        name="ACTZERO8",
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_identity",
        act_bits=8,
        act_dtype_name="test_zero",
        quantize_dequantize=lambda w: w.clone(),
        activation_quantize_dequantize=lambda x: torch.zeros_like(x),
    )


def _counted_fp_spec(name: str):
    codebook = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    return fr.FormatSpec(
        name=name,
        weight_bits=4,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_counted_fp",
        quantize_dequantize=lambda w: fr._rtn_fp_codebook(w, codebook, 0),
        activation_quantize_dequantize=lambda x: x.clone(),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def test_select_formats_uses_current_neighbors_and_bf16():
    stats = {"layer": _stat()}
    costs = {
        "layer": {
            "NVFP4": {"predicted_dloss": 1.05},
            "MXFP8": {"predicted_dloss": 1.00},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    assignment = {"layer": "MXFP8"}

    got = select_formats_for_l3(stats, costs, assignment, "layer", _specs())

    assert got == ("NVFP4", "MXFP8", "BF16")


def test_tail_forward_from_layer_matches_full_forward_from_layer_output():
    model = _TailToy().eval()
    x = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    with torch.no_grad():
        hidden = x
        for idx in range(3):
            hidden = model.layers[idx](hidden)
        perturbed = hidden + torch.tensor([[0.1, -0.2], [0.05, 0.1]])

        tail_logits = tail_forward_from_layer(model, 2, (x,), {}, perturbed)

        handle = model.layers[2].register_forward_hook(
            lambda _module, _args, _output: perturbed
        )
        try:
            full_logits = model(x).logits
        finally:
            handle.remove()

    assert torch.allclose(tail_logits, full_logits, atol=1e-6, rtol=1e-6)


def test_select_formats_limits_rich_menu_to_current_neighbors_and_bf16():
    stats = {"layer": _stat()}
    formats = (
        "NVFP4",
        "MXFP6_E3M2",
        "MXFP6_E2M3",
        "FP8_E4M3",
        "FP8_E5M2",
        "MXFP8",
        "BF16",
    )
    costs = {
        "layer": {
            fmt: {"predicted_dloss": float(i)}
            for i, fmt in enumerate(formats)
        }
    }
    assignment = {"layer": "FP8_E4M3"}
    specs = [fr.get_format(name) for name in formats]

    got = select_formats_for_l3(stats, costs, assignment, "layer", specs)

    assert len(got) == 4
    assert got == ("MXFP6_E3M2", "FP8_E4M3", "FP8_E5M2", "BF16")


def test_select_l3_neighborhood_caps_and_keeps_safety_layers():
    stats = {f"layer{i}": _stat() for i in range(20)}
    costs = {}
    assignment = {}
    for i in range(20):
        name = f"layer{i}"
        assignment[name] = "NVFP4"
        costs[name] = {
            "NVFP4": {"predicted_dloss": 1.0},
            "MXFP8": {"predicted_dloss": 2.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    assignment["layer0"] = "MXFP8"
    costs["layer0"] = {
        "NVFP4": {"predicted_dloss": 1.04},
        "MXFP8": {"predicted_dloss": 1.0},
        "BF16": {"predicted_dloss": 0.0},
    }
    assignment["layer5"] = "MXFP8"
    costs["layer5"] = {
        "NVFP4": {"predicted_dloss": 2.0},
        "MXFP8": {"predicted_dloss": 1.0},
        "BF16": {"predicted_dloss": 0.0},
    }
    costs["layer19"] = {
        "NVFP4": {"predicted_dloss": 100.0},
        "MXFP8": {"predicted_dloss": 200.0},
        "BF16": {"predicted_dloss": 0.0},
    }

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=0.05,
        max_fraction=0.15,
        safety_fraction=0.05,
    )

    selected_by_name = {entry.name: entry for entry in selected}
    assert len(selected) == 3
    assert "uncertain" in selected_by_name["layer0"].reasons
    assert "confident_non_cheapest" in selected_by_name["layer5"].reasons
    assert "high_l2_cost" in selected_by_name["layer19"].reasons

    ranked_stats = {f"ranked{i}": _stat() for i in range(10)}
    ranked_costs = {}
    ranked_assignment = {}
    for i in range(10):
        name = f"ranked{i}"
        ranked_assignment[name] = "NVFP4"
        ranked_costs[name] = {
            "NVFP4": {"predicted_dloss": 1.0},
            "MXFP8": {"predicted_dloss": 2.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    for i, benefit in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
        name = f"ranked{i}"
        ranked_assignment[name] = "MXFP8"
        ranked_costs[name] = {
            "NVFP4": {"predicted_dloss": 10.0 - benefit},
            "MXFP8": {"predicted_dloss": 10.0},
            "BF16": {"predicted_dloss": 0.0},
        }

    ranked = select_l3_neighborhood(
        ranked_stats,
        ranked_costs,
        ranked_assignment,
        _specs(),
        uncertainty_rel_tol=0.0,
        min_fraction=0.0,
        max_fraction=0.30,
        safety_fraction=0.0,
    )

    assert [entry.name for entry in ranked] == ["ranked3", "ranked4", "ranked5"]
    assert all("confident_non_cheapest" in entry.reasons for entry in ranked)


def test_select_l3_neighborhood_includes_confident_non_cheapest():
    stats = {f"layer{i}": _stat() for i in range(6)}
    costs = {
        name: {
            "NVFP4": {"predicted_dloss": 10.0},
            "MXFP6_E3M2": {"predicted_dloss": 1.0},
            "MXFP8": {"predicted_dloss": 2.0},
            "BF16": {"predicted_dloss": 0.0},
        }
        for name in stats
    }
    assignment = {name: "MXFP6_E3M2" for name in stats}
    specs = [fr.get_format(n) for n in ("NVFP4", "MXFP6_E3M2", "MXFP8", "BF16")]

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        specs,
        min_fraction=0.0,
        max_fraction=1.0,
        safety_fraction=0.0,
    )

    assert {entry.name for entry in selected} == set(stats)
    assert len(selected) > 1
    assert all(entry.reasons == ("confident_non_cheapest",) for entry in selected)


def test_select_l3_neighborhood_excludes_packed_experts(capsys):
    packed = dict(_stat())
    packed.update({
        "num_experts": 4,
        "_packed_experts_module": "model.layers.0.mlp.experts",
        "_packed_param": "gate_up_proj",
    })
    stats = {
        "model.layers.0.mlp.experts.gate_up_proj": packed,
        "model.layers.0.self_attn.q_proj": _stat(),
    }
    assignment = {name: "MXFP8" for name in stats}
    costs = {name: _cost_table() for name in stats}

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=1.0,
        max_fraction=1.0,
        safety_fraction=0.0,
    )
    out = capsys.readouterr().out

    assert [entry.name for entry in selected] == ["model.layers.0.self_attn.q_proj"]
    assert "model.layers.0.mlp.experts.gate_up_proj" in out
    assert "packed MoE expert tensor" in out


def test_global_l3_neighborhood_excludes_packed_experts(capsys):
    packed = dict(_stat())
    packed.update({
        "num_experts": 4,
        "_packed_experts_module": "model.layers.0.mlp.experts",
        "_packed_param": "down_proj",
    })
    stats = {
        "model.layers.0.mlp.experts.down_proj": packed,
        "model.layers.0.self_attn.q_proj": _stat(),
    }
    assignment = {name: "MXFP8" for name in stats}
    costs = {name: _cost_table() for name in stats}

    selected = pc.build_global_l3_neighborhood(stats, costs, assignment, _specs())
    out = capsys.readouterr().out

    assert [entry.name for entry in selected] == ["model.layers.0.self_attn.q_proj"]
    assert "model.layers.0.mlp.experts.down_proj" in out
    assert "packed MoE expert tensor" in out


def test_select_l3_neighborhood_passes_dense_only():
    stats = {
        "model.layers.0.self_attn.q_proj": _stat(),
        "model.layers.0.self_attn.k_proj": _stat(),
    }
    assignment = {name: "MXFP8" for name in stats}
    costs = {name: _cost_table() for name in stats}

    selected = select_l3_neighborhood(
        stats,
        costs,
        assignment,
        _specs(),
        min_fraction=1.0,
        max_fraction=1.0,
        safety_fraction=0.0,
    )

    assert {entry.name for entry in selected} == set(stats)


def test_build_l3_candidates_uses_propagated_end_kl_only():
    stats = {"layer": _stat()}
    propagated = {
        "layer": {
            "NVFP4": {"propagated_end_kl": 0.5, "downstream_output_mse": 10.0},
            "MXFP8": {"output_mse": 0.01},
            "BF16": {"propagated_end_kl": 0.0},
        }
    }

    cands = build_l3_candidates(stats, propagated, _specs())

    assert [c.fmt for c in cands["layer"]] == ["NVFP4", "BF16"]
    assert [c.predicted_dloss for c in cands["layer"]] == [0.5, 0.0]


def test_solve_frozen_l3_neighborhood_respects_remaining_budget():
    stats = {name: _stat(n_params=100) for name in ("a", "b", "frozen")}
    assignment = {"a": "MXFP8", "b": "MXFP8", "frozen": "MXFP8"}
    candidates = {
        "a": [
            Candidate("NVFP4", 4.0, 50, 3.0),
            Candidate("MXFP8", 8.0, 100, 1.0),
            Candidate("BF16", 16.0, 200, 0.0),
        ],
        "b": [
            Candidate("NVFP4", 4.0, 50, 0.1),
            Candidate("MXFP8", 8.0, 100, 1.0),
            Candidate("BF16", 16.0, 200, 0.0),
        ],
    }

    solved, chosen = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=8.0,
        bit_precision=0.001,
    )

    assert solved["frozen"] == "MXFP8"
    assert solved["a"] == "MXFP8"
    assert solved["b"] == "NVFP4"
    assert chosen["b"].fmt == "NVFP4"


def test_solve_frozen_l3_neighborhood_falls_back_when_precision_too_tight(monkeypatch):
    stats = {f"layer{i}": _stat(n_params=100) for i in range(15)}
    assignment = {name: "NVFP4" for name in stats}
    candidates = {
        name: [
            Candidate("NVFP4", 4.0, 50, 1.0),
            Candidate("MXFP8", 8.0, 100, 0.0),
            Candidate("BF16", 16.0, 200, 2.0),
        ]
        for name in stats
    }
    monkeypatch.setattr(pc, "solve_allocation", lambda *_args, **_kwargs: None)

    solved, chosen, meta = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=4.4091,
        bit_precision=0.001,
        return_metadata=True,
    )

    used_bits = sum(8.0 * chosen[name].memory_bytes for name in chosen)
    total_params = sum(entry["n_params"] for entry in stats.values())
    assert used_bits <= 4.4091 * total_params + 1e-6
    assert list(solved.values()).count("MXFP8") == 1
    assert meta["frozen_dp_precision_used"] == "greedy"
    assert meta["frozen_dp_greedy"]["accepted"] == 1

    over_stats = {f"over{i}": _stat(n_params=100) for i in range(3)}
    over_assignment = {name: "MXFP8" for name in over_stats}
    over_candidates = {
        name: [
            Candidate("NVFP4", 4.0, 50, 1.0),
            Candidate("MXFP8", 8.0, 100, 1.0),
            Candidate("BF16", 16.0, 200, 2.0),
        ]
        for name in over_stats
    }

    solved, chosen, meta = solve_frozen_l3_neighborhood(
        over_stats,
        over_assignment,
        over_candidates,
        _specs(),
        target_bits=7.8,
        bit_precision=0.001,
        budget_tolerance=0.05,
        return_metadata=True,
    )

    used_bits = sum(8.0 * chosen[name].memory_bytes for name in chosen)
    total_params = sum(entry["n_params"] for entry in over_stats.values())
    ceiling_bits = 7.8 * total_params * 1.05
    assert used_bits <= ceiling_bits + 1e-6
    assert list(solved.values()).count("NVFP4") == 1
    assert meta["frozen_dp_precision_used"] == "greedy"
    assert meta["frozen_dp_greedy"]["accepted_budget_reducing_nonworse"] == 1


def test_solve_frozen_l3_neighborhood_greedy_swaps_dominated_current(monkeypatch):
    stats = {"layer": _stat(n_params=100)}
    assignment = {"layer": "MXFP6_E3M2"}
    candidates = {
        "layer": [
            Candidate("NVFP4", 4.0, 50, 0.1),
            Candidate("MXFP6_E3M2", 6.0, 75, 1.0),
            Candidate("BF16", 16.0, 200, 2.0),
        ]
    }
    monkeypatch.setattr(pc, "solve_allocation", lambda *_args, **_kwargs: None)

    # The target is just below the minimum-bpp candidate, so the removed
    # min-bpp shortcut would have engaged at precision=0.01 before greedy.
    solved, _chosen, meta = solve_frozen_l3_neighborhood(
        stats,
        assignment,
        candidates,
        _specs(),
        target_bits=3.995,
        bit_precision=0.001,
        budget_tolerance=0.01,
        return_metadata=True,
    )

    assert solved["layer"] == "NVFP4"
    assert meta["frozen_dp_precision_used"] == "greedy"
    assert meta["frozen_dp_greedy"]["accepted"] == 1


def test_solve_frozen_l3_neighborhood_still_raises_when_frozen_exceeds_budget():
    stats = {name: _stat(n_params=100) for name in ("open", "frozen")}
    assignment = {"open": "MXFP8", "frozen": "BF16"}
    candidates = {"open": [Candidate("NVFP4", 4.0, 50, 0.0)]}

    with pytest.raises(FrozenBudgetError, match="frozen L2 choices already exceed"):
        solve_frozen_l3_neighborhood(
            stats,
            assignment,
            candidates,
            _specs(),
            target_bits=4.0,
            bit_precision=0.5,
        )


def test_measure_propagated_costs_pairs_candidate_with_target_bf16_baseline(tmp_path):
    model = _AmplifyingToy().eval()
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    neighborhood = [
        L3NeighborhoodEntry(
            name="l1",
            current_format="BF16",
            formats=("ZERO4", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        )
    ]
    calib = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    costs = measure_propagated_costs(
        model,
        assignment,
        neighborhood,
        calib,
        [_zero_spec(), fr.get_format("BF16")],
        work_root=tmp_path,
        max_lanes_per_batch=4,
    )

    zero = costs["l1"]["ZERO4"]
    assert zero["propagated_end_kl"] > 0.1
    assert zero["downstream_output_mse"] > 0.0
    assert costs["l1"]["BF16"]["propagated_end_kl"] == 0.0


def _assert_l3_costs_close(actual, expected):
    assert set(actual) == set(expected)
    for name, per_name in expected.items():
        assert set(actual[name]) == set(per_name)
        for fmt, expected_entry in per_name.items():
            actual_entry = actual[name][fmt]
            assert set(actual_entry) == set(expected_entry)
            for key, expected_value in expected_entry.items():
                actual_value = actual_entry[key]
                if isinstance(expected_value, float):
                    assert actual_value == pytest.approx(
                        expected_value,
                        abs=1e-6,
                        rel=1e-6,
                    )
                else:
                    assert actual_value == expected_value


def test_measure_propagated_costs_cached_tail_inputs_are_equivalent(tmp_path):
    assignment = {
        "layers.0.a": "MXFP8",
        "layers.0.b": "MXFP8",
        "layers.1.a": "MXFP8",
        "layers.1.b": "MXFP8",
    }
    neighborhood = [
        L3NeighborhoodEntry(
            name="layers.0.a",
            current_format="MXFP8",
            formats=("NVFP4", "MXFP8", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        ),
        L3NeighborhoodEntry(
            name="layers.0.b",
            current_format="MXFP8",
            formats=("NVFP4", "MXFP8", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        ),
    ]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )
    specs = [fr.get_format(name) for name in ("NVFP4", "MXFP8", "BF16")]

    def _measure(*, cache_tail_layer_inputs: bool):
        model = _TwoProjTailToy().eval()
        costs = measure_propagated_costs(
            model,
            assignment,
            neighborhood,
            calib,
            specs,
            work_root=tmp_path,
            max_lanes_per_batch=3,
            tail_only=True,
            cache_tail_layer_inputs=cache_tail_layer_inputs,
        )
        return costs, model.forward_calls

    baseline, baseline_forward_calls = _measure(
        cache_tail_layer_inputs=False,
    )
    cached, cached_forward_calls = _measure(cache_tail_layer_inputs=True)
    _assert_l3_costs_close(
        cached,
        baseline,
    )
    assert cached_forward_calls < baseline_forward_calls


def test_measure_override_paired_kl_cached_tail_inputs_are_equivalent(tmp_path):
    assignment = {
        "layers.0.a": "MXFP8",
        "layers.0.b": "MXFP8",
        "layers.1.a": "MXFP8",
        "layers.1.b": "MXFP8",
    }
    overrides = [
        {"layers.1.a": "NVFP4", "layers.1.b": "NVFP4"},
        {"layers.1.a": "BF16", "layers.1.b": "NVFP4"},
        {"layers.1.a": "NVFP4", "layers.1.b": "BF16"},
    ]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )

    def _measure(*, tail_only: bool, cache_tail_layer_inputs: bool):
        model = _TwoProjTailToy().eval()
        values = measure_override_paired_kl_deltas(
            model,
            assignment,
            overrides,
            calib,
            work_root=tmp_path,
            max_lanes_per_batch=2,
            tail_only=tail_only,
            cache_tail_layer_inputs=cache_tail_layer_inputs,
        )
        return values, model.forward_calls

    baseline, baseline_forward_calls = _measure(
        tail_only=False,
        cache_tail_layer_inputs=False,
    )
    cached, cached_forward_calls = _measure(
        tail_only=True,
        cache_tail_layer_inputs=True,
    )

    assert cached == pytest.approx(baseline, abs=1e-6, rel=1e-6)
    assert cached_forward_calls < baseline_forward_calls


def test_paired_tail_cuda_graph_key_tracks_override_state(tmp_path, monkeypatch):
    assignment = {
        "layers.0.a": "MXFP8",
        "layers.0.b": "MXFP8",
        "layers.1.a": "MXFP8",
        "layers.1.b": "MXFP8",
    }
    overrides = [
        {"layers.1.a": "NVFP4", "layers.1.b": "NVFP4"},
        {"layers.1.a": "BF16", "layers.1.b": "NVFP4"},
        {"layers.1.a": "NVFP4", "layers.1.b": "BF16"},
    ]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )
    seen_state_keys = []

    class _FakeTailGraphCache:
        enabled = True

        def run(
            self,
            model,
            layer_idx,
            layer_args,
            layer_kwargs,
            hidden_state,
            *,
            lane_count,
            state_key=None,
        ):
            seen_state_keys.append(state_key)
            return pc._tail_forward_eager(
                model,
                layer_idx,
                layer_args,
                layer_kwargs,
                hidden_state,
            )

        def clear(self):
            pass

    monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "1")
    monkeypatch.setattr(
        pc,
        "_TailCudaGraphCache",
        lambda *, enabled: _FakeTailGraphCache(),
    )

    values = measure_override_paired_kl_deltas(
        _TwoProjTailToy().eval(),
        assignment,
        overrides,
        calib,
        work_root=tmp_path,
        max_lanes_per_batch=2,
        tail_only=True,
        cache_tail_layer_inputs=True,
    )

    assert len(values) == len(overrides)
    assert len(seen_state_keys) >= len(overrides)
    assert all(key is not None for key in seen_state_keys)
    assert len(set(seen_state_keys)) == len(overrides)


def test_prequant_cache_bit_exact_with_inline(tmp_path, monkeypatch):
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    neighborhood = [
        L3NeighborhoodEntry(
            name="l1",
            current_format="BF16",
            formats=("ZERO4", "IDENT8", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        )
    ]
    specs = [_zero_spec(), _identity8_spec(), fr.get_format("BF16")]
    calib = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    cache_model = _AmplifyingToy().eval()
    cache = pc.build_quant_weight_cache(cache_model, neighborhood, specs)
    for spec in specs[:-1]:
        cached = cache.cache[("l1", spec.name)]
        inline = pc.apply_format_quantization(cache_model.l1.weight.data, spec).to(
            dtype=cache_model.l1.weight.dtype,
            device=cache_model.l1.weight.device,
        ).contiguous()
        assert torch.equal(cached, inline)
        assert _tensor_digest(cached) == _tensor_digest(inline)

    monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "0")
    monkeypatch.setenv("PRISMAQUANT_L3_PREQUANT_CACHE", "0")
    inline_costs = measure_propagated_costs(
        _AmplifyingToy().eval(),
        assignment,
        neighborhood,
        calib,
        specs,
        work_root=tmp_path,
        max_lanes_per_batch=4,
    )
    monkeypatch.setenv("PRISMAQUANT_L3_PREQUANT_CACHE", "1")
    cached_costs = measure_propagated_costs(
        _AmplifyingToy().eval(),
        assignment,
        neighborhood,
        calib,
        specs,
        work_root=tmp_path,
        max_lanes_per_batch=4,
    )

    _assert_l3_costs_close(cached_costs, inline_costs)


def test_l3_lane_prequant_cache_avoids_per_forward_weight_rtn(tmp_path, monkeypatch):
    specs = [
        _counted_fp_spec("COUNTED_FP4_A"),
        _counted_fp_spec("COUNTED_FP4_B"),
        fr.get_format("BF16"),
    ]
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    neighborhood = [
        L3NeighborhoodEntry(
            name=name,
            current_format="BF16",
            formats=("COUNTED_FP4_A", "COUNTED_FP4_B", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        )
        for name in ("l1", "l2")
    ]
    calls = {"count": 0}
    original = fr._rtn_fp_codebook

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "0")
    monkeypatch.setenv("PRISMAQUANT_L3_PREQUANT_CACHE", "1")
    monkeypatch.setattr(fr, "_rtn_fp_codebook", _counted)

    measure_propagated_costs(
        _AmplifyingToy().eval(),
        assignment,
        neighborhood,
        torch.randn(5, 2),
        specs,
        work_root=tmp_path,
        max_lanes_per_batch=8,
        tail_only=False,
    )

    assert calls["count"] == 4


def test_l3_frozen_perturbed_cache_avoids_per_forward_context_rtn(
    tmp_path,
    monkeypatch,
):
    spec = _counted_fp_spec("COUNTED_CONTEXT_FP4")
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)

    class _WideToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(33, 33, bias=False)
            self.l2 = nn.Linear(33, 33, bias=False)
            self.l3 = nn.Linear(33, 33, bias=False)

        def forward(self, x):
            return SimpleNamespace(logits=self.l3(self.l2(self.l1(x))))

    calls = {"count": 0}
    original = fr._rtn_fp_codebook

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fr, "_rtn_fp_codebook", _counted)

    def _run(cache_enabled: bool) -> int:
        calls["count"] = 0
        monkeypatch.setenv("PRISMAQUANT_L3_CUDA_GRAPHS", "0")
        monkeypatch.setenv(
            "PRISMAQUANT_L3_FROZEN_PERTURBED_CACHE",
            "1" if cache_enabled else "0",
        )
        measure_propagated_costs(
            _WideToy().eval(),
            {"l1": "BF16", "l2": spec.name, "l3": spec.name},
            [
                L3NeighborhoodEntry(
                    name="l1",
                    current_format="BF16",
                    formats=("ZERO4", "BF16"),
                    margin=0.0,
                    l2_current_cost=0.0,
                )
            ],
            torch.randn(4, 33),
            [_zero_spec(), fr.get_format("BF16")],
            work_root=tmp_path,
            max_lanes_per_batch=2,
            tail_only=False,
        )
        return calls["count"]

    assert _run(cache_enabled=True) == 2
    assert _run(cache_enabled=False) == 8


def test_perturbed_frozen_weight_cache_bit_exact_and_reused(tmp_path, monkeypatch):
    spec = _counted_fp_spec("COUNTED_PERTURBED_FP4")
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    model = nn.Sequential(
        nn.Linear(33, 33, bias=False),
        nn.Linear(33, 33, bias=False),
    ).eval()
    assignment = {"0": spec.name, "1": spec.name}
    builder = PerturbedActivationCache(
        model,
        assignment,
        tmp_path,
        input_rows=0,
        cal_hash="test",
    )
    calls = {"count": 0}
    original = fr._rtn_fp_codebook

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fr, "_rtn_fp_codebook", _counted)
    with builder.frozen_weight_cache():
        assert calls["count"] == 2
        for idx, layer in enumerate(model):
            cached = builder._frozen_weight_cache[(id(layer), "weight")]
            inline = spec.quantize_dequantize(layer.weight.data.detach().clone()).to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            ).contiguous()
            assert torch.equal(cached, inline), idx
        assert calls["count"] == 4
        builder.install()
        try:
            x = torch.randn(1, 33)
            for _ in range(5):
                model(x)
        finally:
            builder.remove()

    assert calls["count"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graphs_bit_exact_with_eager(tmp_path, monkeypatch):
    assignment = {"layers.0.proj": "BF16"}
    neighborhood = [
        L3NeighborhoodEntry(
            name="layers.0.proj",
            current_format="BF16",
            formats=("ZERO4", "BF16"),
            margin=0.0,
            l2_current_cost=0.0,
        )
    ]
    specs = [_zero_spec(), fr.get_format("BF16")]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )

    def _measure(graphs_enabled: bool):
        monkeypatch.setenv(
            "PRISMAQUANT_L3_CUDA_GRAPHS",
            "1" if graphs_enabled else "0",
        )
        return measure_propagated_costs(
            _TailToy().eval().cuda(),
            assignment,
            neighborhood,
            calib,
            specs,
            work_root=tmp_path,
            max_lanes_per_batch=4,
            tail_only=True,
            output_mse_names=[],
        )

    eager = _measure(False)
    graphed = _measure(True)
    _assert_l3_costs_close(graphed, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coord_lane_batched_with_cuda_graphs_matches_eager(tmp_path, monkeypatch):
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    candidate_flips = [("l1", zero.name), ("l2", zero.name)]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )

    def _measure(graphs_enabled: bool):
        monkeypatch.setenv(
            "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
            "1" if graphs_enabled else "0",
        )
        model = _AmplifyingToy().eval().cuda()
        ref_log_probs = cache_reference_log_probs(
            model,
            calib,
            next(model.parameters()).device,
        )
        return pc.measure_lane_batched_kl_deltas(
            model,
            assignment,
            candidate_flips,
            calib,
            ref_log_probs,
            work_root=tmp_path,
            max_lanes_per_batch=2,
        )

    eager = _measure(False)
    graphed = _measure(True)
    assert graphed == pytest.approx(eager, abs=1e-9, rel=0.0)


def test_coord_lane_batched_mixed_depth_matches_sequential(tmp_path, monkeypatch):
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_CUDA_GRAPHS", "0")
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    candidate_flips = [("l1", zero.name), ("l2", zero.name)]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )

    def _measure(max_lanes: int):
        torch.manual_seed(123)
        model = _AmplifyingToy().eval()
        ref_log_probs = cache_reference_log_probs(
            model,
            calib,
            next(model.parameters()).device,
        )
        return pc.measure_lane_batched_kl_deltas(
            model,
            assignment,
            candidate_flips,
            calib,
            ref_log_probs,
            work_root=tmp_path,
            max_lanes_per_batch=max_lanes,
            use_cuda_graphs=False,
        )

    sequential = _measure(1)
    batched = _measure(2)
    assert batched == pytest.approx(sequential, abs=1e-9, rel=0.0)


def test_override_set_kl_batches_multi_target_lanes(tmp_path, monkeypatch):
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_CUDA_GRAPHS", "0")
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    overrides = [
        {"l1": zero.name, "l2": zero.name},
        {"l2": zero.name, "l3": zero.name},
    ]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )

    def _measure(max_lanes: int):
        torch.manual_seed(321)
        model = _AmplifyingToy().eval()
        ref_log_probs = cache_reference_log_probs(
            model,
            calib,
            next(model.parameters()).device,
        )
        return pc.measure_override_set_kl(
            model,
            assignment,
            overrides,
            calib,
            ref_log_probs,
            work_root=tmp_path,
            max_lanes_per_batch=max_lanes,
            use_cuda_graphs=False,
        )

    sequential = _measure(1)
    batched = _measure(2)
    assert batched == pytest.approx(sequential, abs=1e-9, rel=0.0)


def test_override_set_kl_uses_production_weight_cache_for_targets(tmp_path, monkeypatch):
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_CUDA_GRAPHS", "0")
    assignment = {"l1": "BF16", "l2": "BF16", "l3": "BF16"}
    overrides = [{"l1": zero.name}]
    calib = torch.tensor(
        [[1.0, -1.0], [0.5, -0.25], [-0.75, 0.25]],
        dtype=torch.float32,
    )
    model = _AmplifyingToy().eval()
    ref_log_probs = cache_reference_log_probs(
        model,
        calib,
        next(model.parameters()).device,
    )

    class _ProductionCache:
        activation_max_abs = {}
        activation_scales = {}

        def get(self, name, fmt):
            if name == "l1" and fmt == zero.name:
                return model.l1.weight.detach().clone()
            return None

    raw = pc.measure_override_set_kl(
        model,
        assignment,
        overrides,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=1,
        use_cuda_graphs=False,
    )
    cleaned = pc.measure_override_set_kl(
        model,
        assignment,
        overrides,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=1,
        use_cuda_graphs=False,
        production_weight_cache=_ProductionCache(),
    )

    assert raw[0] > 1e-6
    assert cleaned == pytest.approx([0.0], abs=1e-9, rel=0.0)


def test_override_set_replay_matches_eager(tmp_path, monkeypatch):
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    assignment = {
        f"model.layers.{idx}.proj": "BF16"
        for idx in range(4)
    }
    overrides = [
        {
            "model.layers.1.proj": zero.name,
            "model.layers.2.proj": zero.name,
        },
        {"model.layers.2.proj": zero.name},
    ]
    calib = torch.linspace(-1.0, 1.0, steps=3 * 5 * 8).reshape(3, 5, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare():
        torch.manual_seed(9876)
        model = _CacheReplayCausalLM(width=8, layers=4).eval().to(device)
        ref_log_probs = cache_reference_log_probs(model, calib, device)
        return model, ref_log_probs

    model, ref_log_probs = _prepare()
    eager = pc.measure_override_set_kl(
        model,
        assignment,
        overrides,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=2,
        include_activation_quant=False,
        use_cuda_graphs=False,
    )

    model, ref_log_probs = _prepare()
    replay_cache = LayerHiddenStateCache(model)
    replay_cache.populate(
        assignment,
        calib,
        device=str(device),
        dtype=torch.float32,
        include_activation_quant=False,
    )
    replay = pc.measure_override_set_kl(
        model,
        assignment,
        overrides,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=2,
        replay_cache=replay_cache,
        include_activation_quant=False,
        use_cuda_graphs=False,
        use_replay_cache=True,
    )

    assert replay == pytest.approx(eager, abs=1e-9, rel=0.0)


def test_coord_lane_replay_graph_capture_no_shape_mismatch(tmp_path, monkeypatch):
    pc._COORD_LANE_CUDA_GRAPH_REGISTRY.clear()
    pc._CUDA_GRAPH_WARNED_LABELS.clear()
    zero = _zero_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "1")
    assignment = {
        f"model.layers.{idx}.proj": "BF16"
        for idx in range(4)
    }
    candidate_flips = [
        ("model.layers.1.proj", zero.name),
        ("model.layers.2.proj", zero.name),
    ]
    calib = torch.linspace(-1.0, 1.0, steps=3 * 5 * 8).reshape(3, 5, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare():
        torch.manual_seed(1234)
        model = _CacheReplayCausalLM(width=8, layers=4).eval().to(device)
        ref_log_probs = cache_reference_log_probs(model, calib, device)
        replay_cache = LayerHiddenStateCache(model)
        replay_cache.populate(
            assignment,
            calib,
            device=str(device),
            dtype=torch.float32,
        )
        return model, ref_log_probs, replay_cache

    def _measure(graphs_enabled: bool):
        monkeypatch.setenv(
            "PRISMAQUANT_COORD_LANE_CUDA_GRAPHS",
            "1" if graphs_enabled else "0",
        )
        model, ref_log_probs, replay_cache = _prepare()
        return pc.measure_lane_batched_kl_deltas(
            model,
            assignment,
            candidate_flips,
            calib,
            ref_log_probs,
            work_root=tmp_path,
            max_lanes_per_batch=2,
            replay_cache=replay_cache,
        )

    eager = _measure(False)
    warnings = []
    monkeypatch.setattr(
        pc,
        "_warn_cuda_graph_fallback_once",
        lambda label, exc: warnings.append((label, exc)),
    )
    graphed = _measure(True)

    assert graphed == pytest.approx(eager, abs=1e-9, rel=0.0)
    assert not warnings
    if torch.cuda.is_available():
        assert pc._COORD_LANE_CUDA_GRAPH_REGISTRY.entries
        assert not pc._COORD_LANE_CUDA_GRAPH_REGISTRY.disabled_keys


def test_coord_lane_replay_respects_disabled_activation_quant(tmp_path, monkeypatch):
    zero = _zero_spec()
    act_zero = _act_zero_identity_weight_spec()
    monkeypatch.setitem(fr.REGISTRY, zero.name, zero)
    monkeypatch.setitem(fr.REGISTRY, act_zero.name, act_zero)
    assignment = {
        f"model.layers.{idx}.proj": act_zero.name
        for idx in range(3)
    }
    candidate_flips = [("model.layers.1.proj", zero.name)]
    calib = torch.linspace(-1.0, 1.0, steps=2 * 4 * 8).reshape(2, 4, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare():
        torch.manual_seed(4321)
        model = _CacheReplayCausalLM(width=8, layers=3).eval().to(device)
        ref_log_probs = cache_reference_log_probs(model, calib, device)
        return model, ref_log_probs

    model, ref_log_probs = _prepare()
    eager = pc.measure_lane_batched_kl_deltas(
        model,
        assignment,
        candidate_flips,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=1,
        include_activation_quant=False,
        use_cuda_graphs=False,
    )

    model, ref_log_probs = _prepare()
    replay_cache = LayerHiddenStateCache(model)
    replay_cache.populate(
        assignment,
        calib,
        device=str(device),
        dtype=torch.float32,
        include_activation_quant=False,
    )
    replay = pc.measure_lane_batched_kl_deltas(
        model,
        assignment,
        candidate_flips,
        calib,
        ref_log_probs,
        work_root=tmp_path,
        max_lanes_per_batch=1,
        replay_cache=replay_cache,
        include_activation_quant=False,
        use_cuda_graphs=False,
        use_replay_cache=True,
    )

    assert replay == pytest.approx(eager, abs=1e-9, rel=0.0)


def test_lane_batch_memory_check_falls_back(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_L3_MAX_LANES_MEM_HEADROOM_GB", "8")
    monkeypatch.setattr(pc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        pc.torch.cuda,
        "mem_get_info",
        lambda _device=None: (1, 16 * 1024 ** 3),
    )

    adjusted = pc._adjust_l3_max_lanes_for_memory(
        8,
        torch.ones(2, 4),
        torch.device("cuda"),
    )

    assert adjusted == 4


def test_toy_l3_propagation_differs_from_local_cost_and_flips_pick(tmp_path):
    model = _AmplifyingToy().eval()
    stats = {
        "l1": {
            "n_params": 4,
            "in_features": 2,
            "out_features": 2,
            "h_trace": 1.0,
            "_memory_bytes_by_format": {
                "ZERO4": 2,
                "IDENT8": 4,
                "BF16": 8,
            },
        }
    }
    l2_costs = {
        "l1": {
            "ZERO4": {"predicted_dloss": 0.01},
            "IDENT8": {"predicted_dloss": 0.02},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    assignment = {"l1": "ZERO4", "l2": "BF16", "l3": "BF16"}
    specs = [_zero_spec(), _identity8_spec(), fr.get_format("BF16")]
    selected = select_l3_neighborhood(
        stats,
        l2_costs,
        {"l1": "ZERO4"},
        specs,
        min_fraction=1.0,
        max_fraction=1.0,
    )
    calib = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)

    l3_costs = measure_propagated_costs(
        model,
        assignment,
        selected,
        calib,
        specs,
        work_root=tmp_path,
        max_lanes_per_batch=8,
    )
    l3_candidates = build_l3_candidates(stats, l3_costs, specs)
    solved, _chosen = solve_frozen_l3_neighborhood(
        stats,
        {"l1": "ZERO4"},
        l3_candidates,
        specs,
        target_bits=8.0,
        bit_precision=0.5,
    )

    assert l3_costs["l1"]["ZERO4"]["propagated_end_kl"] != pytest.approx(
        l2_costs["l1"]["ZERO4"]["predicted_dloss"]
    )
    assert l3_costs["l1"]["ZERO4"]["propagated_end_kl"] > (
        l3_costs["l1"]["IDENT8"]["propagated_end_kl"]
    )
    assert solved["l1"] == "IDENT8"
