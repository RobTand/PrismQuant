from __future__ import annotations

from copy import deepcopy
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.prismasnap import PrismaSnapSearchConfig
import prismaquant.prismasnap_moe as moe
import prismaquant.prismasnap_moe_checkpoint as checkpoint


class _CensusSource:
    """Minimal live-source stand-in for ``_layer_source_graph``'s binding leg.

    Derives every header from the same census the graph reads, so an
    agreeing checkpoint needs no duplicated shape literals.  ``overrides``
    injects a disagreeing header; ``missing`` injects the residency refusal
    ``_Checkpoint.metadata`` raises when a shard is not local.
    """

    def __init__(self, tensor_rows, *, overrides=None, missing=()):
        self._rows = tensor_rows
        self._overrides = dict(overrides or {})
        self._missing = set(missing)
        self.reads: list[str] = []

    def metadata(self, key: str) -> tuple[tuple[int, ...], str]:
        self.reads.append(key)
        if key in self._missing:
            raise RuntimeError(
                f"PrismaSnap tensor {key!r} lacks required local shards"
            )
        if key in self._overrides:
            return self._overrides[key]
        row = self._rows.get(key)
        if row is None:
            raise RuntimeError(f"PrismaSnap graph references absent tensor {key!r}")
        return tuple(int(value) for value in row["shape"]), str(row["dtype"])


def _config() -> PrismaSnapSearchConfig:
    return PrismaSnapSearchConfig(
        alphas=(0.0,),
        max_rounds=1,
        stage=False,
        polish=False,
        polish_top=0,
        polish_pool=0,
    )


def test_fp64_router_routes_and_complete_shared_gated_output_are_invariant() -> None:
    generator = torch.Generator().manual_seed(20260825)
    tokens, hidden, experts, intermediate, shared_intermediate = 5, 16, 4, 16, 32
    x = torch.randn(tokens, hidden, generator=generator, dtype=torch.float64)
    gamma = torch.randn(hidden, generator=generator, dtype=torch.float64)
    router = torch.randn(experts, hidden, generator=generator, dtype=torch.float64)
    gate_up = torch.randn(
        experts, 2 * intermediate, hidden, generator=generator, dtype=torch.float64
    )
    down = torch.randn(
        experts, hidden, intermediate, generator=generator, dtype=torch.float64
    )
    shared_gate_up = torch.randn(
        2 * shared_intermediate, hidden, generator=generator, dtype=torch.float64
    )
    shared_down = torch.randn(
        hidden, shared_intermediate, generator=generator, dtype=torch.float64
    )
    shared_output_gate = torch.randn(
        1, hidden, generator=generator, dtype=torch.float64
    )
    norm_scale = torch.tensor([0.5, 1.0, 2.0, 4.0] * 4, dtype=torch.float64)
    routed_scales = torch.tensor(
        [[0.25, 0.5, 1.0, 2.0] * 4] * experts, dtype=torch.float64
    )
    shared_scale = torch.tensor(
        [0.25, 0.5, 1.0, 2.0] * 8, dtype=torch.float64
    )

    result = moe.fp64_router_and_expert_invariance(
        hidden=x,
        norm_gamma=gamma,
        norm_scale=norm_scale,
        router_weight=router,
        gate_up=gate_up,
        down=down,
        expert_scales=routed_scales,
        top_k=2,
        shared_gate_up=[shared_gate_up],
        shared_down=[shared_down],
        shared_output_gate_weight=[shared_output_gate],
        shared_expert_scales=[shared_scale],
    )

    assert result["routing_changed"] == 0.0
    assert result["router_logit_max_abs"] <= 1e-10
    assert result["route_weight_max_abs"] <= 1e-10
    assert result["routed_output_max_abs"] <= 1e-10


def test_packed_expert_materialization_uses_declared_axes_and_only_up_rows() -> None:
    experts, hidden, intermediate = 2, 16, 16
    gate_up = torch.arange(
        experts * 2 * intermediate * hidden, dtype=torch.float64
    ).reshape(experts, 2 * intermediate, hidden) + 1.0
    down = torch.arange(
        experts * hidden * intermediate, dtype=torch.float64
    ).reshape(experts, hidden, intermediate) + 1.0
    scales = torch.stack(
        (
            torch.tensor([0.5, 1.0, 2.0, 4.0] * 4),
            torch.tensor([4.0, 2.0, 1.0, 0.5] * 4),
        )
    ).to(torch.float64)

    folded_gate_up = moe.apply_packed_expert_slice_transform(
        gate_up,
        scales,
        expert_axis=0,
        channel_axis=1,
        expert=None,
        channel_start=intermediate,
        channel_stop=2 * intermediate,
        operation="multiply",
    )
    folded_down = moe.apply_packed_expert_slice_transform(
        down,
        scales,
        expert_axis=0,
        channel_axis=2,
        expert=None,
        channel_start=0,
        channel_stop=intermediate,
        operation="divide",
    )
    assert torch.equal(folded_gate_up[:, :intermediate], gate_up[:, :intermediate])
    assert torch.equal(
        folded_gate_up[:, intermediate:],
        gate_up[:, intermediate:] * scales[:, :, None],
    )
    assert torch.equal(folded_down, down / scales[:, None, :])

    x = torch.linspace(-1.0, 1.0, hidden, dtype=torch.float64).repeat(3, 1)
    for expert in range(experts):
        gate, up = (x @ gate_up[expert].T).chunk(2, dim=-1)
        before = (F.silu(gate) * up) @ down[expert].T
        folded_gate, folded_up = (x @ folded_gate_up[expert].T).chunk(2, dim=-1)
        after = (F.silu(folded_gate) * folded_up) @ folded_down[expert].T
        torch.testing.assert_close(after, before, rtol=1e-13, atol=1e-7)


def test_expert_search_uses_exact_post_folded_bf16_bytes_and_joint_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(7)
    experts, hidden, intermediate = 2, 16, 16
    gate_up_weight = torch.randn(
        experts, 2 * intermediate, hidden, generator=generator
    ).to(torch.bfloat16)
    down_weight = torch.randn(
        experts, hidden, intermediate, generator=generator
    ).to(torch.bfloat16)
    gate_up_importance = torch.rand(experts, hidden, generator=generator) + 0.1
    down_importance = torch.rand(experts, intermediate, generator=generator) + 0.1
    post_scale = torch.linspace(0.55, 1.45, hidden, dtype=torch.float64)
    packed_gate_up = moe.PackedGateUp(
        name="model.layers.0.mlp.experts.gate_up_proj",
        weight=gate_up_weight,
        importance=gate_up_importance,
        gate_rows=(0, intermediate),
        up_rows=(intermediate, 2 * intermediate),
    )
    packed_down = moe.PackedDown(
        name="model.layers.0.mlp.experts.down_proj",
        weight=down_weight,
        importance=down_importance,
    )
    captured: list[list[object]] = []

    def fake_search(consumers, reference, *, config):
        del config
        captured.append(list(consumers))
        return torch.ones(reference.numel(), dtype=torch.float64), {}

    monkeypatch.setattr(moe, "search_diagonal_scale", fake_search)
    scales, _records = moe.search_packed_up_down_scales(
        packed_gate_up,
        packed_down,
        config=_config(),
        post_norm_scale=post_scale,
    )

    assert tuple(scales.shape) == (experts, intermediate)
    expected_weight = (
        gate_up_weight.to(torch.float64) / post_scale.view(1, 1, -1)
    ).to(torch.bfloat16)
    expected_importance = (
        gate_up_importance.to(torch.float64) * post_scale.square().view(1, -1)
    ).to(gate_up_importance.dtype)
    for expert, consumers in enumerate(captured):
        gate, up, down_consumer = consumers
        assert gate.mode == "stationary"
        assert up.mode == "row"
        assert down_consumer.mode == "column_inverse"
        assert gate.codec_group == up.codec_group
        assert down_consumer.codec_group != up.codec_group
        assert torch.equal(gate.weight, expected_weight[expert, :intermediate])
        assert torch.equal(up.weight, expected_weight[expert, intermediate:])
        assert torch.equal(gate.importance, expected_importance[expert])
        assert torch.equal(up.importance, expected_importance[expert])


def _packed_qwen_graph_inputs():
    profile = Qwen3_5Profile()
    layer, hidden, intermediate, experts = 0, 16, 32, 2
    contract = profile.prismasnap_moe_layer_contract(layer)
    assert contract is not None

    def source(recipe: str) -> str:
        return profile.source_tensor_name(recipe)

    tensor_rows: dict[str, dict[str, object]] = {}

    def tensor(recipe: str, shape: tuple[int, ...]) -> None:
        tensor_rows[source(recipe)] = {"shape": list(shape), "dtype": "BF16"}

    prefix = f"model.layers.{layer}"
    mlp = f"{prefix}.mlp"
    tensor(f"{prefix}.input_layernorm.weight", (hidden,))
    tensor(f"{prefix}.post_attention_layernorm.weight", (hidden,))
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensor(f"{prefix}.self_attn.{leaf}.weight", (hidden, hidden))
    tensor(f"{mlp}.gate.weight", (experts, hidden))
    tensor(f"{mlp}.shared_expert_gate.weight", (1, hidden))
    # Direct packed Parameters intentionally have no trailing `.weight`.
    tensor(f"{mlp}.experts.gate_up_proj", (experts, 2 * intermediate, hidden))
    tensor(f"{mlp}.experts.down_proj", (experts, hidden, intermediate))
    for leaf in ("gate_proj", "up_proj"):
        tensor(f"{mlp}.shared_expert.{leaf}.weight", (intermediate, hidden))
    tensor(f"{mlp}.shared_expert.down_proj.weight", (hidden, intermediate))

    qkv_importance = np.linspace(0.5, 1.5, hidden, dtype=np.float32)
    o_importance = np.linspace(1.5, 0.5, hidden, dtype=np.float32)
    stats: dict[str, dict[str, object]] = {}
    for leaf in ("q_proj", "k_proj", "v_proj"):
        stats[f"{prefix}.self_attn.{leaf}"] = {
            "in_features": hidden,
            "out_features": hidden,
            "act_sq_sum": qkv_importance.copy(),
        }
    stats[f"{prefix}.self_attn.o_proj"] = {
        "in_features": hidden,
        "out_features": hidden,
        "act_sq_sum": o_importance,
    }
    expert_tokens = np.asarray([3.0, 5.0], dtype=np.float64)
    stats[f"{mlp}.experts.gate_up_proj"] = {
        "in_features": hidden,
        "out_features": 2 * intermediate,
        "num_experts": experts,
        "expert_act_sq_sum": np.ones((experts, hidden), dtype=np.float32),
        "expert_tokens": expert_tokens.copy(),
        "_packed_param": "gate_up_proj",
    }
    stats[f"{mlp}.experts.down_proj"] = {
        "in_features": intermediate,
        "out_features": hidden,
        "num_experts": experts,
        "expert_act_sq_sum": np.ones((experts, intermediate), dtype=np.float32),
        "expert_tokens": expert_tokens.copy(),
        "_packed_param": "down_proj",
    }
    for leaf, width in (
        ("gate_proj", hidden),
        ("up_proj", hidden),
        ("down_proj", intermediate),
    ):
        stats[f"{mlp}.shared_expert.{leaf}"] = {
            "in_features": width,
            "out_features": intermediate if leaf != "down_proj" else hidden,
            "act_sq_sum": np.ones(width, dtype=np.float32),
        }
    return profile, stats, tensor_rows, hidden, experts


def test_qwen_profile_graph_accepts_direct_packed_parameters_without_weight() -> None:
    profile, stats, tensor_rows, hidden, experts = _packed_qwen_graph_inputs()
    contract = checkpoint._validate_profile_contract(
        profile.prismasnap_moe_layer_contract(0), layer=0, profile=profile
    )
    assert contract["router"].endswith("mlp.gate.weight")
    assert contract["packed_routed"]["gate_up"].endswith("gate_up_proj")
    assert not contract["packed_routed"]["gate_up"].endswith(".weight")
    assert contract["shared_experts"][0]["output_gate"].endswith(
        "shared_expert_gate.weight"
    )

    graph = checkpoint._layer_source_graph(
        layer=0,
        hidden_size=hidden,
        expected_experts=experts,
        stats=stats,
        profile=profile,
        source=_CensusSource(tensor_rows),
        tensor_rows=tensor_rows,
    )
    assert graph["routed_layout"] == "packed_3d"
    assert str(graph["packed_gate_up"]).endswith("experts.gate_up_proj")
    assert not str(graph["packed_gate_up"]).endswith(".weight")
    assert len(graph["input_consumers"]) == 3
    assert all("o_proj" not in name for name in graph["input_consumers"])
    expected_graph_sha = checkpoint._moe_plan_graph_sha256(
        layer=0,
        input_norm=str(graph["input_norm"]),
        norm_parameter_offset=float(graph["norm_parameter_offset"]),
        input_consumers=graph["input_consumers"],
        post_norm=str(graph["post_norm"]),
        router=str(graph["router"]),
        routed_layout=str(graph["routed_layout"]),
        experts=int(graph["experts"]),
        intermediate=int(graph["intermediate"]),
        packed_gate_up=str(graph["packed_gate_up"]),
        packed_down=str(graph["packed_down"]),
        per_expert_roles=[],
        shared_roles=graph["shared_roles"],
    )
    assert graph["graph_sha256"] == expected_graph_sha
    mutated_shared = deepcopy(graph["shared_roles"])
    mutated_shared[0]["output_gate"] += ".wrong"
    assert checkpoint._moe_plan_graph_sha256(
        layer=0,
        input_norm=str(graph["input_norm"]),
        norm_parameter_offset=float(graph["norm_parameter_offset"]),
        input_consumers=graph["input_consumers"],
        post_norm=str(graph["post_norm"]),
        router=str(graph["router"]),
        routed_layout=str(graph["routed_layout"]),
        experts=int(graph["experts"]),
        intermediate=int(graph["intermediate"]),
        packed_gate_up=str(graph["packed_gate_up"]),
        packed_down=str(graph["packed_down"]),
        per_expert_roles=[],
        shared_roles=mutated_shared,
    ) != expected_graph_sha


def test_per_expert_source_layout_reuses_packed_live_probe_and_rejects_mixing() -> None:
    profile, stats, tensor_rows, hidden, experts = _packed_qwen_graph_inputs()
    rows = deepcopy(tensor_rows)
    mlp_recipe = "model.layers.0.mlp"
    mlp = profile.source_tensor_name(mlp_recipe)
    rows.pop(f"{mlp}.experts.gate_up_proj")
    rows.pop(f"{mlp}.experts.down_proj")
    for expert in range(experts):
        for leaf, shape in (
            ("gate_proj", (32, hidden)),
            ("up_proj", (32, hidden)),
            ("down_proj", (hidden, 32)),
        ):
            recipe = f"{mlp_recipe}.experts.{expert}.{leaf}.weight"
            rows[profile.source_tensor_name(recipe)] = {
                "shape": list(shape),
                "dtype": "BF16",
            }
    graph = checkpoint._layer_source_graph(
        layer=0,
        hidden_size=hidden,
        expected_experts=experts,
        stats=stats,
        profile=profile,
        source=_CensusSource(rows),
        tensor_rows=rows,
    )
    assert graph["routed_layout"] == "per_expert_2d"
    assert graph["routed_probe"]["topology"] == "packed_live_model"
    assert len(graph["per_expert_roles"]) == experts

    rows[f"{mlp}.experts.gate_up_proj"] = {
        "shape": [experts, 64, hidden],
        "dtype": "BF16",
    }
    rows[f"{mlp}.experts.down_proj"] = {
        "shape": [experts, hidden, 32],
        "dtype": "BF16",
    }
    with pytest.raises(RuntimeError, match="mixed packed and per-expert"):
        checkpoint._layer_source_graph(
            layer=0,
            hidden_size=hidden,
            expected_experts=experts,
            stats=stats,
            profile=profile,
            source=_CensusSource(rows),
            tensor_rows=rows,
        )


@pytest.mark.parametrize("failure", ["shared_gate_shape", "packed_bias", "unknown_consumer"])
def test_qwen_packed_graph_fails_closed_on_shape_bias_and_unknown_consumers(
    failure: str,
) -> None:
    profile, stats, tensor_rows, hidden, experts = _packed_qwen_graph_inputs()
    rows = deepcopy(tensor_rows)
    mlp = profile.source_tensor_name("model.layers.0.mlp")
    if failure == "shared_gate_shape":
        rows[f"{mlp}.shared_expert_gate.weight"]["shape"] = [2, hidden]
        match = "shared expert 0 shape/dtype"
    elif failure == "packed_bias":
        rows[f"{mlp}.experts.gate_up_proj.bias"] = {
            "shape": [64],
            "dtype": "BF16",
        }
        match = "projection bias"
    else:
        rows[f"{mlp}.mystery_proj.weight"] = {
            "shape": [32, hidden],
            "dtype": "BF16",
        }
        match = "does not close every MLP hidden-input consumer"
    with pytest.raises(RuntimeError, match=match):
        checkpoint._layer_source_graph(
            layer=0,
            hidden_size=hidden,
            expected_experts=experts,
            stats=stats,
            profile=profile,
            source=_CensusSource(rows),
            tensor_rows=rows,
        )


def _graph_kwargs(profile, stats, rows, hidden, experts, source):
    return {
        "layer": 0,
        "hidden_size": hidden,
        "expected_experts": experts,
        "stats": stats,
        "profile": profile,
        "source": source,
        "tensor_rows": rows,
    }


def test_moe_graph_refuses_a_census_that_the_live_source_does_not_confirm() -> None:
    """Defect 1: `source` is the live-header binding the census cannot supply."""

    profile, stats, tensor_rows, hidden, experts = _packed_qwen_graph_inputs()
    mlp = profile.source_tensor_name("model.layers.0.mlp")

    # Baseline: an agreeing live source plans, and it really was consulted.
    agreeing = _CensusSource(tensor_rows)
    graph = checkpoint._layer_source_graph(
        **_graph_kwargs(profile, stats, tensor_rows, hidden, experts, agreeing)
    )
    planned = {
        str(graph["input_norm"]),
        str(graph["post_norm"]),
        str(graph["router"]),
        str(graph["packed_gate_up"]),
        str(graph["packed_down"]),
        *(str(name) for name in graph["input_consumers"]),
    }
    assert planned <= set(agreeing.reads)

    # A census row the live shard header contradicts must refuse.  The census
    # alone cannot see this: on the external-manifest path nothing re-opens a
    # shard, so only `source` can price the bytes the plan will actually read.
    drifted = _CensusSource(
        tensor_rows,
        overrides={f"{mlp}.experts.down_proj": ((experts, hidden, 999), "BF16")},
    )
    with pytest.raises(RuntimeError, match="live source header differs"):
        checkpoint._layer_source_graph(
            **_graph_kwargs(profile, stats, tensor_rows, hidden, experts, drifted)
        )

    # A planned operand whose shard is not local on this worker must refuse at
    # graph time, not later inside `_plan_layer` after staging exists.
    absent = _CensusSource(tensor_rows, missing={f"{mlp}.experts.gate_up_proj"})
    with pytest.raises(RuntimeError, match="lacks required local shards"):
        checkpoint._layer_source_graph(
            **_graph_kwargs(profile, stats, tensor_rows, hidden, experts, absent)
        )

    # And the parameter may not be optional: a None source is a refusal, never
    # a census-only bypass.
    with pytest.raises(RuntimeError, match="requires the live source checkpoint"):
        checkpoint._layer_source_graph(
            **_graph_kwargs(profile, stats, tensor_rows, hidden, experts, None)
        )


def test_moe_graph_binds_per_expert_operands_and_not_rejected_probes() -> None:
    """The bound set is the selected layout, not every name `source_key` saw."""

    profile, stats, tensor_rows, hidden, experts = _packed_qwen_graph_inputs()
    rows = deepcopy(tensor_rows)
    mlp_recipe = "model.layers.0.mlp"
    mlp = profile.source_tensor_name(mlp_recipe)
    rows.pop(f"{mlp}.experts.gate_up_proj")
    rows.pop(f"{mlp}.experts.down_proj")
    for expert in range(experts):
        for leaf, shape in (
            ("gate_proj", (32, hidden)),
            ("up_proj", (32, hidden)),
            ("down_proj", (hidden, 32)),
        ):
            recipe = f"{mlp_recipe}.experts.{expert}.{leaf}.weight"
            rows[profile.source_tensor_name(recipe)] = {
                "shape": list(shape),
                "dtype": "BF16",
            }
    source = _CensusSource(rows)
    graph = checkpoint._layer_source_graph(
        **_graph_kwargs(profile, stats, rows, hidden, experts, source)
    )
    assert graph["routed_layout"] == "per_expert_2d"
    read = set(source.reads)
    # Every selected per-expert operand was re-read from the live source.
    for role in graph["per_expert_roles"]:
        for key in ("gate", "up", "down"):
            assert str(role[key]) in read
    # The packed names `source_key` resolved while probing the layout were
    # rejected by the layout choice and must not be demanded of the source.
    assert f"{mlp}.experts.gate_up_proj" not in read
    assert f"{mlp}.experts.down_proj" not in read

    drifted = _CensusSource(
        rows,
        overrides={
            str(graph["per_expert_roles"][experts - 1]["down"]): ((hidden, 31), "BF16")
        },
    )
    with pytest.raises(RuntimeError, match="live source header differs"):
        checkpoint._layer_source_graph(
            **_graph_kwargs(profile, stats, rows, hidden, experts, drifted)
        )


def test_moe_schema_and_algorithm_are_categorically_separate_from_dense() -> None:
    import prismaquant.prismasnap as dense
    import prismaquant.prismasnap_checkpoint as dense_checkpoint

    assert checkpoint.MOE_PLAN_SCHEMA != dense_checkpoint.PLAN_SCHEMA
    assert checkpoint.MOE_PLAN_SET_SCHEMA != dense_checkpoint.PLAN_SET_SCHEMA
    assert checkpoint.MOE_PROVENANCE_SCHEMA != dense_checkpoint.PROVENANCE_SCHEMA
    assert moe.PRISMASNAP_MOE_ALGORITHM != dense.PRISMASNAP_ALGORITHM
    search = moe.moe_search_contract(_config())
    assert search["promotion"] == moe.PRISMASNAP_MOE_PROMOTION
    assert search["expert_coverage_policy"] == "all_experts_routed"
    assert search["routed_gate_up_global_scope"] == "per_expert_joint_gate_up"


def test_all_experts_routed_probe_policy_refuses_a_cold_expert(tmp_path) -> None:
    probe = tmp_path / "probe.pkl"
    with probe.open("wb") as handle:
        pickle.dump(
            {
                "stats": {
                    "model.layers.0.mlp.experts.gate_up_proj": {
                        "in_features": 16,
                        "out_features": 32,
                        "num_experts": 2,
                        "expert_act_sq_sum": np.ones((2, 16), dtype=np.float32),
                        "expert_tokens": np.asarray([8.0, 0.0]),
                        "_packed_param": "gate_up_proj",
                    }
                },
                "meta": {},
            },
            handle,
        )
    with pytest.raises(RuntimeError, match="complete per-expert routed importance"):
        checkpoint._load_moe_probe(probe)
