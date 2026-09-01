"""The per-expert checkpoint bridge must compute the packed A-side, not a cousin.

`build_weight_resolver` maps a card unit to ONE checkpoint key, which fits a
packed `[E, M, N]` routed-expert parameter. GLM-5.3-Flash stores every expert as
its own 2-D Linear weight and splits the fused pair in two, so its 84 packed
units -- 97% of the parameters -- resolved to nothing and silently priced their
A-side at 0.0 on a lane whose own attested contract says NVFP4 is "Real A4 on
BOTH the dense and the packed-expert route".

`packed_act_dloss_per_expert` streams those per-expert tensors instead of
stacking 19 GiB of gate_up. The risk in reordering a reduction is that it stops
being the same reduction, so the load-bearing test here is the agreement one:
streamed must equal `_activation_dloss_packed` on the stacked equivalent.
"""
import numpy as np
import pytest
import torch

from prismaquant.aqua_activation_cost import (
    packed_act_dloss_per_expert, per_expert_weight_keys,
)
from prismaquant.format_cost_protocol import _activation_dloss_packed
from prismaquant.sensitivity_card import SensitivityUnit, UnitTopology

E, M, N = 4, 6, 8          # experts, out_features (gate+up), in_features


def _unit(name="model.layers.0.mlp.experts.gate_up_proj"):
    rng = np.random.default_rng(0)
    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="gate_up"),
        out_features=M, in_features=N, n_params=E * M * N, n_tokens=512,
        h_trace_raw=1.0, h_w2_sum_raw=0.0, w_norm_sq=1.0, w_max_abs=1.0,
        expert_g_sq_sum=rng.random((E, M)).astype(np.float32) + 0.1,
        expert_act_sq_sum=rng.random((E, N)).astype(np.float32) + 0.1,
        expert_tokens=np.array([128, 64, 32, 0], dtype=np.int64),
    )


def test_streamed_equals_the_stacked_packed_reduction():
    unit = _unit()
    rng = np.random.default_rng(7)
    per_expert = [torch.tensor(rng.standard_normal((M, N)), dtype=torch.float32)
                  for _ in range(E)]
    stacked = torch.stack(per_expert).numpy()
    var = rng.random((E, N)) + 0.05

    keys = [[f"e{e}"] for e in range(E)]
    lookup = {f"e{e}": per_expert[e] for e in range(E)}
    streamed = packed_act_dloss_per_expert(
        unit, keys, "unused", var, handles=lookup.__getitem__)
    reference = _activation_dloss_packed(unit, stacked, var)
    assert streamed == pytest.approx(reference, rel=1e-9)


def test_fused_siblings_concatenate_along_the_output_axis():
    """gate then up -- the order `expert_g_sq_sum`'s rows are indexed in.

    Concatenating the other way silently pairs each expert's up-projection rows
    with the gradient statistics of its gate rows, which is a wrong number that
    still has the right shape.
    """
    unit = _unit()
    rng = np.random.default_rng(11)
    gate = [torch.tensor(rng.standard_normal((M // 2, N)), dtype=torch.float32)
            for _ in range(E)]
    up = [torch.tensor(rng.standard_normal((M // 2, N)), dtype=torch.float32)
          for _ in range(E)]
    var = rng.random((E, N)) + 0.05
    lookup = {}
    keys = []
    for e in range(E):
        lookup[f"g{e}"], lookup[f"u{e}"] = gate[e], up[e]
        keys.append([f"g{e}", f"u{e}"])
    streamed = packed_act_dloss_per_expert(
        unit, keys, "unused", var, handles=lookup.__getitem__)
    stacked = torch.stack([torch.cat([gate[e], up[e]], dim=0)
                           for e in range(E)]).numpy()
    assert streamed == pytest.approx(_activation_dloss_packed(unit, stacked, var),
                                     rel=1e-9)


def test_a_zero_token_expert_contributes_no_activation_cost():
    """Expert 3 saw no calibration tokens; its sigma row is zero.

    `expert_act_sigma` returns a zero row rather than a fitted distribution, and
    the streamed reduction must carry that through as 0.0 -- inventing an A-side
    for the least-evidenced expert is worse than reporting none.
    """
    unit = _unit()
    rng = np.random.default_rng(3)
    per_expert = [torch.tensor(rng.standard_normal((M, N)), dtype=torch.float32)
                  for _ in range(E)]
    var = rng.random((E, N)) + 0.05
    lookup = {f"e{e}": per_expert[e] for e in range(E)}
    keys = [[f"e{e}"] for e in range(E)]
    full = packed_act_dloss_per_expert(unit, keys, "u", var,
                                       handles=lookup.__getitem__)
    var_zeroed = var.copy()
    var_zeroed[3] = 0.0
    zeroed = packed_act_dloss_per_expert(unit, keys, "u", var_zeroed,
                                         handles=lookup.__getitem__)
    assert zeroed < full


def test_per_expert_keys_are_found_and_ordered():
    wm = {}
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wm[f"model.layers.0.mlp.experts.{e}.{proj}.weight"] = "s.safetensors"
    keys = per_expert_weight_keys("model.layers.0.mlp.experts.gate_up_proj",
                                  wm, n_experts=E)
    assert keys == [[f"model.layers.0.mlp.experts.{e}.gate_proj.weight",
                     f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
                    for e in range(E)]
    down = per_expert_weight_keys("model.layers.0.mlp.experts.down_proj",
                                  wm, n_experts=E)
    assert down == [[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]
                    for e in range(E)]


def test_a_packed_layout_declines_rather_than_guessing():
    """No per-expert keys -> None, so the caller falls back to the single-key
    resolver instead of taking a special case that would price nothing."""
    assert per_expert_weight_keys(
        "model.layers.0.mlp.experts.gate_up_proj",
        {"model.layers.0.mlp.experts.gate_up_proj": "s.safetensors"},
        n_experts=E) is None
    assert per_expert_weight_keys("model.layers.0.mlp.down_proj", {},
                                  n_experts=E) is None
