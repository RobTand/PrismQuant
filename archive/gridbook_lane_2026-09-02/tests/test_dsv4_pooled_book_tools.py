"""Producer half of campaign rule R1: the stack-keyed routed book burn.

The consumer half (`build_cb_learned_bundle --routed-book-keying stack`,
`load_banked_cbl_book`, the export's split-book gate) landed first and expects
a `gate_up_proj` shard whose identity is the fused per-expert stack weighted
by the packed target's own imatrix entry.  These tests pin the producer to
that contract without a GPU or a checkpoint: the fused row order, the packed
entry's derivation (the export's own per-expert mean) and the refusals that
keep the pooled arm's imatrix spelling unique.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.build_cb_learned_bundle import _stack_col_weights
from prismaquant.cb_warm_state import tensor_value_identity
from tools import dsv4_onlaw_book_burn as ob
from tools import dsv4_onlaw_book_select as sel
from tools import dsv4_packed_col_weights as pcw

_E, _ROWS, _IN = 3, 4, 8
_LAYER = 7


class _Profile:
    def __init__(self, fused=("gate_proj", "up_proj")):
        self._fused = tuple(fused)

    def packed_expert_parent_for_projection(self, projection):
        return "down_proj" if projection == "down_proj" else "gate_up_proj"

    def packed_expert_projection_names(self, packed):
        return self._fused if packed == "gate_up_proj" else ("down_proj",)


def _per_expert_col_weights() -> dict[str, torch.Tensor]:
    col = {"model.layers.7.self_attn.wq_b": torch.ones(5)}
    for expert in range(_E):
        for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            col[f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"] = (
                torch.arange(_IN, dtype=torch.float32) * (index + 1)
                + expert * 100.0
            )
    return col


def test_augmentation_is_the_exports_per_expert_mean():
    col = _per_expert_col_weights()
    augmented, added = pcw.augment_packed_expert_col_weights(col, _Profile())
    gate_up = f"model.layers.{_LAYER}.mlp.experts.gate_up_proj"
    down = f"model.layers.{_LAYER}.mlp.experts.down_proj"
    assert added == [down, gate_up]
    assert tuple(augmented[gate_up].shape) == (_E, 1, _IN)
    assert tuple(augmented[down].shape) == (_E, 1, _IN)
    for expert in range(_E):
        gate = col[f"model.layers.{_LAYER}.mlp.experts.{expert}.gate_proj"]
        up = col[f"model.layers.{_LAYER}.mlp.experts.{expert}.up_proj"]
        assert torch.equal(augmented[gate_up][expert, 0], (gate + up) / 2)
        assert torch.equal(
            augmented[down][expert, 0],
            col[f"model.layers.{_LAYER}.mlp.experts.{expert}.down_proj"],
        )
    # The per-expert entries (the role arm's identity) are untouched.
    for name, value in col.items():
        assert torch.equal(augmented[name], value)
    # The builder's stack reshape reads the entry back as the burn hashes it.
    stacked = _stack_col_weights(
        gate_up, experts=_E, in_features=_IN, col_weights=augmented
    )
    assert tensor_value_identity(stacked) == tensor_value_identity(
        augmented[gate_up]
    )


def test_augmentation_refuses_a_second_spelling_but_accepts_an_equal_one():
    col = _per_expert_col_weights()
    augmented, _added = pcw.augment_packed_expert_col_weights(col, _Profile())
    gate_up = f"model.layers.{_LAYER}.mlp.experts.gate_up_proj"
    again, added = pcw.augment_packed_expert_col_weights(augmented, _Profile())
    assert added == []
    assert torch.equal(again[gate_up], augmented[gate_up])
    broadcast = dict(col)
    broadcast[gate_up] = torch.ones(1, 1, _IN)
    with pytest.raises(ValueError, match="two spellings"):
        pcw.augment_packed_expert_col_weights(broadcast, _Profile())


def test_populations_follow_the_keying_and_the_abi_order():
    assert ob.populations("stack") == ("gate_up_proj", "down_proj")
    assert ob.populations("role") == ("gate_proj", "up_proj", "down_proj")
    assert ob.population_members("gate_up_proj", "stack", _Profile()) == (
        "gate_proj", "up_proj",
    )
    assert ob.population_members("down_proj", "stack", _Profile()) == (
        "down_proj",
    )
    assert ob.population_members("up_proj", "role", _Profile()) == ("up_proj",)
    with pytest.raises(ValueError, match="ABI expects"):
        ob.population_members(
            "gate_up_proj", "stack", _Profile(fused=("up_proj", "gate_proj"))
        )
    with pytest.raises(ValueError, match="not a routed stack key"):
        ob.population_members("gate_proj", "stack", _Profile())
    assert sel.selection_filename("stack") == "CB_ROUTED_MOE_BOOK_SELECTION.stack.json"
    assert sel.selection_filename("role") == "CB_ROUTED_MOE_BOOK_SELECTION.json"


def _fake_load_projection(col):
    def load(layer, projection, *, device, identity, all_col_weights,
             model_to_shard, model_to_ckpt, scale_map):
        assert layer == _LAYER
        qnames = [
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            for expert in range(_E)
        ]
        offset = {"gate_proj": 0.0, "up_proj": 1000.0, "down_proj": 0.0}[projection]
        weight = torch.stack([
            torch.arange(_ROWS * _IN, dtype=torch.float32).reshape(_ROWS, _IN)
            + offset + expert * 10_000.0
            for expert in range(_E)
        ])
        return {
            "qnames": qnames,
            "weight": weight,
            "col_weights": torch.stack(
                [col[q].reshape(1, -1) for q in qnames]
            ),
            "activation_rows": object(),
            "observed_activation_files": 1,
            "cold_experts": [expert for expert in range(_E) if projection == "up_proj" and expert == 2],
        }
    return load


def test_stack_population_is_the_builders_fused_weight(monkeypatch):
    col = _per_expert_col_weights()
    augmented, _added = pcw.augment_packed_expert_col_weights(col, _Profile())
    monkeypatch.setattr(ob, "load_projection", _fake_load_projection(col))
    data = ob.load_population(
        _LAYER, "gate_up_proj", keying="stack", profile=_Profile(),
        device=torch.device("cpu"), identity=None, all_col_weights=augmented,
        model_to_shard={}, model_to_ckpt={}, scale_map={},
    )
    weight = data["weight"]
    assert tuple(weight.shape) == (_E, 2 * _ROWS, _IN)
    # Every expert's gate rows, then its up rows: exactly provide_weight.
    for expert in range(_E):
        assert float(weight[expert, 0, 0]) == expert * 10_000.0
        assert float(weight[expert, _ROWS, 0]) == expert * 10_000.0 + 1000.0
    gate_up = f"model.layers.{_LAYER}.mlp.experts.gate_up_proj"
    assert tensor_value_identity(data["col_weights"]) == tensor_value_identity(
        _stack_col_weights(
            gate_up, experts=_E, in_features=_IN, col_weights=augmented
        )
    )
    assert data["activation_rows"] is None
    assert data["cold_experts"] == [2]
    assert data["observed_activation_files"] == 2
    assert len(data["qnames"]) == 2 * _E

    # A one-projection population is load_projection unchanged: the same
    # tensors under the same name, so a role-keyed down_proj bank already
    # satisfies the stack-keyed request.
    down = ob.load_population(
        _LAYER, "down_proj", keying="stack", profile=_Profile(),
        device=torch.device("cpu"), identity=None, all_col_weights=augmented,
        model_to_shard={}, model_to_ckpt={}, scale_map={},
    )
    role = ob.load_population(
        _LAYER, "down_proj", keying="role", profile=None,
        device=torch.device("cpu"), identity=None, all_col_weights=col,
        model_to_shard={}, model_to_ckpt={}, scale_map={},
    )
    assert tuple(down["weight"].shape) == (_E, _ROWS, _IN)
    assert tensor_value_identity(down["weight"]) == tensor_value_identity(role["weight"])
    assert tensor_value_identity(down["col_weights"]) == tensor_value_identity(role["col_weights"])
    assert down["activation_rows"] is not None


def test_stack_population_refuses_a_missing_or_foreign_packed_entry(monkeypatch):
    col = _per_expert_col_weights()
    monkeypatch.setattr(ob, "load_projection", _fake_load_projection(col))
    common = dict(
        keying="stack", profile=_Profile(), device=torch.device("cpu"),
        identity=None, model_to_shard={}, model_to_ckpt={}, scale_map={},
    )
    with pytest.raises(SystemExit, match="no packed col_weights entry"):
        ob.load_population(_LAYER, "gate_up_proj", all_col_weights=col, **common)
    foreign = dict(col)
    foreign[f"model.layers.{_LAYER}.mlp.experts.gate_up_proj"] = torch.ones(1, 1, _IN)
    with pytest.raises(SystemExit, match="not the export's per-expert mean"):
        ob.load_population(
            _LAYER, "gate_up_proj", all_col_weights=foreign, **common
        )
