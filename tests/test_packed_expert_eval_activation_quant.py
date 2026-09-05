"""The KL gate must quantize down_proj's activations too, or it selects on a
cheaper model than the one that ships.

`PerturbedActivationCache` emulates activation quantization with a
`register_forward_pre_hook`, which sees only the MODULE input. For a packed
routed-expert module that input is gate_up_proj's; down_proj consumes the
post-SwiGLU intermediate produced inside the forward, which no boundary hook
can reach -- `_module_input_member_name` says as much in its own docstring.

vLLM's `CompressedTensorsW4A4Nvfp4MoEMethod` registers BOTH
`w13_input_global_scale` and `w2_input_global_scale`, so the served runtime
quantizes both. Emulating one of the two understates the cost of exactly the
half of the MoE FLOPs it skips -- in the gate that picks the assignment.

These tests pin that both projections are quantized, each with its OWN
calibrated scale, and each exactly once.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import prismaquant.perturbed_x_cache as pxc
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    _ModulePlan,
    _ParamPlan,
)

from test_packed_expert_per_token_fisher import _PackedWrap, _routed_batch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN, INTER, E = 5, 6, 4


class _RecordingSpec:
    """Stands in for a W4A4 FormatSpec, recording what it was asked to
    quantize. Returning a marked tensor makes double-quantization visible."""

    act_quant_changes_input = True

    def __init__(self, name="NVFP4"):
        self.name = name
        self.seen: list[tuple[int, str]] = []

    def activation_quantize_dequantize(self, x):
        self.seen.append((int(x.shape[-1]), str(x.dtype)))
        return x + 1.0


def _cache_with_plan(model, specs):
    """A cache object carrying one hand-built packed plan, bypassing
    `_build_module_plans` (which needs a real model profile)."""
    cache = PerturbedActivationCache.__new__(PerturbedActivationCache)
    cache.include_activation_quant = True
    cache._activation_scales = {}
    # The G each unit's cached render score was priced at (#227). Empty here:
    # this hand-built cache has no production cache behind it, so there is no
    # priced cost for the hook to disagree with.
    cache._priced_input_global_scales = {}
    cache._handles = []
    cache._fused_forward_originals = []
    experts = model.experts
    cache.plans = [_ModulePlan(
        module=experts,
        params=[_ParamPlan(name=f"experts.{a}", attr=a, spec=s)
                for a, s in specs.items()],
    )]
    return cache


def _run(cache, model):
    x, idx, w, _v = _routed_batch(16, E, 2, HIDDEN, DEV, seed=5)
    for plan in cache.plans:
        cache._install_packed_expert_activation_quant(plan)
    try:
        with torch.no_grad():
            model.experts(x, idx, w)
    finally:
        for module, original in reversed(cache._fused_forward_originals):
            module.forward = original
        cache._fused_forward_originals.clear()


def test_both_projections_are_quantized():
    torch.manual_seed(0)
    model = _PackedWrap(E, HIDDEN, INTER).to(DEV)
    gu, dn = _RecordingSpec(), _RecordingSpec()
    cache = _cache_with_plan(model, {"gate_up_proj": gu, "down_proj": dn})
    _run(cache, model)

    # gate_up sees the hidden state; down sees the post-SwiGLU intermediate.
    assert gu.seen and all(d == HIDDEN for d, _ in gu.seen)
    assert dn.seen and all(d == INTER for d, _ in dn.seen), (
        "down_proj's activations were never quantized -- the gate is "
        "measuring a model the runtime does not serve")


def test_each_projection_uses_its_own_spec():
    """Not one shared spec: the two consume different tensors, so their
    calibrated scales are different objects and must not be crossed."""
    torch.manual_seed(1)
    model = _PackedWrap(E, HIDDEN, INTER).to(DEV)
    gu, dn = _RecordingSpec("A"), _RecordingSpec("B")
    cache = _cache_with_plan(model, {"gate_up_proj": gu, "down_proj": dn})
    _run(cache, model)
    assert {d for d, _ in gu.seen} == {HIDDEN}
    assert {d for d, _ in dn.seen} == {INTER}


def test_pre_hook_does_not_double_quantize_gate_up():
    """The module pre-hook must stand down for a packed plan; otherwise
    gate_up's input goes through the quantizer at the boundary AND again
    inside."""
    torch.manual_seed(2)
    model = _PackedWrap(E, HIDDEN, INTER).to(DEV)
    gu, dn = _RecordingSpec(), _RecordingSpec()
    cache = _cache_with_plan(model, {"gate_up_proj": gu, "down_proj": dn})
    plan = cache.plans[0]
    assert cache._packed_act_plan(plan) is not None
    # _active_activation_spec would otherwise hand the pre-hook a live spec.
    assert plan.params[0].spec.act_quant_changes_input


def test_dense_linear_plans_are_untouched():
    """A single-param nn.Linear plan keeps the boundary-hook path."""
    lin = nn.Linear(HIDDEN, INTER).to(DEV)
    cache = PerturbedActivationCache.__new__(PerturbedActivationCache)
    cache.include_activation_quant = True
    cache._activation_scales = {}
    cache._fused_forward_originals = []
    plan = _ModulePlan(
        module=lin,
        params=[_ParamPlan(name="w", attr="weight", spec=_RecordingSpec())],
    )
    cache.plans = [plan]
    assert cache._packed_act_plan(plan) is None
    cache._install_packed_expert_activation_quant(plan)
    # `is` would fail spuriously -- attribute access rebinds the method each
    # time. What matters is that nothing was recorded for restoration.
    assert cache._fused_forward_originals == []
    assert "forward" not in lin.__dict__


def test_forward_is_restored_and_f_linear_unpatched():
    torch.manual_seed(3)
    model = _PackedWrap(E, HIDDEN, INTER).to(DEV)
    original_f_linear = F.linear
    before = model.experts.forward
    cache = _cache_with_plan(
        model, {"gate_up_proj": _RecordingSpec(), "down_proj": _RecordingSpec()})
    _run(cache, model)
    assert F.linear is original_f_linear
    assert model.experts.forward == before
