"""CPU mathematical oracles for signed joint weight/activation projections.

These fixtures use arbitrary resident float32 weight deltas and synthetic
activation quantizers. They establish the local projection identity, not a
served format's numerical accuracy, model quality, or performance.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from prismaquant.format_registry import FormatSpec
from prismaquant.joint_aura import SignedJointProjectionLease


def test_static_served_qdq_owner_is_shared_and_never_preclips(monkeypatch):
    from dataclasses import replace
    from prismaquant.format_registry import get_format
    from prismaquant.joint_aura import activation_identity
    from prismaquant.nvfp4_activation_contract import NVFP4_SERVED_ACTIVATION_CONTRACT
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "1")
    monkeypatch.setenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", "0")
    contract = replace(NVFP4_SERVED_ACTIVATION_CONTRACT, measured_as_served=True)
    def forbidden_dynamic(x):
        raise AssertionError("served static path must use its shared owner")
    spec_a = replace(get_format("NVFP4"), name="static_a", static_activation_contract=contract,
                     activation_quantize_dequantize=forbidden_dynamic)
    spec_b = replace(spec_a, name="static_b", activation_quantize_dequantize=lambda x: x * 0)
    torch.manual_seed(510)
    weight = torch.randn(3, 16)
    x = torch.randn(2, 16) * 2
    gradient = torch.randn(2, 3)
    delta = torch.randn_like(weight) * 0.1
    module = _linear(weight)
    maximum = {"unit": 1.0}
    receipt = activation_identity(spec_a, maximum, "unit")
    assert receipt["clip_enabled"] is False
    assert receipt["input_global_scale"] == contract.input_global_scale_from_max_abs(1.0)
    qx = contract.quantize_dequantize(x, receipt["input_global_scale"])
    with SignedJointProjectionLease({"unit": module}, {"unit": {"static_a": spec_a, "static_b": spec_b}},
                                    {("unit", "static_a"): delta, ("unit", "static_b"): delta * 2},
                                    activation_max_abs=maximum) as lease:
        lease.begin_probe()
        module(x).backward(gradient)
        result = lease.finish_probe()
        assert lease.telemetry["qdq_calls"] == 1
        assert lease.telemetry["operator_gemms"] == 1
    _assert_projection(result[("unit", "static_a")], _oracle(weight, delta, x, qx, gradient))
    _assert_projection(result[("unit", "static_b")], _oracle(weight, delta * 2, x, qx, gradient))


def _spec(
    name: str,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
    *,
    act_bits: int | None = 8,
) -> FormatSpec:
    return FormatSpec(
        name=name,
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="fp32",
        weight_element_dtype="int8",
        act_bits=act_bits,
        activation_quantize_dequantize=quantizer,
    )


def _linear(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def _oracle(
    weight: torch.Tensor,
    delta_weight: torch.Tensor,
    x: torch.Tensor,
    quantized_x: torch.Tensor,
    output_gradient: torch.Tensor,
) -> dict[str, float]:
    """Materialize both outputs independently of the production contraction."""
    w, dw, baseline, quantized, gradient = (
        value.detach().double()
        for value in (weight, delta_weight, x, quantized_x, output_gradient)
    )
    dx = quantized - baseline
    reference_output = baseline @ w.T
    candidate_output = quantized @ (w + dw).T
    return {
        "weight": float(((baseline @ dw.T) * gradient).sum()),
        "activation": float(((dx @ w.T) * gradient).sum()),
        "mixed": float(((dx @ dw.T) * gradient).sum()),
        "total": float(((candidate_output - reference_output) * gradient).sum()),
    }


def _assert_projection(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert set(actual) == {"weight", "activation", "mixed", "total"}
    for component, value in expected.items():
        assert isinstance(actual[component], float), component
        assert actual[component] == pytest.approx(value, rel=3e-5, abs=3e-6), component
    assert actual["total"] == pytest.approx(
        actual["weight"] + actual["activation"] + actual["mixed"],
        rel=3e-5,
        abs=3e-6,
    )


def _lease(layer: nn.Linear, spec: FormatSpec, dw: torch.Tensor):
    return SignedJointProjectionLease(
        {"layer": layer},
        {"layer": {spec.name: spec}},
        {("layer", spec.name): dw},
        activation_max_abs={},
    )


@pytest.mark.parametrize("input_shape", [(3,), (5, 3), (2, 4, 3)])
def test_full_output_residual_matches_fp64_oracle_without_changing_baseline(input_shape):
    generator = torch.Generator().manual_seed(237)
    weight = torch.randn(4, 3, generator=generator)
    dw = torch.randn(4, 3, generator=generator) * 0.125
    bias = torch.randn(4, generator=generator)
    x = torch.randn(input_shape, generator=generator, requires_grad=True)
    gradient = torch.randn((*input_shape[:-1], 4), generator=generator)
    original_x = x.detach().clone()
    layer = _linear(weight, bias)

    def quantizer(value):
        return torch.round(value * 2) / 2

    quantized_x = quantizer(original_x)
    expected = _oracle(weight, dw, original_x, quantized_x, gradient)
    with _lease(layer, _spec("synthetic", quantizer), dw) as lease:
        lease.begin_probe()
        output = layer(x)
        torch.testing.assert_close(output, torch.nn.functional.linear(x, weight, bias))
        (output * gradient).sum().backward()
        projections = lease.finish_probe()

    assert set(projections) == {("layer", "synthetic")}
    _assert_projection(projections[("layer", "synthetic")], expected)
    torch.testing.assert_close(x.detach(), original_x, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.detach(), weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias.detach(), bias, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, gradient @ weight)
    torch.testing.assert_close(layer.weight.grad, gradient.reshape(-1, 4).T @ original_x.reshape(-1, 3))


def test_weight_and_activation_projections_cancel_before_squaring():
    weight = torch.tensor([[0.0, -1.0]])
    dw = torch.tensor([[1.0, 0.0]])
    x = torch.tensor([[1.0, 0.0]])
    gradient = torch.ones(1, 1)
    quantizer = lambda value: value + torch.tensor([0.0, 1.0])
    layer = _linear(weight)
    with _lease(layer, _spec("cancel", quantizer), dw) as lease:
        lease.begin_probe()
        layer(x).sum().backward()
        projection = lease.finish_probe()[("layer", "cancel")]

    _assert_projection(projection, _oracle(weight, dw, x, quantizer(x), gradient))
    assert projection == {"weight": 1.0, "activation": -1.0, "mixed": 0.0, "total": 0.0}
    assert projection["weight"] ** 2 + projection["activation"] ** 2 == 2.0


def test_mixed_term_is_the_entire_joint_residual_when_both_marginals_vanish():
    weight = torch.zeros(1, 1)
    dw = torch.tensor([[2.0]])
    x = torch.zeros(1, 1)
    gradient = torch.tensor([[-0.5]])
    quantizer = lambda value: value + 3.0
    layer = _linear(weight)
    with _lease(layer, _spec("mixed", quantizer), dw) as lease:
        lease.begin_probe()
        (layer(x) * gradient).sum().backward()
        projection = lease.finish_probe()[("layer", "mixed")]

    _assert_projection(projection, _oracle(weight, dw, x, quantizer(x), gradient))
    assert projection == {"weight": 0.0, "activation": 0.0, "mixed": -3.0, "total": -3.0}


def test_repeated_invocations_sum_signed_projections_before_squaring():
    weight = torch.tensor([[2.0, -3.0], [1.0, 4.0]])
    dw = torch.tensor([[0.5, 1.0], [-0.25, 0.5]])
    x = torch.tensor([[1.0, 2.0], [-2.0, 3.0]])
    gradient = torch.tensor([[2.0, -1.0], [3.0, 0.5]])
    quantizer = lambda value: value + 0.25
    individual = _oracle(weight, dw, x, quantizer(x), gradient)
    assert all(value != 0.0 for value in individual.values())
    layer = _linear(weight)
    with _lease(layer, _spec("reused", quantizer), dw) as lease:
        lease.begin_probe()
        first, second = layer(x), layer(x.clone())
        ((first - second) * gradient).sum().backward()
        projection = lease.finish_probe()[("layer", "reused")]

    _assert_projection(projection, dict.fromkeys(individual, 0.0))


def test_repeated_invocations_keep_each_input_and_output_gradient_paired():
    weight = torch.tensor([[1.25, -0.5], [0.75, 2.0]])
    dw = torch.tensor([[0.25, -0.5], [0.125, 0.75]])
    inputs = [torch.tensor([[0.1, 1.1]]), torch.tensor([[2.6, -0.2], [0.7, -1.1]])]
    gradients = [torch.tensor([[2.0, -1.0]]), torch.tensor([[-1.0, 0.5], [1.5, 3.0]])]
    quantizer = lambda value: torch.round(value)
    expected = dict.fromkeys(("weight", "activation", "mixed", "total"), 0.0)
    for x, gradient in zip(inputs, gradients):
        for component, value in _oracle(weight, dw, x, quantizer(x), gradient).items():
            expected[component] += value
    layer = _linear(weight)
    with _lease(layer, _spec("reused", quantizer), dw) as lease:
        lease.begin_probe()
        outputs = [layer(x) for x in inputs]
        sum((output * gradient).sum() for output, gradient in zip(outputs, gradients)).backward()
        projection = lease.finish_probe()[("layer", "reused")]

    _assert_projection(projection, expected)


@pytest.mark.parametrize("act_bits", [None, 16])
def test_identity_activation_declaration_bypasses_quantizer(act_bits):
    def forbidden_quantizer(_value):
        raise AssertionError("A16 must not invoke activation QDQ")

    weight = torch.tensor([[1.0, -2.0]])
    dw = torch.tensor([[0.5, -0.25]])
    x = torch.tensor([[2.0, 3.0]])
    gradient = torch.tensor([[-2.0]])
    layer = _linear(weight)
    with _lease(layer, _spec("a16", forbidden_quantizer, act_bits=act_bits), dw) as lease:
        lease.begin_probe()
        (layer(x) * gradient).sum().backward()
        projection = lease.finish_probe()[("layer", "a16")]

    _assert_projection(projection, _oracle(weight, dw, x, x, gradient))
    assert projection["weight"] == projection["total"] == -0.5
    assert projection["activation"] == projection["mixed"] == 0.0


def test_candidates_sharing_quantizer_reuse_one_qdq_per_invocation():
    calls = []

    def shared_quantizer(value):
        calls.append(value.detach().clone())
        return value + 0.25

    weight = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    layer = _linear(weight)
    specs = {name: _spec(name, shared_quantizer) for name in ("first", "second", "third")}
    deltas = {
        ("layer", name): torch.full_like(weight, (index + 1) * 0.125)
        for index, name in enumerate(specs)
    }
    x = torch.tensor([[1.0, -1.5]])
    gradient = torch.tensor([[2.0, -0.5]])
    with SignedJointProjectionLease(
        {"layer": layer}, {"layer": specs}, deltas, activation_max_abs={}
    ) as lease:
        lease.begin_probe()
        (layer(x) * gradient).sum().backward()
        projections = lease.finish_probe()

    assert len(calls) == 1
    torch.testing.assert_close(calls[0], x, rtol=0, atol=0)
    assert set(projections) == set(deltas)
    for key, dw in deltas.items():
        _assert_projection(projections[key], _oracle(weight, dw, x, x + 0.25, gradient))


def test_distinct_quantizer_closures_with_same_name_do_not_share_qdq():
    calls = []

    def make_quantizer(offset):
        def quantizer(value):
            calls.append(offset)
            return value + offset
        return quantizer

    quantizers = [make_quantizer(0.25), make_quantizer(-0.5)]
    assert quantizers[0].__qualname__ == quantizers[1].__qualname__
    weight = torch.tensor([[1.0, 2.0]])
    layer = _linear(weight)
    specs = {str(index): _spec(str(index), qdq) for index, qdq in enumerate(quantizers)}
    deltas = {("layer", name): torch.full_like(weight, 0.5) for name in specs}
    x = torch.tensor([[2.0, -1.0]])
    with SignedJointProjectionLease(
        {"layer": layer}, {"layer": specs}, deltas, activation_max_abs={}
    ) as lease:
        lease.begin_probe()
        layer(x).sum().backward()
        projections = lease.finish_probe()

    assert sorted(calls) == [-0.5, 0.25]
    for index, offset in enumerate((0.25, -0.5)):
        key = ("layer", str(index))
        _assert_projection(
            projections[key], _oracle(weight, deltas[key], x, x + offset, torch.ones(1, 1))
        )


def test_unexecuted_modules_and_outputs_unused_by_loss_return_zero():
    weight = torch.tensor([[2.0, -1.0]])
    modules = {name: _linear(weight) for name in ("used", "unused_output", "unexecuted")}
    quantizer = lambda value: value + 0.5
    spec = _spec("synthetic", quantizer)
    specs = {name: {spec.name: spec} for name in modules}
    deltas = {(name, spec.name): torch.ones_like(weight) for name in modules}
    x = torch.tensor([[2.0, 1.0]])
    with SignedJointProjectionLease(modules, specs, deltas, activation_max_abs={}) as lease:
        lease.begin_probe()
        unused_output = modules["unused_output"](x)
        modules["used"](x).sum().backward()
        projections = lease.finish_probe()

    assert unused_output.requires_grad
    assert set(projections) == set(deltas)
    assert projections[("used", spec.name)]["total"] != 0.0
    for name in ("unused_output", "unexecuted"):
        assert projections[(name, spec.name)] == dict.fromkeys(
            ("weight", "activation", "mixed", "total"), 0.0
        )


def test_begin_probe_clears_previous_projections_and_exit_removes_observation():
    calls = []

    def quantizer(value):
        calls.append(True)
        return value + 0.5

    weight = torch.tensor([[2.0]])
    dw = torch.tensor([[0.25]])
    layer = _linear(weight)
    x = torch.tensor([[1.0]])
    with _lease(layer, _spec("reset", quantizer), dw) as lease:
        for sign in (1.0, -1.0):
            lease.begin_probe()
            (layer(x) * sign).sum().backward()
            _assert_projection(
                lease.finish_probe()[("layer", "reset")],
                _oracle(weight, dw, x, x + 0.5, torch.full((1, 1), sign)),
            )
    assert len(calls) == 2
    layer(x).sum().backward()
    assert len(calls) == 2


@pytest.mark.parametrize("bad_shape", [(1, 3), (2, 2), (2,)])
def test_mismatched_weight_delta_shape_is_refused(bad_shape):
    layer = _linear(torch.ones(1, 2))
    spec = _spec("bad_shape", lambda value: value)
    with pytest.raises((ValueError, RuntimeError), match="(?i)shape"):
        with _lease(layer, spec, torch.zeros(bad_shape)) as lease:
            lease.begin_probe()
            layer(torch.ones(1, 2)).sum().backward()
            lease.finish_probe()


def test_nonresident_weight_delta_device_is_refused_without_cuda():
    layer = _linear(torch.ones(1, 2))
    spec = _spec("bad_device", lambda value: value)
    with pytest.raises((ValueError, RuntimeError), match="(?i)device|resident"):
        with _lease(layer, spec, torch.empty(1, 2, device="meta")) as lease:
            lease.begin_probe()
            layer(torch.ones(1, 2)).sum().backward()
            lease.finish_probe()


def test_activation_qdq_shape_change_is_refused():
    layer = _linear(torch.ones(1, 2))
    spec = _spec("bad_qdq_shape", lambda value: value[..., :1])
    with pytest.raises((ValueError, RuntimeError), match="(?i)shape"):
        with _lease(layer, spec, torch.ones_like(layer.weight)) as lease:
            lease.begin_probe()
            layer(torch.ones(1, 2)).sum().backward()
            lease.finish_probe()


def test_activation_qdq_device_change_is_refused_without_cuda():
    layer = _linear(torch.ones(1, 2))
    spec = _spec("bad_qdq_device", lambda value: torch.empty_like(value, device="meta"))
    with pytest.raises((ValueError, RuntimeError), match="(?i)device|residen"):
        with _lease(layer, spec, torch.ones_like(layer.weight)) as lease:
            lease.begin_probe()
            layer(torch.ones(1, 2)).sum().backward()
            lease.finish_probe()
