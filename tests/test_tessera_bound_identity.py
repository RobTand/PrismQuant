"""Actual producer identities remain exact across a bounded immutable unit."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def fixture(*, projected=False):
    from prismaquant import tessera_campaign as tc, tessera_hessian as th
    from prismaquant.tessera_formats import parse_tessera_format_name
    name = "model.layers.0.proj"
    weight = torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256).to(torch.bfloat16) / 1024
    hessian = torch.eye(256)
    source = th.activation_source({name: hessian}, th.calibration_identity(
        "fixture", [torch.arange(16).reshape(1, 16)], fit_tokens=16))
    formats = [f"TESSERA_{family}_K1_R{rung}" for family in ("BF16", "E4M3")
               for rung in (896, 1024, 1152)] + ["TESSERA_E2M1_K2_R896"]
    anchors = []
    for fmt in formats:
        family, rung = parse_tessera_format_name(fmt)
        anchors.append(tc.CampaignAnchor(qname=name, format_name=fmt, family=family.name,
            body_rate_q256=rung, dloss=0.1, dloss_stderr=0.0, memory_bytes=8192,
            bits_per_param=4.0, activation_contract="fixture", activation_quantized=True,
            wire_bytes=8192, seconds=0.1, hessian_applied=True,
            input_global_scale=0.125 if "E2M1" in fmt else None))
    projection = None if not projected else dict(tensor=name + ".weight", source_tensor="packed.weight",
        source_layout="packed", source_slice={"expert": 0}, expert=0,
        projection="gate_proj", group="w13", rows=32, cols=256)
    kwargs = dict(weights={name: weight}, menus={name: [SimpleNamespace(format_name=f) for f in formats]},
        calibration_source=source, static_scales={name: 0.125},
        projected_units={} if projection is None else {name: projection})
    return tc, name, weight, hessian, source, anchors, projection, kwargs


@pytest.mark.parametrize("projected", [False, True])
def test_bound_unit_reuses_actual_source_h_hashes_with_identical_producer_identity(monkeypatch, projected):
    from tessera import cached_unit
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture(projected=projected)
    expected = [tc._checkpoint_anchor_identity(anchor, **kwargs) for anchor in anchors]
    actual_hash = cached_unit.tensor_identity
    calls = []
    def observed(value):
        calls.append(id(value))
        return actual_hash(value)
    monkeypatch.setattr(cached_unit, "tensor_identity", observed)
    with tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"]) as bound:
        actual = [tc._checkpoint_anchor_identity(anchor, **kwargs, bound_unit=bound) for anchor in anchors]
        assert actual == expected
        assert calls.count(id(weight)) == 1
        assert calls.count(id(hessian)) == 1
        actual[0]["source"]["sha256"] = "0" * 64
        assert tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound) == expected[0]
    with pytest.raises(ValueError, match="closed"):
        tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound)


@pytest.mark.parametrize("change", ["source_values", "source_storage", "source_view", "h_values", "h_replaced", "settings", "provenance", "projection"])
def test_bound_unit_refuses_lifetime_mutation(change):
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture(projected=True)
    bound = tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
        calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"])
    if change == "source_values":
        weight[0, 0] += 1
    elif change == "source_storage":
        weight.data = weight.clone()
    elif change == "source_view":
        kwargs["weights"][name] = weight.t().contiguous().t()
    elif change == "h_values":
        hessian[0, 0] += 1
    elif change == "h_replaced":
        source.hessians[name] = hessian.clone()
    elif change == "settings":
        object.__setattr__(source, "ldlq_sigma", source.ldlq_sigma + 1)
    elif change == "provenance":
        source.provenance["fit_tokens"] += 1
    else:
        projection["source_slice"]["expert"] += 1
    from tessera.errors import GrammarError
    with pytest.raises((ValueError, RuntimeError, GrammarError), match="changed|moved|differs|bound"):
        tc._checkpoint_anchor_identity(anchors[0], **kwargs, bound_unit=bound)


def test_bound_unit_keeps_per_anchor_static_scale_and_hessian_gates():
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture()
    with tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"]) as bound:
        with pytest.raises(RuntimeError, match="Hessian applicability"):
            tc._checkpoint_anchor_identity(replace(anchors[0], hessian_applied=False), **kwargs, bound_unit=bound)
        with pytest.raises(tc.ActivationScaleContractError):
            tc._checkpoint_anchor_identity(replace(anchors[-1], input_global_scale=0.25), **kwargs, bound_unit=bound)


def test_bound_unit_refuses_nonfinite_source():
    tc, name, weight, hessian, source, anchors, projection, kwargs = fixture()
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        tc.bind_checkpoint_unit_identity(anchors, source_weight=weight,
            calibration_source=source, projected_unit=projection, static_scales=kwargs["static_scales"])
