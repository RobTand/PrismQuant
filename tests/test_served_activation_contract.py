"""The served activation contract is a property of the spec (#205).

A Tessera W4A4 rung is served by vLLM's ``scaled_fp4_quant`` with a STATIC
per-unit ``input_global_scale`` and UE4M3 block scales.  The campaign already
prices its anchors that way (#196); the assignment-KL hooks and the production
cache scorer priced the same rung through NVFP4's dynamic FP32-scale RTN
because both asked ``canonical_format_name(spec.name) == "NVFP4"`` to decide
which activation quantizer a spec serves, and a Tessera name is not "NVFP4".

One rule, one home: ``FormatSpec.static_activation_contract`` names the
served contract and its per-unit G rule; every consumer reads it from the
spec.  Codex's proof (``prismaquant_activation_consumers.py``) is the shape of
every case below: G=1 on a 1e-3 block underflows the UE4M3 scale to byte zero,
so the served activation is exactly 0 where the dynamic quantiser keeps 1e-3.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_activation_contract as owner
from prismaquant.perturbed_x_cache import _activation_qdq
from prismaquant.production_weight_cache import (
    _local_forward_render_score,
    _render_score_record,
)

Q = "model.layers.0.self_attn.o_proj"
W4A4 = "TESSERA_E2M1_K2_R896"
ENV = "PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES"


def _g_of(max_abs: float) -> float:
    return owner.input_global_scale_from_max_abs(
        max_abs, policy=owner.resolve_input_global_scale_policy())


@pytest.fixture
def underflow_case():
    """max_abs chosen so G == 1.0 exactly; a 1e-3 block then serves as 0."""
    amax = _g_of(1.0)          # 6.0 under the legacy policy
    g = _g_of(amax)            # 1.0
    assert g == 1.0, g
    x = torch.full((1, 16), 1e-3, dtype=torch.bfloat16)
    served = fr.nvfp4_activation_qdq_served(x, g)
    assert float(served[0, 0]) == 0.0, "the case must underflow to be a case"
    return amax, g, x, served


# --------------------------------------------------------------- the property

def test_a_w4a4_tessera_spec_names_the_served_static_contract():
    spec = fr.get_format(W4A4)
    contract = spec.static_activation_contract
    assert contract is not None
    assert contract.execution == owner.NVFP4_ACTIVATION_EXECUTION
    assert contract.group_size == owner.FP4_GROUP_SIZE
    # The served contract IS the measurement contract for a Tessera rung: the
    # plugin has no dynamic path, so there is no "screen baseline" to keep.
    assert contract.measured_as_served is True
    # Its G rule is the owner's, at the resolved policy -- the same number
    # the campaign stamps and the export ships.
    assert contract.input_global_scale_from_max_abs(3.0) == _g_of(3.0)
    x = torch.randn(4, 64)
    torch.testing.assert_close(
        contract.quantize_dequantize(x, 2.0),
        owner.nvfp4_activation_qdq_served(x, 2.0), rtol=0, atol=0)


def test_stock_nvfp4_carries_the_same_contract_behind_its_screen_default():
    contract = fr.get_format("NVFP4").static_activation_contract
    assert contract is not None
    assert contract.execution == owner.NVFP4_ACTIVATION_EXECUTION
    # Stock NVFP4's measurement default stays the dynamic screen baseline;
    # the served emulation remains the env opt-in (runtime_flags.md).
    assert contract.measured_as_served is False


def test_formats_without_a_static_scale_carry_no_contract():
    from prismaquant.tessera_formats import recipe_from_wire_names
    from prismaquant.tessera_render import synthesize_tessera_spec

    for name in ("NVFP4A16", "FP8_E4M3", "FP8_DYNAMIC", "MXFP8_E4M3", "BF16"):
        assert fr.get_format(name).static_activation_contract is None, name
    # A Tessera rung served through the FP8 channel route is dynamic W8A8;
    # weight-only E4M3 over a block plane is A16.  Neither has a static G.
    w8 = synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024",
        recipe=recipe_from_wire_names(1, "channel", "window", 8),
        shape=(2048, 4096))
    assert w8.static_activation_contract is None
    kernel = synthesize_tessera_spec(
        "TESSERA_E4M3_K1_R1024", recipe=recipe_from_wire_names(2, "lut16"))
    assert kernel.static_activation_contract is None


# ------------------------------------------------------------------ the hooks

def test_tessera_hook_qdq_is_the_owned_served_oracle(monkeypatch, underflow_case):
    """Codex's proof, inverted: the Tessera hook serves what the plugin serves."""
    amax, g, x, served = underflow_case
    spec = fr.get_format(W4A4)
    for env in (None, "1"):
        if env is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, env)
        hook = _activation_qdq(x, spec, {Q: amax}, Q)
        assert torch.equal(hook, served), (
            f"env={env}: Tessera hook priced {float(hook[0, 0])} where the "
            f"served oracle gives {float(served[0, 0])}")


def test_stock_nvfp4_hook_keeps_its_screen_default_and_opt_in(
        monkeypatch, underflow_case):
    amax, g, x, served = underflow_case
    stock = fr.get_format("NVFP4")
    monkeypatch.delenv(ENV, raising=False)
    base = _activation_qdq(x, stock, {Q: amax}, Q)
    torch.testing.assert_close(
        base, stock.activation_quantize_dequantize(x.clamp(-amax, amax)))
    assert not torch.equal(base, served)
    monkeypatch.setenv(ENV, "1")
    assert torch.equal(_activation_qdq(x, stock, {Q: amax}, Q), served)


def test_tessera_hook_refuses_by_name_without_a_calibrated_scale(underflow_case):
    _, _, x, _ = underflow_case
    spec = fr.get_format(W4A4)
    with pytest.raises(owner.ActivationScaleContractError, match=Q):
        _activation_qdq(x, spec, {}, Q)
    with pytest.raises(owner.ActivationScaleContractError, match=Q):
        _activation_qdq(x, spec, {Q: 0.0}, Q)


# ----------------------------------------------------------- the cache scorer

def test_cache_score_for_a_tessera_row_prices_the_served_contract(underflow_case):
    amax, g, x, _ = underflow_case
    w = torch.ones((1, 16), dtype=torch.float32)   # W == rendered W: A-side only
    record = _render_score_record(
        qname=Q, fmt=W4A4, render_format=W4A4,
        reference_weight=w, rendered_weight=w,
        activations=x, activation_max_abs=amax)
    served_score = _local_forward_render_score(
        reference_weight=w, rendered_weight=w, activations=x,
        activation_quantize=lambda t: fr.nvfp4_activation_qdq_served(t, g),
        activation_max_abs=None)[0]
    assert served_score > 0.0
    assert record["score"] == pytest.approx(served_score, rel=0, abs=0)
    assert record["activation_quantized"] is True
    # The clamp lives inside the static scale, not in a pre-clip.
    assert record["activation_clipped"] is False
    # The record retains its static calibration maximum ...
    assert record["activation_max_abs"] == amax
    # ... and the G it was priced at.
    assert record["input_global_scale"] == g


def test_cache_score_for_a_tessera_row_refuses_without_its_scale(underflow_case):
    _, _, x, _ = underflow_case
    w = torch.ones((1, 16), dtype=torch.float32)
    with pytest.raises(owner.ActivationScaleContractError, match=Q):
        _render_score_record(
            qname=Q, fmt=W4A4, render_format=W4A4,
            reference_weight=w, rendered_weight=w,
            activations=x, activation_max_abs=None)


def test_stock_nvfp4_cache_score_is_unchanged(underflow_case):
    """The stock row keeps clip + dynamic RTN (screen default), and records
    the clip maximum as before (test_render_score_clips_nvfp4_...)."""
    amax, g, x, served = underflow_case
    w = torch.ones((1, 16), dtype=torch.float32)
    record = _render_score_record(
        qname=Q, fmt="NVFP4", render_format="NVFP4",
        reference_weight=w, rendered_weight=w,
        activations=x, activation_max_abs=amax)
    dynamic = _local_forward_render_score(
        reference_weight=w, rendered_weight=w, activations=x,
        activation_quantize=fr.get_format("NVFP4").activation_quantize_dequantize,
        activation_max_abs=amax)[0]
    assert record["score"] == dynamic
    assert record["activation_clipped"] is True
    assert record["activation_max_abs"] == amax
    # Priced under the dynamic quantiser: no static G to record.
    assert record["input_global_scale"] is None


def test_the_cache_fill_computes_max_abs_for_a_tessera_only_fill():
    """Without this the Tessera record can never retain a maximum: the fill
    only measured max|x| when a stock NVFP4 format was in the set."""
    from prismaquant.production_weight_cache import (
        _formats_need_static_activation_max,
    )

    assert _formats_need_static_activation_max({W4A4}) is True
    assert _formats_need_static_activation_max({"NVFP4"}) is True
    assert _formats_need_static_activation_max({"FP8_DYNAMIC", "BF16"}) is False
    assert _formats_need_static_activation_max(set()) is False


# ------------------------------------------------------- the KL preflight gate

class _Refusing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(64, 32, bias=False)

    def forward(self, input_ids):
        raise AssertionError("the forward must not run: the refusal is preflight")


def test_kl_measurement_refuses_a_served_contract_without_scale_identity(
        tmp_path, monkeypatch):
    from prismaquant.kl_measurement import measure_assignment_kl

    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    model = _Refusing()
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    refs = [torch.log_softmax(torch.randn(1, 3, 5), dim=-1)]
    with pytest.raises(owner.ActivationScaleContractError, match="proj"):
        measure_assignment_kl(
            model, {"proj": W4A4}, calib_ids, refs,
            work_root=tmp_path, kl_scope="full_sequence")


def test_kl_preflight_is_quiet_for_stock_nvfp4_and_a16(tmp_path, monkeypatch):
    """The gate is the served-measurement contract, not the W4A4 shape: stock
    NVFP4 without a production cache still measures under its screen default
    (the forward then runs -- and this model refuses to, which is the proof)."""
    from prismaquant.kl_measurement import measure_assignment_kl

    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    model = _Refusing()
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    refs = [torch.log_softmax(torch.randn(1, 3, 5), dim=-1)]
    for fmt in ("NVFP4", "NVFP4A16"):
        with pytest.raises(AssertionError, match="forward must not run"):
            measure_assignment_kl(
                model, {"proj": fmt}, calib_ids, refs,
                work_root=tmp_path, kl_scope="full_sequence")
