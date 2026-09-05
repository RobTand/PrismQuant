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
import torch.nn as nn

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
#: The compatibility input the two shipped policies are selected with.
POLICY_ENV = "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE"
LEGACY = owner.LEGACY_INPUT_GLOBAL_SCALE_POLICY
FULL = owner.FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY


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


# ------------------------------------- the fill gate must be total (#218)
#
# #213 replaced a string membership test with a registry lookup, and a lookup
# can raise where a membership test could only answer.  Two inputs reach this
# gate that the lookup cannot survive, and both kill a production cache fill
# at `fill_production_weight_cache` before one tensor is rendered.

def test_the_fill_gate_answers_for_the_mixed_case_registry_row():
    """``INT4_W4A16_g128`` is the ONE registry row whose name is mixed case,
    and it is reachable: ``validation_harness._entry_format_name`` maps the
    4-bit entry of a precision plan to it, so an assignment carries it into
    ``render_formats_by_qname`` and hence into ``render_base_fmt_set``.

    The fill hands this gate ``_render_base_format``'s output, which
    upper-cases.  The gate must still ANSWER -- that row is W4A16, it has no
    static activation contract, so no calibrated maximum is needed -- rather
    than raise ``KeyError`` naming the registry (#218).
    """
    from prismaquant.production_weight_cache import (
        _format_uses_static_activation_clip,
        _formats_need_static_activation_max,
        _render_base_format,
    )

    # What the fill actually builds at production_weight_cache.py:6790.
    render_base_fmt_set = {
        _render_base_format(f) for f in ("INT4_W4A16_g128", "BF16")
    }
    assert "INT4_W4A16_G128" in render_base_fmt_set

    assert _format_uses_static_activation_clip("INT4_W4A16_g128") is False
    assert _format_uses_static_activation_clip("INT4_W4A16_G128") is False
    assert _formats_need_static_activation_max(render_base_fmt_set) is False


def test_the_fill_gate_is_total_over_names_the_registry_does_not_own():
    """A predicate over a requested format MENU has to be total.  A menu may
    carry a name this registry does not own, and "not a format we know" is an
    answer -- False, there is no static activation contract to calibrate for
    -- not a bare ``KeyError`` raised from inside an activation-scale gate.
    A refusal, if the project wants one, belongs where the menu is validated.
    """
    from prismaquant.production_weight_cache import (
        _format_uses_static_activation_clip,
        _formats_need_static_activation_max,
        _is_cb_format_name,
    )

    junk = "NOT_A_REGISTERED_FORMAT_pq218"
    assert _format_uses_static_activation_clip(junk) is False
    assert _formats_need_static_activation_max({junk, "BF16"}) is False
    # The sibling predicate over the same values already answered False here;
    # they now share one resolver, so they cannot drift apart again.
    assert _is_cb_format_name(junk) is False


# ------------------- the policy a cached cost was priced under (#227)
#
# A W4A4 score is an activation-aware cost, and the static scale it is priced
# at is `FP4_MAX[*FP8_MAX] / max_abs`: the SAME calibration maximum prices
# G=1 under `legacy_6_over_calibration_amax.v1` and G=448 under
# `full_e4m3_range_448x6_over_calibration_amax.v1`, and the served oracle at
# those two G underflows a 1e-3 block to zero at one and keeps it at the
# other.  Measured for RobTand/prismaquant#227 on the frozen `17ca6930` tree
# and re-derived by the assertions below: amax=6, BF16 [1e-3]*16,
# W = rendered-W = ones(1,16).
#
#   policy 0: G=1,   score=0.00025571882724761963, hook QDQ=0
#   policy 1: G=448, score=5.622899834634154e-07,  hook QDQ=0.00104522705078125
#   cache render identity under policy 0 == identity under policy 1: True
#
# That last line is the defect: the numbers are each right under their own
# policy (#205's positive control), and nothing refused when a cache priced
# under one was resumed and KL-validated under the other.

PRICED_UNDER_POLICY = {
    "0": (LEGACY, 1.0, 0.00025571882724761963, 0.0),
    "1": (FULL, 448.0, 5.622899834634154e-07, 0.00104522705078125),
}


@pytest.fixture
def policy_case():
    """amax=6 (so G is exactly 1.0 or 448.0) and the underflowing block."""
    x = torch.full((1, 16), 1e-3, dtype=torch.bfloat16)
    w = torch.ones((1, 16), dtype=torch.float32)
    return 6.0, x, w


def _score_record(amax, x, w, fmt=W4A4, **kwargs):
    return _render_score_record(
        qname=Q, fmt=fmt, render_format=fmt,
        reference_weight=w, rendered_weight=w,
        activations=x, activation_max_abs=amax, **kwargs)


def test_the_cache_render_identity_binds_the_resolved_scale_policy(monkeypatch):
    """The pre-fix line: two policies, one identity.

    The resolved levers are where every env-valued render input is turned into
    a stamped value, and the directory render identity carries them; the scale
    policy joins ``nvfp4_scale_rule`` there, so the mismatch is named rather
    than silent.
    """
    import prismaquant.production_weight_cache as pwc

    identities = {}
    resolved = {}
    for setting in ("0", "1"):
        monkeypatch.setenv(POLICY_ENV, setting)
        levers = pwc._resolve_production_render_levers(
            {"tessera_weights_only": True})
        resolved[setting] = levers
        identities[setting] = pwc.build_production_cache_render_identity(
            render_scope="format-menu",
            requested_formats=[W4A4],
            levers=levers,
            mechanism_plan=pwc._resolve_render_mechanism_plan(levers),
            calib_hash="a" * 64,
            eligible_qnames=[Q],
            render_formats_by_qname={Q: [W4A4]},
            max_act_rows=1,
        )
    # The pre-fix behaviour, stated first: these two were equal, so a cache
    # priced at G=1 admitted a resume that quantizes activations at G=448.
    assert identities["0"] != identities["1"], (
        "the production cache render identity does not bind the resolved "
        "activation-scale policy")
    difference = pwc.first_identity_difference(
        identities["0"], identities["1"])
    assert difference == (
        f"levers.{pwc.RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY}", LEGACY, FULL)
    for setting, levers in resolved.items():
        assert levers[pwc.RENDER_LEVER_INPUT_GLOBAL_SCALE_POLICY] == (
            PRICED_UNDER_POLICY[setting][0])


def test_a_render_score_records_the_policy_its_static_scale_came_from(
        monkeypatch, policy_case):
    amax, x, w = policy_case
    for setting, (policy, g, score, _hook) in PRICED_UNDER_POLICY.items():
        monkeypatch.setenv(POLICY_ENV, setting)
        record = _score_record(amax, x, w)
        assert record["input_global_scale"] == g
        assert record["input_global_scale_policy"] == policy
        assert record["score"] == pytest.approx(score, rel=1e-9)


def test_a_dynamically_scored_row_records_no_policy(monkeypatch, policy_case):
    """Stock NVFP4 keeps its screen default, so nothing about its cost depends
    on the static-scale policy -- and it must not be invalidated by one."""
    amax, x, w = policy_case
    monkeypatch.setenv(POLICY_ENV, "0")
    record = _score_record(amax, x, w, fmt="NVFP4")
    assert record["input_global_scale"] is None
    assert record["input_global_scale_policy"] is None


def test_one_resolved_policy_prices_a_whole_operation(monkeypatch, policy_case):
    """A caller that resolved ONE policy for a fill passes it, and the scorer
    prices with that one -- not with whatever the environment says now."""
    amax, x, w = policy_case
    monkeypatch.setenv(POLICY_ENV, "0")
    record = _score_record(amax, x, w, input_global_scale_policy=FULL)
    assert record["input_global_scale"] == 448.0
    assert record["input_global_scale_policy"] == FULL
    assert record["score"] == pytest.approx(
        PRICED_UNDER_POLICY["1"][2], rel=1e-9)


def test_resume_refuses_a_retained_cost_priced_under_another_policy(
        monkeypatch, policy_case):
    """The retained-score half: reusing the weights can be legitimate, reusing
    their activation-aware cost under another policy is not."""
    from prismaquant.production_weight_cache import (
        _check_resumed_render_score_policies as check,
        _render_score_record_key,
    )

    amax, x, w = policy_case
    monkeypatch.setenv(POLICY_ENV, "0")
    record = _score_record(amax, x, w)
    key = _render_score_record_key(Q, W4A4)

    # Unchanged policy: the cost is admitted, and it IS checked (1 row).
    assert check({key: record}, policy=LEGACY, where="resume") == 1

    with pytest.raises(owner.ActivationScalePolicyMismatchError) as excinfo:
        check({key: record}, policy=FULL, where="resume")
    message = str(excinfo.value)
    assert Q in message and "1.0" in message and "448.0" in message
    assert LEGACY in message and FULL in message

    # A record written before the policy stamp existed is still checked: its
    # recorded G against the G this policy derives from its recorded maximum.
    legacy_record = {
        k: v for k, v in record.items() if k != "input_global_scale_policy"
    }
    with pytest.raises(owner.ActivationScalePolicyMismatchError, match=Q):
        check({key: legacy_record}, policy=FULL, where="resume")
    assert check({key: legacy_record}, policy=LEGACY, where="resume") == 1

    # A dynamically scored row does not depend on the policy and is not
    # invalidated by it.
    stock = _score_record(amax, x, w, fmt="NVFP4")
    assert check(
        {_render_score_record_key(Q, "NVFP4"): stock},
        policy=FULL, where="resume") == 0


def test_the_kl_hook_refuses_an_activation_priced_at_another_scale(
        monkeypatch, policy_case):
    """The measurement half: the hook derives G from the calibrated maximum
    and the CURRENT policy, so it must compare it against the G the cost it is
    measured against was priced at."""
    amax, x, _w = policy_case
    spec = fr.get_format(W4A4)
    monkeypatch.setenv(POLICY_ENV, "0")
    served = fr.nvfp4_activation_qdq_served(x, 1.0)
    # Control: the cache priced this unit at the same G the hook applies.
    assert torch.equal(_activation_qdq(x, spec, {Q: amax}, Q, {Q: 1.0}), served)
    # And with no priced G at all (no production cache) nothing changes.
    assert torch.equal(_activation_qdq(x, spec, {Q: amax}, Q, {}), served)
    with pytest.raises(owner.ActivationScalePolicyMismatchError, match=Q):
        _activation_qdq(x, spec, {Q: amax}, Q, {Q: 448.0})


def test_the_cache_the_scorer_and_the_hook_use_one_effective_g(
        monkeypatch, policy_case):
    """Fused siblings share ONE unified maximum, hence ONE static G -- and the
    record, the cache's priced-scale provenance and the hook must all be that
    same G under the policy in force."""
    from prismaquant.production_weight_cache import (
        ProductionWeightCache,
        _render_score_record_key,
        production_cache_priced_input_global_scales,
    )

    amax, x, w = policy_case
    siblings = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    ]
    for setting, (_policy, g, _score, hook_value) in PRICED_UNDER_POLICY.items():
        monkeypatch.setenv(POLICY_ENV, setting)
        records = {
            _render_score_record_key(name, W4A4): _render_score_record(
                qname=name, fmt=W4A4, render_format=W4A4,
                reference_weight=w, rendered_weight=w,
                activations=x, activation_max_abs=amax)
            for name in siblings
        }
        assert {r["input_global_scale"] for r in records.values()} == {g}
        cache = ProductionWeightCache(
            weights={},
            levers={},
            activation_max_abs={name: amax for name in siblings},
            metadata={"render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": len(records),
                "records": records,
            }},
        )
        priced = production_cache_priced_input_global_scales(cache)
        assert priced == {name: g for name in siblings}
        spec = fr.get_format(W4A4)
        for name in siblings:
            hook = _activation_qdq(
                x, spec, cache.activation_max_abs, name, priced)
            assert float(hook[0, 0]) == hook_value


class _OneLinear(nn.Module):
    """The shape ``_build_module_plans`` accepts for a W4A4 rung, as the
    preflight tests above it use."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 32, bias=False)


def _cache_priced_at(g: float, amax: float):
    from prismaquant.production_weight_cache import (
        ProductionWeightCache,
        _render_score_record_key,
    )

    return ProductionWeightCache(
        weights={},
        levers={},
        activation_max_abs={"proj": amax},
        metadata={"render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": 1,
            "records": {
                _render_score_record_key("proj", W4A4): {
                    "qname": "proj",
                    "format": W4A4,
                    "activation_max_abs": amax,
                    "input_global_scale": g,
                    "input_global_scale_policy": (
                        LEGACY if g == 1.0 else FULL),
                },
            },
        }},
    )


def test_the_kl_preflight_names_every_unit_priced_at_another_scale(
        tmp_path, monkeypatch, policy_case):
    """As with the missing-maximum gate, the refusal is preflight and lists
    every affected unit -- a maximum EXISTING says nothing about the policy it
    was priced under, which is exactly what the earlier gate could not see."""
    from prismaquant.kl_measurement import measure_assignment_kl
    from prismaquant.perturbed_x_cache import PerturbedActivationCache

    amax, _x, _w = policy_case
    monkeypatch.setenv(POLICY_ENV, "0")
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    monkeypatch.setenv("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "0")

    matched = PerturbedActivationCache(
        _OneLinear(), {"proj": W4A4}, tmp_path / "matched",
        input_rows=0, cal_hash="c" * 32,
        production_weight_cache=_cache_priced_at(1.0, amax))
    assert matched.served_activation_scale_gaps() == []
    assert matched.served_activation_policy_conflicts() == []

    stale = PerturbedActivationCache(
        _OneLinear(), {"proj": W4A4}, tmp_path / "stale",
        input_rows=0, cal_hash="c" * 32,
        production_weight_cache=_cache_priced_at(448.0, amax))
    # The maximum is present under either policy: the OLD gate stays quiet.
    assert stale.served_activation_scale_gaps() == []
    assert stale.served_activation_policy_conflicts() == ["proj"]

    model = _OneLinear()
    calib_ids = torch.ones(1, 3, dtype=torch.long)
    refs = [torch.log_softmax(torch.randn(1, 3, 5), dim=-1)]
    with pytest.raises(
            owner.ActivationScalePolicyMismatchError, match="proj"):
        measure_assignment_kl(
            model, {"proj": W4A4}, calib_ids, refs,
            work_root=tmp_path, kl_scope="full_sequence",
            production_weight_cache=_cache_priced_at(448.0, amax))
