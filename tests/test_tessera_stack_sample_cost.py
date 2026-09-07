"""A sampled routed-expert stack prices as ONE unit, in the allocator's currency.

The campaign encodes one Linear at a time, but vLLM loads a packed ``[E, M, N]``
expert tensor under a single quantization scheme. PrismaQuant #290: an
allocation fed a probe expanded per expert was refused by the explicit Tessera
serving scope for every unit
(``campaign-01/README.md`` section 7a: ``model.layers.18.feed_forward.experts.0.w2:
conflicting router_path/expert_id topology``), because the expansion drops
``_packed_experts_module`` while keeping a numeric ``expert_id`` and a null
``router_path``. Keying the cost table at the packed parameter's own qname --
the row the probe already has -- removes the need for any bridge.

These tests pin three things:

* the ESTIMATOR: Horvitz-Thompson over the drawn experts is unbiased for the
  stack total, and a census through the stack path prices identically to the
  per-expert expansion (``tier2_per_expert_counterfactual``);
* the CURRENCY: the number written into ``output_mse`` is the one
  ``allocator_solver.predicted_dloss`` multiplies correctly, i.e.
  ``0.5 * h_stack * output_mse`` reproduces the summed per-expert dloss;
* the SCOPE: the stack row resolves ``routed_moe`` on
  ``tessera_serving_scope``'s EXISTING packed branch, and a per-expert row
  without the packed topology still refuses (principle 14 -- the scope may not
  invent a topology it was not given).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "tessera_stack_lfm_layer18.json"
STEM = "model.layers.18.feed_forward.experts"
MODULE = "model.layers.18.feed_forward.experts"
IMAGE = "example/runtime@sha256:" + "b" * 64


def _campaign():
    return pytest.importorskip("prismaquant.tessera_campaign")


def _fixture():
    return json.loads(FIXTURE.read_text())


def _anchor(campaign, qname, family, fmt, q256, dloss, **over):
    values = dict(
        qname=qname, family=family, format_name=fmt, body_rate_q256=q256,
        dloss=float(dloss), dloss_stderr=0.0, memory_bytes=1024,
        bits_per_param=4.0, activation_contract="bfloat16",
        activation_quantized=False, wire_bytes=1100, seconds=1.0,
        hessian_applied=True, input_global_scale=None)
    values.update(over)
    return campaign.CampaignAnchor(**values)


def _packed_probe_row(num_experts, per_expert, *, packed_param="gate_up_proj",
                      out_features=8, in_features=4):
    """The probe's own packed row -- the shape the allocator prices bytes from."""
    return {
        "h_trace": float(math.fsum(per_expert)),
        "h_trace_per_expert": [float(v) for v in per_expert],
        "num_experts": int(num_experts),
        "_packed_experts_module": MODULE,
        "_packed_param": packed_param,
        "out_features": out_features,
        "in_features": in_features,
        "n_params": num_experts * out_features * in_features,
        "router_path": None,
        "expert_id": None,
    }


def _profile():
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
    return Lfm2MoeProfile()


def _sample(campaign, probe_row, *, qname=f"{STEM}.gate_up_proj",
            sampled=None, pi=None, seed=7, design="pps_wor"):
    experts = range(probe_row["num_experts"]) if sampled is None else sampled
    experts = list(experts)
    probs = {e: 1.0 for e in experts} if pi is None else pi
    return campaign.stack_sample_from_probe(
        qname, probe_row, _profile(), sampled_experts=experts,
        inclusion_prob=probs, seed=seed, design=design)


def _payload(campaign, anchors, samples, *, menus=None):
    return campaign.campaign_cost_payload(
        anchors, menus or {}, loo={},
        provenance={"provenance": {"hessian": {"supplied": True}}},
        stack_samples=samples)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def test_horvitz_thompson_is_unbiased_for_the_stack_total():
    """Averaged over draws, the estimate equals the full 32-expert sum.

    Sampling is PPS on the probe's own Fisher weights with a certainty stratum,
    so the check is on the ESTIMATOR, not on one lucky draw: the mean over many
    independent draws must sit inside the standard error of that mean.
    """
    campaign = _campaign()
    rng = random.Random(20260906)
    n_experts, n_draw = 32, 8
    h = [rng.lognormvariate(0.0, 1.0) for _ in range(n_experts)]
    mse = [rng.lognormvariate(-3.0, 0.45) for _ in range(n_experts)]
    truth = math.fsum(hi * mi for hi, mi in zip(h, mse))

    # pi_i = min(1, c*h_i) with c the root of sum_i min(1, c*h_i) = n_draw.
    def probs(c):
        return [min(1.0, c * hi) for hi in h]
    lo, hi_c = 0.0, n_draw / min(h)
    for _ in range(200):
        mid = (lo + hi_c) / 2
        if math.fsum(probs(mid)) < n_draw:
            lo = mid
        else:
            hi_c = mid
    pi = probs((lo + hi_c) / 2)
    assert math.fsum(pi) == pytest.approx(n_draw, rel=1e-9)

    estimates = []
    for trial in range(4000):
        draw_rng = random.Random(trial)
        drawn = [e for e in range(n_experts) if draw_rng.random() < pi[e]]
        if len([e for e in drawn if pi[e] < 1.0]) < 2:
            continue
        estimates.append(math.fsum(h[e] * mse[e] / pi[e] for e in drawn))
    mean = math.fsum(estimates) / len(estimates)
    spread = math.sqrt(math.fsum((v - mean) ** 2 for v in estimates)
                       / (len(estimates) - 1))
    # Two standard errors OF THE MEAN, which is the tolerance an unbiasedness
    # claim actually has; the per-draw spread is far wider and is the point.
    assert abs(mean - truth) < 2.0 * spread / math.sqrt(len(estimates))

    # And the same arithmetic through the module under test, on one draw.
    drawn = [e for e in range(n_experts) if pi[e] >= 1.0] or [0]
    drawn = sorted(set(drawn) | {e for e in range(n_experts) if pi[e] < 1.0})
    probe = _packed_probe_row(n_experts, h, packed_param="down_proj",
                              out_features=4)
    sample = _sample(campaign, probe, qname=f"{STEM}.down_proj",
                     sampled=drawn, pi={e: pi[e] for e in drawn})
    contributions = {e: h[e] * mse[e] for e in drawn}
    total, stderr, m = campaign._horvitz_thompson_stack(sample, contributions)
    assert total == pytest.approx(
        math.fsum(h[e] * mse[e] / pi[e] for e in drawn), rel=1e-12)
    assert m == len([e for e in drawn if pi[e] < 1.0])
    assert stderr > 0.0


def test_a_census_has_a_true_zero_sampling_error_and_one_draw_refuses():
    campaign = _campaign()
    probe = _packed_probe_row(4, [1.0, 2.0, 3.0, 4.0], packed_param="down_proj")
    census = _sample(campaign, probe, qname=f"{STEM}.down_proj")
    total, stderr, m = campaign._horvitz_thompson_stack(
        census, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0})
    assert (total, stderr, m) == (10.0, 0.0, 0)
    one = _sample(campaign, probe, qname=f"{STEM}.down_proj",
                  sampled=[0, 1], pi={0: 1.0, 1: 0.5})
    with pytest.raises(campaign.StackSampleError, match="non-certainty draw"):
        campaign._horvitz_thompson_stack(one, {0: 1.0, 1: 2.0})


def test_zero_inclusion_probability_is_only_legal_for_a_zero_weight_expert():
    campaign = _campaign()
    probe = _packed_probe_row(3, [1.0, 0.0, 2.0], packed_param="down_proj")
    ok = _sample(campaign, probe, qname=f"{STEM}.down_proj",
                 sampled=[0, 1, 2], pi={0: 1.0, 1: 0.0, 2: 1.0})
    campaign._validate_stack_sample(ok)
    bad = _sample(campaign, probe, qname=f"{STEM}.down_proj",
                  sampled=[0, 1, 2], pi={0: 0.0, 1: 0.0, 2: 1.0})
    with pytest.raises(campaign.StackSampleError, match="cannot be drawn"):
        campaign._validate_stack_sample(bad)


def test_weights_that_do_not_sum_to_the_probe_h_trace_refuse():
    """The currency assumes sum_e h_e == h_trace; a drifted vector is refused."""
    campaign = _campaign()
    probe = _packed_probe_row(4, [1.0, 2.0, 3.0, 4.0], packed_param="down_proj")
    probe["h_trace"] = 11.0  # not 10.0
    sample = _sample(campaign, probe, qname=f"{STEM}.down_proj")
    with pytest.raises(campaign.StackSampleError, match="does not equal"):
        campaign._validate_stack_sample(sample)


# ---------------------------------------------------------------------------
# The currency
# ---------------------------------------------------------------------------

def test_stack_row_prices_exactly_what_the_expanded_experts_would():
    """A census through the stack path == the per-expert expansion's sum.

    ``tier2_per_expert_counterfactual.expand_packed_expert_rows`` gives each
    member ``h_trace_per_expert[e] / R``. The stack row must be the quantity
    that, multiplied by the PROBE row's ``h_trace``, reproduces the sum of
    those members' ``predicted_dloss``.
    """
    campaign = _campaign()
    from prismaquant.allocator_solver import predicted_dloss

    h = [3.0, 1.0, 4.0, 2.0]
    mse = {(0, "w1"): 0.05, (0, "w3"): 0.03, (1, "w1"): 0.02, (1, "w3"): 0.07,
           (2, "w1"): 0.01, (2, "w3"): 0.06, (3, "w1"): 0.04, (3, "w3"): 0.08}
    probe = _packed_probe_row(4, h)
    anchors = {}
    for (expert, role), value in mse.items():
        q = f"{STEM}.{expert}.{role}"
        anchors[q] = {"TESSERA_E2M1_K1": [
            _anchor(campaign, q, "TESSERA_E2M1_K1", "TESSERA_E2M1_K1_R256",
                    256, value)]}
    sample = _sample(campaign, probe)
    payload = _payload(campaign, anchors, {sample.packed_qname: sample})

    row = payload["costs"][f"{STEM}.gate_up_proj"]["TESSERA_E2M1_K1_R256"]
    expanded = math.fsum(predicted_dloss(h[e] / 2.0, mse[(e, role)])
                         for e in range(4) for role in ("w1", "w3"))
    assert predicted_dloss(probe["h_trace"], row["output_mse"]) == pytest.approx(
        expanded, rel=1e-12)
    assert row["dloss_stderr"] == 0.0
    assert row["cost_source"] == "tessera_campaign_measured_stack_sample"
    assert row["currency"] == campaign.CURRENCY
    assert "predicted_dloss" not in row and "weight_mse" not in row


def test_the_allocator_takes_the_output_mse_branch_for_a_stack_row():
    """Pricing must go through ``_prices_from_output_mse``, not weight-only."""
    campaign = _campaign()
    from prismaquant import allocator_candidates as ac
    from prismaquant.allocator_solver import predicted_dloss

    h = [3.0, 1.0, 4.0, 2.0]
    anchors = {
        f"{STEM}.{e}.{role}": {"TESSERA_E2M1_K1": [
            _anchor(campaign, f"{STEM}.{e}.{role}", "TESSERA_E2M1_K1",
                    "TESSERA_E2M1_K1_R256", 256, 0.01 * (e + 1))]}
        for e in range(4) for role in ("w1", "w3")}
    probe = _packed_probe_row(4, h)
    sample = _sample(campaign, probe)
    payload = _payload(campaign, anchors, {sample.packed_qname: sample})
    row = payload["costs"][f"{STEM}.gate_up_proj"]["TESSERA_E2M1_K1_R256"]

    assert ac._prices_from_output_mse(probe, row) is True
    assert ac.cost_entry_predicted_dloss(probe, row) == pytest.approx(
        predicted_dloss(probe["h_trace"], row["output_mse"]), rel=1e-12)


# ---------------------------------------------------------------------------
# The decision unit: one row, no double counting
# ---------------------------------------------------------------------------

def test_measured_members_are_not_also_cost_keys():
    campaign = _campaign()
    h = [1.0, 2.0, 3.0, 4.0]
    anchors = {
        f"{STEM}.{e}.{role}": {"TESSERA_E2M1_K1": [
            _anchor(campaign, f"{STEM}.{e}.{role}", "TESSERA_E2M1_K1",
                    "TESSERA_E2M1_K1_R256", 256, 0.02)]}
        for e in range(4) for role in ("w1", "w3")}
    # A dense unit measured in the same campaign still gets its own key.
    dense = "model.layers.18.self_attn.o_proj"
    anchors[dense] = {"TESSERA_E2M1_K1": [
        _anchor(campaign, dense, "TESSERA_E2M1_K1", "TESSERA_E2M1_K1_R256",
                256, 0.5)]}
    sample = _sample(campaign, _packed_probe_row(4, h))
    payload = _payload(campaign, anchors, {sample.packed_qname: sample})
    assert set(payload["costs"]) == {f"{STEM}.gate_up_proj", dense}
    block = payload["costs"][f"{STEM}.gate_up_proj"][
        "TESSERA_E2M1_K1_R256"]["sampled_experts"]
    # The members survive as evidence on the row, just not as decisions.
    assert sorted(block["members"]) == [0, 1, 2, 3]
    assert {m["qname"] for m in block["members"][0]} == {
        f"{STEM}.0.w1", f"{STEM}.0.w3"}


def test_a_stack_measured_whole_may_not_also_be_estimated_from_a_sample():
    campaign = _campaign()
    packed = f"{STEM}.gate_up_proj"
    anchors = {
        packed: {"TESSERA_E2M1_K1": [
            _anchor(campaign, packed, "TESSERA_E2M1_K1",
                    "TESSERA_E2M1_K1_R256", 256, 0.02)]},
        f"{STEM}.0.w1": {"TESSERA_E2M1_K1": [
            _anchor(campaign, f"{STEM}.0.w1", "TESSERA_E2M1_K1",
                    "TESSERA_E2M1_K1_R256", 256, 0.02)]},
    }
    sample = _sample(campaign, _packed_probe_row(1, [1.0]), sampled=[0])
    with pytest.raises(campaign.StackSampleError, match="measured whole"):
        _payload(campaign, anchors, {packed: sample})


def test_members_measured_under_different_contracts_do_not_average():
    campaign = _campaign()
    anchors = {}
    for e in range(2):
        for role in ("w1", "w3"):
            q = f"{STEM}.{e}.{role}"
            contract = "bfloat16" if e == 0 else "nvfp4_static"
            anchors[q] = {"TESSERA_E2M1_K1": [
                _anchor(campaign, q, "TESSERA_E2M1_K1", "TESSERA_E2M1_K1_R256",
                        256, 0.02, activation_contract=contract)]}
    sample = _sample(campaign, _packed_probe_row(2, [1.0, 1.0]))
    with pytest.raises(campaign.StackSampleError, match="activation_contract"):
        _payload(campaign, anchors, {sample.packed_qname: sample})


def test_both_packed_parameters_promote_into_one_serving_unit():
    """gate_up_proj and down_proj are one format decision, by the profile."""
    profile = _profile()
    keys = {profile.packed_expert_format_group(f"{STEM}.{p}")
            for p in ("gate_up_proj", "down_proj")}
    assert len(keys) == 1 and None not in keys


# ---------------------------------------------------------------------------
# The serving scope
# ---------------------------------------------------------------------------

def _target():
    from prismaquant import tessera_serving_scope as scope
    return scope.serving_target_from_args(argparse.Namespace(
        tessera_runtime_image=IMAGE, tessera_execution_mode="eager",
        tessera_residency="resident", tessera_platform="sm_121"))


def test_stack_rows_resolve_routed_moe_with_no_fallback():
    """The whole point of #290: the explicit scope resolves every unit."""
    from prismaquant import tessera_serving_scope as scope

    campaign = _campaign()
    stats = {
        f"{STEM}.gate_up_proj": _packed_probe_row(4, [1.0, 2.0, 3.0, 4.0]),
        f"{STEM}.down_proj": _packed_probe_row(
            4, [1.0, 2.0, 3.0, 4.0], packed_param="down_proj", out_features=4),
        "model.layers.18.self_attn.o_proj": {
            "router_path": None, "expert_id": None,
            "out_features": 8, "in_features": 8, "n_params": 64},
    }
    contexts = scope.context_by_unit_from_stats(_target(), stats, _profile())
    assert {n: c.structure for n, c in contexts.items()} == {
        f"{STEM}.gate_up_proj": "routed_moe",
        f"{STEM}.down_proj": "routed_moe",
        "model.layers.18.self_attn.o_proj": "dense"}

    # And the topology the scope reads is on the cost row too, so a payload
    # carries its own structure rather than depending on a probe re-read.
    anchors = {
        f"{STEM}.{e}.{role}": {"TESSERA_E2M1_K1": [
            _anchor(campaign, f"{STEM}.{e}.{role}", "TESSERA_E2M1_K1",
                    "TESSERA_E2M1_K1_R256", 256, 0.02)]}
        for e in range(4) for role in ("w1", "w3")}
    sample = _sample(campaign, stats[f"{STEM}.gate_up_proj"])
    payload = _payload(campaign, anchors, {sample.packed_qname: sample})
    row = payload["costs"][f"{STEM}.gate_up_proj"]["TESSERA_E2M1_K1_R256"]
    assert scope.unit_structure_from_stats(
        f"{STEM}.gate_up_proj", row, _profile()) == "routed_moe"


def test_a_per_expert_row_without_packed_topology_still_refuses():
    """Principle 14: the scope may not invent a topology it was not given.

    This is the refusal campaign-01 section 7a hit on all nine runs, and it must
    survive #290 -- the fix is to stop EXPANDING the probe, not to teach the
    scope to guess.
    """
    from prismaquant import tessera_serving_scope as scope

    expanded_child = {"router_path": None, "expert_id": 0}
    with pytest.raises(ValueError, match="conflicting router_path/expert_id"):
        scope.unit_structure_from_stats(f"{STEM}.0.w2", expanded_child, _profile())


# ---------------------------------------------------------------------------
# On real measured anchors
# ---------------------------------------------------------------------------

def _fixture_anchors(campaign, data, packed_param, roles):
    anchors = {}
    for qname, rows in data["member_anchors"].items():
        if qname.rsplit(".", 1)[-1] not in roles:
            continue
        by_family = {}
        for row in rows:
            by_family.setdefault(row["family"], []).append(
                campaign.CampaignAnchor(qname=qname, **{
                    k: v for k, v in row.items() if k != "qname"}))
        anchors[qname] = by_family
    return anchors


def test_lfm_layer18_census_matches_the_per_expert_expansion():
    """On the real campaign-01 anchors, a 32-expert census is exact."""
    campaign = _campaign()
    from prismaquant.allocator_solver import predicted_dloss

    data = _fixture()
    packed_qname = f"{STEM}.down_proj"
    probe = data["packed_probe_rows"][packed_qname]
    anchors = _fixture_anchors(campaign, data, "down_proj", {"w2"})
    assert len(anchors) == 32
    sample = campaign.stack_sample_from_probe(
        packed_qname, probe, _profile(),
        sampled_experts=range(32), inclusion_prob={e: 1.0 for e in range(32)},
        seed=1, design="census")
    payload = _payload(campaign, anchors, {packed_qname: sample})
    rows = payload["costs"][packed_qname]
    assert rows, "the fixture's shared rungs must produce stack rows"

    fmt = "TESSERA_BF16_K1_R256"
    row = rows[fmt]
    per_expert = {q: next(a for a in v if a["format_name"] == fmt)["dloss"]
                  for q, v in data["member_anchors"].items()
                  if q.endswith(".w2")}
    expanded = math.fsum(
        predicted_dloss(probe["h_trace_per_expert"][e], per_expert[f"{STEM}.{e}.w2"])
        for e in range(32))
    assert predicted_dloss(probe["h_trace"], row["output_mse"]) == pytest.approx(
        expanded, rel=1e-9)
    assert row["dloss_stderr"] == 0.0
    assert row["sampled_experts"]["variance_estimator"] == "census"
    assert row["num_experts"] == 32
    assert row["_packed_experts_module"] == probe["_packed_experts_module"]


def test_lfm_layer18_sample_brackets_the_census_it_estimates():
    """A PPS sample of the real stack lands within a few SE of the truth."""
    campaign = _campaign()
    data = _fixture()
    packed_qname = f"{STEM}.down_proj"
    probe = data["packed_probe_rows"][packed_qname]
    anchors = _fixture_anchors(campaign, data, "down_proj", {"w2"})
    fmt = "TESSERA_BF16_K1_R256"
    h = probe["h_trace_per_expert"]

    def probs(c):
        return [min(1.0, c * v) for v in h]
    lo, hi = 0.0, 16.0 / min(v for v in h if v > 0)
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.fsum(probs(mid)) < 16:
            lo = mid
        else:
            hi = mid
    pi = probs((lo + hi) / 2)
    rng = random.Random(4711)
    drawn = sorted(e for e in range(32) if pi[e] >= 1.0 or rng.random() < pi[e])
    assert len([e for e in drawn if pi[e] < 1.0]) >= 2

    sample = campaign.stack_sample_from_probe(
        packed_qname, probe, _profile(), sampled_experts=drawn,
        inclusion_prob={e: pi[e] for e in drawn}, seed=4711)
    row = _payload(campaign, anchors, {packed_qname: sample})[
        "costs"][packed_qname][fmt]

    per_expert = {q: next(a for a in v if a["format_name"] == fmt)["dloss"]
                  for q, v in data["member_anchors"].items()
                  if q.endswith(".w2")}
    truth = math.fsum(h[e] * per_expert[f"{STEM}.{e}.w2"]
                      for e in range(32)) / probe["h_trace"]
    assert row["dloss_stderr"] > 0.0
    assert abs(row["output_mse"] - truth) < 4.0 * row["dloss_stderr"]
    block = row["sampled_experts"]
    assert block["variance_estimator"] == "hartley_rao"
    assert block["n_sampled"] == len(drawn) and block["n_experts"] == 32
    assert block["n_random_stratum"] == len([e for e in drawn if pi[e] < 1.0])


def test_fixture_records_the_probe_it_was_distilled_from():
    """A committed fixture that cannot say where it came from is a guess."""
    data = _fixture()
    sources = data["sources"]
    probe = next(k for k in sources if k.endswith("probe.pkl"))
    assert len(sources[probe]["sha256"]) == 64
    assert "not the campaign's own calibration draw" in data["note"].replace(
        "NOT", "not")
