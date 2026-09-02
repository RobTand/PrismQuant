"""The anchor campaign: what it measures, what it predicts, and the difference.

A Tessera family addresses thousands of rungs per unit and every one of them
needs its own encode (see ``test_no_embedded_axis_on_the_wire`` -- the
"embedded ladder" is a *decode-time completion* axis and does not exist on the
serialised wire). So the cost stage cannot be "price every rung": it measures a
few anchors per (unit, family) and interpolates between them, refuses to
extrapolate past them, and stamps every row with which of the two it was.

These tests pin the three properties that make that honest:

* a measured row is priced from its own measurement, and an interpolated row
  is priced from the family's own output-space fit -- never from a weight-space
  number wearing an output-space field name;
* a rung outside the measured envelope is **omitted**, not extrapolated;
* the render that was priced IS the decoded wire, and the wire is stored.
"""
import math
import os

import pytest

torch = pytest.importorskip("torch")

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Tessera encodes need CUDA")


# ---------------------------------------------------------------------------
# Anchor placement (host-only)
# ---------------------------------------------------------------------------

def test_anchor_schedule_spans_the_family_ends():
    from prismaquant.tessera_campaign import anchor_schedule

    rates = anchor_schedule(128, 896, 3)
    assert rates[0] == 128 and rates[-1] == 896
    assert len(set(rates)) == 3
    assert rates == sorted(rates)


def test_next_anchor_splits_the_worst_predicted_interval():
    from prismaquant.tessera_campaign import next_anchor_rate

    rates = [128, 512, 896]
    loo = {"per_anchor": {"512": {"abs_log2_error": 2.0}}}
    nxt = next_anchor_rate(rates, loo)
    assert nxt is not None
    assert 128 < nxt < 896
    assert nxt not in rates


# ---------------------------------------------------------------------------
# The payload's own contract (host-only)
# ---------------------------------------------------------------------------

def _anchor(qname, family, rung, dloss, *, bytes_=1000):
    from prismaquant.tessera_campaign import CampaignAnchor

    return CampaignAnchor(
        qname=qname, family=family, format_name=f"{family}_R{rung}",
        body_rate_q256=rung, dloss=dloss, dloss_stderr=0.0,
        memory_bytes=bytes_, bits_per_param=rung / 256.0,
        activation_contract="w4a4-nvfp4-e2m1-group16-ue4m3",
        activation_quantized=True, wire_bytes=bytes_, seconds=1.0,
    )


def _menu(qname, family, rungs):
    from prismaquant.tessera_menu import expand_tessera_menu, MENU_RESEARCH

    rows = expand_tessera_menu((2048, 1024), mode=MENU_RESEARCH)
    return {qname: [r for r in rows
                    if r.family == family and r.body_rate_q256 in rungs]}


def test_measured_and_interpolated_rows_are_told_apart():
    from prismaquant.allocator_candidates import (
        cost_entry_is_band_interpolated, cost_entry_source,
    )
    from prismaquant.tessera_campaign import campaign_cost_payload

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    anchors = {q: {fam: [_anchor(q, fam, r, d) for r, d in
                         ((128, 1e-2), (512, 1e-3), (896, 1e-4))]}}
    payload = campaign_cost_payload(
        anchors, _menu(q, fam, {128, 384, 512, 896}), loo={}, provenance={})
    rows = payload["costs"][q]
    measured = rows[f"{fam}_R512"]
    assert cost_entry_source({}, measured) == "tessera_campaign_measured"
    assert not cost_entry_is_band_interpolated(measured)
    interp = rows[f"{fam}_R384"]
    assert cost_entry_source({}, interp) == "tessera_campaign_interpolated"
    assert cost_entry_is_band_interpolated(interp)


def test_an_interpolated_row_is_priced_in_output_space():
    """Not from weight_mse via h_trace: the fit IS an output-space number.

    The failure this pins is silent and expensive -- an interpolated row that
    fell through to the weight-only branch would be multiplied by the unit's
    Fisher trace, double-counting a transfer the number already contains, and
    the DP would rank Tessera's rungs against each other on a scale nothing
    else on the menu uses.
    """
    from prismaquant.allocator_candidates import (
        cost_entry_activation_pricing_branch, cost_entry_predicted_dloss,
    )
    from prismaquant.tessera_campaign import campaign_cost_payload

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    anchors = {q: {fam: [_anchor(q, fam, r, d) for r, d in
                         ((128, 1e-2), (512, 1e-3), (896, 1e-4))]}}
    payload = campaign_cost_payload(
        anchors, _menu(q, fam, {128, 384, 512, 896}), loo={}, provenance={})
    row = payload["costs"][q][f"{fam}_R384"]
    stats = {"h_trace": 1e6, "n_params": 2048 * 1024}
    priced = cost_entry_predicted_dloss(stats, row, format_name=f"{fam}_R384")
    assert math.isclose(
        priced, 0.5 * stats["h_trace"] * row["output_mse"], rel_tol=1e-9)
    assert "predicted_dloss" not in row
    branch = cost_entry_activation_pricing_branch(
        stats, row, f"{fam}_R384", None)
    assert "interpolated" in branch or "measured" in branch, branch


def test_a_rung_outside_the_envelope_is_omitted_not_extrapolated():
    from prismaquant.tessera_campaign import campaign_cost_payload

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    anchors = {q: {fam: [_anchor(q, fam, r, d) for r, d in
                         ((384, 1e-3), (512, 5e-4), (640, 1e-4))]}}
    payload = campaign_cost_payload(
        anchors, _menu(q, fam, {128, 384, 512, 640, 896}), loo={},
        provenance={})
    rows = payload["costs"][q]
    assert f"{fam}_R128" not in rows
    assert f"{fam}_R896" not in rows
    assert f"{fam}_R512" in rows


def test_non_monotone_anchors_are_recorded_not_laundered():
    from prismaquant.tessera_campaign import campaign_cost_payload

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    anchors = {q: {fam: [_anchor(q, fam, r, d) for r, d in
                         ((128, 1e-4), (512, 1e-3), (896, 1e-2))]}}
    payload = campaign_cost_payload(
        anchors, _menu(q, fam, {128, 384, 512, 896}), loo={}, provenance={})
    assert payload["non_interpolable"], payload["non_interpolable"]
    rows = payload["costs"][q]
    # The measured rows survive; nothing between them was invented.
    assert f"{fam}_R384" not in rows
    assert f"{fam}_R128" in rows


# ---------------------------------------------------------------------------
# The wire, and the render that is the wire
# ---------------------------------------------------------------------------

@CUDA
def test_no_embedded_axis_on_the_wire():
    """Two rungs are two encodes; neither wire is a truncation of the other.

    Recorded as a test because the task this campaign was designed for was
    briefed on the opposite premise -- "one deep encode per unit, then exact
    decodes of every lower rate". Tessera's embedded axis is a decode-time
    COMPLETION axis (``encode_linear`` writes ``completion=0``,
    ``build_unit_artifact`` writes exactly one terminal, and a WINDOW body has
    no completion axis at all), so it produces weights, not wires. Every rung
    on the serialised wire costs its own encode, which is the fact that makes
    an anchor campaign necessary rather than merely convenient.
    """
    from prismaquant.tessera_campaign import _encode_and_render

    torch.manual_seed(0)
    W = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    _, lo = _encode_and_render(W, "TESSERA_E2M1_K2_R384")
    _, hi = _encode_and_render(W, "TESSERA_E2M1_K2_R512")
    assert len(hi) > len(lo)
    assert not bytes(hi).startswith(bytes(lo))


@CUDA
def test_the_priced_render_is_the_decoded_wire():
    """Principle 8, at the one place it could silently break.

    ``_encode_and_render`` passes ``verify=False`` to the encoder because the
    encoder's ``verify`` compares its in-memory reconstruction against the
    decoded artifact and this campaign only ever uses the latter. That is the
    right call only if the two agree, which nothing downstream re-checks --
    so it is pinned here, once.
    """
    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact

    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_render import _grid_for

    torch.manual_seed(0)
    W = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    family, rung = parse_tessera_format_name("TESSERA_E2M1_K2_R384")
    unit = encode_linear(
        W, grid=_grid_for(family), q256=int(rung),
        name="TESSERA_E2M1_K2_R384", verify=True)
    decoded = read_unit_artifact(unit.blob, device="cuda")
    assert decoded.shape == W.shape


@CUDA
@pytest.mark.slow
def test_the_same_weight_rate_costs_differently_on_the_two_routes():
    """Every candidate is priced AS SERVED -- and the two routes differ.

    A Tessera rung's price is not a property of its weight rate alone: the
    E2M1 families execute under NVFP4's W4A4 contract and the E4M3 family
    under per-token FP8, so the activation leg the render is scored against
    is a different quantiser. This measures both rungs' output MSE twice --
    once under the route's own activation contract, once under the identity --
    and asserts the RATIO differs between the routes. Asserting only that each
    route differs from identity would pass on a harness that applied one
    contract to both.
    """
    from prismaquant import format_registry as fr
    from prismaquant.production_weight_cache import _local_forward_render_score
    from prismaquant.tessera_campaign import _encode_and_render

    torch.manual_seed(0)
    W = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    X = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    ratios = {}
    for name in ("TESSERA_E2M1_K2_R896", "TESSERA_E4M3_K1_R896"):
        render, _ = _encode_and_render(W, name)
        spec = fr.get_format(name)
        under_route = _local_forward_render_score(
            reference_weight=W.float(), rendered_weight=render.float(),
            activations=X,
            activation_quantize=spec.activation_quantize_dequantize,
            activation_max_abs=None)[0]
        under_identity = _local_forward_render_score(
            reference_weight=W.float(), rendered_weight=render.float(),
            activations=X,
            activation_quantize=lambda t: t, activation_max_abs=None)[0]
        ratios[name] = float(under_route) / max(float(under_identity), 1e-30)

    e2m1, e4m3 = (ratios["TESSERA_E2M1_K2_R896"],
                  ratios["TESSERA_E4M3_K1_R896"])
    assert e2m1 > 1.05, ratios          # W4A4 is a real, priced A leg
    assert e2m1 / e4m3 > 1.5, ratios    # and the two routes are not the same
