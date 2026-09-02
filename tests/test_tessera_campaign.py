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
import pathlib

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
    # Weights-only deliberately: this pins a property of the WIRE (that a
    # deeper encode is not a superset of a shallower one), and the encoder's
    # weights-only bytes are unchanged by the H-aware default.
    _, lo = _encode_and_render(W, "TESSERA_E2M1_K2_R384",
                               hessian_required=False)
    _, hi = _encode_and_render(W, "TESSERA_E2M1_K2_R512",
                               hessian_required=False)
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
        # Weights-only on both arms, so the A-leg ratio is the only thing
        # that differs between the routes -- an H would be a second variable.
        render, _ = _encode_and_render(W, name, hessian_required=False)
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


# ---------------------------------------------------------------------------
# The Hessian contract
# ---------------------------------------------------------------------------

def test_the_hessian_kwarg_pin_matches_the_pinned_encoder():
    """A stale API pin fails here, not silently downgrades a price.

    ``TESSERA_HESSIAN_KWARG`` is a claim about the pinned ``tessera.export``.
    If it names a parameter the pinned encoder does not have, every encode
    would quietly drop the Hessian, so the claim is checked rather than
    trusted (principle 14).
    """
    import inspect

    from tessera import export as texport

    from prismaquant.tessera_render import TESSERA_HESSIAN_KWARG

    if TESSERA_HESSIAN_KWARG is None:
        pytest.skip("H-aware encoder branch not pinned yet")
    params = inspect.signature(texport.encode_linear_planes).parameters
    assert TESSERA_HESSIAN_KWARG in params, sorted(params)


def test_encoding_without_a_hessian_is_refused_not_silently_downgraded():
    from prismaquant.tessera_render import (
        HessianContractError, encode_tessera_unit,
        tessera_encoder_hessian_status,
    )

    status = tessera_encoder_hessian_status()
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    if status["accepted"]:
        # The encoder can take one; then a *missing* H is the refusal.
        with pytest.raises(HessianContractError):
            encode_tessera_unit(w, "TESSERA_E2M1_K2_R512", hessian=None)
    else:
        # The encoder cannot take one; then requiring one is the refusal, and
        # the message must name the reason rather than say "unsupported".
        with pytest.raises(HessianContractError) as excinfo:
            encode_tessera_unit(w, "TESSERA_E2M1_K2_R512",
                                hessian=torch.eye(256))
        assert "TESSERA_HESSIAN_KWARG" in str(excinfo.value)


@CUDA
def test_weights_only_is_reachable_but_only_deliberately():
    from prismaquant.tessera_render import encode_tessera_unit

    w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    render, blob = encode_tessera_unit(
        w, "TESSERA_E2M1_K2_R512", hessian_required=False)
    assert render.shape == w.shape and len(blob) > 0


def test_the_seam_passes_the_hessian_through_when_the_encoder_takes_it():
    """The pass-through itself, independent of whether the branch is merged.

    Monkeypatches an encoder that accepts the kwarg and asserts the seam hands
    it the tensor. Without this the whole contract is untested until the pin
    moves, which is exactly when nobody re-reads it.
    """
    import prismaquant.tessera_render as tr

    seen = {}

    class _Unit:
        blob = b""

    def fake_encode_linear_planes(weight, *, grid, q256, name, verify=True,
                                  gram=None, token_count=None, **kw):
        seen["gram"] = gram
        seen["token_count"] = token_count
        seen["q256"] = q256
        return None

    def fake_encode_linear(weight, **kwargs):
        fake_encode_linear_planes(weight, **kwargs)
        return _Unit()

    H = torch.eye(256)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(tr._tessera_export, "encode_linear_planes",
                       fake_encode_linear_planes, raising=True)
        monkey.setattr(tr._tessera_export, "encode_linear",
                       fake_encode_linear, raising=True)
        monkey.setattr(tr, "TESSERA_HESSIAN_KWARG", "gram", raising=False)
        tr._encoder_accepts_hessian.cache_clear()
        monkey.setattr(
            "tessera.unit_artifact.read_unit_artifact",
            lambda blob, device=None: torch.zeros(64, 256), raising=True)
        tr.encode_tessera_unit(
            torch.randn(64, 256, dtype=torch.bfloat16),
            "TESSERA_E2M1_K2_R512", hessian=H, token_count=4096)
    finally:
        monkey.undo()
        tr._encoder_accepts_hessian.cache_clear()
    assert seen["gram"] is H, seen
    assert seen["token_count"] == 4096
    assert seen["q256"] == 512


def test_a_missing_qname_in_the_hessian_map_is_a_hard_failure():
    """The landmine this project has already stepped on once.

    A render whose activation lookup misses and falls through to RTN raises
    nothing and produces a plausible tensor. The Hessian map is keyed the same
    way, so the miss is made loud here.
    """
    from prismaquant.tessera_campaign import _measure_anchor
    from prismaquant.tessera_render import HessianContractError

    with pytest.raises(HessianContractError) as excinfo:
        _measure_anchor(
            qname="model.layers.0.self_attn.q_proj",
            weight=torch.randn(64, 256, dtype=torch.bfloat16),
            activations=torch.randn(8, 256),
            format_name="TESSERA_E2M1_K2_R512",
            cache=None, wire_dir=pathlib.Path("."),
            hessians={"model.layers.0.self_attn.k_proj": torch.eye(256)},
            token_count=4096, hessian_required=True,
        )
    assert "q_proj" in str(excinfo.value)


def test_the_hessian_sees_every_row_not_just_the_scored_ones():
    """H is XtX over ALL calibration rows; the score's cap must not reach it.

    A 256-row cap on a 3072-column Linear would give a rank-256 Hessian. The
    accumulation therefore runs before the keep's early return, and this is
    the test that says so: with ``max_rows`` far below the rows the module
    sees, H must still equal the full Gram.
    """
    from prismaquant.tessera_campaign import _collect_activations

    torch.manual_seed(0)
    cols, batches, rows_per_batch = 32, 3, 40

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(cols, 8, bias=False)

        def forward(self, x):
            return self.lin(x)

    net = _Net().eval()
    tokens = [torch.randn(rows_per_batch, cols) for _ in range(batches)]
    rows, hess, seen = _collect_activations(
        net, ["lin"], tokens, max_rows=5, device="cpu", want_hessian=True)

    every = torch.cat(tokens, dim=0).to(torch.float32)
    assert seen["lin"] == batches * rows_per_batch
    assert rows["lin"].shape[0] == 5           # the score's cap still binds
    torch.testing.assert_close(hess["lin"], every.t() @ every,
                               rtol=1e-4, atol=1e-4)


def test_one_cost_table_carries_one_hessian_identity():
    from prismaquant.tessera_menu import assert_uniform_hessian_identity

    ident = {"supplied": True, "text_sha": "abc", "token_count": 4096,
             "kwarg": "gram"}
    ok = {"q": {"TESSERA_E4M3_K1_R512": {"hessian_identity": ident},
                "TESSERA_E2M1_K2_R512": {"hessian_identity": dict(ident)}}}
    assert assert_uniform_hessian_identity(ok)["stamped_rows"] == 2

    mixed = {"q": {
        "TESSERA_E4M3_K1_R512": {"hessian_identity": ident},
        "TESSERA_E2M1_K2_R512": {"hessian_identity": {
            "supplied": False, "text_sha": None, "token_count": 0,
            "kwarg": None}}}}
    with pytest.raises(ValueError, match="mixes Hessian identities"):
        assert_uniform_hessian_identity(mixed)

    # Absence of a claim is reported, not read as agreement.
    bare = {"q": {"TESSERA_E4M3_K1_R512": {}}}
    got = assert_uniform_hessian_identity(bare)
    assert got["unstamped_rows"] == 1 and got["supplied"] is None


def test_the_token_sha_identifies_the_calibration_draw():
    from prismaquant.tessera_campaign import _token_sha

    a = [torch.arange(0, 16).reshape(1, 16)]
    b = [torch.arange(0, 16).reshape(1, 16)]
    c = [torch.arange(1, 17).reshape(1, 16)]
    assert _token_sha(a) == _token_sha(b)
    assert _token_sha(a) != _token_sha(c)


def test_every_payload_row_carries_the_hessian_identity():
    """Measured AND interpolated rows, or the guard has nothing to compare.

    The identity is what makes a half-H-aware cost table detectable. If only
    the measured branch stamped it, a table could be 3 stamped rows and 766
    unstamped ones and still pass a uniformity check.
    """
    from prismaquant.tessera_campaign import (
        CampaignAnchor, campaign_cost_payload,
    )
    from prismaquant.tessera_menu import (
        assert_uniform_hessian_identity, expand_tessera_menu,
    )

    prov = {"provenance": {"hessian": {
        "supplied": False, "mode": "off", "text_sha": "deadbeef",
        "token_count": 0, "kwarg": None}}}
    menus = {"q": expand_tessera_menu((512, 512), mode="research",
                                      tp_degree=1)}
    anchors = {"q": {"TESSERA_E2M1_K2": [
        CampaignAnchor(
            qname="q", family="TESSERA_E2M1_K2",
            format_name=f"TESSERA_E2M1_K2_R{r}", body_rate_q256=r,
            dloss=1.0 / r, dloss_stderr=0.0, memory_bytes=1,
            bits_per_param=r / 256, activation_contract="fp4_e2m1",
            activation_quantized=True, wire_bytes=1, seconds=0.1)
        for r in (128, 512, 896)]}}
    payload = campaign_cost_payload(anchors, menus, loo={}, provenance=prov)
    rows = payload["costs"]["q"]
    assert len(rows) > 100
    measured = [v for v in rows.values() if v.get("output_mse_measured")]
    interpolated = [v for v in rows.values()
                    if not v.get("output_mse_measured")]
    assert measured and interpolated
    for row in rows.values():
        assert row["hessian_identity"]["text_sha"] == "deadbeef"
        assert row["hessian_identity"]["supplied"] is False
    got = assert_uniform_hessian_identity(payload["costs"])
    assert got["unstamped_rows"] == 0
    assert got["stamped_rows"] == len(rows)
    assert got["supplied"] is False


# ---------------------------------------------------------------------------
# The production render seam: render_production_weight owns Tessera
# ---------------------------------------------------------------------------

def test_render_production_weight_does_not_fall_to_the_registry_for_tessera():
    """A TESSERA fmt must not reach the registry's weights-only QDQ.

    Before this seam existed, ``render_production_weight``'s cascade ended at
    ``weighted_quantize_dequantize`` -> the synthesized registry
    ``quantize_dequantize`` -> ``render_tessera_weight``, which is a *weights-
    only reconstruction* and not the decoded wire. Both halves of that are
    silent: an allocator-chosen Tessera unit would have been cached, KL-scored
    and (eventually) exported from bytes nobody encoded. The refusal below is
    the observable proof the interception happens: the registry path would have
    returned a tensor, this raises.
    """
    from prismaquant.production_weight_cache import render_production_weight
    from prismaquant.tessera_render import HessianContractError

    w = torch.randn(64, 256, dtype=torch.bfloat16)
    with pytest.raises(HessianContractError):
        render_production_weight(
            w, "TESSERA_E2M1_K2_R256",
            qname="model.layers.0.self_attn.q_proj",
            activations={},          # the key misses -- the known landmine
            levers={},
        )


def test_a_wrong_shaped_activation_is_refused_not_reshaped():
    from prismaquant.production_weight_cache import render_production_weight
    from prismaquant.tessera_render import HessianContractError

    w = torch.randn(64, 256, dtype=torch.bfloat16)
    with pytest.raises(HessianContractError, match="wrong-key"):
        render_production_weight(
            w, "TESSERA_E2M1_K2_R256",
            qname="q",
            activations={"q": torch.randn(32, 128)},
            levers={},
        )


def test_weights_only_on_the_production_seam_is_a_stamped_lever():
    """``tessera_weights_only`` is the deliberate opt-out, and it is the ONLY
    way a Tessera unit renders without an H on this build."""
    from prismaquant.tessera_render import (
        encode_tessera_unit, render_tessera_production,
    )

    if not torch.cuda.is_available():
        pytest.skip("Tessera encodes need CUDA")
    w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    got = render_tessera_production(
        w, "TESSERA_E2M1_K2_R256", qname="q",
        activations={}, levers={"tessera_weights_only": True},
    )
    want, _blob = encode_tessera_unit(
        w, "TESSERA_E2M1_K2_R256", hessian_required=False)
    # The production render IS the decoded wire, bit for bit -- not a second
    # reconstruction that happens to agree to a tolerance.
    assert torch.equal(got, want)
    assert got.shape == w.shape and got.dtype == w.dtype


def test_token_count_is_only_sent_when_the_encoder_takes_one(monkeypatch):
    """Pinning the H kwarg must not TypeError every encode on a signature that
    takes H but no count."""
    import prismaquant.tessera_render as tr

    seen = {}

    def fake_encode_linear(weight, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after argument capture")

    monkeypatch.setattr(tr, "TESSERA_HESSIAN_KWARG", "hessians")
    monkeypatch.setattr(
        tr, "_encoder_accepts_hessian",
        lambda: (True, ("weight", "grid", "q256", "name", "hessians")))
    monkeypatch.setattr(
        tr._tessera_export, "encode_linear", fake_encode_linear)
    w = torch.randn(8, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="argument capture"):
        tr.encode_tessera_unit(
            w, "TESSERA_E2M1_K2_R256",
            hessian=torch.eye(32), token_count=4096)
    assert "hessians" in seen
    assert "token_count" not in seen


def test_the_weights_only_lever_forms_no_hessian_at_all(monkeypatch):
    """``tessera_weights_only`` must mean weights-only *after* the pin too.

    Forming H and letting ``encode_tessera_unit`` drop it because the pinned
    encoder cannot take one is correct exactly until the day the kwarg lands --
    at which point a lever named "weights only" starts shipping H-aware bytes
    under a ``supplied=false`` stamp. The monkeypatch simulates that day.
    """
    import prismaquant.tessera_render as tr

    seen = {}

    def fake_encode_linear(weight, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after argument capture")

    monkeypatch.setattr(tr, "TESSERA_HESSIAN_KWARG", "hessians")
    monkeypatch.setattr(
        tr, "_encoder_accepts_hessian",
        lambda: (True, ("weight", "grid", "q256", "name", "hessians")))
    monkeypatch.setattr(
        tr._tessera_export, "encode_linear", fake_encode_linear)

    w = torch.randn(8, 32, dtype=torch.bfloat16)
    acts = {"q": torch.randn(64, 32)}
    with pytest.raises(RuntimeError, match="argument capture"):
        tr.render_tessera_production(
            w, "TESSERA_E2M1_K2_R256", qname="q", activations=acts,
            levers={"tessera_weights_only": True},
        )
    assert "hessians" not in seen

    seen.clear()
    with pytest.raises(RuntimeError, match="argument capture"):
        tr.render_tessera_production(
            w, "TESSERA_E2M1_K2_R256", qname="q", activations=acts, levers={},
        )
    assert "hessians" in seen


def test_a_production_cache_miss_does_not_fall_back_to_the_registry():
    """``STRICT_PRODUCTION_CACHE=0`` is not permission to price other bytes.

    Both miss paths end at the format's registry ``quantize_dequantize``; for
    Tessera that is the weights-only reconstruction, so the fallback would put
    a different tensor behind the same format name with nothing raised. CB
    already refuses there for the same reason.
    """
    from prismaquant.weight_session import WeightSession

    lin = torch.nn.Linear(32, 8, bias=False).to(torch.bfloat16)
    session = WeightSession(torch.nn.Sequential(lin),
                            production_weight_cache=None,
                            strict_production_cache=False)
    # The qname map normally comes from a model profile; registering the
    # alias directly keeps this test about the fallback and not about
    # architecture discovery.
    session._linear_by_qname["lin"] = (lin, "weight")
    with pytest.raises(RuntimeError, match="Tessera"):
        session._format_weight("lin", "TESSERA_E2M1_K2_R256")
