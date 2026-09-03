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


def test_a_refused_surface_is_not_reported_as_a_perfect_fit():
    """A missing leave-one-out error is not a zero one.

    The campaign's per-surface readout reads ``max_abs_log2_error`` with a
    default.  A surface that refused to exist has no such key, so the default
    used to make the one surface that does not interpolate at all read as the
    one that interpolates perfectly -- and closed its gate, on the record, at
    exactly the surface the adaptive loop exists to keep spending on.
    """

    from prismaquant.tessera_campaign import _loo_for, _loo_refused, _surface_loo
    from prismaquant.tessera_rate_surface import leave_one_anchor_out

    q, fam = "m.0.q_proj", "TESSERA_E2M1_K2"
    bad = [_anchor(q, fam, r, d) for r, d in
           ((128, 1e-4), (512, 1e-3), (896, 1e-2))]
    good = [_anchor(q, fam, r, d) for r, d in
            ((128, 1e-2), (512, 1e-3), (896, 1e-4))]

    refused = _loo_for(bad, leave_one_anchor_out)
    assert _loo_refused(refused), refused
    assert _surface_loo(refused, 0.25) == (None, False)

    fitted = _loo_for(good, leave_one_anchor_out)
    assert not _loo_refused(fitted)
    value, closed = _surface_loo(fitted, 0.25)
    assert value is not None and closed is (value <= 0.25)

    # And a surface that was never built at all is not a perfect fit either.
    assert _surface_loo(None, 0.25) == (None, False)


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

def test_the_pinned_encoder_accepts_every_keyword_the_source_emits():
    """A stale API pin fails here, not silently downgrades a price.

    ``ActivationSource.for_unit`` decides which keywords an H-aware encode
    needs; ``encode_linear_planes`` is what receives them. Neither is asserted
    in PrismaQuant -- the probe asks the object what it emits and the
    signature what it takes, and the seam refuses when they do not match
    (principle 14).
    """
    import inspect

    from tessera import export as texport

    from prismaquant.tessera_render import (
        _encoder_accepts_hessian, tessera_encoder_hessian_status,
    )

    accepted, required, _params = _encoder_accepts_hessian()
    params = inspect.signature(texport.encode_linear_planes).parameters
    # The UNION over planes: the emitted set is plane-dependent (s6b's grouped
    # words have no metric-aware refit, so that plane emits no
    # ``refit_metric``), and this tuple is the seam's forwarded-key whitelist.
    # Probing one plane would narrow it and make the seam refuse another
    # plane's legitimate keyword as unknown.
    assert required == ("ldl", "ldl_block", "refit_metric", "refit_reach_floor")
    assert accepted, sorted(params)
    assert all(k in params for k in required), sorted(params)
    # And the recipe travels from Tessera's own defaults, not from a comment --
    # including the fact that the refit objective is now keyed BY PLANE. A
    # receipt that quoted one objective would print a true statement about one
    # plane over an artifact built on another.
    assert tessera_encoder_hessian_status()["recipe"] == {
        "ldlq_sigma": 1.0, "ldlq_block": 32,
        "refit_objective": {"channel": "hessian", "lut16": "h^1.0",
                            "s6b": "plain"},
        "refit_reach_floor": False,
    }
    # Plain dicts all the way down: this block is stamped into cost payloads.
    import json
    json.dumps(tessera_encoder_hessian_status()["recipe"])


def test_the_hessian_applies_exactly_where_tessera_says_it_does():
    """The predicate and Tessera's raise site agree, on real encodes.

    ``rung_accepts_hessian`` DERIVES its answer -- does the pinned
    ``ActivationSource`` emit an H-bearing keyword for this rung's plane --
    rather than restating Tessera's condition, which principle 14 forbids. This
    is the attestation: build real activation kwargs **on that rung's own
    resolved plane**, encode, and require that the encode succeeds exactly when
    the predicate said it would.

    The roster is derived, not named. It used to be five hand-written format
    strings, which is the "pin the rule, not the roster" defect: when the
    ``ANCHOR_BUDGET_BITS`` wall became the TCQ body's rather than the grid's
    and admitted ``BF16_K1`` -- a twelfth family, a third window width -- the
    five names did not notice, and the new family
    went unprobed by the test whose whole subject is "every family". So the
    roster is now asked for: ``menu_families()`` (the families that both render
    and serialise) crossed with the **distinct wire recipes** each one resolves
    to over ``realisable_rungs``, keeping the lowest rung that exhibits each.
    That is exactly the set of (body, plane, window width, span) the predicate
    can be asked about, at the cheapest rung per combination, and a Tessera
    release that adds a family or splits a window band grows it on its own.

    Each rep gets its own kwargs, because that is the contract: the refit
    objective is keyed by scale plane, and one shared kwargs dict -- what this
    test used to build -- prices the other planes under the first's objective.
    That construction is why the old verdict map read ``False`` for E2M1: it
    probed a LUT-plane encode with the CHANNEL plane's answer.

    Every verdict is ``True``, and that is a Tessera fact, not a taste.
    Tessera deleted both CHANNEL-only guards on 2026-09-02
    (``tessera-ldlq-lut-plane-served-2026-09-02.md``); on the LUT plane the
    H-aware wire is served KL 0.5310 against the weights-only wire's 0.6404 at
    identical bytes. So weights-only on the W4A4 route is a downgrade after
    all, and pricing it that way priced bytes the exporter does not write. The
    assertion is therefore ``all(...)`` over a derived roster rather than a
    literal map: a map would be a roster again, and would have to be edited
    every time the menu grows.
    """
    from tessera.errors import GrammarError

    from prismaquant.tessera_formats import (
        realisable_rungs, scale_plane_name, tessera_wire_recipe,
    )
    from prismaquant.tessera_hessian import (
        activation_source, calibration_identity, encoder_kwargs, hessian_from_rows,
    )
    from prismaquant.tessera_menu import menu_families
    from prismaquant.tessera_render import rung_accepts_hessian

    reps: list[tuple[object, int, object]] = []
    for spec in menu_families():
        cheapest: dict[tuple, tuple] = {}
        for rung in realisable_rungs(spec):
            wire = tessera_wire_recipe(spec, rung)
            key = (wire.body.name, scale_plane_name(wire.scale_plane),
                   wire.window_bits, wire.span)
            cheapest.setdefault(key, (spec, rung, wire))
        reps.extend(cheapest.values())

    # Not counts. A ``>= 4`` is today's roster wearing a number -- the same
    # defect one level up. The properties instead: the sweep is non-empty (an
    # emptied ``menu_families`` would otherwise pass vacuously), every family
    # the menu offers is represented (one whose ``realisable_rungs`` came back
    # empty would vanish silently), and every scale plane the menu resolves to
    # is probed, which is what "the predicate is asked about every plane" means.
    families = list(menu_families())
    assert reps
    assert {spec.name for spec, _, _ in reps} == {s.name for s in families}, reps
    menu_planes = {scale_plane_name(tessera_wire_recipe(s, r).scale_plane)
                   for s in families for r in realisable_rungs(s)}
    assert {scale_plane_name(w.scale_plane) for _, _, w in reps} == menu_planes, reps

    rows = torch.randn(64, 256)
    source = activation_source(
        {"q": hessian_from_rows(rows)},
        calibration_identity("corpus", [torch.arange(4)], fit_tokens=64))
    weight = torch.randn(64, 256)

    verdicts = {}
    for spec, rung, wire in reps:
        name = f"TESSERA_{spec.base}_K{spec.arity}_R{rung}"
        kwargs = encoder_kwargs(source, "q", 256, scale_plane=wire.scale_plane)
        predicted = rung_accepts_hessian(name, wire)
        try:
            # Around the seam deliberately: the seam's whole job is to act on
            # the predicate, so asking it would test the predicate against
            # itself.
            from prismaquant.tessera_render import _grid_for, _tessera_export
            _tessera_export.encode_linear(
                weight, grid=_grid_for(spec), q256=rung, name=name,
                verify=False, **kwargs)
            actual = True
        except GrammarError:
            actual = False
        assert predicted == actual, (name, predicted, actual)
        verdicts[name] = actual
    assert all(verdicts.values()), verdicts


def test_a_block_plane_rung_is_h_aware_too_and_the_bytes_prove_it():
    """The 4-bit route's H is applied, not excused.

    This used to assert the opposite -- that an E2M1 rung "has no H-aware
    encode, so its weights-only bytes ARE its shipping bytes", and that kwargs
    passed to one must be refused. Tessera deleted the two CHANNEL-only guards
    that made it true on 2026-09-02, and measured what the block was costing:
    served KL 0.5310 H-aware against 0.6404 weights-only at identical bytes
    (``tessera-ldlq-lut-plane-served-2026-09-02.md``). PrismaQuant's own export
    lane hands the encode to Tessera's exporter, which applies the H on every
    plane, so the old behaviour priced bytes the export does not write.

    The check is on the bytes, not on the absence of a raise: an H that changes
    nothing was dropped somewhere between the source and ``encode_linear``,
    which is the failure this seam exists to make impossible.
    """
    from prismaquant.tessera_formats import (
        parse_tessera_format_name, tessera_wire_recipe,
    )
    from prismaquant.tessera_hessian import (
        activation_source, calibration_identity, encoder_kwargs, hessian_from_rows,
    )
    from prismaquant.tessera_render import (
        encode_tessera_unit, rung_accepts_hessian,
    )

    torch.manual_seed(0)
    fmt = "TESSERA_E2M1_K2_R512"
    weight = torch.randn(64, 256)
    wire = tessera_wire_recipe(*parse_tessera_format_name(fmt))
    assert scale_plane_is(wire, "lut16"), wire
    assert rung_accepts_hessian(fmt, wire), "the LUT plane admits LDLQ and a refit"

    source = activation_source(
        {"q": hessian_from_rows(torch.randn(64, 256))},
        calibration_identity("corpus", [torch.arange(4)], fit_tokens=64))
    kwargs = encoder_kwargs(source, "q", 256, scale_plane=wire.scale_plane)

    h_aware, h_blob = encode_tessera_unit(
        weight, fmt, activation_kwargs=kwargs, recipe=wire)
    plain, plain_blob = encode_tessera_unit(
        weight, fmt, hessian_required=False, recipe=wire)
    assert h_aware.shape == weight.shape and len(h_blob) > 0
    assert len(h_blob) == len(plain_blob), (
        "the H must change the bytes, not the byte count")
    assert not torch.equal(h_aware, plain), (
        "an H that changes nothing was dropped before encode_linear")


def scale_plane_is(recipe, name: str) -> bool:
    from prismaquant.tessera_formats import scale_plane_name

    return scale_plane_name(recipe.scale_plane) == name


def test_encoding_without_a_hessian_is_refused_not_silently_downgraded():
    from prismaquant.tessera_render import (
        HessianContractError, encode_tessera_unit,
    )

    w = torch.randn(64, 256, dtype=torch.bfloat16)
    with pytest.raises(HessianContractError, match="missing input"):
        encode_tessera_unit(w, "TESSERA_E4M3_K1_R1024", activation_kwargs=None)


def test_h_aware_kwargs_under_a_weights_only_encode_are_refused():
    """The other direction, and it is just as silent.

    Encoding H-aware bytes while stamping ``supplied=false`` puts an
    unreproducible artifact behind a reproducible label. Neither direction is
    allowed to be a quiet default.
    """
    from prismaquant.tessera_formats import (
        parse_tessera_format_name, tessera_wire_recipe,
    )
    from prismaquant.tessera_hessian import (
        activation_source, calibration_identity, encoder_kwargs, hessian_from_rows,
    )
    from prismaquant.tessera_render import (
        HessianContractError, encode_tessera_unit,
    )

    rows = torch.randn(64, 256)
    source = activation_source(
        {"q": hessian_from_rows(rows)},
        calibration_identity("corpus", [torch.arange(4)], fit_tokens=64))
    wire = tessera_wire_recipe(
        *parse_tessera_format_name("TESSERA_E4M3_K1_R1024"))
    kwargs = encoder_kwargs(source, "q", 256, scale_plane=wire.scale_plane)
    with pytest.raises(HessianContractError, match="applied and unrecorded"):
        encode_tessera_unit(
            torch.randn(64, 256), "TESSERA_E4M3_K1_R1024",
            activation_kwargs=kwargs, hessian_required=False, recipe=wire)


@CUDA
def test_weights_only_is_reachable_but_only_deliberately():
    from prismaquant.tessera_render import encode_tessera_unit

    w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    render, blob = encode_tessera_unit(
        w, "TESSERA_E2M1_K2_R512", hessian_required=False)
    assert render.shape == w.shape and len(blob) > 0


def test_the_seam_forwards_the_activation_source_and_nothing_else():
    """What reaches ``encode_linear`` is exactly what ``for_unit`` produced.

    Captured on the real encoder call rather than a simulated one: the pin now
    takes an H, so this is the pass-through itself and not a rehearsal of it.
    A keyword the source did not emit is refused rather than forwarded --
    otherwise this seam becomes a second place where encode settings are
    chosen, which is the drift ``ActivationSource`` exists to prevent.
    """
    import prismaquant.tessera_render as tr
    from prismaquant.tessera_formats import (
        parse_tessera_format_name, tessera_wire_recipe,
    )
    from prismaquant.tessera_hessian import (
        activation_source, calibration_identity, encoder_kwargs, hessian_from_rows,
    )

    rows = torch.randn(64, 256)
    source = activation_source(
        {"q": hessian_from_rows(rows)},
        calibration_identity("corpus", [torch.arange(4)], fit_tokens=64))
    wire = tessera_wire_recipe(
        *parse_tessera_format_name("TESSERA_E4M3_K1_R1024"))
    kwargs = encoder_kwargs(source, "q", 256, scale_plane=wire.scale_plane)

    seen = {}
    real = tr._tessera_export.encode_linear

    def capturing(weight, **kw):
        seen.update(kw)
        return real(weight, **kw)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(tr._tessera_export, "encode_linear", capturing)
        tr.encode_tessera_unit(
            torch.randn(64, 256, dtype=torch.bfloat16),
            "TESSERA_E4M3_K1_R1024", activation_kwargs=kwargs)
    finally:
        monkey.undo()
    assert seen["q256"] == 1024
    assert torch.equal(seen["ldl"], kwargs["ldl"])
    assert torch.equal(seen["refit_metric"], kwargs["refit_metric"])
    assert seen["ldl_block"] == kwargs["ldl_block"]
    assert seen["refit_reach_floor"] == kwargs["refit_reach_floor"]

    from prismaquant.tessera_render import HessianContractError
    with pytest.raises(HessianContractError, match="not keywords"):
        tr.encode_tessera_unit(
            torch.randn(64, 256), "TESSERA_E4M3_K1_R1024",
            activation_kwargs={**kwargs, "scale_refit": 9})


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
            format_name="TESSERA_E4M3_K1_R1024",
            cache=None, wire_dir=pathlib.Path("."),
            activation_kwargs_for=lambda name, plane: {},
            hessian_required=True,
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
    from prismaquant.tessera_hessian import token_ids_sha256

    a = [torch.arange(0, 16).reshape(1, 16)]
    b = [torch.arange(0, 16).reshape(1, 16)]
    c = [torch.arange(1, 17).reshape(1, 16)]
    assert token_ids_sha256(a) == token_ids_sha256(b)
    assert token_ids_sha256(a) != token_ids_sha256(c)


# ---------------------------------------------------------------------------
# The two Hessian sources: the campaign, and the production render
# ---------------------------------------------------------------------------

def test_the_campaign_and_the_production_render_form_one_hessian():
    """Both paths call ``hessian_from_rows``, and it is bit-identical.

    Two paths hand Tessera an H -- the anchor campaign and
    ``render_tessera_production`` under ``ProductionWeightCache``. If they
    formed it differently (a dtype, an accumulation order, a normalisation)
    the campaign would price bytes the cache does not render, and principle 8
    would be broken with nothing raising. There is one formation, so this
    asserts the thing that makes that true rather than the two call sites.
    """
    from prismaquant.tessera_hessian import hessian_from_rows

    rows = torch.randn(37, 64, dtype=torch.bfloat16)
    # The campaign captures [tokens, in]; the render cache may hand back
    # [batch, seq, in]. One matrix, or they are two draws.
    flat = hessian_from_rows(rows)
    shaped = hessian_from_rows(rows.reshape(1, 37, 64))
    assert torch.equal(flat, shaped)
    assert flat.dtype is torch.float32, "bf16 loses the small eigenvalues LDLQ reads"
    reference = rows.to(torch.float32)
    assert torch.equal(flat, reference.t() @ reference)


def test_one_identity_function_answers_for_both_paths():
    """Same draw -> same triple; a different draw -> a different triple.

    The second half is what stops the guard being vacuous: an identity that
    never changes compares equal to everything, which is exactly how the merge
    guard this project already paid for went 8/13 vacuous.
    """
    from prismaquant.tessera_hessian import (
        HESSIAN_IDENTITY_FIELDS, calibration_identity,
    )

    batches = [torch.arange(0, 16).reshape(1, 16)]
    other = [torch.arange(1, 17).reshape(1, 16)]
    a = calibration_identity("corpus", batches, fit_tokens=16)
    b = calibration_identity("corpus", batches, fit_tokens=16)
    assert all(f in a for f in HESSIAN_IDENTITY_FIELDS), sorted(a)
    assert a == b
    assert calibration_identity("corpus", other, fit_tokens=16) != a
    assert calibration_identity("other", batches, fit_tokens=16) != a
    assert calibration_identity("corpus", batches, fit_tokens=32) != a


def test_a_production_render_refuses_rows_with_no_named_draw():
    """Bytes shaped by an H are not reproducible from the weights.

    So the capture has to be named, and the render refuses rather than
    encoding H-aware bytes whose provenance nothing can state. Tessera's own
    ``ActivationSource`` refuses the same way one level down; this refusal is
    earlier and names the lever.
    """
    from prismaquant.tessera_render import (
        HessianContractError, render_tessera_production,
    )

    weight = torch.randn(32, 64)
    rows = torch.randn(48, 64)
    with pytest.raises(HessianContractError, match="tessera_hessian_identity"):
        render_tessera_production(
            weight, "TESSERA_E4M3_K1_R1024", qname="m.up",
            activations={"m.up": rows}, levers={})


def test_the_production_render_is_h_aware_and_the_lever_is_not(tmp_path):
    """The lever is a different encode, not a differently-labelled one.

    ``tessera_weights_only`` must produce the byte-identical weights-only
    encode, and the default must not: if the two agreed, the H was silently
    dropped somewhere between the cache and ``encode_linear``, which is the
    failure this whole seam exists to make impossible.
    """
    from prismaquant.tessera_hessian import calibration_identity
    from prismaquant.tessera_render import (
        encode_tessera_unit, render_tessera_production,
    )

    torch.manual_seed(0)
    weight = torch.randn(32, 256)
    rows = torch.randn(64, 256)
    identity = calibration_identity("corpus", [torch.arange(8)], fit_tokens=64)
    fmt = "TESSERA_E4M3_K1_R1024"

    h_aware = render_tessera_production(
        weight, fmt, qname="m.up", activations={"m.up": rows},
        levers={"tessera_hessian_identity": identity})
    weights_only = render_tessera_production(
        weight, fmt, qname="m.up", activations=None,
        levers={"tessera_weights_only": True})
    direct, _blob = encode_tessera_unit(weight, fmt, hessian_required=False)

    assert torch.equal(weights_only, direct), (
        "the lever must be the plain encode, byte for byte")
    assert not torch.equal(h_aware, weights_only), (
        "an H that changes nothing was dropped between the cache and the encoder")


def test_activation_kwargs_are_rung_independent_but_plane_dependent():
    """One block-LDL per unit PER PLANE, not one per rate and not one per unit.

    The campaign memoises these across a surface's anchors, and the memo key is
    what this pins. Two halves:

    * **Rung-independent, still.** No rate reaches the call and the signature
      has nowhere to put one, so a rate cannot leak into the memo and price
      every anchor at the first one's rate.
    * **Plane-dependent, newly.** Tessera keys the refit objective by scale
      plane -- ``hessian`` on a CHANNEL row scale, ``h^1.0`` on the LUT plane's
      coupled blocks -- because the two answers were measured separately and
      disagree. A memo keyed by unit alone would price a unit's second family
      under the first family's objective, silently. The expensive half (the
      block-LDL) is genuinely shared, which is what keeps the hoist worth
      having: the bound is one factorisation per unit per plane.
    """
    import inspect

    from tessera.manifest import ScalePlaneKind

    from prismaquant.tessera_hessian import (
        activation_source, calibration_identity, encoder_kwargs, hessian_from_rows,
    )

    rows = torch.randn(64, 256)
    identity = calibration_identity("corpus", [torch.arange(8)], fit_tokens=64)
    source = activation_source({"m.up": hessian_from_rows(rows)}, identity)

    plane = ScalePlaneKind.CHANNEL
    first = encoder_kwargs(source, "m.up", 256, scale_plane=plane)
    second = encoder_kwargs(source, "m.up", 256, scale_plane=plane)
    assert sorted(first) == ["ldl", "ldl_block", "refit_metric", "refit_reach_floor"]
    for key in ("ldl", "refit_metric"):
        assert torch.equal(first[key], second[key])

    other = encoder_kwargs(source, "m.up", 256, scale_plane=ScalePlaneKind.LUT)
    assert torch.equal(first["ldl"], other["ldl"]), (
        "the block-LDL is a function of the Hessian alone; if it moved with the "
        "plane the hoist would be worthless")
    assert first["refit_metric"].shape != other["refit_metric"].shape or \
        not torch.equal(first["refit_metric"], other["refit_metric"]), (
        "the refit metric must move with the plane -- CHANNEL takes the exact "
        "quadratic, LUT a diagonal power -- or the memo may drop the plane")

    # No rate reaches this call at all -- the signature has nowhere to put one.
    assert "q256" not in inspect.signature(encoder_kwargs).parameters
    # And the plane is required, with no default: a caller that omits it would
    # otherwise be served one plane's objective for another plane's wire.
    plane_param = inspect.signature(encoder_kwargs).parameters["scale_plane"]
    assert plane_param.default is inspect.Parameter.empty
    assert plane_param.kind is inspect.Parameter.KEYWORD_ONLY


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
            w, "TESSERA_E4M3_K1_R1024",
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
            w, "TESSERA_E4M3_K1_R1024",
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


def test_the_weights_only_lever_forms_no_hessian_at_all():
    """``tessera_weights_only`` means weights-only, on a pin that takes an H.

    Forming H and letting the seam drop it was correct exactly until the day
    the encoder could consume one -- at which point a lever named "weights
    only" would start shipping H-aware bytes under a ``supplied=false`` stamp.
    The pin now takes one, so this is the real check: nothing the
    ``ActivationSource`` emits may reach ``encode_linear`` under the lever, and
    everything it emits must reach it without.
    """
    import prismaquant.tessera_render as tr
    from prismaquant.tessera_hessian import calibration_identity

    seen: dict = {}
    real = tr._tessera_export.encode_linear

    def capturing(weight, **kw):
        seen.clear()
        seen.update(kw)
        return real(weight, **kw)

    w = torch.randn(64, 256, dtype=torch.bfloat16)
    acts = {"q": torch.randn(64, 256)}
    identity = calibration_identity("corpus", [torch.arange(4)], fit_tokens=64)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(tr._tessera_export, "encode_linear", capturing)
        tr.render_tessera_production(
            w, "TESSERA_E4M3_K1_R1024", qname="q", activations=acts,
            levers={"tessera_weights_only": True})
        assert "ldl" not in seen and "refit_metric" not in seen, sorted(seen)
        tr.render_tessera_production(
            w, "TESSERA_E4M3_K1_R1024", qname="q", activations=acts,
            levers={"tessera_hessian_identity": identity})
        assert "ldl" in seen and "refit_metric" in seen, sorted(seen)
    finally:
        monkey.undo()


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


def test_the_aura_dw_fallback_refuses_tessera():
    """``aura_cost``'s ``dW`` fallback is the one whose guard is off by default.

    ``require_production_cache`` defaults to False, AURA is the default
    ``COST_MODE``, and the fallback ends at the registry
    ``quantize_dequantize``. The orchestrator refuses the ``TESSERA`` token
    before this stage (``run-pipeline.sh:69``), but a direct call does not, and
    "unreachable through one entry point" is not the same as closed.
    """
    from prismaquant.aura_cost import _delta_w

    w = torch.randn(8, 32, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Tessera"):
        _delta_w("lin", "TESSERA_E2M1_K2_R256", w, None)


def test_the_perturbed_x_cache_fallback_refuses_tessera():
    """The third fallback, pinned like the other two.

    `_apply_weight_quant` is reached with no production cache, so `q` stays
    None and the registry render is the next thing it would do.
    """
    from prismaquant import format_registry as fr
    from prismaquant.perturbed_x_cache import (
        PerturbedActivationCache, _ModulePlan, _ParamPlan,
    )

    lin = torch.nn.Linear(32, 8, bias=False)
    plan = _ModulePlan(module=lin, params=[_ParamPlan(
        name="lin", attr="weight",
        spec=fr.get_format("TESSERA_E2M1_K2_R256"))])
    cache = PerturbedActivationCache.__new__(PerturbedActivationCache)
    cache._materialized_frozen_weight_depth = 0
    cache._frozen_weight_cache = None
    cache._production_weight_cache = None
    with pytest.raises(RuntimeError, match="Tessera"):
        cache._apply_weight_quant(plan)


def test_a_non_tessera_render_does_not_import_the_tessera_package():
    """Asking "is this mine?" must not drag in the answer's dependencies.

    `render_production_weight` and the three cache-miss fallbacks are on the
    hot path of every *non*-Tessera format too. Both `tessera_formats` and
    `tessera_render` require the `tessera` package at import, so routing the
    predicate through either would have made Tessera a hard dependency of the
    shipping NVFP4 pipeline. The sweep cannot see this — it always runs with
    Tessera on the path — so the import is blocked in a subprocess.
    """
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    script = r'''
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "tessera" or name.startswith("tessera."):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, _Block())
for m in [m for m in sys.modules if m == "tessera" or m.startswith("tessera.")]:
    del sys.modules[m]
import torch
from prismaquant.production_weight_cache import render_production_weight
w = torch.randn(64, 256, dtype=torch.bfloat16)
r = render_production_weight(w, "FP8_DYNAMIC", qname="q",
                             activations={"q": torch.randn(128, 256)},
                             levers={})
assert tuple(r.shape) == (64, 256)
from prismaquant.weight_session import WeightSession
lin = torch.nn.Linear(32, 8, bias=False).to(torch.bfloat16)
s = WeightSession(torch.nn.Sequential(lin), production_weight_cache=None,
                  strict_production_cache=False)
s._linear_by_qname["lin"] = (lin, "weight")
try:
    s._format_weight("lin", "TESSERA_E2M1_K2_R256")
    raise SystemExit("weight_session did not refuse")
except RuntimeError as e:
    assert "Tessera" in str(e), e
from prismaquant.aura_cost import _delta_w
try:
    _delta_w("lin", "TESSERA_E2M1_K2_R256", torch.randn(8, 32), None)
    raise SystemExit("aura_cost did not refuse")
except RuntimeError as e:
    assert "Tessera" in str(e), e
print("OK")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env.setdefault("TMPDIR", "/home/rob/tmp")
    out = subprocess.run([sys.executable, "-c", script], cwd=str(repo),
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "OK" in out.stdout
