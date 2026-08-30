"""Gate for the continuous trellis rate surface (research-only).

The module turns a handful of measured anchors into a dense, exactly-priced
candidate menu.  The things that can go wrong are all failures of honesty
rather than of arithmetic: a rung silently extrapolated beyond what was
measured, an interpolated value reported as measured, two objectives mixed in
one DP, or a non-monotone anchor set laundered into a smooth-looking cost.
Each of those has a test here.
"""

from __future__ import annotations

import math
import statistics

import pytest

from prismaquant.trellis_allocator import build_trellis_allocator_candidate
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    E4M3_FAMILY,
    get_trellis_family,
    native_code_value,
    LAYOUT_TIGHT_OFFSETS,
    TrellisFormatError,
    validate_schedule,
)
from prismaquant.trellis_rate_surface import (
    CONVERTED_LABEL_SUFFIX,
    PROVENANCE_INTERPOLATED,
    PROVENANCE_MEASURED,
    CurrencyConversionStamp,
    TrellisKappaEvidence,
    TrellisRateSurface,
    allocation_regret,
    convert_surface_currency,
    densify_rate_surface,
    fit_rate_surface,
    leave_one_anchor_out,
    rate_surface_solver_menu,
    uniform_column_schedule,
)

UNIT = "model.layers.0.mlp.down_proj"
SHAPE = (512, 512)
CURRENCY = "aura_adjoint"


def alphabet(rate: int) -> tuple[int, ...]:
    """A distinct, NaN-free E4M3 alphabet, sorted the way the wire demands.

    The wire requires the alphabet ordered by DECODED VALUE then code, which
    is not byte order: 0x80 is -0.0 and sorts below every positive code.
    """

    spec = get_trellis_family(E4M3_FAMILY)
    required = 1 << (rate + 1)
    codes = [code for code in range(256) if code not in (0x7F, 0xFF)]
    if required > len(codes):
        # Rate 7 wants 256 slots from a 254-value finite grid.  The wire
        # allows exactly this corner by spending both zero byte patterns.
        codes.extend((0x00, 0x80))
        required = len(codes)
    codes.sort(key=lambda code: (native_code_value(spec, code), code))
    # Take a contiguous run around zero so the subset stays value-ordered.
    start = (len(codes) - required) // 2
    return tuple(codes[start:start + required])


def alphabets_for(schedule) -> dict[int, tuple[int, ...]]:
    return {rate: alphabet(rate) for rate in sorted(set(schedule)) if rate < 8}


def anchor(rate_q256: int, dloss: float, stderr: float = 0.0, unit: str = UNIT):
    schedule = uniform_column_schedule(
        SHAPE[1], rate_q256, family=E4M3_FAMILY,
    )
    return build_trellis_allocator_candidate(
        unit,
        SHAPE,
        family=E4M3_FAMILY,
        body_rate_q256=rate_q256,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets=alphabets_for(schedule),
        predicted_dloss=dloss,
        predicted_dloss_stderr=stderr,
    )


def surface(anchors=((1024, 1.0), (1280, 0.25), (1536, 0.0625))):
    return fit_rate_surface(
        [anchor(rate, dloss) for rate, dloss in anchors],
        currency=CURRENCY,
    )


# --------------------------------------------------------------------------
# The schedule constructor: the rate is interpolated, the BYTES never are.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate_q256", [1024, 1088, 1152, 1280, 1536, 1792])
def test_uniform_schedule_is_wire_legal_and_hits_the_rate(rate_q256):
    schedule = uniform_column_schedule(512, rate_q256, family=E4M3_FAMILY)
    # The wire itself is the authority on legality, not this test's opinion.
    validate_schedule(
        E4M3_FAMILY, rate_q256, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    )
    assert len(schedule) == 512
    # tight_offsets tolerates less than one physical body bit of drift.
    assert abs(sum(schedule) - rate_q256 * 512 / 256) < 1.0


# E2M1 is where the low-bpp mandate lives, and it is the harder family for
# this constructor: bypass_rate is 4, so a block packed with rate-4 columns
# has zero coded steps and the wire refuses it.  These rates are inside the
# research range [256, 1016], not an unused corner.
@pytest.mark.parametrize("rate_q256", [384, 512, 640, 768, 896, 1008])
def test_uniform_schedule_is_wire_legal_on_e2m1(rate_q256):
    schedule = uniform_column_schedule(512, rate_q256, family=E2M1_FAMILY)
    validate_schedule(
        E2M1_FAMILY, rate_q256, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    )
    assert abs(sum(schedule) - rate_q256 * 512 / 256) < 1.0
    # Every superblock keeps enough coded steps, which is only true because
    # the remainder is spread rather than packed at the front.
    for start in range(0, 512, 256):
        block = schedule[start:start + 256]
        assert sum(value < 4 for value in block) >= 8


def test_remainder_is_spread_not_packed():
    # Packing the dearer columns contiguously would give block 0 all of them.
    schedule = uniform_column_schedule(512, 896, family=E2M1_FAMILY)
    first = sum(schedule[:256])
    second = sum(schedule[256:])
    assert abs(first - second) <= 1


def test_uniform_schedule_is_flat_within_one_bit():
    schedule = uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
    assert max(schedule) - min(schedule) <= 1
    assert set(schedule) == {4, 5}


def test_uniform_schedule_refuses_a_partial_block():
    # A short final block is legal on the wire, but its rate accounting is
    # the caller's to declare; guessing it here would be a silent choice.
    with pytest.raises(TrellisFormatError, match="multiple of 256"):
        uniform_column_schedule(500, 1152, family=E4M3_FAMILY)


def test_uniform_schedule_refuses_rates_needing_illegal_columns():
    with pytest.raises(TrellisFormatError):
        uniform_column_schedule(512, 200, family=E4M3_FAMILY)


# --------------------------------------------------------------------------
# The surface contract.
# --------------------------------------------------------------------------


def test_one_anchor_is_a_point_not_a_surface():
    with pytest.raises(TrellisFormatError, match="at least two"):
        fit_rate_surface([anchor(1024, 1.0)], currency=CURRENCY)


def test_non_monotone_anchors_are_refused_not_smoothed():
    # A rate that costs MORE loss than a cheaper one is a measurement
    # problem.  Interpolating through it would launder it into a cost.
    with pytest.raises(TrellisFormatError, match="strictly decrease"):
        fit_rate_surface(
            [anchor(1024, 0.25), anchor(1280, 1.0)], currency=CURRENCY,
        )


def test_anchors_may_not_repeat_a_rate():
    with pytest.raises(TrellisFormatError, match="repeat a rate"):
        fit_rate_surface(
            [anchor(1024, 1.0), anchor(1024, 0.5)], currency=CURRENCY,
        )


def test_currency_must_be_declared():
    with pytest.raises(TrellisFormatError, match="currency"):
        fit_rate_surface(
            [anchor(1024, 1.0), anchor(1280, 0.25)], currency="",
        )


def test_predict_reproduces_anchors_exactly():
    rd = surface()
    for rate, dloss in ((1024, 1.0), (1280, 0.25), (1536, 0.0625)):
        assert rd.predict(rate) == pytest.approx(dloss, rel=0, abs=0)
        assert rd.provenance(rate) == PROVENANCE_MEASURED


def test_predict_interpolates_in_log_space():
    rd = surface()
    # Midway between 1.0 and 0.25 in log2 is the geometric mean, 0.5 --
    # not the arithmetic mean 0.625.  Distortion decays geometrically with
    # rate, so a linear-in-D interpolation would be biased high everywhere.
    assert rd.predict(1152) == pytest.approx(0.5, rel=1e-12)
    assert rd.provenance(1152) == PROVENANCE_INTERPOLATED


def test_extrapolation_is_refused_never_guessed():
    rd = surface()
    for outside in (1023, 1537, 2040):
        with pytest.raises(TrellisFormatError, match="outside the measured"):
            rd.predict(outside)


def test_interpolated_stderr_never_shrinks_by_interpolating():
    rd = fit_rate_surface(
        [anchor(1024, 1.0, 0.01), anchor(1280, 0.25, 0.4)],
        currency=CURRENCY,
    )
    assert rd.predict_stderr(1024) == pytest.approx(0.01)
    assert rd.predict_stderr(1280) == pytest.approx(0.4)
    # Between them it inherits the WIDER bracketing anchor, not the nearer.
    assert rd.predict_stderr(1088) == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Densification: exact bytes, honest labels.
# --------------------------------------------------------------------------


def test_densify_prices_every_rung_with_real_bytes():
    rd = surface()
    rungs = (1024, 1088, 1152, 1216, 1280)
    schedule = uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=rungs,
        alphabets=alphabets_for(schedule),
    )
    assert len(built) == len(rungs)
    assert [record.body_rate_q256 for record in built] == list(rungs)
    # Bytes are exact and strictly increase with rate: nothing about the
    # footprint is interpolated, only the loss.
    footprints = [record.footprint["total_bytes"] for record in built]
    assert footprints == sorted(footprints)
    assert len(set(footprints)) == len(footprints)


def test_densify_labels_interpolated_rungs_as_such():
    rd = surface()
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1152, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
    )
    labels = {
        record.body_rate_q256: record.variant_label for record in built
    }
    assert labels[1024] == PROVENANCE_MEASURED
    assert labels[1280] == PROVENANCE_MEASURED
    assert labels[1152] == PROVENANCE_INTERPOLATED


def test_menu_refuses_to_mix_objectives():
    rd = surface()
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
    )
    with pytest.raises(TrellisFormatError, match="one currency"):
        rate_surface_solver_menu(
            {"a": built, "b": built},
            currencies={"a": "aura_adjoint", "b": "output_mse"},
        )


def test_menu_builds_for_one_declared_currency():
    rd = surface()
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
    )
    menu = rate_surface_solver_menu(
        {UNIT: built}, currencies={UNIT: CURRENCY},
    )
    assert set(menu) == {UNIT}
    assert len(menu[UNIT]) == 2


# --------------------------------------------------------------------------
# Diagnostic vs gate.
# --------------------------------------------------------------------------


def test_loo_is_near_zero_on_a_log_linear_surface():
    # dloss halves every 128 q256: interpolation is exact by construction,
    # so any residual here is the implementation's own error.
    anchors = tuple(
        (1024 + 128 * step, 2.0 ** (-step)) for step in range(5)
    )
    report = leave_one_anchor_out(surface(anchors))
    assert report["interior_anchors"] == 3
    assert report["max_abs_log2_error"] < 1e-12


def test_loo_exposes_saturation_the_way_the_ladder_shows_it():
    # The real E4M3 ladder saturates against the scalar ceiling: ~5.4 dB per
    # bit, then 2.22 for the last step.  A surface shaped like that must
    # report a real interpolation error, not hide it.
    anchors = (
        (1024, 1.0),
        (1280, 0.29),
        (1536, 0.085),
        (1792, 0.051),
    )
    report = leave_one_anchor_out(surface(anchors))
    assert report["max_abs_log2_error"] > 0.1


def test_regret_is_zero_when_the_surface_is_exact():
    rates = (1024, 1280, 1536)
    truth = {
        "u0": {1024: 1.0, 1280: 0.25, 1536: 0.0625},
        "u1": {1024: 2.0, 1280: 0.60, 1536: 0.2000},
    }
    surfaces = {
        name: TrellisRateSurface(
            unit_name=name,
            family=E4M3_FAMILY,
            layout=LAYOUT_TIGHT_OFFSETS,
            currency=CURRENCY,
            anchor_q256=rates,
            anchor_dloss=tuple(truth[name][rate] for rate in rates),
            anchor_stderr=(0.0, 0.0, 0.0),
        )
        for name in truth
    }
    unit_bytes = {
        name: {1024: 1000, 1280: 1250, 1536: 1500} for name in truth
    }
    report = allocation_regret(
        surfaces, truth, unit_bytes=unit_bytes, byte_budget=2600,
    )
    assert report["regret_pct"] == pytest.approx(0.0)
    assert report["assignment_agreement"] == 1.0


def test_regret_is_reported_when_interpolation_misranks():
    # u1's true curve has a kink the two anchors cannot see, so the surface
    # overstates the value of its middle rung and the allocator misspends.
    rates = (1024, 1536)
    truth = {
        "u0": {1024: 1.0, 1280: 0.90, 1536: 0.80},
        "u1": {1024: 1.0, 1280: 0.99, 1536: 0.10},
    }
    surfaces = {
        name: TrellisRateSurface(
            unit_name=name,
            family=E4M3_FAMILY,
            layout=LAYOUT_TIGHT_OFFSETS,
            currency=CURRENCY,
            anchor_q256=rates,
            anchor_dloss=tuple(truth[name][rate] for rate in rates),
            anchor_stderr=(0.0, 0.0),
        )
        for name in truth
    }
    unit_bytes = {
        name: {1024: 1000, 1280: 1250, 1536: 1500} for name in truth
    }
    report = allocation_regret(
        surfaces, truth, unit_bytes=unit_bytes, byte_budget=2500,
    )
    # The point is that regret is REPORTED and finite, not that it is large:
    # the one-anchor campaign's lesson is that a noisy surrogate can still
    # make near-omniscient decisions.
    assert report["regret_pct"] >= 0.0
    assert 0.0 <= report["assignment_agreement"] <= 1.0
    assert math.isfinite(report["true_loss_deciding_on_interpolated"])


# --------------------------------------------------------------------------
# Currency conversion: measured per-unit kappa, honest uncertainty.
# --------------------------------------------------------------------------

SSE_CURRENCY = "trellis_weighted_sse"
MEASUREMENT_SOURCE = (
    "sparklina:/home/rob/dq-runs/trellis-currency-20260829/"
    "currency_gap_qwen3-0.6b_analysis.json"
)
MEASUREMENT_SCHEMA = "prismaquant.trellis.currency_gap.analysis.v1"


def sse_surface(anchors=((1024, 1.0, 0.05), (1280, 0.25, 0.02),
                         (1536, 0.0625, 0.004)), unit=UNIT):
    return fit_rate_surface(
        [anchor(rate, dloss, stderr, unit=unit)
         for rate, dloss, stderr in anchors],
        currency=SSE_CURRENCY,
    )


def evidence(kappa=(2.0e-4, 8.0e-4, 4.0e-4), rates=(1024, 1280, 1536),
             unit=UNIT, source=SSE_CURRENCY, target=CURRENCY):
    return TrellisKappaEvidence(
        unit_name=unit,
        source_currency=source,
        target_currency=target,
        rate_q256=tuple(rates),
        kappa=tuple(kappa),
        measurement_source=MEASUREMENT_SOURCE,
        measurement_schema=MEASUREMENT_SCHEMA,
    )


def test_conversion_applies_the_measured_geomean_kappa():
    # The campaign's per-Linear scalar is the GEOMETRIC mean over rates:
    # float(np.exp(np.mean(np.log(kap)))).  (2e-4 * 8e-4 * 4e-4)^(1/3)
    # is exactly 4e-4; the arithmetic mean would be ~4.67e-4.
    rd = convert_surface_currency(sse_surface(), evidence())
    assert rd.currency == CURRENCY
    assert rd.unit_name == UNIT
    for rate, dloss in ((1024, 1.0), (1280, 0.25), (1536, 0.0625)):
        assert rd.predict(rate) == pytest.approx(4.0e-4 * dloss, rel=1e-12)
    # Interpolated rungs scale identically: log-space interpolation
    # commutes with a positive scalar.
    assert rd.predict(1152) == pytest.approx(4.0e-4 * 0.5, rel=1e-12)
    # Provenance of the interpolation axis is unchanged by conversion.
    assert rd.provenance(1024) == PROVENANCE_MEASURED
    assert rd.provenance(1152) == PROVENANCE_INTERPOLATED


def test_conversion_stamp_is_structured_and_traceable():
    rd = convert_surface_currency(sse_surface(), evidence())
    stamp = rd.conversion
    assert isinstance(stamp, CurrencyConversionStamp)
    assert stamp.unit_name == UNIT
    assert stamp.source_currency == SSE_CURRENCY
    assert stamp.target_currency == CURRENCY
    assert stamp.kappa == pytest.approx(4.0e-4, rel=1e-12)
    assert stamp.kappa_spread_maxmin == pytest.approx(4.0, rel=1e-12)
    assert stamp.evidence_rate_q256 == (1024, 1280, 1536)
    assert stamp.evidence_kappa == (2.0e-4, 8.0e-4, 4.0e-4)
    assert stamp.measurement_source == MEASUREMENT_SOURCE
    assert stamp.measurement_schema == MEASUREMENT_SCHEMA
    # A gate reads a dict, never prose.
    body = stamp.as_dict()
    assert body["source_currency"] == SSE_CURRENCY
    assert body["target_currency"] == CURRENCY
    assert body["kappa_predictive_log_sd"] > 0.0
    # An unconverted surface carries no stamp.
    assert sse_surface().conversion is None


def test_zero_spread_evidence_scales_stderr_without_widening():
    # All per-rate kappas equal: the conversion introduces no additional
    # uncertainty, so relative stderr is preserved exactly.
    rd = convert_surface_currency(
        sse_surface(), evidence(kappa=(3.0e-4, 3.0e-4, 3.0e-4)),
    )
    assert rd.predict_stderr(1024) == pytest.approx(3.0e-4 * 0.05, rel=1e-12)
    assert rd.predict_stderr(1536) == pytest.approx(3.0e-4 * 0.004, rel=1e-12)


def test_kappa_spread_widens_stderr_by_the_predictive_log_sd():
    # stderr' = kappa * dloss * sqrt((stderr/dloss)^2 + s^2 * (1 + 1/m)),
    # with s the ddof-1 sd of log kappa: the scaled measurement error and
    # the conversion's own predictive uncertainty, in quadrature.  Computed
    # independently here from the raw evidence.
    kappas = (2.0e-4, 8.0e-4, 4.0e-4)
    ev = evidence(kappa=kappas)
    rd = convert_surface_currency(sse_surface(), ev)
    kappa = math.exp(math.fsum(math.log(k) for k in kappas) / len(kappas))
    sigma = statistics.stdev(
        [math.log(k) for k in kappas]
    ) * math.sqrt(1.0 + 1.0 / len(kappas))
    for rate, dloss, stderr in ((1024, 1.0, 0.05), (1280, 0.25, 0.02),
                                (1536, 0.0625, 0.004)):
        expected = kappa * dloss * math.sqrt(
            (stderr / dloss) ** 2 + sigma ** 2
        )
        assert rd.predict_stderr(rate) == pytest.approx(expected, rel=1e-12)
    # A wide-spread unit comes out with visibly wider RELATIVE error bars
    # than a tight one: the 35x-spread outlier must not look as trustworthy
    # as a 1.6x-median unit.
    tight = convert_surface_currency(
        sse_surface(), evidence(kappa=(3.9e-4, 4.0e-4, 4.1e-4)),
    )
    rel_wide = rd.predict_stderr(1024) / rd.predict(1024)
    rel_tight = tight.predict_stderr(1024) / tight.predict(1024)
    assert rel_wide > 2.0 * rel_tight


def test_conversion_refuses_another_units_evidence():
    # The global-scalar arm was REFUSED by measurement (an order of
    # magnitude worse allocation regret than per-Linear kappa); reusing one
    # unit's kappa for another is that arm.
    other = evidence(unit="model.layers.1.mlp.down_proj")
    with pytest.raises(TrellisFormatError, match="per-unit only"):
        convert_surface_currency(sse_surface(), other)


def test_a_bare_scalar_is_not_evidence():
    # One number cannot carry the per-rate spread, so it cannot price the
    # conversion's own uncertainty -- and the API refuses it outright.
    with pytest.raises(TrellisFormatError, match="two or more rates"):
        evidence(kappa=(4.0e-4,), rates=(1024,))


def test_evidence_ratios_must_be_positive_and_finite():
    for bad in (0.0, -1.0e-4, math.nan, math.inf):
        with pytest.raises(TrellisFormatError, match="positive and finite"):
            evidence(kappa=(2.0e-4, bad, 4.0e-4))


def test_conversion_refuses_a_bridge_the_evidence_did_not_measure():
    # Evidence measuring output_mse -> aura cannot convert an SSE surface.
    mismatched = evidence(source="output_mse")
    with pytest.raises(TrellisFormatError, match="do not price this bridge"):
        convert_surface_currency(sse_surface(), mismatched)


def test_identity_conversion_is_refused():
    with pytest.raises(TrellisFormatError, match="DIFFERENT objectives"):
        evidence(source=CURRENCY, target=CURRENCY)


def test_chained_conversion_is_refused():
    converted = convert_surface_currency(sse_surface(), evidence())
    onward = evidence(source=CURRENCY, target="held_out_kl")
    with pytest.raises(TrellisFormatError, match="already currency-converted"):
        convert_surface_currency(converted, onward)


def test_conversion_refuses_kappa_beyond_its_measured_envelope():
    # kappa's near-rate-independence is a fact about the rates it was
    # measured at; the surface's 1536 anchor lies beyond evidence measured
    # only on [1024, 1280], and applying the scalar there would be the same
    # extrapolation predict() refuses.
    short = evidence(kappa=(2.0e-4, 8.0e-4), rates=(1024, 1280))
    with pytest.raises(TrellisFormatError, match="measured rate envelope"):
        convert_surface_currency(sse_surface(), short)


def test_densify_marks_candidates_from_a_converted_surface():
    rd = convert_surface_currency(sse_surface(), evidence())
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1152, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
    )
    labels = {
        record.body_rate_q256: record.variant_label for record in built
    }
    assert labels[1024] == PROVENANCE_MEASURED + CONVERTED_LABEL_SUFFIX
    assert labels[1152] == PROVENANCE_INTERPOLATED + CONVERTED_LABEL_SUFFIX
    assert labels[1280] == PROVENANCE_MEASURED + CONVERTED_LABEL_SUFFIX


def test_converted_surface_joins_the_dp_menu_unconverted_does_not():
    other_unit = "model.layers.1.mlp.down_proj"
    sse = sse_surface()
    # A second unit priced natively in AURA currency.
    native = fit_rate_surface(
        [anchor(1024, 2.0e-4, unit=other_unit),
         anchor(1280, 0.5e-4, unit=other_unit)],
        currency=CURRENCY,
    )
    kwargs = dict(
        q256_values=(1024, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
    )
    native_built = densify_rate_surface(native, SHAPE, **kwargs)
    # Unconverted, the SSE unit cannot share a DP with an AURA unit ...
    sse_built = densify_rate_surface(sse, SHAPE, **kwargs)
    with pytest.raises(TrellisFormatError, match="one currency"):
        rate_surface_solver_menu(
            {UNIT: sse_built, other_unit: native_built},
            currencies={UNIT: sse.currency, other_unit: CURRENCY},
        )
    # ... and after a MEASURED conversion it genuinely can: the currency
    # string is now truthfully the DP's, not a relabelling.
    converted = convert_surface_currency(sse, evidence())
    converted_built = densify_rate_surface(converted, SHAPE, **kwargs)
    menu = rate_surface_solver_menu(
        {UNIT: converted_built, other_unit: native_built},
        currencies={UNIT: converted.currency, other_unit: CURRENCY},
    )
    assert set(menu) == {UNIT, other_unit}
