from dataclasses import FrozenInstanceError
import math

import pytest

from prismaquant.anchored_shape import (
    AnchoredShapeError,
    LogShapeObservation,
    audit_anchored,
    fit_anchor_correction,
    fit_centered_log_shape,
    predict_anchored,
)


def _panel(offsets=(0.0, 2.0)):
    features = {str(k): (float(k), float(k * k)) for k in range(7)}
    observations = [
        LogShapeObservation(str(unit), str(k), 10.0 ** (offset - 0.7*k + 0.04*k*k))
        for unit, offset in enumerate(offsets) for k in (0, 2, 4, 6)
    ]
    return fit_centered_log_shape(observations, features)


def test_two_anchors_recover_unit_tilt_over_curved_shared_shape():
    shape = _panel()
    coords = {str(k): float(k) for k in range(7)}
    truth = {str(k): 10.0 ** (1.3 - 0.5*k + 0.04*k*k) for k in range(7)}
    one = fit_anchor_correction(shape, {"0": truth["0"]}, coords)
    two = fit_anchor_correction(shape, {k: truth[k] for k in ("0", "6")}, coords)
    assert predict_anchored(one, "5") == pytest.approx(truth["5"] / 10)
    for key, value in truth.items():
        assert predict_anchored(two, key) == pytest.approx(value, rel=3e-14)


def test_shared_shape_is_invariant_to_each_panel_units_level():
    first, shifted = _panel(), _panel((-190.0, 230.0))
    assert shifted.coefficients == pytest.approx(first.coefficients, abs=2e-14)
    assert shifted.log_shape_by_key == pytest.approx(first.log_shape_by_key, abs=1e-13)


def test_shared_shape_rank_deficiency_and_duplicate_observations_refuse():
    rows = [LogShapeObservation("u", str(k), 2.0**-k) for k in range(3)]
    with pytest.raises(AnchoredShapeError, match="rank"):
        fit_centered_log_shape(rows, {str(k): (k, 2*k) for k in range(3)})
    with pytest.raises(AnchoredShapeError, match="duplicate"):
        fit_centered_log_shape([*rows, rows[0]], {str(k): (k,) for k in range(3)})
    with pytest.raises(AnchoredShapeError, match="fewer than two"):
        fit_centered_log_shape(rows[:1], {str(k): (k,) for k in range(3)})


@pytest.mark.parametrize("offset,step", [(1e15, 0.5), (-1e300, 1e290), (0.0, 1e-290)])
def test_correction_is_stable_under_coordinate_translation_and_scaling(offset, step):
    shape = _panel()
    coords = {str(k): offset + step*k for k in range(7)}
    anchors = {"0": 2.0, "6": 0.02}
    fit = fit_anchor_correction(shape, anchors, coords)
    baseline = fit_anchor_correction(shape, anchors, {str(k): k for k in range(7)})
    # The extreme-offset input's float coordinates themselves lose ~1e-6 of
    # their spacing. Compare against the represented coordinate geometry.
    for key in ("1", "3", "5"):
        position = (coords[key] - coords["0"]) / (coords["6"] - coords["0"])
        expected_log = (math.log10(anchors["0"]) + shape.log_shape_by_key[key]
                        + position*(math.log10(anchors["6"]/anchors["0"])
                                    - shape.log_shape_by_key["6"]))
        assert predict_anchored(fit, key) == pytest.approx(10**expected_log, rel=2e-14)
    assert predict_anchored(baseline, "0") == 2.0


def test_exact_measured_anchors_survive_overdetermined_residual_fit():
    shape = _panel()
    anchors = {"0": 0.123456789, "3": 7.1, "6": 0.0003456789}
    correction = fit_anchor_correction(shape, anchors, {str(k): k for k in range(7)})
    for key, value in anchors.items():
        assert predict_anchored(correction, key) == value


def test_audit_is_independent_immutable_and_does_not_refit():
    shape = _panel()
    coords = {str(k): k for k in range(7)}
    anchors = {"0": 2.0, "6": 0.02}
    correction = fit_anchor_correction(shape, anchors, coords)
    before = predict_anchored(correction, "3")
    audit = audit_anchored(correction, {"3": before*100})
    assert audit.rows[0].predicted == before
    assert audit.rows[0].measured == before*100
    assert audit.max_absolute_log10_error == pytest.approx(2.0)
    assert predict_anchored(correction, "3") == before
    refit = fit_anchor_correction(shape, {**anchors, "3": before*100}, coords)
    assert predict_anchored(refit, "3") == before*100
    assert audit.rows[0].predicted == before
    with pytest.raises(AnchoredShapeError, match="anchor"):
        audit_anchored(correction, {"0": anchors["0"]})
    with pytest.raises(FrozenInstanceError):
        audit.rows[0].predicted = 1.0
    anchors["0"] = 4.0
    coords["3"] = 999.0
    assert predict_anchored(correction, "0") == 2.0
    assert predict_anchored(correction, "3") == before
    with pytest.raises(TypeError):
        correction.anchors["0"] = 1.0


def test_extreme_costs_use_logs_without_overflowing_ratios():
    features = {"a": (0.0,), "b": (1.0,), "c": (2.0,)}
    shape = fit_centered_log_shape([
        LogShapeObservation("u", "a", 1e-300),
        LogShapeObservation("u", "c", 1e300),
    ], features)
    correction = fit_anchor_correction(shape, {"a": 1e-300, "c": 1e300},
                                       {"a": 0, "b": 1, "c": 2})
    assert predict_anchored(correction, "a") == 1e-300
    assert predict_anchored(correction, "b") == pytest.approx(1.0)
    assert predict_anchored(correction, "c") == 1e300
    assert math.isfinite(audit_anchored(correction, {"b": 1e-300}).max_absolute_log10_error)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf, True])
def test_invalid_observations_refuse(bad):
    with pytest.raises(AnchoredShapeError):
        LogShapeObservation("u", "a", bad)
    with pytest.raises(AnchoredShapeError):
        fit_anchor_correction(_panel(), {"0": bad}, {str(k): k for k in range(7)})


def test_unsupported_missing_duplicate_and_nonfinite_coordinates_refuse():
    shape = _panel()
    coords = {str(k): k for k in range(7)}
    with pytest.raises(AnchoredShapeError, match="coordinate"):
        fit_anchor_correction(shape, {"0": 1.0}, {"0": 0})
    for value in (math.nan, math.inf, True, coords["2"]):
        with pytest.raises(AnchoredShapeError, match="coordinate"):
            fit_anchor_correction(shape, {"0": 1.0}, {**coords, "1": value})
    with pytest.raises(AnchoredShapeError, match="unknown"):
        fit_anchor_correction(shape, {"missing": 1.0}, coords)
    correction = fit_anchor_correction(shape, {"0": 1.0}, coords)
    with pytest.raises(AnchoredShapeError, match="unknown"):
        predict_anchored(correction, "missing")
    with pytest.raises(AnchoredShapeError, match="empty"):
        audit_anchored(correction, {})


def test_prediction_outside_finite_positive_float_range_refuses():
    features = {"a": (0.0,), "b": (1.0,), "c": (10.0,)}
    shape = fit_centered_log_shape([
        LogShapeObservation("u", "a", 1.0),
        LogShapeObservation("u", "b", 10.0),
    ], features)
    for anchors in ({"a": 1e300}, {"a": 1e-300, "b": 1e-310}):
        fit = fit_anchor_correction(shape, anchors, {"a": 0, "b": 1, "c": 10})
        with pytest.raises(AnchoredShapeError, match="representable"):
            predict_anchored(fit, "c")
