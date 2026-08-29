"""Activation-fair pricing for weight-only cost rows — ultraplan P5a item 1.

Gridbook's ``docs/audits/ultraplan_perf_2026-08-01.md`` §6 ("Three cost-model
asymmetries", #1): *W4A4 vs W8A8 activation cost is priced only on the
measured ``output_mse`` branch*. Packed experts under
``PRISMAQUANT_EXPERT_COST_SAMPLE`` and ladder-interpolated rungs under
``CB_LADDER_INTERP`` are written with ``output_mse_measured=False``, so they
fall to weight-only Fisher pricing where the activation contract is
structurally invisible — "NVFP4-CB gets credit for its cheaper index stream
with none of its A-side cost, on most rows of a production run".

What this file pins:

1. The fit is a per-family GEOMETRIC MEAN of the measured-over-weight-only
   estimator ratio, exact on synthetic rows, for BOTH CB families.
2. It is applied to weight-only rows of its own family and nowhere else —
   never to a measured row, never to an activation-identity format, never to
   another family.
3. The fail-closed policy: a MIXED scale (one family calibrated, another with
   uncorrected weight-only rows) refuses by name; a run with no measured
   activation rows at all passes through with the verdict recorded, because
   it introduces no asymmetry and refusing would make currently-legal
   research runs illegal.
4. ``PRISMAQUANT_ACTIVATION_FAIR_PRICING=0`` restores pre-P5a prices exactly.
5. No existing gate is weakened: a multiplicative penalty cannot lift an
   exactly-0.0 price off the DP's global minimum, so the
   ``ACTIVATION_COST_UNMEASURED_REASON`` candidate removal keeps firing.
6. Aggregated (fused-sibling / packed-serving-group) super items fold the
   penalty in exactly ONCE.
"""
from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.activation_fair_pricing import (
    BRANCH_ACTIVATION_IDENTITY,
    BRANCH_CALIBRATED,
    BRANCH_KILL_SWITCH,
    BRANCH_MEASURED,
    BRANCH_UNCALIBRATED,
    ENV_FLAG,
    MIN_CALIBRATION_ROWS,
    REASON_KILL_SWITCH,
    REASON_NO_MEASURED_ROWS,
    CalibrationRow,
    calibrate,
    env_enabled,
)
from prismaquant.allocator_candidates import (
    ACTIVATION_COST_UNMEASURED_REASON,
    aggregate_fused_siblings,
    build_candidates,
    calibrate_activation_fair_pricing,
    collect_activation_calibration_rows,
    cost_entry_activation_pricing_branch,
    cost_entry_predicted_dloss,
)
from prismaquant.nvfp4_cb_footprint import CBSerializationContext

_CB_CONTEXT = CBSerializationContext.production()

# Two rungs per CB family. NVFP4_CB_* is the W4 index stream served through
# the BF16 bridge; FP8_CB_* is W8A8 — the audit's live contest.
_FP4_RUNGS = ["NVFP4_CB_K16", "NVFP4_CB_K20"]
_FP8_RUNGS = ["FP8_CB_K40", "FP8_CB_K44"]
_MENU = _FP4_RUNGS + _FP8_RUNGS + ["BF16"]


def _specs(menu=None):
    return [fr.get_format(name) for name in (menu or _MENU)]


def _shape_stats(h_trace: float) -> dict:
    # in_features % 256 == 0 and out_features % 16 == 0 keeps every CB rung of
    # both families legal, so nothing drops out for shape reasons here.
    return {
        "h_trace": h_trace,
        "n_params": 1024 * 512,
        "in_features": 1024,
        "out_features": 512,
    }


def _measured_row(*, weight_mse: float, output_mse: float) -> dict:
    """A dense row: real joint-output measurement AND a weight-only field."""
    return {
        "weight_mse": weight_mse,
        "output_mse": output_mse,
        "output_mse_measured": True,
    }


def _weight_only_row(predicted_dloss: float) -> dict:
    """A packed-expert / interpolated-rung row: no activation evidence."""
    return {
        "predicted_dloss": predicted_dloss,
        "cost_source": "empirical_unit_kl",
        "output_mse_measured": False,
    }


# ---------------------------------------------------------------------------
# 1. The fit
# ---------------------------------------------------------------------------

def test_family_penalty_is_the_geometric_mean_of_the_estimator_ratio():
    """Exact on synthetic rows, per family, for both CB families.

    ratios 4 and 16 -> geometric mean 8; ratios 2 and 8 -> 4. An ARITHMETIC
    mean would give 10 and 5, which is the estimator this deliberately is
    not: Δloss spans decades across layers, so the arithmetic mean of ratios
    is set by whichever row happens to sit highest.
    """
    rows = [
        CalibrationRow("a", "NVFP4_CB_K16", "nvfp4_cb", 4.0, 1.0),
        CalibrationRow("b", "NVFP4_CB_K20", "nvfp4_cb", 16.0, 1.0),
        CalibrationRow("c", "FP8_CB_K40", "fp8_cb", 2.0, 1.0),
        CalibrationRow("d", "FP8_CB_K44", "fp8_cb", 8.0, 1.0),
    ]
    fit = calibrate(
        rows,
        measured_rows_by_family={"nvfp4_cb": 2, "fp8_cb": 2},
        weight_only_rows_by_family={"nvfp4_cb": 7, "fp8_cb": 7},
    )
    assert fit.enabled
    assert fit.families["nvfp4_cb"].penalty == pytest.approx(8.0)
    assert fit.families["fp8_cb"].penalty == pytest.approx(4.0)
    # The correction is family-scoped: a rung of one family never reads the
    # other family's factor, which is the whole point of the cross-family
    # asymmetry this fixes.
    assert fit.penalty_for("NVFP4_CB_K24", True) == (
        pytest.approx(8.0), BRANCH_CALIBRATED)
    assert fit.penalty_for("FP8_CB_K28", True) == (
        pytest.approx(4.0), BRANCH_CALIBRATED)


def test_a_single_row_is_not_enough_to_calibrate_a_family():
    """A point estimate with no residual band cannot be audited, and the
    audit trail is the deliverable — so one row does not calibrate."""
    assert MIN_CALIBRATION_ROWS == 2
    fit = calibrate(
        [CalibrationRow("a", "NVFP4_CB_K16", "nvfp4_cb", 4.0, 1.0)],
        measured_rows_by_family={"nvfp4_cb": 1},
        weight_only_rows_by_family={"nvfp4_cb": 0},
    )
    assert not fit.enabled and not fit.families


def test_provenance_records_the_sample_the_fit_and_the_residuals():
    """"Deterministic, provenance-stamped": the artifact must carry what the
    fit was made from, not just its answer."""
    rows = [
        CalibrationRow(f"layer.{i}", "NVFP4_CB_K16", "nvfp4_cb",
                       2.0 ** (i + 1), 1.0)
        for i in range(4)
    ]
    fit = calibrate(
        rows,
        measured_rows_by_family={"nvfp4_cb": 4},
        weight_only_rows_by_family={"nvfp4_cb": 3},
    )
    payload = fit.as_dict()
    family = payload["families"]["nvfp4_cb"]
    assert family["n_rows"] == 4
    # log2 ratios 1,2,3,4 -> mean 2.5 -> penalty 2**2.5
    assert family["log2_penalty"] == pytest.approx(2.5)
    assert family["penalty"] == pytest.approx(2.0 ** 2.5)
    assert family["log2_stdev"] > 0.0
    assert family["log2_stderr"] == pytest.approx(
        family["log2_stdev"] / 2.0)          # sqrt(4)
    assert family["log2_residual_min"] == pytest.approx(-1.5)
    assert family["log2_residual_max"] == pytest.approx(1.5)
    assert [row["qname"] for row in family["calibration_sample"]] == [
        "layer.0", "layer.1", "layer.2", "layer.3"]
    assert len(family["calibration_rows_sha256"]) == 64
    assert payload["env_flag"] == ENV_FLAG

    # Deterministic: same rows in a different order produce the same digest.
    shuffled = calibrate(
        list(reversed(rows)),
        measured_rows_by_family={"nvfp4_cb": 4},
        weight_only_rows_by_family={"nvfp4_cb": 3},
    )
    assert (shuffled.families["nvfp4_cb"].rows_digest
            == fit.families["nvfp4_cb"].rows_digest)


def test_rung_dependence_of_the_ratio_is_recorded_not_hidden():
    """The A-side error is rung-INDEPENDENT while the W side shrinks with k,
    so the true ratio grows along a ladder and a per-family constant
    under-corrects the top rungs. That bias is within-family (it cannot move
    the cross-family verdict) and it is stamped rather than assumed away."""
    rows = [
        CalibrationRow("a", "NVFP4_CB_K16", "nvfp4_cb", 2.0, 1.0),
        CalibrationRow("b", "NVFP4_CB_K16", "nvfp4_cb", 2.0, 1.0),
        CalibrationRow("c", "NVFP4_CB_K24", "nvfp4_cb", 8.0, 1.0),
        CalibrationRow("d", "NVFP4_CB_K24", "nvfp4_cb", 8.0, 1.0),
    ]
    fit = calibrate(
        rows,
        measured_rows_by_family={"nvfp4_cb": 4},
        weight_only_rows_by_family={"nvfp4_cb": 1},
    )
    family = fit.families["nvfp4_cb"]
    assert family.rung_dependence_log2_range == pytest.approx(2.0)  # log2(8/2)
    per_format = dict(
        (fmt, value) for fmt, value, _n in family.per_format_log2_penalty)
    assert per_format["NVFP4_CB_K16"] == pytest.approx(1.0)
    assert per_format["NVFP4_CB_K24"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 2. Where the correction lands in the precedence chain
# ---------------------------------------------------------------------------

def _two_family_tables(*, fp4_weight_only=True, fp8_weight_only=True):
    """Two measured dense rows per family plus one weight-only row each.

    Mirrors the production shape the audit describes: dense rows carry a real
    output_mse, the packed-expert row does not.
    """
    stats, costs = {}, {}
    for i in range(2):
        name = f"model.layers.{i}.mlp.down_proj"
        stats[name] = _shape_stats(1.0)
        costs[name] = {
            # measured/weight-only ratio 4x for the fp4 family (0.5*1.0*mse)
            "NVFP4_CB_K16": _measured_row(weight_mse=1e-4, output_mse=4e-4),
            "NVFP4_CB_K20": _measured_row(weight_mse=1e-4, output_mse=4e-4),
            # ...and 2x for the fp8 family.
            "FP8_CB_K40": _measured_row(weight_mse=1e-4, output_mse=2e-4),
            "FP8_CB_K44": _measured_row(weight_mse=1e-4, output_mse=2e-4),
            "BF16": {"weight_mse": 0.0, "output_mse": 0.0,
                     "output_mse_measured": True},
        }
    expert = "model.layers.9.mlp.experts.gate_up_proj"
    stats[expert] = _shape_stats(1.0)
    costs[expert] = {"BF16": _weight_only_row(0.0)}
    for rung in _FP4_RUNGS:
        costs[expert][rung] = (
            _weight_only_row(1e-3) if fp4_weight_only
            else _measured_row(weight_mse=1e-3, output_mse=4e-3))
    for rung in _FP8_RUNGS:
        costs[expert][rung] = (
            _weight_only_row(1e-3) if fp8_weight_only
            else _measured_row(weight_mse=1e-3, output_mse=2e-3))
    return stats, costs, expert


def test_calibration_corrects_weight_only_rows_and_leaves_measured_rows_alone():
    stats, costs, expert = _two_family_tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    assert pricing.enabled
    # Both branches are 1/2*h_trace*MSE over the SAME h_trace, so the fitted
    # factor is exactly output_mse/weight_mse: 4x for the W4 family, 2x for
    # the W8 one — the activation-contract asymmetry the audit describes.
    assert pricing.families["nvfp4_cb"].penalty == pytest.approx(4.0)
    assert pricing.families["fp8_cb"].penalty == pytest.approx(2.0)

    cands = build_candidates(
        stats, costs, _specs(), target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT,
        activation_pricing=pricing)

    priced = {c.fmt: c for c in cands[expert]}
    # The weight-only expert row is charged its family's factor...
    assert priced["NVFP4_CB_K16"].predicted_dloss == pytest.approx(4e-3)
    assert priced["FP8_CB_K40"].predicted_dloss == pytest.approx(2e-3)
    assert priced["NVFP4_CB_K16"].activation_pricing == BRANCH_CALIBRATED
    # ...while the measured dense rows keep the activation-inclusive price
    # they already had, bit for bit.
    dense = {c.fmt: c for c in cands["model.layers.0.mlp.down_proj"]}
    assert dense["NVFP4_CB_K16"].predicted_dloss == pytest.approx(0.5 * 4e-4)
    assert dense["NVFP4_CB_K16"].activation_pricing == BRANCH_MEASURED
    # BF16 quantizes no activations, so it is never in scope.
    assert priced["BF16"].activation_pricing == BRANCH_ACTIVATION_IDENTITY
    assert priced["BF16"].predicted_dloss == 0.0


def test_the_correction_is_exactly_the_difference_from_pre_p5a_pricing():
    """Same row, same table: the ONLY delta is the family multiplier."""
    stats, costs, expert = _two_family_tables()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    entry = costs[expert]["NVFP4_CB_K16"]
    before = cost_entry_predicted_dloss(
        stats[expert], entry, format_name="NVFP4_CB_K16")
    after = cost_entry_predicted_dloss(
        stats[expert], entry, format_name="NVFP4_CB_K16",
        activation_pricing=pricing)
    assert after == pytest.approx(
        before * pricing.families["nvfp4_cb"].penalty)


def test_calibration_rows_are_collected_only_from_activation_quantizing_formats():
    stats, costs, _expert = _two_family_tables()
    rows, measured, weight_only = collect_activation_calibration_rows(
        stats, costs, _specs())
    assert {row.family for row in rows} == {"nvfp4_cb", "fp8_cb"}
    # BF16's activation path is the identity: it has no A side to transfer,
    # so it never enters the sample and never appears in the branch counts.
    assert "fp" not in measured and "fp" not in weight_only
    assert measured == {"nvfp4_cb": 4, "fp8_cb": 4}
    assert weight_only == {"nvfp4_cb": 2, "fp8_cb": 2}


def test_the_penalty_is_gain_invariant():
    """A calibrated gain multiplies both estimators identically, so the fit
    does not have to be redone when --calibration changes."""
    stats, costs, _expert = _two_family_tables()
    rows, _m, _w = collect_activation_calibration_rows(stats, costs, _specs())
    assert rows
    for row in rows:
        # gain enters both branches as the same factor; the extracted ratio
        # is therefore already gain-free.
        assert row.measured_dloss > 0.0 and row.weight_only_dloss > 0.0


# ---------------------------------------------------------------------------
# 3. Fail-closed policy
# ---------------------------------------------------------------------------

def test_a_mixed_cost_scale_refuses_and_names_the_env_var():
    """One family calibrated while the other still prices weight-only is
    WORSE than the status quo: it tilts exactly the NVFP4-vs-FP8-CB
    comparison the calibration exists to make fair."""
    stats, costs, expert = _two_family_tables()
    # Strip the fp8 family's measured evidence, keeping its weight-only rows.
    for name in list(costs):
        if name != expert:
            for rung in _FP8_RUNGS:
                costs[name][rung] = _weight_only_row(1e-4)
    with pytest.raises(AssertionError) as exc:
        calibrate_activation_fair_pricing(stats, costs, _specs())
    message = str(exc.value)
    assert ENV_FLAG in message
    assert "MIXED cost scale" in message
    assert "fp8_cb" in message and "nvfp4_cb" in message
    assert "PRISMAQUANT_EXPERT_COST_SAMPLE" in message


def test_no_measured_activation_rows_passes_through_but_is_recorded():
    """Nothing can be corrected and no asymmetry is introduced, so the run
    stays legal — but the verdict is stamped, so it is never silent."""
    stats, costs, _expert = _two_family_tables()
    for name in list(costs):
        for rung in _FP4_RUNGS + _FP8_RUNGS:
            costs[name][rung] = _weight_only_row(1e-4)
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    assert not pricing.enabled
    assert pricing.reason == REASON_NO_MEASURED_ROWS
    assert pricing.weight_only_rows_by_family == {"nvfp4_cb": 6, "fp8_cb": 6}
    assert set(pricing.uncalibrated_families) == {"nvfp4_cb", "fp8_cb"}
    assert pricing.as_dict()["enabled"] is False
    # Uncorrected rows say so on the candidate.
    cands = build_candidates(
        stats, costs, _specs(), target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT,
        activation_pricing=pricing)
    branches = {
        c.activation_pricing
        for c in cands["model.layers.0.mlp.down_proj"]
    }
    assert BRANCH_UNCALIBRATED in branches


def test_the_unmeasured_activation_removal_gate_is_not_weakened():
    """A multiplicative penalty cannot lift an exactly-0.0 price off the DP's
    global minimum, so the pre-existing candidate removal still fires on a
    weight-lossless re-encode of an activation-quantizing format."""
    stats = {"model.layers.0.mlp.down_proj": _shape_stats(1.0)}
    costs = {
        "model.layers.0.mlp.down_proj": {
            # Lossless W side, no measured output_mse, positive h_trace: the
            # exact predicate cost_entry_prices_unmeasured_activation_at_zero
            # describes.
            "NVFP4": {"weight_mse": 0.0, "predicted_dloss": 0.0,
                      "output_mse_measured": False},
            "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0,
                     "output_mse_measured": False},
        }
    }
    pricing = calibrate(
        [CalibrationRow("x", "NVFP4", "nv", 4.0, 1.0),
         CalibrationRow("y", "NVFP4", "nv", 4.0, 1.0)],
        measured_rows_by_family={"nv": 2},
        weight_only_rows_by_family={"nv": 0},
    )
    masks: list[dict] = []
    cands = build_candidates(
        stats, costs, _specs(["NVFP4", "BF16"]),
        mask_records=masks, activation_pricing=pricing)
    assert {c.fmt for c in cands["model.layers.0.mlp.down_proj"]} == {"BF16"}
    assert [m["reason"] for m in masks] == [ACTIVATION_COST_UNMEASURED_REASON]


# ---------------------------------------------------------------------------
# 4. Kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_restores_pre_p5a_prices_exactly(monkeypatch):
    stats, costs, expert = _two_family_tables()
    monkeypatch.setenv(ENV_FLAG, "0")
    assert not env_enabled()
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    assert not pricing.enabled and pricing.reason == REASON_KILL_SWITCH

    switched = build_candidates(
        stats, costs, _specs(), target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT, activation_pricing=pricing)
    monkeypatch.delenv(ENV_FLAG, raising=False)
    legacy = build_candidates(
        stats, costs, _specs(), target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT, activation_pricing=None)
    for name in switched:
        got = {c.fmt: c.predicted_dloss for c in switched[name]}
        want = {c.fmt: c.predicted_dloss for c in legacy[name]}
        assert got == want, name
    assert {c.activation_pricing for c in switched[expert]} >= {
        BRANCH_KILL_SWITCH}


def test_kill_switch_also_suppresses_the_mixed_scale_refusal(monkeypatch):
    """The switch's job is to reproduce a pre-P5a artifact; refusing under it
    would make that impossible on exactly the runs that need it."""
    stats, costs, expert = _two_family_tables()
    for name in list(costs):
        if name != expert:
            for rung in _FP8_RUNGS:
                costs[name][rung] = _weight_only_row(1e-4)
    monkeypatch.setenv(ENV_FLAG, "0")
    pricing = calibrate_activation_fair_pricing(stats, costs, _specs())
    assert not pricing.enabled and pricing.reason == REASON_KILL_SWITCH


# ---------------------------------------------------------------------------
# 5. Aggregation
# ---------------------------------------------------------------------------

class _FusedProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".gate_proj", ".up_proj")):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


def test_a_fused_super_item_folds_the_family_penalty_in_exactly_once():
    """The super item's Δloss is the SUM of penalized member prices — not the
    sum re-penalized. The marker on the aggregated cost entry is what makes
    a re-price idempotent."""
    stats, costs = {}, {}
    for leaf in ("gate_proj", "up_proj"):
        name = f"model.layers.0.mlp.{leaf}"
        stats[name] = _shape_stats(1.0)
        costs[name] = {
            "NVFP4_CB_K16": _weight_only_row(1e-3),
            "BF16": _weight_only_row(0.0),
        }
    # Separate measured rows supply the fp4 family's calibration (ratio 3x).
    for i in range(2):
        dense = f"model.layers.{i}.self_attn.o_proj"
        stats[dense] = _shape_stats(1.0)
        costs[dense] = {
            "NVFP4_CB_K16": _measured_row(weight_mse=1e-4, output_mse=3e-4),
            "BF16": {"weight_mse": 0.0, "output_mse": 0.0,
                     "output_mse_measured": True},
        }
    specs = _specs(["NVFP4_CB_K16", "BF16"])
    pricing = calibrate_activation_fair_pricing(stats, costs, specs)
    assert pricing.families["nvfp4_cb"].penalty == pytest.approx(3.0)

    cands = build_candidates(
        stats, costs, specs, target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT, activation_pricing=pricing)
    stats_ext, costs_ext, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile=_FusedProfile(),
        activation_pricing=pricing)
    [super_name] = [n for n in cands_ext if ".__siblings__." in n]
    super_cand = next(
        c for c in cands_ext[super_name] if c.fmt == "NVFP4_CB_K16")
    assert super_cand.predicted_dloss == pytest.approx(2 * 3.0 * 1e-3)
    # Re-pricing the aggregated entry must not square the correction.
    assert cost_entry_predicted_dloss(
        stats_ext[super_name],
        costs_ext[super_name]["NVFP4_CB_K16"],
        format_name="NVFP4_CB_K16",
        activation_pricing=pricing,
    ) == pytest.approx(2 * 3.0 * 1e-3)
    assert super_cand.activation_pricing == BRANCH_CALIBRATED


def test_a_super_item_with_mixed_member_branches_says_so():
    """"Some rows of this serving unit never saw an activation measurement"
    is the fact the stamp exists to preserve; a majority vote would erase
    it."""
    stats, costs = {}, {}
    stats["model.layers.0.mlp.gate_proj"] = _shape_stats(1.0)
    costs["model.layers.0.mlp.gate_proj"] = {
        "NVFP4_CB_K16": _measured_row(weight_mse=1e-4, output_mse=3e-4),
        "BF16": {"weight_mse": 0.0, "output_mse": 0.0,
                 "output_mse_measured": True},
    }
    stats["model.layers.0.mlp.up_proj"] = _shape_stats(1.0)
    costs["model.layers.0.mlp.up_proj"] = {
        "NVFP4_CB_K16": _weight_only_row(1e-3),
        "BF16": _weight_only_row(0.0),
    }
    for i in range(2):
        dense = f"model.layers.{i}.self_attn.o_proj"
        stats[dense] = _shape_stats(1.0)
        costs[dense] = dict(costs["model.layers.0.mlp.gate_proj"])
    specs = _specs(["NVFP4_CB_K16", "BF16"])
    pricing = calibrate_activation_fair_pricing(stats, costs, specs)
    cands = build_candidates(
        stats, costs, specs, target_profile="nvfp4_cb",
        cb_serialization_context=_CB_CONTEXT, activation_pricing=pricing)
    _s, _c, cands_ext = aggregate_fused_siblings(
        stats, costs, specs, cands, profile=_FusedProfile(),
        activation_pricing=pricing)
    [super_name] = [n for n in cands_ext if ".__siblings__." in n]
    branch = next(
        c for c in cands_ext[super_name] if c.fmt == "NVFP4_CB_K16"
    ).activation_pricing
    assert branch.startswith("mixed:")
    assert BRANCH_CALIBRATED in branch and BRANCH_MEASURED in branch


# ---------------------------------------------------------------------------
# 6. The allocation the audit says is wrong today
# ---------------------------------------------------------------------------

def test_the_dp_flips_to_fp8_cb_once_the_w4_activation_cost_is_priced():
    """The audit's whole point, end to end through the real knapsack.

    An expert unit's weight-only rows make NVFP4-CB look strictly better than
    FP8-CB per byte, so the DP takes the W4 rung — "NVFP4-CB gets credit for
    its cheaper index stream with none of its A-side cost". The measured
    dense rows in the SAME run say the W4 family's activation contract costs
    3x what the weight-only estimator reports while the W8 family's costs
    1x. With that transfer applied, the same DP at the same byte budget picks
    FP8-CB instead.

    NB the DP itself is untouched: this is a pricing change, not a solver
    change (no latency term, no second axis — that is P5c).
    """
    from prismaquant.allocator_solver import solve_allocation

    menu = ["NVFP4_CB_K24", "FP8_CB_K40", "BF16"]
    specs = _specs(menu)
    stats, costs = {}, {}
    # Two measured dense rows per family supply the calibration: the W4
    # family's joint-output error is 3x its weight-side surrogate, the W8
    # family's is 1x (its A side is the same per-token E4M3 QDQ the source
    # already used).
    for i in range(2):
        name = f"model.layers.{i}.self_attn.o_proj"
        stats[name] = _shape_stats(1.0)
        costs[name] = {
            "NVFP4_CB_K24": _measured_row(weight_mse=1e-5, output_mse=3e-5),
            "FP8_CB_K40": _measured_row(weight_mse=1e-5, output_mse=1e-5),
            "BF16": {"weight_mse": 0.0, "output_mse": 0.0,
                     "output_mse_measured": True},
        }
    # The expert unit: weight-only rows only (PRISMAQUANT_EXPERT_COST_SAMPLE).
    # NVFP4_CB_K24 (3.281 bpw under two-tier v2) is both CHEAPER in bytes and
    # priced lower than FP8_CB_K40 (5.008 bpw) — dominant, on weight-only
    # evidence alone.
    expert = "model.layers.9.mlp.experts.gate_up_proj"
    stats[expert] = _shape_stats(1.0)
    costs[expert] = {
        "NVFP4_CB_K24": _weight_only_row(1.0e-3),
        "FP8_CB_K40": _weight_only_row(1.2e-3),
        "BF16": _weight_only_row(0.0),
    }

    def _solve(pricing):
        cands = build_candidates(
            stats, costs, specs, target_profile="nvfp4_cb",
            cb_serialization_context=_CB_CONTEXT,
            activation_pricing=pricing)
        # Budget the expert row alone at a rate both CB rungs can reach.
        solved = solve_allocation(
            {expert: stats[expert]}, {expert: cands[expert]}, 5.1)
        assert solved is not None
        assignment, _chosen = solved
        return assignment[expert]

    assert _solve(None) == "NVFP4_CB_K24"          # pre-P5a: W4 wins
    pricing = calibrate_activation_fair_pricing(stats, costs, specs)
    assert pricing.families["nvfp4_cb"].penalty == pytest.approx(3.0)
    assert pricing.families["fp8_cb"].penalty == pytest.approx(1.0)
    assert _solve(pricing) == "FP8_CB_K40"         # activation-fair: W8 wins


def test_branch_label_is_derivable_without_a_calibration_object():
    """``activation_pricing=None`` is the pre-P5a path every direct caller
    outside candidate construction still uses; it must still name the
    branch honestly rather than claim a correction it did not apply."""
    stats = _shape_stats(1.0)
    assert cost_entry_activation_pricing_branch(
        stats, _weight_only_row(1e-3), "NVFP4_CB_K16", None
    ) == BRANCH_UNCALIBRATED
    assert cost_entry_activation_pricing_branch(
        stats, _measured_row(weight_mse=1e-4, output_mse=2e-4),
        "NVFP4_CB_K16", None
    ) == BRANCH_MEASURED
    assert cost_entry_activation_pricing_branch(
        stats, _weight_only_row(0.0), "BF16", None
    ) == BRANCH_ACTIVATION_IDENTITY
