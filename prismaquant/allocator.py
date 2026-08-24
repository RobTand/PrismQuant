#!/usr/bin/env python3
"""allocator.py — multi-choice knapsack mixed-precision assignment.

Given:
  - per-Linear empirical Fisher diagonal trace (from sensitivity_probe.py)
  - per-(Linear, format) measured quantization cost (from measure_quant_cost.py)
  - a bit budget (target average bits per parameter)
  - a format registry (any subset of registered formats)

Solve for a per-Linear format assignment that minimizes total predicted
loss increase subject to the bit budget.

Derivation of the per-(layer, format) predicted loss term
---------------------------------------------------------
Let L be the per-token loss (negative log-likelihood). Quantizing layer
ℓ's weight tensor W by ΔW = W_q - W produces a perturbed loss whose
expectation under the calibration distribution admits the standard
second-order expansion:

    E[ΔL] ≈ 0.5 · ΔW · F · ΔWᵀ                         (1)

where F is the Fisher information matrix of L w.r.t. W. Replacing F by
its diagonal (the standard HAWQ-V1 simplification) and approximating
F_ww by the empirical Fisher diagonal F̂_ww = E_token[(∂L/∂W_w)²]:

    E[ΔL] ≈ 0.5 · Σ_w F̂_ww · (ΔW_w)²                   (2)

Under the further assumption that the per-weight quantization error
(ΔW_w)² and the per-weight Fisher diagonal F̂_ww are uncorrelated across
w (which is the same assumption HAWQ already makes when it summarizes a
layer by a single scalar), this collapses to the product of two
per-layer scalars:

    E[ΔL] ≈ 0.5 · H_trace · MSE_W                       (3)

where
    H_trace = Σ_w F̂_ww            (per-token Fisher diagonal trace)
    MSE_W   = (1/n_w) · Σ_w (ΔW_w)²

Both quantities are produced by upstream stages:
    H_trace ← sensitivity_probe.py / FisherAccumulator (`h_trace`)
    MSE_W   ← measure_quant_cost.py (per-(layer, format) `weight_mse`)

So we use eq. (3) directly. There is no `* d_out` factor; the previous
implementation carried one but it does not appear in the derivation —
it was a holdover from an earlier output-side formulation that mixed
units and was off by a per-layer multiplicative constant that varies
with d_out.

For MoE experts an additional route-probability normalization is folded
into H_trace inside the probe so that sparsely-routed experts' Fisher
contributions are on the same per-token footing as dense layers'.

Solver:
  Multi-choice knapsack via DP with bit-budget discretization (we round
  bit costs to 0.001-bit bins). For 35B with ~300 Linears × 8 formats ×
  ~5000 budget bins, runtime is under 1s.

Fused-projection siblings (q/k/v/o, gate/up, ...) are post-processed:
  all siblings promoted to the highest format chosen for any of them,
  to match vLLM's fused-tensor loader constraints. Since promotion can
  push achieved bits past the requested budget, the DP is re-run with a
  tightened target until achieved is within tolerance.

Optional empirical calibration:
  If `--calibration` points at a JSON containing
  `calibrated_gains[fmt] = α_fmt`, the predicted Δloss for format f is
  multiplied by α_f before the DP runs. The historical tiny-bakeoff
  producer for this payload is archived; production recipes normally run
  uncalibrated and validate assignments with direct KL measurement.

Auto-Pareto knee via Kneedle (Satopää et al.). The primary recommendation
uses a post-cliff log-error knee: when the frontier spans multiple orders of
magnitude, the first half of the log-error range is treated as the
catastrophic region and Kneedle is applied to the remaining operational
frontier. The global log-error and raw-linear knees are still reported as
diagnostics. (We implement Kneedle here rather than depend on the `kneed`
PyPI package because the post-cliff log-error variant — trimming the
catastrophic prefix before applying the knee — has no equivalent in `kneed`,
and the curve is tiny so the dependency would not pay for itself.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import (
    Candidate,
    _shape_from_stats,
    compute_achieved,
    compute_assignment_predicted_dloss,
    legal_formats_from_candidates,
    promote_fused,
    promote_moe_pair,
    promote_serving_units,
    solve_with_promotion,
)
from .allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _FUSED_SIBLING_MARKER,
    _PACKED_GROUP_MARKER,
    _format_kernel_supports_shape,
    _is_passthrough_format,
    _passthrough_source_ok,
    _scan_source_dtype_manifest,
    aggregate_fused_siblings,
    aggregate_packed_serving_groups,
    build_candidates,
    calibrate_activation_fair_pricing,
    check_stats_format_applicability,
    expand_fused_sibling_assignment,
    expand_packed_group_assignment,
    packed_role_split_profile,
    selection_serving_lane_provenance,
    serialized_candidate_payload,
    summarize_applicability_masks,
)
from .fixed_head import (
    allow_pinned_lifts_lm_head,
    is_lm_head_name,
    parse_allow_pinned,
)
from .nvfp4_cb_footprint import (
    CB_ASSIGNMENT_IDENTITIES_FIELD,
    CB_TENSOR_IDENTITY_FIELD,
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_assignment_serialization_stamps,
    cb_serialization_context_from_env,
    cb_serialization_context_stamp,
    cb_tensor_payload_breakdown,
    is_cb_format,
    validate_cb_cost_provenance,
    whole_artifact_budget_stamp,
)
from .production_weight_cache import (
    project_cb_render_identity,
    validate_cb_render_provenance,
)
from .footprint import (
    NVFP4_WEIGHT_ONLY_STATS_KEY,
    nvfp4_global_sidecar_bytes,
)
from .serving_profiles import (
    check_serving_format,
    require_per_role_expert_scheme_support,
    resolve_target_profile,
    serving_lane_catalog,
    serving_lane_route,
    serving_profile_names,
)
from .cb_ladder_cross_family import (
    cross_family_verdict_from_cost_payload,
)
from .serve_constraints import (
    ServeConstraintContext,
    ServeConstraintError,
    ServeSLOs,
    WorkloadMix,
    evaluate_assignment as evaluate_serve_constraints,
    fastest_feasible_summary,
    rejection_record,
)
from .serve_dispatch_table import DispatchTableError, load_dispatch_table
from .decision_units import block_id_from_qname
from .layer_config import LAYER_CONFIG_META_KEY
from .schemas import validate_cost_payload, validate_probe_payload


_KNEE_DIAGNOSTIC_MIN_LOG_SPAN_DECADES = 1.0
_KNEE_DIAGNOSTIC_TAIL_MIDPOINT_FRACTION = 0.5


def _serialized_format_rates(
    specs: list[fr.FormatSpec],
    stats: Mapping[str, Mapping],
    cb_serialization_context: CBSerializationContext | None,
) -> dict[str, float]:
    """Artifact-faithful menu ordering, independent of input menu order.

    CB FormatSpec rates are deliberately incomplete.  Rank each format by the
    exact payload it would use across the available Linear shapes; for CB this
    includes FP8 row scales and once-only codebook sidecars. Shapes a format
    cannot serialize are omitted (the applicability gate will remove them
    later). A name tie-break makes the result deterministic.
    """
    rates: dict[str, float] = {}
    for spec in specs:
        shapes: dict[str, tuple[int, ...]] = {}
        total_params = 0
        for name, entry in stats.items():
            if not isinstance(entry, Mapping):
                continue
            shape = _shape_from_stats(dict(entry))
            if len(shape) < 2 or any(int(dim) <= 0 for dim in shape):
                continue
            try:
                if is_cb_format(spec.name):
                    if cb_serialization_context is None:
                        continue
                    # The exact accountant owns the divisibility/shape gate.
                    from .nvfp4_cb_footprint import cb_tensor_payload_breakdown

                    cb_tensor_payload_breakdown(
                        spec.name,
                        shape,
                        qname=str(name),
                        context=cb_serialization_context,
                    )
                else:
                    spec.memory_bytes_for_shape(shape)
            except (ValueError, AssertionError):
                continue
            shapes[str(name)] = shape
            total_params += int(math.prod(shape))
        if total_params <= 0:
            rates[spec.name] = float(spec.effective_bits)
            continue
        if is_cb_format(spec.name):
            payload = cb_assignment_payload_breakdown(
                {name: spec.name for name in shapes},
                shapes,
                context=cb_serialization_context,
            )
            total_bytes = int(payload["total_bytes"])
        else:
            total_bytes = sum(
                int(spec.memory_bytes_for_shape(shape))
                for shape in shapes.values()
            )
        rates[spec.name] = 8.0 * total_bytes / float(total_params)
    return rates


def _sort_specs_by_serialized_rate(
    specs: list[fr.FormatSpec],
    stats: Mapping[str, Mapping],
    cb_serialization_context: CBSerializationContext | None,
) -> tuple[list[fr.FormatSpec], dict[str, float]]:
    rates = _serialized_format_rates(specs, stats, cb_serialization_context)
    return (
        sorted(specs, key=lambda spec: (rates[spec.name], spec.name)),
        rates,
    )
_RD_LOG_LINEAR_R2_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Kneedle knee detection
# ---------------------------------------------------------------------------
def _kneedle_convex_decreasing(x: list[float], y: list[float]) -> int:
    """Return index of the knee in a convex-decreasing curve."""
    if len(x) < 3:
        return 0
    xs = [xi for xi in x]
    ys = [yi for yi in y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return 0
    x_norm = [(xi - xmin) / (xmax - xmin) for xi in xs]
    y_norm = [(yi - ymin) / (ymax - ymin) for yi in ys]
    # For a convex-decreasing curve, the knee is the point with max
    # distance below the chord from (0,1) to (1,0).
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    # Convex-decreasing, so we want the most-negative diff (max dip).
    return min(range(len(diffs)), key=lambda i: diffs[i])


def _log_error_values(y: list[float]) -> list[float]:
    """Return log10(dloss) with non-positive values floored at the smallest
    positive point.

    A measured dloss <= 0 means "at the measurement floor" (realistic: an
    all-passthrough rung on an FP8-native source has total dloss exactly 0),
    not "10^6x better than the best positive point". The old floor of
    ``min_positive * 1e-6`` injected a ~6-decade fake cliff below the real
    curve, compressing it and dragging the Kneedle to the curve start.
    Flooring at ``min_positive`` itself keeps such points 0 decades below
    the smallest real point, so the knee stays on the measured curve.
    """
    finite_positive = [
        float(v) for v in y
        if math.isfinite(float(v)) and float(v) > 0.0
    ]
    if not finite_positive:
        return [0.0 for _ in y]
    floor = min(finite_positive)
    return [math.log10(max(float(v), floor)) for v in y]


def _log_error_tail_start(y: list[float]) -> int:
    """Return the first index in the operational log-error frontier.

    Full Pareto curves often contain a catastrophic low-bit prefix where error
    drops by orders of magnitude before the useful shipping frontier starts.
    Applying Kneedle over the whole log range can then select the transition
    out of catastrophe instead of the real tradeoff knee. Trim that prefix once
    the curve spans at least one decade, starting at the first point in the
    lower half of the log-error range. Keep at least three points for Kneedle.
    """
    logs = _log_error_values(y)
    if len(logs) < 3:
        return 0
    ymin, ymax = min(logs), max(logs)
    if ymax - ymin < _KNEE_DIAGNOSTIC_MIN_LOG_SPAN_DECADES:
        return 0
    threshold = ymin + _KNEE_DIAGNOSTIC_TAIL_MIDPOINT_FRACTION * (ymax - ymin)
    idx = next((i for i, v in enumerate(logs) if v <= threshold), 0)
    return min(max(idx, 0), max(len(logs) - 3, 0))


def kneedle_raw_linear(x: list[float], y: list[float]) -> int:
    """Return the historical raw-error Kneedle index."""
    return _kneedle_convex_decreasing(x, y)


def kneedle_log_error_global(x: list[float], y: list[float]) -> int:
    """Return Kneedle over the full log10(error) range."""
    return _kneedle_convex_decreasing(x, _log_error_values(y))


def kneedle_log_error(x: list[float], y: list[float]) -> int:
    """Return the primary post-cliff Kneedle index on log10(error)."""
    start = _log_error_tail_start(y)
    local = _kneedle_convex_decreasing(
        x[start:],
        _log_error_values(y[start:]),
    )
    return start + local


def kneedle(x: list[float], y: list[float]) -> int:
    """Return the default allocator knee index.

    The allocator's Δloss can span orders of magnitude. Running Kneedle
    on raw error lets the largest point dominate the normalized curve and
    tends to pick the first large absolute drop. Running Kneedle over the
    full log-error range can still pick the transition out of catastrophic
    error. The default therefore runs on the post-cliff log-error frontier;
    global-log and raw-linear knees remain available for diagnostics and
    backwards comparisons.
    """
    return kneedle_log_error(x, y)


def _pareto_knee_summary(curve: list[dict]) -> dict:
    feasible = [row for row in curve if row.get("feasible")]
    if len(feasible) < 3:
        return {"enabled": False, "reason": "too_few_feasible_points"}
    xs = [float(r["achieved_bits"]) for r in feasible]
    error_key = "predicted_dloss"
    ys = [float(r.get(error_key, r["predicted_dloss"])) for r in feasible]

    def _record(mode: str, idx: int) -> dict:
        row = feasible[idx]
        record = {
            "mode": mode,
            "target_bits": float(row["target_bits"]),
            "achieved_bits": float(row["achieved_bits"]),
            "predicted_dloss": float(row["predicted_dloss"]),
            "kneedle_dloss": float(row.get(error_key, row["predicted_dloss"])),
            "kneedle_error_source": error_key,
            "index": int(idx),
        }
        for field in (
            "aux_fixed_predicted_dloss",
            "fixed_predicted_dloss",
            "total_predicted_dloss_with_aux",
        ):
            if field in row:
                record[field] = float(row[field])
        return record

    raw_idx = kneedle_raw_linear(xs, ys)
    global_log_idx = kneedle_log_error_global(xs, ys)
    log_idx = kneedle_log_error(xs, ys)
    return {
        "enabled": True,
        "primary": "log_error",
        "diagnostic_thresholds": {
            "tail_min_log_span_decades": float(
                _KNEE_DIAGNOSTIC_MIN_LOG_SPAN_DECADES
            ),
            "tail_midpoint_fraction": float(
                _KNEE_DIAGNOSTIC_TAIL_MIDPOINT_FRACTION
            ),
        },
        "log_error": _record("log_error", log_idx),
        "global_log_error": _record("global_log_error", global_log_idx),
        "raw_linear": _record("raw_linear", raw_idx),
        "rd_curve": _rd_curve_diagnostic(feasible),
    }


def _rd_curve_diagnostic(feasible: list[dict]) -> dict:
    """Is the rate-distortion curve log-linear (no intrinsic knee)?

    The Aura 27B finding (handover 2026-06-05): the surrogate RD curve is a clean
    exponential, KL ~ 10^(-a*bpp), i.e. a straight line in (bpp, log10 Δloss).
    On such a curve the kneedle has no fixed answer — its "knee" moves with the
    axis scaling (the 27B knee swung 7.5 -> 12 bpp across raw/log/golden axes),
    so it is a *diagnostic*, not a ship-point. This fits a least-squares line to
    log10(Δloss) vs achieved bpp and reports R^2. The R^2 cutoff is a
    diagnostic threshold only, reported into the sidecar payload; it is not a
    shipping selector. A genuine sensitivity cliff would kink the line (low R^2)
    and *then* a knee is meaningful.
    """
    pts = [r for r in feasible if float(r.get("predicted_dloss", 0.0)) > 0.0]
    if len(pts) < 3:
        return {"available": False, "reason": "too_few_positive_points"}
    xs = [float(r["achieved_bits"]) for r in pts]
    ys = [math.log10(float(r["predicted_dloss"])) for r in pts]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return {"available": False, "reason": "degenerate_bpp_range"}
    a = (n * sxy - sx * sy) / denom            # decades of Δloss per bit (<0)
    b = (sy - a * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    log_linear = r2 >= _RD_LOG_LINEAR_R2_THRESHOLD
    return {
        "available": True,
        "model": "log10(predicted_dloss) = a*bpp + b",
        "slope_decades_per_bit": float(a),
        "intercept": float(b),
        "r2": float(r2),
        "diagnostic_thresholds": {
            "log_linear_r2": float(_RD_LOG_LINEAR_R2_THRESHOLD),
        },
        "log_linear": bool(log_linear),
        "intrinsic_knee": bool(not log_linear),
        "note": (
            f"RD curve is log-linear (R^2>={_RD_LOG_LINEAR_R2_THRESHOLD:g}): "
            "no intrinsic knee; the kneedle "
            "is axis-dependent. Select ship bpp by byte budget (--target-disk-gb) "
            "or measured saturation, not curvature."
            if log_linear else
            f"RD curve deviates from log-linear "
            f"(R^2<{_RD_LOG_LINEAR_R2_THRESHOLD:g}): a curvature knee may be "
            "meaningful here; still prefer a byte budget when shipping to a card."
        ),
    }


_GOLDEN_RATIO_INV = (5.0 ** 0.5 - 1.0) / 2.0  # 0.6180339887...


def refine_knee_golden(
    solve_fn,
    knee_summary: dict,
    curve: list[dict],
    *,
    tol: float = 0.03,
    max_evals: int = 24,
):
    """Golden-section refinement of the coarse log-error knee.

    The Pareto sweep runs over a coarse (linearly spaced) target grid, so the
    Kneedle can only land *on a grid point*. At large model scale a 0.05-bpp
    miss is real size/quality left on the table. After the coarse sweep brackets
    the knee, this golden-section search (a binary-search-family optimizer for
    the unimodal dip-below-the-chord) re-solves the sub-second DP at interior
    target budgets and returns the budget that maximizes the perpendicular dip
    below the bracket chord in (achieved_bits, log10 error) space — the true
    knee to ~``tol`` bpp. Returns ``(refined_record | None, extra_curve_pts)``.

    ``solve_fn(target_bits) -> (assignment|None, achieved_bits, predicted_dloss, _)``
    """
    if not knee_summary.get("enabled"):
        return None, []
    feasible = sorted(
        (r for r in curve if r.get("feasible")),
        key=lambda r: float(r["achieved_bits"]),
    )
    if len(feasible) < 3:
        return None, []
    primary = knee_summary.get(knee_summary.get("primary", "log_error"))
    if not primary:
        return None, []
    ki = min(
        range(len(feasible)),
        key=lambda i: abs(
            float(feasible[i]["target_bits"]) - float(primary["target_bits"])
        ),
    )
    lo_t = float(feasible[max(ki - 1, 0)]["target_bits"])
    hi_t = float(feasible[min(ki + 1, len(feasible) - 1)]["target_bits"])
    if hi_t - lo_t <= tol:
        return None, []

    # Same floor convention as _log_error_values: a dloss <= 0 is "at the
    # measurement floor", not 300 decades better. Flooring a zero bracket
    # endpoint at 1e-300 made the chord vertical and dragged the refined
    # knee to the opposite bracket edge.
    positive_dloss = [
        float(r["predicted_dloss"]) for r in feasible
        if math.isfinite(float(r["predicted_dloss"]))
        and float(r["predicted_dloss"]) > 0.0
    ]
    ylog_floor = min(positive_dloss) if positive_dloss else 1.0e-300

    def _ylog(dloss: float) -> float:
        return math.log10(max(float(dloss), ylog_floor))

    pL, pH = solve_fn(lo_t), solve_fn(hi_t)
    if pL[0] is None or pH[0] is None:
        return None, []
    xL, yL = float(pL[1]), _ylog(pL[2])
    xH, yH = float(pH[1]), _ylog(pH[2])
    if xH == xL:
        return None, []

    evaluated: list[tuple[float, float, tuple]] = []  # (dip, target, solve_result)

    def _eval(target: float):
        p = solve_fn(target)
        if p[0] is None:
            return -math.inf
        x, y = float(p[1]), _ylog(p[2])
        ychord = yL + (yH - yL) * (x - xL) / (xH - xL)
        dip = ychord - y  # convex-decreasing curve: the knee dips below the chord
        evaluated.append((dip, float(target), p))
        return dip

    a, b = lo_t, hi_t
    c = b - _GOLDEN_RATIO_INV * (b - a)
    d = a + _GOLDEN_RATIO_INV * (b - a)
    fc, fd = _eval(c), _eval(d)
    evals = 2
    while (b - a) > tol and evals < max_evals:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN_RATIO_INV * (b - a)
            fc = _eval(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN_RATIO_INV * (b - a)
            fd = _eval(d)
        evals += 1

    if not evaluated:
        return None, []
    best_dip, best_target, best_p = max(evaluated, key=lambda t: t[0])
    refined = {
        "mode": "log_error_golden_refined",
        "target_bits": float(best_target),
        "achieved_bits": float(best_p[1]),
        "predicted_dloss": float(best_p[2]),
        "kneedle_dloss": float(best_p[2]),
        "kneedle_error_source": "predicted_dloss",
        "coarse_target_bits": float(primary["target_bits"]),
        "coarse_achieved_bits": float(primary["achieved_bits"]),
        "bracket_target_bits": [lo_t, hi_t],
        "evals": int(evals),
        "tol_bits": float(tol),
    }
    extra = [
        {
            "target_bits": float(t), "achieved_bits": float(p[1]),
            "predicted_dloss": float(p[2]), "feasible": True, "knee_refine": True,
        }
        for (_, t, p) in evaluated if p[0] is not None
    ]
    return refined, extra


def _allowed_format(target_profile: str, name: str, fmt: str) -> bool:
    decision = check_serving_format(target_profile, name, fmt)
    if not decision.legal and decision.detail.startswith("unknown target profile"):
        raise ValueError(decision.detail)
    return decision.legal


def filter_candidates_for_profile(
    candidates: dict[str, list[Candidate]],
    target_profile: str,
) -> dict[str, list[Candidate]]:
    out = {}
    for name, cands in candidates.items():
        kept = [c for c in cands if _allowed_format(target_profile, name, c.fmt)]
        if kept:
            out[name] = kept
    return out


def _format_cli_choices() -> tuple[str, ...]:
    return tuple(sorted(set(fr.REGISTRY) | set(fr.FORMAT_ALIASES)))


# ---------------------------------------------------------------------------
# Visual encoder override
# ---------------------------------------------------------------------------
# Phase 1 visual-encoder support: the probe's text-only calibration does not
# exercise the visual tower, so per-Linear Fisher gradients for
# `model.visual.blocks.*` Linears are zero — the knapsack DP has no
# sensitivity signal to allocate on. Rather than let every visual Linear
# default to the cheapest format or go through stale passthrough, we accept
# a single uniform target format (`BF16`, `NVFP4`, or `FP8_DYNAMIC`) and assign
# every visual Linear to it. BF16 (the default) reproduces the previous
# passthrough behavior; quantized formats shrink the tower to quantized
# storage using the same math the body gets.
#
# Phase 2 (tracked separately) will replace this override with a real
# multimodal Fisher: load images + text, run full forward through the
# visual encoder → projector → body LM, capture per-Linear empirical Fisher
# gradients, and feed those into the allocator's closed-form Δloss. That
# requires a multimodal dataset loader, multimodal tokenizer wiring, and a
# probe path that doesn't strip the visual tower — none of which ship in
# Phase 1.
_VISUAL_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")


def _is_visual_linear(name: str) -> bool:
    """True when `name` refers to a Linear inside the visual encoder.

    Matches both the raw HF checkpoint form (`model.visual.blocks.*`) and
    the post-remap form (`visual.blocks.*`) so the override behaves the
    same regardless of which side of `profile.live_to_recipe_name` the
    allocator's stats dictionary landed on.
    """
    return bool(_VISUAL_PREFIX_RE.match(name))


def _mark_weight_only_nvfp4_stats(
    stats: Mapping[str, Mapping[str, object]],
    profile,
) -> dict[str, dict[str, object]]:
    """Stamp targets that the mixed exporter emits as stock W4A16.

    The exporter classifies a sidecar target by the same profile mapping:
    ``checkpoint_to_live_name(..., multimodal=False) is None`` means text
    calibration cannot cover that module, so stock NVFP4 drops
    ``input_global_scale``. Persist that decision in the accounting stats
    rather than re-deriving it from a visual/audio/MTP name heuristic.
    """
    mapper = getattr(profile, "checkpoint_to_live_name", None)
    if not callable(mapper):
        return {str(name): dict(entry) for name, entry in stats.items()}
    marked: dict[str, dict[str, object]] = {}
    for raw_name, raw_entry in stats.items():
        name = str(raw_name)
        entry = dict(raw_entry)
        try:
            live_name = mapper(name + ".weight", multimodal=False)
        except TypeError:
            live_name = mapper(name + ".weight")
        if live_name is None:
            entry[NVFP4_WEIGHT_ONLY_STATS_KEY] = True
        marked[name] = entry
    return marked


def apply_visual_format_override(
    assignment: dict[str, str],
    visual_format: str,
) -> dict[str, str]:
    """Force every visual-encoder Linear in `assignment` to `visual_format`.

    Called after the knapsack DP + fused-sibling promotion so the override
    wins even if the solver would have picked a different format per
    per-Linear sensitivity noise (which is meaningless for visual Linears
    under text-only calibration — see module comment above).

    `visual_format="BF16"` is a no-op when a visual Linear already has no
    allocator entry (the export's existing passthrough keeps it at BF16);
    we still write `BF16` into the returned assignment so the layer_config
    round-trip is explicit and downstream tooling (export, validate) has a
    uniform record of the decision.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if _is_visual_linear(name):
            out[name] = visual_format
    return out


def apply_mtp_format_override(
    assignment: dict[str, str],
    mtp_format: str,
) -> dict[str, str]:
    """Force MTP Linears to a recipe-level format.

    The production vLLM path currently validates main-target logits and keeps
    MTP in BF16 until speculative-decode acceptance is measured.  This override
    is applied after DP/fused-sibling promotion so a sensitive MTP projection
    cannot be accidentally quantized by the allocator.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if name.startswith("mtp."):
            out[name] = mtp_format
    return out


def _is_mtp_linear(name: str) -> bool:
    """True when `name` refers to an MTP Linear-like quantization target."""
    return str(name).startswith("mtp.")


def _find_candidate_for_format(
    candidates: dict[str, list[Candidate]],
    name: str,
    fmt: str,
) -> Candidate | None:
    """Return the scored candidate for `name` at canonical format `fmt`."""
    canonical = fr.get_format(fmt).name
    for cand in candidates.get(name, []):
        if fr.get_format(cand.fmt).name == canonical:
            return cand
    return None


def _validate_assignment_candidate_membership(
    assignment: dict[str, str],
    candidates: dict[str, list[Candidate]],
    *,
    fixed_chosen_candidates: dict[str, Candidate] | None = None,
) -> None:
    """Fail if promotion assigned a format no candidate row allowed."""
    available = {
        name: {fr.get_format(cand.fmt).name for cand in per_name}
        for name, per_name in candidates.items()
    }
    for name, cand in (fixed_chosen_candidates or {}).items():
        available.setdefault(name, set()).add(fr.get_format(cand.fmt).name)

    violations = []
    for name, fmt in sorted(assignment.items()):
        if name not in available:
            continue
        canonical = fr.get_format(fmt).name
        if canonical not in available[name]:
            violations.append((name, canonical, sorted(available[name])))
    if not violations:
        return

    sample = "\n  ".join(
        f"{name}: promoted to {fmt}, available={choices}"
        for name, fmt, choices in violations[:10]
    )
    raise SystemExit(
        "[alloc] serving-unit promotion assigned a format that was not "
        "present in the per-Linear candidate set. This would defer legality "
        "repair to export and can recreate mixed serving units. Sample:\n"
        f"  {sample}"
    )


# Role tokens used to bucket the bit-attribution report. Best-effort: anything
# unrecognized buckets as "unknown" rather than failing, so new architectures
# degrade gracefully instead of crashing a read-only diagnostic.
_BIT_ATTR_KNOWN_ROLES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",            # attention
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",           # MLP / fused
    "in_proj_a", "in_proj_b", "in_proj_ba", "in_proj_qkv",         # linear attn
    "in_proj_qkvz", "in_proj_z", "out_proj",
    "lm_head", "embed_tokens",                                     # head / embed
})


def _parse_role_from_qname(qname: str) -> str:
    """Best-effort role token for bit-attribution bucketing.

    Distinguishes routed-expert and shared-expert projections from dense MLP
    projections (so e.g. ``experts.7.down_proj`` does not collapse into the
    dense ``down_proj`` bucket). Returns ``"unknown"`` on any name that doesn't
    match a known pattern; this is a diagnostic, not a correctness path.
    """
    name = qname[4:] if qname.startswith("mtp.") else qname
    parts = name.split(".")
    if not parts:
        return "unknown"
    last = parts[-1]
    # Unpacked routed expert: ...experts.<N>.<role>
    if len(parts) >= 3 and parts[-2].isdigit() and "expert" in parts[-3]:
        return f"expert.{last}" if last in _BIT_ATTR_KNOWN_ROLES else "expert.unknown"
    # Packed routed expert (3D param, no per-expert index): ...experts.<role>
    if len(parts) >= 2 and parts[-2] == "experts":
        return f"expert.{last}" if last in _BIT_ATTR_KNOWN_ROLES else "expert.unknown"
    # Shared expert: ...shared_expert(s).<role>
    if len(parts) >= 2 and parts[-2] in ("shared_expert", "shared_experts"):
        return f"shared_expert.{last}" if last in _BIT_ATTR_KNOWN_ROLES else "shared_expert.unknown"
    if last in _BIT_ATTR_KNOWN_ROLES:
        return last
    return "unknown"


def _bit_attr_block_sort_key(block_id: str) -> tuple:
    """Sort layers.<N> numerically; non-layer blocks (lm_head, etc.) sort last."""
    match = re.search(r"layers\.(\d+)", str(block_id))
    if match:
        return (0, int(match.group(1)), str(block_id))
    return (1, 0, str(block_id))


def _build_bit_attribution(
    assignment_expanded: dict[str, str],
    candidates: dict[str, list[Candidate]],
    stats_entry_for,
    format_specs: dict[str, "fr.FormatSpec"],
    cb_serialization_context: CBSerializationContext | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Build (buckets, per_linear_rows, body_totals) for the bit-attribution
    report over the FINAL resolved body assignment.

    Read-only: derives everything from the already-resolved assignment,
    scored candidates, and probe stats. Visual / MTP Linears are excluded
    (auxiliary to the body budget). Bits are recovered the same way
    ``compute_achieved`` does, so the report's body bpp matches the artifact.
    Entries with no scored candidate (expanded fused-sibling members, experts
    priced at super-item granularity) report ``predicted_dloss=None`` rather
    than fabricating a value.
    """
    buckets: dict[tuple[str, str], dict] = {}
    per_linear: list[dict] = []
    body_tensor_bits = 0.0
    body_params = 0
    cb_assignment: dict[str, str] = {}
    cb_shapes: dict[str, tuple[int, ...]] = {}

    for name, fmt in assignment_expanded.items():
        if _is_visual_linear(name) or _is_mtp_linear(name):
            continue
        entry = stats_entry_for(name)
        n_params = None
        h_trace = None
        if isinstance(entry, dict):
            if entry.get("n_params") is not None:
                n_params = int(entry["n_params"])
            if entry.get("h_trace") is not None:
                h_trace = float(entry["h_trace"])

        cand = _find_candidate_for_format(candidates, name, fmt)
        bits = None
        pred_dloss = None
        if is_cb_format(fmt):
            if cb_serialization_context is None:
                raise ValueError(
                    f"bit attribution cannot price {name}={fmt} without "
                    "CBSerializationContext"
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    f"bit attribution cannot price {name}={fmt} without shape stats"
                )
            shape = _shape_from_stats(entry)
            item = cb_tensor_payload_breakdown(
                fmt,
                shape,
                qname=name,
                context=cb_serialization_context,
            )
            bits = 8.0 * int(item["tensor_payload_bytes"])
            cb_assignment[name] = fmt
            cb_shapes[name] = shape
            if cand is not None:
                pred_dloss = float(getattr(cand, "predicted_dloss", 0.0))
        elif cand is not None:
            bits = 8.0 * cand.memory_bytes
            pred_dloss = float(getattr(cand, "predicted_dloss", 0.0))
        elif isinstance(entry, dict):
            mem_map = entry.get("_memory_bytes_by_format")
            if isinstance(mem_map, dict) and fmt in mem_map:
                bits = 8.0 * mem_map[fmt]
            elif fmt in format_specs and n_params:
                shape = _shape_from_stats(entry)
                bits = format_specs[fmt].effective_bits_for_shape(shape) * n_params

        if bits is not None and fmt == "NVFP4" and isinstance(entry, dict):
            bits += 8.0 * nvfp4_global_sidecar_bytes(
                name,
                _shape_from_stats(entry),
                weight_only=bool(
                    entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                ),
            )

        bpp = (bits / n_params) if (bits is not None and n_params) else None
        block_id = block_id_from_qname(name)
        role = _parse_role_from_qname(name)
        per_linear.append({
            "qname": name,
            "block_id": block_id,
            "role": role,
            "format": fmt,
            "bits": bits,
            "bpp": bpp,
            "n_params": n_params,
            "h_trace": h_trace,
            "predicted_dloss": pred_dloss,
        })

        bucket = buckets.setdefault((block_id, role), {
            "block_id": block_id,
            "role": role,
            "n_linears": 0,
            "bits_total": 0.0,
            "n_params_total": 0,
            "format_counts": defaultdict(int),
            "sum_predicted_dloss": 0.0,
            "predicted_dloss_n": 0,
            "sum_h_trace": 0.0,
            "h_trace_n": 0,
        })
        bucket["n_linears"] += 1
        bucket["format_counts"][fmt] += 1
        if bits is not None:
            bucket["bits_total"] += bits
            body_tensor_bits += bits
        if n_params:
            bucket["n_params_total"] += n_params
            body_params += n_params
        if pred_dloss is not None:
            bucket["sum_predicted_dloss"] += pred_dloss
            bucket["predicted_dloss_n"] += 1
        if h_trace is not None:
            bucket["sum_h_trace"] += h_trace
            bucket["h_trace_n"] += 1

    bucket_list: list[dict] = []
    for bucket in buckets.values():
        n_params_total = bucket["n_params_total"]
        bucket_list.append({
            "block_id": bucket["block_id"],
            "role": bucket["role"],
            "n_linears": bucket["n_linears"],
            "bits_total": bucket["bits_total"],
            "n_params_total": n_params_total,
            "mean_bpp": (bucket["bits_total"] / n_params_total) if n_params_total else None,
            "format_counts": dict(bucket["format_counts"]),
            "sum_predicted_dloss": (
                bucket["sum_predicted_dloss"] if bucket["predicted_dloss_n"] else None),
            "predicted_dloss_coverage": f"{bucket['predicted_dloss_n']}/{bucket['n_linears']}",
            "sum_h_trace": bucket["sum_h_trace"] if bucket["h_trace_n"] else None,
        })

    bucket_list.sort(key=lambda r: (_bit_attr_block_sort_key(r["block_id"]), r["role"]))
    per_linear.sort(key=lambda r: (_bit_attr_block_sort_key(r["block_id"]), r["role"], r["qname"]))
    cb_shared_sidecar_bits = 0.0
    if cb_assignment:
        cb_payload = cb_assignment_payload_breakdown(
            cb_assignment,
            cb_shapes,
            context=cb_serialization_context,
        )
        cb_shared_sidecar_bits = 8.0 * int(
            cb_payload["codebook_sidecar_bytes"]
        )
    body_bits = body_tensor_bits + cb_shared_sidecar_bits
    totals = {
        "body_bits": body_bits,
        "body_tensor_payload_bits": body_tensor_bits,
        "body_shared_cb_sidecar_bits": cb_shared_sidecar_bits,
        "body_assignment_payload_bits": body_bits,
        "body_quantizable_params": body_params,
        "body_bits_per_param": (body_bits / body_params) if body_params else None,
        "n_body_linears": len(per_linear),
    }
    return bucket_list, per_linear, totals


def _write_bit_attribution_reports(
    json_path: str | None,
    csv_path: str | None,
    *,
    target_bits: float,
    achieved_bits: float,
    assignment_expanded: dict[str, str],
    candidates: dict[str, list[Candidate]],
    stats_entry_for,
    format_specs: dict[str, "fr.FormatSpec"],
    cb_serialization_context: CBSerializationContext | None = None,
) -> None:
    """Write the bit-attribution JSON / CSV and print a compact per-role rollup.

    No-op when neither output path is set."""
    if not json_path and not csv_path:
        return
    buckets, per_linear, totals = _build_bit_attribution(
        assignment_expanded,
        candidates,
        stats_entry_for,
        format_specs,
        cb_serialization_context,
    )

    if totals["body_quantizable_params"]:
        reconciled = float(totals["body_bits_per_param"])
        if not math.isclose(
            reconciled,
            float(achieved_bits),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                "bit attribution does not reconcile with final exact body "
                f"assignment bpp: report={reconciled}, final={achieved_bits}"
            )

    if json_path:
        payload = {
            "schema": "prismaquant.allocator.bit_attribution.v2",
            "target_bits": float(target_bits),
            "achieved_bits": float(achieved_bits),
            "body_bits_per_param": totals["body_bits_per_param"],
            "body_assignment_payload_bits": totals[
                "body_assignment_payload_bits"
            ],
            "body_tensor_payload_bits": totals["body_tensor_payload_bits"],
            "body_shared_cb_sidecar_bits": totals[
                "body_shared_cb_sidecar_bits"
            ],
            "reconciliation": (
                "body_assignment_payload_bits = body_tensor_payload_bits + "
                "body_shared_cb_sidecar_bits"
            ),
            "body_quantizable_params": totals["body_quantizable_params"],
            "n_body_linears": totals["n_body_linears"],
            "buckets": buckets,
        }
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[alloc] bit attribution (per block/role) → {path}", flush=True)

    if csv_path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "qname", "block_id", "role", "format",
                "bits", "bpp", "n_params", "h_trace", "predicted_dloss",
            ])
            for row in per_linear:
                writer.writerow([
                    row["qname"], row["block_id"], row["role"], row["format"],
                    "" if row["bits"] is None else f"{row['bits']:.6g}",
                    "" if row["bpp"] is None else f"{row['bpp']:.6g}",
                    "" if row["n_params"] is None else row["n_params"],
                    "" if row["h_trace"] is None else f"{row['h_trace']:.6g}",
                    "" if row["predicted_dloss"] is None else f"{row['predicted_dloss']:.6g}",
                ])
        print(f"[alloc] bit attribution (per Linear) → {path}", flush=True)

    # Compact stdout rollup by role across all blocks — the at-a-glance answer
    # to "where did the budget go?" without opening the JSON.
    role_roll: dict[str, dict] = {}
    for row in per_linear:
        agg = role_roll.setdefault(row["role"], {
            "n": 0, "bits": 0.0, "params": 0, "fmts": defaultdict(int)})
        agg["n"] += 1
        agg["fmts"][row["format"]] += 1
        if row["bits"] is not None:
            agg["bits"] += row["bits"]
        if row["n_params"]:
            agg["params"] += row["n_params"]
    print("[alloc] bit attribution by role (body only):", flush=True)
    for role in sorted(role_roll, key=lambda r: -(role_roll[r]["bits"])):
        agg = role_roll[role]
        mean_bpp = (agg["bits"] / agg["params"]) if agg["params"] else float("nan")
        fmt_mix = " ".join(
            f"{fmt}:{n}" for fmt, n in sorted(agg["fmts"].items(), key=lambda kv: -kv[1]))
        print(f"    {role:>22}  n={agg['n']:>4}  mean_bpp={mean_bpp:6.3f}  [{fmt_mix}]",
              flush=True)


def discover_visual_linear_stats_from_source(
    model_path: str,
    *,
    strict: bool = False,
) -> dict[str, dict[str, object]]:
    """Scan source safetensors for visual-Linear names and exact shapes.

    Returned keys are the basename (``.weight`` stripped); values carry the
    same ``n_params``/``in_features``/``out_features`` fields as probe stats.
    Keeping shapes is required for exact Pareto byte pricing when the text-only
    probe did not instantiate the visual tower.

    The probe's text-only staging strips the visual tower, so visual
    Linears never appear in the probe or cost pickles. This helper lets
    the allocator emit a layer_config entry for them anyway when
    `--visual-format` is non-BF16 — the exporter can then quantize each
    of them uniformly under the requested format. Without this scan, the
    allocator has no way to enumerate visual Linear names (there is no
    in-memory visual module at allocation time).
    """
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    candidates: list[tuple[str, tuple[int, ...], str]] = []
    source_tensor_names: set[str] = set()
    scan_errors: list[str] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        source_tensor_names.update(str(name) for name in wm)
        # Index file carries only names, not shapes. We need to open each
        # referenced shard once to read rank.
        from collections import defaultdict as _dd
        by_shard: dict[str, list[str]] = _dd(list)
        for key, shard in wm.items():
            if not key.endswith(".weight"):
                continue
            if not _VISUAL_PREFIX_RE.match(key):
                continue
            by_shard[shard].append(key)
        try:
            from safetensors import safe_open
        except ImportError as exc:
            if strict:
                raise RuntimeError(
                    "exact visual source discovery requires safetensors"
                ) from exc
            return {}
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                scan_errors.append(
                    f"missing indexed shard {shard!r} for {keys[:4]}"
                )
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        tensor_slice = sf.get_slice(k)
                        shape = tuple(tensor_slice.get_shape())
                        dtype = str(tensor_slice.get_dtype()).upper()
                    except Exception as exc:
                        scan_errors.append(f"{shard}:{k}: {exc}")
                        continue
                    candidates.append((k, shape, dtype))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError as exc:
            if strict:
                raise RuntimeError(
                    "exact visual source discovery requires safetensors"
                ) from exc
            return {}
        import os as _os
        if not src.exists():
            if strict:
                raise FileNotFoundError(
                    f"visual source checkpoint does not exist: {src}"
                )
            return {}
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                source_tensor_names.update(str(key) for key in sf.keys())
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _VISUAL_PREFIX_RE.match(k):
                        continue
                    try:
                        tensor_slice = sf.get_slice(k)
                        shape = tuple(tensor_slice.get_shape())
                        dtype = str(tensor_slice.get_dtype()).upper()
                    except Exception as exc:
                        scan_errors.append(f"{f}:{k}: {exc}")
                        continue
                    candidates.append((k, shape, dtype))

    if strict and scan_errors:
        raise RuntimeError(
            "exact visual source discovery was incomplete: "
            + "; ".join(scan_errors[:8])
        )

    # Only rank-2 weights are Linear-like; conv1d / norms / biases are
    # kept at BF16 passthrough regardless of --visual-format.
    # Additionally, blacklist known rank-2 tensors that live in
    # `nn.Parameter` / `nn.Embedding` modules (NOT `nn.Linear`), which
    # the compressed-tensors loader in vLLM cannot consume. Example:
    # `model.visual.pos_embed.weight` is an Embedding-like learned
    # parameter with shape (num_pos, hidden) — rank-2 but NOT a Linear.
    # Quantizing it produces `pos_embed.input_global_scale` etc. which
    # vLLM's VL runtime rejects with `KeyError: pos_embed.input_global_scale`
    # because its `model.visual.pos_embed` is a bare Parameter, not a
    # quantizable Linear module.
    _NON_LINEAR_RE = re.compile(
        r"(?:^|\.)("
        r"pos_embed"            # positional embedding (nn.Parameter/Embedding)
        r"|rotary_emb"          # rotary pos embed cache
        r")(?:\.|$)"
    )
    out: dict[str, dict[str, object]] = {}
    for name, shape, dtype in candidates:
        if len(shape) != 2:
            continue
        if _NON_LINEAR_RE.search(name):
            continue
        qname = name[:-len(".weight")] if name.endswith(".weight") else name
        out_features, in_features = (int(shape[0]), int(shape[1]))
        entry = {
            "n_params": out_features * in_features,
            "out_features": out_features,
            "in_features": in_features,
            NVFP4_WEIGHT_ONLY_STATS_KEY: True,
            "source_dtype": (
                "bf16" if dtype == "BF16"
                else "fp8" if dtype.startswith("F8_")
                else "fp16" if dtype == "F16"
                else "fp32" if dtype == "F32"
                else dtype.lower()
            ),
            "has_fp8_scale": any(
                f"{qname}.{suffix}" in source_tensor_names
                for suffix in ("weight_scale_inv", "weight_scale")
            ),
        }
        previous = out.setdefault(qname, entry)
        if previous != entry:
            raise RuntimeError(
                f"visual Linear {qname!r} appears with conflicting source "
                f"shapes: {previous} versus {entry}"
            )
    return dict(sorted(out.items()))


def discover_visual_linears_from_source(model_path: str) -> list[str]:
    """Backwards-compatible name-only view of source visual Linears."""
    return list(discover_visual_linear_stats_from_source(model_path))


def validate_source_visual_passthrough_contract(
    source_visual_stats: Mapping[str, Mapping[str, object]],
    visual_format: str,
) -> None:
    """Reject a source-only visual passthrough whose dtype does not match."""
    canonical = fr.get_format(visual_format).name
    if canonical not in PASSTHROUGH_SOURCE_REQUIREMENTS:
        return
    mismatched = sorted(
        name
        for name, entry in source_visual_stats.items()
        if (
            not _passthrough_source_ok(
                canonical,
                str(entry.get("source_dtype", "unknown")),
            )
            or (
                canonical == "FP8_SOURCE"
                and not bool(entry.get("has_fp8_scale", False))
            )
        )
    )
    if not mismatched:
        return
    sample = [
        (
            name,
            source_visual_stats[name].get("source_dtype", "unknown"),
            bool(source_visual_stats[name].get("has_fp8_scale", False)),
        )
        for name in mismatched[:8]
    ]
    raise ValueError(
        f"--visual-format={canonical} is a passthrough contract, but "
        "source-only visual Linears have incompatible source dtype/scale "
        "contracts "
        f"(sample={sample}). The exporter copies these tensors verbatim, so "
        "pricing them as a re-encode would make the assignment budget false. "
        "Choose an explicit re-encoding format or use a matching source."
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def validate_default_profile_format_menu(
    model_profile,
    specs_sorted,
    *,
    allow_default_profile: bool = False,
) -> None:
    """Reject multi-format menus when only DefaultProfile was resolved."""
    from .model_profiles import DefaultProfile

    using_default_profile = isinstance(model_profile, DefaultProfile)
    distinct_fmt_names = sorted({s.name for s in specs_sorted})
    if (
        using_default_profile
        and len(distinct_fmt_names) > 1
        and not allow_default_profile
    ):
        raise SystemExit(
            "[alloc] ERROR: multi-format menu "
            f"({distinct_fmt_names}) resolved to DefaultProfile (no probe "
            "meta['model'] / --model-override, or the model architecture is "
            "not registered) -> DefaultProfile only enforces its fallback "
            "fused groups (qkv_proj/gate_up_proj and DeltaNet "
            "in_proj_ba/in_proj_qkvz). It cannot guarantee unknown "
            "architecture-specific coherence or packed-MoE expert uniformity, which "
            "risks an unservable or silently-corrupt artifact. Pass "
            "--model-override <model> so detect_profile resolves the real "
            "profile, or --allow-default-profile to proceed anyway."
        )


def incomplete_fused_group_dp_exclusions(
    stats: dict,
    costs: dict,
    model_profile,
    allocation_excluded=(),
) -> list[str]:
    """Linears to exclude from DP because their fused group is incomplete.

    Excluding them from the mutable body assignment leaves them absent from
    ``layer_config``; export then keeps those present-but-incomplete fused
    siblings as BF16 passthrough instead of producing missing scale tensors.
    """
    from .decision_units import incomplete_fused_group_members

    incomplete_members = incomplete_fused_group_members(
        set(stats) | set(costs), model_profile)
    return sorted(incomplete_members - set(allocation_excluded))


def tied_lm_head_dp_exclusions(
    stats: dict,
    costs: dict,
    model_profile,
    model_path: str | None,
    allocation_excluded=(),
) -> list[str]:
    """Linears to exclude from DP because they are a tied LM head.

    When the checkpoint declares `tie_word_embeddings` and ships no head
    tensor, `lm_head.weight` IS the input embedding — one storage, one set
    of source bytes. Quantizing it would quantize every token embedding
    (which the pipeline prices in the non-quantizable floor and no probe /
    cost measurement observes), and there is no `lm_head.weight` in the
    source manifest to move out of that floor. So the tie makes the head
    structurally passthrough-only, independent of the serving profile's
    pins: this exclusion deliberately ignores `--allow-pinned lm_head`,
    which exists to trade an *independent* head's bytes for quality.

    Post-fix probes emit no tied-head row at all (the shard schedule skips
    it); this also covers probes built before that fix.
    """
    if not model_path:
        return []
    from .tied_embeddings import lm_head_is_tied_alias

    if not lm_head_is_tied_alias(model_path, profile=model_profile):
        return []
    excluded = {
        name for name in (set(stats) | set(costs))
        if is_lm_head_name(name, model_profile)
    }
    return sorted(excluded - set(allocation_excluded))


def validate_final_serving_promotion_noop(
    before: dict[str, str],
    after: dict[str, str],
) -> None:
    if before == after:
        return
    changed = [
        (name, before.get(name), after.get(name))
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    ]
    sample = changed[:8]
    raise SystemExit(
        "[alloc] ERROR: final serving-unit promotion changed the emitted "
        "assignment after achieved_bits/Delta-loss were computed; metrics are "
        f"stale. Changed {len(changed)} entries, sample={sample}. Move this "
        "coupling before solve_with_promotion or recompute accounting."
    )


def renormalize_probe_fisher(stats: dict, meta: dict, *,
                             allow_legacy: bool = False) -> int | None:
    """Recompute h_trace/h_w2_sum/h_trace_per_expert from the stored RAW
    accumulators with the one correct shared denominator: the global calib
    token count.

    Older incremental probes finalized `h_trace = h_trace_raw /
    n_tokens_seen` PER ROW, where per-expert Linears only count their
    ROUTED tokens — inflating a rarely-routed expert's Fisher by
    (global/routed), i.e. inverted importance weighting. The typical
    inflation is ~n_experts/top_k (≈32x on a 256-expert top-8 model);
    the degenerate 1-routed-token case reaches ~33,000x at a 32k-token
    calibration. Dense rows saw exactly the global count, so their values
    are unchanged; probes written by the fixed finalize renormalize to
    identical values — the recompute is idempotent (see PER-ROW STAMPS).

    PER-ROW STAMPS WIN. A row that already carries ``h_trace_norm_tokens``
    was written by a fixed finalize (the field was introduced with it) and
    is already divided by the global count of the pass that PRODUCED it —
    which is not always this probe's meta count. `incremental_probe` merges
    the multimodal visual pass (finalized at ``n_samples x max_text_len``,
    e.g. 8x128) into the body probe, and `merge_probe_pickles` keeps the
    FIRST shard's meta (the body's, e.g. 32x1024), so recomputing every row
    from the meta count would rescale the visual rows by ~32x away from
    what their writer computed — and would not be idempotent. Honouring the
    row stamp keeps each row on its own writer's denominator and makes the
    recompute a true no-op for any probe written by the fixed finalize.

    Returns the probe-wide token count (the meta fallback used for
    unstamped rows), or None when nothing could be renormalized. A row
    with raw accumulators, no stamp, and no usable meta count is a HARD
    ERROR (SystemExit) — silently allocating on routed-token-normalized
    Fisher mis-spends the bit budget; `allow_legacy=True`
    (--allow-legacy-fisher-norm) downgrades it to a warning and keeps the
    probe's stored h_trace values.
    """
    meta = meta or {}
    meta_tokens = int(meta.get("fisher_norm_tokens", 0) or 0)
    if meta_tokens <= 0:
        meta_tokens = (int(meta.get("nsamples", 0) or 0)
                       * int(meta.get("seqlen", 0) or 0))

    renormed = 0
    unresolvable = 0
    denominators: dict[int, int] = {}
    for entry in stats.values():
        if not isinstance(entry, dict) or "h_trace_raw" not in entry:
            continue
        # The row's own writer-stamped denominator, else the probe-wide one.
        row_tokens = int(entry.get("h_trace_norm_tokens", 0) or 0)
        if row_tokens <= 0:
            row_tokens = meta_tokens
        if row_tokens <= 0:
            unresolvable += 1
            continue
        entry["h_trace"] = float(entry["h_trace_raw"]) / row_tokens
        if "h_w2_sum_raw" in entry:
            entry["h_w2_sum"] = (
                float(entry["h_w2_sum_raw"]) / row_tokens)
        per_raw = entry.get("h_trace_per_expert_raw")
        if per_raw is not None:
            entry["h_trace_per_expert"] = [
                float(v) / row_tokens for v in per_raw]
        # Keep the probe-side stamp truthful on legacy pickles.
        entry["h_trace_norm_tokens"] = row_tokens
        denominators[row_tokens] = denominators.get(row_tokens, 0) + 1
        renormed += 1

    if unresolvable:
        msg = (
            "probe rows carry raw Fisher accumulators but no usable token "
            "count: neither a per-row h_trace_norm_tokens stamp nor "
            f"meta.fisher_norm_tokens / nsamples+seqlen ({unresolvable} of "
            f"{len(stats)} rows). h_trace cannot be renormalized by the "
            "global calib token count; legacy per-row normalization inflates "
            "routed-expert rows by (global/routed) — typically "
            "~n_experts/top_k — and the allocator would mis-spend the bit "
            "budget on rarely-routed experts. Re-run the probe with current "
            "code (it stamps meta.fisher_norm_tokens), or pass "
            "--allow-legacy-fisher-norm to proceed with the stored legacy "
            "values anyway.")
        if not allow_legacy:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        print(f"[alloc] WARNING (--allow-legacy-fisher-norm): {msg}",
              flush=True)

    if renormed:
        mix = ", ".join(f"{n}x{cnt} rows"
                        for n, cnt in sorted(denominators.items()))
        print(f"[alloc] Fisher renormalized: {renormed}/{len(stats)} rows "
              f"→ h_trace_raw / global calib tokens [{mix}] "
              "(per-expert routed-count normalization corrected)", flush=True)
        if meta_tokens > 0:
            return meta_tokens
        # Meta-less probe carried entirely by row stamps: report the
        # denominator the most rows share (the body pass').
        return max(denominators.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return None


_FISHER_CAP_ROLE_RE = re.compile(r"layers\.(\d+)\.[a-z_]+\.([a-z_]+)$")


def _fisher_cap_role(qname: str) -> str | None:
    """Role bucket for the robust-Fisher clip, or ``None`` to skip the row.

    Deliberately the REFERENCE TOOL's grouping
    (``/home/rob/dq-runs/robust_fisher_clip.py``), not the wider
    ``_parse_role_from_qname`` used for bit attribution: it requires exactly
    one container segment between ``layers.<N>.`` and the leaf, so dense
    attention/MLP projections bucket by their leaf role
    (``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/``up_proj``/
    ``down_proj``) and everything else — packed and unpacked MoE experts,
    shared experts, MTP/visual sidecars, non-``layers`` names — is skipped.
    That is the grouping the 4B research result was measured under; widening
    it would change the medians and is a separate, unmeasured lever.
    """
    m = _FISHER_CAP_ROLE_RE.search(qname)
    return m.group(2) if m else None


def clip_probe_fisher_outliers(stats: dict, meta: dict | None = None, *,
                               cap_multiplier: float | None = None
                               ) -> dict | None:
    """OPT-IN robust-Fisher clip: cap each row's ``h_trace`` at
    ``K x median(h_trace)`` over its role bucket.

    Off unless ``PRISMAQUANT_FISHER_CAP_MULTIPLIER`` is set (or
    ``cap_multiplier`` is passed explicitly); when off this is a byte-identical
    no-op and returns ``None``. Research lever — a per-role h_trace clip at
    K=3 measured ~5% better WikiText PPL at 6.0 bpp on Qwen3-4B (2026-05-19),
    never promoted to a default because it has no served A/B.

    Rationale: ``predicted_dloss = 1/2 h_trace MSE`` is linear in ``h_trace``,
    so a handful of heavy-tailed Fisher rows can capture the DP's whole bit
    budget. Clipping the tail bounds that leverage without touching the rest
    of the distribution (values below the cap are unchanged).

    ``h_w2_sum`` is rescaled by the same ratio so the derived cost math stays
    consistent with the clipped scalar, exactly as the reference tool does.
    The raw accumulators (``h_trace_raw`` / ``h_w2_sum_raw``) are NOT touched
    — they remain the source of truth for renormalization, so this must run
    AFTER :func:`renormalize_probe_fisher`, never before.

    Returns a summary dict (``cap_multiplier``, ``role_median``, ``role_cap``,
    ``n_clipped``, ``n_considered``) when active, else ``None``.
    """
    import os
    import statistics

    if cap_multiplier is None:
        raw = os.environ.get("PRISMAQUANT_FISHER_CAP_MULTIPLIER")
        if raw is None or not str(raw).strip():
            return None
        try:
            cap_multiplier = float(raw)
        except (TypeError, ValueError):
            raise SystemExit(
                "[alloc] ERROR: PRISMAQUANT_FISHER_CAP_MULTIPLIER must be a "
                f"float multiple of the per-role median, got {raw!r}.")
    K = float(cap_multiplier)
    if not (K > 0.0) or not math.isfinite(K):
        raise SystemExit(
            "[alloc] ERROR: PRISMAQUANT_FISHER_CAP_MULTIPLIER must be a "
            f"finite value > 0 (it is a multiple of the role median), got {K}.")

    by_role: dict[str, list[float]] = defaultdict(list)
    for qname, entry in stats.items():
        if not isinstance(entry, dict):
            continue
        role = _fisher_cap_role(qname)
        if role is None:
            continue
        by_role[role].append(float(entry.get("h_trace", 0.0) or 0.0))

    role_median = {r: statistics.median(v) for r, v in by_role.items() if v}
    role_cap = {r: K * m for r, m in role_median.items()}

    n_clipped = 0
    n_considered = 0
    for qname, entry in stats.items():
        if not isinstance(entry, dict):
            continue
        role = _fisher_cap_role(qname)
        if role is None:
            continue
        cap = role_cap.get(role)
        if cap is None:
            continue
        n_considered += 1
        old = float(entry.get("h_trace", 0.0) or 0.0)
        if old <= cap:
            continue
        entry["h_trace"] = cap
        old_w2 = entry.get("h_w2_sum")
        if old_w2 is not None and old > 0:
            entry["h_w2_sum"] = float(old_w2) * (cap / old)
        n_clipped += 1

    summary = {
        "schema": "prismaquant.robust_fisher_clip.v1",
        "cap_multiplier": K,
        "role_median": role_median,
        "role_cap": role_cap,
        "n_clipped": n_clipped,
        "n_considered": n_considered,
    }
    if isinstance(meta, dict):
        meta.setdefault("clip_history", []).append(summary)
    print(f"[alloc] robust Fisher clip ACTIVE (research lever, "
          f"PRISMAQUANT_FISHER_CAP_MULTIPLIER={K:g}): clipped {n_clipped} of "
          f"{n_considered} role-bucketed rows at K x per-role median "
          f"({len(role_median)} roles)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument(
        "--accept-research-cost-table",
        action="store_true",
        help="Explicitly accept a table stamped as the sanctioned study-grade "
             "assembled-segments lane. This never weakens production CB "
             "provenance checks for unstamped tables.",
    )
    ap.add_argument(
        "--research-cost-base",
        default=None,
        help="With --accept-research-cost-table, production v2 base pickle "
             "to assemble into --costs before allocation.",
    )
    ap.add_argument(
        "--research-cost-segments-dir",
        default=None,
        help="With --accept-research-cost-table, complete layer_*.pkl store "
             "to assemble over --research-cost-base before allocation.",
    )
    ap.add_argument("--model-override", default=None,
                    help="Override the model path stored in probe.pkl's meta. "
                         "Useful when re-running allocator against a probe "
                         "whose container-side paths no longer exist (e.g., "
                         "the original source was at /src/qwen36 in a prior "
                         "container run but is now only accessible via a "
                         "different mount). Overrides both profile detection "
                         "and visual-Linear source discovery.")
    ap.add_argument("--allow-default-profile", action="store_true",
                    help="Permit a multi-format allocation to run on "
                         "DefaultProfile when no model path is available "
                         "(probe meta['model'] unset and no --model-override). "
                         "DefaultProfile enforces only fallback fused groups "
                         "(qkv_proj/gate_up_proj and DeltaNet "
                         "in_proj_ba/in_proj_qkvz), so unknown "
                         "architecture-specific merged columns or packed-MoE "
                         "expert constraints may produce an unservable or "
                         "silently-corrupt artifact. Off by default: "
                         "the allocator hard-errors instead and asks for "
                         "--model-override.")
    ap.add_argument("--allow-legacy-fisher-norm", action="store_true",
                    help="Permit allocation from a probe whose rows carry "
                         "neither a per-row h_trace_norm_tokens stamp nor "
                         "meta fisher_norm_tokens / nsamples+seqlen, i.e. whose "
                         "h_trace cannot be renormalized by the global calib "
                         "token count. Such probes carry the legacy per-row "
                         "normalization that inflates routed-expert Fisher "
                         "by (global/routed) — typically ~n_experts/top_k. "
                         "Off by default: the allocator hard-errors and asks "
                         "for a re-probe. Use only to reproduce historical "
                         "allocations.")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--target-disk-gb", type=float, default=None,
                    help="Fit-the-card ship selection: instead of --target-bits, "
                         "the card sets the CONSTRAINT and predicted Δloss the "
                         "OBJECTIVE — among allocations whose exact tensor-data "
                         "payload plus --artifact-overhead-reserve-bytes fits "
                         "this many decimal GB, ship the one with the LOWEST "
                         "predicted "
                         "Δloss (ties -> larger footprint; more bits is not "
                         "monotonically better, so filling the card is a proxy, "
                         "not the objective). Bisects the sub-second DP between "
                         "Pareto grid rungs to search denser fitting "
                         "allocations. The exporter then measures all regular "
                         "files recursively and fails closed against this same "
                         "budget. Overrides --target-bits for the emitted "
                         "layer_config, and writes selection.json (objective, "
                         "search ceiling, full ratchet trace) beside "
                         "--pareto-csv. Needs the source model path (probe "
                         "meta.model / --model-override) to size the "
                         "non-quantizable floor (lm_head/embed/norms). A "
                         "measured fixed head (--lm-head-format FP8_E4M3) "
                         "lowers the floor without putting head parameters "
                         "into reported body bpp; --allow-pinned lm_head is "
                         "the separate research/DP path.")
    ap.add_argument(
        "--artifact-overhead-reserve-bytes",
        type=int,
        default=None,
        help=(
            "Required with --target-disk-gb. Conservative upper bound for "
            "all bytes outside safetensors tensor-data spans (container "
            "headers, JSON/config, tokenizer/processor assets and other "
            "regular files). Selection gates tensor payload + this reserve; "
            "the exporter then stats every regular file recursively and "
            "fails closed against the original whole-artifact budget."
        ),
    )
    ap.add_argument(
        "--exclude-source-prefix",
        action="append",
        default=None,
        metavar="PREFIX",
        help=(
            "Repeatable. Declare that this artifact does NOT ship the source "
            "tensors under PREFIX, because they ship as a separate artifact. "
            "Byte accounting otherwise counts every unassigned source tensor "
            "at source precision (correct for a verbatim tensor, wrong for "
            "one that is not here at all), which UNDER-fills a byte budget by "
            "the excluded mass and is invisible downstream because the "
            "artifact still comes in under budget. Live case: DSv4-Flash "
            "ships its draft head as its own directory, so the target "
            "artifact passes --exclude-source-prefix mtp. and the 10.863 GB "
            "of MTP spans leave the floor. Exact (resolved against the "
            "per-tensor source manifest, each span charged at most once); a "
            "prefix matching nothing is a hard error."
        ),
    )
    ap.add_argument(
        "--cb-scale-coding",
        choices=("v1", "two_tier"),
        default=None,
        help=(
            "Exact CB producer scale layout used for candidate and artifact "
            "bytes. Required when the body/auxiliary formats contain a CB "
            "rung; production "
            "passes two_tier. v1 is explicit legacy-write reproduction only."
        ),
    )
    ap.add_argument(
        "--cb-codebook-source",
        choices=("lattice", "learned"),
        default=None,
        help=(
            "Exact CB sidecar sharing policy. Required when any body/auxiliary "
            "format is CB so codebook identity/bytes cannot be guessed. The "
            "production CLI currently accepts lattice only; learned is "
            "rejected until every stage consumes one immutable value-bearing "
            "bundle."
        ),
    )
    ap.add_argument(
        "--cb-codebook-source-scope",
        choices=("none", "fp8", "all"),
        default=None,
        help=(
            "Family-scoped source contract. none is the byte-identical "
            "all-lattice default; fp8 keeps NVFP4 lattice and binds FP8_CB "
            "to immutable learned cells; all is research-only/NO-GO on NVFP4."
        ),
    )
    ap.add_argument(
        "--cb-codebook-bundle",
        default=None,
        help=(
            "Immutable value-bearing .pqcb used by every learned render "
            "stage. Required when the effective source scope is not none."
        ),
    )
    ap.add_argument(
        "--cb-codebook-digests",
        default=None,
        help=(
            "Reserved learned-CB digest manifest. The production CLI rejects "
            "this digest-only contract because it does not supply codebook "
            "values; direct research accounting APIs still accept digests."
        ),
    )
    ap.add_argument(
        "--cb-scale-sweep",
        choices=("0", "1"),
        default=None,
        help="Exact CB scale-search contract; required with CB formats.",
    )
    ap.add_argument(
        "--cb-scale-sweep-scope",
        choices=("none", "nvfp4", "fp8", "all"),
        default=None,
        help=(
            "Optional per-family scale-search scope. Unset preserves the "
            "legacy --cb-scale-sweep bool and its byte-identical stamp."
        ),
    )
    ap.add_argument(
        "--cb-ldlq",
        choices=("0", "1"),
        default=None,
        help="Exact CB feedback-assignment contract; required with CB formats.",
    )
    ap.add_argument(
        "--cb-ldlq-scope",
        choices=("none", "nvfp4", "all"),
        default=None,
        help=(
            "Which CB family the exporter will LDLQ. Authoritative over "
            "--cb-ldlq when given. The stamp must match what the export "
            "actually renders, or the per-tensor identity preflight fails: "
            "'nvfp4' means NVFP4_CB is LDLQ and FP8_CB stays raw. LDLQ is "
            "byte-neutral, so this changes the recorded contract, not bytes."
        ),
    )
    ap.add_argument(
        "--cb-minchain",
        choices=("0", "1"),
        default="0",
        help=(
            "Exact CB monotone min-chain encoder contract (default: 0). "
            "The production pipeline passes this explicitly."
        ),
    )
    ap.add_argument(
        "--cb-encode-tier",
        choices=("fast", "balanced", "max"),
        default=None,
        help="Resolved CB encoder tier; required with CB formats.",
    )
    ap.add_argument(
        "--cb-col-weights",
        default=None,
        help=(
            "Exact imatrix pickle used by CB cost/cache/export. Required "
            "when any allocated or fixed auxiliary format is CB; allocator "
            "validates its value-bearing render identity before solving."
        ),
    )
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--allow-pinned", default="",
                    help="Comma-separated qname substrings to UN-pin (e.g. "
                         "'lm_head') so the allocator chooses their format by "
                         "budget-value instead of force-excluding them as BF16. "
                         "Requires the cost file to carry candidates for them "
                         "(aura_cost --include-lm-head) + the probe their "
                         "n_params. Empty = current behavior (backwards-compat). "
                         "Shipping a quantized pinned name also needs cache "
                         "render + export packing + vLLM serving support "
                         "(lm_head: yes, native lane; embed_tokens: yes on the "
                         "GRIDBOOK lane only, via the quantized_embedding "
                         "declaration -- the native compressed-tensors "
                         "embedding path accepts weight-only INT and RAISES "
                         "for FP8/NVFP4, so a stock config_groups entry does "
                         "not mis-serve, it refuses to load).")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
    ap.add_argument("--knee-refine-tol", type=float, default=0.03,
                    help="Golden-section knee refinement tolerance in bpp "
                         "(default 0.03). The coarse Pareto grid only lands the "
                         "Kneedle on a grid point; refinement re-solves the DP "
                         "inside the knee bracket to pin it to this resolution.")
    ap.add_argument("--knee-refine", action="store_true",
                    help="Opt-in: golden-section refine the (diagnostic) "
                         "log-error knee, re-solving the DP at interior budgets "
                         "and folding the samples into the curve + Pareto "
                         "manifest. Default OFF: the AURA RD curve is log-linear "
                         "(see the rd_curve diagnostic) so the knee is "
                         "axis-dependent — ship by --target-disk-gb or measured "
                         "saturation instead. NB enabling this in "
                         "validated-surrogate mode adds real-GPU KL passes for "
                         "the injected interior assignments.")
    ap.add_argument(
        "--applicability-report",
        default=None,
        help=(
            "Optional JSON sidecar for candidates masked before DP by source, "
            "profile, divisibility, or runtime kernel-shape constraints. "
            "Defaults to format_applicability.json beside --pareto-csv."
        ),
    )
    ap.add_argument(
        "--pareto-output-dir",
        default=None,
        help=(
            "Optional directory where each feasible Pareto point is written "
            "as a per-Linear assignment JSON suitable for "
            "kl_sensitivity_probe --seed-assignment."
        ),
    )
    ap.add_argument("--no-fused-promote", action="store_true",
                    help="Skip fused-projection sibling promotion")
    ap.add_argument("--no-fused-aggregation", action="store_true",
                    help="Disable pre-DP aggregation of fused siblings "
                         "(qkv_proj / gate_up_proj). Falls back to the "
                         "legacy promote_fused post-pass with tightening "
                         "retries. Pre-aggregation is strictly better "
                         "for hitting the target bit budget exactly on "
                         "dense models; use this flag only for "
                         "back-compat experiments.")
    ap.add_argument("--no-packed-aggregation", action="store_true",
                    help="Disable pre-DP aggregation of packed-MoE expert "
                         "serving groups into whole-group DP units. Falls "
                         "back to per-row DP pricing with the "
                         "promote_serving_units post-pass repairing group "
                         "coherence. Aggregation prices the DP and the "
                         "serving constraint identically; use this flag "
                         "only for back-compat experiments.")
    ap.add_argument("--packed-role-split", action="store_true",
                    help="Split each packed-MoE expert serving group into "
                         "TWO DP units per layer: gate+up projections and "
                         "down projections (default: one per-layer-uniform "
                         "unit). Hard-errors unless the resolved serving "
                         "profile declares supports_per_role_expert_schemes "
                         "(e.g. gguf: GGUF stacks expert tensors per "
                         "projection, so each role can carry its own type; "
                         "vLLM packed-MoE loads one scheme per FusedMoE "
                         "layer). The split threads through the profile "
                         "view, so serving promotion stays consistent with "
                         "the DP units.")
    ap.add_argument("--enforce-family-coherence", action="store_true",
                    help="Error (instead of warn) if the format set contains "
                         "multiple candidates for the same bit tier (e.g. "
                         "NVFP4 and MXFP4 both at 4 bits)")
    ap.add_argument("--bit-precision", type=float, default=0.0001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.0001 → ~50000 bins). "
                         "Measured on MiniMax-M2.7: going from 0.001 to 0.0001 "
                         "cuts predicted Δloss ~10%% at the same bit budget. "
                         "Coarser values (0.01) leave 40%% on the table.")
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/numpy threads for DP (0 = default)")
    ap.add_argument("--target-profile",
                    choices=serving_profile_names(),
                    default=None,
                    help="Serving/backend constraint profile loaded from "
                         "prismaquant/serving_profile_specs. Defaults to "
                         "the detected model profile's configured serving "
                         "profile, or research when none is declared.")
    ap.add_argument("--target-profile-default",
                    default="research",
                    help="Fallback serving profile when neither "
                         "--target-profile nor the architecture's "
                         "spec.default_serving_profile names one. The "
                         "production orchestrator passes vllm_packed_moe; "
                         "the bare-CLI default stays `research` (re-vet R11).")
    ap.add_argument("--calibration", default=None,
                    help="Optional path to a JSON containing "
                         "'calibrated_gains[fmt] = α_fmt'. When present, "
                         "the per-(layer, format) predicted Δloss "
                         "is multiplied by α_fmt before the DP runs.")
    ap.add_argument("--overshoot-tolerance", type=float, default=0.01,
                    help="Maximum allowed overshoot (bits/param) of the "
                         "achieved budget over the requested target after "
                         "fused-sibling promotion. The DP is re-run with a "
                         "tightened target until overshoot is within tol.")
    ap.add_argument("--visual-format",
                    choices=_format_cli_choices(),
                    default="BF16",
                    help="Uniform format for all visual-encoder Linears "
                         "(`model.visual.blocks.*`). Phase 1 fallback: "
                         "assigned to every visual Linear when "
                         "--visual-sensitivity=uniform OR when --visual-"
                         "sensitivity=fisher but the probe / cost pickles "
                         "don't carry real visual Fisher data. BF16 (default) "
                         "reproduces passthrough behavior; NVFP4 / "
                         "FP8_DYNAMIC shrink the tower to quantized storage via the "
                         "existing RTN math at export time.")
    ap.add_argument("--visual-sensitivity",
                    choices=["fisher", "uniform"],
                    default="fisher",
                    help="How visual-encoder Linears are assigned. Visual "
                         "Linears are auxiliary to the language-model budget: "
                         "they are stamped with --visual-format and excluded "
                         "from default bpp/Δloss accounting. 'fisher' keeps "
                         "measured visual cost rows available for audit "
                         "metadata when present; 'uniform' uses only the "
                         "Phase 1 source-scan override.")
    ap.add_argument("--mtp-format",
                    choices=_format_cli_choices(),
                    default="BF16",
                    help="Uniform format for MTP Linears. BF16 is the "
                         "production default until MTP speculative-decode "
                         "acceptance is validated for quantized MTP weights.")
    ap.add_argument(
        "--lm-head-format",
        choices=_format_cli_choices(),
        default="BF16",
        help=(
            "Fixed auxiliary format for the language-model output head. "
            "BF16 preserves the historical profile pin. A non-BF16 value "
            "removes only lm_head from the DP, records its exact serialized "
            "payload inside whole-artifact accounting, and excludes its "
            "parameters from reported body bpp. Use --allow-pinned lm_head "
            "instead for the independent research path where the DP chooses "
            "the head format; a non-BF16 fixed value and that research "
            "override are mutually exclusive."
        ),
    )
    ap.add_argument("--bit-attribution-json", default=None,
                    help="Optional path: write a read-only 'where did the "
                         "budget go' report bucketing the final body "
                         "assignment by (block, role) with per-bucket bpp, "
                         "format mix, summed predicted_dloss and h_trace.")
    ap.add_argument("--bit-attribution-csv", default=None,
                    help="Optional path: write the flat per-Linear bit "
                         "attribution table (qname, block, role, format, "
                         "bits, bpp, n_params, h_trace, predicted_dloss).")
    # ---- Hard serving constraints: the second selection axis (P5c) ----
    # format-speed-policy.md §1's constrained problem. Latency is NEVER
    # blended into the objective (no lambda, no phase-weighted serve_ms):
    # these are hard constraints, an assignment that misses one is INFEASIBLE,
    # and the objective stays minimum predicted Delta-loss among the feasible.
    # Supply NONE of them and every code path is byte-identical to the
    # pre-P5c allocator apart from a stamp recording that constraints were
    # absent. See prismaquant/serve_constraints.py for the aggregation model
    # and its named assumptions.
    ap.add_argument("--serve-dispatch-table", default=None,
                    help="Path to a measured serve dispatch table "
                         "(prismaquant.serve_dispatch_table.v1): per "
                         "(format-family, phase, M-regime, lane) relative "
                         "serving costs, every row citing its source. An "
                         "example built only from published Gridbook "
                         "measurements ships at "
                         "prismaquant/serve_dispatch_tables/"
                         "gridbook_gb10_2026-08-01.example.json and is "
                         "PROPOSAL DATA, not a qualified serving model.")
    ap.add_argument("--serve-workload-mix", default=None,
                    help="Workload M-regime mix, e.g. "
                         "'prefill:dense_prefill_1400=1.0,"
                         "decode:decode_batch1=1.0'. Per-phase weights must "
                         "sum to 1.0. There is deliberately no default: "
                         "policy §1 forbids a default workload mix hidden in "
                         "the allocator.")
    ap.add_argument("--slo-prefill-p95-ttft-ms", type=float, default=None,
                    help="Hard constraint: predicted p95 TTFT (ms) must be "
                         "<= this. Prefill and decode are separate "
                         "constraints and are never blended.")
    ap.add_argument("--slo-decode-p95-itl-ms", type=float, default=None,
                    help="Hard constraint: predicted p95 inter-token latency "
                         "(ms) must be <= this.")
    ap.add_argument("--slo-decode-p05-tps", type=float, default=None,
                    help="Hard constraint: predicted p05 decode throughput "
                         "(tok/s) must be >= this.")
    ap.add_argument("--serve-device-budget-bytes", type=int, default=None,
                    help="Hard constraint: resident weight bytes + "
                         "--serve-kv-bytes + --serve-peak-scratch-bytes must "
                         "be <= this.")
    ap.add_argument("--serve-kv-bytes", type=int, default=0,
                    help="Operator-supplied KV-cache bytes for the device "
                         "memory constraint (not modelled by the allocator).")
    ap.add_argument("--serve-peak-scratch-bytes", type=int, default=0,
                    help="Operator-supplied peak scratch bytes for the "
                         "device memory constraint (not modelled by the "
                         "allocator).")
    args = ap.parse_args()

    effective_cb_source_scope = args.cb_codebook_source_scope
    if effective_cb_source_scope is None:
        effective_cb_source_scope = (
            "all" if args.cb_codebook_source == "learned" else "none"
        )
    if effective_cb_source_scope != "none" and not args.cb_codebook_bundle:
        raise SystemExit(
            "[alloc] ERROR: learned CB requires an immutable value-bearing "
            "codebook bundle before reading probe/cost inputs; pass "
            "--cb-codebook-bundle. Digest-only identity cannot render values."
        )

    if args.target_disk_gb is not None:
        if not math.isfinite(args.target_disk_gb) or args.target_disk_gb <= 0:
            raise SystemExit(
                "[alloc] ERROR: --target-disk-gb must be a positive finite "
                "decimal-GB budget"
            )
        if (
            args.artifact_overhead_reserve_bytes is None
            or args.artifact_overhead_reserve_bytes <= 0
        ):
            raise SystemExit(
                "[alloc] ERROR: --target-disk-gb is a whole-artifact hard "
                "budget and requires a positive "
                "--artifact-overhead-reserve-bytes. The selector can price "
                "tensor data deterministically, but safetensors headers, "
                "JSON and copied tokenizer/processor files require an "
                "operator-supplied conservative reserve; final export is "
                "measured recursively and fails closed."
            )
    elif args.artifact_overhead_reserve_bytes is not None:
        raise SystemExit(
            "[alloc] ERROR: --artifact-overhead-reserve-bytes is meaningful "
            "only with --target-disk-gb"
        )

    # ---- Hard serving constraints, resolved once (ultraplan P5c) ----
    # Built before any expensive work so a malformed table, an unbalanced
    # workload mix, or an SLO with nothing to price it fails on the command
    # line rather than after a multi-minute solve. An INACTIVE context (no
    # table / no SLOs) makes every downstream call a no-op that only stamps
    # "constraints were absent" — the pre-P5c behaviour, byte for byte.
    try:
        serve_dispatch = (
            load_dispatch_table(args.serve_dispatch_table)
            if args.serve_dispatch_table else None
        )
        serve_context = ServeConstraintContext(
            table=serve_dispatch,
            mix=WorkloadMix.parse(args.serve_workload_mix),
            slos=ServeSLOs(
                p95_ttft_ms=args.slo_prefill_p95_ttft_ms,
                p95_itl_ms=args.slo_decode_p95_itl_ms,
                p05_tps=args.slo_decode_p05_tps,
                device_budget_bytes=args.serve_device_budget_bytes,
                kv_bytes=int(args.serve_kv_bytes or 0),
                peak_scratch_bytes=int(args.serve_peak_scratch_bytes or 0),
            ),
        )
        serve_context.validate()
    except (DispatchTableError, ServeConstraintError) as exc:
        raise SystemExit(f"[alloc] ERROR: {exc}") from None
    if serve_context.active:
        print(
            "[alloc] serving constraints ACTIVE (ultraplan P5c): table="
            f"{serve_dispatch.table_id!r} status={serve_dispatch.status!r}, "
            f"mix={args.serve_workload_mix!r}, "
            f"SLOs={serve_context.slos.as_dict()}. These are HARD "
            "constraints: an assignment that misses one is infeasible, never "
            "scored worse. The objective is unchanged (min predicted Δloss); "
            "no λ blending exists. Table-driven latency is PROPOSAL DATA — "
            "the served NATIVE-PARITY protocol is the release gate.",
            flush=True,
        )
    elif serve_dispatch is not None or args.serve_workload_mix:
        print(
            "[alloc] serving constraints INACTIVE: a dispatch table and/or "
            "workload mix was supplied but no SLO, so no serving constraint "
            "is evaluated and selection is the pre-P5c byte-budget objective.",
            flush=True,
        )

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Detect the model profile from the probe's metadata. The probe
    # writes `meta.model` when it runs, so we can look up the HF
    # config at that path and map it to a registered ModelProfile.
    # Profile governs fused-sibling promotion (allocator's
    # `promote_fused`) and the vLLM-internal name remap
    # (`build_quantization_config` via export_native_compressed).
    from .model_profiles import detect_profile, DefaultProfile
    model_profile = DefaultProfile()
    with open(args.probe, "rb") as f:
        _probe_peek = pickle.load(f)
    validate_probe_payload(_probe_peek, args.probe)
    probe_model_path = _probe_peek.get("meta", {}).get("model")
    del _probe_peek
    if args.model_override:
        probe_model_path = args.model_override
        print(f"[alloc] model-override: {probe_model_path}", flush=True)
    used_default_fallback = False
    if probe_model_path:
        model_profile = detect_profile(probe_model_path)
        print(f"[alloc] model profile: {model_profile.name} "
              f"(derived from {probe_model_path})", flush=True)
    else:
        # No model path (probe lacks meta['model'] and no --model-override):
        # we fall back to DefaultProfile, whose fallback fused map covers only
        # qkv_proj, gate_up_proj, and the known DeltaNet in_proj_ba/qkvz
        # groups. Unknown architecture-specific merged columns remain
        # invisible, so fused-sibling promotion can leave them with mixed
        # formats -> an unservable / silently-corrupt checkpoint. A
        # SINGLE-format menu is always safe (every Linear gets the same format,
        # so fused groups are trivially coherent); a multi-format menu is gated
        # to a hard error below unless --allow-default-profile is set.
        used_default_fallback = True
        print(
            "[alloc] WARNING: no model path (probe meta['model'] is unset and "
            "no --model-override) -> using DefaultProfile. Architecture-"
            "specific fused-sibling groups will NOT be enforced. Pass "
            "--model-override <model> (or rebuild the probe with meta['model'] "
            "set) so detect_profile resolves the real profile.", flush=True)
    target_profile = resolve_target_profile(
        model_profile, args.target_profile,
        default=str(args.target_profile_default or "research"),
    )
    if target_profile not in serving_profile_names():
        raise SystemExit(f"[alloc] ERROR: unknown target profile {target_profile!r}")
    print(f"[alloc] target profile: {target_profile}", flush=True)

    if bool(args.research_cost_base) != bool(args.research_cost_segments_dir):
        raise SystemExit(
            "[alloc] ERROR: --research-cost-base and "
            "--research-cost-segments-dir must be supplied together"
        )
    if args.research_cost_base:
        if not args.accept_research_cost_table:
            raise SystemExit(
                "[alloc] ERROR: assembling segmented research costs requires "
                "--accept-research-cost-table"
            )
        from .research_cost_acceptance import assemble_research_cost_table
        try:
            _assembled, _manifest = assemble_research_cost_table(
                args.research_cost_base,
                args.research_cost_segments_dir,
                output_path=args.costs,
            )
        except ValueError as exc:
            raise SystemExit(f"[alloc] ERROR: research cost assembly: {exc}") from None
        print(
            "[alloc] RESEARCH COST ACCEPTED: assembled "
            f"{_manifest['assembled_row_count']} rows x "
            f"{len(_manifest['formats'])} formats from "
            f"{_manifest['layer_count']} x {_manifest['rows_per_layer']} "
            f"layer rows -> {args.costs}",
            flush=True,
        )

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    validate_probe_payload(probe, args.probe)
    validate_cost_payload(cost_data, args.costs)
    from .research_cost_acceptance import (
        accepted_cost_provenance,
        propagated_cost_provenance,
    )
    try:
        research_cost_provenance = accepted_cost_provenance(cost_data)
    except ValueError as exc:
        raise SystemExit(f"[alloc] ERROR: {exc}") from None
    if research_cost_provenance is not None and not args.accept_research_cost_table:
        raise SystemExit(
            "[alloc] ERROR: cost table is research-stamped; pass "
            "--accept-research-cost-table to acknowledge its study-grade "
            "assembled provenance"
        )
    if args.accept_research_cost_table and research_cost_provenance is None:
        raise SystemExit(
            "[alloc] ERROR: --accept-research-cost-table cannot bless an "
            "unstamped table; assemble it through the sanctioned path first"
        )
    if research_cost_provenance is not None:
        print(
            "[alloc] RESEARCH COST ACCEPTANCE ACTIVE: production CB reuse, "
            "serialized-payload, lattice-coverage, and render-scope guards "
            "remain unchanged outside this exact stamped table",
            flush=True,
        )
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    # ---- Fisher renormalization ----
    # One shared denominator (the global calib token count) recomputed
    # from the stored raw accumulators; hard error on probes that cannot
    # be renormalized unless --allow-legacy-fisher-norm. See
    # renormalize_probe_fisher for the full story.
    renormalize_probe_fisher(stats, probe.get("meta", {}),
                             allow_legacy=args.allow_legacy_fisher_norm)

    # ---- Robust Fisher clip (opt-in research lever) ----
    # PRISMAQUANT_FISHER_CAP_MULTIPLIER=K caps each dense row's h_trace at
    # K x its role's median. Unset => byte-identical no-op. Must run AFTER
    # the renormalization above (it clips the finalized scalar, not the raw
    # accumulator). See clip_probe_fisher_outliers.
    clip_probe_fisher_outliers(stats, probe.get("meta", {}))

    allow_pinned = list(parse_allow_pinned(args.allow_pinned))
    lm_head_format_canonical = fr.get_format(args.lm_head_format).name
    lm_head_dp_unpinned = allow_pinned_lifts_lm_head(
        model_profile, allow_pinned
    )
    if lm_head_dp_unpinned and lm_head_format_canonical != "BF16":
        raise SystemExit(
            "[alloc] ERROR: --lm-head-format fixes lm_head outside the body "
            "DP, while --allow-pinned lm_head asks the DP to choose it. "
            "These modes are mutually exclusive. Drop --allow-pinned for "
            f"the fixed {lm_head_format_canonical} production recipe, or "
            "leave --lm-head-format=BF16 for the research/DP path."
        )
    fixed_lm_head_quantized = (
        not lm_head_dp_unpinned and lm_head_format_canonical != "BF16"
    )
    allocation_excluded = []
    for name in sorted(set(stats) | set(costs)):
        if model_profile.is_pinned_name(name):
            if fixed_lm_head_quantized and is_lm_head_name(
                name, model_profile
            ):
                # Retain the row until the fixed-auxiliary pass below. It is
                # removed before the body DP and priced exactly there.
                continue
            if any(tok in name for tok in allow_pinned):
                continue  # opt-in: let the allocator choose this name's format
            allocation_excluded.append(name)
    tied_head_added = tied_lm_head_dp_exclusions(
        stats, costs, model_profile, probe_model_path, allocation_excluded)
    allocation_excluded.extend(tied_head_added)
    fixed_tied_heads = [
        name for name in tied_head_added
        if fixed_lm_head_quantized and is_lm_head_name(name, model_profile)
    ]
    if fixed_tied_heads:
        raise SystemExit(
            "[alloc] ERROR: a fixed quantized lm_head cannot share storage "
            "with the non-quantizable input embedding; quantizing it would "
            f"also quantize the embedding ({fixed_tied_heads}). Keep "
            "--lm-head-format=BF16 for tied embeddings."
        )
    if tied_head_added:
        print(
            "[alloc] tied LM head (shares storage with the non-quantizable "
            f"input embedding) excluded from DP budget: {tied_head_added} "
            "— not overridable by --allow-pinned", flush=True)
    if allow_pinned:
        print(f"[alloc] --allow-pinned active for {allow_pinned}: these "
              "profile-pinned names enter the DP budget (allocator chooses "
              "their format by cost-per-byte)", flush=True)
    if fixed_lm_head_quantized:
        print(
            f"[alloc] --lm-head-format={lm_head_format_canonical}: lm_head "
            "is a fixed auxiliary assignment (outside body bpp/DP, inside "
            "exact artifact bytes)",
            flush=True,
        )
    # vLLM fused-load invariant: a fused-sibling group missing a member (e.g.
    # Gemma4 k_eq_v full-attention layers synthesize v=k and ship no v_proj /
    # v_scale) cannot be partially quantized — the present members must ship
    # BF16, else the fused load KeyErrors on a non-existent scale param.
    # Generic + profile-driven (no model-specific code here).
    incomplete_added = incomplete_fused_group_dp_exclusions(
        stats, costs, model_profile, allocation_excluded)
    allocation_excluded.extend(incomplete_added)
    if incomplete_added:
        print(
            "[alloc] incomplete fused-sibling groups → BF16 (vLLM fused-load "
            f"invariant): {len(incomplete_added)} Linears "
            f"(sample: {incomplete_added[:8]})",
            flush=True,
        )
    if allocation_excluded:
        excluded = set(allocation_excluded)
        stats = {name: value for name, value in stats.items() if name not in excluded}
        costs = {name: value for name, value in costs.items() if name not in excluded}
        print(
            "[alloc] profile-pinned names excluded from DP budget: "
            f"{len(allocation_excluded)} Linears "
            f"(sample: {allocation_excluded[:8]})",
            flush=True,
        )
    stats = _mark_weight_only_nvfp4_stats(stats, model_profile)
    accounting_stats = dict(stats)

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    cb_serialization_context = None
    cb_col_weights = None
    cb_cost_render_identity = None
    cb_requested_names = [spec.name for spec in specs]
    cb_requested_names.extend((
        lm_head_format_canonical,
        fr.get_format(args.mtp_format).name,
        fr.get_format(args.visual_format).name,
    ))
    if any(is_cb_format(name) for name in cb_requested_names):
        if (
            args.cb_scale_coding is None
            or args.cb_codebook_source is None
            or args.cb_scale_sweep is None
            or args.cb_ldlq is None
            or args.cb_encode_tier is None
        ):
            raise SystemExit(
                "[alloc] ERROR: a CB format is present in the body or fixed "
                "auxiliary assignment but exact serialized "
                "and renderer context is missing. Pass --cb-scale-coding, "
                "--cb-codebook-source, --cb-scale-sweep, --cb-ldlq, "
                "--cb-minchain, and "
                "--cb-encode-tier; refusing implicit render defaults."
            )
        if args.cb_codebook_digests is not None:
            raise SystemExit(
                "[alloc] ERROR: --cb-codebook-digests is a digest-only legacy "
                "contract and cannot supply learned values. Pass "
                "--cb-codebook-bundle instead."
            )
        if args.cb_col_weights is None:
            raise SystemExit(
                "[alloc] ERROR: CB allocation requires --cb-col-weights so "
                "the cost table can be checked against the exact imatrix "
                "that cache/KL/export consume"
            )
        try:
            with open(args.cb_col_weights, "rb") as fh:
                cb_col_weights = pickle.load(fh)
        except Exception as exc:
            raise SystemExit(
                f"[alloc] ERROR: cannot load CB col-weights "
                f"{args.cb_col_weights}: {exc}"
            ) from None
        if not isinstance(cb_col_weights, Mapping):
            raise SystemExit(
                "[alloc] ERROR: --cb-col-weights must contain a qname -> "
                "tensor mapping"
            )
        try:
            cb_env = {
                "CB_SCALE_CODING": args.cb_scale_coding,
                "CB_CODEBOOK_SOURCE": args.cb_codebook_source,
                "CB_SCALE_SWEEP": args.cb_scale_sweep,
                "PRISMAQUANT_CB_LDLQ": args.cb_ldlq,
                "PRISMAQUANT_CB_MINCHAIN": args.cb_minchain,
                "PRISMAQUANT_CB_ENCODE_TIER": args.cb_encode_tier,
            }
            if args.cb_codebook_source_scope is not None:
                cb_env["CB_CODEBOOK_SOURCE_SCOPE"] = (
                    args.cb_codebook_source_scope
                )
            if args.cb_scale_sweep_scope is not None:
                cb_env["CB_SCALE_SWEEP_SCOPE"] = args.cb_scale_sweep_scope
            if args.cb_ldlq_scope is not None:
                cb_env["PRISMAQUANT_CB_LDLQ_SCOPE"] = args.cb_ldlq_scope
            if args.cb_codebook_bundle is not None:
                cb_env["CB_CODEBOOK_BUNDLE"] = args.cb_codebook_bundle
            cb_serialization_context = cb_serialization_context_from_env(
                cb_env,
                require_explicit=True,
                where="allocator CB producer context",
            )
        except ValueError as exc:
            raise SystemExit(f"[alloc] ERROR: {exc}") from None
        print(
            "[alloc] CB serialized payload: "
            f"scale_coding={cb_serialization_context.scale_coding} "
            f"layout_version={cb_serialization_context.layout_version} "
            f"codebook_source={cb_serialization_context.codebook_source} "
            f"codebook_source_scope={cb_serialization_context.codebook_source_scope} "
            f"scale_sweep={cb_serialization_context.scale_sweep} "
            f"scale_sweep_scope={cb_serialization_context.scale_sweep_scope} "
            f"ldlq={cb_serialization_context.ldlq} "
            f"encode_tier={cb_serialization_context.encode_tier} "
            f"renderer_abi={cb_serialization_context.renderer_abi}",
            flush=True,
        )
        if research_cost_provenance is None:
            try:
                validate_cb_cost_provenance(
                    cost_data,
                    cb_requested_names,
                    context=cb_serialization_context,
                    where=f"allocator cost cache {args.costs}",
                )
                _stored_context, cb_cost_render_identity = (
                    validate_cb_render_provenance(
                        cost_data,
                        expected_context=cb_serialization_context,
                        col_weights=cb_col_weights,
                        where=f"allocator cost cache {args.costs}",
                    )
                )
            except ValueError as exc:
                raise SystemExit(f"[alloc] ERROR: {exc}") from None
    specs_sorted, serialized_rates = _sort_specs_by_serialized_rate(
        specs,
        accounting_stats,
        cb_serialization_context,
    )

    # Fused-coherence guard: a multi-format menu under DefaultProfile cannot
    # enforce architecture-specific fused-sibling coherence (e.g. Qwen3.x
    # DeltaNet in_proj_ba) or packed-MoE expert uniformity, so the allocation
    # can ship an unservable or silently-corrupt checkpoint. DefaultProfile is
    # the RESOLVED profile either when no model path was available OR when the
    # model architecture is not registered -- key off the resolved profile, not
    # just the no-model fallback, so an unrecognized model with a path is also
    # caught (and so packed-MoE archs are protected by the same gate). Fail
    # PROACTIVELY here rather than rely only on the export's last-line hard-fail.
    # Single-format menus are always coherent; --allow-default-profile is the
    # explicit escape for vanilla transformers. The menu is de-duped by format
    # NAME so an alias/duplicate cannot false-trigger.
    validate_default_profile_format_menu(
        model_profile,
        specs_sorted,
        allow_default_profile=args.allow_default_profile,
    )

    if args.packed_role_split:
        # A role split emits different expert formats for gate_up vs down
        # projections of the SAME MoE layer — only loadable when the
        # serving lane keys expert schemes per projection. Hard-error
        # unless the resolved serving profile declares the capability
        # (SystemExit inside on unsupported/unknown profiles).
        require_per_role_expert_scheme_support(
            target_profile, flag="--packed-role-split")
        # Split packed expert groups into gate_up / down serving units by
        # wrapping the profile view: DP aggregation AND serving promotion
        # both key groups through packed_expert_format_group, so wrapping
        # keeps them consistent (role units atomic, final promotion still a
        # validated no-op). Must run after the isinstance-based
        # DefaultProfile gate above.
        model_profile = packed_role_split_profile(model_profile)
        print("[alloc] --packed-role-split: packed expert groups keyed as "
              "(layer, {gate_up|down}) serving units "
              f"(profile {target_profile!r} declares per-role expert "
              "schemes)", flush=True)

    # --- Format-family coherence check -----------------------------------
    # A sensible format ladder has at most ONE format per bit tier. Having
    # both NVFP4 and MXFP4 (or MXFP6_E3M2 and MXFP6_E2M3) means the allocator
    # picks between them based on tiny measurement noise per-layer, which
    # produces a serving mess: two separate kernel paths for the same tier.
    #
    # We bucket formats by exact serialized rate rounded to 0.25 and warn when a
    # bucket has more than one member. If --enforce-family-coherence is
    # set we error instead.
    from collections import Counter as _Counter
    buckets: dict[float, list[str]] = {}
    for s in specs_sorted:
        key = round(serialized_rates[s.name] * 4) / 4
        buckets.setdefault(key, []).append(s.name)
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    if collisions:
        msg = ("format set has multiple candidates at the same bit tier; "
               "the allocator will pick among them based on per-layer RTN "
               "noise, which is usually not what you want:\n"
               + "\n".join(f"  {k} bits: {v}" for k, v in collisions.items())
               + "\nRecommended bundles (vLLM serving, today):\n"
               "  Production     : NVFP4,FP8_DYNAMIC,BF16\n"
               "  MX research    : MXFP4,MXFP8_E4M3,BF16\n"
               "  Experimental   : NVFP4,MXFP6_E3M2,FP8_DYNAMIC,BF16   "
               "(MXFP6 hardware-supported on Blackwell, vLLM kernels not yet landed)")
        if args.enforce_family_coherence:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        else:
            print(f"[alloc] WARNING: {msg}", flush=True)
    mtp_format_canonical = fr.get_format(args.mtp_format).name
    visual_format_canonical = fr.get_format(args.visual_format).name
    rank_specs = {s.name: s for s in specs_sorted}
    rank_specs.setdefault(
        lm_head_format_canonical,
        fr.get_format(lm_head_format_canonical),
    )
    rank_specs.setdefault(mtp_format_canonical, fr.get_format(mtp_format_canonical))
    rank_specs.setdefault(visual_format_canonical, fr.get_format(visual_format_canonical))
    rank_specs_sorted, _rank_serialized_rates = _sort_specs_by_serialized_rate(
        list(rank_specs.values()),
        accounting_stats,
        cb_serialization_context,
    )
    format_rank = {s.name: i for i, s in enumerate(rank_specs_sorted)}
    format_specs = {s.name: s for s in rank_specs_sorted}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({serialized_rates[s.name]:.4f}b)' for s in specs_sorted]}")

    # Optional empirical calibration: per-format scalar gain α_f. When
    # absent, all gains default to 1.0.
    calibrated_gains: dict[str, float] = {}
    if args.calibration:
        with open(args.calibration) as f:
            cal_payload = json.load(f)
        cal_raw = cal_payload.get("calibrated_gains") or {}
        for fmt_name, gain_val in cal_raw.items():
            try:
                calibrated_gains[fmt_name] = float(gain_val)
            except (TypeError, ValueError):
                continue
        if calibrated_gains:
            print(f"[alloc] calibration loaded from {args.calibration}: "
                  f"{ {k: round(v, 4) for k, v in calibrated_gains.items()} }",
                  flush=True)
        else:
            print(f"[alloc] WARNING: {args.calibration} has no usable "
                  f"calibrated_gains; running uncalibrated", flush=True)

    # Source-dtype manifest drives passthrough-integrity filtering and the
    # exact candidate-payload <= source-payload gate in build_candidates.
    # None when model path is unknown — legacy/offline allocations cannot
    # evaluate either source-dependent rule.
    source_manifest: dict[str, str] | None = None
    if probe_model_path:
        source_manifest = _scan_source_dtype_manifest(
            probe_model_path, model_profile)
        if source_manifest is not None and not source_manifest:
            # Empty scan = the model dir has no safetensors to classify
            # (config-only dirs, probe-only flows). No evidence is not
            # evidence of mismatch: fall back to legacy gating (BF16
            # passthrough allowed) rather than mapping every name to
            # "unknown" and silently stripping the BF16 rung.
            print("[alloc] source-dtype manifest EMPTY (no safetensors "
                  f"found under {probe_model_path}) — passthrough formats "
                  "allowed WITHOUT source verification", flush=True)
            source_manifest = None
        if source_manifest:
            # Report EVERY kind the scan found, not a hardcoded fp8/bf16 pair.
            # A checkpoint whose units are mxfp4 and fp8_ue8m0 printed "0 fp8,
            # 238 bf16" under the old line — which reads as "nothing was
            # classified" precisely when the classification is the interesting
            # part, and hides a newly censused native format entirely.
            kinds = Counter(source_manifest.values())
            summary = ", ".join(
                f"{count} {kind}" for kind, count in sorted(kinds.items())
            )
            gated = ", ".join(sorted(PASSTHROUGH_SOURCE_REQUIREMENTS))
            print(f"[alloc] source-dtype manifest: {summary} "
                  f"(gates {gated} per source)",
                  flush=True)

    # A requested production CB rung needs a measured row everywhere it is
    # otherwise legal. `build_candidates` historically skipped absent/error
    # rows, which could silently prune the entire CB menu after an imatrix or
    # learned-codebook render failure and let the run select stock/BF16.
    body_cb_specs = [spec for spec in specs_sorted if is_cb_format(spec.name)]
    mtp_cb_spec = (
        fr.get_format(mtp_format_canonical)
        if is_cb_format(mtp_format_canonical)
        else None
    )
    visual_cb_spec = (
        fr.get_format(visual_format_canonical)
        if is_cb_format(visual_format_canonical)
        else None
    )
    incomplete_cb_costs: list[tuple[str, str, str]] = []
    for name, stats_entry in stats.items():
        if _is_mtp_linear(name):
            required_specs = [mtp_cb_spec] if mtp_cb_spec is not None else []
        elif _is_visual_linear(name):
            required_specs = (
                [visual_cb_spec] if visual_cb_spec is not None else []
            )
        else:
            required_specs = body_cb_specs
        source_kind = (
            source_manifest.get(name, "unknown")
            if source_manifest is not None
            else None
        )
        per_name_costs = costs.get(name)
        for spec in required_specs:
            verdict = check_stats_format_applicability(
                stats_entry,
                spec,
                qname=name,
                source_kind=source_kind,
                target_profile=target_profile,
                cb_serialization_context=cb_serialization_context,
            )
            if not verdict.legal:
                continue
            row = None
            if isinstance(per_name_costs, Mapping):
                for candidate_name in dict.fromkeys(
                    (spec.name, *fr.aliases_for(spec.name))
                ):
                    if candidate_name in per_name_costs:
                        row = per_name_costs[candidate_name]
                        break
            reason = None
            if not isinstance(row, Mapping):
                reason = "missing"
            elif "error" in row:
                reason = f"error={row.get('error')!r}"
            if reason is not None:
                incomplete_cb_costs.append((str(name), spec.name, reason))
    if incomplete_cb_costs:
        sample = "; ".join(
            f"{name}={fmt} ({reason})"
            for name, fmt, reason in incomplete_cb_costs[:8]
        )
        raise SystemExit(
            "[alloc] ERROR: production CB cost coverage is incomplete for "
            f"{len(incomplete_cb_costs)} legal (tensor, format) pair(s): "
            f"{sample}. CB export is imatrix-weighted, so missing/error rows "
            "cannot be silently pruned from the menu. Rebuild the cost/cache "
            "with complete production col_weights or remove the CB rung."
        )

    # ---- Activation-fair pricing (ultraplan P5a) ----
    # ONE per-family calibration for the whole run, fit before any candidate
    # is built so the body, MTP and visual menus cannot end up on three
    # different scales. See activation_fair_pricing for the functional form,
    # the fail-closed policy, and the PRISMAQUANT_ACTIVATION_FAIR_PRICING
    # kill switch (this call raises AssertionError on a mixed scale).
    activation_pricing_stats = (
        {
            name: entry for name, entry in stats.items()
            if not is_lm_head_name(name, model_profile)
        }
        if fixed_lm_head_quantized else stats
    )
    activation_pricing_costs = (
        {
            name: entry for name, entry in costs.items()
            if not is_lm_head_name(name, model_profile)
        }
        if fixed_lm_head_quantized else costs
    )
    activation_pricing = calibrate_activation_fair_pricing(
        activation_pricing_stats,
        activation_pricing_costs,
        specs_sorted,
    )
    if activation_pricing.enabled:
        print(
            "[alloc] activation-fair pricing: "
            + ", ".join(
                f"{family} x{fit.penalty:.4g} (n={fit.n_rows}, "
                f"log2 sd={fit.log2_stdev:.3f}, rung-dep "
                f"{fit.rung_dependence_log2_range:.3f})"
                for family, fit in sorted(activation_pricing.families.items())
            ),
            flush=True,
        )
    else:
        print(
            "[alloc] activation-fair pricing DISABLED "
            f"({activation_pricing.reason}): weight-only-priced rows keep "
            "their pre-P5a prices, so any W4A4-vs-W8A8 comparison drawn "
            "from this run is NOT activation-fair "
            f"(weight_only rows by family: "
            f"{dict(sorted(activation_pricing.weight_only_rows_by_family.items()))})",
            flush=True,
        )

    # ---- Cross-family CB-ladder symmetry verdict (ultraplan P5a item 2) ----
    # Computed by the cost stage on its own held-out units; the allocator
    # republishes it so a consumer reading only the allocation artifacts can
    # see whether this run's ladder fits support a cross-family (NVFP4-CB vs
    # FP8-CB) claim at all. A failure is surfaced, never fatal: the
    # allocation is still solvable, only the cross-family verdict is not
    # publishable.
    cross_family_verdict = cross_family_verdict_from_cost_payload(cost_data)
    if cross_family_verdict is not None:
        line = (
            "[alloc] CB ladder cross-family symmetry: "
            f"{str(cross_family_verdict.get('verdict', '?')).upper()} — "
            f"{cross_family_verdict.get('detail', '')}"
        )
        if not cross_family_verdict.get(
                "cross_family_comparison_publishable", False):
            print(f"[alloc] WARNING: {line[8:]}", flush=True)
        else:
            print(line, flush=True)

    candidate_mask_records: list[dict] = []
    candidates = build_candidates(
        stats, costs, specs_sorted, calibrated_gains,
        source_manifest=source_manifest,
        target_profile=target_profile,
        mask_records=candidate_mask_records,
        cb_serialization_context=cb_serialization_context,
        activation_pricing=activation_pricing,
    )
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    fixed_format_assignment: dict[str, str] = {}
    fixed_stats: dict[str, dict] = {}
    fixed_chosen_candidates: dict[str, Candidate] = {}
    fixed_lm_head_names: set[str] = set()
    fixed_lm_head_cost_pricing = (
        "direct_terminal_measurement_no_body_activation_transfer"
        if fixed_lm_head_quantized else None
    )

    if fixed_lm_head_quantized:
        head_probe_names = sorted(
            name for name in stats
            if is_lm_head_name(name, model_profile)
        )
        if not head_probe_names:
            raise SystemExit(
                f"[alloc] --lm-head-format={lm_head_format_canonical} was "
                "requested, but the probe has no lm_head row. Re-run the "
                "probe with --include-lm-head."
            )
        head_names_without_costs = [
            name for name in head_probe_names if name not in costs
        ]
        if head_names_without_costs:
            raise SystemExit(
                f"[alloc] --lm-head-format={lm_head_format_canonical} was "
                f"requested, but the cost pickle lacks {len(head_names_without_costs)} "
                f"lm_head row(s): {head_names_without_costs[:8]}. Re-run "
                "cost measurement with --include-lm-head."
            )
        head_stats = {name: stats[name] for name in head_probe_names}
        head_costs = {name: costs[name] for name in head_probe_names}
        head_candidates = build_candidates(
            head_stats,
            head_costs,
            [fr.get_format(lm_head_format_canonical)],
            calibrated_gains,
            source_manifest=source_manifest,
            target_profile=target_profile,
            mask_records=candidate_mask_records,
            cb_serialization_context=cb_serialization_context,
            # The fixed head is selected from direct terminal-head evidence.
            # The body family transfer fits a measured/output-space scale for
            # ordinary internal Linears and must not multiply that terminal
            # measurement a second time.  Passing None preserves the row's
            # own predicted_dloss/cost_source precedence while making the
            # absence of body activation transfer explicit.
            activation_pricing=None,
        )
        missing_head_candidates = [
            name for name in head_probe_names
            if _find_candidate_for_format(
                head_candidates,
                name,
                lm_head_format_canonical,
            ) is None
        ]
        if missing_head_candidates:
            raise SystemExit(
                f"[alloc] --lm-head-format={lm_head_format_canonical} "
                "requires a measured, serveable candidate for every head, "
                f"but {len(missing_head_candidates)} are missing: "
                f"{missing_head_candidates[:8]}. Rebuild the head cost for "
                "this format and verify the target serving profile supports "
                "quantized lm_head."
            )
        fixed_lm_head_names = set(head_probe_names)
        fixed_format_assignment.update({
            name: lm_head_format_canonical for name in head_probe_names
        })
        fixed_stats.update(head_stats)
        fixed_chosen_candidates.update({
            name: candidate
            for name in head_probe_names
            if (
                candidate := _find_candidate_for_format(
                    head_candidates,
                    name,
                    lm_head_format_canonical,
                )
            ) is not None
        })
        stats = {
            name: value for name, value in stats.items()
            if name not in fixed_lm_head_names
        }
        costs = {
            name: value for name, value in costs.items()
            if name not in fixed_lm_head_names
        }
        candidates = {
            name: value for name, value in candidates.items()
            if name not in fixed_lm_head_names
        }
        print(
            f"[alloc] --lm-head-format={lm_head_format_canonical}: fixed "
            f"{len(fixed_lm_head_names)} lm_head Linear(s) before DP as "
            "auxiliary to body bpp/\u0394loss accounting",
            flush=True,
        )

    mtp_names_without_costs = sorted(
        n for n in stats if _is_mtp_linear(n) and n not in costs
    )
    if mtp_names_without_costs and mtp_format_canonical != "BF16":
        sample = ", ".join(mtp_names_without_costs[:8])
        raise SystemExit(
            f"[alloc] --mtp-format={mtp_format_canonical} was requested, "
            f"but the cost pickle lacks {len(mtp_names_without_costs)} MTP "
            f"Linear(s). Sample: {sample}. Re-run cost measurement with "
            "--include-mtp."
        )
    if mtp_names_without_costs:
        print(
            "[alloc] WARNING: probe has MTP Linears absent from the cost "
            f"pickle; leaving {len(mtp_names_without_costs)} MTP Linears "
            "out of allocator accounting. Re-run cost measurement with "
            "--include-mtp to budget them explicitly.",
            flush=True,
        )
    mtp_names = sorted(n for n in stats if _is_mtp_linear(n) and n in costs)
    if mtp_names:
        mtp_stats = {name: stats[name] for name in mtp_names}
        mtp_costs = {name: costs[name] for name in mtp_names}
        mtp_candidates = build_candidates(
            mtp_stats,
            mtp_costs,
            [fr.get_format(mtp_format_canonical)],
            calibrated_gains,
            source_manifest=source_manifest,
            target_profile=target_profile,
            mask_records=candidate_mask_records,
            cb_serialization_context=cb_serialization_context,
            activation_pricing=activation_pricing,
        )
        missing_mtp_candidates = [
            name for name in mtp_names
            if _find_candidate_for_format(
                mtp_candidates,
                name,
                mtp_format_canonical,
            ) is None
        ]
        if missing_mtp_candidates:
            sample = ", ".join(missing_mtp_candidates[:8])
            raise SystemExit(
                f"[alloc] --mtp-format={mtp_format_canonical} requires "
                "measured, serveable MTP cost candidates, but "
                f"{len(missing_mtp_candidates)} MTP Linear(s) are missing "
                f"that candidate. Sample: {sample}. Re-run cost measurement "
                "with --include-mtp for the requested MTP format."
            )
        fixed_format_assignment.update({
            name: mtp_format_canonical for name in mtp_names
        })
        fixed_stats.update({name: stats[name] for name in mtp_names})
        mtp_chosen_candidates = {
            name: _find_candidate_for_format(
                mtp_candidates,
                name,
                mtp_format_canonical,
            )
            for name in mtp_names
        }
        fixed_chosen_candidates.update({
            name: cand for name, cand in mtp_chosen_candidates.items()
            if cand is not None
        })
        if mtp_names:
            fixed_names = set(mtp_names)
            stats = {
                name: value for name, value in stats.items()
                if name not in fixed_names
            }
            costs = {
                name: value for name, value in costs.items()
                if name not in fixed_names
            }
            candidates = {
                name: value for name, value in candidates.items()
                if name not in fixed_names
            }
            print(
                f"[alloc] --mtp-format={mtp_format_canonical}: fixed "
                f"{len(mtp_names)} MTP Linears before DP "
                "as auxiliary to body bpp/Δloss accounting",
                flush=True,
            )

    visual_names = sorted(n for n in stats if _is_visual_linear(n))
    visual_aux_candidates: dict[str, Candidate] = {}
    if visual_names:
        visual_cost_names = [name for name in visual_names if name in costs]
        if visual_cost_names and args.visual_sensitivity == "fisher":
            visual_stats = {name: stats[name] for name in visual_cost_names}
            visual_costs = {name: costs[name] for name in visual_cost_names}
            visual_candidates = build_candidates(
                visual_stats,
                visual_costs,
                [fr.get_format(visual_format_canonical)],
                calibrated_gains,
                source_manifest=source_manifest,
                target_profile=target_profile,
                mask_records=candidate_mask_records,
                cb_serialization_context=cb_serialization_context,
                activation_pricing=activation_pricing,
            )
            visual_aux_candidates = {
                name: cand for name in visual_cost_names
                if (
                    cand := _find_candidate_for_format(
                        visual_candidates,
                        name,
                        visual_format_canonical,
                    )
                ) is not None
            }
        fixed_format_assignment.update({
            name: visual_format_canonical for name in visual_names
        })
        fixed_stats.update({name: stats[name] for name in visual_names})
        fixed_chosen_candidates.update(visual_aux_candidates)
        visual_names_set = set(visual_names)
        stats = {
            name: value for name, value in stats.items()
            if name not in visual_names_set
        }
        costs = {
            name: value for name, value in costs.items()
            if name not in visual_names_set
        }
        candidates = {
            name: value for name, value in candidates.items()
            if name not in visual_names_set
        }
        print(
            f"[alloc] --visual-format={visual_format_canonical}: fixed "
            f"{len(visual_names)} visual Linears as auxiliary to body "
            f"bpp/Δloss accounting"
            + (
                f" ({len(visual_aux_candidates)} measured cost rows tracked)"
                if visual_aux_candidates else ""
            ),
            flush=True,
        )

    # Text-only probes can omit the complete visual tower. Discover those
    # source-only Linears *before* Pareto records are built so every candidate
    # JSON, CB identity, payload price, and budget stamp covers the exact full
    # assignment later emitted by the selector/exporter. The historical late
    # insertion made Pareto files deltas over the final layer_config and could
    # stamp a different (occasionally smaller) artifact than the one shipped.
    source_visual_stats = (
        discover_visual_linear_stats_from_source(
            probe_model_path,
            strict=bool(
                args.target_disk_gb is not None
                or visual_format_canonical != "BF16"
            ),
        )
        if probe_model_path
        else {}
    )
    source_visual_stats = _mark_weight_only_nvfp4_stats(
        source_visual_stats,
        model_profile,
    )
    source_only_visual_stats = {
        name: entry
        for name, entry in source_visual_stats.items()
        if name not in fixed_format_assignment
    }
    if source_only_visual_stats and is_cb_format(visual_format_canonical):
        sample = sorted(source_only_visual_stats)[:8]
        raise SystemExit(
            "[alloc] ERROR: source-only visual Linears cannot be assigned a "
            f"CB format without measured imatrix/cost rows (sample={sample}). "
            "Run multimodal probing so these Linears have production "
            "col_weights, or choose a non-CB --visual-format."
        )
    try:
        validate_source_visual_passthrough_contract(
            source_visual_stats,
            visual_format_canonical,
        )
    except ValueError as exc:
        raise SystemExit(f"[alloc] ERROR: {exc}") from None
    if source_only_visual_stats:
        fixed_format_assignment.update({
            name: visual_format_canonical for name in source_only_visual_stats
        })
        fixed_stats.update(source_only_visual_stats)
        print(
            f"[alloc] --visual-format={visual_format_canonical}: added "
            f"{len(source_only_visual_stats)} source-only visual Linears "
            "to every full Pareto assignment before byte pricing",
            flush=True,
        )

    pre_aggregation_availability = {
        spec.name: sum(
            1 for per_name in candidates.values()
            if any(c.fmt == spec.name for c in per_name)
        )
        for spec in specs_sorted
    }

    fixed_total_params = sum(
        int(entry.get("n_params", 0) or 0) for entry in fixed_stats.values()
    )

    def _bits_for_stats_entry(entry: dict, fmt: str, qname: str) -> float:
        shape = _shape_from_stats(entry)
        payload_bytes, _identity, _sidecar_identity = serialized_candidate_payload(
            fr.get_format(fmt),
            shape,
            qname=qname,
            cb_serialization_context=cb_serialization_context,
        )
        global_bytes = (
            nvfp4_global_sidecar_bytes(
                qname,
                shape,
                weight_only=bool(
                    entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                ),
            )
            if fmt == "NVFP4"
            else 0
        )
        return 8.0 * (payload_bytes + global_bytes)

    fixed_total_bits = sum(
        _bits_for_stats_entry(fixed_stats[name], fmt, name)
        for name, fmt in fixed_format_assignment.items()
        if name in fixed_stats
    )
    fixed_cb_assignment = {
        name: fmt
        for name, fmt in fixed_format_assignment.items()
        if name in fixed_stats and is_cb_format(fmt)
    }
    if fixed_cb_assignment:
        if cb_serialization_context is None:
            raise AssertionError(
                "fixed CB assignment reached payload reporting without a "
                "CBSerializationContext"
            )
        fixed_cb_payload = cb_assignment_payload_breakdown(
            fixed_cb_assignment,
            {
                name: _shape_from_stats(fixed_stats[name])
                for name in fixed_cb_assignment
            },
            context=cb_serialization_context,
        )
        fixed_total_bits += 8.0 * int(
            fixed_cb_payload["codebook_sidecar_bytes"]
        )
    fixed_total_dloss = sum(
        float(cand.predicted_dloss)
        for cand in fixed_chosen_candidates.values()
    )

    applicability_report_path = (
        Path(args.applicability_report)
        if args.applicability_report
        else Path(args.pareto_csv).with_name("format_applicability.json")
    )
    applicability_report_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-Linear legal-format sets for serving-unit promotion, snapshotted
    # BEFORE aggregation: both promotion call sites below run on the EXPANDED
    # per-Linear assignment, where an aggregated super-item's candidate list no
    # longer describes its individual members. Auxiliary MTP/visual names were
    # already removed from `candidates` and stay absent here on purpose — they
    # are format-PINNED, not legality-restricted (their candidates were built
    # from a one-format menu), and promotion treats an absent name as
    # unconstrained.
    per_linear_legal_formats = legal_formats_from_candidates(candidates)

    # Pre-aggregate packed-MoE serving groups (e.g. DeepSeek-V4's 768
    # per-expert Linears per layer) into single multi-choice DP units.
    # The serving runtime loads each group under ONE format, so the DP
    # must price the whole-group move — pricing per row while
    # promote_serving_units charges the group is a ~1000x mismatch:
    # expert rows top the per-bin ranking, the feasibility tightening
    # over-corrects, and attention/shared rows starve while headroom goes
    # unused. With groups as first-class DP units, post-DP MoE promotion
    # is a validated no-op.
    if not args.no_packed_aggregation:
        stats, costs, candidates = aggregate_packed_serving_groups(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains,
            activation_pricing=activation_pricing)
        packed_groups = sum(
            1 for n in candidates if _PACKED_GROUP_MARKER in n)
        packed_member_rows = sum(
            len(stats[n].get("_packed_group_members", ()))
            for n in candidates if _PACKED_GROUP_MARKER in n
        )
        print(f"[alloc] packed-serving-group aggregation: {packed_groups} "
              f"groups ({packed_member_rows} member Linears priced as "
              "whole-group DP units)")

    # Pre-aggregate fused siblings (qkv_proj, gate_up_proj, ...) into
    # single DP items. The DP can't pick mixed-sibling solutions because
    # there's only one item per group — so promote_fused becomes a no-op
    # on aggregated items and the overshoot-tightening loop collapses to
    # a single pass on well-behaved models. Must run AFTER the MoE
    # aggregation (it skips `.__fused__.` and packed-group entries
    # explicitly).
    if not args.no_fused_aggregation:
        stats, costs, candidates = aggregate_fused_siblings(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains,
            activation_pricing=activation_pricing)
        sib_groups = sum(1 for n in candidates if _FUSED_SIBLING_MARKER in n)
        print(f"[alloc] fused-sibling aggregation: {sib_groups} groups "
              f"(qkv_proj / gate_up_proj / ...)")

    candidates = filter_candidates_for_profile(candidates, target_profile)

    post_aggregation_availability = {
        spec.name: sum(
            1 for per_name in candidates.values()
            if any(c.fmt == spec.name for c in per_name)
        )
        for spec in specs_sorted
    }
    mutable_total_params = sum(
        int(stats[n].get("n_params", 0) or 0) for n in candidates
    )
    if fixed_total_params:
        fixed_bpp = fixed_total_bits / max(fixed_total_params, 1)
        print(
            "[alloc] auxiliary fixed-format contribution "
            "(excluded from body budget/loss): "
            f"{fixed_total_params:,} params, {fixed_bpp:.4f} bpp, "
            f"Δloss={fixed_total_dloss:.4e}",
            flush=True,
        )
    applicability_payload = {
        "schema": "prismaquant.format_applicability.v1",
        "target_profile": target_profile,
        "model_profile": getattr(model_profile, "name", ""),
        "formats": [spec.name for spec in specs_sorted],
        "probe": str(args.probe),
        "costs": str(args.costs),
        "pre_aggregation_candidate_availability": pre_aggregation_availability,
        "post_aggregation_candidate_availability": post_aggregation_availability,
        "fixed_format_assignment": {
            "lm_head_format": lm_head_format_canonical,
            "lm_head_mode": (
                "dp" if lm_head_dp_unpinned
                else "fixed" if fixed_lm_head_quantized
                else "profile_pinned_bf16"
            ),
            "lm_head_cost_pricing": fixed_lm_head_cost_pricing,
            "mtp_format": mtp_format_canonical,
            "visual_format": visual_format_canonical,
            "linears": len(fixed_format_assignment),
            "linears_by_kind": dict(Counter(
                "lm_head" if name in fixed_lm_head_names
                else "mtp" if _is_mtp_linear(name)
                else "visual" if _is_visual_linear(name)
                else "other"
                for name in fixed_format_assignment
            )),
            "params": fixed_total_params,
            "assignment_payload_bits_total": fixed_total_bits,
            "assignment_payload_bits_scope": (
                "fixed_assignment_tensor_payload_including_deduplicated_"
                "cb_sidecars"
            ),
            "predicted_dloss": fixed_total_dloss,
            "budget_scope": "auxiliary_excluded_from_body_budget",
        },
        # Ultraplan P5a/P5b: how every candidate's activation contract was
        # priced, whether this run's per-family ladder fits are
        # cross-comparable, and the concrete serving route each format rides.
        "activation_fair_pricing": activation_pricing.as_dict(),
        "cb_ladder_cross_family_verdict": cross_family_verdict,
        "serving_lanes": serving_lane_catalog(target_profile),
        **summarize_applicability_masks(
            candidate_mask_records,
            source_census_present=source_manifest is not None,
            source_census_units=(
                len(source_manifest) if source_manifest is not None else None
            ),
        ),
    }
    applicability_report_path.write_text(
        json.dumps(applicability_payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"[alloc] format applicability → {applicability_report_path}")

    def _stats_entry_for_assignment_name(name: str) -> dict | None:
        entry = stats.get(name)
        if isinstance(entry, dict):
            return entry
        fixed_entry = fixed_stats.get(name)
        if isinstance(fixed_entry, dict):
            return fixed_entry
        original_entry = accounting_stats.get(name)
        if isinstance(original_entry, dict):
            return original_entry
        return None

    def _expand_assignment_for_seed_json(
        assignment: dict[str, str],
        *,
        include_auxiliary: bool = True,
    ) -> dict[str, str]:
        """Expand DP super-items into the per-Linear seed-assignment shape.

        The allocator can solve over packed-serving-group and fused-sibling
        super-items. The KL probe's seed path wants ordinary module qnames;
        it already handles legality, pinning, and fused coherence, but giving
        it expanded names preserves the intended frontier point instead of
        making the super-item markers look like unknown entries.
        """
        expanded = dict(assignment)
        if not args.no_packed_aggregation:
            expanded = expand_packed_group_assignment(expanded, stats)
        if not args.no_fused_aggregation:
            expanded = expand_fused_sibling_assignment(expanded, stats)
        if include_auxiliary:
            expanded.update(fixed_format_assignment)
        return promote_serving_units(
            expanded,
            format_rank,
            profile=model_profile,
            legal_formats=per_linear_legal_formats,
        )

    _serve_lane_cache: dict[str, object] = {}

    def _serve_lane_for(name: str, fmt: str):
        """The concrete serving-lane route one assigned unit would ride (P5b).

        Same resolution order as ``selection_serving_lane_provenance``: prefer
        the ``Candidate`` the DP actually saw, and re-resolve from the target
        profile for expanded members of aggregated super items, which have no
        candidate of their own. This is what makes the constraint axis price a
        rung whose fused lane the pinned Gridbook version does NOT instantiate
        with its FALLBACK route's numbers instead of the fused lane's.
        """
        for cand in candidates.get(name, ()):
            if cand.fmt == fmt:
                return cand.serving_lane
        if fmt not in _serve_lane_cache:
            _serve_lane_cache[fmt] = serving_lane_route(target_profile, fmt)
        return _serve_lane_cache[fmt]

    def _serve_feasibility(
        expanded_assignment: Mapping[str, str],
        *,
        resident_bytes: int | None = None,
    ):
        """Policy §1's hard serving constraints for one expanded assignment.

        Inactive context -> a stamp and ``feasible=True``; nothing downstream
        changes. The stats view is the same three-map precedence the footprint
        accounting uses, so a super-item-expanded name resolves to real
        parameter counts (the aggregation is parameter-share weighted).
        """
        return evaluate_serve_constraints(
            expanded_assignment,
            {**accounting_stats, **fixed_stats, **stats},
            serve_context,
            lane_for=_serve_lane_for,
            resident_bytes=resident_bytes,
        )

    def _assignment_payload_totals(
        assignment: Mapping[str, str],
        *,
        require_all_stats: bool,
    ) -> dict[str, float | int | list[str]]:
        """Exact assignment-scope tensor payload, including shared CB tables.

        Candidate memory remains the additive DP proposal cost.  This function
        is the non-additive exact filter/reporting path: CB tables are charged
        once per physical identity and NVFP4 global scale tensors are included.
        """
        total = 0.0
        params = 0
        cb_assignment: dict[str, str] = {}
        cb_shapes: dict[str, tuple[int, ...]] = {}
        missing: list[str] = []
        for name, fmt in assignment.items():
            entry = _stats_entry_for_assignment_name(name)
            if not isinstance(entry, dict):
                missing.append(name)
                continue
            params += int(entry.get("n_params", 0) or 0)
            if is_cb_format(fmt):
                cb_assignment[name] = fmt
                cb_shapes[name] = _shape_from_stats(entry)
                continue
            # Super-items (packed groups, fused siblings) carry exact
            # per-format byte sums; their stats entries have no single
            # (out, in) shape, so the shape fallback is only for plain rows.
            memory_map = entry.get("_memory_bytes_by_format")
            if isinstance(memory_map, dict) and fmt in memory_map:
                total += 8.0 * memory_map[fmt]
                if fmt == "NVFP4":
                    total += 8.0 * nvfp4_global_sidecar_bytes(
                        name,
                        _shape_from_stats(entry),
                        weight_only=bool(
                            entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                        ),
                    )
                continue
            shape = _shape_from_stats(entry)
            payload_bytes, _identity, _sidecar_identity = serialized_candidate_payload(
                fr.get_format(fmt),
                shape,
                qname=name,
                cb_serialization_context=cb_serialization_context,
            )
            total += 8.0 * payload_bytes
            if fmt == "NVFP4":
                total += 8.0 * nvfp4_global_sidecar_bytes(
                    name,
                    shape,
                    weight_only=bool(
                        entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                    ),
                )
        if missing and require_all_stats:
            raise AssertionError(
                "exact assignment payload has no shape/stats for "
                f"{len(missing)} tensor(s): {sorted(missing)[:12]}"
            )
        cb_tensor_bits = 0.0
        cb_sidecar_bits = 0.0
        if cb_assignment:
            if cb_serialization_context is None:
                raise AssertionError(
                    "CB assignment reached exact bit reporting without a "
                    "CBSerializationContext"
                )
            payload = cb_assignment_payload_breakdown(
                cb_assignment,
                cb_shapes,
                context=cb_serialization_context,
            )
            cb_tensor_bits = 8.0 * int(payload["tensor_payload_bytes"])
            cb_sidecar_bits = 8.0 * int(payload["codebook_sidecar_bytes"])
            total += cb_tensor_bits + cb_sidecar_bits
        return {
            "bits_total": float(total),
            "quantizable_params": int(params),
            "bits_per_param": float(total) / max(params, 1),
            "cb_tensor_bits": float(cb_tensor_bits),
            "cb_shared_sidecar_bits": float(cb_sidecar_bits),
            "missing_stats_names": sorted(missing),
        }

    def _assignment_bits_total(assignment: dict[str, str]) -> float:
        return float(_assignment_payload_totals(
            assignment,
            require_all_stats=True,
        )["bits_total"])

    _solve_cache: dict[float, tuple] = {}
    # Solver diagnostics per target, kept beside the memo so a cache hit never
    # loses them: an INFEASIBLE rung's only explanation lives here.
    _solve_diagnostics: dict[float, dict] = {}

    def _solve_for_target(target_bits: float):
        """Solve additively, then exact-filter non-additive shared payloads.

        Shared CB sidecars are assignment activation costs rather than legal
        per-candidate additive costs.  The DP therefore proposes assignments;
        every proposal is expanded and exact-priced, and an over-target result
        tightens/re-solves.  This enforces feasibility but is deliberately not
        advertised as a globally optimal mixed-sidecar solve.
        """
        cache_key = round(float(target_bits), 9)
        cached = _solve_cache.get(cache_key)
        if cached is None:
            cached = _solve_for_target_uncached(target_bits)
            _solve_cache[cache_key] = cached
        assign, achieved_r, total, mutable_total = cached
        if assign is not None:
            assign = dict(assign)
        return assign, achieved_r, total, mutable_total

    def _solve_for_target_uncached(target_bits: float):
        requested_target = float(target_bits)
        mutable_target_bits = requested_target
        if mutable_total_params <= 0:
            if fixed_total_params > 0 and mutable_target_bits >= 0.0:
                return {}, 0.0, 0.0, 0.0
            return None, float("nan"), float("inf"), float("inf")
        if mutable_target_bits < 0.0:
            return None, float("nan"), float("inf"), float("inf")
        outer_diag: dict = {
            "solver_contract": (
                "additive_candidate_proposal_then_exact_assignment_filter"
            ),
            "global_optimality_claimed": False,
            "exact_filter_trace": [],
        }
        _solve_diagnostics[round(requested_target, 9)] = outer_diag
        for attempt in range(16):
            proposal_diag: dict = {}
            assign, solver_achieved = solve_with_promotion(
                stats,
                candidates,
                mutable_target_bits,
                format_specs,
                format_rank,
                args.bit_precision,
                no_fused_promote=args.no_fused_promote,
                overshoot_tolerance=args.overshoot_tolerance,
                profile=model_profile,
                diagnostics=proposal_diag,
            )
            outer_diag.update({
                key: value
                for key, value in proposal_diag.items()
                if key not in {"achieved_bits"}
            })
            if assign is None:
                return None, float("nan"), float("inf"), float("inf")
            expanded = _expand_assignment_for_seed_json(
                assign,
                include_auxiliary=False,
            )
            exact = _assignment_payload_totals(
                expanded,
                require_all_stats=True,
            )
            exact_achieved = float(exact["bits_per_param"])
            outer_diag["exact_filter_trace"].append({
                "attempt": attempt,
                "proposal_target_bits": float(mutable_target_bits),
                "solver_additive_candidate_bpp": float(solver_achieved),
                "exact_assignment_payload_bpp": exact_achieved,
                "cb_shared_sidecar_bits": float(
                    exact["cb_shared_sidecar_bits"]
                ),
                "feasible": bool(
                    exact_achieved
                    <= requested_target + args.overshoot_tolerance
                ),
            })
            if exact_achieved <= requested_target + args.overshoot_tolerance:
                outer_diag["achieved_bits"] = exact_achieved
                outer_diag["solver_additive_candidate_bpp"] = float(
                    solver_achieved
                )
                outer_diag["exact_assignment_payload_bpp"] = exact_achieved
                mutable_total = compute_assignment_predicted_dloss(
                    assign, candidates
                )
                return assign, exact_achieved, mutable_total, mutable_total
            overage = exact_achieved - requested_target
            next_target = (
                mutable_target_bits
                - overage
                - max(float(args.bit_precision), 1e-9)
            )
            if next_target >= mutable_target_bits or next_target < 0.0:
                break
            mutable_target_bits = next_target
        outer_diag["feasible"] = False
        outer_diag["reason"] = "exact_assignment_payload_filter_exhausted"
        return None, float("nan"), float("inf"), float("inf")

    def _cb_stamps_for_assignment(
        assignment: Mapping[str, str],
    ) -> dict[str, str]:
        cb_names = {
            name: fmt for name, fmt in assignment.items()
            if is_cb_format(fmt)
        }
        if not cb_names:
            return {}
        if cb_serialization_context is None:
            raise AssertionError(
                "CB assignment reached stamp emission without a "
                "CBSerializationContext"
            )
        shapes: dict[str, tuple[int, ...]] = {}
        for name in cb_names:
            entry = _stats_entry_for_assignment_name(name)
            if not isinstance(entry, dict):
                raise AssertionError(
                    f"{name}: cannot stamp CB serialization identity without "
                    "probe stats"
                )
            shapes[name] = _shape_from_stats(entry)
        stamps = cb_assignment_serialization_stamps(
            cb_names,
            shapes,
            context=cb_serialization_context,
        )
        # Candidate construction persisted the identity it priced. Assert the
        # expanded assignment still agrees before a Pareto/KL/export consumer
        # can mistake a promotion or stale aggregation record for exact bytes.
        for name, fmt in cb_names.items():
            entry = _stats_entry_for_assignment_name(name)
            identities = (
                entry.get("_serialized_identity_by_format")
                if isinstance(entry, dict)
                else None
            )
            if isinstance(identities, Mapping) and fmt in identities:
                if str(identities[fmt]) != stamps[name]:
                    raise AssertionError(
                        f"{name}: selected {fmt} serialization identity "
                        "differs from the candidate priced by the allocator"
                    )
        return stamps

    def _cb_render_identity_for_assignment(
        expanded_assignment: Mapping[str, str],
    ) -> dict | None:
        selected_scope = {
            str(name): (str(fmt),)
            for name, fmt in expanded_assignment.items()
            if is_cb_format(fmt)
        }
        if not selected_scope:
            return None
        if cb_cost_render_identity is None or cb_col_weights is None:
            if research_cost_provenance is not None:
                # This absence is the exact production guard the explicit
                # research-cost acceptance acknowledges. Do not fabricate a
                # render identity; carry the research manifest instead so the
                # exporter can demand its own independent acknowledgement.
                return None
            raise RuntimeError(
                "CB assignment has no validated value-bearing cost render "
                "identity"
            )
        return project_cb_render_identity(
            cb_cost_render_identity,
            selected_scope,
            col_weights=cb_col_weights,
            where="allocator selected CB assignment",
        )

    pareto_seed_records: list[dict] = []

    # Pareto sweep.
    targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = []
    for t in targets:
        assign, achieved, total, mutable_total = _solve_for_target(t)
        if assign is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        format_counts = defaultdict(int)
        format_params = defaultdict(int)
        for name, fmt in assign.items():
            format_counts[fmt] += 1
            entry = _stats_entry_for_assignment_name(name)
            if isinstance(entry, dict):
                format_params[fmt] += int(entry.get("n_params", 0) or 0)
        total_with_aux = total + fixed_total_dloss
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": total,
            "variable_predicted_dloss": mutable_total,
            "aux_fixed_predicted_dloss": fixed_total_dloss,
            "fixed_predicted_dloss": fixed_total_dloss,
            "total_predicted_dloss_with_aux": total_with_aux,
            "aux_fixed_assignment_payload_bits_total": fixed_total_bits,
            "aux_fixed_params": fixed_total_params,
            **{f"layers_{k}": v for k, v in format_counts.items()},
            **{f"params_{k}": v for k, v in format_params.items()},
        })
        if args.pareto_output_dir:
            expanded = _expand_assignment_for_seed_json(assign)
            budget_expanded = _expand_assignment_for_seed_json(
                assign,
                include_auxiliary=False,
            )
            expanded_counts = defaultdict(int)
            for fmt in expanded.values():
                expanded_counts[fmt] += 1
            pareto_cb_render_identity = _cb_render_identity_for_assignment(
                expanded
            )
            pareto_seed_records.append({
                "target_bits": float(t),
                "achieved_bits": float(achieved),
                "predicted_dloss": float(total),
                "variable_predicted_dloss": float(mutable_total),
                "aux_fixed_predicted_dloss": float(fixed_total_dloss),
                "fixed_predicted_dloss": float(fixed_total_dloss),
                "total_predicted_dloss_with_aux": float(total_with_aux),
                "assignment": expanded,
                "format_counts": dict(sorted(expanded_counts.items())),
                "bits_total": _assignment_bits_total(budget_expanded),
                "bits_total_with_aux": _assignment_bits_total(expanded),
                CB_ASSIGNMENT_IDENTITIES_FIELD: _cb_stamps_for_assignment(
                    expanded
                ),
                **({
                    "cb_render_identity": pareto_cb_render_identity,
                } if pareto_cb_render_identity is not None else {}),
            })

    # Coarse Kneedle, then golden-section refinement inside the knee bracket so
    # the knee isn't snapped to the coarse target grid (re-solves the sub-second
    # DP at interior budgets; pins the knee to ~--knee-refine-tol bpp). Additive:
    # the legacy coarse knees stay under their keys; the refined knee is added as
    # ``knee_summary["refined"]`` and its samples folded into the curve + seeds.
    knee_summary = _pareto_knee_summary(curve)
    if knee_summary.get("enabled") and args.knee_refine:
        refined, extra_pts = refine_knee_golden(
            _solve_for_target, knee_summary, curve, tol=args.knee_refine_tol,
        )
        if refined is not None:
            refined["aux_fixed_predicted_dloss"] = float(fixed_total_dloss)
            refined["fixed_predicted_dloss"] = float(fixed_total_dloss)
            refined["total_predicted_dloss_with_aux"] = (
                refined["predicted_dloss"] + float(fixed_total_dloss)
            )
            knee_summary["refined"] = refined
            seen = {round(float(r.get("target_bits", -1)), 6) for r in curve}
            for pt in extra_pts:
                if round(float(pt["target_bits"]), 6) not in seen:
                    curve.append(pt)
            curve.sort(key=lambda r: float(r.get("target_bits", 0.0)))
            if args.pareto_output_dir:
                r_assign, r_ach, r_tot, r_mut = _solve_for_target(refined["target_bits"])
                if r_assign is not None:
                    r_exp = _expand_assignment_for_seed_json(r_assign)
                    r_bud = _expand_assignment_for_seed_json(
                        r_assign, include_auxiliary=False)
                    r_counts = defaultdict(int)
                    for fmt in r_exp.values():
                        r_counts[fmt] += 1
                    refined_cb_render_identity = (
                        _cb_render_identity_for_assignment(r_exp)
                    )
                    pareto_seed_records.append({
                        "target_bits": float(refined["target_bits"]),
                        "achieved_bits": float(r_ach),
                        "predicted_dloss": float(r_tot),
                        "variable_predicted_dloss": float(r_mut),
                        "aux_fixed_predicted_dloss": float(fixed_total_dloss),
                        "fixed_predicted_dloss": float(fixed_total_dloss),
                        "total_predicted_dloss_with_aux": float(r_tot + fixed_total_dloss),
                        "assignment": r_exp,
                        "format_counts": dict(sorted(r_counts.items())),
                        "bits_total": _assignment_bits_total(r_bud),
                        "bits_total_with_aux": _assignment_bits_total(r_exp),
                        CB_ASSIGNMENT_IDENTITIES_FIELD: _cb_stamps_for_assignment(
                            r_exp
                        ),
                        **({
                            "cb_render_identity": refined_cb_render_identity,
                        } if refined_cb_render_identity is not None else {}),
                        "knee_refined": True,
                    })

    # Output Pareto CSV (includes golden-section refinement samples)
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")
    knee_path = Path(args.pareto_csv).with_suffix(".knees.json")
    knee_path.write_text(json.dumps(knee_summary, indent=2, sort_keys=True) + "\n")
    print(f"[alloc] Pareto knees → {knee_path}")
    if knee_summary.get("refined"):
        _r, _c = knee_summary["refined"], knee_summary["log_error"]
        print(f"[alloc] knee refined (golden-section): coarse achieved="
              f"{_c['achieved_bits']:.3f} → refined achieved={_r['achieved_bits']:.3f} "
              f"(target={_r['target_bits']:.3f}, {_r['evals']} DP evals, "
              f"±{_r['tol_bits']}b)")

    # --- deterministic tensor payload + conservative artifact bound --------
    # The byte budget is the CONSTRAINT and measured KL is the OBJECTIVE, but
    # `select_validated_frontier` cannot see the card: it reads only the
    # per-point KL rows. Pricing each candidate here — through the SAME
    # footprint.assignment_artifact_bytes the allocator's own byte-budget
    # selector uses.  That function prices safetensors tensor-data spans, not
    # a directory.  Under a whole-artifact budget we add the explicit operator
    # reserve; the exporter later measures every regular file and hard-fails.
    _footprint_ctx: dict[str, object] = {}

    def _partition_source_total(_fp, src_total, src_manifest, *, where,
                                assigned_names=()):
        """Apply --exclude-source-prefix, or pass the total through.

        Raises SystemExit (not ValueError) on a bad prefix ON PURPOSE: both
        callers of the pricing scalars sit behind `except Exception` clauses
        that degrade to "pricing unavailable", and a prefix that silently
        excluded nothing is the exact failure this flag exists to prevent —
        it under-fills the budget by the excluded mass and every downstream
        number stays self-consistent. SystemExit is a BaseException, so it
        passes through those clauses to the operator.
        """
        if not getattr(args, "exclude_source_prefix", None):
            return int(src_total), None
        try:
            part = _fp.partitioned_source_total_bytes(
                src_manifest, int(src_total), args.exclude_source_prefix,
                context=where, assigned_names=assigned_names)
        except ValueError as exc:
            raise SystemExit(f"[alloc] ERROR: {exc}") from None
        print(
            f"[alloc] source partition ({where}): excluding "
            f"{', '.join(part['excluded_prefixes'])} removes "
            f"{part['n_excluded']} source tensors / "
            f"{part['excluded_source_bytes'] / 1e9:.3f} GB; this artifact is "
            f"priced against {part['source_total_bytes'] / 1e9:.3f} GB of "
            f"{int(src_total) / 1e9:.3f} GB",
            flush=True)
        return int(part["source_total_bytes"]), part

    def _footprint_scalars():
        if _footprint_ctx or not probe_model_path:
            return _footprint_ctx or None
        from . import footprint as _fp
        try:
            src_total, src_by_dtype = _fp.source_checkpoint_bytes(probe_model_path)
            src_manifest = _fp.source_tensor_bytes_manifest(
                probe_model_path,
                name_map=getattr(model_profile, "checkpoint_to_live_name", None),
                expert_parent_for_projection=getattr(
                    model_profile, "packed_expert_parent_for_projection", None),
            )
            priced_total, _part = _partition_source_total(
                _fp, src_total, src_manifest,
                where="pareto candidate footprint",
                assigned_names={**accounting_stats, **fixed_stats, **stats})
            _footprint_ctx.update({
                "fp": _fp,
                "source_total_bytes": priced_total,
                "regime": _fp.source_regime(src_by_dtype),
                "source_manifest": src_manifest,
                "stats": {**accounting_stats, **fixed_stats, **stats},
            })
        except Exception as exc:  # pricing is additive; never break allocation
            print(f"[alloc] WARNING: Pareto footprint pricing unavailable: {exc}",
                  flush=True)
            return None
        return _footprint_ctx

    def _artifact_size_for(expanded_assignment):
        ctx = _footprint_scalars()
        if not ctx:
            return None
        try:
            info = ctx["fp"].assignment_artifact_bytes(
                expanded_assignment, ctx["stats"],
                source_total_bytes=ctx["source_total_bytes"],
                source_manifest=ctx["source_manifest"],
                regime=ctx["regime"],
                context="pareto candidate footprint",
                cb_serialization_context=cb_serialization_context,
            )
            if info["n_missing_stats"]:
                sample = ", ".join(info["missing_stats_names"][:10])
                raise ValueError(
                    f"{info['n_missing_stats']} assigned Linear(s) have no "
                    f"shape stats and cannot receive an exact artifact price: "
                    f"{sample}"
                )
            tensor_payload_bytes = int(info["artifact_payload_bytes"])
            reserve_bytes = int(args.artifact_overhead_reserve_bytes or 0)
            return {
                "artifact_tensor_payload_bytes": tensor_payload_bytes,
                "artifact_tensor_payload_scope": info["artifact_byte_scope"],
                **({
                    "whole_artifact_upper_bound_bytes": (
                        tensor_payload_bytes + reserve_bytes
                    ),
                    "artifact_bytes": tensor_payload_bytes + reserve_bytes,
                    "artifact_byte_scope": (
                        "selection_upper_bound_tensor_payload_plus_"
                        "operator_non_tensor_reserve"
                    ),
                } if args.target_disk_gb is not None else {}),
            }
        except Exception as exc:
            if args.target_disk_gb is not None:
                raise SystemExit(
                    "[alloc] ERROR: exact Pareto artifact pricing failed "
                    f"under --target-disk-gb: {exc}"
                ) from None
            print(f"[alloc] WARNING: could not price a Pareto candidate: {exc}",
                  flush=True)
            return None

    if args.pareto_output_dir:
        for record in pareto_seed_records:
            sized = _artifact_size_for(record["assignment"])
            if sized is not None:
                record.update(sized)

        # Collapse the Pareto set to the rungs that can actually ship. On an
        # 11-rung sweep under a card, 8 of those rungs are decided before a
        # single KL is measured — the narrowing is what makes byte-budget
        # selection ~3 KL evals instead of 11. Computed from the priced grid,
        # never hardcoded, and skipped (loudly) when pricing is unavailable.
        if args.target_disk_gb is not None and pareto_seed_records:
            from . import footprint as _fp_gb
            budget_bytes_pareto = int(
                math.floor(float(args.target_disk_gb) * _fp_gb.GB)
            )
            priced = [r for r in pareto_seed_records
                      if r.get("whole_artifact_upper_bound_bytes") is not None]
            if len(priced) != len(pareto_seed_records):
                print("[alloc] WARNING: byte-budget Pareto narrowing skipped — "
                      f"{len(pareto_seed_records) - len(priced)} of "
                      f"{len(pareto_seed_records)} candidates could not be "
                      "priced; measuring the full sweep", flush=True)
            else:
                ordered = sorted(pareto_seed_records,
                                 key=lambda r: r["whole_artifact_upper_bound_bytes"])
                fits = [i for i, r in enumerate(ordered)
                        if r["whole_artifact_upper_bound_bytes"] <= budget_bytes_pareto]
                if fits:
                    top = fits[-1]
                    keep_positions = {max(0, top - 1), top,
                                      min(len(ordered) - 1, top + 1)}
                    reason = (
                        f"largest fitting rung achieved="
                        f"{ordered[top]['achieved_bits']:.3f} bpp at "
                        f"{ordered[top]['whole_artifact_upper_bound_bytes'] / _fp_gb.GB:.3f}GB "
                        "selection upper bound")
                else:
                    keep_positions = set(range(min(2, len(ordered))))
                    reason = ("NOTHING fits the card; keeping the cheapest "
                              "rungs so the measurement shows the shortfall")
                keep = {id(ordered[i]) for i in sorted(keep_positions)}
                before = len(pareto_seed_records)
                pareto_seed_records = [r for r in pareto_seed_records
                                       if id(r) in keep]
                print(f"[alloc] byte-budget Pareto narrowing: {before} -> "
                      f"{len(pareto_seed_records)} rungs "
                      f"(card={args.target_disk_gb:.2f}GB; {reason})",
                      flush=True)

    if args.pareto_output_dir:
        out_dir = Path(args.pareto_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        seen_payloads: set[str] = set()
        for idx, record in enumerate(pareto_seed_records):
            assignment = record["assignment"]
            digest_src = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(digest_src.encode()).hexdigest()[:12]
            if digest in seen_payloads:
                continue
            seen_payloads.add(digest)
            label = (
                f"allocator_target_{record['target_bits']:.4f}"
                f"_achieved_{record['achieved_bits']:.4f}_{digest}"
            ).replace(".", "p")
            path = out_dir / f"{label}.json"
            record_budget_stamp = None
            if args.target_disk_gb is not None:
                record_budget_bytes = int(math.floor(
                    float(args.target_disk_gb) * 1_000_000_000.0
                ))
                upper = record.get("whole_artifact_upper_bound_bytes")
                payload_bytes = record.get("artifact_tensor_payload_bytes")
                if (
                    isinstance(upper, int)
                    and isinstance(payload_bytes, int)
                    and upper <= record_budget_bytes
                ):
                    record_budget_stamp = whole_artifact_budget_stamp(
                        budget_bytes=record_budget_bytes,
                        selection_tensor_payload_bytes=payload_bytes,
                        selection_non_tensor_reserve_bytes=int(
                            args.artifact_overhead_reserve_bytes
                        ),
                        selection_assignment=assignment,
                        excluded_source_prefixes=(
                            getattr(args, "exclude_source_prefix", None) or ()
                        ),
                    )
            payload = {
                "schema": "prismaquant.allocator.pareto_assignment.v2",
                "label": label,
                "source": "allocator_pareto",
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "bits_total_with_aux": float(record["bits_total_with_aux"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "total_predicted_dloss_with_aux": float(
                    record["total_predicted_dloss_with_aux"]
                ),
                "format_counts": record["format_counts"],
                "artifact_bytes": record.get("artifact_bytes"),
                "artifact_byte_scope": record.get("artifact_byte_scope"),
                "artifact_tensor_payload_bytes": record.get(
                    "artifact_tensor_payload_bytes"
                ),
                "artifact_tensor_payload_scope": record.get(
                    "artifact_tensor_payload_scope"
                ),
                "whole_artifact_upper_bound_bytes": record.get(
                    "whole_artifact_upper_bound_bytes"
                ),
                "target_profile": target_profile,
                "assignment": dict(sorted(assignment.items())),
                # Two independent facts, so two independent guards. A CB
                # assignment always carries per-tensor serialized identities,
                # but it carries a render identity only when the cost table had
                # a validated one to project from: under
                # ``--accept-research-cost-table``
                # ``_cb_render_identity_for_assignment`` deliberately returns
                # None rather than fabricate one (see its comment), and
                # ``cb_cost_render_identity`` is never assigned on that path at
                # all. Gating the render-identity fields on the *identities*
                # field therefore KeyErrors on every research-cost run that
                # writes a Pareto point with any CB format. The final-assignment
                # writer already guards on the render identity itself; this
                # makes the Pareto writer agree with it.
                **({
                    CB_ASSIGNMENT_IDENTITIES_FIELD: dict(sorted(
                        record.get(CB_ASSIGNMENT_IDENTITIES_FIELD, {}).items()
                    )),
                } if record.get(CB_ASSIGNMENT_IDENTITIES_FIELD) else {}),
                **({
                    "cb_serialized_payload": record[
                        "cb_render_identity"
                    ]["cb_serialized_payload"],
                    "cb_render_identity": record["cb_render_identity"],
                } if record.get("cb_render_identity") is not None else {}),
                **({
                    "whole_artifact_budget": record_budget_stamp,
                } if record_budget_stamp is not None else {}),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            manifest_rows.append({
                "label": label,
                "path": str(path),
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "bits_total_with_aux": float(record["bits_total_with_aux"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "total_predicted_dloss_with_aux": float(
                    record["total_predicted_dloss_with_aux"]
                ),
                "variable_predicted_dloss": float(record["variable_predicted_dloss"]),
                "aux_fixed_predicted_dloss": float(
                    record["aux_fixed_predicted_dloss"]
                ),
                "fixed_predicted_dloss": float(record["fixed_predicted_dloss"]),
                "format_counts": record["format_counts"],
                "artifact_bytes": record.get("artifact_bytes"),
                "artifact_byte_scope": record.get("artifact_byte_scope"),
                "artifact_tensor_payload_bytes": record.get(
                    "artifact_tensor_payload_bytes"
                ),
                "whole_artifact_upper_bound_bytes": record.get(
                    "whole_artifact_upper_bound_bytes"
                ),
            })
        (out_dir / "manifest.json").write_text(json.dumps({
            "schema": "prismaquant.allocator.pareto_manifest.v1",
            "probe": str(args.probe),
            "costs": str(args.costs),
            "target_profile": target_profile,
            "target_disk_gb": (float(args.target_disk_gb)
                               if args.target_disk_gb is not None else None),
            "artifact_overhead_reserve_bytes": (
                int(args.artifact_overhead_reserve_bytes)
                if args.artifact_overhead_reserve_bytes is not None
                else None
            ),
            "formats": [s.name for s in specs_sorted],
            "target_bits": [float(x) for x in targets],
            "knees": knee_summary,
            "candidates": manifest_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(
            f"[alloc] Pareto seed assignments → {out_dir} "
            f"({len(manifest_rows)} unique)",
            flush=True,
        )

    # Kneedle
    if knee_summary.get("enabled"):
        knee = knee_summary["log_error"]
        raw = knee_summary["raw_linear"]
        print(f"[alloc] suggested knee (log-error): "
              f"target={knee['target_bits']}, "
              f"achieved={knee['achieved_bits']:.3f}, "
              f"Δloss={knee['kneedle_dloss']:.3e} "
              f"({knee['kneedle_error_source']})")
        print(f"[alloc] raw-linear knee: target={raw['target_bits']}, "
              f"achieved={raw['achieved_bits']:.3f}, "
              f"Δloss={raw['kneedle_dloss']:.3e} "
              f"({raw['kneedle_error_source']})")
        rd = knee_summary.get("rd_curve", {})
        if rd.get("available"):
            print(f"[alloc] RD curve: log10(Δloss)≈{rd['slope_decades_per_bit']:.3f}"
                  f"·bpp+{rd['intercept']:.3f}  R²={rd['r2']:.4f}  "
                  + ("LOG-LINEAR → no intrinsic knee; ship by --target-disk-gb or "
                     "measured saturation (kneedle is axis-dependent diagnostic)."
                     if rd.get("log_linear") else
                     "has curvature → a knee may be meaningful; still prefer a "
                     "byte budget when shipping to a card."))

    # Print table
    print(f"\n  target  achieved     {'Δloss body':>20}   " + "   ".join(
        f"{s.name[:11]:>11}" for s in specs_sorted))
    for row in curve:
        if not row.get("feasible"):
            print(f"  {row['target_bits']:>6.3f}  INFEASIBLE")
            continue
        fmt_str = "   ".join(
            f"{row.get(f'layers_{s.name}', 0):>11,}" for s in specs_sorted)
        dloss_str = f"{row['predicted_dloss']:.4e}"
        print(f"  {row['target_bits']:>6.3f}  {row['achieved_bits']:>7.3f}  "
              f"{dloss_str:>20}   {fmt_str}")

    # ----- Whole-artifact budget ("fit the card") ship-bpp selection -----
    # When --target-disk-gb is given, the ship bpp is set by the card, not by
    # --target-bits: the card supplies the CONSTRAINT.  Selection uses a
    # conservative upper bound (exact tensor spans + required non-tensor
    # reserve); final export stats the whole directory and fails closed. The RD
    # curve is log-linear, so there is no intrinsic knee to
    # find — see rd_curve diagnostic) and predicted Δloss supplies the
    # OBJECTIVE (minimize it among the allocations that fit). This is a
    # selector over Pareto candidates whose feasibility test needs no
    # measurement, then a bisection of the same sub-second DP between grid
    # rungs to search denser fitting allocations.
    selected_whole_artifact_budget_stamp = None
    if args.target_disk_gb is not None:
        from . import footprint as _fp
        from .saturation_select import select_under_byte_budget
        if not probe_model_path:
            raise SystemExit(
                "[alloc] --target-disk-gb needs the source model path (probe "
                "meta.model or --model-override) to size the non-quantizable "
                "floor (lm_head/embed/norms).")
        # What the ratchet optimizes, recorded in selection.json: the shipped
        # objective must be recoverable from the artifact, not inferred from
        # the code version that produced it.
        _RATCHET_OBJECTIVE = "min_predicted_dloss__ties_to_larger_footprint"
        budget_bytes = int(math.floor(float(args.target_disk_gb) * _fp.GB))
        overhead_reserve_bytes = int(args.artifact_overhead_reserve_bytes)
        src_total, src_by_dtype = _fp.source_checkpoint_bytes(probe_model_path)
        regime = _fp.source_regime(src_by_dtype)  # recorded for reporting only
        # Per-tensor source-byte manifest: each re-encoded Linear is charged
        # its ACTUAL header byte span (weight + scale siblings), never a
        # regime-wide per-param rate. On mixed sources (an MXFP4-packed
        # DSv4-Flash checkpoint: I8 nibble experts + E8M0 scales, F8
        # attention, BF16 floor) the old regime accounting removed more
        # bytes than the checkpoint holds, driving the floor negative.
        # `expert_parent_for_projection` bridges the per-expert-on-disk /
        # packed-live MoE layouts (…experts.{i}.gate_proj -> the packed
        # …experts.gate_up_proj the allocator names), same mapping the
        # layer-streaming pack bridge uses.
        src_manifest = _fp.source_tensor_bytes_manifest(
            probe_model_path,
            name_map=getattr(model_profile, "checkpoint_to_live_name", None),
            expert_parent_for_projection=getattr(
                model_profile, "packed_expert_parent_for_projection", None),
        )
        # Must apply the SAME partition the Pareto sweep priced against, or
        # the narrowing pass and the ratchet would disagree about which rungs
        # fit the card.
        src_total_full = int(src_total)
        src_total, source_partition = _partition_source_total(
            _fp, src_total, src_manifest, where="byte-budget selection",
            assigned_names={**accounting_stats, **fixed_stats, **stats})

        # Stats view for the footprint accounting: the same three-map
        # precedence `_stats_entry_for_assignment_name` uses (aggregated DP
        # stats > fixed MTP/visual > pre-aggregation accounting stats), so
        # every name in an EXPANDED assignment resolves and the shared
        # function prices exactly the tensors the inlined copy did.
        _footprint_stats = {**accounting_stats, **fixed_stats, **stats}

        def _artifact_for_target(t: float):
            assign_t, ach_t, tot_t, _mut = _solve_for_target(t)
            if assign_t is None:
                return None
            expanded_t = _expand_assignment_for_seed_json(assign_t)
            ctx = f"byte-budget selector (target_bits={t:.3f})"
            # ONE accounting path: footprint.assignment_artifact_bytes owns
            # the manifest -> resolve -> check-non-negative -> floor sequence
            # (shared with floor_bytes_for_model, whose exact agreement the
            # footprint tests pin) plus the body-byte sum. Inlining a third
            # copy here meant the shipped selector was the one path that
            # agreement was never tested against.
            #
            # Both failure modes are hard errors raised BEFORE any selection
            # number is consumed: a re-encoded name the manifest cannot
            # resolve (its source bytes stay in the floor while its quantized
            # bytes are still added — on a packed-MoE model the whole expert
            # mass, after which every rung reads "below the floor"), and two
            # names resolving to the same source span (bytes removed twice, so
            # an over-budget artifact reads as fitting).
            try:
                info = _fp.assignment_artifact_bytes(
                    expanded_t, _footprint_stats,
                    source_total_bytes=int(src_total),
                    source_manifest=src_manifest,
                    regime=regime,
                    context=ctx,
                    cb_serialization_context=cb_serialization_context,
                )
            except ValueError as exc:
                # House idiom for an operator-facing fatal in main(): a
                # SystemExit line, not a raw traceback. The footprint
                # messages already name the offending tensors.
                raise SystemExit(f"[alloc] ERROR: {exc}") from None
            # A name with no stats entry is priced at source precision by
            # assignment_artifact_bytes (left in the floor, no body bytes) —
            # correct for a verbatim tensor, but for a name the DP actually
            # allocated it silently under-counts the body. It also breaks the
            # provable byte-for-byte identity with the inlined accounting this
            # replaced (which subtracted every expanded name from the floor).
            # Any such name is an allocator/probe bug, so refuse rather than
            # ship a number that no longer means what the label says.
            if info["n_missing_stats"]:
                sample = ", ".join(info["missing_stats_names"][:10])
                raise SystemExit(
                    f"[alloc] ERROR: {info['n_missing_stats']} allocated "
                    f"Linear(s) have no probe stats entry, so their exported "
                    f"bytes cannot be priced ({ctx}): {sample}"
                    + (", …" if info["n_missing_stats"] > 10 else "")
                    + ". The byte-budget selector would price them at source "
                    "precision and under-count the artifact. Fix the probe / "
                    "expansion so every allocated name carries stats.")
            # The SECOND hard axis (ultraplan P5c), evaluated on the same
            # exact expanded assignment the byte axis just priced — after
            # super-item expansion and serving-unit promotion, so the verdict
            # is about the assignment that would actually ship. Resident bytes
            # are the exact serialized tensor payload (assumption A6).
            serve = _serve_feasibility(
                expanded_t,
                resident_bytes=int(info["artifact_payload_bytes"]),
            )
            return {
                "target_bits": float(t), "achieved_bits": float(ach_t),
                "bpp": float(ach_t), "dloss": float(tot_t),
                "assignment": expanded_t,
                "tensor_payload_bytes": int(info["artifact_payload_bytes"]),
                "whole_artifact_upper_bound_bytes": int(
                    info["artifact_payload_bytes"]
                ) + overhead_reserve_bytes,
                "floor_bytes": float(info["floor_bytes"]),
                "serve": serve,
            }

        grid = []
        for row in curve:
            if not row.get("feasible"):
                continue
            r = _artifact_for_target(float(row["target_bits"]))
            if r is not None:
                grid.append(r)

        def _menu_floor_target_bits() -> float | None:
            """Exact bpp of the cheapest allocation this run's MENU can reach.

            Every DP unit takes its own cheapest candidate; the proposal is
            expanded and priced through the same exact accountant
            ``_solve_for_target`` filters against, so the returned bpp is a
            target the solver can actually land on (an epsilon-below value
            like the format's nominal rate is NOT: per-shape row scales and
            the shared CB sidecar push the achievable mean a hair above it,
            and the rung comes back INFEASIBLE).

            ``None`` when a unit has no candidate at all or the proposal
            cannot be priced — both are upstream legality/coverage bugs that
            their own gates report; this probe stays silent about them.
            """
            cheapest_fmt: dict[str, str] = {}
            for cand_name, cand_list in candidates.items():
                if not cand_list:
                    return None
                cheapest_fmt[cand_name] = min(
                    cand_list, key=lambda c: (c.memory_bytes, c.fmt)).fmt
            try:
                expanded_floor = _expand_assignment_for_seed_json(
                    cheapest_fmt, include_auxiliary=False)
                totals = _assignment_payload_totals(
                    expanded_floor, require_all_stats=True)
            except Exception:
                return None
            floor_bits = float(totals["bits_per_param"])
            if not math.isfinite(floor_bits) or floor_bits <= 0.0:
                return None
            return floor_bits

        def _extend_grid_to_menu_floor() -> dict:
            """Probe below the swept grid before calling a budget infeasible.

            The Pareto grid is a COARSE SAMPLE of the RD curve, but the
            byte-budget selector reads its cheapest point as "the cheapest
            allocation". That reading holds only while the grid actually
            reaches the bottom of the format menu, and ``--pareto-targets``
            defaults to 4.5..8.25 — a range written for a 4-bit/8-bit menu.
            On the CB menu (NVFP4-CB is ~2.03 bpp) every sampled point is
            more than twice as dense as the operator's budget, so a budget
            the menu can meet with room to spare is rejected as "below the
            floor" and the remedy printed with it ("raise the budget, widen
            the menu") is exactly backwards. Measured on DeepSeek-V4-Flash:
            a 92.0 GB budget whose menu floor is 87.2 GB was reported
            infeasible against a "cheapest allocation" of 172.4 GB, which is
            simply the 4.5 bpp rung.

            This runs ONLY after the swept grid has already failed, so a run
            that selects today keeps its exact grid, ratchet and selection.
            A run that would have died gets the rungs it was missing, and a
            budget genuinely below the menu floor gets a truthful floor in
            the error instead of an artifact of where the sweep happened to
            start.
            """
            record: dict = {
                "swept_min_target_bits": (min(targets) if targets else None),
                "menu_floor_target_bits": None,
                "added_targets": [],
                "infeasible_targets": [],
                "reason": None,
            }
            floor_bits = _menu_floor_target_bits()
            record["menu_floor_target_bits"] = floor_bits
            if floor_bits is None:
                record["reason"] = "menu_floor_not_priceable"
                return record
            swept_min = record["swept_min_target_bits"]
            if swept_min is None or float(swept_min) <= floor_bits + 1e-9:
                record["reason"] = "swept_grid_already_reaches_menu_floor"
                return record
            record["reason"] = "swept_grid_min_target_above_menu_floor"
            # Geometric fill: densest sampling at the floor, where a budget
            # that the sweep missed by this much almost always sits.
            n_fill = 8
            ratio = (float(swept_min) / floor_bits) ** (1.0 / n_fill)
            for step in range(n_fill):
                probe_t = floor_bits * (ratio ** step)
                probe = _artifact_for_target(float(probe_t))
                if probe is None:
                    record["infeasible_targets"].append(float(probe_t))
                    continue
                grid.append(probe)
                record["added_targets"].append(float(probe_t))
            grid.sort(key=lambda c: float(c["target_bits"]))
            return record

        # select_under_byte_budget is used here as the FEASIBILITY gate
        # (feasible / below_floor / rejected_next). Its own `chosen` is the
        # largest-footprint fitting candidate, which is deliberately NOT the
        # ship pick — see the objective note below; it is recorded as
        # `max_bytes_pick_*` so the two objectives stay comparable in the
        # artifact.
        sel = select_under_byte_budget(
            grid,
            budget_bytes,
            bytes_key="whole_artifact_upper_bound_bytes",
        )
        grid_extension: dict | None = None
        if not sel["feasible"]:
            grid_extension = _extend_grid_to_menu_floor()
            if grid_extension["added_targets"]:
                print(
                    "[alloc] byte-budget: the swept Pareto grid bottoms out "
                    f"at {grid_extension['swept_min_target_bits']:.4f} bpp, "
                    "above this menu's cheapest allocation "
                    f"({grid_extension['menu_floor_target_bits']:.4f} bpp) — "
                    f"probing {len(grid_extension['added_targets'])} rung(s) "
                    "below the sweep so 'the cheapest allocation' means the "
                    "cheapest the MENU can reach, not the cheapest that was "
                    "sampled. Pass --pareto-targets covering the menu to see "
                    "these rungs on the Pareto curve too.",
                    flush=True,
                )
                sel = select_under_byte_budget(
                    grid,
                    budget_bytes,
                    bytes_key="whole_artifact_upper_bound_bytes",
                )

        rd = knee_summary.get("rd_curve") if isinstance(knee_summary, dict) else None
        selection = {
            "schema": "prismaquant.allocator.byte_budget_selection.v3",
            "mode": "byte-budget",
            "target_disk_gb": float(args.target_disk_gb),
            "budget_bytes": budget_bytes,
            "artifact_overhead_reserve_bytes": overhead_reserve_bytes,
            "source_total_bytes": float(src_total),
            "source_regime": regime,
            "source_accounting": "per_tensor_manifest_v2",
            # Present only when --exclude-source-prefix partitioned the ship.
            # Without it `source_total_bytes` above would not reconcile with
            # the checkpoint on disk and a later reader would have no way to
            # tell a partitioned artifact from a mis-measured one.
            **({
                "source_partition": {
                    "excluded_prefixes": list(
                        source_partition["excluded_prefixes"]),
                    "excluded_source_bytes": int(
                        source_partition["excluded_source_bytes"]),
                    "n_excluded_source_tensors": int(
                        source_partition["n_excluded"]),
                    "source_total_bytes_unpartitioned": src_total_full,
                    "rationale": (
                        "tensors under these prefixes ship as a separate "
                        "artifact and are absent from this one"
                    ),
                },
            } if source_partition else {}),
            "footprint_path": "footprint.assignment_artifact_bytes",
            "source_bytes_per_param": int(
                _fp.dominant_source_bytes_per_param(src_by_dtype)),
            "selection_feasibility_test": (
                "exact_tensor_payload_bytes + operator_non_tensor_reserve_bytes "
                "<= whole_artifact_budget_bytes"
            ),
            "final_feasibility_test": (
                "stat_all_regular_files_recursive <= whole_artifact_budget_bytes"
            ),
            "feasibility_test": (
                "selection_whole_artifact_upper_bound_bytes <= budget_bytes; "
                "final_recursive_inventory_fail_closed"
            ),
            "ratchet_objective": _RATCHET_OBJECTIVE,
            "feasible": bool(sel["feasible"]),
            "below_floor": bool(sel["below_floor"]),
            "lm_head_unpinned": bool(lm_head_dp_unpinned),
            "lm_head_format": lm_head_format_canonical,
            "lm_head_mode": (
                "dp" if lm_head_dp_unpinned
                else "fixed" if fixed_lm_head_quantized
                else "profile_pinned_bf16"
            ),
            # Present only when the swept grid failed and rungs below it were
            # probed; ``menu_floor_target_bits`` is what "the cheapest
            # allocation" is measured against from here on.
            "pareto_grid_extension": grid_extension,
            "rd_curve": rd,
            "grid": [
                {"target_bits": c["target_bits"],
                 "achieved_bits": c["achieved_bits"],
                 "tensor_payload_gb": c["tensor_payload_bytes"] / _fp.GB,
                 "whole_artifact_upper_bound_gb": (
                     c["whole_artifact_upper_bound_bytes"] / _fp.GB
                 ),
                 "dloss": c["dloss"],
                 "fits": c["whole_artifact_upper_bound_bytes"] <= budget_bytes}
                for c in grid
            ],
            **propagated_cost_provenance(research_cost_provenance),
        }
        sel_path = Path(args.pareto_csv).with_name("selection.json")
        if not sel["feasible"]:
            cheapest = sel.get("rejected_next") or (grid[0] if grid else None)
            cheapest_gb = (
                cheapest["whole_artifact_upper_bound_bytes"] / _fp.GB
                if cheapest else float("nan")
            )
            selection["cheapest_whole_artifact_upper_bound_gb"] = cheapest_gb
            sel_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
            # The floor probe already ran (above), so this number is the
            # menu's floor and not the sweep's — EXCEPT when the probe itself
            # could not be priced or came back INFEASIBLE, which is the one
            # case where "cheapest" may still be an artifact of the sweep.
            # Say so rather than let the operator read a sampling boundary as
            # a property of the model.
            floor_caveat = ""
            if grid_extension is not None and not grid_extension["added_targets"]:
                if grid_extension["reason"] == "menu_floor_not_priceable":
                    floor_caveat = (
                        " NOTE: the menu's own cheapest allocation could not "
                        "be priced, so this figure is the cheapest SAMPLED "
                        "rung; widen --pareto-targets before trusting it as "
                        "a floor.")
                elif grid_extension["infeasible_targets"]:
                    floor_caveat = (
                        " NOTE: rungs below the sweep were probed down to "
                        f"{grid_extension['menu_floor_target_bits']:.4f} bpp "
                        "and the solver could not land any of them, so this "
                        "figure is the cheapest SAMPLED rung.")
            raise SystemExit(
                f"[alloc] --target-disk-gb={args.target_disk_gb:.3f} is below the "
                f"floor: the cheapest allocation is {cheapest_gb:.3f}GB. Raise the "
                "budget, fix lm_head to a smaller measured format "
                "(--lm-head-format FP8_E4M3), unpin it for a research DP "
                "(--allow-pinned lm_head), or widen the format menu. "
                f"Selection written to {sel_path}." + floor_caveat)

        # Among allocations whose conservative selection upper bound fits the
        # card, ship the
        # one with the LOWEST predicted Δloss — ties broken toward the larger
        # footprint (spend the budget only when it costs nothing in predicted
        # quality). "Fill the card" was a proxy for that, valid only while
        # more bytes implied lower Δloss; it does not (5.5 bpp has beaten 6.0
        # bpp on served PPL, and serving-unit promotion can flip a group into
        # a denser-but-worse format), so ratcheting on MAX bytes could
        # actively select a denser artifact with WORSE predicted Δloss than a
        # sparser one that also fits. Same objective and same tie-break the
        # solver's own feasible-iterate ratchet uses (solve_with_promotion).
        #
        # Δloss is comparable across rungs: every rung's value is
        # compute_assignment_predicted_dloss over the SAME DP item set (the
        # multi-choice knapsack assigns every item exactly one candidate at
        # every target), in the same ½·h_trace·MSE units, and the fixed
        # auxiliary (MTP/visual) Δloss excluded from all of them is a
        # rung-invariant constant, so it cannot reorder them.
        #
        # Selection feasibility is tensor spans + reserve <= budget. Final
        # feasibility is intentionally deferred to the exporter, which stats
        # the recursive regular-file set and fails closed. The ratchet is
        # seeded at the grid pick already proven selection-feasible.
        def _fits_bytes(cand) -> bool:
            return (
                cand is not None
                and cand["whole_artifact_upper_bound_bytes"] <= budget_bytes
            )

        def _serve_ok(cand) -> bool:
            """The second hard axis. Always True when constraints are absent.

            Deliberately a separate predicate from the byte test so the two
            reasons a probe was rejected stay distinguishable in the trace —
            and so that with no constraints supplied ``_fits`` is the
            pre-P5c function, evaluated on the pre-P5c inputs.
            """
            if cand is None:
                return False
            serve = cand.get("serve")
            return serve is None or bool(serve.feasible)

        def _fits(cand) -> bool:
            """Feasibility = BOTH hard axes (policy §1).

            Bytes and SLOs are constraints, never terms in the objective. A
            probe that misses either is removed from the candidate set; it is
            not ranked below the others. The ratchet below then minimises
            predicted Δloss over exactly the survivors, with its tie-break
            unchanged.
            """
            return _fits_bytes(cand) and _serve_ok(cand)

        def _beats(cand, best) -> bool:
            """The ratchet objective: min Δloss, ties -> larger footprint."""
            if best is None:
                return True
            if cand["dloss"] != best["dloss"]:
                return cand["dloss"] < best["dloss"]
            return (
                cand["whole_artifact_upper_bound_bytes"]
                > best["whole_artifact_upper_bound_bytes"]
            )

        best = None
        emit_target = None
        ratchet_trace: list[dict] = []
        # Every probe the SLO axis removed, and which limit removed it. Empty
        # (and omitted from selection.json) when constraints are inactive.
        serve_rejections: list[dict] = []
        serve_probe_rows: list[dict] = []

        def _record_serve(cand, target: float, stage: str) -> None:
            """Note a probe's serving verdict; nothing when constraints are off."""
            if cand is None or not serve_context.active:
                return
            serve = cand["serve"]
            serve_probe_rows.append({
                "label": f"{stage}@{float(target):.4f}",
                "feasible": bool(serve.feasible),
                "predicted": dict(serve.predicted),
            })
            if not serve.feasible:
                serve_rejections.append(rejection_record(
                    serve,
                    stage=stage,
                    target_bits=float(target),
                    achieved_bits=float(cand["achieved_bits"]),
                    dloss=float(cand["dloss"]),
                ))

        def _consider(cand, target: float, stage: str) -> bool:
            """Ratchet ``cand`` in; record the probe. Returns whether it fits."""
            nonlocal best, emit_target
            fits = _fits(cand)
            accepted = bool(fits and _beats(cand, best))
            _record_serve(cand, target, stage)
            ratchet_trace.append({
                "stage": stage,
                "target_bits": float(target),
                "achieved_bits": (
                    float(cand["achieved_bits"]) if cand is not None else None),
                "tensor_payload_gb": (
                    cand["tensor_payload_bytes"] / _fp.GB
                    if cand is not None else None
                ),
                "whole_artifact_upper_bound_gb": (
                    cand["whole_artifact_upper_bound_bytes"] / _fp.GB
                    if cand is not None else None
                ),
                "dloss": float(cand["dloss"]) if cand is not None else None,
                "fits": fits,
                "accepted": accepted,
                # Additive and CONDITIONAL: with no constraints supplied the
                # trace rows are byte-identical to the pre-P5c allocator's.
                **({
                    "fits_bytes": _fits_bytes(cand),
                    "serve_feasible": (
                        bool(cand["serve"].feasible)
                        if cand is not None else None),
                    "serve_binding_constraint": (
                        cand["serve"].binding_constraint
                        if cand is not None else None),
                    "serve_violated_constraints": (
                        list(cand["serve"].violation_names())
                        if cand is not None else None),
                } if serve_context.active else {}),
            })
            if accepted:
                best, emit_target = cand, float(target)
            return fits

        fitting = [c for c in grid if _fits(c)]
        for cand in grid:
            _record_serve(cand, float(cand["target_bits"]), "grid")
        grid_pick = None
        for cand in fitting:
            if _beats(cand, grid_pick):
                grid_pick = cand

        if grid_pick is None:
            # Bytes alone were satisfiable (checked above) but the serving
            # constraints removed every grid rung. That is an INFEASIBLE
            # problem, not a reason to relax an SLO silently: policy §1 makes
            # these hard constraints. Write the evidence, then exit naming the
            # limits that bound.
            binding = sorted({
                str(r["binding_constraint"]) for r in serve_rejections
                if r.get("binding_constraint")
            })
            selection.update({
                "feasible": False,
                "serve_constraints_infeasible": True,
                "serve_constraints": {
                    **serve_context.stamp_inactive(),
                    "active": True,
                    "reason": "no_grid_rung_satisfied_the_serving_constraints",
                    "binding_constraints": binding,
                    "rejected_assignments": serve_rejections,
                },
            })
            sel_path.write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n")
            raise SystemExit(
                "[alloc] every allocation that fits the "
                f"{args.target_disk_gb:.3f}GB card misses a hard serving "
                f"constraint ({', '.join(binding) or 'unpriced'}). Policy §1 "
                "makes these constraints, not penalties: nothing here is "
                "'close enough'. Raise the SLO, widen --formats, change "
                "--serve-workload-mix, or supply a dispatch table that prices "
                f"the families in this menu. Selection written to {sel_path}.")

        # Near-lossless cap: all of the most expensive format, +1 bit of slack
        # so the DP can actually reach it.
        search_hi_cap = float(max(int(s.weight_bits) for s in specs_sorted)) + 1.0
        # Bound the ratchet by the cheapest grid rung that does NOT fit:
        # bisecting toward the near-lossless cap on a card that only fits
        # the low rungs wastes dozens of expensive DP solves at high-bin
        # targets. HEURISTIC: this assumes disk(target) is only LOCALLY
        # non-monotone — if a fitting allocation existed above the first
        # non-fitting rung, the tightened ceiling would forgo it, and the
        # min-Δloss objective makes that premise weaker, not stronger. The
        # ratchet never accepts a probe worse than the proven grid pick, so
        # the downside is bounded at shipping the grid pick, never worse —
        # but it IS a forgone option, so both the cap and the tightening are
        # recorded in selection.json rather than left invisible.
        over_budget = [
            c for c in grid
            if c["whole_artifact_upper_bound_bytes"] > budget_bytes
        ]
        tightening_rung = (
            min(float(c["target_bits"]) for c in over_budget)
            if over_budget else None)
        search_hi = (search_hi_cap if tightening_rung is None
                     else min(search_hi_cap, tightening_rung))

        _consider(grid_pick, float(grid_pick["target_bits"]), "grid_pick")
        top = _artifact_for_target(search_hi)  # densest possible (all-expensive)
        has_slack = _consider(top, search_hi, "search_hi_cap")
        bisection_skipped_reason = None
        if has_slack:
            # The cap itself fits the card: nothing denser exists to bisect
            # toward. NB the cap is only *considered*, not forced — if the
            # grid pick has strictly lower predicted Δloss it still ships.
            bisection_skipped_reason = "search_hi_fits_budget"
        elif search_hi <= float(grid_pick["target_bits"]) + 0.005:
            # The tightened ceiling landed at/below the grid pick's target, so
            # the bracket is empty and the bisection below would break on
            # iteration 0 — it ships the grid pick with no exploration. That
            # used to be silent; name it.
            bisection_skipped_reason = (
                "tightened_search_hi_at_or_below_grid_pick_target")
        else:
            a_t, b_t = float(grid_pick["target_bits"]), search_hi
            for _ in range(40):
                if (b_t - a_t) <= 0.005:
                    break
                mid = 0.5 * (a_t + b_t)
                if _consider(_artifact_for_target(mid), mid, "bisect"):
                    a_t = mid
                else:
                    b_t = mid
        chosen_info = best

        max_bytes_pick = sel["chosen"]  # what "fill the card" would have shipped
        args.target_bits = float(emit_target)  # override emit target below
        selected_whole_artifact_budget_stamp = whole_artifact_budget_stamp(
            budget_bytes=budget_bytes,
            selection_tensor_payload_bytes=int(
                chosen_info["tensor_payload_bytes"]
            ),
            selection_non_tensor_reserve_bytes=overhead_reserve_bytes,
            selection_assignment=chosen_info["assignment"],
            excluded_source_prefixes=(
                getattr(args, "exclude_source_prefix", None) or ()
            ),
        )
        selection.update({
            "has_slack": bool(has_slack),
            "chosen_target_bits": float(emit_target),
            "chosen_achieved_bits": float(chosen_info["achieved_bits"]),
            "predicted_tensor_payload_gb": (
                chosen_info["tensor_payload_bytes"] / _fp.GB
            ),
            "predicted_whole_artifact_upper_bound_gb": (
                chosen_info["whole_artifact_upper_bound_bytes"] / _fp.GB
            ),
            "predicted_floor_gb": chosen_info["floor_bytes"] / _fp.GB,
            "predicted_body_tensor_payload_gb": (
                chosen_info["tensor_payload_bytes"]
                - chosen_info["floor_bytes"]
            ) / _fp.GB,
            "predicted_dloss": float(chosen_info["dloss"]),
            "predicted_dloss_scope": (
                "dp_body_items_only__excludes_fixed_auxiliary_mtp_visual"),
            "selection_headroom_gb": (
                budget_bytes
                - chosen_info["whole_artifact_upper_bound_bytes"]
            ) / _fp.GB,
            "grid_pick_target_bits": float(grid_pick["target_bits"]),
            "grid_pick_dloss": float(grid_pick["dloss"]),
            # The tightened ceiling the ratchet actually searched under, the
            # untightened near-lossless cap, and what tightened it: a denser
            # allocation forgone by the tightening leaves a trace here.
            "search_hi_target_bits": float(search_hi),
            "search_hi_cap_target_bits": float(search_hi_cap),
            "search_hi_tightened_by_rung": (
                float(tightening_rung) if tightening_rung is not None else None),
            "search_hi_tightened": bool(search_hi < search_hi_cap),
            "bisection_ran": bool(bisection_skipped_reason is None),
            "bisection_skipped_reason": bisection_skipped_reason,
            "ratchet_trace": ratchet_trace,
            # The retired MAX-bytes objective, kept for auditability: when
            # these agree the objective change was a no-op on this run.
            "max_bytes_pick_target_bits": (
                float(max_bytes_pick["target_bits"])
                if max_bytes_pick is not None else None),
            "max_bytes_pick_dloss": (
                float(max_bytes_pick["dloss"])
                if max_bytes_pick is not None else None),
            "max_bytes_grid_pick_agrees": bool(
                max_bytes_pick is not None
                and max_bytes_pick["target_bits"] == grid_pick["target_bits"]),
            "whole_artifact_budget": selected_whole_artifact_budget_stamp,
            # Ultraplan P5a/P5b provenance for the SHIPPED assignment: how
            # each selected unit's activation cost was priced, whether this
            # run's per-family ladder fits are cross-comparable, and which
            # selected rungs ride a backed fused mid-M lane vs the
            # expand+GEMM fallback.
            "activation_fair_pricing": activation_pricing.as_dict(),
            "cb_ladder_cross_family_verdict": cross_family_verdict,
            "serving_lane_provenance": selection_serving_lane_provenance(
                chosen_info["assignment"], candidates, target_profile),
            # Ultraplan P5c: which hard serving constraints were active, which
            # probed assignments the axis REJECTED and for which SLO, and
            # which constraint binds at the shipped optimum. Present on every
            # run; when no table/SLO was supplied it says exactly that and
            # nothing else in this file differs from the pre-P5c allocator.
            "serve_constraints": (
                {
                    **chosen_info["serve"].as_dict(),
                    "rejected_assignments": serve_rejections,
                    "n_probes_rejected_by_serving_constraints": len(
                        serve_rejections),
                    "binding_constraint_at_optimum": (
                        chosen_info["serve"].binding_constraint),
                    "fastest_feasible_reference": fastest_feasible_summary(
                        serve_probe_rows,
                        scope_note=(
                            "Scope: the assignments this byte-budget ratchet "
                            "PROBED (grid rungs, the near-lossless cap, and "
                            "the bisection midpoints) — not an enumeration of "
                            "the globally feasible set. Treat it as the "
                            "fastest feasible assignment SEEN, and say so "
                            "wherever it is used as a denominator."
                        ),
                    ),
                }
                if serve_context.active
                else serve_context.stamp_inactive()
            ),
        })
        sel_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(
            f"[alloc] byte-budget: card={args.target_disk_gb:.2f}GB ({regime} src) "
            f"-> ship {chosen_info['achieved_bits']:.3f} bpp "
            f"(selection upper bound "
            f"{chosen_info['whole_artifact_upper_bound_bytes'] / _fp.GB:.3f}GB: "
            f"tensor floor "
            f"{chosen_info['floor_bytes'] / _fp.GB:.3f} + body "
            f"{(chosen_info['tensor_payload_bytes'] - chosen_info['floor_bytes']) / _fp.GB:.3f} "
            f"+ reserve {overhead_reserve_bytes / _fp.GB:.3f}, "
            f"headroom {(budget_bytes - chosen_info['whole_artifact_upper_bound_bytes']) / _fp.GB:.3f}GB, "
            f"Δloss={chosen_info['dloss']:.4e} = min over "
            f"{sum(1 for r in ratchet_trace if r['fits'])} fitting probes)"
            + ("  [card has slack beyond near-lossless]" if has_slack else "")
            + (f"  [search_hi tightened {search_hi_cap:.3f}->{search_hi:.3f} by "
               f"the cheapest non-fitting rung; denser allocations above it "
               f"were not explored]" if search_hi < search_hi_cap else "")
            + (f"  [bisection skipped: {bisection_skipped_reason}]"
               if bisection_skipped_reason else "")
            + f" -> {sel_path}",
            flush=True,
        )
        if serve_context.active:
            chosen_serve = chosen_info["serve"]
            print(
                "[alloc] serving constraints: "
                + "; ".join(
                    f"{c.name} {c.predicted if c.predicted is None else round(c.predicted, 4)}"
                    f" {c.direction} {c.limit} {c.units} "
                    f"[{'ok' if c.satisfied else 'VIOLATED'}]"
                    for c in chosen_serve.checks
                )
                + f" | binding={chosen_serve.binding_constraint} | "
                + f"{len(serve_rejections)} probed assignment(s) rejected by "
                + "the SLO axis. Proposal data (table-driven); the served "
                + "NATIVE-PARITY protocol is the release gate.",
                flush=True,
            )

    # Emit chosen layer_config for target_bits.
    assignment, achieved, total, mutable_total = _solve_for_target(args.target_bits)
    if assignment is None:
        # An infeasibility exit must say WHICH wall was hit: a target under the
        # format floor needs a different --formats menu, while a target the
        # floor clears but serving-group promotion overshoots needs a slightly
        # looser target. The solver recorded both; report them.
        d = _solve_diagnostics.get(round(float(args.target_bits), 9), {})
        floor = d.get("min_bits")
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}."
            + (f" The format floor (cheapest legal format everywhere) is "
               f"{floor:.3f} bpp" if isinstance(floor, (int, float)) else "")
            + (f", and that floor solve itself promotes to "
               f"{d['floor_achieved_bits']:.3f} bpp"
               if isinstance(d.get("floor_achieved_bits"), (int, float)) else "")
            + (f". Closest of {d.get('evals', 0)} DP solves: "
               f"{d['closest_achieved_bits']:.3f} bpp"
               if isinstance(d.get("closest_achieved_bits"), (int, float))
               else "")
            + (f" against an overshoot tolerance of "
               f"{d['overshoot_tolerance']}" if "overshoot_tolerance" in d
               else "")
            + ". Raise --target-bits above the achievable value above, or "
              "widen --formats so the floor drops.")
    print(
        f"[alloc] target_bits={args.target_bits}: "
        f"achieved_bits={achieved:.3f}, Δloss={total:.3e}",
        flush=True,
    )

    assignment_expanded = dict(assignment)

    # Expand packed-serving-group super-items back to per-tensor entries.
    if not args.no_packed_aggregation:
        assignment_expanded = expand_packed_group_assignment(
            assignment_expanded, stats)

    # Expand fused-sibling super-Linears (qkv_proj / gate_up_proj).
    if not args.no_fused_aggregation:
        assignment_expanded = expand_fused_sibling_assignment(
            assignment_expanded, stats)

    assignment_expanded.update(fixed_format_assignment)
    assignment_before_serving_promotion = dict(assignment_expanded)

    # vLLM's FusedMoE requires all projections of the same expert to share
    # one scheme. Packed groups are first-class DP units, so this promotion
    # is a validated no-op (validate_final_serving_promotion_noop below);
    # it stays as the serve-time coherence backstop for the un-aggregated
    # paths (--no-packed-aggregation / --no-fused-aggregation). It is handed
    # per-Linear legality so the shared format it lands on is runnable for
    # every member, and it is still a no-op here by construction: an
    # aggregated unit's format came from the intersection of its members'
    # candidate sets, and the un-aggregated paths were already promoted
    # against the same sets inside solve_with_promotion.
    assignment_expanded = promote_serving_units(
        assignment_expanded,
        format_rank,
        profile=model_profile,
        legal_formats=per_linear_legal_formats,
    )
    validate_final_serving_promotion_noop(
        assignment_before_serving_promotion,
        assignment_expanded,
    )

    # Visual-encoder Linears are auxiliary to the language-model budget.
    # Stamp them with --visual-format for export, but keep them out of the
    # body DP frontier, default bpp, and default Δloss.
    visual_format = visual_format_canonical
    visual_sensitivity = args.visual_sensitivity

    def _visual_fisher_available(stats_d: dict, costs_d: dict) -> bool:
        """True when both the probe and cost pickles carry real visual
        entries — the signal a multimodal calibration pass ran."""
        any_visual_stats = any(_is_visual_linear(n) for n in stats_d)
        any_visual_costs = any(_is_visual_linear(n) for n in costs_d)
        return any_visual_stats and any_visual_costs

    if visual_sensitivity == "fisher" and visual_names:
        print(
            "[alloc] --visual-sensitivity=fisher found visual Linears, "
            "but visual assignments are auxiliary to the body budget; "
            f"using --visual-format={visual_format} for the layer_config.",
            flush=True,
        )
    elif visual_sensitivity == "fisher" and not _visual_fisher_available(stats, costs):
        print("[alloc] --visual-sensitivity=fisher requested but probe / "
              "cost pickles have no visual Linear entries; falling back "
              f"to --visual-format={visual_format} (Phase 1 uniform).",
              flush=True)

    visual_names_src = sorted(source_visual_stats)

    if visual_names_src:
        for vname in visual_names_src:
            assignment_expanded[vname] = visual_format
        print(f"[alloc] --visual-format={visual_format}: assigned "
              f"{len(visual_names_src)} visual Linears uniformly "
              f"(source={probe_model_path})", flush=True)
    elif visual_format != "BF16":
        print(f"[alloc] --visual-format={visual_format}: no visual "
              f"Linears found in source checkpoint — override is a "
              f"no-op", flush=True)

    _validate_assignment_candidate_membership(
        assignment_expanded,
        candidates,
        fixed_chosen_candidates=fixed_chosen_candidates,
    )

    mtp_count = sum(1 for n in assignment_expanded if n.startswith("mtp."))
    if mtp_count:
        mtp_fmts = {
            assignment_expanded[n]
            for n in assignment_expanded
            if n.startswith("mtp.")
        }
        expected_mtp_fmts = {mtp_format_canonical}
        if mtp_fmts != expected_mtp_fmts:
            raise AssertionError(
                f"MTP assignment drifted after fixed-format accounting: "
                f"expected {sorted(expected_mtp_fmts)}, got "
                f"{sorted(mtp_fmts)}"
            )
        print(
            f"[alloc] --mtp-format={mtp_format_canonical}: assigned "
            f"{mtp_count} MTP Linears uniformly",
            flush=True,
        )

    if fixed_lm_head_names:
        head_fmts = {
            assignment_expanded[name]
            for name in fixed_lm_head_names
            if name in assignment_expanded
        }
        if head_fmts != {lm_head_format_canonical}:
            raise AssertionError(
                "lm_head assignment drifted after fixed-format accounting: "
                f"expected {[lm_head_format_canonical]}, got "
                f"{sorted(head_fmts)}"
            )
        print(
            f"[alloc] --lm-head-format={lm_head_format_canonical}: assigned "
            f"{len(fixed_lm_head_names)} lm_head Linear(s) uniformly",
            flush=True,
        )

    # Passthrough-integrity belt-and-suspenders. The filter in
    # build_candidates drops mismatched FP8_SOURCE / BF16 per-Linear
    # candidate, but downstream aggregation + promotion (fused
    # siblings, MoE expert-unity) can in principle push a format onto
    # a group whose members have heterogeneous source dtypes. On
    # modern checkpoints this doesn't happen (siblings share source
    # dtype), but if it ever does we want a loud early failure rather
    # than a broken export artifact.
    if source_manifest:
        violations: list[tuple[str, str, str]] = []
        for name, fmt in assignment_expanded.items():
            if not _is_passthrough_format(fmt):
                continue
            kind = source_manifest.get(name)
            if kind is None:
                # Visual and MTP assignments are stamped as auxiliary formats
                # outside the language-model source manifest by design.
                if _is_visual_linear(name) or _is_mtp_linear(name):
                    continue
                kind = "unknown"
            if not _passthrough_source_ok(fmt, kind):
                violations.append((name, fmt, kind))
        if violations:
            head = "\n  ".join(
                f"{n}: picked {f} but source is {k} "
                f"(requires {PASSTHROUGH_SOURCE_REQUIREMENTS[f]})"
                for n, f, k in violations[:10]
            )
            raise SystemExit(
                f"[alloc] passthrough-integrity violation: "
                f"{len(violations)} Linears have a passthrough format "
                f"picked over a mismatched source dtype. Sample:\n"
                f"  {head}\n"
                "The per-Linear filter should have excluded these — "
                "investigate fused-sibling / MoE-unity promotion. Note: "
                "fp16/fp32 sources have NO passthrough format by design "
                "(fp16→bf16 drops 3 mantissa bits — not lossless); allocate "
                "a quantized format for them or extend "
                "PASSTHROUGH_SOURCE_REQUIREMENTS deliberately."
            )

    final_body_assignment = {
        name: fmt
        for name, fmt in assignment_expanded.items()
        if (
            not _is_visual_linear(name)
            and not _is_mtp_linear(name)
            and name not in fixed_lm_head_names
        )
    }
    final_body_payload = _assignment_payload_totals(
        final_body_assignment,
        require_all_stats=True,
    )
    final_body_achieved = float(final_body_payload["bits_per_param"])
    if not math.isclose(
        final_body_achieved,
        float(achieved),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssertionError(
            "final expanded assignment payload does not reconcile with the "
            f"exact-filtered solve: final={final_body_achieved}, "
            f"solve={achieved}"
        )
    final_assignment_payload = _assignment_payload_totals(
        assignment_expanded,
        require_all_stats=False,
    )

    # ---- Hard serving constraints on the SHIPPED assignment (P5c) ----
    # The byte-budget selector already filtered its probes; this is the check
    # on the emit path, which the plain --target-bits mode reaches without any
    # ratchet. Policy §1 makes an SLO miss INFEASIBLE, so a violation exits
    # rather than shipping a layer_config that quietly does not meet the
    # deployment constraint the operator stated. Inactive -> no evaluation, no
    # stamp, no behaviour change.
    final_serve_feasibility = None
    if serve_context.active:
        final_serve_feasibility = _serve_feasibility(
            final_body_assignment,
            resident_bytes=int(
                round(float(final_body_payload["bits_total"]) / 8.0)),
        )
        if not final_serve_feasibility.feasible:
            violated = ", ".join(
                f"{c.name}: predicted={c.predicted} {c.direction} "
                f"{c.limit} {c.units}"
                + (f" (unpriced: {c.unpriced_reason})"
                   if c.unpriced_reason else "")
                for c in final_serve_feasibility.violations
            )
            raise SystemExit(
                "[alloc] ERROR: the emitted assignment at "
                f"target_bits={args.target_bits} misses a hard serving "
                f"constraint — {violated}. Binding: "
                f"{final_serve_feasibility.binding_constraint}. Policy §1 "
                "(docs/lanes/nvfp4-cb/format-speed-policy.md) makes these "
                "hard constraints, not penalties: there is no λ that trades "
                "them against predicted Δloss. Raise the SLO, widen "
                "--formats, adjust --serve-workload-mix, or supply a dispatch "
                "table that prices this menu's format families."
            )

    final_cb_serialization_stamps = _cb_stamps_for_assignment(
        assignment_expanded
    )
    final_cb_render_identity = _cb_render_identity_for_assignment(
        assignment_expanded
    )
    layer_cfg = {}
    for name, fmt in assignment_expanded.items():
        if fmt in format_specs:
            layer_cfg[name] = format_specs[fmt].autoround_config()
        else:
            # Visual format outside the body's format set (e.g., user
            # passed --formats NVFP4,BF16 plus --visual-format MXFP8_E4M3).
            # Resolve from the global registry.
            layer_cfg[name] = fr.get_format(fmt).autoround_config()
        if name in final_cb_serialization_stamps:
            layer_cfg[name][CB_TENSOR_IDENTITY_FIELD] = (
                final_cb_serialization_stamps[name]
            )

    # The resolved serving profile travels WITH the assignment (re-vet R11 /
    # debt D4). Before this, it landed only in the side report
    # format_applicability.json, which export never reads, so the exporter
    # re-resolved the profile from the architecture spec and could legality-
    # audit under a different one than the allocator solved with — measured
    # 2026-07-11: 226 dense FP8 Linears silently coerced to BF16 on the Hy3
    # compressed-tensors export. PRISMAQUANT_TARGET_PROFILE remains the
    # override for direct exporter invocations.
    layer_cfg[LAYER_CONFIG_META_KEY] = {
        "schema": "prismaquant.layer_config_meta.v1",
        "target_profile": target_profile,
        "target_profile_requested": args.target_profile,
        "target_profile_default": str(args.target_profile_default or "research"),
        "lm_head_format": lm_head_format_canonical,
        "lm_head_mode": (
            "dp" if lm_head_dp_unpinned
            else "fixed" if fixed_lm_head_quantized
            else "profile_pinned_bf16"
        ),
        "lm_head_cost_pricing": fixed_lm_head_cost_pricing,
        "target_bits": float(args.target_bits),
        "achieved_bits": final_body_achieved,
        "achieved_bits_scope": (
            "body_assignment_tensor_payload_including_deduplicated_cb_sidecars"
        ),
        "body_assignment_payload_bits_total": float(
            final_body_payload["bits_total"]
        ),
        "body_assignment_quantizable_params": int(
            final_body_payload["quantizable_params"]
        ),
        "body_shared_cb_sidecar_bits": float(
            final_body_payload["cb_shared_sidecar_bits"]
        ),
        "solver_contract": (
            "additive_candidate_proposal_then_exact_assignment_filter"
        ),
        "global_optimality_claimed": False,
        # The artifact-wide CB context the per-tensor identities above were
        # computed under. Without it the exporter cannot know which contract
        # produced them: it reads this key
        # (`cb_serialization_metadata_from_assignment_payload`) to decide
        # whether to claim the W4A4 activation contract. Absent, it falls back
        # to no-contract/payload-v2, drops the 4-byte input_global_scale, and
        # EVERY per-tensor stamp mismatches -- measured 2026-08-08 on DSv4:
        # 24851/24851 mismatched, which is the root cause of the four earlier
        # "CB per-layer serialization identity mismatch" export failures.
        **({"cb_serialized_payload": cb_serialization_context_stamp(
                cb_serialization_context,
                formats=sorted({
                    str(fmt) for fmt in assignment_expanded.values()
                    if is_cb_format(str(fmt))
                }) or None,
            )}
           if cb_serialization_context is not None
           and final_cb_serialization_stamps else {}),
        **propagated_cost_provenance(research_cost_provenance),
        "assignment_payload_bits_total": (
            float(final_assignment_payload["bits_total"])
            if not final_assignment_payload["missing_stats_names"]
            else None
        ),
        "assignment_payload_bits_scope": (
            "all_assignment_tensor_payload_including_deduplicated_cb_sidecars"
        ),
        "assignment_payload_missing_stats_names": final_assignment_payload[
            "missing_stats_names"
        ],
        **({
            "cb_serialized_payload": final_cb_render_identity[
                "cb_serialized_payload"
            ],
            "cb_render_identity": final_cb_render_identity,
        } if final_cb_render_identity is not None else {}),
        **({
            "whole_artifact_budget": selected_whole_artifact_budget_stamp,
        } if selected_whole_artifact_budget_stamp is not None else {}),
        # Only when the constraint axis actually ran, so an unconstrained run
        # writes byte-identical layer-config metadata (the "constraints were
        # absent" stamp lives in selection.json, which every byte-budget run
        # writes anyway).
        **({
            "serve_constraints": final_serve_feasibility.as_dict(),
        } if final_serve_feasibility is not None else {}),
    }

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    counts = defaultdict(int)
    for fmt in assignment_expanded.values():
        counts[fmt] += 1
    print(
        f"\n[alloc] target={args.target_bits} "
        f"exact_assignment_payload_bpp={final_body_achieved:.3f}"
    )
    for fmt, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:>14}: {n:>5} layers")
    print(f"\nLayer config → {out}")
    print(f"Feed to AutoRound via --layer_config {out}")

    # Optional read-only "where did the budget go?" attribution over the final
    # resolved body assignment. Derived from already-resolved data; no re-probe.
    _write_bit_attribution_reports(
        args.bit_attribution_json,
        args.bit_attribution_csv,
        target_bits=args.target_bits,
        achieved_bits=final_body_achieved,
        assignment_expanded=assignment_expanded,
        candidates=candidates,
        stats_entry_for=_stats_entry_for_assignment_name,
        format_specs=format_specs,
        cb_serialization_context=cb_serialization_context,
    )


if __name__ == "__main__":
    main()
