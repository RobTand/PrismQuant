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
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import (
    Candidate,
    _shape_from_stats,
    compute_achieved,
    compute_assignment_predicted_dloss,
    promote_fused,
    promote_moe_pair,
    promote_serving_units,
    solve_with_promotion,
)
from .allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _FUSED_SIBLING_MARKER,
    _format_kernel_supports_shape,
    _is_passthrough_format,
    _passthrough_source_ok,
    _scan_source_dtype_manifest,
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
    summarize_applicability_masks,
)
from .serving_profiles import (
    check_serving_format,
    resolve_target_profile,
    serving_profile_names,
)
from .decision_units import block_id_from_qname
from .schemas import validate_cost_payload, validate_probe_payload



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
    finite_positive = [
        float(v) for v in y
        if math.isfinite(float(v)) and float(v) > 0.0
    ]
    if not finite_positive:
        return [0.0 for _ in y]
    floor = max(min(finite_positive) * 1.0e-6, 1.0e-300)
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
    if ymax - ymin < 1.0:
        return 0
    threshold = 0.5 * (ymin + ymax)
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
        "log_error": _record("log_error", log_idx),
        "global_log_error": _record("global_log_error", global_log_idx),
        "raw_linear": _record("raw_linear", raw_idx),
    }




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
    body_bits = 0.0
    body_params = 0

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
        if cand is not None:
            bits = 8.0 * cand.memory_bytes
            pred_dloss = float(getattr(cand, "predicted_dloss", 0.0))
        elif isinstance(entry, dict):
            mem_map = entry.get("_memory_bytes_by_format")
            if isinstance(mem_map, dict) and fmt in mem_map:
                bits = 8.0 * mem_map[fmt]
            elif fmt in format_specs and n_params:
                shape = _shape_from_stats(entry)
                bits = format_specs[fmt].effective_bits_for_shape(shape) * n_params

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
            body_bits += bits
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
    totals = {
        "body_bits": body_bits,
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
) -> None:
    """Write the bit-attribution JSON / CSV and print a compact per-role rollup.

    No-op when neither output path is set."""
    if not json_path and not csv_path:
        return
    buckets, per_linear, totals = _build_bit_attribution(
        assignment_expanded, candidates, stats_entry_for, format_specs)

    if json_path:
        payload = {
            "schema": "prismaquant.allocator.bit_attribution.v1",
            "target_bits": float(target_bits),
            "achieved_bits": float(achieved_bits),
            "body_bits_per_param": totals["body_bits_per_param"],
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


def discover_visual_linears_from_source(model_path: str) -> list[str]:
    """Scan the source safetensors index for `model.visual.blocks.*.weight`
    entries with rank-2 shapes — these are the Linear modules the visual
    encoder exposes.

    Returned names are the basename (`.weight` stripped) so they slot
    directly into the allocator's assignment dictionary and the exporter's
    quantize-by-recipe dispatch.

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
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
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
        except ImportError:
            return []
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        import os as _os
        if not src.exists():
            return []
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _VISUAL_PREFIX_RE.match(k):
                        continue
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))

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
    out: list[str] = []
    for name, shape in candidates:
        if len(shape) != 2:
            continue
        if _NON_LINEAR_RE.search(name):
            continue
        out.append(name[:-len(".weight")] if name.endswith(".weight") else name)
    return sorted(set(out))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--model-override", default=None,
                    help="Override the model path stored in probe.pkl's meta. "
                         "Useful when re-running allocator against a probe "
                         "whose container-side paths no longer exist (e.g., "
                         "the original source was at /src/qwen36 in a prior "
                         "container run but is now only accessible via a "
                         "different mount). Overrides both profile detection "
                         "and visual-Linear source discovery.")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
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
    ap.add_argument("--calibration", default=None,
                    help="Optional path to a JSON containing "
                         "'calibrated_gains[fmt] = α_fmt'. When present, "
                         "the per-(layer, format) predicted Δloss "
                         "is multiplied by α_fmt before the DP runs.")
    ap.add_argument("--propagated-sensitivity-report", default=None,
                    help="Optional sensitivity_propagated_group_report JSON. "
                         "When provided, propagated KL is folded into the "
                         "allocator costs before candidate construction.")
    ap.add_argument("--propagated-sensitivity-scale", type=float, default=1.0,
                    help="Multiplier for --propagated-sensitivity-report "
                         "penalties. The report's current-format unit KL "
                         "is preserved at scale=1 and distributed once over "
                         "fused siblings by added-bit share.")
    ap.add_argument("--propagated-sensitivity-score-field",
                    default="propagated_kl",
                    help="Numeric report row field to fold into allocator "
                         "costs when --propagated-sensitivity-report is set.")
    ap.add_argument("--propagated-sensitivity-format-extrapolation",
                    choices=("local_mse_ratio", "current_only", "bits_interp"),
                    default="local_mse_ratio",
                    help="How to extrapolate measured current-format "
                         "propagated sensitivity across alternative "
                         "candidate formats.")
    ap.add_argument("--propagated-sensitivity-target-format", default=None,
                    help="Override target format for propagated-sensitivity "
                         "bit-share accounting. Defaults to the report's "
                         "target_format.")
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
    ap.add_argument("--bit-attribution-json", default=None,
                    help="Optional path: write a read-only 'where did the "
                         "budget go' report bucketing the final body "
                         "assignment by (block, role) with per-bucket bpp, "
                         "format mix, summed predicted_dloss and h_trace.")
    ap.add_argument("--bit-attribution-csv", default=None,
                    help="Optional path: write the flat per-Linear bit "
                         "attribution table (qname, block, role, format, "
                         "bits, bpp, n_params, h_trace, predicted_dloss).")
    args = ap.parse_args()

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
    if probe_model_path:
        model_profile = detect_profile(probe_model_path)
        print(f"[alloc] model profile: {model_profile.name} "
              f"(derived from {probe_model_path})", flush=True)
    target_profile = resolve_target_profile(model_profile, args.target_profile)
    if target_profile not in serving_profile_names():
        raise SystemExit(f"[alloc] ERROR: unknown target profile {target_profile!r}")
    print(f"[alloc] target profile: {target_profile}", flush=True)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    validate_probe_payload(probe, args.probe)
    validate_cost_payload(cost_data, args.costs)
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    allocation_excluded = []
    for name in sorted(set(stats) | set(costs)):
        if model_profile.is_pinned_name(name):
            allocation_excluded.append(name)
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
    accounting_stats = dict(stats)

    propagated_sensitivity_summary: dict | None = None
    if args.propagated_sensitivity_report:
        from .propagated_sensitivity_costs import (
            apply_propagated_sensitivity_penalty,
        )

        with open(args.propagated_sensitivity_report) as f:
            propagated_report = json.load(f)
        costs, propagated_sensitivity_summary = apply_propagated_sensitivity_penalty(
            costs,
            stats=stats,
            report=propagated_report,
            scale=float(args.propagated_sensitivity_scale),
            target_format=args.propagated_sensitivity_target_format,
            score_field=args.propagated_sensitivity_score_field,
            format_extrapolation=args.propagated_sensitivity_format_extrapolation,
        )
        print(
            "[alloc] propagated sensitivity costs: "
            f"report={args.propagated_sensitivity_report} "
            f"scale={propagated_sensitivity_summary['scale']:.6g} "
            f"adjusted={propagated_sensitivity_summary['adjusted_entries']} "
            f"skipped={propagated_sensitivity_summary['skipped']} "
            f"penalty={propagated_sensitivity_summary['total_scaled_member_penalty']:.6g}",
            flush=True,
        )

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)

    # --- Format-family coherence check -----------------------------------
    # A sensible format ladder has at most ONE format per bit tier. Having
    # both NVFP4 and MXFP4 (or MXFP6_E3M2 and MXFP6_E2M3) means the allocator
    # picks between them based on tiny measurement noise per-layer, which
    # produces a serving mess: two separate kernel paths for the same tier.
    #
    # We bucket formats by effective_bits rounded to 0.25 and warn when a
    # bucket has more than one member. If --enforce-family-coherence is
    # set we error instead.
    from collections import Counter as _Counter
    buckets: dict[float, list[str]] = {}
    for s in specs_sorted:
        key = round(s.effective_bits * 4) / 4
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
    rank_specs.setdefault(mtp_format_canonical, fr.get_format(mtp_format_canonical))
    rank_specs.setdefault(visual_format_canonical, fr.get_format(visual_format_canonical))
    rank_specs_sorted = sorted(rank_specs.values(), key=lambda s: s.effective_bits)
    format_rank = {s.name: i for i, s in enumerate(rank_specs_sorted)}
    format_specs = {s.name: s for s in rank_specs_sorted}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({s.effective_bits:.2f}b)' for s in specs_sorted]}")

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

    # Source-dtype manifest drives passthrough-integrity filtering in
    # build_candidates. None when model path is unknown — candidates
    # fall back to cost-pickle-only gating (pre-passthrough behavior).
    source_manifest: dict[str, str] | None = None
    if probe_model_path:
        source_manifest = _scan_source_dtype_manifest(
            probe_model_path, model_profile)
        if source_manifest:
            n_fp8 = sum(1 for v in source_manifest.values() if v == "fp8")
            n_bf16 = sum(1 for v in source_manifest.values() if v == "bf16")
            print(f"[alloc] source-dtype manifest: {n_fp8} fp8, "
                  f"{n_bf16} bf16 (gates FP8_SOURCE/BF16 per source)",
                  flush=True)

    candidate_mask_records: list[dict] = []
    candidates = build_candidates(
        stats, costs, specs_sorted, calibrated_gains,
        source_manifest=source_manifest,
        target_profile=target_profile,
        mask_records=candidate_mask_records,
    )
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    fixed_format_assignment: dict[str, str] = {}
    fixed_stats: dict[str, dict] = {}
    fixed_chosen_candidates: dict[str, Candidate] = {}
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
        fixed_format_assignment = {
            name: mtp_format_canonical for name in mtp_names
        }
        fixed_stats = {name: stats[name] for name in mtp_names}
        fixed_chosen_candidates = {
            name: _find_candidate_for_format(
                mtp_candidates,
                name,
                mtp_format_canonical,
            )
            for name in mtp_names
        }
        fixed_chosen_candidates = {
            name: cand for name, cand in fixed_chosen_candidates.items()
            if cand is not None
        }
        if fixed_format_assignment:
            fixed_names = set(fixed_format_assignment)
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
                f"{len(fixed_format_assignment)} MTP Linears before DP "
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

    def _bits_for_stats_entry(entry: dict, fmt: str) -> float:
        shape = _shape_from_stats(entry)
        return 8.0 * fr.get_format(fmt).memory_bytes_for_shape(shape)

    fixed_total_bits = sum(
        _bits_for_stats_entry(fixed_stats[name], fmt)
        for name, fmt in fixed_format_assignment.items()
        if name in fixed_stats
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

    # Pre-aggregate fused siblings (qkv_proj, gate_up_proj, ...) into
    # single DP items. The DP can't pick mixed-sibling solutions because
    # there's only one item per group — so promote_fused becomes a no-op
    # on aggregated items and the overshoot-tightening loop collapses to
    # a single pass on well-behaved models. Must run AFTER the MoE
    # aggregation (it skips `.__fused__.` entries explicitly).
    if not args.no_fused_aggregation:
        stats, costs, candidates = aggregate_fused_siblings(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains)
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
        "propagated_sensitivity_costs": propagated_sensitivity_summary,
        "pre_aggregation_candidate_availability": pre_aggregation_availability,
        "post_aggregation_candidate_availability": post_aggregation_availability,
        "fixed_format_assignment": {
            "mtp_format": mtp_format_canonical,
            "visual_format": visual_format_canonical,
            "linears": len(fixed_format_assignment),
            "linears_by_kind": dict(Counter(
                "mtp" if _is_mtp_linear(name)
                else "visual" if _is_visual_linear(name)
                else "other"
                for name in fixed_format_assignment
            )),
            "params": fixed_total_params,
            "bits_total": fixed_total_bits,
            "predicted_dloss": fixed_total_dloss,
            "budget_scope": "auxiliary_excluded_from_body_budget",
        },
        **summarize_applicability_masks(candidate_mask_records),
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

    def _solve_for_target(target_bits: float):
        """Solve the body-budget DP at one target bit budget."""
        mutable_target_bits = float(target_bits)
        if mutable_total_params <= 0:
            if fixed_total_params > 0 and mutable_target_bits >= 0.0:
                return {}, 0.0, 0.0, 0.0
            return None, float("nan"), float("inf"), float("inf")
        if mutable_target_bits < 0.0:
            return None, float("nan"), float("inf"), float("inf")
        assign, achieved_r = solve_with_promotion(
            stats, candidates, mutable_target_bits, format_specs, format_rank,
            args.bit_precision,
            no_fused_promote=args.no_fused_promote,
            overshoot_tolerance=args.overshoot_tolerance,
            profile=model_profile,
        )
        if assign is None:
            return None, float("nan"), float("inf"), float("inf")
        mutable_total = compute_assignment_predicted_dloss(assign, candidates)
        total = mutable_total
        return assign, achieved_r, total, mutable_total

    def _expand_assignment_for_seed_json(
        assignment: dict[str, str],
        *,
        include_auxiliary: bool = True,
    ) -> dict[str, str]:
        """Expand DP super-items into the per-Linear seed-assignment shape.

        The allocator can solve over fused-sibling super-items.
        The KL probe's seed path wants ordinary module qnames; it already
        handles legality, pinning, and fused coherence, but giving it expanded
        names preserves the intended frontier point instead of making the
        super-item markers look like unknown entries.
        """
        expanded = dict(assignment)
        if not args.no_fused_aggregation:
            expanded = expand_fused_sibling_assignment(expanded, stats)
        if include_auxiliary:
            expanded.update(fixed_format_assignment)
        return promote_serving_units(expanded, format_rank, profile=model_profile)

    def _assignment_bits_total(assignment: dict[str, str]) -> float:
        total = 0.0
        for name, fmt in assignment.items():
            entry = _stats_entry_for_assignment_name(name)
            if not isinstance(entry, dict):
                continue
            shape = _shape_from_stats(entry)
            total += 8.0 * fr.get_format(fmt).memory_bytes_for_shape(shape)
        return float(total)

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
            "aux_fixed_bits_total": fixed_total_bits,
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
            })

    # Output Pareto CSV
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")
    knee_summary = _pareto_knee_summary(curve)
    knee_path = Path(args.pareto_csv).with_suffix(".knees.json")
    knee_path.write_text(json.dumps(knee_summary, indent=2, sort_keys=True) + "\n")
    print(f"[alloc] Pareto knees → {knee_path}")

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
            payload = {
                "schema": "prismaquant.allocator.pareto_assignment.v1",
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
                "assignment": dict(sorted(assignment.items())),
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
            })
        (out_dir / "manifest.json").write_text(json.dumps({
            "schema": "prismaquant.allocator.pareto_manifest.v1",
            "probe": str(args.probe),
            "costs": str(args.costs),
            "propagated_sensitivity_costs": propagated_sensitivity_summary,
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

    # Emit chosen layer_config for target_bits.
    assignment, achieved, total, mutable_total = _solve_for_target(args.target_bits)
    if assignment is None:
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}. "
            "Consider raising the target or widening the format set.")
    print(
        f"[alloc] target_bits={args.target_bits}: "
        f"achieved_bits={achieved:.3f}, Δloss={total:.3e}",
        flush=True,
    )

    assignment_expanded = dict(assignment)

    # Expand fused-sibling super-Linears (qkv_proj / gate_up_proj).
    if not args.no_fused_aggregation:
        assignment_expanded = expand_fused_sibling_assignment(
            assignment_expanded, stats)

    assignment_expanded.update(fixed_format_assignment)

    # vLLM's FusedMoE requires all projections of the same expert to share
    # one scheme. This keeps per-Linear assignments serveable without
    # collapsing experts into allocator super-items.
    assignment_expanded = promote_serving_units(
        assignment_expanded,
        format_rank,
        profile=model_profile,
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

    if probe_model_path:
        visual_names_src = discover_visual_linears_from_source(probe_model_path)
    else:
        visual_names_src = []

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
                # Not in manifest — likely a visual Linear stamped via
                # --visual-format (bypasses the manifest by design) or
                # a name the profile rewrite didn't map. Skip.
                continue
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
                "investigate fused-sibling / MoE-unity promotion."
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

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    counts = defaultdict(int)
    for fmt in assignment_expanded.values():
        counts[fmt] += 1
    print(f"\n[alloc] target={args.target_bits} achieved={achieved:.3f}")
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
        achieved_bits=achieved,
        assignment_expanded=assignment_expanded,
        candidates=candidates,
        stats_entry_for=_stats_entry_for_assignment_name,
        format_specs=format_specs,
    )


if __name__ == "__main__":
    main()
