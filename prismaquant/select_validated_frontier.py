"""Select a measured frontier point from assignment-KL validation output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.layer_config import (
    LAYER_CONFIG_META_KEY,
    canonicalize_format,
    is_layer_config_meta_key,
    read_layer_config_metadata,
)
from prismaquant.saturation_select import find_saturation_bpp
from prismaquant.nvfp4_cb_footprint import (
    CB_ASSIGNMENT_IDENTITIES_FIELD,
    CB_TENSOR_IDENTITY_FIELD,
    assignment_serialization_sha256,
    cb_serialization_metadata_from_assignment_payload,
    cb_serialization_context_from_stamp,
    is_cb_format,
    whole_artifact_budget_from_assignment_payload,
)


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _saturation_pick(frontier: Sequence[Mapping], z: float) -> tuple[int, dict]:
    """Saturation B* over the measured frontier (the unconstrained selector).

    Builds the bpp grid + a (kl, kl_stderr) lookup from the measured lower
    envelope and runs ``find_saturation_bpp``: B* is the lowest bpp whose KL is
    within z * combined stderr of the highest-bpp asymptote. Returns the chosen
    frontier index and the full saturation result (trace/slopes/measured) for
    the summary. ``no_noise_floor`` is set when the frontier carries no positive
    per-bpp stderr (single-rep validation): the band is then 0, so B* collapses
    to the asymptote (ship the most bits) — a safe but uninformative degenerate
    that the caller must surface (run validation with --calib-repeats>=4).
    """
    grid = [float(r["bpp"]) for r in frontier]
    kl_by = {float(r["bpp"]): float(r["kl"]) for r in frontier}

    def _se(r):
        se = r.get("kl_stderr")
        try:
            se = float(se)
        except (TypeError, ValueError):
            return 0.0
        return se if math.isfinite(se) and se > 0.0 else 0.0

    se_by = {float(r["bpp"]): _se(r) for r in frontier}
    result = find_saturation_bpp(
        grid, lambda b: (kl_by[b], se_by[b]), z=z,
    )
    result["no_noise_floor"] = not any(v > 0.0 for v in se_by.values())
    bstar = result["bpp"]
    idx = min(range(len(frontier)), key=lambda i: abs(float(frontier[i]["bpp"]) - bstar))
    return idx, result


def _load_assignment(path: str | Path) -> dict[str, str]:
    payload = _load_json(path)
    return _assignment_from_payload(payload, where=str(path))


def _assignment_from_payload(
    payload: Mapping[str, object],
    *,
    where: str,
) -> dict[str, str]:
    raw = payload.get("assignment") if isinstance(payload, Mapping) else None
    if raw is None and isinstance(payload, Mapping):
        raw = payload
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected assignment JSON object")
    # Entries may be format-name strings ({qname: "NVFP4"}) or AutoRound-style
    # dicts ({qname: {"data_type": "nv_fp", "bits": 4, ...}}); str().upper() on
    # a dict silently fabricates a garbage format name. Strings go through the
    # registry canonicalizer (which keeps FP8_SOURCE & friends); dicts go
    # through the layer-config parser. Unknown names still fail loudly at
    # fr.get_format in _layer_config_from_assignment.
    return {
        str(name): (
            fr.canonical_format_name(fmt.strip().upper())
            if isinstance(fmt, str)
            else canonicalize_format(fmt)
        )
        for name, fmt in raw.items()
        if str(name).strip() and not is_layer_config_meta_key(name)
    }


def _layer_config_from_assignment(
    assignment: Mapping[str, str],
    *,
    cb_serialization_stamps: Mapping[str, object] | None = None,
) -> dict:
    out = {}
    for name, fmt in sorted(assignment.items()):
        out[str(name)] = fr.get_format(str(fmt).strip().upper()).autoround_config()
        if cb_serialization_stamps is not None and name in cb_serialization_stamps:
            out[str(name)][CB_TENSOR_IDENTITY_FIELD] = str(
                cb_serialization_stamps[name]
            )
    return out


def _log_error_values(values: Sequence[float]) -> list[float]:
    """Map measured KL values to log10 for the kneedle.

    Non-positive values are floored at the smallest positive measured value
    itself. A measured KL <= 0 (fp32 round-off on a near-passthrough
    assignment; realistic on FP8-native sources) is indistinguishable from
    "at the floor of what this validation run can resolve" — it is *not*
    evidence the point is orders of magnitude better than every real point.
    Flooring at min_positive places such points exactly 0 decades below the
    smallest real point; any lower floor (the old ``min_positive * 1e-6``)
    fabricates a multi-decade cliff in normalized log-space that compresses
    the real curve and flips the kneedle to the curve start, i.e. the worst
    point on the ship path.
    """
    finite_positive = [
        float(value) for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite_positive:
        return [0.0 for _ in values]
    floor = min(finite_positive)
    return [math.log10(max(float(value), floor)) for value in values]


def _kneedle_convex_decreasing(
    points: Sequence[Mapping[str, float]],
    *,
    log_error: bool = True,
) -> int:
    """Return knee index for points sorted by increasing bpp, decreasing KL."""
    if len(points) < 3:
        return min(
            range(len(points)),
            key=lambda i: (float(points[i]["kl"]), float(points[i]["bpp"])),
        )
    xs = [float(p["bpp"]) for p in points]
    ys = [float(p["kl"]) for p in points]
    if log_error:
        ys = _log_error_values(ys)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return min(range(len(points)), key=lambda i: (ys[i], xs[i]))
    x_norm = [(x - xmin) / (xmax - xmin) for x in xs]
    y_norm = [(y - ymin) / (ymax - ymin) for y in ys]
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    return min(range(len(diffs)), key=lambda i: diffs[i])


def kneedle_comparison(points: Sequence[Mapping[str, float]]) -> dict:
    if len(points) < 3:
        return {"enabled": False, "reason": "too_few_frontier_points"}

    def _record(mode: str, idx: int) -> dict:
        row = points[idx]
        return {
            "mode": mode,
            "label": row.get("label"),
            "bpp": float(row["bpp"]),
            "kl": float(row["kl"]),
            "index": int(idx),
        }

    log_idx = _kneedle_convex_decreasing(points, log_error=True)
    raw_idx = _kneedle_convex_decreasing(points, log_error=False)
    return {
        "enabled": True,
        "primary": "log_error",
        "log_error": _record("log_error", log_idx),
        "raw_linear": _record("raw_linear", raw_idx),
    }


TAIL_VETO_COLUMNS: tuple[str, ...] = ("kl_p95", "kl_p99", "kl_max", "nll_mean", "nll_p99")
TAIL_VETO_CHOICES: tuple[str, ...] = ("none", "kl_p99", "kl_max", "nll_p99")
#: Per-repeat values of each tail statistic, emitted by
#: `validate_assignments_kl._kl_repeat_summary` at zero extra forward cost.
TAIL_REPEAT_COLUMNS: tuple[str, ...] = tuple(f"{c}_repeats" for c in TAIL_VETO_COLUMNS)
#: The contract statistic (R9/D1, ruled 2026-07-30). `kl_max` — the worst
#: sequence — is the statistic that would have caught the broken 27B that passed
#: on the mean while 80% of its prompts were bad (§7.2's ship gate already
#: guards the p99 per-prompt NLL for the same reason).
DEFAULT_TAIL_VETO: str = "kl_max"
#: `--tail-eta auto`: derive the slack instead of picking one (house rule 2).
DEFAULT_TAIL_ETA: str = "auto"
#: Exit status when the selector refuses to certify a rate-axis pick (#117).
#: The recipe files are still written -- the byte-matched uniform control is
#: built FROM the candidate plan downstream -- but the pipeline must not walk
#: on to export on an uncertified selection.
RATE_AXIS_UNCERTIFIED_EXIT: int = 2


def rate_axis_rungs(assignment: Mapping[str, str]) -> list[str]:
    """Sorted unique rate-axis rung names in a selected assignment.

    The axis is defined by the code that owns the name grammar
    (``format_registry.is_tessera_format_name``), never by a list restated
    here: today the only rate-axis container is Tessera, and a future
    container adds itself by widening the definition this reads, in the commit
    that declares its lane.
    """
    return sorted({
        str(fmt)
        for fmt in assignment.values()
        if fr.is_tessera_format_name(fmt)
    })


def rate_axis_candidate_status(
    *,
    selected: Mapping,
    rungs: Sequence[str],
    n_rows: int,
) -> dict:
    """The uncertified-selection record stamped on a rate-axis pick (#117).

    Measured 2026-09-02: a Tessera allocation served 2.00x worse KL than a
    byte-matched uniform arm while every check this stage owns passed, and an
    oracle over the same menu reaches only 0.941x of uniform -- so no ranking
    this stage can do closes the gap. The validated frontier re-ranks the
    allocator's own Pareto rows and carries no uniform arm, so the pick ships
    as a candidate with the comparison, the bytes, and what would pass named
    in data, not just in a log line.
    """
    return {
        "status": "outstanding",
        "rate_axis_formats": list(rungs),
        "selected_label": selected.get("label"),
        "selected_measured_kl": selected.get("kl"),
        "selected_bpp": selected.get("bpp"),
        "selected_artifact_bytes": selected.get("artifact_bytes"),
        "compared": (
            f"re-ranked {n_rows} allocator Pareto rows on measured KL; "
            "no byte-matched uniform arm in the validation set"
        ),
        "to_pass": (
            "build the byte-matched uniform control from the candidate plan, "
            "serve it beside the candidate, and close the shipcard "
            "uniform_control slot (shipcard_cli fill-control); verify and "
            "publish refuse until then"
        ),
    }


def tail_eta_auto(row: Mapping, column: str) -> tuple[float, str]:
    """Derived tail-veto slack for one incumbent row: `stderr(tail)/mean(tail)`.

    **The derivation.** The veto asks whether a candidate's tail is *really*
    worse than the incumbent's, and the only scale on which "really" means
    anything is the noise of the statistic itself. `--calib-repeats` already
    re-measures every row on independent calibration draws, so each tail
    statistic arrives as a small sample (`<column>_repeats`); its relative
    standard error — `std/sqrt(n)` over `mean` — is exactly "how much this tail
    moves when nothing about the assignment changes". A candidate inside that
    band is not distinguishable from the incumbent and is admitted; one outside
    it is a real tail regression and is refused. Floored at 0 (a negative or
    non-finite spread means no slack, not negative slack).

    This is deliberately *not* a constant: §5's single-seed history (+10% that
    flipped to −5.2% across repeats; between-seed std ~0.02) is the reason a
    hand-set eta would be either vacuous or arbitrary at different scales.

    Returns `(eta, source)` where `source` is one of `derived`,
    `single_repeat` (n<2 — no spread exists, so the slack degrades to a strict
    0 and the caller must say so out loud), `absent` (pre-R9 row that carries no
    per-repeat tails) or `degenerate` (non-positive/non-finite mean).
    """
    repeats = row.get(f"{column}_repeats")
    if repeats is None:
        return 0.0, "absent"
    try:
        vals = [float(v) for v in repeats]
    except (TypeError, ValueError):
        return 0.0, "absent"
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0, "single_repeat"
    mean = sum(vals) / len(vals)
    if not math.isfinite(mean) or mean <= 0.0:
        return 0.0, "degenerate"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    stderr = math.sqrt(max(var, 0.0)) / math.sqrt(len(vals))
    eta = stderr / mean
    if not math.isfinite(eta):
        return 0.0, "degenerate"
    return max(eta, 0.0), "derived"


def _tail_value(row: Mapping, column: str) -> float | None:
    value = row.get(column)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def tail_veto_inert_reason(rows: Sequence[Mapping], column: str | None) -> str | None:
    """Why the veto cannot bind on these rows, or None when it can.

    With the veto DEFAULT-ON, a validation JSON written before R9 carries no
    tail column at all: vetoing every row would empty the frontier and turn a
    stale input into a crash. The veto goes inert instead — loudly, never
    silently — and the run behaves exactly as it did before R9.
    """
    if column is None:
        return None
    if not any(_tail_value(row, column) is not None for row in rows):
        return "tail_column_absent_on_every_row"
    return None


def _row_metric(row: Mapping, metric: str) -> float | None:
    candidates: tuple[str, ...]
    # ``kl_mean`` is the canonical mean key (R28); ``last_token_kl`` is the
    # deprecated alias kept one cycle, and stays first so rows written by both
    # the old and new writer resolve identically.
    if metric == "ucb":
        candidates = (
            "kl_ucb", "validation_kl_ucb", "last_token_kl_ucb",
            "last_token_kl", "kl_mean", "kl",
        )
    else:
        candidates = ("last_token_kl", "kl_mean", "validation_kl", "kl")
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _artifact_bytes_for_row(row: Mapping) -> int | None:
    """Conservative whole-artifact selection bound, or None when unpriced.

    A raw safetensors tensor-span estimate is deliberately not accepted as a
    directory budget. Modern allocator payloads expose
    ``whole_artifact_upper_bound_bytes`` (tensor spans + operator reserve) and
    keep ``artifact_bytes`` only as a scope-stamped compatibility alias.
    """
    direct = row.get("whole_artifact_upper_bound_bytes")
    if direct is None and str(row.get("artifact_byte_scope", "")).startswith(
        "selection_upper_bound_"
    ):
        direct = row.get("artifact_bytes")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            return None
    path = row.get("path")
    if not path:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    value = None
    if isinstance(payload, Mapping):
        value = payload.get("whole_artifact_upper_bound_bytes")
        if value is None and str(payload.get("artifact_byte_scope", "")).startswith(
            "selection_upper_bound_"
        ):
            value = payload.get("artifact_bytes")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def measured_rows(
    results: Sequence[Mapping],
    *,
    metric: str = "kl",
) -> list[dict]:
    """Return finite measured KL/bpp rows sorted by bpp."""
    rows: list[dict] = []
    for row in results:
        kl = _row_metric(row, metric)
        bpp = row.get("bpp")
        path = row.get("path")
        label = row.get("label")
        if kl is None or bpp is None or path is None:
            continue
        kl_f = float(kl)
        bpp_f = float(bpp)
        if not (math.isfinite(kl_f) and math.isfinite(bpp_f)):
            continue
        rows.append({
            "label": str(label or Path(str(path)).stem),
            "path": str(path),
            "kl": kl_f,
            "bpp": bpp_f,
            "format_counts": dict(row.get("format_counts", {}) or {}),
            "changed_vs_base": int(row.get("changed_vs_base", 0) or 0),
            "assignment_hash": row.get("assignment_hash"),
            "assignment_sha256": row.get("assignment_sha256"),
            "mse": dict(row.get("mse", {}) or {}),
            # The surrogate the allocator optimized. validate_assignments_kl emits
            # it nested as mse.predicted_dloss_sum; a top-level surrogate_loss (e.g.
            # test fixtures or legacy rows) takes precedence when present. Without
            # this fallback the surrogate-vs-KL Spearman below silently never fires.
            "surrogate_loss": (
                row.get("surrogate_loss")
                if row.get("surrogate_loss") is not None
                else (row.get("mse") or {}).get("predicted_dloss_sum")
            ),
            # Conservative whole-artifact selection bound. Final exact
            # recursive bytes are enforced by the exporter.
            "artifact_bytes": _artifact_bytes_for_row(row),
            **({
                "resolved_assignment_payload": dict(
                    row["resolved_assignment_payload"]
                ),
                "resolved_assignment_payload_sha256": row.get(
                    "resolved_assignment_payload_sha256"
                ),
            } if isinstance(
                row.get("resolved_assignment_payload"), Mapping
            ) else {}),
            "kl_repeats": list(row.get("kl_repeats", []) or []),
            "kl_std": row.get("kl_std"),
            "kl_stderr": row.get("kl_stderr"),
            "kl_ucb": row.get("kl_ucb", row.get("validation_kl_ucb")),
            # R9 tail columns, passed through when validate_assignments_kl
            # emitted them. Absent on pre-R9 rows -> the veto reports
            # 'tail_missing' rather than silently admitting.
            **{
                column: row[column]
                for column in TAIL_VETO_COLUMNS + TAIL_REPEAT_COLUMNS
                if row.get(column) is not None
            },
        })
    rows.sort(key=lambda r: (r["bpp"], r["kl"], r["label"]))
    return rows


def _frontier_from_rows(
    rows: Sequence[Mapping],
    *,
    kl_noise_floor: float = 0.0,
    tail_veto: str | None = None,
    tail_eta: float | str = 0.0,
    vetoed: list | None = None,
) -> list[dict]:
    """Return the eta-dominance lower envelope of measured rows.

    ``rows`` must already be sorted by (bpp, kl); a point enters the envelope
    only when it improves the running best KL by more than the noise floor.

    **Tail veto (D1/R9).** CLAUDE.md §5 rule 4: KL is a *screening* metric, and
    a lower mean can hide a heavier tail — the shipped 27B PrismaSCOUT has a
    worse max-prompt NLL than the artifact it beat on mean KL. With
    ``tail_veto`` naming a column (``kl_p99``/``kl_max``/``nll_p99``), a row
    that improves mean KL is admitted only if its tail also holds:
    ``row[tail] <= incumbent[tail] * (1 + tail_eta)``. The incumbent is the
    last admitted frontier point, so the tail is required to be non-increasing
    along the envelope exactly as the mean is.

    ``tail_eta`` is a number, or ``"auto"`` — the derived slack, recomputed
    against each incumbent as that row's between-repeat relative stderr (see
    ``tail_eta_auto``). ``"auto"`` on a row with a single calibration repeat is
    a strict 0; the CLI says so out loud.

    ``tail_veto=None`` (and ``--tail-veto none``) is byte-identical to the
    pre-R9 behavior — no column is read and no row is vetoed. The CLI default
    is ``kl_max`` (ruled 2026-07-30); vetoed rows are appended to ``vetoed``
    with a ``veto_reason`` so a rejection is visible in the summary rather than
    silent.
    """
    frontier: list[dict] = []
    best_kl = float("inf")
    floor = max(float(kl_noise_floor), 0.0)
    column = str(tail_veto) if tail_veto and tail_veto != "none" else None
    # Pre-R9 input: nothing to veto on. Go inert rather than veto every row and
    # hand back an empty frontier (the caller reports the reason).
    if tail_veto_inert_reason(rows, column) is not None:
        column = None
    auto_eta = isinstance(tail_eta, str) and tail_eta.strip().lower() == "auto"
    eta = 0.0 if auto_eta else float(tail_eta)
    incumbent_tail: float | None = None
    incumbent_row: Mapping | None = None
    for row in rows:
        if not (row["kl"] < best_kl - floor - 1e-12):
            continue
        if column is not None:
            value = _tail_value(row, column)
            if value is None:
                if vetoed is not None:
                    vetoed.append({
                        **dict(row),
                        "veto_reason": "tail_missing",
                        "veto_column": column,
                    })
                continue
            eta_source = "explicit"
            if auto_eta and incumbent_row is not None:
                eta, eta_source = tail_eta_auto(incumbent_row, column)
            if (
                incumbent_tail is not None
                and value > incumbent_tail * (1.0 + eta) + 1e-12
            ):
                if vetoed is not None:
                    vetoed.append({
                        **dict(row),
                        "veto_reason": "tail_regression",
                        "veto_column": column,
                        "veto_value": value,
                        "veto_incumbent": incumbent_tail,
                        "veto_limit": incumbent_tail * (1.0 + eta),
                        "veto_eta": float(eta),
                        "veto_eta_source": eta_source,
                    })
                continue
            incumbent_tail = value
            incumbent_row = row
        frontier.append(row)
        best_kl = row["kl"]
    return frontier


def measured_frontier(
    results: Sequence[Mapping],
    *,
    metric: str = "kl",
    kl_noise_floor: float = 0.0,
    tail_veto: str | None = None,
    tail_eta: float | str = 0.0,
    vetoed: list | None = None,
) -> list[dict]:
    """Return non-dominated measured KL/bpp points sorted by bpp.

    A point is dominated when a lower-or-equal bpp assignment already has
    lower-or-equal KL. Kneedle should operate on this measured lower envelope,
    not on noisy interior points. See ``_frontier_from_rows`` for ``tail_veto``.
    """
    return _frontier_from_rows(
        measured_rows(results, metric=metric),
        kl_noise_floor=kl_noise_floor,
        tail_veto=tail_veto,
        tail_eta=tail_eta,
        vetoed=vetoed,
    )


def practical_knee(
    frontier: Sequence[Mapping],
    *,
    rel_eps: float = 0.005,
    abs_eps: float = 0.0,
    kl_noise_floor: float = 0.0,
) -> dict | None:
    if not frontier:
        return None
    best = min(frontier, key=lambda row: (float(row["kl"]), float(row["bpp"])))
    tol = max(
        float(abs_eps),
        float(kl_noise_floor),
        abs(float(best["kl"])) * max(float(rel_eps), 0.0),
    )
    eligible = [
        row for row in frontier
        if float(row["kl"]) <= float(best["kl"]) + tol + 1e-12
    ]
    chosen = min(eligible, key=lambda row: (float(row["bpp"]), float(row["kl"])))
    out = dict(chosen)
    out["best_kl_label"] = best["label"]
    out["best_kl"] = float(best["kl"])
    out["tolerance"] = float(tol)
    return out


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0 for _ in ordered]
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][0] == ordered[idx][0]:
            end += 1
        rank = (idx + end - 1) / 2.0
        for _value, original_idx in ordered[idx:end]:
            ranks[original_idx] = rank
        idx = end
    return ranks


def spearman_rank_correlation(rows: Sequence[Mapping]) -> float | None:
    paired = [
        (float(row["surrogate_loss"]), float(row["kl"]))
        for row in rows
        if row.get("surrogate_loss") is not None
        and math.isfinite(float(row["surrogate_loss"]))
        and math.isfinite(float(row["kl"]))
    ]
    if len(paired) < 3:
        return None
    xs = _rank([item[0] for item in paired])
    ys = _rank([item[1] for item in paired])
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(num / (den_x * den_y))


def worst_rank_inversion(rows: Sequence[Mapping]) -> dict | None:
    """Surface the single most-misranked pair of measured rows.

    Uses the same (surrogate_loss, kl) pairing as ``spearman_rank_correlation``
    so the two agree on which rows count. Returns the pair whose surrogate-rank
    ordering disagrees most strongly with the measured-KL-rank ordering, i.e.
    the pair maximizing ``rank_kl_gap`` among discordant pairs. This checks
    surrogate-vs-KL fidelity only; it says nothing about held-out PPL, which is
    measured post-export and is not joined here. Returns ``None`` when fewer than
    three usable pairs exist (same guard as the Spearman).
    """
    paired = [
        {
            "label": str(row.get("label") or row.get("path") or f"point[{idx}]"),
            "surrogate_loss": float(row["surrogate_loss"]),
            "kl": float(row["kl"]),
        }
        for idx, row in enumerate(rows)
        if row.get("surrogate_loss") is not None
        and math.isfinite(float(row["surrogate_loss"]))
        and math.isfinite(float(row["kl"]))
    ]
    if len(paired) < 3:
        return None
    sur_ranks = _rank([p["surrogate_loss"] for p in paired])
    kl_ranks = _rank([p["kl"] for p in paired])

    worst = None
    worst_gap = 0.0
    for i in range(len(paired)):
        for j in range(i + 1, len(paired)):
            # Discordant: surrogate orders i,j one way, KL the other way.
            sur_order = sur_ranks[i] - sur_ranks[j]
            kl_order = kl_ranks[i] - kl_ranks[j]
            if sur_order == 0.0 or kl_order == 0.0:
                continue
            if (sur_order > 0) == (kl_order > 0):
                continue  # concordant, no inversion
            gap = abs(kl_ranks[i] - kl_ranks[j])
            if gap > worst_gap:
                worst_gap = gap
                worst = (i, j)
    if worst is None:
        return None

    i, j = worst
    # Order the reported pair so "better" = lower surrogate_loss (predicted best).
    a, b = (paired[i], paired[j]) if paired[i]["surrogate_loss"] <= paired[j]["surrogate_loss"] else (paired[j], paired[i])
    direction = "worse" if a["kl"] > b["kl"] else "better"
    verdict = (
        f"surrogate ranked '{a['label']}' better than '{b['label']}' "
        f"(predicted_dloss {a['surrogate_loss']:.6g} < {b['surrogate_loss']:.6g}) "
        f"but measured KL was {direction} ({a['kl']:.6g} vs {b['kl']:.6g})"
    )
    return {
        "predicted_best_label": a["label"],
        "predicted_best_surrogate_loss": a["surrogate_loss"],
        "predicted_best_kl": a["kl"],
        "predicted_worse_label": b["label"],
        "predicted_worse_surrogate_loss": b["surrogate_loss"],
        "predicted_worse_kl": b["kl"],
        "rank_gap": float(worst_gap),
        "verdict": verdict,
    }


def _row_identity(row: Mapping) -> tuple:
    return (
        str(row.get("label")),
        str(row.get("path")),
        float(row["bpp"]),
        float(row["kl"]),
    )


def _row_kl_stderr(row: Mapping) -> float | None:
    stderr = row.get("kl_stderr")
    try:
        stderr = float(stderr)
    except (TypeError, ValueError):
        return None
    return stderr if math.isfinite(stderr) and stderr > 0.0 else None


def leave_one_out_kneedle_diagnostic(
    frontier: Sequence[Mapping],
    selected: Mapping,
    *,
    tolerance_bpp: float = 0.1,
    kl_noise_floor: float = 0.0,
    all_rows: Sequence[Mapping] | None = None,
    tail_veto: str | None = None,
    tail_eta: float | str = 0.0,
) -> dict:
    """Leave-one-out stability of the kneedle pick.

    For each frontier point, drop it from the *full* measured row set
    (``all_rows``, when provided), rebuild the eta-dominance envelope, and
    re-run the kneedle on that rebuilt envelope. Dropping a frontier point can
    let a previously-dominated interior point re-enter the envelope; freezing
    the envelope (the old behavior, still the fallback when ``all_rows`` is
    omitted) understates the instability.

    KL-axis stability tolerance (no arbitrary constants):
    - an explicit positive ``kl_noise_floor`` wins (source "kl_noise_floor");
    - otherwise the knee point's measured repeat stderr — ``kl_stderr`` from
      validate_assignments_kl's ``_kl_repeat_summary`` — is the measured noise
      scale of the pick: an LOO KL shift within one stderr is
      indistinguishable from measurement noise (source "repeat_stderr");
    - with neither, the tolerance is strict 0: single-rep validation carries
      no measured noise scale, so any shift counts as unstable
      (source "strict").
    ``stability_tolerance_source`` in the output labels which one applied.
    """
    if len(frontier) < 4:
        return {"enabled": False, "reason": "too_few_frontier_points"}
    rows = list(all_rows) if all_rows else [dict(row) for row in frontier]
    rows.sort(key=lambda r: (float(r["bpp"]), float(r["kl"]), str(r.get("label"))))
    selected_bpp = float(selected["bpp"])
    selected_kl = float(selected["kl"])
    picks: list[dict] = []
    for dropped in frontier:
        dropped_key = _row_identity(dropped)
        subset_rows = [row for row in rows if _row_identity(row) != dropped_key]
        subset = _frontier_from_rows(
            subset_rows,
            kl_noise_floor=kl_noise_floor,
            tail_veto=tail_veto,
            tail_eta=tail_eta,
        )
        if len(subset) < 3:
            continue
        chosen = subset[_kneedle_convex_decreasing(subset)]
        picks.append({
            "dropped_label": dropped["label"],
            "selected_label": chosen["label"],
            "bpp": float(chosen["bpp"]),
            "kl": float(chosen["kl"]),
        })
    if not picks:
        return {"enabled": False, "reason": "no_leave_one_out_picks"}
    max_bpp_shift = max(abs(row["bpp"] - selected_bpp) for row in picks)
    max_kl_shift = max(abs(row["kl"] - selected_kl) for row in picks)
    if float(kl_noise_floor) > 0.0:
        kl_tolerance = float(kl_noise_floor)
        tolerance_source = "kl_noise_floor"
    else:
        stderr = _row_kl_stderr(selected)
        if stderr is not None:
            kl_tolerance = stderr
            tolerance_source = "repeat_stderr"
        else:
            kl_tolerance = 0.0
            tolerance_source = "strict"
    stable = (
        max_bpp_shift <= max(float(tolerance_bpp), 0.0) + 1e-12
        and max_kl_shift <= kl_tolerance + 1e-12
    )
    return {
        "enabled": True,
        "stable": bool(stable),
        "max_bpp_shift": float(max_bpp_shift),
        "max_kl_shift": float(max_kl_shift),
        "tolerance_bpp": float(tolerance_bpp),
        "kl_noise_floor": float(kl_noise_floor),
        "kl_stability_tolerance": float(kl_tolerance),
        "stability_tolerance_source": tolerance_source,
        "picks": picks,
    }


def select_frontier_point(
    results: Sequence[Mapping],
    *,
    mode: str = "kneedle",
    metric: str = "kl",
    kl_noise_floor: float = 0.0,
    practical_rel_eps: float = 0.005,
    practical_abs_eps: float = 0.0,
    knee_tolerance_bpp: float = 0.1,
    unstable_policy: str = "keep-kneedle",
    sat_z: float = 2.0,
    tail_veto: str | None = None,
    tail_eta: float | str = 0.0,
    vetoed: list | None = None,
    budget_bytes: float | None = None,
) -> tuple[dict, list[dict]]:
    rows = measured_rows(results, metric=metric)
    frontier = _frontier_from_rows(
        rows,
        kl_noise_floor=kl_noise_floor,
        tail_veto=tail_veto,
        tail_eta=tail_eta,
        vetoed=vetoed,
    )
    if not frontier:
        raise ValueError("no finite measured KL/bpp points found")
    if mode == "budget":
        # Byte budget = constraint, measured KL = objective (re-vet R1). The
        # frontier is the KL lower envelope and bytes are monotone in bpp, so
        # the min-KL fitting FRONTIER row is the min-KL fitting row overall.
        if budget_bytes is None:
            raise ValueError("--mode budget requires a byte budget "
                             "(--target-disk-gb)")
        unpriced = [row["label"] for row in frontier
                    if row.get("artifact_bytes") is None]
        if unpriced:
            raise ValueError(
                "--mode budget needs a tensor-payload + non-tensor-reserve "
                "whole-artifact upper bound on every frontier "
                f"row; {len(unpriced)} are unpriced (e.g. {unpriced[:3]}). "
                "Re-run the allocator with --target-disk-gb and an explicit "
                "--artifact-overhead-reserve-bytes so it stamps the bound into "
                "Pareto assignment payloads.")
        fitting = [i for i, row in enumerate(frontier)
                   if float(row["artifact_bytes"]) <= float(budget_bytes)]
        if not fitting:
            cheapest = min(float(row["artifact_bytes"]) for row in frontier)
            raise ValueError(
                f"no measured allocation fits the {budget_bytes / 1e9:.3f}GB "
                f"budget; the cheapest measured artifact is "
                f"{cheapest / 1e9:.3f}GB. Raise the budget or widen the "
                "format menu.")
        idx = min(fitting, key=lambda i: (frontier[i]["kl"], -frontier[i]["bpp"]))
    elif mode == "best-kl":
        idx = min(range(len(frontier)), key=lambda i: (frontier[i]["kl"], frontier[i]["bpp"]))
    elif mode == "saturation":
        idx, _sat = _saturation_pick(frontier, sat_z)
    elif mode == "lowest-bpp":
        idx = 0
    elif mode == "practical-knee":
        practical = practical_knee(
            frontier,
            rel_eps=practical_rel_eps,
            abs_eps=practical_abs_eps,
            kl_noise_floor=kl_noise_floor,
        )
        idx = next(
            i for i, row in enumerate(frontier)
            if practical is not None and row["label"] == practical["label"]
        )
    elif mode == "kneedle":
        idx = _kneedle_convex_decreasing(frontier)
        diagnostic = leave_one_out_kneedle_diagnostic(
            frontier,
            frontier[idx],
            tolerance_bpp=knee_tolerance_bpp,
            kl_noise_floor=kl_noise_floor,
            all_rows=rows,
            tail_veto=tail_veto,
            tail_eta=tail_eta,
        )
        if diagnostic.get("enabled") and not diagnostic.get("stable", True):
            if unstable_policy == "best-kl":
                idx = min(range(len(frontier)), key=lambda i: (frontier[i]["kl"], frontier[i]["bpp"]))
            elif unstable_policy == "practical-knee":
                practical = practical_knee(
                    frontier,
                    rel_eps=practical_rel_eps,
                    abs_eps=practical_abs_eps,
                    kl_noise_floor=kl_noise_floor,
                )
                idx = next(
                    i for i, row in enumerate(frontier)
                    if practical is not None and row["label"] == practical["label"]
                )
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    return frontier[idx], frontier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a measured-kneedle assignment from validate_assignments_kl output",
    )
    parser.add_argument("--validation-json", required=True)
    parser.add_argument(
        "--mode",
        choices=("kneedle", "best-kl", "lowest-bpp", "practical-knee",
                 "saturation", "budget"),
        default="kneedle",
        help="Frontier pick. 'budget' = min measured KL among the rows whose "
             "tensor payload plus operator-supplied non-tensor reserve fits "
             "--target-disk-gb; export then enforces exact recursive file "
             "bytes (re-vet R1: the card is the constraint, measured KL is "
             "the objective). "
             "'saturation' = unconstrained bit-rate selector: "
             "lowest bpp whose KL is within --sat-z stderr of the high-bpp "
             "asymptote (needs a real per-bpp stderr, i.e. validate with "
             "--calib-repeats>=4). 'kneedle' is axis-dependent and a diagnostic "
             "on a log-linear RD curve.",
    )
    parser.add_argument(
        "--metric",
        choices=("kl", "ucb"),
        default="kl",
        help="Metric used for frontier construction. 'ucb' uses kl_ucb when present.",
    )
    parser.add_argument("--kl-noise-floor", type=float, default=0.0)
    parser.add_argument(
        "--tail-veto",
        choices=TAIL_VETO_CHOICES,
        default=DEFAULT_TAIL_VETO,
        help="D1 tail veto: additionally require the named tail column to be "
             "non-increasing along the frontier, so a mean-KL win that "
             "regresses the tail is not admitted (CLAUDE.md §5 rule 4). "
             "Columns come from validate_assignments_kl's per-sequence "
             "emission and share the gold lane's key names. DEFAULT "
             f"'{DEFAULT_TAIL_VETO}' (ruled 2026-07-30): the worst sequence is "
             "the statistic that would have caught the broken 27B that passed "
             "on the mean while 80%% of its prompts were bad. The asymmetry is "
             "the reason it is safe on by default — a spurious veto only makes "
             "the pick MORE conservative (a higher-bpp, lower-tail point), and "
             "it is never silent: every refusal is retained in the summary's "
             "vetoed_rows with its veto_reason. 'none' restores the pre-R9 "
             "envelope byte-for-byte.",
    )
    parser.add_argument(
        "--tail-eta", type=str, default=DEFAULT_TAIL_ETA,
        help="Slack on the tail veto: a row is admitted when "
             "row[tail] <= incumbent[tail] * (1 + tail_eta). DEFAULT 'auto' "
             "DERIVES it (house rule 2) as the incumbent's between-repeat "
             "relative stderr of the tail statistic, i.e. how much that tail "
             "moves when nothing about the assignment changes; a single "
             "calibration repeat has no spread, so auto degrades to a strict "
             "0.0 and says so. An explicit number wins.",
    )
    parser.add_argument("--sat-z", type=float, default=2.0,
                        help="Significance multiplier on the combined per-bpp "
                             "stderr for --mode saturation (2.0 ~= 95%).")
    parser.add_argument("--practical-rel-eps", type=float, default=0.005)
    parser.add_argument("--practical-abs-eps", type=float, default=0.0)
    parser.add_argument("--knee-tolerance-bpp", type=float, default=0.1)
    parser.add_argument(
        "--unstable-policy",
        choices=("keep-kneedle", "best-kl", "practical-knee"),
        default="keep-kneedle",
    )
    parser.add_argument(
        "--target-disk-gb", type=float, default=None,
        help="Byte budget in decimal GB. Required by --mode budget; ignored "
             "by the other picks (recorded in the summary either way).")
    parser.add_argument("--output-layer-config", required=True)
    parser.add_argument("--output-assignment", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(argv)
    requested_budget_bytes = None
    if args.mode == "budget":
        if (
            args.target_disk_gb is None
            or not math.isfinite(args.target_disk_gb)
            or args.target_disk_gb <= 0
        ):
            parser.error(
                "--mode budget requires a positive finite --target-disk-gb"
            )
        requested_budget_bytes = int(math.floor(args.target_disk_gb * 1e9))
    # This stage is pure JSON/frontier post-processing. GPU-or-bust applies to
    # tensor hot paths (probe, render, export, validation), not this selector.
    payload = _load_json(args.validation_json)
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list):
        raise ValueError("--validation-json must contain a results list")

    tail_veto = None if args.tail_veto == "none" else args.tail_veto
    tail_eta_raw = str(args.tail_eta).strip()
    tail_eta_auto_mode = tail_eta_raw.lower() == "auto"
    if tail_eta_auto_mode:
        tail_eta_arg: float | str = "auto"
    else:
        try:
            tail_eta_arg = float(tail_eta_raw)
        except ValueError:
            parser.error(
                f"--tail-eta must be a number or 'auto', got {args.tail_eta!r}")
    vetoed_rows: list[dict] = []
    selected, frontier = select_frontier_point(
        results,
        mode=args.mode,
        metric=args.metric,
        kl_noise_floor=args.kl_noise_floor,
        practical_rel_eps=args.practical_rel_eps,
        practical_abs_eps=args.practical_abs_eps,
        knee_tolerance_bpp=args.knee_tolerance_bpp,
        unstable_policy=args.unstable_policy,
        sat_z=args.sat_z,
        tail_veto=tail_veto,
        tail_eta=tail_eta_arg,
        vetoed=vetoed_rows,
        budget_bytes=requested_budget_bytes,
    )
    saturation = None
    if args.mode == "saturation":
        if args.metric == "ucb":
            # The band is z * combined stderr; with metric=ucb the frontier 'kl'
            # is already mean+k*stderr, so the band would double-count the noise.
            # Saturation wants the raw mean — warn rather than silently inflate.
            print("[frontier-select] WARNING: --mode saturation with --metric "
                  "ucb double-counts uncertainty (UCB already folds stderr into "
                  "kl, and the saturation band re-adds it); use --metric kl.",
                  flush=True)
        _sidx, saturation = _saturation_pick(frontier, args.sat_z)
    practical = practical_knee(
        frontier,
        rel_eps=args.practical_rel_eps,
        abs_eps=args.practical_abs_eps,
        kl_noise_floor=args.kl_noise_floor,
    )
    knee_cmp = kneedle_comparison(frontier)
    diagnostic_rows = measured_rows(results, metric=args.metric)
    loo = (
        leave_one_out_kneedle_diagnostic(
            frontier,
            selected,
            tolerance_bpp=args.knee_tolerance_bpp,
            kl_noise_floor=args.kl_noise_floor,
            all_rows=diagnostic_rows,
            tail_veto=tail_veto,
            tail_eta=tail_eta_arg,
        )
        if args.mode == "kneedle"
        else {"enabled": False, "reason": "mode_not_kneedle"}
    )
    rank_corr = spearman_rank_correlation(diagnostic_rows)
    worst_inversion = worst_rank_inversion(diagnostic_rows)
    # What the veto actually did, recorded rather than inferred: whether it
    # could bind at all on this input, and — under 'auto' — the slack each
    # admitted incumbent contributed.
    tail_inert_reason = tail_veto_inert_reason(diagnostic_rows, tail_veto)
    eta_resolved: list[dict] | None = None
    if tail_veto is not None and tail_inert_reason is None and tail_eta_auto_mode:
        eta_resolved = []
        for row in frontier:
            eta_value, eta_source = tail_eta_auto(row, tail_veto)
            eta_resolved.append({
                "label": row.get("label"),
                "eta": float(eta_value),
                "source": eta_source,
            })
    resolved_payload = selected.get("resolved_assignment_payload")
    if isinstance(resolved_payload, Mapping):
        selected_payload = dict(resolved_payload)
        resolved_payload_sha256 = hashlib.sha256(
            json.dumps(
                selected_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if selected.get("resolved_assignment_payload_sha256") != (
            resolved_payload_sha256
        ):
            raise ValueError(
                "resolved measured assignment payload does not match the "
                "identity bound to its KL result"
            )
        assignment = _assignment_from_payload(
            selected_payload,
            where=f"resolved measured assignment {selected.get('label')!r}",
        )
        actual_assignment_sha256 = assignment_serialization_sha256(assignment)
        row_assignment_sha256 = selected.get("assignment_sha256")
        payload_assignment_sha256 = selected_payload.get("assignment_sha256")
        if (
            row_assignment_sha256 != actual_assignment_sha256
            or payload_assignment_sha256 != actual_assignment_sha256
        ):
            raise ValueError(
                "resolved measured assignment does not match the exact "
                "assignment identity bound to its KL result"
            )
    else:
        # Backwards-compatible path for validation files produced before the
        # validator persisted its exact base+candidate merge. New production
        # validation always takes the resolved branch above.
        selected_payload = _load_json(selected["path"])
        assignment = _load_assignment(selected["path"])
    # A rate-axis pick (#117) is a candidate until a byte-matched uniform arm
    # corroborates it. The validated frontier re-ranks the allocator's own
    # Pareto rows and carries no uniform arm, so no input to this stage can
    # corroborate -- the absence is structural, and the served comparison that
    # would close it is one this stage cannot run. Recorded in data below and
    # refused at the end: the recipe files the control loop needs are still
    # written, but the pipeline must not walk on to export certified.
    rate_axis = rate_axis_rungs(assignment)
    uniform_control_status = (
        rate_axis_candidate_status(
            selected=selected, rungs=rate_axis, n_rows=len(results),
        )
        if rate_axis else None
    )
    selected_cb_context, selected_cb_stamps = (
        cb_serialization_metadata_from_assignment_payload(selected_payload)
        if isinstance(selected_payload, Mapping)
        else (None, {})
    )
    selected_cb_names = {
        str(name) for name, fmt in assignment.items() if is_cb_format(fmt)
    }
    if selected_cb_names and selected_cb_context is None:
        raise ValueError(
            "selected CB assignment is missing its global serialized-payload "
            "context"
        )
    if selected_cb_context is not None and not selected_cb_stamps:
        raise ValueError(
            "selected CB assignment carries a global serialized-payload "
            "context but no per-layer identities; refusing to carry a stale "
            "global stamp onto unverifiable tensors"
        )
    if selected_cb_names and not selected_cb_stamps:
        raise ValueError(
            "selected CB assignment is missing its per-layer serialization "
            "identities"
        )
    if selected_cb_stamps and selected_cb_context is None:
        raise ValueError(
            "selected CB assignment carries per-layer serialization identities "
            "without their global context"
        )
    if selected_cb_stamps:
        stamped_names = set(selected_cb_stamps)
        missing = sorted(selected_cb_names - stamped_names)
        extra = sorted(stamped_names - selected_cb_names)
        if missing or extra:
            raise ValueError(
                "selected CB assignment serialization identities do not match "
                f"its CB tensors: missing={missing[:8]}, extra={extra[:8]}"
            )
    selected_cb_render_identity = selected_payload.get("cb_render_identity")
    if selected_cb_names:
        from prismaquant.production_weight_cache import (
            validate_cb_render_provenance,
        )

        selected_context_object = cb_serialization_context_from_stamp(
            selected_cb_context,
            where="selected frontier CB context",
        )
        _render_context, selected_cb_render_identity = (
            validate_cb_render_provenance(
                selected_payload,
                expected_context=selected_context_object,
                expected_formats_by_qname={
                    name: (assignment[name],)
                    for name in sorted(selected_cb_names)
                },
                where="selected frontier CB render identity",
            )
        )
    elif selected_cb_render_identity is not None:
        raise ValueError(
            "selected non-CB assignment carries a stale CB render identity"
        )
    selected_budget = whole_artifact_budget_from_assignment_payload(
        selected_payload,
        where=f"selected frontier assignment {selected['path']}",
        assignment=assignment,
    )
    if args.mode == "budget":
        if selected_budget is None:
            raise ValueError(
                "selected budget-mode assignment has no whole_artifact_budget "
                "stamp; refusing to emit an assignment without the exporter's "
                "hard recursive-byte gate"
            )
        stamped_budget = int(selected_budget["budget_bytes"])
        if stamped_budget != requested_budget_bytes:
            raise ValueError(
                "selected assignment whole-artifact budget differs from "
                f"--target-disk-gb: stamp={stamped_budget}B, "
                f"requested={requested_budget_bytes}B"
            )
        selected_upper = selected.get("artifact_bytes")
        stamped_upper = int(
            selected_budget["selection_whole_artifact_upper_bound_bytes"]
        )
        if selected_upper is None or int(selected_upper) != stamped_upper:
            raise ValueError(
                "selected row whole-artifact upper bound does not reconcile "
                f"with its assignment stamp: row={selected_upper!r}, "
                f"stamp={stamped_upper}B"
            )
    selected_cb_stamps_arg = selected_cb_stamps or None
    layer_config = _layer_config_from_assignment(
        assignment,
        cb_serialization_stamps=selected_cb_stamps_arg,
    )

    layer_config_path = Path(args.output_layer_config)
    # This stage OVERWRITES the allocator's layer_config.json, so it must carry
    # the allocator's reserved metadata forward — the exporter reads the
    # resolved serving profile from there (re-vet R11), and dropping it would
    # re-open the allocator/export profile split this run just closed.
    carried = dict(read_layer_config_metadata(layer_config_path))
    # The selected payload, not the overwritten destination file, owns
    # assignment-coupled identities.  Otherwise selecting a non-CB point after
    # a CB allocator run carries a stale global stamp while dropping every
    # per-layer identity, which exporters previously accepted via truthiness.
    carried.pop("cb_serialized_payload", None)
    carried.pop("cb_render_identity", None)
    carried.pop("whole_artifact_budget", None)
    if uniform_control_status is not None:
        carried["uniform_control"] = uniform_control_status
    if (
        carried
        or selected_cb_context is not None
        or selected_budget is not None
    ):
        carried["selected_by"] = f"validated_frontier:{args.mode}"
        carried["selected_label"] = selected.get("label")
        carried["selected_achieved_bits"] = selected.get("bpp")
        if selected_cb_context is not None:
            carried["cb_serialized_payload"] = dict(selected_cb_context)
            carried["cb_render_identity"] = selected_cb_render_identity
        if selected_budget is not None:
            carried["whole_artifact_budget"] = dict(selected_budget)
        layer_config[LAYER_CONFIG_META_KEY] = carried
    layer_config_path.parent.mkdir(parents=True, exist_ok=True)
    layer_config_path.write_text(json.dumps(layer_config, indent=2, sort_keys=True) + "\n")

    assignment_payload = {
        "schema": "prismaquant.validated_frontier_assignment.v1",
        "selection_mode": args.mode,
        "selected": selected,
        "assignment": dict(sorted(assignment.items())),
        **({
            "cb_serialized_payload": dict(selected_cb_context),
            "cb_render_identity": selected_cb_render_identity,
        } if selected_cb_context is not None else {}),
        **({
            CB_ASSIGNMENT_IDENTITIES_FIELD: dict(sorted(
                (str(name), str(value))
                for name, value in selected_cb_stamps.items()
            )),
        } if selected_cb_stamps else {}),
        **({
            "whole_artifact_budget": dict(selected_budget),
        } if selected_budget is not None else {}),
    }
    assignment_path = Path(args.output_assignment)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps(assignment_payload, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema": "prismaquant.validated_frontier_selection.v1",
        "validation_json": str(Path(args.validation_json)),
        "selection_mode": args.mode,
        "target_disk_gb": args.target_disk_gb,
        "selected_artifact_bytes": selected.get("artifact_bytes"),
        "metric": args.metric,
        "selected": selected,
        "frontier": frontier,
        "practical_knee": practical,
        "kneedle_comparison": knee_cmp,
        "leave_one_out": loo,
        "saturation": saturation,
        "surrogate_spearman": rank_corr,
        "surrogate_worst_rank_inversion": worst_inversion,
        "kl_noise_floor": float(args.kl_noise_floor),
        "tail_veto": {
            "column": tail_veto,
            "eta": tail_eta_arg,
            "eta_mode": "auto" if tail_eta_auto_mode else "explicit",
            "eta_resolved": eta_resolved,
            "inert_reason": tail_inert_reason,
            "n_vetoed": len(vetoed_rows),
        },
        "vetoed_rows": vetoed_rows,
        "practical_rel_eps": float(args.practical_rel_eps),
        "practical_abs_eps": float(args.practical_abs_eps),
        "unstable_policy": args.unstable_policy,
        "n_results": len(results),
        "n_frontier": len(frontier),
        "uniform_control": uniform_control_status,
        "output_layer_config": str(layer_config_path),
        "output_assignment": str(assignment_path),
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    mse = selected.get("mse", {}) if isinstance(selected, Mapping) else {}
    mse_msg = ""
    if isinstance(mse, Mapping) and mse.get("output_mse_sum") is not None:
        mse_msg = f" output_mse={float(mse['output_mse_sum']):.6g}"
    print(
        "[frontier-select] selected "
        f"{selected['label']} bpp={selected['bpp']:.6f} "
        f"KL={selected['kl']:.8g}{mse_msg} mode={args.mode}",
        flush=True,
    )
    if tail_veto is not None and tail_inert_reason is not None:
        print(
            f"[frontier-select] WARNING: tail-veto={tail_veto} is INERT on this "
            f"input ({tail_inert_reason}) — no row carries the column, so the "
            "frontier is the pre-R9 mean-only envelope. Re-run "
            "validate_assignments_kl at this commit to get the tail columns.",
            flush=True,
        )
    elif tail_veto is not None:
        eta_desc = (
            "auto" if tail_eta_auto_mode
            else f"{float(tail_eta_arg):g}"
        )
        print(
            f"[frontier-select] tail-veto={tail_veto} eta={eta_desc}: "
            f"{len(vetoed_rows)} row(s) refused entry to the frontier",
            flush=True,
        )
        if tail_eta_auto_mode:
            for entry in (eta_resolved or []):
                print(
                    f"[frontier-select]   eta[{entry['label']}]="
                    f"{entry['eta']:.6g} ({entry['source']})",
                    flush=True,
                )
            degraded = sorted({
                e["source"] for e in (eta_resolved or [])
                if e["source"] != "derived"
            })
            if degraded:
                print(
                    "[frontier-select] WARNING: --tail-eta auto fell back to a "
                    f"STRICT 0 on at least one incumbent ({', '.join(degraded)}): "
                    "there is no between-seed spread to derive a slack from, so "
                    "the veto is exact-comparison — and single-seed tails are "
                    "noisy (§5: a +10% reading has flipped to -5.2% across "
                    "repeats). Run validate_assignments_kl with "
                    "--calib-repeats>=4 to derive a real slack.",
                    flush=True,
                )
        for row in vetoed_rows:
            print(
                f"[frontier-select]   vetoed {row['label']} "
                f"bpp={float(row['bpp']):.6f} KL={float(row['kl']):.8g} "
                f"reason={row['veto_reason']}"
                + (
                    f" {tail_veto}={float(row['veto_value']):.8g} > "
                    f"limit={float(row['veto_limit']):.8g}"
                    if row.get("veto_value") is not None else ""
                ),
                flush=True,
            )
    if saturation is not None:
        print(
            "[frontier-select] saturation B*="
            f"{saturation['bpp']:.6f} (KL={saturation['kl_at_bstar']:.8g}, "
            f"asymptote@{saturation['asymptote_bpp']:.4f}="
            f"{saturation['kl_asymptote']:.8g}, z={saturation['z']}, "
            f"{saturation['n_measurements']} probes)",
            flush=True,
        )
        if saturation.get("no_noise_floor"):
            print(
                "[frontier-select] WARNING: frontier has no positive per-bpp "
                "stderr -> saturation band is 0 and B* collapsed to the "
                "asymptote (most bits). Re-run validate_assignments_kl with "
                "--calib-repeats>=4 for a real noise floor.",
                flush=True,
            )
    if knee_cmp.get("enabled"):
        log_k = knee_cmp["log_error"]
        raw_k = knee_cmp["raw_linear"]
        print(
            "[frontier-select] kneedle log-error="
            f"{log_k['label']}@{log_k['bpp']:.6f} "
            f"raw-linear={raw_k['label']}@{raw_k['bpp']:.6f}",
            flush=True,
        )
    if rank_corr is not None:
        print(
            "[frontier-select] surrogate-vs-KL fidelity: "
            f"spearman={rank_corr:.4f} (1.0=perfect, surrogate-vs-KL only)",
            flush=True,
        )
        if worst_inversion is not None:
            print(
                f"[frontier-select] worst rank-inversion: {worst_inversion['verdict']}",
                flush=True,
            )
    else:
        print(
            "[frontier-select] surrogate-vs-KL fidelity: unavailable "
            "(need >=3 measured points carrying predicted_dloss_sum)",
            flush=True,
        )
    print(f"[frontier-select] layer_config -> {layer_config_path}", flush=True)
    print(f"[frontier-select] summary -> {summary_path}", flush=True)
    if uniform_control_status is not None:
        selected_bytes = selected.get("artifact_bytes")
        byte_clause = (
            f"{int(selected_bytes)} bytes (bpp {selected.get('bpp')})"
            if selected_bytes is not None
            else f"unpriced bytes (bpp {selected.get('bpp')})"
        )
        print(
            f"[frontier-select] REFUSED: {selected.get('label')} is a "
            f"rate-axis pick ({len(rate_axis)} distinct rung(s), e.g. "
            f"{', '.join(rate_axis[:4])}) and this stage cannot corroborate "
            "it against a byte-matched uniform arm -- the validation set "
            f"holds {len(results)} allocator Pareto rows and no uniform arm, "
            "and the corroborating measurement (served gold KL on both arms) "
            "is one this stage cannot run. Compared: "
            f"{selected.get('label')} KL={selected.get('kl'):.8g} re-ranked "
            f"among {len(results)} Pareto rows; uniform arm: ABSENT. At "
            f"{byte_clause}. This pick ships as a CANDIDATE, not a "
            "selection: build the byte-matched uniform control from the "
            "candidate plan, serve it beside the candidate, and close the "
            "shipcard uniform_control slot (shipcard_cli fill-control) -- "
            "verify and publish refuse until then (prismaquant#117).",
            flush=True,
        )
        return RATE_AXIS_UNCERTIFIED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
