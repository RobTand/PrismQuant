#!/usr/bin/env python3
"""Locate the Pareto knee, and say what happened to the menu-exhaustion cliff.

HEADLINE OF DELIVERABLE (3). The old-menu curve put the global knee at
103.6 GB / 2.499 bpp -- far outside the 88-92 GB shave window -- but it also
had a brutal local CLIFF at 88.8-90.2 GB costing 3,891 Delta-loss per GB. That
cliff was not physics: at 88 GB the DP needs 2.055 bpp while the old menu's
cheapest expert rung was NVFP4_CB_K14 at 2.031, so it sat within 1% of its own
floor with nothing left to trade. K12 (1.781) and K13 (1.906) put real headroom
underneath it. This script asks whether that changed the answer.

FRAMES. Bytes and bits are related by an affine map whose intercept is the
immutable floor, so releasing the MTP block moves the chord endpoints and can
move the knee. The knee is therefore reported in BOTH frames:

    (b) MTP in the floor      GB = floor + reserve + bpp * params/8
    (c) MTP bytes released    same, with floor reduced by 10,862,838,300 B

Comparing them answers a question the grid alone cannot: whether the draft
heads change WHERE the efficient budget is, not merely how much loss it costs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MTP_BYTES = 10_862_838_300
OLD_KNEE_GB, OLD_KNEE_BPP = 103.612, 2.4994
OLD_CLIFF = {"segment_gb": (88.803, 90.230), "dloss_per_gb": -3890.6}
NEAR_FREE_MULTIPLE = 3.0          # "near free" = within 3x the 92-GB-adjacent rate


def load_curve(pareto_csv: Path) -> list[tuple[float, float]]:
    rows = [
        (float(r["achieved_bits"]), float(r["predicted_dloss"]))
        for r in csv.DictReader(pareto_csv.open())
        if r.get("feasible") == "True"
    ]
    return sorted(set(rows))


def affine_from_selection(selection: Path) -> tuple[float, float]:
    """Return (intercept_gb, gb_per_bpp) for GB = intercept + bpp * slope."""
    sel = json.loads(selection.read_text())
    bits = float(sel["chosen_achieved_bits"])
    total = float(sel["predicted_whole_artifact_upper_bound_gb"])
    floor = float(sel["predicted_floor_gb"])
    reserve = float(sel.get("artifact_overhead_reserve_bytes", 268435456)) / 1e9
    return floor + reserve, (total - floor - reserve) / bits


def kneedle(xs: list[float], ys: list[float]) -> int:
    """Index of maximum deviation from the chord joining the endpoints.

    The same construction that produced the old-menu 103.6 GB figure, kept
    identical so old and new are comparable rather than merely both plausible.
    """
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    if x1 == x0 or y1 == y0:
        return 0
    nx = [(x - x0) / (x1 - x0) for x in xs]
    ny = [(y - y0) / (y1 - y0) for y in ys]
    dev = [abs(nx[i] - ny[i]) for i in range(len(nx))]
    return max(range(len(dev)), key=lambda i: dev[i])


def analyse(curve, intercept, slope, frame: str) -> dict:
    pts = sorted((intercept + b * slope, d, b) for b, d in curve)
    gbs = [p[0] for p in pts]
    dls = [p[1] for p in pts]
    ki = kneedle(gbs, dls)

    segments = []
    for i in range(1, len(pts)):
        g0, d0, _ = pts[i - 1]
        g1, d1, _ = pts[i]
        if g1 == g0:
            continue
        segments.append({
            "from_gb": round(g0, 3), "to_gb": round(g1, 3),
            "dloss_per_gb": round((d1 - d0) / (g1 - g0), 1),
        })

    # "3x the 92-GB-adjacent rate" is ambiguous and the two readings give
    # materially different answers, so both are reported rather than one being
    # silently picked. SHAVE-RELEVANT is the rate immediately BELOW 92 -- what
    # the first GB of the shave actually costs -- and it is the stricter,
    # more decision-honest reference. The segment ABOVE 92 is what you would
    # pay to buy quality back, which is a different question.
    def _rate_below(limit):
        below = [s for s in segments if s["to_gb"] <= limit]
        return abs(below[-1]["dloss_per_gb"]) if below else 0.0

    def _rate_spanning(limit):
        return next((abs(s["dloss_per_gb"]) for s in segments
                     if s["from_gb"] <= limit <= s["to_gb"]), 0.0)

    def _boundary(ref):
        if ref <= 0:
            return None
        for seg in sorted((s for s in segments if s["to_gb"] <= 92.0),
                          key=lambda s: -s["to_gb"]):
            if abs(seg["dloss_per_gb"]) > NEAR_FREE_MULTIPLE * ref:
                return seg["to_gb"]
        return None

    ref = _rate_below(92.0)                 # shave-relevant (strict)
    ref_above = _rate_spanning(92.0)        # buy-back rate (lenient)
    boundary = _boundary(ref)
    boundary_lenient = _boundary(ref_above)

    window = [s for s in segments if 86.0 <= s["from_gb"] <= 94.0]
    old_lo, old_hi = OLD_CLIFF["segment_gb"]
    overlapping = [s for s in window
                   if s["to_gb"] > old_lo and s["from_gb"] < old_hi]
    worst = min((s["dloss_per_gb"] for s in overlapping), default=None)

    return {
        "frame": frame,
        "knee_gb": round(gbs[ki], 3),
        "knee_bpp": round(pts[ki][2], 4),
        "knee_dloss": round(dls[ki], 1),
        "knee_inside_shave_window_88_92": 88.0 <= gbs[ki] <= 92.0,
        "near_free_zone": {
            "reference_rate_shave_relevant_dloss_per_gb": round(ref, 1),
            "reference_rate_buyback_dloss_per_gb": round(ref_above, 1),
            "threshold_multiple": NEAR_FREE_MULTIPLE,
            "boundary_gb": boundary,
            "boundary_gb_lenient_reference": boundary_lenient,
            "meaning": (
                "budgets down to boundary_gb cost within "
                f"{NEAR_FREE_MULTIPLE}x the rate already being accepted at "
                "92 GB; below it the shave gets materially more expensive"
                if boundary is not None else
                "no segment below 92 GB exceeds the threshold: the curve is "
                "locally flat and the whole window is near-free"
            ),
        },
        "cliff_region_87_93": window,
        "old_menu_cliff": {
            **OLD_CLIFF,
            "new_menu_worst_dloss_per_gb_in_same_span": worst,
            "flattened": (None if worst is None
                          else abs(worst) < abs(OLD_CLIFF["dloss_per_gb"])),
            "flattening_factor": (None if not worst else
                                  round(OLD_CLIFF["dloss_per_gb"] / worst, 2)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True, type=Path)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    curve = load_curve(args.pareto)
    intercept, slope = affine_from_selection(args.selection)
    report = {
        "schema": "prismaquant.pareto_knee_analysis.v1",
        "method": "max deviation from the endpoint chord (kneedle), identical "
                  "to the construction used on the old-menu curve",
        "old_menu": {"knee_gb": OLD_KNEE_GB, "knee_bpp": OLD_KNEE_BPP,
                     "menu": "NVFP4_CB_K14/K15 + FP8_CB_K36 + MXFP4_SOURCE"},
        "n_pareto_points": len(curve),
        "frame_note": (
            "The (b)->(c) frame shift is a pure TRANSLATION of the GB axis by "
            "the MTP bytes, and the kneedle normalisation (x-x0)/(x1-x0) is "
            "translation-invariant -- so the knee is necessarily the SAME "
            "allocation in both frames, with its GB label 10.863 lower in (c). "
            "The two frames are reported because the question was asked, and "
            "because a disagreement between them would indicate an "
            "implementation error rather than a finding."
        ),
        "frames": {
            "b_mtp_in_floor": analyse(curve, intercept, slope,
                                      "MTP in floor"),
            "c_mtp_released": analyse(curve, intercept - MTP_BYTES / 1e9,
                                      slope, "MTP bytes released"),
        },
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
