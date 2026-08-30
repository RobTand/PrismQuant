#!/usr/bin/env python3
"""Does the four-family menu verdict change when the CB arms are rendered at
the encode tier production actually ships?

THE DEFECT. Every CB cell in `docs/design/format_menu_2026-08-29.md` was
rendered at `encode_tier="max"`, hardcoded in the two ladder drivers
(`fp8_ladder.py:568` `cb_arm_fp8`, `:715` `cb_arm_fp8_learned`;
`hull_sweep.py:995` and `:1046` `cb_arm_two_tier`) and recorded in neither
results file.  Production ships `balanced`: it is the library default
(`prismaquant/nvfp4_cb_formats.py:173`), the pipeline default
(`prismaquant/run-pipeline.sh:217`), and the explicit setting in every shipped
driver that names the variable (10 under `scripts/`, 2 under `tools/`; none set
`max`).  `max` is the exhaustive *sweep*, not an exhaustive *search* -- its
`_candidate_scales` window is `amax / linspace(448, 448*4/6, 16)`, i.e. scales
in [1.0, 1.5] x amax/448 anchored on the fixed lattice's grid-max
(`nvfp4_cb_formats.py:872-883`), and the module's own note at `:180` predicts
what follows: "the reach, not granularity, sets quality ... BEATS max at
+-34%".  On the fp8 grid that window is binding, so `max` is a *lower* bound
for fp8 CB, not an upper one.

WHAT THIS SCRIPT DOES.  It re-solves the identical per-tensor Lagrangian that
`verdict/menu_selection_4families.py` solves -- same corpus, same metric, same
budgets, same tie-break to fewer bits -- with exactly one substitution: the six
`fp8_cb@K` arms take their `weighted_sse` from the shipped-tier (`balanced`)
render measured in `fp8_learned_tier_ab.json`.  Everything else is byte-for-byte
the published input.  It then diffs the selected / never-selected / contested
sets against `verdict/menu_selection_4families.json`.

WHY THE SUBSTITUTION IS LEGITIMATE, AND WHAT GATES IT.  Five refusals, all
fatal:

  1. the four joinability gates the published solver already applies
     (`weighted_energy` to 1e-9, `source_weight_sha256`, fp8 `selfcheck.status`
     PASS, rectangular corpus);
  2. every `max`-tier cell of the A/B must reproduce the published ladder cell
     in `weighted_snr_db` to 1e-9 -- otherwise the A/B is not the ladder and its
     `balanced` cells are not a tier comparison;
  3. ... and in `exact_bpw` exactly -- so the substitution moves error and
     nothing else.  bpw is tier-independent by construction (same k, same rows);
  4. the fp8-CB row-scale plane must be charged once and only once.  The
     published ladder files predate `apply_cb_scale_fix.py`, so their
     `exact_bpw` is exactly `k/8` and the published solver adds the plane
     post-hoc; the tier A/B was rendered by the post-fix driver and charges it
     natively.  The gate is therefore an identity, not an absence:
     `ladder exact_bpw + cb_scale_correction_bpw == A/B exact_bpw`, checked per
     cell.  (This is also an independent confirmation of open item 4 in the
     format-menu doc: the two paths land on the same bytes.)
  5. the A/B carries only the fp8 arms, so the fp4-CB and both trellis families
     are read from the published files untouched.

WHAT IT CANNOT DECIDE.  The fp4-CB arms stay at `max` here, because the tier is
measured not to matter on that grid: -2.6e-10 to +0 dB over
{k=8, 13, 18} x 3 tensors, and bit-identical on 24/24 tensors at k=18
(`encode_tier_fidelity.json`, `encode_tier_fidelity_full24.json`).  That is a
measured bound at three index widths, not a proof at all eighteen.  And this is
corpus SSE on 24 DeepSeek-V4-Flash MoE expert tensors -- a screen, per
principle 3, not a serving result.

It also emits, unconditionally, the per-tensor learned-fp8-CB-versus-E4M3-
trellis comparison at matched bytes (`learned_fp8_vs_trellis_matched_bytes`).
The Lagrangian sweep answers "what does a budget select"; that block answers
"at the same byte count, which wire is more accurate", which is the claim
section 1 of the format-menu doc makes.  A pair counts as byte-matched only
when the two arms' median `exact_bpw` agree to 0.05; when a bracket holds no
matched pair the nearest one per rung is reported and tagged
`byte_matched: false`, so an unmatched pair can never be read as dominance.

`--with-learned-fp8` additionally puts the per-tensor learned fp8-CB book on the
menu.  That arm is EXPOSURE-SIZING ONLY and must not be read as a menu change:
its K44/K48 rungs sit above this corpus's ~4.7-bit content ceiling, so the
direction of the learned-book win is valid but its size is corpus-scoped and a
bf16-corpus re-check is owed first.

Usage (CPU only, ~seconds):
    python3 docs/design/resolve_menu_at_shipped_tier.py
    python3 docs/design/resolve_menu_at_shipped_tier.py --with-learned-fp8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUN = Path("/home/rob/dq-runs/trellis-hull-20260828")
FOURBIT = RUN / "hull_results.json"
FP8 = {
    "production": RUN / "fp8_ladder_production_row_fp32.json",
    "penalty": RUN / "fp8_ladder_two_tier.json",
}
TIER_AB = RUN / "fp8_learned_tier_ab.json"
PUBLISHED_VERDICT = RUN / "verdict" / "menu_selection_4families.json"

FOURBIT_PREFIXES = ("cb_two_tier@", "tcq_two_tier@")
FP8_PREFIXES = ("fp8_cb@", "tcq_e4m3@")
FP8_CB_RUNGS = (28, 32, 36, 40, 44, 48)
SHIPPED_TIER = "balanced"
LADDER_TIER = "max"
RTOL = 1e-9

FAMILY = {
    "cb_two_tier": "fp4 lattice CB",
    "tcq_two_tier": "E2M1 trellis",
    "fp8_cb": "fp8 lattice CB",
    "fp8_cb_learned": "fp8 learned CB (exposure only)",
    "tcq_e4m3": "E4M3 trellis",
}


def close(a, b):
    return abs(a - b) <= RTOL * abs(b)


def cb_scale_correction_bpw(arm_name, arm, shape):
    """Mirror of the published solver's post-hoc charge, kept so this script
    reproduces it exactly on a pre-fix input.  On the post-fix ladder it must
    return 0 -- asserted by the caller."""
    if not arm_name.startswith("fp8_cb@"):
        return 0.0
    if float(arm["footprint"].get("scale_bpw", 0.0)):
        return 0.0
    rows, columns = (int(v) for v in shape)
    return rows * 4 * 8 / float(rows * columns)


def load_tier_ab():
    doc = json.loads(TIER_AB.read_text())
    gate = doc.get("reproduction_gate", {})
    if gate.get("status") != "PASS":
        raise SystemExit(
            f"REFUSING: {TIER_AB.name} reproduction_gate is "
            f"{gate.get('status', 'ABSENT')!r}; its max-tier cells do not "
            f"reproduce the ladder, so its balanced cells are not a tier A/B")
    return doc["per_tensor"]


def load(bracket, tier_ab, with_learned):
    four = json.loads(FOURBIT.read_text())["per_tensor"]
    fp8_doc = json.loads(FP8[bracket].read_text())
    check = fp8_doc.get("selfcheck", {})
    if check.get("status") != "PASS":
        raise SystemExit(
            f"REFUSING to join {FP8[bracket].name}: selfcheck.status="
            f"{check.get('status', 'ABSENT')!r}")
    fp8 = fp8_doc["per_tensor"]

    tensors, swapped, deltas = [], 0, []
    for name, rec4 in four.items():
        rec8 = fp8.get(name)
        if rec8 is None:
            raise SystemExit(f"{name}: absent from the fp8 ladder")
        if not close(float(rec8["weighted_energy"]),
                     float(rec4["weighted_energy"])):
            raise SystemExit(f"{name}: weighted_energy differs; the SSE "
                             f"denominators are not the same object")
        if rec8["source_weight_sha256"] != rec4["source_weight_sha256"]:
            raise SystemExit(f"{name}: different source weight bytes")
        ab = tier_ab.get(name)
        if ab is None:
            raise SystemExit(f"{name}: absent from the tier A/B; the tier "
                             f"substitution would cover part of the corpus")
        if not close(float(ab["weighted_energy"]),
                     float(rec4["weighted_energy"])):
            raise SystemExit(f"{name}: tier A/B weighted_energy differs")
        if ab["source_weight_sha256"] != rec4["source_weight_sha256"]:
            raise SystemExit(f"{name}: tier A/B encoded different bytes")

        numel = float(rec4["numel"])
        rungs = []
        for rec, prefixes in ((rec4, FOURBIT_PREFIXES), (rec8, FP8_PREFIXES)):
            for arm_name, arm in rec["arms"].items():
                if not arm_name.startswith(prefixes):
                    continue
                sse = float(arm["weighted_sse"])
                bpw = float(arm["footprint"]["exact_bpw"])
                bpw += cb_scale_correction_bpw(arm_name, arm, rec["shape"])

                if arm_name.startswith("fp8_cb@"):
                    k = int(arm_name.split("@")[1])
                    ref = ab["arms"][f"fp8_cb@{k}|{LADDER_TIER}"]
                    if not close(float(ref["weighted_snr_db"]),
                                 float(arm["weighted_snr_db"])):
                        raise SystemExit(
                            f"{name}/{arm_name}: the A/B's max cell "
                            f"({ref['weighted_snr_db']}) does not reproduce "
                            f"the ladder ({arm['weighted_snr_db']})")
                    if float(ref["footprint"]["exact_bpw"]) != bpw:
                        raise SystemExit(
                            f"{name}/{arm_name}: A/B bpw "
                            f"{ref['footprint']['exact_bpw']} != ladder-plus-"
                            f"row-plane {bpw}; the row scale plane is charged "
                            f"differently on the two paths, so the "
                            f"substitution would move bytes as well as error")
                    ship = ab["arms"][f"fp8_cb@{k}|{SHIPPED_TIER}"]
                    if float(ship["footprint"]["exact_bpw"]) != bpw:
                        raise SystemExit(
                            f"{name}/{arm_name}: shipped-tier bpw differs from "
                            f"the ladder's; bpw must be tier-independent")
                    deltas.append((k, float(ship["weighted_snr_db"])
                                   - float(ref["weighted_snr_db"])))
                    sse = float(ship["weighted_sse"])
                    swapped += 1

                rungs.append((arm_name, bpw, sse, bpw * numel))

        if with_learned:
            for k in FP8_CB_RUNGS:
                arm = ab["arms"][f"fp8_cb_learned@{k}|{SHIPPED_TIER}"]
                bpw = float(arm["footprint"]["exact_bpw"])   # FP16 book charge
                rungs.append((f"fp8_cb_learned@{k}", bpw,
                              float(arm["weighted_sse"]), bpw * numel))

        tensors.append({"name": name, "numel": numel, "rungs": rungs})
    return tensors, swapped, deltas


def matched_byte_pairs(bracket, tier_ab, tol_bpw=0.05):
    """Per-tensor learned-fp8-CB versus E4M3-trellis comparison at matched bytes.

    The Lagrangian sweep answers "what does a budget select"; it does not answer
    "at the same byte count, which wire is more accurate".  This does, pairwise
    and per tensor, so the crossover claim has a producer rather than a
    recollection.  Only pairs whose median bpw agree to `tol_bpw` are called
    byte-matched; the rest are reported with their gap so no one reads an
    unmatched pair as a dominance result.
    """
    fp8 = json.loads(FP8[bracket].read_text())["per_tensor"]
    out = []
    for k in FP8_CB_RUNGS:
        for rate in ("2.0", "3.0", "4.0", "5.0", "6.0"):
            ddb, dbpw = [], []
            for name, rec in fp8.items():
                book = tier_ab[name]["arms"][f"fp8_cb_learned@{k}|{SHIPPED_TIER}"]
                wire = rec["arms"][f"tcq_e4m3@{rate}"]
                ddb.append(float(book["weighted_snr_db"])
                           - float(wire["weighted_snr_db"]))
                dbpw.append(float(book["footprint"]["exact_bpw"])
                            - float(wire["footprint"]["exact_bpw"]))
            ddb.sort()
            dbpw.sort()
            n = len(ddb)
            median_bpw = dbpw[n // 2]
            out.append({"book": f"fp8_cb_learned@{k}", "wire": f"tcq_e4m3@{rate}",
                        "byte_matched": abs(median_bpw) <= tol_bpw,
                        "book_wins_db": sum(1 for d in ddb if d > 0), "n": n,
                        "median_delta_db": ddb[n // 2],
                        "min_delta_db": ddb[0], "max_delta_db": ddb[-1],
                        "median_delta_bpw": median_bpw})
    matched = [r for r in out if r["byte_matched"]]
    if matched:
        return matched
    # No pair lands within tolerance -- report the nearest one per book rung so
    # the absence is legible.  An unmatched pair is not a dominance result and
    # is tagged as such.
    nearest = {}
    for r in out:
        cur = nearest.get(r["book"])
        if cur is None or abs(r["median_delta_bpw"]) < abs(
                cur["median_delta_bpw"]):
            nearest[r["book"]] = r
    return sorted(nearest.values(), key=lambda r: r["book"])


def solve(tensors, lam):
    """Ties break to FEWER bits -- conservative for a retirement argument."""
    picks, bits, cost = [], 0.0, 0.0
    for tensor in tensors:
        best = None
        for arm_name, _bpw, sse, tensor_bits in tensor["rungs"]:
            value = sse + lam * tensor_bits
            if best is None or value < best[0] - 1e-30 or (
                    abs(value - best[0]) <= 1e-30 and tensor_bits < best[3]):
                best = (value, arm_name, sse, tensor_bits)
        picks.append((tensor["name"], best[1]))
        bits += best[3]
        cost += best[2]
    return picks, bits, cost


def bisect(tensors, target_bpw, total_numel):
    target = target_bpw * total_numel
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        _, bits, _ = solve(tensors, mid)
        if bits > target:
            lo = mid
        else:
            hi = mid
    picks, bits, cost = solve(tensors, hi)
    return hi, picks, bits, cost


def run(bracket, tier_ab, with_learned):
    tensors, swapped, deltas = load(bracket, tier_ab, with_learned)
    total = sum(t["numel"] for t in tensors)
    all_rungs = sorted({r[0] for t in tensors for r in t["rungs"]})
    budgets = [round(1.30 + 0.10 * i, 2)
               for i in range(int((6.60 - 1.30) / 0.10) + 1)]
    ever = {}
    print(f"\n=== bracket: {bracket} "
          f"({swapped} fp8_cb cells swapped max -> {SHIPPED_TIER}) ===")
    head = f"{'budget':>7} {'achieved':>9} {'wSSE':>12}   selection"
    print(head)
    print("-" * len(head))
    for target in budgets:
        _lam, picks, bits, cost = bisect(tensors, target, total)
        counts = {}
        for _, arm in picks:
            counts[arm] = counts.get(arm, 0) + 1
            ever[arm] = ever.get(arm, 0) + 1
        pretty = ", ".join(f"{a}x{n}" for a, n in
                           sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"{target:>7.2f} {bits/total:>9.4f} {cost:>12.4f}   {pretty}")
    never = [r for r in all_rungs if r not in ever]
    by_family = {}
    for arm in ever:
        by_family.setdefault(FAMILY.get(arm.split("@")[0], "?"), []).append(arm)
    print(f"\nSELECTED ({len(ever)}/{len(all_rungs)}):")
    for fam, arms in sorted(by_family.items()):
        print(f"  {fam:32s} {', '.join(sorted(arms))}")
    print(f"NEVER SELECTED ({len(never)}): {', '.join(never) or '(none)'}")
    return {"bracket": bracket, "menu_size": len(all_rungs),
            "cells_swapped": swapped,
            "ever_selected": ever, "never_selected": never,
            "selected_by_family": {k: sorted(v) for k, v in by_family.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-learned-fp8", action="store_true",
                    help="add the per-tensor learned fp8-CB book to the menu; "
                         "EXPOSURE-SIZING ONLY (corpus-ceiling caveat applies)")
    ap.add_argument("--out", default=None,
                    help="write the report JSON here (default: alongside the "
                         "published verdict, *.shipped_tier[.learned].json)")
    args = ap.parse_args()

    tier_ab = load_tier_ab()
    reports = {"schema": "trellis.menu_selection_shipped_tier.v1",
               "shipped_tier": SHIPPED_TIER, "ladder_tier": LADDER_TIER,
               "with_learned_fp8": bool(args.with_learned_fp8),
               "sources": [str(FOURBIT), str(FP8["production"]),
                           str(FP8["penalty"]), str(TIER_AB)]}
    for bracket in ("production", "penalty"):
        reports[bracket] = run(bracket, tier_ab, args.with_learned_fp8)

    a = set(reports["production"]["ever_selected"])
    b = set(reports["penalty"]["ever_selected"])
    retired = sorted(set(reports["production"]["never_selected"])
                     & set(reports["penalty"]["never_selected"]))
    contested = sorted(a ^ b)
    selected_both = sorted(a & b)
    reports["agreement"] = {"retired_under_both": retired,
                            "contested": contested,
                            "selected_under_both": selected_both}
    print("\n=== BRACKET AGREEMENT (shipped tier) ===")
    print(f"retired under BOTH ({len(retired)}): {', '.join(retired)}")
    print(f"contested ({len(contested)}): {', '.join(contested) or '(none)'}")
    print(f"selected under BOTH ({len(selected_both)}): "
          f"{', '.join(selected_both)}")

    pub = json.loads(PUBLISHED_VERDICT.read_text())["agreement"]
    diff = {}
    for key, pub_key in (("retired_under_both", "retired_under_both"),
                         ("contested", "contested")):
        was, now = set(pub[pub_key]), set(reports["agreement"][key])
        diff[key] = {"added": sorted(now - was), "removed": sorted(was - now)}
    reports["diff_vs_published_max_tier"] = diff
    print("\n=== DIFF vs the published max-tier verdict ===")
    flipped = False
    for key, d in diff.items():
        if d["added"] or d["removed"]:
            flipped = True
            print(f"  {key}: +{d['added']} -{d['removed']}")
    if not flipped:
        print("  NO CHANGE: the same rungs are retired, contested and "
              "selected at the shipped tier as at max.")
    reports["verdict_changed"] = flipped

    pairs = {b: matched_byte_pairs(b, tier_ab) for b in ("production", "penalty")}
    reports["learned_fp8_vs_trellis_matched_bytes"] = pairs
    print("\n=== learned fp8-CB book vs E4M3 trellis, per tensor, "
          "matched bytes (NOT a verdict: exposure sizing, and the corpus "
          "content ceiling scopes the MAGNITUDE) ===")
    for bracket, rows in pairs.items():
        for r in rows:
            print(f"  {bracket:10s} {r['book']:20s} vs {r['wire']:14s} "
                  f"book wins {r['book_wins_db']}/{r['n']}  "
                  f"median {r['median_delta_db']:+7.3f} dB "
                  f"[{r['min_delta_db']:+.3f},{r['max_delta_db']:+.3f}]  "
                  f"median Dbpw {r['median_delta_bpw']:+.4f}"
                  f"{'' if r['byte_matched'] else '  NOT byte-matched'}")

    out = Path(args.out) if args.out else PUBLISHED_VERDICT.with_suffix(
        ".shipped_tier.learned.json" if args.with_learned_fp8
        else ".shipped_tier.json")
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
