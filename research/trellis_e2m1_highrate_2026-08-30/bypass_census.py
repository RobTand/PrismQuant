#!/usr/bin/env python3
"""Schedule census for the E2M1 trellis ladder: how much of each rung's budget
is spent on BYPASS columns rather than on shaping.

A bypass column carries `bypass_rate` (4) bits and no alphabet
(`stage6_worker.used_alphabets` drops rate 4 outright), so it IS scalar RTN
E2M1 on whatever scale plane the arm uses -- it carries no coding gain by
construction.  If a rung's budget goes to promoting whole columns to bypass
instead of to shaping, the rung's measured SNR is a mixture of shaped columns
(which can carry gain) and RTN columns (which cannot), and a mixture like that
lands on the scalar line for a reason that is a SCHEDULE property, not a format
property.

Pure schedule arithmetic: no encode, no GPU.  Both arms share the schedule
builder, but not the column weights -- arm C weights come from the snapped
two-tier plane and arm D's from the declared plane -- so both are censused.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, "/home/rob/dq-runs/trellis-hull-20260828")

import numpy as np
import torch

import hull_sweep as H

W, C, P, S4, TF = H.W, H.C, H.P, H.S4, H.TF
SPEC = TF.FAMILIES[TF.E2M1_FAMILY]
RATES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
         3.25, 3.5, 3.75, 3.9375, 3.96875)
OUT = Path("/home/rob/dq-runs/trellis-hull-20260828/e2m1_bypass_census.json")


def main() -> int:
    entries = json.loads(H.INPUT_MANIFEST.read_text())["entries"]
    names = [str(e["name"]) for e in entries]
    per = {}
    for index, name in enumerate(names, start=1):
        packed, raw_scale, importance = W.load_compact(name)
        weight = W.dequant_mxfp4(packed, raw_scale, torch.device("cpu"))
        eff = P.eff_scale_plane(weight)
        snapped, _, _, _ = H.two_tier_snap_plane(eff)
        cols = int(weight.shape[1])
        cell = {"shape": list(map(int, weight.shape)), "columns": cols,
                "arms": {}}
        for lane, plane in (("tcq_two_tier", snapped), ("tcq_v1", eff)):
            _, _, _, enc = H.context_from_plane(weight, importance, plane)
            colw = S4.column_weight(enc)
            rows = {}
            for rate in RATES:
                try:
                    sched, _ = S4.build_schedules(rate, cols, colw,
                                                  include_variants=False)
                except AssertionError as exc:
                    # At the family ceiling every 256-block must keep exactly
                    # MIN_TRELLIS_STEPS=8 coded positions, so the global RWF
                    # has ZERO slack to move a bit between blocks.  Some
                    # column-weight spreads cannot be rebalanced into that.
                    # This is a reachability fact about the rung, recorded
                    # rather than crashed on.
                    rows[str(rate)] = {"unreachable": True,
                                       "reason": str(exc)}
                    continue
                s = sched["rwf"]
                counts = {str(r): int((s == r).sum()) for r in (1, 2, 3, 4)}
                n_bypass = counts["4"]
                shaped = cols - n_bypass
                rows[str(rate)] = {
                    "counts": counts,
                    "bypass_columns": n_bypass,
                    "shaped_columns": shaped,
                    "bypass_fraction": n_bypass / cols,
                    "bypass_bit_share": (n_bypass * SPEC.bypass_rate)
                    / int(s.sum()),
                    "mean_rate_shaped_columns": (
                        float(s[s < 4].mean()) if shaped else None),
                    "min_coded_steps_per_block": int(
                        (s.reshape(-1, 256) < 4).sum(1).min()),
                }
            cell["arms"][lane] = rows
        per[name] = cell
        print(f"[{index}/{len(names)}] {name}", flush=True)

    summary = {}
    for lane in ("tcq_two_tier", "tcq_v1"):
        summary[lane] = {}
        for rate in RATES:
            k = str(rate)
            ok = [n for n in names
                  if not per[n]["arms"][lane][k].get("unreachable")]
            unreachable = [n for n in names if n not in ok]
            if not ok:
                summary[lane][k] = {"body_rate": rate,
                                    "unreachable_on_all_tensors": True}
                continue
            fr = [per[n]["arms"][lane][k]["bypass_fraction"] for n in ok]
            bs = [per[n]["arms"][lane][k]["bypass_bit_share"] for n in ok]
            tot_cols = sum(per[n]["columns"] for n in ok)
            tot_by = sum(per[n]["arms"][lane][k]["bypass_columns"] for n in ok)
            counts = {r: sum(per[n]["arms"][lane][k]["counts"][r]
                             for n in ok) for r in ("1", "2", "3", "4")}
            summary[lane][k] = {
                "body_rate": rate,
                "body_rate_q256": int(round(rate * 256)),
                "corpus_columns": tot_cols,
                "corpus_counts_by_rate": counts,
                "corpus_bypass_columns": tot_by,
                "corpus_bypass_fraction": tot_by / tot_cols,
                "bypass_fraction_median": statistics.median(fr),
                "bypass_fraction_min": min(fr),
                "bypass_fraction_max": max(fr),
                "bypass_bit_share_median": statistics.median(bs),
                "min_coded_steps_per_block_over_corpus": min(
                    per[n]["arms"][lane][k]["min_coded_steps_per_block"]
                    for n in ok),
                "tensors_measured": len(ok),
                "tensors_unreachable": unreachable,
            }
    OUT.write_text(json.dumps({
        "schema": "trellis.e2m1_bypass_census.v1",
        "family": SPEC.family, "bypass_rate": SPEC.bypass_rate,
        "shaped_max_rate": SPEC.shaped_max_rate,
        "min_trellis_steps": TF.MIN_TRELLIS_STEPS,
        "note": ("a bypass column carries no alphabet "
                 "(stage6_worker.used_alphabets drops rate 4) and is scalar "
                 "RTN E2M1 on the arm's plane; it carries no coding gain"),
        "summary": summary, "per_tensor": per}, indent=1))

    for lane in ("tcq_two_tier", "tcq_v1"):
        print(f"\n{lane}")
        print(f"{'body':>8} {'q256':>5} {'r1':>7} {'r2':>7} {'r3':>7} "
              f"{'r4=bypass':>10} {'bypass%':>8} {'bitshare%':>10} "
              f"{'shapedmean':>11}")
        for rate in RATES:
            s = summary[lane][str(rate)]
            if s.get("unreachable_on_all_tensors"):
                print(f"{rate:>8} {'':>5} UNREACHABLE on every tensor")
                continue
            c = s["corpus_counts_by_rate"]
            sm = [per[n]["arms"][lane][str(rate)].get(
                      "mean_rate_shaped_columns")
                  for n in names
                  if not per[n]["arms"][lane][str(rate)].get("unreachable")]
            sm = [v for v in sm if v is not None]
            print(f"{rate:>8} {s['body_rate_q256']:>5} {c['1']:>7} {c['2']:>7} "
                  f"{c['3']:>7} {c['4']:>10} "
                  f"{100*s['corpus_bypass_fraction']:>7.2f}% "
                  f"{100*s['bypass_bit_share_median']:>9.2f}% "
                  f"{(statistics.median(sm) if sm else float('nan')):>11.4f}"
                  + ("" if not s["tensors_unreachable"]
                     else f"  UNREACHABLE on {len(s['tensors_unreachable'])}"))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
