#!/usr/bin/env python3
"""Separate the TRELLIS CODE gain from the RATE-SHAPING gain.

A trellis rung at body rate R does not put every column at rate R: reverse
water-filling shapes per-column rates by importance.  So "trellis at body R vs
flat scalar at R bits" credits the trellis lane with BOTH the code and the
shaping.  This control gives the scalar side the SAME RWF schedule: every
column scheduled at rate r is quantized by RTN onto the weighted-MSE-optimal
2^r-level subset of the E2M1 grid (the whole grid at r=4, which is what a
bypass column is).  The remaining difference is the code, and only the code.

Declared (v1) plane only -- that is the arm the NVFP4 comparison uses, and the
plane where NVFP4 itself lives.  No trellis encode here; the trellis side comes
from the published ladder.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, "/home/rob/dq-runs/trellis-hull-20260828")

import numpy as np
import torch

import hull_sweep as H

W, C, P, S4 = H.W, H.C, H.P, H.S4
RATES = (1.5, 2.0, 2.5)          # the published arm-D rungs
OUT = Path("/home/rob/dq-runs/trellis-hull-20260828/shaped_scalar_control.json")


def main() -> int:
    pub = json.loads((Path("/home/rob/dq-runs/trellis-hull-20260828") /
                      "hull_results.menuext-20260829.json").read_text())["per_tensor"]
    entries = json.loads(H.INPUT_MANIFEST.read_text())["entries"]
    names = [str(e["name"]) for e in entries]
    per = {}
    for index, name in enumerate(names, start=1):
        packed, raw_scale, importance = W.load_compact(name)
        weight = W.dequant_mxfp4(packed, raw_scale, torch.device("cpu"))
        eff = P.eff_scale_plane(weight)
        x, pes, metric_w, enc_w = H.context_from_plane(weight, importance, eff)
        energy = C.weighted_sse(weight, torch.zeros_like(weight), metric_w)
        columns = int(weight.shape[1])
        cell = {"weighted_energy": energy, "numel": int(weight.numel()),
                "rungs": {}}
        for rate in RATES:
            sched, _ = S4.build_schedules(rate, columns, S4.column_weight(enc_w),
                                          include_variants=False)
            sched = torch.as_tensor(np.asarray(sched["rwf"]))
            total, classes = 0.0, {}
            for r in (1, 2, 3, 4):
                idx = (sched == r).nonzero(as_tuple=True)[0]
                if not idx.numel():
                    continue
                w_s = weight.index_select(1, idx)
                m_s = metric_w.index_select(1, idx)
                x_s = x.index_select(1, idx)
                e_s = enc_w.index_select(1, idx)
                p_s = pes.index_select(1, idx)
                m_levels = (1 << r) if r < 4 else len(P.E2M1_LEVELS)
                levels, _ = P.best_level_subset(x_s, e_s, m_levels)
                recon = P.quantize_to_levels(x_s, levels, p_s)
                sse = C.weighted_sse(w_s, recon.to(weight.dtype), m_s)
                total += sse
                classes[str(r)] = {
                    "columns": int(idx.numel()), "n_levels": m_levels,
                    "levels": [float(v) for v in levels], "wsse": sse,
                    "energy": C.weighted_sse(w_s, torch.zeros_like(w_s), m_s)}
            trellis_db = pub[name]["arms"][f"tcq_v1@{rate}"]["weighted_snr_db"]
            shaped_db = 10.0 * math.log10(energy / total)
            cell["rungs"][str(rate)] = {
                "shaped_scalar_wsse": total, "shaped_scalar_db": shaped_db,
                "trellis_db": trellis_db,
                "code_only_gain_db": trellis_db - shaped_db,
                "classes": classes}
        per[name] = cell
        print(f"[{index}/{len(names)}] {name}  " + "  ".join(
            f"R{r}: code_gain {cell['rungs'][str(r)]['code_only_gain_db']:+.3f}"
            for r in RATES), flush=True)

    summary = {}
    for rate in RATES:
        g = [per[n]["rungs"][str(rate)]["code_only_gain_db"] for n in names]
        sd = [per[n]["rungs"][str(rate)]["shaped_scalar_db"] for n in names]
        td = [per[n]["rungs"][str(rate)]["trellis_db"] for n in names]
        summary[str(rate)] = {
            "shaped_scalar_db_median": statistics.median(sd),
            "trellis_db_median": statistics.median(td),
            "code_only_gain_db_median": statistics.median(g),
            "code_only_gain_db_min": min(g), "code_only_gain_db_max": max(g),
            "tensors_positive": sum(1 for v in g if v > 0), "n": len(g)}
    OUT.write_text(json.dumps({
        "schema": "trellis.shaped_scalar_control.v1",
        "plane": "v1 declared group16_fp8_e4m3_0p5_bpw",
        "note": ("scalar side carries the SAME RWF per-column schedule as the "
                 "trellis rung, so the residual is the trellis CODE alone and "
                 "not the rate shaping"),
        "summary": summary, "per_tensor": per}, indent=1))

    print(f"\n{'body':>6} {'shaped scalar':>14} {'trellis':>9} "
          f"{'CODE-ONLY gain':>15} {'positive':>10}")
    for rate in RATES:
        s = summary[str(rate)]
        print(f"{rate:>6} {s['shaped_scalar_db_median']:>14.3f} "
              f"{s['trellis_db_median']:>9.3f} "
              f"{s['code_only_gain_db_median']:>+15.3f} "
              f"{s['tensors_positive']:>7}/{s['n']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
