#!/usr/bin/env python3
"""Merge the E2M1 trellis ladder across its measured band and price it against
scalar NVFP4 on the same 24 tensors.

Rows 1.0-3.0 come from the sealed sweep + its 08-29 menu extension; rows
3.25-3.96875 come from `e2m1_highrate.py`.  The incumbent comes from
`format-ladder-20260829/scalar_incumbents.json`.

PRICING, both planes, never mixed inside one comparison:
  research  = the hull's `tcq_two_tier` 0.28125 bpw two-tier scale plane.
              Comparable to the published ladder.  NOT on the wire.
  attested  = the family's declared `group16_fp8_e4m3_0p5_bpw` plane, read from
              the SAME row's `production_payload_v1`.  Comparable to NVFP4
              (which is priced on exactly that plane) and to production.
The NVFP4 comparison uses ATTESTED only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path("/home/rob/dq-runs/trellis-hull-20260828")
PUBLISHED = ROOT / "hull_results.menuext-20260829.json"
HIGHRATE = ROOT / "e2m1_highrate_rows.json"
SCALAR = Path("/home/rob/dq-runs/format-ladder-20260829/scalar_incumbents.json")
OUT = ROOT / "e2m1_highrate_results.json"

LOW_RATES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
HIGH_RATES = (3.25, 3.5, 3.75, 3.9375, 3.96875)
# Arm D exists on the published ladder only at 1.5/2.0/2.5; the high-rate run
# adds 3.0 and 3.25-3.96875.  2.5 is measured on BOTH and is a control.
ARM_D_LOW = (1.5, 2.0, 2.5)


def db(nsse: float) -> float:
    return -10.0 * math.log10(nsse)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--highrate", type=Path, default=HIGHRATE)
    parser.add_argument("--scalar", type=Path, default=SCALAR)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pub = json.loads(args.published.read_text())["per_tensor"]
    hr_doc = json.loads(args.highrate.read_text())
    hr = hr_doc["per_tensor"]
    sc = json.loads(args.scalar.read_text())
    spt = sc["per_tensor"]
    names = sorted(pub)
    assert set(names) == set(hr) == set(spt), "corpus mismatch"

    numel = {n: spt[n]["numel"] for n in names}
    total_numel = sum(numel.values())
    nv = {n: spt[n]["arms"]["scalar_nvfp4"]["weighted_nsse"] for n in names}
    nv_bpw = {n: spt[n]["arms"]["scalar_nvfp4"]["footprint"]["exact_bpw"]
              for n in names}
    # scalar_incumbents.py's own corpus aggregate: numel-weighted mean nsse.
    nv_corpus_nsse = sum(nv[n] * numel[n] for n in names) / total_numel
    nv_corpus_db = db(nv_corpus_nsse)
    nv_median_db = statistics.median(db(nv[n]) for n in names)
    nv_corpus_bpw = (sum(nv_bpw[n] * numel[n] for n in names) / total_numel)

    arm_c_rates, arm_c_incomplete = complete_rates(
        pub, hr, names, "tcq_two_tier", (*LOW_RATES, *HIGH_RATES), LOW_RATES)
    rungs = build_rungs(pub, hr, names, numel, total_numel, nv, nv_bpw,
                        "tcq_two_tier", arm_c_rates, LOW_RATES)
    arm_d_rates, arm_d_incomplete = complete_rates(
        pub, hr, names, "tcq_v1", (*ARM_D_LOW, 3.0, *HIGH_RATES), ARM_D_LOW)
    rungs_attested_render = build_rungs(
        pub, hr, names, numel, total_numel, nv, nv_bpw,
        "tcq_v1", arm_d_rates, ARM_D_LOW)

    def crossing(key_db: str, key_bpw: str, target_db: float,
                 table=None):
        """Linear interpolation of the ladder onto `target_db`. INTERPOLATED
        where it lands between two measured rungs, EXTRAPOLATED beyond the
        top rung (the family's ceiling), and said so."""
        pts = [(r[key_bpw], r[key_db]) for r in (table or rungs)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if (y0 - target_db) * (y1 - target_db) <= 0 and y1 != y0:
                return {"bpw": x0 + (target_db - y0) * (x1 - x0) / (y1 - y0),
                        "kind": "interpolated between measured rungs",
                        "between_body_rates": None}
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        slope = (yb - ya) / (xb - xa)
        return {"bpw": xb + (target_db - yb) / slope,
                "kind": ("EXTRAPOLATED beyond the family's ceiling -- the "
                         "ladder never reaches this dB inside the format"),
                "slope_db_per_bpw_at_top": slope}

    payload = {
        "schema": "trellis.e2m1_highrate_analysis.v1",
        "question": ("does the E2M1 (4-bit) trellis beat scalar NVFP4 at "
                     "<= 4.5 bpw on the 24-tensor DSv4 routed-expert corpus"),
        "corpus": {"n_tensors": len(names), "names": names,
                   "numel_per_tensor": sorted(set(numel.values()))},
        "incumbent_nvfp4": {
            "bpw_attested": nv_corpus_bpw,
            "wsnr_db_corpus": nv_corpus_db,
            "wsnr_db_median": nv_median_db,
            "aggregate_definition": ("corpus = numel-weighted mean of "
                                     "per-tensor weighted_nsse, converted to "
                                     "dB (scalar_incumbents.py's own rule); "
                                     "median = median of per-tensor dB"),
            "scope": sc["scope"], "scale_rule": sc["scale_rule"],
        },
        "plane_pricing": {
            "research_bpw": ("hull tcq_two_tier plane, 9 B/superblock/row = "
                             "0.28125 bpw. RESEARCH: the E2M1 family declares "
                             "exactly one wire scale contract and this is not "
                             "it. Comparable to the published ladder only."),
            "attested_bpw": ("group16_fp8_e4m3_0p5_bpw, the family's declared "
                             "contract and an _scaled_mm operand that cannot "
                             "be compressed away. This is the column the "
                             "NVFP4 comparison uses."),
            "constant_offset_bpw": 0.5 - 0.28125,
            "caveat": ("quality at every rung was RENDERED on the two-tier "
                       "plane and is here PRICED at the attested plane. The "
                       "same-rate tcq_v1 arm (rendered AND priced on the "
                       "attested plane) bounds that substitution at "
                       "-0.25..+0.42 dB per tensor, median -0.14/-0.25/-0.01 "
                       "dB at R=1.5/2.0/2.5 -- see plane_ab below."),
        },
        "plane_ab": plane_ab(pub, names),
        "rungs_research_render": rungs,
        "rungs_attested_render": rungs_attested_render,
        "incomplete_rungs_excluded": {
            "tcq_two_tier": arm_c_incomplete,
            "tcq_v1": arm_d_incomplete,
        },
        "primary_comparison": (
            "rungs_attested_render (arm D / tcq_v1): rendered AND priced on "
            "group16_fp8_e4m3_0p5_bpw, the same plane scalar NVFP4 is priced "
            "on. rungs_research_render (arm C / tcq_two_tier) is the "
            "published ladder's arm and is carried for continuity; its "
            "attested-plane bpw column is a PRICE applied to a render done on "
            "another plane."),
        "crossover_attested_render_median": crossing(
            "wsnr_db_median", "bpw_attested_median", nv_median_db,
            rungs_attested_render),
        "crossover_attested_render_corpus": crossing(
            "wsnr_db_corpus", "bpw_attested_corpus", nv_corpus_db,
            rungs_attested_render),
        "crossover_research_render_attested_price_median": crossing(
            "wsnr_db_median", "bpw_attested_median", nv_median_db),
        "crossover_research_render_research_price_median": crossing(
            "wsnr_db_median", "bpw_research_median", nv_median_db),
        "family_ceiling": {
            "q256": 1016, "body_rate": 1016 / 256,
            "why": ("mathematical_q256_bounds = (256, (256-8)*4 + 8*3) = "
                    "(256, 1016): bypass_rate 4 at up to 248 of 256 positions "
                    "plus shaped_max_rate 3 at the MIN_TRELLIS_STEPS=8 that "
                    "must stay genuinely coded"),
        },
        "scope_limits": [
            "weight-only corpus SSE (W*A16), not served KL and not PPL",
            "RTN render on BOTH sides; production NVFP4 renders under GPTQ+JSO,"
            " so the incumbent here is a LOWER bound on production NVFP4",
            "24 DSv4 routed-expert tensors whose source is MXFP4, so the "
            "content ceiling is ~5.3 bits and high-rate rungs measure less "
            "than they would on a bf16 source",
            "no served A/B, no kernel measurement, no W4A4 activation side",
            "the trellis two-tier scale plane does not exist on the wire",
        ],
        "inputs": {
            "published": str(args.published),
            "highrate": str(args.highrate),
            "highrate_receipt": hr_doc["receipt"],
            "scalar": str(args.scalar),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"NVFP4 incumbent: {nv_corpus_bpw:.4f} bpw  "
          f"{nv_corpus_db:.4f} dB corpus / {nv_median_db:.4f} dB median\n")
    hdr = (f"{'body':>8} {'q256':>5} {'bpw_res':>8} {'bpw_att':>8} "
           f"{'medSNR':>8} {'corpSNR':>8} {'beat':>6} {'beat<=bpw':>10}")
    for label, table in (("ARM C  tcq_two_tier (research render)", rungs),
                         ("ARM D  tcq_v1 (ATTESTED render + price)",
                          rungs_attested_render)):
        print(f"\n{label}")
        print(hdr)
        print("-" * len(hdr))
        for r in table:
            print(f"{r['body_rate']:>8} {r['body_rate_q256']:>5} "
                  f"{r['bpw_research_median']:>8.4f} "
                  f"{r['bpw_attested_median']:>8.4f} "
                  f"{r['wsnr_db_median']:>8.3f} {r['wsnr_db_corpus']:>8.3f} "
                  f"{r['beats_nvfp4_per_tensor']:>4}/24 "
                  f"{r['beats_nvfp4_per_tensor_at_or_below_nvfp4_bpw']:>8}/24")
    print()
    for label, key in (
            ("arm D, median", "crossover_attested_render_median"),
            ("arm D, corpus", "crossover_attested_render_corpus"),
            ("arm C priced attested, median",
             "crossover_research_render_attested_price_median"),
            ("arm C priced research, median",
             "crossover_research_render_research_price_median")):
        c = payload[key]
        print(f"crossover ({label}): {c['bpw']:.4f} bpw -- {c['kind']}")
    print(f"\nwrote {args.out}")
    return 0


def complete_rates(pub, hr, names, lane, rates, low_rates):
    """Keep only corpus-complete rungs and record every missing tensor.

    A high-rate schedule may be unreachable for one tensor because the minimum
    trellis-length guard cannot be rebalanced.  Such a rung is not a 24-tensor
    comparison and must never be silently averaged over the remaining rows.
    """
    available = []
    incomplete = []
    for rate in rates:
        src = pub if rate in low_rates else hr
        key = f"{lane}@{rate}"
        missing = [name for name in names if key not in src[name]["arms"]]
        if missing:
            incomplete.append({
                "body_rate": rate,
                "missing_tensors": missing,
                "reason": "rung is not corpus-complete",
            })
        else:
            available.append(rate)
    return tuple(available), incomplete


def build_rungs(pub, hr, names, numel, total_numel, nv, nv_bpw,
                lane, rates, low_rates):
    """One ladder table for one lane.  For `tcq_v1` the research column is a
    copy of the attested column: arm D has only ever had the declared plane,
    so there is no second price to report and none is invented."""
    rungs = []
    for rate in rates:
        src = pub if rate in low_rates else hr
        rows = {n: src[n]["arms"][f"{lane}@{rate}"] for n in names}
        res_bpw = {n: rows[n]["footprint"]["exact_bpw"] for n in names}
        att_bpw = {n: rows[n]["footprint"].get(
            "production_payload_v1", rows[n]["footprint"])["exact_bpw"]
            for n in names}
        nsse = {n: rows[n]["weighted_nsse"] for n in names}
        beat = [n for n in names if db(nsse[n]) > db(nv[n])]
        beat_fair = [n for n in beat if att_bpw[n] <= nv_bpw[n] + 1e-12]
        rungs.append({
            "lane": lane,
            "body_rate": rate,
            "body_rate_q256": int(round(rate * 256)),
            "bpw_research_corpus": sum(res_bpw[n] * numel[n]
                                       for n in names) / total_numel,
            "bpw_attested_corpus": sum(att_bpw[n] * numel[n]
                                       for n in names) / total_numel,
            "bpw_research_median": statistics.median(res_bpw.values()),
            "bpw_attested_median": statistics.median(att_bpw.values()),
            "wsnr_db_median": statistics.median(db(nsse[n]) for n in names),
            "wsnr_db_corpus": db(sum(nsse[n] * numel[n]
                                     for n in names) / total_numel),
            "beats_nvfp4_per_tensor": len(beat),
            "beats_nvfp4_per_tensor_at_or_below_nvfp4_bpw": len(beat_fair),
            "per_tensor_db": {n: db(nsse[n]) for n in names},
            "source": ("hull_results.menuext-20260829.json" if rate in low_rates
                       else "e2m1_highrate_rows.json"),
        })
    return rungs


def plane_ab(pub, names):
    """tcq_v1 (rendered AND priced on the attested plane) vs tcq_two_tier at
    the same body rate: how much the plane substitution is worth, measured."""
    out = {}
    for rate in (1.5, 2.0, 2.5):
        a = [pub[n]["arms"][f"tcq_v1@{rate}"]["weighted_snr_db"] for n in names]
        b = [pub[n]["arms"][f"tcq_two_tier@{rate}"]["weighted_snr_db"]
             for n in names]
        d = [x - y for x, y in zip(a, b)]
        out[str(rate)] = {"attested_minus_research_db_median":
                          statistics.median(d),
                          "min": min(d), "max": max(d)}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
