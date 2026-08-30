#!/usr/bin/env python3
"""The MEASURED scalar ladder on the E2M1 grid: is the 6.02 dB/bit line a fair
baseline, or is NVFP4-at-4-bits itself off it?

For m in {2,4,8,15} levels (= 1,2,3,4 bits/weight) this renders RTN onto the
weighted-MSE-optimal m-level subset of the 15 distinct E2M1 levels, on the
declared group-16 plane, with the sweep's own importance weights and scorer.
`best_level_subset` is exhaustive and is the routine the trellis uses to pick
its own alphabet, so the scalar ladder and the trellis ladder are chosen the
same way and differ only in the coder.

No trellis encode, no GPU trellis: this is the baseline arm on its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, "/home/rob/dq-runs/trellis-hull-20260828")

import torch

import hull_sweep as H

W, C, P = H.W, H.C, H.P
OUT_BY_CORPUS = {
    "dsv4": Path("/home/rob/dq-runs/trellis-hull-20260828/"
                 "scalar_subgrid_ladder.json"),
    "bf16": Path("/home/rob/dq-runs/trellis-hull-20260828/"
                 "scalar_subgrid_ladder_bf16.json"),
}
# 15 distinct levels is the whole E2M1 grid (16 codes, duplicate zero), i.e.
# 4 bits/weight -- the same alphabet scalar NVFP4 uses.
M_BY_BITS = {1: 2, 2: 4, 3: 8, 4: len(P.E2M1_LEVELS)}


def load_dsv4(name, entries):
    """MXFP4-source DSv4 routed experts: 21-27 distinct values per tensor."""
    packed, raw_scale, importance = W.load_compact(name)
    return W.dequant_mxfp4(packed, raw_scale, torch.device("cpu")), importance


def load_bf16(name, entries):
    """bf16-source Qwen3-4B MLP: ~5100 distinct values per tensor.

    Same reader contract the bf16 W4A4 ladder uses (bf16 on disk, fp32 in
    hand, hashes checked by `bf16_ladder.load_tensor`)."""
    import bf16_ladder as B
    raw, imp = B.load_tensor(entries[name])
    return raw.to(torch.float32), imp.to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("dsv4", "bf16"), default="dsv4")
    ap.add_argument("--v1-only", action="store_true",
                    help="skip the two-tier research plane")
    args = ap.parse_args()
    from prismaquant import format_registry as FR
    if args.corpus == "dsv4":
        entries = {str(e["name"]): e
                   for e in json.loads(H.INPUT_MANIFEST.read_text())["entries"]}
        load = load_dsv4
    else:
        import bf16_ladder as B
        _manifest, _names, entries = B.load_corpus()
        load = load_bf16
    OUT = OUT_BY_CORPUS[args.corpus]
    names = list(entries)
    per = {}
    for index, name in enumerate(names, start=1):
        weight, importance = load(name, entries)
        eff = P.eff_scale_plane(weight)
        H.assert_context_parity(weight, importance, eff)
        # The two-tier RESEARCH plane is only needed to partner arm C, which
        # exists on the DSv4 study.  The bf16 question is about the SCALAR
        # last-bit step on the declared plane, so it is not paid for there.
        planes = [("v1_attested", eff)]
        if not args.v1_only:
            snapped, _, _, _ = H.two_tier_snap_plane(eff)
            planes.append(("two_tier_research", snapped))
        _, _, metric_w, _ = H.context_from_plane(weight, importance, eff)
        energy = C.weighted_sse(weight, torch.zeros_like(weight), metric_w)
        cell = {"shape": list(map(int, weight.shape)),
                "numel": int(weight.numel()), "weighted_energy": energy,
                "planes": {}}
        # One scalar ladder PER PLANE: a trellis arm must be compared with the
        # scalar partner on ITS OWN plane, or the plane difference is scored
        # as a coding gain.
        for plane_name, plane in planes:
            x, pes, _, enc_w = H.context_from_plane(weight, importance, plane)
            bits_out = {}
            for bits, m in M_BY_BITS.items():
                levels, _ = P.best_level_subset(x, enc_w, m)
                recon = P.quantize_to_levels(x, levels, pes)
                sse = C.weighted_sse(weight, recon.to(weight.dtype), metric_w)
                bits_out[str(bits)] = {
                    "n_levels": m, "levels": [float(v) for v in levels],
                    "wsse": sse, "wnsse": sse / energy,
                    "db": 10.0 * math.log10(energy / sse)}
            cell["planes"][plane_name] = {
                "distinct_x_values": int(torch.unique(x).numel()),
                "bits": bits_out}
        cell["bits"] = cell["planes"]["v1_attested"]["bits"]
        cell["distinct_x_values"] = (
            cell["planes"]["v1_attested"]["distinct_x_values"])
        nv = FR.get_format("NVFP4").quantize_dequantize(weight)
        nv_sse = C.weighted_sse(weight, nv.to(weight.dtype), metric_w)
        cell["nvfp4"] = {"wsse": nv_sse, "wnsse": nv_sse / energy,
                         "db": 10.0 * math.log10(energy / nv_sse)}
        per[name] = cell
        print(f"[{index}/{len(names)}] {name}  " + "  ".join(
            f"{b}b={cell['bits'][str(b)]['db']:6.3f}" for b in M_BY_BITS)
            + f"  NVFP4={cell['nvfp4']['db']:6.3f}"
            + f"  distinct_x={cell['distinct_x_values']}", flush=True)

    tot = sum(c["numel"] for c in per.values())
    by_plane = {}
    for plane_name in ([p for p in ("v1_attested", "two_tier_research")
                        if p in per[names[0]]["planes"]]):
        d = {}
        for bits in M_BY_BITS:
            g = sum(per[n]["planes"][plane_name]["bits"][str(bits)]["wnsse"]
                    * per[n]["numel"] for n in names) / tot
            d[str(bits)] = {
                "bits_per_weight": bits, "n_levels": M_BY_BITS[bits],
                "db_corpus": -10 * math.log10(g),
                "db_median": statistics.median(
                    per[n]["planes"][plane_name]["bits"][str(bits)]["db"]
                    for n in names)}
        by_plane[plane_name] = d
    summary = dict(by_plane["v1_attested"])
    g = sum(per[n]["nvfp4"]["wnsse"] * per[n]["numel"] for n in names) / tot
    summary["nvfp4"] = {"bits_per_weight": 4, "db_corpus": -10 * math.log10(g),
                        "db_median": statistics.median(
                            per[n]["nvfp4"]["db"] for n in names)}
    slopes = {f"{a}->{b}": summary[str(b)]["db_corpus"]
              - summary[str(a)]["db_corpus"]
              for a, b in ((1, 2), (2, 3), (3, 4))}
    OUT.write_text(json.dumps({
        "schema": "trellis.scalar_subgrid_ladder.v1",
        "corpus": args.corpus,
        "corpus_note": (
            "dsv4 = MXFP4-source DSv4 routed experts (21-27 distinct source "
            "values, the ~4.7-bit content ceiling); bf16 = bf16-source "
            "Qwen3-4B MLP (~5100 distinct source values). The 3->4 dB/bit "
            "step is the diagnostic: a large step means the 15-level grid is "
            "nearly INTERPOLATING a small discrete source, which is a "
            "property of the source dtype and not of NVFP4."),
        "note": ("RTN onto the exhaustively-optimal m-level subset of the "
                 "E2M1 grid, declared group-16 plane, sweep's own importance "
                 "and scorer. The 4-bit row uses the whole grid and is "
                 "therefore the same alphabet scalar NVFP4 uses."),
        "summary_v1_attested_plane": summary,
        "summary_by_plane": by_plane, "measured_db_per_bit": slopes,
        "ideal_db_per_bit": 6.0206, "per_tensor": per}, indent=1))

    for plane_name in by_plane:
        print(f"\nMEASURED scalar ladder on the E2M1 grid -- {plane_name}")
        print(f"{'bits':>5} {'levels':>7} {'db_corpus':>10} {'db_median':>10}")
        for bits in M_BY_BITS:
            s = by_plane[plane_name][str(bits)]
            print(f"{bits:>5} {s['n_levels']:>7} {s['db_corpus']:>10.3f} "
                  f"{s['db_median']:>10.3f}")
    s = summary["nvfp4"]
    print(f"{'NVFP4':>5} {'16':>7} {s['db_corpus']:>10.3f} "
          f"{s['db_median']:>10.3f}")
    print("\nmeasured dB per bit (corpus): " + "  ".join(
        f"{k}={v:+.3f}" for k, v in slopes.items())
        + "   [ideal scalar = +6.021]")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
