#!/usr/bin/env python3
"""The coding-gain table: trellis minus best-subset scalar at MATCHED rate,
restricted to the columns actually scheduled at that rate.

One derivation, both corpora, so the DSv4 and bf16 columns sit in one table
with nothing re-derived between them:

  dsv4 = DSv4 routed experts, MXFP4 source (21-27 distinct source values)
  bf16 = Qwen3-4B DENSE MLP, bf16 source (~5100 distinct source values)
         -- this corpus fixes the SOURCE-DTYPE confound and nothing else.
         It is dense Qwen3-4B: NOT MoE experts, NOT GLM.

The scalar partner is RTN onto the exhaustively-optimal 2^r-level subset of the
E2M1 grid, on the arm's OWN plane, with the same importance and the same
scorer.  Two fitting scopes are reported: `oracle` fits the subset on the rate-r
columns themselves (a TOUGHER baseline than the trellis gets, whose alphabet is
fit tensor-wide, so a positive oracle gain is conservative) and `shared` fits it
tensor-wide (the trellis alphabet's own scope).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path("/home/rob/dq-runs/trellis-hull-20260828")
LABEL = {"dsv4": "DSv4 routed experts, MXFP4 source",
         "bf16": "Qwen3-4B DENSE MLP, bf16 source (NOT MoE, NOT GLM)",
         "glm": "GLM-5.3-Flash expert 0 + dense MLP, bf16 source"}


def _summarize(per, names):
    rows = []
    for lane in ("tcq_two_tier", "tcq_v1"):
        keys = sorted({k for n in names for k in per[n]["arms"]
                       if k.startswith(f"{lane}@")},
                      key=lambda k: float(k.split("@")[1]))
        for key in keys:
            rate = float(key.split("@")[1])
            for r in (1, 2, 3, 4):
                cells = [per[n]["arms"][key]["subset_split"][str(r)]
                         for n in names
                         if key in per[n]["arms"]
                         and per[n]["arms"][key].get("subset_split")
                         and str(r) in per[n]["arms"][key]["subset_split"]]
                if len(cells) < len(names) / 2:
                    continue
                entry = {
                    "lane": lane, "body_rate": rate, "column_class": r,
                    "matched": abs(rate - r) < 1e-9,
                    "tensors": len(cells),
                    "columns_median": statistics.median(
                        c["columns"] for c in cells),
                    "trellis_db_median": statistics.median(
                        c["trellis_db"] for c in cells),
                    "nvfp4_db_median": statistics.median(
                        c["nvfp4_db"] for c in cells),
                }
                for scope in ("oracle", "shared"):
                    gain = [c[f"scalar_subgrid_{scope}"]["coding_gain_db"]
                            for c in cells]
                    entry[f"scalar_{scope}_db_median"] = statistics.median(
                        c[f"scalar_subgrid_{scope}"]["db"] for c in cells)
                    entry[f"coding_gain_{scope}_db_median"] = statistics.median(gain)
                    entry[f"coding_gain_{scope}_db_min"] = min(gain)
                    entry[f"coding_gain_{scope}_db_max"] = max(gain)
                    entry[f"coding_gain_{scope}_positive"] = sum(v > 0 for v in gain)
                rows.append(entry)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, action="append", required=True,
                    help="e2m1_highrate rows file (repeatable)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "coding_gain_table.json")
    args = ap.parse_args()

    payload = {"schema": "trellis.coding_gain_table.v2",
               "definition": __doc__, "corpora": {}}
    for path in args.rows:
        doc = json.loads(path.read_text())
        corpus = doc["receipt"].get("corpus", "dsv4")
        per = doc["per_tensor"]
        names = list(per)
        populations = sorted({str(per[name].get("population", corpus))
                              for name in names})
        by_population = {}
        for population in populations:
            population_names = [
                name for name in names
                if str(per[name].get("population", corpus)) == population
            ]
            by_population[population] = {
                "n_tensors": len(population_names),
                "rows": _summarize(per, population_names),
            }
        payload["corpora"][corpus] = {
            "label": LABEL[corpus], "source": str(path),
            "n_tensors": len(names),
            "aggregation_contract": (
                "population-separated; no pooled median" if corpus == "glm"
                else "single population"
            ),
            "populations": by_population,
        }

    args.out.write_text(json.dumps(payload, indent=1))

    for corpus, blob in payload["corpora"].items():
        print(f"\n=== {corpus}: {blob['label']}  ({blob['n_tensors']} tensors)")
        for population, population_blob in blob["populations"].items():
            print(f"--- population={population} "
                  f"({population_blob['n_tensors']} tensors)")
            print("MATCHED-RATE coding gain "
                  "(columns scheduled at rate r, trellis minus best 2^r subset)")
            hdr = (f"{'lane':>13} {'body':>6} {'r':>2} {'cols':>7} "
                   f"{'trellis':>8} {'scalar':>8} {'GAIN':>7} {'pos':>6}")
            print(hdr); print("-" * len(hdr))
            for e in population_blob["rows"]:
                if not e["matched"]:
                    continue
                print(f"{e['lane']:>13} {e['body_rate']:>6} {e['column_class']:>2} "
                      f"{e['columns_median']:>7.0f} "
                      f"{e['trellis_db_median']:>8.3f} "
                      f"{e['scalar_oracle_db_median']:>8.3f} "
                      f"{e['coding_gain_oracle_db_median']:>+7.3f} "
                      f"{e['coding_gain_oracle_positive']:>3}/{e['tensors']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
