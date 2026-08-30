#!/usr/bin/env python3
"""Does a per-tensor learned CB book change the 4-bit format verdict?

THE QUESTION. Every fp4-CB rung in docs/design/format_menu_2026-08-29.md was
measured with a FIXED LATTICE book. 15 of 18 rungs retire there, and the
verdict's own section 7 names the learned arm as an open hole: a learned book
is strictly better at the same index width, so a retirement measured against
the fixed lattice is measured against CB's weaker arm.

THE ANSWER IS BOUNDED BEFORE IT IS MEASURED, and this script checks the bound
rather than trusting it. `cb_book_bits` charges 4 bits/element because every
codeword element is a legal e2m1 level (tcq_pilot.py:461-465, LEVEL_BITS=4).
PRODUCTION ships FP16 books = 16 bits/element. So the arm as run is
UNDER-charged 4x, and the honest reading needs both brackets:

  as_run   -- the 4 bits/element charge the driver recorded
  fp16     -- 4x that, what production would actually pay

A learned rung only beats a fixed-lattice one if it wins by more than its own
book costs. Reporting only `as_run` would hand the learned arm a discount it
does not get in an artifact -- the mirror of the plane-grant error recorded in
removing_a_confound_can_install_its_mirror.md.

WHAT IT CANNOT DECIDE. 24 DeepSeek-V4-Flash MoE expert tensors (corrected
2026-08-29 from "GLM"; `stage6_inputs_manifest.json` `source_shard_sha256` names
the `dsv4-flash-0731` shards), w1/w2/w3 over 8 layers. No
attention weight and no dense Linear is in this corpus.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/home/rob/dq-runs/trellis-hull-20260828/hull_results.learnedcb-20260829.json")
d = json.loads(P.read_text())
pt = d["per_tensor"]

def rows(arm_prefix):
    out = {}
    for tname, t in pt.items():
        for aname, a in t["arms"].items():
            if not aname.startswith(arm_prefix + "@"):
                continue
            k = aname.split("@", 1)[1]
            fp = a["footprint"]
            out.setdefault(k, []).append({
                "tensor": tname,
                "bpw": fp["exact_bpw"],
                "book_bpw": fp.get("codebook_side_bpw", 0.0),
                "db": a["weighted_snr_db"],
            })
    return out

fixed = rows("cb_two_tier")
fixed = {k: v for k, v in fixed.items()}          # cb_two_tier@K
learned = rows("cb_two_tier_learned")

common = sorted(set(fixed) & set(learned), key=lambda s: int(s))
print(f"corpus: {len(pt)} tensors | rungs compared: {len(common)}\n")
print(f"{'K':>3} {'fixed dB':>9} {'learn dB':>9} {'ΔdB':>7} "
      f"{'bpw(as-run)':>12} {'Δbpw run':>9} {'Δbpw fp16':>10} "
      f"{'verdict as-run':>15} {'verdict fp16':>14}")

flips = {"as_run": 0, "fp16": 0}
for k in common:
    f = {r["tensor"]: r for r in fixed[k]}
    l = {r["tensor"]: r for r in learned[k]}
    ts = sorted(set(f) & set(l))
    if not ts:
        continue
    d_db = statistics.mean(l[t]["db"] - f[t]["db"] for t in ts)
    fb = statistics.mean(f[t]["bpw"] for t in ts)
    lb = statistics.mean(l[t]["bpw"] for t in ts)
    book = statistics.mean(l[t]["book_bpw"] for t in ts)
    # fp16 bracket: the book plane costs 4x what the driver charged.
    lb16 = lb + 3.0 * book
    # A rung "wins" only if it improves dB at no worse bpw. Both are averages
    # over the same tensor set, so the comparison is paired.
    win_run = d_db > 0 and lb <= fb + 1e-12
    win_16 = d_db > 0 and lb16 <= fb + 1e-12
    flips["as_run"] += bool(win_run)
    flips["fp16"] += bool(win_16)
    print(f"{k:>3} {statistics.mean(f[t]['db'] for t in ts):9.3f} "
          f"{statistics.mean(l[t]['db'] for t in ts):9.3f} {d_db:+7.3f} "
          f"{lb:12.5f} {lb-fb:+9.5f} {lb16-fb:+10.5f} "
          f"{'learned' if win_run else 'fixed':>15} "
          f"{'learned' if win_16 else 'fixed':>14}")

print(f"\nrungs where learned dominates at no extra bytes: "
      f"as-run {flips['as_run']}/{len(common)}, fp16 {flips['fp16']}/{len(common)}")
print("A learned book is never FREE: it always costs bytes, so it can only\n"
      "dominate by being better at the SAME bpw, which the exact-byte\n"
      "accounting makes impossible above zero book size. What matters for the\n"
      "verdict is whether learned-at-K beats fixed-at-K+1 -- checked next.")

print("\n--- learned@K vs fixed@K+1 (the byte-matched comparison) ---")
print(f"{'K':>3} {'learn@K dB':>11} {'fixed@K+1 dB':>13} {'ΔdB':>7} "
      f"{'bpw learn':>10} {'bpw fixed':>10} {'bpw learn fp16':>15} {'verdict':>22}")
beats_run = beats_16 = total = 0
for k in common:
    kn = str(int(k) + 1)
    if kn not in fixed:
        continue
    l = {r["tensor"]: r for r in learned[k]}
    f = {r["tensor"]: r for r in fixed[kn]}
    ts = sorted(set(f) & set(l))
    if not ts:
        continue
    total += 1
    ldb = statistics.mean(l[t]["db"] for t in ts)
    fdb = statistics.mean(f[t]["db"] for t in ts)
    lb = statistics.mean(l[t]["bpw"] for t in ts)
    fb = statistics.mean(f[t]["bpw"] for t in ts)
    book = statistics.mean(l[t]["book_bpw"] for t in ts)
    lb16 = lb + 3.0 * book
    ok_run = ldb > fdb and lb <= fb
    ok_16 = ldb > fdb and lb16 <= fb
    beats_run += ok_run
    beats_16 += ok_16
    v = ("learned BOTH" if ok_16 else
         "learned as-run only" if ok_run else "fixed")
    print(f"{k:>3} {ldb:11.3f} {fdb:13.3f} {ldb-fdb:+7.3f} "
          f"{lb:10.5f} {fb:10.5f} {lb16:15.5f} {v:>22}")
print(f"\nlearned@K beats fixed@K+1 at no more bytes: "
      f"as-run {beats_run}/{total}, fp16 {beats_16}/{total}")
print("\nA verdict counts only where BOTH brackets agree "
      "(removing_a_confound_can_install_its_mirror).")
