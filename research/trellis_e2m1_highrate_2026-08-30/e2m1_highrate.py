#!/usr/bin/env python3
"""Measure the UNMEASURED high-rate band of the E2M1 (4-bit) trellis ladder.

The sealed hull sweep stopped at body rate 2.25; the 2026-08-29 menu extension
carried it to 3.0.  `TCQ_E2M1_R256`'s own mathematical bound is
``mathematical_q256_bounds == (256, 1016)`` -> body rate 3.96875, because a
block may spend `bypass_rate` (4) at up to 248 of 256 positions so long as
`MIN_TRELLIS_STEPS` (8) remain genuinely coded.  Everything from 3.0 to 3.96875
was never measured, and that is exactly the band where a W4A4 trellis would have
to beat scalar NVFP4 (4.0 body + 0.5 plane = 4.5 bpw).

WHY A SIBLING DRIVER AND NOT `hull_sweep.py --extra-rates`:
`hull_sweep`'s checkpoint identity contains the whole `plan`, so adding rates
invalidates every checkpoint and re-funds all 25 CB rungs on 24 tensors (the
08-29 menu extension cost 3h15m for exactly that reason).  `cb_two_tier` encode
is ~80% of ladder cost and CB is already retired 24/24 across the overlapping
band, so re-buying it answers nothing.  This driver imports `hull_sweep` and
calls its OWN functions -- same corpus, same hashes, same plane snap, same
context, same alphabets, same schedule builder, same footprint accountant, same
scorer -- so the rows it writes are the rows `hull_sweep` would have written.

SELF-CHECK, NOT SELF-CERTIFICATION: rate 3.0 is re-measured here and compared
against the 08-29 row.  If this driver does not reproduce that row, its new
rows are not comparable to the published ladder and it refuses.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, "/home/rob/dq-runs/trellis-hull-20260828")

import numpy as np
import torch

import hull_sweep as H

W, C, P, S4, TF = H.W, H.C, H.P, H.S4, H.TF

# --- corpus configuration --------------------------------------------------
# dsv4 = MXFP4-source DSv4 routed experts (21-27 distinct source values).
# bf16 = bf16-source Qwen3-4B DENSE MLP (~5100 distinct source values).  The
# bf16 corpus fixes the SOURCE-DTYPE confound and NOTHING else: it is dense
# Qwen3-4B, not MoE experts and not GLM.  Label it that way everywhere.
CORPUS_LABEL = {
    "dsv4": "DSv4 routed experts, MXFP4 source",
    "bf16": "Qwen3-4B DENSE MLP, bf16 source (NOT MoE, NOT GLM)",
}
CONTROL_RATE = 3.0
NEW_RATES = (3.25, 3.5, 3.75, 3.9375, 3.96875)
# 3.96875 is the family's mathematical ceiling but is NOT universally
# reachable: at 1016 q256 every 256-block must keep exactly MIN_TRELLIS_STEPS=8
# coded positions, leaving the global reverse-water-fill zero slack to move a
# bit between blocks, and on at least one corpus tensor the rebalance is
# infeasible.  A refusal is recorded per (tensor, rate), never crashed on, and
# 3.9375 is the highest rung reachable on all 24.
MID_RATES = (1.5, 2.0, 2.5)
PUBLISHED = Path("/home/rob/dq-runs/trellis-hull-20260828/"
                 "hull_results.menuext-20260829.json")
BF16_PUBLISHED = Path("/home/rob/dq-runs/trellis-bf16-20260829/"
                      "bf16_w4a4_results.json")
# On bf16 the campaign question is the coding gain at matched rate, so the
# rungs are the integer rates the coordinator named plus 2.5 to read the TREND
# (the gain RISES with rate on DSv4: 1.34 -> 1.88 -> 2.18, and whether that
# survives on a continuous-density source is what the estimate assumed).
BF16_RATES = (1.0, 2.0, 2.5, 3.0)
# Deterministic given the same encoder/inputs/device.  A torch/triton skew
# flips DISCRETE encode decisions and shows at ~1e-3, which no tolerance
# absorbs -- it is diagnosed, not tolerated.  Same bar hull_sweep uses.
CONTROL_RTOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("dsv4", "bf16"), default="dsv4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-control-drift", action="store_true",
                    help="record a failed 3.0 control instead of refusing")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA required (principle 7)")
    device = torch.device("cuda")

    if args.corpus == "dsv4":
        published = json.loads(PUBLISHED.read_text())["per_tensor"]
        entries = {str(e["name"]): e
                   for e in json.loads(
                       H.INPUT_MANIFEST.read_text())["entries"]}
        names = list(entries)
        rate_plan = (*MID_RATES, CONTROL_RATE, *NEW_RATES)
        control_keys = (f"tcq_two_tier@{CONTROL_RATE}",
                        "tcq_two_tier@2.5", "tcq_v1@2.5")
    else:
        import bf16_ladder as B
        _m, _n, entries = B.load_corpus()
        names = list(entries)
        rate_plan = BF16_RATES
        # The bf16 W4A4 ladder publishes arm C at 1.0-2.25 on the two-tier
        # plane.  2.0 is in BOTH it and our plan, so it is the control.  Arm D
        # has no published bf16 row and is therefore uncontrolled here; that is
        # stated, not hidden.
        published = (json.loads(BF16_PUBLISHED.read_text())["cells"]
                     if BF16_PUBLISHED.exists() else {})
        control_keys = ("tcq_two_tier@2.0",)
    if args.limit:
        names = names[:args.limit]

    env = H.current_env()
    receipt = {
        "schema": "trellis.e2m1_highrate.v1",
        "started_at_unix_s": time.time(),
        "question": ("does the E2M1 trellis, above body rate 2.25 and up to "
                     "its 3.96875 mathematical ceiling, beat scalar NVFP4 "
                     "(4.5 bpw) at equal or smaller bpw on the same corpus"),
        "control_rungs": ["tcq_two_tier@3.0", "tcq_two_tier@2.5",
                          "tcq_v1@2.5"],
        "arms_measured": ["tcq_two_tier (research 0.28125 plane)",
                          "tcq_v1 (ATTESTED group16_fp8_e4m3_0p5_bpw plane, "
                          "rendered AND priced there)"],
        "control_source": str(PUBLISHED),
        "control_rtol": CONTROL_RTOL,
        "corpus": args.corpus,
        "corpus_label": CORPUS_LABEL[args.corpus],
        "rate_plan": list(rate_plan),
        "new_rates": list(NEW_RATES),
        "mathematical_q256_bounds": list(
            TF.FAMILIES[TF.E2M1_FAMILY].mathematical_q256_bounds),
        "arms": ["tcq_two_tier (arm C): trellis rendered on the snapped RESEARCH two-tier plane, priced at both planes", "tcq_v1 (arm D): trellis rendered AND priced on the family's DECLARED group16_fp8_e4m3_0p5_bpw plane -- the arm the NVFP4 comparison needs, since NVFP4 is priced on exactly that plane"],
        "pricing": ("every row carries BOTH prices: exact_bpw at the RESEARCH "
                    "0.28125 bpw two-tier plane (comparable to the published "
                    "ladder) and production_payload_v1.exact_bpw at the "
                    "family's ATTESTED group16_fp8_e4m3_0p5_bpw plane "
                    "(comparable to NVFP4 and to production)"),
        "environment": env,
    }

    out: dict[str, dict] = {}
    for index, name in enumerate(names, start=1):
        entry = entries[name]
        if args.corpus == "dsv4":
            packed, raw_scale, importance = W.load_compact(name)
            for label, value, key in (
                    ("weight", packed, "source_weight_sha256"),
                    ("scale", raw_scale, "source_scale_sha256"),
                    ("importance", importance, "importance_sha256")):
                if H.tensor_sha256(value) != entry[key]:
                    raise SystemExit(f"{name}: compact {label} hash mismatch")
            weight = W.dequant_mxfp4(packed, raw_scale, device)
            importance = importance.to(device, torch.float32)
        else:
            import bf16_ladder as B
            raw, imp = B.load_tensor(entry)   # hashes checked inside
            weight = raw.to(device, torch.float32)
            importance = imp.to(device, torch.float32)
        rows, columns = map(int, weight.shape)
        eff = P.eff_scale_plane(weight)
        H.assert_context_parity(weight, importance, eff)
        _, _, metric_w, _ = H.context_from_plane(weight, importance, eff)
        weighted_energy = C.weighted_sse(weight, torch.zeros_like(weight),
                                         metric_w)
        plain_energy = C.plain_sse(weight, torch.zeros_like(weight))

        # Arm D context: rendered AND priced on the family's DECLARED
        # group16_fp8_e4m3_0p5_bpw plane.  This is the arm that answers the
        # NVFP4 question without any plane substitution -- NVFP4 is priced on
        # exactly this plane -- so it is measured here alongside arm C rather
        # than left to a caveat.
        x_v1, pes_v1, _, enc_v1 = H.context_from_plane(weight, importance, eff)
        codes_v1, _, alpha_v1 = W.alphabets(x_v1, enc_v1)
        colw_v1 = S4.column_weight(enc_v1)

        snapped, _, _, snap_stats = H.two_tier_snap_plane(eff)
        x_tt, pes_tt, _, enc_tt = H.context_from_plane(weight, importance,
                                                       snapped)
        codes_tt, _, alpha_tt = W.alphabets(x_tt, enc_tt)
        colw_tt = S4.column_weight(enc_tt)

        cell = {"shape": [rows, columns], "numel": rows * columns,
                "weighted_energy": weighted_energy,
                "plain_energy": plain_energy,
                "two_tier_plane_sha256": H.tensor_sha256(snapped),
                "arms": {}}

        # scalar NVFP4 RTN on this tensor, for the per-subset contrast: a
        # bypass column IS this, so "trellis on bypass columns" and "NVFP4 on
        # bypass columns" must agree up to the scale plane, while a genuine
        # coding gain can only live on the SHAPED columns.
        from prismaquant import format_registry as FR
        nvfp4_recon = FR.get_format("NVFP4").quantize_dequantize(weight)

        def scalar_subgrid(x_cols, enc_cols, pes_cols, w_cols, m_levels):
            """RTN onto the weighted-MSE-optimal m-level subset of the E2M1
            grid -- the honest same-rate, same-grid, same-plane partner for a
            shaped trellis column.

            `P.best_level_subset` is EXHAUSTIVE (C(15,2)=105, C(15,4)=1365,
            C(15,8)=6435) and is the SAME routine the trellis uses to choose
            its OWN alphabet, so the two sides differ in the coder and nothing
            else.  A rate-R trellis step gets 2^(R+1) levels and picks among
            2^R of them per state; the scalar partner gets a free choice among
            2^R levels.  Equal bits, equal grid, equal plane.
            """
            levels, _ = P.best_level_subset(x_cols, enc_cols, m_levels)
            recon_s = P.quantize_to_levels(x_cols, levels, pes_cols)
            return levels, recon_s

        # The "shared" scope fits the subset on the WHOLE tensor, so it is
        # identical across every column class and every rung of one lane.
        # Memoized: on the bf16 corpus (25M elements/tensor) recomputing it
        # per class per rung is most of the run.
        shared_memo: dict = {}

        def subset_split(recon, rate, colw_lane, x_lane, enc_lane, pes_lane,
                         lane_tag):
            """Per-schedule-rate weighted SNR, trellis vs scalar NVFP4, over
            the SAME columns.  Answers whether the shaped columns carry gain
            the bypass columns cannot."""
            try:
                sched, _ = S4.build_schedules(rate, columns, colw_lane,
                                              include_variants=False)
            except AssertionError:
                return None
            sched = torch.as_tensor(np.asarray(sched["rwf"]),
                                    device=weight.device)
            out = {}
            for r in (1, 2, 3, 4):
                mask = sched == r
                n = int(mask.sum())
                if not n:
                    continue
                cols_idx = mask.nonzero(as_tuple=True)[0]
                w_s = weight.index_select(1, cols_idx)
                m_s = metric_w.index_select(1, cols_idx)
                e = C.weighted_sse(w_s, torch.zeros_like(w_s), m_s)
                t = C.weighted_sse(w_s, recon.index_select(1, cols_idx)
                                   .to(weight.dtype), m_s)
                v = C.weighted_sse(w_s, nvfp4_recon.index_select(1, cols_idx)
                                   .to(weight.dtype), m_s)
                row = {
                    "columns": n, "energy": e,
                    "trellis_wsse": t, "nvfp4_wsse": v,
                    "trellis_db": 10.0 * math.log10(e / t),
                    "nvfp4_db": 10.0 * math.log10(e / v),
                    "trellis_minus_nvfp4_db": 10.0 * math.log10(v / t),
                    "bits_per_weight_here": r,
                    "nvfp4_bits_per_weight": 4,
                }
                # --- the same-grid scalar control, the real coding gain ----
                x_s = x_lane.index_select(1, cols_idx)
                enc_s = enc_lane.index_select(1, cols_idx)
                pes_s = pes_lane.index_select(1, cols_idx)
                # rate 4 is bypass: no alphabet at all, the whole grid.
                m_levels = (1 << r) if r < 4 else len(P.E2M1_LEVELS)
                for scope in ("oracle", "shared"):
                    if scope == "oracle":
                        levels, _ = scalar_subgrid(
                            x_s, enc_s, pes_s, w_s, m_levels)
                    else:
                        key = (lane_tag, m_levels)
                        if key not in shared_memo:
                            shared_memo[key] = P.best_level_subset(
                                x_lane, enc_lane, m_levels)[0]
                        levels = shared_memo[key]
                    recon_s = P.quantize_to_levels(x_s, levels, pes_s)
                    q = C.weighted_sse(w_s, recon_s.to(weight.dtype), m_s)
                    row[f"scalar_subgrid_{scope}"] = {
                        "levels": [float(v) for v in levels],
                        "n_levels": m_levels,
                        "wsse": q,
                        "db": 10.0 * math.log10(e / q),
                        "coding_gain_db": 10.0 * math.log10(q / t),
                        "subset_fit_scope": (
                            "these columns only (ORACLE: a tougher baseline "
                            "than the trellis gets, whose alphabet is fit "
                            "tensor-wide -- so a positive gain here is "
                            "conservative)" if scope == "oracle" else
                            "whole tensor (the SAME fitting scope the trellis "
                            "alphabet gets)"),
                    }
                out[str(r)] = row
            return out

        def emit(key, arm, rung, recon, seconds, footprint, extra=None,
                 _cell=cell):
            row = W.metric_row(weight, recon, metric_w, weighted_energy,
                               plain_energy, seconds, footprint)
            row.update({"arm": arm, "rung": rung})
            if extra:
                row.update(dict(extra))
            row["subset_split"] = subset_split(
                recon, rung,
                *( (colw_v1, x_v1, enc_v1, pes_v1, "v1") if arm == "tcq_v1"
                   else (colw_tt, x_tt, enc_tt, pes_tt, "two_tier") ))
            _cell["arms"][key] = row

        unreachable = []
        for lane, plane_args, coding, colw in (
                ("tcq_two_tier", (x_tt, enc_tt, pes_tt, codes_tt),
                 H.SCALE_CODING_TWO_TIER, colw_tt),
                ("tcq_v1", (x_v1, enc_v1, pes_v1, codes_v1),
                 H.SCALE_CODING_V1, colw_v1)):
            x, enc, pes, codes = plane_args
            for rate in rate_plan:
                try:
                    H.emit_trellis(name, lane, rate, weight, x, enc, pes,
                                   codes, colw, coding, "triton", emit)
                except AssertionError as exc:
                    unreachable.append({"lane": lane, "rate": rate,
                                        "reason": str(exc)})
                    print(f"      {lane}@{rate}: UNREACHABLE ({exc})",
                          flush=True)
        cell["unreachable_rungs"] = unreachable

        # --- the control -------------------------------------------------
        checks = {}
        footprint_equal = True
        for control_key in control_keys:
            if (name not in published
                    or control_key not in published[name].get("arms", {})
                    or control_key not in cell["arms"]):
                checks[f"{control_key}.MISSING"] = {
                    "mine": None, "published": None, "rel": 0.0}
                continue
            mine = cell["arms"][control_key]
            theirs = published[name]["arms"][control_key]
            for field in ("weighted_sse", "weighted_nsse", "weighted_snr_db",
                          "plain_sse", "plain_snr_db"):
                a, b = float(mine[field]), float(theirs[field])
                checks[f"{control_key}.{field}"] = {
                    "mine": a, "published": b,
                    "rel": abs(a - b) / max(abs(b), 1e-300)}
            footprint_equal = footprint_equal and (
                mine["footprint"]["total_bytes"]
                == theirs["footprint"]["total_bytes"]
                and mine["footprint"]["body_rate_q256"]
                == theirs["footprint"]["body_rate_q256"])
        measured = [c["rel"] for c in checks.values()
                    if c["mine"] is not None]
        worst = max(measured) if measured else float("nan")
        if not measured:
            footprint_equal = False
        status = ("pass" if (measured and worst <= CONTROL_RTOL
                             and footprint_equal)
                  else ("uncontrolled" if not measured else "fail"))
        cell["control"] = {"status": status, "worst_relative": worst,
                           "footprint_equal": footprint_equal,
                           "checks": checks}
        print(f"[{index}/{len(names)}] {name}: control {status.upper()} "
              f"(worst rel {worst:.3e})", flush=True)
        for rate in rate_plan:
            c_row = cell["arms"].get(f"tcq_two_tier@{rate}")
            d_row = cell["arms"].get(f"tcq_v1@{rate}")
            if c_row is None or d_row is None:
                print(f"      R={rate:<8} (unreachable on one or both arms)",
                      flush=True)
                continue
            print(f"      R={rate:<8} armC {c_row['weighted_snr_db']:7.3f} dB "
                  f"@{c_row['footprint']['exact_bpw']:.4f} research / "
                  f"{c_row['footprint']['production_payload_v1']['exact_bpw']:.4f} "
                  f"attested | armD {d_row['weighted_snr_db']:7.3f} dB "
                  f"@{d_row['footprint']['exact_bpw']:.4f} attested",
                  flush=True)
        if status == "fail" and not args.allow_control_drift:
            raise SystemExit(
                f"FATAL: {name}: a control rung does not reproduce the "
                f"published row (worst relative {worst:.3e}, footprint_equal "
                f"{footprint_equal}). These rows are NOT comparable to the "
                f"published ladder; re-run under the pinned environment "
                f"(hull_sweep.py --print-container-command).")
        out[name] = cell
        args.out.write_text(json.dumps(
            {"receipt": {**receipt, "partial": True,
                         "tensors_done": len(out)},
             "per_tensor": out}, indent=1))

    receipt["completed_at_unix_s"] = time.time()
    receipt["tensors_done"] = len(out)
    receipt["status"] = "ok"
    receipt["control_verdict"] = {
        n: c["control"]["status"] for n, c in out.items()}
    args.out.write_text(json.dumps(
        {"receipt": {**receipt, "partial": False}, "per_tensor": out},
        indent=1))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
