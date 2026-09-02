#!/usr/bin/env python3
"""Dense-tier A/B: vanilla NVFP4 (4.5 bpw) vs FP8-CB K36 (4.5 bpw).

Answers the open menu question (2026-07-20): would native NVFP4 have beaten
FP8-CB K36 on the dense/attention/shared tier at matched bits? Measures BOTH
formats through ONE code path -- format_registry RTN qdq -> weight-MSE
(unweighted AND activation-column-weighted) x probe h_trace -- so the
comparison is internally consistent with the allocation's local cost
convention. Reads source weights through the same profile-aware BF16-value
decoder as CB cost/export, so native FP8 and declared MXFP4 checkpoints are
compared after their serialized scales are applied; GPU-first.

Model-general since 2026-07-30 (format_choice_4p5.md Stage 0): the target list
falls back to the probe's own Linear inventory when the work dir has no
exported CB config, the source-tensor lookup is namespace-robust
(``model.language_model.layers.*`` vs ``model.layers.*``), h_trace is read out
of ``probe['stats'][name]['h_trace']``, and the act-column weighting is
optional (a work dir with no ``cb_col_weights.pkl`` still reports the
unweighted and h_trace-weighted columns).

  ab_nvfp4_vs_k36_dense.py [--work /home/rob/dq-runs/prod-hy3-nvfp4cb-2p9]
                           [--source /home/rob/dq-runs/hy3-prod/source]
                           [--exclude .experts.,.layers.80.]
                           [--limit N]   # first N dense units (debug)

Output: per-role win rates + cost-ratio geomeans + a per-unit CSV.
"""
import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch

# Resolve the checkout containing this script.  A hard-coded developer clone
# silently imported a stale branch when this tool ran from a clean worktree.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from prismaquant.format_registry import REGISTRY, canonical_format_name  # noqa: E402
from prismaquant import nvfp4_cb_formats as cb           # noqa: E402
from prismaquant.export_nvfp4_cb_streaming import _LazySkeleton  # noqa: E402
from prismaquant.model_profiles import detect_profile  # noqa: E402


def load_st_headers(src):
    hdrs = {}
    for st in sorted(glob.glob(f"{src}/*.safetensors")):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            hdrs[k] = (st, v)
    return hdrs


def load_source_weight(skeleton, name, device):
    """Load one checkpoint weight under the canonical producer value contract.

    ``_LazySkeleton.dequant_weight`` applies profile-resolved block-FP8 or
    MXFP4 scales and rounds through the BF16 model-load value.  Calling it here
    keeps this screen bit-comparable with CB cost/export and hard-fails on an
    unscaled float8 tensor instead of treating storage codes as values.
    """
    return skeleton.dequant_weight(name).to(device, torch.bfloat16)


# --- namespace-robust source lookup -----------------------------------------
# Probe/allocator names are the HF-module names with the multimodal wrapper
# collapsed (``model.layers.7...``); the checkpoint keeps the wrapper
# (``model.language_model.layers.7...``). Strip wrapper segments that sit
# between a leading ``model`` and the first structural segment, and index the
# checkpoint by the collapsed name. Idempotent for already-collapsed names,
# so single-namespace models (Hy3) are unaffected.
_WRAPPER_SEGMENTS = ("language_model", "text_model", "transformer", "model",
                     "decoder")


def collapse_namespace(name: str) -> str:
    parts = name.split(".")
    if not parts or parts[0] != "model":
        return name
    i = 1
    while i < len(parts) and parts[i] in _WRAPPER_SEGMENTS:
        i += 1
    return ".".join(["model"] + parts[i:])


def build_source_index(hdrs, profile=None):
    idx = {}
    for k in hdrs:
        idx.setdefault(collapse_namespace(k), k)
        if profile is not None:
            live = profile.checkpoint_to_live_name(k)
            if live is not None:
                idx.setdefault(str(live), k)
    return idx


def resolve_source_key(src_idx, hdrs, target):
    wname = target + ".weight"
    if wname in hdrs:
        return wname
    return src_idx.get(collapse_namespace(wname))


# --- target inventory --------------------------------------------------------
def targets_from_export(work, exclude):
    """Shipped CB config groups (original Hy3 path): gives the shipped rung."""
    path = f"{work}/exported_nvfp4_cb/quant_config.json"
    if not os.path.exists(path):
        return None
    cfg = json.load(open(path))
    targets = {}
    for g in cfg["config_groups"].values():
        s = g.get("scheme")
        if not s:
            continue
        rung = f"{'NVFP4_CB' if s['grid'] == 'fp4' else 'FP8_CB'}_K{s['k']}"
        for t in g["targets"]:
            if any(x in t for x in exclude):
                continue
            targets[t] = rung
    return targets


def targets_from_probe(probe, exclude):
    """Fallback: the probe's own Linear inventory (no shipped rung known)."""
    stats = probe.get("stats", probe) if isinstance(probe, dict) else probe
    return {t: "-" for t in stats
            if not any(x in t for x in exclude)}


def h_trace_of(probe, name):
    """h_trace lives in probe['stats'][name]['h_trace'] on every production
    probe; tolerate a bare {name: value} mapping too."""
    if not isinstance(probe, dict):
        return 1.0
    stats = probe.get("stats")
    if isinstance(stats, dict) and name in stats:
        e = stats[name]
        if isinstance(e, dict):
            return float(e.get("h_trace", e.get("h_trace_raw", 1.0)) or 1.0)
        return float(e)
    ht = probe.get("h_trace")
    if isinstance(ht, dict) and name in ht:
        return float(ht[name])
    return 1.0


def unit_role(t):
    if ".experts." in t:
        return "expert"
    if "shared_mlp" in t or "shared_experts" in t:
        return "shared"
    return "dense/attn"


def unit_leaf(t):
    parts = t.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", "--work-dir", dest="work",
                    default="/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9")
    ap.add_argument("--source", "--model-path", dest="source",
                    default="/home/rob/dq-runs/hy3-prod/source")
    ap.add_argument("--exclude", default=".experts.,.layers.80.",
                    help="comma-separated substrings to drop from the target "
                         "inventory")
    ap.add_argument("--col-weights", default=None,
                    help="imatrix column weights pkl "
                         "(default <work>/artifacts/cb_col_weights.pkl; "
                         "act-weighted columns are omitted when absent)")
    ap.add_argument("--fmt-a", default="NVFP4")
    ap.add_argument("--fmt-b", default="FP8_CB_K36")
    ap.add_argument("--out", default=None, help="per-unit CSV path")
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--cb-imatrix-fit", action="store_true",
                    help="fit the CB codebook with the imatrix column weights "
                         "(production-faithful render); default is the "
                         "unweighted fit the original Hy3 screen used")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "GPU-first: refusing CPU run"
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    exclude = [x for x in args.exclude.split(",") if x]

    with open(f"{args.work}/artifacts/probe.pkl", "rb") as f:
        probe = pickle.load(f)

    targets = targets_from_export(args.work, exclude)
    src_of_targets = "exported_nvfp4_cb/quant_config.json"
    if targets is None:
        targets = targets_from_probe(probe, exclude)
        src_of_targets = "probe.pkl inventory"

    cwpath = args.col_weights or f"{args.work}/artifacts/cb_col_weights.pkl"
    colw = None
    if os.path.exists(cwpath):
        with open(cwpath, "rb") as f:
            colw = pickle.load(f)
    else:
        print(f"[warn] no col-weights at {cwpath}: act-weighted columns "
              f"UNAVAILABLE for this model", flush=True)

    spec_a = REGISTRY[canonical_format_name(args.fmt_a)]
    spec_b = REGISTRY[canonical_format_name(args.fmt_b)]
    hdrs = load_st_headers(args.source)
    skeleton = _LazySkeleton(args.source)
    profile = detect_profile(args.source)
    src_idx = build_source_index(hdrs, profile)
    print(f"[cfg] work={args.work}\n[cfg] source={args.source}\n"
          f"[cfg] targets={len(targets)} from {src_of_targets}; "
          f"A={spec_a.name} B={spec_b.name}; "
          f"col_weights={'yes' if colw is not None else 'NO'}; "
          f"cb_imatrix_fit={args.cb_imatrix_fit}", flush=True)

    def cost(W, spec, cw, fit_cw=None):
        # fp32 in for BOTH formats: the CB qdq path expects fp32, and cost is
        # measured in fp32 per the cross-layer-additivity finding.
        Wf = W.float()
        q = spec.quantize_dequantize(Wf, fit_cw) if fit_cw is not None \
            else spec.quantize_dequantize(Wf)
        d2 = (q.float() - Wf) ** 2
        um = d2.mean().item()                       # unweighted weight-MSE
        wm = float("nan")
        if cw is not None:
            w = cw.to(dev, torch.float32).clamp_min(0)
            w = w / w.mean().clamp_min(1e-30)
            wm = (d2 * w[None, :]).mean().item()    # act-col-weighted
        return um, wm

    def diagnostics(W):
        """Cheap per-unit shape/outlier descriptors for the post-hoc analysis
        of which units favour the fixed-grid format."""
        Wf = W.float()
        rms_row = Wf.pow(2).mean(dim=1).clamp_min(1e-30).sqrt()
        rowmax = Wf.abs().amax(dim=1)
        rowratio = (rowmax / rms_row).mean().item()
        rms = Wf.pow(2).mean().clamp_min(1e-30).sqrt()
        kurt = (Wf.pow(4).mean() / Wf.pow(2).mean().clamp_min(1e-30) ** 2 - 3.0)
        return rowratio, float(kurt.item()), (rowmax.max() / rms).item()

    rows = []
    items = sorted(targets.items())
    if args.limit:
        items = items[: args.limit]
    for i, (t, rung) in enumerate(items):
        key = resolve_source_key(src_idx, hdrs, t)
        if key is None:
            print(f"[skip] {t}: no source tensor", flush=True)
            continue
        meta = hdrs[key][1]
        if len(meta["shape"]) != 2:
            print(f"[skip] {t}: shape {meta['shape']} not 2-D", flush=True)
            continue
        W = load_source_weight(skeleton, key, dev)
        h = h_trace_of(probe, t)
        cw = colw.get(t) if hasattr(colw, "get") else None
        fit_cw = (cw.to(dev, torch.float32)
                  if (args.cb_imatrix_fit and cw is not None) else None)
        u_n, w_n = cost(W, spec_a, cw)
        u_k, w_k = cost(W, spec_b, cw, fit_cw=fit_cw)
        rowratio, kurt, maxratio = diagnostics(W)
        n_out, n_in = int(meta["shape"][0]), int(meta["shape"][1])
        del W
        torch.cuda.empty_cache()
        rows.append((t, unit_role(t), unit_leaf(t), rung, h,
                     u_n, u_k, w_n, w_k, n_out, n_in,
                     rowratio, kurt, maxratio))
        if i % 50 == 0:
            print(f"[{i}/{len(items)}] {t}: {spec_a.name} {u_n:.3e} vs "
                  f"{spec_b.name} {u_k:.3e} "
                  f"({spec_b.name if u_k < u_n else spec_a.name} wins)",
                  flush=True)

    out = args.out or f"{args.work}/ab_nvfp4_vs_k36.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["target", "role", "leaf", "shipped_rung", "h_trace",
                     "nvfp4_mse", "k36_mse", "nvfp4_wmse", "k36_wmse",
                     "out_features", "in_features",
                     "rowmax_over_rms", "excess_kurtosis", "maxabs_over_rms"])
        wr.writerows(rows)

    have_act = colw is not None
    schemes = [("unweighted weight-MSE", 5, 6)]
    if have_act:
        schemes.append(("act-column-weighted weight-MSE", 7, 8))

    for label, ui, ki in schemes:
        print(f"\n== {label}: {spec_a.name}(4.5) vs {spec_b.name}(4.5) ==")
        for group_idx, group_name in ((1, "role"), (2, "leaf")):
            by = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for r in rows:
                for key in (r[group_idx], "ALL") if group_idx == 1 else (r[group_idx],):
                    b = by[key]
                    b[0] += 1                                   # n
                    win = r[ki] < r[ui]
                    b[1] += 1 if win else 0                     # count wins
                    b[2] += math.log(max(r[ki], 1e-300) / max(r[ui], 1e-300))
                    b[3] += r[4]                                # sum h_trace
                    b[4] += r[4] if win else 0.0                # h of winners
                    b[5] += r[4] * r[ui]                        # sum h*mse A
                    b[6] += r[4] * r[ki]                        # sum h*mse B
            if group_idx == 2:
                print(f"  -- by {group_name} --")
            for g, (n, kw, lg, hs, hw, da, db) in sorted(
                    by.items(), key=lambda kv: (-kv[1][0], kv[0])):
                print(f"  {g:28s} n={n:4d}  {spec_b.name} wins {kw:4d} "
                      f"({100*kw/max(n,1):5.1f}%)  geomean B/A = "
                      f"{math.exp(lg/max(n,1)):.3f}  "
                      f"h-weighted win {100*hw/max(hs,1e-30):5.1f}%  "
                      f"sum(h*mse) B/A = {db/max(da,1e-300):.3f}")

    if not have_act:
        print("\n[note] act-column-weighted columns NOT computed "
              "(no cb_col_weights.pkl in this work dir).")
    print(f"\nper-unit CSV: {out}")


if __name__ == "__main__":
    main()
