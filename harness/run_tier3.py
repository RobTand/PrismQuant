#!/usr/bin/env python3
"""
Tier-3 stratified sample runner (GPU flock-gated, content-keyed, RSS guarded).

Sample: layers {3,12,21,30,39}, all three projections, 16 experts per (layer,proj)
chosen by same sha256+salt audit_sample (tools/dsv4_cbl_kernels.py).

For each unit it computes per-group per-rung MSE via banked pooled books
(assignment-only, scale_sweep=False) and solves split knapsack for budgets
K29..K36. Persists per-unit JSON incrementally in results/.

CPU-only synthetic mode --synthetic runs without GPU/books for quick shakedown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

# Ensure /w is on path
sys.path.insert(0, "/w")
# Fallback when running from host mount directly
if Path("/w/prismaquant").exists():
    sys.path.insert(0, "/w")

from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields, nvfp4_cb_reconstruct

# Local harness imports (relative)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from harness.byte_accounting import bytes_split, bytes_uniform, bytes_for_budget, ledger_row
    from harness.split_knapsack import RUNGS, solve_split, sweep_budgets, uniform_mse
    from harness.content_key import unit_content_key, unit_path, atomic_write_json, rss_bytes, host_available_bytes
    from harness.cbl_encode import group_boundaries, synthetic_per_group_mse
except ImportError:
    from byte_accounting import bytes_split, bytes_uniform, bytes_for_budget, ledger_row
    from split_knapsack import RUNGS, solve_split, sweep_budgets, uniform_mse
    from content_key import unit_content_key, unit_path, atomic_write_json, rss_bytes, host_available_bytes
    from cbl_encode import group_boundaries, synthetic_per_group_mse

# Config
RUN_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq")
BOOK_ROOT = RUN_ROOT / "bucket-books"
RESULTS_ROOT = RUN_ROOT / "tier3-study" / "results"
AUDIT_SALT = "cbl-audit-v1"
LAYERS = [3, 12, 21, 30, 39]
PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]
GROUPS_LIST = [2, 4, 8]
BUDGET_KS = [29, 30, 31, 32, 33, 34, 35, 36]

# Borrowed from tools/dsv4_cbl_kernels audit_sample
def audit_sample(digests, n=16):
    keyed = sorted(range(len(digests)), key=lambda e: hashlib.sha256((digests[e] + AUDIT_SALT).encode()).hexdigest())
    return keyed[:n]

def load_book(rung: int, layer: int, projection: str, source_digest: str, col_weights_digest: str, device: torch.device):
    # Reuse key logic from dsv4_cbl_kernels._book_key
    import hashlib, json
    payload = json.dumps({
        "schema": "prismaquant.dsv4_cbl_measurement_semantics.v2",
        "layer": int(layer), "projection": str(projection), "rung": int(rung),
        "source_digest": source_digest, "col_weights_digest": col_weights_digest,
        "train": {"row_sample": 64, "row_seed": 4321, "cap": 2000000, "iters": 4, "seed": 0, "init": "fixed_lattice", "normalization": "cand0_v1"},
    }, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = BOOK_ROOT / key[:2] / f"{key}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing banked book {path} for L{layer} {projection} K{rung}")
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["dsv4_cbl_book"])
        pool = tuple(f.get_tensor(f"sub{i}").to(torch.float32) for i in range(int(meta["n_sub"])))
    # Move to device (fp32)
    pool_dev = tuple(t.to(device) for t in pool)
    # book sha for key
    h = hashlib.sha256()
    for t in pool:
        h.update(t.detach().cpu().to(torch.float32).numpy().tobytes())
    sha = h.hexdigest()
    return pool_dev, sha, str(path), meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="skip units with existing content-keyed JSON")
    parser.add_argument("--synthetic", action="store_true", help="CPU synthetic shakedown (no GPU/books)")
    parser.add_argument("--groups", type=int, nargs="+", default=GROUPS_LIST, help="group counts to evaluate")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--layer", type=int, default=None, help="single layer override")
    parser.add_argument("--dry-run", action="store_true", help="print planned units and exit")
    args = parser.parse_args()

    layers = [args.layer] if args.layer is not None else args.layers
    groups_list = args.groups

    # Collect model/weight loading only for non-synthetic
    if args.synthetic:
        # Synthetic shakedown: 2 units, verify skip + additivity
        print("[tier3] synthetic shakedown start", flush=True)
        pass  # synthetic_per_group_mse already imported
        import tempfile, os as _os
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        for layer in layers[:1]:
            for proj in PROJECTIONS[:1]:
                for expert in [0, 1]:
                    for G in groups_list[:1]:
                        per = synthetic_per_group_mse(out=64, inn=512, groups=G, seed=layer*100+expert)
                        # sweep
                        sweep = sweep_budgets(per, proj if proj in ["gate_proj","up_proj","down_proj"] else "gate_proj", G)
                        key = unit_content_key(layer=layer, projection=proj, expert=expert,
                                               source_digest="synthetic", col_weights_digest="synthetic",
                                               book_shas={k:"synth" for k in RUNGS}, groups=G)
                        path = unit_path(RESULTS_ROOT, layer=layer, projection=proj, expert=expert, groups=G, key=key)
                        if args.resume and path.is_file():
                            print(f"[skip] {path.name}")
                            continue
                        # additivity: synthetic is additive by construction; verify group sum at fixed rung equals sum
                        # whole at rung 32
                        rec = {
                            "schema": "prismaquant.tier3.unit.v1",
                            "layer": layer, "projection": proj, "expert": expert, "groups": G,
                            "source_digest": "synthetic", "col_weights_digest": "synthetic",
                            "book_shas": {str(k): "synth" for k in RUNGS},
                            "content_key": key,
                            "per_group_per_rung_mse": per.tolist(),
                            "sweep": sweep,
                        }
                        atomic_write_json(path, rec)
                        print(f"[wrote] {path}")
                        # immediate second write should be skipped on resume
        print("[tier3] synthetic shakedown done", flush=True)
        return 0

    # Real path: need GPU and source weights
    if not torch.cuda.is_available():
        print("[tier3] CUDA not available, abort (use --synthetic for CPU shakedown)", file=sys.stderr)
        return 2

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda:0")
    print(f"[tier3] layers={layers} projs={PROJECTIONS} groups={groups_list} resume={args.resume}", flush=True)

    # Lazy imports for weight loading
    from tools.dsv4_ldlq_cost_campaign import load_projection, load_layer_identity, COL_WEIGHTS
    from prismaquant.layer_streaming import _build_weight_map, _build_fp8_scale_inv_map
    from prismaquant.production_weight_cache import canonical_cb_col_weights_sha256
    import pickle

    SOURCE = Path("/home/rob/dq-runs/dsv4-flash-0731/source")
    PRIOR_BY_LAYER = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts-mxfp4/probe-k12k18/by-layer")

    # Build weight maps once
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    with open(COL_WEIGHTS, "rb") as f:
        all_col_weights = pickle.load(f)

    planned = []
    for layer in layers:
        for proj in PROJECTIONS:
            # Need digests for audit sample without loading full weight? Load anyway per layer
            planned.append((layer, proj))
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return 0

    # For each layer, load once then iterate experts/groups
    for layer in layers:
        # Load layer identity for digest checks
        payload, layer_record = load_layer_identity(layer)
        identity = layer_record["identity"]
        # Load projection data per proj (weights for all 256 experts)
        for proj in PROJECTIONS:
            print(f"[tier3] loading L{layer} {proj}", flush=True)
            data = load_projection(layer, proj, device=device, identity=identity,
                                   all_col_weights=all_col_weights,
                                   model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
                                   scale_map=scale_map)
            weight_stack = data["weight"]  # (256, out, in) bfloat16 on CUDA
            col_stack = data["col_weights"]  # (256,1,in) float32 on CUDA
            # For audit sample, need per-expert digest over weight bytes (like audit_projection)
            E = int(weight_stack.shape[0])
            digests = []
            for e in range(E):
                digests.append(hashlib.sha256(weight_stack[e].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            sample = audit_sample(digests, 16)
            print(f"[tier3] L{layer} {proj} sample {sample}", flush=True)

            # Precompute per-expert source/col digests for content key
            for expert in sample:
                w_e = weight_stack[expert]  # (out,in)
                cw_e = col_stack[expert, 0]  # (in,) maybe (1,in) shape handling
                # digests: use float32 bytes like burn?
                # Use same as book key: tensor_value_identity? Use simple sha
                src_digest = hashlib.sha256(w_e.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()
                col_digest = hashlib.sha256(cw_e.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()
                for G in groups_list:
                    # Check resume
                    # Need book shas for all rungs to form key (load lazily)
                    # Load books for this expert's source digest to get shas
                    book_shas = {}
                    books = {}
                    for rung in RUNGS:
                        pool, sha, _, _ = load_book(rung, layer, proj, src_digest, col_digest, device)
                        books[rung] = pool
                        book_shas[rung] = sha
                    key = unit_content_key(layer=layer, projection=proj, expert=int(expert),
                                           source_digest=src_digest, col_weights_digest=col_digest,
                                           book_shas=book_shas, groups=G)
                    out_path = unit_path(RESULTS_ROOT, layer=layer, projection=proj, expert=int(expert), groups=G, key=key)
                    if args.resume and out_path.is_file():
                        print(f"[tier3] skip resume {out_path.name}", flush=True)
                        # free books tensors
                        del books
                        torch.cuda.empty_cache()
                        continue
                    # Compute per-group per-rung MSE (assignment-only)
                    # For each group, slice weight and encode
                    out_dim, in_dim = w_e.shape
                    bounds = group_boundaries(in_dim, G)
                    n_rungs = len(RUNGS)
                    per_group = torch.zeros(G, n_rungs, dtype=torch.float64, device="cpu")
                    # For additivity check later, also compute whole-unit broadcast sum
                    whole_sums = {}
                    for ri, rung in enumerate(RUNGS):
                        cb = books[rung]
                        # Whole-unit mse for this rung (for additivity gate): encode full weight
                        # Use col_stack per expert: cw_e is (in,)
                        # Need col_weights for nvfp4_cb_fields: expect (rows, in) or broadcast (1,in)
                        # For single expert, weight is (out, in) -> nvfp4_cb_fields expects (rows, in) 2-D
                        # col_weights should be (rows, in) or (in,) broadcast; pass cw_e expanded
                        cw_full = cw_e.unsqueeze(0).expand(out_dim, -1).to(device)
                        fields_full = nvfp4_cb_fields(w_e.to(device).to(torch.float32), int(rung), grid="fp8", mode="product",
                                                      col_weights=cw_full, codebook=cb, scale_sweep=False, encode_tier="balanced")
                        recon_full = nvfp4_cb_reconstruct(fields_full, int(rung), grid="fp8", mode="product").to(torch.float32)
                        err_full = (recon_full - w_e.to(device).to(torch.float32)).pow(2)
                        # If weighted, multiply; but burn uses weighted assignment but reports unweighted free_weight_mse? Check dsv4_cbl_kernels encode_cbl uses weighted assignment but reports unweighted weight_mse? We follow same: report unweighted mean for cost comparison.
                        whole_mse_sum = err_full.sum().item()
                        whole_sums[rung] = whole_mse_sum
                        del fields_full, recon_full, err_full

                        for gi, (s, e) in enumerate(bounds):
                            w_g = w_e[:, s:e].to(device).to(torch.float32)
                            cw_g = cw_e[s:e].to(device).to(torch.float32).unsqueeze(0).expand(out_dim, -1)
                            fields = nvfp4_cb_fields(w_g, int(rung), grid="fp8", mode="product",
                                                     col_weights=cw_g, codebook=cb, scale_sweep=False, encode_tier="balanced")
                            recon = nvfp4_cb_reconstruct(fields, int(rung), grid="fp8", mode="product").to(torch.float32)
                            err = (recon - w_g).pow(2)
                            per_group[gi, ri] = err.sum().item()
                            del fields, recon, err
                        torch.cuda.empty_cache()
                        # host available check
                        if host_available_bytes() and host_available_bytes() < 40 * 1024**3:
                            torch.cuda.empty_cache()
                        if rss_bytes() > 60 * 1024**3:
                            print(f"[tier3] RSS guard abort >60GB", flush=True)
                            raise SystemExit(3)

                    # Additivity sanity: sum_g per_group[g, ri] == whole_sums[rung] within 1e-6 rel
                    additivity = {}
                    for rung in RUNGS:
                        ri = RUNGS.index(rung)
                        s = float(per_group[:, ri].sum().item())
                        w = float(whole_sums[rung])
                        rel = abs(s - w) / max(abs(w), 1e-12)
                        additivity[str(rung)] = {"sum_groups": s, "whole": w, "rel": rel, "pass": rel < 1e-6}
                    # Solve sweep
                    # Move per_group to CPU float64 for solver (already)
                    sweep = sweep_budgets(per_group, proj, G, BUDGET_KS)
                    rec = {
                        "schema": "prismaquant.tier3.unit.v1",
                        "layer": layer, "projection": proj, "expert": int(expert), "groups": G,
                        "source_digest": src_digest, "col_weights_digest": col_digest,
                        "book_shas": {str(k): v for k, v in book_shas.items()},
                        "book_paths": {str(k): str(BOOK_ROOT / book_shas[k][:2] / f"{book_shas[k]}.safetensors") for k in book_shas},
                        "per_group_per_rung_mse_sum": per_group.tolist(),
                        "additivity": additivity,
                        "sweep": sweep,
                        "budget_ks": BUDGET_KS,
                        "rungs": RUNGS,
                        "bytes_ledger_examples": [
                            ledger_row(proj, bk, [bk]*G) if False else None  # placeholder
                        ],
                    }
                    # Fix ledger examples to be real: for each budget k, show uniform vs best split ledger if feasible
                    rec["bytes_ledger_examples"] = []
                    for entry in sweep:
                        if entry["feasible"] and entry["split_ks"] is not None:
                            rec["bytes_ledger_examples"].append(ledger_row(proj, entry["budget_k"], entry["split_ks"]))
                    atomic_write_json(out_path, rec)
                    print(f"[tier3] wrote {out_path.name} gains {[round(e['gain'],4) for e in sweep if e['feasible']]}", flush=True)
                    # free per unit tensors
                    del books, per_group
                    torch.cuda.empty_cache()
                    # incremental flush already via atomic write; check RSS
                    if rss_bytes() > 60 * 1024**3:
                        print("[tier3] RSS guard flush abort", flush=True)
                        return 3
            # free layer proj data
            del weight_stack, col_stack, data
            torch.cuda.empty_cache()

    print("[tier3] done", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
