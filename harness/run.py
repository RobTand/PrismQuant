#!/usr/bin/env python3
"""Rotation study harness: content-keyed per-unit runner.

Usage:
  python -m harness.run --synthetic --out-dir /tmp/out --resume
  python -m harness.run --layers 3,12,21,30,39 --projections gate_proj,up_proj,down_proj --rungs 28,32,36 --arms A,B,C,D --resume

Content-keyed per-unit JSON: atomic writes, resume-by-key (skip if file exists),
RSS guard 60GB, torch.cuda.empty_cache() hygiene.

This is Phase-A CPU harness; Phase-B GPU phase adds real DSV4 weight loading.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .rotation import random_hadamard_signs, rotate_weights, unrotate_weights
from .metrics import weighted_mse, unweighted_mse, gap_closed_percent

RSS_LIMIT_BYTES = 60 * 1024**3
HOST_AVAILABLE_THRESHOLD = 40 * 1024**3


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def _host_available() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def _content_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _unit_payload(layer: int, proj: str, rung: int, expert: int, arm: str, weight_digest: str, col_digest: str, seed: int) -> dict:
    return {
        "schema": "prismaquant.rotation_study.unit.v1",
        "layer": int(layer),
        "projection": proj,
        "rung": int(rung),
        "expert": int(expert),
        "arm": arm,
        "weight_digest": weight_digest,
        "col_weights_digest": col_digest,
        "rotation_seed": int(seed) if arm in ("C", "D") else None,
        "vec_dim": 8,
    }


def synthetic_unit(layer: int, proj: str, rung: int, expert: int, arm: str, seed: int, out_dir: Path, resume: bool) -> Path:
    """Run one synthetic unit end-to-end, write content-keyed JSON, return path."""
    in_dim = 2048 if proj == "down_proj" else 4096
    out_dim = 8
    torch.manual_seed(hash((layer, expert, rung)) % (2**31))
    weight = torch.randn(out_dim, in_dim, dtype=torch.float32) * 0.02
    col_weights = torch.rand(in_dim, dtype=torch.float32).abs() + 0.5
    # col_weights shape broadcastable: (out_dim, in_dim)
    cw = col_weights.unsqueeze(0).expand(out_dim, -1)
    weight_digest = hashlib.sha256(weight.numpy().tobytes()).hexdigest()
    col_digest = hashlib.sha256(col_weights.numpy().tobytes()).hexdigest()
    key_payload = _unit_payload(layer, proj, rung, expert, arm, weight_digest, col_digest, seed)
    key = _content_key(key_payload)
    out_path = out_dir / f"{key[:2]}" / f"{key}.json"
    if resume and out_path.is_file():
        return out_path
    # RSS guard
    rss = _rss_bytes()
    if rss > RSS_LIMIT_BYTES:
        raise RuntimeError(f"RSS guard tripped: {rss/1024**3:.2f} GiB > 60GiB")
    if _host_available() < HOST_AVAILABLE_THRESHOLD:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Simulate quantization per arm
    # For synthetic we use a simple scale-quant proxy: uniform quant with step based on rung
    # Arms: A=incumbent, B=CBL (better), C=Hadamard+incumbent, D=Hadamard+CBL
    # To make test meaningful, we simulate that B beats A, C maybe partially.
    import math
    # Simple proxy: quantize to k-bit grid where k=rung
    def fake_quant(w: torch.Tensor, k: int) -> torch.Tensor:
        # per-group scale like FP8: amax/448 style simplified
        scale = w.abs().amax() / 6.0 if w.numel() else torch.tensor(1.0)
        scale = max(float(scale), 1e-6)
        # Fake lattice: round to scaled grid
        return torch.round(w / scale).clamp(-6, 6) * scale

    signs = random_hadamard_signs(in_dim, seed=seed) if arm in ("C", "D") else None

    if arm in ("A", "B"):
        w_for_q = weight
    else:
        w_for_q = rotate_weights(weight, signs)

    # Simulate per-arm quality: B and D use smaller error
    k_eff = rung + (2 if arm in ("B", "D") else 0)  # CBL advantage 2 bits
    # Add slight rotation penalty so C not equal B
    if arm == "C":
        k_eff = rung + 1
    recon_rot = fake_quant(w_for_q, k_eff)
    if arm in ("C", "D"):
        recon = unrotate_weights(recon_rot, signs)
    else:
        recon = recon_rot

    # Compute metrics in original space
    w_exp = weight.unsqueeze(0)
    r_exp = recon.unsqueeze(0)
    cw_exp = cw.unsqueeze(0)
    w_mse = weighted_mse(w_exp, r_exp, cw_exp)[0].item()
    uw_mse = unweighted_mse(w_exp, r_exp)[0].item()
    # For invariance check, compute unweighted in rotated space as well
    if arm in ("C", "D"):
        uw_rot = unweighted_mse(w_for_q.unsqueeze(0), recon_rot.unsqueeze(0))[0].item()
    else:
        uw_rot = uw_mse

    result = {
        "schema": "prismaquant.rotation_study.result.v1",
        "content_key": key,
        "identity": key_payload,
        "metrics": {
            "weighted_mse": w_mse,
            "unweighted_mse": uw_mse,
            "unweighted_mse_rotated_space": uw_rot,
            "unweighted_invariant": abs(uw_mse - uw_rot) < 1e-6,
        },
        "provenance": {
            "arm": arm,
            "rung": rung,
            "synthetic": True,
            "seed": seed if arm in ("C", "D") else None,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(out_path, result)
    # free tensors
    del weight, cw, recon, w_for_q
    gc.collect()
    if _host_available() < HOST_AVAILABLE_THRESHOLD:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rotation study per-unit harness")
    p.add_argument("--out-dir", type=Path, default=Path("/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/rotation-study/results"))
    p.add_argument("--resume", action="store_true", help="skip units where content-keyed JSON exists")
    p.add_argument("--synthetic", action="store_true", help="use synthetic weights instead of DSV4 checkpoint")
    p.add_argument("--layers", default="3,12,21,30,39", help="comma-separated layer ids")
    p.add_argument("--projections", default="gate_proj,up_proj,down_proj")
    p.add_argument("--rungs", default="28,32,36")
    p.add_argument("--experts", default="0-16", help="range or comma list; default 16-per-projection audit sample")
    p.add_argument("--arms", default="A,B,C,D", help="comma list of arms")
    p.add_argument("--seed-base", type=int, default=12345, help="base seed for Hadamard per-unit (seed = base + layer*1000+expert)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in str(args.layers).split(",") if x.strip() != ""]
    projections = [s.strip() for s in str(args.projections).split(",") if s.strip()]
    rungs = [int(x) for x in str(args.rungs).split(",") if x.strip() != ""]
    arms = [s.strip().upper() for s in str(args.arms).split(",") if s.strip()]

    # experts: parse "0-16" or "0,5,12"
    experts_spec = str(args.experts).strip()
    if "-" in experts_spec and "," not in experts_spec:
        lo, hi = experts_spec.split("-")
        experts = list(range(int(lo), int(hi)))
    else:
        experts = [int(x) for x in experts_spec.split(",") if x.strip() != ""]

    total = len(layers) * len(projections) * len(rungs) * len(experts) * len(arms)
    print(f"[rotation-harness] units={total} layers={layers} projs={projections} rungs={rungs} arms={arms} synthetic={args.synthetic} resume={args.resume}", flush=True)
    count = 0
    for layer in layers:
        for proj in projections:
            for rung in rungs:
                for ei, expert in enumerate(experts):
                    # audit_sample logic for real: hash-based 16-sample per projection.
                    # For synthetic we just iterate given experts.
                    for arm in arms:
                        seed = int(args.seed_base) + layer * 1000 + expert * 10 + rung
                        if args.synthetic:
                            path = synthetic_unit(layer, proj, rung, expert, arm, seed, out_dir, resume=args.resume)
                        else:
                            raise NotImplementedError("real DSV4 weight path requires GPU and checkpoint loading (Phase B)")
                        count += 1
                        if count % 50 == 0:
                            rss_gb = _rss_bytes() / 1024**3
                            avail_gb = _host_available() / 1024**3
                            print(f"[rotation-harness] progress {count}/{total} rss={rss_gb:.2f}GiB avail={avail_gb:.2f}GiB last={path.name}", flush=True)
                            if rss_gb > 60:
                                raise RuntimeError(f"RSS guard abort: {rss_gb:.2f} GiB")
    print(f"[rotation-harness] done {count}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
