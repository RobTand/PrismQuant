"""Does input-side rotation cut Tessera's ACTIVATION leg under W4A4?

Why this needed running.  `R_IN_ONLY` rotation was measured at 0.987x -- 1.3%
WORSE -- and written off.  That measurement was weight-space, and
`reconstruct_unit` ends with `undo_rotation(...)`, so the rotated basis was
undone before scoring: the activations never saw the rotation at all.  It was
scored on the one axis where an input-basis rotation cannot help.

Under W4A4 the activation term dominates -- backing it out of NVFP4's two
measured faces gives 0.0856, larger than EXL3 K=4's entire weight error
(0.05653).  With Tessera's current weights, a 42% cut to that leg would reach
EXL3 parity.  Input-side rotation is the standard tool against activation
outliers (QuaRot, SpinQuant, EXL3's own suh/svh) and it is the one rotation
Tessera is allowed: doc S7 permits R_IN_ONLY precisely because "rotating the
input axis is undone by rotating the activation with the same orthogonal
matrix, which the consumer can fold into the preceding op".

So this keeps the rotated basis instead of undoing it, rotates the cached
activations by the same block Hadamard, and quantizes them there.

  y = X W^T = (X H)(W H)^T   for orthogonal H   (hadamard_block is normalised)

Arms are scored functionally on the held-out half of real cached GLM
activations.  A screen, not a served KL (principle 3).
"""
import argparse, json
from fractions import Fraction

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.diagonals import apply_rotation, hadamard_block, _block_size
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"


def rotate(t, size):
    """Block Hadamard on the last (input-feature) axis -- the same transform
    `apply_rotation` puts on the weight, so it cancels in y = X W^T."""
    cols = t.shape[-1]
    return (t.reshape(-1, cols // size, size)
             @ hadamard_block(size, t.device)).reshape(t.shape)


def tessera_hat(w, rotation):
    """Reconstructed weight.  For R_IN_ONLY, returned in the ROTATED basis --
    re-applying the rotation to reconstruct_unit's output cancels its internal
    undo_rotation, both being orthogonal."""
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    rates = bresenham_rate_schedule(Fraction(grid.rate_cap), w.shape[1],
                                    cap=grid.rate_cap)
    forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
    unit = encode_unit(w, forests, rates, CC, rotation=rotation,
                       with_diagonals=False, completion=0, group=32, half=16)
    hat = reconstruct_unit(unit, forests, CC)
    if rotation is RotationState.R_IN_ONLY:
        hat = apply_rotation(hat, RotationState.R_IN_ONLY)[0]
    return hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--out", default="experiments/results/tessera_rotation_w4a4.json")
    a = ap.parse_args()

    m = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    rows = []
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        x = blob["inputs"].float().cuda()
        if x.shape[0] > 2 * a.tokens:
            x = x[: 2 * a.tokens]
        half = x.shape[0] // 2
        x_fit, x_ev = x[:half].contiguous(), x[half:].contiguous()
        size = _block_size(x.shape[-1])

        # Static G is fit per basis: the rotated activations are a different
        # distribution, which is the entire point of rotating them.
        g_pl = select_mse_grid_input_global_scale([x_fit])
        xr_fit, xr_ev = rotate(x_fit, size), rotate(x_ev, size)
        g_rot = select_mse_grid_input_global_scale([xr_fit])
        xq, xrq = nvfp4_activation_qdq_served(x_ev, g_pl).float(), \
                  nvfp4_activation_qdq_served(xr_ev, g_rot).float()

        keys = [k for k in m if f".layers.{layer}.mlp.experts." in k
                and k.endswith(".weight")]
        picked = 0
        for k in sorted(keys):
            with safe_open(f"{MODEL}/{m[k]}", framework="pt") as f:
                w = f.get_tensor(k)
            if w.shape[1] != x.shape[-1]:
                continue                      # down_proj: no cached input
            w = w.cuda().float()
            wr = apply_rotation(w, RotationState.R_IN_ONLY)[0]
            y = x_ev @ w.T
            ny = torch.linalg.norm(y)
            rel = lambda yh: (torch.linalg.norm(yh - y) / ny).item()

            # 1. activation leg alone, exact weights
            act_plain = rel(xq @ w.T)
            act_rot = rel(xrq @ wr.T)
            # 2. weight leg alone, exact activations  (reproduces the 0.987x)
            hp, hr = tessera_hat(w, RotationState.NONE), \
                     tessera_hat(w, RotationState.R_IN_ONLY)
            w_plain, w_rot = rel(x_ev @ hp.T), rel(xr_ev @ hr.T)
            # 3. the composite that actually ships under W4A4
            both_plain, both_rot = rel(xq @ hp.T), rel(xrq @ hr.T)
            rows.append({"layer": layer, "expert": k.split("experts.")[1],
                         "block": size,
                         "act_plain": act_plain, "act_rot": act_rot,
                         "w_plain": w_plain, "w_rot": w_rot,
                         "both_plain": both_plain, "both_rot": both_rot})
            print(f"  L{layer} {k.split('experts.')[1][:18]:<18} "
                  f"act {act_plain:.5f}->{act_rot:.5f} ({act_rot/act_plain:.3f}x)  "
                  f"W {w_plain:.5f}->{w_rot:.5f} ({w_rot/w_plain:.3f}x)  "
                  f"W4A4 {both_plain:.5f}->{both_rot:.5f} ({both_rot/both_plain:.3f}x)")
            del w, wr, hp, hr
            picked += 1
            if picked >= a.experts:
                break
        del x, x_fit, x_ev, xq, xrq, xr_fit, xr_ev
        torch.cuda.empty_cache()

    if rows:
        import statistics as st
        print()
        for key, lab in (("act", "activation leg"), ("w", "weight leg"),
                         ("both", "W4A4 composite")):
            r = [x[f"{key}_rot"] / x[f"{key}_plain"] for x in rows]
            print(f"  {lab:<16} rotation is {st.mean(r):.3f}x "
                  f"(min {min(r):.3f} max {max(r):.3f}) over {len(r)} projections")
        need = st.mean([x["act_rot"] / x["act_plain"] for x in rows])
        print(f"\n  a 42% cut (0.58x) reaches EXL3 parity; measured {need:.3f}x")
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
