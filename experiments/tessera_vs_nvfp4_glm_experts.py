"""Tessera E2M1_K2 @ 4.0 bpp vs NVFP4 @ 4.5 bpp, on real GLM-5.3-Flash experts.

Weight-space rel_err only -- a screen, never a promotion metric (principle 3).
It is reported because the gap it shows is large enough to change what gets
built next, not because it settles anything.

Both arms are RTN-class: NVFP4 here is the registry's plain quantize_dequantize,
without GPTQ or JSO, which are worth several percent. That makes NVFP4 the
*weak* arm, so a Tessera loss here is a floor on the real gap, not a ceiling.
"""
import json, statistics as st, sys
from fractions import Fraction
import torch
from safetensors import safe_open
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode
import prismaquant.format_registry as fr

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"


def experts(layer=20, n=6):
    m = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in m if f".layers.{layer}.mlp.experts." in k
            and k.endswith(".weight")][:n]
    for k in keys:
        with safe_open(f"{MODEL}/{m[k]}", framework="pt") as f:
            yield k.split("experts.")[1].replace(".weight", ""), f.get_tensor(k).cuda()


def render(w, q256, rot=RotationState.NONE, diag=False, part="coset"):
    grid = tuple_grid(E2M1_GRID, 2, partition=part)
    rates = bresenham_rate_schedule(Fraction(q256 * 2, 256), w.shape[1],
                                    cap=grid.rate_cap)
    forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
    unit = encode_unit(w, forests, rates, CC, rotation=rot, with_diagonals=diag,
                       completion=0, group=32, half=16)
    return reconstruct_unit(unit, forests, CC)


def main():
    nv = fr.get_format("NVFP4")
    arms = [("plain", RotationState.NONE, False), ("+diag", RotationState.NONE, True),
            ("+rot", RotationState.R_IN_ONLY, False),
            ("+rot+diag", RotationState.R_IN_ONLY, True)]
    acc = {a: [] for a, _, _ in arms}
    nvs = []
    for name, w in experts():
        wf = w.float(); den = wf.norm()
        err = lambda t: ((t.float() - wf).norm() / den).item()
        nvs.append(err(nv.quantize_dequantize(w)))
        for a, rot, diag in arms:
            acc[a].append(err(render(w, 896, rot, diag)))
    base = st.mean(nvs)
    print(f"NVFP4@4.5 rel_err {base:.5f}")
    for a, _, _ in arms:
        mean = st.mean(acc[a])
        print(f"  Tessera 4.0 {a:<10} {mean:.5f}   ratio {mean/base:.4f}")


if __name__ == "__main__":
    sys.exit(main())
