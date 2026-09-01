"""What the routed-expert menu actually offers, priced on the served contract.

``glm53_bit_trade.py`` asked whether FP8 attention buys expert bits, and
answered yes by 10.7x.  **Its gain leg is refuted.**  It bought
``TESSERA_E4M3_K1_R951`` as a "4.2148 bpp" rung, and two independent facts kill
that: the rung is priced by ``tessera-rung-is-not-a-rate`` at **7.5 bpp** (body
and completion sum to the cap, so a family has one size, not a band), and its
grid digest is not in ``SERIALISABLE_GRIDS``, so it cannot be written at all.
There is no Tessera rung between 4.0 and 8.

So the freed bytes cannot buy a *rate*.  They can only buy a **format change on
a subset of layers** -- packed MoE is uniform within a layer, not across them.
This measures every rung the DP may actually pick for a routed expert, on the
contract each one serves, so the trade can be re-stated on a real menu:

    TESSERA_E2M1_K1  3.5000 bpp  W4A16  (kernel lane)
    TESSERA_E2M1_K2  4.0000 bpp  W4A16  (kernel lane)
    NVFP4            4.5000 bpp  W4A4   (flashinfer_b12x on this route)
    FP8_E4M3         8.0234 bpp  W8A8   (vLLM dynamic per-token activation)

Same harness as ``tessera_vs_nvfp4_served_contract.py``: real cached
routed-expert inputs from the GLM-5.3-Flash BF16 probe, tokens split fit/eval,
every arm scored on the disjoint ``X_eval``.  NVFP4 and FP8 weights are the
**production** render (GPTQ + static_act_order + JSO); Tessera's encoder sees
no activations at all, so the render asymmetry favours the 8- and 4.5-bit arms
and is stated rather than netted.

Scope: gate_proj/up_proj only -- the probe caches one input per packed-expert
entry at hidden dim, so ``down_proj`` (~1/3 of expert params) is unpriced.  This
is a functional screen on cached activations, not a served KL; principle 3 says
it selects nothing.
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
from prismaquant.production_weight_cache import render_production_weight
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served,
    select_mse_grid_input_global_scale,
)

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"


def activations(layer):
    blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                      map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()   # fit, eval


def experts(layer, n, in_features):
    m = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in m if f".layers.{layer}.mlp.experts." in k and k.endswith(".weight")]
    out = []
    for k in keys:
        with safe_open(f"{MODEL}/{m[k]}", framework="pt") as f:
            w = f.get_tensor(k)
        if w.shape[1] != in_features:      # down_proj: no cached input, skip
            continue
        out.append((k.split("experts.")[1].replace(".weight", ""), w.cuda()))
        if len(out) == n:
            break
    return out


def tessera(w, arity):
    """The top -- and only serialisable -- rung of the E2M1 family at ``arity``."""
    grid = tuple_grid(E2M1_GRID, arity, partition="coset") if arity > 1 else E2M1_GRID
    rates = bresenham_rate_schedule(Fraction(grid.rate_cap), w.shape[1],
                                    cap=grid.rate_cap)
    forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
    unit = encode_unit(w, forests, rates, CC, rotation=RotationState.NONE,
                       with_diagonals=False, completion=0, group=32, half=16)
    return reconstruct_unit(unit, forests, CC)


def main():
    nv = fr.get_format("NVFP4")
    fp8 = fr.get_format("FP8_E4M3")
    shape = (4096, 2048)
    print("registry bpp: "
          f"NVFP4 {nv.effective_bits_for_shape(shape):.4f}  "
          f"FP8_E4M3 {fp8.effective_bits_for_shape(shape):.4f}  "
          f"TESSERA_E2M1_K1 "
          f"{fr.get_format('TESSERA_E2M1_K1_R768').effective_bits_for_shape(shape):.4f}  "
          f"TESSERA_E2M1_K2 "
          f"{fr.get_format('TESSERA_E2M1_K2_R896').effective_bits_for_shape(shape):.4f}\n")

    hdr = (f"{'layer':>5} {'tensor':<12} {'T-K1 3.5':>9} {'T-K2 4.0':>9} "
           f"{'NV 4.5 W4A4':>12} {'NV 4.5 W4A16':>13} {'FP8 8.0 W8A8':>13}")
    print(hdr)
    rows = []
    for layer in (5, 20, 42):
        x_fit, x = activations(layer)
        g = select_mse_grid_input_global_scale([x_fit], device="cuda")
        x_nv = nvfp4_activation_qdq_served(x, g).float()
        x_fp8 = fp8.activation_quantize_dequantize(x).float()
        for name, w in experts(layer, 2, x.shape[1]):
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            err = lambda y: ((y - ref).norm() / den).item()
            levers = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}
            w_nv = render_production_weight(
                w, "NVFP4", qname=name, activations={name: x_fit}, levers=levers,
            ).float()
            w_fp8 = render_production_weight(
                w, "FP8_E4M3", qname=name, activations={name: x_fit}, levers=levers,
            ).float()
            if torch.equal(w_nv, nv.quantize_dequantize(w).float()):
                raise SystemExit(
                    f"{name}: NVFP4 production render is bit-identical to RTN -- "
                    "the activation key did not land; see the qname memory"
                )
            row = dict(
                layer=layer, tensor=name,
                t_k1=err(x @ tessera(w, 1).float().T),
                t_k2=err(x @ tessera(w, 2).float().T),
                nv_a4=err(x_nv @ w_nv.T),
                nv_a16=err(x @ w_nv.T),
                fp8_a8=err(x_fp8 @ w_fp8.T),
            )
            rows.append(row)
            print(f"{layer:>5} {name:<12} {row['t_k1']:>9.5f} {row['t_k2']:>9.5f} "
                  f"{row['nv_a4']:>12.5f} {row['nv_a16']:>13.5f} "
                  f"{row['fp8_a8']:>13.5f}")
            del w, wf, ref, w_nv, w_fp8
            torch.cuda.empty_cache()
        print(f"      (static G = {g:.4g})")
        del x_fit, x, x_nv, x_fp8
        torch.cuda.empty_cache()

    print("\n" + "=" * len(hdr))
    mean = {k: st.mean(r[k] for r in rows)
            for k in ("t_k1", "t_k2", "nv_a4", "nv_a16", "fp8_a8")}
    print(f"{'mean':>5} {'':<12} {mean['t_k1']:>9.5f} {mean['t_k2']:>9.5f} "
          f"{mean['nv_a4']:>12.5f} {mean['nv_a16']:>13.5f} {mean['fp8_a8']:>13.5f}")
    print(f"\nn = {len(rows)} projections, 3 layers")
    print(f"NVFP4-as-served / Tessera-4.0 = {mean['nv_a4'] / mean['t_k2']:.4f}  "
          "(>1 means the 4.5 bpp rung is WORSE than the 4.0 one)")
    print(f"FP8-as-served  / Tessera-4.0 = {mean['fp8_a8'] / mean['t_k2']:.4f}")
    print(f"Tessera 3.5 -> 4.0            = "
          f"{(mean['t_k2'] / mean['t_k1'] - 1) * 100:+.1f}% per half-bit")
    json.dump(rows, open("/mnt/shared/tessera-kl/glm53_expert_menu.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
