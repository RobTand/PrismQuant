"""Is the trade good?  Measure both sides of "FP8 attention buys expert bits".

``docs/measurements/glm53-body-budget-2026-09-01.md`` shows that Mia leaves
15.44 GiB at 16 bpp on 2.6% of GLM's parameters, and that pricing attention and
``lm_head`` at FP8 frees 7.692 GiB -- worth **+0.2171 bpp on the routed
experts** at identical total size.  That is byte arithmetic.  It says the trade
is *available*; it does not say it is *good*.

This measures both sides on real cached activations:

    cost  : other_body Linears, BF16 -> FP8_E4M3       (what the trade spends)
    gain  : expert Linears, 4.0000 bpp -> 4.2285 bpp   (what the trade buys)

Both in the same currency -- relative functional error on ``y = X W^T``, which
is what the activations make measurable -- so the two legs are comparable
without a Fisher weighting that would import a second set of assumptions.

**The expert rungs are not the same family.**  ``TESSERA_E2M1_K2`` caps at
exactly 4.000 bpp (``q256`` max 896), so the higher rung cannot come from it;
``TESSERA_E4M3_K1`` is the only *serialisable* family covering the band (the
LM* Lloyd-Max families reach it but have no wire identity -- their values are
fitted to the tensor and no identifier reproduces them).  So the gain leg is
also a family change, and that is stated rather than hidden: it is what the
allocator would actually have to pick.

Held out throughout: FP8 and Tessera see no activations, but the reference and
every arm are scored on ``X_eval``, disjoint from the ``X_fit`` half, so this
harness cannot be flattered by an in-sample render the way the damp_sweep
evaluator once was.
"""
import json
import statistics as st
import sys
from fractions import Fraction

import torch
from safetensors import safe_open

import prismaquant.format_registry as fr
from prismaquant.tessera_render import render_tessera_weight

MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
INDEX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

# The freed bytes buy +0.2171 bpp on top of 4.0000, so the ceiling is 4.2171
# and the highest rung at or under it is q256=951 (4.2148).  Rounding UP would
# price a rung the budget cannot afford and quietly flatter the gain leg.
EXPERT_LOW = "TESSERA_E2M1_K2_R896"      # 4.0000 bpp
EXPERT_HIGH = "TESSERA_E4M3_K1_R951"     # 4.2148 bpp


def load(name):
    with safe_open(f"{MODEL}/{INDEX[name]}", framework="pt") as f:
        return f.get_tensor(name).cuda()


def split_act(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()


def rel_err(x, w_ref, w_q):
    ref = x @ w_ref.T
    return ((x @ w_q.T - ref).norm() / ref.norm()).item()


def main():
    fp8 = fr.get_format("FP8_E4M3")
    print(f"registry bpp: {EXPERT_LOW} = "
          f"{fr.get_format(EXPERT_LOW).effective_bits_for_shape((4096, 2048)):.4f}, "
          f"{EXPERT_HIGH} = "
          f"{fr.get_format(EXPERT_HIGH).effective_bits_for_shape((4096, 2048)):.4f}, "
          f"FP8_E4M3 = {fp8.effective_bits_for_shape((4096, 2048)):.4f}\n")

    # ---- cost leg: what BF16 -> FP8 does to the attention/dense body --------
    print("COST leg -- other_body BF16 -> FP8_E4M3 (rel functional error)")
    cost = []
    for layer in (5, 20, 42):
        for proj in ("mlp__gate_proj", "mlp__up_proj", "mlp__down_proj",
                     "mlp__shared_experts__gate_proj"):
            path = f"{ACT}/model__language_model__layers__{layer}__{proj}.pt"
            name = ("model.language_model.layers."
                    f"{layer}.{proj.replace('__', '.')}.weight")
            if name not in INDEX:
                continue
            try:
                _, x = split_act(path)
            except FileNotFoundError:
                continue
            w = load(name)
            if w.shape[1] != x.shape[1]:
                del w; torch.cuda.empty_cache(); continue
            e = rel_err(x, w.float(), fp8.quantize_dequantize(w).float())
            cost.append(e)
            print(f"  L{layer:<3d} {proj:<32s} {e:.6f}")
            del w, x
            torch.cuda.empty_cache()

    # ---- gain leg: what +0.215 bpp does to the routed experts --------------
    print("\nGAIN leg -- experts 4.0000 -> 4.2148 bpp (rel functional error)")
    gain = []
    for layer in (5, 20, 42):
        _, x = split_act(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt")
        keys = [k for k in INDEX
                if f".layers.{layer}.mlp.experts." in k and k.endswith(".weight")]
        picked = 0
        for k in keys:
            w = load(k)
            if w.shape[1] != x.shape[1] or w.shape[0] % 2:
                del w; torch.cuda.empty_cache(); continue
            wf = w.float()
            lo = rel_err(x, wf, render_tessera_weight(w, EXPERT_LOW).float())
            hi = rel_err(x, wf, render_tessera_weight(w, EXPERT_HIGH).float())
            gain.append((lo, hi))
            print(f"  L{layer:<3d} {k.split('experts.')[1][:26]:<26s} "
                  f"{lo:.6f} -> {hi:.6f}  ({(hi/lo - 1)*100:+.1f}%)")
            del w, wf
            torch.cuda.empty_cache()
            picked += 1
            if picked == 2:
                break
        del x
        torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    c = st.mean(cost)
    lo = st.mean(l for l, _ in gain)
    hi = st.mean(h for _, h in gain)
    print(f"cost: other_body picks up {c:.6f} rel error going BF16 -> FP8")
    print(f"      on {7648.2 / 320120.0 * 100:.2f}% of quantizable params")
    print(f"gain: experts go {lo:.6f} -> {hi:.6f} rel error ({(hi/lo - 1)*100:+.1f}%)")
    print(f"      on {304405.8 / 320120.0 * 100:.2f}% of quantizable params")
    print()
    if hi < lo:
        print("The extra expert bits REDUCE expert error, and the FP8 leg pays "
              f"{c:.6f} on 2.4% of the params to get it.")
    else:
        print("The higher rung did NOT reduce expert error -- the family change "
              "(E2M1_K2 -> E4M3_K1) costs more than the 0.215 bpp buys, and the "
              "trade as stated does not hold. Report this, do not bury it.")
    print("\nCaveat: relative functional error is a screen, not a KL. It ranks; "
          "it does not promote (principle 3).")


if __name__ == "__main__":
    sys.exit(main())
