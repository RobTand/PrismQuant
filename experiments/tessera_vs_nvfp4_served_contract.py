"""Contract-matched: Tessera(W4A16) @ 4.0 bpp vs NVFP4(W4A4) @ 4.5 bpp.

The weight-space screen in ``tessera_vs_nvfp4_glm_experts.py`` compares two
dequantized weights.  That is not how either format serves.  On the GLM route
NVFP4 runs **W4A4** -- ``flashinfer_b12x`` quantizes the activation to FP4 too
-- while the Tessera kernel lane decodes to bf16 weights and consumes a bf16
activation (W4A16), exactly as EXL3 does.  Pricing NVFP4 without its activation
leg prices it as it does not deploy, which is the mistake
``tessera-scale-plane-is-not-a-neutral-control`` records in the other
direction.

So this measures the functional error each format actually delivers:

    y_ref = X @ W.T                                   (bf16 source, fp32 math)
    NVFP4 W4A4 = qdq_act(X, G) @ qdq_w(W).T           (as served)
    NVFP4 W4A16 = X @ qdq_w(W).T                      (decomposition only)
    Tessera    = X @ W_tessera.T                      (as served, kernel lane)

NVFP4's weight leg is rendered by the **production** path (GPTQ +
static_act_order + JSO), not RTN, so neither leg of the NVFP4 arm is
handicapped.  **The activation tokens are split.**  GPTQ fits its Hessian on
``X_fit`` and the static ``G`` is calibrated on ``X_fit``; every arm is scored
on the disjoint ``X_eval``.  Scored in-sample, GPTQ's render looked 3.3x better
than RTN -- a number that would have flattered NVFP4 exactly the way this
project's damp_sweep evaluator once did.  Tessera's encoder sees no activations
at all, so without the split the two arms were not even measured on the same
footing.  The render is asserted to differ from RTN before it is scored:
``render_activations_keyed_by_qname`` records that a wrong activation key
silently falls back to RTN and raises nothing.

``X`` is *real* cached routed-expert input activations from the GLM-5.3-Flash
BF16 probe, not a synthetic draw -- the distinction that
``tessera-source-model-must-match-the-encoder`` exists to enforce.  ``G`` is
chosen by the production policy ``select_mse_grid_input_global_scale``, which
minimizes serve-QDQ MSE, so the NVFP4 activation leg is at its *best* static
setting.  NVFP4's weight leg remains RTN (no GPTQ/JSO), so the weight side
still favours Tessera; both biases are stated rather than netted.

Scope limit: the probe caches one input per packed-expert entry, at hidden dim,
so this covers gate_proj/up_proj only.  down_proj consumes the intermediate
activation, which was not cached, and is excluded -- roughly a third of expert
params are therefore unpriced here.  And this is a functional screen on cached
activations, not a served KL: principle 3 says it selects nothing.
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


def tessera(w, q256=896):
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    rates = bresenham_rate_schedule(Fraction(q256 * 2, 256), w.shape[1], cap=grid.rate_cap)
    forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
    unit = encode_unit(w, forests, rates, CC, rotation=RotationState.NONE,
                       with_diagonals=False, completion=0, group=32, half=16)
    return reconstruct_unit(unit, forests, CC)


def main():
    nv = fr.get_format("NVFP4")
    print(f"{'layer':>5} {'tensor':<12} {'NVFP4 W4A4':>11} {'NVFP4 W4A16':>12} "
          f"{'Tessera':>9} {'T/served':>9}")
    ratios = []
    for layer in (5, 20, 42):
        x_fit, x = activations(layer)
        g = select_mse_grid_input_global_scale([x_fit], device="cuda")
        xq = nvfp4_activation_qdq_served(x, g).float()
        for name, w in experts(layer, 2, x.shape[1]):
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            err = lambda y: ((y - ref).norm() / den).item()
            rtn = nv.quantize_dequantize(w).float()
            wq = render_production_weight(
                w, "NVFP4", qname=name, activations={name: x_fit},
                levers={"gptq": True, "static_act_order": True,
                        "joint_scale_opt": True},
            ).float()
            if torch.equal(wq, rtn):
                raise SystemExit(
                    f"{name}: production render is bit-identical to RTN -- the "
                    "activation key did not land; see the qname memory"
                )
            a4 = err(xq @ wq.T)
            a16 = err(x @ wq.T)
            te = err(x @ tessera(w).float().T)
            ratios.append(te / a4)
            print(f"{layer:>5} {name:<12} {a4:>11.5f} {a16:>12.5f} {te:>9.5f} "
                  f"{te/a4:>9.4f}")
            del w, wf, ref, wq, rtn
            torch.cuda.empty_cache()
        print(f"      (static G = {g:.4g})")
    print(f"\nmean Tessera/NVFP4-as-served ratio: {st.mean(ratios):.4f}  "
          f"(n={len(ratios)})")


if __name__ == "__main__":
    sys.exit(main())
