#!/usr/bin/env python3
"""calibrate_allocator.py — empirical calibration for PrismaQuant frontier points.

Given:
  - sensitivity probe pickle
  - measured per-format cost pickle
  - a target set of average-bit budgets

This script:
  1. Rebuilds PrismaQuant assignments for each target
  2. Applies the chosen native formats in-memory to a real model
  3. Measures actual KL against the BF16 reference logits on a small
     calibration corpus
  4. Fits per-format scalar gains α_f by non-negative least squares so
     that  Σ_f α_f · S_f(pt)  ≈  ΔKL_pt  across the bake-off frontier,
     where  S_f(pt) is the allocator's own per-candidate predicted Δloss
     contribution grouped by format at frontier point pt.

The fitted gains land in the output JSON under `calibrated_gains` and
are consumed by `allocator.py --calibration <this output>`. With a
single global gain (all α_f equal) the DP's choice is invariant; the
per-format fit lets calibration actually shift the chosen recipe.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    iter_quantizable_tensors,
    kl_divergence,
    load_wikitext_calibration,
    stage_multimodal,
)
from prismaquant import format_registry as fr
from prismaquant.allocator import (
    aggregate_moe_candidates,
    build_candidates,
    compute_achieved,
    expand_moe_assignment,
    kneedle,
    promote_fused,
    solve_allocation,
)
from prismaquant.allocator_candidates import cost_entry_predicted_dloss
from prismaquant.schemas import validate_cost_payload, validate_probe_payload


def load_inputs(probe_path: Path, costs_path: Path, fmt_names: list[str]):
    import pickle

    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    with open(costs_path, "rb") as f:
        cost_data = pickle.load(f)
    validate_probe_payload(probe, str(probe_path))
    validate_cost_payload(cost_data, str(costs_path))

    stats = probe["stats"]
    costs = cost_data["costs"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)
    return stats, costs, specs_sorted


def per_format_predicted_breakdown(
    assignment: dict[str, str],
    stats_alloc: dict,
    costs_alloc: dict,
) -> dict[str, float]:
    """Sum the per-(layer, format) predicted Δloss contributions, grouped
    by the assigned format. Returned dict maps fmt → S_f(pt)."""
    out: dict[str, float] = {}
    for name, fmt in assignment.items():
        entry = {}
        for alias in fr.aliases_for(fmt):
            entry = costs_alloc.get(name, {}).get(alias, {})
            if entry:
                break
        contrib = cost_entry_predicted_dloss(stats_alloc[name], entry)
        out[fmt] = out.get(fmt, 0.0) + contrib
    return out


def build_curve(stats: dict, costs: dict, specs_sorted, targets: list[float], bit_precision: float,
                no_fused_promote: bool, expert_granularity: str):
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs_sorted}

    candidates = build_candidates(stats, costs, specs_sorted)
    stats_alloc = stats
    costs_alloc = costs
    if expert_granularity == "layer":
        stats_alloc, costs_alloc, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates
        )

    curve = []
    for t in targets:
        result = solve_allocation(stats_alloc, candidates, t, bit_precision)
        if result is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        assignment, _chosen = result
        if not no_fused_promote:
            assignment = promote_fused(assignment, format_rank)
        achieved, _ = compute_achieved(stats_alloc, assignment, format_specs)
        per_fmt = per_format_predicted_breakdown(assignment, stats_alloc, costs_alloc)
        total_pred = sum(per_fmt.values())
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": total_pred,
            "predicted_dloss_by_format": per_fmt,
            "assignment": assignment,
            "stats_scope": "aggregated" if expert_granularity == "layer" else "expert",
        })
    return curve, stats_alloc, costs_alloc, format_rank


def fit_calibrated_gains(
    results: list[dict],
    baseline_kl: float,
    *,
    floor: float = 1e-3,
    ceiling: float = 1e3,
) -> tuple[dict[str, float], dict]:
    """Fit per-format scalar gains α_f via non-negative least squares so
    that  Σ_f α_f · S_f(pt)  ≈  ΔKL_pt  across the measured frontier.

    Returns (gains, diagnostics). Gains are clamped to [`floor`, `ceiling`]
    to keep the allocator numerically well-behaved when a frontier point
    contributes negligible information about a particular format.

    A "format with all-zero columns" cannot be identified — its gain is
    set to 1.0 (no correction). This happens for BF16 in any bake-off
    that includes a BF16 bucket, since BF16 weight_mse is exactly zero.

    The fit is intentionally simple (NNLS, no regularization beyond
    bounds): the bake-off frontier typically has only 3 points, so any
    heavier model would over-fit. With ≥3 points the per-format fit is
    over-determined for 2 active formats (the typical NVFP4+MXFP8 case).
    """
    measured = np.array(
        [float(r["actual_last_token_kl"]) - float(baseline_kl) for r in results],
        dtype=np.float64,
    )
    fmts = sorted({fmt for r in results for fmt in r.get("predicted_dloss_by_format", {})})
    if not fmts or measured.size == 0:
        return {}, {"reason": "no measured frontier points"}

    A = np.zeros((len(results), len(fmts)), dtype=np.float64)
    for i, r in enumerate(results):
        breakdown = r.get("predicted_dloss_by_format", {})
        for j, f in enumerate(fmts):
            A[i, j] = float(breakdown.get(f, 0.0))

    active_cols = np.where(A.sum(axis=0) > 0)[0]
    diagnostics: dict = {
        "frontier_points": int(measured.size),
        "fmts_in_fit": [fmts[j] for j in active_cols],
        "fmts_unidentifiable": [fmts[j] for j in range(len(fmts))
                                if j not in active_cols],
    }
    gains = {f: 1.0 for f in fmts}
    if active_cols.size == 0:
        diagnostics["reason"] = "all per-format predictions zero (BF16-only?)"
        return gains, diagnostics

    A_active = A[:, active_cols]

    # Non-negative least squares via SciPy if available, else a small
    # active-set fallback that handles the typical 1-3 unknown case.
    try:
        from scipy.optimize import nnls  # type: ignore
        x, residual = nnls(A_active, measured)
    except Exception:
        x = _fallback_nnls(A_active, measured)
        residual = float(np.linalg.norm(A_active @ x - measured))

    # Clamp away from extreme values that would arise if a frontier point
    # has unexpectedly tiny per-format contribution.
    for j_idx, j in enumerate(active_cols):
        g = float(x[j_idx])
        if not np.isfinite(g):
            g = 1.0
        gains[fmts[j]] = float(min(max(g, floor), ceiling))

    diagnostics["residual"] = float(residual)
    diagnostics["fit_predicted"] = (A @ np.array(
        [gains[f] for f in fmts], dtype=np.float64,
    )).tolist()
    diagnostics["measured_delta_kl"] = measured.tolist()
    return gains, diagnostics


def _fallback_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tiny NNLS substitute used when SciPy isn't available.

    Closed-form least squares followed by a single nonneg projection +
    refit on the active set. Adequate for the 1-3-unknown case typical
    here, not a general NNLS solver.
    """
    if A.shape[1] == 0:
        return np.zeros(0)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(A.shape[1])
    if (x >= 0).all():
        return x
    active = np.where(x >= 0)[0]
    if active.size == 0:
        return np.zeros(A.shape[1])
    x2, *_ = np.linalg.lstsq(A[:, active], b, rcond=None)
    full = np.zeros(A.shape[1])
    full[active] = np.maximum(x2, 0.0)
    return full


def build_module_param_map(model):
    out = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        out[full_name] = (mod, attr)
        bare_name = full_name[:-7] if full_name.endswith(".weight") else full_name
        out[bare_name] = (mod, attr)
        if full_name.startswith("model."):
            out[f"model.language_model.{full_name[len('model.') :]}"] = (mod, attr)
            out[
                f"model.language_model.{bare_name[len('model.') :]}"
                if bare_name.startswith("model.")
                else f"model.language_model.{bare_name}"
            ] = (mod, attr)
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    if xr.std() == 0 or yr.std() == 0:
        return None
    return float(np.corrcoef(xr.astype(np.float64), yr.astype(np.float64))[0, 1])


@torch.no_grad()
def measure_avg_last_token_kl(model, calib_ids: torch.Tensor, ref_log_probs, device: torch.device) -> float:
    kls = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i : i + 1].to(device)
        logits = model(batch).logits[:, -1:, :]
        teacher = ref_log_probs[i][:, -1:, :]
        kls.append(float(kl_divergence(logits, teacher).item()))
    return sum(kls) / max(len(kls), 1)


def apply_recipe_in_place(model, assignment_expanded: dict[str, str], quant_map: dict[str, tuple]):
    originals = {}
    for name, fmt in assignment_expanded.items():
        target = quant_map.get(name)
        if target is None:
            continue
        mod, attr = target
        original = getattr(mod, attr).data.detach().clone()
        originals[name] = (mod, attr, original)
        q = fr.get_format(fmt).quantize_dequantize(original)
        getattr(mod, attr).data.copy_(q.to(device=original.device, dtype=original.dtype))
    return originals


def install_activation_hooks(
    assignment_expanded: dict[str, str],
    quant_map: dict[str, tuple],
):
    module_specs = {}
    skipped = []
    for name, fmt in assignment_expanded.items():
        target = quant_map.get(name)
        if target is None:
            continue
        mod, _attr = target
        spec = fr.get_format(fmt)
        key = id(mod)
        prev = module_specs.get(key)
        if prev is None:
            module_specs[key] = (mod, spec, [name], fmt)
            continue
        prev_mod, prev_spec, prev_names, prev_fmt = prev
        prev_names.append(name)
        if prev_spec.name != spec.name:
            skipped.append(
                {
                    "module": type(mod).__name__,
                    "weights": sorted(prev_names),
                    "formats": sorted({prev_fmt, fmt}),
                }
            )
            module_specs[key] = (prev_mod, None, prev_names, prev_fmt)

    handles = []
    active = []
    for mod, spec, names, fmt in module_specs.values():
        if spec is None:
            continue
        if spec.act_bits is None or spec.act_bits >= 16:
            continue
        quant_fn = spec.activation_quantize_dequantize

        def _pre_hook(_mod, args, kwargs, quant_fn=quant_fn):
            if args:
                x = args[0]
                qx = quant_fn(x)
                args = (qx,) + tuple(args[1:])
            if kwargs and "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = quant_fn(kwargs["hidden_states"])
            return args, kwargs

        handles.append(mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        active.append(
            {
                "module": type(mod).__name__,
                "weights": sorted(names),
                "format": fmt,
            }
        )
    return handles, active, skipped


def restore_in_place(originals: dict):
    for _name, (mod, attr, original) in originals.items():
        getattr(mod, attr).data.copy_(original)


def select_targets(curve: list[dict], mode: str) -> list[int]:
    feasible = [i for i, row in enumerate(curve) if row.get("feasible")]
    if not feasible:
        return []
    if mode == "all":
        return feasible
    if mode == "knee":
        rows = [curve[i] for i in feasible]
        knee_local = kneedle([r["achieved_bits"] for r in rows], [r["predicted_dloss"] for r in rows])
        return [feasible[knee_local]]
    if mode == "baseline,knee,high":
        rows = [curve[i] for i in feasible]
        knee_local = kneedle([r["achieved_bits"] for r in rows], [r["predicted_dloss"] for r in rows])
        picks = {feasible[0], feasible[knee_local], feasible[-1]}
        return sorted(picks)
    raise ValueError(f"unknown selection mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--formats", required=True,
                    help="Comma-separated format names, e.g. NVFP4,MXFP8")
    ap.add_argument("--pareto-targets", default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25")
    ap.add_argument("--selection", default="baseline,knee,high",
                    choices=["baseline,knee,high", "knee", "all"])
    ap.add_argument("--bit-precision", type=float, default=0.001)
    ap.add_argument("--expert-granularity", choices=["layer", "expert"], default="layer")
    ap.add_argument("--no-fused-promote", action="store_true")
    ap.add_argument("--n-calib-samples", type=int, default=4)
    ap.add_argument("--calib-seqlen", type=int, default=128)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    stats, costs, specs_sorted = load_inputs(Path(args.probe), Path(args.costs), fmt_names)
    curve, stats_alloc, _costs_alloc, format_rank = build_curve(
        stats,
        costs,
        specs_sorted,
        [float(x) for x in args.pareto_targets.split(",")],
        args.bit_precision,
        args.no_fused_promote,
        args.expert_granularity,
    )
    selected = select_targets(curve, args.selection)
    if not selected:
        raise SystemExit("no feasible points to calibrate")

    model_arg = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    staged, cleanup = stage_multimodal(model_arg)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = args.device
        device = torch.device(device_str)
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer_kwargs = dict(trust_remote_code=True)
        if Path(staged).exists():
            load_kwargs["local_files_only"] = True
            tokenizer_kwargs["local_files_only"] = True
        if device.type == "cuda":
            load_kwargs["device_map"] = device_str

        model = AutoModelForCausalLM.from_pretrained(
            staged,
            **load_kwargs,
        )
        if device.type != "cuda":
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        quant_map = build_module_param_map(model)

        calib_ids = load_wikitext_calibration(tokenizer, args.n_calib_samples, args.calib_seqlen)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)
        baseline_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)

        results = []
        for idx in selected:
            row = curve[idx]
            assignment = row["assignment"]
            if args.expert_granularity == "layer":
                assignment_expanded = expand_moe_assignment(assignment, stats_alloc)
            else:
                assignment_expanded = assignment

            originals = apply_recipe_in_place(model, assignment_expanded, quant_map)
            hook_handles, active_hooks, skipped_hooks = install_activation_hooks(
                assignment_expanded, quant_map
            )
            try:
                actual_kl = measure_avg_last_token_kl(model, calib_ids, ref_log_probs, device)
            finally:
                for handle in hook_handles:
                    handle.remove()
                restore_in_place(originals)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            counts = {}
            for fmt in assignment.values():
                counts[fmt] = counts.get(fmt, 0) + 1

            results.append({
                "curve_index": idx,
                "target_bits": row["target_bits"],
                "achieved_bits": row["achieved_bits"],
                "predicted_dloss": row["predicted_dloss"],
                "predicted_dloss_by_format": row.get(
                    "predicted_dloss_by_format", {}),
                "actual_last_token_kl": actual_kl,
                "delta_from_baseline_kl": actual_kl - baseline_kl,
                "format_counts": counts,
                "activation_hook_count": len(active_hooks),
                "activation_hook_modules": active_hooks,
                "activation_hook_skipped": skipped_hooks,
            })
            print(
                f"[cal] idx={idx} target={row['target_bits']:.3f} "
                f"achieved={row['achieved_bits']:.3f} pred={row['predicted_dloss']:.4e} "
                f"kl={actual_kl:.4e} hooks={len(active_hooks)} skipped={len(skipped_hooks)}",
                flush=True,
            )

        predicted = np.array([r["predicted_dloss"] for r in results], dtype=np.float64)
        actual = np.array([r["actual_last_token_kl"] for r in results], dtype=np.float64)
        pearson = (
            float(np.corrcoef(predicted, actual)[0, 1])
            if len(results) >= 2 and predicted.std() > 0 and actual.std() > 0
            else None
        )
        spearman = _spearman_corr(predicted, actual)

        calibrated_gains, calibration_diag = fit_calibrated_gains(
            results, baseline_kl,
        )
        if calibrated_gains:
            print(
                f"[cal] fitted gains: "
                f"{ {k: round(v, 4) for k, v in calibrated_gains.items()} } "
                f"(residual={calibration_diag.get('residual', float('nan')):.3e})",
                flush=True,
            )

        out = {
            "model": args.model,
            "formats": fmt_names,
            "pareto_targets": args.pareto_targets,
            "selection": args.selection,
            "bit_precision": args.bit_precision,
            "expert_granularity": args.expert_granularity,
            "baseline_last_token_kl": baseline_kl,
            "correlation_predicted_vs_actual_pearson": pearson,
            "correlation_predicted_vs_actual_spearman": spearman,
            "calibrated_gains": calibrated_gains,
            "calibration_diagnostics": calibration_diag,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[cal] wrote {args.output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
