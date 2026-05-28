#!/usr/bin/env python3
"""Measure fusion-unit KL costs and synthesize allocator cost tables.

The legacy allocator objective is local: h_trace times an output/weight error
for one Linear at a time. That misses cancellation and damping inside units
that are fused by the serving/export path, especially q/k/v and gate/up. This
module measures the quantity the allocator actually needs:

    KL(teacher || model with one fused decision unit rendered at format F)

The resulting group KL is shared back to member Linears only to preserve the
existing cost.pkl handoff. With fused-sibling aggregation enabled in the
allocator, those shares sum back to the original measured group KL.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.calibration_data import _dtype_from_name
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.kl_measurement import measure_override_set_kl
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.perturbed_x_cache import load_text_model_under_work_root
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.sensitivity_probe import load_calibration
from prismaquant.serving_profiles import resolve_target_profile, serving_profile_names


SCHEMA = "prismaquant.grouped_kl_cost.v1"


@dataclass(frozen=True)
class GroupedKLUnit:
    name: str
    members: tuple[str, ...]


def canonical_cost_name(qname: str) -> str:
    """Canonicalize qnames enough to join probe/cost/group payloads."""
    name = str(qname)
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    prefix = "model.language_model."
    if name.startswith(prefix):
        name = "model." + name[len(prefix):]
    return name


def _recipe_name(full_name: str, attr: str, profile) -> str:
    qname = (
        full_name[:-len(".weight")]
        if attr == "weight" and full_name.endswith(".weight")
        else full_name
    )
    if profile is not None:
        try:
            qname = profile.live_to_recipe_name(qname)
        except Exception:
            pass
    return canonical_cost_name(qname)


def discover_grouped_kl_units(
    model: nn.Module,
    profile=None,
    *,
    include_singletons: bool = True,
    include_pinned: bool = False,
) -> tuple[list[GroupedKLUnit], dict[str, object]]:
    """Return fused-sibling decision units supported by hook KL measurement.

    Only nn.Linear weight parameters are included. Packed MoE expert tensors
    are intentionally skipped here and handled by the baseline-cost fallback
    during cost synthesis.
    """
    grouped: dict[str, list[str]] = {}
    seen_params: set[int] = set()
    skipped: list[dict[str, str]] = []

    for full_name, module, attr in iter_quantizable_tensors(model, profile):
        param = getattr(module, attr, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        pid = id(param)
        if pid in seen_params:
            continue
        seen_params.add(pid)

        qname = _recipe_name(full_name, attr, profile)
        if profile is not None and not include_pinned:
            try:
                if profile.is_pinned_name(qname):
                    skipped.append({"qname": qname, "reason": "pinned"})
                    continue
            except Exception:
                pass
        if not isinstance(module, nn.Linear) or attr != "weight":
            skipped.append({"qname": qname, "reason": "not_nn_linear_weight"})
            continue

        group_key = None
        if profile is not None:
            try:
                group_key = profile.fused_sibling_group(qname)
            except Exception:
                group_key = None
        if group_key is None:
            if not include_singletons:
                skipped.append({"qname": qname, "reason": "singleton"})
                continue
            group_key = qname
        grouped.setdefault(canonical_cost_name(str(group_key)), []).append(qname)

    units = [
        GroupedKLUnit(name=name, members=tuple(sorted(set(members))))
        for name, members in sorted(grouped.items())
        if members
    ]
    diag = {
        "unit_count": len(units),
        "member_count": sum(len(unit.members) for unit in units),
        "skipped": skipped,
    }
    return units, diag


def _member_shapes(model: nn.Module, profile=None) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for full_name, module, attr in iter_quantizable_tensors(model, profile):
        param = getattr(module, attr, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        qname = _recipe_name(full_name, attr, profile)
        shapes.setdefault(qname, tuple(int(v) for v in param.shape))
    return shapes


def _legal_formats_for_unit(
    unit: GroupedKLUnit,
    formats: Sequence[str],
    *,
    shapes: Mapping[str, tuple[int, ...]],
    target_profile: str | None,
) -> list[str]:
    legal: list[str] = []
    for raw_fmt in formats:
        spec = fr.get_format(raw_fmt)
        fmt = fr.canonical_format_name(spec.name)
        if fmt == "BF16":
            legal.append(fmt)
            continue
        ok = True
        for member in unit.members:
            shape = shapes.get(member)
            if shape is None:
                ok = False
                break
            verdict = check_format_applicability(
                shape,
                spec,
                qname=member,
                target_profile=target_profile,
            )
            if not verdict.legal:
                ok = False
                break
        if ok:
            legal.append(fmt)
    return list(dict.fromkeys(legal))


@torch.no_grad()
def capture_reference_log_probs(
    model: nn.Module,
    calib_ids: torch.Tensor,
    *,
    device: torch.device,
    kl_scope: str,
) -> list[torch.Tensor]:
    if kl_scope not in {"last_token", "full_sequence"}:
        raise ValueError("kl_scope must be 'last_token' or 'full_sequence'")
    ref: list[torch.Tensor] = []
    for i in range(int(calib_ids.size(0))):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits
        if kl_scope == "last_token":
            logits = logits[:, -1:, :]
        ref.append(F.log_softmax(logits.float(), dim=-1).cpu())
        del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.type == "cuda" and kl_scope == "last_token":
        ref = [t.to(device) for t in ref]
    elif kl_scope == "full_sequence":
        print(
            "[grouped-kl] full_sequence reference cache stores "
            "n_calib x seqlen x vocab fp32 logprobs; keep calibration small",
            flush=True,
        )
    return ref


def _parse_formats(formats_csv: str) -> list[str]:
    formats = [
        fr.canonical_format_name(item.strip())
        for item in str(formats_csv).split(",")
        if item.strip()
    ]
    return list(dict.fromkeys(formats))


def _parse_levers(csv_value: str) -> dict[str, bool]:
    if not csv_value:
        return {}
    return {item.strip(): True for item in csv_value.split(",") if item.strip()}


def load_production_cache(path: str, *, cache_dir_override: str | None, lru_gb: float):
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    if cache_dir_override and hasattr(cache, "relocate"):
        cache.relocate(cache_dir_override)
    if float(lru_gb) > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(float(lru_gb) * 1024 ** 3))
    return cache


@torch.no_grad()
def measure_grouped_kl_costs(
    model: nn.Module,
    calib_ids: torch.Tensor,
    *,
    formats: Sequence[str],
    profile=None,
    target_profile: str | None,
    production_weight_cache=None,
    work_root: Path,
    kl_scope: str = "full_sequence",
    max_lanes_per_batch: int = 4,
    calib_microbatch_size: int = 1,
    include_activation_quant: bool = True,
    use_cuda_graphs: bool = False,
) -> dict:
    device = next(model.parameters()).device
    units, unit_diag = discover_grouped_kl_units(model, profile)
    shapes = _member_shapes(model, profile)

    labels: list[tuple[str, str]] = []
    overrides: list[dict[str, str]] = []
    unit_formats: dict[str, list[str]] = {}
    for unit in units:
        legal = _legal_formats_for_unit(
            unit,
            formats,
            shapes=shapes,
            target_profile=target_profile,
        )
        unit_formats[unit.name] = legal
        for fmt in legal:
            if fr.canonical_format_name(fmt) == "BF16":
                continue
            labels.append((unit.name, fmt))
            overrides.append({member: fmt for member in unit.members})

    if production_weight_cache is not None:
        # Production grouped KL should be production-faithful, not silently RTN.
        os.environ.setdefault("PRISMAQUANT_STRICT_PRODUCTION_CACHE", "1")
        groups_by_name = {unit.name: unit.members for unit in units}
        required_pairs = sorted({
            (member, fmt)
            for unit_name, fmt in labels
            for member in groups_by_name.get(unit_name, ())
            if fr.canonical_format_name(fmt) != "BF16"
        })
        if required_pairs and hasattr(production_weight_cache, "resolve_key"):
            missing = [
                (qname, fmt)
                for qname, fmt in required_pairs
                if production_weight_cache.resolve_key(qname, fmt) is None
            ]
            if missing:
                raise RuntimeError(
                    "ProductionWeightCache coverage failure for grouped KL: "
                    f"{len(missing)} missing measured pairs; sample={missing[:5]}"
                )

    print(
        f"[grouped-kl] units={len(units)} members={unit_diag['member_count']} "
        f"measurements={len(overrides)} formats={list(formats)} "
        f"scope={kl_scope} target_profile={target_profile}",
        flush=True,
    )
    print("[grouped-kl] capturing BF16 reference logprobs", flush=True)
    t0 = time.monotonic()
    ref_log_probs = capture_reference_log_probs(
        model,
        calib_ids,
        device=device,
        kl_scope=kl_scope,
    )
    print(f"[grouped-kl] reference captured in {time.monotonic() - t0:.1f}s", flush=True)

    values = measure_override_set_kl(
        model,
        {},
        overrides,
        calib_ids,
        ref_log_probs,
        work_root=work_root,
        max_lanes_per_batch=max_lanes_per_batch,
        profile=profile,
        kl_scope=kl_scope,
        calib_microbatch_size=calib_microbatch_size,
        include_activation_quant=include_activation_quant,
        use_cuda_graphs=use_cuda_graphs,
        production_weight_cache=production_weight_cache,
    )

    results: dict[str, dict[str, float]] = {
        unit.name: {"BF16": 0.0}
        for unit in units
    }
    for (unit_name, fmt), value in zip(labels, values):
        results.setdefault(unit_name, {})[fmt] = float(value)

    return {
        "schema": SCHEMA,
        "results": results,
        "groups": {unit.name: list(unit.members) for unit in units},
        "formats": list(formats),
        "unit_formats": unit_formats,
        "kl_scope": kl_scope,
        "diagnostics": unit_diag,
    }


def _legacy_grouped_payload(payload: Mapping) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    groups_out: dict[str, list[str]] = {}
    results_out: dict[str, dict[str, float]] = {}
    groups = payload.get("groups", {})
    results = payload.get("results", {})
    for block, group_map in groups.items():
        block_results = results.get(block, results.get(str(block), {}))
        for group_name, members in group_map.items():
            unit = f"block_{block}.{group_name}"
            groups_out[unit] = [str(m) for m in members]
            per_fmt = block_results.get(group_name, {})
            results_out[unit] = {
                fr.canonical_format_name(fmt): float(value)
                for fmt, value in per_fmt.items()
            }
            results_out[unit].setdefault("BF16", 0.0)
    return groups_out, results_out


def grouped_payload_groups_and_results(
    payload: Mapping,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    schema = str(payload.get("schema", ""))
    if schema == "prismaquant.kl_grouped_probe.v1":
        return _legacy_grouped_payload(payload)
    groups = {
        str(unit): [str(member) for member in members]
        for unit, members in dict(payload.get("groups", {})).items()
    }
    results = {
        str(unit): {
            fr.canonical_format_name(fmt): float(value)
            for fmt, value in dict(per_fmt).items()
        }
        for unit, per_fmt in dict(payload.get("results", {})).items()
    }
    for unit in groups:
        results.setdefault(unit, {}).setdefault("BF16", 0.0)
    return groups, results


def synthesize_grouped_cost_payload(
    grouped_payload: Mapping,
    baseline_cost_payload: Mapping,
    *,
    source_label: str | None = None,
) -> dict:
    groups, results = grouped_payload_groups_and_results(grouped_payload)
    baseline_costs = dict(baseline_cost_payload["costs"])
    formats = [
        fr.canonical_format_name(fmt)
        for fmt in baseline_cost_payload.get("formats", [])
    ]
    member_to_group: dict[str, tuple[str, int]] = {}
    for unit, members_raw in groups.items():
        members = [canonical_cost_name(member) for member in members_raw]
        n = max(len(members), 1)
        for member in members:
            member_to_group[member] = (unit, n)

    grouped_entries = 0
    fallback_entries = 0
    output_costs: dict[str, dict[str, dict]] = {}
    for qname, per_name_raw in baseline_costs.items():
        cname = canonical_cost_name(qname)
        per_name = dict(per_name_raw)
        unit_info = member_to_group.get(cname)
        synthesized: dict[str, dict] = {}
        for fmt in formats:
            fmt_c = fr.canonical_format_name(fmt)
            if fmt_c == "BF16":
                synthesized[fmt_c] = {
                    "predicted_dloss": 0.0,
                    "weight_mse": 0.0,
                    "output_mse": 0.0,
                    "rel_output_mse": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "bf16_zero",
                }
                continue
            if unit_info is not None:
                unit, n_members = unit_info
                per_fmt = results.get(unit, {})
                if fmt_c in per_fmt:
                    synthesized[fmt_c] = {
                        "predicted_dloss": float(per_fmt[fmt_c]) / max(n_members, 1),
                        "weight_mse": 0.0,
                        "output_mse": 0.0,
                        "rel_output_mse": 0.0,
                        "output_mse_measured": False,
                        "cost_source": "grouped_kl_share",
                        "grouped_kl_unit": unit,
                        "grouped_kl_members": int(n_members),
                    }
                    grouped_entries += 1
                    continue
            fallback = None
            for alias in fr.aliases_for(fmt_c):
                if alias in per_name:
                    fallback = dict(per_name[alias])
                    break
            if fallback is None and fmt_c in per_name:
                fallback = dict(per_name[fmt_c])
            if fallback is None:
                fallback = {"error": "missing baseline cost"}
            else:
                fallback["cost_source"] = fallback.get("cost_source", "fallback_baseline")
            synthesized[fmt_c] = fallback
            fallback_entries += 1
        output_costs[qname] = synthesized

    return {
        "schema": "prismaquant.grouped_kl_share_cost.v1",
        "costs": output_costs,
        "formats": formats,
        "meta": {
            "grouped_kl_source": source_label,
            "baseline_schema": baseline_cost_payload.get("schema"),
            "baseline_meta": baseline_cost_payload.get("meta"),
            "kl_scope": grouped_payload.get("kl_scope"),
            "grouped_entries": int(grouped_entries),
            "fallback_entries": int(fallback_entries),
        },
    }


def _load_pickle(path: str | Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="/home/rob/dq-runs/calibration/diverse-v1.jsonl")
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=1024)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--output", required=True, help="Output grouped-KL pickle")
    parser.add_argument("--baseline-cost", default=None)
    parser.add_argument("--output-cost", default=None, help="Allocator-compatible grouped cost.pkl")
    parser.add_argument("--target-profile", choices=serving_profile_names(), default=None)
    parser.add_argument("--kl-scope", choices=["last_token", "full_sequence"], default="full_sequence")
    parser.add_argument("--max-lanes-per-batch", type=int, default=4)
    parser.add_argument("--calib-microbatch", type=int, default=1)
    parser.add_argument("--enable-cuda-graphs", action="store_true")
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument("--candidate-recipe", choices=["production", "raw"], default="production")
    parser.add_argument("--production-weight-cache", default=None)
    parser.add_argument("--production-cache-dir", default=None)
    parser.add_argument("--production-cache-output", default=None)
    parser.add_argument("--production-cache-dir-override", default=None)
    parser.add_argument(
        "--production-cache-levers",
        default="gptq,static_act_order,joint_scale_opt",
    )
    parser.add_argument("--production-cache-max-act-rows", type=int, default=512)
    parser.add_argument("--production-cache-lru-gb", type=float, default=16.0)
    args = parser.parse_args(argv)

    require_cuda_hot_path("grouped_kl_cost", args.device)
    dtype = _dtype_from_name(args.dtype)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = (
        Path(args.work_root)
        if args.work_root
        else out_path.parent / "grouped_kl_work"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    model = load_text_model_under_work_root(
        args.model,
        device=args.device,
        dtype=dtype,
        work_root=work_root,
        device_map=args.device_map,
    )
    try:
        profile = detect_profile(args.model)
    except Exception:
        profile = DefaultProfile()
    target_profile = resolve_target_profile(profile, args.target_profile)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calib_ids = load_calibration(
        tokenizer,
        args.dataset,
        int(args.n_calib_samples),
        int(args.calib_seqlen),
        calib_seed=int(args.calib_seed),
    )

    formats = _parse_formats(args.formats)
    production_cache = None
    if args.candidate_recipe == "production":
        if args.production_weight_cache:
            print(
                f"[grouped-kl] loading production cache {args.production_weight_cache}",
                flush=True,
            )
            production_cache = load_production_cache(
                args.production_weight_cache,
                cache_dir_override=args.production_cache_dir_override,
                lru_gb=float(args.production_cache_lru_gb),
            )
        else:
            units, _diag = discover_grouped_kl_units(model, profile)
            qnames = sorted({member for unit in units for member in unit.members})
            cache_formats = [fmt for fmt in formats if fmt != "BF16"]
            cache_dir = Path(args.production_cache_dir or (work_root / "production_weight_cache"))
            cache_out = Path(args.production_cache_output or (work_root / "production_weight_cache.pkl"))
            print(
                f"[grouped-kl] building production cache qnames={len(qnames)} "
                f"formats={cache_formats} dir={cache_dir}",
                flush=True,
            )
            production_cache = fill_production_weight_cache(
                model,
                calib_ids,
                qnames,
                formats=cache_formats,
                levers=_parse_levers(args.production_cache_levers),
                max_act_rows=int(args.production_cache_max_act_rows),
                cache_dir=cache_dir,
            )
            cache_out.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_out, "wb") as fh:
                pickle.dump(production_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if float(args.production_cache_lru_gb) > 0 and hasattr(production_cache, "enable_lru"):
                production_cache.enable_lru(int(float(args.production_cache_lru_gb) * 1024 ** 3))
            print(f"[grouped-kl] wrote production cache manifest {cache_out}", flush=True)

    payload = measure_grouped_kl_costs(
        model,
        calib_ids,
        formats=formats,
        profile=profile,
        target_profile=target_profile,
        production_weight_cache=production_cache,
        work_root=work_root,
        kl_scope=args.kl_scope,
        max_lanes_per_batch=int(args.max_lanes_per_batch),
        calib_microbatch_size=int(args.calib_microbatch),
        include_activation_quant=not bool(args.no_activation_quant),
        use_cuda_graphs=bool(args.enable_cuda_graphs),
    )
    payload["meta"] = {
        "model": args.model,
        "n_calib_samples": int(args.n_calib_samples),
        "calib_seqlen": int(args.calib_seqlen),
        "calib_seed": int(args.calib_seed),
        "target_profile": target_profile,
        "candidate_recipe": args.candidate_recipe,
        "production_weight_cache": args.production_weight_cache,
        "production_cache_levers": _parse_levers(args.production_cache_levers),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[grouped-kl] wrote {out_path}", flush=True)

    if args.output_cost:
        if not args.baseline_cost:
            raise SystemExit("--output-cost requires --baseline-cost")
        baseline = _load_pickle(args.baseline_cost)
        cost_payload = synthesize_grouped_cost_payload(
            payload,
            baseline,
            source_label=str(out_path),
        )
        cost_path = Path(args.output_cost)
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cost_path, "wb") as fh:
            pickle.dump(cost_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[grouped-kl] wrote allocator cost {cost_path} "
            f"(grouped_entries={cost_payload['meta']['grouped_entries']} "
            f"fallback_entries={cost_payload['meta']['fallback_entries']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
