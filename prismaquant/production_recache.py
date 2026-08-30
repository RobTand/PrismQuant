"""Production-faithful activation re-cache helpers.

The initial production cache is rendered from BF16-upstream activations.
At runtime, upstream Linears are quantized, so downstream activation ranges can
shift.  This module replays calibration with production weights installed and
re-fits the per-Linear ``activation_max_abs`` values consumed by export and
perturbed-X measurement.
"""
from __future__ import annotations

import json
import pickle
import tempfile
import time
import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.decision_units import fused_group_key
from prismaquant.layer_config import load_assignment as _load_assignment
from prismaquant.memory_management import model_device as _model_device
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    calibration_data_hash,
    iter_calibration_forwards,
)
from prismaquant.sensitivity_probe import _prismaquant_temp_parent


def _unify_fused_max_abs(
    max_abs: Mapping[str, float],
    *,
    profile=None,
) -> dict[str, float]:
    groups: dict[str, list[str]] = {}
    for qname, value in max_abs.items():
        if value <= 0:
            continue
        group = fused_group_key(profile, qname) if profile else qname
        groups.setdefault(group, []).append(qname)

    unified: dict[str, float] = {}
    for members in groups.values():
        shared = max(float(max_abs[m]) for m in members)
        for member in members:
            unified[member] = shared
    return unified


def production_cache_keys_for_assignment(
    production_weight_cache,
    assignment: Mapping[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return concrete non-BF16 cache keys and missing assignment entries."""
    if hasattr(production_weight_cache, "assignment_keys"):
        return production_weight_cache.assignment_keys(assignment)

    keys: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for qname, fmt in assignment.items():
        fmt_canon = fr.canonical_format_name(str(fmt))
        if fmt_canon == "BF16":
            continue
        key = (
            production_weight_cache.resolve_key(qname, fmt_canon)
            if hasattr(production_weight_cache, "resolve_key")
            else None
        )
        if key is None:
            missing.append((str(qname), fmt_canon))
            continue
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys, missing


def preload_production_cache_for_assignment(
    production_weight_cache,
    assignment: Mapping[str, str],
    *,
    max_resident_bytes: int | None = None,
    max_workers: int = 4,
    require: bool = False,
    progress: bool = True,
) -> dict[str, object]:
    """Preload the concrete rendered weights needed by ``assignment``.

    This is an explicit guardrail against disk-bound recache replay.  If the
    selected assignment cannot fit inside the resident cache budget, ``require``
    turns that into a hard failure instead of silently streaming from NVMe.
    """
    if hasattr(production_weight_cache, "prefetch_assignment"):
        return production_weight_cache.prefetch_assignment(
            assignment,
            max_resident_bytes=max_resident_bytes,
            max_workers=max_workers,
            require=require,
            progress=progress,
            log_prefix="[prod-recache]",
        )

    keys, missing = production_cache_keys_for_assignment(
        production_weight_cache,
        assignment,
    )
    nbytes = (
        production_weight_cache.estimate_nbytes(keys)
        if hasattr(production_weight_cache, "estimate_nbytes")
        else 0
    )
    budget = (
        int(max_resident_bytes)
        if max_resident_bytes is not None and int(max_resident_bytes) > 0
        else None
    )
    stats: dict[str, object] = {
        "keys": len(keys),
        "missing": len(missing),
        "bytes": int(nbytes),
        "budget_bytes": int(budget or 0),
        "loaded": 0,
        "skipped": False,
    }
    if missing:
        stats["missing_sample"] = missing[:8]
        msg = (
            f"production cache missing {len(missing)} assignment entries; "
            f"sample={missing[:8]}"
        )
        if require:
            raise RuntimeError(msg)
        if progress:
            print(f"[prod-recache] WARNING: {msg}", flush=True)
    if budget is not None and nbytes > budget:
        stats["skipped"] = True
        msg = (
            "production cache preload would exceed resident budget: "
            f"{nbytes / 1024**3:.2f} GiB needed, "
            f"{budget / 1024**3:.2f} GiB budget"
        )
        if require:
            raise RuntimeError(msg)
        if progress:
            print(f"[prod-recache] WARNING: {msg}; skipping preload", flush=True)
        return stats

    if progress:
        print(
            "[prod-recache] preloading production cache: "
            f"{len(keys)} entries, {nbytes / 1024**3:.2f} GiB",
            flush=True,
        )
    loaded = production_weight_cache.prefetch(keys, max_workers=max_workers)
    stats["loaded"] = int(loaded)
    if progress:
        print(
            f"[prod-recache] preloaded {loaded}/{len(keys)} production "
            "cache entries",
            flush=True,
        )
    return stats


@torch.no_grad()
def measure_production_activation_max_abs(
    model: nn.Module,
    calibration_data,
    assignment: Mapping[str, str],
    production_weight_cache,
    *,
    profile=None,
    include_activation_quant: bool = True,
    microbatch_size: int = 1,
    preload_production_cache: bool = False,
    preload_max_bytes: int | None = None,
    preload_max_workers: int = 4,
    require_preload: bool = False,
    progress: bool = True,
    temp_parent: str | Path | None = None,
) -> dict[str, float]:
    """Measure activation max-abs under quantized upstream weights.

    ``assignment`` must be the concrete production assignment that will be
    exported.  Candidate caches with multiple possible formats per Linear are
    intentionally not accepted here because the upstream graph would be
    ambiguous.
    """
    if production_weight_cache is None:
        raise ValueError("production_weight_cache is required for recache")
    if not assignment:
        return {}

    started = time.monotonic()
    device = _model_device(model)
    cal_hash = calibration_data_hash(calibration_data)
    if preload_production_cache:
        preload_production_cache_for_assignment(
            production_weight_cache,
            assignment,
            max_resident_bytes=preload_max_bytes,
            max_workers=preload_max_workers,
            require=require_preload,
            progress=progress,
        )
    tmp_parent = Path(temp_parent) if temp_parent is not None else _prismaquant_temp_parent()
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="prismaquant_recache_",
        dir=str(tmp_parent),
    ) as tmp:
        builder = PerturbedActivationCache(
            model,
            assignment,
            tmp,
            input_rows=0,
            cal_hash=cal_hash,
            profile=profile,
            production_weight_cache=production_weight_cache,
            include_activation_quant=include_activation_quant,
        )
        builder.install()
        try:
            for args, kwargs in iter_calibration_forwards(
                calibration_data,
                device,
                microbatch_size=microbatch_size,
            ):
                call_kwargs = dict(kwargs)
                call_kwargs.setdefault("use_cache", False)
                try:
                    model(*args, **call_kwargs)
                except TypeError:
                    model(*args, **kwargs)
        finally:
            builder.remove()
        measured = dict(builder.max_abs)

    unified = _unify_fused_max_abs(measured, profile=profile)
    if progress:
        print(
            "[prod-recache] measured activation max_abs for "
            f"{len(unified)} Linears in {time.monotonic() - started:.1f}s",
            flush=True,
        )
    return unified


def apply_activation_max_abs_to_cache(
    production_weight_cache,
    activation_max_abs: Mapping[str, float],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Update a ProductionWeightCache with re-fitted activation ranges.

    PACKED-expert scales (3-D tensors: several projection params on ONE
    module plan) are PRESERVED from the cache, never replaced: the recache
    replay's ``_capture`` records max|module INPUT| under every param name of
    a module plan, so a packed experts module would record the hidden-state
    max for BOTH ``gate_up_proj`` and ``down_proj`` — but ``down_proj``'s
    real input is the routed post-SwiGLU intermediate, a different tensor
    entirely. The render-time per-projection scales calibrated by
    ``fill_packed_expert_cache_entries`` (routed rows, correct tensor per
    projection) are the ones the frontier's real KL validated; silently
    swapping them at recache would ship a served ``down_proj`` W4A4 input
    scale selection never measured (principle #8: export must ship the
    bytes/scales selection rode on). Per-expert-INDEXED names
    (``...experts.5.down_proj`` — plain nn.Linear modules, DSv4/MiniMax
    layouts) are NOT preserved: each is its own module plan, so the replay
    capture is the correct per-projection tensor and the re-fit is the whole
    point of this stage.
    """
    from prismaquant.production_weight_cache import (
        is_packed_expert_param_qname,
    )

    previous = dict(getattr(production_weight_cache, "activation_max_abs", {}) or {})
    values = {
        str(k): float(v)
        for k, v in activation_max_abs.items()
        if v > 0 and not is_packed_expert_param_qname(str(k))
    }
    preserved_expert_scales = 0
    for key, prev in previous.items():
        if is_packed_expert_param_qname(str(key)):
            values[str(key)] = float(prev)
            preserved_expert_scales += 1
    production_weight_cache.activation_max_abs = values or None
    production_weight_cache.activation_scales = production_weight_cache.activation_max_abs
    meta = dict(getattr(production_weight_cache, "metadata", {}) or {})
    recache_meta = dict(metadata or {})
    recache_meta.setdefault("status", "applied")
    recache_meta.setdefault("n_activation_max_abs", len(values))
    if preserved_expert_scales:
        recache_meta.setdefault(
            "n_packed_expert_scales_preserved", preserved_expert_scales)
    # Delta over the RE-FITTED subset only — preserved packed-expert keys are
    # identical by construction and would dilute the summary to ~1.0 ratios.
    refit_prev = {
        k: v for k, v in previous.items()
        if not is_packed_expert_param_qname(str(k))
    }
    refit_new = {
        k: v for k, v in values.items()
        if not is_packed_expert_param_qname(str(k))
    }
    delta = activation_max_abs_delta_summary(refit_prev, refit_new)
    if delta:
        recache_meta.setdefault("activation_max_abs_delta", delta)
    meta["activation_recache"] = recache_meta
    production_weight_cache.metadata = meta


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def activation_max_abs_delta_summary(
    before: Mapping[str, float] | None,
    after: Mapping[str, float] | None,
) -> dict[str, float | int]:
    """Summarize how much re-cache moved activation ranges."""
    before = before or {}
    after = after or {}
    ratios: list[float] = []
    for qname, old in before.items():
        new = after.get(qname)
        if old and old > 0 and new and new > 0:
            ratios.append(float(new) / float(old))
    if not ratios:
        return {}

    changed_1pct = sum(abs(r - 1.0) > 0.01 for r in ratios)
    changed_5pct = sum(abs(r - 1.0) > 0.05 for r in ratios)
    return {
        "n_common": len(ratios),
        "n_before": len(before),
        "n_after": len(after),
        "ratio_min": min(ratios),
        "ratio_p05": _percentile(ratios, 0.05),
        "ratio_p50": _percentile(ratios, 0.50),
        "ratio_p95": _percentile(ratios, 0.95),
        "ratio_max": max(ratios),
        "changed_gt_1pct": changed_1pct,
        "changed_gt_5pct": changed_5pct,
    }


def assignment_digest(assignment: Mapping[str, str]) -> str:
    """Stable digest for the concrete assignment used during re-cache."""
    payload = json.dumps(
        {str(k): str(v) for k, v in sorted(assignment.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def recache_production_weight_cache(
    model: nn.Module,
    calibration_data,
    assignment: Mapping[str, str],
    production_weight_cache,
    *,
    profile=None,
    include_activation_quant: bool = True,
    microbatch_size: int = 1,
    preload_production_cache: bool = False,
    preload_max_bytes: int | None = None,
    preload_max_workers: int = 4,
    require_preload: bool = False,
    progress: bool = True,
    write_sidecar: bool = True,
    temp_parent: str | Path | None = None,
) -> dict[str, float]:
    """Measure and apply production-faithful activation max-abs values."""
    max_abs = measure_production_activation_max_abs(
        model,
        calibration_data,
        assignment,
        production_weight_cache,
        profile=profile,
        include_activation_quant=include_activation_quant,
        microbatch_size=microbatch_size,
        preload_production_cache=preload_production_cache,
        preload_max_bytes=preload_max_bytes,
        preload_max_workers=preload_max_workers,
        require_preload=require_preload,
        progress=progress,
        temp_parent=temp_parent,
    )
    apply_activation_max_abs_to_cache(
        production_weight_cache,
        max_abs,
        metadata={
            "include_activation_quant": bool(include_activation_quant),
            "microbatch_size": int(microbatch_size),
            "assignment_sha256": assignment_digest(assignment),
            "assignment_entries": int(len(assignment)),
        },
    )
    cache_dir = getattr(production_weight_cache, "cache_dir", None)
    if write_sidecar and cache_dir:
        sidecar = Path(cache_dir) / "activation_max_abs.json"
        # Write the APPLIED scales (post packed-expert preservation), not the
        # raw replay captures — the raw dict carries the wrong-tensor
        # module-input max under packed down_proj names, and a later
        # shard-resume rebuild merges this sidecar back into the cache
        # wholesale ("sidecar wins"). Persisting the rejected values would
        # re-open the exact resurrection path the apply step closes.
        applied = dict(
            getattr(production_weight_cache, "activation_max_abs", {}) or {})
        sidecar.write_text(json.dumps(applied, indent=2))
        delta = (
            getattr(production_weight_cache, "metadata", {}) or {}
        ).get("activation_recache", {}).get("activation_max_abs_delta")
        if delta:
            delta_sidecar = Path(cache_dir) / "activation_max_abs_delta.json"
            delta_sidecar.write_text(json.dumps(delta, indent=2))
            if progress:
                print(
                    "[prod-recache] activation max_abs after/before "
                    f"ratio p50={delta['ratio_p50']:.4g} "
                    f"p95={delta['ratio_p95']:.4g} "
                    f"max={delta['ratio_max']:.4g}; "
                    f"moved>5%={delta['changed_gt_5pct']}/{delta['n_common']}",
                    flush=True,
                )
    return max_abs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-fit ProductionWeightCache activation scales under "
        "quantized upstream weights."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-config", required=True)
    parser.add_argument("--production-weight-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir-override", default=None)
    parser.add_argument("--production-cache-lru-gb", type=float, default=64.0)
    parser.add_argument(
        "--dataset",
        default="/home/rob/dq-runs/calibration/diverse-v1.jsonl",
    )
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument(
        "--production-cache-prefetch",
        choices=("auto", "off", "require"),
        default="require",
        help="Preload assignment-required rendered weights before replay. "
             "'auto' preloads only when they fit the LRU/resident budget; "
             "'require' fails instead of allowing an NVMe-bound replay.",
    )
    parser.add_argument("--production-cache-prefetch-workers", type=int, default=4)
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="Measure ranges with production weights installed but without "
        "activation quantization in the replay hooks.",
    )
    parser.add_argument(
        "--no-write-sidecar",
        action="store_true",
        help="Do not update activation_max_abs sidecars in the production "
             "cache directory. Use when reusing a shared source cache for "
             "an ablation or smoke run.",
    )
    args = parser.parse_args(argv)
    from prismaquant.unit_sharding import SHARD_ENV, resolve_shard_spec

    shard = resolve_shard_spec()
    if shard is not None:
        raise SystemExit(
            f"[production-recache] ERROR: {SHARD_ENV}={shard.label} is set, "
            "but the recache is a whole-model calibration replay with the "
            "selected production weights installed — not a per-unit render. "
            "A shard holding a subset of the assignment would measure a "
            "different activation distribution, so its activation_max_abs "
            "would not be the full run's. Run this stage unsharded on the "
            "merged cache."
        )

    from transformers import AutoTokenizer

    from prismaquant.calibration_data import _dtype_from_name
    from prismaquant.model_profiles import detect_profile_with_warning
    from prismaquant.perturbed_x_cache import load_text_model_under_work_root
    from prismaquant.sensitivity_probe import load_calibration

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.production_weight_cache, "rb") as fh:
        cache = pickle.load(fh)
    if args.cache_dir_override:
        cache.relocate(args.cache_dir_override)
    if args.production_cache_lru_gb > 0 and hasattr(cache, "enable_lru"):
        cache.enable_lru(int(float(args.production_cache_lru_gb) * 1024**3))

    work_root = Path(args.work_root) if args.work_root else output.parent / "recache_work"
    work_root.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_from_name(args.dtype)
    from prismaquant.gpu_guard import require_cuda_hot_path

    require_cuda_hot_path("production_recache", args.device)
    if args.device_map not in (None, "cuda"):
        raise RuntimeError(
            "production_recache requires a CUDA-resident model. CPU/offload "
            f"device_map={args.device_map!r} is not allowed."
        )
    model = load_text_model_under_work_root(
        args.model,
        device=args.device,
        dtype=dtype,
        work_root=work_root,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calib_ids = load_calibration(
        tokenizer,
        args.dataset,
        args.n_calib_samples,
        args.calib_seqlen,
    )
    assignment = _load_assignment(args.layer_config)
    profile = detect_profile_with_warning(
        args.model,
        entrypoint="production-recache",
    )
    # Per-expert-on-disk MoE loaded via a text-only modeling class leaves the
    # packed experts zero-initialized (from_pretrained can't pack them); restore
    # from source so the activation max-abs recapture (incl. the down_proj input
    # scale) reflects the real routed-expert output. No-op if already loaded.
    from .layer_streaming import fill_packed_experts_from_source
    fill_packed_experts_from_source(model, args.model, profile, progress=True)
    max_abs = recache_production_weight_cache(
        model,
        calib_ids,
        assignment,
        cache,
        profile=profile,
        include_activation_quant=not args.no_activation_quant,
        microbatch_size=args.microbatch_size,
        preload_production_cache=args.production_cache_prefetch != "off",
        preload_max_bytes=(
            getattr(cache, "_lru_max_bytes", 0) or None
        ),
        preload_max_workers=args.production_cache_prefetch_workers,
        require_preload=args.production_cache_prefetch == "require",
        progress=True,
        write_sidecar=not args.no_write_sidecar,
        temp_parent=work_root,
    )
    compacted = (
        cache.compact_for_pickle()
        if hasattr(cache, "compact_for_pickle")
        else 0
    )
    if compacted:
        print(
            f"[prod-recache] compacted {compacted} resident cache tensors "
            "back to path references before writing",
            flush=True,
        )
    with open(output, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[prod-recache] wrote {output} with {len(max_abs)} activation scales",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
