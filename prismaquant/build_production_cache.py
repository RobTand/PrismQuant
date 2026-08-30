"""Build a production-faithful δw cache for a model checkpoint.

Renders W_tilde[name, fmt] using the export pipeline's activation-aware
passes (GPTQ damp-sweep + scale_sweep on NVFP4; GPTQ damp-sweep on
FP8_DYNAMIC/FP8_E4M3; explicit MX formats only when requested; passthrough
on BF16) and saves a pickle that
PerturbedActivationCache can load via ``production_weight_cache=...``.

By default this standalone CLI renders the explicit ``--formats`` menu for
all quantizable Linears. Pipeline callers should pass ``--render-scope
assignment --render-layer-config layer_config.json`` to render only the
concrete non-BF16 entries the export assignment will consume.

This CLI is GPU-or-bust and refuses CPU execution.

Usage:

    python -m prismaquant.build_production_cache \\
        --model /path/to/model \\
        --output /work/production_cache.pkl \\
        --formats NVFP4 \\
        --n-calib-samples 8 \\
        --calib-seqlen 256

The output pickle is a ``ProductionWeightCache`` keyed by
``(qname, fmt_canonical)``. Payloads are either resident tensors or references
into the configured streaming cache directory.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import (
    iter_quantizable_tensors,
    stage_multimodal,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.model_profiles import detect_profile_with_warning
from prismaquant.mtp_production_cache import (
    fill_profile_mtp_production_cache,
)
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.production_recache import _load_assignment
from prismaquant.production_weight_cache import (
    fill_production_weight_cache,
)
from prismaquant.sensitivity_probe import load_calibration


def _load_col_weights(path, formats) -> dict | None:
    """Load the imatrix pickle for a weighted-family render (re-vet R3).

    Refuses the two ways this can silently produce a confounded cache: a menu
    that contains a weighted-render family with no vector (the render would be
    unweighted while the exporter ships weighted bytes), and a vector supplied
    for a menu that has no weighted family at all (nothing would consume it, so
    the operator's intent is not what happened).
    """
    import pickle

    import torch

    from prismaquant.production_weight_cache import _weighted_render_family

    weighted = sorted({
        fmt for fmt in formats if _weighted_render_family(fmt) is not None
    })
    if not path:
        if weighted:
            raise SystemExit(
                "[build-prod-cache] ERROR: the format menu contains "
                f"weighted-render formats {weighted} but no --col-weights was "
                "supplied. Their exporters ALWAYS render imatrix-weighted, so "
                "an unweighted cache render is not the bytes that ship (the "
                "rendering confound). Pass --col-weights "
                "<work>/artifacts/cb_col_weights.pkl."
            )
        return None
    if not weighted:
        raise SystemExit(
            "[build-prod-cache] ERROR: --col-weights was supplied but the "
            f"format menu {sorted(set(formats))} has no weighted-render "
            "family (CB / GGUF); nothing would consume the vector and every "
            "render would be byte-identical without it. Drop the flag."
        )
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    loaded = {str(k): torch.as_tensor(v) for k, v in raw.items()}
    print(
        f"[build-prod-cache] col-weights: {len(loaded)} entries from {path} "
        f"(weighted-render formats in menu: {weighted})",
        flush=True,
    )
    return loaded


def _explicit_cb_render_context(formats):
    """Resolve CB producer settings once, at the CLI boundary.

    Library render/cache code receives the resulting object explicitly and
    therefore cannot reinterpret a stored cache under a later environment.
    """
    from prismaquant.nvfp4_cb_footprint import (
        cb_serialization_context_from_env,
        is_cb_format,
    )

    cb_formats = sorted({str(fmt) for fmt in formats if is_cb_format(str(fmt))})
    if not cb_formats:
        return None
    if "PRISMAQUANT_CB_MINCHAIN" not in os.environ:
        raise SystemExit(
            "[build-prod-cache] ERROR: ProductionWeightCache producer: "
            "missing explicit CB producer setting "
            "['PRISMAQUANT_CB_MINCHAIN']"
        )
    try:
        return cb_serialization_context_from_env(
            require_explicit=True,
            where="ProductionWeightCache producer",
        )
    except ValueError as exc:
        raise SystemExit(f"[build-prod-cache] ERROR: {exc}") from exc


def _resolve_unit_shard(args):
    """Resolve PRISMAQUANT_UNIT_SHARD once, at the CLI boundary.

    Unset is a byte-identical no-op: nothing downstream changes shape. When
    set, refuse the paths a unit shard cannot mean:

    * packed-MoE experts — ``fill_packed_expert_cache_entries`` enumerates
      and reservoir-samples its own modules off a second shared generator, so
      every shard would render every expert (duplicates the merge would
      reject) with rows that are not the full run's rows;
    * the recache replay (``--recache-layer-config``) — it is a whole-model
      calibration forward with the production weights installed, not a
      per-unit render, and a shard holding a subset of the weights would
      measure a different activation distribution.
    """
    from prismaquant.unit_sharding import SHARD_ENV, resolve_shard_spec

    try:
        spec = resolve_shard_spec()
    except ValueError as exc:
        raise SystemExit(f"[build-prod-cache] ERROR: {exc}") from exc
    if spec is None:
        return None
    if args.recache_layer_config:
        raise SystemExit(
            f"[build-prod-cache] ERROR: {SHARD_ENV}={spec.label} with "
            "--recache-layer-config. The production recache is a whole-model "
            "calibration replay with the selected weights installed, not a "
            "per-unit render; a shard would replay with a subset of the "
            "assignment installed. Shard the render, merge, then recache."
        )
    if args.render_packed_experts:
        raise SystemExit(
            f"[build-prod-cache] ERROR: {SHARD_ENV}={spec.label} with "
            "--render-packed-experts. Packed-MoE expert renders are not "
            "unit-sharded in v1 (see prismaquant/unit_sharding.py); every "
            "shard would render every expert."
        )
    return spec


def _model_has_packed_experts(model: nn.Module, profile) -> bool:
    from prismaquant.sensitivity_probe import _is_packed_experts_module
    return any(
        _is_packed_experts_module(m, profile)
        for _, m in model.named_modules()
    )


def render_format_menu_packed_experts(
    cache,
    model: nn.Module,
    calib_ids,
    formats,
    *,
    profile,
    cache_dir=None,
    module_token_budget: int = 32768,
    render_mode: str = "batched",
) -> dict:
    """Eagerly render packed-MoE experts for a format-menu frontier cache.

    Format-menu builds have no single assignment, but the validated frontier
    must be able to SELECT each expert's format by real KL, so render packed
    experts into the shared cache at the cheap rung real-KL actually picks
    under the route-flip floor — NVFP4 (batched, fast); BF16 is passthrough
    (no render). FP8 experts are rendered LAZILY per-Pareto-point in
    validate_assignments_kl (M4): FP8 is Pareto-dominated on routed experts,
    so eager-rendering all packed tensors at FP8 (~64 GB / ~1 hr, no batched
    path) is wasted — keep FP8 on the menu and render only the rare point
    that proposes it.

    Returns the merged coverage dict (also merged into
    ``cache.metadata['packed_expert_coverage']``).
    """
    from prismaquant.production_weight_cache import (
        fill_packed_expert_cache_entries,
    )
    from prismaquant import format_registry as _fr

    eager_fmts = [
        f for f in formats
        if _fr.canonical_format_name(str(f)) == "NVFP4"
    ]
    merged: dict = {}
    if not eager_fmts:
        print(
            "[build-prod-cache] WARNING: packed-MoE experts present but "
            "no NVFP4 in the format menu; experts not pre-rendered — "
            "frontier real-KL expert selection unavailable.",
            flush=True,
        )
        return merged
    for ef in eager_fmts:
        cov = fill_packed_expert_cache_entries(
            cache, model, calib_ids,
            force_format=ef,
            levers=cache.levers,
            profile=profile,
            cache_dir=cache_dir,
            module_token_budget=module_token_budget,
            render_mode=render_mode,
        )
        if cov:
            merged.update(cov)
    if merged:
        if cache.metadata is None:
            cache.metadata = {}
        cache.metadata.setdefault(
            "packed_expert_coverage", {}).update(merged)
    return merged


def validate_render_assignment_cache_coverage(cache, render_assignment) -> None:
    """Fail if a production render assignment has any uncached non-BF16 entry.

    ``ProductionWeightCache.assignment_keys`` intentionally exempts packed
    expert names for downstream prefetch callers that may run without a packed
    render cache. Build-side validation is stricter: after
    ``fill_packed_expert_cache_entries`` runs, every concrete non-BF16
    assignment entry must be present or the export would fall back/raise later.
    """
    from prismaquant import format_registry as fr

    missing: list[tuple[str, str]] = []
    for qname, fmt in (render_assignment or {}).items():
        fmt_canon = fr.canonical_format_name(str(fmt))
        if fmt_canon == "BF16":
            continue
        if cache.resolve_key(str(qname), fmt_canon) is None:
            missing.append((str(qname), fmt_canon))
    failed = list((cache.failed or {}).keys())
    if missing or failed:
        samples = missing[:5] + failed[:5]
        raise RuntimeError(
            f"ProductionWeightCache assignment coverage failure: "
            f"{len(missing)} misses, {len(failed)} failed "
            f"renders; sample={samples}"
        )


def _filter_assignment_to_include_qnames(
    render_assignment,
    include_qnames: Sequence[str] | None,
) -> dict[str, str]:
    """Project assignment coverage onto the exact rendered stripe."""
    if include_qnames is None:
        return dict(render_assignment or {})
    allowed = {str(qname) for qname in include_qnames}
    return {
        str(qname): fmt
        for qname, fmt in (render_assignment or {}).items()
        if str(qname) in allowed
    }


def _load_cache_calibration(tokenizer, args) -> torch.Tensor:
    if args.dataset:
        return load_calibration(
            tokenizer,
            args.dataset,
            args.n_calib_samples,
            args.calib_seqlen,
            calib_seed=args.calib_seed,
        )
    return load_wikitext_calibration_windowed(
        tokenizer,
        args.n_calib_samples,
        args.calib_seqlen,
        split=args.calib_split,
        seed=args.calib_seed,
    )


def _run_streaming(args, formats, levers, dtype, unit_shard=None) -> int:
    """Streaming per-layer render path for models too large to load whole.

    No whole-model from_pretrained and no calibration forward: dense + packed
    activations come from the probe's --activation-cache-dir, and each decoder
    layer is materialized on demand through the streaming model.
    """
    from prismaquant.streaming_production_cache import (
        fill_production_weight_cache_streaming,
    )
    format_plan = None
    if args.format_plan:
        from prismaquant.source_class_format_plan import load_format_plan

        format_plan = load_format_plan(args.format_plan)

    layer_config = args.render_layer_config or args.recache_layer_config
    if args.render_scope == "assignment" and not layer_config:
        print(
            "[build-prod-cache] FAIL: --streaming requires "
            "--render-layer-config for --render-scope assignment",
            flush=True,
        )
        return 2
    if not args.cache_dir:
        print(
            "[build-prod-cache] FAIL: --streaming requires --cache-dir "
            "(the streamed cache is disk-backed; peak memory is one layer)",
            flush=True,
        )
        return 2
    if not args.activation_cache_dir:
        print(
            "[build-prod-cache] FAIL: --streaming requires "
            "--activation-cache-dir (the probe's per-Linear activation cache)",
            flush=True,
        )
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = require_cuda_hot_path("build_production_cache")
    print(f"[build-prod-cache] streaming device={device}", flush=True)

    render_assignment = (
        _load_assignment(layer_config)
        if args.render_scope == "assignment" else None
    )
    if render_assignment is not None:
        non_bf16 = sum(
            1 for fmt in render_assignment.values()
            if str(fmt).strip().upper() != "BF16"
        )
        print(
            f"[build-prod-cache] streaming assignment render scope: "
            f"{non_bf16} non-BF16 entries from {layer_config}",
            flush=True,
        )
    else:
        print(
            "[build-prod-cache] streaming full format menu: every eligible "
            f"Linear x {len(formats)} requested formats; renders are consumed "
            "synchronously and discarded",
            flush=True,
        )
    render_formats = list(formats)
    if render_assignment is not None:
        render_formats.extend(render_assignment.values())
    col_weights = _load_col_weights(args.col_weights, render_formats)
    cb_serialization_context = _explicit_cb_render_context(render_formats)

    # Streaming consumes a probe activation cache instead of replaying the
    # calibration forward, but its pair stamps still bind the exact token
    # corpus. Tokenization is cheap and ensures selected-assignment rerender
    # cannot silently use a different calibration contract.
    from transformers import AutoTokenizer

    local_only = Path(args.model).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    calib_ids = _load_cache_calibration(tokenizer, args)
    calib_hash = calibration_data_hash(calib_ids)

    include_qnames = None
    if args.include_qnames_file:
        include_path = Path(args.include_qnames_file)
        include_qnames = [
            line.strip()
            for line in include_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not include_qnames:
            print(
                "[build-prod-cache] FAIL: --include-qnames-file contains no "
                "qnames",
                flush=True,
            )
            return 2

    skip_tokens = (
        list(args.skip_qnames) if args.skip_qnames is not None else None
    )
    t0 = time.monotonic()
    cache = fill_production_weight_cache_streaming(
        args.model,
        render_assignment=render_assignment,
        activation_cache_dir=args.activation_cache_dir,
        formats=formats,
        levers=levers,
        cache_dir=args.cache_dir,
        device=device,
        dtype=dtype,
        skip_tokens=skip_tokens,
        expert_render_mode=args.expert_render_mode,
        expert_module_token_budget=args.expert_token_budget,
        h_detail_dir=args.h_detail_dir,
        col_weights=col_weights,
        cb_serialization_context=cb_serialization_context,
        render_scope=args.render_scope,
        retain_rendered=(args.render_scope == "assignment"),
        calibration_hash=calib_hash,
        resume=args.resume,
        max_act_rows=args.max_act_rows,
        include_qnames=include_qnames,
        format_plan=(
            format_plan.formats_by_qname()
            if format_plan is not None else None
        ),
        format_plan_identity=(
            format_plan.identity_sha256
            if format_plan is not None else None
        ),
        unit_shard=unit_shard,
    )
    if render_assignment is not None:
        profile = detect_profile_with_warning(
            args.model,
            entrypoint="build-prod-cache/mtp",
        )
        fill_profile_mtp_production_cache(
            cache,
            args.model,
            profile=profile,
            activation_cache_dir=args.activation_cache_dir,
            formats=formats,
            render_assignment=render_assignment,
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
            max_act_rows=args.max_act_rows,
            h_detail_dir=args.h_detail_dir,
            include_qnames=include_qnames,
            col_weights=col_weights,
            cb_serialization_context=cb_serialization_context,
        )
    elapsed = time.monotonic() - t0

    if unit_shard is not None and render_assignment is not None:
        shard_units = set((cache.metadata or {})["unit_shard"]["unit_names"])
        render_assignment = {
            q: fmt for q, fmt in render_assignment.items() if q in shard_units
        }
    try:
        if render_assignment is not None:
            coverage_assignment = _filter_assignment_to_include_qnames(
                render_assignment, include_qnames,
            )
            validate_render_assignment_cache_coverage(
                cache, coverage_assignment,
            )
        else:
            artifacts = cache.metadata.get("transient_render_artifacts", {})
            expected = int(cache.metadata.get("requested_entries", -1))
            observed = int(artifacts.get("entries", -2))
            if observed != expected or cache.failed:
                raise RuntimeError(
                    "streamed format-menu consumption coverage failure: "
                    f"expected={expected} consumed={observed} "
                    f"failed={len(cache.failed)}"
                )
        print("[build-prod-cache] coverage check passed", flush=True)
    except RuntimeError as e:
        if args.allow_incomplete:
            print(f"[build-prod-cache] WARNING: {e}", flush=True)
            print(
                "[build-prod-cache] --allow-incomplete: writing cache anyway.",
                flush=True,
            )
        else:
            print(f"[build-prod-cache] FAIL: {e}", flush=True)
            return 2

    compacted = (
        cache.compact_for_pickle()
        if hasattr(cache, "compact_for_pickle")
        else 0
    )
    if compacted:
        print(
            f"[build-prod-cache] compacted {compacted} resident cache tensors "
            "back to path references before writing",
            flush=True,
        )
    with open(output_path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[build-prod-cache] wrote {len(cache)} entries to "
        f"{output_path} ({elapsed:.1f}s)",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build production δw cache")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats",
        default="NVFP4",
        help="Comma-separated formats to render. FP8_DYNAMIC is accepted "
        "as an alias for FP8_E4M3 and uses GPTQ damp-sweep. MXFP8/E5M2 "
        "are explicit opt-in research/legacy formats.",
    )
    p.add_argument(
        "--format-plan",
        default=None,
        help="Identity-bound source-class format plan. Streaming format-menu "
        "renders intersect the requested family with each qname's exact "
        "legal menu; assignment scope refuses an illegal planned cell.",
    )
    p.add_argument(
        "--render-scope",
        choices=("format-menu", "assignment"),
        default="format-menu",
        help="format-menu renders every requested format for every eligible "
        "Linear. assignment renders only the concrete non-BF16 entries from "
        "--render-layer-config. The pipeline defaults to assignment to avoid "
        "wasting compute on unused cache entries.",
    )
    p.add_argument(
        "--render-layer-config",
        default=None,
        help="Concrete layer_config.json assignment used when "
        "--render-scope=assignment. Non-BF16 entries are rendered exactly; "
        "BF16 entries are ignored because they do not need cache weights.",
    )
    p.add_argument("--n-calib-samples", type=int, default=8)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration source accepted by sensitivity_probe "
        "(HF dataset id, .jsonl, or .txt). When omitted, preserves the "
        "historical wikitext-2 windowed loader.",
    )
    p.add_argument(
        "--expert-gate-dataset",
        default=None,
        help="Optional corpus DISJOINT from --dataset for the packed-expert "
        "GPTQ-vs-RTN do-no-harm gate. When set, each expert's gate is judged "
        "on this corpus's routed rows (and GPTQ fits on all fit-corpus rows) "
        "instead of a same-corpus held-out slice — a same-domain holdout "
        "cannot catch calibration-domain overfit (the 2026-06-09 35B served "
        "inversion). Same source formats as --dataset.",
    )
    p.add_argument(
        "--expert-gate-samples", type=int, default=None,
        help="Sample count for --expert-gate-dataset (default: --n-calib-samples).",
    )
    p.add_argument(
        "--expert-gate-seqlen", type=int, default=None,
        help="Sequence length for --expert-gate-dataset (default: --calib-seqlen).",
    )
    p.add_argument(
        "--expert-token-budget", type=int, default=32768,
        help="Per-module reservoir budget (tokens) for packed-expert GPTQ "
        "fit activations. CPU-resident, but on unified-memory hosts it still "
        "consumes the shared pool: budget × hidden × 4B × n_modules.",
    )
    p.add_argument(
        "--expert-gate-token-budget", type=int, default=None,
        help="Per-module reservoir budget for the cross-domain gate corpus "
        "(default: --expert-token-budget). The gate only judges (needs "
        "~eval_rows_per_expert routed rows/expert), so this can be much "
        "smaller than the fit budget.",
    )
    p.add_argument(
        "--expert-render-mode", default="batched",
        choices=["batched", "per_expert"],
        help="Packed-expert render path. 'batched' vectorizes GPTQ across "
        "experts (fixed damp, no JSO/act-order; ~13 min/35B). 'per_expert' "
        "runs every expert through render_production_weight — the IDENTICAL "
        "GPTQ+damp_sweep+act_order+JSO stack dense Linears get (production "
        "homogeneity; ~16h/35B).",
    )
    p.add_argument(
        "--render-packed-experts", action="store_true",
        help="Format-menu builds only: eagerly render packed-MoE experts at "
        "the NVFP4 rung so the validated frontier can SELECT expert formats "
        "by real KL (M4). Pass this for the FRONTIER cache build; leave it "
        "off for render-score cost caches, whose expert renders would have "
        "no consumer (~60 GB and hours wasted on a 35B). Assignment-scope "
        "builds render experts from the assignment regardless of this flag.",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument(
        "--max-act-rows",
        type=int,
        default=512,
        help="Max activation rows kept per Linear for GPTQ covariance. "
        "GPTQ is O(in_features^2); rows just need to span the input "
        "subspace well.",
    )
    p.add_argument(
        "--enable",
        default="gptq,static_act_order,joint_scale_opt",
        help="Comma-separated levers to enable. Currently honored: "
        "{none, gptq, static_act_order, joint_scale_opt}. "
        "Use none for RTN-only rendering with no local production levers. "
        "Default `gptq,static_act_order,joint_scale_opt` ships GPTQ "
        "(with the always-on per-Linear damp sweep), static activation "
        "ordering where the format supports it, plus JSO. FP8_DYNAMIC "
        "uses GPTQ damp-sweep; static activation ordering and JSO are "
        "ignored for FP8 because its served representation is per-row "
        "scaled FP8 dynamic. "
        "scale_sweep regresses end-to-end KL on Qwen3-4B and was dropped "
        "from defaults 2026-05-15. "
        "fisher_gptq is an archived legacy name and must not be used for "
        "V1 production artifacts. "
        "Joint NVFP4 sibling globals + calibrated input_global_scale are "
        "computed unconditionally when NVFP4 is in the format menu. "
        "static_act_order applies to production microscaling GPTQ formats "
        "(NVFP4, MXFP4, MXFP8); joint_scale_opt applies only to NVFP4. "
        "MXFP4/MXFP8 use the canonical E8M0 scale rule inside GPTQ when "
        "explicitly requested. "
        "NVFP4 block scaling follows PRISMAQUANT_NVFP4_SCALE_RULE.",
    )
    p.add_argument(
        "--disable",
        default="",
        help="Comma-separated levers to FORCE off (overrides defaults). "
        "Use this to run RTN-only ablations, e.g. --disable gptq,scale_sweep.",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write the cache even if validate_coverage finds missing "
        "(qname, fmt) entries.  Default: fail loudly.  Downstream "
        "consumers running with PRISMAQUANT_STRICT_PRODUCTION_CACHE=1 "
        "will refuse to use an incomplete cache anyway.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to stream per-Linear weight tensors to (one .pt "
        "per (qname, fmt)).  When set, fill peak memory is bounded by "
        "the largest single render rather than the full cache size.  "
        "The pickle becomes a small manifest; PerturbedActivationCache "
        "lazy-loads each weight on first access at hook time.  Required "
        "for arbitrarily-large models (e.g. 27B+ on a 121 GB UMA box).",
    )
    p.add_argument(
        "--h-detail-dir",
        default=None,
        help="Optional h-detail directory from incremental_probe. When "
        "'fisher_gptq' is enabled, g2_per_token vectors from this directory "
        "weight NVFP4 GPTQ/scale-sweep and explicit microscaled-FP8 "
        "objectives.",
    )
    p.add_argument(
        "--skip-qnames",
        nargs="*",
        default=None,
        help="Substrings on qname components that should be EXCLUDED from "
        "the cache fill. Default: the active model profile's pinned_names "
        "(typically lm_head/head). Pass --skip-qnames with no values to "
        "disable this skip.",
    )
    p.add_argument(
        "--include-qnames-file",
        default=None,
        help="Optional newline-delimited qname allowlist. After normal "
        "profile/pinned skips, only qnames in this file are rendered. Used "
        "by staged production-render cost to render FP8_DYNAMIC only for "
        "the high-error NVFP4 tail.",
    )
    p.add_argument(
        "--recache-layer-config",
        default=None,
        help="Optional concrete layer_config.json assignment. When set, "
        "after rendering the cache, replay calibration with those production "
        "weights installed and re-fit activation_max_abs for export.",
    )
    p.add_argument(
        "--recache-microbatch-size",
        type=int,
        default=1,
        help="Calibration microbatch size for the production activation "
        "re-cache replay.",
    )
    p.add_argument(
        "--no-recache-activation-quant",
        action="store_true",
        help="During re-cache, install production weights but leave activation "
        "quantization disabled in replay hooks.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Render one decoder layer at a time on top of the streaming "
        "model (no whole-model from_pretrained) so 100B+ / 295B checkpoints "
        "fit on a 121 GB box. Assignment scope retains only the selected "
        "weights. Format-menu scope is CB-only: it renders and synchronously "
        "scores the complete menu, persists identity-bound scalar receipts, "
        "and discards every rendered tensor. Requires --cache-dir and "
        "--activation-cache-dir; assignment scope additionally requires "
        "--render-layer-config.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a streaming CB build only from exact per-pair identity "
        "sidecars in --cache-dir. Transient menu receipts and retained "
        "assignment shards are both validated fail-closed.",
    )
    p.add_argument(
        "--activation-cache-dir",
        default=None,
        help="Probe activation cache directory. Streaming body renders consume "
        "its per-Linear and per-experts-module rows in place of a fresh "
        "calibration forward. Resident format-menu builds also use its MTP "
        "rows to append profile-synthesized mtp.* Linears; an explicit "
        "non-BF16 mtp.* assignment requires it.",
    )
    p.add_argument(
        "--col-weights",
        default=None,
        help="Per-input-column imatrix pickle ({qname: tensor}, e.g. "
        "artifacts/cb_col_weights.pkl) applied to the weighted-render "
        "families ONLY (CB codebook rungs, GGUF k-quants) — re-vet R3 / CB "
        "Milestone C. Their exporters always render weighted, so without this "
        "a cached-menu render of those formats is unfaithful to the shipped "
        "bytes (the rendering confound the lane gates were written to avoid). "
        "NVFP4/FP8/MX/BF16 renders are bit-identical with or without it.",
    )
    args = p.parse_args(argv)
    if args.format_plan and not args.streaming:
        p.error("--format-plan requires --streaming")
    unit_shard = _resolve_unit_shard(args)

    # Opt-in deterministic CUDA path. The default lever ablations on small
    # models show ~2-4% per-Linear weight variance across re-runs of the
    # same configuration, driven by non-deterministic CUDA reduction order
    # inside the GPTQ Cholesky + U-update. Enable
    # PRISMAQUANT_DETERMINISTIC=1 to force bit-reproducible builds at a
    # ~5-15% throughput cost. Required for any A/B comparison where the
    # expected gain is comparable to or smaller than the noise floor.
    if os.environ.get("PRISMAQUANT_DETERMINISTIC", "0").lower() in {"1", "true", "yes"}:
        import torch as _torch_det
        # cuBLAS workspace fix is the main lever for deterministic matmul
        # reductions; torch.use_deterministic_algorithms(True) additionally
        # forces some kernels into slow paths that produce numerically
        # different results — those differences then cascade into NaN KL
        # downstream even though per-Linear MSE looks healthy. Keep only
        # the cuBLAS workspace + seeded RNGs for now.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch_det.manual_seed(int(args.calib_seed))
        if _torch_det.cuda.is_available():
            _torch_det.cuda.manual_seed_all(int(args.calib_seed))
        print(
            f"[build-prod-cache] deterministic mode ON "
            f"(CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
            f"seed={args.calib_seed}, use_deterministic_algorithms=False)",
            flush=True,
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().upper() for f in args.formats.split(",") if f.strip()]
    levers = {
        name: True for name in (
            x.strip() for x in args.enable.split(",")
        ) if name
    }
    for name in (x.strip() for x in args.disable.split(",")):
        if name:
            levers[name] = False

    dtype = _dtype_from_name(args.dtype)

    if args.streaming:
        return _run_streaming(args, formats, levers, dtype, unit_shard)

    staged, cleanup = stage_multimodal(args.model)
    device = require_cuda_hot_path("build_production_cache")
    print(f"[build-prod-cache] device={device}", flush=True)
    try:
        local_only = Path(staged).exists()
        tokenizer = AutoTokenizer.from_pretrained(
            staged, trust_remote_code=True, local_files_only=local_only,
        )
        calib_ids = _load_cache_calibration(tokenizer, args)
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = "cuda"
        try:
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        except ValueError as exc:
            if "requires `accelerate`" not in str(exc) and "requires accelerate" not in str(exc):
                raise
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
            model.to(device)
        if device.type != "cuda":
            model.to(device)
        model.eval()
        profile = detect_profile_with_warning(
            args.model,
            entrypoint="build-prod-cache",
        )
        # Per-expert-on-disk MoE loaded through a text-only modeling class
        # (e.g. qwen3_5_moe_text) whose WeightsMapper doesn't pack experts:
        # from_pretrained leaves the packed params zero. Restore them from the
        # source so activation-scale calibration (esp. the down_proj input
        # scale, derived from the expert weights) sees the real experts.
        # No-op when experts already loaded correctly.
        from .layer_streaming import fill_packed_experts_from_source
        fill_packed_experts_from_source(model, args.model, profile, progress=True)
        skip_tokens = list(
            args.skip_qnames
            if args.skip_qnames is not None
            else profile.pinned_names()
        )
        qnames: list[str] = []
        skipped: list[str] = []
        for full_name, mod, attr in iter_quantizable_tensors(model, profile):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            # Exact dotted-token match against --skip-qnames substrings.
            tokens = qname.split(".")
            if any(s in tokens for s in skip_tokens):
                skipped.append(qname)
                continue
            qnames.append(qname)
        print(
            f"[build-prod-cache] {len(qnames)} quantizable Linears, "
            f"formats={formats}, levers={sorted(levers)}",
            flush=True,
        )
        if skipped:
            print(
                f"[build-prod-cache] skipped {len(skipped)} qnames matching "
                f"{skip_tokens} (typically pinned-BF16 in polish): "
                f"{skipped if len(skipped) <= 5 else skipped[:5] + ['...']}",
                flush=True,
            )
        include_qnames = None
        if args.include_qnames_file:
            include_path = Path(args.include_qnames_file)
            allowed = {
                line.strip()
                for line in include_path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            before = len(qnames)
            qnames = [q for q in qnames if q in allowed]
            include_qnames = sorted(allowed)
            print(
                f"[build-prod-cache] include-qnames-file={include_path} "
                f"kept {len(qnames)}/{before} qnames",
                flush=True,
            )

        recache_assignment = (
            _load_assignment(args.recache_layer_config)
            if args.recache_layer_config else None
        )
        render_assignment = None
        if args.render_scope == "assignment":
            layer_config = args.render_layer_config or args.recache_layer_config
            if not layer_config:
                print(
                    "[build-prod-cache] FAIL: --render-scope=assignment "
                    "requires --render-layer-config",
                    flush=True,
                )
                return 2
            render_assignment = _load_assignment(layer_config)
            non_bf16 = sum(
                1 for fmt in render_assignment.values()
                if str(fmt).strip().upper() != "BF16"
            )
            print(
                f"[build-prod-cache] assignment render scope: "
                f"{non_bf16} non-BF16 entries from {layer_config}",
                flush=True,
            )
        render_formats = list(formats) + list(
            (render_assignment or {}).values()
        ) + list(
            (recache_assignment or {}).values()
        )
        col_weights = _load_col_weights(args.col_weights, render_formats)
        cb_serialization_context = _explicit_cb_render_context(render_formats)
        t0 = time.monotonic()
        cache = fill_production_weight_cache(
            model, calib_ids, qnames,
            formats=formats,
            render_assignment=render_assignment,
            levers=levers,
            max_act_rows=args.max_act_rows,
            cache_dir=args.cache_dir,
            recache_pass=recache_assignment is not None,
            recache_assignment=recache_assignment,
            recache_profile=profile,
            recache_include_activation_quant=not args.no_recache_activation_quant,
            recache_microbatch_size=args.recache_microbatch_size,
            h_detail_dir=args.h_detail_dir,
            col_weights=col_weights,
            cb_serialization_context=cb_serialization_context,
            unit_shard=unit_shard,
        )
        # R14: stamp the calibration identity onto the cache so every artifact
        # derived from it (production_render_cost's cost table) can be checked
        # for disjointness against the selection split, instead of that
        # guarantee resting on the driver passing the right flag.
        if cache.metadata is None:
            cache.metadata = {}
        cache.metadata["calib_hash"] = calibration_data_hash(calib_ids)
        # Transformers suppresses the Qwen MTP sidecar, so it is absent from
        # the resident CausalLM graph above. Synthesize the profile module and
        # append its exact production renders from the probe's existing MTP
        # activation rows. Concrete non-BF16 MTP assignments fail closed when
        # any module/source/row/cache pair is missing; body-only legacy menu
        # builds remain unchanged when no activation cache was supplied.
        fill_profile_mtp_production_cache(
            cache,
            args.model,
            profile=profile,
            activation_cache_dir=args.activation_cache_dir,
            formats=formats,
            render_assignment=(render_assignment or recache_assignment),
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
            max_act_rows=args.max_act_rows,
            h_detail_dir=args.h_detail_dir,
            include_qnames=include_qnames,
            col_weights=col_weights,
            cb_serialization_context=cb_serialization_context,
        )
        # Render packed-MoE experts through the SAME deliberate path. They are
        # 3-D packed tensors, not nn.Linear, so fill_production_weight_cache
        # skips them; without this they would be RTN'd by omission at export
        # (a severe NVFP4 quality regression — banned). Requires a concrete
        # assignment (which format each expert gets).
        expert_assignment = render_assignment or recache_assignment
        expert_coverage: dict = {}
        if (
            unit_shard is not None
            and expert_assignment is not None
            and _model_has_packed_experts(model, profile)
        ):
            from prismaquant.unit_sharding import SHARD_ENV
            raise SystemExit(
                f"[build-prod-cache] ERROR: {SHARD_ENV}={unit_shard.label} on "
                "a model with packed-MoE experts. Packed-expert renders are "
                "not unit-sharded in v1: every shard would render every "
                "expert. Render this model unsharded."
            )
        if expert_assignment is not None:
            from prismaquant.production_weight_cache import (
                fill_packed_expert_cache_entries,
            )
            gate_calib_ids = None
            if args.expert_gate_dataset:
                gate_calib_ids = load_calibration(
                    tokenizer,
                    args.expert_gate_dataset,
                    args.expert_gate_samples or args.n_calib_samples,
                    args.expert_gate_seqlen or args.calib_seqlen,
                    calib_seed=args.calib_seed,
                )
            expert_coverage = fill_packed_expert_cache_entries(
                cache, model, calib_ids,
                render_assignment=expert_assignment,
                levers=cache.levers,
                profile=profile,
                cache_dir=args.cache_dir,
                module_token_budget=args.expert_token_budget,
                render_mode=args.expert_render_mode,
                gate_calib_ids=gate_calib_ids,
                gate_token_budget=args.expert_gate_token_budget,
                col_weights=col_weights,
                cb_serialization_context=cb_serialization_context,
            )
            if expert_coverage:
                if cache.metadata is None:
                    cache.metadata = {}
                cache.metadata["packed_expert_coverage"] = expert_coverage
        elif args.render_packed_experts and _model_has_packed_experts(
                model, profile):
            render_format_menu_packed_experts(
                cache, model, calib_ids, formats,
                profile=profile,
                cache_dir=args.cache_dir,
                module_token_budget=args.expert_token_budget,
                render_mode=args.expert_render_mode,
            )
        elif _model_has_packed_experts(model, profile):
            print(
                "[build-prod-cache] packed-MoE experts present; format-menu "
                "build without --render-packed-experts leaves them "
                "unrendered (correct for render-score cost caches; the "
                "frontier cache build must pass the flag or the validated "
                "frontier cannot select expert formats).",
                flush=True,
            )

        elapsed = time.monotonic() - t0
        # Strict coverage validation: every (qname, NVFP4) must be present
        # before we ship.  Catches naming-alias mismatches, GPTQ Cholesky
        # failures, and any other silent gaps that would otherwise fall
        # through to RTN at hook time.
        try:
            # A shard is complete when ITS units are complete; the merge
            # tool is what gates the full enumeration.  Narrow by shard
            # first, then by --include-qnames-file, in the same order the
            # streaming path applies them.
            coverage_qnames = qnames
            coverage_assignment = render_assignment
            if unit_shard is not None:
                shard_units = set(
                    (cache.metadata or {})["unit_shard"]["unit_names"]
                )
                coverage_qnames = [q for q in qnames if q in shard_units]
                if coverage_assignment is not None:
                    coverage_assignment = {
                        q: fmt
                        for q, fmt in coverage_assignment.items()
                        if q in shard_units
                    }
            if coverage_assignment is not None:
                coverage_assignment = _filter_assignment_to_include_qnames(
                    coverage_assignment, include_qnames,
                )
                validate_render_assignment_cache_coverage(
                    cache, coverage_assignment)
            else:
                cache.validate_coverage(coverage_qnames, formats)
            print("[build-prod-cache] coverage check passed", flush=True)
        except RuntimeError as e:
            if args.allow_incomplete:
                print(f"[build-prod-cache] WARNING: {e}", flush=True)
                print(
                    "[build-prod-cache] --allow-incomplete: writing cache "
                    "anyway.  Downstream consumers running with "
                    "PRISMAQUANT_STRICT_PRODUCTION_CACHE=1 will refuse "
                    "this cache.",
                    flush=True,
                )
            else:
                print(f"[build-prod-cache] FAIL: {e}", flush=True)
                print(
                    "[build-prod-cache] aborting.  Pass --allow-incomplete "
                    "to write the cache anyway, or fix the underlying "
                    "render failures.",
                    flush=True,
                )
                return 2

        compacted = (
            cache.compact_for_pickle()
            if hasattr(cache, "compact_for_pickle")
            else 0
        )
        if compacted:
            print(
                f"[build-prod-cache] compacted {compacted} resident cache "
                "tensors back to path references before writing",
                flush=True,
            )
        with open(output_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[build-prod-cache] wrote {len(cache)} entries to "
            f"{output_path} ({elapsed:.1f}s)",
            flush=True,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
