#!/usr/bin/env python3
"""incremental_measure_quant_cost.py — per-(Linear, format) quantization
error, streamed shard-by-shard on top of `layer_streaming`.

This is the unified cost path. There is no whole-model `from_pretrained`
load anymore: we mirror the Step-2 `incremental_probe` architecture —
one `StreamingContext` built once with the head (embed + norm + lm_head
+ rotary) resident and every decoder layer offloaded to disk / on meta.
Each shard installs its decoder layers on demand, runs
`measure_batched_gpu` / `measure_unbatched` + `_measure_packed_experts`
on the matching Linears, writes a per-shard pickle, then unloads.

MTP is folded in as a built-in shard kind: when `--include-mtp` (default
True), a shard's regex like `^mtp\\.layers\\.0\\.` triggers synthesis of
the profile's MTP module (`profile.build_mtp_module`), loading of the
source MTP safetensors (`profile.read_mtp_source_state_dict` /
`profile.load_mtp_state_dict`), enumeration of MTP Linears + packed
experts, then the same measurement pipeline against the MTP activation
cache (the probe writes those activations to the same
`--activation-cache-dir` when its own `--include-mtp` was set).

Small models pay the no-op cost of a cache that can hold every layer
resident; large models drain the cache to disk as needed. The per-shard
pickle format matches `measure_quant_cost.run_cost_pass` — allocator
consumes either unchanged.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

from prismaquant.incremental_shards import (
    annotate_incremental_shard as annotate_cost_shard,
    read_pickle as _read_pickle,
)

# Must be set before the cuda allocator initializes. On Spark's UMA,
# cuda and cpu share one LPDDR5X pool; without `expandable_segments`
# the caching allocator hoards freed blocks, causing the OS to swap
# while torch's bookkeeping still thinks it has headroom.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from . import format_registry as fr
from .incremental_probe import (
    build_shard_schedule,
    load_num_hidden_layers,
)
from .measure_quant_cost import (
    ActivationIndex,
    HDetailIndex,
    _accumulate_result,
    _cb_col_weights_lookup,
    _expert_cost_sample_split,
    _extrapolate_expert_costs,
    _finalize_results,
    _measure_packed_experts,
    _normalize_fisher_output_mse_row_weights,
    cost_payload_provenance,
    cb_render_provenance_for_results,
    h_detail_expected_norm_tokens,
    measure_batched_gpu,
    measure_unbatched,
    prepare_cost_context,
    resolve_cost_target_name,
    start_mem_watchdog,
)
from .streaming_model import (
    StreamingContext,
    _build_streaming_context,
    _classify_shard,
)
from .nvfp4_cb_footprint import (
    cb_serialization_context_from_env,
    validate_cb_cost_provenance,
)


# ---------------------------------------------------------------------------
# Per-shard pickle merge helpers (unchanged public API vs. prior version)
# ---------------------------------------------------------------------------
# The pipeline COST_MODE this table was produced under (re-vet R2 precondition
# (i)). `cost.pkl` is the same path under every mode, so without this stamp a
# re-run silently allocates on the previous mode's estimator.
_PIPELINE_COST_MODE = ""


def _cost_mode_provenance(
    specs: list[fr.FormatSpec] | None = None,
) -> dict:
    out = {"cost_mode": _PIPELINE_COST_MODE} if _PIPELINE_COST_MODE else {}
    if specs:
        out.update(cost_payload_provenance(specs))
    return out


def merge_cost_pickles(paths: list[Path], output_path: Path):
    merged_costs = {}
    merged_formats = None
    shard_metas = []
    cb_render_identities: list[dict[str, object]] = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        costs = data["costs"]
        overlap = set(merged_costs) & set(costs)
        if overlap:
            raise ValueError(f"cost shards overlap on {len(overlap)} entries")
        merged_costs.update(costs)
        if merged_formats is None:
            merged_formats = data.get("formats", [])
        elif list(data.get("formats", [])) != list(merged_formats):
            raise ValueError(
                f"cost shard {path} format menu differs from the first "
                f"shard: {data.get('formats', [])!r} != {merged_formats!r}"
            )
        has_cb = any(
            fr.get_format(name).family in {"nvfp4_cb", "fp8_cb"}
            for name in data.get("formats", [])
        )
        if has_cb:
            validate_cb_cost_provenance(
                data,
                data.get("formats", []),
                context=cb_serialization_context_from_env(
                    require_explicit=True,
                    where="incremental cost merge",
                ),
                where=f"incremental cost shard {path}",
            )
        identity = (data.get("provenance") or {}).get(
            "cb_render_identity"
        )
        if has_cb and costs and not isinstance(identity, dict):
            raise ValueError(
                f"incremental cost shard {path} has measured CB rows but no "
                "source/imatrix-complete render identity"
            )
        if isinstance(identity, dict):
            cb_render_identities.append(identity)
        shard_metas.append(data.get("meta", {}))

    merged_cb_identity = None
    if cb_render_identities:
        from .production_weight_cache import merge_cb_render_identities

        qnames = sorted({
            str(name)
            for identity in cb_render_identities
            for name in identity["col_weights_qnames"]
        })
        current_col_weights = {
            name: _cb_col_weights_lookup(name) for name in qnames
        }
        missing_col = sorted(
            name for name, value in current_col_weights.items()
            if value is None
        )
        if missing_col:
            raise ValueError(
                "incremental cost merge is missing current CB col_weights; "
                f"sample={missing_col[:8]}"
            )
        merged_cb_identity = merge_cb_render_identities(
            cb_render_identities,
            col_weights=current_col_weights,
            where="incremental cost merge",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        h_detail_dirs = set()
        for shard_meta in shard_metas:
            if not isinstance(shard_meta, dict):
                continue
            h_detail_dir = shard_meta.get("h_detail_dir")
            if h_detail_dir is None:
                nested = shard_meta.get("incremental_shard")
                if isinstance(nested, dict):
                    h_detail_dir = nested.get("h_detail_dir")
            if h_detail_dir is not None:
                h_detail_dirs.add(h_detail_dir)
        merged_provenance = _cost_mode_provenance(
            [fr.get_format(name) for name in (merged_formats or [])]
        )
        if merged_cb_identity is not None:
            merged_provenance.update({
                "cb_serialized_payload": dict(
                    merged_cb_identity["cb_serialized_payload"]
                ),
                "cb_render_identity": merged_cb_identity,
            })
        pickle.dump({
            "costs": merged_costs,
            "formats": merged_formats or [],
            "provenance": merged_provenance,
            "meta": {
                "incremental": True,
                "n_shards": len(paths),
                "shards": shard_metas,
                "h_detail_dir": (
                    next(iter(h_detail_dirs))
                    if len(h_detail_dirs) == 1
                    else None
                ),
            },
        }, f)


def _expected_cost_shard_meta(*,
                              model: str,
                              probe_path: Path,
                              linear_include: str,
                              shard_idx: int,
                              activation_cache_dir: str,
                              mode: str,
                              chunk_size: int,
                              h_detail_dir: str | None,
                              formats: list[str],
                              render_path: str = "registry",
                              n_linears_expected: int = 0) -> dict[str, Any]:
    meta = {
        "model": model,
        "probe": str(probe_path),
        "activation_cache_dir": str(Path(activation_cache_dir)),
        "linear_include": linear_include,
        "mode": mode,
        "chunk_size": chunk_size,
        "h_detail_dir": str(Path(h_detail_dir)) if h_detail_dir else None,
        "shard_idx": shard_idx,
        "formats": list(formats),
        "n_linears_expected": int(n_linears_expected),
        # Persisted in each shard's annotated metadata, so changing the FP4
        # layout or codebook-sharing policy invalidates shard reuse before a
        # merged cost table can be stamped with a different identity.
        **cost_payload_provenance([fr.get_format(name) for name in formats]),
    }
    if render_path != "registry":
        meta["render_path"] = render_path
    return meta


def _scheduled_probe_targets(stats: dict[str, Any],
                             shard_regexes: list[str]) -> set[str]:
    """Return probe rows that are actually reachable by the shard schedule."""
    compiled = [re.compile(expr) for expr in shard_regexes]
    return {
        name for name in stats
        if any(expr.search(name) for expr in compiled)
    }


def cost_shard_is_reusable(path: Path, expected_meta: dict[str, Any]) -> bool:
    try:
        data = _read_pickle(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "costs" not in data or "meta" not in data:
        return False
    if not isinstance(data["costs"], dict):
        return False
    if data["costs"] and any(
        fr.get_format(name).family in {"nvfp4_cb", "fp8_cb"}
        for name in data.get("formats", [])
    ):
        # A reusable measured-CB shard would need its exact decoded source
        # tensors reloaded and re-hashed before acceptance. The skip path has
        # deliberately not materialized that layer yet, so fail closed and
        # rebuild. Empty visual/MTP slots remain reusable because they carry
        # no CB row/value claim.
        return False
    # Empty-costs guard: a shard whose target regex is known to match at
    # least one probed Linear but produced zero entries is the signature
    # of an earlier buggy run (pre-fix lm_head / stale meta). Force a
    # recompute so the merge doesn't silently carry forward the hole.
    expected_n = expected_meta.get("n_linears_expected")
    if expected_n is not None and expected_n > 0 and len(data["costs"]) == 0:
        return False
    meta = dict(data.get("meta", {}))
    meta.update(meta.get("incremental_shard", {}))
    actual_formats = data.get("formats", [])
    if list(actual_formats) != list(expected_meta.get("formats", [])):
        return False
    for key, expected in expected_meta.items():
        if key in ("formats", "n_linears_expected"):
            continue
        if meta.get(key) != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Shared measurement step: run the configured measure path + packed experts
# against a nn.Module whose relevant Linears have already been materialized.
# Returns the final results dict keyed by canonical Linear name.
# ---------------------------------------------------------------------------
def _run_cost_measurement(
    module: nn.Module,
    *,
    act_cache: ActivationIndex,
    target_names: set[str],
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    log_prefix: str,
    profile=None,
    render_path: str = "registry",
) -> dict:
    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "batched" if device.startswith("cuda") else "unbatched"
    render_path = str(render_path or "registry")
    print(f"{log_prefix} mode={chosen_mode} render_path={render_path} "
          f"targets={len(target_names)}", flush=True)

    if render_path == "production":
        results = _measure_production_render_dense(
            module,
            act_cache=act_cache,
            target_names=target_names,
            specs=specs,
            device=device,
            dtype=dtype,
            h_detail=h_detail,
            profile=profile,
            log_prefix=log_prefix,
        )
    elif chosen_mode == "batched":
        results = measure_batched_gpu(
            module, act_cache, target_names, specs, device, dtype,
            chunk_size=chunk_size, h_detail=h_detail, profile=profile)
    else:
        results = measure_unbatched(
            module, act_cache, target_names, specs, device, dtype,
            h_detail=h_detail, profile=profile)

    packed_accum: dict[str, dict] = {}
    _measure_packed_experts(
        module, target_names, specs, device, dtype, packed_accum,
        act_cache=act_cache, h_detail=h_detail, profile=profile)
    if packed_accum:
        results.update(_finalize_results(packed_accum))
        print(f"{log_prefix} measured {len(packed_accum)} packed-expert tensors",
              flush=True)

    return results


def _production_cost_render_levers() -> dict[str, object]:
    return {
        "gptq": True,
        "joint_scale_opt": True,
        "scale_sweep": False,
        "static_act_order": True,
        "weighted_vq": True,
        "nvfp4_scale_rule": os.environ.get(
            "PRISMAQUANT_NVFP4_SCALE_RULE",
            "static_6",
        ),
    }


def _measure_production_render_dense(
    module: nn.Module,
    *,
    act_cache: ActivationIndex,
    target_names: set[str],
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    h_detail: "HDetailIndex | None",
    profile=None,
    log_prefix: str,
) -> dict:
    """Measure dense Linear costs using the production render primitive.

    This path is deliberately one-Linear-at-a-time: production render may run
    GPTQ and progressive gates that depend on the target's activation rows.
    Packed experts are still measured by ``_measure_packed_experts`` in the
    caller, matching the current split-export path for packed MoE experts.
    """
    from .production_weight_cache import render_production_weight

    dev = torch.device(device)
    accum: dict[str, dict[str, dict]] = {}
    render_gate_meta: dict[tuple[str, str], dict[str, object]] = {}
    processed = 0
    tstart = time.time()
    # Stratified per-expert subsample (PRISMAQUANT_EXPERT_COST_SAMPLE) —
    # the same lever measure_unbatched / measure_batched_gpu apply. Under
    # the default COST_MODE=production-render-score this is the path
    # DSv4's per-expert Linears (256 experts x 3 projections x 43 layers)
    # actually take, so without wiring it here the lever silently does
    # nothing. Skipped experts get their group's sampled mean after
    # finalize; export still quantizes every expert exactly.
    measure_names, expert_extrapolate = _expert_cost_sample_split(target_names)
    n_total = len(measure_names)
    if expert_extrapolate:
        print(f"{log_prefix} production-render expert sample: "
              f"{n_total} measured, {len(expert_extrapolate)} extrapolated",
              flush=True)
    levers = _production_cost_render_levers()
    cb_context = (
        cb_serialization_context_from_env(
            require_explicit=True,
            where="incremental production render cost",
        )
        if any(
            spec.family in {"nvfp4_cb", "fp8_cb"} for spec in specs
        )
        else None
    )

    for name, mod in module.named_modules():
        canonical_name = resolve_cost_target_name(name, target_names, profile)
        if not isinstance(mod, nn.Linear) or canonical_name not in measure_names:
            continue
        if canonical_name not in act_cache:
            continue

        W = mod.weight.detach().to(device=dev, dtype=dtype)
        X_cpu, row_indices = act_cache.load_with_row_indices(canonical_name)
        X = X_cpu.to(device=dev, dtype=dtype)
        y_ref = X @ W.T
        ref_energy = float(y_ref.float().pow(2).mean().item())

        h_full = None
        gq_rows = None
        if h_detail is not None and canonical_name in h_detail:
            blob = h_detail.load_blob(canonical_name)
            try:
                h_full = HDetailIndex.h_diag_from_blob(blob).to(dev).float()
                if h_full.shape != W.shape:
                    h_full = None
            except Exception:
                h_full = None
            gq_rows = _normalize_fisher_output_mse_row_weights(
                blob.get("g2_per_token") if isinstance(blob, dict) else None,
                row_indices,
                int(X.shape[0]),
                dev,
            )

        activations = {canonical_name: X.detach()}
        for spec in specs:
            fmt = fr.canonical_format_name(spec.name)
            try:
                if fmt == "BF16":
                    W_hat = spec.quantize_dequantize(W.clone())
                    gate_trace = None
                else:
                    gate_trace = []
                    W_hat = render_production_weight(
                        W.clone(),
                        fmt,
                        qname=canonical_name,
                        activations=activations,
                        levers=levers,
                        fisher_row_weights=gq_rows,
                        col_weights=(
                            _cb_col_weights_lookup(canonical_name)
                            if spec.family in {"nvfp4_cb", "fp8_cb"}
                            else None
                        ),
                        cb_serialization_context=cb_context,
                        gate_trace=gate_trace,
                    )
                    gptq_steps = [
                        step for step in gate_trace
                        if isinstance(step, dict)
                        and step.get("mechanism") == "gptq"
                    ]
                    if gptq_steps:
                        step = gptq_steps[-1]
                        render_gate_meta[(canonical_name, spec.name)] = {
                            "render_gate_candidate": step.get("candidate"),
                            "render_gate_package": list(
                                step.get("package", [])
                                if isinstance(step.get("package"), list)
                                else []
                            ),
                            "render_gate_accepted": bool(
                                step.get("accepted", False)
                            ),
                            "render_gate_candidates": [
                                {
                                    "label": cand.get("label"),
                                    "score": cand.get("score"),
                                    "package": cand.get("package", []),
                                }
                                for cand in step.get("candidates", [])
                                if isinstance(cand, dict)
                            ],
                        }
                X_hat = spec.activation_quantize_dequantize(X.clone())
                err = (W - W_hat).float()
                y_q = X_hat @ W_hat.T
                y_err_sq = (y_ref - y_q).float().pow(2)
                fisher_output_mse = None
                if gq_rows is not None:
                    fisher_output_mse = float(
                        (y_err_sq * gq_rows.unsqueeze(1)).mean().item()
                    )
                predicted_dloss = None
                if h_full is not None:
                    predicted_dloss = float(0.5 * (h_full * err.pow(2)).sum().item())
                _accumulate_result(
                    accum,
                    canonical_name,
                    spec.name,
                    float(err.pow(2).mean().item()),
                    float(y_err_sq.mean().item()),
                    float(y_err_sq.mean().item()) / max(ref_energy, 1e-12),
                    predicted_dloss=predicted_dloss,
                    fisher_output_mse=fisher_output_mse,
                )
            except Exception as e:
                accum.setdefault(canonical_name, {})[spec.name] = {
                    "error": str(e),
                    "render_path": "production",
                }

        processed += 1
        if processed % 16 == 0:
            elapsed = time.time() - tstart
            eta = elapsed / processed * max(n_total - processed, 0)
            print(f"{log_prefix} production-render {processed}/{n_total} "
                  f"eta={eta:.0f}s", flush=True)

    out = _finalize_results(accum)
    # Fill skipped experts before the render_path stamp so extrapolated
    # rows carry the same provenance as their sampled sources.
    _extrapolate_expert_costs(out, expert_extrapolate)
    for per_name in out.values():
        for entry in per_name.values():
            if isinstance(entry, dict):
                entry.setdefault("render_path", "production")
    for (name, fmt), meta in render_gate_meta.items():
        entry = out.get(name, {}).get(fmt)
        if isinstance(entry, dict) and "error" not in entry:
            entry.update(meta)
    return out


# ---------------------------------------------------------------------------
# Body / lm_head shard runner — install the decoder layers matched by the
# shard regex, run the cost pipeline on Linears (and packed experts) in
# them, write a shard pickle, unload.
# ---------------------------------------------------------------------------
def _run_body_cost_shard(
    ctx: StreamingContext,
    *,
    linear_include: str,
    shard_kind: str,
    probe_stats: dict[str, dict],
    act_cache: ActivationIndex,
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    output_path: str,
    model_name: str,
    probe_path: str,
    render_path: str = "registry",
):
    model = ctx.model
    from .model_profiles import DeadVendoredOverrideError, profile_from_model
    try:
        profile = profile_from_model(model)
    except DeadVendoredOverrideError:
        # This shard is about to price quantization for real. A dead override
        # means the layers it installs and measures came from UPSTREAM
        # modelling code, so every number it writes is priced against the
        # wrong model -- refuse rather than shard on (#202).
        raise
    except Exception:
        profile = None
    num_layers = ctx.num_layers
    layers_prefix = ctx.layers_prefix

    # Figure out which decoder layers (by index) this shard touches.
    # lm_head / purely-top-level shards don't need any decoder layer
    # installed — their target is a resident module already materialized
    # at context setup.
    inc = re.compile(linear_include)
    body_layers_needed: set[int] = set()
    if shard_kind == "body":
        for L in range(num_layers):
            probe_key = f"{layers_prefix}{L}."
            # Any probe stat whose key starts with this layer prefix and
            # matches the include regex means we need the layer installed.
            for name in probe_stats:
                if name.startswith(probe_key) and inc.search(name):
                    body_layers_needed.add(L)
                    break

    # Pre-filter the target set to what actually matches this shard.
    # target_names partitions into:
    #   - body-layer-scoped Linears: matched via the shard regex and
    #     covered by the install loop below.
    #   - resident Linears (lm_head, root-level projections): matched
    #     by the shard regex but with no decoder-layer prefix. They
    #     are already resident on device from `_build_streaming_context`
    #     (head modules were pinned with resident_device), so no install
    #     is needed — `model.get_submodule(name)` during measurement
    #     finds them directly.
    target_names: set[str] = {
        n for n in probe_stats if inc.search(n)
    }
    if not target_names:
        print(f"[incremental-cost] shard has no matching Linears "
              f"(include={linear_include!r}); writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "provenance": _cost_mode_provenance(specs),
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": shard_kind,
                },
            }, f)
        return

    # Install the relevant decoder layers (body path) or none (lm_head
    # shard — resolver has nothing to install at layer granularity;
    # lm_head is already resident from the context setup).
    installed = []
    pressure_trim_bytes = 0
    ctx.layer_cache.set_priority_layers(body_layers_needed)
    ctx.configure_runtime_pressure_floor()
    try:
        t_install = time.time()
        for L in sorted(body_layers_needed):
            ctx.install(L)
            installed.append(L)
        if installed:
            print(f"[incremental-cost] installed layers {installed[0]}..{installed[-1]} "
                  f"({len(installed)}) in {time.time()-t_install:.1f}s",
                  flush=True)

        results = _run_cost_measurement(
            model,
            act_cache=act_cache,
            target_names=target_names,
            specs=specs,
            device=device,
            dtype=dtype,
            mode=mode,
            chunk_size=chunk_size,
            h_detail=h_detail,
            log_prefix=f"[incremental-cost/{shard_kind}]",
            profile=profile,
            render_path=render_path,
        )
        shard_provenance = cb_render_provenance_for_results(
            model,
            results,
            specs,
            profile=profile,
            render_levers=(
                _production_cost_render_levers()
                if render_path == "production" else None
            ),
            where=f"incremental {shard_kind} CB cost shard",
        )
    finally:
        for L in installed:
            pressure_trim_bytes += int(ctx.unload(L) or 0)
        ctx.layer_cache.set_priority_layers(set())
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if pressure_trim_bytes:
        print(f"[incremental-cost] pressure-trimmed "
              f"{pressure_trim_bytes/(1024**3):.1f} GB from layer cache",
              flush=True)

    missing_from_results = [n for n in target_names if n not in results]
    if missing_from_results:
        print(f"[incremental-cost] WARNING: {len(missing_from_results)} Linears "
              f"produced no measurement (cache miss or skipped)", flush=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "provenance": {
                **_cost_mode_provenance(),
                **shard_provenance,
            },
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "mode": ("batched" if mode == "auto" and device.startswith("cuda")
                         else ("unbatched" if mode == "auto" else mode)),
                "shard_kind": shard_kind,
                "render_path": render_path,
            },
        }, f)
    print(f"[incremental-cost] wrote {out_path} ({len(results)} entries)",
          flush=True)


# ---------------------------------------------------------------------------
# MTP shard runner — synthesize MtpModule, load `mtp.*` weights from
# safetensors, enumerate targets, run the same measurement pipeline. The
# body context provides the text_config + device/dtype; lm_head is the
# body's resident output embedding.
# ---------------------------------------------------------------------------
def _run_mtp_cost_shard(
    ctx: StreamingContext,
    *,
    model_path: str,
    linear_include: str,
    probe_stats: dict[str, dict],
    act_cache: ActivationIndex,
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    output_path: str,
    model_name: str,
    probe_path: str,
    render_path: str = "registry",
):
    inc = re.compile(linear_include)
    from .model_profiles import profile_from_model
    # The MTP module layout + its checkpoint prefix are profile
    # decisions, so this lookup is no longer best-effort: without a
    # profile there is nothing to build. (The `profile` local is also
    # passed to the packed-expert helpers further down, which do
    # tolerate None — but reaching this shard at all means has_mtp().)
    profile = profile_from_model(ctx.model)
    # Prune the probe stats to this MTP shard's regex before building
    # anything — if there's nothing to measure, emit an empty pickle.
    shard_targets: set[str] = {n for n in probe_stats if inc.search(n)}
    if not shard_targets:
        print(f"[incremental-cost/mtp] shard has no matching MTP tensors "
              f"(include={linear_include!r}); writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "provenance": _cost_mode_provenance(specs),
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": "mtp",
                },
            }, f)
        return

    device_t = torch.device(device)
    text_config = ctx.model.config
    inner_mtp = profile.build_mtp_module(text_config)
    if inner_mtp is None:
        raise RuntimeError(
            f"profile '{profile.name}' declares has_mtp() but "
            f"build_mtp_module() returned None — the MTP cost shard "
            f"cannot be measured. Either implement build_mtp_module() or "
            f"set has_mtp() -> False.")
    mtp_wrapper = nn.Module()
    mtp_wrapper.add_module("mtp", inner_mtp)
    mtp_wrapper.to(device=device_t, dtype=dtype)
    mtp_wrapper.eval()

    raw = profile.read_mtp_source_state_dict(model_path)
    missing, extra = profile.load_mtp_state_dict(inner_mtp, raw)
    loaded = len(raw) - len(missing)
    print(f"[incremental-cost/mtp] loaded {loaded}/{len(raw)} mtp weights "
          f"(missing={len(missing)}, module_params_unset={len(extra)})",
          flush=True)

    for p in mtp_wrapper.parameters():
        p.requires_grad_(False)

    # Enumerate the MTP module's Linears + packed experts to intersect
    # with the shard's include regex AND with what the probe actually
    # tracked (so we don't measure things the probe never instrumented).
    mtp_linears: set[str] = {
        n for n, m in mtp_wrapper.named_modules()
        if isinstance(m, nn.Linear) and not n.endswith(".mlp.gate")
    }
    from .sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )
    for name, module in mtp_wrapper.named_modules():
        if not _is_packed_experts_module(module, profile):
            continue
        for pn in _packed_experts_param_names(module, profile):
            mtp_linears.add(f"{name}.{pn}")

    target_names = shard_targets & mtp_linears
    if not target_names:
        print(f"[incremental-cost/mtp] shard include regex {linear_include!r} "
              f"matches probe stats but not live MTP module; writing empty pickle",
              flush=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "costs": {},
                "formats": [s.name for s in specs],
                "provenance": _cost_mode_provenance(specs),
                "meta": {
                    "model": model_name,
                    "probe": probe_path,
                    "n_linears": 0,
                    "mode": mode,
                    "shard_kind": "mtp",
                },
            }, f)
        del mtp_wrapper, inner_mtp, raw
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return

    try:
        results = _run_cost_measurement(
            mtp_wrapper,
            act_cache=act_cache,
            target_names=target_names,
            specs=specs,
            device=device,
            dtype=dtype,
            mode=mode,
            chunk_size=chunk_size,
            h_detail=h_detail,
            log_prefix="[incremental-cost/mtp]",
            profile=profile,
            render_path=render_path,
        )
        shard_provenance = cb_render_provenance_for_results(
            mtp_wrapper,
            results,
            specs,
            profile=profile,
            render_levers=(
                _production_cost_render_levers()
                if render_path == "production" else None
            ),
            where="incremental MTP CB cost shard",
        )
    finally:
        del mtp_wrapper, inner_mtp, raw
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "provenance": {
                **_cost_mode_provenance(),
                **shard_provenance,
            },
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "mode": ("batched" if mode == "auto" and device.startswith("cuda")
                         else ("unbatched" if mode == "auto" else mode)),
                "shard_kind": "mtp",
                "render_path": render_path,
            },
        }, f)
    print(f"[incremental-cost/mtp] wrote {out_path} ({len(results)} entries)",
          flush=True)


# ---------------------------------------------------------------------------
# Visual shard — text-only staging strips visual blocks, so we emit an
# empty pickle matching the slot so merge layout stays consistent with
# the probe's shard list.
# ---------------------------------------------------------------------------
def _write_empty_cost_shard(
    output_path: str, *, shard_kind: str,
    specs: list[fr.FormatSpec], model_name: str, probe_path: str,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "costs": {},
            "formats": [s.name for s in specs],
            "provenance": _cost_mode_provenance(specs),
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": 0,
                "mode": "skipped",
                "shard_kind": shard_kind,
            },
        }, f)


# ---------------------------------------------------------------------------
# Visual cost shard runner — Phase 2 multimodal support.
# Loads the multimodal-staged model (vision_config preserved) and runs
# `measure_batched_gpu` / `measure_unbatched` against cached activations
# for the visual Linears matched by this shard's regex. The 35B visual
# tower is ~1 GB BF16; the full 35B model fits in 128 GB. On 122B-scale
# models the whole-model load OOMs and we gracefully emit an empty shard
# so the allocator's --visual-format override can take over.
# ---------------------------------------------------------------------------
def _run_visual_cost_shard(
    *,
    model_path: str,
    linear_include: str,
    probe_stats: dict[str, dict],
    act_cache: "ActivationIndex",
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    mode: str,
    chunk_size: int,
    h_detail: "HDetailIndex | None",
    output_path: str,
    model_name: str,
    probe_path: str,
    mm_ctx: "StreamingContext | None" = None,
    mm_offload_folder: str | None = None,
    render_path: str = "registry",
) -> "StreamingContext | None":
    """Measure per-(visual-Linear, format) cost using an incremental
    multimodal streaming context (visual tower fully resident; body
    decoder layers on meta — they are never touched by the visual cost
    measurement, which operates on activation cache + resident visual
    Linear weights only).

    Intersects `linear_include` with the probe stats and with live
    visual Linears from the streaming context, runs the shared
    measurement pipeline, writes the shard pickle.

    If `mm_ctx` is provided, reuses it (built once and shared across
    every visual shard in the run). Otherwise builds a multimodal
    context on first call (needs `mm_offload_folder`). Returns the
    (possibly newly-built) context for the caller to cache.

    Returns None on failure (empty pickle written, allocator's
    --visual-format override takes over).
    """
    inc = re.compile(linear_include)
    shard_targets = {n for n in probe_stats if inc.search(n)}
    if not shard_targets:
        print(f"[incremental-cost/visual] shard has no matching visual "
              f"tensors in probe stats (include={linear_include!r}); "
              f"writing empty pickle", flush=True)
        _write_empty_cost_shard(
            output_path, shard_kind="visual", specs=specs,
            model_name=model_name, probe_path=probe_path,
        )
        return mm_ctx

    # Build the multimodal streaming context once on first visual shard,
    # then reuse across subsequent visual shards. Visual tower is fully
    # resident (2-3 GB even at 122B scale); body decoder layers live on
    # meta and are never materialized by the visual cost path.
    if mm_ctx is None:
        if mm_offload_folder is None:
            raise ValueError(
                "_run_visual_cost_shard: mm_ctx=None requires mm_offload_folder")
        print(f"[incremental-cost/visual] building multimodal streaming "
              f"context for {len(shard_targets)} target tensors "
              f"(shard={linear_include!r})", flush=True)
        try:
            mm_ctx = _build_streaming_context(
                model_path,
                device=torch.device(device),
                dtype=dtype,
                offload_folder=mm_offload_folder,
                log_prefix="[incremental-cost/visual]",
                multimodal=True,
                visual_requires_grad=False,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "oom" in msg
                    or isinstance(e, (torch.cuda.OutOfMemoryError,
                                      MemoryError))):
                print(f"[incremental-cost/visual] multimodal streaming "
                      f"context build OOM ({type(e).__name__}); writing "
                      f"empty pickle. The allocator's --visual-format "
                      f"override will assign a uniform format to visual "
                      f"Linears.", flush=True)
                _write_empty_cost_shard(
                    output_path, shard_kind="visual", specs=specs,
                    model_name=model_name, probe_path=probe_path,
                )
                return None
            raise

    if mm_ctx.visual_module is None:
        print(f"[incremental-cost/visual] streaming context has no "
              f"visual module; writing empty pickle", flush=True)
        _write_empty_cost_shard(
            output_path, shard_kind="visual", specs=specs,
            model_name=model_name, probe_path=probe_path,
        )
        return mm_ctx

    model = mm_ctx.model
    from .model_profiles import DeadVendoredOverrideError, profile_from_model
    try:
        profile = profile_from_model(model)
    except DeadVendoredOverrideError:
        # Same as the body shard above (#202): a dead override prices the
        # visual tower against upstream modelling code.
        raise
    except Exception:
        profile = None
    live_linears = {n for n, m in model.named_modules()
                    if isinstance(m, nn.Linear)
                    and getattr(m, "weight", None) is not None
                    and not m.weight.is_meta}
    target_names = shard_targets & live_linears
    if not target_names:
        print(f"[incremental-cost/visual] probe stats matched the include "
              f"regex but no live visual Linears with the same name; "
              f"writing empty pickle", flush=True)
        _write_empty_cost_shard(
            output_path, shard_kind="visual", specs=specs,
            model_name=model_name, probe_path=probe_path,
        )
        return mm_ctx

    results = _run_cost_measurement(
        model,
        act_cache=act_cache,
        target_names=target_names,
        specs=specs,
        device=device,
        dtype=dtype,
        mode=mode,
        chunk_size=chunk_size,
        h_detail=h_detail,
        log_prefix="[incremental-cost/visual]",
        profile=profile,
        render_path=render_path,
    )
    shard_provenance = cb_render_provenance_for_results(
        model,
        results,
        specs,
        profile=profile,
        render_levers=(
            _production_cost_render_levers()
            if render_path == "production" else None
        ),
        where="incremental visual CB cost shard",
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "provenance": {
                **_cost_mode_provenance(),
                **shard_provenance,
            },
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "mode": ("batched" if mode == "auto" and device.startswith("cuda")
                         else ("unbatched" if mode == "auto" else mode)),
                "shard_kind": "visual",
                "render_path": render_path,
            },
        }, f)
    print(f"[incremental-cost/visual] wrote {out_path} "
          f"({len(results)} entries)", flush=True)
    return mm_ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cost-mode", default="",
                    help="Pipeline COST_MODE this table is produced under; "
                         "stamped into provenance['cost_mode'] so a re-run "
                         "cannot silently reuse another mode's table "
                         "(re-vet R2).")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--activation-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--formats", default="")
    ap.add_argument("--skip-missing-activations", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mode", choices=["auto", "batched", "unbatched"],
                    default="auto")
    ap.add_argument("--render-path", choices=["registry", "production"],
                    default="registry",
                    help="Weight renderer used for dense Linears. "
                         "`registry` is the fast RTN path; `production` "
                         "uses production_weight_cache.render_production_weight "
                         "so FP8/MXFP8 costs include GPTQ gates. Packed "
                         "experts keep the current export-faithful packed "
                         "measurement path.")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--swap-grow-limit-mb", type=int, default=256)
    ap.add_argument("--min-mem-available-mb", type=int, default=2048)
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--layers-per-shard", default="1",
                    help='Int, or "auto" to derive from available RAM + model size.')
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--include-mtp", action="store_true", default=True,
                    help="Measure cost for profile-declared MTP layers.")
    ap.add_argument("--no-include-mtp", action="store_false", dest="include_mtp")
    ap.add_argument("--include-visual", action="store_true", default=True,
                    help="Measure cost for profile-declared visual encoder "
                         "blocks. Text-only staging may still emit empty "
                         "shard pickles for unsupported multimodal paths.")
    ap.add_argument("--no-include-visual", action="store_false",
                    dest="include_visual")
    ap.add_argument("--include-lm-head", action="store_true", default=True,
                    help="Measure cost for the profile-declared LM head.")
    ap.add_argument("--no-include-lm-head", action="store_false",
                    dest="include_lm_head")
    ap.add_argument("--h-detail-dir", default=None,
                    help="If set, read per-Linear Fisher H diagonal (from "
                         "incremental_probe's --h-detail-dir) and emit "
                         "per-(layer, format) predicted_dloss and "
                         "Fisher row-weighted fisher_output_mse alongside "
                         "weight_mse in cost.pkl.")
    args = ap.parse_args()
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("incremental_measure_quant_cost", args.device)
    global _PIPELINE_COST_MODE
    _PIPELINE_COST_MODE = str(getattr(args, "cost_mode", "") or "")

    n_layers = load_num_hidden_layers(args.model)
    start = max(0, args.start_layer)
    end = n_layers if args.end_layer is None else min(args.end_layer, n_layers)
    if start >= end:
        raise SystemExit(f"empty layer range: start={start} end={end}")

    # Resolve --layers-per-shard: int literal or "auto" (hardware-adaptive).
    # Cost script has no --nsamples / --seqlen of its own — those belong to
    # the probe run. Pull them from the probe pickle's meta so the autoscale
    # memory model uses the same calibration shape that produced the probe.
    lps_arg = str(args.layers_per_shard).strip()
    if lps_arg.lower() in ("auto", ""):
        from .autoscale import pick_layers_per_shard
        try:
            with open(args.probe, "rb") as _f:
                _probe_meta = (pickle.load(_f).get("meta") or {})
        except Exception:
            _probe_meta = {}
        _nsamples = int(_probe_meta.get("nsamples", 32))
        _seqlen = int(_probe_meta.get("seqlen", 1024))
        lps, lps_diag = pick_layers_per_shard(
            args.model, nsamples=_nsamples, seqlen=_seqlen,
        )
        print(f"[cost] layers_per_shard=auto -> {lps} "
              f"(nsamples={_nsamples}, seqlen={_seqlen}, "
              f"available={lps_diag.get('available_gb',0):.1f} GB, "
              f"per_layer_active={lps_diag.get('per_layer_active_gb',0):.2f} GB)",
              flush=True)
        args.layers_per_shard = lps
    else:
        args.layers_per_shard = int(lps_arg)

    schedule = build_shard_schedule(
        model_path=args.model,
        num_body_layers=n_layers,
        body_layers_per_shard=args.layers_per_shard,
        body_layer_range=(start, end),
        include_mtp=args.include_mtp,
        include_visual=args.include_visual,
        include_lm_head=args.include_lm_head,
        unified_body_sweep=False,
    )
    shard_regexes = schedule.regexes()
    n_body_shards = sum(1 for entry in schedule if entry.kind == "body")
    n_extra_shards = len(schedule) - n_body_shards
    print(f"[incremental-cost] shard regexes: {len(shard_regexes)} total "
          f"(body={n_body_shards}, extras={n_extra_shards})",
          flush=True)

    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # Shared probe / activation / spec context — probe pickle stays in
    # memory so each shard can just intersect its regex against the stats.
    #
    # prepare_cost_context sees the whole probe pickle, including optional
    # regions such as MTP and visual tower rows. Incremental runs can
    # deliberately exclude those regions, so scope missing-activation
    # validation to the actual shard schedule below.
    probe, stats, act_cache, _, missing_act_all, _, specs = prepare_cost_context(
        probe_path=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        formats_csv=args.formats,
        skip_missing_activations=True,
    )
    scheduled_targets = _scheduled_probe_targets(stats, shard_regexes)
    missing_act = sorted(
        name for name in missing_act_all
        if name in scheduled_targets
    )
    ignored_missing = len(missing_act_all) - len(missing_act)
    if ignored_missing:
        print(f"[incremental-cost] ignored {ignored_missing} missing "
              "activation(s) outside the requested shard schedule",
              flush=True)
    if missing_act and not args.skip_missing_activations:
        preview = ", ".join(missing_act[:8])
        suffix = " ..." if len(missing_act) > 8 else ""
        raise SystemExit(
            f"{len(missing_act)} scheduled Linears missing activation "
            f"({preview}{suffix}); pass --skip-missing-activations to proceed."
        )

    # H-detail is optional. Built against the full probe stat set so the
    # intersect-per-shard logic just checks `name in h_detail` when needed.
    h_detail: "HDetailIndex | None" = None
    if args.h_detail_dir:
        detail_path = Path(args.h_detail_dir)
        if detail_path.exists():
            # Gated on the probe's per-row Fisher denominators: a stale
            # h-detail dir (v3 blobs = per-ROUTED-token on expert rows, or
            # a dir from a different calibration size) is refused here
            # rather than silently mis-scaling predicted_dloss.
            h_detail = HDetailIndex(
                detail_path, set(stats.keys()),
                expected_norm_tokens=h_detail_expected_norm_tokens(probe))
            print(f"[incremental-cost] h-detail cache: {len(h_detail)} Linears "
                  "→ per-weight Δloss + Fisher row-weighted output MSE",
                  flush=True)
        else:
            print(f"[incremental-cost] WARN: h-detail dir {detail_path} not "
                  f"found; falling back to scalar proxy", flush=True)

    # Compute per-shard paths + expected metas up front so we can decide
    # whether to even bother loading the streaming context.
    shard_paths = [shard_dir / f"cost_shard_{i:03d}.pkl"
                   for i in range(len(shard_regexes))]
    expected_metas = [
        _expected_cost_shard_meta(
            model=args.model,
            probe_path=args.probe,
            linear_include=linear_include,
            shard_idx=i,
            activation_cache_dir=args.activation_cache_dir,
            mode=args.mode,
            chunk_size=args.chunk_size,
            h_detail_dir=args.h_detail_dir,
            formats=[s.name for s in specs],
            render_path=args.render_path,
            n_linears_expected=sum(
                1 for n in stats if re.search(linear_include, n)
            ),
        )
        for i, linear_include in enumerate(shard_regexes)
    ]
    all_reusable = all(
        shard_paths[i].exists()
        and cost_shard_is_reusable(shard_paths[i], expected_metas[i])
        for i in range(len(shard_regexes))
    )

    ctx: StreamingContext | None = None
    # Separate multimodal context for visual shards (visual tower
    # resident). Built lazily on first visual shard; reused across all
    # visual shards in this run.
    mm_ctx: StreamingContext | None = None
    mm_offload_folder = str(work_dir / "streaming_offload_mm")

    def _ensure_ready():
        nonlocal ctx
        if ctx is not None:
            return
        offload_folder = str(work_dir / "streaming_offload")
        ctx_local = _build_streaming_context(
            args.model,
            device=torch.device(args.device),
            dtype=dtype,
            offload_folder=offload_folder,
            log_prefix="[incremental-cost]",
        )
        # Watchdog is armed AFTER model skeleton is up — baseline is the
        # post-context state, so the trigger reflects measurement-phase
        # growth only (e.g. a leak in the activation cache path), not
        # transient churn from skeleton build / layer install.
        if not args.no_watchdog:
            start_mem_watchdog(
                swap_grow_limit_mb=args.swap_grow_limit_mb,
                min_mem_available_mb=args.min_mem_available_mb,
            )
        ctx = ctx_local

    try:
        if not all_reusable:
            _ensure_ready()

        for shard_idx, linear_include in enumerate(shard_regexes):
            shard_path = shard_paths[shard_idx]
            expected_meta = expected_metas[shard_idx]
            if shard_path.exists() and cost_shard_is_reusable(shard_path, expected_meta):
                print(f"[incremental-cost] reuse shard {shard_idx}: {shard_path}",
                      flush=True)
                continue
            if shard_path.exists():
                print(f"[incremental-cost] stale shard {shard_idx}: "
                      f"recomputing {shard_path}", flush=True)

            kind = _classify_shard(linear_include)
            print(f"[incremental-cost] shard {shard_idx} ({kind}): "
                  f"include={linear_include!r}", flush=True)
            _ensure_ready()

            if kind in ("body", "lm_head"):
                _run_body_cost_shard(
                    ctx,
                    linear_include=linear_include,
                    shard_kind=kind,
                    probe_stats=stats,
                    act_cache=act_cache,
                    specs=specs,
                    device=args.device,
                    dtype=dtype,
                    mode=args.mode,
                    chunk_size=args.chunk_size,
                    h_detail=h_detail,
                    output_path=str(shard_path),
                    model_name=args.model,
                    probe_path=args.probe,
                    render_path=args.render_path,
                )
            elif kind == "mtp":
                _run_mtp_cost_shard(
                    ctx,
                    model_path=args.model,
                    linear_include=linear_include,
                    probe_stats=stats,
                    act_cache=act_cache,
                    specs=specs,
                    device=args.device,
                    dtype=dtype,
                    mode=args.mode,
                    chunk_size=args.chunk_size,
                    h_detail=h_detail,
                    output_path=str(shard_path),
                    model_name=args.model,
                    probe_path=args.probe,
                    render_path=args.render_path,
                )
            elif kind == "visual":
                # Phase 2 multimodal path: if the probe's multimodal pass
                # populated visual Linear stats + activations, measure
                # them the same way body Linears are measured. Uses an
                # incremental multimodal streaming context (visual tower
                # resident; body on meta) so 122B-scale models don't
                # blow memory. On context-build OOM or no visual tower
                # in model, fall back to an empty pickle and let the
                # allocator's --visual-format override take over.
                mm_ctx = _run_visual_cost_shard(
                    model_path=args.model,
                    linear_include=linear_include,
                    probe_stats=stats,
                    act_cache=act_cache,
                    specs=specs,
                    device=args.device,
                    dtype=dtype,
                    mode=args.mode,
                    chunk_size=args.chunk_size,
                    h_detail=h_detail,
                    output_path=str(shard_path),
                    model_name=args.model,
                    probe_path=args.probe,
                    mm_ctx=mm_ctx,
                    mm_offload_folder=mm_offload_folder,
                    render_path=args.render_path,
                )
            else:
                # Other unclassified shard kinds — keep the empty-pickle
                # fallback for safety.
                print(f"[incremental-cost] unknown shard kind {kind!r} "
                      f"(include={linear_include!r}); writing empty pickle",
                      flush=True)
                _write_empty_cost_shard(
                    str(shard_path), shard_kind=kind, specs=specs,
                    model_name=args.model, probe_path=args.probe,
                )

            annotate_cost_shard(shard_path, expected_meta)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            # Per-shard intermediates (activation snapshots, H-detail tensors,
            # result dicts) accumulate pinned CPU memory. The CUDA allocator
            # catch-frees back to the pool, but Python/CPU lingers — swap
            # grows ~40 MB/shard on 122B without this gc sweep.
            import gc as _gc
            _gc.collect()
    finally:
        if ctx is not None:
            ctx.shutdown()
        if mm_ctx is not None:
            mm_ctx.shutdown()

    merge_cost_pickles(shard_paths, Path(args.output))
    # Coverage gate: every body layer the probe measured must appear in
    # the merged cost table. A silent hole here flows straight into the
    # allocation (the allocator only allocates what has costs) and the
    # export passes the hole through as passthrough bytes.
    import pickle as _pkl
    import re as _re
    with open(args.probe, "rb") as _pf:
        _probe_stats = _pkl.load(_pf).get("stats", {})
    with open(args.output, "rb") as _cf:
        _cost_table = _pkl.load(_cf).get("costs", {})
    _cost_names = set(_cost_table)
    # Rows a measurement exception poisoned. measure_batched_gpu aborts the
    # shard by default, so reaching here means someone ran with
    # PRISMAQUANT_COST_FAIL_FAST=0 — the table must still never reach the
    # allocator, which cannot tell a failed row from an unoffered format.
    _failed = sorted(
        f"{_n}/{_f}"
        for _n, _row in _cost_table.items()
        if isinstance(_row, dict)
        for _f, _e in _row.items()
        if isinstance(_e, dict) and _e.get("cost_measurement_failed")
    )
    if _failed:
        print(f"[incremental-cost] FATAL: {len(_failed)} (Linear, format) "
              f"rows failed measurement and carry no usable cost, e.g. "
              f"{_failed[:5]} — refusing to hand the allocator a cost table "
              f"with fail-open holes.", flush=True)
        raise SystemExit(2)
    _lay = _re.compile(r"\blayers\.(\d+)\.")
    def _layers_of(names):
        out = set()
        for n in names:
            m = _lay.search(str(n))
            if m:
                out.add(int(m.group(1)))
        return out
    _missing = sorted(_layers_of(_probe_stats) - _layers_of(_cost_names))
    if _missing:
        print(f"[incremental-cost] FATAL: merged cost table has no entries "
              f"for probed body layers {_missing} — refusing to hand the "
              f"allocator a cost table with silent holes.", flush=True)
        raise SystemExit(2)
    print(f"[incremental-cost] wrote merged cost to {args.output}", flush=True)


if __name__ == "__main__":
    main()
