"""Production-cache rendering for profile-synthesized MTP Linears.

Transformers deliberately does not instantiate the Qwen3.5/3.6 MTP sidecar,
so the resident body model passed to :func:`fill_production_weight_cache`
cannot discover or render ``mtp.*`` recipe units.  The probe and cost stages
already synthesize that module through the model profile and persist its input
rows in the shared activation cache.  This module connects those existing
pieces to the existing production renderer and ``ProductionWeightCache``.

There is intentionally no second rendering implementation here.  Source
weights and activation rows are validated before any output is written, then
``streaming_production_cache._render_dense_layer`` performs the same
GPU-resident GPTQ/JSO render and cache storage used by the streaming body path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    resolve_cost_target_name,
)
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    _FisherRowWeightCache,
    _fused_sibling_leaf_mapping_from_profile,
    _is_cb_format_name,
    _weighted_render_family,
    _write_render_score_sidecar,
)
from prismaquant.streaming_production_cache import _render_dense_layer


MTP_RENDER_METADATA_SCHEMA = (
    "prismaquant.production_weight_cache.mtp_render.v1"
)


def _canon_fmt(fmt: str) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _nonbf16_mtp_assignment(
    assignment: Mapping[str, str] | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for qname, fmt in (assignment or {}).items():
        qname_s = str(qname)
        if not qname_s.startswith("mtp."):
            continue
        fmt_canon = _canon_fmt(fmt)
        if fmt_canon != "BF16":
            out[qname_s] = fmt_canon
    return out


def _cache_dir_for_append(
    cache: ProductionWeightCache,
    cache_dir: str | Path | None,
) -> Path | None:
    cache_path = Path(cache.cache_dir) if cache.cache_dir else None
    requested_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None and requested_path is not None:
        if cache_path.resolve() != requested_path.resolve():
            raise RuntimeError(
                "MTP production render cache-dir mismatch: existing cache "
                f"uses {cache_path}, append requested {requested_path}"
            )
    path = requested_path or cache_path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        if cache.cache_dir is None:
            cache.cache_dir = str(path)
    return path


def _existing_render_score_records(
    cache: ProductionWeightCache,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if cache.metadata is None:
        cache.metadata = {}
    score_meta = cache.metadata.get("render_scores")
    if score_meta is None:
        score_meta = {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": 0,
            "records": {},
        }
        cache.metadata["render_scores"] = score_meta
    if not isinstance(score_meta, dict):
        raise RuntimeError(
            "ProductionWeightCache render_scores metadata is not a mapping"
        )
    records_raw = score_meta.get("records", {})
    if not isinstance(records_raw, Mapping):
        raise RuntimeError(
            "ProductionWeightCache render_scores.records is not a mapping"
        )
    records: dict[str, dict[str, object]] = {}
    for key, value in records_raw.items():
        if not isinstance(value, Mapping):
            raise RuntimeError(
                "ProductionWeightCache render score is not a mapping: "
                f"{key!r}"
            )
        # MTP records are replaced as one fail-closed append scope.  This
        # prevents a previous menu/assignment from leaving stale scalar costs
        # in a manifest even though those tensor keys were not requested now.
        if not str(key).startswith("mtp."):
            records[str(key)] = dict(value)
    return score_meta, records


def _load_exact_mtp_wrapper(
    model_path: str,
    profile,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, nn.Linear], int]:
    from transformers import AutoConfig

    local_only = Path(model_path).exists()
    cfg = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    text_config = getattr(cfg, "text_config", cfg)
    inner = profile.build_mtp_module(text_config)
    if inner is None:
        raise RuntimeError(
            f"profile '{profile.name}' declares MTP but "
            "build_mtp_module() returned None"
        )

    wrapper = nn.Module()
    wrapper.add_module("mtp", inner)
    wrapper.to(dtype=dtype)

    raw = profile.read_mtp_source_state_dict(model_path)
    if not raw:
        raise RuntimeError(
            f"profile '{profile.name}' declares MTP but the checkpoint has "
            f"no source tensors under {profile.mtp_source_prefix()!r}"
        )
    unmatched_source, module_unset = profile.load_mtp_state_dict(inner, raw)
    if unmatched_source or module_unset:
        raise RuntimeError(
            "MTP source/module coverage failure: "
            f"unmatched_source={unmatched_source[:8]} "
            f"module_params_unset={module_unset[:8]}"
        )

    wrapper.to(device=device)
    wrapper.eval()
    for param in wrapper.parameters():
        param.requires_grad_(False)

    linears = {
        name: module
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear)
        and not name.endswith(".mlp.gate")
    }
    if not linears:
        raise RuntimeError(
            f"profile '{profile.name}' synthesized MTP with no quantizable "
            "nn.Linear modules"
        )
    return wrapper, linears, len(raw)


def _preflight_activation_rows(
    act_index: ActivationIndex,
    linears: Mapping[str, nn.Linear],
    profile,
) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    failures: list[str] = []
    for qname, module in linears.items():
        canonical = resolve_cost_target_name(qname, act_index, profile)
        if canonical not in act_index:
            failures.append(f"{qname}: missing row (canonical={canonical})")
            continue
        try:
            inputs, _ = act_index.load_with_row_indices(canonical)
        except Exception as exc:
            failures.append(
                f"{qname}: unreadable activation row "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if not isinstance(inputs, torch.Tensor):
            failures.append(f"{qname}: activation inputs are not a tensor")
            continue
        if inputs.ndim != 2:
            failures.append(
                f"{qname}: activation inputs must be rank-2, got "
                f"shape={tuple(inputs.shape)}"
            )
            continue
        if int(inputs.shape[0]) <= 0:
            failures.append(f"{qname}: activation row set is empty")
            continue
        if int(inputs.shape[1]) != int(module.in_features):
            failures.append(
                f"{qname}: activation width={int(inputs.shape[1])} does not "
                f"match in_features={int(module.in_features)}"
            )
            continue
        row_counts[qname] = int(inputs.shape[0])
    if failures:
        raise RuntimeError(
            "MTP activation-cache coverage failure: "
            f"{len(failures)} invalid/missing rows; sample={failures[:8]}"
        )
    return row_counts


def _preflight_weighted_rows(
    formats_by_qname: Mapping[str, Sequence[str]],
    linears: Mapping[str, nn.Linear],
    col_weights: Mapping[str, torch.Tensor] | None,
) -> None:
    for qname, formats in formats_by_qname.items():
        for fmt in formats:
            if _is_cb_format_name(fmt):
                raise RuntimeError(
                    "MTP ProductionWeightCache rendering does not yet "
                    f"publish the identity-bound CB pair contract: "
                    f"{qname}@{fmt}"
                )
            if _weighted_render_family(fmt) is None:
                continue
            value = None if col_weights is None else col_weights.get(qname)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"weighted MTP render {qname}@{fmt} has no col_weights"
                )
            if value.numel() != int(linears[qname].in_features):
                raise RuntimeError(
                    f"weighted MTP render {qname}@{fmt} col_weights has "
                    f"{value.numel()} entries; expected "
                    f"{int(linears[qname].in_features)}"
                )


def fill_profile_mtp_production_cache(
    cache: ProductionWeightCache,
    model_path: str,
    *,
    profile,
    activation_cache_dir: str | Path | None,
    formats: Sequence[str] = ("NVFP4",),
    render_assignment: Mapping[str, str] | None = None,
    cache_dir: str | Path | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_act_rows: int = 256,
    h_detail_dir: str | Path | None = None,
    include_qnames: Sequence[str] | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    cb_serialization_context=None,
    progress: bool = True,
) -> int:
    """Append exact profile-synthesized ``mtp.*`` production renders.

    A concrete assignment renders only its non-BF16 MTP pairs and requires an
    activation cache.  A format-menu build preserves the historical body-only
    behavior when no activation cache is supplied; when supplied, every live
    MTP Linear is appended for the menu's non-BF16 formats.

    Returns the number of rendered ``(qname, format)`` pairs.  Every requested
    module, source tensor, activation row, and cache result is checked exactly;
    an explicit MTP request never degrades to RTN or passthrough.
    """
    assignment_scope = render_assignment is not None
    allowed = (
        {str(qname) for qname in include_qnames}
        if include_qnames is not None else None
    )
    allowlisted_mtp = (
        {qname for qname in allowed if qname.startswith("mtp.")}
        if allowed is not None else set()
    )
    explicit = _nonbf16_mtp_assignment(render_assignment)
    if allowed is not None:
        explicit = {
            qname: fmt for qname, fmt in explicit.items()
            if qname in allowed
        }
    if assignment_scope and not explicit:
        return 0
    if not assignment_scope and allowed is not None and not allowlisted_mtp:
        return 0

    menu_formats = tuple(dict.fromkeys(
        _canon_fmt(fmt) for fmt in formats if str(fmt).strip()
    ))
    menu_formats = tuple(fmt for fmt in menu_formats if fmt != "BF16")
    if not assignment_scope and not menu_formats:
        return 0

    # Profile capability is checked BEFORE the activation cache, so a model
    # that declares no MTP is a no-op regardless of what `act/` looks like.
    # The reverse order aborted the whole cache build for Gemma4/LFM2/MiniMax/
    # DSv4/GLM — none of which have MTP — whenever `act/` had been pruned to
    # reclaim disk mid-run, which the >=10%-free rule makes routine.
    has_mtp = bool(profile is not None and profile.has_mtp())
    if not has_mtp:
        if explicit or allowlisted_mtp:
            profile_name = getattr(profile, "name", "<none>")
            raise RuntimeError(
                f"assignment requests quantized mtp.* units but profile "
                f"'{profile_name}' does not declare MTP"
            )
        return 0

    if activation_cache_dir is None:
        if explicit or allowlisted_mtp:
            raise RuntimeError(
                "non-BF16 mtp.* assignment requires --activation-cache-dir "
                "from the MTP probe; production render cannot fabricate its "
                "GPTQ/JSO rows"
            )
        return 0
    activation_path = Path(activation_cache_dir)
    if not activation_path.is_dir():
        raise RuntimeError(
            f"MTP activation cache directory does not exist: {activation_path}"
        )

    device_t = torch.device(device)
    wrapper, live_linears, source_tensor_count = _load_exact_mtp_wrapper(
        model_path,
        profile,
        device=device_t,
        dtype=dtype,
    )
    try:
        if assignment_scope:
            unknown = sorted(set(explicit) - set(live_linears))
            if unknown:
                raise RuntimeError(
                    "MTP assignment/module coverage failure: requested names "
                    f"are absent from the synthesized profile module: "
                    f"{unknown[:8]}"
                )
            selected_linears = {
                qname: live_linears[qname] for qname in sorted(explicit)
            }
            formats_by_qname = {
                qname: (explicit[qname],) for qname in sorted(explicit)
            }
        else:
            unknown = sorted(allowlisted_mtp - set(live_linears))
            if unknown:
                raise RuntimeError(
                    "MTP include-qnames/module coverage failure: allowlisted "
                    f"names are absent from the synthesized profile module: "
                    f"{unknown[:8]}"
                )
            selected_linears = {
                qname: module
                for qname, module in sorted(live_linears.items())
                if allowed is None or qname in allowed
            }
            formats_by_qname = {
                qname: menu_formats for qname in selected_linears
            }

        act_index = ActivationIndex(activation_path, selected_linears)
        activation_rows = _preflight_activation_rows(
            act_index,
            selected_linears,
            profile,
        )
        _preflight_weighted_rows(
            formats_by_qname,
            selected_linears,
            col_weights,
        )

        expected_pairs = {
            (qname, fmt)
            for qname, fmts in formats_by_qname.items()
            for fmt in fmts
        }

        cache_dir_path = _cache_dir_for_append(cache, cache_dir)
        if cache.activation_max_abs is None:
            cache.activation_max_abs = {}
            cache.activation_scales = cache.activation_max_abs
        if cache.failed is None:
            cache.failed = {}
        if cache.metadata is None:
            cache.metadata = {}
        previous = cache.metadata.get("mtp_render")
        previous_entries = (
            int(previous.get("entries", 0))
            if isinstance(previous, Mapping) else 0
        )
        requested_entries = int(cache.metadata.get("requested_entries", 0))
        body_entries = requested_entries - previous_entries
        if body_entries < 0:
            raise RuntimeError(
                "ProductionWeightCache requested_entries is smaller than its "
                "prior mtp_render entry count"
            )
        # Replace the MTP append as one exact scope. Orphaned disk shards are
        # harmless and deliberately left recoverable, but stale manifest keys,
        # failures, or activation scales must not leak across stripes/reruns.
        for key in list(cache.weights):
            if key[0].startswith("mtp.") and key not in expected_pairs:
                cache.weights.pop(key)
        for key in list(cache.failed):
            if key[0].startswith("mtp."):
                cache.failed.pop(key)
        for qname in list(cache.activation_max_abs):
            if qname.startswith("mtp.") and qname not in selected_linears:
                cache.activation_max_abs.pop(qname)
        score_meta, render_score_records = _existing_render_score_records(cache)

        fused_mapping = _fused_sibling_leaf_mapping_from_profile(profile)
        fisher_rows = (
            _FisherRowWeightCache(h_detail_dir, fused_mapping or None)
            if bool(cache.levers.get("fisher_gptq", False)) and h_detail_dir
            else None
        )
        rendered = _render_dense_layer(
            wrapper,
            selected_linears,
            render_formats_by_qname=formats_by_qname,
            act_index=act_index,
            cache=cache,
            levers=cache.levers,
            cache_dir_path=cache_dir_path,
            profile=profile,
            device=device_t,
            fisher_rows=fisher_rows,
            render_score_records=render_score_records,
            col_weights=col_weights,
            cb_serialization_context=cb_serialization_context,
            retain_rendered=True,
            consume_render=None,
            consumer_identity=None,
            calibration_hash=str(cache.metadata.get("calib_hash") or "") or None,
            resume=False,
            max_act_rows=max_act_rows,
            cb_pair_identities={},
            cb_pair_artifacts={},
            transient_results={},
            cb_git_commit=None,
            cb_producer_source_sha256=None,
            trellis_plans={},
            trellis_pair_identities={},
            trellis_pair_artifacts={},
            trellis_git_commit=None,
            trellis_producer_source_sha256=None,
            joint_scale_modules=selected_linears,
            progress=False,
        )

        missing = sorted(
            (qname, fmt)
            for qname, fmt in expected_pairs
            if cache.resolve_key(qname, fmt) is None
        )
        failed = sorted(
            key for key in cache.failed if key in expected_pairs
        )
        if rendered != len(expected_pairs) or missing or failed:
            raise RuntimeError(
                "MTP ProductionWeightCache coverage failure: "
                f"requested={len(expected_pairs)} rendered={rendered} "
                f"missing={missing[:8]} failed={failed[:8]}"
            )

        score_meta["schema"] = "prismaquant.production_render_scores.v1"
        score_meta["entries"] = int(len(render_score_records))
        score_meta["records"] = dict(sorted(render_score_records.items()))

        cache.metadata["requested_entries"] = (
            body_entries + len(expected_pairs)
        )
        cache.metadata["mtp_render"] = {
            "schema": MTP_RENDER_METADATA_SCHEMA,
            "scope": "assignment" if assignment_scope else "format-menu",
            "entries": int(len(expected_pairs)),
            "qnames": sorted(selected_linears),
            "formats": sorted({
                fmt for fmts in formats_by_qname.values() for fmt in fmts
            }),
            "formats_by_qname": {
                qname: list(formats_by_qname[qname])
                for qname in sorted(formats_by_qname)
            },
            "source_prefix": profile.mtp_source_prefix(),
            "source_tensor_count": int(source_tensor_count),
            "activation_rows": dict(sorted(activation_rows.items())),
            "max_act_rows": int(max_act_rows),
        }
        _write_render_score_sidecar(
            None if cache_dir_path is None else cache_dir_path / "render_scores.json",
            render_score_records,
        )
        if progress:
            print(
                f"[prod-cache/mtp] rendered {rendered} exact MTP cache "
                f"entries from {len(selected_linears)} activation rows",
                flush=True,
            )
        return rendered
    finally:
        del wrapper
        if device_t.type == "cuda":
            torch.cuda.empty_cache()
