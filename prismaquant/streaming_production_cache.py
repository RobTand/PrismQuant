"""Streaming per-layer production weight-cache fill for very large models.

Renders GPTQ/JSO production weights for a checkpoint too large to load whole
(e.g. Tencent Hy3, 295B, 192 experts/layer). Mirrors the shard-by-shard
architecture of ``incremental_measure_quant_cost``: one ``StreamingContext``
with the head (embed + rotary + lm_head) resident and every decoder layer
offloaded to disk / on meta. Each layer is installed on demand, its assigned
non-BF16 dense Linears and packed experts are rendered from the probe's
activation cache, then the layer is unloaded. Only one decoder layer is
resident at a time, so peak memory is head + one layer + working set.

The rendered ``ProductionWeightCache`` is byte-identical in SEMANTICS to the
resident ``fill_production_weight_cache`` path: same ``(qname, fmt)`` keys, the
same ``activation_max_abs`` / ``packed_expert_coverage`` / ``levers`` metadata,
and the same coverage contract — export / recache consume it unchanged.

Unlike the resident path there is no whole-model forward pass here: dense
activation rows come from the probe's per-Linear activation cache (keyed by the
canonical Linear name) and each packed-experts module's input snapshot comes
from the same cache (keyed by the experts-module name). Routing is recomputed
offline from that snapshot + the resident gate weight, exactly as the resident
packed render does.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

# Must be set before the cuda allocator initializes — matches the streaming
# cost path so the caching allocator doesn't hoard freed blocks on the GB10
# unified-memory pool.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    resolve_cost_target_name,
)
from prismaquant.production_weight_cache import (
    CB_CACHE_PAIR_IDENTITY_SCHEMA,
    CB_RENDER_IDENTITY_METADATA_KEY,
    ProductionWeightCache,
    _FisherRowWeightCache,
    _build_cb_transient_consumer_receipt,
    _cache_pair_identity_filename,
    _canonical_json_sha256,
    _canonical_rendered_weight_tensor,
    _cb_cache_tensor_identity,
    _fused_sibling_leaf_mapping_from_profile,
    _render_base_format,
    _render_score_record,
    _render_score_record_key,
    _resolve_production_render_levers,
    _resolve_render_mechanism_plan,
    _store_rendered_weight_entry,
    _cache_weight_filename,
    _extend_production_cache_cb_render_identity,
    _is_cb_format_name,
    _production_cache_git_commit,
    _production_cache_source_sha256,
    _combined_source_weights_sha256,
    _validate_cb_cache_pair_resume,
    _write_cb_cache_pair_sidecar,
    bind_production_cache_cb_source_weights,
    build_cb_cache_pair_identity,
    fill_packed_expert_cache_entries,
    first_identity_difference,
    identity_value_for_error,
    render_production_weight,
    validate_cb_render_identity_metadata,
)
from prismaquant.streaming_model import _build_streaming_context


def _canon_fmt(fmt: str) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _layer_index_of(qname: str, layers_prefix: str) -> int | None:
    """Return the decoder-layer index a qname lives under, or None (head)."""
    if not qname.startswith(layers_prefix):
        return None
    rest = qname[len(layers_prefix):]
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _nonbf16_assignment(render_assignment: Mapping[str, str]) -> dict[str, str]:
    """Canonicalize and drop BF16 entries (dense keys are live qnames)."""
    out: dict[str, str] = {}
    for qname, fmt in render_assignment.items():
        fmt_canon = _canon_fmt(fmt)
        if fmt_canon == "BF16":
            continue
        out[str(qname)] = fmt_canon
    return out


def _eligible_dense_qname_modules(
    model: nn.Module,
    profile,
    skip_tokens: Sequence[str],
    include_qnames: set[str] | None = None,
) -> dict[str, nn.Module]:
    """Map live qname -> module for every eligible dense Linear (resident
    selection semantics: same ``iter_quantizable_tensors`` + pinned-skip
    filter that ``build_production_cache`` applies)."""
    out: dict[str, nn.Module] = {}
    skip = set(skip_tokens)
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if any(s in qname.split(".") for s in skip):
            continue
        if include_qnames is not None and qname not in include_qnames:
            continue
        out[qname] = mod
    return out


def _render_dense_layer(
    model: nn.Module,
    layer_dense_modules: Mapping[str, nn.Module],
    *,
    render_formats_by_qname: Mapping[str, Sequence[str]],
    act_index: ActivationIndex,
    cache: ProductionWeightCache,
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    device: torch.device,
    fisher_rows: _FisherRowWeightCache | None,
    render_score_records: dict[str, dict[str, object]],
    col_weights: Mapping[str, torch.Tensor] | None,
    cb_serialization_context,
    retain_rendered: bool,
    consume_render: Callable[..., Mapping[str, object]] | None,
    consumer_identity: Mapping[str, object] | None,
    calibration_hash: str | None,
    resume: bool,
    max_act_rows: int,
    cb_pair_identities: dict[tuple[str, str], dict[str, object]],
    cb_pair_artifacts: dict[str, dict[str, object]],
    transient_results: dict[str, dict[str, object]],
    cb_git_commit: str | None,
    cb_producer_source_sha256: str | None,
    joint_scale_modules: Mapping[str, nn.Module] | None = None,
    declared_cold_qnames: frozenset[str] = frozenset(),
    progress: bool,
) -> int:
    """Render this layer's dense menu from the activation cache.

    Reproduces ``fill_production_weight_cache``'s per-Linear render exactly:
    joint fused-sibling NVFP4 globals (siblings are resident in the same layer),
    fused-group-unified calibrated max_abs, then
    ``render_production_weight`` per (qname, fmt) stored via the shared
    ``_store_rendered_weight_entry``.
    """
    from prismaquant.decision_units import fused_group_key
    from prismaquant.export_native_compressed import _compute_nvfp4_joint_global

    layer_formats: dict[str, tuple[str, ...]] = {}
    for qname in layer_dense_modules:
        fmts = tuple(render_formats_by_qname.get(qname, ()))
        if fmts:
            layer_formats[qname] = fmts
    qname_to_module = {
        q: layer_dense_modules[q] for q in layer_formats
    }
    if not qname_to_module:
        return 0

    _extend_production_cache_cb_render_identity(
        cache,
        layer_formats,
        cb_serialization_context=cb_serialization_context,
        col_weights=col_weights,
    )
    dense_cb_source_weights = {
        qname: qname_to_module[qname].weight.detach()
        for qname, fmts in layer_formats.items()
        if any(_is_cb_format_name(fmt) for fmt in fmts)
    }
    if dense_cb_source_weights:
        bind_production_cache_cb_source_weights(
            cache,
            dense_cb_source_weights,
            require_complete=False,
            where="streaming ProductionWeightCache dense source binding",
        )

    expected_rerenders: dict[tuple[str, str], dict[str, object]] = {}
    completed: set[tuple[str, str]] = set()
    if dense_cb_source_weights:
        if not calibration_hash:
            raise ValueError(
                "streaming CB render requires an exact calibration_hash for "
                "per-pair identity"
            )
        cb_identity = cache.metadata.get(CB_RENDER_IDENTITY_METADATA_KEY)
        cb_layer_scope = {
            qname: tuple(
                fmt for fmt in fmts if _is_cb_format_name(fmt)
            )
            for qname, fmts in layer_formats.items()
            if any(_is_cb_format_name(fmt) for fmt in fmts)
        }
        validated_context = validate_cb_render_identity_metadata(
            cb_identity,
            expected_context=cb_serialization_context,
            expected_formats_by_qname=cb_layer_scope,
            require_source_complete=False,
            where="streaming ProductionWeightCache dense pair identity",
        )
        if cb_git_commit is None or cb_producer_source_sha256 is None:
            raise RuntimeError("streaming CB producer identity was not resolved")
        for qname, fmts in cb_layer_scope.items():
            weight_dtype = qname_to_module[qname].weight.dtype
            for fmt in fmts:
                key = (qname, fmt)
                pair_identity = build_cb_cache_pair_identity(
                    cb_identity,
                    qname=qname,
                    fmt=fmt,
                    calibration_hash=calibration_hash,
                    git_commit=cb_git_commit,
                    source_weight_dtype=weight_dtype,
                    cb_serialization_context=cb_serialization_context,
                    render_input_contract={
                        "path": "dense",
                        "max_act_rows": int(max_act_rows),
                    },
                    validated_context=validated_context,
                    producer_source_sha256=cb_producer_source_sha256,
                )
                cb_pair_identities[key] = pair_identity
                if cache_dir_path is None:
                    continue
                shard_path = cache_dir_path / _cache_weight_filename(qname, fmt)
                sidecar_path = (
                    cache_dir_path / _cache_pair_identity_filename(qname, fmt)
                )
                if not resume and (shard_path.exists() or sidecar_path.exists()):
                    raise RuntimeError(
                        "streaming production cache destination is not fresh "
                        f"for {qname}@{fmt}; pass resume=True only when the "
                        "identity-bound sidecar should be admitted"
                    )
                if not resume:
                    continue
                admitted = _validate_cb_cache_pair_resume(
                    cache_dir_path=cache_dir_path,
                    qname=qname,
                    fmt=fmt,
                    expected_identity=pair_identity,
                    require_render_score=True,
                    allow_missing_shard=True,
                    require_consumer_receipt=not retain_rendered,
                    expected_consumer_identity=(
                        consumer_identity if not retain_rendered else None
                    ),
                )
                if admitted is None:
                    continue
                score = admitted.get("render_score")
                if not isinstance(score, Mapping):
                    raise RuntimeError(
                        f"admitted CB pair {qname}@{fmt} has no render score"
                    )
                score_key = _render_score_record_key(qname, fmt)
                if shard_path.is_file():
                    if not retain_rendered:
                        raise RuntimeError(
                            f"transient CB resume found an unexpected retained "
                            f"shard for {qname}@{fmt}: {shard_path}"
                        )
                    cache.weights[key] = _cache_weight_filename(qname, fmt)
                    completed.add(key)
                elif retain_rendered:
                    expected_rerenders[key] = admitted
                else:
                    receipt = admitted.get("consumer_receipt")
                    result = (
                        receipt.get("result")
                        if isinstance(receipt, Mapping) else None
                    )
                    if not isinstance(result, Mapping):
                        raise RuntimeError(
                            f"admitted transient CB pair {qname}@{fmt} has no "
                            "consumer result"
                        )
                    transient_results[score_key] = dict(result)
                    completed.add(key)
                render_score_records[score_key] = dict(score)
                cb_pair_artifacts[score_key] = {
                    "identity": pair_identity,
                    "tensor": admitted.get("tensor"),
                    "render_score": dict(score),
                    "render_score_sha256": admitted.get(
                        "render_score_sha256"
                    ),
                    "consumer_receipt": admitted.get("consumer_receipt"),
                    "retained_weight": bool(shard_path.is_file()),
                }

    render_base_fmts = {
        _render_base_format(f)
        for fmts in layer_formats.values()
        for f in fmts
    }
    needs_nvfp4 = "NVFP4" in render_base_fmts

    # Joint fused-sibling NVFP4 global (max across q/k/v or gate/up). Restricted
    # to this layer's non-BF16 dense qnames; siblings are co-resident. The
    # synthetic all-NVFP4 assignment matches the resident derivation.
    joint_globals: dict[str, torch.Tensor] = {}
    if needs_nvfp4:
        joint_scope = (
            dict(joint_scale_modules)
            if joint_scale_modules is not None
            else qname_to_module
        )
        joint_globals = _compute_nvfp4_joint_global(
            model,
            {q: "NVFP4" for q in joint_scope},
            profile=profile,
        )

    # Per-Linear calibrated max_abs, unified (max) across fused sibling groups
    # — reproduces the resident block; the value drives only the export
    # activation scale (metadata), never the rendered weight.
    if needs_nvfp4:
        per_qname_max_abs: dict[str, float] = {}
        for qname in qname_to_module:
            canonical = resolve_cost_target_name(qname, act_index, profile)
            if canonical not in act_index:
                continue
            X, _ = act_index.load_with_row_indices(canonical)
            mx = float(X.abs().max().item())
            if mx > 0:
                per_qname_max_abs[qname] = mx
        groups: dict[str, list[str]] = defaultdict(list)
        for qname in per_qname_max_abs:
            gk = (
                fused_group_key(profile, qname)
                if profile is not None else qname
            )
            groups[gk].append(qname)
        for members in groups.values():
            shared = max(per_qname_max_abs[m] for m in members)
            for m in members:
                cache.activation_max_abs[m] = shared

    rendered = 0
    for qname, mod in qname_to_module.items():
        pending_formats = [
            fmt for fmt in layer_formats[qname]
            if (qname, fmt) not in completed
        ]
        if not pending_formats:
            continue
        weight = mod.weight.data
        canonical = resolve_cost_target_name(qname, act_index, profile)
        cold_declared = qname in declared_cold_qnames
        if canonical not in act_index and not cold_declared:
            raise RuntimeError(
                "[stream-prod-cache] no cached activations for dense Linear "
                f"{qname} (canonical={canonical}); the probe activation cache "
                "must cover every non-BF16 assignment entry — streaming render "
                "cannot fabricate the GPTQ Hessian."
            )
        if cold_declared:
            if canonical in act_index:
                raise RuntimeError(
                    f"{qname}: declared never-routed expert unexpectedly has "
                    "cached activations; refusing the cold-render exception"
                )
            X_cpu = torch.empty(
                (0, int(weight.shape[1])), dtype=torch.float32,
            )
        else:
            X_cpu, _ = act_index.load_with_row_indices(canonical)
            if max_act_rows > 0:
                X_cpu = X_cpu[:max_act_rows]
        # Activation-residency landmine: act-cache tensors are CPU-resident;
        # move to the compute device + fp32 explicitly or GPTQ silently runs
        # on CPU (and diverges from the resident render dtype).
        X = X_cpu.to(device=device, dtype=torch.float32)
        # A provenance-authorized never-routed expert has no activation rows
        # by definition.  Keep the mapping empty so LDLQ takes its explicit
        # raw-render fallback; an empty matrix would look like a successfully
        # loaded (but degenerate) calibration input to downstream code.
        activations = {} if cold_declared else {qname: X}
        joint = joint_globals.get(qname)
        max_abs = cache.activation_max_abs.get(qname)
        # Export input_global_scale: MUST match the resident render loop and
        # the exporter (the igs convention is a ±14-37% served-KL knob).
        export_scale = None
        if max_abs is not None and max_abs > 0:
            from prismaquant.export_native_compressed import (
                _nvfp4_input_global_scale_from_max_abs,
            )
            export_scale = _nvfp4_input_global_scale_from_max_abs(
                float(max_abs))
        row_weights = (
            fisher_rows.get(qname)
            if (fisher_rows is not None
                and bool(levers.get("fisher_gptq", False)))
            else None
        )
        for fmt in pending_formats:
            render_fmt = _render_base_format(fmt)
            gate_trace: list[dict[str, object]] = []
            timed_cb_pair = (
                cache_dir_path is not None and _is_cb_format_name(fmt)
            )
            if timed_cb_pair and weight.device.type == "cuda":
                torch.cuda.synchronize(weight.device)
            encode_started = time.perf_counter()
            w_dq = render_production_weight(
                weight, render_fmt,
                qname=qname,
                activations=activations,
                levers=levers,
                joint_global_real=joint,
                input_global_scale=export_scale,
                fisher_row_weights=row_weights,
                col_weights=(
                    None if col_weights is None else col_weights.get(qname)
                ),
                cb_serialization_context=cb_serialization_context,
                gate_trace=gate_trace,
                ldlq_missing_activation_ok=cold_declared,
            )
            if timed_cb_pair and weight.device.type == "cuda":
                torch.cuda.synchronize(weight.device)
            encode_seconds = (
                time.perf_counter() - encode_started
                if timed_cb_pair else 0.0
            )
            score_key = _render_score_record_key(qname, fmt)
            render_score = _render_score_record(
                qname=qname,
                fmt=fmt,
                render_format=render_fmt,
                reference_weight=weight,
                rendered_weight=w_dq,
                activations=X,
                activation_max_abs=max_abs,
            )
            render_score_records[score_key] = render_score
            canonical_render = _canonical_rendered_weight_tensor(
                w_dq,
                weight_dtype=weight.dtype,
            )
            expected_sidecar = expected_rerenders.get((qname, fmt))
            if expected_sidecar is not None:
                observed_tensor = _cb_cache_tensor_identity(canonical_render)
                tensor_difference = first_identity_difference(
                    expected_sidecar.get("tensor"),
                    observed_tensor,
                    path="tensor",
                )
                if tensor_difference is not None:
                    field, stored, rerendered = tensor_difference
                    raise RuntimeError(
                        "selected CB assignment re-render differs from the "
                        f"streamed scalar for {qname}@{fmt} at '{field}': "
                        f"stored={identity_value_for_error(stored)} "
                        f"rerendered={identity_value_for_error(rerendered)}; "
                        "refusing publication"
                    )

            receipt = None
            if not retain_rendered:
                if consume_render is None:
                    result = dict(render_score)
                else:
                    result = consume_render(
                        qname=qname,
                        fmt=fmt,
                        reference_weight=weight,
                        rendered_weight=canonical_render,
                        render_score=render_score,
                    )
                if not isinstance(result, Mapping):
                    raise TypeError(
                        "streamed render consumer must return a Mapping for "
                        f"{qname}@{fmt}, got {type(result).__name__}"
                    )
                # A receipt hashes the full canonical tensor and exists to
                # authenticate a durable transient sidecar. Production-anchor
                # AURA has no pair sidecar/cache directory; its exact consumer
                # contract is already nested in the checkpoint identity, so a
                # throwaway per-anchor tensor hash only burns CPU and UMA
                # bandwidth on the hot path.
                if cache_dir_path is not None:
                    if consumer_identity is None:
                        raise ValueError(
                            "durable transient streamed render requires "
                            "consumer_identity"
                        )
                    receipt = _build_cb_transient_consumer_receipt(
                        qname=qname,
                        fmt=fmt,
                        tensor=canonical_render,
                        render_score=render_score,
                        consumer_identity=consumer_identity,
                        result=result,
                    )
                transient_results[score_key] = dict(result)

            if cache_dir_path is not None and _is_cb_format_name(fmt):
                pair_identity = cb_pair_identities.get((qname, fmt))
                if pair_identity is None:
                    raise RuntimeError(
                        f"streaming CB pair identity missing for {qname}@{fmt}"
                    )
                _write_cb_cache_pair_sidecar(
                    cache_dir_path=cache_dir_path,
                    qname=qname,
                    fmt=fmt,
                    identity=pair_identity,
                    tensor=canonical_render,
                    encode_seconds=encode_seconds,
                    render_score=render_score,
                    consumer_receipt=receipt,
                    retained_weight=retain_rendered,
                )
                cb_pair_artifacts[score_key] = {
                    "identity": pair_identity,
                    "tensor": _cb_cache_tensor_identity(canonical_render),
                    "render_score": render_score,
                    "render_score_sha256": receipt.get("render_score_sha256")
                    if isinstance(receipt, Mapping) else None,
                    "consumer_receipt": receipt,
                    "retained_weight": bool(retain_rendered),
                }
            if retain_rendered:
                _store_rendered_weight_entry(
                    weights=cache.weights,
                    cache_dir_path=cache_dir_path,
                    qname=qname,
                    fmt=fmt,
                    tensor=w_dq,
                    weight_dtype=weight.dtype,
                    durable=_is_cb_format_name(fmt),
                )
            rendered += 1
            del canonical_render, w_dq
        del X
        activations.clear()
    if progress and rendered:
        print(f"[stream-prod-cache] rendered {rendered} dense Linears",
              flush=True)
    return rendered


PRODUCTION_ANCHOR_RENDERER_IDENTITY_SCHEMA = (
    "prismaquant.streaming_production_anchor_renderer.identity.v1"
)
_ANCHOR_PASSTHROUGH_FORMATS = frozenset({
    "BF16",
    "FP8_SOURCE",
    "FP8_BLOCK_UE8M0_SOURCE",
    "MXFP4_SOURCE",
})
_COLD_EXPERT_RULE = "unrouted_expert_neutral_prior:layer_routed_mean"


class StreamedProductionAnchorRenderer:
    """Render an exact sparse anchor plan while one decoder layer is live.

    The object deliberately owns no model residency mechanism.  Its caller
    installs a layer through the existing streaming context, then calls
    :meth:`render_layer` with that layer's live ``nn.Linear`` modules.  Each
    production render can either be returned in memory or passed directly to
    a transient consumer and removed from the temporary
    ``ProductionWeightCache`` immediately; no rendered-weight shard or full
    menu is published. The transient path bounds live rendered weights to one
    pair while a caller retains its smaller derived state.

    ``formats_by_qname`` is the complete anchor plan, not a candidate menu.
    Multiple formats for one qname are supported because a unit can have one
    measured level in each legal ``(family, basis)`` segment.  Passthrough
    terminals are rejected here: they are exact source/BF16 allocator cells,
    never synthesized weights.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        act_index: ActivationIndex,
        formats_by_qname: Mapping[str, Sequence[str]],
        levers: Mapping[str, object],
        profile,
        device: torch.device | str,
        col_weights: Mapping[str, torch.Tensor] | None,
        cb_serialization_context,
        calibration_hash: str,
        arm_identity: Mapping[str, object],
        model_identity: Mapping[str, object],
        cold_expert_provenance: Mapping[str, object] | None = None,
        max_act_rows: int = 512,
        h_detail_dir: str | Path | None = None,
        producer_git_commit: str | None = None,
        producer_source_sha256: str | None = None,
        transient_consumer_identity: Mapping[str, object] | None = None,
    ) -> None:
        from prismaquant.cost_stage_checkpoint import canonical_json
        from prismaquant.cost_streaming import validate_streamed_model_identity
        from prismaquant.routed_experts import (
            profile_declared_routed_expert_targets,
            resolve_routed_expert_profile,
        )

        if not isinstance(arm_identity, Mapping) or not arm_identity:
            raise ValueError(
                "production anchor renderer requires an exact nonempty "
                "arm_identity"
            )
        if not isinstance(calibration_hash, str) or not calibration_hash:
            raise ValueError(
                "production anchor renderer requires an exact calibration_hash"
            )
        exact_model_identity = validate_streamed_model_identity(
            model_identity, where="production anchor renderer"
        )
        if int(max_act_rows) < 1:
            raise ValueError("production anchor max_act_rows must be positive")

        canonical_plan: dict[str, tuple[str, ...]] = {}
        for raw_qname, raw_formats in sorted(formats_by_qname.items()):
            qname = str(raw_qname)
            if isinstance(raw_formats, (str, bytes)):
                raise TypeError(
                    f"production anchor plan {qname!r} formats must be a "
                    "sequence, not a string"
                )
            formats: list[str] = []
            for raw_fmt in raw_formats:
                fmt = _canon_fmt(str(raw_fmt))
                if fmt in _ANCHOR_PASSTHROUGH_FORMATS:
                    raise ValueError(
                        f"production anchor plan attempts to synthesize "
                        f"passthrough terminal {qname}@{fmt}"
                    )
                if fmt in formats:
                    raise ValueError(
                        f"production anchor plan duplicates {qname}@{fmt}"
                    )
                formats.append(fmt)
            if not formats:
                raise ValueError(
                    f"production anchor plan has no measured anchor for {qname}"
                )
            canonical_plan[qname] = tuple(formats)
        if not canonical_plan:
            raise ValueError("production anchor plan is empty")

        modules = dict(model.named_modules())
        invalid_modules = sorted(
            qname for qname in canonical_plan
            if not isinstance(modules.get(qname), nn.Linear)
        )
        if invalid_modules:
            raise ValueError(
                "production anchor plan contains qnames that are not live "
                f"nn.Linear modules; sample={invalid_modules[:8]}"
            )

        profile = resolve_routed_expert_profile(model, profile)
        raw_cold = dict(cold_expert_provenance or {})
        if raw_cold and raw_cold.get("rule") != _COLD_EXPERT_RULE:
            raise ValueError(
                "production anchor cold-expert provenance has unsupported "
                f"rule {raw_cold.get('rule')!r}"
            )
        raw_names = raw_cold.get("names", ()) if raw_cold else ()
        if isinstance(raw_names, (str, bytes)):
            raise ValueError(
                "production anchor cold-expert provenance names is malformed"
            )
        cold_qnames = frozenset(str(name) for name in raw_names)
        unexpected_cold = sorted(cold_qnames - set(canonical_plan))
        if unexpected_cold:
            raise ValueError(
                "production anchor cold-expert declaration contains unplanned "
                f"qnames; sample={unexpected_cold[:8]}"
            )
        routed_targets = set(
            profile_declared_routed_expert_targets(model, profile)
        )
        not_routed = sorted(cold_qnames - routed_targets)
        if not_routed:
            raise ValueError(
                "production anchor cold exception is limited to profile-"
                f"declared routed experts; sample={not_routed[:8]}"
            )

        # Probe-side key, not the live module name: DSv4-Flash keys its probe
        # and activation cache under the RAW per-expert names, so resolving
        # through canonical_linear_name alone reports every routed expert as
        # missing.  resolve_cost_target_name is the one rule both conventions
        # go through, and it stays fail-closed when neither spelling is cached.
        missing_activations = {
            qname
            for qname in canonical_plan
            if resolve_cost_target_name(qname, act_index, profile)
            not in act_index
        }
        if missing_activations != set(cold_qnames):
            undeclared = sorted(missing_activations - set(cold_qnames))
            falsely_declared = sorted(set(cold_qnames) - missing_activations)
            raise RuntimeError(
                "production anchor activation coverage differs from the exact "
                "cold-expert declaration; "
                f"undeclared_missing={undeclared[:8]} "
                f"declared_but_present={falsely_declared[:8]}"
            )

        self.model = model
        self.act_index = act_index
        self.formats_by_qname = canonical_plan
        self.profile = profile
        self.device = torch.device(device)
        self.col_weights = col_weights
        self.cb_serialization_context = cb_serialization_context
        self.calibration_hash = calibration_hash
        self.max_act_rows = int(max_act_rows)
        self.cold_qnames = cold_qnames
        self.levers = _resolve_production_render_levers(levers)
        self.levers.setdefault("weighted_vq", True)
        self.mechanism_plan = _resolve_render_mechanism_plan(self.levers)
        fused_mapping = _fused_sibling_leaf_mapping_from_profile(profile)
        self.fisher_rows = (
            _FisherRowWeightCache(h_detail_dir, fused_mapping or None)
            if bool(self.levers.get("fisher_gptq", False)) and h_detail_dir
            else None
        )
        self.cache = ProductionWeightCache(
            weights={},
            levers=dict(self.levers),
            activation_max_abs={},
            failed={},
            cache_dir=None,
            metadata={},
        )
        _extend_production_cache_cb_render_identity(
            self.cache,
            self.formats_by_qname,
            cb_serialization_context=self.cb_serialization_context,
            col_weights=self.col_weights,
            render_levers=self.levers,
            render_mechanism_plan=self.mechanism_plan,
        )
        self.producer_git_commit = (
            str(producer_git_commit)
            if producer_git_commit is not None
            else _production_cache_git_commit()
        )
        self.producer_source_sha256 = (
            str(producer_source_sha256)
            if producer_source_sha256 is not None
            else _production_cache_source_sha256()
        )
        self.transient_consumer_identity = (
            canonical_json(
                transient_consumer_identity,
                where="production anchor transient consumer identity",
            )
            if transient_consumer_identity is not None
            else None
        )
        identity = {
            "schema": PRODUCTION_ANCHOR_RENDERER_IDENTITY_SCHEMA,
            "formats_by_qname": {
                qname: list(formats)
                for qname, formats in self.formats_by_qname.items()
            },
            "requested_entries": sum(
                len(formats) for formats in self.formats_by_qname.values()
            ),
            "calibration_hash": self.calibration_hash,
            "max_act_rows": self.max_act_rows,
            "arm_identity": dict(arm_identity),
            # This complete checkpoint-shard/value-map identity is the source
            # binding authority on partial resume.  Per-layer CB source hashes
            # cannot become complete when an already-checkpointed unit is
            # intentionally not rematerialized; the immutable streamed-model
            # content identity is the equivalent stronger binding.
            "source_model": exact_model_identity,
            "source_weight_binding": (
                "complete_streamed_model_content_identity"
            ),
            "cold_expert_provenance": raw_cold,
            "cb_render_identity": self.cache.metadata.get(
                CB_RENDER_IDENTITY_METADATA_KEY
            ),
            "producer_git_commit": self.producer_git_commit,
            "producer_source_sha256": self.producer_source_sha256,
            "retention": "one_render_or_explicit_layer_mapping",
            "transient_consumer_identity": (
                dict(self.transient_consumer_identity)
                if self.transient_consumer_identity is not None
                else None
            ),
        }
        self.identity = canonical_json(
            identity, where="production anchor renderer identity"
        )
        self.render_count = 0
        self.max_live_rendered = 0
        # Lazily bound source identities for units outside any CB scope
        # (stock plans run no CB source binding); see
        # source_weight_identity_for.
        self._stock_source_identities: dict[str, dict[str, object]] = {}

    def render_layer(
        self,
        *,
        layer: int,
        modules: Mapping[str, nn.Linear],
        formats_by_qname: Mapping[str, Sequence[str]],
    ) -> dict[tuple[str, str], torch.Tensor]:
        """Render only the requested missing anchors for one installed layer."""
        del layer  # residency/layer ordering is owned by the caller
        requested: dict[str, tuple[str, ...]] = {}
        for qname, raw_formats in formats_by_qname.items():
            name = str(qname)
            if name not in self.formats_by_qname:
                raise RuntimeError(
                    f"unplanned production anchor render requested for {name}"
                )
            formats = tuple(_canon_fmt(fmt) for fmt in raw_formats)
            if not formats or not set(formats).issubset(
                self.formats_by_qname[name]
            ):
                raise RuntimeError(
                    f"production anchor render request for {name} differs "
                    "from its identity-bound plan"
                )
            requested[name] = formats
        if not requested:
            return {}
        missing_modules = sorted(set(requested) - set(modules))
        if missing_modules:
            raise RuntimeError(
                "production anchor render was not given live modules for "
                f"{missing_modules[:8]}"
            )
        if self.cache.weights:
            raise RuntimeError(
                "production anchor renderer retained weights across layers"
            )

        layer_scope = {
            qname: module
            for qname, module in modules.items()
            if qname in self.formats_by_qname
        }
        render_scores: dict[str, dict[str, object]] = {}
        _render_dense_layer(
            self.model,
            layer_scope,
            render_formats_by_qname=requested,
            act_index=self.act_index,
            cache=self.cache,
            levers=self.levers,
            cache_dir_path=None,
            profile=self.profile,
            device=self.device,
            fisher_rows=self.fisher_rows,
            render_score_records=render_scores,
            col_weights=self.col_weights,
            cb_serialization_context=self.cb_serialization_context,
            retain_rendered=True,
            consume_render=None,
            consumer_identity=None,
            calibration_hash=self.calibration_hash,
            resume=False,
            max_act_rows=self.max_act_rows,
            cb_pair_identities={},
            cb_pair_artifacts={},
            transient_results={},
            cb_git_commit=self.producer_git_commit,
            cb_producer_source_sha256=self.producer_source_sha256,
            joint_scale_modules=layer_scope,
            declared_cold_qnames=self.cold_qnames,
            progress=False,
        )
        expected = {
            (qname, fmt)
            for qname, formats in requested.items()
            for fmt in formats
        }
        observed = set(self.cache.weights)
        if observed != expected:
            raise RuntimeError(
                "production anchor renderer did not materialize its exact "
                f"layer plan: missing={sorted(expected - observed)[:8]} "
                f"unexpected={sorted(observed - expected)[:8]}"
            )
        self.max_live_rendered = max(self.max_live_rendered, len(observed))
        rendered = {
            key: self.cache.weights.pop(key) for key in sorted(expected)
        }
        self.render_count += len(rendered)
        return rendered

    def render_layer_transient(
        self,
        *,
        layer: int,
        modules: Mapping[str, nn.Linear],
        formats_by_qname: Mapping[str, Sequence[str]],
        consume_render: Callable[..., Mapping[str, object]],
        consumer_identity: Mapping[str, object],
    ) -> tuple[tuple[str, str], ...]:
        """Render and consume one pair at a time for an installed layer.

        This is the memory-bounded counterpart to :meth:`render_layer`.
        Rendering still goes through ``_render_dense_layer`` and the shared
        ``ProductionWeightCache`` identity machinery; only retention changes.
        The callback must finish deriving all value-bearing state before it
        returns because the rendered tensor is released immediately.
        """
        del layer  # residency/layer ordering is owned by the caller
        if not callable(consume_render):
            raise TypeError(
                "transient production anchor consumer is not callable"
            )
        if not isinstance(consumer_identity, Mapping) or not consumer_identity:
            raise ValueError(
                "transient production anchor consumer identity is empty"
            )
        if self.transient_consumer_identity is None:
            raise RuntimeError(
                "production anchor renderer has no identity-bound transient "
                "consumer contract"
            )
        from prismaquant.cost_stage_checkpoint import canonical_json

        supplied_consumer_identity = canonical_json(
            consumer_identity,
            where="production anchor transient consumer identity",
        )
        if supplied_consumer_identity != self.transient_consumer_identity:
            raise RuntimeError(
                "production anchor transient consumer identity differs from "
                "the renderer's identity-bound contract"
            )
        requested: dict[str, tuple[str, ...]] = {}
        for qname, raw_formats in formats_by_qname.items():
            name = str(qname)
            if name not in self.formats_by_qname:
                raise RuntimeError(
                    f"unplanned production anchor render requested for {name}"
                )
            formats = tuple(_canon_fmt(fmt) for fmt in raw_formats)
            if not formats or not set(formats).issubset(
                self.formats_by_qname[name]
            ):
                raise RuntimeError(
                    f"production anchor render request for {name} differs "
                    "from its identity-bound plan"
                )
            requested[name] = formats
        if not requested:
            return ()
        missing_modules = sorted(set(requested) - set(modules))
        if missing_modules:
            raise RuntimeError(
                "production anchor render was not given live modules for "
                f"{missing_modules[:8]}"
            )
        if self.cache.weights:
            raise RuntimeError(
                "production anchor renderer retained weights across layers"
            )

        layer_scope = {
            qname: module
            for qname, module in modules.items()
            if qname in self.formats_by_qname
        }
        expected = {
            (qname, fmt)
            for qname, formats in requested.items()
            for fmt in formats
        }
        observed: list[tuple[str, str]] = []
        observed_set: set[tuple[str, str]] = set()

        def _consume_once(**kwargs) -> Mapping[str, object]:
            key = (
                str(kwargs.get("qname")),
                _canon_fmt(str(kwargs.get("fmt"))),
            )
            if key not in expected:
                raise RuntimeError(
                    "production anchor renderer emitted an unexpected "
                    f"transient pair {key}"
                )
            if key in observed_set:
                raise RuntimeError(
                    "production anchor renderer emitted a duplicate "
                    f"transient pair {key}"
                )
            result = consume_render(**kwargs)
            if not isinstance(result, Mapping):
                raise TypeError(
                    "transient production anchor consumer must return a "
                    f"Mapping for {key}, got {type(result).__name__}"
                )
            observed.append(key)
            observed_set.add(key)
            return result

        render_scores: dict[str, dict[str, object]] = {}
        _render_dense_layer(
            self.model,
            layer_scope,
            render_formats_by_qname=requested,
            act_index=self.act_index,
            cache=self.cache,
            levers=self.levers,
            cache_dir_path=None,
            profile=self.profile,
            device=self.device,
            fisher_rows=self.fisher_rows,
            render_score_records=render_scores,
            col_weights=self.col_weights,
            cb_serialization_context=self.cb_serialization_context,
            retain_rendered=False,
            consume_render=_consume_once,
            consumer_identity=consumer_identity,
            calibration_hash=self.calibration_hash,
            resume=False,
            max_act_rows=self.max_act_rows,
            cb_pair_identities={},
            cb_pair_artifacts={},
            transient_results={},
            cb_git_commit=self.producer_git_commit,
            cb_producer_source_sha256=self.producer_source_sha256,
            joint_scale_modules=layer_scope,
            declared_cold_qnames=self.cold_qnames,
            progress=False,
        )
        if observed_set != expected:
            raise RuntimeError(
                "production anchor renderer did not consume its exact layer "
                f"plan: missing={sorted(expected - observed_set)[:8]} "
                f"unexpected={sorted(observed_set - expected)[:8]}"
            )
        if self.cache.weights:
            raise RuntimeError(
                "transient production anchor renderer retained a weight"
            )
        self.max_live_rendered = max(self.max_live_rendered, 1)
        self.render_count += len(observed)
        return tuple(observed)

    def source_weight_identity_for(
        self, qname: str,
    ) -> dict[str, object]:
        """Return the unit's source-weight value identity.

        A CB render binds it during source binding and this method reads it
        back rather than hashing twice.  A stock (non-CB) unit runs no CB
        source binding at all, so its identity is bound here lazily from the
        live source weight -- the render is transient and the live module
        weight is never mutated, so the tensor hashed is exactly the source.
        A unit *inside* a CB scope with no binding stays a hard refusal: that
        is a lost identity, not a stock plan.
        """
        identity = self.cache.metadata.get(CB_RENDER_IDENTITY_METADATA_KEY)
        shapes = (
            identity.get("source_weights_shapes")
            if isinstance(identity, Mapping) else None
        )
        content = (
            identity.get("source_weights_content_sha256")
            if isinstance(identity, Mapping) else None
        )
        name = str(qname)
        if (
            isinstance(shapes, Mapping)
            and isinstance(content, Mapping)
            and name in shapes
            and name in content
        ):
            return {
                "shape": [int(dim) for dim in shapes[name]],
                "sha256": str(content[name]).lower(),
            }
        cb_scope = (
            identity.get("cb_formats_by_qname")
            if isinstance(identity, Mapping) else None
        )
        if isinstance(cb_scope, Mapping) and name in cb_scope:
            raise RuntimeError(
                f"production anchor renderer has no bound source identity "
                f"for {name}"
            )
        if name not in self.formats_by_qname:
            raise RuntimeError(
                f"production anchor renderer was asked for the source "
                f"identity of unplanned unit {name}"
            )
        cached = self._stock_source_identities.get(name)
        if cached is not None:
            return dict(cached)
        from prismaquant.production_weight_cache import (
            _source_weight_value_identity,
        )

        weight = self.model.get_submodule(name).weight
        shape, digest = _source_weight_value_identity(weight.data)
        bound = {
            "shape": [int(dim) for dim in shape],
            "sha256": str(digest).lower(),
        }
        self._stock_source_identities[name] = bound
        return dict(bound)

    def bind_completed_source_weight_identities(
        self,
        records: Mapping[str, Mapping[str, object]],
    ) -> dict[str, object]:
        """Finalize sparse renderer provenance from rendered + resumed units.

        The AURA journal restores records for units skipped on resume.  This
        method combines them with freshly bound rows and publishes a
        source-complete *sparse anchor* CB identity; it never claims that the
        unrendered ladder cells were materialized.
        """
        from prismaquant.cost_stage_checkpoint import canonical_json

        if set(records) != set(self.formats_by_qname):
            raise RuntimeError(
                "production anchor completed source identity scope differs "
                f"from the anchor plan: missing="
                f"{sorted(set(self.formats_by_qname) - set(records))[:8]} "
                f"unexpected={sorted(set(records) - set(self.formats_by_qname))[:8]}"
            )
        normalized: dict[str, dict[str, object]] = {}
        for name in sorted(records):
            row = records[name]
            shape = [int(dim) for dim in row["shape"]]
            digest = str(row["sha256"]).lower()
            module = self.model.get_submodule(name)
            if shape != [int(dim) for dim in module.weight.shape]:
                raise RuntimeError(
                    f"production anchor source shape differs for {name}"
                )
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise RuntimeError(
                    f"production anchor source SHA-256 is invalid for {name}"
                )
            normalized[name] = {"shape": shape, "sha256": digest}

        completed = dict(self.identity)
        completed["source_weights"] = {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": normalized,
            "identity_sha256": _canonical_json_sha256(
                normalized,
                where="production anchor source-weight identity",
            ),
        }
        cb_identity = completed.get("cb_render_identity")
        if isinstance(cb_identity, Mapping):
            cb_identity = dict(cb_identity)
            cb_qnames = list(cb_identity["cb_formats_by_qname"])
            cb_shapes = {
                name: normalized[name]["shape"] for name in cb_qnames
            }
            cb_content = {
                name: normalized[name]["sha256"] for name in cb_qnames
            }
            cb_identity.update({
                "source_weights_complete": True,
                "source_weights_shapes": cb_shapes,
                "source_weights_content_sha256": cb_content,
                "source_weights_sha256": _combined_source_weights_sha256(
                    cb_shapes, cb_content
                ),
                "render_scope": "sparse_production_anchors",
            })
            validate_cb_render_identity_metadata(
                cb_identity,
                expected_context=self.cb_serialization_context,
                expected_formats_by_qname=self.formats_by_qname,
                require_source_complete=True,
                where="completed production anchor renderer",
            )
            completed["cb_render_identity"] = cb_identity
        self.identity = canonical_json(
            completed,
            where="completed production anchor renderer identity",
        )
        return self.identity


def _render_packed_layer(
    model: nn.Module,
    layer_experts_qnames: Sequence[str],
    *,
    act_index: ActivationIndex,
    cache: ProductionWeightCache,
    render_assignment: Mapping[str, str],
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    module_token_budget: int,
    max_rows_per_expert: int,
    render_mode: str,
    col_weights: Mapping[str, torch.Tensor] | None,
    cb_serialization_context,
    progress: bool,
) -> dict:
    """Render this layer's packed experts via the shared packed-expert path,
    feeding each experts-module's input snapshot from the probe act cache."""
    module_acts: dict[str, torch.Tensor] = {}
    for experts_qname in layer_experts_qnames:
        if experts_qname not in act_index:
            continue
        module_acts[experts_qname] = act_index.load(experts_qname)
    if not module_acts:
        return {}
    return fill_packed_expert_cache_entries(
        cache, model, None,
        render_assignment=render_assignment,
        levers=levers,
        profile=profile,
        module_token_budget=module_token_budget,
        max_rows_per_expert=max_rows_per_expert,
        cache_dir=cache_dir_path,
        render_mode=render_mode,
        module_acts_override=module_acts,
        col_weights=col_weights,
        cb_serialization_context=cb_serialization_context,
        progress=progress,
    )


def _experts_qnames_by_layer(
    model: nn.Module, profile, layers_prefix: str, num_layers: int,
) -> dict[int | None, list[str]]:
    """Group packed-experts module names by decoder-layer index (structure
    only — safe while layers are on meta)."""
    from prismaquant.sensitivity_probe import _is_packed_experts_module

    out: dict[int | None, list[str]] = defaultdict(list)
    for name, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        out[_layer_index_of(name, layers_prefix)].append(name)
    return out


def _streaming_cb_render_scope(
    model: nn.Module,
    *,
    dense_modules: Mapping[str, nn.Module],
    experts_by_layer: Mapping[int | None, Sequence[str]],
    render_assignment: Mapping[str, str],
    assignment_nonbf16: Mapping[str, str],
    requested_formats: Sequence[str],
    render_formats_by_qname: Mapping[str, Sequence[str]],
    render_scope: str,
    profile,
) -> dict[str, tuple[str, ...]]:
    """Resolve the exact live-qname CB scope before the first shard write."""
    from prismaquant.sensitivity_probe import _packed_experts_param_names

    scope: dict[str, tuple[str, ...]] = {}
    if render_scope == "format-menu":
        del requested_formats
        scope.update({
            qname: tuple(
                fmt for fmt in render_formats_by_qname.get(qname, ())
                if _is_cb_format_name(fmt)
            )
            for qname in dense_modules
            if any(
                _is_cb_format_name(fmt)
                for fmt in render_formats_by_qname.get(qname, ())
            )
        })
    else:
        for qname in dense_modules:
            fmt = assignment_nonbf16.get(qname)
            if fmt is not None and _is_cb_format_name(fmt):
                scope[qname] = (fmt,)

    if render_scope != "assignment":
        return dict(sorted(scope.items()))

    modules = dict(model.named_modules())
    for experts_qname in sorted({
        name for names in experts_by_layer.values() for name in names
    }):
        mod = modules[experts_qname]
        for pn in _packed_experts_param_names(mod, profile):
            full = f"{experts_qname}.{pn}" if experts_qname else pn
            try:
                recipe_key = profile.live_to_recipe_name(full)
            except Exception:
                recipe_key = full
            fmt = render_assignment.get(recipe_key)
            if fmt is None and recipe_key != full:
                fmt = render_assignment.get(full)
            if fmt is None:
                continue
            canonical = _canon_fmt(fmt)
            if _is_cb_format_name(canonical):
                scope[full] = (canonical,)
    return dict(sorted(scope.items()))


def run_streaming_render(
    model: nn.Module,
    *,
    layers_prefix: str,
    num_layers: int,
    render_assignment: Mapping[str, str] | None,
    act_index: ActivationIndex,
    formats: Sequence[str],
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    skip_tokens: Sequence[str],
    device: torch.device,
    expert_render_mode: str = "batched",
    expert_module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    h_detail_dir: str | Path | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    cb_serialization_context=None,
    render_scope: str = "assignment",
    retain_rendered: bool | None = None,
    consume_render: Callable[..., Mapping[str, object]] | None = None,
    consumer_identity: Mapping[str, object] | None = None,
    calibration_hash: str | None = None,
    resume: bool = False,
    max_act_rows: int = 512,
    include_qnames: Sequence[str] | None = None,
    format_plan: Mapping[str, Sequence[str]] | None = None,
    format_plan_identity: str | None = None,
    install=None,
    unload=None,
    set_priority=None,
    progress: bool = True,
) -> ProductionWeightCache:
    """Layer-by-layer render loop over a (streaming or resident) model.

    ``install``/``unload``/``set_priority`` are the ``StreamingContext`` hooks
    for a real streamed checkpoint. When ``None`` (an already-resident model,
    used by the tests) the loop just renders each layer in place — the render
    math is identical, only weight residency differs.
    """
    if render_scope not in {"assignment", "format-menu"}:
        raise ValueError(f"unsupported streaming render_scope={render_scope!r}")
    if render_scope == "assignment" and render_assignment is None:
        raise ValueError("assignment streaming render requires render_assignment")
    if retain_rendered is None:
        retain_rendered = render_scope == "assignment"
    if render_scope == "assignment" and not retain_rendered:
        raise ValueError(
            "assignment streaming render must retain the selected weights"
        )
    if max_act_rows < 1:
        raise ValueError("max_act_rows must be positive")

    levers = _resolve_production_render_levers(levers)
    mechanism_plan = _resolve_render_mechanism_plan(levers)
    render_assignment = dict(render_assignment or {})
    assignment_nonbf16 = _nonbf16_assignment(render_assignment)
    requested_formats = tuple(
        dict.fromkeys(_canon_fmt(f) for f in formats if str(f).strip())
    )
    if render_scope == "format-menu" and not retain_rendered:
        unsupported = [
            fmt for fmt in requested_formats
            if fmt != "BF16" and not _is_cb_format_name(fmt)
        ]
        if unsupported:
            raise ValueError(
                "transient streamed format-menu is proven deterministic only "
                f"for CB formats; unsupported={unsupported}"
            )
        if consume_render is not None and consumer_identity is None:
            raise ValueError(
                "a custom streamed render consumer requires an exact "
                "consumer_identity"
            )
        if consumer_identity is None:
            consumer_identity = {
                "schema": (
                    "prismaquant.production_weight_cache."
                    "production_render_score_consumer.v1"
                ),
                "consumer": "production_render_score",
            }

    cache = ProductionWeightCache(
        weights={},
        levers=dict(levers),
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={},
    )

    fused_mapping = (
        _fused_sibling_leaf_mapping_from_profile(profile)
        if profile is not None else {}
    )
    fisher_rows = (
        _FisherRowWeightCache(h_detail_dir, fused_mapping or None)
        if (bool(levers.get("fisher_gptq", False)) and h_detail_dir)
        else None
    )

    include_qname_set = (
        {str(name) for name in include_qnames}
        if include_qnames is not None else None
    )
    dense_modules = _eligible_dense_qname_modules(
        model,
        profile,
        skip_tokens,
        include_qname_set,
    )
    if render_scope == "assignment":
        render_formats_by_qname = {
            qname: (fmt,)
            for qname, fmt in assignment_nonbf16.items()
            if qname in dense_modules
        }
    else:
        menu = tuple(fmt for fmt in requested_formats if fmt != "BF16")
        render_formats_by_qname = {
            qname: menu for qname in dense_modules if menu
        }
    if format_plan is not None:
        canonical_plan = {
            str(qname): tuple(dict.fromkeys(
                _canon_fmt(fmt) for fmt in formats
            ))
            for qname, formats in format_plan.items()
        }
        missing = sorted(set(dense_modules) - set(canonical_plan))
        if missing:
            raise ValueError(
                "streaming format plan does not cover every selected live "
                f"Linear; sample={missing[:8]}"
            )
        planned_universe = {
            fmt for formats in canonical_plan.values() for fmt in formats
        }
        if render_scope == "assignment":
            illegal = sorted(
                (qname, formats[0])
                for qname, formats in render_formats_by_qname.items()
                if formats[0] in planned_universe
                and formats[0] not in canonical_plan[qname]
            )
            if illegal:
                raise ValueError(
                    "streaming assignment contains source-rate-illegal "
                    f"format-plan cells; sample={illegal[:8]}"
                )
        else:
            render_formats_by_qname = {
                qname: tuple(
                    fmt for fmt in formats
                    if fmt not in planned_universe
                    or fmt in canonical_plan[qname]
                )
                for qname, formats in render_formats_by_qname.items()
            }
            render_formats_by_qname = {
                qname: formats
                for qname, formats in render_formats_by_qname.items()
                if formats
            }
    per_layer_dense: dict[int | None, dict[str, nn.Module]] = defaultdict(dict)
    for qname, mod in dense_modules.items():
        per_layer_dense[_layer_index_of(qname, layers_prefix)][qname] = mod
    per_layer_experts = _experts_qnames_by_layer(
        model, profile, layers_prefix, num_layers,
    )

    cb_scope = _streaming_cb_render_scope(
        model,
        dense_modules=dense_modules,
        experts_by_layer=per_layer_experts,
        render_assignment=render_assignment,
        assignment_nonbf16=assignment_nonbf16,
        requested_formats=requested_formats,
        render_formats_by_qname=render_formats_by_qname,
        render_scope=render_scope,
        profile=profile,
    )
    _extend_production_cache_cb_render_identity(
        cache,
        cb_scope,
        cb_serialization_context=cb_serialization_context,
        col_weights=col_weights,
        render_levers=levers,
        render_mechanism_plan=mechanism_plan,
    )
    cb_git_commit = _production_cache_git_commit() if cb_scope else None
    cb_producer_source_sha256 = (
        _production_cache_source_sha256() if cb_scope else None
    )
    render_score_records: dict[str, dict[str, object]] = {}
    transient_results: dict[str, dict[str, object]] = {}
    cb_pair_identities: dict[tuple[str, str], dict[str, object]] = {}
    cb_pair_artifacts: dict[str, dict[str, object]] = {}
    coverage: dict[str, dict[str, object]] = {}

    def _process_layer(L: int | None) -> None:
        dense = per_layer_dense.get(L, {})
        experts = per_layer_experts.get(L, [])
        if not dense and not experts:
            return
        did_install = False
        if L is not None and install is not None:
            if set_priority is not None:
                set_priority({L})
            install(L)
            did_install = True
        try:
            _render_dense_layer(
                model, dense,
                render_formats_by_qname=render_formats_by_qname,
                act_index=act_index,
                cache=cache,
                levers=levers,
                cache_dir_path=cache_dir_path,
                profile=profile,
                device=device,
                fisher_rows=fisher_rows,
                render_score_records=render_score_records,
                col_weights=col_weights,
                cb_serialization_context=cb_serialization_context,
                retain_rendered=bool(retain_rendered),
                consume_render=consume_render,
                consumer_identity=consumer_identity,
                calibration_hash=calibration_hash,
                resume=resume,
                max_act_rows=max_act_rows,
                cb_pair_identities=cb_pair_identities,
                cb_pair_artifacts=cb_pair_artifacts,
                transient_results=transient_results,
                cb_git_commit=cb_git_commit,
                cb_producer_source_sha256=cb_producer_source_sha256,
                progress=progress,
            )
            if experts and render_scope == "assignment":
                cov = _render_packed_layer(
                    model, experts,
                    act_index=act_index,
                    cache=cache,
                    render_assignment=render_assignment,
                    levers=levers,
                    cache_dir_path=cache_dir_path,
                    profile=profile,
                    module_token_budget=expert_module_token_budget,
                    max_rows_per_expert=max_rows_per_expert,
                    render_mode=expert_render_mode,
                    col_weights=col_weights,
                    cb_serialization_context=cb_serialization_context,
                    progress=progress,
                )
                coverage.update(cov)
        finally:
            if did_install:
                unload(L)
                if set_priority is not None:
                    set_priority(set())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for L in range(num_layers):
        _process_layer(L)
    # Head / root-level Linears (rare — lm_head is normally pinned-skipped) are
    # resident throughout; render them last with no install.
    _process_layer(None)

    if cb_scope:
        bind_production_cache_cb_source_weights(
            cache,
            {},
            require_complete=True,
            where="streaming ProductionWeightCache final source binding",
        )

    # The packed-expert append writes its own render-score records, coverage
    # and counters onto the cache as each layer is rendered
    # (``_finalize_packed_expert_cache_metadata``). This finalizer must MERGE
    # with them: ``render_formats_by_qname`` is dense-only (it is built from
    # ``dense_modules``), so a plain overwrite would drop every packed record
    # and under-count ``requested_entries`` by the packed key count — the
    # exact union then refuses the shard it just built.
    if coverage or "packed_expert_coverage" in cache.metadata:
        merged_coverage = dict(cache.metadata.get("packed_expert_coverage") or {})
        merged_coverage.update(coverage)
        cache.metadata["packed_expert_coverage"] = merged_coverage
    existing_scores = cache.metadata.get("render_scores")
    packed_score_records = (
        dict(existing_scores.get("records") or {})
        if isinstance(existing_scores, Mapping) else {}
    )
    merged_score_records = {**packed_score_records, **render_score_records}
    # Keep the dense plan sum as the dense-hole detector it is; add only the
    # packed records the packed path actually produced.
    requested_entries = sum(
        len(fmts) for fmts in render_formats_by_qname.values()
    ) + len(packed_score_records)
    cache.metadata.update({
        "render_scope": render_scope,
        "render_retention": (
            "materialized" if retain_rendered else "transient-consumed"
        ),
        "requested_formats": list(requested_formats),
        "requested_entries": int(requested_entries),
        "streaming": True,
        "calib_hash": calibration_hash,
        "format_plan_identity_sha256": format_plan_identity,
        "render_mechanism_order": [
            {
                "name": spec.name,
                "operation": spec.operation,
                "scope": spec.scope,
                "gate_metric": spec.gate_metric,
            }
            for spec in mechanism_plan.ordered
        ],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": int(len(merged_score_records)),
            "records": dict(sorted(merged_score_records.items())),
        },
    })
    if not retain_rendered:
        cache.metadata["transient_render_artifacts"] = {
            "schema": (
                "prismaquant.production_weight_cache."
                "transient_render_artifacts.v1"
            ),
            "entries": int(len(cb_pair_artifacts)),
            "records": dict(sorted(cb_pair_artifacts.items())),
            "consumer_identity": dict(consumer_identity or {}),
            "consumer_results": dict(sorted(transient_results.items())),
        }
    if cb_pair_identities:
        canonical_pairs = {
            f"{qname}|{fmt}": pair
            for (qname, fmt), pair in sorted(cb_pair_identities.items())
        }
        cache.metadata["cb_cache_pair_identity"] = {
            "schema": "prismaquant.production_weight_cache.cb_pair_set.v1",
            "pair_schema": CB_CACHE_PAIR_IDENTITY_SCHEMA,
            "entries": len(canonical_pairs),
            "identity_sha256": _canonical_json_sha256(
                canonical_pairs,
                where="streaming CB cache pair identity set",
            ),
            "published_entries": len(cb_pair_artifacts),
            "artifact_sha256": _canonical_json_sha256(
                cb_pair_artifacts,
                where="streaming CB cache pair artifact set",
            ),
            "calibration_hashes": sorted({
                str(pair["calibration_hash"])
                for pair in canonical_pairs.values()
            }),
            "git_commits": sorted({
                str(pair["git_commit"])
                for pair in canonical_pairs.values()
            }),
            "producer_source_sha256": sorted({
                str(pair["producer_source_sha256"])
                for pair in canonical_pairs.values()
            }),
        }
    if cache.failed:
        cache.metadata["render_failures"] = {
            f"{q}|{fmt}": str(err)
            for (q, fmt), err in sorted(cache.failed.items())
        }
    return cache


def _priority_setter(ctx):
    """Set the resident-layer priority AND re-arm the memory-pressure floor
    before each install — mirrors the streaming cost path so a UMA-pressured
    box pre-evicts instead of OOMing on the layer read."""
    def _set(layers: set) -> None:
        ctx.layer_cache.set_priority_layers(layers)
        ctx.configure_runtime_pressure_floor()
    return _set


def fill_production_weight_cache_streaming(
    model_path: str,
    *,
    render_assignment: Mapping[str, str] | None,
    activation_cache_dir: str | Path,
    formats: Sequence[str],
    levers: Mapping[str, object] | None,
    cache_dir: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_tokens: Sequence[str] | None = None,
    expert_render_mode: str = "batched",
    expert_module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    h_detail_dir: str | Path | None = None,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    cb_serialization_context=None,
    render_scope: str = "assignment",
    retain_rendered: bool | None = None,
    consume_render: Callable[..., Mapping[str, object]] | None = None,
    consumer_identity: Mapping[str, object] | None = None,
    calibration_hash: str | None = None,
    resume: bool = False,
    max_act_rows: int = 512,
    include_qnames: Sequence[str] | None = None,
    format_plan: Mapping[str, Sequence[str]] | None = None,
    format_plan_identity: str | None = None,
    offload_folder: str | Path | None = None,
    progress: bool = True,
) -> ProductionWeightCache:
    """Build a production δw cache one decoder layer at a time.

    No whole-model ``from_pretrained``: a ``StreamingContext`` keeps only the
    head resident and streams each decoder layer's weights on demand. Requires
    ``cache_dir`` (disk streaming — the pickle is a manifest) and a probe
    activation cache produced with the same calibration.
    """
    device = torch.device(device)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    if offload_folder is None:
        offload_folder = cache_dir_path / "streaming_offload"

    ctx = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=str(offload_folder),
        log_prefix="[stream-prod-cache]",
    )
    try:
        model = ctx.model
        try:
            from prismaquant.model_profiles import profile_from_model
            profile = profile_from_model(model)
        except Exception:
            profile = None
        if skip_tokens is None:
            skip_tokens = (
                list(profile.pinned_names())
                if profile is not None
                and hasattr(profile, "pinned_names")
                else []
            )
        act_index = ActivationIndex(Path(activation_cache_dir), [])
        cache = run_streaming_render(
            model,
            layers_prefix=ctx.layers_prefix,
            num_layers=ctx.num_layers,
            render_assignment=render_assignment,
            act_index=act_index,
            formats=formats,
            levers=levers,
            cache_dir_path=cache_dir_path,
            profile=profile,
            skip_tokens=skip_tokens,
            device=device,
            expert_render_mode=expert_render_mode,
            expert_module_token_budget=expert_module_token_budget,
            max_rows_per_expert=max_rows_per_expert,
            h_detail_dir=h_detail_dir,
            col_weights=col_weights,
            cb_serialization_context=cb_serialization_context,
            render_scope=render_scope,
            retain_rendered=retain_rendered,
            consume_render=consume_render,
            consumer_identity=consumer_identity,
            calibration_hash=calibration_hash,
            resume=resume,
            max_act_rows=max_act_rows,
            include_qnames=include_qnames,
            format_plan=format_plan,
            format_plan_identity=format_plan_identity,
            install=ctx.install,
            unload=ctx.unload,
            set_priority=_priority_setter(ctx),
            progress=progress,
        )
    finally:
        ctx.shutdown()
    return cache
