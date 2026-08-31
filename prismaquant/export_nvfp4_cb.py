"""Materialize a PrismaQuant recipe as an NVFP4-CB / FP8-CB checkpoint.

Sibling of :mod:`prismaquant.export_gguf` — the same skeleton-requantize
strategy, but the container is safetensors + a custom compressed-tensors-**style**
``quant_config.json`` whose scheme vocabulary (``nvfp4_cb`` / ``fp8_cb``) only
the out-of-tree vLLM plugin understands (docs/lanes/nvfp4-cb/serving-kernel.md
§2). It is explicitly **not** stock compressed-tensors (whose schemes cannot
express codebooks) — do not route a CB assignment through
:mod:`prismaquant.export_native_compressed`; that exporter hard-fails on CB.

Pipeline: read the bf16 HF skeleton (config.json + *.safetensors), VQ-pack each
target Linear with the **same** weighted closure the cost measured
(:func:`prismaquant.nvfp4_cb_formats.nvfp4_cb_pack`), copy every non-target
tensor verbatim (bf16 passthrough), and emit:

  * ``<name>.cb_qweight``  uint8 (rows, bytes_per_row) — the §1 superblock byte
    stream (index bits + fp4 versioned scale plane: production-v2 two-tier,
    explicit legacy-v1 E4M3-direct; fp8 index bits only);
  * ``<name>.weight_scale`` fp32 (out_features,) — fp8 families only (fp8 has no
    on-disk scale plane; the plane is per-output-channel);
  * ``<name>.input_global_scale`` fp32 (1,) — production-contracted FP4-CB
    targets only, calibrated under the sole top-level static W4A4 execution
    contract; historical/research payloads omit both scalar and contract;
  * ``cb_codebook.<ref>.<fmt>[.sub{i}]`` fp16 — the resolved codebook, shipped
    **once** per (ref, format): ``ref = "lattice"`` for the fixed lattice,
    ``ref = "<role>"`` for a shared per-(role) learned codebook;
  * ``config.json`` (verbatim + a ``quantization_config`` pointer) and
    ``quant_config.json`` (the custom scheme + provenance).

Bit-layout + tensor-naming + config-schema contract: docs/lanes/nvfp4-cb/LAYOUT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from functools import wraps
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import (
    CB_FORMAT_NAMES,
    family_for,
    parse_format_name,
    parse_producer_format_name,
    subtable_bit_widths,
)
from prismaquant.cb_export_config import (
    build_quant_config,
    build_quantized_embedding_declaration,
    codebook_tensor_names as _codebook_tensor_names,
    codebook_tensors as _codebook_tensors,
)
from prismaquant.trellis_formats import (
    ALL_LEGAL_TRELLIS_FORMAT_NAMES as _ALL_LEGAL_TRELLIS_FORMAT_NAMES,
    E2M1_FAMILY as _TRELLIS_E2M1_FAMILY,
    E4M3_FAMILY as _TRELLIS_E4M3_FAMILY,
    parse_trellis_format_name as _parse_trellis_format_name,
)
from prismaquant.trellis_wire import TrellisWire as _TrellisWire
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown as _trellis_breakdown
from prismaquant.layer_config import load_assignment, read_layer_config_metadata
from prismaquant.shard_layout import (
    DEFAULT_SHARD_BYTES,
    container_names,
    plan_shards,
    tensor_payload_identity,
    write_shard_index,
)
from prismaquant.export_output_safety import (
    prepare_fresh_export_directory,
    transactional_directory_output,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_payload_summary,
    cb_serialization_metadata_from_assignment_payload,
    cb_serialization_context_from_env,
    codebook_source_for_format,
    effective_codebook_source_scope,
    scale_sweep_for_format,
    cb_tensor_payload_breakdown,
    finalize_cb_export_artifact_inventory,
    resolve_cb_encode_tier,
    whole_artifact_budget_from_assignment_payload,
    validate_cb_sidecar_tensors,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)
from prismaquant.prismasnap_contract import refuse_prismasnap_lane_before_output
from prismaquant.nvfp4_activation_contract import (
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_EXECUTION,
    build_execution_contract,
    calibrated_input_global_scales_with_sources,
    input_global_scale_tensor,
    resolve_input_global_scale_policy,
)
from prismaquant.routed_moe_codebooks import (
    ROUTED_MOE_CBL_BANK_RUNGS,
    RoutedMoECodebookRole,
    describe_split_book_refusal,
    fused_targets_with_split_books,
    learned_role_qnames_for_packed,
    logical_role_qname,
    split_role_rows,
    stacked_role_col_weights,
)

# This exporter's own declaration of what the mixed CB container can carry —
# exactly the coverage gate in `export_cb` below: the CB rung families, the two
# stock-CT schemes the plugin delegates to vLLM's CompressedTensors path
# (NVFP4, FP8_E4M3 <- FP8_DYNAMIC), the verbatim FP8_SOURCE passthrough, the
# BF16 container passthrough, and the Gridbook trellis families
# (TCQ_E2M1_Rxxx / TCQ_E4M3_Rxxx) whose wire is the only carrier. The
# `nvfp4_cb` serving profile's export lane derives its format menu from this
# constant (serving_profile_specs/nvfp4_cb.json), so the allocator can never
# spend budget on a rung this exporter would hard-fail on.
EXPORTABLE_FORMATS = CB_FORMAT_NAMES | frozenset(
    {"NVFP4", "FP8_E4M3", "FP8_SOURCE", "BF16"}
) | frozenset(_ALL_LEGAL_TRELLIS_FORMAT_NAMES)


def _preflight_assignment_before_output_transaction(function):
    """Validate assignment syntax and target policy before staging output.

    The exporter deliberately reloads the assignment inside the transaction as
    part of its ordinary render path.  This outer preflight exists so an
    unsupported public format spelling or a format denied by the allocator's
    stamped target profile fails without creating a destination or a preserved
    ``.tmp-*`` sibling.  The explicit environment override matches the native
    exporter; unstamped legacy recipes retain their historical generic path.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        layer_config_path = bound.arguments["layer_config_path"]
        assignment = load_assignment(layer_config_path)
        metadata = read_layer_config_metadata(layer_config_path)
        target_profile = str(
            os.environ.get("PRISMAQUANT_TARGET_PROFILE")
            or metadata.get("target_profile")
            or ""
        ).strip()
        if target_profile:
            from prismaquant.serving_profiles import check_serving_format

            refused = []
            for qname, fmt in assignment.items():
                # Trellis is gated by lane_eligibility (gridbook.lane-eligibility.v3),
                # not by the CB profile's format_rules. The CB profile's allow list
                # deliberately does not enumerate the 2546 trellis rungs, and
                # check_serving_format would refuse them before the lane gate
                # ever runs. Skip trellis here; the per-artifact lane gate below
                # is the authority for those bytes (principle 14).
                if _is_trellis_fmt(str(fmt)):
                    continue
                decision = check_serving_format(target_profile, qname, fmt)
                if not decision.legal:
                    refused.append({
                        "qname": qname,
                        "format": fmt,
                        "reason": decision.reason,
                        "rule": decision.rule,
                    })
            if refused:
                raise ValueError(
                    f"{function.__name__}: target profile {target_profile!r} "
                    "refuses assignment format(s) before output transaction: "
                    f"{refused[:8]}"
                )
        return function(*args, **kwargs)

    return wrapped


def _git_commit() -> str:
    from prismaquant.aura_cost import _git_commit as _aura_git_commit

    return _aura_git_commit() or "unknown"


def _parse_cb_format(fmt: str) -> tuple[str, str, int] | None:
    """``NVFP4_CB_K{k}`` -> (fp4, product, k); ``FP8_CB_K{k}`` -> (fp8,
    product, k). None for non-CB. (``NVFP4_CB_S{k}`` was the signed family,
    deleted 2026-08-17; `parse_format_name` no longer recognizes it.)"""
    parsed = parse_format_name(str(fmt).strip().upper())
    if parsed is None:
        return None
    family, k = parsed
    return family.grid, family.mode, k


def _parse_producer_cb_format(fmt: str) -> tuple[str, str, int] | None:
    """Parse only CB rungs legal in a newly emitted artifact."""

    parsed = parse_producer_format_name(str(fmt).strip().upper())
    if parsed is None:
        return None
    family, k = parsed
    return family.grid, family.mode, k


def _role_of(qname: str) -> str:
    """Shared-codebook grouping key — the Linear's projection role (last qname
    component), e.g. ``model.layers.3.mlp.gate_proj`` -> ``gate_proj``."""
    return qname.split(".")[-1]


# ---------------------------------------------------------------------------
# Trellis helpers — the Gridbook trellis wire is the only carrier (WO-C rule 1)
# ---------------------------------------------------------------------------

def _parse_trellis_format(fmt: str):
    """Return (TrellisFamily, body_rate_q256) or None for non-trellis."""
    return _parse_trellis_format_name(str(fmt).strip().upper())


def _is_trellis_fmt(fmt: str) -> bool:
    return _parse_trellis_format(fmt) is not None


def _trellis_fused_group_key(qname: str, profile) -> str | None:
    """Reuse the single fused-sibling grouping the allocator uses.

    WO-C rule 3: vLLM merges q/k/v and gate/up and per-role wires cannot be
    concatenated — each carries its own alphabets, schedule and padding.
    gridbook/config.py refuses such a target *by name*. This helper mirrors
    that hard runtime fact rather than inventing a second rule.
    """
    try:
        from prismaquant.nvfp4_activation_contract import (
            fused_sibling_group_key as _fused_key,
        )

        return _fused_key(qname, profile=profile, tolerate_profile_errors=False)
    except Exception:
        return None


def _is_routed_moe_trellis_target(qname: str, profile) -> bool:
    """Dense only (WO-C rule 4). The pinned contract publishes no routed_moe
    trellis cell; a routed/packed-MoE unit assigned a trellis rung must fail
    export closed naming the missing cell.

    Detection reuses the profile when present, otherwise falls back to the
    lexical “.experts.” convention. This is intentionally conservative:
    false-positive dense is fail-closed via the lane gate, false-positive MoE
    is a loud shape failure, so neither silently ships.
    """
    if profile is not None:
        try:
            packed = getattr(profile, "packed_expert_param_names", lambda: frozenset())()
            if packed:
                # Any qname that resolves to a packed expert parent is MoE.
                per_expert_regex = getattr(profile, "per_expert_moe_regex", lambda: None)()
                if per_expert_regex:
                    import re as _re

                    pat = _re.compile(
                        per_expert_regex[len("re:") :]
                        if per_expert_regex.startswith("re:")
                        else per_expert_regex
                    )
                    # Check both raw and profile-mapped live name
                    candidates = [str(qname)]
                    try:
                        mapped = profile.checkpoint_to_live_name(str(qname) + ".weight")
                        if mapped:
                            candidates.append(str(mapped))
                    except Exception:
                        pass
                    for cand in candidates:
                        base = cand[:-7] if cand.endswith(".weight") else cand
                        if pat.match(base) or ".experts." in base:
                            return True
        except Exception:
            pass
    return ".experts." in str(qname)


def _load_trellis_wire_bytes(
    qname: str,
    fmt: str,
    trellis_wire_cache: dict | None,
) -> bytes | None:
    """Return wire bytes for one trellis unit from the ProductionWeightCache seam.

    WO-C C1: The wire bytes come from ProductionWeightCache (WO-B retention).
    If WO-B is not merged, code against the seam: try the explicit
    trellis_wire_cache dict first, then ProductionWeightCache.get_trellis_wire_bytes
    if that method exists, then fail closed. Do NOT re-encode.
    """
    # 1) Explicit dict supplied by caller (test seam or staged cache)
    if trellis_wire_cache is not None:
        # dict may be keyed by (qname, fmt) or qname alone
        key = (str(qname), str(fmt).strip().upper())
        alt = str(qname)
        if key in trellis_wire_cache:
            val = trellis_wire_cache[key]
            return bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else val.numpy().tobytes() if hasattr(val, "numpy") else bytes(val)
        if alt in trellis_wire_cache:
            val = trellis_wire_cache[alt]
            return bytes(val) if isinstance(val, (bytes, bytearray, memoryview)) else val.numpy().tobytes() if hasattr(val, "numpy") else bytes(val)
        # Also try upper fmt
        for k, v in trellis_wire_cache.items():
            if isinstance(k, tuple) and k[0] == str(qname):
                return bytes(v) if isinstance(v, (bytes, bytearray, memoryview)) else bytes(v)
    # 2) ProductionWeightCache seam — WO-B
    try:
        from prismaquant.production_weight_cache import ProductionWeightCache  # type: ignore

        # If a global cache instance has been stashed on the exporter via
        # environment or attribute, try it. Otherwise, this is a seam call
        # that will fail closed below.
        # The production path will set trellis_wire_cache explicitly, so this
        # branch is only for the merged WO-B world where the cache object
        # itself is passed as trellis_wire_cache.
        if hasattr(trellis_wire_cache, "get_trellis_wire_bytes"):
            b = trellis_wire_cache.get_trellis_wire_bytes(str(qname), str(fmt))  # type: ignore[attr-defined]
            if b is not None:
                return bytes(b)
    except Exception:
        pass
    return None


def _load_skeleton(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from a HF safetensors dir (single file or sharded)."""
    index = model_dir / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index.exists():
        shards = sorted({
            v for v in json.loads(index.read_text())["weight_map"].values()
        })
        for shard in shards:
            tensors.update(load_file(str(model_dir / shard)))
        return tensors
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {model_dir}")
    return load_file(str(single))


def _decoded_cb_source_weight(
    skeleton: dict[str, torch.Tensor],
    weight_key: str,
    *,
    model_weight_name: str,
    fp8_scale_inv_map,
) -> torch.Tensor:
    from prismaquant.cb_source_decode import cb_source_weight_bf16_value

    return cb_source_weight_bf16_value(
        skeleton[weight_key],
        model_weight_name=model_weight_name,
        fp8_scale_inv_map=fp8_scale_inv_map,
    )


# --- Nested-prefix skeleton name resolution (hybrid Qwen3.6-27B / Hy3 / DSv4).
# The allocator's recipe qnames are the text-only-staged names
# (`model.layers.N.*`); the on-disk checkpoint nests the LM under an infix
# (`model.language_model.layers.N.*`). The profile knows the structure and maps
# both directions — never hard-code the infix. ---

def _pack_skeleton_experts(
    skeleton: dict,
    profile,
    fp8_scale_inv_map=None,
    target_qnames: set[str] | None = None,
    member_qnames_out: dict[
        str, dict[tuple[str, int], str]
    ] | None = None,
) -> int:
    """Per-expert-on-disk MoE checkpoints (Qwen3.5-MoE / Ornith): assemble
    the packed ``<experts>.gate_up_proj/.down_proj`` skeleton tensors the CB
    targets name, via layer_streaming's tested bridge. No-op for dense or
    already-packed checkpoints.

    When ``target_qnames`` is supplied, only those packed parents are
    assembled. This is required for partial mixed assignments: an omitted or
    BF16 expert bank must stay in its original per-expert checkpoint layout,
    which is the layout architecture-specific vLLM loaders consume.

    Memory discipline: the bridge is invoked once PER packed group (the
    ``live_param_shape`` gate restricts each call), so the transient is one
    expert stack (~1 GB at 35B), not the whole model's expert bytes doubled.

    ``member_qnames_out`` optionally receives the exact canonical
    ``(projection, expert_id) -> qname`` inputs consumed for each produced
    stack.  Learned per-role books need those names to bind each role slice to
    the same per-expert imatrix identities the bundle records.  Already-packed
    rank-3 inputs have no such provenance and intentionally produce no entry.
    """
    if profile is None:
        return 0
    regex = getattr(profile, "per_expert_moe_regex", lambda: None)()
    pnames = getattr(profile, "packed_expert_param_names",
                     lambda: frozenset())()
    if not regex or not pnames:
        return 0
    from prismaquant.layer_streaming import _pack_per_expert_into_packed
    pat = re.compile(regex[len("re:"):] if regex.startswith("re:") else regex)

    def is_per_expert(name: str) -> bool:
        if pat.match(name):
            return True
        try:
            live = profile.checkpoint_to_live_name(name + ".weight")
            if live is not None and live.endswith(".weight"):
                live = live[:-len(".weight")]
            if live is not None and pat.match(live):
                return True
        except Exception:
            pass
        try:
            return bool(pat.match(profile.to_vllm_internal_name(name)))
        except Exception:
            return False

    requested = None if target_qnames is None else set(target_qnames)

    def is_requested(packed_full: str) -> bool:
        if requested is None:
            return True
        variants = {packed_full}
        canon = _canonical_qname(packed_full, profile)
        if canon is not None:
            variants.add(canon)
        return bool(variants & requested)

    def packed_source_parts(source_name: str):
        """Return ``(packed_parent, expert_id, projection)`` for either a
        checkpoint or profile-mapped live per-expert source name."""
        name = (source_name[:-len(".weight")]
                if source_name.endswith(".weight") else source_name)
        try:
            head, proj = name.rsplit(".", 1)
            experts_path, idx = head.rsplit(".", 1)
        except ValueError:
            return None
        if not idx.isdigit():
            return None
        parent = profile.packed_expert_parent_for_projection(proj)
        if parent is None:
            return None
        return f"{experts_path}.{parent}", int(idx), proj

    # Pre-derive each packed group's expected shape from its per-expert
    # members (E, sum of fused projection out-dims, in).
    members: dict[str, dict[int, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict))
    member_names: dict[str, dict[tuple[str, int], str]] = defaultdict(dict)
    for key, t in skeleton.items():
        name = key[:-len(".weight")] if key.endswith(".weight") else key
        if not is_per_expert(name):
            continue
        try:
            live_weight_name = profile.checkpoint_to_live_name(key)
        except Exception:
            live_weight_name = None
        live_weight_name = live_weight_name or key
        raw_parts = packed_source_parts(key)
        live_parts = packed_source_parts(live_weight_name)
        candidate_parents = {
            parts[0] for parts in (raw_parts, live_parts)
            if parts is not None
        }
        if requested is not None and not any(
                is_requested(parent) for parent in candidate_parents):
            continue
        if key.endswith(".weight") and (
            t.dtype == torch.float8_e4m3fn
            or live_weight_name in (fp8_scale_inv_map or {})
        ):
            raise ValueError(
                "resident CB export cannot safely assemble profile-scaled "
                "FP8/MXFP4 per-expert tensors before dequantization; use "
                "export_nvfp4_cb_streaming for this source"
            )
        if raw_parts is None:
            continue
        packed_full, idx, proj = raw_parts
        members[packed_full][idx][proj] = tuple(t.shape)
        parent = packed_full.rsplit(".", 1)[-1]
        source_order = tuple(profile.packed_expert_projection_names(parent))
        try:
            logical_order = tuple(
                profile.vllm_fused_moe_scheme_projection_names(parent)
            )
        except Exception:
            logical_order = source_order
        logical_proj = proj
        if (
            proj in source_order
            and len(source_order) == len(logical_order)
        ):
            logical_proj = logical_order[source_order.index(proj)]
        member_base = (
            live_weight_name[:-len(".weight")]
            if live_weight_name.endswith(".weight")
            else live_weight_name
        )
        if member_base.rsplit(".", 1)[-1] != logical_proj:
            member_base = member_base.rsplit(".", 1)[0] + "." + logical_proj
        member_key = (logical_proj, int(idx))
        previous = member_names[packed_full].get(member_key)
        if previous is not None and previous != member_base:
            raise ValueError(
                f"{packed_full}: learned routed member {member_key} maps to "
                f"both {previous!r} and {member_base!r}"
            )
        member_names[packed_full][member_key] = member_base
    expected: dict[str, tuple] = {}
    for packed_full, by_e in members.items():
        parent = packed_full.rsplit(".", 1)[1]
        order = tuple(profile.packed_expert_projection_names(parent))
        shapes0 = by_e[min(by_e)]
        if any(p not in shapes0 for p in order):
            continue
        out_rows = sum(shapes0[p][0] for p in order)
        in_f = shapes0[order[0]][1]
        expected[packed_full] = (max(by_e) + 1, out_rows, in_f)

    produced = 0
    for packed_full, shape in expected.items():
        n = _pack_per_expert_into_packed(
            skeleton,
            is_per_expert=is_per_expert,
            parent_for_projection=profile.packed_expert_parent_for_projection,
            projection_names_for=profile.packed_expert_projection_names,
            live_param_shape=(
                lambda name, _t=packed_full, _s=shape:
                _s if name == _t else None),
        )
        if packed_full in skeleton:
            skeleton[packed_full + ".weight"] = skeleton.pop(packed_full)
        if member_qnames_out is not None:
            mapping = dict(member_names.get(packed_full, {}))
            if mapping:
                variants = {packed_full}
                canonical = _canonical_qname(packed_full, profile)
                if canonical is not None:
                    variants.add(canonical)
                for variant in variants:
                    previous = member_qnames_out.get(variant)
                    if previous is not None and previous != mapping:
                        raise ValueError(
                            f"{variant}: conflicting routed member maps"
                        )
                    member_qnames_out[variant] = mapping
        produced += n
    if produced:
        print(f"[export-cb] packed {produced} per-expert MoE groups into "
              f"stacked skeleton tensors")
    return produced


def _try_resolve_direct_packed_expert(qname, skeleton, profile):
    """Resolve a direct packed expert parameter key (no ``.weight`` suffix).

    Packed expert containers expose 3-D ``nn.Parameter`` objects directly on
    some HF models. A checkpoint saved from that live representation therefore
    legitimately contains ``...experts.gate_up_proj`` rather than
    ``...experts.gate_up_proj.weight``. Accept only profile-declared packed
    parents under an ``experts`` container, and require rank 3 so an unrelated
    suffix-less tensor can never be mistaken for a quantization target.
    """
    if profile is None:
        return None
    try:
        packed_names = frozenset(profile.packed_expert_param_names())
    except Exception:
        return None
    if not packed_names:
        return None
    candidates = [qname]
    try:
        mapped = profile.source_tensor_name(qname)
        if mapped not in candidates:
            candidates.append(mapped)
    except Exception:
        pass
    for key in candidates:
        if key not in skeleton or "." not in key:
            continue
        parent, leaf = key.rsplit(".", 1)
        if not parent.endswith(".experts") or leaf not in packed_names:
            continue
        if hasattr(skeleton, "get_shape"):
            shape = tuple(skeleton.get_shape(key))
        else:
            shape = tuple(skeleton[key].shape)
        if len(shape) != 3:
            raise ValueError(
                f"{qname}: direct packed expert source {key!r} must be rank-3 "
                f"[experts, out_features, in_features], got {shape}")
        return key
    return None


def _source_tensor_key(qname, profile, suffix=".weight"):
    """Profile-mapped CHECKPOINT key for the tensor ``qname + suffix``.

    The naming rules are written against whole TENSOR names: DSv4's shared- and
    routed-expert renames anchor on a following ``.`` (``ffn.shared_experts.
    gate_proj.`` -> ``ffn.shared_experts.w1.``) and hy3's router/expert-bias
    rules anchor on ``.weight$``. Mapping the BARE module qname and appending
    the suffix afterwards therefore skips every such rule; the suffix has to be
    attached BEFORE the rewrite. On DSv4-Flash the bare-name convention left
    33,153 of 33,325 selectable leaves (129 shared-expert + 33,024 routed-expert
    projections) resolving to ``gate_proj``/``up_proj``/``down_proj``
    checkpoint keys that do not exist.
    """
    return profile.source_tensor_name(qname + suffix)


def _source_module_name(qname, profile):
    """Profile-mapped CHECKPOINT module base for ``qname`` (no tensor suffix).

    Derived through the ``.weight`` tensor name so the trailing-dot- and
    ``.weight$``-anchored rules fire, then stripped back to the module base.
    A rule that rewrites the suffix itself (none today) falls back to the
    bare-name mapping rather than returning a truncated key."""
    mapped = _source_tensor_key(qname, profile, ".weight")
    if mapped.endswith(".weight"):
        return mapped[: -len(".weight")]
    return profile.source_tensor_name(qname)


def _try_resolve_skeleton(qname, skeleton, profile, suffix=".weight"):
    """Recipe qname -> actual skeleton key, or None if neither the direct name
    nor the profile-mapped (checkpoint-convention) name is present. For the
    default weight lookup, also accepts a validated direct 3-D packed-expert
    parameter key (the native LFM live/save representation)."""
    direct = qname + suffix
    if direct in skeleton:
        return direct
    if profile is not None:
        mapped = _source_tensor_key(qname, profile, suffix)
        if mapped in skeleton:
            return mapped
    if suffix == ".weight":
        return _try_resolve_direct_packed_expert(
            qname, skeleton, profile)
    return None


def _resolve_skeleton(qname, skeleton, profile, suffix=".weight"):
    """Strict `_try_resolve_skeleton`: raise listing both names tried."""
    key = _try_resolve_skeleton(qname, skeleton, profile, suffix)
    if key is not None:
        return key
    tried = [qname + suffix]
    if profile is not None:
        tried.append(_source_tensor_key(qname, profile, suffix))
        if suffix == ".weight":
            tried.extend([qname, _source_module_name(qname, profile)])
    raise KeyError(
        f"{qname}: no skeleton tensor for {suffix!r} (tried {tried})")


def _export_base_name(qname, profile, skeleton=None, *,
                      assume_resolvable=False):
    """Recipe qname -> the base name the EXPORTED tensor + its config_groups
    target must carry. The profile's checkpoint mapping is only TRUSTED when
    the mapped name actually resolves in the skeleton — a text-only snapshot
    inside a multimodal config shell (qwen35-0.8B: Qwen3_5ForConditional-
    Generation + text_config but model.layers.* keys) otherwise gets every
    config target mis-namespaced under model.language_model.* while the
    tensor writer's fallback uses the real names (2026-07-22 S-rung run:
    nothing resolved at serve, all layers loaded unquantized, crash).

    ``assume_resolvable`` is the caller's assertion that this target HAS a
    resolved source even though no single skeleton key carries its name — the
    packed-expert stacks, whose source is a set of per-expert checkpoint
    tensors (DSv4: ``layers.N.ffn.experts.{i}.w{1,2,3}``) and whose packed
    parent ``layers.N.ffn.experts.gate_up_proj`` therefore never appears on
    disk. Without it the existence check silently demotes exactly those
    targets to the LIVE spelling and the manifest ships two namespaces
    (``layers.N.attn.*`` beside ``model.layers.N.mlp.experts.*``), which
    gridbook resolves as no-scheme rather than rejecting."""
    if profile is None:
        return qname
    mapped = _source_module_name(qname, profile)
    if skeleton is not None and mapped != qname and not assume_resolvable:
        if (mapped + ".weight") not in skeleton and mapped not in skeleton:
            return qname
    return mapped


def _canonical_qname(ckpt_qname, profile):
    """Skeleton (checkpoint) module qname -> canonical recipe qname, or None if
    the profile drops the key (visual/audio/`.weight_scale_inv`)."""
    if profile is None:
        return ckpt_qname
    live = profile.checkpoint_to_live_name(ckpt_qname + ".weight",
                                           multimodal=False)
    return live[:-len(".weight")] if live else None


def _vecs_and_wq(w: torch.Tensor, cw: torch.Tensor | None, grid: str):
    """One-shot scaled 8-dim vectors + per-vector weights for one Linear (the
    same scaling the encoder feeds the VQ search) — mirrors the exp1b driver's
    shared-codebook pooling."""
    w2d = w.reshape(-1, w.shape[-1]).to(torch.float32)
    vectors, _, _ = cb._scale_and_vectorize(w2d, grid)
    wq = None
    if cw is not None:
        # Broadcast against the ORIGINAL shape first so stacked-expert
        # per-expert weights ((E, 1, in) — the gguf _qw_blocks precedent)
        # slice correctly before the row flatten.
        cw2d = torch.broadcast_to(
            cw.to(w2d.device, torch.float32), tuple(w.shape)
        ).reshape(w2d.shape).contiguous()
        wq = cb._col_weight_vectors(cw2d)
    return vectors, wq


def _train_shared_codebook(weights, cws, *, grid, mode, k, seed, iters,
                           train_cap):
    """One learned codebook over a role's pooled scaled vectors (the exp1b
    shared-per-role logic): product -> n_sub grid-snapped sub-tables; full ->
    one (2^k, 8) table. (The signed positive-magnitude mode was deleted
    2026-08-17 -- no native Gridbook kernel serves the n_sub=1 layout.)"""
    vlist, wlist = [], []
    for w, cw in zip(weights, cws):
        v, wq = _vecs_and_wq(w, cw, grid)
        vlist.append(v)
        wlist.append(wq if wq is not None else torch.ones_like(v))
    vec = torch.cat(vlist, 0)
    wq = torch.cat(wlist, 0)
    if vec.shape[0] > train_cap:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(vec.shape[0], generator=g)[:train_cap].to(
            vec.device)
        vec, wq = vec[idx], wq[idx]
    if mode == "product":
        n_sub = family_for(grid, mode).n_sub
        sub_dim = cb.VEC_DIM // n_sub
        bits = subtable_bit_widths(k, mode, n_sub)
        subs = []
        for i, b in enumerate(bits):
            xs = vec[:, i * sub_dim:(i + 1) * sub_dim]
            ws = wq[:, i * sub_dim:(i + 1) * sub_dim]
            init_i = cb.fixed_lattice(b, grid, sub_dim).to(vec.device)
            subs.append(cb.learn_codebook(xs, b, grid=grid, col_weights=ws,
                                          init=init_i, iters=iters,
                                          seed=seed).cpu())
        return tuple(subs)
    return cb.learn_codebook(vec, k, grid=grid, col_weights=wq, iters=iters,
                             seed=seed).cpu()


def _write_cb_containers(
    out_tensors: dict[str, torch.Tensor],
    out_dir: Path,
    shard_bytes: int,
) -> tuple[list[str], dict[str, str]]:
    """Publish the CB weight containers in the HF-standard shard layout.

    The resident exporter already holds the whole tensor dict, so this is the
    same partition rule the streaming writer applies to its entry sequence --
    emit order, one budget, an oversized tensor gets its own container -- with
    the dict's insertion order as the emit order. Returns the published
    container names in index order and the per-tensor content digests that
    ``shard_layout.tensor_payload_identity`` reduces to the layout-invariant
    identity (the streaming writer takes the same digests in its write pass).
    """
    sizes = [
        (name, int(tensor.numel() * tensor.element_size()))
        for name, tensor in out_tensors.items()
    ]
    groups = plan_shards(sizes, shard_bytes)
    names = container_names(len(groups))
    tensor_sha256: dict[str, str] = {}
    for name, group in zip(names, groups):
        payload = {key: out_tensors[key].contiguous() for key in group}
        for key, tensor in payload.items():
            tensor_sha256[key] = hashlib.sha256(
                tensor.detach().cpu().contiguous()
                .flatten().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        save_file(
            payload,
            str(out_dir / name),
            metadata={"format": "pt", "quant_method": "gridbook"},
        )
    if len(names) > 1:
        write_shard_index(
            out_dir,
            {key: name for name, group in zip(names, groups) for key in group},
            sum(nbytes for _name, nbytes in sizes),
        )
        print(f"[export-cb] published {len(names)} safetensors shard(s) + "
              "model.safetensors.index.json", flush=True)
    return names, tensor_sha256


@refuse_prismasnap_lane_before_output(lane="Gridbook/codebook")
@_preflight_assignment_before_output_transaction
@transactional_directory_output(
    source_parameter="model_dir",
    output_parameter="out_dir",
    where="export_nvfp4_cb",
)
def export_nvfp4_cb(
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: dict[str, torch.Tensor],
    *,
    shared_codebook_spec: dict | None = None,
    device: str | None = None,
    scale_sweep: bool = True,
    scale_coding: str = cb.SCALE_CODING_TWO_TIER,
    allow_unstamped_research: bool = False,
    allow_research_cost_selection: bool = False,
    allow_per_role_books: bool = False,
    allow_unbacked_route: str | None = None,
    non_native_target: str | None = None,
    activation_cache_dir: str | Path | None = None,
    activation_scale_policy: str | None = None,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    producer_policy: str | None = None,
    producer_runtime_contract: dict | str | Path | None = None,
    trellis_wire_cache: dict | object | None = None,
) -> dict[str, int]:
    """Export a CB checkpoint. See module docstring / LAYOUT.md for the layout.

    ``col_weights`` maps each CB-target qname to its per-input-column importance
    (imatrix / Fisher). ``shared_codebook_spec`` (or None) selects the codebook
    source:

      * ``None`` / ``{"source": "lattice"}`` — the deterministic fixed lattice,
        shipped as one shared FP16 sidecar table set per format;
      * ``{"source": "learned", "train": True, "iters", "seed", "train_cap"}`` —
        a shared per-(role) learned codebook trained here on pooled vectors;
      * ``{"source": "learned", "codebooks": {role: cb_obj}}`` — use provided
        per-role codebooks (a missing role for a target hard-fails).

    ``allow_per_role_books`` overrides the split-book ship gate (campaign rule
    R1): a fused routed weight whose scheme would name more than one codebook
    refuses unless this is passed, and passing it stamps the fact onto the
    shipcard. This exporter reads role-keyed bundles only; a bundle whose
    routed books were pooled per ``(layer, stack, rung)`` belongs to the
    streaming exporter and fails closed here on its missing per-role cell.

    ``scale_coding``: ``"two_tier"`` (production layout v2; fp4 targets write
    4k+9 bytes per superblock) or explicit legacy ``"v1"`` (4k+16). Readers
    remain backward compatible with v1; new artifacts default to v2.

    ``shard_bytes`` is the per-container byte budget, 1 GiB by default, the
    same flag and partition rule as the compressed-tensors lane. One resulting
    shard keeps ``model.safetensors``; more than one publishes
    ``model-XXXXX-of-YYYYY.safetensors`` plus ``model.safetensors.index.json``.
    Pass a budget at least as large as the artifact for the legacy layout.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if scale_coding not in (cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    out_dir = prepare_fresh_export_directory(
        model_dir,
        out_dir,
        where="export_nvfp4_cb",
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = shared_codebook_spec or {}
    source = str(spec.get("source", "lattice")).lower()
    if source not in ("lattice", "learned"):
        raise ValueError(f"shared_codebook_spec source must be lattice/learned,"
                         f" got {source!r}")
    _env_cb_context = cb_serialization_context_from_env()
    _scoped_bundle_export = (
        effective_codebook_source_scope(_env_cb_context) != "none"
    )
    if _scoped_bundle_export:
        if source != _env_cb_context.codebook_source:
            raise ValueError(
                "export_nvfp4_cb: shared_codebook_spec source differs from "
                "CB_CODEBOOK_SOURCE_SCOPE/CB_CODEBOOK_BUNDLE"
            )
        source = _env_cb_context.codebook_source
    elif source == "learned" and bool(spec.get("train", False)) and not (
        allow_unstamped_research
    ):
        raise ValueError(
            "export_nvfp4_cb: production export-time learned-codebook "
            "retraining is forbidden; build CB_CODEBOOK_BUNDLE before cost. "
            "The legacy trainer is available only with "
            "allow_unstamped_research=True."
        )

    assignment = load_assignment(layer_config_path)
    from prismaquant.rtx4090_qwen38_policy import (
        prepare_rtx4090_export_policy,
    )

    _strict_producer = prepare_rtx4090_export_policy(
        model_dir=model_dir,
        assignment=assignment,
        producer_policy=producer_policy,
        runtime_contract=producer_runtime_contract,
        where="export_nvfp4_cb",
    )
    _recipe_payload = json.loads(Path(layer_config_path).read_text())
    _recipe_cb_context_stamp, _recipe_cb_tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(_recipe_payload)
    )
    _recipe_meta = _recipe_payload.get("__prismaquant__", {})
    from prismaquant.research_cost_acceptance import (
        enforce_research_export_acknowledgement,
    )
    _research_cost_selection = enforce_research_export_acknowledgement(
        _recipe_payload,
        acknowledged=allow_research_cost_selection,
        where="export_nvfp4_cb",
    )
    _recipe_cb_render_identity = _recipe_payload.get("cb_render_identity")
    if _recipe_cb_render_identity is None and isinstance(_recipe_meta, dict):
        _recipe_cb_render_identity = _recipe_meta.get("cb_render_identity")
    production_recipe_stamped = (
        _recipe_cb_context_stamp is not None or bool(_recipe_cb_tensor_stamps)
    )
    _claimed_activation_contract = (
        _recipe_cb_context_stamp.get("activation_contract")
        if isinstance(_recipe_cb_context_stamp, dict)
        else None
    )
    if _claimed_activation_contract not in (
        None,
        NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    ):
        raise ValueError(
            "export_nvfp4_cb: unsupported activation contract "
            f"{_claimed_activation_contract!r}"
        )
    _whole_artifact_budget = whole_artifact_budget_from_assignment_payload(
        _recipe_payload,
        where="export_nvfp4_cb layer config",
        assignment=assignment,
    )
    skeleton = _load_skeleton(model_dir)

    # Reuse the compressed-tensors codecs + scheme templates for stock rungs —
    # NEVER reimplement packing. M19 scale-fidelity: `_quantize_2d` renders and
    # packs from ONE scale selection, so the shipped scales ARE the render's.
    from prismaquant import format_registry as _fr
    from prismaquant.export_native_compressed import (
        _quantize_2d as _ct_quantize_2d,
        compute_nvfp4_global_real as _ct_nvfp4_global_real,
        NVFP4_SCHEME as _NVFP4_SCHEME,
        FP8_E4M3_SCHEME as _FP8_E4M3_SCHEME,
    )
    from prismaquant.model_profiles import detect_profile as _detect_profile
    # Stock rungs the mixed container carries CT-style (plugin delegates them to
    # vLLM's CompressedTensors path). FP8_DYNAMIC canonicalizes to FP8_E4M3.
    # FP8_SOURCE is a PASSTHROUGH scheme (verbatim fp8 weight + scale_inv copy),
    # not a _quantize_2d target — handled separately below.
    _STOCK_CT_SCHEMES = {"NVFP4": _NVFP4_SCHEME, "FP8_E4M3": _FP8_E4M3_SCHEME}
    # Profile drives nested-prefix skeleton name resolution (hybrid VLMs); None
    # for a flat checkpoint (recipe names == checkpoint names, resolver no-ops).
    try:
        _profile = _detect_profile(str(model_dir))
    except Exception:
        _profile = None
    from prismaquant.cb_source_decode import build_cb_source_fp8_scale_map

    # One profile-aware source contract for every CB/stock render.  This map
    # also carries the checkpoint-declared block shape and declared MXFP4 set;
    # missing scales fail closed in cb_source_weight_bf16_value.
    _source_fp8_scale_map = build_cb_source_fp8_scale_map(model_dir)
    # --- Coverage gate: classify every assigned format into CB / stock-CT /
    # trellis / BF16-passthrough (the mixed container, LAYOUT.md §4; "FP8 in
    # every recipe"). ---
    cb_targets: dict[str, tuple[str, str, int]] = {}   # qname -> (grid,mode,k)
    stock_targets: dict[str, str] = {}                 # qname -> "NVFP4"|"FP8_E4M3"
    trellis_targets: dict[str, tuple[str, int]] = {}   # qname -> (family, body_rate_q256)
    source_targets: list[str] = []                     # FP8_SOURCE passthrough
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        trellis_parsed = _parse_trellis_format(fmt)
        if trellis_parsed is not None:
            family_obj, rate = trellis_parsed
            trellis_targets[qname] = (family_obj.family, int(rate))
            continue
        parsed = _parse_producer_cb_format(fmt)
        if parsed is not None:
            cb_targets[qname] = parsed
            continue
        canon = _fr.canonical_format_name(fmt)
        if canon == "FP8_SOURCE":
            source_targets.append(qname)
            continue
        if canon in _STOCK_CT_SCHEMES:
            stock_targets[qname] = canon
            continue
        illegal.append((qname, fmt))
    if illegal:
        raise ValueError(
            f"assignment contains formats the mixed CB container cannot carry: "
            f"{sorted({f for _, f in illegal})} — it carries the CB families "
            f"+ stock NVFP4/FP8_DYNAMIC (CT-delegated) + FP8_SOURCE "
            f"(verbatim fp8 passthrough) + trellis TCQ_E2M1/TCQ_E4M3 + BF16 passthrough only")

    # --- Trellis hard runtime facts (WO-C rules 3 + 4) --------------------
    # Rule 3: Fused modules cannot be trellis — per-role wires cannot be
    # concatenated (each carries its own alphabets, schedule, padding).
    # gridbook/config.py refuses such a target *by name*; mirror that here.
    for qname in list(trellis_targets):
        fused_key = _trellis_fused_group_key(qname, _profile)
        if fused_key is not None:
            raise ValueError(
                f"{qname}: fused modules cannot be trellis — vLLM merges "
                f"{fused_key} and per-role wires cannot be concatenated "
                "(each carries its own alphabets, rate schedule and row "
                "padding). gridbook/config.py refuses such a target by name. "
                "On a Qwen-shaped architecture the trellis-eligible Linears are "
                "the unfused ones: o_proj and down_proj. Fused siblings must "
                "take a non-trellis format (NVFP4/FP8/BF16) or go in ignore. "
                "This is a hard runtime fact, not a policy you may relax."
            )
        if _is_routed_moe_trellis_target(qname, _profile):
            raise ValueError(
                f"{qname}: routed/packed-MoE trellis is not served — the pinned "
                "Gridbook 0.9.1 contract (gridbook.runtime-contract.v12) publishes "
                "no routed_moe trellis cell for any family/rate. A "
                "routed/packed-MoE unit assigned a trellis rung must fail export "
                "closed with this message naming the missing cell."
            )

    # Per-expert-on-disk MoE checkpoints: assemble only packed parents that
    # are actually quantized. Packing every detected bank mutates omitted/BF16
    # LFM layers from their loadable ``experts.E.w1/w2/w3.weight`` layout into
    # aggregate ``gate_up_proj/down_proj.weight`` passthrough tensors that
    # vLLM's architecture loader cannot consume.
    _expert_stack_members: dict[
        str, dict[tuple[str, int], str]
    ] = {}
    _pack_skeleton_experts(
        skeleton,
        _profile,
        fp8_scale_inv_map=_source_fp8_scale_map,
        target_qnames=set(cb_targets) | set(stock_targets),
        member_qnames_out=_expert_stack_members,
    )
    # FP8_SOURCE is deliberately excluded above: it is a byte-verbatim
    # passthrough contract and must already resolve to a source weight plus its
    # scale_inv sibling. Synthesizing a packed stack would change that payload
    # and still would not produce the required packed scale tensor.

    # Sidecar stock targets (visual/audio — modules the profile's LM mapping
    # drops): ship WEIGHT-ONLY (W4A16). Text-only calibration has no visual
    # activations to derive a static input scale from, and vLLM's vision
    # tower builds the weight-only CT variant (no input_global_scale param —
    # the W4A4 tensor set failed to load, 2026-07-22).
    sidecar_stock = {q for q in stock_targets
                     if _canonical_qname(q, _profile) is None}

    # Quantized token embeddings (`quantized_embedding` declaration).  A third
    # class of stock target, stricter than the sidecar: the bytes are packed by
    # the same CT codec, but the unit is claimed by GridBook's embedding method
    # rather than by a config group.  Both halves of that matter.
    #
    #   * PACKED, not verbatim.  vLLM's compressed-tensors embedding path
    #     accepts weight-only INT schemes and RAISES for FP8/NVFP4, so a stock
    #     config group naming the embedding does not mis-route the artifact --
    #     it refuses to load it.  Hence the declaration, and hence the unit is
    #     kept out of `config_groups` below.
    #   * WEIGHT-ONLY.  A lookup has no input activation to scale, and the
    #     serving method registers no `input_global_scale` parameter, so an
    #     emitted one is an unmatched checkpoint key at load.  Same suppression
    #     the visual sidecar already needs, for a different reason.
    #
    # Detected by two INDEPENDENT conditions that must agree: the name contract
    # the probe and the declaration builder both use, and the checkpoint's own
    # vocab_size.  A name that matches without the shape (or a stock target
    # shaped like a vocab table under another name) raises rather than picking
    # a branch quietly -- getting this wrong ships an artifact whose embedding
    # is dispatched as a Linear, or a Linear dispatched as a lookup.
    def _declared_vocab_size() -> int:
        # Multimodal checkpoints keep the LM's vocab under `text_config`; a
        # wrapper config with no top-level vocab_size would otherwise read as
        # zero and disable the shape half of the cross-check silently.
        cfg = json.loads((model_dir / "config.json").read_text())
        for holder in (cfg, cfg.get("text_config") or {},
                       cfg.get("language_config") or {}):
            if isinstance(holder, dict) and holder.get("vocab_size"):
                return int(holder["vocab_size"])
        return 0

    _vocab_rows = _declared_vocab_size()

    def _is_embedding_name(q: str) -> bool:
        return q == "model.embed_tokens" or q.endswith(".embed_tokens")

    embedding_stock: dict[str, str] = {}
    for _q, _f in stock_targets.items():
        _w = skeleton.get(_try_resolve_skeleton(_q, skeleton, _profile) or "")
        _rows = int(_w.shape[0]) if _w is not None and _w.dim() == 2 else -1
        _named = _is_embedding_name(_q)
        _shaped = _vocab_rows > 0 and _rows == _vocab_rows and not (
            _q == "lm_head" or _q.endswith(".lm_head"))
        if _named != _shaped:
            raise ValueError(
                f"{_q}: cannot classify as a token embedding — the name says "
                f"{_named} but the checkpoint shape says {_shaped} "
                f"(rows={_rows}, vocab_size={_vocab_rows}). An embedding is "
                "served by GridBook's lookup method and a Linear by a config "
                "group; the two dispatches are not interchangeable, so this "
                "refuses rather than guessing.")
        if _named:
            embedding_stock[_q] = _f
    if embedding_stock:
        # Read the record back through the consumer's rules before any bytes
        # are written: an unroutable format or an lm_head slipped into the
        # recipe must fail the export, not the load.
        build_quantized_embedding_declaration(embedding_stock)
        # An embedding is not a sidecar tower: it keeps its `model.` prefix in
        # vLLM's module tree and is claimed by the declaration, not by a
        # weight-only config group. Both memberships would otherwise fire on a
        # profile whose LM mapping does not name the embedding.
        sidecar_stock -= set(embedding_stock)

    # FP8_SOURCE is PASSTHROUGH-ONLY (PASSTHROUGH_SOURCE_REQUIREMENTS): legal
    # only where the source `.weight` is already fp8_e4m3fn with a
    # `.weight_scale_inv` sibling. The allocator's passthrough-integrity
    # filter should drop it otherwise — hard-fail here so a stale manifest
    # never ships a re-synthesized (8-bpp-wasting) FP8 tensor.
    for qname in source_targets:
        wname = _try_resolve_skeleton(qname, skeleton, _profile)
        scale_entry = _source_fp8_scale_map.get(qname + ".weight")
        sname = scale_entry[1] if scale_entry is not None else None
        w = skeleton.get(wname) if wname else None
        if (
            w is None
            or w.dtype != torch.float8_e4m3fn
            or sname is None
            or sname not in skeleton
        ):
            raise ValueError(
                f"{qname}: assigned FP8_SOURCE but source is not native FP8 "
                f"(weight dtype={None if w is None else w.dtype}, "
                f"has resolved scale={sname is not None and sname in skeleton}). "
                "FP8_SOURCE is "
                f"passthrough-only — never synthesize it.")

    for qname, (grid, mode, k) in cb_targets.items():
        wname = _try_resolve_skeleton(qname, skeleton, _profile)
        if wname is None:
            raise ValueError(
                f"{qname}: assigned {grid}/{mode} k{k} but no weight tensor for "
                f"it in the skeleton (tried {qname}.weight + the "
                f"profile-mapped checkpoint name)")
        in_f = int(skeleton[wname].shape[-1])
        if in_f % cb.SUPERBLOCK != 0:
            raise ValueError(
                f"{qname}: in_features={in_f} is not a multiple of "
                f"{cb.SUPERBLOCK}; fall back to a coarser legal rung or BF16 "
                f"(no block-32 CB rung in Phase 0)")
        if qname not in col_weights:
            raise ValueError(
                f"{qname}: CB target has no col_weights entry — exporting "
                f"unweighted bytes would silently diverge from the "
                f"imatrix-weighted cost measurement (no silent RTN)")
        cwn = col_weights[qname].numel()
        n_exp = (int(skeleton[wname].shape[0])
                 if skeleton[wname].dim() == 3 else 1)
        if cwn not in (in_f, n_exp * in_f):
            raise ValueError(
                f"{qname}: col_weights has {cwn} elements but the weight "
                f"wants {in_f} (shared) or {n_exp}x{in_f} (per-expert, "
                f"(E,1,in)) — the imatrix does not describe this checkpoint")

    # --- Stock NVFP4 fused-sibling coherence: q/k/v (and gate/up) that all land
    # on NVFP4 MUST share one weight_global_scale, or vLLM's fused loader sees
    # inconsistent per-tensor global scales. Take the max over each fused group
    # and override every sibling's pack (mirrors export_native_compressed). ---
    for qname in stock_targets:
        if _try_resolve_skeleton(qname, skeleton, _profile) is None:
            raise ValueError(
                f"{qname}: assigned {stock_targets[qname]} but no weight tensor "
                f"for it in the skeleton (tried {qname}.weight + the "
                f"profile-mapped checkpoint name)")
    _nvfp4_shared_global: dict[str, torch.Tensor] = {}
    _nvfp4_groups: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        if _f != "NVFP4":
            continue
        _gk = (_profile.fused_sibling_group(_q)
               if _profile is not None else None) or _q
        _nvfp4_groups.setdefault(_gk, []).append(_q)
    for _members in _nvfp4_groups.values():
        _grs = [_ct_nvfp4_global_real(
                    _decoded_cb_source_weight(
                        skeleton,
                        _resolve_skeleton(_m, skeleton, _profile),
                        model_weight_name=_m + ".weight",
                        fp8_scale_inv_map=_source_fp8_scale_map,
                    ).to(device),
                    16)
                for _m in _members]
        _shared = torch.stack([g.reshape(()) for g in _grs]).max()
        for _m in _members:
            _nvfp4_shared_global[_m] = _shared

    def _resident_export_target(qname: str) -> str:
        resolved = _resolve_skeleton(qname, skeleton, _profile)
        return (
            resolved[:-len(".weight")]
            if resolved.endswith(".weight")
            else resolved
        )

    # Versioned fused-W4A4 activation contract.  The CB serialization stamp is
    # the production/research boundary: v3 claims the contract and therefore
    # requires complete calibrated coverage; v2/unstamped research emits
    # neither JSON contract nor scalar tensors and is fused-ineligible.
    fp4_activation_targets = {
        qname
        for qname, (grid, _mode, _k) in cb_targets.items()
        if grid == "fp4"
    } | {
        qname
        for qname, fmt in stock_targets.items()
        if fmt == "NVFP4" and qname not in sidecar_stock
        and qname not in embedding_stock
    }
    activation_execution_contract = None
    activation_scales_by_physical_target: dict[str, float] = {}
    activation_scale_policy_id = None

    # -------------------------------------------------------------------
    # Trellis wire retrieval + E2M1 A-side static scale (WO-C C1/C2)
    # -------------------------------------------------------------------
    # The wire is the only carrier (rule 1). The exporter must NOT re-encode.
    # Bytes come from ProductionWeightCache (WO-B). Code against the seam:
    # trellis_wire_cache may be a dict, a ProductionWeightCache instance, or
    # None — if no cache has a wire for a selected unit, fail closed.
    trellis_wires: dict[str, bytes] = {}
    trellis_parsed: dict[str, _TrellisWire] = {}
    trellis_e2m1_targets: set[str] = set()
    trellis_e4m3_targets: set[str] = set()
    trellis_input_global_scales: dict[str, float] = {}
    trellis_input_global_scale_sources: dict[str, str] = {}
    for qname, (family, rate) in trellis_targets.items():
        # Retrieve wire bytes from cache seam
        blob = _load_trellis_wire_bytes(qname, assignment[qname], trellis_wire_cache)
        if blob is None:
            raise ValueError(
                f"{qname}: trellis wire not found in ProductionWeightCache "
                f"for format {assignment[qname]} — export refuses to re-encode "
                "(principle 8: one rendering, surrogate and exported bytes "
                "identical). WO-B lands the retention; if that branch is not "
                "merged into yours yet this is the seam. Provide "
                "trellis_wire_cache={{qname: wire_bytes}} or a "
                "ProductionWeightCache with get_trellis_wire_bytes, and say so "
                "in the commit message."
            )
        # Parse and validate — the wire is self-describing and the only
        # carrier of schedule/alphabets/scale plane/row padding.
        try:
            wire = _TrellisWire.from_bytes(blob)
        except Exception as exc:
            raise ValueError(
                f"{qname}: trellis wire blob is not a canonical "
                f"{_TrellisWire.__name__} payload: {exc}"
            ) from exc
        # Family/rate must match assignment; wire_bytes length is already
        # implied by the blob but we gate on the parsed fields too.
        if wire.family != family or wire.body_rate_q256 != rate:
            raise ValueError(
                f"{qname}: wire family/rate {wire.family} R{wire.body_rate_q256} "
                f"does not match assignment {family} R{rate}"
            )
        # Shape check against skeleton — dense Linear only (rule 4)
        wkey = _try_resolve_skeleton(qname, skeleton, _profile)
        if wkey is not None:
            shp = tuple(int(d) for d in skeleton[wkey].shape)
            if len(shp) == 2:
                if (wire.rows, wire.columns) != (shp[0], shp[1]):
                    raise ValueError(
                        f"{qname}: wire geometry {wire.rows}x{wire.columns} "
                        f"does not match skeleton shape {shp}"
                    )
            elif len(shp) == 3:
                raise ValueError(
                    f"{qname}: trellis wire is dense-only but skeleton is "
                    f"rank-3 packed MoE {shp}"
                )
        trellis_wires[qname] = bytes(blob)
        trellis_parsed[qname] = wire
        if family == _TRELLIS_E2M1_FAMILY:
            trellis_e2m1_targets.add(qname)
        elif family == _TRELLIS_E4M3_FAMILY:
            trellis_e4m3_targets.add(qname)

    # E2M1 A-side static scale: the one genuinely new quantity (WO-C C2).
    # Derive it from calibration activations exactly as NVFP4 derives
    # input_global_scale (compute_nvfp4_input_global_scale /
    # nvfp4_activation_contract.input_global_scale_from_max_abs). Reuse those
    # functions because the execution contract is identical:
    # e2m1_group16_ue4m3_static.
    if trellis_e2m1_targets:
        # Fused-sibling unification does not apply — trellis units are unfused
        # by rule 3. State explicitly rather than leaving silently unhandled.
        # (No call to _unify_input_global_scales_across_fused_siblings.)
        if activation_cache_dir is None:
            raise ValueError(
                "export_nvfp4_cb: E2M1 trellis requires activation_cache_dir; "
                "missing activations for a trellis unit is fail-closed — there "
                "is no defensible default for an activation scale"
            )
        # Resolve policy once — same policy as NVFP4 W4A4
        trellis_activation_policy = resolve_input_global_scale_policy(
            activation_scale_policy
        )
        # For trellis, we reuse the NVFP4 calibration helper to load max_abs
        # per target, but without fused grouping (unfused singletons). Missing
        # activations fail closed.
        from prismaquant.nvfp4_activation_contract import (
            input_global_scale_from_max_abs as _trellis_scale_from_max,
            load_activation_cache_max_abs as _trellis_load_max,
        )

        # Attempt to load via the calibrated helper; if activations are present
        # in the cache dir, use them. Fallback to direct max_abs compute.
        # We intentionally do NOT use the packed-MoE supplemental path — trellis
        # is dense only.
        try:
            # Try the full helper which handles fused groups; for trellis it
            # will be singleton groups, but we still getSource validation.
            from prismaquant.nvfp4_activation_contract import (
                calibrated_input_global_scales_with_sources as _calib_with_src,
            )

            trellis_scales_tmp, trellis_sources_tmp = _calib_with_src(
                trellis_e2m1_targets,
                activation_cache_dir=activation_cache_dir,
                policy=trellis_activation_policy,
                profile=_profile,
            )
            trellis_input_global_scales.update(trellis_scales_tmp)
            trellis_input_global_scale_sources.update(trellis_sources_tmp)
        except Exception:
            # Fallback: direct per-target max_abs load
            max_abs_map = {}
            # Try loading via ActivationIndex directly
            try:
                max_abs_map = _trellis_load_max(
                    activation_cache_dir, trellis_e2m1_targets
                )
            except Exception as exc:
                raise ValueError(
                    "export_nvfp4_cb: E2M1 trellis activation calibration "
                    f"failed for {sorted(trellis_e2m1_targets)}: {exc}"
                ) from exc
            missing = [t for t in trellis_e2m1_targets if t not in max_abs_map]
            if missing:
                raise ValueError(
                    f"export_nvfp4_cb: E2M1 trellis activation contract has no "
                    f"calibrated input for {missing!r}; production export "
                    "refuses an incomplete scale mapping"
                )
            for qname in trellis_e2m1_targets:
                max_abs = float(max_abs_map[qname])
                scale = _trellis_scale_from_max(
                    max_abs, policy=trellis_activation_policy
                )
                trellis_input_global_scales[qname] = float(scale)
                trellis_input_global_scale_sources[qname] = "target_activation_cache"

    # --- Resolve codebooks, grouped by (physical ref, format).  Production
    # learned cells come only from the immutable pre-render bundle.  The old
    # pooled trainer remains behind allow_unstamped_research for reproducible
    # historical experiments; it is never reached by a stamped export. ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))

    learned_bundle = None
    if _scoped_bundle_export:
        from prismaquant.cb_learned_bundle import load_bundle_cached

        learned_bundle = load_bundle_cached(_env_cb_context.codebook_bundle_path)

    # (ref, fmt) -> codebook object; learned production refs are one qname per
    # cell (and therefore one layer/role/rung), while lattice stays canonical.
    codebooks: dict[tuple[str, str], object] = {}
    # qname -> (ref, fmt, codebook, source_kind)
    target_cb: dict[str, tuple[str, str, object, str]] = {}
    routed_role_plans: dict[
        str, tuple[RoutedMoECodebookRole, ...]
    ] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    cb_group_target_names: dict[tuple[str, str], list[str]] = {}
    for qname, (grid, mode, k) in cb_targets.items():
        fmt = assignment[qname]
        kind = (
            codebook_source_for_format(fmt, _env_cb_context)
            if _scoped_bundle_export
            else source
        )
        logical_qnames = learned_role_qnames_for_packed(qname)
        if kind == "learned":
            from prismaquant.cb_learned_bundle import (
                refuse_routed_moe_learned,
            )

            refuse_routed_moe_learned(
                qname,
                routed_moe=bool(logical_qnames),
            )
        if _scoped_bundle_export and kind == "learned" and logical_qnames:
            if learned_bundle is None:
                raise AssertionError(
                    f"{qname}/{fmt}: learned routed roles need a bundle"
                )
            if (
                grid != "fp8"
                or mode != "product"
                or k not in ROUTED_MOE_CBL_BANK_RUNGS
            ):
                raise ValueError(
                    f"{qname}/{fmt}: routed learned CBL is limited to "
                    "FP8_CB_K28..K33 covered by the immutable CBL bank"
                )
            member_qnames = _expert_stack_members.get(qname)
            if member_qnames is None:
                raise ValueError(
                    f"{qname}/{fmt}: learned routed per-role books require "
                    "the exact per-expert checkpoint member map and member "
                    "col_weights; an already-packed rank-3 source has no "
                    "member aliases to validate"
                )
            physical_weight = _decoded_cb_source_weight(
                skeleton,
                _resolve_skeleton(qname, skeleton, _profile),
                model_weight_name=qname + ".weight",
                fp8_scale_inv_map=_source_fp8_scale_map,
            )
            if physical_weight.ndim != 3:
                raise ValueError(
                    f"{qname}/{fmt}: routed learned role source must be "
                    f"rank-3, got {tuple(physical_weight.shape)}"
                )
            if len(logical_qnames) == 2:
                fused_rows = int(physical_weight.shape[1])
                if fused_rows % 2:
                    raise ValueError(
                        f"{qname}: fused gate/up row count {fused_rows} is odd"
                    )
                output_rows = (fused_rows // 2, fused_rows // 2)
            else:
                output_rows = (int(physical_weight.shape[1]),)

            unresolved_roles: list[RoutedMoECodebookRole] = []
            for logical_qname, rows in zip(
                logical_qnames, output_rows, strict=True
            ):
                projection = logical_qname.rsplit(".", 1)[-1]
                role_cw, members = stacked_role_col_weights(
                    packed_qname=qname,
                    projection=projection,
                    member_qnames=member_qnames,
                    col_weights=col_weights,
                )
                unresolved_roles.append(RoutedMoECodebookRole(
                    projection=projection,
                    qname=logical_qname,
                    ref=logical_qname,
                    format_name=fmt,
                    codebook=None,
                    col_weights=role_cw,
                    output_rows=rows,
                    member_qnames=members,
                ))

            resolved_roles: list[RoutedMoECodebookRole] = []
            for unresolved, role_weight in split_role_rows(
                physical_weight, tuple(unresolved_roles)
            ):
                role_book = learned_bundle.codebook_for(
                    unresolved.qname,
                    fmt,
                    weight=role_weight,
                    col_weights=unresolved.col_weights,
                )
                role = RoutedMoECodebookRole(
                    projection=unresolved.projection,
                    qname=unresolved.qname,
                    ref=unresolved.ref,
                    format_name=unresolved.format_name,
                    codebook=role_book,
                    col_weights=unresolved.col_weights,
                    output_rows=unresolved.output_rows,
                    member_qnames=unresolved.member_qnames,
                )
                group_key = (role.ref, fmt)
                if group_key in by_group:
                    raise ValueError(
                        f"{qname}: duplicate routed learned group {group_key}"
                    )
                codebooks[group_key] = role.codebook
                by_group[group_key] = [qname]
                physical_target = _resident_export_target(qname)
                cb_group_target_names[group_key] = [
                    logical_role_qname(physical_target, role.projection)
                ]
                resolved_roles.append(role)
            routed_role_plans[qname] = tuple(resolved_roles)
            continue
        ref = (
            qname if _scoped_bundle_export and kind == "learned"
            else (_role_of(qname) if kind == "learned" else "lattice")
        )
        by_group.setdefault((ref, fmt), []).append(qname)

    # --- SPLIT-BOOK SHIP GATE (campaign rule R1), the streaming exporter's
    # twin. Structural and producer-side: count the distinct codebooks one
    # fused routed weight's scheme would name. The runtime consequence lives
    # only in the human-facing message.
    split_book_targets = fused_targets_with_split_books({
        _resident_export_target(routed_qname): {
            role.projection: role.ref for role in routed_roles
        }
        for routed_qname, routed_roles in routed_role_plans.items()
        if len(routed_roles) > 1
    })
    if split_book_targets and not allow_per_role_books:
        raise ValueError(describe_split_book_refusal(split_book_targets))

    for (ref, fmt), qnames in by_group.items():
        if qnames[0] in routed_role_plans:
            continue
        grid, mode, k = cb_targets[qnames[0]]
        kind = (
            codebook_source_for_format(fmt, _env_cb_context)
            if _scoped_bundle_export
            else source
        )
        if kind == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
        elif _scoped_bundle_export:
            if len(qnames) != 1 or learned_bundle is None:
                raise AssertionError(
                    f"{ref}/{fmt}: learned bundle cell must own one qname"
                )
            qname = qnames[0]
            weight = _decoded_cb_source_weight(
                skeleton,
                _resolve_skeleton(qname, skeleton, _profile),
                model_weight_name=qname + ".weight",
                fp8_scale_inv_map=_source_fp8_scale_map,
            )
            codebooks[(ref, fmt)] = learned_bundle.codebook_for(
                qname,
                fmt,
                weight=weight,
                col_weights=col_weights[qname],
            )
        else:
            role = ref
            if train:
                weights = [
                    _decoded_cb_source_weight(
                        skeleton,
                        _resolve_skeleton(q, skeleton, _profile),
                        model_weight_name=q + ".weight",
                        fp8_scale_inv_map=_source_fp8_scale_map,
                    ).to(device)
                    for q in qnames
                ]
                cws = [col_weights[q].to(device) for q in qnames]
                codebooks[(ref, fmt)] = _train_shared_codebook(
                    weights, cws, grid=grid, mode=mode, k=k, seed=seed,
                    iters=iters, train_cap=train_cap)
            elif role in provided:
                codebooks[(ref, fmt)] = provided[role]
            else:
                raise ValueError(
                    f"role {role!r} ({fmt}): codebook_source=learned but no "
                    f"codebook supplied and train=False — missing learned "
                    f"sidecar for {len(qnames)} tensor(s)")
        for q in qnames:
            target_cb[q] = (ref, fmt, codebooks[(ref, fmt)], kind)

    # Bind byte pricing to the exact physical sidecar refs this artifact will
    # write.  This identity is shared by allocation/reporting/export checks;
    # no producer path silently assumes the legacy-v1 scale plane.
    materialized_codebook_tensors = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in _codebook_tensors(ref, fmt, codebook).items()
    }
    selected_codebook_digests = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, tensor in materialized_codebook_tensors.items()
    }
    materialized_codebook_digests = (
        dict(_env_cb_context.codebook_content_digests or {})
        if _scoped_bundle_export
        else {}
    )
    for name, digest in selected_codebook_digests.items():
        previous = materialized_codebook_digests.get(name)
        if previous is not None and previous != digest:
            raise ValueError(
                f"export_nvfp4_cb: selected codebook {name!r} differs from "
                "the immutable bundle digest"
            )
        materialized_codebook_digests[name] = digest
    selected_refs_by_format = {
        qname: {
            fmt: _codebook_tensor_names(ref, fmt, codebook)
        }
        for qname, (ref, fmt, codebook, _kind) in target_cb.items()
    }
    for roles in routed_role_plans.values():
        for role in roles:
            selected_refs_by_format[role.qname] = {
                role.format_name: _codebook_tensor_names(
                    role.ref, role.format_name, role.codebook
                )
            }
    if _scoped_bundle_export:
        refs_by_format = {
            str(qname): dict(by_format)
            for qname, by_format in (
                _env_cb_context.codebook_refs_by_qname_format or {}
            ).items()
        }
        for qname, by_format in selected_refs_by_format.items():
            target_formats = refs_by_format.setdefault(qname, {})
            for fmt, refs in by_format.items():
                previous = target_formats.get(fmt)
                if previous is not None and tuple(
                    (previous,) if isinstance(previous, str) else previous
                ) != tuple(refs):
                    raise ValueError(
                        f"{qname}/{fmt}: exporter refs differ from immutable "
                        "bundle refs"
                    )
                target_formats[fmt] = refs
    else:
        refs_by_format = None
    serialization_context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        codebook_source_scope=(
            _env_cb_context.codebook_source_scope
            if _scoped_bundle_export else None
        ),
        scale_sweep=(
            _env_cb_context.scale_sweep
            if _scoped_bundle_export else bool(scale_sweep)
        ),
        scale_sweep_scope=(
            _env_cb_context.scale_sweep_scope
            if _scoped_bundle_export else None
        ),
        ldlq=_env_cb_context.ldlq,
        ldlq_scope=getattr(_env_cb_context, "ldlq_scope", "all" if _env_cb_context.ldlq else "none"),
        minchain=_env_cb_context.minchain,
        minchain_version=_env_cb_context.minchain_version,
        encode_tier=_env_cb_context.encode_tier,
        activation_contract=_claimed_activation_contract,
        activation_execution=(
            NVFP4_ACTIVATION_EXECUTION
            if _claimed_activation_contract is not None
            else None
        ),
        codebook_refs=None if _scoped_bundle_export else {
            qname: _codebook_tensor_names(ref, fmt, codebook)
            for qname, (ref, fmt, codebook, _kind) in target_cb.items()
        },
        codebook_refs_by_qname_format=refs_by_format,
        codebook_content_digests=materialized_codebook_digests,
        codebook_bundle_path=(
            _env_cb_context.codebook_bundle_path
            if _scoped_bundle_export else None
        ),
    )
    # --- ROUTE-STATUS GATE (campaign rule R3, principle 9). ----------------
    # The in-memory twin of the streaming gate, at the matching pre-write
    # point. Both call one helper so the verdict cannot drift between the two
    # export paths; only the spelling of the facts differs (this exporter
    # holds real tensors, so shapes come from the skeleton rather than from a
    # safetensors header).
    from .cb_route_status_gate import gate_cb_export_units

    def _route_gate_shape(qname: str) -> tuple[int, ...]:
        wname = _try_resolve_skeleton(qname, skeleton, _profile)
        if wname is None:
            raise ValueError(
                f"{qname}: route-status gate cannot resolve a weight tensor")
        return tuple(int(v) for v in skeleton[wname].shape)

    if _strict_producer is not None:
        # The unreleased Ada candidate is qualified by the supplied Gridbook
        # v11 contract, not by the repository's historical serving pin used by
        # the generic gate.  The strict stamp is closed and cannot represent a
        # generic override, non-native disposition, or fallback route.
        from prismaquant.rtx4090_qwen38_policy import (
            rtx4090_route_status_stamp,
        )

        cb_route_status_provenance = rtx4090_route_status_stamp(
            _strict_producer[1], assignment
        )
        # Trellis on a strict policy is not defined — the policy pins an
        # Ada-era v11 contract that has no trellis cells. Fail closed if a
        # trellis rung sneaked into a strict assignment.
        if trellis_targets:
            raise ValueError(
                "export_nvfp4_cb: strict producer policy does not support "
                f"trellis units {sorted(trellis_targets)}; use the generic "
                "Gridbook 0.9.1 route gate"
            )
    else:
        # Include trellis in the same gate — one table, one verdict.
        # Trellis routed MoE is dense-only, so routed_units for trellis is
        # empty (the earlier hard refusal would have already fired). The gate
        # still needs shapes for trellis: use the wire geometry when a wire
        # is present, else the skeleton.
        def _trellis_route_shape(qname: str) -> tuple[int, ...]:
            if qname in trellis_parsed:
                w = trellis_parsed[qname]
                return (int(w.rows), int(w.columns))
            return _route_gate_shape(qname)

        def _combined_shape(qname: str) -> tuple[int, ...]:
            if qname in trellis_targets:
                return _trellis_route_shape(qname)
            return _route_gate_shape(qname)

        all_quantized = (*cb_targets, *stock_targets, *trellis_targets)
        # For trellis, routed detection is via the earlier dense-only gate;
        # no trellis MoE should survive, but keep the routed set empty for
        # the mixed call so the gate does not mis-classify a trellis Linear
        # as MoE due to “.experts.” in a non-MoE name.
        cb_route_status_provenance = gate_cb_export_units(
            assignment=assignment,
            quantized_targets=all_quantized,
            routed_units=_expert_stack_members,
            role_split_units=(
                qname for qname, roles in routed_role_plans.items() if roles
            ),
            shape_of=_combined_shape,
            allow_unbacked_route=allow_unbacked_route,
            non_native_target=non_native_target,
            exporter="export_nvfp4_cb",
        )
        # WO-C C3: Verify every trellis unit's cell explicitly — a rung may be
        # a producer candidate (formats table) and still have no attested serving
        # cell (lane_eligibility). The gate already fails on unattested, but
        # we want a pointed message for the trellis rate-table gap.
        if trellis_targets:
            from prismaquant.gridbook_lane_eligibility import (
                load_eligibility_table,
                load_published_formats,
                unit_structural_facts,
                resolve_unit_route,
            )
            _trellis_table = load_eligibility_table()
            _trellis_published = load_published_formats()
            # Derive target_platform from the same profile the gate used
            _target_platform = None
            try:
                from prismaquant.serving_profiles import load_serving_profile
                import os as _os_gate

                _tp_name = (
                    _os_gate.environ.get("PRISMAQUANT_TARGET_PROFILE")
                    or read_layer_config_metadata(layer_config_path).get(
                        "target_profile"
                    )
                    or "nvfp4_cb"
                )
                _target_platform = load_serving_profile(_tp_name).target_platform or None
            except Exception:
                _target_platform = None
            # Default to sm_121 per WO-C C3 when no profile declares it — the
            # trellis lane is sm_121 only, and the gate without a platform says
            # unattested with no route to name, which is correct but vague.
            if not _target_platform:
                _target_platform = "sm_121"
            # Only enforce the pointed trellis rung-vs-cell message when
            # the artifact does NOT carry an explicit non-native or override
            # declaration — otherwise the gate's own disposition (stamped as
            # declared_non_native_target / explicit_override) is the correct
            # outcome, not a refusal.
            _allow_override = bool(
                (allow_unbacked_route or os.environ.get("PQ_CB_ROUTE_STATUS_OVERRIDE"))
                or (non_native_target or os.environ.get("PQ_CB_NON_NATIVE_TARGET"))
            )
            for qname, (family, rate) in trellis_targets.items():
                is_routed = _is_routed_moe_trellis_target(qname, _profile)
                shape = _trellis_route_shape(qname)
                facts = unit_structural_facts(
                    qname,
                    assignment[qname],
                    is_routed_moe=is_routed,
                    role_split=False,
                    in_features=shape[1] if len(shape) >= 2 else shape[-1],
                    out_features=shape[0],
                    published_formats=_trellis_published,
                )
                route = resolve_unit_route(facts, _trellis_table, platform=_target_platform)
                if route.route_status not in ("backed", "backed_with_serve_flag"):
                    if _allow_override:
                        continue
                    # Produce the exact WO-C message naming the missing cell
                    raise ValueError(
                        f"{qname}: trellis {family} R{rate} has no backed serving "
                        f"cell on platform {_target_platform!r} (route_status="
                        f"{route.route_status!r}, reason={route.unattested_reason!r}). "
                        "The formats table lists candidate_rungs_q256 "
                        "[384, 512, 640, 768, 896] but the pinned contract's "
                        "lane_eligibility cells list rungs_q256: [512] only. A "
                        "rung may be a producer candidate and still have no "
                        "attested serving cell — export must gate on the cell, "
                        "not the candidate list. Unless the artifact carries an "
                        "explicit non-native-target declaration or per-run "
                        "override (stamped on shipcard), this refuses."
                    )

    validate_cb_serialization_context_stamp(
        _recipe_cb_context_stamp,
        serialization_context,
        where="export_nvfp4_cb",
    )
    from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

    _ldlq_telemetry_qnames = {
        qname for qname in cb_targets
        if _ldlq_for_format(assignment[qname], serialization_context)
    }
    ldlq_telemetry = None
    if _ldlq_telemetry_qnames:
        from prismaquant.cb_ldlq_gate_telemetry import (
            LDLQGateTelemetryCollector,
        )

        ldlq_telemetry = LDLQGateTelemetryCollector(
            expected_qnames=_ldlq_telemetry_qnames,
            kernel_stamp=cb.canonical_ldlq_kernel_stamp(),
        )
    ldlq_activation_loader = None
    if serialization_context.ldlq:
        if activation_cache_dir is None:
            raise ValueError(
                "export_nvfp4_cb: LDLQ requires activation_cache_dir"
            )
        from prismaquant.cb_ldlq import CBLDLQActivationLoader

        ldlq_activation_loader = CBLDLQActivationLoader(
            activation_cache_dir,
            model_dir=model_dir,
            profile=_profile,
            replay_device=device,
        )
    if production_recipe_stamped and cb_targets:
        validate_cb_assignment_serialization_stamps(
            {qname: assignment[qname] for qname in cb_targets},
            {
                qname: tuple(int(dim) for dim in skeleton[
                    _resolve_skeleton(qname, skeleton, _profile)
                ].shape)
                for qname in cb_targets
            },
            context=serialization_context,
            stamps=_recipe_cb_tensor_stamps,
            where="export_nvfp4_cb",
        )
    if cb_targets and _recipe_cb_render_identity is not None:
        from prismaquant.production_weight_cache import (
            validate_cb_render_identity_metadata,
        )

        validate_cb_render_identity_metadata(
            _recipe_cb_render_identity,
            expected_context=serialization_context,
            expected_formats_by_qname={
                qname: (assignment[qname],) for qname in sorted(cb_targets)
            },
            col_weights=col_weights,
            require_minchain_cells=serialization_context.minchain,
            where="export_nvfp4_cb assignment render identity",
        )
    elif (
        cb_targets
        and production_recipe_stamped
        and _research_cost_selection is None
    ):
        raise ValueError(
            "export_nvfp4_cb: stamped production CB assignment is missing "
            "its value-bearing render identity"
        )
    elif cb_targets and not (
        allow_unstamped_research or _research_cost_selection is not None
    ):
        raise ValueError(
            "export_nvfp4_cb: CB export requires a value-bearing render "
            "identity; pass allow_unstamped_research=True only for an "
            "explicit non-production experiment"
        )
    elif _recipe_cb_render_identity is not None:
        raise ValueError(
            "export_nvfp4_cb: non-CB assignment carries a stale CB render "
            "identity"
        )

    # Fit only after every persisted producer identity is validated.  Besides
    # giving deterministic failure ordering, this avoids loading a potentially
    # multi-GB activation cache for an assignment that cannot be exported.
    if _claimed_activation_contract is not None and fp4_activation_targets:
        if activation_cache_dir is None:
            raise ValueError(
                "export_nvfp4_cb: production FP4 activation contract requires "
                "activation_cache_dir; refusing uncalibrated fused W4A4"
            )
        activation_scale_policy_id = resolve_input_global_scale_policy(
            activation_scale_policy
        )
        from prismaquant.moe_imatrix import (
            synthesize_packed_expert_activation_samples,
        )

        packed_candidates = {
            qname for qname in fp4_activation_targets
            if qname.endswith((".gate_up_proj", ".down_proj"))
        }
        supplemental_samples = (
            synthesize_packed_expert_activation_samples(
                model_dir,
                activation_cache_dir,
                packed_candidates,
                _profile,
                device=device,
            )
            if packed_candidates and _profile is not None
            else {}
        )
        (
            logical_scales,
            activation_calibration_sources,
        ) = calibrated_input_global_scales_with_sources(
            fp4_activation_targets,
            activation_cache_dir=activation_cache_dir,
            policy=activation_scale_policy_id,
            profile=_profile,
            supplemental_activations=supplemental_samples,
            calibration_device=device,
        )
        (
            activation_execution_contract,
            activation_scales_by_physical_target,
        ) = build_execution_contract(
            logical_scales,
            policy=activation_scale_policy_id,
            target_name=_resident_export_target,
            calibration_sources=activation_calibration_sources,
            profile=_profile,
        )

    # --- Pack targets; copy everything else verbatim. ---
    out_tensors: dict[str, torch.Tensor] = {}
    cb_tensor_blobs: dict[str, torch.Tensor] = {}
    cb_serialized_shapes: dict[str, tuple[int, ...]] = {}
    cb_output_tensor_names: set[str] = set()
    verified_cb_source_qnames: set[str] = set()
    actual_cb_tensor_bytes = 0
    emitted_activation_scale_targets: set[str] = set()
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    packed_qnames = set(cb_targets)
    source_qnames = set(source_targets)
    # scale_inv siblings of FP8_SOURCE targets are emitted verbatim in the
    # source branch below; skip them in the passthrough else-branch so they
    # are neither double-emitted nor added to the ignore list.
    _source_scale_keys = {
        _source_fp8_scale_map[q + ".weight"][1]
        for q in source_qnames
        if q + ".weight" in _source_fp8_scale_map
    }
    _source_scale_keys.discard(None)
    _consumed_source_scale_keys = {
        _source_fp8_scale_map[q + ".weight"][1]
        for q in set(cb_targets) | set(stock_targets) | source_qnames
        if q + ".weight" in _source_fp8_scale_map
    }
    _consumed_source_scale_keys.discard(None)

    for name, tensor in skeleton.items():
        # `name` is the CHECKPOINT key; `ckpt_qname` its module base (drives the
        # EXPORTED tensor names — vLLM's convention, incl. the language_model
        # infix); `canon` is the canonical recipe qname that assignment /
        # col_weights / cb_targets are keyed by (nested -> canonical).
        if name.endswith(".weight"):
            ckpt_qname = name[:-len(".weight")]
        elif _try_resolve_direct_packed_expert(name, skeleton, _profile) == name:
            ckpt_qname = name
        else:
            ckpt_qname = None
        canon = _canonical_qname(ckpt_qname, _profile) if ckpt_qname else None
        if canon is None and ckpt_qname is not None and (
                ckpt_qname in stock_targets or ckpt_qname in cb_targets):
            # Sidecar modules the profile's LM mapping drops (visual/audio)
            # but the recipe DOES assign (e.g. VISUAL_FORMAT=NVFP4): their
            # recipe qnames are already checkpoint-form, so classify by the
            # raw name. Without this the config pass promised a stock group
            # for 110 visual Linears while the write pass copied raw BF16 —
            # a split-brain artifact vLLM cannot load (2026-07-22 27B).
            canon = ckpt_qname
        if name in _consumed_source_scale_keys:
            continue
        if canon in source_qnames:
            # FP8_SOURCE passthrough: copy the native fp8 `.weight` verbatim
            # and rename `.weight_scale_inv` -> `.weight_scale` (bytes
            # verbatim, fp32) — EXACTLY as the CT streaming exporter does
            # (export_native_compressed:5711), so stock compressed-tensors
            # block-fp8 delegation reads it unchanged. No dequant/requant
            # round-trip; NOT added to ignore (it is an FP8_SOURCE group).
            out_tensors[ckpt_qname + ".weight"] = tensor.contiguous()
            sname = _source_fp8_scale_map[canon + ".weight"][1]
            out_tensors[ckpt_qname + ".weight_scale"] = skeleton[sname].to(
                torch.float32).contiguous()
            counts["FP8_SOURCE"] += 1
            continue
        if canon in trellis_targets:
            # Gridbook trellis wire — the only carrier (WO-C rule 1). The
            # schedule, alphabets, block offsets and scale plane exist nowhere
            # else; never emit them as separate tensors and never emit a
            # [rows, row_stride] payload rectangle. Every scale is derived from
            # the blob, never loaded beside it — except E2M1's
            # trellis_input_global_scale, which is genuinely not a wire fact.
            family, rate = trellis_targets[canon]
            blob = trellis_wires[canon]
            wire = trellis_parsed[canon]
            # Emit wire_bytes as opaque uint8 1-D
            wire_tensor = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            export_base = _export_base_name(canon, _profile, skeleton, assume_resolvable=False)
            # Resolve export base to checkpoint naming
            # ckpt_qname already is checkpoint base; use it for output but ensure
            # it matches the canonical export mapping
            out_tensors[export_base + ".wire_bytes"] = wire_tensor.contiguous()
            counts[assignment[canon]] += 1
            # Verify wire_bytes length matches scheme declaration (consumer gate)
            # The scheme will declare wire_bytes = len(blob)
            if family == _TRELLIS_E2M1_FAMILY:
                # E2M1 A-side static scale — the one genuinely new quantity (WO-C C2)
                scale_val = trellis_input_global_scales.get(canon)
                if scale_val is None:
                    raise ValueError(
                        f"{canon}: missing trellis_input_global_scale for E2M1 "
                        f"R{rate} — calibration activations absent, fail closed"
                    )
                scale_tensor = torch.tensor([float(scale_val)], dtype=torch.float32)
                out_tensors[export_base + ".trellis_input_global_scale"] = scale_tensor.contiguous()
            continue
        if canon in packed_qnames:
            grid, mode, k = cb_targets[canon]
            routed_roles = routed_role_plans.get(canon)
            if routed_roles is None:
                _ref, fmt, codebook, _kind = target_cb[canon]
                cbook = _to_device(codebook, device)
            else:
                fmt = assignment[canon]
                cbook = None
            w = _decoded_cb_source_weight(
                skeleton,
                name,
                model_weight_name=canon + ".weight",
                fp8_scale_inv_map=_source_fp8_scale_map,
            ).to(device)
            if _recipe_cb_render_identity is not None:
                from prismaquant.production_weight_cache import (
                    validate_cb_render_source_weight,
                )

                validate_cb_render_source_weight(
                    _recipe_cb_render_identity,
                    canon,
                    w,
                    where="export_nvfp4_cb source tensor",
                )
                verified_cb_source_qnames.add(canon)
            from prismaquant.nvfp4_cb_footprint import _ldlq_for_format

            ldlq_for_this = _ldlq_for_format(fmt, serialization_context)
            if routed_roles is not None and ldlq_for_this:
                raise ValueError(
                    f"{canon}/{fmt}: routed learned per-role CBL does not "
                    "yet carry one physical-tensor LDLQ gate telemetry "
                    "record; use the production FP8 learned / NVFP4 LDLQ "
                    "scope instead of enabling LDLQ on FP8"
                )
            ldlq_gate_info: dict[str, object] | None = (
                {} if ldlq_for_this else None
            )
            if routed_roles is None:
                packed, fields = cb.nvfp4_cb_pack(
                    w, k, grid=grid, mode=mode,
                    col_weights=col_weights[canon].to(device),
                    codebook=cbook,
                    scale_sweep=scale_sweep_for_format(
                        fmt, serialization_context
                    ),
                    scale_coding=(scale_coding if grid == "fp4"
                                  else cb.SCALE_CODING_V1),
                    encode_tier=serialization_context.encode_tier,
                    ldlq=ldlq_for_this,
                    activation_rows=(
                        ldlq_activation_loader.load(
                            canon,
                            stack_size=(
                                int(w.shape[0]) if w.dim() == 3 else None
                            ),
                        )
                        if ldlq_for_this
                        and ldlq_activation_loader is not None else None
                    ),
                    ldlq_gate_info_out=ldlq_gate_info,
                )
            else:
                packed_parts: list[torch.Tensor] = []
                scale_parts: list[torch.Tensor] = []
                for role, role_weight in split_role_rows(w, routed_roles):
                    role_packed, role_fields = cb.nvfp4_cb_pack(
                        role_weight,
                        k,
                        grid=grid,
                        mode=mode,
                        col_weights=role.col_weights.to(device),
                        codebook=_to_device(role.codebook, device),
                        scale_sweep=scale_sweep_for_format(
                            fmt, serialization_context
                        ),
                        scale_coding=cb.SCALE_CODING_V1,
                        encode_tier=serialization_context.encode_tier,
                        ldlq=False,
                    )
                    packed_parts.append(role_packed.reshape(
                        w.shape[0], role.output_rows, -1
                    ))
                    scale_parts.append(role_fields["scales"].reshape(
                        w.shape[0], role.output_rows
                    ))
                packed = torch.cat(packed_parts, dim=1).contiguous()
                fields = {"scales": torch.cat(scale_parts, dim=1).contiguous()}
            if ldlq_for_this:
                assert ldlq_telemetry is not None
                assert ldlq_gate_info is not None
                ldlq_telemetry.record(
                    qname=canon,
                    shape=tuple(int(dim) for dim in w.shape),
                    grid=grid,
                    mode=mode,
                    k=k,
                    gate_info=ldlq_gate_info,
                )
            if w.dim() == 3:
                # Stacked packed experts: keep the expert axis explicit —
                # uint8 (E, out, bytes_per_row); fp8 per-channel scales
                # (E, out). LAYOUT.md §3 (stacked experts).
                packed = packed.reshape(w.shape[0], w.shape[1], -1)
            packed_out = packed.to(torch.uint8).cpu().contiguous()
            payload = cb_tensor_payload_breakdown(
                fmt,
                tuple(int(dim) for dim in w.shape),
                qname=canon,
                context=serialization_context,
            )
            packed_bytes = packed_out.numel() * packed_out.element_size()
            if packed_bytes != payload["packed_weight_bytes"]:
                raise AssertionError(
                    f"{canon}: serialized cb_qweight is {packed_bytes}B, "
                    f"accounting expected {payload['packed_weight_bytes']}B"
                )
            packed_name = ckpt_qname + ".cb_qweight"
            out_tensors[packed_name] = packed_out
            cb_output_tensor_names.add(packed_name)
            scale_bytes = 0
            input_scale_bytes = 0
            if grid == "fp8":
                ws = fields["scales"].reshape(
                    *w.shape[:-1]).to(torch.float32).cpu().contiguous()
                scale_bytes = ws.numel() * ws.element_size()
                if scale_bytes != payload["fp8_row_scale_bytes"]:
                    raise AssertionError(
                        f"{canon}: serialized weight_scale is {scale_bytes}B, "
                        "accounting expected "
                        f"{payload['fp8_row_scale_bytes']}B"
                    )
                scale_name = ckpt_qname + ".weight_scale"
                out_tensors[scale_name] = ws
                cb_output_tensor_names.add(scale_name)
            elif payload["fp8_row_scale_bytes"]:
                raise AssertionError(
                    f"{canon}: FP4-CB unexpectedly priced an FP8 row scale"
                )
            if grid == "fp4" and _claimed_activation_contract is not None:
                try:
                    input_scale = activation_scales_by_physical_target[
                        ckpt_qname
                    ]
                except KeyError:
                    raise AssertionError(
                        f"{canon}: claimed FP4 activation contract has no "
                        f"physical scalar for {ckpt_qname!r}"
                    ) from None
                input_scale_tensor = input_global_scale_tensor(input_scale)
                input_scale_bytes = (
                    input_scale_tensor.numel()
                    * input_scale_tensor.element_size()
                )
                input_scale_name = ckpt_qname + ".input_global_scale"
                out_tensors[input_scale_name] = input_scale_tensor
                cb_output_tensor_names.add(input_scale_name)
                emitted_activation_scale_targets.add(ckpt_qname)
            if input_scale_bytes != payload["input_global_scale_bytes"]:
                raise AssertionError(
                    f"{canon}: serialized input_global_scale is "
                    f"{input_scale_bytes}B, accounting expected "
                    f"{payload['input_global_scale_bytes']}B"
                )
            actual = packed_bytes + scale_bytes + input_scale_bytes
            if actual != payload["tensor_payload_bytes"]:
                raise AssertionError(
                    f"{canon}: emitted {actual}B of CB tensor payload, "
                    f"accounting expected {payload['tensor_payload_bytes']}B"
                )
            cb_serialized_shapes[canon] = tuple(int(dim) for dim in w.shape)
            actual_cb_tensor_bytes += actual
            counts[fmt] += 1
        elif canon in stock_targets:
            # Stock rung: CT-pack via the shared compressed-tensors codec
            # (RTN default levers = the render the allocator cost measured; the
            # packed scales are the render's, M19). Emit the CT suffix tensors
            # verbatim; NOT added to the ignore list (it is quantized).
            fmt = stock_targets[canon]
            override = (_nvfp4_shared_global.get(canon)
                        if fmt == "NVFP4" else None)
            packed = _ct_quantize_2d(
                _decoded_cb_source_weight(
                    skeleton,
                    name,
                    model_weight_name=canon + ".weight",
                    fp8_scale_inv_map=_source_fp8_scale_map,
                ).to(device),
                fmt,
                nvfp4_global_real_override=override,
                input_global_scale_override=(
                    activation_scales_by_physical_target.get(ckpt_qname)
                    if fmt == "NVFP4"
                    else None
                ),
            )
            for suffix, t in packed.items():
                if "input" in suffix and (canon in sidecar_stock
                                          or canon in embedding_stock):
                    continue        # weight-only: sidecar tower / lookup table
                out_tensors[f"{ckpt_qname}.{suffix}"] = t.cpu().contiguous()
                if suffix == "input_global_scale":
                    emitted_activation_scale_targets.add(ckpt_qname)
            counts[assignment[canon]] += 1
        else:
            # Verbatim (BF16 passthrough, norms, embeddings, visual encoder,
            # lm_head) under the checkpoint name; 2-D unquantized Linears go to
            # the ignore list by their checkpoint/vLLM name.
            out_tensors[name] = tensor.contiguous()
            if ckpt_qname is not None and tensor.dim() >= 2:
                ignore.append(ckpt_qname)
            counts["copied"] += 1

    # --- Codebook tensors: shipped once per (ref, fmt) in a NON-safetensors-
    # globbed sidecar (cb_codebooks.pqcb) so vLLM's weight loader never sees
    # these non-parameter tensors. The plugin loads them explicitly via the
    # config's codebook_file pointer (external Gridbook runtime:
    # config.get_codebooks -> load_file(model_dir/cb_codebooks.pqcb)), keyed by each
    # scheme's codebook_ref. Sidecar-only: NOT written into model.safetensors. ---
    cb_tensor_blobs.update(materialized_codebook_tensors)
    codebook_file = "cb_codebooks.pqcb" if cb_tensor_blobs else None

    if set(cb_serialized_shapes) != set(cb_targets):
        missing = sorted(set(cb_targets) - set(cb_serialized_shapes))
        extra = sorted(set(cb_serialized_shapes) - set(cb_targets))
        raise AssertionError(
            "CB serialized-payload coverage does not match assignment: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if activation_execution_contract is not None and (
        emitted_activation_scale_targets
        != set(activation_scales_by_physical_target)
    ):
        raise AssertionError(
            "NVFP4 activation scalar coverage differs from the claimed "
            "execution contract: missing="
            f"{sorted(set(activation_scales_by_physical_target) - emitted_activation_scale_targets)[:8]}, "
            "extra="
            f"{sorted(emitted_activation_scale_targets - set(activation_scales_by_physical_target))[:8]}"
        )
    serialized_payload = cb_assignment_payload_breakdown(
        {qname: assignment[qname] for qname in cb_targets},
        cb_serialized_shapes,
        context=serialization_context,
    )
    if _recipe_cb_context_stamp is not None or _recipe_cb_tensor_stamps:
        validate_cb_assignment_serialization_stamps(
            {qname: assignment[qname] for qname in cb_targets},
            cb_serialized_shapes,
            context=serialization_context,
            stamps=_recipe_cb_tensor_stamps,
            where="export_nvfp4_cb",
        )
    if actual_cb_tensor_bytes != serialized_payload["tensor_payload_bytes"]:
        raise AssertionError(
            f"emitted CB tensor payload is {actual_cb_tensor_bytes}B, "
            "assignment accounting expected "
            f"{serialized_payload['tensor_payload_bytes']}B"
        )
    validate_cb_sidecar_tensors(
        serialized_payload,
        cb_tensor_blobs,
        where="export_nvfp4_cb",
    )
    serialized_payload_summary = cb_payload_summary(serialized_payload)

    if (
        _recipe_cb_render_identity is not None
        and verified_cb_source_qnames != set(cb_targets)
    ):
        raise AssertionError(
            "resident CB source-value validation did not cover the exact "
            "assignment: missing="
            f"{sorted(set(cb_targets) - verified_cb_source_qnames)[:8]}, "
            "extra="
            f"{sorted(verified_cb_source_qnames - set(cb_targets))[:8]}"
        )

    def _delegated_target_name(qname: str) -> str:
        # Sidecar towers lose the leading model. prefix in vLLM's module tree;
        # the LM and all physical tensor names retain their canonical names.
        if qname in sidecar_stock and qname.startswith("model."):
            return qname[len("model."):]
        return qname

    post_allocation_refinement = None
    _meta_ref = _recipe_payload.get("__prismaquant__", {})
    if isinstance(_meta_ref, dict) and "post_allocation_refinement" in _meta_ref:
        from prismaquant.cb_ldlq_refinement import validate_refinement_provenance

        post_allocation_refinement = validate_refinement_provenance(
            _meta_ref.get("post_allocation_refinement"),
            where="export_nvfp4_cb post_allocation_refinement",
        )
    # Inject trellis groups (WO-C C1): the wire is the only carrier.
    # Each trellis Linear gets its own group with format TRELLIS and a
    # scheme that is the consumer's gate input, not prose.
    trellis_scheme_groups: dict[str, dict] = {}
    for qname, (family, rate) in trellis_targets.items():
        wire = trellis_parsed[qname]
        blob_len = len(trellis_wires[qname])
        # Scheme copies the reference shape from
        # tools/make_trellis_smoke_checkpoint.py verbatim (WO-C contract):
        # family, body_rate_q256, rows, columns, wire_bytes.
        # The wire schema “gridbook.trellis.wire.v1” is pinned via the wire
        # header itself; the scheme repeats only the gate-relevant fields.
        scheme = {
            "family": family,
            "body_rate_q256": int(rate),
            "rows": int(wire.rows),
            "columns": int(wire.columns),
            "wire_bytes": int(blob_len),
        }
        # One target per group — per-role wires cannot be concatenated, so
        # grouping them would be the fused-module error.
        export_target = _resident_export_target(qname)
        trellis_scheme_groups[qname] = {
            "format": "TRELLIS",
            "targets": [export_target],
            "scheme": scheme,
        }

    quant_config = build_quant_config(
        assignment=assignment,
        cb_targets=cb_targets,
        source_targets=source_targets,
        # Embedding units are packed like a stock target but claimed by the
        # `quantized_embedding` declaration, so they must not also appear in a
        # config group -- the consumer refuses a unit owned by two dispatches.
        stock_targets={q: f for q, f in stock_targets.items()
                       if q not in embedding_stock},
        quantized_embedding_units=embedding_stock or None,
        by_group=by_group,
        cb_group_target_names=cb_group_target_names,
        codebooks=codebooks,
        col_weights=col_weights,
        codebook_tensors_by_name=cb_tensor_blobs,
        ignore=ignore,
        codebook_file=codebook_file,
        scale_coding=scale_coding,
        codebook_source=serialization_context.codebook_source,
        serialized_payload_summary=serialized_payload_summary,
        serialization_context=serialization_context,
        cb_render_identity=_recipe_cb_render_identity,
        research_cost_selection=_research_cost_selection,
        post_allocation_refinement=post_allocation_refinement,
        activation_execution_contract=activation_execution_contract,
        git_commit=_git_commit(),
        cb_target_name=_resident_export_target,
        delegated_target_name=_delegated_target_name,
        weight_only_stock_targets=sidecar_stock,
        streaming_provenance=None,
        include_tensor_formats=True,
    )
    # Merge trellis groups after build_quant_config — that builder owns the CB
    # vocabulary and must not be taught the trellis scheme; merging here keeps
    # the one-carrier invariant in this exporter alone.
    if trellis_scheme_groups:
        # Assign consecutive group ids after the CB/stock groups
        next_idx = len(quant_config.get("config_groups", {}))
        for qname in sorted(trellis_scheme_groups):
            quant_config["config_groups"][f"group_{next_idx}"] = trellis_scheme_groups[qname]
            next_idx += 1
        # Mix of families => mixed-precision (WO-C C1)
        quant_config["format"] = "mixed-precision"
        # Purge any trellis qnames that accidentally landed in ignore (they
        # are quantized, not passthrough). The earlier loop already avoided
        # adding them, but a profile that drops a key could leave a stale entry.
        quant_config["ignore"] = [n for n in quant_config.get("ignore", []) if n not in trellis_targets]
        # Also ensure BF16-assigned linears that are actually trellis are not
        # in ignore via the skeleton verbatim path — they were never added.
    if _strict_producer is not None:
        _strict_runtime_contract, _strict_policy_stamp = _strict_producer
        quant_config["format"] = "fp8_cb"
        quant_config["provenance"]["producer_policy"] = _strict_policy_stamp
        from prismaquant.rtx4090_artifact_census import (
            bind_rtx4090_source_provenance,
        )

        bind_rtx4090_source_provenance(
            quant_config, _strict_policy_stamp
        )
        quant_config["provenance"]["imatrix_sha256"] = (
            quant_config["provenance"]["cb_render_identity"][
                "col_weights_sha256"
            ]
        )

    # --- Write safetensors (params only) + the codebook sidecar + configs. ---
    published_containers, _tensor_sha256 = _write_cb_containers(
        out_tensors, out_dir, int(shard_bytes))
    _strict_content_receipt = None
    if _strict_producer is not None:
        from prismaquant.shipcard import verify_safetensors_content_once

        _strict_content_receipt = verify_safetensors_content_once(
            out_dir,
            expected_weight_manifest=None,
            expected_tensor_sha256=_tensor_sha256,
            expected_files=published_containers,
        )
    # Layout-invariant payload identity: `model_sha` binds container filenames
    # and sizes, so it moves with the shard budget; this digest does not.
    quant_config["provenance"]["tensor_payload_identity"] = (
        tensor_payload_identity(
            _tensor_sha256,
            include_tensor_sha256=_strict_producer is not None,
        )
    )
    # The route census principle 12 requires next to any bpp or KL claim; the
    # streaming exporter stamps the identical key. Its shape makes an
    # unattested lane impossible to read as a clean one.
    quant_config["provenance"]["cb_route_status"] = cb_route_status_provenance
    if codebook_file:
        # The .pqcb is a plain safetensors blob under a non-globbed extension:
        # the plugin reads it with safetensors.load_file, vLLM's *.safetensors
        # weight globber skips it (LAYOUT.md §3 codebook contract).
        save_file({k: v.contiguous() for k, v in cb_tensor_blobs.items()},
                  str(out_dir / codebook_file),
                  metadata={"format": "pt", "quant_method": "gridbook"})
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    # WO-C C1: format becomes mixed-precision when assignment mixes families
    _quant_format = "fp8_cb" if _strict_producer is not None else "nvfp4_cb"
    if trellis_targets:
        # Trellis alone uses mixed-precision per the smoke checkpoint contract;
        # mixing trellis with any CB/stock also mixes.
        _has_other_quant = bool(cb_targets or stock_targets or source_targets)
        if _has_other_quant or trellis_targets:
            _quant_format = "mixed-precision"
            quant_config["format"] = "mixed-precision"
        else:
            _quant_format = quant_config.get("format", "mixed-precision")
    else:
        # No trellis: decide mixed from CB vs stock mixing
        _families = set()
        if cb_targets:
            _families.add("cb")
        if stock_targets:
            _families.update(stock_targets.values())
        if source_targets:
            _families.add("source")
        if len(_families) > 1:
            _quant_format = "mixed-precision"
            quant_config["format"] = "mixed-precision"
    config["quantization_config"] = {
        "quant_method": "gridbook",
        "format": _quant_format,
        "config_file": "quant_config.json",
        **({"codebook_file": codebook_file} if codebook_file else {}),
    }
    # WO-C C3: selection_serving_lane_provenance — structured serve flags
    # The CB gate already stamped requires_serve_flags; mirror it into the
    # legacy selection_serving_lane_provenance key so a gate can read it.
    if trellis_targets:
        quant_config["provenance"]["selection_serving_lane_provenance"] = cb_route_status_provenance
        # Also record trellis-specific provenance explicitly for docs
        quant_config["provenance"]["trellis_route_status"] = cb_route_status_provenance
        # Ensure route histogram and serve flags are visible next to bpp (principle 12)
        if cb_route_status_provenance.get("requires_serve_flags"):
            quant_config["provenance"]["requires_serve_flags"] = cb_route_status_provenance["requires_serve_flags"]
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Copy tokenizer / generation / multimodal sidecars verbatim (best effort).
    # The multimodal preprocessor configs are REQUIRED for VLM checkpoints
    # (e.g. Qwen3-VL): vLLM's input processor calls
    # `image_processor.from_pretrained(model_dir)` at load and hard-fails
    # without preprocessor_config.json — the artifact will not serve. Copy the
    # chat template too so chat/tool serving matches the source.
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt",
                "preprocessor_config.json", "video_preprocessor_config.json",
                "processor_config.json", "chat_template.jinja",
                "chat_template.json"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    if ldlq_telemetry is not None:
        ldlq_telemetry.publish(out_dir, quant_config)
    # Open the refusal record before inventory finalization: the preliminary
    # quant_config binds the CB identity, while shipcard.json itself must be
    # measured by the recursive inventory and the hard artifact budget.
    from prismaquant.shipcard import (
        open_cb_export_shipcard,
        safetensors_content_receipt_manifest,
    )

    open_cb_export_shipcard(
        out_dir,
        quant_config,
        source_model=model_dir,
        layer_config_path=layer_config_path,
        exporter="export_nvfp4_cb",
        weight_content_manifest=(
            safetensors_content_receipt_manifest(_strict_content_receipt)
            if _strict_content_receipt is not None
            else None
        ),
        build_extra={
            "routed_codebook_books": {
                "keying": ["role"] if routed_role_plans else [],
                "pooled_stack_units": 0,
                "per_role_units": len(routed_role_plans),
                "fused_targets_with_split_books": sorted(split_book_targets),
                "per_role_books_override": bool(
                    split_book_targets and allow_per_role_books
                ),
            },
        },
    )
    # Final measured bytes are a separate scope from CB tensor-data pricing:
    # include safetensors headers, JSON, tokenizer files, and every other
    # regular file.  The helper embeds a self-consistent inventory in
    # quant_config.json and re-checks the exact CB spans in the final files.
    finalize_cb_export_artifact_inventory(
        out_dir,
        quant_config,
        serialized_payload=serialized_payload_summary,
        cb_tensor_names=sorted(cb_output_tensor_names),
        codebook_file=codebook_file,
        expected_model_files=published_containers,
        whole_artifact_budget_bytes=(
            int(_whole_artifact_budget["budget_bytes"])
            if _whole_artifact_budget is not None
            else None
        ),
    )
    if _strict_producer is not None:
        from prismaquant.rtx4090_qwen38_policy import (
            is_rtx4090_validation_only_policy,
            validate_rtx4090_quant_config_manifest,
        )

        validate_rtx4090_quant_config_manifest(
            quant_config,
            runtime_contract=_strict_runtime_contract,
            allow_unreleasable_validation_only=(
                is_rtx4090_validation_only_policy(_strict_policy_stamp)
            ),
            artifact_dir=out_dir,
            artifact_content_receipt=_strict_content_receipt,
            where="export_nvfp4_cb finalized RTX4090 manifest",
        )
    return dict(counts)


def _to_device(codebook, device):
    if isinstance(codebook, (tuple, list)):
        return tuple(t.to(device) for t in codebook)
    return codebook.to(device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="HF model dir (config.json + *.safetensors, bf16)")
    ap.add_argument("--layer-config", required=True,
                    help="assignment JSON (qname -> CB format)")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES,
                    help="Approx per-shard size in bytes (default 1 GiB), the "
                         "same flag, default, and partition rule as "
                         "export_native_compressed. A single tensor larger "
                         "than this still gets its own shard. One resulting "
                         "shard is published as model.safetensors with no "
                         "index; pass a value at least as large as the "
                         "artifact to reproduce the legacy single-container "
                         "layout.")
    ap.add_argument("--col-weights", required=True,
                    help="pickle: {qname: per-column importance tensor}")
    ap.add_argument(
        "--activation-cache-dir",
        default=None,
        help="probe activation cache used to calibrate the versioned static "
        "W4A4 input_global_scale contract",
    )
    ap.add_argument(
        "--activation-scale-policy",
        default=None,
        choices=sorted((
            "legacy_6_over_calibration_amax.v1",
            "full_e4m3_range_448x6_over_calibration_amax.v1",
            "mse_grid_calibrated.v1",
        )),
        help="explicit static W4A4 scale policy; default preserves the "
        "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE compatibility setting",
    )
    ap.add_argument("--codebook-source", default="lattice",
                    choices=["lattice", "learned"],
                    help="fixed lattice sidecar or shared per-role "
                    "learned codebooks trained at export time")
    ap.add_argument("--codebook-iters", type=int, default=4)
    ap.add_argument("--codebook-seed", type=int, default=0)
    ap.add_argument("--no-scale-sweep", action="store_true",
                    help="one-shot amax/grid-max scale (A/B only; default is "
                    "the joint scale sweep, IQ-rendering parity)")
    ap.add_argument(
        "--allow-unstamped-research",
        action="store_true",
        help="unsafe research-only escape hatch for a bare CB assignment; "
        "production recipes must carry a source/imatrix-complete render "
        "identity",
    )
    ap.add_argument(
        "--allow-research-cost-selection",
        action="store_true",
        help="explicitly acknowledge export of an allocation derived from "
             "the sanctioned study-grade assembled cost table; recorded in "
             "artifact provenance",
    )
    ap.add_argument(
        "--allow-per-role-books",
        action="store_true",
        help="ship fused routed weights whose scheme names more than one "
        "codebook (books burned per (layer, projection, rung) rather than "
        "pooled per (layer, stack, rung), campaign rule R1). Refused by "
        "default; passing it stamps the acknowledgement onto the shipcard.",
    )
    ap.add_argument("--scale-coding", default=cb.SCALE_CODING_TWO_TIER,
                    choices=[cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER],
                    help="fp4 scale coding: production layout-v2 two-tier "
                    "super+sub coding (default), or explicit legacy v1 e4m3 "
                    "plane for backward-compatible artifacts")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--producer-policy",
        default=None,
        help="opt-in strict producer policy id; the RTX4090 policy requires "
             "--producer-runtime-contract",
    )
    ap.add_argument(
        "--producer-runtime-contract",
        default=None,
        help="explicit Gridbook v11 runtime_contract.json for the strict "
             "producer policy (never inferred from the current v4 pin)",
    )
    args = ap.parse_args(argv)
    from .prismasnap_contract import refuse_prismasnap_for_unvalidated_lane

    refuse_prismasnap_for_unvalidated_lane(
        args.model_dir, lane="Gridbook/codebook"
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("export_nvfp4_cb")

    with open(args.col_weights, "rb") as fh:
        col_weights = pickle.load(fh)
    col_weights = {k: torch.as_tensor(v) for k, v in col_weights.items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep,
        scale_coding=args.scale_coding,
        allow_unstamped_research=args.allow_unstamped_research,
        allow_research_cost_selection=args.allow_research_cost_selection,
        allow_per_role_books=args.allow_per_role_books,
        activation_cache_dir=args.activation_cache_dir,
        activation_scale_policy=args.activation_scale_policy,
        shard_bytes=args.shard_bytes,
        producer_policy=args.producer_policy,
        producer_runtime_contract=args.producer_runtime_contract,
    )
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
