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
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import (
    CB_FORMAT_NAMES,
    family_for,
    parse_format_name,
    subtable_bit_widths,
)
from prismaquant.cb_export_config import (
    build_quant_config,
    codebook_tensor_names as _codebook_tensor_names,
    codebook_tensors as _codebook_tensors,
)
from prismaquant.layer_config import load_assignment
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
    cb_tensor_payload_breakdown,
    finalize_cb_export_artifact_inventory,
    resolve_cb_encode_tier,
    whole_artifact_budget_from_assignment_payload,
    validate_cb_sidecar_tensors,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)
from prismaquant.nvfp4_activation_contract import (
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_EXECUTION,
    build_execution_contract,
    calibrated_input_global_scales_with_sources,
    input_global_scale_tensor,
    resolve_input_global_scale_policy,
)

# This exporter's own declaration of what the mixed CB container can carry —
# exactly the coverage gate in `export_cb` below: the CB rung families, the two
# stock-CT schemes the plugin delegates to vLLM's CompressedTensors path
# (NVFP4, FP8_E4M3 <- FP8_DYNAMIC), the verbatim FP8_SOURCE passthrough and the
# BF16 container passthrough. The `nvfp4_cb` serving profile's export lane
# derives its format menu from this constant
# (serving_profile_specs/nvfp4_cb.json), so the allocator can never spend budget
# on a rung this exporter would hard-fail on.
EXPORTABLE_FORMATS = CB_FORMAT_NAMES | frozenset(
    {"NVFP4", "FP8_E4M3", "FP8_SOURCE", "BF16"}
)

def _git_commit() -> str:
    from prismaquant.aura_cost import _git_commit as _aura_git_commit

    return _aura_git_commit() or "unknown"


def _parse_cb_format(fmt: str) -> tuple[str, str, int] | None:
    """``NVFP4_CB_K{k}`` -> (fp4, product, k); ``NVFP4_CB_S{k}`` -> (fp4,
    signed, k); ``FP8_CB_K{k}`` -> (fp8, product, k). None for non-CB."""
    parsed = parse_format_name(str(fmt).strip().upper())
    if parsed is None:
        return None
    family, k = parsed
    return family.grid, family.mode, k


def _role_of(qname: str) -> str:
    """Shared-codebook grouping key — the Linear's projection role (last qname
    component), e.g. ``model.layers.3.mlp.gate_proj`` -> ``gate_proj``."""
    return qname.split(".")[-1]


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
    shared-per-role logic): signed -> positive magnitude table; product ->
    n_sub grid-snapped sub-tables; full -> one (2^k, 8) table."""
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
    if mode == "signed":
        signed_bits = subtable_bit_widths(
            k,
            "signed",
            family_for(grid, "signed").n_sub,
        )[0]
        return cb.learn_codebook(vec.abs(), signed_bits, grid=grid,
                                 col_weights=wq, positive=True, iters=iters,
                                 seed=seed).cpu()
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
    activation_cache_dir: str | Path | None = None,
    activation_scale_policy: str | None = None,
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

    ``scale_coding``: ``"two_tier"`` (production layout v2; fp4 targets write
    4k+9 bytes per superblock) or explicit legacy ``"v1"`` (4k+16). Readers
    remain backward compatible with v1; new artifacts default to v2.
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

    assignment = load_assignment(layer_config_path)
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
    # Fail-closed gate: production scope with gate disabled is research-only
    if isinstance(_recipe_cb_context_stamp, dict):
        _scope = str(_recipe_cb_context_stamp.get("ldlq_scope", "none")).strip().lower()
        if _scope != "none":
            from prismaquant.nvfp4_cb_formats import _ldlq_gate_enabled

            if not _ldlq_gate_enabled():
                raise RuntimeError(
                    f"export_nvfp4_cb: production LDLQ scope {_scope!r} requires PRISMAQUANT_CB_LDLQ_GATE=1; "
                    "ungated LDLQ is research-only without a context-stamped production artifact"
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
    # BF16-passthrough (the mixed container, LAYOUT.md §4; "FP8 in every
    # recipe"). ---
    cb_targets: dict[str, tuple[str, str, int]] = {}   # qname -> (grid,mode,k)
    stock_targets: dict[str, str] = {}                 # qname -> "NVFP4"|"FP8_E4M3"
    source_targets: list[str] = []                     # FP8_SOURCE passthrough
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
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
            f"(verbatim fp8 passthrough) + BF16 passthrough only")

    # Per-expert-on-disk MoE checkpoints: assemble only packed parents that
    # are actually quantized. Packing every detected bank mutates omitted/BF16
    # LFM layers from their loadable ``experts.E.w1/w2/w3.weight`` layout into
    # aggregate ``gate_up_proj/down_proj.weight`` passthrough tensors that
    # vLLM's architecture loader cannot consume.
    _pack_skeleton_experts(
        skeleton,
        _profile,
        fp8_scale_inv_map=_source_fp8_scale_map,
        target_qnames=set(cb_targets) | set(stock_targets),
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
    }
    activation_execution_contract = None
    activation_scales_by_physical_target: dict[str, float] = {}
    activation_scale_policy_id = None

    # --- Resolve/train codebooks, grouped by (ref, format). ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))

    # (ref, fmt) -> codebook object; ref = "lattice" or role.
    codebooks: dict[tuple[str, str], object] = {}
    # qname -> (ref, fmt, codebook, source_kind)
    target_cb: dict[str, tuple[str, str, object, str]] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for qname, (grid, mode, k) in cb_targets.items():
        fmt = assignment[qname]
        ref = _role_of(qname) if source == "learned" else "lattice"
        by_group.setdefault((ref, fmt), []).append(qname)

    for (ref, fmt), qnames in by_group.items():
        grid, mode, k = cb_targets[qnames[0]]
        if source == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
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
            kind = "learned"
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
    materialized_codebook_digests = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, tensor in materialized_codebook_tensors.items()
    }
    _env_cb_context = cb_serialization_context_from_env()
    serialization_context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        scale_sweep=bool(scale_sweep),
        ldlq=_env_cb_context.ldlq,
        ldlq_scope=getattr(_env_cb_context, "ldlq_scope", "all" if _env_cb_context.ldlq else "none"),
        minchain=_env_cb_context.minchain,
        minchain_version=_env_cb_context.minchain_version,
        encode_tier=resolve_cb_encode_tier(),
        activation_contract=_claimed_activation_contract,
        activation_execution=(
            NVFP4_ACTIVATION_EXECUTION
            if _claimed_activation_contract is not None
            else None
        ),
        codebook_refs={
            qname: _codebook_tensor_names(ref, fmt, codebook)
            for qname, (ref, fmt, codebook, _kind) in target_cb.items()
        },
        codebook_content_digests=materialized_codebook_digests,
    )
    validate_cb_serialization_context_stamp(
        _recipe_cb_context_stamp,
        serialization_context,
        where="export_nvfp4_cb",
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
        if canon in packed_qnames:
            grid, mode, k = cb_targets[canon]
            ref, fmt, codebook, _ = target_cb[canon]
            cbook = _to_device(codebook, device)
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
            packed, fields = cb.nvfp4_cb_pack(
                w, k, grid=grid, mode=mode,
                col_weights=col_weights[canon].to(device),
                codebook=cbook, scale_sweep=scale_sweep,
                scale_coding=(scale_coding if grid == "fp4"
                              else cb.SCALE_CODING_V1),
                encode_tier=serialization_context.encode_tier,
                ldlq=ldlq_for_this,
                activation_rows=(
                    ldlq_activation_loader.load(
                        canon,
                        stack_size=(int(w.shape[0]) if w.dim() == 3 else None),
                    )
                    if ldlq_for_this and ldlq_activation_loader is not None else None
                ))
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
                if canon in sidecar_stock and "input" in suffix:
                    continue        # weight-only sidecar group (see above)
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
    quant_config = build_quant_config(
        assignment=assignment,
        cb_targets=cb_targets,
        source_targets=source_targets,
        stock_targets=stock_targets,
        by_group=by_group,
        codebooks=codebooks,
        col_weights=col_weights,
        codebook_tensors_by_name=cb_tensor_blobs,
        ignore=ignore,
        codebook_file=codebook_file,
        scale_coding=scale_coding,
        codebook_source=source,
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

    # --- Write safetensors (params only) + the codebook sidecar + configs. ---
    save_file(out_tensors, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "quant_method": "gridbook"})
    if codebook_file:
        # The .pqcb is a plain safetensors blob under a non-globbed extension:
        # the plugin reads it with safetensors.load_file, vLLM's *.safetensors
        # weight globber skips it (LAYOUT.md §3 codebook contract).
        save_file({k: v.contiguous() for k, v in cb_tensor_blobs.items()},
                  str(out_dir / codebook_file),
                  metadata={"format": "pt", "quant_method": "gridbook"})
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_file": "quant_config.json",
        **({"codebook_file": codebook_file} if codebook_file else {}),
    }
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
        expected_model_files=["model.safetensors"],
        whole_artifact_budget_bytes=(
            int(_whole_artifact_budget["budget_bytes"])
            if _whole_artifact_budget is not None
            else None
        ),
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
    ap.add_argument("--scale-coding", default=cb.SCALE_CODING_TWO_TIER,
                    choices=[cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER],
                    help="fp4 scale coding: production layout-v2 two-tier "
                    "super+sub coding (default), or explicit legacy v1 e4m3 "
                    "plane for backward-compatible artifacts")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
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
        activation_cache_dir=args.activation_cache_dir,
        activation_scale_policy=args.activation_scale_policy,
    )
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
