#!/usr/bin/env python3
"""Derive dual-basis cost: NVFP4 LDLQ (scope nvfp4) + FP8 preserved.

Three planes (all preserved distinctly, never relabeled):
  1. NVFP4_CB raw  — immutable cost bank (RUN_ROOT/shards), never overwritten.
  2. NVFP4_CB LDLQ — measured cost/allocator/export plane, scope nvfp4, gated.
  3. FP8_CB raw    — preserved deep-equal from canonical cost_merged.pkl
                   (fp8_source=preserved_deep_equal_from_raw_merged), no
                   reinterpolation in this derive. Historical archived campaigns
                   did cross-family NVFP4→FP8 interpolation; future FP8
                   interpolation must use raw/no-LDLQ values, never the LDLQ plane.

Per-tensor identities stamp ldlq:true only for NVFP4 family when scope=nvfp4.
LDLQ is low-rate NVFP4 K12-K18 candidate only, gated by direct activation
replay output_mse (not diagonal col-weighted), with pooled missing-expert prior
identical to CBLDLQActivationLoader and fail-closed on malformed/missing.
No served KL/PPL evidence; not production-default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

# Worktree-local import only; never import from stale pq-nestpilot worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Shared fail-closed ownership helper for packed-projection data.
# Attempts factor-cache clear, always drops holder["data"] before empty_cache,
# always attempts empty_cache even if clear failed, preserves body exception
# as context and chains multiple cleanup failures.
# Do not use `except Exception: raise` or broad `pass`; do not use a helper
# that merely deletes its own argument while caller still owns `data`.
# ---------------------------------------------------------------------------
import sys as _sys_cleanup  # local alias for cleanup helper


def _finalize_packed_holder_cleanup(holder: dict, device: torch.device) -> None:
    """Shared cleanup for holder = {"data": <packed dict>}. Fail-closed.

    Order: attempt factor clear; always drop holder ownership; always attempt
    empty_cache even if clear failed. Preserves body exception plus every
    cleanup failure via ExceptionGroup (or cause chain on older Python) without
    fragile __context__ overwrite and without swallowing KeyboardInterrupt.
    """
    body_exc = _sys_cleanup.exc_info()[1]
    clear_exc: BaseException | None = None
    empty_exc: BaseException | None = None
    del_exc: BaseException | None = None
    try:
        from prismaquant.nvfp4_cb_formats import clear_ldlq_factor_cache

        clear_ldlq_factor_cache()
    except BaseException as exc:  # noqa: BLE001
        clear_exc = exc
    # Always drop owner's data reference before empty_cache, even if clear failed
    try:
        if "data" in holder:
            del holder["data"]
    except BaseException as exc:  # noqa: BLE001
        del_exc = exc
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except BaseException as exc:  # noqa: BLE001
            empty_exc = exc
    # Aggregate cleanup failures
    cleanup_errs: list[BaseException] = []
    for e in (clear_exc, del_exc, empty_exc):
        if e is not None:
            cleanup_errs.append(e)
    if not cleanup_errs:
        return
    # If body also failed, aggregate body + cleanups
    if body_exc is not None:
        # Prefer ExceptionGroup (Python 3.11+) for inspectable aggregation
        try:
            raise BaseExceptionGroup("body and cleanup failures", [body_exc, *cleanup_errs])  # type: ignore[attr-defined]
        except NameError:
            # Fallback: chain via context without overwriting __context__ of existing
            # Use first cleanup as primary, attach others via __context__ chain cautiously
            primary = cleanup_errs[0]
            # Build chain: body -> primary -> rest
            cur = primary
            for extra in cleanup_errs[1:]:
                extra.__cause__ = cur  # type: ignore[attr-defined]
                cur = extra
            cur.__cause__ = body_exc  # type: ignore[attr-defined]
            raise cur
    # Body succeeded but cleanup failed: raise cleanup aggregation
    if len(cleanup_errs) == 1:
        raise cleanup_errs[0]
    try:
        raise BaseExceptionGroup("cleanup failures", cleanup_errs)  # type: ignore[attr-defined]
    except NameError:
        primary = cleanup_errs[0]
        cur = primary
        for extra in cleanup_errs[1:]:
            extra.__cause__ = cur  # type: ignore[attr-defined]
            cur = extra
        raise cur

from prismaquant import format_registry as fr
from prismaquant.cb_ldlq_fused_activation import concat_equal_member_samples as FUSED_ACTIVATION_POLICY_V1
from prismaquant.nvfp4_cb_footprint import CBSerializationContext, cb_fields_for_context, cb_serialization_context_stamp
from prismaquant.nvfp4_cb_formats import (
    canonical_nvfp4_cb_single_output_mse,
    canonical_nvfp4_cb_payload_identity,
    encode_packed_parent_leaf_local,
    iter_nvfp4_cb_recon_chunks,
    nvfp4_cb_reconstruct,
)

# Authoritative packed-MoE planning — single source, no duplication.
# Derive imports only the public production helpers; no private leading-underscore
# APIs. This guarantees derive and streaming export share exact planning, pooling,
# and materialization order.
from prismaquant.export_nvfp4_cb_streaming import (
    get_packed_expert_col_weights,
    get_packed_expert_projection_names,
    get_packed_moe_planning,
    get_expert_weight,
    open_packed_weight_source,
)

THIS_FILE = Path(__file__).resolve()

# Campaign data paths are intentionally pinned to the DSv4-Flash-0731
# cost-ldlq burn (single immutable source). They are defaults, not globals:
# the derive tool is campaign-specific and its provenance (hashes, content_key,
# manifests) binds the exact source/col-weights/act cache used. Overriding
# via env (PQ_DERIVE_*) or CLI (--run-root etc.) is supported for testing
# or re-use on a different campaign checkout, but the override is explicit,
# does not affect any global cache, and the resulting artifact's manifest
# records the overridden paths/hashes so drift is detectable. If no override
# is given the pinned defaults are used byte-for-byte.
def _env_path(env_name: str, default: str) -> Path:
    val = os.environ.get(env_name)
    return Path(val) if val and str(val).strip() else Path(default)

RUN_ROOT = _env_path("PQ_DERIVE_RUN_ROOT", "/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq")
DERIVED_ROOT = _env_path("PQ_DERIVE_DERIVED_ROOT", "/home/rob/dq-runs/dsv4-flash-0731/ldlq-reexport-2026-08-07/derived-cost")
DERIVED_SHARDS = DERIVED_ROOT / "shards"
DERIVED_WARM = DERIVED_ROOT / "warm-state-nvfp4"
DERIVED_CHECKPOINTS = DERIVED_ROOT / "projection_checkpoints"
DERIVED_DENSE_CHECKPOINTS = DERIVED_ROOT / "dense_checkpoints"
DERIVED_RAW_PLANE = DERIVED_ROOT / "raw_plane"
SOURCE = _env_path("PQ_DERIVE_SOURCE", "/home/rob/dq-runs/dsv4-flash-0731/source")
BY_LAYER = _env_path("PQ_DERIVE_BY_LAYER", "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts-mxfp4/probe-k12k18/by-layer")
COL_WEIGHTS = _env_path("PQ_DERIVE_COL_WEIGHTS", "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts-mxfp4/cb_col_weights.pkl")
ACT_ROOT = _env_path("PQ_DERIVE_ACT_ROOT", "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/act")
RAW_SHARDS = RUN_ROOT / "shards"
RAW_MERGED = _env_path("PQ_DERIVE_RAW_MERGED", "/home/rob/dq-runs/dsv4-flash-0731/cost-ldlq/burn-afast/cost_merged.pkl")
DERIVED_MERGED = DERIVED_ROOT / "cost_merged_derived.pkl"

CTX_NVFP4_LDLQ = CBSerializationContext.production(ldlq_scope="nvfp4", encode_tier="balanced")
CTX_RAW = CBSerializationContext.production(ldlq_scope="none", encode_tier="balanced")
NVFP4_RUNGS = tuple(range(12, 19))

# ---------------------------------------------------------------------------
# Authoritative production planning — single source, no duplication.
# Derive shares the exact planning implementation with the streaming exporter
# via the public helper get_packed_moe_planning. No private underscore APIs,
# no duplicated regex, no hardcoded fallback enumeration.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cached_packed_planning(source_str: str) -> tuple[Any, dict[str, Any], dict[str, Any], frozenset[str]]:
    """Source-keyed planning cache — distinct SOURCE values get distinct entries."""
    return get_packed_moe_planning(source_str)


def _get_packed_planning() -> tuple[Any, dict[str, Any], dict[str, Any], frozenset[str]]:
    """Return (profile, expert_groups, expert_stack_members, packed_param_names).

    SOURCE is authoritative: CLI --source overrides PQ_DERIVE_SOURCE env.
    Import-time env is only the fallback when CLI is absent (SOURCE global
    already reflects that). No main-time env re-override. Cache is source-keyed
    and cleared whenever SOURCE changes via CLI.
    """
    return _cached_packed_planning(str(SOURCE))


def _clear_packed_planning_cache() -> None:
    """Clear the planning cache — must be called whenever SOURCE changes."""
    _cached_packed_planning.cache_clear()

_ACT_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")
SCHEMA = "prismaquant.dsv4_afast_layer_shard.v3"
PROJECTION_RUNG_SCHEMA = "prismaquant.dsv4_dual_basis_projection_rung.v3"
DENSE_RUNG_SCHEMA = "prismaquant.dsv4_dual_basis_dense_rung.v1"

# ---------------------------------------------------------------------------
# Hashing / atomic file helpers (ported from protected campaign, with digest
# provenance instead of external import).  Hot-loop filesystem hashing is
# eliminated: col/source/tool/module SHAs are cached once per process.
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()

def content_sha256_float32(tensor: torch.Tensor) -> str:
    v = torch.as_tensor(tensor).detach().cpu().to(torch.float32).contiguous()
    return hashlib.sha256(v.numpy().astype("<f4", copy=False).tobytes()).hexdigest()

def _sha(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as h:
        pickle.dump(payload, h, protocol=pickle.HIGHEST_PROTOCOL)
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def atomic_bytes_copy(src: Path, dst: Path) -> str:
    """Atomic copy of raw plane shard; returns sha256 of written bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    data = src.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dst)
    # verify copy is bit-identical
    assert sha256_file(dst) == h == sha256_file(src)
    return h

def require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"packed NVFP4 LDLQ requires CUDA, got device={device} cuda_available={torch.cuda.is_available()}")

def require_ldlq_gate_enabled() -> None:
    from prismaquant.nvfp4_cb_formats import _ldlq_gate_enabled

    if not _ldlq_gate_enabled():
        raise RuntimeError(
            "derive_dual_basis_packed requires PRISMAQUANT_CB_LDLQ_GATE=1 (gated LDLQ); refusing to derive with gate disabled (hostile env would silently produce ungated LDLQ)"
        )

# Cached precompute: avoid hashing 466MB col-weights or source index in hot loops.
@lru_cache(maxsize=1)
def _cached_col_weights_sha256() -> str:
    return sha256_file(COL_WEIGHTS)

@lru_cache(maxsize=1)
def _load_col_weights_cached() -> dict[str, Any]:
    """Load and deserialize the 466 MB col-weights mapping once per process."""
    return pickle.loads(COL_WEIGHTS.read_bytes())

@lru_cache(maxsize=1)
def _cached_source_index_sha256() -> str:
    return sha256_file(SOURCE / "model.safetensors.index.json")

@lru_cache(maxsize=1)
def _cached_tool_sha256() -> str:
    return sha256_file(THIS_FILE)

@lru_cache(maxsize=1)
def _cached_module_shas() -> dict[str, str]:
    base = Path(__file__).resolve().parents[1]
    return {
        "derive_tool": sha256_file(THIS_FILE),
        "nvfp4_cb_footprint": sha256_file(base / "prismaquant/nvfp4_cb_footprint.py"),
        "nvfp4_cb_formats": sha256_file(base / "prismaquant/nvfp4_cb_formats.py"),
        "cb_ldlq": sha256_file(base / "prismaquant/cb_ldlq.py"),
    }

def _activation_evidence_digest(qname: str) -> str:
    """Content digest for dense activation evidence: SHA of file bytes or empty sentinel."""
    p = act_path(qname)
    if not p.is_file():
        return "missing"
    return sha256_file(p)

# ---------------------------------------------------------------------------
# Ported verified loader logic (local worktree-owned, with source-file digest
# provenance).  Do NOT import from /home/rob/pq-nestpilot-wt at runtime.
# ---------------------------------------------------------------------------

def act_path(qname: str) -> Path:
    return ACT_ROOT / (_ACT_FNAME_SUB.sub("__", qname) + ".pt")

def load_direct_activation(qname: str, width: int) -> torch.Tensor:
    path = act_path(qname)
    if not path.is_file():
        return torch.empty((0, width), dtype=torch.float32)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    value = blob.get("inputs") if isinstance(blob, dict) else None
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{qname}: activation entry has no rank-2 inputs")
    if int(value.shape[1]) != int(width):
        raise ValueError(f"{qname}: activation width {value.shape[1]} != weight width {width}")
    return value.detach().to(torch.float32).contiguous()

def load_layer_identity(layer: int) -> tuple[dict, dict]:
    from prismaquant.production_weight_cache import validate_cb_render_identity_metadata
    path = BY_LAYER / f"layer_{layer:03d}.pkl"
    payload = pickle.loads(path.read_bytes())
    identity = copy.deepcopy(payload["provenance"]["cb_render_identity"])
    historical_missing_ldlq = "ldlq" not in identity.get("cb_serialized_payload", {})
    if historical_missing_ldlq:
        identity["cb_serialized_payload"]["ldlq"] = False  # type: ignore[index]
    validate_cb_render_identity_metadata(
        identity,
        require_source_complete=True,
        where=f"DSV4 verified by-layer store layer {layer}",
    )
    foreign = [q for q in payload["costs"] if not str(q).startswith(f"model.layers.{layer}.")]
    if foreign:
        raise AssertionError(f"layer {layer}: by-layer store holds foreign qnames (e.g. {foreign[0]})")
    if Path(payload["meta"]["model"]).resolve() != SOURCE.resolve():
        raise AssertionError(f"layer {layer}: source path mismatch")
    if Path(payload["meta"]["incremental_shard"]["activation_cache_dir"]).resolve() != ACT_ROOT.resolve():
        raise AssertionError(f"layer {layer}: activation cache mismatch")
    return payload, {
        "identity": identity,
        "path": str(path),
        "sha256": sha256_file(path),
        "historical_ldlq_missing_inferred_false": historical_missing_ldlq,
    }

def load_packed_projection(
    layer: int,
    packed_proj: str,
    *,
    device: torch.device,
    identity: Mapping[str, Any],
    all_col_weights: Mapping[str, Any],
    model_to_shard: Mapping[str, str],
    model_to_ckpt: Mapping[str, str],
    scale_map: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one authoritative packed parent (e.g. gate_up_proj) with fused semantics.

    Materializes weights in the exact gate-then-up order and shape as the
    public materializer, pools col-weights with the same mean rule, and loads
    the same activation tuple as the streaming exporter.
    """
    from prismaquant.production_weight_cache import validate_cb_render_source_weight
    from prismaquant.cb_ldlq import CBLDLQActivationLoader

    profile, expert_groups, expert_stack_members, packed_names = _get_packed_planning()
    if packed_proj not in packed_names:
        raise AssertionError(f"{packed_proj!r} not in authoritative packed set {sorted(packed_names)} for SOURCE {SOURCE}")
    projections = get_packed_expert_projection_names(profile, packed_proj)
    packed_qname = f"model.layers.{layer}.mlp.experts.{packed_proj}"
    # Find exactly one (prefix, group) with f"{prefix}.{packed_proj}" == packed_qname
    matched = [(p, g) for p, g in expert_groups.items() if f"{p}.{packed_proj}" == packed_qname]
    if len(matched) != 1:
        raise AssertionError(f"{packed_qname}: expected exactly one prefix match, got {len(matched)} { [p for p,_ in matched]}")
    prefix, group = matched[0]
    expected_members = expert_stack_members.get(packed_qname)
    if expected_members is None or len(expected_members) == 0:
        raise AssertionError(f"{packed_qname}: missing from authoritative planning members {sorted(expert_stack_members)}")
    # Derive expert ids per projection from expected map, require identical contiguous ids
    # Expert IDs must be exactly 0..N-1; activation tuple indexing and export topology are zero-based.
    ids_per_proj: dict[str, list[int]] = {}
    for proj in projections:
        ids = sorted({eid for (p, eid) in expected_members if p == proj})
        if not ids:
            raise AssertionError(f"{packed_qname}: projection {proj} has no members")
        if ids != list(range(len(ids))):
            raise AssertionError(f"{packed_qname}: projection {proj} ids must be exactly list(range(len(ids))) got {ids} (nonzero-start contiguous IDs are rejected because activation tuple indexing and export topology are zero-based)")
        ids_per_proj[proj] = ids
    first_ids = next(iter(ids_per_proj.values()))
    for proj, ids in ids_per_proj.items():
        if ids != first_ids:
            raise AssertionError(f"{packed_qname}: ids differ across projections {proj} {ids} vs {first_ids}")
    if len(expected_members) != len(first_ids) * len(projections):
        raise AssertionError(f"{packed_qname}: member count {len(expected_members)} != {len(first_ids)}*{len(projections)}")
    # Validate group coverage matches expected ids
    for proj in projections:
        if proj not in group:
            raise AssertionError(f"{packed_qname}: prefix group missing projection {proj}")
        grp_ids = sorted(group[proj].keys())
        if grp_ids != first_ids:
            raise AssertionError(f"{packed_qname}: group ids for {proj} {grp_ids} != expected {first_ids}")
    expert_ids = first_ids
    n_experts = len(expert_ids)
    # Loader
    loader = CBLDLQActivationLoader(
        activation_cache_dir=ACT_ROOT,
        model_dir=SOURCE,
        profile=profile,
        expert_stack_members=expert_stack_members,
        replay_device=str(device) if device.type == "cuda" else None,
    )
    loaded = loader.load(packed_qname, stack_size=n_experts)
    if not isinstance(loaded, tuple) or len(loaded) != n_experts:
        raise AssertionError(f"{packed_qname}: loader returned {type(loaded)} len {len(loaded) if isinstance(loaded, tuple) else 'N/A'} != {n_experts}")
    loader_rows = loaded
    for idx, a in enumerate(loader_rows):
        if not isinstance(a, torch.Tensor):
            raise AssertionError(f"{packed_qname} expert {idx}: loader row not Tensor")
        if a.ndim != 2:
            raise AssertionError(f"{packed_qname} expert {idx}: rank {a.ndim} != 2")
    # Determine expected width from first nonempty row (never use expert 0 as oracle).
    # If weight in_features is authoritative, it will be validated after materialization;
    # here we only require mutual consistency among observed rows.
    _first_nonempty = next((x for x in loader_rows if x.numel() > 0 and x.ndim == 2 and int(x.shape[0]) > 0), None)
    if _first_nonempty is not None:
        _expected_w = int(_first_nonempty.shape[1])
        for idx, a in enumerate(loader_rows):
            if a.numel() > 0 and int(a.shape[1]) != _expected_w:
                raise AssertionError(f"{packed_qname} expert {idx}: width {a.shape[1]} != expected {_expected_w} (first nonempty)")
    # else all cold -> no width to validate; gate will stay raw
    # Ownership: retain original unfilled rows as authoritative; cold derived
    # directly from empties — no pooled allocation. Eligible-only gate needs no
    # pooled prior; cold rows remain exactly raw.
    original_rows = tuple(loader_rows)
    activation_rows = original_rows
    cold = tuple(i for i, v in enumerate(original_rows) if v.numel() == 0 or (v.ndim == 2 and int(v.shape[0]) == 0))
    # Build qnames per leaf from expected map (no DSV4 construction)
    qnames_per_leaf: dict[str, list[str]] = {p: [] for p in projections}
    for proj in projections:
        qnames_per_leaf[proj] = [expected_members[(proj, eid)] for eid in expert_ids]
    # Validate col-weights for every expected base qname
    for proj in projections:
        for qname in qnames_per_leaf[proj]:
            cw = torch.as_tensor(all_col_weights[qname]).to(torch.float32).contiguous()
            if list(cw.shape) != list(identity["col_weights_shapes"][qname]):  # type: ignore[index]
                raise AssertionError(f"{qname}: col-weight shape mismatch")
            if content_sha256_float32(cw) != identity["col_weights_content_sha256"][qname]:  # type: ignore[index]
                raise AssertionError(f"{qname}: col-weight digest mismatch")
    # Open source only through public opener
    skel = open_packed_weight_source(str(SOURCE))
    # Materialize via public single materializer, validating every leaf
    leaf_dims: dict[str, int] = {}
    weights: list[torch.Tensor] = []
    observed_files = 0
    # For pooling cross-check, collect per-expert leaf cws
    per_expert_leaf_cws: list[list[torch.Tensor]] = []
    for eid in expert_ids:
        leaf_cws: list[torch.Tensor] = []
        for proj in projections:
            qn = expected_members[(proj, eid)]
            leaf_cws.append(torch.as_tensor(all_col_weights[qn]).to(torch.float32))
        per_expert_leaf_cws.append(leaf_cws)
    for idx, eid in enumerate(expert_ids):
        # on_member validates checkpoint_base against authoritative group (no heuristic) and logical_qname
        def _make_on_member(cur_eid: int):
            def _on_member(proj: str, inner_eid: int, checkpoint_base: str, logical_qname: str, decoded: torch.Tensor):
                # Validate checkpoint_base against authoritative checkpoint-native group mapping
                exp_ckpt = group[proj][inner_eid]
                if checkpoint_base != exp_ckpt:
                    raise AssertionError(f"{packed_qname} expert {inner_eid} proj {proj}: checkpoint_base {checkpoint_base!r} != authoritative group {exp_ckpt!r}")
                # Validate logical_qname matches authoritative logical member map
                exp_logical = expected_members[(proj, inner_eid)]
                if logical_qname != exp_logical:
                    raise AssertionError(f"{packed_qname} expert {inner_eid} proj {proj}: logical_qname {logical_qname!r} != expected {exp_logical!r}")
                # Validate source weight using logical_qname (recipe identity)
                validate_cb_render_source_weight(identity, logical_qname, decoded, where=f"DSV4 layer {layer} source")
                d = int(decoded.shape[0])
                if proj not in leaf_dims:
                    leaf_dims[proj] = d
                elif leaf_dims[proj] != d:
                    raise AssertionError(f"{packed_qname} leaf {proj} dim {d} != prior {leaf_dims[proj]}")
            return _on_member
        fused = get_expert_weight(skel, profile, prefix, packed_proj, group, eid, logical_members=expected_members, on_member=_make_on_member(eid))
        weights.append(fused)
        x = loader_rows[idx]
        if x.numel() > 0 and x.shape[0] > 0:
            observed_files += 1
            pooled = torch.stack(per_expert_leaf_cws[idx]).mean(dim=0) if len(per_expert_leaf_cws[idx]) > 1 else per_expert_leaf_cws[idx][0]
            derived = x.square().mean(dim=0)
            if not torch.allclose(derived, pooled, rtol=1e-6, atol=1e-8):
                delta = float((derived - pooled).abs().max().item())
                if delta > 1e-3:
                    raise AssertionError(f"{packed_qname} expert {eid}: activation/pooled col-weight mismatch max_abs={delta}")
    weight_stack_cpu = torch.stack(weights).contiguous()
    # Build slice boundaries from recorded leaf dims (preserve profile order)
    leaf_out_dims = [leaf_dims[p] for p in projections]
    slice_boundaries: dict[str, tuple[int, int]] = {}
    off = 0
    for proj, dim in zip(projections, leaf_out_dims):
        slice_boundaries[proj] = (off, off + dim)
        off += dim
    if int(weight_stack_cpu.shape[1]) != off:
        raise AssertionError(f"{packed_qname}: fused width {weight_stack_cpu.shape[1]} != sum leaf dims {off}")
    # Single fused transfer/allocation to BF16 (avoids transient FP32 GPU stack)
    # weight_stack_cpu is explicitly contiguous; transfer is one allocation, no trailing contiguous() copy.
    weight_stack = weight_stack_cpu.to(device=device, dtype=torch.bfloat16)
    if not weight_stack.is_contiguous():
        raise AssertionError("fused weight transfer result not contiguous — fail closed rather than copying")
    del weight_stack_cpu
    # Authoritative leaf col-weights (E,1,C) per leaf — the ONLY encoder authority for multi-member parents.
    # A pooled diagnostic may exist but is never used for encoding/cost/export identity.
    leaf_col_weights: dict[str, torch.Tensor] = {}
    for proj in projections:
        leaf_names = qnames_per_leaf[proj]
        leaf_cws = [torch.as_tensor(all_col_weights[q]).to(torch.float32).contiguous() for q in leaf_names]
        leaf_stack = torch.stack(leaf_cws).unsqueeze(1).to(device).contiguous()
        if not leaf_stack.is_contiguous():
            raise AssertionError(f"{packed_qname} leaf {proj} col-weights transfer not contiguous")
        leaf_col_weights[proj] = leaf_stack
    # Pooled diagnostic (non-authoritative) via public helper — single transfer, assert contiguous, no extra copy
    pooled_map = get_packed_expert_col_weights(all_col_weights, {packed_qname: expected_members}, profile)
    pooled = pooled_map[packed_qname]
    col_stack_diag = pooled.to(device)
    if not col_stack_diag.is_contiguous():
        raise AssertionError("pooled col-weights transfer not contiguous — fail closed")
    # Cross-check manual mean may only assert equality
    _manual_pooled = torch.stack([torch.stack(lc).mean(dim=0) if len(lc) > 1 else lc[0] for lc in per_expert_leaf_cws]).unsqueeze(1).to(device)
    if not torch.allclose(col_stack_diag.cpu().float(), _manual_pooled.cpu().float(), rtol=1e-6, atol=1e-8):
        raise AssertionError(f"{packed_qname}: public pooling result drift vs manual mean")
    return {
        "packed_proj": packed_proj,
        "projections": tuple(projections),
        "qnames_per_leaf": qnames_per_leaf,
        "weight": weight_stack,
        "leaf_col_weights": dict(leaf_col_weights),
        "col_weights": col_stack_diag,
        "col_weights_pooled_diag": col_stack_diag,
        "col_weights_pooled_diagnostic": col_stack_diag,
        "activation_rows": activation_rows,
        "activation_rows_original": original_rows,
        "observed_activation_files": observed_files,
        "cold_experts": list(cold),
        "loader_used": True,
        "slice_boundaries": dict(slice_boundaries),
        "leaf_out_dims": list(leaf_out_dims),
        "col_weight_pooling": "mean_of_member_vectors",
        "col_weight_pooling_diagnostic": "mean_of_member_vectors",
        "member_order": list(projections),
    }


# Backward compat shim: old entry point — now fail-closed, only known leaf/parent via profile.
def load_projection(
    layer: int,
    projection: str,
    *,
    device: torch.device,
    identity: Mapping[str, Any],
    all_col_weights: Mapping[str, Any],
    model_to_shard: Mapping[str, str],
    model_to_ckpt: Mapping[str, str],
    scale_map: Mapping[str, Any],
) -> dict[str, Any]:
    profile, _, _, packed_names = _get_packed_planning()
    if projection in packed_names:
        return load_packed_projection(
            layer, projection, device=device, identity=identity, all_col_weights=all_col_weights, model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt, scale_map=scale_map
        )
    for packed_proj in packed_names:
        projs = get_packed_expert_projection_names(profile, packed_proj)
        if projection in projs:
            packed_data = load_packed_projection(
                layer, packed_proj, device=device, identity=identity, all_col_weights=all_col_weights, model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt, scale_map=scale_map
            )
            start, end = packed_data["slice_boundaries"][projection]
            weight_leaf = packed_data["weight"][:, start:end, :].contiguous()
            # Use authoritative qnames for leaf col stack (no DSV4 construction)
            leaf_qnames = packed_data["qnames_per_leaf"][projection]
            leaf_cws = [torch.as_tensor(all_col_weights[q]).to(torch.float32).contiguous() for q in leaf_qnames]
            col_stack_leaf = torch.stack(leaf_cws).unsqueeze(1).to(device).contiguous()
            return {
                "qnames": leaf_qnames,
                "weight": weight_leaf,
                "col_weights": col_stack_leaf,
                "activation_rows": packed_data["activation_rows"],
                "observed_activation_files": packed_data["observed_activation_files"],
                "cold_experts": packed_data["cold_experts"],
                "loader_used": True,
                "packed_parent": packed_proj,
                "slice": (start, end),
            }
    raise AssertionError(f"projection {projection!r} not in authoritative packed set {sorted(packed_names)} nor in any leaf set")

def per_slice_mse(weight: torch.Tensor, recon: torch.Tensor) -> list[float]:
    return (weight - recon).float().square().mean(dim=(1, 2)).detach().cpu().tolist()

def per_slice_weighted_mse(weight: torch.Tensor, recon: torch.Tensor, col_weights: torch.Tensor) -> list[float]:
    err2 = (weight - recon).float().square()
    cw = torch.broadcast_to(col_weights.to(err2), err2.shape)
    # Parenthesize full quotient before detach/cpu/tolist — operator precedence fix.
    return ((err2 * cw).sum(dim=(1, 2)) / cw.sum(dim=(1, 2)).clamp_min(1e-30)).detach().cpu().tolist()

def per_slice_output_mse(weight: torch.Tensor, recon: torch.Tensor, act_rows: tuple[torch.Tensor, ...]) -> list[float]:
    # After fill_empty, every expert has rows (pooled); still handle empty as inf for fail-closed diagnostics.
    out: list[float] = []
    for idx in range(int(weight.shape[0])):
        act = act_rows[idx]
        if act is None or act.numel() == 0 or act.shape[0] == 0:
            out.append(float("inf"))
            continue
        w = weight[idx].to(torch.float32)
        r = recon[idx].to(torch.float32) if recon.ndim == 3 else recon.to(torch.float32)
        err = (act.to(w.device, torch.float32) @ (w - r).T).pow(2).mean().item()
        out.append(float(err))
    return out

def per_slice_rel_output_mse(weight: torch.Tensor, recon: torch.Tensor, act_rows: tuple[torch.Tensor, ...], output_mse: list[float]) -> list[float]:
    rel: list[float] = []
    for idx in range(int(weight.shape[0])):
        act = act_rows[idx]
        if act is None or act.numel() == 0:
            rel.append(float("inf"))
            continue
        w = weight[idx].to(torch.float32)
        ref_energy = float((act.to(w.device, torch.float32) @ w.T).pow(2).mean().item())
        rel.append(output_mse[idx] / max(ref_energy, 1e-12))
    return rel

# ---------------------------------------------------------------------------
# Compact gate helper: extract exact indexed expert decision, scalar
# raw/LDLQ MSE, reason, missing status. Full gate_info stays only in
# projection checkpoint; per-row stores compact dict.
# ---------------------------------------------------------------------------

def compact_gate_for_expert(gate_info: Mapping[str, Any], expert_idx: int, leaf: str | None = None) -> dict[str, Any]:
    """Extract compact per-expert decision. When leaf is supplied, uses exact per_leaf_kept[leaf][expert]."""
    gate = str(gate_info.get("gate", "unknown"))
    out: dict[str, Any] = {"gate": gate}
    # Leaf-local path: per_leaf_kept is authoritative, per_expert_kept is AND across leaves (non-authoritative for pricing)
    if leaf is not None and "per_leaf_kept" in gate_info:
        per_leaf = gate_info["per_leaf_kept"]
        if not isinstance(per_leaf, Mapping) or leaf not in per_leaf:
            raise AssertionError(f"per_leaf_kept missing leaf {leaf!r}")
        kept_list = per_leaf[leaf]
        try:
            kept = bool(kept_list[expert_idx])
        except (IndexError, TypeError, KeyError) as exc:
            raise AssertionError(f"per_leaf_kept[{leaf!r}] malformed for expert {expert_idx}: {exc}") from exc
        out["kept_ldlq"] = kept
        # Use per-leaf raw/ldlq vectors when available
        raw_per_leaf = gate_info.get("raw_mse_per_expert_per_leaf", {})
        ldlq_per_leaf = gate_info.get("ldlq_mse_per_expert_per_leaf", {})
        if isinstance(raw_per_leaf, Mapping) and leaf in raw_per_leaf and isinstance(ldlq_per_leaf, Mapping) and leaf in ldlq_per_leaf:
            try:
                out["raw_mse"] = float(raw_per_leaf[leaf][expert_idx])
                out["ldlq_mse"] = float(ldlq_per_leaf[leaf][expert_idx])
            except (IndexError, TypeError, ValueError) as exc:
                raise AssertionError(f"per-leaf mse lists malformed for leaf {leaf!r} expert {expert_idx}: {exc}") from exc
        else:
            # Fallback to fused vectors if per-leaf not present (single-leaf parents)
            raw_list = gate_info.get("raw_mse_per_expert")
            ldlq_list = gate_info.get("ldlq_mse_per_expert")
            if raw_list is not None and ldlq_list is not None:
                try:
                    out["raw_mse"] = float(raw_list[expert_idx])
                    out["ldlq_mse"] = float(ldlq_list[expert_idx])
                except (IndexError, TypeError, ValueError) as exc:
                    raise AssertionError(f"mse lists malformed for expert {expert_idx}: {exc}") from exc
        if "missing_experts" in gate_info:
            me = gate_info["missing_experts"]
            if not isinstance(me, (list, tuple, set)):
                raise AssertionError("missing_experts not list/tuple/set")
            if expert_idx in set(me):
                out["missing_activation"] = True
        # Stamp leaf identity and basis
        out["leaf"] = str(leaf)
        if "reason" in gate_info:
            out["reason"] = str(gate_info["reason"])
        if "metric" in gate_info:
            out["metric"] = str(gate_info["metric"])
        else:
            out["metric"] = "activation_output_mse"
        return out
    if "per_expert_kept" in gate_info:
        kept_list = gate_info["per_expert_kept"]
        try:
            kept = bool(kept_list[expert_idx])
        except (IndexError, TypeError, KeyError) as exc:
            raise AssertionError(f"per_expert_kept malformed for expert {expert_idx}: {exc}") from exc
        out["kept_ldlq"] = kept
        raw_list = gate_info.get("raw_mse_per_expert")
        ldlq_list = gate_info.get("ldlq_mse_per_expert")
        if raw_list is not None and ldlq_list is not None:
            try:
                out["raw_mse"] = float(raw_list[expert_idx])
                out["ldlq_mse"] = float(ldlq_list[expert_idx])
            except (IndexError, TypeError, ValueError) as exc:
                raise AssertionError(f"mse lists malformed for expert {expert_idx}: {exc}") from exc
        elif raw_list is not None or ldlq_list is not None:
            raise AssertionError("per_expert gate_info has mismatched mse lists")
        if "missing_experts" in gate_info:
            me = gate_info["missing_experts"]
            if not isinstance(me, (list, tuple, set)):
                raise AssertionError("missing_experts not list/tuple/set")
            if expert_idx in set(me):
                out["missing_activation"] = True
        if "reason" in gate_info:
            out["reason"] = str(gate_info["reason"])
        if "metric" in gate_info:
            out["metric"] = str(gate_info["metric"])
        else:
            out["metric"] = "activation_output_mse"
        return out
    if "kept_ldlq" in gate_info:
        out["kept_ldlq"] = bool(gate_info["kept_ldlq"])
        if "raw_mse" in gate_info or "ldlq_mse" in gate_info:
            if "raw_mse" not in gate_info or "ldlq_mse" not in gate_info:
                raise AssertionError("kept_ldlq gate missing paired mse fields")
            try:
                out["raw_mse"] = float(gate_info["raw_mse"])
                out["ldlq_mse"] = float(gate_info["ldlq_mse"])
            except (TypeError, ValueError) as exc:
                raise AssertionError(f"mse fields malformed: {exc}") from exc
        if "reason" in gate_info:
            out["reason"] = str(gate_info["reason"])
        if "metric" in gate_info:
            out["metric"] = str(gate_info["metric"])
        else:
            out["metric"] = "activation_output_mse"
        if gate.startswith("raw_fallback"):
            out["missing_activation"] = True
        return out
    if "reason" in gate_info:
        out["reason"] = str(gate_info["reason"])
    if "metric" in gate_info:
        out["metric"] = str(gate_info["metric"])
    return out

def compact_gate_for_dense(gate_info: Mapping[str, Any]) -> dict[str, Any]:
    """Compact gate for dense (non-packed) — single decision. Fail-closed on malformed."""
    gate = str(gate_info.get("gate", "unknown"))
    out: dict[str, Any] = {"gate": gate}
    if "kept_ldlq" in gate_info:
        out["kept_ldlq"] = bool(gate_info["kept_ldlq"])
    if "raw_mse" in gate_info or "ldlq_mse" in gate_info:
        if "raw_mse" not in gate_info or "ldlq_mse" not in gate_info:
            raise AssertionError("dense gate missing paired mse fields")
        try:
            out["raw_mse"] = float(gate_info["raw_mse"])
            out["ldlq_mse"] = float(gate_info["ldlq_mse"])
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"dense mse fields malformed: {exc}") from exc
    if "reason" in gate_info:
        out["reason"] = str(gate_info["reason"])
    if "metric" in gate_info:
        out["metric"] = str(gate_info["metric"])
    else:
        out["metric"] = "activation_output_mse"
    if gate.startswith("raw_fallback"):
        out["missing_activation"] = True
    return out


# ---------------------------------------------------------------------------
# Content-keyed atomic projection-rung checkpoints (fail-closed, no catch-to-raw)
# ---------------------------------------------------------------------------

def _packed_activation_evidence_identity(
    activation_rows: Sequence[torch.Tensor],
    act_root: Path,
    qname_prefix: str,
    *,
    member_order: Sequence[str] | None = None,
    slice_boundaries: Mapping[str, tuple[int, int]] | None = None,
    col_weight_pooling: str | None = None,
    cold_experts: Sequence[int] | None = None,
    split_infos: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic activation evidence for packed projections — full content of all 256 tensors.

    Includes member order, slice boundaries, col-weight pooling contract,
    cold indices, split policy/version, per-expert digests and fresh_text flag
    so checkpoint identity, content keys, and telemetry all bind the exact
    evidence and split. Uses original unfilled rows as authoritative.
    """
    from prismaquant.nvfp4_cb_formats import _SPLIT_POLICY, _SPLIT_VERSION

    h = hashlib.sha256()
    h.update(str(act_root.resolve()).encode())
    h.update(b"|")
    h.update(qname_prefix.encode())
    h.update(b"|")
    h.update(b"loader=prismaquant.cb_ldlq.CBLDLQActivationLoader|")
    h.update(b"pooling=none_eligible_only_no_pooled_prior|")
    h.update(f"fused_activation_policy={FUSED_ACTIVATION_POLICY_V1}|".encode())
    h.update(f"split_policy={_SPLIT_POLICY}|split_version={_SPLIT_VERSION}|fresh_text=false|".encode())
    if cold_experts is not None:
        h.update(f"cold_experts={sorted(cold_experts)}|".encode())
    if member_order is not None:
        h.update(f"member_order={list(member_order)}|".encode())
    if slice_boundaries is not None:
        ordered = {k: slice_boundaries[k] for k in (member_order or sorted(slice_boundaries))}
        h.update(f"slice_boundaries={ordered}|".encode())
    if col_weight_pooling is not None:
        h.update(f"col_weight_pooling={col_weight_pooling}|".encode())
    h.update(b"col_weight_pooling_contract=mean_of_member_vectors|")
    if split_infos is not None:
        # Include per-expert split digests deterministically
        h.update(json.dumps(list(split_infos), sort_keys=True, separators=(",", ":")).encode())
        h.update(b"|")
    row_counts: list[int] = []
    for idx, a in enumerate(activation_rows):
        h.update(f"expert={idx}|".encode())
        if not isinstance(a, torch.Tensor):
            h.update(b"not_tensor|")
            row_counts.append(0)
            continue
        h.update(f"shape={tuple(a.shape)}|dtype={str(a.dtype)}|".encode())
        row_counts.append(int(a.shape[0]) if a.ndim == 2 else 0)
        if a.numel() > 0:
            b = a.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            h.update(f"bytes_len={len(b)}|".encode())
            h.update(b)
        else:
            h.update(b"empty|")
    out: dict[str, Any] = {
        "act_root": str(act_root.resolve()),
        "qname_prefix": str(qname_prefix),
        "loader_contract": "prismaquant.cb_ldlq.CBLDLQActivationLoader",
        "pooling": "none_eligible_only_no_pooled_prior",
        "fused_activation_policy": FUSED_ACTIVATION_POLICY_V1,
        "fused_activation_order": list(member_order) if member_order is not None else None,
        "row_counts": list(row_counts),
        "evidence_sha256": h.hexdigest(),
        "gate": "activation_output_mse",
        "col_weight_pooling": col_weight_pooling or "mean_of_member_vectors",
        "split_policy": _SPLIT_POLICY,
        "split_version": _SPLIT_VERSION,
        "fresh_text": False,
    }
    if cold_experts is not None:
        out["cold_experts"] = sorted(cold_experts)
        out["cold_indices"] = sorted(cold_experts)
    if split_infos is not None:
        out["split_infos"] = [dict(s) for s in split_infos]
        # Also store aggregated digests for quick provenance
        out["fit_digests"] = [s.get("fit_digest") for s in split_infos if isinstance(s, dict)]
        out["holdout_digests"] = [s.get("holdout_digest") for s in split_infos if isinstance(s, dict)]
    if member_order is not None:
        out["member_order"] = list(member_order)
    if slice_boundaries is not None:
        out["slice_boundaries"] = {k: list(v) for k, v in slice_boundaries.items()}
    return out

def projection_checkpoint_identity(
    layer: int, projection: str, rung: int, *, by_layer_sha256: str, col_weights_sha256: str, source_index_sha256: str, context_stamp: Mapping[str, Any], tool_sha256: str, activation_evidence: Mapping[str, Any] | None = None, member_order: Sequence[str] | None = None, slice_boundaries: Mapping[str, tuple[int, int]] | None = None, col_weight_pooling: str | None = None, expert_count: int,
) -> dict[str, Any]:
    """Checkpoint identity for a packed parent (gate_up_proj) or single (down_proj).

    Uses the single-sourced split policy from nvfp4_cb_formats and stamps
    fresh_text=false. Includes cold/split digests via activation_evidence so
    content keys and resume validation change on any evidence/split mutation.
    """
    from prismaquant.nvfp4_cb_formats import _SPLIT_POLICY, _SPLIT_VERSION

    if not isinstance(expert_count, int) or expert_count <= 0:
        raise ValueError(f"projection_checkpoint_identity: expert_count must be positive int, got {expert_count!r}")
    base: dict[str, Any] = {
        "schema": PROJECTION_RUNG_SCHEMA,
        "layer": int(layer),
        "projection": str(projection),
        "rung": int(rung),
        "format": f"NVFP4_CB_K{rung}",
        "by_layer_sha256": str(by_layer_sha256),
        "col_weights_sha256": str(col_weights_sha256),
        "source_index_sha256": str(source_index_sha256),
        "context_stamp": dict(context_stamp),
        "tool_sha256": str(tool_sha256),
        "packed_parent": str(projection),
        "expert_count": int(expert_count),
        "fused_activation_policy": FUSED_ACTIVATION_POLICY_V1,
        "fused_activation_order": list(member_order) if member_order is not None else None,
        "split_policy": _SPLIT_POLICY,
        "split_version": _SPLIT_VERSION,
        "fresh_text": False,
    }
    if member_order is not None:
        base["member_order"] = list(member_order)
    if slice_boundaries is not None:
        base["slice_boundaries"] = {k: list(v) for k, v in slice_boundaries.items()}
    if col_weight_pooling is not None:
        base["col_weight_pooling"] = str(col_weight_pooling)
    else:
        base["col_weight_pooling"] = "mean_of_member_vectors"
    if activation_evidence is not None:
        base["activation_evidence"] = dict(activation_evidence)
    return base

def validated_projection_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise AssertionError(f"stale or corrupt projection checkpoint {path}: unreadable ({type(exc).__name__}: {exc})") from exc
    for key in ("schema", "content_key", "identity", "result"):
        if key not in payload:
            raise AssertionError(f"stale checkpoint {path}: missing {key!r}")
    if payload.get("schema") != PROJECTION_RUNG_SCHEMA:
        raise AssertionError(f"stale checkpoint {path}: schema mismatch {payload.get('schema')!r} expected {PROJECTION_RUNG_SCHEMA!r}")
    if payload.get("identity") != dict(expected_identity):
        raise AssertionError(f"stale checkpoint {path}: identity mismatch")
    if payload.get("content_key") != _sha(expected_identity):
        raise AssertionError(f"stale checkpoint {path}: content_key mismatch")
    if "expert_count" not in expected_identity:
        raise AssertionError(f"stale checkpoint {path}: identity missing expert_count (stale schema)")
    # Fused activation policy must be stamped and equal v1 (bumped v2->v3)
    if expected_identity.get("fused_activation_policy") != FUSED_ACTIVATION_POLICY_V1:
        raise AssertionError(f"stale checkpoint {path}: identity missing or mismatched fused_activation_policy {expected_identity.get('fused_activation_policy')!r} expected {FUSED_ACTIVATION_POLICY_V1!r} (v2->v3 bump)")
    if payload.get("identity", {}).get("fused_activation_policy") != FUSED_ACTIVATION_POLICY_V1:
        raise AssertionError(f"stale checkpoint {path}: payload identity fused_activation_policy missing/mismatch")
    try:
        expert_count = int(expected_identity["expert_count"])
    except Exception as exc:
        raise AssertionError(f"stale checkpoint {path}: expert_count not int {exc}") from exc
    if expert_count <= 0:
        raise AssertionError(f"stale checkpoint {path}: expert_count must be positive, got {expert_count}")
    # Activation evidence must also carry fused policy when present
    act_ev = expected_identity.get("activation_evidence")
    if isinstance(act_ev, dict) and act_ev.get("fused_activation_policy") != FUSED_ACTIVATION_POLICY_V1:
        raise AssertionError(f"stale checkpoint {path}: activation_evidence fused_activation_policy missing/mismatch")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise AssertionError(f"stale checkpoint {path}: missing result mapping")
    if result.get("format") != f"NVFP4_CB_K{expected_identity['rung']}":
        raise AssertionError(f"stale checkpoint {path}: rung mismatch {result.get('format')} vs K{expected_identity['rung']}")
    # member_order is required identity field for strict validation
    expected_member_order = expected_identity.get("member_order")
    if not isinstance(expected_member_order, list) or len(expected_member_order) == 0:
        raise AssertionError(f"stale checkpoint {path}: identity member_order missing or not non-empty list {expected_member_order!r}")
    # result must carry member_order and match identity
    result_member_order = result.get("member_order")
    if not isinstance(result_member_order, list):
        raise AssertionError(f"stale checkpoint {path}: result member_order missing or not list")
    if result_member_order != list(expected_member_order):
        raise AssertionError(f"stale checkpoint {path}: result member_order {result_member_order} != identity {expected_member_order}")
    member_order = list(expected_member_order)
    # result expert_count must match and be int
    result_expert_count = result.get("expert_count")
    if not isinstance(result_expert_count, int) or int(result_expert_count) != expert_count:
        raise AssertionError(f"stale checkpoint {path}: result expert_count {result.get('expert_count')} != identity {expert_count}")
    # result-level packed metric/gate vectors must exist, be list, exact length
    for k in ("weight_mse_per_expert", "weighted_mse_per_expert", "output_mse_per_expert", "rel_output_mse_per_expert", "n_activation_rows_per_expert", "weight_mse_fused_per_expert", "output_mse_fused_per_expert"):
        v = result.get(k)
        if not isinstance(v, list):
            raise AssertionError(f"stale checkpoint {path}: result {k} missing or not list")
        if len(v) != expert_count:
            raise AssertionError(f"stale checkpoint {path}: result {k} length {len(v)} != expert_count {expert_count}")
    # qnames flat must exist, be list, total count exactly expert_count*len(member_order)
    qnames_flat = result.get("qnames")
    if not isinstance(qnames_flat, list):
        raise AssertionError(f"stale checkpoint {path}: result qnames missing or not list")
    expected_total = expert_count * len(member_order)
    if len(qnames_flat) != expected_total:
        raise AssertionError(f"stale checkpoint {path}: qnames length {len(qnames_flat)} != expected total {expected_total} (=expert_count*len(member_order))")
    # per_leaf and qnames_per_leaf must exist as mappings, keys exactly member_order, reject extras/missing
    per_leaf = result.get("per_leaf")
    if not isinstance(per_leaf, Mapping):
        raise AssertionError(f"stale checkpoint {path}: per_leaf missing or not mapping (required)")
    if set(per_leaf.keys()) != set(member_order):
        raise AssertionError(f"stale checkpoint {path}: per_leaf keys {sorted(per_leaf.keys())} != member_order {sorted(member_order)} (extras/missing rejected)")
    qnames_per_leaf = result.get("qnames_per_leaf")
    if not isinstance(qnames_per_leaf, Mapping):
        raise AssertionError(f"stale checkpoint {path}: qnames_per_leaf missing or not mapping (required)")
    if set(qnames_per_leaf.keys()) != set(member_order):
        raise AssertionError(f"stale checkpoint {path}: qnames_per_leaf keys {sorted(qnames_per_leaf.keys())} != member_order {sorted(member_order)}")
    # per_leaf metric vectors strict
    required_leaf_metrics = ("weight_mse_per_expert", "output_mse_per_expert", "rel_output_mse_per_expert")
    for leaf_proj, leaf_data in per_leaf.items():
        if not isinstance(leaf_data, Mapping):
            raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} not mapping")
        for k in required_leaf_metrics:
            if k not in leaf_data:
                raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} missing required {k}")
            v = leaf_data.get(k)
            if not isinstance(v, list):
                raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} {k} not list")
            if len(v) != expert_count:
                raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} {k} length {len(v)} != expert_count {expert_count}")
        # reject extra metric keys beyond required + weighted if present? Any extra still checked for length but we enforce no missing; extras are allowed only if they are also list length exact? To enforce reject extras strictly, ensure no extra keys beyond the required set plus weighted?
        # For strictness, reject any leaf_data key that is not in required set plus weighted
        allowed_leaf_metrics = set(required_leaf_metrics) | {"weighted_mse_per_expert"}
        for k in leaf_data.keys():
            if k not in allowed_leaf_metrics:
                # Still require it be list length expert_count if present, but flag extras as stale
                raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} unexpected extra metric {k!r}")
        # weighted is required if present in required set? Check presence
        if "weighted_mse_per_expert" not in leaf_data:
            raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} missing weighted_mse_per_expert")
        v = leaf_data.get("weighted_mse_per_expert")
        if not isinstance(v, list) or len(v) != expert_count:
            raise AssertionError(f"stale checkpoint {path}: per_leaf {leaf_proj} weighted_mse_per_expert not list len {expert_count}")
    total_qnames = 0
    for leaf_proj, qns in qnames_per_leaf.items():
        if not isinstance(qns, list):
            raise AssertionError(f"stale checkpoint {path}: qnames_per_leaf {leaf_proj} not list")
        if len(qns) != expert_count:
            raise AssertionError(f"stale checkpoint {path}: qnames_per_leaf {leaf_proj} length {len(qns)} != expert_count {expert_count}")
        total_qnames += len(qns)
    expected_total_q = expert_count * len(member_order)
    if total_qnames != expected_total_q:
        raise AssertionError(f"stale checkpoint {path}: qnames_per_leaf total {total_qnames} != expected {expected_total_q}")
    # gate vectors: if gate_info present, validate per-expert vectors lengths
    gate_info = result.get("gate_info")
    if not isinstance(gate_info, Mapping):
        raise AssertionError(f"stale checkpoint {path}: gate_info missing or not mapping (required)")
    for gk in ("per_expert_kept", "raw_mse_per_expert", "ldlq_mse_per_expert"):
        if gk not in gate_info:
            raise AssertionError(f"stale checkpoint {path}: gate_info missing required {gk}")
        gv = gate_info[gk]
        if not isinstance(gv, list):
            raise AssertionError(f"stale checkpoint {path}: gate_info {gk} not list")
        if len(gv) != expert_count:
            raise AssertionError(f"stale checkpoint {path}: gate_info {gk} length {len(gv)} != expert_count {expert_count}")
    warm_path = result.get("warm_state_path")
    if warm_path and not Path(str(warm_path)).is_file():
        raise AssertionError(f"stale checkpoint {path}: warm state missing {warm_path!r}")
    return payload

def dense_checkpoint_identity(
    layer: int, qname: str, *, col_weights_content_sha256: str, activation_evidence_sha256: str, by_layer_sha256: str, col_weights_sha256: str, source_index_sha256: str, context_stamp: Mapping[str, Any], tool_sha256: str,
) -> dict[str, Any]:
    from prismaquant.nvfp4_cb_formats import _SPLIT_POLICY as _DPOL, _SPLIT_VERSION as _DVER

    return {
        "schema": DENSE_RUNG_SCHEMA,
        "layer": int(layer),
        "qname": str(qname),
        "rungs": list(NVFP4_RUNGS),
        "col_weights_content_sha256": str(col_weights_content_sha256),
        "activation_evidence_sha256": str(activation_evidence_sha256),
        "by_layer_sha256": str(by_layer_sha256),
        "col_weights_sha256": str(col_weights_sha256),
        "source_index_sha256": str(source_index_sha256),
        "context_stamp": dict(context_stamp),
        "tool_sha256": str(tool_sha256),
        "split_policy": _DPOL,
        "split_version": _DVER,
        "fresh_text": False,
    }

def validated_dense_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise AssertionError(f"stale or corrupt dense checkpoint {path}: unreadable ({type(exc).__name__}: {exc})") from exc
    for key in ("schema", "content_key", "identity", "results"):
        if key not in payload:
            raise AssertionError(f"stale dense checkpoint {path}: missing {key!r}")
    if payload.get("schema") != DENSE_RUNG_SCHEMA:
        raise AssertionError(f"stale dense checkpoint {path}: schema mismatch {payload.get('schema')!r}")
    if payload.get("identity") != dict(expected_identity):
        raise AssertionError(f"stale dense checkpoint {path}: identity mismatch")
    if payload.get("content_key") != _sha(expected_identity):
        raise AssertionError(f"stale dense checkpoint {path}: content_key mismatch")
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise AssertionError(f"stale dense checkpoint {path}: missing results")
    # must cover exactly NVFP4_RUNGS
    if set(int(k) for k in results.keys()) != set(NVFP4_RUNGS):
        raise AssertionError(f"stale dense checkpoint {path}: rung coverage mismatch {sorted(results.keys())}")
    return payload

def encode_nvfp4_rung_packed(
    layer: int, packed_proj: str, rung: int, data: Mapping[str, Any], device: torch.device, write_warm_state: bool = True, prepared_evidence=None,
) -> dict[str, Any]:
    """Encode one authoritative packed parent (gate_up_proj or down_proj) with per-leaf held-out gating.

    Performs fused fixed-codebook LDLQ on the authoritative E x fused_R x C tensor
    (eligible-only, batched/chunked) but makes an independent held-out
    activation_output_mse decision per leaf (gate_proj vs up_proj) and per expert.
    The fixed parent/codebook/scales remain shared; only disjoint output-row
    assignment/sign slices differ. Cold/insufficient experts are raw for both leaves.
    All metrics are measured on the immutable holdout split (never FIT rows).
    No full E×R×C reconstruction is ever materialized (chunked decode, bounded
    to PRISMAQUANT_CB_RECON_EXPERT_CHUNK).
    """
    require_cuda(device)
    format_name = f"NVFP4_CB_K{rung}"
    spec = fr.get_format(format_name)
    torch.cuda.synchronize()
    start = time.perf_counter()
    projections: tuple[str, ...] = tuple(data.get("projections", (packed_proj,)))
    slice_boundaries: dict[str, tuple[int, int]] = dict(data.get("slice_boundaries", {packed_proj: (0, int(data["weight"].shape[1]))}))
    if len(projections) > 1:
        leaf_dims = list(data.get("leaf_out_dims", []))
        if leaf_dims and len(set(leaf_dims)) != 1:
            raise AssertionError(f"{packed_proj}: leaf widths not equal {leaf_dims} — allocator additive grouping requires equal gate/up widths")
    E, R, C = map(int, data["weight"].shape)
    # Resolve prepared evidence for holdout-authentic gating (single truthful owner).
    # If caller supplied prepared_evidence, use it; else build once from activation_rows.
    from prismaquant.nvfp4_cb_formats import (
        prepare_ldlq_gate_evidence as _prep,
    )
    _prepared = prepared_evidence if prepared_evidence is not None else _prep(data["activation_rows"], qname=f"model.layers.{layer}.mlp.experts.{packed_proj}")
    # Validate prepared expert count matches weight stack
    if len(_prepared.original_rows) != E:
        raise AssertionError(f"{packed_proj}: prepared evidence expert count {len(_prepared.original_rows)} != weight stack {E}")
    holdout_rows: tuple[torch.Tensor, ...] = tuple(_prepared.holdout_rows)
    # --- Shared leaf-local encode/stitch via pure helper (single source) ---
    weight = data["weight"]
    leaf_col_weights_map = data.get("leaf_col_weights")
    if len(projections) > 1 and leaf_col_weights_map is None:
        raise AssertionError(
            f"{packed_proj}: multi-member parent requires leaf_col_weights keys {sorted(projections)}, "
            f"got None — pooled fallback deleted; caller must supply exact leaf maps"
        )
    if len(projections) > 1:
        if not isinstance(leaf_col_weights_map, dict) or set(leaf_col_weights_map.keys()) != set(projections):
            raise AssertionError(f"{packed_proj}: multi-member parent requires leaf_col_weights keys {sorted(projections)}, got {sorted(leaf_col_weights_map.keys()) if isinstance(leaf_col_weights_map, dict) else type(leaf_col_weights_map)}")
        leaf_cw_for_helper = dict(leaf_col_weights_map)
    else:
        # Single-leaf: use leaf map if present, else pooled col_weights as leaf entry (preserves byte identity)
        if isinstance(leaf_col_weights_map, dict) and projections[0] in leaf_col_weights_map:
            leaf_cw_for_helper = {projections[0]: leaf_col_weights_map[projections[0]]}
        else:
            # Fallback to pooled for single-leaf backward compat (no multi-member fallback)
            cw_pooled = data.get("col_weights")
            if cw_pooled is None:
                raise AssertionError(f"{packed_proj}: single-leaf requires col_weights")
            # Pooled shape may be (E,1,C) already; use as leaf
            leaf_cw_for_helper = {projections[0]: cw_pooled}
    # Delegate to shared pure helper — final fields/gate must come from there (no second implementation)
    final_fields, gate_info = encode_packed_parent_leaf_local(
        weight, rung, grid="fp4", mode="product",
        member_order=list(projections),
        slice_boundaries=dict(slice_boundaries),
        leaf_col_weights=leaf_cw_for_helper,
        prepared=_prepared,
    )
    per_leaf_kept = gate_info.get("per_leaf_kept", {})
    holdout_rows = tuple(_prepared.holdout_rows)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Final payload metrics on holdout (chunked) — compute weight/weighted/output/rel for final stitched fields
    weight_mse_fused: list[float] = [0.0] * E
    weighted_mse_fused: list[float] = [0.0] * E
    output_mse_fused: list[float] = [0.0] * E
    rel_output_mse_fused: list[float] = [0.0] * E
    per_leaf_acc_final: dict[str, dict[str, list[float]]] = {
        proj: {
            "weight_mse_per_expert": [0.0] * E,
            "weighted_mse_per_expert": [0.0] * E,
            "output_mse_per_expert": [0.0] * E,
            "rel_output_mse_per_expert": [0.0] * E,
        }
        for proj in projections
    }
    weight_dtype = weight.dtype
    for (first, last), chunk_recon, _chunk_fields in iter_nvfp4_cb_recon_chunks(final_fields, rung, grid="fp4", mode="product"):
        chunk_recon_bf16 = chunk_recon.to(weight_dtype)
        del chunk_recon
        for local in range(last - first):
            gidx = first + local
            w_bf16 = weight[gidx]
            r_bf16 = chunk_recon_bf16[local]
            diff_bf16 = w_bf16 - r_bf16
            diff_f32 = diff_bf16.float()
            weight_mse_fused[gidx] = float(diff_f32.pow(2).mean().item())
            # For multi-leaf, fused weighted uses stitched leaf colweights: compute as mean of leaf weighted (equal leaf widths) for consistency
            if len(projections) > 1 and isinstance(leaf_col_weights_map, dict):
                # Leaf-local weighted: average of per-leaf weighted MSEs (equal gate/up widths validated)
                # Compute after per-leaf loop for exactness; placeholder weighted fused as mean of leaf weighted later
                cw_raw = None
                cw_b = None
                denom = None
                weighted_mse_fused[gidx] = float("nan")  # placeholder, filled after per-leaf loop
            else:
                cw_raw = leaf_cw_for_helper[projections[0]][gidx]
                cw_b = torch.broadcast_to(cw_raw.to(w_bf16.device, torch.float32), w_bf16.shape)
                denom = float(cw_b.sum().clamp_min(1e-30).item())
                weighted_mse_fused[gidx] = float(((diff_f32.pow(2) * cw_b).sum().item() / max(denom, 1e-30)))
            # output on holdout
            act_h = holdout_rows[gidx]
            if act_h is None or (isinstance(act_h, torch.Tensor) and (act_h.numel() == 0 or int(act_h.shape[0]) == 0)):
                output_mse_fused[gidx] = float("inf")
                rel_output_mse_fused[gidx] = float("inf")
            else:
                act_f = torch.as_tensor(act_h).to(w_bf16.device, torch.float32)
                try:
                    out_mse = canonical_nvfp4_cb_single_output_mse(w_bf16, r_bf16, act_f, spec)
                except ValueError:
                    out_mse = float("inf")
                output_mse_fused[gidx] = out_mse
                if out_mse == float("inf"):
                    rel_output_mse_fused[gidx] = float("inf")
                else:
                    w_f32 = w_bf16.to(torch.float32)
                    ref_energy = float((act_f.to(w_f32.device) @ w_f32.T).pow(2).mean().item())
                    rel_output_mse_fused[gidx] = out_mse / max(ref_energy, 1e-12)
            # Compute per-leaf metrics with authoritative leaf colweights
            leaf_weighted_vals = []
            leaf_weighted_denom = []
            leaf_weighted_numer = []
            for proj in projections:
                s, e = slice_boundaries[proj]
                w_leaf_bf16 = w_bf16[s:e, :].contiguous()
                r_leaf_bf16 = r_bf16[s:e, :].contiguous()
                diff_leaf = (w_leaf_bf16 - r_leaf_bf16).float()
                per_leaf_acc_final[proj]["weight_mse_per_expert"][gidx] = float(diff_leaf.pow(2).mean().item())
                # Use leaf-colweights when available (multi-leaf), else single-leaf authority
                if len(projections) > 1 and isinstance(leaf_col_weights_map, dict):
                    cw_leaf_src = leaf_col_weights_map[proj][gidx]
                else:
                    cw_leaf_src = leaf_cw_for_helper[projections[0]][gidx]
                cw_leaf_b = torch.broadcast_to(cw_leaf_src.to(w_leaf_bf16.device, torch.float32), w_leaf_bf16.shape)
                denom_leaf = float(cw_leaf_b.sum().clamp_min(1e-30).item())
                numer_leaf = float((diff_leaf.pow(2) * cw_leaf_b).sum().item())
                w_leaf_weighted = float(numer_leaf / max(denom_leaf, 1e-30))
                per_leaf_acc_final[proj]["weighted_mse_per_expert"][gidx] = w_leaf_weighted
                leaf_weighted_vals.append(w_leaf_weighted)
                leaf_weighted_denom.append(denom_leaf)
                leaf_weighted_numer.append(numer_leaf)
                if act_h is None or (isinstance(act_h, torch.Tensor) and (act_h.numel() == 0 or int(act_h.shape[0]) == 0)):
                    per_leaf_acc_final[proj]["output_mse_per_expert"][gidx] = float("inf")
                    per_leaf_acc_final[proj]["rel_output_mse_per_expert"][gidx] = float("inf")
                else:
                    act_f2 = torch.as_tensor(act_h).to(w_leaf_bf16.device, torch.float32)
                    try:
                        out_mse_leaf = canonical_nvfp4_cb_single_output_mse(w_leaf_bf16, r_leaf_bf16, act_f2, spec)
                    except ValueError:
                        out_mse_leaf = float("inf")
                    per_leaf_acc_final[proj]["output_mse_per_expert"][gidx] = float(out_mse_leaf)
                    if out_mse_leaf == float("inf"):
                        per_leaf_acc_final[proj]["rel_output_mse_per_expert"][gidx] = float("inf")
                    else:
                        w_leaf_f32 = w_leaf_bf16.to(torch.float32)
                        ref_leaf = float((act_f2.to(w_leaf_f32.device) @ w_leaf_f32.T).pow(2).mean().item())
                        per_leaf_acc_final[proj]["rel_output_mse_per_expert"][gidx] = float(out_mse_leaf / max(ref_leaf, 1e-12))
            # Fill fused weighted: true piecewise total weighted squared-error / total mass
            if len(projections) > 1 and isinstance(leaf_col_weights_map, dict):
                if leaf_weighted_vals and all(v == v and v != float("inf") for v in leaf_weighted_vals):
                    total_numer = float(sum(leaf_weighted_numer))
                    total_denom = float(sum(leaf_weighted_denom))
                    if total_denom > 1e-30 and total_numer == total_numer:
                        weighted_mse_fused[gidx] = float(total_numer / max(total_denom, 1e-30))
                    else:
                        weighted_mse_fused[gidx] = float("inf")
                    # Also store diagnostic mean for comparison (non-authoritative)
                elif leaf_weighted_vals:
                    weighted_mse_fused[gidx] = float("inf")
        del chunk_recon_bf16, _chunk_fields
    per_leaf: dict[str, dict[str, Any]] = {}
    for proj in projections:
        per_leaf[proj] = {
            "weight_mse_per_expert": per_leaf_acc_final[proj]["weight_mse_per_expert"],
            "weighted_mse_per_expert": per_leaf_acc_final[proj]["weighted_mse_per_expert"],
            "output_mse_per_expert": per_leaf_acc_final[proj]["output_mse_per_expert"],
            "rel_output_mse_per_expert": per_leaf_acc_final[proj]["rel_output_mse_per_expert"],
        }
    n_rows = [int(a.shape[0]) if a is not None and isinstance(a, torch.Tensor) and a.numel() > 0 else 0 for a in holdout_rows]
    # Canonical payload identity from exact assembled bytes (pure shared helper, no source-weight hashing, no truncation, no caught errors)
    payload_identity = canonical_nvfp4_cb_payload_identity(
        final_fields, rung, grid="fp4", mode="product",
        member_order=list(projections), slice_boundaries=slice_boundaries,
        per_leaf_kept=per_leaf_kept,
        format_name=format_name,
    )
    warm_path = None
    if write_warm_state:
        from prismaquant.cb_warm_state import CBWarmStateStore, build_warm_record
        logical_qname = f"model.layers.{layer}.mlp.experts.{packed_proj}"
        rec = build_warm_record(qname=logical_qname, format_name=format_name, source_weight=data["weight"], col_weights=data["col_weights"], context=CTX_NVFP4_LDLQ, fields=final_fields)
        warm_path = CBWarmStateStore(DERIVED_WARM).write(rec)
    fields = final_fields
    # Build qnames list per leaf for backward compat: flattened
    qnames_flat: list[str] = []
    for proj in projections:
        qnames_flat.extend(data["qnames_per_leaf"][proj])
    expert_count = int(data["weight"].shape[0])
    result: dict[str, Any] = {
        "schema": "prismaquant.dsv4_nvfp4_projection_rung.v1",
        "layer": layer,
        "projection": packed_proj,
        "packed_parent": packed_proj,
        "expert_count": expert_count,
        "member_order": list(projections),
        "slice_boundaries": {k: list(v) for k, v in slice_boundaries.items()},
        "col_weight_pooling": data.get("col_weight_pooling", "mean_of_member_vectors"),
        "fused_activation_policy": FUSED_ACTIVATION_POLICY_V1,
        "fused_activation_order": list(projections),
        "format": format_name,
        "rung": rung,
        "qnames": qnames_flat,
        "qnames_per_leaf": {k: list(v) for k, v in data["qnames_per_leaf"].items()},
        "weight_mse_per_expert": per_leaf[projections[0]]["weight_mse_per_expert"] if len(projections) == 1 else weight_mse_fused,
        "weighted_mse_per_expert": per_leaf[projections[0]]["weighted_mse_per_expert"] if len(projections) == 1 else weighted_mse_fused,
        "weighted_mse_diagnostic_mean_per_expert": [float(sum(per_leaf[proj]["weighted_mse_per_expert"][e] for proj in projections)/len(projections)) for e in range(E)] if len(projections) > 1 else list(weighted_mse_fused),
        "output_mse_per_expert": per_leaf[projections[0]]["output_mse_per_expert"] if len(projections) == 1 else output_mse_fused,
        "rel_output_mse_per_expert": per_leaf[projections[0]]["rel_output_mse_per_expert"] if len(projections) == 1 else rel_output_mse_fused,
        "per_leaf": per_leaf,
        "weight_mse_fused_per_expert": weight_mse_fused,
        "output_mse_fused_per_expert": output_mse_fused,
        "n_activation_rows_per_expert": n_rows,
        "elapsed_seconds": elapsed,
        "warm_state_path": str(warm_path) if warm_path else None,
        "payload_identity": dict(payload_identity),
        "payload_hash": payload_identity["full_hash"],
        "payload_hash_algo": payload_identity["abi"],
        "physical_parent": packed_proj,
        "physical_rung": rung,
        "cold_experts": list(data["cold_experts"]),
        "observed_activation_files": int(data["observed_activation_files"]),
        "encoder": {"ldlq": True, "scope": "nvfp4", "batch_experts": True, "encode_tier": "balanced", "gate": str(gate_info.get("gate")), "packed_parent": packed_proj, "member_order": list(projections), "equal_leaf_widths_validated": True if len(projections) == 1 else bool(len(set(data.get("leaf_out_dims", []))) == 1)},
        "gate_info": dict(gate_info),
        "leaf_widths_equal": True if len(projections) == 1 else bool(len(set(data.get("leaf_out_dims", []))) == 1),
        "leaf_row_hashes": dict(payload_identity["row_hashes"]),
    }
    # Validate all vectors match expert_count
    for k in ("weight_mse_per_expert", "output_mse_per_expert", "rel_output_mse_per_expert", "n_activation_rows_per_expert"):
        if len(result[k]) != expert_count:
            raise AssertionError(f"{packed_proj} result {k} length {len(result[k])} != expert_count {expert_count}")
    for proj, leaf in per_leaf.items():
        for k in ("weight_mse_per_expert", "output_mse_per_expert", "rel_output_mse_per_expert"):
            if len(leaf[k]) != expert_count:
                raise AssertionError(f"{packed_proj} per_leaf {proj} {k} length mismatch")
        if len(data["qnames_per_leaf"][proj]) != expert_count:
            raise AssertionError(f"{packed_proj} qnames_per_leaf {proj} length mismatch")
    return result

def derive_one_projection_rung(layer: int, projection: str, rung: int, device: torch.device, write_warm_state: bool = False) -> dict[str, Any]:
    """Smoke helper: 256 results, no shard write, no warm state by default.

    Accepts either a packed parent (gate_up_proj/down_proj) or a leaf name
    (gate_proj/up_proj). Leaf requests are served via the fused parent and
    sliced to the leaf's metrics, so smoke proves fused semantics.
    """
    if write_warm_state:
        raise ValueError("smoke must not write warm state: pass write_warm_state=False")
    require_cuda(device)
    _, rec = load_layer_identity(layer)
    identity = rec["identity"]
    col_weights_all = _load_col_weights_cached()
    from prismaquant.layer_streaming import _build_fp8_scale_inv_map, _build_weight_map
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    # Map leaf to packed parent via authoritative planning — no hardcoded projection list.
    # Fail-closed: unknown leaf not in any packed parent's projection set.
    profile, _, _, packed_names = _get_packed_planning()
    target_packed = projection
    leaf_for_smoke: str | None = None
    if projection in packed_names:
        target_packed = projection
    else:
        # Search for leaf in packed parents' projection sets via public helper
        found = None
        for _pp in packed_names:
            _projs = get_packed_expert_projection_names(profile, _pp)
            if projection in _projs:
                found = _pp
                break
        if found is not None:
            target_packed = found
            leaf_for_smoke = projection
        elif projection not in packed_names:
            # Unknown projection for this SOURCE — fail closed, no broad hardcoded fallback
            raise AssertionError(f"projection {projection!r} not in authoritative packed set {sorted(packed_names)} nor in any leaf set")
    # Load the authoritative packed parent (fused for gate_up) with ownership-bound cleanup
    # No local alias to data is held across the empty_cache boundary; holder is the sole owner.
    holder: dict = {"data": load_packed_projection(layer, target_packed, device=device, identity=identity, all_col_weights=col_weights_all, model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt, scale_map=scale_map)}
    try:
        result = encode_nvfp4_rung_packed(layer, target_packed, rung, holder["data"], device, write_warm_state=False)
        # If smoke asked for a leaf slice of a fused parent, return that leaf's metrics shaped like the old leaf result
        expert_count = int(result.get("expert_count", len(result.get("weight_mse_per_expert", []))))
        if leaf_for_smoke is not None and leaf_for_smoke in result.get("per_leaf", {}):
            leaf = result["per_leaf"][leaf_for_smoke]
            leaf_result = dict(result)
            leaf_result["projection"] = leaf_for_smoke
            leaf_result["packed_parent"] = target_packed
            leaf_result["weight_mse_per_expert"] = leaf["weight_mse_per_expert"]
            leaf_result["weighted_mse_per_expert"] = leaf["weighted_mse_per_expert"]
            leaf_result["output_mse_per_expert"] = leaf["output_mse_per_expert"]
            leaf_result["rel_output_mse_per_expert"] = leaf["rel_output_mse_per_expert"]
            assert len(leaf_result["weight_mse_per_expert"]) == expert_count
            assert len(leaf_result["output_mse_per_expert"]) == expert_count
            gi = leaf_result["gate_info"]
            assert "gate" in gi, "gate_info missing gate field"
            return leaf_result
        assert len(result["weight_mse_per_expert"]) == expert_count
        assert len(result["output_mse_per_expert"]) == expert_count
        assert len(result["rel_output_mse_per_expert"]) == expert_count
        assert len(result["n_activation_rows_per_expert"]) == expert_count
        gi = result["gate_info"]
        assert "gate" in gi, "gate_info missing gate field"
        return result
    finally:
        # Shared fail-closed ownership: clear, drop holder before empty, chain errors
        _finalize_packed_holder_cleanup(holder, device)

# ---------------------------------------------------------------------------
# Dense handling: load once per tensor, measure all rungs (avoid 466MB per rung)
# Dense now has content-keyed atomic all-rung checkpoints.
# ---------------------------------------------------------------------------

def load_dense_tensor(
    qname: str, layer: int, device: torch.device, identity: Mapping[str, Any], all_col_weights: Mapping[str, Any], model_to_shard: Mapping[str, str], model_to_ckpt: Mapping[str, str], scale_map: Mapping[str, Any],
) -> dict[str, Any]:
    from prismaquant.layer_streaming import _read_layer_to_device
    from prismaquant.production_weight_cache import validate_cb_render_source_weight
    cw = torch.as_tensor(all_col_weights[qname]).to(torch.float32).contiguous()
    if list(cw.shape) != list(identity["col_weights_shapes"][qname]):  # type: ignore[index]
        raise AssertionError(f"{qname}: col-weight shape mismatch")
    if content_sha256_float32(cw) != identity["col_weights_content_sha256"][qname]:  # type: ignore[index]
        raise AssertionError(f"{qname}: col-weight digest mismatch")
    loaded = _read_layer_to_device(qname + ".weight", model_to_shard, model_to_ckpt, torch.bfloat16, device, fp8_scale_inv_map=scale_map)
    if set(loaded) != {qname + ".weight"}:
        raise AssertionError(f"{qname}: source resolved {sorted(loaded)}")
    w = loaded[qname + ".weight"].contiguous()
    validate_cb_render_source_weight(identity, qname, w, where=f"DSV4 layer {layer} dense source")
    # Dense activation: direct file only (no pooled prior); missing is fail-closed (empty tensor).
    act = load_direct_activation(qname, int(w.shape[1])).to(device) if w.ndim == 2 else torch.empty((0, int(w.shape[-1])), dtype=torch.float32, device=device)
    return {"qname": qname, "weight": w, "col_weights": cw.to(device), "activation_rows": act}

def measure_dense_all_rungs(dense_loaded: dict[str, Any], rungs: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Single explicit path: raw fields, LDLQ reassignment on fit rows (ungated), canonical holdout comparison, selected fields.

    Prepared evidence is built once outside the rung loop and the same
    prepared object (same fit tensor ids) is reused across all seven rungs
    via the canonical gated path, so the factor is built once and hits cache
    on later rungs. No duplicated manual split/gate logic.
    """
    results: dict[int, dict[str, Any]] = {}
    w = dense_loaded["weight"]
    cw = dense_loaded["col_weights"]
    act = dense_loaded["activation_rows"]
    qname = dense_loaded["qname"]
    from prismaquant.nvfp4_cb_formats import (
        nvfp4_cb_fields as _raw_fields_fn,
        canonical_nvfp4_cb_single_output_mse as _canon_single,
        prepare_ldlq_gate_evidence as _prepare_dense,
        ldlq_reassign_cb_fields_gated as _gated_dense,
    )

    # Prepare once outside the loop — same objects reused for factor cache hit across rungs
    # Once-prepared fail-closed: do not swallow preparation errors or re-split per rung.
    # Malformed/insufficient evidence is represented by the prepared sentinel itself
    # (eligible empty, has_observed_fit false) which stays raw with zero factors.
    # Programmer errors propagate rather than silently reparsing.
    prepared_dense = _prepare_dense(act, qname=qname)
    for rung in rungs:
        format_name = f"NVFP4_CB_K{rung}"
        spec = fr.get_format(format_name)
        raw_fields = _raw_fields_fn(w, rung, grid="fp4", mode="product", col_weights=cw)
        # Canonical gate path with prepared evidence (reuses same fit tensor objects)
        fields, gate_info = _gated_dense(w, raw_fields, cw, act, grid="fp4", mode="product", k=rung, prepared=prepared_dense)
        recon = nvfp4_cb_reconstruct(fields, rung, grid="fp4", mode="product").to(w.dtype)
        # Final measurement on full original rows for comparability (canonical)
        w_f = w.to(torch.float32)
        r_f = recon.to(torch.float32)
        weight_mse = float((w_f - r_f).pow(2).mean().item())
        if act.numel() == 0 or int(act.shape[0]) == 0:
            output_mse = float("inf")
            rel = float("inf")
            n_rows = 0
            cost_source = "raw_immutable_copy" if not gate_info.get("kept_ldlq") else "ldlq_direct_measured"
        else:
            act_f = act.to(w_f.device, torch.float32)
            # Use canonical helper for final full-row measurement
            try:
                output_mse = _canon_single(w, recon, act_f, spec)
            except ValueError:
                output_mse = float("inf")
            ref_energy = float((act_f @ w_f.T).pow(2).mean().item())
            rel = output_mse / max(ref_energy, 1e-12) if output_mse != float("inf") else float("inf")
            n_rows = int(act.shape[0])
            # Truthful provenance: only LDLQ-kept with finite canonical measurement is direct measured
            if gate_info.get("kept_ldlq") and output_mse != float("inf"):
                cost_source = "ldlq_direct_measured"
            else:
                cost_source = "raw_immutable_copy" if act.numel() > 0 else "raw_fallback_missing_activation"
        results[rung] = {
            "weight_mse": weight_mse,
            "output_mse": output_mse,
            "rel_output_mse": rel,
            "n_activation_rows": n_rows,
            "cost_source": cost_source,
            "ldlq_scope": "nvfp4",
            "gate_info": dict(gate_info),
            "gate": str(gate_info.get("gate")),
        }
    return results

# ---------------------------------------------------------------------------
# Rung coverage validation: measure exactly NVFP4 formats present in raw menu
# ---------------------------------------------------------------------------

def _expected_nvfp4_formats_from_raw(raw_costs: Mapping[str, Any]) -> list[str]:
    """Derive NVFP4 format list from raw menu and validate uniform coverage."""
    # Find first qname with NVFP4 entry
    first_qname = None
    expected: set[str] | None = None
    for qname, row in raw_costs.items():
        nvfp4 = sorted([k for k in row.keys() if str(k).startswith("NVFP4_CB")])
        if expected is None:
            expected = set(nvfp4)
            first_qname = qname
            if not expected:
                raise AssertionError(f"{qname}: raw has no NVFP4 formats")
        else:
            if set(nvfp4) != expected:
                raise AssertionError(f"{qname}: NVFP4 coverage mismatch: expected {sorted(expected)} from {first_qname} but got {sorted(nvfp4)}")
    if expected is None:
        raise AssertionError("raw_costs empty")
    return sorted(expected)

def _nvfp4_rungs_from_formats(formats: Sequence[str]) -> list[int]:
    return [int(f.split("_K")[-1]) for f in formats]

# ---------------------------------------------------------------------------
# Shard merge: preserve FP8 deep equality, replace NVFP4 only, rebuild identity
# ---------------------------------------------------------------------------

def _nvfp4_formats_for_ctx(ctx: CBSerializationContext) -> list[str]:
    # NVFP4 CB rungs present in cost-ldlq shards: K12-K18 inclusive
    return [f"NVFP4_CB_K{k}" for k in NVFP4_RUNGS]

def build_derived_identity(raw_identity: Mapping[str, Any]) -> dict[str, Any]:
    new_identity = copy.deepcopy(dict(raw_identity))
    # Derive formats from raw identity's canonical stamp — never hardcode.
    raw_sc = raw_identity.get("serialization_context") if isinstance(raw_identity.get("serialization_context"), dict) else {}
    lattice = raw_sc.get("lattice_codebook_sha256_by_format") if isinstance(raw_sc, dict) else {}
    if isinstance(lattice, Mapping) and lattice:
        all_formats = sorted(str(k) for k in lattice.keys())
    else:
        raw_formats = raw_identity.get("formats") if isinstance(raw_identity.get("formats"), (list, tuple)) else None
        if raw_formats and any(str(f).startswith("NVFP4_CB") or str(f).startswith("FP8_CB") for f in raw_formats):
            all_formats = sorted(str(f) for f in raw_formats if str(f).startswith("NVFP4_CB") or str(f).startswith("FP8_CB"))
        else:
            raise ValueError("build_derived_identity: raw identity has no lattice_codebook and no formats list — cannot derive CB formats, refusing to hardcode")
    if not all_formats:
        raise ValueError("build_derived_identity: derived all_formats empty — fail closed")
    new_stamp = cb_serialization_context_stamp(CTX_NVFP4_LDLQ, formats=all_formats)
    new_identity["serialization_context"] = new_stamp
    # Preserve raw identity: do not overwrite implementation_sha256, record as raw
    raw_impl = copy.deepcopy(raw_identity.get("implementation_sha256"))
    if raw_impl is not None:
        new_identity["raw_implementation_sha256"] = raw_impl
        # Keep original implementation_sha256 unchanged (schema-required producer field)
        new_identity["implementation_sha256"] = raw_impl
    # Add dual-basis specific implementation digests separately
    mod_shas = _cached_module_shas()
    new_identity["dual_basis_implementation_sha256"] = {
        "derive_tool": mod_shas["derive_tool"],
        "nvfp4_cb_footprint": mod_shas["nvfp4_cb_footprint"],
        "nvfp4_cb_formats": mod_shas["nvfp4_cb_formats"],
        "cb_ldlq": mod_shas["cb_ldlq"],
    }
    new_identity["dual_basis_derivation"] = {
        "derivation": "dual_basis_nvfp4_ldlq_scope_nvfp4_direct_7rungs",
        "derive_tool_sha256": _cached_tool_sha256(),
        "tool_path": str(THIS_FILE),
    }
    new_identity["derive_tool_sha256"] = _cached_tool_sha256()
    new_identity["dual_basis_tool_sha256"] = _cached_tool_sha256()
    # Use cached SHAs to avoid hot-loop hashing
    new_identity["source_index_sha256"] = _cached_source_index_sha256()
    new_identity["col_weights_sha256"] = _cached_col_weights_sha256()
    new_identity["activation_cache_dir"] = str(ACT_ROOT.resolve())
    new_identity["by_layer_sha256"] = new_identity.get("verified_base_layer_sha256")  # preserved
    return new_identity

def validate_no_fp8_drift(raw_costs: Mapping[str, Any], derived_costs: Mapping[str, Any]) -> None:
    for qname, raw_row in raw_costs.items():
        derived_row = derived_costs.get(qname)
        if derived_row is None:
            raise AssertionError(f"{qname}: missing in derived costs")
        raw_fp8 = {k: v for k, v in raw_row.items() if str(k).startswith("FP8_CB")}
        derived_fp8 = {k: v for k, v in derived_row.items() if str(k).startswith("FP8_CB")}
        if raw_fp8 != derived_fp8:
            # Deep equality required; even added keys are drift.
            raise AssertionError(f"{qname}: FP8 drift (raw keys {sorted(raw_fp8)} vs derived {sorted(derived_fp8)}); raw FP8 must remain byte-identical")

def derive_layer_full(layer: int, device: torch.device) -> Path:
    require_ldlq_gate_enabled()
    raw_path = RAW_SHARDS / f"layer_{layer:03d}.pkl"
    if not raw_path.is_file():
        raise FileNotFoundError(f"layer {layer} missing raw shard {raw_path}")
    raw_payload = pickle.loads(raw_path.read_bytes())
    if raw_payload.get("schema") != SCHEMA:
        raise ValueError(f"layer {layer} bad schema {raw_payload.get('schema')!r}, expected {SCHEMA!r}")
    if raw_payload.get("content_key") != _sha(raw_payload["identity"]):
        raise ValueError(f"layer {layer} content_key does not equal sha256(canonical identity)")
    # Precompute SHAs once per layer (not per rung/expert)
    raw_sha256 = sha256_file(raw_path)
    by_layer_rec_payload, rec = load_layer_identity(layer)
    identity = rec["identity"]
    by_layer_sha256 = rec["sha256"]
    source_index_sha256 = _cached_source_index_sha256()
    col_weights_sha256 = _cached_col_weights_sha256()
    tool_sha256 = _cached_tool_sha256()
    context_stamp = cb_serialization_context_stamp(CTX_NVFP4_LDLQ, formats=[f"NVFP4_CB_K{k}" for k in NVFP4_RUNGS] + [f"FP8_CB_K{k}" for k in range(28, 39)])
    # Validate NVFP4 coverage: measure exactly formats present in raw menu
    expected_nvfp4_formats = _expected_nvfp4_formats_from_raw(raw_payload["costs"])
    # Ensure raw uses exactly NVFP4_RUNGS (reject extra/missing if raw deviates from declared NVFP4_RUNGS menu)
    expected_from_rungs = [f"NVFP4_CB_K{k}" for k in NVFP4_RUNGS]
    if sorted(expected_nvfp4_formats) != sorted(expected_from_rungs):
        # If raw has different rungs, we still only measure raw's present set but flag mismatch with declared RUNG menu
        # For strictness, require raw matches NVFP4_RUNGS exactly; otherwise abort.
        raise AssertionError(f"layer {layer}: NVFP4 rung set {sorted(expected_nvfp4_formats)} != declared NVFP4_RUNGS {sorted(expected_from_rungs)}")
    nvfp4_rungs_to_measure = _nvfp4_rungs_from_formats(expected_nvfp4_formats)
    # Atomically preserve raw plane (hash-verified, never overwrite)
    raw_copy = DERIVED_RAW_PLANE / f"layer_{layer:03d}.pkl"
    if not raw_copy.is_file():
        atomic_bytes_copy(raw_path, raw_copy)
    else:
        if sha256_file(raw_copy) != raw_sha256:
            raise AssertionError(f"layer {layer}: raw_plane copy drift")
    # Prepare derived payload as deep copy
    payload = copy.deepcopy(raw_payload)
    # Load shared maps ONCE per layer (not per rung, not per projection loop)
    # 466 MB col-weights mapping is deserialized once per process and reused.
    col_weights_all = _load_col_weights_cached()
    from prismaquant.layer_streaming import _build_fp8_scale_inv_map, _build_weight_map
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    # For each authoritative packed parent (gate_up_proj, down_proj): load fused stack once,
    # encode once per rung on the fused E x fused_R x C tensor, slice fused reconstruction
    # to emit per-leaf metrics. This matches the streaming export's DSV4 path exactly.
    manifest_gate_buckets: dict[str, int] = {}
    profile, _, _, packed_names = _get_packed_planning()
    if not packed_names:
        raise AssertionError(f"layer {layer}: no packed expert projections detected for SOURCE {SOURCE}")
    for packed_proj in sorted(packed_names):
        # Sole owner is holder; no local alias survives empty_cache boundary
        holder = {"data": load_packed_projection(layer, packed_proj, device=device, identity=identity, all_col_weights=col_weights_all, model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt, scale_map=scale_map)}
        try:
            projections = tuple(holder["data"]["projections"])
            slice_boundaries = dict(holder["data"]["slice_boundaries"])
            member_order = list(holder["data"]["member_order"])
            col_pooling = str(holder["data"]["col_weight_pooling"])
            # Single truthful owner for prepared gate evidence — retain same fit tensor objects across all missing rungs
            from prismaquant.nvfp4_cb_formats import prepare_ldlq_gate_evidence
            prepared = prepare_ldlq_gate_evidence(holder["data"]["activation_rows"], qname=f"model.layers.{layer}.mlp.experts.{packed_proj}")
            # Hold prepared in holder so fit_filled tensors stay alive across rungs and are not GC'd; also ensures same objects for factor cache reuse
            holder["prepared"] = prepared
            split_infos_for_evidence = list(prepared.split_infos)
            activation_evidence = _packed_activation_evidence_identity(
                holder["data"]["activation_rows"], ACT_ROOT, f"model.layers.{layer}.mlp.experts.{packed_proj}",
                member_order=member_order, slice_boundaries=slice_boundaries, col_weight_pooling=col_pooling,
                cold_experts=holder["data"]["cold_experts"], split_infos=split_infos_for_evidence,
            )
            expert_count = int(holder["data"]["weight"].shape[0])
            # Resume-aware: determine missing rungs first (validates identities/covers every checkpoint)
            ckpt_info_by_rung: dict[int, tuple[dict, Path]] = {}
            existing_by_rung: dict[int, dict | None] = {}
            for rung in nvfp4_rungs_to_measure:
                ckpt_identity = projection_checkpoint_identity(
                    layer, packed_proj, rung, by_layer_sha256=by_layer_sha256, col_weights_sha256=col_weights_sha256, source_index_sha256=source_index_sha256, context_stamp=context_stamp, tool_sha256=tool_sha256, activation_evidence=activation_evidence, member_order=member_order, slice_boundaries=slice_boundaries, col_weight_pooling=col_pooling, expert_count=expert_count,
                )
                ckpt_path = DERIVED_CHECKPOINTS / f"layer_{layer:03d}_{packed_proj}_K{rung}.pkl"
                existing = validated_projection_checkpoint(ckpt_path, ckpt_identity)
                ckpt_info_by_rung[rung] = (ckpt_identity, ckpt_path)
                existing_by_rung[rung] = existing
            # No explicit full-row prewarm — factor is populated by first missing rung's gated LDLQ on FIT rows, later rungs hit cache via same prepared object. All-resumed paths do zero factor work beyond identity validation.
            for rung in nvfp4_rungs_to_measure:
                ckpt_identity, ckpt_path = ckpt_info_by_rung[rung]
                existing = existing_by_rung[rung]
                if existing is not None:
                    res = existing["result"]
                else:
                    res = encode_nvfp4_rung_packed(layer, packed_proj, rung, holder["data"], device, write_warm_state=True, prepared_evidence=prepared)
                    ckpt_payload = {"schema": PROJECTION_RUNG_SCHEMA, "content_key": _sha(ckpt_identity), "identity": ckpt_identity, "result": res}
                    atomic_pickle(ckpt_path, ckpt_payload)
                g = str(res["gate_info"].get("gate"))
                manifest_gate_buckets[g] = manifest_gate_buckets.get(g, 0) + 1
                per_leaf = res.get("per_leaf", {})
                # For single-projection parents, per_leaf may be empty; fall back to fused leaf
                for leaf_proj in projections:
                    leaf_metrics = per_leaf.get(leaf_proj) if per_leaf else None
                    if leaf_metrics is None:
                        # Single parent: metrics are at top-level
                        leaf_metrics = {
                            "weight_mse_per_expert": res["weight_mse_per_expert"],
                            "output_mse_per_expert": res["output_mse_per_expert"],
                            "rel_output_mse_per_expert": res["rel_output_mse_per_expert"],
                        }
                    qnames_leaf = holder["data"]["qnames_per_leaf"][leaf_proj]
                    for idx, qname in enumerate(qnames_leaf):
                        fmt = f"NVFP4_CB_K{rung}"
                        if fmt not in expected_nvfp4_formats:
                            raise AssertionError(f"{qname}: NVFP4 format {fmt} not in raw menu {expected_nvfp4_formats}")
                        raw_entry = raw_payload["costs"][qname].get(fmt, {})
                        raw_weight_mse = float(raw_entry.get("weight_mse", 0.0)) if isinstance(raw_entry, dict) else 0.0
                        raw_output_mse = float(raw_entry.get("output_mse", 0.0)) if isinstance(raw_entry, dict) else 0.0
                        # Use leaf-authoritative kept and per-leaf MSEs
                        compact = compact_gate_for_expert(res["gate_info"], idx, leaf=leaf_proj)
                        kept = bool(compact.get("kept_ldlq"))
                        ldlq_mse_val = float(leaf_metrics["output_mse_per_expert"][idx])
                        raw_holdout_mse = float(compact.get("raw_mse", raw_output_mse)) if isinstance(compact.get("raw_mse"), (int, float)) else raw_output_mse
                        # Require finite positive raw holdout denominator for ratio; otherwise raw winner
                        has_finite_raw_holdout = raw_holdout_mse == raw_holdout_mse and raw_holdout_mse not in (float("inf"), float("-inf")) and raw_holdout_mse > 1e-30
                        has_finite_ldlq = ldlq_mse_val == ldlq_mse_val and ldlq_mse_val not in (float("inf"), float("-inf"))
                        # Anchored ratio pricing: chosen_holdout / raw_holdout * immutable raw anchor
                        if kept and has_finite_raw_holdout and has_finite_ldlq and compact.get("gate") not in ("raw_fallback_missing_activation", "raw_fallback_malformed_activation"):
                            ratio = float(ldlq_mse_val / raw_holdout_mse) if raw_holdout_mse > 0 else 1.0
                            if not (ratio == ratio and ratio not in (float("inf"), float("-inf")) and ratio > 0 and ratio < 1e6):
                                ratio = 1.0
                                kept = False
                            else:
                                # High-rate regressions gate to raw when ratio > 1 (regression)
                                # but per-leaf decision already ensured ldlq <= raw, so ratio <=1; still guard
                                if ratio > 1.0 + 1e-12:
                                    ratio = 1.0
                                    kept = False
                        else:
                            ratio = 1.0
                        if kept and has_finite_raw_holdout and has_finite_ldlq and compact.get("gate") not in ("raw_fallback_missing_activation", "raw_fallback_malformed_activation"):
                            # Anchored NV cost: immutable raw anchor scaled by holdout ratio
                            # Ratio is authoritative; raw anchor preserved
                            anchored_weight = float(raw_weight_mse * ratio) if raw_weight_mse == raw_weight_mse and raw_weight_mse not in (float("inf"),) else float(leaf_metrics["weight_mse_per_expert"][idx])
                            anchored_output = float(raw_output_mse * ratio) if raw_output_mse == raw_output_mse and raw_output_mse not in (float("inf"),) else ldlq_mse_val
                            anchored_rel = float(raw_entry.get("rel_output_mse", leaf_metrics["rel_output_mse_per_expert"][idx]) * ratio) if isinstance(raw_entry.get("rel_output_mse"), (int,float)) and raw_entry.get("rel_output_mse") == raw_entry.get("rel_output_mse") else float(leaf_metrics["rel_output_mse_per_expert"][idx])
                            entry = {
                                "weight_mse": anchored_weight,
                                "output_mse": anchored_output,
                                "rel_output_mse": anchored_rel,
                                "n_activation_rows": int(res["n_activation_rows_per_expert"][idx]),
                                "cost_source": "holdout_ratio_adjusted_raw_anchor",
                                "cost_source_detail": "ldlq_holdout_ratio_adjusted_raw_anchor",
                                "ldlq_scope": "nvfp4",
                                "gate": str(compact.get("gate")),
                                "gate_decision": compact,
                                "packed_parent": packed_proj,
                                "member_order": list(member_order),
                                "slice_boundaries": {k: list(v) for k, v in slice_boundaries.items()},
                                "col_weight_pooling": col_pooling,
                                "raw_source_weight_mse": raw_weight_mse,
                                "raw_source_output_mse": raw_output_mse,
                                "raw_holdout_mse": float(raw_holdout_mse),
                                "ldlq_holdout_mse": float(ldlq_mse_val),
                                "holdout_ratio": float(ratio),
                                "holdout_n_eval_rows": int(res["n_activation_rows_per_expert"][idx]),
                                "holdout_evidence_digest": str(activation_evidence.get("evidence_sha256", "")),
                                "holdout_split_identity": f"{res['gate_info'].get('split_policy')}:{res['gate_info'].get('split_version')}",
                                "physical_parent": str(packed_proj),
                                "physical_rung": int(rung),
                                "provenance": {
                                    "derived_from_raw_sha256": raw_sha256,
                                    "projection_checkpoint": str(ckpt_path),
                                    "projection_checkpoint_content_key": _sha(ckpt_identity),
                                    "packed_parent": packed_proj,
                                    "anchor_identity": f"{qname}:{fmt}:raw_anchor_sha256_{hashlib.sha256(json.dumps(raw_entry, sort_keys=True).encode()).hexdigest()[:16]}" if isinstance(raw_entry, dict) else "",
                                },
                            }
                        else:
                            # Raw winner: preserve immutable raw anchor at ratio 1, label heldout evidence correctly
                            if not isinstance(raw_entry, dict) or "weight_mse" not in raw_entry:
                                raise AssertionError(f"{qname} {fmt}: raw immutable copy missing finite entry")
                            entry = copy.deepcopy(raw_entry)
                            entry["cost_source"] = "raw_immutable_copy"
                            entry["cost_source_detail"] = "raw_holdout_ratio_1_preserved_anchor"
                            entry["gate"] = str(compact.get("gate"))
                            entry["gate_decision"] = compact
                            entry["raw_source_weight_mse"] = raw_weight_mse
                            entry["raw_source_output_mse"] = raw_output_mse
                            entry["raw_holdout_mse"] = float(raw_holdout_mse) if has_finite_raw_holdout else float("inf")
                            entry["ldlq_holdout_mse"] = float(ldlq_mse_val) if has_finite_ldlq else float("inf")
                            entry["holdout_ratio"] = 1.0
                            entry["holdout_n_eval_rows"] = int(res["n_activation_rows_per_expert"][idx])
                            entry["holdout_evidence_digest"] = str(activation_evidence.get("evidence_sha256", ""))
                            entry["holdout_split_identity"] = f"{res['gate_info'].get('split_policy')}:{res['gate_info'].get('split_version')}"
                            entry["physical_parent"] = str(packed_proj)
                            entry["physical_rung"] = int(rung)
                            # Cold arms have no heldout evidence — never label as direct heldout
                            if idx in set(res.get("gate_info", {}).get("cold_experts", [])) or not has_finite_raw_holdout:
                                entry["holdout_evidence"] = "cold_no_heldout_eval_rows"
                                entry["provenance_note"] = "cold arm uses exact leaf-local raw bytes and immutable raw anchor at ratio 1; no heldout evidence"
                            entry["provenance"] = {
                                "derived_from_raw_sha256": raw_sha256,
                                "projection_checkpoint": str(ckpt_path),
                                "projection_checkpoint_content_key": _sha(ckpt_identity),
                                "packed_parent": packed_proj,
                                "copied_from_raw": True,
                                "anchor_identity": f"{qname}:{fmt}:raw_anchor_sha256_{hashlib.sha256(json.dumps(raw_entry, sort_keys=True).encode()).hexdigest()[:16]}" if isinstance(raw_entry, dict) else "",
                            }
                        payload["costs"][qname][fmt] = entry
        finally:
            _finalize_packed_holder_cleanup(holder, device)
    # Ensure no stale factors before dense work (fail-closed)
    from prismaquant.nvfp4_cb_formats import clear_ldlq_factor_cache

    clear_ldlq_factor_cache()
    # Dense qnames: collect once, load once per tensor, measure all rungs via dense checkpoints
    dense_qnames = [q for q in raw_payload["costs"] if q.startswith(f"model.layers.{layer}.") and ".experts." not in q]
    for qname in dense_qnames:
        dense_loaded = load_dense_tensor(qname, layer, device, identity, col_weights_all, model_to_shard, model_to_ckpt, scale_map)
        # Dense missing activation: never store inf. Reuse finite canonical RAW entry unchanged
        # and stamp explicit raw_fallback_missing_activation gate/provenance; export does same.
        # If raw lacks finite values, fail closed.
        act = dense_loaded["activation_rows"]
        is_missing = act.numel() == 0 or act.shape[0] == 0
        col_content_sha = content_sha256_float32(dense_loaded["col_weights"])
        act_evidence_sha = _activation_evidence_digest(qname)
        dckpt_identity = dense_checkpoint_identity(layer, qname, col_weights_content_sha256=col_content_sha, activation_evidence_sha256=act_evidence_sha, by_layer_sha256=by_layer_sha256, col_weights_sha256=col_weights_sha256, source_index_sha256=source_index_sha256, context_stamp=context_stamp, tool_sha256=tool_sha256)
        qname_safe = _ACT_FNAME_SUB.sub("__", qname)
        dckpt_path = DERIVED_DENSE_CHECKPOINTS / f"layer_{layer:03d}_{qname_safe}.pkl"
        existing_dense = validated_dense_checkpoint(dckpt_path, dckpt_identity)
        if existing_dense is not None:
            dense_results_raw = existing_dense["results"]
            dense_results = {int(k): v for k, v in dense_results_raw.items()}
        else:
            if is_missing:
                # Reuse raw finite entries directly — never invent inf cost
                dense_results = {}
                for rung in nvfp4_rungs_to_measure:
                    fmt = f"NVFP4_CB_K{rung}"
                    raw_ent = raw_payload["costs"][qname].get(fmt)
                    if not isinstance(raw_ent, Mapping):
                        raise AssertionError(f"{qname} {fmt}: missing raw entry for fallback")
                    # Require finite numeric cost values
                    for k in ("weight_mse", "output_mse", "rel_output_mse"):
                        v = raw_ent.get(k)
                        if not isinstance(v, (int, float)) or not __import__("math").isfinite(float(v)):
                            raise AssertionError(f"{qname} {fmt}: raw {k} not finite ({v!r}) — fail closed")
                    # Use raw entry verbatim but stamp fallback gate
                    dense_results[rung] = {
                        "weight_mse": float(raw_ent["weight_mse"]),
                        "output_mse": float(raw_ent["output_mse"]),
                        "rel_output_mse": float(raw_ent.get("rel_output_mse", raw_ent["output_mse"])),
                        "n_activation_rows": int(raw_ent.get("n_activation_rows", 0)),
                        "cost_source": str(raw_ent.get("cost_source", "raw")),
                        "ldlq_scope": "nvfp4",
                        "gate": "raw_fallback_missing_activation",
                        "gate_info": {
                            "gate": "raw_fallback_missing_activation",
                            "kept_ldlq": False,
                            "reason": "dense activation evidence missing — reused finite canonical RAW entry",
                            "metric": "activation_output_mse",
                            "missing_activation": True,
                        },
                    }
            else:
                dense_results = measure_dense_all_rungs(dense_loaded, nvfp4_rungs_to_measure)
                for rung, ent in dense_results.items():
                    gate = str(ent.get("gate", ""))
                    if (ent["output_mse"] == float("inf") or ent["rel_output_mse"] == float("inf")) and not gate.startswith("raw_fallback"):
                        raise AssertionError(f"{qname} K{rung}: unexpected inf without fallback gate {gate}")
            dckpt_payload = {"schema": DENSE_RUNG_SCHEMA, "content_key": _sha(dckpt_identity), "identity": dckpt_identity, "results": {str(k): v for k, v in dense_results.items()}}
            atomic_pickle(dckpt_path, dckpt_payload)
        for rung, entry in dense_results.items():
            fmt = f"NVFP4_CB_K{rung}"
            if fmt not in expected_nvfp4_formats:
                raise AssertionError(f"{qname}: NVFP4 format {fmt} not in raw menu")
            dense_gate_compact = compact_gate_for_dense(entry.get("gate_info", {}))
            # For missing case, ensure we preserve finite raw values and correct gate
            if is_missing:
                # Verify we are not labeling fallback as ldlq_direct_measured
                if str(entry.get("cost_source")) == "ldlq_direct_measured":
                    raise AssertionError(f"{qname} K{rung}: missing activation must not be labeled ldlq_direct_measured")
            compact_entry = {
                "weight_mse": float(entry["weight_mse"]),
                "output_mse": float(entry["output_mse"]),
                "rel_output_mse": float(entry["rel_output_mse"]),
                "n_activation_rows": int(entry["n_activation_rows"]),
                "cost_source": str(entry.get("cost_source", "ldlq_direct_measured" if not is_missing else "raw")),
                "ldlq_scope": "nvfp4",
                "gate": str(dense_gate_compact.get("gate")),
                "gate_decision": dense_gate_compact,
            }
            if not __import__("math").isfinite(compact_entry["weight_mse"]) or not __import__("math").isfinite(compact_entry["output_mse"]):
                raise AssertionError(f"{qname} K{rung}: compact entry has inf/nan — dense missing must reuse finite raw")
            g = str(dense_gate_compact.get("gate"))
            manifest_gate_buckets[g] = manifest_gate_buckets.get(g, 0) + 1
            payload["costs"][qname][fmt] = compact_entry
        del dense_loaded
    # Validate FP8 deep equality before writing (interpolation provenance stays outside FP8 dicts)
    validate_no_fp8_drift(raw_payload["costs"], payload["costs"])
    # Rebuild shard identity exactly per schema, content_key = sha256(canonical identity)
    new_identity = build_derived_identity(raw_payload["identity"])
    payload["identity"] = new_identity
    payload["content_key"] = _sha(new_identity)
    # Add derived provenance (separate from costs)
    payload["derived_from"] = {
        "raw_shard": str(raw_path),
        "raw_sha256": raw_sha256,
        "raw_copy": str(raw_copy),
        "derivation": "dual_basis_nvfp4_ldlq_scope_nvfp4_direct_7rungs",
        "gate_metric": "activation_output_mse",
        "ldlq_scope": "nvfp4",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "derived_tool_sha256": tool_sha256,
        "col_weights_sha256": col_weights_sha256,
        "source_index_sha256": source_index_sha256,
        "activation_cache_dir": str(ACT_ROOT.resolve()),
        "gate_summary": dict(manifest_gate_buckets),
    }
    # Validate schema and content_key before write
    if payload.get("schema") != SCHEMA:
        raise AssertionError("derived schema mismatch")
    if payload.get("content_key") != _sha(new_identity):
        raise AssertionError("derived content_key mismatch")
    out = DERIVED_SHARDS / f"layer_{layer:03d}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    if pickle.loads(out.read_bytes()).get("content_key") != payload["content_key"]:
        raise AssertionError("derived shard write verification failed")
    return out

def build_manifests() -> None:
    """Write separate raw and derived manifests with scope stamp and hashes and gate telemetry."""
    raw_manifest: dict[str, Any] = {
        "schema": "prismaquant.dsv4_dual_basis_raw_plane_manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_index_sha256": _cached_source_index_sha256(),
        "col_weights_sha256": _cached_col_weights_sha256(),
        "activation_cache_dir": str(ACT_ROOT.resolve()),
        "by_layer_dir": str(BY_LAYER.resolve()),
        "derive_tool_sha256": _cached_tool_sha256(),
        "raw_shards": {},
        "scope": "raw_nvfp4_bank_scope_none",
        "campaign_paths": {
            "RUN_ROOT": str(RUN_ROOT.resolve()),
            "RAW_SHARDS": str(RAW_SHARDS.resolve()),
            "RAW_MERGED": str(RAW_MERGED.resolve()),
            "SOURCE": str(SOURCE.resolve()),
            "BY_LAYER": str(BY_LAYER.resolve()),
            "COL_WEIGHTS": str(COL_WEIGHTS.resolve()),
            "ACT_ROOT": str(ACT_ROOT.resolve()),
            "DERIVED_ROOT": str(DERIVED_ROOT.resolve()),
            "allow_mixed_campaign_paths": bool(globals().get("_CAMPAIGN_MIX_ALLOW", False)),
            "mixed_sources": list(globals().get("_CAMPAIGN_MIX_SOURCES", [])),
        },
    }
    for p in sorted(RAW_SHARDS.glob("layer_*.pkl")):
        raw_manifest["raw_shards"][p.name] = sha256_file(p)
    atomic_json(DERIVED_ROOT / "raw_plane_manifest.json", raw_manifest)
    # Gate telemetry aggregation: scan projection and dense checkpoints for observed counts/reasons — fail-closed on malformed.
    gate_counts: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    if DERIVED_CHECKPOINTS.is_dir():
        for p in DERIVED_CHECKPOINTS.glob("*.pkl"):
            try:
                payload = pickle.loads(p.read_bytes())
            except Exception as exc:
                raise AssertionError(f"gate scan failed for checkpoint {p.name}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise AssertionError(f"checkpoint {p.name} payload not mapping")
            gi = payload.get("result", {}).get("gate_info", {}) if isinstance(payload.get("result"), Mapping) else {}
            if not isinstance(gi, Mapping):
                raise AssertionError(f"checkpoint {p.name} gate_info not mapping")
            g = str(gi.get("gate", "unknown"))
            gate_counts[g] = gate_counts.get(g, 0) + 1
            if "reason" in gi:
                r = str(gi["reason"])
                gate_reasons[r] = gate_reasons.get(r, 0) + 1
            if "missing_experts" in gi:
                if not isinstance(gi["missing_experts"], (list, tuple)):
                    raise AssertionError(f"checkpoint {p.name} missing_experts not list")
                gate_reasons["missing_experts"] = gate_reasons.get("missing_experts", 0) + len(gi["missing_experts"])
    if DERIVED_DENSE_CHECKPOINTS.is_dir():
        for p in DERIVED_DENSE_CHECKPOINTS.glob("*.pkl"):
            try:
                payload = pickle.loads(p.read_bytes())
            except Exception as exc:
                raise AssertionError(f"gate scan failed for dense checkpoint {p.name}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise AssertionError(f"dense checkpoint {p.name} payload not mapping")
            results = payload.get("results", {})
            if not isinstance(results, Mapping):
                raise AssertionError(f"dense checkpoint {p.name} results not mapping")
            for ent in results.values():
                if not isinstance(ent, Mapping):
                    raise AssertionError(f"dense checkpoint {p.name} ent not mapping")
                gi = ent.get("gate_info", {})
                if not isinstance(gi, Mapping):
                    raise AssertionError(f"dense checkpoint {p.name} gate_info not mapping")
                g = str(gi.get("gate", "unknown"))
                gate_counts[g] = gate_counts.get(g, 0) + 1
                if "reason" in gi:
                    r = str(gi["reason"])
                    gate_reasons[r] = gate_reasons.get(r, 0) + 1
    # Also aggregate from derived shards if checkpoints missing (fallback) — fail-closed.
    dense_gate_counts: dict[str, int] = {}
    if DERIVED_SHARDS.is_dir():
        for p in sorted(DERIVED_SHARDS.glob("layer_*.pkl")):
            try:
                payload = pickle.loads(p.read_bytes())
            except Exception as exc:
                raise AssertionError(f"gate scan failed for shard {p.name}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise AssertionError(f"shard {p.name} payload not mapping")
            costs = payload.get("costs", {})
            if not isinstance(costs, Mapping):
                raise AssertionError(f"shard {p.name} costs not mapping")
            for row in costs.values():
                if not isinstance(row, Mapping):
                    raise AssertionError(f"shard {p.name} row not mapping")
                for fmt, ent in row.items():
                    if str(fmt).startswith("NVFP4_CB"):
                        if not isinstance(ent, Mapping):
                            raise AssertionError(f"shard {p.name} {fmt} not mapping")
                        g = str(ent.get("gate", "unknown"))
                        dense_gate_counts[g] = dense_gate_counts.get(g, 0) + 1
    # Source-lineage: prove FP8 untouched without fabricating cross-family provenance.
    # Aggregate cost_source counts and validate FP8 deep equality fail-closed.
    _raw_cost_source_counts: dict[str, int] = {}
    _derived_nvfp4_cost_source_counts: dict[str, int] = {}
    # Build aggregated cost maps for FP8 drift check (deep equality over all FP8 dicts).
    _raw_agg: dict[str, dict[str, Any]] = {}
    _derived_agg: dict[str, dict[str, Any]] = {}
    if DERIVED_RAW_PLANE.is_dir():
        for p in sorted(DERIVED_RAW_PLANE.glob("layer_*.pkl")):
            try:
                pl = pickle.loads(p.read_bytes())
            except Exception as exc:
                raise AssertionError(f"raw plane {p.name} unreadable: {exc}") from exc
            costs = pl.get("costs")
            if not isinstance(costs, Mapping):
                raise AssertionError(f"raw plane {p.name} missing costs mapping")
            for qname, row in costs.items():
                if not isinstance(row, Mapping):
                    raise AssertionError(f"raw plane {p.name} qname {qname!r} row not mapping")
                for fmt, ent in row.items():
                    if not isinstance(ent, Mapping):
                        raise AssertionError(f"raw plane {p.name} {qname} {fmt} not mapping")
                    if "cost_source" in ent:
                        cs = str(ent.get("cost_source"))
                        _raw_cost_source_counts[cs] = _raw_cost_source_counts.get(cs, 0) + 1
                # Aggregate for FP8 check
                if qname not in _raw_agg:
                    _raw_agg[qname] = {}
                for fmt, ent in row.items():
                    if str(fmt).startswith("FP8_CB"):
                        _raw_agg[qname][fmt] = ent
    if DERIVED_SHARDS.is_dir():
        for p in sorted(DERIVED_SHARDS.glob("layer_*.pkl")):
            try:
                pl = pickle.loads(p.read_bytes())
            except Exception as exc:
                raise AssertionError(f"derived shard {p.name} unreadable: {exc}") from exc
            costs = pl.get("costs")
            if not isinstance(costs, Mapping):
                raise AssertionError(f"derived shard {p.name} missing costs mapping")
            for qname, row in costs.items():
                if not isinstance(row, Mapping):
                    raise AssertionError(f"derived shard {p.name} qname {qname!r} row not mapping")
                for fmt, ent in row.items():
                    if not isinstance(ent, Mapping):
                        raise AssertionError(f"derived shard {p.name} {qname} {fmt} not mapping")
                    if str(fmt).startswith("NVFP4_CB") and "cost_source" in ent:
                        cs = str(ent.get("cost_source"))
                        _derived_nvfp4_cost_source_counts[cs] = _derived_nvfp4_cost_source_counts.get(cs, 0) + 1
                if qname not in _derived_agg:
                    _derived_agg[qname] = {}
                for fmt, ent in row.items():
                    if str(fmt).startswith("FP8_CB"):
                        _derived_agg[qname][fmt] = ent
    # Fail-closed FP8 invariant: every FP8 dict must be deep-equal to raw plane.
    # Require nonzero coverage for every present derived shard — never emit true if no evidence.
    _fp8_total_raw = sum(len(v) for v in _raw_agg.values())
    _fp8_total_derived = sum(len(v) for v in _derived_agg.values())
    _fp8_verified = False
    if DERIVED_SHARDS.is_dir() and any(DERIVED_SHARDS.glob("layer_*.pkl")):
        if _fp8_total_raw == 0 or _fp8_total_derived == 0:
            raise AssertionError(f"FP8 coverage empty: raw {_fp8_total_raw} derived {_fp8_total_derived} — cannot prove untouched")
        if _fp8_total_raw != _fp8_total_derived:
            raise AssertionError(f"FP8 coverage mismatch: raw {_fp8_total_raw} vs derived {_fp8_total_derived}")
        validate_no_fp8_drift(_raw_agg, _derived_agg)
        _fp8_verified = True
    else:
        # No derived shards yet — cannot prove, record false
        _fp8_verified = False
    derived_manifest: dict[str, Any] = {
        "schema": "prismaquant.dsv4_dual_basis_derived_manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_index_sha256": _cached_source_index_sha256(),
        "col_weights_sha256": _cached_col_weights_sha256(),
        "activation_cache_dir": str(ACT_ROOT.resolve()),
        "by_layer_dir": str(BY_LAYER.resolve()),
        "derive_tool_sha256": _cached_tool_sha256(),
        "module_shas": _cached_module_shas(),
        "context_stamp": cb_serialization_context_stamp(CTX_NVFP4_LDLQ, formats=[f"NVFP4_CB_K{k}" for k in NVFP4_RUNGS] + [f"FP8_CB_K{k}" for k in range(28, 39)]),
        "fused_activation_policy": FUSED_ACTIVATION_POLICY_V1,
        "fused_activation_order": "profile_order_concat_equal_member_samples",
        "ldlq_scope": "nvfp4",
        "nvfp4_rungs": list(NVFP4_RUNGS),
        "derived_shards": {},
        "projection_checkpoints_dir": str(DERIVED_CHECKPOINTS.resolve()),
        "dense_checkpoints_dir": str(DERIVED_DENSE_CHECKPOINTS.resolve()),
        "warm_state_dir": str(DERIVED_WARM.resolve()),
        "campaign_paths": {
            "RUN_ROOT": str(RUN_ROOT.resolve()),
            "RAW_SHARDS": str(RAW_SHARDS.resolve()),
            "RAW_MERGED": str(RAW_MERGED.resolve()),
            "SOURCE": str(SOURCE.resolve()),
            "BY_LAYER": str(BY_LAYER.resolve()),
            "COL_WEIGHTS": str(COL_WEIGHTS.resolve()),
            "ACT_ROOT": str(ACT_ROOT.resolve()),
            "DERIVED_ROOT": str(DERIVED_ROOT.resolve()),
            "allow_mixed_campaign_paths": bool(globals().get("_CAMPAIGN_MIX_ALLOW", False)),
            "mixed_sources": list(globals().get("_CAMPAIGN_MIX_SOURCES", [])),
        },
        "gate_telemetry": {
            "projection_gate_counts": dict(gate_counts),
            "gate_reasons": dict(gate_reasons),
            "derived_shard_nvfp4_gate_counts": dict(dense_gate_counts),
        },
        "source_lineage": {
            "raw_cost_source_counts": dict(_raw_cost_source_counts),
            "derived_nvfp4_cost_source_counts": dict(_derived_nvfp4_cost_source_counts),
            "fp8_source": "preserved_deep_equal_from_raw_merged" if _fp8_verified else "unverified_no_fp8_evidence",
            "fp8_proven_untouched": bool(_fp8_verified),
            "fp8_coverage_raw": _fp8_total_raw,
            "fp8_coverage_derived": _fp8_total_derived,
            "verified_scope": "nvfp4" if _fp8_verified else "none",
            "note": "FP8 dicts remain deep-equal to canonical raw_merged; no reinterpolation in this derive; historical cross-family interpolation is archived, future FP8 interpolation must use raw/no-LDLQ values never LDLQ plane",
        },
    }
    if DERIVED_SHARDS.is_dir():
        for p in sorted(DERIVED_SHARDS.glob("layer_*.pkl")):
            derived_manifest["derived_shards"][p.name] = sha256_file(p)
    atomic_json(DERIVED_ROOT / "derived_manifest.json", derived_manifest)

# ---------------------------------------------------------------------------
# Derived merged cost artifact: immutable source 109,058,144 bytes, 33,325 costs
# ---------------------------------------------------------------------------

def _expected_raw_merged_sha() -> str:
    # Narrowly explicit expected-SHA override for deliberately different campaign.
    # Precedence: CLI > env > canonical.
    cli = globals().get("_EXPECTED_RAW_MERGED_SHA_CLI", "")
    if cli and str(cli).strip():
        v = str(cli).strip().lower()
        if not __import__("re").fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"--expected-raw-merged-sha must be 64 hex chars, got {v!r}")
        return v
    env = os.environ.get("PQ_DERIVE_EXPECTED_RAW_MERGED_SHA", "").strip().lower()
    if env:
        if not __import__("re").fullmatch(r"[0-9a-f]{64}", env):
            raise ValueError(f"PQ_DERIVE_EXPECTED_RAW_MERGED_SHA must be 64 hex chars, got {env!r}")
        return env
    return "03bb8dac46744cccb03018f982196dc35f92e3553254fe5acf6ca49265127801"

def _validate_raw_merged(payload: Mapping[str, Any], path: Path) -> None:
    expected_size = 109058144
    expected_sha256 = _expected_raw_merged_sha()
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise AssertionError(f"raw merged size mismatch: expected {expected_size} got {actual_size} for {path}")
    if set(payload.keys()) != {"costs", "formats", "provenance", "meta"}:
        raise AssertionError(f"raw merged schema keys mismatch: {sorted(payload.keys())}")
    if len(payload.get("costs", {})) != 33325:
        raise AssertionError(f"raw merged costs count {len(payload.get('costs',{}))} != 33325")
    # Canonical SHA enforcement always — regardless of path/symlink/override.
    # Prevents same-size tampering. Only narrowly explicit expected-SHA param can override.
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise AssertionError(f"raw merged sha256 mismatch: expected {expected_sha256} got {actual_sha} for {path} (recorded expected {expected_sha256}, actual {actual_sha})")
    prov = payload.get("provenance", {})
    if not isinstance(prov, Mapping):
        raise AssertionError("raw merged provenance not mapping")

def build_derived_merged(require_all_43: bool = True) -> Path:
    """Build derived merged cost artifact from 43 derived layer shards.

    Deep-copy source merged payload, replace only NVFP4 entries from derived
    shards, require every FP8_CB dict deep-equal source, preserve other
    formats, update provenance to scope nvfp4 and add dual_basis block.
    Atomically write DERIVED_MERGED. Never patches source.
    """
    require_ldlq_gate_enabled()
    if not RAW_MERGED.is_file():
        raise FileNotFoundError(f"raw merged missing {RAW_MERGED}")
    raw_merged_payload = pickle.loads(RAW_MERGED.read_bytes())
    raw_merged_sha = sha256_file(RAW_MERGED)
    _validate_raw_merged(raw_merged_payload, RAW_MERGED)
    # Require all 43 derived shards exist and validate
    derived_shards = sorted(DERIVED_SHARDS.glob("layer_*.pkl"))
    if require_all_43:
        if len(derived_shards) != 43:
            raise AssertionError(f"derived merged requires 43 shards, found {len(derived_shards)}; partial layer commands only write shards/checkpoints/manifests, do not run full merge")
        # Also check each shard's content_key and schema
        for p in derived_shards:
            payload = pickle.loads(p.read_bytes())
            if payload.get("schema") != SCHEMA:
                raise AssertionError(f"derived shard {p.name} bad schema {payload.get('schema')}")
            if payload.get("content_key") != _sha(payload["identity"]):
                raise AssertionError(f"derived shard {p.name} content_key mismatch")
    else:
        if not derived_shards:
            raise AssertionError("no derived shards to merge")
    # Deep-copy source
    derived_merged = copy.deepcopy(raw_merged_payload)
    # Build lookup from derived shards: qname -> {fmt: entry}
    derived_costs: dict[str, dict[str, Any]] = {}
    for p in derived_shards:
        payload = pickle.loads(p.read_bytes())
        for qname, row in payload["costs"].items():
            if qname not in derived_costs:
                derived_costs[qname] = {}
            for fmt, ent in row.items():
                if str(fmt).startswith("NVFP4_CB"):
                    derived_costs[qname][fmt] = copy.deepcopy(ent)
    # Validate row/format/coverage: every raw row's NVFP4 set must match derived
    for qname, raw_row in raw_merged_payload["costs"].items():
        if qname not in derived_costs:
            raise AssertionError(f"{qname}: missing in derived shards (needed for merged)")
        raw_nvfp4 = {k for k in raw_row.keys() if str(k).startswith("NVFP4_CB")}
        derived_nvfp4 = {k for k in derived_costs[qname].keys() if str(k).startswith("NVFP4_CB")}
        if raw_nvfp4 != derived_nvfp4:
            raise AssertionError(f"{qname}: NVFP4 coverage mismatch raw {sorted(raw_nvfp4)} vs derived {sorted(derived_nvfp4)}")
        # Ensure no extra NVFP4 added
        extra = derived_nvfp4 - raw_nvfp4
        if extra:
            raise AssertionError(f"{qname}: derived adds extra NVFP4 {extra} absent from raw menu")
    # Replace only NVFP4 entries in deep-copied source
    for qname in derived_merged["costs"]:
        raw_row = raw_merged_payload["costs"][qname]
        derived_row_nvfp4 = derived_costs.get(qname, {})
        for fmt in list(derived_merged["costs"][qname].keys()):
            if str(fmt).startswith("NVFP4_CB"):
                # Replace with derived NVFP4 entry
                if fmt not in derived_row_nvfp4:
                    raise AssertionError(f"{qname} {fmt}: derived NVFP4 missing for replacement")
                derived_merged["costs"][qname][fmt] = copy.deepcopy(derived_row_nvfp4[fmt])
        # Require every FP8_CB dict deep-equal source (no drift)
        for fmt, val in raw_row.items():
            if str(fmt).startswith("FP8_CB"):
                if derived_merged["costs"][qname].get(fmt) != val:
                    raise AssertionError(f"{qname} {fmt}: FP8 drift in merged (must remain byte-identical to raw merged)")
        # Preserve other formats (BF16 etc) implicitly by not touching them; assert they deep-equal
        for fmt, val in raw_row.items():
            if not str(fmt).startswith("NVFP4_CB") and not str(fmt).startswith("FP8_CB"):
                if derived_merged["costs"][qname].get(fmt) != val:
                    raise AssertionError(f"{qname} {fmt}: non-NVFP4/non-FP8 format drift")
    # Update provenance to scope nvfp4 and add dual_basis block
    prov = derived_merged.get("provenance", {})
    # Preserve raw merged SHA and manifests
    raw_manifest_path = DERIVED_ROOT / "raw_plane_manifest.json"
    derived_manifest_path = DERIVED_ROOT / "derived_manifest.json"
    raw_manifest_sha = sha256_file(raw_manifest_path) if raw_manifest_path.is_file() else None
    derived_manifest_sha = sha256_file(derived_manifest_path) if derived_manifest_path.is_file() else None
    # Update cb_serialized_payload scope to nvfp4 if present
    new_prov = copy.deepcopy(prov)
    # Build ONE authoritative canonical CB stamp and use it identically top and
    # nested — validate_cb_render_provenance requires exact equality.
    # Keep all other render-identity bindings (col_weights, source) unchanged.
    _raw_stamp = new_prov.get("cb_serialized_payload") if isinstance(new_prov.get("cb_serialized_payload"), dict) else {}
    _raw_lattice_formats = list(_raw_stamp.get("lattice_codebook_sha256_by_format", {}).keys()) if isinstance(_raw_stamp, dict) else []
    if not _raw_lattice_formats:
        # Fallback to NVFP4+FP8 union
        _raw_lattice_formats = [f"NVFP4_CB_K{k}" for k in NVFP4_RUNGS] + [f"FP8_CB_K{k}" for k in range(28, 39)]
    canonical_formats = sorted(set(str(f) for f in _raw_lattice_formats))
    canonical_stamp = cb_serialization_context_stamp(CTX_NVFP4_LDLQ, formats=canonical_formats)
    canonical_stamp["ldlq_scope"] = "nvfp4"
    if "cb_serialized_payload" in new_prov and isinstance(new_prov["cb_serialized_payload"], dict):
        new_prov["cb_serialized_payload"] = copy.deepcopy(canonical_stamp)
    if "cb_render_identity" in new_prov and isinstance(new_prov["cb_render_identity"], dict):
        nested = new_prov["cb_render_identity"]
        if "cb_serialized_payload" in nested and isinstance(nested["cb_serialized_payload"], dict):
            # Use the SAME canonical stamp for nested — exact equality required.
            nested["cb_serialized_payload"] = copy.deepcopy(canonical_stamp)
            if "post_allocation_refinement" in nested:
                del nested["post_allocation_refinement"]
        if "post_allocation_refinement" in new_prov:
            del new_prov["post_allocation_refinement"]
    # Add dual_basis block — source lineage is fail-closed, no hard-coded flags.
    # Gate summaries: aggregate from derived merged (observed telemetry).
    gate_summary: dict[str, int] = {}
    for row in derived_merged["costs"].values():
        for fmt, ent in row.items():
            if str(fmt).startswith("NVFP4_CB"):
                if not isinstance(ent, Mapping):
                    raise AssertionError(f"dual_basis gate summary malformed entry {fmt}")
                g = str(ent.get("gate", "unknown"))
                gate_summary[g] = gate_summary.get(g, 0) + 1
    # Count cost_source for lineage without fabricating FP8 provenance — fail-closed on malformed.
    _raw_counts: dict[str, int] = {}
    for row in raw_merged_payload["costs"].values():
        if not isinstance(row, Mapping):
            raise AssertionError("raw_merged costs row not mapping")
        for fmt, ent in row.items():
            if str(fmt).startswith("NVFP4_CB"):
                if not isinstance(ent, Mapping):
                    raise AssertionError(f"raw_merged NVFP4 entry {fmt} not mapping")
                if "cost_source" in ent:
                    cs = str(ent.get("cost_source"))
                    _raw_counts[cs] = _raw_counts.get(cs, 0) + 1
    _nvfp4_counts: dict[str, int] = {}
    for row in derived_merged["costs"].values():
        if not isinstance(row, Mapping):
            raise AssertionError("derived_merged costs row not mapping")
        for fmt, ent in row.items():
            if str(fmt).startswith("NVFP4_CB"):
                if not isinstance(ent, Mapping):
                    raise AssertionError(f"derived_merged NVFP4 entry {fmt} not mapping")
                cs = str(ent.get("cost_source", "unknown"))
                _nvfp4_counts[cs] = _nvfp4_counts.get(cs, 0) + 1
    # FP8 deep equality already validated above via the per-qname loop and validate_no_fp8_drift;
    # require nonzero equal coverage and deep equality before setting verified.
    _fp8_raw_cnt = sum(1 for row in raw_merged_payload["costs"].values() for k in row.keys() if str(k).startswith("FP8_CB"))
    _fp8_derived_cnt = sum(1 for row in derived_merged["costs"].values() for k in row.keys() if str(k).startswith("FP8_CB"))
    if _fp8_raw_cnt == 0 or _fp8_derived_cnt == 0:
        raise AssertionError(f"FP8 coverage empty: raw {_fp8_raw_cnt} derived {_fp8_derived_cnt} — cannot prove untouched")
    if _fp8_raw_cnt != _fp8_derived_cnt:
        raise AssertionError(f"FP8 coverage mismatch: raw {_fp8_raw_cnt} vs derived {_fp8_derived_cnt}")
    # Deep equality already validated via per-qname loop; also run canonical validator
    validate_no_fp8_drift(raw_merged_payload["costs"], derived_merged["costs"])
    _fp8_verified_merged = True
    new_prov["dual_basis"] = {
        "raw_merged_sha256": raw_merged_sha,
        "raw_merged_path": str(RAW_MERGED.resolve()),
        "raw_manifest_sha256": raw_manifest_sha,
        "derived_manifest_sha256": derived_manifest_sha,
        "activation_cache_dir": str(ACT_ROOT.resolve()),
        "col_weights_sha256": _cached_col_weights_sha256(),
        "source_index_sha256": _cached_source_index_sha256(),
        "derive_tool_sha256": _cached_tool_sha256(),
        "module_shas": _cached_module_shas(),
        "gate_summary": dict(gate_summary),
        "nvfp4_cost_source_counts": dict(_nvfp4_counts),
        "raw_nvfp4_cost_source_counts": dict(_raw_counts),
        "fp8_source": "preserved_deep_equal_from_raw_merged" if _fp8_verified_merged else "unverified",
        "fp8_proven_untouched": bool(_fp8_verified_merged),
        "fp8_coverage_raw": _fp8_raw_cnt,
        "fp8_coverage_derived": _fp8_derived_cnt,
        "nvfp4_scope": "nvfp4",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "FP8 dicts remain deep-equal to canonical raw_merged; no reinterpolation in this derive; historical cross-family interpolation is archived, future FP8 interpolation must use raw/no-LDLQ values never LDLQ plane",
    }
    # Ensure ldlq_scope at top-level provenance also set
    new_prov["ldlq_scope"] = "nvfp4"
    derived_merged["provenance"] = new_prov
    # Also update top-level formats to ensure still consistent (should already contain NVFP4 etc)
    # --- Real validators: fail-closed on the in-memory payload before any atomic write ---
    from prismaquant.nvfp4_cb_footprint import (
        validate_cb_serialization_context_stamp,
        validate_cb_cost_provenance,
    )
    from prismaquant.production_weight_cache import validate_cb_render_provenance

    # Validate top-level CB serialization stamp matches the declared context.
    # Use the actual CB format menu from the payload's top-level stamp.
    _cb_formats = [k for k in derived_merged.get("formats", []) if str(k).startswith("NVFP4_CB") or str(k).startswith("FP8_CB")]
    if _cb_formats:
        validate_cb_cost_provenance(derived_merged, _cb_formats, context=CTX_NVFP4_LDLQ, where="derived merged cb_cost")
        validate_cb_serialization_context_stamp(derived_merged["provenance"].get("cb_serialized_payload"), CTX_NVFP4_LDLQ, where="derived merged cb_serialized_payload")
        # Nested render identity: validate with the real render provenance validator (not just stamp).
        _render_ident = derived_merged["provenance"].get("cb_render_identity")
        if isinstance(_render_ident, Mapping) and isinstance(_render_ident.get("cb_serialized_payload"), Mapping):
            validate_cb_serialization_context_stamp(_render_ident.get("cb_serialized_payload"), CTX_NVFP4_LDLQ, where="derived merged cb_render_identity")
        # Full render provenance — checks value-bearing identity, source/col-weights bindings, and top-level match.
        # This is the real production validator, not a superficial name check.
        validate_cb_render_provenance(derived_merged, expected_context=CTX_NVFP4_LDLQ, where="derived merged cb_render")
    # Atomically write
    DERIVED_MERGED.parent.mkdir(parents=True, exist_ok=True)
    tmp = DERIVED_MERGED.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(derived_merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, DERIVED_MERGED)
    # Verify not patched source
    assert sha256_file(RAW_MERGED) == raw_merged_sha
    assert sha256_file(DERIVED_MERGED) != raw_merged_sha
    # Row/format/coverage validation already done; final sanity
    assert len(pickle.loads(DERIVED_MERGED.read_bytes())["costs"]) == 33325
    return DERIVED_MERGED

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one packed projection/rung, 256 results, no shard write, no warm state")
    parser.add_argument("--layer", type=int, default=0)
    # Projection accepts either packed parent (gate_up_proj/down_proj) or leaf (gate_proj/up_proj/down_proj) for backward compat; leaf is mapped to packed parent.
    parser.add_argument("--projection", type=str, default="down_proj")
    parser.add_argument("--rung", type=int, default=12, choices=list(NVFP4_RUNGS))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--write-warm-state", action="store_true", help="allow smoke to write warm state (must be false in smoke)")
    parser.add_argument("--build-merged", action="store_true", help="build derived merged artifact (requires all 43 shards)")
    parser.add_argument("--merged-no-require-all", action="store_true", help="allow merged build without 43 shards (testing only)")
    parser.add_argument("--run-root", type=str, default=None, help="override RUN_ROOT (default campaign pinned; env PQ_DERIVE_RUN_ROOT also)")
    parser.add_argument("--derived-root", type=str, default=None, help="override DERIVED_ROOT (env PQ_DERIVE_DERIVED_ROOT)")
    parser.add_argument("--source", type=str, default=None, help="override SOURCE model dir (env PQ_DERIVE_SOURCE)")
    parser.add_argument("--by-layer", type=str, default=None, help="override BY_LAYER dir (env PQ_DERIVE_BY_LAYER)")
    parser.add_argument("--col-weights", type=str, dest="col_weights", default=None, help="override COL_WEIGHTS pickle (env PQ_DERIVE_COL_WEIGHTS)")
    parser.add_argument("--act-root", type=str, default=None, help="override ACT_ROOT (env PQ_DERIVE_ACT_ROOT)")
    parser.add_argument("--raw-merged", type=str, default=None, help="override RAW_MERGED pickle (env PQ_DERIVE_RAW_MERGED)")
    parser.add_argument("--expected-raw-merged-sha", type=str, default=None, dest="expected_raw_merged_sha", help="narrowly explicit expected SHA for raw merged (env PQ_DERIVE_EXPECTED_RAW_MERGED_SHA); required to use deliberately different campaign")
    parser.add_argument("--allow-mixed-campaign-paths", action="store_true", help="explicit opt-in to allow --run-root override while related defaults remain pinned (records opt-in in manifest); without this, mixing fails closed")
    args = parser.parse_args()
    # Apply CLI overrides for campaign paths (explicit, no global drift beyond this invocation)
    global RUN_ROOT, DERIVED_ROOT, DERIVED_SHARDS, DERIVED_WARM, DERIVED_CHECKPOINTS, DERIVED_DENSE_CHECKPOINTS, DERIVED_RAW_PLANE, SOURCE, BY_LAYER, COL_WEIGHTS, ACT_ROOT, RAW_SHARDS, RAW_MERGED, DERIVED_MERGED
    if args.run_root is not None:
        RUN_ROOT = Path(args.run_root)
        RAW_SHARDS = RUN_ROOT / "shards"
    if args.derived_root is not None:
        DERIVED_ROOT = Path(args.derived_root)
        DERIVED_SHARDS = DERIVED_ROOT / "shards"
        DERIVED_WARM = DERIVED_ROOT / "warm-state-nvfp4"
        DERIVED_CHECKPOINTS = DERIVED_ROOT / "projection_checkpoints"
        DERIVED_DENSE_CHECKPOINTS = DERIVED_ROOT / "dense_checkpoints"
        DERIVED_RAW_PLANE = DERIVED_ROOT / "raw_plane"
        DERIVED_MERGED = DERIVED_ROOT / "cost_merged_derived.pkl"
    if args.source is not None:
        SOURCE = Path(args.source)
    if args.by_layer is not None:
        BY_LAYER = Path(args.by_layer)
    if args.col_weights is not None:
        COL_WEIGHTS = Path(args.col_weights)
    if args.act_root is not None:
        ACT_ROOT = Path(args.act_root)
    # Determine whether RAW_MERGED was explicitly set via env or CLI.
    _raw_merged_env_explicit = bool(os.environ.get("PQ_DERIVE_RAW_MERGED") and str(os.environ.get("PQ_DERIVE_RAW_MERGED")).strip())
    _raw_merged_cli_explicit = args.raw_merged is not None
    if args.raw_merged is not None:
        RAW_MERGED = Path(args.raw_merged)
    elif (args.run_root is not None or bool(os.environ.get("PQ_DERIVE_RUN_ROOT", "").strip())) and not _raw_merged_env_explicit and not _raw_merged_cli_explicit:
        # RUN_ROOT env or CLI without explicit RAW_MERGED must derive RAW_MERGED from resolved RUN_ROOT
        # Path source is derived_from_run_root
        RAW_MERGED = RUN_ROOT / "burn-afast" / "cost_merged.pkl"
        # Update path source tracking for manifest
        if "RAW_MERGED" in globals().get("_CAMPAIGN_PATH_SOURCES", {}):
            globals()["_CAMPAIGN_PATH_SOURCES"]["RAW_MERGED"] = "derived_from_run_root"
    # Narrowly explicit expected SHA for different campaign
    globals()["_EXPECTED_RAW_MERGED_SHA_CLI"] = str(args.expected_raw_merged_sha or "").strip().lower()
    # Clear cached SHAs that depend on overridden paths so manifest reflects override.
    # Also clear the 466 MB col-weights payload cache and planning cache.
    # CLI --source is authoritative and overrides PQ_DERIVE_SOURCE env;
    # import-time env is only the fallback when CLI is absent. No main-time
    # env re-override: if CLI set SOURCE, env is ignored for planning.
    _cached_col_weights_sha256.cache_clear()
    _load_col_weights_cached.cache_clear()
    _cached_source_index_sha256.cache_clear()
    _cached_tool_sha256.cache_clear()
    _cached_module_shas.cache_clear()
    _clear_packed_planning_cache()
    # Campaign mixing guard: fail-closed unless explicit opt-in.
    # Detect RUN_ROOT overrides from CLI or env (resolved paths + provenance).
    _allow_mixed = bool(args.allow_mixed_campaign_paths or str(os.environ.get("PQ_DERIVE_ALLOW_MIXED_CAMPAIGN", "")).strip().lower() in {"1", "true", "yes", "on"})
    globals()["_CAMPAIGN_MIX_ALLOW"] = _allow_mixed
    globals()["_CAMPAIGN_MIX_SOURCES"] = []
    # Determine RUN_ROOT provenance
    _run_root_cli = args.run_root is not None
    _run_root_env = bool(os.environ.get("PQ_DERIVE_RUN_ROOT", "").strip())
    _run_root_overridden = _run_root_cli or _run_root_env
    # Also handle DERIVED_ROOT overrides for completeness
    _derived_root_cli = args.derived_root is not None
    _derived_root_env = bool(os.environ.get("PQ_DERIVE_DERIVED_ROOT", "").strip())
    # Record every resolved path + source for manifest
    def _path_source(cli: bool, env: bool) -> str:
        if cli:
            return "cli"
        if env:
            return "env"
        return "default"
    globals()["_CAMPAIGN_PATH_SOURCES"] = {
        "RUN_ROOT": _path_source(_run_root_cli, _run_root_env),
        "RAW_MERGED": _path_source(bool(args.raw_merged), _raw_merged_env_explicit),
        "SOURCE": _path_source(bool(args.source), bool(os.environ.get("PQ_DERIVE_SOURCE", "").strip())),
        "BY_LAYER": _path_source(bool(args.by_layer), bool(os.environ.get("PQ_DERIVE_BY_LAYER", "").strip())),
        "COL_WEIGHTS": _path_source(bool(args.col_weights), bool(os.environ.get("PQ_DERIVE_COL_WEIGHTS", "").strip())),
        "ACT_ROOT": _path_source(bool(args.act_root), bool(os.environ.get("PQ_DERIVE_ACT_ROOT", "").strip())),
        "DERIVED_ROOT": _path_source(_derived_root_cli, _derived_root_env),
    }
    if _run_root_overridden or args.derived_root is not None or _derived_root_env:
        _default_source = Path("/home/rob/dq-runs/dsv4-flash-0731/source").resolve()
        _default_by_layer = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts-mxfp4/probe-k12k18/by-layer").resolve()
        _default_col = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts-mxfp4/cb_col_weights.pkl").resolve()
        _default_act = Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/act").resolve()
        _mix_sources = []
        # Compare resolved paths to detect mixing (env or CLI)
        if SOURCE.resolve() == _default_source and not (args.source or os.environ.get("PQ_DERIVE_SOURCE", "").strip()):
            # SOURCE still default while RUN_ROOT env/cli overridden -> mixing
            if _run_root_overridden:
                _mix_sources.append("SOURCE")
        # More precise: if RUN_ROOT overridden (cli or env) and any of the related defaults still pinned, it's mixing
        if _run_root_overridden:
            if not (args.source or os.environ.get("PQ_DERIVE_SOURCE", "").strip()) and SOURCE.resolve() == _default_source:
                if "SOURCE" not in _mix_sources:
                    _mix_sources.append("SOURCE")
            if not (args.by_layer or os.environ.get("PQ_DERIVE_BY_LAYER", "").strip()) and BY_LAYER.resolve() == _default_by_layer:
                _mix_sources.append("BY_LAYER")
            if not (args.col_weights or os.environ.get("PQ_DERIVE_COL_WEIGHTS", "").strip()) and COL_WEIGHTS.resolve() == _default_col:
                _mix_sources.append("COL_WEIGHTS")
            if not (args.act_root or os.environ.get("PQ_DERIVE_ACT_ROOT", "").strip()) and ACT_ROOT.resolve() == _default_act:
                _mix_sources.append("ACT_ROOT")
        globals()["_CAMPAIGN_MIX_SOURCES"] = list(_mix_sources)
        if _mix_sources and not _allow_mixed:
            raise SystemExit(
                f"[derive] ERROR: RUN_ROOT overridden (cli={_run_root_cli} env={_run_root_env}) but related defaults still pinned to old campaign: {', '.join(_mix_sources)}; "
                "refusing to mix campaigns without explicit --allow-mixed-campaign-paths or PQ_DERIVE_ALLOW_MIXED_CAMPAIGN=1"
            )
        if _mix_sources and _allow_mixed:
            print(f"[derive] WARNING: RUN_ROOT overridden but related defaults still pinned to old campaign: {', '.join(_mix_sources)}; allowed via explicit opt-in, manifest will record the mix", flush=True)
    if args.smoke:
        if args.write_warm_state:
            raise ValueError("smoke must not write warm state: --write-warm-state is forbidden with --smoke")
        # Smoke no-writes static validation: snapshot ALL derived locations before/after
        # including DERIVED_RAW_PLANE and tmp files.
        def _snap(p: Path):
            return set(p.glob("*.pkl")) if p.is_dir() else set()
        def _snap_all(p: Path):
            return set(p.rglob("*")) if p.is_dir() else set()
        before_shards = _snap(DERIVED_SHARDS)
        before_warm = _snap_all(DERIVED_WARM)
        before_ckpt = _snap(DERIVED_CHECKPOINTS)
        before_dense_ckpt = _snap(DERIVED_DENSE_CHECKPOINTS)
        before_raw_plane = _snap(DERIVED_RAW_PLANE)
        # Also snapshot any tmp files in derived root
        before_tmp = set(DERIVED_ROOT.glob("*.tmp")) if DERIVED_ROOT.is_dir() else set()
        device = torch.device("cuda")
        require_cuda(device)
        res = derive_one_projection_rung(args.layer, args.projection, args.rung, device, write_warm_state=False)
        after_shards = _snap(DERIVED_SHARDS)
        after_warm = _snap_all(DERIVED_WARM)
        after_ckpt = _snap(DERIVED_CHECKPOINTS)
        after_dense_ckpt = _snap(DERIVED_DENSE_CHECKPOINTS)
        after_raw_plane = _snap(DERIVED_RAW_PLANE)
        after_tmp = set(DERIVED_ROOT.glob("*.tmp")) if DERIVED_ROOT.is_dir() else set()
        if before_shards != after_shards or before_warm != after_warm or before_ckpt != after_ckpt or before_dense_ckpt != after_dense_ckpt or before_raw_plane != after_raw_plane or before_tmp != after_tmp:
            raise AssertionError(
                f"smoke wrote derived files: shards {after_shards - before_shards} warm {after_warm - before_warm} "
                f"ckpt {after_ckpt - before_ckpt} dense_ckpt {after_dense_ckpt - before_dense_ckpt} "
                f"raw_plane {after_raw_plane - before_raw_plane} tmp {after_tmp - before_tmp}"
            )
        print(json.dumps({
            "layer": res["layer"], "projection": res["projection"], "rung": res["rung"],
            "weight_mse_per_expert": res["weight_mse_per_expert"][:3],
            "output_mse_per_expert": res["output_mse_per_expert"][:3],
            "rel_output_mse_per_expert": res["rel_output_mse_per_expert"][:3],
            "n_activation_rows_per_expert": res["n_activation_rows_per_expert"][:3],
            "count": len(res["weight_mse_per_expert"]),
            "gate": res["gate_info"].get("gate"),
            "per_expert_kept": res["gate_info"].get("per_expert_kept"),
        }, indent=2))
        print("[smoke] passed 256 results with output_mse/rel/n_rows and exact gate_info", flush=True)
        return 0
    if args.build_merged:
        require_all = not args.merged_no_require_all
        out = build_derived_merged(require_all_43=require_all)
        print(f"derived merged written {out} size {out.stat().st_size}", flush=True)
        return 0
    if args.layers is not None:
        device = torch.device("cuda")
        require_cuda(device)
        for lyr in args.layers:
            derive_layer_full(lyr, device)
        build_manifests()
        print(f"derived {len(args.layers)} layers", flush=True)
        return 0
    print("full derive not executed (use --smoke or --layers or --build-merged)", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
