"""Empirical routed-MoE expert costs for the AURA hybrid recipe.

AURA's smooth per-Linear cost is route-flip-blind on routed experts (Step A,
2026-06-29: Spearman drops 0.45->0.35 under faithful dW; predicted NVFP4/FP8
ratios 2-49x vs measured 1.1-1.5x), so expert costs are MEASURED, not
modeled: per MoE layer the serving unit = all profile-coupled routed expert
tensors (packed Parameters or per-expert Linears; they must share one format —
vLLM FusedMoE constraint), and the unit cost of a format is the end-to-end
mean-token KL(BF16 || unit-quantized) with everything else left at source
precision. The unit KL is split across the member tensors proportionally to
n_params so the allocator's per-member aggregation charges it exactly once.

The quantizer is plain RTN ``quantize_dequantize`` from the format registry —
the same estimator contract as the AURA non-expert cost (RTN-vs-GPTQ dW is a
wash at fp4 and RTN is *better* at fp8 on the served 27B A/B); the deliberate
GPTQ render happens later in the production cache, and real-KL frontier
selection (M4) judges the actual rendered bytes.

FP8 stays IN the expert menu (standing decision 2026-06-29): it is
Pareto-dominated on routed experts (~1.3x lower KL for 2x bits), and the
right place for that fact to act is the allocator's DP + the real-KL
frontier — not a hardcoded ban here.

This module also performs the hybrid merge that previously lived as a
one-off in /home/rob/dq-runs/aura-35b/: ``--merge-base`` unions these expert
rows into an AURA (non-expert) cost payload, and ``--backfill-base`` copies
rows for any name the merged payload still lacks (MTP / visual sidecars the
AURA pass never sees) from the baseline incremental cost.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.cb_layout import (
    VEC_DIM,
    parse_format_name,
    subtable_bit_widths,
)
from prismaquant.cb_ladder_cross_family import (
    PROVENANCE_KEY as CROSS_FAMILY_PROVENANCE_KEY,
)
from prismaquant.cb_ladder_cross_family import verdict_from_unit_kls
from prismaquant.emu_forward_kl import _qdq_accepts_col_weights
from prismaquant.nvfp4_cb_footprint import (
    cb_cost_provenance,
    cb_quantize_dequantize_for_context,
    cb_serialization_context_from_env,
    validate_cb_cost_provenance,
)
from prismaquant.routed_experts import (
    UnpackedExpertLinear,
    profile_declared_routed_expert_targets,
    profile_declared_unpacked_expert_linears,
    resolve_routed_expert_profile,
)

SCHEMA = "prismaquant.expert_empirical_cost.v1"
PASSTHROUGH_FORMATS = {"BF16", "FP8_SOURCE"}
# CB families render the WHOLE stack in one qdq call (the export convention:
# fp4 derives one per-stack global; fp8 per-row scales) — measured render ==
# shipped bytes, never chunked (moe_cb_design.md §3).
_CB_FAMILIES = {"nvfp4_cb", "fp8_cb"}
# Both CB families carry k-rung ladders (k index bits per 8-weight vector).
# Ladders are PER (family, mode) — NVFP4_CB_K and FP8_CB_K are
# different grids/codings and never share one log-linear fit. The RD-law fit
# is holdout-gated per unit, so admitting a family costs nothing when the law
# fails there (falls back to full measurement).
# RD-law ladder interpolation (moe_cb_design.md §3.4): D(k) = C * 2^(-k/4),
# validated +-3% on weighted-recon at 0.6B but UNPROVEN on unit-KL — so it is
# opt-in and holdout-gated PER UNIT (a failed holdout falls back to full
# measurement for that unit).
_LADDER_SLOPE_BITS = 0.25


def _log(msg: str) -> None:
    print(f"[expert-cost {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _canon_formats(formats: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for raw in formats:
        name = fr.canonical_format_name(str(raw).strip())
        if name and name not in seen:
            seen.append(name)
    return seen


_CALIB_BATCH_ENV = "PRISMAQUANT_EXPERT_CALIB_BATCH"


def _calib_batch() -> int:
    """Calibration sequences per forward. Default 1 preserves the historical
    per-sequence numerics exactly; >1 batches independent windows (semantics
    identical, per-position arithmetic may differ at reassociation level —
    baseline and quantized arms always use the SAME batching, so the KL
    comparison stays internally consistent). Becomes the dominant-wall knob
    once expert sampling shrinks the encode side."""
    return max(1, int(os.environ.get(_CALIB_BATCH_ENV, "1") or 1))


@torch.no_grad()
def _baseline_logprobs(
    model, calib_ids: torch.Tensor,
    capture_units: Sequence[tuple[str, object]] | None = None,
    capture_rows: int = 4096,
    *,
    forward_model=None,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict]:
    """Baseline log-probs; optionally capture each expert module's INPUT rows
    during the same forwards (bounded to ``capture_rows`` per unit) for the
    imatrix replay — no separate pass, no activation-cache dependency."""
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []
    if capture_units:
        def _mk_hook(qn):
            def _hook(_mod, args, _kwargs, _out):
                xs = captured.setdefault(qn, [])
                have = sum(t.shape[0] for t in xs)
                if have >= capture_rows:
                    return
                x = args[0].detach()
                x = x.reshape(-1, x.shape[-1])
                xs.append(x[: capture_rows - have].cpu())
            return _hook
        for qn, mod in capture_units:
            handles.append(mod.register_forward_hook(
                _mk_hook(qn), with_kwargs=True))
    try:
        out = []
        bs = _calib_batch()
        n_total = calib_ids.shape[0]
        t0 = time.time()
        for i in range(0, n_total, bs):
            logits = (forward_model or model)(
                calib_ids[i:i + bs]
            ).logits.float()
            out.append(F.log_softmax(logits, dim=-1).cpu())
            done = min(i + bs, n_total)
            dt = time.time() - t0
            tps = done * calib_ids.shape[1] / max(dt, 1e-9)
            _log(f"baseline forward {done}/{n_total} windows "
                 f"(batch={bs}, {dt:.0f}s elapsed, {tps:.0f} tok/s)")
    finally:
        for h in handles:
            h.remove()
    if capture_units is None:
        return out
    unit_x = {qn: torch.cat(xs, dim=0) for qn, xs in captured.items() if xs}
    return out, unit_x


@torch.no_grad()
def _replay_down_proj_col_weights(
    mod, parent_mod, router, X: torch.Tensor,
) -> torch.Tensor:
    """Per-expert down_proj imatrix ``(E, 1, inter)`` by replaying the routed
    forward on captured module inputs X: route -> per-expert gate_up ->
    activation -> intermediate; pool mean-square over that expert's routed
    tokens. down_proj's input (the per-expert intermediate) is never
    activation-cached — the packed-expert hook sees only the MODULE input —
    so this replay is the only faithful source (the same reason
    measure_quant_cost leaves down_proj unweighted in its pooled path).
    Experts with no routed tokens in the capture get the mean of the routed
    experts' vectors (a neutral prior, recorded by the caller)."""
    from prismaquant.measure_quant_cost import _packed_router_topk

    gate_up = mod.gate_up_proj
    E = int(gate_up.shape[0])
    inter = int(gate_up.shape[1]) // 2
    dev = gate_up.device
    Xd = X.to(device=dev, dtype=gate_up.dtype)
    act_fn = getattr(mod, "act_fn", F.silu)
    route_fn = getattr(parent_mod, "route_tokens_to_experts", None)
    if callable(route_fn):
        top_k_index, _tw = route_fn(router(Xd))
    else:
        top_k_index, _tw = _packed_router_topk(
            router, Xd, e_score_correction_bias=getattr(
                parent_mod, "e_score_correction_bias", None),
            expert_bias=getattr(parent_mod, "expert_bias", None))
    out = torch.zeros(E, inter, dtype=torch.float32, device=dev)
    hit = torch.zeros(E, dtype=torch.bool)
    for e in range(E):
        tok = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
        if tok.numel() == 0:
            continue
        g, u = F.linear(Xd[tok], gate_up[e]).chunk(2, dim=-1)
        inter_act = (act_fn(g) * u).float()
        out[e] = inter_act.pow(2).mean(dim=0)
        hit[e] = True
    if bool(hit.any()) and not bool(hit.all()):
        out[~hit] = out[hit].mean(dim=0)
    elif not bool(hit.any()):
        out[:] = 1.0
    return out.reshape(E, 1, inter).cpu()


@torch.no_grad()
def ensure_unit_col_weights(
    model, units, col_weights: dict, unit_x: Mapping[str, torch.Tensor],
) -> list[str]:
    """Fill missing packed-expert col_weights entries in place.

    gate_up_proj: pooled module-input second moment (identical op to the
    exporter's builder — full rows, fp32, mean over dim 0).
    down_proj: the per-expert intermediate replay above.
    Returns the names added (caller persists them back to the shared
    col-weights pickle so the EXPORTER ships the same weighting — the
    lockstep contract)."""
    from prismaquant.measure_quant_cost import (
        _packed_experts_parent_module,
        _packed_experts_router,
    )
    added: list[str] = []
    for qn, mod in units:
        X = unit_x.get(qn)
        gu_name, dn_name = f"{qn}.gate_up_proj", f"{qn}.down_proj"
        if gu_name not in col_weights:
            if X is None:
                raise ValueError(f"{qn}: no captured input rows for the "
                                 f"gate_up imatrix (unit never routed?)")
            col_weights[gu_name] = (
                X.float().pow(2).mean(dim=0).reshape(1, 1, -1))
            added.append(gu_name)
        if dn_name not in col_weights and hasattr(mod, "down_proj"):
            if X is None:
                raise ValueError(f"{qn}: no captured input rows for the "
                                 f"down_proj imatrix replay")
            parent = _packed_experts_parent_module(model, qn)
            router = _packed_experts_router(parent)
            if router is None:
                raise ValueError(f"{qn}: no router found for the down_proj "
                                 f"imatrix replay")
            col_weights[dn_name] = _replay_down_proj_col_weights(
                mod, parent, router, X)
            added.append(dn_name)
    return added


@torch.no_grad()
def _expert_sample_idx(num_experts: int, sample: int) -> torch.Tensor | None:
    """Deterministic stratified expert subsample (the local cost path's
    linspace pattern — even coverage of the expert index range)."""
    if sample <= 0 or num_experts <= sample:
        return None
    return torch.linspace(0, num_experts - 1, sample).round().long().unique()


def _quantize_unit_inplace(
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
) -> None:
    """Render every member of one expert serving unit in-place in ``fmt``.

    CB families use the imatrix-WEIGHTED VQ render on the whole stack (the
    exporter convention — measuring an unweighted render while the exporter
    ships weighted bytes is the rendering-confound class, so a CB format
    with no col_weights entry for a member hard-fails; the encode tier is
    inherited from PRISMAQUANT_CB_ENCODE_TIER via the registry closure).

    ``sample_idx`` (expert subsampling): quantize ONLY those expert slices,
    leaving the rest BF16 — the caller extrapolates the partial unit KL.
    Each sampled expert's render is identical to its full-stack render (the
    CB encode is per-expert-row independent; per-expert col_weights slices
    keep the weighted-render contract), so sampling changes COVERAGE, never
    the bytes measured for a covered expert.
    """
    spec = fr.get_format(fmt)
    qdq = spec.quantize_dequantize
    if spec.family in _CB_FAMILIES:
        context = cb_serialization_context_from_env()

        def qdq(weight, col_weights=None, *, qname=None):
            return cb_quantize_dequantize_for_context(
                spec,
                weight,
                context=context,
                qname=qname,
                col_weights=col_weights,
            )
    weighted = _qdq_accepts_col_weights(spec)
    for pn in param_names:
        w = getattr(mod, pn).data
        full = f"{unit_qname}.{pn}" if unit_qname else pn
        if spec.family in _CB_FAMILIES:
            cw = (col_weights or {}).get(full)
            if weighted and cw is None:
                raise ValueError(
                    f"{full}: CB format {fmt} needs a col_weights entry — "
                    f"the deliberate CB render is imatrix-weighted; an "
                    f"unweighted unit-KL would measure bytes the exporter "
                    f"never ships (pass --col-weights)")
            cw_dev = cw.to(w.device)
            if sample_idx is not None:
                idx = sample_idx.to(w.device)
                cw_s = (cw_dev[idx] if cw_dev.ndim >= 3
                        and cw_dev.shape[0] == w.shape[0] else cw_dev)
                w[idx] = qdq(
                    w[idx].float(), col_weights=cw_s, qname=full
                ).to(w.dtype)
            else:
                w.copy_(qdq(
                    w.float(), col_weights=cw_dev, qname=full
                ).to(w.dtype))
        elif spec.family == "nv":
            # NV formats derive one per-TENSOR global scale from
            # whatever slice they are given, while export ships one
            # global PER EXPERT. Chunk-batching would share a global
            # across the chunk and make the measured KL depend on the
            # --expert-chunk knob; quantize per expert slice instead
            # (mirrors measure_quant_cost._batched_quantize, which does
            # the per-slice loop for exactly this reason).
            experts = (sample_idx.tolist() if sample_idx is not None
                       else range(w.shape[0]))
            for e in experts:
                w[e] = qdq(w[e].float()).to(w.dtype)
        else:
            # Scale-local formats are chunk-invariant, so batching is
            # safe: FP8_E4M3/FP8_E5M2 reshape to (-1, in) and scale each
            # output row independently (fp8_dynamic_weight_qdq), and
            # group/block-scaled formats (MX) never cross the expert
            # boundary within a row.
            if sample_idx is not None:
                idx = sample_idx.to(w.device)
                w[idx] = qdq(w[idx].float()).to(w.dtype)
            else:
                for e in range(0, w.shape[0], expert_chunk):
                    w[e:e + expert_chunk] = qdq(
                        w[e:e + expert_chunk].float()).to(w.dtype)


@torch.no_grad()
def _unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
    per_window: bool = False,
    forward_model=None,
) -> float | tuple[float, list[float]]:
    """Mean-token KL(BF16 || model-with-this-unit-quantized).

    With ``sample_idx``, only those expert slices are quantized (and cloned
    for restore) — the caller owns the extrapolation to the full unit.

    With ``per_window`` also returns the per-calibration-window mean KLs the
    aggregate is built from — free (the loop already runs window by window)
    and the only between-draw noise datum either cost chain has. The CB
    ladder's holdout gate derives its tolerance from it rather than from a
    bare constant (``_cb_ladder_holdout_tol``)."""
    if sample_idx is None:
        originals = {pn: getattr(mod, pn).data.clone() for pn in param_names}
    else:
        originals = {pn: getattr(mod, pn).data[
            sample_idx.to(getattr(mod, pn).device)].clone()
            for pn in param_names}
    try:
        _quantize_unit_inplace(
            mod, param_names, fmt, expert_chunk=expert_chunk,
            col_weights=col_weights, unit_qname=unit_qname,
            sample_idx=sample_idx)
        total = 0.0
        n_tok = 0
        windows: list[float] = []
        bs = _calib_batch()
        for bi, i in enumerate(range(0, calib_ids.shape[0], bs)):
            lp = F.log_softmax(
                (forward_model or model)(
                    calib_ids[i:i + bs]
                ).logits.float(), -1)
            bl = baseline[bi].to(lp.device)
            kl = (bl.exp() * (bl - lp)).sum(-1)
            wsum = float(kl.sum().item())
            total += wsum
            n_tok += kl.numel()
            if per_window:
                windows.append(wsum / max(kl.numel(), 1))
        mean = total / max(n_tok, 1)
        return (mean, windows) if per_window else mean
    finally:
        for pn in param_names:
            w = getattr(mod, pn).data
            if sample_idx is None:
                w.copy_(originals[pn])
            else:
                w[sample_idx.to(w.device)] = originals[pn]


class _UnpackedExpertUnit(NamedTuple):
    """One routed serving unit backed by per-expert ``nn.Linear`` rows."""

    qname: str
    group_key: str
    members: tuple[UnpackedExpertLinear, ...]
    roles: tuple[tuple[str, tuple[str, ...]], ...]
    num_experts: int
    members_by_target: dict[str, dict[tuple[str, int], str]]


def _expert_profile_call(profile, accessor: str, *args):
    method = getattr(profile, accessor, None)
    if not callable(method):
        raise RuntimeError(
            f"profile {type(profile).__name__} cannot render routed experts: "
            f"missing callable {accessor}()"
        )
    try:
        return method(*args)
    except Exception as exc:
        raise RuntimeError(
            f"profile {type(profile).__name__} could not determine routed-"
            f"expert layout via {accessor}()"
        ) from exc


def _unpacked_expert_units(model, profile) -> list[_UnpackedExpertUnit]:
    """Validate and group profile-declared per-expert Linear units.

    Classification already happened through ``packed_expert_format_group``.
    The remaining profile accessors declare how those physical rows form the
    packed serving tensors used by the quantizer/exporter.  Missing projection
    coverage, non-contiguous expert ids, or an accessor that cannot answer is
    a hard error; none of those states is equivalent to a dense model.
    """
    discovered = profile_declared_unpacked_expert_linears(model, profile)
    grouped: dict[str, list[UnpackedExpertLinear]] = {}
    for member in discovered:
        grouped.setdefault(member.unit_qname, []).append(member)

    units: list[_UnpackedExpertUnit] = []
    for qname in sorted(grouped):
        members = sorted(grouped[qname], key=lambda member: member.qname)
        group_keys = {member.group_key for member in members}
        if len(group_keys) != 1:
            raise RuntimeError(
                f"{qname}: profile {type(profile).__name__} assigned the "
                f"unpacked expert rows to conflicting serving-format groups: "
                f"{sorted(group_keys)!r}"
            )

        by_projection: dict[str, dict[int, UnpackedExpertLinear]] = {}
        for member in members:
            per_expert = by_projection.setdefault(member.projection_name, {})
            if member.expert_id in per_expert:
                raise RuntimeError(
                    f"{qname}: duplicate routed expert projection "
                    f"{member.projection_name!r} for expert {member.expert_id}"
                )
            per_expert[member.expert_id] = member
        expected_ids: list[int] | None = None
        for projection, per_expert in sorted(by_projection.items()):
            ids = sorted(per_expert)
            if ids != list(range(len(ids))):
                raise RuntimeError(
                    f"{qname}: non-contiguous expert ids for {projection}: "
                    f"{ids[:16]!r}"
                )
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(
                    f"{qname}: expert ids differ across projections; "
                    f"{projection} has {ids[:16]!r}, expected "
                    f"{expected_ids[:16]!r}"
                )
        if not expected_ids:
            raise RuntimeError(f"{qname}: routed expert group has no members")

        projections_by_parent: dict[str, set[str]] = {}
        for projection in by_projection:
            parent = _expert_profile_call(
                profile, "packed_expert_parent_for_projection", projection
            )
            if not isinstance(parent, str) or not parent:
                raise RuntimeError(
                    f"{qname}: profile {type(profile).__name__} grouped "
                    f"projection {projection!r} as routed but did not declare "
                    "its packed serving parent"
                )
            projections_by_parent.setdefault(parent, set()).add(projection)

        roles: list[tuple[str, tuple[str, ...]]] = []
        members_by_target: dict[str, dict[tuple[str, int], str]] = {}
        consumed: set[str] = set()
        for parent in sorted(projections_by_parent):
            declared = _expert_profile_call(
                profile, "packed_expert_projection_names", parent
            )
            try:
                projections = tuple(str(name) for name in declared)
            except TypeError as exc:
                raise RuntimeError(
                    f"{qname}: profile {type(profile).__name__} returned a "
                    f"malformed projection order for packed parent {parent!r}"
                ) from exc
            actual = projections_by_parent[parent]
            if not projections or set(projections) != actual:
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} needs exactly profile-"
                    f"declared projections {projections!r}, found "
                    f"{sorted(actual)!r}"
                )
            if len(set(projections)) != len(projections):
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} repeats a projection "
                    f"in {projections!r}"
                )
            consumed.update(projections)
            target = f"{qname}.{parent}"
            members_by_target[target] = {
                (projection, expert_id):
                    by_projection[projection][expert_id].qname
                for projection in projections
                for expert_id in expected_ids
            }
            roles.append((parent, projections))

            # Packing concatenates projections along the output dimension.
            # Every row in one role must therefore share an input width, and
            # every expert must expose the same physical shape per projection.
            input_widths: set[int] = set()
            for projection in projections:
                shapes = {
                    tuple(int(dim) for dim in member.module.weight.shape)
                    for member in by_projection[projection].values()
                }
                if len(shapes) != 1:
                    raise RuntimeError(
                        f"{qname}: {projection} shapes differ across experts: "
                        f"{sorted(shapes)!r}"
                    )
                shape = next(iter(shapes))
                if len(shape) != 2:
                    raise RuntimeError(
                        f"{qname}: profile-declared unpacked Linear "
                        f"{projection} has non-matrix shape {shape!r}"
                    )
                input_widths.add(shape[1])
            if len(input_widths) != 1:
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} projections disagree "
                    f"on input width: {sorted(input_widths)!r}"
                )
        if consumed != set(by_projection):
            raise RuntimeError(
                f"{qname}: routed expert layout did not consume projections "
                f"{sorted(set(by_projection) - consumed)!r}"
            )
        units.append(_UnpackedExpertUnit(
            qname=qname,
            group_key=next(iter(group_keys)),
            members=tuple(members),
            roles=tuple(roles),
            num_experts=len(expected_ids),
            members_by_target=members_by_target,
        ))
    return units


def _virtual_packed_module(unit: _UnpackedExpertUnit) -> nn.Module:
    """Materialize the export-equivalent packed stacks for one live unit."""
    by_key = {
        (member.projection_name, member.expert_id): member
        for member in unit.members
    }
    packed = nn.Module()
    for parent, projections in unit.roles:
        expert_rows = []
        for expert_id in range(unit.num_experts):
            tensors = [
                by_key[(projection, expert_id)].module.weight.detach()
                for projection in projections
            ]
            expert_rows.append(
                tensors[0] if len(tensors) == 1
                else torch.cat(tensors, dim=0)
            )
        setattr(
            packed,
            parent,
            nn.Parameter(torch.stack(expert_rows), requires_grad=False),
        )
    return packed


def _scatter_virtual_packed_module(
    unit: _UnpackedExpertUnit,
    packed: nn.Module,
    expert_ids: Sequence[int],
) -> None:
    by_key = {
        (member.projection_name, member.expert_id): member
        for member in unit.members
    }
    for parent, projections in unit.roles:
        tensor = getattr(packed, parent).data
        offset = 0
        for projection in projections:
            width = int(by_key[(projection, 0)].module.weight.shape[0])
            for expert_id in expert_ids:
                by_key[(projection, int(expert_id))].module.weight.data.copy_(
                    tensor[int(expert_id), offset:offset + width]
                )
            offset += width


@torch.no_grad()
def _unpacked_unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    unit: _UnpackedExpertUnit,
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    sample_idx: torch.Tensor | None = None,
    per_window: bool = False,
    forward_model=None,
) -> float | tuple[float, list[float]]:
    """Unit KL for per-expert Linears via the shipped packed-stack render."""
    if sample_idx is None:
        expert_ids = list(range(unit.num_experts))
    else:
        expert_ids = [int(value) for value in sample_idx.tolist()]
    selected = set(expert_ids)
    originals = {
        member.qname: member.module.weight.data.clone()
        for member in unit.members
        if member.expert_id in selected
    }
    packed = _virtual_packed_module(unit)
    param_names = [parent for parent, _projections in unit.roles]
    try:
        _quantize_unit_inplace(
            packed,
            param_names,
            fmt,
            expert_chunk=expert_chunk,
            col_weights=col_weights,
            unit_qname=unit.qname,
            sample_idx=sample_idx,
        )
        _scatter_virtual_packed_module(unit, packed, expert_ids)
        # The packed stacks are a full copy of the unit's expert mass
        # (~25 GB on a GLM-5.3 layer) and are redundant once scattered
        # back into the live members. Holding them through the forward
        # streams stacked a third expert-mass copy on top of `originals`
        # + resident weights and wedged the box twice; free the blocks
        # back to the shared pool before the eval forwards.
        for parent, _projections in unit.roles:
            delattr(packed, parent)
        del packed
        torch.cuda.empty_cache()
        total = 0.0
        n_tok = 0
        windows: list[float] = []
        bs = _calib_batch()
        for bi, i in enumerate(range(0, calib_ids.shape[0], bs)):
            lp = F.log_softmax(
                (forward_model or model)(
                    calib_ids[i:i + bs]
                ).logits.float(), -1
            )
            bl = baseline[bi].to(lp.device)
            kl = (bl.exp() * (bl - lp)).sum(-1)
            wsum = float(kl.sum().item())
            total += wsum
            n_tok += kl.numel()
            if per_window:
                windows.append(wsum / max(kl.numel(), 1))
        mean = total / max(n_tok, 1)
        return (mean, windows) if per_window else mean
    finally:
        for member in unit.members:
            original = originals.get(member.qname)
            if original is not None:
                member.module.weight.data.copy_(original)
        # Return the clone's ~25 GB to the shared pool before the next
        # format's eval clones again (unified memory: allocator-cached
        # blocks are invisible to `avail` and to other processes).
        originals.clear()
        torch.cuda.empty_cache()


def _cb_ladder_split(measured_fmts: Sequence[str]):
    """Split the menu's CB rungs into PER-(family, mode) ladders, each
    (kmap, anchors, holdout, predicted) for RD-law interpolation. A family
    with < 4 rungs is skipped (anchors+holdout would measure everything
    anyway). At exactly 4 rungs the two extremes anchor the line and one
    middle rung is the holdout, predicting the other (25% fewer encodes); at
    >= 5 rungs three anchors give a least-squares fit. Returns a list of
    ladders, or None when no family pays."""
    fams: dict[str, dict[str, int]] = {}
    for f in measured_fmts:
        parsed = parse_format_name(f)
        if parsed is not None:
            family, k = parsed
            fams.setdefault(family.prefix, {})[f] = k
    raw_anchors = os.environ.get("PRISMAQUANT_CB_LADDER_ANCHORS", "").strip()
    raw_holdout = os.environ.get("PRISMAQUANT_CB_LADDER_HOLDOUT", "").strip()
    if bool(raw_anchors) != bool(raw_holdout):
        raise ValueError(
            "explicit CB ladder planning requires both "
            "PRISMAQUANT_CB_LADDER_ANCHORS and "
            "PRISMAQUANT_CB_LADDER_HOLDOUT"
        )
    explicit_anchors = [
        item.strip().upper() for item in raw_anchors.split(",") if item.strip()
    ]
    explicit_holdout = raw_holdout.upper() if raw_holdout else ""
    explicit_family = ""
    if explicit_anchors:
        parsed_explicit = [parse_format_name(name) for name in
                           explicit_anchors + [explicit_holdout]]
        if any(item is None for item in parsed_explicit):
            raise ValueError("explicit CB ladder plan contains a non-CB format")
        families = {item[0].prefix for item in parsed_explicit}
        if len(families) != 1:
            raise ValueError("explicit CB ladder plan crosses format families")
        if len(set(explicit_anchors)) != len(explicit_anchors):
            raise ValueError("explicit CB ladder anchors contain duplicates")
        if len(explicit_anchors) < 2 or explicit_holdout in explicit_anchors:
            raise ValueError(
                "explicit CB ladder needs at least two distinct anchors and "
                "a separate holdout"
            )
        explicit_family = next(iter(families))

    ladders = []
    explicit_applied = False
    for fam, kmap in sorted(fams.items()):
        if len(kmap) < 4:
            continue
        by_k = sorted(kmap, key=kmap.get)
        if explicit_family and fam == explicit_family:
            required = set(explicit_anchors + [explicit_holdout])
            missing = sorted(required - set(kmap))
            if missing:
                raise ValueError(
                    f"explicit CB ladder plan is absent from the measured "
                    f"menu: {missing}"
                )
            anchors = list(explicit_anchors)
            holdout = explicit_holdout
            explicit_applied = True
        else:
            if len(by_k) == 4:
                anchors = [by_k[0], by_k[-1]]
            else:
                anchors = [by_k[0], by_k[len(by_k) // 2], by_k[-1]]
            rest = [f for f in by_k if f not in anchors]
            holdout = rest[len(rest) // 2]
        rest = [f for f in by_k if f not in anchors]
        predicted = [f for f in rest if f != holdout]
        if predicted:
            ladders.append((kmap, anchors, holdout, predicted))
    if explicit_family and not explicit_applied:
        raise ValueError(
            f"explicit CB ladder family {explicit_family!r} is absent from "
            "the measured menu"
        )
    return ladders or None


def _ladder_family_prefix(kmap: Mapping[str, int]) -> str:
    """The CB family prefix a ladder's kmap belongs to.

    ``_cb_ladder_split`` already buckets by ``family.prefix``, so every key of
    one kmap shares a family; recovering it here keeps the splitter's 4-tuple
    shape (both cost chains and their tests unpack it) while letting the
    cross-family check know WHICH curve each residual came from.
    """
    for name in sorted(kmap):
        parsed = parse_format_name(name)
        if parsed is not None:
            return str(parsed[0].prefix)
    return ""


def _cb_ladder_signed_residual(kmap: Mapping[str, int],
                               anchors: Sequence[str],
                               values: Mapping[str, float],
                               holdout: str) -> float | None:
    """Signed relative holdout residual ``(predicted - measured)/|measured|``.

    ``_cb_ladder_gate`` keeps only the magnitude, which is the right input to
    a per-unit accept/reject. The cross-family symmetry check needs the SIGN:
    two families can miss by the same amount in opposite directions, and that
    is precisely the biased-estimator state the audit's gate exists to catch
    (``cb_ladder_cross_family``). Refits the same shared law, so the two
    numbers cannot disagree about which law produced them.
    """
    law = _cb_ladder_law(kmap, anchors, values)
    if law is None:
        return None
    try:
        meas = float(values[holdout])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(meas) or meas == 0.0:
        return None
    resid = (law.predict(holdout) - meas) / abs(meas)
    return resid if math.isfinite(resid) else None


def _fit_floor_law(ks: Sequence[float], ds: Sequence[float]):
    """Solve D(k) = F + C * 2^(-b*k) exactly through three (k, D) anchors
    (unequal spacing; bisection on b). The floor term matters for the FP8_CB
    family, whose error flattens toward the E4M3 grid's own floor at high k —
    a pure log-linear law systematically misses there (0.6B smoke: 60-90%
    holdout rejection). Returns (F, C, b) with F >= 0, or None when the
    anchors are non-monotone/degenerate (caller falls back to log-linear;
    the holdout gate rules either way)."""
    pts = sorted(zip(ks, ds))
    (k1, d1), (k2, d2), (k3, d3) = pts
    if not (d1 > d2 > d3 > 0.0):
        return None
    target = (d1 - d2) / (d2 - d3)

    def ratio(b):
        r1, r2, r3 = (2.0 ** (-b * k) for k in (k1, k2, k3))
        den = r2 - r3
        if den <= 0.0:
            return float("inf")
        return (r1 - r2) / den

    lo, hi = 1e-6, 4.0
    if ratio(lo) > target:            # decays faster than pure exponential
        return None
    while ratio(hi) < target and hi < 64.0:
        hi *= 2.0
    if ratio(hi) < target:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < target:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    r1 = 2.0 ** (-b * k1)
    r2 = 2.0 ** (-b * k2)
    if r1 - r2 <= 0.0:
        return None
    C = (d1 - d2) / (r1 - r2)
    F = d1 - C * r1
    if F < 0.0 or C <= 0.0:
        return None
    return F, C, b


def _cb_ladder_rate_factor(fmt_name: str, k: int) -> float:
    """Exact per-sub rate factor R(k) = sum_i 2^(-2*b_i/d_i) under the
    ceil-first bit split — the theory-faithful rate variable for the CB
    ladder. The smooth 2^(-alpha*k) law treats k as evenly divisible, but at
    k % n_sub != 0 the ceil-first split gives some sub-tables one bit more:
    the true error carries a +4-6% SAWTOOTH by split phase that a smooth law
    cannot represent — and the (28, 38, 48) anchors + k=39 holdout sit on
    DIFFERENT phases, which is exactly where the 8-12% holdout rejections
    came from (2026-07-21 27B cost run). R is exact per phase and reduces to
    the old law at even splits (R(4m) = 4*2^(-m) for fp8)."""
    parsed = parse_format_name(fmt_name)
    if parsed is None:
        raise ValueError(f"not a producer CB format: {fmt_name!r}")
    family, parsed_k = parsed
    if int(k) != parsed_k:
        raise ValueError(
            f"CB rung mismatch: {fmt_name!r} encodes k={parsed_k}, got {k}"
        )
    widths = subtable_bit_widths(parsed_k, family.mode, family.n_sub)
    sub_dim = VEC_DIM // family.n_sub
    return sum(2.0 ** (-2.0 * width / sub_dim) for width in widths)


class _LadderLaw(NamedTuple):
    """A fitted CB-ladder law: the prediction closure plus the branch name
    that produced it (for provenance in the gate's log)."""
    predict: Callable[[str], float]
    name: str


def _cb_ladder_law(kmap: Mapping[str, int], anchors: Sequence[str],
                   values: Mapping[str, float]) -> _LadderLaw | None:
    """Fit ONE metric on the anchors. THE shared CB-ladder law.

    Chain (the holdout gate arbitrates accept/reject of the whole chain, so
    every branch is a proposal only):
      1. Split-aware FLOORED LINEAR law D = F + C*R(k), with R the exact
         ceil-first per-sub rate factor — plain linear least squares in
         (1, R); kills both the high-k floor miss AND the k%n_sub sawtooth.
         (F clamped to 0 -> C is refit through the origin.)
      2. The smooth floor law D = F + C*2^(-b*k) (exact 3-anchor solve; the
         fp8 family flattens toward the E4M3 grid floor at high k).
      3. Log-linear LS (the original law).

    Both cost chains call this: the dense per-tensor path
    (``measure_quant_cost._ladder_metric_fit``) and the expert unit-KL path
    (``_cb_ladder_fit``). They were separate implementations until R20
    (2026-07-30) and the expert side carried NO R(k) term at all, so the
    ceil-first sawtooth that motivated the dense change (commit 5184892 —
    the (28,38,48)+k=39 phase mismatch behind 8-12% holdout rejections on
    the 2026-07-21 27B run) was still costing the expert ladder its
    holdouts.

    Returns None if any anchor value is unusable."""
    try:
        xs = [float(kmap[f]) for f in anchors]
        vs = [float(values[f]) for f in anchors]
    except (KeyError, TypeError):
        return None
    if len(anchors) >= 2:
        # -- 1. split-aware floored linear LS ------------------------------
        rs = [_cb_ladder_rate_factor(f, kmap[f]) for f in anchors]
        n = float(len(rs))
        mr = sum(rs) / n
        mv = sum(vs) / n
        den = sum((r - mr) ** 2 for r in rs)
        if den > 0.0:
            C = sum((r - mr) * (v - mv) for r, v in zip(rs, vs)) / den
            F = mv - C * mr
            if C > 0.0 and F >= 0.0:
                return _LadderLaw(
                    lambda f, _F=F, _C=C, _km=kmap: float(
                        _F + _C * _cb_ladder_rate_factor(f, _km[f])),
                    "floored_linear_R")
            if C > 0.0 and F < 0.0:
                # floor clamped to 0: refit C through the origin
                den0 = sum(r * r for r in rs)
                if den0 > 0.0:
                    C0 = sum(r * v for r, v in zip(rs, vs)) / den0
                    if C0 > 0.0:
                        return _LadderLaw(
                            lambda f, _C=C0, _km=kmap: float(
                                _C * _cb_ladder_rate_factor(f, _km[f])),
                            "linear_R_origin")
    if len(anchors) == 3:
        # -- 2. smooth floor law (exact solve) -----------------------------
        fl = _fit_floor_law(xs, vs)
        if fl is not None:
            F, C, b = fl
            return _LadderLaw(
                lambda f, _F=F, _C=C, _b=b, _km=kmap:
                    float(_F + _C * 2.0 ** (-_b * _km[f])),
                "floor_law")
    # -- 3. log-linear LS --------------------------------------------------
    ys = [math.log2(max(v, 1e-20)) for v in vs]
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    b = (-sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
         if denom > 0 else _LADDER_SLOPE_BITS)
    a = my + b * mx
    return _LadderLaw(
        lambda f, _a=a, _b=b, _km=kmap: float(2.0 ** (_a - _b * _km[f])),
        "log_linear")


def _cb_ladder_holdout_tol(kmap: Mapping[str, int], anchors: Sequence[str],
                           values: Mapping[str, float], holdout: str,
                           floor: float,
                           windows: Mapping[str, Sequence[float]] | None
                           ) -> float:
    """Derive the holdout-gate tolerance from the rungs' MEASUREMENT noise.

    ``encode_tiers.md`` §B states the rule — *"trust the fit only where the
    holdout error clears the between-seed cost noise"* — so the threshold is
    that noise, not a taste constant (house rule 2). ``windows`` carries each
    measured rung's per-calibration-window values; the expert stage gets them
    for free because every unit KL is already a mean over independent
    calibration windows (``_unit_kl(per_window=True)``).

    The estimate is PAIRED: for each window w the law is refitted on that
    window's anchors and the holdout residual ``r_w`` recomputed. Anchors and
    holdout are measured on the SAME windows, so their errors are strongly
    common-mode; the spread of ``r_w`` isolates exactly the noise that
    survives the pairing, and its systematic part (the law's misfit) drops
    out of a standard deviation. The tolerance is the standard error of that
    mean residual, relative to the holdout::

        tol = stdev(r_w) / sqrt(n_windows) / |v_holdout|

    Returns ``floor`` when the datum is ABSENT or degenerate:

    * the dense per-tensor path measures each ``(tensor, format)`` exactly
      once — the accumulator's ``_count`` is 1 — so it has no between-draw
      spread to offer. (A FIT RESIDUAL is not a substitute: it cannot
      separate measurement noise from law misfit, so a systematically wrong
      law would inflate its own tolerance and open the gate exactly where it
      must close. Measured: on free-rate exponential anchors that estimator
      returns tol 252% against a 127% miss.)
    * fewer than 2 windows, ragged window counts, or an exactly-zero spread
      (a synthetic/degenerate estimator with no resolution).

    ``floor`` defaults to the historical bare 0.10 and is the value the
    shipped runs used; where the datum IS present the derived tolerance
    REPLACES it, which is the literal §B rule and can tighten as well as
    loosen. The accept/reject rate is logged so a ladder that the derived
    tolerance has closed is visible rather than silent."""
    if not windows:
        return floor
    need = list(anchors) + [holdout]
    if any(f not in windows for f in need):
        return floor
    n = len(windows[holdout])
    if n < 2 or any(len(windows[f]) != n for f in need):
        return floor
    resid = []
    for w in range(n):
        vw = {f: float(windows[f][w]) for f in need}
        law_w = _cb_ladder_law(kmap, anchors, vw)
        if law_w is None:
            return floor
        resid.append(law_w.predict(holdout) - vw[holdout])
    mr = sum(resid) / n
    var = sum((r - mr) ** 2 for r in resid) / (n - 1)
    se = math.sqrt(var / n)
    try:
        vh = abs(float(values[holdout]))
    except (KeyError, TypeError):
        return floor
    if se <= 0.0 or vh <= 0.0:
        return floor
    return se / vh


def _cb_ladder_gate(kmap: Mapping[str, int], anchors: Sequence[str],
                    values: Mapping[str, float], holdout: str,
                    tol_floor: float,
                    windows: Mapping[str, Sequence[float]] | None = None):
    """Fit + holdout-gate ONE metric. THE shared gate for both cost chains.

    Returns ``(law | None, holdout_rel_err, tol)``. ``law`` is None when the
    anchors are unusable (rel_err inf) or when the holdout misses by more
    than the tolerance; the caller then MEASURES the predicted rungs
    (encode_tiers.md §B/§C — a tensor/unit that defies the law never
    receives an interpolated cost)."""
    law = _cb_ladder_law(kmap, anchors, values)
    tol = _cb_ladder_holdout_tol(kmap, anchors, values, holdout, tol_floor,
                                 windows)
    if law is None:
        return None, float("inf"), tol
    try:
        meas = float(values[holdout])
    except (KeyError, TypeError):
        return None, float("inf"), tol
    rel = abs(law.predict(holdout) - meas) / max(abs(meas), 1e-20)
    if rel > tol:
        return None, rel, tol
    return law, rel, tol


def _cb_ladder_fit(kls: Mapping[str, float], kmap: Mapping[str, int],
                   anchors: Sequence[str], holdout: str,
                   predicted: Sequence[str], tol: float,
                   windows: Mapping[str, Sequence[float]] | None = None):
    """Fit the anchors and predict, through the SHARED law + gate.

    ``tol`` is the tolerance used when no measurement-noise datum is
    available; when ``windows`` (per-rung per-calibration-window values) is
    supplied the gate derives the tolerance from it instead
    (``_cb_ladder_holdout_tol``).

    Returns ``(predicted_kls | None, holdout_rel_err, tol_used)``."""
    law, rel, tol_used = _cb_ladder_gate(kmap, anchors, kls, holdout, tol,
                                         windows)
    if law is None:
        return None, rel, tol_used
    return {f: law.predict(f) for f in predicted}, rel, tol_used


def measure_expert_unit_costs(
    model,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    ladder_interp: bool = False,
    ladder_tol: float = 0.10,
    expert_sample: int = 0,
    max_units: int = 0,
    unit_filter: str | None = None,
    forward_model=None,
    baseline_logprobs: list[torch.Tensor] | None = None,
    synthesize_col_weights: bool = True,
) -> tuple[dict, dict, dict]:
    """Measure empirical KL costs for profile-declared routed experts.

    Returns ``(stats, costs, unit_kls)`` where stats/costs are
    allocator-payload row dicts keyed by full member names and ``unit_kls``
    maps ``experts_qname -> {fmt: unit_kl}``.
    """
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    profile = resolve_routed_expert_profile(model, profile)
    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    # First classify every routed target through the profile-owned, rank-free
    # predicate.  Physical-layout discovery below is allowed to inspect rank
    # only to choose a renderer; it must account for every classified target.
    routed_targets = set(profile_declared_routed_expert_targets(model, profile))
    packed_units = [
        (qn, m) for qn, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    unpacked_units = _unpacked_expert_units(model, profile)
    packed_qnames = {qn for qn, _mod in packed_units}
    unpacked_qnames = {unit.qname for unit in unpacked_units}
    overlap = sorted(packed_qnames & unpacked_qnames)
    if overlap:
        raise RuntimeError(
            "profile-declared routed expert units expose both packed "
            f"Parameters and unpacked Linears at {overlap[:8]!r}; refusing "
            "to measure overlapping state twice"
        )

    accounted: set[str] = {
        member.qname
        for unit in unpacked_units
        for member in unit.members
    }
    for qn, mod in packed_units:
        for param_name in _packed_experts_param_names(mod, profile):
            accounted.add(f"{qn}.{param_name}" if qn else str(param_name))
    if accounted != routed_targets:
        missing = sorted(routed_targets - accounted)
        unexpected = sorted(accounted - routed_targets)
        raise RuntimeError(
            "profile-declared routed expert discovery could not form an "
            "empirical serving unit for every target; "
            f"unaccounted={missing[:8]!r}, non-profile targets="
            f"{unexpected[:8]!r}"
        )

    units: list[tuple[str, str, object]] = [
        ("packed", qn, mod) for qn, mod in packed_units
    ] + [
        ("unpacked", unit.qname, unit) for unit in unpacked_units
    ]
    if unit_filter:
        pat = re.compile(unit_filter)
        units = [record for record in units if pat.search(record[1])]
    if max_units > 0:
        units = units[:max_units]
    if progress:
        _log(f"{len(units)} expert serving units; measured formats: "
             f"{measured_fmts} (menu {menu})"
             + (f"; expert_sample={expert_sample}" if expert_sample else ""))
    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    # Cleared up front so a caller cannot read a PREVIOUS call's cross-family
    # verdict off this function after an early return (no units / no measured
    # formats) — a stale symmetry certificate is worse than none.
    measure_expert_unit_costs.last_cross_family_verdict = None
    if not units or not measured_fmts:
        return stats, costs, unit_kls

    has_cb = any(fr.get_format(f).family in _CB_FAMILIES
                 for f in measured_fmts)
    if has_cb:
        # Packed modules can synthesize missing imatrix entries from a module
        # input capture. Unpacked per-expert Linears already have one harvested
        # entry per member; those are pooled below with the exporter's exact
        # virtual-stack rule and missing coverage is a hard error.
        if col_weights is None:
            col_weights = {}
        selected_packed = [
            (qn, mod) for kind, qn, mod in units if kind == "packed"
        ]
        if baseline_logprobs is not None:
            baseline = baseline_logprobs
            added = []
            if selected_packed and synthesize_col_weights:
                raise ValueError(
                    "precomputed baseline cannot synthesize packed-expert "
                    "col_weights because its module-input capture is absent"
                )
        elif selected_packed and synthesize_col_weights:
            baseline, unit_x = _baseline_logprobs(
                model, calib_ids, capture_units=selected_packed,
                forward_model=forward_model)
            added = ensure_unit_col_weights(
                model, selected_packed, col_weights, unit_x
            )
            del unit_x
        else:
            baseline = _baseline_logprobs(
                model, calib_ids, forward_model=forward_model)
            added = []
        if added and progress:
            _log(f"synthesized {len(added)} packed-expert imatrix entries "
                 f"(module-input pool / down_proj replay): "
                 f"{added[:4]}{'...' if len(added) > 4 else ''}")
        measure_expert_unit_costs.last_added_col_weights = added
        # The packed-expert pooling rule: fuse gate/up in declared order, and
        # average their per-member imatrix vectors per expert.
        from prismaquant.routed_experts import packed_expert_col_weights
        for kind, _qn, unit in units:
            if kind == "unpacked":
                col_weights = packed_expert_col_weights(
                    col_weights, unit.members_by_target, profile
                )
    else:
        baseline = (
            baseline_logprobs
            if baseline_logprobs is not None
            else _baseline_logprobs(
                model, calib_ids, forward_model=forward_model
            )
        )
        measure_expert_unit_costs.last_added_col_weights = []
    ladder = _cb_ladder_split(measured_fmts) if ladder_interp else None
    # Visible accept/reject rate for the holdout gate (R20): a ladder that
    # is silently rejecting most units is paying full measurement cost PLUS
    # the anchors, and the operator must be able to see that from the log.
    ladder_accept = 0
    ladder_reject = 0
    for unit_kind, qn, storage in units:
        if unit_kind == "packed":
            mod = storage
            pnames = list(_packed_experts_param_names(mod, profile))
            n_params_unit = sum(
                int(getattr(mod, pn).numel()) for pn in pnames
            )
            num_experts = int(getattr(mod, pnames[0]).shape[0])
            row_members = [
                (
                    f"{qn}.{pn}" if qn else pn,
                    getattr(mod, pn),
                    {
                        "num_experts": num_experts,
                        "_packed_experts_module": qn,
                        "_packed_param": pn,
                    },
                )
                for pn in pnames
            ]
            for pn in pnames:
                t = getattr(mod, pn)
                if not bool((t != 0).any()):
                    raise RuntimeError(
                        f"{qn}.{pn}: packed-expert stack is ALL ZERO — the "
                        f"checkpoint's per-expert weights were never mapped "
                        f"into the packed class param (the zero-expert "
                        f"calibration bug). A unit KL measured now would read "
                        f"exactly 0 for every format. Fill via "
                        f"layer_streaming.fill_packed_experts_from_source "
                        f"before measuring."
                    )
        else:
            unit = storage
            num_experts = unit.num_experts
            n_params_unit = sum(
                int(member.module.weight.numel()) for member in unit.members
            )
            row_members = [
                (
                    member.qname,
                    member.module.weight,
                    {"_unpacked_expert_unit": qn},
                )
                for member in unit.members
            ]
            if not any(
                bool((member.module.weight != 0).any())
                for member in unit.members
            ):
                raise RuntimeError(
                    f"{qn}: profile-declared unpacked expert unit is ALL "
                    "ZERO; an empirical unit KL would silently read zero for "
                    "every format"
                )
        # One stratified subsample SHARED across every format of the unit, so
        # inter-format comparability (what the allocator consumes) is exact
        # even under sampling; the extrapolation to the full unit rides on
        # cross-expert additivity (validated fp32-additive in this repo) and
        # is scaled by expert count (uniform stacks).
        sample_idx = _expert_sample_idx(num_experts, expert_sample)
        kl_scale = (float(num_experts) / float(sample_idx.numel())
                    if sample_idx is not None else 1.0)

        # Per-window unit KLs of every MEASURED rung: the between-draw noise
        # datum the ladder's holdout gate derives its tolerance from. Free —
        # _unit_kl already loops window by window.
        kl_windows: dict[str, list[float]] = {}

        def kl_of(fmt):
            if unit_kind == "packed":
                out = _unit_kl(
                    model, calib_ids, baseline, mod, pnames, fmt,
                    expert_chunk=expert_chunk, col_weights=col_weights,
                    unit_qname=qn, sample_idx=sample_idx,
                    per_window=ladder is not None,
                    forward_model=forward_model)
            else:
                out = _unpacked_unit_kl(
                    model, calib_ids, baseline, unit, fmt,
                    expert_chunk=expert_chunk, col_weights=col_weights,
                    sample_idx=sample_idx, per_window=ladder is not None,
                    forward_model=forward_model)
            if ladder is None:
                return kl_scale * out
            mean, windows = out
            kl_windows[fmt] = [kl_scale * w for w in windows]
            return kl_scale * mean

        if ladder is None:
            kls = {fmt: kl_of(fmt) for fmt in measured_fmts}
        else:
            predicted_all = {f for (_, _, _, pred) in ladder for f in pred}
            kls = {fmt: kl_of(fmt) for fmt in measured_fmts
                   if fmt not in predicted_all}
            ladder_meta_all = []
            for kmap, anchors, holdout, predicted in ladder:
                # Recorded for EVERY ladder, accepted or not: the family the
                # curve belongs to and the SIGNED holdout residual. Both are
                # inputs to the cross-family symmetry gate (ultraplan P5a
                # item 2, cb_ladder_cross_family) — a per-family accept rate
                # alone cannot tell a symmetric miss from a biased one.
                family = _ladder_family_prefix(kmap)
                signed = _cb_ladder_signed_residual(
                    kmap, anchors, kls, holdout)
                pred_kls, rel, tol_used = _cb_ladder_fit(
                    kls, kmap, anchors, holdout, predicted, ladder_tol,
                    kl_windows)
                if pred_kls is None:
                    # Holdout gate FAILED for this unit/family: fall back to
                    # full measurement (recon-validated, KL-unproven law).
                    ladder_reject += 1
                    if progress:
                        _log(f"  {qn}: ladder holdout rel_err {rel:.1%} > "
                             f"{tol_used:.1%} — measuring {predicted}")
                    kls.update({fmt: kl_of(fmt) for fmt in predicted})
                    ladder_meta_all.append(
                        {"accepted": False, "family": family,
                         "holdout": holdout,
                         "holdout_rel_err": round(rel, 4),
                         "holdout_signed_rel_resid": (
                             round(signed, 6) if signed is not None else None),
                         "holdout_tol": round(tol_used, 4),
                         "anchors": anchors})
                else:
                    ladder_accept += 1
                    ladder_meta_all.append({
                        "accepted": True, "family": family,
                        "holdout_rel_err": round(rel, 4),
                        "holdout_signed_rel_resid": (
                            round(signed, 6) if signed is not None else None),
                        "holdout_tol": round(tol_used, 4),
                        "anchors": anchors, "holdout": holdout,
                        "predicted": predicted,
                    })
                    kls.update(pred_kls)
            kls["_ladder"] = ladder_meta_all
        ladder_meta = kls.pop("_ladder", None)
        unit_kls[qn] = dict(kls)
        if ladder_meta is not None:
            unit_kls[qn]["_ladder"] = ladder_meta
        if sample_idx is not None:
            unit_kls[qn]["_sampling"] = {
                "num_experts": num_experts,
                "sampled": int(sample_idx.numel()),
                "scale": round(kl_scale, 4),
            }
        for full, tensor, expert_metadata in row_members:
            npm = int(tensor.numel())
            shape = list(tensor.shape)
            if unit_kind == "packed":
                in_features = int(shape[2])
                out_features = int(shape[1])
            else:
                in_features = int(shape[1])
                out_features = int(shape[0])
            stats[full] = {
                # h_trace is meaningless for an empirically-costed unit; the
                # allocator consumes predicted_dloss directly. 0.0 marks
                # "do not fall back to h_trace x weight_mse" for this row.
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": in_features,
                "out_features": out_features,
                **expert_metadata,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                # Split the UNIT cost across members by n_params so the
                # per-member sum re-assembles exactly one unit KL.
                row[fmt] = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
            for fmt in menu:
                if fmt in PASSTHROUGH_FORMATS:
                    row[fmt] = {
                        "predicted_dloss": 0.0,
                        "cost_source": "passthrough_zero",
                        "output_mse_measured": False,
                    }
            costs[full] = row
        if progress:
            _log(f"  {qn}: " + "  ".join(
                f"{fmt} unit KL = {kls[fmt]:.4e}" for fmt in measured_fmts)
                + f"  (n_params={n_params_unit / 1e6:.0f}M, "
                  f"experts={num_experts})")
    if ladder is not None:
        n_gate = ladder_accept + ladder_reject
        _log(f"CB ladder holdout gate: {ladder_accept}/{n_gate} accepted "
             f"({ladder_accept / max(n_gate, 1):.0%}), {ladder_reject} "
             f"rejected -> measured (tolerance derived per unit from the "
             f"between-window noise of the measured rungs; "
             f"{ladder_tol:.0%} where that datum is degenerate)")
        # Cross-family symmetry gate (ultraplan P5a item 2). A failure does
        # not abort the run — it says the CROSS-FAMILY verdict this run's
        # ladder fits could support is not publishable — but it must be loud
        # here and travel into the artifact provenance.
        verdict = verdict_from_unit_kls(unit_kls)
        _log(f"CB ladder cross-family symmetry: {verdict['verdict'].upper()} "
             f"— {verdict['detail']}")
        measure_expert_unit_costs.last_cross_family_verdict = verdict
    else:
        measure_expert_unit_costs.last_cross_family_verdict = None
    return stats, costs, unit_kls


EXPERT_CHECKPOINT_IDENTITY_SCHEMA = (
    "prismaquant.expert_empirical_checkpoint.identity.v1"
)
EXPERT_CHECKPOINT_STAGE = "expert empirical cost"


def _tensor_value_stamp(tensor: torch.Tensor) -> dict[str, object]:
    value = torch.as_tensor(tensor).detach().to("cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _streamed_expert_unit_records(model, profile, *, unit_filter, max_units):
    """Return serving units in the resident measurer's exact order."""
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    packed = [
        ("packed", qname, mod)
        for qname, mod in model.named_modules()
        if _is_packed_experts_module(mod, profile)
    ]
    unpacked = [
        ("unpacked", unit.qname, unit)
        for unit in _unpacked_expert_units(model, profile)
    ]
    records = packed + unpacked
    if unit_filter:
        pattern = re.compile(str(unit_filter))
        records = [record for record in records if pattern.search(record[1])]
    if max_units > 0:
        records = records[:max_units]
    identities: list[dict[str, object]] = []
    for kind, qname, storage in records:
        if kind == "packed":
            members = []
            for param_name in _packed_experts_param_names(storage, profile):
                tensor = getattr(storage, param_name)
                members.append({
                    "qname": f"{qname}.{param_name}" if qname else param_name,
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                })
        else:
            members = [
                {
                    "qname": member.qname,
                    "shape": [
                        int(dim) for dim in member.module.weight.shape
                    ],
                    "dtype": str(member.module.weight.dtype),
                }
                for member in storage.members
            ]
        identities.append({
            "qname": str(qname),
            "storage": str(kind),
            "members": members,
        })
    return records, identities


def _expert_checkpoint_identity(
    *,
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    col_weights: Mapping[str, torch.Tensor],
    unit_identities: Sequence[Mapping[str, object]],
    model_identity: Mapping[str, object],
    expert_chunk: int,
    ladder_interp: bool,
    ladder_tol: float,
    expert_sample: int,
    max_units: int,
    unit_filter: str | None,
    identity_extra: Mapping[str, object] | None,
) -> dict[str, object]:
    from prismaquant.cost_stage_checkpoint import canonical_json
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.production_weight_cache import (
        _production_cache_source_sha256,
    )

    calib = calib_ids.detach().to("cpu").contiguous()
    from prismaquant.cost_streaming import validate_streamed_model_identity

    exact_model_identity = validate_streamed_model_identity(
        model_identity, where="expert empirical checkpointing"
    )
    identity = {
        "schema": EXPERT_CHECKPOINT_IDENTITY_SCHEMA,
        "git_commit": _git_commit(),
        "producer_source_sha256": _production_cache_source_sha256(),
        "model": exact_model_identity,
        "profile": {
            "module": type(profile).__module__,
            "class": type(profile).__qualname__,
        },
        "calibration": {
            "shape": [int(dim) for dim in calib.shape],
            "dtype": str(calib.dtype),
            "sha256": hashlib.sha256(
                calib.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "calib_hash": calibration_data_hash(calib_ids),
        },
        "formats": [str(fmt) for fmt in formats],
        "units": [dict(record) for record in unit_identities],
        "imatrix": {
            str(name): _tensor_value_stamp(value)
            for name, value in sorted(col_weights.items())
        },
        # This stamp value-binds the selected learned bundle/lattice books,
        # serialization layout, encode tier, and every CB menu semantic.
        "cb_cost_provenance": cb_cost_provenance(formats),
        "measurement_dtype": str(runner.dtype),
        "expert_chunk": int(expert_chunk),
        "calib_batch": int(_calib_batch()),
        "ladder_interp": bool(ladder_interp),
        "ladder_tol": float(ladder_tol),
        "expert_sample": int(expert_sample),
        "max_units": int(max_units),
        "unit_filter": unit_filter,
        "extra": dict(identity_extra or {}),
    }
    return canonical_json(identity, where="expert empirical checkpoint identity")


def _write_expert_unit_checkpoint(
    root: Path,
    *,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    """Patchable publication seam used by interruption tests."""
    from prismaquant.cost_stage_checkpoint import write_unit

    write_unit(
        root,
        stage=EXPERT_CHECKPOINT_STAGE,
        qname=qname,
        identity_sha256=identity_sha256,
        state=state,
    )


def _validate_expert_checkpoint_state(
    qname: str, state: Mapping[str, object]
) -> dict[str, object]:
    expected = {"stats", "costs", "unit_kls"}
    if set(state) != expected:
        raise RuntimeError(
            f"expert unit checkpoint for {qname} has fields "
            f"{sorted(state)!r}, expected {sorted(expected)!r}; refusing "
            "reuse or recompute"
        )
    for field in expected:
        if not isinstance(state[field], Mapping):
            raise RuntimeError(
                f"expert unit checkpoint for {qname} has invalid {field}; "
                "refusing reuse or recompute"
            )
    unit_kls = state["unit_kls"]
    if set(unit_kls) != {qname}:
        raise RuntimeError(
            f"expert unit checkpoint for {qname} carries unit_kls keys "
            f"{sorted(unit_kls)!r}; refusing reuse or recompute"
        )
    return dict(state)


def measure_expert_unit_costs_streamed(
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    ladder_interp: bool = False,
    ladder_tol: float = 0.10,
    expert_sample: int = 0,
    max_units: int = 0,
    unit_filter: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    formats_by_qname: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict, dict, dict]:
    """Measure routed serving units with one decoder layer resident at once.

    The numerical core is the resident ``measure_expert_unit_costs`` function:
    identical qdq, fp32 log-softmax/KL, window order, ladder, and row builder.
    Streaming changes only model residency.  A target layer stays pinned for
    the complete serving unit so its temporary qdq is restored before the
    context can cache/unload it.
    """
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    profile = resolve_routed_expert_profile(runner.model, profile)
    menu = _canon_formats(formats)
    weights = {
        str(name): torch.as_tensor(value)
        for name, value in dict(col_weights or {}).items()
    }
    records, unit_identities = _streamed_expert_unit_records(
        runner.model,
        profile,
        unit_filter=unit_filter,
        max_units=max_units,
    )
    qnames = [str(record[1]) for record in records]
    if len(qnames) != len(set(qnames)):
        raise RuntimeError(
            "streamed expert discovery produced duplicate serving-unit qnames"
        )
    # Every serving unit is layer-local by serving contract. Resolve this
    # before opening the journal so malformed topology cannot acquire state.
    for qname in qnames:
        runner.layer_index_for_qname(qname)

    menu_by_unit = {qname: menu for qname in qnames}
    canonical_format_plan = None
    if formats_by_qname is not None:
        canonical_format_plan = {
            str(name): tuple(_canon_formats(values))
            for name, values in formats_by_qname.items()
        }
        planned_universe = {
            fmt
            for values in canonical_format_plan.values()
            for fmt in values
        }
        menu_by_unit = {}
        for identity in unit_identities:
            unit_qname = str(identity["qname"])
            members = [str(row["qname"]) for row in identity["members"]]
            missing = sorted(
                member for member in members
                if member not in canonical_format_plan
            )
            if missing:
                raise ValueError(
                    f"streamed expert format plan does not cover serving "
                    f"unit {unit_qname}; sample={missing[:8]}"
                )
            projected = {
                tuple(
                    fmt for fmt in menu
                    if fmt not in planned_universe
                    or fmt in canonical_format_plan[member]
                )
                for member in members
            }
            if len(projected) != 1:
                raise ValueError(
                    "streamed expert format plan straddles one serving unit "
                    f"{unit_qname}: {sorted(projected)}"
                )
            unit_menu = next(iter(projected))
            if not unit_menu:
                raise ValueError(
                    f"streamed expert format plan leaves {unit_qname} empty"
                )
            menu_by_unit[unit_qname] = unit_menu

    completed: dict[str, dict[str, object]] = {}
    journal_root: Path | None = None
    journal_identity_sha256: str | None = None
    if checkpoint_dir is not None:
        if model_identity is None:
            raise RuntimeError(
                "streamed expert checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        extra = dict(checkpoint_identity_extra or {})
        if canonical_format_plan is not None:
            if "formats_by_qname" in extra:
                raise ValueError(
                    "checkpoint_identity_extra cannot override "
                    "formats_by_qname"
                )
            extra["formats_by_qname"] = {
                name: list(values)
                for name, values in canonical_format_plan.items()
            }
        identity = _expert_checkpoint_identity(
            runner=runner,
            profile=profile,
            calib_ids=calib_ids,
            formats=menu,
            col_weights=weights,
            unit_identities=unit_identities,
            model_identity=model_identity,
            expert_chunk=expert_chunk,
            ladder_interp=ladder_interp,
            ladder_tol=ladder_tol,
            expert_sample=expert_sample,
            max_units=max_units,
            unit_filter=unit_filter,
            identity_extra=extra,
        )
        from prismaquant.cost_stage_checkpoint import prepare_journal

        journal_root, journal_identity_sha256, raw_completed = prepare_journal(
            checkpoint_dir,
            stage=EXPERT_CHECKPOINT_STAGE,
            resume=resume,
            identity=identity,
            qnames=qnames,
        )
        completed = {
            qname: _validate_expert_checkpoint_state(qname, state)
            for qname, state in raw_completed.items()
        }

    pending = [qname for qname in qnames if qname not in completed]
    baseline = None
    if pending:
        # Resume identity was validated above, before this first forward.
        baseline = _baseline_logprobs(
            runner.model,
            calib_ids,
            forward_model=runner,
        )

    results: dict[str, dict[str, object]] = dict(completed)
    t_units0 = time.time()
    for u_i, qname in enumerate(pending):
        if progress:
            _log(f"streamed unit {qname} ({u_i + 1}/{len(pending)})")
        t_unit0 = time.time()
        exact_filter = rf"\A{re.escape(qname)}\Z"
        unit_menu = menu_by_unit[qname]
        with runner.pin_layer_for_qname(qname):
            unit_stats, unit_costs, unit_kls = measure_expert_unit_costs(
                runner.model,
                profile,
                calib_ids,
                unit_menu,
                expert_chunk=expert_chunk,
                progress=False,
                col_weights=weights,
                ladder_interp=ladder_interp,
                ladder_tol=ladder_tol,
                expert_sample=expert_sample,
                max_units=0,
                unit_filter=exact_filter,
                forward_model=runner,
                baseline_logprobs=baseline,
                # Packed streamed units must receive complete persisted
                # imatrix values. Synthesis needs a second mutable live unit
                # plus router replay and is intentionally fail-closed here.
                synthesize_col_weights=False,
            )
        if set(unit_kls) != {qname}:
            raise RuntimeError(
                f"streamed expert unit {qname} produced unit_kls keys "
                f"{sorted(unit_kls)!r}"
            )
        state = {
            "stats": unit_stats,
            "costs": unit_costs,
            "unit_kls": unit_kls,
        }
        results[qname] = state
        if journal_root is not None:
            assert journal_identity_sha256 is not None
            _write_expert_unit_checkpoint(
                journal_root,
                qname=qname,
                identity_sha256=journal_identity_sha256,
                state=state,
            )
        if progress:
            dt = time.time() - t_unit0
            rate = (time.time() - t_units0) / (u_i + 1)
            eta_min = rate * (len(pending) - u_i - 1) / 60.0
            _log(f"unit {qname} done in {dt:.0f}s "
                 f"({sorted(unit_kls[qname])}); "
                 f"~{eta_min:.0f} min left for {len(pending) - u_i - 1} units")

    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    for qname in qnames:
        state = results[qname]
        stats.update(state["stats"])
        costs.update(state["costs"])
        unit_kls.update(state["unit_kls"])
    if ladder_interp and unit_kls:
        measure_expert_unit_costs.last_cross_family_verdict = (
            verdict_from_unit_kls(unit_kls)
        )
    else:
        measure_expert_unit_costs.last_cross_family_verdict = None
    measure_expert_unit_costs.last_added_col_weights = []
    return stats, costs, unit_kls


@torch.no_grad()
def _fork_quantized_unit(
    runner,
    batch,
    layer: int,
    unit_kind: str,
    storage,
    qn: str,
    fmt: str,
    *,
    profile,
    expert_chunk: int,
    col_weights: Mapping[str, torch.Tensor],
    rows_meta: dict,
) -> torch.Tensor:
    """One fork step: run this unit's layer with an export-equivalent
    quantized COPY of its expert weights on the baseline boundary input.

    The live module is never mutated — each expert Parameter attribute is
    swapped for the quantized copy for exactly one layer call and the
    ORIGINAL Parameter object is reattached in ``finally`` (so the
    streaming cache's install/unload bookkeeping sees the same objects).
    The only transient is the quantized copy itself (~one unit of expert
    mass); there is no restore clone.

    Also captures ``rows_meta[qn]`` (member names/shapes/metadata) on the
    first call, while the layer is resident.
    """
    from prismaquant.sensitivity_probe import _packed_experts_param_names

    device = runner.device
    swaps: list[tuple] = []
    holder = nn.Module()
    if unit_kind == "packed":
        mod = storage
        pnames = list(_packed_experts_param_names(mod, profile))
        for pn in pnames:
            live = getattr(mod, pn)
            if not bool((live != 0).any()):
                raise RuntimeError(
                    f"{qn}.{pn}: packed-expert stack is ALL ZERO at fork "
                    "time — the zero-expert calibration bug; a unit KL "
                    "measured now would read exactly 0 for every format"
                )
            setattr(holder, pn, nn.Parameter(
                live.data.clone(), requires_grad=False))
        _quantize_unit_inplace(
            holder, pnames, fmt,
            expert_chunk=expert_chunk, col_weights=col_weights,
            unit_qname=qn,
        )
        if qn not in rows_meta:
            num_experts = int(getattr(mod, pnames[0]).shape[0])
            members = []
            for pn in pnames:
                shape = list(getattr(mod, pn).shape)
                members.append((
                    f"{qn}.{pn}" if qn else pn,
                    int(getattr(mod, pn).numel()),
                    int(shape[2]),
                    int(shape[1]),
                    {
                        "num_experts": num_experts,
                        "_packed_experts_module": qn,
                        "_packed_param": pn,
                    },
                ))
            rows_meta[qn] = {
                "n_params_unit": sum(m[1] for m in members),
                "members": members,
            }
        for pn in pnames:
            swaps.append((mod, pn, getattr(mod, pn),
                          getattr(holder, pn)))
    else:
        unit = storage
        packed = _virtual_packed_module(unit)
        param_names = [parent for parent, _projections in unit.roles]
        _quantize_unit_inplace(
            packed, param_names, fmt,
            expert_chunk=expert_chunk, col_weights=col_weights,
            unit_qname=qn,
        )
        for parent, _projections in unit.roles:
            setattr(holder, parent, getattr(packed, parent))
            delattr(packed, parent)
        del packed
        if qn not in rows_meta:
            members = [(
                member.qname,
                int(member.module.weight.numel()),
                int(member.module.weight.shape[1]),
                int(member.module.weight.shape[0]),
                {"_unpacked_expert_unit": qn},
            ) for member in unit.members]
            rows_meta[qn] = {
                "n_params_unit": sum(m[1] for m in members),
                "members": members,
            }
        by_key = {
            (member.projection_name, member.expert_id): member
            for member in unit.members
        }
        for parent, projections in unit.roles:
            stack = getattr(holder, parent)
            offset = 0
            for projection in projections:
                width = int(by_key[(projection, 0)].module.weight.shape[0])
                for expert_id in range(unit.num_experts):
                    mod = by_key[(projection, expert_id)].module
                    swaps.append((
                        mod, "weight", mod.weight,
                        nn.Parameter(
                            stack[expert_id, offset:offset + width],
                            requires_grad=False),
                    ))
                offset += width
    try:
        for mod, attr, _orig, q in swaps:
            setattr(mod, attr, q if isinstance(q, nn.Parameter)
                    else nn.Parameter(q, requires_grad=False))
        fork_h = runner.isolated_layer(
            batch, layer,
            batch.activations_cpu[layer].to(device),
            pass_state=None,
        )
    finally:
        for mod, attr, orig, _q in swaps:
            setattr(mod, attr, orig)
    del swaps, holder
    return fork_h


@torch.no_grad()
def measure_expert_unit_costs_forked(
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    max_units: int = 0,
    unit_filter: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
) -> tuple[dict, dict, dict]:
    """Forked-stream expert unit KLs: O(1) body streams instead of O(units).

    The window-major streamed driver re-streams the whole body for EVERY
    forward (a 306 GB source at 16 windows = ~1.1 TB of disk per unit,
    ~47 TB for a 42-unit run — measured 2026-08-26 on GLM-5.3-Flash, GPU
    idling at ~25% while the nvme pinned at 2 GB/s). This driver streams
    the body THREE times total: one boundary-capture pass (baseline
    activations at every decoder boundary + baseline logits), then one
    pass per measured format in which every unit's quantized stream is
    forked from its layer's baseline boundary and advanced through the
    live layers together with all other forks.

    Semantics per unit are identical to the window-major path: exactly one
    unit quantized (export-equivalent packed render), everything else at
    source precision, KL(BF16 || quantized) from fp32 log-softmax at the
    head. Downstream route-flips propagate through the fork's own suffix,
    so the empirical route-flip floor is preserved. Differences are
    reassociation-class only (all windows forward in one batch instead of
    ``_calib_batch()`` chunks).

    The live model is NEVER mutated: the fork's layer call swaps each
    member Linear's ``weight`` Parameter for a view into the quantized
    packed stacks and reattaches the ORIGINAL Parameter objects in
    ``finally`` — no restore clone exists, so the triple-expert-mass peak
    that wedged the window-major runs (resident + clone + packed) cannot
    recur; the transient is the packed render alone.

    Fail-closed scope (v1): profiles with per-pass layer state, CB
    formats (imatrix synthesis not wired), expert subsampling and the CB
    ladder are refused — use the window-major driver for those. Packed
    and unpacked units are both handled; the packed all-zero-stack guard
    fires at fork time (the unpacked variant relies on the window driver's
    discovery-time check if you re-enable it there).
    Checkpoint granularity is the whole run: per-unit rows are journaled
    only after every format pass completes, so a mid-pass crash re-pays
    the passes (~1 h at GLM scale), never the journal identity.
    """
    if profile.new_forward_pass_state() != {}:
        raise RuntimeError(
            "forked expert eval requires stateless layer passes; profile "
            f"{type(profile).__name__} declares per-pass state — use the "
            "window-major streamed driver"
        )
    profile = resolve_routed_expert_profile(runner.model, profile)
    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    cb_fmts = [f for f in measured_fmts
               if fr.get_format(f).family in _CB_FAMILIES]
    if cb_fmts:
        raise RuntimeError(
            f"forked expert eval does not support CB formats {cb_fmts}; "
            "use the window-major streamed driver"
        )
    weights = {
        str(name): torch.as_tensor(value)
        for name, value in dict(col_weights or {}).items()
    }
    records, unit_identities = _streamed_expert_unit_records(
        runner.model,
        profile,
        unit_filter=unit_filter,
        max_units=max_units,
    )
    for kind, qname, _storage in records:
        if kind not in ("packed", "unpacked"):
            raise RuntimeError(
                f"forked expert eval got unknown unit kind {kind!r} for "
                f"{qname}"
            )
    qnames = [str(record[1]) for record in records]
    if len(qnames) != len(set(qnames)):
        raise RuntimeError(
            "forked expert discovery produced duplicate serving-unit qnames"
        )
    unit_by_qname = {str(qn): unit for _kind, qn, unit in records}
    kind_by_qname = {str(qn): kind for kind, qn, _unit in records}
    layer_of = {qn: runner.layer_index_for_qname(qn) for qn in qnames}

    completed: dict[str, dict[str, object]] = {}
    journal_root: Path | None = None
    journal_identity_sha256: str | None = None
    if checkpoint_dir is not None:
        if model_identity is None:
            raise RuntimeError(
                "forked expert checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        extra = dict(checkpoint_identity_extra or {})
        if "eval_driver" in extra:
            raise ValueError(
                "checkpoint_identity_extra cannot override eval_driver"
            )
        extra["eval_driver"] = "forked-stream/1"
        identity = _expert_checkpoint_identity(
            runner=runner,
            profile=profile,
            calib_ids=calib_ids,
            formats=menu,
            col_weights=weights,
            unit_identities=unit_identities,
            model_identity=model_identity,
            expert_chunk=expert_chunk,
            ladder_interp=False,
            ladder_tol=0.0,
            expert_sample=0,
            max_units=max_units,
            unit_filter=unit_filter,
            identity_extra=extra,
        )
        from prismaquant.cost_stage_checkpoint import prepare_journal

        journal_root, journal_identity_sha256, raw_completed = prepare_journal(
            checkpoint_dir,
            stage=EXPERT_CHECKPOINT_STAGE,
            resume=resume,
            identity=identity,
            qnames=qnames,
        )
        completed = {
            qname: _validate_expert_checkpoint_state(qname, state)
            for qname, state in raw_completed.items()
        }
    pending = [qn for qn in qnames if qn not in completed]

    device = runner.device
    num_layers = runner.num_layers

    def _avail_swap_gb() -> tuple[float, float]:
        vals = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                    vals[key] = int(line.split()[1])
        swap = (vals.get("SwapTotal", 0) - vals.get("SwapFree", 0)) / 1048576
        return vals.get("MemAvailable", 0) / 1048576, swap

    unit_kl_means: dict[str, dict[str, float]] = {qn: {} for qn in pending}
    unit_kl_windows: dict[str, dict[str, list[float]]] = {
        qn: {} for qn in pending
    }
    # Row-shape metadata captured at fork time (first format pass), while
    # the unit's layer is resident — by row-building time it is unloaded.
    rows_meta: dict[str, dict[str, object]] = {}
    baseline_lp: list[torch.Tensor] | None = None
    batch = None

    if pending:
        n_windows = int(calib_ids.shape[0])
        _log(f"forked driver: {len(pending)}/{len(qnames)} units pending, "
             f"formats {measured_fmts}, {n_windows} windows in one batch")
        t0 = time.time()
        batch = runner.capture_boundaries(calib_ids)
        avail, swap = _avail_swap_gb()
        _log(f"boundary pass done in {time.time() - t0:.0f}s "
             f"({len(batch.activations_cpu)} boundaries, "
             f"avail {avail:.0f}G, swap {swap:.1f}G)")
        logits = runner.tail_logits(
            batch, batch.activations_cpu[-1].to(device)
        )
        baseline_lp = [
            F.log_softmax(logits[w].float(), dim=-1).cpu()
            for w in range(n_windows)
        ]
        del logits
        torch.cuda.empty_cache()

        for f_i, fmt in enumerate(measured_fmts):
            t_pass0 = time.time()
            forks: dict[str, torch.Tensor] = {}
            for depth in range(runner.prefetch_lookahead):
                runner.context.schedule_prefetch(depth)
            for layer in range(num_layers):
                runner.context.install(
                    layer,
                    require_prefetched=runner.require_prefetched_residency,
                )
                runner.context.schedule_prefetch(
                    layer + runner.prefetch_lookahead
                )
                try:
                    # Advance every existing fork through this live layer
                    # BEFORE forking at it, so a new fork's hidden is not
                    # double-advanced.
                    for qn in list(forks):
                        forks[qn] = runner.isolated_layer(
                            batch, layer, forks[qn], pass_state=None
                        )
                    for qn in pending:
                        if layer_of[qn] != layer:
                            continue
                        forks[qn] = _fork_quantized_unit(
                            runner,
                            batch,
                            layer,
                            kind_by_qname[qn],
                            unit_by_qname[qn],
                            qn,
                            fmt,
                            profile=profile,
                            expert_chunk=expert_chunk,
                            col_weights=weights,
                            rows_meta=rows_meta,
                        )
                        torch.cuda.empty_cache()
                finally:
                    runner.context.unload(layer)
                if progress and (
                    layer % 5 == 4 or layer == num_layers - 1
                ):
                    avail, swap = _avail_swap_gb()
                    rate = (time.time() - t_pass0) / (layer + 1)
                    eta_min = rate * (num_layers - layer - 1) / 60.0
                    _log(f"fork pass {fmt} ({f_i + 1}/"
                         f"{len(measured_fmts)}): layer "
                         f"{layer + 1}/{num_layers}, {len(forks)} forks "
                         f"live, avail {avail:.0f}G swap {swap:.1f}G, "
                         f"~{eta_min:.0f} min left in pass")
            for qn in pending:
                logits = runner.tail_logits(batch, forks.pop(qn))
                total = 0.0
                n_tok = 0
                windows: list[float] = []
                for w in range(n_windows):
                    lp = F.log_softmax(logits[w].float(), dim=-1)
                    bl = baseline_lp[w].to(lp.device)
                    kl = (bl.exp() * (bl - lp)).sum(-1)
                    wsum = float(kl.sum().item())
                    total += wsum
                    n_tok += kl.numel()
                    windows.append(wsum / max(kl.numel(), 1))
                    del lp, bl, kl
                del logits
                unit_kl_means[qn][fmt] = total / max(n_tok, 1)
                unit_kl_windows[qn][fmt] = windows
            torch.cuda.empty_cache()
            _log(f"fork pass {fmt} done in "
                 f"{(time.time() - t_pass0) / 60.0:.1f} min")

    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    results: dict[str, dict[str, object]] = dict(completed)
    for qn in pending:
        kls = unit_kl_means[qn]
        # Row construction mirrors measure_expert_unit_costs field for
        # field — the merge and the allocator must not be able to tell the
        # drivers apart. Shapes come from rows_meta (captured at fork time,
        # while the layer was resident).
        meta = rows_meta[qn]
        n_params_unit = int(meta["n_params_unit"])
        unit_stats: dict = {}
        unit_costs: dict = {}
        for full, npm, in_features, out_features, member_meta in (
            meta["members"]
        ):
            unit_stats[full] = {
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": in_features,
                "out_features": out_features,
                **member_meta,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                row[fmt] = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
            for fmt in menu:
                if fmt in PASSTHROUGH_FORMATS:
                    row[fmt] = {
                        "predicted_dloss": 0.0,
                        "cost_source": "passthrough_zero",
                        "output_mse_measured": False,
                    }
            unit_costs[full] = row
        state = {
            "stats": unit_stats,
            "costs": unit_costs,
            "unit_kls": {qn: dict(kls)},
            "kl_windows": {
                fmt: list(vals)
                for fmt, vals in unit_kl_windows[qn].items()
            },
        }
        results[qn] = state
        if journal_root is not None:
            assert journal_identity_sha256 is not None
            _write_expert_unit_checkpoint(
                journal_root,
                qname=qn,
                identity_sha256=journal_identity_sha256,
                state=state,
            )
        if progress:
            _log(f"  {qn}: " + "  ".join(
                f"{fmt} unit KL = {kls[fmt]:.4e}"
                for fmt in measured_fmts))
    for qn in qnames:
        state = results[qn]
        stats.update(state["stats"])
        costs.update(state["costs"])
        unit_kls.update(state["unit_kls"])
    measure_expert_unit_costs.last_cross_family_verdict = None
    measure_expert_unit_costs.last_added_col_weights = []
    return stats, costs, unit_kls


def merge_cost_payloads(
    base: Mapping[str, object],
    expert_stats: Mapping[str, object],
    expert_costs: Mapping[str, object],
    *,
    formats: Sequence[str],
    replace_experts: bool = False,
) -> dict:
    """Union base non-expert rows with empirical expert rows.

    AURA lane (``replace_experts=False``): collisions are an error —
    aura_cost must have been run with ``--allow-packed-expert-omission``
    (its guard fail-fasts otherwise), so no name may be costed by both
    estimators.

    CB lane (``replace_experts=True``): the COST_MODE=local payload DOES
    cost the expert stacks (smoothly — route-flip-blind); those rows are
    REPLACED by the empirical ones and recorded in provenance, non-expert
    rows stay untouched (moe_cb_design.md §3).
    """
    merged = dict(base)
    base_stats = dict(base.get("stats", {}) or {})
    base_costs = dict(base.get("costs", {}) or {})
    overlap = set(base_costs) & set(expert_costs)
    if overlap and not replace_experts:
        raise RuntimeError(
            f"hybrid merge collision: {len(overlap)} names costed by BOTH "
            f"the base payload and the expert empirical pass (e.g. "
            f"{sorted(overlap)[:3]}). The base run must omit routed experts "
            f"(or pass replace_experts for the CB-lane replace semantics).")
    canonical_formats = _canon_formats(formats)
    validate_cb_cost_provenance(
        base,
        canonical_formats,
        context=cb_serialization_context_from_env(),
        where="expert empirical merge base",
    )
    if overlap:
        for name in overlap:
            base_costs.pop(name)
            base_stats.pop(name, None)
        prov = dict(merged.get("provenance", {}) or {})
        prov["replaced_smooth_expert_rows"] = sorted(overlap)
        merged["provenance"] = prov
    base_stats.update(expert_stats)
    base_costs.update(expert_costs)
    merged["stats"] = base_stats
    merged["costs"] = base_costs
    merged["schema"] = SCHEMA
    merged["formats"] = canonical_formats
    return merged


def backfill_missing_from_base(
    payload: dict,
    base_cost: Mapping[str, object],
) -> list[str]:
    """Copy rows for names the payload lacks from the baseline cost pkl.

    Covers MTP / visual sidecars the AURA pass never sees (the synthesized
    MTP module lives outside the CausalLM the cost harness loads). Returns
    the backfilled names, and records them in provenance for honesty: these
    rows carry the baseline estimator, not the AURA adjoint.
    """
    formats = list(payload.get("formats", []) or [])
    context = cb_serialization_context_from_env()
    validate_cb_cost_provenance(
        payload,
        formats,
        context=context,
        where="expert empirical backfill destination",
    )
    validate_cb_cost_provenance(
        base_cost,
        formats,
        context=context,
        where="expert empirical backfill source",
    )
    base_costs = dict(base_cost.get("costs", {}) or {})
    base_stats = dict(base_cost.get("stats", {}) or {})
    added: list[str] = []
    for name, row in base_costs.items():
        if name in payload["costs"]:
            continue
        payload["costs"][name] = row
        if name in base_stats and name not in payload["stats"]:
            payload["stats"][name] = base_stats[name]
        added.append(name)
    return sorted(added)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Empirical routed-MoE expert cost (+ hybrid merge)")
    p.add_argument("--model", required=True)
    p.add_argument("--cost-mode", default="",
                   help="Pipeline COST_MODE stamped into "
                        "provenance['cost_mode'] (re-vet R2).")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats", default="NVFP4,FP8_DYNAMIC,BF16",
        help="Expert format menu. Non-passthrough formats are measured; "
        "BF16/FP8_SOURCE rows are passthrough-zero.")
    p.add_argument(
        "--format-plan",
        default=None,
        help="Identity-bound source-class format plan. Streaming measurement "
        "intersects the global menu for every serving unit and refuses a "
        "group whose members straddle planned menus.",
    )
    p.add_argument("--n-calib-samples", type=int, default=16)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument(
        "--dataset", default=None,
        help="Optional calibration source (HF id, .jsonl, .txt) via "
        "sensitivity_probe.load_calibration; default is the WikiText "
        "windowed loader (matches aura_cost).")
    p.add_argument("--expert-chunk", type=int, default=16,
                   help="Experts quantized per in-place RTN chunk.")
    p.add_argument(
        "--merge-base", default=None,
        help="AURA non-expert cost pkl to union the expert rows into "
        "(the hybrid recipe). Output = merged payload.")
    p.add_argument(
        "--backfill-base", default=None,
        help="Baseline incremental cost pkl; rows for names still missing "
        "after the merge (MTP/visual sidecars) are copied from it.")
    p.add_argument(
        "--replace-experts", action="store_true",
        help="CB-lane merge semantics: the COST_MODE=local base payload "
        "costs expert stacks smoothly (route-flip-blind); REPLACE those "
        "rows with the empirical ones (recorded in provenance) instead of "
        "treating the collision as an error.")
    p.add_argument(
        "--col-weights", default=None,
        help="Pickle {qname: per-input-column importance} (the CB "
        "exporter's imatrix). REQUIRED when the menu contains CB formats: "
        "their deliberate render is imatrix-weighted, and the measured "
        "unit-KL must be of the bytes the exporter ships.")
    p.add_argument(
        "--cb-ladder-interp", action="store_true",
        help="RD-law ladder interpolation for NVFP4_CB_K rungs (measure "
        "anchors + holdout, predict the rest; holdout-gated PER UNIT). "
        "Also enabled by PRISMAQUANT_CB_LADDER_INTERP=1. Default OFF — "
        "the law is recon-validated but KL-unproven (encode_tiers.md §B).")
    p.add_argument("--ladder-holdout-tol", type=float, default=0.10,
                   help="FLOOR on the max holdout relative error that "
                   "accepts a unit's ladder fit; the gate derives its own "
                   "tolerance from the anchors' residual noise "
                   "(encode_tiers.md B) and uses the larger. Above it the "
                   "unit measures every rung.")
    p.add_argument(
        "--expert-sample", type=int, default=0,
        help="Quantize only a stratified subsample of N experts per unit and "
        "extrapolate the unit KL by expert count. REFUTED for CB-fidelity "
        "menus (encode_tiers.md §C: the bf16 unit KL is a perturbation "
        "floor — S=1 of 256 already reads ~90%% of the full-stack KL, so "
        "count-scaling over-predicts ~10x and rung ranks drown in floor "
        "noise). Kept for floor-regime probing and for coarse-format menus "
        "where unit KLs sit far above the floor. 0 = full stack (default). "
        "For CB rungs use the LOCAL cost's PRISMAQUANT_EXPERT_COST_SAMPLE "
        "instead (MSE sampling is unbiased; KL sampling is not).")
    p.add_argument(
        "--max-units", type=int, default=0,
        help="Measure only the first N units (0 = all). Validation/"
        "sharding aid.")
    p.add_argument(
        "--unit-filter", default=None,
        help="Regex on the experts qname; only matching units are "
             "measured. Validation/sharding aid.")
    p.add_argument(
        "--checkpoint-dir", default=None,
        help="Durable identity-bound per-serving-unit checkpoint directory.")
    p.add_argument(
        "--resume", action="store_true",
        help="Validate and reuse completed qname-keyed unit shards; any "
             "identity mismatch refuses reuse and recomputation.")
    p.add_argument(
        "--streaming", action="store_true",
        help="Use the existing decoder-layer prefetch/cache context instead "
             "of loading the expanded source model resident.")
    p.add_argument(
        "--streaming-offload-dir", default=None,
        help="Streaming model work directory. Defaults below the checkpoint "
             "or output directory and is never placed in /tmp.")
    p.add_argument("--device", default="cuda")
    return p


def _streaming_model_identity(
    runner,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Stable source manifest used to bind expert end-to-end KL resumes."""
    from prismaquant.cost_streaming import build_streamed_model_identity

    return build_streamed_model_identity(
        runner,
        source_model,
        identity_cache_path=identity_cache_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume and not args.checkpoint_dir:
        raise SystemExit("--resume requires --checkpoint-dir")
    if args.format_plan and not args.streaming:
        raise SystemExit("--format-plan requires --streaming")

    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("expert_empirical_cost", args.device)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.model_profiles import detect_profile

    staged, _cleanup = stage_multimodal(args.model)
    # Detection is the authority for routed-expert membership and also
    # installs any architecture-owned vendored modelling override. Resolve it
    # before constructing the model; a fallback-after-load would recreate the
    # silent wrong-topology failure this empirical path exists to prevent.
    profile = detect_profile(staged)
    local_only = Path(staged).exists()
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    _log(f"loading {args.model} (staged={staged}) bf16 ...")
    model = None
    streamed_runner = None
    if args.streaming:
        from prismaquant.cost_streaming import build_streamed_causal_lm

        offload_dir = args.streaming_offload_dir
        if not offload_dir:
            anchor = Path(args.checkpoint_dir or args.output).parent
            offload_dir = str(anchor / "expert-cost-streaming-offload")
        streamed_runner = build_streamed_causal_lm(
            staged,
            device=torch.device(args.device),
            dtype=torch.bfloat16,
            offload_folder=offload_dir,
            profile=profile,
        )
        model = streamed_runner.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            staged, dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=local_only, attn_implementation="eager",
            device_map=args.device,
        ).eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
    # Per-expert-on-disk -> packed-in-class checkpoints (Qwen3.5-MoE /
    # Ornith): the text class lacks the per-expert->packed mapper, so the
    # packed params load ZERO-INITIALIZED — the zero-expert calibration bug
    # (quantizing zeros is a no-op; every unit KL reads exactly 0). Fill from
    # the source shards; measure_expert_unit_costs also hard-fails on
    # all-zero stacks so this class of silent garbage can never recur.
    from prismaquant.layer_streaming import fill_packed_experts_from_source
    # NOTE: pass the ORIGINAL model dir, not the staged text-only view — the
    # staging rewrites index keys to `model.layers.*` while the profile's
    # source_tensor_name keeps the checkpoint's own nesting
    # (`model.language_model.*`), so against the staged index the fill's
    # prefix filter silently matches nothing (how production_recache calls it).
    fill_src = args.model if Path(args.model).exists() else staged
    filled = (
        0
        if streamed_runner is not None
        else fill_packed_experts_from_source(
            model, fill_src, profile, progress=True
        )
    )
    if filled:
        _log(f"filled {filled} packed-expert params from source shards")

    if args.dataset:
        from prismaquant.sensitivity_probe import load_calibration
        calib = load_calibration(
            tok, args.dataset, args.n_calib_samples, args.calib_seqlen,
            calib_seed=args.calib_seed)
    else:
        from prismaquant.calibration_data import (
            load_wikitext_calibration_windowed,
        )
        calib = load_wikitext_calibration_windowed(
            tok, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed)
    calib = calib.to(args.device)

    formats = _canon_formats(
        [f for f in args.formats.split(",") if f.strip()])
    source_format_plan = None
    if args.format_plan:
        from prismaquant.source_class_format_plan import load_format_plan

        source_format_plan = load_format_plan(args.format_plan)
    col_weights = None
    if args.col_weights:
        with open(args.col_weights, "rb") as fh:
            col_weights = {k: torch.as_tensor(v)
                           for k, v in pickle.load(fh).items()}
    ladder_interp = bool(args.cb_ladder_interp) or (
        os.environ.get("PRISMAQUANT_CB_LADDER_INTERP", "0") == "1")
    if col_weights is None:
        col_weights = {}
    if streamed_runner is not None:
        try:
            streamed_model_identity = None
            if args.checkpoint_dir:
                streamed_model_identity = _streaming_model_identity(
                    streamed_runner,
                    args.model,
                    identity_cache_path=(
                        Path(args.checkpoint_dir)
                        / "streamed_model_identity.json"
                    ),
                )
            eval_driver = os.environ.get(
                "PRISMAQUANT_EXPERT_EVAL_DRIVER", "window"
            ).strip().lower()
            if eval_driver not in ("window", "forked"):
                raise SystemExit(
                    f"PRISMAQUANT_EXPERT_EVAL_DRIVER={eval_driver!r} is not "
                    "one of: window, forked"
                )
            if eval_driver == "forked":
                # O(1) body streams instead of O(units): see
                # measure_expert_unit_costs_forked. Scope guards live in the
                # driver (stateless pass, no CB, unpacked units); the CLI
                # features it does not take are refused here, loudly.
                refused = {
                    "--expert-sample": args.expert_sample,
                    "--cb-ladder-interp": ladder_interp,
                    "--format-plan": source_format_plan is not None,
                }
                on = sorted(k for k, v in refused.items() if v)
                if on:
                    raise SystemExit(
                        "PRISMAQUANT_EXPERT_EVAL_DRIVER=forked does not "
                        f"support {on}; unset them or use the window driver"
                    )
                stats, costs, unit_kls = measure_expert_unit_costs_forked(
                    streamed_runner,
                    profile,
                    calib,
                    formats,
                    expert_chunk=args.expert_chunk,
                    col_weights=col_weights,
                    max_units=args.max_units,
                    unit_filter=args.unit_filter,
                    checkpoint_dir=args.checkpoint_dir,
                    resume=args.resume,
                    model_identity=streamed_model_identity,
                )
            else:
                stats, costs, unit_kls = measure_expert_unit_costs_streamed(
                    streamed_runner,
                    profile,
                    calib,
                    formats,
                    expert_chunk=args.expert_chunk,
                    col_weights=col_weights,
                    ladder_interp=ladder_interp,
                    ladder_tol=args.ladder_holdout_tol,
                    expert_sample=args.expert_sample,
                    max_units=args.max_units,
                    unit_filter=args.unit_filter,
                    checkpoint_dir=args.checkpoint_dir,
                    resume=args.resume,
                    model_identity=streamed_model_identity,
                    checkpoint_identity_extra=(
                        {
                            "source_format_plan_identity_sha256": (
                                source_format_plan.identity_sha256
                            )
                        }
                        if source_format_plan is not None else None
                    ),
                    formats_by_qname=(
                        source_format_plan.formats_by_qname()
                        if source_format_plan is not None else None
                    ),
                )
        finally:
            streamed_runner.shutdown()
    else:
        stats, costs, unit_kls = measure_expert_unit_costs(
            model, profile, calib, formats, expert_chunk=args.expert_chunk,
            col_weights=col_weights, ladder_interp=ladder_interp,
            ladder_tol=args.ladder_holdout_tol,
            expert_sample=args.expert_sample, max_units=args.max_units,
            unit_filter=args.unit_filter)
    added_cw = getattr(
        measure_expert_unit_costs, "last_added_col_weights", [])
    if added_cw and args.col_weights:
        # Persist the synthesized packed-expert imatrix entries back to the
        # SHARED col-weights pickle: the exporter must ship the identical
        # weighting the cost measured (lockstep contract). Atomic replace.
        tmp = args.col_weights + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({k: v.cpu() if hasattr(v, "cpu") else v
                         for k, v in col_weights.items()}, fh)
        os.replace(tmp, args.col_weights)
        _log(f"persisted {len(added_cw)} synthesized imatrix entries into "
             f"{args.col_weights}")

    provenance = {
        "schema": SCHEMA,
        "git_commit": _git_commit(),
        "model": args.model,
        "dataset": args.dataset or f"wikitext:{args.calib_split}",
        "n_calib_samples": int(calib.shape[0]),
        "calib_seqlen": int(calib.shape[1]),
        "calib_seed": args.calib_seed,
        "calib_sha256": hashlib.sha256(
            calib.cpu().numpy().tobytes()).hexdigest(),
        "expert_units": len(unit_kls),
        "unit_kls": unit_kls,
        "formats_measured": [
            f for f in formats if f not in PASSTHROUGH_FORMATS],
        "col_weights": args.col_weights,
        "cb_ladder_interp": ladder_interp,
        "encode_tier": os.environ.get("PRISMAQUANT_CB_ENCODE_TIER"),
        "expert_sample": int(args.expert_sample),
        "max_units": int(args.max_units),
        "unit_filter": args.unit_filter,
        # Cross-family ladder symmetry (ultraplan P5a item 2). None when no
        # ladder ran; the allocator reads it back through
        # cb_ladder_cross_family.cross_family_verdict_from_cost_payload and
        # republishes it in its own diagnostics/selection provenance.
        CROSS_FAMILY_PROVENANCE_KEY: getattr(
            measure_expert_unit_costs, "last_cross_family_verdict", None),
        **cb_cost_provenance(formats),
    }

    if args.merge_base:
        with open(args.merge_base, "rb") as fh:
            base = pickle.load(fh)
        payload = merge_cost_payloads(
            base, stats, costs, formats=formats,
            replace_experts=bool(args.replace_experts))
        prov = dict(payload.get("provenance", {}) or {})
        prov["expert_empirical_cost"] = provenance
        prov["merge_base"] = args.merge_base
        payload["provenance"] = prov
        _log(f"merged {len(costs)} expert member rows into "
             f"{args.merge_base} ({len(payload['costs'])} total)")
    else:
        payload = {
            "schema": SCHEMA,
            "formats": formats,
            "stats": stats,
            "costs": costs,
            "provenance": provenance,
        }

    if args.backfill_base:
        with open(args.backfill_base, "rb") as fh:
            base_cost = pickle.load(fh)
        added = backfill_missing_from_base(payload, base_cost)
        prov = dict(payload.get("provenance", {}) or {})
        prov["backfilled_from_base"] = added
        prov["backfill_base"] = args.backfill_base
        payload["provenance"] = prov
        if added:
            _log(f"backfilled {len(added)} sidecar rows from "
                 f"{args.backfill_base}: {added[:5]}"
                 f"{' ...' if len(added) > 5 else ''}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"wrote {out}: {len(payload['costs'])} cost rows "
         f"({len(unit_kls)} expert units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
