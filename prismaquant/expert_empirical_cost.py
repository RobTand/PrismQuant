"""Empirical packed-MoE expert costs for the AURA hybrid recipe.

AURA's smooth per-Linear cost is route-flip-blind on routed experts (Step A,
2026-06-29: Spearman drops 0.45->0.35 under faithful dW; predicted NVFP4/FP8
ratios 2-49x vs measured 1.1-1.5x), so expert costs are MEASURED, not
modeled: per MoE layer the serving unit = all packed expert tensors of that
module (they must share one format — vLLM FusedMoE constraint), and the unit
cost of a format is the end-to-end mean-token KL(BF16 || unit-quantized)
with everything else left at source precision. The unit KL is split across
the member tensors proportionally to n_params so the allocator's per-member
aggregation charges it exactly once.

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
from prismaquant.render_score import persisted_cell_score_fields

SCHEMA = "prismaquant.expert_empirical_cost.v1"
PASSTHROUGH_FORMATS = {"BF16", "FP8_SOURCE"}
# CB families render the WHOLE stack in one qdq call (the export convention:
# fp4 derives one per-stack global; fp8 per-row scales) — measured render ==
# shipped bytes, never chunked (moe_cb_design.md §3).
_CB_FAMILIES = {"nvfp4_cb", "fp8_cb"}
# Both CB families carry k-rung ladders (k index bits per 8-weight vector).
# Ladders are PER (family, mode) — NVFP4_CB_K, NVFP4_CB_S, FP8_CB_K are
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


def _cell_score_fields(
    cell_scores: Mapping[str, object] | None,
    qname: str,
    fmt: str,
) -> dict[str, object]:
    if not isinstance(cell_scores, Mapping):
        return {}
    per_name = cell_scores.get(qname)
    if not isinstance(per_name, Mapping):
        return {}
    for alias in fr.aliases_for(fmt):
        record = per_name.get(alias)
        if isinstance(record, Mapping):
            return persisted_cell_score_fields(record)
    return {}


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
        for i in range(0, calib_ids.shape[0], bs):
            logits = model(calib_ids[i:i + bs]).logits.float()
            out.append(F.log_softmax(logits, dim=-1).cpu())
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
                parent_mod, "e_score_correction_bias", None))
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

        def qdq(weight, col_weights=None):
            return cb_quantize_dequantize_for_context(
                spec,
                weight,
                context=context,
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
                    w[idx].float(), col_weights=cw_s).to(w.dtype)
            else:
                w.copy_(qdq(
                    w.float(), col_weights=cw_dev).to(w.dtype))
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
                model(calib_ids[i:i + bs]).logits.float(), -1)
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
    cell_scores: Mapping[str, object] | None = None,
) -> tuple[dict, dict, dict]:
    """Measure per-serving-unit empirical KL costs for packed-MoE experts.

    Returns ``(stats, costs, unit_kls)`` where stats/costs are
    allocator-payload row dicts keyed by full member names and ``unit_kls``
    maps ``experts_qname -> {fmt: unit_kl}``.
    """
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    units = [
        (qn, m) for qn, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    if unit_filter:
        pat = re.compile(unit_filter)
        units = [(qn, m) for qn, m in units if pat.search(qn)]
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
        # Capture module inputs during the baseline forwards and synthesize
        # any missing packed-expert imatrix entries (down_proj is NEVER in
        # the harvested cache — its input is the per-expert intermediate).
        if col_weights is None:
            col_weights = {}
        baseline, unit_x = _baseline_logprobs(
            model, calib_ids, capture_units=units)
        added = ensure_unit_col_weights(model, units, col_weights, unit_x)
        if added and progress:
            _log(f"synthesized {len(added)} packed-expert imatrix entries "
                 f"(module-input pool / down_proj replay): "
                 f"{added[:4]}{'...' if len(added) > 4 else ''}")
        measure_expert_unit_costs.last_added_col_weights = added
        del unit_x
    else:
        baseline = _baseline_logprobs(model, calib_ids)
        measure_expert_unit_costs.last_added_col_weights = []
    ladder = _cb_ladder_split(measured_fmts) if ladder_interp else None
    # Visible accept/reject rate for the holdout gate (R20): a ladder that
    # is silently rejecting most units is paying full measurement cost PLUS
    # the anchors, and the operator must be able to see that from the log.
    ladder_accept = 0
    ladder_reject = 0
    for qn, mod in units:
        pnames = list(_packed_experts_param_names(mod, profile))
        n_params_unit = sum(int(getattr(mod, pn).numel()) for pn in pnames)
        num_experts = int(getattr(mod, pnames[0]).shape[0])
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
                    f"before measuring.")
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
            out = _unit_kl(
                model, calib_ids, baseline, mod, pnames, fmt,
                expert_chunk=expert_chunk, col_weights=col_weights,
                unit_qname=qn, sample_idx=sample_idx,
                per_window=ladder is not None)
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
        for pn in pnames:
            tensor = getattr(mod, pn)
            npm = int(tensor.numel())
            full = f"{qn}.{pn}" if qn else pn
            shape = list(tensor.shape)
            stats[full] = {
                # h_trace is meaningless for an empirically-costed unit; the
                # allocator consumes predicted_dloss directly. 0.0 marks
                # "do not fall back to h_trace x weight_mse" for this row.
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": int(shape[2]),
                "out_features": int(shape[1]),
                "num_experts": num_experts,
                "_packed_experts_module": qn,
                "_packed_param": pn,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                # Split the UNIT cost across members by n_params so the
                # per-member sum re-assembles exactly one unit KL.
                entry = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
                entry.update(_cell_score_fields(cell_scores, full, fmt))
                row[fmt] = entry
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
            f"{sorted(overlap)[:3]}). The base run must omit packed experts "
            f"(or pass replace_experts for the CB-lane replace semantics).")
    canonical_formats = _canon_formats(formats)
    validate_cb_cost_provenance(
        base,
        canonical_formats,
        context=cb_serialization_context_from_env(),
        where="expert empirical merge base",
    )
    expert_costs_for_merge = dict(expert_costs)
    if overlap:
        for name in overlap:
            base_row = base_costs[name]
            expert_row = expert_costs_for_merge[name]
            if isinstance(base_row, Mapping) and isinstance(expert_row, Mapping):
                # Enrichment belongs to the merged payload.  The empirical
                # payload is also returned to callers as a standalone result,
                # so do not mutate its nested rows while preserving score
                # metadata from the smooth base table.
                expert_row = dict(expert_row)
                for fmt, base_entry in base_row.items():
                    if not isinstance(base_entry, Mapping):
                        continue
                    target = expert_row.get(fmt)
                    if not isinstance(target, Mapping):
                        continue
                    target = dict(target)
                    for field, value in persisted_cell_score_fields(
                        base_entry
                    ).items():
                        target.setdefault(field, value)
                    expert_row[fmt] = target
                expert_costs_for_merge[name] = expert_row
            base_costs.pop(name)
            base_stats.pop(name, None)
        prov = dict(merged.get("provenance", {}) or {})
        prov["replaced_smooth_expert_rows"] = sorted(overlap)
        merged["provenance"] = prov
    base_stats.update(expert_stats)
    base_costs.update(expert_costs_for_merge)
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
        description="Empirical packed-MoE expert cost (+ hybrid merge)")
    p.add_argument("--model", required=True)
    p.add_argument("--cost-mode", default="",
                   help="Pipeline COST_MODE stamped into "
                        "provenance['cost_mode'] (re-vet R2).")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats", default="NVFP4,FP8_DYNAMIC,BF16",
        help="Expert format menu. Non-passthrough formats are measured; "
        "BF16/FP8_SOURCE rows are passthrough-zero.")
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
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("expert_empirical_cost", args.device)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.model_profiles import detect_profile_with_warning

    staged, _cleanup = stage_multimodal(args.model)
    local_only = Path(staged).exists()
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    _log(f"loading {args.model} (staged={staged}) bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        staged, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=local_only, attn_implementation="eager",
        device_map=args.device,
    ).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    profile = detect_profile_with_warning(
        staged, entrypoint="expert-empirical-cost")
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
    filled = fill_packed_experts_from_source(
        model, fill_src, profile, progress=True)
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
    col_weights = None
    if args.col_weights:
        with open(args.col_weights, "rb") as fh:
            col_weights = {k: torch.as_tensor(v)
                           for k, v in pickle.load(fh).items()}
    ladder_interp = bool(args.cb_ladder_interp) or (
        os.environ.get("PRISMAQUANT_CB_LADDER_INTERP", "0") == "1")
    if col_weights is None:
        col_weights = {}
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
