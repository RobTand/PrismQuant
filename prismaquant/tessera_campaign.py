"""Price Tessera's continuous rate axis by anchor campaign.

Four families address thousands of rungs at every shape, and one encode is
seconds.
Rendering the menu is not a cost model, it is a week of a shared GPU.  So this
stage renders a small **measured anchor** set per (unit, family) and fills the
rest of the axis from ``tessera_rate_surface``, which interpolates monotonically
in ``(q256, log2 dloss)`` between bracketing anchors and **refuses to
extrapolate**.  How many anchors that is, is not a constant: round 1 measures
the two endpoints and the middle of each family's legal range, and every
following round adds one anchor wherever leave-one-anchor-out says the
interpolation is worst, until the LOO gate closes or the round budget runs out.

Priced as served
----------------
Every measured rung is scored by the production scorer,
``production_weight_cache._local_forward_render_score`` -- the same function the
render-cost stage uses -- with the **route's own activation quantiser** applied
to the calibration rows.  A Tessera rung on the E2M1 grid over a block plane
decodes to an NVFP4 tensor and executes W4A4, so its cost is measured against an
A4 input; a rung on E4M3 over a CHANNEL plane executes W8A8 and is measured
against a per-token FP8 input.  The same *weight* rate therefore costs
differently on the two routes, which is the whole reason the allocator may not
rank them on weight error alone.

Rendering identity, and the wire
--------------------------------
The render is not ``render_tessera_weight``'s reconstruction; it is
``read_unit_artifact(encode_linear(...).blob)`` -- **the bytes, decoded**.  So the
cache entry holds the wire beside the dequantised render, the export leg writes
exactly the bytes this cost was measured on, and the identity holds by
construction rather than by two code paths agreeing (principle 8).

What this stage does NOT do
---------------------------
It measures ``output_mse`` under the route's activation contract, which is the
render-cost currency (``COST_MODE=production-render-score``'s
``--score-field output_mse``).  It is not the AURA adjoint: AURA prices ``dW``
against a KL-Fisher weight gradient and applies the A side afterwards as a
calibrated per-family multiplier, and a Tessera family has no such calibration
yet.  The payload declares its own currency so nothing downstream can mistake
one for the other.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import tessera_hessian as th

__all__ = [
    "SCHEMA",
    "CampaignAnchor",
    "anchor_schedule",
    "campaign_cost_payload",
    "main",
    "next_anchor_rate",
]

SCHEMA = "prismaquant.tessera_campaign_cost.v1"

#: The currency every row is measured or interpolated in.  Named, because the
#: allocator's cost-source precedence reads a field, not a convention.
CURRENCY = "output_mse_under_route_activation_contract"

#: Round 1's anchors: both endpoints plus the middle.  The endpoints are
#: mandatory rather than chosen -- a surface that does not span its family's
#: legal range would have to extrapolate to price the ends, and
#: ``TesseraRateSurface.predict`` refuses that.  Three is the minimum at which
#: leave-one-anchor-out has anything to drop.
ROUND_ONE_ANCHORS = 3


@dataclass(frozen=True)
class CampaignAnchor:
    """One measured (unit, family, rung) point."""

    qname: str
    family: str
    format_name: str
    body_rate_q256: int
    dloss: float
    dloss_stderr: float
    memory_bytes: int
    bits_per_param: float
    activation_contract: str
    activation_quantized: bool
    wire_bytes: int
    seconds: float
    #: Was a Hessian actually applied to these bytes?  A property of the RUNG'S
    #: WIRE and not of the run: only a CHANNEL scale plane admits LDLQ and the
    #: H refit, so every E2M1 rung is False even under ``--hessian require``.
    #: Stamped per row rather than per table, because a mixed-family campaign
    #: is legitimately half H-aware and a table-level flag would have to lie in
    #: one direction or the other.
    hessian_applied: bool = False


def anchor_schedule(lo: int, hi: int, count: int) -> list[int]:
    """``count`` rungs spanning ``[lo, hi]``, endpoints included.

    Evenly spaced in rate because the surface interpolates in rate; the
    *adaptive* rounds are what put anchors where the curve actually bends, and
    they are driven by measured leave-one-out error rather than by a prior
    about where a rate-distortion curve is steep.
    """
    if hi <= lo:
        return [lo]
    count = max(2, int(count))
    if count == 2:
        return [lo, hi]
    step = (hi - lo) / (count - 1)
    out = sorted({int(round(lo + i * step)) for i in range(count)})
    out[0], out[-1] = lo, hi
    return sorted(set(out))


def next_anchor_rate(
    rates: Sequence[int], loo: Mapping[str, object],
) -> "int | None":
    """The rung the next round should measure, or None if there is no room.

    The interval to split is the one whose *interior* anchor the rest of the
    surface predicts worst -- leave-one-anchor-out's own per-anchor error, so
    the campaign spends its next encode where the interpolation is measurably
    failing rather than where a heuristic guesses it might.  With fewer than
    three anchors LOO has no interior to report, and the widest gap is the only
    information available; that fallback is stated rather than silent.
    """
    ordered = sorted(int(r) for r in rates)
    if len(ordered) < 2:
        return None

    def midpoint(left: int, right: int) -> "int | None":
        if right - left < 2:
            return None
        mid = (left + right) // 2
        return mid if mid not in ordered else None

    per_anchor = loo.get("per_anchor") if isinstance(loo, Mapping) else None
    if isinstance(per_anchor, Sequence) and per_anchor:
        worst = max(per_anchor, key=lambda e: abs(float(e["log2_error"])))
        rate = int(worst["q256"])
        index = ordered.index(rate)
        left = midpoint(ordered[index - 1], rate)
        right = midpoint(rate, ordered[index + 1])
        for candidate in (left, right):
            if candidate is not None:
                return candidate
    gaps = sorted(
        ((right - left, left, right)
         for left, right in zip(ordered, ordered[1:])),
        reverse=True,
    )
    for _width, left, right in gaps:
        candidate = midpoint(left, right)
        if candidate is not None:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _encode_and_render(weight, format_name: str, *, activation_kwargs=None,
                       hessian_required: bool = True, recipe=None):
    """``(render, blob)`` for one rung: the bytes, and what they decode to.

    A thin adapter over ``tessera_render.encode_tessera_unit``, which is the
    **one** function in this tree that calls Tessera's byte path.  Everything
    that made this function worth having -- the render is
    ``read_unit_artifact(blob)``, i.e. the bytes decoded rather than a second
    reconstruction (principle 8), and ``verify=False`` because the tensor
    ``verify`` would compare against is one nothing downstream sees -- now
    lives there, so a caller cannot reach the encoder around it and skip the
    Hessian contract.
    """
    from .tessera_render import encode_tessera_unit

    return encode_tessera_unit(
        weight, format_name,
        activation_kwargs=activation_kwargs,
        hessian_required=bool(hessian_required), verify=False,
        recipe=recipe,
    )


def _measure_anchor(
    *, qname: str, weight, activations, format_name: str, cache, wire_dir: Path,
    activation_kwargs_for=None, hessian_required: bool = True,
):
    """Render one rung, price it as served, and store the wire beside it.

    ``activation_kwargs_for`` is a callable ``(qname, scale_plane) -> encoder
    kwargs`` -- the block-LDL of that unit's regularised ``XᵀX`` and the
    plane's refit metric -- which the caller memoises per ``(unit, plane)``.
    Per unit alone would be wrong and silently so: the refit objective is keyed
    by scale plane, so a memo that ignored the plane would price a unit's
    second family under the first family's objective.  Per rung would be
    wasteful: no rate reaches that call, and a twelve-anchor surface would
    otherwise factorise the same Hessian twelve times.  The plane is constant
    across every rung of a family and differs between families, so the bound is
    one factorisation per unit per family-plane.

    A **missing key is a hard failure**, never a silently H-free encode: this
    codebase has already been bitten once by a render whose activation lookup
    missed and quietly fell back to RTN, raising nothing, and an H-free encode
    of a rung whose shipping bytes are H-aware is exactly that bug with
    different bytes.
    """
    import torch

    from . import format_registry as fr
    from .production_weight_cache import (
        _local_forward_render_score, _store_rendered_weight_entry,
    )
    from .tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from .tessera_render import HessianContractError

    from .tessera_render import rung_accepts_hessian

    spec = fr.get_format(format_name)
    family, rung = parse_tessera_format_name(format_name)
    # ONE resolve for this anchor. The plane the kwargs are built on, the plane
    # the predicate reads and the plane the encode writes are this object --
    # not three lookups that agree only while nothing clears the recipe memo.
    wire = tessera_wire_recipe(family, rung)
    activation_kwargs = None
    # Whether an H can be applied is a property of the RUNG'S WIRE, not of the
    # run, and it is DERIVED from what the pinned ActivationSource emits for
    # that wire's plane rather than restated here (``rung_accepts_hessian``).
    # It reads True on every plane PrismaQuant resolves at this pin.
    hessian_required = bool(hessian_required) and rung_accepts_hessian(
        format_name, wire)
    if hessian_required:
        if activation_kwargs_for is None:
            raise HessianContractError(
                f"{qname}: hessian_required=True but no activation-kwargs "
                "source was passed to _measure_anchor")
        activation_kwargs = activation_kwargs_for(qname, wire.scale_plane)
        if not activation_kwargs:
            raise HessianContractError(
                f"{qname}: no Hessian for this Linear. A lookup that misses "
                "must not fall through to a weights-only encode.")
    started = time.time()
    render, blob = _encode_and_render(
        weight, format_name, recipe=wire, activation_kwargs=activation_kwargs,
        hessian_required=hessian_required,
    )
    elapsed = time.time() - started

    score, metric, quantized, _clipped = _local_forward_render_score(
        reference_weight=weight,
        rendered_weight=render,
        activations=activations,
        activation_quantize=spec.activation_quantize_dequantize,
        # Tessera's A side is the serving format's own dynamic quantiser, taken
        # by reference; no calibrated static clip applies (that is NVFP4's
        # tensor-level scale, and ``_format_uses_static_activation_clip``
        # names only NVFP4 for it).
        activation_max_abs=None,
    )
    if metric != "output_mse":
        raise RuntimeError(f"unexpected render-score metric {metric!r}")
    hessian_applied = bool(activation_kwargs)

    _store_rendered_weight_entry(
        weights=cache.weights,
        qname=qname,
        fmt=format_name,
        tensor=render,
        cache_dir_path=Path(cache.cache_dir) if cache.cache_dir else None,
        weight_dtype=torch.bfloat16,
    )
    # The wire, beside the render.  A ``.tessera`` shard per (qname, rung),
    # named the way the cache names its weight shards, so the export leg can
    # find the exact bytes this row was priced on instead of re-encoding.
    wire_path = wire_dir / f"{qname.replace('.', '__')}__{format_name}.tessera"
    tmp = wire_path.with_suffix(".tessera.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, wire_path)

    bits = spec.bits_for_shape(tuple(weight.shape))
    return CampaignAnchor(
        qname=qname,
        family=family.name,
        format_name=format_name,
        body_rate_q256=int(rung),
        dloss=float(score),
        # One calibration draw: the stderr a single measurement supports is
        # zero, and inventing one would put a fabricated uncertainty into the
        # allocator's UCB hedge.  Reported as zero and named as such.
        dloss_stderr=0.0,
        memory_bytes=int(spec.memory_bytes_for_shape(tuple(weight.shape))),
        bits_per_param=float(bits) / max(1, int(weight.numel())),
        activation_contract=str(spec.act_dtype_name or "a16"),
        activation_quantized=bool(quantized),
        wire_bytes=len(blob),
        seconds=elapsed,
        hessian_applied=hessian_applied,
    )


# ---------------------------------------------------------------------------
# The dense payload
# ---------------------------------------------------------------------------

def campaign_cost_payload(
    anchors: Mapping[str, Mapping[str, list[CampaignAnchor]]],
    menus: Mapping[str, list],
    *,
    loo: Mapping[str, Mapping[str, dict]],
    provenance: dict,
) -> dict:
    """Turn measured anchors plus a legal menu into a cost payload.

    Measured rungs keep their measurement and are marked so; every other legal
    rung inside the measured envelope is interpolated and marked so; a rung
    outside the envelope is **omitted**, because ``TesseraRateSurface.predict``
    refuses to extrapolate and a menu row the surface will not price is a row
    nothing measured.
    """
    from .tessera_rate_surface import (
        PROVENANCE_INTERPOLATED, PROVENANCE_MEASURED, TesseraRateSurface,
    )

    # One identity for every row this payload writes, so a cost table that is
    # half H-aware and half weights-only is detectable downstream instead of
    # being an invisible merge of two encoders -- the exact shape of the
    # encoder-drift bug this project has already paid for once.
    _prov = dict(provenance.get("provenance", {}))
    _h = dict(_prov.get("hessian", {})) if isinstance(
        _prov.get("hessian"), dict) else {}
    hessian_identity = {
        "supplied": bool(_h.get("supplied", False)),
        # The legacy spelling, kept because ``assert_uniform_hessian_identity``
        # compares tables written before Tessera's triple existed.
        "text_sha": _h.get("text_sha"),
        "token_count": _h.get("token_count"),
        # Tessera's own required triple, so a row can be checked against the
        # ActivationSource that would encode it.
        "text_sha256": _h.get("text_sha256"),
        "fit_ids_sha256": _h.get("fit_ids_sha256"),
        "fit_tokens": _h.get("fit_tokens"),
        "kwarg": tuple(_h.get("kwargs", ())) or _h.get("kwarg"),
    }

    costs: dict[str, dict[str, dict]] = {}
    formats: set[str] = set()
    surfaces: dict[str, dict[str, TesseraRateSurface]] = {}
    refused: list[dict] = []
    for qname, by_family in anchors.items():
        rows: dict[str, dict] = {}
        measured_names: set[str] = set()
        for family, family_anchors in by_family.items():
            ordered = sorted(family_anchors, key=lambda a: a.body_rate_q256)
            for anchor in ordered:
                rows[anchor.format_name] = {
                    # ``output_mse`` ONLY, deliberately no ``predicted_dloss``.
                    # In this codebase ``output_mse`` is a raw MSE and
                    # ``predicted_dloss`` is already the 1/2*h_trace*mse
                    # product; writing the MSE into both fields would have the
                    # weight-only branch price a Tessera rung ~h_trace/2 times
                    # low against every other format on the menu. Leaving the
                    # field out puts these rows on the same branch, in the same
                    # currency, as production_render_cost's.
                    "output_mse": anchor.dloss,
                    "output_mse_measured": True,
                    "cost_source": "tessera_campaign_measured",
                    "currency": CURRENCY,
                    "tessera_provenance": PROVENANCE_MEASURED,
                    "tessera_family": family,
                    "tessera_body_rate_q256": anchor.body_rate_q256,
                    "activation_quantized": anchor.activation_quantized,
                    "activation_contract": anchor.activation_contract,
                    "wire_bytes": anchor.wire_bytes,
                    "encode_seconds": anchor.seconds,
                    "hessian_identity": {
                        **hessian_identity, "applied": bool(anchor.hessian_applied),
                    },
                }
                measured_names.add(anchor.format_name)
                formats.add(anchor.format_name)
            try:
                surface = TesseraRateSurface(
                    unit_name=qname,
                    family=family,
                    layout="tight",
                    currency=CURRENCY,
                    anchor_q256=tuple(a.body_rate_q256 for a in ordered),
                    anchor_dloss=tuple(a.dloss for a in ordered),
                    anchor_stderr=tuple(a.dloss_stderr for a in ordered),
                )
            except Exception as exc:
                # A non-monotone anchor set is a measurement fact, and the
                # surface refuses to launder it into a cost.  Record it and
                # keep the measured rows; the family is then priced only where
                # it was measured, which is the honest reduction.
                refused.append({
                    "qname": qname, "family": family,
                    "reason": "non_interpolable_anchors",
                    "detail": str(exc),
                    "anchor_q256": [a.body_rate_q256 for a in ordered],
                    "anchor_dloss": [a.dloss for a in ordered],
                })
                continue
            surfaces.setdefault(qname, {})[family] = surface
            low, high = surface.q256_range
            for rung in menus.get(qname, []):
                if rung.family != family or rung.format_name in measured_names:
                    continue
                if not low <= rung.body_rate_q256 <= high:
                    continue
                rows[rung.format_name] = {
                    # Same currency as the measured rows above, and for the
                    # same reason: no ``predicted_dloss`` field.
                    "output_mse": surface.predict(rung.body_rate_q256),
                    "output_mse_measured": False,
                    "cost_source": "tessera_campaign_interpolated",
                    "currency": CURRENCY,
                    "tessera_provenance": PROVENANCE_INTERPOLATED,
                    "tessera_family": family,
                    "tessera_body_rate_q256": rung.body_rate_q256,
                    "activation_contract": rung.admission.activation_contract,
                    # An interpolated row inherits the applicability of the
                    # anchors it was interpolated between; they share a family,
                    # so they share a scale plane and therefore an answer.
                    "hessian_identity": {
                        **hessian_identity,
                        "applied": bool(ordered[0].hessian_applied),
                    },
                }
                formats.add(rung.format_name)
        if rows:
            costs[qname] = rows
    payload = dict(provenance)
    payload.update({
        "schema": SCHEMA,
        "costs": costs,
        "formats": sorted(formats),
        "currency": CURRENCY,
        "leave_one_anchor_out": {
            q: {f: dict(v) for f, v in by_f.items()} for q, by_f in loo.items()
        },
        "non_interpolable": refused,
    })
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_activations(model, targets, tokens, max_rows: int, device,
                         *, want_hessian: bool = False):
    """One forward pass per calibration batch, per Linear.

    Returns ``(rows, hessians, token_counts)``.

    Two different things come out of the same hook, and they have different
    row budgets on purpose:

    * ``rows`` is capped at ``max_rows`` because it feeds
      ``_local_forward_render_score``, whose cost is linear in rows and whose
      job is to *rank* rungs.
    * ``hessians`` is ``XᵀX`` accumulated over **every** calibration row the
      Linear sees, because it feeds the encoder, whose job is to *build* the
      bytes.  Capping it at ``max_rows`` would hand a 3072-column
      ``down_proj`` a rank-256 Hessian -- rank-deficient by a factor of
      twelve, and wrong in a way no downstream check would catch.  So the
      accumulation runs before the keep's early return, not after it.

    Accumulated in fp32 on the model's device and moved to CPU once at the
    end, matching how the kept rows are handled.
    """
    import torch

    store: dict[str, list] = {name: [] for name in targets}
    kept: dict[str, int] = {name: 0 for name in targets}
    hess: dict[str, object] = {name: None for name in targets}
    seen: dict[str, int] = {name: 0 for name in targets}
    handles = []

    def make_hook(name):
        def hook(_module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            flat = x.detach().reshape(-1, x.shape[-1])
            if want_hessian:
                # Every row, before any cap: see the docstring.
                f32 = flat.to(dtype=torch.float32)
                gram = f32.t() @ f32
                if hess[name] is None:
                    hess[name] = gram
                else:
                    hess[name] += gram
                seen[name] += int(f32.shape[0])
            room = max_rows - kept[name]
            if room <= 0:
                return
            take = flat[:room].to(dtype=torch.float32, device="cpu")
            store[name].append(take)
            kept[name] += int(take.shape[0])
        return hook

    modules = dict(model.named_modules())
    for name in targets:
        handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            for batch in tokens:
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
    rows = {
        name: (torch.cat(chunks, dim=0) if chunks else None)
        for name, chunks in store.items()
    }
    hessians = {
        name: (None if h is None else h.to(device="cpu"))
        for name, h in hess.items()
    } if want_hessian else {}
    return rows, hessians, dict(seen)


def _calibration_tokens(model_path: str, n: int, seqlen: int, seed: int):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    # The corpus text travels beside the ids: Tessera's identity triple wants
    # both, and it is right to want both -- two tokenizers over one corpus are
    # two calibrations that a text sha alone would call the same, and one
    # tokenizer over two corpora is two calibrations an id sha would separate
    # only by luck.
    generator = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(int(n)):
        start = int(torch.randint(
            0, max(1, ids.shape[0] - seqlen - 1), (1,), generator=generator,
        ).item())
        out.append(ids[start:start + seqlen].unsqueeze(0))
    return out, text


def expand_menus_for_targets(weights, targets, *, mode, tp_degree,
                             parallel_kind) -> dict[str, list]:
    """One Tessera menu per distinct shape, shared across same-shape units.

    ``expand_tessera_menu`` takes nothing but the shape and the run
    configuration -- no argument identifies the unit, and ``MenuRung`` carries
    no unit field -- so two units of one shape get identical lists.  Units
    repeat shapes ~1500:1 on a production MoE, so expanding per Linear repeats
    the same answer thousands of times; keying by shape expands once per
    distinct answer instead.  Exact rather than approximate: same arguments,
    same list.  The lists are shared, not copied -- downstream only iterates
    them -- which is also what makes ``menu_cache_shapes``' retention the
    thing that bounds the work.
    """
    from .tessera_menu import expand_tessera_menu

    by_shape: dict[tuple, list] = {}
    menus: dict[str, list] = {}
    for name in targets:
        shape = tuple(weights[name].shape)
        if shape not in by_shape:
            by_shape[shape] = expand_tessera_menu(
                shape, mode=mode, tp_degree=tp_degree,
                parallel_kind=parallel_kind,
            )
        menus[name] = by_shape[shape]
    return menus


def main(argv: "Sequence[str] | None" = None) -> int:
    import torch

    from . import format_registry as fr
    from .production_weight_cache import ProductionWeightCache
    from .tessera_menu import MENU_MODES, PARALLEL_NONE, menu_mode
    from .tessera_rate_surface import leave_one_anchor_out
    from .tessera_render import (
        HessianContractError, tessera_encoder_hessian_status,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="cost payload (.pkl)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="resumable per-anchor JSON; defaults beside --out")
    ap.add_argument("--menu-mode", default=None, choices=sorted(MENU_MODES))
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-act-rows", type=int, default=512)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="price every Nth decoder layer (1 = every Linear)")
    ap.add_argument("--anchors", type=int, default=ROUND_ONE_ANCHORS)
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="hard stop on adaptive rounds (0 = governed by "
                         "--anchor-budget instead). Rounds are not the "
                         "budget: a round adds ONE anchor to each surface "
                         "that is still failing its gate, so capping rounds "
                         "caps how far the worst surface can be improved, "
                         "which is the opposite of what the adaptive loop is "
                         "for.")
    ap.add_argument("--anchor-budget", type=int, default=12,
                    help="max anchors per (fused group, family) surface. The "
                         "adaptive loop keeps splitting the worst-predicted "
                         "interval until every member's LOO clears "
                         "--loo-gate or this budget is spent, and the payload "
                         "records which of the two stopped each surface.")
    ap.add_argument("--loo-gate", type=float, default=0.25,
                    help="max |log2 error| an interpolated surface may carry")
    ap.add_argument("--tp-degree", type=int, default=1)
    ap.add_argument("--max-artifact-bpp", type=float, default=0.0,
                    help="cap the MEASURED envelope at this artifact bpp "
                         "(0 = each family's whole legal range). A budget "
                         "decision, not a menu one: rungs above the envelope "
                         "stay in the menu and are simply not priced, because "
                         "the surface refuses to extrapolate.")
    ap.add_argument("--hessian", default="require", choices=("require", "off"),
                    help="'require' (default) prices every rung with the "
                         "per-unit XtX from this run's calibration "
                         "activations, which is what the encoder's shipping "
                         "default consumes; it REFUSES if the pinned encoder "
                         "cannot take one. 'off' prices weights-only "
                         "deliberately and stamps hessian.supplied=false on "
                         "every row, so a weights-only price can never be "
                         "read as a shipping one.")
    ap.add_argument("--deadline-seconds", type=float, default=0.0,
                    help="stop starting new anchors after this much wall time")
    args = ap.parse_args(argv)

    mode = menu_mode(args.menu_mode)
    hessian_status = tessera_encoder_hessian_status()
    if args.hessian == "require" and not hessian_status["accepted"]:
        # Refuse before the model load, not after an hour of encodes.
        raise HessianContractError(
            "--hessian require: " + str(hessian_status["reason"]) + ". The "
            "encoder's shipping default consumes a per-unit Hessian (LDLQ + "
            "full-H row-scale refit), so a campaign that prices without one "
            "prices bytes that are not the bytes that ship. Re-run with "
            "--hessian off to price weights-only deliberately -- every row "
            "and the payload are stamped hessian.supplied=false -- or pin "
            "prismaquant.tessera_render.TESSERA_HESSIAN_KWARG once the "
            "H-aware encoder branch is merged."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wire_dir = cache_dir / "wire"
    wire_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        Path(args.out).with_suffix(".anchors.json")
    )

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    from .model_profiles import detect_profile

    profile = detect_profile(args.model)
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.endswith("lm_head") or "embed" in name:
            continue
        targets.append(name)
    if args.layer_stride > 1:
        import re as _re
        keep = []
        for name in targets:
            match = _re.search(r"\.layers\.(\d+)\.", name)
            if match is None or int(match.group(1)) % args.layer_stride == 0:
                keep.append(name)
        targets = keep
    print(f"[campaign] {len(targets)} target Linears, mode={mode}, "
          f"device={device}", flush=True)

    tokens, corpus_text = _calibration_tokens(
        args.model, args.nsamples, args.seqlen, args.seed)
    want_h = args.hessian == "require"
    acts, hessians, hessian_rows = _collect_activations(
        model, targets, tokens, args.max_act_rows, device,
        want_hessian=want_h)
    # For the log line and the run-level provenance only; every encode is
    # given its own Linear's count.
    hessian_token_count = (
        max(hessian_rows.values()) if hessian_rows else 0)
    hessian_token_min = (
        min(hessian_rows.values()) if hessian_rows else 0)
    print(f"[campaign] activations collected "
          f"(hessian={args.hessian}, rows/Linear "
          f"{hessian_token_min}..{hessian_token_count})",
          flush=True)

    # The draw's identity, in Tessera's own required vocabulary and built by
    # the function the production render also calls. An ActivationSource
    # refuses a provenance missing any of the three, which is what makes
    # "cost.pkl and the production cache are the same draw" a checkable claim.
    hessian_identity = th.calibration_identity(
        corpus_text, tokens,
        fit_tokens=int(hessian_token_count),
        source="wikitext-2-raw-v1/train",
        split_role="calibration",
        model=str(args.model),
        seed=int(args.seed),
        nsamples=int(args.nsamples),
        seqlen=int(args.seqlen),
        fit_tokens_min=int(hessian_token_min),
    )

    weights = {name: dict(model.named_modules())[name].weight.detach()
               for name in targets}
    del model
    torch.cuda.empty_cache()

    # ONE ActivationSource for the whole campaign, and ONE set of encoder
    # keywords per unit PER SCALE PLANE. The block-LDL is a function of the
    # unit's Hessian alone -- not of its rate -- so a twelve-anchor surface
    # would otherwise factorise the same [in, in] matrix twelve times. The
    # refit metric is a function of the Hessian AND the plane, because Tessera
    # keys the refit objective by plane (the exact quadratic on a CHANNEL row
    # scale, a diagonal power on the LUT plane's coupled blocks) and the two
    # measured answers disagree. The plane is a property of the family, not of
    # the rung, so the memo is keyed by (unit, plane) and the bound is one
    # factorisation per unit per family-plane. The source is built from the
    # same functions the production render calls (``tessera_hessian``), so the
    # campaign's price and the cache's render are one rendering of one draw
    # (principle 8).
    calibration_source = None
    if want_h:
        calibration_source = th.activation_source(hessians, hessian_identity)

    @functools.lru_cache(maxsize=None)
    def _activation_kwargs_for(name: str, scale_plane) -> dict:
        return th.encoder_kwargs(
            calibration_source, name,
            int(weights[name].shape[1]), device, scale_plane=scale_plane)

    cache = ProductionWeightCache(
        weights={}, levers={"tessera_campaign": True},
        cache_dir=str(cache_dir),
        metadata={"schema": SCHEMA, "menu_mode": mode},
    )
    menus = expand_menus_for_targets(
        weights, targets, mode=mode, tp_degree=args.tp_degree,
        parallel_kind=PARALLEL_NONE,
    )

    measured: dict[str, dict[str, list[CampaignAnchor]]] = {}
    if checkpoint.is_file():
        raw = json.loads(checkpoint.read_text())
        for row in raw.get("anchors", []):
            anchor = CampaignAnchor(**row)
            measured.setdefault(anchor.qname, {}).setdefault(
                anchor.family, []).append(anchor)
        print(f"[campaign] resumed {sum(len(v) for f in measured.values() for v in f.values())} "
              f"anchors from {checkpoint}", flush=True)

    def flush_checkpoint() -> None:
        rows = [
            vars(a) for by_f in measured.values()
            for anchors in by_f.values() for a in anchors
        ]
        tmp = checkpoint.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"schema": SCHEMA, "anchors": rows}))
        os.replace(tmp, checkpoint)

    started = time.time()
    deadline = float(args.deadline_seconds)
    stopped_early = False

    def out_of_time() -> bool:
        return deadline > 0 and (time.time() - started) > deadline

    # The measured ENVELOPE per (unit, family): the rung range anchors span.
    # Capped by --max-artifact-bpp, which is a wall-clock budget and says so:
    # the menu keeps every legal rung, and the ones above the envelope are
    # left unpriced rather than extrapolated into.
    cap = float(args.max_artifact_bpp)
    rates_by_unit: dict[str, dict[str, set[int]]] = {}
    for name in targets:
        per_family: dict[str, set[int]] = {}
        for rung in menus[name]:
            if cap > 0 and rung.bpp > cap:
                continue
            per_family.setdefault(rung.family, set()).add(rung.body_rate_q256)
        rates_by_unit[name] = per_family

    # Anchors are placed per FUSED GROUP, not per (unit, family).
    #
    # The cap is a WIRE bpp and the wire->body map is shape-dependent (the
    # CHANNEL plane amortises over rows), so solving each member's top anchor
    # independently put fused siblings on different body grids even when they
    # share ``in_features`` and therefore share the realisable set: on
    # Qwen3-0.6B ``q_proj`` topped out at R1388 where ``k/v_proj`` topped out
    # at R1372, and every bisected anchor below inherited the offset. A
    # measured-only table then has almost nothing in the intersection of a
    # group's menus -- the qkv E4M3 intersection was exactly one rung -- which
    # made the group's measured baseline weak for a reason that has nothing to
    # do with Tessera. One grid per group, from the intersection of its
    # members' realisable sets, and every member measures the same rungs.
    def _group_key(name: str) -> str:
        try:
            key = profile.fused_sibling_group(name)
        except Exception:
            key = None
        return f"g:{key}" if key else f"u:{name}"

    anchor_groups: dict[str, list[str]] = {}
    for name in targets:
        anchor_groups.setdefault(_group_key(name), []).append(name)
    for key in anchor_groups:
        anchor_groups[key].sort()

    # The rungs every member of the group can realise, per family. A family
    # missing from one member is not a group family at all: a shared grid over
    # a rung one sibling cannot build is not shared.
    group_rates: dict[str, dict[str, list[int]]] = {}
    for key, members in anchor_groups.items():
        per_family: dict[str, list[int]] = {}
        families = set.intersection(*[
            set(rates_by_unit[m].keys()) for m in members]) if members else set()
        for family in sorted(families):
            shared = set.intersection(*[
                rates_by_unit[m][family] for m in members])
            if shared:
                per_family[family] = sorted(shared)
        group_rates[key] = per_family

    print("[campaign] anchor groups: "
          + ", ".join(
              f"{key}({len(members)})" for key, members
              in sorted(anchor_groups.items())), flush=True)

    def _snap(rate: int, allowed: Sequence[int]) -> "int | None":
        """The realisable rung nearest ``rate``, or None if there is none.

        ``anchor_schedule`` and ``next_anchor_rate`` both work in rate space
        and can name a q256 no member can build; snapping keeps the group's
        grid shared, which is the whole point of placing anchors per group.
        """
        if not allowed:
            return None
        return min(allowed, key=lambda r: (abs(int(r) - int(rate)), int(r)))

    # Round 1, breadth-first over units, so a deadline yields every unit priced
    # at the same depth rather than a prefix priced deeply and a tail not at all.
    budget = int(args.anchor_budget)
    # Why the group's own worst member drives the split: the grid is shared,
    # so a rung added for one sibling is measured for all of them anyway. The
    # gate closes for a GROUP surface only when it closes for every member.
    surface_stop: dict[tuple[str, str], str] = {}
    round_index = 0
    while True:
        round_index += 1
        if int(args.max_rounds) > 0 and round_index > int(args.max_rounds):
            print(f"[campaign] --max-rounds {args.max_rounds} reached",
                  flush=True)
            break
        pending: list[tuple[str, str, int]] = []
        for key, members in sorted(anchor_groups.items()):
            for family, allowed in group_rates[key].items():
                # The grid is what EVERY member actually measured: one
                # member's failed encode must not let the group think it has
                # an anchor there.
                grid = sorted(set.intersection(*[
                    {a.body_rate_q256
                     for a in measured.get(m, {}).get(family, [])}
                    for m in members
                ])) if members else []
                if round_index == 1:
                    want = [
                        _snap(r, allowed) for r in
                        anchor_schedule(allowed[0], allowed[-1], args.anchors)
                    ]
                    for rate in sorted({r for r in want if r is not None}):
                        pending.extend(
                            (m, family, rate) for m in members
                            if rate not in sorted(
                                a.body_rate_q256 for a
                                in measured.get(m, {}).get(family, []))
                        )
                    continue
                if len(grid) >= budget:
                    surface_stop.setdefault((key, family), "anchor_budget")
                    continue
                worst = 0.0
                worst_loo: dict = {}
                ready = True
                interpolable = 0
                for m in members:
                    anchors = measured.get(m, {}).get(family, [])
                    if len(anchors) < 3:
                        ready = False
                        break
                    member_loo = _loo_for(anchors, leave_one_anchor_out)
                    if _loo_refused(member_loo):
                        # A refused surface has no leave-one-out error, and a
                        # missing error is not a zero one.  Reading it as 0.0
                        # would say "this surface interpolates perfectly" about
                        # the one surface that does not interpolate at all --
                        # and would close the gate on it.  It contributes
                        # nothing to ``worst`` instead, so the group keeps
                        # spending anchors for whichever members can still use
                        # them, and the refusal is reported as itself.
                        continue
                    interpolable += 1
                    value = float(
                        member_loo.get("max_abs_log2_error", 0.0) or 0.0)
                    if value > worst:
                        worst, worst_loo = value, member_loo
                if not ready:
                    continue
                if not interpolable:
                    surface_stop.setdefault((key, family), "non_interpolable")
                    continue
                if worst <= args.loo_gate:
                    surface_stop.setdefault((key, family), "gate_closed")
                    continue
                nxt = next_anchor_rate(grid, worst_loo)
                nxt = _snap(nxt, allowed) if nxt is not None else None
                if nxt is None or nxt in grid:
                    surface_stop.setdefault((key, family), "no_room")
                    continue
                pending.extend((m, family, nxt) for m in members)
        if not pending:
            print(f"[campaign] round {round_index}: nothing pending", flush=True)
            break
        print(f"[campaign] round {round_index}: {len(pending)} anchors",
              flush=True)
        for index, (name, family, rung) in enumerate(pending):
            if out_of_time():
                stopped_early = True
                print("[campaign] deadline reached; stopping", flush=True)
                break
            fmt = f"{family}_R{rung}"
            activations = acts.get(name)
            if activations is None:
                continue
            try:
                anchor = _measure_anchor(
                    qname=name, weight=weights[name].to(device),
                    activations=activations.to(device),
                    format_name=fmt, cache=cache, wire_dir=wire_dir,
                    activation_kwargs_for=(
                        _activation_kwargs_for if want_h else None),
                    hessian_required=want_h,
                )
            except HessianContractError:
                # Never absorbed into "one anchor failed": a contract refusal
                # is about every row this run would write, not this one.
                raise
            except Exception as exc:
                print(f"[campaign] {name} {fmt}: FAILED {type(exc).__name__}: "
                      f"{exc}", flush=True)
                continue
            measured.setdefault(name, {}).setdefault(family, []).append(anchor)
            if index % 10 == 0:
                flush_checkpoint()
                print(f"[campaign] r{round_index} {index}/{len(pending)} "
                      f"{name} {fmt} mse={anchor.dloss:.6g} "
                      f"{anchor.seconds:.1f}s", flush=True)
        flush_checkpoint()
        if stopped_early:
            break

    loo: dict[str, dict[str, dict]] = {}
    for name, by_family in measured.items():
        for family, anchors in by_family.items():
            if len(anchors) >= 3:
                loo.setdefault(name, {})[family] = _loo_for(
                    anchors, leave_one_anchor_out)
    loo_pre = loo

    provenance = {
        "provenance": {
            "menu_mode": mode,
            "tp_degree": int(args.tp_degree),
            "model": str(args.model),
            "nsamples": int(args.nsamples),
            "seqlen": int(args.seqlen),
            "max_act_rows": int(args.max_act_rows),
            "layer_stride": int(args.layer_stride),
            "anchors_round_one": int(args.anchors),
            "max_rounds": int(args.max_rounds),
            "anchor_budget": int(args.anchor_budget),
            "rounds_run": int(round_index),
            "anchor_placement": "fused_group",
            "anchor_groups": {
                key: list(members)
                for key, members in sorted(anchor_groups.items())
            },
            # Per surface: what it cost and whether its gate closed. The
            # adaptive loop's whole purpose is to spend encodes where the
            # interpolation is measurably failing, so "how many anchors and
            # how many seconds did that take, and did it work" is the readout
            # that says whether the budget was the binding constraint.
            "surfaces": {
                name: {
                    family: {
                        "anchors": len(anchors),
                        "encode_seconds": round(
                            sum(float(a.seconds) for a in anchors), 3),
                        "rungs": sorted(a.body_rate_q256 for a in anchors),
                        # ``None``, never 0.0, when there is no leave-one-out
                        # error to report: a refused surface did not fit its
                        # own anchors perfectly, it failed to become a surface,
                        # and a zero here would read as the former.
                        "loo_max_abs_log2_error": _surface_loo(
                            loo_pre.get(name, {}).get(family),
                            float(args.loo_gate))[0],
                        "gate_closed": _surface_loo(
                            loo_pre.get(name, {}).get(family),
                            float(args.loo_gate))[1],
                        "non_interpolable": _loo_refused(
                            loo_pre.get(name, {}).get(family) or {}),
                        "stopped_by": (
                            "non_interpolable"
                            if _loo_refused(
                                loo_pre.get(name, {}).get(family) or {})
                            else surface_stop.get(
                                (_group_key(name), family), "round_limit")),
                    }
                    for family, anchors in sorted(by_family.items())
                }
                for name, by_family in sorted(measured.items())
            },
            "loo_gate": float(args.loo_gate),
            "max_artifact_bpp": float(args.max_artifact_bpp),
            "stopped_early": bool(stopped_early),
            "wall_seconds": time.time() - started,
            "cache_dir": str(cache_dir),
            "wire_dir": str(wire_dir),
            "tessera_commit": os.environ.get("TESSERA_COMMIT", ""),
            "hessian": {
                "supplied": bool(want_h),
                "mode": str(args.hessian),
                "reason": str(hessian_status["reason"]),
                "consumed_by": (
                    "prismaquant.tessera_render.encode_tessera_unit"
                    + (" -> tessera.export.ActivationSource.for_unit -> "
                       "tessera.export.encode_linear_planes("
                       + ", ".join(f"{k}=" for k in hessian_status["kwargs"])
                       + ")" if want_h else "")),
                "kwargs": list(hessian_status["kwargs"]),
                "recipe": dict(hessian_status["recipe"]),
                "token_count": int(hessian_token_count),
                "token_count_min": int(hessian_token_min),
                # The identity triple Tessera requires, plus its context. The
                # legacy ``text_sha`` spelling is kept because older cost
                # tables carry it and ``assert_uniform_hessian_identity``
                # compares tables across runs.
                **dict(hessian_identity),
                "text_sha": hessian_identity["fit_ids_sha256"],
            },
        },
    }
    payload = campaign_cost_payload(
        measured, menus, loo=loo, provenance=provenance)
    payload["menu_sizes"] = {n: len(m) for n, m in menus.items()}
    payload["anchor_counts"] = {
        n: {f: len(a) for f, a in by_f.items()} for n, by_f in measured.items()
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(payload, handle)
    total = sum(len(rows) for rows in payload["costs"].values())
    print(f"[campaign] wrote {args.out}: {len(payload['costs'])} units, "
          f"{total} priced rungs, {len(payload['formats'])} distinct formats",
          flush=True)
    return 0


def _loo_refused(member_loo) -> bool:
    """Did the surface refuse to exist at all?

    :func:`_loo_for` reports a refusal as an ``error`` key.  Everywhere else
    reads ``max_abs_log2_error`` with a default, so the distinction between
    "fits perfectly" and "is not a surface" has to be asked for explicitly.
    """

    return bool(member_loo) and "error" in member_loo


def _surface_loo(member_loo, gate: float):
    """``(loo_error, gate_closed)`` for one surface, refusal-aware."""

    if not member_loo or _loo_refused(member_loo):
        return None, False
    value = float(member_loo.get("max_abs_log2_error", 0.0) or 0.0)
    return value, bool(value <= gate)


def _loo_for(anchors, leave_one_anchor_out) -> dict:
    from .tessera_rate_surface import TesseraRateSurface

    ordered = sorted(anchors, key=lambda a: a.body_rate_q256)
    try:
        surface = TesseraRateSurface(
            unit_name=ordered[0].qname,
            family=ordered[0].family,
            layout="tight",
            currency=CURRENCY,
            anchor_q256=tuple(a.body_rate_q256 for a in ordered),
            anchor_dloss=tuple(a.dloss for a in ordered),
            anchor_stderr=tuple(a.dloss_stderr for a in ordered),
        )
    except Exception as exc:
        return {"error": str(exc)}
    return dict(leave_one_anchor_out(surface))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
